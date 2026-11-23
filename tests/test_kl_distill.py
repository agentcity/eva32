"""KL-дистилляция: формат логитов учителя, эталонный KL и путь обучения.

Задача 0107. Тесты намеренно не требуют torch: разбор формата и опорный
расчёт живут в ``qat/kl.py`` на чистом Python, поэтому в обычном
``python3 -m unittest`` проверяются и они. Torch-путь (``qat.train._kl_sum``,
``PairDataset._teacher_logits``) проверяется отдельными тестами, которые
пропускаются, если torch нет.

Что здесь закрепляется:

* KL совпадает с ручным расчётом (замкнутая формула для двух классов),
  включая множитель ``T²`` и температуру;
* ``KL(p‖p) = 0`` — на этом держится сверка выравнивания позиций учителя;
* ``parse_teacher_logits`` принимает ровно те строки, что и обучение
  (``A - 1 … A + 1``), и падает на чужом словаре;
* парсер файла логитов строг: битый JSON, отсутствующее поле и «рваные»
  строки — ошибка с номером строки, а не молчаливый пропуск;
* ``tiny_qat.teacher_logits_rows`` берёт позиции ``P-1 … P+A-2`` — сдвиг на
  один токен даёт другой KL, и это ловится;
* регрессия: пара длиннее ``max_seq_len`` (обрезка) с логитами учителя не
  выбрасывается. До 0107 длина ответа считалась до обрезки, и на 15,6%
  датасета KL-путь оставался без данных (``dropped_bad_logits`` на каждой
  паре → «не удалось собрать микро-батч»);
* top-k (0111): формат ``{"topk": {k, indices, logits}}``, KL по top-k учителя
  с полным словарём ученика совпадает с ручной формулой и с torch-путём, KL по
  объединению двух top-k равен 0 на совпадающих носителях и ``inf`` при
  ``-inf``-хвосте; ``PairStore`` читает файл потоково (память не растёт с
  числом пар), ``PairDataset`` превращает top-k в индексы/логиты тензорами.

Запуск::

    python3 -m unittest tests.test_kl_distill -v
    PYTHONPATH=.pylibs python3.13 -m unittest tests.test_kl_distill -v
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qat import kl  # noqa: E402 - путь настраивается выше

SCRIPT = ROOT / "scripts" / "tiny_qat.py"


def _load_tiny():
    """Загрузить scripts/tiny_qat.py как модуль (scripts/ — не пакет)."""
    spec = importlib.util.spec_from_file_location("tiny_qat_kl", SCRIPT)
    if spec is None or spec.loader is None:  # pragma: no cover - файл на месте
        raise RuntimeError(f"не загружается {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


class KlReferenceTest(unittest.TestCase):
    """Опорный расчёт: замкнутые формулы, а не второй такой же код."""

    def test_two_class_manual(self) -> None:
        """Учитель равномерный, ученик ``[ln 3, 0]``: KL = ln 2 - ½ ln 3.

        q = softmax([ln 3, 0]) = [3/4, 1/4]; p = [1/2, 1/2];
        KL = ½ ln(½ / ¾) + ½ ln(½ / ¼) = ½ ln(⅔) + ½ ln 2 = ln 2 − ½ ln 3.
        """
        teacher = [0.0, 0.0]
        student = [math.log(3.0), 0.0]
        expected = math.log(2.0) - 0.5 * math.log(3.0)
        self.assertAlmostEqual(kl.kl_divergence(teacher, student), expected, places=12)
        self.assertAlmostEqual(expected, 0.143_841_035_8, places=9)

    def test_identical_is_zero(self) -> None:
        logits = [1.5, -2.0, 0.25, 7.0]
        self.assertAlmostEqual(kl.kl_divergence(logits, logits), 0.0, places=12)

    def test_temperature_scaling(self) -> None:
        """KL_T = T²·KL(softmax(t/T)‖softmax(s/T)) — проверяем по формуле.

        Для равномерного учителя и ученика ``[ln 3, 0]``:
        ``KL_T = -T²/2 · ln(4r / (1+r)²)``, ``r = 3**(1/T)``.
        """
        teacher = [0.0, 0.0]
        student = [math.log(3.0), 0.0]
        for temperature in (0.5, 1.0, 2.0, 4.0):
            ratio = 3.0 ** (1.0 / temperature)
            expected = -0.5 * math.log(4.0 * ratio / (1.0 + ratio) ** 2) * temperature ** 2
            got = kl.kl_divergence(teacher, student, temperature)
            self.assertAlmostEqual(got, expected, places=12, msg=f"T={temperature}")

    def test_temperature_softens(self) -> None:
        """С ростом T распределение ближе к равномерному: энтропия растёт.

        (Сам ``T²·KL`` при этом немонотонен — это множитель градиента, а не
        «штраф за остроту»; поэтому проверяется именно смягчение.)
        """
        values = [3.0, 0.0, -1.0]
        entropies = []
        for temperature in (0.5, 1.0, 2.0, 8.0):
            probs = [math.exp(v) for v in kl.log_softmax(values, temperature)]
            entropies.append(-sum(p * math.log(p) for p in probs))
        self.assertEqual(entropies, sorted(entropies))
        self.assertGreater(entropies[-1], entropies[0])

    def test_temperature_definition(self) -> None:
        """KL_T = T²·Σ p·ln(p/q), p и q — явные softmax(logits/T) из теста."""
        teacher = [3.0, 0.0, -1.0]
        student = [-1.0, 2.0, 0.5]

        def softmax(values: list[float], temperature: float) -> list[float]:
            top = max(value / temperature for value in values)
            exp = [math.exp(value / temperature - top) for value in values]
            return [value / sum(exp) for value in exp]

        for temperature in (0.5, 1.0, 3.0):
            p = softmax(teacher, temperature)
            q = softmax(student, temperature)
            expected = sum(pi * math.log(pi / qi)
                           for pi, qi in zip(p, q)) * temperature ** 2
            self.assertAlmostEqual(
                kl.kl_divergence(teacher, student, temperature), expected, places=10,
                msg=f"T={temperature}",
            )

    def test_t_squared_is_gradient_neutral(self) -> None:
        """T² — не «масштаб красивее»: при T→1 результат равен обычному KL."""
        teacher = [0.3, -0.2, 1.1]
        student = [0.0, 0.5, -0.7]
        self.assertAlmostEqual(
            kl.kl_divergence(teacher, student, 1.0),
            kl.kl_divergence(teacher, student),
            places=15,
        )

    def test_asymmetric(self) -> None:
        """KL не симметрична — значит, порядок аргументов не перепутан."""
        teacher = [2.0, 0.0, -2.0]
        student = [0.0, 0.0, 0.0]
        forward = kl.kl_divergence(teacher, student)
        backward = kl.kl_divergence(student, teacher)
        self.assertGreater(forward, 0.0)
        self.assertNotAlmostEqual(forward, backward, places=6)

    def test_bad_arguments(self) -> None:
        with self.assertRaises(ValueError):
            kl.kl_divergence([0.0, 1.0], [0.0, 1.0, 2.0])
        with self.assertRaises(ValueError):
            kl.kl_divergence([0.0], [0.0], temperature=0.0)

    def test_log_softmax_is_stable(self) -> None:
        values = kl.log_softmax([1e4, 0.0, -1e4])
        self.assertAlmostEqual(math.exp(values[0]), 1.0, places=12)
        self.assertAlmostEqual(sum(math.exp(v) for v in values), 1.0, places=12)
        self.assertTrue(all(math.isfinite(v) for v in values))

    def test_sum_over_rows(self) -> None:
        teacher = [[0.0, 0.0], [1.0, -1.0]]
        student = [[math.log(3.0), 0.0], [0.0, 1.0]]
        expected = (math.log(2.0) - 0.5 * math.log(3.0)) + kl.kl_divergence(
            teacher[1], student[1]
        )
        self.assertAlmostEqual(kl.kl_div_sum(teacher, student), expected, places=12)
        with self.assertRaises(ValueError):
            kl.kl_div_sum(teacher, student[:1])


class ParseRowTest(unittest.TestCase):
    """Допуски формата: те же, что у ``PairDataset._teacher_logits``."""

    def _rows(self, count: int, width: int = 4) -> list[list[float]]:
        return [[float(i + j) for j in range(width)] for i in range(count)]

    def test_accepted_lengths(self) -> None:
        answer_len = 5  # PairDataset считает длину до добавления eos
        for count in (answer_len - 1, answer_len, answer_len + 1):
            raw = self._rows(count)
            self.assertIs(kl.parse_teacher_logits(raw, answer_len, 4), raw,
                          msg=f"строк {count}: должны приниматься без копии")

    def test_rejected_lengths(self) -> None:
        answer_len = 5
        for count in (0, 1, answer_len - 2, answer_len + 2, 50):
            raw = self._rows(count) if count else []
            self.assertIsNone(kl.parse_teacher_logits(raw, answer_len, 4),
                              msg=f"строк {count}: должны отбрасываться")

    def test_rejected_shapes(self) -> None:
        self.assertIsNone(kl.parse_teacher_logits(None, 5))
        self.assertIsNone(kl.parse_teacher_logits("нет", 5))
        self.assertIsNone(kl.parse_teacher_logits([], 5))
        self.assertIsNone(kl.parse_teacher_logits([[], [], [], []], 5))
        self.assertIsNone(kl.parse_teacher_logits([[1.0, 2.0]] * 3 + [[1.0]], 5))
        self.assertIsNone(kl.parse_teacher_logits([[1.0, 2.0], "x"], 5))

    def test_vocab_mismatch_is_error(self) -> None:
        """Словарь учителя ≠ ученика — это не «пропустить пример»."""
        with self.assertRaises(kl.LogitsFormatError) as ctx:
            kl.parse_teacher_logits(self._rows(5, width=6), 5, vocab_size=4)
        self.assertIn("словарь ученика", str(ctx.exception))
        # без vocab_size ширину не с чем сверять — принимаем
        self.assertIsNotNone(kl.parse_teacher_logits(self._rows(5, width=6), 5))

    def test_value_checks(self) -> None:
        good = self._rows(3, width=2)
        self.assertIs(kl.parse_teacher_logits(good, 3, 2, check_values=True), good)
        for bad_row in ([float("nan"), 0.0], [float("inf"), 0.0], ["x", 0.0], [None, 0.0],
                        [True, 0.0]):
            raw = [list(bad_row), [0.0, 1.0], [1.0, 0.0]]
            self.assertIsNone(kl.parse_teacher_logits(raw, 3, 2, check_values=True),
                              msg=f"строка {bad_row} должна отбрасываться")


class LogitsFileTest(unittest.TestCase):
    """Парсер файла логитов: строгий, с номерами строк."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "pairs_logits.jsonl"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _row(self, rows: int = 3, width: int = 2) -> dict:
        return {
            "question": "вопрос",
            "answer": "ответ",
            "logits": [[float(i + j) / 10.0 for j in range(width)] for i in range(rows)],
        }

    def test_round_trip(self) -> None:
        rows = [self._row(3), self._row(4)]
        written = kl.write_logits_file(self.path, rows)
        self.assertEqual(written, 2)
        back = kl.read_logits_file(self.path, vocab_size=2)
        self.assertEqual(len(back), 2)
        self.assertEqual(back[0]["question"], "вопрос")
        self.assertEqual(back[0]["answer"], "ответ")
        self.assertEqual(len(back[0]["logits"]), 3)
        self.assertEqual(len(back[0]["logits"][0]), 2)
        # округление записи — 5 знаков: (2 + 1) / 10
        self.assertAlmostEqual(back[0]["logits"][2][1], 0.3, places=6)
        self.assertEqual(kl.read_logits_file(self.path, vocab_size=2, limit=1)[0]
                         ["logits"][0][0], back[0]["logits"][0][0])

    def test_teacher_logits_alias(self) -> None:
        row = self._row()
        row["teacher_logits"] = row.pop("logits")
        self.path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        back = kl.read_logits_file(self.path, vocab_size=2)
        self.assertEqual(len(back), 1)
        self.assertEqual(len(back[0]["logits"]), 3)

    def test_broken_json_reports_line(self) -> None:
        self.path.write_text(
            json.dumps(self._row(), ensure_ascii=False) + "\n{битый\n", encoding="utf-8"
        )
        with self.assertRaises(kl.LogitsFormatError) as ctx:
            kl.read_logits_file(self.path, vocab_size=2)
        self.assertIn(":2:", str(ctx.exception))
        self.assertIn("битый JSON", str(ctx.exception))

    def test_missing_fields(self) -> None:
        for broken in (
            {"answer": "a", "logits": [[0.0, 1.0]]},
            {"question": "q", "logits": [[0.0, 1.0]]},
            {"question": "q", "answer": "a"},
            {"question": "q", "answer": "a", "logits": []},
            {"question": "q", "answer": "a", "logits": {"0": 1}},
        ):
            self.path.write_text(json.dumps(broken) + "\n", encoding="utf-8")
            with self.assertRaises(kl.LogitsFormatError):
                kl.read_logits_file(self.path, vocab_size=2)

    def test_ragged_rows(self) -> None:
        row = self._row()
        row["logits"][1] = [0.0]
        self.path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        with self.assertRaises(kl.LogitsFormatError):
            kl.read_logits_file(self.path, vocab_size=2)

    def test_vocab_mismatch_reports_line(self) -> None:
        row = self._row(width=3)
        self.path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        with self.assertRaises(kl.LogitsFormatError) as ctx:
            kl.read_logits_file(self.path, vocab_size=2)
        self.assertIn(":1:", str(ctx.exception))
        self.assertIn("словарь ученика", str(ctx.exception))

    def test_missing_and_empty_file(self) -> None:
        with self.assertRaises(kl.LogitsFormatError):
            kl.read_logits_file(self.path)
        self.path.write_text("\n\n", encoding="utf-8")
        with self.assertRaises(kl.LogitsFormatError):
            kl.read_logits_file(self.path)


class TeacherRowsTest(unittest.TestCase):
    """Позиции логитов учителя: ``P-1 … P+A-2``."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tiny = _load_tiny()

    def test_positions(self) -> None:
        sequence = list(range(10))  # логит на позиции i предсказывает метку i+1
        prompt_len, answer_len = 4, 3
        rows = self.tiny.teacher_logits_rows(sequence, prompt_len, answer_len)
        self.assertEqual(rows, [3, 4, 5])  # P-1=3, P+A-2=5
        # ученик берёт тот же срез: logits[P-1 : P-1+A]
        self.assertEqual(rows, sequence[prompt_len - 1:prompt_len - 1 + answer_len])

    def test_length_and_errors(self) -> None:
        sequence = list(range(10))
        self.assertEqual(len(self.tiny.teacher_logits_rows(sequence, 1, 9)), 9)
        with self.assertRaises(self.tiny.TinyError):
            self.tiny.teacher_logits_rows(sequence, 0, 3)
        with self.assertRaises(self.tiny.TinyError):
            self.tiny.teacher_logits_rows(sequence, 8, 5)
        with self.assertRaises(self.tiny.TinyError):
            self.tiny.teacher_logits_rows(sequence, 2, 0)


class FakeTokenizer:
    """Один символ → один id; eos фиксированный. Для PairDataset без torch-модели."""

    def __init__(self, eos_id: int = 1) -> None:
        self.eos_token_id = eos_id

    def __call__(self, text, add_special_tokens=False):  # noqa: ARG002 - API HF
        return SimpleNamespace(input_ids=[2 + (ord(char) % 5) for char in text])


@unittest.skipUnless(torch_available(), "torch не установлен — путь обучения проверяется на аренде")
class TorchPathTest(unittest.TestCase):
    """Torch-путь: ``_kl_sum``, ``PairDataset._teacher_logits``, обрезка пар.

    Экземпляр крошечной модели не нужен: ``_kl_sum`` принимает готовые срезы
    логитов, поэтому проверяются выравнивание, температура и словарь.
    """

    @classmethod
    def setUpClass(cls) -> None:
        import torch  # noqa: PLC0415

        from qat import train  # noqa: PLC0415

        cls.torch = torch
        cls.train = train

    def _teacher(self, rows: int, vocab: int = 5) -> "object":
        torch = self.torch
        generator = torch.Generator().manual_seed(7)
        return torch.randn(rows, vocab, generator=generator)

    def test_torch_matches_reference(self) -> None:
        teacher = self._teacher(4)
        student = self._teacher(4)
        expected = kl.kl_div_sum(teacher.tolist(), student.tolist(), temperature=1.5)
        got = float(kl.kl_div_sum_torch(self.torch, student, teacher, 1.5).item())
        self.assertAlmostEqual(got, expected, places=4)

    def test_vocab_mismatch_in_torch(self) -> None:
        with self.assertRaises(ValueError):
            kl.kl_div_sum_torch(self.torch, self._teacher(2, 5), self._teacher(2, 6))

    def _batch(self, *, shift: int):
        """Батч из одного примера: промпт 3 токена, ответ 3 токена.

        Логиты ученика покрывают позиции ``logits_start = P-1 = 2``:
        ``student[0]`` предсказывает метку с индексом 3.
        """
        torch = self.torch
        seq_len, vocab = 6, 5
        labels = torch.tensor([[-100, -100, -100, 2, 3, 4]], dtype=torch.long)
        base = self._teacher(seq_len, vocab)  # логиты для позиций 0..seq_len-1
        logits_start = 2
        logits = base[logits_start:].unsqueeze(0).clone()  # позиции 2..5
        teacher_rows = base[2:5].clone()  # предсказывают метки 3,4,5
        if shift:
            teacher_rows = base[3:6].clone()
        return logits, labels, [teacher_rows]

    def test_kl_sum_aligned(self) -> None:
        logits, labels, teacher = self._batch(shift=0)
        got = float(self.train._kl_sum(
            self.torch, logits, labels, teacher, logits_start=2,
            temperature=1.0, chunk_size=2,
        ).item())
        self.assertAlmostEqual(got, 0.0, places=6)

    def test_kl_sum_detects_off_by_one(self) -> None:
        """Сдвинутые на токен логиты учителя дают заметную KL — их видно."""
        logits, labels, teacher = self._batch(shift=1)
        got = float(self.train._kl_sum(
            self.torch, logits, labels, teacher, logits_start=2,
            temperature=1.0, chunk_size=2,
        ).item())
        self.assertGreater(got, 1e-3)

    def test_kl_sum_matches_reference_chunks(self) -> None:
        """Кусочный подсчёт (chunk_size=2) совпадает с опорным по всем строкам."""
        logits, labels, teacher = self._batch(shift=1)
        got = float(self.train._kl_sum(
            self.torch, logits, labels, teacher, logits_start=2,
            temperature=2.0, chunk_size=2,
        ).item())
        expected = kl.kl_div_sum(teacher[0].tolist(), logits[0, 0:3].tolist(), 2.0)
        self.assertAlmostEqual(got, expected, places=4)

    def test_fp16_teacher_tensor(self) -> None:
        """Логиты из ``PairDataset`` приходят в float16 — путь обязан работать."""
        logits, labels, teacher = self._batch(shift=0)
        teacher = [row.to(self.torch.float16) for row in teacher]
        got = float(self.train._kl_sum(
            self.torch, logits, labels, teacher, logits_start=2,
            temperature=1.0, chunk_size=2,
        ).item())
        self.assertGreaterEqual(got, 0.0)
        self.assertLess(got, 1e-2)  # только округление float16
        self.assertTrue(math.isfinite(got))

    def test_kl_sum_no_supervised(self) -> None:
        torch = self.torch
        labels = torch.full((1, 6), -100, dtype=torch.long)
        logits = self._teacher(5).unsqueeze(0)
        got = float(self.train._kl_sum(
            self.torch, logits, labels, [self._teacher(3)], logits_start=0,
            temperature=1.0, chunk_size=4,
        ).item())
        self.assertEqual(got, 0.0)

    def test_kl_sum_vocab_mismatch(self) -> None:
        logits, labels, _ = self._batch(shift=0)
        with self.assertRaises(RuntimeError):
            self.train._kl_sum(
                self.torch, logits, labels, [self._teacher(3, vocab=6)],
                logits_start=2, temperature=1.0, chunk_size=4,
            )


@unittest.skipUnless(torch_available(), "torch не установлен — путь обучения проверяется на аренде")
class DatasetLogitsTest(unittest.TestCase):
    """``PairDataset``: отбор строк логитов и обрезка длинных пар."""

    @classmethod
    def setUpClass(cls) -> None:
        import torch  # noqa: PLC0415

        from qat import data  # noqa: PLC0415

        cls.torch = torch
        cls.data = data

    def _dataset(self, **kwargs) -> "object":
        options = dict(max_seq_len=40, long_pairs="truncate", min_answer_tokens=1,
                       use_teacher_logits=True, vocab_size=4)
        options.update(kwargs)
        return self.data.PairDataset(
            [{"question": "q", "answer": "x" * 100}], FakeTokenizer(), **options
        )

    def _raw(self, rows: int, width: int = 4) -> list[list[float]]:
        return [[float((i + j) % 7) for j in range(width)] for i in range(rows)]

    def test_teacher_logits_tensor(self) -> None:
        dataset = self._dataset()
        tensor = dataset._teacher_logits(self._raw(5), answer_len=5)
        self.assertEqual(tuple(tensor.shape), (5, 4))
        self.assertEqual(tensor.dtype, self.torch.float16)

    def test_teacher_logits_drops(self) -> None:
        dataset = self._dataset()
        self.assertIsNone(dataset._teacher_logits(None, answer_len=5))
        self.assertIsNone(dataset._teacher_logits(self._raw(3), answer_len=5))
        ragged = self._raw(5)
        ragged[3] = ragged[3][:2]  # рваная строка: чинится только формой
        self.assertIsNone(dataset._teacher_logits(ragged, answer_len=5))
        bad = self._raw(5)
        bad[2][1] = float("nan")
        self.assertIsNone(dataset._teacher_logits(bad, answer_len=5))
        self.assertEqual(dataset.stats_snapshot().get("dropped_bad_logits"), 3)
        self.assertEqual(dataset.stats_snapshot().get("dropped_no_logits"), 1)

    def test_vocab_mismatch_raises_data_error(self) -> None:
        dataset = self._dataset()
        with self.assertRaises(self.data.DataError):
            dataset._teacher_logits(self._raw(5, width=6), answer_len=5)

    def test_truncated_pair_keeps_logits(self) -> None:
        """Регрессия 0107: длина ответа — после обрезки, а не до неё."""
        dataset = self._dataset()
        answer_ids = len(FakeTokenizer()("x" * 100).input_ids) + 1  # + eos
        prompt_ids = len(dataset._prompt_ids("q"))
        used = dataset.max_seq_len - prompt_ids
        self.assertLess(used, answer_ids, "пара должна реально обрезаться")
        dataset.rows[0]["logits"] = self._raw(used)  # логиты сняты с обрезки
        sample = dataset.encode(0)
        self.assertIsNotNone(sample, "обрезанная пара с логитами не должна теряться")
        self.assertEqual(len(sample["input_ids"]), dataset.max_seq_len)
        self.assertEqual(tuple(sample["teacher_logits"].shape), (used, 4))
        self.assertEqual(sample["labels"][prompt_ids], sample["input_ids"][prompt_ids])
        self.assertEqual(sample["labels"][-1], 1)  # eos на месте
        self.assertEqual(dataset.stats_snapshot().get("truncated_long"), 1)
        self.assertEqual(dataset.stats_snapshot().get("dropped_bad_logits", 0), 0)

    def test_missing_logits_drops_example(self) -> None:
        dataset = self._dataset()
        dataset.rows[0]["logits"] = None
        self.assertIsNone(dataset.encode(0))
        self.assertEqual(dataset.stats_snapshot().get("dropped_no_logits"), 1)

    def test_collate_mixes_rejected(self) -> None:
        dataset = self._dataset()
        prompt_ids = len(dataset._prompt_ids("q"))
        dataset.rows[0]["logits"] = self._raw(dataset.max_seq_len - prompt_ids)
        good = dataset.encode(0)
        self.assertIsNotNone(good)
        batch = self.data.collate_samples([good], pad_token_id=0)
        self.assertEqual(len(batch["teacher_logits"]), 1)
        with self.assertRaises(self.data.DataError):
            self.data.collate_samples(
                [good, {"input_ids": [2, 3], "labels": [-100, 3], "teacher_logits": None}],
                pad_token_id=0,
            )


class TopKParseTest(unittest.TestCase):
    """Формат top-k: k = ширина, индексы в словаре и без дублей, логиты конечны."""

    def _raw(self, rows: int = 2, k: int = 3, vocab: int = 10) -> dict:
        indices = [list(range(row * k, row * k + k)) for row in range(rows)]
        logits = [[float(k - j) for j in range(k)] for _ in range(rows)]
        return {"k": k, "indices": indices, "logits": logits}

    def test_accepts_valid(self) -> None:
        parsed = kl.parse_topk_logits(self._raw(), 2, 10)
        self.assertIsInstance(parsed, kl.TopKLogits)
        self.assertEqual(parsed.k, 3)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed.indices[1], (3, 4, 5))

    def test_k_mismatch_with_width(self) -> None:
        raw = self._raw()
        raw["k"] = 2
        self.assertIsNone(kl.parse_topk_logits(raw, 2, 10))
        self.assertIsNone(kl.parse_topk_logits({"indices": [[0, 1]], "logits": [[1.0]]}, 1, 10))

    def test_ragged_and_empty(self) -> None:
        raw = self._raw()
        raw["indices"][1] = [0]
        self.assertIsNone(kl.parse_topk_logits(raw, 2, 10))
        self.assertIsNone(kl.parse_topk_logits(None, 2, 10))
        self.assertIsNone(kl.parse_topk_logits({"k": 1, "indices": [], "logits": []}, 2, 10))
        self.assertIsNone(kl.parse_topk_logits("нет", 2, 10))

    def test_bad_indices(self) -> None:
        for indices in (
            [[0, 0, 1]],        # дубль
            [[0, 1, 99]],       # вне словаря
            [[0, 1, -1]],       # отрицательный
            [[0, 1, 2.5]],      # не целое
            [[0, 1, True]],     # bool — не индекс
        ):
            raw = {"k": 3, "indices": indices, "logits": [[3.0, 2.0, 1.0]]}
            self.assertIsNone(kl.parse_topk_logits(raw, 1, 10), msg=str(indices))

    def test_bad_values(self) -> None:
        for values in (
            [[3.0, float("nan"), 1.0]],
            [[3.0, float("inf"), 1.0]],
            [[1.0, 2.0, 3.0]],   # не по убыванию
            [[3.0, "x", 1.0]],
        ):
            raw = {"k": 3, "indices": [[0, 1, 2]], "logits": values}
            self.assertIsNone(kl.parse_topk_logits(raw, 1, 10), msg=str(values))

    def test_rows_tolerance(self) -> None:
        answer_len = 4
        for rows in (answer_len - 1, answer_len, answer_len + 1):
            raw = self._raw(rows)
            self.assertIsNotNone(kl.parse_topk_logits(raw, answer_len, 100),
                                 msg=f"строк {rows}")
        for rows in (0, 1, answer_len + 2):
            raw = self._raw(rows) if rows else {"k": 3, "indices": [], "logits": []}
            self.assertIsNone(kl.parse_topk_logits(raw, answer_len, 100),
                              msg=f"строк {rows}")

    def test_k_wider_than_vocab_is_error(self) -> None:
        with self.assertRaises(kl.LogitsFormatError) as ctx:
            kl.parse_topk_logits(self._raw(k=4), 1, 3)
        self.assertIn("шире словаря", str(ctx.exception))

    def test_max_k_is_error(self) -> None:
        """Файл не той ширины, что ждёт прогон, — ошибка, а не «пропустим пару»."""
        with self.assertRaises(kl.LogitsFormatError) as ctx:
            kl.parse_topk_logits(self._raw(k=4), 1, 10, max_k=2)
        self.assertIn("max_top_k", str(ctx.exception))

    def test_topk_from_logits(self) -> None:
        values = [0.5, 3.0, -1.0, 2.0, 7.0]
        indices, logits = kl.topk_from_logits(values, 3)
        self.assertEqual(indices, [4, 1, 3])
        self.assertEqual(logits, [values[i] for i in indices])
        with self.assertRaises(ValueError):
            kl.topk_from_logits(values, 0)
        with self.assertRaises(ValueError):
            kl.topk_from_logits(values, 6)
        with self.assertRaises(ValueError):
            kl.topk_from_logits([1.0, float("nan")], 2)


class TopKFileTest(unittest.TestCase):
    """Запись и чтение top-k файла: строго, с номерами строк."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "pairs_topk.jsonl"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _topk(self, rows: int = 2, k: int = 3):
        return kl.TopKLogits(
            [list(range(row * k, row * k + k)) for row in range(rows)],
            [[float(k - j) for j in range(k)] for _ in range(rows)],
        )

    def _row(self, rows: int = 2, k: int = 3) -> dict:
        return {"question": "вопрос", "answer": "ответ", "topk": self._topk(rows, k)}

    def test_round_trip(self) -> None:
        written = kl.write_topk_file(self.path, [self._row(), self._row(3)])
        self.assertEqual(written, 2)
        back = kl.read_topk_file(self.path, vocab_size=16, max_k=3)
        self.assertEqual(len(back), 2)
        self.assertEqual(back[0]["topk"].k, 3)
        self.assertEqual(back[0]["topk"].indices[0], (0, 1, 2))
        self.assertEqual(len(back[1]["topk"]), 3)
        self.assertIsNone(back[0]["logits"])
        self.assertEqual(back[0]["k"], 3)

    def test_write_from_full_logits_truncates(self) -> None:
        row = {"question": "q", "answer": "a",
               "logits": [[0.0, 5.0, 1.0, 3.0], [1.0, 0.0, 2.0, 4.0]]}
        kl.write_topk_file(self.path, [row], k=2)
        back = kl.read_topk_file(self.path, vocab_size=4, max_k=2)
        self.assertEqual(back[0]["topk"].indices[0], (1, 3))
        self.assertEqual(back[0]["topk"].logits[0], (5.0, 3.0))

    def test_write_requires_k_for_full_logits(self) -> None:
        with self.assertRaises(kl.LogitsFormatError):
            kl.write_topk_file(self.path, [{"question": "q", "answer": "a",
                                            "logits": [[1.0, 0.0]]}])

    def test_write_rejects_rerounding_topk(self) -> None:
        with self.assertRaises(kl.LogitsFormatError) as ctx:
            kl.write_topk_file(self.path, [self._row(k=3)], k=2)
        self.assertIn("хвост потерян", str(ctx.exception))

    def test_read_topk_rejects_full_vocab_row(self) -> None:
        kl.write_logits_file(self.path, [{"question": "q", "answer": "a",
                                          "logits": [[1.0, 0.0]]}])
        with self.assertRaises(kl.LogitsFormatError) as ctx:
            kl.read_topk_file(self.path, vocab_size=2)
        self.assertIn("top-k", str(ctx.exception))

    def test_read_topk_reports_line_and_reason(self) -> None:
        bad = {"question": "q", "answer": "a",
               "topk": {"k": 2, "indices": [[0, 0]], "logits": [[1.0, 0.0]]}}
        self.path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
        with self.assertRaises(kl.LogitsFormatError) as ctx:
            kl.read_topk_file(self.path, vocab_size=2)
        self.assertIn(":1:", str(ctx.exception))
        self.assertIn("повторяется", str(ctx.exception))

    def test_flat_aliases(self) -> None:
        """topk_indices/topk_logits — та же запись, «плоскими» полями."""
        row = {"question": "q", "answer": "a",
               "topk_indices": [[2, 0]], "topk_logits": [[1.0, -1.0]]}
        self.path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        back = kl.read_topk_file(self.path, vocab_size=3)
        self.assertEqual(back[0]["topk"].indices, ((2, 0),))

    def test_read_all_streams_and_validates(self) -> None:
        kl.write_topk_file(self.path, [self._row()])
        streamed = list(kl.iter_logits_file(self.path, vocab_size=16))
        self.assertEqual(len(streamed), 1)
        self.assertIsInstance(streamed[0]["topk"], kl.TopKLogits)


class TopKLTest(unittest.TestCase):
    """KL по top-k: ручная формула, совпадение с полным словарём, объединение."""

    def test_full_coverage_equals_exact_kl(self) -> None:
        teacher = [1.5, -2.0, 0.25, 7.0]
        student = [0.3, 0.1, -1.0, 4.0]
        indices, values = kl.topk_from_logits(teacher, len(teacher))
        self.assertAlmostEqual(
            kl.kl_divergence_topk_full(indices, values, student),
            kl.kl_divergence(teacher, student), places=12,
        )

    def test_truncated_matches_manual_formula(self) -> None:
        """Ручная сумма по k индексам: softmax по k у учителя, по V у ученика."""
        teacher = [2.0, 1.0, 0.0, -1.0]
        student = [0.5, -0.5, 1.0, 0.0]
        k = 2
        indices, values = kl.topk_from_logits(teacher, k)
        log_t = kl.log_softmax(values)
        log_s = kl.log_softmax(student)
        expected = sum(math.exp(lt) * (lt - log_s[index])
                       for index, lt in zip(indices, log_t))
        self.assertAlmostEqual(
            kl.kl_divergence_topk_full(indices, values, student), expected, places=12,
        )
        # усечение не равно полной KL, но того же порядка на гладких логитах
        self.assertGreater(kl.kl_divergence_topk_full(indices, values, student), 0.0)

    def test_union_identical_support_is_zero(self) -> None:
        values = [3.0, 1.0, -2.0]
        indices = [7, 2, 5]
        self.assertEqual(kl.kl_divergence_topk(indices, values, indices, values), 0.0)

    def test_union_renormalizes_over_union(self) -> None:
        """Носители разные: KL считается по объединению, а не по одному из них."""
        teacher_idx, teacher_vals = [0, 1], [3.0, 2.0]
        student_idx, student_vals = [1, 2], [1.0, 0.0]
        got = kl.kl_divergence_topk(teacher_idx, teacher_vals, student_idx, student_vals,
                                    missing="min")
        teacher_vector = [3.0, 2.0, min(teacher_vals)]  # индекс 2 — floor учителя
        student_vector = [min(student_vals), 1.0, 0.0]   # индекс 0 — floor ученика
        expected = kl.kl_divergence(teacher_vector, student_vector)
        self.assertAlmostEqual(got, expected, places=12)

    def test_union_missing_neg_inf_is_infinite(self) -> None:
        """У ученика нет индекса, где у учителя масса: это honest inf, не сбой."""
        self.assertEqual(
            kl.kl_divergence_topk([0, 1], [2.0, 1.0], [2, 3], [2.0, 1.0]),
            math.inf,
        )
        finite = kl.kl_divergence_topk([0, 1], [2.0, 1.0], [2, 3], [2.0, 1.0],
                                       missing="min")
        self.assertTrue(math.isfinite(finite))
        self.assertGreater(finite, 0.0)
        with self.assertRaises(ValueError):
            kl.kl_divergence_topk([0], [1.0], [0], [1.0], missing="что-то")

    def test_temperature(self) -> None:
        """T² масштабирует результат: ручная формула при той же T."""
        teacher, student = [2.0, 0.0, -1.0, 1.0], [0.0, 0.5, 0.2, -0.3]
        indices, values = kl.topk_from_logits(teacher, 3)
        for temperature in (0.5, 1.0, 2.5):
            log_t = kl.log_softmax(values, temperature)
            log_s = kl.log_softmax(student, temperature)
            manual = sum(math.exp(lt) * (lt - log_s[i])
                         for i, lt in zip(indices, log_t)) * temperature ** 2
            self.assertAlmostEqual(
                kl.kl_divergence_topk_full(indices, values, student, temperature),
                manual, places=12, msg=f"T={temperature}",
            )

    def test_sum_topk_both_shapes(self) -> None:
        teacher = kl.TopKLogits([[3, 1], [0, 2]], [[2.0, 1.0], [1.5, 0.5]])
        student_full = [[0.0, 1.0, 2.0, -1.0], [1.0, 0.0, -1.0, 2.0]]
        student_topk = kl.TopKLogits([[1, 2], [0, 2]], [[1.0, 2.0], [1.0, -1.0]])
        full_sum = kl.kl_div_sum_topk(teacher, student_full)
        self.assertGreater(full_sum, 0.0)
        union_sum = kl.kl_div_sum_topk(teacher, student_topk, missing="min")
        self.assertTrue(math.isfinite(union_sum))
        with self.assertRaises(ValueError):
            kl.kl_div_sum_topk(teacher, student_full[:1])
        with self.assertRaises(ValueError):
            # индекс учителя 9 вне словаря ученика (4)
            kl.kl_div_sum_topk(kl.TopKLogits([[9]], [[1.0]]), [[0.0, 1.0, 2.0, 3.0]])


@unittest.skipUnless(torch_available(), "torch не установлен — путь обучения проверяется на аренде")
class TopKTorchTest(unittest.TestCase):
    """torch-путь top-k: gather по индексам учителя, сверка с эталоном."""

    @classmethod
    def setUpClass(cls) -> None:
        import torch  # noqa: PLC0415

        from qat import train  # noqa: PLC0415

        cls.torch = torch
        cls.train = train

    def test_matches_reference(self) -> None:
        torch = self.torch
        generator = torch.Generator().manual_seed(11)
        teacher = torch.randn(4, 5, generator=generator)
        student = torch.randn(4, 7, generator=generator)  # словарь ученика шире
        indices = torch.topk(teacher, 3, dim=-1).indices
        values = teacher.gather(1, indices)
        got = float(kl.kl_div_sum_topk_torch(torch, student, indices, values).item())
        expected = kl.kl_div_sum_topk(
            kl.TopKLogits(indices.tolist(), values.tolist()), student.tolist(),
        )
        self.assertAlmostEqual(got, expected, places=4)

    def test_fp16_teacher_logits(self) -> None:
        torch = self.torch
        generator = torch.Generator().manual_seed(3)
        teacher = torch.randn(2, 9, generator=generator)
        student = teacher.clone()
        indices = torch.topk(teacher, 4, dim=-1).indices
        values = teacher.gather(1, indices)
        expected = float(kl.kl_div_sum_topk_torch(torch, student, indices, values).item())
        got = float(kl.kl_div_sum_topk_torch(
            torch, student, indices, values.to(torch.float16)).item())
        # fp16-логиты: отличие только в округлении (шум ~1e-3), не в формуле.
        self.assertGreater(got, 0.0)
        self.assertAlmostEqual(got, expected, places=2)
        self.assertTrue(math.isfinite(got))

    def test_index_out_of_vocab_raises(self) -> None:
        torch = self.torch
        student = torch.zeros(1, 4)
        with self.assertRaises(ValueError):
            kl.kl_div_sum_topk_torch(torch, student,
                                     torch.tensor([[9]]), torch.tensor([[1.0]]))
        with self.assertRaises(ValueError):
            kl.kl_div_sum_topk_torch(torch, student,
                                     torch.tensor([[0, 1]]), torch.tensor([[1.0]]))
        with self.assertRaises(ValueError):
            kl.kl_div_sum_topk_torch(torch, student,
                                     torch.tensor([[]], dtype=torch.long),
                                     torch.tensor([[]], dtype=torch.float))

    def _batch(self, *, shift: int = 0):
        """Батч из одного примера: промпт 3 токена, ответ 3 токена (как в 0107)."""
        torch = self.torch
        seq_len, vocab = 6, 5
        labels = torch.tensor([[-100, -100, -100, 2, 3, 4]], dtype=torch.long)
        base = torch.randn(seq_len, vocab, generator=torch.Generator().manual_seed(7))
        logits = base[2:].unsqueeze(0).clone()
        teacher_rows = base[2:5].clone()  # предсказывают метки 3,4,5
        if shift:
            teacher_rows = base[3:6].clone()
        indices = torch.topk(teacher_rows, 2, dim=-1).indices
        values = teacher_rows.gather(1, indices)
        return logits, labels, [{"indices": indices, "logits": values}]

    def test_kl_sum_dispatch_topk_aligned(self) -> None:
        """Совпадающие веса: torch-путь совпадает с эталоном по top-k.

        KL здесь не 0: учитель перенормирован на top-k, поэтому остаётся
        ``-log(p_масса_topk)`` — это цена усечения, а не сдвиг позиций.
        """
        logits, labels, teacher = self._batch()
        got = float(self.train._kl_sum(
            self.torch, logits, labels, teacher, logits_start=2,
            temperature=1.0, chunk_size=2,
        ).item())
        expected = kl.kl_div_sum_topk(
            kl.TopKLogits(teacher[0]["indices"].tolist(),
                          teacher[0]["logits"].tolist()),
            logits[0, 0:3].tolist(),
        )
        self.assertAlmostEqual(got, expected, places=4)
        self.assertGreater(got, 0.0)

    def test_kl_sum_dispatch_topk_detects_shift(self) -> None:
        logits, labels, teacher = self._batch(shift=1)
        got = float(self.train._kl_sum(
            self.torch, logits, labels, teacher, logits_start=2,
            temperature=1.0, chunk_size=2,
        ).item())
        self.assertGreater(got, 1e-3)

    def test_kl_sum_topk_index_beyond_vocab(self) -> None:
        torch = self.torch
        labels = torch.tensor([[-100, -100, -100, 2, 3, 4]], dtype=torch.long)
        logits = torch.randn(1, 4, 5)
        teacher = [{"indices": torch.tensor([[0, 99]]),
                    "logits": torch.tensor([[1.0, 0.0]])}]
        with self.assertRaises(RuntimeError):
            self.train._kl_sum(self.torch, logits, labels, teacher, logits_start=2,
                               temperature=1.0, chunk_size=2)


@unittest.skipUnless(torch_available(), "torch не установлен — PairStore/PairDataset проверяются с torch")
class StreamingDataTest(unittest.TestCase):
    """Потоковое чтение: PairStore не держит логиты, PairDataset отдаёт top-k."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        from qat import data  # noqa: PLC0415

        self.data = data

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write(self, name: str, rows: list[dict]) -> Path:
        path = self.dir / name
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path

    def test_store_reads_rows_by_offset(self) -> None:
        path = self._write("pairs.jsonl", [
            {"question": "q1", "answer": "a1", "logits": [[1.0, 0.0], [0.0, 1.0]]},
            {"question": "q2", "answer": "a2", "logits": [[0.5, 0.5], [0.1, 0.9]]},
        ])
        store = self.data.PairStore(path)
        self.assertEqual(len(store), 2)
        self.assertEqual(store[1]["question"], "q2")
        self.assertEqual(len(store[0]["logits"]), 2)
        self.assertEqual([row["question"] for row in store], ["q1", "q2"])
        with self.assertRaises(IndexError):
            store[5]

    def test_store_returns_topk_raw(self) -> None:
        path = self._write("topk.jsonl", [
            {"question": "q", "answer": "a",
             "topk": {"k": 2, "indices": [[0, 1]], "logits": [[1.0, 0.0]]}},
        ])
        store = self.data.PairStore(path)
        row = store[0]
        self.assertIsNone(row["logits"])
        self.assertEqual(row["topk"]["k"], 2)

    def test_store_skips_rows_without_question(self) -> None:
        path = self._write("mixed.jsonl", [
            {"answer": "a"},
            {"question": "q", "answer": "a"},
        ])
        store = self.data.PairStore(path)
        self.assertEqual(len(store), 1)
        self.assertEqual(store.skipped, 1)

    def test_store_broken_json_reports_line(self) -> None:
        path = self.dir / "broken.jsonl"
        path.write_text('{"question": "q", "answer": "a"}\n{битый\n', encoding="utf-8")
        with self.assertRaises(self.data.DataError) as ctx:
            self.data.PairStore(path)
        self.assertIn(":2:", str(ctx.exception))

    def test_load_pairs_without_logits(self) -> None:
        path = self._write("pairs.jsonl", [
            {"question": "q", "answer": "a", "logits": [[1.0, 0.0]]},
        ])
        rows = self.data.load_pairs(path, with_logits=False)
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]["logits"])
        rows_with = self.data.load_pairs(path)
        self.assertEqual(rows_with[0]["logits"], [[1.0, 0.0]])

    def test_dataset_turns_topk_into_tensors(self) -> None:
        raw = {"k": 2, "indices": [[0, 1], [1, 0]], "logits": [[2.0, 1.0], [1.5, 0.5]]}
        dataset = self.data.PairDataset(
            [{"question": "q", "answer": "x"}], FakeTokenizer(), max_seq_len=40,
            use_teacher_logits=True, vocab_size=4, expected_top_k=2,
        )
        dataset.rows[0]["topk"] = raw
        sample = dataset.encode(0)
        self.assertIsNotNone(sample)
        topk = sample["teacher_topk"]
        self.assertEqual(tuple(topk["indices"].shape), (2, 2))
        self.assertEqual(topk["indices"].dtype, self.data.torch.long)
        self.assertEqual(topk["logits"].dtype, self.data.torch.float32)
        self.assertIsNone(sample["teacher_logits"])
        batch = self.data.collate_samples([sample], pad_token_id=0)
        self.assertEqual(batch["teacher_logits"][0]["indices"].tolist(), [[0, 1], [1, 0]])

    def test_dataset_drops_bad_topk(self) -> None:
        dataset = self.data.PairDataset(
            [{"question": "q", "answer": "x"}], FakeTokenizer(), max_seq_len=40,
            use_teacher_logits=True, vocab_size=4,
        )
        for raw in (
            {"k": 2, "indices": [[0, 0]], "logits": [[1.0, 0.0]]},   # дубль
            {"k": 2, "indices": [[0, 1]], "logits": [[1.0, 9.0]]},   # не по убыванию
            {"k": 2, "indices": [[0, 1]], "logits": [[1.0]]},        # рваная
        ):
            dataset.rows[0]["topk"] = raw
            self.assertIsNone(dataset.encode(0), msg=str(raw))
        self.assertEqual(dataset.stats_snapshot().get("dropped_bad_topk"), 3)

    def test_dataset_expected_top_k_mismatch_raises(self) -> None:
        dataset = self.data.PairDataset(
            [{"question": "q", "answer": "x"}], FakeTokenizer(), max_seq_len=40,
            use_teacher_logits=True, vocab_size=4, expected_top_k=64,
        )
        dataset.rows[0]["topk"] = {"k": 2, "indices": [[0, 1]], "logits": [[1.0, 0.0]]}
        with self.assertRaises(self.data.DataError) as ctx:
            dataset.encode(0)
        self.assertIn("ожидает k=64", str(ctx.exception))

    def test_memory_does_not_grow_with_pairs(self) -> None:
        """Потоковое чтение: пик памяти PairStore не растёт с числом пар.

        Тот же формат, тот же размер пары: 2 пары против 16. ``PairStore``
        разбирает одну строку за раз (пик ~ одна пара), ``read_logits_file``
        держит все — его пик обязан вырасти.
        """
        import tracemalloc  # noqa: PLC0415

        from qat import kl as kl_mod  # noqa: PLC0415

        def make(count: int) -> Path:
            rows = [{"question": f"q{index}", "answer": f"a{index}",
                     "logits": [[float((row + col) % 7) for col in range(200)]
                                for row in range(60)]}
                    for index in range(count)]
            return self._write(f"wide{count}.jsonl", rows)

        def peak(call: Any) -> int:
            tracemalloc.start()
            try:
                call()
                _current, top = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            return top

        small, large = make(2), make(16)

        def stream(path: Path) -> None:
            store = self.data.PairStore(path)
            for index in range(len(store)):
                store[index]

        small_store = peak(lambda: stream(small))
        large_store = peak(lambda: stream(large))
        small_all = peak(lambda: kl_mod.read_logits_file(small, vocab_size=200))
        large_all = peak(lambda: kl_mod.read_logits_file(large, vocab_size=200))
        self.assertLess(large_store, small_store * 2.0 + 2 ** 16,
                        "потоковое чтение не должно расти с числом пар")
        self.assertGreater(large_all, small_all * 4.0,
                           "чтение всего файла обязано расти с числом пар")



if __name__ == "__main__":
    unittest.main()
