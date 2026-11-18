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
  паре → «не удалось собрать микро-батч»).

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


if __name__ == "__main__":
    unittest.main()
