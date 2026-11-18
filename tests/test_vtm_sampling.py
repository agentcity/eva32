"""Паритет SAMP: C-интерпретатор и Python-эталон (задача 0051).

Стратегии SAMP (0 argmax, 1 temperature, 2 top-k, 3 top-p) и параметры из
меты реализованы дважды — в `src/vtm/interp.c` и `tools/vtm_ref.py`. Главная
гарантия проекта — паритет C↔Python, поэтому здесь один и тот же контейнер
исполняют оба интерпретатора и сравнивают последовательность токенов
посимвольно: одинаковый `sample_seed` и одинаковые логиты обязаны дать
одинаковый вывод.

C-рантайм вызывается как `src/vtm/vtm run ... --tokens 1 --max-new N`.
Если он не собран (`cd src/vtm && make`) или собран под другую версию
контейнера, тест пропускается, как и в tests/test_vtm_full.py.

Запуск:
    python3 -m unittest tests.test_vtm_sampling -v
    python3 -m unittest discover -s tests
"""

import math
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools import vtm_ref  # noqa: E402

VTM_BIN = os.path.join(ROOT, "src", "vtm", "vtm")

#: Логиты без явного лидера: любая стратегия «ходит» по нескольким токенам,
#: поэтому расхождение ГПСЧ или порядка сортировки сразу видно.
LOGITS = [0.4, -0.2, 1.1, 0.0, 0.7]
#: Те же логиты для штрафа за повторы: 1.1 — «зацикленный» фаворит,
#: после штрафа 2.0 его перебивает 0.4.
LOOP_LOGITS = [0.4, 0.0, 1.1, 0.0]


def _samp_program(logits, strategy, meta):
    """Контейнер «logits -> SAMP(strategy) -> HALT» с dst-тензором token."""
    tensors = [
        vtm_ref.Tensor(
            name="logits",
            kind=vtm_ref.KIND_F32,
            dims=(len(logits),),
            data=[float(value) for value in logits],
        ),
        vtm_ref.Tensor(
            name="token",
            kind=vtm_ref.KIND_F32,
            dims=(1,),
            data=[-1.0],
        ),
    ]
    program = [
        vtm_ref.Instr(op=vtm_ref.Op.SAMP, dst=1, a=0, imm=strategy),
        vtm_ref.Instr(op=vtm_ref.Op.HALT),
    ]
    return vtm_ref.VTM(tensors=tensors, program=program, meta=dict(meta))


class _SamplingParityCase(unittest.TestCase):
    max_new = 16

    @classmethod
    def setUpClass(cls):
        if not os.access(VTM_BIN, os.X_OK):
            raise unittest.SkipTest(
                "C-рантайм не собран: src/vtm/vtm отсутствует "
                "(cd src/vtm && make)"
            )

    def run_case(self, logits, strategy, meta, max_new=None):
        """Собрать контейнер и сравнить последовательности C и Python."""
        max_new = self.max_new if max_new is None else max_new
        model = _samp_program(logits, strategy, meta)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "samp.vtm")
            vtm_ref.write_vtm(path, model.tensors, model.program, model.meta)

            expected = vtm_ref.execute(path, [1], max_new).generated
            proc = subprocess.run(
                [VTM_BIN, "run", path, "--tokens", "1",
                 "--max-new", str(max_new)],
                cwd=ROOT, capture_output=True, text=True, timeout=180,
            )
            if proc.returncode != 0 and "версия контейнера" in proc.stderr:
                self.skipTest(
                    "C-рантайм устарел относительно tools/vtm_ref.py: "
                    + proc.stderr.strip()
                )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            line = next(
                (line for line in proc.stdout.splitlines()
                 if line.startswith("вывод:")),
                None,
            )
            self.assertIsNotNone(line, proc.stdout)
            got = [int(token) for token in line.split(":", 1)[1].split()]
        self.assertEqual(len(got), max_new)
        self.assertEqual(got, expected)
        return got

    def test_argmax(self):
        self.run_case(LOGITS, vtm_ref.SAMP_ARGMAX, {}, max_new=4)

    def test_temperature_zero_is_argmax(self):
        got = self.run_case(LOGITS, vtm_ref.SAMP_TEMPERATURE,
                            {"temperature": 0.0}, max_new=4)
        argmax = self.run_case(LOGITS, vtm_ref.SAMP_ARGMAX, {},
                               max_new=4)
        self.assertEqual(got, argmax)

    def test_temperature(self):
        self.run_case(LOGITS, vtm_ref.SAMP_TEMPERATURE,
                      {"temperature": 0.8, "sample_seed": 12345})

    def test_top_k(self):
        self.run_case(LOGITS, vtm_ref.SAMP_TOPK,
                      {"top_k": 3, "temperature": 1.0,
                       "sample_seed": 7})

    def test_top_k_one_is_argmax(self):
        got = self.run_case(LOGITS, vtm_ref.SAMP_TOPK,
                            {"top_k": 1, "temperature": 0.9,
                             "sample_seed": 7}, max_new=4)
        self.assertEqual(len(set(got)), 1)
        self.assertEqual(got[0], 2)  # индекс максимума в LOGITS

    def test_top_p(self):
        self.run_case(LOGITS, vtm_ref.SAMP_TOPP,
                      {"top_p": 0.7, "temperature": 1.0,
                       "sample_seed": 99})

    def test_top_p_one_does_not_cut(self):
        # top_p = 1.0 не отсекает ничего: вывод совпадает с temperature.
        plain = self.run_case(LOGITS, vtm_ref.SAMP_TEMPERATURE,
                              {"temperature": 1.0, "sample_seed": 3})
        full = self.run_case(LOGITS, vtm_ref.SAMP_TOPP,
                             {"top_p": 1.0, "temperature": 1.0,
                              "sample_seed": 3})
        self.assertEqual(plain, full)

    def test_repetition_penalty_breaks_loop(self):
        # Без штрафа argmax зациклен на токене 2; штраф 4.0 снимает его:
        # 1.1 / 4.0 = 0.275 < 0.4 — фаворит перестаёт побеждать.
        plain = self.run_case(LOOP_LOGITS, vtm_ref.SAMP_ARGMAX, {},
                              max_new=6)
        self.assertEqual(plain, [2] * 6)
        got = self.run_case(
            LOOP_LOGITS, vtm_ref.SAMP_ARGMAX,
            {"repetition_penalty": 4.0, "repetition_window": 2},
            max_new=6,
        )
        self.assertNotEqual(got, plain)
        # 1.1 / 4.0 = 0.275 < 0.4 — фаворит перестаёт побеждать.
        self.assertEqual(got[1], 0)

    def test_repetition_penalty_with_sampling(self):
        # Окно меняет распределение, поэтому паритет проверяет и историю.
        self.run_case(LOOP_LOGITS, vtm_ref.SAMP_TEMPERATURE,
                      {"temperature": 1.0, "repetition_penalty": 2.0,
                       "repetition_window": 3, "sample_seed": 5})

    def test_wide_vocab_parity(self):
        # 64 логита и 120 сэмплов: если C и Python разойдутся в порядке
        # суммирования или в ГПСЧ, это видно почти сразу. Сравнение точное:
        # при одном seed обе реализации детерминированы.
        logits = [
            ((index * 2654435761) % 1000) / 1000.0 - 0.5
            for index in range(64)
        ]
        self.run_case(
            logits, vtm_ref.SAMP_TOPP,
            {"temperature": 0.7, "top_p": 0.95, "sample_seed": 123456},
            max_new=120,
        )

    def test_python_is_deterministic(self):
        model = _samp_program(LOGITS, vtm_ref.SAMP_TEMPERATURE,
                              {"temperature": 1.0, "sample_seed": 2024})
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "samp.vtm")
            vtm_ref.write_vtm(path, model.tensors, model.program, model.meta)
            first = vtm_ref.execute(path, [1], 12).generated
            again = vtm_ref.execute(path, [1], 12).generated
        self.assertEqual(first, again)
        self.assertGreater(len(set(first)), 1)


class TestSamplingParams(unittest.TestCase):
    """Разбор параметров SAMP: дефолты, клампы, вложенные ключи."""

    def test_defaults(self):
        params = vtm_ref._samp_params({})
        self.assertEqual(params["temperature"], 1.0)
        self.assertEqual(params["top_k"], 0)
        self.assertEqual(params["top_p"], 1.0)
        self.assertEqual(params["repetition_penalty"], 1.0)
        self.assertEqual(params["repetition_window"], 64)
        self.assertEqual(params["sample_seed"], 0)
        self.assertFalse(params["has_seed"])

    def test_explicit(self):
        params = vtm_ref._samp_params({
            "temperature": 0.25,
            "top_k": 7,
            "top_p": 0.9,
            "repetition_penalty": 1.3,
            "repetition_window": 12,
            "sample_seed": 42,
        })
        self.assertEqual(params["temperature"], 0.25)
        self.assertEqual(params["top_k"], 7)
        self.assertEqual(params["top_p"], 0.9)
        self.assertEqual(params["repetition_penalty"], 1.3)
        self.assertEqual(params["repetition_window"], 12)
        self.assertEqual(params["sample_seed"], 42)
        self.assertTrue(params["has_seed"])

    def test_nested_key_and_clamps(self):
        params = vtm_ref._samp_params({
            "model": {"temperature": 0.5, "top_p": 2.0, "top_k": -3,
                      "repetition_penalty": 0.0,
                      "repetition_window": 10_000},
            "temperature": 0.75,
        })
        self.assertEqual(params["temperature"], 0.75)   # верхний уровень
        self.assertEqual(params["top_p"], 1.0)          # >= 1 не отсекает
        self.assertEqual(params["top_k"], 0)            # отрицательный -> 0
        self.assertEqual(params["repetition_penalty"], 1.0)
        self.assertEqual(params["repetition_window"], 256)

    def test_negative_temperature_is_greedy(self):
        params = vtm_ref._samp_params({"temperature": -1.0})
        self.assertEqual(params["temperature"], 0.0)

    def test_seed_must_be_finite_and_non_negative(self):
        self.assertFalse(vtm_ref._samp_params({"sample_seed": -1})["has_seed"])
        self.assertFalse(
            vtm_ref._samp_params({"sample_seed": math.inf})["has_seed"]
        )
        self.assertTrue(vtm_ref._samp_params({"sample_seed": 0})["has_seed"])


class TestSamplerEdgeCases(unittest.TestCase):
    """Края без C: NaN, один токен, top-p = 0, imm вне 0..3."""

    def test_single_token(self):
        sampler = vtm_ref._Sampler({})
        self.assertEqual(sampler.choose([5.0], vtm_ref.SAMP_TOPP), 0)

    def test_all_nan_is_token_zero(self):
        sampler = vtm_ref._Sampler({"temperature": 1.0})
        self.assertEqual(
            sampler.choose([math.nan, math.nan], vtm_ref.SAMP_TEMPERATURE),
            0,
        )

    def test_top_p_zero_is_argmax(self):
        sampler = vtm_ref._Sampler({"top_p": 0.0})
        for _ in range(5):
            self.assertEqual(
                sampler.choose([0.1, 0.9, 0.3], vtm_ref.SAMP_TOPP), 1
            )

    def test_unknown_strategy(self):
        model = _samp_program(LOGITS, 9, {})
        with self.assertRaises(ValueError):
            vtm_ref.execute(model, [1], 1)


if __name__ == "__main__":
    unittest.main()
