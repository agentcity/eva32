"""Паритет SAMP: C-интерпретатор и Python-эталон (задача 0051).

Стратегии SAMP (0 argmax, 1 temperature, 2 top-k, 3 top-p) и параметры из
меты реализованы дважды — в `src/vtm/interp.c` и `tools/vtm_ref.py`. Главная
гарантия проекта — паритет C↔Python, поэтому здесь один и тот же контейнер
исполняют оба интерпретатора и сравнивают последовательность токенов
посимвольно: одинаковый `sample_seed` и одинаковые логиты обязаны дать
одинаковый вывод.

Задача 0080 добавила сюда паритет по кольцевому KV-окну: контейнер
прогоняется на 2×ctx токенах (позиция уходит за длину кэша), C и Python
сравниваются по самим логитам, а не только по выбранному токену.

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
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools import parity_real, vtm_compile, vtm_ref  # noqa: E402

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


# ---------------------------------------------------------------------------
# Кольцевое KV-окно (задача 0076) в Python-эталоне (задача 0080)
# ---------------------------------------------------------------------------


class TestContextWindowParsing(unittest.TestCase):
    """`meta["context_window"]`: те же клампы, что `kv_window()` в C."""

    def test_missing_and_broken_values_mean_no_window(self):
        for meta in (
            {},
            {"context_window": 0},
            {"context_window": -4},
            {"context_window": 0.5},
            {"context_window": 2_147_483_648},
            {"context_window": "8"},
            {"context_window": True},
            {"context_window": math.nan},
            {"context_window": math.inf},
        ):
            self.assertEqual(vtm_ref._context_window(meta), 0, meta)

    def test_valid_values_and_nesting(self):
        self.assertEqual(vtm_ref._context_window({"context_window": 4}), 4)
        self.assertEqual(vtm_ref._context_window({"context_window": 4.9}), 4)
        self.assertEqual(vtm_ref._context_window({"context_window": 1}), 1)
        # Как C-сканер: ключ на любом уровне, внешний приоритетнее.
        self.assertEqual(
            vtm_ref._context_window(
                {"model": {"context_window": 3}, "context_window": 7}
            ),
            7,
        )


class TestKvRingWindow(unittest.TestCase):
    """Кольцо без C: 5 проходов при ctx=2 и окне 1.

    E[0]=[1,0], E[1]=[0,1], E[2]=[1,1]; K=V=эмбеддинг токена прохода.
    `context_window=1` оставляет ATTN только текущий ключ, поэтому на
    позиции 4 (token 1) out обязан быть ровно [0, 1]; с окном длины кэша
    (2) в out подмешивается токен позиции 3 — это и различает окна.
    """

    TOKENS = [0, 1, 2, 0, 1]

    @classmethod
    def _model(cls, context_window=None):
        e = vtm_ref.Tensor.f32("e", (3, 2), [1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        x = vtm_ref.Tensor.f32("x", (2,), [0.0, 0.0])
        k_cache = vtm_ref.Tensor.f32("kc", (2, 2), [0.0] * 4)
        v_cache = vtm_ref.Tensor.f32("vc", (2, 2), [0.0] * 4)
        q = vtm_ref.Tensor.f32("q", (1, 1, 2), [0.0, 0.0])
        out = vtm_ref.Tensor.f32("out", (1, 1, 2), [0.0, 0.0])
        program = [
            vtm_ref.Instr(vtm_ref.Op.EMB, dst=1, a=0,
                          flags=vtm_ref.IFLAG_EMB_INPUT),
            vtm_ref.Instr(vtm_ref.Op.KVW, dst=2, a=1, b=3, c=1),
            vtm_ref.Instr(vtm_ref.Op.CPY, dst=4, a=1),
            vtm_ref.Instr(vtm_ref.Op.ATTN, dst=5, a=4, b=2, c=3,
                          flags=vtm_ref.IFLAG_CAUSAL),
            vtm_ref.Instr(vtm_ref.Op.HALT),
        ]
        meta = {} if context_window is None else {"context_window": context_window}
        return vtm_ref.VTM(
            tensors=[e, x, k_cache, v_cache, q, out],
            program=program,
            meta=meta,
        )

    def test_position_past_cache_is_not_an_error(self):
        # 5 проходов при ctx=2: раньше KVW падал на позиции 2.
        result = vtm_ref.execute(self._model(context_window=1), self.TOKENS, 0)
        self.assertEqual(result.steps, len(self.TOKENS))
        self.assertIsNotNone(result.get("out"))

    def test_window_one_sees_only_current_key(self):
        result = vtm_ref.execute(self._model(context_window=1), self.TOKENS, 0)
        self.assertEqual(result.get("out"), [0.0, 1.0])
        # win=1: пишется только слот 0, слот 1 так и остался нулевым.
        self.assertEqual(result.get("kc"), [0.0, 1.0, 0.0, 0.0])

    def test_full_cache_window_mixes_previous_token(self):
        result = vtm_ref.execute(self._model(), self.TOKENS, 0)
        out = result.get("out")
        self.assertIsNotNone(out)
        score = 1.0 / math.sqrt(2.0)
        expected = math.exp(score) / (math.exp(score) + 1.0)
        self.assertAlmostEqual(out[0], 1.0 - expected, places=6)
        self.assertAlmostEqual(out[1], expected, places=6)
        # Оба слота кольца (ctx=2) использованы.
        self.assertEqual(result.get("kc"), [0.0, 1.0, 1.0, 0.0])

    def test_negative_position_is_still_an_error(self):
        model = self._model(context_window=1)
        # imm=-1 на позиции 0: отрицательный слот кольца запрещён, как в C.
        model.program[1] = vtm_ref.Instr(
            vtm_ref.Op.KVW, dst=2, a=1, b=3, c=1, imm=-1
        )
        with self.assertRaises(ValueError):
            vtm_ref.execute(model, [0], 0)


#: Проба для паритета по логитам: `vtm run` логиты не печатает, поэтому
#: тест собирает крошечный C-бинарник (во временном каталоге, не в репозитории)
#: и линкует его с `src/vtm/libvtm.a`. `%.17g` — чтобы max|Δ| не упирался в
#: точность печати.
_KV_WINDOW_PROBE = r'''/* Снять logits последнего прохода .vtm. */
#include "vtm.h"

#include <locale.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    vtm_t *machine;
    const vtm_tensor *logits;
    int position = 0;
    int arg;
    uint32_t i;

    if (argc < 3) {
        fprintf(stderr, "нужны model.vtm и токены\n");
        return 2;
    }
    setlocale(LC_NUMERIC, "C");  /* десятичная точка, а не запятая */
    machine = vtm_load(argv[1]);
    if (machine == NULL) {
        fprintf(stderr, "vtm_load: %s\n", vtm_last_error());
        return 1;
    }
    for (arg = 2; arg < argc; arg++) {
        vtm_set_position(machine, position);
        vtm_set_token(machine, atoi(argv[arg]));
        if (vtm_run(machine, NULL, NULL) != VTM_OK) {
            fprintf(stderr, "прогон %d: %s\n", position,
                    vtm_error_message(machine));
            vtm_free(machine);
            return 1;
        }
        position++;
    }
    logits = vtm_tensor_by_name(machine, "logits");
    if (logits == NULL || logits->kind != VTM_TENSOR_F32
        || logits->f32 == NULL) {
        fprintf(stderr, "в контейнере нет fp32-тензора logits\n");
        vtm_free(machine);
        return 1;
    }
    for (i = 0; i < logits->n_elements; i++)
        printf("%.17g\n", (double)logits->f32[i]);
    vtm_free(machine);
    return 0;
}
'''


class TestKvWindowParity(unittest.TestCase):
    """C ↔ Python на 2×ctx токенах: кольцо и `context_window`.

    Модель — синтетический слой `vtm_compile.make_synthetic(mode="attn")`:
    EMB → RMSN → Q/K/V → ROPE → KVW → ATTN → … → logits. Длина KV-кэша 8,
    прогонов 16 = 2·ctx, поэтому позиции 8..15 затирают старейшие слоты.

    Порог 1e-6 на скалярном бэкенде; фактически числа совпадают побитово
    (max|Δ| = 0.0, проба печатает `%.17g`). На `auto`/SIMD к этому
    добавляется ~1e-6…1e-5 из-за другого порядка суммирования в NEON/AVX2 —
    это проверяет отдельный тест с порогом 1e-4.
    """

    ctx = 8
    vocab = 32

    @classmethod
    def setUpClass(cls):
        if not os.access(VTM_BIN, os.X_OK):
            raise unittest.SkipTest(
                "C-рантайм не собран: src/vtm/vtm отсутствует "
                "(cd src/vtm && make)"
            )
        if not parity_real.VTM_LIB.is_file():
            raise unittest.SkipTest(
                "C-библиотека не собрана: src/vtm/libvtm.a отсутствует "
                "(cd src/vtm && make)"
            )
        cls.cc = parity_real.find_c_compiler()
        if cls.cc is None:
            raise unittest.SkipTest("нет C-компилятора (cc/clang/gcc)")
        cls.workdir = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.workdir.cleanup)
        workdir = Path(cls.workdir.name)
        source = workdir / "kv_window_probe.c"
        source.write_text(_KV_WINDOW_PROBE, encoding="utf-8")
        cls.probe = workdir / "kv_window_probe"
        proc = subprocess.run(
            [
                cls.cc, "-std=c99", "-O2", "-Wall", "-Wextra",
                "-I", str(parity_real.VTM_INCLUDE),
                "-o", str(cls.probe), str(source),
                str(parity_real.VTM_LIB), "-lm",
            ],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise unittest.SkipTest(
                "C-проба не собралась: " + proc.stderr.strip()[-500:]
            )
        cls.model = vtm_compile.make_synthetic(seed=5, size=32, mode="attn")

    def _parity(self, context_window, backend):
        """Прогнать 2·ctx токенов в C и Python; вернуть (max|Δ|, logits…)."""
        meta = dict(self.model.meta)
        if context_window is None:
            meta.pop("context_window", None)
        else:
            meta["context_window"] = context_window
        # Один и тот же тензорный граф, отличается только мета.
        model = vtm_ref.VTM(
            tensors=self.model.tensors,
            program=self.model.program,
            meta=meta,
        )
        tokens = [
            (index * 7 + 1) % self.vocab for index in range(2 * self.ctx)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "kv_window.vtm")
            vtm_ref.write_vtm(path, model.tensors, model.program, meta)
            expected = vtm_ref.execute(path, tokens, 0).logits
            env = dict(os.environ)
            env["VTM_BACKEND"] = backend
            proc = subprocess.run(
                [str(self.probe), path, *[str(token) for token in tokens]],
                capture_output=True, text=True, env=env, timeout=180,
            )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIsNotNone(expected)
        got = [float(value) for value in proc.stdout.split()]
        self.assertEqual(len(got), len(expected), proc.stdout)
        deltas = [abs(a - b) for a, b in zip(expected, got)]
        self.assertEqual(
            vtm_ref._argmax(expected), vtm_ref._argmax(got),
            f"argmax разошёлся при context_window={context_window}",
        )
        return max(deltas), expected, got

    def test_ring_at_two_contexts(self):
        # context_window нет: win = ctx = 8, позиции 8..15 работают по кольцу.
        delta, _, _ = self._parity(None, "scalar")
        self.assertLessEqual(delta, 1e-6, f"max|Δ|={delta:.3e}")

    def test_context_window_shorter_than_cache(self):
        # win = 4 < ctx: в кэше живут только последние 4 ключа.
        delta, _, _ = self._parity(self.ctx // 2, "scalar")
        self.assertLessEqual(delta, 1e-6, f"max|Δ|={delta:.3e}")

    def test_context_window_longer_than_cache_is_clamped(self):
        # context_window > ctx: окно упирается в длину кэша (как kv_window).
        delta, clamped, _ = self._parity(4 * self.ctx, "scalar")
        self.assertLessEqual(delta, 1e-6, f"max|Δ|={delta:.3e}")
        # Тот же контейнер без ключа обязан дать те же логиты.
        _, plain, _ = self._parity(None, "scalar")
        self.assertEqual(clamped, plain)

    def test_default_backend_ring(self):
        # auto = SIMD, если CPU умеет: другой порядок суммирования даёт
        # ~1e-6…1e-5, но кольцо обязано остаться тем же (порог 1e-4).
        delta, _, _ = self._parity(self.ctx // 2, "auto")
        self.assertLessEqual(delta, 1e-4, f"max|Δ|={delta:.3e}")


if __name__ == "__main__":
    unittest.main()
