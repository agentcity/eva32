"""Паритет CONV4/GDELTA: C-интерпретатор и Python-эталон (задача 0176).

Эталон истины — `tools/vtm_ref.py` (op 0x12/0x13). Здесь один и тот же
контейнер исполняют оба интерпретатора:

* `vtm_ref` строит `.vtm` из случайных тензоров;
* крошечная C-проба (во временном каталоге, не в репозитории) линкуется с
  `src/vtm/libvtm.a`, гоняет токены по одному (`vtm_run`) и пачкой
  (`vtm_run_batch`) и печатает `y` и состояние после каждого шага;
* сравниваются `y` и состояние с порогом **1e-6**. Состояние в C — f32 и
  округляется на каждом шаге, как в эталоне (`_f32` в `vtm_ref`).

Ручные числа из `tests/test_gated_delta.py` проверяет C-тест
`tests/vtm/test_vtm.c` (`test_conv4`/`test_gdelta_manual`), здесь — случайные
формы, несколько value-голов на одну key-голову, eps и сброс состояния.

Запуск (после `cd src/vtm && make`):

    python3 -m unittest discover -s tests/vtm -p 'test_gated_delta_parity.py' -v

Если C-рантайм/библиотека не собраны или нет компилятора — тест
пропускается, как остальные паритетные тесты.
"""

from __future__ import annotations

import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools import vtm_ref  # noqa: E402

VTM_BIN = ROOT / "src" / "vtm" / "vtm"
VTM_LIB = ROOT / "src" / "vtm" / "libvtm.a"
VTM_INCLUDE = ROOT / "src" / "vtm"

EMB_INPUT = int(vtm_ref.IFLAG_EMB_INPUT)
#: Порог задачи — 1e-6 на элемент.
TOL = 1e-6

_PROBE = r'''/* Снять y и состояние рекуррентного слоя после прогонов .vtm. */
#include "vtm.h"

#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int emit(vtm_t *m, const char *name, int step) {
    const vtm_tensor *t = vtm_tensor_by_name(m, name);
    uint32_t i;

    if (t == NULL || t->kind < VTM_TENSOR_F32 || t->kind > VTM_TENSOR_BF16
        || t->f32 == NULL) {
        fprintf(stderr, "нет fp-тензора %s\n", name);
        return 1;
    }
    printf("B %d %s %u\n", step, name, (unsigned)t->n_elements);
    for (i = 0; i < t->n_elements; i++)
        printf("%.17g\n", (double)t->f32[i]);
    return 0;
}

int main(int argc, char **argv) {
    vtm_t *m;
    const char *state_name;
    int step = 0, arg, batch = 0, i;
    int32_t ids[64];

    if (argc < 5) {
        fprintf(stderr,
                "usage: probe model.vtm state_name seq|batch token...\n");
        return 2;
    }
    setlocale(LC_NUMERIC, "C");
    state_name = argv[2];
    batch = strcmp(argv[3], "batch") == 0;
    m = vtm_load(argv[1]);
    if (m == NULL) {
        fprintf(stderr, "vtm_load: %s\n", vtm_last_error());
        return 1;
    }
    if (batch) {
        int n = argc - 4;
        if (n > 64) n = 64;
        for (i = 0; i < n; i++) ids[i] = (int32_t)atoi(argv[4 + i]);
        if (vtm_run_batch(m, ids, n) != VTM_OK) {
            fprintf(stderr, "vtm_run_batch: %s\n", vtm_error_message(m));
            vtm_free(m);
            return 1;
        }
        if (emit(m, "y", 0) != 0 || emit(m, state_name, 0) != 0) {
            vtm_free(m);
            return 1;
        }
        vtm_free(m);
        return 0;
    }
    for (arg = 4; arg < argc; arg++) {
        vtm_set_position(m, step);
        vtm_set_token(m, atoi(argv[arg]));
        if (vtm_run(m, NULL, NULL) != VTM_OK) {
            fprintf(stderr, "vtm_run %d: %s\n", step, vtm_error_message(m));
            vtm_free(m);
            return 1;
        }
        if (emit(m, "y", step) != 0 || emit(m, state_name, step) != 0) {
            vtm_free(m);
            return 1;
        }
        step++;
    }
    /* Сброс и повтор первого токена: состояние нулевое, числа — как в
     * новом execute(). */
    vtm_reset(m);
    vtm_set_position(m, 0);
    vtm_set_token(m, atoi(argv[4]));
    if (vtm_run(m, NULL, NULL) != VTM_OK) {
        fprintf(stderr, "vtm_run reset: %s\n", vtm_error_message(m));
        vtm_free(m);
        return 1;
    }
    if (emit(m, "y", -1) != 0 || emit(m, state_name, -1) != 0) {
        vtm_free(m);
        return 1;
    }
    vtm_free(m);
    return 0;
}
'''


def _parse_probe(text: str) -> dict[tuple[int, str], list[float]]:
    lines = text.splitlines()
    blocks: dict[tuple[int, str], list[float]] = {}
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) != 4 or parts[0] != "B":
            raise AssertionError(f"непонятная строка пробы: {lines[i]!r}")
        step = int(parts[1])
        name = parts[2]
        count = int(parts[3])
        values = [float(value) for value in lines[i + 1:i + 1 + count]]
        if len(values) != count:
            raise AssertionError("проба оборвалась на значениях")
        blocks[(step, name)] = values
        i += 1 + count
    return blocks


def conv4_model(*, width: int, w: list[float], x_table: list[float],
                n_tokens: int) -> vtm_ref.VTM:
    tensors = [
        vtm_ref.Tensor.f32("x", (width,), [0.0] * width),
        vtm_ref.Tensor.f32("w", (width, 4), w),
        vtm_ref.Tensor.state("conv.state", (3, width)),
        vtm_ref.Tensor.f32("y", (width,), [0.0] * width),
        vtm_ref.Tensor.f32("x.table", (n_tokens, width), x_table),
    ]
    program = [
        vtm_ref.Instr(op=int(vtm_ref.Op.EMB), flags=EMB_INPUT, dst=0, a=4),
        vtm_ref.Instr(op=int(vtm_ref.Op.CONV4), dst=3, a=0, b=1, c=2),
    ]
    return vtm_ref.VTM(tensors=tensors, program=program, meta={})


def gdelta_model(*, n_k_heads: int, n_v_heads: int, key_dim: int,
                 value_dim: int, q_tokens, k_tokens, v_tokens,
                 beta_tokens, g_tokens, z_tokens, norm_w,
                 eps: int = 0) -> vtm_ref.VTM:
    n_keys = n_k_heads * key_dim
    n_values = n_v_heads * value_dim
    tensors = [
        vtm_ref.Tensor.f32("q", (n_k_heads, key_dim), [0.0] * n_keys),
        vtm_ref.Tensor.f32("k", (n_k_heads, key_dim), [0.0] * n_keys),
        vtm_ref.Tensor.state(
            "gd.state", (n_v_heads, key_dim, value_dim)
        ),
        vtm_ref.Tensor.f32("v", (n_v_heads, value_dim), [0.0] * n_values),
        vtm_ref.Tensor.f32("beta", (n_v_heads,), [0.0] * n_v_heads),
        vtm_ref.Tensor.f32("g", (n_v_heads,), [0.0] * n_v_heads),
        vtm_ref.Tensor.f32("z", (n_v_heads, value_dim), [0.0] * n_values),
        vtm_ref.Tensor.f32("norm_w", (value_dim,), norm_w),
        vtm_ref.Tensor.f32("y", (n_v_heads, value_dim), [0.0] * n_values),
    ]
    tables = {
        "q": (q_tokens, (len(q_tokens), n_keys)),
        "k": (k_tokens, (len(k_tokens), n_keys)),
        "v": (v_tokens, (len(v_tokens), n_values)),
        "beta": (beta_tokens, (len(beta_tokens), n_v_heads)),
        "g": (g_tokens, (len(g_tokens), n_v_heads)),
        "z": (z_tokens, (len(z_tokens), n_values)),
    }
    table_index: dict[str, int] = {}
    for name, (rows, dims) in tables.items():
        table_index[name] = len(tensors)
        flat = [float(value) for row in rows for value in row]
        tensors.append(vtm_ref.Tensor.f32(f"{name}.table", dims, flat))
    program = [
        vtm_ref.Instr(op=int(vtm_ref.Op.EMB), flags=EMB_INPUT, dst=0,
                      a=table_index["q"]),
        vtm_ref.Instr(op=int(vtm_ref.Op.EMB), flags=EMB_INPUT, dst=1,
                      a=table_index["k"]),
        vtm_ref.Instr(op=int(vtm_ref.Op.EMB), flags=EMB_INPUT, dst=3,
                      a=table_index["v"]),
        vtm_ref.Instr(op=int(vtm_ref.Op.EMB), flags=EMB_INPUT, dst=4,
                      a=table_index["beta"]),
        vtm_ref.Instr(op=int(vtm_ref.Op.EMB), flags=EMB_INPUT, dst=5,
                      a=table_index["g"]),
        vtm_ref.Instr(op=int(vtm_ref.Op.EMB), flags=EMB_INPUT, dst=6,
                      a=table_index["z"]),
        vtm_ref.Instr(op=int(vtm_ref.Op.GDELTA), dst=8, a=0, b=1, c=2,
                      imm=int(eps)),
    ]
    return vtm_ref.VTM(tensors=tensors, program=program, meta={})


class _GatedParityCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.access(VTM_BIN, os.X_OK):
            raise unittest.SkipTest(
                "C-рантайм не собран: src/vtm/vtm отсутствует "
                "(cd src/vtm && make)"
            )
        if not VTM_LIB.is_file():
            raise unittest.SkipTest(
                "C-библиотека не собрана: src/vtm/libvtm.a отсутствует "
                "(cd src/vtm && make)"
            )
        cls.cc = (os.environ.get("CC") or shutil.which("cc")
                  or shutil.which("clang") or shutil.which("gcc"))
        if not cls.cc:
            raise unittest.SkipTest("нет C-компилятора (cc/clang/gcc)")
        cls.workdir = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.workdir.cleanup)
        workdir = Path(cls.workdir.name)
        source = workdir / "gated_probe.c"
        source.write_text(_PROBE, encoding="utf-8")
        cls.probe = workdir / "gated_probe"
        proc = subprocess.run(
            [
                cls.cc, "-std=c99", "-O2", "-Wall", "-Wextra",
                "-I", str(VTM_INCLUDE),
                "-o", str(cls.probe), str(source),
                str(VTM_LIB), "-lm", "-lpthread",
            ],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise unittest.SkipTest(
                "C-проба не собралась: " + proc.stderr.strip()[-500:]
            )

    def _write(self, model: vtm_ref.VTM, path: Path) -> None:
        vtm_ref.write_vtm(path, model.tensors, model.program, model.meta)

    def _probe(self, path: Path, state_name: str, tokens, mode="seq"):
        proc = subprocess.run(
            [str(self.probe), str(path), state_name, mode,
             *[str(token) for token in tokens]],
            capture_output=True, text=True, timeout=180,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return _parse_probe(proc.stdout)

    def _assert_sequence(self, path: Path, state_name: str, tokens,
                         blocks) -> None:
        for step in range(len(tokens)):
            expected = vtm_ref.execute(path, list(tokens[:step + 1]), 0)
            for name, key in (("y", "y"), (state_name, state_name)):
                got = blocks[(step, name)]
                want = expected.get(key)
                self.assertIsNotNone(want, f"эталон не вернул {key}")
                self.assertEqual(len(got), len(want), f"{name} шаг {step}")
                delta = max(
                    (abs(a - b) for a, b in zip(got, want)), default=0.0
                )
                self.assertLessEqual(
                    delta, TOL, f"{name} шаг {step}: max|Δ|={delta:.3e}"
                )

    def _assert_reset(self, path: Path, state_name: str, tokens,
                      blocks) -> None:
        expected = vtm_ref.execute(path, [tokens[0]], 0)
        for name, key in (("y", "y"), (state_name, state_name)):
            got = blocks[(-1, name)]
            want = expected.get(key)
            self.assertIsNotNone(want)
            delta = max(
                (abs(a - b) for a, b in zip(got, want)), default=0.0
            )
            self.assertLessEqual(
                delta, TOL, f"после vtm_reset {name}: max|Δ|={delta:.3e}"
            )


class TestConv4Parity(_GatedParityCase):
    def test_random_sequence(self):
        rng = random.Random(20261125)
        width = 5
        n_tokens = 4
        w = [rng.uniform(-3.0, 3.0) for _ in range(width * 4)]
        x_table = [rng.uniform(-2.0, 2.0) for _ in range(n_tokens * width)]
        model = conv4_model(width=width, w=w, x_table=x_table,
                            n_tokens=n_tokens)
        tokens = list(range(n_tokens))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "conv.vtm"
            self._write(model, path)
            state = vtm_ref.read_vtm(str(path)).by_name("conv.state")
            self.assertTrue(state.is_state)
            self.assertEqual(state.offset, vtm_ref.KV_NO_OFFSET)
            self.assertIsNone(state.data)
            blocks = self._probe(path, "conv.state", tokens)
            self._assert_sequence(path, "conv.state", tokens, blocks)
            self._assert_reset(path, "conv.state", tokens, blocks)


class TestGdeltaParity(_GatedParityCase):
    def make_model(self, eps: int):
        rng = random.Random(17013)
        n_k_heads, n_v_heads = 2, 4
        key_dim, value_dim = 3, 5
        n_tokens = 5
        q = [[rng.uniform(-2.0, 2.0) for _ in range(n_k_heads * key_dim)]
             for _ in range(n_tokens)]
        k = [[rng.uniform(-2.0, 2.0) for _ in range(n_k_heads * key_dim)]
             for _ in range(n_tokens)]
        v = [[rng.uniform(-3.0, 3.0) for _ in range(n_v_heads * value_dim)]
             for _ in range(n_tokens)]
        beta = [[rng.uniform(-2.0, 2.0) for _ in range(n_v_heads)]
                for _ in range(n_tokens)]
        # g — лог затухания, α = exp(g) ∈ (0, 1]; край g = 0 — без затухания.
        g = [[rng.uniform(-1.0, 0.0) for _ in range(n_v_heads)]
             for _ in range(n_tokens)]
        z = [[rng.uniform(-2.0, 2.0) for _ in range(n_v_heads * value_dim)]
             for _ in range(n_tokens)]
        norm_w = [rng.uniform(-2.0, 2.0) for _ in range(value_dim)]
        model = gdelta_model(
            n_k_heads=n_k_heads, n_v_heads=n_v_heads,
            key_dim=key_dim, value_dim=value_dim,
            q_tokens=q, k_tokens=k, v_tokens=v, beta_tokens=beta,
            g_tokens=g, z_tokens=z, norm_w=norm_w, eps=eps,
        )
        return model, list(range(n_tokens))

    def _run(self, eps: int):
        model, tokens = self.make_model(eps)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gd.vtm"
            self._write(model, path)
            state = vtm_ref.read_vtm(str(path)).by_name("gd.state")
            self.assertTrue(state.is_state)
            self.assertFalse(state.is_kv_cache)
            self.assertEqual(state.offset, vtm_ref.KV_NO_OFFSET)
            self.assertEqual(vtm_ref.tensor_payload_size(state), 0)
            blocks = self._probe(path, "gd.state", tokens)
            self._assert_sequence(path, "gd.state", tokens, blocks)
            self._assert_reset(path, "gd.state", tokens, blocks)
            # Пачка (vtm_run_batch) обязана дать то же состояние/выход, что
            # последовательность: состояние сквозное, токены идут по порядку.
            batch = self._probe(path, "gd.state", tokens, mode="batch")
            expected = vtm_ref.execute(path, tokens, 0)
            for name, key in (("y", "y"), ("gd.state", "gd.state")):
                got = batch[(0, name)]
                want = expected.get(key)
                self.assertIsNotNone(want)
                delta = max(
                    (abs(a - b) for a, b in zip(got, want)), default=0.0
                )
                self.assertLessEqual(
                    delta, TOL, f"пачка {name}: max|Δ|={delta:.3e}"
                )

    def test_zero_eps(self):
        self._run(0)

    def test_nonzero_eps(self):
        # imm = 1000 → eps = 1e-3: заметно и в L2-нормировке q/k, и в RMSNorm.
        self._run(1000)

    def test_qwen_head_layout_small_dims(self):
        # Форма Qwen3.5 по головам (Hk=16, Hv=32, kh = h/2), но маленькие
        # Dk = Dv = 4: проверяет раскладку банка и группировку value-голов на
        # нескольких token-шагах без 128-мерного тензора.
        rng = random.Random(3509)
        n_k_heads, n_v_heads = 16, 32
        key_dim = value_dim = 4
        n_tokens = 3
        q = [[rng.uniform(-1.0, 1.0) for _ in range(n_k_heads * key_dim)]
             for _ in range(n_tokens)]
        k = [[rng.uniform(-1.0, 1.0) for _ in range(n_k_heads * key_dim)]
             for _ in range(n_tokens)]
        v = [[rng.uniform(-1.0, 1.0) for _ in range(n_v_heads * value_dim)]
             for _ in range(n_tokens)]
        beta = [[rng.uniform(-1.0, 1.0) for _ in range(n_v_heads)]
                for _ in range(n_tokens)]
        g = [[rng.uniform(-0.5, 0.0) for _ in range(n_v_heads)]
             for _ in range(n_tokens)]
        z = [[rng.uniform(-1.0, 1.0) for _ in range(n_v_heads * value_dim)]
             for _ in range(n_tokens)]
        norm_w = [rng.uniform(-1.0, 1.0) for _ in range(value_dim)]
        model = gdelta_model(
            n_k_heads=n_k_heads, n_v_heads=n_v_heads,
            key_dim=key_dim, value_dim=value_dim,
            q_tokens=q, k_tokens=k, v_tokens=v, beta_tokens=beta,
            g_tokens=g, z_tokens=z, norm_w=norm_w, eps=0,
        )
        tokens = list(range(n_tokens))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gd_qwen_heads.vtm"
            self._write(model, path)
            blocks = self._probe(path, "gd.state", tokens)
            self._assert_sequence(path, "gd.state", tokens, blocks)

    def test_manual_numbers_from_python_test(self):
        # Ручной пример tests/test_gated_delta.py: 1/1 голова, Dk = Dv = 2,
        # q = k = [1,0], β = 0.5, α = 1, z = 1, norm_w = 1.
        silu1 = 1.0 / (1.0 + math.exp(-1.0))
        model = gdelta_model(
            n_k_heads=1, n_v_heads=1, key_dim=2, value_dim=2,
            q_tokens=[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            k_tokens=[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            v_tokens=[[2.0, 4.0], [4.0, 2.0], [1.0, 1.0]],
            beta_tokens=[[0.0], [0.0], [0.0]],
            g_tokens=[[0.0], [0.0], [0.0]],
            z_tokens=[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            norm_w=[1.0, 1.0],
        )
        want_y = [
            [0.4623620423, 0.9247240846],
            [0.8073191449, 0.6458553159],
            [0.7849748684, 0.6728356015],
        ]
        want_state = [
            [1.0, 2.0, 0.0, 0.0],
            [2.5, 2.0, 0.0, 0.0],
            [1.75, 1.5, 0.0, 0.0],
        ]
        tokens = [0, 1, 2]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gd_manual.vtm"
            self._write(model, path)
            blocks = self._probe(path, "gd.state", tokens)
            for step, (y_want, state_want) in enumerate(
                    zip(want_y, want_state)):
                y_got = blocks[(step, "y")]
                state_got = blocks[(step, "gd.state")]
                for got, want in zip(y_got, y_want):
                    self.assertLessEqual(abs(got - want), TOL,
                                         f"y шаг {step}: {got} != {want}")
                for got, want in zip(state_got, state_want):
                    self.assertLessEqual(abs(got - want), TOL,
                                         f"state шаг {step}: {got} != {want}")
            # silu(1) — ручная константа из tests/test_gated_delta.py,
            # а не вызов кода операции.
            self.assertLessEqual(abs(silu1 - 0.7310585786300049), 1e-12)


if __name__ == "__main__":
    unittest.main()
