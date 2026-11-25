"""CONV4/GDELTA: Python-эталон Gated Delta Net и короткой свёртки (задача 0176).

Этап 1a — **только Python**: `tools/vtm_ref.py` исполняет два новых опкода,
C-часть (`src/vtm/interp.c`) отложена до задачи после 0181. Проверяется:

1. **CONV4** (0x12) — причинная depthwise-свёртка с ядром 4 (Qwen3.5,
   `linear_conv_kernel_dim: 4`): ручной пример на двух каналах и четырёх
   токенах, состояние `[3, width]` между проходами;
2. **GDELTA** (0x13) — шаг gated delta rule (arXiv 2412.06464) плюс выходной
   RMS-гейт: ручной пример на двух шагах, L2-нормировка q/k, затухание,
   дельта-поправка, несколько value-голов на одну key-голову;
3. **свойства**: нулевое состояние и `v = 0` → нулевой выход; сброс состояния
   между `execute` (новый буфер, а не остаток прошлого прогона); в контейнере
   у состояния нет payload'а (`VTM_TENSOR_FLAG_STATE`);
4. **форма Qwen3.5**: 16 key-голов, 32 value-головы по 128, состояние f32.

Ручные числа. Все примеры считаны ниже в комментариях по формулам из
`docs/qwen35.md`; тест не переиспользует код операции, а повторяет арифметику
явно. `silu(1) = 1/(1+e⁻¹) = 0.7310585786300049`.

Запуск:
    python3 -m unittest tests.test_gated_delta -v
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools import vtm_ref  # noqa: E402

EMB_INPUT = int(vtm_ref.IFLAG_EMB_INPUT)


def silu_one() -> float:
    """`silu(1)` — независимо от `_silu` эталона: это ручная константа."""
    return 1.0 / (1.0 + math.exp(-1.0))


SILU1 = silu_one()


def _almost(case: unittest.TestCase, got, want, places: int = 6) -> None:
    case.assertEqual(len(got), len(want), f"длина {len(got)} != {len(want)}")
    for index, (left, right) in enumerate(zip(got, want)):
        case.assertAlmostEqual(
            float(left), float(right), places=places,
            msg=f"элемент {index}: {left!r} != {right!r}",
        )


#: Ручной пример CONV4: два канала, ядро `[1, 10, 100, 1000]` (+2-й канал).
#: Порядок `w[c, j]` — построчный: `[1,10,100,1000, 2,20,200,2000]`.
W_CONV4 = [1.0, 10.0, 100.0, 1000.0, 2.0, 20.0, 200.0, 2000.0]
#: Входы по токенам: t = 0..3 → `[t+1, 1]`.
X_CONV4 = [1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0]


def conv4_model(*, w=None, state_dims=(3, 2)) -> vtm_ref.VTM:
    """Крошечная программа `EMB → CONV4`: вход зависит от номера токена.

    Тензоры: `x` (2), `w` [2, 4], состояние `[3, 2]`, `y` (2), таблица входов
    [4, 2]. Инструкции: EMB (dst=0, a=4), CONV4 (dst=3, a=0, b=1, c=2).
    """
    tensors = [
        vtm_ref.Tensor.f32("x", (2,), [0.0, 0.0]),
        vtm_ref.Tensor.f32("w", (2, 4), W_CONV4 if w is None else w),
        vtm_ref.Tensor.state("conv.state", state_dims),
        vtm_ref.Tensor.f32("y", (2,), [0.0, 0.0]),
        vtm_ref.Tensor.f32("x.table", (4, 2), X_CONV4),
    ]
    program = [
        vtm_ref.Instr(op=int(vtm_ref.Op.EMB), flags=EMB_INPUT, dst=0, a=4),
        vtm_ref.Instr(op=int(vtm_ref.Op.CONV4), dst=3, a=0, b=1, c=2),
    ]
    return vtm_ref.VTM(tensors=tensors, program=program, meta={})


def gd_model(
    *,
    n_k_heads: int,
    n_v_heads: int,
    key_dim: int,
    value_dim: int,
    q_tokens,
    k_tokens,
    v_tokens,
    beta_tokens,
    g_tokens,
    z_tokens,
    norm_w,
    eps: int = 0,
) -> vtm_ref.VTM:
    """Крошечная программа `EMB×6 → GDELTA`; входы заданы по токенам.

    Банк GDELTA — шесть подряд идущих тензоров (часть ABI `vtm.h`):
    `state` [Hv, Dk, Dv] (флаг STATE), `v` [Hv, Dv], `beta` [Hv],
    `g` [Hv], `z` [Hv, Dv], `norm_w` [Dv]. `q`/`k` лежат до банка.
    """
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


def _single_head_model(**kwargs) -> vtm_ref.VTM:
    """Одна key- и одна value-голова, Dk = Dv = 2, состояние [[·,·],[·,·]]."""
    params = {
        "n_k_heads": 1, "n_v_heads": 1, "key_dim": 2, "value_dim": 2,
        "norm_w": [1.0, 1.0],
    }
    params.update(kwargs)
    return gd_model(**params)


class TestConv4(unittest.TestCase):
    """Ядро 4: `y[c] = Σ_j w[c,j]·s[j,c]`, s = [x_{t-3}, x_{t-2}, x_{t-1}, x_t]."""

    def _run(self, tokens, model=None):
        return vtm_ref.execute(model or conv4_model(), tokens, 0)

    def test_manual_two_channels(self):
        # x по токенам: [1,1], [2,1], [3,1], [4,1]; w = [1,10,100,1000] и
        # [2,20,200,2000]. Считаем вручную:
        # t0: s = [0,0,0,x0] → 1000·1 = 1000;  2000·1 = 2000
        # t1: s = [0,0,x0,x1] → 100·1 + 1000·2 = 2100;  200·1 + 2000·1 = 2200
        # t2: s = [0,x0,x1,x2] → 10·1 + 100·2 + 1000·3 = 3210;
        #                       20·1 + 200·1 + 2000·1 = 2220
        # t3: s = [x0,x1,x2,x3] → 1·1 + 10·2 + 100·3 + 1000·4 = 4321;
        #                         2·1 + 20·1 + 200·1 + 2000·1 = 2222
        expected = [
            ([1000.0, 2000.0], [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]),
            ([2100.0, 2200.0], [0.0, 0.0, 1.0, 1.0, 2.0, 1.0]),
            ([3210.0, 2220.0], [1.0, 1.0, 2.0, 1.0, 3.0, 1.0]),
            ([4321.0, 2222.0], [2.0, 1.0, 3.0, 1.0, 4.0, 1.0]),
        ]
        model = conv4_model()
        for step, (want_y, want_state) in enumerate(expected, start=1):
            result = vtm_ref.execute(model, list(range(step)), 0)
            # Выход последнего прохода — ровно y_{step-1}, а состояние — три
            # последних входа: их видно в снимке рабочей области.
            self.assertEqual(result.get("y"), want_y, f"t={step - 1}")
            self.assertEqual(result.get("conv.state"), want_state, f"t={step - 1}")

    def test_zero_state_and_zero_input(self):
        # Нулевой вход при нулевом состоянии: y = 0 и состояние остаётся
        # нулевым — свёртка не имеет смещения (bias).
        zero_model = conv4_model()
        zeros = vtm_ref.Tensor.f32("x.table", (4, 2), [0.0] * 8)
        zero_model.tensors[-1] = zeros
        result = vtm_ref.execute(zero_model, [0, 1, 2], 0)
        self.assertEqual(result.get("y"), [0.0, 0.0])
        self.assertEqual(result.get("conv.state"), [0.0] * 6)

    def test_state_reset_between_executions(self):
        model = conv4_model()
        first = vtm_ref.execute(model, [0], 0)
        long_run = vtm_ref.execute(model, [0, 1, 2, 3], 0)
        self.assertEqual(long_run.get("y"), [4321.0, 2222.0])
        # Следующий execute — новый буфер состояния: ни y, ни состояние не
        # помнят предыдущий прогон (сброс = не новый контейнер, а новый
        # интерпретатор).
        again = vtm_ref.execute(model, [0], 0)
        self.assertEqual(again.get("y"), first.get("y"))
        self.assertEqual(again.get("conv.state"), first.get("conv.state"))
        self.assertEqual(again.get("conv.state"), [0.0, 0.0, 0.0, 0.0, 1.0, 1.0])

    def test_container_round_trip_has_no_state_payload(self):
        model = conv4_model()
        with tempfile.TemporaryDirectory(prefix="gd-conv-") as tmp:
            path = Path(tmp) / "conv.vtm"
            vtm_ref.write_vtm(path, model.tensors, model.program, {})
            loaded = vtm_ref.read_vtm(str(path))
            state = loaded.by_name("conv.state")
            self.assertTrue(state.is_state)
            self.assertFalse(state.is_kv_cache)
            self.assertIsNone(state.data)
            # Состояние — буфер вне файла: payload'а нет, offset — маркер.
            self.assertEqual(vtm_ref.tensor_payload_size(state), 0)
            self.assertEqual(state.offset, vtm_ref.KV_NO_OFFSET)
            self.assertTrue(loaded.flags & vtm_ref.VFLAG_KV_OUT_OF_FILE)
            # Из файла считается то же, что из памяти.
            from_file = vtm_ref.execute(path, [0, 1, 2, 3], 0)
            from_memory = vtm_ref.execute(model, [0, 1, 2, 3], 0)
            self.assertEqual(from_file.get("y"), from_memory.get("y"))
            self.assertEqual(
                from_file.get("conv.state"), from_memory.get("conv.state")
            )

    def test_stream_writer_skips_state_payload(self):
        # Потоковый писатель (им пользуется импорт больших моделей) тоже не
        # должен писать состояние в секцию данных: только запись таблицы.
        tensors = [
            vtm_ref.Tensor.placeholder("x", (2,)),
            vtm_ref.Tensor.placeholder("w", (2, 4)),
            vtm_ref.Tensor.state("conv.state", (3, 2)),
            vtm_ref.Tensor.placeholder("y", (2,)),
            vtm_ref.Tensor.placeholder("x.table", (4, 2)),
        ]
        program = [
            vtm_ref.Instr(op=int(vtm_ref.Op.EMB), flags=EMB_INPUT, dst=0, a=4),
            vtm_ref.Instr(op=int(vtm_ref.Op.CONV4), dst=3, a=0, b=1, c=2),
        ]
        with tempfile.TemporaryDirectory(prefix="gd-stream-") as tmp:
            path = Path(tmp) / "stream.vtm"
            vtm_ref.write_vtm_stream(path, tensors, program, {})
            loaded = vtm_ref.read_vtm(str(path))
            state = loaded.by_name("conv.state")
            self.assertTrue(state.is_state)
            self.assertEqual(vtm_ref.tensor_payload_size(state), 0)
            self.assertEqual(state.offset, vtm_ref.KV_NO_OFFSET)
            self.assertTrue(loaded.flags & vtm_ref.VFLAG_KV_OUT_OF_FILE)
            # Размер файла считает тот же `tensor_payload_size`: состояния в
            # нём нет, иначе оценка разошлась бы с диском.
            self.assertEqual(
                os.path.getsize(path), loaded.estimated_file_size()
            )

    def test_shape_errors(self):
        with self.assertRaises(ValueError):
            vtm_ref.execute(
                conv4_model(w=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), [0], 0
            )
        with self.assertRaises(ValueError):
            vtm_ref.execute(conv4_model(state_dims=(2, 2)), [0], 0)


class TestGatedDeltaManual(unittest.TestCase):
    """Ручной пример GDELTA: 1 key-голова, 1 value-голова, Dk = Dv = 2."""

    def _three_step_model(self) -> vtm_ref.VTM:
        # Токен 0: q = k = [1,0], v = [2,4], beta = 0 (β = 0.5), g = 0 (α = 1),
        #          z = [1,1], norm_w = [1,1].
        #   S₁ = 0 + 0.5·v·k̂ᵀ = [[1,2],[0,0]];  u = S₀k̂ = 0;
        #   o = S₁q̂ = [1,2]; rms = √((1+4)/2) = √2.5;
        #   y = o/rms·silu(1) = [0.4623620423, 0.9247240846].
        # Токен 1: те же q,k, но v = [4,2]:
        #   u = S₁k̂ = [1,2]; γ = [4,2] − 1·[1,2] = [3,0];
        #   S₂ = [[1,2],[0,0]] + 0.5·γ·k̂ᵀ = [[2.5,2],[0,0]];
        #   o = [2.5,2]; rms = √((6.25+4)/2) = √5.125;
        #   y = [2.5,2]·silu(1)/√5.125 = [0.8073191449, 0.6458553159].
        # Токен 2: v = [1,1]:
        #   u = S₂k̂ = [2.5,2]; γ = [1,1] − [2.5,2] = [−1.5,−1];
        #   S₃ = [[2.5,2],[0,0]] + 0.5·γ·k̂ᵀ = [[1.75,1.5],[0,0]];
        #   o = [1.75,1.5]; rms = √((3.0625+2.25)/2) = √2.65625;
        #   y = [1.75,1.5]·silu(1)/√2.65625 = [0.7849748684, 0.6728356015].
        return _single_head_model(
            q_tokens=[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            k_tokens=[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            v_tokens=[[2.0, 4.0], [4.0, 2.0], [1.0, 1.0]],
            beta_tokens=[[0.0], [0.0], [0.0]],
            g_tokens=[[0.0], [0.0], [0.0]],
            z_tokens=[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            norm_w=[1.0, 1.0],
        )

    def test_manual_three_steps(self):
        model = self._three_step_model()
        first = vtm_ref.execute(model, [0], 0)
        # S₁ построчно: (dk=0) [1,2], (dk=1) [0,0].
        self.assertEqual(first.get("gd.state"), [1.0, 2.0, 0.0, 0.0])
        _almost(self, first.get("y"), [SILU1 / math.sqrt(2.5),
                                       2.0 * SILU1 / math.sqrt(2.5)])
        # Те же числа в десятичном виде — чтобы ошибка ориентации состояния
        # или перепутанный γ не прошли только за счёт «той же формулы».
        _almost(self, first.get("y"), [0.4623620423, 0.9247240846])

        second = vtm_ref.execute(model, [0, 1], 0)
        self.assertEqual(second.get("gd.state"), [2.5, 2.0, 0.0, 0.0])
        _almost(self, second.get("y"), [2.5 * SILU1 / math.sqrt(5.125),
                                        2.0 * SILU1 / math.sqrt(5.125)])
        _almost(self, second.get("y"), [0.8073191449, 0.6458553159])

        third = vtm_ref.execute(model, [0, 1, 2], 0)
        self.assertEqual(third.get("gd.state"), [1.75, 1.5, 0.0, 0.0])
        _almost(self, third.get("y"), [1.75 * SILU1 / math.sqrt(2.65625),
                                       1.5 * SILU1 / math.sqrt(2.65625)])
        _almost(self, third.get("y"), [0.7849748684, 0.6728356015])

    def test_two_value_heads_share_key_head(self):
        # Hv = 2, Hk = 1 → kh = h / 2: обе value-головы берут ключ головы 0.
        # q = k = [1,0]; β = 0.5; v₀ = [2,0], v₁ = [0,1]; α = 1.
        # S₀ = [[1,0],[0,0]], S₁ = [[0,0],[0,0.5]] (вторая голова: β·v₁k̂ᵀ).
        # o₀ = S₀q̂ = [1,0], rms = √0.5 → y₀ = [√2·silu(1), 0]
        # o₁ = [0,0.5], rms = √0.125 → y₁ = [0, √2·silu(1)].
        model = gd_model(
            n_k_heads=1, n_v_heads=2, key_dim=2, value_dim=2,
            q_tokens=[[1.0, 0.0]],
            k_tokens=[[1.0, 0.0]],
            v_tokens=[[2.0, 0.0, 0.0, 1.0]],
            beta_tokens=[[0.0, 0.0]],
            g_tokens=[[0.0, 0.0]],
            z_tokens=[[1.0, 1.0, 1.0, 1.0]],
            norm_w=[1.0, 1.0],
        )
        result = vtm_ref.execute(model, [0], 0)
        self.assertEqual(
            result.get("gd.state"),
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
        )
        root_two = math.sqrt(2.0)
        _almost(self, result.get("y"),
                [root_two * SILU1, 0.0, 0.0, root_two * SILU1])

    def test_l2_norm_of_q_and_k(self):
        # q = [3,0], k = [0,5] → q̂ = [1,0], k̂ = [0,1] (нормировка по L2).
        # S = 0.5·v·k̂ᵀ = [[0,0],[1,2]]; o = S·q̂ = [0,0] → y = 0 (q ⊥ k).
        model = _single_head_model(
            q_tokens=[[3.0, 0.0]],
            k_tokens=[[0.0, 5.0]],
            v_tokens=[[2.0, 4.0]],
            beta_tokens=[[0.0]],
            g_tokens=[[0.0]],
            z_tokens=[[1.0, 1.0]],
            norm_w=[1.0, 1.0],
        )
        result = vtm_ref.execute(model, [0], 0)
        self.assertEqual(result.get("gd.state"), [0.0, 0.0, 1.0, 2.0])
        self.assertEqual(result.get("y"), [0.0, 0.0])

    def test_output_gate_and_norm_weight(self):
        # z = 0 → silu(0) = 0, выход нулевой, хотя состояние не нулевое.
        gate_zero = _single_head_model(
            q_tokens=[[1.0, 0.0]], k_tokens=[[1.0, 0.0]],
            v_tokens=[[2.0, 4.0]], beta_tokens=[[0.0]], g_tokens=[[0.0]],
            z_tokens=[[0.0, 0.0]], norm_w=[1.0, 1.0],
        )
        result = vtm_ref.execute(gate_zero, [0], 0)
        self.assertEqual(result.get("gd.state"), [1.0, 2.0, 0.0, 0.0])
        self.assertEqual(result.get("y"), [0.0, 0.0])
        # norm_w = [2,3] взвешивает после RMS-нормировки:
        # o = [1,2], rms = √2.5, y = [2,6]·silu(1)/√2.5.
        weighted = _single_head_model(
            q_tokens=[[1.0, 0.0]], k_tokens=[[1.0, 0.0]],
            v_tokens=[[2.0, 4.0]], beta_tokens=[[0.0]], g_tokens=[[0.0]],
            z_tokens=[[1.0, 1.0]], norm_w=[2.0, 3.0],
        )
        result = vtm_ref.execute(weighted, [0], 0)
        _almost(self, result.get("y"),
                [2.0 * SILU1 / math.sqrt(2.5), 6.0 * SILU1 / math.sqrt(2.5)])


class TestGatedDeltaProperties(unittest.TestCase):
    """Свойства, на которые опирается контракт: нули, затухание, сброс."""

    def test_zero_state_with_zero_v_gives_zero_output(self):
        # При нулевом состоянии o = S_t·q̂ = β·(k̂·q̂)·v: если v = 0, то и
        # S_t = 0, и выход нулевой. Это и есть «нулевое состояние → нулевой
        # выход» в честной формулировке (при произвольном v выход не ноль!).
        model = _single_head_model(
            q_tokens=[[1.0, 0.0], [0.0, 1.0]],
            k_tokens=[[1.0, 0.0], [0.0, 1.0]],
            v_tokens=[[0.0, 0.0], [0.0, 0.0]],
            beta_tokens=[[0.0], [0.0]],
            g_tokens=[[0.0], [0.0]],
            z_tokens=[[1.0, 1.0], [1.0, 1.0]],
            norm_w=[1.0, 1.0],
        )
        result = vtm_ref.execute(model, [0, 1], 0)
        self.assertEqual(result.get("gd.state"), [0.0] * 4)
        self.assertEqual(result.get("y"), [0.0, 0.0])

    def test_zero_q_gives_zero_output_but_updates_state(self):
        # q = 0 → o = 0 при любом состоянии; состояние при этом обновляется
        # (v ≠ 0), то есть «выход 0» не значит «шаг не сделан».
        model = _single_head_model(
            q_tokens=[[0.0, 0.0]],
            k_tokens=[[1.0, 0.0]],
            v_tokens=[[2.0, 4.0]],
            beta_tokens=[[0.0]],
            g_tokens=[[0.0]],
            z_tokens=[[1.0, 1.0]],
            norm_w=[1.0, 1.0],
        )
        result = vtm_ref.execute(model, [0], 0)
        self.assertEqual(result.get("y"), [0.0, 0.0])
        self.assertEqual(result.get("gd.state"), [1.0, 2.0, 0.0, 0.0])

    def test_decay_halves_state(self):
        # Токен 0 создаёт S = [[1,2],[0,0]]. Токен 1: k = 0, v = 0, α = 0.5
        # (g = ln 0.5) → u = 0, γ = 0, S₂ = 0.5·S₁ = [[0.5,1],[0,0]].
        model = _single_head_model(
            q_tokens=[[1.0, 0.0], [0.0, 0.0]],
            k_tokens=[[1.0, 0.0], [0.0, 0.0]],
            v_tokens=[[2.0, 4.0], [0.0, 0.0]],
            beta_tokens=[[0.0], [0.0]],
            g_tokens=[[0.0], [math.log(0.5)]],
            z_tokens=[[1.0, 1.0], [1.0, 1.0]],
            norm_w=[1.0, 1.0],
        )
        first = vtm_ref.execute(model, [0], 0)
        self.assertEqual(first.get("gd.state"), [1.0, 2.0, 0.0, 0.0])
        second = vtm_ref.execute(model, [0, 1], 0)
        _almost(self, second.get("gd.state"), [0.5, 1.0, 0.0, 0.0])
        self.assertEqual(second.get("y"), [0.0, 0.0])

    def test_state_reset_between_executions(self):
        model = _single_head_model(
            q_tokens=[[1.0, 0.0], [1.0, 0.0]],
            k_tokens=[[1.0, 0.0], [1.0, 0.0]],
            v_tokens=[[2.0, 4.0], [4.0, 2.0]],
            beta_tokens=[[0.0], [0.0]],
            g_tokens=[[0.0], [0.0]],
            z_tokens=[[1.0, 1.0], [1.0, 1.0]],
            norm_w=[1.0, 1.0],
        )
        first = vtm_ref.execute(model, [0], 0)
        long_run = vtm_ref.execute(model, [0, 1], 0)
        self.assertEqual(long_run.get("gd.state"), [2.5, 2.0, 0.0, 0.0])
        # Новый execute — состояние с нуля, а не остаток прошлого прогона.
        again = vtm_ref.execute(model, [0], 0)
        self.assertEqual(again.get("gd.state"), first.get("gd.state"))
        self.assertEqual(again.get("y"), first.get("y"))

    def test_container_round_trip_has_no_state_payload(self):
        model = _single_head_model(
            q_tokens=[[1.0, 0.0]], k_tokens=[[1.0, 0.0]],
            v_tokens=[[2.0, 4.0]], beta_tokens=[[0.0]], g_tokens=[[0.0]],
            z_tokens=[[1.0, 1.0]], norm_w=[1.0, 1.0],
        )
        with tempfile.TemporaryDirectory(prefix="gd-state-") as tmp:
            path = Path(tmp) / "gd.vtm"
            vtm_ref.write_vtm(path, model.tensors, model.program, {})
            loaded = vtm_ref.read_vtm(str(path))
            state = loaded.by_name("gd.state")
            self.assertTrue(state.is_state)
            self.assertEqual(tuple(state.dims), (1, 2, 2))
            self.assertEqual(vtm_ref.tensor_payload_size(state), 0)
            self.assertEqual(state.offset, vtm_ref.KV_NO_OFFSET)
            self.assertTrue(loaded.flags & vtm_ref.VFLAG_KV_OUT_OF_FILE)
            from_file = vtm_ref.execute(path, [0], 0)
            self.assertEqual(from_file.get("gd.state"), [1.0, 2.0, 0.0, 0.0])
            _almost(self, from_file.get("y"),
                    [SILU1 / math.sqrt(2.5), 2.0 * SILU1 / math.sqrt(2.5)])

    def test_bank_and_shape_errors(self):
        # Банк из шести тензоров: обрезанный банк — отказ, а не чтение мусора.
        # Здесь в таблице только q, k, состояние, v, beta, g: c+5 = 7 за
        # границей, GDELTA обязан сказать это до чтения z/norm_w.
        truncated = vtm_ref.VTM(
            tensors=[
                vtm_ref.Tensor.f32("q", (1, 2), [1.0, 0.0]),
                vtm_ref.Tensor.f32("k", (1, 2), [1.0, 0.0]),
                vtm_ref.Tensor.state("gd.state", (1, 2, 2)),
                vtm_ref.Tensor.f32("v", (1, 2), [2.0, 4.0]),
                vtm_ref.Tensor.f32("beta", (1,), [0.0]),
                vtm_ref.Tensor.f32("g", (1,), [0.0]),
            ],
            program=[
                vtm_ref.Instr(op=int(vtm_ref.Op.GDELTA), dst=0, a=0, b=1, c=2,
                              imm=0),
            ],
            meta={},
        )
        with self.assertRaises(ValueError):
            vtm_ref.execute(truncated, [0], 0)
        # z не той формы.
        bad_z = _single_head_model(
            q_tokens=[[1.0, 0.0]], k_tokens=[[1.0, 0.0]],
            v_tokens=[[2.0, 4.0]], beta_tokens=[[0.0]], g_tokens=[[0.0]],
            z_tokens=[[1.0, 1.0]], norm_w=[1.0, 1.0],
        )
        bad_z.tensors[6] = vtm_ref.Tensor.f32("z", (1, 1), [1.0])
        with self.assertRaises(ValueError):
            vtm_ref.execute(bad_z, [0], 0)
        # Отрицательный eps (imm) запрещён.
        with self.assertRaises(ValueError):
            vtm_ref.execute(
                _single_head_model(
                    q_tokens=[[1.0, 0.0]], k_tokens=[[1.0, 0.0]],
                    v_tokens=[[2.0, 4.0]], beta_tokens=[[0.0]],
                    g_tokens=[[0.0]], z_tokens=[[1.0, 1.0]],
                    norm_w=[1.0, 1.0], eps=-1,
                ),
                [0], 0,
            )


class TestQwenShape(unittest.TestCase):
    """Форма Qwen3.5: Hk = 16, Hv = 32, Dk = Dv = 128, состояние f32."""

    def test_state_and_output_shapes(self):
        n_k_heads, n_v_heads = 16, 32
        key_dim = value_dim = 128
        # q/k/v — детерминированные маленькие значения (без random: тест
        # должен быть воспроизводим). v = 0 → при нулевом состоянии выход 0.
        q = [((index % 7) - 3) * 0.05 for index in range(n_k_heads * key_dim)]
        k = [((index % 5) - 2) * 0.05 for index in range(n_k_heads * key_dim)]
        model = gd_model(
            n_k_heads=n_k_heads, n_v_heads=n_v_heads,
            key_dim=key_dim, value_dim=value_dim,
            q_tokens=[q], k_tokens=[k],
            v_tokens=[[0.0] * (n_v_heads * value_dim)],
            beta_tokens=[[0.0] * n_v_heads],
            g_tokens=[[0.0] * n_v_heads],
            z_tokens=[[1.0] * (n_v_heads * value_dim)],
            norm_w=[1.0] * value_dim,
        )
        state = model.by_name("gd.state")
        self.assertEqual(tuple(state.dims), (n_v_heads, key_dim, value_dim))
        self.assertEqual(state.n_elements, 32 * 128 * 128)
        result = vtm_ref.execute(model, [0], 0)
        self.assertEqual(len(result.get("y")), n_v_heads * value_dim)
        self.assertEqual(result.get("y"), [0.0] * (n_v_heads * value_dim))
        self.assertEqual(result.get("gd.state"), [0.0] * state.n_elements)
        self.assertTrue(all(math.isfinite(value) for value in result.get("y")))


if __name__ == "__main__":
    unittest.main()
