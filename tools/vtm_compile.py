#!/usr/bin/env python3
"""Компилятор графа VTM: Qwen-подобный декодер-слой -> байткод `.vtm`.

Модуль делает три вещи:

1. `compile_layer(cfg)` — программа одного декодер-слоя;
2. `compile_model(cfg)` — эмбеддинг + N слоёв + финальный RMSN + lm_head;
3. `make_synthetic(seed, size)` — маленькая детерминированная модель с
   известным входом, на которой C-интерпретатор сверяется с Python-эталоном.

Граф слоя (порядок инструкций фиксирован, `docs/vtm.md` этап 2):

```
x ──RMSN(norm1)── h1 ──TMV(q_proj)── q ──ROPE──┐
                    ├──TMV(k_proj)── k ──ROPE──┤
                    └──TMV(v_proj)── v ────────┴──KVW(k,v @ pos)
q ──ATTN(кэш)── attn ──TMV(o_proj)── o ──ADD(x,o)── x
x ──RMSN(norm2)── h2 ─┬─TMV(gate_proj)─ gate ─SILU─┐
                      └─TMV(up_proj)─── up ──MUL── mul ──TMV(down_proj)── down
x ──ADD(x,down)── x
```

Эмбеддинг (`EMB`), финальный `RMSN`, `lm_head` (TMV по тритам или MM по
fp32-матрице) и `SAMP` + `HALT` добавляет `compile_model`.

Соглашения (совпадают с текущим `src/vtm/`, кроме масштаба тритов)
==================================================================

* позиция: `pos = state.pos + imm`, компилятор везде пишет `imm = 0`;
  хост двигает позицию между проходами (`vtm_set_position` в C);
* `EMB` с флагом `EMB_INPUT` (0x01) берёт токен прохода
  (`vtm_set_token` в C), без флага — статический `imm`;
* `ROPE`: `b` — число голов (не индекс тензора), `imm = 0`;
  флаг `ROPE_INTERLEAVED` (0x02) — пары `(2i, 2i+1)`, иначе половинки;
* `KVW`: `a` — K, `c` — V, `dst` — K-кэш `(ctx, kv_dim)`,
  `b` — V-кэш `(ctx, kv_dim)`; оба кэша не сбрасываются между проходами;
* `ATTN`: Q/out `(heads, q_len, head_dim)`, K/V — 2-D кэши, GQA через
  `group = n_heads / n_kv_heads`, `q_start = pos + imm`;
* `CPY` копирует тензор целиком;
* eps в `RMSN` — `imm/1000` (float32, как C); Qwen eps=1e-6 честно
  округляется до `imm = 0`, поэтому синтетика использует eps=1e-3
  (`imm = 1`), чтобы эталон не делил на строго нулевой RMS;
* масштабы тритов — absmax на блок 128, применяются к частичной сумме TMV
  (**единственное расхождение с текущим C**: там `scales[wi >> 3]`, 160);
* `meta["rope_theta"]` и `meta["rope_layout"]` ("half" по умолчанию) —
  обязательная часть контейнера; C v0.1 их пока не читает (theta=10000).
* инструкция: `op|flags|dst|a|b|c|reserved(2)|imm` — `imm` по смещению 12,
  как в C.

Синтетика (`make_synthetic`)
============================

`size=128` -> hidden=128, heads=4, head_dim=32, kv_heads=1, intermediate=256,
словарь=128, context_len=8. Веса — случайные триты из `random.Random(seed)`
с настоящими absmax-масштабами, вход фиксирован. В `meta["synthetic"]` лежит
вход (`tokens`), в `meta["expected"]` — эталонные логиты, их сумма и argmax:
C-тест запускает файл на этих токенах и сверяет числа (с точностью 1e-4;
при согласованных правилах совпадение побитовое).

Режимы:
  * `mode="mlp"` — только RMSN/TMV/SILU/MUL/ADD (первый гейт C-теста,
    без RoPE, KV-кэша и внимания; вход — готовый тензор `x`, один проход,
    `tokens=[]`; C вызывает `vtm_run` один раз, токен не нужен);
  * `mode="attn"` — полный слой с EMB/ROPE/KVW/ATTN (второй гейт; вход —
    эмбеддинг `emb.w` по токену прохода, токены `[0, 1, 2, 3]`).

CLI:

```
python3 tools/vtm_compile.py --synthetic --out /tmp/synth.vtm
python3 tools/vtm_compile.py --synthetic --mode mlp --size 128 --seed 1 --out /tmp/synth_mlp.vtm
python3 tools/vtm_compile.py --qwen-config qat/configs/qwen1.5b.yaml
```

`--qwen-config` — только каркас: реальные веса не читаются, файл не пишется,
печатается план (число тензоров/инструкций и оценка размера). Запись
контейнера без весов запрещена (`vtm_ref.write_vtm` бросит `ValueError`).
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import random
import re
import sys
from dataclasses import dataclass
from typing import Any, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _path in (_ROOT, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:  # импорт как пакета (tests/*, `python3 -m tools.vtm_compile`)
    from tools import vtm_ref
except ImportError:  # pragma: no cover - запуск как `python3 tools/vtm_compile.py`
    import vtm_ref  # type: ignore[no-redef]

Op = vtm_ref.Op


# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------


@dataclass
class LayerConfig:
    """Гиперпараметры одного декодер-слоя (Qwen-подобная схема)."""

    hidden: int = 128
    n_heads: int = 4
    n_kv_heads: int = 1
    head_dim: int = 32
    intermediate: int = 256
    eps: float = 1e-3
    rope_theta: float = 10000.0
    rope_layout: str = "half"
    context_len: int = 8
    block_size: int = vtm_ref.SCALE_BLOCK
    scale_dtype: str = "float32"

    def __post_init__(self) -> None:
        for name in ("hidden", "n_heads", "n_kv_heads", "head_dim", "intermediate", "context_len"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} должен быть > 0, получено {value}")
            setattr(self, name, value)
        if self.n_heads % self.n_kv_heads:
            raise ValueError(
                f"n_heads={self.n_heads} не делятся на n_kv_heads={self.n_kv_heads}"
            )
        if self.head_dim % 2:
            raise ValueError(f"head_dim={self.head_dim} должен быть чётным (RoPE)")
        if self.eps < 0:
            raise ValueError("eps не может быть отрицательным")
        if self.rope_layout not in ("half", "interleaved"):
            raise ValueError(f"rope_layout={self.rope_layout!r}, допустимо half|interleaved")
        if self.block_size != vtm_ref.SCALE_BLOCK:
            raise ValueError(
                f"VTM v0.1: масштаб только на {vtm_ref.SCALE_BLOCK} тритов, "
                f"получено block_size={self.block_size}"
            )
        if self.scale_dtype not in ("float32", "float16"):
            raise ValueError(f"scale_dtype={self.scale_dtype!r}")

    @property
    def q_dim(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def eps_imm(self) -> int:
        return int(round(self.eps * 1000.0))


@dataclass
class ModelConfig(LayerConfig):
    """Полная модель: словарь, число слоёв и способ lm_head."""

    vocab: int = 128
    n_layers: int = 1
    #: "mm" — плотная fp32-матрица через MM (политика QAT: lm_head не
    #: тернаризуется), "tmv" — тритовая матрица.
    lm_head: str = "mm"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.vocab <= 0:
            raise ValueError("vocab должен быть > 0")
        if self.n_layers <= 0:
            raise ValueError("n_layers должен быть > 0")
        if self.lm_head not in ("tmv", "mm"):
            raise ValueError(f"lm_head={self.lm_head!r}, допустимо tmv|mm")


# ---------------------------------------------------------------------------
# Построение графа
# ---------------------------------------------------------------------------


class Graph:
    """Буфер тензоров и инструкций; индекс в списке = индекс в таблице."""

    def __init__(self, cfg: LayerConfig) -> None:
        self.cfg = cfg
        self.tensors: list[vtm_ref.Tensor] = []
        self.program: list[vtm_ref.Instr] = []
        self._names: dict[str, int] = {}

    def add(self, tensor: vtm_ref.Tensor) -> int:
        if tensor.name in self._names:
            raise ValueError(f"повторяющееся имя тензора в графе: {tensor.name!r}")
        index = len(self.tensors)
        self.tensors.append(tensor)
        self._names[tensor.name] = index
        return index

    def act(self, name: str, dims: Sequence[int]) -> int:
        """Рабочий fp32-тензор (активация), инициализированный нулями."""
        return self.add(vtm_ref.Tensor.f32(name, dims, [0.0] * vtm_ref.prod(dims)))

    def emit(
        self,
        op: vtm_ref.Op,
        dst: int = 0,
        a: int = 0,
        b: int = 0,
        c: int = 0,
        flags: int = 0,
        imm: int = 0,
    ) -> None:
        self.program.append(vtm_ref.Instr(int(op), flags, dst, a, b, c, imm))


def _spec_trits(spec: Any, n_values: int) -> tuple[list[int], list[float]]:
    """Достать (триты, масштабы) из спецификации веса."""
    if isinstance(spec, vtm_ref.Tensor):
        if spec.kind != vtm_ref.KIND_TRITS:
            raise ValueError(f"вес {spec.name!r}: ожидались триты, получено {spec.kind_name}")
        trits = list(spec.trits or [])
        scales = list(spec.scales or [])
        if len(trits) != n_values:
            raise ValueError(f"вес {spec.name!r}: {len(trits)} тритов, ожидалось {n_values}")
        return trits, scales
    if isinstance(spec, dict):
        if "values" in spec:
            return vtm_ref.quantize_trits(spec["values"])
        if "trits" not in spec:
            raise ValueError("спецификация веса: нужен ключ 'trits' или 'values'")
        trits = [int(t) for t in spec["trits"]]
        if len(trits) != n_values:
            raise ValueError(f"тритов {len(trits)}, ожидалось {n_values}")
        scales = spec.get("scales")
        if scales is None:
            scales = vtm_ref.Tensor.ternary("tmp", (1, n_values), trits).scales or []
        return trits, [float(s) for s in scales]
    if isinstance(spec, (list, tuple)):
        return vtm_ref.quantize_trits(spec)
    raise ValueError(f"непонятная спецификация веса: {type(spec).__name__}")


def _spec_floats(spec: Any, n_values: int) -> list[float]:
    """Достать плоский список float из спецификации fp-веса."""
    if isinstance(spec, vtm_ref.Tensor):
        values = spec.dense()
    elif isinstance(spec, dict):
        values = spec.get("values", spec.get("data"))
    else:
        values = spec
    if values is None:
        raise ValueError("спецификация fp-веса без значений")
    result = [float(v) for v in values]
    if len(result) != n_values:
        raise ValueError(f"значений {len(result)}, ожидалось {n_values}")
    return result


def matrix_tensor(
    graph: Graph,
    name: str,
    rows: int,
    cols: int,
    spec: Any,
    dense: bool = False,
) -> int:
    """Добавить матрицу весов: тритовую (TMV) или плотную fp32 (MM/каркас)."""
    n_values = rows * cols
    if spec is None:
        kind = vtm_ref.KIND_F32 if dense else vtm_ref.KIND_TRITS
        return graph.add(vtm_ref.Tensor.placeholder(name, (rows, cols), kind))
    if dense:
        return graph.add(vtm_ref.Tensor.f32(name, (rows, cols), _spec_floats(spec, n_values)))
    trits, scales = _spec_trits(spec, n_values)
    return graph.add(
        vtm_ref.Tensor.ternary(
            name, (rows, cols), trits, scales, graph.cfg.scale_dtype
        )
    )


def vector_tensor(graph: Graph, name: str, dims: Sequence[int], spec: Any) -> int:
    """Добавить вектор весов нормы (fp32)."""
    n_values = vtm_ref.prod(dims)
    if spec is None:
        return graph.add(vtm_ref.Tensor.placeholder(name, dims, vtm_ref.KIND_F32))
    return graph.add(vtm_ref.Tensor.f32(name, dims, _spec_floats(spec, n_values)))


# ---------------------------------------------------------------------------
# Слой
# ---------------------------------------------------------------------------


def compile_layer(
    cfg: LayerConfig,
    weights: dict[str, Any] | None = None,
    graph: Graph | None = None,
    prefix: str = "l0",
    x_index: int | None = None,
) -> tuple[Graph, int]:
    """Собрать программу одного декодер-слоя.

    `weights` — словарь с ключами `norm1`, `norm2` (векторы fp32 длины hidden)
    и `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`,
    `down_proj` (матрицы). Матрицу можно задать списком float (тогда она
    квантуется в триты), словарём `{"trits": [...], "scales": [...]}`,
    `{"values": [...]}` или `vtm_ref.Tensor`. `None` — каркас без весов.

    `x_index` — индекс общего residual-тензора `x`; если `None`, слой создаёт
    свой `f"{prefix}.x"`. Возвращает `(Graph, индекс x)` — так слои
    выстраиваются в модель, не копируя residual-поток.

    Программа слоя не содержит `HALT`: в модели за слоями идут финальный
    RMSN, lm_head и `SAMP`.
    """
    g = graph if graph is not None else Graph(cfg)
    w = weights or {}
    hidden = cfg.hidden
    ctx = cfg.context_len

    if x_index is None:
        x_index = g.act(f"{prefix}.x", (hidden,))
    x = x_index

    # --- self-attention -----------------------------------------------------
    h1 = g.act(f"{prefix}.h1", (hidden,))
    norm1 = vector_tensor(g, f"{prefix}.norm1.w", (hidden,), w.get("norm1"))
    g.emit(Op.RMSN, dst=h1, a=x, b=norm1, imm=cfg.eps_imm)

    # Q/K/V — (heads, 1, head_dim): так их видит и Python, и C (ATTN ждёт 3-D).
    q = g.act(f"{prefix}.q", (cfg.n_heads, 1, cfg.head_dim))
    k = g.act(f"{prefix}.k", (cfg.n_kv_heads, 1, cfg.head_dim))
    v = g.act(f"{prefix}.v", (cfg.n_kv_heads, 1, cfg.head_dim))
    q_w = matrix_tensor(g, f"{prefix}.q_proj.w", cfg.q_dim, hidden, w.get("q_proj"))
    k_w = matrix_tensor(g, f"{prefix}.k_proj.w", cfg.kv_dim, hidden, w.get("k_proj"))
    v_w = matrix_tensor(g, f"{prefix}.v_proj.w", cfg.kv_dim, hidden, w.get("v_proj"))
    g.emit(Op.TMV, dst=q, a=q_w, b=h1)
    g.emit(Op.TMV, dst=k, a=k_w, b=h1)
    g.emit(Op.TMV, dst=v, a=v_w, b=h1)

    rope_flags = (
        vtm_ref.IFLAG_ROPE_INTERLEAVED if cfg.rope_layout == "interleaved" else 0
    )
    g.emit(Op.ROPE, dst=q, a=q, b=cfg.n_heads, flags=rope_flags, imm=0)
    g.emit(Op.ROPE, dst=k, a=k, b=cfg.n_kv_heads, flags=rope_flags, imm=0)

    k_cache = g.act(f"{prefix}.kcache", (ctx, cfg.kv_dim))
    v_cache = g.act(f"{prefix}.vcache", (ctx, cfg.kv_dim))
    g.emit(Op.KVW, dst=k_cache, a=k, b=v_cache, c=v, imm=0)

    attn = g.act(f"{prefix}.attn", (cfg.n_heads, 1, cfg.head_dim))
    g.emit(
        Op.ATTN, dst=attn, a=q, b=k_cache, c=v_cache,
        flags=vtm_ref.IFLAG_CAUSAL, imm=0,
    )

    out = g.act(f"{prefix}.o", (hidden,))
    o_w = matrix_tensor(g, f"{prefix}.o_proj.w", hidden, cfg.q_dim, w.get("o_proj"))
    g.emit(Op.TMV, dst=out, a=o_w, b=attn)
    g.emit(Op.ADD, dst=x, a=x, b=out)

    # --- FFN ----------------------------------------------------------------
    h2 = g.act(f"{prefix}.h2", (hidden,))
    norm2 = vector_tensor(g, f"{prefix}.norm2.w", (hidden,), w.get("norm2"))
    g.emit(Op.RMSN, dst=h2, a=x, b=norm2, imm=cfg.eps_imm)

    inter = cfg.intermediate
    gate = g.act(f"{prefix}.gate", (inter,))
    up = g.act(f"{prefix}.up", (inter,))
    gate_w = matrix_tensor(g, f"{prefix}.gate_proj.w", inter, hidden, w.get("gate_proj"))
    up_w = matrix_tensor(g, f"{prefix}.up_proj.w", inter, hidden, w.get("up_proj"))
    g.emit(Op.TMV, dst=gate, a=gate_w, b=h2)
    g.emit(Op.TMV, dst=up, a=up_w, b=h2)
    g.emit(Op.SILU, dst=gate, a=gate)

    mul = g.act(f"{prefix}.mul", (inter,))
    g.emit(Op.MUL, dst=mul, a=gate, b=up)

    down = g.act(f"{prefix}.down", (hidden,))
    down_w = matrix_tensor(g, f"{prefix}.down_proj.w", hidden, inter, w.get("down_proj"))
    g.emit(Op.TMV, dst=down, a=down_w, b=mul)
    g.emit(Op.ADD, dst=x, a=x, b=down)

    return g, x


# ---------------------------------------------------------------------------
# Модель
# ---------------------------------------------------------------------------


def _as_layer_weights(weights: dict[str, Any] | None, n_layers: int) -> list[Any]:
    if not weights:
        return [None] * n_layers
    layers = weights.get("layers")
    if layers is None:
        return [None] * n_layers
    if len(layers) != n_layers:
        raise ValueError(f"слоёв в весах {len(layers)}, а в конфиге {n_layers}")
    return list(layers)


def compile_model(
    cfg: ModelConfig,
    weights: dict[str, Any] | None = None,
    meta_extra: dict[str, Any] | None = None,
) -> vtm_ref.VTM:
    """Собрать программу всей модели: EMB -> N слоёв -> RMSN -> lm_head.

    `weights` (все ключи опциональны): `embed` (vocab×hidden fp32), `layers`
    (список словарей для `compile_layer`), `final_norm` (hidden), `lm_head`
    (матрица; для `lm_head="tmv"` — триты, для `"mm"` — плотная fp32).
    Без весов получается каркас: его нельзя записать в файл, но можно оценить
    по размеру.
    """
    g = Graph(cfg)
    w = weights or {}
    hidden = cfg.hidden

    # --- эмбеддинг ----------------------------------------------------------
    embed = matrix_tensor(g, "emb.w", cfg.vocab, hidden, w.get("embed"), dense=True)
    x = g.act("x", (hidden,))
    g.emit(Op.EMB, dst=x, a=embed, flags=vtm_ref.IFLAG_EMB_INPUT, imm=0)

    # --- слои ---------------------------------------------------------------
    layers = _as_layer_weights(w, cfg.n_layers)
    for index, layer_weights in enumerate(layers):
        compile_layer(
            cfg,
            layer_weights if isinstance(layer_weights, dict) else None,
            graph=g,
            prefix=f"l{index}",
            x_index=x,
        )

    # --- финальная норма и lm_head -----------------------------------------
    final_norm = vector_tensor(g, "final_norm.w", (hidden,), w.get("final_norm"))
    hfinal = g.act("hfinal", (hidden,))
    g.emit(Op.RMSN, dst=hfinal, a=x, b=final_norm, imm=cfg.eps_imm)

    # 2-D (vocab, 1): этого требует MM в C; TMV принимает любую форму.
    logits = g.act("logits", (cfg.vocab, 1))
    if cfg.lm_head == "tmv":
        lm_head = matrix_tensor(g, "lm_head.w", cfg.vocab, hidden, w.get("lm_head"))
        g.emit(Op.TMV, dst=logits, a=lm_head, b=hfinal)
    else:
        lm_head = matrix_tensor(
            g, "lm_head.w", cfg.vocab, hidden, w.get("lm_head"), dense=True
        )
        x_column = g.act("x.col", (hidden, 1))
        g.emit(Op.CPY, dst=x_column, a=hfinal)
        g.emit(Op.MM, dst=logits, a=lm_head, b=x_column)

    token = g.act("token", (1,))
    g.emit(Op.SAMP, dst=token, a=logits, imm=0)
    g.emit(Op.HALT)

    meta: dict[str, Any] = {
        "format": "vtm",
        "format_version": vtm_ref.VTM_VERSION,
        "generator": "tools/vtm_compile.py",
        "rope_theta": cfg.rope_theta,
        "rope_layout": cfg.rope_layout,
        "n_layers": cfg.n_layers,
        "model": dataclasses.asdict(cfg),
        "sampling": "greedy",
    }
    if meta_extra:
        meta.update(meta_extra)
    return vtm_ref.VTM(tensors=g.tensors, program=g.program, meta=meta)


# ---------------------------------------------------------------------------
# Синтетика
# ---------------------------------------------------------------------------


def _synth_matrix(
    graph: Graph, rng: random.Random, name: str, rows: int, cols: int
) -> int:
    """Случайная тритовая матрица с настоящими absmax-масштабами."""
    values = [rng.uniform(-1.0, 1.0) for _ in range(rows * cols)]
    trits, scales = vtm_ref.quantize_trits(values)
    return graph.add(vtm_ref.Tensor.ternary(name, (rows, cols), trits, scales))


def _synth_layer_weights(
    cfg: ModelConfig, rng: random.Random
) -> dict[str, Any]:
    """Веса слоя в виде, который принимает `compile_layer` (уже триты)."""
    hidden, inter = cfg.hidden, cfg.intermediate
    norms = lambda: [rng.uniform(0.5, 1.5) for _ in range(hidden)]  # noqa: E731

    def matrix(rows: int, cols: int) -> dict[str, Any]:
        values = [rng.uniform(-1.0, 1.0) for _ in range(rows * cols)]
        trits, scales = vtm_ref.quantize_trits(values)
        return {"trits": trits, "scales": scales}

    return {
        "norm1": norms(),
        "norm2": norms(),
        "q_proj": matrix(cfg.q_dim, hidden),
        "k_proj": matrix(cfg.kv_dim, hidden),
        "v_proj": matrix(cfg.kv_dim, hidden),
        "o_proj": matrix(hidden, cfg.q_dim),
        "gate_proj": matrix(inter, hidden),
        "up_proj": matrix(inter, hidden),
        "down_proj": matrix(hidden, inter),
    }


def make_synthetic(seed: int = 1, size: int = 128, mode: str = "attn") -> vtm_ref.VTM:
    """Маленькая детерминированная модель с известным входом.

    `mode="mlp"` — один проход без внимания (первый гейт C-теста),
    `mode="attn"` — полный слой с RoPE/KV-кэшем/ATTN (второй гейт).
    Возвращает `VTM`, в `meta["synthetic"]` — вход, в `meta["expected"]` —
    контрольные числа (`logits`, `sum`, `argmax`).
    """
    if mode not in ("mlp", "attn"):
        raise ValueError(f"mode={mode!r}, допустимо mlp|attn")
    if size <= 0 or size % 8:
        raise ValueError("size должен быть положительным и кратным 8")
    if size // 4 < 2 or (size // 4) % 2:
        raise ValueError(f"head_dim={size // 4} должен быть чётным и >= 2")

    hidden = size
    cfg = ModelConfig(
        hidden=hidden,
        n_heads=4,
        n_kv_heads=1,
        head_dim=hidden // 4,
        intermediate=2 * hidden,
        vocab=hidden,
        n_layers=1,
        context_len=8,
        eps=1e-3,
        rope_theta=10000.0,
        rope_layout="half",
    )
    rng = random.Random(int(seed))
    g = Graph(cfg)

    if mode == "mlp":
        x = g.add(
            vtm_ref.Tensor.f32("x", (hidden,), [rng.uniform(-1.0, 1.0) for _ in range(hidden)])
        )
        h1 = g.act("h1", (hidden,))
        norm1 = vector_tensor(
            g, "norm1.w", (hidden,), [rng.uniform(0.5, 1.5) for _ in range(hidden)]
        )
        g.emit(Op.RMSN, dst=h1, a=x, b=norm1, imm=cfg.eps_imm)

        gate = g.act("gate", (cfg.intermediate,))
        up = g.act("up", (cfg.intermediate,))
        gate_w = _synth_matrix(g, rng, "gate_proj.w", cfg.intermediate, hidden)
        up_w = _synth_matrix(g, rng, "up_proj.w", cfg.intermediate, hidden)
        g.emit(Op.TMV, dst=gate, a=gate_w, b=h1)
        g.emit(Op.TMV, dst=up, a=up_w, b=h1)
        g.emit(Op.SILU, dst=gate, a=gate)
        mul = g.act("mul", (cfg.intermediate,))
        g.emit(Op.MUL, dst=mul, a=gate, b=up)

        down = g.act("down", (hidden,))
        down_w = _synth_matrix(g, rng, "down_proj.w", hidden, cfg.intermediate)
        g.emit(Op.TMV, dst=down, a=down_w, b=mul)
        g.emit(Op.ADD, dst=x, a=x, b=down)

        h2 = g.act("h2", (hidden,))
        norm2 = vector_tensor(
            g, "norm2.w", (hidden,), [rng.uniform(0.5, 1.5) for _ in range(hidden)]
        )
        g.emit(Op.RMSN, dst=h2, a=x, b=norm2, imm=cfg.eps_imm)

        logits = g.act("logits", (cfg.vocab,))
        lm_head = _synth_matrix(g, rng, "lm_head.w", cfg.vocab, hidden)
        g.emit(Op.TMV, dst=logits, a=lm_head, b=h2)
        g.emit(Op.HALT)
        tokens: list[int] = []
    else:
        # Вход — через эмбеддинг и EMB_INPUT: так синтетику прогоняет и C
        # (хост ставит vtm_set_token перед каждым vtm_run).
        emb = g.add(
            vtm_ref.Tensor.f32(
                "emb.w",
                (cfg.vocab, hidden),
                [rng.uniform(-1.0, 1.0) for _ in range(cfg.vocab * hidden)],
            )
        )
        x = g.act("x", (hidden,))
        g.emit(Op.EMB, dst=x, a=emb, flags=vtm_ref.IFLAG_EMB_INPUT, imm=0)

        layer_weights = _synth_layer_weights(cfg, rng)
        compile_layer(cfg, layer_weights, graph=g, prefix="l0", x_index=x)

        final_norm = vector_tensor(
            g, "final_norm.w", (hidden,), [rng.uniform(0.5, 1.5) for _ in range(hidden)]
        )
        hfinal = g.act("hfinal", (hidden,))
        g.emit(Op.RMSN, dst=hfinal, a=x, b=final_norm, imm=cfg.eps_imm)

        logits = g.act("logits", (cfg.vocab, 1))
        lm_head = _synth_matrix(g, rng, "lm_head.w", cfg.vocab, hidden)
        g.emit(Op.TMV, dst=logits, a=lm_head, b=hfinal)
        g.emit(Op.HALT)
        tokens = [0, 1, 2, 3]

    meta: dict[str, Any] = {
        "format": "vtm",
        "format_version": vtm_ref.VTM_VERSION,
        "generator": "tools/vtm_compile.py",
        "rope_theta": cfg.rope_theta,
        "rope_layout": cfg.rope_layout,
        "n_layers": cfg.n_layers,
        "model": dataclasses.asdict(cfg),
        "synthetic": {
            "seed": int(seed),
            "size": int(size),
            "mode": mode,
            "tokens": tokens,
            "max_new": 0,
            "eps_imm": cfg.eps_imm,
            "scale_block": vtm_ref.SCALE_BLOCK,
            "inputs_tensor": "emb.w" if mode == "attn" else "x",
        },
    }
    vtm = vtm_ref.VTM(tensors=g.tensors, program=g.program, meta=meta)
    result = vtm_ref.execute(vtm, tokens, 0)
    if result.logits is None:  # pragma: no cover - защита от ошибки в графе
        raise RuntimeError("синтетика не дала логитов: в программе нет SAMP и тензора 'logits'")
    logits = result.logits
    meta["expected"] = {
        "logits": logits,
        "sum": sum(logits),
        "argmax": vtm_ref._argmax(logits),
        "argmax_logit": logits[vtm_ref._argmax(logits)],
        "first8": logits[:8],
    }
    return vtm


# ---------------------------------------------------------------------------
# Qwen-каркас: только план, без весов
# ---------------------------------------------------------------------------

#: Архитектуры, которые мы знаем по имени (значения из HF-конфигов).
_KNOWN_ARCH: dict[str, dict[str, Any]] = {
    "Qwen/Qwen2.5-1.5B-Instruct": {
        "hidden": 1536,
        "n_heads": 12,
        "n_kv_heads": 2,
        "head_dim": 128,
        "intermediate": 8960,
        "vocab": 151936,
        "n_layers": 28,
        "rope_theta": 1_000_000.0,
    },
    "Qwen/Qwen2.5-1.5B": {
        "hidden": 1536,
        "n_heads": 12,
        "n_kv_heads": 2,
        "head_dim": 128,
        "intermediate": 8960,
        "vocab": 151936,
        "n_layers": 28,
        "rope_theta": 1_000_000.0,
    },
}


def read_model_name(config_path: str | os.PathLike[str]) -> str:
    """Достать `model.name_or_path` из YAML без PyYAML (он не в stdlib)."""
    with open(config_path, "r", encoding="utf-8") as handle:
        text = handle.read()
    match = re.search(r"^\s*name_or_path:\s*[\"']?([^\"'#\s]+)", text, re.MULTILINE)
    if not match:
        raise ValueError(f"в {config_path} нет model.name_or_path")
    return match.group(1)


def qwen_plan(config_path: str | os.PathLike[str]) -> tuple[str, ModelConfig, vtm_ref.VTM]:
    """Собрать каркас модели по Qwen-конфигу (веса не читаются)."""
    name = read_model_name(config_path)
    arch = _KNOWN_ARCH.get(name)
    if arch is None:
        known = ", ".join(sorted(_KNOWN_ARCH))
        raise ValueError(
            f"архитектура {name!r} неизвестна компилятору v0.1; известны: {known}"
        )
    cfg = ModelConfig(**arch)
    vtm = compile_model(cfg)
    return name, cfg, vtm


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_synthetic(vtm: vtm_ref.VTM, path: str | None) -> None:
    synthetic = vtm.meta["synthetic"]
    expected = vtm.meta["expected"]
    ops = vtm.op_counts()
    print(f"VTM синтетика: mode={synthetic['mode']} size={synthetic['size']} "
          f"seed={synthetic['seed']}")
    print(f"тензоров: {len(vtm.tensors)}, инструкций: {len(vtm.program)}")
    print("операции: " + ", ".join(f"{name}×{count}" for name, count in sorted(ops.items())))
    print(f"вход: tokens={synthetic['tokens']} (inputs={synthetic['inputs_tensor']})")
    print(f"эталон: sum={expected['sum']:.9f} argmax={expected['argmax']} "
          f"logit={expected['argmax_logit']:.9f}")
    print("logits[0:8]: " + " ".join(f"{value:.9f}" for value in expected["first8"]))
    if path:
        print(f"файл: {path} ({os.path.getsize(path)} байт)")
        loaded = vtm_ref.read_vtm(path)
        check = vtm_ref.execute(loaded, synthetic["tokens"], synthetic["max_new"])
        if check.logits is None:
            raise RuntimeError("после чтения файла логитов нет")
        delta = max(
            abs(a - b) for a, b in zip(check.logits, expected["logits"])
        )
        print(f"round-trip: тензоров {len(loaded.tensors)}, инструкций "
              f"{len(loaded.program)}, max|Δ логитов| = {delta:.3e}")
    else:
        print("файл не записан (--out не задан)")
    print("сверка C: логиты последнего прохода против meta.expected.logits, допуск 1e-4")


def _print_plan(name: str, cfg: ModelConfig, vtm: vtm_ref.VTM) -> None:
    ops = vtm.op_counts()
    print(f"каркас: {name}")
    print(
        f"hidden={cfg.hidden} heads={cfg.n_heads}/{cfg.n_kv_heads} "
        f"head_dim={cfg.head_dim} intermediate={cfg.intermediate} "
        f"vocab={cfg.vocab} layers={cfg.n_layers} rope_theta={cfg.rope_theta:g}"
    )
    print(f"тензоров: {len(vtm.tensors)}, инструкций: {len(vtm.program)}")
    print("операции слоя×N: " + ", ".join(f"{k}×{v}" for k, v in sorted(ops.items())))
    weight_bytes = sum(
        vtm_ref.align_up(vtm_ref.tensor_payload_size(t))
        for t in vtm.tensors
        if t.name.endswith(".w")
    )
    total = vtm.estimated_file_size()
    print(
        f"оценка: веса {weight_bytes / 2**20:.1f} МиБ, файл ~{total / 2**20:.1f} МиБ"
    )
    print(
        "файл не записывается: реальные веса не загружены "
        "(--weights в v0.1 не реализован), это только план."
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Компилятор графа VTM (Python-эталон, без torch/numpy)"
    )
    parser.add_argument("--synthetic", action="store_true", help="собрать синтетическую модель")
    parser.add_argument("--seed", type=int, default=1, help="seed синтетики (по умолчанию 1)")
    parser.add_argument("--size", type=int, default=128, help="hidden синтетики (128)")
    parser.add_argument("--mode", choices=("mlp", "attn"), default="attn",
                        help="mlp — без внимания, attn — полный слой (по умолчанию)")
    parser.add_argument("--out", help="куда записать .vtm (для --synthetic)")
    parser.add_argument("--qwen-config", help="YAML Qwen-конфига (только план, без весов)")
    args = parser.parse_args(argv)

    if args.synthetic and args.qwen_config:
        parser.error("--synthetic и --qwen-config взаимоисключающие")
    if not args.synthetic and not args.qwen_config:
        parser.error("нужен --synthetic или --qwen-config")

    try:
        if args.qwen_config:
            name, cfg, plan = qwen_plan(args.qwen_config)
            _print_plan(name, cfg, plan)
            if args.out:
                print(f"--out {args.out} проигнорирован: контейнер без весов не пишется.")
            return 0

        vtm = make_synthetic(seed=args.seed, size=args.size, mode=args.mode)
        if args.out:
            out_dir = os.path.dirname(os.path.abspath(args.out))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            vtm_ref.write_vtm(args.out, vtm.tensors, vtm.program, vtm.meta)
    except (ValueError, OSError) as exc:
        print(f"ошибка: {exc}", file=sys.stderr)
        return 2
    _print_synthetic(vtm, args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
