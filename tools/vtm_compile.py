#!/usr/bin/env python3
"""Компилятор графа VTM: Qwen-подобный декодер-слой -> байткод `.vtm`.

Модуль делает три вещи:

1. `compile_layer(cfg)` — программа одного декодер-слоя;
2. `compile_model(cfg | manifest)` — эмбеддинг + N слоёв + финальный RMSN +
   lm_head; принимает готовые веса или манифест экспорта QAT
   (`export/manifest.json` + `ternary.trit` + `non_ternary.bin`);
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
python3 tools/vtm_compile.py --manifest export/manifest.json --out model.vtm
python3 tools/vtm_compile.py --qwen-config qat/configs/qwen1.5b.yaml
```

`--manifest` — единственный путь собрать настоящую модель: он читает
`manifest.json` экспорта QAT, тритовые сегменты `ternary.trit` и нетернарные
веса `non_ternary.bin` (эмбеддинги, финальная норма, lm_head, bias). При
`tie_word_embeddings: true` матрица lm_head не дублируется: `MM` берёт ту же
таблицу, что `EMB` (индекс `emb.w`), поэтому размер контейнера не растёт на
`vocab × hidden × 4` байт.

Если в манифесте `architecture.embed == "trit"` (задача 0043), таблица словаря
лежит не в `non_ternary.bin`, а тритовым сегментом `model.embed_tokens.weight`
в `ternary.trit`: `EMB` читает её как kind 0 (C — через LUT243), а tied lm_head
по-прежнему считается `MM` по `emb.w`, только A уже тритовая. `embed == "int8"`
компилятор отвергает: kind 4 в контейнере пока не реализован.

`--qwen-config` — только каркас: реальные веса не читаются, файл не пишется,
печатается план (число тензоров/инструкций и оценка размера). Запись
контейнера без весов запрещена (`vtm_ref.write_vtm` бросит `ValueError`).
"""

from __future__ import annotations

import argparse
import array
import dataclasses
import hashlib
import json
import os
import random
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
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
    #: Qwen/Llama-стиль: lm_head.weight — это та же таблица, что
    #: embed_tokens.weight. В контейнере отдельного тензора нет, MM читает
    #: `emb.w`; экономия vocab × hidden × 4 байт.
    tie_word_embeddings: bool = False
    #: Где лежит таблица словаря: "none" — fp32 в non_ternary.bin,
    #: "trit" — сегмент ternary.trit (kind: trit, задача 0043).
    #: "int8" в контейнере пока не поддержан (нужен kind 4).
    embed_quant: str = "none"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.embed_quant = str(self.embed_quant or "none").lower()
        if self.embed_quant not in ("none", "trit", "int8"):
            raise ValueError(
                f"embed_quant={self.embed_quant!r}, допустимо none|trit|int8"
            )
        if self.embed_quant == "int8":
            raise ValueError(
                "int8-эмбеддинги в контейнере отложены: нужен kind 4 "
                "(int8 + масштабы) и его чтение в src/vtm/loader.c и "
                "tools/vtm_ref.py; пока поддержаны none и trit"
            )
        if self.vocab <= 0:
            raise ValueError("vocab должен быть > 0")
        if self.n_layers <= 0:
            raise ValueError("n_layers должен быть > 0")
        if self.lm_head not in ("tmv", "mm"):
            raise ValueError(f"lm_head={self.lm_head!r}, допустимо tmv|mm")
        if self.tie_word_embeddings and self.lm_head == "tmv":
            raise ValueError(
                "tie_word_embeddings вместе с lm_head='tmv' невозможен: "
                "tied-таблица хранится как fp32-эмбеддинг, а TMV ждёт триты; "
                "тритовые эмбеддинги идут через EMB + MM (embed_quant='trit')"
            )


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
    """Добавить матрицу весов: тритовую (TMV) или плотную fp32 (MM/каркас).

    `spec` может быть готовым `vtm_ref.Tensor` из манифеста: тогда он
    добавляется в граф без копирования значений (важно для эмбеддингов
    1.5B-модели — это сотни мегабайт). Имя тензора приводится к короткому
    имени VTM (в таблице контейнера максимум 24 байта).
    """
    n_values = rows * cols
    if spec is None:
        kind = vtm_ref.KIND_F32 if dense else vtm_ref.KIND_TRITS
        return graph.add(vtm_ref.Tensor.placeholder(name, (rows, cols), kind))
    if isinstance(spec, vtm_ref.Tensor):
        if tuple(spec.dims) != (rows, cols):
            raise ValueError(
                f"вес {name!r}: форма {spec.dims}, ожидалась {(rows, cols)}"
            )
        if dense and spec.kind == vtm_ref.KIND_TRITS:
            raise ValueError(f"вес {name!r}: нужен плотный fp-тензор, а пришли триты")
        if not dense and spec.kind != vtm_ref.KIND_TRITS:
            raise ValueError(
                f"вес {name!r}: нужны триты, а пришёл {spec.kind_name}"
            )
        spec.name = name
        return graph.add(spec)
    if dense:
        return graph.add(vtm_ref.Tensor.f32(name, (rows, cols), _spec_floats(spec, n_values)))
    trits, scales = _spec_trits(spec, n_values)
    return graph.add(
        vtm_ref.Tensor.ternary(
            name, (rows, cols), trits, scales, graph.cfg.scale_dtype
        )
    )


def vector_tensor(graph: Graph, name: str, dims: Sequence[int], spec: Any) -> int:
    """Добавить вектор весов (нормы, bias) в fp32.

    Готовый `Tensor` переиспользуется (смена имени/формы при том же числе
    элементов) — так bias из манифеста не копируется лишний раз.
    """
    n_values = vtm_ref.prod(dims)
    if spec is None:
        return graph.add(vtm_ref.Tensor.placeholder(name, dims, vtm_ref.KIND_F32))
    if isinstance(spec, vtm_ref.Tensor):
        if spec.n_elements != n_values:
            raise ValueError(
                f"вес {name!r}: {spec.n_elements} элементов, ожидалось {n_values}"
            )
        if spec.kind == vtm_ref.KIND_TRITS:
            raise ValueError(f"вес {name!r}: нужен плотный fp-тензор, а пришли триты")
        spec.name = name
        spec.dims = tuple(int(d) for d in dims)
        return graph.add(spec)
    return graph.add(vtm_ref.Tensor.f32(name, dims, _spec_floats(spec, n_values)))


def add_bias(
    graph: Graph, dst: int, dims: Sequence[int], spec: Any, name: str
) -> None:
    """Прибавить bias к тензору `dst` (Qwen/Llama кладут bias в q/k/v)."""
    if spec is None:
        return
    bias = vector_tensor(graph, name, dims, spec)
    graph.emit(Op.ADD, dst=dst, a=dst, b=bias)


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
    `{"values": [...]}` или `vtm_ref.Tensor`. Необязательные `q_bias`,
    `k_bias`, `v_bias`, `o_bias`, `gate_bias`, `up_bias`, `down_bias`
    (векторы fp32) добавляются через `ADD`: Qwen2.5 кладёт bias в q/k/v.
    `None` — каркас без весов.

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
    add_bias(g, q, (cfg.n_heads, 1, cfg.head_dim), w.get("q_bias"), f"{prefix}.q_bias")
    add_bias(g, k, (cfg.n_kv_heads, 1, cfg.head_dim), w.get("k_bias"), f"{prefix}.k_bias")
    add_bias(g, v, (cfg.n_kv_heads, 1, cfg.head_dim), w.get("v_bias"), f"{prefix}.v_bias")

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
    add_bias(g, out, (hidden,), w.get("o_bias"), f"{prefix}.o_bias")
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
    add_bias(g, gate, (inter,), w.get("gate_bias"), f"{prefix}.gate_bias")
    add_bias(g, up, (inter,), w.get("up_bias"), f"{prefix}.up_bias")
    g.emit(Op.SILU, dst=gate, a=gate)

    mul = g.act(f"{prefix}.mul", (inter,))
    g.emit(Op.MUL, dst=mul, a=gate, b=up)

    down = g.act(f"{prefix}.down", (hidden,))
    down_w = matrix_tensor(g, f"{prefix}.down_proj.w", hidden, inter, w.get("down_proj"))
    g.emit(Op.TMV, dst=down, a=down_w, b=mul)
    add_bias(g, down, (hidden,), w.get("down_bias"), f"{prefix}.down_bias")
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
    cfg: ModelConfig | str | os.PathLike[str] | dict[str, Any],
    weights: dict[str, Any] | None = None,
    meta_extra: dict[str, Any] | None = None,
) -> vtm_ref.VTM:
    """Собрать программу всей модели: EMB -> N слоёв -> RMSN -> lm_head.

    Первый аргумент — либо `ModelConfig`, либо путь/словарь манифеста
    экспорта QAT. Во втором случае веса читаются из `manifest.json`,
    `ternary.trit` и `non_ternary.bin` (см. `load_export`), а `weights`
    должен быть `None`: смешивать источники нельзя.

    `weights` (все ключи опциональны): `embed` (vocab×hidden: fp32-список или
    тритовый `vtm_ref.Tensor`), `layers` (список словарей для
    `compile_layer`), `final_norm` (hidden), `lm_head` (матрица; для
    `lm_head="tmv"` — триты, для `"mm"` — плотная fp32).
    При `tie_word_embeddings=True` ключ `lm_head` игнорируется, а `MM` читает
    ту же таблицу, что `EMB` (отдельного тензора в контейнере нет).
    Без весов получается каркас: его нельзя записать в файл, но можно оценить
    по размеру.
    """
    if not isinstance(cfg, ModelConfig):
        if weights is not None:
            raise TypeError(
                "compile_model(manifest): weights нельзя передавать вместе с "
                "манифестом — веса берутся из ternary.trit/non_ternary.bin"
            )
        export = load_export(cfg)
        cfg = export.cfg
        weights = export.weights
        merged = dict(export.meta)
        merged.update(meta_extra or {})
        meta_extra = merged

    g = Graph(cfg)
    w = weights or {}
    hidden = cfg.hidden

    # --- эмбеддинг ----------------------------------------------------------
    # Таблица словаря может быть плотной (fp32/fp16/bf16) или тритовой:
    # при quantize_embeddings: trit экспорт кладёт её сегментом в
    # ternary.trit, и EMB читает триты (C — через LUT243, задача 0043).
    embed_spec = w.get("embed")
    if isinstance(embed_spec, vtm_ref.Tensor) and embed_spec.kind == vtm_ref.KIND_TRITS:
        if tuple(embed_spec.dims) != (cfg.vocab, hidden):
            raise ValueError(
                f"emb.w: форма {tuple(embed_spec.dims)}, ожидалась "
                f"{(cfg.vocab, hidden)} (тритовая таблица словаря)"
            )
        embed_spec.name = "emb.w"
        embed = g.add(embed_spec)
        embed_storage = "trit"
    else:
        embed = matrix_tensor(g, "emb.w", cfg.vocab, hidden, embed_spec, dense=True)
        embed_storage = "fp32"
    if cfg.embed_quant == "trit" and embed_storage != "trit":
        raise ValueError(
            "в architecture embed='trit', а таблица словаря плотная: "
            "экспорт должен положить model.embed_tokens.weight в ternary.trit "
            "(kind: trit), либо уберите embed=trit из манифеста"
        )
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
    tied = bool(cfg.tie_word_embeddings)
    if tied:
        # lm_head.weight — та же таблица: MM(A=emb.w, B=x.col) даёт E·x.
        # Отдельного тензора нет, поэтому размер не растёт на vocab×hidden×4.
        x_column = g.act("x.col", (hidden, 1))
        g.emit(Op.CPY, dst=x_column, a=hfinal)
        g.emit(Op.MM, dst=logits, a=embed, b=x_column)
    elif cfg.lm_head == "tmv":
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
        "vocab_size": cfg.vocab,
        "hidden_size": cfg.hidden,
        "tied_embeddings": tied,
        "tie_word_embeddings": tied,
        "lm_head": cfg.lm_head,
        # Где лежит таблица словаря: по этому ключу C/B-код и отладка
        # понимают, что EMB читает триты (kind 0), а не fp32.
        "embed": embed_storage,
        "quantize_embeddings": "trit" if embed_storage == "trit" else "none",
        "rms_norm_eps": cfg.eps,
        "context_len": cfg.context_len,
        "scale_block": vtm_ref.SCALE_BLOCK,
        "model": dataclasses.asdict(cfg),
        "sampling": "greedy",
    }
    if meta_extra:
        meta.update(meta_extra)
    return vtm_ref.VTM(tensors=g.tensors, program=g.program, meta=meta)


# ---------------------------------------------------------------------------
# Манифест экспорта QAT -> веса компилятора
# ---------------------------------------------------------------------------

MANIFEST_FORMAT = "eva32.ternary.manifest"
MANIFEST_VERSION = 2
DEFAULT_CONTEXT_LEN = 2048

_LAYER_RE = re.compile(r"(?:^|\.)(?:layers|h)\.(\d+)\.")
_MATRIX_KEYS = {
    "self_attn.q_proj": "q_proj", "attn.q_proj": "q_proj",
    "attention.q_proj": "q_proj", "q_proj": "q_proj",
    "self_attn.k_proj": "k_proj", "attn.k_proj": "k_proj",
    "attention.k_proj": "k_proj", "k_proj": "k_proj",
    "self_attn.v_proj": "v_proj", "attn.v_proj": "v_proj",
    "attention.v_proj": "v_proj", "v_proj": "v_proj",
    "self_attn.o_proj": "o_proj", "attn.o_proj": "o_proj",
    "attention.o_proj": "o_proj", "o_proj": "o_proj",
    "mlp.gate_proj": "gate_proj", "ffn.gate_proj": "gate_proj",
    "feed_forward.gate_proj": "gate_proj", "gate_proj": "gate_proj",
    "mlp.up_proj": "up_proj", "ffn.up_proj": "up_proj",
    "feed_forward.up_proj": "up_proj", "up_proj": "up_proj",
    "mlp.down_proj": "down_proj", "ffn.down_proj": "down_proj",
    "feed_forward.down_proj": "down_proj", "down_proj": "down_proj",
}
_NORM_KEYS = {
    "input_layernorm": "norm1", "ln1": "norm1", "norm1": "norm1",
    "attn_norm": "norm1",
    "post_attention_layernorm": "norm2", "ln2": "norm2", "norm2": "norm2",
    "ffn_norm": "norm2",
}
_MATRIX_ORDER = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)
_BIAS_ORDER = (
    "q_bias", "k_bias", "v_bias", "o_bias",
    "gate_bias", "up_bias", "down_bias",
)


def _human_bytes(size: int) -> str:
    value = float(size)
    for unit in ("Б", "КиБ", "МиБ", "ГиБ", "ТиБ"):
        if value < 1024.0 or unit == "ТиБ":
            return f"{value:.1f} {unit}" if unit != "Б" else f"{int(value)} Б"
        value /= 1024.0
    return f"{size} Б"


@dataclass
class Export:
    """Разобранный экспорт QAT, готовый к `compile_model`."""

    cfg: ModelConfig
    weights: dict[str, Any]
    meta: dict[str, Any]
    manifest: dict[str, Any]
    path: str
    trit_params: int = 0
    fp_params: int = 0
    tied: bool = False


def _weight_role(name: str) -> tuple[str, int | None, str]:
    """Роль имени тензора QAT/HF: (kind, layer, key).

    kind: embed | final_norm | lm_head | matrix | bias | norm | unknown.
    """
    base = name
    is_bias = name.endswith(".bias")
    if is_bias:
        base = name[:-len(".bias")]
    elif name.endswith(".weight"):
        base = name[:-len(".weight")]
    if base in ("model.embed_tokens", "embed_tokens", "embed", "emb"):
        return "embed", None, "embed"
    if base in ("model.norm", "transformer.ln_f", "final_norm", "ln_f", "norm"):
        return "final_norm", None, "final_norm"
    if base == "lm_head":
        return "lm_head", None, "lm_head"
    match = _LAYER_RE.search(name)
    if not match:
        return "unknown", None, base
    layer = int(match.group(1))
    tail = name[match.end():]
    if tail.endswith(".weight"):
        tail = tail[:-len(".weight")]
    elif tail.endswith(".bias"):
        tail = tail[:-len(".bias")]
    if tail in _NORM_KEYS:
        key = _NORM_KEYS[tail]
        return ("norm" if not is_bias else "unknown"), layer, key
    key = _MATRIX_KEYS.get(tail)
    if key is None:
        return "unknown", layer, tail
    if is_bias:
        return "bias", layer, key[:-len("_proj")] + "_bias"
    return "matrix", layer, key


def _layer_matrix_dims(cfg: ModelConfig, key: str) -> tuple[int, int]:
    hidden, q_dim, kv_dim = cfg.hidden, cfg.q_dim, cfg.kv_dim
    inter = cfg.intermediate
    return {
        "q_proj": (q_dim, hidden),
        "k_proj": (kv_dim, hidden),
        "v_proj": (kv_dim, hidden),
        "o_proj": (hidden, q_dim),
        "gate_proj": (inter, hidden),
        "up_proj": (inter, hidden),
        "down_proj": (hidden, inter),
    }[key]


def _layer_bias_dim(cfg: ModelConfig, key: str) -> int:
    return {
        "q_bias": cfg.q_dim,
        "k_bias": cfg.kv_dim,
        "v_bias": cfg.kv_dim,
        "o_bias": cfg.hidden,
        "gate_bias": cfg.intermediate,
        "up_bias": cfg.intermediate,
        "down_bias": cfg.hidden,
    }[key]


def _expect_shape(shape: Sequence[int], expected: Sequence[int], what: str) -> None:
    got = tuple(int(x) for x in shape)
    want = tuple(int(x) for x in expected)
    if got != want:
        raise ValueError(f"{what}: форма {got}, ожидалась {want}")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_file(path: Path, record: Any = None, verify: bool = True) -> bytes:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"не читается {path}: {exc}") from exc
    expected = record.get("sha256") if isinstance(record, dict) else None
    if verify and expected:
        actual = _sha256(data)
        if actual != expected:
            raise ValueError(
                f"{path.name}: sha256 {actual[:16]}… не совпал с манифестом "
                f"{str(expected)[:16]}…; файл повреждён или пересобран без манифеста"
            )
    return data


def _decode_fp_values(payload: bytes, shape: Sequence[int], dtype: str,
                      what: str) -> list[float]:
    n_values = vtm_ref.prod(shape)
    if dtype == "float32":
        item = 4
        fmt = "f"
    elif dtype == "float16":
        item = 2
        fmt = "e"
    elif dtype in ("bfloat16", "bf16"):
        item = 2
        fmt = None
    else:
        raise ValueError(f"{what}: неизвестный dtype {dtype!r}")
    expected = n_values * item
    if len(payload) != expected:
        raise ValueError(
            f"{what}: {len(payload)} байт, ожидалось {expected} "
            f"({n_values} × {item}, dtype={dtype})"
        )
    if fmt is not None:
        return list(struct.unpack(f"<{n_values}{fmt}", payload))
    words = struct.unpack(f"<{n_values}H", payload)
    return [vtm_ref._bf16_to_f32(word) for word in words]


def _non_ternary_tensor(
    entry: dict[str, Any], payload: bytes, short_name: str
) -> vtm_ref.Tensor:
    name = str(entry.get("name") or "?")
    shape = [int(x) for x in entry.get("shape") or []]
    if not shape or any(x <= 0 for x in shape):
        raise ValueError(f"{name}: плохая shape={entry.get('shape')!r}")
    dtype = str(entry.get("dtype") or "float32")
    kind = {
        "float32": vtm_ref.KIND_F32,
        "float16": vtm_ref.KIND_F16,
        "bfloat16": vtm_ref.KIND_BF16,
        "bf16": vtm_ref.KIND_BF16,
    }.get(dtype)
    if kind is None:
        raise ValueError(f"{name}: неизвестный dtype {dtype!r}")
    values = _decode_fp_values(payload, shape, dtype, name)
    return vtm_ref.Tensor(name=short_name, kind=kind, dims=shape, data=values)


def _short_weight_name(kind: str, layer: int | None, key: str) -> str:
    """Короткое имя (≤24 байт) для таблицы тензоров VTM."""
    if kind == "embed":
        return "emb.w"
    if kind == "final_norm":
        return "final_norm.w"
    if kind == "lm_head":
        return "lm_head.w"
    if layer is None:
        return key
    if kind in ("matrix", "norm"):
        return f"l{layer}.{key}.w"
    return f"l{layer}.{key}"


def _manifest_architecture(manifest: dict[str, Any]) -> ModelConfig:
    arch = manifest.get("architecture")
    if not isinstance(arch, dict):
        raise ValueError(
            "в манифесте нет architecture: переэкспортируйте чекпоинт "
            "(`python3 -m qat.export ...`, format_version 2) или добавьте "
            "раздел architecture вручную"
        )
    required = (
        "hidden", "n_heads", "n_kv_heads", "head_dim",
        "intermediate", "vocab", "n_layers",
    )
    missing = [key for key in required if arch.get(key) is None]
    if missing:
        raise ValueError(
            "в architecture нет полей: " + ", ".join(missing)
            + "; n_heads/head_dim можно задать при экспорте "
            "(--n-heads/--head-dim)"
        )
    quant = manifest.get("quantization") or {}
    block_size = int(quant.get("block_size", vtm_ref.SCALE_BLOCK))
    if block_size != vtm_ref.SCALE_BLOCK:
        raise ValueError(
            f"манифест собран с block_size={block_size}, а VTM v0.1 хранит "
            f"масштаб только на {vtm_ref.SCALE_BLOCK} тритов"
        )
    try:
        return ModelConfig(
            hidden=int(arch["hidden"]),
            n_heads=int(arch["n_heads"]),
            n_kv_heads=int(arch["n_kv_heads"]),
            head_dim=int(arch["head_dim"]),
            intermediate=int(arch["intermediate"]),
            vocab=int(arch["vocab"]),
            n_layers=int(arch["n_layers"]),
            context_len=int(arch.get("context_len") or DEFAULT_CONTEXT_LEN),
            eps=float(arch.get("rms_norm_eps", 1e-6)),
            rope_theta=float(arch.get("rope_theta", 10000.0)),
            rope_layout=str(arch.get("rope_layout", "half")),
            lm_head=str(arch.get("lm_head", "mm")),
            tie_word_embeddings=bool(arch.get("tie_word_embeddings", False)),
            embed_quant=str(
                arch.get("embed")
                or (manifest.get("embeddings") or {}).get("mode")
                or "none"
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"architecture в манифесте некорректна: {exc}") from exc


def load_export(
    source: str | os.PathLike[str] | dict[str, Any],
    verify: bool = True,
) -> Export:
    """Прочитать манифест QAT и собрать `weights` для `compile_model`.

    Источник — путь к `manifest.json` (файлы `ternary.trit` и
    `non_ternary.bin` ищутся рядом) или уже разобранный словарь манифеста
    (тогда относительные пути считаются от текущего каталога).
    """
    if isinstance(source, dict):
        manifest = dict(source)
        base = Path.cwd()
        path_label = "<dict>"
    else:
        manifest_path = Path(source).expanduser().resolve()
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ValueError(f"не читается манифест {manifest_path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"{manifest_path}: битый JSON ({exc})") from exc
        base = manifest_path.parent
        path_label = str(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError(f"{path_label}: манифест должен быть JSON-объектом")
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError(
            f"{path_label}: format={manifest.get('format')!r}, "
            f"ожидался {MANIFEST_FORMAT!r}"
        )
    version = manifest.get("format_version")
    if version != MANIFEST_VERSION:
        raise ValueError(
            f"{path_label}: format_version={version!r}; компилятор понимает "
            f"{MANIFEST_VERSION}. Манифест v1 не содержит нетернарных весов "
            "(эмбеддинги/норма/lm_head) — переэкспортируйте чекпоинт"
        )

    cfg = _manifest_architecture(manifest)
    files = manifest.get("files") or {}

    # --- тритовые сегменты --------------------------------------------------
    trit_name = "ternary.trit"
    trit_data = _read_file(base / trit_name, files.get(trit_name), verify)

    weights: dict[str, Any] = {"layers": [{} for _ in range(cfg.n_layers)]}
    trit_params = 0
    fp_params = 0
    unknown: list[str] = []

    def _layer_bucket(layer: int, what: str) -> dict[str, Any]:
        if not 0 <= layer < cfg.n_layers:
            raise ValueError(f"{what}: слой {layer} вне n_layers={cfg.n_layers}")
        return weights["layers"][layer]

    for entry in manifest.get("tensors") or []:
        name = str(entry.get("name") or "?")
        shape = [int(x) for x in entry.get("shape") or []]
        offset = int(entry.get("trit_offset", entry.get("offset", -1)))
        length = int(entry.get("trit_length", entry.get("length", -1)))
        if offset < 0 or length <= 0 or offset + length > len(trit_data):
            raise ValueError(
                f"{name}: сегмент [{offset}, {offset + length}) вне "
                f"ternary.trit ({len(trit_data)} байт)"
            )
        kind, layer, key = _weight_role(name)
        if kind == "embed":
            # Тритовая таблица словаря (задача 0043): экспорт кладёт её
            # сегментом ternary.trit при quantize_embeddings: trit.
            if cfg.embed_quant != "trit":
                raise ValueError(
                    f"{name}: тритовая таблица словаря, а в architecture "
                    f"embed={cfg.embed_quant!r}; переэкспортируйте чекпоинт "
                    "с quantize_embeddings: trit или оставьте эмбеддинги fp32"
                )
            _expect_shape(shape, (cfg.vocab, cfg.hidden), name)
            trits, scales, scale_dtype = vtm_ref.decode_trit_segment(
                trit_data[offset:offset + length]
            )
            weights["embed"] = vtm_ref.Tensor(
                name="emb.w",
                kind=vtm_ref.KIND_TRITS,
                dims=shape,
                trits=array.array("b", trits),
                scales=scales,
                scale_dtype=scale_dtype,
            )
        elif kind in ("bias", "norm", "final_norm"):
            raise ValueError(
                f"{name}: триты допустимы только для матриц Linear, lm_head "
                f"и таблицы словаря; нормы/bias должны лежать в "
                f"non_ternary.bin"
            )
        elif kind == "lm_head":
            if cfg.tie_word_embeddings:
                continue  # tied: отдельная матрица не нужна
            if cfg.lm_head != "tmv":
                raise ValueError(
                    f"{name}: в architecture lm_head={cfg.lm_head!r}, "
                    "а в экспорте тритовая матрица; переэкспортируйте с "
                    "lm_head='tmv' или оставьте lm_head в fp32"
                )
            _expect_shape(shape, (cfg.vocab, cfg.hidden), name)
            trits, scales, scale_dtype = vtm_ref.decode_trit_segment(
                trit_data[offset:offset + length]
            )
            weights["lm_head"] = vtm_ref.Tensor(
                name="lm_head.w",
                kind=vtm_ref.KIND_TRITS,
                dims=shape,
                trits=array.array("b", trits),
                scales=scales,
                scale_dtype=scale_dtype,
            )
        elif kind == "matrix":
            expected = _layer_matrix_dims(cfg, key)
            _expect_shape(shape, expected, name)
            decoded = vtm_ref.decode_trit_segment(trit_data[offset:offset + length])
            tensor = vtm_ref.Tensor(
                name=f"l{layer}.{key}.w",
                kind=vtm_ref.KIND_TRITS,
                dims=shape,
                trits=array.array("b", decoded[0]),
                scales=decoded[1],
                scale_dtype=decoded[2],
            )
            _layer_bucket(layer, name)[key] = tensor
        elif kind == "unknown":
            unknown.append(name)
            continue
        else:  # pragma: no cover - роли выше перечислены
            raise ValueError(f"{name}: неожиданная роль {kind!r}")
        if len(shape) != 2 or not all(shape):
            raise ValueError(f"{name}: у матрицы плохая shape={shape}")
        trit_params += vtm_ref.prod(shape)

    # --- нетернарные веса ---------------------------------------------------
    section = manifest.get("non_ternary") or {}
    if not section.get("stored_in_this_export"):
        raise ValueError(
            f"{path_label}: нетернарные веса не экспортированы "
            "(non_ternary.stored_in_this_export != true): эмбеддингов и "
            "финальной нормы нет, собрать полный граф не из чего. "
            "Переэкспортируйте чекпоинт (format_version 2)"
        )
    non_file = str(section.get("file") or "non_ternary.bin")
    non_data = _read_file(base / non_file, files.get(non_file), verify)
    for entry in section.get("tensors") or []:
        name = str(entry.get("name") or "?")
        kind, layer, key = _weight_role(name)
        if kind == "unknown":
            unknown.append(name)
            continue
        if entry.get("tied_to") or entry.get("alias_of"):
            # Ссылка на уже выгруженный тензор: отдельных байт у неё нет.
            continue
        offset = int(entry.get("offset", -1))
        length = int(entry.get("length", -1))
        if offset < 0 or length < 0 or offset + length > len(non_data):
            raise ValueError(
                f"{name}: диапазон [{offset}, {offset + length}) вне {non_file} "
                f"({len(non_data)} байт)"
            )
        shape = [int(x) for x in entry.get("shape") or []]
        payload = non_data[offset:offset + length]
        tensor = _non_ternary_tensor(
            entry, payload, _short_weight_name(kind, layer, key)
        )
        if kind == "embed":
            if cfg.embed_quant == "trit":
                raise ValueError(
                    f"{name}: в architecture embed='trit', а в "
                    f"{non_file} лежит плотная таблица; манифест противоречив"
                )
            _expect_shape(shape, (cfg.vocab, cfg.hidden), name)
            if cfg.tie_word_embeddings and tensor.kind != vtm_ref.KIND_F32:
                raise ValueError(
                    f"{name}: tied lm_head считается через MM по этой же "
                    f"таблице; нужен fp32 или тритовая таблица (embed=trit), "
                    f"а пришёл {tensor.kind_name}"
                )
            weights["embed"] = tensor
        elif kind == "final_norm":
            _expect_shape(shape, (cfg.hidden,), name)
            weights["final_norm"] = tensor
        elif kind == "lm_head":
            if cfg.tie_word_embeddings:
                continue
            if cfg.lm_head == "tmv":
                raise ValueError(
                    f"{name}: в architecture lm_head='tmv', а тензор плотный"
                )
            _expect_shape(shape, (cfg.vocab, cfg.hidden), name)
            weights["lm_head"] = tensor
        elif kind in ("matrix", "norm", "bias"):
            bucket = _layer_bucket(layer, name)
            if kind == "matrix":
                _expect_shape(shape, _layer_matrix_dims(cfg, key), name)
                bucket[key] = tensor
            elif kind == "norm":
                _expect_shape(shape, (cfg.hidden,), name)
                bucket[key] = tensor
            else:
                _expect_shape(shape, (_layer_bias_dim(cfg, key),), name)
                bucket[key] = tensor
        else:  # pragma: no cover
            raise ValueError(f"{name}: неожиданная роль {kind!r}")
        fp_params += tensor.n_elements

    if unknown:
        preview = ", ".join(sorted(set(unknown))[:8])
        raise ValueError(
            "компилятор не знает, куда положить тензоры: " + preview
            + (" …" if len(set(unknown)) > 8 else "")
            + ". Поддерживаются Qwen2/Llama-имена (model.layers.N...)"
        )

    # --- обязательные части графа ------------------------------------------
    if "embed" not in weights:
        if cfg.embed_quant == "trit":
            raise ValueError(
                "в экспорте нет тритовой таблицы словаря: architecture "
                "embed='trit', но сегмента model.embed_tokens.weight нет в "
                "ternary.trit и записи с kind: trit в манифесте тоже нет"
            )
        raise ValueError(
            "в экспорте нет эмбеддингов (model.embed_tokens.weight): "
            "полный граф без них не собрать"
        )
    if "final_norm" not in weights:
        raise ValueError(
            "в экспорте нет финальной нормы (model.norm.weight): "
            "полный граф без неё не собрать"
        )
    if not cfg.tie_word_embeddings and "lm_head" not in weights:
        raise ValueError(
            "lm_head.weight нет в экспорте, а tie_word_embeddings=false; "
            "экспортируйте lm_head в non_ternary.bin"
        )
    for index, layer in enumerate(weights["layers"]):
        missing = [key for key in _MATRIX_ORDER if key not in layer]
        if missing:
            raise ValueError(
                f"слой {index}: нет матриц {', '.join(missing)}; "
                "манифест неполный или имена слоёв другие"
            )
        for key in ("norm1", "norm2"):
            if key not in layer:
                raise ValueError(f"слой {index}: нет {key}")

    training = manifest.get("training") or {}
    meta: dict[str, Any] = {
        "source_manifest": path_label,
        "base_model": manifest.get("base_model"),
        "revision": (manifest.get("base_model") or {}).get("revision"),
        "step": training.get("step"),
        "checkpoint": training.get("checkpoint"),
        "manifest_format": manifest.get("format"),
        "manifest_version": version,
        "quantization": manifest.get("quantization"),
    }
    return Export(
        cfg=cfg,
        weights=weights,
        meta=meta,
        manifest=manifest,
        path=path_label,
        trit_params=trit_params,
        fp_params=fp_params,
        tied=cfg.tie_word_embeddings,
    )


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
#: context_len — не max_position_embeddings модели, а длина KV-кэша, которую
#: компилятор кладёт в контейнер: кэш занимает ctx × layers × kv_dim × 8 байт
#: (K и V, fp32). 32768 превратили бы контейнер в ~3 ГиБ одних нулей, поэтому
#: по умолчанию берём рабочую длину VTM, а не максимум Qwen.
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
        "eps": 1e-6,
        "tie_word_embeddings": True,
        "context_len": 2048,
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
        "eps": 1e-6,
        "tie_word_embeddings": True,
        "context_len": 2048,
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


def _print_manifest(
    export: Export, vtm: vtm_ref.VTM, path: str | None
) -> None:
    cfg = export.cfg
    ops = vtm.op_counts()
    base = export.manifest.get("base_model") or {}
    print(f"манифест: {export.path}")
    print(
        "источник: "
        f"{base.get('name_or_path') or '?'} @ {base.get('revision') or '?'}"
        f", шаг {(export.manifest.get('training') or {}).get('step', '?')}"
    )
    print(
        f"архитектура: hidden={cfg.hidden} heads={cfg.n_heads}/{cfg.n_kv_heads} "
        f"head_dim={cfg.head_dim} intermediate={cfg.intermediate} "
        f"vocab={cfg.vocab} layers={cfg.n_layers} context={cfg.context_len} "
        f"rope_theta={cfg.rope_theta:g} eps={cfg.eps:g}"
    )
    if export.tied:
        saved = cfg.vocab * cfg.hidden * 4
        print(
            "tied lm_head: да — отдельного тензора нет, MM читает emb.w "
            f"(экономия {_human_bytes(saved)})"
        )
    else:
        print(f"lm_head: отдельная матрица ({cfg.lm_head})")
    embed = export.weights.get("embed")
    if embed is not None:
        payload = (
            vtm_ref.tensor_payload_size(embed)
            if embed.trits is not None or embed.data is not None
            else 0
        )
        fp32_bytes = cfg.vocab * cfg.hidden * 4
        if embed.kind == vtm_ref.KIND_TRITS:
            print(
                f"эмбеддинги: trit, сегмент ternary.trit "
                f"({_human_bytes(payload)} против {_human_bytes(fp32_bytes)} fp32, "
                f"EMB читает триты через LUT243"
                + ("; tied lm_head — MM по той же таблице)" if export.tied else ")")
            )
        else:
            print(
                f"эмбеддинги: {embed.kind_name} в non_ternary.bin "
                f"({_human_bytes(fp32_bytes)})"
            )
    print(f"тензоров: {len(vtm.tensors)}, инструкций: {len(vtm.program)}")
    print("операции: " + ", ".join(f"{name}×{count}" for name, count in sorted(ops.items())))
    cache_bytes = cfg.n_layers * cfg.context_len * cfg.kv_dim * 2 * 4
    print(
        f"параметры: тритов {export.trit_params:,}, fp {export.fp_params:,}; "
        f"KV-кэш {_human_bytes(cache_bytes)}"
    )
    if path:
        print(f"файл: {path} ({_human_bytes(os.path.getsize(path))})")
        print(
            "проверка C: vtm run <файл> --tokens ... --max-new 1; "
            "логиты сверять с meta.expected больше нечем — контейнер "
            "собран из чекпоинта, эталона HF в нём нет"
        )
    else:
        print("файл не записан (--out не задан)")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Компилятор графа VTM (Python-эталон, без torch/numpy)"
    )
    parser.add_argument("--synthetic", action="store_true", help="собрать синтетическую модель")
    parser.add_argument("--seed", type=int, default=1, help="seed синтетики (по умолчанию 1)")
    parser.add_argument("--size", type=int, default=128, help="hidden синтетики (128)")
    parser.add_argument("--mode", choices=("mlp", "attn"), default="attn",
                        help="mlp — без внимания, attn — полный слой (по умолчанию)")
    parser.add_argument("--out", help="куда записать .vtm (--synthetic/--manifest)")
    parser.add_argument("--qwen-config", help="YAML Qwen-конфига (только план, без весов)")
    parser.add_argument(
        "--manifest",
        help="manifest.json экспорта QAT (нужны ternary.trit и non_ternary.bin рядом)",
    )
    args = parser.parse_args(argv)

    modes = [bool(args.synthetic), bool(args.qwen_config), bool(args.manifest)]
    if sum(modes) != 1:
        parser.error("нужен ровно один режим: --synthetic, --qwen-config или --manifest")
    if args.manifest and not args.out:
        parser.error("--manifest требует --out: контейнер нужно куда-то записать")

    try:
        if args.qwen_config:
            name, cfg, plan = qwen_plan(args.qwen_config)
            _print_plan(name, cfg, plan)
            if args.out:
                print(f"--out {args.out} проигнорирован: контейнер без весов не пишется.")
            return 0

        if args.manifest:
            export = load_export(args.manifest)
            vtm = compile_model(export.cfg, export.weights, export.meta)
            out_dir = os.path.dirname(os.path.abspath(args.out))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            vtm_ref.write_vtm(args.out, vtm.tensors, vtm.program, vtm.meta)
            _print_manifest(export, vtm, args.out)
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
