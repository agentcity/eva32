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
* KV-кэши помечены `VTM_TENSOR_FLAG_KV_CACHE` (бит 0 поля по смещению 50 в
  таблице тензоров) и в контейнер не пишутся (задача 0101): в файле остаётся
  только запись таблицы, а буфер `ctx × kv_dim` создаёт загрузчик (C — в
  арене, Python — списком нулей). Для 1.5B-раскладки это −112 МиБ файла;
  старые контейнеры, где кэш лежал внутри, читаются как раньше. Компилятор
  ставит и флаг заголовка `VFLAG_KV_OUT_OF_FILE` (0x0004, задача 0118) —
  через `vtm_ref.write_vtm`, который видит KV-кэши в графе; offset кэшей
  тогда `KV_NO_OFFSET` (0xFFFFFFFF), чтобы загрузчик до 0101 отказал, а не
  прочитал мусор. Суммарный размер кэшей ограничен `MAX_KV_BYTES` (512 МиБ);
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
python3 tools/vtm_compile.py --synthetic --shape 7b --out /tmp/synth_7b.vtm
python3 tools/vtm_compile.py --synthetic --shape 32b --out /tmp/synth_32b.vtm
python3 tools/vtm_compile.py --manifest export/manifest.json --out model.vtm
python3 tools/vtm_compile.py --qwen-config qat/configs/qwen1.5b.yaml
```

Большие формы и KV-бюджет (задача 0135)
========================================

KV-кэш (`2 × слоёв × ctx × kv_dim × 4` байт) живёт в арене, а не в файле, но
загрузчик ограничивает его сумму `VTM_MAX_KV_BYTES` = 512 МиБ (в C задаётся
`-D`). Компилятор считает то же число и **сам урезает `context_len`**
(2048 → 1024 → 512), печатая в stderr предупреждение с числами; решение
пишется в `meta["kv_cache"]`, а фактическое окно — в `meta["context_window"]`.
Полный контекст возвращается пересборкой рантайма с
`-DVTM_MAX_KV_BYTES=...` (команда печатается). Если не влезает даже 512 —
`ValueError` с этой командой: молча не отказываем.

`--synthetic --shape 1.5b|7b|14b|32b` собирает настоящий граф формы (те же
тензоры/инструкции/KV-кэши, что у обученной модели) с нулевыми весами:
`vtm_ref.write_vtm_stream` пишет payload нулями, поэтому 32B не требует ни HF,
ни миллиардов тритов в памяти. Файл пишется для 1.5B (~0,34 ГиБ), 7B (~1,5 ГиБ)
и 14B (~3,0 ГиБ); 32B-раскладка — ~6,9 ГиБ, а формат v0.2 адресует файл
u32-смещениями (предел 4 ГиБ), поэтому `--out` для 32B даёт внятную ошибку, а
`--shape 32b` без `--out` печатает план и KV-бюджет. Проверка C-рантаймом:

```
src/vtm/vtm run /tmp/synth_14b.vtm --tokens 1 --max-new 1
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
по-прежнему считается `MM` по `emb.w`, только A уже тритовая.

Если `architecture.embed == "int8"` (задача 0044), таблица словаря — int8-сегмент
(kind 4): `dtype: "int8"` в записи `non_ternary.bin`, payload — ровно
`INT8`-сегмент (`"INT8"`, version, flags, n_values, n_scales, масштабы, q-байты;
см. `vtm_ref.decode_int8_segment`). `EMB` в Python читает `q · scale`; C с 0170
держит таблицу упакованной: `EMB` разворачивает одну строку, `MM` (tied lm_head)
считает GEMV по int8-таблице (`vtm_i8_*`), в арене — только fp32-масштабы.
`VTM_I8_EMB=0` возвращает fp32-развёртку, если нужен прежний путь.

Если в манифесте `architecture.lm_head == "q4"` (импорт GGUF, задача 0153),
матрицы Linear и отдельный lm_head лежат Q4-сегментами kind 5 в `q4.bin`
(`dtype: "q4_0"`, `q4_offset`/`q4_length`). `load_export` разбирает сегмент
`vtm_ref.decode_q4_segment_ex` и отдаёт `Tensor.q4(...)`; `matrix_tensor`
принимает kind 5, а `TMV` считает упакованные нибблы без fp32-копии
(ядра 0152). `lm_head='q4'` — это `TMV` по kind 5; при
`tie_word_embeddings: true` отдельной Q4-матрицы нет и работает прежний
`MM` по таблице словаря. Транспонированный Q4 (`VTM_FLAG_TMV_TRANSPOSED`)
ядра не поддерживают: компилятор отвергает такой граф с внятной ошибкой, а
не пишет контейнер, который C посчитает неверно.

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
import mmap
import os
import random
import re
import struct
import sys
import time
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

#: Флаг TMV «матрица хранится как [cols, rows]» (VTM_FLAG_TMV_TRANSPOSED,
#: `vtm.h`). Q4-ядра 0152 такую ориентацию не поддерживают: загрузчик C
#: отказывает `VTM_E_UNSUPPORTED`, а компилятор обязан отказать раньше —
#: иначе в контейнер попадёт граф, который невозможно исполнить.
TMV_FLAG_TRANSPOSED = 0x04


def q4_transposed_error(name: str) -> str:
    """Текст отказа «транспонированный Q4/Q4_K не поддержан» (одно место)."""
    return (
        f"вес {name!r}: Q4/Q4_K/Q6_K с VTM_FLAG_TMV_TRANSPOSED не поддержан — "
        "ядра 0152/0165 (vtm_q4_*, vtm_q4k_*, vtm_q6k_*) читают матрицу "
        "только как [rows, cols] и считают W·x. Транспонированный Q4 — "
        "отдельная задача; переэкспортируйте вес в обычной ориентации"
    )


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
    #: тернаризуется), "tmv" — тритовая матрица, "q4" — Q4-сегмент kind 5
    #: (импорт GGUF, задача 0153) через TMV.
    lm_head: str = "mm"
    #: Qwen/Llama-стиль: lm_head.weight — это та же таблица, что
    #: embed_tokens.weight. В контейнере отдельного тензора нет, MM читает
    #: `emb.w`; экономия vocab × hidden × 4 байт.
    tie_word_embeddings: bool = False
    #: Где лежит таблица словаря: "none" — fp32 в non_ternary.bin,
    #: "trit" — сегмент ternary.trit (kind: trit, задача 0043),
    #: "int8" — INT8-сегмент в non_ternary.bin (kind 4, задача 0044).
    embed_quant: str = "none"
    #: MoE (задача 0172): вместо плотного FFN — роутер, top-k и разреженные
    #: эксперты. `intermediate` при этом не используется (у Qwen3-MoE там
    #: dense-размер, которого в графе нет), размер эксперта — `expert_inter`.
    moe: bool = False
    n_experts: int = 0
    top_k: int = 0
    expert_inter: int = 0
    #: QK-norm (Qwen3): RMSH по head_dim на Q и K до RoPE.
    qk_norm: bool = False
    #: Общий эксперт (Qwen2-MoE: shared_expert): плотный FFN, прибавляется к
    #: residual рядом с суммой экспертов. 0 — общего эксперта нет (Qwen3).
    shared_inter: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.moe:
            if self.n_experts <= 0:
                raise ValueError("moe=true, а n_experts <= 0")
            if not 0 < self.top_k <= self.n_experts:
                raise ValueError(
                    f"top_k={self.top_k} вне 1..n_experts={self.n_experts}"
                )
            if self.expert_inter <= 0:
                raise ValueError("moe=true, а expert_inter <= 0")
            if self.shared_inter < 0:
                raise ValueError("shared_inter не может быть отрицательным")
        self.embed_quant = str(self.embed_quant or "none").lower()
        if self.embed_quant not in ("none", "trit", "int8"):
            raise ValueError(
                f"embed_quant={self.embed_quant!r}, допустимо none|trit|int8"
            )
        if self.vocab <= 0:
            raise ValueError("vocab должен быть > 0")
        if self.n_layers <= 0:
            raise ValueError("n_layers должен быть > 0")
        if self.lm_head not in ("tmv", "mm", "q4"):
            raise ValueError(
                f"lm_head={self.lm_head!r}, допустимо tmv|mm|q4"
            )
        if self.tie_word_embeddings and self.lm_head == "tmv":
            raise ValueError(
                "tie_word_embeddings вместе с lm_head='tmv' невозможен: "
                "tied-таблица хранится как fp32-эмбеддинг, а TMV ждёт триты; "
                "тритовые эмбеддинги идут через EMB + MM (embed_quant='trit')"
            )
        # lm_head='q4' при tied — не противоречие: отдельной матрицы нет,
        # MM читает таблицу словаря (так пишет манифест импорт GGUF 0153).



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

    def kv_cache(self, name: str, dims: Sequence[int]) -> int:
        """K/V-кэш внимания `(ctx, kv_dim)` — без содержимого в контейнере.

        В файл попадает только запись таблицы (имя/kind/dims): кэш пуст на
        старте, его заполняет KVW. Для 1.5B-раскладки это экономит 112 МиБ
        (28 слоёв × ctx 2048 × kv_dim 256 × 2 × 4). Читатель (C и Python)
        создаёт буфер по dims, поэтому исполнение не меняется.
        """
        return self.add(vtm_ref.Tensor.kv_cache(name, dims))

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
    transposed: bool = False,
) -> int:
    """Добавить матрицу весов: тритовую/Q4 (TMV) или плотную fp32 (MM/каркас).

    `spec` может быть готовым `vtm_ref.Tensor` из манифеста: тогда он
    добавляется в граф без копирования значений (важно для эмбеддингов
    1.5B-модели — это сотни мегабайт). Имя тензора приводится к короткому
    имени VTM (в таблице контейнера максимум 24 байта).

    Для TMV (`dense=False`) принимаются триты (kind 0) и Q4 (kind 5, задача
    0154). `transposed=True` означает, что граф хочет Wᵀ·x: для Q4 это
    не поддержано ядрами — сразу `ValueError` (внятный отказ вместо
    контейнера, который C не исполнит).
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
        if dense and spec.kind in (
            vtm_ref.KIND_TRITS, vtm_ref.KIND_Q4,
            vtm_ref.KIND_Q4K, vtm_ref.KIND_Q6K,
        ):
            raise ValueError(
                f"вес {name!r}: нужен плотный fp-тензор, а пришли "
                f"{spec.kind_name}"
            )
        if not dense and spec.kind not in (
            vtm_ref.KIND_TRITS, vtm_ref.KIND_Q4,
            vtm_ref.KIND_Q4K, vtm_ref.KIND_Q6K,
        ):
            raise ValueError(
                f"вес {name!r}: нужны триты или Q4, а пришёл {spec.kind_name}"
            )
        if transposed and spec.kind in (
            vtm_ref.KIND_Q4, vtm_ref.KIND_Q4K, vtm_ref.KIND_Q6K
        ):
            raise ValueError(q4_transposed_error(name))
        spec.name = name
        return graph.add(spec)
    if transposed:
        # Спецификация без готового Tensor: Q4 тут неоткуда взять (триты
        # квантуются локально), поэтому флаг пока не с чем сочетать. Не
        # игнорируем его молча — иначе будущий Q4-путь снова станет тихим.
        raise ValueError(
            f"вес {name!r}: transposed=True поддержан только для готового "
            "Tensor (Q4 транспонировать нельзя); для плотного/тритового веса "
            "соберите матрицу в нужной ориентации"
        )
    if dense:
        return graph.add(vtm_ref.Tensor.f32(name, (rows, cols), _spec_floats(spec, n_values)))
    trits, scales = _spec_trits(spec, n_values)
    return graph.add(
        vtm_ref.Tensor.ternary(
            name, (rows, cols), trits, scales, graph.cfg.scale_dtype
        )
    )


def check_q4_tmv_support(
    tensors: Sequence[vtm_ref.Tensor], program: Sequence[vtm_ref.Instr]
) -> None:
    """Отказать, если граф просит транспонированный Q4 в TMV.

    Ядра kind 5 (`vtm_q4_tmv`/`vtm_q4_mm`, задача 0152) читают матрицу
    только как `[rows, cols]`. Компилятор сам `VTM_FLAG_TMV_TRANSPOSED` не
    ставит, но проверка ловит граф, собранный извне (или будущий путь), до
    записи контейнера: C иначе вернёт `VTM_E_UNSUPPORTED` уже на исполнении.
    """
    for index, instr in enumerate(program):
        if int(instr.op) != Op.TMV or not (int(instr.flags) & TMV_FLAG_TRANSPOSED):
            continue
        matrix = tensors[int(instr.a)]
        if matrix.kind in (
            vtm_ref.KIND_Q4, vtm_ref.KIND_Q4K, vtm_ref.KIND_Q6K
        ):
            raise ValueError(
                q4_transposed_error(matrix.name) + f" (инструкция TMV #{index})"
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


def expert_bank(
    graph: Graph, name: str, dims: Sequence[int], spec: Any
) -> int:
    """Добавить банк экспертов MoE: kind 6/7, dims [n_experts, rows, cols].

    Банк — это все эксперты одного веса (gate/up/down) в одном тензоре:
    блоки GGML лежат подряд, эксперт `e` занимает `rows·cols/256` блоков с
    начала. Так ядро читает только выбранные эксперты, а не всю матрицу, и
    в таблице тензоров остаётся одна запись на вес вместо 128.

    `spec` — готовый `vtm_ref.Tensor` (окно в mmap q4.bin из импорта GGUF).
    Три инструкции MOE требуют, чтобы gate, up и down шли в таблице подряд
    (c, c+1, c+2) — компилятор добавляет их именно так, без вставок между
    вызовами (см. compile_layer и vtm.h).
    """
    dims = tuple(int(d) for d in dims)
    if spec is None:
        return graph.add(vtm_ref.Tensor.placeholder(name, dims, vtm_ref.KIND_Q4K))
    if not isinstance(spec, vtm_ref.Tensor):
        raise ValueError(
            f"банк {name!r}: ожидался готовый Tensor из импорта, а пришло "
            f"{type(spec).__name__}"
        )
    if spec.kind not in (vtm_ref.KIND_Q4K, vtm_ref.KIND_Q6K, vtm_ref.KIND_Q4):
        raise ValueError(
            f"банк {name!r}: нужен kind 6/7 (Q4_K/Q6_K), а пришёл "
            f"{spec.kind_name}"
        )
    if tuple(spec.dims) != dims:
        raise ValueError(
            f"банк {name!r}: форма {tuple(spec.dims)}, ожидалась {dims}"
        )
    spec.name = name
    return graph.add(spec)


def _expert_bank_dims(cfg: ModelConfig, key: str) -> tuple[int, ...]:
    """Форма банка по роли: gate/up — [n_exp, inter, hidden], down — наоборот."""
    if key in ("expert_gate", "expert_up"):
        return (cfg.n_experts, cfg.expert_inter, cfg.hidden)
    if key == "expert_down":
        return (cfg.n_experts, cfg.hidden, cfg.expert_inter)
    raise ValueError(f"не экспертный ключ: {key!r}")


def _shared_dims(cfg: ModelConfig, key: str) -> tuple[int, int]:
    inter = cfg.shared_inter
    return {
        "shared_gate": (inter, cfg.hidden),
        "shared_up": (inter, cfg.hidden),
        "shared_down": (cfg.hidden, inter),
    }[key]


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

    # QK-norm (Qwen3): RMSN по head_dim на каждую голову, ДО RoPE. RMSN
    # нормирует весь тензор, поэтому здесь отдельный опкод RMSH — иначе 32
    # головы получили бы один общий масштаб.
    if cfg.qk_norm:
        q_norm = vector_tensor(g, f"{prefix}.q_norm.w", (cfg.head_dim,), w.get("q_norm"))
        k_norm = vector_tensor(g, f"{prefix}.k_norm.w", (cfg.head_dim,), w.get("k_norm"))
        g.emit(Op.RMSH, dst=q, a=q, b=q_norm, imm=cfg.eps_imm)
        g.emit(Op.RMSH, dst=k, a=k, b=k_norm, imm=cfg.eps_imm)

    rope_flags = (
        vtm_ref.IFLAG_ROPE_INTERLEAVED if cfg.rope_layout == "interleaved" else 0
    )
    g.emit(Op.ROPE, dst=q, a=q, b=cfg.n_heads, flags=rope_flags, imm=0)
    g.emit(Op.ROPE, dst=k, a=k, b=cfg.n_kv_heads, flags=rope_flags, imm=0)

    k_cache = g.kv_cache(f"{prefix}.kcache", (ctx, cfg.kv_dim))
    v_cache = g.kv_cache(f"{prefix}.vcache", (ctx, cfg.kv_dim))
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

    if cfg.moe:
        # MoE (задача 0172): роутер -> top-k -> взвешенная сумма экспертов.
        # Всё одной инструкцией MOE: индекс эксперта зависит от данных, а
        # граф из TMV его не выражает (TMV берёт матрицу из инструкции).
        # gate/up/down добавляются строго подряд — это часть ABI MOE.
        router = matrix_tensor(
            g, f"{prefix}.router.w", cfg.n_experts, hidden, w.get("router"),
            dense=True,
        )
        gate_bank = expert_bank(
            g, f"{prefix}.gate_bank",
            _expert_bank_dims(cfg, "expert_gate"), w.get("expert_gate"),
        )
        up_bank = expert_bank(
            g, f"{prefix}.up_bank",
            _expert_bank_dims(cfg, "expert_up"), w.get("expert_up"),
        )
        down_bank = expert_bank(
            g, f"{prefix}.down_bank",
            _expert_bank_dims(cfg, "expert_down"), w.get("expert_down"),
        )
        if (up_bank != gate_bank + 1 or down_bank != gate_bank + 2):
            raise ValueError(
                "MOE: банки gate/up/down обязаны идти в таблице тензоров "
                "подряд (c, c+1, c+2) — между ними добавлен другой тензор"
            )
        g.emit(Op.MOE, dst=x, a=router, b=h2, c=gate_bank, imm=cfg.top_k)

        # Общий эксперт (Qwen2-MoE): обычный плотный FFN, прибавляется к
        # residual рядом с суммой экспертов. У Qwen3-30B-A3B его нет
        # (shared_inter=0), поддержка — ради 35B/будущих моделей.
        if cfg.shared_inter:
            sinter = cfg.shared_inter
            sgate = g.act(f"{prefix}.sgate", (sinter,))
            sup = g.act(f"{prefix}.sup", (sinter,))
            sgate_w = matrix_tensor(
                g, f"{prefix}.shared_gate.w", sinter, hidden, w.get("shared_gate")
            )
            sup_w = matrix_tensor(
                g, f"{prefix}.shared_up.w", sinter, hidden, w.get("shared_up")
            )
            g.emit(Op.TMV, dst=sgate, a=sgate_w, b=h2)
            g.emit(Op.TMV, dst=sup, a=sup_w, b=h2)
            g.emit(Op.SILU, dst=sgate, a=sgate)
            smul = g.act(f"{prefix}.smul", (sinter,))
            g.emit(Op.MUL, dst=smul, a=sgate, b=sup)
            sdown = g.act(f"{prefix}.sdown", (hidden,))
            sdown_w = matrix_tensor(
                g, f"{prefix}.shared_down.w", hidden, sinter, w.get("shared_down")
            )
            g.emit(Op.TMV, dst=sdown, a=sdown_w, b=smul)
            g.emit(Op.ADD, dst=x, a=x, b=sdown)
        return g, x

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
    *,
    max_kv_bytes: int | None = None,
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

    `max_kv_bytes` — кэп KV-кэшей (по умолчанию `vtm_ref.MAX_KV_BYTES`, то же,
    что `VTM_MAX_KV_BYTES` в C). Если кэш при `context_len` не влезает,
    компилятор сам урезает контекст (2048 → 1024 → 512), печатает
    предупреждение с числами в stderr и пишет решение в `meta["kv_cache"]`;
    если не влезает даже 512 — `ValueError` с командой пересборки `-D`.
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

    cfg, kv_info = fit_context_window(cfg, max_bytes=max_kv_bytes)
    if kv_info["reduced"]:
        print(kv_warning_message(kv_info), file=sys.stderr)

    g = Graph(cfg)
    w = weights or {}
    hidden = cfg.hidden

    # --- эмбеддинг ----------------------------------------------------------
    # Таблица словаря может быть плотной (fp32/fp16/bf16) или тритовой:
    # при quantize_embeddings: trit экспорт кладёт её сегментом в
    # ternary.trit, и EMB читает триты (C — через LUT243, задача 0043).
    embed_spec = w.get("embed")
    if isinstance(embed_spec, vtm_ref.Tensor) and embed_spec.kind in (
        vtm_ref.KIND_TRITS,
        vtm_ref.KIND_INT8,
        vtm_ref.KIND_Q4K,
        vtm_ref.KIND_Q6K,
    ):
        if tuple(embed_spec.dims) != (cfg.vocab, hidden):
            raise ValueError(
                f"emb.w: форма {tuple(embed_spec.dims)}, ожидалась "
                f"{(cfg.vocab, hidden)} (квантованная таблица словаря)"
            )
        embed_spec.name = "emb.w"
        embed = g.add(embed_spec)
        embed_storage = {
            vtm_ref.KIND_TRITS: "trit",
            vtm_ref.KIND_INT8: "int8",
            vtm_ref.KIND_Q4K: "q4_k",
            vtm_ref.KIND_Q6K: "q6_k",
        }[embed_spec.kind]
    else:
        embed = matrix_tensor(g, "emb.w", cfg.vocab, hidden, embed_spec, dense=True)
        embed_storage = "fp32"
    if cfg.embed_quant == "trit" and embed_storage != "trit":
        raise ValueError(
            "в architecture embed='trit', а таблица словаря плотная: "
            "экспорт должен положить model.embed_tokens.weight в ternary.trit "
            "(kind: trit), либо уберите embed=trit из манифеста"
        )
    if cfg.embed_quant == "int8" and embed_storage != "int8":
        raise ValueError(
            "в architecture embed='int8', а таблица словаря не int8: "
            "экспорт должен положить model.embed_tokens.weight сегментом "
            "INT8 (dtype: int8) в non_ternary.bin, либо уберите embed=int8 "
            "из манифеста"
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
    elif cfg.lm_head == "q4":
        # Q4-матрица (kind 5) из q4.bin импорта GGUF: TMV считает упакованные
        # нибблы, fp32-копии нет. Ориентация — обычная [vocab, hidden].
        lm_head = matrix_tensor(g, "lm_head.w", cfg.vocab, hidden, w.get("lm_head"))
        if g.tensors[lm_head].kind not in (
            vtm_ref.KIND_Q4, vtm_ref.KIND_Q4K, vtm_ref.KIND_Q6K
        ):
            raise ValueError(
                f"lm_head='q4', а матрица {g.tensors[lm_head].kind_name}; "
                "импорт GGUF --format q4 кладёт lm_head Q4-сегментом, "
                "для тритов/плотной матрицы поставьте lm_head='tmv'/'mm'"
            )
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
        # MoE (0172): по этим ключам tools/moe_program.py и отладка понимают,
        # что FFN — разреженный, и сколько экспертов считается на токен.
        "moe": bool(cfg.moe),
        "n_experts": int(cfg.n_experts),
        "top_k": int(cfg.top_k),
        "expert_intermediate": int(cfg.expert_inter),
        "qk_norm": bool(cfg.qk_norm),
        "shared_expert_intermediate": int(cfg.shared_inter),
        # Где лежит таблица словаря: по этому ключу C/B-код и отладка
        # понимают, что EMB читает триты (kind 0), а не fp32.
        "embed": embed_storage,
        "quantize_embeddings": (
            embed_storage if embed_storage in ("trit", "int8") else "none"
        ),
        # C берёт eps RMSN из меты (приоритетнее imm/1000), а Python-эталон
        # (`vtm_ref.execute`) — только imm/1000. Если положить сюда точный
        # 1e-6, C и Python разойдутся на ~1e-3 относительной ошибки (на
        # малых активациях это видно в логитах). Поэтому в мету идёт eps,
        # представимый в imm (тот, что реально применяется), а исходный —
        # рядом: `rms_norm_eps_exact` и `model.eps`.
        "rms_norm_eps": cfg.eps_imm / 1000.0,
        "rms_norm_eps_exact": cfg.eps,
        "context_len": cfg.context_len,
        "scale_block": vtm_ref.SCALE_BLOCK,
        "model": dataclasses.asdict(cfg),
        "sampling": "greedy",
        # Решение по KV-бюджету видно без чтения stderr: сколько байт кэш,
        # какой кэп и что было запрошено. Ключ context_window появляется
        # только при урезании — кольцо KV равно длине буфера, поведение
        # исполнения не меняется, но потребитель меты видит фактическое окно.
        "kv_cache": {
            "bytes": int(kv_info["kv_bytes"]),
            "max_bytes": int(kv_info["max_bytes"]),
            "requested_context": int(kv_info["requested_context"]),
            "requested_kv_bytes": int(kv_info["requested_kv_bytes"]),
            "context_len": int(cfg.context_len),
            "reduced": bool(kv_info["reduced"]),
        },
    }
    if kv_info["reduced"]:
        meta["context_window"] = cfg.context_len
    if meta_extra:
        meta.update(meta_extra)
    check_q4_tmv_support(g.tensors, g.program)
    return vtm_ref.VTM(tensors=g.tensors, program=g.program, meta=meta)


# ---------------------------------------------------------------------------
# Манифест экспорта QAT -> веса компилятора
# ---------------------------------------------------------------------------

MANIFEST_FORMAT = "eva32.ternary.manifest"
MANIFEST_VERSION = 2
DEFAULT_CONTEXT_LEN = 2048

#: Контексты, между которыми компилятор урезает `context_len`, когда KV-кэш
#: не влезает в кэп загрузчика (`MAX_KV_BYTES`, в C — `VTM_MAX_KV_BYTES`).
#: 2048 — рабочая длина, на которой валидированы 1.5B/3B; 1024/512 — шаги
#: урезания (задача 0135). Дальше 512 не идём: 1.5B-форма на 512 занимает
#: 28 МиБ, и контекст короче 512 для чата уже не полезен — тогда честная
#: ошибка с командой пересборки, а не молчание.
CONTEXT_STEPS: tuple[int, ...] = (2048, 1024, 512)


def kv_cache_bytes(cfg: LayerConfig, context_len: int | None = None) -> int:
    """Байты KV-кэшей в арене: `2 × n_layers × ctx × kv_dim × 4` (K и V).

    В контейнер кэши не пишутся (задача 0101), но загрузчик выделяет под них
    буфер в арене и проверяет сумму по `VTM_MAX_KV_BYTES`; компилятор обязан
    считать ровно то же число, иначе запишет нечитаемый файл.
    """
    ctx = int(cfg.context_len if context_len is None else context_len)
    n_layers = int(getattr(cfg, "n_layers", 1))
    return 2 * n_layers * ctx * cfg.kv_dim * 4


def kv_warning_message(info: dict[str, Any]) -> str:
    """Предупреждение «ctx урезан» с числами (печатает компилятор)."""
    return (
        "предупреждение: KV-кэш при context_len="
        f"{info['requested_context']} — "
        f"{_human_bytes(info['requested_kv_bytes'])} "
        f"({info['n_layers']} слоёв × 2 × {info['requested_context']} × "
        f"{info['kv_dim']} × 4 Б), кэп загрузчика VTM_MAX_KV_BYTES = "
        f"{_human_bytes(info['max_bytes'])}; context_len уменьшен до "
        f"{info['chosen']} ({_human_bytes(info['kv_bytes'])}). "
        "Полный контекст — только с поднятым кэпом: "
        + vtm_ref.kv_rebuild_hint(info["requested_kv_bytes"])
    )


def kv_cap_error_message(info: dict[str, Any]) -> str:
    """Ошибка «не влезает даже минимальный ctx»: числа и команда пересборки."""
    smallest = info["candidates"][-1]
    return (
        "KV-кэш не влезает в кэп загрузчика VTM_MAX_KV_BYTES = "
        f"{_human_bytes(info['max_bytes'])}: при context_len="
        f"{info['requested_context']} он равен "
        f"{_human_bytes(info['requested_kv_bytes'])} "
        f"({info['n_layers']} слоёв × 2 × {info['requested_context']} × "
        f"{info['kv_dim']} × 4 Б), а при минимальном context_len={smallest} — "
        f"{_human_bytes(info['kv_bytes'])}. Соберите рантайм с бо́льшим кэпом: "
        + vtm_ref.kv_rebuild_hint(info["requested_kv_bytes"])
    )


def fit_context_window(
    cfg: ModelConfig,
    *,
    max_bytes: int | None = None,
    steps: Sequence[int] = CONTEXT_STEPS,
) -> tuple[ModelConfig, dict[str, Any]]:
    """Вписать KV-кэш в кэп загрузчика, при необходимости уменьшив `ctx`.

    Кандидаты — запрошенный `context_len` и шаги `steps` меньше него, по
    убыванию (2048 → 1024 → 512). Берётся первый, чей KV-кэш не больше
    `max_bytes` (по умолчанию `vtm_ref.MAX_KV_BYTES` = 512 МиБ — то же число,
    что `VTM_MAX_KV_BYTES` в `src/vtm/loader.c`). Кэп инклюзивный: ровно
    512 МиБ (32B при ctx 1024) влезает.

    Возвращает `(конфиг с финальным context_len, сводка)`; сводка попадает в
    мету контейнера (`meta["kv_cache"]`), чтобы урезание было видно без
    чтения stderr. Если не влезает ни один кандидат — `ValueError` с числами
    и командой пересборки `-DVTM_MAX_KV_BYTES=...` (молча не отказываем).
    """
    budget = vtm_ref.MAX_KV_BYTES if max_bytes is None else int(max_bytes)
    if budget <= 0:
        raise ValueError(f"бюджет KV должен быть > 0, получено {budget}")
    requested = int(cfg.context_len)
    candidates: list[int] = [requested]
    seen = {requested}
    for step in steps:
        step = int(step)
        if 0 < step < requested and step not in seen:
            seen.add(step)
            candidates.append(step)
    candidates.sort(reverse=True)
    values = {ctx: kv_cache_bytes(cfg, ctx) for ctx in candidates}
    chosen = next((ctx for ctx in candidates if values[ctx] <= budget), None)
    info: dict[str, Any] = {
        "requested_context": requested,
        "candidates": candidates,
        "kv_bytes_by_context": values,
        "requested_kv_bytes": values[requested],
        "max_bytes": budget,
        "n_layers": int(getattr(cfg, "n_layers", 1)),
        "kv_dim": int(cfg.kv_dim),
        "chosen": chosen,
        "kv_bytes": values[chosen if chosen is not None else candidates[-1]],
        "reduced": chosen is not None and chosen != requested,
    }
    if chosen is None:
        raise ValueError(kv_cap_error_message(info))
    if chosen == requested:
        return cfg, info
    return dataclasses.replace(cfg, context_len=chosen), info

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
    q4_params: int = 0
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
    if is_bias:
        key = _MATRIX_KEYS.get(tail)
        if key is not None:
            return "bias", layer, key[:-len("_proj")] + "_bias"
        return "unknown", layer, tail
    # MoE (задача 0172): роутер, экспертные банки и общий эксперт. Имена —
    # HF-раскладка Qwen2/3-MoE; в GGUF им соответствуют ffn_gate_inp,
    # ffn_{gate,up,down}_exps и ffn_{gate,up,down}_shexp (маппинг — в
    # tools/moe_import.py).
    if tail in ("mlp.gate", "moe.gate", "feed_forward.gate", "router"):
        return "router", layer, "router"
    if tail in ("mlp.experts.gate_proj", "experts.gate_proj", "moe.experts.gate_proj"):
        return "expert_gate", layer, "expert_gate"
    if tail in ("mlp.experts.up_proj", "experts.up_proj", "moe.experts.up_proj"):
        return "expert_up", layer, "expert_up"
    if tail in ("mlp.experts.down_proj", "experts.down_proj", "moe.experts.down_proj"):
        return "expert_down", layer, "expert_down"
    if tail in ("mlp.shared_expert.gate_proj", "shared_expert.gate_proj"):
        return "shared_gate", layer, "shared_gate"
    if tail in ("mlp.shared_expert.up_proj", "shared_expert.up_proj"):
        return "shared_up", layer, "shared_up"
    if tail in ("mlp.shared_expert.down_proj", "shared_expert.down_proj"):
        return "shared_down", layer, "shared_down"
    if tail in ("self_attn.q_norm", "attn.q_norm", "attention.q_norm"):
        return "norm", layer, "q_norm"
    if tail in ("self_attn.k_norm", "attn.k_norm", "attention.k_norm"):
        return "norm", layer, "k_norm"
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


def _sha256(data: Any) -> str:
    """sha256 от bytes или от read-only mmap: кусками, без копии файла."""
    digest = hashlib.sha256()
    if isinstance(data, memoryview):
        view = data.cast("B")
        for start in range(0, len(view), 8 << 20):
            digest.update(view[start:start + (8 << 20)])
    else:
        digest.update(data)
    return digest.hexdigest()


#: Файлы больше этого размера читаются через mmap: 7B q4.bin — 3,98 ГБ, и
#: `read_bytes()` плюс нарезка сегментов давали пик ~7 ГиБ на машине с 16 ГиБ.
MMAP_THRESHOLD = 256 << 20


def _read_file(path: Path, record: Any = None, verify: bool = True,
               mapped: bool = False) -> Any:
    """Прочитать файл манифеста: bytes или (для больших) read-only mmap.

    ``mapped=True`` — окно в файл вместо копии в память. Байты те же, поэтому
    контейнер получается побайтово идентичным; sha256 при включённой проверке
    считается кусками по отображению, без материализации всего файла.
    """
    expected = record.get("sha256") if isinstance(record, dict) else None
    use_map = mapped and path.stat().st_size > MMAP_THRESHOLD
    try:
        if use_map:
            with path.open("rb") as handle:
                data: Any = memoryview(
                    mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
                )
        else:
            data = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"не читается {path}: {exc}") from exc
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
    if dtype == "int8":
        # Таблица словаря (kind 4, задача 0044): payload — INT8-сегмент.
        qdata, scales, scale_dtype = vtm_ref.decode_int8_segment(payload)
        if len(qdata) != vtm_ref.prod(shape):
            raise ValueError(
                f"{name}: в INT8-сегменте {len(qdata)} значений, а shape даёт "
                f"{vtm_ref.prod(shape)}"
            )
        return vtm_ref.Tensor(
            name=short_name,
            kind=vtm_ref.KIND_INT8,
            dims=shape,
            qdata=qdata,
            scales=scales,
            scale_dtype=scale_dtype,
        )
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
    if kind in ("matrix", "norm", "router", "expert_gate", "expert_up",
                "expert_down", "shared_gate", "shared_up", "shared_down"):
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
            moe=bool(arch.get("moe", False)),
            n_experts=int(arch.get("n_experts") or 0),
            top_k=int(arch.get("top_k") or 0),
            expert_inter=int(arch.get("expert_intermediate") or 0),
            qk_norm=bool(arch.get("qk_norm", False)),
            shared_inter=int(arch.get("shared_expert_intermediate") or 0),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"architecture в манифесте некорректна: {exc}") from exc


def _is_q4_entry(entry: dict[str, Any]) -> bool:
    """Запись таблицы тензоров манифеста — Q4-сегмент (kind 5/6/7)?

    Импорт GGUF (0153) пишет `dtype: "q4_0"`, `kind: "q4"` и смещения
    `q4_offset`/`q4_length`; нативный импорт (0165) — `dtype/kind:
    "q4_k"|"q6_k"` и те же смещения; поддерживаем все признаки, чтобы
    манифест, собранный вручную, тоже читался.
    """
    dtype = str(entry.get("dtype") or "").lower()
    kind = str(entry.get("kind") or "").lower()
    return (
        dtype.startswith("q4") or dtype.startswith("q6")
        or kind == "q4" or kind in ("q4_k", "q6_k")
        or "q4_offset" in entry
    )


def _q4k_kind_of(entry: dict[str, Any]) -> int | None:
    """kind контейнера для нативной записи (6 — Q4_K, 7 — Q6_K) или None."""
    dtype = str(entry.get("dtype") or "").lower()
    kind = str(entry.get("kind") or "").lower()
    if kind == "q6_k" or dtype == "q6_k":
        return vtm_ref.KIND_Q6K
    if kind == "q4_k" or dtype == "q4_k":
        return vtm_ref.KIND_Q4K
    return None


def load_export(
    source: str | os.PathLike[str] | dict[str, Any],
    verify: bool = True,
) -> Export:
    """Прочитать манифест QAT/импорта GGUF и собрать `weights` для `compile_model`.

    Источник — путь к `manifest.json` (файлы `ternary.trit`, `q4.bin` и
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

    # --- сегменты весов: триты (ternary.trit) и Q4 (q4.bin) -----------------
    trit_name = "ternary.trit"
    trit_data = _read_file(base / trit_name, files.get(trit_name), verify)
    q4_name = str((manifest.get("q4") or {}).get("file") or "q4.bin")
    segment_entries = manifest.get("tensors") or []
    q4_entries = [entry for entry in segment_entries if _is_q4_entry(entry)]
    q4_data = (
        _read_file(base / q4_name, files.get(q4_name), verify, mapped=True)
        if q4_entries or q4_name in files
        else b""
    )

    weights: dict[str, Any] = {"layers": [{} for _ in range(cfg.n_layers)]}
    trit_params = 0
    q4_params = 0
    fp_params = 0
    unknown: list[str] = []

    def _layer_bucket(layer: int, what: str) -> dict[str, Any]:
        if not 0 <= layer < cfg.n_layers:
            raise ValueError(f"{what}: слой {layer} вне n_layers={cfg.n_layers}")
        return weights["layers"][layer]

    for entry in segment_entries:
        name = str(entry.get("name") or "?")
        shape = [int(x) for x in entry.get("shape") or []]
        role, layer, key = _weight_role(name)

        if _is_q4_entry(entry):
            # Q4-сегмент kind 5: заголовок + масштабы (+ min у Q4_1) + нибблы.
            offset = int(entry.get("q4_offset", entry.get("offset", -1)))
            length = int(entry.get("q4_length", entry.get("length", -1)))
            if offset < 0 or length <= 0 or offset + length > len(q4_data):
                raise ValueError(
                    f"{name}: Q4-сегмент [{offset}, {offset + length}) вне "
                    f"{q4_name} ({len(q4_data)} байт)"
                )
            if role in ("bias", "norm", "final_norm"):
                raise ValueError(
                    f"{name}: Q4 допустим только для матриц Linear и lm_head; "
                    "нормы/bias должны лежать в non_ternary.bin"
                )
            native_kind = _q4k_kind_of(entry)
            if role == "embed" and native_kind is None:
                raise ValueError(
                    f"{name}: Q4 (kind 5) в таблице словаря не поддержана — "
                    "EMB ждёт fp32/int8/триты или нативные блоки Q4_K/Q6_K "
                    "(kind 6/7, они читаются ядром EMB без fp32-копии)"
                )
            if role == "embed":
                tensor_name = "emb.w"
            elif role in (
                "matrix", "expert_gate", "expert_up", "expert_down",
                "shared_gate", "shared_up", "shared_down",
            ):
                tensor_name = f"l{layer}.{key}.w"
            else:
                tensor_name = "lm_head.w"
            if native_kind is not None:
                # kind 6/7 (0165): блоки GGML как есть, окно в mmap q4.bin.
                # Масштабы и min внутри блоков, отдельных массивов нет.
                qdata, decoded_kind = vtm_ref.decode_q4k_segment_ex(
                    q4_data[offset:offset + length], native_kind
                )
                if len(qdata) != vtm_ref.prod(shape):
                    raise ValueError(
                        f"{name}: в сегменте {len(qdata)} значений, а shape "
                        f"{tuple(shape)} даёт {vtm_ref.prod(shape)}"
                    )
                tensor = vtm_ref.Tensor.q4k(
                    tensor_name, shape, qdata, kind=decoded_kind
                )
            else:
                qdata, scales, scale_dtype, asym, mins = (
                    vtm_ref.decode_q4_segment_ex(q4_data[offset:offset + length])
                )
                if len(qdata) != vtm_ref.prod(shape):
                    raise ValueError(
                        f"{name}: в Q4-сегменте {len(qdata)} значений, а shape "
                        f"{tuple(shape)} даёт {vtm_ref.prod(shape)}"
                    )
                tensor = vtm_ref.Tensor.q4(
                    tensor_name, shape, qdata, scales, scale_dtype,
                    asymmetric=asym, mins=mins,
                )
            if role == "lm_head":
                if cfg.tie_word_embeddings:
                    continue  # tied: отдельная матрица не нужна
                if cfg.lm_head != "q4":
                    raise ValueError(
                        f"{name}: в architecture lm_head={cfg.lm_head!r}, "
                        "а в экспорте Q4-матрица; переэкспортируйте с "
                        "lm_head='q4' (импорт GGUF: --format q4)"
                    )
                _expect_shape(shape, (cfg.vocab, cfg.hidden), name)
                weights["lm_head"] = tensor
            elif role == "embed":
                _expect_shape(shape, (cfg.vocab, cfg.hidden), name)
                weights["embed"] = tensor
            elif role == "matrix":
                expected = _layer_matrix_dims(cfg, key)
                _expect_shape(shape, expected, name)
                _layer_bucket(layer, name)[key] = tensor
            elif role in ("expert_gate", "expert_up", "expert_down"):
                if not cfg.moe:
                    raise ValueError(
                        f"{name}: экспертный банк, а в architecture нет "
                        "moe=true — граф не MoE"
                    )
                _expect_shape(shape, _expert_bank_dims(cfg, key), name)
                _layer_bucket(layer, name)[key] = tensor
            elif role in ("shared_gate", "shared_up", "shared_down"):
                if not cfg.shared_inter:
                    raise ValueError(
                        f"{name}: общий эксперт, а в architecture "
                        "shared_expert_intermediate=0"
                    )
                _expect_shape(shape, _shared_dims(cfg, key), name)
                _layer_bucket(layer, name)[key] = tensor
            elif role == "unknown":
                unknown.append(name)
                continue
            else:  # pragma: no cover - роли выше перечислены
                raise ValueError(f"{name}: неожиданная роль {role!r}")
            if role not in (
                "expert_gate", "expert_up", "expert_down",
                "shared_gate", "shared_up", "shared_down",
            ) and (len(shape) != 2 or not all(shape)):
                raise ValueError(f"{name}: у матрицы плохая shape={shape}")
            q4_params += vtm_ref.prod(shape)
            continue

        # --- тритовый сегмент (ternary.trit) --------------------------------
        offset = int(entry.get("trit_offset", entry.get("offset", -1)))
        length = int(entry.get("trit_length", entry.get("length", -1)))
        if offset < 0 or length <= 0 or offset + length > len(trit_data):
            raise ValueError(
                f"{name}: сегмент [{offset}, {offset + length}) вне "
                f"ternary.trit ({len(trit_data)} байт)"
            )
        if role == "embed":
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
        elif role in ("bias", "norm", "final_norm"):
            raise ValueError(
                f"{name}: триты допустимы только для матриц Linear, lm_head "
                f"и таблицы словаря; нормы/bias должны лежать в "
                f"non_ternary.bin"
            )
        elif role == "lm_head":
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
        elif role == "matrix":
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
        elif role == "unknown":
            unknown.append(name)
            continue
        else:  # pragma: no cover - роли выше перечислены
            raise ValueError(f"{name}: неожиданная роль {role!r}")
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
    non_data = _read_file(base / non_file, files.get(non_file), verify, mapped=True)
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
            if tensor.kind == vtm_ref.KIND_INT8 and cfg.embed_quant != "int8":
                raise ValueError(
                    f"{name}: int8-таблица словаря, а в architecture "
                    f"embed={cfg.embed_quant!r}; переэкспортируйте чекпоинт "
                    "с quantize_embeddings: int8"
                )
            _expect_shape(shape, (cfg.vocab, cfg.hidden), name)
            if cfg.tie_word_embeddings and tensor.kind not in (
                vtm_ref.KIND_F32,
                vtm_ref.KIND_INT8,
            ):
                raise ValueError(
                    f"{name}: tied lm_head считается через MM по этой же "
                    f"таблице; нужен fp32, int8 (embed=int8) или тритовая "
                    f"таблица (embed=trit), а пришёл {tensor.kind_name}"
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
            if cfg.lm_head == "q4":
                raise ValueError(
                    f"{name}: в architecture lm_head='q4', а тензор плотный; "
                    "импортируйте GGUF с --format q4 (lm_head ляжет "
                    "Q4-сегментом в q4.bin) или переэкспортируйте с "
                    "lm_head='mm'"
                )
            _expect_shape(shape, (cfg.vocab, cfg.hidden), name)
            weights["lm_head"] = tensor
        elif kind == "router":
            if not cfg.moe:
                raise ValueError(
                    f"{name}: роутер MoE, а в architecture нет moe=true"
                )
            _expect_shape(shape, (cfg.n_experts, cfg.hidden), name)
            _layer_bucket(layer, name)["router"] = tensor
        elif kind in ("expert_gate", "expert_up", "expert_down"):
            if not cfg.moe:
                raise ValueError(
                    f"{name}: экспертный банк, а в architecture нет moe=true"
                )
            _expect_shape(shape, _expert_bank_dims(cfg, key), name)
            _layer_bucket(layer, name)[key] = tensor
        elif kind in ("shared_gate", "shared_up", "shared_down"):
            if not cfg.shared_inter:
                raise ValueError(
                    f"{name}: общий эксперт, а shared_expert_intermediate=0"
                )
            _expect_shape(shape, _shared_dims(cfg, key), name)
            _layer_bucket(layer, name)[key] = tensor
        elif kind in ("matrix", "norm", "bias"):
            bucket = _layer_bucket(layer, name)
            if kind == "matrix":
                _expect_shape(shape, _layer_matrix_dims(cfg, key), name)
                bucket[key] = tensor
            elif kind == "norm":
                expect = (
                    (cfg.head_dim,) if key in ("q_norm", "k_norm")
                    else (cfg.hidden,)
                )
                _expect_shape(shape, expect, name)
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
        if cfg.embed_quant == "int8":
            raise ValueError(
                "в экспорте нет int8-таблицы словаря: architecture "
                "embed='int8', но записи model.embed_tokens.weight с "
                "dtype: int8 в non_ternary.bin нет"
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
    if cfg.moe:
        required_keys = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "router", "expert_gate", "expert_up", "expert_down",
        ]
        if cfg.qk_norm:
            required_keys += ["q_norm", "k_norm"]
        if cfg.shared_inter:
            required_keys += ["shared_gate", "shared_up", "shared_down"]
    else:
        required_keys = list(_MATRIX_ORDER)
    for index, layer in enumerate(weights["layers"]):
        missing = [key for key in required_keys if key not in layer]
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
        q4_params=q4_params,
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
# Большие формы: синтетика без HF
# ---------------------------------------------------------------------------

#: Формы Qwen2.5, которые компилятор собирает в режиме `--synthetic --shape`
#: (и узнаёт по имени в `--qwen-config`). Значения — из HF-конфигов; словарь
#: 152064, tied lm_head (MM по тритовой таблице), eps 1e-6.
#:
#: KV-кэш при ctx 2048 (2 × слоёв × ctx × kv_dim × 4 Б): 1.5B — 112 МиБ,
#: 7B — 224 МиБ, 14B — 768 МиБ, 32B — 1024 МиБ. Кэп загрузчика 512 МиБ,
#: поэтому `fit_context_window` урежет 14B и 32B до ctx 1024 (384/512 МиБ);
#: 7B и 1.5B остаются на 2048.
#:
#: 32B-раскладка в packed20 — ~6,9 ГиБ, а контейнер v0.2 адресует файл
#: u32-смещениями (предел 4 ГиБ): граф и KV-бюджет поддержаны, но записать
#: такой файл формат не может — `write_vtm_stream` откажет с числами и
#: вариантами (u64-смещения/шардирование). 7B (~1,5 ГиБ) и 14B (~3,0 ГиБ)
#: пишутся и запускаются C-рантаймом.
SYNTHETIC_SHAPES: dict[str, dict[str, Any]] = {
    "1.5b": {
        "hidden": 1536,
        "n_heads": 12,
        "n_kv_heads": 2,
        "head_dim": 128,
        "intermediate": 8960,
        "vocab": 151936,
        "n_layers": 28,
        "context_len": 2048,
        "rope_theta": 1_000_000.0,
        "eps": 1e-6,
        "lm_head": "mm",
        "tie_word_embeddings": True,
    },
    "7b": {
        "hidden": 3584,
        "n_heads": 28,
        "n_kv_heads": 4,
        "head_dim": 128,
        "intermediate": 18944,
        "vocab": 152064,
        "n_layers": 28,
        "context_len": 2048,
        "rope_theta": 1_000_000.0,
        "eps": 1e-6,
        "lm_head": "mm",
        "tie_word_embeddings": True,
    },
    "14b": {
        "hidden": 5120,
        "n_heads": 40,
        "n_kv_heads": 8,
        "head_dim": 128,
        "intermediate": 13824,
        "vocab": 152064,
        "n_layers": 48,
        "context_len": 2048,
        "rope_theta": 1_000_000.0,
        "eps": 1e-6,
        "lm_head": "mm",
        "tie_word_embeddings": True,
    },
    "32b": {
        "hidden": 5120,
        "n_heads": 40,
        "n_kv_heads": 8,
        "head_dim": 128,
        "intermediate": 27648,
        "vocab": 152064,
        "n_layers": 64,
        "context_len": 2048,
        "rope_theta": 1_000_000.0,
        "eps": 1e-6,
        "lm_head": "mm",
        "tie_word_embeddings": True,
    },
}


def make_synthetic_shape(
    shape: str,
    *,
    seed: int = 7,
    context_len: int | None = None,
    max_kv_bytes: int | None = None,
) -> vtm_ref.VTM:
    """Синтетическая модель большой формы (1.5B/7B/14B/32B) без весов и HF.

    Граф — настоящий: `compile_model` строит те же тензоры, инструкции и
    KV-кэши, что у обученной модели формы. Значений нет: все веса — заглушки,
    а `vtm_ref.write_vtm_stream` пишет вместо payload'ов нули. Этого хватает,
    чтобы проверить сборку, загрузку C-рантаймом и проход по 1 токену на
    реальных размерностях (7B — ~1,5 ГиБ файла, 14B — ~3,0 ГиБ), не поднимая
    в памяти списки на миллиарды тритов и не качая HF. 32B (~6,9 ГиБ) в
    контейнер v0.2 не влезает (u32-смещения, предел 4 ГиБ): для него доступен
    граф/план без записи и честная ошибка при `--out`.

    `context_len` переопределяет форму; KV-бюджет (`max_kv_bytes`) передаётся
    в `compile_model`, который сам урежет ctx и напечатает предупреждение.

    eps=1e-3, а не 1e-6 из HF: `imm` в RMSN — тысячные, 1e-6 честно
    округляется до 0, а при нулевых весах это 0/0 = NaN. На форму eps не
    влияет (как в `make_synthetic`).
    """
    key = str(shape).strip().lower()
    arch = SYNTHETIC_SHAPES.get(key)
    if arch is None:
        known = ", ".join(sorted(SYNTHETIC_SHAPES))
        raise ValueError(f"форма {shape!r} неизвестна; известны: {known}")
    params = dict(arch)
    params["eps"] = 1e-3
    if context_len is not None:
        params["context_len"] = int(context_len)
    cfg = ModelConfig(**params)
    embed = vtm_ref.Tensor.placeholder(
        "emb.w", (cfg.vocab, cfg.hidden), vtm_ref.KIND_TRITS
    )
    vtm = compile_model(cfg, weights={"embed": embed}, max_kv_bytes=max_kv_bytes)
    # Активации `Graph.act` создаются нулями; потоковому писателю они не нужны,
    # а на 32B это ~60 МиБ списков.
    for tensor in vtm.tensors:
        if tensor.data is not None:
            tensor.data = None
    vtm.meta["generator"] = "tools/vtm_compile.py"
    vtm.meta["synthetic"] = {
        "kind": "shape",
        "shape": key,
        "seed": int(seed),
        "tokens": [1, 2, 3],
        "max_new": 0,
        "inputs_tensor": "emb.w",
        # Явно: контейнер без значений, проверяются форма/загрузка/проход.
        "weights": "zeros",
    }
    return vtm


# ---------------------------------------------------------------------------
# Qwen-каркас: только план, без весов
# ---------------------------------------------------------------------------

#: Архитектуры, которые мы знаем по имени (значения из HF-конфигов).
#: context_len — не max_position_embeddings модели, а длина KV-кэша: кэш
#: занимает ctx × layers × kv_dim × 8 байт (K и V, fp32) в памяти. В контейнер
#: он больше не пишется (задача 0101, VTM_TENSOR_FLAG_KV_CACHE), поэтому
#: контейнер не растёт от ctx; но буфер в арене всё равно 112 МиБ на 1.5B, так
#: что 32768 по-прежнему не берём — это рабочая длина VTM, а не максимум Qwen.
#: Сами формы — в `SYNTHETIC_SHAPES` (один источник чисел для --shape и плана).
_KNOWN_ARCH: dict[str, dict[str, Any]] = {
    "Qwen/Qwen2.5-1.5B-Instruct": dict(SYNTHETIC_SHAPES["1.5b"]),
    "Qwen/Qwen2.5-1.5B": dict(SYNTHETIC_SHAPES["1.5b"]),
    "Qwen/Qwen2.5-7B-Instruct": dict(SYNTHETIC_SHAPES["7b"]),
    "Qwen/Qwen2.5-7B": dict(SYNTHETIC_SHAPES["7b"]),
    "Qwen/Qwen2.5-14B-Instruct": dict(SYNTHETIC_SHAPES["14b"]),
    "Qwen/Qwen2.5-14B": dict(SYNTHETIC_SHAPES["14b"]),
    "Qwen/Qwen2.5-32B-Instruct": dict(SYNTHETIC_SHAPES["32b"]),
    "Qwen/Qwen2.5-32B": dict(SYNTHETIC_SHAPES["32b"]),
}


def read_model_name(config_path: str | os.PathLike[str]) -> str:
    """Достать `model.name_or_path` из YAML без PyYAML (он не в stdlib)."""
    with open(config_path, "r", encoding="utf-8") as handle:
        text = handle.read()
    match = re.search(r"^\s*name_or_path:\s*[\"']?([^\"'#\s]+)", text, re.MULTILINE)
    if not match:
        raise ValueError(f"в {config_path} нет model.name_or_path")
    return match.group(1)


def qwen_plan(
    config_path: str | os.PathLike[str],
    *,
    max_kv_bytes: int | None = None,
) -> tuple[str, ModelConfig, vtm_ref.VTM]:
    """Собрать каркас модели по Qwen-конфигу (веса не читаются)."""
    name = read_model_name(config_path)
    arch = _KNOWN_ARCH.get(name)
    if arch is None:
        known = ", ".join(sorted(_KNOWN_ARCH))
        raise ValueError(
            f"архитектура {name!r} неизвестна компилятору v0.1; известны: {known}"
        )
    cfg = ModelConfig(**arch)
    vtm = compile_model(cfg, max_kv_bytes=max_kv_bytes)
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


def kv_budget_lines(vtm: vtm_ref.VTM) -> list[str]:
    """Строки отчёта о KV-бюджете из `meta["kv_cache"]` (пусто — меты нет)."""
    kv = vtm.meta.get("kv_cache") or {}
    if not kv:
        return []
    context = int(kv.get("context_len") or vtm.meta.get("context_len") or 0)
    lines = [
        f"KV-кэш: {_human_bytes(int(kv['bytes']))} при context_len={context} "
        f"(кэп {_human_bytes(int(kv['max_bytes']))})"
    ]
    if kv.get("reduced"):
        lines.append(
            f"  урезано с context_len={kv['requested_context']} "
            f"({_human_bytes(int(kv['requested_kv_bytes']))} не влезало); "
            "полный контекст — пересборка: "
            + vtm_ref.kv_rebuild_hint(int(kv["requested_kv_bytes"]))
        )
    return lines


def _print_plan(name: str, cfg: ModelConfig, vtm: vtm_ref.VTM) -> None:
    ops = vtm.op_counts()
    context = int(vtm.meta.get("context_len") or cfg.context_len)
    print(f"каркас: {name}")
    print(
        f"hidden={cfg.hidden} heads={cfg.n_heads}/{cfg.n_kv_heads} "
        f"head_dim={cfg.head_dim} intermediate={cfg.intermediate} "
        f"vocab={cfg.vocab} layers={cfg.n_layers} context={context} "
        f"rope_theta={cfg.rope_theta:g}"
    )
    print(f"тензоров: {len(vtm.tensors)}, инструкций: {len(vtm.program)}")
    print("операции слоя×N: " + ", ".join(f"{k}×{v}" for k, v in sorted(ops.items())))
    for line in kv_budget_lines(vtm):
        print(line)
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
    context = int(vtm.meta.get("context_len") or cfg.context_len)
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
        f"vocab={cfg.vocab} layers={cfg.n_layers} context={context} "
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
            if embed.trits is not None
            or embed.data is not None
            or embed.qdata is not None
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
        elif embed.kind == vtm_ref.KIND_INT8:
            print(
                f"эмбеддинги: int8 (kind 4), сегмент non_ternary.bin "
                f"({_human_bytes(payload)} против {_human_bytes(fp32_bytes)} fp32, "
                f"масштабы на блок {vtm_ref.INT8_SCALE_BLOCK}; загрузчик "
                f"материализует fp32 в арене"
                + ("; tied lm_head — MM по той же таблице)" if export.tied else ")")
            )
        else:
            print(
                f"эмбеддинги: {embed.kind_name} в non_ternary.bin "
                f"({_human_bytes(fp32_bytes)})"
            )
        # Гранулярность масштаба таблицы на формат не влияет (масштабов всё
        # равно ceil(n/128), посрочный режим раскладывает масштаб строки по её
        # блокам), но в отчёте это видно: 0143.
        scale_mode = str(
            (export.manifest.get("cold_export") or {}).get("embed_scale")
            or (export.manifest.get("embeddings") or {}).get("scale_mode")
            or "block"
        )
        if scale_mode != "block":
            print(
                f"масштаб таблицы: {scale_mode} (один масштаб на строку, "
                f"разложен по блокам {cfg.block_size}; раскладка сегмента прежняя)"
            )
    print(f"тензоров: {len(vtm.tensors)}, инструкций: {len(vtm.program)}")
    print("операции: " + ", ".join(f"{name}×{count}" for name, count in sorted(ops.items())))
    if export.q4_params:
        files = export.manifest.get("files") or {}
        q4_bytes = int((files.get("q4.bin") or {}).get("bytes") or 0)
        q4_section = export.manifest.get("q4") or {}
        native = bool(q4_section.get("native"))
        kinds = "kind 6/7 (Q4_K/Q6_K нативно)" if native else "kind 5"
        lm_line = (
            "lm_head — MM по таблице словаря (tied)"
            if export.tied else f"lm_head — TMV по {kinds}"
        )
        print(
            f"Q4: {export.q4_params:,} весов {kinds} в q4.bin "
            f"({_human_bytes(q4_bytes)}), {lm_line}, без fp32-материализации"
        )
    cache_bytes = kv_cache_bytes(cfg, context)
    params_line = (
        f"параметры: тритов {export.trit_params:,}, "
        f"Q4 {export.q4_params:,}, fp {export.fp_params:,}; "
        f"KV-кэш {_human_bytes(cache_bytes)} в арене, в контейнере его нет"
    )
    print(params_line)
    for line in kv_budget_lines(vtm):
        print(line)
    if path:
        print(f"файл: {path} ({_human_bytes(os.path.getsize(path))})")
        print(
            "проверка C: vtm run <файл> --tokens ... --max-new 1; "
            "логиты сверять с meta.expected больше нечем — контейнер "
            "собран из чекпоинта, эталона HF в нём нет"
        )
    else:
        print("файл не записан (--out не задан)")


def _print_shape(
    vtm: vtm_ref.VTM,
    path: str | None,
    *,
    build_s: float,
    write_s: float | None = None,
    integrity: bool = False,
) -> None:
    """Отчёт по большой синтетической форме: размеры, KV, время сборки."""
    synthetic = vtm.meta.get("synthetic") or {}
    model = vtm.meta.get("model") or {}
    ops = vtm.op_counts()
    print(
        f"VTM синтетика формы {synthetic.get('shape', '?')}: "
        f"hidden={model.get('hidden')} "
        f"heads={model.get('n_heads')}/{model.get('n_kv_heads')} "
        f"head_dim={model.get('head_dim')} "
        f"intermediate={model.get('intermediate')} "
        f"vocab={model.get('vocab')} layers={model.get('n_layers')}"
    )
    print(
        f"тензоров: {len(vtm.tensors)} (предел таблицы {vtm_ref.MAX_TENSORS}), "
        f"инструкций: {len(vtm.program)}"
    )
    print("операции: " + ", ".join(f"{name}×{count}" for name, count in sorted(ops.items())))
    for line in kv_budget_lines(vtm):
        print(line)
    estimated = vtm.estimated_file_size()
    print(f"оценка размера: {_human_bytes(estimated)}")
    if estimated >= 2 ** 32:
        print(
            "файл не влезет в контейнер v0.2: смещения u32 (предел 4 ГиБ); "
            "нужен формат с u64-смещениями (ADR) или шардирование модели"
        )
    print(f"сборка графа: {build_s:.2f} с")
    if path:
        actual = os.path.getsize(path)
        print(
            f"файл: {path} ({_human_bytes(actual)}), запись "
            f"{write_s if write_s is not None else 0.0:.2f} с"
        )
        print("веса — нули (vtm_ref.write_vtm_stream): для проверки C: "
              f"src/vtm/vtm run {path} --tokens 1 --max-new 1")
        print("целостность: " + ("sha256 записан" if integrity else
                                "не пишется (VTM_WRITE_HASH=1 включит)"))
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
    parser.add_argument(
        "--shape",
        choices=tuple(sorted(SYNTHETIC_SHAPES)),
        help="полная форма для --synthetic: 1.5b/7b/14b/32b (без HF; --size/--mode "
             "не используются)",
    )
    parser.add_argument(
        "--ctx",
        type=int,
        help="context_len для --shape (по умолчанию из формы: 2048)",
    )
    parser.add_argument(
        "--max-kv-bytes",
        type=int,
        help="кэп KV-кэшей в байтах (по умолчанию vtm_ref.MAX_KV_BYTES = 512 МиБ, "
             "как VTM_MAX_KV_BYTES в C)",
    )
    parser.add_argument(
        "--integrity",
        action="store_true",
        help="считать sha256 при записи --shape (по умолчанию решает VTM_WRITE_HASH)",
    )
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
    if args.shape and not args.synthetic:
        parser.error("--shape работает только с --synthetic")
    if args.ctx is not None and not (args.synthetic and args.shape):
        parser.error("--ctx работает только с --synthetic --shape")

    try:
        if args.qwen_config:
            name, cfg, plan = qwen_plan(args.qwen_config, max_kv_bytes=args.max_kv_bytes)
            _print_plan(name, cfg, plan)
            if args.out:
                print(f"--out {args.out} проигнорирован: контейнер без весов не пишется.")
            return 0

        if args.manifest:
            export = load_export(args.manifest)
            vtm = compile_model(
                export.cfg, export.weights, export.meta,
                max_kv_bytes=args.max_kv_bytes,
            )
            out_dir = os.path.dirname(os.path.abspath(args.out))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            # MoE (задача 0172): контейнер с разреженными экспертами НЕ
            # получает integrity-блок. Загрузчик, увидев его, сверяет sha256
            # всей секции данных — то есть читает все веса на старте и
            # занимает RSS размером с модель. Ровно это MoE и обходит: 19 ГиБ
            # весов, из которых на токен нужны ~1,2 ГБ. Целостность при этом
            # не теряется: импорт пишет blocks_sha256 каждого сегмента
            # (tools/moe_import.py) и сверяет регионы с GGUF. Для плотных
            # моделей поведение прежнее — хеш пишется.
            integrity = bool(args.integrity) or not export.cfg.moe
            vtm_ref.write_vtm(
                args.out, vtm.tensors, vtm.program, vtm.meta,
                integrity=integrity,
            )
            _print_manifest(export, vtm, args.out)
            return 0

        if args.shape:
            started = time.perf_counter()
            vtm = make_synthetic_shape(
                args.shape,
                seed=args.seed,
                context_len=args.ctx,
                max_kv_bytes=args.max_kv_bytes,
            )
            build_s = time.perf_counter() - started
            write_s: float | None = None
            if args.out:
                out_dir = os.path.dirname(os.path.abspath(args.out))
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                started = time.perf_counter()
                vtm_ref.write_vtm_stream(
                    args.out, vtm.tensors, vtm.program, vtm.meta,
                    integrity=True if args.integrity else None,
                )
                write_s = time.perf_counter() - started
            _print_shape(
                vtm, args.out, build_s=build_s, write_s=write_s,
                integrity=bool(args.integrity),
            )
            return 0

        vtm = make_synthetic(seed=args.seed, size=args.size, mode=args.mode)
        if args.out:
            out_dir = os.path.dirname(os.path.abspath(args.out))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            vtm_ref.write_vtm(args.out, vtm.tensors, vtm.program, vtm.meta, integrity=True)
    except (ValueError, OSError) as exc:
        print(f"ошибка: {exc}", file=sys.stderr)
        return 2
    _print_synthetic(vtm, args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
