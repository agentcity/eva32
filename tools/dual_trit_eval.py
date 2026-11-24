#!/usr/bin/env python3
"""Два трита на вес (≈2 бита): сколько возвращает двойной размер (задача 0148).

Зачем
=====

Обычная третизация без обучения ломает Qwen2.5-0.5B: PPL по ответам
растёт в ~10^6 раз, декод — мусор (0136/0143, ``tools/parity_cold.py``).
Здесь проверяется одна гипотеза: ``W ≈ s1·A + s2·B``, где ``A`` и ``B`` —
триты, даёт больше уровней при вдвое большем размере. Окупается ли это?

Метод
=====

Только HF-сторона, без контейнера: для каждого ``nn.Linear`` (только
q/k/v/o/gate/up/down — таблица словаря и ``lm_head`` в этом опыте не
трогаются, это отдельная ось из 0143) вес раскладывается **посрочно**:

1. жадно: ``s1 = absmax(строка)``, ``A = clamp(round_half_away(w/s1), -1, 1)``,
   остаток ``r = w - s1·A``, ``s2 = absmax(r)``, ``B = round_half_away(r/s2)``;
2. один шаг наименьших квадратов по ``(s1, s2)`` при замороженных ``A, B``:
   решается 2×2 нормальная система ``G·s = [A·w, B·w]``, ``G = [[A·A, A·B],
   [A·B, B·B]]``. Вырожденные строки (``det ≈ 0``, нулевой остаток, ``B``
   коллинеарен ``A``) остаются на жадном решении; отрицательный ``s2`` после
   МНК эквивалентен смене знака ``B`` и хранится как положительный.

Уровни: у пары тритов 9 состояний ``{-1,0,+1}²``, но вдоль строки это
9 значений ``s1·a + s2·b`` (до 5 различных модулей: ``0, s2, |s1-s2|, s1,
s1+s2``). «4 уровня ≈ 2 бита» — неточная формулировка: пара тритов стоит
``2·1.6 = 3.2`` бита на вес, а не 2. Честный 4-уровневый код в 2 бита
(``int2``) в 1,6 раза меньше — это отдельная строка в таблице размеров.

Perplexity
==========

Способ ровно тот же, что в 0136/0143: импортируются
``tools.parity_cold.read_text_pairs``, ``build_examples`` и
``hf_answer_ppl`` — «Вопрос: …\\nОтвет:», маска на промпт, 20 первых пар
``data/train.jsonl``, лимит 128 токенов. Ничего не копируется и не
переписывается: числа сравнимы с 0143 (bf16 = 16,13).

Размеры
=======

Считаются по тем же правилам, что формат TRT1 (``docs/trit-format.md``):
20 тритов в 32-битном слове (1,6 бита), хвост выравнивается блоками по 160
тритов = 32 байта, масштабы fp16. Для двух тритов — два сегмента (две
таблицы масштабов). Оценка 3B — аналитическая, по конфигу Qwen2.5-3B
(параметры Linear и таблицы берутся из ``config.json``, веса не нужны).

Запуск
======

::

    PYTHONPATH=.pylibs python3.13 tools/dual_trit_eval.py \\
        --ppl-lines 20 --ppl-max-tokens 128

    # только план и размеры, без torch
    python3 tools/dual_trit_eval.py --dry-run

    # lean-режим под своп: bf16-хранилище (вдвое меньше RAM) и только декод
    PYTHONPATH=.pylibs python3.13 tools/dual_trit_eval.py \\
        --variants dual_block --quant-dtype bfloat16 --skip-ppl

Коды возврата: 0 — прогон сделан; 2 — нет torch/модели или плохие аргументы.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent.parent
#: ``tools`` — не пакет (namespace package), корень нужен в sys.path.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
#: Локальные зависимости (torch/transformers) — только если их ещё нет.

# Только чтение: способ замера PPL обязан совпадать с 0136/0143.
from tools import parity_cold as pc  # noqa: E402

def _enable_local_pylibs() -> None:
    """Добавить ``.pylibs`` в путь — только при запуске скриптом.

    Импорт модуля из тестов не должен менять ``sys.path``: иначе каталог
    ``.pylibs`` (собранный под 3.13) попадает в интерпретатор 3.14, ``import
    torch`` падает и навсегда травит ``sys.modules`` — после этого чужие тесты
    видят «torch есть» и падают вместо скипа.
    """
    path = ROOT / ".pylibs"
    if path.is_dir() and str(path) not in sys.path:
        sys.path.append(str(path))


#: Блок третизации из ``qat.ternary`` (эталон 0143).
BLOCK_SIZE = 128
#: Упаковка формата: 20 тритов в 32-битное слово.
TRITS_PER_WORD = 20
#: Блок хранения: 8 слов = 160 тритов = 32 байта.
TRITS_PER_STORAGE_BLOCK = 160
STORAGE_BLOCK_BYTES = 32
#: Масштабы в формате — fp16.
SCALE_BYTES = 2

#: Какие матрицы квантуются. ``lm_head``/``embed_tokens`` — не Linear-ось.
LINEAR_SUFFIXES = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
)

#: Варианты в порядке отчёта. ``dual_row`` — главный, остальные — опора.
ALL_VARIANTS = (
    "bf16",
    "trit_block",       # 1 трит, блок 128 — эталон 0143
    "trit_row",         # 1 трит, посрочно — честный пэр для dual_row
    "dual_row_greedy",  # 2 трита посрочно, без шага МНК
    "dual_row",         # 2 трита посрочно, жадно + один шаг МНК
    "dual_block",       # 2 трита, блок 128 (как поедет в формат)
    "int8_row",         # int8 посрочно — «сколько стоит точность рядом»
)

VARIANT_TITLES = {
    "bf16": "bf16 (эталон)",
    "trit_block": "1 трит, блок 128",
    "trit_row": "1 трит, посрочно",
    "dual_row_greedy": "2 трита, жадно",
    "dual_row": "2 трита, жадно+МНК",
    "dual_block": "2 трита, блок 128",
    "int8_row": "int8, посрочно",
}

#: Какой вид представления соответствует варианту в таблице размеров.
SIZE_KIND = {
    "bf16": "bf16",
    "trit_block": "trit_block",
    "trit_row": "trit_row",
    "dual_row_greedy": "dual_row",
    "dual_row": "dual_row",
    "dual_block": "dual_block",
    "int8_row": "int8_row",
}

#: Опора для оценки 3B, если конфига в кэше нет (Qwen2.5-3B-Instruct).
FALLBACK_3B_CONFIG = {
    "hidden_size": 2048,
    "intermediate_size": 11008,
    "num_hidden_layers": 36,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "vocab_size": 151936,
    "tie_word_embeddings": True,
}

DEFAULT_PPL_LINES = 20
DEFAULT_PPL_MAX_TOKENS = 128
DEFAULT_DECODE_TOKENS = 24
DEFAULT_JSON = ROOT / "data" / "dual_trit" / "eval.json"
#: Два промпта для декода: факт и инструкция (как в DEFAULT_PROMPTS 0136).
DECODE_PROMPTS = (
    "Столица Франции — это",
    "Как приготовить борщ? Опиши шаги коротко.",
)


class DualTritError(RuntimeError):
    """Ошибка прогона, которую видно пользователю без трейсбека."""


# ---------------------------------------------------------------------------
# Чистая математика: одна строка весов (её проверяют тесты без torch)
# ---------------------------------------------------------------------------


def round_half_away(value: float) -> int:
    """Округлить половину от нуля — правило ``qat.ternary``.

    ``round()`` в Python и ``torch.round()`` округляют половину к чётному
    (0.5 → 0), поэтому для квантования нужно своё правило.
    """
    if not math.isfinite(value):
        raise ValueError(f"нельзя округлить неконечное значение: {value!r}")
    if value >= 0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def _trit(value: float, scale: float) -> int:
    """Один вес в один трит при заданном масштабе (0 при нулевом масштабе)."""
    if scale <= 0.0:
        return 0
    return max(-1, min(1, round_half_away(value / scale)))


def single_trit(values: Sequence[float]) -> tuple[tuple[int, ...], float]:
    """Посрочная третизация: ``(триты, s = absmax строки)``."""
    vals = [float(value) for value in values]
    scale = max((abs(value) for value in vals), default=0.0)
    return tuple(_trit(value, scale) for value in vals), scale


def reconstruct_single(trits: Sequence[int], scale: float) -> list[float]:
    """``w ≈ trit · s``."""
    return [float(trit) * float(scale) for trit in trits]


@dataclass(frozen=True)
class RowDecomposition:
    """Разложение строки на два трита: ``w ≈ s1·A + s2·B``.

    ``greedy_s1/greedy_s2`` — масштабы до шага МНК (для проверки, что МНК
    не ухудшил), ``ls_applied`` — сошлась ли нормальная система, ``flipped``
    — пришлось ли менять знак ``B`` из-за отрицательного ``s2``.
    """

    s1: float
    s2: float
    a: tuple[int, ...]
    b: tuple[int, ...]
    greedy_s1: float
    greedy_s2: float
    ls_applied: bool
    flipped: bool


def dual_trit_row(values: Sequence[float], *, ls_step: bool = True) -> RowDecomposition:
    """Жадное разложение строки на два трита + (опционально) один шаг МНК.

    Шаг МНК заморожен на ``A, B`` из жадного прохода: решается только
    ``min_{s1,s2} ||w - s1·A - s2·B||²``. Поэтому МНК **не может** ухудшить
    ошибку жадного решения для тех же ``A, B`` — это проверяется тестом.
    """
    vals = [float(value) for value in values]
    if not vals:
        return RowDecomposition(0.0, 0.0, (), (), 0.0, 0.0, False, False)
    s1 = max(abs(value) for value in vals)
    a = tuple(_trit(value, s1) for value in vals)
    remainder = [value - s1 * trit for value, trit in zip(vals, a)]
    s2 = max((abs(value) for value in remainder), default=0.0)
    b = tuple(_trit(value, s2) for value in remainder)
    greedy_s1, greedy_s2 = s1, s2
    applied = False
    flipped = False
    if ls_step and s1 > 0.0 and s2 > 0.0:
        aa = sum(trit * trit for trit in a)
        bb = sum(trit * trit for trit in b)
        ab = sum(x * y for x, y in zip(a, b))
        aw = sum(trit * value for trit, value in zip(a, vals))
        bw = sum(trit * value for trit, value in zip(b, vals))
        det = aa * bb - ab * ab
        # det = ||A||²||B||² - (A·B)² — всегда ≥ 0; около нуля система
        # вырождена (B почти коллинеарен A), тогда оставляем жадное решение.
        if det > 1e-9 * max(aa * bb, 1.0):
            ls_s1 = (aw * bb - bw * ab) / det
            ls_s2 = (bw * aa - aw * ab) / det
            if math.isfinite(ls_s1) and math.isfinite(ls_s2) and ls_s1 > 0.0:
                if ls_s2 < 0.0:
                    # s2 < 0 — то же самое, что положительный s2 с -B:
                    # храним положительный масштаб, трит меняет знак.
                    ls_s2 = -ls_s2
                    b = tuple(-trit for trit in b)
                    flipped = True
                s1, s2 = ls_s1, ls_s2
                applied = True
    return RowDecomposition(s1, s2, a, b, greedy_s1, greedy_s2, applied, flipped)


def reconstruct_dual(row: RowDecomposition) -> list[float]:
    """``w ≈ s1·A + s2·B``."""
    return [
        row.s1 * float(a) + row.s2 * float(b)
        for a, b in zip(row.a, row.b)
    ]


def dual_trit_levels(s1: float, s2: float) -> tuple[float, ...]:
    """Различные значения пары тритов — их 9, а не 4 (см. докстринг модуля)."""
    values = {round(s1 * a + s2 * b, 12) for a in (-1, 0, 1) for b in (-1, 0, 1)}
    return tuple(sorted(values))


def relative_error(values: Sequence[float], approx: Sequence[float]) -> float:
    """``||w - w_hat|| / ||w||`` (0 при нулевом весе)."""
    numerator = math.fsum(
        (float(value) - float(other)) ** 2 for value, other in zip(values, approx)
    )
    denominator = math.fsum(float(value) ** 2 for value in values)
    if denominator <= 0.0:
        return 0.0
    return math.sqrt(numerator / denominator)


# ---------------------------------------------------------------------------
# Размеры по правилам формата (без torch: чистые числа)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SizeEstimate:
    """Размер сегмента: данные + масштабы, и эффективные биты на параметр."""

    kind: str
    params: int
    data_bytes: int
    scale_bytes: int

    @property
    def total_bytes(self) -> int:
        return int(self.data_bytes) + int(self.scale_bytes)

    @property
    def bits_per_param(self) -> float:
        if self.params <= 0:
            return 0.0
        return 8.0 * self.total_bytes / self.params


def packed_trit_bytes(n_values: int) -> int:
    """Байты под триты в формате TRT1: 160 тритов = 32 байта, хвост добит."""
    if n_values <= 0:
        return 0
    blocks = (int(n_values) + TRITS_PER_STORAGE_BLOCK - 1) // TRITS_PER_STORAGE_BLOCK
    return blocks * STORAGE_BLOCK_BYTES


def scale_block_count(n_values: int) -> int:
    """Число блоков масштаба 128 (как в ``qat.ternary``)."""
    if n_values <= 0:
        return 0
    return (int(n_values) + BLOCK_SIZE - 1) // BLOCK_SIZE


def size_estimate(kind: str, params: int, rows: int) -> SizeEstimate:
    """Оценить размер матрицы(целиком) в заданном представлении.

    ``rows`` — число групп масштаба при посрочном режиме (для блочных
    вариантов игнорируется). ``int2_block`` — честный 4-уровневый код в
    2 бита: он нужен, чтобы сравнить «два трита» не с Q4, а с тем, что
    реально соответствует формулировке «4 уровня ≈ 2 бита».
    """
    params = int(params)
    rows = max(int(rows), 0)
    if params <= 0:
        return SizeEstimate(kind, 0, 0, 0)
    if kind in ("bf16", "fp16"):
        return SizeEstimate(kind, params, 2 * params, 0)
    if kind == "trit_block":
        return SizeEstimate(kind, params, packed_trit_bytes(params),
                            scale_block_count(params) * SCALE_BYTES)
    if kind == "trit_row":
        return SizeEstimate(kind, params, packed_trit_bytes(params),
                            rows * SCALE_BYTES)
    if kind == "dual_block":
        return SizeEstimate(kind, params, 2 * packed_trit_bytes(params),
                            2 * scale_block_count(params) * SCALE_BYTES)
    if kind == "dual_row":
        return SizeEstimate(kind, params, 2 * packed_trit_bytes(params),
                            2 * rows * SCALE_BYTES)
    if kind == "int8_row":
        return SizeEstimate(kind, params, params, rows * SCALE_BYTES)
    if kind == "int2_block":
        return SizeEstimate(kind, params, (params + 3) // 4,
                            scale_block_count(params) * SCALE_BYTES)
    raise DualTritError(f"неизвестный вид представления: {kind!r}")


@dataclass(frozen=True)
class LinearShape:
    """Форма «Linear-части» модели — считает размеры без загрузки весов."""

    hidden: int
    intermediate: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int
    tied: bool

    @property
    def rows(self) -> int:
        """Строк (групп масштаба) во всех Linear — для посрочных режимов."""
        per_layer = (
            self.hidden                 # q_proj (heads·head_dim = hidden)
            + self.kv_heads * self.head_dim  # k_proj
            + self.kv_heads * self.head_dim  # v_proj
            + self.hidden               # o_proj
            + self.intermediate         # gate_proj
            + self.intermediate         # up_proj
            + self.hidden               # down_proj
        )
        return per_layer * self.layers

    @property
    def linear_params(self) -> int:
        return self.params_per_layer * self.layers

    @property
    def params_per_layer(self) -> int:
        hidden, inter = self.hidden, self.intermediate
        kv = self.kv_heads * self.head_dim
        return (
            hidden * hidden          # q_proj
            + kv * hidden            # k_proj
            + kv * hidden            # v_proj
            + hidden * hidden        # o_proj
            + inter * hidden         # gate_proj
            + inter * hidden         # up_proj
            + hidden * inter         # down_proj
        )

    @property
    def embed_params(self) -> int:
        """Таблица словаря; при ``tie_word_embeddings`` она же и lm_head."""
        return self.vocab * self.hidden

    @property
    def embed_rows(self) -> int:
        """Строк таблицы — для посрочных масштабов int8."""
        return self.vocab

    @property
    def total_params(self) -> int:
        return self.linear_params + self.embed_params


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    """Достать поле из dict (config.json) или HF-конфига."""
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def linear_shape_from_config(config: Any) -> LinearShape:
    """Форма Linear-части из конфига Qwen2 (или похожего)."""
    hidden = int(_config_value(config, "hidden_size", 0) or 0)
    intermediate = int(_config_value(config, "intermediate_size", 0) or 0)
    layers = int(_config_value(config, "num_hidden_layers", 0) or 0)
    heads = int(_config_value(config, "num_attention_heads", 0) or 0)
    kv_heads = int(_config_value(config, "num_key_value_heads", heads) or heads)
    head_dim = _config_value(config, "head_dim", None)
    if not head_dim:
        head_dim = hidden // heads if heads else 0
    vocab = int(_config_value(config, "vocab_size", 0) or 0)
    tied = bool(_config_value(config, "tie_word_embeddings", True))
    if min(hidden, intermediate, layers, heads, head_dim, vocab) <= 0:
        raise DualTritError("в конфиге не хватает размеров модели")
    return LinearShape(hidden, intermediate, layers, heads, kv_heads,
                       int(head_dim), vocab, tied)


def shape_from_model(model: Any) -> LinearShape:
    """Форма по фактическим модулям — чтобы отчёт не зависел от догадок."""
    linear_params = 0
    for _name, module in linear_targets(model):
        linear_params += int(module.weight.numel())
    config = model.config
    shape = linear_shape_from_config(config)
    if shape.linear_params != linear_params:
        raise DualTritError(
            f"конфиг обещает {shape.linear_params} параметров Linear, "
            f"а в модели {linear_params} — формула не сходится"
        )
    return shape


def size_rows(shape: LinearShape, kinds: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Таблица размеров Linear (и таблица словаря отдельно) без загрузки весов."""
    out: dict[str, dict[str, Any]] = {}
    for kind in kinds:
        estimate = size_estimate(kind, shape.linear_params, shape.rows)
        out[kind] = {
            "kind": kind,
            "linear_bytes": estimate.total_bytes,
            "bits_per_param": estimate.bits_per_param,
            "embed_bf16_bytes": shape.embed_params * 2,
            "embed_trit_bytes": size_estimate(
                "trit_block", shape.embed_params, 0).total_bytes,
            "embed_int8_bytes": size_estimate(
                "int8_row", shape.embed_params, shape.embed_rows).total_bytes,
        }
    return out


def file_bytes(linear: SizeEstimate, embed_kind: str, shape: LinearShape) -> int:
    """Размер «полного файла» = Linear в этом виде + таблица в ``embed_kind``."""
    if embed_kind in ("bf16", "fp16"):
        embed = shape.embed_params * 2
    elif embed_kind == "trit_block":
        embed = size_estimate("trit_block", shape.embed_params, 0).total_bytes
    elif embed_kind == "int8":
        embed = size_estimate("int8_row", shape.embed_params, shape.embed_rows).total_bytes
    else:
        raise DualTritError(f"неизвестный вид таблицы: {embed_kind!r}")
    return int(linear.total_bytes) + int(embed)


# ---------------------------------------------------------------------------
# Имена модулей (без torch: выбор матриц проверяется тестом)
# ---------------------------------------------------------------------------


def is_linear_name(name: str) -> bool:
    """Квантуется ли модуль с таким именем (только 7 матриц Qwen2)."""
    return name.rsplit(".", 1)[-1] in LINEAR_SUFFIXES


def linear_targets(model: Any) -> list[tuple[str, Any]]:
    """Список ``(имя, nn.Linear)`` для квантования — без ``lm_head``/эмбеддингов."""
    torch = _torch()
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and is_linear_name(name)
    ]


# ---------------------------------------------------------------------------
# Torch: квантование матриц
# ---------------------------------------------------------------------------


def _torch():
    try:
        import torch  # noqa: PLC0415 - только в прогоне
    except ImportError as exc:  # pragma: no cover - зависит от машины
        raise DualTritError(
            "нужен torch (.pylibs): PYTHONPATH=.pylibs python3.13 "
            "tools/dual_trit_eval.py"
        ) from exc
    return torch


def _round_half_away_torch(tensor: Any) -> Any:
    torch = _torch()
    return torch.sign(tensor) * torch.floor(tensor.abs() + 0.5)


def _flatten_for_blocks(weight: Any, block_size: int | None) -> tuple[Any, tuple[int, ...]]:
    """Привести вес к строкам: посрочно (``None``) или к блокам 128.

    Возвращает ``(X, shape)``: ``X`` — 2D ``[групп, ширина]``, ``shape`` —
    исходная форма (для обратного reshape с обрезкой padding'а).
    """
    torch = _torch()
    original = tuple(weight.shape)
    if weight.ndim != 2:
        raise DualTritError(f"ожидалась 2D-матрица Linear, получено {original}")
    rows, cols = int(weight.shape[0]), int(weight.shape[1])
    if block_size is None:
        return weight.reshape(rows, cols), original
    blocks = (cols + block_size - 1) // block_size
    pad = blocks * block_size - cols
    flat = weight
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    return flat.reshape(rows * blocks, block_size), original


def _restore_rows(flat: Any, original: tuple[int, ...]) -> Any:
    """Обратный reshape после :func:`_flatten_for_blocks` (без padding'а)."""
    rows, cols = original
    return flat.reshape(rows, -1)[:, :cols].reshape(original)


@dataclass
class WeightStats:
    """Сколько стоила одна матрица: ошибка и ветки разложения."""

    relative_error: float
    ls_applied: int = 0
    ls_fallback: int = 0
    flipped: int = 0


def trit_weight(weight: Any, block_size: int | None = None) -> tuple[Any, WeightStats]:
    """Третизация (1 трит): absmax блока (или строки), ``s`` — на блок."""
    torch = _torch()
    with torch.no_grad():
        w = weight.detach().float()
        flat, original = _flatten_for_blocks(w, block_size)
        scales = flat.abs().amax(dim=1, keepdim=True)
        safe = torch.where(scales > 0, scales, torch.ones_like(scales))
        trits = _round_half_away_torch(flat / safe).clamp(-1.0, 1.0)
        trits = torch.where(scales > 0, trits, torch.zeros_like(trits))
        deq = _restore_rows(trits * scales, original)
        error = _frobenius_error(w, deq)
    return deq, WeightStats(relative_error=error)


def dual_trit_weight(
    weight: Any,
    block_size: int | None = None,
    *,
    ls_step: bool = True,
) -> tuple[Any, WeightStats]:
    """Два трита на вес: жадно + (опционально) один шаг МНК по ``s1, s2``."""
    torch = _torch()
    with torch.no_grad():
        w = weight.detach().float()
        flat, original = _flatten_for_blocks(w, block_size)
        # 1. Жадно: s1 = absmax, A = round(w/s1), остаток r, s2 = absmax(r).
        s1 = flat.abs().amax(dim=1)
        safe1 = torch.where(s1 > 0, s1, torch.ones_like(s1))
        a = _round_half_away_torch(flat / safe1[:, None]).clamp(-1.0, 1.0)
        a = torch.where((s1 > 0)[:, None], a, torch.zeros_like(a))
        remainder = flat - s1[:, None] * a
        s2 = remainder.abs().amax(dim=1)
        safe2 = torch.where(s2 > 0, s2, torch.ones_like(s2))
        b = _round_half_away_torch(remainder / safe2[:, None]).clamp(-1.0, 1.0)
        b = torch.where((s2 > 0)[:, None], b, torch.zeros_like(b))

        greedy_s1, greedy_s2 = s1.clone(), s2.clone()
        ls_applied = 0
        flipped = 0
        if ls_step:
            aa = (a * a).sum(dim=1)
            bb = (b * b).sum(dim=1)
            ab = (a * b).sum(dim=1)
            aw = (a * flat).sum(dim=1)
            bw = (b * flat).sum(dim=1)
            det = aa * bb - ab * ab
            ok = (det > 1e-6 * torch.clamp(aa * bb, min=1.0)) & (aa > 0) & (bb > 0)
            det_safe = torch.where(ok, det, torch.ones_like(det))
            ls_s1 = (aw * bb - bw * ab) / det_safe
            ls_s2 = (bw * aa - aw * ab) / det_safe
            good = ok & (ls_s1 > 0) & torch.isfinite(ls_s1) & torch.isfinite(ls_s2)
            s1 = torch.where(good, ls_s1, greedy_s1)
            s2_ls = torch.where(good, ls_s2, greedy_s2)
            # Отрицательный s2 — это тот же вес с -B; храним положительный.
            flip = good & (s2_ls < 0)
            b = torch.where(flip[:, None], -b, b)
            s2 = s2_ls.abs()
            ls_applied = int(good.sum())
            flipped = int(flip.sum())
        deq = _restore_rows(s1[:, None] * a + s2[:, None] * b, original)
        error = _frobenius_error(w, deq)
    return deq, WeightStats(
        relative_error=error,
        ls_applied=ls_applied,
        ls_fallback=int(s1.numel() - ls_applied) if ls_step else int(s1.numel()),
        flipped=flipped,
    )


def int8_row_weight(weight: Any) -> tuple[Any, WeightStats]:
    """Симметричный int8 посрочно: ``s = absmax/127``, ``q = round(w/s)``."""
    torch = _torch()
    with torch.no_grad():
        w = weight.detach().float()
        rows, cols = w.shape
        flat = w.reshape(rows, cols)
        scales = flat.abs().amax(dim=1, keepdim=True) / 127.0
        safe = torch.where(scales > 0, scales, torch.ones_like(scales))
        quantized = _round_half_away_torch(flat / safe).clamp(-127.0, 127.0)
        quantized = torch.where(scales > 0, quantized, torch.zeros_like(quantized))
        deq = (quantized * scales).reshape(w.shape)
        error = _frobenius_error(w, deq)
    return deq, WeightStats(relative_error=error)


def _frobenius_error(reference: Any, approx: Any) -> float:
    """``||w - w_hat||_F / ||w||_F`` на тензорах."""
    torch = _torch()
    numerator = torch.linalg.vector_norm((reference - approx).double())
    denominator = torch.linalg.vector_norm(reference.double())
    if float(denominator) <= 0.0:
        return 0.0
    return float(numerator / denominator)


def quantize_model(model: Any, variant: str, *, ls_step: bool = True) -> dict[str, Any]:
    """Заменить веса всех целевых Linear на их квантованную версию.

    Возвращает отчёт: сколько матриц/параметров, глобальная относительная
    ошибка веса и счётчики веток разложения. Модель меняется **на месте**.
    """
    if variant == "bf16":
        return {"variant": variant, "modules": 0, "params": 0,
                "relative_error": 0.0, "ls_applied": 0, "ls_fallback": 0,
                "flipped": 0, "seconds": 0.0}
    targets = linear_targets(model)
    if not targets:
        raise DualTritError("в модели не нашлось ни одной q/k/v/o/gate/up/down матрицы")
    started = time.monotonic()
    params = 0
    error_num = 0.0
    error_den = 0.0
    ls_applied = ls_fallback = flipped = 0
    for _name, module in targets:
        weight = module.weight.detach()
        # Квадрат нормы исходного веса — до copy_: weight делит хранилище
        # с module.weight, и после подмены он уже квантованный.
        original_sq = float(weight.detach().double().pow(2).sum())
        if variant == "trit_block":
            deq, stats = trit_weight(weight, BLOCK_SIZE)
        elif variant == "trit_row":
            deq, stats = trit_weight(weight, None)
        elif variant == "dual_row_greedy":
            deq, stats = dual_trit_weight(weight, None, ls_step=False)
        elif variant == "dual_row":
            deq, stats = dual_trit_weight(weight, None, ls_step=ls_step)
        elif variant == "dual_block":
            deq, stats = dual_trit_weight(weight, BLOCK_SIZE, ls_step=ls_step)
        elif variant == "int8_row":
            deq, stats = int8_row_weight(weight)
        else:
            raise DualTritError(f"неизвестный вариант: {variant!r}")
        with _torch().no_grad():
            module.weight.copy_(deq.reshape(module.weight.shape).to(module.weight.dtype))
        count = int(weight.numel())
        params += count
        error_num += (stats.relative_error ** 2) * original_sq
        error_den += original_sq
        ls_applied += stats.ls_applied
        ls_fallback += stats.ls_fallback
        flipped += stats.flipped
        del deq, stats
    relative = math.sqrt(error_num / error_den) if error_den > 0 else 0.0
    return {
        "variant": variant,
        "modules": len(targets),
        "params": params,
        "relative_error": relative,
        "ls_applied": ls_applied,
        "ls_fallback": ls_fallback if variant in ("dual_row", "dual_block") else 0,
        "flipped": flipped,
        "seconds": time.monotonic() - started,
    }


# ---------------------------------------------------------------------------
# Декод двух ответов
# ---------------------------------------------------------------------------


def decode_answers(
    torch: Any,
    tokenizer: Any,
    model: Any,
    prompts: Sequence[str],
    *,
    max_new_tokens: int,
    device: Any,
) -> list[dict[str, Any]]:
    """Жадный декод по двум промптам (chat-template, как у Instruct-модели)."""
    answers: list[dict[str, Any]] = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:  # noqa: BLE001 - нет шаблона: обычный текст
            text = prompt
        encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                use_cache=True,
                pad_token_id=int(getattr(tokenizer, "pad_token_id", 0) or 0),
                eos_token_id=getattr(tokenizer, "eos_token_id", None),
            )
        new_tokens = generated[0, input_ids.shape[1]:]
        answers.append({
            "prompt": prompt,
            "text": tokenizer.decode(new_tokens, skip_special_tokens=True).strip(),
            "tokens": int(new_tokens.numel()),
        })
    return answers


# ---------------------------------------------------------------------------
# Прогон
# ---------------------------------------------------------------------------


def _cached_snapshot(repo: str) -> Path | None:
    """Локальный снапшот из ``.hfcache`` (сеть не нужна)."""
    base = ROOT / ".hfcache" / "hub" / ("models--" + repo.replace("/", "--")) / "snapshots"
    if not base.is_dir():
        return None
    found = sorted(path for path in base.iterdir() if (path / "config.json").is_file())
    return found[-1] if found else None


def default_model() -> str:
    """Путь к Qwen2.5-0.5B из кэша; иначе — имя репозитория."""
    snapshot = _cached_snapshot("Qwen/Qwen2.5-0.5B-Instruct")
    return str(snapshot) if snapshot is not None else "Qwen/Qwen2.5-0.5B-Instruct"


def three_b_config() -> dict[str, Any]:
    """Конфиг 3B из кэша (только ``config.json``, веса не читаются)."""
    snapshot = _cached_snapshot("Qwen/Qwen2.5-3B-Instruct")
    if snapshot is not None:
        return json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    return dict(FALLBACK_3B_CONFIG)


def parse_variants(text: str) -> list[str]:
    """Разобрать ``--variants``, сохранив порядок и отбросив дубли."""
    variants: list[str] = []
    for raw in str(text).split(","):
        name = raw.strip()
        if not name:
            continue
        if name == "all":
            for item in ALL_VARIANTS:
                if item not in variants:
                    variants.append(item)
            continue
        if name not in ALL_VARIANTS:
            raise DualTritError(
                f"неизвестный вариант {name!r}; допустимы: {', '.join(ALL_VARIANTS)}"
            )
        if name not in variants:
            variants.append(name)
    if not variants:
        raise DualTritError("пустой список вариантов")
    return variants


def build_ppl_examples(tokenizer: Any, args: argparse.Namespace, vocab: int) -> list[dict[str, Any]]:
    """Примеры PPL — ровно через ``tools.parity_cold`` (числа сравнимы с 0143)."""
    pairs = pc.read_text_pairs(args.ppl_data, args.ppl_lines)
    examples, dropped = pc.build_examples(
        tokenizer, pairs, max_tokens=args.ppl_max_tokens, vocab=vocab
    )
    if not examples:
        raise DualTritError("perplexity: после обрезки не осталось пар")
    if dropped:
        print(f"[dual_trit] PPL: отброшено {dropped} пар", file=sys.stderr)
    return examples


def peak_rss_mb() -> float:
    """Пик RSS процесса (байты на macOS, килобайты на Linux)."""
    try:
        import resource  # noqa: PLC0415
    except ImportError:  # pragma: no cover - Windows
        return 0.0
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / 1e6
    return usage / 1024.0


def run(args: argparse.Namespace) -> int:
    variants = parse_variants(args.variants)
    torch, tokenizer, model = pc.load_hf("bfloat16", args.model)
    device = pc.resolve_device(torch, args.device)
    results: dict[str, Any] = {}
    shape: LinearShape | None = None
    examples: list[dict[str, Any]] = []
    answers: dict[str, list[dict[str, Any]]] = {}
    started = time.monotonic()

    # Токенизатор и примеры берём один раз: модель перезагружается под
    # каждый вариант, чтобы не держать в RAM две копии весов (2 ГБ fp32).
    # Словарь — из конфига модели: у токенизатора Qwen ``vocab_size`` 151643
    # без added-токенов, а id доходят до 151935.
    vocab = int(model.config.vocab_size)
    if not args.skip_ppl:
        examples = build_ppl_examples(tokenizer, args, vocab)
        print(
            f"[dual_trit] PPL: строк {len(examples)}, токенов ответа "
            f"{sum(item['answer_tokens'] for item in examples)}, лимит "
            f"{args.ppl_max_tokens}", file=sys.stderr,
        )
    del model
    gc.collect()

    for variant in variants:
        print(f"[dual_trit] вариант {variant}: загрузка", file=sys.stderr)
        dtype = "bfloat16" if variant == "bf16" else args.quant_dtype
        torch, _tok, model = pc.load_hf(dtype, args.model)
        model.to(device).eval()
        if shape is None:
            shape = shape_from_model(model)
            print(
                f"[dual_trit] Linear: {shape.linear_params} параметров, "
                f"{shape.rows} строк; таблица {shape.embed_params}", file=sys.stderr,
            )
        stage = time.monotonic()
        quant = quantize_model(model, variant)
        quant["quantize_seconds"] = time.monotonic() - stage
        print(
            f"[dual_trit] {variant}: относительная ошибка веса "
            f"{quant['relative_error']:.4g}", file=sys.stderr,
        )
        if args.skip_ppl:
            ppl = {"perplexity": None, "tokens": 0, "nll_sum": 0.0}
        else:
            ppl = pc.hf_answer_ppl(torch, model, examples, progress=False)
            print(
                f"[dual_trit] {variant}: PPL {ppl['perplexity']:.6g} "
                f"({ppl['tokens']} токенов, {ppl['nll_sum']:.4f} NLL)",
                file=sys.stderr,
            )
        decoded: list[dict[str, Any]] = []
        if not args.no_decode:
            decoded = decode_answers(
                torch, tokenizer, model, DECODE_PROMPTS,
                max_new_tokens=args.decode_tokens, device=device,
            )
            answers[variant] = decoded
        results[variant] = {
            "ppl": ppl["perplexity"],
            "tokens": ppl["tokens"],
            "nll_sum": ppl["nll_sum"],
            "quant": quant,
            "answers": decoded,
        }
        del model
        gc.collect()
        if device.type == "cuda":  # pragma: no cover - на CPU не бывает
            torch.cuda.empty_cache()

    baseline = results.get("bf16", {}).get("ppl")
    for item in results.values():
        item["ratio_vs_bf16"] = (
            item["ppl"] / baseline if baseline and item["ppl"] is not None else None
        )
    assert shape is not None
    sizes = {
        variant: size_rows(shape, [SIZE_KIND[variant]])[SIZE_KIND[variant]]
        for variant in variants
    }
    result = {
        "format": "eva32.dual_trit.eval",
        "format_version": 1,
        "model": args.model,
        "device": str(device),
        "quant_dtype": args.quant_dtype,
        "ppl": {
            "data": str(Path(args.ppl_data).resolve()),
            "lines": len(examples),
            "tokens": results[variants[0]]["tokens"],
            "max_tokens": int(args.ppl_max_tokens),
            "method": "tools.parity_cold.hf_answer_ppl",
            "prompt_format": f"{pc.PROMPT_PREFIX}…{pc.PROMPT_SUFFIX}",
        },
        "decode_tokens": int(args.decode_tokens),
        "shape": {
            "hidden": shape.hidden,
            "intermediate": shape.intermediate,
            "layers": shape.layers,
            "linear_params": shape.linear_params,
            "linear_rows": shape.rows,
            "embed_params": shape.embed_params,
            "total_params": shape.total_params,
        },
        "variants": results,
        "sizes": sizes,
        "decode_prompts": list(DECODE_PROMPTS),
        "timings": {"total_seconds": time.monotonic() - started},
        "peak_rss_mb": peak_rss_mb(),
        "three_b": three_b_report(three_b_config()),
    }
    report = format_report(result)
    print()
    print(report)
    if args.json:
        path = Path(args.json).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"\nрезультат: {path}")
    return 0


def three_b_report(config: dict[str, Any]) -> dict[str, Any]:
    """Аналитическая оценка 3B: размеры под два трита, один трит, int2, Q4."""
    shape = linear_shape_from_config(config)
    rows: dict[str, Any] = {}
    for kind in ("trit_block", "dual_block", "dual_row", "int2_block", "int8_row", "bf16"):
        estimate = size_estimate(kind, shape.linear_params, shape.rows)
        rows[kind] = {
            "linear_bytes": estimate.total_bytes,
            "bits_per_param": estimate.bits_per_param,
            "file_trit_embed": file_bytes(estimate, "trit_block", shape),
            "file_int8_embed": file_bytes(estimate, "int8", shape),
        }
    return {
        "config": {
            "hidden": shape.hidden,
            "intermediate": shape.intermediate,
            "layers": shape.layers,
            "linear_params": shape.linear_params,
            "linear_rows": shape.rows,
            "embed_params": shape.embed_params,
            "tied": shape.tied,
        },
        "sizes": rows,
        # Ориентиры из задачи/0143: Q4 GGUF 3B ≈ 2 ГБ, один трит = 688 МБ.
        "reference": {"q4_gguf_bytes": 2 * 10 ** 9, "one_trit_bytes": 688 * 10 ** 6},
    }


def _mb(value: float) -> str:
    return f"{value / 1e6:,.1f}".replace(",", " ")


def _ratio_text(ratio: float | None) -> str:
    """Кратко, но без потери порядка: ×7,44e+03 читается лучше «7440»."""
    if ratio is None:
        return "—"
    if ratio < 1e3:
        return f"{ratio:.3f}"
    if ratio < 1e5:
        return f"{ratio:,.0f}".replace(",", " ")
    return f"{ratio:.3g}"


def format_report(result: dict[str, Any]) -> str:
    """Текстовая таблица: PPL, ×bf16, размер, биты на параметр, декод."""
    ppl = result["ppl"]
    shape = result["shape"]
    lines = [
        "=== 0148: два трита на вес (≈2 бита) ===",
        f"модель: {result['model']}",
        f"PPL: {ppl['data']}, строк {ppl['lines']}, токенов ответа "
        f"{ppl['tokens']}, лимит {ppl['max_tokens']}",
        f"метод: {ppl['method']} (тот же, что 0136/0143; bf16 = 16,13)",
        f"dtype квантованных весов: {result.get('quant_dtype', 'float32')} "
        "(fp32 — как эталон 0143; bf16 — lean-режим под своп)",
        f"Linear: {shape['linear_params']:,} параметров, {shape['linear_rows']:,} строк; "
        f"таблица {shape['embed_params']:,}".replace(",", " "),
        f"пик RSS: {result['peak_rss_mb']:.0f} МБ, "
        f"время {result['timings']['total_seconds']:.0f} с",
        "",
        "вариант                    PPL          ×bf16    Linear МБ  бит/пар  файл МБ*  err веса",
    ]
    for variant, item in result["variants"].items():
        sizes = result["sizes"].get(variant)
        ratio_text = _ratio_text(item.get("ratio_vs_bf16"))
        if sizes is None:
            linear_mb = bits = file_mb = "—"
        else:
            linear_mb = _mb(sizes["linear_bytes"])
            bits = f"{sizes['bits_per_param']:.2f}"
            file_mb = _mb(sizes["linear_bytes"] + sizes["embed_bf16_bytes"])
        ppl_text = f"{item['ppl']:.6g}" if item["ppl"] is not None else "—"
        lines.append(
            f"{VARIANT_TITLES.get(variant, variant):<26} {ppl_text:>12}  "
            f"{ratio_text:>8}  {linear_mb:>9}  {bits:>7}  {file_mb:>9}  "
            f"{item['quant']['relative_error']:.3g}"
        )
    lines.append("* файл = Linear в этом виде + таблица словаря в bf16 (272 МБ): "
                 "таблица — отдельная ось (0143), здесь не квантуется.")
    lines.append("")
    lines.append("декод (жадно, chat-template, "
                 + str(result.get("decode_tokens", DEFAULT_DECODE_TOKENS))
                 + " токенов):")
    for variant, item in result["variants"].items():
        lines.append(f"[{variant}]")
        for answer in item.get("answers") or []:
            text = " ".join(answer["text"].split())
            if len(text) > 220:
                text = text[:217] + "…"
            lines.append(f"  ? {answer['prompt']}")
            lines.append(f"  > {text if text else '(пусто)'}")
    lines.append("")
    three_b = result["three_b"]
    config = three_b["config"]
    reference = three_b["reference"]
    one_trit = three_b["sizes"]["trit_block"]["file_trit_embed"]
    dual_trit = three_b["sizes"]["dual_block"]
    int2 = three_b["sizes"]["int2_block"]
    lines.extend([
        "=== оценка 3B (аналитически, по config.json, веса не нужны) ===",
        f"Linear {config['linear_params']:,} параметров, таблица "
        f"{config['embed_params']:,}".replace(",", " "),
        "",
        "вид                 Linear, ГБ  бит/пар  файл+trit табл.  файл+int8 табл.",
    ])
    for kind, item in three_b["sizes"].items():
        lines.append(
            f"{kind:<18} {item['linear_bytes'] / 1e9:>9.3f}  "
            f"{item['bits_per_param']:>7.2f}  "
            f"{item['file_trit_embed'] / 1e9:>13.3f}  "
            f"{item['file_int8_embed'] / 1e9:>13.3f}"
        )
    lines.extend([
        "",
        f"ориентиры: Q4 GGUF 3B ≈ {reference['q4_gguf_bytes'] / 1e9:.1f} ГБ, "
        f"один трит = {reference['one_trit_bytes'] / 1e6:.0f} МБ "
        f"(наша оценка {one_trit / 1e6:.0f} МБ).",
        f"два трита на 3B: {dual_trit['file_trit_embed'] / 1e9:.2f} ГБ "
        f"(trit-таблица) / {dual_trit['file_int8_embed'] / 1e9:.2f} ГБ "
        f"(int8-таблица) — против {int2['file_trit_embed'] / 1e9:.2f} ГБ "
        f"у честного 4-уровневого int2 (2,0 бита).",
    ])
    return "\n".join(lines)


def dry_run(args: argparse.Namespace) -> int:
    """План и размеры без torch: проверка аргументов и арифметики."""
    variants = parse_variants(args.variants)
    shape = linear_shape_from_config(three_b_config())
    model_path = Path(args.model)
    print("=== dual_trit_eval: dry-run ===")
    print(f"модель: {args.model}"
          f"{'' if model_path.is_dir() else ' (локального каталога нет — прогон упадёт)'}")
    print(f"PPL: {args.ppl_data}, строк {args.ppl_lines}, "
          f"лимит {args.ppl_max_tokens} токенов (метод parity_cold)")
    print(f"варианты: {', '.join(variants)}")
    print(f"декод: {'нет' if args.no_decode else f'{len(DECODE_PROMPTS)} промпта × ' + str(args.decode_tokens) + ' токенов'}")
    print()
    print("оценка размера Qwen2.5-3B (Linear + таблица):")
    for kind in ("trit_block", "dual_row", "dual_block", "int2_block", "int8_row"):
        estimate = size_estimate(kind, shape.linear_params, shape.rows)
        print(
            f"  {kind:<12} Linear {estimate.total_bytes / 1e9:.3f} ГБ, "
            f"{estimate.bits_per_param:.2f} бит/параметр, файл с trit-таблицей "
            f"{file_bytes(estimate, 'trit_block', shape) / 1e9:.3f} ГБ"
        )
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Аргументы: по умолчанию — ровно режим 0143 (20 строк, 128 токенов)."""
    parser = argparse.ArgumentParser(
        description="Два трита на вес: PPL и размеры (задача 0148)",
    )
    parser.add_argument("--model", default=default_model(),
                        help="HF-модель (по умолчанию — Qwen2.5-0.5B из .hfcache)")
    parser.add_argument("--ppl-data", default=str(pc.DEFAULT_PPL_DATA))
    parser.add_argument("--ppl-lines", type=int, default=DEFAULT_PPL_LINES)
    parser.add_argument("--ppl-max-tokens", type=int, default=DEFAULT_PPL_MAX_TOKENS)
    parser.add_argument("--variants", default=",".join(ALL_VARIANTS),
                        help="через запятую: " + ", ".join(ALL_VARIANTS))
    parser.add_argument("--decode-tokens", type=int, default=DEFAULT_DECODE_TOKENS)
    parser.add_argument("--no-decode", action="store_true",
                        help="не декодировать ответы (быстрее)")
    parser.add_argument("--skip-ppl", action="store_true",
                        help="только декод: PPL не считать (нужен, когда "
                             "машина в свопе и 20 проходов не влезают)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quant-dtype", choices=("float32", "bfloat16"),
                        default="float32",
                        help="dtype весов квантованной модели (bf16 — вдвое "
                             "меньше RAM, но медленнее и грубее)")
    parser.add_argument("--json", default=str(DEFAULT_JSON),
                        help="куда писать JSON ('' — не писать)")
    parser.add_argument("--dry-run", action="store_true",
                        help="только план и размеры, без torch")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
    except SystemExit as exc:  # argparse уже напечатал ошибку
        return int(exc.code) if isinstance(exc.code, int) else 2
    try:
        if args.dry_run:
            return dry_run(args)
        return run(args)
    except DualTritError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n[dual_trit] прервано", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
