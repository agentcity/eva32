#!/usr/bin/env python3
"""Разреженный второй трит: точная поправка только там, где остаток велик (0149).

Идея
====

``W ≈ s1·A`` — один трит на вес, 3 уровня. Остаток ``r = W - s1·A`` не
исчезает: у весов около ``±s1/2`` он почти равен самому весу. Второй трит
``B`` поверх него (``W ≈ s1·A + s2·B``) возвращает 9 уровней, но стоит ещё
1,6 бита на **каждый** вес. Здесь проверяется компромисс: платить за ``B``
только у доли весов ``p``, где поправка нужна больше всего.

Два критерия отбора — и это главный содержательный вопрос задачи:

* ``residual`` — берём веса с наибольшим ``|r|``. Это ровно те места, где
  базовый трит ошибся сильнее всего; поправка адресная.
* ``magnitude`` — берём веса с наибольшим ``|W|``. Это классическая история
  про выбросы, но у **этой** схемы она может не сработать: после ``A``
  крупнейший по модулю вес блока уже представлен точно (``A = ±1``,
  ``s1 = |W|max``), и его остаток около нуля. То есть критерий по ``|W|``
  тратит поправку там, где ошибки почти нет. Числа показывают, так ли это.

Доля ``p`` фиксирована **внутри блока** 128 весов: ``k = round(p·128)``
(5 % → 6, 10 % → 13, 20 % → 26). Это не «доля от матрицы»: фиксированное
``k`` на блок делает работу декодера одинаковой во всех блоках и позволяет
считать индексы без счётчика на блок (см. «Размер»). Цена — фактическая
доля чуть отличается от номинала (6/128 = 4,69 %, 13/128 = 10,16 %,
26/128 = 20,31 %); в отчёте печатается фактическая.

Метод
=====

Только HF-сторона, без контейнера, как в 0143/0148: квантуются 7 матриц
слоя (q/k/v/o/gate/up/down), ``lm_head``/``embed_tokens`` не трогаются —
таблица словаря это отдельная ось (0143), иначе число не с чем сравнивать.
Разложение блока (``BLOCK_SIZE = 128``, как ``qat.ternary``):

1. ``s1 = absmax(block)``, ``A = clamp(round_half_away(W/s1), -1, 1)``;
2. ``r = W - s1·A``;
3. отбор ``k`` весов блока по ``|r|`` или по ``|W|`` (tie-break по позиции);
4. ``s2 = absmax(r по выбранным)``, ``B = clamp(round_half_away(r/s2), -1, 1)``
   — только у выбранных, у остальных ``B = 0`` и трит не хранится.

Шага наименьших квадратов, как в 0148, здесь нет намеренно: вопрос задачи —
не «сколько даёт двойной размер», а «сколько даёт точечная поправка при
малом размере». Лишняя свобода ``(s1, s2)`` смазала бы сравнение критериев.

Perplexity
==========

Способ ровно тот же, что в 0136/0143/0148: импортируются
``tools.parity_cold.read_text_pairs``, ``build_examples``, ``hf_answer_ppl``
и ``load_hf`` — «Вопрос: …\\nОтвет:», маска на промпт, 20 первых пар
``data/train.jsonl``, лимит 128 токенов (2047 токенов ответа). Ничего не
копируется: числа сравнимы с 0143 (bf16 = 16,13).

Размер
======

Считается по правилам формата TRT1 (``docs/trit-format.md``): 20 тритов в
32-битном слове (1,6 бита), хвост выравнивается блоками по 160 тритов =
32 байта; масштабы fp16 (2 байта). Три сегмента:

* **база** — триты всех весов + ``s1`` на блок 128;
* **остаток** — триты только выбранных, упакованы **единым потоком** матрицы
  (не по блоку: 26 тритов в блоке 160 — это 32 байта на блок, то есть
  +2 байта на вес, вся экономия съедается) + ``s2`` на блок;
* **индексы** — какая позиция в блоке выбрана. Два варианта:
  ``offsets`` — ``k`` смещений по 7 бит (0..127) на блок, без счётчика;
  ``bitmap`` — 1 бит на вес, маска на всю матрицу.
  Стоимость на вес: ``offsets`` = ``p·7`` бит, ``bitmap`` = 1 бит, поэтому
  ``offsets`` дешевле при ``p < 1/7 ≈ 14,3 %``, а на 20 % дешевле bitmap.
  ``--index auto`` (по умолчанию) берёт дешевле; для 5 % и 10 % это
  смещения, для 20 % — битовая карта.

Размер считается **по фактическим числам прогона** (число блоков, число
выбранных весов по каждой матрице, упаковка каждой матрицы отдельно), а не
по формуле «примерно». Оценка 3B — аналитическая по ``config.json``
(параметры Linear и блоки из конфига, веса не нужны).

Запуск
======

::

    PYTHONPATH=.pylibs python3.13 tools/sparse_trit_eval.py \\
        --ppl-lines 20 --ppl-max-tokens 128

    # только план и размеры, без torch
    python3 tools/sparse_trit_eval.py --dry-run

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

# Только чтение: способ замера PPL обязан совпадать с 0136/0143/0148.
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


#: Блок базы из ``qat.ternary`` (эталон 0143). И блок отбора второго трита.
BLOCK_SIZE = 128
#: Упаковка формата: 20 тритов в 32-битное слово.
TRITS_PER_WORD = 20
#: Блок хранения: 8 слов = 160 тритов = 32 байта.
TRITS_PER_STORAGE_BLOCK = 160
STORAGE_BLOCK_BYTES = 32
#: Масштабы в формате — fp16.
SCALE_BYTES = 2
#: Смещение внутри блока 128 — 7 бит.
OFFSET_BITS = 7
#: Битовая карта — 1 бит на вес.
BITMAP_BITS = 1

#: Критерии отбора: остаток ``|r|`` и магнитуда ``|W|``.
CRITERION_RESIDUAL = "residual"
CRITERION_MAGNITUDE = "magnitude"
SELECTION_CRITERIA = (CRITERION_RESIDUAL, CRITERION_MAGNITUDE)
CRITERION_TITLES = {
    CRITERION_RESIDUAL: "по |r| (остаток)",
    CRITERION_MAGNITUDE: "по |W| (магнитуда)",
}

#: Доли из задачи: 5 %, 10 %, 20 %.
DEFAULT_FRACTIONS = (0.05, 0.10, 0.20)

#: Схемы индексов.
INDEX_OFFSETS = "offsets"
INDEX_BITMAP = "bitmap"
INDEX_AUTO = "auto"
INDEX_KINDS = (INDEX_OFFSETS, INDEX_BITMAP)

#: Какие матрицы квантуются. ``lm_head``/``embed_tokens`` — не Linear-ось.
LINEAR_SUFFIXES = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
)

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
DEFAULT_JSON = ROOT / "data" / "sparse_trit" / "eval.json"
#: Два промпта для декода — те же, что в 0148 (факт и инструкция).
DECODE_PROMPTS = tuple(pc.DEFAULT_PROMPTS[:2])

#: Ориентиры из 0143 и задачи — печатаются рядом с нашими числами.
REFERENCE = {
    "bf16_ppl": 16.13,
    "trit1_ratio": 7.3e5,
    "int8_ratio": 1.001,
    "three_b_one_trit_bytes": 688 * 10 ** 6,
    "three_b_q4_gguf_bytes": 1740 * 10 ** 6,
}


class SparseTritError(RuntimeError):
    """Ошибка прогона, которую видно пользователю без трейсбека."""


# ---------------------------------------------------------------------------
# Чистая математика: блок весов (её проверяют тесты без torch)
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


def trit_of(value: float, scale: float) -> int:
    """Один вес в один трит при заданном масштабе (0 при нулевом масштабе)."""
    if scale <= 0.0:
        return 0
    return max(-1, min(1, round_half_away(value / scale)))


def selected_per_block(fraction: float, block_size: int = BLOCK_SIZE) -> int:
    """Сколько весов блока корректируется: ``k = round_half_away(p·B)``.

    Доля задаётся на блок, а не на матрицу: так у декодера одинаковая работа
    во всех блоках, а индекс не требует счётчика (``k`` известно из ``p``).
    ``fraction = 0`` — база без остатка (1 трит), ``1`` — оба трита у всех.
    """
    if not (0.0 <= float(fraction) <= 1.0):
        raise SparseTritError(f"доля вне [0, 1]: {fraction!r}")
    if block_size <= 0:
        raise SparseTritError(f"плохой размер блока: {block_size!r}")
    return max(0, min(int(block_size), round_half_away(float(fraction) * block_size)))


def actual_fraction(fraction: float, block_size: int = BLOCK_SIZE) -> float:
    """Фактическая доля с учётом округления ``k`` (6/128 = 4,69 %, не 5 %)."""
    return selected_per_block(fraction, block_size) / float(block_size)


@dataclass(frozen=True)
class BlockSparse:
    """Разложение матрицы (плоского списка) на базу + разреженный остаток.

    Всё по блокам ``block_size``: ``s1[i]``/``s2[i]`` — масштабы блока,
    ``a[i]``/``b[i]`` — триты блока, ``selected[i]`` — позиции, где ``b``
    хранится (остальные ``b`` равны нулю и в формате отсутствуют).
    """

    block_size: int
    fraction: float
    criterion: str
    k: int
    n_values: int
    s1: tuple[float, ...]
    a: tuple[tuple[int, ...], ...]
    s2: tuple[float, ...]
    b: tuple[tuple[int, ...], ...]
    selected: tuple[tuple[int, ...], ...]

    @property
    def blocks(self) -> int:
        return len(self.s1)

    @property
    def selected_total(self) -> int:
        return sum(len(item) for item in self.selected)

    @property
    def fraction_actual(self) -> float:
        return self.selected_total / float(self.n_values) if self.n_values else 0.0


def _block_scale(values: Sequence[float]) -> float:
    return max((abs(float(value)) for value in values), default=0.0)


def sparse_block_decompose(
    values: Sequence[float],
    *,
    fraction: float,
    criterion: str,
    block_size: int = BLOCK_SIZE,
) -> BlockSparse:
    """Эталон разложения на чистом Python (torch-путь повторяет его).

    Хвост добивается нулями до целого блока — нули не влияют ни на ``s1``,
    ни на отбор (``|r| = 0``), а ``n_values`` хранит настоящую длину, чтобы
    восстановление не тянуло padding.
    """
    if criterion not in SELECTION_CRITERIA:
        raise SparseTritError(
            f"неизвестный критерий {criterion!r}; допустимы: "
            f"{', '.join(SELECTION_CRITERIA)}"
        )
    vals = [float(value) for value in values]
    if not vals:
        return BlockSparse(int(block_size), float(fraction), criterion,
                           selected_per_block(fraction, block_size), 0,
                           (), (), (), (), ())
    k = selected_per_block(fraction, block_size)
    padded = list(vals)
    pad = (-len(padded)) % block_size
    if pad:
        padded.extend([0.0] * pad)
    s1_all: list[float] = []
    a_all: list[tuple[int, ...]] = []
    s2_all: list[float] = []
    b_all: list[tuple[int, ...]] = []
    sel_all: list[tuple[int, ...]] = []
    for start in range(0, len(padded), block_size):
        block = padded[start:start + block_size]
        scale1 = _block_scale(block)
        a = [trit_of(value, scale1) for value in block]
        remainder = [value - scale1 * trit for value, trit in zip(block, a)]
        key = remainder if criterion == CRITERION_RESIDUAL else block
        # tie-break по позиции: результат не зависит от порядка равных по модулю.
        order = sorted(range(len(block)), key=lambda i: (-abs(key[i]), i))[:k]
        selected = tuple(sorted(order))
        scale2 = max((abs(remainder[i]) for i in selected), default=0.0)
        b = [0] * len(block)
        for i in selected:
            b[i] = trit_of(remainder[i], scale2)
        s1_all.append(scale1)
        a_all.append(tuple(a))
        s2_all.append(scale2)
        b_all.append(tuple(b))
        sel_all.append(selected)
    return BlockSparse(int(block_size), float(fraction), criterion, k,
                       len(vals), tuple(s1_all), tuple(a_all), tuple(s2_all),
                       tuple(b_all), tuple(sel_all))


def reconstruct_sparse(decomposition: BlockSparse) -> list[float]:
    """``W ≈ s1·A + s2·B`` (вторая часть только у выбранных позиций)."""
    out: list[float] = []
    for scale1, a, scale2, b, selected in zip(
        decomposition.s1, decomposition.a, decomposition.s2,
        decomposition.b, decomposition.selected,
    ):
        chosen = set(selected)
        for position, trit in enumerate(a):
            value = scale1 * float(trit)
            if position in chosen:
                value += scale2 * float(b[position])
            out.append(value)
    return out[:decomposition.n_values]


def relative_error(values: Sequence[float], approx: Sequence[float]) -> float:
    """``||w - w_hat|| / ||w||`` (0 при нулевом весе)."""
    numerator = math.fsum(
        (float(value) - float(other)) ** 2 for value, other in zip(values, approx)
    )
    denominator = math.fsum(float(value) ** 2 for value in values)
    if denominator <= 0.0:
        return 0.0
    return math.sqrt(numerator / denominator)


def residual_share(values: Sequence[float], decomposition: BlockSparse) -> float:
    """Какую долю энергии ошибки базы забирает разреженный остаток (0..1).

    Считается по блокам: ``||s2·B||²`` против ``||r||²`` до правки. Число
    показывает, честно ли критерий бьёт в цель: для ``residual`` оно должно
    быть много больше ``p``, для ``magnitude`` — около ``p·(что попало)``.
    """
    vals = [float(value) for value in values]
    if not vals:
        return 0.0
    padded = list(vals)
    pad = (-len(padded)) % decomposition.block_size
    if pad:
        padded.extend([0.0] * pad)
    residual_sq = 0.0
    fixed_sq = 0.0
    step = decomposition.block_size
    for index, start in enumerate(range(0, len(padded), step)):
        block = padded[start:start + step]
        scale1 = decomposition.s1[index]
        a = decomposition.a[index]
        remainder = [value - scale1 * trit for value, trit in zip(block, a)]
        residual_sq += math.fsum(value * value for value in remainder)
        chosen = set(decomposition.selected[index])
        for position in chosen:
            fixed_sq += (decomposition.s2[index] * float(decomposition.b[index][position])) ** 2
    if residual_sq <= 0.0:
        return 0.0
    return fixed_sq / residual_sq


# ---------------------------------------------------------------------------
# Размеры по правилам формата (без torch: чистые числа)
# ---------------------------------------------------------------------------


def packed_trit_bytes(n_values: int) -> int:
    """Байты под триты в формате TRT1: 160 тритов = 32 байта, хвост добит."""
    if n_values <= 0:
        return 0
    blocks = (int(n_values) + TRITS_PER_STORAGE_BLOCK - 1) // TRITS_PER_STORAGE_BLOCK
    return blocks * STORAGE_BLOCK_BYTES


def scale_block_count(n_values: int, block_size: int = BLOCK_SIZE) -> int:
    """Число блоков масштаба 128 (как в ``qat.ternary``)."""
    if n_values <= 0:
        return 0
    return (int(n_values) + int(block_size) - 1) // int(block_size)


def offsets_bytes(selected: int) -> int:
    """Смещения по 7 бит каждое, упакованы подряд (без выравнивания по блоку)."""
    if selected <= 0:
        return 0
    return (int(selected) * OFFSET_BITS + 7) // 8


def bitmap_bytes(params: int) -> int:
    """Битовая карта: 1 бит на вес, вся матрица."""
    if params <= 0:
        return 0
    return (int(params) * BITMAP_BITS + 7) // 8


def index_bytes_for(index_kind: str, params: int, selected: int) -> int:
    """Байты индексов выбранной схемой."""
    if index_kind == INDEX_OFFSETS:
        return offsets_bytes(selected)
    if index_kind == INDEX_BITMAP:
        return bitmap_bytes(params)
    raise SparseTritError(f"неизвестная схема индексов: {index_kind!r}")


def choose_index_kind(fraction: float, block_size: int = BLOCK_SIZE) -> str:
    """Дешевле смещения или карта: ``p·7`` бит против 1 бита на вес.

    Порог — 1/7 ≈ 14,3 %: на 5/10 % смещения дешевле, на 20 % — карта.
    """
    per_weight_bits = actual_fraction(fraction, block_size) * OFFSET_BITS
    return INDEX_OFFSETS if per_weight_bits <= BITMAP_BITS else INDEX_BITMAP


@dataclass(frozen=True)
class SizeBreakdown:
    """Размер представления: сегменты по отдельности и итог."""

    kind: str
    index_kind: str
    params: int
    blocks: int
    selected: int
    base_trit_bytes: int
    base_scale_bytes: int
    second_trit_bytes: int
    second_scale_bytes: int
    index_bytes: int

    @property
    def total_bytes(self) -> int:
        return (
            int(self.base_trit_bytes) + int(self.base_scale_bytes)
            + int(self.second_trit_bytes) + int(self.second_scale_bytes)
            + int(self.index_bytes)
        )

    @property
    def bits_per_param(self) -> float:
        if self.params <= 0:
            return 0.0
        return 8.0 * self.total_bytes / self.params

    @property
    def fraction(self) -> float:
        return self.selected / float(self.params) if self.params else 0.0


def sum_breakdowns(
    parts: Sequence[SizeBreakdown], *, kind: str, index_kind: str
) -> SizeBreakdown:
    """Сложить размеры матриц: файл — это сумма сегментов, а не среднее."""
    return SizeBreakdown(
        kind=kind,
        index_kind=index_kind,
        params=sum(int(part.params) for part in parts),
        blocks=sum(int(part.blocks) for part in parts),
        selected=sum(int(part.selected) for part in parts),
        base_trit_bytes=sum(int(part.base_trit_bytes) for part in parts),
        base_scale_bytes=sum(int(part.base_scale_bytes) for part in parts),
        second_trit_bytes=sum(int(part.second_trit_bytes) for part in parts),
        second_scale_bytes=sum(int(part.second_scale_bytes) for part in parts),
        index_bytes=sum(int(part.index_bytes) for part in parts),
    )


#: Сколько сегментов у представления: нужно и для отчёта, и для тестов.
SEGMENT_FIELDS = (
    "base_trit_bytes", "base_scale_bytes", "second_trit_bytes",
    "second_scale_bytes", "index_bytes",
)


def sparse_breakdown(
    params: int,
    fraction: float,
    *,
    block_size: int = BLOCK_SIZE,
    index: str = INDEX_AUTO,
    blocks: int | None = None,
) -> SizeBreakdown:
    """Размер матрицы (или суммы матриц) в схеме «база + разреженный остаток».

    ``blocks`` можно передать явно — тогда формула не пересчитывает блоки
    из ``params`` (у Qwen2.5 все ``cols`` кратны 128, поэтому совпадает).
    """
    params = int(params)
    if params <= 0:
        return SizeBreakdown("sparse", index, 0, 0, 0, 0, 0, 0, 0, 0)
    k = selected_per_block(fraction, block_size)
    blocks = scale_block_count(params, block_size) if blocks is None else int(blocks)
    selected = blocks * k
    if index == INDEX_AUTO:
        index = choose_index_kind(fraction, block_size)
    if index not in INDEX_KINDS:
        raise SparseTritError(f"неизвестная схема индексов: {index!r}")
    # Индексы не нужны, если выбраны все веса (k = B): второй трит есть везде.
    index_bytes = 0 if k >= block_size else index_bytes_for(index, params, selected)
    return SizeBreakdown(
        kind="sparse",
        index_kind=index,
        params=params,
        blocks=blocks,
        selected=selected,
        base_trit_bytes=packed_trit_bytes(params),
        base_scale_bytes=blocks * SCALE_BYTES,
        second_trit_bytes=packed_trit_bytes(selected),
        second_scale_bytes=blocks * SCALE_BYTES if k > 0 else 0,
        index_bytes=index_bytes,
    )


def trit_breakdown(params: int, *, block_size: int = BLOCK_SIZE,
                   blocks: int | None = None) -> SizeBreakdown:
    """Один трит: база без остатка (эталон 0143)."""
    return sparse_breakdown(params, 0.0, block_size=block_size, index=INDEX_OFFSETS,
                            blocks=blocks)


def dual_breakdown(params: int, *, block_size: int = BLOCK_SIZE,
                   blocks: int | None = None) -> SizeBreakdown:
    """Два трита у всех весов: верхняя граница схемы (p = 100 %)."""
    return sparse_breakdown(params, 1.0, block_size=block_size,
                            index=INDEX_OFFSETS, blocks=blocks)


# ---------------------------------------------------------------------------
# Формы модели (без загрузки весов: считает размеры и проверяет сходимость)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LinearShape:
    """Форма «Linear-части» модели — те же 7 матриц, что квантуются."""

    hidden: int
    intermediate: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int
    tied: bool

    @property
    def columns(self) -> tuple[tuple[str, int, int], ...]:
        """``(имя, строки, столбцы)`` семи матриц одного слоя (порядок Qwen2)."""
        kv = self.kv_heads * self.head_dim
        return (
            ("q_proj", self.hidden, self.hidden),
            ("k_proj", kv, self.hidden),
            ("v_proj", kv, self.hidden),
            # o_proj принимает выход внимания: heads·head_dim (у Qwen2.5 это
            # ровно hidden), а не kv_heads·head_dim — тут легко ошибиться.
            ("o_proj", self.hidden, self.heads * self.head_dim),
            ("gate_proj", self.intermediate, self.hidden),
            ("up_proj", self.intermediate, self.hidden),
            ("down_proj", self.hidden, self.intermediate),
        )

    @property
    def rows(self) -> int:
        """Строк (групп масштаба посрочных режимов) во всех Linear."""
        return sum(rows for _name, rows, _cols in self.columns) * self.layers

    @property
    def params_per_layer(self) -> int:
        return sum(rows * cols for _name, rows, cols in self.columns)

    @property
    def blocks_per_layer(self) -> int:
        """Блоки по 128: ``rows·ceil(cols/128)`` на матрицу."""
        return sum(
            rows * ((cols + BLOCK_SIZE - 1) // BLOCK_SIZE)
            for _name, rows, cols in self.columns
        )

    @property
    def linear_params(self) -> int:
        return self.params_per_layer * self.layers

    @property
    def linear_blocks(self) -> int:
        return self.blocks_per_layer * self.layers

    @property
    def embed_params(self) -> int:
        """Таблица словаря; при ``tie_word_embeddings`` она же и lm_head."""
        return self.vocab * self.hidden

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
        raise SparseTritError("в конфиге не хватает размеров модели")
    return LinearShape(hidden, intermediate, layers, heads, kv_heads,
                       int(head_dim), vocab, tied)


def shape_from_model(model: Any) -> LinearShape:
    """Форма по фактическим модулям — чтобы отчёт не зависел от догадок."""
    shape = linear_shape_from_config(model.config)
    measured = sum(int(module.weight.numel()) for _n, module in linear_targets(model))
    if measured != shape.linear_params:
        raise SparseTritError(
            f"конфиг обещает {shape.linear_params} параметров Linear, "
            f"а в модели {measured} — формула не сходится"
        )
    return shape


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
        raise SparseTritError(
            "нужен torch (.pylibs): PYTHONPATH=.pylibs python3.13 "
            "tools/sparse_trit_eval.py"
        ) from exc
    return torch


def _round_half_away_torch(tensor: Any) -> Any:
    torch = _torch()
    return torch.sign(tensor) * torch.floor(tensor.abs() + 0.5)


def _flatten_blocks(weight: Any, block_size: int | None) -> tuple[Any, tuple[int, ...]]:
    """Привести вес к ``[блоков, block_size]`` (или к строкам при ``None``)."""
    torch = _torch()
    original = tuple(weight.shape)
    if weight.ndim != 2:
        raise SparseTritError(f"ожидалась 2D-матрица Linear, получено {original}")
    rows, cols = int(weight.shape[0]), int(weight.shape[1])
    if block_size is None:
        return weight.reshape(rows, cols), original
    blocks = (cols + block_size - 1) // block_size
    pad = blocks * block_size - cols
    flat = weight
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    return flat.reshape(rows * blocks, block_size), original


def _restore_blocks(flat: Any, original: tuple[int, ...]) -> Any:
    """Обратный reshape после :func:`_flatten_blocks` (без padding'а)."""
    rows, cols = original
    return flat.reshape(rows, -1)[:, :cols].reshape(original)


@dataclass
class WeightStats:
    """Сколько стоила одна матрица: ошибка, блоки, выбранные веса.

    ``residual_energy`` / ``fixed_energy`` — энергия остатка базы и энергия
    поправки ``s2·B``: их отношение показывает, насколько адресно критерий
    бьёт в ошибку (предел — ``p``, если бы остаток был равномерным).
    """

    relative_error: float
    blocks: int = 0
    selected: int = 0
    k: int = 0
    residual_energy: float = 0.0
    fixed_energy: float = 0.0


def sparse_trit_weight(
    weight: Any,
    fraction: float,
    criterion: str,
    block_size: int = BLOCK_SIZE,
) -> tuple[Any, WeightStats]:
    """База + разреженный второй трит для одной матрицы.

    Повторяет :func:`sparse_block_decompose` построчно-по-блочно: ``s1`` —
    absmax блока, ``A`` — трит, ``r`` — остаток, отбор ``k`` весов по
    ``|r|``/``|W|``, ``s2`` — absmax остатка по выбранным, ``B`` — трит.
    """
    torch = _torch()
    if criterion not in SELECTION_CRITERIA:
        raise SparseTritError(
            f"неизвестный критерий {criterion!r}; допустимы: "
            f"{', '.join(SELECTION_CRITERIA)}"
        )
    k = selected_per_block(fraction, block_size)
    with torch.no_grad():
        w = weight.detach().float()
        flat, original = _flatten_blocks(w, block_size)
        blocks = int(flat.shape[0])
        s1 = flat.abs().amax(dim=1)
        safe1 = torch.where(s1 > 0, s1, torch.ones_like(s1))
        a = _round_half_away_torch(flat / safe1[:, None]).clamp(-1.0, 1.0)
        a = torch.where((s1 > 0)[:, None], a, torch.zeros_like(a))
        remainder = flat - s1[:, None] * a
        selected = 0
        if k > 0:
            key = remainder.abs() if criterion == CRITERION_RESIDUAL else flat.abs()
            indices = key.topk(k, dim=1, largest=True, sorted=False).indices
            mask = torch.zeros_like(flat, dtype=torch.bool)
            mask.scatter_(1, indices, True)
            selected = int(mask.sum())
            masked = torch.where(mask, remainder.abs(), torch.zeros_like(remainder))
        else:
            mask = torch.zeros_like(flat, dtype=torch.bool)
            masked = torch.zeros_like(remainder)
        s2 = masked.amax(dim=1)
        safe2 = torch.where(s2 > 0, s2, torch.ones_like(s2))
        b = _round_half_away_torch(remainder / safe2[:, None]).clamp(-1.0, 1.0)
        # Второй трит существует только у выбранных; у остальных ровно 0.
        b = torch.where(mask & (s2 > 0)[:, None], b, torch.zeros_like(b))
        correction = s2[:, None] * b
        residual_energy = float((remainder * remainder).sum())
        fixed_energy = float((correction * correction).sum())
        deq = _restore_blocks(s1[:, None] * a + correction, original)
        error = _frobenius_error(w, deq)
        del flat, a, remainder, b, mask, correction
    return deq, WeightStats(relative_error=error, blocks=blocks,
                            selected=selected, k=k,
                            residual_energy=residual_energy,
                            fixed_energy=fixed_energy)


def _frobenius_error(reference: Any, approx: Any) -> float:
    """``||w - w_hat||_F / ||w||_F`` на тензорах."""
    torch = _torch()
    numerator = torch.linalg.vector_norm((reference - approx).double())
    denominator = torch.linalg.vector_norm(reference.double())
    if float(denominator) <= 0.0:
        return 0.0
    return float(numerator / denominator)


# ---------------------------------------------------------------------------
# Варианты прогона
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Variant:
    """Что меряем: ``bf16``, ``trit1`` (база) или ``sparse`` (доля + критерий)."""

    name: str
    fraction: float = 0.0
    criterion: str = ""

    @property
    def is_sparse(self) -> bool:
        return self.name == "sparse"

    @property
    def key(self) -> str:
        if self.name != "sparse":
            return self.name
        return f"{self.criterion}_{_percent(self.fraction)}"

    @property
    def title(self) -> str:
        if self.name == "bf16":
            return "bf16 (эталон)"
        if self.name == "trit1":
            return "1 трит (база, 0 %)"
        return f"{self.criterion} {_percent(self.fraction)} %"

    @property
    def needs_quant(self) -> bool:
        return self.name != "bf16"


def _percent(fraction: float) -> str:
    """5 % / 10 % / 20 % — без хвоста «.0»."""
    value = round(float(fraction) * 100.0, 6)
    return f"{value:g}"


def build_variants(
    fractions: Sequence[float],
    criteria: Sequence[str],
    *,
    baselines: bool = True,
) -> list[Variant]:
    """Список вариантов: bf16, база, затем доли × критерии."""
    for criterion in criteria:
        if criterion not in SELECTION_CRITERIA:
            raise SparseTritError(
                f"неизвестный критерий {criterion!r}; допустимы: "
                f"{', '.join(SELECTION_CRITERIA)}"
            )
    variants: list[Variant] = []
    if baselines:
        variants.extend([Variant("bf16"), Variant("trit1")])
    for fraction in fractions:
        for criterion in criteria:
            variants.append(Variant("sparse", float(fraction), criterion))
    if not variants:
        raise SparseTritError("пустой список вариантов")
    return variants


def parse_fractions(text: str) -> list[float]:
    """Разобрать ``--fractions``: 0.05 — доля, 5 — проценты; дубли отброшены."""
    fractions: list[float] = []
    for raw in str(text).split(","):
        item = raw.strip()
        if not item:
            continue
        try:
            value = float(item)
        except ValueError as exc:
            raise SparseTritError(f"плохая доля {item!r}") from exc
        if value > 1.0:
            value /= 100.0
        if not (0.0 < value <= 1.0):
            raise SparseTritError(f"доля вне (0, 1]: {item!r}")
        if value not in fractions:
            fractions.append(value)
    if not fractions:
        raise SparseTritError("пустой список долей")
    return fractions


def parse_criteria(text: str) -> list[str]:
    """Разобрать ``--criteria``, сохранив порядок и отбросив дубли."""
    criteria: list[str] = []
    for raw in str(text).split(","):
        item = raw.strip()
        if not item:
            continue
        if item not in SELECTION_CRITERIA:
            raise SparseTritError(
                f"неизвестный критерий {item!r}; допустимы: "
                f"{', '.join(SELECTION_CRITERIA)}"
            )
        if item not in criteria:
            criteria.append(item)
    if not criteria:
        raise SparseTritError("пустой список критериев")
    return criteria


# ---------------------------------------------------------------------------
# Квантование модели и фактический размер
# ---------------------------------------------------------------------------


def quantize_model(model: Any, variant: Variant) -> dict[str, Any]:
    """Заменить веса целевых Linear и посчитать **фактический** размер.

    Размер считается по каждой матрице отдельно: упаковка тритов выравнивается
    хвостом, поэтому сумма упакованных матриц больше «формулы по параметрам».
    Модель меняется **на месте**.
    """
    if variant.name == "bf16":
        return _empty_quant(variant, "bf16")
    targets = linear_targets(model)
    if not targets:
        raise SparseTritError("в модели не нашлось ни одной q/k/v/o/gate/up/down матрицы")
    started = time.monotonic()
    fraction = variant.fraction if variant.is_sparse else 0.0
    criterion = variant.criterion if variant.is_sparse else CRITERION_RESIDUAL
    params = 0
    error_num = 0.0
    error_den = 0.0
    residual_energy = 0.0
    fixed_energy = 0.0
    parts: list[SizeBreakdown] = []
    for _name, module in targets:
        weight = module.weight.detach()
        # Квадрат нормы исходного веса — до copy_: weight делит хранилище
        # с module.weight, и после подмены он уже квантованный.
        original_sq = float(weight.detach().double().pow(2).sum())
        count = int(weight.numel())
        deq, stats = sparse_trit_weight(weight, fraction, criterion, BLOCK_SIZE)
        with _torch().no_grad():
            module.weight.copy_(deq.reshape(module.weight.shape).to(module.weight.dtype))
        params += count
        error_num += (stats.relative_error ** 2) * original_sq
        error_den += original_sq
        residual_energy += stats.residual_energy
        fixed_energy += stats.fixed_energy
        k = stats.k
        # Схема индексов выбирается по доле; для k >= B индексы не нужны.
        index_kind = choose_index_kind(fraction, BLOCK_SIZE) if k > 0 else INDEX_OFFSETS
        index_bytes = 0 if k >= BLOCK_SIZE else index_bytes_for(
            index_kind, count, stats.selected)
        parts.append(SizeBreakdown(
            kind=variant.key,
            index_kind=index_kind,
            params=count,
            blocks=stats.blocks,
            selected=stats.selected,
            base_trit_bytes=packed_trit_bytes(count),
            base_scale_bytes=stats.blocks * SCALE_BYTES,
            second_trit_bytes=packed_trit_bytes(stats.selected),
            second_scale_bytes=stats.blocks * SCALE_BYTES if k > 0 else 0,
            index_bytes=index_bytes,
        ))
        del deq, stats
    index_kind = choose_index_kind(fraction, BLOCK_SIZE) if fraction > 0 else INDEX_OFFSETS
    total = sum_breakdowns(parts, kind=variant.key, index_kind=index_kind)
    bitmap_total = SizeBreakdown(
        kind="bitmap", index_kind=INDEX_BITMAP, params=total.params,
        blocks=total.blocks, selected=total.selected,
        base_trit_bytes=total.base_trit_bytes,
        base_scale_bytes=total.base_scale_bytes,
        second_trit_bytes=total.second_trit_bytes,
        second_scale_bytes=total.second_scale_bytes,
        index_bytes=0,
    )
    relative = math.sqrt(error_num / error_den) if error_den > 0 else 0.0
    # Индексов нет, когда второй трит не хранится вовсе (база): обе «схемы»
    # вырождаются в 0 байт, и это надо показать честно, а не 1 бит на вес.
    overrides_bitmap = bitmap_bytes(total.params) if k > 0 else 0
    overrides_offsets = offsets_bytes(total.selected) if k > 0 else 0
    return {
        "variant": variant.key,
        "name": variant.name,
        "fraction": fraction,
        "criterion": criterion,
        "k": selected_per_block(fraction, BLOCK_SIZE) if fraction > 0 else 0,
        "modules": len(targets),
        "params": params,
        "blocks": total.blocks,
        "selected": total.selected,
        "fraction_actual": total.fraction,
        "relative_error": relative,
        "residual_energy": residual_energy,
        "fixed_energy": fixed_energy,
        "fixed_share": (
            fixed_energy / residual_energy if residual_energy > 0.0 else 0.0
        ),
        "seconds": time.monotonic() - started,
        "size": _breakdown_dict(total),
        # Альтернативная схема индексов — чтобы отчёт мог сравнить честно.
        "size_bitmap": _breakdown_dict(bitmap_total, index_override=overrides_bitmap),
        "size_offsets": _breakdown_dict(
            bitmap_total, index_override=overrides_offsets),
    }


def _empty_quant(variant: Variant, kind: str) -> dict[str, Any]:
    return {
        "variant": variant.key, "name": variant.name, "fraction": 0.0,
        "criterion": "", "k": 0, "modules": 0, "params": 0, "blocks": 0,
        "selected": 0, "fraction_actual": 0.0, "relative_error": 0.0,
        "residual_energy": 0.0, "fixed_energy": 0.0, "fixed_share": 0.0,
        "seconds": 0.0, "size": None, "size_bitmap": None, "size_offsets": None,
        "kind": kind,
    }


def _breakdown_dict(
    breakdown: SizeBreakdown, *, index_override: int | None = None
) -> dict[str, Any]:
    """Разрез размера: сегменты по отдельности (JSON-совместимо)."""
    index_bytes = (
        int(breakdown.index_bytes) if index_override is None else int(index_override)
    )
    total = breakdown.total_bytes - breakdown.index_bytes + index_bytes
    params = int(breakdown.params)
    return {
        "index_kind": breakdown.index_kind,
        "params": params,
        "blocks": int(breakdown.blocks),
        "selected": int(breakdown.selected),
        "base_trit_bytes": int(breakdown.base_trit_bytes),
        "base_scale_bytes": int(breakdown.base_scale_bytes),
        "second_trit_bytes": int(breakdown.second_trit_bytes),
        "second_scale_bytes": int(breakdown.second_scale_bytes),
        "index_bytes": index_bytes,
        "total_bytes": total,
        "bits_per_param": 8.0 * total / params if params else 0.0,
    }


def table_bytes(params: int, kind: str = "trit") -> int:
    """Байты таблицы словаря: ``trit`` (эталон 0143) или ``bf16``."""
    params = int(params)
    if params <= 0:
        return 0
    if kind == "trit":
        return trit_breakdown(params).total_bytes
    if kind in ("bf16", "fp16"):
        return params * 2
    raise SparseTritError(f"неизвестный вид таблицы: {kind!r}")


# ---------------------------------------------------------------------------
# PPL и декод
# ---------------------------------------------------------------------------


def build_ppl_examples(
    tokenizer: Any, args: argparse.Namespace, vocab: int
) -> list[dict[str, Any]]:
    """Примеры PPL — ровно через ``tools.parity_cold`` (числа сравнимы с 0143)."""
    pairs = pc.read_text_pairs(args.ppl_data, args.ppl_lines)
    examples, dropped = pc.build_examples(
        tokenizer, pairs, max_tokens=args.ppl_max_tokens, vocab=vocab
    )
    if not examples:
        raise SparseTritError("perplexity: после обрезки не осталось пар")
    if dropped:
        print(f"[sparse_trit] PPL: отброшено {dropped} пар", file=sys.stderr)
    return examples


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


# ---------------------------------------------------------------------------
# Оценка 3B
# ---------------------------------------------------------------------------


def three_b_report(
    config: Any,
    fractions: Sequence[float],
    criteria: Sequence[str],
) -> dict[str, Any]:
    """Аналитическая оценка 3B: база, доли, два трита, Q4 — по ``config.json``.

    ``criteria`` принимается для симметрии с прогоном, но на размер не влияет:
    число выбранных весов ``k`` задаётся только долей.
    """
    shape = linear_shape_from_config(config)
    blocks = shape.linear_blocks
    entries: dict[str, Any] = {}

    def add(kind: str, title: str, breakdown: SizeBreakdown) -> None:
        entries[kind] = {
            "title": title,
            "linear_bytes": breakdown.total_bytes,
            "bits_per_param": breakdown.bits_per_param,
            "selected": breakdown.selected,
            "index_kind": breakdown.index_kind,
            "index_bytes": breakdown.index_bytes,
            "file_trit_bytes": breakdown.total_bytes + table_bytes(shape.embed_params, "trit"),
            "file_bf16_bytes": breakdown.total_bytes + table_bytes(shape.embed_params, "bf16"),
        }

    add("trit1", "1 трит (база)", trit_breakdown(shape.linear_params, blocks=blocks))
    for fraction in fractions:
        add(
            f"sparse_{_percent(fraction)}",
            f"sparse {_percent(fraction)} %",
            sparse_breakdown(shape.linear_params, fraction, blocks=blocks),
        )
    add("dual", "два трита (100 %)", dual_breakdown(shape.linear_params, blocks=blocks))
    return {
        "config": {
            "hidden": shape.hidden,
            "intermediate": shape.intermediate,
            "layers": shape.layers,
            "linear_params": shape.linear_params,
            "linear_blocks": blocks,
            "embed_params": shape.embed_params,
            "tied": shape.tied,
        },
        "entries": entries,
        "reference": dict(REFERENCE),
    }


# ---------------------------------------------------------------------------
# Отчёт
# ---------------------------------------------------------------------------


def _mb(value: float) -> str:
    return f"{value / 1e6:,.1f}".replace(",", " ")


def _gi(value: float) -> str:
    return f"{value / 1e9:,.3f}".replace(",", " ")


def _ratio_text(ratio: float | None) -> str:
    """Кратко, но без потери порядка: ×7,3e+05 читается лучше «730000»."""
    if ratio is None:
        return "—"
    if ratio < 1e3:
        return f"×{ratio:.3f}"
    if ratio < 1e5:
        return f"×{ratio:,.0f}".replace(",", " ")
    return f"×{ratio:.3g}"


def _ppl_text(value: float | None) -> str:
    if value is None:
        return "—"
    if value < 1e5:
        return f"{value:.2f}"
    return f"{value:.3g}"


def best_variant(results: dict[str, Any]) -> str | None:
    """Лучший (наименьший PPL) разреженный вариант; ``trit1`` — не кандидат."""
    best_key: str | None = None
    best_ppl = math.inf
    for key, item in results.items():
        if not item.get("fraction"):
            continue
        value = item.get("ppl")
        if value is None or not math.isfinite(value):
            continue
        if value < best_ppl:
            best_ppl = float(value)
            best_key = key
    return best_key


def verdict_lines(result: dict[str, Any]) -> list[str]:
    """Вывод по числам: помогает ли остаток, насколько, и чего это стоит."""
    results = result["variants"]
    base = results.get("trit1", {}).get("ppl")
    bf16 = results.get("bf16", {}).get("ppl")
    best_key = result.get("best_variant")
    lines: list[str] = []
    if best_key is None:
        return ["вывод: ни один разреженный вариант не дал конечного PPL — "
                "сравнивать нечего"]
    best = results[best_key]
    best_ppl = best.get("ppl")
    if base and best_ppl:
        gain = float(base) / float(best_ppl)
        lines.append(
            f"лучший разреженный вариант: {best['title']}, PPL {_ppl_text(best_ppl)} "
            f"против {_ppl_text(base)} у базы — выигрыш {gain:.2f}×"
        )
        if bf16 and best_ppl:
            lines.append(
                f"до bf16 ({_ppl_text(bf16)}) остаётся {float(best_ppl) / float(bf16):.3g}×: "
                "разреженный остаток сам по себе модель не спасает"
                if best_ppl > bf16 * 2 else
                f"PPL {_ppl_text(best_ppl)} при bf16 {_ppl_text(bf16)} — "
                f"{float(best_ppl) / float(bf16):.3g}×, то есть сопоставимо"
            )
    # Сравнение критериев при одной доле: главный содержательный вопрос.
    for fraction in sorted({float(item["fraction"]) for item in results.values()
                            if item.get("fraction")}):
        keys = {item["criterion"]: key for key, item in results.items()
                if float(item["fraction"]) == fraction}
        if CRITERION_RESIDUAL in keys and CRITERION_MAGNITUDE in keys:
            residual = results[keys[CRITERION_RESIDUAL]].get("ppl")
            magnitude = results[keys[CRITERION_MAGNITUDE]].get("ppl")
            if residual and magnitude:
                lines.append(
                    f"{_percent(fraction)} %: остаток {_ppl_text(residual)} против "
                    f"магнитуды {_ppl_text(magnitude)} — "
                    + ("остаток лучше" if residual < magnitude else "магнитуда лучше")
                    + f" в {max(residual, magnitude) / min(residual, magnitude):.2f}×"
                )
    # Цена: сколько стоит каждый процент.
    size = best.get("size") or {}
    if size:
        base_size = result.get("base_size") or {}
        if base_size:
            lines.append(
                f"цена: Linear {_mb(size['total_bytes'])} МБ против "
                f"{_mb(base_size['total_bytes'])} МБ у базы — "
                f"+{100.0 * (size['total_bytes'] / base_size['total_bytes'] - 1.0):.0f} % "
                f"({size['bits_per_param']:.2f} бит/параметр)"
            )
    return lines


def format_report(result: dict[str, Any]) -> str:
    """Текстовая таблица: доля × критерий → PPL → фактический размер."""
    ppl = result["ppl"]
    shape = result["shape"]
    lines = [
        "=== 0149: база-трит + разреженный второй трит (Qwen2.5-0.5B) ===",
        f"модель: {result['model']}",
        f"PPL: {ppl['data']}, строк {ppl['lines']}, токенов ответа "
        f"{ppl['tokens']}, лимит {ppl['max_tokens']}",
        f"метод: {ppl['method']} (как 0136/0143; bf16 = "
        f"{REFERENCE['bf16_ppl']:.2f})",
        f"Linear: {shape['linear_params']} параметров, {shape['linear_blocks']} "
        f"блоков по {BLOCK_SIZE}; таблица {shape['embed_params']} (bf16, "
        f"не квантуется) — числа с пробелами-разделителями ниже",
        f"пик RSS: {result['peak_rss_mb']:.0f} МБ, "
        f"время {result['timings']['total_seconds']:.0f} с",
        "",
        "доля   критерий          k/блок  факт.%       PPL    ×bf16    "
        "err  снято err   Linear МБ  бит/пар  индекс МБ",
    ]
    for key, item in result["variants"].items():
        title = item["title"]
        if item["name"] == "bf16":
            fraction = "—"
            k_text = "—"
            actual = "—"
            size = None
            err = "0"
            fixed = "—"
        elif item["name"] == "trit1":
            fraction = "0 %"
            k_text = "0"
            actual = "0.00"
            size = item.get("size")
            err = f"{item['quant']['relative_error']:.3g}"
            fixed = "0.0"
        else:
            fraction = f"{_percent(item['fraction'])} %"
            k_text = str(item["quant"]["k"])
            actual = f"{100.0 * item['quant']['fraction_actual']:.2f}"
            size = item.get("size")
            err = f"{item['quant']['relative_error']:.3g}"
            fixed = f"{100.0 * item['quant']['fixed_share']:.1f}"
        if size is None:
            linear_mb = bits = index_text = "—"
        else:
            linear_mb = _mb(size["total_bytes"])
            bits = f"{size['bits_per_param']:.2f}"
            index_text = _mb(size["index_bytes"])
        lines.append(
            f"{fraction:<6} {title:<18} {k_text:>6} {actual:>8} "
            f"{_ppl_text(item.get('ppl')):>10} {_ratio_text(item.get('ratio_vs_bf16')):>9} "
            f"{err:>7} {fixed:>10} {linear_mb:>10} {bits:>8} {index_text:>9}"
        )
    lines.append(
        f"* «Linear МБ» — фактический размер (триты базы + триты остатка + "
        f"масштабы fp16 + индексы), посчитан по каждой матрице отдельно; "
        f"файл = Linear + таблица словаря в 1 трит ({_mb(shape['embed_trit_bytes'])} МБ)."
    )
    lines.append(
        "* «снято err» — какую долю энергии ошибки базы (Σr²) забрала поправка "
        "Σ(s2·B)²: у идеально адресного отбора это ~p (5 % → 5 %), "
        "у случайного — p·(средний |r|² / max |r|²)."
    )
    lines.append(
        "* индекс: смещения по 7 бит на выбранный вес (дешевле при доле "
        "< 1/7) или битовая карта 1 бит на вес; берётся дешевле."
    )
    lines.append("")
    lines.append("декод (жадно, chat-template, "
                 + str(result.get("decode_tokens", DEFAULT_DECODE_TOKENS))
                 + " токенов):")
    answers = result.get("answers") or {}
    if not answers:
        lines.append("  (декод не запускался)")
    for key, items in answers.items():
        item = result["variants"].get(key, {})
        lines.append(f"[{item.get('title', key)}]")
        for answer in items:
            text = " ".join(str(answer["text"]).split())
            if len(text) > 220:
                text = text[:217] + "…"
            lines.append(f"  ? {answer['prompt']}")
            lines.append(f"  > {text if text else '(пусто)'}")
    lines.append("")
    lines.extend(_three_b_lines(result["three_b"]))
    lines.append("")
    lines.append("=== вывод ===")
    lines.extend(verdict_lines(result))
    return "\n".join(lines)


def _three_b_lines(three_b: dict[str, Any]) -> list[str]:
    config = three_b["config"]
    entries = three_b["entries"]
    reference = three_b["reference"]
    lines = [
        "=== оценка 3B (аналитически, по config.json, веса не нужны) ===",
        f"Linear {config['linear_params']:,} параметров, {config['linear_blocks']:,} "
        f"блоков; таблица {config['embed_params']:,}".replace(",", " "),
        "",
        "вид                        Linear, МБ  бит/пар  индекс МБ  файл+trit табл., МБ",
    ]
    for entry in entries.values():
        lines.append(
            f"{entry['title']:<26} {_mb(entry['linear_bytes']):>10} "
            f"{entry['bits_per_param']:>8.2f} {_mb(entry['index_bytes']):>10} "
            f"{_mb(entry['file_trit_bytes']):>19}"
        )
    one_trit = entries["trit1"]["file_trit_bytes"]
    q4 = reference["three_b_q4_gguf_bytes"]
    lines.extend([
        "",
        f"ориентиры: 1 трит 3B = {_mb(reference['three_b_one_trit_bytes'])} МБ "
        f"(наша оценка {_mb(one_trit)} МБ), Q4 GGUF 3B = {_mb(q4)} МБ.",
    ])
    # Самая дешёвая из заказанных долей — против Q4 и против базы.
    sparse_keys = [key for key in entries if key.startswith("sparse_")]
    if sparse_keys:
        smallest = min(sparse_keys, key=lambda key: float(key.split("_", 1)[1]))
        entry = entries[smallest]
        lines.append(
            f"минимальная доля {smallest.split('_', 1)[1]} %: "
            f"{_mb(entry['file_trit_bytes'])} МБ — это "
            f"{entry['file_trit_bytes'] / q4 * 100:.0f} % от Q4 "
            f"и {entry['file_trit_bytes'] / one_trit:.2f}× от одного трита."
        )
    return lines


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


def run(args: argparse.Namespace) -> int:
    fractions = parse_fractions(args.fractions)
    criteria = parse_criteria(args.criteria)
    variants = build_variants(fractions, criteria, baselines=not args.skip_baselines)
    torch, tokenizer, model = pc.load_hf("bfloat16", args.model)
    device = pc.resolve_device(torch, args.device)
    results: dict[str, Any] = {}
    shape: LinearShape | None = None
    answers: dict[str, list[dict[str, Any]]] = {}
    started = time.monotonic()

    # Токенизатор и примеры берём один раз: модель перезагружается под
    # каждый вариант, чтобы не держать в RAM две копии весов.
    vocab = int(model.config.vocab_size)
    examples = build_ppl_examples(tokenizer, args, vocab)
    print(
        f"[sparse_trit] PPL: строк {len(examples)}, токенов ответа "
        f"{sum(item['answer_tokens'] for item in examples)}, лимит "
        f"{args.ppl_max_tokens}", file=sys.stderr,
    )
    del model
    gc.collect()

    for variant in variants:
        print(f"[sparse_trit] вариант {variant.title}: загрузка", file=sys.stderr)
        dtype = "bfloat16" if variant.name == "bf16" else "float32"
        torch, _tok, model = pc.load_hf(dtype, args.model)
        model.to(device).eval()
        if shape is None:
            shape = shape_from_model(model)
            print(
                f"[sparse_trit] Linear: {shape.linear_params} параметров, "
                f"{shape.linear_blocks} блоков; таблица {shape.embed_params}",
                file=sys.stderr,
            )
        quant = quantize_model(model, variant)
        if quant.get("relative_error") is not None and variant.needs_quant:
            print(
                f"[sparse_trit] {variant.key}: ошибка веса "
                f"{quant['relative_error']:.4g}, выбрано {quant['selected']} "
                f"весов ({100.0 * quant['fraction_actual']:.2f} %), "
                f"Linear {_mb(quant['size']['total_bytes'])} МБ",
                file=sys.stderr,
            )
        ppl = pc.hf_answer_ppl(torch, model, examples, progress=False)
        print(
            f"[sparse_trit] {variant.key}: PPL {ppl['perplexity']:.6g} "
            f"({ppl['tokens']} токенов, {ppl['nll_sum']:.4f} NLL)",
            file=sys.stderr,
        )
        results[variant.key] = {
            "variant": variant.key,
            "name": variant.name,
            "title": variant.title,
            "fraction": variant.fraction,
            "criterion": variant.criterion,
            "k": quant.get("k", 0),
            "fraction_actual": quant.get("fraction_actual", 0.0),
            "ppl": ppl["perplexity"],
            "tokens": ppl["tokens"],
            "nll_sum": ppl["nll_sum"],
            "quant": quant,
            "size": quant.get("size"),
        }
        del model
        gc.collect()
        if device.type == "cuda":  # pragma: no cover - на CPU не бывает
            torch.cuda.empty_cache()

    baseline = results.get("bf16", {}).get("ppl")
    for item in results.values():
        item["ratio_vs_bf16"] = (
            item["ppl"] / baseline if baseline and item.get("ppl") else None
        )

    best_key = best_variant(results)
    if best_key is not None and not args.no_decode:
        best = results[best_key]
        print(f"[sparse_trit] декод лучшего варианта {best_key}", file=sys.stderr)
        torch, _tok, model = pc.load_hf("float32", args.model)
        model.to(device).eval()
        variant = Variant("sparse", float(best["fraction"]), str(best["criterion"]))
        quantize_model(model, variant)
        answers[best_key] = decode_answers(
            torch, tokenizer, model, DECODE_PROMPTS,
            max_new_tokens=args.decode_tokens, device=device,
        )
        del model
        gc.collect()

    assert shape is not None
    trit_table = table_bytes(shape.embed_params, "trit")
    result = {
        "format": "eva32.sparse_trit.eval",
        "format_version": 1,
        "task": "0149",
        "model": args.model,
        "device": str(device),
        "ppl": {
            "data": str(Path(args.ppl_data).resolve()),
            "lines": len(examples),
            "tokens": results[variants[0].key]["tokens"],
            "max_tokens": int(args.ppl_max_tokens),
            "method": "tools.parity_cold.hf_answer_ppl",
            "prompt_format": f"{pc.PROMPT_PREFIX}…{pc.PROMPT_SUFFIX}",
            "reference_bf16": REFERENCE["bf16_ppl"],
        },
        "decode_tokens": int(args.decode_tokens),
        "shape": {
            "hidden": shape.hidden,
            "intermediate": shape.intermediate,
            "layers": shape.layers,
            "linear_params": shape.linear_params,
            "linear_blocks": shape.linear_blocks,
            "embed_params": shape.embed_params,
            "embed_trit_bytes": trit_table,
            "total_params": shape.total_params,
        },
        "variants": results,
        "best_variant": best_key,
        "base_size": _breakdown_dict(
            trit_breakdown(shape.linear_params, blocks=shape.linear_blocks)
        ),
        "answers": answers,
        "decode_prompts": list(DECODE_PROMPTS),
        "timings": {"total_seconds": time.monotonic() - started},
        "peak_rss_mb": peak_rss_mb(),
        "three_b": three_b_report(three_b_config(), fractions, criteria),
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


def dry_run(args: argparse.Namespace) -> int:
    """План и размеры без torch: проверка аргументов и арифметики."""
    fractions = parse_fractions(args.fractions)
    criteria = parse_criteria(args.criteria)
    variants = build_variants(fractions, criteria, baselines=not args.skip_baselines)
    shape = linear_shape_from_config(three_b_config())
    model_path = Path(args.model)
    print("=== sparse_trit_eval: dry-run ===")
    print(f"модель: {args.model}"
          f"{'' if model_path.is_dir() else ' (локального каталога нет — прогон упадёт)'}")
    print(f"PPL: {args.ppl_data}, строк {args.ppl_lines}, "
          f"лимит {args.ppl_max_tokens} токенов (метод parity_cold)")
    print(f"варианты: {', '.join(variant.title for variant in variants)}")
    print("доля, к/блок, факт.доля: " + "; ".join(
        f"{_percent(fraction)} % → {selected_per_block(fraction)}/"
        f"{BLOCK_SIZE} = {100.0 * actual_fraction(fraction):.2f} %"
        for fraction in fractions
    ))
    print("индекс: " + "; ".join(
        f"{_percent(fraction)} % → {choose_index_kind(fraction)}"
        for fraction in fractions
    ))
    print(f"декод: {'нет' if args.no_decode else f'{len(DECODE_PROMPTS)} промпта × ' + str(args.decode_tokens) + ' токенов'}")
    print()
    print("оценка размера Qwen2.5-3B (Linear, без таблицы):")
    blocks = shape.linear_blocks
    for fraction in fractions:
        breakdown = sparse_breakdown(shape.linear_params, fraction, blocks=blocks)
        print(
            f"  {_percent(fraction):>3} %: {_mb(breakdown.total_bytes)} МБ, "
            f"{breakdown.bits_per_param:.2f} бит/параметр, "
            f"индекс {breakdown.index_kind} {_mb(breakdown.index_bytes)} МБ"
        )
    trit = trit_breakdown(shape.linear_params, blocks=blocks)
    dual = dual_breakdown(shape.linear_params, blocks=blocks)
    print(f"  база: {_mb(trit.total_bytes)} МБ, {trit.bits_per_param:.2f} бит/параметр")
    print(f"  два трита: {_mb(dual.total_bytes)} МБ, {dual.bits_per_param:.2f} бит/параметр")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Аргументы: по умолчанию — ровно режим 0143/0148 (20 строк, 128 токенов)."""
    parser = argparse.ArgumentParser(
        description="Разреженный второй трит: PPL и фактический размер (задача 0149)",
    )
    parser.add_argument("--model", default=default_model(),
                        help="HF-модель (по умолчанию — Qwen2.5-0.5B из .hfcache)")
    parser.add_argument("--ppl-data", default=str(pc.DEFAULT_PPL_DATA))
    parser.add_argument("--ppl-lines", type=int, default=DEFAULT_PPL_LINES)
    parser.add_argument("--ppl-max-tokens", type=int, default=DEFAULT_PPL_MAX_TOKENS)
    parser.add_argument("--fractions", default=",".join(_percent(f) for f in DEFAULT_FRACTIONS),
                        help="доли остатка: 0.05 (доля) или 5 (проценты), через запятую")
    parser.add_argument("--criteria", default=",".join(SELECTION_CRITERIA),
                        help="критерии отбора: " + ", ".join(SELECTION_CRITERIA))
    parser.add_argument("--skip-baselines", action="store_true",
                        help="не мерить bf16 и 1 трит (только доли)")
    parser.add_argument("--decode-tokens", type=int, default=DEFAULT_DECODE_TOKENS)
    parser.add_argument("--no-decode", action="store_true",
                        help="не декодировать ответы (быстрее)")
    parser.add_argument("--device", default="cpu", choices=("cpu", "auto"),
                        help="устройство HF: замер идёт на CPU (parity_cold.hf_answer_ppl "
                             "не переносит входы на GPU), auto = cpu без CUDA")
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
    except SparseTritError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except pc.ColdParityError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n[sparse_trit] прервано", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
