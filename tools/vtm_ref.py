#!/usr/bin/env python3
"""VTM v0.2 — эталонная Python-реализация: контейнер, исполнитель, семантика.

Это **наша истина**. C-интерпретатор (`src/vtm/`) обязан воспроизводить
поведение этого модуля; если результаты расходятся — прав Python, а не C.
Здесь нет torch и numpy: только stdlib (`struct`, `math`, `json`) и
`tools/trit_codec.py` для упаковки тритов.

Контейнер
=========

Все числа little-endian, выравнивание данных — 32 байта.

```
0                       заголовок 32 байта
  0  magic   "VTM1"                4
  4  version u16                   2
  6  flags   u16                   2
  8  n_instr u32                   4   число инструкций
 12  n_tensors u32                 4   число записей в таблице тензоров
 16  n_program u32                 4   размер секции программы в байтах (= n_instr*16)
 20  program_offset u32            4   абсолютное смещение программы в файле
 24  meta_offset u32               4   абсолютное смещение JSON-метаданных (0 = нет)
 28  meta_len u32                  4   длина JSON-метаданных в байтах (без padding)
32                       таблица тензоров: n_tensors * 52 байта
  0  name[24]  UTF-8, NUL-дополнение
 24  kind u8                        0 триты (TRT1), 1 fp32, 2 fp16, 3 bf16
 25  ndim u8                        1..4
 26  dims u32[4]                    недостающие измерения = 0
 42  offset u32                     смещение ОТ НАЧАЛА СЕКЦИИ ДАННЫХ, кратно 32
 46  scale_offset u32               0xFFFFFFFF (VTM_NO_OFFSET) — масштабы лежат
                                    внутри TRT1-сегмента по offset; для fp — 0
                                    (читатель принимает и offset+18 как синоним)
 50  reserved u16                   0
     (20 из 32 байт заголовка — это поля спеки; 12 байт «резерва» заняты
      program_offset/meta_offset/meta_len, что обратно совместимо: C может
      игнорировать эти три слова и читать программу с конца файла.)
data_start = align32(32 + 52*n_tensors)
секция данных:      payload'ы тензоров подряд, каждый выровнен на 32 байта
meta_start = data_start + data_size   (JSON, опционально, дополнен нулями до 32)
program_offset = meta_start + align32(meta_len)  — программа идёт ПОСЛЕДНЕЙ
программа:          n_instr * 16 байт, файл заканчивается ровно на ней
```

Почему программа в конце: C-читатель, который ничего не знает про метаданные,
всё равно может взять её как `file_size - n_instr*16`; `program_offset` из
заголовка — тот же ответ.

Версия 2 (ADR 0011): `dims` — `u32 × 4`, запись тензора 52 байта. Файлы v0.1
(version=1, запись 44 байта) не читаются: `read_vtm` бросает `ValueError`
«формат v0.1 не поддерживается» — ценных артефактов v0.1 нет, синтетика
пересобирается за секунды.

Тензор тритов (kind 0) — сегмент TRT1 (см. `docs/trit-format.md`):

```
"TRT1" | version u8 (=1) | flags u8 (бит0: 1=f32 масштабы, 0=f16)
       | n_trits u64 | n_scale u32
масштабы: n_scale * (4 или 2) байта
слова:    ceil(n_trits/160)*8 слов по 4 байта (20 тритов в слове, pack20)
```

`n_scale = ceil(n_trits/128)` — масштаб на **128** тритов (блок квантования
QAT), хотя упаковочный блок — 160 тритов (8 слов = 32 байта). Это сознательное
расширение `docs/trit-format.md` §2 (там масштаб на 160; вопрос открыт в §8):
VTM фиксирует 128. Пересечение границ строки блоком 128 невозможно, если
`cols` кратно 128, — компилятор всегда порождает такие матрицы.

**Расхождение с `src/vtm/` на 2026-11-18**: C-загрузчик ждёт `n_blocks` в
TRT1 как число 160-тритовых блоков и один масштаб на 160 тритов
(`scales[wi >> 3]` в `backend_scalar.c`). Чтобы C читал файлы этого
компилятора, ему нужно: считать `n_scale = ceil(n_trits/128)` из заголовка,
`n_words = ceil(n_trits/160)*8`, а в TMV брать `scales[(i*cols+j)/128]`.
Пока это не сделано, `read_vtm`/`execute` Python — эталон, а C разойдётся.

Инструкция (16 байт):

```
opcode u8 | flags u8 | dst u16 | a u16 | b u16 | c u16 | reserved u16 | imm i32
```

Порядок как в C: 10 байт полей, 2 байта резерва (выравнивание i32), затем
`imm` — смещение 12, как в `src/vtm/interp.c`. `dst/a/b/c` — индексы в
таблице тензоров (не указатели); исключение — `b` у `ROPE`: это число
голов, а не индекс (но C проверяет его как индекс, поэтому компилятор
держит `heads < n_tensors`). `imm` — целое (позиция/eps/токен),
`flags` — локальны для операции: `CAUSAL` (0x01) у `SM`/`ATTN`,
`EMB_INPUT` (0x01) у `EMB`, `ROPE_INTERLEAVED` (0x02) у `ROPE`.

Семантика операций v0.1
=======================

Все операции читают тензоры `a`, `b`, `c` и пишут в `dst`. Арифметика
считается в double, результат каждой операции округляется до float32 —
ровно как в C (арена хранит float). Запись в `dst == a` допустима: значение
сначала читается целиком.

Позиция: `pos = state.pos + imm` (`state.pos` — номер прохода в `execute`;
в C это `vtm_set_position()`). Компилятор пишет `imm = 0`, так что
`imm` — смещение, а не абсолютная позиция.

`TMV` — `y = W · x`, троичная матрица на вектор.
    `a` — матрица (rows, cols) row-major, `b` — вектор длины cols.
    Масштабы применяются **к результату TMV, не к весам**:
    `y[i] = sum_b scale_b * sum_{j in block b} trit[i,j] * x[j]`,
    где `b = (i*cols + j) // 128`. При `cols % 128 == 0` блок не пересекает
    границу строки, и это эквивалентно одному масштабу на строку.
    **Для C**: масштаб — на 128 тритов (блок QAT), а не на 160; текущий
    `backend_scalar.c` берёт `scales[wi >> 3]` (160) и обязан перейти на 128.
    Если `a` — fp32/fp16/bf16 (плотная матрица), `y[i] = Σ W[i,j]·x[j]`
    без масштабов (расширение Python; C v0.1 TMV по тритам).

`MM` — `Y = A · B`, только fp: `A` (m,k), `B` (k,n), `dst` — (m,n) row-major.

`RMSN` — `y = x / sqrt(mean(x²) + eps) · w`, `imm` = **eps в тысячных**:
    `eps = (float)imm / 1000.0f` (C округляет до float32 — повторяем).
    Нормализация по всему вектору: `n = x.n_elements`, `w` длины `n`.
    (C-ядро тоже нормализует весь тензор; компилятор подаёт только 1-D.)
    Если `sqrt(...) == 0` (строго нулевой вектор и eps=0) — результат нули.
    Qwen eps=1e-6 в тысячных округляется до `imm = 0` — осознанная потеря
    точности v0.1; синтетика использует eps=1e-3 (`imm = 1`).

`ROPE` — поворот `x` по позиции: `a` — `x` (обычно формы
    `(n_heads, head_dim)`), `b` — **число голов** (не индекс тензора!),
    `head_dim = n_elements / b`; `dst` — тот же размер.
    `flags & 0x02` (`ROPE_INTERLEAVED`) — пары `(2k, 2k+1)`; иначе half
    (HF/Qwen `rotate_half`) — пары `(k, k + head_dim/2)`:
        `y[k]   = x[k]·cos − x[k+half]·sin`
        `y[k+half] = x[k+half]·cos + x[k]·sin`
    `inv_freq[k] = theta ** (-2k/head_dim)`, `theta = meta["rope_theta"]`
    (по умолчанию 10000; C v0.1 жёстко 10000 — для Qwen с theta=1e6 C
    обязан научиться читать theta из контейнера). Угол `pos · inv_freq[k]`,
    `head_dim` чётный. Раскладка по умолчанию — half (как Qwen/Llama).

`SM` — softmax по последней оси, численно устойчивый (`exp(x - max)`).
    `flags & CAUSAL`: в строке `r` остаются `j <= r + pos` (как в C-ядре),
    остальные элементы — 0. Для квадратной матрицы при pos=0 это обычная
    нижнетреугольная маска. Для вектора-строки маска не меняет результат.

`SILU` — `x · sigmoid(x)` поэлементно, формула как в C: `v / (1 + exp(-v))`
    (при очень отрицательном `v` — 0, без переполнения exp).

`MUL` — `dst = a · b` поэлементно; `ADD` — `dst = a + b`. Размеры совпадают
    (broadcasting в v0.1 нет).

`EMB` — `y = E[token]`, `a` — матрица (vocab, hidden).
    `flags & 0x01` (`EMB_INPUT`) — токен текущего прохода (host:
    `vtm_set_token`; компилятор всегда ставит этот флаг), иначе `imm` — id.

`KVW` — записать K,V в кэш на позицию: `a` — K, `c` — V (по `kv_dim`),
    `dst` — K-кэш `(ctx, kv_dim)`, `b` — V-кэш `(ctx, kv_dim)`.
    Позиция `pos = state.pos + imm`. Оба кэша не сбрасываются между
    проходами: исполнитель находит их по инструкциям KVW (`dst` и `b`).

`ATTN` — `out = softmax(Q·Kᵀ/√head_dim)·V`:
    `a` — Q `(n_heads, q_len, head_dim)`; `b` — K-кэш
    `(ctx, n_kv_heads*head_dim)`, `c` — V-кэш той же формы;
    `dst` — `(n_heads, q_len, head_dim)`. `q_start = state.pos + imm`,
    валидны ключи `0..q_start+q_len-1`. GQA:
    `group = n_heads / n_kv_heads`, голова `h` идёт в kv-голову `h // group`.
    Масштаб `1/sqrt(head_dim)`, softmax с вычитанием максимума.
    `flags & CAUSAL`: ключ `j` виден запросу `qi`, только если
    `j <= q_start + qi`. Для одного токена (`q_len = 1`, `q_start = pos`)
    это всё `0..pos` — ровно декод.

`SAMP` — выбрать токен из логитов `a`; `imm` — номер стратегии: 0 argmax
    (первый максимум), 1 temperature, 2 top-k, 3 top-p; `imm` вне 0..3 —
    `ValueError`. Параметры (temperature, top_k, top_p, repetition_penalty,
    repetition_window, sample_seed) — из меты контейнера, штраф за повторы —
    по окну последних токенов. ГПСЧ (xorshift64*, засев splitmix64) и порядок
    вычислений совпадают с C побитово: при одном `seed` оба интерпретатора
    дают ту же последовательность токенов (docs/vtm.md, «Сэмплинг»). `dst`
    (опционально) — тензор, куда пишется id как float; исполнитель помнит
    логиты в результате.

`CPY` — копировать весь `a` в `dst` (число элементов равно, выбора строки нет).

`SCAL` — `dst = a · (imm/1000)`.

`HALT` — останов прохода.

Проход и генерация
==================

`execute(vtm, tokens, max_new)` прогоняет программу по одному токену за проход
(так же, как хост вызывает `vtm_run()` в C). Позиция прохода — номер шага
(`0, 1, 2, ...`); она подставляется в `pos = state.pos + imm`, токен прохода
доступен `EMB` с флагом `EMB_INPUT`. Если `tokens` пуст, делается ровно один
проход с позицией 0 (синтетика без эмбеддинга). `max_new` — сколько токенов
сгенерировать после промпта (0 = только промпт, возвращаются логиты последнего
прохода).

Ограничения v0.1
================

* нет broadcasting и многомерных matmul: MM только 2-D;
* eps в тысячных не выражает 1e-6 (см. RMSN);
* C v0.1 не читает метаданные: `rope_theta` (Qwen 1e6) и `rope_layout`
  он пока не знает — синтетика использует theta=10000/half, чтобы гейт
  проходил без этого;
* в контейнере нет контрольной суммы: битые данные ловятся по magic, версии,
  диапазонам и длинам, но не по содержимому;
* тритовые тензоры ограничены 4 ГиБ смещениями u32; файл больше — ошибка.
"""

from __future__ import annotations

import json
import math
import os
import struct
import sys
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Iterable, Sequence

try:  # импорт как пакета: tests/*, `python3 -m tools.vtm_compile`
    from tools.trit_codec import pack20, unpack20
except ImportError:  # pragma: no cover - запуск как `python3 tools/vtm_ref.py`
    from trit_codec import pack20, unpack20

# ---------------------------------------------------------------------------
# Константы формата
# ---------------------------------------------------------------------------

VTM_MAGIC = b"VTM1"
VTM_VERSION = 2
VTM_VERSION_V01 = 1
TRT1_MAGIC = b"TRT1"
TRT1_VERSION = 1
TRT1_HEADER_SIZE = 18

HEADER_SIZE = 32
TENSOR_ENTRY_SIZE = 52
INSTR_SIZE = 16
ALIGNMENT = 32
NAME_SIZE = 24
MAX_DIMS = 4
MAX_TENSORS = 0xFFFF
#: Одно измерение тензора — u32; больше не влезает в запись таблицы.
MAX_DIM_VALUE = 0xFFFFFFFF

TRITS_PER_WORD = 20
TRITS_PER_STORAGE_BLOCK = 160
WORDS_PER_STORAGE_BLOCK = TRITS_PER_STORAGE_BLOCK // TRITS_PER_WORD  # 8
SCALE_BLOCK = 128

#: scale_offset тритов: масштабы лежат внутри TRT1-сегмента (как в C).
NO_OFFSET = 0xFFFFFFFF

KIND_TRITS = 0
KIND_F32 = 1
KIND_F16 = 2
KIND_BF16 = 3
KIND_NAMES = {
    KIND_TRITS: "trits",
    KIND_F32: "fp32",
    KIND_F16: "fp16",
    KIND_BF16: "bf16",
}

#: Флаги контейнера (заголовок).
VFLAG_F32_SCALES = 0x0001  # масштабы тритов по умолчанию float32
VFLAG_CAUSAL = 0x0002      # модель причинная (LM)

#: Стратегии SAMP (imm инструкции, задача 0051; см. docs/vtm.md).
SAMP_ARGMAX = 0
SAMP_TEMPERATURE = 1
SAMP_TOPK = 2
SAMP_TOPP = 3
VFLAG_KNOWN = VFLAG_F32_SCALES | VFLAG_CAUSAL

#: Флаги инструкции — локальны для opcode (как в C).
IFLAG_CAUSAL = 0x01            # SM, ATTN
IFLAG_EMB_INPUT = 0x01         # EMB: токен из состояния хоста
IFLAG_INPLACE = 0x02           # подсказка: dst может совпадать с a
IFLAG_ROPE_INTERLEAVED = 0x02  # ROPE: пары (2i, 2i+1), иначе половинки
IFLAG_KNOWN = IFLAG_CAUSAL | IFLAG_INPLACE


class Op(IntEnum):
    """Коды операций v0.1 (docs/vtm.md, «Набор операций»)."""

    TMV = 0x01
    MM = 0x02
    RMSN = 0x03
    ROPE = 0x04
    SM = 0x05
    SILU = 0x06
    MUL = 0x07
    ADD = 0x08
    EMB = 0x09
    KVW = 0x0A
    ATTN = 0x0B
    SAMP = 0x0C
    CPY = 0x0D
    SCAL = 0x0E
    HALT = 0x0F


OP_NAMES = {int(op): op.name for op in Op}

_HEADER_STRUCT = struct.Struct("<4sHHIII")
_RESERVED_STRUCT = struct.Struct("<III")
_TENSOR_STRUCT = struct.Struct("<24sBB4IIIH")
_INSTR_STRUCT = struct.Struct("<BBHHHH2xi")
_TRT1_STRUCT = struct.Struct("<4sBBQI")

if _HEADER_STRUCT.size + _RESERVED_STRUCT.size != HEADER_SIZE:  # pragma: no cover
    raise RuntimeError("заголовок VTM должен быть 32 байта")
if _TENSOR_STRUCT.size != TENSOR_ENTRY_SIZE:  # pragma: no cover
    raise RuntimeError("запись таблицы тензоров должна быть 52 байта")
if _INSTR_STRUCT.size != INSTR_SIZE:  # pragma: no cover
    raise RuntimeError("инструкция VTM должна быть 16 байт")
if _TRT1_STRUCT.size != TRT1_HEADER_SIZE:  # pragma: no cover
    raise RuntimeError("заголовок TRT1 должен быть 18 байт")


# ---------------------------------------------------------------------------
# Мелкие помощники
# ---------------------------------------------------------------------------


def align_up(value: int, alignment: int = ALIGNMENT) -> int:
    """Округлить вверх до `alignment` (по умолчанию 32)."""
    if alignment <= 0:
        raise ValueError("alignment должен быть положительным")
    if value < 0:
        raise ValueError("выравнивать отрицательное смещение нельзя")
    return (value + alignment - 1) // alignment * alignment


def n_scale_blocks(n_values: int, block_size: int = SCALE_BLOCK) -> int:
    """Сколько блоков масштаба нужно на `n_values` элементов."""
    if n_values < 0:
        raise ValueError("число элементов не может быть отрицательным")
    if block_size <= 0:
        raise ValueError("block_size должен быть положительным")
    return (n_values + block_size - 1) // block_size


def storage_words(n_trits: int) -> int:
    """Слов pack20 с округлением до 160-тритовых (32-байтовых) блоков."""
    if n_trits < 0:
        raise ValueError("число тритов не может быть отрицательным")
    return (n_trits + TRITS_PER_STORAGE_BLOCK - 1) // TRITS_PER_STORAGE_BLOCK * WORDS_PER_STORAGE_BLOCK


def prod(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def _f32(value: float) -> float:
    """Округлить до float32 — чтобы память и файл давали одно и то же число."""
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def round_half_away(value: float) -> int:
    """Округление «половина от нуля» — как в `qat/ternary.py`."""
    if not math.isfinite(value):
        raise ValueError(f"нельзя округлить неконечное значение: {value!r}")
    if value >= 0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def _f32_to_bf16(value: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    lsb = (bits >> 16) & 1
    return ((bits + 0x7FFF + lsb) >> 16) & 0xFFFF


def _bf16_to_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", int(bits) << 16))[0]


def _check_finite(values: Sequence[float], what: str) -> None:
    for value in values:
        if not math.isfinite(value):
            raise ValueError(f"{what}: неконечное значение {value!r}")


# ---------------------------------------------------------------------------
# Тензоры
# ---------------------------------------------------------------------------


@dataclass
class Tensor:
    """Запись таблицы тензоров + полезная нагрузка.

    `data` — значения fp32/fp16/bf16 (в памяти всегда float),
    `trits`/`scales` — триты и масштабы блоков 128.
    `data is None` / `trits is None` означает «тензор без полезной нагрузки»
    (каркас графа без весов): такой тензор нельзя записать в файл, но можно
    оценить по размеру.
    """

    name: str
    kind: int
    dims: tuple[int, ...]
    data: list[float] | None = None
    trits: list[int] | None = None
    scales: list[float] | None = None
    scale_dtype: str = "float32"
    block_size: int = SCALE_BLOCK
    offset: int = 0
    scale_offset: int = 0

    def __post_init__(self) -> None:
        self.dims = tuple(int(d) for d in self.dims)
        if not 1 <= len(self.dims) <= MAX_DIMS:
            raise ValueError(f"тензор {self.name!r}: ndim={len(self.dims)}, допустимо 1..{MAX_DIMS}")
        if any(d <= 0 for d in self.dims):
            raise ValueError(f"тензор {self.name!r}: размеры должны быть > 0, получено {self.dims}")
        if any(d > MAX_DIM_VALUE for d in self.dims):
            raise ValueError(
                f"тензор {self.name!r}: размер {max(self.dims)} не влезает в u32 "
                f"(максимум {MAX_DIM_VALUE})"
            )
        name_bytes = self.name.encode("utf-8") if isinstance(self.name, str) else b""
        if not self.name or b"\0" in name_bytes:
            raise ValueError(f"тензор {self.name!r}: имя пустое или содержит NUL")
        if len(name_bytes) > NAME_SIZE:
            raise ValueError(
                f"тензор {self.name!r}: имя {len(name_bytes)} байт, максимум {NAME_SIZE}"
            )
        if self.kind not in KIND_NAMES:
            raise ValueError(f"тензор {self.name!r}: неизвестный kind={self.kind}")
        if self.scale_dtype not in ("float32", "float16"):
            raise ValueError(f"тензор {self.name!r}: scale_dtype={self.scale_dtype!r}")
        if self.kind == KIND_TRITS:
            if self.block_size != SCALE_BLOCK:
                raise ValueError(
                    f"тензор {self.name!r}: VTM v0.1 хранит масштаб только на {SCALE_BLOCK} тритов"
                )
            if self.data is not None:
                raise ValueError(f"тензор {self.name!r}: тритовый тензор не имеет fp-данных")
            if self.trits is not None:
                if len(self.trits) != self.n_elements:
                    raise ValueError(
                        f"тензор {self.name!r}: {len(self.trits)} тритов, "
                        f"а dims дают {self.n_elements}"
                    )
                for trit in self.trits:
                    if trit not in (-1, 0, 1):
                        raise ValueError(f"тензор {self.name!r}: трит {trit!r} вне {{-1,0,1}}")
                if self.scales is None:
                    raise ValueError(f"тензор {self.name!r}: триты без масштабов")
                expected = n_scale_blocks(self.n_elements, self.block_size)
                if len(self.scales) != expected:
                    raise ValueError(
                        f"тензор {self.name!r}: масштабов {len(self.scales)}, ожидалось {expected}"
                    )
                _check_finite(self.scales, f"тензор {self.name!r}: масштаб")
                if any(s < 0 for s in self.scales):
                    raise ValueError(f"тензор {self.name!r}: масштаб не может быть отрицательным")
        else:
            if self.trits is not None or self.scales is not None:
                raise ValueError(f"тензор {self.name!r}: fp-тензор не имеет тритов")
            if self.data is not None:
                if len(self.data) != self.n_elements:
                    raise ValueError(
                        f"тензор {self.name!r}: {len(self.data)} значений, "
                        f"а dims дают {self.n_elements}"
                    )
                _check_finite(self.data, f"тензор {self.name!r}")

    # -- удобные конструкторы ------------------------------------------------

    @property
    def n_elements(self) -> int:
        return prod(self.dims)

    @property
    def kind_name(self) -> str:
        return KIND_NAMES[self.kind]

    @classmethod
    def f32(cls, name: str, dims: Sequence[int], values: Iterable[float]) -> "Tensor":
        """Плотный fp32-тензор; значения округляются до float32."""
        return cls(name=name, kind=KIND_F32, dims=tuple(dims), data=[_f32(v) for v in values])

    @classmethod
    def fp16(cls, name: str, dims: Sequence[int], values: Iterable[float]) -> "Tensor":
        return cls(name=name, kind=KIND_F16, dims=tuple(dims), data=[float(v) for v in values])

    @classmethod
    def bf16(cls, name: str, dims: Sequence[int], values: Iterable[float]) -> "Tensor":
        return cls(name=name, kind=KIND_BF16, dims=tuple(dims), data=[float(v) for v in values])

    @classmethod
    def ternary(
        cls,
        name: str,
        dims: Sequence[int],
        trits: Iterable[int],
        scales: Sequence[float] | None = None,
        scale_dtype: str = "float32",
    ) -> "Tensor":
        """Тритовый тензор. Без масштабов — absmax по тритам (0 или 1)."""
        trits_list = [int(t) for t in trits]
        n_values = prod(dims)
        if len(trits_list) != n_values:
            raise ValueError(f"тензор {name!r}: {len(trits_list)} тритов, dims дают {n_values}")
        if scales is None:
            scales_list = []
            for start in range(0, n_values, SCALE_BLOCK):
                block = trits_list[start:start + SCALE_BLOCK]
                scales_list.append(1.0 if any(block) else 0.0)
        else:
            scales_list = [_f32(float(s)) for s in scales]
        return cls(
            name=name,
            kind=KIND_TRITS,
            dims=tuple(dims),
            trits=trits_list,
            scales=scales_list,
            scale_dtype=scale_dtype,
        )

    @classmethod
    def placeholder(cls, name: str, dims: Sequence[int], kind: int = KIND_F32) -> "Tensor":
        """Тензор без данных — для каркаса графа (веса ещё не загружены)."""
        return cls(name=name, kind=kind, dims=tuple(dims))

    # -- данные --------------------------------------------------------------

    def dense(self) -> list[float]:
        """Плотные значения: для тритов — `trit * scale`, для fp — как есть."""
        if self.kind == KIND_TRITS:
            if self.trits is None or self.scales is None:
                raise ValueError(f"тензор {self.name!r}: нет данных тритов")
            block = self.block_size
            scales = self.scales
            return [trit * scales[i // block] for i, trit in enumerate(self.trits)]
        if self.data is None:
            raise ValueError(f"тензор {self.name!r}: нет данных")
        return list(self.data)


def quantize_trits(
    values: Iterable[float], block_size: int = SCALE_BLOCK
) -> tuple[list[int], list[float]]:
    """Квантовать веса в (триты, absmax-масштабы по 128) — правило QAT.

    `scale = absmax(блок)`, `trit = clamp(round_half_away(w/scale), -1, 1)`;
    у нулевого блока масштаб 0 и все триты 0. Числа округляются до float32,
    чтобы результат совпадал с контейнером, прочитанным с диска.
    """
    vals = [_f32(float(v)) for v in values]
    _check_finite(vals, "quantize_trits")
    scales: list[float] = []
    for start in range(0, len(vals), block_size):
        block = vals[start:start + block_size]
        scales.append(_f32(max(abs(v) for v in block)) if block else 0.0)
    trits: list[int] = []
    for i, value in enumerate(vals):
        scale = scales[i // block_size]
        if scale == 0.0:
            trits.append(0)
        else:
            trits.append(max(-1, min(1, round_half_away(value / scale))))
    return trits, scales


# ---------------------------------------------------------------------------
# Инструкции
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Instr:
    """Одна инструкция VTM (16 байт)."""

    op: int
    flags: int = 0
    dst: int = 0
    a: int = 0
    b: int = 0
    c: int = 0
    imm: int = 0

    def __post_init__(self) -> None:
        if int(self.op) not in OP_NAMES:
            raise ValueError(f"неизвестный opcode: {self.op!r}")
        if not 0 <= int(self.flags) <= 0xFF:
            raise ValueError(f"flags не влезает в u8: {self.flags!r}")
        if int(self.flags) & ~IFLAG_KNOWN:
            raise ValueError(f"неизвестные флаги инструкции: 0x{int(self.flags):02x}")
        for field_name in ("dst", "a", "b", "c"):
            value = int(getattr(self, field_name))
            if not 0 <= value <= 0xFFFF:
                raise ValueError(f"{field_name} не влезает в u16: {value}")
        if not -(2 ** 31) <= int(self.imm) < 2 ** 31:
            raise ValueError(f"imm не влезает в i32: {self.imm!r}")

    def pack(self) -> bytes:
        return _INSTR_STRUCT.pack(
            int(self.op), int(self.flags), int(self.dst), int(self.a),
            int(self.b), int(self.c), int(self.imm),
        )

    @classmethod
    def unpack_from(cls, data: bytes, offset: int = 0) -> "Instr":
        if offset < 0 or offset + INSTR_SIZE > len(data):
            raise ValueError(
                f"инструкция по смещению {offset} выходит за границы буфера "
                f"({len(data)} байт)"
            )
        op, flags, dst, a, b, c, imm = _INSTR_STRUCT.unpack_from(data, offset)
        return cls(op=op, flags=flags, dst=dst, a=a, b=b, c=c, imm=imm)


# ---------------------------------------------------------------------------
# Полезная нагрузка тензоров: TRT1 и fp
# ---------------------------------------------------------------------------


def tensor_payload_size(tensor: Tensor) -> int:
    """Размер payload'а тензора в байтах (без выравнивания)."""
    if tensor.kind == KIND_TRITS:
        n_values = tensor.n_elements
        scale_bytes = 4 if tensor.scale_dtype == "float32" else 2
        return (
            TRT1_HEADER_SIZE
            + n_scale_blocks(n_values, tensor.block_size) * scale_bytes
            + storage_words(n_values) * 4
        )
    if tensor.kind == KIND_F32:
        return tensor.n_elements * 4
    if tensor.kind in (KIND_F16, KIND_BF16):
        return tensor.n_elements * 2
    raise ValueError(f"тензор {tensor.name!r}: неизвестный kind={tensor.kind}")


def encode_trit_segment(tensor: Tensor) -> bytes:
    """Собрать TRT1-сегмент: заголовок + масштабы + слова pack20."""
    if tensor.kind != KIND_TRITS:
        raise ValueError(f"тензор {tensor.name!r}: это не триты")
    if tensor.trits is None or tensor.scales is None:
        raise ValueError(
            f"тензор {tensor.name!r}: нет тритов — граф собран без весов, "
            "записывать в контейнер нечего"
        )
    n_trits = tensor.n_elements
    n_scales = n_scale_blocks(n_trits, tensor.block_size)
    if len(tensor.scales) != n_scales:
        raise ValueError(
            f"тензор {tensor.name!r}: масштабов {len(tensor.scales)}, ожидалось {n_scales}"
        )
    flags = 0x01 if tensor.scale_dtype == "float32" else 0x00
    header = _TRT1_STRUCT.pack(TRT1_MAGIC, TRT1_VERSION, flags, n_trits, n_scales)
    scale_code = "f" if tensor.scale_dtype == "float32" else "e"
    try:
        scales_bytes = struct.pack(f"<{n_scales}{scale_code}", *tensor.scales)
    except (struct.error, OverflowError) as exc:
        raise ValueError(f"тензор {tensor.name!r}: масштаб не влезает в {tensor.scale_dtype}") from exc
    words = pack20(tensor.trits)
    expected = storage_words(n_trits) * 4
    if len(words) > expected:  # pragma: no cover - защита от рассинхрона констант
        raise ValueError(
            f"тензор {tensor.name!r}: упаковано {len(words)} байт при лимите {expected}"
        )
    words += b"\x00" * (expected - len(words))
    return header + scales_bytes + words


def decode_trit_segment(data: bytes) -> tuple[list[int], list[float], str]:
    """Разобрать TRT1-сегмент; лишние байты в конце (padding) игнорируются."""
    if len(data) < TRT1_HEADER_SIZE:
        raise ValueError(f"TRT1-сегмент короче заголовка: {len(data)} < {TRT1_HEADER_SIZE}")
    magic, version, flags, n_trits, n_scales = _TRT1_STRUCT.unpack_from(data, 0)
    if magic != TRT1_MAGIC:
        raise ValueError(f"чужой magic TRT1-сегмента: {magic!r}")
    if version != TRT1_VERSION:
        raise ValueError(f"неизвестная версия TRT1-сегмента: {version}")
    if flags & ~0x01:
        raise ValueError(f"неизвестные флаги TRT1-сегмента: 0x{flags:02x}")
    expected_scales = n_scale_blocks(n_trits, SCALE_BLOCK)
    if n_scales != expected_scales:
        raise ValueError(
            f"в заголовке TRT1 {n_scales} масштабов, ожидалось {expected_scales} "
            f"для {n_trits} тритов"
        )
    scale_dtype = "float32" if flags & 0x01 else "float16"
    scale_size = 4 if scale_dtype == "float32" else 2
    n_words = storage_words(n_trits)
    expected_len = TRT1_HEADER_SIZE + n_scales * scale_size + n_words * 4
    if len(data) < expected_len:
        raise ValueError(
            f"TRT1-сегмент обрезан: {len(data)} байт, ожидалось {expected_len}"
        )
    scale_code = "f" if scale_dtype == "float32" else "e"
    scales = list(struct.unpack_from(f"<{n_scales}{scale_code}", data, TRT1_HEADER_SIZE))
    words_offset = TRT1_HEADER_SIZE + n_scales * scale_size
    words = data[words_offset:words_offset + n_words * 4]
    trits = unpack20(words, n_trits)
    return trits, [float(s) for s in scales], scale_dtype


def encode_tensor_payload(tensor: Tensor) -> bytes:
    """Payload тензора в том виде, в каком он лежит в секции данных."""
    if tensor.kind == KIND_TRITS:
        return encode_trit_segment(tensor)
    if tensor.data is None:
        raise ValueError(
            f"тензор {tensor.name!r}: нет данных — граф собран без весов, "
            "записывать в контейнер нечего"
        )
    n = tensor.n_elements
    _check_finite(tensor.data, f"тензор {tensor.name!r}")
    if tensor.kind == KIND_F32:
        return struct.pack(f"<{n}f", *tensor.data)
    if tensor.kind == KIND_F16:
        try:
            return struct.pack(f"<{n}e", *tensor.data)
        except (struct.error, OverflowError) as exc:
            raise ValueError(f"тензор {tensor.name!r}: значение не влезает в fp16") from exc
    if tensor.kind == KIND_BF16:
        return struct.pack(f"<{n}H", *(_f32_to_bf16(v) for v in tensor.data))
    raise ValueError(f"тензор {tensor.name!r}: неизвестный kind={tensor.kind}")  # pragma: no cover


def decode_tensor_values(tensor: Tensor, payload: bytes) -> None:
    """Заполнить `tensor` значениями из payload'а (payload уже отрезан по размеру)."""
    if tensor.kind == KIND_TRITS:
        trits, scales, scale_dtype = decode_trit_segment(payload)
        if len(trits) != tensor.n_elements:
            raise ValueError(
                f"тензор {tensor.name!r}: в сегменте {len(trits)} тритов, "
                f"а dims дают {tensor.n_elements}"
            )
        tensor.trits = trits
        tensor.scales = scales
        tensor.scale_dtype = scale_dtype
        return
    n = tensor.n_elements
    if tensor.kind == KIND_F32:
        if len(payload) < n * 4:
            raise ValueError(f"тензор {tensor.name!r}: fp32-данные обрезаны")
        tensor.data = list(struct.unpack_from(f"<{n}f", payload, 0))
        return
    if len(payload) < n * 2:
        raise ValueError(f"тензор {tensor.name!r}: fp16/bf16-данные обрезаны")
    if tensor.kind == KIND_F16:
        tensor.data = [float(v) for v in struct.unpack_from(f"<{n}e", payload, 0)]
        return
    if tensor.kind == KIND_BF16:
        words = struct.unpack_from(f"<{n}H", payload, 0)
        tensor.data = [_bf16_to_f32(w) for w in words]
        return
    raise ValueError(f"тензор {tensor.name!r}: неизвестный kind={tensor.kind}")  # pragma: no cover


# ---------------------------------------------------------------------------
# Контейнер
# ---------------------------------------------------------------------------


@dataclass
class VTM:
    """Разобранный/собранный контейнер."""

    tensors: list[Tensor]
    program: list[Instr]
    meta: dict[str, Any]
    version: int = VTM_VERSION
    flags: int = VFLAG_F32_SCALES | VFLAG_CAUSAL
    data_offset: int = 0
    program_offset: int = 0
    meta_offset: int = 0
    meta_len: int = 0
    path: str | None = None

    # -- доступ по имени -----------------------------------------------------

    def index_of(self, name: str) -> int:
        for index, tensor in enumerate(self.tensors):
            if tensor.name == name:
                return index
        raise KeyError(f"тензора {name!r} нет в контейнере")

    def by_name(self, name: str) -> Tensor:
        return self.tensors[self.index_of(name)]

    def values(self, name: str) -> list[float]:
        return self.by_name(name).dense()

    def op_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for instr in self.program:
            key = OP_NAMES[int(instr.op)]
            counts[key] = counts.get(key, 0) + 1
        return counts

    def estimated_file_size(self) -> int:
        """Размер файла, который получится после `write_vtm`."""
        table_end = HEADER_SIZE + TENSOR_ENTRY_SIZE * len(self.tensors)
        data_start = align_up(table_end)
        data_size = 0
        for tensor in self.tensors:
            data_size = align_up(data_size + tensor_payload_size(tensor))
        meta_bytes = _encode_meta(self.meta) if self.meta else b""
        meta_offset = data_start + data_size
        program_offset = meta_offset + align_up(len(meta_bytes))
        return program_offset + len(self.program) * INSTR_SIZE


def _encode_meta(meta: dict[str, Any]) -> bytes:
    if not isinstance(meta, dict):
        raise ValueError(f"meta должен быть словарём, получено {type(meta).__name__}")
    try:
        text = json.dumps(meta, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"meta не сериализуется в JSON: {exc}") from exc
    return text.encode("utf-8")


def _as_tensor(item: Any) -> Tensor:
    if isinstance(item, Tensor):
        return item
    if isinstance(item, dict):
        return Tensor(**item)
    raise ValueError(f"ожидался Tensor, получено {type(item).__name__}")


def _as_instr(item: Any) -> Instr:
    if isinstance(item, Instr):
        return item
    if isinstance(item, (tuple, list)):
        return Instr(*item)
    if isinstance(item, dict):
        return Instr(**item)
    raise ValueError(f"ожидалась Instr, получено {type(item).__name__}")


def _validate_program(program: Sequence[Instr], n_tensors: int) -> None:
    for position, instr in enumerate(program):
        for field_name in ("dst", "a", "b", "c"):
            if int(instr.op) == Op.ROPE and field_name == "b":
                # У ROPE поле b — число голов, а не индекс тензора.
                # C (src/vtm/loader.c) проверяет b как индекс (< n_tensors),
                # поэтому компилятор обязан держать heads < n_tensors.
                if not 1 <= int(instr.b) < max(n_tensors, 1):
                    raise ValueError(
                        f"инструкция #{position}: ROPE b (число голов) = {instr.b}, "
                        f"должно быть 1..{n_tensors - 1}"
                    )
                continue
            index = int(getattr(instr, field_name))
            if n_tensors == 0:
                if index != 0:
                    raise ValueError(
                        f"инструкция #{position}: {field_name}={index}, "
                        "а тензоров в контейнере нет"
                    )
            elif index >= n_tensors:
                raise ValueError(
                    f"инструкция #{position}: {field_name}={index} за границей "
                    f"таблицы тензоров ({n_tensors})"
                )


def _validate_names(tensors: Sequence[Tensor]) -> None:
    seen: set[str] = set()
    for tensor in tensors:
        if tensor.name in seen:
            raise ValueError(f"повторяющееся имя тензора: {tensor.name!r}")
        seen.add(tensor.name)


def write_vtm(
    path: str | os.PathLike[str],
    tensors: Sequence[Tensor | dict[str, Any]],
    program: Sequence[Instr | tuple[Any, ...] | dict[str, Any]],
    meta: dict[str, Any] | None = None,
) -> VTM:
    """Собрать контейнер `.vtm` строго по спеке и записать на диск.

    Возвращает собранный `VTM` (с проставленными offset'ами) — его можно
    сразу передать в `execute`. Порядок `tensors` = порядок таблицы, индексы
    в `program` ссылаются на него.
    """
    tensor_list = [_as_tensor(item) for item in tensors]
    program_list = [_as_instr(item) for item in program]
    meta_dict: dict[str, Any] = dict(meta or {})
    _validate_names(tensor_list)
    _validate_program(program_list, len(tensor_list))
    if len(tensor_list) > MAX_TENSORS:
        raise ValueError(f"тензоров {len(tensor_list)}, максимум {MAX_TENSORS}")

    payloads = [encode_tensor_payload(tensor) for tensor in tensor_list]

    n_tensors = len(tensor_list)
    table_end = HEADER_SIZE + TENSOR_ENTRY_SIZE * n_tensors
    data_start = align_up(table_end)

    data_size = 0
    for tensor, payload in zip(tensor_list, payloads):
        tensor.offset = data_size
        tensor.scale_offset = NO_OFFSET if tensor.kind == KIND_TRITS else 0
        if tensor.offset % ALIGNMENT:
            raise RuntimeError(  # pragma: no cover - защита от ошибки в align_up
                f"тензор {tensor.name!r}: offset не выровнен"
            )
        data_size = align_up(data_size + len(payload))

    meta_bytes = _encode_meta(meta_dict) if meta_dict else b""
    meta_offset = data_start + data_size if meta_bytes else 0
    program_offset = data_start + data_size + align_up(len(meta_bytes))
    program_bytes = b"".join(instr.pack() for instr in program_list)
    file_size = program_offset + len(program_bytes)
    if file_size >= 2 ** 32:
        raise ValueError(
            f"VTM адресует файл 32-битными смещениями: {file_size} байт "
            "не влезает в контейнер"
        )

    flags = VFLAG_CAUSAL
    if all(t.kind != KIND_TRITS or t.scale_dtype == "float32" for t in tensor_list):
        flags |= VFLAG_F32_SCALES

    header = _HEADER_STRUCT.pack(
        VTM_MAGIC, VTM_VERSION, flags, len(program_list), n_tensors, len(program_bytes)
    ) + _RESERVED_STRUCT.pack(program_offset, meta_offset, len(meta_bytes))

    table = bytearray()
    for tensor in tensor_list:
        name_bytes = tensor.name.encode("utf-8")
        dims = list(tensor.dims) + [0] * (MAX_DIMS - len(tensor.dims))
        table += _TENSOR_STRUCT.pack(
            name_bytes, tensor.kind, len(tensor.dims), *dims,
            tensor.offset, tensor.scale_offset, 0,
        )

    with open(path, "wb") as handle:
        handle.write(header)
        handle.write(bytes(table))
        handle.write(b"\x00" * (data_start - table_end))
        for payload in payloads:
            handle.write(payload)
            handle.write(b"\x00" * (align_up(len(payload)) - len(payload)))
        if meta_bytes:
            handle.write(meta_bytes)
            handle.write(b"\x00" * (align_up(len(meta_bytes)) - len(meta_bytes)))
        handle.write(program_bytes)

    written = os.path.getsize(path)
    if written != file_size:  # pragma: no cover - защита от рассинхрона раскладки
        raise RuntimeError(f"записано {written} байт, ожидалось {file_size}")

    vtm = VTM(
        tensors=tensor_list,
        program=program_list,
        meta=meta_dict,
        version=VTM_VERSION,
        flags=flags,
        data_offset=data_start,
        program_offset=program_offset,
        meta_offset=meta_offset,
        meta_len=len(meta_bytes),
        path=os.fspath(path),
    )
    return vtm


def _payload_extent(
    data: bytes, start: int, kind: int, n_elements: int
) -> tuple[int, str | None]:
    """Размер payload'а по заголовку TRT1 (для тритов) или по kind/dims."""
    if kind == KIND_TRITS:
        if start + TRT1_HEADER_SIZE > len(data):
            raise ValueError("TRT1-сегмент выходит за конец файла")
        magic, version, flags, n_trits, n_scales = _TRT1_STRUCT.unpack_from(data, start)
        if magic != TRT1_MAGIC:
            raise ValueError(f"чужой magic TRT1-сегмента: {magic!r}")
        if version != TRT1_VERSION:
            raise ValueError(f"неизвестная версия TRT1-сегмента: {version}")
        if flags & ~0x01:
            raise ValueError(f"неизвестные флаги TRT1-сегмента: 0x{flags:02x}")
        if n_trits != n_elements:
            raise ValueError(
                f"в TRT1-сегменте {n_trits} тритов, а dims дают {n_elements}"
            )
        expected_scales = n_scale_blocks(n_trits, SCALE_BLOCK)
        if n_scales != expected_scales:
            raise ValueError(
                f"в TRT1-сегменте {n_scales} масштабов, ожидалось {expected_scales}"
            )
        scale_dtype = "float32" if flags & 0x01 else "float16"
        scale_bytes = 4 if scale_dtype == "float32" else 2
        size = TRT1_HEADER_SIZE + n_scales * scale_bytes + storage_words(n_trits) * 4
        return size, scale_dtype
    if kind == KIND_F32:
        return n_elements * 4, None
    if kind in (KIND_F16, KIND_BF16):
        return n_elements * 2, None
    raise ValueError(f"неизвестный kind тензора: {kind}")


def read_vtm(path: str | os.PathLike[str]) -> VTM:
    """Прочитать контейнер и вернуть `VTM`. Любая порча данных — `ValueError`."""
    with open(path, "rb") as handle:
        data = handle.read()

    if len(data) < HEADER_SIZE:
        raise ValueError(f"файл короче заголовка VTM: {len(data)} < {HEADER_SIZE}")
    magic, version, flags, n_instr, n_tensors, n_program = _HEADER_STRUCT.unpack_from(data, 0)
    if magic != VTM_MAGIC:
        raise ValueError(f"чужой magic контейнера: {magic!r}")
    if version == VTM_VERSION_V01:
        raise ValueError(
            "формат v0.1 не поддерживается: пересоберите контейнер в v0.2 "
            "(dims u32, запись тензора 52 байта)"
        )
    if version != VTM_VERSION:
        raise ValueError(f"неизвестная версия контейнера: {version}")
    if n_tensors > MAX_TENSORS:
        raise ValueError(f"n_tensors={n_tensors} больше предела {MAX_TENSORS}")

    table_end = HEADER_SIZE + TENSOR_ENTRY_SIZE * n_tensors
    if table_end > len(data):
        raise ValueError(
            f"таблица тензоров ({n_tensors} записей) выходит за конец файла "
            f"({len(data)} байт)"
        )
    if n_program not in (0, n_instr * INSTR_SIZE):
        raise ValueError(
            f"n_program={n_program} не совпадает с n_instr*16={n_instr * INSTR_SIZE}"
        )
    if n_instr * INSTR_SIZE > len(data) - HEADER_SIZE:
        raise ValueError(f"n_instr={n_instr} не влезает в файл")

    program_offset, meta_offset, meta_len = _RESERVED_STRUCT.unpack_from(data, HEADER_SIZE - 12)
    if program_offset == 0:
        program_offset = len(data) - n_instr * INSTR_SIZE
    if program_offset < table_end:
        raise ValueError(f"program_offset={program_offset} заходит на таблицу тензоров")
    program_end = program_offset + n_instr * INSTR_SIZE
    if program_end > len(data):
        raise ValueError(
            f"секция программы ({n_instr} инструкций) выходит за конец файла"
        )

    data_start = align_up(table_end)
    if data_start > program_offset:
        raise ValueError("секция данных начинается позже программы")

    if meta_len:
        if meta_offset < data_start or meta_offset > len(data):
            raise ValueError(f"meta_offset={meta_offset} вне секции данных")
        if meta_offset + meta_len > len(data):
            raise ValueError("JSON-метаданные выходят за конец файла")
        if meta_offset > program_offset:
            raise ValueError("JSON-метаданные начинаются позже программы")
        meta_end = meta_offset
        if meta_offset + meta_len > program_offset:
            raise ValueError("JSON-метаданные налезают на программу")
    else:
        meta_end = program_offset
    if meta_end < data_start:
        raise ValueError("секция данных пуста или отрицательна")

    # --- таблица тензоров ---------------------------------------------------
    tensors: list[Tensor] = []
    ranges: list[tuple[int, int, str]] = []
    for index in range(n_tensors):
        entry_offset = HEADER_SIZE + TENSOR_ENTRY_SIZE * index
        raw_name, kind, ndim, d0, d1, d2, d3, offset, scale_offset, _reserved = (
            _TENSOR_STRUCT.unpack_from(data, entry_offset)
        )
        name_bytes = raw_name.split(b"\x00", 1)[0]
        try:
            name = name_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"тензор #{index}: имя не в UTF-8") from exc
        if not name:
            raise ValueError(f"тензор #{index}: пустое имя")
        if kind not in KIND_NAMES:
            raise ValueError(f"тензор {name!r}: неизвестный kind={kind}")
        if not 1 <= ndim <= MAX_DIMS:
            raise ValueError(f"тензор {name!r}: ndim={ndim}, допустимо 1..{MAX_DIMS}")
        dims = tuple((d0, d1, d2, d3)[:ndim])
        if any(d == 0 for d in dims):
            raise ValueError(f"тензор {name!r}: нулевое измерение в {dims}")
        if offset % ALIGNMENT:
            raise ValueError(f"тензор {name!r}: offset={offset} не выровнен на {ALIGNMENT}")

        tensor = Tensor(name=name, kind=kind, dims=dims, offset=offset, scale_offset=scale_offset)
        size, scale_dtype = _payload_extent(data, data_start + offset, kind, tensor.n_elements)
        start = data_start + offset
        end = start + size
        if end > meta_end:
            raise ValueError(
                f"тензор {name!r}: данные ({size} байт по offset={offset}) выходят "
                f"за секцию данных"
            )
        if kind == KIND_TRITS:
            if scale_offset not in (0, NO_OFFSET, offset + TRT1_HEADER_SIZE):
                raise ValueError(
                    f"тензор {name!r}: scale_offset={scale_offset}, ожидалось "
                    f"{NO_OFFSET} (масштабы внутри TRT1)"
                )
            tensor.scale_offset = NO_OFFSET
            tensor.scale_dtype = scale_dtype or "float32"
        elif scale_offset != 0:
            raise ValueError(f"тензор {name!r}: scale_offset должен быть 0 для fp-тензора")
        decode_tensor_values(tensor, data[start:end])
        ranges.append((start, end, name))
        tensors.append(tensor)

    ranges.sort()
    for (prev_start, prev_end, prev_name), (next_start, _next_end, next_name) in zip(
        ranges, ranges[1:]
    ):
        if next_start < prev_end:
            raise ValueError(f"тензоры {prev_name!r} и {next_name!r} перекрываются в файле")

    # --- программа ----------------------------------------------------------
    program: list[Instr] = []
    for position in range(n_instr):
        instr = Instr.unpack_from(data, program_offset + position * INSTR_SIZE)
        for field_name in ("dst", "a", "b", "c"):
            if int(instr.op) == Op.ROPE and field_name == "b":
                if int(instr.b) < 1:
                    raise ValueError(
                        f"инструкция #{position}: ROPE b (число голов) = {instr.b}"
                    )
                continue
            index = int(getattr(instr, field_name))
            if n_tensors == 0:
                if index != 0:
                    raise ValueError(
                        f"инструкция #{position}: {field_name}={index}, а тензоров нет"
                    )
            elif index >= n_tensors:
                raise ValueError(
                    f"инструкция #{position}: {field_name}={index} за границей "
                    f"таблицы тензоров ({n_tensors})"
                )
        program.append(instr)

    # --- метаданные ---------------------------------------------------------
    meta: dict[str, Any] = {}
    if meta_len:
        try:
            decoded_meta = json.loads(data[meta_offset:meta_offset + meta_len].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"JSON-метаданные не читаются: {exc}") from exc
        if not isinstance(decoded_meta, dict):
            raise ValueError("JSON-метаданные должны быть объектом")
        meta = decoded_meta

    return VTM(
        tensors=tensors,
        program=program,
        meta=meta,
        version=version,
        flags=flags,
        data_offset=data_start,
        program_offset=program_offset,
        meta_offset=meta_offset,
        meta_len=meta_len,
        path=os.fspath(path),
    )


# ---------------------------------------------------------------------------
# Эталонный исполнитель
# ---------------------------------------------------------------------------


@dataclass
class ExecResult:
    """Результат `execute`: токены, логиты и снимок рабочей области."""

    prompt: list[int]
    generated: list[int]
    logits: list[float] | None
    steps: int
    workspace: dict[str, list[float]]
    halted: bool = False

    @property
    def tokens(self) -> list[int]:
        return self.prompt + self.generated

    def get(self, name: str) -> list[float] | None:
        return self.workspace.get(name)


@dataclass
class _State:
    pos: int = 0
    token: int = 0
    sampled: int | None = None
    logits: list[float] | None = None
    halted: bool = False


def _tmv_trits(
    trits: Sequence[int],
    scales: Sequence[float],
    x: Sequence[float],
    rows: int,
    cols: int,
    block_size: int = SCALE_BLOCK,
) -> list[float]:
    """`y = W·x` с масштабом блока, применённым к частичной сумме."""
    if len(x) != cols:
        raise ValueError(f"TMV: вектор длины {len(x)}, а матрица просит {cols}")
    if len(trits) != rows * cols:
        raise ValueError(f"TMV: тритов {len(trits)}, ожидалось {rows * cols}")
    y = [0.0] * rows
    for i in range(rows):
        base = i * cols
        acc = 0.0
        j = 0
        while j < cols:
            block = (base + j) // block_size
            end = min(cols, (block + 1) * block_size - base)
            partial = 0.0
            for k in range(j, end):
                trit = trits[base + k]
                if trit:
                    partial += trit * x[k]
            acc += partial * scales[block]
            j = end
        y[i] = acc
    return y


def _tmv_dense(
    weights: Sequence[float], x: Sequence[float], rows: int, cols: int
) -> list[float]:
    if len(x) != cols:
        raise ValueError(f"TMV: вектор длины {len(x)}, а матрица просит {cols}")
    if len(weights) != rows * cols:
        raise ValueError(f"TMV: весов {len(weights)}, ожидалось {rows * cols}")
    y = [0.0] * rows
    for i in range(rows):
        base = i * cols
        acc = 0.0
        for j in range(cols):
            acc += weights[base + j] * x[j]
        y[i] = acc
    return y


def _matmul(a: Sequence[float], b: Sequence[float], m: int, k: int, n: int) -> list[float]:
    if len(a) != m * k:
        raise ValueError(f"MM: в A {len(a)} значений, ожидалось {m * k}")
    if len(b) != k * n:
        raise ValueError(f"MM: в B {len(b)} значений, ожидалось {k * n}")
    y = [0.0] * (m * n)
    for i in range(m):
        a_base = i * k
        y_base = i * n
        for j in range(n):
            acc = 0.0
            for t in range(k):
                acc += a[a_base + t] * b[t * n + j]
            y[y_base + j] = acc
    return y


def _rmsnorm(x: Sequence[float], w: Sequence[float], eps: float) -> list[float]:
    """RMSN по всему вектору, как C-ядро: `x / sqrt(mean(x²)+eps) · w`."""
    n = len(x)
    if len(w) != n:
        raise ValueError(f"RMSN: весов {len(w)}, ожидалось {n}")
    if n == 0:
        return []
    mean_sq = 0.0
    for value in x:
        mean_sq += value * value
    mean_sq /= n
    denom = math.sqrt(mean_sq + eps)
    if denom == 0.0:
        return [0.0] * n
    return [value / denom * w[i] for i, value in enumerate(x)]


def _softmax_last(x: Sequence[float], n_last: int, causal: bool, offset: int = 0) -> list[float]:
    """Softmax по последней оси; CAUSAL оставляет `j <= r + offset` (как в C)."""
    if n_last <= 0:
        raise ValueError("SM: последняя ось должна быть > 0")
    if len(x) % n_last:
        raise ValueError(f"SM: {len(x)} значений не делится на {n_last}")
    rows = len(x) // n_last
    out = [0.0] * len(x)
    for row_index in range(rows):
        base = row_index * n_last
        limit = n_last - 1
        if causal:
            limit = min(n_last - 1, row_index + offset)
        peak = -math.inf
        for j in range(limit + 1):
            value = x[base + j]
            if math.isnan(value):
                continue
            if value > peak:
                peak = value
        if peak == -math.inf:
            continue
        exps = [0.0] * (limit + 1)
        total = 0.0
        for j in range(limit + 1):
            value = x[base + j]
            if math.isnan(value):
                continue
            weight = math.exp(value - peak)
            exps[j] = weight
            total += weight
        if total <= 0.0 or not math.isfinite(total):
            continue
        for j in range(limit + 1):
            out[base + j] = exps[j] / total
    return out


def _rope(
    x: Sequence[float],
    heads: int,
    head_dim: int,
    pos: int,
    theta: float = 10000.0,
    interleaved: bool = False,
) -> list[float]:
    """RoPE: half-раскладка (Qwen/Llama) или пары (2i, 2i+1)."""
    if heads <= 0:
        raise ValueError(f"ROPE: heads={heads} должно быть > 0")
    if head_dim < 2 or head_dim % 2:
        raise ValueError(f"ROPE: head_dim={head_dim} должен быть чётным и >= 2")
    if len(x) != heads * head_dim:
        raise ValueError(f"ROPE: {len(x)} значений, ожидалось {heads * head_dim}")
    half = head_dim // 2
    inv_freq = [theta ** (-2.0 * k / head_dim) for k in range(half)]
    out = [0.0] * len(x)
    for head in range(heads):
        base = head * head_dim
        for k in range(half):
            angle = pos * inv_freq[k]
            cos = math.cos(angle)
            sin = math.sin(angle)
            if interleaved:
                first = base + 2 * k
                second = base + 2 * k + 1
            else:
                first = base + k
                second = base + half + k
            value_a = x[first]
            value_b = x[second]
            out[first] = value_a * cos - value_b * sin
            out[second] = value_b * cos + value_a * sin
    return out


def _silu(value: float) -> float:
    """`v / (1 + exp(-v))` — та же формула, что в C-ядре (без overflow)."""
    if value < -700.0:
        return 0.0
    return value / (1.0 + math.exp(-value))


def _argmax(values: Sequence[float]) -> int:
    best = 0
    best_value = values[0]
    for index in range(1, len(values)):
        if values[index] > best_value:
            best = index
            best_value = values[index]
    return best


# --- SAMP: стратегии, параметры из меты и ГПСЧ -----------------------------
#
# Паритет с C (docs/vtm.md, «Сэмплинг»): imm — номер стратегии 0..3,
# параметры — из меты, один сэмпл без перехода в argmax потребляет ровно одно
# число xorshift64*. Всё считается в float (Python) над теми же значениями,
# что C читает как f32, поэтому последовательности совпадают побитово.

_SAMP_MASK64 = (1 << 64) - 1
_SAMP_MAX_WINDOW = 256
_SAMP_DEFAULTS = {
    "temperature": 1.0,
    "top_k": 0,
    "top_p": 1.0,
    "repetition_penalty": 1.0,
    "repetition_window": 64,
    "sample_seed": 0,
    "has_seed": False,
}


def _samp_mix64(value: int) -> int:
    """splitmix64 — только для засева xorshift64* (как в C)."""
    value = (value + 0x9E3779B97F4A7C15) & _SAMP_MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _SAMP_MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _SAMP_MASK64
    return value ^ (value >> 31)


def _samp_meta_number(meta: dict, key: str) -> float | None:
    """Первое числовое значение ключа на минимальной глубине — как C-сканер.

    C ищет ключ по тексту JSON на любом уровне вложенности и при нескольких
    вхождениях берёт самое внешнее; здесь тот же порядок обхода.
    """
    found: list[tuple[int, float]] = []

    def walk(value: object, depth: int) -> None:
        if isinstance(value, dict):
            for name, child in value.items():
                if (
                    name == key
                    and isinstance(child, (int, float))
                    and not isinstance(child, bool)
                ):
                    found.append((depth, float(child)))
                walk(child, depth + 1)
        elif isinstance(value, list):
            for item in value:
                walk(item, depth + 1)

    walk(meta, 0)
    if not found:
        return None
    return min(found, key=lambda pair: pair[0])[1]


def _samp_params(meta: dict) -> dict:
    """Разбор и кламп параметров SAMP — ровно как samp_params_from_meta()."""
    params = dict(_SAMP_DEFAULTS)
    number = _samp_meta_number

    value = number(meta, "temperature")
    if value is not None:
        if math.isnan(value) or value < 0.0:
            params["temperature"] = 0.0
        elif value > 1e6:
            params["temperature"] = 1e6
        else:
            params["temperature"] = value
    value = number(meta, "top_k")
    if value is not None:
        if value <= 0.0:
            params["top_k"] = 0
        elif value > 1e9:
            params["top_k"] = 1_000_000_000
        else:
            params["top_k"] = int(value)
    value = number(meta, "top_p")
    if value is not None:
        if math.isnan(value):
            params["top_p"] = _SAMP_DEFAULTS["top_p"]
        elif value <= 0.0:
            params["top_p"] = 0.0
        elif value >= 1.0:
            params["top_p"] = 1.0
        else:
            params["top_p"] = value
    value = number(meta, "repetition_penalty")
    if value is not None:
        if math.isnan(value) or value <= 0.0:
            params["repetition_penalty"] = _SAMP_DEFAULTS["repetition_penalty"]
        elif value > 1e6:
            params["repetition_penalty"] = 1e6
        else:
            params["repetition_penalty"] = value
    value = number(meta, "repetition_window")
    if value is not None:
        if value <= 0.0:
            params["repetition_window"] = 0
        elif value > _SAMP_MAX_WINDOW:
            params["repetition_window"] = _SAMP_MAX_WINDOW
        else:
            params["repetition_window"] = int(value)
    value = number(meta, "sample_seed")
    if (
        value is not None
        and math.isfinite(value)
        and 0.0 <= value < 9007199254740992.0
    ):
        params["sample_seed"] = int(value)
        params["has_seed"] = True
    return params


class _Sampler:
    """Состояние SAMP машины: параметры, ГПСЧ и окно последних токенов."""

    def __init__(self, meta: dict) -> None:
        self.params = _samp_params(meta)
        self.history: list[int] = []
        seed = self.params["sample_seed"] ^ 0x9E3779B97F4A7C15
        self.rng = _samp_mix64(seed)
        if self.rng == 0:
            self.rng = 0x2545F4914F6CDD1D

    def _next(self) -> int:
        value = self.rng
        value ^= (value << 13) & _SAMP_MASK64
        value ^= value >> 7
        value ^= (value << 17) & _SAMP_MASK64
        value &= _SAMP_MASK64
        self.rng = value
        return (value * 0x2545F4914F6CDD1D) & _SAMP_MASK64

    def _uniform(self) -> float:
        return (self._next() >> 11) * (1.0 / 9007199254740992.0)

    def _penalty_active(self) -> bool:
        params = self.params
        return (
            params["repetition_penalty"] != 1.0
            and params["repetition_window"] > 0
            and bool(self.history)
        )

    def _penalized(self, token: int, value: float) -> float:
        if not self._penalty_active():
            return value
        window = self.history[-self.params["repetition_window"]:]
        if token not in window:
            return value
        penalty = self.params["repetition_penalty"]
        return value / penalty if value > 0.0 else value * penalty

    def _greedy(self, logits: Sequence[float]) -> int:
        best = 0
        best_value = self._penalized(0, logits[0])
        for index in range(1, len(logits)):
            value = self._penalized(index, logits[index])
            if value > best_value:
                best = index
                best_value = value
        return best

    def choose(self, logits: Sequence[float], strategy: int) -> int:
        params = self.params
        temperature = params["temperature"]
        greedy = (
            strategy == 0
            or (strategy == 1 and temperature <= 0.0)
            or (strategy == 2 and (temperature <= 0.0 or params["top_k"] == 1))
            or (strategy == 3 and (temperature <= 0.0 or params["top_p"] <= 0.0))
        )
        if greedy:
            token = self._greedy(logits)
        else:
            token = self._sample(logits, strategy)
        self.history.append(token)
        del self.history[:-_SAMP_MAX_WINDOW]
        return token

    def _sample(self, logits: Sequence[float], strategy: int) -> int:
        params = self.params
        values = [
            -math.inf if math.isnan(float(value)) else self._penalized(index, float(value))
            for index, value in enumerate(logits)
        ]
        order = sorted(range(len(values)), key=lambda index: (-values[index], index))
        kept = len(order)
        top = values[order[0]]
        if not math.isfinite(top):
            return order[0]
        if strategy == 2 and 0 < params["top_k"] < kept:
            kept = params["top_k"]
        weights = [
            math.exp((values[order[index]] - top) / params["temperature"])
            for index in range(kept)
        ]
        total = sum(weights)
        if strategy == 3 and params["top_p"] < 1.0 and total > 0.0:
            need = params["top_p"] * total
            upto = 0.0
            for index in range(len(weights)):
                upto += weights[index]
                if upto >= need:
                    kept = index + 1
                    break
            weights = weights[:kept]
            total = sum(weights)
        if total > 0.0 and math.isfinite(total):
            threshold = self._uniform() * total
            cumulative = 0.0
            for index in range(kept):
                cumulative += weights[index]
                if cumulative >= threshold:
                    return order[index]
            return order[kept - 1]
        return order[0]




class _Interpreter:
    """Исполнение одной программы; состояние живёт между проходами."""

    def __init__(self, vtm: VTM) -> None:
        self.vtm = vtm
        self.tensors = vtm.tensors
        self.program = vtm.program
        self.meta = vtm.meta
        self.theta = float(self.meta.get("rope_theta", 10000.0))
        self.rope_layout = str(self.meta.get("rope_layout", "half"))
        self.sampler = _Sampler(self.meta)
        #: индексы тензоров-кэшей KV: они не сбрасываются между проходами
        #: KVW пишет в dst (K-кэш) и b (V-кэш): их не сбрасываем между проходами.
        self.cache_indices = {
            index
            for instr in self.program
            if int(instr.op) == Op.KVW
            for index in (int(instr.dst), int(instr.b))
        }
        self.workspace: dict[int, list[float]] = {}

    # -- доступ к значениям --------------------------------------------------

    def _read(self, index: int) -> list[float]:
        if index in self.workspace:
            return self.workspace[index]
        tensor = self.tensors[index]
        if tensor.kind == KIND_TRITS:
            raise ValueError(
                f"тензор {tensor.name!r}: тритовая матрица допустима только в TMV/MM"
            )
        if tensor.data is None:
            raise ValueError(f"тензор {tensor.name!r}: нет данных для чтения")
        return tensor.data

    def _dense(self, index: int) -> list[float]:
        if index in self.workspace:
            return list(self.workspace[index])
        return self.tensors[index].dense()

    def _write(self, index: int, values: Sequence[float]) -> None:
        expected = self.tensors[index].n_elements
        if len(values) != expected:
            raise ValueError(
                f"тензор {self.tensors[index].name!r}: записано {len(values)} значений, "
                f"ожидалось {expected}"
            )
        self.workspace[index] = [_f32(v) for v in values]

    def _ensure_cache(self, index: int) -> list[float]:
        if index not in self.workspace:
            self.workspace[index] = [0.0] * self.tensors[index].n_elements
        return self.workspace[index]

    # -- проход --------------------------------------------------------------

    def begin_step(self) -> None:
        """Сбросить рабочую область, сохранив KV-кэш."""
        self.workspace = {
            index: values
            for index, values in self.workspace.items()
            if index in self.cache_indices
        }

    def run_step(self, state: _State) -> None:
        for instr in self.program:
            self._execute(instr, state)
            if state.halted:
                break

    # -- операции ------------------------------------------------------------

    def _execute(self, instr: Instr, state: _State) -> None:
        op = int(instr.op)
        if op == Op.TMV:
            matrix = self.tensors[int(instr.a)]
            x = self._read(int(instr.b))
            if matrix.kind == KIND_TRITS:
                if matrix.trits is None or matrix.scales is None:
                    raise ValueError(f"тензор {matrix.name!r}: нет тритов")
                if len(matrix.dims) != 2:
                    raise ValueError(f"TMV: матрица {matrix.name!r} должна быть 2-D")
                rows, cols = matrix.dims
                y = _tmv_trits(
                    matrix.trits, matrix.scales, x, rows, cols, matrix.block_size
                )
            else:
                if len(matrix.dims) != 2:
                    raise ValueError(f"TMV: матрица {matrix.name!r} должна быть 2-D")
                rows, cols = matrix.dims
                y = _tmv_dense(matrix.dense(), x, rows, cols)
            self._write(int(instr.dst), y)
        elif op == Op.MM:
            a = self._dense(int(instr.a))
            b = self._dense(int(instr.b))
            a_dims = self.tensors[int(instr.a)].dims
            b_dims = self.tensors[int(instr.b)].dims
            if len(a_dims) != 2 or len(b_dims) != 2:
                raise ValueError("MM: оба операнда должны быть 2-D")
            m, k = a_dims
            k2, n = b_dims
            if k != k2:
                raise ValueError(f"MM: размеры не сходятся: {a_dims} × {b_dims}")
            self._write(int(instr.dst), _matmul(a, b, m, k, n))
        elif op == Op.RMSN:
            x = self._read(int(instr.a))
            w = self._read(int(instr.b))
            if instr.imm < 0:
                raise ValueError(f"RMSN: eps (imm) не может быть отрицательным: {instr.imm}")
            # C округляет eps до float32 — повторяем, чтобы числа совпали.
            eps = _f32(instr.imm / 1000.0)
            self._write(int(instr.dst), _rmsnorm(x, w, eps))
        elif op == Op.ROPE:
            x = self._read(int(instr.a))
            heads = int(instr.b)
            if heads <= 0:
                raise ValueError(f"ROPE: heads (b) = {heads}, должно быть > 0")
            n_values = self.tensors[int(instr.a)].n_elements
            if n_values % heads:
                raise ValueError(
                    f"ROPE: {n_values} элементов не делятся на {heads} голов"
                )
            head_dim = n_values // heads
            pos = state.pos + int(instr.imm)
            interleaved = bool(instr.flags & IFLAG_ROPE_INTERLEAVED) or (
                self.rope_layout == "interleaved"
            )
            self._write(
                int(instr.dst),
                _rope(x, heads, head_dim, pos, self.theta, interleaved),
            )
        elif op == Op.SM:
            x = self._read(int(instr.a))
            n_last = self.tensors[int(instr.a)].dims[-1]
            causal = bool(instr.flags & IFLAG_CAUSAL)
            self._write(
                int(instr.dst),
                _softmax_last(x, n_last, causal, state.pos + int(instr.imm)),
            )
        elif op == Op.SILU:
            x = self._read(int(instr.a))
            self._write(int(instr.dst), [_silu(value) for value in x])
        elif op == Op.MUL:
            a = self._read(int(instr.a))
            b = self._read(int(instr.b))
            if len(a) != len(b):
                raise ValueError(f"MUL: размеры не совпадают: {len(a)} и {len(b)}")
            self._write(int(instr.dst), [x * y for x, y in zip(a, b)])
        elif op == Op.ADD:
            a = self._read(int(instr.a))
            b = self._read(int(instr.b))
            if len(a) != len(b):
                raise ValueError(f"ADD: размеры не совпадают: {len(a)} и {len(b)}")
            self._write(int(instr.dst), [x + y for x, y in zip(a, b)])
        elif op == Op.EMB:
            table = self.tensors[int(instr.a)]
            if len(table.dims) != 2:
                raise ValueError("EMB: таблица должна быть (vocab, hidden)")
            vocab, hidden = table.dims
            if instr.flags & IFLAG_EMB_INPUT:
                token = state.token
            else:
                token = int(instr.imm)
            if not 0 <= token < vocab:
                raise ValueError(f"EMB: токен {token} вне словаря {vocab}")
            values = table.dense()
            self._write(int(instr.dst), values[token * hidden:(token + 1) * hidden])
        elif op == Op.KVW:
            k_cache_tensor = self.tensors[int(instr.dst)]
            v_cache_tensor = self.tensors[int(instr.b)]
            if len(k_cache_tensor.dims) != 2 or len(v_cache_tensor.dims) != 2:
                raise ValueError(
                    f"KVW: кэши {k_cache_tensor.name!r}/{v_cache_tensor.name!r} "
                    "должны быть (ctx, kv_dim)"
                )
            ctx, kv_dim = k_cache_tensor.dims
            if v_cache_tensor.dims != (ctx, kv_dim):
                raise ValueError(
                    f"KVW: K- и V-кэш разной формы: {k_cache_tensor.dims} и "
                    f"{v_cache_tensor.dims}"
                )
            pos = state.pos + int(instr.imm)
            if not 0 <= pos < ctx:
                raise ValueError(f"KVW: позиция {pos} вне кэша длины {ctx}")
            k = self._read(int(instr.a))
            v = self._read(int(instr.c))
            if len(k) != kv_dim or len(v) != kv_dim:
                raise ValueError(
                    f"KVW: K/V длины {len(k)}/{len(v)}, кэш ждёт {kv_dim}"
                )
            k_cache = self._ensure_cache(int(instr.dst))
            v_cache = self._ensure_cache(int(instr.b))
            base = pos * kv_dim
            k_cache[base:base + kv_dim] = [_f32(value) for value in k]
            v_cache[base:base + kv_dim] = [_f32(value) for value in v]
        elif op == Op.ATTN:
            q = self._read(int(instr.a))
            q_dims = self.tensors[int(instr.a)].dims
            if len(q_dims) != 3:
                raise ValueError("ATTN: Q должен быть (n_heads, q_len, head_dim)")
            n_heads, q_len, head_dim = q_dims
            k_cache_tensor = self.tensors[int(instr.b)]
            v_cache_tensor = self.tensors[int(instr.c)]
            if len(k_cache_tensor.dims) != 2 or len(v_cache_tensor.dims) != 2:
                raise ValueError(
                    f"ATTN: K/V-кэши {k_cache_tensor.name!r}/{v_cache_tensor.name!r} "
                    "должны быть (ctx, kv_dim)"
                )
            ctx, kv_dim = k_cache_tensor.dims
            if v_cache_tensor.dims != (ctx, kv_dim):
                raise ValueError(
                    f"ATTN: K- и V-кэш разной формы: {k_cache_tensor.dims} и "
                    f"{v_cache_tensor.dims}"
                )
            if head_dim <= 0 or kv_dim % head_dim:
                raise ValueError(
                    f"ATTN: kv_dim={kv_dim} не делится на head_dim={head_dim}"
                )
            n_kv_heads = kv_dim // head_dim
            if n_heads % n_kv_heads:
                raise ValueError(
                    f"ATTN: {n_heads} голов не делятся на {n_kv_heads} kv-голов"
                )
            group = n_heads // n_kv_heads
            q_start = state.pos + int(instr.imm)
            if q_start < 0:
                raise ValueError(f"ATTN: q_start={q_start} отрицателен")
            key_count = q_start + q_len
            if key_count > ctx:
                raise ValueError(
                    f"ATTN: ключей {key_count} больше длины кэша {ctx}"
                )
            k_cache = self.workspace.get(int(instr.b), [0.0] * (ctx * kv_dim))
            v_cache = self.workspace.get(int(instr.c), [0.0] * (ctx * kv_dim))
            causal = bool(instr.flags & IFLAG_CAUSAL)
            scale = 1.0 / math.sqrt(head_dim)
            out = [0.0] * (n_heads * q_len * head_dim)
            for head in range(n_heads):
                kv_head = head // group
                for qi in range(q_len):
                    q_base = (head * q_len + qi) * head_dim
                    out_base = q_base
                    peak = -math.inf
                    scores = [None] * key_count
                    for j in range(key_count):
                        if causal and j > q_start + qi:
                            continue
                        base = j * kv_dim + kv_head * head_dim
                        dot = 0.0
                        for d in range(head_dim):
                            dot += q[q_base + d] * k_cache[base + d]
                        value = dot * scale
                        scores[j] = value
                        if value > peak:
                            peak = value
                    if peak == -math.inf:
                        continue
                    exps = [0.0] * key_count
                    total = 0.0
                    for j, value in enumerate(scores):
                        if value is None:
                            continue
                        weight = math.exp(value - peak)
                        exps[j] = weight
                        total += weight
                    if total <= 0.0 or not math.isfinite(total):
                        continue
                    for d in range(head_dim):
                        acc = 0.0
                        for j in range(key_count):
                            if scores[j] is None:
                                continue
                            base = j * kv_dim + kv_head * head_dim
                            acc += (exps[j] / total) * v_cache[base + d]
                        out[out_base + d] = acc
            self._write(int(instr.dst), out)
        elif op == Op.SAMP:
            logits = self._read(int(instr.a))
            if not logits:
                raise ValueError("SAMP: пустые логиты")
            strategy = int(instr.imm)
            if strategy < 0 or strategy > 3:
                raise ValueError(
                    f"SAMP: стратегия imm={strategy} не реализована (0..3)"
                )
            token = self.sampler.choose(logits, strategy)
            state.sampled = token
            state.logits = [float(value) for value in logits]
            self._write(int(instr.dst), [float(token)])
        elif op == Op.CPY:
            source = self._read(int(instr.a))
            dest = self.tensors[int(instr.dst)]
            if len(source) != dest.n_elements:
                raise ValueError(
                    f"CPY: элементов {len(source)}, а у {dest.name!r} "
                    f"{dest.n_elements}"
                )
            self._write(int(instr.dst), source)
        elif op == Op.SCAL:
            x = self._read(int(instr.a))
            factor = instr.imm / 1000.0
            self._write(int(instr.dst), [value * factor for value in x])
        elif op == Op.HALT:
            state.halted = True
        else:  # pragma: no cover - Instr уже проверил opcode
            raise ValueError(f"неизвестная операция: {op}")


def _fallback_logits(vtm: VTM, workspace: dict[int, list[float]]) -> list[float] | None:
    """Если SAMP не встретился — взять тензор `logits` (или `out`) из снимка."""
    for name in ("logits", "out"):
        try:
            index = vtm.index_of(name)
        except KeyError:
            continue
        if index in workspace:
            return [float(value) for value in workspace[index]]
    return None


def execute(
    vtm: VTM | str | os.PathLike[str],
    tokens: Sequence[int] = (),
    max_new: int = 0,
) -> ExecResult:
    """Эталонно исполнить программу.

    `tokens` — промпт (ids; пустой список = один проход без эмбеддинга),
    `max_new` — сколько новых токенов сгенерировать. Позиция прохода — номер
    шага; инструкции видят `pos = state.pos + imm`, токен прохода — EMB с
    флагом `EMB_INPUT`.
    """
    if not isinstance(vtm, VTM):
        vtm = read_vtm(vtm)
    prompt = [int(token) for token in tokens]
    max_new = int(max_new)
    if max_new < 0:
        raise ValueError("max_new не может быть отрицательным")

    interpreter = _Interpreter(vtm)
    state = _State()
    steps = (len(prompt) + max_new) if prompt else 1
    logits: list[float] | None = None
    generated: list[int] = []
    halted = False

    for step in range(steps):
        interpreter.begin_step()
        state.pos = step
        if prompt:
            state.token = (
                prompt[step] if step < len(prompt) else generated[step - len(prompt)]
            )
        else:
            state.token = 0
        state.sampled = None
        state.logits = None
        state.halted = False
        interpreter.run_step(state)
        halted = halted or state.halted
        if state.logits is not None:
            logits = state.logits
        if (
            prompt
            and state.sampled is not None
            and step >= len(prompt) - 1
            and len(generated) < max_new
        ):
            generated.append(int(state.sampled))

    if logits is None:
        logits = _fallback_logits(vtm, interpreter.workspace)

    workspace = {
        vtm.tensors[index].name: list(values)
        for index, values in interpreter.workspace.items()
    }
    return ExecResult(
        prompt=prompt,
        generated=generated,
        logits=logits,
        steps=steps,
        workspace=workspace,
        halted=halted,
    )


def summary(vtm: VTM) -> dict[str, Any]:
    """Краткая сводка по контейнеру — для CLI и логов."""
    kinds: dict[str, int] = {}
    data_bytes = 0
    for tensor in vtm.tensors:
        kinds[tensor.kind_name] = kinds.get(tensor.kind_name, 0) + 1
        data_bytes += align_up(tensor_payload_size(tensor))
    return {
        "file": vtm.path,
        "version": vtm.version,
        "flags": vtm.flags,
        "tensors": len(vtm.tensors),
        "kinds": kinds,
        "data_bytes": data_bytes,
        "instructions": len(vtm.program),
        "ops": vtm.op_counts(),
        "meta_keys": sorted(vtm.meta),
        "estimated_file_bytes": vtm.estimated_file_size(),
    }


def _main(argv: Sequence[str]) -> int:
    if len(argv) != 2 or argv[0] not in ("info", "summary"):
        print("использование: python3 tools/vtm_ref.py info <файл.vtm>", file=sys.stderr)
        return 2
    vtm = read_vtm(argv[1])
    print(json.dumps(summary(vtm), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main(sys.argv[1:]))
