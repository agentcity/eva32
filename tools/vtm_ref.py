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
 24  kind u8                        0 триты (TRT1), 1 fp32, 2 fp16, 3 bf16,
                                    4 int8 + масштабы (INT8-сегмент),
                                    5 Q4: 4 бита + масштабы (Q4-сегмент)
 25  ndim u8                        1..4
 26  dims u32[4]                    недостающие измерения = 0
 42  offset u32                     смещение ОТ НАЧАЛА СЕКЦИИ ДАННЫХ, кратно 32
                                    (у KV-кэша — не адрес: 0 или 0xFFFFFFFF
                                     при флаге VFLAG_KV_OUT_OF_FILE)
 46  scale_offset u32               0xFFFFFFFF (VTM_NO_OFFSET) — масштабы лежат
                                    внутри TRT1-сегмента по offset; для fp — 0
                                    (читатель принимает и offset+18 как синоним)
 50  flags/резерв u16              0 — обычный тензор; бит 0 — KV-кэш
                                    (VTM_TENSOR_FLAG_KV_CACHE, payload'а нет);
                                    биты 1..2 — хранение тритов (0 packed20,
                                    1 native; задача 0057)
     (20 из 32 байт заголовка — это поля спеки; 12 байт «резерва» заняты
      program_offset/meta_offset/meta_len, что обратно совместимо: C может
      игнорировать эти три слова и читать программу с конца файла.)
data_start = align32(32 + 52*n_tensors)
секция данных:      payload'ы тензоров подряд, каждый выровнен на 32 байта
meta_start = data_start + data_size   (JSON, опционально, дополнен нулями до 32)
program_offset = meta_start + align32(meta_len)  — программа идёт ПОСЛЕДНЕЙ
программа:          n_instr * 16 байт, файл заканчивается ровно на ней
```

KV-кэши вне контейнера (задачи 0101, 0118). Поле по смещению 50 было резервом и
всегда 0; младший бит занят ролью `KV_CACHE`. Такой тензор существует только
в таблице (`name`/`kind`/`dims`), а его payload в секцию данных **не пишется**:
`offset` — не адрес, `scale_offset = 0`. Кэш пуст на старте и заполняется KVW,
поэтому хранить под него `ctx × kv_dim × 4` байт в файле незачем — для
1.5B-раскладки (28 слоёв, ctx 2048, kv_dim 256) это 112 МиБ. Читатель создаёт
буфер сам (Python — список нулей, C — буфер в арене); старые контейнеры
присылают 0 в этом поле и читаются прежним путём, с содержимым из файла.
Тензор с флагом обязан быть fp (kind 1..3); неизвестные биты читатель
игнорирует.

Новый вид контейнера (задача 0118) отмечается флагом заголовка
`VFLAG_KV_OUT_OF_FILE` (0x0004) и `offset == KV_NO_OFFSET` (0xFFFFFFFF) у
каждого KV-кэша. Старый загрузчик до 0101 этот бит игнорирует, зато проверяет
выравнивание offset и отказывает с `VTM_E_FORMAT`, не читая данные: мусор в
кэш вместо отказа — ровно тот риск, ради которого флаг введён. Контейнеры без
флага (0101 и старше) по-прежнему несут `offset = 0` и читаются как раньше.
Суммарный размер KV-кэшей ограничен `MAX_KV_BYTES` (512 МиБ): 192-байтный файл
иначе просит `calloc` до 4 ГиБ (предел `VTM_MAX_ELEMENTS`).

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
расширение `docs/trit-format.md` §2 (там масштаб на 160; вопрос открыт в §9):
VTM фиксирует 128. Пересечение границ строки блоком 128 невозможно, если
`cols` кратно 128, — компилятор всегда порождает такие матрицы.

Хранение тритов (задача 0057). У TRT1 есть version 2: биты 1..2 байта `flags`
задают хранение — `0` `packed20` (20 тритов в u32, как version 1) или `1`
`native` (1 знаковый байт на трит, 4 трита в слове). Массив слов/байтов
занимает `storage_words(n, storage) · 4` байт: для `packed20` —
`ceil(n/160)·8` слов, для `native` — `ceil(n/4)` слов. Version 1 всегда
`packed20`; так писали все контейнеры до 0057, и байты не меняются. Тот же
идентификатор дублируется в битах 1..2 `flags` записи тензора; читатель
сверяет дубликаты. `native` в 5 раз больше `packed20` (32 → 160 байт на 160
тритов), зато читается без делений на 3. C-загрузчик нормализует оба хранения
в `packed20` в арене (`src/vtm/trit.h`, `vtm_trit_store`), поэтому
интерпретатор о хранении не знает.

Тензор int8 (kind 4, задача 0044) — сегмент с тем же 18-байтовым заголовком:

```
"INT8" | version u8 (=1) | flags u8 (бит0: 1=f32 масштабы, 0=f16)
       | n_values u64 | n_scale u32
масштабы: n_scale * (4 или 2) байта
q:        n_values знаковых байт (q в -127..127)
```

`n_scale = ceil(n_values/128)`, `scale_offset = NO_OFFSET`, `w = q · scale`.
`-128` запрещён: диапазон симметричный, как у `Int8Embedding` из QAT.

Тензор Q4 (kind 5, задача 0151) — сегмент с заголовком 18 байт (Q4_1 — 22):

```
"Q4_0" | version u8 (=1) | flags u8 (бит0: 1=f32 масштабы, 0=f16)
       | n_values u64 | n_groups u32
[Q4_1] | n_mins u32                       — явное число min (== n_groups)
масштабы: n_groups * (4 или 2) байта
[Q4_1] min: n_mins * (4 или 2) байта
q:        n_groups * 16 байт (2 ниббла на байт)
```

`n_groups = ceil(n_values/32)`, `scale_offset = NO_OFFSET`. Группа — 32
значения в 16 байтах, раскладка как в GGML Q4_0 (её ждут ядра 0152 и импорт
GGUF 0153): байт `j` группы хранит значение `j` в младшем ниббле и значение
`j+16` в старшем. `Q4_0`: `w = (nibble − 8) · scale` (значения 0..15
кодируют −8..7); `Q4_1`: `w = nibble · scale + min`, `min` — fp32 на группу.
Явное `n_mins` у Q4_1 нужно, чтобы отсутствие min в обрезанном сегменте не
приняли за выравнивающий padding и не прочитали как Q4_0.
Хвост последней группы (если `n_values` не кратно 32) дополняется нибблом 8
и читателем игнорируется. Масштабы конечны и неотрицательны, `min` конечен.

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
* контроль целостности — по желанию писателя (write_vtm(..., integrity=True)
  или VTM_WRITE_HASH=1): sha256 секции данных и меты лежит в meta["integrity"];
  старые контейнеры без блока читаются как раньше, VTM_SKIP_HASH=1 отключает
  проверку;
* тритовые тензоры ограничены 4 ГиБ смещениями u32; файл больше — ошибка.
"""

from __future__ import annotations

import hashlib
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
#: Версия TRT1 по умолчанию: только pack20, так писали контейнеры до 0057.
TRT1_VERSION = 1
#: Версия TRT1 с явным хранением тритов: биты 1..2 flags (задача 0057).
#: Отдельная версия — чтобы читатель до 0057 отверг native с понятной ошибкой
#: версии, а не прочитал байты как слова pack20.
TRT1_VERSION_STORAGE = 2
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

#: Хранение тритов (задача 0057): pack20 — 20 тритов в u32 (как раньше);
#: native — 1 знаковый байт на трит. Идентификатор дублируется в flags записи
#: тензора (биты 1..2; бит 0 — KV_CACHE) и в TRT1 version 2 (биты 1..2 flags).
STORAGE_PACKED20 = "packed20"
STORAGE_NATIVE = "native"
TRIT_STORAGE_IDS = {STORAGE_PACKED20: 0, STORAGE_NATIVE: 1}
TRIT_STORAGE_NAMES = {0: STORAGE_PACKED20, 1: STORAGE_NATIVE}
TFLAG_STORAGE_SHIFT = 1
TFLAG_STORAGE_MASK = 0x0006
NATIVE_TRITS_PER_WORD = 4

#: scale_offset тритов: масштабы лежат внутри TRT1-сегмента (как в C).
NO_OFFSET = 0xFFFFFFFF

KIND_TRITS = 0
KIND_F32 = 1
KIND_F16 = 2
KIND_BF16 = 3
#: kind 4: знаковый int8 (-127..127) + масштабы блока 128 (задача 0044).
KIND_INT8 = 4
#: kind 5: Q4 — 4 бита (0..15) + масштабы группы 32 (задача 0151).
KIND_Q4 = 5
KIND_NAMES = {
    KIND_TRITS: "trits",
    KIND_F32: "fp32",
    KIND_F16: "fp16",
    KIND_BF16: "bf16",
    KIND_INT8: "int8",
    KIND_Q4: "q4",
}

#: Сегмент int8-тензора (kind 4): тот же 18-байтовый заголовок, что у TRT1,
#: но magic "INT8". По offset: заголовок, масштабы, n_elements байт q.
#: w = q · scale, масштаб на блок 128, q в -127..127 (симметрично, zero-point 0).
INT8_MAGIC = b"INT8"
INT8_VERSION = 1
INT8_HEADER_SIZE = 18
INT8_SCALE_BLOCK = 128
INT8_MIN = -127
INT8_MAX = 127

#: Сегмент Q4-тензора (kind 5): заголовок 18 байт, как у INT8/TRT1
#: (у Q4_1 — 22: после n_groups идёт явное n_mins).
#: magic "Q4_0" (симметричный, w = (q-8)·scale) или "Q4_1" (w = q·scale+min).
#: Группа — 32 значения в 16 байтах, раскладка GGML: байт j хранит значение j
#: в младшем ниббле и j+16 в старшем. Масштаб — fp32 (или fp16 по флагу) на
#: группу; у Q4_1 после масштабов лежит столько же min.
Q4_MAGIC = b"Q4_0"
Q4_MAGIC_ASYM = b"Q4_1"
Q4_VERSION = 1
Q4_HEADER_SIZE = 18
#: У Q4_1 после n_groups идёт явное n_mins u32 (для проверки, что min
#: действительно есть и их ровно n_groups, а не «нули из padding»).
Q4_ASYM_HEADER_SIZE = 22
Q4_SCALE_BLOCK = 32
Q4_GROUP_BYTES = Q4_SCALE_BLOCK // 2  # 16 байт = 32 ниббла
Q4_MIN = 0
Q4_MAX = 15
#: Ниббл нуля в симметричном Q4_0: (8 - 8) · scale = 0. Им же дополняется
#: хвост последней группы; читатель обрезает его по n_values.
Q4_ZERO_NIBBLE = 8
Q4_FLAG_F32_SCALES = 0x01

#: Флаги записи таблицы тензоров (поле по смещению 50, бывший «резерв»).
#: TFLAG_KV_CACHE — K/V-кэш внимания: содержимого в файле нет (offset=0),
#: читатель создаёт буфер сам (см. докстроку модуля, «KV-кэши вне контейнера»).
#: Как и в C (src/vtm/vtm.h), осмыслен только бит 0; остальные биты
#: сохраняются, но поведение не меняют — контейнер будущей версии читается.
TFLAG_KV_CACHE = 0x0001

#: offset KV-кэша в контейнере с флагом VFLAG_KV_OUT_OF_FILE: payload'а нет,
#: поэтому поле — не адрес («смещения нет», как NO_OFFSET у scale_offset).
#: Старый загрузчик видит здесь невыровненное смещение и отказывает, не читая
#: данные (задача 0118).
KV_NO_OFFSET = 0xFFFFFFFF

#: Кэп на суммарный размер KV-кэшей при загрузке (в C — VTM_MAX_KV_BYTES).
#: 1.5B-раскладка (ctx 2048, kv_dim 256, 28 слоёв) — 112 МиБ, запас ×4,5
#: (ctx 8192); 192-байтный файл не должен просить 4 ГиБ. Значения в двух
#: загрузчиках обязаны совпадать: tools/vtm_ref.py — эталон для C.
MAX_KV_BYTES = 512 << 20

#: Флаги контейнера (заголовок).
VFLAG_F32_SCALES = 0x0001  # масштабы тритов по умолчанию float32
VFLAG_CAUSAL = 0x0002      # модель причинная (LM)
#: KV-кэши (если есть) лежат вне файла: у их записей offset == KV_NO_OFFSET.
#: Контейнер с этим флагом обязан нести хотя бы один KV-кэш; старые
#: контейнеры присылают 0 и читаются прежним путём (задача 0118).
VFLAG_KV_OUT_OF_FILE = 0x0004

#: Стратегии SAMP (imm инструкции, задача 0051; см. docs/vtm.md).
SAMP_ARGMAX = 0
SAMP_TEMPERATURE = 1
SAMP_TOPK = 2
SAMP_TOPP = 3
VFLAG_KNOWN = VFLAG_F32_SCALES | VFLAG_CAUSAL | VFLAG_KV_OUT_OF_FILE

#: Предел int в C: позиция KVW/ATTN и context_window за него не выходят.
INT_MAX = 2_147_483_647

#: Контроль целостности (задача 0066). Блок лежит в мете под этим ключом:
#: {"algo": "sha256", "data_sha256": "<hex>", "meta_sha256": "<hex>"}.
#: data_sha256 — хеш секции данных; meta_sha256 — хеш JSON меты, в котором
#: значение meta_sha256 заменено 64 символами '0' (иначе хеш невычислим).
INTEGRITY_KEY = "integrity"
INTEGRITY_ALGO = "sha256"
INTEGRITY_DATA_FIELD = "data_sha256"
INTEGRITY_META_FIELD = "meta_sha256"
INTEGRITY_SKIP_ENV = "VTM_SKIP_HASH"
#: Включить запись хеша в контейнер, не меняя вызовы write_vtm: полезно
#: компилятору (tools/vtm_compile.py), который зовёт write_vtm без флага.
INTEGRITY_WRITE_ENV = "VTM_WRITE_HASH"

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


def storage_words(n_trits: int, storage: str = STORAGE_PACKED20) -> int:
    """Сколько 4-байтовых слов занимает массив тритов в хранении `storage`.

    packed20 округляет до 160-тритовых (32-байтовых) блоков, native — до слова
    (4 трита). Формулы обязаны совпадать с `vtm_trit_storage_words()` в
    `src/vtm/trit.c`.
    """
    if n_trits < 0:
        raise ValueError("число тритов не может быть отрицательным")
    if storage == STORAGE_PACKED20:
        return (n_trits + TRITS_PER_STORAGE_BLOCK - 1) // TRITS_PER_STORAGE_BLOCK * WORDS_PER_STORAGE_BLOCK
    if storage == STORAGE_NATIVE:
        return (n_trits + NATIVE_TRITS_PER_WORD - 1) // NATIVE_TRITS_PER_WORD
    raise ValueError(f"неизвестное хранение тритов: {storage!r}")


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

    Исключение — `flags & TFLAG_KV_CACHE` (K/V-кэш внимания, задача 0101):
    данных у него нет и в файл они не пишутся, но это не каркас, а полноценный
    тензор — читатель сам создаёт буфер нужного размера по `dims`.
    """

    name: str
    kind: int
    dims: tuple[int, ...]
    data: list[float] | None = None
    trits: list[int] | None = None
    scales: list[float] | PackedF32 | None = None
    scale_dtype: str = "float32"
    block_size: int = SCALE_BLOCK
    offset: int = 0
    scale_offset: int = 0
    #: Флаги записи таблицы (поле 50 в файле), см. TFLAG_KV_CACHE.
    flags: int = 0
    #: Хранение тритов (задача 0057): `packed20` (по умолчанию) или `native`.
    #: Осмысленно только для kind 0 (иначе — ошибка): у native TRT1 пишется
    #: version 2, а идентификатор дублируется в flags записи (биты 1..2).
    storage: str = STORAGE_PACKED20
    #: kind 4: знаковые значения int8 (-127..127). В файл идут байтами,
    #: в памяти плотные значения считаются как q · scale (`dense()`).
    #: У kind 5 это `PackedQ4` (окно в байты сегмента) — см. `Tensor.q4`.
    qdata: list[int] | PackedQ4 | PackedInt8 | None = None
    #: kind 5: Q4_1 — по одному min (нулю квантования) на группу 32;
    #: у Q4_0 (симметричный) `None`. Может быть `PackedF32` (без списка).
    q4_mins: list[float] | PackedF32 | None = None
    #: kind 5: False — Q4_0 (`(q-8)·scale`), True — Q4_1 (`q·scale + min`).
    q4_asymmetric: bool = False

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
        if self.storage not in TRIT_STORAGE_IDS:
            raise ValueError(
                f"тензор {self.name!r}: неизвестное хранение тритов "
                f"{self.storage!r}, допустимо packed20|native"
            )
        if self.kind != KIND_TRITS and self.storage != STORAGE_PACKED20:
            raise ValueError(
                f"тензор {self.name!r}: хранение {self.storage!r} осмысленно "
                "только для тритов (kind 0)"
            )
        if self.scale_dtype not in ("float32", "float16"):
            raise ValueError(f"тензор {self.name!r}: scale_dtype={self.scale_dtype!r}")
        if not 0 <= int(self.flags) <= 0xFFFF:
            raise ValueError(
                f"тензор {self.name!r}: flags={self.flags}, а поле таблицы — u16"
            )
        if self.flags & TFLAG_KV_CACHE:
            # Кэш существует только в таблице: содержимое не пишется и не
            # читается (его создаёт исполнитель), поэтому данные запрещены —
            # иначе легко решить, что они попадут в файл.
            if self.kind in (KIND_TRITS, KIND_INT8, KIND_Q4):
                raise ValueError(
                    f"тензор {self.name!r}: KV-кэш обязан быть fp-тензором (kind {self.kind})"
                )
            if (self.data is not None or self.trits is not None
                    or self.scales is not None or self.q4_mins is not None):
                raise ValueError(
                    f"тензор {self.name!r}: у KV-кэша нет данных — буфер создаёт "
                    "читатель (Tensor.kv_cache())"
                )
            return
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
        elif self.kind == KIND_INT8:
            if self.block_size != SCALE_BLOCK:
                raise ValueError(
                    f"тензор {self.name!r}: int8 хранит масштаб только на "
                    f"{SCALE_BLOCK} элементов"
                )
            if self.data is not None or self.trits is not None:
                raise ValueError(
                    f"тензор {self.name!r}: int8-тензор не имеет fp-данных/тритов"
                )
            if self.qdata is not None:
                if len(self.qdata) != self.n_elements:
                    raise ValueError(
                        f"тензор {self.name!r}: {len(self.qdata)} int8-значений, "
                        f"а dims дают {self.n_elements}"
                    )
                if not isinstance(self.qdata, PackedInt8):
                    # У `PackedInt8` диапазон проверен при разборе сегмента
                    # (`first_below`): обход 311 млн значений не нужен.
                    for value in self.qdata:
                        if not INT8_MIN <= int(value) <= INT8_MAX:
                            raise ValueError(
                                f"тензор {self.name!r}: int8-значение {value!r} "
                                f"вне {INT8_MIN}..{INT8_MAX}"
                            )
                if self.scales is None:
                    raise ValueError(f"тензор {self.name!r}: int8 без масштабов")
                expected = n_scale_blocks(self.n_elements, self.block_size)
                if len(self.scales) != expected:
                    raise ValueError(
                        f"тензор {self.name!r}: масштабов {len(self.scales)}, "
                        f"ожидалось {expected}"
                    )
                if isinstance(self.scales, PackedF32):
                    # Окно в байты сегмента: один проход, без списков.
                    self.scales.check_finite_nonneg(
                        f"тензор {self.name!r}: масштаб"
                    )
                else:
                    _check_finite(self.scales, f"тензор {self.name!r}: масштаб")
                    if any(s < 0 for s in self.scales):
                        raise ValueError(
                            f"тензор {self.name!r}: масштаб не может быть "
                            "отрицательным"
                        )
            elif self.scales is not None:
                raise ValueError(
                    f"тензор {self.name!r}: int8-масштабы без qdata"
                )
        elif self.kind == KIND_Q4:
            # По умолчанию поле block_size = 128 (канон тритов/int8); для Q4
            # канон — 32, поэтому дефолт приводим к нему, а иное значение
            # отвергаем.
            if self.block_size == SCALE_BLOCK:
                self.block_size = Q4_SCALE_BLOCK
            if self.block_size != Q4_SCALE_BLOCK:
                raise ValueError(
                    f"тензор {self.name!r}: Q4 хранит масштаб только на "
                    f"{Q4_SCALE_BLOCK} значений"
                )
            if self.data is not None or self.trits is not None:
                raise ValueError(
                    f"тензор {self.name!r}: Q4-тензор не имеет fp-данных/тритов"
                )
            if self.qdata is not None:
                if len(self.qdata) != self.n_elements:
                    raise ValueError(
                        f"тензор {self.name!r}: {len(self.qdata)} Q4-значений, "
                        f"а dims дают {self.n_elements}"
                    )
                if not isinstance(self.qdata, PackedQ4):
                    # У `PackedQ4` значение — ниббл байта, то есть 0..15 по
                    # построению; обход 2,77 млрд значений был бы минутами.
                    for value in self.qdata:
                        if not Q4_MIN <= int(value) <= Q4_MAX:
                            raise ValueError(
                                f"тензор {self.name!r}: Q4-значение {value!r} "
                                f"вне {Q4_MIN}..{Q4_MAX}"
                            )
                expected = n_scale_blocks(self.n_elements, Q4_SCALE_BLOCK)
                if self.scales is None:
                    raise ValueError(f"тензор {self.name!r}: Q4 без масштабов")
                if len(self.scales) != expected:
                    raise ValueError(
                        f"тензор {self.name!r}: масштабов {len(self.scales)}, "
                        f"ожидалось {expected}"
                    )
                if isinstance(self.scales, PackedF32):
                    # Окно в байты сегмента: один проход, без списков.
                    self.scales.check_finite_nonneg(
                        f"тензор {self.name!r}: масштаб"
                    )
                else:
                    _check_finite(self.scales, f"тензор {self.name!r}: масштаб")
                    if any(s < 0 for s in self.scales):
                        raise ValueError(
                            f"тензор {self.name!r}: масштаб не может быть "
                            "отрицательным"
                        )
                if self.q4_asymmetric:
                    if self.q4_mins is None:
                        raise ValueError(
                            f"тензор {self.name!r}: Q4_1 без min (нулей квантования)"
                        )
                    if len(self.q4_mins) != expected:
                        raise ValueError(
                            f"тензор {self.name!r}: Q4_1 min {len(self.q4_mins)}, "
                            f"ожидалось {expected}"
                        )
                    _check_finite(self.q4_mins, f"тензор {self.name!r}: Q4_1 min")
                elif self.q4_mins is not None:
                    raise ValueError(
                        f"тензор {self.name!r}: Q4_0 не имеет min — "
                        "поставьте q4_asymmetric=True для Q4_1"
                    )
            else:
                if self.scales is not None or self.q4_mins is not None:
                    raise ValueError(
                        f"тензор {self.name!r}: Q4-масштабы/min без qdata"
                    )
        else:
            if (self.trits is not None or self.scales is not None
                    or self.qdata is not None or self.q4_mins is not None):
                raise ValueError(
                    f"тензор {self.name!r}: fp-тензор не имеет тритов/int8/Q4"
                )
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

    @property
    def is_kv_cache(self) -> bool:
        """1 — K/V-кэш: payload'а в файле нет, буфер создаёт читатель."""
        return bool(getattr(self, "flags", 0) & TFLAG_KV_CACHE)

    @classmethod
    def kv_cache(cls, name: str, dims: Sequence[int]) -> "Tensor":
        """K/V-кэш внимания `(ctx, kv_dim)`: в контейнере только запись таблицы.

        Содержимое не пишется: кэш пуст на старте, его заполняет KVW. Загрузчик
        (C — арена, Python — список нулей) создаёт буфер по `dims`, поэтому
        тензор занимает `ctx × kv_dim × 4` байт только в памяти.
        """
        return cls(name=name, kind=KIND_F32, dims=tuple(dims), flags=TFLAG_KV_CACHE)

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
        storage: str = STORAGE_PACKED20,
    ) -> "Tensor":
        """Тритовый тензор. Без масштабов — absmax по тритам (0 или 1).

        `storage` — физическое хранение (задача 0057): `packed20` или `native`
        (1 знаковый байт на трит). Логические триты и результаты одни и те же.
        """
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
            storage=storage,
        )

    @classmethod
    def int8(
        cls,
        name: str,
        dims: Sequence[int],
        qdata: Iterable[int],
        scales: Sequence[float],
        scale_dtype: str = "float32",
    ) -> "Tensor":
        """int8-тензор (kind 4): q в -127..127, один масштаб на блок 128."""
        return cls(
            name=name,
            kind=KIND_INT8,
            dims=tuple(dims),
            # Окна (`PackedInt8`/`PackedF32`) не материализуются: у 3B таблица
            # словаря — 311 млн значений, это ~2,5 ГиБ списком.
            qdata=(
                qdata if isinstance(qdata, PackedInt8)
                else [int(value) for value in qdata]
            ),
            scales=(
                scales if isinstance(scales, PackedF32)
                else [_f32(float(value)) for value in scales]
            ),
            scale_dtype=scale_dtype,
        )

    @classmethod
    def q4(
        cls,
        name: str,
        dims: Sequence[int],
        qdata: Iterable[int],
        scales: Sequence[float],
        scale_dtype: str = "float32",
        *,
        asymmetric: bool = False,
        mins: Sequence[float] | None = None,
    ) -> "Tensor":
        """Q4-тензор (kind 5): нибблы 0..15, масштаб на группу 32.

        `asymmetric=False` — Q4_0 (`w = (q-8)·scale`); `True` — Q4_1
        (`w = q·scale + min`, `mins` обязательны и по числу равны scales).
        """
        # Ленивые окна (`PackedQ4`/`PackedF32`) не материализуются: у 3B Q4
        # список нибблов — это ~22 ГиБ, список fp32-масштабов — ещё ~2,8 ГиБ.
        # Окна уже ссылаются на байты сегмента, а их числа — точные float32,
        # поэтому `_f32` для них не меняет значение.
        return cls(
            name=name,
            kind=KIND_Q4,
            dims=tuple(dims),
            qdata=(
                qdata if isinstance(qdata, PackedQ4)
                else [int(value) for value in qdata]
            ),
            scales=(
                scales if isinstance(scales, PackedF32)
                else [_f32(float(value)) for value in scales]
            ),
            scale_dtype=scale_dtype,
            q4_asymmetric=bool(asymmetric),
            q4_mins=(
                mins if isinstance(mins, PackedF32)
                else ([_f32(float(value)) for value in mins]
                      if mins is not None else None)
            ),
        )

    @classmethod
    def placeholder(cls, name: str, dims: Sequence[int], kind: int = KIND_F32) -> "Tensor":
        """Тензор без данных — для каркаса графа (веса ещё не загружены)."""
        return cls(name=name, kind=kind, dims=tuple(dims))

    # -- данные --------------------------------------------------------------

    def dense(self) -> list[float]:
        """Плотные значения: триты — `trit·scale`, int8 — `q·scale`, fp — как есть."""
        if self.kind == KIND_TRITS:
            if self.trits is None or self.scales is None:
                raise ValueError(f"тензор {self.name!r}: нет данных тритов")
            block = self.block_size
            scales = self.scales
            return [trit * scales[i // block] for i, trit in enumerate(self.trits)]
        if self.kind == KIND_INT8:
            if self.qdata is None or self.scales is None:
                raise ValueError(f"тензор {self.name!r}: нет int8-данных")
            block = self.block_size
            scales = self.scales
            # _f32(...) — та же единственная операция float32, что в C
            # (int8 → float, умножение на float32-масштаб).
            return [_f32(value * scales[i // block])
                    for i, value in enumerate(self.qdata)]
        if self.kind == KIND_Q4:
            if self.qdata is None or self.scales is None:
                raise ValueError(f"тензор {self.name!r}: нет Q4-данных")
            if self.q4_asymmetric and self.q4_mins is None:
                raise ValueError(f"тензор {self.name!r}: Q4_1 без min")
            block = Q4_SCALE_BLOCK
            scales = self.scales
            mins = self.q4_mins
            values: list[float] = []
            for index, nibble in enumerate(self.qdata):
                group = index // block
                if self.q4_asymmetric:
                    if mins is None:  # pragma: no cover - проверено в __post_init__
                        raise ValueError(f"тензор {self.name!r}: Q4_1 без min")
                    values.append(_f32(nibble * scales[group] + mins[group]))
                else:
                    values.append(_f32((nibble - Q4_ZERO_NIBBLE) * scales[group]))
            return values
        if self.data is None:
            raise ValueError(f"тензор {self.name!r}: нет данных")
        return list(self.data)


def quantize_int8(
    values: Iterable[float], block_size: int = SCALE_BLOCK
) -> tuple[list[int], list[float]]:
    """Квантовать веса в (int8, масштабы по 128) — правило `Int8Embedding`.

    `scale = absmax(блок) / 127`, `q = clamp(round(w/scale), -127, 127)`;
    у нулевого блока масштаб `1e-8/127` и все q = 0 (так делает QAT:
    `qat/model.py:Int8Embedding`). Округление — к ближайшему чётному, как
    `torch.round`; отношение считается в float32, чтобы совпасть с QAT.
    """
    vals = [_f32(float(v)) for v in values]
    _check_finite(vals, "quantize_int8")
    scales: list[float] = []
    for start in range(0, len(vals), block_size):
        block = vals[start:start + block_size]
        absmax = max(abs(v) for v in block) if block else 0.0
        scales.append(_f32(absmax / 127.0) if absmax > 0 else _f32(1e-8 / 127.0))
    qdata: list[int] = []
    for i, value in enumerate(vals):
        scale = scales[i // block_size]
        ratio = _f32(value / scale) if scale else 0.0
        qdata.append(max(INT8_MIN, min(INT8_MAX, int(round(ratio)))))
    return qdata, scales


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


#: Порция при потоковой записи (байт); 1 МиБ — компромисс между числом
#: системных вызовов и лишней памятью. Объявлена здесь, потому что её
#: используют и ленивые Q4-последовательности (`PackedQ4.iter_packed`).
STREAM_CHUNK = 1 << 20


def _pack_q4_group(values: Sequence[int], pad: int) -> bytes:
    """Одна группа из ≤32 нибблов: байт `j` = значение `j` + `j+16 << 4`."""
    out = bytearray(Q4_GROUP_BYTES)
    n = len(values)
    for within in range(Q4_SCALE_BLOCK):
        value = int(values[within]) if within < n else int(pad)
        offset = within & 15
        if within < 16:
            out[offset] |= value
        else:
            out[offset] |= value << 4
    return bytes(out)


class PackedF32(Sequence[float]):
    """Последовательность fp32 поверх байт контейнера — без list на весь тензор.

    У 3B Q4 масштаб на каждую группу 32 — это 86,6 млн значений: `list[float]`
    занял бы ~2,8 ГиБ, а представление здесь — просто окно в те же байты
    (масштабы уже лежат в сегменте). Чтение отдаёт float, `to_bytes()` —
    исходные байты (little-endian), поэтому сегмент переписывается побайтово.
    """

    __slots__ = ("_view", "_raw", "_n_values")

    def __init__(self, data: Any, offset: int, n_values: int) -> None:
        n_values = int(n_values)
        if n_values < 0:
            raise ValueError("PackedF32: отрицательное число значений")
        needle = n_values * 4
        chunk = memoryview(data)[offset:offset + needle]
        if len(chunk) != needle:
            raise ValueError(
                f"PackedF32: обрезано {len(chunk)} из {needle} байт"
            )
        self._raw = chunk
        if sys.byteorder == "little":
            self._view = chunk.cast("f")
        else:  # pragma: no cover - экзотическая платформа
            import array as _array

            decoded = _array.array("f")
            decoded.frombytes(chunk)
            decoded.byteswap()
            self._view = decoded
        self._n_values = n_values

    def __len__(self) -> int:
        return self._n_values

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, slice):
            return [float(value) for value in self._view[index]]
        return float(self._view[index])

    def __iter__(self):
        return iter(self._view)

    def to_bytes(self) -> bytes:
        """Байты fp32 в порядке little-endian (ровно как в контейнере)."""
        # `_raw` — исходный срез байт сегмента: на любом порядке байт хоста
        # это ровно те байты, что лежат в файле.
        return self._raw.tobytes()

    def iter_chunks(self, chunk: int = 1 << 20):
        """Куски исходных байт без копии — для потоковой записи."""
        step = max(4, (int(chunk) // 4) * 4)
        raw = self._raw
        needle = self._n_values * 4
        for start in range(0, needle, step):
            yield raw[start:min(start + step, needle)]

    def check_finite_nonneg(self, what: str) -> None:
        """Один проход: конечность и неотрицательность (как в Tensor/сегменте)."""
        bad_finite = -1
        bad_negative = -1
        for index, value in enumerate(self._view):
            if bad_finite < 0 and not math.isfinite(value):
                bad_finite = index
            elif bad_negative < 0 and value < 0:
                bad_negative = index
            if bad_finite >= 0 and bad_negative >= 0:
                break
        if bad_finite >= 0:
            raise ValueError(
                f"{what}: неконечное значение {float(self._view[bad_finite])!r}"
            )
        if bad_negative >= 0:
            raise ValueError(f"{what} не может быть отрицательным")

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, (PackedF32, list, tuple)):
            if len(other) != self._n_values:
                return False
            return all(left == right for left, right in zip(self, other))
        return NotImplemented

    def __repr__(self) -> str:  # pragma: no cover - только для отладки
        return f"PackedF32(n={self._n_values})"


class PackedQ4(Sequence[int]):
    """Нибблы Q4 поверх уже упакованных байт: `list[int]` на весь тензор не нужен.

    У 3B Q4 это 2,77 млрд значений (~22 ГиБ списком Python) — поэтому
    `decode_q4_segment_ex` отдаёт это окно, а не список. Раскладка групп —
    как в `pack_q4`/GGML Q4_0, поэтому запись сегмента — копирование байт:
    ни распаковка в нибблы, ни повторная упаковка не нужны.

    `Sequence[int]` сохраняет совместимость: `len`, индексация, срезы,
    итерация и сравнение со списком работают как раньше (для мелких тензоров
    разница не видна).
    """

    __slots__ = ("_packed", "_n_values")

    def __init__(self, packed: Any, n_values: int) -> None:
        n_values = int(n_values)
        if n_values < 0:
            raise ValueError("Q4: отрицательное число значений")
        n_groups = n_scale_blocks(n_values, Q4_SCALE_BLOCK)
        needle = n_groups * Q4_GROUP_BYTES
        view = memoryview(packed)[:needle]
        if len(view) != needle:
            raise ValueError(
                f"Q4-данные обрезаны: {len(view)} байт, ожидалось {needle} "
                f"({n_groups} групп по {Q4_SCALE_BLOCK})"
            )
        self._packed = view
        self._n_values = n_values

    @classmethod
    def from_segment(cls, data: Any, offset: int, n_values: int) -> "PackedQ4":
        """Окно на упакованные нибблы сегмента (без копирования байт)."""
        n_groups = n_scale_blocks(int(n_values), Q4_SCALE_BLOCK)
        needle = n_groups * Q4_GROUP_BYTES
        view = memoryview(data)[offset:offset + needle]
        if len(view) != needle:
            raise ValueError(
                f"Q4-данные обрезаны: {len(view)} байт, ожидалось {needle}"
            )
        return cls(view, n_values)

    def __len__(self) -> int:
        return self._n_values

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(self._n_values))]
        index = int(index)
        n_values = self._n_values
        if index < 0:
            index += n_values
        if not 0 <= index < n_values:
            raise IndexError(f"Q4: индекс {index} вне 0..{n_values - 1}")
        within = index & (Q4_SCALE_BLOCK - 1)
        byte = self._packed[(index >> 5) * Q4_GROUP_BYTES + (within & 15)]
        return (byte & 0x0F) if within < 16 else (byte >> 4)

    def __iter__(self):
        packed = self._packed
        n_values = self._n_values
        full_groups = n_values // Q4_SCALE_BLOCK
        for group in range(full_groups):
            base = group * Q4_GROUP_BYTES
            # Сначала младшие нибблы (значения 0..15), затем старшие (16..31):
            # порядок значений, а не порядок байт.
            for within in range(16):
                yield packed[base + within] & 0x0F
            for within in range(16):
                yield packed[base + within] >> 4
        for index in range(full_groups * Q4_SCALE_BLOCK, n_values):
            yield self[index]

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, (PackedQ4, list, tuple)):
            if len(other) != self._n_values:
                return False
            return all(left == right for left, right in zip(self, other))
        return NotImplemented

    def __repr__(self) -> str:  # pragma: no cover - только для отладки
        return f"PackedQ4(n={self._n_values})"

    # -- запись ---------------------------------------------------------------

    def _last_group_bytes(self) -> bytes:
        """Последняя группа с padding-нибблом 8 — ровно как `pack_q4`."""
        groups = n_scale_blocks(self._n_values, Q4_SCALE_BLOCK)
        start = (groups - 1) * Q4_SCALE_BLOCK
        return _pack_q4_group(
            [self[index] for index in range(start, self._n_values)],
            Q4_ZERO_NIBBLE,
        )

    def to_packed(self) -> bytes:
        """Упакованные байты (с padding'ом хвоста, как у `pack_q4`)."""
        if self._n_values % Q4_SCALE_BLOCK == 0:
            return self._packed.tobytes()
        out = bytearray(self._packed.tobytes())
        out[-Q4_GROUP_BYTES:] = self._last_group_bytes()
        return bytes(out)

    def iter_packed(self, chunk: int = STREAM_CHUNK):
        """Куски упакованных байт без копии — для потоковой записи сегмента."""
        full_groups = self._n_values // Q4_SCALE_BLOCK
        step = max(1, int(chunk) // Q4_GROUP_BYTES)
        group = 0
        while group < full_groups:
            take = min(step, full_groups - group)
            yield self._packed[
                group * Q4_GROUP_BYTES:(group + take) * Q4_GROUP_BYTES
            ]
            group += take
        if self._n_values % Q4_SCALE_BLOCK:
            yield self._last_group_bytes()


class PackedInt8(Sequence[int]):
    """int8-значения поверх байт сегмента: без `list[int]` на весь тензор.

    У 3B таблица словаря — это 311 млн int8-значений: `list[int]` занял бы
    ~2,5 ГиБ (плюс столько же при копировании в `Tensor.int8`), хотя байты
    уже лежат в `non_ternary.bin`. Окно отдаёт те же числа и те же байты.
    """

    __slots__ = ("_view", "_n_values")

    def __init__(self, data: Any, offset: int, n_values: int) -> None:
        n_values = int(n_values)
        if n_values < 0:
            raise ValueError("INT8: отрицательное число значений")
        chunk = memoryview(data)[offset:offset + n_values]
        if len(chunk) != n_values:
            raise ValueError(
                f"INT8-данные обрезаны: {len(chunk)} байт, ожидалось {n_values}"
            )
        # 'b' — знаковый байт: -128..127, как struct '<b'.
        self._view = chunk.cast("b")
        self._n_values = n_values

    @classmethod
    def from_segment(cls, data: Any, offset: int,
                     n_values: int) -> "PackedInt8":
        return cls(data, offset, n_values)

    def __len__(self) -> int:
        return self._n_values

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, slice):
            return [int(value) for value in self._view[index]]
        return int(self._view[index])

    def __iter__(self):
        return (int(value) for value in self._view)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, (PackedInt8, list, tuple)):
            if len(other) != self._n_values:
                return False
            return all(left == right for left, right in zip(self, other))
        return NotImplemented

    def __repr__(self) -> str:  # pragma: no cover - только для отладки
        return f"PackedInt8(n={self._n_values})"

    def to_bytes(self) -> bytes:
        """Байты q-значений (дополнительный код, как в контейнере)."""
        return self._view.tobytes()

    def iter_chunks(self, chunk: int = STREAM_CHUNK):
        """Куски q-байт без копии — для потоковой записи."""
        step = max(1, int(chunk))
        view = self._view
        for start in range(0, self._n_values, step):
            yield view[start:start + step]

    def first_below(self, limit: int) -> int:
        """Индекс первого значения < limit (или -1): для сообщения об ошибке."""
        view = self._view
        for index in range(self._n_values):
            if view[index] < limit:
                return index
        return -1


def pack_q4(values: Iterable[int], pad: int = Q4_ZERO_NIBBLE) -> bytes:
    """Упаковать нибблы 0..15 в группы по 32 значения (16 байт на группу).

    Раскладка — как в GGML Q4_0 и в ядрах 0152: байт `j` группы хранит
    значение `j` в младшем ниббле, `j+16` — в старшем. Хвост последней
    группы дополняется нибблом `pad` (по умолчанию 8 — ноль Q4_0).

    Память — одна группа на ходу (у `PackedQ4` — вообще копия байт): список
    значений на весь тензор не материализуется, иначе 3B Q4 требует ~22 ГиБ.
    """
    if isinstance(values, PackedQ4) and int(pad) == Q4_ZERO_NIBBLE:
        return values.to_packed()
    if not Q4_MIN <= int(pad) <= Q4_MAX:
        raise ValueError(f"Q4: padding-ниббл {pad} вне {Q4_MIN}..{Q4_MAX}")
    out = bytearray()
    group: list[int] = []
    index = 0
    for value in values:
        value = int(value)
        if not Q4_MIN <= value <= Q4_MAX:
            raise ValueError(
                f"Q4: значение #{index} = {value} вне {Q4_MIN}..{Q4_MAX}"
            )
        group.append(value)
        index += 1
        if len(group) == Q4_SCALE_BLOCK:
            out += _pack_q4_group(group, pad)
            group.clear()
    if group:
        out += _pack_q4_group(group, pad)
    return bytes(out)


def unpack_q4(data: bytes, n_values: int) -> list[int]:
    """Развернуть `n_values` нибблов Q4 (лишний padding игнорируется)."""
    if n_values < 0:
        raise ValueError("Q4: отрицательное число значений")
    n_groups = n_scale_blocks(n_values, Q4_SCALE_BLOCK)
    need = n_groups * Q4_GROUP_BYTES
    if len(data) < need:
        raise ValueError(
            f"Q4-данные обрезаны: {len(data)} байт, ожидалось {need} "
            f"({n_groups} групп по {Q4_SCALE_BLOCK})"
        )
    values: list[int] = []
    for index in range(n_values):
        group, within = divmod(index, Q4_SCALE_BLOCK)
        byte = data[group * Q4_GROUP_BYTES + (within & 15)]
        values.append((byte & 0x0F) if within < 16 else (byte >> 4))
    return values


def quantize_q4(
    values: Iterable[float],
    block_size: int = Q4_SCALE_BLOCK,
    *,
    symmetric: bool = True,
) -> tuple[list[int], list[float]] | tuple[list[int], list[float], list[float]]:
    """Квантовать веса в Q4 (нибблы 0..15, масштаб на группу 32).

    Симметричный Q4_0: `scale = absmax/7`, `q = clamp(round_half_away(w/scale),
    -8, 7) + 8` — крайние ±absmax представимы, ноль кодируется нибблом 8.
    Асимметричный Q4_1: `scale = (max-min)/15`, `min` — настоящий минимум
    блока, `q = clamp(round_half_away((w-min)/scale), 0, 15)`, `w = q·scale+min`.
    У нулевого блока масштаб 0 (Q4_1 — ещё и min = значение блока). Для Q4_0
    возвращается пара `(qdata, scales)`, для Q4_1 — тройка
    `(qdata, scales, mins)`. Числа округляются до float32, чтобы совпасть с
    контейнером после чтения с диска.
    """
    if block_size != Q4_SCALE_BLOCK:
        raise ValueError(
            f"Q4 хранит масштаб только на {Q4_SCALE_BLOCK} значений"
        )
    vals = [_f32(float(v)) for v in values]
    _check_finite(vals, "quantize_q4")
    qdata: list[int] = []
    scales: list[float] = []
    mins_list: list[float] = []
    for start in range(0, len(vals), block_size):
        block = vals[start:start + block_size]
        if symmetric:
            absmax = max(abs(v) for v in block) if block else 0.0
            scale = _f32(absmax / 7.0) if absmax > 0 else 0.0
            scales.append(scale)
            for value in block:
                if scale == 0.0:
                    qdata.append(Q4_ZERO_NIBBLE)
                else:
                    signed = max(-8, min(7, round_half_away(
                        _f32(value / scale)
                    )))
                    qdata.append(signed + Q4_ZERO_NIBBLE)
        else:
            lo = min(block) if block else 0.0
            hi = max(block) if block else 0.0
            if hi == lo:
                scale = 0.0
                zero = _f32(lo)
            else:
                scale = _f32((hi - lo) / 15.0)
                zero = _f32(lo)
            scales.append(scale)
            mins_list.append(zero)
            for value in block:
                if scale == 0.0:
                    qdata.append(0)
                else:
                    qdata.append(max(0, min(15, round_half_away(
                        _f32((value - zero) / scale)
                    ))))
    if symmetric:
        return qdata, scales
    return qdata, scales, mins_list


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
    """Размер payload'а тензора в байтах (без выравнивания).

    У KV-кэша payload'а нет (TFLAG_KV_CACHE) — 0 байт: место под кэш
    выделяется при загрузке, а не хранится в файле (задача 0101).
    """
    if getattr(tensor, "flags", 0) & TFLAG_KV_CACHE:
        return 0
    if tensor.kind == KIND_TRITS:
        n_values = tensor.n_elements
        scale_bytes = 4 if tensor.scale_dtype == "float32" else 2
        return (
            TRT1_HEADER_SIZE
            + n_scale_blocks(n_values, tensor.block_size) * scale_bytes
            + storage_words(n_values, tensor.storage) * 4
        )
    if tensor.kind == KIND_INT8:
        n_values = tensor.n_elements
        scale_bytes = 4 if tensor.scale_dtype == "float32" else 2
        return (
            INT8_HEADER_SIZE
            + n_scale_blocks(n_values, INT8_SCALE_BLOCK) * scale_bytes
            + n_values
        )
    if tensor.kind == KIND_Q4:
        n_values = tensor.n_elements
        n_groups = n_scale_blocks(n_values, Q4_SCALE_BLOCK)
        scale_bytes = 4 if tensor.scale_dtype == "float32" else 2
        aux_bytes = n_groups * scale_bytes * (2 if tensor.q4_asymmetric else 1)
        header = Q4_HEADER_SIZE + (4 if tensor.q4_asymmetric else 0)
        return header + aux_bytes + n_groups * Q4_GROUP_BYTES
    if tensor.kind == KIND_F32:
        return tensor.n_elements * 4
    if tensor.kind in (KIND_F16, KIND_BF16):
        return tensor.n_elements * 2
    raise ValueError(f"тензор {tensor.name!r}: неизвестный kind={tensor.kind}")


def _int8_segment_header(tensor: Tensor) -> tuple[bytes, int]:
    """Проверить int8-тензор и собрать заголовок сегмента (без данных)."""
    if tensor.kind != KIND_INT8:
        raise ValueError(f"тензор {tensor.name!r}: это не int8")
    if tensor.qdata is None or tensor.scales is None:
        raise ValueError(
            f"тензор {tensor.name!r}: нет int8-данных — граф собран без весов, "
            "записывать в контейнер нечего"
        )
    n_values = tensor.n_elements
    n_scales = n_scale_blocks(n_values, INT8_SCALE_BLOCK)
    if len(tensor.scales) != n_scales:
        raise ValueError(
            f"тензор {tensor.name!r}: масштабов {len(tensor.scales)}, ожидалось {n_scales}"
        )
    if len(tensor.qdata) != n_values:
        raise ValueError(
            f"тензор {tensor.name!r}: {len(tensor.qdata)} int8-значений, "
            f"а dims дают {n_values}"
        )
    flags = 0x01 if tensor.scale_dtype == "float32" else 0x00
    header = _TRT1_STRUCT.pack(INT8_MAGIC, INT8_VERSION, flags, n_values, n_scales)
    return header, n_scales


def _scales_bytes(values: Sequence[float], n_scales: int, dtype: str,
                  name: str) -> bytes:
    """Байты масштабов: у `PackedF32` — те же байты без перепаковки."""
    if isinstance(values, PackedF32):
        return values.to_bytes()
    scale_code = "f" if dtype == "float32" else "e"
    try:
        return struct.pack(f"<{n_scales}{scale_code}", *values)
    except (struct.error, OverflowError) as exc:
        raise ValueError(
            f"тензор {name!r}: масштаб не влезает в {dtype}"
        ) from exc


def iter_int8_payload(tensor: Tensor, chunk: int = STREAM_CHUNK):
    """Куски payload'а INT8-тензора — без `bytes` на весь сегмент."""
    header, n_scales = _int8_segment_header(tensor)
    yield header
    scales = tensor.scales
    assert scales is not None  # проверено в _int8_segment_header
    if isinstance(scales, PackedF32):
        yield from scales.iter_chunks(chunk)
    else:
        yield _scales_bytes(scales, n_scales, tensor.scale_dtype, tensor.name)
    qdata = tensor.qdata
    assert qdata is not None  # проверено в _int8_segment_header
    if isinstance(qdata, PackedInt8):
        yield from qdata.iter_chunks(chunk)
    else:
        # q — знаковые байты: -1 -> 0xFF, 127 -> 0x7F.
        yield bytes(value & 0xFF for value in qdata)


def encode_int8_segment(tensor: Tensor) -> bytes:
    """Собрать INT8-сегмент: 18-байтовый заголовок + масштабы + q-байты.

    Для больших тензоров предпочтителен `iter_int8_payload`: здесь по
    определению получается один `bytes` на весь сегмент.
    """
    return b"".join(iter_int8_payload(tensor))


def decode_int8_segment(
    data: bytes,
) -> tuple[PackedInt8, PackedF32, str]:
    """Разобрать INT8-сегмент; лишние байты в конце (padding) игнорируются.

    q и масштабы — окна в байты `data` (`PackedInt8`/`PackedF32`), а не
    списки: таблица словаря 3B — это 311 млн значений (~2,5 ГиБ списком),
    хотя в `non_ternary.bin` они уже лежат байтами. API совместим со
    списком (`len`, индексация, итерация, сравнение).
    """
    if len(data) < INT8_HEADER_SIZE:
        raise ValueError(
            f"INT8-сегмент короче заголовка: {len(data)} < {INT8_HEADER_SIZE}"
        )
    magic, version, flags, n_values, n_scales = _TRT1_STRUCT.unpack_from(data, 0)
    if magic != INT8_MAGIC:
        raise ValueError(f"чужой magic INT8-сегмента: {magic!r}")
    if version != INT8_VERSION:
        raise ValueError(f"неизвестная версия INT8-сегмента: {version}")
    if flags & ~0x01:
        raise ValueError(f"неизвестные флаги INT8-сегмента: 0x{flags:02x}")
    expected_scales = n_scale_blocks(n_values, INT8_SCALE_BLOCK)
    if n_scales != expected_scales:
        raise ValueError(
            f"в заголовке INT8 {n_scales} масштабов, ожидалось {expected_scales} "
            f"для {n_values} элементов"
        )
    scale_dtype = "float32" if flags & 0x01 else "float16"
    scale_size = 4 if scale_dtype == "float32" else 2
    expected_len = INT8_HEADER_SIZE + n_scales * scale_size + n_values
    if len(data) < expected_len:
        raise ValueError(
            f"INT8-сегмент обрезан: {len(data)} байт, ожидалось {expected_len}"
        )
    if scale_dtype == "float32":
        scales: PackedF32 | list[float] = PackedF32(
            data, INT8_HEADER_SIZE, n_scales
        )
    else:
        scales = [
            float(value) for value in struct.unpack_from(
                f"<{n_scales}e", data, INT8_HEADER_SIZE
            )
        ]
    q_offset = INT8_HEADER_SIZE + n_scales * scale_size
    qdata = PackedInt8.from_segment(data, q_offset, n_values)
    # Проверка диапазона — одно чтение min (затем поиск индекса только при
    # ошибке): побайтовый обход 311 млн значений на 3B не нужен.
    first_bad = qdata.first_below(INT8_MIN)
    if first_bad >= 0:
        raise ValueError(
            f"INT8-сегмент: q #{first_bad} = {qdata[first_bad]} "
            f"меньше {INT8_MIN}"
        )
    return qdata, scales, scale_dtype


def _q4_segment_header(tensor: Tensor) -> tuple[bytes, int]:
    """Проверить Q4-тензор и собрать заголовок сегмента (без данных).

    Возвращает `(header, n_groups)`. Отдельно от `encode_q4_segment`, чтобы
    потоковый писатель мог проверить тензор до создания файла и писать
    масштабы/нибблы кусками.
    """
    if tensor.kind != KIND_Q4:
        raise ValueError(f"тензор {tensor.name!r}: это не Q4")
    if tensor.qdata is None or tensor.scales is None:
        raise ValueError(
            f"тензор {tensor.name!r}: нет Q4-данных — граф собран без весов, "
            "записывать в контейнер нечего"
        )
    n_values = tensor.n_elements
    n_groups = n_scale_blocks(n_values, Q4_SCALE_BLOCK)
    if len(tensor.scales) != n_groups:
        raise ValueError(
            f"тензор {tensor.name!r}: масштабов {len(tensor.scales)}, "
            f"ожидалось {n_groups}"
        )
    if len(tensor.qdata) != n_values:
        raise ValueError(
            f"тензор {tensor.name!r}: {len(tensor.qdata)} Q4-значений, "
            f"а dims дают {n_values}"
        )
    asymmetric = bool(tensor.q4_asymmetric)
    if asymmetric:
        if tensor.q4_mins is None:
            raise ValueError(
                f"тензор {tensor.name!r}: Q4_1 без min — тихо писать Q4_0 нельзя"
            )
        if len(tensor.q4_mins) != n_groups:
            raise ValueError(
                f"тензор {tensor.name!r}: Q4_1 min {len(tensor.q4_mins)}, "
                f"ожидалось {n_groups}"
            )
    elif tensor.q4_mins is not None:
        raise ValueError(
            f"тензор {tensor.name!r}: Q4_0 не имеет min (q4_asymmetric=False)"
        )
    if tensor.scale_dtype not in ("float32", "float16"):
        raise ValueError(
            f"тензор {tensor.name!r}: scale_dtype={tensor.scale_dtype!r}"
        )
    magic = Q4_MAGIC_ASYM if asymmetric else Q4_MAGIC
    flags = Q4_FLAG_F32_SCALES if tensor.scale_dtype == "float32" else 0
    header = _TRT1_STRUCT.pack(magic, Q4_VERSION, flags, n_values, n_groups)
    if asymmetric:
        # Явное число min: читатель обязан сверить его с ceil(n/32), иначе
        # отсутствие min можно спутать с выравнивающим padding'ом.
        header += struct.pack("<I", n_groups)
    return header, n_groups


def _q4_aux_bytes(values: Sequence[float], n_groups: int, dtype: str,
                  name: str, what: str) -> bytes:
    """Байты массива масштабов/min: у `PackedF32` — без перепаковки."""
    if isinstance(values, PackedF32):
        return values.to_bytes()
    scale_code = "f" if dtype == "float32" else "e"
    try:
        return struct.pack(f"<{n_groups}{scale_code}", *values)
    except (struct.error, OverflowError) as exc:
        raise ValueError(
            f"тензор {name!r}: {what} не влезает в {dtype}"
        ) from exc


def iter_q4_payload(tensor: Tensor, chunk: int = STREAM_CHUNK):
    """Куски payload'а Q4-тензора — без `bytes` на весь сегмент.

    Порядок байт тот же, что у `encode_q4_segment`: заголовок, масштабы,
    [min Q4_1], упакованные нибблы. У ленивого `PackedQ4` нибблы пишутся
    прямо из байт сегмента (окно, не копия), у списка — пакуются по группам.
    """
    header, n_groups = _q4_segment_header(tensor)
    yield header
    dtype = tensor.scale_dtype
    if isinstance(tensor.scales, PackedF32):
        yield from tensor.scales.iter_chunks(chunk)
    else:
        yield _q4_aux_bytes(tensor.scales, n_groups, dtype, tensor.name, "масштаб")
    if tensor.q4_asymmetric:
        mins = tensor.q4_mins
        assert mins is not None  # проверено в _q4_segment_header
        if isinstance(mins, PackedF32):
            yield from mins.iter_chunks(chunk)
        else:
            yield _q4_aux_bytes(mins, n_groups, dtype, tensor.name, "min")
    qdata = tensor.qdata
    assert qdata is not None  # проверено в _q4_segment_header
    if isinstance(qdata, PackedQ4):
        yield from qdata.iter_packed(chunk)
    else:
        yield pack_q4(qdata)


def encode_q4_segment(tensor: Tensor) -> bytes:
    """Собрать Q4-сегмент: заголовок, масштабы, [min Q4_1], упакованные нибблы.

    Для больших тензоров предпочтителен `iter_q4_payload` (потоковая запись):
    эта функция по определению возвращает один `bytes` на весь сегмент.
    """
    return b"".join(iter_q4_payload(tensor))


def decode_q4_segment_ex(
    data: bytes,
) -> tuple[PackedQ4, PackedF32, str, bool, PackedF32 | None]:
    """Разобрать Q4-сегмент: нибблы, масштабы, dtype, asym, min'ы.

    Нибблы и масштабы — **окна в байты `data`** (`PackedQ4`/`PackedF32`), а не
    списки: у 3B Q4 это 2,77 млрд нибблов и 86,6 млн масштабов, то есть ~22 ГиБ
    и ~2,8 ГиБ списками Python. API совместим со списком (`len`, индексация,
    итерация, сравнение), поэтому мелкие тесты не меняются. Материализовать
    список можно явно: `list(qdata)`.

    Лишние байты в конце (выравнивание payload'а до 32) игнорируются.
    """
    if len(data) < Q4_HEADER_SIZE:
        raise ValueError(
            f"Q4-сегмент короче заголовка: {len(data)} < {Q4_HEADER_SIZE}"
        )
    magic, version, flags, n_values, n_groups = _TRT1_STRUCT.unpack_from(data, 0)
    if magic not in (Q4_MAGIC, Q4_MAGIC_ASYM):
        raise ValueError(f"чужой magic Q4-сегмента: {magic!r}")
    if version != Q4_VERSION:
        raise ValueError(f"неизвестная версия Q4-сегмента: {version}")
    if flags & ~Q4_FLAG_F32_SCALES:
        raise ValueError(f"неизвестные флаги Q4-сегмента: 0x{flags:02x}")
    asymmetric = magic == Q4_MAGIC_ASYM
    expected_groups = n_scale_blocks(n_values, Q4_SCALE_BLOCK)
    if n_groups != expected_groups:
        raise ValueError(
            f"в заголовке Q4 {n_groups} групп, ожидалось {expected_groups} "
            f"для {n_values} значений (группа {Q4_SCALE_BLOCK})"
        )
    header_size = Q4_HEADER_SIZE
    if asymmetric:
        if len(data) < Q4_ASYM_HEADER_SIZE:
            raise ValueError(
                f"Q4_1-сегмент короче заголовка: {len(data)} < "
                f"{Q4_ASYM_HEADER_SIZE} (нет n_mins)"
            )
        n_mins = struct.unpack_from("<I", data, Q4_HEADER_SIZE)[0]
        if n_mins != expected_groups:
            raise ValueError(
                f"Q4_1: n_mins={n_mins}, ожидалось {expected_groups} "
                f"(по одному min на группу {Q4_SCALE_BLOCK}) — тихо считать "
                "Q4_0 нельзя"
            )
        header_size = Q4_ASYM_HEADER_SIZE
    scale_dtype = "float32" if flags & Q4_FLAG_F32_SCALES else "float16"
    scale_size = 4 if scale_dtype == "float32" else 2
    scales_offset = header_size
    mins_offset = scales_offset + n_groups * scale_size
    q_offset = mins_offset + (n_groups * scale_size if asymmetric else 0)
    expected_len = q_offset + n_groups * Q4_GROUP_BYTES
    if len(data) < expected_len:
        raise ValueError(
            f"Q4-сегмент обрезан: {len(data)} байт, ожидалось {expected_len}"
        )
    if scale_dtype == "float32":
        scales: PackedF32 | list[float] = PackedF32(data, scales_offset, n_groups)
        mins: PackedF32 | list[float] | None = (
            PackedF32(data, mins_offset, n_groups) if asymmetric else None
        )
    else:
        # fp16-масштабы: 2 байта на значение, окно без копии не сделать тем же
        # `cast('f')`; их число у 3B — 86,6 млн, но fp16 здесь ~173 МиБ,
        # поэтому список допустим только для небольших тензоров.
        scale_code = "e"
        scales = [float(s) for s in struct.unpack_from(
            f"<{n_groups}{scale_code}", data, scales_offset
        )]
        mins = (
            [float(s) for s in struct.unpack_from(
                f"<{n_groups}{scale_code}", data, mins_offset
            )]
            if asymmetric else None
        )
    # Один проход вместо двух: сначала сообщаем о неконечном масштабе (как
    # раньше), затем об отрицательном.
    bad_finite = -1
    bad_negative = -1
    for index, scale in enumerate(scales):
        if bad_finite < 0 and not math.isfinite(scale):
            bad_finite = index
        elif bad_negative < 0 and scale < 0:
            bad_negative = index
        if bad_finite >= 0 and bad_negative >= 0:
            break
    if bad_finite >= 0:
        raise ValueError(
            f"Q4-сегмент: масштаб #{bad_finite} = {scales[bad_finite]!r} "
            "(должен быть конечным и неотрицательным)"
        )
    if bad_negative >= 0:
        raise ValueError(
            f"Q4-сегмент: масштаб #{bad_negative} = {scales[bad_negative]!r} "
            "(должен быть конечным и неотрицательным)"
        )
    if mins is not None:
        for index, value in enumerate(mins):
            if not math.isfinite(value):
                raise ValueError(
                    f"Q4-сегмент: min #{index} = {value!r} не конечный"
                )
    # Нибблы — байты: любое значение 0..15 по построению, побайтовый обход
    # 2,77 млрд значений не нужен (и занял бы минуты).
    qdata = PackedQ4.from_segment(data, q_offset, n_values)
    return qdata, scales, scale_dtype, asymmetric, mins


def decode_q4_segment_values(
    data: bytes,
) -> tuple[list[int], list[float], str, bool, list[float] | None]:
    """Старый вид разбора — списками: для мелких тестов, не для 3B.

    Тонкая обёртка над `decode_q4_segment_ex`; на 3B материализует ~22 ГиБ
    нибблов, поэтому в потоковых путях не используется.
    """
    qdata, scales, scale_dtype, asymmetric, mins = decode_q4_segment_ex(data)
    return (
        list(qdata),
        list(scales),
        scale_dtype,
        asymmetric,
        None if mins is None else list(mins),
    )


def decode_q4_segment(data: bytes) -> tuple[list[int], list[float], str]:
    """Разобрать Q4-сегмент (совместимый вид: без asym/min)."""
    qdata, scales, scale_dtype, _asymmetric, _mins = decode_q4_segment_ex(data)
    return qdata, scales, scale_dtype


def encode_trit_segment(tensor: Tensor) -> bytes:
    """Собрать TRT1-сегмент: заголовок + масштабы + триты в его хранении.

    `packed20` пишется TRT1 version 1 (байт-в-байт как до 0057), `native` —
    version 2 с идентификатором хранения в битах 1..2 flags. Байт-в-байт
    совместимость важна: старые контейнеры и старые читатели не меняются.
    """
    if tensor.kind != KIND_TRITS:
        raise ValueError(f"тензор {tensor.name!r}: это не триты")
    if tensor.trits is None or tensor.scales is None:
        raise ValueError(
            f"тензор {tensor.name!r}: нет тритов — граф собран без весов, "
            "записывать в контейнер нечего"
        )
    storage = tensor.storage
    if storage not in TRIT_STORAGE_IDS:  # pragma: no cover - проверено в Tensor
        raise ValueError(f"тензор {tensor.name!r}: хранение {storage!r}")
    n_trits = tensor.n_elements
    n_scales = n_scale_blocks(n_trits, tensor.block_size)
    if len(tensor.scales) != n_scales:
        raise ValueError(
            f"тензор {tensor.name!r}: масштабов {len(tensor.scales)}, ожидалось {n_scales}"
        )
    version = TRT1_VERSION if storage == STORAGE_PACKED20 else TRT1_VERSION_STORAGE
    flags = 0x01 if tensor.scale_dtype == "float32" else 0x00
    flags |= TRIT_STORAGE_IDS[storage] << TFLAG_STORAGE_SHIFT
    header = _TRT1_STRUCT.pack(TRT1_MAGIC, version, flags, n_trits, n_scales)
    scale_code = "f" if tensor.scale_dtype == "float32" else "e"
    try:
        scales_bytes = struct.pack(f"<{n_scales}{scale_code}", *tensor.scales)
    except (struct.error, OverflowError) as exc:
        raise ValueError(f"тензор {tensor.name!r}: масштаб не влезает в {tensor.scale_dtype}") from exc
    expected = storage_words(n_trits, storage) * 4
    if storage == STORAGE_PACKED20:
        words = pack20(tensor.trits)
    else:
        # 1 трит = 1 знаковый байт: -1 -> 0xFF, 0 -> 0x00, +1 -> 0x01.
        words = bytes(trit & 0xFF for trit in tensor.trits)
    if len(words) > expected:  # pragma: no cover - защита от рассинхрона констант
        raise ValueError(
            f"тензор {tensor.name!r}: упаковано {len(words)} байт при лимите {expected}"
        )
    words += b"\x00" * (expected - len(words))
    return header + scales_bytes + words


def _trit_segment_header(
    data: bytes, where: str = "TRT1-сегмент"
) -> tuple[str, int, int, int, int]:
    """Проверить заголовок TRT1 и вернуть (storage, flags, n_trits, n_scales, scale_size)."""
    if len(data) < TRT1_HEADER_SIZE:
        raise ValueError(f"{where} короче заголовка: {len(data)} < {TRT1_HEADER_SIZE}")
    magic, version, flags, n_trits, n_scales = _TRT1_STRUCT.unpack_from(data, 0)
    if magic != TRT1_MAGIC:
        raise ValueError(f"чужой magic {where}: {magic!r}")
    if version == TRT1_VERSION:
        # До 0057: только pack20, бит 0 — fp32-масштабы.
        if flags & ~0x01:
            raise ValueError(f"неизвестные флаги {where}: 0x{flags:02x}")
        storage = STORAGE_PACKED20
    elif version == TRT1_VERSION_STORAGE:
        if flags & ~0x07:
            raise ValueError(f"неизвестные флаги {where}: 0x{flags:02x}")
        storage = TRIT_STORAGE_NAMES.get((flags >> TFLAG_STORAGE_SHIFT) & 0x03)
        if storage is None:
            raise ValueError(
                f"{where}: неизвестное хранение тритов "
                f"{(flags >> TFLAG_STORAGE_SHIFT) & 0x03}"
            )
    else:
        raise ValueError(f"неизвестная версия {where}: {version}")
    expected_scales = n_scale_blocks(n_trits, SCALE_BLOCK)
    if n_scales != expected_scales:
        raise ValueError(
            f"в заголовке {where} {n_scales} масштабов, ожидалось {expected_scales} "
            f"для {n_trits} тритов"
        )
    scale_size = 4 if flags & 0x01 else 2
    return storage, flags, n_trits, n_scales, scale_size


def decode_trit_segment_ex(
    data: bytes,
) -> tuple[list[int], list[float], str, str]:
    """Разобрать TRT1-сегмент: триты, масштабы, dtype масштабов, хранение.

    Лишние байты в конце (padding) игнорируются.
    """
    storage, flags, n_trits, n_scales, scale_size = _trit_segment_header(data)
    scale_dtype = "float32" if flags & 0x01 else "float16"
    n_words = storage_words(n_trits, storage)
    expected_len = TRT1_HEADER_SIZE + n_scales * scale_size + n_words * 4
    if len(data) < expected_len:
        raise ValueError(
            f"TRT1-сегмент обрезан: {len(data)} байт, ожидалось {expected_len}"
        )
    scale_code = "f" if scale_dtype == "float32" else "e"
    scales = list(struct.unpack_from(f"<{n_scales}{scale_code}", data, TRT1_HEADER_SIZE))
    words_offset = TRT1_HEADER_SIZE + n_scales * scale_size
    words = data[words_offset:words_offset + n_words * 4]
    if storage == STORAGE_PACKED20:
        trits = unpack20(words, n_trits)
    else:
        raw = struct.unpack(f"<{n_words * 4}b", words)
        for index in range(n_trits):
            if raw[index] not in (-1, 0, 1):
                raise ValueError(
                    f"TRT1-сегмент: native-трит #{index} = {raw[index]} вне -1..1"
                )
        trits = list(raw[:n_trits])
    return trits, [float(s) for s in scales], scale_dtype, storage


def decode_trit_segment(data: bytes) -> tuple[list[int], list[float], str]:
    """Разобрать TRT1-сегмент (совместимый вид: без хранения)."""
    trits, scales, scale_dtype, _storage = decode_trit_segment_ex(data)
    return trits, scales, scale_dtype


def encode_tensor_payload(tensor: Tensor) -> bytes:
    """Payload тензора в том виде, в каком он лежит в секции данных."""
    if getattr(tensor, "flags", 0) & TFLAG_KV_CACHE:
        # KV-кэш в файл не пишется: тензор живёт только в таблице.
        return b""
    if tensor.kind == KIND_TRITS:
        return encode_trit_segment(tensor)
    if tensor.kind == KIND_INT8:
        return encode_int8_segment(tensor)
    if tensor.kind == KIND_Q4:
        return encode_q4_segment(tensor)
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
        trits, scales, scale_dtype, storage = decode_trit_segment_ex(payload)
        if len(trits) != tensor.n_elements:
            raise ValueError(
                f"тензор {tensor.name!r}: в сегменте {len(trits)} тритов, "
                f"а dims дают {tensor.n_elements}"
            )
        tensor.trits = trits
        tensor.scales = scales
        tensor.scale_dtype = scale_dtype
        tensor.storage = storage
        return
    if tensor.kind == KIND_INT8:
        qdata, scales, scale_dtype = decode_int8_segment(payload)
        if len(qdata) != tensor.n_elements:
            raise ValueError(
                f"тензор {tensor.name!r}: в сегменте {len(qdata)} int8-значений, "
                f"а dims дают {tensor.n_elements}"
            )
        tensor.qdata = qdata
        tensor.scales = scales
        tensor.scale_dtype = scale_dtype
        return
    if tensor.kind == KIND_Q4:
        qdata, scales, scale_dtype, asymmetric, mins = decode_q4_segment_ex(payload)
        if len(qdata) != tensor.n_elements:
            raise ValueError(
                f"тензор {tensor.name!r}: в сегменте {len(qdata)} Q4-значений, "
                f"а dims дают {tensor.n_elements}"
            )
        tensor.qdata = qdata
        tensor.scales = scales
        tensor.scale_dtype = scale_dtype
        tensor.q4_asymmetric = asymmetric
        tensor.q4_mins = mins
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

# ---------------------------------------------------------------------------
# Контроль целостности (задача 0066)
# ---------------------------------------------------------------------------


def _is_sha256_hex(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _integrity_requested(explicit: bool | None) -> bool:
    """Записывать ли хеш: явный параметр, иначе VTM_WRITE_HASH=1."""
    if explicit is not None:
        return bool(explicit)
    value = os.environ.get(INTEGRITY_WRITE_ENV, "")
    return value not in ("", "0")


def _json_string_value_offset(blob: bytes, name: str) -> int | None:
    """Смещение значения "name": "<64 hex>" в сыром JSON или None.

    Тот же поиск, что и в C-загрузчике: сырые байты, без разбора JSON, чтобы
    испорченная мета не «теряла» блок целостности.
    """
    key = b'"' + name.encode("ascii") + b'"'
    start = blob.find(key)
    while start != -1:
        i = start + len(key)
        while i < len(blob) and blob[i:i + 1] in b" \t\r\n":
            i += 1
        if blob[i:i + 1] != b":":
            start = blob.find(key, start + 1)
            continue
        i += 1
        while i < len(blob) and blob[i:i + 1] in b" \t\r\n":
            i += 1
        if blob[i:i + 1] != b'"':
            start = blob.find(key, start + 1)
            continue
        i += 1
        value = blob[i:i + 64]
        if len(value) == 64 and all(ch in b"0123456789abcdef" for ch in value):
            if blob[i + 64:i + 65] == b'"':
                return i
        start = blob.find(key, start + 1)
    return None


def _json_object_key_present(blob: bytes, name: str) -> bool:
    """Есть ли в сыром JSON ключ "name", значение которого — объект."""
    key = b'"' + name.encode("ascii") + b'"'
    start = blob.find(key)
    while start != -1:
        i = start + len(key)
        while i < len(blob) and blob[i:i + 1] in b" \t\r\n":
            i += 1
        if blob[i:i + 1] == b":":
            i += 1
            while i < len(blob) and blob[i:i + 1] in b" \t\r\n":
                i += 1
            if blob[i:i + 1] == b"{":
                return True
        start = blob.find(key, start + 1)
    return False


def _hash_meta(meta_bytes: bytes, meta_value_offset: int) -> str:
    """sha256(мета), где значение meta_sha256 заменено 64 символами '0'."""
    zeroed = (
        meta_bytes[:meta_value_offset]
        + b"0" * 64
        + meta_bytes[meta_value_offset + 64:]
    )
    return hashlib.sha256(zeroed).hexdigest()


def _verify_integrity(meta_bytes: bytes, data: bytes) -> None:
    """Сверить сырые поля integrity. Блока нет — старый контейнер.

    Работаем по сырым байтам, а не по разобранному JSON: испорченная мета
    может перестать парситься, но подмена обязана дать ошибку integrity,
    а не «битый JSON».
    """
    data_pos = _json_string_value_offset(meta_bytes, INTEGRITY_DATA_FIELD)
    meta_pos = _json_string_value_offset(meta_bytes, INTEGRITY_META_FIELD)
    if data_pos is None and meta_pos is None:
        if _json_object_key_present(meta_bytes, INTEGRITY_KEY):
            raise ValueError(
                "integrity: блок в мете есть, но data_sha256/meta_sha256 "
                "в нём отсутствуют (блок повреждён)"
            )
        return
    if data_pos is None or meta_pos is None:
        raise ValueError(
            "integrity: в мете нет пары data_sha256/meta_sha256 "
            "(блок повреждён)"
        )
    expected_data = meta_bytes[data_pos:data_pos + 64].decode("ascii")
    expected_meta = meta_bytes[meta_pos:meta_pos + 64].decode("ascii")

    if os.environ.get(INTEGRITY_SKIP_ENV, "") not in ("", "0"):
        print(
            f"VTM: {INTEGRITY_SKIP_ENV}=1 — проверка sha256 контейнера "
            "пропущена (данные не сверены)",
            file=sys.stderr,
        )
        return

    actual_data = hashlib.sha256(data).hexdigest()
    if actual_data != expected_data:
        raise ValueError(
            "integrity: sha256 секции данных не совпал: "
            f"в файле {actual_data[:16]}…, в мете {expected_data[:16]}…; "
            f"контейнер повреждён ({INTEGRITY_SKIP_ENV}=1 — пропустить)"
        )
    actual_meta = _hash_meta(meta_bytes, meta_pos)
    if actual_meta != expected_meta:
        raise ValueError(
            "integrity: sha256 метаданных не совпал: "
            f"в файле {actual_meta[:16]}…, в мете {expected_meta[:16]}…; "
            f"контейнер повреждён ({INTEGRITY_SKIP_ENV}=1 — пропустить)"
        )


def file_size_limit_error(file_size: int) -> str:
    """Ошибка «контейнер больше 4 ГиБ»: VTM адресует файл u32-смещениями.

    Формат v0.2 не представляет модель, у которой секция данных больше 4 ГиБ:
    32B-раскладка Qwen2.5 (64 слоя, hidden 5120, intermediate 27648) — это
    ~6,9 ГиБ тритов в packed20. Ошибка называет числа и варианты, чтобы не
    выглядеть загадочным «не влезает».
    """
    return (
        "VTM адресует файл 32-битными смещениями: "
        f"{file_size} байт ({file_size / 2 ** 30:.2f} ГиБ) не влезает в "
        "контейнер (предел 4 ГиБ). Раскладка 32B (64 слоя, hidden 5120, "
        "intermediate 27648) — это ~6,9 ГиБ тритов, формат v0.2 её не "
        "представляет; 7B (~1,5 ГиБ) и 14B (~3,0 ГиБ) влезают. Для 32B нужен "
        "формат с u64-смещениями (ADR) или шардирование модели."
    )


def kv_rebuild_hint(required_bytes: int) -> str:
    """Команда пересборки C-рантайма с поднятым кэпом KV (`VTM_MAX_KV_BYTES`).

    Кэп `MAX_KV_BYTES` — предел загрузчика (`src/vtm/loader.c`), а не формата:
    он задаётся на сборке через `-D`. Компилятор называет точное число байт,
    которого хватает форме, чтобы сообщение можно было скопировать в шелл.
    `CPPFLAGS` передаётся переменной окружения (а не аргументом `make`), чтобы
    `Makefile` дописал к нему свои `-D` (например, `-DVTM_NO_AVX2`).
    """
    return (
        'cd src/vtm && make clean && '
        f'CPPFLAGS="-D_POSIX_C_SOURCE=200809L -DVTM_MAX_KV_BYTES={int(required_bytes)}" make'
    )


def _streamed_payload_kind(tensor: Tensor) -> str | None:
    """Вид потокового payload'а тензора: "q4", "int8" или None (в память).

    Поток требует ленивого окна: у Q4/INT8 из сегмента это `PackedQ4`/
    `PackedInt8`, и тогда `bytes` на весь сегмент (1,6 ГиБ q4 + 0,3 ГиБ
    int8 у 3B) не материализуется.
    """
    if getattr(tensor, "flags", 0) & TFLAG_KV_CACHE:
        return None
    if tensor.kind == KIND_Q4 and isinstance(tensor.qdata, PackedQ4):
        return "q4"
    if tensor.kind == KIND_INT8 and isinstance(tensor.qdata, PackedInt8):
        return "int8"
    return None


def write_vtm(
    path: str | os.PathLike[str],
    tensors: Sequence[Tensor | dict[str, Any]],
    program: Sequence[Instr | tuple[Any, ...] | dict[str, Any]],
    meta: dict[str, Any] | None = None,
    *,
    integrity: bool | None = None,
) -> VTM:
    """Собрать контейнер `.vtm` строго по спеке и записать на диск.

    Возвращает собранный `VTM` (с проставленными offset'ами) — его можно
    сразу передать в `execute`. Порядок `tensors` = порядок таблицы, индексы
    в `program` ссылаются на него.

    `integrity=True` кладёт в мету `meta["integrity"]` с sha256 секции данных
    и самой меты (задача 0066). По умолчанию это делает `VTM_WRITE_HASH=1`,
    чтобы компилятор мог включить защиту, не меняя вызов; без этой опции
    файл без KV-кэшей пишется байт-в-байт как раньше (старые тесты и
    контейнеры не ломаются). Загрузчик проверяет блок всегда, когда он есть;
    `VTM_SKIP_HASH=1` отключает сверку.

    Контейнер с KV-кэшами (TFLAG_KV_CACHE) получает флаг заголовка
    `VFLAG_KV_OUT_OF_FILE` и offset `KV_NO_OFFSET` у кэшей (задача 0118):
    старый загрузчик до 0101 видит невыровненный offset и отказывает, а не
    читает мусор. Суммарный размер кэшей проверяется по `MAX_KV_BYTES` —
    ровно так же, как это сделает загрузчик, чтобы не записать нечитаемый
    файл.

    Q4/INT8 с ленивыми окнами (`decode_q4_segment_ex` даёт `PackedQ4`,
    `decode_int8_segment` — `PackedInt8`) пишутся **потоком**, кусками по
    `STREAM_CHUNK`: `bytes` на весь сегмент (у 3B — 1,6 ГиБ q4 и 0,3 ГиБ
    int8) не материализуется. Байты контейнера при этом те же, что при
    записи из списков: раскладка сегмента не зависит от представления
    значений (сверка sha256 — в тестах `tests/test_q4_stream.py`).
    """
    tensor_list = [_as_tensor(item) for item in tensors]
    program_list = [_as_instr(item) for item in program]
    meta_dict: dict[str, Any] = dict(meta or {})
    _validate_names(tensor_list)
    _validate_program(program_list, len(tensor_list))
    if len(tensor_list) > MAX_TENSORS:
        raise ValueError(f"тензоров {len(tensor_list)}, максимум {MAX_TENSORS}")

    kv_caches = [t for t in tensor_list if getattr(t, "flags", 0) & TFLAG_KV_CACHE]
    kv_bytes = sum(t.n_elements * 4 for t in kv_caches)
    if kv_bytes > MAX_KV_BYTES:
        raise ValueError(
            f"KV-кэши: суммарный размер {kv_bytes} байт больше предела "
            f"{MAX_KV_BYTES} (MAX_KV_BYTES); уменьшите ctx или kv_dim. "
            "Если контекст нужен целиком, кэп поднимается на сборке рантайма: "
            + kv_rebuild_hint(kv_bytes)
        )

    # Payload'ы мелких тензоров собираются в память (как раньше: ошибка
    # «нет данных» должна всплыть до создания файла). Q4 с ленивыми окнами
    # пишется потоком: `bytes` на 1,6 ГиБ сегмента не нужен, достаточно
    # проверить тензор и взять размер из dims.
    payloads: list[bytes | None] = []
    for tensor in tensor_list:
        streamed = _streamed_payload_kind(tensor)
        if streamed == "q4":
            _q4_segment_header(tensor)  # валидация без материализации
            payloads.append(None)
        elif streamed == "int8":
            _int8_segment_header(tensor)
            payloads.append(None)
        else:
            payloads.append(encode_tensor_payload(tensor))
    write_integrity = _integrity_requested(integrity)

    n_tensors = len(tensor_list)
    table_end = HEADER_SIZE + TENSOR_ENTRY_SIZE * n_tensors
    data_start = align_up(table_end)

    digest = hashlib.sha256() if write_integrity else None
    has_kv = bool(kv_caches)
    data_size = 0
    for tensor, payload in zip(tensor_list, payloads):
        size = tensor_payload_size(tensor) if payload is None else len(payload)
        if getattr(tensor, "flags", 0) & TFLAG_KV_CACHE:
            # KV-кэш места в секции данных не занимает: payload'а нет, размер
            # известен из dims. У контейнера нового вида offset — маркер
            # KV_NO_OFFSET (не адрес): старый загрузчик до 0101 проверяет
            # выравнивание offset и отказывает, не читая данные. Без флага
            # (0101 и старше) сохраняем прежний offset=0 — совместимость.
            tensor.offset = KV_NO_OFFSET if has_kv else 0
            tensor.scale_offset = 0
            continue
        tensor.offset = data_size
        tensor.scale_offset = (
            NO_OFFSET if tensor.kind in (KIND_TRITS, KIND_INT8, KIND_Q4) else 0
        )
        if tensor.offset % ALIGNMENT:
            raise RuntimeError(  # pragma: no cover - защита от ошибки в align_up
                f"тензор {tensor.name!r}: offset не выровнен"
            )
        start = data_size
        data_size = align_up(start + size)

    block: dict[str, Any] | None = None
    if digest is not None:
        # Хеш секции данных считается при записи (Q4 идёт потоком, и до неё
        # байтов ещё нет). Длина меты не зависит от значения: оба хеша —
        # 64 hex-символа, поэтому offset'ы считаются по плейсхолдерам.
        block = {
            "algo": INTEGRITY_ALGO,
            "data_sha256": "0" * 64,
            INTEGRITY_META_FIELD: "0" * 64,
        }
        meta_dict.pop(INTEGRITY_KEY, None)
        meta_dict[INTEGRITY_KEY] = block

    meta_bytes = _encode_meta(meta_dict) if meta_dict else b""
    meta_len = len(meta_bytes)
    meta_offset = data_start + data_size if meta_bytes else 0
    program_offset = data_start + data_size + align_up(len(meta_bytes))
    program_bytes = b"".join(instr.pack() for instr in program_list)
    file_size = program_offset + len(program_bytes)
    if file_size >= 2 ** 32:
        raise ValueError(file_size_limit_error(file_size))

    flags = VFLAG_CAUSAL
    if all(t.kind != KIND_TRITS or t.scale_dtype == "float32" for t in tensor_list):
        flags |= VFLAG_F32_SCALES
    if has_kv:
        # Новый вид контейнера (0118): KV-кэши вне файла, их offset —
        # KV_NO_OFFSET. Бита 0x0004 старый загрузчик не знает и игнорирует,
        # отказ даёт offset (см. write_vtm).
        flags |= VFLAG_KV_OUT_OF_FILE

    header = _HEADER_STRUCT.pack(
        VTM_MAGIC, VTM_VERSION, flags, len(program_list), n_tensors, len(program_bytes)
    ) + _RESERVED_STRUCT.pack(program_offset, meta_offset, len(meta_bytes))

    table = bytearray()
    for tensor in tensor_list:
        name_bytes = tensor.name.encode("utf-8")
        dims = list(tensor.dims) + [0] * (MAX_DIMS - len(tensor.dims))
        tflags = int(getattr(tensor, "flags", 0))
        if tensor.kind == KIND_TRITS:
            # Хранение дублируется в flags записи (биты 1..2): читателю, который
            # не разбирает TRT1, видно, что payload — не pack20.
            tflags = (tflags & ~TFLAG_STORAGE_MASK) | (
                TRIT_STORAGE_IDS[tensor.storage] << TFLAG_STORAGE_SHIFT
            )
        table += _TENSOR_STRUCT.pack(
            name_bytes, tensor.kind, len(tensor.dims), *dims,
            tensor.offset, tensor.scale_offset, tflags,
        )

    with open(path, "wb") as handle:
        handle.write(header)
        handle.write(bytes(table))
        handle.write(b"\x00" * (data_start - table_end))
        for tensor, payload in zip(tensor_list, payloads):
            size = tensor_payload_size(tensor) if payload is None else len(payload)
            if payload is None:
                # Q4-сегмент потоком: заголовок, масштабы, нибблы — куски
                # прямо из байт сегмента (без bytes на весь тензор).
                pieces = (
                    iter_int8_payload(tensor)
                    if tensor.kind == KIND_INT8 else iter_q4_payload(tensor)
                )
                written_payload = 0
                for piece in pieces:
                    handle.write(piece)
                    if digest is not None:
                        digest.update(piece)
                    written_payload += len(piece)
                if written_payload != size:  # pragma: no cover - защита
                    raise RuntimeError(
                        f"тензор {tensor.name!r}: записано {written_payload} "
                        f"байт payload'а, ожидалось {size}"
                    )
                if digest is not None:
                    digest.update(b"\x00" * (align_up(size) - size))
                handle.write(b"\x00" * (align_up(size) - size))
                continue
            handle.write(payload)
            if digest is not None:
                digest.update(payload)
                digest.update(
                    b"\x00" * (align_up(len(payload)) - len(payload))
                )
            handle.write(b"\x00" * (align_up(len(payload)) - len(payload)))
        if meta_bytes:
            if digest is not None:
                # data_sha256 известен только теперь; сериализация с настоящим
                # значением и плейсхолдером meta_sha256 — вход для второго хеша
                # (как в write_vtm_stream и в _verify_integrity).
                block[INTEGRITY_DATA_FIELD] = digest.hexdigest()
                block[INTEGRITY_META_FIELD] = "0" * 64
                meta_bytes = _encode_meta(meta_dict)
                block[INTEGRITY_META_FIELD] = hashlib.sha256(meta_bytes).hexdigest()
                meta_bytes = _encode_meta(meta_dict)
                if len(meta_bytes) != meta_len:  # pragma: no cover - защита
                    raise RuntimeError(
                        f"длина меты изменилась: {len(meta_bytes)} != {meta_len}"
                    )
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


def _empty_payload_prefix(tensor: Tensor) -> tuple[bytes, int]:
    """Заголовок payload'а «без данных» и полный размер payload'а.

    Для тритов это TRT1-заголовок с нулевыми масштабами (веса 0, слова
    pack20 тоже нулевые), для int8 — INT8-заголовок с нулевыми q, для fp —
    ничего. Так
    потоковый писатель (`write_vtm_stream`) отдаёт контейнер нужной формы,
    не материализуя веса: 7B-раскладка — это ~1,5 ГиБ тритов, а 32B — ~7 ГиБ.
    """
    if getattr(tensor, "flags", 0) & TFLAG_KV_CACHE:
        return b"", 0
    size = tensor_payload_size(tensor)
    if tensor.kind == KIND_TRITS:
        storage = tensor.storage
        version = (
            TRT1_VERSION if storage == STORAGE_PACKED20 else TRT1_VERSION_STORAGE
        )
        flags = 0x01 if tensor.scale_dtype == "float32" else 0x00
        flags |= TRIT_STORAGE_IDS[storage] << TFLAG_STORAGE_SHIFT
        n_scales = n_scale_blocks(tensor.n_elements, tensor.block_size)
        header = _TRT1_STRUCT.pack(
            TRT1_MAGIC, version, flags, tensor.n_elements, n_scales
        )
        return header, size
    if tensor.kind == KIND_INT8:
        flags = 0x01 if tensor.scale_dtype == "float32" else 0x00
        n_scales = n_scale_blocks(tensor.n_elements, tensor.block_size)
        header = _TRT1_STRUCT.pack(
            INT8_MAGIC, INT8_VERSION, flags, tensor.n_elements, n_scales
        )
        return header, size
    if tensor.kind == KIND_Q4:
        flags = Q4_FLAG_F32_SCALES if tensor.scale_dtype == "float32" else 0
        n_groups = n_scale_blocks(tensor.n_elements, Q4_SCALE_BLOCK)
        header = _TRT1_STRUCT.pack(
            Q4_MAGIC_ASYM if tensor.q4_asymmetric else Q4_MAGIC,
            Q4_VERSION, flags, tensor.n_elements, n_groups,
        )
        if tensor.q4_asymmetric:
            header += struct.pack("<I", n_groups)
        return header, size
    return b"", size


_ZEROS = bytes(STREAM_CHUNK)


def write_vtm_stream(
    path: str | os.PathLike[str],
    tensors: Sequence[Tensor | dict[str, Any]],
    program: Sequence[Instr | tuple[Any, ...] | dict[str, Any]],
    meta: dict[str, Any] | None = None,
    *,
    integrity: bool | None = None,
    chunk: int = STREAM_CHUNK,
) -> VTM:
    """Записать контейнер потоком, не держа payload'ы весов в памяти.

    Раскладка байт-в-байт та же, что у `write_vtm` (заголовок, таблица,
    выровненная секция данных, мета, программа), но вместо значений тензоров
    в секцию данных идут нули: fp-тензоры — 4/2 нулевых байта на элемент, у
    тритов — валидный TRT1 с нулевыми масштабами (это триты 0), у int8 —
    нулевые q. Контейнер читается `read_vtm` и C-загрузчиком, но модель в нём
    «пустая»: логиты нулевые, argmax = 0.

    Зачем: синтетика больших форм (7B/14B/32B) нужна, чтобы проверить сборку,
    загрузку и проход на реальных размерностях, а не значениях. `write_vtm`
    требует данных в каждом тензоре и на 32B съел бы ~7 ГиБ списков Python;
    здесь память — один буфер `chunk` байт, а файл пишется последовательно.

    KV-кэши, как и в `write_vtm`, в файл не пишутся (флаг
    `VFLAG_KV_OUT_OF_FILE`, offset `KV_NO_OFFSET`), а их суммарный размер
    проверяется по `MAX_KV_BYTES` — ровно так же, как это сделает загрузчик.
    """
    tensor_list = [_as_tensor(item) for item in tensors]
    program_list = [_as_instr(item) for item in program]
    meta_dict: dict[str, Any] = dict(meta or {})
    _validate_names(tensor_list)
    _validate_program(program_list, len(tensor_list))
    if len(tensor_list) > MAX_TENSORS:
        raise ValueError(f"тензоров {len(tensor_list)}, максимум {MAX_TENSORS}")
    if chunk <= 0:
        raise ValueError(f"chunk={chunk} должен быть > 0")
    for tensor in tensor_list:
        if getattr(tensor, "flags", 0) & TFLAG_KV_CACHE:
            continue
        if (
            tensor.data is not None
            or tensor.trits is not None
            or tensor.scales is not None
            or tensor.qdata is not None
        ):
            # Иначе явные веса молча превратились бы в нули — это потеря
            # данных, а не оптимизация.
            raise ValueError(
                f"тензор {tensor.name!r}: write_vtm_stream пишет нули, а у "
                "тензора уже есть данные; обнулите data/trits/scales или "
                "пишите контейнер через write_vtm"
            )

    kv_caches = [t for t in tensor_list if getattr(t, "flags", 0) & TFLAG_KV_CACHE]
    kv_bytes = sum(t.n_elements * 4 for t in kv_caches)
    if kv_bytes > MAX_KV_BYTES:
        raise ValueError(
            f"KV-кэши: суммарный размер {kv_bytes} байт больше предела "
            f"{MAX_KV_BYTES} (MAX_KV_BYTES); уменьшите ctx или kv_dim. "
            "Если контекст нужен целиком, кэп поднимается на сборке рантайма: "
            + kv_rebuild_hint(kv_bytes)
        )

    write_integrity = _integrity_requested(integrity)
    n_tensors = len(tensor_list)
    table_end = HEADER_SIZE + TENSOR_ENTRY_SIZE * n_tensors
    data_start = align_up(table_end)
    has_kv = bool(kv_caches)

    digest = hashlib.sha256() if write_integrity else None
    layout: list[tuple[Tensor, bytes, int]] = []
    data_size = 0
    for tensor in tensor_list:
        if getattr(tensor, "flags", 0) & TFLAG_KV_CACHE:
            tensor.offset = KV_NO_OFFSET if has_kv else 0
            tensor.scale_offset = 0
            layout.append((tensor, b"", 0))
            continue
        prefix, size = _empty_payload_prefix(tensor)
        tensor.offset = data_size
        tensor.scale_offset = (
            NO_OFFSET if tensor.kind in (KIND_TRITS, KIND_INT8, KIND_Q4) else 0
        )
        if tensor.offset % ALIGNMENT:  # pragma: no cover - защита от align_up
            raise RuntimeError(
                f"тензор {tensor.name!r}: offset не выровнен"
            )
        layout.append((tensor, prefix, size))
        data_size = align_up(data_size + size)

    if digest is not None:
        # Плейсхолдеры той же длины, что настоящие хеши (64 hex-символа):
        # длина меты не должна зависеть от момента, когда data_sha256 станет
        # известен, иначе program_offset разойдётся с записанными байтами.
        block = {
            "algo": INTEGRITY_ALGO,
            "data_sha256": "0" * 64,
            INTEGRITY_META_FIELD: "0" * 64,
        }
        meta_dict.pop(INTEGRITY_KEY, None)
        meta_dict[INTEGRITY_KEY] = block

    meta_bytes = _encode_meta(meta_dict) if meta_dict else b""
    meta_offset = data_start + data_size if meta_bytes else 0
    program_offset = data_start + data_size + align_up(len(meta_bytes))
    program_bytes = b"".join(instr.pack() for instr in program_list)
    file_size = program_offset + len(program_bytes)
    if file_size >= 2 ** 32:
        raise ValueError(file_size_limit_error(file_size))

    flags = VFLAG_CAUSAL
    if all(t.kind != KIND_TRITS or t.scale_dtype == "float32" for t in tensor_list):
        flags |= VFLAG_F32_SCALES
    if has_kv:
        flags |= VFLAG_KV_OUT_OF_FILE

    header = _HEADER_STRUCT.pack(
        VTM_MAGIC, VTM_VERSION, flags, len(program_list), n_tensors,
        len(program_bytes),
    ) + _RESERVED_STRUCT.pack(program_offset, meta_offset, len(meta_bytes))

    table = bytearray()
    for tensor in tensor_list:
        name_bytes = tensor.name.encode("utf-8")
        dims = list(tensor.dims) + [0] * (MAX_DIMS - len(tensor.dims))
        tflags = int(getattr(tensor, "flags", 0))
        if tensor.kind == KIND_TRITS:
            tflags = (tflags & ~TFLAG_STORAGE_MASK) | (
                TRIT_STORAGE_IDS[tensor.storage] << TFLAG_STORAGE_SHIFT
            )
        table += _TENSOR_STRUCT.pack(
            name_bytes, tensor.kind, len(tensor.dims), *dims,
            tensor.offset, tensor.scale_offset, tflags,
        )

    def _write_zeros(handle: Any, count: int) -> None:
        remaining = int(count)
        while remaining > 0:
            piece = _ZEROS[: min(chunk, remaining)]
            handle.write(piece)
            if digest is not None:
                digest.update(piece)
            remaining -= len(piece)

    with open(path, "wb") as handle:
        handle.write(header)
        handle.write(bytes(table))
        handle.write(b"\x00" * (data_start - table_end))
        # Секция данных начинается после выравнивания таблицы: её ведущие
        # нули в хеш не входят (в `write_vtm` хешируются только payload'ы
        # и их выравнивание).
        for _tensor, prefix, size in layout:
            if prefix:
                handle.write(prefix)
                if digest is not None:
                    digest.update(prefix)
            _write_zeros(handle, size - len(prefix))
            tail = align_up(size) - size
            if tail:
                handle.write(b"\x00" * tail)
                if digest is not None:
                    digest.update(b"\x00" * tail)
        if meta_bytes:
            if digest is not None:
                # data_sha256 известен только после данных. Сначала пишем его
                # в мету и пере-сериализуем (длина та же: 64 hex), затем
                # считаем meta_sha256 по сериализации, где это поле — нули,
                # как в `write_vtm` и в проверке `_verify_integrity`.
                block[INTEGRITY_DATA_FIELD] = digest.hexdigest()
                block[INTEGRITY_META_FIELD] = "0" * 64
                meta_bytes = _encode_meta(meta_dict)
                block[INTEGRITY_META_FIELD] = hashlib.sha256(meta_bytes).hexdigest()
                meta_bytes = _encode_meta(meta_dict)
            handle.write(meta_bytes)
            handle.write(b"\x00" * (align_up(len(meta_bytes)) - len(meta_bytes)))
        handle.write(program_bytes)

    written = os.path.getsize(path)
    if written != file_size:  # pragma: no cover - защита от рассинхрона раскладки
        raise RuntimeError(f"записано {written} байт, ожидалось {file_size}")

    return VTM(
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


def _payload_extent(
    data: bytes, start: int, kind: int, n_elements: int, flags: int = 0
) -> tuple[int, str | None]:
    """Размер payload'а по заголовку TRT1 (для тритов) или по kind/dims.

    У KV-кэша (flags & TFLAG_KV_CACHE) payload'а нет: 0 байт, читать нечего.
    """
    if flags & TFLAG_KV_CACHE:
        if kind in (KIND_TRITS, KIND_INT8, KIND_Q4):
            raise ValueError(
                f"KV-кэш обязан быть fp-тензором, а запись говорит kind={kind}"
            )
        return 0, None
    if kind == KIND_TRITS:
        if start + TRT1_HEADER_SIZE > len(data):
            raise ValueError("TRT1-сегмент выходит за конец файла")
        header = data[start:start + TRT1_HEADER_SIZE]
        storage, flags, n_trits, n_scales, scale_bytes = _trit_segment_header(header)
        if n_trits != n_elements:
            raise ValueError(
                f"в TRT1-сегменте {n_trits} тритов, а dims дают {n_elements}"
            )
        scale_dtype = "float32" if flags & 0x01 else "float16"
        size = (
            TRT1_HEADER_SIZE
            + n_scales * scale_bytes
            + storage_words(n_trits, storage) * 4
        )
        return size, scale_dtype
    if kind == KIND_INT8:
        if start + INT8_HEADER_SIZE > len(data):
            raise ValueError("INT8-сегмент выходит за конец файла")
        magic, version, flags, n_values, n_scales = _TRT1_STRUCT.unpack_from(data, start)
        if magic != INT8_MAGIC:
            raise ValueError(f"чужой magic INT8-сегмента: {magic!r}")
        if version != INT8_VERSION:
            raise ValueError(f"неизвестная версия INT8-сегмента: {version}")
        if flags & ~0x01:
            raise ValueError(f"неизвестные флаги INT8-сегмента: 0x{flags:02x}")
        if n_values != n_elements:
            raise ValueError(
                f"в INT8-сегменте {n_values} значений, а dims дают {n_elements}"
            )
        expected_scales = n_scale_blocks(n_values, INT8_SCALE_BLOCK)
        if n_scales != expected_scales:
            raise ValueError(
                f"в INT8-сегменте {n_scales} масштабов, ожидалось "
                f"{expected_scales}"
            )
        scale_dtype = "float32" if flags & 0x01 else "float16"
        scale_size = 4 if scale_dtype == "float32" else 2
        return INT8_HEADER_SIZE + n_scales * scale_size + n_values, scale_dtype
    if kind == KIND_Q4:
        if start + Q4_HEADER_SIZE > len(data):
            raise ValueError("Q4-сегмент выходит за конец файла")
        magic, version, qflags, n_values, n_groups = _TRT1_STRUCT.unpack_from(
            data, start
        )
        if magic not in (Q4_MAGIC, Q4_MAGIC_ASYM):
            raise ValueError(f"чужой magic Q4-сегмента: {magic!r}")
        if version != Q4_VERSION:
            raise ValueError(f"неизвестная версия Q4-сегмента: {version}")
        if qflags & ~Q4_FLAG_F32_SCALES:
            raise ValueError(f"неизвестные флаги Q4-сегмента: 0x{qflags:02x}")
        if n_values != n_elements:
            raise ValueError(
                f"в Q4-сегменте {n_values} значений, а dims дают {n_elements}"
            )
        expected_groups = n_scale_blocks(n_values, Q4_SCALE_BLOCK)
        if n_groups != expected_groups:
            raise ValueError(
                f"в Q4-сегменте {n_groups} групп, ожидалось "
                f"{expected_groups}"
            )
        scale_dtype = "float32" if qflags & Q4_FLAG_F32_SCALES else "float16"
        scale_size = 4 if scale_dtype == "float32" else 2
        asymmetric = magic == Q4_MAGIC_ASYM
        header_size = Q4_HEADER_SIZE
        if asymmetric:
            if start + Q4_ASYM_HEADER_SIZE > len(data):
                raise ValueError("Q4_1-сегмент короче заголовка (нет n_mins)")
            n_mins = struct.unpack_from("<I", data, start + Q4_HEADER_SIZE)[0]
            if n_mins != expected_groups:
                raise ValueError(
                    f"Q4_1: n_mins={n_mins}, ожидалось {expected_groups}"
                )
            header_size = Q4_ASYM_HEADER_SIZE
        return (
            header_size
            + n_groups * scale_size * (2 if asymmetric else 1)
            + n_groups * Q4_GROUP_BYTES,
            scale_dtype,
        )
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
    # Задача 0118: флаг «KV-кэши вне файла». Старые контейнеры его не ставят
    # и читаются прежним путём; контейнер с флагом обязан нести KV-кэши, а их
    # offset — маркер KV_NO_OFFSET, а не адрес.
    kv_flag = bool(flags & VFLAG_KV_OUT_OF_FILE)
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
    kv_bytes = 0
    has_kv = False
    for index in range(n_tensors):
        entry_offset = HEADER_SIZE + TENSOR_ENTRY_SIZE * index
        raw_name, kind, ndim, d0, d1, d2, d3, offset, scale_offset, tflags = (
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
        if tflags & TFLAG_KV_CACHE:
            # KV-кэш: payload'а в файле нет, offset — не адрес. В контейнере
            # нового вида (флаг VFLAG_KV_OUT_OF_FILE) это маркер
            # KV_NO_OFFSET; старый загрузчик до 0101 на нём отказывает, а не
            # читает данные по offset=0 как мусор. Контейнеры без флага
            # (0101 и старше) по-прежнему несут offset=0.
            if kind in (KIND_TRITS, KIND_INT8, KIND_Q4):
                raise ValueError(f"тензор {name!r}: KV-кэш обязан быть fp-тензором")
            if scale_offset != 0:
                raise ValueError(
                    f"тензор {name!r}: у KV-кэша scale_offset={scale_offset}, ожидался 0"
                )
            if kv_flag:
                if offset != KV_NO_OFFSET:
                    raise ValueError(
                        f"тензор {name!r}: KV-кэш с флагом KV_OUT_OF_FILE, "
                        f"offset={offset}, ожидался 0x{KV_NO_OFFSET:08X} "
                        "(payload'а в файле нет)"
                    )
            elif offset != 0:
                raise ValueError(
                    f"тензор {name!r}: KV-кэш без флага KV_OUT_OF_FILE, "
                    f"offset={offset}, ожидался 0"
                )
            kv_bytes += prod(dims) * 4
            if kv_bytes > MAX_KV_BYTES:
                raise ValueError(
                    f"KV-кэши: суммарный размер {kv_bytes} байт больше предела "
                    f"{MAX_KV_BYTES} (MAX_KV_BYTES); тензор {name!r} — "
                    f"{prod(dims)} элементов"
                )
            has_kv = True
            tensor = Tensor(
                name=name, kind=kind, dims=dims, offset=offset,
                scale_offset=0, flags=tflags,
            )
            tensors.append(tensor)
            continue
        if offset % ALIGNMENT:
            raise ValueError(f"тензор {name!r}: offset={offset} не выровнен на {ALIGNMENT}")

        entry_storage = (tflags >> TFLAG_STORAGE_SHIFT) & 0x03
        if kind == KIND_TRITS and entry_storage not in TRIT_STORAGE_NAMES:
            raise ValueError(
                f"тензор {name!r}: неизвестное хранение тритов "
                f"{entry_storage} в flags"
            )
        tensor = Tensor(
            name=name, kind=kind, dims=dims, offset=offset,
            scale_offset=scale_offset, flags=tflags,
            storage=(
                TRIT_STORAGE_NAMES.get(entry_storage, STORAGE_PACKED20)
                if kind == KIND_TRITS
                else STORAGE_PACKED20
            ),
        )
        size, scale_dtype = _payload_extent(
            data, data_start + offset, kind, tensor.n_elements, tflags
        )
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
        elif kind == KIND_INT8:
            if scale_offset != NO_OFFSET:
                raise ValueError(
                    f"тензор {name!r}: scale_offset={scale_offset}, ожидалось "
                    f"{NO_OFFSET} (масштабы внутри INT8-сегмента)"
                )
            tensor.scale_offset = NO_OFFSET
            tensor.scale_dtype = scale_dtype or "float32"
        elif kind == KIND_Q4:
            if scale_offset != NO_OFFSET:
                raise ValueError(
                    f"тензор {name!r}: scale_offset={scale_offset}, ожидалось "
                    f"{NO_OFFSET} (масштабы внутри Q4-сегмента)"
                )
            tensor.scale_offset = NO_OFFSET
            tensor.scale_dtype = scale_dtype or "float32"
        elif scale_offset != 0:
            raise ValueError(
                f"тензор {name!r}: scale_offset должен быть 0 для fp-тензора"
            )
        expected_storage = tensor.storage
        decode_tensor_values(tensor, data[start:end])
        if kind == KIND_TRITS and tensor.storage != expected_storage:
            raise ValueError(
                f"тензор {name!r}: хранение в flags ({expected_storage}) не "
                f"совпадает с TRT1 ({tensor.storage})"
            )
        ranges.append((start, end, name))
        tensors.append(tensor)

    # Флаг обещает новый вид контейнера: в нём обязан быть хотя бы один
    # KV-кэш. Обратное не требуется — старые контейнеры с кэшами внутри и
    # 0101-контейнеры без флага читаются по-прежнему (задача 0118).
    if kv_flag and not has_kv:
        raise ValueError(
            "флаг KV_OUT_OF_FILE поднят, но в таблице нет KV-кэшей"
        )

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
        # Контроль целостности (задача 0066): блок есть — сверяем всегда,
        # кроме VTM_SKIP_HASH=1. Старые контейнеры без блока читаются как
        # раньше.
        _verify_integrity(
            data[meta_offset:meta_offset + meta_len],
            data[data_start:meta_offset],
        )
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


def _meta_number(meta: dict, key: str) -> float | None:
    """Первое числовое значение ключа на минимальной глубине — как C-сканер.

    C ищет ключ по тексту JSON на любом уровне вложенности и при нескольких
    вхождениях берёт самое внешнее; здесь тот же порядок обхода. Числа из
    `bool` не считаются числом (в C `true`/`false` не разбирает `strtod`).
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


#: Прежнее имя помощника: так его звал разбор SAMP (задача 0051).
_samp_meta_number = _meta_number


def _context_window(meta: dict) -> int:
    """`meta["context_window"]` -> длина кольца или 0, если ключа нет/он битый.

    Ровно как `kv_window()` в `src/vtm/interp.c`: берётся конечное число в
    диапазоне `[1, INT_MAX]` и усекается к int; всё остальное (NaN, `0`,
    отрицательное, строка, bool, слишком большое) — «ключа нет», окно равно
    длине KV-кэша.
    """
    value = _meta_number(meta, "context_window")
    if value is None or not math.isfinite(value):
        return 0
    if value < 1.0 or value > float(INT_MAX):
        return 0
    return int(value)


def _samp_params(meta: dict) -> dict:
    """Разбор и кламп параметров SAMP — ровно как samp_params_from_meta()."""
    params = dict(_SAMP_DEFAULTS)
    number = _meta_number

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
        #: Длина кольца KV-окна: `min(ctx, meta["context_window"])`, как в C.
        #: 0 — ключа нет (или он битый), окно равно длине кэша.
        self.context_window = _context_window(self.meta)
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
        # KV-кэш из контейнера нового формата (TFLAG_KV_CACHE) данных не
        # имеет: буфер создаётся здесь, нулями — ровно как кэш из старого
        # контейнера, где в файле лежали нули. Так оба формата дают одно
        # поведение (ATTN до первого KVW видит нули, а не «нет данных»).
        # Берём все помеченные тензоры, а не только dst/b KVW: в C такой
        # тензор тоже существует и читается как нули, даже если программа до
        # него не пишет.
        for index, tensor in enumerate(self.tensors):
            if tensor.is_kv_cache:
                self.workspace[index] = [0.0] * tensor.n_elements

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

    def _window(self, ctx: int) -> int:
        """Эффективное окно кольца: `min(ctx, context_window)` (C: kv_window)."""
        if 0 < self.context_window < ctx:
            return self.context_window
        return ctx

    # -- проход --------------------------------------------------------------

    def begin_step(self) -> None:
        """Сбросить рабочую область, сохранив KV-кэш."""
        self.workspace = {
            index: values
            for index, values in self.workspace.items()
            if index in self.cache_indices or self.tensors[index].is_kv_cache
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
            if pos < 0 or pos > INT_MAX:
                raise ValueError(f"KVW: позиция {pos} вне диапазона")
            k = self._read(int(instr.a))
            v = self._read(int(instr.c))
            if len(k) != kv_dim or len(v) != kv_dim:
                raise ValueError(
                    f"KVW: K/V длины {len(k)}/{len(v)}, кэш ждёт {kv_dim}"
                )
            k_cache = self._ensure_cache(int(instr.dst))
            v_cache = self._ensure_cache(int(instr.b))
            # Кольцо (задача 0076): токен глобальной позиции p живёт в слоте
            # p % win, win = min(ctx, context_window). Позиция >= ctx — не
            # ошибка: она затирает самый старый слот (C: op_kvw).
            win = self._window(ctx)
            base = (pos % win) * kv_dim
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
            # Верхней границы «меньше ctx» больше нет: 2-D кэш стал кольцом
            # (0076), длинный диалог законен. Осталась защита от битого imm —
            # та же, что в C (`qs > INT_MAX - q_len`).
            if q_start < 0 or q_start > INT_MAX - q_len:
                raise ValueError(
                    f"ATTN: q_start={q_start} вне диапазона (q_len={q_len})"
                )
            key_count = q_start + q_len
            k_cache = self.workspace.get(int(instr.b), [0.0] * (ctx * kv_dim))
            v_cache = self.workspace.get(int(instr.c), [0.0] * (ctx * kv_dim))
            causal = bool(instr.flags & IFLAG_CAUSAL)
            win = self._window(ctx)
            scale = 1.0 / math.sqrt(head_dim)
            out = [0.0] * (n_heads * q_len * head_dim)
            for head in range(n_heads):
                kv_head = head // group
                for qi in range(q_len):
                    q_base = (head * q_len + qi) * head_dim
                    out_base = q_base
                    # Видимые ключи запроса qi — [lo, hi] в глобальных
                    # позициях. Кэш — кольцо: ATTN читает только последние
                    # win ключей, слот глобальной позиции j — `j % win`
                    # (хронологический обход, как в C), поэтому q_start >= ctx
                    # не ошибка.
                    hi = q_start + qi if causal else key_count - 1
                    if hi > key_count - 1:
                        hi = key_count - 1
                    lo = hi - win + 1
                    if lo < 0:
                        lo = 0
                    if hi < lo:
                        continue
                    scores: list[float] = []
                    peak = -math.inf
                    for j in range(lo, hi + 1):
                        base = (j % win) * kv_dim + kv_head * head_dim
                        dot = 0.0
                        for d in range(head_dim):
                            dot += q[q_base + d] * k_cache[base + d]
                        value = dot * scale
                        scores.append(value)
                        if value > peak:
                            peak = value
                    if peak == -math.inf:
                        continue
                    # Онлайн-softmax C копит p·V и делит на сумму в конце;
                    # здесь тот же порядок, чтобы паритет не терял биты.
                    acc = [0.0] * head_dim
                    total = 0.0
                    for offset, value in enumerate(scores):
                        weight = math.exp(value - peak)
                        total += weight
                        if weight == 0.0:
                            continue
                        base = ((lo + offset) % win) * kv_dim + kv_head * head_dim
                        for d in range(head_dim):
                            acc[d] += weight * v_cache[base + d]
                    if total <= 0.0 or not math.isfinite(total):
                        continue
                    for d in range(head_dim):
                        out[out_base + d] = acc[d] / total
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
