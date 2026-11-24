#!/usr/bin/env python3
"""Импорт GGUF (ollama) -> манифест и веса VTM.

Зачем
=====

Ollama отдаёт модели в GGUF. Чтобы Qwen из `~/.ollama/models/blobs` попала в
ВТМ, нужен разбор контейнера GGUF (заголовок, метаданные, таблица тензоров,
типы квантования) и преобразование в наш формат: манифест
`eva32.ternary.manifest` v2 + файлы весов, которые читает
`tools/vtm_compile.py`.

Что поддерживается
==================

Чтение (v2 и v3, little-endian):

* заголовок, метаданные всех типов: скаляры и небольшие числовые массивы
  читаются, строковые массивы (словарь токенизатора) пропускаются, но их
  длина запоминается — по ней находится `vocab`, если его нет в метаданных;
* таблица тензоров: имя, размерности (порядок GGUF, ne[0] — младшая),
  тип, смещение от секции данных, размер в байтах;
* типы: **F32, F16, BF16, Q4_0, Q4_1, Q8_0, Q4_K, Q6_K**. Q4_K/Q6_K нужны,
  потому что ollama раздаёт Qwen в Q4_K_M: это супер-блок 256 значений из
  8 (у Q6_K — 16) под-блоков, 6-битные масштабы/минимумы (`get_scale_min_k4`),
  fp16 `d`/`dmin`. Остальные типы (Q5_K, Q3_K, IQ*, TQ*, F64, I8…) отвергаются
  с именем типа и списком тензоров — не «на удачу».

Раскладка GGUF: `dims = [ne0, ne1, ...]`, `ne0` — самая быстрая (входная)
размерность, поэтому форма HF — это `reversed(dims)`. Значения
`blk.N.*` отображаются в Qwen/Llama-имена (`model.layers.N...`), которые
понимает `tools/vtm_compile.py`.

Что пишется
===========

`--format q4` (по умолчанию для Linear):

* `q4.bin` — Q4-сегменты kind 5 (контракт 0151/0152, формат задан
  `tools/vtm_ref.py:encode_q4_segment`). По умолчанию — **Q4_1** (задача
  0160): magic `"Q4_1"`, 22-байтовый заголовок (`n_groups` + явное
  `n_mins u32`), затем масштабы и min'ы на группу **32** (fp32 или fp16 —
  `--q4-scale-dtype`), затем нибблы в раскладке GGML (байт `j` = значение
  `j` в младшем ниббле, `j+16` — в старшем),
  `min = min(группа)`, `scale = (max-min)/15`,
  `q = clamp(round_half_away((w-min)/scale), 0, 15)`, `w = q·scale + min`.
  `--q4-kind q4_0` возвращает прежний симметричный вид (`"Q4_0"`, 18 байт
  заголовка, `scale = absmax/7`, `q = clamp(round_half_away(w/scale),-8,7)+8`,
  `w = (q-8)·scale`) — он нужен 7B, см. «7B и 4 ГиБ»;
* `non_ternary.bin` — таблица словаря INT8-сегментом kind 4 (блок 128,
  `scale = absmax/127`, `q ∈ -127..127`, `w = q·scale`) при `--embed int8`,
  плотная fp32/fp16 при `--embed fp32|fp16` или отсутствует при
  `--embed trit` (тогда таблица — TRT1-сегмент в `ternary.trit`); нормы и
  bias — всегда fp32. Нормы/bias — меньше 0,1 % параметров, поэтому fp32
  (точность важнее байтов); эмбеддинг — int8, потому что у 7B это
  544 МЭ против 2,2 ГЭ в fp32 и 1,1 ГЭ в fp16.

Почему два флага, а не один
===========================

Предел контейнера — u32-смещения, то есть 4 ГиБ (ADR 0016). 7B Qwen2.5
Q4_K_M — это 7,07 млрд весов в Linear+lm_head и ещё 0,545 млрд в таблице
словаря:

| Вариант | Размер | Доля от 4 ГиБ |
|---|---|---|
| fp32-масштабы + int8-эмбеддинг (было) | 4,98 ГБ | 116 % |
| fp16-масштабы + int8-эмбеддинг | 4,54 ГБ | 105,7 % |
| fp16-масштабы + тритовый эмбеддинг | 4,10 ГБ | 95 % |

Оба варианта поддержаны форматом (флаг fp16-масштабов в Q4-сегменте,
TRT1-таблица словаря с `quantize_embeddings: trit`), но импортёр писал
только fp32-масштабы и int8-эмбеддинг. Отсюда `--q4-scale-dtype fp32|fp16`
и `--embed int8|trit|fp16`: fp16 экономит 0,44 ГБ, триты в таблице — ещё
0,44 ГБ. Для 7B без обоих 4 ГиБ не достигаются — это не «настройка на
вкус», а единственная достижимая комбинация.

Q4_1 вместо Q4_0: откуда 10 % и куда они делись
==============================================

Шаг переквантования неизбежен: значения K-блока расквантовываются в fp32 и
квантуются заново по нашему правилу. Но правило решает, сколько при этом
теряется. K-блок хранит масштаб **и минимум** на под-блок 32; симметричный
Q4_0 (`scale = absmax/7`, `w = (q-8)·scale`) минимум выбрасывает. Если 32
веса лежат узкой полосой вдали от нуля (у под-блоков Q4_K это обычное дело),
из 16 уровней работают 3–6, шаг сетки в разы больше нужного — отсюда 10 %
rel-RMS и «Столица Франции - это Пгврф» вместо Парижа (задача 0160).

Q4_1 (`w = q·scale + min`) возвращает минимум: `min = min(32)`,
`scale = (max-min)/15`. Цена — второй массив (4 байта на группу 32 при
fp32-масштабах, 2 при fp16). `--stats` печатает относительную RMS-ошибку
реконструкции по ролям — численную оценку ступени.

Замер на реальном `qwen2.5:3b` (Q4_K_M), одни и те же де-квантованные
веса, два правила:

| Множество | Q4_K→Q4_0 (sym) | Q4_K→Q4_1 (asym) |
|---|---|---|
| 1 слой + срез словаря 4096 (77,1 МЭ) | 9,96 % (макс. ошибка 0,049) | **5,63 %** (макс. ошибка 0,040) |
| полная модель (2,77 млрд весов Linear) | — | **5,1 %** (макс. ошибка 0,049) |

То есть ошибка падает почти вдвое, а текст оживает: «Столица Франции - это
Пгврф» (Q4_0) → «Столицей Франции является Париж.» (Q4_1), ровно как у
ollama. Глобальный rel-RMS при этом меняется в 1,8 раза, а не в 10: ошибка
симметричного правила **неравномерна** — она сосредоточена в тех группах,
где полоса весов далеко от нуля, и таких групп достаточно, чтобы сломать
несколько слоёв, но мало для среднего по всем весам. Отсюда правило: на
rel-RMS смотреть как на индикатор, а текст проверять отдельно.
int8-эмбеддинг из Q6_K — **0,6 %** (не менялся: int8-сетка симметрична, но
у неё на блок 128 ступеней вчетверо больше).

Для F16/BF16/F32/Q4_0/Q4_1/Q8_0 потерь меньше: это та же сетка, но с
другим масштабом.

7B и 4 ГиБ
==========

Предел контейнера — u32-смещения, 4 ГиБ (ADR 0016). Q4_1 стоит +2 байта
min на группу 32 при fp16-масштабах (+0,0625 Б/вес) и +4 байта при fp32
(+0,125 Б/вес). Для 3B (2,36 млрд весов в Linear) это ~+0,15 ГБ, для 7B —
~+0,44 ГБ: единственная схема, влезавшая у 7B (fp16-масштабы + тритовый
эмбеддинг, 4,10 ГБ), с Q4_1 даёт 4,54 ГБ, то есть 106 % предела. Поэтому
умолчание Q4_1, а 7B импортируется явным `--q4-kind q4_0` — это осознанная
плата за качество на моделях, которые влезают, а не «настройка на вкус».

fp16-масштаб выбирается **до** квантования нибблов: контейнер хранит
округлённый до fp16 масштаб, и C-ядро считает `w = (q-8)·scale_fp16`.
Квантовать по fp32-масштабу и хранить fp16 — значит сложить две ступени
ошибки вместо одной. Переполнение fp16 (> 65504) — отказ с числом, а не
`inf` в контейнере.

`--format trit` — Linear расквантовываются в fp32 и тернаризуются
(масштаб absmax на блок 128, `trit = clamp(round_half_away(w/scale), -1, 1)`)
в TRT1-сегменты `ternary.trit` (kind 0, `packed20`); эмбеддинг — int8 или
триты (`--embed`), нормы/bias — fp32. Для таблицы словаря берётся правило
TWN (`scale = 1,5·mean(|w|)`): постичь-absmax на Q6_K-эмбеддинге 7B даёт
rel-RMS 77 %, а TWN — 43–49 % при тех же байтах сегмента (формат правило не
фиксирует). Качество здесь не цель (триты грубее Q4), а формат и сквозная
проводка. Полный 7B в тритах — ~1,5 ГиБ вместо ~4,4 ГиБ Q4_K.

Компилятор и kind 5
===================

`tools/vtm_compile.py` (задача 0154) читает `q4.bin`: запись с
`dtype: "q4_1"` (или `"q4_0"`), `q4_offset`/`q4_length` разбирается
`vtm_ref.decode_q4_segment_ex` и становится `Tensor(kind=KIND_Q4)`;
`architecture.lm_head='q4'` — `TMV` по kind 5. TRT1- и int8-таблица словаря
компилятор тоже знает (`architecture.embed` = `trit`/`int8`). Поэтому
манифест импорта в обеих схемах собирается без правок компилятора, о чём
раздел `compiler` (`ready`, `missing`) и сообщает.

Память и потоковость
====================

Модель целиком в RAM не поднимается: тензор читается блоками по
`CHUNK_VALUES` значений, квантуется/пишется и отпускается. Хвост сегмента
(Q4-нибблы, int8-байты) складывается в `SpooledTemporaryFile` (в памяти до
64 МиБ, дальше — временный файл): по формату масштабы идут перед данными,
а масштабы становятся известны только после прохода. Пик RSS на 7B —
десятки мегабайт независимо от размера тензора.

Ограничения
===========

* Поддержаны Qwen2/Llama-имена. Qwen3 (`attn_q_norm` и т. п.) отвергается
  по имени тензора.
* `rope.scaling` (linear/yarn) не реализован в VTM — импорт отказывает, а не
  пишет модель с неверным RoPE.
* `<arch>.attention.head_count`/`head_count_kv` массивом по слоям не
  поддерживаются (берётся только одинаковое значение).
* `--limit-layers N` / `--limit-vocab V` дают **частичную** модель: это
  дымовой тест формата, а не качество. Манифест помечается
  `"partial": true`.

CLI
===

```
python3 tools/gguf_import.py info ~/.ollama/models/blobs/sha256-... [--json]
python3 tools/gguf_import.py import model.gguf --out export/gguf-qwen --format q4
python3 tools/gguf_import.py import model.gguf --out export/gguf-7b --format q4 \
    --q4-kind q4_0 --q4-scale-dtype fp16 --embed trit   # 7B Q4 в предел 4 ГиБ
python3 tools/gguf_import.py import model.gguf --out export/gguf-3b --format q4 \
    --stats                                      # Q4_1 (asym) — умолчание
python3 tools/gguf_import.py import model.gguf --out /tmp/slice --format trit \
    --limit-layers 1 --limit-vocab 4096 --stats
```

Затем:

```
python3 tools/vtm_compile.py --manifest /tmp/slice/manifest.json --out /tmp/slice/model.vtm
src/vtm/vtm run /tmp/slice/model.vtm --tokens 1,2 --max-new 1
```

Импорт можно звать и из кода: `read_gguf`, `infer_config`,
`map_tensor_name`, `import_gguf`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
import tempfile
import time
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

# ---------------------------------------------------------------------------
# Константы формата
# ---------------------------------------------------------------------------

GGUF_MAGIC = b"GGUF"
GGUF_VERSIONS = (2, 3)
DEFAULT_ALIGNMENT = 32
MAX_METADATA = 1 << 20
MAX_TENSORS = 1 << 17
MAX_STRING = 1 << 30

#: Сколько значений тензора читать за один заход (f32-буфер ~4 МиБ).
CHUNK_VALUES = 1 << 20
#: Порог, после которого хвост сегмента уходит из RAM во временный файл.
SPOOL_LIMIT = 64 << 20

Q4_MAGIC = b"Q4_0"
#: Асимметричный Q4_1 (ADR 0017): `w = q·scale + min`, min на группу 32.
Q4_MAGIC_ASYM = b"Q4_1"
Q4_VERSION = 1
Q4_HEADER_SIZE = 18
#: У Q4_1 после `n_groups` идёт явное `n_mins u32` (== n_groups): без него
#: отсутствие min не отличить от выравнивающего padding'а.
Q4_ASYM_HEADER_SIZE = 22
Q4_SCALE_BLOCK = 32
Q4_GROUP_BYTES = 16
Q4_ZERO_NIBBLE = 8
#: Крайние нибблы Q4_1: 0 — это min группы, 15 — max.
Q4_NIBBLE_MIN = 0
Q4_NIBBLE_MAX = 15
Q4_SCALE_MAX = 7.0
#: Бит 0 flags Q4-сегмента: 1 — масштабы fp32, 0 — fp16 (ADR 0017).
Q4_FLAG_F32_SCALES = 0x01
#: Сколько байт занимает один масштаб Q4-сегмента.
Q4_SCALE_BYTES = {"float32": 4, "float16": 2}
TRT1_MAGIC = b"TRT1"
TRT1_VERSION = 1
TRT1_HEADER_SIZE = 18
TRITS_PER_WORD = 20
TRITS_PER_STORAGE_BLOCK = 160
TRIT_SCALE_BLOCK = 128
#: Множитель правила TWN для тритового эмбеддинга: scale = 1,5·mean(|w|).
#: На реальном Qwen2.5-7B эмбеддинге даёт rel-RMS 49 % против 77 % у absmax.
TRIT_MEAN_FACTOR = 1.5
INT8_MAGIC = b"INT8"
INT8_VERSION = 1
INT8_SCALE_BLOCK = 128
INT8_MIN = -127
INT8_MAX = 127

#: 18-байтовый заголовок сегментов kind 4/5: magic, version, flags,
#: n_values u64, n_scale u32 (тот же `_TRT1_STRUCT`, что в tools/vtm_ref.py).
_SEGMENT_HEADER = struct.Struct("<4sBBQI")
_F32 = struct.Struct("<f")
_F16 = struct.Struct("<e")

MANIFEST_FORMAT = "eva32.ternary.manifest"
MANIFEST_VERSION = 2
Q4_FILE = "q4.bin"
NON_TERNARY_FILE = "non_ternary.bin"
MANIFEST_FILE = "manifest.json"

#: Предел контейнера: смещения u32, то есть 4 ГиБ (ADR 0016). По нему
#: выбирается `--q4-kind auto`: Q4_1 дороже на 2–4 байта на группу 32, и у 7B
#: эта цена выводит за предел.
CONTAINER_LIMIT = 1 << 32
#: Запас на заголовок графа, инструкции, integrity-хэши и выравнивание.
CONTAINER_MARGIN = 16 << 20

#: Допустимые хранилища таблицы словаря: int8 (kind 4), триты (kind 0) и
#: плотные fp32/fp16 (kind 1/2). Имена-синонимы понимает CLI.
EMBED_MODES = ("int8", "trit", "float32", "float16")
#: Вид Q4-сегмента (kind 5): Q4_1 — импорт по умолчанию (минимум группы,
#: 4 байта/группа дороже), Q4_0 — симметричный, нужен 7B: у него бюджет
#: 4 ГиБ (ADR 0016) не сходится с min'ами (см. докстроку модуля).
Q4_KINDS = ("q4_0", "q4_1")
_Q4_KIND_ALIASES = {
    "q4_0": "q4_0",
    "q4-0": "q4_0",
    "q40": "q4_0",
    "sym": "q4_0",
    "symmetric": "q4_0",
    "q4_1": "q4_1",
    "q4-1": "q4_1",
    "q41": "q4_1",
    "asym": "q4_1",
    "asymmetric": "q4_1",
}
_EMBED_ALIASES = {
    "int8": "int8",
    "trit": "trit",
    "fp32": "float32",
    "float32": "float32",
    "f32": "float32",
    "fp16": "float16",
    "float16": "float16",
    "f16": "float16",
}
_SCALE_DTYPE_ALIASES = {
    "fp32": "float32",
    "float32": "float32",
    "f32": "float32",
    "fp16": "float16",
    "float16": "float16",
    "f16": "float16",
}


class GgufError(ValueError):
    """GGUF не читается или не поддерживается (текст — для человека)."""


# ---------------------------------------------------------------------------
# Типы GGML
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GgufType:
    """Описание типа GGML: сколько значений в блоке и сколько байт блок."""

    type_id: int
    name: str
    block: int
    block_bytes: int
    kind: str  # f32 | f16 | bf16 | q4_0 | q4_1 | q8_0 | q4_k | q6_k

    @property
    def bytes_per_value(self) -> float:
        return self.block_bytes / self.block


GGUF_TYPES: dict[int, GgufType] = {
    0: GgufType(0, "F32", 1, 4, "f32"),
    1: GgufType(1, "F16", 1, 2, "f16"),
    2: GgufType(2, "Q4_0", 32, 18, "q4_0"),
    3: GgufType(3, "Q4_1", 32, 20, "q4_1"),
    8: GgufType(8, "Q8_0", 32, 34, "q8_0"),
    12: GgufType(12, "Q4_K", 256, 144, "q4_k"),
    14: GgufType(14, "Q6_K", 256, 210, "q6_k"),
    30: GgufType(30, "BF16", 1, 2, "bf16"),
}

#: Имена всех известных ggml-типов — чтобы отказ называл тип, а не число.
GGUF_TYPE_NAMES: dict[int, str] = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 4: "Q4_2", 5: "Q4_3",
    6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
    16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL",
    21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32",
    27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16",
    31: "Q4_0_4_4", 32: "Q4_0_4_8", 33: "Q4_0_8_8", 34: "TQ1_0",
    35: "TQ2_0", 36: "IQ4_NL_4_4", 37: "IQ4_NL_4_8", 38: "IQ4_NL_8_8",
    39: "MXFP4",
}

SUPPORTED_TYPE_NAMES = ", ".join(spec.name for spec in GGUF_TYPES.values())


def type_label(type_id: int) -> str:
    """Имя типа для сообщений: «Q4_K (12)» или «тип 99 (неизвестен)»."""
    name = GGUF_TYPE_NAMES.get(type_id)
    if name is None:
        return f"тип {type_id} (неизвестен)"
    return f"{name} ({type_id})"


# ---------------------------------------------------------------------------
# Чтение GGUF
# ---------------------------------------------------------------------------

#: Тип значения метаданных -> (struct-формат, размер). Индексы — значения
#: `gguf_metadata_value_type` из спецификации GGUF.
_VALUE_FORMATS: dict[int, str] = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
    10: "<Q", 11: "<q", 12: "<d",
}
_VALUE_SIZES: dict[int, int] = {
    vtype: struct.calcsize(fmt) for vtype, fmt in _VALUE_FORMATS.items()
}


@dataclass(frozen=True)
class TensorInfo:
    """Запись таблицы тензоров GGUF."""

    name: str
    dims: tuple[int, ...]  # порядок GGUF: ne0 — младшая (входная)
    type_id: int
    offset: int  # от начала секции данных

    @property
    def spec(self) -> GgufType:
        try:
            return GGUF_TYPES[self.type_id]
        except KeyError:
            raise GgufError(
                f"тензор {self.name!r}: тип {type_label(self.type_id)} "
                f"не поддерживается (поддержаны: {SUPPORTED_TYPE_NAMES})"
            ) from None

    @property
    def n_elements(self) -> int:
        result = 1
        for dim in self.dims:
            result *= int(dim)
        return result

    @property
    def n_bytes(self) -> int:
        spec = self.spec
        if self.n_elements % spec.block:
            raise GgufError(
                f"тензор {self.name!r}: {self.n_elements} значений не кратны "
                f"блоку {spec.block} типа {spec.name}"
            )
        return self.n_elements // spec.block * spec.block_bytes

    @property
    def hf_shape(self) -> tuple[int, ...]:
        """Форма в понимании HF/QAT: старшая размерность первой."""
        return tuple(reversed(self.dims))


@dataclass
class GgufFile:
    """Разобранный заголовок GGUF: метаданные, тензоры, смещение данных."""

    path: Path
    version: int
    metadata: dict[str, Any]
    array_lengths: dict[str, int]
    tensors: list[TensorInfo]
    data_offset: int
    alignment: int
    file_size: int

    def tensor(self, name: str) -> TensorInfo:
        for info in self.tensors:
            if info.name == name:
                return info
        raise GgufError(f"в GGUF нет тензора {name!r}")

    def has_tensor(self, name: str) -> bool:
        return any(info.name == name for info in self.tensors)

    def metadata_scalar(self, key: str) -> Any:
        return self.metadata.get(key)

    # -- чтение данных -------------------------------------------------------

    def iter_tensor_chunks(
        self, info: TensorInfo, n_values: int | None = None
    ) -> Iterator[array]:
        """Отдать значения тензора (fp32) кусками, начиная с первого.

        `n_values` ограничивает чтение префиксом (срез словаря/слоёв):
        реального смещения это не требует — строки таблицы словаря лежат
        подряд с начала. Границы кусков кратны блоку типа, поэтому блок
        никогда не разрезается; хвост последнего куска обрезается.
        """
        spec = info.spec
        total = info.n_elements
        if n_values is None:
            n_values = total
        if not 0 < n_values <= total:
            raise GgufError(
                f"тензор {info.name!r}: запрошено {n_values} значений, "
                f"в тензоре {total}"
            )
        read_values = _align_up(n_values, spec.block)
        start_byte = self.data_offset + info.offset
        chunk_values = max(
            spec.block, (CHUNK_VALUES // spec.block) * spec.block
        )
        with open(self.path, "rb") as handle:
            handle.seek(start_byte)
            done = 0
            while done < read_values:
                want = min(chunk_values, read_values - done)
                raw = handle.read(want // spec.block * spec.block_bytes)
                if len(raw) != want // spec.block * spec.block_bytes:
                    raise GgufError(
                        f"тензор {info.name!r}: файл обрезан на значении "
                        f"{done} (нужно {want}, прочитано {len(raw)} байт)"
                    )
                values = _dequantize(spec, raw, want)
                done += want
                if done > n_values:
                    values = values[: n_values - (done - want)]
                yield values

    def raw_tensor(self, info: TensorInfo) -> bytes:
        """Сырые байты тензора (для тестов и `info`)."""
        with open(self.path, "rb") as handle:
            handle.seek(self.data_offset + info.offset)
            data = handle.read(info.n_bytes)
        if len(data) != info.n_bytes:
            raise GgufError(f"тензор {info.name!r}: файл обрезан")
        return data


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _read_string(handle: Any, where: str) -> str:
    raw = handle.read(8)
    if len(raw) != 8:
        raise GgufError(f"{where}: файл обрезан на длине строки")
    (length,) = struct.unpack("<Q", raw)
    if length > MAX_STRING:
        raise GgufError(f"{where}: строка длиной {length} байт — это порча данных")
    data = handle.read(length)
    if len(data) != length:
        raise GgufError(f"{where}: файл обрезан на строке")
    return data.decode("utf-8", "replace")


def _skip_array(handle: Any, elem_type: int, count: int, where: str) -> None:
    """Пропустить массив метаданных, не материализуя его."""
    if count < 0 or count > (1 << 32):
        raise GgufError(f"{where}: массив длиной {count} — порча данных")
    if elem_type in _VALUE_SIZES:
        need = count * _VALUE_SIZES[elem_type]
        if need:
            handle.seek(need, os.SEEK_CUR)
        return
    if elem_type == 8:  # массив строк: длины читаем, байты пропускаем
        for _ in range(count):
            raw = handle.read(8)
            if len(raw) != 8:
                raise GgufError(f"{where}: массив строк обрезан")
            (length,) = struct.unpack("<Q", raw)
            if length > MAX_STRING:
                raise GgufError(f"{where}: строка длиной {length} в массиве")
            if length:
                handle.seek(length, os.SEEK_CUR)
        return
    if elem_type == 9:
        raise GgufError(f"{where}: вложенные массивы в метаданных не поддержаны")
    raise GgufError(f"{where}: неизвестный тип элемента массива {elem_type}")


def _read_metadata_value(
    handle: Any, value_type: int, key: str, array_lengths: dict[str, int]
) -> Any:
    fmt = _VALUE_FORMATS.get(value_type)
    if fmt is not None:
        size = _VALUE_SIZES[value_type]
        raw = handle.read(size)
        if len(raw) != size:
            raise GgufError(f"метаданные: файл обрезан на {key!r}")
        return struct.unpack(fmt, raw)[0]
    if value_type == 8:
        return _read_string(handle, f"метаданные {key!r}")
    if value_type == 9:
        head = handle.read(12)
        if len(head) != 12:
            raise GgufError(f"метаданные: файл обрезан на массиве {key!r}")
        elem_type, count = struct.unpack("<IQ", head)
        array_lengths[key] = int(count)
        if elem_type in _VALUE_FORMATS and count <= 4096:
            fmt = _VALUE_FORMATS[elem_type]
            size = _VALUE_SIZES[elem_type]
            raw = handle.read(size * count)
            if len(raw) != size * count:
                raise GgufError(f"метаданные: массив {key!r} обрезан")
            if count == 0:
                return []
            return list(struct.unpack(f"<{count}{fmt[1:]}", raw))
        _skip_array(handle, elem_type, int(count), f"метаданные {key!r}")
        return None
    raise GgufError(
        f"метаданные {key!r}: тип значения {value_type} не поддержан"
    )


def read_gguf(path: str | os.PathLike[str]) -> GgufFile:
    """Прочитать заголовок GGUF (v2/v3) без загрузки весов."""
    path = Path(path).expanduser()
    try:
        size = path.stat().st_size
        handle = open(path, "rb")
    except OSError as exc:
        raise GgufError(f"не открывается {path}: {exc}") from exc
    with handle:
        magic = handle.read(4)
        if magic != GGUF_MAGIC:
            raise GgufError(
                f"{path.name}: это не GGUF (magic {magic!r}, ожидался b'GGUF')"
            )
        raw = handle.read(4)
        if len(raw) != 4:
            raise GgufError(f"{path.name}: файл обрезан на версии")
        (version,) = struct.unpack("<I", raw)
        if version not in GGUF_VERSIONS:
            if version == 1:
                raise GgufError(
                    f"{path.name}: GGUF v1 не поддерживается (нужны v2/v3)"
                )
            raise GgufError(
                f"{path.name}: версия GGUF {version} не поддерживается "
                f"(нужны v2/v3)"
            )
        head = handle.read(16)
        if len(head) != 16:
            raise GgufError(f"{path.name}: файл обрезан на заголовке")
        n_tensors, n_kv = struct.unpack("<QQ", head)
        if n_kv > MAX_METADATA:
            raise GgufError(f"{path.name}: {n_kv} метаданных — порча данных")
        if n_tensors > MAX_TENSORS:
            raise GgufError(f"{path.name}: {n_tensors} тензоров — порча данных")

        metadata: dict[str, Any] = {}
        array_lengths: dict[str, int] = {}
        for _ in range(n_kv):
            key = _read_string(handle, f"{path.name}: метаданные")
            raw = handle.read(4)
            if len(raw) != 4:
                raise GgufError(f"{path.name}: файл обрезан на типе {key!r}")
            (value_type,) = struct.unpack("<I", raw)
            metadata[key] = _read_metadata_value(
                handle, value_type, key, array_lengths
            )

        tensors: list[TensorInfo] = []
        for index in range(n_tensors):
            name = _read_string(handle, f"{path.name}: тензор #{index}")
            raw = handle.read(4)
            if len(raw) != 4:
                raise GgufError(f"{path.name}: файл обрезан на {name!r}")
            (n_dims,) = struct.unpack("<I", raw)
            if not 1 <= n_dims <= 4:
                raise GgufError(
                    f"{path.name}: тензор {name!r} с {n_dims} измерениями "
                    "(поддержано 1..4)"
                )
            raw = handle.read(8 * n_dims)
            if len(raw) != 8 * n_dims:
                raise GgufError(f"{path.name}: файл обрезан на dims {name!r}")
            dims = struct.unpack(f"<{n_dims}Q", raw)
            raw = handle.read(12)
            if len(raw) != 12:
                raise GgufError(f"{path.name}: файл обрезан на {name!r}")
            type_id, offset = struct.unpack("<IQ", raw)
            tensors.append(
                TensorInfo(
                    name=name,
                    dims=tuple(int(dim) for dim in dims),
                    type_id=int(type_id),
                    offset=int(offset),
                )
            )

        position = handle.tell()
        alignment = metadata.get("general.alignment", DEFAULT_ALIGNMENT)
        try:
            alignment = int(alignment)
        except (TypeError, ValueError):
            raise GgufError(
                f"{path.name}: general.alignment={alignment!r} не число"
            ) from None
        if alignment <= 0 or alignment & (alignment - 1) or alignment > (1 << 20):
            raise GgufError(
                f"{path.name}: general.alignment={alignment} не степень двойки "
                "в разумных пределах"
            )
        data_offset = _align_up(position, alignment)

    if data_offset > size:
        raise GgufError(
            f"{path.name}: секция данных начинается за файлом "
            f"({data_offset} > {size})"
        )
    for info in tensors:
        try:
            n_bytes = info.n_bytes
        except GgufError:
            continue  # неподдержанный тип проверится при импорте
        if info.offset + n_bytes > size - data_offset:
            raise GgufError(
                f"{path.name}: тензор {info.name!r} выходит за файл: "
                f"offset {info.offset} + {n_bytes} > {size - data_offset}"
            )
    return GgufFile(
        path=path,
        version=version,
        metadata=metadata,
        array_lengths=array_lengths,
        tensors=tensors,
        data_offset=data_offset,
        alignment=alignment,
        file_size=size,
    )


# ---------------------------------------------------------------------------
# Деквантование в fp32
# ---------------------------------------------------------------------------

_LOW_NIBBLE = bytes(i & 0x0F for i in range(256))
_HIGH_NIBBLE = bytes(i >> 4 for i in range(256))
_TWO_BITS_SHIFT = {
    shift: bytes((i >> shift) & 0x03 for i in range(256))
    for shift in (0, 2, 4, 6)
}
_EMPTY16 = bytes(16)


def _pack_nibbles(low: bytes, high: bytes) -> bytes:
    """Склеить два массива 4-битных значений в байты `low | high << 4`."""
    return (
        int.from_bytes(low, "little") | (int.from_bytes(high, "little") << 4)
    ).to_bytes(len(low), "little")


def _scale_min_k4(j: int, scales: bytes) -> tuple[int, int]:
    """6-битные (scale, min) под-блока `j` из 12 байт Q4_K — как ggml."""
    if j < 4:
        return scales[j] & 0x3F, scales[j + 4] & 0x3F
    return (
        (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4),
        (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
    )


def _dequantize(spec: GgufType, raw: bytes, n_values: int) -> array:
    """Расквантовать блоки `raw` в fp32-массив длины `n_values`."""
    kind = spec.kind
    if kind == "f32":
        values = array("f")
        values.frombytes(raw)
        if sys.byteorder != "little":
            values.byteswap()
        return values
    if kind == "f16":
        return array("f", struct.unpack(f"<{n_values}e", raw))
    if kind == "bf16":
        words = struct.unpack(f"<{n_values}H", raw)
        bits = array("I", (word << 16 for word in words))
        if sys.byteorder != "little":
            bits.byteswap()
        values = array("f")
        values.frombytes(bits.tobytes())
        return values
    if kind == "q4_0":
        return _dequant_q4_0(raw, spec)
    if kind == "q4_1":
        return _dequant_q4_1(raw, spec)
    if kind == "q8_0":
        return _dequant_q8_0(raw, spec)
    if kind == "q4_k":
        return _dequant_q4_k(raw, spec)
    if kind == "q6_k":
        return _dequant_q6_k(raw, spec)
    raise GgufError(f"тип {spec.name} не поддержан деквантованием")


def _dequant_q4_0(raw: bytes, spec: GgufType) -> array:
    out = array("f")
    extend = out.extend
    for offset in range(0, len(raw), spec.block_bytes):
        (d,) = _F16.unpack_from(raw, offset)
        group = raw[offset + 2: offset + 18]
        table = [d * (value - 8) for value in range(16)]
        extend(map(table.__getitem__, group.translate(_LOW_NIBBLE)))
        extend(map(table.__getitem__, group.translate(_HIGH_NIBBLE)))
    return out


def _dequant_q4_1(raw: bytes, spec: GgufType) -> array:
    out = array("f")
    extend = out.extend
    for offset in range(0, len(raw), spec.block_bytes):
        (d, m) = struct.unpack_from("<ee", raw, offset)
        group = raw[offset + 4: offset + 20]
        table = [d * value + m for value in range(16)]
        extend(map(table.__getitem__, group.translate(_LOW_NIBBLE)))
        extend(map(table.__getitem__, group.translate(_HIGH_NIBBLE)))
    return out


def _dequant_q8_0(raw: bytes, spec: GgufType) -> array:
    out = array("f")
    extend = out.extend
    for offset in range(0, len(raw), spec.block_bytes):
        (d,) = _F16.unpack_from(raw, offset)
        extend(d * value for value in struct.unpack_from("<32b", raw, offset + 2))
    return out


def _dequant_q4_k(raw: bytes, spec: GgufType) -> array:
    out = array("f")
    extend = out.extend
    for offset in range(0, len(raw), spec.block_bytes):
        (d, dmin) = struct.unpack_from("<ee", raw, offset)
        scales = raw[offset + 4: offset + 16]
        qs = raw[offset + 16: offset + 144]
        for chunk in range(4):  # 64 значений на 32 байта qs
            group = qs[chunk * 32: chunk * 32 + 32]
            low = group.translate(_LOW_NIBBLE)
            high = group.translate(_HIGH_NIBBLE)
            for nibbles, sub in ((low, 2 * chunk), (high, 2 * chunk + 1)):
                scale, minimum = _scale_min_k4(sub, scales)
                dd = d * scale
                mm = dmin * minimum
                table = [dd * value - mm for value in range(16)]
                extend(map(table.__getitem__, nibbles))
    return out


def _dequant_q6_k(raw: bytes, spec: GgufType) -> array:
    out = array("f")
    extend = out.extend
    for offset in range(0, len(raw), spec.block_bytes):
        (d,) = _F16.unpack_from(raw, offset + 208)
        scales = struct.unpack_from("<16b", raw, offset + 192)
        # Блок Q6_K — 256 значений: ql 128 байт, qh 64 байта, scales 16 байт.
        # Внутри него два независимых 128-значных куска (по 4 под-блока).
        for group in (0, 1):
            base = offset + group * 64
            ql = raw[base: base + 64]
            qh = raw[offset + 128 + group * 32:
                     offset + 128 + group * 32 + 32]
            low0 = ql[:32].translate(_LOW_NIBBLE)
            high0 = ql[:32].translate(_HIGH_NIBBLE)
            low1 = ql[32:64].translate(_LOW_NIBBLE)
            high1 = ql[32:64].translate(_HIGH_NIBBLE)
            quads = (
                _pack_nibbles(low0, qh.translate(_TWO_BITS_SHIFT[0])),
                _pack_nibbles(low1, qh.translate(_TWO_BITS_SHIFT[2])),
                _pack_nibbles(high0, qh.translate(_TWO_BITS_SHIFT[4])),
                _pack_nibbles(high1, qh.translate(_TWO_BITS_SHIFT[6])),
            )
            for quad_index, quads_data in enumerate(quads):
                for half in (0, 1):
                    multiplier = d * scales[group * 8 + quad_index * 2 + half]
                    table = [multiplier * (value - 32) for value in range(64)]
                    extend(map(
                        table.__getitem__,
                        quads_data[half * 16: half * 16 + 16],
                    ))
    return out


# ---------------------------------------------------------------------------
# Квантование: Q4_0 (kind 5) и INT8 (kind 4)
# ---------------------------------------------------------------------------


def _f32(value: float) -> float:
    return _F32.unpack(_F32.pack(float(value)))[0]


def _f16(value: float) -> float:
    """Округлить до float16 (как это сделает формат при записи масштаба).

    Переполнение — не «обрезать до inf»: масштаб `absmax/7 > 65504` означает,
    что в fp16 тензор не записать, и импорт должен отказать с числом, а не
    положить inf в контейнер.
    """
    try:
        return _F16.unpack(_F16.pack(float(value)))[0]
    except (OverflowError, struct.error) as exc:
        raise GgufError(
            f"масштаб {value!r} не влезает в float16 (максимум 65504): "
            "для этого тензора нужен --q4-scale-dtype fp32"
        ) from exc


def _normalize_embed_mode(mode: str) -> str:
    normalized = _EMBED_ALIASES.get(str(mode).lower())
    if normalized is None:
        raise GgufError(
            f"embed={mode!r}, допустимо int8|trit|fp16|fp32 "
            "(синонимы: float16/float32)"
        )
    return normalized


def _normalize_scale_dtype(dtype: str) -> str:
    normalized = _SCALE_DTYPE_ALIASES.get(str(dtype).lower())
    if normalized is None:
        raise GgufError(
            f"q4_scale_dtype={dtype!r}, допустимо fp32|fp16 "
            "(синонимы: float32/float16)"
        )
    return normalized


def _normalize_q4_kind(kind: str) -> str:
    """Нормализовать вид Q4: q4_1/q4_0 или "auto" (выбрать по размеру)."""
    text = str(kind).lower()
    if text in ("auto", "", "none"):
        return "auto"
    normalized = _Q4_KIND_ALIASES.get(text)
    if normalized is None:
        raise GgufError(
            f"q4_kind={kind!r}, допустимо auto (по умолчанию: q4_1, если "
            "контейнер влезает в 4 ГиБ, иначе q4_0) | q4_1 (asym) | q4_0 "
            "(sym; синонимы: asym/q4-1 и sym/q4-0)"
        )
    return normalized


def estimate_container_bytes(
    plans: Iterable["PlannedTensor"],
    *,
    q4_asym: bool,
    scale_dtype: str,
) -> int:
    """Оценка суммы payload'ов контейнера в байтах (для выбора q4_kind).

    Считается по плану импорта, до чтения весов: Q4-сегменты (заголовок,
    масштабы [+ min], нибблы), INT8/TRT1-сегменты, плотные fp32/fp16 и
    выравнивание сегментов по 32 байта. Это именно оценка бюджета
    (ADR 0016: u32-смещения), а не размер готового файла.
    """
    scale_bytes = Q4_SCALE_BYTES[scale_dtype]
    total = 0
    for plan in plans:
        n_values = plan.n_values
        if plan.target == "q4":
            groups = (n_values + Q4_SCALE_BLOCK - 1) // Q4_SCALE_BLOCK
            total += (
                (Q4_ASYM_HEADER_SIZE if q4_asym else Q4_HEADER_SIZE)
                + groups * scale_bytes * (2 if q4_asym else 1)
                + groups * Q4_GROUP_BYTES
            )
        elif plan.target == "int8":
            blocks = (n_values + INT8_SCALE_BLOCK - 1) // INT8_SCALE_BLOCK
            total += Q4_HEADER_SIZE + blocks * 4 + n_values
        elif plan.target == "trit":
            scales = (n_values + TRIT_SCALE_BLOCK - 1) // TRIT_SCALE_BLOCK
            words = (
                (n_values + TRITS_PER_STORAGE_BLOCK - 1)
                // TRITS_PER_STORAGE_BLOCK * 8
            )
            total += TRT1_HEADER_SIZE + scales * 4 + words * 4
        elif plan.target == "f16":
            total += n_values * 2
        else:  # f32
            total += n_values * 4
        total += 32  # выравнивание сегмента
    return total


def _round_half_away(value: float) -> int:
    if value >= 0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


_TRIT_TO_DIGIT = {-1: 0, 0: 1, 1: 2}


def _pack20(trits: Sequence[int]) -> bytes:
    """Упаковать триты по 20 в u32 (младший трит — младший разряд).

    Хвост дополняется нулевыми тритами (цифра 1) до 20 — ровно как
    `tools/trit_codec.py:pack20` и `qat/ternary.py:pack20`.
    """
    values = list(trits)
    pad = (-len(values)) % TRITS_PER_WORD
    values.extend([0] * pad)
    out = bytearray()
    for start in range(0, len(values), TRITS_PER_WORD):
        word = 0
        for index, trit in enumerate(values[start:start + TRITS_PER_WORD]):
            word += _TRIT_TO_DIGIT[int(trit)] * (3 ** index)
        out += word.to_bytes(4, "little")
    return bytes(out)


def _trit_segment_header(n_values: int, n_scale: int) -> bytes:
    return _SEGMENT_HEADER.pack(
        TRT1_MAGIC, TRT1_VERSION, 0x01, n_values, n_scale
    )


def _trit_words_padded(n_values: int) -> int:
    return (n_values + TRITS_PER_STORAGE_BLOCK - 1) // TRITS_PER_STORAGE_BLOCK * 8


def _q4_nibble(value: float, scale: float) -> int:
    """Ниббл Q4_0 по правилу 0151: w=(q-8)·scale, |w| <= 7·scale.

    Отношение округляется до float32 перед округлением к целому — ровно как
    `vtm_ref.quantize_q4`, иначе редкие значения на границе .5 дают другой
    ниббл (и сверка с эталоном 0151 расходится).
    """
    signed = _round_half_away(_f32(value / scale))
    if signed < -8:
        signed = -8
    elif signed > 7:
        signed = 7
    return signed + Q4_ZERO_NIBBLE


def _q4_nibble_asym(value: float, scale: float, minimum: float) -> int:
    """Ниббл Q4_1: w = q·scale + min, q в 0..15 (как `quantize_q4` asym).

    Как и у Q4_0, отношение приводится к float32 до округления: эталон
    `vtm_ref.quantize_q4` делает `_f32((w - min) / scale)`, и без этого
    редкие значения на границе .5 дали бы другой ниббл.
    """
    q = _round_half_away(_f32((value - minimum) / scale))
    if q < Q4_NIBBLE_MIN:
        return Q4_NIBBLE_MIN
    if q > Q4_NIBBLE_MAX:
        return Q4_NIBBLE_MAX
    return q


def _q4_encode_block(
    block: array, scale_dtype: str = "float32"
) -> tuple[float, bytes]:
    """Один блок Q4_0 (симметричный) из 32 значений: (масштаб, 16 байт).

    Импорт это правило **не** использует: Q4_K несёт минимум под-блока, и
    отбрасывать его — та самая ступень в 10 % rel-RMS (см. докстроку модуля).
    Функция остаётся эталоном для сравнения и для тестов формата Q4_0.

    При `scale_dtype="float16"` масштаб округляется до fp16 **до** выбора
    нибблов: контейнер хранит fp16-масштаб, и C-ядро считает
    `w = (q-8)·scale_fp16`. Если квантовать по fp32-масштабу, а хранить fp16,
    ошибка сложилась бы из двух ступеней.
    """
    top = max(block)
    bottom = min(block)
    absmax = top if top > -bottom else -bottom
    if not math.isfinite(absmax):
        raise GgufError(
            "Q4: в блоке неконечное значение — исходный тензор повреждён"
        )
    if absmax == 0.0:
        return 0.0, bytes([0x88]) * Q4_GROUP_BYTES
    scale = _f32(absmax / Q4_SCALE_MAX)
    if scale_dtype == "float16":
        scale = _f16(scale)
    if scale == 0.0:
        # absmax > 0, но масштаб не представим (fp16-субнормал): все веса
        # блока ниже разрешения сетки, единственное честное приближение — 0.
        return 0.0, bytes([0x88]) * Q4_GROUP_BYTES
    low = bytes(_q4_nibble(value, scale) for value in block[:16])
    high = bytes(_q4_nibble(value, scale) for value in block[16:])
    return scale, _pack_nibbles(low, high)


def _q4_encode_asym_block(
    block: array, scale_dtype: str = "float32"
) -> tuple[float, float, bytes]:
    """Один блок Q4_1 из 32 значений: (min, масштаб, 16 байт нибблов).

    Правило то же, что у `vtm_ref.quantize_q4(..., symmetric=False)`:
    `min = min(блок)`, `scale = (max - min)/15`,
    `q = clamp(round_half_away((w - min)/scale), 0, 15)`. Ноль группы —
    вырожденный случай: `scale = 0`, `min` равен значению блока, все нибблы
    0 — реконструкция `0·0 + min` точная (терять нечего, в отличие от Q4_0,
    где нулевой блок не представим без потери знака).

    Q4_K-подблок хранит и масштаб, и минимум 32 значений; поэтому `min`
    группы возвращает почти всю информацию, которую терял Q4_0 (`absmax/7`
    без минимума).

    При `scale_dtype="float16"` масштаб и min округляются до fp16 **до**
    выбора нибблов: контейнер хранит именно эти числа, и C-ядро считает
    `w = q·scale_fp16 + min_fp16`.
    """
    top = max(block)
    bottom = min(block)
    if not (math.isfinite(top) and math.isfinite(bottom)):
        raise GgufError(
            "Q4_1: в блоке неконечное значение — исходный тензор повреждён"
        )
    if top == bottom:
        # Вырожденная группа: одна точка, q=0, реконструкция точная.
        minimum = _f32(bottom)
        if scale_dtype == "float16":
            minimum = _f16(minimum)
        return minimum, 0.0, bytes(Q4_GROUP_BYTES)
    minimum = _f32(bottom)
    scale = _f32((_f32(top) - minimum) / (Q4_NIBBLE_MAX - Q4_NIBBLE_MIN))
    if scale_dtype == "float16":
        scale = _f16(scale)
        minimum = _f16(minimum)
    if scale == 0.0:
        # (max-min) не представим в fp16-субнормалах: все веса блока лежат
        # ниже разрешения сетки, единственное честное приближение — min.
        return minimum, 0.0, bytes(Q4_GROUP_BYTES)
    low = bytes(
        _q4_nibble_asym(value, scale, minimum) for value in block[:16]
    )
    high = bytes(
        _q4_nibble_asym(value, scale, minimum) for value in block[16:]
    )
    return minimum, scale, _pack_nibbles(low, high)


@dataclass
class _Stats:
    """Относительная ошибка реконструкции (для отчёта о ступени потери)."""

    values: int = 0
    sum_sq: float = 0.0
    err_sq: float = 0.0
    max_abs: float = 0.0

    def add(self, value: float, reconstructed: float) -> None:
        self.values += 1
        self.sum_sq += value * value
        error = value - reconstructed
        self.err_sq += error * error
        if abs(error) > self.max_abs:
            self.max_abs = abs(error)

    def merge(self, other: "_Stats") -> None:
        self.values += other.values
        self.sum_sq += other.sum_sq
        self.err_sq += other.err_sq
        self.max_abs = max(self.max_abs, other.max_abs)

    @property
    def rel_rms(self) -> float:
        if self.values == 0 or self.sum_sq == 0.0:
            return 0.0
        return math.sqrt(self.err_sq / self.sum_sq)


class _TritEncoder:
    """Накопитель TRT1-сегмента kind 0 (packed20) для формата компилятора.

    Масштаб — на блок 128, трит — `clamp(round_half_away(w/scale), -1, 1)`.
    Правило масштаба задаётся `scale_rule`:

    * `absmax` — как `qat/ternary.py` и Python-эталон VTM (Linear `--format
      trit`);
    * `mean` — `scale = 1,5 · mean(|w|)` (TWN). Для таблицы словаря это
      заметно точнее: у 7B Q6_K-эмбеддинга постичь absmax даёт rel-RMS 77 %,
      а mean — 49 % (обучать таблицу мы не можем, это размен формата, а не
      качества обучения). Формат не фиксирует правило — только то, что
      масштаб конечен и неотрицателен, — поэтому обе схемы читаются
      компилятором и C одинаково.

    Слова pack20 идут после масштабов, поэтому сначала копятся в spool, а
    заголовок пишется в конце.
    """

    def __init__(self, collect_stats: bool = False, scale_rule: str = "absmax"):
        if scale_rule not in ("absmax", "mean"):
            raise ValueError(f"scale_rule={scale_rule!r}")
        self.scales = array("f")
        self.body = tempfile.SpooledTemporaryFile(max_size=SPOOL_LIMIT)
        self.n_values = 0
        self.collect_stats = bool(collect_stats)
        self.scale_rule = scale_rule
        self.stats = _Stats()
        self._pending: list[int] = []

    def _block_scale(self, block: array) -> float:
        if self.scale_rule == "mean":
            total = 0.0
            for value in block:
                total += value if value >= 0.0 else -value
            return _f32(total / len(block) * TRIT_MEAN_FACTOR) if total else 0.0
        absmax = max(max(block), -min(block))
        if not math.isfinite(absmax):
            raise GgufError("триты: в блоке неконечное значение")
        return _f32(absmax) if absmax > 0 else 0.0

    def add_values(self, values: array) -> None:
        if len(values) % TRIT_SCALE_BLOCK:
            raise GgufError(
                "триты: длина куска не кратна блоку 128 — ошибка нарезки"
            )
        append_scale = self.scales.append
        pending = self._pending
        body = self.body
        for start in range(0, len(values), TRIT_SCALE_BLOCK):
            block = values[start:start + TRIT_SCALE_BLOCK]
            scale = self._block_scale(block)
            append_scale(scale)
            for value in block:
                if scale == 0.0:
                    trit = 0
                else:
                    trit = _round_half_away(value / scale)
                    if trit < -1:
                        trit = -1
                    elif trit > 1:
                        trit = 1
                pending.append(trit)
                if self.collect_stats:
                    self.stats.add(value, trit * scale)
            while len(pending) >= TRITS_PER_WORD:
                body.write(_pack20(pending[:TRITS_PER_WORD]))
                del pending[:TRITS_PER_WORD]
        self.n_values += len(values)

    def n_scale(self) -> int:
        return (self.n_values + TRIT_SCALE_BLOCK - 1) // TRIT_SCALE_BLOCK

    def write_to(self, sink: "_Sink") -> int:
        if self.n_values == 0:
            raise GgufError("пустой TRT1-сегмент")
        words = _trit_words_padded(self.n_values)
        while self._pending:
            take = self._pending[:TRITS_PER_WORD]
            self._pending = self._pending[TRITS_PER_WORD:]
            take.extend([0] * (TRITS_PER_WORD - len(take)))
            self.body.write(_pack20(take))
        have = self.body.tell() // 4
        if have > words:
            raise GgufError(
                f"тритов упаковано в {have} слов, а формат допускает {words}"
            )
        if have < words:
            self.body.seek(0, os.SEEK_END)
            self.body.write(bytes((words - have) * 4))
        self.body.seek(0)
        n_scale = self.n_scale()
        if len(self.scales) != n_scale:
            raise GgufError(
                f"триты: масштабов {len(self.scales)}, ожидалось {n_scale}"
            )
        start = sink.offset
        sink.write(_trit_segment_header(self.n_values, n_scale))
        scales = array("f", self.scales)
        if sys.byteorder != "little":
            scales.byteswap()
        sink.write(scales.tobytes())
        while True:
            chunk = self.body.read(1 << 20)
            if not chunk:
                break
            sink.write(chunk)
        return sink.offset - start

    def close(self) -> None:
        self.body.close()


class _SegmentEncoder:
    """Накопитель сегмента kind 4/5: масштабы (+ min Q4_1) и хвост в spool.

    `scale_dtype` относится к Q4 (kind 5): fp32 (flags 0x01) или fp16
    (flags 0x00). У INT8 маска флагов та же по смыслу, но импортёр всегда
    пишет fp32 — менять там нечего (int8-эмбеддинг и так вдвое меньше Q4).

    `asymmetric` — писать Q4_1 вместо Q4_0: у каждой группы 32 значения
    появляется свой `min` (`w = q·scale + min`). Импорт GGUF Q4_K/Q6_K
    использует именно его: исходный K-блок несёт минимум под-блока, и
    симметричный Q4_0 его выбрасывает — отсюда 10 % rel-RMS (см. докстроку
    модуля). `min` копится в `array("f")` параллельно масштабам: у самого
    большого тензора 3B это ~10 млн чисел на массив (40 МБ), а не fp32-копия
    весов.
    """

    def __init__(
        self,
        kind: str,
        collect_stats: bool = False,
        scale_dtype: str = "float32",
        asymmetric: bool = False,
    ):
        if kind not in ("q4", "int8"):
            raise ValueError(kind)
        if scale_dtype not in Q4_SCALE_BYTES:
            raise ValueError(f"scale_dtype={scale_dtype!r}")
        if asymmetric and kind != "q4":
            raise ValueError(
                f"asymmetric=True имеет смысл только для Q4, а kind={kind!r}"
            )
        self.kind = kind
        self.scale_dtype = scale_dtype
        self.asymmetric = bool(asymmetric)
        self.scales = array("f")
        #: min'ы Q4_1 (по одному на группу 32); у Q4_0/INT8 — пусто.
        self.mins = array("f")
        self.body = tempfile.SpooledTemporaryFile(max_size=SPOOL_LIMIT)
        self.n_values = 0
        self.collect_stats = bool(collect_stats)
        self.stats = _Stats()

    @property
    def magic(self) -> bytes:
        """Magic сегмента Q4: "Q4_1" у асимметричного, иначе "Q4_0"."""
        if self.kind != "q4":
            return INT8_MAGIC
        return Q4_MAGIC_ASYM if self.asymmetric else Q4_MAGIC

    @property
    def header_size(self) -> int:
        if self.kind == "q4" and self.asymmetric:
            return Q4_ASYM_HEADER_SIZE
        return Q4_HEADER_SIZE

    @property
    def flags(self) -> int:
        """Байт flags сегмента: бит 0 — масштабы fp32."""
        return Q4_FLAG_F32_SCALES if self.scale_dtype == "float32" else 0x00

    def add_values(self, values: array) -> None:
        if not values:
            return
        if self.kind == "q4":
            self._add_q4(values)
        else:
            self._add_int8(values)

    def _add_q4(self, values: array) -> None:
        if len(values) % Q4_SCALE_BLOCK:
            raise GgufError(
                "Q4: длина куска не кратна 32 — внутренняя ошибка нарезки"
            )
        append_scale = self.scales.append
        append_min = self.mins.append
        body = self.body
        asymmetric = self.asymmetric
        for start in range(0, len(values), Q4_SCALE_BLOCK):
            block = values[start:start + Q4_SCALE_BLOCK]
            if asymmetric:
                minimum, scale, packed = _q4_encode_asym_block(
                    block, self.scale_dtype
                )
                append_min(minimum)
            else:
                scale, packed = _q4_encode_block(block, self.scale_dtype)
            append_scale(scale)
            body.write(packed)
            if self.collect_stats:
                zero = minimum if asymmetric else Q4_ZERO_NIBBLE
                for index, value in enumerate(block):
                    nibble = (
                        (packed[index & 15] & 0x0F)
                        if index < 16 else (packed[index & 15] >> 4)
                    )
                    if asymmetric:
                        # Нулевой масштаб у Q4_1 не теряет min: реконструкция
                        # `0·0 + min` — это значение вырожденной группы.
                        self.stats.add(value, nibble * scale + minimum)
                    elif scale != 0.0:
                        self.stats.add(value, (nibble - zero) * scale)
                    else:
                        # Нулевой/непредставимый масштаб Q4_0: реконструкция —
                        # 0, но ошибку надо учесть, иначе stats занижает
                        # потерю.
                        self.stats.add(value, 0.0)
        self.n_values += len(values)

    def _add_int8(self, values: array) -> None:
        append_scale = self.scales.append
        body = self.body
        total = len(values)
        start = 0
        while start < total:
            stop = min(start + INT8_SCALE_BLOCK, total)
            block = values[start:stop]
            top = max(block)
            bottom = min(block)
            absmax = top if top > -bottom else -bottom
            if absmax > 0.0:
                scale = _f32(absmax / 127.0)
            else:
                scale = _f32(1e-8 / 127.0)
            append_scale(scale)
            soft = scale if scale else 1.0
            for value in block:
                q = int(round(_f32(value / soft)))
                if q < INT8_MIN:
                    q = INT8_MIN
                elif q > INT8_MAX:
                    q = INT8_MAX
                body.write(bytes((q & 0xFF,)))
                if self.collect_stats:
                    self.stats.add(value, q * scale)
            start = stop
        self.n_values += total

    def n_scale(self) -> int:
        if self.kind == "q4":
            return (self.n_values + Q4_SCALE_BLOCK - 1) // Q4_SCALE_BLOCK
        return (self.n_values + INT8_SCALE_BLOCK - 1) // INT8_SCALE_BLOCK

    def write_to(self, sink: "_Sink", *, magic: bytes, flags: int | None = None) -> int:
        """Записать сегмент в приёмник и вернуть его длину в байтах.

        `flags=None` — взять из `scale_dtype` (Q4); для INT8 вызывающий
        передаёт 0x01 явно, как раньше. У Q4_1 порядок байт — как в
        `vtm_ref.encode_q4_segment`: заголовок 22 байта (magic, version,
        flags, n_values, n_groups, n_mins), масштабы, min'ы, нибблы.
        """
        if self.n_values == 0:
            raise GgufError("пустой сегмент Q4/INT8")
        if self.kind == "q4" and flags is None:
            flags = self.flags
        if flags is None:
            raise GgufError("сегмент: flags не заданы")
        n_scale = self.n_scale()
        if len(self.scales) != n_scale:
            raise GgufError(
                f"сегмент: масштабов {len(self.scales)}, ожидалось {n_scale} "
                f"на {self.n_values} значений"
            )
        header = _SEGMENT_HEADER.pack(
            magic, Q4_VERSION, flags, self.n_values, n_scale
        )
        if self.kind == "q4" and self.asymmetric:
            if len(self.mins) != n_scale:
                raise GgufError(
                    f"сегмент Q4_1: min {len(self.mins)}, ожидалось {n_scale} "
                    f"на {self.n_values} значений"
                )
            # Явное число min: читатель (и C-загрузчик) обязан сверить его с
            # ceil(n/32), иначе отсутствие min можно спутать с padding'ом.
            header += struct.pack("<I", n_scale)
        start = sink.offset
        sink.write(header)
        for chunk in self._aux_bytes(self.scales):
            sink.write(chunk)
        if self.kind == "q4" and self.asymmetric:
            for chunk in self._aux_bytes(self.mins):
                sink.write(chunk)
        self.body.seek(0)
        while True:
            chunk = self.body.read(1 << 20)
            if not chunk:
                break
            sink.write(chunk)
        return sink.offset - start

    def _aux_bytes(self, values: array) -> Iterator[bytes]:
        """Байты массива масштабов/min: fp32 — как есть, fp16 — через struct.

        Значения уже приведены к целевому типу при квантовании блока, поэтому
        здесь только сериализация. fp16 пишется кусками: у 7B на один тензор
        приходится до 2,1 млн масштабов, и `struct.pack` на них целиком — это
        лишние сотни мегабайт пика.
        """
        if self.scale_dtype == "float32":
            payload = array("f", values)
            if sys.byteorder != "little":
                payload.byteswap()
            yield payload.tobytes()
            return
        chunk = 1 << 16
        for start in range(0, len(values), chunk):
            part = values[start:start + chunk]
            yield struct.pack(f"<{len(part)}e", *part)

    def close(self) -> None:
        self.body.close()


# ---------------------------------------------------------------------------
# Имена тензоров GGUF -> Qwen/Llama
# ---------------------------------------------------------------------------

#: Глобальные тензоры Qwen/Llama.
GLOBAL_TENSORS: dict[str, tuple[str, str, str]] = {
    "token_embd.weight": ("model.embed_tokens.weight", "embed", "embed"),
    "output_norm.weight": ("model.norm.weight", "final_norm", "final_norm"),
    "output.weight": ("lm_head.weight", "lm_head", "lm_head"),
}

#: Суффиксы внутри `blk.N.` -> (HF-суффикс, роль, ключ компилятора).
LAYER_TENSORS: dict[str, tuple[str, str, str]] = {
    "attn_norm.weight": ("input_layernorm.weight", "norm", "norm1"),
    "ffn_norm.weight": ("post_attention_layernorm.weight", "norm", "norm2"),
    "attn_q.weight": ("self_attn.q_proj.weight", "matrix", "q_proj"),
    "attn_k.weight": ("self_attn.k_proj.weight", "matrix", "k_proj"),
    "attn_v.weight": ("self_attn.v_proj.weight", "matrix", "v_proj"),
    "attn_output.weight": ("self_attn.o_proj.weight", "matrix", "o_proj"),
    "ffn_gate.weight": ("mlp.gate_proj.weight", "matrix", "gate_proj"),
    "ffn_up.weight": ("mlp.up_proj.weight", "matrix", "up_proj"),
    "ffn_down.weight": ("mlp.down_proj.weight", "matrix", "down_proj"),
    "attn_q.bias": ("self_attn.q_proj.bias", "bias", "q_bias"),
    "attn_k.bias": ("self_attn.k_proj.bias", "bias", "k_bias"),
    "attn_v.bias": ("self_attn.v_proj.bias", "bias", "v_bias"),
    "attn_output.bias": ("self_attn.o_proj.bias", "bias", "o_bias"),
    "ffn_gate.bias": ("mlp.gate_proj.bias", "bias", "gate_bias"),
    "ffn_up.bias": ("mlp.up_proj.bias", "bias", "up_bias"),
    "ffn_down.bias": ("mlp.down_proj.bias", "bias", "down_bias"),
}

_MATRIX_KEYS = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")


def _split_layer_name(name: str) -> tuple[int | None, str]:
    if not name.startswith("blk."):
        return None, name
    rest = name[4:]
    head, _, tail = rest.partition(".")
    if not head.isdigit() or not tail:
        return None, name
    return int(head), tail


def map_tensor_name(name: str) -> tuple[str, str, int | None, str]:
    """GGUF-имя -> (HF-имя, роль, слой, ключ). Роли — как в qat/export.py.

    `KeyError`/`GgufError` с текстом, если имя не из Qwen2/Llama-семейства.
    """
    global_entry = GLOBAL_TENSORS.get(name)
    if global_entry is not None:
        hf_name, role, key = global_entry
        return hf_name, role, None, key
    layer, tail = _split_layer_name(name)
    if layer is not None:
        entry = LAYER_TENSORS.get(tail)
        if entry is not None:
            hf_tail, role, key = entry
            return f"model.layers.{layer}.{hf_tail}", role, layer, key
    raise GgufError(
        f"тензор {name!r}: имя не из Qwen2/Llama (поддержаны token_embd, "
        "output_norm, output и blk.N.{attn_*, ffn_*}); для Qwen3 и других "
        "архитектур с дополнительными нормами импорт не реализован"
    )


# ---------------------------------------------------------------------------
# Конфиг модели
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    architecture: str
    name: str
    hidden: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    intermediate: int
    vocab: int
    n_layers: int
    context_len: int
    max_context: int
    rope_theta: float
    rms_norm_eps: float
    tie_word_embeddings: bool
    n_layers_source: int = 0
    vocab_source: int = 0

    @property
    def q_dim(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    def as_dict(self) -> dict[str, Any]:
        return {
            "hidden": self.hidden,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "intermediate": self.intermediate,
            "vocab": self.vocab,
            "n_layers": self.n_layers,
            "context_len": self.context_len,
            "max_context": self.max_context,
            "rope_theta": self.rope_theta,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_layout": "half",
            "tie_word_embeddings": self.tie_word_embeddings,
        }


def _uniform_int(value: Any, what: str) -> int:
    if isinstance(value, (list, tuple)):
        if not value:
            raise GgufError(f"метаданные {what}: пустой массив")
        first = int(value[0])
        if any(int(item) != first for item in value):
            raise GgufError(
                f"метаданные {what}: разное значение по слоям "
                f"({sorted(set(int(item) for item in value))[:4]}…) — "
                "поэлементные head_count не поддержаны"
            )
        return first
    if isinstance(value, bool) or value is None:
        raise GgufError(f"метаданные {what}: ожидалось число, получено {value!r}")
    return int(value)


def _meta_int(meta: dict[str, Any], key: str, default: Any = None) -> int:
    if key not in meta or meta[key] is None:
        if default is None:
            raise GgufError(f"в метаданных нет {key!r} — это не Qwen-модель?")
        return int(default)
    return _uniform_int(meta[key], key)


def _meta_float(meta: dict[str, Any], key: str, default: float) -> float:
    value = meta.get(key)
    if value is None:
        return float(default)
    if isinstance(value, (list, tuple)):
        return float(_uniform_int(value, key))
    return float(value)


def infer_config(
    gguf: GgufFile,
    *,
    tie: str = "auto",
    limit_layers: int | None = None,
    limit_vocab: int | None = None,
) -> ModelConfig:
    """Собрать конфиг Qwen из метаданных GGUF и проверить его целостность."""
    arch = gguf.metadata.get("general.architecture")
    if not isinstance(arch, str) or not arch:
        raise GgufError("в метаданных нет general.architecture")
    name = str(gguf.metadata.get("general.name") or gguf.path.name)
    prefix = f"{arch}."

    rope_scaling = gguf.metadata.get(f"{prefix}rope.scaling.type")
    if rope_scaling not in (None, "", "none"):
        raise GgufError(
            f"rope.scaling.type={rope_scaling!r}: масштабирование RoPE "
            "(linear/yarn) в VTM не реализовано — импорт остановлен, "
            "чтобы не писать модель с неверной позиционной кодировкой"
        )

    n_layers_source = _meta_int(gguf.metadata, prefix + "block_count")
    hidden = _meta_int(gguf.metadata, prefix + "embedding_length")
    n_heads = _meta_int(gguf.metadata, prefix + "attention.head_count")
    n_kv_heads = _meta_int(
        gguf.metadata, prefix + "attention.head_count_kv", default=n_heads
    )
    intermediate = _meta_int(gguf.metadata, prefix + "feed_forward_length")
    context_len = _meta_int(gguf.metadata, prefix + "context_length", default=2048)
    rope_theta = _meta_float(gguf.metadata, prefix + "rope.freq_base", 10000.0)
    eps = _meta_float(
        gguf.metadata, prefix + "attention.layer_norm_rms_epsilon", 1e-6
    )
    head_dim = _meta_int(
        gguf.metadata,
        prefix + "attention.key_length",
        default=hidden // n_heads if n_heads else 0,
    )

    for label, value in (
        ("block_count", n_layers_source), ("embedding_length", hidden),
        ("attention.head_count", n_heads),
        ("attention.head_count_kv", n_kv_heads),
        ("feed_forward_length", intermediate), ("head_dim", head_dim),
        ("context_length", context_len),
    ):
        if value <= 0:
            raise GgufError(f"конфиг: {label}={value} — некорректно")
    if hidden % head_dim:
        raise GgufError(
            f"конфиг: hidden={hidden} не делится на head_dim={head_dim}"
        )
    if n_heads % n_kv_heads:
        raise GgufError(
            f"конфиг: n_heads={n_heads} не делятся на n_kv_heads={n_kv_heads}"
        )
    if hidden != n_heads * head_dim:
        raise GgufError(
            f"конфиг: hidden={hidden} != n_heads×head_dim="
            f"{n_heads * head_dim}; VTM ждёт Qwen/Llama-раскладку"
        )
    if not math.isfinite(rope_theta) or rope_theta <= 0:
        raise GgufError(f"конфиг: rope_theta={rope_theta!r}")
    if not math.isfinite(eps) or eps < 0:
        raise GgufError(f"конфиг: rms_norm_eps={eps!r}")

    # vocab: метаданные -> dims token_embd -> длина массива токенов.
    vocab = None
    if gguf.metadata.get(prefix + "vocab_size") is not None:
        vocab = _meta_int(gguf.metadata, prefix + "vocab_size")
    if gguf.has_tensor("token_embd.weight"):
        embed_dims = gguf.tensor("token_embd.weight").dims
        if len(embed_dims) != 2:
            raise GgufError(
                f"token_embd.weight: {len(embed_dims)} измерений, ожидалось 2"
            )
        if int(embed_dims[0]) != hidden:
            raise GgufError(
                f"token_embd.weight: ne0={embed_dims[0]}, а hidden={hidden}"
            )
        if vocab is None:
            vocab = int(embed_dims[1])
        elif vocab != int(embed_dims[1]):
            raise GgufError(
                f"vocab_size={vocab} не совпал с token_embd ne1={embed_dims[1]}"
            )
    if vocab is None:
        vocab = gguf.array_lengths.get("tokenizer.ggml.tokens")
    if not vocab or vocab <= 0:
        raise GgufError(
            "не удалось определить vocab: нет vocab_size, "
            "token_embd.weight и tokenizer.ggml.tokens"
        )

    if tie == "auto":
        tie_flag = not gguf.has_tensor("output.weight")
    elif tie in ("yes", "no"):
        tie_flag = tie == "yes"
    else:
        raise GgufError(f"tie={tie!r}, допустимо auto|yes|no")
    if tie_flag and not gguf.has_tensor("token_embd.weight"):
        raise GgufError("tie_word_embeddings=true, но token_embd.weight нет")
    if not tie_flag and not gguf.has_tensor("output.weight"):
        raise GgufError(
            "output.weight нет, а --tie no: нечем считать lm_head"
        )

    if limit_layers is not None:
        if limit_layers <= 0:
            raise GgufError(f"--limit-layers={limit_layers}, нужно > 0")
        if limit_layers > n_layers_source:
            raise GgufError(
                f"--limit-layers={limit_layers} > слоёв в модели "
                f"({n_layers_source})"
            )
        n_layers = limit_layers
    else:
        n_layers = n_layers_source
    if limit_vocab is not None:
        if limit_vocab <= 0:
            raise GgufError(f"--limit-vocab={limit_vocab}, нужно > 0")
        if limit_vocab > vocab:
            raise GgufError(
                f"--limit-vocab={limit_vocab} > vocab={vocab}"
            )
        out_vocab = limit_vocab
    else:
        out_vocab = vocab

    return ModelConfig(
        architecture=arch,
        name=name,
        hidden=hidden,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        intermediate=intermediate,
        vocab=out_vocab,
        n_layers=n_layers,
        context_len=context_len,
        max_context=context_len,
        rope_theta=rope_theta,
        rms_norm_eps=eps,
        tie_word_embeddings=tie_flag,
        n_layers_source=n_layers_source,
        vocab_source=vocab,
    )


# ---------------------------------------------------------------------------
# План импорта
# ---------------------------------------------------------------------------


@dataclass
class PlannedTensor:
    info: TensorInfo
    gguf_name: str
    hf_name: str
    role: str
    layer: int | None
    key: str
    shape: tuple[int, ...]
    n_values: int
    target: str  # q4 | int8 | f32 | f16


_ROLE_ORDER = {
    "embed": 0, "matrix": 1, "norm": 2, "bias": 3,
    "final_norm": 4, "lm_head": 5,
}


def _role_order(plan: PlannedTensor) -> tuple[int, int, str]:
    return (
        _ROLE_ORDER.get(plan.role, 9),
        -1 if plan.layer is None else plan.layer,
        plan.key,
    )


def plan_import(
    gguf: GgufFile,
    cfg: ModelConfig,
    *,
    fmt: str,
    embed_mode: str,
) -> list[PlannedTensor]:
    """Проверить имена/формы всех тензоров и решить, что куда писать.

    `embed_mode` — уже нормализованный (`_normalize_embed_mode`):
    int8 | trit | float32 | float16. `trit` кладёт таблицу словаря
    TRT1-сегментом в `ternary.trit` (kind 0), как это делает QAT-экспорт с
    `quantize_embeddings: trit`; так 7B Q4 с fp16-масштабами влезает в 4 ГиБ
    (ADR 0016): int8-таблица стоит 0,53 ГиБ против 0,12 ГиБ у тритов.
    """
    if fmt not in ("q4", "trit"):
        raise GgufError(f"format={fmt!r}, допустимо q4|trit")
    if embed_mode not in EMBED_MODES:
        raise GgufError(
            f"embed={embed_mode!r}, допустимо int8|trit|fp16|fp32"
        )
    if cfg.tie_word_embeddings and embed_mode == "float16":
        raise GgufError(
            "tie_word_embeddings=true, а --embed fp16: компилятор требует "
            "для tied-таблицы fp32, int8 или триты (fp16 он отвергает) — "
            "выберите --embed int8, trit или fp32"
        )

    # Типы и имена проверяются до записи: частично записанный экспорт хуже,
    # чем отказ с полным списком проблем.
    unsupported: dict[int, list[str]] = {}
    for info in gguf.tensors:
        if info.type_id not in GGUF_TYPES:
            unsupported.setdefault(info.type_id, []).append(info.name)
            continue
        try:
            info.n_bytes  # проверка кратности блоку
        except GgufError as exc:
            raise GgufError(str(exc)) from None
    if unsupported:
        details = []
        for type_id, names in sorted(unsupported.items()):
            preview = ", ".join(names[:3])
            if len(names) > 3:
                preview += f" и ещё {len(names) - 3}"
            details.append(f"{type_label(type_id)}: {len(names)} тензоров ({preview})")
        raise GgufError(
            "неподдерживаемые типы квантования: " + "; ".join(details)
            + f". Поддержаны: {SUPPORTED_TYPE_NAMES}. "
            "Конвертируйте модель в Q4_0/Q4_K/Q8_0/F16 (например, "
            "`ollama pull` другой тег или `llama-quantize`), либо добавьте "
            "деквант типа в tools/gguf_import.py"
        )

    plans: list[PlannedTensor] = []
    seen_roles: set[tuple[str, int | None, str]] = set()
    layer_keys: dict[int, set[str]] = {index: set() for index in range(cfg.n_layers)}
    unknown: list[str] = []

    for info in gguf.tensors:
        try:
            hf_name, role, layer, key = map_tensor_name(info.name)
        except GgufError:
            unknown.append(info.name)
            continue
        if layer is not None and layer >= cfg.n_layers:
            continue  # срез --limit-layers: остальные слои не нужны
        if role == "lm_head" and cfg.tie_word_embeddings:
            continue  # tied: таблица уже есть как embed
        shape = info.hf_shape
        if role in ("embed", "lm_head"):
            expected = (cfg.vocab, cfg.hidden)
            n_values = cfg.vocab * cfg.hidden
            if shape[0] < expected[0] or shape[1] != expected[1]:
                raise GgufError(
                    f"{info.name}: форма {shape}, а нужно не меньше "
                    f"{expected} (срез словаря берётся с начала строк)"
                )
        else:
            expected = _expected_shape(cfg, role, layer, key, info.name)
            if shape != expected:
                raise GgufError(
                    f"{info.name}: форма {shape}, ожидалась {expected}"
                )
            n_values = info.n_elements
        if (
            fmt == "q4"
            and role in ("matrix", "lm_head")
            and n_values % Q4_SCALE_BLOCK
        ):
            raise GgufError(
                f"{info.name}: {n_values} значений не кратны группе Q4 "
                f"{Q4_SCALE_BLOCK} — такой тензор нельзя записать как kind 5"
            )
        if (
            (fmt == "trit" or embed_mode == "trit")
            and role in ("matrix", "lm_head", "embed")
            and n_values % TRIT_SCALE_BLOCK
        ):
            raise GgufError(
                f"{info.name}: {n_values} значений не кратны блоку тритов "
                f"{TRIT_SCALE_BLOCK} — такой тензор нельзя записать в TRT1"
            )
        if embed_mode == "int8" and role == "embed":
            target = "int8"
        elif embed_mode == "trit" and role == "embed":
            target = "trit"
        elif role in ("norm", "final_norm", "bias"):
            target = "f32"
        elif embed_mode in ("float32", "float16") and role == "embed":
            target = "f32" if embed_mode == "float32" else "f16"
        elif role in ("matrix", "lm_head"):
            target = "q4" if fmt == "q4" else "trit"
        else:
            target = "f32"
        token = (role, layer, key)
        if token in seen_roles:
            raise GgufError(f"дубликат тензора по роли {role}/{layer}/{key}")
        seen_roles.add(token)
        if layer is not None:
            layer_keys[layer].add(key)
        plans.append(
            PlannedTensor(
                info=info,
                gguf_name=info.name,
                hf_name=hf_name,
                role=role,
                layer=layer,
                key=key,
                shape=expected,
                n_values=n_values,
                target=target,
            )
        )

    if unknown:
        preview = ", ".join(sorted(unknown)[:8])
        if len(unknown) > 8:
            preview += f" … (всего {len(unknown)})"
        raise GgufError(
            "неизвестные тензоры в GGUF: " + preview
            + ". Qwen3/другие архитектуры с доп. нормами не поддержаны; "
            "уберите их или добавьте отображение в tools/gguf_import.py"
        )

    # Полнота графа: без embed/final_norm/lm_head и матриц слоя компилятор
    # соберёт не то или откажет позже — лучше сказать здесь.
    roles = {(plan.role, plan.layer, plan.key) for plan in plans}
    if ("embed", None, "embed") not in roles:
        raise GgufError("в GGUF нет token_embd.weight: полный граф не собрать")
    if ("final_norm", None, "final_norm") not in roles:
        raise GgufError("в GGUF нет output_norm.weight: полный граф не собрать")
    if not cfg.tie_word_embeddings and ("lm_head", None, "lm_head") not in roles:
        raise GgufError("в GGUF нет output.weight, а tied-эмбеддинги выключены")
    required = {"q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj", "norm1", "norm2"}
    for index in range(cfg.n_layers):
        missing = sorted(required - layer_keys.get(index, set()))
        if missing:
            raise GgufError(
                f"слой {index}: нет тензоров {', '.join(missing)} — "
                "манифест был бы неполным"
            )
    plans.sort(key=_role_order)
    return plans


def _expected_shape(
    cfg: ModelConfig, role: str, layer: int | None, key: str, name: str
) -> tuple[int, ...]:
    if role == "final_norm" or key == "final_norm":
        return (cfg.hidden,)
    if role == "embed" or key == "embed":
        return (cfg.vocab, cfg.hidden)
    if role == "lm_head" or key == "lm_head":
        return (cfg.vocab, cfg.hidden)
    if key.endswith("_bias"):
        dims = {
            "q_bias": cfg.q_dim, "k_bias": cfg.kv_dim, "v_bias": cfg.kv_dim,
            "o_bias": cfg.hidden, "gate_bias": cfg.intermediate,
            "up_bias": cfg.intermediate, "down_bias": cfg.hidden,
        }
        return (dims[key],)
    if key in ("norm1", "norm2"):
        return (cfg.hidden,)
    dims = {
        "q_proj": (cfg.q_dim, cfg.hidden),
        "k_proj": (cfg.kv_dim, cfg.hidden),
        "v_proj": (cfg.kv_dim, cfg.hidden),
        "o_proj": (cfg.hidden, cfg.q_dim),
        "gate_proj": (cfg.intermediate, cfg.hidden),
        "up_proj": (cfg.intermediate, cfg.hidden),
        "down_proj": (cfg.hidden, cfg.intermediate),
    }
    if key not in dims:  # pragma: no cover - план проверяет роли
        raise GgufError(f"{name}: неизвестная роль {role}/{key}")
    return dims[key]


# ---------------------------------------------------------------------------
# Запись файлов и манифеста
# ---------------------------------------------------------------------------


class _Sink:
    """Файл назначения с sha256 и счётчиком байт."""

    def __init__(self, path: Path):
        self.path = path
        self.handle = open(path, "wb")
        self.sha256 = hashlib.sha256()
        self.offset = 0

    def write(self, data: bytes) -> None:
        self.handle.write(data)
        self.sha256.update(data)
        self.offset += len(data)

    def align(self, alignment: int = 32) -> None:
        pad = (-self.offset) % alignment
        if pad:
            self.write(bytes(pad))

    def close(self) -> None:
        self.handle.close()

    def abort(self) -> None:
        try:
            self.handle.close()
        finally:
            try:
                self.path.unlink()
            except OSError:
                pass


@dataclass
class ImportResult:
    config: ModelConfig
    manifest: dict[str, Any]
    manifest_path: Path
    q4_path: Path | None
    non_ternary_path: Path
    source_bytes: int
    q4_bytes: int
    non_ternary_bytes: int
    stats: dict[str, _Stats] = field(default_factory=dict)
    q4_params: int = 0
    trit_params: int = 0

    @property
    def compiler_ready(self) -> bool:
        return bool((self.manifest.get("compiler") or {}).get("ready"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_plain_tensor(
    sink: _Sink, plan: PlannedTensor, values: array, stats: _Stats | None
) -> int:
    start = sink.offset
    if plan.target == "f32":
        payload = array("f", values)
        if sys.byteorder != "little":
            payload.byteswap()
        sink.write(payload.tobytes())
    elif plan.target == "f16":
        chunk = 1 << 16
        for offset in range(0, len(values), chunk):
            part = values[offset: offset + chunk]
            sink.write(struct.pack(f"<{len(part)}e", *part))
    else:  # pragma: no cover - цель проверена планом
        raise GgufError(f"{plan.hf_name}: не fp-цель {plan.target}")
    if stats is not None:
        for value in values:
            stats.add(value, value)
    return sink.offset - start


def import_gguf(
    source: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    *,
    fmt: str = "q4",
    embed_mode: str = "int8",
    q4_scale_dtype: str = "float32",
    q4_kind: str = "auto",
    tie: str = "auto",
    limit_layers: int | None = None,
    limit_vocab: int | None = None,
    stats: bool = False,
    force: bool = False,
    progress: bool = False,
) -> ImportResult:
    """Прочитать GGUF и записать манифест + веса VTM в `out_dir`.

    `fmt="q4"` — Linear в kind 5 (`q4.bin`); `fmt="trit"` — Linear в триты
    (`ternary.trit`, kind 0). Компилятор (0154) читает оба.

    `q4_scale_dtype` — fp32 или fp16 для масштабов Q4-сегмента: цена fp32 —
    1 бит на вес (у 7B это +0,44 ГиБ, то есть выход за 4 ГиБ, ADR 0016).

    `q4_kind` — `auto` (по умолчанию) | `q4_1` | `q4_0`. Q4_1 пишет `min`
    на группу 32 и почти не теряет точность на Q4_K-подблоках; цена — 2–4
    байта на группу 32. `auto` выбирает Q4_1, если оценка контейнера
    влезает в предел 4 ГиБ (ADR 0016), иначе Q4_0 — и сообщает выбор с
    числами в stderr: у 7B Q4_1 с fp16-масштабами и тритовым эмбеддингом
    даёт ~4,55 ГБ (106 % предела), у 3B — ~2,4 ГБ.

    `embed_mode` — int8 | trit | float32 | float16. `trit` кладёт таблицу
    словаря TRT1-сегментом в `ternary.trit` (kind 0) и уменьшает её с 0,53
    до ~0,12 ГиБ у 7B — это и есть разница между 105,7 % и 95 % предела.
    """
    started = time.time()
    fmt = str(fmt).lower()
    if fmt not in ("q4", "trit"):
        raise GgufError(f"format={fmt!r}, допустимо q4|trit")
    embed_mode = _normalize_embed_mode(embed_mode)
    q4_scale_dtype = _normalize_scale_dtype(q4_scale_dtype)
    q4_kind = _normalize_q4_kind(q4_kind)
    gguf = read_gguf(source)
    cfg = infer_config(
        gguf, tie=tie, limit_layers=limit_layers, limit_vocab=limit_vocab
    )
    plans = plan_import(gguf, cfg, fmt=fmt, embed_mode=embed_mode)

    # Вид Q4 зависит от бюджета: 3B Q4_1 влезает с запасом, 7B — нет.
    auto_kind = q4_kind == "auto"
    if fmt == "q4":
        asym_bytes = estimate_container_bytes(
            plans, q4_asym=True, scale_dtype=q4_scale_dtype
        )
        if q4_kind == "auto":
            if asym_bytes + CONTAINER_MARGIN <= CONTAINER_LIMIT:
                q4_kind = "q4_1"
                reason = (
                    f"оценка контейнера {asym_bytes / 2**30:.2f} ГиБ "
                    f"≤ {CONTAINER_LIMIT / 2**30:.0f} ГиБ — берём Q4_1 "
                    "(min на группу 32)"
                )
            else:
                sym_bytes = estimate_container_bytes(
                    plans, q4_asym=False, scale_dtype=q4_scale_dtype
                )
                q4_kind = "q4_0"
                reason = (
                    f"оценка Q4_1 {asym_bytes / 2**30:.2f} ГиБ > "
                    f"{CONTAINER_LIMIT / 2**30:.0f} ГиБ (ADR 0016), Q4_0 — "
                    f"{sym_bytes / 2**30:.2f} ГиБ — берём Q4_0"
                )
            print(f"gguf_import: q4_kind=auto → {q4_kind}: {reason}",
                  file=sys.stderr)
        chosen_bytes = (
            asym_bytes if q4_kind == "q4_1"
            else estimate_container_bytes(
                plans, q4_asym=False, scale_dtype=q4_scale_dtype
            )
        )
        if chosen_bytes + CONTAINER_MARGIN > CONTAINER_LIMIT:
            # Даже выбранный вид не влезает: подсказка, что менять, а не
            # «сборка упала на 4 ГиБ» через час.
            hint = (
                "уменьшите вес: --q4-scale-dtype fp16 (−0,44 ГБ у 7B)"
                if q4_scale_dtype == "float32" else "уменьшите вес"
            )
            if embed_mode == "int8":
                hint += ", --embed trit (−0,44 ГБ у 7B)"
            explicit = "задан явно" if not auto_kind else "выбран auto"
            print(
                f"gguf_import: q4_kind={q4_kind} ({explicit}), но оценка "
                f"контейнера {chosen_bytes / 2**30:.2f} ГиБ > "
                f"{CONTAINER_LIMIT / 2**30:.0f} ГиБ (ADR 0016, u32-смещения): "
                f"{hint}",
                file=sys.stderr,
            )
    q4_asym = q4_kind == "q4_1"

    out = Path(out_dir).expanduser()
    if out.exists():
        if not out.is_dir():
            raise GgufError(f"{out} существует и это не каталог")
        entries = list(out.iterdir())
        if entries and not force:
            names = ", ".join(sorted(entry.name for entry in entries[:5]))
            raise GgufError(
                f"{out} не пуст ({names}{'…' if len(entries) > 5 else ''}); "
                "укажите --force, чтобы перезаписать импорт"
            )
    else:
        out.mkdir(parents=True)

    q4_plans = [plan for plan in plans if plan.target == "q4"]
    trit_plans = [plan for plan in plans if plan.target == "trit"]
    q4_path = out / Q4_FILE if q4_plans else None
    trit_path = out / "ternary.trit"
    non_path = out / NON_TERNARY_FILE
    manifest_path = out / MANIFEST_FILE

    q4_sink = _Sink(q4_path) if q4_path is not None else None
    trit_sink = _Sink(trit_path) if trit_plans else None
    non_sink = _Sink(non_path)
    segment_entries: list[dict[str, Any]] = []
    non_entries: list[dict[str, Any]] = []
    stats_by_role: dict[str, _Stats] = {}
    total_params = 0
    q4_params = 0
    trit_params = 0
    int8_params = 0
    fp_params = 0
    source_values_q4 = 0
    source_values_trit = 0

    def _role_stats(role: str) -> _Stats | None:
        if not stats:
            return None
        return stats_by_role.setdefault(role, _Stats())

    def _entry(plan: PlannedTensor, offset: int, length: int, kind: str) -> dict:
        return {
            "name": plan.hf_name,
            "role": plan.role,
            "layer": plan.layer,
            "key": plan.key,
            "shape": list(plan.shape),
            "kind": kind,
            "dtype": q4_kind if kind == "q4" else "trit",
            "offset": offset,
            "length": length,
            ("q4_offset" if kind == "q4" else "trit_offset"): offset,
            ("q4_length" if kind == "q4" else "trit_length"): length,
            "params": plan.n_values,
            "source_tensor": plan.gguf_name,
            "source_type": plan.info.spec.name,
        }

    try:
        for plan in plans:
            total_params += plan.n_values
            if plan.target == "q4":
                assert q4_sink is not None
                encoder = _SegmentEncoder(
                    "q4", collect_stats=stats, scale_dtype=q4_scale_dtype,
                    asymmetric=q4_asym,
                )
                for chunk in gguf.iter_tensor_chunks(plan.info, plan.n_values):
                    encoder.add_values(chunk)
                q4_sink.align(32)
                length = encoder.write_to(q4_sink, magic=encoder.magic)
                encoder.close()
                q4_params += plan.n_values
                source_values_q4 += plan.info.n_bytes
                segment_entries.append(
                    _entry(plan, q4_sink.offset - length, length, "q4")
                )
                if stats:
                    stats_by_role.setdefault("q4", _Stats()).merge(encoder.stats)
                if progress:
                    print(
                        f"  q4   {plan.hf_name:<47} "
                        f"{plan.n_values / 1e6:7.1f} МЭ → {length / 1e6:7.1f} МБ",
                        file=sys.stderr,
                    )
            elif plan.target == "trit":
                assert trit_sink is not None
                encoder = _TritEncoder(
                    collect_stats=stats,
                    # Таблица словаря — единственный тензор, где absmax-правило
                    # теряет слишком много: см. docstring _TritEncoder.
                    scale_rule="mean" if plan.role == "embed" else "absmax",
                )
                for chunk in gguf.iter_tensor_chunks(plan.info, plan.n_values):
                    encoder.add_values(chunk)
                trit_sink.align(32)
                length = encoder.write_to(trit_sink)
                encoder.close()
                trit_params += plan.n_values
                source_values_trit += plan.info.n_bytes
                segment_entries.append(
                    _entry(plan, trit_sink.offset - length, length, "trit")
                )
                if stats:
                    # Ошибка тритовой таблицы словаря — отдельная ступень:
                    # её надо мерить против int8, а не смешивать с матрицами.
                    key = "embed(trit)" if plan.role == "embed" else "trit"
                    stats_by_role.setdefault(key, _Stats()).merge(
                        encoder.stats
                    )
                if progress:
                    print(
                        f"  trit {plan.hf_name:<47} "
                        f"{plan.n_values / 1e6:7.1f} МЭ → {length / 1e6:7.1f} МБ",
                        file=sys.stderr,
                    )
            elif plan.target == "int8":
                encoder = _SegmentEncoder("int8", collect_stats=stats)
                for chunk in gguf.iter_tensor_chunks(plan.info, plan.n_values):
                    encoder.add_values(chunk)
                non_sink.align(32)
                length = encoder.write_to(non_sink, magic=INT8_MAGIC, flags=0x01)
                encoder.close()
                int8_params += plan.n_values
                non_entries.append({
                    "name": plan.hf_name,
                    "role": plan.role,
                    "layer": plan.layer,
                    "key": plan.key,
                    "shape": list(plan.shape),
                    "dtype": "int8",
                    "offset": non_sink.offset - length,
                    "length": length,
                    "params": plan.n_values,
                    "source_tensor": plan.gguf_name,
                    "source_type": plan.info.spec.name,
                })
                if stats:
                    stats_by_role.setdefault("int8", _Stats()).merge(encoder.stats)
                if progress:
                    print(
                        f"  int8 {plan.hf_name:<47} "
                        f"{plan.n_values / 1e6:7.1f} МЭ → {length / 1e6:7.1f} МБ",
                        file=sys.stderr,
                    )
            else:
                role_stats = _role_stats(plan.role)
                non_sink.align(32)
                start = non_sink.offset
                for chunk in gguf.iter_tensor_chunks(plan.info, plan.n_values):
                    _write_plain_tensor(non_sink, plan, chunk, role_stats)
                length = non_sink.offset - start
                fp_params += plan.n_values
                non_entries.append({
                    "name": plan.hf_name,
                    "role": plan.role,
                    "layer": plan.layer,
                    "key": plan.key,
                    "shape": list(plan.shape),
                    "dtype": "float32" if plan.target == "f32" else "float16",
                    "offset": start,
                    "length": length,
                    "params": plan.n_values,
                    "source_tensor": plan.gguf_name,
                    "source_type": plan.info.spec.name,
                })

        # tied lm_head: отдельного тензора нет, компилятор читает ту же таблицу.
        if cfg.tie_word_embeddings:
            # Таблица может лежать и в non_ternary.bin (int8/fp), и в
            # ternary.trit (триты): ищем её в обоих списках, а ссылку кладём в
            # non_ternary — компилятор пропускает записи с tied_to, не читая
            # по ним байт.
            embed_entry = next(
                (entry for entry in non_entries if entry["role"] == "embed"),
                None,
            )
            if embed_entry is None:
                embed_entry = next(
                    (entry for entry in segment_entries
                     if entry["role"] == "embed"),
                    None,
                )
            assert embed_entry is not None  # план гарантирует embed
            non_entries.append({
                "name": "lm_head.weight",
                "role": "lm_head",
                "layer": None,
                "key": "lm_head",
                "shape": list(embed_entry["shape"]),
                "dtype": embed_entry["dtype"],
                "tied_to": embed_entry["name"],
                "params": embed_entry["params"],
            })
    except BaseException:
        if q4_sink is not None:
            q4_sink.abort()
        if trit_sink is not None:
            trit_sink.abort()
        non_sink.abort()
        raise

    if q4_sink is not None:
        q4_sink.close()
    if trit_sink is not None:
        trit_sink.close()
    non_sink.close()

    # Структурная проверка записанного: заголовки сегментов и границы.
    if q4_path is not None:
        _verify_q4_file(q4_path, segment_entries, q4_scale_dtype, q4_asym)
    if trit_plans:
        _verify_trit_file(trit_path, segment_entries)
    _verify_int8_file(non_path, non_entries)

    q4_bytes = q4_path.stat().st_size if q4_path is not None else 0
    trit_bytes = trit_path.stat().st_size if trit_plans else 0
    non_bytes = non_path.stat().st_size
    if not trit_plans:
        # Компилятор всегда открывает ternary.trit (даже в q4-экспорте).
        trit_path.write_bytes(b"")
    files: dict[str, dict[str, Any]] = {
        "ternary.trit": {
            "bytes": trit_bytes,
            "sha256": _sha256_file(trit_path),
        },
        NON_TERNARY_FILE: {
            "bytes": non_bytes,
            "sha256": _sha256_file(non_path),
        },
    }
    if q4_path is not None:
        files[Q4_FILE] = {
            "bytes": q4_bytes,
            "sha256": _sha256_file(q4_path),
        }

    partial = (
        cfg.n_layers != cfg.n_layers_source
        or cfg.vocab != cfg.vocab_source
        or limit_layers is not None
        or limit_vocab is not None
    )
    architecture = cfg.as_dict()
    if fmt == "q4":
        architecture["lm_head"] = "q4"
    else:
        architecture["lm_head"] = "mm" if cfg.tie_word_embeddings else "tmv"
    # Компилятор понимает три значения: int8 (kind 4 в non_ternary.bin),
    # trit (TRT1 в ternary.trit) и none (плотная fp32/fp16-таблица).
    architecture["embed"] = (
        embed_mode if embed_mode in ("int8", "trit") else "none"
    )

    embed_rule = {
        "int8": (
            "int8 (kind 4): scale=absmax(128)/127; "
            "q=clamp(round(w/scale),-127,127); w=q*scale"
        ),
        "trit": (
            "триты (kind 0): scale=1.5*mean(|w|) на блок 128 (TWN); "
            "trit=clamp(round_half_away(w/scale),-1,1); w=trit*scale"
        ),
        "float32": "плотная fp32-таблица (kind 1), без квантования",
        "float16": "плотная fp16-таблица (kind 2), без квантования",
    }[embed_mode]

    quantization = {
        "block_size": 128,
        "scale_dtype": "float32",
        "target": fmt,
        "rule": (
            ("Linear -> Q4_1 (kind 5, asym): min=min(32); "
             "scale=(max-min)/15; "
             "q=clamp(round_half_away((w-min)/scale),0,15); w=q*scale+min")
            if fmt == "q4" and q4_asym else
            ("Linear -> Q4_0 (kind 5, sym): scale=absmax(32)/7; "
             "q=clamp(round_half_away(w/scale),-8,7)+8; w=(q-8)*scale")
            if fmt == "q4" else
            "Linear -> триты (kind 0): scale=absmax(128); "
            "trit=clamp(round_half_away(w/scale),-1,1); w=trit*scale"
        ),
        "q4_block_size": Q4_SCALE_BLOCK,
        "q4_kind": q4_kind if fmt == "q4" else None,
        "q4_scale_dtype": q4_scale_dtype,
        "q4_scale_bytes": Q4_SCALE_BYTES[q4_scale_dtype],
        "q4_flags": Q4_FLAG_F32_SCALES if q4_scale_dtype == "float32" else 0,
        "q4_asymmetric": bool(q4_asym) if fmt == "q4" else None,
        "q4_estimated_bytes": (
            estimate_container_bytes(
                plans, q4_asym=q4_asym, scale_dtype=q4_scale_dtype
            )
            if fmt == "q4" else None
        ),
        "container_limit_bytes": CONTAINER_LIMIT,
        "q4_layout": "ggml: байт j = значение j (low), j+16 (high)",
        "trit_block_size": TRIT_SCALE_BLOCK,
        "trit_layout": "packed20: 20 тритов в u32, 160 тритов в 32 байтах",
        "trit_scale_rule": (
            "Linear — absmax(128) (как qat/ternary.py); "
            "эмбеддинг — 1.5*mean(|w|) (TWN, меньше потерь на Q6_K-таблице)"
            if embed_mode == "trit" else
            "absmax(128) (как qat/ternary.py)"
        ),
        "embed_rule": embed_rule,
        "source_types": sorted({plan.info.spec.name for plan in plans}),
    }
    scale_label = "f32" if q4_scale_dtype == "float32" else "f16"
    q4_section = None
    if q4_path is not None:
        q4_section = {
            "file": Q4_FILE,
            "kind": 5,
            "block_size": Q4_SCALE_BLOCK,
            "scale_dtype": q4_scale_dtype,
            "flags": Q4_FLAG_F32_SCALES if q4_scale_dtype == "float32" else 0,
            "asymmetric": bool(q4_asym),
            "layout": (
                "ggml: байт j = значение j (low), j+16 (high); "
                "w = q*scale + min" if q4_asym else
                "ggml: байт j = значение j (low), j+16 (high); "
                "w = (q-8)*scale"
            ),
            "nibble_min": Q4_NIBBLE_MIN,
            "nibble_max": Q4_NIBBLE_MAX,
            "nibble_zero": None if q4_asym else Q4_ZERO_NIBBLE,
            "nibble_note": (
                "у Q4_1 min группы — это ниббл 0, max — ниббл 15; "
                "фиксированного «нуля» нет" if q4_asym else
                "у Q4_0 ниббл 8 — ноль, диапазон -8..7"
            ),
            "segment": {
                "magic": "Q4_1" if q4_asym else "Q4_0",
                "version": Q4_VERSION,
                "header_bytes": (
                    Q4_ASYM_HEADER_SIZE if q4_asym else Q4_HEADER_SIZE
                ),
                "n_mins": (
                    "n_groups (u32 после n_groups)" if q4_asym else None
                ),
                "order": (
                    f"header, scales {scale_label}, mins {scale_label}, "
                    "nibbles" if q4_asym else
                    f"header, scales {scale_label}, nibbles"
                ),
                "aligned": 32,
                "note": (
                    "ровно tools/vtm_ref.py:encode_q4_segment "
                    "(Tensor.q4(..., asymmetric=True))" if q4_asym else
                    "ровно tools/vtm_ref.py:encode_q4_segment "
                    "(Tensor.q4(..., asymmetric=False))"
                ),
            },
            "tensors": len(q4_plans),
            "params": q4_params,
        }
    trit_section = None
    if trit_plans:
        trit_section = {
            "file": "ternary.trit",
            "kind": 0,
            "magic": "TRT1",
            "version": TRT1_VERSION,
            "block_size": TRIT_SCALE_BLOCK,
            "scale_dtype": "float32",
            "storage": "packed20",
            "layout": "header, scales f32, words pack20 (20 тритов в u32)",
            "segment": {
                "magic": "TRT1",
                "version": TRT1_VERSION,
                "header_bytes": TRT1_HEADER_SIZE,
                "aligned": 32,
                "note": "tools/vtm_ref.py:encode_trit_segment (kind 0)",
            },
            "contains_embed": embed_mode == "trit",
            "tensors": len(trit_plans),
            "params": trit_params,
        }
    # Запись таблицы словаря — в том файле, куда её положил план: int8/fp
    # в non_ternary.bin, триты — в ternary.trit (tensors[]).
    embed_entry = next(
        (entry for entry in non_entries if entry["role"] == "embed"), None
    )
    embed_segment = None
    if embed_entry is None:
        embed_segment = next(
            (entry for entry in segment_entries if entry["role"] == "embed"),
            None,
        )
    embed_dtype = {
        "int8": "int8",
        "trit": "trit",
        "float32": "float32",
        "float16": "float16",
    }[embed_mode]
    embed_bytes = 0
    if embed_entry is not None:
        embed_bytes = int(embed_entry["length"])
    elif embed_segment is not None:
        embed_bytes = int(embed_segment["length"])
    embeddings = {
        "mode": embed_mode if embed_mode in ("int8", "trit") else "none",
        "tensor": "model.embed_tokens.weight",
        "storage": NON_TERNARY_FILE if embed_entry is not None else "ternary.trit",
        "dtype": embed_dtype,
        "params": (
            embed_entry["params"] if embed_entry is not None
            else (embed_segment or {}).get("params", 0)
        ),
        "bytes": embed_bytes,
        "block_size": (
            INT8_SCALE_BLOCK if embed_mode == "int8"
            else (TRIT_SCALE_BLOCK if embed_mode == "trit" else None)
        ),
        "rule": quantization["embed_rule"],
    }
    if embed_segment is not None:
        embeddings["segment"] = {
            "magic": "TRT1",
            "version": TRT1_VERSION,
            "kind": 0,
            "storage": "packed20",
            "trit_offset": embed_segment["trit_offset"],
            "trit_length": embed_segment["trit_length"],
        }
    # Компилятор 0154 читает и q4.bin (kind 5), и TRT1/int8-эмбеддинг:
    # манифест импорта собирается без правок компилятора.
    compiler_missing: list[str] = []
    manifest = {
        "format": MANIFEST_FORMAT,
        "format_version": MANIFEST_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generator": "tools/gguf_import.py",
        "source": {
            "path": str(gguf.path),
            "file_bytes": gguf.file_size,
            "gguf_version": gguf.version,
            "name": cfg.name,
            "architecture": cfg.architecture,
            "tensor_count": len(gguf.tensors),
            "data_offset": gguf.data_offset,
            "alignment": gguf.alignment,
            "metadata": _json_safe({
                key: value
                for key, value in gguf.metadata.items()
                if isinstance(value, (str, int, float, bool))
                and (
                    key.startswith("general.")
                    or key.startswith(cfg.architecture + ".")
                )
            }),
        },
        "architecture": architecture,
        "quantization": quantization,
        "q4": q4_section,
        "trit": trit_section,
        "embeddings": embeddings,
        "files": files,
        "totals": {
            "params": total_params,
            "q4_params": q4_params,
            "trit_params": trit_params,
            "int8_params": int8_params,
            "fp_params": fp_params,
            "source_bytes": gguf.file_size,
            "q4_bytes": q4_bytes,
            "trit_bytes": trit_bytes,
            "non_ternary_bytes": non_bytes,
            "params_source_bytes": source_values_q4 + source_values_trit,
        },
        "tensors": segment_entries,
        "non_ternary": {
            "stored_in_this_export": True,
            "file": NON_TERNARY_FILE,
            "dtype": "mixed" if embed_mode == "int8" else "float32",
            "count": len(non_entries),
            "params": int8_params + fp_params,
            "tensors": non_entries,
            "note": (
                "эмбеддинг — "
                + (
                    "INT8-сегмент (kind 4, блок 128)"
                    if embed_mode == "int8" else
                    "в ternary.trit, триты (kind 0)"
                    if embed_mode == "trit" else
                    f"плотный {embed_dtype}"
                )
                + "; нормы/bias — fp32; квантованные матрицы — в "
                + ("q4.bin" if fmt == "q4" else "ternary.trit")
                + " и описаны в tensors[]"
            ),
        },
        "compiler": {
            "ready": True,
            "format": fmt,
            "missing": compiler_missing,
            "note": (
                "манифест читается tools/vtm_compile.py без изменений: "
                + (
                    "q4.bin (kind 5, Q4_1/asym) в Tensor(kind=Q4, "
                    "q4_asymmetric=True), lm_head='q4' → TMV"
                    if fmt == "q4" and q4_asym else
                    "q4.bin (kind 5, Q4_0/sym) в Tensor(kind=Q4), "
                    "lm_head='q4' → TMV"
                    if fmt == "q4" else
                    "Linear в TRT1, lm_head='tmv'"
                )
                + (
                    ", эмбеддинг — TRT1-сегмент (kind 0)"
                    if embed_mode == "trit" else
                    ", эмбеддинг — INT8-сегмент (kind 4)"
                    if embed_mode == "int8" else
                    f", эмбеддинг — плотный {embed_dtype}"
                )
                + ("" if q4_scale_dtype == "float32" else
                   "; fp16-масштабы Q4 разбираются decode_q4_segment_ex")
            ),
        },
        "partial": bool(partial),
        "partial_note": (
            "ЧАСТИЧНАЯ модель (--limit-layers/--limit-vocab): для дымового "
            "теста, не для качества" if partial else None
        ),
        "import_seconds": round(time.time() - started, 3),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return ImportResult(
        config=cfg,
        manifest=manifest,
        manifest_path=manifest_path,
        q4_path=q4_path,
        non_ternary_path=non_path,
        source_bytes=gguf.file_size,
        q4_bytes=q4_bytes,
        non_ternary_bytes=non_bytes,
        stats=stats_by_role,
        q4_params=q4_params,
        trit_params=trit_params,
    )


def _verify_trit_file(path: Path, entries: Sequence[dict[str, Any]]) -> None:
    """Проверить, что TRT1-сегменты на месте и их заголовки согласованы."""
    size = path.stat().st_size
    with open(path, "rb") as handle:
        for entry in entries:
            if entry.get("kind") != "trit":
                continue
            offset = int(entry["trit_offset"])
            length = int(entry["trit_length"])
            if offset < 0 or length < TRT1_HEADER_SIZE or offset + length > size:
                raise GgufError(
                    f"{entry['name']}: TRT1-сегмент вне {path.name}"
                )
            handle.seek(offset)
            header = handle.read(TRT1_HEADER_SIZE)
            magic, version, flags, n_values, n_scale = _SEGMENT_HEADER.unpack(
                header
            )
            if magic != TRT1_MAGIC or version != TRT1_VERSION:
                raise GgufError(
                    f"{entry['name']}: чужой заголовок TRT1 "
                    f"{magic!r}/{version}"
                )
            if n_values != entry["params"]:
                raise GgufError(
                    f"{entry['name']}: в TRT1 {n_values} тритов, "
                    f"в манифесте {entry['params']}"
                )
            expected = (
                TRT1_HEADER_SIZE + n_scale * 4
                + _trit_words_padded(n_values) * 4
            )
            if length < expected:
                raise GgufError(
                    f"{entry['name']}: TRT1-сегмент {length} байт, нужно "
                    f"{expected}"
                )


def _verify_q4_file(
    path: Path,
    entries: Sequence[dict[str, Any]],
    scale_dtype: str = "float32",
    asymmetric: bool = True,
) -> None:
    """Проверить, что каждый Q4-сегмент на месте и его заголовок согласован.

    `scale_dtype` входит в проверку длины: fp16-масштабы вдвое короче, и
    «сегмент длиной как у fp32» у fp16 означал бы лишние байты (или чужой
    flags), то есть рассинхрон манифеста с файлом.

    Импорт по умолчанию пишет Q4_1: magic `"Q4_1"`, заголовок 22 байта с
    явным `n_mins`, после масштабов ещё массив min. Проверка повторяет то,
    что сделает C-загрузчик (`loader.c`): неверный `n_mins` — отказ, а не
    «прочитать как Q4_0». `asymmetric=False` — старый симметричный Q4_0
    (`--q4-kind q4_0`, нужен 7B ради предела 4 ГиБ).
    """
    scale_bytes = Q4_SCALE_BYTES[scale_dtype]
    expected_flags = Q4_FLAG_F32_SCALES if scale_dtype == "float32" else 0
    expected_magic = Q4_MAGIC_ASYM if asymmetric else Q4_MAGIC
    header_size = Q4_ASYM_HEADER_SIZE if asymmetric else Q4_HEADER_SIZE
    aux_arrays = 2 if asymmetric else 1
    size = path.stat().st_size
    with open(path, "rb") as handle:
        for entry in entries:
            if entry.get("kind") != "q4":
                continue  # в tensors[] может быть и TRT1-эмбеддинг
            offset = int(entry["q4_offset"])
            length = int(entry["q4_length"])
            if offset < 0 or length < header_size or offset + length > size:
                raise GgufError(
                    f"{entry['name']}: сегмент [{offset}, {offset + length}) "
                    f"вне {path.name} ({size} байт)"
                )
            handle.seek(offset)
            header = handle.read(header_size)
            magic, version, flags, n_values, n_groups = _SEGMENT_HEADER.unpack(
                header[:Q4_HEADER_SIZE]
            )
            if magic != expected_magic or version != Q4_VERSION:
                raise GgufError(
                    f"{entry['name']}: чужой заголовок Q4-сегмента {magic!r}/"
                    f"{version} (ожидался {expected_magic!r})"
                )
            n_mins = (
                struct.unpack("<I", header[Q4_HEADER_SIZE:])[0]
                if asymmetric else 0
            )
            if flags != expected_flags:
                raise GgufError(
                    f"{entry['name']}: flags Q4-сегмента 0x{flags:02x}, "
                    f"ожидалось 0x{expected_flags:02x} для масштабов "
                    f"{scale_dtype}"
                )
            if n_values != entry["params"]:
                raise GgufError(
                    f"{entry['name']}: в заголовке {n_values} значений, "
                    f"в манифесте {entry['params']}"
                )
            expected_groups = (n_values + Q4_SCALE_BLOCK - 1) // Q4_SCALE_BLOCK
            if n_groups != expected_groups:
                raise GgufError(
                    f"{entry['name']}: групп {n_groups}, ожидалось "
                    f"{expected_groups}"
                )
            if asymmetric and n_mins != expected_groups:
                raise GgufError(
                    f"{entry['name']}: n_mins={n_mins}, ожидалось "
                    f"{expected_groups}"
                )
            expected_length = (
                header_size + n_groups * scale_bytes * aux_arrays
                + n_groups * Q4_GROUP_BYTES
            )
            if length < expected_length:
                raise GgufError(
                    f"{entry['name']}: сегмент {length} байт, а по заголовку "
                    f"нужно {expected_length}"
                )


def _verify_int8_file(path: Path, entries: Sequence[dict[str, Any]]) -> None:
    size = path.stat().st_size
    with open(path, "rb") as handle:
        for entry in entries:
            if entry.get("tied_to") or entry.get("dtype") != "int8":
                continue
            offset = int(entry["offset"])
            length = int(entry["length"])
            if offset < 0 or length < 18 or offset + length > size:
                raise GgufError(
                    f"{entry['name']}: INT8-сегмент вне {path.name}"
                )
            handle.seek(offset)
            header = handle.read(18)
            magic, version, flags, n_values, n_scale = _SEGMENT_HEADER.unpack(header)
            if magic != INT8_MAGIC or version != INT8_VERSION:
                raise GgufError(
                    f"{entry['name']}: чужой заголовок INT8-сегмента {magic!r}"
                )
            if n_values != entry["params"]:
                raise GgufError(
                    f"{entry['name']}: в INT8 {n_values} значений, "
                    f"в манифесте {entry['params']}"
                )
            expected = (
                18 + n_scale * 4 + n_values
            )
            if length < expected:
                raise GgufError(
                    f"{entry['name']}: INT8-сегмент {length} байт, нужно "
                    f"{expected}"
                )


# ---------------------------------------------------------------------------
# info и CLI
# ---------------------------------------------------------------------------


def _human_bytes(size: float) -> str:
    value = float(size)
    for unit in ("Б", "КиБ", "МиБ", "ГиБ", "ТиБ"):
        if value < 1024.0 or unit == "ТиБ":
            return f"{value:.1f} {unit}" if unit != "Б" else f"{int(value)} Б"
        value /= 1024.0
    return f"{int(size)} Б"


def describe(gguf: GgufFile) -> dict[str, Any]:
    """Сводка для `info`: конфиг, типы, поддержка."""
    types: dict[int, dict[str, Any]] = {}
    for info in gguf.tensors:
        record = types.setdefault(info.type_id, {
            "type": type_label(info.type_id),
            "name": GGUF_TYPE_NAMES.get(info.type_id, "?"),
            "tensors": 0,
            "params": 0,
            "bytes": 0,
            "supported": info.type_id in GGUF_TYPES,
            "examples": [],
        })
        record["tensors"] += 1
        record["params"] += info.n_elements
        if info.type_id in GGUF_TYPES:
            record["bytes"] += info.n_bytes
        if len(record["examples"]) < 3:
            record["examples"].append(info.name)
    config: dict[str, Any] = {}
    config_error = None
    try:
        cfg = infer_config(gguf)
        config = cfg.as_dict()
        config.update({
            "name": cfg.name,
            "architecture": cfg.architecture,
            "n_layers_source": cfg.n_layers_source,
            "vocab_source": cfg.vocab_source,
            "tie_word_embeddings": cfg.tie_word_embeddings,
        })
    except GgufError as exc:
        config_error = str(exc)
    unsupported = [
        record for record in types.values() if not record["supported"]
    ]
    return {
        "path": str(gguf.path),
        "file_bytes": gguf.file_size,
        "gguf_version": gguf.version,
        "data_offset": gguf.data_offset,
        "alignment": gguf.alignment,
        "tensor_count": len(gguf.tensors),
        "metadata": _json_safe({
            key: value
            for key, value in gguf.metadata.items()
            if isinstance(value, (str, int, float, bool))
        }),
        "config": config,
        "config_error": config_error,
        "types": [types[key] for key in sorted(types)],
        "supported": not unsupported and config_error is None,
        "unsupported_types": unsupported,
    }


def print_info(gguf: GgufFile) -> None:
    summary = describe(gguf)
    print(f"файл: {summary['path']} ({_human_bytes(summary['file_bytes'])})")
    print(
        f"GGUF v{summary['gguf_version']}, тензоров {summary['tensor_count']}, "
        f"выравнивание {summary['alignment']}, данные с "
        f"0x{summary['data_offset']:x}"
    )
    config = summary["config"]
    if config:
        print(
            f"модель: {config['name']} ({config['architecture']})\n"
            f"  hidden={config['hidden']}, слоёв={config['n_layers_source']}, "
            f"голов={config['n_heads']}/{config['n_kv_heads']}, "
            f"head_dim={config['head_dim']}, "
            f"intermediate={config['intermediate']}, "
            f"vocab={config['vocab_source']}, "
            f"ctx={config['context_len']}, "
            f"rope_theta={config['rope_theta']:g}, "
            f"eps={config['rms_norm_eps']:g}, "
            f"tie={config['tie_word_embeddings']}"
        )
    else:
        print(f"конфиг: НЕ разобран — {summary['config_error']}")
    print("типы тензоров:")
    for record in summary["types"]:
        status = "поддерживается" if record["supported"] else "ОТКАЗ"
        size = (
            _human_bytes(record["bytes"]) if record["supported"] else "—"
        )
        examples = ", ".join(record["examples"])
        print(
            f"  {record['type']:<16} {record['tensors']:4d} тензоров, "
            f"{record['params'] / 1e6:9.1f} МЭ, {size:>10}  {status}"
            f"  [{examples}]"
        )
    if summary["supported"]:
        print("импорт возможен: все типы поддержаны, имена/формы проверяемы")
    else:
        if summary["unsupported_types"]:
            names = ", ".join(
                record["type"] for record in summary["unsupported_types"]
            )
            print(f"импорт невозможен: неподдержанные типы {names}")
        if summary["config_error"]:
            print(f"импорт невозможен: {summary['config_error']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gguf_import.py",
        description=(
            "Импорт GGUF (ollama) в VTM: разбор заголовка/метаданных/тензоров "
            "и запись манифеста eva32.ternary.manifest v2 + q4.bin / "
            "non_ternary.bin. Без внешних зависимостей."
        ),
        epilog=(
            "Примеры:\n"
            "  python3 tools/gguf_import.py info ~/.ollama/models/blobs/sha256-...\n"
            "  python3 tools/gguf_import.py import model.gguf --out export/gguf-q4\n"
            "  python3 tools/gguf_import.py import model.gguf --out /tmp/slice "
            "--format trit --limit-layers 1 --limit-vocab 4096\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    info = sub.add_parser("info", help="показать заголовок, конфиг и типы")
    info.add_argument("model", help="путь к .gguf (или blob ollama)")
    info.add_argument("--json", action="store_true", help="вывод в JSON")

    imp = sub.add_parser("import", help="записать манифест и веса VTM")
    imp.add_argument("model", help="путь к .gguf (или blob ollama)")
    imp.add_argument("--out", required=True, help="каталог результата")
    imp.add_argument(
        "--format", choices=("q4", "trit"), default="q4",
        help="q4 — kind 5 в q4.bin (по умолчанию); trit — TRT1/kind 0 в "
             "ternary.trit; компилятор читает оба",
    )
    imp.add_argument(
        "--q4-scale-dtype", choices=("fp32", "fp16", "float32", "float16"),
        default="fp32",
        help="масштабы Q4-сегментов: fp32 (1 бит/вес, по умолчанию) или "
             "fp16 (0,5 бита/вес; для 7B — единственный способ влезть в "
             "4 ГиБ, ADR 0016)",
    )
    imp.add_argument(
        "--q4-kind", choices=("auto", "q4_1", "q4_0", "asym", "sym"),
        default="auto",
        help="вид Q4-сегмента (kind 5): auto (по умолчанию) — q4_1, если "
             "оценка контейнера влезает в 4 ГиБ (3B), иначе q4_0 (7B: Q4_1 "
             "даёт 106 %% предела); q4_1 — min на группу 32, почти без потерь "
             "на Q4_K-подблоках (+2–4 байта/группа); q4_0 — симметричный",
    )
    imp.add_argument(
        "--embed", choices=("int8", "trit", "fp16", "fp32", "float16", "float32"),
        default="int8",
        help="хранение таблицы словаря: int8 (kind 4, по умолчанию), trit "
             "(TRT1/kind 0 в ternary.trit; 7B — 0,12 ГиБ вместо 0,53 ГиБ), "
             "fp16/fp32 — плотная (kind 2/1). Синонимы: float16/float32",
    )
    imp.add_argument(
        "--tie", choices=("auto", "yes", "no"), default="auto",
        help="tie_word_embeddings: auto — по наличию output.weight",
    )
    imp.add_argument(
        "--limit-layers", type=int, default=None,
        help="ЧАСТИЧНАЯ модель: только первые N слоёв (дымовой тест)",
    )
    imp.add_argument(
        "--limit-vocab", type=int, default=None,
        help="ЧАСТИЧНАЯ модель: только первые V строк словаря (дымовой тест)",
    )
    imp.add_argument(
        "--stats", action="store_true",
        help="считать относительную RMS-ошибку переквантования",
    )
    imp.add_argument(
        "--progress", action="store_true", help="печатать каждый тензор в stderr"
    )
    imp.add_argument(
        "--force", action="store_true", help="перезаписать непустой --out"
    )
    imp.add_argument("--json", action="store_true", help="итог в JSON")
    return parser


def _print_result(result: ImportResult) -> None:
    cfg = result.config
    print(
        f"импорт: {cfg.name} ({cfg.architecture}) -> {result.manifest_path.parent}"
    )
    print(
        f"  конфиг: hidden={cfg.hidden}, слоёв={cfg.n_layers}"
        f"{' (из ' + str(cfg.n_layers_source) + ')' if cfg.n_layers != cfg.n_layers_source else ''}, "
        f"голов={cfg.n_heads}/{cfg.n_kv_heads}, head_dim={cfg.head_dim}, "
        f"vocab={cfg.vocab}"
        f"{' (из ' + str(cfg.vocab_source) + ')' if cfg.vocab != cfg.vocab_source else ''}, "
        f"rope_theta={cfg.rope_theta:g}, eps={cfg.rms_norm_eps:g}, "
        f"tie={cfg.tie_word_embeddings}"
    )
    q4 = result.manifest.get("q4")
    trit = result.manifest.get("trit")
    if q4:
        kind = "Q4_1 (asym, min на группу)" if q4["asymmetric"] else "Q4_0 (sym)"
        print(
            f"  Linear → {kind}, kind 5, масштабы {q4['scale_dtype']}: "
            f"{q4['params'] / 1e6:.1f} МЭ, "
            f"q4.bin {_human_bytes(result.q4_bytes)}"
        )
    if trit:
        trit_bytes = result.manifest["files"]["ternary.trit"]["bytes"]
        where = "триты + эмбеддинг" if trit.get("contains_embed") else "триты"
        print(
            f"  Linear → {where} (kind 0): {trit['params'] / 1e6:.1f} МЭ, "
            f"ternary.trit {_human_bytes(trit_bytes)}"
        )
    embed = result.manifest["embeddings"]
    print(
        f"  эмбеддинг → {embed['dtype']}: {embed['params'] / 1e6:.1f} МЭ, "
        f"{embed.get('storage')} {_human_bytes(embed['bytes'])}"
    )
    total_bytes = (
        result.q4_bytes + result.non_ternary_bytes
        + int(result.manifest["files"]["ternary.trit"]["bytes"])
    )
    print(
        f"  исходник {_human_bytes(result.source_bytes)} → "
        f"файлы {_human_bytes(total_bytes)} "
        f"({total_bytes / result.source_bytes:.2f}×)"
    )
    if result.manifest.get("partial"):
        print(
            "  ВНИМАНИЕ: частичная модель (--limit-layers/--limit-vocab) — "
            "только для дымового теста, не для качества"
        )
    for role, stats in sorted(result.stats.items()):
        if stats.values:
            print(
                f"  переквантование {role}: относительная RMS-ошибка "
                f"{stats.rel_rms:.1%}, max|Δ|={stats.max_abs:.4g}"
            )
    compiler = result.manifest["compiler"]
    if compiler["ready"]:
        print("  компилятор: манифест читается без изменений")
    else:
        print("  компилятор: НЕ готов — манифест описывает kind 5, а")
        for item in compiler["missing"]:
            print(f"    - {item}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "info":
            gguf = read_gguf(args.model)
            if args.json:
                print(json.dumps(describe(gguf), ensure_ascii=False, indent=2))
            else:
                print_info(gguf)
            return 0
        result = import_gguf(
            args.model,
            args.out,
            fmt=args.format,
            embed_mode=args.embed,
            q4_scale_dtype=args.q4_scale_dtype,
            q4_kind=args.q4_kind,
            tie=args.tie,
            limit_layers=args.limit_layers,
            limit_vocab=args.limit_vocab,
            stats=args.stats,
            force=args.force,
            progress=args.progress,
        )
    except GgufError as exc:
        print(f"gguf_import: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"gguf_import: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps({
            "manifest": str(result.manifest_path),
            "q4": str(result.q4_path) if result.q4_path else None,
            "non_ternary": str(result.non_ternary_path),
            "config": result.config.as_dict(),
            "compiler_ready": result.compiler_ready,
            "stats": {
                role: {
                    "values": stats.values,
                    "rel_rms": stats.rel_rms,
                    "max_abs": stats.max_abs,
                }
                for role, stats in result.stats.items()
            },
        }, ensure_ascii=False, indent=2))
    else:
        _print_result(result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
