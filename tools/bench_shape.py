#!/usr/bin/env python3
"""Синтетический контейнер 1.5B-формы: размер, память и скорость без обучения.

Все замеры VTM до сих пор делались на demo-модели (4 млн параметров), поэтому
никто не знал, что будет на реальной раскладке Qwen2.5-1.5B. Обучать модель для
этого не нужно: интерпретатору важны форма и объём весов, а не их значения.
Модуль собирает ``.vtm`` нужной формы со случайными тритами и меряет:

* размер файла (и разбивку: триты, масштабы, fp32, KV-кэш, программа);
* время загрузки и RSS после загрузки;
* TTFT (первый токен) и декод (токенов/с);
* всё это при ``VTM_THREADS=1/4/8`` и с mmap / ``VTM_NO_MMAP=1``.

Контейнер собирается из настоящего графа компилятора: ``vtm_compile.compile_model``
даёт ту же программу и тот же список тензоров, что и для обученной модели, а
данные подставляются случайными. Поэтому числа относятся к настоящему
байткоду Qwen-формы, а не к «примерно такой же» синтетике.

Почему не ``vtm_ref.write_vtm``: он держит весь payload в памяти (``tensors`` →
``bytes``), а на 1.5B это ~345 МиБ payload'а плюс 1,5 млрд тритов списком.
Здесь тот же формат пишется потоком, тензор за тензором; раскладка сверяется с
``VTM.estimated_file_size()`` (публичный расчёт самого ``vtm_ref``), так что
контейнер байт-в-байт по раскладке и читается ``vtm_ref.read_vtm``/C-загрузчиком.

Случайные триты — валидные слова pack20 (каждое < 3²⁰), масштабы блока 128 —
константа (по умолчанию 0,02, типичный порядок для тритовой модели). Значения
весов на скорость не влияют: ядро всё равно проходит все триты.

Формы:

    python3 tools/bench_shape.py --dry-run                 # только оценка
    python3 tools/bench_shape.py                           # 1.5B-форма + замеры
    python3 tools/bench_shape.py --hidden 512 --layers 4 --vocab 8192 --ctx 512
    python3 tools/bench_shape.py --reduced                 # уменьшенная 1.5B
    python3 tools/bench_shape.py --json > shape.json

Тяжёлый прогон в тестах — под ``VTM_BENCH_SHAPE_HEAVY=1``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools import vtm_compile, vtm_ref  # noqa: E402

try:  # RSS текущего процесса (сборка контейнера) — POSIX, как и весь рантайм
    import resource
except ImportError:  # pragma: no cover - Windows не поддерживается и в C
    resource = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Константы формата (сверяются с vtm_ref тестом)
# ---------------------------------------------------------------------------

TRT1_MAGIC = vtm_ref.TRT1_MAGIC
TRT1_VERSION = vtm_ref.TRT1_VERSION
TRT1_HEADER_SIZE = vtm_ref.TRT1_HEADER_SIZE
HEADER_SIZE = vtm_ref.HEADER_SIZE
TENSOR_ENTRY_SIZE = vtm_ref.TENSOR_ENTRY_SIZE
INSTR_SIZE = vtm_ref.INSTR_SIZE
ALIGNMENT = vtm_ref.ALIGNMENT
MAX_DIMS = vtm_ref.MAX_DIMS
NO_OFFSET = vtm_ref.NO_OFFSET
KIND_TRITS = vtm_ref.KIND_TRITS
KIND_F32 = vtm_ref.KIND_F32
SCALE_BLOCK = vtm_ref.SCALE_BLOCK
TRITS_PER_STORAGE_BLOCK = vtm_ref.TRITS_PER_STORAGE_BLOCK
HEADER_FLAGS = vtm_ref.VFLAG_F32_SCALES | vtm_ref.VFLAG_CAUSAL

_HEADER = struct.Struct("<4sHHIII")
_RESERVED = struct.Struct("<III")
_TENSOR_ENTRY = struct.Struct("<24sBB4IIIH")
_TRT1 = struct.Struct("<4sBBQI")

#: Старший байт слова pack20 → заведомо допустимый (≤ 206).
#:
#: Слово pack20 — это число в системе по основанию 3 (``Σ dᵢ·3ⁱ ≤ 3²⁰−1``),
#: а не четыре байта-группы по 243: 243⁴ = 3²⁰, но 243 ≠ 256, поэтому байты
#: нельзя переводить по отдельности (``unpack20`` отверг бы слово). Достаточно
#: ограничить старший байт: ``206·2²⁴ + 2²⁴−1 < 3²⁰``, тогда любое слово
#: валидно. Старший байт теряет 19 % диапазона (0..206 вместо 0..255) — для
#: бенчмарка это не важно: важна длина и распаковка, а не распределение тритов.
_HIGH_BYTE = bytes(value % 207 for value in range(256))

#: Размер порции случайных/нулевых байт при записи payload'а.
CHUNK = 1 << 20

#: Сколько секунд ждать один прогон (1.5B на скалярном бэкенде — секунды на токен).
DEFAULT_TIMEOUT = 3600.0

#: Меньше этого числа тритов форма — не 1.5B, и вердикты «30 ток/с / 450 МиБ»
#: к ней не применяются (сравнивать игрушку с оценкой бессмысленно).
VERDICT_MIN_TRITS = 100_000_000


class BenchShapeError(RuntimeError):
    """Ошибка бенчмарка: битая форма, нет бинарника, непонятный вывод."""


# ---------------------------------------------------------------------------
# Форма
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Shape:
    """Форма Qwen-подобного декодера (по умолчанию — Qwen2.5-1.5B).

    ``heads``/``kv_heads``/``head_dim`` не обязаны давать ``hidden`` (у Qwen3
    head_dim свой), компилятор это не проверяет — проверяется только чётность
    ``head_dim`` (RoPE) и кратность ``heads`` числу ``kv_heads`` (GQA).
    """

    hidden: int = 1536
    layers: int = 28
    heads: int = 12
    kv_heads: int = 2
    head_dim: int = 128
    intermediate: int = 8960
    vocab: int = 151936
    ctx: int = 2048
    rope_theta: float = 1_000_000.0
    eps: float = 1e-6
    seed: int = 7
    trit_scale: float = 0.02

    def __post_init__(self) -> None:
        for name in ("hidden", "layers", "heads", "kv_heads", "head_dim",
                     "intermediate", "vocab", "ctx", "seed"):
            value = int(getattr(self, name))
            if value <= 0:
                raise BenchShapeError(f"{name} должен быть > 0, получено {value}")
            object.__setattr__(self, name, value)
        if self.heads % self.kv_heads:
            raise BenchShapeError(
                f"heads={self.heads} не делятся на kv_heads={self.kv_heads}"
            )
        if self.head_dim % 2:
            raise BenchShapeError(f"head_dim={self.head_dim} должен быть чётным (RoPE)")
        if not self.rope_theta > 0:
            raise BenchShapeError(f"rope_theta должен быть > 0: {self.rope_theta}")
        if self.eps < 0:
            raise BenchShapeError(f"eps не может быть отрицательным: {self.eps}")
        if not self.trit_scale > 0:
            raise BenchShapeError(f"trit_scale должен быть > 0: {self.trit_scale}")

    @property
    def q_dim(self) -> int:
        return self.heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.kv_heads * self.head_dim

    def as_dict(self) -> dict[str, Any]:
        return {
            "hidden": self.hidden,
            "layers": self.layers,
            "heads": self.heads,
            "kv_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "intermediate": self.intermediate,
            "vocab": self.vocab,
            "ctx": self.ctx,
            "rope_theta": self.rope_theta,
            "eps": self.eps,
            "seed": self.seed,
            "trit_scale": self.trit_scale,
        }

    def slug(self) -> str:
        return f"h{self.hidden}_l{self.layers}_v{self.vocab}_c{self.ctx}"

    def reduced(self) -> "Shape":
        """Уменьшенная форма той же архитектуры: ~1/16 тритов.

        Половина hidden, четверть слоёв/intermediate/vocab/ctx — ровно тот
        совет, который печатается в предупреждении о тяжёлой генерации.
        """
        return Shape(
            hidden=max(8, self.hidden // 2),
            layers=max(1, self.layers // 4),
            heads=self.heads,
            kv_heads=self.kv_heads,
            head_dim=self.head_dim,
            intermediate=max(8, self.intermediate // 4),
            vocab=max(8, self.vocab // 4),
            ctx=max(8, self.ctx // 4),
            rope_theta=self.rope_theta,
            eps=self.eps,
            seed=self.seed,
            trit_scale=self.trit_scale,
        )


DEFAULT_SHAPE = Shape()


def model_config(shape: Shape) -> "vtm_compile.ModelConfig":
    """Конфиг компилятора под форму: trit-эмбеддинги, tied lm_head, MM."""
    return vtm_compile.ModelConfig(
        hidden=shape.hidden,
        n_heads=shape.heads,
        n_kv_heads=shape.kv_heads,
        head_dim=shape.head_dim,
        intermediate=shape.intermediate,
        vocab=shape.vocab,
        n_layers=shape.layers,
        context_len=shape.ctx,
        eps=shape.eps,
        rope_theta=shape.rope_theta,
        lm_head="mm",
        tie_word_embeddings=True,
        embed_quant="trit",
    )


def shape_param_counts(shape: Shape) -> dict[str, int]:
    """Параметры формы по формуле — без сборки графа.

    Ровно те тензоры, которые эмитит ``compile_layer``/``compile_model``:
    на слой ``2·q·hidden`` (q_proj, o_proj), ``2·kv·hidden`` (k_proj, v_proj),
    ``3·intermediate·hidden`` (gate, up, down); плюс таблица словаря
    ``vocab·hidden`` (tied lm_head отдельного тензора не имеет).
    """
    q_dim = shape.heads * shape.head_dim
    kv_dim = shape.kv_heads * shape.head_dim
    per_layer = 2 * q_dim * shape.hidden + 2 * kv_dim * shape.hidden + \
        3 * shape.intermediate * shape.hidden
    embed = shape.vocab * shape.hidden
    return {
        "trits_per_layer": per_layer,
        "trit_weights": per_layer * shape.layers,
        "trit_embed": embed,
        "trit_params": per_layer * shape.layers + embed,
        "fp32_norms": 2 * shape.hidden * shape.layers + shape.hidden,
        "kv_cache_elems": 2 * shape.ctx * kv_dim * shape.layers,
    }


def trit_payload_size(n_trits: int) -> int:
    """Размер TRT1-сегмента (заголовок + масштабы + слова pack20)."""
    return (
        TRT1_HEADER_SIZE
        + vtm_ref.n_scale_blocks(n_trits) * 4
        + vtm_ref.storage_words(n_trits) * 4
    )


def encode_meta(meta: Mapping[str, Any]) -> bytes:
    """JSON меты ровно как у ``vtm_ref`` (sort_keys, utf-8 без escape-последовательностей)."""
    if not meta:
        return b""
    try:
        text = json.dumps(meta, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise BenchShapeError(f"мета не сериализуется в JSON: {exc}") from exc
    return text.encode("utf-8")


# ---------------------------------------------------------------------------
# План контейнера: граф компилятора + оценка размера
# ---------------------------------------------------------------------------


@dataclass
class ContainerPlan:
    """Готовый граф формы: тензоры (без данных), программа, мета и размер."""

    shape: Shape
    vtm: "vtm_ref.VTM"
    meta: dict[str, Any]
    file_size: int
    trit_bytes: int
    scale_bytes: int
    fp32_bytes: int
    kv_cache_bytes: int
    trit_params: int
    data_bytes: int

    @property
    def tensors(self) -> list["vtm_ref.Tensor"]:
        return self.vtm.tensors

    @property
    def program(self) -> list["vtm_ref.Instr"]:
        return self.vtm.program

    def summary(self) -> dict[str, Any]:
        return {
            "shape": self.shape.as_dict(),
            "file_bytes": self.file_size,
            "file_mib": _mib(self.file_size),
            "tensors": len(self.vtm.tensors),
            "instructions": len(self.vtm.program),
            "trit_params": self.trit_params,
            "trit_mib": _mib(self.trit_bytes),
            "scale_mib": _mib(self.scale_bytes),
            "fp32_mib": _mib(self.fp32_bytes),
            "kv_cache_mib": _mib(self.kv_cache_bytes),
            "data_mib": _mib(self.data_bytes),
        }


def build_plan(shape: Shape) -> ContainerPlan:
    """Собрать граф формы: настоящая программа компилятора, тензоры без данных.

    ``compile_model`` даёт ровно ту программу и тот список тензоров, что и для
    обученной модели; данные не нужны ни для размера, ни для скорости. Рабочие
    тензоры (активации) создаются с нулями — они сразу выбрасываются
    (``data=None``), а payload пишется потоком. KV-кэш (117 МиБ на 1.5B) в
    контейнер не пишется вовсе (``TFLAG_KV_CACHE``, задача 0101): его буфер
    создаёт загрузчик.
    """
    cfg = model_config(shape)
    embed = vtm_ref.Tensor.placeholder(
        "emb.w", (shape.vocab, shape.hidden), KIND_TRITS
    )
    try:
        vtm = vtm_compile.compile_model(cfg, weights={"embed": embed})
    except (ValueError, TypeError) as exc:  # несовместимая форма — не наша вина
        raise BenchShapeError(f"форма не собирается компилятором: {exc}") from exc

    for tensor in vtm.tensors:
        if tensor.kind != KIND_TRITS:
            tensor.data = None

    meta = dict(vtm.meta)
    counts = shape_param_counts(shape)
    meta["generator"] = "tools/bench_shape.py"
    meta["shape_bench"] = {
        "weights": "random trits",
        "trained": False,
        "seed": shape.seed,
        "trit_scale": shape.trit_scale,
        "trit_params": counts["trit_params"],
        "trit_embed": counts["trit_embed"],
        "trit_weights": counts["trit_weights"],
    }
    meta["synthetic"] = {
        "kind": "shape",
        "seed": shape.seed,
        "tokens": [1, 2, 3],
        "max_new": 0,
        "inputs_tensor": "emb.w",
    }
    vtm.meta = meta

    tensors = vtm.tensors
    trit_bytes = scale_bytes = fp32_bytes = kv_bytes = trit_params = data_bytes = 0
    for tensor in tensors:
        size = vtm_ref.tensor_payload_size(tensor)
        data_bytes += size
        if tensor_is_kv_cache(tensor):
            # В файле кэша нет, но в памяти он есть: ctx × kv_dim × 4 на K и V.
            kv_bytes += tensor.n_elements * 4
            continue
        if tensor.kind == KIND_TRITS:
            trit_params += tensor.n_elements
            scale_bytes += vtm_ref.n_scale_blocks(tensor.n_elements) * 4
            trit_bytes += size - vtm_ref.n_scale_blocks(tensor.n_elements) * 4
        else:
            fp32_bytes += size
    return ContainerPlan(
        shape=shape,
        vtm=vtm,
        meta=meta,
        file_size=vtm.estimated_file_size(),
        trit_bytes=trit_bytes,
        scale_bytes=scale_bytes,
        fp32_bytes=fp32_bytes,
        kv_cache_bytes=kv_bytes,
        trit_params=trit_params,
        data_bytes=data_bytes,
    )


# ---------------------------------------------------------------------------
# Потоковая запись контейнера
# ---------------------------------------------------------------------------


def _write_random_words(handle: Any, n_bytes: int, rng: Any) -> None:
    """Случайные валидные слова pack20 (каждое < 3²⁰).

    ``n_bytes`` кратно 4 (так устроен ``storage_words``): старшие байты слов —
    индексы 3, 7, 11, … — переводятся таблицей ``_HIGH_BYTE``; это срез и
    ``bytes.translate``, то есть C-скорость на сотнях мегабайт.
    """
    if n_bytes % 4:
        raise BenchShapeError(f"payload тритов должен быть кратен 4: {n_bytes}")
    left = n_bytes
    while left > 0:
        step = min(CHUNK, left)
        raw = bytearray(rng.getrandbits(8 * step).to_bytes(step, "little"))
        raw[3::4] = bytes(raw[3::4]).translate(_HIGH_BYTE)
        handle.write(bytes(raw))
        left -= step


def _write_trit_payload(handle: Any, tensor: "vtm_ref.Tensor", shape: Shape,
                        rng: Any) -> None:
    n_trits = tensor.n_elements
    n_scales = vtm_ref.n_scale_blocks(n_trits)
    handle.write(_TRT1.pack(TRT1_MAGIC, TRT1_VERSION, 0x01, n_trits, n_scales))
    handle.write(struct.pack("<f", float(shape.trit_scale)) * n_scales)
    _write_random_words(handle, vtm_ref.storage_words(n_trits) * 4, rng)


def _is_norm_weight(name: str) -> bool:
    return name == "final_norm.w" or name.endswith((".norm1.w", ".norm2.w"))


def tensor_is_kv_cache(tensor: "vtm_ref.Tensor") -> bool:
    """KV-кэш внимания: в файле только запись таблицы (задача 0101).

    Флаг ``TFLAG_KV_CACHE`` появился в формате v0.2 позже самого формата, и
    контейнеры без него тоже валидны, поэтому проверка через ``getattr``: старый
    ``vtm_ref`` без поля ``flags`` считает все тензоры обычными.
    """
    return bool(getattr(tensor, "flags", 0) & getattr(vtm_ref, "TFLAG_KV_CACHE", 0))


def _write_f32_payload(handle: Any, tensor: "vtm_ref.Tensor") -> None:
    """fp32-тензор: нормы — единицы (как обученные), остальное — нули.

    Активации и KV-кэш в контейнере всё равно перезаписываются; единичные
    нормы нужны, чтобы граф не вырождался в ноль на первом же слое.
    """
    value = 1.0 if _is_norm_weight(tensor.name) else 0.0
    unit = struct.pack("<f", value)
    block = unit * (CHUNK // 4)
    left = tensor.n_elements * 4
    while left > 0:
        step = min(len(block), left)
        handle.write(block[:step])
        left -= step


def _write_fp_payload(handle: Any, tensor: "vtm_ref.Tensor") -> None:
    """fp16/bf16 — нулями: в графе формы они не встречаются, но формат знаем."""
    width = 2
    block = b"\x00" * CHUNK
    left = tensor.n_elements * width
    while left > 0:
        step = min(CHUNK, left)
        handle.write(block[:step])
        left -= step


def _write_payload(handle: Any, tensor: "vtm_ref.Tensor", shape: Shape,
                   rng: Any) -> None:
    if tensor_is_kv_cache(tensor):
        # У KV-кэша payload'а в файле нет: буфер создаёт загрузчик (calloc).
        return
    if tensor.kind == KIND_TRITS:
        _write_trit_payload(handle, tensor, shape, rng)
    elif tensor.kind == KIND_F32:
        _write_f32_payload(handle, tensor)
    else:
        _write_fp_payload(handle, tensor)


def tensor_table(tensors: Sequence["vtm_ref.Tensor"]) -> bytes:
    """Таблица тензоров (52 байта на запись) — после проставления offset'ов."""
    out = bytearray()
    for tensor in tensors:
        dims = list(tensor.dims) + [0] * (MAX_DIMS - len(tensor.dims))
        out += _TENSOR_ENTRY.pack(
            tensor.name.encode("utf-8"), tensor.kind, len(tensor.dims), *dims,
            tensor.offset, tensor.scale_offset,
            int(getattr(tensor, "flags", 0)),
        )
    return bytes(out)


def write_container(path: str | os.PathLike[str], plan: ContainerPlan,
                    *, seed: int | None = None) -> int:
    """Записать контейнер потоком; вернуть размер файла.

    Раскладка — как у ``vtm_ref.write_vtm``: заголовок 32, таблица 52·N,
    выравнивание данных и каждого payload'а на 32, затем мета и программа.
    Размер сверяется с ``VTM.estimated_file_size()`` — если формула разошлась
    с записью, это ошибка, а не «примерно столько».
    """
    import random

    rng = random.Random(plan.shape.seed if seed is None else int(seed))
    tensors = plan.tensors
    program = plan.program
    n_tensors = len(tensors)

    table_end = HEADER_SIZE + TENSOR_ENTRY_SIZE * n_tensors
    data_start = vtm_ref.align_up(table_end)

    offset = 0
    layout: list[tuple["vtm_ref.Tensor", int]] = []
    for tensor in tensors:
        if tensor_is_kv_cache(tensor):
            # Раскладка ровно как у vtm_ref.write_vtm: у кэша offset'ы нулевые,
            # места в секции данных он не занимает (задача 0101).
            tensor.offset = 0
            tensor.scale_offset = 0
            layout.append((tensor, 0))
            continue
        size = vtm_ref.tensor_payload_size(tensor)
        tensor.offset = offset
        tensor.scale_offset = NO_OFFSET if tensor.kind == KIND_TRITS else 0
        layout.append((tensor, size))
        offset = vtm_ref.align_up(offset + size)
    data_size = offset

    meta_bytes = encode_meta(plan.meta)
    meta_offset = data_start + data_size
    program_offset = meta_offset + vtm_ref.align_up(len(meta_bytes))
    program_bytes = b"".join(instr.pack() for instr in program)
    file_size = program_offset + len(program_bytes)
    if file_size != plan.file_size:
        raise BenchShapeError(
            f"раскладка разошлась с vtm_ref: запись {file_size} байт, "
            f"оценка {plan.file_size} байт"
        )
    if file_size >= 2 ** 32:
        raise BenchShapeError(
            f"контейнер {file_size} байт не влезает в 32-битные смещения VTM"
        )

    header = _HEADER.pack(
        vtm_ref.VTM_MAGIC, vtm_ref.VTM_VERSION, HEADER_FLAGS,
        len(program), n_tensors, len(program_bytes),
    ) + _RESERVED.pack(program_offset, meta_offset, len(meta_bytes))
    table = tensor_table(tensors)

    parent = os.path.dirname(os.path.abspath(os.fspath(path)))
    os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(header)
        handle.write(table)
        handle.write(b"\x00" * (data_start - table_end))
        for tensor, size in layout:
            _write_payload(handle, tensor, plan.shape, rng)
            padding = vtm_ref.align_up(size) - size
            if padding:
                handle.write(b"\x00" * padding)
        if meta_bytes:
            handle.write(meta_bytes)
            handle.write(b"\x00" * (vtm_ref.align_up(len(meta_bytes)) - len(meta_bytes)))
        handle.write(program_bytes)

    written = os.path.getsize(path)
    if written != file_size:  # pragma: no cover - защита от рассинхрона
        raise BenchShapeError(f"записано {written} байт, ожидалось {file_size}")
    return written


# ---------------------------------------------------------------------------
# Разбор вывода процессов
# ---------------------------------------------------------------------------

_RE_STATS = re.compile(
    r"stats:\s*токенов=(\d+),\s*время=([0-9.]+)\s*с,\s*токенов/с=([0-9.]+)"
)
_RE_COUNTERS = re.compile(
    r"тензоров:\s*(\d+),\s*инструкций:\s*(\d+),\s*бэкенд:\s*(\S+)"
)
_RE_MAC_RSS = re.compile(r"^\s*(\d+)\s+maximum resident set size", re.MULTILINE)
_RE_LINUX_RSS = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)", re.I)
_RE_MAC_WALL = re.compile(r"^\s*([0-9.]+)\s+real\b", re.MULTILINE)
_RE_LINUX_WALL = re.compile(
    r"Elapsed \(wall clock\) time[^\n]*?:\s*(\d+(?::\d+)+(?:\.\d+)?|\d+\.\d+)",
    re.I,
)


@dataclass
class Stats:
    """Строка ``stats:`` рантайма (только реальные проходы, без загрузки)."""

    tokens: int
    seconds: float
    tok_per_s: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "tokens": self.tokens,
            "seconds": self.seconds,
            "tok_per_s": self.tok_per_s,
        }


def _seconds_from_clock(text: str) -> float | None:
    """``h:mm:ss`` / ``m:ss`` / ``ss`` (формат GNU time) → секунды."""
    parts = text.strip().split(":")
    try:
        values = [float(part) for part in parts]
    except ValueError:
        return None
    if not values:
        return None
    total = 0.0
    for value in values:
        total = total * 60.0 + value
    return total


def parse_time_output(text: str) -> dict[str, Any]:
    """Разобрать вывод ``/usr/bin/time``: пиковый RSS и wall-время.

    Понимает обе ветки: BSD/macOS (``-l``: «N  maximum resident set size»,
    байты, и «S real») и GNU (``-v``: "Maximum resident set size (kbytes)",
    «Elapsed (wall clock) time»). Возвращает ``{}``, если строк нет — тогда
    замер остаётся без RSS, а не падает.
    """
    if not isinstance(text, str):
        raise BenchShapeError("parse_time_output: ожидался текст")
    result: dict[str, Any] = {}
    match = _RE_MAC_RSS.search(text)
    if match:
        result["max_rss_bytes"] = int(match.group(1))
    else:
        match = _RE_LINUX_RSS.search(text)
        if match:
            result["max_rss_bytes"] = int(match.group(1)) * 1024
    match = _RE_MAC_WALL.search(text)
    if match:
        result["wall_s"] = float(match.group(1))
    else:
        match = _RE_LINUX_WALL.search(text)
        if match:
            seconds = _seconds_from_clock(match.group(1))
            if seconds is not None:
                result["wall_s"] = seconds
    return result


def parse_stats_output(text: str) -> Stats | None:
    """Разобрать строку ``stats:`` из stderr; нет строки — ``None``."""
    if not isinstance(text, str):
        raise BenchShapeError("parse_stats_output: ожидался текст")
    match = _RE_STATS.search(text)
    if not match:
        return None
    return Stats(
        tokens=int(match.group(1)),
        seconds=float(match.group(2)),
        tok_per_s=float(match.group(3)),
    )


def parse_run_header(text: str) -> dict[str, Any]:
    """Разобрать шапку ``vtm run``/``vtm inspect``: счётчики и бэкенд."""
    match = _RE_COUNTERS.search(text or "")
    if not match:
        return {}
    return {
        "tensors": int(match.group(1)),
        "instructions": int(match.group(2)),
        "backend": match.group(3),
    }


# ---------------------------------------------------------------------------
# Запуск процессов
# ---------------------------------------------------------------------------

#: (команда, таймаут, окружение) -> CompletedProcess
Runner = Callable[[Sequence[str], "float | None", "Mapping[str, str] | None"],
                  "subprocess.CompletedProcess[str]"]


def default_runner(
    command: Sequence[str],
    timeout: float | None,
    env: Mapping[str, str] | None,
) -> "subprocess.CompletedProcess[str]":
    """Настоящий запуск; тесты подменяют runner целиком."""
    return subprocess.run(
        list(command), capture_output=True, text=True, timeout=timeout, env=dict(env) if env else None
    )


def run_command(
    command: Sequence[str],
    *,
    runner: Runner | None = None,
    timeout: float | None = None,
    env: Mapping[str, str] | None = None,
) -> tuple["subprocess.CompletedProcess[str]", float]:
    """Запустить команду, вернуть (процесс, wall-секунды). Ошибки — BenchShapeError."""
    run = default_runner if runner is None else runner
    shown = " ".join(str(part) for part in command)
    started = time.perf_counter()
    try:
        proc = run(list(command), timeout, env)
    except subprocess.TimeoutExpired as exc:
        raise BenchShapeError(f"таймаут {timeout:g} с: {shown}") from exc
    except OSError as exc:
        raise BenchShapeError(f"не удалось запустить {shown}: {exc}") from exc
    wall = time.perf_counter() - started
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip().splitlines()
        tail = "; ".join(stderr[-3:]) if stderr else "без stderr"
        raise BenchShapeError(f"код {proc.returncode}: {shown}: {tail}")
    return proc, wall


def build_inspect_command(vtm_bin: str, model: str) -> list[str]:
    return [str(vtm_bin), "inspect", str(model)]


def build_run_command(
    vtm_bin: str,
    model: str,
    tokens: Sequence[int],
    max_new: int,
    *,
    backend: str | None = None,
) -> list[str]:
    if not tokens:
        raise BenchShapeError("build_run_command: пустой промпт")
    if max_new < 0:
        raise BenchShapeError("build_run_command: max_new < 0")
    command = [
        str(vtm_bin), "run", str(model),
        "--tokens", ",".join(str(int(token)) for token in tokens),
        "--max-new", str(int(max_new)),
        "--stats",
    ]
    if backend:
        command += ["--backend", str(backend)]
    return command


def time_utility_command() -> list[str] | None:
    """Обёртка для RSS: ``/usr/bin/time -l`` (macOS) или ``-v`` (GNU)."""
    path = "/usr/bin/time"
    if not os.path.exists(path):
        return None
    return [path, "-l"] if sys.platform == "darwin" else [path, "-v"]


def wrap_time(command: Sequence[str], time_cmd: Sequence[str] | None) -> list[str]:
    if not time_cmd:
        return [str(part) for part in command]
    return [*(str(part) for part in time_cmd), *(str(part) for part in command)]


def calibrate_time_overhead(
    *, runner: Runner | None = None, time_cmd: Sequence[str] | None,
    repeats: int = 3,
) -> float:
    """Накладные расходы обёртки ``time`` (запуск пустого процесса).

    Без этой поправки «время загрузки» включало бы fork/exec самого `/usr/bin/time`
    (~5–10 мс), а на mmap-загрузке это заметная доля.
    """
    if not time_cmd:
        return 0.0
    samples: list[float] = []
    for _ in range(max(1, repeats)):
        try:
            _proc, wall = run_command(
                wrap_time(["/usr/bin/true"], time_cmd), runner=runner, timeout=30.0
            )
        except BenchShapeError:  # pragma: no cover - экзотика без /usr/bin/true
            return 0.0
        samples.append(wall)
    samples.sort()
    return samples[len(samples) // 2]


# ---------------------------------------------------------------------------
# Замер одной конфигурации
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MeasureConfig:
    """Что именно меряем: бэкенд × потоки × mmap."""

    threads: int = 1
    mmap: bool = True
    backend: str | None = None  # None — не трогать VTM_BACKEND (по умолчанию auto)
    prompt_tokens: int = 1
    max_new: int = 4

    def label(self) -> str:
        return (
            f"{self.backend or 'auto'}/thr{self.threads}/"
            f"{'mmap' if self.mmap else 'no-mmap'}"
        )

    def env(self, base: Mapping[str, str] | None = None) -> dict[str, str]:
        env = dict(os.environ if base is None else base)
        env["VTM_THREADS"] = str(int(self.threads))
        if self.mmap:
            env.pop("VTM_NO_MMAP", None)
        else:
            env["VTM_NO_MMAP"] = "1"
        if self.backend:
            env["VTM_BACKEND"] = str(self.backend)
        else:
            env.pop("VTM_BACKEND", None)
        return env


def prompt_token_ids(shape: Shape, count: int, seed: int = 7) -> list[int]:
    """Детерминированные id промпта в пределах словаря."""
    import random

    rng = random.Random(int(seed))
    return [rng.randrange(shape.vocab) for _ in range(max(1, int(count)))]


def measure_config(
    vtm_bin: str,
    model: str,
    config: MeasureConfig,
    shape: Shape,
    *,
    runner: Runner | None = None,
    time_cmd: Sequence[str] | None = None,
    overhead_s: float = 0.0,
    timeout: float | None = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Замер: загрузка → TTFT (один проход) → декод (N проходов).

    ``stats`` рантайма считает только реальные ``vtm_run`` (таймер стартует
    после загрузки): прогон с ``--max-new 1`` — это ровно первый токен (TTFT при
    промпте в 1 токен), а декод = ``(T_N − T_1)/(N − 1)``. Загрузка меряется
    отдельно по ``vtm inspect``.
    """
    env = config.env()
    tokens = prompt_token_ids(shape, config.prompt_tokens, seed=shape.seed)

    inspect_proc, inspect_wall = run_command(
        wrap_time(build_inspect_command(vtm_bin, model), time_cmd),
        runner=runner, timeout=timeout, env=env,
    )
    inspect_time = parse_time_output(inspect_proc.stderr or "")
    load_s = max(0.0, inspect_wall - overhead_s)
    load_rss = inspect_time.get("max_rss_bytes")
    header = parse_run_header((inspect_proc.stdout or "") + (inspect_proc.stderr or ""))

    first_proc, first_wall = run_command(
        wrap_time(build_run_command(vtm_bin, model, tokens, 1), time_cmd),
        runner=runner, timeout=timeout, env=env,
    )
    first_stats = parse_stats_output(first_proc.stderr or "")
    first_time = parse_time_output(first_proc.stderr or "")
    ttft_s = first_stats.seconds if first_stats else max(0.0, first_wall - load_s - overhead_s)
    peak_rss = first_time.get("max_rss_bytes") or load_rss

    decode_s: float | None = None
    decode_tok_s: float | None = None
    total_s: float | None = None
    stats_s = ttft_s
    if config.max_new > 1:
        many_proc, many_wall = run_command(
            wrap_time(
                build_run_command(vtm_bin, model, tokens, config.max_new), time_cmd
            ),
            runner=runner, timeout=timeout, env=env,
        )
        many_stats = parse_stats_output(many_proc.stderr or "")
        many_time = parse_time_output(many_proc.stderr or "")
        total_s = many_stats.seconds if many_stats else many_wall
        stats_s = total_s
        # На крошечной форме разница T_N − T_1 может утонуть в разрешении
        # таймера; тогда декода честно нет, а не «миллиард ток/с».
        delta = total_s - ttft_s
        if delta > 0:
            decode_s = delta / (config.max_new - 1)
            decode_tok_s = 1.0 / decode_s
        peak_rss = many_time.get("max_rss_bytes") or peak_rss

    return {
        "backend": header.get("backend") or config.backend,
        "backend_requested": config.backend,
        "threads": config.threads,
        "mmap": config.mmap,
        "label": config.label(),
        "prompt_tokens": len(tokens),
        "max_new": config.max_new,
        "tensors": header.get("tensors"),
        "instructions": header.get("instructions"),
        "load_s": load_s,
        "load_rss_mib": _mib(load_rss) if load_rss else None,
        "ttft_s": ttft_s,
        "decode_s_per_token": decode_s,
        "decode_tok_s": decode_tok_s,
        "generation_s": stats_s,
        "peak_rss_mib": _mib(peak_rss) if peak_rss else None,
    }


# ---------------------------------------------------------------------------
# Отчёт
# ---------------------------------------------------------------------------


def _mib(value: Any) -> float | None:
    """Байты → МиБ (одна единица во всём отчёте, без «МБ по-разному»)."""
    if value is None:
        return None
    return round(float(value) / (1024.0 * 1024.0), 1)


def pick_best(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """Строка с лучшим декодом (для вердикта по ток/с и RSS)."""
    best = None
    for row in rows:
        value = row.get("decode_tok_s")
        if value is None:
            continue
        if best is None or value > best["decode_tok_s"]:
            best = row
    return best


def verdict(expected: float | None, observed: float | None, *,
            tolerance: float = 0.25, unit: str = "") -> dict[str, Any] | None:
    """Сравнить замер с ожиданием: подтверждена / опровергнута и во сколько раз."""
    if expected is None or observed is None or expected <= 0:
        return None
    ratio = observed / expected
    if 1.0 - tolerance <= ratio <= 1.0 + tolerance:
        text = "подтверждена"
    elif ratio < 1.0:
        text = f"опровергнута: в {1.0 / ratio:.1f}× ниже"
    else:
        text = f"опровергнута: в {ratio:.1f}× выше"
    return {
        "expected": expected,
        "observed": observed,
        "ratio": round(ratio, 3),
        "tolerance": tolerance,
        "verdict": text,
        "unit": unit,
    }


def build_report(
    plan: ContainerPlan,
    rows: Sequence[Mapping[str, Any]],
    *,
    container: Mapping[str, Any],
    expect_tok_s: float | None = 30.0,
    expect_rss_mib: float | None = 450.0,
    notes: Sequence[str] = (),
) -> dict[str, Any]:
    best = pick_best(rows)
    tok_s = best.get("decode_tok_s") if best else None
    rss = best.get("peak_rss_mib") if best else None
    scalar_rows = [row for row in rows if row.get("backend_requested") == "scalar"]
    note_list = list(notes)
    comparable = plan.trit_params >= VERDICT_MIN_TRITS
    if not comparable:
        note_list.append(
            "форма меньше {limit} млн тритов: абсолютные оценки 30 ток/с и "
            "450 МиБ к ней не применяются (вердикт не считается)".format(
                limit=VERDICT_MIN_TRITS // 1_000_000
            )
        )
    return {
        "shape": plan.shape.as_dict(),
        "container": dict(container),
        "measurements": [dict(row) for row in rows],
        "verdict": {
            "tok_s": verdict(expect_tok_s, tok_s, unit="ток/с") if comparable else None,
            "rss": verdict(expect_rss_mib, rss, tolerance=0.15, unit="МиБ")
            if comparable else None,
            "best_label": best.get("label") if best else None,
            "scalar_ref": pick_best(scalar_rows).get("decode_tok_s")
            if scalar_rows else None,
        },
        "notes": note_list,
    }


def _fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def format_measurements(rows: Sequence[Mapping[str, Any]]) -> str:
    """Таблица замеров: бэкенд, mmap, потоки, загрузка, RSS, TTFT, декод."""
    header = ["бэкенд", "mmap", "пот.", "загрузка,с", "RSS,МиБ", "TTFT,с",
              "декод,ток/с", "пик RSS,МиБ"]
    body: list[list[str]] = []
    for row in rows:
        body.append([
            str(row.get("backend") or "?"),
            "да" if row.get("mmap") else "нет",
            str(row.get("threads", "")),
            _fmt(row.get("load_s"), 3),
            _fmt(row.get("load_rss_mib")),
            _fmt(row.get("ttft_s"), 3),
            _fmt(row.get("decode_tok_s"), 2),
            _fmt(row.get("peak_rss_mib")),
        ])
    widths = [len(cell) for cell in header]
    for line in body:
        widths = [max(width, len(cell)) for width, cell in zip(widths, line)]
    # Числовые колонки — по правому краю, текст — по левому.
    right = {3, 4, 5, 6, 7}

    def render(cells: Sequence[str]) -> str:
        parts = []
        for index, (width, cell) in enumerate(zip(widths, cells)):
            parts.append(cell.rjust(width) if index in right else cell.ljust(width))
        return "  ".join(parts).rstrip()

    lines = [render(header), render(["-" * width for width in widths])]
    lines += [render(line) for line in body]
    return "\n".join(lines)


def container_lines(container: Mapping[str, Any]) -> list[str]:
    """Строки «контейнер: …» — общие для сухого прогона и полного отчёта."""
    shape = container.get("shape") or {}
    lines = []
    if shape:
        lines.append(
            "форма: hidden={hidden}, слоёв={layers}, голов={heads}/{kv_heads}, "
            "head_dim={head_dim}, intermediate={intermediate}, vocab={vocab}, "
            "ctx={ctx}".format(**shape)
        )
    lines.append(
        (
            "контейнер: {mib} МиБ ({bytes} байт), тензоров {tensors}, "
            "инструкций {instructions}, тритов {trit_params}"
        ).format(
            mib=_fmt(container.get("file_mib"), 1),
            bytes=container.get("file_bytes"),
            tensors=container.get("tensors"),
            instructions=container.get("instructions"),
            trit_params=f"{int(container.get('trit_params') or 0):,}".replace(",", " "),
        )
    )
    lines.append(
        (
            "  триты {trit} МиБ, масштабы {scale} МиБ, fp32 {fp32} МиБ, "
            "KV-кэш {kv} МиБ"
        ).format(
            trit=_fmt(container.get("trit_mib")),
            scale=_fmt(container.get("scale_mib")),
            fp32=_fmt(container.get("fp32_mib")),
            kv=_fmt(container.get("kv_cache_mib")),
        )
    )
    if container.get("build_s") is not None:
        lines.append(
            "  сборка графа: {build} с, пик RSS сборки {rss} МиБ".format(
                build=_fmt(container.get("build_s")),
                rss=_fmt(container.get("build_rss_mib"), 0),
            )
        )
    if container.get("write_s") is not None:
        lines.append(f"  запись на диск: {_fmt(container.get('write_s'))} с")
    return lines


def format_plan(plan: ContainerPlan, *, build_s: float | None = None,
                build_rss_mib: float | None = None,
                write_s: float | None = None) -> str:
    """Текст «что будет собрано» — для ``--dry-run``/``--build-only``."""
    container = dict(plan.summary())
    container["build_s"] = build_s
    container["build_rss_mib"] = build_rss_mib
    container["write_s"] = write_s
    return "\n".join(container_lines(container))


def format_report(report: Mapping[str, Any]) -> str:
    """Текст отчёта: форма, контейнер, таблица замеров, вердикт."""
    lines = container_lines(report["container"])
    lines.append("")
    lines.append(format_measurements(report["measurements"]))
    verdicts = report.get("verdict") or {}
    tok = verdicts.get("tok_s")
    rss = verdicts.get("rss")
    lines.append("")
    if tok:
        lines.append(
            "оценка {expected:g} ток/с: лучший замер {observed} ток/с "
            "({label}) — {verdict}".format(
                expected=tok["expected"], observed=_fmt(tok["observed"], 2),
                label=verdicts.get("best_label"), verdict=tok["verdict"],
            )
        )
    if rss:
        lines.append(
            "оценка {expected:g} МиБ RSS: пик {observed} МиБ — {verdict}".format(
                expected=rss["expected"], observed=_fmt(rss["observed"]),
                verdict=rss["verdict"],
            )
        )
    if verdicts.get("scalar_ref") is not None:
        lines.append(
            "скалярный бэкенд (лучший): {value} ток/с".format(
                value=_fmt(verdicts["scalar_ref"], 2)
            )
        )
    for note in report.get("notes") or ():
        lines.append(f"примечание: {note}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _int_list(text: str) -> list[int]:
    values: list[int] = []
    for part in str(text).replace(" ", "").split(","):
        if not part:
            continue
        try:
            value = int(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"ожидалось целое: {part!r}") from exc
        if value <= 0:
            raise argparse.ArgumentTypeError(f"значение должно быть > 0: {value}")
        values.append(value)
    if not values:
        raise argparse.ArgumentTypeError("пустой список")
    return values


def _backend_list(text: str) -> list[str | None]:
    values: list[str | None] = []
    for part in str(text).replace(" ", "").split(","):
        if not part:
            continue
        if part in ("auto", "default", ""):
            values.append(None)
        elif part in ("scalar", "lut", "simd", "direct", "neon", "avx2"):
            values.append(part)
        else:
            raise argparse.ArgumentTypeError(f"неизвестный бэкенд: {part!r}")
    if not values:
        values.append(None)
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bench_shape.py",
        description=(
            "Синтетический .vtm формы Qwen2.5-1.5B (случайные триты) и замеры: "
            "размер, RSS, загрузка, TTFT, декод."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Примеры:\n"
            "  python3 tools/bench_shape.py --dry-run\n"
            "  python3 tools/bench_shape.py --json > runs/bench_shape.json\n"
            "  python3 tools/bench_shape.py --hidden 512 --layers 4 --vocab 8192 --ctx 512\n"
            "  python3 tools/bench_shape.py --threads 1,8 --mmap both --backends auto,scalar\n"
        ),
    )
    parser.add_argument("--hidden", type=int, default=DEFAULT_SHAPE.hidden)
    parser.add_argument("--layers", type=int, default=DEFAULT_SHAPE.layers)
    parser.add_argument("--heads", type=int, default=DEFAULT_SHAPE.heads)
    parser.add_argument("--kv-heads", type=int, default=DEFAULT_SHAPE.kv_heads)
    parser.add_argument("--head-dim", type=int, default=DEFAULT_SHAPE.head_dim)
    parser.add_argument("--intermediate", type=int, default=DEFAULT_SHAPE.intermediate)
    parser.add_argument("--vocab", type=int, default=DEFAULT_SHAPE.vocab)
    parser.add_argument("--ctx", type=int, default=DEFAULT_SHAPE.ctx)
    parser.add_argument("--seed", type=int, default=DEFAULT_SHAPE.seed,
                        help="seed случайных тритов и id промпта (7)")
    parser.add_argument("--trit-scale", type=float, default=DEFAULT_SHAPE.trit_scale,
                        help="константа масштаба блока 128 (0.02)")
    parser.add_argument("--reduced", action="store_true",
                        help="уменьшенная форма (1/2 hidden, 1/4 слоёв/vocab/ctx)")
    parser.add_argument("--out", default=None,
                        help="путь .vtm (по умолчанию runs/bench_shape/shape_<форма>.vtm)")
    parser.add_argument("--reuse", action="store_true",
                        help="не собирать заново, если --out уже есть")
    parser.add_argument("--dry-run", action="store_true",
                        help="только оценка размера, без записи и замеров")
    parser.add_argument("--build-only", action="store_true",
                        help="собрать контейнер и остановиться")
    parser.add_argument("--threads", type=_int_list, default=[1, 4, 8],
                        help="список VTM_THREADS (1,4,8)")
    parser.add_argument("--mmap", choices=("both", "yes", "no"), default="both",
                        help="mmap/без mmap (VTM_NO_MMAP=1)")
    parser.add_argument("--backends", type=_backend_list, default=[None],
                        help="auto,scalar,simd,... (по умолчанию auto)")
    parser.add_argument("--prompt-tokens", type=int, default=1,
                        help="длина промпта в токенах (1 — чистый первый проход)")
    parser.add_argument("--max-new", type=int, default=4,
                        help="сколько токенов генерировать для декода (4)")
    parser.add_argument("--repeats", type=int, default=2,
                        help="прогонов на конфигурацию, берётся лучший декод (2)")
    parser.add_argument("--vtm-bin", default=os.path.join(ROOT, "src", "vtm", "vtm"),
                        help="путь к собранному vtm")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--expect-tok-s", type=float, default=30.0)
    parser.add_argument("--expect-rss-mib", type=float, default=450.0)
    parser.add_argument("--json", action="store_true", help="отчёт в JSON")
    return parser


def shape_from_args(args: argparse.Namespace) -> Shape:
    shape = Shape(
        hidden=args.hidden,
        layers=args.layers,
        heads=args.heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        intermediate=args.intermediate,
        vocab=args.vocab,
        ctx=args.ctx,
        seed=args.seed,
        trit_scale=args.trit_scale,
    )
    return shape.reduced() if args.reduced else shape


def default_out_path(shape: Shape) -> str:
    return os.path.join(ROOT, "runs", "bench_shape", f"shape_{shape.slug()}.vtm")


def measure_config_list(
    vtm_bin: str,
    model: str,
    shape: Shape,
    *,
    threads: Iterable[int],
    mmap_modes: Iterable[bool],
    backends: Iterable[str | None],
    prompt_tokens: int,
    max_new: int,
    runner: Runner | None = None,
    time_cmd: Sequence[str] | None = None,
    overhead_s: float = 0.0,
    timeout: float | None = DEFAULT_TIMEOUT,
    repeats: int = 2,
    progress: Callable[[MeasureConfig], None] | None = None,
) -> list[dict[str, Any]]:
    """Полная матрица «бэкенд × mmap × потоки».

    ``repeats`` — сколько раз гонять конфигурацию; берётся лучший декод (шум
    чужой нагрузки на машине легко даёт 2×, а «лучший» — устойчивая нижняя
    оценка времени на токен). Все замеры конфигурации кладутся в
    ``decode_tok_s_samples``, чтобы разброс был виден в отчёте.
    """
    repeats = max(1, int(repeats))
    rows: list[dict[str, Any]] = []
    for backend in backends:
        for use_mmap in mmap_modes:
            for thread_count in threads:
                config = MeasureConfig(
                    threads=int(thread_count),
                    mmap=bool(use_mmap),
                    backend=backend,
                    prompt_tokens=int(prompt_tokens),
                    max_new=int(max_new),
                )
                if progress is not None:
                    progress(config)
                samples = [
                    measure_config(
                        vtm_bin, model, config, shape, runner=runner,
                        time_cmd=time_cmd, overhead_s=overhead_s, timeout=timeout,
                    )
                    for _ in range(repeats)
                ]
                row = max(
                    samples,
                    key=lambda item: -1.0 if item.get("decode_tok_s") is None
                    else item["decode_tok_s"],
                )
                row["repeats"] = repeats
                row["decode_tok_s_samples"] = [
                    sample.get("decode_tok_s") for sample in samples
                ]
                row["load_s_samples"] = [sample.get("load_s") for sample in samples]
                rows.append(row)
    return rows


def _process_max_rss_bytes() -> int | None:
    if resource is None:  # pragma: no cover
        return None
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _resolve_vtm_bin(path: str) -> str:
    if os.path.exists(path) and os.access(path, os.X_OK):
        return path
    raise BenchShapeError(
        f"нет исполняемого {path}: соберите `make -C src/vtm vtm` "
        "или укажите --vtm-bin"
    )


def _warning(shape: Shape, plan: ContainerPlan, *, out: str) -> None:
    """Предупреждение о тяжёлой генерации + уменьшенная форма."""
    small = shape.reduced()
    counts = shape_param_counts(small)
    print(
        "bench_shape: генерация контейнера — {mib} МиБ, {trits} тритов; "
        "это займёт секунды и до ~1 ГиБ RAM на пике.\n"
        "  файл: {out}\n"
        "  если это слишком много: --reduced (hidden={rh}, слоёв={rl}, "
        "vocab={rv}, ctx={rc}, ~{rt} млн тритов) — та же архитектура, "
        "числа будут другого порядка.".format(
            mib=_fmt(plan.file_size / (1024.0 * 1024.0), 1),
            trits=f"{plan.trit_params:,}".replace(",", " "),
            out=out,
            rh=small.hidden, rl=small.layers, rv=small.vocab, rc=small.ctx,
            rt=f"{counts['trit_params'] / 1e6:.0f}",
        ),
        file=sys.stderr,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: Runner | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    if args.max_new < 2 and not (args.dry_run or args.build_only):
        print(
            "bench_shape: для декода нужен --max-new >= 2 "
            "(TTFT меряется прогоном с --max-new 1)",
            file=sys.stderr,
        )
        return 2
    try:
        shape = shape_from_args(args)
    except BenchShapeError as exc:
        print(f"bench_shape: {exc}", file=sys.stderr)
        return 2

    out = args.out or default_out_path(shape)
    started = time.perf_counter()
    plan = build_plan(shape)
    build_s = time.perf_counter() - started
    build_rss = _process_max_rss_bytes()

    if plan.file_size > 100 * 1024 * 1024:
        _warning(shape, plan, out=out)
    if args.dry_run:
        if args.json:
            print(json.dumps({
                "shape": shape.as_dict(),
                "container": {**plan.summary(), "build_s": round(build_s, 2),
                              "build_rss_mib": _mib(build_rss)},
                "dry_run": True,
            }, ensure_ascii=False, indent=2))
        else:
            print(format_plan(plan, build_s=round(build_s, 2),
                              build_rss_mib=_mib(build_rss)))
        return 0

    write_s: float | None = None
    if args.reuse and os.path.exists(out):
        size = os.path.getsize(out)
        if size != plan.file_size:
            print(
                f"bench_shape: {out} ({size} байт) не совпадает по размеру с "
                f"формой ({plan.file_size} байт) — пересобираю",
                file=sys.stderr,
            )
            write_started = time.perf_counter()
            size = write_container(out, plan)
            write_s = time.perf_counter() - write_started
    else:
        write_started = time.perf_counter()
        size = write_container(out, plan)
        write_s = time.perf_counter() - write_started
        print(
            f"bench_shape: записан {out} ({_fmt(size / (1024.0 * 1024.0), 1)} МиБ) "
            f"за {write_s:.1f} с",
            file=sys.stderr,
        )

    container = {
        **plan.summary(),
        "path": out,
        "file_bytes": size,
        "file_mib": _mib(size),
        "build_s": round(build_s, 2),
        "build_rss_mib": _mib(build_rss),
        "write_s": round(write_s, 2) if write_s is not None else None,
    }
    if args.build_only:
        if args.json:
            print(json.dumps({"shape": shape.as_dict(), "container": container},
                             ensure_ascii=False, indent=2))
        else:
            print(format_plan(plan, build_s=container["build_s"],
                              build_rss_mib=container["build_rss_mib"],
                              write_s=container["write_s"]))
        return 0

    mmap_modes = {
        "both": [True, False], "yes": [True], "no": [False],
    }[args.mmap]
    try:
        vtm_bin = _resolve_vtm_bin(args.vtm_bin)
        time_cmd = time_utility_command()
        overhead_s = calibrate_time_overhead(runner=runner, time_cmd=time_cmd)
        rows = measure_config_list(
            vtm_bin, out, shape,
            threads=args.threads,
            mmap_modes=mmap_modes,
            backends=args.backends,
            prompt_tokens=args.prompt_tokens,
            max_new=args.max_new,
            repeats=args.repeats,
            runner=runner,
            time_cmd=time_cmd,
            overhead_s=overhead_s,
            timeout=args.timeout,
            progress=None if args.json else (
                lambda config: print(f"bench_shape: {config.label()} …", file=sys.stderr)
            ),
        )
    except BenchShapeError as exc:
        print(f"bench_shape: {exc}", file=sys.stderr)
        return 1

    notes = [
        "веса случайные (триты ±1, масштаб {scale:g}), обучения не было: "
        "на скорость и размер это не влияет".format(scale=shape.trit_scale),
        "TTFT — один проход (промпт {n} токен), без батчевого prefill: "
        "промпт из P токенов даёт ~P таких проходов".format(n=args.prompt_tokens),
        "обёртка /usr/bin/time добавляет ~{over:.0f} мс, они вычтены из загрузки".format(
            over=overhead_s * 1000.0),
        "каждая строка — лучший из {repeats} прогонов (машина занята другими "
        "агентами, разброс декода виден в decode_tok_s_samples)".format(
            repeats=args.repeats),
    ]
    report = build_report(
        plan, rows, container=container,
        expect_tok_s=args.expect_tok_s, expect_rss_mib=args.expect_rss_mib,
        notes=notes,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(format_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
