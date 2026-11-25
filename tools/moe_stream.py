#!/usr/bin/env python3
"""Потоковый импорт MoE GGUF -> `.vtm`: без `q4.bin` и без второй копии.

Зачем
=====

Полный Qwen3-30B-A3B Q4_K_M — это GGUF 18,5 ГБ. Прежний путь
(`tools/moe_import.py` -> `q4.bin` 18,9 ГБ -> `tools/vtm_compile.py` ->
`.vtm` 18,9 ГБ) требует 56 ГБ диска одновременно, а на рабочей машине их 12.
При этом `q4.bin` — чистая копия блоков GGUF: те же 18,9 ГБ, прочитанные и
записанные только для того, чтобы компилятор прочитал их ещё раз.

Здесь один проход:

    GGUF (локальный файл или HTTP Range)
        -> таблица тензоров и план (как в moe_import)
        -> каркас контейнера через vtm_ref.write_vtm_stream
        -> payload'ы сегментов пишутся в секцию данных на свои смещения
        -> .part переименовывается в .vtm

Промежуточных файлов нет: единственный временный файл — префикс GGUF
(метаданные + таблица тензоров, 16 МБ у Qwen3) для HTTP-источника, и он
удаляется до начала записи контейнера. Поэтому пик диска = размер контейнера
(при `--keep-temp` — плюс этот префикс).

Почему через `write_vtm_stream`
===============================

Раскладка контейнера (заголовок, таблица, выравнивание, мета, программа,
выбор v0.2/v0.3 по размеру) обязана совпадать с непоточным путём байт-в-байт.
Поэтому её считает тот же писатель, что и раньше: `write_vtm_stream` пишет
каркас с нулевыми payload'ами, после чего веса записываются на свои
смещения через `pwrite`. Нули в секции данных — это ровно те байты, которые
`write_vtm` и так пишет для активаций (они инициализированы нулями), а
payload'ы весов перезаписываются целиком. Итог: те же байты, что у
`write_vtm`, но данные ни разу не материализуются в памяти.

Цена решения — нулевой проход по секции данных (запись файла дважды: каркас,
затем веса). На срезе 4 слоя это ~2 ГБ лишней записи; выигрыш — 18,9 ГБ
`q4.bin`, без которых полная модель вообще не собирается на 12 ГиБ.

Память
======

Тензоры читаются кусками по `--chunk-bytes` (4 МиБ): 1 МиБ значений fp32
дают ~32 МБ пика на кусок, банки Q4_K/Q6_K копируются блоками как есть.
Пик RSS не зависит от размера модели (замер — в отчёте задачи 0174).

Проверка
========

* `--verify` (по умолчанию) перечитывает payload'ы сегментов из готового
  контейнера и сверяет sha256 с тем, что записано;
* `--compare REF.vtm` сравнивает сегменты и программу с контейнером,
  собранным непоточным путём (`moe_import` + `vtm_compile`), — таблица
  тензоров, смещения, payload'ы и программа обязаны совпасть побайтово.

CLI
===

    python3 tools/moe_stream.py --gguf /tmp/moe/slice4.gguf --out /tmp/moe/stream4.vtm
    python3 tools/moe_stream.py --gguf <url> --out /tmp/moe/full.vtm --limit-layers 4
    python3 tools/moe_stream.py --gguf <url> --out /tmp/moe/full.vtm --dry-run

`--dry-run` строит план и каркас в памяти (данные не читаются и не пишутся) и
печатает ожидаемый размер контейнера: так оценивается полная 30B-модель, даже
когда диска на неё нет.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _path in (_ROOT, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Импорт как пакета (`tests/*`, `python3 -m tools.moe_stream`) и как скрипта
# (`python3 tools/moe_stream.py`) — ровно как в `vtm_compile`: важно, чтобы
# `vtm_ref` был ОДНИМ модулем. Иначе `vtm_compile` собрал бы граф из объектов
# `tools.vtm_ref.Tensor`, а мы держали бы `vtm_ref.Tensor` — два разных класса
# с одинаковыми именами, и `isinstance` в компиляторе молча не срабатывает.
try:  # pragma: no cover - выбор ветки зависит от способа запуска
    from tools import gguf_import as G
    from tools import moe_import as MI
    from tools import moe_slice as MS
    from tools import vtm_compile as VC
    from tools import vtm_ref as VR
except ImportError:  # pragma: no cover
    import gguf_import as G
    import moe_import as MI
    import moe_slice as MS
    import vtm_compile as VC
    import vtm_ref as VR

#: Кусок чтения/записи по умолчанию. 4 МиБ: ~32 МБ пика на fp32-кусок и
#: достаточно крупные pwrite, чтобы не тонуть в syscall'ах.
CHUNK_BYTES = 4 << 20
#: У HTTP каждый Range — отдельное соединение (TLS handshake на запрос),
#: поэтому кусок крупнее: 32 МиБ вместо 4.
HTTP_CHUNK_BYTES = 32 << 20
USER_AGENT = "eva32-moe-stream/1"

#: Поля меты, которые у потокового пути отличаются по построению (у него нет
#: `manifest.json`): при сравнении с непоточным контейнером они исключаются.
VOLATILE_META_KEYS = frozenset(
    {"source_manifest", "source_gguf", "stream_import", "generator"}
)

_KIND_BY_TARGET = {
    "q4k": VR.KIND_Q4K,
    "q6k": VR.KIND_Q6K,
    "f32": VR.KIND_F32,
}


class StreamImportError(ValueError):
    """GGUF/план/контейнер не сходятся — текст для человека."""


class SourceError(RuntimeError):
    """Источник GGUF недоступен (файл или HTTP Range)."""


# ---------------------------------------------------------------------------
# Источник байтов GGUF: локальный файл или HTTP Range
# ---------------------------------------------------------------------------


class GgufSource:
    """Абстракция «откуда читать байты GGUF».

    Нужна, чтобы план и запись контейнера не знали, откуда приходят блоки:
    локальный файл читается `pread`, удалённый — HTTP Range (как в
    `tools/moe_slice.py`). Ни один метод не отдаёт больше `chunk` байт, кроме
    явного `read`, который вызывается только для префикса заголовка.
    """

    label: str = "?"
    size: int = 0
    chunk: int = CHUNK_BYTES

    def read(self, start: int, length: int) -> bytes:
        raise NotImplementedError

    def chunk_for(self, requested: int | None) -> int:
        """Кусок чтения: явный (если задан) или собственный по умолчанию."""
        return int(requested) if requested else int(self.chunk)

    def iter_range(
        self, start: int, length: int, chunk: int | None = None
    ) -> Iterator[bytes]:
        step = max(1, int(chunk or self.chunk))
        done = 0
        while done < length:
            take = min(step, length - done)
            piece = self.read(start + done, take)
            if len(piece) != take:
                raise SourceError(
                    f"{self.label}: диапазон {start + done}+{take} дал "
                    f"{len(piece)} байт"
                )
            yield piece
            done += take

    def close(self) -> None:  # pragma: no cover - переопределяется
        pass

    def describe(self) -> dict[str, Any]:
        return {"label": self.label, "bytes": self.size, "kind": "?"}


class LocalSource(GgufSource):
    """Локальный GGUF: `pread` по открытому fd, без mmap и без копии."""

    def __init__(self, path: str | os.PathLike[str],
                 *, chunk: int = CHUNK_BYTES) -> None:
        self.path = Path(path)
        try:
            self._fd = os.open(self.path, os.O_RDONLY)
        except OSError as exc:
            raise SourceError(f"не открывается {self.path}: {exc}") from exc
        self.size = os.fstat(self._fd).st_size
        self.chunk = max(1, int(chunk))
        self.label = str(self.path)

    def read(self, start: int, length: int) -> bytes:
        if start < 0 or length < 0 or start + length > self.size:
            raise SourceError(
                f"{self.label}: диапазон {start}+{length} за файлом "
                f"({self.size} байт)"
            )
        first = os.pread(self._fd, length, start)
        if len(first) == length:
            return first
        pieces = [first]
        done = len(first)
        while done < length:
            piece = os.pread(self._fd, length - done, start + done)
            if not piece:
                raise SourceError(
                    f"{self.label}: файл обрезан на байте {start + done}"
                )
            pieces.append(piece)
            done += len(piece)
        return b"".join(pieces)

    def close(self) -> None:
        fd = getattr(self, "_fd", None)
        if fd is not None:
            self._fd = None  # type: ignore[assignment]
            os.close(fd)

    def describe(self) -> dict[str, Any]:
        return {"label": self.label, "bytes": self.size, "kind": "local"}


class HttpSource(GgufSource):
    """GGUF по HTTP Range: ничего не скачивается целиком.

    Диапазоны берутся тем же `moe_slice._fetch`, что и у среза (повторы,
    таймаут, проверка длины), чтобы поведение при обрывах было одинаковым.
    """

    def __init__(self, url: str, *, chunk: int = HTTP_CHUNK_BYTES,
                 timeout: float = 120.0) -> None:
        self.url = url
        self.label = _short_url(url)
        self.size = _http_content_length(url, timeout=timeout)
        self.chunk = max(1, int(chunk))
        self.timeout = timeout

    def read(self, start: int, length: int) -> bytes:
        if start < 0 or length < 0 or start + length > self.size:
            raise SourceError(
                f"{self.label}: диапазон {start}+{length} за файлом "
                f"({self.size} байт)"
            )
        try:
            return MS._fetch(self.url, start, start + length - 1)
        except MS.SliceError as exc:
            raise SourceError(f"{self.label}: {exc}") from exc

    def chunk_for(self, requested: int | None) -> int:
        # HTTP: каждый Range — отдельное соединение, поэтому крупный кусок
        # выгоднее, чем переданный лимит по умолчанию (4 МиБ).
        if requested:
            return max(int(requested), int(self.chunk))
        return int(self.chunk)

    def describe(self) -> dict[str, Any]:
        return {"label": self.label, "bytes": self.size, "kind": "http",
                "url": self.url}


def _short_url(url: str) -> str:
    """URL без query (там подписанный токен CDN, в отчёте он не нужен)."""
    base = url.split("?", 1)[0]
    return base if len(base) <= 200 else base[:197] + "…"


def _http_content_length(url: str, *, timeout: float = 120.0) -> int:
    """Размер GGUF по HTTP: HEAD, при отказе — Range 0-0 и Content-Range."""
    request = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": USER_AGENT}
    )
    header = ""
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            header = response.headers.get("Content-Length") or ""
            if header.isdigit():
                return int(header)
            content_range = response.headers.get("Content-Range") or ""
            size = _content_range_size(content_range)
            if size is not None:
                return size
    except (urllib.error.URLError, OSError) as exc:
        # HEAD поддерживают не все зеркала — ниже пробуем Range.
        header = f"{type(exc).__name__}: {exc}"
    request = urllib.request.Request(
        url, headers={"Range": "bytes=0-0", "User-Agent": USER_AGENT}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            size = _content_range_size(
                response.headers.get("Content-Range") or ""
            )
            if size is not None:
                return size
            header = response.headers.get("Content-Length") or header
    except (urllib.error.URLError, OSError) as exc:
        raise SourceError(
            f"не удалось узнать размер {_short_url(url)}: {exc}"
        ) from exc
    raise SourceError(
        f"сервер не сообщил размер {_short_url(url)} (Content-Length={header!r})"
    )


def _content_range_size(value: str) -> int | None:
    """Размер файла из `Content-Range: bytes 0-0/18556685824`."""
    if "/" not in value:
        return None
    tail = value.rsplit("/", 1)[1].strip()
    return int(tail) if tail.isdigit() else None


def open_source(
    spec: str | os.PathLike[str], *, chunk: int | None = None
) -> GgufSource:
    """Открыть GGUF: `http(s)://` — HTTP Range, иначе локальный файл."""
    text = str(spec)
    if text.startswith(("http://", "https://")):
        return HttpSource(text, chunk=HTTP_CHUNK_BYTES if chunk is None else chunk)
    return LocalSource(text, chunk=CHUNK_BYTES if chunk is None else chunk)


# ---------------------------------------------------------------------------
# Заголовок GGUF
# ---------------------------------------------------------------------------


def _check_tensors_fit(gguf: G.GgufFile, source: GgufSource) -> None:
    """Для удалённого GGUF: каждый тензор обязан влезать в файл.

    Локальный файл это проверяет `read_gguf(check_data=True)`, но у HTTP
    прочитан только префикс, поэтому сверяем с настоящим размером файла —
    иначе обрезанный/подменённый GGUF всплыл бы на середине записи.
    """
    for info in gguf.tensors:
        try:
            n_bytes = info.n_bytes
        except G.GgufError:
            continue  # неподдержанный тип проверит plan_moe
        if info.offset + n_bytes > source.size - gguf.data_offset:
            raise SourceError(
                f"{source.label}: тензор {info.name!r} выходит за файл "
                f"(offset {info.offset} + {n_bytes} > "
                f"{source.size - gguf.data_offset})"
            )


def load_gguf(
    source: GgufSource,
    work: Path,
    *,
    keep_temp: bool = False,
) -> tuple[G.GgufFile, Path | None, int]:
    """Прочитать метаданные и таблицу тензоров, не читая весов.

    Возвращает `(gguf, temp_path, temp_bytes)`. Для локального файла
    временных файлов нет. Для HTTP префикс кладётся во временный файл (его
    требует `gguf_import.read_gguf`) и удаляется сразу после разбора, если не
    `keep_temp`: к началу записи контейнера его на диске уже нет.
    """
    if isinstance(source, LocalSource):
        try:
            gguf = G.read_gguf(source.path)
        except G.GgufError as exc:
            raise StreamImportError(str(exc)) from exc
        return gguf, None, 0

    last_error: Exception | None = None
    temp_path: Path | None = None
    prefix_bytes = 0
    for size in MS.PREFIX_STEPS:
        want = min(size, source.size)
        prefix = source.read(0, want)
        prefix_bytes = len(prefix)
        fd, name = tempfile.mkstemp(prefix=".moe-head-", dir=work)
        temp_path = Path(name)
        with os.fdopen(fd, "wb") as handle:
            handle.write(prefix)
        try:
            gguf = G.read_gguf(temp_path, check_data=False)
        except G.GgufError as exc:
            last_error = exc
            if not keep_temp:
                temp_path.unlink(missing_ok=True)
                temp_path = None
            continue
        _check_tensors_fit(gguf, source)
        if not keep_temp:
            temp_path.unlink(missing_ok=True)
            temp_path = None
        return gguf, temp_path, prefix_bytes
    if temp_path is not None and not keep_temp:
        temp_path.unlink(missing_ok=True)
    raise StreamImportError(
        f"заголовок GGUF не разобрался за {prefix_bytes} байт: {last_error}"
    )


# ---------------------------------------------------------------------------
# План: какие тензоры откуда брать
# ---------------------------------------------------------------------------


@dataclass
class Job:
    """Один сегмент: план из GGUF, тензор в графе и окно в источнике."""

    plan: MI.Planned
    tensor: VR.Tensor
    kind: int
    n_values: int
    source_offset: int  # абсолютное смещение данных в GGUF
    n_bytes: int        # сколько байт источника читать
    payload_bytes: int  # сколько байт payload'а в контейнере
    #: sha256 payload'а (заголовок сегмента + блоки, или fp32-байты).
    payload_sha256: str = ""
    #: sha256 только блоков kind 6/7 — как `blocks_sha256` в манифесте
    #: `moe_import` (сверка с непоточным импортом без разбора контейнера).
    blocks_sha256: str = ""


def _placeholder_name(plan: MI.Planned) -> str:
    """Короткое имя (≤24 байт) для временного тензора графа.

    `compile_model` всё равно переименует его (`emb.w`, `lN.q_proj.w`,
    `lN.gate_bank`), но `Tensor.__post_init__` проверяет длину имени сразу,
    а HF-имена (`model.layers.0.self_attn.q_proj.weight`) в 24 байта не лезут.
    """
    if plan.layer is None:
        return {
            "embed": "emb.w",
            "final_norm": "final_norm.w",
            "lm_head": "lm_head.w",
        }[plan.role]
    return f"l{plan.layer}.{plan.key}.w"


def build_import_plan(
    cfg: MI.MoeConfig, plans: Sequence[MI.Planned], data_offset: int
) -> tuple[dict[str, Any], list[Job]]:
    """Собрать `weights` для `compile_model` и список сегментов для записи.

    Тензоры-заглушки (kind/dims без данных) — ровно те объекты, которые
    компилятор положит в граф. Позже по ним же находятся смещения в
    контейнере (`tensor.offset`), поэтому связь «сегмент GGUF ↔ запись
    таблицы» не теряется.
    """
    weights: dict[str, Any] = {"layers": [{} for _ in range(cfg.n_layers)]}
    jobs: list[Job] = []
    for plan in plans:
        kind = _KIND_BY_TARGET[plan.target]
        n_values = VR.prod(plan.shape)
        info = plan.info
        spec = info.spec
        if plan.target in ("q4k", "q6k"):
            if n_values % G.Q4K_BLOCK_VALUES:
                raise StreamImportError(
                    f"{info.name}: {n_values} значений не кратны блоку GGML "
                    f"{G.Q4K_BLOCK_VALUES}"
                )
            n_bytes = n_values // G.Q4K_BLOCK_VALUES * spec.block_bytes
            payload_bytes = VR.Q4K_HEADER_SIZE + n_bytes
        else:
            # fp32-роли: читаем ceil(block) значений, пишем ровно n_values*4.
            read_values = G._align_up(n_values, spec.block)
            n_bytes = read_values // spec.block * spec.block_bytes
            payload_bytes = n_values * 4
        tensor = VR.Tensor(name=_placeholder_name(plan), kind=kind,
                           dims=tuple(plan.shape))
        job = Job(
            plan=plan,
            tensor=tensor,
            kind=kind,
            n_values=n_values,
            source_offset=data_offset + info.offset,
            n_bytes=n_bytes,
            payload_bytes=payload_bytes,
        )
        jobs.append(job)
        if plan.role == "embed":
            weights["embed"] = tensor
        elif plan.role == "final_norm":
            weights["final_norm"] = tensor
        elif plan.role == "lm_head":
            weights["lm_head"] = tensor
        else:
            weights["layers"][plan.layer][plan.key] = tensor
    return weights, jobs


def _quantization_meta() -> dict[str, Any]:
    """Раздел quantization манифеста — тот же, что пишет `moe_import`."""
    return {
        "scheme": "ggml-native",
        "block_size": G.TRIT_SCALE_BLOCK,
        "ggml_block": G.Q4K_BLOCK_VALUES,
        "note": "Q4_K/Q6_K блоки скопированы байт-в-байт (ADR 0018)",
    }


def build_model(
    cfg: MI.MoeConfig,
    plans: Sequence[MI.Planned],
    gguf: G.GgufFile,
    source: GgufSource,
) -> tuple[VR.VTM, list[Job]]:
    """Собрать граф/программу компилятора на заглушках и подготовить сегменты.

    Конфиг для `compile_model` берётся из `architecture`, которую построил бы
    `moe_import` (`MoeConfig.as_architecture`), и разбирается тем же
    `vtm_compile._manifest_architecture`, что и непоточный путь: так
    `ModelConfig` (а значит и программа) совпадает по построению.
    """
    weights, jobs = build_import_plan(cfg, plans, gguf.data_offset)
    manifest = {
        "format": MI.MANIFEST_FORMAT,
        "format_version": MI.MANIFEST_VERSION,
        "architecture": cfg.as_architecture(),
        "quantization": _quantization_meta(),
        "base_model": {"name": cfg.name},
    }
    try:
        model_cfg = VC._manifest_architecture(manifest)  # noqa: SLF001
    except ValueError as exc:  # pragma: no cover - защита от рассинхрона
        raise StreamImportError(f"architecture не собралась: {exc}") from exc
    meta_extra = {
        # Непоточный путь пишет сюда путь к manifest.json; у потокового его
        # нет — ключ остаётся (схема меты одна), но со значением None.
        "source_manifest": None,
        "source_gguf": source.label,
        "stream_import": True,
        "base_model": {"name": cfg.name},
        "revision": None,
        "step": None,
        "checkpoint": None,
        "manifest_format": MI.MANIFEST_FORMAT,
        "manifest_version": MI.MANIFEST_VERSION,
        "quantization": _quantization_meta(),
    }
    try:
        model = VC.compile_model(model_cfg, weights, meta_extra)
    except ValueError as exc:
        raise StreamImportError(f"граф не собрался: {exc}") from exc

    # Активации графа создаются как fp32-нули и в контейнере обязаны остаться
    # нулями. `write_vtm_stream` отказывается писать тензор с данными (чтобы
    # нули не подменили веса молча), поэтому данные активаций снимаем: это те
    # же нули, но выраженные раскладкой, а не списком.
    weight_ids = {id(job.tensor) for job in jobs}
    for tensor in model.tensors:
        if id(tensor) in weight_ids or tensor.is_kv_cache:
            continue
        if tensor.kind != VR.KIND_F32 or tensor.data is None:
            raise StreamImportError(
                f"тензор {tensor.name!r}: не вес и не fp32-активация — "
                "граф изменился, потоковая запись не знает, что с ним делать"
            )
        if any(tensor.data):
            raise StreamImportError(
                f"тензор {tensor.name!r}: активация не нулевая — "
                "write_vtm_stream записал бы нули вместо значений"
            )
        tensor.data = None
    attached = {id(tensor) for tensor in model.tensors}
    lost = [job.plan.name for job in jobs if id(job.tensor) not in attached]
    if lost:
        raise StreamImportError(
            "веса не попали в граф: " + ", ".join(sorted(lost)[:5])
            + (" …" if len(lost) > 5 else "")
        )
    return model, jobs


# ---------------------------------------------------------------------------
# Запись payload'ов в секцию данных
# ---------------------------------------------------------------------------


def _pwrite_all(fd: int, data: Any, offset: int) -> int:
    """`os.pwrite` может записать меньше — дописываем; возвращаем длину."""
    view = memoryview(data)
    written = 0
    while written < len(view):
        chunk = os.pwrite(fd, view[written:], offset + written)
        if chunk <= 0:  # pragma: no cover - защита от странной ФС
            raise OSError(f"pwrite вернул {chunk} на смещении {offset + written}")
        written += chunk
    return written


def _fp32_payload_chunks(
    source: GgufSource, job: Job, chunk_values: int
) -> Iterator[bytes]:
    """Куски fp32-payload'а (little-endian) из GGUF-тензора.

    Значения деквантуются тем же `gguf_import._dequantize`, что и в
    `moe_import`, и пакуются как `struct.pack("<Nf", ...)` — но через
    `array('f')`: у 1 МиБ значений это те же байты без списка Python-float.
    """
    spec = job.plan.info.spec
    step = max(spec.block, (int(chunk_values) // spec.block) * spec.block)
    remaining = job.n_values
    raw_done = 0
    while remaining > 0:
        read_values = min(step, G._align_up(remaining, spec.block))
        raw_n = read_values // spec.block * spec.block_bytes
        raw = source.read(job.source_offset + raw_done, raw_n)
        if len(raw) != raw_n:  # pragma: no cover - источник это уже проверил
            raise SourceError(
                f"{job.plan.info.name}: прочитано {len(raw)} из {raw_n} байт"
            )
        raw_done += raw_n
        values = G._dequantize(spec, raw, read_values)
        if len(values) > remaining:
            values = values[:remaining]
        if not all(map(math.isfinite, values)):
            raise StreamImportError(
                f"{job.plan.info.name}: в тензоре есть nan/inf — "
                "контейнер с такими весами не читается (vtm_ref._check_finite)"
            )
        if sys.byteorder != "little":  # pragma: no cover - x86/arm64 little
            values = array("f", values)
            values.byteswap()
        yield values.tobytes()
        remaining -= len(values)


def fill_container(
    path: str | os.PathLike[str],
    model: VR.VTM,
    jobs: Sequence[Job],
    source: GgufSource,
    *,
    chunk_bytes: int = CHUNK_BYTES,
    quiet: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Записать веса в секцию данных каркаса (нули на смещениях тензоров).

    Каркас уже записан `write_vtm_stream`: заголовок, таблица, мета,
    программа и нулевые payload'ы на своих местах. Здесь каждый сегмент
    перезаписывается побайтово (`pwrite`), поэтому итоговый файл совпадает с
    тем, что записал бы `write_vtm` с теми же тензорами.
    """
    ordered = sorted(jobs, key=lambda job: job.tensor.offset)
    data_offset = model.data_offset
    chunk_values = max(1024, int(chunk_bytes) // 4)
    report: dict[str, Any] = {
        "segments": 0,
        "bytes": 0,
        "q4_bytes": 0,
        "fp32_bytes": 0,
        "seconds": 0.0,
    }
    started = time.time()
    fd = os.open(path, os.O_WRONLY)
    try:
        for job in ordered:
            name = job.plan.info.name
            dst = data_offset + job.tensor.offset
            payload_digest = hashlib.sha256()
            blocks_digest = hashlib.sha256()
            written = 0
            if job.kind in (VR.KIND_Q4K, VR.KIND_Q6K):
                header, n_blocks = VR.q4k_segment_header(job.kind, job.n_values)
                expected = VR.Q4K_HEADER_SIZE + (
                    n_blocks * (VR.Q4K_BLOCK_BYTES if job.kind == VR.KIND_Q4K
                                else VR.Q6K_BLOCK_BYTES)
                )
                if expected != job.payload_bytes:  # pragma: no cover
                    raise StreamImportError(
                        f"{name}: payload {expected} байт, план ждал "
                        f"{job.payload_bytes}"
                    )
                written += _pwrite_all(fd, header, dst)
                payload_digest.update(header)
                copied = 0
                for piece in source.iter_range(
                    job.source_offset, job.n_bytes, chunk_bytes
                ):
                    if copied + len(piece) > job.n_bytes:
                        piece = piece[: job.n_bytes - copied]
                    if not piece:
                        break
                    written += _pwrite_all(fd, piece, dst + written)
                    payload_digest.update(piece)
                    blocks_digest.update(piece)
                    copied += len(piece)
                    if copied >= job.n_bytes:
                        break
                if copied != job.n_bytes:
                    raise StreamImportError(
                        f"{name}: скопировано {copied} байт из "
                        f"{job.n_bytes} — источник оборвался"
                    )
                job.blocks_sha256 = blocks_digest.hexdigest()
                report["q4_bytes"] += written
            else:
                for piece in _fp32_payload_chunks(source, job, chunk_values):
                    written += _pwrite_all(fd, piece, dst + written)
                    payload_digest.update(piece)
                report["fp32_bytes"] += written
            if written != job.payload_bytes:
                raise StreamImportError(
                    f"{name}: записано {written} байт payload'а, ожидалось "
                    f"{job.payload_bytes} — раскладка контейнера разошлась с "
                    "планом (write_vtm_stream посчитал другой размер)"
                )
            job.payload_sha256 = payload_digest.hexdigest()
            report["segments"] += 1
            report["bytes"] += written
            if not quiet:
                line = (
                    f"  {job.plan.role:13s} {name:44s} "
                    f"{str(tuple(job.plan.shape)):20s} "
                    f"{written / 1e6:8.1f} МБ"
                )
                print(line, file=sys.stderr, flush=True)
                if progress is not None:
                    progress(line)
    finally:
        os.close(fd)
    report["seconds"] = round(time.time() - started, 2)
    return report


# ---------------------------------------------------------------------------
# Разбор готового контейнера без загрузки весов
# ---------------------------------------------------------------------------


@dataclass
class SegmentRef:
    """Запись таблицы тензоров + вычисленный размер payload'а."""

    index: int
    name: str
    kind: int
    dims: tuple[int, ...]
    offset: int
    scale_offset: int
    flags: int

    @property
    def n_values(self) -> int:
        return VR.prod(self.dims)

    @property
    def payload_bytes(self) -> int:
        # KV-кэш в файле не хранится: у записи только флаг, offset-маркер.
        if self.flags & VR.TFLAG_KV_CACHE:
            return 0
        return payload_bytes_of(self.kind, self.dims)


@dataclass
class ContainerIndex:
    """Всё, что нужно для проверки контейнера, без секции данных."""

    path: str
    file_size: int
    version: int
    flags: int
    n_instr: int
    n_tensors: int
    data_start: int
    meta_offset: int
    meta_len: int
    program_offset: int
    segments: list[SegmentRef]
    meta: dict[str, Any]
    program: bytes

    def by_name(self) -> dict[str, SegmentRef]:
        return {segment.name: segment for segment in self.segments}


def payload_bytes_of(kind: int, dims: Sequence[int]) -> int:
    """Размер payload'а тензора в контейнере по kind и dims."""
    n_values = VR.prod(tuple(int(d) for d in dims))
    if kind == VR.KIND_Q4K:
        return VR.Q4K_HEADER_SIZE + n_values // VR.Q4K_BLOCK_VALUES * VR.Q4K_BLOCK_BYTES
    if kind == VR.KIND_Q6K:
        return VR.Q4K_HEADER_SIZE + n_values // VR.Q4K_BLOCK_VALUES * VR.Q6K_BLOCK_BYTES
    if kind == VR.KIND_F32:
        return n_values * 4
    raise StreamImportError(
        f"kind {kind}: проверка сегментов поддержана для Q4_K/Q6_K/fp32 "
        "(в MoE-контейнере других видов нет)"
    )


def read_container_index(path: str | os.PathLike[str]) -> ContainerIndex:
    """Прочитать заголовок, таблицу, мету и программу — без секции данных.

    `vtm_ref.read_vtm` читает файл целиком: на 19 ГБ это невозможно, на
    2 ГБ — удвоение RSS. Для сверки сегментов достаточно таблицы.
    """
    path = Path(path)
    try:
        size = path.stat().st_size
        handle = open(path, "rb")
    except OSError as exc:
        raise StreamImportError(f"не открывается {path}: {exc}") from exc
    with handle:
        head = handle.read(VR.HEADER_SIZE)
        if len(head) != VR.HEADER_SIZE:
            raise StreamImportError(
                f"{path.name}: файл короче заголовка VTM ({len(head)} байт)"
            )
        magic, version, flags, n_instr, n_tensors, n_program = (
            VR._HEADER_STRUCT.unpack_from(head, 0)  # noqa: SLF001
        )
        if magic != VR.VTM_MAGIC:
            raise StreamImportError(f"{path.name}: чужой magic {magic!r}")
        if version == VR.VTM_VERSION_V02:
            entry_size = VR.TENSOR_ENTRY_SIZE
            tensor_struct = VR._TENSOR_STRUCT  # noqa: SLF001
            program_offset, meta_offset, meta_len = (
                VR._RESERVED_STRUCT.unpack_from(  # noqa: SLF001
                    head, VR.HEADER_SIZE - 12
                )
            )
        elif version == VR.VTM_VERSION_V03:
            entry_size = VR.TENSOR_ENTRY_SIZE_V03
            tensor_struct = VR._TENSOR_STRUCT_V03  # noqa: SLF001
            meta_offset, meta_len = (
                VR._RESERVED_STRUCT_V03.unpack_from(  # noqa: SLF001
                    head, VR.HEADER_SIZE - 12
                )
            )
            program_offset = size - n_instr * VR.INSTR_SIZE
        else:
            raise StreamImportError(
                f"{path.name}: версия контейнера {version}, поддержаны "
                f"{VR.VTM_VERSION_V02}/{VR.VTM_VERSION_V03}"
            )
        table_end = VR.HEADER_SIZE + entry_size * n_tensors
        if table_end > size:
            raise StreamImportError(
                f"{path.name}: таблица тензоров выходит за файл"
            )
        data_start = VR.align_up(table_end)
        handle.seek(VR.HEADER_SIZE)
        table = handle.read(entry_size * n_tensors)
        segments: list[SegmentRef] = []
        for index in range(n_tensors):
            (
                raw_name, kind, ndim, d0, d1, d2, d3, offset, scale_offset,
                tflags,
            ) = tensor_struct.unpack_from(table, index * entry_size)
            dims = tuple(int(d) for d in (d0, d1, d2, d3)[:ndim])
            segments.append(
                SegmentRef(
                    index=index,
                    name=raw_name.split(b"\x00", 1)[0].decode("utf-8"),
                    kind=int(kind),
                    dims=dims,
                    offset=int(offset),
                    scale_offset=int(scale_offset),
                    flags=int(tflags),
                )
            )
        meta: dict[str, Any] = {}
        if meta_len:
            handle.seek(meta_offset)
            raw_meta = handle.read(meta_len)
            try:
                meta = json.loads(raw_meta.decode("utf-8")) or {}
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise StreamImportError(
                    f"{path.name}: мета не JSON ({exc})"
                ) from exc
        handle.seek(program_offset)
        program = handle.read(n_instr * VR.INSTR_SIZE)
    return ContainerIndex(
        path=str(path),
        file_size=size,
        version=int(version),
        flags=int(flags),
        n_instr=int(n_instr),
        n_tensors=int(n_tensors),
        data_start=int(data_start),
        meta_offset=int(meta_offset),
        meta_len=int(meta_len),
        program_offset=int(program_offset),
        segments=segments,
        meta=meta,
        program=program,
    )


def hash_segments(
    index: ContainerIndex, *, chunk: int = CHUNK_BYTES
) -> dict[str, str]:
    """sha256 payload'а каждого сегмента из файла (потоком, без mmap)."""
    digests: dict[str, str] = {}
    with open(index.path, "rb") as handle:
        for segment in index.segments:
            size = segment.payload_bytes
            if not size:  # KV-кэш: payload'а в файле нет
                continue
            start = index.data_start + segment.offset
            if start + size > index.file_size:
                raise StreamImportError(
                    f"{index.path}: payload {segment.name!r} "
                    f"[{start}, {start + size}) за файлом ({index.file_size})"
                )
            handle.seek(start)
            digest = hashlib.sha256()
            left = size
            while left > 0:
                piece = handle.read(min(chunk, left))
                if not piece:
                    raise StreamImportError(
                        f"{index.path}: файл оборвался на {segment.name!r}"
                    )
                digest.update(piece)
                left -= len(piece)
            digests[segment.name] = digest.hexdigest()
    return digests


def verify_container(
    path: str | os.PathLike[str], expected: dict[str, str]
) -> dict[str, Any]:
    """Сверить sha256 сегментов на диске с ожидаемыми."""
    index = read_container_index(path)
    actual = hash_segments(index)
    missing = [name for name in expected if name not in actual]
    mismatched = [
        name for name, digest in expected.items()
        if name in actual and actual[name] != digest
    ]
    return {
        "segments": len(expected),
        "checked": sum(1 for name in expected if name in actual),
        "bytes": sum(
            segment.payload_bytes for segment in index.segments
            if segment.name in expected
        ),
        "missing": missing,
        "mismatched": mismatched,
    }


def _meta_view(meta: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in meta.items()
            if key not in VOLATILE_META_KEYS}


def compare_containers(
    left: str | os.PathLike[str], right: str | os.PathLike[str]
) -> dict[str, Any]:
    """Сравнить два контейнера: таблица, payload'ы, программа, мета.

    Нужен для критерия «сегменты побайтово равны непоточному пути»: `left`
    собран потоково, `right` — через `moe_import` + `vtm_compile`. Мета
    сравнивается без полей, которых у потокового пути нет по построению
    (`VOLATILE_META_KEYS`).
    """
    left_index = read_container_index(left)
    right_index = read_container_index(right)
    mismatches: list[str] = []
    if left_index.n_tensors != right_index.n_tensors:
        mismatches.append(
            f"тензоров: {left_index.n_tensors} != {right_index.n_tensors}"
        )
    if left_index.version != right_index.version:
        mismatches.append(
            f"версия: {left_index.version} != {right_index.version}"
        )
    if left_index.program != right_index.program:
        mismatches.append("секция программы различается")
    right_by_name = right_index.by_name()
    for segment in left_index.segments:
        other = right_by_name.get(segment.name)
        if other is None:
            mismatches.append(f"{segment.name}: нет в эталоне")
            continue
        for attr in ("kind", "dims", "offset", "scale_offset", "flags"):
            mine = getattr(segment, attr)
            theirs = getattr(other, attr)
            if mine != theirs:
                mismatches.append(
                    f"{segment.name}: {attr} {mine!r} != {theirs!r}"
                )
    extra = set(right_by_name) - {segment.name for segment in left_index.segments}
    if extra:
        mismatches.append(
            "лишние в эталоне: " + ", ".join(sorted(extra)[:5])
        )
    left_hashes = hash_segments(left_index)
    right_hashes = hash_segments(right_index)
    for name, digest in left_hashes.items():
        if right_hashes.get(name) != digest:
            mismatches.append(f"{name}: sha256 payload'а не совпал")
    meta_left = _meta_view(left_index.meta)
    meta_right = _meta_view(right_index.meta)
    meta_diff = sorted(
        key for key in set(meta_left) | set(meta_right)
        if meta_left.get(key) != meta_right.get(key)
    )
    return {
        "same": not mismatches,
        "segments": len(left_index.segments),
        "bytes": sum(
            segment.payload_bytes for segment in left_index.segments
        ),
        "mismatches": mismatches,
        "meta_differences": meta_diff,
    }


# ---------------------------------------------------------------------------
# Потоковый импорт
# ---------------------------------------------------------------------------


def _max_rss_bytes() -> int:
    """Пик RSS процесса (macOS — байты, Linux — килобайты)."""
    try:
        import resource
    except ImportError:  # pragma: no cover - Windows
        return 0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak) if sys.platform == "darwin" else int(peak) * 1024


def stream_import(
    gguf: str | os.PathLike[str],
    out: str | os.PathLike[str],
    *,
    limit_layers: int | None = None,
    limit_vocab: int | None = None,
    keep_temp: bool = False,
    verify: bool = True,
    compare: str | os.PathLike[str] | None = None,
    container_version: int | str = "auto",
    chunk_bytes: int | None = None,
    dry_run: bool = False,
    source_factory: Callable[[str], GgufSource] | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    """Импортировать GGUF прямо в `.vtm`, не создавая `q4.bin`.

    Возвращает отчёт: размеры, время, пик RSS, пик диска, результат сверки
    сегментов и сравнения с непоточным контейнером.
    """
    if chunk_bytes is not None and int(chunk_bytes) < 1024:
        raise StreamImportError(f"chunk_bytes={chunk_bytes} маловат")
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work = out_path.parent
    factory = source_factory or open_source
    source = factory(str(gguf))
    started = time.time()
    temp_path: Path | None = None
    temp_bytes = 0
    part = out_path.with_name(out_path.name + ".part")
    try:
        gguf_file, temp_path, temp_bytes = load_gguf(
            source, work, keep_temp=keep_temp
        )
        try:
            cfg = MI.infer_moe_config(
                gguf_file, limit_layers=limit_layers, limit_vocab=limit_vocab
            )
            plans = MI.plan_moe(gguf_file, cfg)
        except (MI.MoeImportError, G.GgufError) as exc:
            raise StreamImportError(str(exc)) from exc
        model, jobs = build_model(cfg, plans, gguf_file, source)
        estimated = model.estimated_file_size(container_version)
        if dry_run:
            return {
                "dry_run": True,
                "gguf": str(gguf),
                "out": str(out_path),
                "architecture": cfg.as_architecture(),
                "tensors": len(model.tensors),
                "segments": len(jobs),
                "q4_bytes": sum(
                    job.payload_bytes for job in jobs
                    if job.kind in (VR.KIND_Q4K, VR.KIND_Q6K)
                ),
                "fp32_bytes": sum(
                    job.payload_bytes for job in jobs if job.kind == VR.KIND_F32
                ),
                "container_bytes": estimated,
                "container_version": _version_for(model, container_version),
                "instr": len(model.program),
                "kv_bytes": (model.meta.get("kv_cache") or {}).get("bytes", 0),
                "seconds": round(time.time() - started, 2),
                "peak_rss_bytes": _max_rss_bytes(),
                "disk_free_bytes": _disk_free(work),
                "source": source.describe(),
                "temp_bytes": temp_bytes,
            }

        write_started = time.time()
        if part.exists():
            part.unlink()
        container = VR.write_vtm_stream(
            part,
            model.tensors,
            model.program,
            model.meta,
            integrity=False,  # как у непоточного пути для MoE (см. docs/moe.md)
            container_version=container_version,
        )
        skeleton_bytes = part.stat().st_size
        write_seconds = time.time() - write_started
        fill = fill_container(
            part, container, jobs, source,
            chunk_bytes=source.chunk_for(chunk_bytes), quiet=quiet,
        )
        expected = {job.tensor.name: job.payload_sha256 for job in jobs}
        verification = verify_container(part, expected) if verify else None
        if verification and (verification["mismatched"]
                              or verification["missing"]):
            raise StreamImportError(
                "сегменты в контейнере не совпали с записанным: "
                + ", ".join(
                    (verification["mismatched"] + verification["missing"])[:5]
                )
            )
        comparison = compare_containers(part, compare) if compare else None
        if comparison and not comparison["same"]:
            raise StreamImportError(
                "контейнер расходится с непоточным эталоном: "
                + "; ".join(comparison["mismatches"][:5])
            )
        final_size = part.stat().st_size
        if final_size != skeleton_bytes:  # pragma: no cover - pwrite не растягивает
            raise StreamImportError(
                f"размер контейнера изменился при записи весов: "
                f"{skeleton_bytes} -> {final_size}"
            )
        os.replace(part, out_path)
        elapsed = time.time() - started
        aux_kept = temp_bytes if (keep_temp and temp_path is not None) else 0
        return {
            "gguf": str(gguf),
            "out": str(out_path),
            "source": source.describe(),
            "container_bytes": final_size,
            "container_version": container.version,
            "tensors": len(model.tensors),
            "segments": len(jobs),
            "instr": len(model.program),
            "q4_bytes": fill["q4_bytes"],
            "fp32_bytes": fill["fp32_bytes"],
            "payload_bytes": fill["bytes"],
            "kv_bytes": (model.meta.get("kv_cache") or {}).get("bytes", 0),
            "skeleton_seconds": round(write_seconds, 2),
            "fill_seconds": fill["seconds"],
            "seconds": round(elapsed, 2),
            "mb_per_second": round(
                (fill["bytes"] + skeleton_bytes) / max(elapsed, 1e-9) / 1e6, 1
            ),
            "peak_rss_bytes": _max_rss_bytes(),
            # Пик диска = контейнер: `.part` — это и есть контейнер (переименование
            # в конце), а префикс HTTP-GGUF удаляется до начала записи.
            "peak_disk_bytes": final_size + aux_kept,
            "container_plus_ratio": (final_size + aux_kept) / max(final_size, 1),
            "temp_path": str(temp_path) if temp_path is not None else None,
            "temp_bytes": temp_bytes,
            "keep_temp": bool(keep_temp),
            "verify": verification,
            "compare": comparison,
            "disk_free_bytes": _disk_free(work),
        }
    except BaseException:
        if not keep_temp:
            try:
                part.unlink(missing_ok=True)
            except OSError:  # pragma: no cover - защита от гонки
                pass
        raise
    finally:
        if temp_path is not None and not keep_temp:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:  # pragma: no cover
                pass
        source.close()


def _version_for(model: VR.VTM, requested: int | str) -> int:
    """Версия, которую выберет писатель (без записи файла)."""
    size_v02 = model.estimated_file_size(2)
    return VR.choose_container_version(size_v02, VR.parse_container_version(requested))


def _disk_free(path: Path) -> int:
    try:
        stat = os.statvfs(path)
    except (AttributeError, OSError):  # pragma: no cover - Windows
        return 0
    return int(stat.f_bavail * stat.f_frsize)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def human_bytes(value: float) -> str:
    units = ("Б", "КиБ", "МиБ", "ГиБ", "ТиБ")
    size = float(value)
    for unit in units:
        if abs(size) < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{value} Б"  # pragma: no cover


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="moe_stream.py",
        description="Потоковый импорт MoE GGUF (Qwen3-30B-A3B) прямо в .vtm",
    )
    parser.add_argument("--gguf", required=True,
                        help="исходный GGUF: путь или http(s)-URL")
    parser.add_argument("--out", required=True, help="куда писать .vtm")
    parser.add_argument("--limit-layers", type=int, default=None,
                        help="взять только первые N слоёв (срез)")
    parser.add_argument("--limit-vocab", type=int, default=None,
                        help="срез таблицы словаря с начала (отладка)")
    parser.add_argument("--chunk-bytes", type=int, default=None,
                        help=f"кусок чтения/записи (по умолчанию {CHUNK_BYTES} "
                             f"для файла, {HTTP_CHUNK_BYTES} для HTTP)")
    parser.add_argument("--container-version", choices=("auto", "2", "3"),
                        default="auto",
                        help="версия контейнера: auto — v0.2, пока ≤4 ГиБ, "
                             "иначе v0.3 (u64)")
    parser.add_argument("--keep-temp", action="store_true",
                        help="не удалять временные файлы (префикс GGUF, .part)")
    parser.add_argument("--no-verify", action="store_true",
                        help="не перечитывать сегменты после записи")
    parser.add_argument("--compare", default=None, metavar="REF.vtm",
                        help="сверить сегменты с непоточным эталоном")
    parser.add_argument("--dry-run", action="store_true",
                        help="только спланировать и оценить размер контейнера")
    parser.add_argument("--json", action="store_true",
                        help="отчёт в JSON (для приёмки)")
    parser.add_argument("--quiet", action="store_true",
                        help="не печатать по тензору")
    return parser


def _print_report(report: dict[str, Any], *, stats: bool) -> None:
    if stats:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if report.get("dry_run"):
        arch = report["architecture"]
        print(
            f"план: {arch['n_layers']} слоёв, {report['tensors']} тензоров "
            f"({report['segments']} весов), {report['instr']} инструкций"
        )
        print(
            f"Q4_K/Q6_K: {human_bytes(report['q4_bytes'])}, "
            f"fp32: {human_bytes(report['fp32_bytes'])}, "
            f"KV-кэш: {human_bytes(report['kv_bytes'])} (в файле нет)"
        )
        print(
            f"контейнер: {human_bytes(report['container_bytes'])} "
            f"(v0.{report['container_version']}), "
            f"свободно на диске: {human_bytes(report['disk_free_bytes'])}"
        )
        if report["container_bytes"] > report["disk_free_bytes"]:
            print(
                "ВНИМАНИЕ: контейнер больше свободного места — запись не "
                "пройдёт (это оценка, файл не создавался)"
            )
        print(f"время планирования: {report['seconds']} с")
        return
    print(f"источник: {report['source']['label']} "
          f"({human_bytes(report['source']['bytes'])})")
    print(
        f"контейнер: {report['out']} — {human_bytes(report['container_bytes'])}, "
        f"v0.{report['container_version']}"
    )
    print(
        f"тензоров: {report['tensors']} ({report['segments']} весов, "
        f"fp32 {report['fp32_bytes'] / 1e6:.1f} МБ), инструкций "
        f"{report['instr']}"
    )
    print(
        f"каркас: {report['skeleton_seconds']} с, веса: "
        f"{report['fill_seconds']} с, всего: {report['seconds']} с "
        f"({report['mb_per_second']} МБ/с записи)"
    )
    print(
        f"пик RSS: {human_bytes(report['peak_rss_bytes'])}, пик диска: "
        f"{human_bytes(report['peak_disk_bytes'])} "
        f"(контейнер × {report['container_plus_ratio']:.2f})"
    )
    verification = report.get("verify")
    if verification:
        status = ("совпали" if not verification["mismatched"]
                  and not verification["missing"] else "РАСХОЖДЕНИЕ")
        print(
            f"сверка сегментов на диске: {verification['checked']} из "
            f"{verification['segments']}, "
            f"{human_bytes(verification['bytes'])} — {status}"
        )
    comparison = report.get("compare")
    if comparison:
        status = "совпали" if comparison["same"] else "РАСХОЖДЕНИЕ"
        print(
            f"сравнение с эталоном: {comparison['segments']} сегментов, "
            f"{human_bytes(comparison['bytes'])} — {status}"
        )
        if comparison["mismatches"]:
            for line in comparison["mismatches"][:5]:
                print(f"  ! {line}")
    if report.get("temp_path"):
        print(f"временный файл (--keep-temp): {report['temp_path']}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = stream_import(
            args.gguf,
            args.out,
            limit_layers=args.limit_layers,
            limit_vocab=args.limit_vocab,
            keep_temp=args.keep_temp,
            verify=not args.no_verify,
            compare=args.compare,
            container_version=args.container_version,
            chunk_bytes=args.chunk_bytes,
            dry_run=args.dry_run,
            quiet=args.quiet,
        )
    except (StreamImportError, SourceError, ValueError, OSError) as exc:
        print(f"ошибка: {exc}", file=sys.stderr)
        return 2
    _print_report(report, stats=args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
