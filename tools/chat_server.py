#!/usr/bin/env python3
"""Веб-адаптер для сравнения двух моделей рядом (задача 0138).

Зачем: чтобы оценить «насколько умна» модель, нужно видеть её ответ и ответ
Qwen одновременно, с временем и скоростью. Скрипт этого не даёт — нужна
страница. Этот модуль поднимает минимальный сервер только на петлевом
интерфейсе и отдаёт одну HTML-страницу со встроенными CSS/JS.

Что внутри:

* ``GET /`` — страница: один вопрос, две колонки (ВТМ и Ollama), выбор модели
  в каждой, время ответа и ток/с;
* ``GET /api/backends`` — закрытый список бэкендов и моделей (без путей и URL);
* ``GET /api/models`` — карточки моделей для панели метрик (задача 0145):
  размер файла (``stat``), параметры и KV-кэш из таблицы тензоров ``.vtm``,
  контекст, тип хранения, источник (демо / холодный экспорт / QAT), словарь,
  слои, бит на параметр, RSS до/после загрузки;
* ``GET /api/system`` — нагрузка и память машины: load average, свободная
  RAM, диск, RSS процесса, ``VTM_THREADS``;
* ``POST /api/chat`` — SSE-поток ``text/event-stream``. События:
  ``start`` (какая модель отвечает, первый токен холодный/тёплый),
  ``delta`` (кусок текста), ``done`` (время, токены, TTFT/prefill/декод,
  RSS-дельта, причина остановки, кэш префикса), ``end`` (CPU-время запроса и
  снимок системы), ``error`` (человеческий текст). Терминальное событие —
  ``data: [DONE]``, как у ``eva/server.py``;
* ``GET /favicon.ico`` — 204, чтобы браузер не шумел 404.

Числа панели либо сняты у источника (``stat``, ``/proc``, ``vm_stat``,
``getrusage``, ``time.process_time``, таблица тензоров ``.vtm``), либо
помечены как оценка; чего снять нельзя — отдаётся ``null`` и в UI это «—».
Сеть у страницы и у сервера одна: локальная Ollama на ``127.0.0.1:11434``
(только ``/api/ps``, чтобы узнать размер резидентной копии).

Границы:

* сервер слушает только ``127.0.0.1``; ``0.0.0.0`` отвергается;
* клиент не передаёт URL: только ``backend`` и ``model`` из закрытого
  реестра. Неизвестный бэкенд/модель — 400 до открытия потока;
* внешних запросов у страницы нет: CSS и JS встроены, CSP запрещает всё,
  кроме inline-стилей/скриптов и запросов к своему origin.

Бэкенды (оба — через библиотеку ``tools/chat.py``, она появилась во время
работы и переиспользуется, а не дублируется):

* ``vtm`` — локальный ``.vtm``: ``chat.build_backend(backend="vtm", ...)``,
  внутри ``eva.backends.VtmBackend``, без сети (ленивая загрузка);
* ``ollama`` — OpenAI-совместимый ``http://127.0.0.1:11434/v1``:
  ``chat.OpenAIChatBackend(base=.../v1, key="")``. Хост зашит в код, клиент
  его не выбирает; ключ из окружения сюда намеренно не подставляется.

HTTP-слой знает только про ``backend``/``model`` из реестра и про
``StreamEvent`` из ``chat.py``; замена бэкенда не трогает страницу.

Запуск::

    python3 tools/chat_server.py --port 8765

Открыть ``http://127.0.0.1:8765/``. Если Ollama установлена, но моделей с
именами по умолчанию нет, перечислите свои::

    python3 tools/chat_server.py --port 8765 \\
        --ollama-model qwen2.5:7b --ollama-model qwen3:8b

Только стандартная библиотека; бэкенды берутся из ``tools/chat.py`` (внутри —
``eva``), их файлы не меняются.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import pathlib
import logging
import os
import queue
import shutil
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence
from urllib.parse import urlsplit

try:  # ru_maxrss — пиковый RSS; на Windows модуля нет, там честное «—»
    import resource
except ImportError:  # pragma: no cover - не наш случай, но падать незачем
    resource = None  # type: ignore[assignment]

__all__ = [
    "BackendSpec",
    "ChatHTTPServer",
    "ClientError",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_PORT",
    "HOST",
    "ModelSpec",
    "OLLAMA_BASE",
    "ChatCliAdapter",
    "StreamPiece",
    "VtmInspect",
    "create_server",
    "default_adapters",
    "default_registry",
    "inspect_vtm",
    "main",
    "system_snapshot",
]

logger = logging.getLogger("chat_server")
logger.addHandler(logging.NullHandler())

#: Слушаем только петлевой интерфейс: наружу этот стенд не выставляется.
HOST = "127.0.0.1"
#: Порт по умолчанию (переопределяется ``--port``).
DEFAULT_PORT = 8765
#: Корень репозитория: ``tools/chat_server.py`` -> ``tools`` -> корень.
ROOT = Path(__file__).resolve().parents[1]
# Скрипт запускают и как ``python3 tools/chat_server.py``: тогда в sys.path
# лежит ``tools``, а не корень, и ``import eva`` не находится. Добавляем корень
# один раз, чтобы адаптеры работали при любом способе запуска.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
#: Ollama — только этот адрес; клиент не может подставить свой.
OLLAMA_BASE = "http://127.0.0.1:11434/v1"
#: Сколько новых токенов просим у обеих моделей (сравнение на равных).
DEFAULT_MAX_TOKENS = 256
#: Потолок тела запроса: вопрос + две ссылки на модели, больше не нужно.
MAX_BODY_BYTES = 64 * 1024
#: Потолок вопроса в символах; защита от случайной вставки книги.
MAX_QUESTION_CHARS = 8000
#: Модели Ollama по умолчанию. Имена можно заменить ``--ollama-model``.
DEFAULT_OLLAMA_MODELS: tuple[tuple[str, str], ...] = (
    ("qwen2.5:7b", "Qwen2.5 7B"),
    ("qwen2.5:3b", "Qwen2.5 3B"),
    ("qwen2.5:1.5b", "Qwen2.5 1.5B"),
    ("qwen3:8b", "Qwen3 8B"),
)
#: Где искать локальные модели. Список не фиксированный: иначе новые `.vtm`
#: появляются на диске и остаются невидимыми в чате (так и случилось с холодным
#: экспортом). Обходим каталоги, а не перечисляем файлы.
VTM_SEARCH_DIRS: tuple[str, ...] = ("runs", "export", "demo")
#: Токенизатор-заглушка: фикстура BPE (287) совпадает с ``runs/tiny``.
#: Для демо-модели (vocab=128) она не подходит — чат вернёт понятную ошибку.
FALLBACK_TOKENIZER = ROOT / "tests" / "vtm" / "fixtures" / "tokenizer.json"
#: Системный промпт EVA32: одинаковый для обеих моделей — сравнение честное.
SYSTEM_PROMPT_PATH = ROOT / "prompts" / "system.assistant.md"

#: Ошибки записи, означающие уход клиента: стрим прекращаем, не шумим в лог.
_CLIENT_GONE = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)


# --------------------------------------------------------------------------- #
# Реестр бэкендов: только то, что разрешено сервером
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelSpec:
    """Модель в закрытом реестре.

    ``id`` — то, что приходит от клиента и попадает в ответы API; ``target`` —
    путь к ``.vtm`` или имя модели Ollama (наружу не отдаётся). ``tokenizer``
    нужен только локальной ВТМ.
    """

    id: str
    label: str
    target: str = ""
    tokenizer: str = ""


@dataclass(frozen=True)
class BackendSpec:
    """Бэкенд и его разрешённые модели."""

    id: str
    label: str
    kind: str
    models: tuple[ModelSpec, ...] = ()


@dataclass
class StreamPiece:
    """Кусок ответа бэкенда: текст, либо терминальный кусок с ``usage``.

    Своя структура, а не ``eva.clients.StreamChunk``: HTTP-слою не нужен
    импорт ``eva``, а тестам — модели. Адаптеры переводят чужой формат сюда.
    """

    text: str = ""
    done: bool = False
    usage: dict[str, Any] = field(default_factory=dict)
    finish_reason: str = ""


def _resolve_tokenizer(model_path: Path, explicit: str | None = None) -> str:
    """Токенизатор ВТМ: явный, из env, рядом с моделью, затем демо-фикстура."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env_value = (os.environ.get("EVA_VTM_TOKENIZER") or "").strip()
    if env_value:
        candidates.append(Path(env_value).expanduser())
    candidates.append(model_path.with_name("tokenizer.json"))
    candidates.append(ROOT / "tokenizer.json")
    candidates.append(FALLBACK_TOKENIZER)
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return ""


def _discover_vtm_models() -> list[pathlib.Path]:
    """Найти локальные `.vtm` в рабочих каталогах (стабильный порядок)."""
    found: list[pathlib.Path] = []
    for base in VTM_SEARCH_DIRS:
        root = ROOT / base
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.vtm")):
            if path.is_file():
                found.append(path)
    return found


def _model_label(path: pathlib.Path) -> str:
    """Человекочитаемая подпись: путь, размер и подсказка о содержимом."""
    size_mb = path.stat().st_size / (1024 * 1024)
    relative = path.relative_to(ROOT).as_posix()
    hint = ""
    if "cold" in relative:
        hint = " — холодный экспорт"
    elif "bench_shape" in relative:
        hint = " — случайные веса (только скорость)"
    elif relative.startswith("demo/"):
        hint = " — демо (vocab=128, без текста)"
    elif "tiny" in relative:
        hint = " — мини-модель (тест)"
    return f"{relative} — {size_mb:.1f} МБ{hint}"


def default_registry(
    *,
    vtm_tokenizer: str | None = None,
    ollama_models: Sequence[tuple[str, str]] | None = None,
) -> tuple[BackendSpec, ...]:
    """Собрать закрытый реестр: локальные ``.vtm`` и модели Ollama.

    ВТМ-модели добавляются только если файл существует: иначе выбор в UI
    предлагал бы заведомо нерабочий вариант. Порядок фиксирован.
    """
    vtm: list[ModelSpec] = []
    for path in _discover_vtm_models():
        relative = path.relative_to(ROOT).as_posix()
        vtm.append(
            ModelSpec(
                id=relative,
                label=_model_label(path),
                target=str(path),
                tokenizer=_resolve_tokenizer(path, vtm_tokenizer),
            )
        )

    catalog = tuple(ollama_models) if ollama_models is not None else DEFAULT_OLLAMA_MODELS
    ollama = tuple(
        ModelSpec(id=name, label=label, target=name) for name, label in catalog
    )
    return (
        BackendSpec(id="vtm", label="ВТМ (локальная)", kind="vtm", models=tuple(vtm)),
        BackendSpec(
            id="ollama", label="Ollama (Qwen)", kind="ollama", models=ollama
        ),
    )


# --------------------------------------------------------------------------- #
# Адаптеры бэкендов
# --------------------------------------------------------------------------- #


class ChatCliAdapter:
    """Адаптер к бэкенду из ``tools/chat.py`` (общая библиотека чата).

    Сборка бэкенда из ``chat.py`` ленивая: страница и ``/api/backends`` не
    грузят ВТМ и не импортируют ``eva``. ``StreamEvent`` из чата переводится
    в :class:`StreamPiece`, а ошибки бэкенда остаются человеческим текстом
    (``chat.py`` уже добавляет подсказки).

    Один адаптер = одна модель; генерации к ней сериализуются замком, потому
    что у ВТМ один C-контекст и параллельный вызов вернул бы ``VtmBusy``.

    Здесь же снимается память загрузки (задача 0145): RSS процесса до и после
    вызова фабрики. Для ВТМ это честная дельта — модель живёт в этом процессе
    через ``ctypes``. ``probe_ps`` — только для Ollama: её процесс чужой,
    поэтому размер резидентной копии спрашивают у ``GET /api/ps``.
    """

    def __init__(
        self,
        factory: Callable[[], Any],
        *,
        label: str = "",
        kind: str = "",
        probe_ps: Callable[[str], dict[str, Any] | None] | None = None,
    ) -> None:
        self._factory = factory
        self.label = label
        self.kind = kind
        self.probe_ps = probe_ps
        self._inner: Any = None
        #: Замер загрузки: RSS до/после, дельта, пик процесса и время сборки.
        self.load_stats: dict[str, Any] | None = None
        # RLock: ``stream`` держит замок на всё время генерации и зовёт
        # ``_ensure``, которому нужен тот же замок.
        self._lock = threading.RLock()

    @property
    def loaded(self) -> bool:
        """Модель уже в памяти этого процесса (а не только в реестре)."""
        return self._inner is not None

    @property
    def cache_prefix(self) -> bool | None:
        """Включён ли у ВТМ кэш префикса; ``None`` — бэкенд не умеет."""
        flag = getattr(self._inner, "cache_prefix", None)
        return flag if isinstance(flag, bool) else None

    def _ensure(self) -> Any:
        if self._inner is None:
            with self._lock:
                if self._inner is None:
                    before = _rss_bytes()
                    started = time.monotonic()
                    inner = self._factory()
                    self._inner = inner
                    after = _rss_bytes()
                    self.load_stats = {
                        "rss_before": before,
                        "rss_after": after,
                        "rss_delta": (
                            after - before
                            if before is not None and after is not None
                            else None
                        ),
                        "peak": _peak_rss_bytes(),
                        "load_ms": round((time.monotonic() - started) * 1000.0, 1),
                    }
        return self._inner

    def stream(
        self, messages: Sequence[Mapping[str, str]], *, max_tokens: int
    ) -> Iterator[StreamPiece]:
        with self._lock:
            inner = self._ensure()
            terminal = False
            iterator = inner.stream(list(messages), max_tokens=max_tokens)
            try:
                for event in iterator:
                    text = getattr(event, "text", "")
                    if isinstance(text, str) and text:
                        yield StreamPiece(text=text)
                    if getattr(event, "done", False):
                        usage = getattr(event, "usage", {}) or {}
                        terminal = True
                        yield StreamPiece(
                            done=True,
                            usage=dict(usage),
                            finish_reason=str(getattr(event, "finish_reason", "") or ""),
                        )
                        break
            finally:
                closer = getattr(iterator, "close", None)
                if callable(closer):
                    closer()
            if not terminal:
                raise RuntimeError(
                    "бэкенд завершил поток без терминального события"
                )


def _chat_cli_factory(
    kind: str, model: ModelSpec, timeout: float
) -> Callable[[], Any]:
    """Фабрика бэкенда из ``tools/chat.py``; импорт и модель — при первом запросе."""

    def factory() -> Any:
        try:
            from tools import chat as chat_cli
        except ImportError as exc:  # pragma: no cover - chat.py есть в репозитории
            raise RuntimeError(
                "нет библиотеки чата: не импортируется tools/chat.py"
            ) from exc
        target = model.target or model.id
        if kind == "vtm":
            return chat_cli.build_backend(
                backend="vtm",
                model=target,
                tokenizer=model.tokenizer or None,
                timeout=timeout,
            )
        if kind == "ollama":
            # Класс из chat.py напрямую, а не build_backend: иначе пустой key
            # подменился бы EVA_BIG_KEY/OPENAI_API_KEY из окружения, и ключ
            # облака ушёл бы на локальный порт. Ключ здесь не нужен.
            return chat_cli.OpenAIChatBackend(
                base=OLLAMA_BASE,
                model=target,
                key="",
                timeout=timeout,
            )
        raise RuntimeError("неизвестный вид бэкенда: {}".format(kind))

    return factory


#: Корень API Ollama (``OLLAMA_BASE`` без ``/v1``) — только для ``/api/ps``.
OLLAMA_ROOT = OLLAMA_BASE.rsplit("/v1", 1)[0]
#: Потолок ответа ``/api/ps``: там список загруженных моделей, не гигабайты.
_OLLAMA_PS_LIMIT = 1024 * 1024


def _probe_ollama_ps(model: str, *, timeout: float = 1.5) -> dict[str, Any] | None:
    """Спросить у локальной Ollama ``GET /api/ps``: ``size`` и ``size_vram``.

    Ollama живёт отдельным процессом, поэтому RSS его модели нам не виден —
    единственный честный источник этих чисел её собственный ``/api/ps``.
    Ошибки (не запущена, таймаут, кривой JSON) — ``None``: в UI будет «—».
    """
    import urllib.request

    try:
        with urllib.request.urlopen(  # noqa: S310 - адрес зашит: 127.0.0.1
            OLLAMA_ROOT + "/api/ps", timeout=timeout
        ) as response:
            raw = response.read(_OLLAMA_PS_LIMIT)
        payload = json.loads(raw.decode("utf-8"))
    except Exception:  # noqa: BLE001 - сеть/JSON: числа просто неизвестны
        return None
    entries = payload.get("models") if isinstance(payload, Mapping) else None
    if not isinstance(entries, list):
        return None
    wanted = str(model).strip()
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        name = str(entry.get("name") or entry.get("model") or "").strip()
        if name != wanted and name.split(":", 1)[0] != wanted.split(":", 1)[0]:
            continue
        return {
            "model": name,
            "size": _as_int(entry.get("size")),
            "size_vram": _as_int(entry.get("size_vram")),
            "expires_at": entry.get("expires_at"),
        }
    return None


def default_adapters(
    registry: Iterable[BackendSpec], *, timeout: float = 120.0
) -> dict[tuple[str, str], Any]:
    """Адаптеры для всех моделей реестра; ВТМ открывается лениво."""
    adapters: dict[tuple[str, str], Any] = {}
    for backend in registry:
        for model in backend.models:
            if backend.kind not in ("vtm", "ollama"):
                continue
            adapters[(backend.id, model.id)] = ChatCliAdapter(
                _chat_cli_factory(backend.kind, model, timeout),
                label="{} / {}".format(backend.id, model.id),
                kind=backend.kind,
                # Сеть тестам не нужна: заглушки ``probe_ps`` не имеют.
                probe_ps=_probe_ollama_ps if backend.kind == "ollama" else None,
            )
    return adapters


# --------------------------------------------------------------------------- #
# Измерения: память процесса, система, контейнер .vtm
# --------------------------------------------------------------------------- #

#: Заголовок контейнера VTM v2: magic, version, flags, n_instr, n_tensors,
#: n_program, program_offset, meta_offset, meta_len (32 байта).
_VTM_HEADER = struct.Struct("<4sHHIIIIII")
#: Запись таблицы тензоров v2: 24+1+1+16+4+4+2 = 52 байта.
_VTM_TENSOR = struct.Struct("<24sBB4IIIH")
#: Заголовок сегмента TRT1/INT8: magic, version, flags, n, n_scale (18 байт).
_VTM_SEGMENT = struct.Struct("<4sBBQI")
#: Роль тензора: бит 0 — KV-кэш, payload которого в файле не лежит.
VTM_TENSOR_FLAG_KV_CACHE = 0x0001
#: ``offset`` тензора, payload которого в файле отсутствует.
VTM_NO_OFFSET = 0xFFFFFFFF
#: Байт на элемент для ``kind``: 1 fp32, 2 fp16, 3 bf16, 4 int8; 0 (trit) — сегмент.
_VTM_KIND_BYTES = {1: 4, 2: 2, 3: 2, 4: 1}
_VTM_KIND_NAMES = {0: "trit", 1: "fp32", 2: "fp16", 3: "bf16", 4: "int8"}
#: Потолки чтения: битый заголовок не должен просить гигабайты.
_VTM_MAX_TENSORS = 200_000
_VTM_MAX_META_BYTES = 4 * 1024 * 1024


def _align32(value: int) -> int:
    return (int(value) + 31) & ~31


def _linux_rss_bytes() -> int | None:
    """RSS из ``/proc/self/status`` (Linux)."""
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


class _DarwinTaskInfo(ctypes.Structure):
    """``struct proc_taskinfo`` из ``<libproc.h>`` (macOS), 96 байт."""

    _fields_ = [
        ("virtual_size", ctypes.c_uint64),
        ("resident_size", ctypes.c_uint64),
        ("total_user", ctypes.c_uint64),
        ("total_system", ctypes.c_uint64),
        ("threads_user", ctypes.c_uint64),
        ("threads_system", ctypes.c_uint64),
        ("policy", ctypes.c_int32),
        ("faults", ctypes.c_int32),
        ("pageins", ctypes.c_int32),
        ("cow_faults", ctypes.c_int32),
        ("messages_sent", ctypes.c_int32),
        ("messages_received", ctypes.c_int32),
        ("syscalls_mach", ctypes.c_int32),
        ("syscalls_unix", ctypes.c_int32),
        ("csw", ctypes.c_int32),
        ("threadnum", ctypes.c_int32),
        ("numrunning", ctypes.c_int32),
        ("priority", ctypes.c_int32),
    ]


#: ``PROC_PIDTASKINFO`` из ``<libproc.h>``.
_PROC_PIDTASKINFO = 4
_LIBPROC: Any = None
_LIBPROC_TRIED = False


def _darwin_rss_bytes() -> int | None:
    """RSS из ``proc_pidinfo()`` (macOS): ровно то, что видит система."""
    global _LIBPROC, _LIBPROC_TRIED
    if not _LIBPROC_TRIED:
        _LIBPROC_TRIED = True
        try:
            name = ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib"
            library = ctypes.CDLL(name)
            library.proc_pidinfo.restype = ctypes.c_int
            library.proc_pidinfo.argtypes = [
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_uint64,
                ctypes.c_void_p,
                ctypes.c_int,
            ]
            _LIBPROC = library
        except (OSError, AttributeError):  # pragma: no cover - среда без libproc
            _LIBPROC = None
    if _LIBPROC is None:  # pragma: no cover - среда без libproc
        return None
    info = _DarwinTaskInfo()
    size = ctypes.sizeof(info)
    try:
        got = _LIBPROC.proc_pidinfo(
            os.getpid(), _PROC_PIDTASKINFO, 0, ctypes.byref(info), size
        )
    except (OSError, ValueError):  # pragma: no cover - защита от ctypes
        return None
    if got != size:
        return None
    return int(info.resident_size)


def _rss_bytes() -> int | None:
    """RSS текущего процесса в байтах; ``None`` — платформа не даёт."""
    if sys.platform.startswith("linux"):
        return _linux_rss_bytes()
    if sys.platform == "darwin":
        return _darwin_rss_bytes()
    return None  # pragma: no cover - Windows/BSD не поддерживаем


def _peak_rss_bytes() -> int | None:
    """Пиковый RSS процесса (``getrusage``): Linux — КиБ, macOS — байты."""
    if resource is None:  # pragma: no cover - Windows
        return None
    try:
        raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (OSError, ValueError):  # pragma: no cover - защита от ctypes
        return None
    return raw if sys.platform == "darwin" else raw * 1024


def process_memory() -> dict[str, Any]:
    """Снимок памяти процесса: текущий RSS, пик и источник числа."""
    rss = _rss_bytes()
    peak = _peak_rss_bytes()
    return {
        "rss": rss,
        "peak": peak if peak is not None else rss,
        "source": "proc" if rss is not None and sys.platform.startswith("linux")
        else ("libproc" if rss is not None else ("getrusage" if peak is not None else None)),
    }


def _linux_ram() -> tuple[int | None, int | None]:
    """``(всего, свободно)`` из ``/proc/meminfo``; свободно — MemAvailable."""
    values: dict[str, int] = {}
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                if key not in ("MemTotal", "MemAvailable"):
                    continue
                values[key] = int(rest.split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        return None, None
    return values.get("MemTotal"), values.get("MemAvailable")


def _darwin_ram() -> tuple[int | None, int | None]:
    """``(всего, свободно)`` по ``vm_stat``: free + inactive + speculative."""
    try:
        completed = subprocess.run(
            ["vm_stat"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - среда
        return None, None
    if completed.returncode != 0:  # pragma: no cover - среда
        return None, None
    lines = completed.stdout.splitlines()
    page_size = 4096
    counters: dict[str, int] = {}
    for line in lines:
        if "page size of" in line:
            try:
                page_size = int(line.split("page size of", 1)[1].split()[0])
            except (IndexError, ValueError):  # pragma: no cover - формат vm_stat
                pass
            continue
        key, _, rest = line.partition(":")
        rest = rest.strip().rstrip(".")
        if key and rest.isdigit():
            counters[key.strip()] = int(rest)
    free_pages = (
        counters.get("Pages free", 0)
        + counters.get("Pages inactive", 0)
        + counters.get("Pages speculative", 0)
    )
    free = free_pages * page_size if counters else None
    try:
        total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):  # pragma: no cover - среда
        total = None
    return total, free


def system_snapshot() -> dict[str, Any]:
    """Нагрузка и память машины; чего нет — ``None`` (в UI «—»)."""
    try:
        load: list[float] | None = [
            round(float(value), 2) for value in os.getloadavg()
        ]
    except (OSError, AttributeError):  # pragma: no cover - не на всех ОС
        load = None

    if sys.platform.startswith("linux"):
        ram_total, ram_free = _linux_ram()
        ram_source = "proc_meminfo" if ram_total is not None else None
    elif sys.platform == "darwin":
        ram_total, ram_free = _darwin_ram()
        ram_source = "vm_stat" if ram_free is not None else None
    else:  # pragma: no cover - Windows/BSD
        ram_total, ram_free, ram_source = None, None, None

    try:
        usage = shutil.disk_usage(str(ROOT))
        disk: dict[str, int] | None = {
            "total": int(usage.total),
            "free": int(usage.free),
            "used": int(usage.used),
        }
    except OSError:  # pragma: no cover - среда без диска
        disk = None

    memory = process_memory()
    return {
        "load": load,
        "cpu_count": os.cpu_count(),
        "ram": {"total": ram_total, "free": ram_free, "source": ram_source},
        "disk": disk,
        "process": {
            "pid": os.getpid(),
            "rss": memory["rss"],
            "peak": memory["peak"],
            "source": memory["source"],
        },
        "vtm_threads": {
            "runtime": os.environ.get("VTM_THREADS") or None,
            "config": os.environ.get("EVA_VTM_THREADS") or None,
            "cpu_count": os.cpu_count(),
        },
        "platform": sys.platform,
    }


@dataclass(frozen=True)
class VtmInspect:
    """Что удалось прочитать в контейнере ``.vtm`` (только файл, без загрузки)."""

    size_bytes: int
    format_version: int
    tensor_count: int
    params: int | None
    vocab: int | None
    layers: int | None
    context: int | None
    kv_bytes: int | None
    kv_in_file: bool | None
    weights_bytes: int | None
    bits_per_param: float | None
    storage: str
    source: str
    meta: Mapping[str, Any] = field(default_factory=dict)


def _tensor_elements(dims: Sequence[int]) -> int:
    """Число элементов по ``dims`` (нули — «измерения кончились»)."""
    total = 1
    for dim in dims:
        if int(dim) <= 0:
            break
        total *= int(dim)
    return max(0, total)


def _read_segment(handle: Any, data_start: int, offset: int) -> dict[str, Any] | None:
    """Разобрать заголовок сегмента TRT1/INT8 и посчитать его размер в файле."""
    if offset in (0, VTM_NO_OFFSET):
        return None
    try:
        handle.seek(data_start + int(offset))
        raw = handle.read(_VTM_SEGMENT.size)
    except (OSError, ValueError):
        return None
    if len(raw) < _VTM_SEGMENT.size:
        return None
    magic, _version, flags, count, scales = _VTM_SEGMENT.unpack(raw)
    if magic not in (b"TRT1", b"INT8"):
        return None
    scale_bytes = 4 if flags & 1 else 2
    trit_storage: str | None = None
    if magic == b"TRT1":
        # Биты 1..2 flags: 0 packed20 (20 тритов в u32), 1 native (4 в u32).
        trit_storage = "native" if (flags >> 1) & 0x3 else "packed20"
        words = (count + 159) // 160 * 8
        if trit_storage == "native":
            words = (count + 3) // 4
        payload = words * 4
    else:
        payload = int(count)
    return {
        "magic": magic.decode("ascii"),
        "bytes": _VTM_SEGMENT.size + int(scales) * scale_bytes + payload,
        "trit_storage": trit_storage,
        "scale_bytes": scale_bytes,
    }


def _meta_int(meta: Mapping[str, Any], *keys: str) -> int | None:
    """Первое целое по цепочке ключей; вложенные — через точку (``model.vocab``)."""
    for key in keys:
        value: Any = meta
        for part in key.split("."):
            if not isinstance(value, Mapping):
                value = None
                break
            value = value.get(part)
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, int) and value > 0:
            return int(value)
        if isinstance(value, float) and value.is_integer() and value > 0:
            return int(value)
    return None


def _vtm_source(meta: Mapping[str, Any], vocab: int | None) -> str:
    """Источник модели по мете; без меты — эвристика (vocab 128 = демо)."""
    if not meta:
        if vocab == 128:
            return "демо (vocab=128, метаданных нет)"
        return "неизвестно (метаданных нет)"
    base = meta.get("base_model")
    names: list[str] = []
    if isinstance(base, Mapping):
        names = [
            str(value)
            for value in (base.get("name_or_path"), base.get("revision"), base.get("dir"))
            if value
        ]
    if meta.get("revision"):
        names.append(str(meta["revision"]))
    if meta.get("checkpoint"):
        names.append(str(meta["checkpoint"]))
    joined = " ".join(names).lower()
    if "qat" in joined or meta.get("step") is not None:
        return "QAT-экспорт"
    if isinstance(meta.get("kv_cache"), Mapping) or (
        isinstance(base, Mapping) and base.get("dir") and not meta.get("checkpoint")
    ):
        return "холодный экспорт"
    if vocab == 128:
        return "демо (vocab=128)"
    return "экспорт (источник в мете не указан)"


def _vtm_storage(
    meta: Mapping[str, Any],
    embed_kind: int | None,
    trit_storage: str | None,
    weight_kinds: Mapping[int, int],
) -> str:
    """Тип хранения: чем лежат веса и эмбеддинг."""
    parts: list[str] = []
    weight_part: list[str] = []
    if weight_kinds.get(0):
        weight_part.append("trit" + (" ({})".format(trit_storage) if trit_storage else ""))
    if weight_kinds.get(4):
        weight_part.append("int8")
    for kind in (2, 3):
        if weight_kinds.get(kind):
            weight_part.append(_VTM_KIND_NAMES[kind])
    if not weight_part and weight_kinds.get(1):
        weight_part.append("fp32")
    if weight_part:
        parts.append("веса: " + ", ".join(weight_part))
    quant = meta.get("quantization")
    scale_dtype = None
    if isinstance(quant, Mapping):
        scale_dtype = quant.get("scale_dtype")
    if scale_dtype is None and isinstance(meta.get("model"), Mapping):
        scale_dtype = meta["model"].get("scale_dtype")  # type: ignore[index]
    if scale_dtype:
        parts.append("масштабы: {}".format(scale_dtype))
    if embed_kind is not None:
        parts.append("эмбеддинг: {}".format(_VTM_KIND_NAMES.get(embed_kind, "?")))
    return "; ".join(parts) if parts else "—"


_VTM_CACHE: dict[str, tuple[int, int, "VtmInspect | None"]] = {}
_VTM_CACHE_LOCK = threading.Lock()


def inspect_vtm(path: str | os.PathLike[str]) -> VtmInspect | None:
    """Прочитать карточку модели из контейнера ``.vtm`` — без загрузки в память.

    Все числа берутся из самого файла: размер — ``stat``, параметры и KV-кэш —
    сумма по таблице тензоров, контекст/словарь/слои — из JSON-метаданных
    (если их нет — из формы тензоров). Файл не найден или это не VTM1 —
    ``None``.

    Разбор кэшируется по ``(путь, размер, mtime)``: у холодной модели 600
    тензоров читаются ~0.1 с, а панель спрашивает карточки после каждого
    ответа. Кэш не врёт: изменился файл — изменился mtime.
    """
    target = Path(path)
    try:
        stat = target.stat()
    except OSError:
        return None
    if not target.is_file():
        return None
    key = os.path.abspath(str(target))
    stamp = (int(stat.st_size), int(stat.st_mtime_ns))
    with _VTM_CACHE_LOCK:
        cached = _VTM_CACHE.get(key)
    if cached is not None and cached[0] == stamp[0] and cached[1] == stamp[1]:
        return cached[2]
    info = _inspect_vtm_file(target, stat)
    with _VTM_CACHE_LOCK:
        _VTM_CACHE[key] = (stamp[0], stamp[1], info)
    return info


def _inspect_vtm_file(target: Path, stat: os.stat_result) -> VtmInspect | None:
    try:
        with open(target, "rb") as handle:
            head = handle.read(_VTM_HEADER.size)
            if len(head) < _VTM_HEADER.size:
                return None
            (
                magic,
                version,
                _flags,
                _n_instr,
                n_tensors,
                _n_program,
                _program_offset,
                meta_offset,
                meta_len,
            ) = _VTM_HEADER.unpack(head)
            if magic != b"VTM1":
                return None
            if n_tensors < 0 or n_tensors > _VTM_MAX_TENSORS:
                return None
            table = handle.read(_VTM_TENSOR.size * n_tensors)
            if len(table) < _VTM_TENSOR.size * n_tensors:
                return None

            meta: dict[str, Any] = {}
            if 0 < meta_len <= _VTM_MAX_META_BYTES and meta_offset + meta_len <= stat.st_size:
                try:
                    handle.seek(meta_offset)
                    parsed = json.loads(handle.read(meta_len).decode("utf-8"))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    parsed = None
                if isinstance(parsed, dict):
                    meta = parsed

            data_start = _align32(_VTM_HEADER.size + _VTM_TENSOR.size * n_tensors)
            params = 0
            weights_bytes = 0
            kv_bytes = 0
            kv_in_file: bool | None = None
            kv_context: int | None = None
            embed_kind: int | None = None
            embed_vocab: int | None = None
            trit_storage: str | None = None
            weight_kinds: dict[int, int] = {}
            layer_ids: set[str] = set()

            for index in range(n_tensors):
                (
                    raw_name,
                    kind,
                    _ndim,
                    d0,
                    d1,
                    d2,
                    d3,
                    offset,
                    _scale_offset,
                    tensor_flags,
                ) = _VTM_TENSOR.unpack_from(table, index * _VTM_TENSOR.size)
                name = raw_name.split(b"\0", 1)[0].decode("utf-8", "replace")
                dims = (d0, d1, d2, d3)
                elements = _tensor_elements(dims)
                if name.startswith("l") and "." in name:
                    layer_ids.add(name.split(".", 1)[0])
                if name == "emb.w":
                    embed_kind = int(kind)
                    embed_vocab = int(d0) if d0 else None
                # KV-кэш: новые контейнеры помечают его флагом, старые — только
                # именем (``l0.kcache``/``l0.vcache``); признак нужен оба.
                is_kv = bool(tensor_flags & VTM_TENSOR_FLAG_KV_CACHE) or (
                    name.endswith("kcache") or name.endswith("vcache")
                )
                if is_kv:
                    per_element = _VTM_KIND_BYTES.get(int(kind))
                    if per_element:
                        kv_bytes += elements * per_element
                    if kv_context is None and d0:
                        kv_context = int(d0)
                    in_file = offset not in (0, VTM_NO_OFFSET)
                    kv_in_file = bool(kv_in_file or in_file) if kv_in_file is not None else in_file
                    continue
                weight = (
                    int(kind) in (0, 4)
                    or name.endswith(".w")
                    or name.endswith("_bias")
                )
                if not weight:
                    continue
                weight_kinds[int(kind)] = weight_kinds.get(int(kind), 0) + 1
                params += elements
                if int(kind) in (0, 4):
                    segment = _read_segment(handle, data_start, int(offset))
                    if segment is not None:
                        weights_bytes += int(segment["bytes"])
                        if int(kind) == 0 and segment["trit_storage"] and trit_storage is None:
                            trit_storage = str(segment["trit_storage"])
                else:
                    per_element = _VTM_KIND_BYTES.get(int(kind))
                    if per_element:
                        weights_bytes += elements * per_element
    except (OSError, ValueError, struct.error):
        return None

    vocab = _meta_int(meta, "vocab_size", "model.vocab", "model.vocab_size") or embed_vocab
    layers = _meta_int(meta, "n_layers", "model.n_layers")
    if layers is None and layer_ids:
        layers = len(layer_ids)
    context = _meta_int(
        meta, "context_len", "model.context_len", "kv_cache.context_len"
    ) or kv_context
    bits_per_param = (
        round(weights_bytes * 8.0 / params, 2) if params and weights_bytes else None
    )
    return VtmInspect(
        size_bytes=int(stat.st_size),
        format_version=int(version),
        tensor_count=int(n_tensors),
        params=params or None,
        vocab=vocab or None,
        layers=layers,
        context=context,
        kv_bytes=kv_bytes or None,
        kv_in_file=kv_in_file,
        weights_bytes=weights_bytes or None,
        bits_per_param=bits_per_param,
        storage=_vtm_storage(meta, embed_kind, trit_storage, weight_kinds),
        source=_vtm_source(meta, vocab),
        meta=meta,
    )


def _model_card(
    backend: BackendSpec,
    model: ModelSpec,
    *,
    adapter: Any = None,
    last: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Карточка модели для ``GET /api/models``: только числа, без абсолютных путей.

    Для ``.vtm`` всё читается из файла (``inspect_vtm``); для Ollama диск и
    параметры неизвестны (``None`` → «—»), зато после ответа появляются
    ``size``/``size_vram`` из её ``/api/ps``.
    """
    card: dict[str, Any] = {
        "backend": backend.id,
        "backend_label": backend.label,
        "model": model.id,
        "label": model.label,
        "kind": backend.kind,
        "disk_bytes": None,
        "params": None,
        "vocab": None,
        "layers": None,
        "context": None,
        "kv_bytes": None,
        "kv_in_file": None,
        "storage": None,
        "source": None,
        "bits_per_param": None,
        "weights_bytes": None,
        "tensors": None,
        "format_version": None,
        "present": None,
        "loaded": None,
        "load": None,
        "cache_prefix": None,
        "last": dict(last) if last else None,
    }
    if backend.kind == "vtm":
        target = Path(model.target or model.id)
        card["present"] = target.is_file()
        info = inspect_vtm(target)
        if info is not None:
            card.update(
                {
                    "disk_bytes": info.size_bytes,
                    "params": info.params,
                    "vocab": info.vocab,
                    "layers": info.layers,
                    "context": info.context,
                    "kv_bytes": info.kv_bytes,
                    "kv_in_file": info.kv_in_file,
                    "storage": info.storage,
                    "source": info.source,
                    "bits_per_param": info.bits_per_param,
                    "weights_bytes": info.weights_bytes,
                    "tensors": info.tensor_count,
                    "format_version": info.format_version,
                }
            )
        elif card["present"]:
            # Файл есть, но это не VTM1: размер всё равно честный (stat),
            # остальное неизвестно — в UI «—», а не выдуманный ноль.
            try:
                card["disk_bytes"] = int(target.stat().st_size)
            except OSError:  # pragma: no cover - файл исчез между проверками
                card["present"] = False
    else:
        card["present"] = True
        card["source"] = "Ollama (метрики — из /api/ps)"

    loaded = getattr(adapter, "loaded", None)
    if isinstance(loaded, bool):
        card["loaded"] = loaded
    load_stats = getattr(adapter, "load_stats", None)
    if isinstance(load_stats, Mapping):
        card["load"] = dict(load_stats)
    prefix = getattr(adapter, "cache_prefix", None)
    if isinstance(prefix, bool):
        card["cache_prefix"] = prefix
    return card


def _model_cards(server: "ChatHTTPServer") -> dict[str, Any]:
    """Карточки всех моделей реестра; последние ответы — из памяти сервера."""
    cards: list[dict[str, Any]] = []
    for backend in server.registry:
        for model in backend.models:
            key = (backend.id, model.id)
            cards.append(
                _model_card(
                    backend,
                    model,
                    adapter=server.adapters.get(key),
                    last=server.last_answers.get(key),
                )
            )
    return {"models": cards, "baseline": server.start_memory, "max_tokens": server.max_tokens}


def _prompt_prefix(
    server: "ChatHTTPServer", key: tuple[str, str], prompt_text: str
) -> dict[str, Any]:
    """Оценка попадания в кэш префикса ВТМ по общему началу промптов.

    C-рантайм (``vtm_api_*``) не отдаёт счётчик попаданий, поэтому точного
    числа нет: считаем длину общего префикса текущего и прошлого промпта.
    Кэш переиспользует промпт прошлого запроса без последнего токена —
    значит попадание возможно, когда прошлый промпт целиком входит в текущий.
    Пометка ``estimated`` в ответе обязательна: это не измерение рантайма.
    """
    with server.state_lock:
        previous = server.prompts.get(key)
        shared = 0
        if previous is not None:
            limit = min(len(previous), len(prompt_text))
            while shared < limit and previous[shared] == prompt_text[shared]:
                shared += 1
        server.prompts[key] = prompt_text
    adapter = server.adapters.get(key)
    enabled = getattr(adapter, "cache_prefix", None)
    if not isinstance(enabled, bool):
        return {
            "enabled": None,
            "hit": None,
            "shared_chars": shared,
            "previous_chars": len(previous) if previous is not None else None,
            "estimated": True,
        }
    hit = None
    if enabled and previous is not None:
        hit = shared >= max(1, len(previous) - 1)
    return {
        "enabled": enabled,
        "hit": hit,
        "shared_chars": shared,
        "previous_chars": len(previous) if previous is not None else None,
        "estimated": True,
    }


# --------------------------------------------------------------------------- #
# Вспомогательные функции
# --------------------------------------------------------------------------- #


class ClientError(Exception):
    """Ошибка клиента: HTTP-статус, код для API и человеческий текст."""

    def __init__(self, message: str, *, status: int = 400, code: str = "invalid_request"):
        super().__init__(message)
        self.status = int(status)
        self.code = code
        self.message = message


def _as_int(value: Any) -> int:
    """Неотрицательное целое из ``usage``; bool и мусор — 0."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float) and value.is_integer():
        return max(0, int(value))
    return 0


def _short_error(exc: BaseException, limit: int = 700) -> str:
    """Текст ошибки без трейсбека, ограниченный по длине."""
    text = str(exc).strip() or type(exc).__name__
    return text if len(text) <= limit else text[: limit - 1] + "…"


def load_system_prompt() -> str:
    """Системный промпт EVA32 из ``prompts/system.assistant.md`` (или "")."""
    try:
        return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _registry_json(
    registry: Sequence[BackendSpec], *, max_tokens: int
) -> dict[str, Any]:
    """Публичное описание реестра: без путей, URL и токенизаторов."""
    backends = [
        {
            "id": backend.id,
            "label": backend.label,
            "models": [
                {"id": model.id, "label": model.label} for model in backend.models
            ],
        }
        for backend in registry
    ]
    by_id = {backend.id: backend for backend in registry}
    left = by_id.get("vtm") or (registry[0] if registry else None)
    right = by_id.get("ollama") or (registry[-1] if registry else None)

    def default_of(spec: BackendSpec | None) -> dict[str, str] | None:
        if spec is None or not spec.models:
            return None
        return {"backend": spec.id, "model": spec.models[0].id}

    return {
        "backends": backends,
        "defaults": {"left": default_of(left), "right": default_of(right)},
        "max_tokens": max_tokens,
    }


def _validate_payload(
    server: "ChatHTTPServer", payload: Any
) -> tuple[str, bool, int, list[dict[str, Any]]]:
    """Проверить запрос и вернуть ``(вопрос, системный, max_tokens, стороны)``.

    Всё, что приходит от клиента, сверяется с реестром сервера: произвольного
    URL, бэкенда или модели в запросе быть не может.
    """
    if not isinstance(payload, Mapping):
        raise ClientError("Тело запроса должно быть JSON-объектом.")

    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ClientError("Поле question: нужен непустой текст вопроса.")
    question = question.strip()
    if len(question) > MAX_QUESTION_CHARS:
        raise ClientError(
            "Вопрос длиннее {} символов — сократите его.".format(MAX_QUESTION_CHARS),
            status=413,
            code="question_too_large",
        )

    use_system = payload.get("use_system", True)
    if not isinstance(use_system, bool):
        raise ClientError("Поле use_system: ожидалось true или false.")

    raw_max = payload.get("max_tokens", server.max_tokens)
    if isinstance(raw_max, bool) or not isinstance(raw_max, int):
        raise ClientError("Поле max_tokens: ожидалось целое число.")
    if raw_max < 1 or raw_max > 1024:
        raise ClientError("Поле max_tokens: допустимо от 1 до 1024.")
    max_tokens = raw_max

    raw_sides = payload.get("sides")
    if not isinstance(raw_sides, list) or len(raw_sides) != 2:
        raise ClientError(
            "Поле sides: нужен ровно список из двух моделей — по одной на колонку."
        )

    catalog = {backend.id: backend for backend in server.registry}
    sides: list[dict[str, Any]] = []
    for index, raw_side in enumerate(raw_sides):
        if not isinstance(raw_side, Mapping):
            raise ClientError("sides[{}]: ожидался объект.".format(index))
        backend_id = raw_side.get("backend")
        model_id = raw_side.get("model")
        if not isinstance(backend_id, str) or backend_id not in catalog:
            raise ClientError(
                "sides[{}].backend: неизвестный бэкенд {!r}; разрешены: {}.".format(
                    index, backend_id, ", ".join(sorted(catalog))
                )
            )
        backend = catalog[backend_id]
        allowed = {model.id: model for model in backend.models}
        if not isinstance(model_id, str) or model_id not in allowed:
            raise ClientError(
                "sides[{}].model: неизвестная модель {!r} для бэкенда {}; "
                "разрешены: {}.".format(
                    index,
                    model_id,
                    backend_id,
                    ", ".join(sorted(allowed)) or "нет",
                )
            )
        adapter = server.adapters.get((backend_id, model_id))
        if adapter is None:
            raise ClientError(
                "sides[{}]: адаптер для {} / {} не собран на сервере.".format(
                    index, backend_id, model_id
                ),
                status=503,
                code="backend_unavailable",
            )
        sides.append(
            {
                "backend": backend_id,
                "backend_label": backend.label,
                "model": model_id,
                "model_label": allowed[model_id].label,
                "adapter": adapter,
            }
        )
    return question, use_system, max_tokens, sides


# --------------------------------------------------------------------------- #
# Работа одной колонки
# --------------------------------------------------------------------------- #


class _SideDone:
    """Внутренний маркер очереди: колонка закончила (в SSE не уходит)."""

    def __init__(self, side: int) -> None:
        self.side = side


def _run_side(
    index: int,
    side: Mapping[str, Any],
    messages: Sequence[Mapping[str, str]],
    max_tokens: int,
    started: float,
    events: "queue.Queue[Any]",
    stop: threading.Event,
    server: Any = None,
    prompt_text: str = "",
) -> None:
    """Сгенерировать одну колонку и сложить события в очередь.

    Пишет в очередь только этот поток; HTTP-поток читает и отправляет SSE.
    Так две модели работают параллельно, а сокет не делят между потоками.

    Здесь же снимаются метрики ответа (задача 0145): TTFT и он же prefill
    (время до первого токена), декод = остаток, причина остановки, был ли
    первый токен холодным (модель грузилась в этом запросе), попадание в кэш
    префикса ВТМ и ``/api/ps`` для Ollama.
    """
    adapter = side["adapter"]
    key = (side["backend"], side["model"])
    loaded_before = getattr(adapter, "loaded", None)
    cold = (not loaded_before) if isinstance(loaded_before, bool) else None
    cache_prefix = None
    if server is not None and getattr(adapter, "kind", "") == "vtm":
        cache_prefix = _prompt_prefix(server, key, prompt_text)
    events.put(
        {
            "type": "start",
            "side": index,
            "backend": side["backend"],
            "backend_label": side["backend_label"],
            "model": side["model"],
            "model_label": side["model_label"],
            "cold": cold,
        }
    )
    first_token_ms: int | None = None
    usage: dict[str, Any] = {}
    finish_reason = ""
    char_count = 0
    completed = False
    try:
        iterator = iter(adapter.stream(messages, max_tokens=max_tokens))
        try:
            for piece in iterator:
                if stop.is_set():
                    break
                text = piece.text if isinstance(piece, StreamPiece) else str(piece)
                if text:
                    char_count += len(text)
                    if first_token_ms is None:
                        first_token_ms = max(
                            0, int((time.monotonic() - started) * 1000)
                        )
                    events.put({"type": "delta", "side": index, "text": text})
                if isinstance(piece, StreamPiece) and piece.done:
                    usage = dict(piece.usage or {})
                    finish_reason = piece.finish_reason or finish_reason
                    break
            completed = True
        finally:
            closer = getattr(iterator, "close", None)
            if callable(closer):
                closer()
    except BaseException as exc:  # noqa: BLE001 - ошибка одной колонки не рвёт другую
        logger.warning("колонка %d (%s): ошибка: %s", index, side["model"], exc)
        if not stop.is_set():
            events.put(
                {
                    "type": "error",
                    "side": index,
                    "backend": side["backend"],
                    "model": side["model"],
                    "message": _short_error(exc),
                    "elapsed_ms": max(
                        0, int((time.monotonic() - started) * 1000)
                    ),
                }
            )
    else:
        if completed and not stop.is_set():
            elapsed_ms = max(1, int((time.monotonic() - started) * 1000))
            prompt_tokens = _as_int(usage.get("prompt_tokens"))
            completion_tokens = _as_int(usage.get("completion_tokens"))
            estimated = False
            if completion_tokens <= 0 and char_count > 0:
                # Провайдер не прислал usage: честно помечаем оценку, а не
                # выдаём её за точный счёт (у ВТМ счёт точный по id).
                completion_tokens = max(1, char_count // 4)
                estimated = True
            # Скорость декода, как в ``tools/chat.py``: токены после первого,
            # а не вместе с ожиданием prefill. Один токен/нет TTFT — по всему
            # времени ответа, иначе делить не на что.
            decode_ms = elapsed_ms - (first_token_ms or 0)
            speed_ms = (
                decode_ms
                if completion_tokens > 1 and decode_ms > 0
                else elapsed_ms
            )
            tokens_per_second = (
                round(completion_tokens / (speed_ms / 1000.0), 2)
                if completion_tokens > 0
                else 0.0
            )
            decode_speed = (
                round(completion_tokens / (decode_ms / 1000.0), 2)
                if completion_tokens > 1 and decode_ms > 0
                else tokens_per_second
            )
            # Доля от времени ответа: prefill — до первого токена (вместе с
            # загрузкой модели, если она была холодной), декод — после.
            prefill_pct = (
                round((first_token_ms or 0) * 100.0 / elapsed_ms, 1)
                if first_token_ms is not None
                else None
            )
            ollama_ps = None
            probe = getattr(adapter, "probe_ps", None)
            if callable(probe):
                try:
                    ollama_ps = probe(side["model"])
                except Exception:  # noqa: BLE001 - чужой сервис не должен рвать ответ
                    ollama_ps = None
            load_stats = getattr(adapter, "load_stats", None)
            if isinstance(cache_prefix, dict) and cache_prefix.get("enabled") is None:
                # Первый ответ мог сам загрузить модель: настройка кэша стала
                # известна только теперь — уточняем попадание, не выдумывая.
                enabled_now = getattr(adapter, "cache_prefix", None)
                if isinstance(enabled_now, bool):
                    cache_prefix["enabled"] = enabled_now
                    previous_chars = cache_prefix.get("previous_chars")
                    if enabled_now and previous_chars:
                        cache_prefix["hit"] = (
                            cache_prefix.get("shared_chars", 0)
                            >= max(1, int(previous_chars) - 1)
                        )
            metrics = {
                "elapsed_ms": elapsed_ms,
                "first_token_ms": first_token_ms,
                "prefill_ms": first_token_ms,
                "prefill_pct": prefill_pct,
                "decode_ms": max(0, decode_ms),
                "tokens_per_second": tokens_per_second,
                "decode_tokens_per_second": decode_speed,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "estimated": estimated,
                "finish_reason": finish_reason or None,
                "first_token": (
                    "холодный" if cold else "тёплый"
                ) if cold is not None else None,
                "cold": cold,
                "cache_prefix": cache_prefix,
                "load": dict(load_stats) if isinstance(load_stats, Mapping) else None,
                "ollama": ollama_ps,
            }
            if server is not None:
                with server.state_lock:
                    server.last_answers[key] = dict(metrics)
            events.put(
                {
                    "type": "done",
                    "side": index,
                    "backend": side["backend"],
                    "model": side["model"],
                    **metrics,
                }
            )
    finally:
        # Маркер — всегда последним: HTTP-поток закрывает стрим, только
        # когда обе колонки прислали и done/error, и маркер.
        events.put(_SideDone(index))


# --------------------------------------------------------------------------- #
# HTML-страница (CSS/JS встроены, внешних ссылок нет)
# --------------------------------------------------------------------------- #

PAGE_HTML = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EVA32 — два ответа рядом</title>
<style>
  :root {
    color-scheme: light dark;
    --bg: #f4f5f7;
    --card: #ffffff;
    --text: #1b1f24;
    --muted: #667085;
    --border: #d9dee7;
    --accent: #2457d6;
    --accent-2: #0f8f6f;
    --danger: #b42318;
    --code-bg: #f8f9fb;
    --shadow: 0 1px 2px rgba(16, 24, 40, .06), 0 8px 24px rgba(16, 24, 40, .04);
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #101318;
      --card: #171b22;
      --text: #e7ebf2;
      --muted: #98a2b3;
      --border: #2a313c;
      --accent: #7ea2ff;
      --accent-2: #4fd1b0;
      --danger: #ff8f85;
      --code-bg: #11151b;
      --shadow: 0 1px 2px rgba(0, 0, 0, .35), 0 10px 30px rgba(0, 0, 0, .25);
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
          "Helvetica Neue", Arial, sans-serif;
  }
  header { max-width: 1440px; margin: 0 auto; padding: 26px 22px 6px; }
  h1 { margin: 0 0 4px; font-size: 22px; letter-spacing: -.02em; }
  .sub { margin: 0; color: var(--muted); font-size: 14px; }
  main { max-width: 1440px; margin: 0 auto; padding: 14px 22px 40px; }
  .composer {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 14px; padding: 16px; box-shadow: var(--shadow);
  }
  label.field { display: block; font-weight: 600; margin-bottom: 6px; }
  textarea {
    width: 100%; min-height: 84px; resize: vertical; padding: 11px 12px;
    border: 1px solid var(--border); border-radius: 10px;
    background: var(--code-bg); color: var(--text);
    font: inherit;
  }
  textarea:focus, select:focus, button:focus-visible {
    outline: 2px solid var(--accent); outline-offset: 1px;
  }
  .controls {
    display: flex; align-items: center; gap: 14px; flex-wrap: wrap;
    margin-top: 12px;
  }
  .check { display: inline-flex; align-items: center; gap: 7px; color: var(--muted); }
  .check input { width: 16px; height: 16px; accent-color: var(--accent); }
  button {
    border: 0; border-radius: 9px; padding: 10px 20px; font: inherit;
    font-weight: 650; color: #fff; background: var(--accent); cursor: pointer;
  }
  button:disabled { opacity: .55; cursor: default; }
  #status { color: var(--muted); font-size: 13px; }
  #status.error { color: var(--danger); }
  .grid {
    display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    gap: 16px; margin-top: 16px;
  }
  @media (max-width: 860px) { .grid { grid-template-columns: 1fr; } }
  .panel {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 14px; box-shadow: var(--shadow);
    display: flex; flex-direction: column; min-height: 320px; overflow: hidden;
  }
  .panel.streaming { border-color: var(--accent); }
  .panel.failed { border-color: var(--danger); }
  .panel-head {
    display: flex; align-items: flex-start; justify-content: space-between;
    gap: 10px; padding: 12px 14px; border-bottom: 1px solid var(--border);
    flex-wrap: wrap;
  }
  .pickers { display: flex; gap: 8px; flex-wrap: wrap; }
  select {
    max-width: 220px; padding: 7px 9px; border: 1px solid var(--border);
    border-radius: 8px; background: var(--code-bg); color: var(--text);
    font: inherit; font-size: 13px;
  }
  .stats {
    display: flex; gap: 10px; flex-wrap: wrap; color: var(--muted);
    font-size: 12px; font-variant-numeric: tabular-nums;
  }
  .stats b { color: var(--text); font-weight: 650; }
  .answer {
    margin: 0; padding: 15px; flex: 1; overflow: auto; max-height: 58vh;
    white-space: pre-wrap; overflow-wrap: anywhere; background: transparent;
    color: var(--text); font: inherit;
  }
  .answer:empty::before {
    content: "Ответ появится здесь…"; color: var(--muted); font-style: italic;
  }
  .answer .error-text {
    color: var(--danger); font-style: normal; white-space: pre-wrap;
  }
  .foot { padding: 0 14px 12px; color: var(--muted); font-size: 12px; }
  .metrics {
    display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    gap: 16px; margin-top: 16px;
  }
  @media (max-width: 860px) { .metrics { grid-template-columns: 1fr; } }
  .metric-col {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 14px; padding: 12px 14px; box-shadow: var(--shadow);
    min-width: 0;
  }
  .metric-col h2 {
    margin: 0 0 8px; font-size: 12px; font-weight: 700; color: var(--muted);
    letter-spacing: .04em; text-transform: uppercase;
  }
  .kv {
    display: grid; grid-template-columns: auto minmax(0, 1fr);
    gap: 2px 10px; margin: 0 0 10px; font-size: 12.5px;
    font-variant-numeric: tabular-nums;
  }
  .kv dt { color: var(--muted); }
  .kv dd { margin: 0; overflow-wrap: anywhere; }
  .kv .sub-head {
    grid-column: 1 / -1; margin-top: 6px; color: var(--muted); font-weight: 700;
  }
  .sysrow {
    grid-column: 1 / -1; display: flex; flex-wrap: wrap; gap: 6px 18px;
    background: var(--card); border: 1px solid var(--border);
    border-radius: 14px; padding: 10px 14px; box-shadow: var(--shadow);
    color: var(--muted); font-size: 12.5px; font-variant-numeric: tabular-nums;
  }
  .sysrow b { color: var(--text); font-weight: 650; }
  .metric-note {
    grid-column: 1 / -1; margin: 0; color: var(--muted); font-size: 11.5px;
  }
  @media (prefers-reduced-motion: no-preference) {
    .panel { transition: border-color .15s ease; }
  }
</style>
</head>
<body>
<header>
  <h1>EVA32 · два ответа рядом</h1>
  <p class="sub">Один вопрос — локальная тритовая модель и Qwen. Время и скорость считает сервер.</p>
</header>
<main>
  <section class="composer">
    <label class="field" for="question">Вопрос</label>
    <textarea id="question" rows="3" placeholder="Например: напиши заголовок для поста о весенней скидке на стрижку"></textarea>
    <div class="controls">
      <label class="check"><input type="checkbox" id="use-system" checked> системный промпт EVA32</label>
      <button id="ask" type="button">Спросить</button>
      <span id="status" role="status"></span>
    </div>
  </section>

  <section class="grid">
    <article class="panel" id="panel-0">
      <div class="panel-head">
        <div class="pickers">
          <select id="backend-0" aria-label="Бэкенд слева"></select>
          <select id="model-0" aria-label="Модель слева"></select>
        </div>
        <div class="stats">
          <span>время: <b id="time-0">—</b></span>
          <span>ток/с: <b id="speed-0">—</b></span>
          <span>токенов: <b id="tokens-0">—</b></span>
        </div>
      </div>
      <pre class="answer" id="answer-0" aria-live="polite"></pre>
      <div class="foot" id="foot-0"></div>
    </article>

    <article class="panel" id="panel-1">
      <div class="panel-head">
        <div class="pickers">
          <select id="backend-1" aria-label="Бэкенд справа"></select>
          <select id="model-1" aria-label="Модель справа"></select>
        </div>
        <div class="stats">
          <span>время: <b id="time-1">—</b></span>
          <span>ток/с: <b id="speed-1">—</b></span>
          <span>токенов: <b id="tokens-1">—</b></span>
        </div>
      </div>
      <pre class="answer" id="answer-1" aria-live="polite"></pre>
      <div class="foot" id="foot-1"></div>
    </article>
  </section>

  <section class="metrics" id="metrics" aria-label="Метрики моделей и системы">
    <article class="metric-col">
      <h2>Модель · <span id="mc-title-0">—</span></h2>
      <dl class="kv">
        <dt>файл на диске</dt><dd id="mc-size-0">—</dd>
        <dt>параметры</dt><dd id="mc-params-0">—</dd>
        <dt>контекст</dt><dd id="mc-ctx-0">—</dd>
        <dt>KV-кэш</dt><dd id="mc-kv-0">—</dd>
        <dt>хранение</dt><dd id="mc-storage-0">—</dd>
        <dt>источник</dt><dd id="mc-source-0">—</dd>
        <dt>словарь</dt><dd id="mc-vocab-0">—</dd>
        <dt>слоёв</dt><dd id="mc-layers-0">—</dd>
        <dt>бит / параметр</dt><dd id="mc-bpp-0">—</dd>
        <dt>RSS до → после</dt><dd id="mc-rss-0">—</dd>
        <dt>пик RSS</dt><dd id="mc-peak-0">—</dd>
        <dt>резидентная копия</dt><dd id="mc-ps-0">—</dd>
      </dl>
      <dl class="kv">
        <dt class="sub-head">последний ответ</dt>
        <dt>TTFT</dt><dd id="ma-ttft-0">—</dd>
        <dt>prefill</dt><dd id="ma-prefill-0">—</dd>
        <dt>декод</dt><dd id="ma-decode-0">—</dd>
        <dt>ток/с декода</dt><dd id="ma-dtps-0">—</dd>
        <dt>всего</dt><dd id="ma-total-0">—</dd>
        <dt>prompt / completion</dt><dd id="ma-tokens-0">—</dd>
        <dt>остановка</dt><dd id="ma-finish-0">—</dd>
        <dt>первый токен</dt><dd id="ma-first-0">—</dd>
        <dt>кэш префикса</dt><dd id="ma-cache-0">—</dd>
      </dl>
    </article>

    <article class="metric-col">
      <h2>Модель · <span id="mc-title-1">—</span></h2>
      <dl class="kv">
        <dt>файл на диске</dt><dd id="mc-size-1">—</dd>
        <dt>параметры</dt><dd id="mc-params-1">—</dd>
        <dt>контекст</dt><dd id="mc-ctx-1">—</dd>
        <dt>KV-кэш</dt><dd id="mc-kv-1">—</dd>
        <dt>хранение</dt><dd id="mc-storage-1">—</dd>
        <dt>источник</dt><dd id="mc-source-1">—</dd>
        <dt>словарь</dt><dd id="mc-vocab-1">—</dd>
        <dt>слоёв</dt><dd id="mc-layers-1">—</dd>
        <dt>бит / параметр</dt><dd id="mc-bpp-1">—</dd>
        <dt>RSS до → после</dt><dd id="mc-rss-1">—</dd>
        <dt>пик RSS</dt><dd id="mc-peak-1">—</dd>
        <dt>резидентная копия</dt><dd id="mc-ps-1">—</dd>
      </dl>
      <dl class="kv">
        <dt class="sub-head">последний ответ</dt>
        <dt>TTFT</dt><dd id="ma-ttft-1">—</dd>
        <dt>prefill</dt><dd id="ma-prefill-1">—</dd>
        <dt>декод</dt><dd id="ma-decode-1">—</dd>
        <dt>ток/с декода</dt><dd id="ma-dtps-1">—</dd>
        <dt>всего</dt><dd id="ma-total-1">—</dd>
        <dt>prompt / completion</dt><dd id="ma-tokens-1">—</dd>
        <dt>остановка</dt><dd id="ma-finish-1">—</dd>
        <dt>первый токен</dt><dd id="ma-first-1">—</dd>
        <dt>кэш префикса</dt><dd id="ma-cache-1">—</dd>
      </dl>
    </article>

    <p class="sysrow" id="sysrow">
      <span>load: <b id="sys-load">—</b></span>
      <span>RAM свободно: <b id="sys-ram">—</b></span>
      <span>диск свободно: <b id="sys-disk">—</b></span>
      <span>CPU запроса: <b id="sys-cpu">—</b></span>
      <span>RSS процесса: <b id="sys-rss">—</b></span>
      <span>пик RSS: <b id="sys-peak">—</b></span>
      <span>VTM_THREADS: <b id="sys-threads">—</b></span>
      <span>ядер: <b id="sys-cpus">—</b></span>
    </p>
    <p class="metric-note">Размер файла — stat; параметры и KV-кэш — сумма по таблице
      тензоров .vtm; RSS — дельта процесса chat_server до и после загрузки
      (модель живёт в этом процессе через ctypes); load/RAM/диск — система.
      «—» значит «недоступно», а не ноль. Кэш префикса ВТМ — оценка по общему
      префиксу промпта: счётчик попаданий C-рантайм не отдаёт. «Резидентная
      копия» — size/size_vram из /api/ps локальной Ollama.</p>
  </section>
</main>
<script>
(function () {
  "use strict";

  var catalog = [];
  var defaults = { left: null, right: null };
  var controller = null;
  var running = false;
  var finishedSides = 0;

  function byId(id) { return document.getElementById(id); }
  function backendOf(id) {
    for (var i = 0; i < catalog.length; i += 1) {
      if (catalog[i].id === id) { return catalog[i]; }
    }
    return null;
  }
  function modelOf(backend, id) {
    if (!backend) { return null; }
    for (var i = 0; i < backend.models.length; i += 1) {
      if (backend.models[i].id === id) { return backend.models[i]; }
    }
    return null;
  }

  function fillBackends(side) {
    var select = byId("backend-" + side);
    select.textContent = "";
    catalog.forEach(function (backend) {
      var option = document.createElement("option");
      option.value = backend.id;
      option.textContent = backend.label;
      if (!backend.models.length) { option.disabled = true; }
      select.appendChild(option);
    });
  }

  function fillModels(side, preferred) {
    var backendSelect = byId("backend-" + side);
    var modelSelect = byId("model-" + side);
    var backend = backendOf(backendSelect.value);
    modelSelect.textContent = "";
    if (!backend) { return; }
    backend.models.forEach(function (model) {
      var option = document.createElement("option");
      option.value = model.id;
      option.textContent = model.label;
      modelSelect.appendChild(option);
    });
    if (preferred && modelOf(backend, preferred)) {
      modelSelect.value = preferred;
    }
  }

  function setSide(side, spec) {
    var backendSelect = byId("backend-" + side);
    if (spec && backendOf(spec.backend)) {
      backendSelect.value = spec.backend;
    } else if (catalog.length) {
      backendSelect.value = catalog[0].id;
    }
    fillModels(side, spec ? spec.model : null);
  }

  function resetPanels() {
    finishedSides = 0;
    [0, 1].forEach(function (side) {
      byId("panel-" + side).className = "panel";
      byId("answer-" + side).textContent = "";
      byId("time-" + side).textContent = "—";
      byId("speed-" + side).textContent = "—";
      byId("tokens-" + side).textContent = "—";
      byId("foot-" + side).textContent = "";
      ["ttft", "prefill", "decode", "dtps", "total", "tokens", "finish",
       "first", "cache"].forEach(function (name) {
        setText("ma-" + name + "-" + side, "—");
      });
    });
  }

  function setStatus(text, isError) {
    var node = byId("status");
    node.textContent = text;
    node.className = isError ? "error" : "";
  }

  function formatSeconds(ms) {
    if (typeof ms !== "number" || ms < 0) { return "—"; }
    return (ms / 1000).toFixed(2) + " с";
  }

  function fmtBytes(value) {
    if (typeof value !== "number" || !isFinite(value) || value < 0) { return "—"; }
    var units = ["Б", "КиБ", "МиБ", "ГиБ", "ТиБ"];
    var index = 0;
    var size = value;
    while (size >= 1024 && index < units.length - 1) {
      size /= 1024;
      index += 1;
    }
    return (index === 0 ? size.toFixed(0) : size.toFixed(1)) + " " + units[index];
  }

  function fmtSignedBytes(value) {
    if (typeof value !== "number" || !isFinite(value)) { return "—"; }
    return (value >= 0 ? "+" : "−") + fmtBytes(Math.abs(value));
  }

  function fmtNum(value) {
    if (typeof value !== "number" || !isFinite(value)) { return "—"; }
    return value.toLocaleString("ru-RU");
  }

  function fmtMs(value) {
    if (typeof value !== "number" || !isFinite(value) || value < 0) { return "—"; }
    if (value < 1000) { return value.toFixed(0) + " мс"; }
    return (value / 1000).toFixed(2) + " с";
  }

  function fmtFreeOf(free, total) {
    if (typeof free !== "number" && typeof total !== "number") { return "—"; }
    return fmtBytes(free) + " из " + fmtBytes(total);
  }

  function setText(id, text) {
    var node = byId(id);
    if (node) { node.textContent = text; }
  }

  function finishText(reason) {
    if (reason === "stop" || reason === "eos") { return "EOS"; }
    if (reason === "length") { return "лимит токенов"; }
    return "—";
  }

  function cacheText(info) {
    if (!info) { return "—"; }
    if (info.enabled === false) { return "выключен"; }
    if (info.enabled === null || info.enabled === undefined) { return "—"; }
    var shared = fmtNum(info.shared_chars) + " симв.";
    if (info.hit === true) { return "да · общий префикс " + shared + " (оценка)"; }
    if (info.hit === false) { return "нет · общий префикс " + shared + " (оценка)"; }
    return "включён, попадание неизвестно";
  }

  function fillSideMetrics(side, event) {
    setText("ma-ttft-" + side, fmtMs(event.first_token_ms));
    var prefill = fmtMs(event.prefill_ms);
    if (prefill !== "—" && typeof event.prefill_pct === "number") {
      prefill += " · " + event.prefill_pct.toFixed(1) + "%";
    }
    setText("ma-prefill-" + side, prefill);
    setText("ma-decode-" + side, fmtMs(event.decode_ms));
    setText("ma-dtps-" + side, fmtNum(event.decode_tokens_per_second));
    setText("ma-total-" + side, fmtMs(event.elapsed_ms));
    setText(
      "ma-tokens-" + side,
      fmtNum(event.prompt_tokens) + " / " + fmtNum(event.completion_tokens) +
        (event.estimated ? "*" : "")
    );
    setText("ma-finish-" + side, finishText(event.finish_reason));
    setText("ma-first-" + side, event.first_token || "—");
    setText("ma-cache-" + side, cacheText(event.cache_prefix));
  }

  function fillCard(index, card) {
    var names = ["size", "params", "ctx", "kv", "storage", "source", "vocab",
                 "layers", "bpp", "rss", "peak", "ps"];
    if (!card) {
      setText("mc-title-" + index, "—");
      names.forEach(function (name) { setText("mc-" + name + "-" + index, "—"); });
      return;
    }
    setText("mc-title-" + index, card.label || card.model || "—");
    setText("mc-size-" + index, fmtBytes(card.disk_bytes));
    setText("mc-params-" + index, fmtNum(card.params));
    setText("mc-ctx-" + index,
      typeof card.context === "number" ? fmtNum(card.context) + " ток." : "—");
    setText("mc-kv-" + index, fmtBytes(card.kv_bytes));
    setText("mc-storage-" + index, card.storage || "—");
    setText("mc-source-" + index, card.source || "—");
    setText("mc-vocab-" + index, fmtNum(card.vocab));
    setText("mc-layers-" + index, fmtNum(card.layers));
    setText("mc-bpp-" + index,
      typeof card.bits_per_param === "number"
        ? card.bits_per_param.toFixed(2) + " бит" : "—");
    var load = card.load || {};
    var rss = "—";
    if (typeof load.rss_before === "number" || typeof load.rss_after === "number") {
      rss = fmtBytes(load.rss_before) + " → " + fmtBytes(load.rss_after);
      if (typeof load.rss_delta === "number") {
        rss += " (" + fmtSignedBytes(load.rss_delta) + ")";
      }
    }
    setText("mc-rss-" + index, rss);
    setText("mc-peak-" + index, fmtBytes(load.peak));
    // Резидентная копия у Ollama — из её /api/ps; у ВТМ такого источника нет
    // (модель в этом процессе, её память — строка RSS выше).
    var ps = card.last && card.last.ollama;
    var psText = "—";
    if (ps && typeof ps.size === "number") {
      psText = fmtBytes(ps.size);
      if (typeof ps.size_vram === "number") {
        psText += " · VRAM " + fmtBytes(ps.size_vram);
      }
    }
    setText("mc-ps-" + index, psText);
    if (card.last) { fillSideMetrics(index, card.last); }
  }

  function fillSystem(snapshot) {
    if (!snapshot) { return; }
    var load = snapshot.load;
    setText("sys-load", Array.isArray(load)
      ? load.map(function (value) { return value.toFixed(2); }).join(" / ")
      : "—");
    var ram = snapshot.ram || {};
    setText("sys-ram", fmtFreeOf(ram.free, ram.total));
    var disk = snapshot.disk || {};
    setText("sys-disk", fmtFreeOf(disk.free, disk.total));
    var process = snapshot.process || {};
    setText("sys-rss", fmtBytes(process.rss));
    setText("sys-peak", fmtBytes(process.peak));
    setThreads(snapshot.vtm_threads);
    setText("sys-cpus", fmtNum(snapshot.cpu_count));
  }

  function setThreads(info) {
    if (!info) { setText("sys-threads", "—"); return; }
    if (info.runtime) { setText("sys-threads", info.runtime); return; }
    if (info.config) { setText("sys-threads", info.config + " (конфиг)"); return; }
    setText("sys-threads", "по умолчанию");
  }

  function fillEndMetrics(metrics) {
    if (!metrics) { return; }
    var cpu = "—";
    if (typeof metrics.cpu_ms === "number") {
      cpu = fmtMs(metrics.cpu_ms);
      if (typeof metrics.cpu_cores === "number") {
        cpu += " · " + metrics.cpu_cores.toFixed(2) + " ядра";
      }
    }
    setText("sys-cpu", cpu);
    var process = metrics.process || {};
    setText("sys-rss", fmtBytes(process.rss));
    setText("sys-peak", fmtBytes(process.peak));
    var load = metrics.load;
    setText("sys-load", Array.isArray(load)
      ? load.map(function (value) { return value.toFixed(2); }).join(" / ")
      : "—");
    var ram = metrics.ram || {};
    setText("sys-ram", fmtFreeOf(ram.free, ram.total));
    var disk = metrics.disk || {};
    setText("sys-disk", fmtFreeOf(disk.free, disk.total));
    setThreads(metrics.vtm_threads);
  }

  async function refreshModels() {
    try {
      var response = await fetch("/api/models",
        { headers: { "Accept": "application/json" } });
      if (!response.ok) { return; }
      var data = await response.json();
      var models = Array.isArray(data.models) ? data.models : [];
      [0, 1].forEach(function (side) {
        var backendId = byId("backend-" + side).value;
        var modelId = byId("model-" + side).value;
        var card = null;
        for (var i = 0; i < models.length; i += 1) {
          if (models[i].backend === backendId && models[i].model === modelId) {
            card = models[i];
            break;
          }
        }
        fillCard(side, card);
      });
    } catch (err) { /* панель метрик не критична для чата */ }
  }

  async function refreshSystem() {
    try {
      var response = await fetch("/api/system",
        { headers: { "Accept": "application/json" } });
      if (!response.ok) { return; }
      fillSystem(await response.json());
    } catch (err) { /* панель метрик не критична для чата */ }
  }

  function appendText(side, text) {
    var answer = byId("answer-" + side);
    answer.appendChild(document.createTextNode(text));
    if (answer.scrollHeight - answer.scrollTop - answer.clientHeight < 140) {
      answer.scrollTop = answer.scrollHeight;
    }
  }

  function appendError(side, message) {
    var answer = byId("answer-" + side);
    var node = document.createElement("span");
    node.className = "error-text";
    node.textContent = (answer.textContent ? "\\n\\n" : "") + "Ошибка: " + message;
    answer.appendChild(node);
  }

  function panelOf(side) {
    return byId("panel-" + side);
  }

  function handleEvent(event) {
    if (event.type === "end") {
      fillEndMetrics(event.metrics);
      refreshModels();
      return;
    }
    var side = event.side;
    if (side !== 0 && side !== 1) { return; }
    if (event.type === "start") {
      panelOf(side).className = "panel streaming";
      byId("foot-" + side).textContent =
        (event.backend_label || event.backend) + " · " +
        (event.model_label || event.model);
    } else if (event.type === "delta") {
      appendText(side, event.text || "");
    } else if (event.type === "done") {
      panelOf(side).className = "panel";
      byId("time-" + side).textContent = formatSeconds(event.elapsed_ms);
      var speed = typeof event.tokens_per_second === "number"
        ? event.tokens_per_second.toFixed(1) : "—";
      byId("speed-" + side).textContent = speed;
      var tokens = typeof event.completion_tokens === "number"
        ? String(event.completion_tokens) : "—";
      byId("tokens-" + side).textContent = tokens + (event.estimated ? "*" : "");
      var details = [];
      if (event.first_token_ms !== null && event.first_token_ms !== undefined) {
        details.push("первый токен " + formatSeconds(event.first_token_ms));
      }
      details.push("промпт " + (event.prompt_tokens || 0) + " ток.");
      if (event.estimated) { details.push("* токены оценены по тексту"); }
      byId("foot-" + side).textContent = details.join(" · ");
      fillSideMetrics(side, event);
      finishedSides += 1;
      if (finishedSides >= 2) { setStatus("готово", false); }
    } else if (event.type === "error") {
      panelOf(side).className = "panel failed";
      appendError(side, event.message || "неизвестная ошибка");
      byId("time-" + side).textContent = formatSeconds(event.elapsed_ms);
      byId("speed-" + side).textContent = "—";
      byId("tokens-" + side).textContent = "—";
      finishedSides += 1;
      if (finishedSides >= 2) { setStatus("готово, были ошибки", true); }
    }
  }

  function handleBlock(block) {
    var lines = block.split("\\n");
    for (var i = 0; i < lines.length; i += 1) {
      var line = lines[i];
      if (line.indexOf("data:") !== 0) { continue; }
      var payload = line.slice(5).trim();
      if (payload === "[DONE]") { return true; }
      try {
        handleEvent(JSON.parse(payload));
      } catch (err) {
        setStatus("не разобрала событие сервера", true);
      }
    }
    return false;
  }

  async function ask() {
    if (running) { return; }
    var question = byId("question").value.trim();
    if (!question) {
      setStatus("нужен вопрос", true);
      byId("question").focus();
      return;
    }
    var sides = [0, 1].map(function (side) {
      return {
        backend: byId("backend-" + side).value,
        model: byId("model-" + side).value
      };
    });
    if (!sides[0].model || !sides[1].model) {
      setStatus("выберите модель в обеих колонках", true);
      return;
    }

    if (controller) { controller.abort(); }
    controller = new AbortController();
    running = true;
    byId("ask").disabled = true;
    resetPanels();
    setStatus("думает…", false);

    try {
      var response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json", "Accept": "text/event-stream" },
        body: JSON.stringify({
          question: question,
          use_system: byId("use-system").checked,
          sides: sides
        }),
        signal: controller.signal
      });

      if (!response.ok) {
        var message = "HTTP " + response.status;
        try {
          var problem = await response.json();
          if (problem && problem.error && problem.error.message) {
            message = problem.error.message;
          }
        } catch (ignored) { /* тело не JSON — оставляем статус */ }
        setStatus(message, true);
        return;
      }

      var reader = response.body.getReader();
      var decoder = new TextDecoder("utf-8");
      var buffer = "";
      var sawDone = false;
      while (!sawDone) {
        var part = await reader.read();
        if (part.done) { break; }
        buffer += decoder.decode(part.value, { stream: true });
        var boundary = buffer.indexOf("\\n\\n");
        while (boundary >= 0) {
          var block = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 2);
          if (handleBlock(block)) { sawDone = true; break; }
          boundary = buffer.indexOf("\\n\\n");
        }
      }
      if (!sawDone) {
        setStatus("поток прервался", true);
      }
    } catch (err) {
      if (err && err.name === "AbortError") {
        setStatus("запрос отменён", false);
      } else {
        setStatus("сервер недоступен: " + (err && err.message ? err.message : err), true);
      }
    } finally {
      running = false;
      byId("ask").disabled = false;
    }
  }

  async function loadCatalog() {
    try {
      var response = await fetch("/api/backends", { headers: { "Accept": "application/json" } });
      if (!response.ok) { throw new Error("HTTP " + response.status); }
      var data = await response.json();
      catalog = Array.isArray(data.backends) ? data.backends : [];
      defaults = data.defaults || { left: null, right: null };
      [0, 1].forEach(function (side) {
        fillBackends(side);
        setSide(side, side === 0 ? defaults.left : defaults.right);
      });
      if (!catalog.length) {
        setStatus("на сервере нет доступных моделей", true);
      }
      refreshModels();
    } catch (err) {
      setStatus("не удалось загрузить список моделей: " + err.message, true);
    }
  }

  byId("ask").addEventListener("click", ask);
  byId("question").addEventListener("keydown", function (event) {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      ask();
    }
  });
  [0, 1].forEach(function (side) {
    byId("backend-" + side).addEventListener("change", function () {
      fillModels(side, null);
      refreshModels();
    });
    byId("model-" + side).addEventListener("change", function () {
      refreshModels();
    });
  });

  loadCatalog();
  refreshSystem();
})();
</script>
</body>
</html>
"""

CSP = (
    "default-src 'none'; "
    "style-src 'unsafe-inline'; "
    "script-src 'unsafe-inline'; "
    "connect-src 'self'; "
    "img-src data:; "
    "base-uri 'none'; "
    "form-action 'none'; "
    "frame-ancestors 'none'"
)


# --------------------------------------------------------------------------- #
# HTTP-сервер
# --------------------------------------------------------------------------- #


class ChatHTTPServer(ThreadingHTTPServer):
    """Сервер сравнения: реестр, адаптеры и системный промпт на всё время."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        registry: Sequence[BackendSpec],
        adapters: Mapping[tuple[str, str], Any],
        system_prompt: str,
        max_tokens: int,
    ) -> None:
        self.registry = tuple(registry)
        self.adapters = dict(adapters)
        self.system_prompt = system_prompt
        self.max_tokens = max(1, int(max_tokens))
        # Метрики (задача 0145): память на старте, последние ответы по моделям
        # и промпты — для оценки кэша префикса. Всё под одним замком.
        self.state_lock = threading.Lock()
        self.start_memory: dict[str, Any] = process_memory()
        self.last_answers: dict[tuple[str, str], dict[str, Any]] = {}
        self.prompts: dict[tuple[str, str], str] = {}
        super().__init__(server_address, ChatRequestHandler)


class ChatRequestHandler(BaseHTTPRequestHandler):
    """HTTP-слой: страница, реестр и SSE-стрим двух ответов."""

    protocol_version = "HTTP/1.1"
    server_version = "eva32-chat/1.0"
    #: Включается в ``_open_sse``: HTTP/1.1 — chunked, HTTP/1.0 — до закрытия.
    _sse_chunked = True

    # -- маршруты ------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - имя задано BaseHTTPRequestHandler
        path = urlsplit(self.path).path
        try:
            if path in ("/", "/index.html"):
                self._send_page()
            elif path == "/api/backends":
                self._send_json(
                    200,
                    _registry_json(
                        self.server.registry, max_tokens=self.server.max_tokens
                    ),
                )
            elif path == "/api/models":
                self._send_json(200, _model_cards(self.server))
            elif path == "/api/system":
                self._send_json(200, system_snapshot())
            elif path == "/health":
                self._send_json(200, {"status": "ok"})
            elif path == "/favicon.ico":
                self.send_response(204)
                self.send_header("Content-Length", "0")
                self.end_headers()
            elif path.startswith("/api/"):
                self._send_json(
                    404,
                    {"error": {"code": "not_found", "message": "Нет такого метода API."}},
                )
            else:
                self._send_text(404, "Страница не найдена.")
        except _CLIENT_GONE:
            self.close_connection = True
        except Exception:  # noqa: BLE001 - наружу только человеческий текст
            logger.exception("GET %s: ошибка", path)
            try:
                self._send_json(
                    500,
                    {
                        "error": {
                            "code": "server_error",
                            "message": "Внутренняя ошибка сервера.",
                        }
                    },
                )
            except _CLIENT_GONE:
                self.close_connection = True

    def do_POST(self) -> None:  # noqa: N802 - имя задано BaseHTTPRequestHandler
        path = urlsplit(self.path).path
        try:
            if path != "/api/chat":
                self._send_json(
                    404,
                    {
                        "error": {
                            "code": "not_found",
                            "message": "Нет такого метода API.",
                        }
                    },
                )
                return
            payload = self._read_json_body()
            question, use_system, max_tokens, sides = _validate_payload(
                self.server, payload
            )
            messages: list[dict[str, str]] = []
            if use_system and self.server.system_prompt:
                messages.append(
                    {"role": "system", "content": self.server.system_prompt}
                )
            messages.append({"role": "user", "content": question})
            self._stream_sides(sides, messages, max_tokens)
        except ClientError as exc:
            self._send_json(
                exc.status,
                {"error": {"code": exc.code, "message": exc.message}},
            )
        except _CLIENT_GONE:
            self.close_connection = True
        except Exception:  # noqa: BLE001 - трейсбек остаётся в логе сервера
            logger.exception("POST %s: ошибка", path)
            try:
                self._send_json(
                    500,
                    {
                        "error": {
                            "code": "server_error",
                            "message": "Внутренняя ошибка сервера.",
                        }
                    },
                )
            except _CLIENT_GONE:
                self.close_connection = True

    # -- тело запроса ---------------------------------------------------------

    def _read_json_body(self) -> Any:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise ClientError(
                "Не указан Content-Length: сервер не знает, сколько читать.",
                status=411,
                code="length_required",
            )
        try:
            length = int(raw_length)
        except (TypeError, ValueError):
            raise ClientError(
                "Некорректный Content-Length.", code="invalid_content_length"
            ) from None
        if length <= 0:
            raise ClientError("Пустое тело запроса.", code="empty_body")
        if length > MAX_BODY_BYTES:
            raise ClientError(
                "Тело запроса больше {} байт.".format(MAX_BODY_BYTES),
                status=413,
                code="payload_too_large",
            )
        body = self.rfile.read(length)
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError:
            raise ClientError(
                "Тело запроса должно быть в UTF-8.", code="invalid_encoding"
            ) from None
        if not text.strip():
            raise ClientError("Пустое тело запроса.", code="empty_body")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ClientError(
                "Тело запроса — не JSON: {}.".format(exc.msg),
                code="invalid_json",
            ) from None

    # -- обычные ответы -------------------------------------------------------

    def _security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")

    def _send_bytes(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self._security_headers()
        self.end_headers()
        if body:
            self.wfile.write(body)
            self.wfile.flush()

    def _send_page(self) -> None:
        body = PAGE_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Content-Security-Policy", CSP)
        self._security_headers()
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def _send_json(self, status: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        self._send_bytes(status, body, "application/json; charset=utf-8")

    def _send_text(self, status: int, text: str) -> None:
        self._send_bytes(
            status, text.encode("utf-8"), "text/plain; charset=utf-8"
        )

    # -- SSE ------------------------------------------------------------------

    def _open_sse(self) -> None:
        self._sse_chunked = self.request_version != "HTTP/1.0"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("X-Accel-Buffering", "no")
        self._security_headers()
        if self._sse_chunked:
            self.send_header("Transfer-Encoding", "chunked")
        else:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()

    def _write_sse(self, payload: bytes) -> bool:
        event = b"data: " + payload + b"\n\n"
        try:
            if self._sse_chunked:
                self.wfile.write(b"%x\r\n" % len(event) + event + b"\r\n")
            else:
                self.wfile.write(event)
            self.wfile.flush()
            return True
        except _CLIENT_GONE:
            self.close_connection = True
            return False

    def _send_sse_data(self, payload: Mapping[str, Any]) -> bool:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        return self._write_sse(body)

    def _send_sse_done(self) -> bool:
        if not self._write_sse(b"[DONE]"):
            return False
        if self._sse_chunked:
            try:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except _CLIENT_GONE:
                self.close_connection = True
                return False
        return True

    def _stream_sides(
        self,
        sides: Sequence[Mapping[str, Any]],
        messages: Sequence[Mapping[str, str]],
        max_tokens: int,
    ) -> None:
        events: "queue.Queue[Any]" = queue.Queue()
        stop = threading.Event()
        started = time.monotonic()
        # CPU-время процесса на запрос: ``process_time`` считает все потоки
        # этого процесса, поэтому снимаем его вокруг всего запроса, а не
        # внутри колонки (две колонки идут параллельно).
        cpu_started = time.process_time()
        prompt_text = "\n".join(
            "{}: {}".format(message.get("role", ""), message.get("content", ""))
            for message in messages
        )
        threads: list[threading.Thread] = []
        for index, side in enumerate(sides):
            thread = threading.Thread(
                target=_run_side,
                args=(
                    index,
                    side,
                    messages,
                    max_tokens,
                    started,
                    events,
                    stop,
                    self.server,
                    prompt_text,
                ),
                name="chat-side-{}".format(index),
                daemon=True,
            )
            thread.start()
            threads.append(thread)

        remaining = len(sides)
        try:
            self._open_sse()
            while remaining > 0:
                item = events.get()
                if isinstance(item, _SideDone):
                    remaining -= 1
                    continue
                if stop.is_set():
                    break
                if not self._send_sse_data(item):
                    stop.set()
                    break
            else:
                wall_ms = max(1, int((time.monotonic() - started) * 1000))
                cpu_ms = round((time.process_time() - cpu_started) * 1000.0, 1)
                try:
                    snapshot = system_snapshot()
                except Exception:  # noqa: BLE001 - метрики не должны рвать ответ
                    logger.exception("снимок системы не снялся")
                    snapshot = {
                        "load": None,
                        "ram": {"total": None, "free": None, "source": None},
                        "disk": None,
                        "cpu_count": os.cpu_count(),
                        "process": {"pid": os.getpid(), "rss": None, "peak": None},
                        "vtm_threads": {"runtime": None, "config": None},
                    }
                metrics = {
                    "wall_ms": wall_ms,
                    "cpu_ms": cpu_ms,
                    # Сколько ядер было занято процессом за время запроса:
                    # 1.0 — одно ядро целиком, 2.0 — два (две колонки).
                    "cpu_cores": round(cpu_ms / wall_ms, 2) if wall_ms else None,
                    "load": snapshot["load"],
                    "ram": snapshot["ram"],
                    "disk": snapshot["disk"],
                    "cpu_count": snapshot["cpu_count"],
                    "process": snapshot["process"],
                    "vtm_threads": snapshot["vtm_threads"],
                    "start_memory": self.server.start_memory,
                }
                if not self._send_sse_data({"type": "end", "metrics": metrics}):
                    stop.set()
                else:
                    self._send_sse_done()
        except _CLIENT_GONE:
            self.close_connection = True
        finally:
            stop.set()
            for thread in threads:
                thread.join(timeout=2.0)

    # -- журнал ---------------------------------------------------------------

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        logger.debug("%s - %s", self.address_string(), format % args)


def create_server(
    port: int = 0,
    *,
    host: str = HOST,
    registry: Sequence[BackendSpec] | None = None,
    adapters: Mapping[tuple[str, str], Any] | None = None,
    system_prompt: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: float = 120.0,
) -> ChatHTTPServer:
    """Собрать сервер; ``port=0`` — свободный порт (удобно тестам).

    ``registry``/``adapters`` инжектируются тестами: реальные модели при этом
    не нужны. ``system_prompt=None`` — взять ``prompts/system.assistant.md``;
    пустая строка отключает системное сообщение.
    """
    if host not in ("127.0.0.1", "::1"):
        raise ValueError(
            "сервер сравнения слушает только петлевой интерфейс (127.0.0.1)"
        )
    if not 0 <= int(port) <= 65535:
        raise ValueError("порт: ожидалось число от 0 до 65535")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
        raise TypeError("max_tokens: ожидалось целое число")
    if max_tokens < 1:
        raise ValueError("max_tokens: ожидалось >= 1")

    resolved = tuple(registry) if registry is not None else default_registry()
    resolved_adapters = (
        dict(adapters)
        if adapters is not None
        else default_adapters(resolved, timeout=timeout)
    )
    prompt = load_system_prompt() if system_prompt is None else str(system_prompt)
    return ChatHTTPServer(
        (host, int(port)),
        registry=resolved,
        adapters=resolved_adapters,
        system_prompt=prompt,
        max_tokens=max_tokens,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Веб-стенд EVA32: два ответа рядом (ВТМ и Ollama).",
        epilog="Слушает только 127.0.0.1. Наружу сервер не выставляется.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="порт на 127.0.0.1 (по умолчанию {})".format(DEFAULT_PORT),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="сколько новых токенов просить у моделей",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="таймаут запроса к Ollama, секунд",
    )
    parser.add_argument(
        "--vtm-tokenizer",
        default="",
        help="tokenizer.json для локальных .vtm (иначе ищется рядом/env)",
    )
    parser.add_argument(
        "--ollama-model",
        action="append",
        default=None,
        metavar="ИМЯ",
        help="модель Ollama (можно несколько раз); по умолчанию Qwen2.5/Qwen3",
    )
    parser.add_argument(
        "--no-system-prompt",
        action="store_true",
        help="не подмешивать prompts/system.assistant.md",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="подробный журнал запросов",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа CLI: поднять сервер и ждать Ctrl+C."""
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        ollama_models = (
            [(name, name) for name in args.ollama_model]
            if args.ollama_model
            else None
        )
        registry = default_registry(
            vtm_tokenizer=args.vtm_tokenizer or None,
            ollama_models=ollama_models,
        )
        server = create_server(
            port=args.port,
            registry=registry,
            system_prompt="" if args.no_system_prompt else None,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
    except (OSError, ValueError, TypeError) as exc:
        print("не удалось запустить сервер: {}".format(exc), file=sys.stderr)
        return 2

    actual_port = server.server_address[1]
    print(
        "чат-стенд: http://{}:{}/  (Ctrl+C — остановить)".format(HOST, actual_port)
    )
    print(
        "модели: {}".format(
            ", ".join(
                "{}={}".format(
                    backend.id,
                    ", ".join(model.id for model in backend.models) or "нет",
                )
                for backend in server.registry
            )
        )
    )
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nостановлен")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
