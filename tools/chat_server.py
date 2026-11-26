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
  RSS-дельта, причина остановки, кэш префикса, ``cancelled``/``cancel_ms``),
  ``end`` (CPU-время запроса и
  снимок системы), ``error`` (человеческий текст). Терминальное событие —
  ``data: [DONE]``, как у ``eva/server.py``;
* ``POST /api/cancel`` — остановить идущую генерацию (задача 0185):
  ``{"side": 0}`` / ``{"side": 1}`` — одну колонку, ``{}`` — обе. Отмена идёт
  до C-рантайма (``vtm_api_cancel`` → ``VTM_API_CANCELLED``), а не только
  «клиент перестал читать»; обрыв SSE и новый вопрос по той же модели
  отменяют генерацию тем же путём. В ``done`` — ``cancelled``, ``cancel_ms``
  и ``cancel_reason``;
* память диалога (задача 0182) — своя история ``user``/``assistant`` на
  колонку: в запрос уходит ``[system, ...история, user]``, ответ дописывается
  в историю после ``done`` (пустой/ошибочный — не дописывается). Бюджет —
  ``--history-tokens`` (0 — без истории) и ``--history-turns``; старые пары
  выбрасываются целиком, системный промпт остаётся. Без токенизатора счёт
  по символам (~3 символа на токен) и помечен как оценка;
* ``POST /api/reset`` — новый диалог: сброс истории колонки
  (``{"side": 0}``) или обеих (``{}``); в шапке колонки — «история: N реплик ·
  ~M токенов», счётчик приходит в ``history`` у ``GET /api/models``;
* ``POST /api/tts`` — озвучка (задачи 0183/0191/0197): ``{"text", "voice"}``
  → аудио (``Cache-Control: no-store``). Content-Type — по сигнатуре:
  ``audio/mpeg`` для MP3 ElevenLabs, ``audio/wav`` для Silero/say/espeak.
  ``voice`` — имя голоса Silero (``baya``/``kseniya``/``xenia``/``aidar``,
  по умолчанию ``baya``); любое другое значение — 400 с перечнем допустимых.
  Текст чистится от ``<think>…</think>``, markdown и URL, длиннее 600
  символов — 400. Синтез — ``eva/tts.py``: ElevenLabs → Silero → системный
  ``say``/``espeak`` → null; сбой облака откатывается на локальный голос, а
  503 «голос недоступен» — только если запасного движка тоже нет; чат при
  этом работает как раньше;
* ``POST /api/load`` / ``POST /api/unload`` — явная загрузка/выгрузка
  ВТМ-модели (задача 0166). Тело — ``{"backend": "vtm", "model": "<id>"}``,
  неизвестная модель — 400, чужой бэкенд (Ollama) — 400: её памятью
  управляет её процесс. Ответ несёт ``loaded``, ``rss_bytes`` (дельта RSS
  процесса до/после; 0 после выгрузки), ``load_ms``, ``requests``;
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

Модель можно держать в памяти и выгружать по бездействию (как ``keep_alive``
у Ollama, только наше решение)::

    python3 tools/chat_server.py --port 8765 --keep-alive 300

``0`` — держать всегда, без флага автовыгрузка выключена (по умолчанию).

Рассуждения (задача 0200): у ВТМ по умолчанию **выключены** — промпт
рендерится настоящим Qwen-шаблоном с пустым закрытым ``<think>``, поэтому
``--max-tokens`` целиком уходит на ответ. ``--think`` включает рассуждение с
бюджетом ``--reasoning-budget`` (по умолчанию 256): при исчерпании бюджета
рассуждение принудительно закрывается, и ответ **досказывается** вторым
прогоном; ``--max-tokens`` тогда — лимит ответа, общий лимит новых токенов =
бюджет + ответ. UI уже умеет сворачивать ``<think>`` в «рассуждение».

Только стандартная библиотека; бэкенды берутся из ``tools/chat.py`` (внутри —
``eva``), их файлы не меняются.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import gc
import json
import math
import pathlib
import logging
import os
import queue
import re
import select
import shutil
import socket
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
    "ChatHistory",
    "ClientError",
    "DEFAULT_HISTORY_TOKENS",
    "DEFAULT_HISTORY_TURNS",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_PORT",
    "HOST",
    "ModelBusy",
    "ModelSpec",
    "OLLAMA_BASE",
    "TTS_MAX_CHARS",
    "ChatCliAdapter",
    "StreamPiece",
    "VtmInspect",
    "clean_tts_text",
    "create_server",
    "default_adapters",
    "default_registry",
    "estimate_tokens",
    "default_tts_speech",
    "inspect_vtm",
    "looks_like_audio",
    "looks_like_wav",
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

from eva.chat_template import DEFAULT_REASONING_BUDGET  # noqa: E402
from eva.plugins import (  # noqa: E402
    DEFAULT_FACT_LIMIT,
    PluginError,
    PluginNotFound,
    PluginRegistry,
    PluginRestartRequired,
    default_registry as default_plugin_registry,
)
from eva.skills.intent import (  # noqa: E402
    FactGuardStream,
    fact_block,
    fact_entries,
    with_fact,
)
from eva.ru_gender import GenderStreamer, feminize  # noqa: E402

#: Ollama — только этот адрес; клиент не может подставить свой.
OLLAMA_BASE = "http://127.0.0.1:11434/v1"
#: Сколько новых токенов просим у обеих моделей (сравнение на равных).
DEFAULT_MAX_TOKENS = 256
#: Бюджет истории диалога на колонку, токенов (задача 0182).
#: ``0`` — истории нет: каждая колонка спрашивается с чистого листа.
DEFAULT_HISTORY_TOKENS = 512
#: Сколько последних реплик (пар вопрос/ответ) держать в истории колонки.
DEFAULT_HISTORY_TURNS = 8
#: Оценка токенов без токенизатора: средний BPE-токен — 3–4 символа
#: латиницы и 2–3 кириллицы, берём 3 (как ``tools/chat.py::estimate_tokens``).
HISTORY_CHARS_PER_TOKEN = 3
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
#:
#: ``demo/`` намеренно вне списка: это тестовая фикстура (vocab=128, генерируется
#: ``make -C src/vtm test``), а не разговорная модель. Её место — в тестах, и в
#: списке чата она только путает: выбор даёт ошибку «token вне vocab».
VTM_SEARCH_DIRS: tuple[str, ...] = ("runs", "export")
#: Токенизатор-заглушка: фикстура BPE (287) совпадает с ``runs/tiny``.
#: Для демо-модели (vocab=128) она не подходит — чат вернёт понятную ошибку.
FALLBACK_TOKENIZER = ROOT / "tests" / "vtm" / "fixtures" / "tokenizer.json"
#: Системный промпт EVA32: одинаковый для обеих моделей — сравнение честное.
#: Боевой промпт (6 правил, ~1500 символов) — для сильных моделей.
SYSTEM_PROMPT_PATH = ROOT / "prompts" / "system.assistant.md"
#: Компактный промпт для квантованных моделей: 4 строки. На длинном промпте
#: 3B Q4_1 теряет следование инструкции и уходит в посторонние темы, на
#: компактном — отвечает по делу; prefill при этом в разы дешевле.
COMPACT_PROMPT_PATH = ROOT / "prompts" / "system.compact.md"
#: Что подставлять по умолчанию: компактный промпт (обе колонки одинаково,
#: иначе сравнение моделей перестаёт быть честным).
DEFAULT_PROMPT_PATH = COMPACT_PROMPT_PATH

#: Ошибки записи, означающие уход клиента: стрим прекращаем, не шумим в лог.
_CLIENT_GONE = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)

#: Как часто HTTP-поток проверяет, не закрыл ли клиент соединение, пока
#: генерация молчит (prefill может идти десятки секунд — писать тогда нечего,
#: и обрыв раньше замечался только на первой записи, задача 0185).
CLIENT_POLL_S = 0.05
#: Сколько ждать отменённую генерацию, прежде чем занять контекст новым
#: вопросом. Отмена — атомарный флаг в C (штатно <100 мс), запас на медленный
#: токен и закрытие потока адаптера.
PREEMPT_TIMEOUT_S = 5.0


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
    #: Сколько длилась загрузка модели именно в этом запросе (``None`` —
    #: модель уже была в памяти). Нужна, чтобы честно вычесть загрузку из
    #: TTFT: метрика адаптера общая и могла быть перезаписана другим запросом.
    load_ms: float | None = None


def _resolve_tokenizer(model_path: Path, explicit: str | None = None) -> str:
    """Токенизатор ВТМ: явный, из env, рядом с моделью, затем демо-фикстура."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env_value = (os.environ.get("EVA_VTM_TOKENIZER") or "").strip()
    if env_value:
        candidates.append(Path(env_value).expanduser())
    # Токенизатор под конкретную модель: у Qwen3 и Qwen2.5 в каталоге лежат
    # разные словари, и общий tokenizer.json даёт «неизвестный token id» на
    # служебных токенах Qwen3. Поэтому сначала ищем файл с именем модели.
    stem = model_path.name.split("-")[0].split(".")[0]
    for name in (stem + "-tokenizer.json", pathlib.Path(stem).stem + "-tokenizer.json"):
        candidates.append(model_path.with_name(name))
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
    vtm = _build_vtm_specs(vtm_tokenizer)
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


def _build_vtm_specs(vtm_tokenizer: str | None) -> list[ModelSpec]:
    """Собрать ВТМ-модели из файлов на диске (без кэша — вызов дешёвый)."""
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
    return vtm


# --------------------------------------------------------------------------- #
# Адаптеры бэкендов
# --------------------------------------------------------------------------- #


class ModelBusy(RuntimeError):
    """Модель занята генерацией: выгрузку сейчас делать нельзя (409)."""


def _open_backend(backend: Any, depth: int = 0) -> bool:
    """Раскрыть бэкенд, чтобы модель оказалась в памяти (не только в объекте).

    ``VtmChatBackend`` из ``tools/chat.py`` не имеет ``open()``: веса читает
    ``VtmBackend.open()`` у вложенного ``_inner``. Идём по цепочке, но не
    глубже двух уровней — чужой объект с методом ``open`` (файл, сокет) здесь
    не ожидается, а случайно дёргать его нельзя. ``False`` — загружать нечего
    (HTTP-клиент Ollama, заглушка в тестах); это не ошибка.
    """
    opener = getattr(backend, "open", None)
    if callable(opener):
        opener()
        return True
    if depth >= 2:
        return False
    nested = getattr(backend, "_inner", None)
    if nested is not None and nested is not backend:
        return _open_backend(nested, depth + 1)
    return False


def _warm_backend(backend: Any) -> bool:
    """Прогреть модель одним коротким forward-проходом.

    ``VtmBackend.open()`` только отображает файл (mmap) и разбирает
    заголовки/таблицу тензоров: страницы весов остаются на диске, и первый
    запрос всё равно платит за них — ровно то, от чего уходим. Один шаг
    генерации проходит через все слои, поэтому после него TTFT запроса — это
    генерация, а не чтение 2,24 ГиБ. Ошибку прогрева не считаем фатальной:
    модель загружена, а запрос покажет ту же ошибку человеческим текстом.
    """
    stream = getattr(backend, "stream", None)
    if not callable(stream):
        return False
    try:
        for _piece in stream(
            [{"role": "user", "content": "а"}], max_tokens=1
        ):
            pass
    except Exception as exc:  # noqa: BLE001 - прогрев не должен ломать загрузку
        logger.warning("прогрев модели не удался: %s", exc)
        return False
    return True


def _close_backend(backend: Any) -> None:
    """Закрыть бэкенд и его вложенный контекст; ошибки — в лог, не наружу.

    Сначала вложенный (``VtmBackend`` держит C-контекст и память весов),
    потом внешний. Двойное закрытие безопасно: ``VtmBackend.close()``
    идемпотентен, а базовая ``ChatBackend.close()`` ничего не делает.
    """
    nested = getattr(backend, "_inner", None)
    targets = (
        [nested, backend]
        if nested is not None and nested is not backend
        else [backend]
    )
    for target in targets:
        closer = getattr(target, "close", None)
        if not callable(closer):
            continue
        try:
            closer()
        except Exception as exc:  # noqa: BLE001 - память важнее исключения
            logger.warning("закрытие бэкенда %r: %s", target, exc)


def _cancel_backend(backend: Any) -> bool:
    """Попросить текущую генерацию бэкенда остановиться (задача 0185).

    Идём по цепочке (``VtmChatBackend`` -> ``VtmBackend``) и предпочитаем
    ``cancel_active()``: он отменяет только идущую генерацию и ничего не
    делает, если её уже нет. Иначе отмена, пришедшая на миллисекунду позже
    конца ответа, «протекла» бы в следующий запрос: у C-ABI ``vtm_api_cancel``
    флаг доходит до ближайшего ``generate()``, если генерации нет.

    ``cancel()`` (без ``cancel_active``) — откат для бэкендов, у которых этот
    метод единственный. Возврат ``True`` — отмену приняли (или запомнили).
    """
    targets = [backend]
    nested = getattr(backend, "_inner", None)
    while (
        nested is not None
        and nested is not backend
        and nested not in targets
        and len(targets) < 3
    ):
        targets.insert(0, nested)
        backend = nested
        nested = getattr(nested, "_inner", None)
    # cancel_active — «отменить только идущее». Если он есть хотя бы на одном
    # уровне, отката на sticky cancel() быть не должно: False здесь значит
    # «отменять нечего», а cancel() в этот момент поставил бы C-флаг, который
    # достанется следующему generate() (проверено: новый вопрос отменялся сам).
    if any(callable(getattr(target, "cancel_active", None)) for target in targets):
        for target in targets:
            action = getattr(target, "cancel_active", None)
            if not callable(action):
                continue
            try:
                return action() is not False
            except Exception as exc:  # noqa: BLE001 - отмена не должна ронять сервер
                logger.warning("отмена %r: %s", target, exc)
        return False
    # Откат для бэкендов без cancel_active (только sticky cancel()).
    for target in targets:
        action = getattr(target, "cancel", None)
        if not callable(action):
            continue
        try:
            result = action()
        except Exception as exc:  # noqa: BLE001 - отмена не должна ронять сервер
            logger.warning("отмена %r: %s", target, exc)
            continue
        if result is not False:
            return True
    return False


def _looks_cancelled(exc: BaseException) -> bool:
    """Похоже ли исключение на штатную отмену, а не на сбой бэкенда."""
    names = {cls.__name__ for cls in type(exc).__mro__}
    return "VtmCancelled" in names or "CancelledError" in names


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

    Задача 0166: модель можно загрузить заранее (:meth:`load`) и выгрузить
    (:meth:`unload`). ``load`` собирает бэкенд, зовёт ``open()`` у
    ``VtmBackend`` (C-контекст и ``mmap`` файла) и прогревает модель одним
    токеном: только после forward-прохода страницы весов реально резидентны,
    поэтому TTFT последующих запросов уже не содержит загрузку. Выгрузка
    зовёт ``close()``, отпускает объект и собирает мусор: следующий запрос
    снова загрузит модель лениво, как раньше.
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
        #: Замер выгрузки: RSS до/после и сколько вернулось системе.
        self.unload_stats: dict[str, Any] | None = None
        #: Сколько запросов обслужено (успешных генераций, с ``done``).
        self.requests = 0
        #: ``time.monotonic()`` последней активности (загрузка или ответ) —
        #: по нему ``--keep-alive`` решает, что модель простаивает.
        self._last_used: float | None = None
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

    # -- статус, загрузка, выгрузка ----------------------------------------

    def status(self) -> dict[str, Any]:
        """Состояние модели для карточки и ответов ``/api/load|unload``.

        ``rss_bytes`` — реальная дельта RSS процесса от загрузки (для ВТМ
        модель живёт здесь же); у выгруженной — 0, у Ollama — ``None``:
        её память в чужом процессе, этот процесс о ней не знает.
        """
        loaded = self.loaded
        stats = self.load_stats if isinstance(self.load_stats, Mapping) else None
        rss_bytes: int | None = None
        load_ms: int | float | None = None
        if self.kind == "vtm":
            if loaded and stats is not None:
                rss_bytes = stats.get("rss_delta")
                load_ms = stats.get("load_ms")
            elif not loaded and self.unload_stats is not None:
                rss_bytes = 0
        return {
            "loaded": loaded,
            "rss_bytes": rss_bytes,
            "load_ms": load_ms,
            "requests": int(self.requests),
            "load": dict(stats) if stats is not None else None,
            "unload": (
                dict(self.unload_stats)
                if isinstance(self.unload_stats, Mapping)
                else None
            ),
        }

    def idle_seconds(self, now: float | None = None) -> float | None:
        """Сколько секунд модель бездействует; ``None`` — не загружена."""
        if not self.loaded:
            return None
        last = self._last_used
        if last is None:  # pragma: no cover - выставляется при загрузке
            return None
        moment = time.monotonic() if now is None else float(now)
        return max(0.0, moment - last)

    def load(self) -> dict[str, Any]:
        """Загрузить модель в память сейчас, не дожидаясь запроса.

        Идемпотентно: у загруженной модели возвращается ``already_loaded``
        и прежние числа, фабрика повторно не зовётся.
        """
        with self._lock:
            already = self._inner is not None
            self._ensure()
            self._last_used = time.monotonic()
            report = self.status()
        report["already_loaded"] = already
        return report

    def unload(self, *, timeout: float = 0.0) -> dict[str, Any]:
        """Выгрузить модель: закрыть контекст, отпустить память.

        ``timeout=0`` — не ждать генерацию: если модель занята, поднимается
        :class:`ModelBusy` (HTTP-слой отдаст 409). Идемпотентно: у выгруженной
        модели ``was_loaded`` = false.
        """
        if timeout > 0:
            acquired = self._lock.acquire(timeout=float(timeout))
        else:
            acquired = self._lock.acquire(blocking=False)
        if not acquired:
            raise ModelBusy(
                "модель занята (идёт загрузка или генерация) — выгрузка не начата"
            )
        try:
            inner = self._inner
            if inner is None:
                report = self.status()
                report["was_loaded"] = False
                report["freed_bytes"] = 0
                report["unload_ms"] = 0.0
                return report
            before = _rss_bytes()
            started = time.monotonic()
            # Сначала отпускаем ссылку, потом закрываем: даже если close()
            # бросит, адаптер не останется с «загруженной» мёртвой моделью.
            self._inner = None
            _close_backend(inner)
            gc.collect()
            after = _rss_bytes()
            freed = (
                before - after
                if before is not None and after is not None
                else None
            )
            self.unload_stats = {
                "rss_before": before,
                "rss_after": after,
                "freed_bytes": freed,
                "unload_ms": round((time.monotonic() - started) * 1000.0, 1),
            }
            report = self.status()
            report["was_loaded"] = True
            report["freed_bytes"] = freed
            report["unload_ms"] = self.unload_stats["unload_ms"]
        finally:
            self._lock.release()
        return report

    def _ensure(self) -> Any:
        if self._inner is None:
            with self._lock:
                if self._inner is None:
                    before = _rss_bytes()
                    started = time.monotonic()
                    inner = self._factory()
                    try:
                        opened = _open_backend(inner)
                        # mmap мало: чтобы «загружено» означало «веса уже в
                        # памяти», прогоняем один токен через все слои.
                        warmed = bool(opened and self.kind == "vtm") and _warm_backend(
                            inner
                        )
                    except BaseException:
                        _close_backend(inner)
                        raise
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
                        # open() раскрыл контекст сразу; false — бэкенд без
                        # явной загрузки (HTTP-клиент, заглушка в тестах).
                        "opened": opened,
                        "warmed": warmed,
                    }
                    self.unload_stats = None
                    self._inner = inner
                    self._last_used = time.monotonic()
        return self._inner

    def stream(
        self, messages: Sequence[Mapping[str, str]], *, max_tokens: int
    ) -> Iterator[StreamPiece]:
        # Одна генерация на адаптер (у ВТМ один C-контекст). Новый вопрос при
        # идущей генерации — сначала отменяем прежнюю (задача 0185): иначе он
        # ждал бы её до max_tokens, как было до отмены. Ждём не больше
        # PREEMPT_TIMEOUT_S: отмена в C — атомарный флаг, обычно <100 мс.
        terminal = False
        if not self._lock.acquire(blocking=False):
            self.cancel()
            if not self._lock.acquire(timeout=PREEMPT_TIMEOUT_S):
                raise ModelBusy(
                    "модель занята: предыдущая генерация не остановилась"
                )
        try:
            loaded_before = self._inner is not None
            inner = self._ensure()
            iterator = inner.stream(list(messages), max_tokens=max_tokens)
            try:
                for event in iterator:
                    text = getattr(event, "text", "")
                    if isinstance(text, str) and text:
                        yield StreamPiece(text=text)
                    if getattr(event, "done", False):
                        usage = getattr(event, "usage", {}) or {}
                        terminal = True
                        # Счётчик обслуженных запросов: генератор закрывают
                        # сразу после терминального куска, поэтому считать
                        # его после цикла уже поздно.
                        self.requests += 1
                        stats_now = (
                            self.load_stats
                            if not loaded_before and isinstance(self.load_stats, Mapping)
                            else None
                        )
                        yield StreamPiece(
                            done=True,
                            usage=dict(usage),
                            finish_reason=str(getattr(event, "finish_reason", "") or ""),
                            load_ms=(
                                stats_now.get("load_ms")
                                if isinstance(stats_now, Mapping)
                                else None
                            ),
                        )
                        break
            finally:
                closer = getattr(iterator, "close", None)
                if callable(closer):
                    closer()
                self._last_used = time.monotonic()
            if not terminal:
                raise RuntimeError(
                    "бэкенд завершил поток без терминального события"
                )
        finally:
            # Потребитель ушёл («стоп», обрыв SSE, новый вопрос) — не оставляем
            # C-машину дорабатывать ответ: закрытие генератора уже остановило
            # поток, а cancel_active страхует бэкенды без cancel-on-close.
            if not terminal:
                _cancel_backend(self._inner)
            self._lock.release()

    def cancel(self) -> bool:
        """Отменить идущую генерацию (задача 0185).

        ``True`` — у бэкенда была активная генерация и отмена принята;
        ``False`` — отменять нечего (или бэкенд не умеет отмену). Вызывается
        из HTTP-потока (``POST /api/cancel``) и из ``stream`` перед новым
        вопросом; на VTM это атомарный ``vtm_api_cancel``.
        """
        inner = self._inner
        if inner is None:
            return False
        return _cancel_backend(inner)


def _chat_cli_factory(
    kind: str,
    model: ModelSpec,
    timeout: float,
    *,
    think: bool = False,
    reasoning_budget: int = DEFAULT_REASONING_BUDGET,
) -> Callable[[], Any]:
    """Фабрика бэкенда из ``tools/chat.py``; импорт и модель — при первом запросе."""
    budget = max(0, int(reasoning_budget))

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
                think=bool(think),
                reasoning_budget=budget,
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
    registry: Iterable[BackendSpec],
    *,
    timeout: float = 120.0,
    think: bool = False,
    reasoning_budget: int = DEFAULT_REASONING_BUDGET,
) -> dict[tuple[str, str], Any]:
    """Адаптеры для всех моделей реестра; ВТМ открывается лениво.

    ``think``/``reasoning_budget`` — политика рассуждений ВТМ: по умолчанию
    выключена (ответ сразу), бюджет действует только при включении.
    """
    adapters: dict[tuple[str, str], Any] = {}
    for backend in registry:
        for model in backend.models:
            if backend.kind not in ("vtm", "ollama"):
                continue
            adapters[(backend.id, model.id)] = ChatCliAdapter(
                _chat_cli_factory(
                    backend.kind,
                    model,
                    timeout,
                    think=think,
                    reasoning_budget=reasoning_budget,
                ),
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
        "unload": None,
        # Задача 0166: статус памяти одной строкой на карточку.
        "rss_bytes": None,
        "load_ms": None,
        "requests": None,
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
    status_of = getattr(adapter, "status", None)
    if callable(status_of):
        try:
            status = status_of()
        except Exception:  # noqa: BLE001 - карточка не должна падать из-за метрик
            logger.exception("статус адаптера не снялся")
            status = None
        if isinstance(status, Mapping):
            if isinstance(status.get("loaded"), bool):
                card["loaded"] = status["loaded"]
            for name in ("rss_bytes", "load_ms", "requests", "load", "unload"):
                if name in status:
                    card[name] = status[name]
    prefix = getattr(adapter, "cache_prefix", None)
    if isinstance(prefix, bool):
        card["cache_prefix"] = prefix
    return card



def _live_model(backend: BackendSpec, model: ModelSpec) -> bool:
    """Модель реально доступна прямо сейчас.

    Реестр собирается один раз при старте, но модели удаляются (чистка
    нерабочих артефактов) и появляются новые. Поэтому перед отдачей в UI
    проверяем файл: удалённые не показываем, иначе список превращается в
    кладбище. Для Ollama проверка ленивая — живой список даёт /api/ps.
    """
    if backend.kind != "vtm":
        return True
    return bool(model.target) and os.path.isfile(model.target)


def _refresh_registry(server: "ChatHTTPServer") -> tuple[BackendSpec, ...]:
    """Пересобрать ВТМ-часть реестра из файлов (дёшево: glob + stat)."""
    if not getattr(server, "dynamic_models", False):
        return tuple(server.registry)
    vtm = _build_vtm_specs(getattr(server, "vtm_tokenizer", None))
    merged: list[BackendSpec] = []
    for backend in server.registry:
        if backend.kind == "vtm":
            merged.append(BackendSpec(id=backend.id, label=backend.label,
                                      kind="vtm", models=tuple(vtm)))
        else:
            merged.append(backend)
    server.registry = tuple(merged)
    return server.registry


def _model_cards(server: "ChatHTTPServer") -> dict[str, Any]:
    """Карточки всех моделей реестра; последние ответы — из памяти сервера."""
    cards: list[dict[str, Any]] = []
    for backend in _refresh_registry(server):
        for model in backend.models:
            # В реальном сервере удалённые файлы не показываем (иначе список
            # превращается в кладбище после чистки); тесты передают свой реестр
            # и должны видеть карточки отсутствующих моделей с present=false.
            if getattr(server, "dynamic_models", False) and not _live_model(backend, model):
                continue
            key = (backend.id, model.id)
            cards.append(
                _model_card(
                    backend,
                    model,
                    adapter=server.adapters.get(key),
                    last=server.last_answers.get(key),
                )
            )
    return {
        "models": cards,
        "baseline": server.start_memory,
        "max_tokens": server.max_tokens,
        # Политика автовыгрузки: null — выключена, 0 — держать всегда.
        "keep_alive": getattr(server, "keep_alive", None),
        # Память диалога (задача 0182): реплики и оценка токенов по колонкам.
        "history": _history_json(server),
    }


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


def load_system_prompt(path: pathlib.Path | None = None) -> str:
    """Системный промпт из ``prompts/`` (по умолчанию — компактный)."""
    target = path or DEFAULT_PROMPT_PATH
    try:
        return target.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def estimate_tokens(text: str) -> int:
    """Оценка токенов по символам: ~3 символа на токен.

    Токенизатор есть только у ВТМ, да и тот про свой словарь; для бюджета
    истории это не нужно. Правило то же, что в ``tools/chat.py``: средний
    BPE-токен — 3–4 символа латиницы и 2–3 кириллицы, поэтому число ниже
    честно помечается как оценка (``estimated: true``), а не выдаётся за
    измерение провайдера.
    """
    if not text:
        return 0
    return max(1, math.ceil(len(text) / HISTORY_CHARS_PER_TOKEN))


class ChatHistory:
    """Память диалога: своя история ``user``/``assistant`` на каждую колонку.

    Задача 0182. Раньше запрос всегда собирался как ``[system, user]``, и
    модель не видела, о чём говорили раньше. Теперь история живёт на сервере
    отдельно для левой и правой колонки: сравнение моделей остаётся честным
    (у каждой свой диалог), а системный промпт — общий и всегда первый.

    Обрезка — по двум лимитам: ``max_turns`` последних реплик (пара
    вопрос/ответ) и ``max_tokens`` бюджета. Старые пары выбрасываются
    целиком — обрубков реплик в истории не бывает. Последняя пара остаётся
    даже если одна не влезает в бюджет: иначе «второй вопрос не видит
    первый» на длинном ответе. ``max_tokens == 0`` выключает историю совсем.

    Потокобезопасно: HTTP-поток читает историю при сборке запроса, а
    дописывает после ``done``; два запроса могут идти одновременно.
    """

    def __init__(
        self,
        *,
        max_tokens: int = DEFAULT_HISTORY_TOKENS,
        max_turns: int = DEFAULT_HISTORY_TURNS,
        counter: Callable[[str], int] | None = None,
    ) -> None:
        self.max_tokens = max(0, int(max_tokens))
        # Одна реплика — минимум одна пара: меньше не имеет смысла.
        self.max_turns = max(1, int(max_turns))
        self._counter = counter or estimate_tokens
        self._lock = threading.Lock()
        self._sides: dict[int, list[dict[str, str]]] = {0: [], 1: []}

    @property
    def enabled(self) -> bool:
        """Есть ли история: ``--history-tokens 0`` выключает её целиком."""
        return self.max_tokens > 0

    def messages(self, side: int) -> list[dict[str, str]]:
        """Копия истории колонки — ровно то, что уйдёт в модель."""
        with self._lock:
            return [dict(message) for message in self._sides.get(side, ())]

    def append(
        self, side: int, question: str, answer: str
    ) -> list[dict[str, str]]:
        """Дописать реплику (вопрос + ответ) и подрезать историю под лимиты."""
        if not self.enabled:
            return []
        with self._lock:
            items = self._sides.setdefault(side, [])
            items.append({"role": "user", "content": question})
            items.append({"role": "assistant", "content": answer})
            self._fit(items)
            return [dict(message) for message in items]

    def reset(self, side: int | None = None) -> None:
        """Забыть историю колонки; ``side=None`` — обеих."""
        with self._lock:
            sides = self._sides.keys() if side is None else (side,)
            for index in sides:
                self._sides[index] = []

    def stats(self, side: int) -> dict[str, Any]:
        """Счётчик для UI: реплик, сообщений, оценка токенов."""
        with self._lock:
            items = self._sides.get(side) or []
            tokens = self._tokens(items)
            return {
                "side": side,
                "turns": len(items) // 2,
                "messages": len(items),
                "tokens": tokens,
                "estimated": True,
            }

    # -- внутреннее ----------------------------------------------------------

    def _tokens(self, messages: Sequence[Mapping[str, str]]) -> int:
        return sum(
            self._counter(str(message.get("content") or ""))
            for message in messages
        )

    def _fit(self, items: list[dict[str, str]]) -> None:
        """Выбросить самые старые пары, пока история не влезет в лимиты."""
        while len(items) > self.max_turns * 2:
            del items[:2]
        while len(items) > 2 and self._tokens(items) > self.max_tokens:
            del items[:2]


def _history_json(server: "ChatHTTPServer") -> dict[str, Any]:
    """История обеих колонок для ``/api/models`` и ``/api/reset``.

    Токены помечены ``estimated: true``: точного токенизатора у сервера нет,
    счёт идёт по символам (~3 символа на токен).
    """
    history = getattr(server, "history", None)
    if not isinstance(history, ChatHistory):
        return {"enabled": False, "estimated": True, "sides": []}
    return {
        "enabled": history.enabled,
        "limit_tokens": history.max_tokens,
        "limit_turns": history.max_turns,
        "estimated": True,
        "sides": [history.stats(0), history.stats(1)],
    }


def _prompt_text(messages: Sequence[Mapping[str, str]]) -> str:
    """Промпт одной колонки строкой — для оценки кэша префикса."""
    return "\n".join(
        "{}: {}".format(message.get("role", ""), message.get("content", ""))
        for message in messages
    )


def _strip_think(text: str) -> str:
    """Убрать рассуждение ``<think>…</think>`` из ответа (включая незакрытый).

    Клиент показывает его отдельным блоком, а в историю он попадать не должен:
    иначе каждый следующий запрос тащит прошлые «размышления» и модель
    тратит на них контекст и время.
    """
    import re as _re

    text = _re.sub(r"(?is)<think(?:ing)?\b[^>]*>.*?(?:</think(?:ing)?>|$)", "", text)
    return text.strip()


def _last_question(messages: Sequence[Mapping[str, str]]) -> str:
    """Вопрос колонки: последнее сообщение пользователя в её запросе."""
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def _fresh_fact(question: str, server: Any = None) -> str:
    """Секция ``СПРАВКА`` из включённых skill-плагинов; пусто — фактов нет.

    Задача 0209: ядро больше не знает про время/курс/котировки — оно спрашивает
    все включённые плагины способности ``skill.fetch`` и склеивает их ответы
    (не больше ``DEFAULT_FACT_LIMIT``). Сеть и права — на стороне плагина:
    рантайм передаёт ему ``PluginContext`` с allowlist.

    Любой сбой этого пути (плагин, сеть, парсинг) — пустая строка: чат не
    должен падать и не должен отвечать 500 из-за недоступного ЦБ.
    ``server`` нужен, чтобы учитывать включение/выключение через
    ``POST /api/plugins`` без перезапуска; без него — общий реестр процесса.
    """
    registry = getattr(server, "plugins", None)
    if not isinstance(registry, PluginRegistry):
        try:
            # Не ``default_registry``: в этом модуле так называется реестр
            # бэкендов, и одноимённый вызов вернул бы кортеж, а не реестр
            # плагинов (задача 0208: факты молча пропадали).
            registry = default_plugin_registry()
        except Exception:  # noqa: BLE001 — данные не стоят сломанного запроса
            logger.exception("плагины: реестр не поднялся")
            return ""
    limit = getattr(server, "facts_limit", DEFAULT_FACT_LIMIT)
    try:
        facts = registry.collect_facts(question, limit=limit)
    except Exception:  # noqa: BLE001 — данные не стоят сломанного запроса
        logger.exception("плагины: сбор фактов упал")
        return ""
    if not facts:
        return ""
    # Задача 0212: не «[Факт: …]», а секция с командой модели. Заголовок
    # печатается один раз, строки данных — по одной на факт.
    line = fact_block(fact.line() for fact in facts)
    logger.debug("справка плагинов: %s", line.replace("\n", " | "))
    return line


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
    # У ВТМ один C-контекст на модель: две колонки с ОДНОЙ моделью отменяют
    # друг друга (вторая генерация снимает первую — «генерация отменена»).
    # Лучше сказать это сразу, чем показать «зависло».
    def _vtm_key(item: Any) -> tuple[str, str] | None:
        if not isinstance(item, dict):
            return None
        if item.get("backend", "vtm") != "vtm":
            return None
        return ("vtm", str(item.get("model") or ""))

    keys = [_vtm_key(item) for item in raw_sides]
    # ВТМ-бэкенд держит ОДИН C-контекст на процесс: две ВТМ-колонки конфликтуют
    # даже с разными моделями («контекст ВТМ занят»), поэтому сравнение
    # допустимо только ВТМ против Ollama. Говорим это сразу, а не через минуту
    # ожидания.
    vtm_sides = [index for index, key in enumerate(keys) if key is not None]
    if len(vtm_sides) > 1:
        raise ClientError(
            "Обе колонки выбраны на ВТМ: у рантайма один C-контекст на процесс, "
            "и вторая генерация получает «контекст занят». Оставьте ВТМ в одной "
            "колонке, а вторую сравните с Ollama."
        )

    sides: list[dict[str, Any]] = []
    for index, raw_side in enumerate(raw_sides):
        if not isinstance(raw_side, Mapping):
            raise ClientError("sides[{}]: ожидался объект.".format(index))
        # Колонка «нет»: пользователь гасит Ollama, чтобы не занимать память, —
        # тогда вторая колонка выключается, а не висит на таймаутах.
        if str(raw_side.get("backend") or "").strip().lower() == "none":
            sides.append(
                {
                    "backend": "none",
                    "backend_label": "выключена",
                    "model": "",
                    "model_label": "",
                    "adapter": None,
                    "disabled": True,
                }
            )
            continue
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


def _validate_model_target(
    server: "ChatHTTPServer", payload: Any
) -> tuple[BackendSpec, ModelSpec, Any]:
    """Проверить тело ``/api/load|unload``: ``(бэкенд, модель, адаптер)``.

    Клиент называет только ``backend``/``model`` из закрытого реестра:
    произвольного пути или URL здесь быть не может. Неизвестная модель —
    400 (как и неизвестный бэкенд). Ollama — 400 с отдельным кодом: её
    памятью управляет её собственный процесс, кнопки грузят только ВТМ.
    """
    if not isinstance(payload, Mapping):
        raise ClientError("Тело запроса должно быть JSON-объектом.")
    backend_id = payload.get("backend", "vtm")
    if not isinstance(backend_id, str) or not backend_id.strip():
        raise ClientError("Поле backend: ожидалась строка (например, vtm).")
    model_id = payload.get("model")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ClientError(
            "Поле model: нужен id модели из реестра (например, "
            "runs/q4/3b-q4_1.vtm)."
        )
    registry = _refresh_registry(server)
    catalog = {backend.id: backend for backend in registry}
    backend = catalog.get(backend_id)
    if backend is None:
        raise ClientError(
            "Неизвестный бэкенд {!r}; разрешены: {}.".format(
                backend_id, ", ".join(sorted(catalog)) or "нет"
            )
        )
    allowed = {model.id: model for model in backend.models}
    model = allowed.get(model_id)
    if model is None:
        raise ClientError(
            "Неизвестная модель {!r} для бэкенда {}; разрешены: {}.".format(
                model_id, backend_id, ", ".join(sorted(allowed)) or "нет"
            )
        )
    if backend.kind != "vtm":
        raise ClientError(
            "Загрузка и выгрузка есть только у ВТМ: памятью Ollama управляет "
            "её собственный процесс (см. keep_alive в её API).",
            code="load_unsupported",
        )
    adapter = server.adapters.get((backend_id, model_id))
    if adapter is None:
        raise ClientError(
            "Адаптер для {} / {} не собран на сервере.".format(
                backend_id, model_id
            ),
            status=503,
            code="backend_unavailable",
        )
    return backend, model, adapter


# --------------------------------------------------------------------------- #
# Работа одной колонки
# --------------------------------------------------------------------------- #


class _CancelState:
    """Кто и когда попросил остановку — для метрик и для колонок.

    Один объект на запрос ``/api/chat``. Отмена бывает:
      * общая — клиент закрыл SSE или пришёл ``POST /api/cancel`` без ``side``;
      * по колонке — ``POST /api/cancel {"side": 0}``: вторая колонка при этом
        продолжает работать (у неё своя модель и свой ``stop``).
    Времена запроса (``requested_at``) и фактической остановки (``stopped_at``)
    дают ``cancel_ms`` в метриках ``done`` — «сколько шло до реальной остановки»,
    а не только «сколько клиент не читал».
    """

    def __init__(self, sides: int) -> None:
        self._lock = threading.Lock()
        self.all = threading.Event()
        self.sides = {index: threading.Event() for index in range(max(1, sides))}
        self.reason = ""
        self.requested_at: dict[int, float] = {}
        self.stopped_at: dict[int, float] = {}

    def request_side(self, side: int, reason: str) -> bool:
        """Отметить отмену колонки; ``True`` — это первая отмена."""
        with self._lock:
            event = self.sides.setdefault(side, threading.Event())
            first = not event.is_set()
            self.requested_at.setdefault(side, time.monotonic())
            self.reason = self.reason or reason
            event.set()
            return first

    def request_all(self, reason: str) -> None:
        for side in list(self.sides):
            self.request_side(side, reason)
        with self._lock:
            self.all.set()

    def requested(self, side: int) -> bool:
        event = self.sides.get(side)
        return self.all.is_set() or (event is not None and event.is_set())

    def stopped(self, side: int) -> None:
        """Зафиксировать фактическую остановку колонки (после закрытия потока)."""
        with self._lock:
            self.stopped_at.setdefault(side, time.monotonic())

    def cancel_ms(self, side: int) -> float | None:
        """Сколько прошло от запроса отмены до реальной остановки, мс."""
        with self._lock:
            started = self.requested_at.get(side)
            finished = self.stopped_at.get(side)
        if started is None or finished is None:
            return None
        return round(max(0.0, finished - started) * 1000.0, 1)

    def was_cancelled(self, side: int) -> bool:
        return side in self.requested_at


class _SideDone:
    """Внутренний маркер очереди: колонка закончила (в SSE не уходит).

    ``answer`` — весь текст ответа колонки, если генерация дошла до ``done``.
    По нему HTTP-поток дописывает историю диалога (задача 0182): пустая
    строка значит «ответа нет» (ошибка, отмена, пустая генерация), и такая
    реплика в историю не попадает.
    """

    def __init__(self, side: int, answer: str = "") -> None:
        self.side = side
        self.answer = answer


def _run_side(
    index: int,
    side: Mapping[str, Any],
    messages: Sequence[Mapping[str, str]],
    max_tokens: int,
    started: float,
    events: "queue.Queue[Any]",
    cancel: "_CancelState",
    server: Any = None,
    prompt_text: str = "",
    facts: Sequence[str] = (),
) -> None:
    """Сгенерировать одну колонку и сложить события в очередь.

    Пишет в очередь только этот поток; HTTP-поток читает и отправляет SSE.
    Так две модели работают параллельно, а сокет не делят между потоками.

    Здесь же снимаются метрики ответа (задача 0145): TTFT и он же prefill
    (время до первого токена), декод = остаток, причина остановки, был ли
    первый токен холодным (модель грузилась в этом запросе), попадание в кэш
    префикса ВТМ и ``/api/ps`` для Ollama.

    Отмена (задача 0185): колонка проверяет ``cancel.requested(index)`` перед
    каждым куском и закрывает поток адаптера. Это и есть фактическая остановка
    C-рантайма (``vtm_api_cancel`` -> ``VTM_API_CANCELLED``): время от запроса
    до неё попадает в ``cancel_ms`` метрик ``done``.
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
    request_load_ms: float | None = None
    char_count = 0
    completed = False
    cancelled = False
    # Полный текст ответа — для истории диалога (задача 0182). Складываем
    # здесь, а не по delta в HTTP-потоке: склейка кусков живёт в этом же
    # цикле, и ответ известен целиком к моменту ``done``.
    answer_parts: list[str] = []
    answer_for_history = ""
    delta_buffer: list[str] = []
    delta_flushed = time.monotonic()
    # Фильтры ответа (задачи 0207/0209): «я понял» → «я поняла» и прочая
    # постобработка живут и в плагинах (``plugins/ru-gender``), и в ядре.
    # Рантайм отдаёт потоковую стадию на каждую колонку (у стримера своё
    # состояние), поэтому два ответа рядом не делят недописанное слово.
    # ``GenderStreamer`` ниже — страховка на случай, если плагин рода не
    # найден или выключен: ``feminize`` идемпотентна, двойной прогон (плагин +
    # ядро) ничего не портит, а требование «Ева в женском роде» не зависит от
    # discovery плагинов.
    filters = (
        server.plugins.stream_filter(scope="chat")
        if server is not None
        else None
    )
    # Страж противоречия (задача 0212): ответ «не могу предоставить» при
    # наличии справки заменяется фактической фразой. Держит только начало
    # ответа (первое предложение), поэтому стриминг не страдает; без справки
    # он прозрачен и обычное «не знаю» остаётся как есть.
    guard = FactGuardStream(facts)
    gender = GenderStreamer()
    try:
        iterator = iter(adapter.stream(messages, max_tokens=max_tokens))
        try:
            try:
                for piece in iterator:
                    if cancel.requested(index):
                        cancelled = True
                        break
                    text = piece.text if isinstance(piece, StreamPiece) else str(piece)
                    if text:
                        char_count += len(text)
                        if first_token_ms is None:
                            first_token_ms = max(
                                0, int((time.monotonic() - started) * 1000)
                            )
                        spoken = filters.feed(text) if filters is not None else text
                        spoken = guard.feed(spoken)
                        spoken = gender.feed(spoken)
                        if spoken:
                            answer_parts.append(spoken)
                            # Склеиваем куски перед отправкой: у ВТМ токен —
                            # это 1–3 символа русского слова (BPE), и
                            # посимвольная выдача выглядит как «печатает по
                            # буквам». Копим до 60 мс или до границы слова,
                            # чтобы глаз видел слова, а не буквы.
                            delta_buffer.append(spoken)
                            now = time.monotonic()
                            boundary = text.endswith(
                                (" ", ".", ",", "!", "?", ":", ";", "\n")
                            )
                            if delta_buffer and (
                                boundary or (now - delta_flushed) >= 0.06
                            ):
                                events.put({
                                    "type": "delta",
                                    "side": index,
                                    "text": "".join(delta_buffer),
                                })
                                delta_buffer.clear()
                                delta_flushed = now
                    if isinstance(piece, StreamPiece) and piece.done:
                        usage = dict(piece.usage or {})
                        finish_reason = piece.finish_reason or finish_reason
                        request_load_ms = piece.load_ms
                        break
            except BaseException as exc:
                # Отмена закрывает поток адаптера: он вправе ответить
                # VtmCancelled/CancelledError. Это штатная остановка, а не
                # сбой колонки — ошибку клиенту не шлём. VtmCancelled без
                # запроса сервера (бэкенд отменили напрямую) тоже считаем
                # остановкой, а не ошибкой.
                if not cancel.requested(index) and not _looks_cancelled(exc):
                    raise
                cancelled = True
            # Бэкенд мог сам корректно закончить поток после отмены (ВТМ:
            # vtm_api_cancel -> VtmCancelled; Ollama: конец ответа): это всё
            # равно остановка по запросу, а не штатный конец генерации.
            if not cancelled and cancel.requested(index):
                cancelled = True
            # Хвост, ждавший границы слова («я понял» без точки в конце), —
            # в тот же поток и в историю: иначе последнее слово ответа
            # осталось бы мужским. Хвост плагина проходит через ядерный
            # стример, чтобы порядок стадий не терялся.
            tail = filters.flush() if filters is not None else ""
            tail = guard.feed(tail) + guard.flush()
            tail = gender.feed(tail) + gender.flush()
            if tail:
                answer_parts.append(tail)
                delta_buffer.append(tail)
            if delta_buffer:
                events.put({"type": "delta", "side": index,
                            "text": "".join(delta_buffer)})
                delta_buffer.clear()
            completed = True
        finally:
            closer = getattr(iterator, "close", None)
            if callable(closer):
                closer()
    except BaseException as exc:  # noqa: BLE001 - ошибка одной колонки не рвёт другую
        tail = filters.flush() if filters is not None else ""
        tail = guard.feed(tail) + guard.flush()
        tail = gender.feed(tail) + gender.flush()
        if tail:
            answer_parts.append(tail)
            delta_buffer.append(tail)
        if delta_buffer:
            events.put({"type": "delta", "side": index,
                        "text": "".join(delta_buffer)})
            delta_buffer.clear()
        logger.warning("колонка %d (%s): ошибка: %s", index, side["model"], exc)
        if not cancel.requested(index):
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
        if completed:
            if cancelled:
                # Время фактической остановки — до сборки метрик: cancel_ms
                # должен попасть в done этого же ответа.
                cancel.stopped(index)
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
            # Задача 0166: если модель грузилась этим запросом, TTFT честно
            # делится на «загрузка» и «генерация». При заранее загруженной
            # модели (кнопка /api/load) загрузки нет и ttft_gen_ms совпадает
            # с first_token_ms. load_ms берём из терминального куска: это
            # число именно этого запроса, а не общий замер адаптера.
            raw_load = request_load_ms
            if isinstance(raw_load, (int, float)) and not isinstance(raw_load, bool):
                request_load_ms = float(raw_load)
            else:
                request_load_ms = None
            ttft_gen_ms = first_token_ms
            if request_load_ms is not None and first_token_ms is not None:
                ttft_gen_ms = max(
                    0, int(round(first_token_ms - request_load_ms))
                )
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
                "finish_reason": "cancelled" if cancelled else (finish_reason or None),
                # Отмена (0185): флаг для панели и время до фактической
                # остановки, а не до конца ответа. У непринятой отмены —
                # cancelled=False, cancel_ms=None.
                "cancelled": bool(cancelled),
                "cancel_ms": cancel.cancel_ms(index),
                "cancel_reason": cancel.reason or None if cancelled else None,
                "first_token": (
                    "холодный" if cold else "тёплый"
                ) if cold is not None else None,
                "cold": cold,
                # TTFT без стоимости загрузки: при warm равен first_token_ms,
                # при cold — за вычетом load_ms этого запроса.
                "ttft_gen_ms": ttft_gen_ms,
                "load_ms": request_load_ms,
                "cache_prefix": cache_prefix,
                "load": dict(load_stats) if isinstance(load_stats, Mapping) else None,
                "ollama": ollama_ps,
            }
            if server is not None:
                with server.state_lock:
                    server.last_answers[key] = dict(metrics)
            # done уходит в поток как есть (формат события не меняется), а
            # полный текст запомним для истории — её допишет HTTP-поток.
            # Отменённый ответ в историю не идёт: он недописан (0185).
            answer_for_history = "" if cancelled else "".join(answer_parts)
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
        # Фактическая остановка: поток адаптера закрыт (C-рантайм уже
        # отпустил токен — vtm_api_cancel). Время до неё — в cancel_ms.
        if cancelled:
            cancel.stopped(index)
        # Маркер — всегда последним: HTTP-поток закрывает стрим, только
        # когда обе колонки прислали и done/error, и маркер. Пустой ответ
        # (ошибка, отмена, пустая генерация) в историю не дописывается.
        events.put(_SideDone(index, answer_for_history))


# --------------------------------------------------------------------------- #
# HTML-страница (CSS/JS встроены, внешних ссылок нет)
# --------------------------------------------------------------------------- #

# Самопроверка звука без слуха: страница проходит тот же путь, что и чат
# (POST /api/tts → decodeAudioData → BufferSource), печатая шаги в DOM.
AUDIO_CHECK_PAGE = """<!doctype html><meta charset="utf-8"><title>audio</title>
<pre id="out">старт</pre>
<script>
var out = document.getElementById("out");
var NL = String.fromCharCode(10);
function log(t) { out.textContent += NL + t; }
async function main() {
  try {
    var ctx = new (window.AudioContext || window.webkitAudioContext)();
    log("ctx: " + ctx.state);
    await ctx.resume();
    log("ctx после resume: " + ctx.state);
    var resp = await fetch("/api/tts", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({text: "Проверка звука"})
    });
    log("http: " + resp.status + " " + resp.headers.get("content-type"));
    var blob = await resp.blob();
    log("blob: " + blob.size + " б, тип " + blob.type);
    var buf = await blob.arrayBuffer();
    log("arraybuffer: " + buf.byteLength);
    var decoded = await new Promise(function (res, rej) { ctx.decodeAudioData(buf, res, rej); });
    log("decoded: " + decoded.duration.toFixed(2) + " с, " + decoded.sampleRate + " Гц");
    var src = ctx.createBufferSource();
    src.buffer = decoded;
    src.connect(ctx.destination);
    src.start(0);
    log("РЕЗУЛЬТАТ: играет");
  } catch (e) {
    log("ОШИБКА: " + (e && (e.name || e.message) ? (e.name + ": " + e.message) : e));
  }
}
main();

// Вариант B — ровно путь чата: blob → createObjectURL → fetch(url) → arrayBuffer
// → decodeAudioData. Нужен, чтобы отличать проблему страницы чата от сервера.
async function variantB() {
  log("--- вариант B (как в чате)");
  try {
    var ctx = new (window.AudioContext || window.webkitAudioContext)();
    await ctx.resume();
    var resp = await fetch("/api/tts", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({text: "Звук включён. Я Ева, слушайте ответы."})
    });
    log("B http: " + resp.status);
    var blob = await resp.blob();
    log("B blob: " + blob.size + " б, тип " + blob.type);
    var url = URL.createObjectURL(blob);
    var again = await fetch(url);
    log("B повторный fetch: " + again.status);
    var buf = await again.arrayBuffer();
    log("B arraybuffer: " + buf.byteLength);
    var decoded = await new Promise(function (res, rej) { ctx.decodeAudioData(buf, res, rej); });
    log("B decoded: " + decoded.duration.toFixed(2) + " с");
    var src = ctx.createBufferSource();
    src.buffer = decoded;
    src.connect(ctx.destination);
    src.start(0);
    log("B РЕЗУЛЬТАТ: играет");
  } catch (e) {
    log("B ОШИБКА: " + (e && (e.name || e.message) ? (e.name + ": " + e.message) : e));
  }
}
variantB();
</script>
"""

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
  /* Озвучка (задача 0183): галочка в шапке колонки и плашка деградации. */
  .check.tts { font-size: 12px; gap: 5px; white-space: nowrap; }
  .check.tts input { width: 14px; height: 14px; }
  .tts-note { color: var(--danger); font-size: 11.5px; }
  .tts-note.clickable { cursor: pointer; text-decoration: underline dotted; }
  .tts-note:empty { display: none; }
  button {
    border: 0; border-radius: 9px; padding: 10px 20px; font: inherit;
    font-weight: 650; color: #fff; background: var(--accent); cursor: pointer;
  }
  button:disabled { opacity: .55; cursor: default; }
  button.mini {
    padding: 6px 11px; font-size: 12px; border-radius: 7px;
    background: var(--accent-2);
  }
  button.mini.alt { background: var(--muted); }
  button.mini.ghost {
    background: transparent; color: var(--muted);
    border: 1px solid var(--border);
  }
  .loadctl { display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
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
  /* Ответ растёт по содержимому: без max-height и без внутренней прокрутки.
     Свернуть длинный ответ можно кнопкой в шапке панели — состояние
     .panel.answer-collapsed ограничивает только сам текст ответа. */
  .answer {
    margin: 0; padding: 15px; flex: 1 0 auto;
    white-space: pre-wrap; overflow-wrap: anywhere; background: transparent;
    color: var(--text); font: inherit;
    max-height: none; overflow: visible;
  }
  .panel.answer-collapsed .answer {
    flex: 0 1 auto; max-height: 18rem; overflow: auto;
  }
  .answer:empty::before {
    content: "Ответ появится здесь…"; color: var(--muted); font-style: italic;
  }
  /* Рассуждение <think>: свёрнуто по умолчанию, раскрывается кликом. */
  .think {
    margin: 12px 14px 0; padding: 8px 12px;
    background: var(--code-bg); border: 1px dashed var(--border);
    border-radius: 10px; color: var(--muted); font-size: 13px;
  }
  .think[hidden] { display: none; }
  .think summary { cursor: pointer; font-weight: 650; color: var(--muted); }
  .think summary:focus-visible {
    outline: 2px solid var(--accent); outline-offset: 2px;
  }
  .think-body {
    margin: 8px 0 0; padding: 0; max-height: 45vh; overflow: auto;
    white-space: pre-wrap; overflow-wrap: anywhere; color: var(--text);
    font: inherit;
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
      <label class="field inline" for="voice">голос
        <select id="voice">
          <option value="baya" selected>baya — живой</option>
          <option value="kseniya">kseniya — выше</option>
          <option value="xenia">xenia — ниже</option>
        </select>
      </label>
      <label class="field inline" for="mode">режим
        <select id="mode">
          <option value="fast">быстро — 3B (слово за 0,25 с)</option>
          <option value="smart">умно — 30B MoE (слово за 1 с)</option>
        </select>
      </label>
      <button id="ask" type="button">Спросить</button>
      <button id="stop" type="button" class="mini ghost" disabled>Стоп</button>
      <span id="status" role="status"></span>
    </div>
  </section>

  <section class="grid">
    <article class="panel" id="panel-0">
      <div class="panel-head">
        <div class="pickers">
          <select id="backend-0" aria-label="Бэкенд слева">
            <option value="none">— нет —</option>
          </select>
          <select id="model-0" aria-label="Модель слева"></select>
        </div>
        <div class="loadctl">
          <button type="button" class="mini" id="load-0" disabled>Загрузить</button>
          <button type="button" class="mini alt" id="unload-0" disabled>Выгрузить</button>
          <button type="button" class="mini ghost" id="new-0">Новый диалог</button>
        </div>
        <div class="stats">
          <span>время: <b id="time-0">—</b></span>
          <span>ток/с: <b id="speed-0">—</b></span>
          <span>токенов: <b id="tokens-0">—</b></span>
          <span>история: <b id="history-0">—</b></span>
        </div>
        <button type="button" class="mini ghost" id="collapse-0" hidden>свернуть</button>
        <label class="check tts"><input type="checkbox" id="tts-0"> озвучивать</label>
        <button class="mini ghost" id="tts-check-0" type="button" title="Проверить звук">▶ звук</button>
        <span class="tts-note" id="tts-note-0" role="status"></span>
      </div>
      <details class="think" id="think-0" hidden>
        <summary id="think-sum-0">рассуждение</summary>
        <pre class="think-body" id="think-body-0"></pre>
      </details>
      <pre class="answer" id="answer-0" aria-live="polite"></pre>
      <div class="foot" id="foot-0"></div>
    </article>

    <article class="panel" id="panel-1">
      <div class="panel-head">
        <div class="pickers">
          <select id="backend-1" aria-label="Бэкенд справа">
            <option value="none">— нет —</option>
          </select>
          <select id="model-1" aria-label="Модель справа"></select>
        </div>
        <div class="loadctl">
          <button type="button" class="mini" id="load-1" disabled>Загрузить</button>
          <button type="button" class="mini alt" id="unload-1" disabled>Выгрузить</button>
          <button type="button" class="mini ghost" id="new-1">Новый диалог</button>
        </div>
        <div class="stats">
          <span>время: <b id="time-1">—</b></span>
          <span>ток/с: <b id="speed-1">—</b></span>
          <span>токенов: <b id="tokens-1">—</b></span>
          <span>история: <b id="history-1">—</b></span>
        </div>
        <button type="button" class="mini ghost" id="collapse-1" hidden>свернуть</button>
        <label class="check tts"><input type="checkbox" id="tts-1"> озвучивать</label>
        <button class="mini ghost" id="tts-check-1" type="button" title="Проверить звук">▶ звук</button>
        <span class="tts-note" id="tts-note-1" role="status"></span>
      </div>
      <details class="think" id="think-1" hidden>
        <summary id="think-sum-1">рассуждение</summary>
        <pre class="think-body" id="think-body-1"></pre>
      </details>
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
        <dt>в памяти</dt><dd id="mc-loaded-0">—</dd>
        <dt>RSS до → после</dt><dd id="mc-rss-0">—</dd>
        <dt>RSS модели</dt><dd id="mc-rssbytes-0">—</dd>
        <dt>загрузка</dt><dd id="mc-loadms-0">—</dd>
        <dt>запросов</dt><dd id="mc-req-0">—</dd>
        <dt>пик RSS</dt><dd id="mc-peak-0">—</dd>
        <dt>резидентная копия</dt><dd id="mc-ps-0">—</dd>
      </dl>
      <dl class="kv">
        <dt class="sub-head">последний ответ</dt>
        <dt>TTFT</dt><dd id="ma-ttft-0">—</dd>
        <dt>TTFT без загрузки</dt><dd id="ma-ttftgen-0">—</dd>
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
        <dt>в памяти</dt><dd id="mc-loaded-1">—</dd>
        <dt>RSS до → после</dt><dd id="mc-rss-1">—</dd>
        <dt>RSS модели</dt><dd id="mc-rssbytes-1">—</dd>
        <dt>загрузка</dt><dd id="mc-loadms-1">—</dd>
        <dt>запросов</dt><dd id="mc-req-1">—</dd>
        <dt>пик RSS</dt><dd id="mc-peak-1">—</dd>
        <dt>резидентная копия</dt><dd id="mc-ps-1">—</dd>
      </dl>
      <dl class="kv">
        <dt class="sub-head">последний ответ</dt>
        <dt>TTFT</dt><dd id="ma-ttft-1">—</dd>
        <dt>TTFT без загрузки</dt><dd id="ma-ttftgen-1">—</dd>
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
      <span>keep-alive: <b id="sys-keepalive">—</b></span>
      <span>ядер: <b id="sys-cpus">—</b></span>
    </p>
    <p class="metric-note">Размер файла — stat; параметры и KV-кэш — сумма по таблице
      тензоров .vtm; RSS — дельта процесса chat_server до и после загрузки
      (модель живёт в этом процессе через ctypes); load/RAM/диск — система.
      «—» значит «недоступно», а не ноль. Кэш префикса ВТМ — оценка по общему
      префиксу промпта: счётчик попаданий C-рантайм не отдаёт. «Резидентная
      копия» — size/size_vram из /api/ps локальной Ollama. Кнопки
      «Загрузить»/«Выгрузить» держат модель в памяти заранее: у загруженной
      модели TTFT не содержит чтения весов с диска. «RSS модели» — дельта RSS
      процесса chat_server от загрузки, «TTFT без загрузки» — TTFT минус
      загрузка этого запроса (у тёплого совпадает с TTFT). Автовыгрузка по
      простою задаётся сервером (--keep-alive, по умолчанию выключена).
      «История» в шапке колонки — реплики этой колонки и оценка их токенов
      (~3 символа на токен): бюджет и число реплик задаёт сервер
      (--history-tokens/--history-turns), «Новый диалог» её очищает.</p>
  </section>
<pre id="selfout" style="display:none"></pre>
</main>
<script>
(function () {
  "use strict";

  var catalog = [];
  var defaults = { left: null, right: null };
  var controller = null;
  var running = false;
  var finishedSides = 0;
  var controlBusy = [false, false];
  // Поток каждого ответа: парсер <think> и счётчик символов рассуждения.
  var streams = [null, null];
  // Автоскролл к концу держится, пока читатель сам не отлистал вверх.
  var stickToBottom = true;

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

  function backendIsNone(side) {
    var select = byId("backend-" + side);
    return !!(select && select.value === "none");
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

  // --- think-парсер: начало (логику проверяет node из tests.test_chat_server) ---
  // Qwen3 кладёт рассуждение в <think>…</think> прямо в поток текста, поэтому
  // тег может прийти разорванным между delta («<thi» + «nk>»). Держим хвост,
  // который ещё может оказаться началом тега, и отдаём его в следующий раз.
  // Всё внутри think уходит отдельным обработчиком: в ответ теги не попадают.
  var THINK_TAG_RE = /<\\/?think(?:ing)?\\s*>/i;
  // Начало тега, оборванное на конце куска: «<», «</», «<thi», «</think »…
  // Такой хвост нельзя показывать — он может дорасти до тега в следующем delta.
  var THINK_TAIL_RE = /^<\\/?(?:t(?:h(?:i(?:n(?:k(?:ing)?)?)?)?)?)?\\s*>?$/;

  function holdThinkTail(text) {
    var start = text.lastIndexOf("<");
    if (start < 0) { return 0; }
    var tail = text.slice(start).toLowerCase();
    return THINK_TAIL_RE.test(tail) ? tail.length : 0;
  }

  function createThinkParser(handlers) {
    var buffer = "";
    var inThink = false;

    function emit(text) {
      if (!text) { return; }
      if (inThink) { handlers.think(text); } else { handlers.answer(text); }
    }

    function feed(chunk) {
      buffer += chunk || "";
      for (;;) {
        var match = THINK_TAG_RE.exec(buffer);
        if (!match) { break; }
        emit(buffer.slice(0, match.index));
        inThink = match[0].charAt(1) !== "/";
        buffer = buffer.slice(match.index + match[0].length);
      }
      var held = holdThinkTail(buffer);
      emit(buffer.slice(0, buffer.length - held));
      buffer = buffer.slice(buffer.length - held);
    }

    // Поток оборвался. В буфере остаётся только начало тега, разорванного
    // вместе с потоком, — как текст его показывать нельзя. Всё содержательное
    // уже ушло в ответ или в рассуждение (незакрытый think — целиком туда).
    function flush() {
      buffer = "";
    }

    return { feed: feed, flush: flush, inThink: function () { return inThink; } };
  }
  // --- think-парсер: конец ---

  // --- TTS-фразы: начало (логику проверяет node из tests.test_chat_server) ---
  // Ответ режется на фразы по мере стрима: конец предложения (. ! ? …) с
  // пробелом после, конец предложения в конце куска (иначе первая фраза
  // ждала бы следующего delta) и перевод строки. Длинная фраза режется по
  // словам под серверный лимит 600 символов.
  var TTS_LIMIT = 600;
  var TTS_END_RE = /[.!?…]+[)\\]"»”']?(?=\\s|$)/;

  function splitTtsChunks(text, limit) {
    var chunks = [];
    var rest = String(text || "").replace(/\\s+/g, " ").trim();
    while (rest.length > limit) {
      var cut = rest.lastIndexOf(" ", limit);
      if (cut <= 0) { cut = limit; }
      chunks.push(rest.slice(0, cut).trim());
      rest = rest.slice(cut).trim();
    }
    if (rest) { chunks.push(rest); }
    return chunks;
  }

  function takeTtsPhrase(buffer, flush) {
    var match = TTS_END_RE.exec(buffer);
    var newline = buffer.indexOf("\\n");
    var cut = -1;
    if (match) { cut = match.index + match[0].length; }
    if (newline >= 0 && (cut < 0 || newline < cut)) { cut = newline + 1; }
    if (cut < 0) {
      if (!flush) { return null; }
      return { phrase: buffer, rest: "" };
    }
    return { phrase: buffer.slice(0, cut), rest: buffer.slice(cut) };
  }

  function createPhraseSplitter(onPhrase) {
    var buffer = "";
    function emit(text) {
      var phrase = String(text || "").trim();
      if (!phrase) { return; }
      splitTtsChunks(phrase, TTS_LIMIT).forEach(onPhrase);
    }
    function feed(chunk) {
      buffer += chunk || "";
      for (;;) {
        var part = takeTtsPhrase(buffer, false);
        if (!part) { break; }
        buffer = part.rest;
        emit(part.phrase);
      }
      // Фраза без единой точки (долгий перечень) не должна ждать конца
      // ответа: режем по последнему пробелу, как только перевалили лимит.
      while (buffer.length > TTS_LIMIT) {
        var space = buffer.lastIndexOf(" ", TTS_LIMIT);
        if (space <= 0) { space = TTS_LIMIT; }
        emit(buffer.slice(0, space));
        buffer = buffer.slice(space);
      }
    }
    function flush() {
      var rest = buffer;
      buffer = "";
      emit(rest);
    }
    return { feed: feed, flush: flush, pending: function () { return buffer; } };
  }
  // --- TTS-фразы: конец ---

  // Очередь озвучки одна на страницу: <audio> тоже один, фразы играют
  // последовательно. Синтез фразы начинается сразу, как она появилась, —
  // пока играет первая, остальные уже готовятся.
  var tts = {
    available: true,
    generation: 0,
    queue: [],
    playing: false,
    ctx: null,
    source: null,
    current: null,
    blocked: null
  };

  function ttsBox(side) { return byId("tts-" + side); }

  // Голос Silero выбирается один на страницу: baya/kseniya/xenia/aidar.
  // Тембр задаёт модель, пост-обработки WAV больше нет.
  function selectedVoice() {
    var select = byId("voice");
    return select ? select.value : "baya";
  }

  function ttsEnabled(side) {
    var box = ttsBox(side);
    return !!(box && box.checked && tts.available);
  }

  function setVoiceNote(side, text) {
    var note = byId("tts-note-" + side);
    if (note) { note.textContent = text || ""; }
  }

  // Web Audio вместо <audio>: политика автоплея проверяется один раз — при
  // resume() в жесте пользователя, — а дальше любые буферы играют без
  // ограничений. С <audio> play() после fetch отклонялся стабильно, и клик по
  // плашке не помогал: элемент уже «сжёг» отказ, повторный play() молчал.
  function audioCtx() {
    if (tts.ctx) { return tts.ctx; }
    var C = window.AudioContext || window.webkitAudioContext;
    if (!C) { return null; }
    try { tts.ctx = new C(); } catch (ignored) { tts.ctx = null; }
    return tts.ctx;
  }

  function unlockSpeech() {
    var ctx = audioCtx();
    if (!ctx) { return null; }
    if (ctx.state === "suspended" && typeof ctx.resume === "function") {
      try { ctx.resume(); } catch (ignored) { /* уже */ }
    }
    return ctx;
  }

  function releaseSpeech() {
    if (tts.source) {
      tts.source.onended = null;
      try { tts.source.stop(); } catch (ignored) { /* уже */ }
      tts.source = null;
    }
    tts.current = null;
    tts.playing = false;
  }

  // Стоп и новый вопрос глушат и звук, и очередь: поколение растёт, и ответы
  // уже отправленных запросов синтеза не попадут в очередь как «новые».
  function stopSpeech() {
    tts.generation += 1;
    tts.queue = [];
    releaseSpeech();
  }

  function voiceFailed(side, reason) {
    tts.available = false;
    stopSpeech();
    [0, 1].forEach(function (index) {
      if (ttsBox(index) && ttsBox(index).checked) {
        setVoiceNote(index, "голос недоступен");
      }
    });
    if (reason) {
      setStatus("голос недоступен — текст идёт как раньше", true);
    }
  }

  // Играем WAV через декодер Web Audio прямо из Blob: onDone — конец,
  // onError(имя ошибки). Blob-URL'ы не используем — CSP connect-src 'self'
  // запрещает fetch(blob:), именно из-за этого звук не шёл.
  function ttsLog(text) {
    if (typeof window !== "undefined" && window.__ttslog) { window.__ttslog(text); }
  }

  function playSpeech(blob, side, onDone, onError) {
    var ctx = unlockSpeech();
    ttsLog("play: ctx " + (ctx ? ctx.state : "нет") + ", blob " + (blob ? blob.size : "нет"));
    if (!ctx) { onError("нет AudioContext"); return; }
    if (!blob || !blob.size) { onError("пустой ответ синтеза"); return; }
    blob.arrayBuffer().then(function (data) {
      ttsLog("play: arraybuffer " + data.byteLength);
      return new Promise(function (resolve, reject) {
        var decoded = ctx.decodeAudioData(data, resolve, reject);
        if (decoded && typeof decoded.then === "function") { decoded.then(resolve, reject); }
      });
    }).then(function (buffer) {
      ttsLog("play: decoded " + (buffer ? buffer.duration.toFixed(2) : "нет"));
      if (!buffer) { throw new Error("декодер вернул пусто"); }
      var src = ctx.createBufferSource();
      src.buffer = buffer;
      src.connect(ctx.destination);
      src.onended = function () { tts.source = null; onDone(); };
      tts.source = src;
      src.start(0);
      ttsLog("play: start ok");
    }).catch(function (err) {
      var name = err ? (err.name ? err.name + ": " + err.message : String(err.message || err)) : "?";
      onError(name);
    });
  }

  function showUnlockNote(side, detail) {
    var note = byId("tts-note-" + side);
    if (!note) { return; }
    note.textContent = "нажмите «▶ звук», чтобы включить" + (detail ? " (" + detail + ")" : "");
    note.classList.add("clickable");
    note.onclick = function () { speakCheck(side); };
  }

  // Проверка звука: разблокировка контекста прямо в жесте, затем короткая
  // фраза. Так слышно, что голос включился, не дожидаясь ответа модели.
  function speakCheck(side) {
    var ctx = unlockSpeech();
    if (!ctx) { setVoiceNote(side, "браузер не поддерживает Web Audio"); return; }
    setVoiceNote(side, "гружу голос… (AudioContext: " + ctx.state + ")");
    // decodeAudioData на ещё не запущенном контексте может не вызвать callback
    // вовсе (видели именно это в Chrome), поэтому сначала дожидаемся resume.
    var ready = (ctx.state === "running" || typeof ctx.resume !== "function")
      ? Promise.resolve()
      : ctx.resume().catch(function () {});
    ready.then(function () {
      return fetch("/api/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: "Звук включён. Я Ева, слушайте ответы.", voice: selectedVoice() })
    }).then(function (resp) {
      if (!resp.ok) { throw new Error("tts " + resp.status); }
      return resp.blob();
    }).then(function (blob) {
      tts.lastBlob = { size: blob.size, type: blob.type, ctx: ctx.state };
      tts.current = { side: side };
      tts.playing = true;
      playSpeech(blob, side, function () {
        releaseSpeech();
        setVoiceNote(side, "");
        tts.available = true;
        if (tts.blocked) {
          var blocked = tts.blocked;
          tts.blocked = null;
          tts.queue.unshift({
            side: blocked.side, url: blocked.url, generation: tts.generation,
            ready: Promise.resolve(blocked.url)
          });
        }
        pumpSpeech();
      }, function (name) {
        releaseSpeech();
        tts.available = false;
        var info = tts.lastBlob || {};
        setVoiceNote(side, "звук не пошёл: " + name +
          " · blob " + (info.size || 0) + " б, тип " + (info.type || "?") +
          ", ctx " + (info.ctx || "?"));
      });
    });
    }).catch(function (err) {
      var note = byId("tts-note-" + side);
      if (note) { note.textContent = "синтез недоступен: " + (err && err.message ? err.message : err); }
    });
  }

  function pumpSpeech() {
    if (tts.playing) { return; }
    var job = tts.queue.shift();
    if (!job) { return; }
    tts.playing = true;
    job.ready.then(function (blob) {
      if (job.generation !== tts.generation) {
        tts.playing = false;
        pumpSpeech();
        return;
      }
      if (!blob) { tts.playing = false; pumpSpeech(); return; }
      tts.current = { side: job.side };
      playSpeech(blob, job.side, function () {
        releaseSpeech();
        pumpSpeech();
      }, function (name) {
        releaseSpeech();
        // Не выбрасываем фразу: держим её до клика «▶ звук» или следующего жеста.
        tts.blocked = { url: url, side: job.side };
        showUnlockNote(job.side, name);
      });
    }).catch(function () {
      tts.playing = false;
      pumpSpeech();
    });
  }

  // Любой клик по «озвучивать» — жест, разрешающий звук: сразу разблокируем.
  [0, 1].forEach(function (side) {
    var box = byId("tts-" + side);
    if (box) { box.addEventListener("change", function () { if (this.checked) { unlockSpeech(); } }); }
  });
  (function () {
    var ask = byId("ask");
    if (ask) { ask.addEventListener("click", function () { unlockSpeech(); }, { once: false }); }
  })();

  function enqueueSpeech(side, phrase) {
    if (!ttsEnabled(side)) { return; }
    var text = String(phrase || "").trim();
    if (!text) { return; }
    var generation = tts.generation;
    var ready = fetch("/api/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json", "Accept": "audio/wav" },
      body: JSON.stringify({ text: text, voice: selectedVoice() })
    }).then(function (response) {
      if (!response.ok) {
        var status = response.status;
        return response.json().catch(function () { return null; }).then(function (data) {
          var message = "HTTP " + status;
          if (data && data.error && data.error.message) {
            message = data.error.message;
          }
          var problem = new Error(message);
          problem.status = status;
          throw problem;
        });
      }
      return response.blob();
    }).then(function (blob) {
      if (generation !== tts.generation || !blob || !blob.size) { return null; }
      return blob;
    }).catch(function (err) {
      if (generation !== tts.generation) { return ""; }
      // 400 — фраза не годится (после чистки пусто, например одна ссылка):
      // молчим только про неё, голос не выключаем.
      if (err && err.status === 400) { return null; }
      voiceFailed(side, err && err.message ? err.message : String(err));
      return null;
    });
    tts.queue.push({ side: side, generation: generation, ready: ready });
    pumpSpeech();
  }

  function resetPanels() {
    finishedSides = 0;
    stickToBottom = true;
    [0, 1].forEach(function (side) {
      byId("panel-" + side).className = "panel";
      resetAnswer(side);
      byId("time-" + side).textContent = "—";
      byId("speed-" + side).textContent = "—";
      byId("tokens-" + side).textContent = "—";
      byId("foot-" + side).textContent = "";
      ["ttft", "ttftgen", "prefill", "decode", "dtps", "total", "tokens",
       "finish", "first", "cache"].forEach(function (name) {
        setText("ma-" + name + "-" + side, "—");
      });
    });
  }

  var statusTimer = null;
  var statusStart = 0;
  var statusPhase = "думает";

  function stopStatusTimer() {
    if (statusTimer) { clearInterval(statusTimer); statusTimer = null; }
  }

  function setStatus(text, isError) {
    var node = byId("status");
    stopStatusTimer();
    node.textContent = text;
    node.className = isError ? "error" : "";
  }

  // Работающий таймер: «думает… 7,2 с» → «отвечает… 1,4 с». Долгий prefill
  // у ВТМ (10–40 с) без таймера выглядит как зависание.
  function startStatusTimer(phase) {
    stopStatusTimer();
    statusStart = Date.now();
    statusPhase = phase;
    var node = byId("status");
    node.className = "";
    function tick() {
      var sec = ((Date.now() - statusStart) / 1000).toFixed(1);
      node.textContent = statusPhase + "… " + sec + " с";
    }
    tick();
    statusTimer = setInterval(tick, 200);
  }

  function switchStatusPhase(phase) {
    if (statusTimer) { statusPhase = phase; }
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

  // История колонки (задача 0182): «N реплик · ~M токенов». Токены — оценка
  // сервера по символам (~3 символа на токен), поэтому подпись честная.
  function historyText(info, enabled) {
    // enabled — флаг всей истории (--history-tokens 0), а не одной колонки.
    if (enabled === false) { return "выключена (0 токенов)"; }
    if (!info) { return "—"; }
    var turns = typeof info.turns === "number" ? info.turns : 0;
    var tokens = typeof info.tokens === "number" ? info.tokens : 0;
    return turns + " реплик · ~" + tokens + " токенов" +
      (info.estimated === false ? "" : " (оценка)");
  }

  function fillHistory(history) {
    if (!history) { return; }
    var sides = Array.isArray(history.sides) ? history.sides : [];
    [0, 1].forEach(function (side) {
      var info = null;
      for (var i = 0; i < sides.length; i += 1) {
        if (sides[i] && sides[i].side === side) { info = sides[i]; break; }
      }
      setText("history-" + side, historyText(info, history.enabled));
    });
  }

  function fillSideMetrics(side, event) {
    setText("ma-ttft-" + side, fmtMs(event.first_token_ms));
    var gen = fmtMs(event.ttft_gen_ms);
    if (typeof event.load_ms === "number" && event.load_ms > 0 && gen !== "—") {
      gen += " (+" + fmtMs(event.load_ms) + " загрузка)";
    }
    setText("ma-ttftgen-" + side, gen);
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

  function setMemoryButtons(index, card) {
    var isVtm = !!(card && card.kind === "vtm" && card.present !== false);
    var isLoaded = !!(card && card.loaded === true);
    var busy = controlBusy[index] === true;
    byId("load-" + index).disabled = busy || !isVtm || isLoaded;
    byId("unload-" + index).disabled = busy || !isVtm || !isLoaded;
  }

  function fillCard(index, card) {
    var names = ["size", "params", "ctx", "kv", "storage", "source", "vocab",
                 "layers", "bpp", "loaded", "rss", "rssbytes", "loadms", "req",
                 "peak", "ps"];
    if (!card) {
      setText("mc-title-" + index, "—");
      names.forEach(function (name) { setText("mc-" + name + "-" + index, "—"); });
      setMemoryButtons(index, null);
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
    var loadedText = "—";
    if (card.loaded === true) {
      loadedText = "да";
      if (typeof card.rss_bytes === "number") {
        loadedText += " · " + fmtSignedBytes(card.rss_bytes);
      }
    } else if (card.loaded === false) {
      loadedText = card.present === false ? "файла нет" : "нет";
    }
    setText("mc-loaded-" + index, loadedText);
    setText("mc-rssbytes-" + index, fmtSignedBytes(card.rss_bytes));
    setText("mc-loadms-" + index, fmtMs(card.load_ms));
    setText("mc-req-" + index, fmtNum(card.requests));
    setMemoryButtons(index, card);
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

  // Ollama может быть выключен специально (освободить память под ВТМ).
  // Тогда правая колонка переключается в «— нет —», а не висит на таймаутах.
  function fallbackEmptySide() {
    var hasOllama = catalog.some(function (item) { return item.id === "ollama"; });
    if (!hasOllama && byId("backend-1") && byId("backend-1").value !== "none") {
      byId("backend-1").value = "none";
      fillModels(1, null);
      setStatus("Ollama недоступен — правая колонка выключена", false);
    }
  }

  async function refreshModels() {
    try {
      var response = await fetch("/api/models",
        { headers: { "Accept": "application/json" } });
      if (!response.ok) { return; }
      var data = await response.json();
      var models = Array.isArray(data.models) ? data.models : [];
      if (typeof data.keep_alive === "number") {
        setText("sys-keepalive", data.keep_alive === 0
          ? "держать всегда" : data.keep_alive + " с простоя");
      } else {
        setText("sys-keepalive", "выключен");
      }
      fillHistory(data.history);
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

  async function loadModel(side, action) {
    if (controlBusy[side]) { return; }
    var backend = byId("backend-" + side).value;
    var model = byId("model-" + side).value;
    if (!backend || !model) { return; }
    controlBusy[side] = true;
    setMemoryButtons(side, null);
    byId("load-" + side).disabled = true;
    byId("unload-" + side).disabled = true;
    setStatus((action === "load" ? "загружаю… " : "выгружаю… ") + model, false);
    var url = action === "load" ? "/api/load" : "/api/unload";
    try {
      var response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json", "Accept": "application/json" },
        body: JSON.stringify({ backend: backend, model: model })
      });
      var data = null;
      try { data = await response.json(); } catch (ignored) { /* не JSON */ }
      if (!response.ok) {
        var message = "HTTP " + response.status;
        if (data && data.error && data.error.message) {
          message = data.error.message;
        }
        setStatus(message, true);
      } else if (action === "load") {
        setStatus(
          "в памяти · " + fmtSignedBytes(data.rss_bytes) +
          " · загрузка " + fmtMs(data.load_ms),
          false
        );
      } else {
        setStatus(
          "выгружена · освобождено " + fmtBytes(data.freed_bytes),
          false
        );
      }
    } catch (err) {
      setStatus("сервер недоступен: " + (err && err.message ? err.message : err), true);
    } finally {
      controlBusy[side] = false;
      refreshModels();
      refreshSystem();
    }
  }

  // Новый диалог колонки (задача 0182): сервер забывает её историю, шапка
  // сразу показывает «0 реплик». Ответ на экране не трогаем — он про уже
  // закончившийся обмен; следующий вопрос начнётся с чистого листа.
  async function resetDialog(side) {
    if (running) { return; }
    setStatus("сбрасываю историю…", false);
    try {
      var response = await fetch("/api/reset", {
        method: "POST",
        headers: { "Content-Type": "application/json", "Accept": "application/json" },
        body: JSON.stringify({ side: side })
      });
      var data = null;
      try { data = await response.json(); } catch (ignored) { /* не JSON */ }
      if (!response.ok) {
        var message = "HTTP " + response.status;
        if (data && data.error && data.error.message) {
          message = data.error.message;
        }
        setStatus(message, true);
        return;
      }
      fillHistory(data.history);
      refreshModels();
      setStatus(
        "история колонки " + (side === 0 ? "слева" : "справа") + " сброшена",
        false
      );
    } catch (err) {
      setStatus("сервер недоступен: " + (err && err.message ? err.message : err), true);
    }
  }

  function appendText(side, text) {
    var answer = byId("answer-" + side);
    answer.appendChild(document.createTextNode(text));
    followStream(side);
    // Текст печатается независимо от звука: сплиттер только копит фразы и
    // отдаёт их в очередь, если «озвучивать» включено.
    var state = streams[side];
    if (state && state.tts) { state.tts.feed(text); }
  }

  function appendThink(side, text) {
    var state = streams[side];
    var box = byId("think-" + side);
    if (!state || !text) { return; }
    if (box.hidden) { box.hidden = false; }
    state.thinkChars += text.length;
    byId("think-body-" + side).appendChild(document.createTextNode(text));
    byId("think-sum-" + side).textContent =
      "рассуждение (" + state.thinkChars + " симв.)";
    followStream(side);
  }

  // Новый ответ: поле ответа и блок рассуждения пустые, парсер — с нуля.
  // Свёрнутость прошлого ответа снимаем: новый ответ показываем целиком.
  function resetAnswer(side) {
    var box = byId("think-" + side);
    box.hidden = true;
    box.open = false;
    byId("think-sum-" + side).textContent = "рассуждение";
    byId("think-body-" + side).textContent = "";
    byId("answer-" + side).textContent = "";
    panelOf(side).classList.remove("answer-collapsed");
    var button = byId("collapse-" + side);
    button.hidden = true;
    button.textContent = "свернуть";
    streams[side] = {
      thinkChars: 0,
      flushed: false,
      tts: createPhraseSplitter(function (phrase) { enqueueSpeech(side, phrase); }),
      parser: createThinkParser({
        answer: function (text) { appendText(side, text); },
        think: function (text) { appendThink(side, text); }
      })
    };
  }

  function deltaAnswer(side, text) {
    if (!streams[side]) { resetAnswer(side); }
    streams[side].parser.feed(text || "");
  }

  // Конец потока: содержательное уже разложено по ответу и рассуждению, в
  // буфере парсера остался только оборванный тег — его не показываем.
  function flushAnswer(side) {
    var state = streams[side];
    if (!state || state.flushed) { return; }
    state.flushed = true;
    state.parser.flush();
    // Конец ответа: недопетая последняя фраза уходит в очередь, уже
    // поставленные фразы доигрываются — очередь не чистим.
    if (state.tts) { state.tts.flush(); }
    updateCollapseButton(side);
  }

  function updateCollapseButton(side) {
    var button = byId("collapse-" + side);
    var collapsed = panelOf(side).classList.contains("answer-collapsed");
    var answer = byId("answer-" + side);
    // Кнопка нужна только длинному ответу: короткий виден целиком и так.
    if (!collapsed && answer.scrollHeight < 520) {
      button.hidden = true;
      return;
    }
    button.hidden = false;
    button.textContent = collapsed ? "развернуть" : "свернуть";
  }

  function toggleCollapse(side) {
    var collapsed = panelOf(side).classList.toggle("answer-collapsed");
    updateCollapseButton(side);
    if (!collapsed) { followStream(side); }
  }

  // Низ блока ответов: страницу тянем к нему, а не к концу документа —
  // иначе автоскролл увозил бы из чата в панель метрик.
  function streamBottom() {
    var grid = document.querySelector(".grid");
    var bottom = grid ? grid.offsetTop + grid.offsetHeight
                      : document.documentElement.scrollHeight;
    return Math.max(0, bottom - window.innerHeight);
  }

  function isNearStreamEnd() {
    var root = document.scrollingElement || document.documentElement;
    return Math.abs(root.scrollTop - streamBottom()) <= 160;
  }

  // Автоскролл к концу: страницу тянем только пока читатель у конца ответов;
  // свёрнутый ответ крутим внутри него самого. Отлистал — не дёргаем.
  function followStream(side) {
    var answer = byId("answer-" + side);
    if (panelOf(side).classList.contains("answer-collapsed")) {
      answer.scrollTop = answer.scrollHeight;
      return;
    }
    if (!stickToBottom) { return; }
    var root = document.scrollingElement || document.documentElement;
    var target = streamBottom();
    if (root.scrollTop < target) { root.scrollTop = target; }
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
    var panel = panelOf(side);
    if (event.type === "start") {
      panel.classList.remove("failed");
      panel.classList.add("streaming");
      byId("foot-" + side).textContent =
        (event.backend_label || event.backend) + " · " +
        (event.model_label || event.model);
    } else if (event.type === "delta") {
      if (statusPhase === "думает") { switchStatusPhase("отвечает"); }
      deltaAnswer(side, event.text || "");
    } else if (event.type === "done") {
      stopStatusTimer();
      flushAnswer(side);
      panel.classList.remove("streaming");
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
      if (finishedSides >= 2) {
        setStatus("готово", false);
        // Обновить карточки: статус «в памяти» и объём должны быть видны сразу.
        refreshModels();
        refreshSystem();
      }
    } else if (event.type === "error") {
      flushAnswer(side);
      panel.classList.remove("streaming");
      panel.classList.add("failed");
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

  // «Стоп» глушит и генерацию, и звук: новый вопрос делает то же самое сам.
  function stopAll() {
    // «Стоп» уходит и по HTTP, а не только закрытием соединения: сервер
    // отменит генерацию в C-рантайме сразу (задача 0185). keepalive — чтобы
    // запрос ушёл даже при закрытии вкладки; ошибку глотаем: если ответ уже
    // кончился, отменять нечего.
    try {
      fetch("/api/cancel", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({}),
        keepalive: true
      }).catch(function (ignored) { /* отменять нечего */ });
    } catch (ignored) { /* старый браузер без keepalive */ }
    if (controller) {
      try { controller.abort(); } catch (ignored) { /* уже завершён */ }
    }
    // Сначала сбрасываем буфер фраз в ответ (текст не теряется), потом
    // глушим: stopSpeech() поднимает поколение и выбрасывает очередь вместе
    // с только что поставленной хвостовой фразой.
    flushAnswer(0);
    flushAnswer(1);
    stopSpeech();
    stopStatusTimer();
    running = false;
    byId("ask").disabled = false;
    byId("stop").disabled = true;
    setStatus("остановлено", false);
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
    stopSpeech();
    controller = new AbortController();
    running = true;
    byId("ask").disabled = true;
    byId("stop").disabled = false;
    resetPanels();
    startStatusTimer("думает");

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
        flushAnswer(0);
        flushAnswer(1);
        setStatus("поток прервался", true);
      }
    } catch (err) {
      // Поток оборвался на середине: недосказанное — в ответ/рассуждение,
      // незакрытый think так и останется рассуждением.
      flushAnswer(0);
      flushAnswer(1);
      if (err && err.name === "AbortError") {
        setStatus("запрос отменён", false);
      } else {
        setStatus("сервер недоступен: " + (err && err.message ? err.message : err), true);
      }
    } finally {
      running = false;
      byId("ask").disabled = false;
      byId("stop").disabled = true;
    }
  }

  async function loadCatalog() {
    try {
      var response = await fetch("/api/backends", { headers: { "Accept": "application/json" } });
      if (!response.ok) { throw new Error("HTTP " + response.status); }
      var data = await response.json();
      catalog = Array.isArray(data.backends) ? data.backends : [];
      defaults = data.defaults || { left: null, right: null };
      fallbackEmptySide();
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
  byId("stop").addEventListener("click", stopAll);
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
    byId("load-" + side).addEventListener("click", function () {
      loadModel(side, "load");
    });
    byId("unload-" + side).addEventListener("click", function () {
      loadModel(side, "unload");
    });
    byId("new-" + side).addEventListener("click", function () {
      resetDialog(side);
    });
    byId("collapse-" + side).addEventListener("click", function () {
      toggleCollapse(side);
    });
    // Галочка «озвучивать»: включение заново пробует голос (сервер мог
    // перезапуститься с TTS), выключение выбрасывает очередь этой колонки.
    var check = byId("tts-check-" + side);
    if (check) { check.addEventListener("click", function () { speakCheck(side); }); }
    byId("tts-" + side).addEventListener("change", function () {
      var box = byId("tts-" + side);
      if (box.checked) {
        tts.available = true;
        setVoiceNote(0, "");
        setVoiceNote(1, "");
        return;
      }
      tts.queue = tts.queue.filter(function (job) { return job.side !== side; });
    });
  });
  // Отлистал вверх — автоскролл за потоком выключается, вернулся к концу —
  // включается снова.
  window.addEventListener("scroll", function () {
    stickToBottom = isNearStreamEnd();
  }, { passive: true });

  // Режим «быстро/умно»: переключает ВТМ-колонки между 3B и 30B.
  // Не трогает Ollama-колонку — сравнение с ней остаётся на месте.
  var MODE_MODELS = {
    fast: "runs/q4/3b-q4k.vtm",
    smart: "runs/q4/qwen3-30b-a3b.vtm"
  };

  function applyMode(mode) {
    var target = MODE_MODELS[mode];
    if (!target) { return; }
    [0, 1].forEach(function (side) {
      if (byId("backend-" + side).value !== "vtm") { return; }
      var select = byId("model-" + side);
      for (var i = 0; i < select.options.length; i += 1) {
        if (select.options[i].value === target) {
          select.value = target;
          break;
        }
      }
    });
    refreshModels();
    setStatus("режим: " + (mode === "fast" ? "быстро (3B)" : "умно (30B MoE)"), false);
  }

  byId("mode").addEventListener("change", function () {
    applyMode(this.value);
  });

  // Автотест страницы чата: /?selftest=1 прогоняет настоящий speakCheck и
  // складывает ошибки в скрытый <pre> — так звук проверяется без слуха.
  if (location.search.indexOf("selftest") >= 0) {
    var selfout = byId("selfout");
    var NL = String.fromCharCode(10);
    var selflog = function (t) { if (selfout) { selfout.textContent += NL + t; } };
    window.addEventListener("error", function (e) {
      selflog("window error: " + (e && e.message ? e.message : e));
    });
    window.__ttslog = selflog;
    setTimeout(function () {
      selflog("ctx before: " + (tts.ctx ? tts.ctx.state : "нет"));
      speakCheck(0);
      setTimeout(function () {
        selflog("note: " + (byId("tts-note-0") ? byId("tts-note-0").textContent : "?"));
        selflog("ctx after: " + (tts.ctx ? tts.ctx.state : "нет"));
        selflog("playing: " + tts.playing + ", source: " + (tts.source ? "есть" : "нет"));
      }, 10000);
    }, 300);
  }

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
# Озвучка ответов (задача 0183): чистка текста и WAV для браузера
# --------------------------------------------------------------------------- #

#: Потолок одной фразы: клиент режет ответ на фразы сам, но прислать может
#: что угодно — длинный кусок отвергается с понятным текстом, а не синтезируется.
TTS_MAX_CHARS = 600
#: Формат ответа по умолчанию. Историческое имя оставлено для совместимости:
#: с задачей 0191 ответ бывает и MP3, поэтому MIME определяется по сигнатуре
#: аудио (:func:`looks_like_audio`), а не этим значением.
TTS_CONTENT_TYPE = "audio/wav"

#: ``<think>…</think>`` (и ``<thinking>``) — рассуждение моделью, не для ушей.
_THINK_BLOCK_RE = re.compile(
    r"<think(?:ing)?\b[^>]*>.*?</think(?:ing)?\s*>", re.IGNORECASE | re.DOTALL
)
#: Незакрытый ``<think>`` в конце: поток оборвался, но рассуждение — не ответ.
_THINK_OPEN_RE = re.compile(r"<think(?:ing)?\b[^>]*>.*\Z", re.IGNORECASE | re.DOTALL)
#: Оторванные теги (закрывающий без пары, куски разметки) — просто выкидываем.
_THINK_TAG_RE = re.compile(r"</?think(?:ing)?\b[^>]*>", re.IGNORECASE)
#: Markdown-ссылка: остаётся текст, адрес не произносится.
_MD_LINK_RE = re.compile(r"\[([^\]\n]*)\]\([^)\s]*(?:\s+\"[^\"]*\")?\)")
#: URL и «www.»-адреса: вслух их не читают.
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
#: Markdown-разметка: ``*`` (курсив/жирный), ``#`` (заголовки), ``` ` ``` (код),
#: ``~`` (зачёркнутый). Прочие символы не трогаем: они часть текста.
_MD_DECOR_RE = re.compile(r"[*#`~]+")
#: Хештег как слово: «#маркетинг» → «хештег маркетинг» (до общей чистки #).
_HASHTAG_RE = re.compile(r"#+(?=[A-Za-zА-Яа-яЁё0-9_])")


def clean_tts_text(text: object) -> str:
    """Привести текст к произносимому виду: без разметки, URL и лишних пробелов.

    Чистит сервер, а не клиент: клиент может прислать что угодно, а сказать
    вслух нужно только то, что предназначено читателю. ``<think>`` вместе с
    содержимым отбрасывается (включая незакрытый в конце), markdown-ссылки
    разворачиваются в текст, URL вырезаются, служебные символы разметки
    заменяются пробелами, пробелы схлопываются.
    """
    value = str(text or "")
    if not value.strip():
        return ""
    value = _THINK_BLOCK_RE.sub(" ", value)
    value = _THINK_OPEN_RE.sub(" ", value)
    value = _THINK_TAG_RE.sub(" ", value)
    value = _MD_LINK_RE.sub(r"\1", value)
    value = _URL_RE.sub(" ", value)
    # Хештег — это слово, а не разметка: «#маркетинг» должно звучать, а не
    # исчезать вместе с решёткой (иначе Silero получает пустоту и молчит).
    value = _HASHTAG_RE.sub(" хештег ", value)
    value = _MD_DECOR_RE.sub(" ", value)
    return " ".join(value.split())


def looks_like_wav(data: bytes | bytearray | None) -> bool:
    """Настоящий ли это RIFF/WAVE."""
    return looks_like_audio(data) == "audio/wav"


def looks_like_audio(data: bytes | bytearray | None) -> str | None:
    """Content-Type по сигнатуре аудио: ``audio/wav``, ``audio/mpeg`` или None.

    Проверяем именно содержимое, а не заявленный клиентом/адаптером тип:

    * WAV — ``RIFF`` + ``WAVE`` в заголовке;
    * MP3 — тег ``ID3`` или кадр с синхрословом ``0xFF 0xFB/0xF3/0xF2/0xFA``
      (старшие 11 бит ``0x7FF``).

    Так ответ ElevenLabs (MP3) и локальных движков (WAV) различается без
    доверия внешнему заголовку и без лишних зависимостей.
    """
    if not isinstance(data, (bytes, bytearray)):
        return None
    raw = bytes(data[:12])
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
        return "audio/wav"
    if len(raw) >= 3 and raw[:3] == b"ID3":
        return "audio/mpeg"
    if len(raw) >= 2 and raw[0] == 0xFF and (raw[1] & 0xE0) == 0xE0:
        return "audio/mpeg"
    return None


# --------------------------------------------------------------------------- #
# Голоса Silero (задача 0197)
# --------------------------------------------------------------------------- #
# Озвучка — только Silero. В селекторах страницы и EVA.app три женских
# голоса (baya/kseniya/xenia); aidar валиден в API и SILERO_SPEAKER, но в UI
# не показывается. Питч-шифта и сглаживания нет: тембр задаёт сама модель.
SILERO_VOICES: tuple[str, ...] = ("baya", "kseniya", "xenia", "aidar")

#: Голос по умолчанию — тот же, что у ``SILERO_SPEAKER`` в ``eva/tts.py``.
DEFAULT_VOICE = "baya"


def default_tts_speech(
    text: str, voice: str | None = None
) -> tuple[bytes, str]:
    """Синтез через ``eva/tts.py``: ElevenLabs → Silero → say/espeak → null.

    Возвращает ``(аудиобайты, content_type)``: ElevenLabs — MP3
    (``audio/mpeg``), локальные движки — WAV (``audio/wav``). Импорт ленивый:
    страница и метрики не должны тянуть ``eva`` без нужды.

    ``voice`` — голос Silero (``baya``/``kseniya``/``xenia``/``aidar``);
    проверка значения и подстановка ``baya`` — в HTTP-слое, чтобы неизвестное
    имя получало 400, а не молчаливую подмену. ``NullTTS`` (нет ни одного
    движка) возвращает ``b""`` — это «голоса нет», а не сбой процесса;
    HTTP-слой превращает пустой ответ в понятный 503. Сбой ElevenLabs
    обрабатывает ``eva.tts``: логирует причину без ключа и откатывается на
    Silero/system, поэтому сюда обычно приходит уже аудио. Исключения
    адаптера не глотаются: их текст уходит клиенту как «голос недоступен»
    (только если запасной движок тоже не настроен).
    """
    try:
        from eva.tts import synthesize_audio
    except ImportError as exc:  # pragma: no cover - eva лежит в репозитории
        raise RuntimeError("нет адаптера синтеза eva/tts.py") from exc
    return synthesize_audio(text, voice=voice or DEFAULT_VOICE)


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
        dynamic_models: bool = False,
        vtm_tokenizer: str | None = None,
        keep_alive: float | None = None,
        history_tokens: int = DEFAULT_HISTORY_TOKENS,
        history_turns: int = DEFAULT_HISTORY_TURNS,
        think: bool = False,
        reasoning_budget: int = DEFAULT_REASONING_BUDGET,
        facts: bool = True,
        facts_limit: int = DEFAULT_FACT_LIMIT,
        plugins: PluginRegistry | None = None,
        tts: Callable[
            [str, str | None], bytes | tuple[bytes, str]
        ] | None = None,
    ) -> None:
        self.registry = tuple(registry)
        # Озвучка (задача 0183): чем синтезировать фразу. Тесты подставляют
        # заглушку, боевой путь — eva/tts.py (ElevenLabs → Silero → say).
        self.tts = tts or default_tts_speech
        self.dynamic_models = bool(dynamic_models)
        self.vtm_tokenizer = vtm_tokenizer
        self.adapters = dict(adapters)
        self.system_prompt = system_prompt
        self.max_tokens = max(1, int(max_tokens))
        # Рассуждения ВТМ (задача 0200): по умолчанию выключены. Бюджет
        # прибавляется к max_tokens, поэтому ответ не теряется никогда.
        self.think = bool(think)
        self.reasoning_budget = max(0, int(reasoning_budget))
        # Свежие данные (задачи 0208/0209): время, курс ЦБ РФ, котировка MOEX.
        # Факт подкладывается в системный блок до генерации — модели не нужен
        # tool-calling, а цифры берутся из источника, а не из весов.
        self.facts = bool(facts)
        #: Сколько фактов максимум просить у плагинов (0 — ни одного).
        self.facts_limit = max(0, int(facts_limit))
        #: Рантайм плагинов (ADR 0022): discovery, включение и способности.
        #: Тесты подставляют свой реестр; боевой путь — plugins/ + config.
        #: Именно ``default_plugin_registry``: ``default_registry`` в этом
        #: модуле — реестр бэкендов, и одноимённый вызов давал кортеж.
        self.plugins = (
            plugins if plugins is not None else default_plugin_registry()
        )
        # Память диалога (задача 0182): история user/assistant на колонку.
        # 0 токенов — истории нет (каждый вопрос с чистого листа).
        self.history = ChatHistory(
            max_tokens=history_tokens, max_turns=history_turns
        )
        # Метрики (задача 0145): память на старте, последние ответы по моделям
        # и промпты — для оценки кэша префикса. Всё под одним замком.
        self.state_lock = threading.Lock()
        self.start_memory: dict[str, Any] = process_memory()
        self.last_answers: dict[tuple[str, str], dict[str, Any]] = {}
        self.prompts: dict[tuple[str, str], str] = {}
        # Задача 0166: автовыгрузка ВТМ по бездействию. None — выключена,
        # 0 — держать всегда, N > 0 — выгрузить после N секунд простоя.
        self.keep_alive = None if keep_alive is None else float(keep_alive)
        self._reaper_stop = threading.Event()
        self._reaper: threading.Thread | None = None
        # Активные SSE-запросы (задача 0185): по ним POST /api/cancel находит,
        # что отменять. Ключ — id(_CancelState); значение — (sides, state).
        self._active_lock = threading.Lock()
        self._active_chats: dict[int, tuple[tuple[Any, ...], Any]] = {}
        super().__init__(server_address, ChatRequestHandler)
        if self.keep_alive is not None and self.keep_alive > 0:
            self.start_reaper()

    # -- отмена идущих генераций (задача 0185) ------------------------------

    def register_chat(self, sides: Sequence[Mapping[str, Any]], cancel: Any) -> int:
        """Запомнить активный запрос, чтобы его можно было отменить по HTTP."""
        token = id(cancel)
        with self._active_lock:
            self._active_chats[token] = (tuple(sides), cancel)
        return token

    def unregister_chat(self, token: int) -> None:
        with self._active_lock:
            self._active_chats.pop(token, None)

    def cancel_chats(self, side: int | None = None, reason: str = "stop") -> int:
        """Отменить идущие генерации: все колонки или только ``side``.

        Возврат — сколько бэкендов приняли отмену (у Ollama отмены нет: её
        колонка просто перестаёт читать поток). ``reason`` попадает в метрики
        ``done`` (``cancel_reason``).
        """
        with self._active_lock:
            chats = list(self._active_chats.values())
        accepted = 0
        for sides, cancel in chats:
            targets = range(len(sides)) if side is None else (side,)
            for index in targets:
                if index < 0 or index >= len(sides) or sides[index].get("disabled"):
                    continue
                cancel.request_side(index, reason)
                adapter = sides[index].get("adapter")
                if adapter is not None and _cancel_backend(adapter):
                    accepted += 1
        return accepted

    def cancel_model(self, backend: str, model: str, reason: str = "superseded") -> int:
        """Отменить идущие генерации этой модели (новый вопрос вместо старого).

        Один C-контекст на модель: пока брошенная генерация считает, новый
        вопрос ждал бы её до ``max_tokens``. Поэтому перед стартом нового
        запроса активные запросы этой же модели отменяются — со штатными
        метриками ``cancelled=true`` у старого запроса (задача 0185).
        """
        with self._active_lock:
            chats = list(self._active_chats.values())
        accepted = 0
        for sides, cancel in chats:
            for index, side in enumerate(sides):
                if side.get("backend") != backend or side.get("model") != model:
                    continue
                if side.get("disabled"):
                    continue
                cancel.request_side(index, reason)
                adapter = side.get("adapter")
                if adapter is not None and _cancel_backend(adapter):
                    accepted += 1
        return accepted

    # -- автовыгрузка по бездействию ---------------------------------------

    def start_reaper(self) -> None:
        """Запустить фоновую нить, которая выгружает простаивающие модели."""
        if self._reaper is not None:
            return
        limit = self.keep_alive or 0.0
        interval = max(0.05, min(1.0, limit / 4.0))
        self._reaper = threading.Thread(
            target=self._reap_loop,
            args=(interval,),
            name="chat-keep-alive",
            daemon=True,
        )
        self._reaper.start()

    def _reap_loop(self, interval: float) -> None:
        while not self._reaper_stop.wait(interval):
            try:
                self.reap_idle()
            except Exception:  # noqa: BLE001 - нить не должна умирать молча
                logger.exception("автовыгрузка: ошибка")

    def reap_idle(self, now: float | None = None) -> list[tuple[str, str]]:
        """Выгрузить ВТМ-модели, простаивавшие дольше ``keep_alive``.

        Возвращает список ``(backend, model)``, которые выгружены. Модель,
        которая сейчас генерирует, пропускается: выгрузка не ждёт и не рвёт
        ответ. ``now`` подменяется тестами, чтобы не спать.
        """
        limit = self.keep_alive
        if not limit or limit <= 0:
            return []
        moment = time.monotonic() if now is None else float(now)
        unloaded: list[tuple[str, str]] = []
        for key, adapter in list(self.adapters.items()):
            if getattr(adapter, "kind", "") != "vtm":
                continue
            idle_of = getattr(adapter, "idle_seconds", None)
            unload = getattr(adapter, "unload", None)
            if not callable(idle_of) or not callable(unload):
                continue
            try:
                idle = idle_of(moment)
            except Exception:  # noqa: BLE001 - чужой адаптер не должен мешать
                logger.exception("автовыгрузка %s: idle_seconds", key)
                continue
            if idle is None or idle < limit:
                continue
            try:
                unload()
            except ModelBusy:
                continue
            except Exception:  # noqa: BLE001
                logger.exception("автовыгрузка %s: ошибка", key)
                continue
            # Кэш префикса живёт в модели: после выгрузки старый промпт —
            # не история попаданий.
            with self.state_lock:
                self.prompts.pop(key, None)
            logger.info("автовыгрузка: %s / %s (простой %.0f с)", key[0], key[1], idle)
            unloaded.append(key)
        return unloaded

    def server_close(self) -> None:
        self._reaper_stop.set()
        reaper = self._reaper
        if reaper is not None:
            reaper.join(timeout=2.0)
        super().server_close()


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
            if path == "/audio-check":
                self._send_bytes(
                    200, AUDIO_CHECK_PAGE.encode("utf-8"), "text/html; charset=utf-8"
                )
            elif path in ("/", "/index.html"):
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
            elif path == "/api/plugins":
                self._send_json(200, self.server.plugins.as_dict())
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
            if path in ("/api/load", "/api/unload"):
                self._handle_memory(path)
                return
            if path == "/api/reset":
                self._handle_reset()
                return
            if path == "/api/cancel":
                self._handle_cancel()
                return
            if path == "/api/tts":
                self._handle_tts()
                return
            if path == "/api/plugins":
                self._handle_plugins()
                return
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
            # Свежие данные (задачи 0208/0209): ядро спрашивает включённые
            # skill-плагины до генерации, факты идут в системный блок обеих
            # колонок. Сеть — только через PluginContext плагина (allowlist
            # хостов, таймаут, кэш); сбой → «не удалось получить», а не
            # выдуманная цифра. Выключенный плагин фактов не даёт.
            fact = ""
            fact_lines: list[str] = []
            if use_system and self.server.facts:
                fact = _fresh_fact(question, self.server)
                # Строки данных без заголовка: страж ответа подставляет
                # именно данные, а не служебную инструкцию.
                fact_lines = fact_entries(fact)
            # Запрос у каждой колонки свой (задача 0182): общий системный
            # промпт, её собственная история и текущий вопрос. Историю берём
            # снимком до старта генерации: дописывать её будет уже HTTP-поток
            # после done, и новый вопрос не должен видеть свой же ответ.
            side_messages: list[list[dict[str, str]]] = []
            for index in range(len(sides)):
                messages: list[dict[str, str]] = []
                if use_system and self.server.system_prompt:
                    messages.append(
                        {"role": "system", "content": self.server.system_prompt}
                    )
                messages.extend(self.server.history.messages(index))
                messages.append({"role": "user", "content": question})
                if fact:
                    messages = with_fact(messages, fact)
                side_messages.append(messages)
            self._stream_sides(
                sides, side_messages, max_tokens, facts=fact_lines
            )
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

    # -- память модели (задача 0166) -----------------------------------------

    def _handle_memory(self, path: str) -> None:
        """``POST /api/load|unload``: заранее загрузить или освободить ВТМ."""
        backend, model, adapter = _validate_model_target(
            self.server, self._read_json_body()
        )
        action = "load" if path == "/api/load" else "unload"
        key = (backend.id, model.id)
        handler = getattr(adapter, action, None)
        if not callable(handler):
            raise ClientError(
                "Адаптер модели не умеет {}.".format(
                    "загружаться" if action == "load" else "выгружаться"
                ),
                code="load_unsupported",
            )
        try:
            result = handler()
        except ModelBusy as exc:
            raise ClientError(str(exc), status=409, code="model_busy") from exc
        except ClientError:
            raise
        except Exception as exc:  # noqa: BLE001 - текст ошибки человеческий
            logger.warning("%s %s / %s: %s", action, key[0], key[1], exc)
            raise ClientError(
                "Не удалось {} модель: {}".format(
                    "загрузить" if action == "load" else "выгрузить",
                    _short_error(exc),
                ),
                code="load_failed",
            ) from exc
        if action == "unload":
            # Кэш префикса живёт в выгруженной модели: старый промпт больше
            # не история попаданий.
            with self.server.state_lock:
                self.server.prompts.pop(key, None)
        body: dict[str, Any] = {
            "backend": backend.id,
            "model": model.id,
            "action": action,
        }
        if isinstance(result, Mapping):
            body.update(result)
        self._send_json(200, body)

    # -- новый диалог (задача 0182) ------------------------------------------

    def _handle_reset(self) -> None:
        """``POST /api/reset``: забыть историю колонки или обеих.

        Тело — JSON-объект: ``{"side": 0}`` (слева), ``{"side": 1}``
        (справа) или ``{}`` / без ``side`` — обе колонки. Ответ несёт
        ``reset`` (какие колонки очищены) и свежий ``history`` — тот же
        счётчик, что отдаёт ``/api/models``, чтобы страница обновила шапку
        без второго запроса.
        """
        payload = self._read_json_body()
        if not isinstance(payload, Mapping):
            raise ClientError("Тело запроса должно быть JSON-объектом.")
        raw_side = payload.get("side")
        if raw_side is None:
            sides = [0, 1]
        else:
            if (
                isinstance(raw_side, bool)
                or not isinstance(raw_side, int)
                or raw_side not in (0, 1)
            ):
                raise ClientError(
                    "Поле side: ожидалось 0 (слева), 1 (справа) или "
                    "отсутствие поля — тогда сбрасываются обе колонки."
                )
            sides = [raw_side]
        for side in sides:
            self.server.history.reset(side)
        self._send_json(
            200,
            {"reset": sides, "history": _history_json(self.server)},
        )

    # -- отмена (задача 0185) -------------------------------------------------

    def _handle_cancel(self) -> None:
        """``POST /api/cancel``: остановить идущую генерацию.

        Тело — JSON-объект: ``{"side": 0}`` / ``{"side": 1}`` — отменить одну
        колонку, ``{}`` или без поля — обе. Ответ ``{"cancelled": n, ...}``,
        где ``n`` — сколько бэкендов приняли отмену (у Ollama своего cancel
        нет: её колонка закрывает поток и прекращает читать ответ).

        Это тот же путь, что и обрыв SSE: отмена доходит до C-рантайма
        атомарным флагом (``VTM_API_CANCELLED``), а не «клиент перестал
        читать». Метрики ``done`` получают ``cancelled=true`` и ``cancel_ms`` —
        время до фактической остановки.
        """
        payload: Any = {}
        if self.headers.get("Content-Length") not in (None, "0"):
            payload = self._read_json_body()
        if payload is None:
            payload = {}
        if not isinstance(payload, Mapping):
            raise ClientError("Тело запроса должно быть JSON-объектом.")
        raw_side = payload.get("side")
        if raw_side is not None and (
            isinstance(raw_side, bool)
            or not isinstance(raw_side, int)
            or raw_side not in (0, 1)
        ):
            raise ClientError(
                "Поле side: ожидалось 0 (слева), 1 (справа) или отсутствие "
                "поля — тогда отменяются обе колонки."
            )
        accepted = self.server.cancel_chats(side=raw_side, reason="stop")
        self._send_json(
            200,
            {"cancelled": accepted, "side": raw_side, "reason": "stop"},
        )

    # -- плагины (задача 0209) -----------------------------------------------

    def _handle_plugins(self) -> None:
        """``POST /api/plugins``: включить/выключить плагин без перезапуска.

        Тело — ``{"id": "time", "enabled": false}``. Ответ — обновлённый
        список плагинов (как ``GET /api/plugins``) плюс ``changed``: так
        страница не делает второй запрос. Плагин, который держит состояние,
        выключать на лету нельзя — 409 с объяснением.
        """
        payload = self._read_json_body()
        if not isinstance(payload, Mapping):
            raise ClientError("Тело запроса должно быть JSON-объектом.")
        plugin_id = payload.get("id")
        if not isinstance(plugin_id, str) or not plugin_id.strip():
            raise ClientError("Поле id: нужен id плагина (например, time).")
        enabled = payload.get("enabled")
        if not isinstance(enabled, bool):
            raise ClientError("Поле enabled: ожидалось true или false.")
        try:
            record = self.server.plugins.set_enabled(
                plugin_id.strip(), enabled
            )
        except PluginNotFound as exc:
            raise ClientError(
                str(exc), status=404, code="plugin_not_found"
            ) from exc
        except PluginRestartRequired as exc:
            raise ClientError(
                str(exc), status=409, code="restart_required"
            ) from exc
        except PluginError as exc:
            raise ClientError(str(exc), code="plugin_error") from exc
        body = self.server.plugins.as_dict()
        body["changed"] = {
            "id": record.id,
            "enabled": record.enabled,
        }
        logger.info(
            "плагин %s: %s", record.id, "включён" if record.enabled else "выключен"
        )
        self._send_json(200, body)

    # -- озвучка (задача 0183) -----------------------------------------------

    def _handle_tts(self) -> None:
        """``POST /api/tts``: фраза → аудио (WAV или MP3) для ``<audio>``.

        Тело — ``{"text": "...", "voice": "..."}``; ``voice`` необязателен:
        это имя голоса Silero (``baya``/``kseniya``/``xenia``/``aidar``), по
        умолчанию ``baya``. Неизвестное имя — 400 со списком допустимых, а не
        молчаливая подмена: иначе опечатка в настройках приложения давала бы
        «не тот голос» без объяснений. Адаптер может вернуть как голые байты,
        так и пару ``(bytes, content_type)``; MIME в ответе берём по
        сигнатуре (:func:`looks_like_audio`): MP3 → ``audio/mpeg``, WAV →
        ``audio/wav``. ``Cache-Control: no-store`` остаётся: фраза одноразовая.

        Всё, что не синтезировалось (нет движка, процесс упал, не аудио), —
        503 с текстом «голос недоступен»: страница покажет это, а чат
        продолжит работать. Откат ElevenLabs → Silero/system делает
        ``eva/tts.py``, поэтому сюда ошибка приходит только если запасного
        голоса тоже нет.

        Женский род (задача 0207): фразу чистим, затем
        :func:`eva.ru_gender.feminize`. Клиент обычно присылает уже
        исправленный текст ответа, но озвучка — отдельный вход: прямой
        ``POST /api/tts`` с «я понял» тоже должен звучать «я поняла».
        """
        payload = self._read_json_body()
        if not isinstance(payload, Mapping):
            raise ClientError("Тело запроса должно быть JSON-объектом.")
        raw_text = payload.get("text")
        if not isinstance(raw_text, str):
            raise ClientError("Поле text: нужен текст фразы для озвучки.")
        raw_voice = payload.get("voice")
        voice = DEFAULT_VOICE
        if raw_voice is not None:
            if not isinstance(raw_voice, str):
                raise ClientError("Поле voice: ожидалась строка с именем голоса.")
            voice = raw_voice.strip().lower() or DEFAULT_VOICE
            if voice not in SILERO_VOICES:
                raise ClientError(
                    "Поле voice: неизвестный голос {!r}. Допустимо: {}.".format(
                        raw_voice, ", ".join(SILERO_VOICES)
                    ),
                    code="unknown_voice",
                )
        # Сначала фильтры плагинов (0207: род; 0209: числа — scope tts), потом
        # чистка разметки. Так речь получает женские формы и слова вместо
        # цифр, а код в бэктиках ещё виден и не переписывается.
        filtered = self.server.plugins.filter_text(raw_text, scope="tts")
        # Страховка ядра (задача 0207): плагин ru-gender мог быть не найден или
        # выключен. feminize идемпотентна, повторный прогон ничего не меняет.
        text = clean_tts_text(feminize(filtered))
        if not text:
            raise ClientError(
                "После чистки текста нечего озвучивать: остались только "
                "разметка и пробелы.",
                code="empty_text",
            )
        if len(text) > TTS_MAX_CHARS:
            raise ClientError(
                "Фраза длиннее {} символов — разбейте её на части.".format(
                    TTS_MAX_CHARS
                ),
                code="text_too_long",
            )
        try:
            result = self.server.tts(text, voice)
        except Exception as exc:  # noqa: BLE001 - любой сбой синтеза = 503
            logger.warning("озвучка: %s", exc)
            raise ClientError(
                "голос недоступен: {}".format(_short_error(exc, 300)),
                status=503,
                code="voice_unavailable",
            ) from exc
        declared_type = ""
        if (
            isinstance(result, tuple)
            and len(result) == 2
            and isinstance(result[0], (bytes, bytearray))
        ):
            data = bytes(result[0])
            declared_type = str(result[1] or "")
        elif isinstance(result, (bytes, bytearray)):
            data = bytes(result)
        else:
            data = b""
        content_type = looks_like_audio(data)
        if content_type is None:
            raise ClientError(
                "голос недоступен: движок синтеза не вернул аудио "
                "(WAV/MP3).",
                status=503,
                code="voice_unavailable",
            )
        if declared_type and declared_type != content_type:
            # Заявленный тип может врать (или быть с параметрами): доверяем
            # сигнатуре, но оставляем след для диагностики.
            logger.debug(
                "озвучка: адаптер заявил %s, по сигнатуре %s",
                declared_type,
                content_type,
            )
        # ``_send_bytes`` выставляет Cache-Control: no-store и прочую защиту.
        self._send_bytes(200, data, content_type)

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
        side_messages: Sequence[Sequence[Mapping[str, str]]],
        max_tokens: int,
        *,
        facts: Sequence[str] = (),
    ) -> None:
        """Отдать обе колонки в SSE и дописать их ответы в историю.

        ``side_messages[i]`` — свой запрос колонки ``i``: системный промпт,
        её собственная история и текущий вопрос (задача 0182). Кэш префикса
        ВТМ считается по промпту своей колонки, а не по чужому.
        """
        events: "queue.Queue[Any]" = queue.Queue()
        # Выключенные колонки («— нет —») не запускают ни модель, ни поток:
        # им хватает start+done, которые отдаются сразу после открытия SSE.
        disabled = [i for i, side in enumerate(sides) if side.get("disabled")]
        active = [i for i in range(len(sides)) if i not in disabled]
        cancel = _CancelState(len(sides))
        # Сначала отменяем брошенные запросы к тем же моделям, и только потом
        # регистрируем свой: иначе cancel_model отменил бы сам себя (0185).
        for index in active:
            side = sides[index]
            try:
                self.server.cancel_model(
                    side["backend"], side["model"], reason="superseded"
                )
            except Exception:  # noqa: BLE001 — отмена старого не должна рвать новый
                logger.exception("отмена предыдущего запроса не удалась")
        chat_token = self.server.register_chat(sides, cancel)
        started = time.monotonic()
        # CPU-время процесса на запрос: ``process_time`` считает все потоки
        # этого процесса, поэтому снимаем его вокруг всего запроса, а не
        # внутри колонки (две колонки идут параллельно).
        cpu_started = time.process_time()
        questions = [_last_question(messages) for messages in side_messages]
        threads: list[threading.Thread] = []
        for index in active:
            side = sides[index]
            messages = list(side_messages[index])
            thread = threading.Thread(
                target=_run_side,
                args=(
                    index,
                    side,
                    messages,
                    max_tokens,
                    started,
                    events,
                    cancel,
                    self.server,
                    _prompt_text(messages),
                    tuple(facts),
                ),
                name="chat-side-{}".format(index),
                daemon=True,
            )
            thread.start()
            threads.append(thread)

        remaining = len(active)
        finished_normally = False
        try:
            self._open_sse()
            for index in disabled:
                side = sides[index]
                self._send_sse_data(
                    {
                        "type": "start",
                        "side": index,
                        "backend": "none",
                        "backend_label": "выключена",
                        "model": "",
                        "model_label": "",
                        "cold": False,
                    }
                )
                self._send_sse_data(
                    {
                        "type": "done",
                        "side": index,
                        "backend": "none",
                        "model": "",
                        "reason": "disabled",
                    }
                )
            while remaining > 0:
                try:
                    # Ждём событие коротко: пока генерация молчит (prefill
                    # бывает десятки секунд), писать нечего, и обрыв клиента
                    # раньше замечался только на первой записи — то есть уже
                    # после первого токена (задача 0185).
                    item = events.get(timeout=CLIENT_POLL_S)
                except queue.Empty:
                    if self._client_gone():
                        logger.info(
                            "SSE: клиент закрыл соединение — отменяю генерацию"
                        )
                        # Соединение закрыто: keep-alive-цикл не должен
                        # читать следующий запрос из мёртвого сокета.
                        self.close_connection = True
                        self._cancel_sides(sides, active, cancel, "client_gone")
                        break
                    continue
                if isinstance(item, _SideDone):
                    remaining -= 1
                    # Ответ дописан — только теперь, после done (и только
                    # непустой и без ошибки; см. _SideDone.answer).
                    self._remember(item.side, questions, item.answer)
                    continue
                if cancel.all.is_set():
                    break
                if not self._send_sse_data(item):
                    self._cancel_sides(sides, active, cancel, "client_gone")
                    break
            else:
                finished_normally = True
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
                    self._cancel_sides(sides, active, cancel, "client_gone")
                else:
                    self._send_sse_done()
        except _CLIENT_GONE:
            self.close_connection = True
        finally:
            self.server.unregister_chat(chat_token)
            # Если стрим не дошёл до конца (обрыв, отмена, исключение), C-рантайм
            # не должен дорабатывать ответ до max_tokens: отменяем и ждём
            # колонки. Отмена идёт через атомарный флаг C-API, поэтому join
            # короткий (штатно <100 мс), запас — на медленный токен.
            if not finished_normally and remaining > 0:
                self._cancel_sides(sides, active, cancel, "teardown")
            for thread in threads:
                thread.join(timeout=2.0)

    def _cancel_sides(
        self,
        sides: Sequence[Mapping[str, Any]],
        indices: Sequence[int],
        cancel: "_CancelState",
        reason: str,
    ) -> int:
        """Попросить колонки остановиться и вернуть число принятых отмен.

        ``cancel.request_side`` нужен и тогда, когда бэкенд отменять нечего
        (Ollama без cancel): колонка увидит запрос и закроет поток.
        """
        accepted = 0
        for index in indices:
            cancel.request_side(index, reason)
            adapter = sides[index].get("adapter")
            if adapter is not None and _cancel_backend(adapter):
                accepted += 1
        return accepted

    def _client_gone(self) -> bool:
        """Закрыл ли клиент соединение — без ожидания следующей записи.

        ``select`` говорит «читаемо», если пришёл EOF (или RST): после тела
        POST клиент больше ничего не присылает, поэтому читаемость == обрыв.
        ``MSG_PEEK`` не забирает байты: данные (если вдруг пришли) остаются
        сокету, а нам важен только факт EOF.
        """
        sock = getattr(self, "connection", None)
        if sock is None:
            return False
        try:
            readable, _, _ = select.select([sock], [], [], 0)
        except (OSError, ValueError):
            return True
        if not readable:
            return False
        try:
            return sock.recv(1, socket.MSG_PEEK) == b""
        except (BlockingIOError, InterruptedError):
            return False
        except OSError:
            return True

    def _remember(
        self, side: int, questions: Sequence[str], answer: str
    ) -> None:
        """Дописать обмен в историю колонки (задача 0182).

        Пустой ответ (ошибка генерации, отмена, пустая выдача) не попадает
        в историю: иначе следующий вопрос «видел» бы пустую реплику
        ассистента. Вопрос тоже не сохраняется — пара дописывается только
        целиком.
        """
        question = questions[side] if 0 <= side < len(questions) else ""
        if not question or not answer.strip():
            return
        # Рассуждение в историю не кладём: <think> съедает половину контекста,
        # из-за чего следующий промпт длиннее, а ответ приходит позже.
        cleaned = _strip_think(answer)
        if not cleaned.strip():
            return
        self.server.history.append(side, question, cleaned)

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
    keep_alive: float | None = None,
    history_tokens: int = DEFAULT_HISTORY_TOKENS,
    history_turns: int = DEFAULT_HISTORY_TURNS,
    think: bool = False,
    reasoning_budget: int = DEFAULT_REASONING_BUDGET,
    facts: bool = True,
    facts_limit: int = DEFAULT_FACT_LIMIT,
    plugins: PluginRegistry | None = None,
    tts: Callable[[str, str | None], bytes] | None = None,
) -> ChatHTTPServer:
    """Собрать сервер; ``port=0`` — свободный порт (удобно тестам).

    ``registry``/``adapters`` инжектируются тестами: реальные модели при этом
    не нужны. ``system_prompt=None`` — взять ``prompts/system.assistant.md``;
    пустая строка отключает системное сообщение. ``keep_alive`` — секунды
    простоя до автовыгрузки ВТМ (``None`` — выключено, ``0`` — держать всегда).
    ``history_tokens`` — бюджет истории колонки в токенах (``0`` — истории
    нет), ``history_turns`` — максимум последних реплик.
    ``think`` — разрешить рассуждение у ВТМ (по умолчанию нет);
    ``reasoning_budget`` — сколько токенов на него, сверх ``max_tokens``.
    ``facts`` — подкладывать ли в системный блок секцию ``СПРАВКА`` со свежими
    данными (время, курс ЦБ, котировка MOEX; ``--no-facts`` выключает).
    ``facts_limit`` — потолок фактов на запрос; ``plugins`` — готовый реестр
    плагинов (по умолчанию ``plugins/`` + ``config/plugins.json``).
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
    if isinstance(history_tokens, bool) or not isinstance(history_tokens, int):
        raise TypeError("history_tokens: ожидалось целое число")
    if history_tokens < 0:
        raise ValueError("history_tokens: ожидалось >= 0 (0 — без истории)")
    if isinstance(history_turns, bool) or not isinstance(history_turns, int):
        raise TypeError("history_turns: ожидалось целое число")
    if history_turns < 1:
        raise ValueError(
            "history_turns: ожидалось >= 1 (выключить историю — history_tokens=0)"
        )
    if keep_alive is not None:
        if isinstance(keep_alive, bool) or not isinstance(keep_alive, (int, float)):
            raise TypeError("keep_alive: ожидалось число секунд или None")
        if keep_alive < 0:
            raise ValueError("keep_alive: ожидалось >= 0 (0 — держать всегда)")
    if isinstance(reasoning_budget, bool) or not isinstance(reasoning_budget, int):
        raise TypeError("reasoning_budget: ожидалось целое число")
    if reasoning_budget < 0:
        raise ValueError("reasoning_budget: ожидалось >= 0")
    if isinstance(facts_limit, bool) or not isinstance(facts_limit, int):
        raise TypeError("facts_limit: ожидалось целое число")
    if facts_limit < 0:
        raise ValueError("facts_limit: ожидалось >= 0")

    resolved = tuple(registry) if registry is not None else default_registry()
    resolved_adapters = (
        dict(adapters)
        if adapters is not None
        else default_adapters(
            resolved,
            timeout=timeout,
            think=bool(think),
            reasoning_budget=int(reasoning_budget),
        )
    )
    prompt = load_system_prompt() if system_prompt is None else str(system_prompt)
    return ChatHTTPServer(
        (host, int(port)),
        registry=resolved,
        adapters=resolved_adapters,
        system_prompt=prompt,
        max_tokens=max_tokens,
        keep_alive=keep_alive,
        history_tokens=history_tokens,
        history_turns=history_turns,
        think=bool(think),
        reasoning_budget=int(reasoning_budget),
        facts=bool(facts),
        facts_limit=int(facts_limit),
        plugins=plugins,
        tts=tts,
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
        help="сколько новых токенов ответа просить у моделей "
        "(с --think это лимит уже ответа)",
    )
    parser.add_argument(
        "--think",
        action="store_true",
        help="разрешить рассуждение <think> у ВТМ с Qwen-шаблоном "
        "(по умолчанию выключено: ответ сразу, без обрыва по max_tokens)",
    )
    parser.add_argument(
        "--reasoning-budget",
        type=int,
        default=DEFAULT_REASONING_BUDGET,
        metavar="N",
        help="токенов на рассуждение при --think (по умолчанию {}); "
        "при исчерпании ответ досказывается вторым прогоном".format(
            DEFAULT_REASONING_BUDGET
        ),
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
        "--keep-alive",
        type=float,
        default=None,
        metavar="СЕК",
        help="выгружать ВТМ после N секунд бездействия (0 — держать всегда; "
        "по умолчанию автовыгрузка выключена)",
    )
    parser.add_argument(
        "--history-tokens",
        type=int,
        default=DEFAULT_HISTORY_TOKENS,
        metavar="ТОК",
        help="бюджет истории диалога на колонку, токенов (оценка ~3 символа "
        "на токен; 0 — без истории, по умолчанию {})".format(
            DEFAULT_HISTORY_TOKENS
        ),
    )
    parser.add_argument(
        "--history-turns",
        type=int,
        default=DEFAULT_HISTORY_TURNS,
        metavar="N",
        help="сколько последних реплик (пар вопрос/ответ) держать в истории "
        "колонки (по умолчанию {})".format(DEFAULT_HISTORY_TURNS),
    )
    parser.add_argument(
        "--no-system-prompt",
        action="store_true",
        help="не подмешивать prompts/system.assistant.md",
    )
    parser.add_argument(
        "--no-facts",
        dest="facts",
        action="store_false",
        help="не подкладывать в системный блок СПРАВКУ со свежими данными "
        "(время, курс ЦБ, котировка MOEX); по умолчанию включено",
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
            keep_alive=args.keep_alive,
            history_tokens=args.history_tokens,
            history_turns=args.history_turns,
            think=bool(args.think),
            reasoning_budget=int(args.reasoning_budget),
            facts=bool(getattr(args, "facts", True)),
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
    if server.keep_alive is None:
        print("автовыгрузка ВТМ: выключена (--keep-alive N включит)")
    elif server.keep_alive <= 0:
        print("автовыгрузка ВТМ: выключена (0 — держать всегда)")
    else:
        print("автовыгрузка ВТМ: после {:.0f} с бездействия".format(server.keep_alive))
    if server.facts:
        print("свежие данные: включены (СПРАВКА со временем, курсом ЦБ, MOEX; --no-facts выключает)")
    else:
        print("свежие данные: выключены (--no-facts)")
    if server.history.enabled:
        print(
            "история диалога: {} реплик, ~{} токенов на колонку "
            "(оценка ~3 символа на токен)".format(
                server.history.max_turns, server.history.max_tokens
            )
        )
    else:
        print("история диалога: выключена (--history-tokens 0)")
    if server.think:
        print(
            "рассуждения ВТМ: включены, бюджет {} ток + {} на ответ "
            "(при исчерпании ответ досказывается)".format(
                server.reasoning_budget, server.max_tokens
            )
        )
    else:
        print(
            "рассуждения ВТМ: выключены (--think включит, "
            "--reasoning-budget N — бюджет)"
        )
    # Голос по умолчанию — Silero (локальный нейроголос). Его worker грузит
    # torch+модель ~1,5–2,5 с, поэтому поднимаем фоном заранее: первая фраза
    # звучит за 0,05–0,2 с вместо 3 с (задача 0195). Нет Silero — no-op.
    try:
        from eva.tts import prewarm_silero_worker

        prewarm_silero_worker()
    except Exception as exc:  # noqa: BLE001 - голос не должен мешать запуску
        logger.info("silero: прогрев не запущен: %s", exc)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nостановлен")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
