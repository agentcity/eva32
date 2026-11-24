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
* ``POST /api/chat`` — SSE-поток ``text/event-stream``. События:
  ``start`` (какая модель отвечает), ``delta`` (кусок текста), ``done``
  (время, токены, ток/с), ``error`` (человеческий текст). Терминальное
  событие — ``data: [DONE]``, как у ``eva/server.py``;
* ``GET /favicon.ico`` — 204, чтобы браузер не шумел 404.

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
import json
import logging
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence
from urllib.parse import urlsplit

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
    "create_server",
    "default_adapters",
    "default_registry",
    "main",
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
#: Демо-модели ВТМ: обе лежат в репозитории и используют тестовый BPE.
DEFAULT_VTM_MODELS: tuple[tuple[str, str], ...] = (
    ("runs/tiny/model.vtm", "runs/tiny/model.vtm — наша мини-модель (текст)"),
    ("demo/model.vtm", "demo/model.vtm — демо (vocab=128, без текста)"),
    ("model.vtm", "model.vtm — копия демо (vocab=128, без текста)"),
)
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
    for relative, label in DEFAULT_VTM_MODELS:
        path = ROOT / relative
        if not path.is_file():
            continue
        vtm.append(
            ModelSpec(
                id=relative,
                label=label,
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
    """

    def __init__(self, factory: Callable[[], Any], *, label: str = "") -> None:
        self._factory = factory
        self.label = label
        self._inner: Any = None
        # RLock: ``stream`` держит замок на всё время генерации и зовёт
        # ``_ensure``, которому нужен тот же замок.
        self._lock = threading.RLock()

    def _ensure(self) -> Any:
        if self._inner is None:
            with self._lock:
                if self._inner is None:
                    self._inner = self._factory()
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
                        yield StreamPiece(done=True, usage=dict(usage))
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
            )
    return adapters


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
) -> None:
    """Сгенерировать одну колонку и сложить события в очередь.

    Пишет в очередь только этот поток; HTTP-поток читает и отправляет SSE.
    Так две модели работают параллельно, а сокет не делят между потоками.
    """
    adapter = side["adapter"]
    events.put(
        {
            "type": "start",
            "side": index,
            "backend": side["backend"],
            "backend_label": side["backend_label"],
            "model": side["model"],
            "model_label": side["model_label"],
        }
    )
    first_token_ms: int | None = None
    usage: dict[str, Any] = {}
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
            events.put(
                {
                    "type": "done",
                    "side": index,
                    "backend": side["backend"],
                    "model": side["model"],
                    "elapsed_ms": elapsed_ms,
                    "first_token_ms": first_token_ms,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "tokens_per_second": tokens_per_second,
                    "estimated": estimated,
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
    });
  });

  loadCatalog();
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
        threads: list[threading.Thread] = []
        for index, side in enumerate(sides):
            thread = threading.Thread(
                target=_run_side,
                args=(index, side, messages, max_tokens, started, events, stop),
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
                if not self._send_sse_data({"type": "end"}):
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
