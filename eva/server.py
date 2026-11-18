"""OpenAI-совместимый HTTP-сервер EVA32 на стандартной библиотеке.

Контракт: ``docs/hybrid.md``, задача ``ai/tasks/0017-eva-server.md``.

Сервер — только транспорт. Он не решает, кто отвечает (скилл, малая модель или
большая), а принимает маршрутизатор снаружи: ``create_server(router=...)``.
В тестах вместо боевого маршрутизатора подставляется заглушка, поэтому тесты
не ходят в сеть и не зависят от моделей.

Ответ ``POST /v1/chat/completions`` повторяет формат OpenAI и дополнен полем
``eva`` (маршрут, скилл, цена, латентность): без него нельзя посчитать
себестоимость запроса.

С ``"stream": true`` сервер отдаёт ``text/event-stream``: чанки
``data: {...}\\n\\n`` с ``choices[0].delta.content`` и финальный ``data: [DONE]``.
Если у инжектированного маршрутизатора есть метод ``stream(request)`` — ответ
уходит по частям по мере генерации; если нет — готовый ответ приходит одним
чанком (без имитации стриминга). Боевая связка (``router=None``) стримит через
``eva.router.stream(req, cfg, deps)``: скилл отдаёт один кусок, модель — поток
от клиента, финальный кусок несёт маршрут и цену. Ошибки до первого чанка —
обычный JSON; обрыв потока после первого чанка — событие ``error`` и
``[DONE]``, сервер продолжает работать.

Ошибки клиента — 4xx с человеческим текстом, ошибка маршрутизатора — 500 с
человеческим текстом и идентификатором запроса; трейсбек уходит в лог сервера
и наружу не попадает.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import threading
import time
import uuid
from collections import deque
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from math import ceil
from typing import Any, Callable
from urllib.parse import urlsplit

__all__ = [
    "AccessGuard",
    "AUTH_HEADER",
    "DEFAULT_MODEL",
    "EvaHTTPServer",
    "EvaRequestHandler",
    "as_router",
    "as_stream_router",
    "build_request",
    "chat_chunk",
    "create_server",
    "default_router",
    "default_stream_router",
    "make_handler",
    "reply_to_openai",
    "version",
]

logger = logging.getLogger("eva.server")
# Библиотека не настраивает логирование: обработчик добавляет приложение
# (eva/__main__.py). NullHandler убирает только последний рубеж logging —
# вывод WARNING в stderr у процесса без настроенного журнала.
logger.addHandler(logging.NullHandler())

SERVICE_NAME = "eva-server"
#: Имя по умолчанию для ответа, если клиент не передал model.
DEFAULT_MODEL = "eva-hybrid"
#: Больше мегабайта тело чата не бывает: защита от случайного мусора.
MAX_BODY_BYTES = 1024 * 1024
#: Заголовок с ключом доступа, как у OpenAI-совместимых клиентов.
AUTH_HEADER = "Authorization"
#: Схема авторизации из контракта; сравнивается без учёта регистра.
_BEARER = "bearer"
#: Окно лимита: запросов в минуту на ключ.
RATE_LIMIT_WINDOW_S = 60.0
#: Больше стольких ключей лимита в памяти не держим: устаревшие вычищаются.
_MAX_RATE_KEYS = 4096

#: Сентинел «поток не дал ни одного куска» для ``next(..., _STREAM_END)``.
_STREAM_END = object()

_VERSION_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "VERSION"
)


def version() -> str:
    """Версия бандла из файла ``VERSION``; ``0.0.0``, если файла нет."""
    try:
        with open(_VERSION_FILE, encoding="utf-8") as handle:
            value = handle.read().strip()
    except OSError:
        return "0.0.0"
    return value or "0.0.0"


# --------------------------------------------------------------------------- #
# Доступ: токен и лимит запросов
# --------------------------------------------------------------------------- #


class AccessGuard:
    """Аутентификация по Bearer-токену и лимит запросов на ключ.

    Потокобезопасен: ``ThreadingHTTPServer`` обслуживает запросы параллельно,
    поэтому счётчики закрыты одним замком. Ключ лимита — предъявленный токен,
    а в режиме без токена — адрес клиента: разные ключи друг другу не мешают,
    а локальный «умный» клиент всё равно не выжжет бюджет.

    Окно скользящее (``window_s``, по умолчанию минута): хранятся отметки
    времени запросов, попавших в окно; ``Retry-After`` — сколько осталось до
    освобождения слота, но не меньше секунды. ``rate_limit_per_min <= 0``
    выключает лимит.
    """

    def __init__(
        self,
        token: str = "",
        rate_limit_per_min: int = 60,
        *,
        window_s: float = RATE_LIMIT_WINDOW_S,
    ) -> None:
        self.token = token.strip()
        self.rate_limit_per_min = max(0, int(rate_limit_per_min))
        self.window_s = max(1.0, float(window_s))
        self._lock = threading.Lock()
        self._hits: dict[str, deque[float]] = {}

    @property
    def enabled(self) -> bool:
        """Включена ли проверка токена."""
        return bool(self.token)

    def authorized(self, presented: str | None) -> bool:
        """Совпадает ли предъявленный токен с настроенным.

        Сравнение постоянного времени: иначе по задержке ответа токен можно
        подбирать посимвольно. Без настроенного токена пропускается любой.
        """
        if not self.token:
            return True
        if not presented:
            return False
        return hmac.compare_digest(
            self.token.encode("utf-8"), presented.strip().encode("utf-8")
        )

    def key_for(self, presented: str | None, client_ip: str = "") -> str:
        """Ключ лимита: токен, если он настроен, иначе адрес клиента."""
        if self.token:
            return presented or ""
        return client_ip or "unknown"

    def check(self, key: str, *, now: float | None = None) -> float:
        """Учитывает запрос: ``0.0`` — можно, иначе ``Retry-After``, сек.

        ``now`` — для тестов: подменяет ``time.monotonic()``, чтобы проверить
        скольжение окна без реального ожидания минуты.
        """
        if self.rate_limit_per_min <= 0:
            return 0.0
        moment = time.monotonic() if now is None else float(now)
        cutoff = moment - self.window_s
        with self._lock:
            self._prune(cutoff)
            hits = self._hits.setdefault(key, deque())
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= self.rate_limit_per_min:
                return max(1.0, hits[0] + self.window_s - moment)
            hits.append(moment)
            return 0.0

    def _prune(self, cutoff: float) -> None:
        """Выкидывает ключи, у которых все отметки старше окна (под замком)."""
        if len(self._hits) <= _MAX_RATE_KEYS:
            return
        stale = [
            key for key, hits in self._hits.items() if not hits or hits[-1] <= cutoff
        ]
        for key in stale:
            del self._hits[key]


# --------------------------------------------------------------------------- #
# Типы запроса
# --------------------------------------------------------------------------- #


def build_request(pairs: list[tuple[str, str]], meta: dict[str, Any]) -> Any:
    """Собирает ``eva.types.Request`` из пар (role, content) и метаданных."""
    from .types import Message, Request

    return Request(
        messages=[Message(role=role, content=content) for role, content in pairs],
        meta=meta,
    )


# --------------------------------------------------------------------------- #
# Маршрутизатор
# --------------------------------------------------------------------------- #

#: Маршрутизатор — функция ``request -> Answer`` (Answer из eva.types).
Router = Callable[[Any], Any]
#: Потоковый маршрутизатор — функция ``request -> iterable`` кусков ответа.
StreamRouter = Callable[[Any], Any]


def as_router(router: Any) -> Router:
    """Приводит инжектированный маршрутизатор к виду ``router(request)``.

    Принимает функцию или объект с методом ``answer(request)``. Модуль
    ``eva.router`` с трёхаргументным ``answer(req, cfg, deps)`` напрямую не
    подходит — для него есть :func:`default_router`.
    """
    if router is None:
        return default_router
    if callable(router):
        return router
    answer = getattr(router, "answer", None)
    if callable(answer):
        return answer
    raise TypeError(
        "router должен быть функцией router(request) -> Answer "
        "или объектом с методом answer(request)"
    )


def as_stream_router(router: Any) -> StreamRouter | None:
    """Достаёт у маршрутизатора необязательный метод ``stream(request)``.

    Маршрутизатор объявляет умение отдавать ответ по частям явно: атрибутом
    ``stream``, который вызывается как ``stream(request)`` и возвращает
    итератор кусков. Куском считается строка (текст дельты), объект/словарь с
    полем ``text`` (тогда текст уходит в ``delta.content``, а ``route``/``usage``
    запоминаются для финального чанка) или словарь с ``delta``. Нет такого
    метода — сервер отдаёт готовый ответ одним чанком.

    Отдельная функция, а не поле в :func:`as_router`: объект с ``answer`` и
    ``stream`` после ``as_router`` превращается в связанный метод и второй
    атрибут теряет.
    """
    if router is None:
        return None
    stream = getattr(router, "stream", None)
    if callable(stream):
        return stream
    return None


def _load_config() -> Any:
    """Конфиг из env: ``config.load()``, если он есть, иначе сам модуль."""
    from . import config as config_module

    load = getattr(config_module, "load", None)
    if callable(load):
        return load()
    return config_module


def default_router(request: Any) -> Any:
    """Боевая связка ``eva.router.answer(req, cfg, deps)``, собранная лениво.

    Лениво — потому что ``/health`` должен отвечать даже тогда, когда
    маршрутизатор ещё не собран; при ошибке клиент получает 500 с человеческим
    текстом, а не трейсбек.
    """
    try:
        from .router import answer
    except ImportError as exc:
        raise RuntimeError(
            "маршрутизатор eva.router не собран: запустите сервер с "
            "router=... или --router МОДУЛЬ:ФАБРИКА"
        ) from exc

    if not callable(answer):
        raise RuntimeError("в eva.router нет функции answer(req, cfg, deps)")
    return answer(request, _load_config(), None)


def default_stream_router(request: Any) -> Any:
    """Боевая потоковая связка ``eva.router.stream(req, cfg, deps)``.

    Отдельная функция, а не атрибут :func:`default_router`: ``as_router``
    превращает объект с ``answer``/``stream`` в связанный метод и теряет
    второй атрибут. Собирается лениво, как и ``default_router``: ``/health``
    и обычный ответ не должны зависеть от готовности потока.
    """
    try:
        from .router import stream
    except ImportError as exc:
        raise RuntimeError(
            "потоковый маршрутизатор eva.router не собран: запустите сервер с "
            "router=... или --router МОДУЛЬ:ФАБРИКА"
        ) from exc

    if not callable(stream):
        raise RuntimeError("в eva.router нет функции stream(req, cfg, deps)")
    return stream(request, _load_config(), None)


# --------------------------------------------------------------------------- #
# Перевод Answer в ответ OpenAI
# --------------------------------------------------------------------------- #


def _field(obj: Any, name: str, default: Any = None) -> Any:
    """Поле объекта или ключ словаря — маршрутизатор может вернуть и то и то."""
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return default


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _elapsed_ms(started: float) -> int:
    """Миллисекунды с момента ``started`` (``time.monotonic()``)."""
    return int((time.monotonic() - started) * 1000)


def _route_summary(route: Any) -> dict[str, Any]:
    if route is None:
        return {"route": "unknown", "reason": "", "confidence": 0.0}
    if isinstance(route, str):
        kind, reason, confidence = route, "", 0.0
    else:
        kind = _field(route, "kind", "unknown")
        reason = _field(route, "reason", "")
        confidence = _field(route, "confidence", 0.0)

    if not isinstance(kind, str) or not kind.strip():
        kind = "unknown"
    if not isinstance(reason, str):
        reason = ""
    confidence = max(0.0, min(1.0, _as_float(confidence)))

    return {
        "route": kind.strip(),
        "reason": reason.strip(),
        "confidence": round(confidence, 4),
    }


def _usage_summary(answer: Any) -> dict[str, int]:
    """``usage`` в форме OpenAI из ответа маршрутизатора."""
    raw_usage = _field(answer, "usage") or {}
    if not isinstance(raw_usage, Mapping):
        raw_usage = {}
    prompt_tokens = max(0, _as_int(raw_usage.get("prompt_tokens")))
    completion_tokens = max(0, _as_int(raw_usage.get("completion_tokens")))
    total_tokens = _as_int(raw_usage.get("total_tokens"))
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _eva_summary(answer: Any, latency_ms: int) -> dict[str, Any]:
    """Поле ``eva``: маршрут, скилл, цена и латентность ответа."""
    skill = _field(answer, "skill")
    if not isinstance(skill, str) or not skill.strip():
        skill = None
    cost_rub = max(0.0, _as_float(_field(answer, "cost_rub")))

    eva = _route_summary(_field(answer, "route"))
    eva["skill"] = skill
    eva["cost_rub"] = round(cost_rub, 4)
    measured = max(0, _as_int(latency_ms))
    eva["latency_ms"] = max(0, _as_int(_field(answer, "latency_ms"), measured))
    return eva


def chat_chunk(
    *,
    chunk_id: str,
    created: int,
    model: str,
    delta: Mapping[str, Any],
    finish_reason: str | None = None,
    eva: Mapping[str, Any] | None = None,
    usage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Один SSE-чанк в формате OpenAI (``chat.completion.chunk``).

    ``delta`` всегда содержит ``content`` (пустая строка, если дельты нет):
    клиент не должен проверять наличие ключа. ``finish_reason`` не ``None``
    только в последнем чанке ответа.
    """
    payload: dict[str, Any] = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {"index": 0, "delta": dict(delta), "finish_reason": finish_reason}
        ],
    }
    if usage is not None:
        payload["usage"] = dict(usage)
    if eva is not None:
        payload["eva"] = dict(eva)
    return payload


def reply_to_openai(
    answer: Any,
    *,
    model: str,
    request_id: str,
    latency_ms: int,
) -> dict[str, Any]:
    """Строит тело ответа OpenAI из ``eva.types.Answer`` (или словаря)."""
    text = _field(answer, "text")
    if not isinstance(text, str):
        raise TypeError("маршрутизатор вернул ответ без текста")

    return {
        "id": "chatcmpl-{}".format(request_id[:24]),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": _usage_summary(answer),
        "eva": _eva_summary(answer, latency_ms),
    }


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #


class _ClientError(Exception):
    """Ошибка клиента: статус, код и текст для ответа."""

    def __init__(
        self, message: str, *, status: int = 400, code: str = "invalid_request"
    ):
        super().__init__(message)
        self.status = status
        self.code = code


def _content_to_text(content: Any) -> str | None:
    """Текст сообщения: строка или список частей ``{"type": "text", "text": ...}``."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if not isinstance(part, Mapping):
                return None
            text = part.get("text")
            if not isinstance(text, str):
                return None
            parts.append(text)
        return "".join(parts)
    return None


def _coerce_stream_piece(item: Any) -> tuple[str, Any]:
    """Кусок потока -> ``(текст дельты, Answer-метаданные | None)``.

    Понимает строку, объект/словарь с ``text`` (Answer-подобный: у него есть
    ``route``, метаданные пойдут в финальный чанк), ``delta`` строкой и
    ``{"delta": {"content": ...}}``. Всё прочее — ошибка маршрутизатора.
    """
    if isinstance(item, str):
        return item, None
    if isinstance(item, bytes):
        try:
            return item.decode("utf-8"), None
        except UnicodeDecodeError as exc:
            raise TypeError("кусок потока не в UTF-8") from exc

    text = _field(item, "text")
    if isinstance(text, str):
        meta = item if _field(item, "route") is not None else None
        return text, meta

    delta = _field(item, "delta")
    if isinstance(delta, str):
        return delta, None
    if isinstance(delta, Mapping):
        content = delta.get("content")
        if isinstance(content, str):
            return content, None

    content = _field(item, "content")
    if isinstance(content, str):
        return content, None

    raise TypeError(
        "кусок потока должен быть строкой или объектом с текстом: {!r}".format(item)
    )


def _close_iterator(iterator: Any) -> None:
    """Закрыть поток роутера, если он это умеет; ошибки закрытия не важны."""
    close = getattr(iterator, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # noqa: BLE001 - закрытие не должно ронять сервер
            logger.debug("ошибка закрытия потока", exc_info=True)


class EvaRequestHandler(BaseHTTPRequestHandler):
    """Один HTTP-запрос. Маршрутизатор задаётся подклассом из :func:`make_handler`."""

    server_version = "EVA32/1.0"
    protocol_version = "HTTP/1.1"
    _router: Router = staticmethod(default_router)
    #: Необязательный ``stream(request)``; ``None`` — стриминга у роутера нет.
    _stream_router: StreamRouter | None = None
    #: Тело SSE пишется chunked (HTTP/1.1) или до закрытия соединения (1.0).
    _sse_chunked = True

    # -- маршрутизация методов ------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - имя задано BaseHTTPRequestHandler
        if not self._check_access():
            return
        path = self._path()
        if path == "/health":
            self._send_health()
        elif path == "/v1/models":
            self._send_models()
        elif path == "/v1/chat/completions":
            self._send_method_not_allowed("POST")
        else:
            self._send_not_found(path)

    def do_POST(self) -> None:  # noqa: N802
        if not self._check_access():
            return
        path = self._path()
        if path == "/v1/chat/completions":
            self._handle_chat()
        elif path in ("/health", "/v1/models"):
            self._send_method_not_allowed("GET")
        else:
            self._send_not_found(path)

    def do_PUT(self) -> None:  # noqa: N802
        if self._check_access():
            self._handle_unsupported_method()

    def do_PATCH(self) -> None:  # noqa: N802
        if self._check_access():
            self._handle_unsupported_method()

    def do_DELETE(self) -> None:  # noqa: N802
        if self._check_access():
            self._handle_unsupported_method()

    def do_OPTIONS(self) -> None:  # noqa: N802
        if not self._check_access():
            return
        path = self._path()
        allow = self._allowed_methods(path)
        if allow is None:
            self._send_not_found(path)
        else:
            self._send_no_content(headers={"Allow": allow})

    # -- доступ ---------------------------------------------------------------

    def _bearer_token(self) -> str | None:
        """Токен из ``Authorization: Bearer ...``; ``None`` — заголовка нет."""
        raw = self.headers.get(AUTH_HEADER)
        if not isinstance(raw, str):
            return None
        scheme, separator, value = raw.strip().partition(" ")
        if not separator or scheme.lower() != _BEARER:
            return None
        token = value.strip()
        return token or None

    def _check_access(self) -> bool:
        """Аутентификация и лимит запросов; ``False`` — ответ уже отправлен.

        ``GET /health`` открыт намеренно: по нему проверяют живость процесса,
        и он ничего не стоит. Всё остальное (включая ``/v1/models`` и
        незнакомые пути) при заданном ``EVA_AUTH_TOKEN`` требует Bearer-токен.
        Проверка идёт до чтения тела: без токена байты не читаем и 413 не
        выдаём — сначала 401.
        """
        if self._path() == "/health":
            return True
        guard: AccessGuard | None = getattr(self.server, "access_guard", None)
        if guard is None:
            return True

        presented = self._bearer_token()
        if not guard.authorized(presented):
            logger.warning(
                "%s: отказ в доступе: нет валидного заголовка %s: Bearer",
                self.address_string(),
                AUTH_HEADER,
            )
            self._send_error(
                401,
                "Нужен заголовок Authorization: Bearer <токен>. Токен задаётся "
                "на сервере переменной EVA_AUTH_TOKEN.",
                "unauthorized",
                headers={"WWW-Authenticate": 'Bearer realm="eva"'},
            )
            return False

        key = guard.key_for(presented, self.client_address[0])
        retry_after = guard.check(key)
        if retry_after <= 0.0:
            return True

        seconds = max(1, int(ceil(retry_after)))
        logger.warning(
            "%s: превышен лимит %s запросов в минуту, повтор через %s с",
            self.address_string(),
            guard.rate_limit_per_min,
            seconds,
        )
        self._send_error(
            429,
            "Превышен лимит запросов: {} в минуту на ключ. "
            "Повторите через {} с.".format(guard.rate_limit_per_min, seconds),
            "rate_limited",
            headers={
                "Retry-After": str(seconds),
                "X-RateLimit-Limit": str(guard.rate_limit_per_min),
                "X-RateLimit-Remaining": "0",
            },
        )
        return False

    # -- эндпоинты ------------------------------------------------------------

    def _send_health(self) -> None:
        started_at = getattr(self.server, "started_at", None)
        uptime = round(time.monotonic() - started_at, 3) if started_at else 0.0
        self._send_json(
            200,
            {
                "status": "ok",
                "service": SERVICE_NAME,
                "version": version(),
                "model": DEFAULT_MODEL,
                "router": getattr(self.server, "router_name", "unknown"),
                "uptime_s": uptime,
            },
        )

    def _send_models(self) -> None:
        self._send_json(
            200,
            {
                "object": "list",
                "data": [
                    {
                        "id": DEFAULT_MODEL,
                        "object": "model",
                        "created": 0,
                        "owned_by": "eva32",
                    }
                ],
            },
        )

    def _handle_chat(self) -> None:
        request_id = uuid.uuid4().hex
        started = time.monotonic()

        try:
            payload = self._read_json_body()
            pairs, meta, model, stream, include_usage = self._parse_chat(payload)
        except _ClientError as exc:
            logger.info("запрос %s отклонён (%s): %s", request_id, exc.code, exc)
            self._send_error(exc.status, str(exc), exc.code)
            return

        if stream:
            self._handle_chat_stream(
                pairs,
                meta,
                model=model,
                include_usage=include_usage,
                request_id=request_id,
                started=started,
            )
            return

        try:
            request = build_request(pairs, meta)
            answer = self._router(request)
            body = reply_to_openai(
                answer,
                model=model,
                request_id=request_id,
                latency_ms=_elapsed_ms(started),
            )
        except Exception:  # noqa: BLE001 - наружу только текст, трейсбек в лог
            logger.exception("ошибка обработки запроса %s", request_id)
            self._send_internal_error(request_id)
            return

        self._send_json(200, body, headers={"X-Request-Id": request_id})

    # -- потоковая выдача -----------------------------------------------------

    def _handle_chat_stream(
        self,
        pairs: list[tuple[str, str]],
        meta: dict[str, Any],
        *,
        model: str,
        include_usage: bool,
        request_id: str,
        started: float,
    ) -> None:
        """SSE: куски из ``router.stream(request)`` или готовый ответ одним чанком.

        Всё, что может упасть до первого чанка (сборка запроса, первый кусок),
        отдаётся обычным JSON-ответом 500. Ошибка после первого чанка уже не
        может стать JSON — уходит событием ``data: {"error": ...}`` и ``[DONE]``.
        """
        body: dict[str, Any] | None = None
        iterator: Any = None

        try:
            request = build_request(pairs, meta)
            stream_fn = self._stream_router
            if stream_fn is None:
                # Роутер не умеет по частям — отдаём готовый ответ одним чанком
                # (честно: это не поток, а один кусок уже сгенерированного текста).
                answer = self._router(request)
                body = reply_to_openai(
                    answer,
                    model=model,
                    request_id=request_id,
                    latency_ms=_elapsed_ms(started),
                )
                first_text = body["choices"][0]["message"]["content"]
                first_meta = None
            else:
                iterator = iter(stream_fn(request))
                first = next(iterator, _STREAM_END)
                if first is _STREAM_END:
                    first_text, first_meta = "", None
                else:
                    first_text, first_meta = _coerce_stream_piece(first)
        except Exception:  # noqa: BLE001 - до первого чанка ещё можно ответить JSON
            _close_iterator(iterator)
            logger.exception("ошибка обработки запроса %s", request_id)
            self._send_internal_error(request_id)
            return

        self._open_sse(request_id)
        created = int(time.time())
        chunk_id = "chatcmpl-{}".format(request_id[:24])

        if body is not None:
            self._send_sse_data(
                chat_chunk(
                    chunk_id=chunk_id,
                    created=created,
                    model=model,
                    delta={"role": "assistant", "content": first_text},
                    finish_reason="stop",
                    eva=body["eva"],
                )
            )
            if include_usage:
                self._send_sse_usage(chunk_id, created, model, body["usage"])
            self._send_sse_done()
            return

        answer_meta = first_meta
        sent_any = False
        try:
            if first_text:
                if not self._send_sse_data(
                    chat_chunk(
                        chunk_id=chunk_id,
                        created=created,
                        model=model,
                        delta={"role": "assistant", "content": first_text},
                    )
                ):
                    return
                sent_any = True

            for item in iterator:
                text, item_meta = _coerce_stream_piece(item)
                if item_meta is not None:
                    answer_meta = item_meta
                if not text:
                    continue
                delta: dict[str, Any] = {"content": text}
                if not sent_any:
                    delta["role"] = "assistant"
                if not self._send_sse_data(
                    chat_chunk(
                        chunk_id=chunk_id,
                        created=created,
                        model=model,
                        delta=delta,
                    )
                ):
                    return
                sent_any = True

            eva = (
                _eva_summary(answer_meta, _elapsed_ms(started))
                if answer_meta is not None
                else None
            )
            if not self._send_sse_data(
                chat_chunk(
                    chunk_id=chunk_id,
                    created=created,
                    model=model,
                    delta={"content": ""},
                    finish_reason="stop",
                    eva=eva,
                )
            ):
                return
            if include_usage and answer_meta is not None:
                self._send_sse_usage(
                    chunk_id, created, model, _usage_summary(answer_meta)
                )
            self._send_sse_done()
        except Exception:  # noqa: BLE001 - заголовки уже отправлены, только SSE
            logger.exception("ошибка потока %s", request_id)
            if self._send_sse_error(request_id):
                self._send_sse_done()
        finally:
            # Клиент мог уйти или поток оборваться: закрываем генератор
            # роутера, чтобы клиент провайдера не висел до сборки мусора.
            _close_iterator(iterator)

    # -- SSE ------------------------------------------------------------------

    def _open_sse(self, request_id: str) -> None:
        """Заголовки SSE. HTTP/1.1 — chunked (keep-alive), HTTP/1.0 — до закрытия."""
        self._sse_chunked = self.request_version != "HTTP/1.0"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Request-Id", request_id)
        # Прокси не должны копить поток в буфере.
        self.send_header("X-Accel-Buffering", "no")
        if self._sse_chunked:
            self.send_header("Connection", "keep-alive")
            self.send_header("Transfer-Encoding", "chunked")
        else:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()

    def _write_sse(self, payload: bytes) -> bool:
        """Пишет одно событие ``data: ...\\n\\n``; ``False`` — клиент ушёл."""
        event = b"data: " + payload + b"\n\n"
        try:
            if self._sse_chunked:
                self.wfile.write(b"%x\r\n" % len(event) + event + b"\r\n")
            else:
                self.wfile.write(event)
            self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("клиент закрыл соединение во время стрима")
            self.close_connection = True
            return False

    def _send_sse_data(self, payload: Mapping[str, Any]) -> bool:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return self._write_sse(body.encode("utf-8"))

    def _send_sse_usage(
        self, chunk_id: str, created: int, model: str, usage: Mapping[str, Any]
    ) -> bool:
        """Отдельный чанк с ``usage`` (``stream_options.include_usage``)."""
        return self._send_sse_data(
            {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [],
                "usage": dict(usage),
            }
        )

    def _send_sse_error(self, request_id: str) -> bool:
        """Ошибка после первого чанка: JSON уже не отправить, шлём событие."""
        return self._send_sse_data(
            {
                "error": {
                    "message": "Внутренняя ошибка сервера. Повторите запрос позже; "
                    "если не поможет, сообщите код {}.".format(request_id),
                    "type": "server_error",
                    "code": "stream_error",
                    "param": None,
                }
            }
        )

    def _send_sse_done(self) -> None:
        """Терминальное событие ``data: [DONE]`` и конец chunked-тела."""
        if not self._write_sse(b"[DONE]"):
            return
        if self._sse_chunked:
            try:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                logger.debug("клиент закрыл соединение до конца стрима")
                self.close_connection = True

    # -- разбор запроса -------------------------------------------------------

    def _read_json_body(self) -> Any:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise _ClientError(
                "Не указан заголовок Content-Length: сервер не знает, сколько "
                "читать.",
                status=411,
                code="length_required",
            )
        try:
            length = int(raw_length)
        except (TypeError, ValueError):
            raise _ClientError(
                "Некорректный заголовок Content-Length: ожидалось целое число.",
                code="invalid_content_length",
            ) from None

        if length <= 0:
            raise _ClientError("Пустое тело запроса.", code="empty_body")
        if length > MAX_BODY_BYTES:
            raise _ClientError(
                "Тело запроса больше {} байт — такой чат не поддерживается.".format(
                    MAX_BODY_BYTES
                ),
                status=413,
                code="payload_too_large",
            )

        body = self.rfile.read(length)
        if not body:
            raise _ClientError("Пустое тело запроса.", code="empty_body")

        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError:
            raise _ClientError(
                "Тело запроса должно быть в кодировке UTF-8.",
                code="invalid_encoding",
            ) from None

        if not text.strip():
            raise _ClientError("Пустое тело запроса.", code="empty_body")

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise _ClientError(
                "Некорректный JSON: {} (строка {}, позиция {}).".format(
                    exc.msg, exc.lineno, exc.colno
                ),
                code="invalid_json",
            ) from None

    def _parse_chat(
        self, payload: Any
    ) -> tuple[list[tuple[str, str]], dict, str, bool, bool]:
        """Разбор тела чата -> (pairs, meta, model, stream, include_usage)."""
        if not isinstance(payload, Mapping):
            raise _ClientError(
                "Тело запроса должно быть JSON-объектом.", code="invalid_body"
            )

        stream_raw = payload.get("stream", False)
        if not isinstance(stream_raw, bool):
            raise _ClientError(
                "Поле stream должно быть true или false.",
                code="invalid_stream",
            )
        stream = stream_raw

        include_usage = False
        stream_options = payload.get("stream_options")
        if stream_options is not None:
            if not isinstance(stream_options, Mapping):
                raise _ClientError(
                    "Поле stream_options должно быть объектом.",
                    code="invalid_stream_options",
                )
            include_usage = stream_options.get("include_usage") is True

        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise _ClientError(
                "Поле messages обязательно и должно быть непустым списком.",
                code="invalid_messages",
            )

        pairs: list[tuple[str, str]] = []
        for index, item in enumerate(messages):
            if not isinstance(item, Mapping):
                raise _ClientError(
                    "messages[{}] должен быть объектом с полями role и content.".format(
                        index
                    ),
                    code="invalid_messages",
                )
            role = item.get("role")
            if not isinstance(role, str) or not role.strip():
                raise _ClientError(
                    "messages[{}].role должен быть непустой строкой.".format(index),
                    code="invalid_messages",
                )
            content = _content_to_text(item.get("content"))
            if content is None:
                raise _ClientError(
                    "messages[{}].content должен быть строкой или списком частей "
                    "с полем text.".format(index),
                    code="invalid_messages",
                )
            pairs.append((role.strip(), content))

        meta: dict[str, Any] = {"channel": "http"}
        user = payload.get("user")
        if isinstance(user, str) and user.strip():
            meta["user_id"] = user.strip()

        requested_model = payload.get("model")
        model = (
            requested_model.strip()
            if isinstance(requested_model, str) and requested_model.strip()
            else DEFAULT_MODEL
        )
        return pairs, meta, model, stream, include_usage

    # -- ответы ---------------------------------------------------------------

    def _path(self) -> str:
        return urlsplit(self.path).path or "/"

    def _allowed_methods(self, path: str) -> str | None:
        if path in ("/health", "/v1/models"):
            return "GET"
        if path == "/v1/chat/completions":
            return "POST"
        return None

    def _handle_unsupported_method(self) -> None:
        path = self._path()
        allow = self._allowed_methods(path)
        if allow is None:
            self._send_not_found(path)
        else:
            self._send_method_not_allowed(allow)

    def _send_not_found(self, path: str) -> None:
        self._send_error(
            404,
            "Эндпоинт {} не найден. Доступны: GET /health, GET /v1/models, "
            "POST /v1/chat/completions.".format(path),
            "not_found",
        )

    def _send_method_not_allowed(self, allow: str) -> None:
        self._send_error(
            405,
            "Метод {} для {} не разрешён. Разрешено: {}.".format(
                self.command, self._path(), allow
            ),
            "method_not_allowed",
            headers={"Allow": allow},
        )

    def _send_json(
        self,
        status: int,
        payload: Any,
        *,
        headers: Mapping[str, str] | None = None,
        close: bool = False,
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        if close:
            # Тело могли не дочитать — keep-alive после ошибки небезопасен.
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        try:
            self.wfile.write(body)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("клиент закрыл соединение до отправки ответа")

    def _send_no_content(self, *, headers: Mapping[str, str] | None = None) -> None:
        """204: у ответа нет тела, поэтому нет и Content-Length (RFC 9110)."""
        self.send_response(204)
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()

    def _send_error(
        self,
        status: int,
        message: str,
        code: str,
        *,
        error_type: str = "invalid_request_error",
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self._send_json(
            status,
            {
                "error": {
                    "message": message,
                    "type": error_type,
                    "code": code,
                    "param": None,
                }
            },
            headers=headers,
            close=True,
        )

    def _send_internal_error(self, request_id: str) -> None:
        """500 с идентификатором запроса; трейсбек остаётся в логе."""
        self._send_error(
            500,
            "Внутренняя ошибка сервера. Повторите запрос позже; "
            "если не поможет, сообщите код {}.".format(request_id),
            "internal_error",
            error_type="server_error",
        )

    # -- журнал ---------------------------------------------------------------

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        logger.info("%s %s", self.address_string(), format % args)

    def log_error(self, format: str, *args: Any) -> None:  # noqa: A002
        logger.warning("%s %s", self.address_string(), format % args)


def make_handler(router: Any) -> type[EvaRequestHandler]:
    """Подкласс обработчика с вшитым маршрутизатором (без глобального состояния).

    ``router`` передаётся как есть: из него берётся и ``answer`` (через
    :func:`as_router`), и необязательный ``stream`` (через
    :func:`as_stream_router`). ``router=None`` — боевая связка: ``answer`` и
    ``stream`` берутся из ``eva.router`` раздельно, потому что это две
    функции модуля, а не методы одного объекта.
    """
    if router is None:
        router_impl: Any = default_router
        stream_impl: Any = default_stream_router
    else:
        router_impl = as_router(router)
        stream_impl = as_stream_router(router)

    class BoundHandler(EvaRequestHandler):
        _router = staticmethod(router_impl)

    BoundHandler._stream_router = (
        staticmethod(stream_impl) if stream_impl is not None else None
    )
    BoundHandler.__name__ = "EvaRequestHandler[{}]".format(
        getattr(router_impl, "__name__", type(router_impl).__name__)
    )
    return BoundHandler


class EvaHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer с отметкой старта для ``/health``."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        handler: type[EvaRequestHandler],
        *,
        router_name: str = "injected",
        access_guard: AccessGuard | None = None,
    ) -> None:
        self.router_name = router_name
        self.access_guard = access_guard
        self.started_at = time.monotonic()
        super().__init__(address, handler)


def _auth_settings() -> tuple[str, int]:
    """(токен, лимит) из ``eva.config`` — только эти две настройки."""
    from . import config as config_module

    return config_module.auth_settings()


def create_server(
    host: str = "127.0.0.1",
    port: int = 0,
    router: Any = None,
    *,
    auth_token: str | None = None,
    rate_limit_per_min: int | None = None,
) -> EvaHTTPServer:
    """Создаёт (но не запускает) сервер; ``port=0`` — свободный порт.

    ``router`` — функция ``router(request) -> Answer`` или объект с методом
    ``answer(request)``. Если у него есть ещё и ``stream(request)``,
    ``stream=true`` отдаёт ответ по частям. Без ``router`` используется боевая
    связка :func:`default_router`, которая собирается лениво.

    Доступ: ``auth_token`` и ``rate_limit_per_min`` по умолчанию берутся из
    окружения (``EVA_AUTH_TOKEN``, ``EVA_RATE_LIMIT_PER_MIN``). ``None`` —
    «взять из окружения», пустая строка и ``0`` — явно выключить проверку
    токена и лимит соответственно. Заданный токен включает требование
    ``Authorization: Bearer`` на всех путях, кроме ``GET /health``.
    """
    env_token, env_limit = _auth_settings()
    token = env_token if auth_token is None else auth_token.strip()
    limit = (
        env_limit
        if rate_limit_per_min is None
        else max(0, int(rate_limit_per_min))
    )

    server = EvaHTTPServer(
        (host, port),
        make_handler(router),
        router_name="default" if router is None else "injected",
        access_guard=AccessGuard(token=token, rate_limit_per_min=limit),
    )
    bound_host, bound_port = server.server_address[:2]
    logger.info("eva-server слушает http://%s:%s", bound_host, bound_port)
    if token:
        logger.info(
            "аутентификация включена (Authorization: Bearer), лимит %s "
            "запросов в минуту на ключ",
            limit,
        )
    else:
        logger.warning(
            "EVA_AUTH_TOKEN не задан: сервер работает без аутентификации. "
            "Кто дотянется до %s:%s, тот и тратит бюджет на большие модели — "
            "наружу не выставлять. Лимит: %s запросов в минуту на адрес клиента.",
            bound_host,
            bound_port,
            limit,
        )
    return server
