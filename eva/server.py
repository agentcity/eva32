"""OpenAI-совместимый HTTP-сервер EVA32 на стандартной библиотеке.

Контракт: ``docs/hybrid.md``, задача ``ai/tasks/0017-eva-server.md``.

Сервер — только транспорт. Он не решает, кто отвечает (скилл, малая модель или
большая), а принимает маршрутизатор снаружи: ``create_server(router=...)``.
В тестах вместо боевого маршрутизатора подставляется заглушка, поэтому тесты
не ходят в сеть и не зависят от моделей.

Ответ ``POST /v1/chat/completions`` повторяет формат OpenAI и дополнен полем
``eva`` (маршрут, скилл, цена, латентность): без него нельзя посчитать
себестоимость запроса.

Ошибки клиента — 4xx с человеческим текстом, ошибка маршрутизатора — 500 с
человеческим текстом и идентификатором запроса; трейсбек уходит в лог сервера
и наружу не попадает.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import urlsplit

__all__ = [
    "DEFAULT_MODEL",
    "EvaHTTPServer",
    "EvaRequestHandler",
    "as_router",
    "build_request",
    "create_server",
    "default_router",
    "make_handler",
    "reply_to_openai",
    "version",
]

logger = logging.getLogger("eva.server")

SERVICE_NAME = "eva-server"
#: Имя по умолчанию для ответа, если клиент не передал model.
DEFAULT_MODEL = "eva-hybrid"
#: Больше мегабайта тело чата не бывает: защита от случайного мусора.
MAX_BODY_BYTES = 1024 * 1024

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

    raw_usage = _field(answer, "usage") or {}
    if not isinstance(raw_usage, Mapping):
        raw_usage = {}
    prompt_tokens = max(0, _as_int(raw_usage.get("prompt_tokens")))
    completion_tokens = max(0, _as_int(raw_usage.get("completion_tokens")))
    total_tokens = _as_int(raw_usage.get("total_tokens"))
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens

    skill = _field(answer, "skill")
    if not isinstance(skill, str) or not skill.strip():
        skill = None
    cost_rub = max(0.0, _as_float(_field(answer, "cost_rub")))

    eva = _route_summary(_field(answer, "route"))
    eva["skill"] = skill
    eva["cost_rub"] = round(cost_rub, 4)
    measured = max(0, _as_int(latency_ms))
    eva["latency_ms"] = max(0, _as_int(_field(answer, "latency_ms"), measured))

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
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
        "eva": eva,
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


class EvaRequestHandler(BaseHTTPRequestHandler):
    """Один HTTP-запрос. Маршрутизатор задаётся подклассом из :func:`make_handler`."""

    server_version = "EVA32/1.0"
    protocol_version = "HTTP/1.1"
    _router: Router = staticmethod(default_router)

    # -- маршрутизация методов ------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - имя задано BaseHTTPRequestHandler
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
        path = self._path()
        if path == "/v1/chat/completions":
            self._handle_chat()
        elif path in ("/health", "/v1/models"):
            self._send_method_not_allowed("GET")
        else:
            self._send_not_found(path)

    def do_PUT(self) -> None:  # noqa: N802
        self._handle_unsupported_method()

    def do_PATCH(self) -> None:  # noqa: N802
        self._handle_unsupported_method()

    def do_DELETE(self) -> None:  # noqa: N802
        self._handle_unsupported_method()

    def do_OPTIONS(self) -> None:  # noqa: N802
        path = self._path()
        allow = self._allowed_methods(path)
        if allow is None:
            self._send_not_found(path)
        else:
            self._send_no_content(headers={"Allow": allow})

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
            pairs, meta, model = self._parse_chat(payload)
        except _ClientError as exc:
            logger.info("запрос %s отклонён (%s): %s", request_id, exc.code, exc)
            self._send_error(exc.status, str(exc), exc.code)
            return

        try:
            request = build_request(pairs, meta)
            answer = self._router(request)
            body = reply_to_openai(
                answer,
                model=model,
                request_id=request_id,
                latency_ms=int((time.monotonic() - started) * 1000),
            )
        except Exception:  # noqa: BLE001 - наружу только текст, трейсбек в лог
            logger.exception("ошибка обработки запроса %s", request_id)
            self._send_error(
                500,
                "Внутренняя ошибка сервера. Повторите запрос позже; "
                "если не поможет, сообщите код {}.".format(request_id),
                "internal_error",
                error_type="server_error",
            )
            return

        self._send_json(200, body, headers={"X-Request-Id": request_id})

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

    def _parse_chat(self, payload: Any) -> tuple[list[tuple[str, str]], dict, str]:
        if not isinstance(payload, Mapping):
            raise _ClientError(
                "Тело запроса должно быть JSON-объектом.", code="invalid_body"
            )

        if payload.get("stream") not in (None, False):
            raise _ClientError(
                "Потоковая выдача (stream=true) не поддерживается: повторите "
                "запрос без stream.",
                code="unsupported_stream",
            )

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
        return pairs, meta, model

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

    # -- журнал ---------------------------------------------------------------

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        logger.info("%s %s", self.address_string(), format % args)

    def log_error(self, format: str, *args: Any) -> None:  # noqa: A002
        logger.warning("%s %s", self.address_string(), format % args)


def make_handler(router: Any) -> type[EvaRequestHandler]:
    """Подкласс обработчика с вшитым маршрутизатором (без глобального состояния)."""
    router_impl = as_router(router)

    class BoundHandler(EvaRequestHandler):
        _router = staticmethod(router_impl)

    BoundHandler.__name__ = "EvaRequestHandler[{}]".format(
        getattr(router_impl, "__name__", type(router_impl).__name__)
    )
    return BoundHandler


class EvaHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer с отметкой старта для ``/health``."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], handler: type[EvaRequestHandler], *,
                 router_name: str = "injected") -> None:
        self.router_name = router_name
        self.started_at = time.monotonic()
        super().__init__(address, handler)


def create_server(
    host: str = "127.0.0.1",
    port: int = 0,
    router: Any = None,
) -> EvaHTTPServer:
    """Создаёт (но не запускает) сервер; ``port=0`` — свободный порт.

    ``router`` — функция ``router(request) -> Answer`` или объект с методом
    ``answer(request)``. Без ``router`` используется боевая связка
    :func:`default_router`, которая собирается лениво.
    """
    router_impl = as_router(router)
    server = EvaHTTPServer(
        (host, port),
        make_handler(router_impl),
        router_name="default" if router is None else "injected",
    )
    logger.info(
        "eva-server слушает http://%s:%s", server.server_address[0],
        server.server_address[1],
    )
    return server
