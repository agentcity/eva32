"""HTTP-клиент к OpenAI-совместимым эндпоинтам.

Контракт: `docs/hybrid.md` (раздел `eva/clients/openai_compat.py`), решение:
`docs/adr/0009-hybrid-router.md`. Только стандартная библиотека.

Клиент один и тот же для малой модели (локальный сервер, `http://127.0.0.1:…`)
и для большой (облако). Отличаются только `base`, `key` и `model`.

    from eva.clients import OpenAIClient, ClientError

    client = OpenAIClient("http://127.0.0.1:11434", "", "eva32-dev",
                          timeout=20, retries=2)
    result = client.chat([{"role": "user", "content": "привет"}])
    result.text, result.usage["cost_rub"], result.raw

    for chunk in client.stream_chat([{"role": "user", "content": "привет"}]):
        if chunk.text:
            print(chunk.text, end="", flush=True)

Устройство:

* сетевые вызовы — `urllib.request.urlopen` внутри `_post_json` (для
  `chat()`) и `_open_stream` (для `stream_chat()`); тесты подменяют
  транспорт, `stream_transport` или `_sleep` — реальных запросов и реальных
  задержек в тестах нет;
* повторяются только таймаут, сетевые ошибки, `429` и `5xx`; пауза —
  экспоненциальный backoff (`backoff * 2**n`), для `429` уважается
  `Retry-After`;
* после исчерпания попыток — `ClientError` с человеческим текстом, статусом,
  числом попыток и видом ошибки (`kind`); ключ в текст/`repr` не попадает;
* стоимость берётся из ответа (`usage.cost_rub`, затем `cost_rub` верхнего
  уровня), иначе считается по локальным тарифам, иначе `0.0`.
"""

from __future__ import annotations

import http.client
import io
import json
import logging
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ClientError",
    "ClientResult",
    "HTTPResponse",
    "OpenAIClient",
    "StreamChunk",
    "StreamResponse",
    "DEFAULT_BACKOFF",
    "DEFAULT_RETRIES",
    "DEFAULT_TIMEOUT",
    "MAX_BACKOFF",
]

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 60.0
DEFAULT_RETRIES = 2
DEFAULT_BACKOFF = 0.5
MAX_BACKOFF = 8.0
MAX_ERROR_BODY = 300
USER_AGENT = "eva32-clients/0.1 (+python-stdlib)"


# --------------------------------------------------------------------------- #
# Сечения для тестов и подмены транспорта
# --------------------------------------------------------------------------- #

def _sleep(seconds: float) -> None:
    """Пауза перед повтором. В тестах подменяется, чтобы не ждать вживую."""
    time.sleep(seconds)


@dataclass(frozen=True)
class HTTPResponse:
    """Ответ транспорта: статус, тело, заголовки (уже без исключений 4xx/5xx)."""

    status: int
    body: bytes = b""
    headers: Mapping[str, str] = field(default_factory=dict)

    def header(self, name: str, default: str | None = None) -> str | None:
        """Регистронезависимый доступ к заголовку."""
        target = name.lower()
        for key, value in self.headers.items():
            if str(key).lower() == target:
                return value
        return default


def _post_json(
    url: str,
    headers: Mapping[str, str],
    body: bytes,
    timeout: float,
) -> HTTPResponse:
    """Транспорт: один POST через urllib. Сетевые сбои — исключениями.

    Ответы 4xx/5xx не бросаются наружу: возвращаются как `HTTPResponse`,
    чтобы клиент сам решал, повторять или нет.
    """
    request = urllib.request.Request(
        url, data=body, headers=dict(headers), method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            payload = response.read()
            response_headers = dict(response.headers.items())
            return HTTPResponse(status=status, body=payload, headers=response_headers)
    except urllib.error.HTTPError as exc:  # он же ответ, просто с ошибкой
        try:
            payload = exc.read()
        except Exception:  # noqa: BLE001 - тело ошибки не критично
            payload = b""
        response_headers = dict(exc.headers.items()) if exc.headers else {}
        return HTTPResponse(status=exc.code, body=payload or b"", headers=response_headers)


@dataclass(frozen=True)
class StreamChunk(Mapping[str, Any]):
    """Один кусок потока от провайдера.

    Успешный поток — последовательность кусков с непустым ``text``; последний
    кусок всегда терминальный (``done=True``) и несёт ``usage``, даже если
    провайдер его не прислал (тогда ``usage == {}``). ``raw`` — разобранный
    JSON события, он нужен для отладки и телеметрии.
    """

    text: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None
    done: bool = False

    _FIELDS = ("text", "usage", "raw", "finish_reason", "done")

    def __getitem__(self, key: str) -> Any:
        if key in self._FIELDS:
            return getattr(self, key)
        raise KeyError(key)

    def __iter__(self):
        return iter(self._FIELDS)

    def __len__(self) -> int:
        return len(self._FIELDS)

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "usage": dict(self.usage),
            "raw": dict(self.raw),
            "finish_reason": self.finish_reason,
            "done": self.done,
        }

    def __repr__(self) -> str:
        text = self.text if len(self.text) <= 40 else self.text[:37] + "…"
        return (
            f"StreamChunk(text={text!r}, finish_reason={self.finish_reason!r}, "
            f"done={self.done}, usage={self.usage!r})"
        )


@dataclass
class StreamResponse:
    """Ответ транспорта на потоковый запрос.

    ``body`` — файлоподобный объект с ``readline()`` и ``close()`` (у
    ``urlopen`` это сам HTTP-ответ). ``status`` есть даже у ошибки: тело
    ошибочного ответа приходит уже прочитанным в ``io.BytesIO``.
    """

    status: int
    body: Any = None
    headers: Mapping[str, str] = field(default_factory=dict)

    def header(self, name: str, default: str | None = None) -> str | None:
        target = name.lower()
        for key, value in self.headers.items():
            if str(key).lower() == target:
                return value
        return default

    def close(self) -> None:
        body = self.body
        if body is not None:
            try:
                body.close()
            except Exception:  # noqa: BLE001 - закрытие не должно ронять поток
                pass


def _open_stream(
    url: str,
    headers: Mapping[str, str],
    body: bytes,
    timeout: float,
) -> StreamResponse:
    """Транспорт потока: POST и чтение SSE по строкам без буферизации всего тела.

    HTTP-ошибки не бросаются: тело читается ограниченно и отдаётся вместе со
    статусом, чтобы ``stream_chat`` сам решил, повторять или нет.
    """
    request = urllib.request.Request(
        url, data=body, headers=dict(headers), method="POST"
    )
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        try:
            payload = exc.read(MAX_ERROR_BODY * 8)
        except Exception:  # noqa: BLE001 - тело ошибки не критично
            payload = b""
        response_headers = dict(exc.headers.items()) if exc.headers else {}
        try:
            exc.close()
        except Exception:  # noqa: BLE001
            pass
        return StreamResponse(
            status=exc.code, body=io.BytesIO(payload or b""), headers=response_headers
        )
    return StreamResponse(
        status=getattr(response, "status", 200),
        body=response,
        headers=dict(response.headers.items()),
    )


# --------------------------------------------------------------------------- #
# Ошибки и результат
# --------------------------------------------------------------------------- #

class ClientError(RuntimeError):
    """Модель не ответила — понятным образом.

    Поля для вызывающего кода (роутер/политика):

    * `kind` — `"timeout"`, `"network"`, `"http"`, `"response"` (200, но
      разобрать тело не удалось), `"stream"` (поток оборвался) или `"config"`;
    * `status` — HTTP-код, если ошибка пришла от сервера, иначе `None`;
    * `attempts` — сколько попыток реально сделано (1…`retries + 1`);
    * `retryable` — была ли ошибка из класса повторяемых (даже если попытки
      кончились);
    * `detail` — короткая выжимка от провайдера, без ключа и без простыни.

    Текст исключения готов к показу человеку и не содержит ключа API.
    """

    def __init__(
        self,
        message: str,
        *,
        kind: str = "http",
        status: int | None = None,
        attempts: int = 0,
        retryable: bool = False,
        detail: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.kind = kind
        self.status = status
        self.attempts = attempts
        self.retryable = retryable
        self.detail = detail
        self.retry_after = retry_after

    def __str__(self) -> str:
        return self.message

    def __repr__(self) -> str:
        return (
            f"ClientError(kind={self.kind!r}, status={self.status!r}, "
            f"attempts={self.attempts}, message={self.message!r})"
        )


@dataclass(frozen=True)
class ClientResult(Mapping[str, Any]):
    """Результат `chat()`: и как объект (`res.text`), и как словарь (`res["text"]`).

    `usage` всегда содержит `prompt_tokens`, `completion_tokens`,
    `total_tokens` и `cost_rub` (рубли, `0.0` если провайдер цену не отдал и
    локальный тариф не задан).
    """

    text: str
    usage: dict[str, Any]
    raw: dict[str, Any]

    _FIELDS = ("text", "usage", "raw")

    def __getitem__(self, key: str) -> Any:
        if key in self._FIELDS:
            return getattr(self, key)
        raise KeyError(key)

    def __iter__(self):
        return iter(self._FIELDS)

    def __len__(self) -> int:
        return len(self._FIELDS)

    def as_dict(self) -> dict[str, Any]:
        return {"text": self.text, "usage": self.usage, "raw": self.raw}

    def __repr__(self) -> str:
        text = self.text if len(self.text) <= 60 else self.text[:57] + "…"
        return (
            f"ClientResult(text={text!r}, usage={self.usage!r}, "
            f"raw=<{len(self.raw)} ключей>)"
        )


# --------------------------------------------------------------------------- #
# Разбор значений
# --------------------------------------------------------------------------- #

def _as_int(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        return max(0, int(float(value)))
    except (TypeError, ValueError, OverflowError):
        return 0


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if number != number or number in (float("inf"), float("-inf")) or number < 0:
        return None
    return number


def _collapse(text: str, limit: int = MAX_ERROR_BODY) -> str:
    flat = " ".join(str(text).split())
    if len(flat) > limit:
        return flat[:limit].rstrip() + "…"
    return flat


def _error_detail(body: bytes | str) -> str:
    """Выжимка причины из тела ошибки: `error.message`, `message` или сырой текст."""
    if isinstance(body, bytes):
        text = body.decode("utf-8", "replace")
    else:
        text = body or ""
    text = text.strip()
    if not text:
        return ""
    parsed: Any = None
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        parsed = None
    if isinstance(parsed, dict):
        error = parsed.get("error")
        candidate: Any = None
        if isinstance(error, dict):
            candidate = error.get("message") or error.get("type") or error.get("code")
        elif isinstance(error, str):
            candidate = error
        if candidate is None:
            candidate = parsed.get("message") or parsed.get("detail")
        if isinstance(candidate, str) and candidate.strip():
            text = candidate
    return _collapse(text)


def _close_quietly(obj: Any) -> None:
    """Закрыть ответ/тело, не давая ошибке закрытия испортить поток."""
    close = getattr(obj, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # noqa: BLE001 - закрытие не должно ронять поток
            pass


def _iter_lines(body: Any):
    """Строки SSE из файлоподобного тела; ``readline`` — точка таймаута."""
    if body is None:
        return
    while True:
        line = body.readline()
        if not line:
            break
        yield line


def _decode_line(raw: Any) -> str:
    """Байты/строка -> строка без завершающего перевода строки."""
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", "replace")
    else:
        text = str(raw)
    return text.rstrip("\r\n")


def _delta_text(choice: Mapping[str, Any]) -> str:
    """Текст дельты из ``choices[0]`` (``delta.content`` или старый ``text``)."""
    delta = choice.get("delta")
    content: Any = None
    if isinstance(delta, Mapping):
        content = delta.get("content")
    elif isinstance(delta, str):
        content = delta
    if content is None:
        message = choice.get("message")
        if isinstance(message, Mapping):
            content = message.get("content")
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        content = "".join(parts)
    if isinstance(content, str):
        return content
    if isinstance(choice.get("text"), str):
        return choice["text"]
    return ""


def _build_endpoint(base: str) -> str:
    """Собрать URL `/chat/completions`, не удваивая `/v1`.

    Понимает все встречающиеся в проекте формы баз:

    * `http://127.0.0.1:11434`        → `…/v1/chat/completions`
    * `https://polza.ai/api`          → `…/api/v1/chat/completions`
    * `https://api.example.com/v1`    → `…/v1/chat/completions`
    * `https://…/v1/chat/completions` → без изменений
    """
    clean = (base or "").strip().rstrip("/")
    if not clean:
        raise ValueError("base не может быть пустым")
    if clean.endswith("/chat/completions"):
        return clean
    if clean.endswith("/v1"):
        return clean + "/chat/completions"
    return clean + "/v1/chat/completions"


def _normalize_messages(messages: Any) -> list[dict[str, str]]:
    """Привести сообщения к `[{"role", "content"}]` для тела запроса.

    Принимает `Message`-объекты (атрибуты `role`/`content`), словари,
    строки (как user-сообщение) и любую их последовательность.
    """
    if isinstance(messages, str):
        items: list[Any] = [messages]
    elif isinstance(messages, Mapping):
        items = [messages]
    else:
        try:
            items = list(messages)
        except TypeError as exc:
            raise TypeError(
                "messages: ожидается список сообщений (dict, Message или str)"
            ) from exc
    if not items:
        raise ValueError("messages пуст: нужно хотя бы одно сообщение")

    normalized: list[dict[str, str]] = []
    for index, item in enumerate(items):
        if isinstance(item, str):
            role, content = "user", item
        elif isinstance(item, Mapping):
            role, content = item.get("role"), item.get("content")
        else:
            role = getattr(item, "role", None)
            content = getattr(item, "content", None)
        if not isinstance(role, str) or not role.strip():
            raise ValueError(f"messages[{index}].role: нужна непустая строка")
        if content is None:
            content = ""
        if not isinstance(content, str):
            raise ValueError(f"messages[{index}].content: нужна строка")
        normalized.append({"role": role.strip(), "content": content})
    return normalized


# --------------------------------------------------------------------------- #
# Клиент
# --------------------------------------------------------------------------- #

class OpenAIClient:
    """Клиент к OpenAI-совместимому `/v1/chat/completions`.

    Параметры
    ---------
    base : str
        Корень API, например `http://127.0.0.1:11434` или
        `https://api.deepseek.com`. `/v1` и `/chat/completions` добавляются
        по необходимости (см. `_build_endpoint`).
    key : str
        Ключ API. Пустая строка — заголовок `Authorization` не отправляется
        (локальной модели ключ обычно не нужен).
    model : str
        Имя модели, уходит в тело запроса.
    timeout : float
        Таймаут одной попытки в секундах, по умолчанию 60.
    retries : int
        Сколько раз повторить после первой неудачи (всего попыток
        `retries + 1`), по умолчанию 2.

    Необязательные (совместимы с контрактом):
    backoff, max_backoff — параметры экспоненциальной паузы;
    price_prompt_rub, price_completion_rub — рубли за 1000 токенов, если
    провайдер не отдаёт `cost_rub` своим полем;
    transport, sleep — точки подмены для тестов (по умолчанию `_post_json`
    и `_sleep` модуля);
    stream_transport — точка подмены для `stream_chat()` (по умолчанию
    используется `transport`, если он задан, иначе `_open_stream`); нужна
    потому, что поток читается по строкам, а не целиком.

    `repr` не печатает ключ.
    """

    def __init__(
        self,
        base: str,
        key: str,
        model: str,
        timeout: float = DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
        *,
        backoff: float = DEFAULT_BACKOFF,
        max_backoff: float = MAX_BACKOFF,
        price_prompt_rub: float = 0.0,
        price_completion_rub: float = 0.0,
        transport: Any = None,
        sleep: Any = None,
        stream_transport: Any = None,
    ) -> None:
        if not isinstance(base, str):
            raise TypeError("base: нужна строка")
        parsed = urllib.parse.urlparse(base.strip())
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(
                f"base: ожидается http(s)://хост[:порт], получено {base!r}"
            )
        if not isinstance(key, str):
            raise TypeError("key: нужна строка (пустая допустима)")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model: нужна непустая строка")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise TypeError("timeout: нужно число секунд")
        if timeout <= 0:
            raise ValueError("timeout: должен быть больше нуля")
        if isinstance(retries, bool) or not isinstance(retries, int):
            raise TypeError("retries: нужно целое число")
        if retries < 0:
            raise ValueError("retries: не может быть отрицательным")
        if isinstance(backoff, bool) or not isinstance(backoff, (int, float)) or backoff < 0:
            raise ValueError("backoff: нужно неотрицательное число секунд")
        if (
            isinstance(max_backoff, bool)
            or not isinstance(max_backoff, (int, float))
            or max_backoff < 0
        ):
            raise ValueError("max_backoff: нужно неотрицательное число секунд")
        for name, price in (
            ("price_prompt_rub", price_prompt_rub),
            ("price_completion_rub", price_completion_rub),
        ):
            if isinstance(price, bool) or not isinstance(price, (int, float)) or price < 0:
                raise ValueError(f"{name}: нужно неотрицательное число рублей")

        self.base = base.strip().rstrip("/")
        self.key = key
        self.model = model.strip()
        self.timeout = float(timeout)
        self.retries = retries
        self.backoff = float(backoff)
        self.max_backoff = float(max_backoff)
        self.price_prompt_rub = float(price_prompt_rub)
        self.price_completion_rub = float(price_completion_rub)
        self.endpoint = _build_endpoint(self.base)
        self._transport = transport
        self._sleep = sleep
        self._stream_transport = stream_transport

    # -- публичное ---------------------------------------------------------- #

    def chat(self, messages: Any, max_tokens: int = 512) -> ClientResult:
        """Один запрос к модели; ретраи и разбор — внутри.

        Возвращает `ClientResult` (`text`, `usage`, `raw`) или бросает
        `ClientError` после исчерпания попыток.
        """
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            raise TypeError("max_tokens: нужно целое число")
        if max_tokens <= 0:
            raise ValueError("max_tokens: должен быть больше нуля")

        payload = {
            "model": self.model,
            "messages": _normalize_messages(messages),
            "max_tokens": max_tokens,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        }
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"

        transport = self._transport or _post_json
        sleep = self._sleep or _sleep
        total_attempts = self.retries + 1

        for attempt in range(total_attempts):
            attempts = attempt + 1
            retry_after: float | None = None
            try:
                response = transport(self.endpoint, headers, body, self.timeout)
            except (TimeoutError, socket.timeout) as exc:  # в 3.10+ это один класс
                failure = self._fail(
                    f"таймаут {self.timeout:g} с",
                    kind="timeout",
                    attempts=attempts,
                    retryable=True,
                    detail=_collapse(str(exc)) or None,
                )
            except urllib.error.URLError as exc:
                reason = getattr(exc, "reason", exc)
                if isinstance(reason, TimeoutError):
                    text, kind = f"таймаут {self.timeout:g} с", "timeout"
                elif isinstance(reason, socket.gaierror):
                    text, kind = f"DNS не разрешил хост: {reason}", "network"
                else:
                    text, kind = f"сеть недоступна: {reason}", "network"
                failure = self._fail(
                    text, kind=kind, attempts=attempts, retryable=True
                )
            except OSError as exc:
                failure = self._fail(
                    f"ошибка сокета: {exc}", kind="network", attempts=attempts, retryable=True
                )
            else:
                if 200 <= response.status < 300:
                    return self._parse(response, attempts=attempts)

                retry_after = self._retry_after(response)
                retryable = response.status == 429 or 500 <= response.status < 600
                reason = http.client.responses.get(response.status, "")
                detail = _error_detail(response.body)
                head = f"HTTP {response.status}"
                if reason:
                    head += f" {reason}"
                message = f"{head}: {detail}" if detail else head
                failure = self._fail(
                    message,
                    kind="http",
                    status=response.status,
                    attempts=attempts,
                    retryable=retryable,
                    detail=detail or None,
                )
                if not retryable:
                    log.error("eva.clients: %s", failure.message)
                    raise failure

            # Сюда попадаем только с повторяемой ошибкой.
            if attempts >= total_attempts:
                log.error("eva.clients: %s", failure.message)
                raise failure
            delay = self._delay(attempt, retry_after)
            log.warning(
                "eva.clients: %s — повтор %d/%d через %.2f с",
                failure.message,
                attempts,
                total_attempts - 1,
                delay,
            )
            sleep(delay)

        # Цикл всегда либо возвращает результат, либо бросает; это страховка.
        raise self._fail(
            "исчерпаны попытки", attempts=total_attempts, retryable=True
        )

    def stream_chat(self, messages: Any, max_tokens: int = 512):
        """Поток кусков ответа через SSE (``stream: true``).

        Возвращает итератор :class:`StreamChunk`: куски с непустым ``text`` по
        мере генерации, затем терминальный кусок (``done=True``) с ``usage``.
        Если провайдер не прислал ``usage``, в терминальном куске ``{}`` —
        вызывающий сам решает, как оценить токены.

        Ошибки аргументов (``max_tokens``, ``messages``) — сразу, как у
        :meth:`chat`; ошибки потока — исключениями :class:`ClientError` при
        итерации: таймаут, сеть, HTTP-статус, некорректный JSON события и
        обрыв потока без ``finish_reason``/``[DONE]``. Повторяется только
        ошибка до первого куска: повторять поток, который уже отдал текст,
        нельзя — клиент получил бы дубли. ``finally`` закрывает ответ
        провайдера, поэтому закрытие итератора (клиент ушёл) не оставляет
        висящих соединений.
        """
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            raise TypeError("max_tokens: нужно целое число")
        if max_tokens <= 0:
            raise ValueError("max_tokens: должен быть больше нуля")

        payload = {
            "model": self.model,
            "messages": _normalize_messages(messages),
            "max_tokens": max_tokens,
            "stream": True,
            # Просим usage отдельным событием: без него цену в потоке не
            # посчитать, а у некоторых провайдеров это единственный способ.
            "stream_options": {"include_usage": True},
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": USER_AGENT,
        }
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        return self._stream_chat(headers, body)

    def _stream_chat(self, headers: Mapping[str, str], body: bytes):
        """Итератор потока с повторами и закрытием; ошибки — ``ClientError``."""
        transport = self._stream_transport or self._transport or _open_stream
        sleep = self._sleep or _sleep
        total_attempts = self.retries + 1
        last_error: ClientError | None = None

        for attempt in range(total_attempts):
            attempts = attempt + 1
            state = {"text": 0}
            try:
                yield from self._stream_once(transport, headers, body, state, attempts)
                return
            except ClientError as exc:
                # Дубли текста недопустимы: после первого куска не повторяем.
                if state["text"] or not exc.retryable or attempts >= total_attempts:
                    raise
                last_error = exc
                delay = self._delay(attempt, exc.retry_after)
                log.warning(
                    "eva.clients: %s — повтор потока %d/%d через %.2f с",
                    exc.message,
                    attempts,
                    total_attempts - 1,
                    delay,
                )
                sleep(delay)

        if last_error is not None:
            raise last_error
        raise self._fail("исчерпаны попытки", attempts=total_attempts, retryable=True)

    def __repr__(self) -> str:
        return (
            f"OpenAIClient(model={self.model!r}, base={self.base!r}, "
            f"timeout={self.timeout:g}, retries={self.retries})"
        )

    # -- внутреннее --------------------------------------------------------- #

    def _fail(
        self,
        reason: str,
        *,
        kind: str,
        attempts: int,
        retryable: bool,
        status: int | None = None,
        detail: str | None = None,
        retry_after: float | None = None,
    ) -> ClientError:
        """Собрать `ClientError` с единым человеческим форматом."""
        total = self.retries + 1
        message = (
            f"{self._label()}: {reason} "
            f"(попыток: {attempts} из {total})"
        )
        message = self._redact(message)
        return ClientError(
            message,
            kind=kind,
            status=status,
            attempts=attempts,
            retryable=retryable,
            detail=self._redact(detail) if detail else None,
            retry_after=retry_after,
        )

    def _label(self) -> str:
        host = urllib.parse.urlparse(self.base).netloc or self.base
        return f"модель {self.model} @ {host}"

    def _redact(self, text: str) -> str:
        """Ключ не должен утечь в логи и текст исключения."""
        if self.key and self.key in text:
            return text.replace(self.key, "***")
        return text

    def _delay(self, attempt: int, retry_after: float | None = None) -> float:
        """Экспоненциальный backoff; для 429 приоритет у `Retry-After`."""
        if retry_after is not None:
            return min(max(0.0, retry_after), self.max_backoff)
        return min(self.backoff * (2 ** attempt), self.max_backoff)

    @staticmethod
    def _retry_after(response: HTTPResponse) -> float | None:
        """`Retry-After` числом секунд; HTTP-дату игнорируем — возьмём backoff."""
        header = getattr(response, "header", None)
        if not callable(header):
            return None
        raw = header("Retry-After")
        if not raw:
            return None
        try:
            value = float(str(raw).strip())
        except (TypeError, ValueError):
            return None
        if value != value or value < 0:  # NaN или отрицательное
            return None
        return value

    def _parse(self, response: HTTPResponse, *, attempts: int) -> ClientResult:
        body = response.body
        text_body = body.decode("utf-8", "replace") if isinstance(body, bytes) else str(body or "")
        status = response.status
        try:
            data = json.loads(text_body)
        except (json.JSONDecodeError, ValueError) as exc:
            detail = _error_detail(text_body)
            raise self._fail(
                f"ответ не JSON: {exc}",
                kind="response",
                status=status,
                attempts=attempts,
                retryable=False,
                detail=detail or None,
            ) from exc
        if not isinstance(data, dict):
            raise self._fail(
                "ответ не объект JSON",
                kind="response",
                status=status,
                attempts=attempts,
                retryable=False,
            )

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            detail = _error_detail(text_body)
            raise self._fail(
                "в ответе нет choices",
                kind="response",
                status=status,
                attempts=attempts,
                retryable=False,
                detail=detail or None,
            )

        first = choices[0] if isinstance(choices[0], Mapping) else {}
        text = self._extract_text(first)
        usage = self._extract_usage(data)
        return ClientResult(text=text, usage=usage, raw=data)

    @staticmethod
    def _extract_text(choice: Mapping[str, Any]) -> str:
        message = choice.get("message")
        content: Any = None
        if isinstance(message, Mapping):
            content = message.get("content")
        elif isinstance(choice.get("text"), str):  # старый /completions
            content = choice.get("text")
        if isinstance(content, list):  # content-части нового формата
            parts = []
            for part in content:
                if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
            content = "".join(parts)
        if not isinstance(content, str):
            content = ""
        return content.strip()

    def _stream_once(
        self,
        transport: Any,
        headers: Mapping[str, str],
        body: bytes,
        state: dict[str, int],
        attempts: int,
    ):
        """Одна попытка потока: открыть, разобрать SSE, закрыть в ``finally``."""
        try:
            response = transport(self.endpoint, headers, body, self.timeout)
        except (TimeoutError, socket.timeout) as exc:
            raise self._fail(
                f"таймаут {self.timeout:g} с",
                kind="timeout",
                attempts=attempts,
                retryable=True,
                detail=_collapse(str(exc)) or None,
            ) from exc
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, TimeoutError):
                text, kind = f"таймаут {self.timeout:g} с", "timeout"
            elif isinstance(reason, socket.gaierror):
                text, kind = f"DNS не разрешил хост: {reason}", "network"
            else:
                text, kind = f"сеть недоступна: {reason}", "network"
            raise self._fail(
                text, kind=kind, attempts=attempts, retryable=True
            ) from exc
        except OSError as exc:
            raise self._fail(
                f"ошибка сокета: {exc}",
                kind="network",
                attempts=attempts,
                retryable=True,
            ) from exc

        stream_body: Any = None
        try:
            status = int(getattr(response, "status", 0) or 0)
            # Подмена транспорта может вернуть уже прочитанное тело (HTTPResponse
            # из chat()) — тогда разбираем его как обычный SSE-буфер.
            stream_body = getattr(response, "body", None)
            if isinstance(stream_body, (bytes, bytearray)):
                stream_body = io.BytesIO(bytes(stream_body))
            elif isinstance(stream_body, str):
                stream_body = io.StringIO(stream_body)
            elif stream_body is None and hasattr(response, "readline"):
                stream_body = response
            if not (200 <= status < 300):
                detail = self._stream_error_detail(stream_body)
                reason = http.client.responses.get(status, "")
                head = f"HTTP {status}" + (f" {reason}" if reason else "")
                message = f"{head}: {detail}" if detail else head
                retryable = status == 429 or 500 <= status < 600
                raise self._fail(
                    message,
                    kind="http",
                    status=status,
                    attempts=attempts,
                    retryable=retryable,
                    detail=detail or None,
                    retry_after=self._retry_after(response),
                )

            usage: dict[str, Any] = {}
            usage_seen = False
            finish_reason: str | None = None
            saw_done = False
            saw_event = False

            try:
                for raw_line in _iter_lines(stream_body):
                    line = _decode_line(raw_line)
                    if not line:
                        continue
                    stripped = line.strip()
                    if not stripped or stripped.startswith(":"):
                        continue
                    field, _, value = stripped.partition(":")
                    if field != "data":
                        # id/event/retry SSE не нужны для chat.completions.
                        continue
                    value = value[1:] if value.startswith(" ") else value
                    if value.strip() == "[DONE]":
                        saw_done = True
                        break
                    if not value:
                        continue
                    saw_event = True
                    data = self._parse_stream_event(value, status, attempts)
                    if "usage" in data or "cost_rub" in data:
                        usage = self._extract_usage(data)
                        usage_seen = True

                    choices = data.get("choices")
                    if not isinstance(choices, list) or not choices:
                        continue
                    first = choices[0] if isinstance(choices[0], Mapping) else {}
                    piece_finish = first.get("finish_reason")
                    if piece_finish is not None:
                        finish_reason = str(piece_finish)
                    piece_text = _delta_text(first)
                    if piece_text:
                        state["text"] += 1
                        yield StreamChunk(
                            text=piece_text,
                            raw=data,
                            finish_reason=piece_finish
                            if isinstance(piece_finish, str)
                            else None,
                        )
            except (TimeoutError, socket.timeout) as exc:
                raise self._fail(
                    f"таймаут {self.timeout:g} с при чтении потока",
                    kind="timeout",
                    status=status,
                    attempts=attempts,
                    retryable=True,
                    detail=_collapse(str(exc)) or None,
                ) from exc
            except (ConnectionResetError, BrokenPipeError, http.client.IncompleteRead) as exc:
                raise self._fail(
                    f"соединение оборвалось во время потока: {exc}",
                    kind="stream",
                    status=status,
                    attempts=attempts,
                    retryable=True,
                ) from exc
            except OSError as exc:
                raise self._fail(
                    f"ошибка сокета во время потока: {exc}",
                    kind="network",
                    status=status,
                    attempts=attempts,
                    retryable=True,
                ) from exc

            if not saw_done and finish_reason is None:
                if not saw_event:
                    raise self._fail(
                        "поток пуст: провайдер не прислал ни одного события",
                        kind="stream",
                        status=status,
                        attempts=attempts,
                        retryable=True,
                    )
                raise self._fail(
                    "поток оборвался: соединение закрыто без finish_reason и [DONE]",
                    kind="stream",
                    status=status,
                    attempts=attempts,
                    retryable=True,
                )

            yield StreamChunk(
                text="",
                usage=usage if usage_seen else {},
                finish_reason=finish_reason,
                done=True,
            )
        finally:
            _close_quietly(stream_body)
            if response is not stream_body:
                _close_quietly(response)

    def _parse_stream_event(
        self, value: str, status: int, attempts: int
    ) -> dict[str, Any]:
        """JSON одного SSE-события; мусор и ошибка провайдера — ``ClientError``."""
        try:
            data = json.loads(value)
        except (json.JSONDecodeError, ValueError) as exc:
            raise self._fail(
                f"поток: некорректный JSON события: {exc}",
                kind="response",
                status=status,
                attempts=attempts,
                retryable=False,
                detail=_collapse(value),
            ) from exc
        if not isinstance(data, dict):
            raise self._fail(
                "поток: событие не объект JSON",
                kind="response",
                status=status,
                attempts=attempts,
                retryable=False,
                detail=_collapse(value),
            )
        if data.get("error") is not None:
            detail = _error_detail(value)
            raise self._fail(
                f"провайдер вернул ошибку в потоке: {detail}" if detail else
                "провайдер вернул ошибку в потоке",
                kind="response",
                status=status,
                attempts=attempts,
                retryable=False,
                detail=detail or None,
            )
        return data

    @staticmethod
    def _stream_error_detail(body: Any) -> str:
        """Выжимка тела ошибочного потокового ответа (тело уже ограничено)."""
        if body is None:
            return ""
        try:
            if isinstance(body, (bytes, bytearray)):
                payload: Any = bytes(body)
            elif hasattr(body, "read"):
                payload = body.read(MAX_ERROR_BODY * 8)
            else:
                payload = body
        except Exception:  # noqa: BLE001 - тело ошибки не критично
            return ""
        if isinstance(payload, bytes):
            return _error_detail(payload)
        return _error_detail(str(payload or ""))

    def _extract_usage(self, data: Mapping[str, Any]) -> dict[str, Any]:
        source = data.get("usage")
        usage_in: Mapping[str, Any] = source if isinstance(source, Mapping) else {}
        prompt = _as_int(usage_in.get("prompt_tokens"))
        completion = _as_int(usage_in.get("completion_tokens"))
        total = _as_int(usage_in.get("total_tokens")) or prompt + completion

        cost = _as_float(usage_in.get("cost_rub"))
        if cost is None:
            cost = _as_float(data.get("cost_rub"))
        if cost is None and (self.price_prompt_rub or self.price_completion_rub):
            cost = (
                prompt / 1000.0 * self.price_prompt_rub
                + completion / 1000.0 * self.price_completion_rub
            )
        if cost is None:
            cost = 0.0
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
            "cost_rub": round(cost, 6),
        }
