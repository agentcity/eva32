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

Устройство:

* единственный сетевой вызов — `urllib.request.urlopen` внутри `_post_json`;
  тесты подменяют `_post_json` (транспорт) или `_sleep` (пауза) — реальных
  запросов и реальных задержек в тестах нет;
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


# --------------------------------------------------------------------------- #
# Ошибки и результат
# --------------------------------------------------------------------------- #

class ClientError(RuntimeError):
    """Модель не ответила — понятным образом.

    Поля для вызывающего кода (роутер/политика):

    * `kind` — `"timeout"`, `"network"`, `"http"`, `"response"` (200, но
      разобрать тело не удалось) или `"config"`;
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
    ) -> None:
        super().__init__(message)
        self.message = message
        self.kind = kind
        self.status = status
        self.attempts = attempts
        self.retryable = retryable
        self.detail = detail

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
    и `_sleep` модуля).

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
        raw = response.header("Retry-After")
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
