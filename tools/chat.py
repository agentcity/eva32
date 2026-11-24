#!/usr/bin/env python3
"""Чат-REPL для проверки моделей: локальный ``.vtm``, ollama, OpenAI-совместимый API.

Зачем: до этой задачи каждую модель приходилось звать по-своему — ``ctypes``
для своей ВТМ, ``curl`` для ollama, клиент ``eva.clients`` для облака. Один
интерфейс нужен, чтобы сравнивать модели глазами: одинаковый системный
промпт, одинаковая история, одинаковые метрики (TTFT, токенов/с, символы,
стоимость).

Запуск::

    python3 tools/chat.py --backend ollama --model qwen2.5:3b
    python3 tools/chat.py --backend vtm --model runs/tiny/model.vtm \\
        --tokenizer tests/vtm/fixtures/tokenizer.json
    python3 tools/chat.py --backend openai --model gpt-4o-mini \\
        --base https://api.example.com/v1 --system prompts/system.assistant.md

    python3 tools/chat.py --backend ollama --model qwen2.5:3b --once "привет"

Устройство:

* **История** — только ``user``/``assistant``; системный промпт хранится
  отдельно и подставляется первым сообщением, поэтому он не теряется ни при
  ``/reset``, ни при обрезке.
* **Обрезка** — перед каждым запросом история укладывается в
  ``--ctx - --max-tokens - запас``: выбрасываются самые старые сообщения
  целиком (без молчаливого урезания текста), последнее сообщение
  пользователя остаётся всегда. Если после обрезки первым стоит ответ
  ассистента (вопрос выброшен), он тоже выбрасывается.
* **Стриминг** — у всех трёх бэкендов по токенам/кускам:
  ВТМ зовёт ``vtm_api_generate`` через ``VtmBackend.stream_ids`` (C-колбэк,
  декодирование чанками); ollama — свой ``POST /api/chat`` с NDJSON
  (построчный JSON, не SSE — так устроен ollama, но парсер понимает и
  ``data:``-строки); OpenAI-совместимый API — SSE через ``eva.clients``.
* **Метрики** — TTFT считается от вызова бэкенда до первого непустого
  куска; токенов/с — по completion-токенам за время после TTFT. Для ВТМ
  токены точные (токенизатор), для HTTP — из ``usage`` провайдера, а если
  провайдер их не прислал, оценка по символам.
* **Стоимость** — берётся из ответа провайдера (``usage.cost_rub``), иначе
  из тарифов ``EVA_BIG_PRICE_PROMPT/COMPLETION`` или ``--price-*``; если
  тариф не задан, честно печатается «— (тариф не задан)». Локальные
  (``vtm``/``ollama``) — 0 ₽.

Команды REPL: ``/reset``, ``/system [текст|-]``, ``/stats``, ``/help``,
``/quit``. ``/system`` без аргумента показывает текущий промпт, ``-``
очищает, ``@файл`` читает промпт из файла.

Тесты (без сети и без GPU): ``python3 -m unittest tests.test_chat_cli``.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# `python3 tools/chat.py` кладёт в sys.path каталог tools, а не корень
# репозитория; eva импортируется лениво, но корень нужен уже для путей.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

__all__ = [
    "ChatBackendError",
    "ChatConfigError",
    "ChatRepl",
    "ChatSession",
    "OllamaChatBackend",
    "OpenAIChatBackend",
    "StreamEvent",
    "TokenCounter",
    "TrimResult",
    "TurnStats",
    "VtmChatBackend",
    "build_backend",
    "build_parser",
    "estimate_tokens",
    "format_stats",
    "format_turn_brief",
    "main",
    "parse_stream_payloads",
    "trim_history",
]

# --------------------------------------------------------------------------- #
# Константы
# --------------------------------------------------------------------------- #

DEFAULT_BACKENDS = ("vtm", "ollama", "openai")
DEFAULT_MAX_TOKENS = 256
DEFAULT_TIMEOUT = 60.0
DEFAULT_OLLAMA_BASE = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:3b"  # Makefile: DEV_MODEL
DEFAULT_CONTEXT = {"vtm": 512, "ollama": 4096, "openai": 8192}

#: Запас на чат-шаблон (роли, разделители) — токенов на запрос.
PROMPT_OVERHEAD = 8
#: Запас на обёртку одного сообщения (роль, перевод строки) — токенов.
MESSAGE_OVERHEAD = 4

MAX_ERROR_BODY = 600
DEFAULT_SYSTEM = ""

_BACKEND_NAMES = {
    "vtm": "локальная ВТМ (ctypes, без сети)",
    "ollama": "ollama /api/chat (NDJSON)",
    "openai": "OpenAI-совместимый /v1/chat/completions (SSE)",
}


# --------------------------------------------------------------------------- #
# Ошибки
# --------------------------------------------------------------------------- #

class ChatConfigError(RuntimeError):
    """CLI настроен так, что бэкенд не собрать (нет модели/base/ключа)."""


class ChatBackendError(RuntimeError):
    """Бэкенд не ответил: сеть, HTTP-статус, поток, генерация."""


# --------------------------------------------------------------------------- #
# Токены: точный счётчик или оценка
# --------------------------------------------------------------------------- #

def estimate_tokens(text: str) -> int:
    """Грубая оценка токенов по символам (для HTTP-бэкендов без токенизатора).

    Средний BPE-токен — 3–4 символа латиницы и 2–3 символа кириллицы;
    берём 3 символа на токен: для обрезки истории этого достаточно, а
    точные числа для ``/stats`` приходят из ``usage`` провайдера.
    """
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 3))


@dataclass(frozen=True)
class TokenCounter:
    """Счётчик токенов: ``exact=True`` — токенизатор модели, иначе оценка."""

    exact: bool = False
    counter: Callable[[str], int] = estimate_tokens

    def __call__(self, text: str) -> int:
        return int(self.counter(text or ""))

    def describe(self) -> str:
        return "точно (токенизатор)" if self.exact else "оценка по символам"


def _message_tokens(message: Mapping[str, str], counter: TokenCounter) -> int:
    return MESSAGE_OVERHEAD + counter(str(message.get("content") or ""))


# --------------------------------------------------------------------------- #
# История и её обрезка
# --------------------------------------------------------------------------- #

Message = dict[str, str]


@dataclass(frozen=True)
class TrimResult:
    """Что уйдёт в модель после обрезки истории."""

    messages: list[Message]
    dropped: int
    budget_tokens: int
    used_tokens: int

    @property
    def trimmed(self) -> bool:
        return self.dropped > 0


def trim_history(
    history: Sequence[Mapping[str, str]],
    *,
    system: str,
    ctx: int,
    max_tokens: int,
    counter: TokenCounter = TokenCounter(),
) -> TrimResult:
    """Уложить историю в контекст, сохранив системный промпт и вопрос.

    Бюджет: ``ctx - max_tokens - PROMPT_OVERHEAD - системный промпт``.
    Идём от конца к началу и набираем сообщения, пока они влезают; старые
    выбрасываются целиком. Последнее сообщение (текущий вопрос) остаётся
    даже если одно больше бюджета — иначе спрашивать будет нечего; в этом
    случае бэкенд сам скажет, что контекст мал.
    """
    system_tokens = 0
    system_message: list[Message] = []
    if system:
        system_tokens = MESSAGE_OVERHEAD + counter(system)
        system_message = [{"role": "system", "content": system}]

    budget = max(0, ctx - max_tokens - PROMPT_OVERHEAD - system_tokens)
    if not history:
        return TrimResult(
            messages=system_message,
            dropped=0,
            budget_tokens=budget,
            used_tokens=system_tokens,
        )

    kept: list[Message] = []
    used = 0
    for message in reversed(history):
        cost = _message_tokens(message, counter)
        if kept and used + cost > budget:
            break
        kept.append(
            {"role": str(message.get("role") or "user"),
             "content": str(message.get("content") or "")}
        )
        used += cost
    kept.reverse()

    # Первый оставшийся — ответ ассистента: его вопрос выброшен обрезкой.
    # Хвост диалога должен начинаться с пользователя (чат-шаблоны и API это
    # ожидают), поэтому такой ответ тоже уходит.
    while len(kept) > 1 and kept[0]["role"] != "user":
        kept.pop(0)

    used = sum(_message_tokens(m, counter) for m in kept)
    messages = system_message + kept
    if system_message:
        used += system_tokens
    return TrimResult(
        messages=messages,
        dropped=len(history) - len(kept),
        budget_tokens=budget,
        used_tokens=used,
    )


# --------------------------------------------------------------------------- #
# Метрики хода
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TurnStats:
    """Метрики одного ответа модели."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    chars: int = 0
    ttft_s: float = 0.0
    elapsed_s: float = 0.0
    cost_rub: float = 0.0
    cost_known: bool = False
    finish_reason: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def ttft_ms(self) -> float:
        return self.ttft_s * 1000.0

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed_s * 1000.0

    @property
    def tps(self) -> float:
        """Completion-токены в секунду после первого токена (TTFT)."""
        if self.completion_tokens <= 0:
            return 0.0
        decode = self.elapsed_s - self.ttft_s
        if decode <= 0 or self.completion_tokens == 1:
            return self.completion_tokens / self.elapsed_s if self.elapsed_s > 0 else 0.0
        return self.completion_tokens / decode

    @property
    def chars_per_token(self) -> float:
        if self.completion_tokens <= 0:
            return 0.0
        return self.chars / self.completion_tokens


@dataclass
class ChatSession:
    """Диалог: системный промпт, история user/assistant, метрики ходов.

    Системный промпт живёт отдельно от ``history`` — поэтому ``/reset`` и
    обрезка его не касаются.
    """

    system: str = DEFAULT_SYSTEM
    ctx: int = 4096
    max_tokens: int = DEFAULT_MAX_TOKENS
    counter: TokenCounter = TokenCounter()
    history: list[Message] = field(default_factory=list)
    turns: list[TurnStats] = field(default_factory=list)
    last_trim: TrimResult | None = None

    def __post_init__(self) -> None:
        if self.ctx <= 0:
            raise ValueError(f"ctx: ожидалось > 0, получено {self.ctx}")
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens: ожидалось > 0, получено {self.max_tokens}")

    # -- история ------------------------------------------------------------

    @property
    def last_turn(self) -> TurnStats | None:
        return self.turns[-1] if self.turns else None

    def add_user(self, text: str) -> None:
        self.history.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        self.history.append({"role": "assistant", "content": text})

    def drop_last(self, role: str | None = None) -> None:
        """Убрать последнее сообщение (откат неудачного хода)."""
        if not self.history:
            return
        if role is not None and self.history[-1].get("role") != role:
            return
        self.history.pop()

    def reset(self) -> None:
        """Очистить диалог и метрики, сохранив системный промпт."""
        self.history.clear()
        self.turns.clear()
        self.last_trim = None

    def set_system(self, text: str) -> None:
        self.system = text

    # -- сборка запроса -----------------------------------------------------

    def build_messages(self) -> tuple[list[Message], TrimResult]:
        result = trim_history(
            self.history,
            system=self.system,
            ctx=self.ctx,
            max_tokens=self.max_tokens,
            counter=self.counter,
        )
        self.last_trim = result
        return result.messages, result

    # -- метрики ------------------------------------------------------------

    def finish_turn(
        self,
        answer: str,
        messages: Sequence[Mapping[str, str]],
        usage: Mapping[str, Any],
        *,
        ttft_s: float,
        elapsed_s: float,
        finish_reason: str | None = None,
    ) -> TurnStats:
        """Собрать :class:`TurnStats`, подставив оценки недостающих чисел."""
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        if prompt_tokens <= 0:
            prompt_tokens = sum(
                _message_tokens(message, self.counter) for message in messages
            )
        completion_tokens = int(usage.get("completion_tokens") or 0)
        if completion_tokens <= 0 and answer:
            completion_tokens = self.counter(answer)
        stats = TurnStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            chars=len(answer),
            ttft_s=max(0.0, float(ttft_s)),
            elapsed_s=max(0.0, float(elapsed_s)),
            cost_rub=float(usage.get("cost_rub") or 0.0),
            cost_known=bool(usage.get("cost_known")),
            finish_reason=finish_reason,
        )
        self.turns.append(stats)
        return stats

    # -- сводка -------------------------------------------------------------

    def totals(self) -> dict[str, Any]:
        turns = len(self.turns)
        return {
            "turns": turns,
            "prompt_tokens": sum(t.prompt_tokens for t in self.turns),
            "completion_tokens": sum(t.completion_tokens for t in self.turns),
            "total_tokens": sum(t.total_tokens for t in self.turns),
            "chars": sum(t.chars for t in self.turns),
            "cost_rub": sum(t.cost_rub for t in self.turns),
            "cost_known": all(t.cost_known for t in self.turns) if turns else True,
            "avg_ttft_ms": (
                sum(t.ttft_ms for t in self.turns) / turns if turns else 0.0
            ),
        }


def _cost_text(stats: TurnStats) -> str:
    if not stats.cost_known:
        return "— (тариф не задан)"
    return f"{stats.cost_rub:.4f} ₽"


def format_turn_brief(stats: TurnStats) -> str:
    """Одна строка после ответа (``--stats``)."""
    return (
        f"[{stats.completion_tokens} ток · {stats.tps:.1f} ток/с · "
        f"TTFT {stats.ttft_ms:.0f} мс · {stats.chars} симв · {_cost_text(stats)}]"
    )


def format_stats(
    session: ChatSession,
    *,
    backend_name: str = "",
    model: str = "",
) -> str:
    """Текст команды ``/stats``: последний ход, сессия, контекст."""
    lines = [f"статистика · {backend_name}/{model}" if backend_name or model else "статистика"]
    last = session.last_turn
    if last is None:
        lines.append("  ходов: 0")
    else:
        lines.extend(
            [
                "  последний ход:",
                f"    TTFT:        {last.ttft_ms:.0f} мс",
                f"    токены:      {last.prompt_tokens} prompt + "
                f"{last.completion_tokens} completion = {last.total_tokens}",
                f"    ток/с:       {last.tps:.1f}",
                f"    символов:    {last.chars} "
                f"({last.chars_per_token:.1f} симв/ток)",
                f"    время:       {last.elapsed_s:.2f} с",
                f"    стоп:        {last.finish_reason or '—'}",
                f"    стоимость:   {_cost_text(last)}",
            ]
        )

    totals = session.totals()
    if totals["turns"]:
        lines.append(
            f"  всего: {totals['turns']} ход(ов), {totals['total_tokens']} токенов, "
            f"{totals['chars']} символов, TTFT сред. {totals['avg_ttft_ms']:.0f} мс"
        )
        total_cost = (
            f"{totals['cost_rub']:.4f} ₽" if totals["cost_known"] else "— (тариф не задан)"
        )
        lines.append(f"  стоимость за сессию: {total_cost}")

    trim = session.last_trim
    dropped = trim.dropped if trim is not None else 0
    used = last.prompt_tokens if last is not None else 0
    lines.append(
        f"  контекст: {used}/{session.ctx} токенов, "
        f"история {len(session.history)} сообщений, обрезано {dropped} в последнем ходе"
    )
    lines.append(f"  счётчик истории: {session.counter.describe()}")
    if session.system:
        lines.append(f"  системный промпт: {len(session.system)} символов")
    else:
        lines.append("  системный промпт: нет")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Транспорт: HTTP-поток и разбор строк
# --------------------------------------------------------------------------- #

@dataclass
class StreamResponse:
    """Ответ транспорта на потоковый запрос.

    ``body`` — файлоподобный объект с ``readline()`` (у ``urlopen`` это сам
    ответ); для HTTP-ошибки тело уже прочитано в ``BytesIO``.
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
        close = getattr(body, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001 — закрытие не должно ронять поток
                pass


def open_stream(
    url: str,
    headers: Mapping[str, str],
    payload: bytes,
    timeout: float,
) -> StreamResponse:
    """POST и вернуть поток, не читая тело целиком (точка подмены в тестах).

    HTTP 4xx/5xx не бросаются: тело ограниченно читается и отдаётся вместе
    со статусом, чтобы бэкенд сам собрал человеческую ошибку.
    """
    request = urllib.request.Request(
        url, data=payload, headers=dict(headers), method="POST"
    )
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read(MAX_ERROR_BODY * 8)
        except Exception:  # noqa: BLE001 — тело ошибки не критично
            body = b""
        response_headers = dict(exc.headers.items()) if exc.headers else {}
        try:
            exc.close()
        except Exception:  # noqa: BLE001
            pass
        return StreamResponse(
            status=exc.code, body=io.BytesIO(body or b""), headers=response_headers
        )
    return StreamResponse(
        status=int(getattr(response, "status", 200) or 200),
        body=response,
        headers=dict(response.headers.items()),
    )


def _decode_line(raw: Any) -> str:
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw).decode("utf-8", "replace").rstrip("\r\n")
    return str(raw).rstrip("\r\n")


def _collapse(text: str, limit: int = MAX_ERROR_BODY) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[:limit].rstrip() + "…"


def parse_stream_payloads(
    body: Any,
    *,
    where: str = "поток",
) -> Iterator[dict[str, Any]]:
    """Строки потока -> JSON-события; понимает SSE (``data:``) и NDJSON.

    ``data: [DONE]`` завершает поток. Пустые строки, комментарии SSE
    (``: keep-alive``) и прочие поля (``event:``/``id:``) пропускаются.
    Некорректный JSON — :class:`ChatBackendError` с куском строки: молча
    терять события нельзя, иначе ответ окажется обрезанным без причины.
    """
    if body is None:
        return
    while True:
        raw = body.readline()
        if not raw:
            break
        line = _decode_line(raw).strip()
        if not line or line.startswith(":"):
            continue
        if line.startswith("data:"):
            value = line[len("data:"):].strip()
            if value == "[DONE]":
                return
            if not value:
                continue
            line = value
        elif line.startswith(("event:", "id:", "retry:")):
            continue
        try:
            data = json.loads(line)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ChatBackendError(
                f"{where}: некорректный JSON события: {exc}; "
                f"строка: {_collapse(line, 160)}"
            ) from exc
        if not isinstance(data, dict):
            raise ChatBackendError(
                f"{where}: событие не объект JSON: {_collapse(line, 160)}"
            )
        error = data.get("error")
        if error:
            detail = ""
            if isinstance(error, Mapping):
                detail = str(error.get("message") or error.get("type") or error)
            else:
                detail = str(error)
            raise ChatBackendError(f"{where}: провайдер вернул ошибку: {_collapse(detail)}")
        yield data


def _read_error_body(body: Any) -> str:
    """Выжимка тела ошибочного ответа (не больше пары сотен байт)."""
    if body is None:
        return ""
    try:
        payload = body.read(MAX_ERROR_BODY * 8) if hasattr(body, "read") else body
    except Exception:  # noqa: BLE001 — тело ошибки не критично
        return ""
    if isinstance(payload, (bytes, bytearray)):
        text = bytes(payload).decode("utf-8", "replace")
    else:
        text = str(payload or "")
    text = text.strip()
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        parsed = None
    if isinstance(parsed, Mapping):
        error = parsed.get("error")
        if isinstance(error, Mapping) and error.get("message"):
            text = str(error["message"])
        elif isinstance(error, str):
            text = error
        elif parsed.get("message"):
            text = str(parsed["message"])
    return _collapse(text)


# --------------------------------------------------------------------------- #
# События стриминга и бэкенды
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class StreamEvent:
    """Кусок ответа: текст или терминальное событие с ``usage``."""

    text: str = ""
    done: bool = False
    finish_reason: str | None = None
    usage: Mapping[str, Any] = field(default_factory=dict)


class ChatBackend:
    """Общий вид бэкенда (duck typing; протокол без наследования).

    * ``name``/``model`` — для баннера и ``/stats``;
    * ``counter`` — :class:`TokenCounter` (точный или оценка);
    * ``stream(messages, max_tokens=...)`` — итератор :class:`StreamEvent`;
    * ``close()`` — освободить ресурсы (у ВТМ — C-контекст).
    """

    name = "?"
    model = "?"
    counter: TokenCounter = TokenCounter()

    def stream(self, messages: Sequence[Mapping[str, str]], *, max_tokens: int) -> Iterator[StreamEvent]:
        raise NotImplementedError

    def close(self) -> None:
        return None


class VtmChatBackend(ChatBackend):
    """Локальная ВТМ: ``vtm_api_generate`` через ``ctypes``, стриминг по токенам.

    Токены считаются точно: промпт кодируется токенизатором модели, ответ
    идёт через ``VtmBackend.stream_ids`` (C-колбэк на каждый токен) и
    декодируется чанками. EOS в completion-токены не входит.
    """

    name = "vtm"

    def __init__(
        self,
        *,
        model: str,
        tokenizer: str | None = None,
        library: str | None = None,
        backend: str = "auto",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        cache_prefix: bool = True,
        environ: Mapping[str, str] | None = None,
        inner: Any = None,
    ) -> None:
        self.model = str(model)
        self.tokenizer_path = tokenizer
        self.cache_prefix = bool(cache_prefix)
        self._inner = inner
        self._counter: TokenCounter | None = None
        if inner is None:
            from eva.backends import VtmBackend  # лениво: без libvtm импорт безопасен

            self._inner = VtmBackend(
                model=self.model,
                tokenizer=tokenizer,
                library=library,
                backend=backend,
                max_tokens=max_tokens,
                cache_prefix=cache_prefix,
                environ=environ,
                prompt_builder=None,
            )
        # точный счётчик включает токенизатор; ошибка токенизатора — на первом
        # же запросе, текст ошибки покажет REPL.
        self._counter = TokenCounter(exact=True, counter=self._count_exact)

    def _count_exact(self, text: str) -> int:
        if not text:
            return 0
        return len(self._inner.encode(text))

    @property
    def counter(self) -> TokenCounter:  # type: ignore[override]
        return self._counter or TokenCounter()

    @property
    def prompt_builder(self) -> Callable[[Any], str]:
        return self._inner.prompt_builder

    def stream(
        self, messages: Sequence[Mapping[str, str]], *, max_tokens: int
    ) -> Iterator[StreamEvent]:
        inner = self._inner
        try:
            prompt = inner.prompt_builder(list(messages))
            prompt_ids = inner.encode(prompt)
            tokenizer = inner.tokenizer
        except Exception as exc:  # noqa: BLE001 — VtmError/TokenizerError -> человеческий текст
            raise ChatBackendError(self._hint(str(exc))) from exc
        if not prompt_ids:
            raise ChatBackendError(
                "vtm: промпт пуст после токенизации; проверьте --system/--tokenizer"
            )
        decoder = tokenizer.stream_decoder()
        eos_id = int(getattr(tokenizer, "eos_id", -1))
        completion = 0
        finish = "stop"
        eos_seen = False
        # Страж зацикливания: слабо квантованная модель иногда не выдаёт EOS и
        # повторяет одну и ту же фразу до лимита. Ловим трёхкратный повтор
        # последних 40 символов и останавливаемся — лучше короткий ответ, чем
        # полотно с эхом.
        guard_text = ""
        #: Сколько символов держим в запасе, чтобы не отдать начало маркера.
        pending = ""
        try:
            for token in inner.stream_ids(prompt_ids, max_new_tokens=max_tokens):
                if eos_id >= 0 and int(token) == eos_id:
                    finish = "stop"
                    eos_seen = True
                    break
                completion += 1
                chunk = decoder.push(int(token))
                if chunk:
                    # Держим «хвост» неотданным: маркер конца («user:»,
                    # <|im_end|>) может прийти в следующем куске, и тогда этот
                    # хвост нужно не отдать, а вырезать.
                    pending += chunk
                    stopped_by_marker = _hit_stop_marker(pending)
                    guard_text = (guard_text + chunk)[-400:]
                    if stopped_by_marker:
                        clean = _strip_stop_markers(pending)
                        if clean:
                            yield StreamEvent(text=clean)
                        pending = ""
                        finish = "stop"
                        eos_seen = True
                        break
                    if _looks_like_loop(guard_text):
                        if pending:
                            yield StreamEvent(text=pending)
                        pending = ""
                        finish = "repeat"
                        eos_seen = True
                        break
                    if len(pending) > _MARKER_LOOKAHEAD:
                        cut = len(pending) - _MARKER_LOOKAHEAD
                        yield StreamEvent(text=pending[:cut])
                        pending = pending[cut:]
            if pending:
                yield StreamEvent(text=pending)
                pending = ""
            tail = decoder.finish()
            if tail:
                yield StreamEvent(text=tail)
            if completion >= max_tokens and not eos_seen:
                finish = "length"
        except ChatBackendError:
            raise
        except Exception as exc:  # noqa: BLE001 — VtmError/VtmBusy/VtmCancelled
            raise ChatBackendError(self._hint(str(exc))) from exc
        yield StreamEvent(
            done=True,
            finish_reason=finish,
            usage={
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": completion,
                "total_tokens": len(prompt_ids) + completion,
                "cost_rub": 0.0,
                "cost_known": True,  # локально: 0 ₽
            },
        )

    def _hint(self, text: str) -> str:
        lowered = text.lower()
        if "токенизатор не задан" in lowered or "tokenizer" in lowered:
            return (
                f"vtm: {text}. Укажите --tokenizer tokenizer.json "
                "(или EVA_VTM_TOKENIZER)"
            )
        if "вне vocab" in lowered or "не сходятся размерности" in lowered:
            return (
                f"vtm: {text}. Похоже, это не текстовая модель: demo/model.vtm — "
                "демо-эмбеддинг (vocab=128). Для чата возьмите текстовую модель и "
                "её токенизатор, например --model runs/tiny/model.vtm "
                "--tokenizer tests/vtm/fixtures/tokenizer.json"
            )
        return f"vtm: {text}"

    def cancel(self) -> None:
        """Прервать текущую генерацию (Ctrl-C в REPL): C-флаг, безопасен."""
        cancel = getattr(self._inner, "cancel", None)
        if callable(cancel):
            try:
                cancel()
            except Exception:  # noqa: BLE001 — отмена не должна ронять REPL
                pass

    def close(self) -> None:
        close = getattr(self._inner, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001 — закрытие не должно ронять выход
                pass


class OllamaChatBackend(ChatBackend):
    """ollama: ``POST /api/chat`` c ``stream: true``.

    ollama отдаёт NDJSON (по строке JSON на событие), а не SSE; парсер
    :func:`parse_stream_payloads` понимает оба формата. Финальное событие
    ``done: true`` несёт ``prompt_eval_count``/``eval_count`` — токены
    берём оттуда.
    """

    name = "ollama"

    def __init__(
        self,
        *,
        base: str = DEFAULT_OLLAMA_BASE,
        model: str = DEFAULT_OLLAMA_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
        transport: Callable[[str, Mapping[str, str], bytes, float], StreamResponse] | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ChatConfigError("ollama: пустое имя модели (--model qwen2.5:3b)")
        self.base = _normalize_base(base, "ollama")
        self.model = model.strip()
        self.endpoint = self.base + "/api/chat"
        self.timeout = float(timeout)
        self._transport = transport or open_stream
        self._options = dict(options or {})

    def stream(
        self, messages: Sequence[Mapping[str, str]], *, max_tokens: int
    ) -> Iterator[StreamEvent]:
        options = dict(self._options)
        options["num_predict"] = int(max_tokens)
        payload = {
            "model": self.model,
            "messages": list(messages),
            "stream": True,
            "options": options,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/x-ndjson",
            "User-Agent": "eva32-chat/0.1 (+python-stdlib)",
        }
        response = self._open(body, headers)
        stream_body = getattr(response, "body", None)
        try:
            if not (200 <= int(getattr(response, "status", 0)) < 300):
                detail = _read_error_body(stream_body)
                status = int(getattr(response, "status", 0))
                hint = ""
                if status == 404:
                    hint = f"; модель загружена? ollama pull {self.model}"
                raise ChatBackendError(
                    f"ollama HTTP {status} @ {self.endpoint}: "
                    f"{detail or 'без тела ответа'}{hint}"
                )
            usage: dict[str, Any] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cost_rub": 0.0,
                "cost_known": True,  # локально: 0 ₽
            }
            for data in parse_stream_payloads(stream_body, where="ollama"):
                message = data.get("message")
                if isinstance(message, Mapping):
                    text = message.get("content")
                    if isinstance(text, str) and text:
                        yield StreamEvent(text=text)
                if data.get("done"):
                    usage["prompt_tokens"] = _as_int(data.get("prompt_eval_count"))
                    usage["completion_tokens"] = _as_int(data.get("eval_count"))
                    usage["total_tokens"] = (
                        usage["prompt_tokens"] + usage["completion_tokens"]
                    )
                    yield StreamEvent(
                        done=True,
                        finish_reason=str(data.get("done_reason") or "stop"),
                        usage=usage,
                    )
                    return
            raise ChatBackendError(
                "ollama: поток оборвался без done:true — сервер закрыл соединение"
            )
        finally:
            response.close()

    def _open(self, body: bytes, headers: Mapping[str, str]) -> StreamResponse:
        try:
            return self._transport(self.endpoint, headers, body, self.timeout)
        except ChatBackendError:
            raise
        except (TimeoutError, OSError) as exc:
            raise ChatBackendError(
                f"ollama не отвечает на {self.base}: {exc}. "
                "Поднимите сервер: make dev-serve (или ollama serve)"
            ) from exc


class OpenAIChatBackend(ChatBackend):
    """OpenAI-совместимый ``/v1/chat/completions`` потоком (SSE).

    Сеть, ретраи, разбор SSE и цена — в ``eva.clients.OpenAIClient`` (он же
    обслуживает малую и большую модели гибрида), здесь только адаптация
    ``StreamChunk`` -> :class:`StreamEvent`.
    """

    name = "openai"

    def __init__(
        self,
        *,
        base: str,
        model: str,
        key: str = "",
        timeout: float = DEFAULT_TIMEOUT,
        retries: int = 1,
        price_prompt_rub: float = 0.0,
        price_completion_rub: float = 0.0,
        stream_transport: Any = None,
    ) -> None:
        if not base:
            raise ChatConfigError("openai: не задан base (--base или EVA_BIG_BASE)")
        if not model:
            raise ChatConfigError("openai: не задана модель (--model или EVA_BIG_MODEL)")
        from eva.clients import OpenAIClient  # лениво: tools не тянет eva без нужды

        self.base = base.strip().rstrip("/")
        self.model = model.strip()
        self._prices = (float(price_prompt_rub), float(price_completion_rub))
        self._client = OpenAIClient(
            base=self.base,
            key=key or "",
            model=self.model,
            timeout=timeout,
            retries=max(0, int(retries)),
            price_prompt_rub=float(price_prompt_rub),
            price_completion_rub=float(price_completion_rub),
            stream_transport=stream_transport,
        )

    def stream(
        self, messages: Sequence[Mapping[str, str]], *, max_tokens: int
    ) -> Iterator[StreamEvent]:
        try:
            chunks = self._client.stream_chat(list(messages), max_tokens=int(max_tokens))
            for chunk in chunks:
                if chunk.done:
                    usage = dict(chunk.usage or {})
                    usage.setdefault("cost_rub", 0.0)
                    usage["cost_known"] = bool(
                        self._prices[0] or self._prices[1]
                        or float(usage.get("cost_rub") or 0.0) > 0
                    )
                    yield StreamEvent(
                        done=True,
                        finish_reason=chunk.finish_reason,
                        usage=usage,
                    )
                    return
                if chunk.text:
                    yield StreamEvent(text=chunk.text)
        except Exception as exc:  # noqa: BLE001 — ClientError и сеть -> ChatBackendError
            if isinstance(exc, ChatBackendError):
                raise
            raise ChatBackendError(f"openai: {exc}") from exc


def _as_int(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        return max(0, int(float(value)))
    except (TypeError, ValueError, OverflowError):
        return 0


def _normalize_base(base: str, name: str) -> str:
    text = (base or "").strip().rstrip("/")
    if not text:
        raise ChatConfigError(f"{name}: пустой base")
    if "://" not in text:
        text = "http://" + text
    parsed = urllib.parse.urlparse(text)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ChatConfigError(
            f"{name}: base={base!r}: ожидается http(s)://хост[:порт]"
        )
    return text


# --------------------------------------------------------------------------- #
# Сборка бэкенда из аргументов/окружения
# --------------------------------------------------------------------------- #

def _env(environ: Mapping[str, str] | None, name: str) -> str:
    source = environ if environ is not None else os.environ
    return str(source.get(name) or "").strip()


def _env_float(
    environ: Mapping[str, str] | None, name: str, default: float = 0.0
) -> float:
    raw = _env(environ, name)
    if not raw:
        return default
    try:
        return float(raw.replace(",", ".").replace("₽", "").strip())
    except ValueError as exc:
        raise ChatConfigError(
            f"{name}: ожидалось число рублей за 1000 токенов, получено {raw!r}"
        ) from exc


def _default_tokenizer(model_path: Path) -> str | None:
    """tokenizer.json рядом с моделью, иначе — фикстура для demo/model.vtm."""
    sibling = model_path.with_name("tokenizer.json")
    if sibling.is_file():
        return str(sibling)
    if model_path.name == "model.vtm" and model_path.parent.name == "demo":
        fixture = ROOT / "tests" / "vtm" / "fixtures" / "tokenizer.json"
        if fixture.is_file():
            return str(fixture)
    return None


def build_backend(
    *,
    backend: str,
    model: str | None = None,
    base: str | None = None,
    key: str | None = None,
    tokenizer: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    price_prompt: float | None = None,
    price_completion: float | None = None,
    cache_prefix: bool | None = None,
    environ: Mapping[str, str] | None = None,
    transport: Any = None,
    stream_transport: Any = None,
) -> ChatBackend:
    """Собрать бэкенд по флагам CLI и окружению (окружение — дефолты).

    Никакой сети здесь нет: HTTP-клиенты только конфигурируются, ВТМ
    грузится лениво при первом запросе.
    """
    name = (backend or "").strip().lower()
    if name not in DEFAULT_BACKENDS:
        raise ChatConfigError(
            f"--backend={backend!r}: ожидалось одно из {', '.join(DEFAULT_BACKENDS)}"
        )

    if name == "vtm":
        from eva.backends import default_model_path

        resolved_model = model or _env(environ, "EVA_VTM_MODEL")
        if not resolved_model:
            default = default_model_path(environ)
            if default is None:
                raise ChatConfigError(
                    "vtm: модель не найдена (EVA_VTM_MODEL/demo/model.vtm); "
                    "соберите демо или укажите --model"
                )
            resolved_model = str(default)
        model_path = Path(resolved_model).expanduser()
        if not model_path.is_file():
            raise ChatConfigError(f"vtm: файла модели нет: {model_path}")
        resolved_tokenizer = (
            tokenizer
            or _env(environ, "EVA_VTM_TOKENIZER")
            or _default_tokenizer(model_path)
        )
        if not resolved_tokenizer:
            raise ChatConfigError(
                f"vtm: рядом с {model_path.name} нет tokenizer.json; "
                "укажите --tokenizer (для текстового чата он обязателен)"
            )
        if not Path(resolved_tokenizer).expanduser().is_file():
            raise ChatConfigError(f"vtm: файла токенизатора нет: {resolved_tokenizer}")
        if cache_prefix is None:
            # По умолчанию кэш префикса включён (второй запрос ~1 с), но
            # EVA_VTM_CACHE=0/no/false его выключает — как в VTMConfig.
            cache_prefix = _env(environ, "EVA_VTM_CACHE").lower() not in (
                "0", "no", "false", "off",
            )
        return VtmChatBackend(
            model=str(model_path),
            tokenizer=str(Path(resolved_tokenizer).expanduser()),
            library=_env(environ, "EVA_VTM_LIB") or None,
            backend=_env(environ, "EVA_VTM_BACKEND") or None,
            max_tokens=max_tokens,
            cache_prefix=bool(cache_prefix),
            environ=environ,
            inner=None,
        )

    if name == "ollama":
        resolved_base = (
            base
            or _env(environ, "OLLAMA_BASE")
            or _env(environ, "OLLAMA_HOST")
            or DEFAULT_OLLAMA_BASE
        )
        resolved_model = model or _env(environ, "OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL
        return OllamaChatBackend(
            base=resolved_base,
            model=resolved_model,
            timeout=timeout,
            transport=transport,
        )

    # openai
    resolved_base = (
        base
        or _env(environ, "EVA_BIG_BASE")
        or _env(environ, "OPENAI_BASE_URL")
        or _env(environ, "TEACHER_BASE")
    )
    if not resolved_base:
        raise ChatConfigError(
            "openai: не задан base; укажите --base или EVA_BIG_BASE "
            "(для облака — например https://api.deepseek.com)"
        )
    resolved_model = (
        model or _env(environ, "EVA_BIG_MODEL") or _env(environ, "TEACHER_MODEL")
    )
    if not resolved_model:
        raise ChatConfigError(
            "openai: не задана модель; укажите --model или EVA_BIG_MODEL"
        )
    resolved_key = (
        key
        or _env(environ, "EVA_BIG_KEY")
        or _env(environ, "OPENAI_API_KEY")
        or _env(environ, "TEACHER_KEY")
    )
    prompt_price = (
        price_prompt
        if price_prompt is not None
        else _env_float(environ, "EVA_BIG_PRICE_PROMPT", 0.0)
    )
    completion_price = (
        price_completion
        if price_completion is not None
        else _env_float(environ, "EVA_BIG_PRICE_COMPLETION", 0.0)
    )
    return OpenAIChatBackend(
        base=resolved_base,
        model=resolved_model,
        key=resolved_key,
        timeout=timeout,
        price_prompt_rub=prompt_price,
        price_completion_rub=completion_price,
        stream_transport=stream_transport,
    )


# --------------------------------------------------------------------------- #
# Системный промпт
# --------------------------------------------------------------------------- #

def resolve_system(value: str | None) -> str:
    """Системный промпт из строки: ``@путь``/существующий файл — прочитать.

    Строка без файла считается текстом промпта. Пустая строка/``-`` —
    «системного промпта нет».
    """
    text = (value or "").strip()
    if not text or text == "-":
        return ""
    path: Path | None = None
    if text.startswith("@"):
        path = Path(text[1:]).expanduser()
    else:
        candidate = Path(text).expanduser()
        if candidate.is_file():
            path = candidate
    if path is None:
        return text
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ChatConfigError(f"не удалось прочитать системный промпт {path}: {exc}") from exc


# --------------------------------------------------------------------------- #
# REPL
# --------------------------------------------------------------------------- #

HELP_TEXT = """команды:
  /reset            очистить историю (системный промпт остаётся)
  /system <текст>   задать системный промпт (@файл — прочитать из файла)
  /system           показать текущий системный промпт
  /system -         убрать системный промпт
  /stats            метрики: TTFT, токены, ток/с, символы, стоимость, контекст
  /help             эта справка
  /quit             выход (или Ctrl-D)
"""


class ChatRepl:
    """Цикл «вопрос -> стриминг ответа -> история».

    Ввод/вывод инъектируется (``stdin``/``stdout``/``stderr``), поэтому
    REPL и команды проверяются тестами без терминала и без сети.
    """

    def __init__(
        self,
        backend: ChatBackend,
        session: ChatSession,
        *,
        stdin: Any = None,
        stdout: Any = None,
        stderr: Any = None,
        show_stats: bool = False,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.backend = backend
        self.session = session
        self.stdin = stdin if stdin is not None else sys.stdin
        self.stdout = stdout if stdout is not None else sys.stdout
        self.stderr = stderr if stderr is not None else sys.stderr
        self.show_stats = show_stats
        self.clock = clock

    # -- вывод --------------------------------------------------------------

    def _out(self, text: str = "") -> None:
        self.stdout.write(text)
        self.stdout.flush()

    def _err(self, text: str = "") -> None:
        self.stderr.write(text)
        self.stderr.flush()

    def greeting(self) -> None:
        self._out(
            f"chat · backend={self.backend.name} · model={self.backend.model} · "
            f"ctx={self.session.ctx} · max_tokens={self.session.max_tokens}\n"
        )
        if self.session.system:
            self._out(f"system: {_collapse(self.session.system, 80)}\n")
        self._out("команды: /reset /system /stats /help /quit\n")

    # -- один ход -----------------------------------------------------------

    def turn(self, text: str, *, prefix: bool = True) -> TurnStats | None:
        """Отправить сообщение, напечатать ответ потоком, записать метрики.

        При ошибке вопрос откатывается из истории: следующий запрос не
        должен содержать вопрос, на который модель не ответила.
        ``prefix=False`` — для ``--once``: в stdout только сам ответ.
        """
        text = text.strip()
        if not text:
            return None
        self.session.add_user(text)
        messages, trim = self.session.build_messages()
        if trim.trimmed:
            self._err(
                f"(история обрезана: выброшено {trim.dropped} сообщений, "
                f"контекст {self.session.ctx})\n"
            )

        start = self.clock()
        first: float | None = None
        chunks: list[str] = []
        usage: Mapping[str, Any] = {}
        finish: str | None = None
        failure: Exception | None = None
        interrupted = False
        try:
            if prefix:
                self._out("assistant> ")
            for event in self.backend.stream(messages, max_tokens=self.session.max_tokens):
                if event.done:
                    usage = dict(event.usage or {})
                    finish = event.finish_reason
                    if event.text:
                        if first is None:
                            first = self.clock()
                        chunks.append(event.text)
                        self._out(event.text)
                    continue
                if not event.text:
                    continue
                if first is None:
                    first = self.clock()
                chunks.append(event.text)
                self._out(event.text)
        except KeyboardInterrupt:
            interrupted = True
            self._cancel_backend()
        except Exception as exc:  # noqa: BLE001 — REPL не должен падать от модели
            failure = exc
        end = self.clock()
        self._out("\n")

        if interrupted or failure is not None:
            self.session.drop_last("user")
            if interrupted:
                self._err("прервано (Ctrl-C); вопрос убран из истории\n")
            else:
                self._err(f"ошибка: {failure}\n")
            return None

        answer = "".join(chunks)
        elapsed = max(0.0, end - start)
        ttft = (first - start) if first is not None else elapsed
        stats = self.session.finish_turn(
            answer,
            messages,
            usage,
            ttft_s=ttft,
            elapsed_s=elapsed,
            finish_reason=finish,
        )
        if answer.strip():
            self.session.add_assistant(answer)
        else:
            self._err("(пустой ответ: в историю не записан)\n")
        if self.show_stats:
            self._err(format_turn_brief(stats) + "\n")
        return stats

    def _cancel_backend(self) -> None:
        cancel = getattr(self.backend, "cancel", None)
        if callable(cancel):
            try:
                cancel()
            except Exception:  # noqa: BLE001 — отмена не должна ронять выход
                pass

    # -- команды ------------------------------------------------------------

    def command(self, line: str) -> bool:
        """Обработать команду; ``True`` — пора выходить из REPL.

        Неизвестная команда печатает подсказку и НЕ уходит в модель.
        """
        parts = line.strip().split(maxsplit=1)
        name = parts[0].lower()
        argument = parts[1] if len(parts) > 1 else ""
        if name in ("/quit", "/exit", "/q"):
            return True
        if name == "/reset":
            removed = len(self.session.history)
            self.session.reset()
            self._out(f"история очищена ({removed} сообщений); системный промпт сохранён\n")
            return False
        if name == "/system":
            if not argument.strip():
                self._out(
                    f"system: {self.session.system or '(нет)'}\n"
                )
                return False
            try:
                new_system = resolve_system(argument)
            except ChatConfigError as exc:
                self._err(f"ошибка: {exc}\n")
                return False
            self.session.set_system(new_system)
            if new_system:
                self._out(
                    f"системный промпт обновлён: {len(new_system)} символов\n"
                )
            else:
                self._out("системный промпт убран\n")
            return False
        if name == "/stats":
            self._out(
                format_stats(
                    self.session,
                    backend_name=self.backend.name,
                    model=self.backend.model,
                )
                + "\n"
            )
            return False
        if name in ("/help", "/?"):
            self._out(HELP_TEXT)
            return False
        self._err(f"неизвестная команда {parts[0]!r}; /help — список команд\n")
        return False

    # -- цикл ---------------------------------------------------------------

    def run(self) -> int:
        """REPL до /quit, EOF или Ctrl-C; возвращает код выхода (130 — Ctrl-C)."""
        self.greeting()
        try:
            while True:
                self._out("you> ")
                line = self.stdin.readline()
                if not line:
                    self._out("\n")
                    return 0
                text = line.strip()
                if not text:
                    continue
                if text.startswith("/"):
                    if self.command(text):
                        return 0
                    continue
                self.turn(text)
        except KeyboardInterrupt:
            self._out("\n")
            self._err("прервано (Ctrl-C)\n")
            return 130


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _positive_int(value: str) -> int:
    try:
        number = int(str(value).strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"ожидалось целое > 0, получено {value!r}") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError(f"ожидалось > 0, получено {number}")
    return number


def _positive_float(value: str) -> float:
    try:
        number = float(str(value).replace(",", ".").strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"ожидалось число > 0, получено {value!r}") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError(f"ожидалось > 0, получено {number}")
    return number


def _non_negative_float(value: str) -> float:
    try:
        number = float(str(value).replace(",", ".").strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"ожидалось число >= 0, получено {value!r}") from exc
    if number < 0:
        raise argparse.ArgumentTypeError(f"ожидалось >= 0, получено {number}")
    return number





#: Служебные маркеры конца: слабо квантованная модель иногда выдаёт их как
#: обычный текст (``<|endoftext|>Human:``), а не через eos_token_id.
_STOP_MARKERS = ("<|endoftext|>", "<|im_end|>", "<|im_start|>user")

#: Модель иногда доигрывает диалог за пользователя: пишет «user:» и продолжает
#: отвечать на собственный вопрос. Ловим начало такой реплики по строке.
_STOP_LINE = re.compile(r"(?m)^[ \t]*(?:user|human|ассистент|assistant)\s*:", re.I)


def _hit_stop_marker(chunk: str) -> bool:
    return any(marker in chunk for marker in _STOP_MARKERS) or bool(_STOP_LINE.search(chunk))


def _strip_stop_markers(chunk: str) -> str:
    """Убрать служебный маркер и всё, что модель «дописала» после него."""
    cut = len(chunk)
    for marker in _STOP_MARKERS:
        index = chunk.find(marker)
        if index >= 0:
            cut = min(cut, index)
    found = _STOP_LINE.search(chunk)
    if found:
        cut = min(cut, found.start())
    return chunk[:cut]


#: Сколько символов ответа держим неотданными: маркер «user:» короче, но
#: запас нужен и на «<|im_end|>» с переносами.
_MARKER_LOOKAHEAD = 16


def _looks_like_loop(text: str, *, words: int = 3, repeats: int = 3,
                     window: int = 400) -> bool:
    """Похоже ли на зацикливание: короткая фраза повторяется много раз подряд.

    Слабо квантованная модель может не выдать EOS и накручивать одну и ту же
    фразу, слегка меняя окружение. Точное совпадение хвоста это не ловит,
    поэтому ищем повтор n-граммы из слов в скользящем окне.
    """
    tail = text[-window:]
    parts = tail.split()
    if len(parts) < words * repeats:
        return False
    counts: dict[str, int] = {}
    for i in range(len(parts) - words + 1):
        key = " ".join(parts[i:i + words])
        counts[key] = counts.get(key, 0) + 1
        if counts[key] >= repeats:
            return True
    return False

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chat.py",
        description=(
            "Чат-REPL для проверки моделей: локальная ВТМ, ollama, "
            "OpenAI-совместимый API. История, стриминг, TTFT/ток-с/стоимость."
        ),
        epilog=(
            "примеры:\n"
            "  python3 tools/chat.py --backend ollama --model qwen2.5:3b\n"
            "  python3 tools/chat.py --backend vtm --model runs/tiny/model.vtm "
            "--tokenizer tests/vtm/fixtures/tokenizer.json\n"
            "  python3 tools/chat.py --backend openai --model gpt-4o-mini "
            "--base https://api.example.com/v1 --system prompts/system.assistant.md\n"
            "  python3 tools/chat.py --backend ollama --model qwen2.5:3b "
            "--once 'привет' --stats\n"
            "\n"
            "окружение (дефолты): EVA_VTM_MODEL/EVA_VTM_TOKENIZER/EVA_VTM_LIB, "
            "OLLAMA_BASE/OLLAMA_HOST/OLLAMA_MODEL, EVA_BIG_BASE/EVA_BIG_MODEL/"
            "EVA_BIG_KEY, EVA_BIG_PRICE_PROMPT/EVA_BIG_PRICE_COMPLETION."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--backend",
        choices=DEFAULT_BACKENDS,
        default="vtm",
        help="бэкенд: vtm (локальный .vtm), ollama, openai (по умолчанию vtm)",
    )
    parser.add_argument("--model", default=None, help="имя модели или путь к .vtm")
    parser.add_argument(
        "--base",
        default=None,
        help="base URL для ollama/openai (ollama: http://127.0.0.1:11434)",
    )
    parser.add_argument(
        "--key",
        default=None,
        help="ключ API (лучше через EVA_BIG_KEY/OPENAI_API_KEY; в вывод не попадает)",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="tokenizer.json для vtm (по умолчанию рядом с моделью)",
    )
    parser.add_argument(
        "--system",
        default="",
        help="системный промпт; @путь или существующий файл читается с диска",
    )
    parser.add_argument(
        "--ctx",
        type=_positive_int,
        default=None,
        help="окно контекста в токенах для обрезки истории (по бэкенду: 512/4096/8192)",
    )
    parser.add_argument(
        "--timeout",
        type=_positive_float,
        default=DEFAULT_TIMEOUT,
        help=f"таймаут HTTP-запроса, сек (по умолчанию {DEFAULT_TIMEOUT:g}; для vtm не нужен)",
    )
    parser.add_argument(
        "--max-tokens",
        type=_positive_int,
        default=DEFAULT_MAX_TOKENS,
        help=f"лимит completion-токенов на ответ (по умолчанию {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--price-prompt",
        type=_non_negative_float,
        default=None,
        help="₽ за 1000 prompt-токенов (для openai; иначе EVA_BIG_PRICE_PROMPT)",
    )
    parser.add_argument(
        "--price-completion",
        type=_non_negative_float,
        default=None,
        help="₽ за 1000 completion-токенов (для openai; иначе EVA_BIG_PRICE_COMPLETION)",
    )
    parser.add_argument(
        "--no-cache-prefix",
        action="store_true",
        help="выключить кэш префикса ВТМ (по умолчанию включён: второй запрос быстрее)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="печатать краткую статистику после каждого ответа",
    )
    parser.add_argument(
        "--once",
        metavar="ТЕКСТ",
        default=None,
        help="один запрос без REPL (удобно для скриптов и проверки)",
    )
    return parser


def _backend_from_args(
    args: argparse.Namespace,
    *,
    environ: Mapping[str, str] | None = None,
    transport: Any = None,
    stream_transport: Any = None,
) -> ChatBackend:
    return build_backend(
        backend=args.backend,
        model=args.model,
        base=args.base,
        key=args.key,
        tokenizer=args.tokenizer,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        price_prompt=args.price_prompt,
        price_completion=args.price_completion,
        cache_prefix=False if args.no_cache_prefix else None,
        environ=environ,
        transport=transport,
        stream_transport=stream_transport,
    )


def _resolve_ctx(args: argparse.Namespace) -> int:
    if args.ctx is not None:
        return int(args.ctx)
    return DEFAULT_CONTEXT.get(str(args.backend), 4096)


def main(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
    stderr: Any = None,
    environ: Mapping[str, str] | None = None,
    transport: Any = None,
    stream_transport: Any = None,
) -> int:
    """Точка входа CLI; возвращает код выхода (0 — ок, 2 — конфиг)."""
    out = stdout if stdout is not None else sys.stdout
    err = stderr if stderr is not None else sys.stderr
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        backend = _backend_from_args(
            args, environ=environ, transport=transport, stream_transport=stream_transport
        )
    except ChatConfigError as exc:
        print(f"chat: {exc}", file=err)
        return 2

    try:
        system = resolve_system(args.system)
    except ChatConfigError as exc:
        print(f"chat: {exc}", file=err)
        return 2

    session = ChatSession(
        system=system,
        ctx=_resolve_ctx(args),
        max_tokens=int(args.max_tokens),
        counter=backend.counter,
    )
    repl = ChatRepl(
        backend,
        session,
        stdin=stdin,
        stdout=out,
        stderr=err,
        show_stats=bool(args.stats),
    )
    try:
        if args.once is not None:
            prompt = str(args.once).strip()
            if not prompt:
                print("chat: --once пуст: нечего отправлять", file=err)
                return 2
            stats = repl.turn(prompt, prefix=False)
            return 0 if stats is not None else 1
        return repl.run()
    except (ChatConfigError, ChatBackendError) as exc:
        print(f"chat: {exc}", file=err)
        return 1
    finally:
        backend.close()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
