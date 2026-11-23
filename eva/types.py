"""Типы гибридного ассистента.

Контракт: ``docs/hybrid.md``. Только стандартная библиотека, без сети,
без импортов из остальных модулей ``eva`` — этот файл является фундаментом,
на который ссылаются все.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["Message", "Request", "Route", "Answer", "ROUTE_KINDS"]

#: Допустимые маршруты: скилл, малая модель, большая модель.
ROUTE_KINDS = ("skill", "small", "big")


@dataclass
class Message:
    """Одно сообщение чата в формате OpenAI (role/content)."""

    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}

    @classmethod
    def coerce(cls, value: Any) -> "Message":
        """Принимает Message, dict или строку (строка = user-сообщение)."""
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(role=str(value.get("role") or "user"), content=str(value.get("content") or ""))
        return cls(role="user", content=str(value or ""))


@dataclass
class Request:
    """Запрос пользователя: сообщения + метаданные канала."""

    messages: list[Message] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.messages = [Message.coerce(m) for m in (self.messages or [])]
        self.meta = dict(self.meta or {})

    @property
    def text(self) -> str:
        """Текст последнего сообщения пользователя (или последнего вообще)."""
        for message in reversed(self.messages):
            if message.role == "user" and (message.content or "").strip():
                return message.content.strip()
        if self.messages:
            return (self.messages[-1].content or "").strip()
        return ""

    def to_dict(self) -> dict[str, Any]:
        return {"messages": [m.to_dict() for m in self.messages], "meta": dict(self.meta)}


@dataclass
class Route:
    """Решение маршрутизатора: кто отвечает и почему.

    ``local`` — ответила ли **своя** модель (задача 0109): ``True`` — локальная
    ВТМ, ``False`` — HTTP/API, ``None`` — маршрут не про малую модель
    (``skill``/``big``) или признак неизвестен. По нему телеметрия и
    ``/metrics`` считают, сколько ответов дала своя модель.

    ``source`` (вычисляемое свойство) — грубее: ``skill``, ``local`` или
    ``api``; в ``to_dict()`` он попадает как отдельное поле.
    """

    kind: str
    reason: str
    confidence: float = 1.0
    local: bool | None = None

    def __post_init__(self) -> None:
        self.kind = str(self.kind).strip().lower()
        self.reason = str(self.reason or "").strip()
        try:
            self.confidence = max(0.0, min(1.0, float(self.confidence)))
        except (TypeError, ValueError):
            self.confidence = 0.0
        # Только настоящий bool: 0/1 и «true» строкой — не маркер ответа.
        self.local = self.local if isinstance(self.local, bool) else None

    @property
    def source(self) -> str | None:
        """``skill`` | ``local`` | ``api``; ``None`` — источник неизвестен."""
        if self.kind == "skill":
            return "skill"
        if self.local is True:
            return "local"
        if self.kind == "big":
            return "api"
        if self.kind == "small" and self.local is False:
            return "api"
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "reason": self.reason,
            "confidence": self.confidence,
            "local": self.local,
            "source": self.source,
        }


@dataclass
class Answer:
    """Ответ ассистента вместе с маршрутом, ценой и латентностью.

    ``prefill_ms`` — сколько заняла prefill-часть (счёт промпта), если
    провайдер/локальный бэкенд сообщил эту границу (задача 0120). ``None`` —
    граница неизвестна: сервер не угадывает её, а оставляет ``prefill_ms`` и
    ``ttft_after_prefill_ms`` в ``/metrics`` пустыми.
    """

    text: str
    route: Route
    skill: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    cost_rub: float = 0.0
    latency_ms: int = 0
    prefill_ms: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.prefill_ms, bool):
            self.prefill_ms = None
        elif self.prefill_ms is not None:
            try:
                value = float(self.prefill_ms)
            except (TypeError, ValueError):
                self.prefill_ms = None
            else:
                self.prefill_ms = value if value >= 0 else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "route": self.route.to_dict(),
            "skill": self.skill,
            "usage": dict(self.usage),
            "cost_rub": self.cost_rub,
            "latency_ms": self.latency_ms,
            "prefill_ms": self.prefill_ms,
        }
