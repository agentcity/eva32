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
    """Решение маршрутизатора: кто отвечает и почему."""

    kind: str
    reason: str
    confidence: float = 1.0

    def __post_init__(self) -> None:
        self.kind = str(self.kind).strip().lower()
        self.reason = str(self.reason or "").strip()
        try:
            self.confidence = max(0.0, min(1.0, float(self.confidence)))
        except (TypeError, ValueError):
            self.confidence = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "reason": self.reason, "confidence": self.confidence}


@dataclass
class Answer:
    """Ответ ассистента вместе с маршрутом, ценой и латентностью."""

    text: str
    route: Route
    skill: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    cost_rub: float = 0.0
    latency_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "route": self.route.to_dict(),
            "skill": self.skill,
            "usage": dict(self.usage),
            "cost_rub": self.cost_rub,
            "latency_ms": self.latency_ms,
        }
