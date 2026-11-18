"""Политика фолбэков, таймаутов и бюджета гибридного маршрутизатора.

Контракт — ``docs/hybrid.md``, решение — ADR 0009. Правила:

* **малая → большая** при любой ошибке/таймауте малой, если большая доступна;
* **большая → малая** только при 429 (лимит провайдера) или исчерпании бюджета;
* бюджет на запрос проверяется **до** вызова платной модели; если платить
  нечем — :class:`BudgetExceeded`, а не тихая трата;
* все причины — человекочитаемым текстом, они попадают в ``Answer.route.reason``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

__all__ = [
    "BUDGET",
    "CONNECTION",
    "ERROR",
    "RATE_LIMIT",
    "TIMEOUT",
    "Budget",
    "BudgetExceeded",
    "ModelUnavailable",
    "PolicyError",
    "budget_downgrade_reason",
    "error_kind",
    "estimate_request_cost",
    "estimate_tokens",
    "fallback_reason",
    "fallback_target",
    "timeout_for",
]

#: Виды сбоев, важные для политики.
RATE_LIMIT = "rate_limit"
TIMEOUT = "timeout"
CONNECTION = "connection"
BUDGET = "budget"
ERROR = "error"

_CAUSE_TEXT = {
    RATE_LIMIT: "лимит запросов (429)",
    BUDGET: "бюджет провайдера исчерпан",
    TIMEOUT: "таймаут",
    CONNECTION: "сетевая ошибка",
    ERROR: "ошибка вызова",
}


# --------------------------------------------------------------------------- #
# исключения
# --------------------------------------------------------------------------- #
class PolicyError(RuntimeError):
    """Базовая ошибка политики маршрутизации."""


class BudgetExceeded(PolicyError):
    """Бюджет на запрос не позволяет выполнить вызов."""

    def __init__(
        self,
        message: str,
        *,
        route_kind: str | None = None,
        needed_rub: float = 0.0,
        budget_rub: float | None = None,
        spent_rub: float = 0.0,
    ) -> None:
        super().__init__(message)
        self.route_kind = route_kind
        self.needed_rub = float(needed_rub)
        self.budget_rub = budget_rub
        self.spent_rub = float(spent_rub)


class ModelUnavailable(PolicyError):
    """Нет ни клиента, ни настроенной модели для маршрута."""

    def __init__(self, message: str, *, route_kind: str | None = None) -> None:
        super().__init__(message)
        self.route_kind = route_kind


# --------------------------------------------------------------------------- #
# бюджет
# --------------------------------------------------------------------------- #
@dataclass
class Budget:
    """Бюджет одного запроса. ``None`` — без лимита, ``0`` — не тратить."""

    max_cost_rub: float | None = None
    spent_rub: float = 0.0

    @classmethod
    def from_config(cls, cfg: Any) -> "Budget":
        limit = getattr(cfg, "max_cost_rub", None)
        return cls(None if limit is None else float(limit), 0.0)

    @property
    def limited(self) -> bool:
        return self.max_cost_rub is not None

    @property
    def remaining(self) -> float | None:
        if not self.limited:
            return None
        return max(0.0, float(self.max_cost_rub) - self.spent_rub)

    def can_afford(self, estimate_rub: float) -> bool:
        if not self.limited:
            return True
        return float(estimate_rub) <= float(self.remaining) + 1e-9

    def charge(self, cost_rub: float) -> float:
        try:
            cost = float(cost_rub or 0.0)
        except (TypeError, ValueError):
            cost = 0.0
        self.spent_rub += max(0.0, cost)
        return self.spent_rub

    def check(self, estimate_rub: float, *, route_kind: str | None = None) -> None:
        """Бросить :class:`BudgetExceeded`, если оценка не влезает в бюджет."""
        if self.can_afford(estimate_rub):
            return
        raise BudgetExceeded(
            (
                f"бюджет на запрос исчерпан: нужно ~{float(estimate_rub):.4f} ₽, "
                f"осталось {float(self.remaining or 0.0):.4f} ₽ из {float(self.max_cost_rub or 0.0):.4f} ₽"
                + (f" (маршрут {route_kind})" if route_kind else "")
            ),
            route_kind=route_kind,
            needed_rub=estimate_rub,
            budget_rub=self.max_cost_rub,
            spent_rub=self.spent_rub,
        )


# --------------------------------------------------------------------------- #
# оценка цены и токенов
# --------------------------------------------------------------------------- #
def estimate_tokens(text: str) -> int:
    """Дешёвая оценка токенов: ~4 символа на токен, минимум 1."""
    return max(1, math.ceil(len(text or "") / 4))


def estimate_request_cost(cfg: Any, kind: str, prompt_text: str, max_tokens: int | None = None) -> float:
    """Оценка сверху: prompt по тексту + ``max_tokens`` completion."""
    completion = getattr(cfg, "max_tokens", 512) if max_tokens is None else max_tokens
    return float(cfg.estimate_cost(kind, estimate_tokens(prompt_text), int(completion)))


def timeout_for(kind: str, cfg: Any) -> float:
    """Таймаут вызова по маршруту (для клиентов и тестов)."""
    if kind == "big":
        return float(getattr(cfg, "big_timeout", 60.0))
    return float(getattr(cfg, "small_timeout", 30.0))


# --------------------------------------------------------------------------- #
# классификация сбоя
# --------------------------------------------------------------------------- #
def _exc_code(exc: BaseException) -> Any:
    for attr in ("status_code", "status", "http_status", "code", "error_kind", "kind"):
        value = getattr(exc, attr, None)
        if value is not None:
            return value
    return None


def error_kind(exc: BaseException) -> str:
    """Привести исключение клиента к одному из видов политики."""
    code = _exc_code(exc)
    if code is not None:
        text = str(code).strip().lower()
        if text.isdigit():
            if text == "429":
                return RATE_LIMIT
        elif text in {"rate_limit", "rate-limit", "rate_limited", "too_many_requests", "429"}:
            return RATE_LIMIT
        elif text in {"timeout", "timed_out", "read_timeout", "timeout_error"}:
            return TIMEOUT
        elif text in {"budget", "budget_exceeded"}:
            return BUDGET
        elif text in {"connection", "connect", "network", "connection_error"}:
            return CONNECTION

    text = str(exc or "").lower()
    if "429" in text or "rate limit" in text or "rate_limit" in text or "too many requests" in text:
        return RATE_LIMIT
    if "бюджет" in text or "budget" in text or "cost limit" in text or "лимит стоимости" in text:
        return BUDGET
    if "timeout" in text or "timed out" in text or "таймаут" in text:
        return TIMEOUT
    if "connection" in text or "connrefused" in text or "network" in text or "unreachable" in text:
        return CONNECTION
    return ERROR


# --------------------------------------------------------------------------- #
# фолбэки
# --------------------------------------------------------------------------- #
def fallback_target(kind: str, cause_kind: str) -> str | None:
    """Куда откатываться: ``small`` — на любую ошибку, ``big`` — только 429/бюджет."""
    if kind == "small":
        return "big"
    if kind == "big" and cause_kind in (RATE_LIMIT, BUDGET):
        return "small"
    return None


def fallback_reason(from_kind: str, to_kind: str, cause_kind: str, detail: str = "") -> str:
    """Причина фолбэка для ``Route.reason``."""
    base = f"fallback {from_kind}→{to_kind}: {_CAUSE_TEXT.get(cause_kind, 'ошибка')}"
    detail = " ".join(str(detail or "").split())
    if detail:
        if len(detail) > 160:
            detail = detail[:157] + "..."
        base += f" — {detail}"
    return base


def budget_downgrade_reason(previous_reason: str, estimate_rub: float, limit_rub: float | None) -> str:
    """Причина понижения маршрута из-за бюджета (big → small)."""
    reason = (
        f"fallback big→small: оценка ~{float(estimate_rub):.4f} ₽ выше бюджета "
        f"{float(limit_rub or 0.0):.4f} ₽ — платный вызов отменён"
    )
    previous_reason = str(previous_reason or "").strip()
    if previous_reason:
        reason += f" (исходный маршрут: {previous_reason})"
    return reason
