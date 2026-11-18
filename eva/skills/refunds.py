"""Скилл ``refunds``: условия возврата, сроки и заявление.

Ответ детерминированный, без модели: 14 дней, условие «не пользовались»,
почта для заявления и сроки зачисления по карте/счёту.
"""

from __future__ import annotations

import re
from typing import Any

from eva.catalog import normalize
from eva.skills.base import Skill, SkillResult, is_third_party, make_result

__all__ = ["RefundsSkill"]

_REFUND_RE = re.compile(
    r"(?:"
    r"возврат\w*"
    r"|возвращ\w*"
    r"|\bверн(?:уть|ите|ете|ет|ут|ем|ул|ула|ули|и|у|утся|ется|уться|етесь)\b"
    r"|отмен\w*\s+подписк\w*"
    r"|отпис\w*"
    r"|расторг\w*"
    r"|деньги\s+обратно"
    r"|\brefund\w*"
    r"|\bmoney\s+back\b"
    r")",
    re.IGNORECASE,
)

_GENERAL = (
    "Возврат возможен в течение 14 дней с момента оплаты, если услугой "
    "не пользовались. [1] Заявление — на почту support@orioncode.ru."
)

_PAYMENT = (
    "При оплате картой деньги возвращаются на ту же карту в течение "
    "5 рабочих дней; при оплате по счёту — на расчётный счёт в течение "
    "10 рабочих дней."
)


class RefundsSkill(Skill):
    """Отвечает на вопросы про возврат денег и отмену подписки."""

    name = "refunds"

    def match(self, text: str, ctx: dict[str, Any] | None = None) -> bool:
        if not text or not text.strip() or is_third_party(text):
            return False
        return bool(_REFUND_RE.search(normalize(text)))

    def run(self, text: str, ctx: dict[str, Any]) -> SkillResult | None:
        if not self.match(text, ctx):
            return None
        value = normalize(text)
        parts: list[str] = []

        if re.search(r"(?:подписк|отпис)", value):
            parts.append(
                "Отменить подписку можно в разделе «Тарифы» личного кабинета. [1]"
            )
        parts.append(_GENERAL)
        parts.append(_PAYMENT)

        if re.search(r"(?:уже\s+пользовал|пользовались|воспользовал)", value):
            parts.append(
                "Если услугой уже воспользовались, возврат по этому условию "
                "не предусмотрен."
            )

        return make_result(self.name, " ".join(parts), sources=["[1]"])
