"""Реестр скиллов EVA32: первый совпавший скилл или ``None``.

Контракт — ``docs/hybrid.md``::

    try_skills(text: str, ctx: dict) -> SkillResult | None

Скиллы v1 (каталог): ``pricing``, ``hours``, ``refunds``, ``contacts``.
Скиллы v2 (вычисляемые, без каталога и без сети): ``time``, ``units``,
``math``, ``currency``. Порядок важен: более точный и более «каталожный»
скилл стоит раньше — новые скиллы не имеют права отобрать вопрос у старого.
Если никто не совпал — реестр молчит, и маршрутизатор передаёт вопрос модели.
"""

from __future__ import annotations

from typing import Any

from eva.skills.base import SkillResult
from eva.skills.calculator import MathSkill
from eva.skills.clock import TimeSkill
from eva.skills.contacts import ContactsSkill
from eva.skills.currency import CurrencySkill
from eva.skills.hours import HoursSkill
from eva.skills.pricing import PricingSkill
from eva.skills.refunds import RefundsSkill
from eva.skills.units import UnitsSkill

__all__ = ["SKILLS", "skills", "try_skills"]

#: Порядок: цена → часы → возврат → контакты → время → единицы → счёт → курс.
SKILLS = (
    PricingSkill(),
    HoursSkill(),
    RefundsSkill(),
    ContactsSkill(),
    TimeSkill(),
    UnitsSkill(),
    MathSkill(),
    CurrencySkill(),
)


def skills() -> list:
    """Список зарегистрированных скиллов (копия, для интроспекции/тестов)."""
    return list(SKILLS)


def try_skills(text: str, ctx: dict[str, Any] | None = None) -> SkillResult | None:
    """Вернуть результат первого скилла, который взял вопрос, иначе ``None``."""
    if not isinstance(text, str) or not text.strip():
        return None
    context = ctx if isinstance(ctx, dict) else {}
    for skill in SKILLS:
        if not skill.match(text, context):
            continue
        result = skill.run(text, context)
        if result:
            return result
    return None
