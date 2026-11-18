"""Скиллы EVA32 — детерминированные ответы из каталога и вычислений.

Пакет реализует контракт ``docs/hybrid.md``:

    from eva.skills import try_skills

    result = try_skills("Сколько стоит тариф «Старт»?", {})
    # {"text": ..., "sources": ["[1]"], "catalog_ids": ["start"], "skill": "pricing"}

Скилл никогда не вызывает модель и не ходит в сеть. Если совпадения нет —
возвращается ``None``.

Скиллы v1 отвечают из каталога (``pricing``, ``hours``, ``refunds``,
``contacts``), скиллы v2 вычисляют (``time``, ``units``, ``math``,
``currency``) и возвращают пустые ``sources``: у вычисленного ответа нет
позиции каталога, приписывать ему ``[1]`` — ложь.
"""

from __future__ import annotations

from eva.skills.base import SOURCE, Skill, SkillResult, get_catalog, make_result
from eva.skills.calculator import MathSkill
from eva.skills.clock import TimeSkill
from eva.skills.contacts import ContactsSkill
from eva.skills.currency import CurrencySkill
from eva.skills.hours import HoursSkill
from eva.skills.pricing import PricingSkill
from eva.skills.refunds import RefundsSkill
from eva.skills.registry import SKILLS, skills, try_skills
from eva.skills.units import UnitsSkill

__all__ = [
    "SOURCE",
    "SKILLS",
    "ContactsSkill",
    "CurrencySkill",
    "HoursSkill",
    "MathSkill",
    "PricingSkill",
    "RefundsSkill",
    "Skill",
    "SkillResult",
    "TimeSkill",
    "UnitsSkill",
    "get_catalog",
    "make_result",
    "skills",
    "try_skills",
]
