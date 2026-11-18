"""Скиллы EVA32 — детерминированные ответы из каталога.

Пакет реализует контракт ``docs/hybrid.md``:

    from eva.skills import try_skills

    result = try_skills("Сколько стоит тариф «Старт»?", {})
    # {"text": ..., "sources": ["[1]"], "catalog_ids": ["start"], "skill": "pricing"}

Скилл никогда не вызывает модель. Если совпадения нет — возвращается ``None``.
"""

from __future__ import annotations

from eva.skills.base import SOURCE, Skill, SkillResult, get_catalog, make_result
from eva.skills.contacts import ContactsSkill
from eva.skills.hours import HoursSkill
from eva.skills.pricing import PricingSkill
from eva.skills.refunds import RefundsSkill
from eva.skills.registry import SKILLS, skills, try_skills

__all__ = [
    "SOURCE",
    "SKILLS",
    "ContactsSkill",
    "HoursSkill",
    "PricingSkill",
    "RefundsSkill",
    "Skill",
    "SkillResult",
    "get_catalog",
    "make_result",
    "skills",
    "try_skills",
]
