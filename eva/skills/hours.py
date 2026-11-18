"""Скилл ``hours``: часы работы офиса и техподдержки.

Факты компании хранятся рядом со скиллом: каталог ``data/catalog.json``
описывает только услуги и тарифы (контракт ``docs/hybrid.md``). Источник
ответа — ``[1]``, как и у остальных скиллов v1.
"""

from __future__ import annotations

import re
from typing import Any

from eva.catalog import normalize
from eva.skills.base import Skill, SkillResult, is_third_party, make_result

__all__ = ["HoursSkill"]

_OFFICE_SUMMARY = (
    "Офис в Москве: по будням с 9:00 до 19:00, в субботу с 10:00 до 16:00, "
    "в воскресенье закрыт."
)
_SUPPORT_SUMMARY = (
    "Техподдержка в чате: с 9:00 до 21:00 по МСК, среднее время ответа — "
    "15 минут; в выходные работает дежурная смена."
)

#: Вопрос именно про режим работы, а не про цену.
_HOURS_RE = re.compile(
    r"(?:"
    r"час[ыов]\s+работы"
    r"|режим\s+работы"
    r"|график\s+работы"
    r"|время\s+работы"
    r"|во\s+сколько\s+(?:\w+\s+){0,2}(?:открыва|закрыва|начина|заканчива|работа|отвеча)"
    r"|до\s+скольки"
    r"|когда\s+(?:\w+\s+){0,2}(?:работа|открыва|закрыва|открыт|закрыт|отвеча)"
    r"|работаете\s+ли|работает\s+ли"
    r"|в\s+какие\s+дни"
    r"|сколько\s+времени\s+(?:работа|отвеча)"
    r"|(?:выходн|суббот|воскресень|будн|праздник)\w*"
    r"|работа\w*\s+в\s+(?:суббот|воскресень|праздник|выходн)\w*"
    r")",
    re.IGNORECASE,
)

#: Без этих слов вопрос про часы может быть про что угодно («в выходные»).
_ENTITY_RE = re.compile(
    r"(?:офис\w*|поддержк\w*|техподдержк\w*|\bчат\w*|\bвы\b|\bваш\w*)",
    re.IGNORECASE,
)

#: Общие формулировки про режим работы — сущность не обязательна.
_GENERIC_HOURS_RE = re.compile(
    r"(?:"
    r"час[ыов]\s+работы"
    r"|режим\s+работы"
    r"|график\s+работы"
    r"|время\s+работы"
    r"|во\s+сколько"
    r"|до\s+скольки"
    r"|когда\s+(?:вы\s+)?(?:работа|открыва|закрыва|открыт|закрыт)"
    r"|работаете\s+ли|работает\s+ли"
    r"|в\s+какие\s+дни"
    r")",
    re.IGNORECASE,
)


class HoursSkill(Skill):
    """Отвечает на вопросы «когда работаете» и «во сколько закрываетесь»."""

    name = "hours"

    def match(self, text: str, ctx: dict[str, Any] | None = None) -> bool:
        if not text or not text.strip() or is_third_party(text):
            return False
        value = normalize(text)
        if not _HOURS_RE.search(value):
            return False
        return bool(_ENTITY_RE.search(value) or _GENERIC_HOURS_RE.search(value))

    def run(self, text: str, ctx: dict[str, Any]) -> SkillResult | None:
        if not self.match(text, ctx):
            return None
        value = normalize(text)

        support = bool(re.search(r"(?:поддержк|техподдержк|чат|отвеча)", value))
        office = bool(re.search(r"(?:офис|приемн|москв)", value)) or not support

        # «Какие часы работы?» без уточнения — рассказываем про оба канала.
        generic = bool(_GENERIC_HOURS_RE.search(value))
        if generic and not support and not re.search(r"(?:офис|приемн|москв)", value):
            office = True
            support = True

        parts: list[str] = []
        if office:
            parts.append(self._office_text(value))
        if support:
            parts.append(_SUPPORT_SUMMARY + " [1]")
        return make_result(self.name, " ".join(parts), sources=["[1]"])

    @staticmethod
    def _office_text(value: str) -> str:
        sunday = "воскресень" in value
        saturday = "суббот" in value
        weekday = "будн" in value or "понедельник" in value or "пятниц" in value

        if sunday and saturday:
            return _OFFICE_SUMMARY + " [1]"
        if sunday:
            return "В воскресенье офис закрыт. [1]"
        if saturday:
            return "В субботу офис работает с 10:00 до 16:00. [1]"
        if weekday:
            return "По будням офис работает с 9:00 до 19:00. [1]"
        return _OFFICE_SUMMARY + " [1]"
