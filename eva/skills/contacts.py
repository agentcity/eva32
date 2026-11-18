"""Скилл ``contacts``: почта, телефон и адрес офиса.

Контактные данные — часть демонстрационного каталога EVA32. Почта
``support@orioncode.ru`` встречается в материалах репозитория; телефон и
адрес — значения каталога, их нужно заменить при выгрузке реальных данных
(задача 0007 заблокирована на внешней выгрузке).
"""

from __future__ import annotations

import re
from typing import Any

from eva.catalog import normalize
from eva.skills.base import Skill, SkillResult, is_third_party, make_result

__all__ = ["ContactsSkill"]

EMAIL = "support@orioncode.ru"
PHONE = "+7 (495) 120-32-32"
ADDRESS = "Москва, ул. Тверская, 7"

_CONTACT_RE = re.compile(
    r"(?:"
    r"контакт\w*"
    r"|телефон\w*"
    r"|почт[аыуео]\w*"
    r"|e-?mail\w*"
    r"|электронн\w*\s+адрес"
    r"|адрес\w*"
    r"|как\s+(?:можно\s+)?(?:с\s+вами\s+)?связаться"
    r"|связаться\s+с\s+вами"
    r"|позвон\w*"
    r"|написать\s+вам"
    r"|где\s+(?:вы\s+)?находитесь"
    r"|где\s+(?:находится|расположен\w*)\s+(?:ваш\s+)?офис\w*"
    r"|где\s+(?:ваш\s+)?офис\w*"
    r"|как\s+добраться"
    r"|реквизит\w*"
    r")",
    re.IGNORECASE,
)


class ContactsSkill(Skill):
    """Отвечает на вопросы «как с вами связаться»."""

    name = "contacts"

    def match(self, text: str, ctx: dict[str, Any] | None = None) -> bool:
        if not text or not text.strip() or is_third_party(text):
            return False
        return bool(_CONTACT_RE.search(normalize(text)))

    def run(self, text: str, ctx: dict[str, Any]) -> SkillResult | None:
        if not self.match(text, ctx):
            return None
        value = normalize(text)

        want_phone = bool(re.search(r"(?:телефон|позвон)", value))
        want_email = bool(re.search(r"(?:почт|e-?mail|электронн)", value))
        want_address = bool(re.search(r"(?:адрес|офис|наход|приехать)", value))
        if not (want_phone or want_email or want_address):
            want_phone = want_email = want_address = True

        parts: list[str] = []
        if want_email:
            parts.append(f"Почта: {EMAIL}.")
        if want_phone:
            parts.append(f"Телефон: {PHONE}.")
        if want_address:
            parts.append(f"Адрес офиса: {ADDRESS}.")
        return make_result(self.name, " ".join(parts) + " [1]", sources=["[1]"])
