"""Базовый протокол скиллов EVA32.

Скилл — детерминированный ответ из каталога. Он **не вызывает модель**:
``match`` решает, его ли это вопрос, ``run`` собирает текст и источники.

Контракт — ``docs/hybrid.md``::

    class Skill:
        name: str
        def match(self, text: str) -> bool
        def run(self, text: str, ctx: dict) -> SkillResult | None

``SkillResult`` — словарь ``{"text", "sources", "catalog_ids"}``. Поле
``skill`` добавлено для маршрутизатора: по нему он пишет имя скилла в
``Answer.skill`` (см. ``eva/router.py``).
"""

from __future__ import annotations

import re
from typing import Any, TypedDict

from eva.catalog import Catalog, load, normalize

__all__ = [
    "SOURCE",
    "Skill",
    "SkillResult",
    "get_catalog",
    "is_third_party",
    "make_result",
]

#: Единственный источник скиллов v1 — каталог услуг и тарифов.
SOURCE = "[1]"

_CATALOG_CACHE: Catalog | None = None

#: Вопрос про чужую компанию — скиллы молчат, отвечает модель с отказом.
#: Кавычки/заглавные буквы проверяем по исходному тексту: ``normalize`` их
#: срезает, и «компания «Ромашка»» иначе стала бы безымянной «компанией».
_THIRD_PARTY_QUOTED_RE = re.compile(
    r"(?:компани\w*|организаци\w*|клиент\w*|конкурент\w*)\s+"
    r"(?:[«\"“][^»\"”]+[»\"”]|№\s*\d+|номер\s*\d+|по\s+названию\s+\w+)",
    re.IGNORECASE,
)
_THIRD_PARTY_NAMED_RE = re.compile(
    r"(?:компани\w*|организаци\w*|клиент\w*|конкурент\w*)\s+[А-ЯЁ][а-яё]+"
)
_THIRD_PARTY_GENERIC_RE = re.compile(
    r"(?:друг\w*|чуж\w*)\s+(?:компани|организаци)\w*",
    re.IGNORECASE,
)


class SkillResult(TypedDict, total=False):
    """Результат скилла: текст, источники и id позиций каталога."""

    text: str
    sources: list[str]
    catalog_ids: list[str]
    skill: str


class Skill:
    """Протокол скилла. Наследники переопределяют ``name``/``match``/``run``."""

    name = ""

    def match(self, text: str, ctx: dict[str, Any] | None = None) -> bool:  # pragma: no cover
        raise NotImplementedError

    def run(self, text: str, ctx: dict[str, Any]) -> SkillResult | None:  # pragma: no cover
        raise NotImplementedError


def get_catalog(ctx: dict[str, Any] | None = None) -> Catalog:
    """Каталог из ``ctx`` (подмена в тестах) или лениво загруженный общий."""
    global _CATALOG_CACHE
    if ctx:
        injected = ctx.get("catalog")
        if isinstance(injected, Catalog):
            return injected
        path = ctx.get("catalog_path")
        if path:
            return load(path)
    if _CATALOG_CACHE is None:
        _CATALOG_CACHE = load()
    return _CATALOG_CACHE


def is_third_party(text: str) -> bool:
    """Спрашивают про данные другой компании? Тогда скилл молчит."""
    raw = str(text or "")
    if _THIRD_PARTY_QUOTED_RE.search(raw) or _THIRD_PARTY_NAMED_RE.search(raw):
        return True
    return bool(_THIRD_PARTY_GENERIC_RE.search(normalize(raw)))


def make_result(
    skill: str,
    text: str,
    *,
    sources: list[str] | None = None,
    catalog_ids: list[str] | None = None,
) -> SkillResult:
    """Собрать ``SkillResult`` с одним источником по умолчанию."""
    return {
        "text": text,
        "sources": list(sources or [SOURCE]),
        "catalog_ids": list(catalog_ids or []),
        "skill": skill,
    }
