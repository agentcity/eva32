"""Скилл ``pricing``: цена, лимиты и заметки по тарифу или услуге.

Ответ детерминированный: берётся позиция каталога, цена форматируется,
источник ``[1]`` попадает и в текст, и в ``sources``. Модель не вызывается.
"""

from __future__ import annotations

import re
from typing import Any

from eva.catalog import (
    Catalog,
    alias_in_tokens,
    stem,
    token_match,
    tokens,
)
from eva.skills.base import Skill, SkillResult, get_catalog, is_third_party, make_result

__all__ = ["PricingSkill"]

#: Вопрос о цене/лимитах/составе, а не о чём-то ещё.
_INTENT_RE = re.compile(
    r"(?:"
    r"сколько\s+(?:сто\w*|плат\w*|обойд\w*|будет|входит|за\b)"
    r"|сколько\s+времени\s+(?:занимает|займет|займут|длится|длятся|нужно|потребуется)"
    r"|стоимост\w*"
    r"|цен[аыуе]\w*"
    r"|прайс\w*"
    r"|поч[её]м"
    r"|что\s+(?:входит|включает)"
    r"|лимит\w*"
    r"|сколько\s+(?:сотрудник|услуг|позиц|человек|мест)\w*"
    r"|есть\s+ли\s+(?:api|интеграц)\w*"
    r"|\bapi\b"
    r"|какие\s+(?:у\s+вас\s+)?тарифы"
    r"|перечислите\s+(?:все\s+)?тарифы"
    r"|список\s+тарифов"
    r"|тарифы\s+и\s+цены"
    r")",
    re.IGNORECASE,
)

#: Просьба перечислить все тарифы — отдельный сценарий, без имени позиции.
_LIST_RE = re.compile(
    r"(?:"
    r"какие\s+(?:у\s+вас\s+)?тарифы"
    r"|перечислите\s+(?:все\s+)?тарифы"
    r"|список\s+тарифов"
    r"|тарифы\s+и\s+цены"
    r"|сколько\s+стоят\s+(?:все\s+)?тарифы"
    r"|цены\s+на\s+тарифы"
    r"|какие\s+тарифы\s+есть"
    r")",
    re.IGNORECASE,
)

#: Слова, которые не делают запрос «чужим»: уточнения, связки, единицы.
_GENERIC_WORDS = (
    # вопрос и вежливость
    "сколько", "что", "чем", "чему", "чего", "кто", "кого", "кому", "где",
    "когда", "куда", "откуда", "почему", "зачем", "как", "какой", "какая",
    "какие", "какое", "каких", "каком", "какую", "который", "которая",
    "которые", "такой", "такая", "такие", "столько", "можно", "нельзя",
    "нужно", "надо", "нужен", "нужна", "нужны", "есть", "был", "была", "было",
    "были", "будет", "будут", "будем", "может", "пожалуйста", "подскажите",
    "скажите", "расскажите", "покажите", "покажи", "дайте", "будьте", "хочу",
    "хотим", "интересует", "вопрос", "спасибо", "здравствуйте", "привет",
    "добрый", "день", "сейчас", "вообще", "также", "тоже", "еще", "уже",
    "только", "просто", "более", "менее", "около", "примерно", "минимум",
    "максимум", "каждый", "любой", "другой", "весь", "все", "вся", "всех",
    "всем", "это", "этот", "эта", "эти", "она", "оно", "они", "он", "мы",
    "нас", "нам", "наш", "наша", "наши", "мне", "меня", "мой", "моя", "мои",
    "себе", "себя", "свой", "своя", "свои", "ваш", "ваша", "ваши", "вас",
    "вам", "вы", "их", "его", "ее", "ним", "них", "него", "нее",
    # единицы и деньги
    "тариф", "услуга", "сервис", "сотрудник", "человек", "люди", "позиция",
    "место", "каталог", "лимит", "ограничение", "срок", "время", "времени", "временем", "день",
    "дня", "дней", "дни", "дню", "днем", "час", "часы", "часов", "часа",
    "часу", "минута", "минуты", "минут", "минуту", "месяц", "месяцы",
    "месяцев", "месяца", "год", "года", "году", "годом", "годы", "годов",
    "годовой", "годовая", "годовых", "подписка", "подписки", "подписку",
    "подписке", "оплата", "оплаты", "оплату", "оплате", "оплатой", "платить",
    "платит", "платят", "платим", "платеж", "платежи", "платежей", "счет",
    "счета", "счету", "счетом", "счете", "карта", "карты", "карту", "карте",
    "картой", "рубль", "рубля", "рублей", "рубли", "рублях", "руб", "цена",
    "цены", "цену", "цене", "ценой", "цен", "стоимость", "стоимости",
    "стоимостью", "стоит", "стоят", "стоить", "скидка", "скидки", "скидку",
    "скидке", "скидкой", "процент", "проценты", "процентов", "api",
    "интеграция", "интеграции", "интеграцию", "доступ", "доступа", "доступен",
    "доступна", "доступно", "запрос", "запросы", "запросов", "сутки", "суток",
    "заказ", "заказа", "заказы", "заказов", "компания", "компании",
    "компанию", "компанией", "команда", "команды", "команду", "команде",
    "проект", "проекта", "проекты", "проектов", "работа", "работы", "работу",
    "работе", "работой", "работает", "работают", "работать", "работаете",
    "работал", "работала", "действует", "действуют", "займет", "займут",
    "занимает", "длится", "длятся", "проходит", "проходят", "получить",
    "получите", "оформить", "оформите", "входит", "включает", "поддерживает",
    "поддерживают", "подключить", "подключать", "подключение", "разместить",
    "размещать", "размещение", "отвечает", "отвечают", "ответ", "ответа", "бесплатно",
    "бесплатный", "включен", "включена", "включено", "предусмотрен",
    "предусмотрена", "предусмотрено", "график", "графика", "режим", "режима",
    "документ", "документы", "документов", "документа", "трёх", "трех",
    "два", "две", "три", "четыре", "пять", "шесть", "семь", "восемь",
    "девять", "десять", "двадцать", "тридцать", "пятьдесят", "сто",
    "тысяча", "тысячи", "тысяч",
    # предлоги, союзы, частицы
    "и", "в", "во", "на", "за", "по", "с", "со", "для", "от", "до", "из",
    "у", "к", "ко", "о", "об", "обо", "при", "про", "а", "но", "или", "либо",
    "же", "ли", "бы", "не", "ни", "без", "над", "под", "через",
)

_GENERIC_STEMS = {stem(word) for word in _GENERIC_WORDS}


def _is_generic(token: str) -> bool:
    if token.isdigit():
        return True
    if len(token) <= 2 and token.isalpha():
        return True
    return stem(token) in _GENERIC_STEMS


def _aliases(item: dict[str, Any]) -> list[str]:
    values = [str(item.get("name") or "")]
    values.extend(str(value) for value in item.get("synonyms") or [])
    return [value for value in values if value.strip()]


def _format_price(item: dict[str, Any]) -> str:
    amount = item.get("price_rub")
    if isinstance(amount, float) and not amount.is_integer():
        number = f"{amount:.2f}".rstrip("0").rstrip(".").replace(".", ",")
    else:
        number = f"{int(amount):,}".replace(",", " ")
    suffix = {"month": "/мес", "order": " за заказ"}.get(str(item.get("period") or ""), "")
    prefix = str(item.get("price_prefix") or "").strip()
    price = f"{number} ₽{suffix}"
    return f"{prefix} {price}" if prefix else price


def _label(item: dict[str, Any]) -> str:
    return "Тариф" if item.get("kind") == "tariff" else "Услуга"


class PricingSkill(Skill):
    """Отвечает на вопросы о цене, лимитах и составе позиции каталога."""

    name = "pricing"

    def __init__(self, catalog: Catalog | None = None):
        self._catalog = catalog

    # ------------------------------------------------------------------ #
    # протокол
    # ------------------------------------------------------------------ #
    def match(self, text: str, ctx: dict[str, Any] | None = None) -> bool:
        if not text or not text.strip() or is_third_party(text):
            return False
        catalog = self._catalog_for(ctx or {})
        return bool(self._items(text, catalog))

    def run(self, text: str, ctx: dict[str, Any]) -> SkillResult | None:
        if not text or not text.strip() or is_third_party(text):
            return None
        catalog = self._catalog_for(ctx)
        items = self._items(text, catalog)
        if not items:
            return None
        return self._answer(items, text)

    # ------------------------------------------------------------------ #
    # выбор позиций
    # ------------------------------------------------------------------ #
    def _catalog_for(self, ctx: dict[str, Any]) -> Catalog:
        return self._catalog if self._catalog is not None else get_catalog(ctx)

    def _items(self, text: str, catalog: Catalog) -> list[dict[str, Any]]:
        if _LIST_RE.search(text):
            return [item for item in catalog.all() if item.get("kind") == "tariff"]
        if not _INTENT_RE.search(text):
            return []
        return self._select(text, catalog)

    def _select(self, text: str, catalog: Catalog) -> list[dict[str, Any]]:
        """Найти упомянутые позиции и отбросить догадки.

        Правила:
        * позиция должна быть названа именем или синонимом;
        * у совпадения должно быть «редкое» слово (не общий ``тариф``);
        * позиция, чьи слова — подмножество слов другой, отбрасывается
          (``доставка`` не тянет все зоны, если названа Москва);
        * в запросе не должно остаться незнакомых слов (``Марс``).
        """
        query = tokens(text)
        if not query:
            return []
        items = catalog.all()

        # Частота слов в именах/синонимах: «тариф» есть у трёх позиций,
        # «старт» — только у одной.
        document_frequency: dict[str, int] = {}
        for item in items:
            unique: set[str] = set()
            for alias in _aliases(item):
                unique.update(tokens(alias))
            for word in unique:
                key = stem(word)
                document_frequency[key] = document_frequency.get(key, 0) + 1

        mentioned: list[tuple[dict[str, Any], set[str]]] = []
        for item in items:
            aliases = _aliases(item)
            if not any(alias_in_tokens(query, alias) for alias in aliases):
                continue
            alias_tokens = {word for alias in aliases for word in tokens(alias)}
            matched = {
                word
                for word in query
                if any(token_match(word, known) for known in alias_tokens)
            }
            mentioned.append((item, matched))

        if not mentioned:
            return []

        candidates = [
            (item, matched)
            for item, matched in mentioned
            if any(document_frequency.get(stem(word), 0) <= 2 for word in matched)
        ]
        if not candidates:
            # Запрос вида «сколько стоит доставка?» или «сколько стоит тариф?»:
            # назван только общий класс — честно показываем все его позиции.
            return self._fallback(query, mentioned)

        selected = [
            (item, matched)
            for item, matched in candidates
            if not any(matched < other for other_item, other in candidates if other_item is not item)
        ]
        selected = self._cover(query, selected)
        return [item for item, _ in selected]

    def _fallback(
        self,
        query: list[str],
        mentioned: list[tuple[dict[str, Any], set[str]]],
    ) -> list[dict[str, Any]]:
        covered = set().union(*(matched for _, matched in mentioned))
        unknown = [word for word in query if word not in covered and not _is_generic(word)]
        if unknown:
            return []
        return [item for item, _ in mentioned]

    def _cover(
        self,
        query: list[str],
        selected: list[tuple[dict[str, Any], set[str]]],
    ) -> list[tuple[dict[str, Any], set[str]]]:
        if not selected:
            return []
        covered = set().union(*(matched for _, matched in selected))
        unknown = [word for word in query if word not in covered and not _is_generic(word)]
        return selected if not unknown else []

    # ------------------------------------------------------------------ #
    # сборка ответа
    # ------------------------------------------------------------------ #
    def _answer(self, items: list[dict[str, Any]], text: str) -> SkillResult:
        sources: list[str] = []
        for item in items:
            for source in item.get("sources") or []:
                if source not in sources:
                    sources.append(source)

        if len(items) == 1:
            text_out = self._single_text(items[0])
        else:
            text_out = self._multi_text(items, text)

        return make_result(
            self.name,
            text_out,
            sources=sources,
            catalog_ids=[str(item.get("id") or "") for item in items],
        )

    def _single_text(self, item: dict[str, Any]) -> str:
        source = (item.get("sources") or [""])[0]
        parts = [f"{_label(item)} «{item['name']}» — {_format_price(item)} {source}."]
        limits = item.get("limits") or []
        notes = item.get("notes") or []
        if limits:
            parts.append("Лимиты: " + "; ".join(str(value) for value in limits) + ".")
        if notes:
            parts.append("Заметки: " + "; ".join(str(value) for value in notes) + ".")
        return " ".join(parts)

    def _multi_text(self, items: list[dict[str, Any]], text: str) -> str:
        detail = bool(re.search(r"(?i)(?:что\s+входит|включает|лимит|api|скидк|срок)", text))
        lines = ["В каталоге:"]
        for item in items:
            source = (item.get("sources") or [""])[0]
            line = f"— {_label(item)} «{item['name']}» — {_format_price(item)} {source}"
            limits = item.get("limits") or []
            if limits:
                line += f"; лимиты: {'; '.join(str(value) for value in limits)}"
            notes = item.get("notes") or []
            if detail and notes:
                line += f"; заметки: {'; '.join(str(value) for value in notes)}"
            lines.append(line)
        return "\n".join(lines)
