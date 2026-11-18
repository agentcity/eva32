"""Скилл ``currency``: курс валют по встроенному справочнику, без сети.

Курс — **статический срез в коде** (``RATES_RUB`` + ``RATES_DATE``), а не
онлайн-котировка: скиллам запрещена сеть, а ответ без модели должен быть
воспроизводим. Поэтому каждый ответ называет дату среза и прямо говорит, что
курс не онлайн — иначе пользователь решит, что видит биржу.

Умеет два вопроса:

* курс: «Какой курс доллара?», «Курс евро к рублю?»;
* конвертация: «Сколько рублей в 100 долларах?», «Переведи 50 евро в рубли»,
  «100 долларов в евро» (кросс-курс через рубль).

Правила осторожности: нужна иностранная валюта (вопросы только про рубли не
наши), каталожные вопросы не перехватываются, при неясной цели конвертации
скилл молчит. ``sources`` пустой — справочник не позиция каталога.
"""

from __future__ import annotations

import re
from typing import Any

from eva.catalog import normalize
from eva.skills.base import Skill, SkillResult, format_number, make_result

__all__ = ["CurrencySkill", "RATES_DATE", "RATES_RUB"]

#: Дата среза справочника. Меняется вручную вместе с ``RATES_RUB``.
RATES_DATE = "2026-11-18"

#: Сколько рублей стоит одна единица валюты. Демонстрационный срез; для боевого
#: курса значения обновляются вручную (сеть в скиллах запрещена).
RATES_RUB: dict[str, float] = {
    "USD": 92.50,
    "EUR": 100.40,
    "CNY": 12.80,
    "GBP": 116.70,
    "KZT": 0.19,
    "JPY": 0.60,
    "BYN": 28.20,
    # Техническая запись рубля: 1 ₽ = 1 ₽. Нужна кросс-курсу «USD → RUB»,
    # чтобы не заводить отдельную ветку на каждую цель.
    "RUB": 1.0,
}

#: Как называть валюту в ответе.
_DISPLAY = {
    "RUB": "₽",
    "USD": "USD",
    "EUR": "EUR",
    "CNY": "CNY",
    "GBP": "GBP",
    "KZT": "KZT",
    "JPY": "JPY",
    "BYN": "BYN",
}

#: Упоминания валюты: (регулярка, код). Рубль — такая же валюта, но вопрос
#: только про рубли не наш: отвечать «100 ₽ = 100 ₽» бессмысленно.
_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"(?:доллар\w*|долл\b|бакс\w*|\busd\b|\$)", "USD"),
    (r"(?:евро|\beur\b|€)", "EUR"),
    (r"(?:юан\w*|\bcny\b|женьминьби)", "CNY"),
    (r"(?:фунт\w*\s+стерлинг\w*|\bgbp\b|£)", "GBP"),
    (r"(?:тенге|\bkzt\b)", "KZT"),
    (r"(?:йен\w*|иен\w*|\bjpy\b)", "JPY"),
    (r"(?:белорусск\w*\s+рубл\w*|\bbyn\b)", "BYN"),
    (r"(?:рубл\w*|\brub\b|₽)", "RUB"),
)

#: Вопрос про курс, а не про конвертацию конкретной суммы.
_RATE_RE = re.compile(
    r"(?:"
    r"курс\w*|котировк\w*"
    r"|сколько\s+(?:стоит|стоят|дают|дадут|будет|будет\s+в)"
    r"|по\s+ч[её]м"
    r"|rub\b"
    r")",
    re.IGNORECASE,
)

#: Связка между суммой и целевой валютой: «100 долларов **в** евро».
_SEPARATOR_RE = re.compile(
    r"(?<![a-zа-я])(?:в|во|на|это|сколько)(?![a-zа-я])|->|→|=>|="
)

#: Каталожные вопросы — не наша зона.
_GUARD_RE = re.compile(
    r"(?:"
    r"тариф\w*|каталог\w*|доставк\w*|возврат\w*|подписк\w*|скидк\w*"
    r"|офис\w*|поддержк\w*|техподдержк\w*"
    r")",
    re.IGNORECASE,
)

_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")


class CurrencySkill(Skill):
    """Курс и конвертация по встроенному справочнику (без сети)."""

    name = "currency"

    def match(self, text: str, ctx: dict[str, Any] | None = None) -> bool:
        if not text or not text.strip():
            return False
        value = normalize(text)
        if _GUARD_RE.search(value):
            return False
        mentions = _mentions(value)
        if not any(code != "RUB" for _start, _end, code in mentions):
            return False
        if _RATE_RE.search(value):
            return True
        return _conversion(value, mentions) is not None

    def run(self, text: str, ctx: dict[str, Any]) -> SkillResult | None:
        if not self.match(text, ctx):
            return None
        value = normalize(text)
        mentions = _mentions(value)
        conversion = _conversion(value, mentions)
        if conversion is not None:
            amount, source, target = conversion
            result = amount * RATES_RUB[source] / RATES_RUB[target]
            return make_result(
                self.name,
                _conversion_text(amount, source, target, result),
                sources=[],
            )
        return make_result(self.name, _rate_text(mentions), sources=[])


# --------------------------------------------------------------------------- #
# разбор
# --------------------------------------------------------------------------- #
def _mentions(value: str) -> list[tuple[int, int, str]]:
    """Упоминания валют: без пересечений, более длинное совпадение важнее."""
    found: list[tuple[int, int, str]] = []
    for pattern, code in _PATTERNS:
        for match in re.finditer(pattern, value):
            found.append((match.start(), match.end(), code))
    found.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    result: list[tuple[int, int, str]] = []
    covered_until = -1
    for start, end, code in found:
        if start >= covered_until:
            result.append((start, end, code))
            covered_until = end
    return result


def _amount_near(value: str, start: int, end: int) -> float | None:
    """Сумма рядом с валютой: «100 долларов», «$100», «100$»."""
    before = re.search(r"(\d+(?:[.,]\d+)?)\s*$", value[:start])
    if before:
        return float(before.group(1).replace(",", "."))
    after = re.match(r"\s*(\d+(?:[.,]\d+)?)", value[end:])
    if after:
        return float(after.group(1).replace(",", "."))
    return None


def _conversion(
    value: str,
    mentions: list[tuple[int, int, str]],
) -> tuple[float, str, str] | None:
    """Найти сумму, исходную валюту и цель или вернуть ``None``."""
    for index, (start, end, code) in enumerate(mentions):
        if code == "RUB":
            continue
        amount = _amount_near(value, start, end)
        if amount is None:
            continue
        target = _target(value, mentions, index, code)
        if target is None or target == code:
            continue
        return amount, code, target
    return None


def _target(
    value: str,
    mentions: list[tuple[int, int, str]],
    index: int,
    source: str,
) -> str | None:
    """Куда конвертируем: другая валюта после суммы, иначе рубли."""
    source_end = mentions[index][1]
    for other_start, _other_end, code in mentions[index + 1 :]:
        if code == source:
            continue
        gap = value[source_end:other_start]
        if _SEPARATOR_RE.search(gap):
            return code
        if code == "RUB" and "рубл" in gap:
            return code
    if re.search(r"рубл\w*", value[source_end:]):
        return "RUB"
    if re.search(r"(?:сколько|переведи|перевести|конверт\w*|выраз\w*)", value):
        return "RUB"
    return None


# --------------------------------------------------------------------------- #
# ответ
# --------------------------------------------------------------------------- #
def _rate_text(mentions: list[tuple[int, int, str]]) -> str:
    codes: list[str] = []
    for _start, _end, code in mentions:
        if code != "RUB" and code in RATES_RUB and code not in codes:
            codes.append(code)
    lines = [f"1 {code} = {_money(RATES_RUB[code])} ₽" for code in codes]
    return (
        f"Курс по встроенному справочнику на {_rates_date()}: "
        + "; ".join(lines)
        + "."
    )


def _conversion_text(amount: float, source: str, target: str, result: float) -> str:
    left = f"{format_number(amount, 0 if float(amount).is_integer() else 2)} {source}"
    right = f"{format_number(result, _result_places(result))} {_DISPLAY[target]}"
    return (
        f"{left} = {right}. Справочник на {_rates_date()}, не онлайн-курс."
    )


def _rates_date() -> str:
    """Дата среза в виде ДД.ММ.ГГГГ — так её читают в ответе."""
    year, month, day = RATES_DATE.split("-")
    return f"{day}.{month}.{year}"


def _money(rate: float) -> str:
    """Деньги в курсе: всегда два знака после запятой («92,50»)."""
    return f"{rate:.2f}".replace(".", ",")


def _result_places(number: float) -> int:
    magnitude = abs(number)
    if magnitude >= 1:
        return 2
    if magnitude >= 0.01:
        return 4
    return 6
