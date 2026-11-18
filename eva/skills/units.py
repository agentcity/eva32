"""Скилл ``units``: конвертация единиц без сети и без модели.

Понимает длину, массу, объём, объём данных, длительность и температуру:

    Переведи 10 км в мили
    5 кг в фунтах
    2 гигабайта в мегабайты
    30 градусов Цельсия в Фаренгейты
    1,5 км в метры

Правила осторожности (скилл молчит, если не уверен):

* нужны **две** единицы, число у исходной и связка «в/во/это/сколько» между
  ними — иначе вопрос уходит модели;
* размерности обязаны совпадать: «10 км в час» — это скорость, а не
  конвертация, такой вопрос модель разберёт лучше;
* неизвестная единица («10 паросов в мили») — молчание;
* каталожные вопросы (цена, доставка, возврат) не перехватываются.

Ответ не из каталога, поэтому ``sources`` пустой.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from eva.catalog import normalize
from eva.skills.base import Skill, SkillResult, format_number, make_result

__all__ = ["UnitsSkill"]


@dataclass(frozen=True)
class _Unit:
    """Единица: размерность, множитель к базовой и формы для ответа."""

    dimension: str
    factor: float
    one: str
    few: str
    many: str


#: Ключ → единица. Множитель — сколько базовых единиц в одной такой.
#: База: метр, грамм, литр, байт, секунда; температура считается отдельно.
_UNITS: dict[str, _Unit] = {
    # длина
    "mm": _Unit("length", 0.001, "мм", "мм", "мм"),
    "cm": _Unit("length", 0.01, "см", "см", "см"),
    "m": _Unit("length", 1.0, "м", "м", "м"),
    "km": _Unit("length", 1000.0, "км", "км", "км"),
    "inch": _Unit("length", 0.0254, "дюйм", "дюйма", "дюймов"),
    "foot": _Unit("length", 0.3048, "фут", "фута", "футов"),
    "yard": _Unit("length", 0.9144, "ярд", "ярда", "ярдов"),
    "mile": _Unit("length", 1609.344, "миля", "мили", "миль"),
    # масса
    "mg": _Unit("mass", 0.001, "мг", "мг", "мг"),
    "g": _Unit("mass", 1.0, "г", "г", "г"),
    "kg": _Unit("mass", 1000.0, "кг", "кг", "кг"),
    "ton": _Unit("mass", 1_000_000.0, "т", "т", "т"),
    "pound": _Unit("mass", 453.59237, "фунт", "фунта", "фунтов"),
    "ounce": _Unit("mass", 28.349523125, "унция", "унции", "унций"),
    # объём
    "ml": _Unit("volume", 0.001, "мл", "мл", "мл"),
    "l": _Unit("volume", 1.0, "л", "л", "л"),
    "gallon": _Unit("volume", 3.785411784, "галлон", "галлона", "галлонов"),
    "pint": _Unit("volume", 0.473176473, "пинта", "пинты", "пинт"),
    # данные (1 КБ = 1024 байта — как в памяти, а не у производителя дисков)
    "byte": _Unit("data", 1.0, "байт", "байта", "байтов"),
    "kb": _Unit("data", 1024.0, "КБ", "КБ", "КБ"),
    "mb": _Unit("data", 1024.0**2, "МБ", "МБ", "МБ"),
    "gb": _Unit("data", 1024.0**3, "ГБ", "ГБ", "ГБ"),
    "tb": _Unit("data", 1024.0**4, "ТБ", "ТБ", "ТБ"),
    # длительность
    "sec": _Unit("duration", 1.0, "сек", "сек", "сек"),
    "min": _Unit("duration", 60.0, "мин", "мин", "мин"),
    "hour": _Unit("duration", 3600.0, "ч", "ч", "ч"),
    "day": _Unit("duration", 86400.0, "сут", "сут", "сут"),
    "week": _Unit("duration", 604800.0, "нед", "нед", "нед"),
    # температура: множитель не используется, конверсия отдельная
    "celsius": _Unit("temperature", 1.0, "°C", "°C", "°C"),
    "fahrenheit": _Unit("temperature", 1.0, "°F", "°F", "°F"),
    "kelvin": _Unit("temperature", 1.0, "K", "K", "K"),
}

#: Формы слов, которые считаются единицей (после normalize: нижний регистр, ё→е).
_FORM_TABLE: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("mm", ("мм", "миллиметр", "миллиметра", "миллиметров", "миллиметры",
            "миллиметре", "миллиметрам", "миллиметрах", "миллиметром")),
    ("cm", ("см", "сантиметр", "сантиметра", "сантиметров", "сантиметры",
            "сантиметре", "сантиметрам", "сантиметрах", "сантиметром")),
    ("m", ("м", "метр", "метра", "метров", "метры", "метре", "метрам",
           "метрах", "метром")),
    ("km", ("км", "километр", "километра", "километров", "километры",
            "километре", "километрам", "километрах", "километром")),
    ("inch", ("дюйм", "дюйма", "дюймов", "дюймы", "дюйме", "дюймах", "дюймом")),
    ("foot", ("фут", "фута", "футов", "футы", "футе", "футах", "футом")),
    ("yard", ("ярд", "ярда", "ярдов", "ярды", "ярде", "ярдах", "ярдом")),
    ("mile", ("миля", "мили", "милю", "миль", "миле", "милях", "милям",
              "милями", "милей")),
    ("mg", ("мг", "миллиграмм", "миллиграмма", "миллиграммов", "миллиграммы",
            "миллиграмме", "миллиграммах")),
    ("g", ("г", "грамм", "грамма", "граммов", "граммы", "грамме", "граммах",
           "граммом")),
    ("kg", ("кг", "килограмм", "килограмма", "килограммов", "килограммы",
            "килограмме", "килограммах", "килограммом")),
    ("ton", ("тонна", "тонны", "тонн", "тонну", "тонне", "тоннах", "тонной",
             "тоннами")),
    ("pound", ("фунт", "фунта", "фунтов", "фунты", "фунте", "фунтах", "фунтом")),
    ("ounce", ("унция", "унции", "унцию", "унций", "унциях", "унцией")),
    ("ml", ("мл", "миллилитр", "миллилитра", "миллилитров", "миллилитры",
            "миллилитре", "миллилитрах")),
    ("l", ("л", "литр", "литра", "литров", "литры", "литре", "литрам",
           "литрах", "литром")),
    ("gallon", ("галлон", "галлона", "галлонов", "галлоны", "галлоне",
                "галлонах", "галлоном")),
    ("pint", ("пинта", "пинты", "пинт", "пинту", "пинте", "пинтах", "пинтой")),
    ("byte", ("байт", "байта", "байтов", "байты", "байте", "байтах", "байтом")),
    ("kb", ("кб", "килобайт", "килобайта", "килобайтов", "килобайты",
            "килобайте", "килобайтах")),
    ("mb", ("мб", "мегабайт", "мегабайта", "мегабайтов", "мегабайты",
            "мегабайте", "мегабайтах")),
    ("gb", ("гб", "гигабайт", "гигабайта", "гигабайтов", "гигабайты",
            "гигабайте", "гигабайтах")),
    ("tb", ("тб", "терабайт", "терабайта", "терабайтов", "терабайты",
            "терабайте", "терабайтах")),
    ("sec", ("сек", "секунда", "секунды", "секунд", "секунду", "секунде",
             "секундах", "секундам", "секундами", "секундой", "с")),
    ("min", ("мин", "минута", "минуты", "минут", "минуту", "минуте",
             "минутах", "минутам", "минутами", "минутой")),
    ("hour", ("ч", "час", "часа", "часов", "часы", "часу", "часе", "часах",
              "часам", "часами", "часом")),
    ("day", ("сутки", "суток", "суткам", "сутках", "сутками")),
    ("week", ("неделя", "недели", "неделю", "недель", "неделе", "неделях",
              "неделям", "неделями", "неделей")),
    ("celsius", ("цельсия", "цельсию", "цельсии", "цельсий", "цельсием",
                 "celsius")),
    ("fahrenheit", ("фаренгейт", "фаренгейта", "фаренгейту", "фаренгейте",
                    "фаренгейты", "фаренгейтов", "фаренгейтом", "fahrenheit")),
    ("kelvin", ("кельвин", "кельвина", "кельвинов", "кельвины", "кельвине",
                "кельвинах", "kelvin")),
)

_FORMS: dict[str, str] = {
    form: key for key, forms in _FORM_TABLE for form in forms
}

_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
_TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)?|[a-zа-я]+")
_DEGREE_RE = re.compile(r"градус\w*")
_SEPARATOR_RE = re.compile(
    r"(?<![a-zа-я])(?:в|во|это|сколько)(?![a-zа-я])|->|→|=>|=|≥"
)

#: Каталожные вопросы — не наша зона: там отвечает скилл цены или модель.
_GUARD_RE = re.compile(
    r"(?:тариф\w*|каталог\w*|доставк\w*|возврат\w*|подписк\w*|скидк\w*"
    r"|стоимост\w*|цен[аыуе]\w*|стоит|руб\w*|₽)",
    re.IGNORECASE,
)

#: Аффинные конверсии температуры.
_TEMPERATURE = {
    ("celsius", "fahrenheit"): lambda value: value * 9 / 5 + 32,
    ("fahrenheit", "celsius"): lambda value: (value - 32) * 5 / 9,
    ("celsius", "kelvin"): lambda value: value + 273.15,
    ("kelvin", "celsius"): lambda value: value - 273.15,
    ("fahrenheit", "kelvin"): lambda value: (value - 32) * 5 / 9 + 273.15,
    ("kelvin", "fahrenheit"): lambda value: (value - 273.15) * 9 / 5 + 32,
}


class UnitsSkill(Skill):
    """Конвертирует единицы, если уверен в исходной и целевой единице."""

    name = "units"

    def match(self, text: str, ctx: dict[str, Any] | None = None) -> bool:
        return _parse(text) is not None

    def run(self, text: str, ctx: dict[str, Any]) -> SkillResult | None:
        parsed = _parse(text)
        if parsed is None:
            return None
        value, source, target = parsed
        converted = _convert(value, source, target)
        if converted is None:  # pragma: no cover — _parse проверяет размерность
            return None
        return make_result(self.name, _answer_text(value, source, target, converted), sources=[])


# --------------------------------------------------------------------------- #
# разбор запроса
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Mention:
    """Упоминание единицы: число (если есть) и границы в тексте."""

    value: float | None
    key: str
    start: int
    end: int


def _prepare(text: str) -> str:
    value = normalize(text)
    value = value.replace("°c", " celsius ").replace("°f", " fahrenheit ")
    return value.replace("°", " ")


def _tokenize(value: str) -> list[tuple[int, int, str]]:
    return [(match.start(), match.end(), match.group(0)) for match in _TOKEN_RE.finditer(value)]


def _merge_numbers(value: str, tokens: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    """«1 200 км» → одно число 1200: пробел — разделитель тысяч.

    Сливаем только когда следующий токен — ровно три цифры и между ними
    один пробел: так «5 100 кг» не станет пятью килограммами и сотней.
    """
    merged: list[tuple[int, int, str]] = []
    index = 0
    while index < len(tokens):
        start, end, token = tokens[index]
        if _NUMBER_RE.fullmatch(token):
            while (
                index + 1 < len(tokens)
                and _NUMBER_RE.fullmatch(tokens[index + 1][2])
                and re.fullmatch(r"\d{3}", tokens[index + 1][2])
                and value[end : tokens[index + 1][0]] == " "
            ):
                index += 1
                token += tokens[index][2]
                end = tokens[index][1]
        merged.append((start, end, token))
        index += 1
    return merged


def _unit_after(
    value: str,
    tokens: list[tuple[int, int, str]],
    index: int,
) -> tuple[str, int, int] | None:
    """Единица сразу после числа: «км» или «градусов цельсия»."""
    if index >= len(tokens):
        return None
    start, end, token = tokens[index]
    key = _FORMS.get(token)
    if key is not None:
        return key, index, end
    if _DEGREE_RE.fullmatch(token) and index + 1 < len(tokens):
        next_start, next_end, next_token = tokens[index + 1]
        if value[end:next_start].strip():
            return None
        scale = _FORMS.get(next_token)
        if scale is not None and _UNITS[scale].dimension == "temperature":
            return scale, index + 1, next_end
    return None


def _signed(value: str, tokens: list[tuple[int, int, str]], index: int) -> float:
    """Знак числа: «-5», «−5» и «минус 5» дают -5."""
    number = float(tokens[index][2].replace(",", "."))
    start = tokens[index][0]
    if start > 0 and value[start - 1] == "-":
        before = value[start - 2] if start >= 2 else " "
        if not before.isalnum():
            return -number
    if index > 0 and tokens[index - 1][2] == "минус":
        return -number
    return number


def _mentions(value: str, tokens: list[tuple[int, int, str]]) -> list[_Mention]:
    found: list[_Mention] = []
    used: set[int] = set()
    for index, (start, _end, token) in enumerate(tokens):
        if not _NUMBER_RE.fullmatch(token):
            continue
        unit = _unit_after(value, tokens, index + 1)
        if unit is None:
            continue
        key, unit_index, unit_end = unit
        for consumed in range(index + 1, unit_index + 1):
            used.add(consumed)
        found.append(_Mention(_signed(value, tokens, index), key, start, unit_end))
    for index, (start, end, token) in enumerate(tokens):
        if index in used:
            continue
        key = _FORMS.get(token)
        if key is not None:
            found.append(_Mention(None, key, start, end))
    found.sort(key=lambda item: item.start)
    return found


def _parse(text: str) -> tuple[float, str, str] | None:
    """Найти пару «число + единица → единица» или вернуть ``None``."""
    if not text or not text.strip():
        return None
    value = _prepare(text)
    if _GUARD_RE.search(value):
        return None
    tokens = _merge_numbers(value, _tokenize(value))
    if not tokens:
        return None

    mentioned = _mentions(value, tokens)
    for left, right in zip(mentioned, mentioned[1:]):
        if (left.value is None) == (right.value is None):
            continue  # нужна ровно одна сторона с числом
        source, target = (left, right) if left.value is not None else (right, left)
        if source.value is None:  # pragma: no cover — проверено выше
            continue
        # Связка всегда между упоминаниями, независимо от того, какое из них
        # исходное: «10 км в мили» и «сколько миль в 10 км» — один и тот же вопрос.
        gap = value[left.end : right.start]
        if not _SEPARATOR_RE.search(gap):
            continue
        if source.key == target.key:
            continue
        if _UNITS[source.key].dimension != _UNITS[target.key].dimension:
            continue
        return source.value, source.key, target.key
    return None


# --------------------------------------------------------------------------- #
# конверсия и ответ
# --------------------------------------------------------------------------- #
def _convert(value: float, source: str, target: str) -> float | None:
    if _UNITS[source].dimension != _UNITS[target].dimension:
        return None
    if _UNITS[source].dimension == "temperature":
        converter = _TEMPERATURE.get((source, target))
        return converter(value) if converter else None
    return value * _UNITS[source].factor / _UNITS[target].factor


def _decimals(number: float) -> int:
    magnitude = abs(number)
    if magnitude >= 100:
        return 2
    if magnitude >= 1:
        return 3
    if magnitude >= 0.001:
        return 4
    return 6


def _plural(value: float, unit: _Unit) -> str:
    number = abs(value)
    if abs(number - round(number)) > 1e-9:
        return unit.few
    count = int(round(number)) % 100
    if 11 <= count <= 14:
        return unit.many
    last = count % 10
    if last == 1:
        return unit.one
    if 2 <= last <= 4:
        return unit.few
    return unit.many


def _answer_text(value: float, source: str, target: str, converted: float) -> str:
    source_unit = _UNITS[source]
    target_unit = _UNITS[target]
    source_places = _decimals(value)
    target_places = _decimals(converted)
    shown_source = round(value, source_places)
    shown_target = round(converted, target_places)
    left = f"{format_number(value, source_places)} {_plural(shown_source, source_unit)}"
    right = f"{format_number(converted, target_places)} {_plural(shown_target, target_unit)}"
    return f"{left} = {right}."
