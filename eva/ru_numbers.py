"""Числа → русские слова для озвучки (задача 0205).

Silero v4_ru **не произносит цифры**: токен-число выпадает из речи, поэтому
«Цена 1500 рублей» звучала втрое короче словесной формы. Нормализация должна
происходить до выбора движка, чтобы слова получали все адаптеры
(Silero/ElevenLabs/``say``), а не только один.

Публичная функция — :func:`numbers_to_words`. Она превращает в слова целые
(0…10¹²), дробные, проценты/рубли/номера, диапазоны и даты, но **не трогает**
числа внутри латиницы и кода (``v2``, ``GPT-4``, ``report_2026.pdf``), URL и
код в бэктиках: такие токены подменяются заглушками и возвращаются как были.
Пробелы и пунктуация вокруг числа сохраняются.

Примеры::

    >>> numbers_to_words("Цена 1500 рублей")
    'Цена одна тысяча пятьсот рублей'
    >>> numbers_to_words("Скидка 20%")
    'Скидка двадцать процентов'
    >>> numbers_to_words("01.02.2026")
    'первое февраля две тысячи двадцать шесть'

Грамматика — именительный падеж; для диапазонов («от десяти до двадцати»)
концы ставятся в родительный. Склонение по падежам предложения не делается:
это нормализация для речи, а не морфологический синтез.
"""

from __future__ import annotations

import re

__all__ = ["cardinal_words", "numbers_to_words"]

#: Верхняя граница нормализации: 10¹² («триллион»). Больше — оставляем
#: цифрами: врать в разрядах хуже, чем промолчать.
MAX_NUMBER = 10 ** 12

# --------------------------------------------------------------------------- #
# словари числительных
# --------------------------------------------------------------------------- #
_UNITS_NOM = {
    "m": (
        "ноль",
        "один",
        "два",
        "три",
        "четыре",
        "пять",
        "шесть",
        "семь",
        "восемь",
        "девять",
    ),
    "f": (
        "ноль",
        "одна",
        "две",
        "три",
        "четыре",
        "пять",
        "шесть",
        "семь",
        "восемь",
        "девять",
    ),
}
_UNITS_GEN = {
    "m": (
        "нуля",
        "одного",
        "двух",
        "трёх",
        "четырёх",
        "пяти",
        "шести",
        "семи",
        "восьми",
        "девяти",
    ),
    "f": (
        "нуля",
        "одной",
        "двух",
        "трёх",
        "четырёх",
        "пяти",
        "шести",
        "семи",
        "восьми",
        "девяти",
    ),
}
_TEENS_NOM = (
    "десять",
    "одиннадцать",
    "двенадцать",
    "тринадцать",
    "четырнадцать",
    "пятнадцать",
    "шестнадцать",
    "семнадцать",
    "восемнадцать",
    "девятнадцать",
)
_TEENS_GEN = (
    "десяти",
    "одиннадцати",
    "двенадцати",
    "тринадцати",
    "четырнадцати",
    "пятнадцати",
    "шестнадцати",
    "семнадцати",
    "восемнадцати",
    "девятнадцати",
)
_TENS_NOM = (
    "",
    "",
    "двадцать",
    "тридцать",
    "сорок",
    "пятьдесят",
    "шестьдесят",
    "семьдесят",
    "восемьдесят",
    "девяносто",
)
_TENS_GEN = (
    "",
    "",
    "двадцати",
    "тридцати",
    "сорока",
    "пятидесяти",
    "шестидесяти",
    "семидесяти",
    "восьмидесяти",
    "девяноста",
)
_HUNDREDS_NOM = (
    "",
    "сто",
    "двести",
    "триста",
    "четыреста",
    "пятьсот",
    "шестьсот",
    "семьсот",
    "восемьсот",
    "девятьсот",
)
_HUNDREDS_GEN = (
    "",
    "ста",
    "двухсот",
    "трёхсот",
    "четырёхсот",
    "пятисот",
    "шестисот",
    "семисот",
    "восьмисот",
    "девятисот",
)

#: Группы разрядов: (именительный, родительный, род числительного).
#: Индекс 0 — тысячи (женский род: «одна тысяча», «две тысячи»).
_GROUPS = (
    (("тысяча", "тысячи", "тысяч"), ("тысячи", "тысяч", "тысяч"), "f"),
    (
        ("миллион", "миллиона", "миллионов"),
        ("миллиона", "миллионов", "миллионов"),
        "m",
    ),
    (
        ("миллиард", "миллиарда", "миллиардов"),
        ("миллиарда", "миллиардов", "миллиардов"),
        "m",
    ),
    (
        ("триллион", "триллиона", "триллионов"),
        ("триллиона", "триллионов", "триллионов"),
        "m",
    ),
)

_MONTHS_GEN = (
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
)
_MONTHS_NOM = (
    "январь",
    "февраль",
    "март",
    "апрель",
    "май",
    "июнь",
    "июль",
    "август",
    "сентябрь",
    "октябрь",
    "ноябрь",
    "декабрь",
)

#: Порядковые числительные для дня месяца — средний род.
_DAY_ORDINALS = {
    1: "первое",
    2: "второе",
    3: "третье",
    4: "четвёртое",
    5: "пятое",
    6: "шестое",
    7: "седьмое",
    8: "восьмое",
    9: "девятое",
    10: "десятое",
    11: "одиннадцатое",
    12: "двенадцатое",
    13: "тринадцатое",
    14: "четырнадцатое",
    15: "пятнадцатое",
    16: "шестнадцатое",
    17: "семнадцатое",
    18: "восемнадцатое",
    19: "девятнадцатое",
    20: "двадцатое",
    30: "тридцатое",
}

#: Слова единиц после числа знаков: процент, рубль.
_PERCENT_FORMS = ("процент", "процента", "процентов")
_RUBLE_FORMS = ("рубль", "рубля", "рублей")


def _plural_index(value: int) -> int:
    """Индекс формы для числа: 1 → 0, 2–4 → 1, 0/5–20/… → 2 (11–14 → 2)."""
    if 11 <= value % 100 <= 14:
        return 2
    last = value % 10
    if last == 1:
        return 0
    if 2 <= last <= 4:
        return 1
    return 2


def _under_1000(value: int, gender: str, case: str) -> str:
    """Число 0…999 словами в нужном роде и падеже (им./род.)."""
    nominative = case == "nom"
    words: list[str] = []
    hundreds, rest = divmod(value, 100)
    if hundreds:
        words.append((_HUNDREDS_NOM if nominative else _HUNDREDS_GEN)[hundreds])
    if rest >= 20:
        tens, rest = divmod(rest, 10)
        words.append((_TENS_NOM if nominative else _TENS_GEN)[tens])
    if rest >= 10:
        words.append((_TEENS_NOM if nominative else _TEENS_GEN)[rest - 10])
    elif rest:
        units = _UNITS_NOM[gender] if nominative else _UNITS_GEN[gender]
        words.append(units[rest])
    return " ".join(words)


def cardinal_words(value: int, *, gender: str = "m", case: str = "nom") -> str:
    """Целое 0…10¹² словами.

    ``gender`` — род единиц (``"f"`` для «одна целая»), ``case`` — ``"nom"``
    (по умолчанию) или ``"gen"`` для концов диапазонов («от десяти»).
    """
    if gender not in ("m", "f"):
        raise ValueError(f"неизвестный род {gender!r}: ожидалось 'm' или 'f'")
    if case not in ("nom", "gen"):
        raise ValueError(f"неизвестный падеж {case!r}: ожидалось 'nom' или 'gen'")
    number = int(value)
    if number < 0 or number > MAX_NUMBER:
        raise ValueError(
            f"число {number} вне диапазона 0…{MAX_NUMBER} "
            "(больше не озвучиваем, чтобы не соврать в разрядах)"
        )
    if number == 0:
        return "ноль"

    parts: list[str] = []
    for index in range(len(_GROUPS), 0, -1):
        chunk = (number // (1000 ** index)) % 1000
        if not chunk:
            continue
        forms_nom, forms_gen, group_gender = _GROUPS[index - 1]
        forms = forms_nom if case == "nom" else forms_gen
        if chunk == 1:
            # «тысяча пятьсот», «миллион», а не «одна тысяча пятьсот».
            parts.append(forms[0])
        else:
            parts.append(_under_1000(chunk, group_gender, case))
            parts.append(forms[_plural_index(chunk)])
    ones = number % 1000
    if ones:
        parts.append(_under_1000(ones, gender, case))
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# защита кода и латиницы
# --------------------------------------------------------------------------- #
#: Заглушки — символы из Private Use Area: они не буквы и не цифры, поэтому
#: ни одно правило нормализации их не задевает. Индекс кодируется отдельным
#: символом, чтобы соседние заглушки не склеивались.
_PUA_START = "\ue000"
_PUA_END = "\ue001"
_PUA_BASE = 0xE100
_PUA_LIMIT = 0xE8FF
_PLACEHOLDER_RE = re.compile("\ue000([\ue100-\ue8ff])\ue001")

#: Токен без пробелов, где есть и латиница, и цифра: ``v2``, ``GPT-4``,
#: ``report_2026.pdf``, ``SILERO_WORKER=0``. Числа внутри не трогаем.
_LATIN_DIGIT_RE = re.compile(
    r"(?<![\w])(?=[^\s]*[A-Za-z])(?=[^\s]*\d)[^\s]+"
)
#: URL: числа в них не читаем (и не превращаем в слова).
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
#: Код в бэктиках — как есть.
_CODE_SPAN_RE = re.compile(r"`[^`\n]*`")
#: Версии/IP: ``1.2.3``, ``192.168.1.1``. Исключение — настоящая дата.
_DOTTED_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+){2,}(?![\w.])")


class _Stash:
    """Временные заглушки для фрагментов, которые нельзя нормализовать."""

    def __init__(self) -> None:
        self._items: list[str] = []

    def __call__(self, match: re.Match[str]) -> str:
        if len(self._items) >= _PUA_LIMIT - _PUA_BASE:
            return match.group(0)  # переполнение: лучше оставить как есть
        self._items.append(match.group(0))
        return _PUA_START + chr(_PUA_BASE + len(self._items) - 1) + _PUA_END

    def restore(self, text: str) -> str:
        def replace(match: re.Match[str]) -> str:
            index = ord(match.group(1)) - _PUA_BASE
            if 0 <= index < len(self._items):
                return self._items[index]
            return match.group(0)

        return _PLACEHOLDER_RE.sub(replace, text)


# --------------------------------------------------------------------------- #
# регулярные выражения нормализации
# --------------------------------------------------------------------------- #
#: Дата ``01.02.2026`` / ``1/2/2026``. Точка в конце предложения не мешает.
_DATE_DMY_RE = re.compile(
    r"(?<![\d./])(\d{1,2})[./](\d{1,2})[./](\d{4})(?!\d)(?!\.\d)"
)
#: Дата ISO ``2026-02-01``.
_DATE_ISO_RE = re.compile(r"(?<![\d-])(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)(?!-\d)")
#: Год-месяц ``2026-02`` (без дня): «февраль две тысячи двадцать шесть».
_DATE_YM_RE = re.compile(r"(?<![\d-])(\d{4})-(\d{1,2})(?!\d)(?!-\d)")

#: Единица после числа: процент, рубль. «рублей» — уже слово, не трогаем.
_UNIT_RE = re.compile(
    r"(?<![\d.,\w])(\d{1,15})(\s*)(%|₽|р\.|руб(?:\.|(?![А-Яа-яЁё])))",
    re.IGNORECASE,
)
#: Разряды через пробел: «1 500», «2 000 000» (обычная запись цен и сумм).
_GROUPED_RE = re.compile(r"(?<![\d,.])(\d{1,3})((?:[ \u00a0\u202f]\d{3})+)(?!\d)")
#: Словами записанный диапазон: «от 5 до 10» → «от пяти до десяти».
_OT_DO_RE = re.compile(
    r"(?<![\w])(от)\s+(\d{1,15})\s+(до)\s+(\d{1,15})(?![\d\w])",
    re.IGNORECASE,
)
#: Диапазон ``10-20`` (дефис, минус, тире) с необязательной единицей.
_RANGE_RE = re.compile(
    r"(?<![\d\w.,\-–—])(\d{1,15})\s*[-–—]\s*(\d{1,15})(?![\d\w])"
    r"(?!\s*[-–—]\s*\d)"
    r"(\s*(?:%|₽|руб(?:\.|(?![А-Яа-яЁё]))))?",
    re.IGNORECASE,
)
#: Дробное ``1,5`` / ``1.5`` (до трёх знаков после запятой).
_FRACTION_RE = re.compile(r"(?<![\d.,])(\d{1,12})[.,](\d{1,3})(?![\d.,])")
#: Номер: ``№ 5`` → «номер пять».
_NUMBER_SIGN_RE = re.compile(r"№\s*(\d{1,15})")
#: Знак числа: ``+7`` → «плюс семь», ``-5`` → «минус пять».
_PLUS_RE = re.compile(r"(?<![\w\d])\+(?=\d)")
_MINUS_RE = re.compile(r"(?<![\w\d])[-–—](?=\d)")
#: Обычное целое: не часть слова, дроби, версии; точка в конце — можно.
_INT_RE = re.compile(r"(?<![\d.,\w])(\d{1,15})(?![\d\w])(?![.,]\d)")
#: Оставшиеся знаки без числа.
_SIGN_WORDS = {"₽": "рублей", "%": "процентов", "№": "номер"}
_SIGN_RE = re.compile(r"[₽%№]")
#: Отдельное «руб.» (без числа перед ним).
_RUB_DOT_RE = re.compile(r"(?<![\w])руб\.", re.IGNORECASE)


def _safe_cardinal(value: int, *, case: str = "nom", gender: str = "m") -> str | None:
    """Слова для числа или ``None``, если оно вне диапазона."""
    try:
        return cardinal_words(value, case=case, gender=gender)
    except ValueError:
        return None


def _day_ordinal(day: int) -> str:
    """День месяца порядковым числительным: 1 → «первое», 31 → «тридцать первое»."""
    known = _DAY_ORDINALS.get(day)
    if known:
        return known
    tens, unit = divmod(day, 10)
    ten_word = _TENS_NOM[tens] if tens else ""
    unit_word = _DAY_ORDINALS.get(unit, "")
    return " ".join(part for part in (ten_word, unit_word) if part)


def _date_words(day: int, month: int, year: int) -> str | None:
    """«первое февраля две тысячи двадцать шесть» или ``None`` для не-даты."""
    if not (1 <= day <= 31 and 1 <= month <= 12 and 1 <= year <= 9999):
        return None
    year_words = _safe_cardinal(year)
    if year_words is None:
        return None
    return f"{_day_ordinal(day)} {_MONTHS_GEN[month - 1]} {year_words}"


def _replace_dmy_date(match: re.Match[str]) -> str:
    day, month, year = (int(group) for group in match.groups())
    words = _date_words(day, month, year)
    return words if words is not None else match.group(0)


def _replace_iso_date(match: re.Match[str]) -> str:
    year, month, day = (int(group) for group in match.groups())
    words = _date_words(day, month, year)
    return words if words is not None else match.group(0)


def _replace_year_month(match: re.Match[str]) -> str:
    year, month = (int(group) for group in match.groups())
    if not (1 <= month <= 12):
        return match.group(0)
    year_words = _safe_cardinal(year)
    if year_words is None:
        return match.group(0)
    return f"{_MONTHS_NOM[month - 1]} {year_words}"


def _join_grouped_digits(match: re.Match[str]) -> str:
    """«1 500» → «1500»: дальше обычное правило чисел прочитает сумму."""
    tail = match.group(2)
    for space in (" ", "\u00a0", "\u202f"):
        tail = tail.replace(space, "")
    return match.group(1) + tail


def _replace_ot_do(match: re.Match[str]) -> str:
    """«от 5 до 10» → «от пяти до десяти» (родительный падеж после от/до)."""
    left = _safe_cardinal(int(match.group(2)), case="gen")
    right = _safe_cardinal(int(match.group(4)), case="gen")
    if left is None or right is None:
        return match.group(0)
    return f"{match.group(1)} {left} {match.group(3)} {right}"


def _unit_word(unit: str, value: int) -> str:
    """Слово единицы с согласованием: «21 процент», «1500 рублей»."""
    cleaned = unit.strip().lower()
    if cleaned == "%":
        return _PERCENT_FORMS[_plural_index(value)]
    # ₽, «руб.», «руб», «р.»
    return _RUBLE_FORMS[_plural_index(value)]


def _replace_unit(match: re.Match[str]) -> str:
    digits, spaces, unit = match.groups()
    number = int(digits)
    words = _safe_cardinal(number)
    if words is None:
        return match.group(0)
    return f"{words}{spaces or ' '}{_unit_word(unit, number)}"


def _replace_range(match: re.Match[str]) -> str:
    first, second, unit = match.groups()
    left = _safe_cardinal(int(first), case="gen")
    right = _safe_cardinal(int(second), case="gen")
    if left is None or right is None:
        return match.group(0)
    words = f"от {left} до {right}"
    if unit:
        # Для диапазона — множественное: «от десяти до двадцати процентов».
        words += f" {_unit_word(unit, 0)}"
    return words


def _replace_fraction(match: re.Match[str]) -> str:
    integer_part, fraction_part = match.groups()
    denominator = 10 ** len(fraction_part)
    if denominator > 1000:
        return match.group(0)  # больше тысячных не озвучиваем
    whole = _safe_cardinal(int(integer_part), gender="f")
    numerator = _safe_cardinal(int(fraction_part), gender="f")
    if whole is None or numerator is None:
        return match.group(0)
    whole_form = (
        "целая"
        if _plural_index(int(integer_part)) == 0
        else "целых"
    )
    denominator_word = {
        10: ("десятая", "десятых"),
        100: ("сотая", "сотых"),
        1000: ("тысячная", "тысячных"),
    }[denominator]
    fraction_form = (
        denominator_word[0]
        if _plural_index(int(fraction_part)) == 0
        else denominator_word[1]
    )
    return f"{whole} {whole_form} {numerator} {fraction_form}"


def _replace_integer(match: re.Match[str]) -> str:
    words = _safe_cardinal(int(match.group(1)))
    return words if words is not None else match.group(1)


def _replace_number_sign(match: re.Match[str]) -> str:
    words = _safe_cardinal(int(match.group(1)))
    if words is None:
        return match.group(0)
    return f"номер {words}"


def _replace_sign(match: re.Match[str]) -> str:
    """Оставшийся знак без числа: ₽ → рублей, % → процентов, № → номер."""
    word = _SIGN_WORDS[match.group(0)]
    string = match.string
    index = match.start()
    prefix = "" if index == 0 or string[index - 1].isspace() else " "
    tail = "" if index + 1 >= len(string) else string[index + 1]
    suffix = "" if not tail or tail.isspace() or tail in ".,;:!?)" else " "
    return f"{prefix}{word}{suffix}"


def _looks_like_date(token: str) -> bool:
    """``01.02.2026`` — дата, её нормализует отдельное правило, не версия."""
    match = _DATE_DMY_RE.fullmatch(token)
    if not match:
        return False
    day, month, year = (int(group) for group in match.groups())
    return _date_words(day, month, year) is not None


def _stash_or_keep_date(match: re.Match[str], stash: _Stash) -> str:
    """Спрятать версию/IP, но оставить настоящую дату для датного правила."""
    return match.group(0) if _looks_like_date(match.group(0)) else stash(match)


def numbers_to_words(text: str) -> str:
    """Заменить числа в ``text`` русскими словами (до выбора TTS-движка).

    Не меняет регистр слов, пробелы и пунктуацию; числа в латинице/коде/URL
    остаются как есть. Пустая строка и текст без чисел возвращаются без
    изменений. Функция идемпотентна: повторный вызов уже нормализованный
    текст не портит.
    """
    if text is None:
        return ""
    value = str(text)
    if not value:
        return value

    stash = _Stash()
    # Порядок важен: сначала прячем код/URL и латиницу с цифрами, чтобы
    # правила чисел не трогали идентификаторы.
    value = _CODE_SPAN_RE.sub(stash, value)
    value = _URL_RE.sub(stash, value)
    value = _LATIN_DIGIT_RE.sub(stash, value)
    value = _DOTTED_RE.sub(lambda match: _stash_or_keep_date(match, stash), value)

    value = _GROUPED_RE.sub(_join_grouped_digits, value)
    value = _OT_DO_RE.sub(_replace_ot_do, value)
    value = _DATE_DMY_RE.sub(_replace_dmy_date, value)
    value = _DATE_ISO_RE.sub(_replace_iso_date, value)
    value = _DATE_YM_RE.sub(_replace_year_month, value)
    value = _RANGE_RE.sub(_replace_range, value)
    value = _UNIT_RE.sub(_replace_unit, value)
    value = _NUMBER_SIGN_RE.sub(_replace_number_sign, value)
    value = _FRACTION_RE.sub(_replace_fraction, value)
    value = _PLUS_RE.sub("плюс ", value)
    value = _MINUS_RE.sub("минус ", value)
    value = _INT_RE.sub(_replace_integer, value)
    value = _SIGN_RE.sub(_replace_sign, value)
    value = _RUB_DOT_RE.sub("рублей", value)
    return stash.restore(value)
