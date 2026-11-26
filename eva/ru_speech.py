"""Символы, сокращения, телефоны и email → произносимые слова (задача 0206).

Silero v4_ru (и ``say``) молча пропускают знаки, которых нет в словаре:
«2 × 3» звучало как «два три», «Цена 50 €» — как «цена пятьдесят», а
``info@example.com`` выпадал целиком. :func:`speech_to_words` — вторая
ступень нормализации после :func:`eva.ru_numbers.numbers_to_words`: она
переводит в слова знаки и сокращения, читает телефоны по цифрам и
расшифровывает email, а числа оставляет уже знакомому нормализатору.

Порядок ступеней важен:

1. **Предправила** — то, что требует смотреть на цифры *до* их перевода:
   «50 $» → «50 долларов» (согласование по числу), «20°C» → «20 градусов
   Цельсия», «5 км²» → «5 километров в квадрате», «8 999 123 45 67» →
   «восемь девять девять…», «т.д.» → «так далее».
2. **Символы** — знаки в слова («×» → «умножить на», «≤» → «не больше»,
   «—» → запятая паузы). Здесь же снимаются кавычки/©®™, hash-теги и лишние
   пробелы (до восстановления кода из заглушек).
3. **Числа** — :func:`eva.ru_numbers.numbers_to_words`: он же отвечает за
   «%» и «₽» с правильным согласованием.

Не трогаются (как и в :mod:`eva.ru_numbers`): код в бэктиках, URL,
латиница с цифрами/знаками (``v2.0``, ``GPT-4``, ``--prefill``, ``EVA_TTS=0``,
IP и версии). Заглушки — свой диапазон PUA, чтобы восстановление одного
нормализатора не подставило фрагмент другого.

Примеры::

    >>> speech_to_words("2 × 3")
    'два умножить на три'
    >>> speech_to_words("5 км²")
    'пять километров в квадрате'
    >>> speech_to_words("Цена 50 €")
    'Цена пятьдесят евро'
    >>> speech_to_words("Пишите на info@example.com")
    'Пишите на info собака example точка com'
    >>> speech_to_words("8 999 123 45 67")
    'восемь девять девять девять один два три четыре пять шесть семь'

Функция идемпотентна: второй прогон по уже нормализованному тексту ничего не
меняет (слов из знаков не остаётся).
"""

from __future__ import annotations

import re

from .ru_numbers import cardinal_words, numbers_to_words

__all__ = ["speech_to_words"]

# --------------------------------------------------------------------------- #
# защита кода, URL и технических токенов
# --------------------------------------------------------------------------- #
#: Свой диапазон PUA: у ``ru_numbers`` заняты 0xE100…0xE8FF со своими
#: разделителями, поэтому здесь другой маркер и другой диапазон. Иначе
#: восстановление чужой заглушки подставило бы не тот фрагмент.
_PUA_OPEN = "\ue002"
_PUA_CLOSE = "\ue003"
_PUA_BASE = 0xE900
_PUA_LIMIT = 0xEFFF
_PLACEHOLDER_RE = re.compile("\ue002([\ue900-\uefff])\ue003")

#: Код в бэктиках — как есть (в том числе двойные пробелы внутри).
_CODE_SPAN_RE = re.compile(r"`[^`\n]*`")
#: URL не нормализуем: их вырезает сервер/чистильщик, а здесь важно не
#: испортить «https://example.com/5?a=1» знаками «/» и «=».
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
#: Токен (без пробелов и кавычек). Режем текст на токены один раз и прячем
#: те, где есть латиница: ``v2.0``, ``GPT-4``, ``--prefill``, ``EVA_TTS=0``,
#: ``report_2026.pdf``. В них «=» — не «равно», а «-» — не минус. Один проход
#: по токенам — чтобы не ловить катастрофический бэктрекинг на длинных
#: строках без латиницы.
_TOKEN_RE = re.compile(r"[^\s«»„“”\"‹›]+")
_LATIN_RE = re.compile(r"[A-Za-z]")

#: Даты прячем от правил «/» и «-»: их разберёт ``numbers_to_words``
#: («01.02.2026» → «первое февраля…», ISO — тоже).
_DATE_DMY_RE = re.compile(r"(?<![\d./])\d{1,2}[./]\d{1,2}[./]\d{4}(?!\d)")
_DATE_ISO_RE = re.compile(r"(?<![\d-])\d{4}-\d{1,2}-\d{1,2}(?!\d)")
_DATE_YM_RE = re.compile(r"(?<![\d-])\d{4}-\d{1,2}(?!\d)(?!-\d)")


class _Stash:
    """Временные заглушки для фрагментов, которые нельзя нормализовать."""

    def __init__(self) -> None:
        self._items: list[str] = []

    def __call__(self, match: "re.Match[str]") -> str:
        if len(self._items) >= _PUA_LIMIT - _PUA_BASE:
            return match.group(0)  # переполнение: лучше оставить как есть
        self._items.append(match.group(0))
        return _PUA_OPEN + chr(_PUA_BASE + len(self._items) - 1) + _PUA_CLOSE

    def put(self, value: str) -> str:
        """Спрятать готовую строку (не по объекту match)."""
        if len(self._items) >= _PUA_LIMIT - _PUA_BASE:
            return value
        self._items.append(value)
        return _PUA_OPEN + chr(_PUA_BASE + len(self._items) - 1) + _PUA_CLOSE

    def restore(self, text: str) -> str:
        def replace(match: "re.Match[str]") -> str:
            index = ord(match.group(1)) - _PUA_BASE
            if 0 <= index < len(self._items):
                return self._items[index]
            return match.group(0)

        return _PLACEHOLDER_RE.sub(replace, text)


# --------------------------------------------------------------------------- #
# словари
# --------------------------------------------------------------------------- #
_DIGIT_WORDS = (
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
)

#: (именительный, 2–4, 5+/0) и род для согласования с числом.
_UNIT_FORMS: dict[str, tuple[tuple[str, str, str], str]] = {
    "шт": (("штука", "штуки", "штук"), "f"),
    "ч": (("час", "часа", "часов"), "m"),
    "мин": (("минута", "минуты", "минут"), "f"),
    "коп": (("копейка", "копейки", "копеек"), "f"),
    "тыс": (("тысяча", "тысячи", "тысяч"), "f"),
    "млн": (("миллион", "миллиона", "миллионов"), "m"),
    "млрд": (("миллиард", "миллиарда", "миллиардов"), "m"),
}

#: Простое слово для единицы без числа («шт.» → «штук»).
_UNIT_PLURAL = {key: forms[2] for key, (forms, _) in _UNIT_FORMS.items()}

#: Валюты, которых нет в ``ru_numbers``: $ € £. Знак «₽» он умеет сам,
#: с согласованием («один рубль», «пять рублей»), поэтому его не трогаем.
_CURRENCY_FORMS: dict[str, tuple[tuple[str, str, str], str]] = {
    "$": (("доллар", "доллара", "долларов"), "m"),
    "€": (("евро", "евро", "евро"), "m"),
    "£": (("фунт", "фунта", "фунтов"), "m"),
}

_GRADUS_FORMS = ("градус", "градуса", "градусов")

#: Длины для площадей и скоростей: «км²» → «километров в квадрате».
_LENGTH_WORDS = {
    "км": "километров",
    "см": "сантиметров",
    "дм": "дециметров",
    "мм": "миллиметров",
    "м": "метров",
}
_SPEED_WORDS = {
    "км": "километров",
    "см": "сантиметров",
    "м": "метров",
}
_TIME_WORDS = {"ч": "в час", "с": "в секунду"}

#: Что за единицу считать: «₽/шт» → «рублей за штуку».
_PER_UNIT_WORDS = {
    "шт": "штуку",
    "кг": "килограмм",
    "л": "литр",
    "км": "километр",
    "ч": "час",
    "мес": "месяц",
    "чел": "человека",
    "день": "день",
    "дня": "день",
}
_CURRENCY_WORDS = {
    "₽": "рублей",
    "руб": "рублей",
    "р": "рублей",
    "$": "долларов",
    "€": "евро",
    "£": "фунтов",
}

#: Знаки-слова: заменяются одним словом (или короткой фразой). Остальные
#: знаки требуют контекста и обрабатываются отдельными функциями.
_WORD_SYMBOLS = {
    "±": "плюс-минус",
    "≤": "не больше",
    "≥": "не меньше",
    "≈": "примерно",
    "×": "умножить на",
    "÷": "делить на",
    "=": "равно",
    "<": "меньше",
    ">": "больше",
    "^": "в степени",
    "~": "примерно",
    "&": "и",
    "‰": "промилле",
    "№": "номер",
    "§": "параграф",
    "→": "стрелка вправо",
    "←": "стрелка влево",
    "↑": "стрелка вверх",
    "↓": "стрелка вниз",
    "½": "одна вторая",
    "¼": "одна четвёртая",
    "¾": "три четвёртых",
    "+": "плюс",
}
_WORD_SYMBOL_RE = re.compile("[" + re.escape("".join(_WORD_SYMBOLS)) + "]")

#: Знаки валют, которые не забрало предправило (валюта без числа).
_CURRENCY_SIGN_RE = re.compile(r"[€$£]")
#: Степени без единицы: «5²» → «пять в квадрате».
_POWER_SIGN_RE = re.compile(r"[²³]")
_POWER_WORDS = {"²": "в квадрате", "³": "в кубе"}
#: Кавычки и служебные знаки копирайта — убрать (не проговариваются).
#: Пробелы по краям тоже съедаем: «© 2026» не должен оставить ведущий пробел.
_DROP_SIGN_RE = re.compile(r"[ \t]*[©®™«»„“”‚‘’‹›\"][ \t]*")
#: Апостроф внутри слова («don't», «об'явление») оставляем: это часть токена.
_APOSTROPHE_RE = re.compile(r"['’]")
#: Непарный бэктик после вынимания кода — тоже убрать.
_BACKTICK_RE = re.compile(r"`")

#: Хештег: «#маркетинг» → «хештег маркетинг», одинокий «#» → «решётка».
_HASHTAG_RE = re.compile(r"#(?=\w)", re.UNICODE)
_LONE_HASH_RE = re.compile(r"#")

#: Тире-пауза и многоточие: → запятая. Пробелы по краям съедаем сами, чтобы
#: не оставлять «слово , слово».
_DASH_RE = re.compile(r"[ \t]*[—–‒][ \t]*")
_ELLIPSIS_RE = re.compile(r"[ \t]*…[ \t]*")

#: Дефис: минус в числовом контексте, пауза — между словами, маркер списка —
#: в начале строки. «по-русски» и «5-3» (диапазон) не трогаются.
_HYPHEN_RE = re.compile(r"[ \t]*[-−][ \t]*")

#: Звёздочка: «умножить на» только рядом с цифрой; остальное — markdown
#: (``**жирный**``, маркер списка) и просто снимается.
_STAR_RE = re.compile(r"[ \t]*\*[ \t]*")

#: Слэш: «делить на» рядом с цифрой, «или» между русскими словами
#: («и/или»), иначе просто пробел.
_SLASH_RE = re.compile(r"[ \t]*/[ \t]*")

_CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")


# --------------------------------------------------------------------------- #
# общие помощники
# --------------------------------------------------------------------------- #
def _plural_index(value: int) -> int:
    """Индекс формы: 1 → 0, 2–4 → 1, 0/5–20/… → 2 (11–14 → 2)."""
    if 11 <= value % 100 <= 14:
        return 2
    last = value % 10
    if last == 1:
        return 0
    if 2 <= last <= 4:
        return 1
    return 2


def _agreed(number: int, forms: tuple[str, str, str], gender: str) -> str:
    """«21 штука», «2 часа», «1500 рублей»: число + слово в нужной форме."""
    return f"{cardinal_words(number, gender=gender)} {forms[_plural_index(number)]}"


def _is_grouped_tail(text: str, start: int) -> bool:
    """Это хвост разрядной записи «1 500» (число уже разбито пробелом)?"""
    return bool(re.search(r"\d[ \u00a0\u202f]$", text[:start]))


def _keep_digits(match: "re.Match[str]", word: str) -> str:
    """Оставить цифры (их прочитает ``ru_numbers``), заменить только знак."""
    return f"{match.group(1)} {word}"


def _edge_word(text: str, start: int, end: int, word: str) -> str:
    """Слово вместо знака с пробелами только там, где они нужны."""
    prefix = "" if start == 0 or text[start - 1].isspace() else " "
    tail = text[end:] if end < len(text) else ""
    suffix = "" if not tail or tail[0].isspace() or tail[0] in ".,;:!?)" else " "
    return f"{prefix}{word}{suffix}"


def _join_neighbours(text: str, start: int, end: int) -> tuple[str, str]:
    """Непробельные соседи слева и справа от знака (для контекста)."""
    before = text[:start].rstrip(" \t")
    after = text[end:].lstrip(" \t")
    return (before[-1] if before else "", after[0] if after else "")


# --------------------------------------------------------------------------- #
# шаг 1: предправила (нужны цифры до их перевода)
# --------------------------------------------------------------------------- #
_EMAIL_RE = re.compile(
    r"(?<![\w.+-])([A-Za-z0-9._%+-]+)@([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)(?![\w-])"
)
_CELSIUS_LATIN_RE = re.compile(
    r"(?<![\d.,])(\d{1,7})[ \t]*°[ \t]*C(?![A-Za-zА-Яа-яЁё])", re.IGNORECASE
)
_CELSIUS_CYRILLIC_RE = re.compile(
    r"(?<![\d.,])(\d{1,7})[ \t]*°С(?![А-Яа-яЁёA-Za-z])", re.IGNORECASE
)
_DEGREE_RE = re.compile(r"(?<![\d.,])(\d{1,7})[ \t]*°(?![ \t]*[CFСcf])")
_AREA_RE = re.compile(
    r"(?<![А-Яа-яЁёA-Za-z])(км|см|дм|мм|м)[ \t]*([²³])", re.IGNORECASE
)
_SPEED_RE = re.compile(
    r"(?<![А-Яа-яЁёA-Za-z])(км|см|м)[ \t]*/[ \t]*(ч|с)(?![А-Яа-яЁёA-Za-z])",
    re.IGNORECASE,
)
_PRICE_PER_RE = re.compile(
    r"(₽|руб\.?|р\.|\$|€|£)[ \t]*/[ \t]*"
    r"(шт\.?|кг|л|км|ч|мес\.?|чел\.?|день|дня)(?![А-Яа-яЁёA-Za-z])",
    re.IGNORECASE,
)
_ABBREV_RE = re.compile(
    r"(?<![А-Яа-яЁё])т\.[ \t]?д\.", re.IGNORECASE
)
_ABBREV_TP_RE = re.compile(r"(?<![А-Яа-яЁё])т\.[ \t]?п\.", re.IGNORECASE)
_ABBREV_TE_RE = re.compile(r"(?<![А-Яа-яЁё])т\.[ \t]?е\.", re.IGNORECASE)
_ULITSA_RE = re.compile(r"(?<![А-Яа-яЁё\w])ул\.(?![А-Яа-яЁё])", re.IGNORECASE)
_GOROD_RE = re.compile(r"(?<![А-Яа-яЁё\w])г\.(?![А-Яа-яЁё])", re.IGNORECASE)
_MSK_RE = re.compile(r"(?<![А-Яа-яЁё\w])мск\.?(?![А-Яа-яЁё\w])", re.IGNORECASE)
_UNIT_ABBREV_RE = re.compile(
    r"(?<![\d.,])(\d{1,3})[ \t]*(шт|ч|мин|коп|тыс|млн|млрд)\.?"
    r"(?![А-Яа-яЁё\w])",
    re.IGNORECASE,
)
_BARE_UNIT_RE = re.compile(
    r"(?<![А-Яа-яЁёA-Za-z.,])(шт|ч|мин|коп|тыс|млн|млрд)\.?"
    r"(?![А-Яа-яЁё\w])",
    re.IGNORECASE,
)
_TIME_RE = re.compile(r"(?<![\d:])(\d{1,2}):(\d{2})(?![\d:])")
_PHONE_CANDIDATE_RE = re.compile(
    r"(?<![\d+/(])\+?\d(?:[\d \t()\-\u00a0\u202f]{5,})\d(?!\d)"
)
_CURRENCY_RE = re.compile(r"(?<![\d.,])(\d{1,15})[ \t]*([$€£])(?!\w)")
_ISO_DATE_FULL_RE = re.compile(r"\d{4}-\d{1,2}-\d{1,2}")


def _replace_email(match: "re.Match[str]") -> str:
    """``info@example.com`` → «info собака example точка com»."""
    local, domain = match.group(1), match.group(2)
    return f"{local} собака {domain.replace('.', ' точка ')}"


def _replace_celsius(match: "re.Match[str]") -> str:
    number = int(match.group(1))
    word = _GRADUS_FORMS[_plural_index(number)]
    if _is_grouped_tail(match.string, match.start()):
        return _keep_digits(match, f"{word} Цельсия")
    return f"{_agreed(number, _GRADUS_FORMS, 'm')} Цельсия"


def _replace_degree(match: "re.Match[str]") -> str:
    number = int(match.group(1))
    if _is_grouped_tail(match.string, match.start()):
        return _keep_digits(match, _GRADUS_FORMS[_plural_index(number)])
    return _agreed(number, _GRADUS_FORMS, "m")


def _space_before(match: "re.Match[str]") -> str:
    """Пробел, если слово встаёт вплотную к цифре/букве («5км²»)."""
    before = match.string[: match.start()]
    return "" if not before or before[-1].isspace() else " "


def _replace_area(match: "re.Match[str]") -> str:
    length, power = match.group(1).lower(), match.group(2)
    return f"{_space_before(match)}{_LENGTH_WORDS[length]} {_POWER_WORDS[power]}"


def _replace_speed(match: "re.Match[str]") -> str:
    length, time = match.group(1).lower(), match.group(2).lower()
    return f"{_space_before(match)}{_SPEED_WORDS[length]} {_TIME_WORDS[time]}"


def _replace_price_per(match: "re.Match[str]") -> str:
    currency = match.group(1).lower().rstrip(".")
    unit = match.group(2).lower().rstrip(".")
    return (
        f"{_space_before(match)}{_CURRENCY_WORDS[currency]} "
        f"за {_PER_UNIT_WORDS[unit]}"
    )


def _replace_unit_abbrev(match: "re.Match[str]") -> str:
    """«2 ч.» → «2 часа», но «1 500 шт.» отдаём числам (разряды)."""
    number = int(match.group(1))
    unit = match.group(2).lower()
    forms, gender = _UNIT_FORMS[unit]
    if _is_grouped_tail(match.string, match.start()):
        return _keep_digits(match, forms[2])
    return _agreed(number, forms, gender)


def _replace_bare_unit(match: "re.Match[str]") -> str:
    return _UNIT_PLURAL[match.group(1).lower()]


def _replace_gorod(match: "re.Match[str]") -> str:
    """«г.» — город, но после года — «года» («2026 г.»)."""
    before = match.string[: match.start()]
    if re.search(r"(?<!\d)(1\d{3}|2\d{3})$", before.rstrip(" \t")):
        return "года"
    return "город"


def _replace_msk(match: "re.Match[str]") -> str:
    return "по московскому времени"


def _replace_currency(match: "re.Match[str]") -> str:
    number = int(match.group(1))
    forms, gender = _CURRENCY_FORMS[match.group(2)]
    if _is_grouped_tail(match.string, match.start()):
        return _keep_digits(match, forms[2])
    return _agreed(number, forms, gender)


def _phone_words(token: str) -> str | None:
    """Телефон по цифрам или ``None``, если это не телефон.

    Порог — 7 цифр (городской номер без кода). Чтобы не перепутать с
    разрядами «1 500 000» и датами, требуем хотя бы два разделителя
    (пробел/скобка/дефис) либо явный «+» и 10+ цифр; запись, где после
    первой все группы ровно по три цифры, считаем разрядами.
    """
    digits = re.sub(r"\D", "", token)
    if len(digits) < 7:
        return None
    core = token.strip()
    if _ISO_DATE_FULL_RE.fullmatch(core):
        return None
    groups = re.findall(r"\d+", core)
    # Длинная группа допустима только у сплошного «+7999…»: иначе это уже
    # артикул/год, а не номер.
    if any(len(group) > 4 for group in groups) and len(groups) > 1:
        return None
    plus = core.startswith("+")
    separators = re.findall(r"[\s()\-\u00a0\u202f]", core)
    if len(separators) < 2 and not (plus and len(digits) >= 10):
        return None
    if (
        len(groups) > 1
        and all(len(group) == 3 for group in groups[1:])
        and not plus
        and not re.search(r"[()-]", core)
    ):
        return None  # «1 500 000» — разряды, а не телефон
    words = [_DIGIT_WORDS[int(digit)] for digit in digits]
    if plus:
        words.insert(0, "плюс")
    return " ".join(words)


def _replace_time(match: "re.Match[str]") -> str:
    """«15:00» → «15 00»: часы и минуты дальше прочитает ru_numbers."""
    return f"{match.group(1)} {match.group(2)}"


def _replace_phone(match: "re.Match[str]") -> str:
    words = _phone_words(match.group(0))
    return words if words is not None else match.group(0)


def _pre_rules(text: str) -> str:
    """Предправила: email, единицы, сокращения, телефоны, валюты."""
    stash = _Stash()
    value = _CODE_SPAN_RE.sub(stash, text)
    value = _URL_RE.sub(stash, value)
    value = _EMAIL_RE.sub(_replace_email, value)
    value = _TIME_RE.sub(_replace_time, value)
    value = _CELSIUS_LATIN_RE.sub(_replace_celsius, value)
    value = _CELSIUS_CYRILLIC_RE.sub(_replace_celsius, value)
    value = _DEGREE_RE.sub(_replace_degree, value)
    value = _AREA_RE.sub(_replace_area, value)
    value = _SPEED_RE.sub(_replace_speed, value)
    value = _PRICE_PER_RE.sub(_replace_price_per, value)
    value = _ABBREV_RE.sub("так далее", value)
    value = _ABBREV_TP_RE.sub("тому подобное", value)
    value = _ABBREV_TE_RE.sub("то есть", value)
    value = _ULITSA_RE.sub("улица", value)
    value = _GOROD_RE.sub(_replace_gorod, value)
    value = _MSK_RE.sub(_replace_msk, value)
    value = _UNIT_ABBREV_RE.sub(_replace_unit_abbrev, value)
    value = _BARE_UNIT_RE.sub(_replace_bare_unit, value)
    value = _PHONE_CANDIDATE_RE.sub(_replace_phone, value)
    value = _CURRENCY_RE.sub(_replace_currency, value)
    return stash.restore(value)


# --------------------------------------------------------------------------- #
# шаг 2: символы → слова
# --------------------------------------------------------------------------- #
def _replace_dash(match: "re.Match[str]") -> str:
    """Тире — пауза-запятая; между цифрами оставляем (это диапазон чисел)."""
    left, right = _join_neighbours(match.string, match.start(), match.end())
    if left.isdigit() and right.isdigit():
        return match.group(0)
    if not left or left in "\n.!?;:" or not right or right in "\n.!?,;:":
        return ""
    return ", "


def _replace_ellipsis(match: "re.Match[str]") -> str:
    """Многоточие — пауза-запятая: «Подождите…» → «Подождите,»."""
    left, right = _join_neighbours(match.string, match.start(), match.end())
    if not left:
        return ""
    return ", " if right else ","


def _replace_hyphen(match: "re.Match[str]") -> str:
    """Дефис: минус в числах, пауза между словами, маркер списка — убрать."""
    text = match.string
    raw = match.group(0)
    left, right = _join_neighbours(text, match.start(), match.end())
    if "−" in raw:  # U+2212 — это знак вычитания, а не дефис/тире
        word = "минус" if right else ""
        return _edge_word(text, match.start(), match.end(), word) if word else ""
    if left.isdigit() and right.isdigit():
        return raw  # «10-20»/«10 - 20» — диапазон, им займётся ru_numbers
    if len(raw) > 1:
        # Дефис отделён пробелами: между словами это пауза-тире.
        if not left or left in "\n.!?;:":
            return "минус " if right.isdigit() else ""  # маркер списка
        if _CYRILLIC_RE.match(left) and _CYRILLIC_RE.match(right):
            return ", "  # «Москва - столица»
    if not left or left in "\n.!?;:([{«\"":
        return "минус " if right.isdigit() else ""  # «-5» / «(-5)»
    if left.isdigit() or right.isdigit():
        return " минус "
    return raw  # «по-русски», «Москва-столица»: дефис внутри слова


def _replace_star(match: "re.Match[str]") -> str:
    left, right = _join_neighbours(match.string, match.start(), match.end())
    if left.isdigit() or right.isdigit():
        return " умножить на "
    return ""  # markdown: «* пункт», остатки выделения


def _replace_slash(match: "re.Match[str]") -> str:
    left, right = _join_neighbours(match.string, match.start(), match.end())
    if left.isdigit() or right.isdigit():
        return " делить на "
    if _CYRILLIC_RE.match(left) and _CYRILLIC_RE.match(right):
        before = re.search(r"[А-Яа-яЁё]+$", match.string[: match.start()])
        after = re.match(r"[А-Яа-яЁё]+", match.string[match.end() :])
        words = (
            before.group(0).lower() if before else "",
            after.group(0).lower() if after else "",
        )
        if "или" in words:
            return " "  # «и/или» → «и или», без второго «или»
        if "и" in words:
            return " или "  # «и/или» с другой стороны
    return " "


def _drop_sign(match: "re.Match[str]") -> str:
    """Снять знак; пробел вернуть, только если он разделяет два слова."""
    before = match.string[: match.start()]
    after = match.string[match.end() :]
    left = before[-1] if before else ""
    right = after[0] if after else ""
    if left and right and not left.isspace() and not right.isspace():
        return " "
    return ""


def _drop_apostrophe(match: "re.Match[str]") -> str:
    """Апостроф внутри слова — часть токена, снаружи — мусор разметки."""
    before = match.string[: match.start()]
    after = match.string[match.end() :]
    left = before[-1] if before else ""
    right = after[0] if after else ""
    if left.isalnum() and right.isalnum():
        return match.group(0)
    return ""


def _replace_hash(match: "re.Match[str]") -> str:
    """Одинокий «#» (не хештег) — «решётка» с нужными пробелами."""
    return _edge_word(match.string, match.start(), match.end(), "решётка")


def _tidy(text: str) -> str:
    """Убрать двойные пробелы и пробел перед запятой/точкой (не в коде)."""
    value = re.sub(r"[ \t]{2,}", " ", text)
    value = re.sub(r"[ \t]+([,.;:!?])", r"\1", value)
    value = re.sub(r"\n[ \t]+", "\n", value)
    return value


def _stash_tech(stash: _Stash):
    """Прятать токены с латиницей: «=», «/», «-», «+» в них — не слова."""

    def replace(match: "re.Match[str]") -> str:
        token = match.group(0)
        if _LATIN_RE.search(token):
            return stash.put(token)
        return token

    return replace


def _symbols_to_words(text: str) -> str:
    """Знаки → слова; код, URL, латинские токены и даты не трогаются."""
    stash = _Stash()
    value = _CODE_SPAN_RE.sub(stash, text)
    value = _URL_RE.sub(stash, value)
    value = _DATE_DMY_RE.sub(stash, value)
    value = _DATE_ISO_RE.sub(stash, value)
    value = _DATE_YM_RE.sub(stash, value)

    # Латинские/технические токены прячем: «=», «/», «-» в них — не слова.
    # Это раньше решётки и кавычек: «C#» и «don't» должны остаться целыми.
    value = _TOKEN_RE.sub(_stash_tech(stash), value)

    value = _HASHTAG_RE.sub("хештег ", value)
    value = _LONE_HASH_RE.sub(_replace_hash, value)
    value = _DROP_SIGN_RE.sub(_drop_sign, value)
    value = _APOSTROPHE_RE.sub(_drop_apostrophe, value)
    value = _BACKTICK_RE.sub("", value)

    value = _DASH_RE.sub(_replace_dash, value)
    value = _ELLIPSIS_RE.sub(_replace_ellipsis, value)
    value = _WORD_SYMBOL_RE.sub(
        lambda match: _edge_word(
            match.string,
            match.start(),
            match.end(),
            _WORD_SYMBOLS[match.group(0)],
        ),
        value,
    )
    value = _HYPHEN_RE.sub(_replace_hyphen, value)
    value = _STAR_RE.sub(_replace_star, value)
    value = _SLASH_RE.sub(_replace_slash, value)
    value = _CURRENCY_SIGN_RE.sub(
        lambda match: _edge_word(
            match.string,
            match.start(),
            match.end(),
            _CURRENCY_WORDS[match.group(0)],
        ),
        value,
    )
    value = _POWER_SIGN_RE.sub(
        lambda match: _edge_word(
            match.string,
            match.start(),
            match.end(),
            _POWER_WORDS[match.group(0)],
        ),
        value,
    )
    return stash.restore(_tidy(value))


# --------------------------------------------------------------------------- #
# публичная функция
# --------------------------------------------------------------------------- #
def speech_to_words(text: object) -> str:
    """Знаки, сокращения, телефоны и email — словами (до выбора TTS-движка).

    Пустая строка возвращается как есть. Функция идемпотентна: повторный
    вызов по уже нормализованному тексту ничего не меняет.
    """
    value = "" if text is None else str(text)
    if not value:
        return value
    value = _pre_rules(value)
    value = _symbols_to_words(value)
    value = numbers_to_words(value)
    return value
