"""Роутер намерений «свежие данные» (задача 0208).

Задача: 3B-модель не умеет tool-calling, поэтому инструмент вызывается **до**
генерации — по вопросу, а не моделью. Роутер отвечает на один вопрос:

    «Это вопрос про время, курс валюты или цену акции? И о чём именно?»

и отдаёт :class:`Intent` с параметром (ISO-код валюты / тикер MOEX). Данные
добывает :mod:`eva.skills.fresh`, а секция ``СПРАВКА`` подкладывается в
системный блок запроса (:func:`fact_line` + :func:`with_fact`) — модель только
формулирует ответ и не выдумывает цифры.

Задача 0212: формулировка блока стала явной инструкцией («используй дословно;
не говори, что данных нет; не цитируй слово «СПРАВКА»»), потому что 3B не
связывала служебную строку ``[Факт: …]`` с вопросом и отвечала «не могу
предоставить». Здесь же живёт страж противоречия (:func:`guard_answer`,
:class:`FactGuardStream`): он срабатывает в ядре, потому что знает справку
этого запроса, а вырезание служебных пометок из готового текста — это
фильтр-плагин ``plugins/ru-facts`` (работает и в веб-чате, и в CLI).

Почему отдельный модуль, а не :mod:`eva.skills.registry`. Скиллы реестра по
контракту ``docs/hybrid.md`` **никогда не ходят в сеть** и отвечают сами,
подменяя модель. Здесь наоборот: сеть разрешена (три источника), но ответ
формулирует модель. Смешивать эти режимы в одном реестре — значит потерять
детерминированность и офлайн-тесты; поэтому реестр не тронут.

Осторожность важнее полноты: если непонятно, о чём вопрос, роутер молчит и
модель отвечает как раньше. Ложный факт хуже отсутствия факта.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Callable, Iterable, Mapping, Sequence

from eva.catalog import normalize
from eva.skills.fresh import (
    DEFAULT_CURRENCIES,
    DEFAULT_TIMEOUT,
    Fact,
    currency_mentions,
    currency_rate,
    degraded_fact,
    format_fact,
    now_info,
    stock_quote,
    ticker_mentions,
)

__all__ = [
    "FACT_SECTION_HINT",
    "FACT_SECTION_TITLE",
    "FactGuardStream",
    "Intent",
    "build_facts",
    "detect",
    "fact_block",
    "fact_entries",
    "fact_line",
    "fact_phrase",
    "facts_for",
    "guard_answer",
    "is_refusal",
    "with_fact",
]

#: Намерения, которые роутер понимает.
KINDS = ("time", "currency", "stock")

# --------------------------------------------------------------------------- #
# Справка: что видит модель вместо служебной строки [Факт: …]
# --------------------------------------------------------------------------- #

#: Заголовок секции. Модель не должна цитировать это слово в ответе.
FACT_SECTION_TITLE = "СПРАВКА"

#: Заголовок секции с явной командой. Формулировка важнее краткости: 3B
#: (задача 0212) отрицала факт, хотя он был рядом, и печатала ``[Факт: …]``.
#: «дословно» + «не говори, что данных нет» + «не цитируй» закрывают все три
#: ошибки живого ответа.
FACT_SECTION_HINT = (
    "СПРАВКА (используй эти данные дословно; не говори, что данных нет; "
    "не цитируй слово «СПРАВКА»):"
)

# --------------------------------------------------------------------------- #
# Время и дата
# --------------------------------------------------------------------------- #

#: «Который час», «сколько времени в Москве», «подскажи точное время».
_TIME_RE = re.compile(
    r"(?:"
    r"котор\w*\s+(?:сейчас\s+|же\s+)?час\w*"
    r"|сколько\s+(?:сейчас\s+)?врем\w*"
    r"|(?:текущ\w+|точн\w+|местн\w+|московск\w+)\s+врем\w*"
    r"|врем\w*\s+(?:сейчас|у\s+вас|в\s+москве|по\s+москве)"
    r"|(?:подскаж\w*|скаж\w*|назов\w*)\s+(?:мне\s+)?(?:сейчас\s+)?"
    r"(?:точн\w+\s+)?врем\w*"
    r"|во\s+сколько\s+(?:сейчас\s+)?(?:врем\w*|час\w*)"
    r"|сколько\s+(?:сейчас\s+)?на\s+часах"
    r")",
    re.IGNORECASE,
)

#: «Какое сегодня число», «какой день недели», «какая дата».
_DATE_RE = re.compile(
    r"(?:"
    r"как(?:ое|ой|ая)\s+(?:сегодня\s+)?(?:число|дата|день(?:\s+недели)?)"
    r"|(?:сегодняшн\w+|текущ\w+)\s+(?:дата|число|день(?:\s+недели)?)"
    r"|какой\s+(?:сегодня\s+)?день(?:\s+недели)?"
    r"|какое\s+число"
    r")",
    re.IGNORECASE,
)

#: Явный вопрос про день недели — тогда в факт добавляем и его.
_WEEKDAY_RE = re.compile(
    r"день\s+недели|какой\s+(?:сегодня\s+)?день", re.IGNORECASE
)

#: «Сколько времени» — не всегда про часы: чаще это длительность. Такие
#: вопросы остаются модели (и каталогу), факт времени им не подкладывается.
_DURATION_RE = re.compile(
    r"(?:"
    r"занима\w*|займ\w*|займу\w*|длит\w*|длят\w*|потребу\w*"
    r"|остал\w*|ждать|ждем|ждём|пути|ехать|лететь|идти"
    r"|работа\w*|офис\w*|поддержк\w*|техподдержк\w*"
    r"|тариф\w*|доставк\w*|возврат\w*|подписк\w*"
    r"|сборк\w*|загрузк\w*|установк\w*|обучен\w*"
    r")",
    re.IGNORECASE,
)

# --------------------------------------------------------------------------- #
# Курс и котировки
# --------------------------------------------------------------------------- #

#: Слова «это про деньги»: курс/цена/стоимость.
_RATE_WORD_RE = re.compile(
    r"(?:"
    r"курс\w*|котировк\w*|по\s+ч[её]м|почём"
    r"|сколько\s+(?:сейчас\s+)?сто(?:ит|ят|ить)"
    r"|цена|стоимость"
    r")",
    re.IGNORECASE,
)

#: Слова «это про бумагу на бирже», а не про товар/услугу.
_STOCK_WORD_RE = re.compile(
    r"(?:"
    r"акци\w*|котировк\w*|бирж\w*|мосбирж\w*|\bmoex\b"
    r"|капитализац\w*|\bipo\b|бумаг\w*"
    r")",
    re.IGNORECASE,
)

#: Общий вопрос о курсе без конкретной валюты: «курс валют», «курс ЦБ».
_ANY_CURRENCY_RE = re.compile(
    r"(?:валют\w*|\bцб\b|центробанк\w*|центральн\w+\s+банк)", re.IGNORECASE
)

#: Служебные вопросы каталога: даже если в них мелькнуло «цена», это не курс
#: и не биржа — их берут скиллы каталога.
_CATALOG_GUARD_RE = re.compile(
    r"(?:"
    r"тариф\w*|подписк\w*|доставк\w*|возврат\w*|скидк\w*|промокод\w*"
    r"|офис\w*|поддержк\w*|техподдержк\w*"
    r")",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Intent:
    """Распознанное намерение: что спросили и о чём именно."""

    kind: str
    params: tuple[str, ...] = ()
    phrase: str = ""
    asks_weekday: bool = False

    @property
    def param(self) -> str | None:
        """Главный параметр: код валюты или тикер (первый, если их несколько)."""
        return self.params[0] if self.params else None


def detect(text: str) -> Intent | None:
    """Определить намерение по вопросу; непонятно → ``None``.

    Порядок важен: время → бумага → валюта. «Курс акций Сбера» — это биржа,
    а не курс ЦБ, поэтому бумага проверяется раньше валюты.
    """
    raw = str(text or "")
    if not raw.strip():
        return None
    value = normalize(raw)
    if not value:
        return None

    # 1. Время и дата. Длительность («сколько времени ехать») — не наши.
    time_hit = _TIME_RE.search(value)
    date_hit = _DATE_RE.search(value)
    if (time_hit or date_hit) and not _DURATION_RE.search(value):
        phrase = (time_hit or date_hit).group(0)
        return Intent(
            "time",
            phrase=phrase,
            asks_weekday=bool(_WEEKDAY_RE.search(value)),
        )

    # 2. Бумага MOEX: нужен и тикер, и «рыночное» слово. «Сбер» без вопроса о
    #    цене — не наша зона (модель ответит сама).
    if _CATALOG_GUARD_RE.search(value):
        return None
    tickers = ticker_mentions(raw)
    if tickers and (_STOCK_WORD_RE.search(value) or _RATE_WORD_RE.search(value)):
        phrase = (_STOCK_WORD_RE.search(value) or _RATE_WORD_RE.search(value))
        return Intent("stock", tuple(tickers), phrase=phrase.group(0))

    # 3. Курс валюты: валюта названа и спросили про курс/цену.
    currencies = [code for code in currency_mentions(value) if code != "RUB"]
    rate_hit = _RATE_WORD_RE.search(value)
    if currencies and rate_hit:
        return Intent("currency", tuple(currencies), phrase=rate_hit.group(0))

    # 4. «Курс валют» без конкретной валюты — отдаём основные.
    if _ANY_CURRENCY_RE.search(value) and rate_hit:
        return Intent("currency", (), phrase=rate_hit.group(0))

    return None


def facts_for(
    intent: Intent,
    *,
    now: datetime | date | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    fetch: Callable[[str, float], bytes] | None = None,
) -> list[Fact]:
    """Собрать факты по намерению; недоступный источник → факт деградации."""
    if intent.kind == "time":
        return [now_info(now, with_weekday=intent.asks_weekday)]

    if intent.kind == "currency":
        codes = intent.params or DEFAULT_CURRENCIES
        facts: list[Fact] = []
        for code in codes:
            fact = currency_rate(code, timeout=timeout, fetch=fetch)
            facts.append(fact if fact is not None else degraded_fact("currency", code))
        return facts

    if intent.kind == "stock":
        facts = []
        for ticker in intent.params:
            fact = stock_quote(ticker, timeout=timeout, fetch=fetch)
            facts.append(fact if fact is not None else degraded_fact("stock", ticker))
        return facts

    return []  # pragma: no cover - detect() других видов не выдаёт


def build_facts(
    text: str,
    *,
    now: datetime | date | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    fetch: Callable[[str, float], bytes] | None = None,
) -> list[Fact]:
    """Намерение + данные одним вызовом: ``[]`` — фактов нет."""
    intent = detect(text)
    if intent is None:
        return []
    return facts_for(intent, now=now, timeout=timeout, fetch=fetch)


def fact_block(entries: Iterable[str]) -> str:
    """Собрать секцию справки: заголовок с командой + строки данных.

    ``[]`` (или строки из одних пробелов) → ``""``: вставлять нечего.
    Заголовок печатается один раз на все строки, поэтому «курс валют» —
    это одна секция с тремя курсами, а не три команды.
    """
    lines = [str(entry).strip() for entry in entries if str(entry).strip()]
    if not lines:
        return ""
    return "\n".join([FACT_SECTION_HINT, *lines])


#: Старый формат служебной строки: ``[Факт: <data> (источник: <source>)]``.
_LEGACY_FACT_RE = re.compile(
    r"\[Факт:\s*(?P<data>.*?)(?:\s*\(источник:\s*(?P<source>[^)]*)\))?\s*\]",
    re.IGNORECASE | re.DOTALL,
)


def _as_entry(line: str) -> str:
    """Строка данных без служебной обёртки (старый ``[Факт: …]`` тоже)."""
    value = str(line or "").strip()
    match = _LEGACY_FACT_RE.search(value)
    if match is None:
        return value
    data = (match.group("data") or "").strip()
    source = (match.group("source") or "").strip()
    if source:
        return f"{data} ({source})"
    return data


def fact_entries(block: str | None) -> list[str]:
    """Разобрать секцию справки обратно в строки данных.

    Нужно стражу: он отвечает **данными** из справки, а не текстом модели.
    Блок мог прийти и в старом формате ``[Факт: …]`` (тесты, чужие вызовы) —
    тогда обёртка снимается.
    """
    lines: list[str] = []
    for raw in str(block or "").splitlines():
        value = raw.strip()
        if not value:
            continue
        if value.upper().startswith(FACT_SECTION_TITLE):
            continue  # заголовок секции — не данные
        entry = _as_entry(value)
        if entry:
            lines.append(entry)
    return lines


def fact_line(
    text: str,
    *,
    now: datetime | date | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    fetch: Callable[[str, float], bytes] | None = None,
) -> str | None:
    """Готовая секция ``СПРАВКА`` (строки через ``\\n``) или ``None``.

    ``None`` значит «роутер не понял вопрос, вставлять нечего». Намерение,
    данные которого получить не удалось, даёт **не** ``None``, а честную
    строку «не удалось получить…»: иначе модель выдумает цифру.
    """
    facts = build_facts(text, now=now, timeout=timeout, fetch=fetch)
    if not facts:
        return None
    block = fact_block(format_fact(fact) for fact in facts)
    return block or None


def with_fact(
    messages: Sequence[Mapping[str, str]],
    line: str | None,
) -> list[dict[str, str]]:
    """Вставить секцию справки в системную часть запроса.

    Если системное сообщение есть — секция дописывается в его конец (модель
    видит команду рядом с данными). Если системного нет — создаётся отдельное
    ``system``-сообщение только со справкой: терять данные из-за того, что
    персона не задана, нельзя. Входной список не меняется.

    Готовая секция (начинается с ``СПРАВКА``) вставляется как есть, одиночная
    строка данных оборачивается в секцию — правило для модели должно быть
    рядом с данными в любом случае.
    """
    result = [dict(message) for message in messages]
    block = (line or "").strip()
    if not block:
        return result
    if not block.upper().startswith(FACT_SECTION_TITLE):
        block = fact_block([block]) or block
    for message in result:
        if message.get("role") == "system":
            content = str(message.get("content") or "").rstrip()
            message["content"] = f"{content}\n\n{block}" if content else block
            return result
    result.insert(0, {"role": "system", "content": block})
    return result


# --------------------------------------------------------------------------- #
# Страж противоречия (задача 0212)
# --------------------------------------------------------------------------- #

#: Отказ модели от данных, которые ей дали. Узко и по делу: широкое «не могу»
#: ловило бы «не могу не отметить» в нормальном ответе.
REFUSAL_RE = re.compile(
    r"(?:"
    r"не\s+могу\s+(?:предоставить|дать|сообщить|подсказать|получить|найти"
    r"|проверить|подтвердить|ответить)"
    r"|нет\s+доступа(?:\s+к)?"
    r"|не\s+имею\s+(?:доступа|информации|данных|возможности|сведений)"
    r"|не\s+располагаю\s+(?:данными|информацией|сведениями)"
    r"|не\s+удалось\s+(?:получить|найти|уточнить)"
    r")",
    re.IGNORECASE,
)

#: Строки деградации: это не данные, подставлять их вместо отказа нельзя.
_DEGRADED_RE = re.compile(
    r"не\s+удалось\s+получить|нет\s+ответа\s+от|источник\s+недоступен",
    re.IGNORECASE,
)

#: Сколько символов начала ответа держим, ожидая отказ. Отказ в живой
#: реплике стоит в первом предложении («К сожалению, я не могу…»); держать
#: весь ответ до точки — значит убить стриминг.
GUARD_HEAD_LIMIT = 80


def is_refusal(text: str) -> bool:
    """Есть ли в ответе отказ от данных (см. :data:`REFUSAL_RE`)."""
    return bool(REFUSAL_RE.search(str(text or "")))


def fact_phrase(entries: Sequence[str]) -> str | None:
    """Короткая фактическая фраза из справки (или ``None``).

    Строки деградации («не удалось получить…») фразой не считаются: заменять
    отказ на «не удалось получить» — значит ничего не сказать. Одна строка —
    как есть, несколько — через ``; ``.
    """
    usable = [
        entry.strip()
        for entry in entries
        if entry.strip() and not _DEGRADED_RE.search(entry)
    ]
    if not usable:
        return None
    phrase = "; ".join(entry.rstrip(".") for entry in usable)
    return phrase if phrase.endswith((".", "!", "?")) else phrase + "."


def guard_answer(text: str, entries: Sequence[str]) -> str:
    """Заменить отказ фактической фразой; без данных отказ остаётся.

    Детерминированно и без второго прогона модели: если ответ противоречит
    справке, пользователю показывается справка.
    """
    value = str(text or "")
    phrase = fact_phrase(entries)
    if phrase is None:
        return value
    # Ответ пустой или состоит из служебного огрызка («СПРАВКА», «[Факт: …») —
    # это не ответ пользователю: показываем справку, даже если формально
    # «отказа» в тексте нет.
    stripped = value.strip().strip("[](){}«»\"'.,:;-— ")
    service_only = not stripped or stripped.lower() in {
        "справка", "справка:", "факт", "факты", "[факт", "источник"
    }
    if is_refusal(value) or service_only:
        return phrase
    return value


class FactGuardStream:
    """Потоковый страж: держит начало ответа, пока не ясно — отказ или нет.

    Отменить уже отправленные в поток символы нельзя, поэтому первый кусок
    ответа буферизуется до первого предложения (или :data:`GUARD_HEAD_LIMIT`
    символов). Увидели отказ — отдаём фразу из справки и глушим остаток
    (модель уже «сказала» отказ, продолжение contradicts факту). Отказа нет —
    отдаём буфер и дальше пропускаем текст как есть.

    Без данных (``entries`` пусты или деградация) страж прозрачен: обычное
    «не знаю» остаётся как есть, стриминг не задерживается.
    """

    def __init__(self, entries: Sequence[str]) -> None:
        self._phrase = fact_phrase(entries)
        self._head: list[str] = []
        self._decided = False
        self._swallow = False

    def __bool__(self) -> bool:
        return self._phrase is not None

    def feed(self, text: str) -> str:
        if not text:
            return ""
        if self._phrase is None:
            return text
        if self._swallow:
            return ""
        if self._decided:
            return text
        self._head.append(text)
        head = "".join(self._head)
        if REFUSAL_RE.search(head):
            self._decided = True
            self._swallow = True
            self._head = []
            return self._phrase
        if _sentence_end(head) or len(head) >= GUARD_HEAD_LIMIT:
            self._decided = True
            self._head = []
            return head
        return ""

    def flush(self) -> str:
        if self._phrase is None:
            # Прозрачный страж: ничего не буферизовали.
            return ""
        if self._swallow or self._decided:
            self._head = []
            return ""
        head = "".join(self._head)
        self._head = []
        self._decided = True
        if head and REFUSAL_RE.search(head):
            self._swallow = True
            return self._phrase
        return head


#: Конец предложения: точка/восклицание/вопрос/перевод строки. Точка внутри
#: числа («82.3») тоже сойдёт за границу — это лишь чуть раньше отдаёт буфер.
_SENTENCE_END_RE = re.compile(r"[.!?…\n]")


def _sentence_end(text: str) -> bool:
    return bool(_SENTENCE_END_RE.search(text))

