"""Роутер намерений «свежие данные» (задача 0208).

Задача: 3B-модель не умеет tool-calling, поэтому инструмент вызывается **до**
генерации — по вопросу, а не моделью. Роутер отвечает на один вопрос:

    «Это вопрос про время, курс валюты или цену акции? И о чём именно?»

и отдаёт :class:`Intent` с параметром (ISO-код валюты / тикер MOEX). Данные
добывает :mod:`eva.skills.fresh`, а строка ``[Факт: … (источник: …)]``
подкладывается в системный блок запроса (:func:`fact_line` +
:func:`with_fact`) — модель только формулирует ответ и не выдумывает цифры.

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
from typing import Callable, Mapping, Sequence

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
    "Intent",
    "build_facts",
    "detect",
    "fact_line",
    "facts_for",
    "with_fact",
]

#: Намерения, которые роутер понимает.
KINDS = ("time", "currency", "stock")

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


def fact_line(
    text: str,
    *,
    now: datetime | date | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    fetch: Callable[[str, float], bytes] | None = None,
) -> str | None:
    """Готовый блок ``[Факт: …]`` (строки через ``\\n``) или ``None``.

    ``None`` значит «роутер не понял вопрос, вставлять нечего». Намерение,
    данные которого получить не удалось, даёт **не** ``None``, а честную
    строку «не удалось получить…»: иначе модель выдумает цифру.
    """
    facts = build_facts(text, now=now, timeout=timeout, fetch=fetch)
    if not facts:
        return None
    return "\n".join(format_fact(fact) for fact in facts)


def with_fact(
    messages: Sequence[Mapping[str, str]],
    line: str | None,
) -> list[dict[str, str]]:
    """Вставить блок ``[Факт]`` в системную часть запроса.

    Если системное сообщение есть — строка дописывается в его конец (модель
    видит правило «используй [Факт]» рядом с данными). Если системного нет —
    создаётся отдельное ``system``-сообщение только с фактом: терять данные
    из-за того, что персона не задана, нельзя. Входной список не меняется.
    """
    result = [dict(message) for message in messages]
    block = (line or "").strip()
    if not block:
        return result
    for message in result:
        if message.get("role") == "system":
            content = str(message.get("content") or "").rstrip()
            message["content"] = f"{content}\n\n{block}" if content else block
            return result
    result.insert(0, {"role": "system", "content": block})
    return result
