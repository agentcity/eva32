"""Свежие данные для контекста: часы, курс ЦБ РФ, котировки MOEX (задача 0208).

Зачем. 3B-модель отвечает из весов: «сейчас 2023 год», курс выдумывает. Лечить
это tool-calling'ом дорого и ненадёжно — маленькая модель не умеет вызывать
инструменты. Поэтому данные добываются **до** генерации, а модель только
формулирует ответ по готовой секции ``СПРАВКА`` (см. ``eva/skills/intent.py``
и вставку в ``tools/chat_server.py`` / ``tools/chat.py``).

Что здесь есть:

* :func:`now_info` — часы сервера, ``Europe/Moscow``, без сети;
* :func:`currency_rate` — курс ЦБ РФ (``XML_daily.asp``, без ключа), кэш 1 час;
* :func:`stock_quote` — цена последней сделки на MOEX ISS (``TQBR``), кэш 60 с;
* :func:`parse_cbr_xml` / :func:`parse_moex_json` — чистые парсеры для тестов.

Правила, общие для всех трёх:

* **таймаут 4 с** на запрос и один повтор только при ошибке проверки
  сертификата (системное хранилище, см. ниже): чат не должен ждать сеть;
* **graceful degradation**: сеть/парсинг/HTTP упали — возвращается ``None``,
  исключение наружу не летит. Текст «не удалось получить» собирает
  :func:`degraded_fact`, его и увидит модель;
* **кэш только успеха**: неудачу не запоминаем, иначе один сбой сети «залипнет»
  на час;
* **без ключей и без внешних пакетов** — только стандартная библиотека.

XML ЦБ разбирается регулярками, а не ``xml.etree``: формат плоский и стабильный
(``<Valute><CharCode>…``), а на этом хосте ``pyexpat`` у сломанной сборки
Python 3.14 не импортируется вовсе — ``ElementTree``/``minidom`` падают ещё до
разбора. Регулярки работают везде и не тянут зависимостей; на фикстуре
настоящего ответа ЦБ (``tests/fixtures/cbr_daily.xml``) проверено.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Callable, Mapping

from eva.skills.clock import MONTHS, MOSCOW_TZ, WEEKDAYS

__all__ = [
    "CBR_TTL",
    "CBR_URL",
    "CURRENCY_ALIASES",
    "CbrDay",
    "CbrRate",
    "DEFAULT_CURRENCIES",
    "DEFAULT_TIMEOUT",
    "Fact",
    "MOEX_TTL",
    "MOEX_URL",
    "Quote",
    "TICKER_ALIASES",
    "currency_mentions",
    "currency_rate",
    "degraded_fact",
    "format_fact",
    "format_number",
    "normalize_currency",
    "normalize_ticker",
    "now_info",
    "parse_cbr_xml",
    "parse_moex_json",
    "reset_cache",
    "stock_quote",
    "ticker_mentions",
]

logger = logging.getLogger("eva.skills.fresh")

#: Курс ЦБ РФ: XML, windows-1251, без ключа. Один запрос — все валюты.
CBR_URL = "https://www.cbr.ru/scripts/XML_daily.asp"
#: Индексы MOEX ISS (IMOEX/RTSI): другая доска и другое поле цены.
MOEX_INDEX_URL = (
    "https://iss.moex.com/iss/engines/stock/markets/index/boards/SNDX/"
    "securities/{ticker}.json?iss.meta=off"
)
#: Состав индекса с весами: кто входит и с какой долей.
MOEX_INDEX_MEMBERS_URL = (
    "https://iss.moex.com/iss/statistics/engines/stock/markets/index/"
    "analytics/{ticker}.json?iss.meta=off&iss.only=analytics&limit=100"
)

#: Имена индексов для человека.
INDEX_TITLES = {
    "IMOEX": "Индекс Мосбиржи",
    "RTSI": "Индекс РТС",
    "RTS": "Индекс РТС",  # MOEX отдаёт SECID=RTS на доске SNDX
}

#: Котировки MOEX ISS: без ключа, одна бумага на запрос.
MOEX_URL = (
    "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/"
    "securities/{ticker}.json"
)
#: Таймаут запроса, секунд. 4 с — внутри оговорённых 3–5 с: чат не должен
#: ждать сеть дольше, чем думает над ответом.
DEFAULT_TIMEOUT = 4.0
#: Курс ЦБ публикуется раз в день — час кэша безопасен.
CBR_TTL = 3600.0
#: Котировка живёт секунды: 60 с — компромисс между свежестью и нагрузкой.
MOEX_TTL = 60.0
#: Представляемся честно: без UA часть CDN отвечает 403.
USER_AGENT = "eva32-chat/0.1 (fresh-data skill)"
#: Валюты, которые отдаём, когда спросили «курс валют» без конкретной.
DEFAULT_CURRENCIES = ("USD", "EUR", "CNY")

# --------------------------------------------------------------------------- #
# Факт: одна строка, которую видит модель
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Fact:
    """Готовая строка данных для секции справки (задача 0212).

    ``data`` — сами данные, ``source`` — откуда они. Служебной обёртки здесь
    нет: секцию ``СПРАВКА`` с командой модели собирает
    :func:`eva.skills.intent.fact_block`, поэтому одна и та же строка годится
    и для системного блока, и для подстановки стражем.
    ``source=None`` — у факта нет источника (деградация: «не удалось
    получить…»), и скобки с источником не печатаются.
    """

    kind: str
    data: str
    source: str | None = None

    def line(self) -> str:
        return format_fact(self)


def format_fact(fact: Fact) -> str:
    """Строка данных: ``26 ноября 2026, 15:30 MSK (часы сервера)``."""
    text = str(fact.data).strip()
    source = str(fact.source or "").strip()
    if source:
        return f"{text} ({source})"
    return text


def format_number(value: Any, places: int = 2) -> str:
    """Число по-русски: пробел между тысячами, запятая перед дробью.

    Деньги печатаются всегда с заданным числом знаков (``285,10``), а не
    «285,1» как в :func:`eva.skills.base.format_number`: курс и цена акции —
    денежные величины, хвостовой ноль в них значим.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number or number in (float("inf"), float("-inf")):
        return "—"
    text = f"{abs(number):,.{max(0, int(places))}f}"
    text = text.replace(",", " ").replace(".", ",")
    return f"-{text}" if number < 0 else text


# --------------------------------------------------------------------------- #
# Часы сервера: без сети
# --------------------------------------------------------------------------- #


def now_info(
    now: datetime | date | None = None,
    *,
    with_weekday: bool = False,
) -> Fact:
    """Текущие дата и время по Москве: «26 ноября 2026, 15:30 MSK».

    Сети не нужно: время берётся из системных часов. ``now`` — подмена для
    тестов и воспроизводимых прогонов (``datetime``/``date``); наивное время
    считается московским.
    """
    moment = _moscow(now)
    stamp = f"{moment.day} {MONTHS[moment.month]} {moment.year}, {moment:%H:%M} MSK"
    if with_weekday:
        stamp = f"{WEEKDAYS[moment.weekday()]}, {stamp}"
    return Fact("time", stamp, "часы сервера")


def _moscow(now: datetime | date | None = None) -> datetime:
    """Момент в московской зоне; ``None`` — системные часы."""
    if now is None:
        return datetime.now(MOSCOW_TZ)
    if isinstance(now, datetime):
        moment = now
    elif isinstance(now, date):
        moment = datetime(now.year, now.month, now.day)
    else:  # pragma: no cover - защита от неверного типа
        raise TypeError(f"now: ожидался datetime/date/None, получено {type(now).__name__}")
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=MOSCOW_TZ)
    return moment.astimezone(MOSCOW_TZ)


# --------------------------------------------------------------------------- #
# Кэш: TTL на успешный ответ
# --------------------------------------------------------------------------- #


def _monotonic() -> float:
    """Монотонные часы (подменяются в тестах срока жизни кэша)."""
    return time.monotonic()


class _TTLCache:
    """Простой кэш «ключ → значение» с абсолютным сроком жизни."""

    def __init__(self, ttl: float) -> None:
        self.ttl = float(ttl)
        self._items: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Any | None:
        item = self._items.get(key)
        if item is None:
            return None
        expires, value = item
        if _monotonic() >= expires:
            self._items.pop(key, None)
            return None
        return value

    def put(self, key: str, value: Any) -> None:
        self._items[key] = (_monotonic() + self.ttl, value)

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:  # pragma: no cover - удобство отладки
        return len(self._items)


_CBR_CACHE = _TTLCache(CBR_TTL)
_MOEX_CACHE = _TTLCache(MOEX_TTL)


def reset_cache() -> None:
    """Сбросить оба кэша (тесты и ручные прогоны)."""
    _CBR_CACHE.clear()
    _MOEX_CACHE.clear()


# --------------------------------------------------------------------------- #
# Сеть: одна точка отказа, её и подменяют тесты
# --------------------------------------------------------------------------- #


#: TLS-контекст запросов: по умолчанию системный бандл, после ошибки проверки
#: сертификата — хранилище доверия ОС (см. :func:`_system_trust_context`).
_CONTEXT: ssl.SSLContext | None = None
_SYSTEM_CONTEXT: ssl.SSLContext | None = None
_SYSTEM_CONTEXT_TRIED = False


def _ssl_context() -> ssl.SSLContext:
    """Контекст TLS по умолчанию (создаётся один раз на процесс)."""
    global _CONTEXT
    if _CONTEXT is None:
        _CONTEXT = ssl.create_default_context()
    return _CONTEXT


def _system_trust_context() -> ssl.SSLContext | None:
    """TLS-контекст с корнями из хранилища доверия ОС (только macOS).

    Зачем это нужно. MOEX ISS отдаёт цепочку от «Russian Trusted Root CA»
    (Минцифры). macOS ей доверяет — Safari и curl работают, — а OpenSSL из
    Homebrew нет: его сборка CA не читает keychain, и запрос падает с
    ``CERTIFICATE_VERIFY_FAILED``. Поэтому при ошибке проверки сертификата
    повторяем запрос с тем же набором корней, что у системы:
    ``security find-certificate -a -p`` (публичные сертификаты, без ключей).

    Проверка сертификата **не отключается** — расширяется только список
    доверенных корней до системного. Не macOS или нет ``security`` → ``None``,
    и вызывающий честно деградирует.
    """
    global _SYSTEM_CONTEXT, _SYSTEM_CONTEXT_TRIED
    if _SYSTEM_CONTEXT_TRIED:
        return _SYSTEM_CONTEXT
    _SYSTEM_CONTEXT_TRIED = True
    if sys.platform != "darwin":
        return None
    security = shutil.which("security")
    if security is None:
        return None
    try:
        result = subprocess.run(
            [security, "find-certificate", "-a", "-p"],
            capture_output=True,
            timeout=10,
            check=False,
        )
    # pragma: no cover - среда без security
    except (OSError, subprocess.SubprocessError) as exc:
        logger.info("хранилище сертификатов ОС недоступно: %s", exc)
        return None
    pem = result.stdout or b""
    if result.returncode != 0 or b"BEGIN CERTIFICATE" not in pem:
        return None
    try:
        context = ssl.create_default_context()
        # cadata в ssl принимает PEM строкой (bytes трактуются как DER).
        context.load_verify_locations(cadata=pem.decode("ascii", "replace"))
    except (ssl.SSLError, ValueError) as exc:  # pragma: no cover - битый PEM
        logger.info("сертификаты ОС не загрузились: %s", exc)
        return None
    _SYSTEM_CONTEXT = context
    return context


def _is_cert_error(exc: BaseException) -> bool:
    """Ошибка именно проверки TLS-сертификата (а не таймаут/обрыв)?"""
    reason = getattr(exc, "reason", None)
    for item in (reason, exc):
        if isinstance(item, ssl.SSLCertVerificationError):
            return True
        if isinstance(item, ssl.SSLError) and "CERTIFICATE_VERIFY_FAILED" in str(item):
            return True
    return "CERTIFICATE_VERIFY_FAILED" in str(exc)


def _open_url(url: str, timeout: float, context: ssl.SSLContext | None) -> Any:
    """Один GET с указанным TLS-контекстом; ответ — файлоподобный объект."""
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/xml, application/json;q=0.9, */*;q=0.1",
        },
    )
    return urllib.request.urlopen(request, timeout=float(timeout), context=context)


def _http_get(url: str, timeout: float) -> bytes:
    """GET без повторов; исключения наружу — их ловит вызывающий скилл.

    Единственный повтор — после ошибки проверки сертификата: тот же запрос с
    системным хранилищем доверия (см. :func:`_system_trust_context`).
    """
    global _CONTEXT
    try:
        with _open_url(url, timeout, _ssl_context()) as response:
            return response.read()
    except (urllib.error.URLError, ssl.SSLError) as exc:
        if not _is_cert_error(exc):
            raise
        fallback = _system_trust_context()
        if fallback is None:
            raise
        _CONTEXT = fallback  # дальше не тратим время на заведомо падающий контекст
        with _open_url(url, timeout, fallback) as response:
            return response.read()


def _fetch(url: str, timeout: float, fetch: Callable[[str, float], bytes] | None) -> Any:
    """Вызов сети: подменённый ``fetch`` или настоящий ``_http_get``."""
    return (fetch or _http_get)(url, float(timeout))


# --------------------------------------------------------------------------- #
# Курс ЦБ РФ
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CbrRate:
    """Одна валюта в ответе ЦБ: сколько рублей стоит ``nominal`` единиц."""

    code: str
    nominal: float
    value: float
    name: str = ""

    @property
    def unit(self) -> float:
        """Рублей за одну единицу валюты (номинал ЦБ бывает 10 и 100)."""
        return self.value / self.nominal if self.nominal else self.value


@dataclass(frozen=True)
class CbrDay:
    """Срез ЦБ на дату (``date`` — как в XML: ``ДД.ММ.ГГГГ``)."""

    date: str
    rates: Mapping[str, CbrRate]

    def get(self, code: str) -> CbrRate | None:
        return self.rates.get((code or "").upper())


#: Блок валюты в XML ЦБ. Формат плоский, вложенности внутри блока нет.
_CBR_VALUTE_RE = re.compile(r"<Valute\b[^>]*>(.*?)</Valute>", re.IGNORECASE | re.DOTALL)
#: Поля блока: имя тега → текст. ``\1`` — тот же тег в закрывающем.
_CBR_FIELD_RE = re.compile(
    r"<(CharCode|Nominal|Value|Name)>(.*?)</\1>", re.IGNORECASE | re.DOTALL
)
#: Дата среза в атрибуте корня: ``<ValCurs Date="26.11.2026" …>``.
_CBR_DATE_RE = re.compile(
    r"<ValCurs\b[^>]*\bDate\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE
)


def parse_cbr_xml(raw: bytes | str) -> CbrDay | None:
    """Разобрать ``XML_daily.asp`` в :class:`CbrDay`; мусор → ``None``.

    Кодировка ответа — windows-1251 (объявлена в прологе), но на всякий случай
    принимаем и уже раскодированную строку.
    """
    text = _decode(raw)
    if not text or "<Valute" not in text:
        return None
    date_match = _CBR_DATE_RE.search(text)
    stamp = _unescape(date_match.group(1).strip()) if date_match else ""
    rates: dict[str, CbrRate] = {}
    for block in _CBR_VALUTE_RE.findall(text):
        fields: dict[str, str] = {}
        for match in _CBR_FIELD_RE.finditer(block):
            fields[match.group(1).lower()] = _unescape(match.group(2).strip())
        code = fields.get("charcode", "").upper()
        value = _to_float(fields.get("value", ""))
        nominal = _to_float(fields.get("nominal", "1")) or 1.0
        if len(code) != 3 or value is None:
            continue
        rates[code] = CbrRate(
            code=code,
            nominal=nominal,
            value=value,
            name=fields.get("name", ""),
        )
    if not rates:
        return None
    return CbrDay(date=stamp, rates=rates)


def _cbr_day(
    *,
    timeout: float,
    fetch: Callable[[str, float], bytes] | None,
    use_cache: bool,
) -> CbrDay | None:
    """Срез ЦБ: из кэша или из сети; любая ошибка → ``None`` (в лог)."""
    if use_cache:
        cached = _CBR_CACHE.get("daily")
        if isinstance(cached, CbrDay):
            return cached
    try:
        raw = _fetch(CBR_URL, timeout, fetch)
        day = parse_cbr_xml(raw)
    except Exception as exc:  # noqa: BLE001 — сеть/парсинг не должны ронять чат
        logger.info("курс ЦБ не получен: %s", exc)
        return None
    if day is None:
        logger.info("курс ЦБ: ответ не разобран (%d байт)", len(raw or b""))
        return None
    if use_cache:
        _CBR_CACHE.put("daily", day)
    return day


def currency_rate(
    code: str,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    fetch: Callable[[str, float], bytes] | None = None,
    use_cache: bool = True,
) -> Fact | None:
    """Курс валюты ЦБ РФ: «1 USD = 78,1234 ₽ (ЦБ РФ на 26.11.2026)».

    ``code`` — ISO-код или русское название («доллар», «евро», «юань»).
    Неизвестная валюта, сбой сети или парсинга → ``None``.
    """
    iso = normalize_currency(code)
    if iso is None:
        return None
    day = _cbr_day(timeout=timeout, fetch=fetch, use_cache=use_cache)
    if day is None:
        return None
    rate = day.get(iso)
    if rate is None:
        logger.info("ЦБ РФ не публикует код %s", iso)
        return None
    data = f"1 {iso} = {format_number(rate.unit, 4)} ₽"
    source = f"ЦБ РФ на {day.date}" if day.date else "ЦБ РФ"
    return Fact("currency", data, source)


# --------------------------------------------------------------------------- #
# Котировки MOEX
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Quote:
    """Котировка бумаги на момент ответа MOEX."""

    ticker: str
    name: str
    price: float
    change: float | None = None
    change_pct: float | None = None
    prev_price: float | None = None
    updated: str | None = None
    board: str = "TQBR"


def parse_moex_index(raw: bytes | str | Mapping[str, Any]) -> Quote | None:
    """Разобрать ответ ISS по ИНДЕКСУ (доска SNDX, поле ``CURRENTVALUE``).

    Отдельная функция, потому что у индекса нет ``LAST``/``PREVPRICE``:
    цена — ``CURRENTVALUE``, изменение — ``LASTCHANGEPRC`` (в пунктах и
    процентах). Свалить это в :func:`parse_moex_json` нельзя: перепутанные
    поля дают правдоподобно неверную цифру (живой случай: «индекс 2915,68»
    при фактических 2333,84).
    """
    payload = raw
    if isinstance(raw, (bytes, str)):
        text = _decode_json(raw)
        if not text:
            return None
        try:
            payload = json.loads(text)
        except ValueError:
            return None
    if not isinstance(payload, Mapping):
        return None
    marketdata = _moex_table(payload.get("marketdata"))
    if not marketdata or not marketdata.get("rows"):
        return None
    row = marketdata["rows"][0]
    price = _first_number(row, ("CURRENTVALUE", "MARKETPRICE", "LASTVALUE"))
    if price is None:
        return None
    change_pct = _first_number(row, ("LASTCHANGEPRC",), allow_zero=True)
    change_abs = _first_number(row, ("LASTCHANGE",), allow_zero=True)
    ticker = str(row.get("SECID") or "").upper()
    return Quote(
        ticker=ticker or "IMOEX",
        name=INDEX_TITLES.get(ticker or "IMOEX", ticker or "индекс"),
        price=price,
        change=change_abs,
        change_pct=change_pct,
        prev_price=None,
        updated=_updated(row),
        board="SNDX",
    )


def parse_moex_members(raw: bytes | str | Mapping[str, Any]) -> list[tuple[str, str, float]]:
    """Состав индекса из ISS: ``[(тикер, имя, вес %), …]`` по убыванию веса.

    Вес в ISS уже в процентах (0.27 = 0,27 %); наружу отдаём как есть.
    """
    payload = raw
    if isinstance(raw, (bytes, str)):
        text = _decode_json(raw)
        if not text:
            return []
        try:
            payload = json.loads(text)
        except ValueError:
            return []
    if not isinstance(payload, Mapping):
        return []
    table = _moex_table(payload.get("analytics"))
    if not table or not table.get("rows"):
        return []
    members: list[tuple[str, str, float]] = []
    for row in table["rows"]:
        ticker = str(row.get("ticker") or row.get("secids") or "").strip().upper()
        if not ticker:
            continue
        name = str(row.get("shortnames") or "").strip()
        weight = _first_number(row, ("weight",), allow_zero=True) or 0.0
        members.append((ticker, name, float(weight)))
    members.sort(key=lambda item: item[2], reverse=True)
    return members


def index_members(
    ticker: str = "IMOEX",
    *,
    timeout: float = DEFAULT_TIMEOUT,
    fetch: Callable[[str, float], bytes] | None = None,
    use_cache: bool = True,
) -> tuple[tuple[str, str, float], ...]:
    """Состав индекса (``IMOEX``/``RTSI``) по весам; сеть упала → пусто."""
    code = str(ticker or "IMOEX").strip().upper() or "IMOEX"
    key = "members:" + code
    if use_cache:
        cached = _MOEX_CACHE.get(key)
        if isinstance(cached, tuple):
            return cached
    url = MOEX_INDEX_MEMBERS_URL.format(ticker=urllib.parse.quote(code))
    try:
        members = tuple(parse_moex_members(_fetch(url, timeout, fetch)))
    except Exception as exc:  # noqa: BLE001 — сеть/парсинг деградируют
        logger.info("состав индекса %s не получен: %s", code, exc)
        return ()
    if members and use_cache:
        # Состав меняется раз в квартал, а ISS отдаёт прошлую сессию:
        # держим дольше котировок.
        _MOEX_CACHE.put(key, members)
    return members


def index_top_fact(
    ticker: str = "IMOEX",
    *,
    limit: int = 5,
    timeout: float = DEFAULT_TIMEOUT,
    fetch: Callable[[str, float], bytes] | None = None,
    use_cache: bool = True,
) -> Fact | None:
    """Топ бумаг индекса по весу + их цены одним фактом.

    «Какие акции в топе индекса Мосбиржи» — это ДВА запроса: состав с весами
    и цена каждой бумаги. Состав берём из ISS analytics (limit=100), цены — из
    обычных котировок (кэш 60 с), поэтому повторный вопрос дешёвый.
    """
    code = str(ticker or "IMOEX").strip().upper() or "IMOEX"
    members = index_members(
        code, timeout=timeout, fetch=fetch, use_cache=use_cache
    )
    if not members:
        return None
    title = INDEX_TITLES.get(code, code)
    parts: list[str] = []
    for member_ticker, member_name, weight in members[: max(1, int(limit))]:
        fact = stock_quote(
            member_ticker, timeout=timeout, fetch=fetch, use_cache=use_cache
        )
        price = ""
        if fact is not None:
            # «SBER (Сбербанк) = 285,17 ₽» → берём цену и процент
            match = re.search(r"= ([\d\s,]+) ₽(.*)", fact.data)
            if match:
                price = f"{match.group(1).strip()} ₽"
        label = member_name or member_ticker
        # Цена — первой: модель охотнее переносит в ответ то, что стоит
        # в начале фрагмента (живой случай: без цен в ответе).
        if price:
            piece = f"{member_ticker} ({label}) — {price}, вес {format_number(weight, 2)} %"
        else:
            piece = f"{member_ticker} ({label}) — вес {format_number(weight, 2)} %"
        parts.append(piece)
    if not parts:
        return None
    return Fact(
        kind="index_top",
        data=(
            f"{title} ({code}), топ по весу (цена, вес): " + "; ".join(parts)
            + ". Назови каждую бумагу с её ценой."
        ),
        source="MOEX ISS",
    )


def parse_moex_json(raw: bytes | str | Mapping[str, Any]) -> Quote | None:
    """Разобрать ответ ISS по бумаге; нет цены → ``None``.

    Берём доску ``TQBR`` (основной режим акций), цену последней сделки
    (``LAST``, с запасными полями), изменение за день к закрытию прошлого дня
    (``CHANGE``) и процент (``LASTTOPREVPRICE``, иначе считаем сами).
    """
    payload = raw
    if isinstance(raw, (bytes, str)):
        # JSON ISS — UTF-8 (в отличие от XML ЦБ в windows-1251).
        text = _decode_json(raw)
        if not text:
            return None
        try:
            payload = json.loads(text)
        except ValueError:
            return None
    if not isinstance(payload, Mapping):
        return None
    securities = _moex_table(payload.get("securities"))
    marketdata = _moex_table(payload.get("marketdata"))
    if not marketdata or not marketdata.get("rows"):
        return None
    row = _board_row(marketdata["rows"]) or {}
    sec = _board_row(securities.get("rows") if securities else []) or {}
    price = _first_number(
        row, ("LAST", "LCURRENTPRICE", "MARKETPRICE", "WAPRICE", "CLOSEPRICE")
    )
    if price is None:
        return None
    prev = _first_number(sec, ("PREVPRICE", "PREVLEGALCLOSEPRICE"))
    change = _first_number(row, ("CHANGE",), allow_zero=True)
    if change is None:
        change = _first_number(row, ("LASTCHANGE",), allow_zero=True)
    if change is None and prev is not None:
        change = price - prev
    percent = _first_number(row, ("LASTTOPREVPRICE",), allow_zero=True)
    if percent is None and change is not None and prev:
        percent = change / prev * 100.0
    ticker = str(row.get("SECID") or sec.get("SECID") or "").upper()
    name = str(sec.get("SHORTNAME") or sec.get("SECNAME") or "").strip()
    return Quote(
        ticker=ticker,
        name=name,
        price=price,
        change=change,
        change_pct=percent,
        prev_price=prev,
        updated=_updated(row),
        board=str(row.get("BOARDID") or "TQBR"),
    )


def index_quote(
    ticker: str = "IMOEX",
    *,
    timeout: float = DEFAULT_TIMEOUT,
    fetch: Callable[[str, float], bytes] | None = None,
    use_cache: bool = True,
) -> Fact | None:
    """Факт по индексу MOEX: «Индекс Мосбиржи (IMOEX) = 2 333,84 п. (-0,52 %)».

    Отдельно от :func:`stock_quote`: у индекса другая доска (``SNDX``) и
    другое поле цены (``CURRENTVALUE``). Перепутать их означает показать
    правдоподобно неверную цифру — именно так случилось с «индексом 2915,68»
    при фактических 2333,84.
    """
    code = str(ticker or "IMOEX").strip().upper() or "IMOEX"
    quote: Quote | None = None
    if use_cache:
        cached = _MOEX_CACHE.get("index:" + code)
        if isinstance(cached, Quote):
            quote = cached
    if quote is None:
        url = MOEX_INDEX_URL.format(ticker=urllib.parse.quote(code))
        try:
            raw = _fetch(url, timeout, fetch)
            quote = parse_moex_index(raw)
        except Exception as exc:  # noqa: BLE001 — сеть/парсинг не роняют чат
            logger.info("индекс %s не получен: %s", code, exc)
            return None
        if quote is None:
            logger.info("индекс %s: ответ не разобран", code)
            return None
        if use_cache:
            _MOEX_CACHE.put("index:" + code, quote)
    title = INDEX_TITLES.get(quote.ticker or code, quote.name or code)
    data = f"{title} ({quote.ticker or code}) = {format_number(quote.price, 2)} п."
    if quote.change_pct is not None:
        data += f", за день {format_number(quote.change_pct, 2)} %"
    if quote.updated:
        data += f", обновлено {quote.updated}"
    return Fact(kind="index", data=data, source="MOEX ISS")


def _moex_table(block: Any) -> dict[str, Any] | None:
    """``{"columns": [...], "data": [[...]]}`` ISS → ``{"rows": [{...}]}``."""
    if not isinstance(block, Mapping):
        return None
    columns = block.get("columns")
    data = block.get("data")
    if not isinstance(columns, list) or not isinstance(data, list):
        return None
    rows: list[dict[str, Any]] = []
    for raw in data:
        if not isinstance(raw, (list, tuple)):
            continue
        rows.append(dict(zip(columns, raw)))
    return {"rows": rows}


def _board_row(rows: Any) -> dict[str, Any] | None:
    """Строка доски ``TQBR``; иначе первая — запрос и так про TQBR."""
    if not isinstance(rows, (list, tuple)):
        return None
    for row in rows:
        if isinstance(row, Mapping) and row.get("BOARDID") in (None, "", "TQBR"):
            return dict(row)
    first = rows[0] if rows else None
    return dict(first) if isinstance(first, Mapping) else None


def _first_number(
    row: Mapping[str, Any],
    keys: tuple[str, ...],
    *,
    allow_zero: bool = False,
) -> float | None:
    """Первое числовое поле из списка; ``None`` и мусор пропускаем."""
    for key in keys:
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        number = float(value)
        if number != number:
            continue
        if number == 0.0 and not allow_zero:
            # Ноль в LAST бывает у бумаги без сделок — это не цена.
            continue
        return number
    return None


def _updated(row: Mapping[str, Any]) -> str | None:
    """Время данных: ``SYSTIME`` (дата и время, MSK) или ``UPDATETIME``."""
    stamp = row.get("SYSTIME")
    if isinstance(stamp, str) and stamp.strip():
        return stamp.strip()
    stamp = row.get("UPDATETIME")
    if isinstance(stamp, str) and stamp.strip():
        return stamp.strip()
    return None


def stock_quote(
    ticker: str,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    fetch: Callable[[str, float], bytes] | None = None,
    use_cache: bool = True,
) -> Fact | None:
    """Котировка MOEX: «SBER (Сбербанк) = 285,10 ₽, за день -2,32 ₽ (-0,81 %)».

    ``ticker`` — тикер MOEX или русское название («Сбер», «Газпром»,
    «Яндекс»). Неизвестная бумага, сбой сети или парсинга → ``None``.
    """
    code = normalize_ticker(ticker)
    if code is None:
        return None
    quote: Quote | None = None
    if use_cache:
        cached = _MOEX_CACHE.get(code)
        if isinstance(cached, Quote):
            quote = cached
    if quote is None:
        url = MOEX_URL.format(ticker=urllib.parse.quote(code))
        try:
            raw = _fetch(url, timeout, fetch)
            quote = parse_moex_json(raw)
        except Exception as exc:  # noqa: BLE001 — сеть/парсинг не должны ронять чат
            logger.info("котировка %s не получена: %s", code, exc)
            return None
        if quote is None:
            logger.info("котировка %s: ответ не разобран", code)
            return None
        if use_cache:
            _MOEX_CACHE.put(code, quote)
    return _quote_fact(quote)


def _quote_fact(quote: Quote) -> Fact:
    """Строка факта из котировки: цена, изменение за день, время MOEX."""
    title = quote.ticker or "бумага"
    if quote.name and quote.name.upper() != title:
        title = f"{title} ({quote.name})"
    data = f"{title} = {format_number(quote.price, 2)} ₽"
    if quote.change is not None:
        change = f"за день {format_number(quote.change, 2)} ₽"
        if quote.change_pct is not None:
            change += f" ({format_number(quote.change_pct, 2)} %)"
        data += f", {change}"
    if quote.prev_price:
        data += f", закрытие вчера {format_number(quote.prev_price, 2)} ₽"
    source = "MOEX ISS"
    stamp = _moex_stamp(quote.updated)
    if stamp:
        source = f"{source}, {stamp} MSK"
    return Fact("stock", data, source)


def _moex_stamp(updated: str | None) -> str:
    """``2026-11-26 14:59:39`` → ``26.11.2026 14:59``; мусор → пусто."""
    if not updated:
        return ""
    text = str(updated).strip()
    match = re.match(
        r"^(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2})", text
    )
    if match:
        year, month, day, hour, minute = match.groups()
        return f"{day}.{month}.{year} {hour}:{minute}"
    match = re.match(r"^(\d{2}):(\d{2})", text)
    if match:  # только время (UPDATETIME без даты) — без даты и печатаем
        return f"{match.group(1)}:{match.group(2)}"
    return ""

# --------------------------------------------------------------------------- #
# Названия → коды: «доллар» → USD, «Сбер» → SBER
# --------------------------------------------------------------------------- #

#: Упоминания валют в вопросе. Порядок важен: более специфичная запись выше.
CURRENCY_ALIASES: tuple[tuple[str, str], ...] = (
    (r"(?:доллар\w*|долл\b|бакс\w*|\busd\b|\$)", "USD"),
    (r"(?:евро|\beur\b|€)", "EUR"),
    (r"(?:юан\w*|женьминьби|\bcny\b|¥)", "CNY"),
    (r"(?:фунт\w*\s+стерлинг\w*|\bgbp\b|£)", "GBP"),
    (r"(?:йен\w*|иен\w*|\bjpy\b)", "JPY"),
    (r"(?:тенге|\bkzt\b)", "KZT"),
    (r"(?:белорусск\w*\s+рубл\w*|\bbyn\b)", "BYN"),
    (r"(?:швейцарск\w*\s+франк\w*|\bchf\b)", "CHF"),
    (r"(?:турецк\w*\s+лир\w*|\btry\b)", "TRY"),
    (r"(?:индийск\w*\s+рупи\w*|\binr\b)", "INR"),
    (r"(?:рубл\w*|\brub\b|₽)", "RUB"),
)

#: Тикеры MOEX и их русские имена. Список не претендует на полноту: он
#: покрывает бумаги, о которых спрашивают в чате, и легко расширяется.
TICKER_ALIASES: dict[str, tuple[str, ...]] = {
    "SBER": ("сбер", "сбербанк", "сбербанка", "sberbank"),
    "GAZP": ("газпром", "газпрома", "gazprom"),
    "YDEX": ("яндекс", "яндекса", "yandex"),
    "LKOH": ("лукойл", "лукойла", "lukoil"),
    "VTBR": ("втб", "vtb"),
    "ROSN": ("роснефть", "роснефти", "rosneft"),
    "GMKN": ("норникель", "норильск", "норильский никель", "nornickel"),
    "MTSS": ("мтс", "mts"),
    "AFLT": ("аэрофлот", "аэрофлота", "aeroflot"),
    "TATN": ("татнефть", "татнефти"),
    "PLZL": ("полюс", "полюса", "polyus"),
    "SNGS": ("сургутнефтегаз", "сургутнефтегаза"),
    "MOEX": ("московская биржа", "мосбирж", "moex"),
    "NVTK": ("новатэк", "новатэка", "novatek"),
    "ALRS": ("алроса", "алросы", "alrosa"),
    "MGNT": ("магнит", "магнита", "magnit"),
    "CHMF": ("северсталь", "северстали", "severstal"),
    "PHOR": ("фосагро", "фосагро", "phosagro"),
    "IRAO": ("интер рао", "интеррао"),
    "FEES": ("россети", "фск"),
}

_CURRENCY_PATTERNS = tuple(
    (re.compile(pattern, re.IGNORECASE), code) for pattern, code in CURRENCY_ALIASES
)
#: Латинский тикер в вопросе: «цена акции SBER». Только заглавные 3–6 букв.
_TICKER_WORD_RE = re.compile(r"(?<![A-Za-zА-Яа-яЁё0-9])([A-Z]{3,6})(?![A-Za-zА-Яа-яЁё0-9])")
#: ISO-код валюты словом: «USD», «cny».
_ISO_CODE_RE = re.compile(r"(?<![A-Za-zА-Яа-яЁё0-9])([A-Za-z]{3})(?![A-Za-zА-Яа-яЁё0-9])")


def normalize_currency(text: str) -> str | None:
    """Код валюты из названия или ISO: «доллар» → ``USD``; неизвестно → ``None``."""
    raw = str(text or "").strip()
    if not raw:
        return None
    upper = raw.upper()
    if len(upper) == 3 and upper.isalpha():
        return upper
    for pattern, code in _CURRENCY_PATTERNS:
        if pattern.search(raw):
            return code
    return None


def currency_mentions(text: str) -> list[str]:
    """Все валюты вопроса по порядку упоминания, без повторов.

    Нужен роутеру: «курс доллара и евро» — это два факта, а не первый
    попавшийся. Рубль возвращается тоже: он важен как валюта вопроса
    («евро к рублю»), но сам по себе курсом не считается.
    """
    found: list[tuple[int, str]] = []
    for pattern, code in _CURRENCY_PATTERNS:
        for match in pattern.finditer(str(text or "")):
            found.append((match.start(), code))
    found.sort()
    result: list[str] = []
    for _start, code in found:
        if code not in result:
            result.append(code)
    return result


#: Падежные окончания существительного: «Сбер» → «Сбера», «Лукойл» → «Лукойла».
#: Список закрытый намеренно: «сбережения» и «магнитный» не должны становиться
#: Сбером и Магнитом, поэтому произвольные хвосты не допускаются.
_CASE_ENDINGS = r"(?:а|я|у|ю|ом|ем|е|ы|и|ой|ей)?"


def _alias_regex(alias: str) -> re.Pattern[str]:
    """Слово-алиас с границами и падежным окончанием, но без «сбережений»."""
    return re.compile(
        r"(?<![А-Яа-яЁёA-Za-z])"
        + re.escape(alias)
        + _CASE_ENDINGS
        + r"(?![А-Яа-яЁёA-Za-z])",
        re.IGNORECASE,
    )


_TICKER_ALIAS_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (_alias_regex(alias), ticker)
    for ticker, aliases in TICKER_ALIASES.items()
    for alias in aliases
)


def normalize_ticker(text: str) -> str | None:
    """Тикер MOEX из тикера или русского названия: «Сбер» → ``SBER``.

    Сначала известные имена (Сбер, Газпром, Яндекс…), затем латинский тикер
    заглавными буквами («SBER», «GAZP»). Неизвестное → ``None``: выдумывать
    бумагу по похожему слову нельзя.
    """
    raw = str(text or "").strip()
    if not raw:
        return None
    upper = raw.upper()
    if upper in TICKER_ALIASES:
        return upper
    for pattern, ticker in _TICKER_ALIAS_PATTERNS:
        if pattern.search(raw):
            return ticker
    match = _TICKER_WORD_RE.search(raw)
    if match:
        return match.group(1)
    return None


def ticker_mentions(text: str) -> list[str]:
    """Тикеры бумаг, упомянутых в вопросе, по порядку и без повторов."""
    raw = str(text or "")
    found: list[tuple[int, int, str]] = []
    for pattern, ticker in _TICKER_ALIAS_PATTERNS:
        for match in pattern.finditer(raw):
            found.append((match.start(), match.end(), ticker))
    for match in _TICKER_WORD_RE.finditer(raw):
        found.append((match.start(), match.end(), match.group(1)))
    # Сначала длинные совпадения (они точнее), затем по позиции в тексте.
    found.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    result: list[str] = []
    covered_until = -1
    for start, end, ticker in found:
        if start < covered_until:
            continue
        covered_until = end
        if ticker not in result:
            result.append(ticker)
    return result


# --------------------------------------------------------------------------- #
# Деградация: честный текст вместо выдуманной цифры
# --------------------------------------------------------------------------- #


def degraded_fact(kind: str, subject: str = "", reason: str | None = None) -> Fact:
    """Факт «данные не получены»: модель должна сказать это, а не выдумать.

    Текст без источника: источник не ответил, ссылаться не на что.
    """
    what = {"currency": "курс", "stock": "котировку", "time": "время"}.get(kind, "данные")
    subject = str(subject or "").strip()
    tail = f" {subject}" if subject else ""
    if reason is None:
        reason = {
            "currency": "нет ответа от ЦБ РФ",
            "stock": "нет ответа от MOEX",
        }.get(kind, "источник недоступен")
    return Fact(kind, f"не удалось получить {what}{tail} — {reason}")


# --------------------------------------------------------------------------- #
# Разбор сырого ответа
# --------------------------------------------------------------------------- #


def _decode(raw: bytes | str) -> str:
    """Байты → строка: windows-1251 (ЦБ) с откатом на UTF-8."""
    if isinstance(raw, str):
        return raw
    if not isinstance(raw, (bytes, bytearray)):
        return ""
    try:
        return bytes(raw).decode("cp1251")
    except UnicodeDecodeError:  # pragma: no cover - cp1251 декодирует почти всё
        return bytes(raw).decode("utf-8", "replace")


def _decode_json(raw: bytes | str) -> str:
    """Байты → строка для JSON: UTF-8, с откатом на windows-1251."""
    if isinstance(raw, str):
        return raw
    if not isinstance(raw, (bytes, bytearray)):
        return ""
    try:
        return bytes(raw).decode("utf-8")
    except UnicodeDecodeError:
        return bytes(raw).decode("cp1251", "replace")


def _to_float(text: str | None) -> float | None:
    """``"78,1234"`` → ``78.1234``; мусор → ``None``."""
    if text is None:
        return None
    cleaned = str(text).replace("\xa0", "").replace(" ", "").replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _unescape(text: str) -> str:
    """Минимальные XML-сущности: ЦБ экранирует ``&`` в названиях."""
    return (
        text.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&apos;", "'")
        .replace("&amp;", "&")
    )
