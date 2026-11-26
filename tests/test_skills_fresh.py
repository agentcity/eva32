"""Свежие данные без сети: часы, парсеры ЦБ РФ и MOEX, кэш, деградация.

Запуск::

    python3 -m unittest tests.test_skills_fresh -v

Задача 0208. Ни один тест не ходит в сеть: сетевой слой (``fresh._http_get``)
подменяется фикстурой настоящего ответа — ``tests/fixtures/cbr_daily.xml``
(windows-1251, как отдаёт ЦБ) и ``tests/fixtures/moex_sber.json`` (ISS).
Таймауты, кэш и «сеть упала → None» проверяются на заглушках.

Почему фикстуры — настоящие ответы: формат ЦБ и MOEX придётся разбирать без
XML-библиотеки (на этом хосте ``pyexpat`` в сборке Python сломан), поэтому
парсер проверяется на реальном тексте, а не на придуманном минимализме.
"""

from __future__ import annotations

import os
import pathlib
import sys
import unittest
from datetime import datetime, timezone
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eva.skills import fresh  # noqa: E402
from eva.skills.clock import MOSCOW_TZ  # noqa: E402
from eva.skills.fresh import (  # noqa: E402
    CBR_TTL,
    DEFAULT_TIMEOUT,
    MOEX_TTL,
    Fact,
    currency_rate,
    degraded_fact,
    format_fact,
    normalize_currency,
    normalize_ticker,
    now_info,
    parse_cbr_xml,
    parse_moex_json,
    reset_cache,
    stock_quote,
)

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"
CBR_XML = (FIXTURES / "cbr_daily.xml").read_bytes()
MOEX_JSON = (FIXTURES / "moex_sber.json").read_bytes()


def fail_fetch(url, timeout):  # pragma: no cover - вызывается только если тест течёт
    raise AssertionError(f"сеть в тесте: {url}")


class FetchStub:
    """Заглушка сети: считает вызовы, отдаёт заданный ответ или падает."""

    def __init__(self, payload=b"", error=None):
        self.payload = payload
        self.error = error
        self.calls = []

    def __call__(self, url, timeout):
        self.calls.append((url, timeout))
        if self.error is not None:
            raise self.error
        return self.payload


class FreshTest(unittest.TestCase):
    def setUp(self):
        reset_cache()


# --------------------------------------------------------------------------- #
# Часы сервера
# --------------------------------------------------------------------------- #


class TestNowInfo(FreshTest):
    def test_format_matches_spec(self):
        fact = now_info(datetime(2026, 11, 26, 15, 30, tzinfo=MOSCOW_TZ))
        self.assertEqual(fact.kind, "time")
        self.assertEqual(fact.data, "26 ноября 2026, 15:30 MSK")
        self.assertEqual(fact.source, "часы сервера")

    def test_weekday_option(self):
        fact = now_info(datetime(2026, 11, 26, 15, 30, tzinfo=MOSCOW_TZ), with_weekday=True)
        self.assertEqual(fact.data, "четверг, 26 ноября 2026, 15:30 MSK")

    def test_utc_is_converted_to_moscow(self):
        fact = now_info(datetime(2026, 11, 26, 12, 30, tzinfo=timezone.utc))
        self.assertIn("15:30 MSK", fact.data)

    def test_naive_time_is_moscow(self):
        fact = now_info(datetime(2026, 11, 26, 9, 5))
        self.assertIn("09:05", fact.data)

    def test_date_object(self):
        from datetime import date

        fact = now_info(date(2026, 1, 2))
        self.assertTrue(fact.data.startswith("2 января 2026"))

    def test_no_network(self):
        with mock.patch.object(fresh, "_http_get", side_effect=AssertionError("сеть")):
            fact = now_info()
        self.assertIn("MSK", fact.data)


# --------------------------------------------------------------------------- #
# Парсер ЦБ РФ
# --------------------------------------------------------------------------- #


class TestCbrParser(FreshTest):
    def test_fixture_parses(self):
        day = parse_cbr_xml(CBR_XML)
        self.assertIsNotNone(day)
        self.assertEqual(day.date, "26.11.2026")
        self.assertEqual(day.get("USD").value, 82.313)
        self.assertEqual(day.get("EUR").value, 90.2397)
        self.assertEqual(day.get("CNY").value, 12.3154)

    def test_nominal_is_respected(self):
        # Иена котируется за 100 единиц: 51,6587 ₽ за 100 → 0,516587 за одну.
        day = parse_cbr_xml(CBR_XML)
        ien = day.get("JPY")
        self.assertEqual(ien.nominal, 100.0)
        self.assertAlmostEqual(ien.unit, 51.6587 / 100, places=9)

    def test_cp1251_names_decoded(self):
        day = parse_cbr_xml(CBR_XML)
        self.assertEqual(day.get("USD").name, "Доллар США")

    def test_accepts_decoded_str(self):
        day = parse_cbr_xml(CBR_XML.decode("cp1251"))
        self.assertEqual(day.get("USD").code, "USD")

    def test_garbage_is_none(self):
        for raw in (b"", b"<html></html>", b"not xml at all", b"<Valute></Valute>"):
            with self.subTest(raw=raw[:20]):
                self.assertIsNone(parse_cbr_xml(raw))

    def test_partial_block_does_not_break_others(self):
        raw = (
            '<ValCurs Date="01.01.2027"><Valute><CharCode>USD</CharCode>'
            "<Nominal>1</Nominal><Value>1 000,50</Value></Valute>"
            "<Valute><CharCode>XXX</CharCode></Valute></ValCurs>"
        )
        day = parse_cbr_xml(raw.encode("cp1251"))
        self.assertEqual(day.date, "01.01.2027")
        self.assertEqual(day.get("USD").value, 1000.5)
        self.assertIsNone(day.get("XXX"))


# --------------------------------------------------------------------------- #
# Курс: факт, кэш, деградация
# --------------------------------------------------------------------------- #


class TestCurrencyRate(FreshTest):
    def test_usd_fact(self):
        stub = FetchStub(CBR_XML)
        fact = currency_rate("USD", fetch=stub)
        self.assertEqual(fact.kind, "currency")
        self.assertEqual(fact.data, "1 USD = 82,3130 ₽")
        self.assertEqual(fact.source, "ЦБ РФ на 26.11.2026")
        self.assertEqual(stub.calls[0][0], fresh.CBR_URL)
        self.assertEqual(stub.calls[0][1], DEFAULT_TIMEOUT)

    def test_russian_names(self):
        stub = FetchStub(CBR_XML)
        self.assertEqual(currency_rate("доллар", fetch=stub).data, "1 USD = 82,3130 ₽")
        self.assertIn("1 EUR", currency_rate("евро", fetch=stub).data)
        self.assertIn("1 CNY", currency_rate("юань", fetch=stub).data)

    def test_cache_hits_once(self):
        stub = FetchStub(CBR_XML)
        first = currency_rate("USD", fetch=stub)
        second = currency_rate("EUR", fetch=stub)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(len(stub.calls), 1, "второй вызов должен брать кэш")

    def test_cache_expires_after_ttl(self):
        stub = FetchStub(CBR_XML)
        clock = {"t": 1000.0}
        with mock.patch.object(fresh, "_monotonic", side_effect=lambda: clock["t"]):
            currency_rate("USD", fetch=stub)
            clock["t"] += CBR_TTL - 1
            currency_rate("USD", fetch=stub)
            self.assertEqual(len(stub.calls), 1)
            clock["t"] += 2
            currency_rate("USD", fetch=stub)
        self.assertEqual(len(stub.calls), 2, "через час кэш обязан истечь")

    def test_no_cache_option(self):
        stub = FetchStub(CBR_XML)
        currency_rate("USD", fetch=stub, use_cache=False)
        currency_rate("USD", fetch=stub, use_cache=False)
        self.assertEqual(len(stub.calls), 2)

    def test_unknown_currency_is_none(self):
        stub = FetchStub(CBR_XML)
        self.assertIsNone(currency_rate("XYZ", fetch=stub))
        self.assertIsNone(currency_rate("абракадабра", fetch=stub))

    def test_network_error_degrades_to_none(self):
        stub = FetchStub(error=OSError("сеть недоступна"))
        self.assertIsNone(currency_rate("USD", fetch=stub))

    def test_timeout_degrades_to_none(self):
        stub = FetchStub(error=TimeoutError("timed out"))
        self.assertIsNone(currency_rate("USD", fetch=stub))

    def test_failure_is_not_cached(self):
        stub = FetchStub(error=OSError("сеть"))
        self.assertIsNone(currency_rate("USD", fetch=stub))
        self.assertIsNone(currency_rate("USD", fetch=stub))
        self.assertEqual(len(stub.calls), 2, "неудачу кэшировать нельзя")

    def test_broken_payload_degrades(self):
        fact = currency_rate("USD", fetch=FetchStub(b"<html>503</html>"))
        self.assertIsNone(fact)

    def test_degraded_fact_text(self):
        fact = degraded_fact("currency", "USD")
        self.assertIn("не удалось получить", fact.data)
        self.assertIn("USD", fact.data)
        self.assertIn("ЦБ РФ", fact.data)
        self.assertIsNone(fact.source)
        self.assertEqual(format_fact(fact), f"[Факт: {fact.data}]")

    def test_timeout_is_within_3_5_seconds(self):
        self.assertGreaterEqual(DEFAULT_TIMEOUT, 3.0)
        self.assertLessEqual(DEFAULT_TIMEOUT, 5.0)


# --------------------------------------------------------------------------- #
# Парсер MOEX
# --------------------------------------------------------------------------- #


class TestMoexParser(FreshTest):
    def test_fixture_parses(self):
        quote = parse_moex_json(MOEX_JSON)
        self.assertIsNotNone(quote)
        self.assertEqual(quote.ticker, "SBER")
        self.assertEqual(quote.name, "Сбербанк")
        self.assertEqual(quote.board, "TQBR")
        self.assertAlmostEqual(quote.price, 285.1)
        self.assertAlmostEqual(quote.change, -2.32)
        self.assertAlmostEqual(quote.change_pct, -0.81)
        self.assertAlmostEqual(quote.prev_price, 287.42)
        self.assertTrue(quote.updated.startswith("2026-11-26"))

    def test_accepts_dict_and_str(self):
        import json

        with self.subTest(kind="dict"):
            self.assertIsNotNone(parse_moex_json(json.loads(MOEX_JSON)))
        with self.subTest(kind="str"):
            self.assertIsNotNone(parse_moex_json(MOEX_JSON.decode("utf-8")))

    def test_cyrillic_names_are_utf8(self):
        quote = parse_moex_json(MOEX_JSON)
        self.assertEqual(quote.name, "Сбербанк")

    def test_unknown_ticker_without_rows_is_none(self):
        payload = {
            "securities": {"columns": ["SECID"], "data": []},
            "marketdata": {"columns": ["SECID", "LAST"], "data": []},
        }
        self.assertIsNone(parse_moex_json(payload))

    def test_no_last_price_is_none(self):
        payload = {
            "securities": {"columns": ["SECID"], "data": [["TEST"]]},
            "marketdata": {"columns": ["SECID", "LAST"], "data": [["TEST", None]]},
        }
        self.assertIsNone(parse_moex_json(payload))

    def test_garbage_is_none(self):
        for raw in (b"", b"{", b"[]", b'{"marketdata": 5}', b"<html></html>"):
            with self.subTest(raw=raw[:12]):
                self.assertIsNone(parse_moex_json(raw))

    def test_change_computed_from_prev_close(self):
        payload = {
            "securities": {
                "columns": ["SECID", "SHORTNAME", "PREVPRICE"],
                "data": [["TEST", "Тест", 100.0]],
            },
            "marketdata": {
                "columns": ["SECID", "BOARDID", "LAST"],
                "data": [["TEST", "TQBR", 102.0]],
            },
        }
        quote = parse_moex_json(payload)
        self.assertAlmostEqual(quote.change, 2.0)
        self.assertAlmostEqual(quote.change_pct, 2.0)


# --------------------------------------------------------------------------- #
# Котировка: факт, URL, кэш, деградация
# --------------------------------------------------------------------------- #


class TestStockQuote(FreshTest):
    def test_sber_fact(self):
        stub = FetchStub(MOEX_JSON)
        fact = stock_quote("Сбер", fetch=stub)
        self.assertEqual(fact.kind, "stock")
        self.assertIn("SBER (Сбербанк) = 285,10 ₽", fact.data)
        self.assertIn("-2,32 ₽", fact.data)
        self.assertIn("-0,81 %", fact.data)
        self.assertIn("287,42 ₽", fact.data)
        self.assertEqual(fact.source, "MOEX ISS, 26.11.2026 15:14 MSK")

    def test_url_is_tqbr_and_ticker(self):
        stub = FetchStub(MOEX_JSON)
        stock_quote("Газпром", fetch=stub)
        url = stub.calls[0][0]
        self.assertIn("/boards/TQBR/securities/GAZP.json", url)
        self.assertTrue(url.startswith("https://iss.moex.com/"))

    def test_cache_hits_once(self):
        stub = FetchStub(MOEX_JSON)
        stock_quote("Сбер", fetch=stub)
        stock_quote("SBER", fetch=stub)
        self.assertEqual(len(stub.calls), 1, "второй вызов должен брать кэш")

    def test_cache_expires_after_ttl(self):
        stub = FetchStub(MOEX_JSON)
        clock = {"t": 500.0}
        with mock.patch.object(fresh, "_monotonic", side_effect=lambda: clock["t"]):
            stock_quote("SBER", fetch=stub)
            clock["t"] += MOEX_TTL + 1
            stock_quote("SBER", fetch=stub)
        self.assertEqual(len(stub.calls), 2)

    def test_network_error_degrades_to_none(self):
        self.assertIsNone(stock_quote("Сбер", fetch=FetchStub(error=OSError("нет"))))

    def test_broken_json_degrades_to_none(self):
        self.assertIsNone(stock_quote("Сбер", fetch=FetchStub(b"<html>")))

    def test_unknown_name_is_none_without_network(self):
        stub = FetchStub(MOEX_JSON)
        self.assertIsNone(stock_quote("Ромашка", fetch=stub))
        self.assertEqual(stub.calls, [])


# --------------------------------------------------------------------------- #
# Названия → коды
# --------------------------------------------------------------------------- #


class TestNormalize(FreshTest):
    def test_currency_aliases(self):
        cases = {
            "доллар": "USD",
            "доллара": "USD",
            "бакс": "USD",
            "$": "USD",
            "USD": "USD",
            "usd": "USD",
            "евро": "EUR",
            "€": "EUR",
            "юань": "CNY",
            "юаня": "CNY",
            "фунт стерлингов": "GBP",
            "йена": "JPY",
            "тенге": "KZT",
            "рубль": "RUB",
        }
        for text, code in cases.items():
            with self.subTest(text=text):
                self.assertEqual(normalize_currency(text), code)

    def test_unknown_currency(self):
        for text in ("абракадабра", "", "долг", "крипта"):
            with self.subTest(text=text):
                self.assertIsNone(normalize_currency(text))

    def test_ticker_aliases(self):
        cases = {
            "Сбер": "SBER",
            "сбербанк": "SBER",
            "Сбербанка": "SBER",
            "Газпром": "GAZP",
            "Яндекс": "YDEX",
            "Лукойл": "LKOH",
            "ВТБ": "VTBR",
            "SBER": "SBER",
            "gazp": "GAZP",
        }
        for text, ticker in cases.items():
            with self.subTest(text=text):
                self.assertEqual(normalize_ticker(text), ticker)

    def test_ticker_word_boundaries(self):
        # «сбережения» — не Сбер, «магнитный» — не Магнит.
        self.assertIsNone(normalize_ticker("сбережения"))
        self.assertIsNone(normalize_ticker("магнитный"))


# --------------------------------------------------------------------------- #
# Формат факта
# --------------------------------------------------------------------------- #


class TestFormatFact(FreshTest):
    def test_with_source(self):
        self.assertEqual(
            format_fact(Fact("time", "26 ноября 2026, 15:30 MSK", "часы сервера")),
            "[Факт: 26 ноября 2026, 15:30 MSK (источник: часы сервера)]",
        )

    def test_without_source(self):
        self.assertEqual(format_fact(Fact("stock", "нет данных")), "[Факт: нет данных]")

    def test_number_formatting(self):
        self.assertEqual(fresh.format_number(3244.5, 2), "3 244,50")
        self.assertEqual(fresh.format_number(-2.32, 2), "-2,32")
        self.assertEqual(fresh.format_number(82.313, 4), "82,3130")
        self.assertEqual(fresh.format_number(None, 2), "None")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
