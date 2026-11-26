"""Роутер намерений «свежие данные»: 20+ фраз, параметры, вставка факта.

Запуск::

    python3 -m unittest tests.test_intent -v

Задача 0208. Сеть не нужна: данные подменены фикстурами ЦБ РФ и MOEX
(``tests/fixtures``), а деградация проверяется заглушкой, которая падает.
"""

from __future__ import annotations

import http.client
import json
import os
import pathlib
import sys
import threading
import unittest
from datetime import datetime
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eva.skills import fresh, intent  # noqa: E402
from eva.skills.clock import MOSCOW_TZ  # noqa: E402
from eva.skills.intent import (  # noqa: E402
    Intent,
    build_facts,
    detect,
    fact_line,
    with_fact,
)
from tools import chat_server  # noqa: E402

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"
CBR_XML = (FIXTURES / "cbr_daily.xml").read_bytes()
MOEX_JSON = (FIXTURES / "moex_sber.json").read_bytes()

#: 15:30 по Москве, 26 ноября 2026 (четверг).
NOW = datetime(2026, 11, 26, 15, 30, tzinfo=MOSCOW_TZ)

#: 26 фраз, по которым роутер обязан сработать: (вопрос, вид, параметр).
POSITIVE = [
    ("Который час?", "time", None),
    ("Сколько сейчас времени?", "time", None),
    ("Сколько времени в Москве?", "time", None),
    ("Подскажи точное время.", "time", None),
    ("Какое сегодня число?", "time", None),
    ("Какая сегодня дата?", "time", None),
    ("Какой сегодня день недели?", "time", None),
    ("Текущее время, пожалуйста", "time", None),
    ("Сколько на часах?", "time", None),
    ("Во сколько сейчас время в Москве?", "time", None),
    ("Курс доллара?", "currency", "USD"),
    ("Какой курс евро к рублю?", "currency", "EUR"),
    ("Курс юаня к рублю", "currency", "CNY"),
    ("Сколько стоит доллар?", "currency", "USD"),
    ("Цена евро в рублях", "currency", "EUR"),
    ("Сколько сейчас стоит евро", "currency", "EUR"),
    ("Доллар сколько стоит", "currency", "USD"),
    ("Курс валют", "currency", None),
    ("Курс ЦБ на сегодня", "currency", None),
    ("Какой сегодня курс валют?", "currency", None),
    ("Цена акции Сбер", "stock", "SBER"),
    ("Сколько стоит Газпром", "stock", "GAZP"),
    ("Котировки Яндекса", "stock", "YDEX"),
    ("Акции Лукойла цена", "stock", "LKOH"),
    ("Курс акций ВТБ", "stock", "VTBR"),
    ("Цена акции SBER", "stock", "SBER"),
    ("Котировки MOEX", "stock", "MOEX"),
]

#: Вопросы, которые роутер обязан оставить модели: длительность, каталог,
#: болтовня, незнакомая компания.
NEGATIVE = [
    "Сколько времени займёт доставка?",
    "Сколько времени ехать до офиса?",
    "Сколько времени ждать ответа поддержки?",
    "Сколько стоит тариф Старт?",
    "Какая цена подписки?",
    "Сколько стоит доставка?",
    "Сколько стоит возврат?",
    "Какой курс обучения?",
    "Кто такой Газпром?",
    "Расскажи про Яндекс",
    "Цена акции Ромашка",
    "Привет",
    "",
    "   ",
]


class FetchStub:
    """Заглушка сети: отдаёт фикстуру или падает, считает вызовы."""

    def __init__(self, payload=b"", error=None):
        self.payload = payload
        self.error = error
        self.calls = []

    def __call__(self, url, timeout):
        self.calls.append(url)
        if self.error is not None:
            raise self.error
        return self.payload


class IntentTest(unittest.TestCase):
    def setUp(self):
        fresh.reset_cache()


class TestDetect(IntentTest):
    def test_positive_phrases(self):
        for question, kind, param in POSITIVE:
            with self.subTest(question=question):
                found = detect(question)
                self.assertIsNotNone(found, question)
                self.assertEqual(found.kind, kind)
                self.assertEqual(found.param, param)

    def test_negative_phrases(self):
        for question in NEGATIVE:
            with self.subTest(question=question):
                self.assertIsNone(detect(question))

    def test_weekday_flag(self):
        self.assertTrue(detect("Какой сегодня день недели?").asks_weekday)
        self.assertFalse(detect("Который час?").asks_weekday)
        self.assertFalse(detect("Какое сегодня число?").asks_weekday)

    def test_phrase_is_reported(self):
        self.assertIn("курс", detect("Какой курс евро?").phrase.lower())
        self.assertIn("акци", detect("Цена акции Сбер").phrase.lower())

    def test_params_property(self):
        self.assertEqual(Intent("currency", ("USD",)).param, "USD")
        self.assertIsNone(Intent("time").param)

    def test_two_currencies_are_both_remembered(self):
        found = detect("Курс доллара и евро")
        self.assertIsNotNone(found)
        self.assertEqual(found.params, ("USD", "EUR"))
        self.assertEqual(found.param, "USD")

    def test_stock_wins_over_currency_word(self):
        # «Курс акций Сбера» — биржа, а не курс ЦБ.
        found = detect("Курс акций Сбера")
        self.assertIsNotNone(found)
        self.assertEqual(found.kind, "stock")
        self.assertEqual(found.param, "SBER")


class TestFactLine(IntentTest):
    def test_no_intent_is_none(self):
        stub = FetchStub(CBR_XML)
        self.assertIsNone(fact_line("Привет", fetch=stub))
        self.assertEqual(stub.calls, [], "без намерения сеть не нужна")

    def test_time_line(self):
        line = fact_line("Который час?", now=NOW)
        self.assertEqual(
            line, "[Факт: 26 ноября 2026, 15:30 MSK (источник: часы сервера)]"
        )

    def test_time_weekday_line(self):
        line = fact_line("Какой сегодня день недели?", now=NOW)
        self.assertTrue(line.startswith("[Факт: четверг, 26 ноября 2026"))

    def test_currency_line(self):
        line = fact_line("Курс доллара", fetch=FetchStub(CBR_XML))
        self.assertEqual(
            line, "[Факт: 1 USD = 82,3130 ₽ (источник: ЦБ РФ на 26.11.2026)]"
        )

    def test_currency_without_param_gives_main_three(self):
        stub = FetchStub(CBR_XML)
        line = fact_line("Курс валют", fetch=stub)
        self.assertEqual(len(line.split("\n")), 3)
        self.assertIn("1 USD", line)
        self.assertIn("1 EUR", line)
        self.assertIn("1 CNY", line)
        self.assertEqual(len(stub.calls), 1, "все валюты — один запрос к ЦБ")

    def test_stock_line(self):
        line = fact_line("Цена акции Сбер", fetch=FetchStub(MOEX_JSON))
        self.assertIn("SBER (Сбербанк) = 285,10 ₽", line)
        self.assertIn("(источник: MOEX ISS, 26.11.2026 15:14 MSK)", line)

    def test_currency_degradation_is_honest(self):
        stub = FetchStub(error=OSError("сеть недоступна"))
        line = fact_line("Курс доллара", fetch=stub)
        self.assertIsNotNone(line, "деградация — это факт, а не молчание")
        self.assertIn("не удалось получить", line)
        self.assertIn("USD", line)
        self.assertIn("ЦБ РФ", line)
        self.assertEqual(len(stub.calls), 1)

    def test_stock_degradation_is_honest(self):
        stub = FetchStub(error=TimeoutError("timed out"))
        line = fact_line("Цена акции Сбер", fetch=stub)
        self.assertIn("не удалось получить", line)
        self.assertIn("SBER", line)
        self.assertIn("MOEX", line)

    def test_build_facts_returns_fact_objects(self):
        facts = build_facts("Который час?", now=NOW)
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].kind, "time")

class StubAdapter:
    """Заглушка бэкенда стенда: запоминает запрос, отвечает одной репликой."""

    def stream(self, messages, *, max_tokens):
        self.calls.append([dict(message) for message in messages])
        yield chat_server.StreamPiece(text="ок")
        yield chat_server.StreamPiece(
            done=True,
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            finish_reason="stop",
        )

    def __init__(self):
        self.calls = []


SERVER_REGISTRY = (
    chat_server.BackendSpec(
        id="vtm",
        label="ВТМ (мок)",
        kind="vtm",
        models=(chat_server.ModelSpec(id="mock", label="Мок"),),
    ),
    chat_server.BackendSpec(
        id="ollama",
        label="Ollama (мок)",
        kind="ollama",
        models=(chat_server.ModelSpec(id="mock-qwen", label="Мок Qwen"),),
    ),
)


class TestServerWiring(IntentTest):
    """[Факт] доходит до запроса колонки через настоящий POST /api/chat."""

    def start(self, *, system_prompt="СИСТЕМА", facts=True):
        left, right = StubAdapter(), StubAdapter()
        server = chat_server.create_server(
            port=0,
            registry=SERVER_REGISTRY,
            adapters={("vtm", "mock"): left, ("ollama", "mock-qwen"): right},
            system_prompt=system_prompt,
            facts=facts,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def stop():
            server.shutdown()
            server.server_close()

        self.addCleanup(stop)
        return server, left, right

    def post(self, server, payload):
        connection = http.client.HTTPConnection(
            "127.0.0.1", server.server_address[1], timeout=10
        )
        try:
            connection.request(
                "POST",
                "/api/chat",
                body=json.dumps(payload),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            response.read()
            return response.status
        finally:
            connection.close()

    def payload(self, question="Который час?", use_system=True):
        return {
            "question": question,
            "use_system": use_system,
            "sides": [
                {"backend": "vtm", "model": "mock"},
                {"backend": "ollama", "model": "mock-qwen"},
            ],
        }

    def test_fact_reaches_both_columns(self):
        server, left, right = self.start()
        with mock.patch.object(
            chat_server, "_fresh_fact", return_value="[Факт: тест (источник: тест)]"
        ):
            status = self.post(server, self.payload())
        self.assertEqual(status, 200)
        for adapter in (left, right):
            with self.subTest(backend=adapter is left):
                messages = adapter.calls[0]
                self.assertEqual(messages[0]["role"], "system")
                self.assertIn("СИСТЕМА", messages[0]["content"])
                self.assertIn("[Факт: тест (источник: тест)]", messages[0]["content"])
                self.assertEqual(
                    messages[-1], {"role": "user", "content": "Который час?"}
                )

    def test_no_fact_when_router_silent(self):
        server, left, _right = self.start()
        with mock.patch.object(chat_server, "_fresh_fact", return_value=None):
            self.post(server, self.payload("Привет"))
        self.assertEqual(left.calls[0][0]["content"], "СИСТЕМА")

    def test_no_fact_when_system_disabled(self):
        server, left, _right = self.start()
        with mock.patch.object(
            chat_server, "_fresh_fact", return_value="[Факт: тест]"
        ) as mocked:
            self.post(server, self.payload(use_system=False))
        self.assertEqual([m["role"] for m in left.calls[0]], ["user"])
        mocked.assert_not_called()

    def test_no_fact_when_flag_disabled(self):
        server, left, _right = self.start(facts=False)
        with mock.patch.object(
            chat_server, "_fresh_fact", return_value="[Факт: тест]"
        ) as mocked:
            self.post(server, self.payload())
        self.assertEqual(left.calls[0][0]["content"], "СИСТЕМА")
        mocked.assert_not_called()

    def test_fact_is_injected_without_system_prompt(self):
        server, left, _right = self.start(system_prompt="")
        with mock.patch.object(
            chat_server, "_fresh_fact", return_value="[Факт: 15:30 MSK]"
        ):
            self.post(server, self.payload())
        messages = left.calls[0]
        self.assertEqual(
            messages[0], {"role": "system", "content": "[Факт: 15:30 MSK]"}
        )
        self.assertEqual(messages[1]["role"], "user")

    def test_router_crash_does_not_break_request(self):
        # Рантайм фактов (в ядре — реестр плагинов 0209) падает → пустая
        # строка и обычный ответ модели, а не 500.
        with mock.patch.object(
            chat_server, "default_plugin_registry", side_effect=RuntimeError("бум")
        ):
            self.assertEqual(chat_server._fresh_fact("Который час?"), "")


class TestWithFact(IntentTest):
    def test_appends_to_existing_system(self):
        messages = [
            {"role": "system", "content": "Ты — Ева."},
            {"role": "user", "content": "Который час?"},
        ]
        result = with_fact(messages, "[Факт: 15:30 MSK]")
        self.assertEqual(result[0]["role"], "system")
        self.assertEqual(result[0]["content"], "Ты — Ева.\n\n[Факт: 15:30 MSK]")
        self.assertEqual(result[1], {"role": "user", "content": "Который час?"})
        self.assertEqual(len(result), 2, "системное сообщение остаётся одно")

    def test_creates_system_when_missing(self):
        messages = [{"role": "user", "content": "Который час?"}]
        result = with_fact(messages, "[Факт: 15:30 MSK]")
        self.assertEqual(result[0], {"role": "system", "content": "[Факт: 15:30 MSK]"})
        self.assertEqual(result[1]["role"], "user")

    def test_empty_line_returns_copy(self):
        messages = [{"role": "user", "content": "привет"}]
        result = with_fact(messages, "")
        self.assertEqual(result, messages)
        self.assertIsNot(result, messages)

    def test_input_is_not_mutated(self):
        original = {"role": "system", "content": "система"}
        messages = [original]
        with_fact(messages, "[Факт: x]")
        self.assertEqual(original["content"], "система")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
