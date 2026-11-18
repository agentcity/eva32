"""Проверки скилла ``currency``: курс по встроенному справочнику, без сети.

Запуск::

    python3 -m unittest tests.test_skills_currency -v
"""

from __future__ import annotations

import inspect
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eva.skills import try_skills  # noqa: E402
from eva.skills.currency import RATES_DATE, RATES_RUB, CurrencySkill  # noqa: E402


class TestCurrencySkill(unittest.TestCase):
    def setUp(self):
        self.skill = CurrencySkill()
        self.ctx = {}

    def run_skill(self, text):
        return self.skill.run(text, self.ctx)

    def test_reference_is_sane(self):
        self.assertEqual(RATES_RUB["RUB"], 1.0)
        self.assertRegex(RATES_DATE, r"^\d{4}-\d{2}-\d{2}$")
        for code, rate in RATES_RUB.items():
            with self.subTest(code=code):
                self.assertGreater(rate, 0.0)

    def test_match(self):
        for question in (
            "Какой курс доллара?",
            "Курс евро к рублю?",
            "Сколько рублей в 100 долларах?",
            "Переведи 50 евро в рубли",
            "100 долларов в евро",
            "$100 в рублях",
        ):
            with self.subTest(question=question):
                self.assertTrue(self.skill.match(question, self.ctx))

    def test_usd_rate(self):
        result = self.run_skill("Какой курс доллара?")
        self.assertEqual(result["skill"], "currency")
        self.assertEqual(result["sources"], [])
        self.assertIn("1 USD = 92,50 ₽", result["text"])
        self.assertIn("18.11.2026", result["text"])

    def test_eur_rate(self):
        result = self.run_skill("Курс евро к рублю?")
        self.assertIn("1 EUR = 100,40 ₽", result["text"])

    def test_usd_to_rub(self):
        result = self.run_skill("Сколько рублей в 100 долларах?")
        self.assertIn("100 USD = 9 250 ₽", result["text"])
        self.assertIn("не онлайн-курс", result["text"])

    def test_eur_to_rub(self):
        result = self.run_skill("Переведи 50 евро в рубли")
        self.assertIn("50 EUR = 5 020 ₽", result["text"])

    def test_cross_rate(self):
        result = self.run_skill("100 долларов в евро")
        self.assertIn("92,13 EUR", result["text"])

    def test_dollar_sign(self):
        result = self.run_skill("$100 в рублях")
        self.assertIn("9 250", result["text"])

    def test_deterministic(self):
        first = self.run_skill("Сколько рублей в 100 долларах?")
        second = self.run_skill("Сколько рублей в 100 долларах?")
        self.assertEqual(first, second)

    def test_rubles_only_is_silent(self):
        for question in (
            "Сколько рублей стоит тариф «Старт»?",
            "Сколько будет 100 рублей?",
            "Сколько стоит доставка?",
        ):
            with self.subTest(question=question):
                self.assertFalse(self.skill.match(question, self.ctx))

    def test_no_amount_no_currency_is_silent(self):
        for question in (
            "100 долларов",
            "Сколько будет 2+2?",
            "Переведи 10 км в мили",
            "Который час?",
        ):
            with self.subTest(question=question):
                self.assertFalse(self.skill.match(question, self.ctx))

    def test_catalog_questions_are_not_ours(self):
        for question in (
            "Сколько стоит тариф «Бизнес»?",
            "Сколько стоит доставка в долларах?",
            "Можно ли вернуть деньги, если платил картой?",
        ):
            with self.subTest(question=question):
                self.assertFalse(self.skill.match(question, self.ctx))

    def test_registry(self):
        result = try_skills("Какой курс доллара?", self.ctx)
        self.assertIsNotNone(result)
        self.assertEqual(result["skill"], "currency")

    def test_no_model_imports(self):
        import eva.skills.currency as module

        source = inspect.getsource(module).lower()
        for forbidden in ("eva.clients", "openai", "requests", "urllib", "http.client"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
