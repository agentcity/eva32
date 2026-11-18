"""Проверки скилла ``units``: конвертация единиц без сети и модели.

Запуск::

    python3 -m unittest tests.test_skills_units -v
"""

from __future__ import annotations

import inspect
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eva.skills import try_skills  # noqa: E402
from eva.skills.units import UnitsSkill  # noqa: E402


class TestUnitsSkill(unittest.TestCase):
    def setUp(self):
        self.skill = UnitsSkill()
        self.ctx = {}

    def run_skill(self, text):
        return self.skill.run(text, self.ctx)

    def test_match(self):
        for question in (
            "Переведи 10 км в мили",
            "5 кг в фунтах",
            "2 гигабайта в мегабайты",
            "30 градусов Цельсия в Фаренгейты",
            "Сколько минут в 2 часах?",
            "1,5 км в метры",
            "10 миль в километрах",
        ):
            with self.subTest(question=question):
                self.assertTrue(self.skill.match(question, self.ctx))

    def test_km_to_miles(self):
        result = self.run_skill("Переведи 10 км в мили")
        self.assertEqual(result["skill"], "units")
        self.assertEqual(result["sources"], [])
        self.assertIn("10 км", result["text"])
        self.assertIn("6,214", result["text"])
        self.assertIn("мили", result["text"])

    def test_kg_to_pounds(self):
        result = self.run_skill("5 кг в фунтах")
        self.assertIn("11,023", result["text"])
        self.assertIn("фунта", result["text"])

    def test_gigabytes_to_megabytes(self):
        result = self.run_skill("2 гигабайта в мегабайты")
        self.assertIn("2 048", result["text"])
        self.assertIn("МБ", result["text"])

    def test_celsius_to_fahrenheit(self):
        result = self.run_skill("30 градусов Цельсия в Фаренгейты")
        self.assertIn("30 °C", result["text"])
        self.assertIn("86 °F", result["text"])

    def test_negative_celsius(self):
        result = self.run_skill("-5 градусов Цельсия в Фаренгейты")
        self.assertIn("-5 °C", result["text"])
        self.assertIn("23 °F", result["text"])

    def test_thousand_separator(self):
        result = self.run_skill("1,5 км в метры")
        self.assertIn("1 500 м", result["text"])

    def test_integer_plural(self):
        result = self.run_skill("1000 м в километрах")
        self.assertIn("1 км", result["text"])

    def test_reversed_order(self):
        result = self.run_skill("Сколько килограммов в 5 фунтах?")
        self.assertIn("кг", result["text"])
        self.assertIn("2,268", result["text"])

    def test_deterministic(self):
        first = self.run_skill("Переведи 10 км в мили")
        second = self.run_skill("Переведи 10 км в мили")
        self.assertEqual(first, second)

    def test_dimension_mismatch_is_silent(self):
        for question in (
            "10 км в час",
            "5 кг в литрах",
            "2 часа в километрах",
            "30 градусов в мили",
        ):
            with self.subTest(question=question):
                self.assertFalse(self.skill.match(question, self.ctx))

    def test_unknown_unit_is_silent(self):
        self.assertFalse(self.skill.match("10 паросов в мили", self.ctx))
        self.assertFalse(self.skill.match("5 попугаев в кг", self.ctx))

    def test_one_unit_is_not_enough(self):
        for question in (
            "Сколько стоит доставка 10 кг?",
            "10 километров",
            "Который час?",
        ):
            with self.subTest(question=question):
                self.assertFalse(self.skill.match(question, self.ctx))

    def test_catalog_questions_are_not_ours(self):
        for question in (
            "Сколько стоит доставка документов по Москве?",
            "Сколько дней даётся на возврат денег?",
            "Сколько стоит тариф «Профи»?",
        ):
            with self.subTest(question=question):
                self.assertFalse(self.skill.match(question, self.ctx))

    def test_registry(self):
        result = try_skills("Переведи 10 км в мили", self.ctx)
        self.assertIsNotNone(result)
        self.assertEqual(result["skill"], "units")

    def test_no_model_imports(self):
        import eva.skills.units as module

        source = inspect.getsource(module).lower()
        for forbidden in ("eva.clients", "openai", "requests", "urllib", "http.client"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
