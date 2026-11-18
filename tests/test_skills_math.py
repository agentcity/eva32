"""Проверки скилла ``math``: арифметика без модели, сети и ``eval``.

Запуск::

    python3 -m unittest tests.test_skills_math -v
"""

from __future__ import annotations

import inspect
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eva.skills import try_skills  # noqa: E402
from eva.skills.calculator import MathSkill  # noqa: E402


class TestMathSkill(unittest.TestCase):
    def setUp(self):
        self.skill = MathSkill()
        self.ctx = {}

    def run_skill(self, text):
        return self.skill.run(text, self.ctx)

    def test_match(self):
        for question in (
            "Сколько будет 2+2?",
            "Вычисли 12 * (3 + 4)",
            "Посчитай 10 / 4",
            "Сколько будет 2^10?",
            "2+2",
            "6×7",
            "Сколько будет 2,5 * 4?",
            "Сколько будет 2 плюс 3?",
        ):
            with self.subTest(question=question):
                self.assertTrue(self.skill.match(question, self.ctx))

    def test_addition(self):
        result = self.run_skill("Сколько будет 2+2?")
        self.assertEqual(result["skill"], "math")
        self.assertEqual(result["sources"], [])
        self.assertEqual(result["text"], "2+2 = 4.")

    def test_parentheses(self):
        result = self.run_skill("Вычисли 12 * (3 + 4)")
        self.assertIn("84", result["text"])

    def test_division(self):
        result = self.run_skill("Сколько будет 10 / 4?")
        self.assertIn("2,5", result["text"])

    def test_power(self):
        result = self.run_skill("Сколько будет 2^10?")
        self.assertIn("1 024", result["text"])

    def test_unicode_operators(self):
        result = self.run_skill("6×7")
        self.assertIn("42", result["text"])

    def test_word_operators(self):
        result = self.run_skill("Сколько будет 2 плюс 3?")
        self.assertIn("5", result["text"])

    def test_decimal_comma(self):
        result = self.run_skill("Сколько будет 2,5 * 4?")
        self.assertIn("10", result["text"])

    def test_division_by_zero_is_answered(self):
        result = self.run_skill("Сколько будет 5/0?")
        self.assertIsNotNone(result)
        self.assertEqual(result["skill"], "math")
        self.assertIn("деление на ноль", result["text"])

    def test_deterministic(self):
        first = self.run_skill("Сколько будет 2+2?")
        second = self.run_skill("Сколько будет 2+2?")
        self.assertEqual(first, second)

    def test_percent_questions_are_not_ours(self):
        # Решение ai/tasks/0091: проценты остаются большой модели, иначе
        # ломается разметка eval-набора (17% от 4800 — маршрут big).
        for question in (
            "Сколько будет 15% от 800?",
            "Посчитай 17% от 4800.",
        ):
            with self.subTest(question=question):
                self.assertFalse(self.skill.match(question, self.ctx))

    def test_catalog_and_other_skills_are_not_ours(self):
        for question in (
            "Сколько стоит тариф «Старт»?",
            "Сколько сотрудников на тарифе «Бизнес»?",
            "Какие часы работы офиса?",
            "Переведи 10 км в мили",
            "Какой курс доллара?",
            "В каком году вышел Python 3.0?",
            "Какая столица Австралии?",
        ):
            with self.subTest(question=question):
                self.assertFalse(self.skill.match(question, self.ctx))

    def test_date_like_text_is_not_an_expression(self):
        for question in (
            "Что было 18/11/2026?",
            "Версия 1.2.3",
            "Позвони 8-800-555-35-35",
        ):
            with self.subTest(question=question):
                self.assertFalse(self.skill.match(question, self.ctx))

    def test_too_large_power_is_silent(self):
        self.assertFalse(self.skill.match("Сколько будет 10^13?", self.ctx))

    def test_registry(self):
        result = try_skills("Сколько будет 2+2?", self.ctx)
        self.assertIsNotNone(result)
        self.assertEqual(result["skill"], "math")

    def test_no_eval_and_no_network(self):
        import eva.skills.calculator as module

        source = inspect.getsource(module).lower()
        for forbidden in ("eva.clients", "openai", "requests", "urllib", "http.client"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)
        for forbidden in ("eval(", "exec("):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
