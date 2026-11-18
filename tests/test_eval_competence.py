"""Golden компетенций: структура и проверки. Прайс здесь не оценивается."""
from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tools.eval_competence as ec  # noqa: E402


class TestCompetenceGolden(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = ec.load_golden()

    def test_has_all_categories(self):
        cats = {r["category"] for r in self.rows}
        self.assertEqual(
            cats,
            {"reasoning", "clarify", "marketing", "writing",
             "client", "language", "behavior", "limits"},
        )

    def test_each_category_has_enough_rows(self):
        for cat in {r["category"] for r in self.rows}:
            with self.subTest(cat=cat):
                self.assertGreaterEqual(
                    len([r for r in self.rows if r["category"] == cat]), 5
                )

    def test_checks_are_known(self):
        for row in self.rows:
            for name in row.get("checks", {}):
                with self.subTest(name=name):
                    self.assertIn(name, ec.KNOWN_CHECKS)

    def test_no_catalog_prices_in_questions(self):
        """Golden — про умение: конкретных цен каталога в вопросах быть не должно."""
        numbers = ec.catalog_numbers()
        for row in self.rows:
            for number in numbers:
                with self.subTest(question=row["question"][:40], number=number):
                    self.assertNotIn(number, row["question"])

    def test_questions_are_unique(self):
        questions = [r["question"] for r in self.rows]
        self.assertEqual(len(set(questions)), len(questions))


class TestCompetenceChecks(unittest.TestCase):
    def _row(self, **checks):
        return {"question": "тест", "category": "reasoning", "checks": checks}

    def test_clarify_requires_question(self):
        bad = ec.check_row(self._row(asks_question=True, min_words=3), "Конечно, сделаю.")
        self.assertIn("не задан уточняющий вопрос", bad)
        good = ec.check_row(self._row(asks_question=True, min_words=3),
                            "Уточните, пожалуйста: для какой аудитории?")
        self.assertEqual(good, [])

    def test_invented_price_is_caught(self):
        bad = ec.check_row(self._row(no_invented_prices=True), "Услуга стоит 5 000 ₽.")
        self.assertIn("выдуманная цена", bad)

    def test_catalog_numbers_are_caught(self):
        numbers = ec.catalog_numbers()
        number = sorted(numbers)[0]
        bad = ec.check_row(self._row(forbid_numbers_from_catalog=True),
                           f"Тариф стоит {number} рублей.", numbers=numbers)
        self.assertTrue(any("цены из каталога" in f for f in bad))

    def test_language_check(self):
        bad = ec.check_row(self._row(min_words=2, max_words=5), "одно")
        self.assertTrue(any("коротко" in f for f in bad))

    def test_must_not_contain(self):
        bad = ec.check_row(self._row(must_not_contain=["OpenAI"]),
                           "Я работаю на OpenAI.")
        self.assertIn("есть запрещённое «OpenAI»", bad)

    def test_evaluate_counts(self):
        rows = [
            {"question": "a", "category": "clarify", "checks": {"asks_question": True}},
            {"question": "b", "category": "clarify", "checks": {"asks_question": True}},
        ]
        report = ec.evaluate(rows, {"a": "Уточните детали?"})
        self.assertEqual(report["per_category"]["clarify"]["прошли"], 1)
        self.assertEqual(len(report["failures"]), 1)

    def test_missing_answer_is_failure(self):
        report = ec.evaluate([{"question": "x", "category": "reasoning", "checks": {}}], {})
        self.assertEqual(report["failures"][0]["failures"], ["нет ответа"])
