"""Проверки скилла ``time``: время и дата без модели и сети.

Запуск::

    python3 -m unittest tests.test_skills_time -v
"""

from __future__ import annotations

import inspect
import os
import sys
import unittest
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eva.skills import try_skills  # noqa: E402
from eva.skills.clock import MOSCOW_TZ, TimeSkill  # noqa: E402

#: 18 ноября 2026 — среда.
NOW = datetime(2026, 11, 18, 8, 54, tzinfo=MOSCOW_TZ)


class TestTimeSkill(unittest.TestCase):
    def setUp(self):
        self.skill = TimeSkill()
        self.ctx = {"now": NOW}

    def run_skill(self, text):
        return self.skill.run(text, self.ctx)

    def test_match(self):
        for question in (
            "Который час?",
            "Сколько сейчас времени?",
            "Подскажи точное время.",
            "Какое сегодня число?",
            "Какая сегодня дата?",
            "Какой сегодня день недели?",
            "Какое сегодня число и который час?",
        ):
            with self.subTest(question=question):
                self.assertTrue(self.skill.match(question, self.ctx))

    def test_time_answer(self):
        result = self.run_skill("Который час?")
        self.assertEqual(result["skill"], "time")
        self.assertEqual(result["sources"], [])
        self.assertIn("08:54", result["text"])
        self.assertIn("московскому времени", result["text"])

    def test_date_answer(self):
        result = self.run_skill("Какое сегодня число?")
        self.assertIn("среда", result["text"])
        self.assertIn("18 ноября 2026 года", result["text"])
        self.assertNotIn("Сейчас", result["text"])

    def test_date_and_time_together(self):
        result = self.run_skill("Какое сегодня число и который час?")
        self.assertIn("18 ноября 2026", result["text"])
        self.assertIn("08:54", result["text"])

    def test_now_from_iso_string(self):
        ctx = {"now": "2026-11-18T08:54:00+03:00"}
        result = self.skill.run("Который час?", ctx)
        self.assertIn("08:54", result["text"])

    def test_now_from_request_meta(self):
        # Роутер кладёт meta запроса в ctx; eval передаёт там фиксированное
        # «сейчас», иначе вопросы о времени невоспроизводимы.
        result = self.skill.run("Который час?", {"meta": {"now": NOW}})
        self.assertIn("08:54", result["text"])

    def test_naive_now_is_moscow(self):
        ctx = {"now": datetime(2026, 11, 18, 8, 54)}
        result = self.skill.run("Который час?", ctx)
        self.assertIn("08:54", result["text"])

    def test_deterministic(self):
        first = self.run_skill("Который час?")
        second = self.run_skill("Который час?")
        self.assertEqual(first, second)

    def test_does_not_steal_duration_questions(self):
        for question in (
            "Сколько времени занимает настройка CRM?",
            "Сколько времени идёт доставка?",
            "Сколько осталось ждать?",
            "Сколько времени вы работаете?",
        ):
            with self.subTest(question=question):
                self.assertFalse(self.skill.match(question, self.ctx))

    def test_does_not_steal_other_skills(self):
        for question in (
            "Какие часы работы офиса?",
            "Сколько стоит тариф «Старт»?",
            "Как вернуть деньги?",
            "Какой у вас телефон?",
            "В каком году вышел Python 3.0?",
            "Какая столица Австралии?",
        ):
            with self.subTest(question=question):
                self.assertFalse(self.skill.match(question, self.ctx))

    def test_registry(self):
        result = try_skills("Который час?", self.ctx)
        self.assertIsNotNone(result)
        self.assertEqual(result["skill"], "time")

    def test_no_model_imports(self):
        import eva.skills.clock as module

        source = inspect.getsource(module).lower()
        for forbidden in ("eva.clients", "openai", "requests", "urllib", "http.client"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
