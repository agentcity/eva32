"""Покрытие новых классов вопросов и защита eval-набора.

Два смысла файла:

1. **Покрытие.** Набор реальных вопросов по новым классам (время, единицы,
   арифметика, курс) должен целиком уходить в скиллы, без модели.
2. **Защита eval-набора.** Ни один новый скилл не имеет права поймать вопрос
   из ``eval/hybrid_golden.jsonl``: там разметка маршрутов, и перехват сломал
   бы 60/60. Этот тест — сторож правки ``match``.

Запуск::

    python3 -m unittest tests.test_skills_probe -v
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eva.skills import SKILLS, try_skills  # noqa: E402
from eva.skills.clock import MOSCOW_TZ  # noqa: E402

GOLDEN = ROOT / "eval" / "hybrid_golden.jsonl"
NEW_SKILLS = ("time", "units", "math", "currency")

#: 18 ноября 2026, среда, 08:54 МСК — «сейчас» для проверок времени.
NOW = datetime(2026, 11, 18, 8, 54, tzinfo=MOSCOW_TZ)

#: Реальные вопросы, которых раньше не было ни у одного скилла.
PROBE: tuple[tuple[str, str, str], ...] = (
    ("time", "Который час?", "08:54"),
    ("time", "Сколько сейчас времени?", "московскому времени"),
    ("time", "Какое сегодня число?", "18 ноября 2026"),
    ("time", "Какой сегодня день недели?", "среда"),
    ("units", "Переведи 10 км в мили", "6,214"),
    ("units", "Сколько килограммов в 5 фунтах?", "2,268"),
    ("units", "2 гигабайта в мегабайты", "2 048"),
    ("units", "30 градусов Цельсия в Фаренгейты", "86 °F"),
    ("units", "Сколько минут в 2 часах?", "120"),
    ("math", "Сколько будет 2+2?", "4"),
    ("math", "Посчитай 12 * (3 + 4)", "84"),
    ("math", "Сколько будет 2^10?", "1 024"),
    ("math", "10 / 4", "2,5"),
    ("math", "Сколько будет 2 плюс 3?", "5"),
    ("currency", "Какой курс доллара?", "92,50"),
    ("currency", "Сколько рублей в 100 долларах?", "9 250"),
    ("currency", "Переведи 50 евро в рубли", "5 020"),
    ("currency", "100 долларов в евро", "92,13"),
)


class TestProbeCoverage(unittest.TestCase):
    def setUp(self):
        self.ctx = {"now": NOW}

    def test_every_probe_question_is_answered_by_its_skill(self):
        for skill, question, expected in PROBE:
            with self.subTest(skill=skill, question=question):
                result = try_skills(question, self.ctx)
                self.assertIsNotNone(result, "вопрос ушёл модели")
                self.assertEqual(result["skill"], skill)
                self.assertIn(expected, result["text"])

    def test_probe_share(self):
        answered = [1 for _skill, question, _ in PROBE if try_skills(question, self.ctx)]
        self.assertEqual(len(answered), len(PROBE))

    def test_new_skills_are_registered(self):
        names = [skill.name for skill in SKILLS]
        for name in NEW_SKILLS:
            with self.subTest(name=name):
                self.assertIn(name, names)

    def test_catalog_skills_still_win_their_questions(self):
        cases = (
            ("Сколько стоит тариф «Старт»?", "pricing"),
            ("Какие часы работы?", "hours"),
            ("Как вернуть деньги?", "refunds"),
            ("Ваш телефон?", "contacts"),
        )
        for question, expected in cases:
            with self.subTest(question=question):
                result = try_skills(question, self.ctx)
                self.assertIsNotNone(result)
                self.assertEqual(result["skill"], expected)


class TestGoldenSetIsNotStolen(unittest.TestCase):
    # 0105: набор вырос до 80 строк. Первые 60 — историческое ядро: новые
    # скиллы по-прежнему не имеют права его перехватывать. Последние 20 —
    # вопросы новых классов, их скиллы, наоборот, обязаны ловить.
    HISTORICAL_ROWS = 60

    @classmethod
    def setUpClass(cls):
        with open(GOLDEN, encoding="utf-8") as handle:
            cls.rows = [json.loads(line) for line in handle if line.strip()]
        cls.ctx = {"now": NOW}
        cls.historical = cls.rows[: cls.HISTORICAL_ROWS]
        cls.new_class_rows = cls.rows[cls.HISTORICAL_ROWS:]

    def test_new_class_rows_are_present(self):
        self.assertGreaterEqual(len(self.new_class_rows), 16)
        self.assertTrue(
            all(row["expect_route"] == "skill" for row in self.new_class_rows)
        )

    def test_no_new_skill_matches_historical_questions(self):
        new_skills = [skill for skill in SKILLS if skill.name in NEW_SKILLS]
        self.assertEqual(len(new_skills), len(NEW_SKILLS))
        for row in self.historical:
            for skill in new_skills:
                with self.subTest(skill=skill.name, question=row["question"]):
                    self.assertFalse(
                        skill.match(row["question"], self.ctx),
                        "новый скилл перехватил вопрос исторического набора",
                    )

    def test_router_the_same_questions_keep_their_route(self):
        # try_skills — то, что зовёт роутер: не-skill строки обязаны молчать,
        # skill-строки — уходить в старые каталожные скиллы.
        catalog_skills = {"pricing", "hours", "refunds", "contacts"}
        for row in self.historical:
            with self.subTest(question=row["question"]):
                result = try_skills(row["question"], self.ctx)
                if row["expect_route"] == "skill":
                    self.assertIsNotNone(result)
                    self.assertIn(result["skill"], catalog_skills)
                else:
                    self.assertIsNone(result)

    def test_new_class_questions_are_caught_by_skills(self):
        seen: set[str] = set()
        for row in self.new_class_rows:
            with self.subTest(question=row["question"]):
                result = try_skills(row["question"], self.ctx)
                self.assertIsNotNone(result, "вопрос нового класса ушёл модели")
                self.assertIn(result["skill"], NEW_SKILLS)
                seen.add(result["skill"])
        self.assertEqual(seen, set(NEW_SKILLS), "не все новые классы сработали")


if __name__ == "__main__":
    unittest.main()
