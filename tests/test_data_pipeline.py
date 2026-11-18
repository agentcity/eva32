"""Тесты конвейера данных: фильтр качества, генераторы вопросов, смесь."""
from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from distill import quality  # noqa: E402
import distill.gen_business_questions as gb  # noqa: E402
import tools.build_mixture as bm  # noqa: E402


class TestQuality(unittest.TestCase):
    def test_accepts_neutral_russian(self):
        self.assertTrue(quality.accept_answer(
            "Тариф «Старт» стоит 990 ₽ в месяц и подходит небольшой команде до трёх человек."
        ))

    def test_rejects_bad_content(self):
        for text in [
            "Трамп и Байден снова спорят о выборах в конгрессе.",
            "Это блядская ситуация, так делать нельзя.",
            "Импортировать numpy как np, определение is_string(value).",
            "Дозировка лекарства и симптом болезни — обратитесь к врачу.",
        ]:
            with self.subTest(text=text[:30]):
                self.assertFalse(quality.accept_answer(text))

    def test_rejects_non_russian(self):
        self.assertFalse(
            quality.accept_answer("This is just an English sentence without Cyrillic.")
        )

    def test_report_counts(self):
        rows = [
            ("Как настроить тариф?", "Тариф настраивается в личном кабинете за пару минут."),
            ("Кто такой Трамп?", "Трамп — политик, обсуждал выборы и конгресс."),
        ]
        rep = quality.report(rows)
        self.assertEqual(rep["всего"], 2)
        self.assertEqual(rep["прошли"], 1)
        self.assertGreaterEqual(rep.get("политика/религия", 0), 1)


class TestBusinessQuestions(unittest.TestCase):
    def test_topic_bank_is_diverse(self):
        rows = gb.build(2000, seed=1)
        topics = {r["meta"]["topic"] for r in rows}
        self.assertGreaterEqual(len(topics), 10)
        self.assertEqual(len({r["question"] for r in rows}), len(rows))

    def test_catalog_questions_have_context(self):
        rows = gb.build_from_catalog(300, seed=1)
        self.assertTrue(rows, "каталог должен давать вопросы")
        for row in rows:
            self.assertTrue(row.get("context"), "грунтованный вопрос без контекста")
            self.assertIn("[1]", row["context"])
            self.assertTrue(row["meta"]["grounded"])

    def test_catalog_question_mentions_item(self):
        rows = gb.build_from_catalog(50, seed=2)
        names = {c["name"] for c in gb.load_catalog()}
        self.assertTrue(any(any(n in r["question"] for n in names) for r in rows))


class TestMixture(unittest.TestCase):
    def test_dedupe_and_layers(self):
        pairs, stats = bm.build(500, seed=1)
        self.assertTrue(pairs, "смесь не должна быть пустой при наличии данных")
        keys = [bm._key(p.question) for p in pairs]
        self.assertEqual(len(set(keys)), len(keys), "дубликаты вопросов в смеси")
        self.assertIn("code", stats)

    def test_split_no_overlap(self):
        pairs, _ = bm.build(200, seed=1)
        train, val = bm.split(pairs, 0.1)
        self.assertFalse({p.question for p in train} & {p.question for p in val})

    def test_weights_respected_with_enough_data(self):
        pairs, stats = bm.build(1000, seed=1)
        self.assertTrue(any(p.layer == "code" for p in pairs))
        self.assertEqual(stats["code"]["нужно"], int(1000 * bm.LAYERS["code"][0]))


if __name__ == "__main__":
    unittest.main()
