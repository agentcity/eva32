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


class TestTranslationChecks(unittest.TestCase):
    """Проверки перевода — без сети: учитель подменяется готовыми строками."""

    def _check(self, q_en, a_en, q_ru, a_ru):
        from tools.translate_pairs import _check
        return _check(q_en, a_en, q_ru, a_ru)

    def test_accepts_good_translation(self):
        reason = self._check(
            "Write a function that reverses a string.",
            "```python\ndef rev(s):\n    return s[::-1]\n```\nThis works by slicing.",
            "Напиши функцию, которая переворачивает строку.",
            "```python\ndef rev(s):\n    return s[::-1]\n```\nЭто работает за счёт среза.",
        )
        self.assertIsNone(reason)

    def test_rejects_untranslated_question(self):
        reason = self._check("Write a function.", "Some text here.",
                             "Write a function.", "Какой-то текст здесь, вполне русский.")
        self.assertEqual(reason, "вопрос не переведён")

    def test_rejects_lost_code_block(self):
        reason = self._check(
            "Write code.",
            "```python\nprint(1)\n```\nExplanation.",
            "Напиши код на Python, пожалуйста.",
            "Вот объяснение без блока кода, но по-русски и достаточно длинное.",
        )
        self.assertIn(reason, {"потеряны блоки кода", "ответ не переведён"})

    def test_rejects_chatty_answer(self):
        reason = self._check(
            "Write code.",
            "Explanation of the code goes here in English.",
            "Напиши код на Python.",
            "Вот перевод: объяснение кода находится здесь и оно достаточно длинное, "
            "чтобы пройти проверку на кириллицу.",
        )
        self.assertEqual(reason, "болтовня вокруг перевода")

    def test_parse_format(self):
        from tools.translate_pairs import _parse
        got = _parse("ВОПРОС: Как дела?\n\nОТВЕТ: Хорошо.")
        self.assertEqual(got, ("Как дела?", "Хорошо."))


class TestSoften(unittest.TestCase):
    def test_no_profanity_no_change(self):
        for text in [
            "```python\ndef f(x):\n    if x:\n        return 1\n    return 0\n```",
            "Обычный ответ  с  двумя  пробелами.",
            "  отступ в начале и в конце  ",
        ]:
            with self.subTest(text=text[:25]):
                self.assertEqual(quality.soften(text), text)

    def test_softens_profanity(self):
        cases = {
            "Это блядская ситуация.": "Это неприемлемая ситуация.",
            "Пиздец, всё упало.": "Неприятность, всё упало.",
            "Это полная хуйня.": "Это полная ерунда.",
            "Дерьмовый код.": "Негодный код.",
        }
        for src, want in cases.items():
            with self.subTest(src=src):
                self.assertEqual(quality.soften(src), want)

    def test_softened_text_is_clean(self):
        for src in ["Это блядская ситуация.", "Пиздец, всё упало.", "Сука, надоело."]:
            with self.subTest(src=src):
                self.assertTrue(quality.is_clean_language(quality.soften(src)))

    def test_code_indentation_survives(self):
        text = "Блядская ошибка:\n```python\nif x:\n    return 1\n```"
        out = quality.soften(text)
        self.assertIn("    return 1", out)
        self.assertNotIn("бляд", out.lower())
