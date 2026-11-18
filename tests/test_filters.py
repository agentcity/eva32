"""Проверки фильтров дистилляции. Запуск: python3 -m unittest discover -s tests -v"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distill import filters  # noqa: E402


class TestForeign(unittest.TestCase):
    def test_cjk(self):
        self.assertTrue(filters.has_foreign("Стоимость 990 ₽/мес. 这是最好的方案。"))

    def test_long_latin(self):
        self.assertTrue(filters.has_foreign("This answer is written in English only for testing"))

    def test_normal_russian(self):
        self.assertFalse(filters.has_foreign("Тариф «Старт» стоит 990 ₽ в месяц."))

    def test_short_latin_ok(self):
        self.assertFalse(filters.has_foreign("Работает через API и CRM."))


class TestPrice(unittest.TestCase):
    def test_rubles(self):
        self.assertTrue(filters.has_invented_price("Тариф стоит 990 ₽ в месяц.", "Сколько стоит?"))

    def test_word_rub(self):
        self.assertTrue(filters.has_invented_price("Цена: 1 500 рублей.", "Сколько?"))

    def test_without_price(self):
        self.assertFalse(filters.has_invented_price("Не нашла в данных. Уточните услугу.", "Сколько?"))

    def test_date_is_not_price(self):
        self.assertFalse(filters.has_invented_price("Возврат в течение 14 дней.", "Условия?"))


class TestCheck(unittest.TestCase):
    def test_good_pair(self):
        q = "Как отменить подписку?"
        a = "Отписаться можно в разделе «Тарифы»: выберите план и нажмите «Отменить подписку»."
        self.assertEqual(filters.check(q, a), [])

    def test_short_answer(self):
        self.assertIn("length", filters.check("Вопрос?", "Да."))

    def test_combined(self):
        a = "Стоимость 990 ₽. 这是最好的方案，建议购买。"
        problems = filters.check("Сколько стоит?", a)
        self.assertIn("foreign", problems)
        self.assertIn("invented_price", problems)


class TestDedupe(unittest.TestCase):
    def test_same_question_different_case(self):
        pairs = [{"question": "Сколько стоит?"}, {"question": "сколько стоит ?"},
                 {"question": "Как оплатить?"}]
        self.assertEqual(len(filters.dedupe(pairs)), 2)


if __name__ == "__main__":
    unittest.main()
