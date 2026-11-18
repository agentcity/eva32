"""Проверки скиллов EVA32. Запуск: python3 -m unittest tests.test_skills -v"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eva.catalog import DEFAULT_PATH, Catalog, load  # noqa: E402
from eva.skills import (  # noqa: E402
    ContactsSkill,
    HoursSkill,
    PricingSkill,
    RefundsSkill,
    try_skills,
)

ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = ROOT / "data" / "catalog.json"


class TestCatalogData(unittest.TestCase):
    """Сам каталог: 10–15 позиций, цены, лимиты, заметки, источники."""

    @classmethod
    def setUpClass(cls):
        cls.catalog = load(CATALOG_PATH)

    def test_default_path(self):
        self.assertEqual(DEFAULT_PATH.resolve(), CATALOG_PATH.resolve())

    def test_size(self):
        self.assertGreaterEqual(len(self.catalog), 10)
        self.assertLessEqual(len(self.catalog), 15)

    def test_item_shape(self):
        for item in self.catalog.all():
            with self.subTest(item=item.get("id")):
                self.assertIn(item["kind"], ("tariff", "service"))
                self.assertTrue(item["name"])
                self.assertIsInstance(item["price_rub"], (int, float))
                self.assertGreaterEqual(item["price_rub"], 0)
                self.assertIsInstance(item["limits"], list)
                self.assertIsInstance(item["notes"], list)
                self.assertTrue(item["id"])

    def test_sources_are_bracketed(self):
        for item in self.catalog.all():
            with self.subTest(item=item["id"]):
                self.assertTrue(item["sources"])
                for source in item["sources"]:
                    self.assertRegex(source, r"^\[\d+\]$")

    def test_ids_unique(self):
        ids = [item["id"] for item in self.catalog.all()]
        self.assertEqual(len(ids), len(set(ids)))

    def test_prices_match_known_catalog(self):
        by_id = {item["id"]: item for item in self.catalog.all()}
        expected = {
            "start": 990,
            "pro": 2490,
            "business": 7900,
            "site-audit": 12000,
            "crm-setup": 34000,
            "social": 19500,
            "ads": 25000,
            "landing": 48000,
            "site-support": 6000,
            "delivery-msk": 490,
            "delivery-mkad": 790,
            "delivery-russia": 1200,
            "extra-seat": 290,
        }
        for item_id, price in expected.items():
            with self.subTest(item=item_id):
                self.assertEqual(by_id[item_id]["price_rub"], price)


class TestCatalogLookup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load(CATALOG_PATH)

    def test_name(self):
        self.assertEqual(self.catalog.get("Старт")["id"], "start")

    def test_synonym(self):
        self.assertEqual(self.catalog.get("тариф Старт")["id"], "start")

    def test_case_quotes_and_yo(self):
        self.assertEqual(self.catalog.get("«СТАРТ»")["id"], "start")
        self.assertEqual(self.catalog.get("услуга «Аудит сайта»")["id"], "site-audit")

    def test_synonym_with_latin(self):
        self.assertEqual(self.catalog.get("СММ")["id"], "social")

    def test_unknown(self):
        self.assertIsNone(self.catalog.get("Марс"))
        self.assertIsNone(self.catalog.get(""))
        self.assertIsNone(self.catalog.get(None))

    def test_all_returns_copies(self):
        items = self.catalog.all()
        items[0]["price_rub"] = -1
        self.assertEqual(self.catalog.all()[0]["price_rub"], 990)

    def test_mentioned(self):
        ids = [item["id"] for item in self.catalog.mentioned("сколько стоит старт")]
        self.assertEqual(ids, ["start"])


class TestCatalogSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load(CATALOG_PATH)

    def test_search_by_name(self):
        ids = [item["id"] for item in self.catalog.search("Сколько стоит CRM?")]
        self.assertIn("crm-setup", ids)

    def test_search_delivery(self):
        ids = [item["id"] for item in self.catalog.search("доставка")]
        self.assertEqual(
            {"delivery-msk", "delivery-mkad", "delivery-russia"}, set(ids)
        )

    def test_search_mkad(self):
        ids = [item["id"] for item in self.catalog.search("МКАД")]
        self.assertIn("delivery-mkad", ids)

    def test_search_empty(self):
        self.assertEqual(self.catalog.search(""), [])
        self.assertEqual(self.catalog.search("часы работы"), [])

    def test_search_returns_copies(self):
        result = self.catalog.search("Старт")
        self.assertTrue(result)
        result[0]["price_rub"] = -1
        self.assertEqual(self.catalog.get("Старт")["price_rub"], 990)


class TestCatalogLoad(unittest.TestCase):
    def _write(self, directory: str, payload) -> Path:
        path = Path(directory) / "catalog.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    def test_minimal_item_without_id(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(
                directory,
                [
                    {
                        "kind": "service",
                        "name": "Тестовая услуга",
                        "price_rub": 123,
                        "sources": ["[1]"],
                        "limits": "один раз",
                        "notes": [],
                    }
                ],
            )
            catalog = load(path)
            item = catalog.get("Тестовая услуга")
            self.assertIsNotNone(item)
            self.assertEqual(item["price_rub"], 123)
            self.assertEqual(item["limits"], ["один раз"])
            self.assertTrue(item["id"])

    def test_string_synonyms(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(
                directory,
                [
                    {
                        "id": "x",
                        "kind": "tariff",
                        "name": "Икс",
                        "synonyms": "икс-тариф",
                        "price_rub": 1,
                        "sources": ["[7]"],
                    }
                ],
            )
            catalog = load(path)
            self.assertEqual(catalog.get("икс-тариф")["id"], "x")
            self.assertEqual(catalog.get("Икс")["id"], "x")

    def test_missing_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(
                directory,
                [{"kind": "service", "name": "Без источника", "price_rub": 1}],
            )
            with self.assertRaises(ValueError):
                load(path)

    def test_bad_kind(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(
                directory,
                [
                    {
                        "kind": "promo",
                        "name": "Промо",
                        "price_rub": 1,
                        "sources": ["[1]"],
                    }
                ],
            )
            with self.assertRaises(ValueError):
                load(path)

    def test_negative_price(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(
                directory,
                [
                    {
                        "kind": "service",
                        "name": "Минус",
                        "price_rub": -1,
                        "sources": ["[1]"],
                    }
                ],
            )
            with self.assertRaises(ValueError):
                load(path)

    def test_duplicate_id(self):
        with tempfile.TemporaryDirectory() as directory:
            item = {
                "id": "same",
                "kind": "service",
                "name": "Один",
                "price_rub": 1,
                "sources": ["[1]"],
            }
            path = self._write(directory, [item, dict(item, name="Два")])
            with self.assertRaises(ValueError):
                load(path)

    def test_breaks_on_broken_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaises(ValueError):
                load(path)

    def test_empty_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, [])
            with self.assertRaises(ValueError):
                load(path)


class SkillTest(unittest.TestCase):
    """Общая обвязка: скиллы получают каталог через ctx."""

    @classmethod
    def setUpClass(cls):
        cls.catalog = load(CATALOG_PATH)

    def ctx(self):
        return {"catalog": self.catalog}


class TestPricing(SkillTest):
    def setUp(self):
        self.skill = PricingSkill()

    def run_skill(self, text):
        return self.skill.run(text, self.ctx())

    def test_answer_start_without_model(self):
        result = self.run_skill("сколько стоит тариф Старт")
        self.assertIsNotNone(result)
        self.assertEqual(result["skill"], "pricing")
        self.assertEqual(result["catalog_ids"], ["start"])
        self.assertEqual(result["sources"], ["[1]"])
        self.assertIn("990", result["text"])
        self.assertIn("₽/мес", result["text"])
        self.assertIn("[1]", result["text"])
        self.assertIn("3 сотрудников", result["text"])
        self.assertIn("100 услуг", result["text"])

    def test_deterministic(self):
        first = self.run_skill("Сколько стоит тариф «Старт»?")
        second = self.run_skill("Сколько стоит тариф «Старт»?")
        self.assertEqual(first, second)

    def test_pro(self):
        result = self.run_skill(
            "Сколько стоит тариф «Профи» в месяц и какая скидка при оплате за год?"
        )
        self.assertEqual(result["catalog_ids"], ["pro"])
        self.assertIn("2 490", result["text"])
        self.assertIn("20%", result["text"])
        self.assertIn("до 10 сотрудников", result["text"])

    def test_business(self):
        result = self.run_skill("Что входит в тариф «Бизнес» и сколько он стоит?")
        self.assertEqual(result["catalog_ids"], ["business"])
        self.assertIn("7 900", result["text"])
        self.assertIn("до 50 сотрудников", result["text"])
        self.assertIn("1 000 запросов", result["text"])

    def test_multi_service(self):
        result = self.run_skill(
            "Сколько стоят аудит сайта, ведение соцсетей и разработка лендинга?"
        )
        self.assertEqual(
            result["catalog_ids"], ["site-audit", "social", "landing"]
        )
        for price in ("12 000", "19 500", "48 000"):
            self.assertIn(price, result["text"])
        self.assertIn("[1]", result["text"])

    def test_delivery_zones(self):
        result = self.run_skill(
            "Сколько стоит доставка документов по Москве и за МКАД?"
        )
        self.assertEqual(result["catalog_ids"], ["delivery-msk", "delivery-mkad"])
        self.assertIn("490", result["text"])
        self.assertIn("790", result["text"])

    def test_delivery_moscow_in(self):
        result = self.run_skill("Сколько стоит доставка в Москве?")
        self.assertEqual(result["catalog_ids"], ["delivery-msk"])
        self.assertIn("490", result["text"])

    def test_delivery_russia(self):
        result = self.run_skill(
            "Сколько стоит доставка за МКАД и по России, и какие сроки?"
        )
        self.assertEqual(result["catalog_ids"], ["delivery-mkad", "delivery-russia"])
        self.assertIn("790", result["text"])
        self.assertIn("от 1 200", result["text"])
        self.assertIn("2–3", result["text"])

    def test_generic_delivery_lists_all_zones(self):
        result = self.run_skill("Сколько стоит доставка?")
        self.assertEqual(
            result["catalog_ids"],
            ["delivery-msk", "delivery-mkad", "delivery-russia"],
        )

    def test_crm_duration(self):
        result = self.run_skill(
            "Сколько стоит настройка CRM под ключ и сколько времени она займёт?"
        )
        self.assertEqual(result["catalog_ids"], ["crm-setup"])
        self.assertIn("34 000", result["text"])
        self.assertIn("5 рабочих дней", result["text"])

    def test_limits_question(self):
        result = self.run_skill(
            "Сколько сотрудников можно подключить на тарифе «Профи»?"
        )
        self.assertEqual(result["catalog_ids"], ["pro"])
        self.assertIn("до 10 сотрудников", result["text"])

    def test_catalog_limit_question(self):
        result = self.run_skill(
            "Сколько услуг можно разместить в каталоге на тарифе «Старт»?"
        )
        self.assertEqual(result["catalog_ids"], ["start"])
        self.assertIn("100 услуг", result["text"])

    def test_api_question(self):
        result = self.run_skill("Есть ли API на тарифе «Профи»?")
        self.assertEqual(result["catalog_ids"], ["pro"])
        self.assertIn("API не предусмотрен", result["text"])

    def test_extra_seat(self):
        result = self.run_skill(
            "Сколько сотрудников входит в «Старт» и сколько стоит дополнительный?"
        )
        self.assertEqual(result["catalog_ids"], ["start", "extra-seat"])
        self.assertIn("290", result["text"])

    def test_list_tariffs(self):
        result = self.run_skill("Какие у вас тарифы?")
        self.assertEqual(
            result["catalog_ids"], ["start", "pro", "business"]
        )
        for price in ("990", "2 490", "7 900"):
            self.assertIn(price, result["text"])

    def test_price_from(self):
        result = self.run_skill("Сколько стоит контекстная реклама?")
        self.assertEqual(result["catalog_ids"], ["ads"])
        self.assertIn("от 25 000", result["text"])

    def test_matches_only_catalog_questions(self):
        self.assertTrue(self.skill.match("Сколько стоит тариф «Старт»?"))
        self.assertFalse(self.skill.match("Какие часы работы?"))
        self.assertFalse(self.skill.match("Как вернуть деньги?"))
        self.assertFalse(self.skill.match("Расскажи анекдот"))

    def test_unknown_service_is_silent(self):
        self.assertIsNone(self.run_skill("Сколько стоит телепортация сотрудников?"))
        self.assertIsNone(self.run_skill("Сколько стоит обучение сотрудников?"))
        self.assertIsNone(self.run_skill("Сколько стоит интеграция с 1С?"))

    def test_unknown_destination_is_silent(self):
        question = "Сколько стоит доставка на Марс?"
        self.assertFalse(self.skill.match(question))
        self.assertIsNone(self.run_skill(question))

    def test_third_party_is_silent(self):
        question = (
            "Покажи, сколько платит за тариф «Бизнес» компания «Ромашка» "
            "и её контактный телефон."
        )
        self.assertFalse(self.skill.match(question))
        self.assertIsNone(self.run_skill(question))

    def test_ambiguous_tariff_is_silent(self):
        self.assertIsNone(self.run_skill("Сколько стоит тариф?"))

    def test_no_model_imports(self):
        import inspect

        import eva.skills.pricing as module

        source = inspect.getsource(module).lower()
        self.assertNotIn("eva.clients", source)
        self.assertNotIn("openai", source)
        self.assertNotIn("requests", source)


class TestHours(SkillTest):
    def setUp(self):
        self.skill = HoursSkill()

    def run_skill(self, text):
        return self.skill.run(text, self.ctx())

    def test_match(self):
        for question in (
            "Какие часы работы?",
            "Во сколько закрывается офис в воскресенье?",
            "Когда работает поддержка?",
            "Работает ли офис в субботу?",
            "Когда офис закрывается?",
            "Во сколько поддержка отвечает?",
            "До скольки вы работаете?",
        ):
            with self.subTest(question=question):
                self.assertTrue(self.skill.match(question))

    def test_not_price_or_refund(self):
        self.assertFalse(self.skill.match("Сколько стоит тариф «Старт»?"))
        self.assertFalse(self.skill.match("Как вернуть деньги?"))

    def test_general_hours(self):
        result = self.run_skill("Какие часы работы?")
        self.assertEqual(result["skill"], "hours")
        self.assertEqual(result["sources"], ["[1]"])
        self.assertEqual(result["catalog_ids"], [])
        self.assertIn("9:00", result["text"])
        self.assertIn("19:00", result["text"])
        self.assertIn("21:00", result["text"])
        self.assertIn("15 минут", result["text"])
        self.assertIn("[1]", result["text"])

    def test_sunday_closed(self):
        result = self.run_skill("Во сколько закрывается офис в воскресенье?")
        self.assertIn("воскресенье", result["text"].lower())
        self.assertIn("закрыт", result["text"].lower())
        self.assertIn("[1]", result["text"])

    def test_saturday(self):
        result = self.run_skill("Работает ли офис в субботу?")
        self.assertIn("10:00", result["text"])
        self.assertIn("16:00", result["text"])

    def test_support_only(self):
        result = self.run_skill("Когда работает поддержка и за сколько она отвечает?")
        self.assertIn("21:00", result["text"])
        self.assertIn("15 минут", result["text"])
        self.assertNotIn("Офис в Москве", result["text"])


class TestRefunds(SkillTest):
    def setUp(self):
        self.skill = RefundsSkill()

    def run_skill(self, text):
        return self.skill.run(text, self.ctx())

    def test_match(self):
        for question in (
            "Как вернуть деньги?",
            "Какие условия возврата?",
            "За сколько вернутся деньги, если я платил картой?",
            "Можно ли отменить подписку?",
        ):
            with self.subTest(question=question):
                self.assertTrue(self.skill.match(question))

    def test_not_hours(self):
        self.assertFalse(self.skill.match("Какие часы работы?"))

    def test_conditions(self):
        result = self.run_skill("Какие условия возврата?")
        self.assertEqual(result["skill"], "refunds")
        self.assertEqual(result["sources"], ["[1]"])
        self.assertIn("14 дней", result["text"])
        self.assertIn("не пользовались", result["text"])
        self.assertIn("support@orioncode.ru", result["text"])
        self.assertIn("5 рабочих дней", result["text"])
        self.assertIn("10 рабочих дней", result["text"])
        self.assertIn("[1]", result["text"])

    def test_cancel_subscription(self):
        result = self.run_skill("Как отменить подписку?")
        self.assertIn("«Тарифы»", result["text"])

    def test_used_service(self):
        result = self.run_skill("Можно ли вернуть деньги, если услугой уже воспользовались?")
        self.assertIn("не предусмотрен", result["text"])

    def test_third_party_is_silent(self):
        self.assertFalse(
            self.skill.match("Верните деньги компании «Ромашка»")
        )


class TestContacts(SkillTest):
    def setUp(self):
        self.skill = ContactsSkill()

    def run_skill(self, text):
        return self.skill.run(text, self.ctx())

    def test_match(self):
        for question in (
            "Ваш телефон?",
            "Какая у вас почта?",
            "Покажи контакты",
            "Где находится офис?",
            "Как связаться с вами?",
        ):
            with self.subTest(question=question):
                self.assertTrue(self.skill.match(question))

    def test_not_price(self):
        self.assertFalse(self.skill.match("Сколько стоит тариф «Старт»?"))

    def test_all_contacts(self):
        result = self.run_skill("Покажи контакты")
        self.assertEqual(result["skill"], "contacts")
        self.assertIn("support@orioncode.ru", result["text"])
        self.assertIn("+7 (495) 120-32-32", result["text"])
        self.assertIn("Тверская", result["text"])
        self.assertIn("[1]", result["text"])

    def test_phone_only(self):
        result = self.run_skill("Какой у вас телефон?")
        self.assertIn("Телефон", result["text"])
        self.assertNotIn("Почта", result["text"])

    def test_email_only(self):
        result = self.run_skill("Какая у вас почта?")
        self.assertIn("Почта", result["text"])
        self.assertNotIn("Телефон", result["text"])

    def test_address_only(self):
        result = self.run_skill("Где находится офис?")
        self.assertIn("Адрес офиса", result["text"])
        self.assertNotIn("Телефон", result["text"])

    def test_third_party_is_silent(self):
        question = "Дай контактный телефон компании «Ромашка»"
        self.assertFalse(self.skill.match(question))


class TestRegistry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load(CATALOG_PATH)

    def ctx(self):
        return {"catalog": self.catalog}

    def test_price_start(self):
        result = try_skills("сколько стоит тариф Старт", self.ctx())
        self.assertIsNotNone(result)
        self.assertEqual(result["skill"], "pricing")
        self.assertIn("990", result["text"])
        self.assertIn("[1]", result["text"])

    def test_hours_question(self):
        result = try_skills("какие часы работы", self.ctx())
        self.assertIsNotNone(result)
        self.assertEqual(result["skill"], "hours")

    def test_refund_question(self):
        result = try_skills("вернуть деньги", self.ctx())
        self.assertIsNotNone(result)
        self.assertEqual(result["skill"], "refunds")

    def test_contacts_question(self):
        result = try_skills("ваш телефон", self.ctx())
        self.assertIsNotNone(result)
        self.assertEqual(result["skill"], "contacts")

    def test_unknown_is_none(self):
        for question in (
            "Расскажи анекдот",
            "Есть ли доставка на Марс?",
            "Сколько стоит телепортация сотрудников?",
            "Какая погода завтра?",
            "Покажи, сколько платит за тариф «Бизнес» компания «Ромашка».",
        ):
            with self.subTest(question=question):
                self.assertIsNone(try_skills(question, self.ctx()))

    def test_empty(self):
        self.assertIsNone(try_skills("", {}))
        self.assertIsNone(try_skills("   ", {}))
        self.assertIsNone(try_skills(None, {}))

    def test_skill_name_present(self):
        result = try_skills("Сколько стоит тариф «Старт»?", self.ctx())
        self.assertEqual(result["skill"], "pricing")

    def test_ctx_catalog_override(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "id": "custom",
                            "kind": "service",
                            "name": "Тестовая услуга",
                            "synonyms": ["тест"],
                            "price_rub": 777,
                            "period": "once",
                            "limits": ["один раз"],
                            "notes": [],
                            "sources": ["[9]"],
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            catalog = load(path)
            result = try_skills(
                "Сколько стоит тестовая услуга?", {"catalog": catalog}
            )
            self.assertEqual(result["catalog_ids"], ["custom"])
            self.assertIn("777", result["text"])
            self.assertEqual(result["sources"], ["[9]"])

    def test_no_skill_raises(self):
        class Boom:
            def __getattr__(self, name):
                raise AssertionError("skills must not touch the model")

        result = try_skills(
            "Сколько стоит тариф «Старт»?",
            {"catalog": self.catalog, "small": Boom(), "big": Boom()},
        )
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
