"""Тесты RAG по каталогу: поиск, формат контекста, вставка в router.

Сети нет: каталог читается из ``data/catalog.json`` (он же в проде), а в
части про router модели подменяются заглушками через ``deps``.
"""

from __future__ import annotations

import unittest

from eva import router
from eva.catalog import DEFAULT_PATH, Catalog, load
from eva.config import Config
from eva.rag import FIELD_WEIGHTS, RagContext, retrieve
from eva.types import Message, Request, Route

CATALOG = load()


# --------------------------------------------------------------------------- #
# заглушки
# --------------------------------------------------------------------------- #
class CapturingClient:
    """Клиент модели без сети: запоминает, какие сообщения ему передали."""

    def __init__(self, text: str = "ответ модели") -> None:
        self.text = text
        self.calls: list[list[Message]] = []

    def chat(self, messages: list[Message], max_tokens: int = 512) -> dict:
        self.calls.append(list(messages))
        return {
            "text": self.text,
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost_rub": 0.0},
            "raw": {"stub": True},
        }


def make_req(text: str = "привет", system: str = "ты ассистент") -> Request:
    return Request(
        messages=[Message("system", system), Message("user", text)],
        meta={"user_id": "rag-test", "channel": "test"},
    )


def small_route(text: str = "", cfg=None) -> Route:
    return Route("small", "classifier: короткий вопрос — хватит малой модели", 0.8)


def big_route(text: str = "", cfg=None) -> Route:
    return Route("big", "classifier: маркер рассуждения — нужна большая", 0.9)


def _item(item_id: str, name: str, **kwargs) -> dict:
    data = {
        "id": item_id,
        "kind": "service",
        "name": name,
        "synonyms": [],
        "price_rub": 100,
        "period": "once",
        "limits": [],
        "notes": [],
        "sources": ["[1]"],
    }
    data.update(kwargs)
    return data


# --------------------------------------------------------------------------- #
# поиск и формат контекста
# --------------------------------------------------------------------------- #
class RagRetrieveTest(unittest.TestCase):
    def test_pro_question_returns_limits_and_source(self) -> None:
        context = retrieve("что входит в тариф Профи", CATALOG)

        self.assertTrue(context, "контекст по «Профи» не должен быть пустым")
        self.assertIn("Профи", context.text)
        self.assertIn("до 10 сотрудников", context.text)
        self.assertIn("500 услуг в каталоге", context.text)
        self.assertIn("2 490", context.text)
        # источник виден и модели, и вызывающему коду
        self.assertEqual(context.sources, ["[1]"])
        self.assertIn("[1]", context.text)
        self.assertIn("Источники: [1]", context.text)
        self.assertEqual(context.ids, ["pro"], "лишние тарифы не должны попасть в контекст")

    def test_context_starts_with_prompt_block_header(self) -> None:
        context = retrieve("сколько стоит Профи", CATALOG)
        self.assertTrue(context.text.startswith("Контекст"))
        # нумерация позиций — с единицы, ровно по числу ids
        self.assertEqual(context.text.count("\n[1] "), 1)
        self.assertEqual(len(context.ids), 1)

    def test_out_of_catalog_questions_are_empty(self) -> None:
        for question in (
            "какая погода в Москве завтра",
            "кто написал «Войну и мир»",
            "напиши код на Python",
            "сколько стоит билет на Марс",
            "как приготовить борщ",
        ):
            with self.subTest(question=question):
                context = retrieve(question, CATALOG)
                self.assertEqual(context.text, "")
                self.assertEqual(context.sources, [])
                self.assertEqual(context.ids, [])
                self.assertFalse(context)

    def test_synonyms_from_card_are_searchable(self) -> None:
        social = retrieve("сколько стоит СММ", CATALOG)
        self.assertEqual(social.ids, ["social"])
        self.assertIn("Ведение соцсетей", social.text)

        audit = retrieve("аудит", CATALOG)
        self.assertEqual(audit.ids, ["site-audit"])

    def test_case_and_yo_are_normalized(self) -> None:
        catalog = Catalog([_item("tree", "Ёлка новогодняя", synonyms=["ёлочный набор"])])

        upper = retrieve("ЕЛКА НОВОГОДНЯЯ", catalog)
        self.assertEqual(upper.ids, ["tree"])

        # «ёлочный» и «елочный» — одно слово
        plain = retrieve("елочный набор", catalog)
        self.assertEqual(plain.ids, ["tree"])

    def test_name_weighs_more_than_notes(self) -> None:
        catalog = Catalog(
            [
                _item("by-name", "Альфа"),
                _item("by-note", "Бета", notes=["Альфа"]),
            ]
        )
        context = retrieve("альфа", catalog)

        self.assertEqual(context.ids[0], "by-name")
        self.assertNotIn("by-note", context.ids, "слабое совпадение по заметкам отсекается")

    def test_limits_weigh_more_than_notes(self) -> None:
        filler = [_item(f"filler-{index}", f"Позиция {index}") for index in range(8)]
        catalog = Catalog(
            [
                _item("by-limits", "Гамма", limits=["редкое"]),
                _item("by-notes", "Дельта", notes=["редкое"]),
                *filler,
            ]
        )
        context = retrieve("редкое", catalog)

        self.assertEqual(context.ids[:2], ["by-limits", "by-notes"])
        self.assertGreater(FIELD_WEIGHTS[0][1], FIELD_WEIGHTS[1][1])
        self.assertGreater(FIELD_WEIGHTS[1][1], FIELD_WEIGHTS[2][1])
        self.assertGreaterEqual(FIELD_WEIGHTS[2][1], FIELD_WEIGHTS[3][1])

    def test_k_limits_positions(self) -> None:
        all_tariffs = retrieve("тарифы", CATALOG, k=3)
        self.assertEqual(all_tariffs.ids, ["start", "pro", "business"])

        two = retrieve("тарифы", CATALOG, k=2)
        self.assertEqual(len(two.ids), 2)

        self.assertFalse(retrieve("тарифы", CATALOG, k=0))
        self.assertFalse(retrieve("тарифы", CATALOG, k=-1))

    def test_empty_query_is_empty(self) -> None:
        for text in ("", "   ", "что как сколько", "!!!", None):
            with self.subTest(text=text):
                self.assertFalse(retrieve(text, CATALOG))

    def test_catalog_can_be_path_or_list(self) -> None:
        from_path = retrieve("Профи", str(DEFAULT_PATH))
        self.assertEqual(from_path.ids, ["pro"])

        from_list = retrieve("Профи", CATALOG.all())
        self.assertEqual(from_list.ids, ["pro"])

    def test_broken_catalog_does_not_raise(self) -> None:
        self.assertFalse(retrieve("Профи", 42))
        self.assertFalse(retrieve("Профи", {"kind": "tariff"}))

    def test_to_dict_shape(self) -> None:
        context = retrieve("Профи", CATALOG)
        data = RagContext(**context.to_dict()).to_dict()
        self.assertEqual(data, context.to_dict())
        self.assertEqual(set(data), {"text", "sources", "ids"})


# --------------------------------------------------------------------------- #
# вставка контекста в сообщения router
# --------------------------------------------------------------------------- #
class RouterContextTest(unittest.TestCase):
    def _answer(self, text: str, deps: dict, route_fn=small_route):
        deps = {"skills": lambda text, ctx: None, "classifier": route_fn, **deps}
        return router.answer(make_req(text), Config(), deps)

    def test_context_is_inserted_for_small_and_user_text_untouched(self) -> None:
        rag = lambda text, catalog: RagContext(  # noqa: E731
            text="Контекст (каталог):\n[1] Тариф «Профи» — до 10 сотрудников",
            sources=["[1]"],
            ids=["pro"],
        )
        small = CapturingClient()
        req = make_req("что входит в тариф Профи")

        answer = router.answer(
            req,
            Config(),
            deps={"skills": lambda text, ctx: None, "classifier": small_route, "small": small, "rag": rag},
        )

        self.assertEqual(answer.route.kind, "small")
        self.assertEqual(answer.text, "ответ модели")
        self.assertEqual(len(small.calls), 1)
        messages = small.calls[0]

        # исходный запрос пользователя не изменился и остался последним
        self.assertEqual(messages[-1].role, "user")
        self.assertEqual(messages[-1].content, "что входит в тариф Профи")
        self.assertEqual([m.content for m in req.messages], ["ты ассистент", "что входит в тариф Профи"])
        # контекст — отдельное system-сообщение после исходного system
        self.assertEqual(messages[0].content, "ты ассистент")
        self.assertEqual(messages[1].role, "system")
        self.assertIn("Контекст", messages[1].content)
        self.assertIn("до 10 сотрудников", messages[1].content)
        self.assertEqual(len(messages), 3)

    def test_context_is_inserted_for_big(self) -> None:
        big = CapturingClient()
        deps = {
            "skills": lambda text, ctx: None,
            "classifier": big_route,
            "big": big,
            "rag": lambda text, catalog: "Контекст: [1] факт",
        }
        answer = router.answer(make_req("объясни тариф Профи"), Config(), deps)

        self.assertEqual(answer.route.kind, "big")
        self.assertEqual(len(big.calls), 1)
        self.assertEqual(big.calls[0][-1].content, "объясни тариф Профи")
        self.assertTrue(any("Контекст" in m.content for m in big.calls[0]))

    def test_empty_context_keeps_messages_as_before(self) -> None:
        small = CapturingClient()
        req = make_req("привет")
        answer = router.answer(
            req,
            Config(),
            deps={
                "skills": lambda text, ctx: None,
                "classifier": small_route,
                "small": small,
                "rag": lambda text, catalog: RagContext(),
            },
        )

        self.assertEqual(answer.route.kind, "small")
        messages = small.calls[0]
        self.assertEqual(len(messages), 2)
        self.assertEqual([m.role for m in messages], ["system", "user"])
        self.assertEqual([m.content for m in messages], ["ты ассистент", "привет"])
        self.assertFalse(any("Контекст" in m.content for m in messages))

    def test_broken_rag_does_not_break_answer(self) -> None:
        def broken(text, catalog):
            raise RuntimeError("rag сломан")

        small = CapturingClient()
        answer = router.answer(
            make_req("сколько стоит Профи"),
            Config(),
            deps={"skills": lambda text, ctx: None, "classifier": small_route, "small": small, "rag": broken},
        )

        self.assertEqual(answer.text, "ответ модели")
        self.assertEqual(len(small.calls[0]), 2)

    def test_skill_route_does_not_call_rag(self) -> None:
        def rag(text, catalog):
            raise AssertionError("rag не должен вызываться на маршруте skill")

        answer = router.answer(
            make_req("сколько стоит тариф Старт"),
            Config(),
            deps={
                "skills": lambda text, ctx: {"skill": "pricing", "text": "990 ₽/мес [1]"},
                "rag": rag,
                "small": CapturingClient(),
            },
        )

        self.assertEqual(answer.route.kind, "skill")
        self.assertEqual(answer.text, "990 ₽/мес [1]")

    def test_real_rag_is_used_without_injection(self) -> None:
        small = CapturingClient()
        answer = router.answer(
            make_req("что входит в тариф Профи"),
            Config(),
            deps={"skills": lambda text, ctx: None, "classifier": small_route, "small": small},
        )

        self.assertEqual(answer.route.kind, "small")
        messages = small.calls[0]
        self.assertEqual(messages[-1].content, "что входит в тариф Профи")
        context = next(m for m in messages if "Контекст" in m.content)
        self.assertIn("Профи", context.content)
        self.assertIn("до 10 сотрудников", context.content)
        self.assertIn("[1]", context.content)


if __name__ == "__main__":
    unittest.main()
