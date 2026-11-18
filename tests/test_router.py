"""Тесты маршрутизатора: маршруты, причины, фолбэки, бюджет.

Сети нет: скиллы, классификатор и клиенты моделей подменяются заглушками
через ``deps`` или через ``mock.patch`` ленивых загрузчиков.
"""

from __future__ import annotations

import contextlib
import inspect
import sys
import unittest
from types import ModuleType
from unittest import mock

from eva import policy, router
from eva.config import Config, load
from eva.policy import Budget, BudgetExceeded, ModelUnavailable, error_kind
from eva.types import Answer, Message, Request, Route


# --------------------------------------------------------------------------- #
# заглушки
# --------------------------------------------------------------------------- #
class FakeClientError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class FakeClient:
    """Клиент модели без сети: пишет вызовы и отдаёт заданный результат."""

    def __init__(
        self,
        text: str = "ответ",
        usage: dict | None = None,
        error: Exception | None = None,
        include_cost: bool = True,
    ):
        self.text = text
        self.usage = usage
        self.error = error
        self.include_cost = include_cost
        self.calls: list[tuple[list[Message], int]] = []

    def chat(self, messages: list[Message], max_tokens: int = 512) -> dict:
        self.calls.append((list(messages), max_tokens))
        if self.error is not None:
            raise self.error
        usage: dict = {"prompt_tokens": 10, "completion_tokens": 5}
        if self.include_cost:
            usage["cost_rub"] = 0.0
        usage.update(self.usage or {})
        return {"text": self.text, "usage": usage, "raw": {"stub": True}}


def make_req(text: str = "привет") -> Request:
    return Request(
        messages=[Message("system", "ты ассистент"), Message("user", text)],
        meta={"user_id": "u-1", "channel": "test"},
    )


def small_route(text: str = "", cfg=None) -> Route:
    return Route("small", "classifier: короткий вопрос — хватит малой модели", 0.8)


def big_route(text: str = "", cfg=None) -> Route:
    return Route("big", "classifier: маркер рассуждения «объясни» — нужна большая", 0.9)


class Deps:
    """Объект-зависимости (проверяем и такой способ подмены)."""

    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            setattr(self, name, value)


# --------------------------------------------------------------------------- #
# типы и конфиг
# --------------------------------------------------------------------------- #
class TypesAndConfigTest(unittest.TestCase):
    def test_request_text_is_last_user_message(self) -> None:
        req = Request(messages=[{"role": "user", "content": "старое"}, {"role": "assistant", "content": "ок"}, {"role": "user", "content": " новое "}])
        self.assertEqual(req.text, "новое")
        self.assertIsInstance(req.messages[0], Message)

    def test_route_confidence_is_clamped_and_reason_kept(self) -> None:
        route = Route(" BIG ", "нужна большая", 7)
        self.assertEqual(route.kind, "big")
        self.assertEqual(route.confidence, 1.0)
        self.assertEqual(route.reason, "нужна большая")

    def test_answer_to_dict_has_route_and_price(self) -> None:
        answer = Answer("текст", Route("small", "причина"), usage={"cost_rub": 0.0})
        data = answer.to_dict()
        self.assertEqual(data["text"], "текст")
        self.assertEqual(data["route"]["kind"], "small")
        self.assertIn("cost_rub", data)

    def test_config_from_env(self) -> None:
        cfg = Config.from_env(
            {
                "EVA_SMALL_BASE": "http://localhost:9000/v1",
                "EVA_SMALL_KEY": "sk-small",
                "EVA_SMALL_MODEL": "eva-small-1",
                "EVA_SMALL_TIMEOUT": "5",
                "EVA_SMALL_RETRIES": "0",
                "EVA_SMALL_PRICE_PROMPT": "0.1",
                "EVA_SMALL_PRICE_COMPLETION": "0.2",
                "EVA_BIG_BASE": "https://api.example/v1",
                "EVA_BIG_KEY": "sk-big",
                "EVA_BIG_MODEL": "big-1",
                "EVA_MAX_COST_RUB": "1.5",
                "EVA_MAX_TOKENS": "256",
                "EVA_CLASSIFIER_LONG_CHARS": "50",
                "EVA_CLASSIFIER_LATIN_RATIO": "0.5",
                "EVA_CLASSIFIER_MARKERS": "объясни, сравни",
            }
        )
        self.assertEqual(cfg.small_base, "http://localhost:9000/v1")
        self.assertEqual(cfg.small_timeout, 5.0)
        self.assertEqual(cfg.small_retries, 0)
        self.assertEqual(cfg.big_model, "big-1")
        self.assertTrue(cfg.big_available)
        self.assertEqual(cfg.max_cost_rub, 1.5)
        self.assertEqual(cfg.max_tokens, 256)
        self.assertEqual(cfg.classifier_long_chars, 50)
        self.assertEqual(cfg.classifier_reason_markers, ("объясни", "сравни"))
        self.assertEqual(cfg.prices("small"), (0.1, 0.2))
        self.assertAlmostEqual(cfg.estimate_cost("small", 1000, 1000), 0.3)

    def test_config_env_defaults_and_unlimited_budget(self) -> None:
        cfg = load({})
        self.assertEqual(cfg.max_cost_rub, 5.0)
        self.assertFalse(cfg.big_available)
        self.assertIsNone(Config.from_env({"EVA_MAX_COST_RUB": "none"}).max_cost_rub)
        self.assertIsNone(Config.from_env({"EVA_MAX_COST_RUB": "-1"}).max_cost_rub)
        self.assertEqual(Config.from_env({"EVA_MAX_COST_RUB": "0"}).max_cost_rub, 0.0)

    def test_config_env_rejects_garbage(self) -> None:
        with self.assertRaises(ValueError):
            Config.from_env({"EVA_MAX_COST_RUB": "дорого"})
        with self.assertRaises(ValueError):
            Config.from_env({"EVA_MAX_TOKENS": "много"})


# --------------------------------------------------------------------------- #
# decide: скиллы → классификатор → эвристика
# --------------------------------------------------------------------------- #
class DecideTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = Config()

    def _providers(self, skills=None, classifier=None):
        """Подменить ленивые загрузчики на время теста."""
        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.object(router, "_load_skills", return_value=skills))
        stack.enter_context(mock.patch.object(router, "_load_classifier", return_value=classifier))
        return stack

    def test_contract_signatures(self) -> None:
        self.assertEqual(list(inspect.signature(router.decide).parameters), ["req", "cfg"])
        answer_params = list(inspect.signature(router.answer).parameters)
        self.assertEqual(answer_params[:3], ["req", "cfg", "deps"])

    def test_decide_finds_skill_first(self) -> None:
        skills = lambda text, ctx: {"skill": "pricing", "text": "Старт — 490 ₽ [1]", "sources": ["[1]"], "catalog_ids": ["start"]}
        classifier = lambda text, cfg: big_route()
        with self._providers(skills=skills, classifier=classifier):
            route = router.decide(make_req("сколько стоит тариф Старт"), self.cfg)
        self.assertEqual(route.kind, "skill")
        self.assertIn("pricing", route.reason)
        self.assertIn("модель не вызывалась", route.reason)
        self.assertEqual(route.confidence, 1.0)

    def test_decide_uses_classifier_route_and_reason(self) -> None:
        with self._providers(skills=None, classifier=big_route):
            route = router.decide(make_req("объясни, почему небо голубое"), self.cfg)
        self.assertEqual(route.kind, "big")
        self.assertEqual(route.reason, "classifier: маркер рассуждения «объясни» — нужна большая")
        self.assertEqual(route.confidence, 0.9)

    def test_decide_classifier_without_reason_gets_one(self) -> None:
        classifier = lambda text, cfg: {"kind": "big", "confidence": 0.7}
        with self._providers(skills=None, classifier=classifier):
            route = router.decide(make_req("напиши код на python"), self.cfg)
        self.assertEqual(route.kind, "big")
        self.assertTrue(route.reason.strip())

    def test_decide_classifier_mapping_and_garbage(self) -> None:
        with self._providers(skills=None, classifier=lambda text, cfg: {"kind": "nonsense"}):
            route = router.decide(make_req("привет"), self.cfg)
        self.assertEqual(route.kind, "small")
        self.assertIn("эвристика", route.reason)

    def test_decide_survives_broken_classifier(self) -> None:
        def broken(text, cfg):
            raise RuntimeError("классификатор сломан")

        with self._providers(skills=None, classifier=broken):
            route = router.decide(make_req("привет"), self.cfg)
        self.assertEqual(route.kind, "small")
        self.assertTrue(route.reason)

    def test_decide_never_returns_skill_without_skill_result(self) -> None:
        classifier = lambda text, cfg: Route("skill", "classifier: факт из каталога")
        with self._providers(skills=None, classifier=classifier):
            route = router.decide(make_req("сколько стоит тариф Старт"), self.cfg)
        self.assertEqual(route.kind, "big")
        self.assertIn("скилл промолчал", route.reason)
        self.assertLessEqual(route.confidence, 0.6)

    def test_decide_silent_skill_falls_through(self) -> None:
        with self._providers(skills=lambda text, ctx: None, classifier=small_route):
            route = router.decide(make_req("привет"), self.cfg)
        self.assertEqual(route.kind, "small")
        self.assertIn("короткий вопрос", route.reason)

    def test_broken_skill_does_not_break_routing(self) -> None:
        def broken(text, ctx):
            raise ValueError("скилл упал")

        with self._providers(skills=broken, classifier=small_route):
            route = router.decide(make_req("привет"), self.cfg)
        self.assertEqual(route.kind, "small")

    def test_heuristic_is_small_for_short_text(self) -> None:
        with self._providers():
            route = router.decide(make_req("привет"), self.cfg)
        self.assertEqual(route.kind, "small")
        self.assertIn("эвристика", route.reason)

    def test_heuristic_is_big_for_reasoning_marker(self) -> None:
        with self._providers():
            route = router.decide(make_req("Объясни, почему небо голубое"), self.cfg)
        self.assertEqual(route.kind, "big")
        self.assertIn("маркер", route.reason)

    def test_heuristic_is_big_for_long_text(self) -> None:
        with self._providers():
            route = router.decide(make_req("а" * 300), self.cfg)
        self.assertEqual(route.kind, "big")
        self.assertIn("длиной", route.reason)

    def test_heuristic_is_big_for_latin_text(self) -> None:
        with self._providers():
            route = router.decide(make_req("hello world this is a test"), self.cfg)
        self.assertEqual(route.kind, "big")
        self.assertIn("латиницы", route.reason)

    def test_empty_request_does_not_crash(self) -> None:
        with self._providers():
            route = router.decide(Request(), self.cfg)
        self.assertEqual(route.kind, "small")
        self.assertIn("пустой", route.reason)

    def test_lazy_import_of_skills_module(self) -> None:
        registry = ModuleType("eva.skills.registry")
        registry.try_skills = lambda text, ctx: {"skill": "hours", "text": "Пн–Пт 10:00–19:00 [1]"}
        package = ModuleType("eva.skills")
        package.registry = registry
        with mock.patch.object(router, "_SKILLS_FN", None):
            with mock.patch.dict(sys.modules, {"eva.skills": package, "eva.skills.registry": registry}):
                route = router.decide(make_req("какие часы работы"), self.cfg)
        self.assertEqual(route.kind, "skill")
        self.assertIn("hours", route.reason)

    def test_lazy_import_of_classifier_module(self) -> None:
        module = ModuleType("eva.classifier")
        module.classify = lambda text, cfg: Route("big", "classifier: нужен код", 0.88)
        with mock.patch.object(router, "_CLASSIFIER_FN", None):
            with mock.patch.object(router, "_load_skills", return_value=None):
                with mock.patch.dict(sys.modules, {"eva.classifier": module}):
                    route = router.decide(make_req("напиши код на python"), self.cfg)
        self.assertEqual(route.kind, "big")
        self.assertIn("нужен код", route.reason)


# --------------------------------------------------------------------------- #
# answer: скиллы
# --------------------------------------------------------------------------- #
class AnswerSkillTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = Config()

    def test_skill_answer_costs_nothing_and_skips_models(self) -> None:
        captured = {}

        def skills(text, ctx):
            captured.update(ctx)
            return {
                "skill": "pricing",
                "text": "Тариф Старт — 490 ₽/мес [1]",
                "sources": ["[1]"],
                "catalog_ids": ["start"],
            }

        broken_client = mock.Mock()
        answer = router.answer(
            make_req("сколько стоит тариф Старт"),
            self.cfg,
            deps={"skills": skills, "small": broken_client, "big": broken_client},
        )
        self.assertEqual(answer.route.kind, "skill")
        self.assertEqual(answer.skill, "pricing")
        self.assertEqual(answer.text, "Тариф Старт — 490 ₽/мес [1]")
        self.assertEqual(answer.cost_rub, 0.0)
        self.assertEqual(answer.usage["prompt_tokens"], 0)
        self.assertIsInstance(answer.latency_ms, int)
        self.assertGreaterEqual(answer.latency_ms, 0)
        self.assertIn("pricing", answer.route.reason)
        broken_client.chat.assert_not_called()
        # контекст скилла: текст, метаданные, история
        self.assertEqual(captured["text"], "сколько стоит тариф Старт")
        self.assertEqual(captured["user_id"], "u-1")
        self.assertEqual(captured["channel"], "test")
        self.assertEqual(captured["messages"][1]["role"], "user")

    def test_skill_answer_without_name(self) -> None:
        deps = {"skills": lambda text, ctx: {"text": "ответ скилла"}}
        with mock.patch.object(router, "_load_classifier", return_value=None):
            answer = router.answer(make_req("вопрос"), self.cfg, deps=deps)
        self.assertEqual(answer.route.kind, "skill")
        self.assertIsNone(answer.skill)
        self.assertIn("skill:", answer.route.reason)


# --------------------------------------------------------------------------- #
# answer: модели, причины, фолбэки
# --------------------------------------------------------------------------- #
class AnswerModelsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = Config()
        self.small = FakeClient("ответ малой", usage={"cost_rub": 0.0})
        self.big = FakeClient("ответ большой", usage={"cost_rub": 0.2})

    def _deps(self, **kwargs):
        deps = {"classifier": small_route}
        deps.update(kwargs)
        return deps

    def test_small_answer_has_route_usage_and_price(self) -> None:
        with mock.patch.object(router, "_load_skills", return_value=None):
            answer = router.answer(make_req(), self.cfg, deps=self._deps(small=self.small))
        self.assertEqual(answer.route.kind, "small")
        self.assertIn("малой модели", answer.route.reason)
        self.assertEqual(answer.text, "ответ малой")
        self.assertEqual(answer.usage["prompt_tokens"], 10)
        self.assertEqual(answer.usage["completion_tokens"], 5)
        self.assertEqual(answer.cost_rub, 0.0)
        self.assertEqual(len(self.small.calls), 1)
        self.assertEqual(self.small.calls[0][1], self.cfg.max_tokens)

    def test_big_answer_passes_route_and_confidence(self) -> None:
        cfg = Config(big_base="https://api.example/v1", big_model="big-1")
        with mock.patch.object(router, "_load_skills", return_value=None):
            answer = router.answer(make_req("объясни"), cfg, deps=self._deps(classifier=big_route, big=self.big))
        self.assertEqual(answer.route.kind, "big")
        self.assertEqual(answer.route.confidence, 0.9)
        self.assertEqual(answer.text, "ответ большой")
        self.assertEqual(answer.cost_rub, 0.2)

    def test_cost_computed_from_config_when_provider_silent(self) -> None:
        cfg = Config(
            small_price_prompt_rub_per_1k=0.1,
            small_price_completion_rub_per_1k=0.2,
        )
        client = FakeClient(
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            include_cost=False,
        )
        with mock.patch.object(router, "_load_skills", return_value=None):
            answer = router.answer(make_req(), cfg, deps=self._deps(small=client))
        self.assertAlmostEqual(answer.cost_rub, 0.002, places=6)
        self.assertAlmostEqual(answer.usage["cost_rub"], 0.002, places=6)

    def test_small_failure_falls_back_to_big_with_reason(self) -> None:
        cfg = Config(big_base="https://api.example/v1", big_model="big-1")
        failing = FakeClient(error=FakeClientError("timed out after 30s"))
        with mock.patch.object(router, "_load_skills", return_value=None):
            answer = router.answer(
                make_req(),
                cfg,
                deps=self._deps(small=failing, big=self.big),
            )
        self.assertEqual(answer.route.kind, "big")
        self.assertIn("fallback small→big", answer.route.reason)
        self.assertIn("таймаут", answer.route.reason)
        self.assertIn("classifier:", answer.route.reason, "исходная причина должна сохраняться")
        self.assertEqual(answer.text, "ответ большой")
        self.assertEqual(len(failing.calls), 1)
        self.assertEqual(len(self.big.calls), 1)

    def test_small_failure_without_big_reraises(self) -> None:
        cfg = Config(big_base="", big_model="")
        failing = FakeClient(error=FakeClientError("connection refused"))
        with mock.patch.object(router, "_load_skills", return_value=None), mock.patch.object(
            router, "_make_client", return_value=None
        ):
            with self.assertRaises(FakeClientError):
                router.answer(make_req(), cfg, deps=self._deps(small=failing))
        self.assertEqual(len(self.big.calls), 0)

    def test_small_failure_when_big_over_budget_reraises(self) -> None:
        cfg = Config(big_base="https://api.example/v1", big_model="big-1", max_cost_rub=0.01)
        failing = FakeClient(error=FakeClientError("boom"))
        with mock.patch.object(router, "_load_skills", return_value=None):
            with self.assertRaises(FakeClientError):
                router.answer(make_req(), cfg, deps=self._deps(small=failing, big=self.big))
        self.assertEqual(len(self.big.calls), 0)

    def test_big_429_falls_back_to_small_with_reason(self) -> None:
        cfg = Config(big_base="https://api.example/v1", big_model="big-1")
        limited = FakeClient(error=FakeClientError("429 Too Many Requests", status=429))
        with mock.patch.object(router, "_load_skills", return_value=None):
            answer = router.answer(
                make_req("объясни"),
                cfg,
                deps=self._deps(classifier=big_route, big=limited, small=self.small),
            )
        self.assertEqual(answer.route.kind, "small")
        self.assertIn("fallback big→small", answer.route.reason)
        self.assertIn("429", answer.route.reason)
        self.assertEqual(answer.text, "ответ малой")
        self.assertEqual(len(limited.calls), 1)
        self.assertEqual(len(self.small.calls), 1)

    def test_big_budget_error_falls_back_to_small(self) -> None:
        cfg = Config(big_base="https://api.example/v1", big_model="big-1")
        limited = FakeClient(error=FakeClientError("provider budget exhausted"))
        with mock.patch.object(router, "_load_skills", return_value=None):
            answer = router.answer(
                make_req("объясни"),
                cfg,
                deps=self._deps(classifier=big_route, big=limited, small=self.small),
            )
        self.assertEqual(answer.route.kind, "small")
        self.assertIn("бюджет", answer.route.reason)

    def test_big_5xx_does_not_fall_back_to_small(self) -> None:
        cfg = Config(big_base="https://api.example/v1", big_model="big-1")
        broken = FakeClient(error=FakeClientError("500 Internal Server Error", status=500))
        with mock.patch.object(router, "_load_skills", return_value=None):
            with self.assertRaises(FakeClientError):
                router.answer(
                    make_req("объясни"),
                    cfg,
                    deps=self._deps(classifier=big_route, big=broken, small=self.small),
                )
        self.assertEqual(len(self.small.calls), 0)

    def test_big_not_retried_after_small_fallback(self) -> None:
        """Малая упала → большая 429: обратно на малую не идём (без цикла)."""
        cfg = Config(big_base="https://api.example/v1", big_model="big-1")
        small = FakeClient(error=FakeClientError("timeout"))
        big = FakeClient(error=FakeClientError("429", status=429))
        with mock.patch.object(router, "_load_skills", return_value=None):
            with self.assertRaises(FakeClientError):
                router.answer(make_req(), cfg, deps=self._deps(small=small, big=big))
        self.assertEqual(len(small.calls), 1)
        self.assertEqual(len(big.calls), 1)

    def test_usage_accepts_object_shape(self) -> None:
        class Usage:
            prompt_tokens = 7
            completion_tokens = 3
            cost_rub = 0.01

        class Result:
            text = "текст из объекта"
            usage = Usage()

        client = mock.Mock()
        client.chat.return_value = Result()
        with mock.patch.object(router, "_load_skills", return_value=None):
            answer = router.answer(make_req(), self.cfg, deps=self._deps(small=client))
        self.assertEqual(answer.text, "текст из объекта")
        self.assertEqual(answer.usage["prompt_tokens"], 7)
        self.assertEqual(answer.cost_rub, 0.01)

    def test_classifier_skill_without_skill_result_goes_big(self) -> None:
        cfg = Config(big_base="https://api.example/v1", big_model="big-1")
        deps = {
            "classifier": lambda text, cfg: Route("skill", "classifier: факт из каталога"),
            "big": self.big,
        }
        with mock.patch.object(router, "_load_skills", return_value=None):
            answer = router.answer(make_req("сколько стоит тариф Старт"), cfg, deps=deps)
        self.assertEqual(answer.route.kind, "big")
        self.assertIn("скилл промолчал", answer.route.reason)
        self.assertEqual(answer.text, "ответ большой")

    def test_deps_as_object_and_nested_clients(self) -> None:
        deps = Deps(classifier=small_route, clients={"small": self.small})
        with mock.patch.object(router, "_load_skills", return_value=None):
            answer = router.answer(make_req(), self.cfg, deps=deps)
        self.assertEqual(answer.route.kind, "small")
        self.assertEqual(len(self.small.calls), 1)


# --------------------------------------------------------------------------- #
# бюджет
# --------------------------------------------------------------------------- #
class BudgetTest(unittest.TestCase):
    def test_budget_downgrades_big_to_small_before_call(self) -> None:
        cfg = Config(big_base="https://api.example/v1", big_model="big-1", max_cost_rub=0.1)
        big = FakeClient("дорого")
        small = FakeClient("дешёво")
        with mock.patch.object(router, "_load_skills", return_value=None):
            answer = router.answer(
                make_req("объясни"),
                cfg,
                deps={"classifier": big_route, "big": big, "small": small},
            )
        self.assertEqual(answer.route.kind, "small")
        self.assertIn("fallback big→small", answer.route.reason)
        self.assertIn("бюджета", answer.route.reason)
        self.assertEqual(len(big.calls), 0, "платный вызов не должен состояться")
        self.assertEqual(len(small.calls), 1)
        self.assertEqual(answer.text, "дешёво")

    def test_over_budget_aborts_instead_of_spending(self) -> None:
        cfg = Config(
            max_cost_rub=0.0001,
            small_price_prompt_rub_per_1k=1.0,
            small_price_completion_rub_per_1k=1.0,
        )
        small = FakeClient("не должно быть вызвано")
        with mock.patch.object(router, "_load_skills", return_value=None):
            with self.assertRaises(BudgetExceeded) as ctx:
                router.answer(make_req(), cfg, deps={"classifier": small_route, "small": small})
        self.assertEqual(len(small.calls), 0)
        self.assertIn("бюджет", str(ctx.exception).lower())
        self.assertEqual(ctx.exception.route_kind, "small")

    def test_big_over_budget_without_small_aborts(self) -> None:
        cfg = Config(big_base="https://api.example/v1", big_model="big-1", max_cost_rub=0.01)
        big = FakeClient("не должно быть вызвано")
        with mock.patch.object(router, "_load_skills", return_value=None), mock.patch.object(
            router, "_make_client", return_value=None
        ):
            with self.assertRaises(BudgetExceeded):
                router.answer(
                    make_req("объясни"),
                    cfg,
                    deps={"classifier": big_route, "big": big},
                )
        self.assertEqual(len(big.calls), 0)

    def test_unlimited_budget_allows_expensive_answer(self) -> None:
        cfg = Config(big_base="https://api.example/v1", big_model="big-1", max_cost_rub=None)
        big = FakeClient("дорого", usage={"cost_rub": 100.0})
        with mock.patch.object(router, "_load_skills", return_value=None):
            answer = router.answer(make_req("объясни"), cfg, deps={"classifier": big_route, "big": big})
        self.assertEqual(answer.route.kind, "big")
        self.assertEqual(answer.cost_rub, 100.0)
        self.assertNotIn("превысила бюджет", answer.route.reason)

    def test_actual_cost_over_budget_is_reported(self) -> None:
        cfg = Config(max_cost_rub=0.001)
        client = FakeClient(usage={"cost_rub": 0.5})
        with mock.patch.object(router, "_load_skills", return_value=None):
            answer = router.answer(make_req(), cfg, deps={"classifier": small_route, "small": client})
        self.assertEqual(answer.route.kind, "small")
        self.assertIn("превысила бюджет", answer.route.reason)

    def test_budget_unit(self) -> None:
        budget = Budget.from_config(Config(max_cost_rub=1.0))
        self.assertTrue(budget.limited)
        self.assertTrue(budget.can_afford(0.5))
        budget.charge(0.75)
        self.assertAlmostEqual(budget.remaining, 0.25)
        self.assertFalse(budget.can_afford(0.3))
        with self.assertRaises(BudgetExceeded):
            budget.check(0.3, route_kind="big")
        unlimited = Budget(None)
        self.assertFalse(unlimited.limited)
        self.assertTrue(unlimited.can_afford(10_000.0))
        self.assertIsNone(unlimited.remaining)


# --------------------------------------------------------------------------- #
# недоступные модели и политика
# --------------------------------------------------------------------------- #
class UnavailableAndPolicyTest(unittest.TestCase):
    def test_no_clients_raises_model_unavailable(self) -> None:
        cfg = Config()
        with mock.patch.object(router, "_load_skills", return_value=None), mock.patch.object(
            router, "_make_client", return_value=None
        ):
            with self.assertRaises(ModelUnavailable):
                router.answer(make_req(), cfg, deps={"classifier": small_route})

    def test_make_client_wires_config_into_client(self) -> None:
        created = {}

        class FakeOpenAIClient:
            def __init__(self, base, key, model, timeout=60, retries=2):
                created.update(base=base, key=key, model=model, timeout=timeout, retries=retries)

        module = ModuleType("eva.clients.openai_compat")
        module.OpenAIClient = FakeOpenAIClient
        cfg = Config(
            big_base="https://api.example/v1",
            big_key="sk-big",
            big_model="big-1",
            big_timeout=7.0,
            big_retries=3,
        )
        with mock.patch.dict(sys.modules, {"eva.clients.openai_compat": module}):
            client = router._make_client("big", cfg)
        self.assertIsInstance(client, FakeOpenAIClient)
        self.assertEqual(created, {"base": "https://api.example/v1", "key": "sk-big", "model": "big-1", "timeout": 7.0, "retries": 3})

    def test_unconfigured_big_has_no_client(self) -> None:
        self.assertIsNone(router._make_client("big", Config(big_base="", big_model="")))

    def test_timeouts_and_estimates(self) -> None:
        cfg = Config(small_timeout=5.0, big_timeout=90.0, max_tokens=100)
        self.assertEqual(policy.timeout_for("small", cfg), 5.0)
        self.assertEqual(policy.timeout_for("big", cfg), 90.0)
        cost = policy.estimate_request_cost(cfg, "big", "привет")
        expected = cfg.estimate_cost("big", policy.estimate_tokens("привет"), 100)
        self.assertAlmostEqual(cost, expected)

    def test_error_kind_detection(self) -> None:
        self.assertEqual(error_kind(FakeClientError("429 Too Many Requests")), policy.RATE_LIMIT)
        self.assertEqual(error_kind(FakeClientError("rate limit", status=429)), policy.RATE_LIMIT)
        self.assertEqual(error_kind(FakeClientError("timed out")), policy.TIMEOUT)
        self.assertEqual(error_kind(FakeClientError("budget exhausted")), policy.BUDGET)
        self.assertEqual(error_kind(FakeClientError("connection refused")), policy.CONNECTION)
        self.assertEqual(error_kind(FakeClientError("что-то пошло не так")), policy.ERROR)

    def test_fallback_matrix(self) -> None:
        self.assertEqual(policy.fallback_target("small", policy.TIMEOUT), "big")
        self.assertEqual(policy.fallback_target("small", policy.ERROR), "big")
        self.assertEqual(policy.fallback_target("big", policy.RATE_LIMIT), "small")
        self.assertEqual(policy.fallback_target("big", policy.BUDGET), "small")
        self.assertIsNone(policy.fallback_target("big", policy.TIMEOUT))
        self.assertIsNone(policy.fallback_target("skill", policy.ERROR))

    def test_every_route_has_reason(self) -> None:
        cfg = Config(big_base="https://api.example/v1", big_model="big-1")
        routes = []

        with mock.patch.object(router, "_load_skills", return_value=lambda text, ctx: {"skill": "hours", "text": "10–19"}):
            routes.append(router.decide(make_req("часы работы"), cfg))
        with mock.patch.object(router, "_load_skills", return_value=None), mock.patch.object(
            router, "_load_classifier", return_value=small_route
        ):
            routes.append(router.decide(make_req("привет"), cfg))
        with mock.patch.object(router, "_load_skills", return_value=None), mock.patch.object(
            router, "_load_classifier", return_value=big_route
        ):
            routes.append(router.decide(make_req("объясни"), cfg))

        self.assertEqual([r.kind for r in routes], ["skill", "small", "big"])
        for route in routes:
            self.assertTrue(route.reason.strip(), f"пустая причина у маршрута {route.kind}")


if __name__ == "__main__":
    unittest.main()
