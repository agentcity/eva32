"""Тесты локальной ВТМ как «маленькой модели» маршрутизатора (задача 0104).

Сети и ``libvtm`` нет: подмены — фейковая локальная модель с формой
``VtmBackend`` (``chat``/``stream_chat``) и фейковый HTTP-клиент. Проверяем:

* маршрут ``small`` уходит в локальную модель, если она передана/настроена;
* ошибка локальной модели (``VtmUnavailable`` и любая другая) не роняет ответ,
  а откатывается на HTTP-клиент малой, затем — на большую по политике;
* маркер ``local`` отличает локальный ответ от HTTP, счётчики не ломаются;
* тот же порядок работает в потоке, но только до первого отданного куска.

Запуск: ``python3 -m unittest tests.test_router_vtm -v``.
"""

from __future__ import annotations

import unittest
from unittest import mock

from eva import router
from eva.backends.vtm_backend import VtmUnavailable
from eva.config import Config
from eva.types import Message, Request, Route


# --------------------------------------------------------------------------- #
# заглушки
# --------------------------------------------------------------------------- #
class FakeLocalModel:
    """Локальная ВТМ без библиотеки: ``chat``/``stream_chat`` как у VtmBackend.

    ``model_path``/``generate`` — duck-typing-признаки локальной модели
    (``base``/``endpoint`` у неё нет, в отличие от HTTP-клиента).
    """

    def __init__(
        self,
        text: str = "ответ локальной ВТМ",
        pieces=("лок", "альный"),
        *,
        usage: dict | None = None,
        error: Exception | None = None,
        stream_error: Exception | None = None,
        midstream_error: Exception | None = None,
    ) -> None:
        self.text = text
        self.pieces = list(pieces)
        self.usage = usage if usage is not None else {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "cost_rub": 0.0,
        }
        self.error = error
        self.stream_error = stream_error
        self.midstream_error = midstream_error
        self.model_path = "/tmp/fake-model.vtm"
        self.calls: list[tuple[list[Message], int]] = []
        self.stream_calls: list[tuple[list[Message], int]] = []

    # -- форма VtmBackend ---------------------------------------------------
    def generate(self, prompt_ids, **_kwargs):
        return list(prompt_ids)

    def chat(self, messages, max_tokens: int = 512):
        self.calls.append((list(messages), max_tokens))
        if self.error is not None:
            raise self.error
        return {"text": self.text, "usage": dict(self.usage), "model": "fake.vtm"}

    def stream_chat(self, messages, max_tokens: int = 512):
        self.stream_calls.append((list(messages), max_tokens))
        if self.stream_error is not None:
            raise self.stream_error
        for index, piece in enumerate(self.pieces):
            if index and self.midstream_error is not None:
                raise self.midstream_error
            yield piece


class FakeFlaggedLocalModel:
    """Локальная модель, помеченная только явным флагом ``is_local``."""

    is_local = True

    def __init__(self, text: str = "флаговая ВТМ") -> None:
        self.text = text
        self.calls: list[list[Message]] = []

    def chat(self, messages, max_tokens: int = 512):
        self.calls.append(list(messages))
        return {"text": self.text, "usage": {"prompt_tokens": 1, "completion_tokens": 1}}


class FakeHttpModel:
    """HTTP-клиент малой/большой модели: ``base`` есть, сети нет."""

    def __init__(
        self,
        text: str = "ответ HTTP-клиента",
        pieces=("http", "-ответ"),
        *,
        usage: dict | None = None,
        error: Exception | None = None,
        stream_error: Exception | None = None,
        midstream_error: Exception | None = None,
    ) -> None:
        self.text = text
        self.pieces = list(pieces)
        self.usage = usage if usage is not None else {
            "prompt_tokens": 20,
            "completion_tokens": 5,
            "cost_rub": 0.02,
        }
        self.error = error
        self.stream_error = stream_error
        self.midstream_error = midstream_error
        self.base = "http://127.0.0.1:9/v1"
        self.endpoint = "http://127.0.0.1:9/v1/chat/completions"
        self.calls: list[tuple[list[Message], int]] = []
        self.stream_calls: list[tuple[list[Message], int]] = []

    def chat(self, messages, max_tokens: int = 512):
        self.calls.append((list(messages), max_tokens))
        if self.error is not None:
            raise self.error
        return {"text": self.text, "usage": dict(self.usage), "raw": {"stub": True}}

    def stream_chat(self, messages, max_tokens: int = 512):
        self.stream_calls.append((list(messages), max_tokens))
        if self.stream_error is not None:
            raise self.stream_error
        for index, piece in enumerate(self.pieces):
            if index and self.midstream_error is not None:
                raise self.midstream_error
            yield piece


class FakeClientError(Exception):
    """Ошибка HTTP-клиента, которую политика умеет классифицировать."""


def make_req(text: str = "привет") -> Request:
    return Request(
        messages=[Message(role="system", content="ты ассистент"), Message(role="user", content=text)],
        meta={"user_id": "u-1", "channel": "test"},
    )


def small_route(text: str = "", cfg=None) -> Route:
    return Route("small", "classifier: короткий вопрос — хватит малой модели", 0.8)


def big_route(text: str = "", cfg=None) -> Route:
    return Route("big", "classifier: нужен код — большая модель", 0.9)


def text_of(chunks) -> str:
    return "".join(chunk.text for chunk in chunks)


class RouterTestCase(unittest.TestCase):
    """Общая обвязка: скиллы молчат, клиенты — только из deps/патчей."""

    def setUp(self) -> None:
        self.cfg = Config()
        patcher = mock.patch.object(router, "_load_skills", return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)


# --------------------------------------------------------------------------- #
# конфиг: EVA_LOCAL_MODEL и приоритет
# --------------------------------------------------------------------------- #
class LocalModelConfigTest(unittest.TestCase):
    def test_local_model_is_off_by_default(self) -> None:
        cfg = Config()
        self.assertFalse(cfg.local_available)
        self.assertEqual(cfg.local_model_path, "")
        self.assertTrue(cfg.local_first, "приоритет локальной модели — по умолчанию")

    def test_local_model_from_env(self) -> None:
        cfg = Config.from_env({"EVA_LOCAL_MODEL": " /models/eva.vtm "})
        self.assertTrue(cfg.local_available)
        self.assertEqual(cfg.local_model, "/models/eva.vtm")
        self.assertEqual(cfg.local_model_path, "/models/eva.vtm")
        self.assertTrue(cfg.local_first)

    def test_priority_flag_from_env(self) -> None:
        for value in ("0", "false", "off", "no", "нет"):
            cfg = Config.from_env({"EVA_LOCAL_MODEL": "m.vtm", "EVA_LOCAL_MODEL_PRIORITY": value})
            self.assertFalse(cfg.local_first, value)
        for value in ("1", "true", "on", "yes", "да"):
            cfg = Config.from_env({"EVA_LOCAL_MODEL": "m.vtm", "EVA_LOCAL_MODEL_PRIORITY": value})
            self.assertTrue(cfg.local_first, value)

    def test_garbage_priority_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Config.from_env({"EVA_LOCAL_MODEL_PRIORITY": "может быть"})

    def test_local_model_path_property_strips_spaces(self) -> None:
        self.assertEqual(Config(local_model="  a.vtm  ").local_model_path, "a.vtm")
        self.assertFalse(Config(local_model="   ").local_available)


# --------------------------------------------------------------------------- #
# распознавание локальной модели
# --------------------------------------------------------------------------- #
class LocalModelDetectionTest(unittest.TestCase):
    def test_duck_typed_local_model(self) -> None:
        self.assertTrue(router.is_local_model(FakeLocalModel()))

    def test_explicit_flag(self) -> None:
        self.assertTrue(router.is_local_model(FakeFlaggedLocalModel()))

    def test_http_client_is_not_local(self) -> None:
        self.assertFalse(router.is_local_model(FakeHttpModel()))
        from eva.clients.openai_compat import OpenAIClient

        self.assertFalse(router.is_local_model(OpenAIClient("http://127.0.0.1:9", "", "m")))

    def test_backends_module_is_local(self) -> None:
        class Vendored:
            def chat(self, messages, max_tokens: int = 512):
                return {"text": "ok"}

        Vendored.__module__ = "eva.backends.vtm_backend"
        self.assertTrue(router.is_local_model(Vendored()))

    def test_none_and_mock_are_not_local(self) -> None:
        self.assertFalse(router.is_local_model(None))
        # Mock отвечает на любой getattr; булевым маркером это считаться не должно.
        self.assertFalse(router.is_local_model(mock.Mock()))

    def test_route_is_local_reads_marker(self) -> None:
        marked = Route("small", "причина")
        marked.local = True
        self.assertIs(router.route_is_local(marked), True)
        self.assertIsNone(router.route_is_local(Route("small", "причина")))
        self.assertIsNone(router.route_is_local(None))


# --------------------------------------------------------------------------- #
# маршрут small идёт в локальную модель
# --------------------------------------------------------------------------- #
class LocalModelRouteTest(RouterTestCase):
    def test_injected_local_model_answers_small_route(self) -> None:
        local = FakeLocalModel()
        http = FakeHttpModel()
        deps = {"classifier": small_route, "small": local, "big": http}
        with mock.patch.object(router, "_make_client", return_value=http):
            answer = router.answer(make_req(), self.cfg, deps=deps)
        self.assertEqual(answer.route.kind, "small")
        self.assertEqual(answer.text, "ответ локальной ВТМ")
        self.assertEqual(len(local.calls), 1, "маршрут обязан идти в локальную ВТМ")
        self.assertEqual(http.calls, [], "HTTP-клиент при живом local не зовётся")
        self.assertIs(answer.usage["local"], True)
        self.assertIs(router.route_is_local(answer.route), True)
        self.assertIs(answer.route.local, True)
        # счётчики не сломаны: токены и цена из usage локальной модели
        self.assertEqual(answer.usage["prompt_tokens"], 11)
        self.assertEqual(answer.usage["completion_tokens"], 7)
        self.assertEqual(answer.usage["cost_rub"], 0.0)
        self.assertEqual(answer.cost_rub, 0.0)

    def test_local_model_with_explicit_flag(self) -> None:
        local = FakeFlaggedLocalModel()
        deps = {"classifier": small_route, "small": local}
        with mock.patch.object(router, "_make_client", return_value=None):
            answer = router.answer(make_req(), self.cfg, deps=deps)
        self.assertEqual(answer.text, "флаговая ВТМ")
        self.assertIs(answer.usage["local"], True)
        self.assertEqual(len(local.calls), 1)

    def test_http_small_route_is_marked_local_false(self) -> None:
        http = FakeHttpModel()
        deps = {"classifier": small_route, "small": http}
        answer = router.answer(make_req(), self.cfg, deps=deps)
        self.assertEqual(answer.text, "ответ HTTP-клиента")
        self.assertIs(answer.usage["local"], False)
        self.assertIs(router.route_is_local(answer.route), False)
        self.assertEqual(len(http.calls), 1)

    def test_configured_local_model_answers_without_deps_small(self) -> None:
        local = FakeLocalModel(text="ВТМ из конфига")
        cfg = Config(local_model="/models/eva.vtm")
        deps = {"classifier": small_route}
        with mock.patch.object(router, "_make_local_client", return_value=local), \
                mock.patch.object(router, "_make_client", return_value=None):
            answer = router.answer(make_req(), cfg, deps=deps)
        self.assertEqual(answer.text, "ВТМ из конфига")
        self.assertEqual(answer.route.kind, "small")
        self.assertIs(answer.usage["local"], True)
        self.assertEqual(len(local.calls), 1)

    def test_priority_off_prefers_http_small(self) -> None:
        local = FakeLocalModel(text="не должно быть")
        http = FakeHttpModel(text="ответ HTTP")
        cfg = Config(local_model="/models/eva.vtm", local_model_priority=False)
        deps = {"classifier": small_route}

        def make_client(kind, _cfg):
            return http if kind == "small" else None

        with mock.patch.object(router, "_make_local_client", return_value=local), \
                mock.patch.object(router, "_make_client", side_effect=make_client):
            answer = router.answer(make_req(), cfg, deps=deps)
        self.assertEqual(answer.text, "ответ HTTP")
        self.assertIs(answer.usage["local"], False)
        self.assertEqual(local.calls, [], "при выключенном приоритете первым решает HTTP")

    def test_priority_off_uses_local_when_http_fails(self) -> None:
        local = FakeLocalModel(text="локальный запасной")
        http = FakeHttpModel(error=FakeClientError("connection refused"))
        cfg = Config(local_model="/models/eva.vtm", local_model_priority=False)
        deps = {"classifier": small_route}

        def make_client(kind, _cfg):
            return http if kind == "small" else None

        with mock.patch.object(router, "_make_local_client", return_value=local), \
                mock.patch.object(router, "_make_client", side_effect=make_client):
            answer = router.answer(make_req(), cfg, deps=deps)
        self.assertEqual(answer.text, "локальный запасной")
        self.assertIs(answer.usage["local"], True)
        self.assertEqual(len(http.calls), 1)
        self.assertEqual(len(local.calls), 1)
        self.assertIn("HTTP-клиент малой модели не ответил", answer.route.reason)

    def test_missing_vtm_backend_falls_back_to_http(self) -> None:
        """Модуля ВТМ нет/он сломан — локальной модели просто нет, HTTP работает."""
        http = FakeHttpModel()
        cfg = Config(local_model="/models/eva.vtm")
        deps = {"classifier": small_route}
        with mock.patch.dict("sys.modules", {"eva.backends.vtm_backend": None}), \
                mock.patch.object(router, "_make_client", return_value=http):
            self.assertIsNone(router._make_local_client(cfg))
            answer = router.answer(make_req(), cfg, deps=deps)
        self.assertEqual(answer.text, "ответ HTTP-клиента")
        self.assertIs(answer.usage["local"], False)

    def test_broken_http_small_config_does_not_break_local(self) -> None:
        """Кривой EVA_SMALL_BASE не отменяет локальную ВТМ (HTTP — запасной)."""
        local = FakeLocalModel(text="ВТМ при кривом HTTP")
        cfg = Config(local_model="/models/eva.vtm", small_base="ftp://nope", small_model="m")
        deps = {"classifier": small_route}
        with mock.patch.object(router, "_make_local_client", return_value=local):
            answer = router.answer(make_req(), cfg, deps=deps)
        self.assertEqual(answer.text, "ВТМ при кривом HTTP")
        self.assertIs(answer.usage["local"], True)


# --------------------------------------------------------------------------- #
# откат при ошибке локальной модели
# --------------------------------------------------------------------------- #
class LocalModelFallbackTest(RouterTestCase):
    def test_vtm_unavailable_falls_back_to_http_small(self) -> None:
        local = FakeLocalModel(error=VtmUnavailable("нет libvtm"))
        http = FakeHttpModel()
        cfg = Config(local_model="/models/eva.vtm")
        deps = {"classifier": small_route, "small": local}

        def make_client(kind, _cfg):
            return http if kind == "small" else None

        with mock.patch.object(router, "_make_client", side_effect=make_client):
            answer = router.answer(make_req(), cfg, deps=deps)
        self.assertEqual(answer.route.kind, "small")
        self.assertEqual(answer.text, "ответ HTTP-клиента")
        self.assertEqual(len(local.calls), 1)
        self.assertEqual(len(http.calls), 1)
        self.assertIs(answer.usage["local"], False)
        self.assertIn("локальная модель не ответила", answer.route.reason)
        self.assertIn("откат на HTTP-клиент малой модели", answer.route.reason)
        # счётчики HTTP-ответа сохранены
        self.assertEqual(answer.usage["prompt_tokens"], 20)
        self.assertAlmostEqual(answer.cost_rub, 0.02, places=6)

    def test_any_local_error_falls_back_to_http(self) -> None:
        local = FakeLocalModel(error=RuntimeError("vtm_api_generate: код -3"))
        http = FakeHttpModel(text="HTTP после сбоя ВТМ")
        deps = {"classifier": small_route, "small": local}
        with mock.patch.object(router, "_make_client", return_value=http):
            answer = router.answer(make_req(), self.cfg, deps=deps)
        self.assertEqual(answer.text, "HTTP после сбоя ВТМ")
        self.assertIs(answer.usage["local"], False)
        self.assertIn("vtm_api_generate", answer.route.reason)

    def test_local_error_without_http_falls_back_to_big(self) -> None:
        local = FakeLocalModel(error=VtmUnavailable("нет модели"))
        big = FakeHttpModel(text="ответ большой", usage={"prompt_tokens": 30, "completion_tokens": 9, "cost_rub": 0.5})
        cfg = Config(big_base="https://api.example/v1", big_model="big-1")
        deps = {"classifier": small_route, "small": local, "big": big}

        def make_client(kind, _cfg):
            return None

        with mock.patch.object(router, "_make_client", side_effect=make_client):
            answer = router.answer(make_req(), cfg, deps=deps)
        self.assertEqual(answer.route.kind, "big")
        self.assertEqual(answer.text, "ответ большой")
        self.assertIn("fallback small→big", answer.route.reason)
        self.assertIn("нет модели", answer.route.reason, "причина локальной ошибки сохраняется")
        self.assertEqual(len(local.calls), 1)
        self.assertEqual(len(big.calls), 1)
        self.assertIs(router.route_is_local(answer.route), None, "big — не локальный маршрут")
        self.assertNotIn("local", answer.usage)

    def test_local_error_without_any_model_reraises(self) -> None:
        local = FakeLocalModel(error=VtmUnavailable("нечего запускать"))
        deps = {"classifier": small_route, "small": local}
        with mock.patch.object(router, "_make_client", return_value=None):
            with self.assertRaises(VtmUnavailable):
                router.answer(make_req(), self.cfg, deps=deps)
        self.assertEqual(len(local.calls), 1)

    def test_local_failure_when_big_over_budget_reraises(self) -> None:
        local = FakeLocalModel(error=VtmUnavailable("нет модели"))
        big = FakeHttpModel()
        cfg = Config(
            big_base="https://api.example/v1",
            big_model="big-1",
            max_cost_rub=0.0,
            big_price_prompt_rub_per_1k=1.0,
            big_price_completion_rub_per_1k=1.0,
        )
        deps = {"classifier": small_route, "small": local, "big": big}
        with mock.patch.object(router, "_make_client", return_value=None):
            with self.assertRaises(VtmUnavailable):
                router.answer(make_req(), cfg, deps=deps)
        self.assertEqual(big.calls, [])


# --------------------------------------------------------------------------- #
# поток
# --------------------------------------------------------------------------- #
class LocalModelStreamTest(RouterTestCase):
    def test_stream_uses_local_model(self) -> None:
        local = FakeLocalModel(pieces=("лок", "альный", " ответ"))
        deps = {"classifier": small_route, "small": local}
        with mock.patch.object(router, "_make_client", return_value=None):
            chunks = list(router.stream(make_req(), self.cfg, deps=deps))
        self.assertEqual(text_of(chunks), "локальный ответ")
        final = chunks[-1]
        self.assertTrue(final.done)
        self.assertEqual(final.route.kind, "small")
        self.assertIs(final.usage["local"], True)
        self.assertIs(router.route_is_local(final.route), True)
        self.assertEqual(len(local.stream_calls), 1)

    def test_stream_local_error_before_first_piece_falls_back_to_http(self) -> None:
        local = FakeLocalModel(stream_error=VtmUnavailable("нет libvtm"))
        http = FakeHttpModel(pieces=("http", "-поток"))
        deps = {"classifier": small_route, "small": local}

        def make_client(kind, _cfg):
            return http if kind == "small" else None

        with mock.patch.object(router, "_make_client", side_effect=make_client):
            chunks = list(router.stream(make_req(), self.cfg, deps=deps))
        self.assertEqual(text_of(chunks), "http-поток")
        final = chunks[-1]
        self.assertIs(final.usage["local"], False)
        self.assertIn("локальная модель не ответила", final.route.reason)
        self.assertEqual(len(local.stream_calls), 1)
        self.assertEqual(len(http.stream_calls), 1)

    def test_stream_local_error_after_first_piece_is_not_retried(self) -> None:
        local = FakeLocalModel(
            pieces=("лок", "альный"), midstream_error=VtmUnavailable("обрыв")
        )
        http = FakeHttpModel(pieces=("http",))
        deps = {"classifier": small_route, "small": local}
        chunks = []
        with mock.patch.object(router, "_make_client", return_value=http):
            with self.assertRaises(VtmUnavailable):
                for chunk in router.stream(make_req(), self.cfg, deps=deps):
                    chunks.append(chunk)
        self.assertEqual(text_of(chunks), "лок", "часть ответа уже у клиента")
        self.assertEqual(http.stream_calls, [], "повтор после первого куска запрещён")

    def test_stream_http_small_is_marked_local_false(self) -> None:
        http = FakeHttpModel()
        deps = {"classifier": small_route, "small": http}
        chunks = list(router.stream(make_req(), self.cfg, deps=deps))
        self.assertIs(chunks[-1].usage["local"], False)
        self.assertIs(router.route_is_local(chunks[-1].route), False)


# --------------------------------------------------------------------------- #
# прочие маршруты не задеты
# --------------------------------------------------------------------------- #
class OtherRoutesTest(RouterTestCase):
    def test_skill_answer_has_no_local_marker(self) -> None:
        deps = {"skills": lambda text, ctx: {"skill": "hours", "text": "10–19"}}
        answer = router.answer(make_req("часы работы"), self.cfg, deps=deps)
        self.assertEqual(answer.route.kind, "skill")
        self.assertNotIn("local", answer.usage)
        self.assertIsNone(router.route_is_local(answer.route))

    def test_big_route_has_no_local_marker(self) -> None:
        big = FakeHttpModel(text="ответ большой")
        deps = {"classifier": big_route, "big": big}
        cfg = Config(big_base="https://api.example/v1", big_model="big-1")
        answer = router.answer(make_req("объясни"), cfg, deps=deps)
        self.assertEqual(answer.route.kind, "big")
        self.assertNotIn("local", answer.usage)
        self.assertIsNone(router.route_is_local(answer.route))

    def test_object_deps_with_local_model(self) -> None:
        class Deps:
            classifier = staticmethod(small_route)
            small = FakeLocalModel(text="ВТМ из объекта deps")

        with mock.patch.object(router, "_make_client", return_value=None):
            answer = router.answer(make_req(), self.cfg, deps=Deps())
        self.assertEqual(answer.text, "ВТМ из объекта deps")
        self.assertIs(answer.usage["local"], True)


if __name__ == "__main__":
    unittest.main()
