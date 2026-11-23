"""Роутер: маркер локальной модели и граница prefill (задачи 0109 и 0120).

Сети и ``libvtm`` нет: подмены — фейковая локальная модель с формой
``VtmBackend`` (``chat``/``stream_chat``) и фейковый HTTP-клиент. Проверяем:

* ``Route.local`` — поле контракта, попадает в ``to_dict()`` как ``local`` и
  ``source`` (``skill``/``local``/``api``); у ``skill``/``big`` — ``None``;
* тот же маркер виден в ``usage["local"]`` и в записи JSONL, а доля своей
  модели считается в агрегате лога (``summary``);
* ``Answer.prefill_ms`` и ``prefill_ms`` финального куска потока заполняются,
  только если клиент модели явно сообщил границу prefill; молчит — ``None``.

Запуск: ``python3 -m unittest tests.test_eva_router -v``.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eva import router
from eva.config import Config
from eva.telemetry import Metrics, log, normalize_record, summary
from eva.types import Answer, Message, Request, Route


# --------------------------------------------------------------------------- #
# заглушки моделей
# --------------------------------------------------------------------------- #
class FakeLocalModel:
    """Локальная ВТМ без библиотеки: ``chat``/``stream_chat`` как у VtmBackend.

    ``prefill_ms`` — граница prefill, которую «сообщает» бэкенд;
    ``report_via`` — где именно: ``result`` (поле в результате/куске) или
    ``attr`` (атрибут ``last_prefill_ms`` после генерации). ``prefill_in`` —
    форма поля: ``timings``, ``usage`` или ``top`` (верхний уровень).
    """

    model_path = "/tmp/fake-model.vtm"

    def __init__(
        self,
        text: str = "ответ локальной ВТМ",
        pieces=("лок", "альный"),
        *,
        usage: dict | None = None,
        prefill_ms: float | None = None,
        report_via: str = "result",
        prefill_in: str = "timings",
    ) -> None:
        self.text = text
        self.pieces = list(pieces)
        self.usage = usage if usage is not None else {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "cost_rub": 0.0,
        }
        self.prefill_ms = prefill_ms
        self.report_via = report_via
        self.prefill_in = prefill_in
        self.last_prefill_ms: float | None = None
        self.calls: list[list[Message]] = []
        self.stream_calls: list[list[Message]] = []

    # -- форма VtmBackend ---------------------------------------------------
    def generate(self, prompt_ids, **_kwargs):
        return list(prompt_ids)

    def _payload(self, text: str, usage: dict) -> dict:
        payload: dict = {"text": text, "usage": dict(usage)}
        if self.prefill_ms is None or self.report_via != "result":
            return payload
        if self.prefill_in == "top":
            payload["prefill_ms"] = self.prefill_ms
        elif self.prefill_in == "usage":
            payload["usage"]["prefill_ms"] = self.prefill_ms
        else:
            payload["timings"] = {"prefill_ms": self.prefill_ms}
        return payload

    def chat(self, messages, max_tokens: int = 512):
        self.calls.append(list(messages))
        if self.report_via == "attr" and self.prefill_ms is not None:
            self.last_prefill_ms = self.prefill_ms
        return self._payload(self.text, self.usage)

    def stream_chat(self, messages, max_tokens: int = 512):
        self.stream_calls.append(list(messages))
        for index, piece in enumerate(self.pieces):
            if index == 0 and self.report_via == "result" and self.prefill_ms is not None:
                yield self._payload(piece, {})
            else:
                yield piece
        if self.report_via == "attr" and self.prefill_ms is not None:
            self.last_prefill_ms = self.prefill_ms


class FakeHttpModel:
    """HTTP-клиент модели: ``base``/``endpoint`` есть, сети нет."""

    def __init__(self, text: str = "ответ HTTP-клиента", pieces=("http", "-ответ")) -> None:
        self.text = text
        self.pieces = list(pieces)
        self.base = "http://127.0.0.1:9/v1"
        self.endpoint = "http://127.0.0.1:9/v1/chat/completions"
        self.calls: list[list[Message]] = []
        self.stream_calls: list[list[Message]] = []

    def chat(self, messages, max_tokens: int = 512):
        self.calls.append(list(messages))
        return {
            "text": self.text,
            "usage": {"prompt_tokens": 20, "completion_tokens": 5, "cost_rub": 0.0},
        }

    def stream_chat(self, messages, max_tokens: int = 512):
        self.stream_calls.append(list(messages))
        for piece in self.pieces:
            yield piece


def make_req(text: str = "привет") -> Request:
    return Request(messages=[Message(role="user", content=text)])


def small_route(_text, _cfg) -> Route:
    return Route("small", "тест: малая модель", 0.8)


def big_route(_text, _cfg) -> Route:
    return Route("big", "тест: большая модель", 0.9)


def text_of(chunks) -> str:
    return "".join(chunk.text for chunk in chunks if not chunk.done)


# --------------------------------------------------------------------------- #
# 0109: поле local в контракте Route
# --------------------------------------------------------------------------- #
class RouteLocalFieldTest(unittest.TestCase):
    def test_default_is_none_and_to_dict_has_marker(self) -> None:
        route = Route("small", "причина")

        self.assertIsNone(route.local)
        self.assertIsNone(route.source)
        data = route.to_dict()
        self.assertIn("local", data)
        self.assertIsNone(data["local"])
        self.assertIsNone(data["source"])

    def test_non_bool_marker_is_not_trusted(self) -> None:
        for value in ("true", 1, 0, None):
            with self.subTest(value=value):
                route = Route("small", "причина", local=value)
                self.assertIsNone(route.local)

    def test_source_is_derived_from_kind_and_marker(self) -> None:
        self.assertEqual(Route("small", "п", local=True).source, "local")
        self.assertEqual(Route("small", "п", local=False).source, "api")
        self.assertIsNone(Route("small", "п").source)
        self.assertEqual(Route("big", "п").source, "api")
        self.assertEqual(Route("skill", "п").source, "skill")


# --------------------------------------------------------------------------- #
# 0109: роутер помечает свои и чужие ответы
# --------------------------------------------------------------------------- #
class RouterLocalMarkerTest(unittest.TestCase):
    cfg = Config()

    def test_local_model_marks_route_and_usage(self) -> None:
        local = FakeLocalModel()
        deps = {"classifier": small_route, "small": local}

        answer = router.answer(make_req(), self.cfg, deps=deps)

        self.assertIs(answer.route.local, True)
        self.assertEqual(answer.route.source, "local")
        self.assertIs(answer.usage["local"], True)
        data = answer.to_dict()
        self.assertIs(data["route"]["local"], True)
        self.assertEqual(data["route"]["source"], "local")

    def test_http_small_marks_route_false(self) -> None:
        http = FakeHttpModel()
        deps = {"classifier": small_route, "small": http}

        answer = router.answer(make_req(), self.cfg, deps=deps)

        self.assertIs(answer.route.local, False)
        self.assertEqual(answer.route.source, "api")
        self.assertIs(answer.usage["local"], False)

    def test_skill_has_no_local_marker(self) -> None:
        deps = {"skills": lambda text, ctx: {"skill": "hours", "text": "10–19"}}

        answer = router.answer(make_req("часы"), self.cfg, deps=deps)

        self.assertEqual(answer.route.kind, "skill")
        self.assertIsNone(answer.route.local)
        self.assertEqual(answer.route.source, "skill")
        self.assertNotIn("local", answer.usage)
        self.assertIsNone(answer.to_dict()["route"]["local"])

    def test_big_is_api_without_local_marker(self) -> None:
        cfg = Config(big_base="https://api.example/v1", big_model="big-1")
        deps = {"classifier": big_route, "big": FakeHttpModel()}

        answer = router.answer(make_req("объясни"), cfg, deps=deps)

        self.assertEqual(answer.route.kind, "big")
        self.assertIsNone(answer.route.local)
        self.assertEqual(answer.route.source, "api")
        self.assertNotIn("local", answer.usage)

    def test_stream_final_chunk_marks_local(self) -> None:
        local = FakeLocalModel(pieces=("лок", "альный"))
        deps = {"classifier": small_route, "small": local}

        chunks = list(router.stream(make_req(), self.cfg, deps=deps))
        final = chunks[-1]

        self.assertTrue(final.done)
        self.assertIs(final.route.local, True)
        self.assertEqual(final.route.source, "local")
        self.assertIs(final.usage["local"], True)
        self.assertIs(final["route"].local, True)


# --------------------------------------------------------------------------- #
# 0109: тот же маркер в JSONL
# --------------------------------------------------------------------------- #
class TelemetryLocalRecordTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "eva.jsonl"

    def _answer(self, *, kind: str, local: bool | None, prefill_ms=None) -> Answer:
        return Answer(
            text="ответ",
            route=Route(kind, "тест", 0.5, local=local),
            usage={"prompt_tokens": 3, "completion_tokens": 4, "cost_rub": 0.0},
            latency_ms=12,
            prefill_ms=prefill_ms,
        )

    def test_local_answer_is_marked_true(self) -> None:
        record = log(self._answer(kind="small", local=True), self.path)

        parsed = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertIs(parsed["local"], True)
        self.assertEqual(parsed["source"], "local")
        self.assertIs(record["local"], True)
        self.assertEqual(record["source"], "local")

    def test_http_answer_is_marked_false(self) -> None:
        log(self._answer(kind="small", local=False), self.path)

        parsed = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertIs(parsed["local"], False)
        self.assertEqual(parsed["source"], "api")

    def test_skill_and_big_have_no_local_block(self) -> None:
        log(self._answer(kind="skill", local=None), self.path)
        log(self._answer(kind="big", local=None), self.path)

        first, second = [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertNotIn("local", first)
        self.assertEqual(first["source"], "skill")
        self.assertNotIn("local", second)
        self.assertEqual(second["source"], "api")

    def test_marker_from_route_dict_is_understood(self) -> None:
        record = normalize_record(
            {
                "route": {"kind": "small", "reason": "тест", "local": True},
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }
        )

        self.assertIs(record["local"], True)
        self.assertEqual(record["source"], "local")

    def test_local_share_in_log_summary(self) -> None:
        log(self._answer(kind="small", local=True), self.path)
        log(self._answer(kind="small", local=True), self.path)
        log(self._answer(kind="small", local=False), self.path)
        log(self._answer(kind="skill", local=None), self.path)

        report = summary(self.path)["total"]

        self.assertEqual(report["local_requests"], 2)
        self.assertAlmostEqual(report["local_share"], 0.5, places=6)

    def test_prefill_is_written_to_record(self) -> None:
        log(self._answer(kind="small", local=True, prefill_ms=180), self.path)

        parsed = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(parsed["prefill_ms"], 180.0)

    def test_prefill_from_usage_is_understood(self) -> None:
        record = normalize_record(
            {
                "route": {"kind": "small", "local": True},
                "usage": {"prompt_tokens": 2, "prefill_ms": 90},
            }
        )

        self.assertEqual(record["prefill_ms"], 90.0)


# --------------------------------------------------------------------------- #
# 0120: граница prefill в Answer и финальном куске потока
# --------------------------------------------------------------------------- #
class PrefillBoundaryTest(unittest.TestCase):
    cfg = Config()

    def _answer_with(self, model) -> Answer:
        deps = {"classifier": small_route, "small": model}
        return router.answer(make_req(), self.cfg, deps=deps)

    def test_prefill_from_result_timings(self) -> None:
        answer = self._answer_with(FakeLocalModel(prefill_ms=120))

        self.assertEqual(answer.prefill_ms, 120.0)
        self.assertEqual(answer.to_dict()["prefill_ms"], 120.0)

    def test_prefill_from_top_level_and_usage(self) -> None:
        for form in ("top", "usage"):
            with self.subTest(form=form):
                answer = self._answer_with(
                    FakeLocalModel(prefill_ms=42.5, prefill_in=form)
                )
                self.assertEqual(answer.prefill_ms, 42.5)

    def test_prefill_from_client_attribute_after_generation(self) -> None:
        answer = self._answer_with(
            FakeLocalModel(prefill_ms=77, report_via="attr")
        )

        self.assertEqual(answer.prefill_ms, 77.0)

    def test_silent_client_leaves_prefill_none(self) -> None:
        for model in (FakeLocalModel(), FakeHttpModel()):
            with self.subTest(model=type(model).__name__):
                self.assertIsNone(self._answer_with(model).prefill_ms)

    def test_http_client_reporting_prefill_is_passed_through(self) -> None:
        class TimingHttp(FakeHttpModel):
            def chat(self, messages, max_tokens: int = 512):
                payload = super().chat(messages, max_tokens)
                payload["timings"] = {"prefill_ms": 33}
                return payload

        self.assertEqual(self._answer_with(TimingHttp()).prefill_ms, 33.0)

    def test_stream_final_chunk_carries_prefill_from_piece(self) -> None:
        deps = {"classifier": small_route, "small": FakeLocalModel(prefill_ms=90)}
        chunks = list(router.stream(make_req(), self.cfg, deps=deps))
        final = chunks[-1]

        self.assertTrue(final.done)
        self.assertEqual(final.prefill_ms, 90.0)
        self.assertEqual(final["prefill_ms"], 90.0)
        self.assertIsNone(chunks[0].prefill_ms, "текстовые куски prefill не несут")

    def test_stream_final_chunk_prefill_from_client_attribute(self) -> None:
        local = FakeLocalModel(prefill_ms=64, report_via="attr")
        deps = {"classifier": small_route, "small": local}

        chunks = list(router.stream(make_req(), self.cfg, deps=deps))

        self.assertEqual(chunks[-1].prefill_ms, 64.0)

    def test_stream_without_data_leaves_prefill_none(self) -> None:
        deps = {"classifier": small_route, "small": FakeLocalModel()}
        chunks = list(router.stream(make_req(), self.cfg, deps=deps))

        self.assertIsNone(chunks[-1].prefill_ms)

    def test_skill_has_no_prefill(self) -> None:
        deps = {"skills": lambda text, ctx: {"skill": "hours", "text": "10–19"}}
        answer = router.answer(make_req("часы"), self.cfg, deps=deps)

        self.assertIsNone(answer.prefill_ms)


# --------------------------------------------------------------------------- #
# 0109: счётчики самой метрики (small_local / small_http и доля)
# --------------------------------------------------------------------------- #
class MetricsLocalCountersTest(unittest.TestCase):
    def test_share_is_computed_over_marked_small_answers(self) -> None:
        metrics = Metrics()
        metrics.observe(route="small", finish="ok", local=True)
        metrics.observe(route="small", finish="ok", local=True)
        metrics.observe(route="small", finish="ok", local=False)
        metrics.observe(route="big", finish="ok", local=False)
        metrics.observe(route="skill", finish="ok")

        snapshot = metrics.snapshot()

        self.assertEqual(snapshot["small_local"], 2)
        self.assertEqual(snapshot["small_http"], 1)
        self.assertEqual(snapshot["local_requests"], 2)
        self.assertAlmostEqual(snapshot["local_share"], 2 / 3, places=6)
        self.assertAlmostEqual(snapshot["local_share_all"], 2 / 5, places=6)
        self.assertEqual(
            snapshot["source"], {"skill": 1, "local": 2, "api": 2}
        )
        self.assertEqual(snapshot["by_route"]["small"]["local"], 2)

    def test_unmarked_small_does_not_land_in_share(self) -> None:
        metrics = Metrics()
        metrics.observe(route="small", finish="ok")
        metrics.observe(route="small", finish="ok", local=True)

        snapshot = metrics.snapshot()

        self.assertEqual(snapshot["small_local"], 1)
        self.assertEqual(snapshot["small_http"], 0)
        self.assertEqual(snapshot["local_share"], 1.0)

    def test_empty_snapshot_has_null_share(self) -> None:
        snapshot = Metrics().snapshot()

        self.assertEqual(snapshot["small_local"], 0)
        self.assertEqual(snapshot["small_http"], 0)
        self.assertIsNone(snapshot["local_share"])
        self.assertIsNone(snapshot["local_share_all"])

    def test_reset_clears_local_counters(self) -> None:
        metrics = Metrics()
        metrics.observe(route="small", finish="ok", local=True)
        metrics.reset()

        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["small_local"], 0)
        self.assertEqual(snapshot["local_requests"], 0)
        self.assertEqual(snapshot["source"], {"skill": 0, "local": 0, "api": 0})

    def test_record_returned_by_observe_has_marker(self) -> None:
        record = Metrics().observe(route="small", finish="ok", local=True)

        self.assertIs(record["local"], True)
        self.assertEqual(record["source"], "local")


if __name__ == "__main__":
    unittest.main()
