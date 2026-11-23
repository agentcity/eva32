"""Метрики сервера и отмена генерации при обрыве SSE (задача 0090).

Запуск::

    python3 -m unittest tests.test_eva_server_metrics -v

Тесты идут в процессе: ни порта, ни сокета, ни сети. Обработчик
``eva.server`` вызывается напрямую, тело запроса подаётся через поддельные
``rfile``/``wfile``, часы — фейковые, маршрутизатор — генератор с известными
кусками. Обрыв клиента — это ``BrokenPipeError`` из поддельного ``wfile``;
так проверяется, что сервер закрывает генератор, не тянет следующие куски и
считает запрос отменённым.

Ручная сверка TTFT (``test_ttft_is_measured_from_request_to_first_chunk``):
часы стартуют в 0, каждый кусок «генерируется» за 0.125 с, метаданные
маршрута приходят третьим куском. Первый чанк отдан на 0.125 с → TTFT 125 мс,
поток закрыт на 0.375 с → полное время 375 мс.
"""

from __future__ import annotations

import io
import json
import sys
import threading
import time
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eva import server  # noqa: E402
from eva.telemetry import Metrics, percentile  # noqa: E402


# --------------------------------------------------------------------------- #
# Поддельные часы, сокет и маршрутизатор
# --------------------------------------------------------------------------- #


class FakeClock:
    """Часы, которые двигает только тест (или генератор между кусками)."""

    def __init__(self, start: float = 0.0) -> None:
        self.value = float(start)
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        return self.value

    def advance(self, seconds: float) -> float:
        self.value += float(seconds)
        return self.value


class FakeWriter:
    """Поддельный ``wfile``: копит байты и умеет оборвать клиента.

    ``fail_after`` — сколько записей (заголовки, чанки) принять, прежде чем
    бросить ``BrokenPipeError``. Так воспроизводится уход клиента без сокета.
    """

    def __init__(self, fail_after: int | None = None) -> None:
        self.fail_after = fail_after
        self.writes: list[bytes] = []
        self.flushes = 0
        self.closed = False

    def write(self, data: bytes) -> int:
        payload = bytes(data)
        if self.fail_after is not None and len(self.writes) >= self.fail_after:
            raise BrokenPipeError("клиент закрыл соединение")
        self.writes.append(payload)
        return len(payload)

    def flush(self) -> None:
        self.flushes += 1

    def close(self) -> None:
        self.closed = True

    @property
    def raw(self) -> bytes:
        return b"".join(self.writes)


@dataclass
class StubRoute:
    kind: str = "small"
    reason: str = "тест"
    confidence: float = 0.9
    #: Маркер своей модели (задача 0109): True/False/None.
    local: bool | None = None


@dataclass
class StubAnswer:
    text: str = "готово"
    route: StubRoute = field(default_factory=StubRoute)
    skill: str | None = None
    usage: dict = field(
        default_factory=lambda: {"prompt_tokens": 3, "completion_tokens": 5}
    )
    cost_rub: float = 0.0
    latency_ms: int = 1
    #: Prefill-часть TTFT, если маршрутизатор её сообщает (задача 0115).
    prefill_ms: float | None = None


class ScriptedStreamRouter:
    """Генератор с известными кусками; между кусками двигает фейковые часы.

    ``pulled`` — сколько кусков сервер успел вытянуть из генератора, ``closed``
    — закрыли ли генератор снаружи (``GeneratorExit``), ``finished`` — доехал
    ли он до конца сам.
    """

    def __init__(
        self,
        clock: FakeClock,
        pieces=("раз", "два", "три"),
        *,
        step: float = 0.125,
        meta: StubAnswer | None = None,
        error: Exception | None = None,
        midstream_error: Exception | None = None,
    ) -> None:
        self.clock = clock
        self.pieces = list(pieces)
        self.step = step
        self.meta = meta
        self.error = error
        self.midstream_error = midstream_error
        self.answer = StubAnswer()
        self.requests = []
        self.pulled = 0
        self.finished = False
        self.closed = False

    def stream(self, request):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        try:
            for index, piece in enumerate(self.pieces):
                if index and self.midstream_error is not None:
                    raise self.midstream_error
                self.clock.advance(self.step)
                self.pulled += 1
                yield piece
            if self.meta is not None:
                self.clock.advance(self.step)
                self.pulled += 1
                yield self.meta
            self.finished = True
        finally:
            self.closed = True

    def __call__(self, request):
        self.requests.append(request)
        self.clock.advance(self.step)
        return self.answer


class FailingRouter:
    """Маршрутизатор, который всегда падает (для проверки счётчика error)."""

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    def __call__(self, request):
        self.calls += 1
        raise self.error


class StaticAnswerRouter:
    """Маршрутизатор без ``stream``: готовый ответ одним куском."""

    def __init__(self, answer: StubAnswer) -> None:
        self.answer = answer
        self.calls = 0

    def __call__(self, request):
        self.calls += 1
        return self.answer


# --------------------------------------------------------------------------- #
# Стенд обработчика без сокета
# --------------------------------------------------------------------------- #


def make_handler(
    router,
    *,
    metrics: Metrics,
    clock: FakeClock,
    writer: FakeWriter | None = None,
    guard: server.AccessGuard | None = None,
    headers: dict | None = None,
    path: str = "/v1/chat/completions",
    method: str = "POST",
):
    """Собрать обработчик так, как его собрал бы ``setup()`` BaseHTTPRequestHandler."""
    handler_class = server.make_handler(router)
    handler = handler_class.__new__(handler_class)
    handler.command = method
    handler.path = path
    handler.requestline = "{} {} HTTP/1.1".format(method, path)
    handler.request_version = "HTTP/1.1"
    handler.headers = dict(headers or {})
    handler.rfile = io.BytesIO(b"")
    handler.wfile = writer if writer is not None else FakeWriter()
    handler.client_address = ("127.0.0.1", 43210)
    handler.close_connection = False
    handler._sse_chunked = True
    handler.server = SimpleNamespace(
        router_name="injected",
        access_guard=guard,
        metrics=metrics,
        clock=clock,
        started_at=0.0,
    )
    return handler


def run_chat(
    router,
    *,
    metrics: Metrics,
    clock: FakeClock,
    writer: FakeWriter | None = None,
    guard: server.AccessGuard | None = None,
    payload: dict | None = None,
):
    """Выполнить ``POST /v1/chat/completions`` в процессе и вернуть обработчик."""
    body = json.dumps(
        {"messages": [{"role": "user", "content": "Привет"}], **(payload or {})},
        ensure_ascii=False,
    ).encode("utf-8")
    handler = make_handler(
        router,
        metrics=metrics,
        clock=clock,
        writer=writer,
        guard=guard,
        headers={"Content-Length": str(len(body)), "Content-Type": "application/json"},
    )
    handler.rfile = io.BytesIO(body)
    handler.do_POST()
    return handler


def run_get(path: str, *, metrics: Metrics, clock: FakeClock, guard=None):
    """Выполнить ``GET path`` в процессе (маршрутизатор не нужен)."""
    handler = make_handler(
        FailingRouter(RuntimeError("маршрутизатор в GET не вызывается")),
        metrics=metrics,
        clock=clock,
        guard=guard,
        path=path,
        method="GET",
    )
    handler.do_GET()
    return handler


def split_http(raw: bytes) -> tuple[int, dict[str, str], bytes]:
    """(статус, заголовки в нижнем регистре, тело) из поддельного ``wfile``."""
    head, _, body = raw.partition(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    status = int(lines[0].split()[1])
    headers: dict[str, str] = {}
    for line in lines[1:]:
        name, _, value = line.partition(":")
        if name.strip():
            headers[name.strip().lower()] = value.strip()
    return status, headers, body


def sse_events(raw: bytes) -> list[str]:
    """Поля ``data:`` событий SSE в порядке появления."""
    text = raw.decode("utf-8", "replace").replace("\r\n", "\n")
    return [
        line[len("data:"):].strip()
        for line in text.split("\n")
        if line.startswith("data:")
    ]


# --------------------------------------------------------------------------- #
# Счётчики и перцентили (telemetry.Metrics)
# --------------------------------------------------------------------------- #


class TestMetricsCounters(unittest.TestCase):
    def test_empty_snapshot_is_zeroed_and_json_safe(self):
        snapshot = Metrics().snapshot()

        self.assertEqual(snapshot["requests"], 0)
        self.assertEqual(snapshot["streams"], 0)
        self.assertEqual(
            snapshot["finish"], {"ok": 0, "stop": 0, "cancel": 0, "error": 0}
        )
        self.assertEqual(
            snapshot["routes"], {"skill": 0, "small": 0, "big": 0, "unknown": 0}
        )
        self.assertEqual(
            snapshot["tokens"], {"prompt": 0, "completion": 0, "total": 0}
        )
        self.assertIsNone(snapshot["ttft_ms"]["p50"])
        self.assertIsNone(snapshot["latency_ms"]["p95"])
        self.assertEqual(snapshot["ttft_ms"]["count"], 0)
        json.dumps(snapshot)  # /metrics отдаёт это как JSON без сюрпризов

    def test_percentiles_use_linear_interpolation(self):
        metrics = Metrics()
        for value in (100, 200, 300, 400):
            metrics.observe(
                route="small",
                finish="ok",
                ttft_ms=value,
                latency_ms=value + 50,
            )

        snapshot = metrics.snapshot()
        ttft = snapshot["ttft_ms"]
        latency = snapshot["latency_ms"]

        self.assertEqual(ttft["count"], 4)
        self.assertEqual(ttft["min"], 100)
        self.assertEqual(ttft["max"], 400)
        self.assertEqual(ttft["avg"], 250)
        self.assertEqual(
            (ttft["p50"], ttft["p95"], ttft["p99"]), (250.0, 385.0, 397.0)
        )
        for q in (50, 95, 99):
            self.assertEqual(ttft["p{}".format(q)], percentile([100, 200, 300, 400], q))
        self.assertEqual(
            (latency["p50"], latency["p95"], latency["p99"]),
            (300.0, 435.0, 447.0),
        )

    def test_finish_routes_and_tokens_are_counted(self):
        metrics = Metrics()
        metrics.observe(
            route="skill",
            finish="ok",
            latency_ms=10,
            prompt_tokens=4,
            completion_tokens=6,
        )
        metrics.observe(
            route="small",
            finish="stop",
            ttft_ms=50,
            latency_ms=20,
            prompt_tokens=10,
            completion_tokens=20,
            stream=True,
        )
        metrics.observe(
            route="big",
            finish="cancel",
            ttft_ms=400,
            latency_ms=30,
            prompt_tokens=100,
            stream=True,
        )
        metrics.observe(route="weird", finish="boom", latency_ms=40)

        snapshot = metrics.snapshot()

        self.assertEqual(snapshot["requests"], 4)
        self.assertEqual(snapshot["streams"], 2)
        self.assertEqual(
            snapshot["finish"], {"ok": 1, "stop": 1, "cancel": 1, "error": 1}
        )
        self.assertEqual(snapshot["cancelled"], 1)
        self.assertEqual(snapshot["errors"], 1)
        self.assertEqual(
            snapshot["routes"], {"skill": 1, "small": 1, "big": 1, "unknown": 1}
        )
        self.assertEqual(
            snapshot["tokens"], {"prompt": 114, "completion": 26, "total": 140}
        )
        self.assertEqual(snapshot["by_route"]["small"]["ttft_ms"]["p50"], 50.0)
        self.assertEqual(snapshot["by_route"]["big"]["finish"]["cancel"], 1)
        self.assertEqual(snapshot["by_route"]["big"]["tokens"], 100)
        self.assertEqual(snapshot["by_route"]["unknown"]["requests"], 1)

    def test_non_stream_request_has_no_ttft_but_has_latency(self):
        metrics = Metrics()
        metrics.observe(route="small", finish="ok", ttft_ms=None, latency_ms=123)

        snapshot = metrics.snapshot()

        self.assertEqual(snapshot["ttft_ms"]["count"], 0)
        self.assertIsNone(snapshot["ttft_ms"]["p50"])
        self.assertEqual(snapshot["latency_ms"]["count"], 1)
        self.assertEqual(snapshot["latency_ms"]["p50"], 123.0)

    def test_bad_values_are_normalized(self):
        metrics = Metrics()
        record = metrics.observe(
            route="BIG",
            finish="CANCELLED",
            ttft_ms=-5,
            latency_ms="abc",
            prompt_tokens=-3,
            completion_tokens="4.6",
            total_tokens=None,
        )

        self.assertEqual(record["route"], "big")
        self.assertEqual(record["finish"], "error")  # неизвестный код
        self.assertIsNone(record["ttft_ms"])
        self.assertEqual(record["latency_ms"], 0.0)
        self.assertEqual(record["prompt_tokens"], 0)
        self.assertEqual(record["completion_tokens"], 5)
        self.assertEqual(record["total_tokens"], 5)

    def test_percentile_window_is_bounded_and_counted(self):
        metrics = Metrics(max_samples=3)
        for value in (10, 20, 30, 40, 50):
            metrics.observe(route="small", finish="ok", ttft_ms=value, latency_ms=value)

        snapshot = metrics.snapshot()

        self.assertEqual(snapshot["requests"], 5)  # счётчик накопительный
        self.assertEqual(snapshot["ttft_ms"]["count"], 3)  # окно перцентилей
        self.assertEqual(snapshot["ttft_ms"]["min"], 30)
        self.assertEqual(snapshot["ttft_ms"]["max"], 50)
        self.assertEqual(snapshot["ttft_ms"]["p50"], 40.0)

    def test_concurrent_observations_are_counted(self):
        metrics = Metrics()

        def worker() -> None:
            for _ in range(100):
                metrics.observe(
                    route="small",
                    finish="ok",
                    prompt_tokens=1,
                    completion_tokens=2,
                    latency_ms=1,
                )

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["requests"], 800)
        self.assertEqual(
            snapshot["tokens"], {"prompt": 800, "completion": 1600, "total": 2400}
        )
        self.assertEqual(snapshot["latency_ms"]["count"], 800)

    def test_reset_clears_everything(self):
        metrics = Metrics()
        metrics.observe(route="big", finish="stop", ttft_ms=5, latency_ms=9)
        metrics.reset()

        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["requests"], 0)
        self.assertEqual(snapshot["routes"]["big"], 0)
        self.assertIsNone(snapshot["ttft_ms"]["p50"])


# --------------------------------------------------------------------------- #
# Метрики потока на сервере
# --------------------------------------------------------------------------- #


class TestServerStreamMetrics(unittest.TestCase):
    def test_ttft_is_measured_from_request_to_first_chunk(self):
        clock = FakeClock()
        meta = StubAnswer(
            text="",
            route=StubRoute(kind="big"),
            usage={"prompt_tokens": 7, "completion_tokens": 5},
        )
        router = ScriptedStreamRouter(clock, pieces=("При", "вет"), meta=meta)
        metrics = Metrics()

        handler = run_chat(
            router,
            metrics=metrics,
            clock=clock,
            payload={"stream": True, "stream_options": {"include_usage": True}},
        )

        status, headers, _ = split_http(handler.wfile.raw)
        self.assertEqual(status, 200)
        self.assertIn("text/event-stream", headers["content-type"])

        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["requests"], 1)
        self.assertEqual(snapshot["streams"], 1)
        self.assertEqual(snapshot["finish"]["stop"], 1)
        self.assertEqual(snapshot["routes"]["big"], 1)
        self.assertEqual(
            snapshot["tokens"], {"prompt": 7, "completion": 5, "total": 12}
        )
        # 0.125 с до первого куска, 0.375 с до терминального чанка.
        self.assertEqual(snapshot["ttft_ms"]["p50"], 125.0)
        self.assertEqual(snapshot["latency_ms"]["p50"], 375.0)
        self.assertEqual(sse_events(handler.wfile.raw)[-1], "[DONE]")

    def test_client_disconnect_before_first_chunk_cancels_generation(self):
        clock = FakeClock()
        router = ScriptedStreamRouter(
            clock, pieces=("раз", "два", "три", "четыре", "пять")
        )
        metrics = Metrics()
        # 1 запись — заголовки SSE, на первом чанке клиент «уходит».
        writer = FakeWriter(fail_after=1)

        handler = run_chat(
            router,
            metrics=metrics,
            clock=clock,
            writer=writer,
            payload={"stream": True},
        )

        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["requests"], 1)
        self.assertEqual(snapshot["finish"]["cancel"], 1)
        self.assertEqual(snapshot["cancelled"], 1)
        self.assertEqual(snapshot["finish"]["stop"], 0)
        self.assertIsNone(snapshot["ttft_ms"]["p50"])
        self.assertNotIn(b"[DONE]", writer.raw)  # обрыв — терминального чанка нет
        self.assertTrue(handler.close_connection)

    def test_client_disconnect_stops_pulling_and_closes_generator(self):
        clock = FakeClock()
        router = ScriptedStreamRouter(
            clock, pieces=("раз", "два", "три", "четыре", "пять")
        )
        metrics = Metrics()
        writer = FakeWriter(fail_after=1)

        run_chat(
            router,
            metrics=metrics,
            clock=clock,
            writer=writer,
            payload={"stream": True},
        )

        self.assertTrue(router.closed, "генератор роутера должен быть закрыт")
        self.assertFalse(router.finished, "генерация не должна доходить до конца")
        self.assertEqual(router.pulled, 1, "лишние куски после обрыва не тянутся")

    def test_disconnect_after_first_chunk_keeps_ttft_and_counts_cancel(self):
        clock = FakeClock()
        router = ScriptedStreamRouter(clock, pieces=("раз", "два", "три"))
        metrics = Metrics()
        # Заголовки + первый чанк проходят, второй чанк уже некуда писать.
        writer = FakeWriter(fail_after=2)

        run_chat(
            router,
            metrics=metrics,
            clock=clock,
            writer=writer,
            payload={"stream": True},
        )

        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["finish"]["cancel"], 1)
        self.assertEqual(snapshot["ttft_ms"]["count"], 1)
        self.assertEqual(snapshot["ttft_ms"]["p50"], 125.0)
        self.assertEqual(router.pulled, 2)
        self.assertTrue(router.closed)

    def test_error_before_first_chunk_is_counted_as_error(self):
        clock = FakeClock()
        router = ScriptedStreamRouter(
            clock, pieces=("раз",), error=RuntimeError("модель не поднялась")
        )
        metrics = Metrics()

        with self.assertLogs("eva.server", level="ERROR"):
            handler = run_chat(
                router, metrics=metrics, clock=clock, payload={"stream": True}
            )

        status, headers, _ = split_http(handler.wfile.raw)
        self.assertEqual(status, 500)
        self.assertIn("application/json", headers["content-type"])

        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["requests"], 1)
        self.assertEqual(snapshot["finish"]["error"], 1)
        self.assertEqual(snapshot["routes"]["unknown"], 1)
        self.assertIsNone(snapshot["ttft_ms"]["p50"])

    def test_error_midstream_is_counted_as_error(self):
        clock = FakeClock()
        router = ScriptedStreamRouter(
            clock,
            pieces=("раз", "два"),
            midstream_error=RuntimeError("секрет sk-live-9"),
        )
        metrics = Metrics()

        with self.assertLogs("eva.server", level="ERROR"):
            handler = run_chat(
                router, metrics=metrics, clock=clock, payload={"stream": True}
            )

        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["finish"]["error"], 1)
        self.assertEqual(snapshot["ttft_ms"]["count"], 1)
        self.assertEqual(sse_events(handler.wfile.raw)[-1], "[DONE]")

    def test_stream_without_metadata_counts_stop_and_unknown_route(self):
        clock = FakeClock()
        router = ScriptedStreamRouter(clock, pieces=("раз", "два"))
        metrics = Metrics()

        run_chat(router, metrics=metrics, clock=clock, payload={"stream": True})

        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["finish"]["stop"], 1)
        self.assertEqual(snapshot["routes"]["unknown"], 1)
        self.assertEqual(snapshot["tokens"]["total"], 0)

    def test_fallback_single_chunk_is_counted_as_stop(self):
        clock = FakeClock()
        # Объект без stream(): сервер отдаёт готовый ответ одним чанком.
        router = StaticAnswerRouter(
            StubAnswer(route=StubRoute(kind="skill"), skill="pricing")
        )
        metrics = Metrics()

        handler = run_chat(
            router, metrics=metrics, clock=clock, payload={"stream": True}
        )

        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["finish"]["stop"], 1)
        self.assertEqual(snapshot["routes"]["skill"], 1)
        self.assertEqual(snapshot["tokens"], {"prompt": 3, "completion": 5, "total": 8})
        self.assertEqual(snapshot["ttft_ms"]["count"], 1)
        self.assertEqual(sse_events(handler.wfile.raw)[-1], "[DONE]")


class TestServerChatMetrics(unittest.TestCase):
    def test_non_stream_success_is_ok_with_latency_but_no_ttft(self):
        clock = FakeClock()
        router = ScriptedStreamRouter(clock, step=0.25)
        router.answer = StubAnswer(
            route=StubRoute(kind="skill"),
            skill="pricing",
            usage={"prompt_tokens": 3, "completion_tokens": 9},
        )
        metrics = Metrics()

        handler = run_chat(router, metrics=metrics, clock=clock, payload={"stream": False})

        status, _, _ = split_http(handler.wfile.raw)
        self.assertEqual(status, 200)

        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["requests"], 1)
        self.assertEqual(snapshot["streams"], 0)
        self.assertEqual(snapshot["finish"]["ok"], 1)
        self.assertEqual(snapshot["routes"]["skill"], 1)
        self.assertEqual(snapshot["tokens"], {"prompt": 3, "completion": 9, "total": 12})
        self.assertEqual(snapshot["ttft_ms"]["count"], 0)
        self.assertEqual(snapshot["latency_ms"]["p50"], 250.0)

    def test_non_stream_router_error_is_counted_as_error(self):
        clock = FakeClock()
        router = FailingRouter(RuntimeError("boom"))
        metrics = Metrics()

        with self.assertLogs("eva.server", level="ERROR"):
            handler = run_chat(router, metrics=metrics, clock=clock)

        status, _, _ = split_http(handler.wfile.raw)
        self.assertEqual(status, 500)

        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["finish"]["error"], 1)
        self.assertEqual(snapshot["routes"]["unknown"], 1)
        self.assertIsNone(snapshot["ttft_ms"]["p50"])

    def test_rejected_request_is_not_counted(self):
        clock = FakeClock()
        metrics = Metrics()
        handler = make_handler(
            FailingRouter(RuntimeError("не вызывается")),
            metrics=metrics,
            clock=clock,
            headers={"Content-Length": "2", "Content-Type": "application/json"},
        )
        handler.rfile = io.BytesIO(b"[]")

        handler.do_POST()

        status, _, _ = split_http(handler.wfile.raw)
        self.assertEqual(status, 400)
        self.assertEqual(metrics.snapshot()["requests"], 0)


# --------------------------------------------------------------------------- #
# GET /metrics
# --------------------------------------------------------------------------- #


class TestMetricsEndpoint(unittest.TestCase):
    def test_metrics_returns_json_with_counters_and_percentiles(self):
        metrics = Metrics()
        for value in (100, 200, 300, 400):
            metrics.observe(
                route="small",
                finish="ok",
                ttft_ms=value,
                latency_ms=value + 50,
            )

        handler = run_get("/metrics", metrics=metrics, clock=FakeClock(5.0))
        status, headers, body = split_http(handler.wfile.raw)

        self.assertEqual(status, 200)
        self.assertIn("application/json", headers["content-type"])
        self.assertEqual(headers["cache-control"], "no-store")

        payload = json.loads(body)
        self.assertEqual(payload["service"], "eva-server")
        self.assertEqual(payload["requests"], 4)
        self.assertEqual(payload["finish"]["ok"], 4)
        self.assertEqual(payload["routes"]["small"], 4)
        self.assertEqual(payload["ttft_ms"]["p50"], 250.0)
        self.assertEqual(payload["ttft_ms"]["p95"], 385.0)
        self.assertEqual(payload["ttft_ms"]["p99"], 397.0)
        self.assertEqual(payload["latency_ms"]["p50"], 300.0)
        self.assertIn("by_route", payload)
        self.assertEqual(payload["by_route"]["small"]["requests"], 4)
        self.assertEqual(payload["uptime_s"], 5.0)

    def test_metrics_on_fresh_server_is_zeroed(self):
        handler = run_get("/metrics", metrics=Metrics(), clock=FakeClock())
        status, _, body = split_http(handler.wfile.raw)

        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertEqual(payload["requests"], 0)
        self.assertEqual(
            payload["finish"], {"ok": 0, "stop": 0, "cancel": 0, "error": 0}
        )
        self.assertIsNone(payload["ttft_ms"]["p50"])
        json.dumps(payload)

    def test_metrics_requires_token_when_auth_is_on(self):
        guard = server.AccessGuard(token="s3cret", rate_limit_per_min=0)
        handler = run_get("/metrics", metrics=Metrics(), clock=FakeClock(), guard=guard)

        status, _, body = split_http(handler.wfile.raw)
        self.assertEqual(status, 401)
        self.assertEqual(json.loads(body)["error"]["code"], "unauthorized")

    def test_metrics_rejects_post(self):
        metrics = Metrics()
        handler = make_handler(
            FailingRouter(RuntimeError("не вызывается")),
            metrics=metrics,
            clock=FakeClock(),
            headers={"Content-Length": "2"},
            path="/metrics",
        )
        handler.rfile = io.BytesIO(b"{}")

        handler.do_POST()

        status, headers, _ = split_http(handler.wfile.raw)
        self.assertEqual(status, 405)
        self.assertEqual(headers["allow"], "GET")


# --------------------------------------------------------------------------- #
# Проводка сервера: часы и накопитель инжектируются, по умолчанию — свои
# --------------------------------------------------------------------------- #


class TestServerWiring(unittest.TestCase):
    def test_server_creates_default_metrics_and_keeps_clock(self):
        # Патчим инициализацию socketserver: сервер нужен без порта и сокета.
        with mock.patch.object(
            server.ThreadingHTTPServer, "__init__", lambda self, *a, **k: None
        ):
            default = server.EvaHTTPServer(("127.0.0.1", 0), object)
            metrics = Metrics()
            clock = FakeClock(42.0)
            injected = server.EvaHTTPServer(
                ("127.0.0.1", 0), object, metrics=metrics, clock=clock
            )

        self.assertIsInstance(default.metrics, Metrics)
        self.assertIs(default.clock, time.monotonic)
        self.assertIs(injected.metrics, metrics)
        self.assertIs(injected.clock, clock)
        self.assertEqual(injected.started_at, 42.0)

    def test_create_server_forwards_metrics_and_clock(self):
        metrics = Metrics()
        clock = FakeClock()
        captured = {}

        class RecordingServer:
            def __init__(self, address, handler, **kwargs):
                captured["address"] = address
                captured["kwargs"] = kwargs
                self.server_address = address

        with mock.patch.object(server, "EvaHTTPServer", RecordingServer):
            server.create_server(
                router=StaticAnswerRouter(StubAnswer()),
                metrics=metrics,
                clock=clock,
                auth_token="",
                rate_limit_per_min=0,
            )

        self.assertEqual(captured["address"], ("127.0.0.1", 0))
        self.assertIs(captured["kwargs"]["metrics"], metrics)
        self.assertIs(captured["kwargs"]["clock"], clock)
        self.assertEqual(captured["kwargs"]["router_name"], "injected")


# --------------------------------------------------------------------------- #
# Прогрев модели при старте и раздельный учёт prefill (задача 0115)
# --------------------------------------------------------------------------- #


class FakeLocalModel:
    """Локальный клиент для прогрева: без libvtm, без модели и без сокета.

    Повторяет только те методы, которые нужны прогреву: ``open()``,
    ``encode()``, ``generate()`` и ``close()``. Ошибки инжектируются, чтобы
    проверить, что неудачный прогрев не отменяет старт сервера.
    """

    def __init__(
        self,
        *,
        error: Exception | None = None,
        encode_error: Exception | None = None,
        tokens=(11, 12, 13),
    ) -> None:
        self.error = error
        self.encode_error = encode_error
        self.tokens = list(tokens)
        self.model_path = "/tmp/fake-model.vtm"
        self.opened = 0
        self.closed = 0
        self.encoded: list[str] = []
        self.calls: list[dict] = []

    def open(self):
        self.opened += 1
        if self.error is not None:
            raise self.error
        return self

    def encode(self, text: str) -> list[int]:
        self.encoded.append(text)
        if self.encode_error is not None:
            raise self.encode_error
        return list(self.tokens)

    def generate(self, ids, *, max_new_tokens=None):
        self.calls.append({"ids": list(ids), "max_new_tokens": max_new_tokens})
        return [1]

    def close(self) -> None:
        self.closed += 1


class TestWarmupWithoutSocket(unittest.TestCase):
    """Прогрев вызывается, метрики заполняются, ошибка не роняет старт."""

    def test_warmup_runs_and_records_metrics(self):
        backend = FakeLocalModel()
        metrics = Metrics()

        record = server.warmup_local_model(backend, metrics=metrics)

        self.assertTrue(record["ok"])
        self.assertIsNone(record["error"])
        self.assertEqual(backend.opened, 1)
        self.assertEqual(backend.encoded, [server.WARMUP_PROMPT])
        self.assertEqual(
            backend.calls,
            [{"ids": [11, 12, 13], "max_new_tokens": server.WARMUP_MAX_NEW_TOKENS}],
        )
        self.assertEqual(record["prefill_tokens"], 3)
        self.assertEqual(record["model"], "/tmp/fake-model.vtm")
        self.assertGreaterEqual(record["warmup_ms"], 0.0)

        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["warmup_ms"], record["warmup_ms"])
        self.assertEqual(snapshot["warmup"]["count"], 1)
        self.assertTrue(snapshot["warmup"]["ok"])
        self.assertEqual(snapshot["warmup"]["prefill_tokens"], 3)
        self.assertEqual(snapshot["warmup"]["model"], "/tmp/fake-model.vtm")
        self.assertIsNone(snapshot["warmup"]["error"])
        # Прогрев — не запрос: счётчики обслуживания он не трогает.
        self.assertEqual(snapshot["requests"], 0)
        self.assertEqual(snapshot["ttft_ms"]["count"], 0)
        json.dumps(snapshot)

    def test_injected_client_is_not_closed(self):
        backend = FakeLocalModel()

        server.warmup_local_model(backend, metrics=Metrics())

        self.assertEqual(backend.closed, 0, "переданный клиент — чужой")

    def test_warmup_failure_is_a_warning_not_a_crash(self):
        backend = FakeLocalModel(error=RuntimeError("libvtm не найдена"))
        metrics = Metrics()

        with self.assertLogs("eva.server", level="WARNING") as logs:
            record = server.warmup_local_model(backend, metrics=metrics)

        self.assertFalse(record["ok"])
        self.assertIn("libvtm не найдена", record["error"])
        self.assertIn("libvtm не найдена", "\n".join(logs.output))
        snapshot = metrics.snapshot()
        self.assertFalse(snapshot["warmup"]["ok"])
        self.assertIn("libvtm не найдена", snapshot["warmup"]["error"])
        self.assertEqual(snapshot["warmup"]["count"], 1)
        json.dumps(snapshot)

    def test_warmup_without_model_is_skipped(self):
        metrics = Metrics()

        self.assertIsNone(server.warmup_local_model(environ={}, metrics=metrics))
        self.assertIsNone(server.warmup_local_model(metrics=metrics))

        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["warmup"]["count"], 0)
        self.assertIsNone(snapshot["warmup_ms"])
        self.assertIsNone(snapshot["warmup"]["ok"])

    def test_env_model_is_loaded_and_closed(self):
        backend = FakeLocalModel()
        metrics = Metrics()

        with mock.patch.object(
            server, "_make_warmup_backend", return_value=backend
        ) as build:
            record = server.warmup_local_model(
                environ={"EVA_VTM_MODEL": "/models/1.5b.vtm"}, metrics=metrics
            )

        build.assert_called_once_with("/models/1.5b.vtm")
        self.assertTrue(record["ok"])
        self.assertEqual(record["model"], "/models/1.5b.vtm")
        self.assertEqual(backend.opened, 1)
        self.assertEqual(backend.closed, 1, "временный бэкенд закрывается")

    def test_warmup_uses_ids_when_tokenizer_is_missing(self):
        backend = FakeLocalModel(encode_error=RuntimeError("нет tokenizer.json"))

        record = server.warmup_local_model(backend, metrics=Metrics())

        self.assertTrue(record["ok"])
        self.assertEqual(backend.calls[0]["ids"], [1, 2, 3])
        self.assertEqual(record["prefill_tokens"], 3)

    def test_server_warms_injected_model_without_socket(self):
        backend = FakeLocalModel()
        metrics = Metrics()

        # Сокет не открываем: нужен только конструктор EvaHTTPServer.
        with mock.patch.object(
            server.ThreadingHTTPServer, "__init__", lambda self, *a, **k: None
        ):
            httpd = server.EvaHTTPServer(
                ("127.0.0.1", 0), object, local_model=backend, metrics=metrics
            )

        self.assertTrue(httpd.warmup["ok"])
        self.assertEqual(metrics.snapshot()["warmup"]["ok"], True)

    def test_server_start_survives_failed_warmup(self):
        metrics = Metrics()
        failing = FakeLocalModel(error=RuntimeError("модель битая"))

        with mock.patch.object(
            server.ThreadingHTTPServer, "__init__", lambda self, *a, **k: None
        ):
            with self.assertLogs("eva.server", level="WARNING"):
                httpd = server.EvaHTTPServer(
                    ("127.0.0.1", 0), object, local_model=failing, metrics=metrics
                )

        self.assertFalse(httpd.warmup["ok"])
        self.assertIn("модель битая", httpd.warmup["error"])
        self.assertFalse(metrics.snapshot()["warmup"]["ok"])

    def test_warmup_can_be_disabled(self):
        backend = FakeLocalModel()
        metrics = Metrics()

        with mock.patch.object(
            server.ThreadingHTTPServer, "__init__", lambda self, *a, **k: None
        ):
            httpd = server.EvaHTTPServer(
                ("127.0.0.1", 0),
                object,
                local_model=backend,
                metrics=metrics,
                warmup=False,
            )

        self.assertIsNone(httpd.warmup)
        self.assertEqual(backend.opened, 0)
        self.assertEqual(metrics.snapshot()["warmup"]["count"], 0)

    def test_create_server_forwards_warmup_settings(self):
        captured = {}

        class RecordingServer:
            def __init__(self, address, handler, **kwargs):
                captured["kwargs"] = kwargs
                self.server_address = address

        local = FakeLocalModel()
        with mock.patch.object(server, "EvaHTTPServer", RecordingServer):
            server.create_server(
                router=StaticAnswerRouter(StubAnswer()),
                auth_token="",
                rate_limit_per_min=0,
                local_model=local,
                warmup=False,
            )

        self.assertIs(captured["kwargs"]["local_model"], local)
        self.assertIs(captured["kwargs"]["warmup"], False)


class TestPrefillMetrics(unittest.TestCase):
    """prefill отдельно от TTFT: граница, остаток и счётчик токенов."""

    def test_ttft_is_split_into_prefill_and_remainder(self):
        metrics = Metrics()
        metrics.observe(
            route="small",
            finish="stop",
            ttft_ms=300,
            prefill_ms=250,
            latency_ms=400,
            prompt_tokens=400,
            prefill_tokens=512,
            stream=True,
        )

        snapshot = metrics.snapshot()

        self.assertEqual(snapshot["ttft_ms"]["p50"], 300.0)
        self.assertEqual(snapshot["prefill_ms"]["p50"], 250.0)
        self.assertEqual(snapshot["ttft_after_prefill_ms"]["p50"], 50.0)
        self.assertEqual(snapshot["prefill_tokens"], 512)
        self.assertEqual(snapshot["by_route"]["small"]["prefill_tokens"], 512)
        self.assertEqual(snapshot["by_route"]["small"]["prefill_ms"]["p50"], 250.0)
        self.assertEqual(
            snapshot["by_route"]["small"]["ttft_after_prefill_ms"]["p50"], 50.0
        )

    def test_non_stream_prefill_has_no_remainder(self):
        metrics = Metrics()
        metrics.observe(
            route="small", finish="ok", ttft_ms=None, prefill_ms=120, latency_ms=200
        )

        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["prefill_ms"]["count"], 1)
        self.assertEqual(snapshot["prefill_ms"]["p50"], 120.0)
        self.assertEqual(snapshot["ttft_after_prefill_ms"]["count"], 0)
        self.assertIsNone(snapshot["ttft_after_prefill_ms"]["p50"])

    def test_prefill_larger_than_ttft_is_not_invented(self):
        metrics = Metrics()
        metrics.observe(ttft_ms=100, prefill_ms=150, latency_ms=200)

        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["prefill_ms"]["p50"], 150.0)
        self.assertEqual(snapshot["ttft_after_prefill_ms"]["count"], 0)

    def test_prefill_tokens_default_to_prompt_tokens(self):
        metrics = Metrics()
        metrics.observe(route="big", finish="ok", prompt_tokens=7, latency_ms=10)

        self.assertEqual(metrics.snapshot()["prefill_tokens"], 7)

    def test_reset_clears_prefill_and_warmup(self):
        metrics = Metrics()
        metrics.observe(ttft_ms=300, prefill_ms=250, prefill_tokens=9)
        metrics.record_warmup(12.5, ok=True, prefill_tokens=3)
        metrics.reset()

        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["prefill_tokens"], 0)
        self.assertEqual(snapshot["prefill_ms"]["count"], 0)
        self.assertEqual(snapshot["ttft_after_prefill_ms"]["count"], 0)
        self.assertEqual(snapshot["warmup"]["count"], 0)
        self.assertIsNone(snapshot["warmup_ms"])


class TestServerPrefillReporting(unittest.TestCase):
    """Сервер: счётчик prefill_tokens и заголовок X-Prefill-Tokens."""

    def test_non_stream_response_reports_prefill(self):
        clock = FakeClock()
        router = ScriptedStreamRouter(clock, step=0.25)
        router.answer = StubAnswer(
            route=StubRoute(kind="small"),
            usage={"prompt_tokens": 3, "completion_tokens": 5},
            prefill_ms=180,
        )
        metrics = Metrics()

        handler = run_chat(
            router, metrics=metrics, clock=clock, payload={"stream": False}
        )

        status, headers, _ = split_http(handler.wfile.raw)
        self.assertEqual(status, 200)
        # «Привет» — 6 символов, оценка промпта ~4 символа на токен.
        self.assertEqual(headers["x-prefill-tokens"], "2")

        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["prefill_tokens"], 2)
        self.assertEqual(snapshot["prefill_ms"]["p50"], 180.0)
        self.assertEqual(snapshot["ttft_after_prefill_ms"]["count"], 0)

    def test_stream_ttft_split_uses_reported_prefill(self):
        clock = FakeClock()
        meta = StubAnswer(
            text="",
            route=StubRoute(kind="big"),
            usage={"prompt_tokens": 7, "completion_tokens": 5},
            prefill_ms=100,
        )
        router = ScriptedStreamRouter(clock, pieces=("При", "вет"), meta=meta)
        metrics = Metrics()

        handler = run_chat(
            router, metrics=metrics, clock=clock, payload={"stream": True}
        )

        status, headers, _ = split_http(handler.wfile.raw)
        self.assertEqual(status, 200)
        self.assertEqual(headers["x-prefill-tokens"], "2")

        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["ttft_ms"]["p50"], 125.0)
        self.assertEqual(snapshot["prefill_ms"]["p50"], 100.0)
        self.assertEqual(snapshot["ttft_after_prefill_ms"]["p50"], 25.0)
        self.assertEqual(snapshot["prefill_tokens"], 2)

    def test_stream_without_prefill_data_leaves_split_empty(self):
        clock = FakeClock()
        router = ScriptedStreamRouter(clock, pieces=("раз", "два"))
        metrics = Metrics()

        run_chat(router, metrics=metrics, clock=clock, payload={"stream": True})

        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["ttft_ms"]["count"], 1)
        self.assertEqual(snapshot["prefill_tokens"], 2)
        self.assertEqual(snapshot["prefill_ms"]["count"], 0)
        self.assertEqual(snapshot["ttft_after_prefill_ms"]["count"], 0)

    def test_aborted_stream_still_counts_prompt_prefill_tokens(self):
        clock = FakeClock()
        router = ScriptedStreamRouter(clock, pieces=("раз", "два", "три"))
        metrics = Metrics()
        writer = FakeWriter(fail_after=1)  # клиент ушёл на первом чанке

        run_chat(
            router,
            metrics=metrics,
            clock=clock,
            writer=writer,
            payload={"stream": True},
        )

        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["finish"]["cancel"], 1)
        # usage при обрыве неизвестен, но длина промпта — известна.
        self.assertEqual(snapshot["prefill_tokens"], 2)

    def test_metrics_endpoint_exposes_prefill_and_warmup(self):
        metrics = Metrics()
        metrics.record_warmup(12.5, ok=True, prefill_tokens=3, model="demo/model.vtm")
        metrics.observe(route="small", finish="stop", ttft_ms=300, prefill_ms=250)

        handler = run_get("/metrics", metrics=metrics, clock=FakeClock())
        status, _, body = split_http(handler.wfile.raw)
        payload = json.loads(body)

        self.assertEqual(status, 200)
        self.assertEqual(payload["warmup_ms"], 12.5)
        self.assertTrue(payload["warmup"]["ok"])
        self.assertEqual(payload["warmup"]["model"], "demo/model.vtm")
        self.assertEqual(payload["prefill_ms"]["p50"], 250.0)
        self.assertEqual(payload["ttft_after_prefill_ms"]["p50"], 50.0)
        json.dumps(payload)


# --------------------------------------------------------------------------- #
# Своя модель в метриках (задача 0109) и реальный роутер (0109 + 0120)
# --------------------------------------------------------------------------- #


class TestLocalModelMetrics(unittest.TestCase):
    """Сервер прокидывает маркер ``local`` в счётчики ``/metrics``."""

    def test_non_stream_local_answer_is_counted(self):
        clock = FakeClock()
        router = StaticAnswerRouter(
            StubAnswer(route=StubRoute(kind="small", local=True))
        )
        metrics = Metrics()

        run_chat(router, metrics=metrics, clock=clock, payload={"stream": False})

        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["small_local"], 1)
        self.assertEqual(snapshot["small_http"], 0)
        self.assertEqual(snapshot["local_requests"], 1)
        self.assertEqual(snapshot["local_share"], 1.0)
        self.assertEqual(snapshot["local_share_all"], 1.0)
        self.assertEqual(snapshot["source"], {"skill": 0, "local": 1, "api": 0})

    def test_stream_http_answer_is_counted(self):
        clock = FakeClock()
        meta = StubAnswer(
            text="",
            route=StubRoute(kind="small", local=False),
            usage={"prompt_tokens": 4, "completion_tokens": 2},
        )
        router = ScriptedStreamRouter(clock, pieces=("http", "-ответ"), meta=meta)
        metrics = Metrics()

        run_chat(router, metrics=metrics, clock=clock, payload={"stream": True})

        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["small_http"], 1)
        self.assertEqual(snapshot["small_local"], 0)
        self.assertEqual(snapshot["local_share"], 0.0)
        self.assertEqual(snapshot["source"], {"skill": 0, "local": 0, "api": 1})

    def test_answer_without_marker_is_not_counted_as_local(self):
        clock = FakeClock()
        router = StaticAnswerRouter(StubAnswer(route=StubRoute(kind="small")))
        metrics = Metrics()

        run_chat(router, metrics=metrics, clock=clock, payload={"stream": False})

        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["small_local"], 0)
        self.assertEqual(snapshot["small_http"], 0)
        self.assertIsNone(snapshot["local_share"])
        self.assertEqual(snapshot["routes"]["small"], 1)

    def test_metrics_endpoint_exposes_local_share(self):
        metrics = Metrics()
        metrics.observe(route="small", finish="ok", local=True)
        metrics.observe(route="small", finish="ok", local=False)

        handler = run_get("/metrics", metrics=metrics, clock=FakeClock())
        status, _, body = split_http(handler.wfile.raw)
        payload = json.loads(body)

        self.assertEqual(status, 200)
        self.assertEqual(payload["small_local"], 1)
        self.assertEqual(payload["small_http"], 1)
        self.assertEqual(payload["local_share"], 0.5)
        self.assertEqual(payload["source"]["local"], 1)
        self.assertEqual(payload["source"]["api"], 1)


class _LocalModelWithTimings:
    """Локальная ВТМ для реального ``eva.router``: отдаёт prefill и local-маркер.

    ``prefill_ms`` попадает в результат/кусок как ``timings.prefill_ms``;
    ``clock`` двигает время между кусками, чтобы TTFT был ненулевым.
    """

    model_path = "/tmp/fake-model.vtm"

    def __init__(self, *, prefill_ms=90, clock=None) -> None:
        self.prefill_ms = prefill_ms
        self.clock = clock
        self.calls = 0

    def generate(self, prompt_ids, **_kwargs):
        return list(prompt_ids)

    def chat(self, messages, max_tokens: int = 512):
        self.calls += 1
        return {
            "text": "локальный ответ",
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "cost_rub": 0.0},
            "timings": {"prefill_ms": self.prefill_ms},
        }

    def stream_chat(self, messages, max_tokens: int = 512):
        self.calls += 1
        if self.clock is not None:
            self.clock.advance(0.125)
        yield {"text": "лок", "timings": {"prefill_ms": self.prefill_ms}}
        yield "альный"


class TestRealRouterLocalAndPrefill(unittest.TestCase):
    """Сквозь ``eva.router`` → сервер: local и prefill доходят до ``/metrics``."""

    def _router(self, model):
        """Объект-маршрутизатор с ``answer`` и ``stream`` — боевой ``eva.router``."""
        from eva import router as eva_router
        from eva.config import Config
        from eva.types import Route

        cfg = Config()
        deps = {
            "classifier": lambda text, _cfg: Route(
                "small", "тест: малая модель", 0.8
            ),
            "small": model,
        }

        class Router:
            def __call__(self, request):
                return eva_router.answer(request, cfg, deps=deps)

            def stream(self, request):
                return eva_router.stream(request, cfg, deps=deps)

        return Router()

    def test_non_stream_answer_reports_local_and_prefill(self):
        clock = FakeClock()
        model = _LocalModelWithTimings(prefill_ms=120)
        metrics = Metrics()

        run_chat(
            self._router(model), metrics=metrics, clock=clock, payload={"stream": False}
        )

        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["small_local"], 1)
        self.assertEqual(snapshot["local_share"], 1.0)
        self.assertEqual(snapshot["prefill_ms"]["p50"], 120.0)

    def test_stream_answer_splits_ttft_by_reported_prefill(self):
        clock = FakeClock()
        model = _LocalModelWithTimings(prefill_ms=90, clock=clock)
        metrics = Metrics()

        run_chat(
            self._router(model), metrics=metrics, clock=clock, payload={"stream": True}
        )

        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["small_local"], 1)
        self.assertEqual(snapshot["prefill_ms"]["p50"], 90.0)
        self.assertEqual(snapshot["ttft_ms"]["p50"], 125.0)
        self.assertEqual(snapshot["ttft_after_prefill_ms"]["p50"], 35.0)

    def test_stream_without_prefill_data_leaves_split_empty(self):
        clock = FakeClock()
        model = _LocalModelWithTimings(prefill_ms=None, clock=clock)
        metrics = Metrics()

        run_chat(
            self._router(model), metrics=metrics, clock=clock, payload={"stream": True}
        )

        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["small_local"], 1)
        self.assertEqual(snapshot["prefill_ms"]["count"], 0)
        self.assertEqual(snapshot["ttft_after_prefill_ms"]["count"], 0)


if __name__ == "__main__":
    unittest.main()
