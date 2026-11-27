"""Лимиты HTTP-сервера и мягкая деградация (задача 0108).

Запуск::

    python3 -m unittest tests.test_server_limits -v

Тесты идут в процессе: ни порта, ни сокета. Обработчик ``eva.server``
собирается напрямую (как в ``tests/test_eva_server_metrics.py``), тело
запроса подаётся через ``io.BytesIO``, ответ читается из поддельного
``wfile``. Клиенты моделей и скилл — фейковые: сети и реальных моделей нет.

Проверяется поведение снаружи: коды 413/429/503/504, заголовок
``Retry-After``, порядок деградации big → локальная модель → скилл и
счётчики в ``GET /metrics``.
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eva import server  # noqa: E402
from eva.telemetry import Metrics  # noqa: E402

CHAT = {"messages": [{"role": "user", "content": "Привет"}]}


# --------------------------------------------------------------------------- #
# Поддельные HTTP-объекты и клиенты
# --------------------------------------------------------------------------- #


class FakeWriter:
    """Поддельный ``wfile``: копит байты ответа и умеет ``flush``."""

    def __init__(self) -> None:
        self.writes: list[bytes] = []
        self.flushes = 0

    def write(self, data: bytes) -> int:
        payload = bytes(data)
        self.writes.append(payload)
        return len(payload)

    def flush(self) -> None:
        self.flushes += 1

    @property
    def raw(self) -> bytes:
        return b"".join(self.writes)


class FakeClient:
    """Фейковый клиент модели: пишет порядок вызовов, умеет падать/тормозить."""

    def __init__(
        self,
        name: str,
        *,
        text: str = "ответ",
        error: Exception | None = None,
        delay: float = 0.0,
    ) -> None:
        self.name = name
        self.text = text
        self.error = error
        self.delay = delay
        self.calls: list[tuple[list, int | None]] = []

    def chat(self, messages, max_tokens=None):
        self.calls.append((list(messages), max_tokens))
        if self.delay:
            time.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return {
            "text": self.text,
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "cost_rub": 0.0},
        }


class FakeSkill:
    """Фейковый скилл: запоминает вызовы и отдаёт заданный результат."""

    def __init__(self, result=None) -> None:
        self.result = result
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, text, ctx):
        self.calls.append((text, ctx))
        return self.result


class FailingRouter:
    """Основной маршрутизатор, который всегда падает (big-путь недоступен)."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error or TimeoutError("big не ответил")
        self.calls = 0

    def __call__(self, request):
        self.calls += 1
        raise self.error


class RecordingRouter:
    """Простой маршрутизатор для проверок, что до него не дошли."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, request):
        self.calls += 1
        return {
            "text": "ok",
            "route": {"kind": "small", "reason": "тест", "confidence": 1.0},
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }


def make_handler(
    router,
    *,
    method: str = "POST",
    path: str = "/v1/chat/completions",
    payload: dict | None = None,
    raw: bytes | None = None,
    guard: server.AccessGuard | None = None,
    max_prompt_tokens: int = 0,
    request_timeout: float = 0.0,
    degradation: server.DegradationChain | None = None,
    httpd: SimpleNamespace | None = None,
):
    """Собрать обработчик так, как его собрал бы ``setup()`` BaseHTTPRequestHandler."""
    body = raw
    if body is None:
        body = json.dumps(payload or CHAT, ensure_ascii=False).encode("utf-8")
    handler_class = server.make_handler(router)
    handler = handler_class.__new__(handler_class)
    handler.command = method
    handler.path = path
    handler.requestline = "{} {} HTTP/1.1".format(method, path)
    handler.request_version = "HTTP/1.1"
    handler.headers = {
        "Content-Length": str(len(body)),
        "Content-Type": "application/json",
    }
    handler.rfile = io.BytesIO(body)
    handler.wfile = FakeWriter()
    handler.client_address = ("127.0.0.1", 43210)
    handler.close_connection = False
    handler._sse_chunked = True
    handler.server = httpd or SimpleNamespace(
        router_name="injected",
        access_guard=guard,
        metrics=Metrics(),
        clock=time.monotonic,
        started_at=time.monotonic(),
        counters=server.ServerCounters(),
        max_prompt_tokens=max_prompt_tokens,
        request_timeout=request_timeout,
        degradation=degradation,
    )
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
    """Поля ``data:`` событий SSE в порядке появления (chunked-кадры не мешают)."""
    text = raw.decode("utf-8", "replace").replace("\r\n", "\n")
    return [
        line[len("data:") :].strip()
        for line in text.split("\n")
        if line.startswith("data:")
    ]


def json_sse_events(raw: bytes) -> list[dict]:
    return [json.loads(event) for event in sse_events(raw) if event != "[DONE]"]


# --------------------------------------------------------------------------- #
# 1. Лимит длины промпта: 413
# --------------------------------------------------------------------------- #


class TestPromptLimit(unittest.TestCase):
    def test_long_prompt_is_413_with_human_text(self):
        router = RecordingRouter()
        long_text = "а" * 400  # ~100 токенов по оценке 4 символа/токен
        handler = make_handler(
            router,
            payload={"messages": [{"role": "user", "content": long_text}]},
            max_prompt_tokens=50,
        )

        handler.do_POST()

        status, headers, body = split_http(handler.wfile.raw)
        error = json.loads(body)["error"]
        self.assertEqual(status, 413)
        self.assertEqual(error["code"], "prompt_too_large")
        self.assertIn("50", error["message"])
        self.assertIn("EVA_MAX_PROMPT_TOKENS", error["message"])
        self.assertNotIn("Traceback", body.decode("utf-8"))
        self.assertEqual(headers["x-max-prompt-tokens"], "50")
        self.assertEqual(router.calls, 0)
        self.assertEqual(
            handler.server.counters.snapshot()["prompt_too_large"], 1
        )

    def test_prompt_at_limit_passes(self):
        router = RecordingRouter()
        # 40 символов = 10 токенов; ровно на границе 413 быть не должно.
        handler = make_handler(
            router,
            payload={"messages": [{"role": "user", "content": "x" * 40}]},
            max_prompt_tokens=10,
        )

        handler.do_POST()

        self.assertEqual(split_http(handler.wfile.raw)[0], 200)
        self.assertEqual(router.calls, 1)

    def test_streaming_long_prompt_is_413_not_sse(self):
        router = RecordingRouter()
        handler = make_handler(
            router,
            payload={
                "messages": [{"role": "user", "content": "x" * 400}],
                "stream": True,
            },
            max_prompt_tokens=50,
        )

        handler.do_POST()

        status, headers, body = split_http(handler.wfile.raw)
        self.assertEqual(status, 413)
        self.assertIn("application/json", headers["content-type"])
        self.assertEqual(json.loads(body)["error"]["code"], "prompt_too_large")
        self.assertEqual(router.calls, 0)

    def test_limit_zero_disables_check(self):
        router = RecordingRouter()
        handler = make_handler(
            router,
            payload={"messages": [{"role": "user", "content": "x" * 4000}]},
            max_prompt_tokens=0,
        )

        handler.do_POST()

        self.assertEqual(split_http(handler.wfile.raw)[0], 200)
        self.assertEqual(router.calls, 1)

    def test_prompt_estimate_is_shared_with_policy(self):
        pairs = [("user", "abcd"), ("assistant", ""), ("user", "абвгд")]
        self.assertEqual(server.estimate_prompt_tokens(pairs), 3)


# --------------------------------------------------------------------------- #
# 2. 429 с Retry-After по маршруту и счётчик
# --------------------------------------------------------------------------- #


class TestRateLimitPerRoute(unittest.TestCase):
    def setUp(self) -> None:
        self.guard = server.AccessGuard(token="", rate_limit_per_min=1)
        self.router = RecordingRouter()

    def post(self, guard=None, path="/v1/chat/completions", method="POST"):
        handler = make_handler(
            self.router,
            guard=guard or self.guard,
            path=path,
            method=method,
        )
        return handler, handler.do_POST if method == "POST" else handler.do_GET

    def test_second_request_is_429_with_retry_after_and_counter(self):
        first = make_handler(self.router, guard=self.guard)
        first.do_POST()
        second = make_handler(self.router, guard=self.guard)
        second.do_POST()

        status, headers, body = split_http(second.wfile.raw)
        error = json.loads(body)["error"]
        self.assertEqual(first.wfile.raw[:15], b"HTTP/1.1 200 OK")
        self.assertEqual(status, 429)
        self.assertEqual(error["code"], "rate_limited")
        self.assertGreaterEqual(int(headers["retry-after"]), 1)
        self.assertEqual(headers["x-ratelimit-limit"], "1")
        self.assertEqual(headers["x-ratelimit-remaining"], "0")
        self.assertEqual(headers["x-ratelimit-route"], "/v1/chat/completions")
        self.assertIn("/v1/chat/completions", error["message"])
        self.assertEqual(
            second.server.counters.snapshot()["rate_limited"], 1
        )
        # 429 не дошёл до маршрутизатора: бюджет модели не тратится.
        self.assertEqual(self.router.calls, 1)

    def test_limit_is_per_route_not_global(self):
        chat_one = make_handler(self.router, guard=self.guard)
        chat_one.do_POST()
        chat_two = make_handler(self.router, guard=self.guard)
        chat_two.do_POST()
        self.assertEqual(split_http(chat_two.wfile.raw)[0], 429)

        # Тот же ключ и тот же guard, но другой маршрут — своя квота.
        metrics_handler = make_handler(
            self.router,
            guard=self.guard,
            method="GET",
            path="/metrics",
        )
        metrics_handler.do_GET()

        status, _, body = split_http(metrics_handler.wfile.raw)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["limits"]["rate_limited"], 0)

    def test_disabled_limit_never_returns_429(self):
        guard = server.AccessGuard(token="", rate_limit_per_min=0)
        for _ in range(3):
            handler = make_handler(self.router, guard=guard)
            handler.do_POST()
            self.assertEqual(split_http(handler.wfile.raw)[0], 200)
        self.assertEqual(self.router.calls, 3)


# --------------------------------------------------------------------------- #
# 3. Порядок деградации big → локальная → скилл → 503
# --------------------------------------------------------------------------- #


class TestDegradationOrder(unittest.TestCase):
    def make_request(self):
        from eva.types import Message, Request

        return Request(messages=[Message(role="user", content="вопрос")])

    def test_big_wins_and_other_sources_are_not_called(self):
        big = FakeClient("big", text="от big")
        local = FakeClient("local", text="от local")
        skill = FakeSkill({"text": "от skill", "skill": "pricing"})
        chain = server.DegradationChain(big=big, local=local, skills=skill)

        answer = chain.answer(self.make_request())

        self.assertEqual(answer.text, "от big")
        self.assertEqual(answer.route.kind, "big")
        self.assertEqual(len(big.calls), 1)
        self.assertEqual(local.calls, [])
        self.assertEqual(skill.calls, [])
        self.assertEqual(chain.stats.snapshot()["degraded"], 0)

    def test_big_failure_falls_back_to_local(self):
        big = FakeClient("big", error=TimeoutError("big таймаут"))
        local = FakeClient("local", text="локальный ответ")
        skill = FakeSkill({"text": "от skill", "skill": "pricing"})
        chain = server.DegradationChain(big=big, local=local, skills=skill)

        answer = chain.answer(self.make_request())

        self.assertEqual(answer.text, "локальный ответ")
        self.assertEqual(answer.route.kind, "small")
        self.assertIn("локальная", answer.route.reason)
        self.assertIn("big", answer.route.reason)
        self.assertEqual(len(big.calls), 1)
        self.assertEqual(len(local.calls), 1)
        self.assertEqual(skill.calls, [])
        stats = chain.stats.snapshot()
        self.assertEqual(stats["degraded"], 1)
        self.assertEqual(stats["served"]["local"], 1)
        self.assertEqual(stats["failures"]["big"], 1)

    def test_big_and_local_failure_falls_back_to_skill(self):
        big = FakeClient("big", error=RuntimeError("big упал"))
        local = FakeClient("local", error=ConnectionError("local упал"))
        skill = FakeSkill({"text": "ответ скилла", "skill": "pricing"})
        chain = server.DegradationChain(big=big, local=local, skills=skill)

        answer = chain.answer(self.make_request())

        self.assertEqual(answer.text, "ответ скилла")
        self.assertEqual(answer.route.kind, "skill")
        self.assertEqual(answer.skill, "pricing")
        self.assertIn("скилл", answer.route.reason)
        self.assertEqual(len(big.calls), 1)
        self.assertEqual(len(local.calls), 1)
        self.assertEqual(len(skill.calls), 1)
        self.assertEqual(chain.stats.snapshot()["served"]["skill"], 1)

    def test_all_sources_fail_raises_503_exception_in_order(self):
        big = FakeClient("big", error=TimeoutError("big таймаут"))
        local = FakeClient("local", error=TimeoutError("local таймаут"))
        skill = FakeSkill(None)
        chain = server.DegradationChain(big=big, local=local, skills=skill)

        with self.assertRaises(server.ServiceUnavailable) as caught:
            chain.answer(self.make_request())

        error = caught.exception
        self.assertEqual(error.stages, ["big", "local", "skill"])
        self.assertIn("big", str(error))
        self.assertIn("local", str(error))
        self.assertIn("недоступны", error.public_message)
        self.assertEqual(chain.stats.snapshot()["unavailable"], 1)
        self.assertEqual(len(big.calls), 1)
        self.assertEqual(len(local.calls), 1)
        self.assertEqual(len(skill.calls), 1)

    def test_skip_big_after_primary_router_failure(self):
        big = FakeClient("big", text="не должен вызываться")
        local = FakeClient("local", text="локальный ответ")
        chain = server.DegradationChain(big=big, local=local)

        answer = chain.answer(self.make_request(), skip=("big",))

        self.assertEqual(answer.route.kind, "small")
        self.assertEqual(big.calls, [])
        self.assertEqual(len(local.calls), 1)


class TestDegradationHandler(unittest.TestCase):
    def test_all_sources_fail_is_503_with_clear_text(self):
        big = FakeClient("big", error=TimeoutError("big таймаут"))
        local = FakeClient("local", error=TimeoutError("local таймаут"))
        chain = server.DegradationChain(big=big, local=local, skills=FakeSkill(None))
        handler = make_handler(chain, degradation=chain)

        handler.do_POST()

        status, headers, body = split_http(handler.wfile.raw)
        error = json.loads(body)["error"]
        self.assertEqual(status, 503)
        self.assertEqual(error["code"], "service_unavailable")
        self.assertEqual(error["type"], "server_error")
        self.assertIn("недоступны", error["message"])
        self.assertIn("Код запроса", error["message"])
        self.assertIn("retry-after", headers)
        self.assertNotIn(b"Traceback", body)
        self.assertEqual(handler.server.counters.snapshot()["unavailable"], 1)

    def test_primary_router_failure_uses_local_without_retrying_big(self):
        router = FailingRouter(TimeoutError("big-клиент недоступен"))
        big = FakeClient("big", text="не должен вызываться")
        local = FakeClient("local", text="локальный ответ")
        chain = server.DegradationChain(big=big, local=local)
        handler = make_handler(router, degradation=chain)

        handler.do_POST()

        status, _, body = split_http(handler.wfile.raw)
        payload = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(payload["choices"][0]["message"]["content"], "локальный ответ")
        self.assertEqual(payload["eva"]["route"], "small")
        self.assertEqual(router.calls, 1)
        self.assertEqual(big.calls, [])  # big уже пробовали — не повторяем
        self.assertEqual(len(local.calls), 1)
        self.assertEqual(handler.server.counters.snapshot()["degraded"], 1)

    def test_degradation_is_visible_in_metrics(self):
        big = FakeClient("big", error=TimeoutError("big таймаут"))
        local = FakeClient("local", text="локальный ответ")
        chain = server.DegradationChain(big=big, local=local)
        handler = make_handler(chain, degradation=chain)
        handler.do_POST()

        metrics_handler = make_handler(
            chain,
            method="GET",
            path="/metrics",
            httpd=handler.server,
        )
        metrics_handler.do_GET()

        status, _, body = split_http(metrics_handler.wfile.raw)
        payload = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(payload["limits"]["degraded"], 1)
        self.assertEqual(payload["limits"]["degraded_by_stage"], {"small": 1})
        self.assertTrue(payload["degraded"])
        self.assertEqual(payload["degradation"]["degraded"], 1)
        self.assertEqual(payload["degradation"]["attempts"]["big"], 1)
        self.assertEqual(payload["degradation"]["attempts"]["local"], 1)
        # Непотоковый запрос учтён обычными метриками маршрута small.
        self.assertEqual(payload["routes"]["small"], 1)

    def test_normal_skill_answer_is_not_counted_as_degraded(self):
        class SkillRouter:
            def __call__(self, request):
                return {
                    "text": "ответ скилла",
                    "route": {"kind": "skill", "reason": "скилл сработал", "confidence": 1.0},
                    "skill": "pricing",
                }

        chain = server.DegradationChain(
            big=FakeClient("big"), local=FakeClient("local")
        )
        handler = make_handler(SkillRouter(), degradation=chain)

        handler.do_POST()

        self.assertEqual(split_http(handler.wfile.raw)[0], 200)
        self.assertEqual(handler.server.counters.snapshot()["degraded"], 0)

    def test_streaming_chain_is_degraded_to_one_chunk(self):
        big = FakeClient("big", error=TimeoutError("big таймаут"))
        local = FakeClient("local", text="локальный ответ")
        chain = server.DegradationChain(big=big, local=local)
        handler = make_handler(
            chain,
            payload={**CHAT, "stream": True},
            degradation=chain,
        )

        handler.do_POST()

        status, headers, _ = split_http(handler.wfile.raw)
        events = json_sse_events(handler.wfile.raw)
        self.assertEqual(status, 200)
        self.assertIn("text/event-stream", headers["content-type"])
        self.assertEqual(events[0]["choices"][0]["delta"]["content"], "локальный ответ")
        self.assertEqual(events[0]["eva"]["route"], "small")
        self.assertEqual(handler.server.counters.snapshot()["degraded"], 1)


# --------------------------------------------------------------------------- #
# 4. Таймаут генерации: 504 и отмена
# --------------------------------------------------------------------------- #


class TestRequestTimeout(unittest.TestCase):
    def test_slow_non_stream_router_returns_504(self):
        def slow_router(request):
            time.sleep(0.4)
            return {"text": "поздно", "route": {"kind": "small", "reason": "тест"}}

        handler = make_handler(slow_router, request_timeout=0.05)
        started = time.monotonic()

        handler.do_POST()

        elapsed = time.monotonic() - started
        status, headers, body = split_http(handler.wfile.raw)
        error = json.loads(body)["error"]
        self.assertEqual(status, 504)
        # Таймаут запроса 0,05 с; в CI ответ приходит с задержкой планировщика,
        # поэтому проверяем «быстро», а не «мгновенно».
        limit = 1.5 if os.environ.get("CI") else 0.35
        self.assertLess(elapsed, limit)
        self.assertEqual(error["code"], "request_timeout")
        self.assertEqual(error["type"], "server_error")
        self.assertIn("0.05", error["message"])
        self.assertIn("EVA_REQUEST_TIMEOUT", error["message"])
        self.assertIn("retry-after", headers)
        self.assertNotIn(b"Traceback", body)
        self.assertEqual(handler.server.counters.snapshot()["timeout"], 1)

    def test_timeout_zero_disables_deadline(self):
        def router(request):
            time.sleep(0.02)
            return {
                "text": "успел",
                "route": {"kind": "small", "reason": "тест"},
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }

        handler = make_handler(router, request_timeout=0)
        handler.do_POST()

        self.assertEqual(split_http(handler.wfile.raw)[0], 200)

    def test_slow_stream_before_first_chunk_is_504_json(self):
        class SlowFirstChunk:
            def __init__(self) -> None:
                self.closed = False

            def __call__(self, request):
                return "не поток"

            def stream(self, request):
                try:
                    time.sleep(0.4)
                    yield "поздно"
                finally:
                    self.closed = True

        router = SlowFirstChunk()
        handler = make_handler(
            router,
            payload={**CHAT, "stream": True},
            request_timeout=0.05,
        )

        handler.do_POST()

        status, headers, body = split_http(handler.wfile.raw)
        self.assertEqual(status, 504)
        self.assertIn("application/json", headers["content-type"])
        self.assertEqual(json.loads(body)["error"]["code"], "request_timeout")
        self.assertEqual(handler.server.counters.snapshot()["timeout"], 1)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and not router.closed:
            time.sleep(0.01)
        self.assertTrue(router.closed, "генератор не был закрыт по таймауту")

    def test_blocking_stream_factory_is_bounded_by_deadline(self):
        class BlockingFactory:
            def __call__(self, request):
                return "не поток"

            def stream(self, request):
                time.sleep(0.4)
                return iter(["поздно"])

        handler = make_handler(
            BlockingFactory(),
            payload={**CHAT, "stream": True},
            request_timeout=0.05,
        )

        handler.do_POST()

        status, _, body = split_http(handler.wfile.raw)
        self.assertEqual(status, 504)
        self.assertEqual(json.loads(body)["error"]["code"], "request_timeout")
        self.assertEqual(handler.server.counters.snapshot()["timeout"], 1)

    def test_stream_timeout_after_first_chunk_sends_sse_error_and_done(self):
        class SlowSecondChunk:
            def __call__(self, request):
                return "не поток"

            def stream(self, request):
                yield "первый"
                time.sleep(0.4)
                yield "второй"

        handler = make_handler(
            SlowSecondChunk(),
            payload={**CHAT, "stream": True},
            request_timeout=0.1,
        )

        handler.do_POST()

        status, headers, raw = split_http(handler.wfile.raw)
        events = sse_events(handler.wfile.raw)
        self.assertEqual(status, 200)
        self.assertIn("text/event-stream", headers["content-type"])
        self.assertEqual(events[-1], "[DONE]")
        self.assertEqual(events.count("[DONE]"), 1)
        errors = [json.loads(event) for event in events if '"error"' in event]
        self.assertEqual(errors[0]["error"]["code"], "request_timeout")
        self.assertEqual(handler.server.counters.snapshot()["timeout"], 1)
        self.assertTrue(raw)

    def test_timeout_cancels_stream_generator(self):
        class CancellableStream:
            def __init__(self) -> None:
                self.closed = False

            def __call__(self, request):
                return "не поток"

            def stream(self, request):
                try:
                    yield "первый"
                    time.sleep(1.0)
                    yield "второй"
                finally:
                    self.closed = True

        router = CancellableStream()
        handler = make_handler(
            router,
            payload={**CHAT, "stream": True},
            request_timeout=0.05,
        )

        handler.do_POST()

        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and not router.closed:
            time.sleep(0.01)
        self.assertTrue(router.closed, "генератор не получил отмену")


# --------------------------------------------------------------------------- #
# 5. Настройки и проводка сервера (без порта)
# --------------------------------------------------------------------------- #


class TestLimitSettings(unittest.TestCase):
    def test_defaults(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                server._limit_settings(),
                (
                    server.DEFAULT_MAX_PROMPT_TOKENS,
                    server.DEFAULT_REQUEST_TIMEOUT_S,
                ),
            )

    def test_values_from_env(self):
        env = {"EVA_MAX_PROMPT_TOKENS": "1234", "EVA_REQUEST_TIMEOUT": "2.5"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(server._limit_settings(), (1234, 2.5))

    def test_invalid_env_fails_before_listen(self):
        with mock.patch.dict(
            os.environ, {"EVA_REQUEST_TIMEOUT": "быстро"}, clear=True
        ):
            with self.assertRaises(ValueError) as caught:
                server.create_server(host="127.0.0.1", port=0)

        self.assertIn("EVA_REQUEST_TIMEOUT", str(caught.exception))

    def test_chain_router_is_recognized_as_degradation(self):
        captured = {}

        class RecordingServer:
            def __init__(self, address, handler, **kwargs):
                captured["kwargs"] = kwargs
                self.server_address = address

        chain = server.DegradationChain(
            big=FakeClient("big"), local=FakeClient("local")
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(server, "EvaHTTPServer", RecordingServer):
                server.create_server(router=chain, auth_token="", rate_limit_per_min=0)

        self.assertIs(captured["kwargs"]["degradation"], chain)
        self.assertEqual(captured["kwargs"]["router_name"], "degradation")

    def test_create_server_forwards_limits_and_degradation(self):
        captured = {}

        class RecordingServer:
            def __init__(self, address, handler, **kwargs):
                captured["address"] = address
                captured["handler"] = handler
                captured["kwargs"] = kwargs
                self.server_address = address

        chain = server.DegradationChain(
            big=FakeClient("big"), local=FakeClient("local")
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch.object(server, "EvaHTTPServer", RecordingServer):
                server.create_server(
                    router=chain,
                    max_prompt_tokens=777,
                    request_timeout=3.5,
                    degradation=chain,
                    auth_token="",
                    rate_limit_per_min=0,
                )

        self.assertEqual(captured["kwargs"]["max_prompt_tokens"], 777)
        self.assertEqual(captured["kwargs"]["request_timeout"], 3.5)
        self.assertIs(captured["kwargs"]["degradation"], chain)
        self.assertEqual(captured["kwargs"]["router_name"], "degradation")


if __name__ == "__main__":
    unittest.main()
