"""Тесты потокового режима гибридного роутера и SSE-клиента.

Запуск: ``python3 -m unittest tests.test_router_stream -v``

Сети нет: клиенты моделей и потоковый транспорт — заглушки; сервер в
интеграционных проверках поднимается только на loopback.
"""

from __future__ import annotations

import http.client
import inspect
import io
import json
import os
import socket
import sys
import threading
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eva import policy, router, server  # noqa: E402
from eva.clients.openai_compat import (  # noqa: E402
    ClientError,
    HTTPResponse,
    OpenAIClient,
    StreamChunk as ClientStreamChunk,
    StreamResponse,
)
from eva.config import Config  # noqa: E402
from eva.router import StreamChunk  # noqa: E402
from eva.types import Message, Request, Route  # noqa: E402

HOST = "127.0.0.1"


# --------------------------------------------------------------------------- #
# заглушки клиентов и маршрутов
# --------------------------------------------------------------------------- #
def make_req(text: str = "привет") -> Request:
    return Request(
        messages=[Message("system", "ты ассистент"), Message("user", text)],
        meta={"user_id": "u-1", "channel": "test"},
    )


def small_route(text: str = "", cfg=None) -> Route:
    return Route("small", "classifier: короткий вопрос — хватит малой модели", 0.8)


def big_route(text: str = "", cfg=None) -> Route:
    return Route("big", "classifier: нужен код — большая модель", 0.9)


class FakeStreamClient:
    """Клиент с ``stream_chat``: куски текста, usage и управляемые ошибки."""

    def __init__(
        self,
        pieces=("раз", "два", "три"),
        *,
        usage: dict | None = None,
        error: Exception | None = None,
        midstream_error: Exception | None = None,
    ) -> None:
        self.pieces = list(pieces)
        self.usage = usage
        self.error = error
        self.midstream_error = midstream_error
        self.calls: list[tuple[list[Message], int]] = []

    def stream_chat(self, messages, max_tokens: int = 512):
        self.calls.append((list(messages), max_tokens))
        if self.error is not None:
            raise self.error
        for index, piece in enumerate(self.pieces):
            if index and self.midstream_error is not None:
                raise self.midstream_error
            yield piece
        if self.usage is not None:
            yield ClientStreamChunk(usage=dict(self.usage), done=True)


class FakeChatClient:
    """Клиент без ``stream_chat``: роутер обязан деградировать до одного куска."""

    def __init__(self, text: str = "ответ целиком", usage: dict | None = None) -> None:
        self.text = text
        self.usage = usage or {"prompt_tokens": 3, "completion_tokens": 2}
        self.calls: list[tuple[list[Message], int]] = []

    def chat(self, messages, max_tokens: int = 512):
        self.calls.append((list(messages), max_tokens))
        return {"text": self.text, "usage": dict(self.usage), "raw": {"stub": True}}


def collect(req: Request, cfg: Config, deps) -> list[StreamChunk]:
    return list(router.stream(req, cfg, deps))


def text_of(chunks) -> str:
    return "".join(chunk.text for chunk in chunks)


# --------------------------------------------------------------------------- #
# контракт router.stream
# --------------------------------------------------------------------------- #
class StreamContractTest(unittest.TestCase):
    def test_signature_is_req_cfg_deps(self) -> None:
        params = list(inspect.signature(router.stream).parameters)
        self.assertEqual(params, ["req", "cfg", "deps"])

    def test_returns_iterator_not_list(self) -> None:
        cfg = Config()
        result = router.stream(make_req(), cfg, {"skills": lambda text, ctx: None})
        self.assertTrue(hasattr(result, "__next__"))
        self.assertIs(iter(result), result)
        result.close()

    def test_stream_is_exported(self) -> None:
        self.assertIn("stream", router.__all__)
        self.assertIn("StreamChunk", router.__all__)


# --------------------------------------------------------------------------- #
# скилл: один чанк и цена 0
# --------------------------------------------------------------------------- #
class SkillStreamTest(unittest.TestCase):
    def test_skill_streams_single_chunk_with_route_and_zero_cost(self) -> None:
        cfg = Config()
        classifier = mock.Mock(side_effect=AssertionError("классификатор не нужен"))
        client = mock.Mock()
        deps = {
            "skills": lambda text, ctx: {
                "skill": "pricing",
                "text": "Тариф «Старт» — 490 ₽/мес [1]",
                "sources": ["[1]"],
            },
            "classifier": classifier,
            "small": client,
            "big": client,
        }

        chunks = collect(make_req("сколько стоит тариф Старт"), cfg, deps)

        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertTrue(chunk.done)
        self.assertEqual(chunk.text, "Тариф «Старт» — 490 ₽/мес [1]")
        self.assertEqual(chunk.route.kind, "skill")
        self.assertEqual(chunk.skill, "pricing")
        self.assertIn("pricing", chunk.route.reason)
        self.assertEqual(chunk.cost_rub, 0.0)
        self.assertEqual(chunk.usage["cost_rub"], 0.0)
        self.assertEqual(chunk.usage["prompt_tokens"], 0)
        self.assertIsInstance(chunk.latency_ms, int)
        classifier.assert_not_called()
        client.chat.assert_not_called()
        client.stream_chat.assert_not_called()

    def test_skill_without_name_still_single_chunk(self) -> None:
        cfg = Config()
        deps = {"skills": lambda text, ctx: {"text": "ответ скилла"}}
        with mock.patch.object(router, "_load_classifier", return_value=None):
            chunks = collect(make_req("вопрос"), cfg, deps)
        self.assertEqual(len(chunks), 1)
        self.assertIsNone(chunks[0].skill)
        self.assertEqual(chunks[0].route.kind, "skill")


# --------------------------------------------------------------------------- #
# модель: поток из клиента, cost в финальном чанке
# --------------------------------------------------------------------------- #
class ModelStreamTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = Config()

    def _deps(self, client, **kwargs):
        deps = {"skills": lambda text, ctx: None, "classifier": small_route}
        deps.update(kwargs)
        deps.setdefault("small", client)
        return deps

    def test_model_streams_three_pieces_then_final_chunk(self) -> None:
        client = FakeStreamClient(
            ("раз", "два", "три"),
            usage={"prompt_tokens": 10, "completion_tokens": 3, "cost_rub": 0.42},
        )

        chunks = collect(make_req(), self.cfg, self._deps(client))

        self.assertEqual(len(chunks), 4)  # 3 куска + финальный
        content = [chunk for chunk in chunks if chunk.text]
        self.assertEqual([chunk.text for chunk in content], ["раз", "два", "три"])
        self.assertTrue(all(not chunk.done for chunk in content))
        self.assertTrue(all(chunk.route is None for chunk in content))
        self.assertEqual(text_of(chunks), "раздватри")

        final = chunks[-1]
        self.assertTrue(final.done)
        self.assertEqual(final.text, "")
        self.assertEqual(final.route.kind, "small")
        self.assertIn("короткий вопрос", final.route.reason)
        self.assertEqual(final.usage["prompt_tokens"], 10)
        self.assertEqual(final.usage["completion_tokens"], 3)
        self.assertEqual(final.cost_rub, 0.42)
        self.assertIsInstance(final.latency_ms, int)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0][1], self.cfg.max_tokens)

    def test_provider_cost_wins_over_tariff(self) -> None:
        cfg = Config(
            small_price_prompt_rub_per_1k=1.0,
            small_price_completion_rub_per_1k=1.0,
        )
        client = FakeStreamClient(
            usage={"prompt_tokens": 10, "completion_tokens": 5, "cost_rub": 0.42}
        )

        final = collect(make_req(), cfg, self._deps(client))[-1]

        self.assertEqual(final.cost_rub, 0.42)

    def test_tokens_without_cost_are_priced_by_config(self) -> None:
        cfg = Config(
            small_price_prompt_rub_per_1k=0.1,
            small_price_completion_rub_per_1k=0.2,
        )
        client = FakeStreamClient(
            usage={"prompt_tokens": 1000, "completion_tokens": 1000}
        )

        final = collect(make_req(), cfg, self._deps(client))[-1]

        self.assertAlmostEqual(final.cost_rub, 0.3, places=6)
        self.assertAlmostEqual(final.usage["cost_rub"], 0.3, places=6)

    def test_silent_provider_gets_local_estimate(self) -> None:
        cfg = Config(
            small_price_prompt_rub_per_1k=1.0,
            small_price_completion_rub_per_1k=2.0,
        )
        client = FakeStreamClient(("раз", "два", "три"), usage=None)
        req = make_req()
        prompt_text = router._prompt_text(req, req.messages)
        expected_prompt = policy.estimate_tokens(prompt_text)
        expected_completion = policy.estimate_tokens("раздватри")
        expected_cost = cfg.estimate_cost("small", expected_prompt, expected_completion)

        final = collect(req, cfg, self._deps(client))[-1]

        self.assertEqual(final.usage["prompt_tokens"], expected_prompt)
        self.assertEqual(final.usage["completion_tokens"], expected_completion)
        self.assertAlmostEqual(final.cost_rub, expected_cost, places=6)

    def test_client_without_stream_chat_answers_one_chunk(self) -> None:
        client = FakeChatClient("ответ целиком")

        chunks = collect(make_req(), self.cfg, self._deps(client))

        self.assertEqual([chunk.text for chunk in chunks if chunk.text], ["ответ целиком"])
        self.assertEqual(chunks[-1].route.kind, "small")
        self.assertEqual(chunks[-1].usage["prompt_tokens"], 3)

    def test_small_failure_before_first_chunk_falls_back_to_big(self) -> None:
        cfg = Config(big_base="https://api.example/v1", big_model="big-1")
        small = FakeStreamClient(
            error=ClientError("таймаут 30 с", kind="timeout", retryable=True)
        )
        big = FakeStreamClient(("боль", "шая"))

        chunks = collect(
            make_req(),
            cfg,
            self._deps(small, big=big),
        )

        self.assertEqual(text_of(chunks), "большая")
        final = chunks[-1]
        self.assertEqual(final.route.kind, "big")
        self.assertIn("fallback small→big", final.route.reason)
        self.assertIn("таймаут", final.route.reason)
        self.assertEqual(len(small.calls), 1)
        self.assertEqual(len(big.calls), 1)

    def test_big_429_falls_back_to_small(self) -> None:
        cfg = Config(big_base="https://api.example/v1", big_model="big-1")
        big = FakeStreamClient(
            error=ClientError("429 Too Many Requests", kind="rate_limit", status=429)
        )
        small = FakeStreamClient(("мал", "ый"))

        chunks = collect(
            make_req("объясни"),
            cfg,
            {
                "skills": lambda text, ctx: None,
                "classifier": big_route,
                "big": big,
                "small": small,
            },
        )

        self.assertEqual(text_of(chunks), "малый")
        self.assertEqual(chunks[-1].route.kind, "small")
        self.assertIn("fallback big→small", chunks[-1].route.reason)

    def test_error_after_first_piece_is_not_retried_on_another_model(self) -> None:
        cfg = Config(big_base="https://api.example/v1", big_model="big-1")
        small = FakeStreamClient(("нач", "дальше"), midstream_error=RuntimeError("обрыв"))
        big = FakeStreamClient()

        iterator = router.stream(make_req(), cfg, self._deps(small, big=big))
        first = next(iterator)
        self.assertEqual(first.text, "нач")
        self.assertFalse(first.done)
        with self.assertRaises(RuntimeError):
            next(iterator)
        self.assertEqual(big.calls, [], "фолбэк после отданного текста запрещён")

    def test_budget_is_checked_before_streaming(self) -> None:
        cfg = Config(
            max_cost_rub=0.0001,
            small_price_prompt_rub_per_1k=1.0,
            small_price_completion_rub_per_1k=1.0,
        )
        client = FakeStreamClient()

        iterator = router.stream(make_req(), cfg, self._deps(client))

        with self.assertRaises(policy.BudgetExceeded):
            next(iterator)
        self.assertEqual(client.calls, [])


# --------------------------------------------------------------------------- #
# SSE-клиент: разбор потока провайдера
# --------------------------------------------------------------------------- #
def sse_event(payload: dict) -> bytes:
    return b"data: " + json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n"


def content_event(text: str) -> bytes:
    return sse_event(
        {"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]}
    )


def finish_event(*, usage: dict | None = None) -> bytes:
    payload = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
    if usage is not None:
        payload["usage"] = usage
    return sse_event(payload)


DONE_EVENT = b"data: [DONE]\n"


class FakeStreamBody:
    """Тело SSE: отдаёт строки, умеет падать на чтении и запоминает закрытие."""

    def __init__(self, lines=(), *, error: Exception | None = None, fail_after: int | None = None) -> None:
        self.lines = list(lines)
        self.index = 0
        self.closed = False
        self.error = error
        self.fail_after = len(self.lines) if fail_after is None else fail_after

    def readline(self):
        if self.error is not None and self.index >= self.fail_after:
            raise self.error
        if self.index >= len(self.lines):
            return b""
        line = self.lines[self.index]
        self.index += 1
        return line

    def close(self) -> None:
        self.closed = True


class FakeStreamTransport:
    """Подмена ``_open_stream``: очередь ответов и запись запросов."""

    def __init__(self, *responses) -> None:
        if not responses:
            raise AssertionError("нужен хотя бы один ответ")
        self.responses = list(responses)
        self.calls = []

    def __call__(self, url, headers, body, timeout):
        self.calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "payload": json.loads(body.decode("utf-8")),
                "timeout": timeout,
            }
        )
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, BaseException):
            raise item
        return item


class SleepRecorder:
    def __init__(self) -> None:
        self.delays = []

    def __call__(self, seconds) -> None:
        self.delays.append(seconds)


def make_stream_client(transport, **kwargs) -> OpenAIClient:
    params = {
        "base": "http://127.0.0.1:11434",
        "key": "sk-test",
        "model": "eva32-dev",
        "timeout": 5,
        "retries": 0,
    }
    params.update(kwargs)
    return OpenAIClient(stream_transport=transport, **params)


class ClientStreamParsingTest(unittest.TestCase):
    def test_parses_deltas_usage_and_done(self) -> None:
        body = FakeStreamBody(
            [
                content_event("При"),
                content_event("вет"),
                finish_event(
                    usage={
                        "prompt_tokens": 7,
                        "completion_tokens": 2,
                        "cost_rub": 0.25,
                    }
                ),
                DONE_EVENT,
            ]
        )
        transport = FakeStreamTransport(StreamResponse(status=200, body=body))
        client = make_stream_client(transport)

        chunks = list(
            client.stream_chat([{"role": "user", "content": "привет"}], max_tokens=64)
        )

        self.assertEqual([c.text for c in chunks if c.text], ["При", "вет"])
        final = chunks[-1]
        self.assertTrue(final.done)
        self.assertEqual(final.finish_reason, "stop")
        self.assertEqual(final.usage["prompt_tokens"], 7)
        self.assertEqual(final.usage["completion_tokens"], 2)
        self.assertEqual(final.usage["total_tokens"], 9)
        self.assertEqual(final.usage["cost_rub"], 0.25)
        self.assertTrue(body.closed, "ответ провайдера должен закрываться")

        call = transport.calls[0]
        self.assertEqual(call["url"], "http://127.0.0.1:11434/v1/chat/completions")
        self.assertTrue(call["payload"]["stream"])
        self.assertEqual(
            call["payload"]["stream_options"], {"include_usage": True}
        )
        self.assertEqual(call["payload"]["max_tokens"], 64)
        self.assertEqual(call["headers"]["Accept"], "text/event-stream")
        self.assertEqual(call["headers"]["Authorization"], "Bearer sk-test")
        self.assertEqual(call["timeout"], 5)

    def test_finish_reason_without_done_is_accepted(self) -> None:
        body = FakeStreamBody([content_event("ок"), finish_event()])
        client = make_stream_client(FakeStreamTransport(StreamResponse(200, body)))

        chunks = list(client.stream_chat("привет"))

        self.assertEqual([c.text for c in chunks if c.text], ["ок"])
        self.assertTrue(chunks[-1].done)
        self.assertEqual(chunks[-1].usage, {}, "usage не выдумывается клиентом")

    def test_truncated_stream_raises_stream_error(self) -> None:
        body = FakeStreamBody([content_event("обрыв")])
        client = make_stream_client(FakeStreamTransport(StreamResponse(200, body)))

        with self.assertRaises(ClientError) as ctx:
            list(client.stream_chat("привет"))

        self.assertEqual(ctx.exception.kind, "stream")
        self.assertTrue(ctx.exception.retryable)
        self.assertIn("оборвался", str(ctx.exception))
        self.assertTrue(body.closed)

    def test_timeout_during_read_raises_timeout(self) -> None:
        body = FakeStreamBody(
            [content_event("нач")], error=socket.timeout("timed out"), fail_after=1
        )
        client = make_stream_client(FakeStreamTransport(StreamResponse(200, body)))

        with self.assertRaises(ClientError) as ctx:
            list(client.stream_chat("привет"))

        self.assertEqual(ctx.exception.kind, "timeout")
        self.assertTrue(body.closed)

    def test_malformed_event_json_is_response_error(self) -> None:
        body = FakeStreamBody([b"data: {broken json}\n"])
        client = make_stream_client(FakeStreamTransport(StreamResponse(200, body)))

        with self.assertRaises(ClientError) as ctx:
            list(client.stream_chat("привет"))

        self.assertEqual(ctx.exception.kind, "response")
        self.assertFalse(ctx.exception.retryable)
        self.assertTrue(body.closed)

    def test_http_error_has_status_and_detail(self) -> None:
        body = io.BytesIO(b'{"error": {"message": "slow down"}}')
        transport = FakeStreamTransport(
            StreamResponse(
                status=429, body=body, headers={"Retry-After": "0.2"}
            )
        )
        client = make_stream_client(transport)

        with self.assertRaises(ClientError) as ctx:
            list(client.stream_chat("привет"))

        self.assertEqual(ctx.exception.status, 429)
        self.assertTrue(ctx.exception.retryable)
        self.assertIn("slow down", str(ctx.exception))
        self.assertTrue(body.closed)

    def test_non_retryable_http_error_is_not_retried(self) -> None:
        body = io.BytesIO(b'{"error": {"message": "bad request"}}')
        transport = FakeStreamTransport(StreamResponse(status=400, body=body))
        client = make_stream_client(transport, retries=2)

        with self.assertRaises(ClientError) as ctx:
            list(client.stream_chat("привет"))

        self.assertEqual(ctx.exception.status, 400)
        self.assertFalse(ctx.exception.retryable)
        self.assertEqual(len(transport.calls), 1)

    def test_retry_before_first_piece_uses_retry_after(self) -> None:
        limited = StreamResponse(
            status=429,
            body=io.BytesIO(b'{"error": {"message": "slow down"}}'),
            headers={"Retry-After": "0.2"},
        )
        good = StreamResponse(
            status=200,
            body=FakeStreamBody([content_event("ок"), DONE_EVENT]),
        )
        transport = FakeStreamTransport(limited, good)
        sleep = SleepRecorder()
        client = make_stream_client(transport, retries=1, sleep=sleep)

        chunks = list(client.stream_chat("привет"))

        self.assertEqual([c.text for c in chunks if c.text], ["ок"])
        self.assertEqual(sleep.delays, [0.2])
        self.assertEqual(len(transport.calls), 2)

    def test_truncated_stream_is_not_retried_after_text(self) -> None:
        truncated = FakeStreamBody([content_event("уже отдано")])
        transport = FakeStreamTransport(StreamResponse(200, truncated))
        sleep = SleepRecorder()
        client = make_stream_client(transport, retries=2, sleep=sleep)

        with self.assertRaises(ClientError) as ctx:
            list(client.stream_chat("привет"))

        self.assertEqual(ctx.exception.kind, "stream")
        self.assertEqual(len(transport.calls), 1, "после текста повтор запрещён")
        self.assertEqual(sleep.delays, [])

    def test_bad_arguments_raise_before_network(self) -> None:
        transport = FakeStreamTransport(
            StreamResponse(status=200, body=FakeStreamBody([DONE_EVENT]))
        )
        client = make_stream_client(transport)

        with self.assertRaises(TypeError):
            client.stream_chat("привет", max_tokens=True)
        with self.assertRaises(ValueError):
            client.stream_chat("привет", max_tokens=0)

        self.assertEqual(transport.calls, [], "сеть не должна дёргаться до проверок")

    def test_injected_buffered_transport_is_accepted(self) -> None:
        """Транспорт chat() может отдать уже прочитанное тело — тоже разбираем."""
        raw = b"".join([content_event("ок"), DONE_EVENT])
        transport = FakeStreamTransport(HTTPResponse(status=200, body=raw))
        client = OpenAIClient(
            base="http://127.0.0.1:11434",
            key="",
            model="eva32-dev",
            timeout=5,
            retries=0,
            transport=transport,
        )

        chunks = list(client.stream_chat("привет"))

        self.assertEqual([c.text for c in chunks if c.text], ["ок"])
        self.assertTrue(chunks[-1].done)

    def test_closing_generator_closes_provider_response(self) -> None:
        body = FakeStreamBody(
            [content_event("раз"), content_event("два"), DONE_EVENT]
        )
        client = make_stream_client(FakeStreamTransport(StreamResponse(200, body)))

        iterator = client.stream_chat("привет")
        self.assertEqual(next(iterator).text, "раз")
        self.assertFalse(body.closed)
        iterator.close()

        self.assertTrue(body.closed, "close() генератора должен закрыть ответ")


# --------------------------------------------------------------------------- #
# сервер: формат событий не меняется, обрыв не роняет процесс
# --------------------------------------------------------------------------- #
class _RouterAdapter:
    """Серверу нужен объект с answer+stream, а eva.router — две функции."""

    def __init__(self, cfg: Config, deps) -> None:
        self.cfg = cfg
        self.deps = deps

    def answer(self, request):
        return router.answer(request, self.cfg, self.deps)

    def stream(self, request):
        return router.stream(request, self.cfg, self.deps)


class ServerHarness:
    def __init__(self, cfg: Config, deps) -> None:
        self.httpd = server.create_server(
            host=HOST, port=0, router=_RouterAdapter(cfg, deps)
        )
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(
            target=self.httpd.serve_forever,
            kwargs={"poll_interval": 0.01},
            daemon=True,
        )
        self.thread.start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)
        assert not self.thread.is_alive(), "поток сервера не завершился"

    def request(self, method, path, body=None):
        connection = http.client.HTTPConnection(HOST, self.port, timeout=5)
        try:
            payload = (
                json.dumps(body, ensure_ascii=False).encode("utf-8")
                if body is not None
                else None
            )
            headers = {"Content-Type": "application/json"} if payload else {}
            connection.request(method, path, body=payload, headers=headers)
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def chat_stream(self, text: str):
        return self.request(
            "POST",
            "/v1/chat/completions",
            {"stream": True, "messages": [{"role": "user", "content": text}]},
        )


def parse_sse(raw: bytes) -> list[str]:
    text = raw.decode("utf-8").replace("\r\n", "\n")
    return [
        line[len("data:") :].strip()
        for line in text.split("\n")
        if line.startswith("data:")
    ]


def parse_sse_chunks(raw: bytes) -> tuple[list[dict], bool]:
    events = parse_sse(raw)
    done = bool(events) and events[-1] == "[DONE]" and events.count("[DONE]") == 1
    return [json.loads(event) for event in events if event != "[DONE]"], done


class ServerStreamIntegrationTest(unittest.TestCase):
    def test_skill_answer_is_one_chunk_with_zero_cost(self) -> None:
        deps = {
            "skills": lambda text, ctx: {
                "skill": "pricing",
                "text": "Старт — 490 ₽ [1]",
                "sources": ["[1]"],
            },
            "classifier": mock.Mock(side_effect=AssertionError("не вызывать")),
        }
        harness = ServerHarness(Config(), deps)
        try:
            status, headers, raw = harness.chat_stream("сколько стоит Старт")

            self.assertEqual(status, 200)
            self.assertIn("text/event-stream", headers["Content-Type"])
            chunks, done = parse_sse_chunks(raw)
            self.assertTrue(done)
            contents = [
                chunk["choices"][0]["delta"]["content"]
                for chunk in chunks
                if chunk.get("choices")
            ]
            self.assertEqual("".join(contents), "Старт — 490 ₽ [1]")
            final = [c for c in chunks if c.get("choices")][-1]
            self.assertEqual(final["choices"][0]["finish_reason"], "stop")
            self.assertEqual(final["eva"]["route"], "skill")
            self.assertEqual(final["eva"]["skill"], "pricing")
            self.assertEqual(final["eva"]["cost_rub"], 0.0)

            health, _, _ = harness.request("GET", "/health")
            self.assertEqual(health, 200)
        finally:
            harness.close()

    def test_model_answer_streams_three_or_more_chunks(self) -> None:
        client = FakeStreamClient(
            ("раз", "два", "три"),
            usage={"prompt_tokens": 10, "completion_tokens": 3, "cost_rub": 0.07},
        )
        deps = {
            "skills": lambda text, ctx: None,
            "classifier": small_route,
            "small": client,
        }
        harness = ServerHarness(Config(), deps)
        try:
            status, _, raw = harness.chat_stream("привет")

            self.assertEqual(status, 200)
            chunks, done = parse_sse_chunks(raw)
            self.assertTrue(done)
            choices = [chunk for chunk in chunks if chunk.get("choices")]
            contents = [chunk["choices"][0]["delta"]["content"] for chunk in choices]
            self.assertEqual(len([text for text in contents if text]), 3)
            self.assertEqual("".join(contents), "раздватри")
            final = choices[-1]
            self.assertEqual(final["eva"]["route"], "small")
            self.assertAlmostEqual(final["eva"]["cost_rub"], 0.07)
        finally:
            harness.close()

    def test_midstream_break_ends_with_error_event_and_server_survives(self) -> None:
        client = FakeStreamClient(
            ("нач", "дальше"), midstream_error=RuntimeError("секрет sk-live-9")
        )
        deps = {
            "skills": lambda text, ctx: None,
            "classifier": small_route,
            "small": client,
        }
        harness = ServerHarness(Config(), deps)
        try:
            with self.assertLogs("eva.server", level="ERROR"):
                status, _, raw = harness.chat_stream("привет")

            self.assertEqual(status, 200)
            events = parse_sse(raw)
            self.assertEqual(events[-1], "[DONE]")
            first = json.loads(events[0])
            self.assertEqual(first["choices"][0]["delta"]["content"], "нач")
            error = json.loads(events[-2])
            self.assertEqual(error["error"]["code"], "stream_error")
            for secret in ("Traceback", "RuntimeError", "sk-live-9", "секрет"):
                self.assertNotIn(secret.encode("utf-8"), raw)

            # Сервер жив: обрыв потока не уронил ни поток, ни процесс.
            health, _, health_body = harness.request("GET", "/health")
            self.assertEqual(health, 200)
            self.assertEqual(json.loads(health_body)["status"], "ok")
        finally:
            harness.close()


class DefaultStreamWiringTest(unittest.TestCase):
    """Боевая связка: create_server() без router обязан стримить router.stream."""

    def test_make_handler_without_router_uses_stream(self) -> None:
        handler = server.make_handler(None)
        self.assertIs(handler._router, server.default_router)
        self.assertIs(handler._stream_router, server.default_stream_router)

    def test_default_stream_router_is_not_as_stream_router_of_default(self) -> None:
        # Контракт as_stream_router(default_router) закреплён тестами сервера:
        # у функции default_router нет метода stream, есть отдельная связка.
        self.assertIsNone(server.as_stream_router(server.default_router))


if __name__ == "__main__":
    unittest.main()
