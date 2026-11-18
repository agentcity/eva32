"""Проверки HTTP-сервера eva-server.

Запуск: ``python3 -m unittest tests.test_server -v``

Сервер поднимается по-настоящему на свободном порту loopback, а маршрутизатор
подменён заглушкой: тесты не ходят во внешнюю сеть и не требуют моделей.
Проверяется поведение снаружи (по HTTP), а не внутренние функции.
"""

import http.client
import json
import os
import socket
import sys
import threading
import time
import unittest
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eva import server  # noqa: E402

HOST = "127.0.0.1"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class StubRoute:
    kind: str = "small"
    reason: str = "заглушка: короткий вопрос"
    confidence: float = 0.8


@dataclass
class StubAnswer:
    text: str = "Здравствуйте! Чем помочь?"
    route: StubRoute = field(default_factory=StubRoute)
    skill: str | None = None
    usage: dict = field(
        default_factory=lambda: {"prompt_tokens": 12, "completion_tokens": 5}
    )
    cost_rub: float = 0.0
    latency_ms: int = 3


class StubRouter:
    """Заглушка маршрутизатора: запоминает запросы и отдаёт заданный ответ."""

    def __init__(self) -> None:
        self.calls = []
        self.answer = StubAnswer()
        self.error = None

    def __call__(self, request):
        self.calls.append(request)
        if self.error is not None:
            raise self.error
        return self.answer

    def reset(self) -> None:
        self.calls.clear()
        self.answer = StubAnswer()
        self.error = None


class StreamingStubRouter:
    """Заглушка, которая умеет отдавать ответ по частям через ``stream``.

    ``error`` бросается до первого куска (ещё можно ответить JSON),
    ``midstream_error`` — после первого (уже только событием SSE).
    """

    def __init__(self, pieces=("При", "вет", "!")) -> None:
        self._default_pieces = list(pieces)
        self.pieces = list(pieces)
        self.calls = []
        self.stream_calls = []
        self.answer = StubAnswer()
        self.error = None
        self.midstream_error = None

    def stream(self, request):
        self.stream_calls.append(request)
        if self.error is not None:
            raise self.error
        for index, piece in enumerate(self.pieces):
            if index and self.midstream_error is not None:
                raise self.midstream_error
            yield piece

    def __call__(self, request):
        self.calls.append(request)
        return self.answer

    def reset(self) -> None:
        self.calls.clear()
        self.stream_calls.clear()
        self.pieces = list(self._default_pieces)
        self.answer = StubAnswer()
        self.error = None
        self.midstream_error = None


class GatedStreamRouter(StreamingStubRouter):
    """Отдаёт второй кусок только после ``release`` — проверка инкрементальности."""

    def __init__(self) -> None:
        super().__init__(pieces=[])
        self.release = threading.Event()
        self.finished = threading.Event()

    def stream(self, request):
        self.stream_calls.append(request)
        yield "Первый "
        self.release.wait(timeout=5)
        yield "второй"
        self.finished.set()


def raw_http(port: int, request_bytes: bytes, timeout: float = 5.0) -> bytes:
    """Сырой HTTP-запрос: нужен там, где http.client не даёт собрать байты."""
    with socket.create_connection((HOST, port), timeout=timeout) as sock:
        sock.sendall(request_bytes)
        chunks = []
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
    return b"".join(chunks)


def parse_raw(raw: bytes) -> tuple[int, bytes]:
    head, _, body = raw.partition(b"\r\n\r\n")
    status_line = head.split(b"\r\n", 1)[0]
    return int(status_line.split()[1]), body


def parse_sse(raw: bytes) -> list[str]:
    """Поля ``data:`` событий SSE в порядке появления.

    Нормализует CRLF, чтобы функция годилась и для тела, уже разобранного
    ``http.client``, и для сырых chunked-кадров из сокета.
    """
    text = raw.decode("utf-8").replace("\r\n", "\n")
    events = []
    for line in text.split("\n"):
        if line.startswith("data:"):
            events.append(line[len("data:") :].strip())
    return events


def parse_sse_json(raw: bytes) -> tuple[list[dict], bool]:
    """(JSON-чанки, был ли ровно один терминальный ``[DONE]`` в конце)."""
    events = parse_sse(raw)
    done = bool(events) and events[-1] == "[DONE]" and events.count("[DONE]") == 1
    chunks = [json.loads(event) for event in events if event != "[DONE]"]
    return chunks, done


class ServerCase(unittest.TestCase):
    """Общий стенд: настоящий сервер на локальном эфемерном порту."""

    router_factory = StubRouter

    @classmethod
    def setUpClass(cls) -> None:
        cls.router = cls.router_factory()
        cls.httpd = server.create_server(host=HOST, port=0, router=cls.router)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(
            target=cls.httpd.serve_forever,
            kwargs={"poll_interval": 0.01},
            name="eva-test-server",
            daemon=True,
        )
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=5)
        assert not cls.thread.is_alive(), "поток сервера не завершился после shutdown"

    def setUp(self) -> None:
        self.router.reset()

    def request(self, method, path, *, body=None, raw=None, headers=None, timeout=5.0):
        connection = http.client.HTTPConnection(HOST, self.port, timeout=timeout)
        request_headers = dict(headers or {})
        if raw is not None:
            payload = raw
        elif body is not None:
            payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
            request_headers.setdefault("Content-Type", "application/json")
        else:
            payload = None
        try:
            connection.request(method, path, body=payload, headers=request_headers)
            response = connection.getresponse()
            data = response.read()
            content_type = response.getheader("Content-Type") or ""
            parsed = (
                json.loads(data)
                if data and content_type.startswith("application/json")
                else None
            )
            return response.status, dict(response.getheaders()), data, parsed
        finally:
            connection.close()

    def post_chat(self, payload, **kwargs):
        return self.request("POST", "/v1/chat/completions", body=payload, **kwargs)


class TestHealth(ServerCase):
    def test_health_returns_status_and_version(self):
        status, headers, _, body = self.request("GET", "/health")

        self.assertEqual(status, 200)
        self.assertIn("application/json", headers["Content-Type"])
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["service"], "eva-server")
        self.assertEqual(body["model"], server.DEFAULT_MODEL)
        with open(os.path.join(REPO_ROOT, "VERSION"), encoding="utf-8") as handle:
            self.assertEqual(body["version"], handle.read().strip())

    def test_health_reports_injected_router(self):
        _, _, _, body = self.request("GET", "/health")

        self.assertEqual(body["router"], "injected")
        self.assertIsInstance(body["uptime_s"], (int, float))
        self.assertGreaterEqual(body["uptime_s"], 0)


class TestChatCompletions(ServerCase):
    def test_openai_response_shape(self):
        status, _, _, body = self.post_chat(
            {
                "model": "orioncode-test",
                "messages": [{"role": "user", "content": "Привет"}],
            }
        )

        self.assertEqual(status, 200)
        self.assertTrue(body["id"].startswith("chatcmpl-"))
        self.assertEqual(body["object"], "chat.completion")
        self.assertIsInstance(body["created"], int)
        self.assertEqual(body["model"], "orioncode-test")
        self.assertEqual(len(body["choices"]), 1)
        choice = body["choices"][0]
        self.assertEqual(choice["index"], 0)
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertEqual(
            choice["message"],
            {"role": "assistant", "content": "Здравствуйте! Чем помочь?"},
        )

    def test_eva_field_carries_route_skill_and_price(self):
        self.router.answer = StubAnswer(
            text="Тариф «Старт» стоит 990 ₽/мес.",
            route=StubRoute(
                kind="skill",
                reason="каталог: тариф «Старт»",
                confidence=0.93,
            ),
            skill="pricing",
            cost_rub=0.0,
            latency_ms=7,
        )

        status, _, _, body = self.post_chat(
            {"messages": [{"role": "user", "content": "Сколько стоит «Старт»?"}]}
        )

        self.assertEqual(status, 200)
        eva = body["eva"]
        self.assertEqual(eva["route"], "skill")
        self.assertEqual(eva["reason"], "каталог: тариф «Старт»")
        self.assertAlmostEqual(eva["confidence"], 0.93)
        self.assertEqual(eva["skill"], "pricing")
        self.assertEqual(eva["cost_rub"], 0.0)
        self.assertEqual(eva["latency_ms"], 7)

    def test_usage_is_openai_shaped(self):
        _, _, _, body = self.post_chat(
            {"messages": [{"role": "user", "content": "Привет"}]}
        )

        usage = body["usage"]
        self.assertEqual(usage["prompt_tokens"], 12)
        self.assertEqual(usage["completion_tokens"], 5)
        self.assertEqual(usage["total_tokens"], 17)

    def test_default_model_when_client_did_not_ask(self):
        _, _, _, body = self.post_chat(
            {"messages": [{"role": "user", "content": "Привет"}]}
        )

        self.assertEqual(body["model"], server.DEFAULT_MODEL)

    def test_router_receives_parsed_request(self):
        self.post_chat(
            {
                "user": "u-42",
                "messages": [
                    {"role": "system", "content": "Ты — EVA."},
                    {"role": "user", "content": "Привет"},
                ],
            }
        )

        self.assertEqual(len(self.router.calls), 1)
        request = self.router.calls[0]
        self.assertEqual([m.role for m in request.messages], ["system", "user"])
        self.assertEqual([m.content for m in request.messages], ["Ты — EVA.", "Привет"])
        self.assertEqual(request.meta, {"channel": "http", "user_id": "u-42"})

    def test_meta_without_user_has_only_channel(self):
        self.post_chat({"messages": [{"role": "user", "content": "Привет"}]})

        self.assertEqual(self.router.calls[0].meta, {"channel": "http"})

    def test_content_parts_are_joined(self):
        self.post_chat(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "При"},
                            {"type": "text", "text": "вет"},
                        ],
                    }
                ]
            }
        )

        self.assertEqual(self.router.calls[0].messages[-1].content, "Привет")

    def test_unicode_is_not_escaped(self):
        text = "Стоимость 990 ₽/мес."
        self.router.answer = StubAnswer(text=text)

        _, _, raw, body = self.post_chat(
            {"messages": [{"role": "user", "content": "Сколько стоит?"}]}
        )

        self.assertEqual(body["choices"][0]["message"]["content"], text)
        self.assertIn("990 ₽", raw.decode("utf-8"))

    def test_router_may_return_dict(self):
        self.router.answer = {
            "text": "Ответ большой модели.",
            "route": {"kind": "big", "reason": "нужны общие знания", "confidence": 0.6},
            "skill": None,
            "usage": {"prompt_tokens": 3, "completion_tokens": 4},
            "cost_rub": 1.25,
            "latency_ms": 41,
        }

        status, _, _, body = self.post_chat(
            {"messages": [{"role": "user", "content": "Объясни, почему небо синее"}]}
        )

        self.assertEqual(status, 200)
        self.assertEqual(
            body["choices"][0]["message"]["content"], "Ответ большой модели."
        )
        self.assertEqual(body["eva"]["route"], "big")
        self.assertIsNone(body["eva"]["skill"])
        self.assertAlmostEqual(body["eva"]["cost_rub"], 1.25)
        self.assertEqual(body["usage"]["total_tokens"], 7)

    def test_server_measures_latency_when_router_does_not(self):
        self.router.answer = {"text": "Ок", "route": {"kind": "small", "reason": ""}}

        _, _, _, body = self.post_chat(
            {"messages": [{"role": "user", "content": "Привет"}]}
        )

        self.assertGreaterEqual(body["eva"]["latency_ms"], 0)
        self.assertEqual(body["eva"]["confidence"], 0.0)
        self.assertEqual(body["eva"]["reason"], "")

    def test_models_endpoint_lists_default_model(self):
        status, _, _, body = self.request("GET", "/v1/models")

        self.assertEqual(status, 200)
        self.assertEqual(body["object"], "list")
        self.assertEqual(body["data"][0]["id"], server.DEFAULT_MODEL)


class TestChatStream(ServerCase):
    """Стриминг: настоящие SSE-чанки от роутера, который умеет отдавать по частям."""

    router_factory = StreamingStubRouter

    def test_stream_returns_openai_chunks_and_done(self):
        status, headers, raw, _ = self.post_chat(
            {
                "stream": True,
                "model": "eva-test",
                "messages": [{"role": "user", "content": "Привет"}],
            }
        )

        self.assertEqual(status, 200)
        self.assertIn("text/event-stream", headers["Content-Type"])
        self.assertEqual(headers["Cache-Control"], "no-cache")
        self.assertEqual(headers["Connection"], "keep-alive")
        self.assertEqual(headers.get("Transfer-Encoding", "").lower(), "chunked")

        chunks, done = parse_sse_json(raw)
        self.assertTrue(done, "поток должен завершаться одним data: [DONE]")
        self.assertEqual(len(chunks), len(self.router.pieces) + 1)

        contents = []
        for chunk in chunks:
            self.assertEqual(chunk["object"], "chat.completion.chunk")
            self.assertEqual(chunk["model"], "eva-test")
            self.assertTrue(chunk["id"].startswith("chatcmpl-"))
            self.assertEqual(len(chunk["choices"]), 1)
            self.assertEqual(chunk["choices"][0]["index"], 0)
            self.assertIsInstance(chunk["choices"][0]["delta"]["content"], str)
            contents.append(chunk["choices"][0]["delta"]["content"])

        self.assertEqual("".join(contents[:-1]), "Привет!")
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")
        for chunk in chunks[:-1]:
            self.assertIsNone(chunk["choices"][0]["finish_reason"])

    def test_stream_sends_chunks_before_router_finishes(self):
        gated = GatedStreamRouter()
        httpd = server.create_server(host=HOST, port=0, router=gated)
        port = httpd.server_address[1]
        thread = threading.Thread(
            target=httpd.serve_forever,
            kwargs={"poll_interval": 0.01},
            daemon=True,
        )
        thread.start()
        sock = None
        try:
            body = json.dumps(
                {
                    "stream": True,
                    "messages": [{"role": "user", "content": "Привет"}],
                }
            ).encode("utf-8")
            sock = socket.create_connection((HOST, port), timeout=5)
            sock.sendall(
                b"POST /v1/chat/completions HTTP/1.1\r\n"
                b"Host: test\r\nContent-Type: application/json\r\n"
                b"Content-Length: %d\r\n\r\n" % len(body) + body
            )

            received = b""
            deadline = time.monotonic() + 5
            while "Первый".encode("utf-8") not in received:
                self.assertLess(time.monotonic(), deadline, "первый чанк не пришёл")
                piece = sock.recv(4096)
                if not piece:
                    self.fail("соединение закрыто до первого чанка")
                received += piece

            # Первый чанк уже у клиента, а генерация ещё ждёт release.
            self.assertFalse(gated.release.is_set())
            self.assertFalse(gated.finished.is_set())

            gated.release.set()
            deadline = time.monotonic() + 5
            while b"[DONE]" not in received:
                self.assertLess(time.monotonic(), deadline, "поток не завершился")
                piece = sock.recv(4096)
                if not piece:
                    self.fail("соединение закрыто до [DONE]")
                received += piece

            chunks, done = parse_sse_json(received)
            self.assertTrue(done)
            text = "".join(c["choices"][0]["delta"]["content"] for c in chunks)
            self.assertEqual(text, "Первый второй")
            self.assertTrue(gated.finished.is_set())
        finally:
            if sock is not None:
                sock.close()
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=5)

    def test_stream_router_receives_parsed_request(self):
        self.post_chat(
            {
                "stream": True,
                "user": "u-7",
                "messages": [
                    {"role": "system", "content": "Ты — EVA."},
                    {"role": "user", "content": "Привет"},
                ],
            }
        )

        self.assertEqual(len(self.router.stream_calls), 1)
        request = self.router.stream_calls[0]
        self.assertEqual([m.role for m in request.messages], ["system", "user"])
        self.assertEqual(request.meta, {"channel": "http", "user_id": "u-7"})

    def test_stream_metadata_item_fills_eva_and_usage(self):
        self.router.pieces = [
            "Ответ",
            StubAnswer(
                text="",
                route=StubRoute(kind="skill", reason="каталог", confidence=0.9),
                skill="pricing",
                usage={"prompt_tokens": 5, "completion_tokens": 7},
                cost_rub=0.5,
            ),
        ]

        status, _, raw, _ = self.post_chat(
            {
                "stream": True,
                "stream_options": {"include_usage": True},
                "messages": [{"role": "user", "content": "Сколько стоит?"}],
            }
        )

        self.assertEqual(status, 200)
        events = parse_sse(raw)
        self.assertEqual(events[-1], "[DONE]")
        chunks = [json.loads(event) for event in events if event != "[DONE]"]

        content_chunks = [c for c in chunks if c["choices"]]
        usage_chunks = [c for c in chunks if not c["choices"]]
        self.assertEqual(len(content_chunks), 2)  # "Ответ" + финальный
        self.assertEqual(content_chunks[0]["choices"][0]["delta"]["content"], "Ответ")
        finish = content_chunks[-1]
        self.assertEqual(finish["choices"][0]["finish_reason"], "stop")
        self.assertEqual(finish["eva"]["route"], "skill")
        self.assertEqual(finish["eva"]["skill"], "pricing")
        self.assertEqual(finish["eva"]["cost_rub"], 0.5)

        self.assertEqual(len(usage_chunks), 1)
        self.assertEqual(usage_chunks[0]["usage"]["total_tokens"], 12)

    def test_non_stream_request_on_streaming_router_is_json(self):
        status, _, raw, body = self.post_chat(
            {"messages": [{"role": "user", "content": "Привет"}]}
        )

        self.assertEqual(status, 200)
        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(body["choices"][0]["message"]["content"], "Здравствуйте! Чем помочь?")
        self.assertEqual(self.router.stream_calls, [])
        self.assertEqual(len(self.router.calls), 1)
        self.assertNotIn(b"data:", raw)

    def test_error_before_first_chunk_is_json(self):
        self.router.error = RuntimeError("модель не поднялась")

        with self.assertLogs("eva.server", level="ERROR"):
            status, headers, raw, body = self.post_chat(
                {"stream": True, "messages": [{"role": "user", "content": "Привет"}]}
            )

        self.assertEqual(status, 500)
        self.assertIn("application/json", headers["Content-Type"])
        self.assertEqual(body["error"]["type"], "server_error")
        self.assertEqual(body["error"]["code"], "internal_error")
        self.assertNotIn(b"data:", raw)
        self.assertNotIn("модель не поднялась".encode("utf-8"), raw)

    def test_error_midstream_ends_with_error_event_and_done(self):
        self.router.midstream_error = RuntimeError("секрет sk-live-9")

        with self.assertLogs("eva.server", level="ERROR"):
            status, _, raw, _ = self.post_chat(
                {"stream": True, "messages": [{"role": "user", "content": "Привет"}]}
            )

        self.assertEqual(status, 200)
        events = parse_sse(raw)
        self.assertEqual(events[-1], "[DONE]")
        first = json.loads(events[0])
        self.assertEqual(first["choices"][0]["delta"]["content"], "При")
        error = json.loads(events[-2])
        self.assertEqual(error["error"]["code"], "stream_error")
        for secret in ("Traceback", "RuntimeError", "sk-live-9", "секрет"):
            self.assertNotIn(secret.encode("utf-8"), raw)


class TestChatStreamFallback(ServerCase):
    """Роутер без ``stream``: отвечаем одним чанком, без имитации потока."""

    def test_router_without_stream_sends_single_chunk(self):
        status, headers, raw, _ = self.post_chat(
            {"stream": True, "messages": [{"role": "user", "content": "Привет"}]}
        )

        self.assertEqual(status, 200)
        self.assertIn("text/event-stream", headers["Content-Type"])
        chunks, done = parse_sse_json(raw)
        self.assertTrue(done)
        self.assertEqual(len(chunks), 1)
        choice = chunks[0]["choices"][0]
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertEqual(
            choice["delta"],
            {"role": "assistant", "content": "Здравствуйте! Чем помочь?"},
        )
        self.assertEqual(chunks[0]["eva"]["route"], "small")
        self.assertEqual(len(self.router.calls), 1)

    def test_fallback_usage_chunk_when_requested(self):
        status, _, raw, _ = self.post_chat(
            {
                "stream": True,
                "stream_options": {"include_usage": True},
                "messages": [{"role": "user", "content": "Привет"}],
            }
        )

        self.assertEqual(status, 200)
        chunks, done = parse_sse_json(raw)
        self.assertTrue(done)
        content_chunks = [c for c in chunks if c["choices"]]
        usage_chunks = [c for c in chunks if not c["choices"]]
        self.assertEqual(len(content_chunks), 1)
        self.assertEqual(len(usage_chunks), 1)
        self.assertEqual(usage_chunks[0]["usage"]["total_tokens"], 17)

    def test_fallback_router_error_is_json(self):
        self.router.error = RuntimeError("boom")

        with self.assertLogs("eva.server", level="ERROR"):
            status, headers, raw, body = self.post_chat(
                {"stream": True, "messages": [{"role": "user", "content": "Привет"}]}
            )

        self.assertEqual(status, 500)
        self.assertIn("application/json", headers["Content-Type"])
        self.assertEqual(body["error"]["code"], "internal_error")
        self.assertNotIn(b"data:", raw)


class TestStreamRouterInjection(unittest.TestCase):
    def test_plain_router_has_no_stream(self):
        self.assertIsNone(server.as_stream_router(StubRouter()))
        self.assertIsNone(server.as_stream_router(server.default_router))
        self.assertIsNone(server.as_stream_router(None))

    def test_object_with_stream_is_detected(self):
        router = StreamingStubRouter()

        detected = server.as_stream_router(router)

        self.assertIsNotNone(detected)
        self.assertEqual(detected, router.stream)

    def test_function_may_carry_stream_attribute(self):
        def router(request):
            return StubAnswer()

        self.assertIsNone(server.as_stream_router(router))
        router.stream = lambda request: iter(["кусок"])
        self.assertIsNotNone(server.as_stream_router(router))


class TestClientErrors(ServerCase):
    def test_broken_json_is_400_without_traceback(self):
        status, _, raw, body = self.post_chat(None, raw=b'{"messages": [')

        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["code"], "invalid_json")
        self.assertIn("JSON", body["error"]["message"])
        self.assertNotIn(b"Traceback", raw)
        self.assertNotIn(b"JSONDecodeError", raw)

    def test_server_survives_broken_json(self):
        self.post_chat(None, raw=b"not json")

        status, _, _, body = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ok")

    def test_empty_body_is_400(self):
        status, _, _, body = self.request(
            "POST", "/v1/chat/completions", raw=b"", headers={"Content-Length": "0"}
        )

        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["code"], "empty_body")

    def test_json_array_is_400(self):
        status, _, _, body = self.request(
            "POST", "/v1/chat/completions", raw=b"[1, 2, 3]"
        )

        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["code"], "invalid_body")

    def test_missing_messages_is_400(self):
        status, _, _, body = self.post_chat({"model": "eva-hybrid"})

        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["code"], "invalid_messages")
        self.assertIn("messages", body["error"]["message"])

    def test_unknown_role_is_400(self):
        status, _, _, body = self.post_chat(
            {"messages": [{"role": "", "content": "Привет"}]}
        )

        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["code"], "invalid_messages")

    def test_unknown_path_is_404(self):
        status, _, _, body = self.request("GET", "/v1/nope")

        self.assertEqual(status, 404)
        self.assertEqual(body["error"]["code"], "not_found")

    def test_get_on_chat_is_405_with_allow(self):
        status, headers, _, body = self.request("GET", "/v1/chat/completions")

        self.assertEqual(status, 405)
        self.assertEqual(headers["Allow"], "POST")
        self.assertEqual(body["error"]["code"], "method_not_allowed")

    def test_options_lists_allowed_methods(self):
        status, headers, _, _ = self.request("OPTIONS", "/v1/chat/completions")

        self.assertEqual(status, 204)
        self.assertEqual(headers["Allow"], "POST")

    def test_post_on_health_is_405(self):
        status, headers, _, _ = self.request(
            "POST", "/health", raw=b"{}", headers={"Content-Type": "application/json"}
        )

        self.assertEqual(status, 405)
        self.assertEqual(headers["Allow"], "GET")

    def test_content_length_not_a_number_is_400(self):
        status, _, _, body = self.request(
            "POST",
            "/v1/chat/completions",
            headers={"Content-Length": "abc"},
        )

        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["code"], "invalid_content_length")

    def test_missing_content_length_is_411(self):
        raw = raw_http(
            self.port,
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: test\r\n\r\n",
        )

        status, body = parse_raw(raw)
        self.assertEqual(status, 411)
        self.assertEqual(json.loads(body)["error"]["code"], "length_required")

    def test_oversized_body_is_413(self):
        raw = raw_http(
            self.port,
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: test\r\n"
            b"Content-Length: 99999999\r\n\r\n",
        )

        status, body = parse_raw(raw)
        self.assertEqual(status, 413)
        self.assertEqual(json.loads(body)["error"]["code"], "payload_too_large")


class TestRouterFailures(ServerCase):
    def test_router_exception_is_500_with_human_text(self):
        self.router.error = RuntimeError("секретный ключ sk-live-123")

        with self.assertLogs("eva.server", level="ERROR") as logs:
            status, _, raw, body = self.post_chat(
                {"messages": [{"role": "user", "content": "Привет"}]}
            )

        self.assertTrue(any(record.exc_info for record in logs.records))
        self.assertEqual(status, 500)
        self.assertEqual(body["error"]["type"], "server_error")
        self.assertEqual(body["error"]["code"], "internal_error")
        self.assertIn("Внутренняя ошибка", body["error"]["message"])
        for secret in ("Traceback", "RuntimeError", "sk-live-123", "секретный"):
            self.assertNotIn(secret.encode("utf-8"), raw)

    def test_server_stays_alive_after_router_exception(self):
        self.router.error = ZeroDivisionError("boom")
        with self.assertLogs("eva.server", level="ERROR"):
            self.post_chat({"messages": [{"role": "user", "content": "Привет"}]})

        status, _, _, body = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ok")

    def test_garbage_answer_is_500(self):
        self.router.answer = object()

        with self.assertLogs("eva.server", level="ERROR"):
            status, _, _, body = self.post_chat(
                {"messages": [{"role": "user", "content": "Привет"}]}
            )

        self.assertEqual(status, 500)
        self.assertEqual(body["error"]["type"], "server_error")


class TestDefaultRouterStartup(unittest.TestCase):
    """Боевой маршрутизатор собирается лениво: /health не зависит от моделей."""

    def test_server_with_default_router_answers_health(self):
        httpd = server.create_server(host=HOST, port=0)
        port = httpd.server_address[1]
        thread = threading.Thread(
            target=httpd.serve_forever,
            kwargs={"poll_interval": 0.01},
            daemon=True,
        )
        thread.start()
        try:
            connection = http.client.HTTPConnection(HOST, port, timeout=5)
            try:
                connection.request("GET", "/health")
                response = connection.getresponse()
                body = json.loads(response.read())
            finally:
                connection.close()

            self.assertEqual(response.status, 200)
            self.assertEqual(body["status"], "ok")
            self.assertEqual(body["router"], "default")
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())


class TestRouterInjection(unittest.TestCase):
    def test_none_gives_default_router(self):
        self.assertIs(server.as_router(None), server.default_router)

    def test_rejects_non_callable(self):
        with self.assertRaises(TypeError):
            server.as_router(object())

    def test_accepts_object_with_answer_method(self):
        class Router:
            def __init__(self):
                self.seen = []

            def answer(self, request):
                self.seen.append(request.text)
                return StubAnswer(text="ok")

        class FakeRequest:
            text = "Привет"

        router = Router()
        callable_router = server.as_router(router)

        self.assertEqual(callable_router(FakeRequest()), StubAnswer(text="ok"))
        self.assertEqual(router.seen, ["Привет"])


if __name__ == "__main__":
    unittest.main()
