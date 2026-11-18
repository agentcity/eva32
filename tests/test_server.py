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


class ServerCase(unittest.TestCase):
    """Общий стенд: настоящий сервер на локальном эфемерном порту."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.router = StubRouter()
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
        self.router.calls.clear()
        self.router.answer = StubAnswer()
        self.router.error = None

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
            parsed = json.loads(data) if data else None
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

    def test_stream_is_rejected_with_human_message(self):
        status, _, _, body = self.post_chat(
            {"stream": True, "messages": [{"role": "user", "content": "Привет"}]}
        )

        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["code"], "unsupported_stream")
        self.assertIn("stream", body["error"]["message"])

    def test_models_endpoint_lists_default_model(self):
        status, _, _, body = self.request("GET", "/v1/models")

        self.assertEqual(status, 200)
        self.assertEqual(body["object"], "list")
        self.assertEqual(body["data"][0]["id"], server.DEFAULT_MODEL)


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
