"""Проверки веб-стенда сравнения двух моделей (задача 0138).

Запуск: ``python3 -m unittest tests.test_chat_server -v``

Тесты поднимают настоящий HTTP-сервер на свободном порту loopback, но
бэкенды подменены заглушками: ни ВТМ, ни Ollama, ни сеть не нужны.
Проверяется поведение снаружи (по HTTP), а не внутренние функции.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import chat_server  # noqa: E402

HOST = "127.0.0.1"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REGISTRY = (
    chat_server.BackendSpec(
        id="vtm",
        label="ВТМ (локальная)",
        kind="vtm",
        models=(chat_server.ModelSpec(id="mock-vtm", label="Мок ВТМ"),),
    ),
    chat_server.BackendSpec(
        id="ollama",
        label="Ollama (Qwen)",
        kind="ollama",
        models=(chat_server.ModelSpec(id="mock-qwen", label="Мок Qwen"),),
    ),
)


class MockAdapter:
    """Заглушка бэкенда: куски текста, usage или ошибка."""

    def __init__(
        self,
        pieces=("Ответ",),
        usage=None,
        error=None,
        delay=0.0,
    ) -> None:
        self.pieces = list(pieces)
        self.usage = (
            dict(usage)
            if usage is not None
            else {
                "prompt_tokens": 5,
                "completion_tokens": len(self.pieces),
                "total_tokens": 5 + len(self.pieces),
            }
        )
        self.error = error
        self.delay = float(delay)
        self.calls = []

    def stream(self, messages, *, max_tokens):
        self.calls.append(
            {
                "messages": [dict(message) for message in messages],
                "max_tokens": max_tokens,
            }
        )
        if self.error is not None:
            raise RuntimeError(self.error)
        for text in self.pieces:
            if self.delay:
                time.sleep(self.delay)
            yield chat_server.StreamPiece(text=text)
        yield chat_server.StreamPiece(done=True, usage=dict(self.usage))


def make_stream_payload(question="Что нового?", use_system=True, sides=None):
    if sides is None:
        sides = [
            {"backend": "vtm", "model": "mock-vtm"},
            {"backend": "ollama", "model": "mock-qwen"},
        ]
    return {
        "question": question,
        "use_system": use_system,
        "sides": sides,
    }


def parse_sse(raw):
    """Разобрать SSE: список JSON-событий и флаг терминального ``[DONE]``."""
    text = raw.decode("utf-8")
    events = []
    done = False
    for block in text.split("\n\n"):
        for line in block.splitlines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload:
                continue
            if payload == "[DONE]":
                done = True
                continue
            events.append(json.loads(payload))
    return events, done


class ServerTestCase(unittest.TestCase):
    """Общий каркас: сервер на свободном порту и два мок-бэкенда."""

    system_prompt = "СИСТЕМА: отвечай коротко."

    def setUp(self) -> None:
        self.vtm = MockAdapter(pieces=("При", "вет"))
        self.ollama = MockAdapter(pieces=("Здравствуйте", "!"))
        self.server = chat_server.create_server(
            port=0,
            registry=REGISTRY,
            adapters={
                ("vtm", "mock-vtm"): self.vtm,
                ("ollama", "mock-qwen"): self.ollama,
            },
            system_prompt=self.system_prompt,
            max_tokens=32,
        )
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            kwargs={"poll_interval": 0.01},
            daemon=True,
        )
        self.thread.start()
        self.port = self.server.server_address[1]
        self.addCleanup(self._stop_server)

    def _stop_server(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5.0)

    def request(self, method, path, body=None, headers=None, timeout=10):
        conn = http.client.HTTPConnection(HOST, self.port, timeout=timeout)
        try:
            conn.request(method, path, body=body, headers=headers or {})
            response = conn.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            conn.close()

    def post_chat(self, payload, timeout=10):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return self.request(
            "POST",
            "/api/chat",
            body=body,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )

    def post_chat_ok(self, payload=None):
        status, headers, raw = self.post_chat(payload or make_stream_payload())
        self.assertEqual(status, 200, raw.decode("utf-8", "replace"))
        self.assertIn("text/event-stream", headers.get("Content-Type", ""))
        events, done = parse_sse(raw)
        self.assertTrue(done, "поток не завершился событием [DONE]")
        return events, headers, raw


class TestPage(ServerTestCase):
    def test_page_is_served(self):
        status, headers, body = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", headers.get("Content-Type", ""))
        page = body.decode("utf-8")
        self.assertIn("<title>EVA32", page)
        self.assertIn('id="question"', page)
        self.assertIn('id="answer-0"', page)
        self.assertIn('id="answer-1"', page)
        self.assertIn("/api/chat", page)
        self.assertIn("/api/backends", page)

    def test_page_has_no_external_references(self):
        status, _, body = self.request("GET", "/")
        self.assertEqual(status, 200)
        page = body.decode("utf-8")
        self.assertNotRegex(page, r"https?://")
        self.assertNotIn("<link", page.lower())
        self.assertNotRegex(page, r"<script[^>]*\bsrc\s*=")
        self.assertNotIn("@import", page)
        self.assertNotRegex(page, r"url\s*\(")
        self.assertNotIn("<iframe", page.lower())
        # Ровно один style и один script — значит, ничего не подгружается.
        self.assertEqual(page.count("<style"), 1)
        self.assertEqual(page.count("<script"), 1)

    def test_page_sets_restrictive_csp(self):
        status, headers, _ = self.request("GET", "/")
        self.assertEqual(status, 200)
        csp = headers.get("Content-Security-Policy", "")
        self.assertIn("default-src 'none'", csp)
        self.assertIn("connect-src 'self'", csp)
        self.assertNotIn("http", csp)

    def test_favicon_and_health(self):
        status, _, body = self.request("GET", "/favicon.ico")
        self.assertEqual(status, 204)
        self.assertEqual(body, b"")

        status, headers, body = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {"status": "ok"})
        self.assertIn("application/json", headers.get("Content-Type", ""))


class TestCatalog(ServerTestCase):
    def test_catalog_lists_only_registered_backends(self):
        status, headers, body = self.request("GET", "/api/backends")
        self.assertEqual(status, 200)
        self.assertIn("application/json", headers.get("Content-Type", ""))
        data = json.loads(body)
        self.assertEqual([item["id"] for item in data["backends"]], ["vtm", "ollama"])
        self.assertEqual(data["defaults"]["left"]["backend"], "vtm")
        self.assertEqual(data["defaults"]["right"]["backend"], "ollama")
        self.assertEqual(data["defaults"]["left"]["model"], "mock-vtm")
        self.assertEqual(data["defaults"]["right"]["model"], "mock-qwen")
        # Наружу не уходят ни URL, ни пути, ни токенизаторы.
        raw = body.decode("utf-8").lower()
        self.assertNotIn("http", raw)
        self.assertNotIn("127.0.0.1", raw)
        self.assertNotIn("/users/", raw)


class TestChatStream(ServerTestCase):
    def test_two_streams_side_by_side(self):
        events, headers, raw = self.post_chat_ok()
        self.assertIn("Cache-Control", headers)
        self.assertTrue(raw.rstrip().endswith(b"data: [DONE]"))

        starts = [event for event in events if event["type"] == "start"]
        self.assertEqual({event["side"] for event in starts}, {0, 1})
        self.assertEqual(starts[0]["backend"], "vtm")
        self.assertEqual(starts[0]["model"], "mock-vtm")
        self.assertEqual(starts[1]["backend"], "ollama")
        self.assertEqual(starts[1]["model"], "mock-qwen")

        texts = {0: "", 1: ""}
        for event in events:
            if event["type"] == "delta":
                texts[event["side"]] += event["text"]
        self.assertEqual(texts[0], "Привет")
        self.assertEqual(texts[1], "Здравствуйте!")

        done = [event for event in events if event["type"] == "done"]
        self.assertEqual(len(done), 2)
        for event in done:
            self.assertGreaterEqual(event["elapsed_ms"], 1)
            self.assertEqual(event["completion_tokens"], 2)
            self.assertGreater(event["tokens_per_second"], 0)
            self.assertIsInstance(event["first_token_ms"], int)
            self.assertFalse(event["estimated"])

        end = [event for event in events if event["type"] == "end"]
        self.assertEqual(len(end), 1)

    def test_same_question_and_system_prompt_go_to_both(self):
        self.post_chat_ok(make_stream_payload(question="Сравни два ответа"))
        self.assertEqual(len(self.vtm.calls), 1)
        self.assertEqual(len(self.ollama.calls), 1)
        for adapter in (self.vtm, self.ollama):
            call = adapter.calls[0]
            self.assertEqual(call["max_tokens"], 32)
            self.assertEqual(
                call["messages"][0],
                {"role": "system", "content": self.system_prompt},
            )
            self.assertEqual(call["messages"][-1]["role"], "user")
            self.assertEqual(
                call["messages"][-1]["content"], "Сравни два ответа"
            )

    def test_system_prompt_can_be_disabled(self):
        self.post_chat_ok(make_stream_payload(use_system=False))
        for adapter in (self.vtm, self.ollama):
            messages = adapter.calls[0]["messages"]
            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[0]["role"], "user")

    def test_one_backend_error_does_not_break_the_other(self):
        self.vtm.error = "модель упала на первом токене"
        events, _, raw = self.post_chat_ok()

        errors = [event for event in events if event["type"] == "error"]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["side"], 0)
        self.assertIn("модель упала", errors[0]["message"])
        self.assertNotIn("Traceback", errors[0]["message"])

        deltas = [event for event in events if event["type"] == "delta"]
        self.assertTrue(any(event["side"] == 1 for event in deltas))
        done = [event for event in events if event["type"] == "done"]
        self.assertEqual([event["side"] for event in done], [1])
        self.assertTrue(raw.rstrip().endswith(b"data: [DONE]"))

    def test_missing_usage_is_marked_as_estimate(self):
        self.vtm.usage = {}
        events, _, _ = self.post_chat_ok()
        done = [
            event
            for event in events
            if event["type"] == "done" and event["side"] == 0
        ]
        self.assertEqual(len(done), 1)
        self.assertTrue(done[0]["estimated"])
        self.assertGreater(done[0]["completion_tokens"], 0)

    def test_stream_is_actually_parallel(self):
        self.vtm.delay = 0.25
        self.vtm.pieces = ("а", "б", "в", "г")
        self.ollama.delay = 0.25
        self.ollama.pieces = ("а", "б", "в", "г")
        started = time.monotonic()
        self.post_chat_ok()
        elapsed = time.monotonic() - started
        # Последовательно было бы ~2 с (8 кусков по 0.25), параллельно — ~1 с.
        self.assertLess(elapsed, 1.8)


class TestValidation(ServerTestCase):
    def assert_client_error(self, response, status=400):
        code, headers, body = response
        self.assertEqual(code, status, body.decode("utf-8", "replace"))
        self.assertIn("application/json", headers.get("Content-Type", ""))
        data = json.loads(body)
        self.assertIn("error", data)
        self.assertTrue(data["error"]["message"])
        return data["error"]

    def test_backend_must_be_in_closed_list(self):
        payload = make_stream_payload(
            sides=[
                {"backend": "http://evil.example", "model": "mock-vtm"},
                {"backend": "ollama", "model": "mock-qwen"},
            ]
        )
        error = self.assert_client_error(self.post_chat(payload))
        self.assertIn("неизвестный бэкенд", error["message"])

    def test_model_must_be_in_closed_list(self):
        payload = make_stream_payload(
            sides=[
                {"backend": "vtm", "model": "../../etc/passwd"},
                {"backend": "ollama", "model": "mock-qwen"},
            ]
        )
        error = self.assert_client_error(self.post_chat(payload))
        self.assertIn("неизвестная модель", error["message"])

    def test_two_sides_required(self):
        error = self.assert_client_error(
            self.post_chat(
                make_stream_payload(
                    sides=[{"backend": "vtm", "model": "mock-vtm"}]
                )
            )
        )
        self.assertIn("ровно", error["message"])

    def test_empty_question_rejected(self):
        self.assert_client_error(self.post_chat(make_stream_payload(question="   ")))

    def test_question_too_large(self):
        error = self.assert_client_error(
            self.post_chat(make_stream_payload(question="я" * 9000)), status=413
        )
        self.assertEqual(error["code"], "question_too_large")

    def test_bad_max_tokens(self):
        payload = make_stream_payload()
        payload["max_tokens"] = 0
        self.assert_client_error(self.post_chat(payload))

    def test_bad_json(self):
        status, headers, body = self.request(
            "POST",
            "/api/chat",
            body=b"{not json",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 400)
        self.assertIn("invalid_json", body.decode("utf-8"))

    def test_unknown_paths(self):
        self.assertEqual(self.request("GET", "/api/nope")[0], 404)
        self.assertEqual(self.request("GET", "/nope")[0], 404)
        status, _, body = self.request(
            "POST",
            "/api/nope",
            body=b"{}",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 404)
        self.assertIn("not_found", body.decode("utf-8"))

    def test_only_loopback_can_be_bound(self):
        with self.assertRaises(ValueError):
            chat_server.create_server(
                host="0.0.0.0",
                registry=REGISTRY,
                adapters={},
                system_prompt="",
            )
        with self.assertRaises(ValueError):
            chat_server.create_server(
                host="192.168.1.10",
                registry=REGISTRY,
                adapters={},
                system_prompt="",
            )


if __name__ == "__main__":
    unittest.main()
