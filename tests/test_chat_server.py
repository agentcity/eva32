"""Проверки веб-стенда сравнения двух моделей (задача 0138) и панели
метрик под чатом (задача 0145).

Запуск: ``python3 -m unittest tests.test_chat_server -v``

Тесты поднимают настоящий HTTP-сервер на свободном порту loopback, но
бэкенды подменены заглушками: ни ВТМ, ни Ollama, ни сеть не нужны.
Проверяется поведение снаружи (по HTTP), а не внутренние функции.

Для карточек моделей тесты собирают синтетический контейнер ``.vtm``
(заголовок + таблица тензоров + JSON-метаданные): так числа панели можно
сверить с тем, что тест сам в файл положил, не имея настоящей модели.
"""

from __future__ import annotations

import http.client
import json
import os
import pathlib
import re
import shutil
import struct
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace

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
        finish_reason="stop",
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
        self.finish_reason = finish_reason
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
        yield chat_server.StreamPiece(
            done=True,
            usage=dict(self.usage),
            finish_reason=self.finish_reason,
        )


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


# --------------------------------------------------------------------------- #
# Синтетический контейнер .vtm: числа в тесте = числа в файле
# --------------------------------------------------------------------------- #

def _align32(value):
    return (value + 31) & ~31


def _trit_segment(n_trits, *, native=False, scale_bytes=4):
    """Сегмент TRT1: 18-байтный заголовок, масштабы, слова (packed20/native)."""
    n_scale = (n_trits + 127) // 128
    flags = 1 if scale_bytes == 4 else 0
    if native:
        words = (n_trits + 3) // 4
        flags |= 1 << 1
    else:
        words = (n_trits + 159) // 160 * 8
    return (
        struct.pack("<4sBBQI", b"TRT1", 2, flags, n_trits, n_scale)
        + b"\0" * (n_scale * scale_bytes)
        + b"\0" * (words * 4)
    )


def build_vtm(
    path,
    *,
    vocab=8,
    hidden=4,
    layers=2,
    context=16,
    meta=None,
    kv_in_file=False,
    native_trits=False,
):
    """Собрать валидный контейнер VTM v2 с предсказуемыми числами.

    Тензоры: ``emb.w`` (trit), по слою ``q_proj.w`` и ``k_proj.w`` (trit),
    ``kcache``/``vcache`` (fp32, KV) и ``final_norm.w`` (fp32). KV-кэш по
    умолчанию вне файла (флаг + ``VTM_NO_OFFSET``), как в новых контейнерах.
    """

    def trit_segment(elements):
        return _trit_segment(elements, native=native_trits)

    tensors = []  # (name, kind, dims, payload, flags, scale_offset)

    def add(name, kind, dims, payload, flags=0, scale_offset=0):
        tensors.append((name, kind, dims, payload, flags, scale_offset))

    kv_dim = hidden // 2
    add(
        "emb.w", 0, (vocab, hidden), trit_segment(vocab * hidden),
        scale_offset=chat_server.VTM_NO_OFFSET,
    )
    for layer in range(layers):
        add(
            "l{}.q_proj.w".format(layer), 0, (hidden, hidden),
            trit_segment(hidden * hidden), scale_offset=chat_server.VTM_NO_OFFSET,
        )
        add(
            "l{}.k_proj.w".format(layer), 0, (kv_dim, hidden),
            trit_segment(kv_dim * hidden), scale_offset=chat_server.VTM_NO_OFFSET,
        )
        for name in ("kcache", "vcache"):
            dims = (context, kv_dim)
            if kv_in_file:
                add(
                    "l{}.{}".format(layer, name), 1, dims,
                    b"\0" * (context * kv_dim * 4),
                )
            else:
                add(
                    "l{}.{}".format(layer, name), 1, dims, b"",
                    flags=chat_server.VTM_TENSOR_FLAG_KV_CACHE,
                )
    add("final_norm.w", 1, (hidden,), b"\0" * (hidden * 4))

    table_size = chat_server._VTM_TENSOR.size * len(tensors)
    data_start = _align32(chat_server._VTM_HEADER.size + table_size)
    records = bytearray()
    data = bytearray()
    for name, kind, dims, payload, flags, scale_offset in tensors:
        out_of_file = bool(flags & chat_server.VTM_TENSOR_FLAG_KV_CACHE) and not kv_in_file
        if out_of_file:
            offset = chat_server.VTM_NO_OFFSET
        else:
            offset = _align32(len(data))
            data += payload
            data += b"\0" * (_align32(len(data)) - len(data))
        padded = tuple(dims) + (0,) * (4 - len(dims))
        records += chat_server._VTM_TENSOR.pack(
            name.encode("utf-8"), kind, len(dims), *padded,
            offset, scale_offset, flags,
        )

    meta_bytes = b"" if meta is None else json.dumps(meta).encode("utf-8")
    meta_offset = data_start + len(data)
    body = bytes(records)
    body += b"\0" * (data_start - chat_server._VTM_HEADER.size - len(body))
    body += bytes(data)
    body += meta_bytes
    body += b"\0" * (_align32(len(meta_bytes)) - len(meta_bytes))
    header = chat_server._VTM_HEADER.pack(
        b"VTM1", 2, 0, 0, len(tensors), 0, 0, meta_offset, len(meta_bytes),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + body)
    return {
        "params": vocab * hidden
        + layers * (hidden * hidden + kv_dim * hidden)
        + hidden,
        "kv_bytes": layers * 2 * context * kv_dim * 4,
        "context": context,
        "layers": layers,
        "vocab": vocab,
        "size_bytes": path.stat().st_size,
    }


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


# --------------------------------------------------------------------------- #
# Метрики (задача 0145): карточки моделей и системный снимок
# --------------------------------------------------------------------------- #

class VtmLikeAdapter(MockAdapter):
    """Заглушка с признаками ВТМ: модель уже в памяти, кэш префикса включён."""

    kind = "vtm"
    cache_prefix = True
    loaded = True


class OllamaProbeAdapter(MockAdapter):
    """Заглушка Ollama: отдаёт ``/api/ps`` из подставной функции, без сети."""

    kind = "ollama"
    loaded = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.probes = []

    def probe_ps(self, model):
        self.probes.append(model)
        return {"model": model, "size": 1234, "size_vram": 1024}


class FakeModel:
    """Внутренний бэкенд для ChatCliAdapter: один текст и терминальный кусок."""

    cache_prefix = True

    def stream(self, messages, *, max_tokens):
        yield SimpleNamespace(text="ок")
        yield SimpleNamespace(
            done=True,
            finish_reason="stop",
            usage={"prompt_tokens": 2, "completion_tokens": 1},
        )


def _greedy_factory(blob_holder, megabytes=12):
    """Фабрика «модели»: выделяет и трогает память, чтобы RSS вырос честно."""

    def factory():
        blob = bytearray(megabytes * 1024 * 1024)
        for offset in range(0, len(blob), 4096):
            blob[offset] = 7
        blob_holder.append(blob)
        return FakeModel()

    return factory


def start_server(registry, adapters, *, max_tokens=32, system_prompt="СИСТЕМА"):
    """Поднять сервер на свободном порту loopback; вернуть ``(server, thread)``."""
    server = chat_server.create_server(
        port=0,
        registry=registry,
        adapters=adapters,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
    )
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    return server, thread


def stop_server(server, thread):
    server.shutdown()
    server.server_close()
    thread.join(timeout=5.0)


def http_json(port, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection(HOST, port, timeout=10)
    try:
        conn.request(method, path, body=body, headers=headers or {})
        response = conn.getresponse()
        return response.status, response.read()
    finally:
        conn.close()


class CardServerTestCase(unittest.TestCase):
    """Сервер с синтетическим ``.vtm``: GET-эндпоинты, без сети и без модели."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="eva32-chat-")
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        self.server = None
        self.thread = None

    def serve(self, registry, adapters=None):
        self.server, self.thread = start_server(
            registry, adapters if adapters is not None else {}, system_prompt=""
        )
        self.addCleanup(stop_server, self.server, self.thread)
        return self.server.server_address[1]

    def get_json(self, path):
        port = self.server.server_address[1]
        status, body = http_json(port, "GET", path)
        self.assertEqual(status, 200, body.decode("utf-8", "replace"))
        return json.loads(body), body

    def registry_for(self, file_name, **build_kwargs):
        path = pathlib.Path(self.tmpdir) / file_name
        expected = build_vtm(path, **build_kwargs)
        registry = (
            chat_server.BackendSpec(
                id="vtm",
                label="ВТМ (локальная)",
                kind="vtm",
                models=(
                    chat_server.ModelSpec(
                        id=file_name, label="Синтетика", target=str(path)
                    ),
                ),
            ),
        )
        return registry, path, expected

    def card_for(self, file_name, **build_kwargs):
        registry, path, expected = self.registry_for(file_name, **build_kwargs)
        self.serve(registry)
        data, raw = self.get_json("/api/models")
        cards = [card for card in data["models"] if card["model"] == file_name]
        self.assertEqual(len(cards), 1, data)
        return cards[0], path, expected, raw


class TestModelCards(CardServerTestCase):
    def test_card_numbers_match_file_and_tensor_table(self):
        card, path, expected, raw = self.card_for(
            "model.vtm", vocab=8, hidden=4, layers=2, context=16
        )
        self.assertEqual(card["disk_bytes"], os.path.getsize(path))
        self.assertEqual(card["disk_bytes"], expected["size_bytes"])
        self.assertEqual(card["params"], expected["params"])
        self.assertEqual(card["kv_bytes"], expected["kv_bytes"])
        self.assertEqual(card["context"], expected["context"])
        self.assertEqual(card["layers"], expected["layers"])
        self.assertEqual(card["vocab"], expected["vocab"])
        self.assertFalse(card["kv_in_file"])
        self.assertTrue(card["present"])
        self.assertEqual(card["tensors"], 10)
        self.assertEqual(card["format_version"], 2)
        self.assertIsInstance(card["bits_per_param"], float)
        self.assertGreater(card["bits_per_param"], 0.0)
        self.assertLess(card["bits_per_param"], 32.0)
        self.assertIn("trit", card["storage"])
        self.assertIn("эмбеддинг: trit", card["storage"])
        # Метаданных нет — честное «неизвестно», а не придуманный источник.
        self.assertEqual(card["source"], "неизвестно (метаданных нет)")
        self.assertIsNone(card["last"])
        self.assertIsNone(card["load"])
        self.assertIsNone(card["loaded"])

    def test_kv_cache_in_file_is_reported(self):
        card, _, expected, _ = self.card_for(
            "kv-in-file.vtm", kv_in_file=True
        )
        self.assertTrue(card["kv_in_file"])
        self.assertEqual(card["kv_bytes"], expected["kv_bytes"])

    def test_meta_gives_context_and_context_len(self):
        meta = {
            "model": {"n_layers": 2, "context_len": 64, "vocab": 8},
            "n_layers": 2,
            "context_len": 64,
            "vocab_size": 8,
            "quantization": {"scale_dtype": "float16"},
        }
        card, _, _, _ = self.card_for(
            "meta.vtm", vocab=8, hidden=4, layers=2, context=16, meta=meta
        )
        self.assertEqual(card["context"], 64)
        self.assertEqual(card["layers"], 2)
        self.assertEqual(card["vocab"], 8)
        self.assertIn("float16", card["storage"])

    def test_source_cold_export_from_meta(self):
        meta = {
            "kv_cache": {"bytes": 4096},
            "base_model": {"dir": "/models/qwen", "name_or_path": "Qwen/Qwen2.5-0.5B"},
            "checkpoint": None,
            "generator": "tools/vtm_compile.py",
        }
        card, _, _, _ = self.card_for("cold.vtm", meta=meta)
        self.assertEqual(card["source"], "холодный экспорт")

    def test_source_qat_from_meta(self):
        meta = {
            "base_model": {"name_or_path": "tiny-qwen2-qat"},
            "step": 500,
            "quantize_embeddings": "trit",
        }
        card, _, _, _ = self.card_for("qat.vtm", meta=meta)
        self.assertEqual(card["source"], "QAT-экспорт")

    def test_demo_heuristic_is_vocab_128(self):
        card, _, _, _ = self.card_for("demo.vtm", vocab=128, hidden=8)
        self.assertEqual(card["vocab"], 128)
        self.assertTrue(card["source"].startswith("демо"), card["source"])

    def test_missing_file_card_has_nulls(self):
        registry = (
            chat_server.BackendSpec(
                id="vtm",
                label="ВТМ",
                kind="vtm",
                models=(
                    chat_server.ModelSpec(
                        id="missing.vtm",
                        label="Нет файла",
                        target=os.path.join(self.tmpdir, "nope.vtm"),
                    ),
                ),
            ),
        )
        self.serve(registry)
        data, _ = self.get_json("/api/models")
        card = data["models"][0]
        self.assertFalse(card["present"])
        self.assertIsNone(card["disk_bytes"])
        self.assertIsNone(card["params"])
        self.assertIsNone(card["source"])
        self.assertIsNone(card["bits_per_param"])

    def test_foreign_file_reports_stat_only(self):
        path = pathlib.Path(self.tmpdir) / "not-vtm.vtm"
        path.write_bytes(b"nope")
        registry = (
            chat_server.BackendSpec(
                id="vtm",
                label="ВТМ",
                kind="vtm",
                models=(
                    chat_server.ModelSpec(
                        id="not-vtm.vtm", label="Мусор", target=str(path)
                    ),
                ),
            ),
        )
        self.serve(registry)
        data, _ = self.get_json("/api/models")
        card = data["models"][0]
        self.assertTrue(card["present"])
        self.assertEqual(card["disk_bytes"], 4)
        self.assertIsNone(card["params"])
        self.assertIsNone(card["storage"])

    def test_cards_do_not_leak_absolute_paths(self):
        card, path, _, raw = self.card_for("model.vtm")
        self.assertEqual(card["model"], "model.vtm")
        self.assertNotIn(str(path).encode(), raw)
        self.assertNotIn(self.tmpdir.encode(), raw)
        self.assertNotIn(b"/Users/", raw)

    def test_inspect_is_cached_until_file_changes(self):
        path = pathlib.Path(self.tmpdir) / "cache.vtm"
        build_vtm(path, vocab=8, hidden=4, layers=2, context=16)
        first = chat_server.inspect_vtm(path)
        self.assertIsNotNone(first)
        second = chat_server.inspect_vtm(path)
        self.assertIs(first, second)
        # Тот же файл с новым mtime — новый разбор (кэш не врёт).
        os.utime(path, (time.time() + 1, time.time() + 1))
        third = chat_server.inspect_vtm(path)
        self.assertIsNotNone(third)
        self.assertIsNot(third, first)
        self.assertEqual(third.size_bytes, os.path.getsize(path))
        self.assertEqual(third.params, first.params)


class TestSystemEndpoint(CardServerTestCase):
    def test_system_snapshot_matches_sources(self):
        self.serve(REGISTRY)
        snapshot, raw = self.get_json("/api/system")
        self.assertIn("load", snapshot)
        self.assertIsInstance(snapshot["cpu_count"], int)
        self.assertGreaterEqual(snapshot["cpu_count"], 1)
        disk = snapshot["disk"]
        self.assertIsNotNone(disk)
        usage = shutil.disk_usage(str(chat_server.ROOT))
        self.assertEqual(disk["free"], usage.free)
        self.assertEqual(disk["total"], usage.total)
        process = snapshot["process"]
        self.assertEqual(process["pid"], os.getpid())
        self.assertTrue(
            process["rss"] is None or process["rss"] > 0,
            "RSS процесса должен быть положительным или недоступным",
        )
        self.assertTrue(process["peak"] is None or process["peak"] > 0)
        load = snapshot["load"]
        self.assertTrue(load is None or len(load) == 3)
        ram = snapshot["ram"]
        self.assertTrue(ram["free"] is None or ram["free"] >= 0)
        self.assertTrue(ram["total"] is None or ram["total"] > 0)
        # Наружу не уходит путь к репозиторию — только числа.
        self.assertNotIn(str(chat_server.ROOT).encode(), raw)

    def test_vtm_threads_reflects_environment(self):
        previous = os.environ.get("VTM_THREADS")
        os.environ["VTM_THREADS"] = "3"
        self.addCleanup(
            lambda: os.environ.pop("VTM_THREADS", None)
            if previous is None
            else os.environ.__setitem__("VTM_THREADS", previous)
        )
        self.serve(REGISTRY)
        snapshot, _ = self.get_json("/api/system")
        self.assertEqual(snapshot["vtm_threads"]["runtime"], "3")
        self.assertEqual(snapshot["platform"], sys.platform)

    def test_ollama_is_only_local(self):
        self.assertEqual(chat_server.OLLAMA_BASE, "http://127.0.0.1:11434/v1")
        self.assertEqual(chat_server.OLLAMA_ROOT, "http://127.0.0.1:11434")


class TestAnswerMetrics(ServerTestCase):
    """Метрики ответа: TTFT/prefill/декод, CPU запроса, остановка, первый токен."""

    def test_done_event_has_timing_breakdown(self):
        events, _, _ = self.post_chat_ok()
        done = [event for event in events if event["type"] == "done"]
        self.assertEqual(len(done), 2)
        for event in done:
            self.assertEqual(event["prefill_ms"], event["first_token_ms"])
            self.assertGreaterEqual(event["decode_ms"], 0)
            self.assertEqual(
                event["decode_ms"],
                max(0, event["elapsed_ms"] - (event["first_token_ms"] or 0)),
            )
            self.assertGreater(event["decode_tokens_per_second"], 0)
            self.assertEqual(event["finish_reason"], "stop")
            # Заглушки не умеют «загружаться» — значит, и холодность неизвестна.
            self.assertIsNone(event["cold"])
            self.assertIsNone(event["first_token"])
            self.assertIsNone(event["cache_prefix"])
            self.assertIsNone(event["load"])

    def test_end_event_has_request_cpu_and_system(self):
        events, _, _ = self.post_chat_ok()
        end = [event for event in events if event["type"] == "end"]
        self.assertEqual(len(end), 1)
        metrics = end[0]["metrics"]
        self.assertGreaterEqual(metrics["wall_ms"], 1)
        self.assertGreaterEqual(metrics["cpu_ms"], 0.0)
        self.assertLess(metrics["cpu_ms"], metrics["wall_ms"] * 60 + 5000)
        self.assertIn("process", metrics)
        self.assertEqual(metrics["process"]["pid"], os.getpid())
        self.assertIn("start_memory", metrics)
        self.assertIn("vtm_threads", metrics)
        self.assertIn("disk", metrics)
        self.assertIn("ram", metrics)

    def test_finish_reason_length_is_reported(self):
        self.server.adapters[("vtm", "mock-vtm")].finish_reason = "length"
        events, _, _ = self.post_chat_ok()
        done = [
            event
            for event in events
            if event["type"] == "done" and event["side"] == 0
        ]
        self.assertEqual(done[0]["finish_reason"], "length")


class TestPrefixCacheMetrics(unittest.TestCase):
    """Кэш префикса ВТМ: настройка и оценка попадания (без сети)."""

    def setUp(self):
        self.vtm = VtmLikeAdapter(pieces=("а", "б"))
        self.ollama = MockAdapter(pieces=("в",))
        self.server, self.thread = start_server(
            REGISTRY,
            {("vtm", "mock-vtm"): self.vtm, ("ollama", "mock-qwen"): self.ollama},
        )
        self.addCleanup(stop_server, self.server, self.thread)
        self.port = self.server.server_address[1]

    def ask(self, question="один и тот же вопрос"):
        payload = json.dumps(
            {
                "question": question,
                "use_system": True,
                "sides": [
                    {"backend": "vtm", "model": "mock-vtm"},
                    {"backend": "ollama", "model": "mock-qwen"},
                ],
            },
            ensure_ascii=False,
        ).encode("utf-8")
        status, raw = http_json(
            self.port,
            "POST",
            "/api/chat",
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 200, raw.decode("utf-8", "replace"))
        events, _ = parse_sse(raw)
        return [
            event
            for event in events
            if event["type"] == "done" and event["side"] == 0
        ][0]

    def test_first_answer_has_no_prefix_history(self):
        done = self.ask()
        self.assertTrue(done["cache_prefix"]["enabled"])
        self.assertIsNone(done["cache_prefix"]["hit"])
        self.assertEqual(done["cache_prefix"]["shared_chars"], 0)
        self.assertTrue(done["cache_prefix"]["estimated"])

    def test_same_prompt_counts_as_hit(self):
        self.ask()
        done = self.ask()
        cache = done["cache_prefix"]
        self.assertTrue(cache["enabled"])
        self.assertTrue(cache["hit"], cache)
        self.assertGreater(cache["shared_chars"], 0)

    def test_different_prompt_is_a_miss(self):
        self.ask("первый вопрос про стрижку")
        done = self.ask("совсем другой вопрос про цену")
        cache = done["cache_prefix"]
        self.assertFalse(cache["hit"], cache)
        self.assertLess(cache["shared_chars"], cache["previous_chars"])

    def test_non_vtm_backend_has_no_cache_section(self):
        events, _, _ = self.post_chat_ok()
        done = [
            event
            for event in events
            if event["type"] == "done" and event["side"] == 1
        ]
        self.assertIsNone(done[0]["cache_prefix"])

    def post_chat_ok(self):
        payload = json.dumps(
            {
                "question": "вопрос",
                "use_system": True,
                "sides": [
                    {"backend": "vtm", "model": "mock-vtm"},
                    {"backend": "ollama", "model": "mock-qwen"},
                ],
            },
            ensure_ascii=False,
        ).encode("utf-8")
        status, raw = http_json(
            self.port,
            "POST",
            "/api/chat",
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 200)
        events, done = parse_sse(raw)
        self.assertTrue(done)
        return events, None, raw


class TestAdapterLoadMetrics(unittest.TestCase):
    """RSS-дельта вокруг загрузки: модель в этом процессе, дельта честная."""

    def test_rss_reader_returns_bytes(self):
        rss = chat_server._rss_bytes()
        if rss is None:
            self.skipTest("платформа не отдаёт RSS процесса")
        self.assertGreater(rss, 0)
        peak = chat_server._peak_rss_bytes()
        self.assertIsNotNone(peak)
        self.assertGreater(peak, 0)

    def test_load_delta_is_positive_and_cold_then_warm(self):
        holder = []
        # 64 МиБ — заметно больше всего, что тесты успевают выделить и
        # освободить раньше: дельта не может «схлопнуться» в 0 из-за
        # переиспользования уже резидентных страниц.
        adapter = chat_server.ChatCliAdapter(
            _greedy_factory(holder, megabytes=64), kind="vtm"
        )
        self.assertFalse(adapter.loaded)
        pieces = list(
            adapter.stream([{"role": "user", "content": "привет"}], max_tokens=4)
        )
        self.assertTrue(any(piece.text for piece in pieces))
        self.assertTrue(pieces[-1].done)
        self.assertEqual(pieces[-1].finish_reason, "stop")
        self.assertTrue(adapter.loaded)
        self.assertTrue(adapter.cache_prefix)
        stats = adapter.load_stats
        self.assertIsNotNone(stats)
        if stats["rss_before"] is None or stats["rss_after"] is None:
            self.skipTest("платформа не отдаёт RSS процесса")
        self.assertGreater(stats["rss_after"], stats["rss_before"])
        self.assertGreater(stats["rss_delta"], 0)
        self.assertGreaterEqual(stats["peak"], stats["rss_after"])
        self.assertGreaterEqual(stats["load_ms"], 0)
        # Второй вызов фабрику не зовёт: модель уже в памяти.
        list(adapter.stream([{"role": "user", "content": "ещё"}], max_tokens=4))
        self.assertEqual(len(holder), 1)


class TestColdWarmInStream(unittest.TestCase):
    """Событие done: первый токен холодный (модель грузилась) или тёплый."""

    def setUp(self):
        self.holder = []
        vtm_adapter = chat_server.ChatCliAdapter(
            _greedy_factory(self.holder, megabytes=4), kind="vtm"
        )
        self.server, self.thread = start_server(
            REGISTRY,
            {
                ("vtm", "mock-vtm"): vtm_adapter,
                ("ollama", "mock-qwen"): MockAdapter(pieces=("в",)),
            },
        )
        self.addCleanup(stop_server, self.server, self.thread)
        self.port = self.server.server_address[1]

    def ask(self):
        payload = json.dumps(
            {
                "question": "вопрос",
                "use_system": True,
                "sides": [
                    {"backend": "vtm", "model": "mock-vtm"},
                    {"backend": "ollama", "model": "mock-qwen"},
                ],
            },
            ensure_ascii=False,
        ).encode("utf-8")
        status, raw = http_json(
            self.port,
            "POST",
            "/api/chat",
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 200)
        events, _ = parse_sse(raw)
        return [
            event
            for event in events
            if event["type"] == "done" and event["side"] == 0
        ][0], events

    def test_first_answer_is_cold_second_is_warm(self):
        first, events = self.ask()
        self.assertTrue(first["cold"])
        self.assertEqual(first["first_token"], "холодный")
        load = first["load"]
        self.assertIsNotNone(load)
        # Дельта бывает и 0, и отрицательной: аллокатор может переиспользовать
        # резидентные страницы, а часть памяти освободиться во время загрузки.
        # Строгая проверка «> 0» — ниже, на отдельной большой аллокации.
        self.assertIsInstance(load["rss_before"], int)
        self.assertIsInstance(load["rss_after"], int)
        self.assertEqual(load["rss_delta"], load["rss_after"] - load["rss_before"])
        end = [event for event in events if event["type"] == "end"][0]
        self.assertGreater(end["metrics"]["process"]["rss"], 0)

        second, _ = self.ask()
        self.assertFalse(second["cold"])
        self.assertEqual(second["first_token"], "тёплый")

    def test_model_cards_show_loaded_model_and_load_delta(self):
        self.ask()
        status, raw = http_json(self.port, "GET", "/api/models")
        self.assertEqual(status, 200)
        data = json.loads(raw)
        card = [item for item in data["models"] if item["model"] == "mock-vtm"][0]
        self.assertTrue(card["loaded"])
        self.assertTrue(card["cache_prefix"])
        self.assertIsNotNone(card["load"])
        self.assertIsInstance(card["load"]["rss_before"], int)
        self.assertIsInstance(card["load"]["rss_after"], int)
        self.assertEqual(
            card["load"]["rss_delta"],
            card["load"]["rss_after"] - card["load"]["rss_before"],
        )
        self.assertIsNotNone(card["last"])
        self.assertEqual(card["last"]["first_token"], "холодный")


class TestOllamaProbe(unittest.TestCase):
    """Ollama: размер резидентной копии берётся из её ``/api/ps`` (заглушка)."""

    def setUp(self):
        self.ollama = OllamaProbeAdapter(pieces=("в",))
        self.server, self.thread = start_server(
            REGISTRY,
            {
                ("vtm", "mock-vtm"): MockAdapter(pieces=("а",)),
                ("ollama", "mock-qwen"): self.ollama,
            },
        )
        self.addCleanup(stop_server, self.server, self.thread)
        self.port = self.server.server_address[1]

    def test_done_event_carries_ollama_ps(self):
        payload = json.dumps(
            {
                "question": "вопрос",
                "use_system": True,
                "sides": [
                    {"backend": "vtm", "model": "mock-vtm"},
                    {"backend": "ollama", "model": "mock-qwen"},
                ],
            },
            ensure_ascii=False,
        ).encode("utf-8")
        status, raw = http_json(
            self.port,
            "POST",
            "/api/chat",
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 200)
        events, _ = parse_sse(raw)
        done = [
            event
            for event in events
            if event["type"] == "done" and event["side"] == 1
        ][0]
        self.assertEqual(done["ollama"]["size"], 1234)
        self.assertEqual(done["ollama"]["size_vram"], 1024)
        self.assertEqual(self.ollama.probes, ["mock-qwen"])
        # У ВТМ такого источника нет — не выдумываем числа.
        vtm_done = [
            event
            for event in events
            if event["type"] == "done" and event["side"] == 0
        ][0]
        self.assertIsNone(vtm_done["ollama"])


class TestMetricsPage(ServerTestCase):
    """Панель под чатом: разметка, запросы к своим эндпоинтам, CSP не ослаблен."""

    def test_page_has_metrics_panel(self):
        status, _, body = self.request("GET", "/")
        self.assertEqual(status, 200)
        page = body.decode("utf-8")
        for element_id in (
            'id="metrics"',
            'id="mc-size-0"',
            'id="mc-kv-1"',
            'id="mc-ps-0"',
            'id="ma-ttft-0"',
            'id="ma-cache-1"',
            'id="sys-load"',
            'id="sys-rss"',
            'id="sys-threads"',
        ):
            self.assertIn(element_id, page)
        self.assertIn("/api/models", page)
        self.assertIn("/api/system", page)

    def test_page_stays_single_style_and_script_without_network(self):
        status, _, body = self.request("GET", "/")
        self.assertEqual(status, 200)
        page = body.decode("utf-8")
        self.assertEqual(page.count("<style"), 1)
        self.assertEqual(page.count("<script"), 1)
        self.assertNotRegex(page, r"https?://")
        self.assertNotRegex(page, r"<script[^>]*\bsrc\s*=")
        self.assertNotRegex(page, r"url\s*\(")

    def test_csp_is_unchanged(self):
        status, headers, _ = self.request("GET", "/")
        self.assertEqual(status, 200)
        csp = headers.get("Content-Security-Policy", "")
        self.assertEqual(csp, chat_server.CSP)
        self.assertIn("default-src 'none'", csp)
        self.assertIn("connect-src 'self'", csp)
        self.assertNotIn("http", csp)


if __name__ == "__main__":
    unittest.main()
