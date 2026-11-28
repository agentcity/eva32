"""Проверки веб-стенда сравнения двух моделей (задача 0138), панели
метрик под чатом (задача 0145) и памяти диалога (задача 0182).

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
import mmap
import os
import pathlib
import re
import shutil
import socket
import struct
import subprocess
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
#: node нужен только для проверок клиентского JS (задача 0178); без него
#: эти тесты пропускаются, а не падают.
NODE = shutil.which("node")

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


class PassThroughPlugins:
    """Рантайм плагинов без фильтров (``plugins/`` не найден или выключен).

    Задача 0207: женский род не должен зависеть от discovery плагинов — ядро
    применяет ``eva.ru_gender`` само. Дактилотипия повторяет минимум
    ``PluginRegistry``: ``stream_filter``/``filter_text``.
    """

    class _Pipeline:
        def feed(self, text):
            return text

        def flush(self):
            return ""

        def __bool__(self):
            return False

    def stream_filter(self, *, scope="chat"):
        return self._Pipeline()

    def filter_text(self, text, *, scope="chat"):
        return text


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
# Клиентский JS страницы: <script>, парсер <think> (задача 0178)
# --------------------------------------------------------------------------- #

THINK_PARSER_START = "// --- think-парсер: начало"
THINK_PARSER_END = "// --- think-парсер: конец"

THINK_HARNESS = r"""
var chunks = JSON.parse(process.argv[2]);
var answer = "";
var think = "";
var parser = createThinkParser({
  answer: function (text) { answer += text; },
  think: function (text) { think += text; }
});
chunks.forEach(function (chunk) { parser.feed(chunk); });
parser.flush();
process.stdout.write(JSON.stringify({
  answer: answer, think: think, inThink: parser.inThink()
}));
"""


def page_html(case):
    """Отдать HTML главной страницы теста."""
    status, _, body = case.request("GET", "/")
    case.assertEqual(status, 200)
    return body.decode("utf-8")


def page_script(page):
    """Единственный <script> страницы (без внешних src — это проверяют тесты)."""
    match = re.search(r"<script>(.*?)</script>", page, re.S)
    if match is None:
        raise AssertionError("на странице нет встроенного <script>")
    return match.group(1)


def think_parser_source(page):
    """Вырезать из страницы самодостаточный блок парсера <think>."""
    script = page_script(page)
    start = script.index("\n", script.index(THINK_PARSER_START)) + 1
    marker = script.index(THINK_PARSER_END)
    end = script.rfind("\n", 0, marker) + 1
    return script[start:end]


def run_think_parser(page, chunks):
    """Прогнать куски потока через клиентский парсер страницы в node."""
    if NODE is None:
        raise unittest.SkipTest("node не установлен")
    source = think_parser_source(page) + THINK_HARNESS
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "think_parser.js")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(source)
        proc = subprocess.run(
            [NODE, path, json.dumps(list(chunks))],
            capture_output=True,
            text=True,
            timeout=60,
        )
    if proc.returncode != 0:
        raise AssertionError("node: " + (proc.stderr or proc.stdout))
    return json.loads(proc.stdout)


def stream_text(events, side=0):
    """Склеить delta одной колонки — ровно то, что видит клиент."""
    return "".join(
        event["text"]
        for event in events
        if event["type"] == "delta" and event["side"] == side
    )


# --------------------------------------------------------------------------- #
# Счётчик истории в шапке колонки (задача 0182): JS из отданной страницы
# --------------------------------------------------------------------------- #

HISTORY_UI_START = "  // История колонки (задача 0182)"
HISTORY_UI_END = "  function fillSideMetrics"

HISTORY_HARNESS = r"""
var out = {};
function setText(id, text) { out[id] = text; }
__SOURCE__
fillHistory({enabled: true, estimated: true, sides: [
  {side: 0, turns: 2, messages: 4, tokens: 37, estimated: true},
  {side: 1, turns: 0, messages: 0, tokens: 0, estimated: true}
]});
var filled = { h0: out["history-0"], h1: out["history-1"] };
fillHistory({enabled: false, sides: [
  {side: 0, turns: 5, messages: 10, tokens: 99, estimated: true},
  {side: 1, turns: 5, messages: 10, tokens: 99, estimated: true}
]});
var off = { h0: out["history-0"], h1: out["history-1"] };
fillHistory(null);
var afterNull = { h0: out["history-0"], h1: out["history-1"] };
process.stdout.write(JSON.stringify({
  filled: filled, off: off, afterNull: afterNull
}));
"""


def history_ui_source(page):
    """Вырезать из страницы блок счётчика истории (historyText/fillHistory)."""
    script = page_script(page)
    start = script.index(HISTORY_UI_START)
    end = script.index(HISTORY_UI_END)
    return script[start:end]


def run_history_ui(page):
    """Прогнать счётчик истории из отданной страницы в node."""
    if NODE is None:
        raise unittest.SkipTest("node не установлен")
    source = HISTORY_HARNESS.replace("__SOURCE__", history_ui_source(page))
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "history_ui.js")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(source)
        proc = subprocess.run(
            [NODE, path], capture_output=True, text=True, timeout=60
        )
    if proc.returncode != 0:
        raise AssertionError("node: " + (proc.stderr or proc.stdout))
    return json.loads(proc.stdout)


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


class TestAnswerUi(ServerTestCase):
    """Длинный ответ не обрезается, `<think>` свёрнут (задача 0178)."""

    def test_page_has_reasoning_block_and_collapse_control(self):
        page = page_html(self)
        for fragment in (
            '<details class="think" id="think-0" hidden>',
            '<details class="think" id="think-1" hidden>',
            '<summary id="think-sum-0">рассуждение</summary>',
            '<pre class="think-body" id="think-body-1"></pre>',
            'id="collapse-0"',
            'id="collapse-1"',
        ):
            self.assertIn(fragment, page)
        # Блок рассуждения стоит до ответа и свёрнут по умолчанию: JS только
        # снимает hidden, атрибут open не выставляется.
        self.assertNotIn("setAttribute(\"open\"", page)
        self.assertLess(page.index('id="think-0"'), page.index('id="answer-0"'))

    def test_answer_rule_has_no_fixed_max_height(self):
        page = page_html(self)
        match = re.search(r"\.answer\s*\{(.*?)\}", page, re.S)
        self.assertIsNotNone(match, "нет правила .answer")
        rule = match.group(1)
        self.assertIn("max-height: none", rule, "ответ снова обрезается по высоте")
        self.assertNotRegex(rule, r"max-height\s*:\s*[\d.]", "у ответа снова потолок высоты")
        self.assertIn("overflow: visible", rule)
        # Ограничение высоты осталось только у явно свёрнутого ответа.
        collapsed = re.search(r"\.panel\.answer-collapsed \.answer\s*\{(.*?)\}", page, re.S)
        self.assertIsNotNone(collapsed)
        self.assertIn("max-height", collapsed.group(1))

    def test_streaming_helpers_are_on_the_page(self):
        page = page_html(self)
        for name in (
            "createThinkParser",
            "function appendThink",
            "function flushAnswer",
            "function updateCollapseButton",
            "function followStream",
            "stickToBottom",
        ):
            self.assertIn(name, page)

    def test_autoscroll_targets_answer_block_not_page_bottom(self):
        # Метрики стоят под чатом: прокрутка к концу документа увозила бы
        # из ответа в панель метрик, поэтому цель — низ .grid.
        page = page_html(self)
        self.assertIn("function streamBottom", page)
        self.assertIn('querySelector(".grid")', page)
        self.assertNotIn("root.scrollTop = root.scrollHeight", page)

    @unittest.skipUnless(NODE, "node не установлен")
    def test_page_script_passes_node_check(self):
        script = page_script(page_html(self))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "page.js")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(script)
            proc = subprocess.run(
                [NODE, "--check", path],
                capture_output=True,
                text=True,
                timeout=60,
            )
        self.assertEqual(proc.returncode, 0, proc.stderr or proc.stdout)


class TestThinkParser(ServerTestCase):
    """Клиентский разбор <think> на разорванном потоке delta."""

    @unittest.skipUnless(NODE, "node не установлен")
    def test_think_goes_to_reasoning_and_never_to_answer(self):
        result = run_think_parser(
            page_html(self),
            ["<think>", "сначала подумаю", "</think>", "Ответ"],
        )
        self.assertEqual(result["answer"], "Ответ")
        self.assertEqual(result["think"], "сначала подумаю")
        self.assertFalse(result["inThink"])

    @unittest.skipUnless(NODE, "node не установлен")
    def test_split_tag_between_deltas_is_glued(self):
        result = run_think_parser(
            page_html(self),
            ["Начало. <thi", "nk>рассуждение</thi", "nk> Конец."],
        )
        self.assertEqual(result["answer"], "Начало.  Конец.")
        self.assertEqual(result["think"], "рассуждение")

    @unittest.skipUnless(NODE, "node не установлен")
    def test_closing_tag_split_after_slash(self):
        result = run_think_parser(
            page_html(self),
            ["<think>думаю</", "think>Ответ"],
        )
        self.assertEqual(result["answer"], "Ответ")
        self.assertEqual(result["think"], "думаю")

    @unittest.skipUnless(NODE, "node не установлен")
    def test_unclosed_think_is_reasoning_to_the_end(self):
        result = run_think_parser(
            page_html(self),
            ["Ответ<think>поток оборвался", " на середине"],
        )
        self.assertEqual(result["answer"], "Ответ")
        self.assertEqual(result["think"], "поток оборвался на середине")
        self.assertTrue(result["inThink"])

    @unittest.skipUnless(NODE, "node не установлен")
    def test_torn_tag_at_stream_end_is_not_shown(self):
        result = run_think_parser(page_html(self), ["Ответ", "<thi"])
        self.assertEqual(result["answer"], "Ответ")
        self.assertEqual(result["think"], "")

    @unittest.skipUnless(NODE, "node не установлен")
    def test_plain_text_and_angle_brackets_survive(self):
        result = run_think_parser(page_html(self), ["a < b, c > d — не тег"])
        self.assertEqual(result["answer"], "a < b, c > d — не тег")
        self.assertEqual(result["think"], "")


class TestThinkStream(ServerTestCase):
    """SSE отдаёт <think> как есть, а клиентский парсер раскладывает его."""

    @unittest.skipUnless(NODE, "node не установлен")
    def test_split_think_from_mock_backend_is_parsed(self):
        # Задержка между кусками больше окна склейки delta (60 мс): сервер
        # обязан отдать тег разрезанным, как в настоящем потоке Qwen3.
        self.vtm.pieces = ["<thi", "nk>рассуждаю вслух", "</thi", "nk>Ответ"]
        self.vtm.delay = 0.09
        events, _, _ = self.post_chat_ok()
        raw = stream_text(events)
        self.assertEqual(raw, "<think>рассуждаю вслух</think>Ответ")
        chunks = [
            event["text"]
            for event in events
            if event["type"] == "delta" and event["side"] == 0
        ]
        self.assertEqual(chunks[0], "<thi", "тег пришёл целым куском — проверка не о разрыве")
        result = run_think_parser(page_html(self), chunks)
        self.assertEqual(result["answer"], "Ответ")
        self.assertEqual(result["think"], "рассуждаю вслух")
        self.assertNotIn("<think>", result["answer"])


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
        # В CI раннер медленный и делится с соседями: порог мягче, проверяем
        # сам факт параллельности, а не скорость железа.
        limit = 3.5 if os.environ.get("CI") else 1.8
        self.assertLess(elapsed, limit)


class TestFeminineSpeech(ServerTestCase):
    """Женский род Евы в потоке ответа (задача 0207).

    Модель 3B пишет «я понял», «готов»; сервер обязан отдать странице и
    истории женские формы, а чужой род (он/ты) и цитаты не трогать.
    """

    def deltas(self, events, side=0):
        return "".join(
            event["text"]
            for event in events
            if event["type"] == "delta" and event["side"] == side
        )

    def test_stream_and_history_are_feminized(self):
        # Куски нарочно рвут слова: стример держит недописанное слово.
        self.vtm.pieces = ("Я всё сдел", "ал, я ", "готов", " помочь.")
        self.ollama.pieces = ("Я пон", "ял.")
        events, _, _ = self.post_chat_ok()
        self.assertEqual(self.deltas(events, 0), "Я всё сделала, я готова помочь.")
        self.assertEqual(self.deltas(events, 1), "Я поняла.")

        # История получает тот же текст, что видел пользователь: следующий
        # вопрос уходит в модель с женскими формами, а не с «сделал».
        self.post_chat_ok(make_stream_payload(question="второй вопрос"))
        second = self.vtm.calls[1]["messages"]
        self.assertIn(
            {"role": "assistant", "content": "Я всё сделала, я готова помочь."},
            second,
        )

    def test_other_person_and_quotes_are_untouched(self):
        self.vtm.pieces = ("Он понял, ты готов. «Я не уверен».",)
        events, _, _ = self.post_chat_ok()
        self.assertEqual(
            self.deltas(events, 0), "Он понял, ты готов. «Я не уверен»."
        )

    def test_plain_answer_is_unchanged(self):
        # Ответ без первого лица не должен даже переставиться по кускам.
        self.vtm.pieces = ("При", "вет")
        events, _, _ = self.post_chat_ok()
        self.assertEqual(self.deltas(events, 0), "Привет")

    def test_core_feminizes_without_plugins(self):
        # Плагины не найдены/выключены: включается страховка ядра (0207).
        self.server.plugins = PassThroughPlugins()
        self.vtm.pieces = ("Я пон", "ял, ", "готов", ".")
        events, _, _ = self.post_chat_ok()
        self.assertEqual(self.deltas(events, 0), "Я поняла, готов.")
        self.assertNotRegex(self.deltas(events, 0), r"понял(?!а)")


class TestChatHistory(ServerTestCase):
    """Память диалога (задача 0182): второй вопрос видит первый."""

    system_prompt = "СИСТЕМА: помни, о чём говорили."

    def ask(self, question="вопрос"):
        events, _, _ = self.post_chat_ok(
            make_stream_payload(question=question)
        )
        return events

    def reset(self, payload=None):
        body = json.dumps(
            payload if payload is not None else {}, ensure_ascii=False
        ).encode("utf-8")
        return self.request(
            "POST",
            "/api/reset",
            body=body,
            headers={"Content-Type": "application/json"},
        )

    def models(self):
        status, _, raw = self.request("GET", "/api/models")
        self.assertEqual(status, 200, raw.decode("utf-8", "replace"))
        return json.loads(raw)

    def history_of(self, side, data=None):
        data = data if data is not None else self.models()
        sides = [item for item in data["history"]["sides"] if item["side"] == side]
        self.assertEqual(len(sides), 1, data["history"])
        return sides[0]

    def test_second_question_sees_the_first(self):
        self.ask("меня зовут Юра")
        self.ask("как меня зовут?")
        self.assertEqual(len(self.vtm.calls), 2)
        first = self.vtm.calls[0]["messages"]
        second = self.vtm.calls[1]["messages"]
        # Первый вопрос — с чистого листа, второй видит первый обмен целиком.
        self.assertEqual([m["role"] for m in first], ["system", "user"])
        self.assertEqual(
            [m["role"] for m in second],
            ["system", "user", "assistant", "user"],
        )
        self.assertEqual(second[0], {"role": "system", "content": self.system_prompt})
        self.assertEqual(
            second[1], {"role": "user", "content": "меня зовут Юра"}
        )
        self.assertEqual(second[2], {"role": "assistant", "content": "Привет"})
        self.assertEqual(
            second[3], {"role": "user", "content": "как меня зовут?"}
        )

    def test_columns_keep_their_own_history(self):
        self.ask("меня зовут Юра")
        self.ask("как меня зовут?")
        vtm_second = self.vtm.calls[1]["messages"]
        ollama_second = self.ollama.calls[1]["messages"]
        # Каждая колонка помнит свой ответ — истории не смешиваются.
        self.assertEqual(vtm_second[2]["content"], "Привет")
        self.assertEqual(ollama_second[2]["content"], "Здравствуйте!")
        vtm_text = [message["content"] for message in vtm_second]
        ollama_text = [message["content"] for message in ollama_second]
        self.assertNotIn("Здравствуйте!", vtm_text)
        self.assertNotIn("Привет", ollama_text)

    def test_history_does_not_leak_into_system_prompt(self):
        self.ask("меня зовут Юра")
        self.ask("как меня зовут?")
        messages = self.vtm.calls[1]["messages"]
        self.assertEqual(messages[0]["content"], self.system_prompt)
        self.assertEqual(
            [message["role"] for message in messages].count("system"), 1
        )

    def test_error_does_not_grow_history(self):
        self.vtm.error = "модель упала"
        self.ask("этот вопрос упадёт")
        self.assertEqual(self.history_of(0)["turns"], 0)
        # Соседняя колонка ответила — у неё история растёт независимо.
        self.assertEqual(self.history_of(1)["turns"], 1)

        self.vtm.error = None
        self.ask("второй вопрос")
        second = self.vtm.calls[1]["messages"]
        self.assertEqual([m["role"] for m in second], ["system", "user"])
        self.assertEqual(second[-1]["content"], "второй вопрос")

    def test_empty_answer_does_not_grow_history(self):
        self.vtm.pieces = ()
        self.ask("пустой ответ")
        self.assertEqual(self.history_of(0)["turns"], 0)

        self.vtm.pieces = ("ответ",)
        self.ask("второй вопрос")
        roles = [m["role"] for m in self.vtm.calls[1]["messages"]]
        self.assertEqual(roles, ["system", "user"])

    def test_counter_in_models_matches_estimate(self):
        data = self.models()["history"]
        self.assertTrue(data["enabled"])
        self.assertEqual(data["limit_tokens"], chat_server.DEFAULT_HISTORY_TOKENS)
        self.assertEqual(data["limit_turns"], chat_server.DEFAULT_HISTORY_TURNS)
        self.assertTrue(data["estimated"])
        self.assertEqual(data["sides"][0]["turns"], 0)

        self.ask("меня зовут Юра")
        data = self.models()["history"]
        stats = [item for item in data["sides"] if item["side"] == 0][0]
        self.assertEqual(stats["turns"], 1)
        self.assertEqual(stats["messages"], 2)
        self.assertEqual(
            stats["tokens"],
            chat_server.estimate_tokens("меня зовут Юра")
            + chat_server.estimate_tokens("Привет"),
        )
        self.assertGreater(stats["tokens"], 0)
        other = [item for item in data["sides"] if item["side"] == 1][0]
        self.assertEqual(
            other["tokens"],
            chat_server.estimate_tokens("меня зовут Юра")
            + chat_server.estimate_tokens("Здравствуйте!"),
        )

    def test_reset_clears_one_column(self):
        self.ask("меня зовут Юра")
        status, headers, raw = self.reset({"side": 0})
        self.assertEqual(status, 200, raw.decode("utf-8", "replace"))
        self.assertIn("application/json", headers.get("Content-Type", ""))
        body = json.loads(raw)
        self.assertEqual(body["reset"], [0])
        self.assertEqual(
            [item["turns"] for item in body["history"]["sides"]], [0, 1]
        )

        self.ask("как меня зовут?")
        vtm_second = self.vtm.calls[1]["messages"]
        self.assertEqual([m["role"] for m in vtm_second], ["system", "user"])
        ollama_second = self.ollama.calls[1]["messages"]
        self.assertEqual(
            [m["role"] for m in ollama_second],
            ["system", "user", "assistant", "user"],
        )

    def test_reset_without_side_clears_both(self):
        self.ask("меня зовут Юра")
        status, _, raw = self.reset({})
        self.assertEqual(status, 200)
        body = json.loads(raw)
        self.assertEqual(body["reset"], [0, 1])
        self.assertEqual(
            [item["turns"] for item in body["history"]["sides"]], [0, 0]
        )
        data = self.models()
        self.assertEqual(
            [self.history_of(side, data)["turns"] for side in (0, 1)], [0, 0]
        )

    def test_bad_reset_side_is_rejected(self):
        for payload in ({"side": 2}, {"side": "0"}, {"side": True}, {"side": -1}):
            status, headers, raw = self.reset(payload)
            self.assertEqual(status, 400, raw.decode("utf-8", "replace"))
            self.assertIn("application/json", headers.get("Content-Type", ""))
            self.assertIn("error", json.loads(raw))

    def test_reset_rejects_non_object_body(self):
        status, _, raw = self.reset([0])
        self.assertEqual(status, 400, raw.decode("utf-8", "replace"))
        self.assertIn("invalid_request", raw.decode("utf-8"))


class HistoryLimitsTestCase(unittest.TestCase):
    """Сервер с явными лимитами истории: обрезка и отключение."""

    system_prompt = "СИСТЕМА: коротко."

    def serve(self, **kwargs):
        self.vtm = MockAdapter(pieces=("Привет",))
        self.ollama = MockAdapter(pieces=("Здравствуйте",))
        server, thread = start_server(
            REGISTRY,
            {("vtm", "mock-vtm"): self.vtm, ("ollama", "mock-qwen"): self.ollama},
            system_prompt=self.system_prompt,
            **kwargs,
        )
        self.addCleanup(stop_server, server, thread)
        return server, server.server_address[1]

    def ask(self, port, question="вопрос"):
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
            port,
            "POST",
            "/api/chat",
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 200, raw.decode("utf-8", "replace"))
        events, done = parse_sse(raw)
        self.assertTrue(done, "поток не завершился событием [DONE]")
        return events

    def models(self, port):
        status, raw = http_json(port, "GET", "/api/models")
        self.assertEqual(status, 200, raw.decode("utf-8", "replace"))
        return json.loads(raw)

    def history_of(self, port, side=0):
        data = self.models(port)["history"]
        return [item for item in data["sides"] if item["side"] == side][0]

    def test_zero_tokens_disables_history(self):
        server, port = self.serve(history_tokens=0)
        self.assertFalse(server.history.enabled)
        self.ask(port, "меня зовут Юра")
        self.ask(port, "как меня зовут?")
        second = self.vtm.calls[1]["messages"]
        self.assertEqual([m["role"] for m in second], ["system", "user"])
        data = self.models(port)["history"]
        self.assertFalse(data["enabled"])
        self.assertEqual([item["turns"] for item in data["sides"]], [0, 0])

    def test_turns_limit_drops_oldest_pairs(self):
        _, port = self.serve(history_turns=2)
        for number in range(1, 5):
            self.ask(port, "вопрос {}".format(number))
        last = self.vtm.calls[-1]["messages"]
        self.assertEqual(
            [m["role"] for m in last],
            ["system", "user", "assistant", "user", "assistant", "user"],
        )
        contents = [m["content"] for m in last]
        # Две последних реплики: третья и четвёртая; первая и вторая ушли.
        self.assertEqual(contents[1], "вопрос 2")
        self.assertEqual(contents[3], "вопрос 3")
        self.assertEqual(contents[5], "вопрос 4")
        self.assertEqual(last[0]["content"], self.system_prompt)
        stats = self.history_of(port)
        self.assertLessEqual(stats["turns"], 2)

    def test_token_budget_keeps_system_and_latest_pair(self):
        _, port = self.serve(history_tokens=120)
        self.vtm.pieces = ("я" * 300,)
        self.vtm.usage = {"prompt_tokens": 5, "completion_tokens": 1}
        self.ask(port, "первый вопрос")
        self.ask(port, "второй вопрос")
        self.ask(port, "третий вопрос")
        last = self.vtm.calls[-1]["messages"]
        texts = [m["content"] for m in last]
        # Системный промпт на месте, старейшая пара выброшена целиком.
        self.assertEqual(last[0]["content"], self.system_prompt)
        self.assertNotIn("первый вопрос", texts)
        self.assertIn("второй вопрос", texts)
        self.assertEqual(texts[-1], "третий вопрос")
        stats = self.history_of(port)
        self.assertLessEqual(stats["tokens"], 120)
        self.assertEqual(stats["turns"], 1)

    def test_history_never_grows_past_limits(self):
        _, port = self.serve()
        for number in range(12):
            self.ask(port, "вопрос номер {}".format(number))
        last = self.vtm.calls[-1]["messages"]
        # system + история (не больше 8 реплик) + текущий вопрос.
        self.assertLessEqual(len(last), 2 * chat_server.DEFAULT_HISTORY_TURNS + 2)
        stats = self.history_of(port)
        self.assertLessEqual(stats["turns"], chat_server.DEFAULT_HISTORY_TURNS)

    def test_server_validates_history_limits(self):
        with self.assertRaises(ValueError):
            chat_server.create_server(
                port=0, registry=REGISTRY, adapters={}, history_tokens=-1
            )
        with self.assertRaises(ValueError):
            chat_server.create_server(
                port=0, registry=REGISTRY, adapters={}, history_turns=0
            )
        with self.assertRaises(TypeError):
            chat_server.create_server(
                port=0, registry=REGISTRY, adapters={}, history_tokens=True
            )
        with self.assertRaises(TypeError):
            chat_server.create_server(
                port=0, registry=REGISTRY, adapters={}, history_turns="8"
            )

    def test_cli_flags(self):
        parser = chat_server._build_parser()
        args = parser.parse_args([])
        self.assertEqual(args.history_tokens, chat_server.DEFAULT_HISTORY_TOKENS)
        self.assertEqual(args.history_turns, chat_server.DEFAULT_HISTORY_TURNS)
        self.assertEqual(
            parser.parse_args(["--history-tokens", "0"]).history_tokens, 0
        )
        self.assertEqual(
            parser.parse_args(["--history-turns", "3"]).history_turns, 3
        )


class TestHistoryPage(ServerTestCase):
    """UI памяти диалога: счётчик в шапке колонки и «Новый диалог»."""

    def test_page_has_history_header_and_reset_button(self):
        page = page_html(self)
        for fragment in (
            'id="history-0"',
            'id="history-1"',
            'id="new-0"',
            'id="new-1"',
            ">Новый диалог<",
            "/api/reset",
            "реплик",
            "токенов",
            "function historyText",
            "function fillHistory",
            "function resetDialog",
        ):
            self.assertIn(fragment, page)

    def test_header_counter_is_filled_from_models(self):
        # Счётчик приходит в /api/models — страница берёт его оттуда, а не
        # считает сама (у клиента нет ни истории, ни токенизатора).
        self.post_chat_ok(make_stream_payload(question="меня зовут Юра"))
        status, _, raw = self.request("GET", "/api/models")
        self.assertEqual(status, 200)
        data = json.loads(raw)
        self.assertIn("history", data)
        self.assertEqual(
            [item["side"] for item in data["history"]["sides"]], [0, 1]
        )
        self.assertEqual(data["history"]["sides"][0]["turns"], 1)

    def test_history_counter_text_in_page(self):
        result = run_history_ui(page_html(self))
        self.assertEqual(
            result["filled"]["h0"], "2 реплик · ~37 токенов (оценка)"
        )
        self.assertEqual(
            result["filled"]["h1"], "0 реплик · ~0 токенов (оценка)"
        )
        # --history-tokens 0: честное «выключена», а не «0 токенов».
        self.assertEqual(result["off"]["h0"], "выключена (0 токенов)")
        self.assertEqual(result["off"]["h1"], "выключена (0 токенов)")
        # Нет data.history — шапка не падает и не затирает прошлое значение.
        self.assertEqual(result["afterNull"], result["off"])


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


def start_server(
    registry,
    adapters,
    *,
    max_tokens=32,
    system_prompt="СИСТЕМА",
    keep_alive=None,
    history_tokens=None,
    history_turns=None,
):
    """Поднять сервер на свободном порту loopback; вернуть ``(server, thread)``."""
    history = {}
    # None — «как на сервере по умолчанию»: тест про историю задаёт лимиты явно.
    if history_tokens is not None:
        history["history_tokens"] = history_tokens
    if history_turns is not None:
        history["history_turns"] = history_turns
    server = chat_server.create_server(
        port=0,
        registry=registry,
        adapters=adapters,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        keep_alive=keep_alive,
        **history,
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

    def reset_histories(self):
        """Новый диалог в обеих колонках (задача 0182)."""
        status, raw = http_json(
            self.port,
            "POST",
            "/api/reset",
            body=b"{}",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 200, raw.decode("utf-8", "replace"))

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

    def test_follow_up_with_history_reuses_prefix(self):
        # История диалога — тоже начало промпта, поэтому второй вопрос в том
        # же диалоге законно попадает в кэш префикса ВТМ (задача 0182).
        self.ask("меня зовут Юра")
        done = self.ask("как меня зовут?")
        cache = done["cache_prefix"]
        self.assertTrue(cache["hit"], cache)
        self.assertEqual(cache["shared_chars"], cache["previous_chars"])

    def test_different_prompt_is_a_miss(self):
        self.ask("первый вопрос про стрижку")
        # С историей прошлый промпт целиком входит в новый (это попадание,
        # см. тест выше). Промах проверяем на новом диалоге: общим остаётся
        # только системный промпт, а вопросы разные.
        self.reset_histories()
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
            'id="mc-loaded-0"',
            'id="mc-rssbytes-1"',
            'id="mc-loadms-0"',
            'id="mc-req-1"',
            'id="load-0"',
            'id="unload-1"',
            'id="ma-ttft-0"',
            'id="ma-ttftgen-1"',
            'id="ma-cache-1"',
            'id="sys-load"',
            'id="sys-rss"',
            'id="sys-threads"',
            'id="sys-keepalive"',
        ):
            self.assertIn(element_id, page)
        self.assertIn("/api/models", page)
        self.assertIn("/api/system", page)
        # Кнопки памяти ходят только в свои эндпоинты.
        self.assertIn("/api/load", page)
        self.assertIn("/api/unload", page)

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


# --------------------------------------------------------------------------- #
# Загрузка/выгрузка модели (задача 0166)
# --------------------------------------------------------------------------- #


class LoadableModel:
    """Заглушка ВТМ с явной загрузкой: ``open()`` «читает веса», ``close()`` их отпускает.

    ``stream`` ставит ``started`` до первого куска — тесту про 409 нужно
    дождаться, что генерация действительно началась, а не спорить с ней.
    """

    cache_prefix = True

    def __init__(
        self, holder, *, megabytes=24, open_delay=0.0, stream_delay=0.0
    ) -> None:
        self.holder = holder
        self.megabytes = int(megabytes)
        self.open_delay = float(open_delay)
        self.stream_delay = float(stream_delay)
        self.opened = 0
        self.closed = 0
        self.streams = 0
        self.started = threading.Event()

    def open(self):
        self.opened += 1
        if self.open_delay:
            time.sleep(self.open_delay)
        if self.megabytes:
            # Анонимное отображение, как веса модели: страницы уходят системе
            # при close(), поэтому RSS-дельта и освобождение видны честно.
            size = self.megabytes * 1024 * 1024
            mapping = mmap.mmap(-1, size)
            for offset in range(0, size, mmap.PAGESIZE):
                mapping[offset] = 7
            self.holder.append(mapping)

    def close(self):
        self.closed += 1
        for mapping in list(self.holder):
            try:
                mapping.close()
            except (BufferError, ValueError):  # pragma: no cover - защита
                pass
        self.holder.clear()

    def stream(self, messages, *, max_tokens):
        self.streams += 1
        self.started.set()
        if self.stream_delay:
            time.sleep(self.stream_delay)
        yield SimpleNamespace(text="ок")
        yield SimpleNamespace(
            done=True,
            finish_reason="stop",
            usage={"prompt_tokens": 3, "completion_tokens": 1},
        )


class LoadableFactory:
    """Фабрика :class:`LoadableModel` с журналом созданных объектов."""

    def __init__(self, holder, **kwargs):
        self.holder = holder
        self.kwargs = kwargs
        self.models = []

    def __call__(self):
        model = LoadableModel(self.holder, **self.kwargs)
        self.models.append(model)
        return model


class MemoryServerTestCase(unittest.TestCase):
    """Сервер с «загружаемой» заглушкой ВТМ: POST /api/load|unload по HTTP."""

    keep_alive = None
    factory_kwargs: dict = {}

    def setUp(self):
        self.server, self.thread, self.port, self.adapter, self.factory = (
            self.make_server(self.keep_alive, **self.factory_kwargs)
        )

    def make_server(self, keep_alive=None, **factory_kwargs):
        holder = []
        factory = LoadableFactory(holder, **factory_kwargs)
        adapter = chat_server.ChatCliAdapter(factory, kind="vtm", label="vtm")
        server, thread = start_server(
            REGISTRY,
            {
                ("vtm", "mock-vtm"): adapter,
                ("ollama", "mock-qwen"): MockAdapter(pieces=("в",)),
            },
            keep_alive=keep_alive,
        )
        self.addCleanup(stop_server, server, thread)
        self.addCleanup(holder.clear)
        return server, thread, server.server_address[1], adapter, factory

    # -- HTTP-помощники ------------------------------------------------------

    def post(self, path, payload, port=None):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return http_json(
            port or self.port,
            "POST",
            path,
            body=body,
            headers={"Content-Type": "application/json"},
        )

    def models(self, port=None):
        status, raw = http_json(port or self.port, "GET", "/api/models")
        self.assertEqual(status, 200, raw.decode("utf-8", "replace"))
        return json.loads(raw)

    def card(self, model="mock-vtm", port=None):
        cards = [item for item in self.models(port)["models"] if item["model"] == model]
        self.assertEqual(len(cards), 1)
        return cards[0]

    def load(self, port=None):
        status, raw = self.post(
            "/api/load", {"backend": "vtm", "model": "mock-vtm"}, port=port
        )
        self.assertEqual(status, 200, raw.decode("utf-8", "replace"))
        return json.loads(raw)

    def unload(self, port=None):
        status, raw = self.post(
            "/api/unload", {"backend": "vtm", "model": "mock-vtm"}, port=port
        )
        self.assertEqual(status, 200, raw.decode("utf-8", "replace"))
        return json.loads(raw)

    def ask(self, port=None):
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
            port or self.port,
            "POST",
            "/api/chat",
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 200, raw.decode("utf-8", "replace"))
        events, done = parse_sse(raw)
        self.assertTrue(done, "поток не завершился событием [DONE]")
        done_events = [
            event for event in events if event["type"] == "done" and event["side"] == 0
        ]
        self.assertEqual(len(done_events), 1, events)
        return done_events[0], events, raw


class TestLoadUnloadAPI(MemoryServerTestCase):
    """Кнопка загружает модель: статус, RSS-дельта, время, счётчик запросов."""

    def test_load_makes_model_resident_and_reports_numbers(self):
        before = self.card()
        self.assertFalse(before["loaded"])
        self.assertIsNone(before["rss_bytes"])
        self.assertIsNone(before["load_ms"])
        self.assertEqual(before["requests"], 0)

        data = self.load()
        self.assertEqual(data["action"], "load")
        self.assertTrue(data["loaded"])
        self.assertFalse(data["already_loaded"])
        self.assertIsNotNone(data["rss_bytes"])
        self.assertGreater(data["rss_bytes"], 0)
        self.assertGreaterEqual(data["load_ms"], 0)
        self.assertEqual(data["requests"], 0)
        self.assertEqual(len(self.factory.models), 1)
        self.assertEqual(self.factory.models[0].opened, 1)
        # open() + прогрев: веса уже прочитаны, а не отложены до запроса.
        self.assertTrue(data["load"]["opened"])
        self.assertTrue(data["load"]["warmed"])
        self.assertGreaterEqual(self.factory.models[0].streams, 1)

        card = self.card()
        self.assertTrue(card["loaded"])
        self.assertEqual(card["rss_bytes"], data["rss_bytes"])
        self.assertEqual(card["load_ms"], data["load_ms"])
        self.assertEqual(card["requests"], 0)
        self.assertIsNotNone(card["load"])
        self.assertEqual(
            card["load"]["rss_delta"],
            card["load"]["rss_after"] - card["load"]["rss_before"],
        )

    def test_second_load_is_idempotent(self):
        first = self.load()
        second = self.load()
        self.assertTrue(second["already_loaded"])
        self.assertEqual(second["rss_bytes"], first["rss_bytes"])
        self.assertEqual(second["load_ms"], first["load_ms"])
        self.assertEqual(len(self.factory.models), 1, "фабрику позвали повторно")

    def test_unload_clears_status_and_next_request_reloads(self):
        loaded = self.load()
        data = self.unload()
        self.assertTrue(data["was_loaded"])
        self.assertFalse(data["loaded"])
        self.assertIsInstance(data["freed_bytes"], int)
        self.assertGreater(data["freed_bytes"], 0, "выгрузка не освободила память")
        self.assertEqual(data["rss_bytes"], 0)
        self.assertIsNone(data["load_ms"])
        self.assertEqual(self.factory.models[0].closed, 1)

        card = self.card()
        self.assertFalse(card["loaded"])
        self.assertEqual(card["rss_bytes"], 0)
        self.assertIsNone(card["load_ms"])
        # История загрузки остаётся в карточке: видно, сколько заняла.
        self.assertIsNotNone(card["load"])
        self.assertEqual(card["load"]["load_ms"], loaded["load_ms"])

        # Без загрузки запрос снова грузит модель сам — прежнее поведение.
        done, _, _ = self.ask()
        self.assertTrue(done["cold"])
        self.assertEqual(len(self.factory.models), 2)
        self.assertTrue(self.card()["loaded"])

    def test_unknown_model_and_backend_are_400(self):
        for payload in (
            {"backend": "vtm", "model": "нет-такой.vtm"},
            {"backend": "nope", "model": "mock-vtm"},
            {"model": ""},
            {"backend": "vtm"},
        ):
            status, raw = self.post("/api/load", payload)
            self.assertEqual(status, 400, (payload, raw))
            error = json.loads(raw)["error"]
            self.assertTrue(error["message"])
            self.assertIn(error["code"], ("invalid_request",))

        # Выгрузка проверяет то же самое.
        status, raw = self.post(
            "/api/unload", {"backend": "vtm", "model": "нет-такой.vtm"}
        )
        self.assertEqual(status, 400)

    def test_ollama_load_is_refused(self):
        status, raw = self.post(
            "/api/load", {"backend": "ollama", "model": "mock-qwen"}
        )
        self.assertEqual(status, 400)
        error = json.loads(raw)["error"]
        self.assertEqual(error["code"], "load_unsupported")
        self.assertIn("Ollama", error["message"])

    def test_adapter_without_load_is_refused(self):
        # Заглушка без load/unload: эндпоинт обязан отказать, а не падать.
        server, thread = start_server(
            REGISTRY,
            {
                ("vtm", "mock-vtm"): MockAdapter(pieces=("а",)),
                ("ollama", "mock-qwen"): MockAdapter(pieces=("в",)),
            },
        )
        self.addCleanup(stop_server, server, thread)
        port = server.server_address[1]
        status, raw = self.post(
            "/api/load", {"backend": "vtm", "model": "mock-vtm"}, port=port
        )
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(raw)["error"]["code"], "load_unsupported")

    def test_unload_is_refused_while_generating(self):
        _, _, port, adapter, factory = self.make_server(stream_delay=0.5)
        status, raw = self.post(
            "/api/load", {"backend": "vtm", "model": "mock-vtm"}, port=port
        )
        self.assertEqual(status, 200)

        results = []

        def chat():
            try:
                payload = json.dumps(
                    {
                        "question": "вопрос",
                        "use_system": False,
                        "sides": [
                            {"backend": "vtm", "model": "mock-vtm"},
                            {"backend": "ollama", "model": "mock-qwen"},
                        ],
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                results.append(
                    http_json(
                        port,
                        "POST",
                        "/api/chat",
                        body=payload,
                        headers={"Content-Type": "application/json"},
                    )
                )
            except Exception as exc:  # pragma: no cover - диагностика
                results.append(exc)

        thread = threading.Thread(target=chat, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 5.0)
        deadline = time.monotonic() + 3.0
        model = None
        while time.monotonic() < deadline:
            # Первый stream — прогрев при загрузке, второй — сам запрос.
            if factory.models and factory.models[0].streams >= 2:
                model = factory.models[0]
                break
            time.sleep(0.01)
        self.assertIsNotNone(model, "генерация не началась")
        self.assertTrue(model.started.is_set())

        status, raw = self.post(
            "/api/unload", {"backend": "vtm", "model": "mock-vtm"}, port=port
        )
        self.assertEqual(status, 409, raw.decode("utf-8", "replace"))
        self.assertEqual(json.loads(raw)["error"]["code"], "model_busy")
        thread.join(timeout=5.0)
        self.assertFalse(thread.is_alive())
        self.assertEqual(model.closed, 0, "модель закрыли во время генерации")
        self.assertTrue(adapter.loaded)

    def test_requests_are_counted_across_loads(self):
        self.load()
        self.ask()
        self.ask()
        self.assertEqual(self.card()["requests"], 2)
        self.unload()
        self.ask()
        self.assertEqual(self.card()["requests"], 3)


class TestColdWarmWithPreload(MemoryServerTestCase):
    """TTFT: у заранее загруженной модели он не содержит загрузки."""

    factory_kwargs = {"open_delay": 0.1, "stream_delay": 0.05, "megabytes": 2}

    def test_preloaded_request_is_warm_and_has_no_load(self):
        self.load()
        done, _, _ = self.ask()
        self.assertFalse(done["cold"])
        self.assertEqual(done["first_token"], "тёплый")
        self.assertIsNone(done["load_ms"])
        self.assertEqual(done["ttft_gen_ms"], done["first_token_ms"])
        self.assertIsInstance(done["first_token_ms"], int)

    def test_cold_request_subtracts_load_from_ttft(self):
        done, _, _ = self.ask()  # модель не загружена: грузится этим запросом
        self.assertTrue(done["cold"])
        self.assertEqual(done["first_token"], "холодный")
        self.assertIsNotNone(done["load_ms"])
        self.assertGreaterEqual(
            done["load_ms"], 80, "задержка open() не попала в замер"
        )
        self.assertGreater(done["first_token_ms"], done["load_ms"])
        self.assertEqual(
            done["ttft_gen_ms"],
            max(0, int(round(done["first_token_ms"] - done["load_ms"]))),
        )
        self.assertLess(done["ttft_gen_ms"], done["first_token_ms"])

    def test_load_ms_is_not_reused_by_next_request(self):
        self.ask()  # холодный, с загрузкой
        second, _, _ = self.ask()  # тёплый
        self.assertFalse(second["cold"])
        self.assertIsNone(second["load_ms"])
        self.assertEqual(second["ttft_gen_ms"], second["first_token_ms"])


class TestKeepAlive(MemoryServerTestCase):
    """Автовыгрузка по бездействию: по умолчанию выключена, N>0 — работает."""

    def test_default_is_disabled(self):
        self.assertIsNone(self.server.keep_alive)
        self.assertIsNone(self.models()["keep_alive"])
        self.load()
        now = self.adapter._last_used + 10_000.0
        self.assertEqual(self.server.reap_idle(now=now), [])
        self.assertTrue(self.card()["loaded"])

    def test_zero_means_keep_forever(self):
        server, _, port, adapter, _ = self.make_server(0)
        self.assertEqual(server.keep_alive, 0.0)
        self.assertEqual(self.models(port)["keep_alive"], 0.0)
        self.load(port=port)
        self.assertEqual(server.reap_idle(now=adapter._last_used + 10_000.0), [])
        self.assertTrue(adapter.loaded)

    def test_reap_idle_unloads_after_the_limit(self):
        server, _, port, adapter, factory = self.make_server(0.05)
        self.load(port=port)
        self.assertEqual(server.keep_alive, 0.05)
        self.assertEqual(self.models(port)["keep_alive"], 0.05)
        last = adapter._last_used
        self.assertEqual(server.reap_idle(now=last + 0.01), [])
        self.assertTrue(adapter.loaded)

        self.assertEqual(
            server.reap_idle(now=last + 1.0), [("vtm", "mock-vtm")]
        )
        self.assertFalse(self.card(port=port)["loaded"])
        self.assertEqual(factory.models[0].closed, 1)
        # Повторный проход ничего не делает: выгружать нечего.
        self.assertEqual(server.reap_idle(now=last + 1000.0), [])

    def test_background_reaper_unloads_idle_model(self):
        server, _, port, adapter, factory = self.make_server(0.05)
        self.load(port=port)
        self.assertTrue(adapter.loaded)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if not adapter.loaded:
                break
            time.sleep(0.02)
        self.assertFalse(adapter.loaded, "фоновая автовыгрузка не сработала")
        self.assertEqual(factory.models[0].closed, 1)
        self.assertFalse(self.card(port=port)["loaded"])

    def test_cli_flag(self):
        parser = chat_server._build_parser()
        self.assertIsNone(parser.parse_args([]).keep_alive)
        self.assertEqual(parser.parse_args(["--keep-alive", "0"]).keep_alive, 0.0)
        self.assertEqual(parser.parse_args(["--keep-alive", "120"]).keep_alive, 120.0)
        with self.assertRaises(ValueError):
            chat_server.create_server(
                port=0, registry=REGISTRY, adapters={}, keep_alive=-1
            )



# --------------------------------------------------------------------------- #
# Озвучка ответов: /api/tts и фразы в клиенте (задача 0183)
# --------------------------------------------------------------------------- #

TTS_SPLIT_START = "// --- TTS-фразы: начало"
TTS_SPLIT_END = "// --- TTS-фразы: конец"

TTS_HARNESS = r"""
var payload = JSON.parse(process.argv[2]);
var phrases = [];
var splitter = createPhraseSplitter(function (phrase) { phrases.push(phrase); });
payload.chunks.forEach(function (chunk) { splitter.feed(chunk); });
var during = phrases.slice();
splitter.flush();
process.stdout.write(JSON.stringify({ during: during, after: phrases }));
"""


def make_wav(payload=2048):
    """RIFF/WAVE с настоящим заголовком: тест сверяет формат, а не звук."""
    samples = b"\0" * int(payload)
    fmt = struct.pack("<IHHIIHH", 16, 1, 1, 22050, 44100, 2, 16)
    body = b"WAVE" + b"fmt " + fmt + b"data" + struct.pack("<I", len(samples)) + samples
    return b"RIFF" + struct.pack("<I", len(body)) + body


def make_mp3():
    """MP3 с ID3-тегом: содержимое важно только сигнатурой, не звуком."""
    return b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\0" * 512


def tts_split_source(page):
    """Вырезать из страницы самодостаточный блок разбиения на фразы."""
    script = page_script(page)
    start = script.index("\n", script.index(TTS_SPLIT_START)) + 1
    end = script.rfind("\n", 0, script.index(TTS_SPLIT_END)) + 1
    return script[start:end]


def run_tts_splitter(page, chunks):
    """Прогнать куски потока через клиентский сплиттер страницы в node."""
    if NODE is None:
        raise unittest.SkipTest("node не установлен")
    source = tts_split_source(page) + TTS_HARNESS
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "tts_split.js")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(source)
        proc = subprocess.run(
            [
                NODE,
                path,
                json.dumps({"chunks": list(chunks)}, ensure_ascii=False),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
    if proc.returncode != 0:
        raise AssertionError("node: " + (proc.stderr or proc.stdout))
    return json.loads(proc.stdout)


class TtsServerCase(unittest.TestCase):
    """Сервер с подменённым синтезом: звук не запускается, WAV — заглушка."""

    def setUp(self) -> None:
        self.calls = []
        self.server = chat_server.create_server(
            port=0,
            registry=REGISTRY,
            adapters={
                ("vtm", "mock-vtm"): MockAdapter(pieces=("Ответ",)),
                ("ollama", "mock-qwen"): MockAdapter(pieces=("Ответ",)),
            },
            system_prompt="СИСТЕМА: отвечай коротко.",
            max_tokens=32,
            tts=self.fake_tts,
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

    def fake_tts(self, text, voice=None):
        self.calls.append({"text": text, "voice": voice})
        return make_wav()

    def request(self, method, path, body=None, headers=None, timeout=10):
        conn = http.client.HTTPConnection(HOST, self.port, timeout=timeout)
        try:
            conn.request(method, path, body=body, headers=headers or {})
            response = conn.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            conn.close()

    def post_tts(self, payload, timeout=10):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return self.request(
            "POST",
            "/api/tts",
            body=body,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )


class TestTtsEndpoint(TtsServerCase):
    """``POST /api/tts``: WAV, чистка текста, лимит и деградация."""

    def test_returns_wav_without_cache(self):
        status, headers, body = self.post_tts({"text": "Привет, мир."})
        self.assertEqual(status, 200, body[:300])
        self.assertEqual(headers.get("Content-Type"), "audio/wav")
        self.assertEqual(headers.get("Cache-Control"), "no-store")
        self.assertEqual(body[:4], b"RIFF")
        self.assertEqual(body[8:12], b"WAVE")
        self.assertEqual(self.calls, [{"text": "Привет, мир.", "voice": "baya"}])

    def test_returns_mp3_with_audio_mpeg_for_tuple_result(self):
        self.server.tts = lambda text, voice=None: (make_mp3(), "audio/mpeg")
        status, headers, body = self.post_tts({"text": "Привет"})
        self.assertEqual(status, 200, body[:300])
        self.assertEqual(headers.get("Content-Type"), "audio/mpeg")
        self.assertEqual(headers.get("Cache-Control"), "no-store")
        self.assertEqual(body, make_mp3())

    def test_mp3_id3_signature_sets_audio_mpeg_without_hint(self):
        self.server.tts = lambda text, voice=None: make_mp3()
        status, headers, body = self.post_tts({"text": "Привет"})
        self.assertEqual(status, 200, body[:300])
        self.assertEqual(headers.get("Content-Type"), "audio/mpeg")

    def test_mp3_frame_sync_signature_sets_audio_mpeg(self):
        frame = b"\xff\xfb\x90\x64" + b"\0" * 256
        self.server.tts = lambda text, voice=None: frame
        status, headers, body = self.post_tts({"text": "Привет"})
        self.assertEqual(status, 200, body[:300])
        self.assertEqual(headers.get("Content-Type"), "audio/mpeg")

    def test_signature_wins_over_wrong_declared_type(self):
        self.server.tts = lambda text, voice=None: (make_wav(), "audio/mpeg")
        status, headers, body = self.post_tts({"text": "Привет"})
        self.assertEqual(status, 200, body[:300])
        self.assertEqual(headers.get("Content-Type"), "audio/wav")
        self.assertEqual(body[:4], b"RIFF")

    def test_empty_tuple_is_503(self):
        self.server.tts = lambda text, voice=None: (b"", "audio/mpeg")
        status, _, body = self.post_tts({"text": "Привет"})
        self.assertEqual(status, 503)
        self.assertEqual(json.loads(body)["error"]["code"], "voice_unavailable")

    def test_tuple_with_non_audio_bytes_is_503(self):
        self.server.tts = lambda text, voice=None: (b"not audio", "audio/mpeg")
        status, _, body = self.post_tts({"text": "Привет"})
        self.assertEqual(status, 503)
        self.assertIn("WAV/MP3", json.loads(body)["error"]["message"])

    def test_cleans_think_markdown_and_urls(self):
        raw = (
            "<think>внутреннее рассуждение</think> **Привет** # заголовок "
            "`код` https://example.com/страница  и [ссылка](http://x.test/y) мир"
        )
        status, _, body = self.post_tts({"text": raw})
        self.assertEqual(status, 200, body[:300])
        self.assertEqual(self.calls[-1]["text"], "Привет заголовок код и ссылка мир")

    def test_feminizes_first_person_before_voice(self):
        # Silero получает уже женские формы: «я понял» → «я поняла».
        status, _, body = self.post_tts({"text": "Я понял, я не готов."})
        self.assertEqual(status, 200, body[:300])
        self.assertEqual(self.calls[-1]["text"], "Я поняла, я не готова.")

    def test_tts_feminizes_without_plugins(self):
        # Прямой /api/tts без плагина рода: ядро применяет feminize само.
        self.server.plugins = PassThroughPlugins()
        status, _, body = self.post_tts({"text": "Я понял."})
        self.assertEqual(status, 200, body[:300])
        self.assertEqual(self.calls[-1]["text"], "Я поняла.")

    def test_tts_keeps_other_person_and_code(self):
        status, _, body = self.post_tts({"text": "Он понял, ты готов: `я понял`"})
        self.assertEqual(status, 200, body[:300])
        # Чужой род не трогаем; код в бэктиках не переписываем,
        # а из озвучки бэктики убирает чистка разметки.
        self.assertEqual(self.calls[-1]["text"], "Он понял, ты готов: я понял")

    def test_unclosed_think_is_not_spoken(self):
        status, _, _ = self.post_tts({"text": "Ответ.<think>долгое рассуждение"})
        self.assertEqual(status, 200)
        self.assertEqual(self.calls[-1]["text"], "Ответ.")

    def test_voice_is_forwarded(self):
        # Пробелы и регистр нормализуются: сервер ждёт id голоса Silero.
        status, _, _ = self.post_tts({"text": "Привет", "voice": " Kseniya "})
        self.assertEqual(status, 200)
        self.assertEqual(self.calls[-1]["voice"], "kseniya")

    def test_all_three_ui_voices_are_accepted(self):
        for voice in ("baya", "kseniya", "xenia"):
            status, _, body = self.post_tts({"text": "Привет", "voice": voice})
            self.assertEqual(status, 200, (voice, body[:200]))
            self.assertEqual(self.calls[-1]["voice"], voice)

    def test_hidden_aidar_is_accepted_by_api(self):
        # aidar скрыт из селекторов, но API и SILERO_SPEAKER его понимают.
        status, _, body = self.post_tts({"text": "Привет", "voice": "aidar"})
        self.assertEqual(status, 200, body[:200])
        self.assertEqual(self.calls[-1]["voice"], "aidar")

    def test_unknown_voice_is_400_with_list(self):
        for voice in ("irina", "soft", "young", "younger", "Milena"):
            status, _, body = self.post_tts({"text": "Привет", "voice": voice})
            self.assertEqual(status, 400, (voice, body[:200]))
            error = json.loads(body)["error"]
            self.assertEqual(error["code"], "unknown_voice")
            for allowed in ("baya", "kseniya", "xenia", "aidar"):
                self.assertIn(allowed, error["message"])
        self.assertEqual(self.calls, [])

    def test_empty_after_cleaning_is_400(self):
        status, _, body = self.post_tts({"text": "**  ##  ``  "})
        self.assertEqual(status, 400, body[:300])
        self.assertEqual(json.loads(body)["error"]["code"], "empty_text")
        self.assertEqual(self.calls, [])

    def test_too_long_is_400_and_not_synthesized(self):
        status, _, body = self.post_tts({"text": "а" * (chat_server.TTS_MAX_CHARS + 1)})
        self.assertEqual(status, 400, body[:300])
        self.assertEqual(json.loads(body)["error"]["code"], "text_too_long")
        self.assertEqual(self.calls, [])

    def test_limit_is_exactly_600_after_cleaning(self):
        status, _, body = self.post_tts({"text": "а" * chat_server.TTS_MAX_CHARS})
        self.assertEqual(status, 200, body[:300])
        self.assertEqual(len(self.calls[-1]["text"]), chat_server.TTS_MAX_CHARS)

    def test_bad_payloads_are_400(self):
        cases = (
            (["не объект"], {}),
            ({}, {}),
            ({"text": 42}, {}),
            ({"text": "Привет", "voice": 7}, {}),
            ({"text": "Привет", "voice": "я" * 65}, {}),
        )
        for payload, _ in cases:
            status, _, body = self.post_tts(payload)
            self.assertEqual(status, 400, (payload, body[:200]))

    def test_unavailable_engine_is_503_and_chat_still_works(self):
        def broken(text, voice=None):
            raise RuntimeError("нет голоса")

        self.server.tts = broken
        status, _, body = self.post_tts({"text": "Привет"})
        self.assertEqual(status, 503, body[:300])
        error = json.loads(body)["error"]
        self.assertEqual(error["code"], "voice_unavailable")
        self.assertIn("голос недоступен", error["message"])

        # Чат при этом работает: озвучка не связана со стримом.
        chat = json.dumps(make_stream_payload(), ensure_ascii=False).encode("utf-8")
        status, headers, _ = self.request(
            "POST",
            "/api/chat",
            body=chat,
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 200)
        self.assertIn("text/event-stream", headers.get("Content-Type", ""))

    def test_null_engine_empty_wav_is_503(self):
        self.server.tts = lambda text, voice=None: b""
        status, _, body = self.post_tts({"text": "Привет"})
        self.assertEqual(status, 503)
        self.assertEqual(json.loads(body)["error"]["code"], "voice_unavailable")

    def test_non_wav_bytes_are_503(self):
        self.server.tts = lambda text, voice=None: b"OggS not a wav at all"
        status, _, body = self.post_tts({"text": "Привет"})
        self.assertEqual(status, 503)
        self.assertIn("WAV", json.loads(body)["error"]["message"])

    def test_get_is_not_allowed(self):
        status, _, _ = self.request("GET", "/api/tts")
        self.assertEqual(status, 404)



class TestTtsPage(ServerTestCase):
    """Галочка «озвучивать», один <audio> и разбиение ответа на фразы."""

    def test_page_has_voice_controls(self):
        page = page_html(self)
        # Селектор — ровно три голоса Silero, baya по умолчанию (задача 0197).
        for fragment in (
            'value="baya" selected',
            'value="kseniya"',
            'value="xenia"',
        ):
            self.assertIn(fragment, page)
        for legacy in (
            'value="irina"', 'value="soft"', 'value="young"', 'value="aidar"',
        ):
            self.assertNotIn(legacy, page)
        for fragment in (
            'id="tts-0"',
            'id="tts-1"',
            'id="tts-note-0"',
            'id="tts-note-1"',
            'id="stop"',
            "озвучивать",
            "/api/tts",
            "function stopSpeech",
            "function createPhraseSplitter",
            "function enqueueSpeech",
            "function stopAll",
            "голос недоступен",
        ):
            self.assertIn(fragment, page)
        # Web Audio, а не <audio>: автоплей блокировал play() после fetch, а
        # AudioContext разблокируется один раз в жесте пользователя. Один
        # источник звука на страницу — фразы играют последовательно.
        # Только Web Audio и только Blob: <audio> и blob-URL'ы убраны совсем.
        # fetch(blob:) запрещён CSP connect-src 'self' — из-за этого звук и не шёл.
        self.assertEqual(page.count("new Audio("), 0, "элемент <audio> больше не нужен")
        self.assertEqual(page.count("createObjectURL"), 0, "blob-URL в чате не используются")
        self.assertIn("AudioContext", page)
        # Декодер вызывается из playSpeech и из лога шагов — оба в одном месте.
        self.assertGreaterEqual(page.count("decodeAudioData"), 1, "нет декодера")
        self.assertIn('byId("stop").addEventListener("click", stopAll)', page)

    def test_new_question_and_stop_clear_speech(self):
        page = page_html(self)
        # Новый вопрос глушит звук до старта генерации, «стоп» — тоже.
        self.assertIn(
            "stopSpeech();\n    controller = new AbortController();", page
        )
        self.assertIn("остановлено", page)
        self.assertIn("tts.queue = [];", page)
        # «Стоп» сначала сбрасывает буфер фраз в текст, потом глушит звук:
        # иначе хвостовая фраза зазвучала бы уже после остановки.
        self.assertIn("flushAnswer(1);\n    stopSpeech();", page)

    @unittest.skipUnless(NODE, "node не установлен")
    def test_first_phrase_ready_before_stream_end(self):
        result = run_tts_splitter(
            page_html(self),
            ["При", "вет. Как де", "ла?\nЕщё текст.", "\nБез точки"],
        )
        self.assertEqual(result["during"], ["Привет.", "Как дела?", "Ещё текст."])
        self.assertEqual(
            result["after"], ["Привет.", "Как дела?", "Ещё текст.", "Без точки"]
        )

    @unittest.skipUnless(NODE, "node не установлен")
    def test_sentence_and_newline_split(self):
        result = run_tts_splitter(page_html(self), ["А.", " Б.\nВ"])
        self.assertEqual(result["during"], ["А.", "Б."])
        self.assertEqual(result["after"], ["А.", "Б.", "В"])

    @unittest.skipUnless(NODE, "node не установлен")
    def test_long_phrase_is_cut_to_server_limit(self):
        words = " ".join(["слово"] * 300)
        result = run_tts_splitter(page_html(self), [words])
        self.assertGreater(len(result["after"]), 1)
        for phrase in result["after"]:
            self.assertLessEqual(len(phrase), chat_server.TTS_MAX_CHARS)
        self.assertEqual(" ".join(result["after"]), words)


# --------------------------------------------------------------------------- #
# Отмена генерации: стоп/обрыв/новый вопрос (задача 0185)
# --------------------------------------------------------------------------- #


class CancellableAdapter:
    """Заглушка с отменой: prefill без токенов и медленный декод.

    Повторяет поведение настоящего бэкенда в той части, которая нужна
    серверу: ``cancel_active()`` поднимает событие и возвращает ``True``
    (есть что отменять), ``stream()`` проверяет его на prefill и между
    токенами и прекращает выдачу. Как ``VtmBackend``, новая генерация
    начинается с чистого события.
    """

    def __init__(
        self,
        *,
        prefill: float = 0.05,
        delay: float = 0.03,
        pieces: tuple[str, ...] = ("ток ",) * 40,
        finish_reason: str = "stop",
    ) -> None:
        self.prefill = float(prefill)
        self.delay = float(delay)
        self.pieces = tuple(pieces)
        self.finish_reason = finish_reason
        self.kind = ""
        self.loaded = True
        self.load_stats = {}
        self.cancelled = threading.Event()
        self.started = threading.Event()
        self.cancel_calls = 0
        self.finished = 0

    def cancel_active(self) -> bool:
        if self.cancelled.is_set():
            return False
        self.cancel_calls += 1
        self.cancelled.set()
        return True

    def cancel(self) -> bool:
        return self.cancel_active()

    def stream(self, messages, *, max_tokens):
        self.cancelled.clear()
        self.started.set()
        try:
            end = time.monotonic() + self.prefill
            while time.monotonic() < end:
                if self.cancelled.is_set():
                    return
                time.sleep(0.01)
            produced = 0
            for text in self.pieces:
                if self.cancelled.is_set():
                    return
                yield chat_server.StreamPiece(text=text)
                produced += 1
                if produced >= max_tokens:
                    break
                time.sleep(self.delay)
            yield chat_server.StreamPiece(
                done=True,
                usage={
                    "prompt_tokens": 5,
                    "completion_tokens": produced,
                    "total_tokens": 5 + produced,
                },
                finish_reason=self.finish_reason,
            )
        finally:
            self.finished += 1


class TestCancellation(ServerTestCase):
    """«Стоп» и обрыв клиента останавливают генерацию, а не только чтение."""

    def setUp(self) -> None:
        super().setUp()
        self.vtm = CancellableAdapter()
        self.ollama = CancellableAdapter(pieces=("б",) * 40)
        self.server.adapters[("vtm", "mock-vtm")] = self.vtm
        self.server.adapters[("ollama", "mock-qwen")] = self.ollama

    # -- помощники ---------------------------------------------------------

    def _chat_in_thread(self, payload):
        """Запустить /api/chat в отдельном потоке; вернуть (поток, результат)."""
        box: dict[str, Any] = {}

        def run() -> None:
            try:
                status, headers, raw = self.post_chat(payload, timeout=30)
                box["status"] = status
                box["headers"] = headers
                box["raw"] = raw
            except BaseException as exc:  # noqa: BLE001 — тест проверит сам
                box["error"] = exc

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return thread, box

    def _events(self, box):
        raw = box.get("raw")
        self.assertIsNotNone(raw, f"нет ответа: {box.get('error')}")
        events, done = parse_sse(raw)
        return events, done

    def _done(self, events, side=0):
        return next(
            (event for event in events
             if event["type"] == "done" and event["side"] == side),
            None,
        )

    def _post_cancel(self, side=None):
        body = json.dumps({} if side is None else {"side": side}).encode("utf-8")
        status, _, raw = self.request(
            "POST", "/api/cancel", body=body,
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 200, raw)
        return json.loads(raw)

    # -- тесты -------------------------------------------------------------

    def test_cancel_endpoint_cancels_side_and_reports_metrics(self):
        thread, box = self._chat_in_thread(make_stream_payload())
        self.assertTrue(self.vtm.started.wait(5.0), "генерация не началась")
        self.assertTrue(self.ollama.started.wait(5.0))
        time.sleep(0.15)
        answer = self._post_cancel(side=0)
        self.assertEqual(answer["cancelled"], 1)
        thread.join(10.0)
        self.assertFalse(thread.is_alive(), "запрос не завершился после отмены")
        events, done = self._events(box)
        self.assertTrue(done)
        left = self._done(events, 0)
        self.assertIsNotNone(left)
        self.assertTrue(left["cancelled"], "колонка не отметила отмену")
        self.assertEqual(left["finish_reason"], "cancelled")
        self.assertIsNotNone(left["cancel_ms"], "нет времени до остановки")
        # Заглушка останавливается сразу: метрика — единицы мс, не секунды.
        self.assertLess(left["cancel_ms"], 100.0)
        self.assertLess(left["completion_tokens"], len(self.vtm.pieces))
        # Отменённый ответ в историю не попадает.
        self.assertEqual(self.server.history.messages(0), [])
        # Вторая колонка не тронута: она доработала до done и не отменена.
        right = self._done(events, 1)
        self.assertIsNotNone(right)
        self.assertFalse(right.get("cancelled"))
        self.assertEqual(self.ollama.cancel_calls, 0)

    def test_cancel_without_active_generation_is_noop(self):
        answer = self._post_cancel()
        self.assertEqual(answer["cancelled"], 0)
        # Следующий запрос не отменён «поздней» отменой.
        events, _headers, _raw = self.post_chat_ok(make_stream_payload())
        self.assertTrue(_raw.endswith(b"data: [DONE]\n\n") or b"[DONE]" in _raw)
        self.assertFalse(self._done(events, 0).get("cancelled"))

    def test_disconnected_client_cancels_without_waiting_for_token(self):
        # Prefill длиной 3 с: ни одного токена ещё нет, и раньше сервер узнал
        # бы об обрыве только на первой записи, т.е. после prefill (0185).
        self.vtm.prefill = 3.0
        conn = http.client.HTTPConnection(HOST, self.port, timeout=10)
        payload = json.dumps(make_stream_payload()).encode("utf-8")
        conn.request(
            "POST", "/api/chat", body=payload,
            headers={"Content-Type": "application/json"},
        )
        response = conn.getresponse()
        self.assertEqual(response.status, 200)
        self.assertTrue(self.vtm.started.wait(5.0))
        moment = time.monotonic()
        # Жёсткий RST: клиент ушёл, как закрытое окно / прерванный fetch.
        conn.sock.setsockopt(
            socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
        )
        conn.sock.close()
        conn.close()
        self.assertTrue(
            self.vtm.cancelled.wait(2.0),
            "обрыв клиента не дошёл до бэкенда за 2 с",
        )
        elapsed_ms = (time.monotonic() - moment) * 1000.0
        cancel_limit = 1000.0 if not os.environ.get("CI") else 3000.0
        self.assertLess(elapsed_ms, cancel_limit, f"отмена ждала {elapsed_ms:.0f} мс")
        # Новый вопрос стартует, не дожидаясь брошенного prefill (3 с).
        self.vtm.prefill = 0.0
        started = time.monotonic()
        events, _headers, _raw = self.post_chat_ok(make_stream_payload(question="новый"))
        self.assertIn(b"[DONE]", _raw)
        # В CI раннер медленнее: проверяем, что запрос НЕ ждёт брошенный
        # prefill (3 с), а не конкретные 2 с.
        limit = 2.0 if not os.environ.get("CI") else 5.0
        self.assertLess(
            time.monotonic() - started, limit,
            "следующий запрос ждал брошенную генерацию",
        )
        self.assertFalse(self._done(events, 0).get("cancelled"))

    def test_new_question_preempts_previous_same_adapter(self):
        # Новый вопрос на той же модели: адаптер отменяет прежнюю генерацию,
        # а не ждёт её до max_tokens (задача 0185).
        inner = CancellableAdapter(prefill=0.05, delay=0.03, pieces=("x",) * 60)
        adapter = chat_server.ChatCliAdapter(lambda: inner)
        self.server.adapters[("vtm", "mock-vtm")] = adapter

        first, box1 = self._chat_in_thread(make_stream_payload(question="первый"))
        self.assertTrue(inner.started.wait(5.0))
        started = time.monotonic()
        second, box2 = self._chat_in_thread(make_stream_payload(question="второй"))
        first.join(10.0)
        second.join(10.0)
        self.assertFalse(first.is_alive(), "первый вопрос не остановился")
        self.assertFalse(second.is_alive(), "второй вопрос не завершился")
        self.assertGreaterEqual(inner.cancel_calls, 1)
        events1, _ = self._events(box1)
        self.assertTrue(self._done(events1, 0)["cancelled"])
        events2, done2 = self._events(box2)
        self.assertTrue(done2)
        self.assertFalse(self._done(events2, 0).get("cancelled"))
        self.assertLess(time.monotonic() - started, 10.0)


if __name__ == "__main__":
    unittest.main()
