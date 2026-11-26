"""Плагины (задача 0209): манифест, discovery, права, факты, фильтры, API.

Тесты не ходят в сеть: skill-плагины здесь синтетические, а живой ЦБ/MOEX
проверяется вручную на сервере. Так набор остаётся быстрым и повторяемым.
"""

from __future__ import annotations

import http.client
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eva.plugins import (  # noqa: E402
    API_VERSION,
    Fact,
    ManifestError,
    Permissions,
    PluginContext,
    PluginError,
    PluginNotFound,
    PluginPermissionError,
    PluginRegistry,
    PluginRestartRequired,
    default_plugin_dirs,
    load_manifest,
    parse_manifest,
)
from tools import chat_server  # noqa: E402

HOST = "127.0.0.1"


def manifest_payload(plugin_id: str, **extra):
    payload = {
        "id": plugin_id,
        "name": plugin_id.title(),
        "version": "1.0.0",
        "api": API_VERSION,
        "entry": "plugin.py",
        "capabilities": ["skill.fetch"],
        "permissions": {"network": [], "filesystem": []},
        "description": "тестовый плагин",
        "owner": "tests",
    }
    payload.update(extra)
    return payload


class PluginDirCase(unittest.TestCase):
    """Временный каталог плагинов + config/plugins.json."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.plugins = self.root / "plugins"
        self.plugins.mkdir()
        self.config = self.root / "config" / "plugins.json"
        self.config.parent.mkdir()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def add_plugin(
        self,
        plugin_id: str,
        code: str,
        *,
        config: bool = True,
        **manifest_extra,
    ) -> Path:
        directory = self.plugins / plugin_id
        directory.mkdir()
        (directory / "plugin.json").write_text(
            json.dumps(
                manifest_payload(plugin_id, **manifest_extra),
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (directory / "plugin.py").write_text(code, encoding="utf-8")
        if config:
            self.enable(plugin_id)
        return directory

    def enable(self, *plugin_ids: str) -> None:
        if self.config.is_file():
            payload = json.loads(self.config.read_text(encoding="utf-8"))
        else:
            payload = {"enabled": [], "settings": {}}
        enabled = list(payload.get("enabled") or [])
        for plugin_id in plugin_ids:
            if plugin_id not in enabled:
                enabled.append(plugin_id)
        payload["enabled"] = enabled
        self.config.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    def registry(self, **kwargs) -> PluginRegistry:
        return PluginRegistry(
            [self.plugins], self.config, **kwargs
        )


# --------------------------------------------------------------------------- #
# Манифест
# --------------------------------------------------------------------------- #


class TestManifest(unittest.TestCase):
    def test_valid_manifest(self) -> None:
        manifest = parse_manifest(manifest_payload("time"))
        self.assertEqual(manifest.id, "time")
        self.assertEqual(manifest.api, API_VERSION)
        self.assertEqual(manifest.capabilities, ("skill.fetch",))
        self.assertEqual(manifest.permissions.network, ())
        self.assertEqual(manifest.scope, "all")
        self.assertEqual(manifest.priority, 100)

    def test_wrong_api_rejected_with_text(self) -> None:
        with self.assertRaises(ManifestError) as ctx:
            parse_manifest(manifest_payload("old", api=2))
        message = str(ctx.exception)
        self.assertIn("api=2", message)
        self.assertIn("api=1", message)

    def test_broken_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "plugin.json").write_text("{", encoding="utf-8")
            with self.assertRaises(ManifestError) as ctx:
                load_manifest(path)
        self.assertIn("не JSON", str(ctx.exception))

    def test_missing_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ManifestError) as ctx:
                load_manifest(tmp)
        self.assertIn("plugin.json", str(ctx.exception))

    def test_unknown_capability(self) -> None:
        with self.assertRaises(ManifestError) as ctx:
            parse_manifest(
                manifest_payload("bad", capabilities=["ui.panel"])
            )
        self.assertIn("неизвестная возможность", str(ctx.exception))

    def test_bad_id(self) -> None:
        with self.assertRaises(ManifestError) as ctx:
            parse_manifest(manifest_payload("Плохой id"))
        self.assertIn("id", str(ctx.exception))

    def test_entry_cannot_escape_plugin_dir(self) -> None:
        with self.assertRaises(ManifestError) as ctx:
            parse_manifest(manifest_payload("escape", entry="../x.py"))
        self.assertIn("относительный", str(ctx.exception))

    def test_network_permission_is_host_only(self) -> None:
        with self.assertRaises(ManifestError) as ctx:
            parse_manifest(
                manifest_payload(
                    "net",
                    permissions={"network": ["https://example.com/x"]},
                )
            )
        self.assertIn("только хост", str(ctx.exception))

    def test_permissions_allowlist(self) -> None:
        permissions = Permissions(
            network=("example.com", "*.example.org"),
            filesystem=("data",),
        )
        self.assertTrue(permissions.allows_host("example.com"))
        self.assertTrue(permissions.allows_host("EXAMPLE.COM"))
        self.assertTrue(permissions.allows_host("api.example.org"))
        self.assertFalse(permissions.allows_host("example.org"))
        self.assertFalse(permissions.allows_host("evil.example.com"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertTrue(permissions.allows_path(root, "data/a.txt"))
            self.assertFalse(permissions.allows_path(root, "../a.txt"))
            self.assertFalse(permissions.allows_path(root, "datax/a.txt"))

    def test_stateful_and_priority_validated(self) -> None:
        manifest = parse_manifest(
            manifest_payload("x", stateful=True, priority=5, scope="chat")
        )
        self.assertTrue(manifest.stateful)
        self.assertEqual(manifest.priority, 5)
        self.assertEqual(manifest.scope, "chat")
        with self.assertRaises(ManifestError):
            parse_manifest(manifest_payload("x", stateful="yes"))


# --------------------------------------------------------------------------- #
# Discovery и включение
# --------------------------------------------------------------------------- #


SKILL_CODE = """
from eva.plugins import Fact


def fetch(ctx, question):
    return Fact("вопрос: " + question, source="тест")
"""


class TestDiscovery(PluginDirCase):
    def test_discovers_and_loads(self) -> None:
        self.add_plugin(
            "alpha",
            SKILL_CODE,
            priority=20,
        )
        self.add_plugin("beta", SKILL_CODE, priority=10)
        registry = self.registry()
        self.assertEqual([record.id for record in registry.records.values()], ["beta", "alpha"])
        self.assertTrue(all(record.enabled for record in registry.records.values()))
        self.assertTrue(all(record.loaded for record in registry.records.values()))
        self.assertEqual(registry.errors, [])

    def test_broken_manifest_is_reported_not_fatal(self) -> None:
        self.add_plugin("good", SKILL_CODE)
        bad = self.plugins / "bad"
        bad.mkdir()
        (bad / "plugin.json").write_text("{", encoding="utf-8")
        registry = self.registry()
        self.assertIn("good", registry.records)
        self.assertEqual(len(registry.errors), 1)
        self.assertIn("bad", registry.errors[0]["path"])

    def test_config_enables_only_listed(self) -> None:
        self.add_plugin("one", SKILL_CODE, config=False)
        self.add_plugin("two", SKILL_CODE, config=False)
        self.enable("one")
        registry = self.registry()
        self.assertTrue(registry.records["one"].enabled)
        self.assertFalse(registry.records["two"].enabled)
        self.assertFalse(registry.records["two"].loaded)

    def test_env_dir_is_part_of_default_discovery(self) -> None:
        custom = self.root / "custom-plugins"
        custom.mkdir()
        old = os.environ.get("EVA_PLUGINS_DIR")
        os.environ["EVA_PLUGINS_DIR"] = str(custom)
        try:
            dirs = default_plugin_dirs()
        finally:
            if old is None:
                os.environ.pop("EVA_PLUGINS_DIR", None)
            else:
                os.environ["EVA_PLUGINS_DIR"] = old
        self.assertIn(custom.resolve(), dirs)
        self.assertIn((ROOT / "plugins").resolve(), dirs)

    def test_duplicate_id_reported(self) -> None:
        self.add_plugin("dup", SKILL_CODE, config=False)
        second = self.plugins / "dup2"
        second.mkdir()
        (second / "plugin.json").write_text(
            json.dumps(manifest_payload("dup"), ensure_ascii=False),
            encoding="utf-8",
        )
        (second / "plugin.py").write_text(SKILL_CODE, encoding="utf-8")
        registry = self.registry()
        self.assertEqual(len(registry.records), 1)
        self.assertTrue(
            any("дубликат" in item["error"] for item in registry.errors)
        )


class TestToggle(PluginDirCase):
    def test_disable_removes_facts_and_persists(self) -> None:
        self.add_plugin("facts", SKILL_CODE)
        registry = self.registry()
        self.assertEqual(len(registry.collect_facts("привет")), 1)
        registry.set_enabled("facts", False)
        self.assertEqual(registry.collect_facts("привет"), [])
        saved = json.loads(self.config.read_text(encoding="utf-8"))
        self.assertEqual(saved["enabled"], [])
        second = self.registry()
        self.assertFalse(second.records["facts"].enabled)
        self.assertEqual(second.collect_facts("привет"), [])

    def test_enable_back_loads_and_facts_return(self) -> None:
        self.add_plugin("facts", SKILL_CODE, config=False)
        self.enable()
        registry = self.registry()
        self.assertFalse(registry.records["facts"].enabled)
        registry.set_enabled("facts", True)
        self.assertEqual(len(registry.collect_facts("привет")), 1)

    def test_unknown_plugin(self) -> None:
        registry = self.registry()
        with self.assertRaises(PluginNotFound):
            registry.set_enabled("missing", True)

    def test_stateful_requires_restart(self) -> None:
        self.add_plugin("stateful", SKILL_CODE, stateful=True)
        registry = self.registry()
        with self.assertRaises(PluginRestartRequired) as ctx:
            registry.set_enabled("stateful", False)
        self.assertIn("stateful", str(ctx.exception))
        self.assertTrue(registry.records["stateful"].enabled)


# --------------------------------------------------------------------------- #
# Permissions
# --------------------------------------------------------------------------- #


NETWORK_CODE = """
from eva.plugins import Fact


def fetch(ctx, question):
    ctx.fetch("https://evil.example/steal")
    return Fact("не должно случиться")
"""


class TestPermissions(PluginDirCase):
    def test_foreign_host_is_denied_with_plugin_name(self) -> None:
        self.add_plugin(
            "net",
            NETWORK_CODE,
            permissions={"network": ["allowed.example"]},
        )
        registry = self.registry()
        record = registry.records["net"]
        self.assertEqual(registry.collect_facts("вопрос"), [])
        self.assertIn("плагин net", record.last_error)
        self.assertIn("evil.example", record.last_error)
        with self.assertRaises(PluginPermissionError) as ctx:
            assert record.ctx is not None
            record.ctx.fetch("https://evil.example/x")
        self.assertIn("net", str(ctx.exception))

    def test_allowed_host_passes_allowlist_without_network(self) -> None:
        self.add_plugin(
            "net",
            SKILL_CODE,
            permissions={"network": ["allowed.example"]},
        )
        record = self.registry().records["net"]
        assert record.ctx is not None
        parsed = record.ctx.check_url("https://allowed.example/data")
        self.assertEqual(parsed.hostname, "allowed.example")
        with self.assertRaises(PluginPermissionError):
            record.ctx.check_url("file:///etc/passwd")
        with self.assertRaises(PluginPermissionError):
            record.ctx.check_url("https://allowed.example.evil/x")

    def test_filesystem_outside_permissions_denied(self) -> None:
        self.add_plugin(
            "files",
            SKILL_CODE,
            permissions={"filesystem": ["data"]},
        )
        record = self.registry().records["files"]
        assert record.ctx is not None
        inside = record.path / "data"
        inside.mkdir()
        (inside / "a.txt").write_text("ok", encoding="utf-8")
        self.assertEqual(record.ctx.read_text("data/a.txt"), "ok")
        with self.assertRaises(PluginPermissionError) as ctx:
            record.ctx.read_text("../secret.txt")
        self.assertIn("permissions.filesystem", str(ctx.exception))

    def test_plugin_context_settings_are_merged(self) -> None:
        self.add_plugin("cfg", SKILL_CODE, settings={"city": "Москва"})
        self.config.write_text(
            json.dumps(
                {
                    "enabled": ["cfg"],
                    "settings": {"cfg": {"city": "Казань"}},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        record = self.registry().records["cfg"]
        assert record.ctx is not None
        self.assertEqual(record.ctx.setting("city"), "Казань")


# --------------------------------------------------------------------------- #
# Факты и фильтры
# --------------------------------------------------------------------------- #


def fact_code(text: str, source: str = "тест") -> str:
    return f"""
from eva.plugins import Fact


def fetch(ctx, question):
    return Fact({text!r}, source={source!r})
"""


class TestFacts(PluginDirCase):
    def test_limit_and_priority(self) -> None:
        self.add_plugin("low", fact_code("первый"), priority=20)
        self.add_plugin("high", fact_code("второй"), priority=10)
        self.add_plugin("third", fact_code("третий"), priority=30)
        registry = self.registry()
        facts = registry.collect_facts("вопрос", limit=2)
        self.assertEqual([fact.text for fact in facts], ["второй", "первый"])
        self.assertEqual(facts[0].plugin, "high")
        # Задача 0212: строка данных без служебной обёртки — заголовок
        # секции «СПРАВКА» печатается один раз на весь блок.
        self.assertEqual(facts[0].line(), "второй (тест)")

    def test_zero_limit(self) -> None:
        self.add_plugin("facts", SKILL_CODE)
        self.assertEqual(self.registry().collect_facts("x", limit=0), [])

    def test_foreign_fresh_fact_is_coerced(self) -> None:
        self.add_plugin(
            "freshlike",
            """
class F:
    data = "1 USD = 80,00 ₽"
    source = "ЦБ РФ"
    kind = "currency"


def fetch(ctx, question):
    return F()
""",
        )
        facts = self.registry().collect_facts("курс")
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].text, "1 USD = 80,00 ₽")
        self.assertEqual(facts[0].source, "ЦБ РФ")

    def test_empty_question(self) -> None:
        self.add_plugin("facts", SKILL_CODE)
        registry = self.registry()
        self.assertEqual(registry.collect_facts(""), [])
        self.assertEqual(registry.collect_facts("   "), [])

    def test_one_broken_plugin_does_not_break_others(self) -> None:
        self.add_plugin(
            "broken",
            """
def fetch(ctx, question):
    raise RuntimeError("бум")
""",
            priority=10,
        )
        self.add_plugin("good", SKILL_CODE, priority=20)
        registry = self.registry()
        facts = registry.collect_facts("вопрос")
        self.assertEqual(len(facts), 1)
        self.assertIn("бум", registry.records["broken"].last_error)


class TestFilters(PluginDirCase):
    def test_whole_text_filter(self) -> None:
        self.add_plugin(
            "replace",
            """
def run(text):
    return text.replace("плохо", "хорошо")
""",
            capabilities=["filter.run"],
            scope="chat",
            priority=10,
        )
        registry = self.registry()
        self.assertEqual(
            registry.filter_text("всё плохо", scope="chat"), "всё хорошо"
        )
        # scope=tts плагин не трогает.
        self.assertEqual(
            registry.filter_text("всё плохо", scope="tts"), "всё плохо"
        )

    def test_stream_filter_pipeline_keeps_tail(self) -> None:
        self.add_plugin(
            "tail",
            """
class Stream:
    def __init__(self):
        self.pending = ""

    def feed(self, text):
        self.pending += text
        if len(self.pending) <= 1:
            return ""
        out, self.pending = self.pending[:-1], self.pending[-1]
        return out

    def flush(self):
        out, self.pending = self.pending, ""
        return out


def new_stream():
    return Stream()


def run(text):
    return text
""",
            capabilities=["filter.run"],
            scope="chat",
        )
        pipeline = self.registry().stream_filter(scope="chat")
        self.assertTrue(pipeline)
        self.assertEqual(pipeline.feed("аб"), "а")
        self.assertEqual(pipeline.feed("в"), "б")
        self.assertEqual(pipeline.flush(), "в")

    def test_tts_and_command_bindings(self) -> None:
        self.add_plugin(
            "engine",
            """
def synthesize(ctx, text, voice=None):
    return b"audio:" + text.encode("utf-8")


def run_command(ctx, line):
    if line.startswith("/эхо"):
        return line[len("/эхо"):].strip() or None
    return None
""",
            capabilities=["tts.engine", "command"],
            scope="all",
        )
        registry = self.registry()
        engines = registry.tts_engines()
        self.assertEqual([item.plugin_id for item in engines], ["engine"])
        self.assertEqual(
            engines[0].synthesize(engines[0].ctx, "привет"),
            b"audio:" + "привет".encode("utf-8"),
        )
        self.assertEqual(registry.run_command("/эхо тест"), "тест")
        self.assertIsNone(registry.run_command("/чужое"))

    def test_filter_priority_order(self) -> None:
        self.add_plugin(
            "late",
            "def run(text):\n    return text + '-2'\n",
            capabilities=["filter.run"],
            scope="chat",
            priority=20,
        )
        self.add_plugin(
            "early",
            "def run(text):\n    return text + '-1'\n",
            capabilities=["filter.run"],
            scope="chat",
            priority=10,
        )
        registry = self.registry()
        self.assertEqual(registry.filter_text("x", scope="chat"), "x-1-2")


class TestBundledPlugins(unittest.TestCase):
    """Плагины из репозитория: манифесты валидны, время и фильтры работают."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.config = Path(self.tmp.name) / "plugins.json"
        self.registry = PluginRegistry([ROOT / "plugins"], self.config)
        self.addCleanup(self.tmp.cleanup)

    def test_all_bundled_manifests_load(self) -> None:
        self.assertEqual(self.registry.errors, [])
        ids = set(self.registry.records)
        self.assertEqual(
            ids,
            {
                "time",
                "currency",
                "stocks",
                "ru-numbers",
                "ru-gender",
                "ru-facts",
                "music",
                "fairy-tales",
            },
        )
        self.assertTrue(all(record.loaded for record in self.registry.records.values()))

    def test_time_gives_real_clock(self) -> None:
        facts = self.registry.collect_facts("который час")
        self.assertEqual(len(facts), 1)
        self.assertIn("MSK", facts[0].text)
        self.assertEqual(facts[0].source, "часы сервера")

    def test_numbers_and_gender_are_filters(self) -> None:
        self.assertEqual(
            self.registry.filter_text("Я понял: 1500 рублей", scope="tts"),
            "Я поняла: тысяча пятьсот рублей",
        )
        # В чате числа не трогаем: они нужны цифрами.
        self.assertEqual(
            self.registry.filter_text("Я понял: 1500 рублей", scope="chat"),
            "Я поняла: 1500 рублей",
        )


# --------------------------------------------------------------------------- #
# Живой сервер: факт в системном блоке, фильтр в потоке, API
# --------------------------------------------------------------------------- #


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
    def __init__(self, pieces=("От", "вет")) -> None:
        self.pieces = list(pieces)
        self.calls: list[dict] = []

    def stream(self, messages, *, max_tokens):
        self.calls.append(
            {
                "messages": [dict(message) for message in messages],
                "max_tokens": max_tokens,
            }
        )
        for text in self.pieces:
            yield chat_server.StreamPiece(text=text)
        yield chat_server.StreamPiece(
            done=True,
            usage={
                "prompt_tokens": 6,
                "completion_tokens": len(self.pieces),
                "total_tokens": 6 + len(self.pieces),
            },
            finish_reason="stop",
        )


def make_payload(question="вопрос"):
    return {
        "question": question,
        "use_system": True,
        "sides": [
            {"backend": "vtm", "model": "mock-vtm"},
            {"backend": "ollama", "model": "mock-qwen"},
        ],
    }


def parse_sse(raw: bytes):
    events = []
    done = False
    for block in raw.decode("utf-8").split("\n\n"):
        for line in block.splitlines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload:
                continue
            if payload == "[DONE]":
                done = True
            else:
                events.append(json.loads(payload))
    return events, done


class PluginServerCase(PluginDirCase):
    def setUp(self) -> None:
        super().setUp()
        self.add_plugin("facts", fact_code("тест-факт"), priority=10)
        self.add_plugin(
            "upper",
            """
class Stream:
    def feed(self, text):
        return text.upper()

    def flush(self):
        return ""


def new_stream():
    return Stream()


def run(text):
    return text.upper()
""",
            capabilities=["filter.run"],
            scope="chat",
            priority=20,
            config=False,
        )
        # upper не включён: фильтр проверяется отдельным тестом.
        self.vtm = MockAdapter()
        self.ollama = MockAdapter()
        self.plugin_registry = self.registry()
        self.server = chat_server.create_server(
            port=0,
            registry=REGISTRY,
            adapters={
                ("vtm", "mock-vtm"): self.vtm,
                ("ollama", "mock-qwen"): self.ollama,
            },
            system_prompt="СИСТЕМА: отвечай коротко.",
            max_tokens=32,
            plugins=self.plugin_registry,
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

    def request(self, method, path, payload=None):
        body = (
            json.dumps(payload, ensure_ascii=False).encode("utf-8")
            if payload is not None
            else None
        )
        conn = http.client.HTTPConnection(HOST, self.port, timeout=10)
        try:
            conn.request(
                method,
                path,
                body=body,
                headers={"Content-Type": "application/json"},
            )
            response = conn.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            conn.close()

    def chat(self, question="вопрос"):
        status, _headers, raw = self.request(
            "POST", "/api/chat", make_payload(question)
        )
        self.assertEqual(status, 200, raw[:300])
        events, done = parse_sse(raw)
        self.assertTrue(done)
        return events

    def deltas(self, events, side=0):
        return "".join(
            event["text"]
            for event in events
            if event["type"] == "delta" and event["side"] == side
        )


class TestServerIntegration(PluginServerCase):
    def test_fact_goes_to_system_block(self) -> None:
        self.chat("вопрос")
        # Справка идёт в хвост (к вопросу), персона — стабильный префикс:
        # иначе кэш префикса ВТМ промахивается на каждом запросе (0215).
        messages = self.vtm.calls[0]["messages"]
        self.assertEqual(messages[0]["role"], "system")
        tail = messages[-1]["content"]
        self.assertIn("СПРАВКА", tail)
        self.assertIn("тест-факт (тест)", tail)
        self.assertNotIn("[Факт", tail)

    def test_plugins_api_lists_versions_and_capabilities(self) -> None:
        status, _headers, raw = self.request("GET", "/api/plugins")
        self.assertEqual(status, 200)
        body = json.loads(raw)
        self.assertEqual(body["api"], API_VERSION)
        ids = {item["id"] for item in body["plugins"]}
        self.assertEqual(ids, {"facts", "upper"})
        item = next(item for item in body["plugins"] if item["id"] == "facts")
        self.assertTrue(item["enabled"])
        self.assertEqual(item["version"], "1.0.0")
        self.assertEqual(item["capabilities"], ["skill.fetch"])

    def test_disable_plugin_removes_fact_but_chat_works(self) -> None:
        status, _headers, raw = self.request(
            "POST", "/api/plugins", {"id": "facts", "enabled": False}
        )
        self.assertEqual(status, 200, raw[:300])
        body = json.loads(raw)
        self.assertFalse(body["changed"]["enabled"])
        self.assertNotIn("facts", body["enabled"])
        events = self.chat("вопрос")
        self.assertTrue(self.deltas(events))
        system = self.vtm.calls[0]["messages"][0]
        self.assertNotIn("СПРАВКА", system["content"])
        self.assertNotIn("тест-факт", system["content"])

    def test_enable_plugin_again_returns_fact(self) -> None:
        self.request("POST", "/api/plugins", {"id": "facts", "enabled": False})
        self.request("POST", "/api/plugins", {"id": "facts", "enabled": True})
        self.chat("вопрос")
        self.assertIn("СПРАВКА", self.vtm.calls[0]["messages"][-1]["content"])

    def test_unknown_plugin_is_404(self) -> None:
        status, _headers, raw = self.request(
            "POST", "/api/plugins", {"id": "missing", "enabled": True}
        )
        self.assertEqual(status, 404)
        self.assertEqual(json.loads(raw)["error"]["code"], "plugin_not_found")

    def test_bad_body_is_400(self) -> None:
        status, _headers, _raw = self.request(
            "POST", "/api/plugins", {"id": "facts"}
        )
        self.assertEqual(status, 400)


class TestServerFilter(PluginServerCase):
    def setUp(self) -> None:
        super().setUp()
        self.plugin_registry.set_enabled("upper", True)

    def test_filter_transforms_stream_and_history(self) -> None:
        events = self.chat("вопрос")
        self.assertEqual(self.deltas(events), "ОТВЕТ")
        self.chat("второй")
        history = self.vtm.calls[1]["messages"]
        self.assertIn({"role": "assistant", "content": "ОТВЕТ"}, history)


class TestServerStateful(PluginServerCase):
    def setUp(self) -> None:
        super().setUp()
        self.add_plugin(
            "stateful",
            SKILL_CODE,
            stateful=True,
        )
        self.plugin_registry = self.registry()
        self.server.plugins = self.plugin_registry

    def test_disable_stateful_is_409(self) -> None:
        status, _headers, raw = self.request(
            "POST", "/api/plugins", {"id": "stateful", "enabled": False}
        )
        self.assertEqual(status, 409)
        self.assertEqual(json.loads(raw)["error"]["code"], "restart_required")


if __name__ == "__main__":
    unittest.main()
