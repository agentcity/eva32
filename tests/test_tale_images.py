"""Картинки к сказкам (задача 0221): контракт, SSE и ``/media``.

Проверяем три границы:

* ``answer`` возвращает словарь ``{"text", "images"}`` только когда для
  сказки есть файлы в ``plugins/fairy-tales/images/<slug>/``;
* сервер вставляет ``image``-события между ``delta`` по ``after_chars``;
* ``GET /media/...`` отдаёт JPEG только из ``plugins/<id>/images/`` и
  отказывает на ``..``, чужой плагин и не-картинку.

Сеть не нужна: картинки уже лежат в репозитории, а плагин для SSE —
заглушка с тем же контрактом.
"""

from __future__ import annotations

import hashlib
import http.client
import importlib.util
import json
import random
import re
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eva.plugins import Permissions, PluginContext  # noqa: E402
from tools import chat_server  # noqa: E402

PLUGIN_DIR = ROOT / "plugins" / "fairy-tales"
HOST = "127.0.0.1"


def _load_plugin_module():
    path = PLUGIN_DIR / "plugin.py"
    name = "_eva_plugin_fairy_tales_images_test"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fairy = _load_plugin_module()


def make_ctx(plugin_dir: Path = PLUGIN_DIR) -> PluginContext:
    return PluginContext(
        "fairy-tales",
        plugin_dir=plugin_dir,
        permissions=Permissions(filesystem=("tales", "images")),
    )


def answer_text(answer):
    """Текст из старого строкового и нового словарного ответа."""
    if isinstance(answer, dict):
        return answer.get("text")
    return answer


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
                continue
            events.append(json.loads(payload))
    return events, done


class TestAnswerContract(unittest.TestCase):
    """Плагин: словарь с картинками, а без картинок — прежняя строка."""

    def setUp(self) -> None:
        self.plugin = fairy.FairyTalesPlugin(rng=random.Random(7))
        self.ctx = make_ctx()

    def test_kolobok_returns_text_and_images(self) -> None:
        result = self.plugin.answer(self.ctx, "Расскажи сказку про колобка")
        self.assertIsInstance(result, dict, "для сказки с картинками нужен словарь")
        assert isinstance(result, dict)
        text = result.get("text")
        images = result.get("images")
        self.assertIsInstance(text, str)
        self.assertIsInstance(images, list)
        assert isinstance(text, str) and isinstance(images, list)
        self.assertIn("Колобок", text)
        self.assertGreaterEqual(len(images), 8)
        self.assertLessEqual(len(images), 10)

        positions = []
        for image in images:
            self.assertIsInstance(image, dict)
            path = image["path"]
            self.assertIsInstance(path, str)
            self.assertTrue(path.startswith("images/kolobok/"), path)
            file = PLUGIN_DIR / path
            self.assertTrue(file.is_file(), f"нет файла {file}")
            self.assertLessEqual(file.stat().st_size, 300 * 1024)
            self.assertEqual(file.read_bytes()[:2], b"\xff\xd8", "не JPEG")
            self.assertTrue(image["alt"])
            after = image["after_chars"]
            self.assertIsInstance(after, int)
            self.assertGreater(after, 0)
            self.assertLess(after, len(text))
            positions.append(after)
        self.assertEqual(positions, sorted(positions), "картинки должны идти по тексту")
        self.assertEqual(len(positions), len(set(positions)), "позиции не повторяются")
        credits = PLUGIN_DIR / "images" / "credits.md"
        self.assertTrue(credits.is_file(), "нет images/credits.md")
        self.assertIn("Public domain", credits.read_text(encoding="utf-8"))

    def test_tale_without_images_returns_plain_string(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tales = root / "tales"
            tales.mkdir()
            (tales / "prostaya.md").write_text(
                "---\n"
                "title: Простая сказка\n"
                "tags: короткая, тест\n"
                "age: 3+\n"
                "source: локальный тест\n"
                "---\n\n"
                "Жили-были дед и баба. И была у них простая история.\n",
                encoding="utf-8",
            )
            plugin = fairy.FairyTalesPlugin(tales_dir=tales, rng=random.Random(1))
            ctx = make_ctx(root)
            result = plugin.answer(ctx, "Расскажи сказку")
            self.assertIsInstance(result, str)
            assert isinstance(result, str)
            self.assertIn("Простая сказка", result)

    def test_render_count_stays_string(self) -> None:
        result = self.plugin.answer(self.ctx, "Сколько у тебя сказок")
        self.assertIsInstance(result, str)


# --------------------------------------------------------------------------- #
# HTTP: SSE с картинками и /media
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Комплекты иллюстраций: 6–10 на сказку, credits, привязка к абзацам
# --------------------------------------------------------------------------- #

#: Сказка корпуса → каталог картинок (у «Сивко-бурко» исторический слаг
#: каталога — «сивка-бурка», его и оставляем).
TALE_IMAGE_SLUGS = {
    "kolobok": "kolobok",
    "repka": "repka",
    "teremok": "teremok",
    "lisa-i-zhuravl": "lisa-i-zhuravl",
    "tri-medvedya": "tri-medvedya",
    "gusi-lebedi": "gusi-lebedi",
    "tsarevna-lyagushka": "tsarevna-lyagushka",
    "baba-yaga": "baba-yaga",
    "sivko-burko": "sivka-burka",
}


def paragraph_ends(text: str) -> list[int]:
    """Позиции концов абзацев: там и должны стоять картинки."""
    ends: list[int] = []
    offset = 0
    for part in text.split("\n\n")[:-1]:
        offset += len(part) + 2
        ends.append(offset)
    return ends


def credits_sections() -> dict[str, str]:
    """Разделы ``credits.md`` по слагу каталога картинок."""
    raw = (PLUGIN_DIR / "images" / "credits.md").read_text(encoding="utf-8")
    sections: dict[str, str] = {}
    for chunk in re.split(r"(?m)^##\s+", raw)[1:]:
        heading, _, body = chunk.partition("\n")
        match = re.search(r"images/([a-z0-9-]+)/", heading)
        if match:
            sections[match.group(1)] = body
    return sections


class TestIllustrationSets(unittest.TestCase):
    """Каждая из девяти сказок: 6–10 картинок, credits, привязка к абзацам."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.plugin = fairy.FairyTalesPlugin(rng=random.Random(3))
        cls.ctx = make_ctx()
        cls.tales = {tale.slug: tale for tale in cls.plugin.load_tales(cls.ctx)}
        cls.credits = credits_sections()

    def test_nine_tales_have_six_to_ten_images(self) -> None:
        for slug, image_slug in TALE_IMAGE_SLUGS.items():
            with self.subTest(slug=slug):
                self.assertIn(slug, self.tales, f"нет сказки {slug}")
                tale = self.tales[slug]
                text = self.plugin.render(tale)
                images = self.plugin.images_for(tale, text, self.ctx)
                self.assertGreaterEqual(
                    len(images), 6, f"{slug}: нужно не меньше 6 картинок"
                )
                self.assertLessEqual(len(images), 10)
                files = sorted(
                    path.name
                    for path in (PLUGIN_DIR / "images" / image_slug).glob("*.jpg")
                )
                self.assertEqual(
                    [image["path"] for image in images],
                    [f"images/{image_slug}/{name}" for name in files],
                    "показываются все файлы каталога, по порядку имён",
                )
                self.assertEqual(
                    files, [f"{index:02d}.jpg" for index in range(1, len(files) + 1)]
                )

                bounds = set(paragraph_ends(text))
                positions = []
                for image in images:
                    after = image["after_chars"]
                    self.assertIsInstance(after, int)
                    self.assertGreater(after, 0)
                    self.assertLess(after, len(text))
                    self.assertIn(after, bounds, "картинка не на границе абзаца")
                    positions.append(after)
                self.assertEqual(positions, sorted(positions))
                self.assertEqual(len(positions), len(set(positions)))

                hashes = set()
                for name in files:
                    path = PLUGIN_DIR / "images" / image_slug / name
                    self.assertLessEqual(
                        path.stat().st_size,
                        300 * 1024,
                        f"{path}: больше 300 КБ",
                    )
                    self.assertEqual(path.read_bytes()[:2], b"\xff\xd8", "не JPEG")
                    digest = hashlib.sha1(path.read_bytes()).hexdigest()
                    self.assertNotIn(digest, hashes, f"{path}: дубль картинки")
                    hashes.add(digest)
                    self.assertIn(
                        f"`{name}`",
                        self.credits.get(image_slug, ""),
                        f"{path}: нет строки в credits.md",
                    )

    def test_credits_have_license_and_author(self) -> None:
        for slug, image_slug in TALE_IMAGE_SLUGS.items():
            with self.subTest(slug=slug):
                section = self.credits.get(image_slug, "")
                self.assertTrue(section, f"нет раздела credits для {image_slug}")
                self.assertTrue(
                    "Public domain" in section or "CC BY" in section,
                    f"{image_slug}: не указана лицензия",
                )
                self.assertRegex(
                    section, r"\|\s*`\d{2}\.jpg`[^\n]*\|[^\n]*\|[^\n]*\|",
                    f"{image_slug}: строки таблицы должны содержать автора и файл",
                )
                self.assertIn("commons.wikimedia.org", section)

    def test_total_size_below_25_mb(self) -> None:
        total = sum(
            path.stat().st_size
            for path in (PLUGIN_DIR / "images").glob("*/*.jpg")
        )
        self.assertLessEqual(total, 25 * 1024 * 1024, "иллюстрации тяжелее 25 МБ")

    def test_positions_spread_over_text(self) -> None:
        """Картинки идут по всей сказке, а не только в начале."""
        for slug, image_slug in TALE_IMAGE_SLUGS.items():
            with self.subTest(slug=slug):
                tale = self.tales[slug]
                text = self.plugin.render(tale)
                positions = [
                    image["after_chars"]
                    for image in self.plugin.images_for(tale, text, self.ctx)
                ]
                self.assertGreater(
                    positions[-1], len(text) // 2, "нет картинок во второй половине"
                )


class MockAdapter:
    """Заглушка бэкенда: один кусок и терминальный ``done``."""

    def __init__(self) -> None:
        self.calls = 0
        self.loaded = True

    def stream(self, messages, *, max_tokens):
        self.calls += 1
        yield chat_server.StreamPiece(text="модель не должна отвечать")
        yield chat_server.StreamPiece(
            done=True,
            usage={"prompt_tokens": 1, "completion_tokens": 1},
            finish_reason="stop",
        )


class IdentityFilter:
    def feed(self, text):
        return text

    def flush(self):
        return ""

    def __bool__(self):
        return False


class FakePlugins:
    """Минимум ``PluginRegistry``: ответы, фильтры и записи для ``/media``."""

    def __init__(self, reply, *, records=None) -> None:
        self.reply = reply
        self.records = records or {}
        self.question = ""

    def answers(self):
        def run(_ctx, question):
            self.question = question
            return self.reply

        return [
            SimpleNamespace(
                plugin_id="fairy-tales",
                run=run,
                ctx=None,
            )
        ]

    def stream_filter(self, *, scope="chat"):
        return IdentityFilter()

    def filter_text(self, text, *, scope="chat"):
        return text


REGISTRY = (
    chat_server.BackendSpec(
        id="vtm",
        label="ВТМ",
        kind="vtm",
        models=(chat_server.ModelSpec(id="mock-vtm", label="Мок"),),
    ),
    chat_server.BackendSpec(
        id="ollama",
        label="Ollama",
        kind="ollama",
        models=(chat_server.ModelSpec(id="mock-qwen", label="Мок"),),
    ),
)


class ImageServerCase(unittest.TestCase):
    def start(self, reply, records=None) -> None:
        self.adapter = MockAdapter()
        self.ollama = MockAdapter()
        self.runtime = FakePlugins(reply, records=records)
        self.server = chat_server.create_server(
            port=0,
            registry=REGISTRY,
            adapters={
                ("vtm", "mock-vtm"): self.adapter,
                ("ollama", "mock-qwen"): self.ollama,
            },
            system_prompt="",
            max_tokens=32,
            plugins=self.runtime,
            facts=False,
        )
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            kwargs={"poll_interval": 0.01},
            daemon=True,
        )
        self.thread.start()
        self.port = self.server.server_address[1]
        self.addCleanup(self._stop)

    def _stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5.0)

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection(HOST, self.port, timeout=10)
        try:
            conn.request(method, path, body=body, headers=headers or {})
            response = conn.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            conn.close()

    def chat(self, question="Расскажи сказку") -> list[dict]:
        payload = {
            "question": question,
            "use_system": False,
            "sides": [
                {"backend": "vtm", "model": "mock-vtm"},
                {"backend": "ollama", "model": "mock-qwen"},
            ],
        }
        status, headers, raw = self.request(
            "POST",
            "/api/chat",
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            {"Content-Type": "application/json"},
        )
        self.assertEqual(status, 200, raw.decode("utf-8", "replace"))
        self.assertIn("text/event-stream", headers.get("Content-Type", ""))
        events, done = parse_sse(raw)
        self.assertTrue(done)
        # Плагин отвечает в обеих колонках одинаково; проверяем левую.
        return [event for event in events if event.get("side") == 0]


class TestImageStream(ImageServerCase):
    def test_image_events_go_between_deltas(self) -> None:
        reply = {
            "text": "0123456789",
            "images": [
                {"path": "images/kolobok/01.jpg", "alt": "первая", "after_chars": 3},
                {"path": "images/kolobok/02.jpg", "alt": "вторая", "after_chars": 7},
            ],
        }
        self.start(reply)
        events = self.chat()
        self.assertEqual(
            [event["type"] for event in events],
            ["start", "delta", "image", "delta", "image", "delta", "done"],
        )
        deltas = "".join(e["text"] for e in events if e["type"] == "delta")
        self.assertEqual(deltas, "0123456789")
        self.assertEqual(
            [len(e["text"]) for e in events if e["type"] == "delta"],
            [3, 4, 3],
            "delta-куски должны резаться по after_chars",
        )
        images = [e for e in events if e["type"] == "image"]
        self.assertEqual(
            [image["url"] for image in images],
            [
                "/media/fairy-tales/kolobok/01.jpg",
                "/media/fairy-tales/kolobok/02.jpg",
            ],
        )
        self.assertEqual([image["alt"] for image in images], ["первая", "вторая"])
        self.assertTrue(all(image["side"] == 0 for image in images))
        # Ответ плагина не пошёл в модель (ни в одну колонку).
        self.assertEqual(self.adapter.calls, 0)
        self.assertEqual(self.ollama.calls, 0)
        done = [e for e in events if e["type"] == "done"][0]
        self.assertEqual(done["source"], "plugin")
        self.assertEqual(done["answer"], "0123456789")

    def test_without_images_single_delta_as_before(self) -> None:
        self.start("Просто текст без картинок.")
        events = self.chat()
        self.assertEqual(
            [event["type"] for event in events],
            ["start", "delta", "done"],
        )
        self.assertEqual(events[1]["text"], "Просто текст без картинок.")
        self.assertEqual(self.adapter.calls, 0)

    def test_unsafe_image_path_is_dropped(self) -> None:
        reply = {
            "text": "Текст",
            "images": [
                {"path": "images/../plugin.py", "alt": "плохо", "after_chars": 1},
                {"path": "/etc/passwd", "alt": "плохо", "after_chars": 1},
                {"path": "images/kolobok/01.jpg", "alt": "хорошо", "after_chars": 1},
            ],
        }
        self.start(reply)
        events = self.chat()
        images = [e for e in events if e["type"] == "image"]
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0]["url"], "/media/fairy-tales/kolobok/01.jpg")


class TestMediaEndpoint(ImageServerCase):
    def setUp(self) -> None:
        self.records = {"fairy-tales": SimpleNamespace(path=PLUGIN_DIR)}
        self.start("ok", records=self.records)

    def test_serves_jpeg_with_cache(self) -> None:
        source = PLUGIN_DIR / "images" / "kolobok" / "01.jpg"
        self.assertTrue(source.is_file())
        status, headers, body = self.request(
            "GET", "/media/fairy-tales/kolobok/01.jpg"
        )
        self.assertEqual(status, 200)
        self.assertEqual(headers.get("Content-Type"), "image/jpeg")
        self.assertEqual(headers.get("Cache-Control"), "max-age=86400")
        self.assertEqual(body, source.read_bytes())
        self.assertEqual(body[:2], b"\xff\xd8")

    def test_refuses_traversal(self) -> None:
        for path in (
            "/media/fairy-tales/../plugin.py",
            "/media/fairy-tales/%2e%2e/plugin.py",
            "/media/fairy-tales/kolobok/../../plugin.py",
            "/media/fairy-tales/..%2fplugin.py",
        ):
            with self.subTest(path=path):
                status, _, body = self.request("GET", path)
                self.assertEqual(status, 404, body)

    def test_refuses_unknown_or_foreign_plugin(self) -> None:
        status, _, _ = self.request("GET", "/media/music/kolobok/01.jpg")
        self.assertEqual(status, 404)
        status, _, _ = self.request("GET", "/media/nope/kolobok/01.jpg")
        self.assertEqual(status, 404)

    def test_refuses_non_image(self) -> None:
        status, _, _ = self.request(
            "GET", "/media/fairy-tales/kolobok/credits.md"
        )
        self.assertEqual(status, 404)
        status, _, _ = self.request(
            "GET", "/media/fairy-tales/plugin.py"
        )
        self.assertEqual(status, 404)

    def test_refuses_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin_dir = Path(tmp)
            images = plugin_dir / "images"
            images.mkdir()
            outside = plugin_dir.parent / "outside.jpg"
            outside.write_bytes(b"\xff\xd8outside")
            (images / "escape.jpg").symlink_to(outside)
            self.records["fairy-tales"] = SimpleNamespace(path=plugin_dir)
            status, _, _ = self.request(
                "GET", "/media/fairy-tales/escape.jpg"
            )
            self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()
