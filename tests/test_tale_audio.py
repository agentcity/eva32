"""Готовая озвучка сказок (задача 0223): запись вместо пересказа.

Проверяем четыре границы:

* фразы «включи запись сказки», «поставь аудиосказку», «сказку в
  исполнении» запускают локальный файл через ``afplay`` (список argv, без
  shell, ``start_new_session``), а «выключи»/«стоп сказку» — завершают;
* если записи нет, плагин честно отвечает и не падает;
* ``silent: true`` доходит до клиента в SSE ``delta``/``done``, чтобы Ева не
  говорила поверх записи;
* в ``audio/credits.md`` есть автор, источник и свободная лицензия, а файлы
  не больше 3 МБ.

Сеть не нужна: файлы записей лежат в репозитории, ``subprocess.Popen``
подменяется моком, а сервер поднимается на свободном порту loopback.
"""

from __future__ import annotations

import http.client
import importlib.util
import json
import random
import re
import signal
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eva.plugins import Permissions, PluginContext  # noqa: E402
from tools import chat_server  # noqa: E402

PLUGIN_DIR = ROOT / "plugins" / "fairy-tales"
AUDIO_DIR = PLUGIN_DIR / "audio"
HOST = "127.0.0.1"

#: Записи, которые лежат в репозитории (см. audio/credits.md).
RECORDINGS = {
    "baba-yaga": "Баба-яга",
    "kot-petuh-i-lisa": "Кот, петух и лиса",
    "teremok": "Теремок",
    "tsarevna-lyagushka": "Царевна-лягушка",
}

START_PHRASES = (
    "Включи запись сказки",
    "Поставь аудиосказку",
    "Сказку в исполнении",
    "Включи аудио сказку",
    "Поставь запись сказки",
    "Включи сказку в исполнении",
)


def _load_plugin_module():
    path = PLUGIN_DIR / "plugin.py"
    name = "_eva_plugin_fairy_tales_audio_test"
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
        permissions=Permissions(filesystem=("tales", "images", "audio")),
    )


def answer_text(answer: object) -> str | None:
    if isinstance(answer, dict):
        value = answer.get("text")
        return value if isinstance(value, str) else None
    return answer if isinstance(answer, str) else None


class FakeProcess:
    """Мок ``Popen``: умеет poll/terminate/wait/kill и запоминает сигналы."""

    def __init__(self, *, exit_code: int | None = None) -> None:
        self.exit_code = exit_code
        self.terminated = False
        self.killed = False
        self.waited = False
        self.signals: list[int] = []

    def poll(self):
        return self.exit_code

    def send_signal(self, sig: int) -> None:
        self.signals.append(sig)

    def terminate(self) -> None:
        self.terminated = True
        self.exit_code = 0

    def wait(self, timeout=None):
        self.waited = True
        return 0

    def kill(self) -> None:
        self.killed = True
        self.exit_code = -9


class AudioCase(unittest.TestCase):
    def setUp(self) -> None:
        self.plugin = fairy.FairyTalesPlugin(rng=random.Random(7))
        self.ctx = make_ctx()

    def start(self, phrase: str, *, exit_code: int | None = None):
        process = FakeProcess(exit_code=exit_code)
        with mock.patch.object(
            fairy.subprocess, "Popen", return_value=process
        ) as popen, mock.patch.object(fairy.time, "sleep"), mock.patch.object(
            fairy.FairyTalesPlugin, "_system_volume", return_value=None
        ), mock.patch.object(
            fairy.FairyTalesPlugin, "_set_system_volume", return_value=False
        ):
            result = self.plugin.answer(self.ctx, phrase)
        return result, popen, process


class TestAudioRecognition(AudioCase):
    def test_start_phrases_play_local_file(self) -> None:
        for phrase in START_PHRASES:
            with self.subTest(phrase=phrase):
                self.plugin = fairy.FairyTalesPlugin(rng=random.Random(7))
                result, popen, _ = self.start(phrase)
                self.assertIsInstance(result, dict, result)
                assert isinstance(result, dict)
                self.assertTrue(result.get("silent"), "запись уже озвучена")
                text = answer_text(result)
                assert text is not None
                self.assertIn("Включаю запись", text)
                self.assertIn("«", text)
                popen.assert_called_once()
                argv, kwargs = popen.call_args
                self.assertEqual(argv[0][0], fairy.AFPLAY)
                path = Path(argv[0][1])
                self.assertEqual(path.suffix, ".mp3")
                self.assertEqual(path.parent, AUDIO_DIR)
                self.assertTrue(path.is_file())
                self.assertNotIn("shell", kwargs)
                self.assertTrue(kwargs.get("start_new_session"))
                self.assertEqual(kwargs.get("stdout"), fairy.subprocess.DEVNULL)
                self.assertEqual(kwargs.get("stderr"), fairy.subprocess.DEVNULL)

    def test_named_tale_picks_matching_recording(self) -> None:
        result, popen, _ = self.start("Включи запись сказки про бабу-ягу")
        assert isinstance(result, dict)
        text = answer_text(result)
        assert text is not None
        self.assertIn("Баба-яга", text)
        argv, _ = popen.call_args
        self.assertEqual(Path(argv[0][1]).stem, "baba-yaga")

    def test_pro_kota_prefers_tale_with_recording(self) -> None:
        # «Кот и лиса» короче и без записи; запись есть у «Кот, петух и лиса».
        result, popen, _ = self.start("Включи запись сказки про кота")
        assert isinstance(result, dict)
        text = answer_text(result)
        assert text is not None
        self.assertIn("Кот, петух и лиса", text)
        argv, _ = popen.call_args
        self.assertEqual(Path(argv[0][1]).stem, "kot-petuh-i-lisa")

    def test_two_columns_ask_same_tale_start_one_process(self) -> None:
        # Сервер зовёт answer для каждой колонки: два одинаковых запроса не
        # должны запускать два afplay, иначе «выключи» убьёт только один.
        process = FakeProcess()
        with mock.patch.object(
            fairy.subprocess, "Popen", return_value=process
        ) as popen, mock.patch.object(fairy.time, "sleep"), mock.patch.object(
            fairy.FairyTalesPlugin, "_system_volume", return_value=None
        ):
            first = self.plugin.answer(self.ctx, "Включи запись сказки про теремок")
            second = self.plugin.answer(self.ctx, "Включи запись сказки про теремок")
        self.assertIsInstance(first, dict)
        self.assertIsInstance(second, dict)
        popen.assert_called_once()

    def test_start_stops_previous_recording(self) -> None:
        first = FakeProcess()
        second = FakeProcess()
        with mock.patch.object(
            fairy.subprocess, "Popen", side_effect=[first, second]
        ), mock.patch.object(fairy.time, "sleep"), mock.patch.object(
            fairy.FairyTalesPlugin, "_system_volume", return_value=None
        ):
            self.plugin.answer(self.ctx, "Включи запись сказки про теремок")
            self.plugin.answer(self.ctx, "Включи запись сказки про бабу-ягу")
        self.assertTrue(first.terminated, "старая запись должна быть остановлена")
        self.assertTrue(first.waited)


class TestAudioStop(AudioCase):
    def test_bare_stop_terminates_active_recording(self) -> None:
        _, _, process = self.start("Включи запись сказки про теремок")
        self.assertFalse(process.terminated)
        with mock.patch.object(fairy.time, "sleep"):
            result = self.plugin.answer(self.ctx, "Выключи")
        self.assertEqual(result, "Выключила запись.")
        self.assertTrue(process.terminated)
        self.assertTrue(process.waited)
        self.assertIn(signal.SIGCONT, process.signals)

    def test_stop_skazku_without_recording_is_honest(self) -> None:
        result = self.plugin.answer(self.ctx, "Стоп сказку")
        self.assertEqual(result, "Запись сейчас не играет.")

    def test_both_columns_get_same_stop_answer(self) -> None:
        # Первая колонка гасит процесс, вторая приходит почти сразу: она
        # должна сказать то же самое, а не «Запись сейчас не играет»/модель.
        self.start("Включи запись сказки про теремок")
        first = self.plugin.answer(self.ctx, "Выключи")
        second = self.plugin.answer(self.ctx, "Выключи")
        self.assertEqual(first, "Выключила запись.")
        self.assertEqual(second, "Выключила запись.")

    def test_bare_stop_without_recording_falls_through(self) -> None:
        # Без активной/недавней записи «выключи» — не наша команда.
        self.assertIsNone(self.plugin.answer(self.ctx, "Выключи"))

    def test_stop_does_not_take_music_command(self) -> None:
        # «выключи музыку» — зона music; без активной записи сказка молчит.
        self.assertIsNone(self.plugin.answer(self.ctx, "Выключи музыку"))
        self.assertIsNone(self.plugin.answer(self.ctx, "Пауза"))


class TestNoRecording(AudioCase):
    def test_tale_without_recording_is_honest(self) -> None:
        result, popen, _ = self.start("Включи запись сказки про колобка")
        self.assertIsInstance(result, str)
        assert isinstance(result, str)
        self.assertIn("Записи", result)
        self.assertIn("могу рассказать сама", result)
        popen.assert_not_called()

    def test_empty_audio_dir_is_honest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin = fairy.FairyTalesPlugin(
                audio_dir=tmp, rng=random.Random(1)
            )
            result = plugin.answer(self.ctx, "Поставь аудиосказку")
            self.assertIsInstance(result, str)
            assert isinstance(result, str)
            self.assertIn("нет", result)
            self.assertIn("могу рассказать сама", result)

    def test_file_over_limit_is_not_played(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "teremok.mp3"
            with open(path, "wb") as handle:
                handle.truncate(fairy.AUDIO_MAX_BYTES + 1)
            plugin = fairy.FairyTalesPlugin(
                audio_dir=tmp, rng=random.Random(1)
            )
            with mock.patch.object(fairy.subprocess, "Popen") as popen:
                result = plugin.answer(self.ctx, "Включи запись сказки про теремок")
            popen.assert_not_called()
            self.assertIsInstance(result, str)
            assert isinstance(result, str)
            self.assertIn("могу рассказать сама", result)

    def test_missing_afplay_is_honest(self) -> None:
        with mock.patch.object(
            fairy.subprocess, "Popen", side_effect=FileNotFoundError("afplay")
        ), mock.patch.object(fairy.time, "sleep"):
            result = self.plugin.answer(self.ctx, "Включи запись сказки про теремок")
        self.assertIsInstance(result, str)
        assert isinstance(result, str)
        self.assertIn("afplay", result)

    def test_immediate_exit_is_honest(self) -> None:
        result, _, process = self.start(
            "Включи запись сказки про теремок", exit_code=1
        )
        self.assertIsInstance(result, str)
        assert isinstance(result, str)
        self.assertIn("не открылся", result)
        self.assertIsNone(self.plugin._audio_process)


class TestCreditsAndFiles(unittest.TestCase):
    """Лицензии и размеры: записи должны быть свободными и лёгкими."""

    def setUp(self) -> None:
        self.credits_path = AUDIO_DIR / "credits.md"
        self.credits = (
            self.credits_path.read_text(encoding="utf-8")
            if self.credits_path.is_file()
            else ""
        )

    def test_credits_exist_and_cover_every_recording(self) -> None:
        self.assertTrue(self.credits_path.is_file(), "нет audio/credits.md")
        self.assertGreater(len(self.credits.strip()), 200)
        for slug, title in RECORDINGS.items():
            with self.subTest(slug=slug):
                self.assertIn(slug, self.credits)
                self.assertIn(title, self.credits)

    def test_licenses_are_free(self) -> None:
        low = self.credits.lower()
        self.assertNotIn("all rights reserved", low)
        self.assertNotIn("все права защищены", low)
        self.assertTrue(
            "publicdomain" in low
            or "cc0" in low
            or "cc-by" in low
            or "public domain" in low
            or "общественное достояние" in low,
            "в credits.md нет свободной лицензии",
        )
        self.assertIn("creativecommons.org/publicdomain", low)

    def test_files_are_small_and_named_by_slug(self) -> None:
        files = sorted(AUDIO_DIR.glob("*"))
        audio = [path for path in files if path.suffix in (".mp3", ".ogg")]
        self.assertEqual({path.stem for path in audio}, set(RECORDINGS))
        for path in audio:
            with self.subTest(file=path.name):
                self.assertGreater(path.stat().st_size, 1000)
                self.assertLessEqual(path.stat().st_size, fairy.AUDIO_MAX_BYTES)
                self.assertIn(path.suffix, (".mp3", ".ogg"))

    def test_credits_has_sources_and_authors(self) -> None:
        self.assertIn("archive.org/details/", self.credits)
        self.assertRegex(self.credits, r"(?i)Афанасьев|Afanasyev")
        self.assertRegex(self.credits, r"(?i)LibriVox")

    def test_manifest_allows_audio_directory(self) -> None:
        import json as json_module

        manifest = json_module.loads(
            (PLUGIN_DIR / "plugin.json").read_text(encoding="utf-8")
        )
        self.assertIn("audio", manifest["permissions"]["filesystem"])


# --------------------------------------------------------------------------- #
# HTTP: silent в SSE delta/done
# --------------------------------------------------------------------------- #


class MockAdapter:
    """Заглушка бэкенда; при ответе плагина не должна вызываться."""

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
    """Минимум ``PluginRegistry`` для ответа способности ``answer``."""

    def __init__(self, reply: object) -> None:
        self.reply = reply
        self.questions: list[str] = []

    def answers(self):
        def run(_ctx, question):
            self.questions.append(question)
            return self.reply

        return [SimpleNamespace(plugin_id="fairy-tales", run=run, ctx=None)]

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


class SilentServerCase(unittest.TestCase):
    def start(self, reply: object) -> None:
        self.adapter = MockAdapter()
        self.ollama = MockAdapter()
        self.runtime = FakePlugins(reply)
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

    def chat(self, question: str) -> list[dict]:
        payload = {
            "question": question,
            "use_system": False,
            "sides": [
                {"backend": "vtm", "model": "mock-vtm"},
                {"backend": "ollama", "model": "mock-qwen"},
            ],
        }
        conn = http.client.HTTPConnection(HOST, self.port, timeout=10)
        try:
            conn.request(
                "POST",
                "/api/chat",
                body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            response = conn.getresponse()
            status = response.status
            raw = response.read()
        finally:
            conn.close()
        self.assertEqual(status, 200, raw.decode("utf-8", "replace"))
        events, done = parse_sse(raw)
        self.assertTrue(done, "поток не завершился [DONE]")
        return events

    def page(self) -> str:
        conn = http.client.HTTPConnection(HOST, self.port, timeout=10)
        try:
            conn.request("GET", "/")
            response = conn.getresponse()
            return response.read().decode("utf-8")
        finally:
            conn.close()


class TestSilentStream(SilentServerCase):
    def test_silent_reaches_delta_and_done(self) -> None:
        self.start({"text": "Включаю запись «Теремок».", "silent": True})
        events = self.chat("Включи запись сказки")
        deltas = [
            event
            for event in events
            if event.get("type") == "delta" and event.get("side") == 0
        ]
        done = [
            event
            for event in events
            if event.get("type") == "done" and event.get("side") == 0
        ]
        self.assertTrue(deltas)
        self.assertEqual("".join(event["text"] for event in deltas),
                         "Включаю запись «Теремок».")
        self.assertTrue(all(event.get("silent") is True for event in deltas))
        self.assertEqual(len(done), 1)
        self.assertTrue(done[0].get("silent"))
        self.assertEqual(done[0].get("finish_reason"), "plugin")
        self.assertEqual(done[0].get("source"), "plugin")
        self.assertEqual(self.adapter.calls, 0, "модель не должна отвечать")

    def test_plain_plugin_answer_is_not_silent(self) -> None:
        self.start("Записи этой сказки нет, могу рассказать сама.")
        events = self.chat("Включи запись сказки про колобка")
        deltas = [
            event
            for event in events
            if event.get("type") == "delta" and event.get("side") == 0
        ]
        done = [
            event
            for event in events
            if event.get("type") == "done" and event.get("side") == 0
        ]
        self.assertTrue(deltas)
        self.assertTrue(all(event.get("silent") is False for event in deltas))
        self.assertFalse(done[0].get("silent"))

    def test_client_page_honours_silent(self) -> None:
        self.start({"text": "Включаю запись.", "silent": True})
        page = self.page()
        self.assertIn("event.silent", page)
        self.assertIn("state.silent", page)
        self.assertIn("!state.silent", page)


if __name__ == "__main__":
    unittest.main()
