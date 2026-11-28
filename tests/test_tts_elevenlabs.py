"""ElevenLabs-провайдер ``eva/tts.py`` на моках: без сети, ключа и звука.

Запуск::

    python3 -m unittest tests.test_tts_elevenlabs -v
    python3 -m unittest discover -s tests

Что проверяется (задача 0191):

* заголовки и тело ``POST /v1/text-to-speech/{voice_id}``: ``xi-api-key``,
  ``output_format``, ``model_id``, ``voice_settings``;
* content-type ``audio/mpeg`` для MP3 и ``audio/wav`` для локального фолбэка;
* кэш: повторная фраза не идёт в сеть, ``ELEVENLABS_CACHE=0`` его отключает,
  LRU держит 50 записей;
* ошибки 401/429/таймаут/сеть/пустой ответ → ``TTSError`` без ключа в тексте;
* автоматический откат на Silero/system и лог без ключа; если локального
  голоса нет — ошибка, а не тишина;
* порядок автовыбора ElevenLabs → Silero → system → null (Piper выключен из
  авто, задача 0197) и перекрытие ``EVA_TTS``.

HTTP подменяется фейковым opener'ом: сеть не используется вообще.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
import urllib.error
import urllib.parse
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eva import tts  # noqa: E402
from tools import chat_server  # noqa: E402

#: Правдоподобные аудиоданные: ID3-тег и WAV-заголовок.
FAKE_MP3 = b"ID3" + b"\x04\x00\x00" + b"\x00" * 512
FAKE_MP3_SYNC = b"\xff\xfb\x90\x64" + b"\x00" * 512
FAKE_WAV = b"RIFF" + b"\x24\x00\x00\x00" + b"WAVE" + b"fmt " + b"\x00" * 900

API_KEY = "sk-test-secret-key"
VOICE_ID = "EXAVITQu4vr4xnSDxMaL"


def _env(**overrides: str) -> dict[str, str]:
    base = {
        "ELEVENLABS_API_KEY": API_KEY,
        "ELEVENLABS_VOICE_ID": VOICE_ID,
    }
    base.update(overrides)
    return base


def _has_system_tts() -> bool:
    """Есть ли на машине системный синтез: say (macOS) или espeak (Linux).

    Тесты запасного пути рассчитаны на `say` и на реальный вывод espeak;
    в окружении без них честнее пропустить, чем требовать установки пакетов.
    """
    import shutil
    return any(shutil.which(name) for name in ("say", "espeak-ng", "espeak"))


HAS_SYSTEM_TTS = _has_system_tts()


class FakeResponse:
    """Минимальный ответ ``urlopen``: read/headers/status/close."""

    def __init__(
        self,
        body: bytes = FAKE_MP3,
        *,
        content_type: str = "audio/mpeg",
        status: int = 200,
    ) -> None:
        self._body = body
        self.status = status
        self.headers = {"Content-Type": content_type}
        self.closed = False

    def read(self) -> bytes:
        return self._body

    def close(self) -> None:
        self.closed = True


class FakeHTTP:
    """Fake opener: записывает Request'ы и отдаёт/бросает заданное по очереди."""

    def __init__(self, *responses) -> None:
        self.requests: list[urllib.request.Request] = []
        self.timeouts: list[float] = []
        self.responses: list[object] = list(responses)

    def __call__(self, request, timeout=None):
        self.requests.append(request)
        self.timeouts.append(timeout)
        if self.responses:
            item = self.responses.pop(0)
        else:
            item = FakeResponse(FAKE_MP3)
        if isinstance(item, BaseException):
            raise item
        return item


def _headers(request: urllib.request.Request) -> dict[str, str]:
    return {str(name).lower(): str(value) for name, value in request.headers.items()}


def make_executable(path: Path) -> Path:
    path.write_bytes(b"#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return path


def _output_path(command: list[str]) -> Path | None:
    for flag in ("--output_file", "-o", "-w"):
        if flag in command:
            index = command.index(flag)
            if index + 1 < len(command):
                return Path(command[index + 1])
    return None


def fake_popen(audio: bytes = FAKE_WAV):
    """Замена ``subprocess.Popen``: пишет WAV в output-файл и быстро выходит."""
    calls: list[list[str]] = []

    class FakePopen:
        def __init__(self, command, stdin=None, stdout=None, stderr=None,
                     text=None, **kwargs):
            calls.append([str(item) for item in command])
            target = _output_path(calls[-1])
            if target is not None:
                target.write_bytes(audio)
            self.returncode: int | None = None
            self._polls = 0

        def poll(self):
            self._polls += 1
            if self._polls >= 2:
                self.returncode = 0
            return self.returncode

        def communicate(self, *args, **kwargs):
            if self.returncode is None:
                self.returncode = 0
            return "", ""

        def kill(self):
            self.returncode = -9

    return FakePopen, calls


class ElevenLabsRequestTests(unittest.TestCase):
    """Тело, заголовки и URL запроса; content-type ответа."""

    def setUp(self) -> None:
        tts.clear_elevenlabs_cache()

    def tearDown(self) -> None:
        tts.clear_elevenlabs_cache()

    def test_request_headers_body_and_url(self):
        http = FakeHTTP(FakeResponse(FAKE_MP3))
        engine = tts.ElevenLabsTTS(env=_env(), opener=http)

        data, content_type = engine.synthesize("  Привет, мир.  ")

        self.assertEqual(data, FAKE_MP3)
        self.assertEqual(content_type, "audio/mpeg")
        self.assertEqual(len(http.requests), 1)
        self.assertEqual(http.timeouts, [tts.DEFAULT_ELEVENLABS_TIMEOUT])
        request = http.requests[0]
        self.assertEqual(request.get_method(), "POST")

        parsed = urllib.parse.urlsplit(request.full_url)
        self.assertEqual(
            parsed.path, f"/v1/text-to-speech/{VOICE_ID}"
        )
        self.assertEqual(
            urllib.parse.parse_qs(parsed.query)["output_format"],
            ["mp3_44100_128"],
        )

        headers = _headers(request)
        self.assertEqual(headers["xi-api-key"], API_KEY)
        self.assertEqual(headers["content-type"], "application/json")
        self.assertEqual(headers["accept"], "audio/mpeg")

        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["text"], "Привет, мир.")  # пробелы обрезаны
        self.assertEqual(payload["model_id"], "eleven_multilingual_v2")
        self.assertEqual(
            payload["voice_settings"],
            {"stability": 0.4, "similarity_boost": 0.75, "style": 0.0},
        )

    def test_env_overrides_voice_model_and_format(self):
        http = FakeHTTP(FakeResponse(FAKE_MP3, content_type="audio/mpeg"))
        engine = tts.ElevenLabsTTS(
            env=_env(
                ELEVENLABS_VOICE_ID="pFZP5JQG7iQjIQuC4Bku",
                ELEVENLABS_MODEL="eleven_turbo_v2_5",
                ELEVENLABS_FORMAT="mp3_22050_32",
            ),
            opener=http,
        )
        engine.synthesize("Привет")
        request = http.requests[0]
        self.assertIn(
            "/v1/text-to-speech/pFZP5JQG7iQjIQuC4Bku",
            request.full_url,
        )
        self.assertIn("output_format=mp3_22050_32", request.full_url)
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["model_id"], "eleven_turbo_v2_5")

    def test_say_voice_preset_is_not_sent_as_elevenlabs_id(self):
        # Страница чата присылает пресет "irina", а не ID ElevenLabs.
        http = FakeHTTP(FakeResponse())
        engine = tts.ElevenLabsTTS(env=_env(), opener=http)
        engine.synthesize("Привет", voice="irina")
        self.assertIn(VOICE_ID, http.requests[0].full_url)
        self.assertNotIn("irina", http.requests[0].full_url)

    def test_voice_override_id_is_used(self):
        other = "pFZP5JQG7iQjIQuC4Bku"
        http = FakeHTTP(FakeResponse())
        engine = tts.ElevenLabsTTS(env=_env(), opener=http)
        engine.synthesize("Привет", voice=other)
        self.assertIn(other, http.requests[0].full_url)

    def test_response_content_type_is_normalized(self):
        http = FakeHTTP(
            FakeResponse(FAKE_MP3, content_type="audio/mpeg; charset=binary")
        )
        engine = tts.ElevenLabsTTS(env=_env(), opener=http)
        data, content_type = engine.synthesize("Привет")
        self.assertEqual(content_type, "audio/mpeg")
        self.assertEqual(data, FAKE_MP3)

    def test_synthesize_to_path_writes_mp3(self):
        http = FakeHTTP(FakeResponse(FAKE_MP3))
        engine = tts.ElevenLabsTTS(env=_env(), opener=http)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "reply.mp3"
            result = engine.synthesize("Привет", target)
            self.assertEqual(result, target)
            self.assertEqual(target.read_bytes(), FAKE_MP3)

    def test_synthesize_timed_reports_engine_and_file(self):
        http = FakeHTTP(FakeResponse(FAKE_MP3))
        engine = tts.ElevenLabsTTS(env=_env(), opener=http)
        with tempfile.TemporaryDirectory() as tmp:
            result = engine.synthesize_timed("Привет", Path(tmp) / "reply.mp3")
            self.assertEqual(result.engine, "elevenlabs")
            self.assertTrue(result.path.is_file())
            self.assertGreaterEqual(result.synth_ms, 0.0)

    def test_module_synthesize_audio_forwards_mp3_type(self):
        http = FakeHTTP(FakeResponse(FAKE_MP3))
        with mock.patch.object(tts.urllib.request, "urlopen", http):
            data, content_type = tts.synthesize_audio("Привет", env=_env())
        self.assertEqual(data, FAKE_MP3)
        self.assertEqual(content_type, "audio/mpeg")


class ElevenLabsCacheTests(unittest.TestCase):
    """LRU-кэш: экономит квоту, повторная фраза не идёт в сеть."""

    def setUp(self) -> None:
        tts.clear_elevenlabs_cache()

    def tearDown(self) -> None:
        tts.clear_elevenlabs_cache()

    def test_second_call_hits_cache(self):
        http = FakeHTTP(FakeResponse(FAKE_MP3), FakeResponse(b"second"))
        engine = tts.ElevenLabsTTS(env=_env(), opener=http)

        first = engine.synthesize("Привет")
        second = engine.synthesize("Привет")

        self.assertEqual(first, second)
        self.assertEqual(len(http.requests), 1)
        info = tts.elevenlabs_cache_info()
        self.assertEqual(info["hits"], 1)
        self.assertEqual(info["size"], 1)

    def test_different_text_calls_network(self):
        http = FakeHTTP(FakeResponse(FAKE_MP3), FakeResponse(b"other"))
        engine = tts.ElevenLabsTTS(env=_env(), opener=http)
        engine.synthesize("Первая")
        engine.synthesize("Вторая")
        self.assertEqual(len(http.requests), 2)

    def test_cache_can_be_disabled(self):
        http = FakeHTTP(FakeResponse(FAKE_MP3), FakeResponse(FAKE_MP3))
        engine = tts.ElevenLabsTTS(
            env=_env(ELEVENLABS_CACHE="0"), opener=http
        )
        engine.synthesize("Привет")
        engine.synthesize("Привет")
        self.assertEqual(len(http.requests), 2)
        self.assertEqual(tts.elevenlabs_cache_info()["size"], 0)

    def test_lru_keeps_last_50_and_evicts_oldest(self):
        http = FakeHTTP()
        engine = tts.ElevenLabsTTS(env=_env(), opener=http)
        for index in range(tts.ELEVENLABS_CACHE_SIZE):
            engine.synthesize(f"фраза {index}")

        self.assertEqual(
            tts.elevenlabs_cache_info()["size"], tts.ELEVENLABS_CACHE_SIZE
        )
        # Обращение к самой старой делает её свежей; добавляем 51-ю.
        engine.synthesize("фраза 0")
        engine.synthesize("фраза 50")
        calls_before = len(http.requests)

        engine.synthesize("фраза 0")  # всё ещё в кэше
        self.assertEqual(len(http.requests), calls_before)
        engine.synthesize("фраза 1")  # вытеснена — снова в сеть
        self.assertEqual(len(http.requests), calls_before + 1)

    def test_shared_cache_survives_new_adapter_instances(self):
        # select_engine создаёт новый адаптер на запрос — кэш должен быть общим.
        http = FakeHTTP(FakeResponse(FAKE_MP3))
        tts.ElevenLabsTTS(env=_env(), opener=http).synthesize("Привет")
        tts.ElevenLabsTTS(env=_env(), opener=http).synthesize("Привет")
        self.assertEqual(len(http.requests), 1)


class ElevenLabsFailureTests(unittest.TestCase):
    """401/429/таймаут/сеть/пусто: понятная ошибка без ключа."""

    def setUp(self) -> None:
        tts.clear_elevenlabs_cache()

    def tearDown(self) -> None:
        tts.clear_elevenlabs_cache()

    def _assert_failure(self, error: BaseException, *fragments: str) -> str:
        http = FakeHTTP(error)
        engine = tts.ElevenLabsTTS(env=_env(), opener=http)
        with self.assertRaises(tts.TTSError) as caught:
            engine.synthesize("Привет")
        message = str(caught.exception)
        for fragment in fragments:
            self.assertIn(fragment, message)
        self.assertNotIn(API_KEY, message)
        return message

    def test_missing_key_is_unavailable_and_mentions_variable(self):
        engine = tts.ElevenLabsTTS(env={})
        self.assertFalse(engine.available())
        with self.assertRaises(tts.TTSUnavailable) as caught:
            engine.synthesize("Привет")
        self.assertIn("ELEVENLABS_API_KEY", str(caught.exception))

    def test_401_is_clear_error_and_hides_key(self):
        self._assert_failure(
            urllib.error.HTTPError(
                "https://api.elevenlabs.io", 401, "Unauthorized", {}, None
            ),
            "401",
        )

    def test_429_is_quota_error(self):
        self._assert_failure(
            urllib.error.HTTPError(
                "https://api.elevenlabs.io", 429, "Too Many Requests", {}, None
            ),
            "429",
            "квота",
        )

    def test_timeout_is_error(self):
        self._assert_failure(TimeoutError("timed out"), "таймаут")

    def test_network_error_is_error(self):
        self._assert_failure(
            urllib.error.URLError("Name or service not known"), "сеть"
        )

    def test_empty_response_is_error(self):
        http = FakeHTTP(FakeResponse(b""))
        engine = tts.ElevenLabsTTS(env=_env(), opener=http)
        with self.assertRaises(tts.TTSError) as caught:
            engine.synthesize("Привет")
        self.assertIn("пустой", str(caught.exception))

    def test_describe_does_not_leak_key(self):
        engine = tts.ElevenLabsTTS(env=_env())
        description = engine.describe()
        self.assertIn("elevenlabs", description)
        self.assertNotIn(API_KEY, description)
        self.assertNotIn(API_KEY, repr(engine.resolve()))


class EngineSelectionElevenLabsTests(unittest.TestCase):
    """Приоритет ElevenLabs и явное перекрытие ``EVA_TTS``."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.piper_bin = make_executable(self.tmp / "piper")
        self.model = self.tmp / "ru_RU-test-medium.onnx"
        self.model.write_bytes(b"onnx")

    def test_auto_prefers_elevenlabs_with_key(self):
        engine = tts.select_engine(env=_env())
        self.assertIsInstance(engine, tts.ElevenLabsTTS)
        self.assertEqual(engine.name, "elevenlabs")

    @unittest.skipUnless(HAS_SYSTEM_TTS, "нет системного TTS (say/espeak)")
    def test_auto_without_key_uses_system_not_piper(self):
        # Без ключа Silero не установлен (env изолирован), Piper не в авто.
        env = {"PIPER_BIN": str(self.piper_bin), "PIPER_MODEL": str(self.model)}
        engine = tts.select_engine(env=env)
        self.assertIsInstance(engine, tts.SystemTTS)
        self.assertEqual(engine.name, "system")

    def test_eva_tts_piper_overrides_elevenlabs(self):
        env = _env(
            EVA_TTS="piper",
            PIPER_BIN=str(self.piper_bin),
            PIPER_MODEL=str(self.model),
        )
        self.assertIsInstance(tts.select_engine(env=env), tts.PiperTTS)

    def test_eva_tts_elevenlabs_without_key_is_explicitly_unavailable(self):
        engine = tts.select_engine(env={"EVA_TTS": "elevenlabs"})
        self.assertIsInstance(engine, tts.ElevenLabsTTS)
        self.assertFalse(engine.available())

    def test_eva_tts_null_overrides_key(self):
        self.assertIsInstance(
            tts.select_engine(env=_env(EVA_TTS="null")), tts.NullTTS
        )


class ElevenLabsFallbackTests(unittest.TestCase):
    """Сбой облака → Silero/system; ошибка только если голоса нет."""

    def setUp(self) -> None:
        tts.clear_elevenlabs_cache()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.piper_bin = make_executable(self.tmp / "piper")
        self.model = self.tmp / "ru_RU-test-medium.onnx"
        self.model.write_bytes(b"onnx")
        self.env = _env(
            PIPER_BIN=str(self.piper_bin), PIPER_MODEL=str(self.model)
        )

    def tearDown(self) -> None:
        tts.clear_elevenlabs_cache()

    @unittest.skipUnless(HAS_SYSTEM_TTS, "нет системного TTS (say/espeak)")
    def test_401_falls_back_to_system_and_logs_without_key(self):
        http = FakeHTTP(
            urllib.error.HTTPError(
                "https://api.elevenlabs.io", 401, "Unauthorized", {}, None
            )
        )
        fake, calls = fake_popen()
        with mock.patch.object(tts.urllib.request, "urlopen", http), \
                mock.patch.object(tts.subprocess, "Popen", fake), \
                self.assertLogs("eva.tts", level="WARNING") as logs:
            data = tts.synthesize("Привет", env=self.env)

        self.assertIsInstance(data, bytes)
        self.assertTrue(data.startswith(b"RIFF"))
        self.assertEqual(len(http.requests), 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(Path(calls[0][0]).name, "say")
        joined = "\n".join(logs.output)
        self.assertIn("откат на system", joined)
        self.assertNotIn(API_KEY, joined)

    @unittest.skipUnless(HAS_SYSTEM_TTS, "нет системного TTS (say/espeak)")
    def test_429_falls_back_to_system(self):
        http = FakeHTTP(
            urllib.error.HTTPError(
                "https://api.elevenlabs.io", 429, "Too Many Requests", {}, None
            )
        )
        fake, _ = fake_popen()
        with mock.patch.object(tts.urllib.request, "urlopen", http), \
                mock.patch.object(tts.subprocess, "Popen", fake):
            data = tts.synthesize("Привет", env=self.env)
        self.assertTrue(data.startswith(b"RIFF"))

    @unittest.skipUnless(HAS_SYSTEM_TTS, "нет системного TTS (say/espeak)")
    def test_timeout_falls_back_to_system(self):
        http = FakeHTTP(TimeoutError("timed out"))
        fake, _ = fake_popen()
        with mock.patch.object(tts.urllib.request, "urlopen", http), \
                mock.patch.object(tts.subprocess, "Popen", fake):
            data = tts.synthesize("Привет", env=self.env)
        self.assertTrue(data.startswith(b"RIFF"))

    @unittest.skipUnless(HAS_SYSTEM_TTS, "нет системного TTS (say/espeak)")
    def test_explicit_elevenlabs_without_key_falls_back_to_system(self):
        # env без ключа: явный ElevenLabs недоступен, должен уйти на system.
        env = {
            "PIPER_BIN": str(self.piper_bin),
            "PIPER_MODEL": str(self.model),
        }
        fake, _ = fake_popen()
        with mock.patch.object(tts.subprocess, "Popen", fake):
            data = tts.synthesize("Привет", prefer="elevenlabs", env=env)
        self.assertTrue(data.startswith(b"RIFF"))

    @unittest.skipUnless(HAS_SYSTEM_TTS, "нет системного TTS (say/espeak)")
    def test_synthesize_audio_reports_wav_after_fallback(self):
        http = FakeHTTP(TimeoutError("timed out"))
        fake, _ = fake_popen()
        with mock.patch.object(tts.urllib.request, "urlopen", http), \
                mock.patch.object(tts.subprocess, "Popen", fake):
            data, content_type = tts.synthesize_audio("Привет", env=self.env)
        self.assertTrue(data.startswith(b"RIFF"))
        self.assertEqual(content_type, "audio/wav")

    def test_no_fallback_raises_instead_of_silence(self):
        http = FakeHTTP(
            urllib.error.HTTPError(
                "https://api.elevenlabs.io", 401, "Unauthorized", {}, None
            )
        )
        with mock.patch.object(tts.urllib.request, "urlopen", http), \
                mock.patch.object(tts, "find_piper_binary", return_value=None), \
                mock.patch.object(tts, "find_say_binary", return_value=None), \
                mock.patch.object(tts, "find_espeak_binary", return_value=None), \
                mock.patch.object(tts.shutil, "which", return_value=None):
            with self.assertRaises(tts.TTSError) as caught:
                tts.synthesize("Привет", env=_env())
        message = str(caught.exception)
        self.assertIn("401", message)
        self.assertIn("запасной", message)
        self.assertNotIn(API_KEY, message)


class ContentTypeTests(unittest.TestCase):
    """Content-Type по формату и сигнатуре, без сети."""

    def test_default_mp3_is_audio_mpeg(self):
        self.assertEqual(
            tts._content_type_for_format("mp3_44100_128"), "audio/mpeg"
        )

    def test_unknown_format_defaults_to_mpeg(self):
        self.assertEqual(
            tts._content_type_for_format("something_new"), "audio/mpeg"
        )

    def test_server_detects_wav_mp3_and_rejects_other_bytes(self):
        self.assertEqual(chat_server.looks_like_audio(FAKE_WAV), "audio/wav")
        self.assertEqual(chat_server.looks_like_audio(FAKE_MP3), "audio/mpeg")
        self.assertEqual(
            chat_server.looks_like_audio(FAKE_MP3_SYNC), "audio/mpeg"
        )
        self.assertIsNone(chat_server.looks_like_audio(b"OggS not audio"))
        self.assertIsNone(chat_server.looks_like_audio(b""))

    def test_default_tts_speech_forwards_content_type(self):
        with mock.patch(
            "eva.tts.synthesize_audio",
            return_value=(FAKE_MP3, "audio/mpeg"),
        ):
            data, content_type = chat_server.default_tts_speech("Привет")
        self.assertEqual(data, FAKE_MP3)
        self.assertEqual(content_type, "audio/mpeg")

    def test_default_tts_speech_does_not_pitch_shift_mp3_preset(self):
        # Пресет "young" — параметр WAV-Piper; MP3 нельзя ресемплировать этими
        # функциями, и портить облачный голос не нужно.
        with mock.patch(
            "eva.tts.synthesize_audio",
            return_value=(FAKE_MP3, "audio/mpeg"),
        ):
            data, content_type = chat_server.default_tts_speech(
                "Привет", voice="young"
            )
        self.assertEqual(data, FAKE_MP3)
        self.assertEqual(content_type, "audio/mpeg")


if __name__ == "__main__":
    unittest.main()
