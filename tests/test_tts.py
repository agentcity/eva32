"""Проверки TTS-адаптера ``eva.tts`` без звука, сети и реальных моделей.

Запуск::

    python3 -m unittest tests.test_tts -v
    python3 -m unittest discover -s tests

Что проверяется:

* выбор движка: Piper → system → null, явный конфиг перекрывает автовыбор;
* отсутствие бинаря Piper/espeak не падает, а уводит на фолбэк;
* команды ``piper``/``say``/``espeak`` и передача текста через stdin;
* mock-процесс: успешный синтез в bytes/файл, ненулевой код, таймаут,
  отсутствие аудиофайла;
* интеграция с ``scripts/voice_loop.py``: адаптер в потоковой озвучке по
  предложениям, ``ThreadPoolExecutor(2)`` и понятная ошибка без TTS.

Тесты не запускают ``piper``, ``say``, ``espeak``, ``afplay`` и не ходят
в сеть: ``Popen`` и поиск бинарей подменяются.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eva import tts  # noqa: E402
from scripts import voice_loop  # noqa: E402

#: Правдоподобный WAV: заголовок + данные, больше любого порога.
FAKE_WAV = b"RIFF" + b"\x00" * 9000


def make_executable(path: Path) -> Path:
    """Создать пустой исполняемый файл — заглушку бинаря."""
    path.write_bytes(b"#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return path


def output_path_of(command: list[str]) -> Path | None:
    """Достать путь WAV из команды piper/say/espeak."""
    for flag in ("--output_file", "-o", "-w"):
        if flag in command:
            index = command.index(flag)
            if index + 1 < len(command):
                return Path(command[index + 1])
    return None


def fake_popen(**spec):
    """Собрать класс-замену ``subprocess.Popen`` и журналы вызовов.

    ``spec``: ``audio`` (байты WAV), ``write`` (писать ли файл), ``polls``
    (через сколько опросов процесс завершится), ``returncode``, ``stdout``,
    ``stderr``. Возвращает ``(класс, calls, stdin_texts)``: ``calls`` — argv
    каждого запуска, ``stdin_texts`` — текст, прочитанный из stdin-файла.
    """
    options = {
        "audio": FAKE_WAV,
        "write": True,
        "polls": 2,
        "returncode": 0,
        "stdout": "",
        "stderr": "",
    }
    options.update(spec)
    calls: list[list[str]] = []
    stdin_texts: list[str | None] = []

    class FakePopen:
        def __init__(
            self,
            command,
            stdin=None,
            stdout=None,
            stderr=None,
            text=None,
            **kwargs,
        ):
            calls.append([str(item) for item in command])
            if stdin is not None and hasattr(stdin, "read"):
                try:
                    stdin.seek(0)
                except (OSError, ValueError):
                    pass
                stdin_texts.append(stdin.read())
            else:
                stdin_texts.append(None)
            self.returncode: int | None = None
            self._polls = 0
            self._stdout = io.StringIO(options["stdout"])
            self._stderr = io.StringIO(options["stderr"])
            target = output_path_of(calls[-1])
            if options["write"] and target is not None:
                target.write_bytes(options["audio"])

        def poll(self):
            self._polls += 1
            if self._polls >= options["polls"]:
                self.returncode = options["returncode"]
            return self.returncode

        def communicate(self, *args, **kwargs):
            if self.returncode is None:
                self.returncode = options["returncode"]
            return self._stdout.read(), self._stderr.read()

        def kill(self):
            self.returncode = -9

        def wait(self, timeout=None):
            if self.returncode is None:
                self.returncode = options["returncode"]
            return self.returncode

    return FakePopen, calls, stdin_texts


class RecordingEngine(tts.TTSAdapter):
    """Адаптер-шпион: записывает вызовы и пишет маленький WAV."""

    name = "recording"

    def __init__(self, *, write: bool = True) -> None:
        self.calls: list[dict[str, object]] = []
        self._write = write
        self._lock = threading.Lock()

    def available(self) -> bool:
        return True

    def describe(self) -> str:
        return "recording (тестовый движок)"

    def build_command(self, text, out_path, *, rate=None, voice=None):
        return [], None

    def synthesize_timed(
        self,
        text,
        out_path,
        *,
        rate=None,
        voice=None,
        timeout=tts.DEFAULT_TIMEOUT,
    ) -> tts.SynthesisResult:
        with self._lock:
            self.calls.append(
                {"text": text, "path": Path(out_path), "rate": rate, "voice": voice}
            )
        target = Path(out_path)
        if self._write:
            target.write_bytes(FAKE_WAV)
        return tts.SynthesisResult(
            path=target,
            synth_ms=1.0,
            first_fragment_ms=0.5,
            engine=self.name,
        )


class EngineSelectionTests(unittest.TestCase):
    """Автовыбор: Silero → system → null; Piper выключен из авто (0197)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.piper_bin = make_executable(self.tmp / "piper")
        self.model = self.tmp / "ru_RU-test-medium.onnx"
        self.model.write_bytes(b"onnx")

    def test_auto_ignores_piper_even_when_binary_and_model_present(self):
        # Задача 0197: Piper-адаптер остался в коде, но автовыбор его не берёт.
        env = {"PIPER_BIN": str(self.piper_bin), "PIPER_MODEL": str(self.model)}
        with mock.patch.object(tts.shutil, "which", return_value=None), \
                mock.patch.object(
                    tts, "find_say_binary", return_value=Path("/fake/say")
                ), mock.patch.object(
                    tts, "find_espeak_binary", return_value=None
                ):
            engine = tts.select_engine(env=env)
        self.assertIsInstance(engine, tts.SystemTTS)
        self.assertEqual(engine.name, "system")

    def test_piper_is_still_available_as_explicit_legacy_choice(self):
        env = {"PIPER_BIN": str(self.piper_bin), "PIPER_MODEL": str(self.model)}
        engine = tts.select_engine("piper", env=env)
        self.assertIsInstance(engine, tts.PiperTTS)
        self.assertTrue(engine.available())
        self.assertIn(self.model.name, engine.describe())

    def test_auto_falls_back_to_system_without_piper(self):
        with mock.patch.object(
            tts, "find_piper_binary", return_value=None
        ), mock.patch.object(
            tts, "find_say_binary", return_value=Path("/fake/say")
        ), mock.patch.object(
            tts, "find_espeak_binary", return_value=None
        ):
            engine = tts.select_engine(env={})
        self.assertIsInstance(engine, tts.SystemTTS)
        self.assertTrue(engine.available())

    def test_auto_falls_back_to_null_when_nothing_available(self):
        with mock.patch.object(
            tts, "find_piper_binary", return_value=None
        ), mock.patch.object(
            tts, "find_say_binary", return_value=None
        ), mock.patch.object(
            tts, "find_espeak_binary", return_value=None
        ), mock.patch.object(tts.shutil, "which", return_value=None):
            engine = tts.select_engine(env={})
        self.assertIsInstance(engine, tts.NullTTS)
        self.assertFalse(engine.available())
        self.assertEqual(engine.synthesize("привет"), b"")

    def test_explicit_engine_overrides_auto(self):
        with mock.patch.object(
            tts, "find_piper_binary", return_value=None
        ), mock.patch.object(tts, "find_say_binary", return_value=None), \
                mock.patch.object(tts, "find_espeak_binary", return_value=None):
            system = tts.select_engine("system", env={})
            piper = tts.select_engine("piper", env={})
            null = tts.select_engine("null", env={})
        self.assertIsInstance(system, tts.SystemTTS)
        self.assertIsInstance(piper, tts.PiperTTS)
        self.assertIsInstance(null, tts.NullTTS)

    def test_explicit_piper_without_binary_reports_unavailable(self):
        engine = tts.select_engine(
            "piper", env={"PIPER_BIN": str(self.tmp / "missing-piper")}
        )
        self.assertIsInstance(engine, tts.PiperTTS)
        self.assertFalse(engine.available())
        with self.assertRaises(tts.TTSUnavailable) as caught:
            engine.synthesize("привет")
        self.assertIn("piper", str(caught.exception).lower())

    def test_broken_piper_env_is_ignored_in_auto(self):
        with mock.patch.object(tts.shutil, "which", return_value=None):
            engine = tts.select_engine(
                "auto", env={"PIPER_BIN": str(self.tmp / "nope-piper")}
            )
        # Битая переменная окружения — не повод падать: работает фолбэк.
        self.assertIsInstance(engine, tts.NullTTS)

    def test_unknown_engine_rejected(self):
        with self.assertRaises(tts.TTSError):
            tts.select_engine("super-tts", env={})

    def test_module_synthesize_without_any_engine_is_silent(self):
        with mock.patch.object(tts.shutil, "which", return_value=None):
            data = tts.synthesize("привет", env={})
        self.assertEqual(data, b"")


class PiperCommandTests(unittest.TestCase):
    """Piper: команда, модель, stdin и явные ошибки конфигурации."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.binary = make_executable(self.tmp / "piper")
        self.model = self.tmp / "ru_RU-test-medium.onnx"
        self.model.write_bytes(b"onnx")

    def test_command_uses_model_and_stdin(self):
        engine = tts.PiperTTS(binary=self.binary, model=self.model)
        command, stdin_text = engine.build_command("Привет", self.tmp / "out.wav")
        self.assertEqual(command[0], str(self.binary))
        self.assertEqual(command[command.index("--model") + 1], str(self.model))
        self.assertEqual(
            command[command.index("--output_file") + 1],
            str(self.tmp / "out.wav"),
        )
        # Текст не в argv (иначе длинная фраза упирается в лимит), а в stdin.
        self.assertEqual(stdin_text, "Привет\n")
        self.assertNotIn("Привет", command)

    def test_rate_maps_to_length_scale(self):
        engine = tts.PiperTTS(binary=self.binary, model=self.model)
        slow, _ = engine.build_command("Привет", self.tmp / "a.wav", rate=90)
        fast, _ = engine.build_command("Привет", self.tmp / "b.wav", rate=360)
        slow_scale = float(slow[slow.index("--length_scale") + 1])
        fast_scale = float(fast[fast.index("--length_scale") + 1])
        self.assertGreater(slow_scale, fast_scale)
        self.assertGreaterEqual(fast_scale, 0.5)
        self.assertLessEqual(slow_scale, 2.0)

    def test_missing_binary_is_unavailable_not_crash(self):
        engine = tts.PiperTTS(
            binary=self.tmp / "no-piper", model=self.model
        )
        self.assertFalse(engine.available())
        with self.assertRaises(tts.TTSUnavailable):
            engine.synthesize("привет")

    def test_missing_model_file_is_clear_error(self):
        engine = tts.PiperTTS(binary=self.binary, model=self.tmp / "nope.onnx")
        self.assertFalse(engine.available())
        with self.assertRaises(tts.TTSUnavailable) as caught:
            engine.synthesize("привет")
        self.assertIn("PIPER_MODEL", str(caught.exception))

    def test_model_discovery_prefers_russian(self):
        english = self.tmp / "en_US-lessac-medium.onnx"
        russian = self.tmp / "ru_RU-irina-medium.onnx"
        for model in (english, russian):
            model.write_bytes(b"onnx")
        found = tts.find_piper_model(env={"PIPER_MODEL_DIR": str(self.tmp)})
        self.assertEqual(found, russian)

    def test_model_env_with_missing_file_is_error(self):
        with self.assertRaises(tts.TTSUnavailable):
            tts.find_piper_model(env={"PIPER_MODEL": str(self.tmp / "nope.onnx")})


class SystemCommandTests(unittest.TestCase):
    """Системный TTS: say на macOS, espeak на Linux."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_say_command_matches_voice_loop_wrapper(self):
        say = make_executable(self.tmp / "say")
        engine = tts.SystemTTS(say, kind="say", voice="Milena")
        command, stdin_text = engine.build_command(
            "Привет", "/tmp/reply.wav", rate=180
        )
        self.assertIsNone(stdin_text)
        self.assertEqual(
            command, voice_loop.build_say_command(say, "Milena", "/tmp/reply.wav", "Привет", rate=180)
        )
        self.assertIn("--data-format=LEI16@16000", command)

    def test_say_ignores_silero_voice_and_uses_milena(self):
        # /api/tts всегда присылает имя голоса Silero; при откате на system
        # `say -v baya` не должен ронять озвучку (задача 0197).
        say = make_executable(self.tmp / "say")
        engine = tts.SystemTTS(say, kind="say")
        for voice in ("baya", "kseniya", "xenia", "aidar"):
            command, _ = engine.build_command(
                "Привет", self.tmp / "out.wav", voice=voice
            )
            self.assertEqual(
                command[command.index("-v") + 1], tts.DEFAULT_SAY_VOICE, voice
            )

    def test_espeak_command_reads_stdin(self):
        espeak = make_executable(self.tmp / "espeak-ng")
        engine = tts.SystemTTS(espeak, kind="espeak", lang="ru")
        command, stdin_text = engine.build_command(
            "Привет", self.tmp / "out.wav", rate=160
        )
        self.assertEqual(command[0], str(espeak))
        self.assertEqual(command[command.index("-w") + 1], str(self.tmp / "out.wav"))
        self.assertEqual(command[command.index("-v") + 1], "ru")
        self.assertEqual(command[command.index("-s") + 1], "160")
        self.assertIn("--stdin", command)
        self.assertEqual(stdin_text, "Привет\n")

    def test_espeak_ignores_say_voice_name(self):
        espeak = make_executable(self.tmp / "espeak")
        engine = tts.SystemTTS(espeak, kind="espeak", lang="ru")
        command, _ = engine.build_command(
            "Привет", self.tmp / "out.wav", voice="Milena"
        )
        # Имя голоса say не должно уходить в espeak как язык.
        self.assertEqual(command[command.index("-v") + 1], "ru")

    def test_system_unavailable_raises(self):
        with mock.patch.object(
            tts, "find_say_binary", return_value=None
        ), mock.patch.object(tts, "find_espeak_binary", return_value=None):
            engine = tts.SystemTTS(env={})
            self.assertFalse(engine.available())
            with self.assertRaises(tts.TTSUnavailable) as caught:
                engine.synthesize("привет")
        self.assertIn("espeak", str(caught.exception))


class MockProcessTests(unittest.TestCase):
    """Синтез через mock-процесс: без реального звука и моделей."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.piper_bin = make_executable(self.tmp / "piper")
        self.model = self.tmp / "ru_RU-test-medium.onnx"
        self.model.write_bytes(b"onnx")

    def _engine(self) -> tts.PiperTTS:
        return tts.PiperTTS(binary=self.piper_bin, model=self.model)

    def test_synthesize_returns_bytes(self):
        fake, calls, stdin_texts = fake_popen()
        with mock.patch.object(tts.subprocess, "Popen", fake):
            data = self._engine().synthesize("Сколько стоит тариф Старт")
        self.assertIsInstance(data, bytes)
        self.assertTrue(data.startswith(b"RIFF"))
        self.assertEqual(len(calls), 1)
        self.assertEqual(stdin_texts, ["Сколько стоит тариф Старт\n"])

    def test_synthesize_writes_path_and_timings(self):
        fake, calls, _ = fake_popen()
        target = self.tmp / "reply.wav"
        with mock.patch.object(tts.subprocess, "Popen", fake):
            result = self._engine().synthesize_timed("Привет", target)
        self.assertEqual(result.path, target)
        self.assertTrue(target.is_file())
        self.assertEqual(result.engine, "piper")
        self.assertGreaterEqual(result.synth_ms, 0.0)
        self.assertIsNotNone(result.first_fragment_ms)
        self.assertEqual(len(calls), 1)

    def test_nonzero_exit_is_error(self):
        fake, _, _ = fake_popen(returncode=1, write=False, stderr="model load failed")
        with mock.patch.object(tts.subprocess, "Popen", fake):
            with self.assertRaises(tts.TTSError) as caught:
                self._engine().synthesize_timed("Привет", self.tmp / "reply.wav")
        self.assertIn("кодом 1", str(caught.exception))
        self.assertIn("model load failed", str(caught.exception))

    def test_timeout_is_error(self):
        fake, _, _ = fake_popen(polls=10**9)
        with mock.patch.object(tts.subprocess, "Popen", fake):
            with self.assertRaises(tts.TTSError) as caught:
                self._engine().synthesize_timed(
                    "Привет", self.tmp / "reply.wav", timeout=0.01
                )
        self.assertIn("не уложился", str(caught.exception))

    def test_missing_audio_is_error(self):
        fake, _, _ = fake_popen(write=False)
        with mock.patch.object(tts.subprocess, "Popen", fake):
            with self.assertRaises(tts.TTSError) as caught:
                self._engine().synthesize_timed("Привет", self.tmp / "reply.wav")
        self.assertIn("не создал аудио", str(caught.exception))

    def test_say_measures_first_fragment_after_header(self):
        say = make_executable(self.tmp / "say")
        fake, calls, stdin_texts = fake_popen()
        with mock.patch.object(tts.subprocess, "Popen", fake):
            result = tts.SystemTTS(say, kind="say", voice="Milena").synthesize_timed(
                "Привет", self.tmp / "reply.wav"
            )
        self.assertIsNotNone(result.first_fragment_ms)
        self.assertIsNone(stdin_texts[0])  # say получает текст в argv
        self.assertEqual(calls[0][-1], "Привет")
        self.assertEqual(result.engine, "system")

    def test_espeak_text_goes_to_stdin(self):
        espeak = make_executable(self.tmp / "espeak-ng")
        fake, _, stdin_texts = fake_popen()
        with mock.patch.object(tts.subprocess, "Popen", fake):
            tts.SystemTTS(espeak, kind="espeak", lang="ru").synthesize_timed(
                "Привет", self.tmp / "reply.wav"
            )
        self.assertEqual(stdin_texts, ["Привет\n"])

    def test_empty_text_rejected(self):
        with self.assertRaises(tts.TTSError):
            self._engine().synthesize("   ")

    def test_missing_output_directory_rejected(self):
        with self.assertRaises(tts.TTSError):
            self._engine().synthesize_timed(
                "Привет", self.tmp / "no-such-dir" / "reply.wav"
            )


class VoiceLoopAdapterTests(unittest.TestCase):
    """voice_loop использует адаптер и сохраняет стриминг по предложениям."""

    SPOKEN = "Первое предложение. Второе предложение. Третье предложение."

    def _route(self):
        return voice_loop.RouteResult(
            question="Сколько стоит тариф Старт",
            answer_text="Тариф «Старт» — 990 ₽/мес [1]. Лимиты. Заметки.",
            spoken_text=self.SPOKEN,
            route_kind="skill",
            route_reason="skill: pricing",
            skill="pricing",
            latency_ms=1.5,
            answered=True,
        )

    def test_voice_loop_synthesize_uses_engine(self):
        engine = RecordingEngine()
        with tempfile.TemporaryDirectory() as tmp:
            result = voice_loop.synthesize(
                "Привет",
                Path(tmp) / "reply.wav",
                voice="Milena",
                rate=180,
                timeout=5.0,
                engine=engine,
            )
            self.assertTrue(result.path.is_file())
        self.assertEqual(engine.calls[0]["text"], "Привет")
        self.assertEqual(engine.calls[0]["rate"], 180)
        self.assertEqual(engine.calls[0]["voice"], "Milena")
        self.assertEqual(result.engine, "recording")

    def test_voice_loop_null_engine_is_unavailable(self):
        # NullTTS означает «озвучивать нечем»: это ошибка TTS, а не тишина.
        self.assertIs(voice_loop.TtsUnavailable, tts.TTSUnavailable)
        self.assertTrue(issubclass(voice_loop.TtsError, tts.TTSError))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(tts.TTSUnavailable):
                voice_loop.synthesize(
                    "Привет", Path(tmp) / "reply.wav", engine=tts.NullTTS()
                )

    def test_run_without_engines_is_graceful(self):
        args = voice_loop.parse_args(
            ["--text", "Сколько стоит тариф Старт", "--json", "--no-play"]
        )
        err = io.StringIO()
        out = io.StringIO()
        with mock.patch.object(
            voice_loop, "ask_hybrid", return_value=self._route()
        ), mock.patch.object(
            voice_loop, "select_engine", return_value=tts.NullTTS()
        ), redirect_stderr(err), redirect_stdout(out):
            code = voice_loop.run(args, cfg=object())
        self.assertEqual(code, 3)
        message = err.getvalue()
        self.assertIn("TTS недоступен", message)
        self.assertIn("Piper", message)
        self.assertNotIn("Traceback", message)
        payload = json.loads(out.getvalue())
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["stage"], "tts")

    def test_run_stream_keeps_sentences_and_order(self):
        engine = RecordingEngine()
        args = voice_loop.parse_args(
            ["--text", "Сколько стоит тариф Старт", "--json", "--no-play"]
        )
        out = io.StringIO()
        with mock.patch.object(
            voice_loop, "ask_hybrid", return_value=self._route()
        ), mock.patch.object(
            voice_loop, "select_engine", return_value=engine
        ), redirect_stdout(out):
            code = voice_loop.run(args, cfg=object())

        self.assertEqual(code, 0)
        payload = json.loads(out.getvalue())
        sentences = voice_loop.split_sentences(self.SPOKEN)
        self.assertEqual(len(sentences), 3)
        # Первое предложение синтезируется синхронно, остальные — параллельно,
        # поэтому порядок записей после первого может быть любым.
        self.assertEqual(engine.calls[0]["text"], sentences[0])
        self.assertEqual(
            sorted(call["text"] for call in engine.calls), sorted(sentences)
        )
        self.assertTrue(payload["timings"]["stream"])
        self.assertEqual(payload["timings"]["chunks"], 3)
        self.assertEqual(len(payload["audio"]["parts"]), 3)
        self.assertEqual(payload["tts"]["engine"], "recording")
        self.assertEqual(voice_loop.STREAM_WORKERS, 2)

    def test_run_no_stream_synthesizes_whole_answer_once(self):
        engine = RecordingEngine()
        args = voice_loop.parse_args(
            [
                "--text",
                "Сколько стоит тариф Старт",
                "--json",
                "--no-play",
                "--no-stream",
            ]
        )
        out = io.StringIO()
        with mock.patch.object(
            voice_loop, "ask_hybrid", return_value=self._route()
        ), mock.patch.object(
            voice_loop, "select_engine", return_value=engine
        ), redirect_stdout(out):
            code = voice_loop.run(args, cfg=object())
        self.assertEqual(code, 0)
        self.assertEqual([call["text"] for call in engine.calls], [self.SPOKEN])

    def test_help_mentions_piper_and_tts_flag(self):
        out = io.StringIO()
        with redirect_stdout(out):
            with self.assertRaises(SystemExit) as caught:
                voice_loop.parse_args(["--help"])
        self.assertEqual(caught.exception.code, 0)
        text = out.getvalue()
        self.assertIn("--tts", text)
        self.assertIn("piper", text)
        self.assertIn("PIPER", text)


if __name__ == "__main__":
    unittest.main()
