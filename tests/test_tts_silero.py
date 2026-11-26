"""Проверки Silero-адаптера ``eva.tts`` без torch, модели, сети и звука.

Запуск::

    python3 -m unittest tests.test_tts_silero -v
    python3 -m unittest discover -s tests

Что проверяется (задача 0193):

* выбор движка: ``EVA_TTS`` → ElevenLabs (ключ) → Silero (установлен) →
  Piper → system → null;
* Silero доступен только когда есть python3.13, vendor/torch-py и веса
  v4_ru; без них — откат на Piper, без исключения при автовыборе;
* команда синтезатора: python, ``tools/silero_speak.py``, путь WAV,
  ``--speaker``, 48 кГц; текст — в stdin;
* окружение процесса: ``PYTHONPATH=vendor/torch-py``, ``TORCH_HOME``,
  ``SILERO_SPEAKER``;
* мок ``subprocess.Popen``: успех, ненулевой код, таймаут;
* кэш повторных фраз (повтор не запускает процесс) и ``SILERO_CACHE=0``;
* откат упавшего Silero на Piper с записью в лог; без запасного движка —
  ошибка, а не тишина;
* пресеты Piper игнорируются, явный голос Silero уважается.

Реальный torch не импортируется: ``Popen`` подменяется, а «установка»
Silero — это пустые файлы в temp-каталоге.
"""

from __future__ import annotations

import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eva import tts  # noqa: E402

#: Правдоподобный WAV: заголовок + данные, больше любого порога.
FAKE_WAV = b"RIFF" + b"\x00" * 9000


def make_executable(path: Path) -> Path:
    """Пустой исполняемый файл — заглушка бинаря/интерпретатора."""
    path.write_bytes(b"#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return path


def output_path_of(command: list[str]) -> Path | None:
    """Достать путь WAV из команды Silero или Piper."""
    for flag in ("--output_file", "-o", "-w"):
        if flag in command and command.index(flag) + 1 < len(command):
            return Path(command[command.index(flag) + 1])
    # Silero: python tools/silero_speak.py OUT.wav ...
    if len(command) > 2 and command[1].endswith("silero_speak.py"):
        return Path(command[2])
    return None


def fake_popen(**spec):
    """Заменить ``subprocess.Popen``; вернуть ``(класс, calls, stdin_texts, envs)``.

    ``fail_silero`` — процесс Silero завершается кодом 1 и не пишет файл;
    остальные (Piper fallback) работают штатно. ``envs`` — окружение,
    переданное в каждый ``Popen``.
    """
    options = {
        "audio": FAKE_WAV,
        "write": True,
        "polls": 1,
        "returncode": 0,
        "stdout": "",
        "stderr": "",
        "fail_silero": False,
    }
    options.update(spec)
    calls: list[list[str]] = []
    stdin_texts: list[str | None] = []
    envs: list[dict[str, str] | None] = []

    class FakePopen:
        def __init__(
            self,
            command,
            stdin=None,
            stdout=None,
            stderr=None,
            text=None,
            env=None,
            **kwargs,
        ):
            command = [str(item) for item in command]
            calls.append(command)
            envs.append(dict(env) if env is not None else None)
            target = output_path_of(command)
            is_silero = len(command) > 1 and command[1].endswith("silero_speak.py")
            if stdin is not None and hasattr(stdin, "read"):
                try:
                    stdin.seek(0)
                except (OSError, ValueError):
                    pass
                stdin_texts.append(stdin.read())
            else:
                stdin_texts.append(None)
            self.returncode = None
            self._polls = 0
            self._stdout = io.StringIO(options["stdout"])
            self._stderr = io.StringIO(options["stderr"])
            failed = bool(options["fail_silero"] and is_silero)
            if failed:
                self._stderr = io.StringIO("torch: boom")
            elif options["write"] and target is not None:
                target.write_bytes(options["audio"])
            self._returncode = options["returncode"] if not failed else 1

        def poll(self):
            self._polls += 1
            if self._polls >= options["polls"]:
                self.returncode = self._returncode
            return self.returncode

        def communicate(self, *args, **kwargs):
            if self.returncode is None:
                self.returncode = self._returncode
            return self._stdout.read(), self._stderr.read()

        def kill(self):
            self.returncode = -9

    return FakePopen, calls, stdin_texts, envs


class InstalledSileroCase(unittest.TestCase):
    """Общая «установка»: пустые, но правдоподобные файлы в temp-каталоге."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.python = make_executable(self.tmp / "python3.13")
        self.torch_dir = self.tmp / "torch-py"
        (self.torch_dir / "torch").mkdir(parents=True)
        (self.torch_dir / "torch" / "__init__.py").write_text("")
        self.model_dir = self.tmp / "silero"
        model = (
            self.model_dir
            / "hub"
            / "snakers4_silero-models_master"
            / "src"
            / "silero"
            / "model"
        )
        model.mkdir(parents=True)
        (model / "v4_ru.pt").write_bytes(b"\x00" * 1_100_000)
        self.script = self.tmp / "silero_speak.py"
        self.script.write_text("")
        self.piper_bin = make_executable(self.tmp / "piper")
        self.piper_model = self.tmp / "ru_RU-test-medium.onnx"
        self.piper_model.write_bytes(b"onnx")
        self.silero_env = {
            "SILERO_PYTHON": str(self.python),
            "SILERO_TORCH_DIR": str(self.torch_dir),
            "SILERO_MODEL_DIR": str(self.model_dir),
            "SILERO_SCRIPT": str(self.script),
        }
        self.piper_env = {
            "PIPER_BIN": str(self.piper_bin),
            "PIPER_MODEL": str(self.piper_model),
        }
        tts.clear_silero_cache()

    def engine(self, *, env=None, speaker=None) -> tts.SileroTTS:
        return tts.SileroTTS(
            python=self.python,
            torch_dir=self.torch_dir,
            model_dir=self.model_dir,
            script=self.script,
            speaker=speaker,
            env=self.silero_env if env is None else env,
        )


class SileroSelectionTests(InstalledSileroCase):
    """Порядок автовыбора и явный ``EVA_TTS``."""

    def test_auto_prefers_silero_over_piper(self):
        env = {**self.silero_env, **self.piper_env}
        engine = tts.select_engine(env=env)
        self.assertIsInstance(engine, tts.SileroTTS)
        self.assertTrue(engine.available())
        self.assertIn("silero", engine.describe())

    def test_elevenlabs_key_wins_over_silero(self):
        env = {**self.silero_env, **self.piper_env, "ELEVENLABS_API_KEY": "sk-x"}
        engine = tts.select_engine(env=env)
        self.assertIsInstance(engine, tts.ElevenLabsTTS)

    def test_without_silero_auto_uses_piper(self):
        empty = self.tmp / "empty-silero"
        empty.mkdir()
        env = {**self.piper_env, "SILERO_MODEL_DIR": str(empty)}
        engine = tts.select_engine(env=env)
        self.assertIsInstance(engine, tts.PiperTTS)

    def test_explicit_piper_overrides_installed_silero(self):
        env = {**self.silero_env, **self.piper_env, "EVA_TTS": "piper"}
        self.assertIsInstance(tts.select_engine(env=env), tts.PiperTTS)

    def test_explicit_silero_without_install_is_unavailable(self):
        engine = tts.select_engine(
            "silero", env={"SILERO_MODEL_DIR": str(self.tmp / "missing")}
        )
        self.assertIsInstance(engine, tts.SileroTTS)
        self.assertFalse(engine.available())
        with self.assertRaises(tts.TTSUnavailable) as caught:
            engine.synthesize("привет")
        self.assertIn("install_silero", str(caught.exception))

    def test_missing_torch_makes_silero_unavailable(self):
        env = dict(self.silero_env)
        env["SILERO_TORCH_DIR"] = str(self.tmp / "no-torch")
        engine = tts.SileroTTS(env=env)
        self.assertFalse(engine.available())
        with self.assertRaises(tts.TTSUnavailable) as caught:
            engine.synthesize("привет")
        self.assertIn("install_silero", str(caught.exception))

    def test_missing_python_makes_silero_unavailable(self):
        env = dict(self.silero_env)
        env["SILERO_PYTHON"] = str(self.tmp / "no-python")
        engine = tts.SileroTTS(env=env)
        self.assertFalse(engine.available())

    def test_empty_model_dir_is_not_install(self):
        empty = self.tmp / "empty-model"
        empty.mkdir()
        env = dict(self.silero_env)
        env["SILERO_MODEL_DIR"] = str(empty)
        self.assertFalse(tts.SileroTTS(env=env).available())

    def test_unknown_speaker_is_clear_error(self):
        engine = self.engine(speaker="robot")
        self.assertFalse(engine.available())
        with self.assertRaises(tts.TTSUnavailable) as caught:
            engine.synthesize("привет")
        self.assertIn("SILERO_SPEAKER", str(caught.exception))

    def test_isolated_env_does_not_autodiscover_vendor_install(self):
        # env-словарь без SILERO_* — явная конфигурация (тесты/встраивание):
        # установленный в репозитории Silero в неё не подмешивается.
        with mock.patch.object(
            tts.shutil, "which", return_value=None
        ), mock.patch.object(tts, "find_say_binary", return_value=None), \
                mock.patch.object(tts, "find_espeak_binary", return_value=None):
            engine = tts.select_engine(env={})
        self.assertIsInstance(engine, tts.NullTTS)


class SileroCommandTests(InstalledSileroCase):
    """Команда и окружение дочернего процесса."""

    def test_command_uses_script_and_48khz(self):
        engine = self.engine()
        command, stdin_text = engine.build_command(
            "Привет", self.tmp / "out.wav"
        )
        self.assertEqual(command[0], str(self.python))
        self.assertEqual(command[1], str(self.script))
        self.assertEqual(command[2], str(self.tmp / "out.wav"))
        self.assertEqual(command[command.index("--speaker") + 1], "baya")
        self.assertEqual(command[command.index("--sample-rate") + 1], "48000")
        self.assertEqual(stdin_text, "Привет\n")
        self.assertNotIn("Привет", command)

    def test_piper_presets_are_ignored(self):
        engine = self.engine()
        command, _ = engine.build_command(
            "Привет", self.tmp / "out.wav", voice="soft"
        )
        self.assertEqual(command[command.index("--speaker") + 1], "baya")

    def test_explicit_silero_voice_is_used(self):
        engine = self.engine()
        command, _ = engine.build_command(
            "Привет", self.tmp / "out.wav", voice="kseniya"
        )
        self.assertEqual(command[command.index("--speaker") + 1], "kseniya")

    def test_env_speaker_is_used(self):
        env = dict(self.silero_env, SILERO_SPEAKER="aidar")
        command, _ = self.engine(env=env).build_command(
            "Привет", self.tmp / "out.wav"
        )
        self.assertEqual(command[command.index("--speaker") + 1], "aidar")

    def test_subprocess_env_isolates_torch_and_model(self):
        engine = self.engine()
        env = engine.subprocess_env()
        self.assertIsNotNone(env)
        assert env is not None
        self.assertEqual(env["PYTHONPATH"], str(self.torch_dir))
        self.assertEqual(env["TORCH_HOME"], str(self.model_dir))
        self.assertEqual(env["SILERO_SPEAKER"], "baya")
        self.assertNotIn(".pylibs", env["PYTHONPATH"])


class SileroProcessTests(InstalledSileroCase):
    """Мок-процесс: успех, таймаут, кэш и откат."""

    def test_synthesize_runs_process_with_stdin_and_env(self):
        fake, calls, stdin_texts, envs = fake_popen()
        target = self.tmp / "reply.wav"
        with mock.patch.object(tts.subprocess, "Popen", fake):
            result = self.engine().synthesize_timed("Привет", target)
        self.assertTrue(target.is_file())
        self.assertEqual(result.engine, "silero")
        self.assertEqual(result.path, target)
        self.assertEqual(len(calls), 1)
        self.assertEqual(stdin_texts, ["Привет\n"])
        self.assertEqual(calls[0][1], str(self.script))
        self.assertEqual(envs[0]["PYTHONPATH"], str(self.torch_dir))
        self.assertEqual(envs[0]["TORCH_HOME"], str(self.model_dir))

    def test_repeated_phrase_is_served_from_cache(self):
        fake, calls, _, _ = fake_popen()
        engine = self.engine()
        with mock.patch.object(tts.subprocess, "Popen", fake):
            first = engine.synthesize_timed("Привет", self.tmp / "a.wav")
            second = engine.synthesize_timed("Привет", self.tmp / "b.wav")
        self.assertEqual(len(calls), 1)
        # В моке оба вызова быстрые; важно, что второй не запускал процесс.
        self.assertGreaterEqual(second.synth_ms, 0.0)
        self.assertLess(second.synth_ms, 500.0)
        self.assertEqual(first.engine, second.engine)
        self.assertTrue((self.tmp / "b.wav").is_file())
        self.assertEqual(tts.silero_cache_info()["hits"], 1)

    def test_cache_can_be_disabled(self):
        fake, calls, _, _ = fake_popen()
        engine = self.engine(env=dict(self.silero_env, SILERO_CACHE="0"))
        with mock.patch.object(tts.subprocess, "Popen", fake):
            engine.synthesize_timed("Привет", self.tmp / "a.wav")
            engine.synthesize_timed("Привет", self.tmp / "b.wav")
        self.assertEqual(len(calls), 2)

    def test_nonzero_exit_is_error(self):
        fake, _, _, _ = fake_popen(returncode=1, write=False, stderr="torch: no")
        with mock.patch.object(tts.subprocess, "Popen", fake):
            with self.assertRaises(tts.TTSError) as caught:
                self.engine().synthesize_timed(
                    "Привет", self.tmp / "reply.wav"
                )
        self.assertIn("кодом 1", str(caught.exception))
        self.assertIn("torch: no", str(caught.exception))

    def test_timeout_is_error(self):
        fake, _, _, _ = fake_popen(polls=10**9)
        with mock.patch.object(tts.subprocess, "Popen", fake):
            with self.assertRaises(tts.TTSError) as caught:
                self.engine().synthesize_timed(
                    "Привет", self.tmp / "reply.wav", timeout=0.01
                )
        self.assertIn("не уложился", str(caught.exception))

    def test_synthesize_audio_marks_silero_wav_as_unpitched(self):
        fake, _, _, _ = fake_popen()
        with mock.patch.object(tts.subprocess, "Popen", fake):
            data, content_type = tts.synthesize_audio(
                "Привет", prefer="silero", env=self.silero_env
            )
        self.assertTrue(data.startswith(b"RIFF"))
        self.assertEqual(content_type, "audio/wav; engine=silero")

    def test_failed_silero_falls_back_to_piper_and_logs(self):
        fake, calls, _, _ = fake_popen(fail_silero=True)
        env = {**self.silero_env, **self.piper_env}
        with mock.patch.object(tts.subprocess, "Popen", fake):
            with self.assertLogs(tts.logger, level="WARNING") as logs:
                data = tts.synthesize("Привет", prefer="silero", env=env)
        self.assertTrue(bytes(data).startswith(b"RIFF"))
        self.assertEqual(len(calls), 2)
        self.assertIn("откат на piper", "\n".join(logs.output))

    def test_failed_silero_without_fallback_raises(self):
        fake, _, _, _ = fake_popen(fail_silero=True)
        with mock.patch.object(tts.subprocess, "Popen", fake), \
                mock.patch.object(tts, "find_piper_binary", return_value=None), \
                mock.patch.object(tts, "find_say_binary", return_value=None), \
                mock.patch.object(tts, "find_espeak_binary", return_value=None), \
                mock.patch.object(tts.shutil, "which", return_value=None):
            with self.assertRaises(tts.TTSError) as caught:
                tts.synthesize("Привет", prefer="silero", env=self.silero_env)
        message = str(caught.exception)
        self.assertIn("Silero", message)
        self.assertIn("запасной", message)


if __name__ == "__main__":
    unittest.main()
