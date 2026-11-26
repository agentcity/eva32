"""Проверки Silero-адаптера ``eva.tts`` без torch, модели, сети и звука.

Запуск::

    python3 -m unittest tests.test_tts_silero -v
    python3 -m unittest discover -s tests

Что проверяется (задачи 0193 и 0195):

* выбор движка: ``EVA_TTS`` → ElevenLabs (ключ) → Silero (установлен) →
  system → null (Piper из автовыбора убран, задача 0197);
* Silero доступен только когда есть python3.13, vendor/torch-py и веса
  v4_ru; без них — откат на system, без исключения при автовыборе;
* протокол worker'а (``tools/silero_worker.py``) без torch: ready/ping/synth,
  WAV в файл и в base64, ошибки на битый JSON/пустой текст/чужой голос,
  выход по ``shutdown`` и по EOF, SIGTERM — штатный код 0;
* клиент worker'а в ``eva/tts.py``: ленивый старт при первой фразе,
  keep-alive (второй вызов не поднимает процесс), ping на старте,
  один worker на процесс (синглтон), ``describe()`` с pid;
* падение worker'а: ровно один перезапуск, повторный сбой → ошибка и откат
  на system; таймаут → процесс убит, следующий вызов поднимает заново;
  ответ ``ok=false`` — ошибка фразы без перезапуска;
* конкурентные вызовы сериализуются lock'ом (ответы не путаются);
* LRU-кэш повторных фраз (worker не запускается вовсе) и ``SILERO_CACHE=0``;
* одноразовый путь ``SILERO_WORKER=0`` (0193) продолжает работать.

Реальный torch не импортируется: ``Popen`` подменяется эмулятором пайпов, а
«установка» Silero — это пустые файлы в temp-каталоге.
"""

from __future__ import annotations

import base64
import io
import json
import signal
import sys
import tempfile
import threading
import time
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eva import tts  # noqa: E402
from tools import silero_worker  # noqa: E402

#: Правдоподобный WAV: заголовок + данные, больше любого порога.
FAKE_WAV = b"RIFF" + b"\x00" * 9000


def make_executable(path: Path) -> Path:
    """Пустой исполняемый файл — заглушка бинаря/интерпретатора."""
    path.write_bytes(b"#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return path


def output_path_of(command: list[str]) -> Path | None:
    """Достать путь WAV из команды одноразового Silero или system."""
    for flag in ("--output_file", "-o", "-w"):
        if flag in command and command.index(flag) + 1 < len(command):
            return Path(command[command.index(flag) + 1])
    # Silero: python tools/silero_speak.py OUT.wav ...
    if len(command) > 2 and command[1].endswith("silero_speak.py"):
        return Path(command[2])
    return None


def fake_popen(**spec):
    """Заменить ``subprocess.Popen`` одноразовым фейком (путь 0193).

    ``fail_silero`` — процесс Silero завершается кодом 1 и не пишет файл;
    остальные (system fallback) работают штатно. Возвращает
    ``(класс, calls, stdin_texts, envs)``.
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


# --------------------------------------------------------------------------- #
# эмулятор worker-процесса: пайпы, JSON-строки, падения и зависания
# --------------------------------------------------------------------------- #
@dataclass
class WorkerBehaviour:
    """Что делает фейковый worker. Счётчики — по всем процессам сразу."""

    fail_start: bool = False
    die_at: set[int] = field(default_factory=set)
    die_always: bool = False
    hang_at: set[int] = field(default_factory=set)
    error: str | None = None
    delay: float = 0.0


class _LineStream:
    """Блокирующий поток строк: ``for line in stream`` как у настоящего пайпа."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._lines: list[str] = []
        self._closed = False

    def push(self, line: str) -> None:
        with self._cond:
            if self._closed:
                return
            self._lines.append(line + "\n")
            self._cond.notify_all()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def readline(self) -> str:
        with self._cond:
            while not self._lines and not self._closed:
                self._cond.wait(10)
            if self._lines:
                return self._lines.pop(0)
            return ""

    def __iter__(self):
        return self

    def __next__(self) -> str:
        line = self.readline()
        if not line:
            raise StopIteration
        return line


class _FakeStdin:
    """Приёмник запросов: каждая строка сразу уходит «процессу».

    Считает одновременные запросы: если клиент не сериализовал вызовы,
    ``max_active`` вырастет больше единицы — тест на lock это заметит.
    """

    def __init__(self, proc, guard: threading.Lock, stats: dict) -> None:
        self._proc = proc
        self._guard = guard
        self._stats = stats

    def write(self, text: str) -> None:
        for line in str(text).splitlines():
            if not line.strip():
                continue
            with self._guard:
                self._stats["active"] += 1
                self._stats["max_active"] = max(
                    self._stats["max_active"], self._stats["active"]
                )
            try:
                self._proc.handle(line)
            finally:
                with self._guard:
                    self._stats["active"] -= 1

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


def fake_worker_popen(**spec):
    """Заменить ``subprocess.Popen`` эмулятором worker'а (+ system для откатов).

    Возвращает ``(класс, behaviour, stats, created)``: ``created`` — все
    поднятые процессы, ``stats`` — счётчики синтезов и конкурентности.
    """
    behaviour = WorkerBehaviour(**spec)
    stats = {
        "synths": 0,
        "active": 0,
        "max_active": 0,
        "commands": [],
        "envs": [],
        "pids": [],
    }
    guard = threading.Lock()
    created: list["FakeWorkerProcess"] = []
    next_pid = [41000]

    class FakeWorkerProcess:
        def __init__(
            self,
            command,
            stdin=None,
            stdout=None,
            stderr=None,
            text=None,
            encoding=None,
            bufsize=None,
            env=None,
            **kwargs,
        ):
            self.command = [str(item) for item in command]
            self.env = dict(env) if env is not None else None
            stats["commands"].append(self.command)
            stats["envs"].append(self.env)
            self.is_worker = bool(
                len(self.command) > 1
                and self.command[1].endswith("silero_worker.py")
            )
            self.returncode = None
            self._cond = threading.Condition()
            self._polls = 0
            self.synth_count = 0
            self.requests: list[dict] = []
            self.stderr_text = ""

            if not self.is_worker:
                # say/espeak и прочие одноразовые движки: пишем WAV и выходим 0.
                self.pid = next_pid[0]
                next_pid[0] += 1
                target = output_path_of(self.command)
                if target is not None:
                    target.write_bytes(FAKE_WAV)
                self.stdin = None
                self.stdout = io.StringIO("")
                self.stderr = io.StringIO("")
                return

            next_pid[0] += 1
            self.pid = next_pid[0]
            stats["pids"].append(self.pid)
            created.append(self)
            self._lines = _LineStream()
            self.stdout = self._lines
            self.stdin = _FakeStdin(self, guard, stats)
            self.stderr = stderr

            if behaviour.fail_start:
                self.stderr_text = "torch: нет модели"
                if stderr is not None:
                    stderr.write(b"torch: net modela\n")
                    try:
                        stderr.flush()
                    except (OSError, ValueError):
                        pass
                self._finish(2)
            else:
                self._lines.push(
                    json.dumps(
                        {
                            "ok": True,
                            "ready": True,
                            "pid": self.pid,
                            "load_ms": 1234.0,
                            "import_ms": 800.0,
                            "model_ms": 434.0,
                            "torch": "fake",
                        }
                    )
                )

        # -- «процесс» ----------------------------------------------------
        def _finish(self, code: int) -> None:
            with self._cond:
                if self.returncode is None:
                    self.returncode = code
                self._cond.notify_all()
            self._lines.close()

        def handle(self, line: str) -> None:
            payload = json.loads(line)
            self.requests.append(payload)
            command = str(payload.get("cmd") or "synth")
            if command == "ping":
                self._lines.push(json.dumps({"ok": True, "pong": True, "pid": self.pid}))
                return
            if command == "shutdown":
                self._lines.push(json.dumps({"ok": True, "bye": True}))
                self._finish(0)
                return
            with guard:
                stats["synths"] += 1
                index = stats["synths"]
            self.synth_count += 1
            if behaviour.die_always or index in behaviour.die_at:
                self._finish(1)
                return
            if index in behaviour.hang_at:
                return
            if behaviour.delay:
                time.sleep(behaviour.delay)
            if behaviour.error:
                self._lines.push(
                    json.dumps({"ok": False, "error": behaviour.error})
                )
                return
            target = Path(str(payload["out"]))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(FAKE_WAV + str(payload.get("text", "")).encode())
            self._lines.push(
                json.dumps(
                    {
                        "ok": True,
                        "ms": 1.0,
                        "duration_s": 0.1,
                        "pcm_bytes": len(FAKE_WAV),
                        "out": str(target),
                        "speaker": payload.get("speaker"),
                    }
                )
            )

        # -- совместимость с Popen ---------------------------------------
        def poll(self):
            if not self.is_worker:
                self._polls += 1
                if self._polls >= 1 and self.returncode is None:
                    self.returncode = 0
            return self.returncode

        def wait(self, timeout=None):
            with self._cond:
                if timeout is None:
                    while self.returncode is None:
                        self._cond.wait(5)
                    return self.returncode
                deadline = time.monotonic() + float(timeout)
                while self.returncode is None:
                    left = deadline - time.monotonic()
                    if left <= 0:
                        raise tts.subprocess.TimeoutExpired(self.command, timeout)
                    self._cond.wait(min(left, 0.05))
                return self.returncode

        def terminate(self):
            self._finish(-15)

        def kill(self):
            self._finish(-9)

        def communicate(self, *args, **kwargs):
            if self.returncode is None:
                self.returncode = 0
            return "", ""

    return FakeWorkerProcess, behaviour, stats, created


# --------------------------------------------------------------------------- #
# «установка» Silero
# --------------------------------------------------------------------------- #
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
        tts.stop_silero_worker()
        self.addCleanup(tts.stop_silero_worker)

    def engine(self, *, env=None, speaker=None) -> tts.SileroTTS:
        return tts.SileroTTS(
            python=self.python,
            torch_dir=self.torch_dir,
            model_dir=self.model_dir,
            script=self.script,
            speaker=speaker,
            env=self.silero_env if env is None else env,
        )

    def one_shot_engine(self, *, env=None) -> tts.SileroTTS:
        """Адаптер в режиме 0193: ``SILERO_WORKER=0`` (одноразовый процесс)."""
        merged = dict(self.silero_env if env is None else env)
        merged["SILERO_WORKER"] = "0"
        return self.engine(env=merged)

    def install_fake(self, **spec):
        """Подменить ``Popen`` эмулятором worker'а; убрать за собой."""
        fake, behaviour, stats, created = fake_worker_popen(**spec)
        patcher = mock.patch.object(tts.subprocess, "Popen", fake)
        patcher.start()
        self.addCleanup(patcher.stop)
        return behaviour, stats, created


# --------------------------------------------------------------------------- #
# выбор движка
# --------------------------------------------------------------------------- #
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

    def test_without_silero_auto_uses_system_not_piper(self):
        # Задача 0197: даже с установленным Piper автовыбор идёт на system.
        empty = self.tmp / "empty-silero"
        empty.mkdir()
        env = {**self.piper_env, "SILERO_MODEL_DIR": str(empty)}
        engine = tts.select_engine(env=env)
        self.assertIsInstance(engine, tts.SystemTTS)
        self.assertEqual(engine.name, "system")

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


# --------------------------------------------------------------------------- #
# одноразовый путь 0193: команда и окружение
# --------------------------------------------------------------------------- #
class SileroCommandTests(InstalledSileroCase):
    """Команда и окружение одноразового процесса."""

    def test_command_uses_script_and_48khz(self):
        engine = self.one_shot_engine()
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

    def test_unknown_voice_falls_back_to_default(self):
        # Пресеты Piper (irina/soft/young) и прочие чужие имена — не голоса
        # Silero: используется настроенный baya, а не подмена на aidar.
        engine = self.one_shot_engine()
        for voice in ("irina", "soft", "young", "younger", "Milena", ""):
            command, _ = engine.build_command(
                "Привет", self.tmp / "out.wav", voice=voice
            )
            self.assertEqual(
                command[command.index("--speaker") + 1], "baya", voice
            )

    def test_each_silero_voice_is_used(self):
        # Три голоса из UI страницы и EVA.app + aidar через API/SILERO_SPEAKER.
        engine = self.one_shot_engine()
        for voice in ("baya", "kseniya", "xenia", "aidar"):
            command, _ = engine.build_command(
                "Привет", self.tmp / "out.wav", voice=voice
            )
            self.assertEqual(
                command[command.index("--speaker") + 1], voice
            )

    def test_explicit_silero_voice_is_used(self):
        engine = self.one_shot_engine()
        command, _ = engine.build_command(
            "Привет", self.tmp / "out.wav", voice="kseniya"
        )
        self.assertEqual(command[command.index("--speaker") + 1], "kseniya")

    def test_env_speaker_is_used(self):
        env = dict(self.silero_env, SILERO_SPEAKER="aidar")
        command, _ = self.one_shot_engine(env=env).build_command(
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


# --------------------------------------------------------------------------- #
# одноразовый путь 0193: процесс
# --------------------------------------------------------------------------- #
class SileroOneShotProcessTests(InstalledSileroCase):
    """``SILERO_WORKER=0``: процесс на фразу (старое поведение 0193)."""

    def test_synthesize_runs_process_with_stdin_and_env(self):
        fake, calls, stdin_texts, envs = fake_popen()
        target = self.tmp / "reply.wav"
        with mock.patch.object(tts.subprocess, "Popen", fake):
            result = self.one_shot_engine().synthesize_timed("Привет", target)
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
        engine = self.one_shot_engine()
        with mock.patch.object(tts.subprocess, "Popen", fake):
            first = engine.synthesize_timed("Привет", self.tmp / "a.wav")
            second = engine.synthesize_timed("Привет", self.tmp / "b.wav")
        self.assertEqual(len(calls), 1)
        self.assertGreaterEqual(second.synth_ms, 0.0)
        self.assertLess(second.synth_ms, 500.0)
        self.assertEqual(first.engine, second.engine)
        self.assertTrue((self.tmp / "b.wav").is_file())
        self.assertEqual(tts.silero_cache_info()["hits"], 1)

    def test_cache_can_be_disabled(self):
        fake, calls, _, _ = fake_popen()
        engine = self.one_shot_engine(
            env=dict(self.silero_env, SILERO_WORKER="0", SILERO_CACHE="0")
        )
        with mock.patch.object(tts.subprocess, "Popen", fake):
            engine.synthesize_timed("Привет", self.tmp / "a.wav")
            engine.synthesize_timed("Привет", self.tmp / "b.wav")
        self.assertEqual(len(calls), 2)

    def test_nonzero_exit_is_error(self):
        fake, _, _, _ = fake_popen(returncode=1, write=False, stderr="torch: no")
        with mock.patch.object(tts.subprocess, "Popen", fake):
            with self.assertRaises(tts.TTSError) as caught:
                self.one_shot_engine().synthesize_timed(
                    "Привет", self.tmp / "reply.wav"
                )
        self.assertIn("кодом 1", str(caught.exception))
        self.assertIn("torch: no", str(caught.exception))

    def test_timeout_is_error(self):
        fake, _, _, _ = fake_popen(polls=10**9)
        with mock.patch.object(tts.subprocess, "Popen", fake):
            with self.assertRaises(tts.TTSError) as caught:
                self.one_shot_engine().synthesize_timed(
                    "Привет", self.tmp / "reply.wav", timeout=0.01
                )
        self.assertIn("не уложился", str(caught.exception))

    def test_failed_silero_falls_back_to_system_and_logs(self):
        fake, calls, _, _ = fake_popen(fail_silero=True)
        env = {**self.silero_env, **self.piper_env, "SILERO_WORKER": "0"}
        with mock.patch.object(tts.subprocess, "Popen", fake):
            with self.assertLogs(tts.logger, level="WARNING") as logs:
                data = tts.synthesize("Привет", prefer="silero", env=env)
        self.assertTrue(bytes(data).startswith(b"RIFF"))
        self.assertEqual(len(calls), 2)
        # Второй процесс — системный say, а не Piper из окружения.
        self.assertEqual(Path(calls[1][0]).name, "say")
        self.assertIn("откат на system", "\n".join(logs.output))


# --------------------------------------------------------------------------- #
# протокол worker'а (без torch)
# --------------------------------------------------------------------------- #
class FakeBackend:
    """Заглушка модели: пишет WAV и отдаёт замеры, как настоящий backend."""

    def __init__(self, *, error: str | None = None) -> None:
        self.speakers = ("baya", "aidar")
        self.torch_version = "fake-2.15"
        self.import_ms = 12.0
        self.model_ms = 34.0
        self.jit_profiling = "off"
        self.error = error
        self.calls: list[tuple[str, str, Path | None, int | None]] = []

    def synthesize(self, text, speaker, out_path, *, sample_rate=None):
        self.calls.append((text, speaker, out_path, sample_rate))
        if self.error:
            raise RuntimeError(self.error)
        info = {
            "ms": 7.5,
            "duration_s": 0.1,
            "sample_rate": sample_rate or 48000,
            "speaker": speaker,
            "pcm_bytes": len(FAKE_WAV),
        }
        if out_path is None:
            info["wav_b64"] = base64.b64encode(FAKE_WAV).decode("ascii")
            return info
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(FAKE_WAV)
        info["out"] = str(out_path)
        return info


def run_worker(requests: list[str], *, backend=None) -> tuple[int, list[dict]]:
    """Прогнать ``serve`` на строках запросов; вернуть (код, ответы)."""
    stdin = io.StringIO("\n".join(requests) + "\n" if requests else "")
    stdout = io.StringIO()
    code = silero_worker.serve(
        backend=backend or FakeBackend(), stdin=stdin, stdout=stdout
    )
    replies = [
        json.loads(line)
        for line in stdout.getvalue().splitlines()
        if line.strip()
    ]
    return code, replies


class SileroWorkerProtocolTests(unittest.TestCase):
    """Протокол ``tools/silero_worker.py``: ready → ping → synth → EOF."""

    def test_ready_line_comes_first(self):
        code, replies = run_worker([])
        self.assertEqual(code, 0)
        self.assertEqual(len(replies), 1)
        ready = replies[0]
        self.assertTrue(ready["ok"])
        self.assertTrue(ready["ready"])
        self.assertEqual(ready["pid"], silero_worker.os.getpid())
        self.assertEqual(ready["protocol"], silero_worker.PROTOCOL_VERSION)
        self.assertEqual(ready["torch"], "fake-2.15")
        self.assertGreaterEqual(ready["rss_mb"], 0.0)
        self.assertGreaterEqual(ready["cpu_s"], 0.0)

    def test_ping_reports_pid_and_uptime(self):
        code, replies = run_worker(['{"cmd": "ping"}'])
        self.assertEqual(code, 0)
        self.assertEqual(len(replies), 2)
        pong = replies[1]
        self.assertTrue(pong["pong"])
        self.assertEqual(pong["pid"], silero_worker.os.getpid())
        self.assertIn("uptime_ms", pong)
        self.assertEqual(pong["requests"], 0)

    def test_synth_writes_wav_and_reports_metrics(self):
        tmp = tempfile.mkdtemp()
        out = Path(tmp) / "reply.wav"
        backend = FakeBackend()
        code, replies = run_worker(
            [json.dumps({"text": "Привет", "speaker": "baya", "out": str(out)})],
            backend=backend,
        )
        self.assertEqual(code, 0)
        reply = replies[1]
        self.assertTrue(reply["ok"])
        self.assertEqual(reply["out"], str(out))
        self.assertEqual(reply["speaker"], "baya")
        self.assertIn("ms", reply)
        self.assertEqual(out.read_bytes(), FAKE_WAV)
        self.assertEqual(backend.calls, [("Привет", "baya", out, 48000)])

    def test_synth_without_out_returns_base64(self):
        backend = FakeBackend()
        code, replies = run_worker(
            [json.dumps({"text": "Привет", "speaker": "aidar"})], backend=backend
        )
        self.assertEqual(code, 0)
        reply = replies[1]
        self.assertTrue(reply["ok"])
        self.assertNotIn("out", reply)
        self.assertEqual(base64.b64decode(reply["wav_b64"]), FAKE_WAV)
        self.assertIsNone(backend.calls[0][2])

    def test_bad_json_does_not_kill_worker(self):
        code, replies = run_worker(["{не json}", '{"cmd": "ping"}'])
        self.assertEqual(code, 0)
        self.assertFalse(replies[1]["ok"])
        self.assertIn("JSON", replies[1]["error"])
        self.assertTrue(replies[2]["pong"])

    def test_empty_text_is_request_error(self):
        code, replies = run_worker(['{"text": "   ", "out": "/tmp/x.wav"}'])
        self.assertEqual(code, 0)
        self.assertFalse(replies[1]["ok"])
        self.assertIn("пустой текст", replies[1]["error"])

    def test_unknown_speaker_lists_allowed(self):
        code, replies = run_worker(
            [json.dumps({"text": "Привет", "speaker": "robot", "out": "/tmp/x.wav"})]
        )
        self.assertEqual(code, 0)
        self.assertFalse(replies[1]["ok"])
        self.assertIn("неизвестный голос", replies[1]["error"])
        self.assertIn("baya", replies[1]["error"])

    def test_synth_error_keeps_worker_alive(self):
        backend = FakeBackend(error="голос сломался")
        code, replies = run_worker(
            [
                json.dumps({"text": "Привет", "speaker": "baya", "out": "/tmp/x.wav"}),
                '{"cmd": "ping"}',
            ],
            backend=backend,
        )
        self.assertEqual(code, 0)
        self.assertFalse(replies[1]["ok"])
        self.assertIn("голос сломался", replies[1]["error"])
        self.assertTrue(replies[2]["pong"])

    def test_unknown_command_is_error(self):
        code, replies = run_worker(['{"cmd": "rewind"}'])
        self.assertEqual(code, 0)
        self.assertFalse(replies[1]["ok"])
        self.assertIn("неизвестная команда", replies[1]["error"])

    def test_shutdown_exits_cleanly(self):
        code, replies = run_worker(['{"cmd": "shutdown"}', '{"cmd": "ping"}'])
        self.assertEqual(code, 0)
        self.assertTrue(replies[1]["bye"])
        # После shutdown worker не обрабатывает следующие строки.
        self.assertEqual(len(replies), 2)

    def test_eof_exits_with_zero(self):
        code, replies = run_worker(['{"cmd": "ping"}'])
        self.assertEqual(code, 0)

    def test_sigterm_handler_exits_zero(self):
        with mock.patch.object(silero_worker.sys, "stderr", io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                silero_worker._on_signal(signal.SIGTERM, None)
        self.assertEqual(caught.exception.code, 0)

    def test_failed_load_reports_not_ready(self):
        with mock.patch.object(
            silero_worker,
            "load_backend",
            side_effect=SystemExit(2),
        ):
            stdout = io.StringIO()
            code = silero_worker.serve(stdin=io.StringIO(""), stdout=stdout)
        self.assertEqual(code, 1)
        reply = json.loads(stdout.getvalue().strip())
        self.assertFalse(reply["ok"])
        self.assertFalse(reply["ready"])
        self.assertIn("stderr", reply["error"])

    def test_proc_stats_reports_positive_rss(self):
        stats = silero_worker._proc_stats()
        self.assertIn("rss_mb", stats)
        self.assertIn("cpu_s", stats)
        self.assertGreater(stats["rss_mb"], 0.0)


# --------------------------------------------------------------------------- #
# worker-клиент: старт, keep-alive, pid, окружение
# --------------------------------------------------------------------------- #
class SileroWorkerStartTests(InstalledSileroCase):
    """Ленивый старт, keep-alive, ping, один процесс на процесс чата."""

    def test_first_phrase_starts_worker_with_protocol(self):
        _, stats, created = self.install_fake()
        engine = self.engine()
        self.assertFalse(tts.silero_worker_info()["running"])
        target = self.tmp / "reply.wav"
        result = engine.synthesize_timed("Привет", target)
        self.assertEqual(result.engine, "silero")
        self.assertGreater(result.synth_ms, 0.0)
        self.assertTrue(target.read_bytes().startswith(b"RIFF"))
        self.assertEqual(len(created), 1)
        process = created[0]
        # Первый запрос — ping (проверка живости), второй — фраза.
        self.assertEqual(process.requests[0], {"cmd": "ping"})
        self.assertEqual(process.requests[1]["text"], "Привет")
        self.assertEqual(process.requests[1]["speaker"], "baya")
        self.assertEqual(process.requests[1]["out"], str(target))
        command = stats["commands"][0]
        self.assertEqual(command[0], str(self.python))
        self.assertTrue(command[1].endswith("silero_worker.py"))
        self.assertEqual(command[command.index("--speaker") + 1], "baya")
        self.assertEqual(command[command.index("--sample-rate") + 1], "48000")

    def test_worker_gets_vendored_torch_env(self):
        _, stats, _ = self.install_fake()
        self.engine().synthesize_timed("Привет", self.tmp / "a.wav")
        env = stats["envs"][0]
        self.assertEqual(env["PYTHONPATH"], str(self.torch_dir))
        self.assertEqual(env["TORCH_HOME"], str(self.model_dir))
        self.assertEqual(env["SILERO_SPEAKER"], "baya")

    def test_second_phrase_reuses_same_process(self):
        _, stats, created = self.install_fake()
        engine = self.engine()
        engine.synthesize_timed("Привет", self.tmp / "a.wav")
        engine.synthesize_timed("Как дела?", self.tmp / "b.wav")
        self.assertEqual(len(created), 1)
        self.assertEqual(len(stats["pids"]), 1)
        self.assertEqual(tts.silero_worker_info()["requests"], 2)

    def test_worker_is_singleton_between_adapters(self):
        _, _, created = self.install_fake()
        first = self.engine()
        second = self.engine()
        first.synthesize_timed("Привет", self.tmp / "a.wav")
        second.synthesize_timed("Привет ещё раз", self.tmp / "b.wav")
        self.assertEqual(len(created), 1)
        self.assertEqual(tts.silero_worker_info()["pid"], created[0].pid)

    def test_cached_phrase_does_not_start_worker(self):
        _, _, created = self.install_fake()
        engine = self.engine()
        engine.synthesize_timed("Привет", self.tmp / "a.wav")
        self.assertEqual(len(created), 1)
        tts.stop_silero_worker()
        engine.synthesize_timed("Привет", self.tmp / "b.wav")
        self.assertEqual(len(created), 1)
        self.assertFalse(tts.silero_worker_info()["running"])
        self.assertEqual(tts.silero_cache_info()["hits"], 1)

    def test_describe_shows_worker_pid(self):
        _, _, created = self.install_fake()
        engine = self.engine()
        self.assertIn("worker не запущен", engine.describe())
        engine.synthesize_timed("Привет", self.tmp / "a.wav")
        text = engine.describe()
        self.assertIn(f"silero (worker, pid {created[0].pid}", text)
        self.assertIn("baya", text)

    def test_voice_and_sample_rate_reach_worker(self):
        _, _, created = self.install_fake()
        self.engine().synthesize_timed(
            "Привет", self.tmp / "a.wav", voice="kseniya"
        )
        synth = created[0].requests[1]
        self.assertEqual(synth["speaker"], "kseniya")

    def test_prewarm_starts_worker_and_warms_up_synth(self):
        _, _, created = self.install_fake()
        engine = self.engine()
        self.assertEqual(engine.prewarm(), created[0].pid)
        info = tts.silero_worker_info()
        self.assertTrue(info["running"])
        # ping + короткий пробный синтез: первый настоящий вызов уже тёплый.
        self.assertEqual(info["requests"], 1)
        engine.synthesize_timed("Привет", self.tmp / "a.wav")
        self.assertEqual(len(created), 1)
        commands = [r.get("cmd") or "synth" for r in created[0].requests]
        self.assertEqual(commands, ["ping", "synth", "synth"])
        self.assertEqual(created[0].requests[1]["text"], "Э.")
        self.assertEqual(tts.silero_worker_info()["requests"], 2)


# --------------------------------------------------------------------------- #
# worker-клиент: падения, таймаут, ошибки
# --------------------------------------------------------------------------- #
class SileroWorkerFailureTests(InstalledSileroCase):
    """Падение, повторный сбой, таймаут и откат на system."""

    def test_crash_triggers_exactly_one_restart(self):
        _, _, created = self.install_fake(die_at={1})
        engine = self.engine()
        target = self.tmp / "reply.wav"
        with self.assertLogs(tts.logger, level="WARNING") as logs:
            result = engine.synthesize_timed("Привет", target)
        self.assertEqual(len(created), 2)
        self.assertTrue(target.read_bytes().startswith(b"RIFF"))
        self.assertGreater(result.synth_ms, 0.0)
        self.assertIn("перезапускаю", "\n".join(logs.output))
        self.assertEqual(tts.silero_worker_info()["pid"], created[1].pid)

    def test_two_crashes_raise_and_fall_back_to_system(self):
        _, stats, created = self.install_fake(die_at={1, 2})
        env = {**self.silero_env, **self.piper_env}
        with self.assertLogs(tts.logger, level="WARNING") as logs:
            data = tts.synthesize("Привет", prefer="silero", env=env)
        self.assertTrue(bytes(data).startswith(b"RIFF"))
        self.assertEqual(len(created), 2)
        # Третий процесс — уже system, а не Piper из окружения.
        self.assertFalse(any("piper" in cmd[0] for cmd in stats["commands"]))
        self.assertTrue(any(Path(cmd[0]).name == "say" for cmd in stats["commands"]))
        self.assertIn("откат на system", "\n".join(logs.output))

    def test_two_crashes_without_fallback_raise(self):
        self.install_fake(die_always=True)
        with mock.patch.object(tts, "find_piper_binary", return_value=None), \
                mock.patch.object(tts, "find_say_binary", return_value=None), \
                mock.patch.object(tts, "find_espeak_binary", return_value=None), \
                mock.patch.object(tts.shutil, "which", return_value=None):
            with self.assertRaises(tts.TTSError) as caught:
                tts.synthesize(
                    "Привет", prefer="silero", env=dict(self.silero_env)
                )
        message = str(caught.exception)
        self.assertIn("Silero", message)
        self.assertIn("запасной", message)

    def test_timeout_kills_worker_and_next_call_restarts(self):
        _, _, created = self.install_fake(hang_at={1})
        env = dict(self.silero_env, SILERO_WORKER_TIMEOUT="0.2")
        engine = self.engine(env=env)
        started = time.perf_counter()
        with self.assertRaises(tts.TTSError) as caught:
            engine.synthesize_timed("Привет", self.tmp / "a.wav")
        elapsed = time.perf_counter() - started
        self.assertIn("не ответил", str(caught.exception))
        self.assertLess(elapsed, 5.0)
        self.assertIsNotNone(created[0].returncode)
        self.assertFalse(tts.silero_worker_info()["running"])
        # Следующий вызов поднимает worker заново и синтезирует.
        result = engine.synthesize_timed("Другая фраза", self.tmp / "b.wav")
        self.assertEqual(len(created), 2)
        self.assertGreater(result.synth_ms, 0.0)
        self.assertTrue((self.tmp / "b.wav").is_file())

    def test_worker_error_reply_is_tts_error_without_restart(self):
        _, _, created = self.install_fake(error="голос не поддержан")
        with self.assertRaises(tts.TTSError) as caught:
            self.engine().synthesize_timed("Привет", self.tmp / "a.wav")
        self.assertIn("голос не поддержан", str(caught.exception))
        self.assertEqual(len(created), 1)
        self.assertTrue(tts.silero_worker_info()["running"])

    def test_failed_worker_start_falls_back_to_system(self):
        _, stats, created = self.install_fake(fail_start=True)
        env = {**self.silero_env, **self.piper_env}
        with self.assertLogs(tts.logger, level="WARNING") as logs:
            data = tts.synthesize("Привет", prefer="silero", env=env)
        self.assertTrue(bytes(data).startswith(b"RIFF"))
        self.assertEqual(len(created), 1)
        self.assertFalse(tts.silero_worker_info()["running"])
        self.assertFalse(any("piper" in cmd[0] for cmd in stats["commands"]))
        self.assertTrue(any(Path(cmd[0]).name == "say" for cmd in stats["commands"]))
        self.assertIn("откат на system", "\n".join(logs.output))

    def test_missing_worker_script_is_unavailable(self):
        env = dict(self.silero_env, SILERO_WORKER_SCRIPT=str(self.tmp / "нет.py"))
        engine = self.engine(env=env)
        # Сломанная установка видна сразу: автовыбор уйдёт на system.
        self.assertFalse(engine.available())
        with self.assertRaises(tts.TTSUnavailable) as caught:
            engine.synthesize_timed("Привет", self.tmp / "a.wav")
        self.assertIn("worker не найден", str(caught.exception))

    def test_one_shot_mode_does_not_require_worker_script(self):
        env = dict(
            self.silero_env,
            SILERO_WORKER="0",
            SILERO_WORKER_SCRIPT=str(self.tmp / "нет.py"),
        )
        self.assertTrue(self.engine(env=env).available())

    def test_stop_silero_worker_is_idempotent(self):
        _, _, created = self.install_fake()
        engine = self.engine()
        engine.synthesize_timed("Привет", self.tmp / "a.wav")
        tts.stop_silero_worker()
        self.assertFalse(tts.silero_worker_info()["running"])
        self.assertEqual(created[0].returncode, 0)
        tts.stop_silero_worker()  # повторно — no-op
        self.assertFalse(tts.silero_worker_info()["running"])


# --------------------------------------------------------------------------- #
# worker-клиент: конкурентные вызовы
# --------------------------------------------------------------------------- #
class SileroWorkerLockTests(InstalledSileroCase):
    """Lock: один запрос за раз, ответы не путаются между потоками."""

    def test_concurrent_calls_are_serialized(self):
        _, stats, created = self.install_fake(delay=0.01)
        engine = self.engine()
        phrases = [f"фраза номер {index}" for index in range(4)]
        results: dict[str, bytes] = {}
        errors: list[BaseException] = []

        def speak(text: str) -> None:
            try:
                target = self.tmp / f"{abs(hash(text))}.wav"
                engine.synthesize_timed(text, target)
                results[text] = target.read_bytes()
            except BaseException as exc:  # noqa: BLE001 — тест должен видеть всё
                errors.append(exc)

        threads = [threading.Thread(target=speak, args=(text,)) for text in phrases]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)

        self.assertEqual(errors, [])
        self.assertEqual(len(results), len(phrases))
        self.assertEqual(len(created), 1)
        self.assertEqual(stats["max_active"], 1)
        # Каждый поток получил свой WAV, а не ответ соседа.
        for text, data in results.items():
            self.assertEqual(data, FAKE_WAV + text.encode())


if __name__ == "__main__":
    unittest.main()
