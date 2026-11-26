"""Готовая озвучка сказок (задачи 0223 и 0231): запись вместо пересказа.

Проверяем пять границ:

* фразы «включи запись сказки», «поставь аудиосказку», «сказку в
  исполнении» запускают локальный файл через ``afplay`` (список argv, без
  shell, ``start_new_session``), а «выключи»/«стоп сказку» — завершают;
* темп — детский 0.9 в argv ``afplay -r``; «медленнее/быстрее/обычный темп»
  меняют его шагом 0.05 в границах 0.75…1.15, а смена во время игры
  перезапускает запись с той же позиции (или честно начинает сначала);
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

    def answer_with(self, phrase: str, process: FakeProcess | None = None):
        """Ответить на команду, подменив ``afplay`` и системный звук.

        Запись уже может играть: Popen мок нужен и для перезапуска с новым
        темпом, а ``_system_volume`` — чтобы тест не трогал звук Mac.
        """
        process = process if process is not None else FakeProcess()
        with mock.patch.object(
            fairy.subprocess, "Popen", return_value=process
        ) as popen, mock.patch.object(fairy.time, "sleep"), mock.patch.object(
            fairy.FairyTalesPlugin, "_system_volume", return_value=None
        ), mock.patch.object(
            fairy.FairyTalesPlugin, "_set_system_volume", return_value=False
        ):
            result = self.plugin.answer(self.ctx, phrase)
        return result, popen, process


def popen_argv(popen) -> list[str]:
    """argv процесса ``afplay`` из мока ``Popen``."""
    call_args = popen.call_args
    assert call_args is not None
    return list(call_args[0][0])


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
                call, kwargs = popen.call_args
                self.assertEqual(call[0][0], fairy.AFPLAY)
                self.assertEqual(call[0][1:3], ["-r", "0.9"], "детский темп")
                path = Path(call[0][-1])
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
        self.assertEqual(Path(popen_argv(popen)[-1]).stem, "baba-yaga")

    def test_pro_kota_prefers_tale_with_recording(self) -> None:
        # «Кот и лиса» короче и без записи; запись есть у «Кот, петух и лиса».
        result, popen, _ = self.start("Включи запись сказки про кота")
        assert isinstance(result, dict)
        text = answer_text(result)
        assert text is not None
        self.assertIn("Кот, петух и лиса", text)
        self.assertEqual(Path(popen_argv(popen)[-1]).stem, "kot-petuh-i-lisa")

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


class TestAudioPace(AudioCase):
    """Темп готовой записи (задача 0231): 0.9 по умолчанию и команды."""

    def play_for(self, seconds: float = 30.0):
        """Включить «Теремок» и сделать вид, что прошло ``seconds`` секунд."""
        _, popen, process = self.start("Включи запись сказки про теремок")
        with self.plugin._audio_lock:
            assert self.plugin._audio_started_at is not None
            self.plugin._audio_started_at -= seconds
        return popen, process

    def test_default_rate_in_argv(self) -> None:
        popen, _ = self.play_for(0.0)
        argv = popen_argv(popen)
        self.assertEqual(argv[:3], [fairy.AFPLAY, "-r", "0.9"])
        self.assertEqual(self.plugin.audio_rate, fairy.DEFAULT_RATE)
        self.assertTrue(argv[-1].endswith(".mp3"))

    def test_slower_restarts_same_position(self) -> None:
        _, old_process = self.play_for(30.0)
        # 30 секунд на 0.9 — это 27 секунд записи.
        self.assertAlmostEqual(self.plugin.audio_position, 27.0, places=1)
        with tempfile.TemporaryDirectory() as tmp:
            trimmed = Path(tmp) / "eva-tale-trim.wav"
            trimmed.write_bytes(b"RIFF" + b"\x00" * 128)
            new_process = FakeProcess()
            with mock.patch.object(
                fairy.FairyTalesPlugin, "_trim_audio", return_value=trimmed
            ) as trim:
                result, popen, _ = self.answer_with("Медленнее", new_process)
            self.assertTrue(old_process.terminated, "старый процесс должен уйти")
            argv = popen_argv(popen)
            self.assertEqual(argv[:3], [fairy.AFPLAY, "-r", "0.85"])
            self.assertEqual(argv[-1], str(trimmed), "продолжаем с обрезки")
            trim.assert_called_once()
            self.assertAlmostEqual(trim.call_args[0][1], 27.0, places=1)
        assert isinstance(result, dict)
        self.assertTrue(result.get("silent"))
        text = answer_text(result)
        assert text is not None
        self.assertIn("85 процентах", text)
        self.assertNotIn("сначала", text)
        self.assertEqual(self.plugin.audio_rate, 0.85)
        self.assertAlmostEqual(self.plugin.audio_position, 27.0, places=1)
        self.assertIs(self.plugin._audio_process, new_process)

    def test_faster_restarts_same_position(self) -> None:
        self.play_for(20.0)
        with tempfile.TemporaryDirectory() as tmp:
            trimmed = Path(tmp) / "eva-tale-fast.wav"
            trimmed.write_bytes(b"RIFF" + b"\x00" * 128)
            with mock.patch.object(
                fairy.FairyTalesPlugin, "_trim_audio", return_value=trimmed
            ):
                result, popen, _ = self.answer_with("Быстрее")
            self.assertEqual(popen_argv(popen)[:3], [fairy.AFPLAY, "-r", "0.95"])
            self.assertAlmostEqual(self.plugin.audio_position, 18.0, places=1)
        text = answer_text(result)
        assert text is not None
        self.assertIn("95 процентах", text)

    def test_honest_restart_when_position_lost(self) -> None:
        self.play_for(15.0)
        new_process = FakeProcess()
        with mock.patch.object(
            fairy.FairyTalesPlugin, "_trim_audio", return_value=None
        ):
            result, popen, _ = self.answer_with("Читай медленнее", new_process)
        argv = popen_argv(popen)
        self.assertEqual(argv[:3], [fairy.AFPLAY, "-r", "0.85"])
        self.assertTrue(argv[-1].endswith(".mp3"), "играем исходный файл")
        self.assertAlmostEqual(self.plugin.audio_position, 0.0, places=1)
        text = answer_text(result)
        assert text is not None
        self.assertIn("сначала", text)
        self.assertIn("не получилось сохранить место", text)

    def test_slower_never_below_minimum(self) -> None:
        self.play_for(10.0)
        with mock.patch.object(
            fairy.FairyTalesPlugin, "_trim_audio", return_value=None
        ):
            for _ in range(5):
                self.answer_with("Читай медленнее")
                self.assertGreaterEqual(self.plugin.audio_rate, fairy.MIN_RATE)
        self.assertEqual(self.plugin.audio_rate, fairy.MIN_RATE)
        with mock.patch.object(
            fairy.FairyTalesPlugin, "_trim_audio", return_value=None
        ):
            result, popen, _ = self.answer_with("Медленнее")
        popen.assert_not_called()
        text = answer_text(result)
        assert text is not None
        self.assertIn("75 процентах", text)
        self.assertIn("нельзя", text)

    def test_faster_never_above_maximum(self) -> None:
        self.play_for(10.0)
        with mock.patch.object(
            fairy.FairyTalesPlugin, "_trim_audio", return_value=None
        ):
            for _ in range(8):
                self.answer_with("Читай быстрее")
                self.assertLessEqual(self.plugin.audio_rate, fairy.MAX_RATE)
        self.assertEqual(self.plugin.audio_rate, fairy.MAX_RATE)
        result, popen, _ = self.answer_with("Быстрее")
        popen.assert_not_called()
        text = answer_text(result)
        assert text is not None
        self.assertIn("115 процентах", text)
        self.assertIn("нельзя", text)

    def test_normal_tempo_returns_one(self) -> None:
        self.play_for(12.0)
        with tempfile.TemporaryDirectory() as tmp:
            trimmed = Path(tmp) / "eva-tale-normal.wav"
            trimmed.write_bytes(b"RIFF" + b"\x00" * 128)
            with mock.patch.object(
                fairy.FairyTalesPlugin, "_trim_audio", return_value=trimmed
            ):
                result, popen, _ = self.answer_with("Обычный темп")
            self.assertEqual(popen_argv(popen)[:3], [fairy.AFPLAY, "-r", "1.0"])
        self.assertEqual(self.plugin.audio_rate, fairy.NORMAL_RATE)
        text = answer_text(result)
        assert text is not None
        self.assertIn("100 процентах", text)

    def test_tempo_setting_survives_between_recordings(self) -> None:
        # «Медленнее» без игры запоминает темп для следующей записи.
        self.play_for(5.0)
        self.plugin.answer(self.ctx, "Выключи")
        result, popen, _ = self.answer_with("Читай медленнее")
        self.assertIsInstance(result, str)
        assert isinstance(result, str)
        self.assertIn("85 процентах", result)
        popen.assert_not_called()
        self.assertEqual(self.plugin.audio_rate, 0.85)
        _, popen, _ = self.start("Включи запись сказки про бабу-ягу")
        self.assertEqual(popen_argv(popen)[:3], [fairy.AFPLAY, "-r", "0.85"])

    def test_pace_commands_keep_one_process(self) -> None:
        _, old_process = self.play_for(10.0)
        with mock.patch.object(
            fairy.FairyTalesPlugin, "_trim_audio", return_value=None
        ):
            second = FakeProcess()
            _, popen, _ = self.answer_with("Медленнее", second)
        self.assertTrue(old_process.terminated)
        self.assertIs(self.plugin._audio_process, second)
        popen.assert_called_once()
        # И «выключи» после смены темпа гасит именно новый процесс.
        self.assertEqual(self.plugin.answer(self.ctx, "Выключи"), "Выключила запись.")
        self.assertTrue(second.terminated)
        self.assertIsNone(self.plugin._audio_process)

    def test_restart_failure_is_honest(self) -> None:
        # afplay пропал между запусками: не делаем вид, что запись играет.
        self.play_for(10.0)
        with mock.patch.object(
            fairy.FairyTalesPlugin, "_trim_audio", return_value=None
        ), mock.patch.object(
            fairy.subprocess, "Popen", side_effect=OSError("afplay")
        ), mock.patch.object(fairy.time, "sleep"):
            result = self.plugin.answer(self.ctx, "Медленнее")
        self.assertIsInstance(result, str)
        assert isinstance(result, str)
        self.assertIn("не получилось", result)
        self.assertIsNone(self.plugin._audio_process)
        self.assertEqual(self.plugin.audio_rate, 0.85)

    def test_trim_file_removed_on_stop(self) -> None:
        self.play_for(10.0)
        with tempfile.TemporaryDirectory() as tmp:
            trimmed = Path(tmp) / "eva-tale-trim.wav"
            trimmed.write_bytes(b"RIFF" + b"\x00" * 128)
            with mock.patch.object(
                fairy.FairyTalesPlugin, "_trim_audio", return_value=trimmed
            ):
                self.answer_with("Медленнее")
            self.assertTrue(trimmed.is_file(), "пока играет — файл нужен")
            self.plugin.answer(self.ctx, "Выключи")
            self.assertFalse(trimmed.is_file(), "после «выключи» временный WAV не нужен")

    def test_bare_pace_without_recording_falls_through(self) -> None:
        fresh = fairy.FairyTalesPlugin(rng=random.Random(1))
        self.assertIsNone(fresh.answer(self.ctx, "Медленнее"))
        self.assertIsNone(fresh.answer(self.ctx, "Быстрее"))
        # С контекстом чтения — это команда, даже если запись ещё не играет.
        self.assertIn(
            "85 процентах", fresh.answer(self.ctx, "Читай медленнее") or ""
        )
        self.assertEqual(fresh.audio_rate, 0.85)

    def test_plain_chat_not_affected(self) -> None:
        fresh = fairy.FairyTalesPlugin(rng=random.Random(1))
        for phrase in (
            "Который час",
            "Расскажи про погоду",
            "Включи музыку",
            "Включи музыку быстрее",
            "Поезд едет быстрее самолёта",
            "Обычный день",
        ):
            with self.subTest(phrase=phrase):
                self.assertIsNone(fresh.answer(self.ctx, phrase))
        self.assertEqual(fresh.audio_rate, fairy.DEFAULT_RATE)

    def test_story_request_with_adverb_is_not_pace(self) -> None:
        # «Расскажи медленно сказку» — просьба сказки, а не команда темпа.
        fresh = fairy.FairyTalesPlugin(rng=random.Random(1))
        result = fresh.answer(self.ctx, "Расскажи медленно сказку про репку")
        text = answer_text(result)
        assert text is not None
        self.assertIn("Слушай сказку", text)
        self.assertNotIn("скорости", text)
        self.assertEqual(fresh.audio_rate, fairy.DEFAULT_RATE)


class TestTrimAudio(unittest.TestCase):
    """``afconvert --offset``: позиция для продолжения в кадрах, не «на глаз»."""

    def test_offset_is_position_in_frames(self) -> None:
        plugin = fairy.FairyTalesPlugin(rng=random.Random(1))
        calls: list[list[str]] = []

        class Result:
            returncode = 0

        def fake_run(argv, **kwargs):
            calls.append(list(argv))
            Path(argv[-1]).write_bytes(b"RIFF" + b"\x00" * 256)
            return Result()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "teremok.mp3"
            path.write_bytes(b"x")
            with mock.patch.object(
                fairy.FairyTalesPlugin, "_audio_probe", return_value=(22050, 138.24)
            ), mock.patch.object(fairy.subprocess, "run", side_effect=fake_run):
                trimmed = plugin._trim_audio(path, 27.0)
            self.assertIsNotNone(trimmed)
            assert trimmed is not None
            self.addCleanup(trimmed.unlink, missing_ok=True)
            argv = calls[0]
            self.assertEqual(argv[0], fairy.AFCONVERT)
            self.assertEqual(
                argv[argv.index("--offset") + 1], str(int(27.0 * 22050))
            )
            self.assertEqual(argv[-3:], [str(path), "-o", str(trimmed)])

    def test_no_trim_when_probe_failed(self) -> None:
        plugin = fairy.FairyTalesPlugin(rng=random.Random(1))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "teremok.mp3"
            path.write_bytes(b"x")
            with mock.patch.object(
                fairy.FairyTalesPlugin, "_audio_probe", return_value=(None, None)
            ), mock.patch.object(fairy.subprocess, "run") as run:
                self.assertIsNone(plugin._trim_audio(path, 10.0))
            run.assert_not_called()

    def test_no_trim_past_the_end(self) -> None:
        plugin = fairy.FairyTalesPlugin(rng=random.Random(1))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "teremok.mp3"
            path.write_bytes(b"x")
            with mock.patch.object(
                fairy.FairyTalesPlugin, "_audio_probe", return_value=(22050, 30.0)
            ), mock.patch.object(fairy.subprocess, "run") as run:
                self.assertIsNone(plugin._trim_audio(path, 29.5))
            run.assert_not_called()


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
