"""Плагин ``music`` (задача 0219): фразы, AppleScript, отказы, радио.

Тесты не запускают Music.app и не ходят в сеть: ``subprocess.run`` (osascript)
и ``subprocess.Popen`` (afplay) подменены. Живая проверка не входит в набор —
на машине без TCC-разрешения она висит на диалоге, поэтому плагин обязан
деградировать в честный текст, что здесь и проверяется.

Запуск::

    python3 -m unittest tests.test_music_control -v
"""

from __future__ import annotations

import importlib.util
import json
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eva.plugins import Permissions, PluginContext, PluginRegistry  # noqa: E402
from eva.plugins.manifest import load_manifest  # noqa: E402

PLUGIN_DIR = ROOT / "plugins" / "music"


def _load_plugin_module():
    """Загрузить ``plugins/music/plugin.py`` без пакета ``plugins``."""
    name = "test_music_plugin_module"
    spec = importlib.util.spec_from_file_location(name, PLUGIN_DIR / "plugin.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:  # pragma: no cover - падение загрузки должно быть видно
        sys.modules.pop(name, None)
        raise
    return module


MUSIC = _load_plugin_module()


class NextPrevPhrases(unittest.TestCase):
    """Переключение трека: шире, чем «следующая» — так люди и говорят."""

    PHRASES = {
        "next": [
            "следующий трек", "следующая песня", "дальше", "переключи",
            "переключи вперёд", "включи следующий трек", "следующую",
            "next",
        ],
        "prev": [
            "предыдущий трек", "предыдущая", "назад", "верни предыдущий трек",
            "верни прошлый трек", "вернись назад", "переключи назад",
            "прошлый трек", "включи предыдущую песню", "prev",
        ],
    }

    def test_phrases_map_to_actions(self) -> None:
        music = MUSIC
        for expected, phrases in self.PHRASES.items():
            for phrase in phrases:
                with self.subTest(phrase=phrase, expected=expected):
                    intent = music._intent(music._normalize(phrase))
                    self.assertIsNotNone(intent, phrase)
                    self.assertEqual(intent[0], expected, phrase)


class Completed:
    """Минимальный ``subprocess.CompletedProcess``."""

    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class FakeProcess:
    """Минимальный Popen для afplay: пишет сигналы и завершение."""

    def __init__(self) -> None:
        self.pid = 4242
        self.signals: list[int] = []
        self.terminated = False
        self.killed = False
        self._returncode: int | None = None

    def poll(self):
        return self._returncode

    def send_signal(self, sig) -> None:
        self.signals.append(sig)

    def terminate(self) -> None:
        self.terminated = True
        self._returncode = 0

    def wait(self, timeout=None) -> int:  # noqa: ANN001
        return 0

    def kill(self) -> None:
        self.killed = True
        self._returncode = -9


class MusicCase(unittest.TestCase):
    """Свежий экземпляр плагина и контекст на каждый тест."""

    def setUp(self) -> None:
        self.plugin = MUSIC.MusicPlugin()
        self.ctx = self.context()

    def context(self, **settings):
        return PluginContext(
            "music",
            plugin_dir=PLUGIN_DIR,
            permissions=Permissions(),
            settings=settings,
        )

    def answer(self, phrase: str, stdout: str = "", **completed):
        """Позвать answer с моком osascript; вернуть (ответ, мок, вызов)."""
        runner = mock.Mock(return_value=Completed(stdout=stdout, **completed))
        with mock.patch.object(MUSIC.subprocess, "run", runner):
            result = self.plugin.answer(self.ctx, phrase)
        call = runner.call_args
        return result, runner, call


# --------------------------------------------------------------------------- #
# Распознавание
# --------------------------------------------------------------------------- #


class TestPhrases(MusicCase):
    """>20 фраз: что услышали, что поняли."""

    CASES = [
        ("включи музыку", ("play", "")),
        ("включи что-нибудь", ("play", "")),
        ("включи что нибудь", ("play", "")),
        ("поставь музыку", ("play", "")),
        ("вруби музыку", ("play", "")),
        ("запусти плеер", ("play", "")),
        ("пауза", ("pause", "")),
        ("поставь на паузу", ("pause", "")),
        ("приостанови", ("pause", "")),
        ("продолжи", ("resume", "")),
        ("продолжи музыку", ("resume", "")),
        ("играй дальше", ("resume", "")),
        ("следующая", ("next", "")),
        ("следующий трек", ("next", "")),
        ("дальше", ("next", "")),
        ("предыдущая", ("prev", "")),
        ("предыдущий трек", ("prev", "")),
        ("назад", ("prev", "")),
        ("выключи музыку", ("stop", "")),
        ("останови музыку", ("stop", "")),
        ("стоп", ("stop", "")),
        ("громче", ("louder", "")),
        ("сделай громче", ("louder", "")),
        ("прибавь звук", ("louder", "")),
        ("тише", ("quieter", "")),
        ("убавь звук", ("quieter", "")),
        ("что играет", ("now", "")),
        ("что сейчас играет", ("now", "")),
        ("какая песня играет", ("now", "")),
        ("что за музыка", ("now", "")),
        ("включи Кино", ("search", "кино")),
        ("поставь группу Кино", ("search", "кино")),
        ("включи песню Кино", ("search", "кино")),
        ("хочу послушать Кино", ("search", "кино")),
        ("включи джаз", ("search", "джаз")),
        ("включи радио", ("radio", "")),
        ("включи радио дфм", ("radio", "дфм")),
        ("включи европу плюс", ("radio", "европу плюс")),
        ("заглуши", ("mute", "")),
        ("включи звук", ("unmute", "")),
    ]

    def test_phrases_are_recognized(self) -> None:
        for phrase, expected in self.CASES:
            with self.subTest(phrase=phrase):
                self.assertEqual(
                    MUSIC._intent(MUSIC._normalize(phrase)), expected
                )

    def test_foreign_phrases_are_not_ours(self) -> None:
        for phrase in (
            "включи свет",
            "выключи свет",
            "включи телевизор",
            "выключи плиту",
            "поставь будильник",
            "который час",
            "расскажи сказку",
            "как дела",
            "что такое джаз",
        ):
            with self.subTest(phrase=phrase):
                self.assertIsNone(MUSIC._intent(MUSIC._normalize(phrase)))

    def test_foreign_phrase_does_not_touch_osascript(self) -> None:
        result, runner, _call = self.answer("включи свет")
        self.assertIsNone(result)
        runner.assert_not_called()


# --------------------------------------------------------------------------- #
# AppleScript
# --------------------------------------------------------------------------- #


class TestAppleScript(MusicCase):
    """Маппинг команд в AppleScript и запрет shell-инъекций."""

    COMMANDS = [
        ("включи музыку", "Включаю музыку.", "tell application \"Music\" to play"),
        ("включи что-нибудь", "Включаю музыку.", "tell application \"Music\" to play"),
        ("продолжи", "Продолжаю.", "tell application \"Music\" to play"),
        ("пауза", "Пауза.", "tell application \"Music\" to pause"),
        ("следующая", "Следующая.", "tell application \"Music\" to next track"),
        ("предыдущая", "Предыдущая.", "tell application \"Music\" to previous track"),
        ("выключи музыку", "Выключила музыку.", "tell application \"Music\" to stop"),
        ("громче", "Сделала громче.", "sound volume"),
        ("тише", "Сделала тише.", "sound volume"),
        ("заглуши", "Звук выключен.", "sound volume to 0"),
        ("включи звук", "Звук включён.", "sound volume to 50"),
    ]

    def test_commands_map_to_osascript(self) -> None:
        for phrase, text, fragment in self.COMMANDS:
            with self.subTest(phrase=phrase):
                result, _runner, call = self.answer(phrase)
                self.assertEqual(result, text)
                argv = call.args[0]
                self.assertIsInstance(argv, list)
                self.assertEqual(argv[:2], ["/usr/bin/osascript", "-e"])
                self.assertIn(fragment, argv[2])
                self.assertNotIn("shell", call.kwargs)
                self.assertIs(call.kwargs.get("check"), False)
                self.assertIs(call.kwargs.get("text"), True)

    def test_volume_step_is_ten(self) -> None:
        _result, _runner, up = self.answer("громче")
        _result, _runner, down = self.answer("тише")
        self.assertIn("currentVolume + 10", up.args[0][2])
        self.assertIn("currentVolume - 10", down.args[0][2])

    def test_timeout_comes_from_settings(self) -> None:
        ctx = self.context(timeout=2)
        runner = mock.Mock(return_value=Completed())
        with mock.patch.object(MUSIC.subprocess, "run", runner):
            self.plugin.answer(ctx, "пауза")
        self.assertEqual(runner.call_args.kwargs["timeout"], 2.0)

    def test_quote_and_backslash_are_escaped(self) -> None:
        literal = MUSIC._applescript_literal('Кино "x" \\ y\nz')
        self.assertEqual(literal, '"Кино \\"x\\" \\\\ y z"')

    def test_answer_does_not_use_shell(self) -> None:
        phrase = 'включи "Кино" & (do shell script "touch /tmp/eva_pwned")'
        result, _runner, call = self.answer(phrase, stdout="not-found")
        self.assertIn("Не нашла в библиотеке", result)
        argv = call.args[0]
        self.assertIsInstance(argv, list)
        self.assertEqual(argv[:2], ["/usr/bin/osascript", "-e"])
        self.assertNotIn("shell", call.kwargs)
        # Пользовательский текст внутри AppleScript-литерала: кавычки из
        # фразы сняты нормализацией, а литерал начинается с системной «"».
        script = argv[2]
        self.assertTrue(script.startswith('set queryText to "'))
        self.assertIn("do shell script touch /tmp/eva_pwned", script)
        self.assertEqual(script.count("queryText to \""), 1)


class TestNowPlaying(MusicCase):
    def test_playing(self) -> None:
        result, _runner, call = self.answer(
            "что играет", stdout="playing\nКино — Звезда по имени Солнце\n"
        )
        self.assertEqual(result, "Играет: Кино — Звезда по имени Солнце.")
        self.assertIn("current track", call.args[0][2])

    def test_paused(self) -> None:
        result, _runner, _call = self.answer(
            "что играет", stdout="paused\nКино — Пачка сигарет\n"
        )
        self.assertEqual(result, "На паузе: Кино — Пачка сигарет.")

    def test_stopped(self) -> None:
        result, _runner, _call = self.answer("что играет", stdout="stopped\n")
        self.assertEqual(result, "Сейчас ничего не играет.")


class TestSearch(MusicCase):
    def test_playlist_is_played_first(self) -> None:
        result, _runner, call = self.answer(
            "включи Кино", stdout="playlist\nМой плейлист Кино\n"
        )
        self.assertEqual(result, "Включаю плейлист «Мой плейлист Кино».")
        script = call.args[0][2]
        playlist_at = script.index("every playlist whose name contains")
        track_at = script.index("every track whose name contains")
        artist_at = script.index("every track whose artist contains")
        self.assertLess(playlist_at, track_at)
        self.assertLess(track_at, artist_at)

    def test_track_by_name(self) -> None:
        result, _runner, _call = self.answer(
            "включи Кино", stdout="track\nКино — Звезда по имени Солнце\n"
        )
        self.assertEqual(result, "Включаю «Кино — Звезда по имени Солнце».")

    def test_track_by_artist(self) -> None:
        result, _runner, _call = self.answer(
            "включи Кино", stdout="artist\nКино — Группа крови\n"
        )
        self.assertEqual(result, "Включаю «Кино — Группа крови».")

    def test_not_found_is_honest(self) -> None:
        result, runner, call = self.answer("включи Кино", stdout="not-found\n")
        self.assertIn("Не нашла в библиотеке", result)
        self.assertIn("Кино", result)
        runner.assert_called_once()
        self.assertIn("кино", call.args[0][2])


# --------------------------------------------------------------------------- #
# Отказы среды
# --------------------------------------------------------------------------- #


class TestUnavailable(MusicCase):
    def test_osascript_missing(self) -> None:
        runner = mock.Mock(side_effect=FileNotFoundError("/usr/bin/osascript"))
        with mock.patch.object(MUSIC.subprocess, "run", runner):
            result = self.plugin.answer(self.ctx, "пауза")
        self.assertEqual(result, "Не могу управлять музыкой: на этом Mac нет osascript.")

    def test_timeout(self) -> None:
        runner = mock.Mock(
            side_effect=MUSIC.subprocess.TimeoutExpired(["osascript"], 5)
        )
        with mock.patch.object(MUSIC.subprocess, "run", runner):
            result = self.plugin.answer(self.ctx, "что играет")
        self.assertIn("Music.app не отвечает", result)
        self.assertIn("автоматизацию", result)

    def test_permission_denied(self) -> None:
        for stderr in (
            "execution error: Not authorized to send Apple events (-1743)",
            "28:40: execution error: Получена ошибка от «Music»: "
            "Произошло нарушение прав. (-10004)",
        ):
            with self.subTest(stderr=stderr):
                result, _runner, _call = self.answer(
                    "пауза", stderr=stderr, returncode=1
                )
                self.assertIn("macOS не разрешила", result)
                self.assertNotIn("Traceback", result)

    def test_object_not_found_error(self) -> None:
        result, _runner, _call = self.answer(
            "пауза",
            stderr="execution error: Can’t get current track. (-1728)",
            returncode=1,
        )
        self.assertIn("не нашла нужный объект", result)

    def test_generic_apple_script_error(self) -> None:
        result, _runner, _call = self.answer(
            "пауза",
            stderr="execution error: Music got an error (-2700)",
            returncode=1,
        )
        self.assertIn("Не удалось управлять Music.app", result)

    def test_error_is_a_string_not_exception(self) -> None:
        runner = mock.Mock(side_effect=OSError("boom"))
        with mock.patch.object(MUSIC.subprocess, "run", runner):
            result = self.plugin.answer(self.ctx, "следующая")
        self.assertIsInstance(result, str)
        self.assertTrue(result)


# --------------------------------------------------------------------------- #
# Радио через afplay
# --------------------------------------------------------------------------- #


class TestRadio(MusicCase):
    def test_default_radio(self) -> None:
        process = FakeProcess()
        popen = mock.Mock(return_value=process)
        with mock.patch.object(MUSIC.subprocess, "Popen", popen), \
                mock.patch.object(MUSIC.subprocess, "run") as runner:
            result = self.plugin.answer(self.ctx, "включи радио")
        self.assertEqual(result, "Включаю радио Европа Плюс.")
        argv = popen.call_args.args[0]
        self.assertEqual(argv[0], "/usr/bin/afplay")
        self.assertEqual(argv[1], "http://ep128.hostingradio.ru:8030/ep128")
        self.assertNotIn("shell", popen.call_args.kwargs)
        runner.assert_not_called()

    def test_named_station(self) -> None:
        process = FakeProcess()
        popen = mock.Mock(return_value=process)
        with mock.patch.object(MUSIC.subprocess, "Popen", popen):
            result = self.plugin.answer(self.ctx, "включи радио дфм")
        self.assertEqual(result, "Включаю радио DFM.")
        self.assertIn("dfm96.aacp", popen.call_args.args[0][1])

    def test_nashe_radio(self) -> None:
        process = FakeProcess()
        popen = mock.Mock(return_value=process)
        with mock.patch.object(MUSIC.subprocess, "Popen", popen):
            result = self.plugin.answer(self.ctx, "включи наше радио")
        self.assertEqual(result, "Включаю радио Наше Радио.")
        self.assertIn("nashe-128.mp3", popen.call_args.args[0][1])

    def test_station_without_word_radio(self) -> None:
        process = FakeProcess()
        popen = mock.Mock(return_value=process)
        with mock.patch.object(MUSIC.subprocess, "Popen", popen):
            result = self.plugin.answer(self.ctx, "включи европу плюс")
        self.assertEqual(result, "Включаю радио Европа Плюс.")

    def test_unknown_station(self) -> None:
        popen = mock.Mock()
        with mock.patch.object(MUSIC.subprocess, "Popen", popen):
            result = self.plugin.answer(self.ctx, "включи радио юность")
        self.assertIn("Такой радиостанции у меня нет", result)
        popen.assert_not_called()

    def test_radio_pause_and_resume(self) -> None:
        process = FakeProcess()
        self.plugin._radio = process
        self.plugin._radio_station = MUSIC._RADIO_STATIONS[0]
        self.assertEqual(self.plugin.answer(self.ctx, "пауза"), "Пауза.")
        self.assertIn(signal.SIGSTOP, process.signals)
        self.assertEqual(self.plugin.answer(self.ctx, "продолжи"), "Продолжаю.")
        self.assertIn(signal.SIGCONT, process.signals)

    def test_radio_stop_terminates_afplay(self) -> None:
        process = FakeProcess()
        self.plugin._radio = process
        self.plugin._radio_station = MUSIC._RADIO_STATIONS[0]
        self.assertEqual(
            self.plugin.answer(self.ctx, "выключи музыку"), "Выключила радио."
        )
        self.assertTrue(process.terminated)
        self.assertIsNone(self.plugin._radio)

    def test_radio_now_playing(self) -> None:
        process = FakeProcess()
        self.plugin._radio = process
        self.plugin._radio_station = MUSIC._RADIO_STATIONS[0]
        self.assertEqual(
            self.plugin.answer(self.ctx, "что играет"),
            "Играет радио Европа Плюс.",
        )

    def test_radio_next_is_honest(self) -> None:
        process = FakeProcess()
        self.plugin._radio = process
        self.plugin._radio_station = MUSIC._RADIO_STATIONS[0]
        self.assertEqual(
            self.plugin.answer(self.ctx, "следующая"),
            "В радио нет следующего трека.",
        )

    def test_afplay_missing(self) -> None:
        with mock.patch.object(
            MUSIC.subprocess, "Popen", side_effect=FileNotFoundError("afplay")
        ):
            result = self.plugin.answer(self.ctx, "включи радио")
        self.assertIn("нет afplay", result)

    def test_new_radio_stops_previous(self) -> None:
        first = FakeProcess()
        second = FakeProcess()
        popen = mock.Mock(side_effect=[first, second])
        with mock.patch.object(MUSIC.subprocess, "Popen", popen):
            self.plugin.answer(self.ctx, "включи радио")
            result = self.plugin.answer(self.ctx, "включи радио дфм")
        self.assertEqual(result, "Включаю радио DFM.")
        self.assertTrue(first.terminated)
        self.assertIs(self.plugin._radio, second)


# --------------------------------------------------------------------------- #
# Манифест и проводка реестра
# --------------------------------------------------------------------------- #


class TestManifestAndRegistry(unittest.TestCase):
    def test_manifest(self) -> None:
        manifest = load_manifest(PLUGIN_DIR)
        self.assertEqual(manifest.id, "music")
        self.assertIn("answer", manifest.capabilities)
        self.assertEqual(manifest.version, "1.0.0")
        self.assertIn("ep128.hostingradio.ru", manifest.permissions.network)
        self.assertIn("dfm.hostingradio.ru", manifest.permissions.network)

    def test_module_answer_and_register(self) -> None:
        self.assertTrue(callable(MUSIC.answer))
        ctx = PluginContext("music", plugin_dir=PLUGIN_DIR)
        plugin = MUSIC.register(ctx)
        self.assertTrue(callable(plugin.answer))

    def test_registry_hook_runs_plugin_before_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "plugins.json"
            config.write_text(
                json.dumps({"enabled": ["music"], "settings": {}}),
                encoding="utf-8",
            )
            registry = PluginRegistry(
                [ROOT / "plugins"], config, include_env=False
            )
            record = registry.get("music")
            self.assertTrue(record.loaded, record.error)
            self.assertTrue(registry.errors == [])
            bindings = registry.answers()
            self.assertEqual([item.plugin_id for item in bindings], ["music"])
            # Не наша тема — None, модель отвечает сама.
            self.assertIsNone(registry.run_answer("включи свет"))
            # Проводка answer(): мок вместо Music.app, без внешних процессов.
            record.obj._osascript = mock.Mock(return_value="")
            self.assertEqual(registry.run_answer("пауза"), "Пауза.")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
