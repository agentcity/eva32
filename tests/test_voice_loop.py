"""Проверки голосового прототипа без звука и без сети.

Запуск::

    python3 -m unittest discover -s tests -v
    python3 -m unittest tests.test_voice_loop -v

Тесты не запускают whisper.cpp, say, afplay и не ходят в сеть: внешние
процессы подменяются, проверяются разбор аргументов, маршрут через скиллы,
подготовка текста к озвучке и форматирование замеров.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import wave
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import voice_loop  # noqa: E402


class ParseArgsTests(unittest.TestCase):
    def test_text_mode(self):
        args = voice_loop.parse_args(["--text", "Сколько стоит тариф Старт"])
        self.assertEqual(args.text, "Сколько стоит тариф Старт")
        self.assertIsNone(args.wav)
        self.assertFalse(args.no_play)
        self.assertFalse(args.json)
        self.assertEqual(args.voice, voice_loop.DEFAULT_VOICE)
        self.assertEqual(args.lang, "ru")

    def test_wav_mode(self):
        args = voice_loop.parse_args(["--wav", "request.wav"])
        self.assertEqual(args.wav, Path("request.wav"))
        self.assertIsNone(args.text)

    def test_requires_input(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                voice_loop.parse_args([])
        self.assertEqual(caught.exception.code, 2)

    def test_text_and_wav_conflict(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                voice_loop.parse_args(["--wav", "request.wav", "--text", "привет"])
        self.assertEqual(caught.exception.code, 2)

    def test_no_play_and_out(self):
        args = voice_loop.parse_args(
            ["--text", "привет", "--no-play", "--out", "/tmp/reply.wav", "--json"]
        )
        self.assertTrue(args.no_play)
        self.assertEqual(args.out, Path("/tmp/reply.wav"))
        self.assertTrue(args.json)

    def test_custom_voice_and_lang(self):
        args = voice_loop.parse_args(
            ["--text", "привет", "--voice", "Milena", "--lang", "ru", "--rate", "180"]
        )
        self.assertEqual(args.voice, "Milena")
        self.assertEqual(args.lang, "ru")
        self.assertEqual(args.rate, 180)


class SpeechTextTests(unittest.TestCase):
    def test_citation_removed(self):
        self.assertEqual(
            voice_loop.to_speech("Тариф «Старт» — 990 ₽ [1]."),
            "Тариф «Старт» — 990 рублей.",
        )

    def test_multiple_citations_removed(self):
        self.assertEqual(
            voice_loop.to_speech("Ответ [1, 2] и ещё [3]."),
            "Ответ и ещё.",
        )

    def test_rubles_per_month(self):
        self.assertEqual(
            voice_loop.to_speech("Тариф «Старт» — 990 ₽/мес [1]."),
            "Тариф «Старт» — 990 рублей в месяц.",
        )

    def test_ruble_plural_forms(self):
        cases = {
            "1 ₽": "1 рубль",
            "2 ₽": "2 рубля",
            "4 ₽": "4 рубля",
            "5 ₽": "5 рублей",
            "11 ₽": "11 рублей",
            "21 ₽": "21 рубль",
            "22 ₽": "22 рубля",
            "25 ₽": "25 рублей",
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(voice_loop.to_speech(source), expected)

    def test_thousands_keep_spacing(self):
        self.assertEqual(
            voice_loop.to_speech("Профи — 2 490 ₽/мес."),
            "Профи — 2 490 рублей в месяц.",
        )

    def test_markdown_marks_removed(self):
        self.assertEqual(
            voice_loop.to_speech("**Важно:** `ответ` в [каталоге](https://example.org)."),
            "Важно: ответ в каталоге.",
        )

    def test_empty(self):
        self.assertEqual(voice_loop.to_speech(""), "")
        self.assertEqual(voice_loop.to_speech("   "), "")


class RouteTests(unittest.TestCase):
    def test_pricing_question_answered_by_skill(self):
        result = voice_loop.ask_hybrid("Сколько стоит тариф Старт")
        self.assertTrue(result.answered)
        self.assertEqual(result.route_kind, "skill")
        self.assertEqual(result.skill, "pricing")
        self.assertIn("990", result.answer_text)
        self.assertIn("Старт", result.spoken_text)
        self.assertNotIn("[1]", result.spoken_text)
        self.assertNotIn("₽", result.spoken_text)
        self.assertGreaterEqual(result.latency_ms, 0.0)

    def test_non_skill_does_not_call_models(self):
        # Если кто-то попытается уйти в модель, тест упадёт: сеть запрещена.
        with mock.patch(
            "eva.router._answer_with_models",
            side_effect=AssertionError("модель вызывать нельзя"),
        ):
            result = voice_loop.ask_hybrid(
                "Почему небо синее?",
                skills_fn=lambda *_args, **_kwargs: None,
            )
        self.assertFalse(result.answered)
        self.assertIn(result.route_kind, ("small", "big"))
        self.assertEqual(result.spoken_text, "")
        self.assertTrue(result.route_reason)

    def test_empty_question_rejected(self):
        with self.assertRaises(voice_loop.VoiceError):
            voice_loop.ask_hybrid("   ")


class WhisperSetupTests(unittest.TestCase):
    def test_help_without_binary(self):
        message = voice_loop.asr_help(None, None)
        self.assertIn("whisper.cpp не найден", message)
        self.assertIn("git clone", message)
        self.assertIn("cmake", message)
        self.assertIn("--text", message)

    def test_help_without_model(self):
        message = voice_loop.asr_help(Path("/opt/whisper-cli"), None)
        self.assertIn("модель не найдена", message)
        self.assertIn("download-ggml-model.sh", message)

    def test_explicit_missing_binary(self):
        with self.assertRaises(voice_loop.AsrUnavailable):
            voice_loop.find_whisper_binary("/nope/whisper-cli")

    def test_explicit_missing_model(self):
        with self.assertRaises(voice_loop.AsrUnavailable):
            voice_loop.find_whisper_model("/nope/ggml-small.bin")

    def test_wav_info(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.wav"
            with wave.open(str(path), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(16000)
                audio.writeframes(b"\x00\x00" * 1600)
            info = voice_loop.wav_info(path)
            self.assertEqual(info["channels"], 1)
            self.assertEqual(info["rate"], 16000)
            self.assertEqual(info["sample_width"], 2)
            self.assertAlmostEqual(info["duration_s"], 0.1, places=3)

    def test_wav_info_rejects_non_wav(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "not.wav"
            path.write_bytes(b"not a wave file")
            with self.assertRaises(voice_loop.AsrUnavailable):
                voice_loop.wav_info(path)

    def test_parse_whisper_stdout(self):
        raw = "[00:00:00.000 --> 00:00:01.000]  Привет\n[00:00:01.000 --> 00:00:02.000]  мир\n"
        self.assertEqual(voice_loop.parse_whisper_stdout(raw), "Привет мир")

    def test_transcribe_builds_whisper_command(self):
        seen = {}

        def fake_run(command, **kwargs):
            seen["command"] = command
            base = Path(command[command.index("-of") + 1])
            Path(str(base) + ".txt").write_text(
                "[00:00:00.000 --> 00:00:01.000]  Сколько стоит тариф Старт\n",
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 0, "", "")

        with mock.patch.object(voice_loop.subprocess, "run", side_effect=fake_run):
            text, elapsed = voice_loop.transcribe_wav(
                Path("request.wav"),
                Path("/fake/whisper-cli"),
                Path("/fake/ggml-small.bin"),
                lang="ru",
                timeout=5,
            )
        self.assertEqual(text, "Сколько стоит тариф Старт")
        self.assertGreaterEqual(elapsed, 0.0)
        command = seen["command"]
        self.assertIn("-otxt", command)
        self.assertIn("-nt", command)
        self.assertIn("-np", command)
        self.assertEqual(command[command.index("-l") + 1], "ru")
        self.assertEqual(command[command.index("-m") + 1], "/fake/ggml-small.bin")

    def test_transcribe_nonzero_exit_is_error(self):
        def fake_run(command, **kwargs):
            return subprocess.CompletedProcess(command, 1, "", "model load failed")

        with mock.patch.object(voice_loop.subprocess, "run", side_effect=fake_run):
            with self.assertRaises(voice_loop.AsrError) as caught:
                voice_loop.transcribe_wav(
                    Path("request.wav"), Path("/fake/whisper-cli"), Path("/fake/model.bin")
                )
        self.assertIn("кодом 1", str(caught.exception))

    def test_transcribe_empty_text_is_error(self):
        def fake_run(command, **kwargs):
            base = Path(command[command.index("-of") + 1])
            Path(str(base) + ".txt").write_text("", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "", "")

        with mock.patch.object(voice_loop.subprocess, "run", side_effect=fake_run):
            with self.assertRaises(voice_loop.AsrError):
                voice_loop.transcribe_wav(
                    Path("request.wav"), Path("/fake/whisper-cli"), Path("/fake/model.bin")
                )



class TtsCommandTests(unittest.TestCase):
    def test_say_command_writes_wav_16k(self):
        command = voice_loop.build_say_command(
            "/usr/bin/say", "Milena", "/tmp/reply.wav", "Привет"
        )
        self.assertEqual(command[0], "/usr/bin/say")
        self.assertIn("-v", command)
        self.assertEqual(command[command.index("-v") + 1], "Milena")
        self.assertIn("--file-format=WAVE", command)
        self.assertIn("--data-format=LEI16@16000", command)
        self.assertEqual(command[-1], "Привет")

    def test_say_command_rate(self):
        command = voice_loop.build_say_command(
            "say", "Milena", "reply.wav", "Привет", rate=180
        )
        self.assertEqual(command[command.index("-r") + 1], "180")


class TimingTests(unittest.TestCase):
    def test_first_sound_and_live_estimate(self):
        timings = voice_loop.Timings(
            asr_ms=120.0,
            route_ms=4.0,
            tts_synth_ms=500.0,
            tts_first_fragment_ms=430.0,
            play_start_ms=3.0,
            play_total_ms=1100.0,
        )
        self.assertAlmostEqual(timings.first_sound_ms, 627.0)
        self.assertAlmostEqual(timings.live_estimate_ms, 554.0)

    def test_no_play_has_no_first_sound(self):
        timings = voice_loop.Timings(
            route_ms=2.0, tts_synth_ms=100.0, tts_first_fragment_ms=80.0
        )
        self.assertIsNone(timings.first_sound_ms)
        self.assertAlmostEqual(timings.live_estimate_ms, 82.0)

    def test_format_contains_stages(self):
        timings = voice_loop.Timings(
            asr_ms=120.0,
            route_ms=4.0,
            tts_synth_ms=500.0,
            tts_first_fragment_ms=430.0,
            play_start_ms=3.0,
            play_total_ms=1100.0,
        )
        text = voice_loop.format_timings(timings)
        self.assertIn("ASR", text)
        self.assertIn("Маршрут", text)
        self.assertIn("TTS", text)
        self.assertIn("До первого звука", text)
        self.assertIn("627.0", text)

    def test_format_text_mode_marks_asr_skipped(self):
        text = voice_loop.format_timings(voice_loop.Timings(route_ms=1.0), no_play=True)
        self.assertIn("ASR пропущен", text)
        self.assertIn("--no-play", text)
        self.assertIn("воспроизведение отключено", text)


class RunWithoutSoundTests(unittest.TestCase):
    def _fake_route(self):
        return voice_loop.RouteResult(
            question="Сколько стоит тариф Старт",
            answer_text="Тариф «Старт» — 990 ₽/мес [1].",
            spoken_text="Тариф «Старт» — 990 рублей в месяц.",
            route_kind="skill",
            route_reason="skill: pricing",
            skill="pricing",
            latency_ms=1.5,
            answered=True,
        )

    def test_text_pipeline_json_no_play(self):
        args = voice_loop.parse_args(["--text", "Сколько стоит тариф Старт", "--json", "--no-play"])
        fake_synthesis = voice_loop.SynthesisResult(
            path=Path("/tmp/does-not-matter.wav"),
            synth_ms=12.0,
            first_fragment_ms=5.0,
        )
        out = io.StringIO()
        with mock.patch.object(voice_loop, "ask_hybrid", return_value=self._fake_route()), mock.patch.object(
            voice_loop, "synthesize", return_value=fake_synthesis
        ), redirect_stdout(out):
            code = voice_loop.run(args, cfg=object())
        self.assertEqual(code, 0)
        payload = json.loads(out.getvalue())
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["route"]["kind"], "skill")
        self.assertEqual(payload["route"]["skill"], "pricing")
        self.assertTrue(payload["audio"]["saved"] is False)
        self.assertFalse(payload["audio"]["played"])
        self.assertAlmostEqual(payload["timings"]["tts_synth_ms"], 12.0)
        self.assertIsNone(payload["timings"]["first_sound_ms"])

    def test_text_pipeline_human_no_play(self):
        args = voice_loop.parse_args(["--text", "Сколько стоит тариф Старт", "--no-play"])
        fake_synthesis = voice_loop.SynthesisResult(
            path=Path("/tmp/does-not-matter.wav"),
            synth_ms=12.0,
            first_fragment_ms=5.0,
        )
        out = io.StringIO()
        with mock.patch.object(voice_loop, "ask_hybrid", return_value=self._fake_route()), mock.patch.object(
            voice_loop, "synthesize", return_value=fake_synthesis
        ), redirect_stdout(out):
            code = voice_loop.run(args, cfg=object())
        self.assertEqual(code, 0)
        text = out.getvalue()
        self.assertIn("Маршрут: skill (pricing)", text)
        self.assertIn("Озвучка: Тариф «Старт» — 990 рублей в месяц.", text)
        self.assertIn("Тайминги, мс:", text)
        self.assertIn("--no-play", text)

    def test_non_skill_prints_honest_note(self):
        args = voice_loop.parse_args(["--text", "Почему небо синее?", "--no-play"])
        route = voice_loop.RouteResult(
            question="Почему небо синее?",
            answer_text="",
            spoken_text="",
            route_kind="big",
            route_reason="маркер рассуждения",
            skill=None,
            latency_ms=0.5,
            answered=False,
        )
        out = io.StringIO()
        with mock.patch.object(voice_loop, "ask_hybrid", return_value=route), redirect_stdout(out):
            code = voice_loop.run(args, cfg=object())
        self.assertEqual(code, 0)
        text = out.getvalue()
        self.assertIn("Модель не вызывалась", text)
        self.assertIn("ollama", text)
        self.assertIn("нет ответа скилла", text)

    def test_missing_whisper_is_message_not_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "request.wav"
            with wave.open(str(wav), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(16000)
                audio.writeframes(b"\x00\x00" * 1600)

            args = voice_loop.parse_args(["--wav", str(wav)])
            err = io.StringIO()
            with mock.patch.object(voice_loop, "find_whisper_binary", return_value=None), mock.patch.object(
                voice_loop, "find_whisper_model", return_value=None
            ), redirect_stderr(err):
                code = voice_loop.run(args, cfg=object())
        self.assertEqual(code, 2)
        message = err.getvalue()
        self.assertIn("ASR недоступен", message)
        self.assertIn("whisper.cpp не найден", message)
        self.assertIn("--text", message)
        self.assertNotIn("Traceback", message)

    def test_missing_wav_reports_error(self):
        args = voice_loop.parse_args(["--wav", "/nope/request.wav"])
        err = io.StringIO()
        with redirect_stderr(err):
            code = voice_loop.run(args, cfg=object())
        self.assertEqual(code, 2)
        self.assertIn("WAV не найден", err.getvalue())


class MainEntryTests(unittest.TestCase):
    def test_main_returns_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "request.wav"
            wav.write_bytes(b"")
            with mock.patch.object(voice_loop, "find_whisper_binary", return_value=None), mock.patch.object(
                voice_loop, "find_whisper_model", return_value=None
            ), redirect_stderr(io.StringIO()):
                code = voice_loop.main(["--wav", str(wav)])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
