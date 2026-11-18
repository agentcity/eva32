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
import threading
import time
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


class SentenceSplitTests(unittest.TestCase):
    """Разбиение ответа на предложения: пунктуация, «₽», кавычки, сокращения."""

    def test_two_sentences(self):
        chunks = voice_loop.split_sentences(
            "Тариф «Старт» — 990 ₽/мес. Лимиты: до 3 сотрудников; 100 услуг."
        )
        self.assertEqual(
            chunks,
            ["Тариф «Старт» — 990 ₽/мес.", "Лимиты: до 3 сотрудников; 100 услуг."],
        )

    def test_ruble_numbers_stay_with_their_sentence(self):
        chunks = voice_loop.split_sentences("Старт — 990 ₽. Профи — 2 490 ₽/мес.")
        self.assertEqual(chunks, ["Старт — 990 ₽.", "Профи — 2 490 ₽/мес."])
        self.assertTrue(all(len(chunk) > 3 for chunk in chunks))

    def test_abbreviations_do_not_split(self):
        chunks = voice_loop.split_sentences(
            "Это долго, т.е. очень долго. Дальше — др. пункт."
        )
        self.assertEqual(chunks, ["Это долго, т.е. очень долго.", "Дальше — др. пункт."])
        self.assertNotIn("т.е.", chunks[1])

    def test_decimal_point_does_not_split(self):
        chunks = voice_loop.split_sentences("Версия 1.5 вышла. Обновитесь.")
        self.assertEqual(chunks, ["Версия 1.5 вышла.", "Обновитесь."])

    def test_initials_do_not_split(self):
        chunks = voice_loop.split_sentences("А. С. Пушкин написал. Это факт.")
        self.assertEqual(chunks, ["А. С. Пушкин написал.", "Это факт."])

    def test_ellipsis_splits(self):
        chunks = voice_loop.split_sentences("Думал… Потом решил. И сделал!")
        self.assertEqual(chunks, ["Думал…", "Потом решил.", "И сделал!"])

    def test_three_dots_split(self):
        chunks = voice_loop.split_sentences("Ну... Ладно.")
        self.assertEqual(chunks, ["Ну...", "Ладно."])

    def test_quote_sentence_is_not_torn(self):
        # Точка внутри «ёлочек» не режет: цитата остаётся одним куском,
        # а следующее за цитатой предложение — отдельный чанк.
        chunks = voice_loop.split_sentences(
            "Он сказал: «Всё хорошо. Завтра продолжим.» И ушёл."
        )
        self.assertEqual(
            chunks,
            ["Он сказал: «Всё хорошо. Завтра продолжим.»", "И ушёл."],
        )

    def test_quote_then_new_sentence_splits(self):
        chunks = voice_loop.split_sentences("«Всё хорошо.» Он ушёл.")
        self.assertEqual(chunks, ["«Всё хорошо.»", "Он ушёл."])

    def test_ellipsis_inside_quote_is_not_torn(self):
        chunks = voice_loop.split_sentences("Он сказал: «Ну…» и ушёл.")
        self.assertEqual(chunks, ["Он сказал: «Ну…» и ушёл."])

    def test_no_punctuation_is_one_chunk(self):
        self.assertEqual(voice_loop.split_sentences("Привет"), ["Привет"])

    def test_empty(self):
        self.assertEqual(voice_loop.split_sentences(""), [])
        self.assertEqual(voice_loop.split_sentences("   \n\t "), [])

    def test_spoken_skill_answer_gives_three_chunks(self):
        spoken = (
            "Тариф «Старт» — 990 рублей в месяц. "
            "Лимиты: до 3 сотрудников; 100 услуг в каталоге. "
            "Заметки: дополнительный сотрудник — 290 рублей в месяц; "
            "при оплате за год скидка 20%."
        )
        chunks = voice_loop.split_sentences(spoken)
        self.assertEqual(len(chunks), 3)
        self.assertTrue(chunks[0].startswith("Тариф «Старт»"))
        self.assertTrue(chunks[2].endswith("20%."))
        self.assertEqual(" ".join(chunks).split(), spoken.split())

    def test_long_sentence_is_wrapped_without_losing_text(self):
        text = "Слово " * 100
        chunks = voice_loop.split_sentences(text, max_chars=120)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 120 for chunk in chunks))
        self.assertEqual(" ".join(chunks).split(), text.split())

    def test_wrap_keeps_number_and_ruble_together(self):
        # Пробел внутри «2 490 ₽» не должен стать разрывом чанка.
        text = "Слово " * 36 + "2 490 ₽ и хвост."
        chunks = voice_loop.split_sentences(text, max_chars=220)
        self.assertTrue(any("2 490 ₽" in chunk for chunk in chunks), chunks)


class StreamSpeakTests(unittest.TestCase):
    """Потоковая озвучка: порядок чанков, параллельный синтез, первый звук."""

    def _fake_synth(self, delay=0.0, return_ms=(40.0, 10.0)):
        calls = []

        def fake(text, out_path, **kwargs):
            calls.append(text)
            if delay:
                time.sleep(delay)
            Path(out_path).write_bytes(b"RIFF")
            return voice_loop.SynthesisResult(
                path=Path(out_path),
                synth_ms=return_ms[0],
                first_fragment_ms=return_ms[1],
            )

        return fake, calls

    def test_order_and_overlap(self):
        texts = ["Первое.", "Второе.", "Третье."]
        synth_started = {text: threading.Event() for text in texts[1:]}
        fake_synth, calls = self._fake_synth()
        played = []
        overlap = {}

        def fake_synthesize(text, out_path, **kwargs):
            if text in synth_started:
                synth_started[text].set()
            return fake_synth(text, out_path, **kwargs)

        def fake_play(path, **kwargs):
            name = Path(path).name
            played.append(name)
            if name.endswith("part00.wav"):
                # Первое предложение уже звучит, а третье должно синтезироваться.
                overlap["third_during_first_play"] = synth_started["Третье."].wait(2.0)
            return voice_loop.PlayResult(start_ms=2.0, total_ms=5.0)

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            voice_loop, "synthesize", side_effect=fake_synthesize
        ), mock.patch.object(voice_loop, "play_audio", side_effect=fake_play):
            stream = voice_loop.speak_stream(texts, Path(tmp), play=True)

        self.assertEqual(
            played, ["reply.part00.wav", "reply.part01.wav", "reply.part02.wav"]
        )
        self.assertEqual(calls, texts)
        self.assertTrue(overlap["third_during_first_play"])
        self.assertEqual(len(stream.chunks), 3)
        self.assertIsNotNone(stream.first_sound_ms)
        self.assertLess(stream.first_sound_ms, stream.synth_total_ms)
        self.assertIsNone(stream.chunks[0].wait_ms)
        self.assertAlmostEqual(stream.chunks[0].play_start_ms, stream.first_sound_ms)

    def test_no_play_does_not_touch_player(self):
        fake_synth, calls = self._fake_synth()
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            voice_loop, "synthesize", side_effect=fake_synth
        ), mock.patch.object(
            voice_loop, "play_audio", side_effect=AssertionError("звук запрещён")
        ):
            stream = voice_loop.speak_stream(["Один.", "Два."], Path(tmp), play=False)
        self.assertEqual(calls, ["Один.", "Два."])
        self.assertIsNone(stream.first_sound_ms)
        self.assertIsNone(stream.play_span_ms)
        self.assertIsNotNone(stream.synth_total_ms)

    def test_empty_sentences_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(voice_loop.TtsError):
                voice_loop.speak_stream(["  ", ""], Path(tmp))

    def test_missing_directory_rejected(self):
        with self.assertRaises(voice_loop.TtsError):
            voice_loop.speak_stream(["Привет."], Path("/nope/eva-voice"))

    def test_concat_wavs(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames = [100, 150, 200]
            parts = []
            for index, count in enumerate(frames):
                path = Path(tmp) / f"part{index}.wav"
                with wave.open(str(path), "wb") as audio:
                    audio.setnchannels(1)
                    audio.setsampwidth(2)
                    audio.setframerate(16000)
                    audio.writeframes(b"\x00\x00" * count)
                parts.append(path)
            target = Path(tmp) / "reply.wav"
            result = voice_loop.concat_wavs(parts, target)
            self.assertEqual(result, target)
            with wave.open(str(target)) as audio:
                self.assertEqual(audio.getnframes(), sum(frames))
                self.assertEqual(audio.getframerate(), 16000)

    def test_concat_rejects_different_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "a.wav"
            second = Path(tmp) / "b.wav"
            for path, rate in ((first, 16000), (second, 22050)):
                with wave.open(str(path), "wb") as audio:
                    audio.setnchannels(1)
                    audio.setsampwidth(2)
                    audio.setframerate(rate)
                    audio.writeframes(b"\x00\x00" * 10)
            with self.assertRaises(voice_loop.TtsError):
                voice_loop.concat_wavs([first, second], Path(tmp) / "out.wav")


class StreamRunTests(unittest.TestCase):
    """run(): потоковый режим по умолчанию и --no-stream для сравнения."""

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

    def _fake_synth(self, calls=None):
        def fake(text, out_path, **kwargs):
            if calls is not None:
                calls.append(text)
            path = Path(out_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(path), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(16000)
                audio.writeframes(b"\x00\x00" * len(text))
            return voice_loop.SynthesisResult(
                path=path, synth_ms=8.0, first_fragment_ms=3.0
            )

        return fake

    def test_stream_json_no_play(self):
        args = voice_loop.parse_args(
            ["--text", "Сколько стоит тариф Старт", "--json", "--no-play"]
        )
        out = io.StringIO()
        with mock.patch.object(
            voice_loop, "ask_hybrid", return_value=self._route()
        ), mock.patch.object(
            voice_loop, "synthesize", side_effect=self._fake_synth()
        ), mock.patch.object(
            voice_loop, "play_audio", side_effect=AssertionError("звук запрещён")
        ), redirect_stdout(out):
            code = voice_loop.run(args, cfg=object())

        self.assertEqual(code, 0)
        payload = json.loads(out.getvalue())
        timings = payload["timings"]
        self.assertTrue(timings["stream"])
        self.assertEqual(timings["chunks"], 3)
        self.assertEqual(len(timings["chunk_stats"]), 3)
        self.assertEqual(timings["chunk_stats"][0]["chars"], len("Первое предложение."))
        self.assertIsNone(timings["first_sound_ms"])
        # --no-play: звук начнётся, когда WAV первого предложения готов.
        self.assertAlmostEqual(timings["first_sound_estimate_ms"], 1.5 + 8.0)
        self.assertAlmostEqual(timings["live_estimate_ms"], 1.5 + 3.0)
        self.assertFalse(payload["audio"]["played"])
        self.assertEqual(len(payload["audio"]["parts"]), 3)

    def test_no_stream_synthesizes_whole_answer_once(self):
        calls = []
        args = voice_loop.parse_args(
            ["--text", "Сколько стоит тариф Старт", "--json", "--no-play", "--no-stream"]
        )
        out = io.StringIO()
        with mock.patch.object(
            voice_loop, "ask_hybrid", return_value=self._route()
        ), mock.patch.object(
            voice_loop, "synthesize", side_effect=self._fake_synth(calls)
        ), redirect_stdout(out):
            code = voice_loop.run(args, cfg=object())

        self.assertEqual(code, 0)
        self.assertEqual(calls, [self.SPOKEN])
        timings = json.loads(out.getvalue())["timings"]
        self.assertFalse(timings["stream"])
        self.assertEqual(timings["chunks"], 0)
        self.assertEqual(timings["chunk_stats"], [])
        self.assertAlmostEqual(timings["tts_synth_ms"], 8.0)
        self.assertIsNone(timings["first_sound_ms"])
        self.assertAlmostEqual(timings["first_sound_estimate_ms"], 1.5 + 8.0)

    def test_no_stream_flag_exists_in_human_mode(self):
        args = voice_loop.parse_args(["--text", "привет", "--no-play", "--no-stream"])
        self.assertTrue(args.no_stream)
        self.assertFalse(voice_loop.parse_args(["--text", "привет"]).no_stream)

    def test_stream_out_concatenates_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "reply.wav"
            args = voice_loop.parse_args(
                ["--text", "Сколько стоит тариф Старт", "--json", "--no-play", "--out", str(target)]
            )
            out = io.StringIO()
            with mock.patch.object(
                voice_loop, "ask_hybrid", return_value=self._route()
            ), mock.patch.object(
                voice_loop, "synthesize", side_effect=self._fake_synth()
            ), redirect_stdout(out):
                code = voice_loop.run(args, cfg=object())

            self.assertEqual(code, 0)
            payload = json.loads(out.getvalue())
            self.assertTrue(payload["audio"]["saved"])
            self.assertEqual(payload["audio"]["path"], str(target))
            chunks = voice_loop.split_sentences(self.SPOKEN)
            with wave.open(str(target)) as audio:
                self.assertEqual(audio.getnframes(), sum(len(chunk) for chunk in chunks))

    def test_stream_human_output_mentions_chunks(self):
        args = voice_loop.parse_args(
            ["--text", "Сколько стоит тариф Старт", "--no-play"]
        )
        out = io.StringIO()
        with mock.patch.object(
            voice_loop, "ask_hybrid", return_value=self._route()
        ), mock.patch.object(
            voice_loop, "synthesize", side_effect=self._fake_synth()
        ), redirect_stdout(out):
            code = voice_loop.run(args, cfg=object())
        self.assertEqual(code, 0)
        text = out.getvalue()
        self.assertIn("TTS, 1-й чанк", text)
        self.assertIn("Чанков", text)
        self.assertIn("(поток по предложениям)", text)
        self.assertIn("Оценка до звука", text)
        self.assertIn("Ориентир live say", text)

    def test_no_stream_human_output_has_no_chunks(self):
        args = voice_loop.parse_args(
            ["--text", "Сколько стоит тариф Старт", "--no-play", "--no-stream"]
        )
        out = io.StringIO()
        with mock.patch.object(
            voice_loop, "ask_hybrid", return_value=self._route()
        ), mock.patch.object(
            voice_loop, "synthesize", side_effect=self._fake_synth()
        ), redirect_stdout(out):
            code = voice_loop.run(args, cfg=object())
        self.assertEqual(code, 0)
        text = out.getvalue()
        self.assertIn("TTS, синтез WAV", text)
        self.assertNotIn("Чанков", text)


class TimingEstimateTests(unittest.TestCase):
    def test_first_sound_estimate_without_play(self):
        timings = voice_loop.Timings(
            asr_ms=120.0, route_ms=4.0, tts_synth_ms=500.0, tts_first_fragment_ms=430.0
        )
        self.assertIsNone(timings.first_sound_ms)
        self.assertAlmostEqual(timings.first_sound_estimate_ms, 624.0)
        self.assertAlmostEqual(timings.to_dict()["first_sound_estimate_ms"], 624.0)
        # Ориентир live say — по первому фрагменту, он всегда меньше.
        self.assertAlmostEqual(timings.live_estimate_ms, 554.0)

    def test_estimate_falls_back_to_fragment(self):
        timings = voice_loop.Timings(route_ms=5.0, tts_first_fragment_ms=300.0)
        self.assertAlmostEqual(timings.first_sound_estimate_ms, 305.0)


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
