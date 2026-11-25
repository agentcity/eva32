"""Локальный бэкенд ВТМ через C-ABI (задача 0089).

Что проверяем на ``demo/model.vtm`` (без GPU и без демона):

* генерация через ``ctypes`` даёт токены, повторный запрос детерминирован;
* стриминг: по одному токену/чанку, склейка == полный ответ;
* отмена из другого потока прекращает генерацию, контекст после неё живой;
* ошибки: нет файла модели, пустой/отрицательный промпт, занятый контекст;
* токенизатор ``tokenizer.json`` (BPE) — round-trip и инкрементальный decode.

Если библиотеки ВТМ нет (не собрана ``libvtm.a`` и нет готовой
``.dylib``/``.so``), тесты бэкенда делают ``skip`` с инструкцией, а не падают:
на чистом клоне без сборки это штатная ситуация.
"""

from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eva.backends import (  # noqa: E402
    BPETokenizer,
    VTMConfig,
    VtmBackend,
    VtmBusy,
    VtmCancelled,
    VtmError,
    VtmUnavailable,
    available,
    build_shared_library,
    ensure_library,
    format_chat_prompt,
)
from eva.backends.tokenizer import TokenizerError  # noqa: E402

DEMO_MODEL = ROOT / "demo" / "model.vtm"
TOKENIZER = ROOT / "tests" / "vtm" / "fixtures" / "tokenizer.json"
PROMPT = [1, 2, 3]


def _skip_without_vtm() -> None:
    """skip, если библиотеку ВТМ не найти и не собрать."""
    try:
        ensure_library()
    except VtmUnavailable as exc:
        raise unittest.SkipTest(f"библиотека ВТМ недоступна: {exc}") from None


def _skip_without_model() -> None:
    if not DEMO_MODEL.is_file():
        raise unittest.SkipTest(
            f"нет {DEMO_MODEL}: соберите демо-модель "
            "`python3 scripts/demo_ternary_model.py --out demo`"
        )


def _demo_backend(**kwargs) -> VtmBackend:
    _skip_without_vtm()
    _skip_without_model()
    params = dict(
        model=str(DEMO_MODEL),
        tokenizer=str(TOKENIZER) if TOKENIZER.is_file() else None,
        max_tokens=4,
    )
    params.update(kwargs)
    return VtmBackend(**params)


class TestConfig(unittest.TestCase):
    def test_defaults(self) -> None:
        cfg = VTMConfig.from_env({})
        self.assertEqual(cfg.backend, "auto")
        self.assertEqual(cfg.max_tokens, 64)
        self.assertEqual(cfg.prefill_batch, 128)
        self.assertFalse(cfg.cache_prefix)
        self.assertIsNone(cfg.threads)
        self.assertIsNone(cfg.library)

    def test_from_env(self) -> None:
        cfg = VTMConfig.from_env(
            {
                "EVA_VTM_MODEL": "/models/m.vtm",
                "EVA_VTM_LIB": "/lib/libvtm.dylib",
                "EVA_VTM_TOKENIZER": "/models/tokenizer.json",
                "EVA_VTM_BACKEND": "scalar",
                "EVA_VTM_MAX_TOKENS": "7",
                "EVA_VTM_THREADS": "2",
                "EVA_VTM_PREFILL_BATCH": "32",
            }
        )
        self.assertEqual(cfg.model, "/models/m.vtm")
        self.assertEqual(cfg.library, "/lib/libvtm.dylib")
        self.assertEqual(cfg.tokenizer, "/models/tokenizer.json")
        self.assertEqual(cfg.backend, "scalar")
        self.assertEqual(cfg.max_tokens, 7)
        self.assertEqual(cfg.threads, 2)
        self.assertEqual(cfg.prefill_batch, 32)
        # 0 из окружения — явный выключатель, а не «не задано».
        self.assertEqual(
            VTMConfig.from_env({"EVA_VTM_PREFILL_BATCH": "0"}).prefill_batch, 0
        )

    def test_arguments_override_environment(self) -> None:
        cfg = VTMConfig.from_env(
            {"EVA_VTM_MODEL": "/from/env.vtm", "EVA_VTM_MAX_TOKENS": "9"},
            model="/from/arg.vtm",
            max_tokens=3,
        )
        self.assertEqual(cfg.model, "/from/arg.vtm")
        self.assertEqual(cfg.max_tokens, 3)

    def test_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            VTMConfig.from_env({"EVA_VTM_MAX_TOKENS": "-1"})
        with self.assertRaises(ValueError):
            VTMConfig.from_env({"EVA_VTM_MAX_TOKENS": "много"})
        with self.assertRaises(ValueError):
            VTMConfig.from_env({"EVA_VTM_BACKEND": "gpu"})
        with self.assertRaises(ValueError):
            VTMConfig.from_env({"EVA_VTM_THREADS": "0"})
        with self.assertRaises(ValueError):
            VTMConfig.from_env({"EVA_VTM_PREFILL_BATCH": "-1"})
        with self.assertRaises(ValueError):
            VTMConfig.from_env({"EVA_VTM_PREFILL_BATCH": "много"})

    def test_tokenizer_next_to_model(self) -> None:
        import shutil
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.vtm"
            model.write_bytes(b"not-a-real-model")
            if TOKENIZER.is_file():
                shutil.copyfile(TOKENIZER, Path(directory) / "tokenizer.json")
            cfg = VTMConfig.from_env({}, model=str(model))
            self.assertEqual(
                cfg.tokenizer,
                str(Path(directory) / "tokenizer.json") if TOKENIZER.is_file() else None,
            )

    def test_backend_reads_environment(self) -> None:
        backend = VtmBackend(
            environ={
                "EVA_VTM_MODEL": "/models/env.vtm",
                "EVA_VTM_MAX_TOKENS": "5",
                "EVA_VTM_BACKEND": "lut",
            },
            tokenizer=None,
        )
        self.assertEqual(backend.model_path, "/models/env.vtm")
        self.assertEqual(backend.config.max_tokens, 5)
        self.assertEqual(backend.config.backend, "lut")

    def test_format_chat_prompt(self) -> None:
        text = format_chat_prompt(
            [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        )
        self.assertEqual(text, "system: s\nuser: u\nassistant:")
        self.assertEqual(format_chat_prompt("raw"), "raw")


class TestLibrary(unittest.TestCase):
    def test_rejects_static_archive(self) -> None:
        static = ROOT / "src" / "vtm" / "libvtm.a"
        if not static.is_file():
            self.skipTest("нет src/vtm/libvtm.a")
        with self.assertRaises(VtmUnavailable) as ctx:
            ensure_library(str(static))
        self.assertIn("статический архив", str(ctx.exception))

    def test_missing_explicit_library(self) -> None:
        with self.assertRaises(VtmUnavailable) as ctx:
            ensure_library("/нет/такой/libvtm.dylib")
        self.assertIn("файла нет", str(ctx.exception))

    def test_library_is_found_or_built(self) -> None:
        _skip_without_vtm()
        path = ensure_library()
        self.assertTrue(path.is_file())
        self.assertIn(path.suffix, (".dylib", ".so", ".dll"))
        self.assertTrue(available())
        # Повторный вызов стабилен (кэш автосборки).
        self.assertEqual(str(ensure_library()), str(path))

    def test_build_requires_source(self) -> None:
        with self.assertRaises(VtmUnavailable):
            build_shared_library("/нет/libvtm.a")


class TestTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not TOKENIZER.is_file():
            raise unittest.SkipTest(f"нет фикстуры {TOKENIZER}")
        cls.tokenizer = BPETokenizer.from_file(str(TOKENIZER))

    def test_roundtrip(self) -> None:
        for text in (
            "hello world",
            "Привет, мир!",
            "Hello, world! 123",
            "  multiple   spaces  ",
            "emoji 😀 test",
            "a\tb\nc",
        ):
            with self.subTest(text=text):
                ids = self.tokenizer.encode(text)
                self.assertTrue(ids)
                self.assertEqual(self.tokenizer.decode(ids), text)

    def test_empty_text(self) -> None:
        self.assertEqual(self.tokenizer.encode(""), [])
        self.assertEqual(self.tokenizer.decode([]), "")

    def test_reference_vectors_match_cpp(self) -> None:
        """Векторы сняты с C++ ``Tokenizer`` (src/vtm/tokenizer.cpp) на этой фикстуре."""
        vectors = {
            "hello world": [259, 264],
            "Привет, мир!": [275, 44, 281, 33],
            "don't stop": [100, 111, 110, 39, 116, 32, 115, 116, 111, 112],
            "emoji 😀 test": [
                101, 109, 111, 106, 105, 32, 282, 152, 128, 32, 116, 101, 115, 116,
            ],
        }
        for text, expected in vectors.items():
            with self.subTest(text=text):
                self.assertEqual(self.tokenizer.encode(text), expected)

    def test_added_special_tokens(self) -> None:
        ids = self.tokenizer.encode("<|im_start|>hi<|im_end|>")
        self.assertEqual(ids[0], 500)
        self.assertEqual(ids[-1], 501)
        self.assertEqual(
            self.tokenizer.decode(ids), "<|im_start|>hi<|im_end|>"
        )
        # allow_special_tokens=False — спецтокены не выдёргиваются.
        plain = self.tokenizer.encode(
            "<|im_end|>", allow_special_tokens=False
        )
        self.assertNotIn(501, plain)

    def test_stream_decoder_matches_decode(self) -> None:
        ids = self.tokenizer.encode("Привет, мир! Hello world 123")
        decoder = self.tokenizer.stream_decoder()
        chunks = [decoder.push(token) for token in ids]
        chunks.append(decoder.finish())
        self.assertEqual("".join(chunks), self.tokenizer.decode(ids))

    def test_stream_decoder_partial_utf8(self) -> None:
        # 0xD0 0xBF — байты «п» (U+043F). По одному токену-байту: первый даёт
        # пустую строку, символ появляется только на втором.
        decoder = self.tokenizer.stream_decoder()
        first = decoder.push(0xD0)
        second = decoder.push(0xBF)
        self.assertEqual(first, "")
        self.assertEqual(first + second + decoder.finish(), "п")

    def test_unknown_id(self) -> None:
        with self.assertRaises(TokenizerError):
            self.tokenizer.decode([10_000])
        with self.assertRaises(TokenizerError):
            self.tokenizer.token(10_000)


class TestVtmGeneration(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = _demo_backend()

    def tearDown(self) -> None:
        self.backend.close()

    def test_generate_returns_tokens(self) -> None:
        ids = self.backend.generate(PROMPT, max_new_tokens=3)
        self.assertEqual(len(ids), 3)
        self.assertTrue(all(isinstance(token, int) and token >= 0 for token in ids))

    def test_generate_is_deterministic(self) -> None:
        first = self.backend.generate(PROMPT, max_new_tokens=4)
        second = self.backend.generate(PROMPT, max_new_tokens=4)
        self.assertEqual(first, second)

    def test_prefill_batch_with_cache_same_tokens(self) -> None:
        """Кэш + пачка (0180): префикс из кэша, хвост пачками — те же токены.

        Пары ``cache_prefix`` × ``prefill_batch`` (0 — выключатель, 2 и 128 —
        пачка) на demo-модели: ответ совпадает с поштучным путём. На SIMD
        батчевые ядра дают ≤1e-5, поэтому сравниваются токены, а не байты.
        """
        runs: list[tuple[bool, int, tuple[int, ...]]] = []
        for cache in (False, True):
            for batch in (0, 2, 128):
                backend = _demo_backend(cache_prefix=cache, prefill_batch=batch)
                try:
                    ids = backend.generate(PROMPT, max_new_tokens=3)
                finally:
                    backend.close()
                runs.append((cache, batch, ids))
        reference = runs[0][2]
        for cache, batch, ids in runs:
            with self.subTest(cache_prefix=cache, prefill_batch=batch):
                self.assertEqual(ids, reference)

    def test_per_call_limit(self) -> None:
        result = self.backend.generate_result(PROMPT, max_new_tokens=2)
        self.assertEqual(result.count, 2)
        self.assertEqual(result.finish_reason, "length")
        # Лимит вызова не портит контекст: следующий вызов даёт больше.
        self.assertEqual(len(self.backend.generate(PROMPT, max_new_tokens=3)), 3)

    def test_stream_ids_one_per_token(self) -> None:
        full = self.backend.generate(PROMPT, max_new_tokens=4)
        streamed = list(self.backend.stream_ids(PROMPT, max_new_tokens=4))
        self.assertEqual(len(streamed), len(full))
        self.assertEqual(streamed, full)

    def test_stream_text_concat_equals_full(self) -> None:
        tokenizer = self.backend.tokenizer
        prompt_ids = tokenizer.encode("abc def")
        tokens = self.backend.generate(prompt_ids, max_new_tokens=4)
        chunks = list(self.backend.stream_text(prompt_ids, max_new_tokens=4))
        self.assertEqual("".join(chunks), tokenizer.decode(tokens))
        self.assertEqual("".join(chunks), self.backend.generate_text(
            prompt_ids, max_new_tokens=4
        ))
        # Демо-токены — валидные ASCII-байты, поэтому чанк на токен и хвоста нет.
        self.assertEqual(len(chunks), len(tokens))

    def test_chat_mapping(self) -> None:
        answer = self.backend.chat([{"role": "user", "content": "abc"}], max_tokens=3)
        self.assertIsInstance(answer["text"], str)
        usage = answer["usage"]
        self.assertGreater(usage["prompt_tokens"], 0)
        self.assertGreater(usage["completion_tokens"], 0)
        self.assertEqual(
            usage["total_tokens"],
            usage["prompt_tokens"] + usage["completion_tokens"],
        )
        self.assertIn(answer["finish_reason"], ("length", "stop", "eos"))
        self.assertEqual(answer["model"], str(DEMO_MODEL))

    def test_stream_chat(self) -> None:
        text = "".join(
            self.backend.stream_chat([{"role": "user", "content": "abc"}], max_tokens=3)
        )
        self.assertIsInstance(text, str)

    def test_empty_prompt(self) -> None:
        with self.assertRaises(VtmError):
            self.backend.generate([], max_new_tokens=2)

    def test_negative_token(self) -> None:
        with self.assertRaises(VtmError) as ctx:
            self.backend.generate([1, -2], max_new_tokens=2)
        self.assertIn("отрицателен", str(ctx.exception))

    def test_missing_model_file(self) -> None:
        backend = VtmBackend(model=str(DEMO_MODEL) + ".нет", max_tokens=2)
        with self.assertRaises(VtmError) as ctx:
            backend.open()
        self.assertIn("не найден", str(ctx.exception))

    def test_corrupt_model_reports_c_abi_error(self) -> None:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".vtm", delete=False) as fh:
            fh.write(b"not a vtm container")
            bad_model = fh.name
        try:
            backend = VtmBackend(model=bad_model, max_tokens=2)
            with self.assertRaises(VtmError) as ctx:
                backend.open()
            # Текст пришёл из vtm_api_last_error(NULL) после неудачного open().
            self.assertIn("vtm_api_open", str(ctx.exception))
            self.assertIn(".vtm", str(ctx.exception))
        finally:
            Path(bad_model).unlink()

    def test_tokenizer_not_configured(self) -> None:
        backend = VtmBackend(model=str(DEMO_MODEL), tokenizer=None, max_tokens=2)
        with self.assertRaises(VtmError) as ctx:
            backend.generate_text(PROMPT)
        self.assertIn("токенизатор не задан", str(ctx.exception))
        backend.close()

    def test_close_is_idempotent_and_blocks_generation(self) -> None:
        self.backend.close()
        self.backend.close()
        self.assertFalse(self.backend.opened)
        with self.assertRaises(VtmError):
            self.backend.generate(PROMPT, max_new_tokens=1)


class TestVtmCancellation(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = _demo_backend(max_tokens=5)

    def tearDown(self) -> None:
        self.backend.close()

    def test_cancel_before_generate_stops_next_run(self) -> None:
        self.backend.open()
        self.backend.cancel()
        with self.assertRaises(VtmCancelled):
            self.backend.generate(PROMPT, max_new_tokens=3)
        # Флаг отмены не «протекает»: следующий запуск нормальный.
        self.assertEqual(len(self.backend.generate(PROMPT, max_new_tokens=3)), 3)

    def test_cancel_from_other_thread(self) -> None:
        started = threading.Event()
        release = threading.Event()
        outcome: list[BaseException] = []

        def on_token(_token: int) -> None:
            started.set()
            release.wait(10.0)

        def run() -> None:
            try:
                self.backend.generate(PROMPT, max_new_tokens=5, on_token=on_token)
            except BaseException as exc:  # noqa: BLE001 — проверяем тип в тесте
                outcome.append(exc)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        self.assertTrue(started.wait(10.0), "генерация не дошла до колбэка")
        self.backend.cancel()  # другой поток, пока generate() внутри C
        release.set()
        thread.join(10.0)
        self.assertFalse(thread.is_alive(), "генерация не остановилась после отмены")
        self.assertEqual(len(outcome), 1, f"ожидалась одна ошибка, получено {outcome}")
        self.assertIsInstance(outcome[0], VtmCancelled)
        self.assertEqual(len(self.backend.generate(PROMPT, max_new_tokens=3)), 3)

    def test_cancel_stream_from_other_thread(self) -> None:
        started = threading.Event()
        release = threading.Event()
        backend = self.backend

        def on_token(_token: int) -> None:
            started.set()
            release.wait(10.0)

        stream = backend.stream_ids(PROMPT, max_new_tokens=5, on_token=on_token)
        first = next(stream)
        self.assertIsInstance(first, int)
        self.assertTrue(started.wait(10.0))
        backend.cancel()
        release.set()
        with self.assertRaises(VtmCancelled):
            for _ in stream:
                pass
        # После отмены и закрытия генератора контекст снова доступен.
        self.assertEqual(len(backend.generate(PROMPT, max_new_tokens=3)), 3)

    def test_busy_context(self) -> None:
        started = threading.Event()
        release = threading.Event()

        def on_token(_token: int) -> None:
            started.set()
            release.wait(10.0)

        stream = self.backend.stream_ids(PROMPT, max_new_tokens=4, on_token=on_token)
        next(stream)
        self.assertTrue(started.wait(10.0))
        with self.assertRaises(VtmBusy):
            self.backend.generate(PROMPT, max_new_tokens=2)
        release.set()
        stream.close()  # останавливает генерацию и отменяет её
        self.assertEqual(len(self.backend.generate(PROMPT, max_new_tokens=3)), 3)


if __name__ == "__main__":
    unittest.main()
