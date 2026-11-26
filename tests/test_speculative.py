"""Тесты спекулятивного декода ВТМ (задача 0203).

Что проверяется без больших моделей:

* C-помощник ``tools/vtm_spec_probe.c`` собирается и грузится (нужны ``cc`` и
  ``src/vtm/libvtm.a``; без них — skip);
* на крошечной модели самоспекуляция (draft == target) даёт **тот же** выход,
  что обычный жадный декод, а на скалярном бэкенде — побитово: батчевые ядра
  скаляра повторяют арифметику поштучного пути;
* учёт статистики (принятые/предложенные, средний префикс) сходится;
* ``--speculative`` в ``tools/chat.py`` подключается и отвергает конфигурации
  без ``--draft``/с тем же файлом/с не-vtm бэкендом.

Паритет на 3B (3 промпта) — отдельный тест под ``EVA_SPEC_MODELS=1``:
он поднимает 2-ГиБ контексты и идёт минуты, поэтому по умолчанию скипается.

Запуск::

    python3 -m unittest tests.test_speculative -v
    EVA_SPEC_MODELS=1 python3 -m unittest tests.test_speculative -v
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eva import speculative as spec  # noqa: E402
from tools import chat as chat_cli  # noqa: E402

TINY_MODEL = ROOT / "runs" / "tiny" / "model.vtm"
TINY_TOKENIZER = ROOT / "tests" / "vtm" / "fixtures" / "tokenizer.json"
DRAFT_3B = ROOT / "runs" / "q4" / "0.5b-q4.vtm"
TARGET_3B = ROOT / "runs" / "q4" / "3b-q4k.vtm"
TOKENIZER_3B = ROOT / "runs" / "q4" / "tokenizer.json"

_HAS_TINY = TINY_MODEL.is_file() and spec.STATIC_LIBRARY.is_file()


def _probe_available() -> bool:
    try:
        spec.build_probe()
        return True
    except spec.SpecUnavailable:
        return False


_HAS_PROBE = _probe_available() if _HAS_TINY else False


@unittest.skipUnless(_HAS_PROBE, "нужны cc и src/vtm/libvtm.a")
class TestProbe(unittest.TestCase):
    def test_probe_builds_and_loads(self) -> None:
        path = spec.build_probe()
        self.assertTrue(path.is_file(), path)
        library = spec._Library(path)  # noqa: SLF001 — проверяем загрузку
        self.assertTrue(hasattr(library.cdll, "vtm_spec_open"))
        self.assertTrue(hasattr(library.cdll, "vtm_spec_batch"))

    def test_backend_and_sampling_metadata(self) -> None:
        model = spec.SpecModel(TINY_MODEL, backend="scalar")
        try:
            self.assertIn("scalar", model.backend)
            sampling = model.sampling()
            self.assertIn("temperature", sampling)
            self.assertIn("repetition_window", sampling)
        finally:
            model.close()

    def test_wrong_backend_fails_loudly(self) -> None:
        with self.assertRaises(spec.SpecError):
            spec.SpecModel(TINY_MODEL, backend="no-such-backend")


@unittest.skipUnless(_HAS_PROBE, "нужны cc и src/vtm/libvtm.a")
class TestSelfSpeculation(unittest.TestCase):
    """draft == target: приём 100%, выход обязан совпасть с обычным."""

    IDS = [1, 2, 3, 4, 5]
    N = 8

    def _decoder(self, k: int) -> spec.SpeculativeDecoder:
        return spec.SpeculativeDecoder(
            TINY_MODEL, TINY_MODEL, k=k, backend="scalar", eos_token_id=-1
        )

    def test_parity_for_several_k(self) -> None:
        for k in (2, 3, 4, 6):
            with self.subTest(k=k):
                decoder = self._decoder(k)
                try:
                    baseline = decoder.target_only(self.IDS, max_new_tokens=self.N)
                    tokens = decoder.generate(self.IDS, max_new_tokens=self.N)
                    self.assertEqual(tokens, baseline)
                    self.assertEqual(len(tokens), self.N)
                    self.assertGreaterEqual(decoder.stats.acceptance_rate, 0.99)
                finally:
                    decoder.close()

    def test_stats_accounting(self) -> None:
        decoder = self._decoder(4)
        try:
            tokens = decoder.generate(self.IDS, max_new_tokens=12)
            stats = decoder.stats
            self.assertEqual(stats.generated_tokens, len(tokens))
            self.assertGreater(stats.rounds, 0)
            self.assertEqual(stats.batch_calls, stats.rounds)
            self.assertLessEqual(stats.drafts_accepted, stats.drafts_proposed)
            self.assertLessEqual(stats.mean_accepted_prefix, 3.0)
            self.assertGreater(stats.acceptance_rate, 0.9)
        finally:
            decoder.close()

    def test_target_only_matches_probe_feed_chain(self) -> None:
        decoder = self._decoder(2)
        try:
            baseline = decoder.target_only(self.IDS, max_new_tokens=6)
        finally:
            decoder.close()
        model = spec.SpecModel(TINY_MODEL, backend="scalar")
        try:
            model.force_greedy()
            token = model.prefill(self.IDS)
            chain = [token]
            for _ in range(5):
                token = model.feed(token)
                chain.append(token)
            self.assertEqual(baseline, chain)
        finally:
            model.close()

    def test_k_must_be_at_least_two(self) -> None:
        with self.assertRaises(ValueError):
            spec.SpeculativeDecoder(TINY_MODEL, TINY_MODEL, k=1)


@unittest.skipUnless(_HAS_TINY, "нет runs/tiny/model.vtm")
class TestChatIntegration(unittest.TestCase):
    def _kwargs(self, **extra: object) -> dict[str, object]:
        base: dict[str, object] = dict(
            backend="vtm",
            model=str(TINY_MODEL),
            tokenizer=str(TINY_TOKENIZER),
        )
        base.update(extra)
        return base

    def test_requires_draft(self) -> None:
        with self.assertRaises(chat_cli.ChatConfigError):
            chat_cli.build_backend(**self._kwargs(speculative=True))

    def test_rejects_non_vtm_backend(self) -> None:
        with self.assertRaises(chat_cli.ChatConfigError):
            chat_cli.build_backend(
                backend="ollama", model="qwen2.5:3b", speculative=True,
                draft=str(TINY_MODEL),
            )

    def test_rejects_same_model_as_draft(self) -> None:
        with self.assertRaises(chat_cli.ChatConfigError):
            chat_cli.build_backend(
                **self._kwargs(speculative=True, draft=str(TINY_MODEL))
            )

    def test_rejects_missing_draft_file(self) -> None:
        with self.assertRaises(chat_cli.ChatConfigError):
            chat_cli.build_backend(
                **self._kwargs(speculative=True, draft=str(ROOT / "nope.vtm"))
            )

    def test_parser_accepts_flags(self) -> None:
        parser = chat_cli.build_parser()
        args = parser.parse_args(
            ["--speculative", "--draft", "x.vtm", "--spec-k", "6"]
        )
        self.assertTrue(args.speculative)
        self.assertEqual(args.draft, "x.vtm")
        self.assertEqual(args.spec_k, 6)


@unittest.skipUnless(
    os.environ.get("EVA_SPEC_MODELS") == "1"
    and DRAFT_3B.is_file()
    and TARGET_3B.is_file()
    and TOKENIZER_3B.is_file()
    and _HAS_PROBE,
    "EVA_SPEC_MODELS=1 и runs/q4/{0.5b-q4,3b-q4k}.vtm нужны для паритета на 3B",
)
class TestParity3B(unittest.TestCase):
    """Greedy-паритет спекуляции и обычного C-ABI декода на 3 промптах."""

    PROMPTS = [
        ("Ты — краткий помощник. Отвечай по-русски.", "Столица Франции — это"),
        ("You are a helpful assistant.", "List three prime numbers and explain why 17 is prime."),
        ("Ты — программист.", "Напиши на Python функцию, которая складывает два числа."),
    ]

    def test_bitwise_parity_three_prompts(self) -> None:
        from eva.backends import VtmBackend
        from eva.backends.tokenizer import BPETokenizer

        tokenizer = BPETokenizer.from_file(str(TOKENIZER_3B))
        prompts = []
        for system, user in self.PROMPTS:
            text = (
                "<|im_start|>system\n" + system + "<|im_end|>\n"
                "<|im_start|>user\n" + user + "<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            prompts.append((user, tokenizer.encode(text)))
        # Эталон (C-ABI) и спекуляция открываются по очереди, а не
        # одновременно: 3B-контекст — 2 ГБ, и на 16 ГиБ лучше не держать два.
        reference = VtmBackend(
            model=str(TARGET_3B), tokenizer=str(TOKENIZER_3B),
            backend="simd", cache_prefix=False, max_tokens=12,
        )
        try:
            expected = [
                reference.generate(prompt, max_new_tokens=12)
                for _, prompt in prompts
            ]
        finally:
            reference.close()
        decoder = spec.SpeculativeDecoder(
            DRAFT_3B, TARGET_3B, k=4, backend="simd",
            eos_token_id=int(getattr(tokenizer, "eos_id", -1)),
        )
        try:
            for (user, prompt), want in zip(prompts, expected):
                tokens = decoder.generate(prompt, max_new_tokens=12)
                self.assertEqual(tokens, want, f"промпт: {user!r}")
                self.assertLessEqual(len(tokens), 12)
        finally:
            decoder.close()


if __name__ == "__main__":
    unittest.main()
