"""Тесты tools/parity_cold.py без torch: парсер, текст, пробы, отчёт.

Запуск:
    python3 -m unittest tests.test_parity_cold -v
    python3 -m unittest discover -s tests

torch/transformers здесь не импортируются: проверяется ровно то, что можно
проверить на машине без GPU-стека. C-проба собирается и сверяется с
``tools/vtm_ref`` на синтетическом контейнере ``demo/model.vtm`` (если есть
``src/vtm/libvtm.a`` и компилятор) — это проверка того, что PPL и логиты из
C совпадают с Python-эталоном, не загружая HF.
"""
import contextlib
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools import parity_cold  # noqa: E402
from tools import vtm_ref  # noqa: E402

DEMO_VTM = os.path.join(ROOT, "demo", "model.vtm")
VTM_LIB = os.path.join(ROOT, "src", "vtm", "libvtm.a")
CC = parity_cold.pr.find_c_compiler()


class StubTokenizer:
    """Минимальный токенизатор: слово → id, без torch и сети."""

    eos_token_id = 99

    def __init__(self, vocab=100):
        self.vocab = vocab

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        ids = [1 + (abs(hash(word)) % (self.vocab - 1)) for word in text.split()]
        return {"input_ids": ids}


def _jsonl(rows):
    return "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"


class TestParser(unittest.TestCase):
    def test_defaults(self):
        args = parity_cold.parse_args([])

        self.assertEqual(args.dtype, "bfloat16")
        self.assertEqual(args.reference_dtype, "float32")
        self.assertEqual(args.threshold, 1e-2)
        self.assertEqual(args.embed_mode, "trit")
        self.assertEqual(args.scale_dtype, "float32")
        self.assertEqual(args.ppl_lines, 200)
        self.assertEqual(args.ppl_max_tokens, 128)
        self.assertEqual(args.limit_tokens, 48)
        self.assertIsNone(args.vtm)
        self.assertIsNone(args.manifest)
        self.assertFalse(args.export)
        self.assertFalse(args.skip_parity)
        self.assertFalse(args.skip_ppl)
        self.assertFalse(args.skip_original)

    def test_all_flags(self):
        args = parity_cold.parse_args([
            "--model", "/models/qwen",
            "--vtm", "/tmp/model.vtm",
            "--workdir", "/tmp/pc",
            "--json", "/tmp/pc/out.json",
            "--prompt", "один", "--prompt", "два",
            "--limit-tokens", "7",
            "--dtype", "float32",
            "--reference-dtype", "bfloat16",
            "--embed-mode", "none",
            "--threshold", "1e-3",
            "--ppl-data", "/tmp/pairs.jsonl",
            "--ppl-lines", "250",
            "--ppl-max-tokens", "64",
            "--skip-original",
        ])

        self.assertEqual(args.model, "/models/qwen")
        self.assertEqual(args.vtm, "/tmp/model.vtm")
        self.assertEqual(args.workdir, "/tmp/pc")
        self.assertEqual(args.json, "/tmp/pc/out.json")
        self.assertEqual(args.prompt, ["один", "два"])
        self.assertEqual(args.limit_tokens, 7)
        self.assertEqual(args.dtype, "float32")
        self.assertEqual(args.reference_dtype, "bfloat16")
        self.assertEqual(args.embed_mode, "none")
        self.assertEqual(args.threshold, 1e-3)
        self.assertEqual(args.ppl_lines, 250)
        self.assertEqual(args.ppl_max_tokens, 64)
        self.assertTrue(args.skip_original)

    def test_default_prompts_are_three_to_five(self):
        args = parity_cold.parse_args([])
        prompts = parity_cold.collect_prompts(args)

        self.assertGreaterEqual(len(prompts), 3)
        self.assertLessEqual(len(prompts), 5)
        self.assertTrue(all(prompt.strip() for prompt in prompts))


class TestText(unittest.TestCase):
    def test_read_pairs_and_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.jsonl"
            path.write_text(
                _jsonl([
                    {"question": "q1", "answer": "a1"},
                    {"question": "", "answer": "a2"},
                    {"question": "q3", "answer": "  "},
                    {"question": "q4", "answer": "a4"},
                ]),
                encoding="utf-8",
            )
            pairs = parity_cold.read_text_pairs(path, 10)

        self.assertEqual(pairs, [("q1", "a1"), ("q4", "a4")])

    def test_read_pairs_limit_and_offset(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.jsonl"
            path.write_text(
                _jsonl([
                    {"question": f"q{index}", "answer": f"a{index}"}
                    for index in range(5)
                ]),
                encoding="utf-8",
            )
            limited = parity_cold.read_text_pairs(path, 2)
            shifted = parity_cold.read_text_pairs(path, 2, offset=3)

        self.assertEqual(limited, [("q0", "a0"), ("q1", "a1")])
        self.assertEqual(shifted, [("q3", "a3"), ("q4", "a4")])

    def test_missing_file_and_bad_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "nope.jsonl"
            with self.assertRaises(parity_cold.ColdParityError):
                parity_cold.read_text_pairs(missing, 1)
            broken = Path(tmp) / "broken.jsonl"
            broken.write_text("{not json}\n", encoding="utf-8")
            with self.assertRaises(parity_cold.ColdParityError):
                parity_cold.read_text_pairs(broken, 1)

    def test_build_example_masks_and_eos(self):
        tokenizer = StubTokenizer()
        example = parity_cold.build_example(
            tokenizer, "привет мир", "ответ", max_tokens=64, vocab=100
        )

        self.assertIsNotNone(example)
        assert example is not None
        self.assertEqual(example["prompt_len"], 4)
        self.assertEqual(example["answer_tokens"], 2)
        self.assertEqual(example["ids"][-1], tokenizer.eos_token_id)
        self.assertFalse(example["truncated"])

    def test_build_example_truncates_and_keeps_eos(self):
        tokenizer = StubTokenizer()
        answer = " ".join(["слово"] * 50)
        example = parity_cold.build_example(
            tokenizer, "короткий вопрос", answer, max_tokens=10, vocab=100
        )

        self.assertIsNotNone(example)
        assert example is not None
        self.assertTrue(example["truncated"])
        self.assertLessEqual(len(example["ids"]), 10)
        self.assertEqual(example["ids"][-1], tokenizer.eos_token_id)

    def test_build_example_rejects_out_of_vocab(self):
        tokenizer = StubTokenizer(vocab=100)
        example = parity_cold.build_example(
            tokenizer, "вопрос", "ответ", max_tokens=64, vocab=5
        )

        self.assertIsNone(example)

    def test_build_examples_counts_dropped(self):
        tokenizer = StubTokenizer(vocab=100)
        long_question = " ".join(["вопрос"] * 20)
        examples, dropped = parity_cold.build_examples(
            tokenizer,
            [("вопрос", "ответ"), (long_question, "ответ")],
            max_tokens=12,
            vocab=100,
        )

        self.assertEqual(len(examples), 1)
        self.assertEqual(dropped, 1)

    def test_build_examples_truncates_long_answer(self):
        tokenizer = StubTokenizer(vocab=100)
        long_answer = " ".join(["слово"] * 100)
        examples, dropped = parity_cold.build_examples(
            tokenizer, [("вопрос", long_answer)], max_tokens=12, vocab=100
        )

        self.assertEqual(len(examples), 1)
        self.assertEqual(dropped, 0)
        self.assertTrue(examples[0]["truncated"])

    def test_sequence_file_text(self):
        text = parity_cold.sequence_file_text([
            {"ids": [10, 11, 12], "prompt_len": 2},
            {"ids": [7, 8], "prompt_len": 1},
        ])

        self.assertEqual(text, "2 10,11,12\n1 7,8\n")

    def test_token_file_text(self):
        self.assertEqual(parity_cold.token_file_text([[1, 2], [3]]), "1,2\n3\n")


class TestMath(unittest.TestCase):
    def test_perplexity(self):
        self.assertAlmostEqual(parity_cold.perplexity(math.log(4.0), 1), 4.0)
        self.assertAlmostEqual(
            parity_cold.perplexity(2 * math.log(4.0), 2), 4.0
        )
        self.assertIsNone(parity_cold.perplexity(1.0, 0))
        self.assertIsNone(parity_cold.perplexity(float("nan"), 5))

    def test_token_nll_uniform(self):
        logits = [0.0, 0.0, 0.0, 0.0]
        self.assertAlmostEqual(parity_cold.token_nll(logits, 0), math.log(4.0))

    def test_token_nll_greedy(self):
        logits = [10.0, 0.0, 0.0]
        self.assertLess(parity_cold.token_nll(logits, 0), 1e-3)
        self.assertGreater(parity_cold.token_nll(logits, 1), 9.0)

    def test_token_nll_rejects_bad_target(self):
        with self.assertRaises(parity_cold.ColdParityError):
            parity_cold.token_nll([1.0, 2.0], 5)

    def test_gap_ratio(self):
        reference = [1.0, 0.5, 0.2]
        ratio = parity_cold.gap_ratio(reference, reference, 0.005)

        self.assertIsNotNone(ratio)
        assert ratio is not None
        self.assertAlmostEqual(ratio, 0.5 / (2 * 0.005))
        self.assertIsNone(parity_cold.gap_ratio([1.0], [1.0], 0.1))
        self.assertIsNone(parity_cold.gap_ratio(reference, reference, 0.0))


class TestProbeOutput(unittest.TestCase):
    LOGITS = (
        "# vocab 4\n"
        '# meta_json {"model": {"context_len": 512}}\n'
        "# seq 0 argmax 2\n"
        "# logits 0 4\n"
        "0.1 0.2 0.9 -0.3\n"
        "# seq 1 argmax 0\n"
        "# logits 1 4\n"
        "0.5 0.4 0.3 0.2\n"
    )

    def test_parse_logits(self):
        parsed = parity_cold.parse_logits_probe_output(self.LOGITS)

        self.assertEqual(parsed["meta"]["vocab"], 4.0)
        self.assertEqual(parsed["argmax"][0], 2)
        self.assertEqual(parsed["argmax"][1], 0)
        self.assertEqual(parsed["logits"][1], [0.5, 0.4, 0.3, 0.2])

    def test_context_len_from_meta_json(self):
        parsed = parity_cold.parse_logits_probe_output(self.LOGITS)

        self.assertEqual(parity_cold._meta_context_len(parsed["meta"]), 512)

    def test_logits_without_header_raises(self):
        with self.assertRaises(parity_cold.ColdParityError):
            parity_cold.parse_logits_probe_output("1 2 3\n")

    def test_logits_count_mismatch_raises(self):
        broken = "# logits 0 4\n0.1 0.2 0.3\n"
        with self.assertRaises(parity_cold.ColdParityError):
            parity_cold.parse_logits_probe_output(broken)

    def test_parse_ppl(self):
        text = (
            "# vocab 4\n"
            "# seq 0 tokens 3 nll 2.5\n"
            "# seq 1 tokens 2 nll 1.5\n"
            "# total_tokens 5\n"
            "# total_nll 4\n"
        )
        parsed = parity_cold.parse_ppl_probe_output(text)

        self.assertEqual(parsed["tokens"], 5)
        self.assertEqual(parsed["nll_sum"], 4.0)
        self.assertEqual(parsed["sequences"][0], {"tokens": 3, "nll": 2.5})
        self.assertEqual(parsed["sequences"][1], {"tokens": 2, "nll": 1.5})

    def test_ppl_without_totals_raises(self):
        with self.assertRaises(parity_cold.ColdParityError):
            parity_cold.parse_ppl_probe_output("# seq 0 tokens 1 nll 1\n")

    def test_ppl_unexpected_line_raises(self):
        with self.assertRaises(parity_cold.ColdParityError):
            parity_cold.parse_ppl_probe_output("42\n")


class TestComparison(unittest.TestCase):
    def test_compare_parity_passes_on_same_logits(self):
        logits = [0.1, 0.5, 1.0, 0.2]
        result = parity_cold.compare_parity(logits, logits, 1e-2)

        self.assertEqual(result["max_abs_delta"], 0.0)
        self.assertTrue(result["argmax_match"])
        self.assertIsNone(result["gap_ratio"])

    def test_compare_parity_argmax_flip_fails(self):
        reference = [0.1, 1.0, 0.0]
        runtime = [0.9, 0.0, 0.0]
        result = parity_cold.compare_parity(reference, runtime, 1e-2)

        self.assertFalse(result["argmax_match"])
        self.assertFalse(result["within_threshold"])
        self.assertIsNotNone(result["gap_ratio"])

    def test_summarize_parity_verdict(self):
        prompts = [
            {
                "index": 0,
                "comparison": {
                    "max_abs_delta": 1e-6,
                    "argmax_match": True,
                    "top5_overlap": 5,
                    "top5_size": 5,
                    "within_threshold": True,
                    "gap_ratio": 100.0,
                },
            },
            {
                "index": 1,
                "comparison": {
                    "max_abs_delta": 3e-6,
                    "argmax_match": True,
                    "top5_overlap": 4,
                    "top5_size": 5,
                    "within_threshold": True,
                    "gap_ratio": 50.0,
                },
            },
        ]
        summary = parity_cold.summarize_parity(prompts, 1e-2)

        self.assertTrue(summary["passed"])
        self.assertEqual(summary["argmax_matches"], 2)
        self.assertEqual(summary["top5_matches"], 1)
        self.assertEqual(summary["worst_prompt"], 1)
        self.assertIn("ПРОЙДЕН", summary["verdict"])
        self.assertGreater(summary["threshold_margin"], 1.0)
        self.assertEqual(summary["min_gap_ratio"], 50.0)

    def test_summarize_parity_empty(self):
        summary = parity_cold.summarize_parity([], 1e-2)

        self.assertFalse(summary["passed"])
        self.assertEqual(summary["prompts"], 0)

    def test_summarize_ppl(self):
        result = {
            "lines": 200,
            "tokens": 1000,
            "hf_original": {"perplexity": 10.0},
            "hf_ternary": {"perplexity": 15.0},
            "vtm_c": {"perplexity": 15.0},
        }
        summary = parity_cold.summarize_ppl(result)

        self.assertAlmostEqual(summary["damage_ratio"], 1.5)
        self.assertAlmostEqual(summary["damage_percent"], 50.0)
        self.assertAlmostEqual(summary["c_vs_hf_relative"], 0.0)

    def test_summarize_ppl_without_original(self):
        result = {"lines": 10, "tokens": 100, "hf_ternary": {"perplexity": 20.0},
                  "vtm_c": {"perplexity": 20.0}}
        summary = parity_cold.summarize_ppl(result)

        self.assertIsNone(summary["damage_ratio"])
        self.assertAlmostEqual(summary["c_vs_hf_relative"], 0.0)


class TestFormatReport(unittest.TestCase):
    def _result(self):
        return {
            "model": {"name": "demo", "dtype": "bfloat16"},
            "reference_dtype": "float32",
            "embed_mode": "trit",
            "threshold": 1e-2,
            "vtm": {"path": "/tmp/model.vtm", "size_bytes": 2048,
                    "sha256": "a" * 64, "mode": "given"},
            "prompts": [
                {
                    "index": 0,
                    "text": "Столица Франции — это",
                    "tokens": 5,
                    "comparison": {
                        "max_abs_delta": 3.8e-06,
                        "argmax_match": True,
                        "argmax_reference": 32,
                        "argmax_runtime": 32,
                        "top5_overlap": 5,
                        "top5_size": 5,
                        "within_threshold": True,
                        "gap_ratio": 250.0,
                    },
                }
            ],
            "parity": {
                "verdict": "ПРОЙДЕН: argmax 1/1, max|Δ| 3.8e-06 < 0.01",
                "top5_matches": 1,
                "prompts": 1,
                "min_gap_ratio": 250.0,
                "threshold_margin": 2600.0,
                "max_abs_delta": 3.8e-06,
                "threshold": 1e-2,
            },
            "quantization_effect": {
                "max_abs_delta": 14.5,
                "argmax_matches": 2,
                "prompts": 5,
            },
            "perplexity": {
                "data": "/tmp/pairs.jsonl",
                "lines": 200,
                "tokens": 38100,
                "hf_original": {"perplexity": 9.54},
                "hf_ternary": {"perplexity": 16.13},
                "vtm_c": {"perplexity": 16.13},
                "summary": {
                    "damage_ratio": 1.69,
                    "damage_percent": 69.0,
                    "c_vs_hf_relative": 1.8e-06,
                },
            },
            "warnings": ["предупреждение"],
        }

    def test_report_contains_tables(self):
        text = parity_cold.format_report(self._result())

        self.assertIn("паритет C ↔ HF", text)
        self.assertIn("Столица Франции", text)
        self.assertIn("итог: ПРОЙДЕН", text)
        self.assertIn("perplexity", text)
        self.assertIn("bf16-оригинал", text)
        self.assertIn("триты, C (.vtm)", text)
        self.assertIn("урон от третизации без QAT: +69%", text)
        self.assertIn("предупреждение", text)

    def test_report_with_skipped_parity(self):
        result = self._result()
        result["prompts"] = []
        result["parity"] = {
            "prompts": 0,
            "passed": True,
            "skipped": True,
            "verdict": "паритет не проверялся (--skip-parity)",
        }
        del result["quantization_effect"]
        text = parity_cold.format_report(result)

        self.assertIn("паритет не проверялся", text)
        self.assertIn("perplexity", text)

    def test_report_without_original(self):
        result = self._result()
        result["prompts"][0]["comparison"]["argmax_match"] = True
        del result["perplexity"]["hf_original"]
        result["perplexity"]["summary"] = {
            "damage_ratio": None,
            "damage_percent": None,
            "c_vs_hf_relative": 0.0,
        }
        text = parity_cold.format_report(result)

        self.assertNotIn("  bf16-оригинал", text)
        self.assertIn("триты, HF", text)


class TestResolveVtm(unittest.TestCase):
    def test_no_source_raises(self):
        args = SimpleNamespace(vtm=None, manifest=None, export=False)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(parity_cold.ColdParityError):
                parity_cold.resolve_vtm(args, Path(tmp))

    def test_missing_vtm_raises(self):
        args = SimpleNamespace(vtm="/nonexistent/model.vtm", manifest=None, export=False)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(parity_cold.ColdParityError):
                parity_cold.resolve_vtm(args, Path(tmp))


class TestDryRun(unittest.TestCase):
    def test_dry_run_without_torch(self):
        args = parity_cold.parse_args([
            "--model", "/nonexistent/model",
            "--vtm", "/nonexistent/model.vtm",
            "--dry-run",
        ])
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = parity_cold.main([
                "--model", "/nonexistent/model",
                "--vtm", "/nonexistent/model.vtm",
                "--dry-run",
            ])
        text = buffer.getvalue()

        del args
        self.assertEqual(code, 0)
        self.assertIn("dry-run", text)
        self.assertIn("perplexity", text)
        self.assertIn("C-компилятор", text)


@unittest.skipUnless(
    CC and os.path.isfile(VTM_LIB) and os.path.isfile(DEMO_VTM),
    "нужны cc, src/vtm/libvtm.a и demo/model.vtm",
)
class TestCProbe(unittest.TestCase):
    """C-проба против tools/vtm_ref: PPL и логиты обязаны совпасть."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.workdir = Path(self.tmp.name)
        self.cc = CC
        self.vtm = DEMO_VTM
        self.sequences = [[5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]
        self.examples = [
            {"ids": [5, 6, 7, 8, 9, 10], "prompt_len": 3},
            {"ids": [11, 12, 13, 14, 15], "prompt_len": 2},
        ]

    def tearDown(self):
        self.tmp.cleanup()

    def _ref_nll(self, ids, prompt_len):
        """NLL токенов ответа по vtm_ref.execute (медленно, но эталонно)."""
        total = 0.0
        for index in range(prompt_len - 1, len(ids) - 1):
            result = vtm_ref.execute(self.vtm, ids[:index + 1])
            logits = result.logits
            assert logits is not None
            total += parity_cold.token_nll(logits, ids[index + 1])
        return total

    def test_logits_match_reference(self):
        parsed = parity_cold.c_last_logits(
            self.workdir, Path(self.vtm), self.sequences, self.cc
        )

        self.assertEqual(parsed["meta"]["vocab"], 128.0)
        for index, ids in enumerate(self.sequences):
            expected = vtm_ref.execute(self.vtm, ids).logits
            assert expected is not None
            got = parsed["logits"][index]
            delta = max(abs(a - b) for a, b in zip(expected, got))
            # Синтетический контейнер: C и vtm_ref копят по-разному, на
            # случайных весах расхождение доходит до ~1e-4 (см. docs/quantization.md).
            self.assertLess(delta, 5e-4, f"seq {index}: max|Δ|={delta:.3g}")

    def test_ppl_matches_reference(self):
        parsed = parity_cold.c_answer_ppl(
            self.workdir, Path(self.vtm), self.examples, self.cc
        )

        reference = sum(
            self._ref_nll(example["ids"], example["prompt_len"])
            for example in self.examples
        )
        self.assertEqual(parsed["tokens"], 3 + 3)
        self.assertAlmostEqual(parsed["nll_sum"], reference, places=4)
        expected_ppl = math.exp(reference / 6)
        self.assertLess(
            abs(parsed["perplexity"] - expected_ppl) / expected_ppl, 1e-5
        )

    def test_probe_reuses_binary(self):
        first = parity_cold.ensure_probe_built(self.workdir, self.cc)
        second = parity_cold.ensure_probe_built(self.workdir, self.cc)

        self.assertEqual(first, second)
        self.assertTrue(first.is_file())

    def test_probe_compile_flags_include_gnu_source(self):
        """getline/ssize_t без -D_GNU_SOURCE на gcc не компилируются."""
        source = parity_cold.probe_source()
        self.assertIn("getline", source)
        self.assertEqual(source.count("int mode_"), 2)


class TestDemoReport(unittest.TestCase):
    def test_format_demo_report(self):
        meta = {
            "dirs": {"plain": "/tmp/demo/plain", "qat": "/tmp/demo/qat"},
            "architecture": {"hidden": 256, "layers": 4, "heads": 4,
                             "kv_heads": 2, "vocab": 287},
            "plain": {"steps": 200, "last_loss": 2.5},
            "qat": {"steps": 150, "last_loss": 2.9},
            "quant": {"embed_mode": "trit"},
        }
        text = parity_cold.format_demo_report(meta)

        self.assertIn("демо-модель", text)
        self.assertIn("plain 200 шагов", text)
        self.assertIn("--skip-original", text)


if __name__ == "__main__":
    unittest.main()
