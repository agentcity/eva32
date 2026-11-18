"""Тесты tools/parity_random.py без запуска C и без сети.

Запуск:
    python3 -m unittest tests.test_parity_random -v
    python3 -m unittest discover -s tests

Проверяется ровно то, что не требует `src/vtm/vtm` и `libvtm.a`:
детерминизм и допустимость генератора конфигураций, тритовые тензоры с
блоком 128/160, сборка контейнера и Python-эталон (без C), сравнение
логитов, сводка/отчёт, JSON и парсер CLI.
"""
import contextlib
import dataclasses
import io
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools import parity_random  # noqa: E402


def tiny_config(index: int = 0, *, block: int = 128, tied: bool = False) -> parity_random.RandomConfig:
    """Маленькая валидная конфигурация: hidden=64 кратно 2*16, 1 слой."""
    return parity_random.RandomConfig(
        index=index,
        seed=0,
        hidden=64,
        n_heads=2,
        n_kv_heads=1,
        head_dim=16,
        intermediate=128,
        vocab=32,
        n_layers=1,
        rope_theta=10000.0,
        rope_layout="half",
        tie_word_embeddings=tied,
        scale_block=block,
        context_len=8,
        eps=1e-3,
        tokens=(1, 5, 7),
    )


# ---------------------------------------------------------------------------
# Генератор конфигураций
# ---------------------------------------------------------------------------


class TestConfigGeneration(unittest.TestCase):
    def test_deterministic(self):
        for seed, index in ((0, 0), (0, 7), (11, 0), (123, 45)):
            with self.subTest(seed=seed, index=index):
                first = parity_random.generate_config(seed, index)
                second = parity_random.generate_config(seed, index)
                self.assertEqual(first, second)
                self.assertEqual(first.to_dict(), second.to_dict())

    def test_generate_configs_matches_single(self):
        configs = parity_random.generate_configs(seed=3, count=6)
        self.assertEqual(len(configs), 6)
        for index, config in enumerate(configs):
            self.assertEqual(config, parity_random.generate_config(3, index))

    def test_seed_and_index_change_config(self):
        self.assertNotEqual(
            parity_random.generate_config(0, 0),
            parity_random.generate_config(1, 0),
        )
        self.assertNotEqual(
            parity_random.generate_config(0, 5),
            parity_random.generate_config(0, 6),
        )

    def test_index_and_seed_are_recorded(self):
        config = parity_random.generate_config(3, 9)
        self.assertEqual(config.index, 9)
        self.assertEqual(config.seed, 3)
        self.assertEqual(config.config_id, "3:9")
        self.assertEqual(config.to_dict()["config_id"], "3:9")

    def test_every_config_is_valid(self):
        for seed in (0, 1, 42):
            for config in parity_random.generate_configs(seed, 40):
                problems = parity_random.validate_config(config)
                self.assertEqual(problems, [], f"seed={seed} cfg={config}")

    def test_ranges_from_task(self):
        for config in parity_random.generate_configs(0, 80):
            with self.subTest(index=config.index):
                self.assertGreaterEqual(config.hidden, 64)
                self.assertLessEqual(config.hidden, 256)
                self.assertEqual(
                    config.hidden % (config.n_heads * config.head_dim), 0
                )
                self.assertTrue(1 <= config.n_heads <= 8)
                self.assertTrue(1 <= config.n_kv_heads <= config.n_heads)
                self.assertEqual(config.n_heads % config.n_kv_heads, 0)
                self.assertTrue(8 <= config.head_dim <= 64)
                self.assertEqual(config.head_dim % 2, 0)
                self.assertTrue(1 <= config.n_layers <= 3)
                self.assertTrue(32 <= config.vocab <= 512)
                self.assertTrue(1e4 <= config.rope_theta <= 1e6)
                self.assertIsInstance(config.tie_word_embeddings, bool)
                self.assertIn(config.scale_block, (128, 160))
                self.assertGreater(config.intermediate, 0)
                self.assertGreaterEqual(config.context_len, len(config.tokens))
                for token in config.tokens:
                    self.assertTrue(0 <= token < config.vocab)

    def test_parameter_coverage(self):
        """Случайный набор обязан покрыть обе ветки tied и оба блока."""
        configs = parity_random.generate_configs(0, 80)
        self.assertEqual({config.scale_block for config in configs}, {128, 160})
        self.assertEqual(
            {config.tie_word_embeddings for config in configs}, {True, False}
        )
        self.assertTrue(any(config.n_kv_heads < config.n_heads for config in configs))
        self.assertTrue(any(config.n_kv_heads == config.n_heads for config in configs))
        self.assertTrue(any(config.rope_layout == "interleaved" for config in configs))
        self.assertTrue(any(config.rope_layout == "half" for config in configs))
        self.assertTrue(any(config.n_layers == 3 for config in configs))
        self.assertTrue(any(config.rope_theta > 1e5 for config in configs))

    def test_negative_index_or_count_rejected(self):
        with self.assertRaises(ValueError):
            parity_random.generate_config(0, -1)
        with self.assertRaises(ValueError):
            parity_random.generate_configs(0, -1)

    def test_validate_reports_violations(self):
        valid = tiny_config()
        self.assertEqual(parity_random.validate_config(valid), [])
        cases = {
            "hidden_range": dataclasses.replace(valid, hidden=512),
            "hidden_not_multiple": dataclasses.replace(valid, hidden=72),
            "heads_range": dataclasses.replace(
                valid, n_heads=9, n_kv_heads=1, hidden=9 * 16
            ),
            "kv_not_divisor": dataclasses.replace(valid, n_heads=3, n_kv_heads=2),
            "head_dim_odd": dataclasses.replace(valid, head_dim=15),
            "layers_range": dataclasses.replace(valid, n_layers=4),
            "vocab_range": dataclasses.replace(valid, vocab=16),
            "theta_range": dataclasses.replace(valid, rope_theta=1e7),
            "rope_layout": dataclasses.replace(valid, rope_layout="neox"),
            "scale_block": dataclasses.replace(valid, scale_block=96),
            "token_out_of_vocab": dataclasses.replace(valid, tokens=(1, 2, 99)),
            "context_too_short": dataclasses.replace(valid, context_len=1),
        }
        for name, config in cases.items():
            with self.subTest(name=name):
                self.assertNotEqual(
                    parity_random.validate_config(config), [], name
                )

    def test_to_model_config(self):
        for block in (128, 160):
            config = tiny_config(block=block)
            model_config = config.to_model_config()
            self.assertEqual(model_config.hidden, config.hidden)
            self.assertEqual(model_config.n_heads, config.n_heads)
            self.assertEqual(model_config.n_kv_heads, config.n_kv_heads)
            self.assertEqual(model_config.head_dim, config.head_dim)
            self.assertEqual(model_config.vocab, config.vocab)
            self.assertEqual(model_config.n_layers, config.n_layers)
            self.assertEqual(model_config.context_len, config.context_len)
            self.assertEqual(model_config.block_size, block)
            self.assertEqual(model_config.tie_word_embeddings, config.tie_word_embeddings)


# ---------------------------------------------------------------------------
# Тритовые тензоры и контейнер
# ---------------------------------------------------------------------------


class TestTritTensor(unittest.TestCase):
    def test_block_128_matches_canon(self):
        values = [((i * 37) % 19 - 9) / 9.0 for i in range(4 * 40)]
        tensor = parity_random.trit_tensor("w", (4, 40), values, 128)
        trits, scales = parity_random.vtm_ref.quantize_trits(values)
        expected = parity_random.vtm_ref.Tensor.ternary("w", (4, 40), trits, scales)
        self.assertEqual(tensor.kind, expected.kind)
        self.assertEqual(tensor.block_size, 128)
        self.assertEqual(tensor.trits, expected.trits)
        self.assertEqual(tensor.scales, expected.scales)

    def test_block_160_scales_and_dense(self):
        n_values = 3 * 50  # 150 тритов: один блок 160
        values = [((i * 11) % 23 - 11) / 11.0 for i in range(n_values)]
        tensor = parity_random.trit_tensor("w", (3, 50), values, 160)
        self.assertEqual(tensor.block_size, 160)
        self.assertEqual(tensor.kind, 0)
        self.assertEqual(tensor.n_elements, n_values)
        self.assertEqual(len(tensor.trits), n_values)
        self.assertEqual(
            len(tensor.scales), parity_random.vtm_ref.n_scale_blocks(n_values, 160)
        )
        self.assertEqual(len(tensor.scales), 1)
        dense = tensor.dense()
        self.assertEqual(len(dense), n_values)
        for index, (trit, scale) in enumerate(
            (trit, tensor.scales[0]) for trit in tensor.trits
        ):
            self.assertAlmostEqual(dense[index], trit * scale, places=6)

    def test_block_160_multiple_blocks(self):
        # 200 тритов: ceil(200/128) == ceil(200/160) == 2 — та самая
        # неоднозначность, из-за которой нужен флаг заголовка 0x8000.
        n_values = 4 * 50
        values = [((i * 7) % 13 - 6) / 6.0 for i in range(n_values)]
        tensor = parity_random.trit_tensor("w", (4, 50), values, 160)
        self.assertEqual(len(tensor.scales), 2)
        self.assertEqual(len(tensor.scales), 2)  # и 128-канон дал бы 2
        self.assertEqual(tensor.scales[0] >= 0.0, True)
        for index, trit in enumerate(tensor.trits):
            self.assertIn(trit, (-1, 0, 1))
            expected = trit * tensor.scales[index // 160]
            self.assertAlmostEqual(tensor.dense()[index], expected, places=6)

    def test_block_160_dense_matches_manual_dequant(self):
        values = [((i * 5) % 17 - 8) / 8.0 for i in range(160)]
        tensor = parity_random.trit_tensor("w", (8, 20), values, 160)
        for index, trit in enumerate(tensor.trits):
            self.assertAlmostEqual(
                tensor.dense()[index], trit * tensor.scales[0], places=6
            )

    def test_zero_block_has_zero_scale(self):
        tensor = parity_random.trit_tensor("w", (2, 8), [0.0] * 16, 160)
        self.assertEqual(tensor.scales, [0.0])
        self.assertEqual(set(tensor.trits), {0})

    def test_bad_block_rejected(self):
        with self.assertRaises(ValueError):
            parity_random.trit_tensor("w", (2, 8), [0.1] * 16, 96)
        with self.assertRaises(ValueError):
            parity_random.trit_tensor("w", (2, 8), [0.1] * 15, 128)

    def test_wrong_dims_rejected(self):
        with self.assertRaises(ValueError):
            parity_random.trit_tensor("w", (2, 8), [0.1] * 15, 128)


class TestBuildAndContainer(unittest.TestCase):
    def test_build_model_and_python_reference(self):
        for tied in (False, True):
            with self.subTest(tied=tied):
                config = tiny_config(tied=tied)
                vtm = parity_random.build_model(config)
                self.assertEqual(vtm.meta["vocab_size"], config.vocab)
                self.assertEqual(vtm.meta["scale_block"], 128)
                self.assertEqual(vtm.meta["parity_random"]["config_id"], config.config_id)
                result = parity_random.vtm_ref.execute(vtm, list(config.tokens), 0)
                self.assertIsNotNone(result.logits)
                self.assertEqual(len(result.logits), config.vocab)
                self.assertTrue(all(math.isfinite(value) for value in result.logits))
                last = vtm.program[-1]
                self.assertEqual(int(last.op), int(parity_random.vtm_ref.Op.HALT))

    def test_write_128_and_roundtrip(self):
        config = tiny_config(block=128)
        vtm = parity_random.build_model(config)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.vtm"
            parity_random.write_container(vtm, path, config)
            self.assertTrue(path.is_file())
            self.assertFalse(
                parity_random.read_header_flags(path)
                & parity_random.HEADER_FLAG_SCALE_BLOCK_160
            )
            # Python-эталон читает контейнер 128 с диска и даёт те же логиты.
            loaded = parity_random.vtm_ref.read_vtm(path)
            in_memory = parity_random.vtm_ref.execute(vtm, list(config.tokens), 0)
            from_disk = parity_random.vtm_ref.execute(
                loaded, list(config.tokens), 0
            )
            self.assertEqual(in_memory.logits, from_disk.logits)

    def test_write_160_sets_header_flag(self):
        config = tiny_config(block=160)
        vtm = parity_random.build_model(config)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.vtm"
            parity_random.write_container(vtm, path, config)
            flags = parity_random.read_header_flags(path)
            self.assertTrue(flags & parity_random.HEADER_FLAG_SCALE_BLOCK_160)
            self.assertTrue(flags & 0x0002)  # VFLAG_CAUSAL не потерян
            self.assertTrue(flags & 0x0001)  # масштабы fp32
            # Python-эталон на блок 160 существует только в памяти:
            # с диска он такой контейнер не читает (канон — 128).
            with self.assertRaises(ValueError):
                parity_random.vtm_ref.read_vtm(path)

    def test_write_160_python_logits_from_memory(self):
        config = tiny_config(block=160)
        vtm = parity_random.build_model(config)
        result = parity_random.vtm_ref.execute(vtm, list(config.tokens), 0)
        self.assertEqual(len(result.logits), config.vocab)

    def test_mark_rejects_foreign_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.vtm"
            path.write_bytes(b"NOPE" + b"\x00" * 32)
            with self.assertRaises(ValueError):
                parity_random.mark_scale_block_160(path)
            with self.assertRaises(ValueError):
                parity_random.read_header_flags(path)

    def test_mark_is_idempotent(self):
        config = tiny_config(block=160)
        vtm = parity_random.build_model(config)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.vtm"
            parity_random.write_container(vtm, path, config)
            first = parity_random.read_header_flags(path)
            parity_random.mark_scale_block_160(path)
            self.assertEqual(parity_random.read_header_flags(path), first)


# ---------------------------------------------------------------------------
# Сравнение логитов
# ---------------------------------------------------------------------------


class TestComparison(unittest.TestCase):
    def test_identical(self):
        comparison = parity_random.compare_logits([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        self.assertEqual(comparison.max_abs_delta, 0.0)
        self.assertTrue(comparison.within_tolerance)
        self.assertTrue(comparison.argmax_match)
        self.assertTrue(comparison.passed)

    def test_within_tolerance(self):
        comparison = parity_random.compare_logits(
            [0.0, 1.0], [1e-7, 1.0 + 1e-7], tolerance=1e-5
        )
        self.assertLess(comparison.max_abs_delta, 1e-5)
        self.assertTrue(comparison.passed)

    def test_outside_tolerance(self):
        comparison = parity_random.compare_logits(
            [0.0, 1.0], [1e-3, 1.0], tolerance=1e-5
        )
        self.assertFalse(comparison.within_tolerance)
        self.assertFalse(comparison.passed)
        self.assertAlmostEqual(comparison.max_abs_delta, 1e-3)

    def test_tolerance_is_strict(self):
        comparison = parity_random.compare_logits(
            [0.0, 0.0], [1e-5, 0.0], tolerance=1e-5
        )
        self.assertFalse(comparison.within_tolerance)

    def test_argmax_mismatch(self):
        comparison = parity_random.compare_logits(
            [1.0, 0.0], [0.0, 1.0], tolerance=1e-5
        )
        self.assertFalse(comparison.argmax_match)
        self.assertEqual(comparison.argmax_reference, 0)
        self.assertEqual(comparison.argmax_runtime, 1)
        self.assertFalse(comparison.passed)

    def test_argmax_tie_uses_first_index(self):
        comparison = parity_random.compare_logits([2.0, 2.0, 1.0], [2.0, 2.0, 1.0])
        self.assertEqual(comparison.argmax_reference, 0)
        self.assertEqual(comparison.argmax_runtime, 0)

    def test_nan_does_not_win(self):
        self.assertEqual(parity_random.parity_real.argmax([1.0, math.nan]), 0)
        self.assertEqual(parity_random.parity_real.argmax([math.nan, 1.0]), 0)
        self.assertEqual(parity_random.parity_real.argmax([math.nan, math.nan]), 0)

    def test_length_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            parity_random.compare_logits([1.0, 2.0], [1.0])

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            parity_random.compare_logits([], [])

    def test_json_safe_for_inf(self):
        comparison = parity_random.compare_logits(
            [0.0, 1.0], [math.inf, 1.0], tolerance=1e-5
        )
        data = comparison.to_dict()
        json.dumps(data, allow_nan=False)
        self.assertEqual(data["max_abs_delta"], "inf")
        self.assertFalse(data["within_tolerance"])


# ---------------------------------------------------------------------------
# Сводка и отчёт
# ---------------------------------------------------------------------------


def make_result(
    index: int,
    delta: float | None,
    *,
    argmax_match: bool = True,
    within: bool | None = None,
    cli_match: bool | None = None,
    error: str | None = None,
    meta_problems: list[str] | None = None,
) -> parity_random.ConfigResult:
    result = parity_random.ConfigResult(
        config=parity_random.generate_config(0, index),
        max_abs_delta=delta,
        argmax_reference=0,
        argmax_runtime=0 if argmax_match else 1,
        argmax_match=argmax_match,
        within_tolerance=(delta is not None and delta < parity_random.DEFAULT_TOLERANCE)
        if within is None
        else within,
        cli_argmax=0,
        cli_match=argmax_match if cli_match is None else cli_match,
        meta_problems=list(meta_problems or []),
        error=error,
    )
    return result


class TestSummaryAndReport(unittest.TestCase):
    def test_worst_is_max_delta(self):
        results = [
            make_result(0, 1e-8),
            make_result(1, 4e-6),
            make_result(2, 3e-7),
        ]
        summary = parity_random.summarize(
            seed=0, count=3, tolerance=1e-5, results=results, workdir="/tmp/x"
        )
        self.assertTrue(summary["ok"])
        self.assertEqual(summary["worst"]["index"], 1)
        self.assertAlmostEqual(summary["worst"]["max_abs_delta"], 4e-6)
        self.assertEqual(summary["passed"], 3)
        self.assertEqual(summary["failures"], [])

    def test_counts_and_failures(self):
        results = [
            make_result(0, 1e-8),
            make_result(1, 2e-5),  # вне порога
            make_result(2, None, error="ValueError: бум"),
        ]
        summary = parity_random.summarize(
            seed=0, count=3, tolerance=1e-5, results=results
        )
        self.assertFalse(summary["ok"])
        self.assertEqual(summary["passed"], 1)
        self.assertEqual(summary["failed"], 2)
        self.assertEqual(summary["errors"], 1)
        self.assertEqual(summary["outside_tolerance"], 1)
        self.assertEqual(summary["argmax_mismatches"], 0)
        self.assertEqual(
            [record["index"] for record in summary["failures"]], [1, 2]
        )

    def test_argmax_mismatch_counted(self):
        results = [make_result(0, 0.0, argmax_match=False)]
        summary = parity_random.summarize(
            seed=0, count=1, tolerance=1e-5, results=results
        )
        self.assertEqual(summary["argmax_mismatches"], 1)
        self.assertFalse(summary["ok"])
        self.assertIn("argmax", summary["failures"][0]["reasons"][0])

    def test_cli_mismatch_fails(self):
        result = make_result(0, 0.0, cli_match=False)
        self.assertFalse(result.passed)
        self.assertTrue(
            any("CLI" in reason for reason in result.failure_reasons())
        )

    def test_meta_problems_fail(self):
        result = make_result(0, 0.0, meta_problems=["rope_theta: C=10000, HF=1e+06"])
        self.assertFalse(result.passed)
        self.assertIn("rope_theta", result.failure_reasons()[0])

    def test_error_has_priority(self):
        result = make_result(0, None, error="RuntimeError: нет C")
        self.assertNotEqual(result.failure_reasons(), [])
        self.assertTrue(result.failure_reasons()[0].startswith("ошибка"))

    def test_report_mentions_worst_and_reproduction(self):
        results = [make_result(0, 1e-8), make_result(12, 4e-6)]
        summary = parity_random.summarize(
            seed=0, count=2, tolerance=1e-5, results=results
        )
        report = parity_random.format_report(summary)
        self.assertIn("худшее расхождение", report)
        self.assertIn("индекс 12", report)
        self.assertIn("--only 12", report)
        self.assertIn("вердикт: PASS", report)
        self.assertIn("конфигураций: 2", report)

    def test_report_lists_failures_with_config(self):
        results = [make_result(3, 9e-5)]
        summary = parity_random.summarize(
            seed=0, count=1, tolerance=1e-5, results=results
        )
        report = parity_random.format_report(summary)
        self.assertIn("провалившиеся конфигурации", report)
        self.assertIn("--only 3", report)
        self.assertIn("max|Δ|", report)
        self.assertIn("вердикт: FAIL", report)

    def test_report_survives_all_errors(self):
        results = [make_result(0, None, error="RuntimeError: нет C")]
        summary = parity_random.summarize(
            seed=0, count=1, tolerance=1e-5, results=results
        )
        self.assertIsNone(summary["worst"])
        report = parity_random.format_report(summary)
        self.assertIn("нет данных", report)
        self.assertIn("вердикт: FAIL", report)

    def test_format_json_roundtrip(self):
        results = [make_result(0, 0.0), make_result(1, 2e-5)]
        summary = parity_random.summarize(
            seed=0, count=2, tolerance=1e-5, results=results, workdir="/tmp/x"
        )
        text = parity_random.format_json(summary)
        parsed = json.loads(text)
        self.assertEqual(parsed["total"], 2)
        self.assertEqual(parsed["worst"]["index"], 1)
        self.assertEqual(parsed["configs"][0]["config"]["seed"], 0)
        # В JSON нет логитов — только конфигурация и числа сверки.
        self.assertNotIn("logits", text)

    def test_record_is_json_safe_with_inf(self):
        result = make_result(0, math.inf)
        record = parity_random._result_record(result)
        json.dumps(record, allow_nan=False)
        self.assertEqual(record["max_abs_delta"], "inf")
        self.assertFalse(record["passed"])

    def test_format_config_line(self):
        config = parity_random.generate_config(0, 0)
        line = parity_random.format_config_line(config)
        self.assertIn(f"индекс {config.index}", line)
        self.assertIn(f"seed {config.seed}", line)
        self.assertIn("block=", line)


class TestReproduction(unittest.TestCase):
    def test_reproduction_command(self):
        config = parity_random.generate_config(7, 12)
        command = parity_random.reproduction_command(config)
        self.assertIn("tools/parity_random.py", command)
        self.assertIn("--seed 7", command)
        self.assertIn("--only 12", command)

    def test_only_config_is_reproducible_without_prefix(self):
        """Конфигурация idx=N не зависит от прогонов 0..N-1."""
        configs = parity_random.generate_configs(5, 20)
        for index in (0, 1, 7, 19):
            self.assertEqual(
                configs[index], parity_random.generate_config(5, index)
            )


# ---------------------------------------------------------------------------
# Парсер CLI
# ---------------------------------------------------------------------------


class TestParser(unittest.TestCase):
    def test_defaults(self):
        args = parity_random.parse_args([])
        self.assertEqual(args.count, parity_random.DEFAULT_COUNT)
        self.assertEqual(args.seed, parity_random.DEFAULT_SEED)
        self.assertAlmostEqual(args.tolerance, 1e-5)
        self.assertFalse(args.json)
        self.assertIsNone(args.only)
        self.assertFalse(args.keep)
        self.assertTrue(str(args.workdir).endswith("parity_random"))

    def test_flags(self):
        args = parity_random.parse_args(
            ["--count", "7", "--seed", "3", "--tolerance", "1e-6", "--json",
             "--keep", "--only", "2,4"]
        )
        self.assertEqual(args.count, 7)
        self.assertEqual(args.seed, 3)
        self.assertAlmostEqual(args.tolerance, 1e-6)
        self.assertTrue(args.json)
        self.assertTrue(args.keep)
        self.assertEqual(args.only, [2, 4])

    def test_parse_only(self):
        self.assertEqual(parity_random.parse_only("0"), [0])
        self.assertEqual(parity_random.parse_only("3,1,3"), [3, 1])
        self.assertEqual(parity_random.parse_only(" 5 , 2 "), [5, 2])

    def test_parse_only_rejects(self):
        for text in ("", ",", "x", "-1", "1,x"):
            with self.subTest(text=text), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    parity_random.parse_args(["--only", text])
                self.assertEqual(caught.exception.code, 2)

    def test_bad_numbers(self):
        for argv in (
            ["--count", "0"],
            ["--count", "-1"],
            ["--count", "x"],
            ["--seed", "-1"],
            ["--tolerance", "0"],
            ["--tolerance", "-1"],
            ["--tolerance", "nan"],
        ):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    parity_random.parse_args(argv)
                self.assertEqual(caught.exception.code, 2)


# ---------------------------------------------------------------------------
# Окружение (без запуска C)
# ---------------------------------------------------------------------------


class TestEnvironment(unittest.TestCase):
    def test_missing_is_empty_or_explains(self):
        env = parity_random.check_environment()
        missing = env.missing()
        if env.ready:
            self.assertEqual(missing, [])
        else:
            self.assertTrue(missing)
            self.assertEqual(env.ready, False)

    def test_probe_source_is_precise(self):
        """Проба печатает 17 цифр: иначе max|Δ| упирается в формат."""
        source = parity_random.probe_source()
        self.assertIn("%.17g", source)
        self.assertNotIn("%.9g", source)
        self.assertIn('vtm_tensor_by_name(machine, "logits")', source)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
