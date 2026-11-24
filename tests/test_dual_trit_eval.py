"""Тесты ``tools/dual_trit_eval.py`` (задача 0148) без torch.

Запуск::

    python3 -m unittest tests.test_dual_trit_eval -v
    python3 -m unittest discover -s tests

Проверяется то, что можно проверить на машине без GPU-стека: чистая
математика разложения строки на два трита, арифметика размеров формата
TRT1, выбор матриц, разбор вариантов и отчёт. Torch-путь (квантование
реальных матриц) проверяется отдельным тестом и скипается, если torch нет:
он обязан совпадать с чистой функцией строчка-в-строчку.
"""
import io
import json
import math
import os
import random
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools import dual_trit_eval as d  # noqa: E402

from qat import ternary  # noqa: E402 - чистые функции, без torch

#: Ссылка на torch для torch-тестов; заполняется :func:`_load_torch`.
torch = None


def _load_torch():
    """Импортировать torch один раз; None, если его нет или импорт отравлен.

    torch 2.15 после НЕУДАЧНОГО импорта оставляет себя в ``sys.modules`` и
    ломает повторный импорт («can only be initialized once»), поэтому
    результат запоминается: не вышло — torch-тесты скипаются, без повторов.
    """
    global torch
    if torch is not None:
        return torch
    try:
        import torch as module
    except Exception:  # noqa: BLE001 - нет .pylibs/другая версия Python
        return None
    torch = module
    return torch


HAS_TORCH = _load_torch() is not None


class TestRoundingAndSingleTrit(unittest.TestCase):
    def test_round_half_away(self):
        self.assertEqual(d.round_half_away(0.5), 1)
        self.assertEqual(d.round_half_away(-0.5), -1)
        self.assertEqual(d.round_half_away(1.5), 2)
        self.assertEqual(d.round_half_away(-1.5), -2)
        self.assertEqual(d.round_half_away(2.49), 2)
        self.assertEqual(d.round_half_away(-2.49), -2)
        self.assertEqual(d.round_half_away(0.0), 0)

    def test_round_half_away_rejects_non_finite(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                d.round_half_away(value)

    def test_single_trit_row(self):
        trits, scale = d.single_trit([1.0, -1.0, 0.4, 0.0])
        self.assertEqual(scale, 1.0)
        self.assertEqual(trits, (1, -1, 0, 0))
        self.assertEqual(d.reconstruct_single(trits, scale), [1.0, -1.0, 0.0, 0.0])

    def test_single_trit_zero_row(self):
        trits, scale = d.single_trit([0.0, 0.0])
        self.assertEqual(scale, 0.0)
        self.assertEqual(trits, (0, 0))
        self.assertEqual(d.reconstruct_single(trits, scale), [0.0, 0.0])


class TestDualTritRow(unittest.TestCase):
    def test_exact_small_row(self):
        """Жадное разложение [1, -1, 0.5, 0] точное: s1=1, s2=0.5.

        Проверяется и шаг МНК: он обязан вернуть те же масштабы (эта строка
        уже оптимальна), а не «уехать» из-за вырожденной системы.
        """
        row = d.dual_trit_row([1.0, -1.0, 0.5, 0.0])
        self.assertEqual(row.a, (1, -1, 1, 0))
        self.assertEqual(row.b, (0, 0, -1, 0))
        self.assertAlmostEqual(row.s1, 1.0)
        self.assertAlmostEqual(row.s2, 0.5)
        self.assertTrue(row.ls_applied)
        self.assertEqual(d.reconstruct_dual(row), [1.0, -1.0, 0.5, 0.0])

    def test_greedy_is_recorded(self):
        row = d.dual_trit_row([0.9, -0.4, 0.2], ls_step=False)
        self.assertEqual(row.greedy_s1, row.s1)
        self.assertEqual(row.greedy_s2, row.s2)
        self.assertFalse(row.ls_applied)
        self.assertFalse(row.flipped)

    def test_ls_never_worse_than_greedy(self):
        rng = random.Random(20261124)
        for _ in range(50):
            values = [rng.gauss(0.0, 0.02) for _ in range(257)]
            greedy = d.dual_trit_row(values, ls_step=False)
            least_squares = d.dual_trit_row(values)
            error_greedy = d.relative_error(values, d.reconstruct_dual(greedy))
            error_ls = d.relative_error(values, d.reconstruct_dual(least_squares))
            self.assertLessEqual(error_ls, error_greedy + 1e-12)

    def test_dual_beats_single_on_typical_row(self):
        rng = random.Random(7)
        values = [rng.gauss(0.0, 0.02) for _ in range(2000)]
        single_trits, single_scale = d.single_trit(values)
        error_single = d.relative_error(
            values, d.reconstruct_single(single_trits, single_scale)
        )
        row = d.dual_trit_row(values)
        error_dual = d.relative_error(values, d.reconstruct_dual(row))
        self.assertLess(error_dual, error_single * 0.8)
        self.assertGreaterEqual(row.s2, 0.0)
        self.assertGreater(row.s1, 0.0)

    def test_levels_are_nine_not_four(self):
        levels = d.dual_trit_levels(1.0, 0.3)
        self.assertEqual(len(levels), 9)
        self.assertEqual(levels[0], -1.3)
        self.assertEqual(levels[-1], 1.3)
        self.assertIn(0.0, levels)

    def test_flip_keeps_positive_scale(self):
        """Если МНК даёт s2 < 0, знак переносится в B, масштаб остаётся ≥ 0."""
        rng = random.Random(11)
        flipped = 0
        for _ in range(300):
            values = [rng.gauss(0.0, 0.02) for _ in range(64)]
            row = d.dual_trit_row(values)
            self.assertGreaterEqual(row.s2, 0.0)
            self.assertGreater(row.s1, 0.0)
            self.assertTrue(all(trit in (-1, 0, 1) for trit in row.a + row.b))
            if row.flipped:
                flipped += 1
        self.assertIsInstance(flipped, int)

    def test_empty_row(self):
        row = d.dual_trit_row([])
        self.assertEqual((row.s1, row.s2, row.a, row.b), (0.0, 0.0, (), ()))
        self.assertEqual(d.reconstruct_dual(row), [])

    def test_degenerate_row_falls_back_to_greedy(self):
        """Остаток нулевой (s2 = 0) — система вырождена, остаётся жадное."""
        row = d.dual_trit_row([1.0, 0.0, -1.0])
        self.assertEqual(row.s2, 0.0)
        self.assertFalse(row.ls_applied)
        self.assertEqual(d.reconstruct_dual(row), [1.0, 0.0, -1.0])


class TestSizes(unittest.TestCase):
    def test_packed_trit_bytes_alignment(self):
        self.assertEqual(d.packed_trit_bytes(0), 0)
        self.assertEqual(d.packed_trit_bytes(1), 32)
        self.assertEqual(d.packed_trit_bytes(160), 32)
        self.assertEqual(d.packed_trit_bytes(161), 64)

    def test_trit_is_1_6_bits(self):
        estimate = d.size_estimate("trit_block", 160 * 1000, 0)
        # 20 тритов в 32-битном слове = 1,6 бита на вес; масштабы 128 → +0,125.
        self.assertAlmostEqual(8.0 * estimate.data_bytes / estimate.params, 1.6,
                               places=6)
        self.assertAlmostEqual(estimate.bits_per_param, 1.725, places=3)

    def test_dual_is_twice_trit(self):
        trit = d.size_estimate("trit_block", 4864 * 896, 0)
        dual = d.size_estimate("dual_block", 4864 * 896, 0)
        self.assertEqual(dual.data_bytes, 2 * trit.data_bytes)
        self.assertEqual(dual.scale_bytes, 2 * trit.scale_bytes)
        self.assertAlmostEqual(dual.bits_per_param, 2 * trit.bits_per_param, places=9)

    def test_row_scales_are_smaller_than_block_scales(self):
        """У строки из 4864 весов один масштаб против 38 блоков по 128."""
        row = d.size_estimate("dual_row", 4864 * 896, 896)
        block = d.size_estimate("dual_block", 4864 * 896, 0)
        self.assertLess(row.scale_bytes, block.scale_bytes)
        self.assertLessEqual(row.total_bytes, block.total_bytes)

    def test_int2_is_smaller_than_dual(self):
        int2 = d.size_estimate("int2_block", 4864 * 896, 0)
        dual = d.size_estimate("dual_block", 4864 * 896, 0)
        self.assertAlmostEqual(int2.bits_per_param, 2.125, places=3)
        self.assertLess(int2.total_bytes, dual.total_bytes)
        self.assertLess(int2.bits_per_param, dual.bits_per_param / 1.5)

    def test_int8_row(self):
        estimate = d.size_estimate("int8_row", 4864 * 896, 896)
        self.assertAlmostEqual(estimate.bits_per_param, 8.003, places=3)

    def test_unknown_kind_raises(self):
        with self.assertRaises(d.DualTritError):
            d.size_estimate("trit16", 100, 0)

    def test_zero_params(self):
        estimate = d.size_estimate("dual_row", 0, 0)
        self.assertEqual(estimate.total_bytes, 0)
        self.assertEqual(estimate.bits_per_param, 0.0)


class TestShape(unittest.TestCase):
    def test_3b_shape_matches_known_parameter_count(self):
        shape = d.linear_shape_from_config(d.FALLBACK_3B_CONFIG)
        # Qwen2.5-3B: 36 слоёв × 77 070 336 Linear-параметров.
        self.assertEqual(shape.linear_params, 2_774_532_096)
        self.assertEqual(shape.rows, 1_032_192)
        self.assertEqual(shape.embed_params, 311_164_928)
        self.assertGreater(shape.total_params, 3_000_000_000)

    def test_0_5b_shape(self):
        config = {
            "hidden_size": 896, "intermediate_size": 4864,
            "num_hidden_layers": 24, "num_attention_heads": 14,
            "num_key_value_heads": 2, "vocab_size": 151936,
            "tie_word_embeddings": True,
        }
        shape = d.linear_shape_from_config(config)
        self.assertEqual(shape.linear_params, 357_826_560)
        self.assertEqual(shape.rows, 304_128)
        self.assertEqual(shape.head_dim, 64)

    def test_missing_fields_raise(self):
        with self.assertRaises(d.DualTritError):
            d.linear_shape_from_config({"hidden_size": 0})

    def test_three_b_report(self):
        report = d.three_b_report(d.FALLBACK_3B_CONFIG)
        sizes = report["sizes"]
        trit = sizes["trit_block"]["file_trit_embed"]
        dual = sizes["dual_block"]["file_trit_embed"]
        dual_row = sizes["dual_row"]
        int2 = sizes["int2_block"]
        # Один трит на 3B ≈ 688 МБ (0143) — наша оценка в тех же 10%.
        self.assertLess(abs(trit - 688e6), 0.1 * 688e6)
        # Два трита ≈ 1,2-1,3 ГБ с trit-таблицей и ≈1,4-1,5 ГБ с int8.
        self.assertTrue(1.1e9 < dual < 1.4e9, dual)
        self.assertTrue(1.3e9 < dual_row["file_int8_embed"] < 1.6e9)
        # Два трита — ровно вдвое дороже одного, но меньше Q4 (~2 ГБ).
        self.assertTrue(1.8 < dual / trit < 2.05, dual / trit)
        self.assertLess(dual, 2 * 10 ** 9)
        # Честный int2 при этом заметно меньше двух тритов.
        self.assertLess(int2["linear_bytes"], dual_row["linear_bytes"])
        self.assertGreater(int2["bits_per_param"], 2.0)
        self.assertLess(int2["bits_per_param"], 2.2)


class TestSelection(unittest.TestCase):
    def test_is_linear_name(self):
        self.assertTrue(d.is_linear_name("model.layers.0.self_attn.q_proj"))
        self.assertTrue(d.is_linear_name("model.layers.0.mlp.down_proj"))
        self.assertFalse(d.is_linear_name("lm_head"))
        self.assertFalse(d.is_linear_name("model.embed_tokens"))

    def test_parse_variants_keeps_order_and_dedupes(self):
        self.assertEqual(
            d.parse_variants("dual_row, bf16, dual_row"),
            ["dual_row", "bf16"],
        )

    def test_parse_variants_all(self):
        self.assertEqual(d.parse_variants("all"), list(d.ALL_VARIANTS))
        self.assertEqual(d.parse_variants("all,bf16"), list(d.ALL_VARIANTS))
        self.assertEqual(d.parse_variants("bf16,all"), list(d.ALL_VARIANTS))

    def test_parse_variants_rejects_unknown(self):
        with self.assertRaises(d.DualTritError):
            d.parse_variants("dual_row,int4")
        with self.assertRaises(d.DualTritError):
            d.parse_variants("")

    def test_every_variant_has_size_kind_and_title(self):
        for variant in d.ALL_VARIANTS:
            self.assertIn(variant, d.SIZE_KIND)
            self.assertIn(variant, d.VARIANT_TITLES)


class TestReport(unittest.TestCase):
    @staticmethod
    def _synthetic() -> dict:
        shape = d.linear_shape_from_config(d.FALLBACK_3B_CONFIG)
        sizes = {
            variant: d.size_rows(shape, [d.SIZE_KIND[variant]])[d.SIZE_KIND[variant]]
            for variant in ("bf16", "dual_row")
        }
        return {
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "ppl": {
                "data": "/tmp/train.jsonl", "lines": 20, "tokens": 2047,
                "max_tokens": 128, "method": "tools.parity_cold.hf_answer_ppl",
            },
            "decode_tokens": 24,
            "shape": {
                "linear_params": shape.linear_params, "linear_rows": shape.rows,
                "embed_params": shape.embed_params,
            },
            "peak_rss_mb": 2400.0,
            "timings": {"total_seconds": 300.0},
            "variants": {
                "bf16": {
                    "ppl": 16.13, "ratio_vs_bf16": 1.0,
                    "quant": {"relative_error": 0.0},
                    "answers": [{"prompt": "Столица Франции — это",
                                 "text": "Париж."}],
                },
                "dual_row": {
                    "ppl": 1.2e5, "ratio_vs_bf16": 7439.0,
                    "quant": {"relative_error": 0.42},
                    "answers": [{"prompt": "Столица Франции — это",
                                 "text": "???"}],
                },
            },
            "sizes": sizes,
            "three_b": d.three_b_report(d.FALLBACK_3B_CONFIG),
        }

    def test_report_has_numbers_and_decode(self):
        text = d.format_report(self._synthetic())
        self.assertIn("0148", text)
        self.assertIn("16.13", text)
        self.assertIn("7 439", text)
        self.assertIn("Париж", text)
        self.assertIn("два трита на 3B", text)
        self.assertNotIn("nan", text.lower())

    def test_report_json_survives_roundtrip(self):
        payload = self._synthetic()
        text = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        restored = json.loads(text)
        self.assertEqual(restored["variants"]["dual_row"]["ppl"], 1.2e5)
        d.format_report(restored)


class TestCli(unittest.TestCase):
    def test_dry_run_without_torch(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = d.main(["--dry-run"])
        self.assertEqual(code, 0)
        text = buffer.getvalue()
        self.assertIn("dual_trit_eval: dry-run", text)
        self.assertIn("dual_block", text)
        self.assertIn("int2_block", text)

    def test_dry_run_bad_variant(self):
        with redirect_stdout(io.StringIO()):
            code = d.main(["--dry-run", "--variants", "int4"])
        self.assertEqual(code, 2)

    def test_quant_dtype_default_is_fp32(self):
        args = d.parse_args(["--dry-run"])
        self.assertEqual(args.quant_dtype, "float32")

    def test_skip_ppl_flag(self):
        args = d.parse_args(["--skip-ppl"])
        self.assertTrue(args.skip_ppl)
        self.assertFalse(d.parse_args([]).skip_ppl)

    def test_default_model_is_cached_qwen(self):
        model = d.default_model()
        if d._cached_snapshot("Qwen/Qwen2.5-0.5B-Instruct") is None:
            self.assertEqual(model, "Qwen/Qwen2.5-0.5B-Instruct")
        else:
            self.assertIn("Qwen2.5-0.5B", model)
            self.assertTrue(Path(model, "config.json").is_file())


@unittest.skipUnless(HAS_TORCH, "torch не установлен (.pylibs) — torch-путь скипнут")
class TestTorchPath(unittest.TestCase):
    def test_dual_matrix_matches_pure_rows(self):
        torch.manual_seed(0)
        weight = torch.randn(8, 129) * 0.02
        deq, stats = d.dual_trit_weight(weight, None, ls_step=True)
        self.assertEqual(tuple(deq.shape), tuple(weight.shape))
        self.assertEqual(stats.ls_applied + stats.ls_fallback, 8)
        for index in range(weight.shape[0]):
            row = weight[index].tolist()
            pure = d.dual_trit_row(row)
            approx = d.reconstruct_dual(pure)
            error_pure = d.relative_error(row, approx)
            error_torch = d.relative_error(row, deq[index].tolist())
            self.assertAlmostEqual(error_pure, error_torch, places=5)

    def test_ls_not_worse_in_torch(self):
        torch.manual_seed(1)
        weight = torch.randn(32, 512) * 0.02
        _ls, stats_ls = d.dual_trit_weight(weight, None, ls_step=True)
        _greedy, stats_greedy = d.dual_trit_weight(weight, None, ls_step=False)
        self.assertLessEqual(stats_ls.relative_error, stats_greedy.relative_error)

    def test_block_mode_shape_and_zero_row(self):
        weight = torch.zeros(3, 300)
        deq, stats = d.dual_trit_weight(weight, d.BLOCK_SIZE)
        self.assertEqual(tuple(deq.shape), (3, 300))
        self.assertEqual(float(deq.abs().sum()), 0.0)
        self.assertEqual(stats.relative_error, 0.0)

    def test_trit_block_matches_qat_rules(self):
        """Один трит, блок 128 — арифметика qat.ternary (absmax, half away)."""
        torch.manual_seed(2)
        weight = torch.randn(2, d.BLOCK_SIZE) * 0.02
        deq, _stats = d.trit_weight(weight, d.BLOCK_SIZE)
        row = weight[0].tolist()
        trits, scales = ternary.quantize(row, block_size=d.BLOCK_SIZE)
        expected = ternary.dequantize(trits, scales, block_size=d.BLOCK_SIZE)
        for got, want in zip(deq[0].tolist(), expected):
            self.assertAlmostEqual(got, want, places=6)

    def test_int8_row_matches_pure(self):
        torch.manual_seed(3)
        weight = torch.randn(4, 64) * 0.02
        deq, _stats = d.int8_row_weight(weight)
        for index in range(4):
            row = weight[index].tolist()
            scale = max(abs(value) for value in row) / 127.0
            expected = [d.round_half_away(value / scale) * scale for value in row]
            for got, want in zip(deq[index].tolist(), expected):
                self.assertAlmostEqual(got, want, places=7)


if __name__ == "__main__":
    unittest.main()
