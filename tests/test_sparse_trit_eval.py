"""Тесты ``tools/sparse_trit_eval.py`` (задача 0149) без torch.

Запуск::

    python3 -m unittest tests.test_sparse_trit_eval -v
    python3 -m unittest discover -s tests

Проверяется то, что можно проверить на машине без GPU-стека: чистая
математика «база + разреженный остаток», выбор доли и критерия, арифметика
фактического размера (триты + масштабы + индексы), разбор вариантов, оценка
3B и отчёт. Torch-путь (квантование реальных матриц) скипается, если torch
нет, но обязан совпадать с чистой функцией — это отдельный тест.

Важное ограничение чистой функции: блоки режутся подряд по плоскому списку,
а torch режет по строкам матрицы. Совпадают они, когда число столбцов кратно
128 (в Qwen2.5 это так для всех семи матриц), поэтому torch-тесты берут
матрицы шириной 128/256.
"""
import io
import json
import os
import random
import subprocess
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools import sparse_trit_eval as s  # noqa: E402

def _torch_available() -> bool:
    """Есть ли **рабочий** torch — проба в отдельном процессе.

    ``import torch`` из ``.pylibs`` при неудаче ставит
    ``sys._torch_import_started``, и все последующие импорты в этом процессе
    падают с «can only be initialized once per process». Тогда чужие тесты
    (``test_embed_variants``, ``test_kl_distill``) не скипаются, а падают в
    ``setUpClass`` — потому что ``has_torch()`` там сделан через
    ``find_spec``, а он видит каталог, а не работоспособность. Поэтому
    пробуем в дочернем процессе: свой процесс не травим.
    """
    env = dict(os.environ)
    parts = [str(ROOT)]
    if (Path(ROOT) / ".pylibs").is_dir():
        parts.append(str(Path(ROOT) / ".pylibs"))
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", "import torch"],
            capture_output=True, text=True, env=env, timeout=180,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


HAS_TORCH = _torch_available()
if HAS_TORCH:  # сам импорт — только если проба сказала «работает»
    import torch  # noqa: E402
else:  # noqa: SIM108 - ветка нужна явно
    torch = None  # noqa: F841 - без torch torch-тесты скипаются


class TestRoundingAndSelection(unittest.TestCase):
    def test_round_half_away(self):
        self.assertEqual(s.round_half_away(0.5), 1)
        self.assertEqual(s.round_half_away(-0.5), -1)
        self.assertEqual(s.round_half_away(2.49), 2)
        self.assertEqual(s.round_half_away(-2.49), -2)
        self.assertEqual(s.round_half_away(0.0), 0)

    def test_round_half_away_rejects_non_finite(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                s.round_half_away(value)

    def test_trit_of(self):
        self.assertEqual(s.trit_of(0.9, 1.0), 1)
        self.assertEqual(s.trit_of(-0.9, 1.0), -1)
        self.assertEqual(s.trit_of(0.4, 1.0), 0)
        self.assertEqual(s.trit_of(0.9, 0.0), 0)

    def test_selected_per_block(self):
        self.assertEqual(s.selected_per_block(0.05), 6)    # 6.4 → 6
        self.assertEqual(s.selected_per_block(0.10), 13)   # 12.8 → 13
        self.assertEqual(s.selected_per_block(0.20), 26)   # 25.6 → 26
        self.assertEqual(s.selected_per_block(0.0), 0)
        self.assertEqual(s.selected_per_block(1.0), s.BLOCK_SIZE)
        self.assertEqual(s.selected_per_block(0.5, block_size=4), 2)

    def test_selected_per_block_rejects_bad_fraction(self):
        with self.assertRaises(s.SparseTritError):
            s.selected_per_block(-0.1)
        with self.assertRaises(s.SparseTritError):
            s.selected_per_block(1.5)

    def test_actual_fraction_is_not_nominal(self):
        """6/128 = 4,69 %, а не 5 % — это печатается в отчёте."""
        self.assertAlmostEqual(s.actual_fraction(0.05), 6 / 128)
        self.assertAlmostEqual(s.actual_fraction(0.10), 13 / 128)
        self.assertAlmostEqual(s.actual_fraction(0.20), 26 / 128)


class TestBlockDecompose(unittest.TestCase):
    def _decompose(self, values, fraction, criterion, block_size=4):
        return s.sparse_block_decompose(
            values, fraction=fraction, criterion=criterion, block_size=block_size
        )

    def test_exact_row_with_all_selected(self):
        """[1, -1, 0.5, 0]: s1=1, A=[1,-1,1,0], r=[0,0,-0.5,0], s2=0.5."""
        block = self._decompose([1.0, -1.0, 0.5, 0.0], 1.0, s.CRITERION_RESIDUAL)
        self.assertEqual(block.s1, (1.0,))
        self.assertEqual(block.a, ((1, -1, 1, 0),))
        self.assertEqual(block.s2, (0.5,))
        self.assertEqual(block.b, ((0, 0, -1, 0),))
        self.assertEqual(block.selected, ((0, 1, 2, 3),))
        self.assertEqual(s.reconstruct_sparse(block), [1.0, -1.0, 0.5, 0.0])

    def test_residual_criterion_picks_largest_remainder(self):
        """Блок [1.0, 0.4, -0.2, 0]: база даёт [1,0,0,0], остаток [0,.4,-.2,0]."""
        block = self._decompose([1.0, 0.4, -0.2, 0.0], 0.25, s.CRITERION_RESIDUAL)
        self.assertEqual(block.selected, ((1,),))     # |r| = 0.4 — максимум
        self.assertAlmostEqual(block.s2[0], 0.4)
        self.assertEqual(block.b[0], (0, 1, 0, 0))

    def test_magnitude_criterion_picks_the_outlier_and_wastes_it(self):
        """Критерий по |W| берёт вес 1.0 — а его остаток уже 0.

        Это и есть проверяемая гипотеза: у базы выброс представлен точно
        (``A = ±1``, ``s1 = |W|max``), поэтому поправка по ``|W|`` уходит
        в ноль. Тест фиксирует такое поведение, а не «качество».
        """
        block = self._decompose([1.0, 0.4, -0.2, 0.0], 0.25, s.CRITERION_MAGNITUDE)
        self.assertEqual(block.selected, ((0,),))
        self.assertEqual(block.s2, (0.0,))
        self.assertEqual(block.b, ((0, 0, 0, 0),))
        self.assertEqual(s.reconstruct_sparse(block), [1.0, 0.0, 0.0, 0.0])

    def test_tie_break_by_position(self):
        """Равные по модулю остатки берутся по порядку — прогон воспроизводим.

        [0.5, 0.25, -0.25, 0]: s1 = 0.5, A = [1, 1, -1, 0], остатки
        [0, -0.25, 0.25, 0] — у позиций 1 и 2 модуль равен, берётся меньшая.
        """
        block = self._decompose([0.5, 0.25, -0.25, 0.0], 0.25, s.CRITERION_RESIDUAL)
        self.assertEqual(block.selected, ((1,),))
        self.assertAlmostEqual(block.s2[0], 0.25)
        self.assertEqual(block.b[0], (0, -1, 0, 0))

    def test_unselected_b_is_zero(self):
        rng = random.Random(11)
        values = [rng.gauss(0.0, 0.02) for _ in range(64)]
        block = self._decompose(values, 0.1, s.CRITERION_RESIDUAL)
        chosen = set(block.selected[0])
        for position, trit in enumerate(block.b[0]):
            if position not in chosen:
                self.assertEqual(trit, 0)
        for item in block.b[0]:
            self.assertIn(item, (-1, 0, 1))

    def test_k_zero_is_the_plain_base(self):
        """Нулевая доля — это ровно один трит: второй сегмент пуст."""
        values = [0.9, -0.4, 0.2, 0.05]
        block = self._decompose(values, 0.0, s.CRITERION_RESIDUAL)
        self.assertEqual(block.k, 0)
        self.assertEqual(block.selected_total, 0)
        self.assertEqual(block.s2, (0.0,))
        self.assertEqual(block.b, ((0, 0, 0, 0),))
        single = self._decompose(values, 0.0, s.CRITERION_MAGNITUDE)
        self.assertEqual(s.reconstruct_sparse(block), s.reconstruct_sparse(single))

    def test_empty_values(self):
        block = self._decompose([], 0.05, s.CRITERION_RESIDUAL)
        self.assertEqual(block.n_values, 0)
        self.assertEqual(s.reconstruct_sparse(block), [])

    def test_padding_keeps_length(self):
        """Хвост добивается нулями, но восстановление возвращает длину входа."""
        values = [0.3, -0.7, 0.1, 0.0, 0.2]
        block = self._decompose(values, 0.5, s.CRITERION_RESIDUAL)
        self.assertEqual(block.n_values, 5)
        self.assertEqual(len(block.s1), 2)      # 5 значений → 2 блока по 4
        self.assertEqual(len(s.reconstruct_sparse(block)), 5)

    def test_more_fraction_means_less_error(self):
        rng = random.Random(20261124)
        values = [rng.gauss(0.0, 0.02) for _ in range(1280)]
        errors = []
        for fraction in (0.0, 0.05, 0.10, 0.20):
            block = s.sparse_block_decompose(
                values, fraction=fraction, criterion=s.CRITERION_RESIDUAL
            )
            errors.append(s.relative_error(values, s.reconstruct_sparse(block)))
        self.assertEqual(errors, sorted(errors, reverse=True))
        self.assertLess(errors[-1], errors[0])

    def test_residual_criterion_beats_magnitude_on_gaussian_weights(self):
        rng = random.Random(7)
        values = [rng.gauss(0.0, 0.02) for _ in range(4096)]
        residual = s.sparse_block_decompose(
            values, fraction=0.10, criterion=s.CRITERION_RESIDUAL
        )
        magnitude = s.sparse_block_decompose(
            values, fraction=0.10, criterion=s.CRITERION_MAGNITUDE
        )
        error_residual = s.relative_error(values, s.reconstruct_sparse(residual))
        error_magnitude = s.relative_error(values, s.reconstruct_sparse(magnitude))
        self.assertLess(error_residual, error_magnitude)

    def test_residual_share_targets_the_error(self):
        """Отбор по |r| снимает больше энергии ошибки, чем доля p."""
        rng = random.Random(3)
        values = [rng.gauss(0.0, 0.02) for _ in range(4096)]
        residual = s.sparse_block_decompose(
            values, fraction=0.05, criterion=s.CRITERION_RESIDUAL
        )
        magnitude = s.sparse_block_decompose(
            values, fraction=0.05, criterion=s.CRITERION_MAGNITUDE
        )
        share_residual = s.residual_share(values, residual)
        share_magnitude = s.residual_share(values, magnitude)
        self.assertGreater(share_residual, s.actual_fraction(0.05))
        self.assertGreater(share_residual, share_magnitude)

    def test_unknown_criterion_raises(self):
        with self.assertRaises(s.SparseTritError):
            s.sparse_block_decompose([1.0], fraction=0.1, criterion="random")

    def test_relative_error_zero_for_zero_weights(self):
        self.assertEqual(s.relative_error([0.0, 0.0], [0.0, 0.0]), 0.0)


class TestSizeMath(unittest.TestCase):
    def test_packed_trit_bytes_alignment(self):
        self.assertEqual(s.packed_trit_bytes(0), 0)
        self.assertEqual(s.packed_trit_bytes(1), 32)
        self.assertEqual(s.packed_trit_bytes(160), 32)
        self.assertEqual(s.packed_trit_bytes(161), 64)

    def test_scale_block_count(self):
        self.assertEqual(s.scale_block_count(0), 0)
        self.assertEqual(s.scale_block_count(128), 1)
        self.assertEqual(s.scale_block_count(129), 2)

    def test_offsets_bytes(self):
        self.assertEqual(s.offsets_bytes(0), 0)
        self.assertEqual(s.offsets_bytes(1), 1)      # 7 бит → 1 байт
        self.assertEqual(s.offsets_bytes(8), 7)      # 56 бит → 7 байт
        self.assertEqual(s.offsets_bytes(16), 14)

    def test_bitmap_bytes(self):
        self.assertEqual(s.bitmap_bytes(0), 0)
        self.assertEqual(s.bitmap_bytes(8), 1)
        self.assertEqual(s.bitmap_bytes(9), 2)

    def test_choose_index_kind_breaks_at_one_seventh(self):
        self.assertEqual(s.choose_index_kind(0.05), s.INDEX_OFFSETS)
        self.assertEqual(s.choose_index_kind(0.10), s.INDEX_OFFSETS)
        self.assertEqual(s.choose_index_kind(0.20), s.INDEX_BITMAP)
        # p·7 бит против 1 бита: 0.14·7 = 0.98 < 1, 0.15·7 = 1.05 > 1.
        self.assertEqual(s.choose_index_kind(0.14), s.INDEX_OFFSETS)
        self.assertEqual(s.choose_index_kind(0.15), s.INDEX_BITMAP)

    def test_index_bytes_for(self):
        self.assertEqual(s.index_bytes_for(s.INDEX_OFFSETS, 1000, 100), 88)
        self.assertEqual(s.index_bytes_for(s.INDEX_BITMAP, 1000, 100), 125)
        with self.assertRaises(s.SparseTritError):
            s.index_bytes_for("hash", 1000, 100)

    def test_trit_breakdown_is_1_6_bits(self):
        """Один трит: 1,6 бита данных + 0,125 бита масштаба = 1,725."""
        estimate = s.trit_breakdown(160 * 1000)
        self.assertAlmostEqual(8.0 * estimate.base_trit_bytes / estimate.params, 1.6,
                               places=6)
        self.assertAlmostEqual(estimate.bits_per_param, 1.725, places=3)

    def test_sparse_breakdown_exact_0_5b(self):
        """Фактические числа таблицы: 0,5B Linear, блоки 128, доли 5/10/20 %."""
        params = 357_826_560
        blocks = params // s.BLOCK_SIZE
        five = s.sparse_breakdown(params, 0.05, blocks=blocks)
        # база 71 565 312 + масштабы 5 591 040 + триты остатка 3 354 624
        # + s2 5 591 040 + смещения 14 676 480.
        self.assertEqual(five.base_trit_bytes, 71_565_312)
        self.assertEqual(five.base_scale_bytes, 5_591_040)
        self.assertEqual(five.second_trit_bytes, 3_354_624)
        self.assertEqual(five.second_scale_bytes, 5_591_040)
        self.assertEqual(five.index_kind, s.INDEX_OFFSETS)
        self.assertEqual(five.index_bytes, 14_676_480)
        self.assertEqual(five.total_bytes, 100_778_496)
        self.assertAlmostEqual(five.bits_per_param, 2.253, places=3)
        self.assertAlmostEqual(five.fraction, 6 / 128, places=12)

        ten = s.sparse_breakdown(params, 0.10, blocks=blocks)
        self.assertEqual(ten.selected, 36_341_760)
        self.assertEqual(ten.index_kind, s.INDEX_OFFSETS)
        self.assertEqual(ten.total_bytes, 121_814_784)

        twenty = s.sparse_breakdown(params, 0.20, blocks=blocks)
        self.assertEqual(twenty.selected, 72_683_520)
        self.assertEqual(twenty.index_kind, s.INDEX_BITMAP)
        self.assertEqual(twenty.index_bytes, 44_728_320)
        self.assertEqual(twenty.total_bytes, 142_012_416)

    def test_sparse_is_cheaper_than_dual_at_5_and_10(self):
        params = 357_826_560
        dual = s.dual_breakdown(params)
        self.assertEqual(dual.total_bytes, 154_312_704)
        self.assertEqual(dual.index_bytes, 0)      # выбраны все — индексы не нужны
        self.assertLess(s.sparse_breakdown(params, 0.05).total_bytes, dual.total_bytes)
        self.assertLess(s.sparse_breakdown(params, 0.10).total_bytes, dual.total_bytes)
        # 20 % со смещениями дороже, чем «два трита у всех»: вот почему bitmap.
        offsets = s.sparse_breakdown(params, 0.20, index=s.INDEX_OFFSETS)
        self.assertGreater(offsets.total_bytes, dual.total_bytes)

    def test_index_off_for_full_selection(self):
        estimate = s.sparse_breakdown(128 * 1000, 1.0)
        self.assertEqual(estimate.index_bytes, 0)
        self.assertEqual(estimate.selected, estimate.params)

    def test_zero_params(self):
        estimate = s.sparse_breakdown(0, 0.05)
        self.assertEqual(estimate.total_bytes, 0)
        self.assertEqual(estimate.bits_per_param, 0.0)
        self.assertEqual(estimate.fraction, 0.0)

    def test_sum_breakdowns(self):
        parts = [s.sparse_breakdown(128 * 100, 0.10) for _ in range(3)]
        total = s.sum_breakdowns(parts, kind="x", index_kind=s.INDEX_OFFSETS)
        self.assertEqual(total.params, sum(part.params for part in parts))
        self.assertEqual(total.total_bytes, sum(part.total_bytes for part in parts))
        self.assertEqual(total.index_bytes, sum(part.index_bytes for part in parts))

    def test_table_bytes(self):
        trit = s.table_bytes(1000, "trit")
        self.assertEqual(trit, s.packed_trit_bytes(1000) + s.scale_block_count(1000) * 2)
        self.assertEqual(s.table_bytes(1000, "bf16"), 2000)
        with self.assertRaises(s.SparseTritError):
            s.table_bytes(1000, "int4")


class TestShape(unittest.TestCase):
    #: Конфиг Qwen2.5-0.5B-Instruct (без head_dim — он выводится из hidden/heads).
    CONFIG_05B = {
        "hidden_size": 896, "intermediate_size": 4864,
        "num_hidden_layers": 24, "num_attention_heads": 14,
        "num_key_value_heads": 2, "vocab_size": 151936,
        "tie_word_embeddings": True,
    }

    def test_half_billion_shape(self):
        shape = s.linear_shape_from_config(self.CONFIG_05B)
        self.assertEqual(shape.head_dim, 64)
        self.assertEqual(shape.linear_params, 357_826_560)
        self.assertEqual(shape.linear_blocks, 2_795_520)
        self.assertEqual(shape.embed_params, 136_134_656)

    def test_three_b_shape(self):
        shape = s.linear_shape_from_config(s.FALLBACK_3B_CONFIG)
        # Qwen2.5-3B: head_dim 128, kv 2 → 256; 36 слоёв.
        self.assertEqual(shape.head_dim, 128)
        self.assertEqual(shape.linear_params, 2_774_532_096)
        self.assertEqual(shape.linear_blocks, 2_774_532_096 // s.BLOCK_SIZE)
        self.assertEqual(shape.embed_params, 311_164_928)
        self.assertGreater(shape.total_params, 3_000_000_000)

    def test_missing_fields_raise(self):
        with self.assertRaises(s.SparseTritError):
            s.linear_shape_from_config({"hidden_size": 0})

    def test_three_b_report(self):
        report = s.three_b_report(
            s.FALLBACK_3B_CONFIG, (0.05, 0.10, 0.20),
            (s.CRITERION_RESIDUAL, s.CRITERION_MAGNITUDE),
        )
        entries = report["entries"]
        one_trit = entries["trit1"]["file_trit_bytes"]
        self.assertLess(abs(one_trit - 688e6), 0.10 * 688e6)
        self.assertLess(entries["sparse_5"]["linear_bytes"],
                        entries["sparse_10"]["linear_bytes"])
        self.assertLess(entries["sparse_10"]["linear_bytes"],
                        entries["sparse_20"]["linear_bytes"])
        self.assertLess(entries["sparse_20"]["linear_bytes"],
                        entries["dual"]["linear_bytes"])
        self.assertLess(entries["dual"]["file_trit_bytes"],
                        report["reference"]["three_b_q4_gguf_bytes"])
        self.assertGreater(entries["sparse_5"]["file_trit_bytes"], one_trit)


class TestVariants(unittest.TestCase):
    def test_build_variants_order(self):
        variants = s.build_variants([0.05], [s.CRITERION_RESIDUAL, s.CRITERION_MAGNITUDE])
        self.assertEqual([variant.key for variant in variants],
                         ["bf16", "trit1", "residual_5", "magnitude_5"])
        self.assertEqual(variants[0].title, "bf16 (эталон)")
        self.assertEqual(variants[1].title, "1 трит (база, 0 %)")
        self.assertEqual(variants[2].title, "residual 5 %")
        self.assertFalse(variants[0].needs_quant)
        self.assertTrue(variants[2].is_sparse)

    def test_build_variants_without_baselines(self):
        variants = s.build_variants([0.05], [s.CRITERION_RESIDUAL], baselines=False)
        self.assertEqual([variant.key for variant in variants], ["residual_5"])

    def test_build_variants_rejects_unknown_criterion(self):
        with self.assertRaises(s.SparseTritError):
            s.build_variants([0.05], ["random"])

    def test_build_variants_rejects_empty(self):
        with self.assertRaises(s.SparseTritError):
            s.build_variants([], [], baselines=False)

    def test_parse_fractions_accepts_share_and_percent(self):
        self.assertEqual(s.parse_fractions("0.05,10,20,5"), [0.05, 0.10, 0.20])
        with self.assertRaises(s.SparseTritError):
            s.parse_fractions("0")
        with self.assertRaises(s.SparseTritError):
            s.parse_fractions("abc")
        with self.assertRaises(s.SparseTritError):
            s.parse_fractions("")

    def test_parse_criteria(self):
        self.assertEqual(s.parse_criteria("magnitude,residual,magnitude"),
                         ["magnitude", "residual"])
        with self.assertRaises(s.SparseTritError):
            s.parse_criteria("random")
        with self.assertRaises(s.SparseTritError):
            s.parse_criteria("")

    def test_best_variant_ignores_baselines(self):
        results = {
            "bf16": {"fraction": 0.0, "ppl": 16.13},
            "trit1": {"fraction": 0.0, "ppl": 1e7},
            "residual_5": {"fraction": 0.05, "ppl": 5e6},
            "magnitude_5": {"fraction": 0.05, "ppl": 9e6},
        }
        self.assertEqual(s.best_variant(results), "residual_5")

    def test_best_variant_empty(self):
        self.assertIsNone(s.best_variant({"bf16": {"fraction": 0.0, "ppl": 16.13}}))
        self.assertIsNone(s.best_variant({}))


class TestReport(unittest.TestCase):
    @staticmethod
    def _synthetic() -> dict:
        params = 357_826_560
        blocks = params // s.BLOCK_SIZE
        base = s._breakdown_dict(s.trit_breakdown(params, blocks=blocks))
        sparse = s._breakdown_dict(s.sparse_breakdown(params, 0.05, blocks=blocks))
        bf16 = s._empty_quant(s.Variant("bf16"), "bf16")
        trit1 = dict(bf16, name="trit1", variant="trit1", size=base,
                     relative_error=0.777, k=0, fraction_actual=0.0)
        five = dict(trit1, name="sparse", variant="residual_5",
                    fraction=0.05, criterion="residual", k=6,
                    fraction_actual=6 / 128, relative_error=0.709,
                    fixed_share=0.182, size=sparse)
        return {
            "task": "0149",
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "device": "cpu",
            "ppl": {
                "data": "/tmp/train.jsonl", "lines": 20, "tokens": 2047,
                "max_tokens": 128, "method": "tools.parity_cold.hf_answer_ppl",
            },
            "decode_tokens": 24,
            "shape": {
                "linear_params": params, "linear_blocks": blocks,
                "embed_params": 136_134_656,
                "embed_trit_bytes": s.table_bytes(136_134_656, "trit"),
            },
            "timings": {"total_seconds": 300.0},
            "peak_rss_mb": 2400.0,
            "variants": {
                "bf16": {
                    "name": "bf16", "title": "bf16 (эталон)", "fraction": 0.0,
                    "criterion": "", "k": 0, "fraction_actual": 0.0,
                    "ppl": 16.13, "tokens": 2047, "nll_sum": 0.0,
                    "quant": bf16, "size": None, "ratio_vs_bf16": 1.0,
                },
                "trit1": {
                    "name": "trit1", "title": "1 трит (база, 0 %)", "fraction": 0.0,
                    "criterion": "", "k": 0, "fraction_actual": 0.0,
                    "ppl": 1.1e7, "tokens": 2047, "nll_sum": 0.0,
                    "quant": trit1, "size": base, "ratio_vs_bf16": 6.8e5,
                },
                "residual_5": {
                    "name": "sparse", "title": "residual 5 %", "fraction": 0.05,
                    "criterion": "residual", "k": 6, "fraction_actual": 6 / 128,
                    "ppl": 6.2e6, "tokens": 2047, "nll_sum": 0.0,
                    "quant": five, "size": sparse, "ratio_vs_bf16": 3.8e5,
                },
            },
            "best_variant": "residual_5",
            "base_size": base,
            "answers": {"residual_5": [
                {"prompt": "Столица Франции — это", "text": "???", "tokens": 3},
            ]},
            "decode_prompts": list(s.DECODE_PROMPTS),
            "three_b": s.three_b_report(
                s.FALLBACK_3B_CONFIG, (0.05,), (s.CRITERION_RESIDUAL,)
            ),
        }

    def test_report_has_the_required_table(self):
        text = s.format_report(self._synthetic())
        self.assertIn("0149", text)
        self.assertIn("16.13", text)          # эталон bf16
        self.assertIn("residual 5 %", text)
        self.assertIn("1 трит (база, 0 %)", text)
        self.assertIn("Linear МБ", text)
        self.assertIn("индекс МБ", text)
        self.assertIn("снято err", text)
        self.assertIn("???", text)            # декод напечатан
        self.assertNotIn("nan", text.lower())

    def test_report_has_verdict(self):
        text = s.format_report(self._synthetic())
        self.assertIn("=== вывод ===", text)
        self.assertIn("выигрыш", text)

    def test_report_survives_json_roundtrip(self):
        payload = self._synthetic()
        text = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        restored = json.loads(text)
        self.assertEqual(restored["variants"]["residual_5"]["ppl"], 6.2e6)
        s.format_report(restored)

    def test_verdict_without_sparse_variants(self):
        payload = self._synthetic()
        payload["variants"] = {"bf16": payload["variants"]["bf16"]}
        payload["best_variant"] = None
        lines = s.verdict_lines(payload)
        self.assertTrue(any("конечного PPL" in line for line in lines))

    def test_format_report_without_decode(self):
        payload = self._synthetic()
        payload["answers"] = {}
        text = s.format_report(payload)
        self.assertIn("декод не запускался", text)


class TestCli(unittest.TestCase):
    def test_dry_run_without_torch(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = s.main(["--dry-run"])
        self.assertEqual(code, 0)
        text = buffer.getvalue()
        self.assertIn("sparse_trit_eval: dry-run", text)
        self.assertIn("индекс:", text)
        self.assertIn("два трита", text)

    def test_dry_run_bad_arguments(self):
        for argv in (["--dry-run", "--criteria", "random"],
                     ["--dry-run", "--fractions", "0"]):
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(s.main(argv), 2)

    def test_dry_run_custom_fractions(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = s.main(["--dry-run", "--fractions", "0.05,0.10",
                           "--criteria", "residual"])
        self.assertEqual(code, 0)
        text = buffer.getvalue()
        self.assertIn("residual 5 %", text)
        self.assertIn("residual 10 %", text)
        self.assertNotIn("magnitude", text)

    def test_default_model_is_cached_qwen(self):
        model = s.default_model()
        if s._cached_snapshot("Qwen/Qwen2.5-0.5B-Instruct") is None:
            self.assertEqual(model, "Qwen/Qwen2.5-0.5B-Instruct")
        else:
            self.assertIn("Qwen2.5-0.5B", model)
            self.assertTrue(Path(model, "config.json").is_file())

    def test_three_b_config_reads_cache(self):
        config = s.three_b_config()
        if s._cached_snapshot("Qwen/Qwen2.5-3B-Instruct") is None:
            self.assertEqual(config, s.FALLBACK_3B_CONFIG)
        else:
            self.assertEqual(int(config["hidden_size"]), 2048)
            self.assertEqual(int(config["num_hidden_layers"]), 36)


@unittest.skipUnless(HAS_TORCH, "torch не установлен (.pylibs) — torch-путь скипнут")
class TestTorchPath(unittest.TestCase):
    def test_matrix_matches_pure_rows(self):
        """Torch-путь обязан совпадать с чистой функцией (ширина кратна 128)."""
        torch.manual_seed(0)
        weight = torch.randn(8, 256) * 0.02
        for criterion in (s.CRITERION_RESIDUAL, s.CRITERION_MAGNITUDE):
            deq, stats = s.sparse_trit_weight(weight, 0.10, criterion)
            self.assertEqual(tuple(deq.shape), tuple(weight.shape))
            self.assertEqual(stats.blocks, 16)
            self.assertEqual(stats.selected, 16 * 13)
            flat = weight.reshape(-1).tolist()
            pure = s.sparse_block_decompose(flat, fraction=0.10, criterion=criterion)
            error_pure = s.relative_error(flat, s.reconstruct_sparse(pure))
            self.assertAlmostEqual(error_pure, stats.relative_error, places=5)

    def test_selected_everywhere_is_the_dual_trit(self):
        """Доля 100 % — это два трита у всех, индексы не нужны."""
        torch.manual_seed(1)
        weight = torch.randn(4, 128) * 0.02
        deq_full, stats = s.sparse_trit_weight(weight, 1.0, s.CRITERION_RESIDUAL)
        self.assertEqual(stats.selected, 4 * 128)
        deq_base, base = s.sparse_trit_weight(weight, 0.0, s.CRITERION_RESIDUAL)
        self.assertEqual(base.selected, 0)
        error_full = s.relative_error(weight.reshape(-1).tolist(),
                                      deq_full.reshape(-1).tolist())
        error_base = s.relative_error(weight.reshape(-1).tolist(),
                                     deq_base.reshape(-1).tolist())
        self.assertLess(error_full, error_base)

    def test_zero_matrix(self):
        weight = torch.zeros(3, 256)
        deq, stats = s.sparse_trit_weight(weight, 0.20, s.CRITERION_RESIDUAL)
        self.assertEqual(float(deq.abs().sum()), 0.0)
        self.assertEqual(stats.relative_error, 0.0)
        self.assertEqual(stats.residual_energy, 0.0)

    def test_residual_beats_magnitude_in_torch(self):
        torch.manual_seed(2)
        weight = torch.randn(64, 256) * 0.02
        _r, residual = s.sparse_trit_weight(weight, 0.10, s.CRITERION_RESIDUAL)
        _m, magnitude = s.sparse_trit_weight(weight, 0.10, s.CRITERION_MAGNITUDE)
        self.assertLess(residual.relative_error, magnitude.relative_error)
        self.assertGreater(residual.fixed_energy, magnitude.fixed_energy)

    def test_quantize_model_counts_and_sizes(self):
        """``quantize_model``: число выбранных = блоки·k, размер собран из матриц."""
        torch.manual_seed(3)

        class _Attention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = torch.nn.Linear(128, 128, bias=False)
                self.k_proj = torch.nn.Linear(128, 128, bias=False)

        class _Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = _Attention()
                self.mlp = torch.nn.Module()
                self.mlp.down_proj = torch.nn.Linear(256, 128, bias=False)

        class _Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([_Layer(), _Layer()])
                self.lm_head = torch.nn.Linear(128, 256, bias=False)
                self.embed_tokens = torch.nn.Embedding(256, 128)

        model = _Model()
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.data.normal_(0.0, 0.02)
        variant = s.Variant("sparse", 0.05, s.CRITERION_RESIDUAL)
        quant = s.quantize_model(model, variant)
        params = sum(module.weight.numel() for name, module in model.named_modules()
                     if isinstance(module, torch.nn.Linear) and s.is_linear_name(name))
        self.assertEqual(quant["params"], int(params))
        self.assertEqual(quant["k"], 6)
        self.assertEqual(quant["selected"], quant["blocks"] * 6)
        self.assertAlmostEqual(quant["fraction_actual"], 6 / 128, places=12)
        size = quant["size"]
        self.assertEqual(size["params"], int(params))
        self.assertEqual(
            size["total_bytes"],
            size["base_trit_bytes"] + size["base_scale_bytes"]
            + size["second_trit_bytes"] + size["second_scale_bytes"]
            + size["index_bytes"],
        )
        self.assertGreater(quant["relative_error"], 0.0)
        self.assertGreater(quant["fixed_energy"], 0.0)
        self.assertLess(quant["fixed_energy"], quant["residual_energy"])

        # bf16 ничего не квантует и размера не имеет.
        empty = s.quantize_model(model, s.Variant("bf16"))
        self.assertEqual(empty["modules"], 0)
        self.assertIsNone(empty["size"])

    def test_quantize_model_base_variant(self):
        torch.manual_seed(4)

        class _Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = torch.nn.Linear(128, 128, bias=False)

        model = _Model()
        base = s.quantize_model(model, s.Variant("trit1"))
        self.assertEqual(base["k"], 0)
        self.assertEqual(base["selected"], 0)
        self.assertEqual(base["size"]["second_trit_bytes"], 0)
        self.assertEqual(base["size"]["index_bytes"], 0)
        self.assertEqual(base["size_bitmap"]["index_bytes"], 0)
        # 1,725 бита — предел для длинного ряда; на 128 весах хвост
        # упаковки добавляет пару сотых.
        self.assertAlmostEqual(base["size"]["bits_per_param"], 1.725, delta=0.02)


if __name__ == "__main__":
    unittest.main()
