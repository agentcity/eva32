"""Умное округление при холодном экспорте (задача 0147).

Что проверяется **без** torch (этот файл не требует GPU-стека):

* сегмент TRT1 из готовых тритов/масштабов побайтово совпадает с
  ``tools/cold_export.py`` (формат не «раздвоился») и читается обратно
  ``qat.ternary.decode_segment``;
* отказы: число масштабов не то, длина тритов не кратна блоку, чужой
  ``scale_dtype``;
* математика AWQ без тензоров: ``s_j = a^α/w^(1−α)``, нормировка по
  геометрическому среднему, зажим, крайние ``alpha``, карта каналов GQA для
  ``o_proj`` и сведение поканальных масштабов к группам;
* манифест: обновляются sha256/``integrity.files``/``sparsity``/
  ``zero_fraction``, смещения и архитектура не меняются, блок ``smart``
  появляется; неизвестный файл в манифесте — отказ;
* сквозной путь без torch: крошечный экспорт ``cold_export`` -> подмена
  сегментов ``apply_updates`` -> ``vtm_compile`` -> исполнение ``vtm_ref``
  (то есть контейнер после умного округления остаётся валидным).

С torch дополнительно:

* GPTQ уменьшает ошибку выхода слоя ``‖XW − XW_t‖²`` против наивного
  округления на синтетике с коррелированными входами;
* масштабы блока в GPTQ — ``absmax`` исходного веса, триты в ``{-1,0,1}``,
  число элементов то же;
* свёртка AWQ не меняет выход слоя (q/k/v/gate/up/down и attention для
  ``o_proj``);
* кодировщик на torch-тензорах совпадает с ``cold_export``.

Запуск:
    python3 -m unittest tests.test_export_smart -v
    PYTHONPATH=.pylibs python3.13 -m unittest tests.test_export_smart -v
"""
import hashlib
import importlib.util
import json
import math
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from qat import export as qat_export  # noqa: E402
from qat import ternary  # noqa: E402
from tests.test_cold_export import (  # noqa: E402
    FakeTensor,
    _floats,
    export_tiny,
    tiny_model,
)
from tools import cold_export, export_smart, vtm_compile, vtm_ref  # noqa: E402


def has_torch() -> bool:
    """Есть ли рабочий torch.

    `find_spec` недостаточно: после неудачного импорта torch остаётся в
    sys.modules и find_spec его находит, а повторный импорт падает навсегда.
    Поэтому проверяем, что модуль живой (есть __version__), и чистим мусор.
    """
    import sys

    if "torch" in sys.modules:
        return getattr(sys.modules["torch"], "__version__", None) is not None
    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):
        return False


HAS_TORCH = has_torch()
requires_torch = unittest.skipUnless(HAS_TORCH, "нужен torch (PYTHONPATH=.pylibs)")


# ---------------------------------------------------------------------------
# Сегмент TRT1 из готовых тритов/масштабов
# ---------------------------------------------------------------------------


class TestSegmentExplicit(unittest.TestCase):
    def test_matches_cold_export_byte_for_byte(self):
        """Триты/масштабы наивного правила -> тот же сегмент, что у cold_export."""
        rows, cols = 5, 256
        values = _floats(rows * cols, 3)
        reference, reference_meta = cold_export.encode_trit_segment(
            FakeTensor((rows, cols), values, "w"), "w"
        )
        trits, scales = ternary.quantize(values, ternary.BLOCK_SIZE)
        data, meta = export_smart.encode_trit_segment_explicit(trits, scales)
        self.assertEqual(data, reference)
        for field in ("params", "n_trits", "n_scale_blocks", "n_words"):
            self.assertEqual(meta[field], reference_meta[field], field)
        self.assertEqual(meta["zero_fraction"], reference_meta["zero_fraction"])

    def test_roundtrip_decode(self):
        trits = ([0, 1, -1] * 86)[:256]
        scales = [1.0, 2.0]
        data, meta = export_smart.encode_trit_segment_explicit(trits, scales)
        decoded_trits, decoded_scales = ternary.decode_segment(data)
        self.assertEqual(decoded_trits, trits)
        self.assertEqual(decoded_scales, [1.0, 2.0])
        self.assertEqual(meta["n_scale_blocks"], 2)
        self.assertEqual(meta["n_trits"], 256)

    def test_float16_scales_flag(self):
        trits = [1, 0, -1, 1] * 32
        data, meta = export_smart.encode_trit_segment_explicit(
            trits, [0.5] * 1, scale_dtype="float16"
        )
        self.assertEqual(data[5], 0)  # байт флагов: fp32 не выставлен
        decoded_trits, decoded_scales = ternary.decode_segment(data)
        self.assertEqual(decoded_trits, trits)
        self.assertAlmostEqual(decoded_scales[0], 0.5, places=6)
        self.assertEqual(meta["n_scale_blocks"], 1)

    def test_rejects_wrong_scale_count(self):
        trits = [0] * 256
        with self.assertRaises(export_smart.SmartExportError):
            export_smart.encode_trit_segment_explicit(trits, [1.0])

    def test_rejects_non_multiple_trits(self):
        with self.assertRaises(export_smart.SmartExportError):
            export_smart.encode_trit_segment_explicit([0] * 200, [1.0])

    def test_rejects_unknown_scale_dtype(self):
        with self.assertRaises(export_smart.SmartExportError):
            export_smart.encode_trit_segment_explicit([0] * 128, [1.0],
                                                     scale_dtype="int4")

    def test_rejects_trit_out_of_range(self):
        with self.assertRaises(Exception):
            export_smart.encode_trit_segment_explicit([2] * 128, [1.0])


# ---------------------------------------------------------------------------
# AWQ: масштабы, группы GQA
# ---------------------------------------------------------------------------


class TestAwqMath(unittest.TestCase):
    def test_formula_and_geomean(self):
        act = [1.0, 100.0, 1.0]
        weight = [1.0, 1.0, 100.0]
        scale = export_smart.awq_scale(act, weight, 0.5)
        product = 1.0
        for value in scale:
            product *= value
        # Нормировка по геометрическому среднему: произведение около 1.
        self.assertAlmostEqual(product ** (1.0 / 3.0), 1.0, places=6)
        # Важный канал (большая активация) получает больший масштаб.
        self.assertGreater(scale[1], scale[0])
        self.assertGreater(scale[0], scale[2])

    def test_alpha_endpoints(self):
        act = [2.0, 4.0]
        weight = [0.5, 8.0]
        only_weight = export_smart.awq_scale(act, weight, 0.0)
        only_act = export_smart.awq_scale(act, weight, 1.0)
        # alpha=0: s ~ 1/w (активации не влияют) -> меньший вес, больший масштаб.
        self.assertGreater(only_weight[0], only_weight[1])
        # alpha=1: s ~ a (веса не влияют) -> большая активация, больший масштаб.
        self.assertGreater(only_act[1], only_act[0])

    def test_scales_clamped(self):
        scale = export_smart.awq_scale(
            [1e-12, 1e12], [1.0, 1.0], 1.0
        )
        self.assertGreaterEqual(min(scale), export_smart.AWQ_SCALE_MIN)
        self.assertLessEqual(max(scale), export_smart.AWQ_SCALE_MAX)

    def test_rejects_mismatched_channels(self):
        with self.assertRaises(export_smart.SmartExportError):
            export_smart.awq_scale([1.0, 2.0], [1.0], 0.5)

    def test_rejects_bad_alpha(self):
        with self.assertRaises(export_smart.SmartExportError):
            export_smart.awq_scale([1.0], [1.0], 1.5)

    def test_rejects_empty(self):
        with self.assertRaises(export_smart.SmartExportError):
            export_smart.awq_scale([], [], 0.5)


class TestGqaGroups(unittest.TestCase):
    def test_map_shape_and_groups(self):
        mapping = export_smart.kv_channel_map(
            896, n_heads=14, n_kv_heads=2, head_dim=64
        )
        self.assertEqual(len(mapping), 896)
        # 128 каналов v: у каждого своя группа из 7 голов запроса.
        self.assertEqual(len(set(mapping)), 128)
        from collections import Counter
        self.assertEqual(set(Counter(mapping).values()), {7})
        # head 0 dim 0 и head 6 dim 0 (та же kv-группа) -> один канал v.
        self.assertEqual(mapping[0], mapping[6 * 64])
        # head 7 уже другая kv-группа.
        self.assertNotEqual(mapping[0], mapping[7 * 64])

    def test_rejects_bad_geometry(self):
        with self.assertRaises(export_smart.SmartExportError):
            export_smart.kv_channel_map(10, n_heads=3, n_kv_heads=2, head_dim=64)
        with self.assertRaises(export_smart.SmartExportError):
            export_smart.kv_channel_map(128, n_heads=2, n_kv_heads=1, head_dim=32)

    def test_group_scales_geomean(self):
        mapping = export_smart.kv_channel_map(
            8, n_heads=2, n_kv_heads=1, head_dim=4
        )
        per_channel = [1.0, 1.0, 1.0, 1.0, 4.0, 4.0, 4.0, 4.0]
        groups, channels = export_smart.group_scales(per_channel, mapping)
        self.assertEqual(len(groups), 4)
        self.assertEqual(len(channels), 8)
        for group, value in enumerate(groups):
            self.assertAlmostEqual(
                math.log(value),
                sum(math.log(per_channel[j]) for j, g in enumerate(mapping) if g == group) / 2.0,
                places=9,
            )
        # В каждом канале — масштаб своей группы.
        for j, group in enumerate(mapping):
            self.assertEqual(channels[j], groups[group])


class TestChannelStats(unittest.TestCase):
    def test_mean_abs_rows(self):
        stats = export_smart.mean_abs_rows([[1.0, -2.0], [3.0, 0.0]])
        self.assertEqual(stats, [2.0, 1.0])

    def test_average_stats(self):
        self.assertEqual(
            export_smart.average_stats([[1.0, 2.0], [3.0, 4.0]]), [2.0, 3.0]
        )

    def test_mean_abs_rows_rejects_empty(self):
        with self.assertRaises(export_smart.SmartExportError):
            export_smart.mean_abs_rows([])

    def test_average_stats_rejects_mismatch(self):
        with self.assertRaises(export_smart.SmartExportError):
            export_smart.average_stats([[1.0], [1.0, 2.0]])


class TestArguments(unittest.TestCase):
    def test_round_modes(self):
        for mode in export_smart.ROUND_MODES:
            self.assertEqual(export_smart.check_round_mode(mode), mode)
        with self.assertRaises(export_smart.SmartExportError):
            export_smart.check_round_mode("awq+")

    def test_calib_lines_bounds(self):
        self.assertEqual(export_smart.check_calib_lines(200), 200)
        with self.assertRaises(export_smart.SmartExportError):
            export_smart.check_calib_lines(0)
        with self.assertRaises(export_smart.SmartExportError):
            export_smart.check_calib_lines(export_smart.MAX_CALIB_LINES + 1)

    def test_alphas(self):
        self.assertEqual(export_smart.parse_alphas(None), export_smart.AWQ_ALPHAS)
        self.assertEqual(export_smart.parse_alphas("0,0.5"), (0.0, 0.5))
        with self.assertRaises(export_smart.SmartExportError):
            export_smart.parse_alphas("0.5,2")
        with self.assertRaises(export_smart.SmartExportError):
            export_smart.parse_alphas("")

    def test_parser_defaults(self):
        parser = export_smart.build_parser()
        args = parser.parse_args(["--model", "/tmp/m", "--out", "/tmp/o"])
        self.assertEqual(args.round, "naive")
        self.assertEqual(args.embed, "int8")
        self.assertEqual(args.calib_lines, export_smart.DEFAULT_CALIB_LINES)
        self.assertEqual(args.calib_skip, export_smart.DEFAULT_CALIB_SKIP)


# ---------------------------------------------------------------------------
# Манифест: хеши, разреженность, сохранность смещений
# ---------------------------------------------------------------------------


def _mini_manifest() -> dict:
    return {
        "format": "eva32.ternary.manifest",
        "format_version": 2,
        "files": {
            "ternary.trit": {"bytes": 100, "sha256": "0" * 64},
            "non_ternary.bin": {"bytes": 20, "sha256": "1" * 64},
        },
        "integrity": {
            "algo": "sha256",
            "files": {"ternary.trit": "0" * 64, "non_ternary.bin": "1" * 64},
        },
        "sparsity": {"zero_fraction": 0.0, "warn_threshold": 0.95,
                     "collapsed": False, "trits": 256},
        "quantization": {"block_size": 128, "scale_dtype": "float32",
                         "rule": "наивное"},
        "tensors": [
            {"name": "a", "n_trits": 128, "zero_fraction": 0.5,
             "trit_offset": 0, "trit_length": 100},
            {"name": "b", "n_trits": 128, "zero_fraction": 0.5,
             "trit_offset": 100, "trit_length": 100},
        ],
        "non_ternary": {
            "tensors": [
                {"name": "model.layers.0.input_layernorm.weight",
                 "dtype": "float32", "offset": 0, "length": 20},
            ]
        },
        "architecture": {"hidden": 128, "n_layers": 1},
    }


class TestManifest(unittest.TestCase):
    def test_updates_hashes_and_keeps_offsets(self):
        manifest = _mini_manifest()
        patched = export_smart.build_smart_manifest(
            manifest,
            round_mode="both",
            file_stats={
                "ternary.trit": {"sha256": "a" * 64, "bytes": 101},
                "non_ternary.bin": {"sha256": "b" * 64, "bytes": 21},
            },
            zero_by_tensor={"a": 0.25},
            zero_counts={},
            info={"calibration": {"lines": 200}},
        )
        self.assertEqual(patched["files"]["ternary.trit"]["sha256"], "a" * 64)
        self.assertEqual(patched["files"]["ternary.trit"]["bytes"], 101)
        self.assertEqual(patched["integrity"]["files"]["ternary.trit"], "a" * 64)
        self.assertEqual(patched["tensors"][0]["zero_fraction"], 0.25)
        self.assertEqual(patched["tensors"][1]["zero_fraction"], 0.5)
        for index in (0, 1):
            self.assertEqual(
                patched["tensors"][index]["trit_offset"],
                manifest["tensors"][index]["trit_offset"],
            )
        self.assertEqual(patched["quantization"]["rounding"], "both")
        self.assertEqual(patched["smart"]["round"], "both")
        self.assertEqual(patched["smart"]["calibration"]["lines"], 200)
        # Исходный манифест не испорчен (deepcopy).
        self.assertEqual(manifest["files"]["ternary.trit"]["sha256"], "0" * 64)
        self.assertNotIn("smart", manifest)

    def test_sparsity_recomputed(self):
        manifest = _mini_manifest()
        manifest["sparsity"]["trits"] = 256
        patched = export_smart.build_smart_manifest(
            manifest,
            round_mode="gptq",
            file_stats={},
            zero_by_tensor={"a": 1.0, "b": 1.0},
            zero_counts={},
            info={},
        )
        self.assertEqual(patched["sparsity"]["zero_fraction"], 1.0)
        self.assertTrue(patched["sparsity"]["collapsed"])

    def test_rejects_unknown_file(self):
        with self.assertRaises(export_smart.SmartExportError):
            export_smart.build_smart_manifest(
                _mini_manifest(),
                round_mode="awq",
                file_stats={"ternary.vtm": {"sha256": "a", "bytes": 1}},
                zero_by_tensor={},
                zero_counts={},
                info={},
            )

    def test_rejects_bad_mode(self):
        with self.assertRaises(export_smart.SmartExportError):
            export_smart.build_smart_manifest(
                _mini_manifest(), round_mode="smart",
                file_stats={}, zero_by_tensor={}, zero_counts={}, info={},
            )


# ---------------------------------------------------------------------------
# Сквозной путь без torch: экспорт -> подмена -> компиляция -> исполнение
# ---------------------------------------------------------------------------


class TestApplyUpdatesEndToEnd(unittest.TestCase):
    def _patch_plan(self, manifest, trit_path):
        """Подменить q_proj слоя 0: триты из наивных, но с перевёрнутым знаком."""
        entry = next(
            item for item in manifest["tensors"]
            if item["name"] == "model.layers.0.self_attn.q_proj.weight"
        )
        # Масштабы берём ровно те, что в наивном сегменте, триты — инверсия:
        # так подмена заведомо валидна по формату и заметна по содержимому.
        blob = Path(trit_path).read_bytes()
        segment = blob[entry["trit_offset"]:entry["trit_offset"] + entry["trit_length"]]
        trits, scales = ternary.decode_segment(segment)
        flipped = [-value for value in trits]
        return {
            "name": entry["name"],
            "trits": flipped,
            "scales": scales,
            "zero": sum(1 for value in flipped if value == 0),
            "changed": sum(1 for a, b in zip(trits, flipped) if a != b),
            "n_trits": len(flipped),
        }

    def test_roundtrip_compiles_and_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, _state = export_tiny(tmp)
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            update = self._patch_plan(manifest, result.trit_path)
            written = export_smart.apply_updates(
                tmp, manifest, {update["name"]: update}, {}, echo=lambda _m: None
            )
            # Файлы реально изменились и хеши совпадают с содержимым.
            for name in ("ternary.trit", "non_ternary.bin"):
                actual = hashlib.sha256((Path(tmp) / name).read_bytes()).hexdigest()
                self.assertEqual(written["files"][name]["sha256"], actual)
            new_manifest = export_smart.build_smart_manifest(
                manifest,
                round_mode="gptq",
                file_stats=written["files"],
                zero_by_tensor=written["zero_by_tensor"],
                zero_counts={},
                info={"calibration": {"lines": 2}},
            )
            Path(result.manifest_path).write_text(
                json.dumps(new_manifest, ensure_ascii=False), encoding="utf-8"
            )
            # Сегмент читается обратно в те же (инвертированные) триты.
            blob = result.trit_path.read_bytes()
            target = next(
                item for item in new_manifest["tensors"] if item["name"] == update["name"]
            )
            segment = blob[
                target["trit_offset"]:target["trit_offset"] + target["trit_length"]
            ]
            patched_trits, _scales = ternary.decode_segment(segment)
            self.assertEqual(patched_trits, update["trits"])
            # Контейнер компилируется и исполняется: формат не сломан.
            export = vtm_compile.load_export(result.manifest_path)
            vtm = vtm_compile.compile_model(export.cfg, export.weights, export.meta)
            out = vtm_ref.execute(vtm, [1, 2, 3, 4], 0)
            self.assertIsNotNone(out.logits)
            self.assertTrue(all(math.isfinite(value) for value in out.logits))

    def test_rejects_unknown_tensor(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, _state = export_tiny(tmp)
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            with self.assertRaises(export_smart.SmartExportError):
                export_smart.apply_updates(
                    tmp, manifest,
                    {"model.layers.0.self_attn.q_proj.weight2": {
                        "trits": [0] * 16, "scales": [1.0], "zero": 16,
                        "changed": 0, "n_trits": 16,
                    }},
                    {}, echo=lambda _m: None,
                )

    def test_rejects_wrong_payload_length(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, _state = export_tiny(tmp)
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            entry = next(
                item for item in manifest["tensors"]
                if item["name"] == "model.layers.0.self_attn.q_proj.weight"
            )
            n_trits = int(entry["n_trits"])
            with self.assertRaises(export_smart.SmartExportError):
                export_smart.apply_updates(
                    tmp, manifest,
                    {entry["name"]: {
                        "trits": [0] * n_trits, "scales": [1.0],
                        "zero": n_trits, "changed": 0, "n_trits": n_trits,
                    }},
                    {}, echo=lambda _m: None,
                )

    def test_rejects_wrong_non_ternary_length(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, _state = export_tiny(tmp)
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            with self.assertRaises(export_smart.SmartExportError):
                export_smart.apply_updates(
                    tmp, manifest, {},
                    {"model.layers.0.input_layernorm.weight": b"\x00" * 4},
                    echo=lambda _m: None,
                )


# ---------------------------------------------------------------------------
# С torch: GPTQ, свёртка AWQ, кодировщик
# ---------------------------------------------------------------------------


@requires_torch
class TestGptq(unittest.TestCase):
    def _problem(self, *, rows=64, cols=512, tokens=128, seed=0, outlier=False):
        import torch

        generator = torch.Generator().manual_seed(seed)
        # Входы как у настоящих слоёв: у части каналов дисперсия на порядки
        # больше (выбросы в активациях). outlier=True — один канал-выброс на
        # ровном фоне: ровно тот случай, где OBQ даёт максимум.
        base = torch.randn(tokens, cols, generator=generator)
        if outlier:
            X = base
            X[:, 7] *= 30.0
        else:
            X = base * torch.logspace(-1, 1, cols)
        W = torch.randn(rows, cols, generator=generator) * 0.05
        H = X.transpose(0, 1) @ X
        return torch, X, W, H

    def test_beats_naive_on_layer_output(self):
        torch, X, W, H = self._problem()
        naive = export_smart.quantize_naive(W)
        trits, scales, stats = export_smart.gptq_quantize(W, H)
        self.assertEqual(tuple(trits.shape), tuple(W.shape))
        self.assertEqual(set(trits.reshape(-1).tolist()) - {-1, 0, 1}, set())
        # Масштабы — absmax блоков исходного веса, не пересчитаны.
        expected_scales = export_smart.block_absmax(W, 128)
        self.assertTrue(torch.equal(scales, expected_scales))
        naive_out = X @ export_smart.dequantize(naive["trits"], naive["scales"]).T
        gptq_out = X @ export_smart.dequantize(trits, scales).T
        reference = X @ W.T
        naive_error = float((naive_out - reference).pow(2).sum())
        gptq_error = float((gptq_out - reference).pow(2).sum())
        self.assertGreater(naive_error, 0.0)
        self.assertLess(
            gptq_error, naive_error,
            f"GPTQ не улучшил ошибку выхода: {gptq_error:.3e} против {naive_error:.3e}",
        )
        # Улучшение заметно больше шума: у GPTQ ошибка выхода ниже.
        self.assertGreater(naive_error / gptq_error, 1.05)
        self.assertEqual(stats["rows"], W.shape[0])
        self.assertEqual(stats["cols"], W.shape[1])

    def test_outlier_channel_helps_most(self):
        """Канал с аномально большой активацией — главный выигрыш GPTQ."""
        torch, X, W, H = self._problem(outlier=True)
        naive = export_smart.quantize_naive(W)
        trits, scales, _stats = export_smart.gptq_quantize(W, H)
        reference = X @ W.T
        naive_error = float(
            (X @ export_smart.dequantize(naive["trits"], naive["scales"]).T
             - reference).pow(2).sum()
        )
        gptq_error = float(
            (X @ export_smart.dequantize(trits, scales).T - reference).pow(2).sum()
        )
        self.assertGreater(naive_error / gptq_error, 2.0)

    def test_zero_block_stays_zero(self):
        torch, X, W, H = self._problem()
        W = W.clone()
        W[:, 128:] = 0.0
        trits, scales, _stats = export_smart.gptq_quantize(W, H)
        self.assertEqual(float(scales[:, 1].abs().max()), 0.0)
        self.assertEqual(set(trits[:, 128:].reshape(-1).tolist()), {0})

    def test_rejects_bad_hessian_shape(self):
        torch, X, W, H = self._problem()
        with self.assertRaises(export_smart.SmartExportError):
            export_smart.gptq_quantize(W, H[:64, :64])

    def test_rejects_non_2d(self):
        import torch

        with self.assertRaises(export_smart.SmartExportError):
            export_smart.gptq_quantize(torch.zeros(4), torch.eye(4))


@requires_torch
class TestFoldExact(unittest.TestCase):
    def _tensors(self):
        import torch

        # Как в Qwen2: q_dim = n_heads*head_dim (8), kv_dim = n_kv*head_dim (4).
        hidden, q_dim, kv_dim, inter = 8, 8, 4, 16
        generator = torch.Generator().manual_seed(7)

        def value(*shape):
            return torch.randn(*shape, generator=generator) * 0.2

        weights = {
            "q_proj": value(q_dim, hidden),
            "k_proj": value(kv_dim, hidden),
            "v_proj": value(kv_dim, hidden),
            "o_proj": value(hidden, q_dim),
            "gate_proj": value(inter, hidden),
            "up_proj": value(inter, hidden),
            "down_proj": value(hidden, inter),
        }
        biases = {"v_proj": value(kv_dim)}
        norms = {"norm1": value(hidden), "norm2": value(hidden)}
        return weights, biases, norms

    def test_fold_keeps_linear_outputs(self):
        import torch

        weights, biases, norms = self._tensors()
        hidden, q_dim, kv_dim, inter = 8, 8, 4, 16
        x_attn = torch.randn(4, hidden) * 0.5
        x_mlp = torch.randn(4, hidden) * 0.5
        h_mlp = torch.randn(4, inter) * 0.5
        before = {
            "q": x_attn @ weights["q_proj"].T,
            "k": x_attn @ weights["k_proj"].T,
            "v": x_attn @ weights["v_proj"].T + biases["v_proj"],
            "gate": x_mlp @ weights["gate_proj"].T,
            "up": x_mlp @ weights["up_proj"].T,
            "down": h_mlp @ weights["down_proj"].T,
        }
        scales = {
            "attn_in": torch.tensor([0.5, 2.0, 1.0, 1.5, 0.7, 1.3, 0.9, 1.1]),
            "mlp_in": torch.tensor([1.2, 0.8, 1.0, 0.5, 1.7, 1.0, 0.6, 1.4]),
            "o_in": torch.tensor([1.0, 1.0, 2.0, 2.0, 0.5, 0.5, 1.0, 1.0]),
            "v_out": torch.tensor([1.0, 2.0, 0.5, 1.0])[:kv_dim],
            "down_in": torch.tensor([1.0, 0.5, 2.0, 1.0, 1.0, 0.25, 3.0, 1.0,
                                     1.0, 0.5, 2.0, 1.0, 1.0, 0.25, 3.0, 1.0]),
        }
        export_smart.fold_awq_layer(
            weights, scales=scales, biases=biases, norms=norms
        )
        after = {
            "q": (x_attn * scales["attn_in"]) @ weights["q_proj"].T,
            "k": (x_attn * scales["attn_in"]) @ weights["k_proj"].T,
            "v": (x_attn * scales["attn_in"]) @ weights["v_proj"].T + biases["v_proj"],
            "gate": (x_mlp * scales["mlp_in"]) @ weights["gate_proj"].T,
            # up умножен ещё на s_down (масштаб промежуточных каналов уходит
            # в down_proj): в свёрнутой модели его выход в этом масштабе.
            "up": ((x_mlp * scales["mlp_in"]) @ weights["up_proj"].T)
                  * scales["down_in"],
            "down": (h_mlp * scales["down_in"]) @ weights["down_proj"].T,
        }
        # v теперь в своём (s_v) масштабе: сравниваем после деления.
        after["v"] = after["v"] / scales["v_out"]
        # up сам по себе умножен на s_down (см. ниже), поэтому сверяем его
        # не отдельно, а в произведении silu(gate)·up — как его читает down.
        for key in ("q", "k", "v", "gate", "down"):
            torch.testing.assert_close(after[key], before[key], rtol=1e-5, atol=1e-5)
        # Выход MLP целиком: silu(gate)·up в масштабе s_down -> down его снимает.
        combined = torch.nn.functional.silu(after["gate"]) * after["up"]
        combined_before = torch.nn.functional.silu(before["gate"]) * before["up"]
        torch.testing.assert_close(combined, combined_before * scales["down_in"],
                                   rtol=1e-5, atol=1e-5)

    def test_attention_fold_with_gqa(self):
        """Масштаб v-канала и обратный масштаб столбцов o не меняют выход."""
        import torch

        torch.manual_seed(3)
        n_heads, n_kv, head_dim, tokens = 2, 1, 4, 3
        q_dim = n_heads * head_dim
        mapping = export_smart.kv_channel_map(
            q_dim, n_heads=n_heads, n_kv_heads=n_kv, head_dim=head_dim
        )
        q = torch.randn(1, n_heads, tokens, head_dim)
        k = torch.randn(1, n_kv, tokens, head_dim)
        v = torch.randn(1, n_kv, tokens, head_dim)
        o_weight = torch.randn(q_dim, q_dim) * 0.1

        groups_per_kv = n_heads // n_kv

        def attention(v_value):
            # GQA: k/v повторяются на все головы запроса своей группы.
            k_rep = k.repeat_interleave(groups_per_kv, dim=1)
            v_rep = v_value.repeat_interleave(groups_per_kv, dim=1)
            scores = torch.einsum("bhqd,bhkd->bhqk", q, k_rep) / math.sqrt(head_dim)
            probs = torch.softmax(scores, dim=-1)
            out = torch.einsum("bhqk,bhkd->bhqd", probs, v_rep)
            return out.reshape(1, tokens, q_dim)

        groups, per_channel = export_smart.group_scales(
            [1.0, 2.0, 4.0, 0.5, 1.0, 2.0, 4.0, 0.5], mapping
        )
        v_scaled = v * torch.tensor(groups).reshape(1, n_kv, 1, head_dim)
        out_before = attention(v) @ o_weight.T
        scaled_weight = o_weight / torch.tensor(per_channel)[None, :]
        out_after = attention(v_scaled) @ scaled_weight.T
        torch.testing.assert_close(out_after, out_before, rtol=1e-4, atol=1e-5)


@requires_torch
class TestTorchEncoder(unittest.TestCase):
    def test_matches_cold_export_torch(self):
        import torch

        generator = torch.Generator().manual_seed(11)
        weight = torch.randn(6, 256, generator=generator) * 0.3
        reference, reference_meta = cold_export.encode_trit_segment(weight, "w")
        quant = ternary.quantize_torch(weight, ternary.BLOCK_SIZE)
        data, meta = export_smart.encode_trit_segment_explicit(
            quant["trits"], quant["scales"]
        )
        self.assertEqual(data, reference)
        self.assertEqual(meta["n_scale_blocks"], reference_meta["n_scale_blocks"])
        self.assertEqual(meta["zero_fraction"], reference_meta["zero_fraction"])

    def test_choose_alpha_returns_member_and_scales(self):
        import torch

        generator = torch.Generator().manual_seed(5)
        W = torch.randn(4, 256, generator=generator) * 0.1
        X = torch.randn(32, 256, generator=generator) * torch.logspace(-1, 1, 256)
        alpha, scale, detail = export_smart.choose_alpha(
            (0.0, 0.5, 1.0),
            samples=X,
            tensors=[W],
            act_absmean=X.abs().mean(dim=0).tolist(),
            weight_absmean=W.abs().mean(dim=0).tolist(),
            block_size=128,
        )
        self.assertIn(alpha, (0.0, 0.5, 1.0))
        self.assertEqual(len(scale), 256)
        self.assertEqual(len(detail["tried"]), 3)
        # Выбранный alpha — действительно лучший из перечисленных.
        best = min(detail["tried"], key=lambda item: item["error"])
        self.assertEqual(best["alpha"], alpha)

    def test_dequantize_matches_trit_times_scale(self):
        """dequantize повторяет ровно то, что читает рантайм: trit×scale."""
        import torch

        generator = torch.Generator().manual_seed(2)
        weight = torch.randn(4, 256, generator=generator)
        quant = ternary.quantize_torch(weight, 128)
        restored = export_smart.dequantize(quant["trits"], quant["scales"], 128)
        flat_trits = quant["trits"].reshape(-1).tolist()
        flat_scales = quant["scales"].reshape(-1).tolist()
        expected = ternary.dequantize(flat_trits, flat_scales, 128)
        self.assertEqual(restored.reshape(-1).tolist(), expected)
        # И это не сам вес: округление действительно что-то потеряло.
        self.assertFalse(torch.equal(restored, weight))


if __name__ == "__main__":
    unittest.main()
