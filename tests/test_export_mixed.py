"""Смешанная точность на экспорте (задача 0146).

Что проверяется без torch:

* политика: разбор ``embed/attn/ffn/first_last/layers.N``, приоритет
  «слой > крайние слои > группа», отказы (неизвестный ключ, дубль, fp16 для
  tied-таблицы);
* раскладка: attn ``fp16`` уходит плотными записями ``float16`` в
  ``non_ternary.bin``, ffn ``trit`` — сегментами TRT1, таблица ``int8`` —
  INT8-сегментом; нормы остаются fp32;
* плотный fp16 читается обратно (точность fp16, не fp32);
* весь ``all-trit`` экспорт **побайтово** совпадает с ``tools/cold_export.py``
  (формат не раздвоился);
* смешанный манифест собирается компилятором (узкий патч
  ``allow_dense_linear_weights``) и исполняется: в контейнере одновременно
  живут trits/fp16/fp32/int8;
* отчёт: таблица «конфигурация → размер → PPL → ×bf16».

Запуск:
    python3 -m unittest tests.test_export_mixed -v
"""
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

from qat import ternary  # noqa: E402
from tools import cold_export, export_mixed, vtm_compile, vtm_ref  # noqa: E402
from tests.test_cold_export import (  # noqa: E402
    FakeTensor,
    tiny_hf_config,
    tiny_model,
)


def export_tiny(directory, policy_text, *, tie=True, force=True, **kwargs):
    policy = export_mixed.Policy.parse(policy_text)
    source = cold_export.DictSource(tiny_model(tie=tie), label="tiny")
    result = export_mixed.export_mixed_model(
        source, tiny_hf_config(tie=tie), directory,
        policy=policy, model_name="tiny/qwen", force=force, **kwargs,
    )
    return result, policy


# ---------------------------------------------------------------------------
# Политика
# ---------------------------------------------------------------------------


class TestPolicy(unittest.TestCase):
    def test_defaults_are_all_trit(self):
        policy = export_mixed.Policy.parse("")
        self.assertEqual(
            (policy.embed, policy.attn, policy.ffn), ("trit", "trit", "trit")
        )
        self.assertIsNone(policy.first_last)
        self.assertEqual(policy.layers, {})

    def test_parses_groups_and_aliases(self):
        policy = export_mixed.Policy.parse(
            "embed=int8,attention=half,mlp=ternary,first_last=float32,layers.3=fp16"
        )
        self.assertEqual(policy.embed, "int8")
        self.assertEqual(policy.attn, "fp16")
        self.assertEqual(policy.ffn, "trit")
        self.assertEqual(policy.first_last, "fp32")
        self.assertEqual(policy.layers, {3: "fp16"})

    def test_precedence_layer_over_edges_over_group(self):
        policy = export_mixed.Policy.parse(
            "attn=fp16,ffn=trit,first_last=fp32,layers.0=trit"
        )
        # Слой 0 перекрыт явно, последний берёт first_last, средний — группу.
        self.assertEqual(policy.dtype_for_matrix("q_proj", 0, 4), "trit")
        self.assertEqual(policy.dtype_for_matrix("q_proj", 3, 4), "fp32")
        self.assertEqual(policy.dtype_for_matrix("q_proj", 2, 4), "fp16")
        self.assertEqual(policy.dtype_for_matrix("gate_proj", 2, 4), "trit")
        # Таблица не подчиняется слоям вообще.
        self.assertEqual(
            policy.dtype_for_layer(0, export_mixed.GROUP_EMBED, 4), "trit"
        )

    def test_single_layer_model_edges(self):
        """При одном слое «первый» и «последний» — один и тот же слой."""
        policy = export_mixed.Policy.parse("attn=trit,first_last=fp16")
        self.assertEqual(policy.dtype_for_matrix("q_proj", 0, 1), "fp16")

    def test_rejects_unknown_key(self):
        with self.assertRaises(export_mixed.MixedExportError) as ctx:
            export_mixed.Policy.parse("atn=fp16")
        self.assertIn("неизвестный ключ", str(ctx.exception))

    def test_rejects_duplicates(self):
        for text in ("attn=fp16,attention=trit", "layers.2=trit,layers.2=fp16"):
            with self.assertRaises(export_mixed.MixedExportError) as ctx:
                export_mixed.Policy.parse(text)
            self.assertIn("дважды", str(ctx.exception))

    def test_rejects_unknown_dtype(self):
        with self.assertRaises(export_mixed.MixedExportError):
            export_mixed.Policy.parse("attn=int4")

    def test_rejects_int8_for_linear(self):
        """INT8 для Linear рантайм не умеет — политика обязана отказать."""
        with self.assertRaises(export_mixed.MixedExportError) as ctx:
            export_mixed.Policy.parse("attn=int8")
        self.assertIn("int8", str(ctx.exception))

    def test_rejects_fp16_for_table_with_hint(self):
        with self.assertRaises(export_mixed.MixedExportError) as ctx:
            export_mixed.Policy.parse("embed=fp16")
        message = str(ctx.exception)
        self.assertIn("tie_word_embeddings", message)
        self.assertIn("int8", message)

    def test_missing_equals_rejected(self):
        with self.assertRaises(export_mixed.MixedExportError):
            export_mixed.Policy.parse("attn")

    def test_to_text_is_canonical(self):
        policy = export_mixed.Policy.parse("ffn=ternary,embed=int8,attn=half")
        self.assertEqual(policy.to_text(), "embed=int8,attn=fp16,ffn=trit")
        policy = export_mixed.Policy.parse("attn=trit,layers.1=fp16,first_last=fp32")
        self.assertEqual(
            policy.to_text(), "embed=trit,attn=trit,ffn=trit,first_last=fp32,"
            "layers.1=fp16"
        )

    def test_suite_is_consistent(self):
        self.assertGreaterEqual(len(export_mixed.SUITE), 4)
        names = [name for name, _ in export_mixed.SUITE]
        self.assertEqual(len(names), len(set(names)))
        for name, text in export_mixed.SUITE:
            with self.subTest(config=name):
                policy = export_mixed.Policy.parse(text)
                self.assertEqual(policy.to_text(), text)


# ---------------------------------------------------------------------------
# Плотные веса
# ---------------------------------------------------------------------------


class TestDensePayload(unittest.TestCase):
    def test_fp32_round_trip_is_exact(self):
        tensor = FakeTensor((2, 4), [0.1, -0.25, 3.0, 0.0, 1.5, -2.0, 0.5, 0.75])
        payload = export_mixed.encode_dense_payload(tensor, "fp32")
        self.assertEqual(len(payload), 8 * 4)
        values = struct.unpack("<8f", payload)
        for got, want in zip(values, list(tensor)):
            self.assertAlmostEqual(got, want, places=6)

    def test_fp16_is_half_precision_not_fp32(self):
        tensor = FakeTensor((4,), [0.1, -0.3, 0.7, 1.0 / 3.0])
        payload = export_mixed.encode_dense_payload(tensor, "fp16")
        self.assertEqual(len(payload), 4 * 2)
        values = struct.unpack("<4e", payload)
        for got, want in zip(values, list(tensor)):
            self.assertAlmostEqual(got, want, places=3)
            self.assertNotEqual(got, want)  # округление действительно было

    def test_fp16_rejects_out_of_range(self):
        tensor = FakeTensor((2,), [1.0, 70000.0])
        with self.assertRaises(export_mixed.MixedExportError):
            export_mixed.encode_dense_payload(tensor, "fp16")


# ---------------------------------------------------------------------------
# Экспорт и раскладка
# ---------------------------------------------------------------------------


class TestExportLayout(unittest.TestCase):
    def test_attn_fp16_ffn_trit_table_int8(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, _policy = export_tiny(tmp, "embed=int8,attn=fp16,ffn=trit")
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["format"], vtm_compile.MANIFEST_FORMAT)
            self.assertEqual(manifest["format_version"], vtm_compile.MANIFEST_VERSION)

            # Таблица словаря — INT8-сегмент в non_ternary.bin.
            embed = next(
                entry for entry in manifest["non_ternary"]["tensors"]
                if entry["role"] == "embed"
            )
            self.assertEqual(embed["dtype"], "int8")
            self.assertEqual(
                manifest["architecture"]["embed"], "int8"
            )
            # tied lm_head ссылается на таблицу, байт не дублируется.
            head = next(
                entry for entry in manifest["non_ternary"]["tensors"]
                if entry["role"] == "lm_head"
            )
            self.assertTrue(head.get("tied_to"))

            # attn — плотные fp16, ffn — триты.
            dense = {
                entry["name"]: entry for entry in manifest["non_ternary"]["tensors"]
                if entry.get("mixed")
            }
            self.assertEqual(len(dense), 2 * len(export_mixed.ATTENTION_KEYS))
            self.assertTrue(all(entry["dtype"] == "float16" for entry in dense.values()))
            self.assertTrue(all(entry["group"] == "attn" for entry in dense.values()))
            trit_names = [entry["name"] for entry in manifest["tensors"]]
            self.assertEqual(len(trit_names), 2 * len(export_mixed.FFN_KEYS))
            self.assertTrue(all(entry["group"] == "ffn"
                                for entry in manifest["tensors"]))

            # Нормы и bias — по-прежнему fp32.
            for entry in manifest["non_ternary"]["tensors"]:
                if entry["role"] in ("norm", "final_norm", "bias"):
                    self.assertEqual(entry["dtype"], "float32")

            # Групповой учёт: байты attn = params × 2.
            groups = manifest["mixed_precision"]["groups"]
            self.assertEqual(groups["attn"]["bytes"], groups["attn"]["params"] * 2)
            self.assertGreater(groups["ffn"]["params"], groups["attn"]["params"])
            self.assertEqual(
                manifest["mixed_precision"]["policy_text"],
                "embed=int8,attn=fp16,ffn=trit",
            )

    def test_first_last_applies_only_to_edges(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, _policy = export_tiny(
                tmp, "embed=trit,attn=trit,ffn=trit,first_last=fp16"
            )
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            dense_layers = {
                entry["layer"] for entry in manifest["non_ternary"]["tensors"]
                if entry.get("mixed")
            }
            self.assertEqual(dense_layers, {0, 1})  # в крошечной модели 2 слоя
            trit_layers = {
                entry["layer"] for entry in manifest["tensors"]
                if entry["layer"] is not None
            }
            self.assertEqual(trit_layers, set())

    def test_layers_override_edges(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, _policy = export_tiny(
                tmp, "attn=trit,ffn=trit,first_last=fp16,layers.0=fp32"
            )
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            kinds = {
                (entry["layer"], entry["key"]): entry["dtype"]
                for entry in manifest["non_ternary"]["tensors"]
                if entry.get("mixed")
            }
            self.assertEqual(kinds[(0, "q_proj")], "float32")
            self.assertEqual(kinds[(1, "q_proj")], "float16")
            self.assertEqual(kinds[(1, "down_proj")], "float16")

    def test_all_trit_matches_cold_export_byte_for_byte(self):
        """Формат не раздвоился: all-trit политика = cold_export --embed trit."""
        with tempfile.TemporaryDirectory() as mixed_dir, \
                tempfile.TemporaryDirectory() as cold_dir:
            mixed, _policy = export_tiny(mixed_dir, "embed=trit,attn=trit,ffn=trit")
            cold = cold_export.export_model(
                cold_export.DictSource(tiny_model(tie=True)), tiny_hf_config(),
                cold_dir, embed_mode="trit", force=True,
            )
            for left, right in (
                (mixed.trit_path, cold.trit_path),
                (mixed.non_ternary_path, cold.non_ternary_path),
            ):
                self.assertEqual(
                    left.read_bytes(), right.read_bytes(),
                    f"{left.name}: смешанный экспорт разошёлся с cold_export",
                )

    def test_embed_fp32_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, _policy = export_tiny(tmp, "embed=fp32,attn=fp16,ffn=trit")
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["architecture"]["embed"], "none")
            embed = next(
                entry for entry in manifest["non_ternary"]["tensors"]
                if entry["role"] == "embed"
            )
            self.assertEqual(embed["dtype"], "float32")
            payload = result.non_ternary_path.read_bytes()[
                embed["offset"]:embed["offset"] + embed["length"]
            ]
            self.assertEqual(len(payload), embed["params"] * 4)

    def test_existing_export_requires_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            export_tiny(tmp, "attn=fp16")
            with self.assertRaises(export_mixed.MixedExportError) as ctx:
                export_tiny(tmp, "attn=fp16", force=False)
            self.assertIn("--force", str(ctx.exception))

    def test_manifest_records_supported_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, _policy = export_tiny(tmp, "attn=fp16")
            mixed = result.manifest["mixed_precision"]
            self.assertEqual(mixed["supported"]["linear"],
                             list(export_mixed.MATRIX_DTYPES))
            self.assertIn("op_tmv", mixed["supported"]["why_no_linear_int8"])
            self.assertIn("MM", mixed["supported"]["why_embed_no_fp16"])
            self.assertIn("compiler_shim", mixed)


# ---------------------------------------------------------------------------
# Компиляция и исполнение
# ---------------------------------------------------------------------------


class TestCompileAndRun(unittest.TestCase):
    def test_mixed_container_holds_all_kinds(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, _policy = export_tiny(tmp, "embed=int8,attn=fp16,ffn=trit")
            container = export_mixed.compile_mixed(
                result.manifest_path, os.path.join(tmp, "model.vtm")
            )
            kinds = container["kinds"]
            self.assertGreaterEqual(kinds.get("fp16", 0), 2)
            self.assertGreaterEqual(kinds.get("trits", 0), 2)
            self.assertEqual(kinds.get("int8"), 1)
            self.assertGreater(container["size_bytes"], 0)
            self.assertEqual(
                container["size_bytes"],
                Path(container["path"]).stat().st_size,
            )

    def test_mixed_container_executes(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, _policy = export_tiny(tmp, "embed=int8,attn=fp16,ffn=trit")
            path = os.path.join(tmp, "model.vtm")
            export_mixed.compile_mixed(result.manifest_path, path)
            loaded = vtm_ref.read_vtm(path)
            out = vtm_ref.execute(loaded, [1, 2, 3, 4], 0)
            self.assertEqual(len(out.logits), 128)
            self.assertTrue(all(math.isfinite(value) for value in out.logits))
            self.assertGreater(max(out.logits) - min(out.logits), 1e-6)

    def test_dense_and_trit_paths_differ(self):
        """fp16-матрица читается как плотная, а не как триты (иначе — баг)."""
        with tempfile.TemporaryDirectory() as tmp:
            result, _policy = export_tiny(tmp, "attn=fp16,ffn=trit")
            path = os.path.join(tmp, "model.vtm")
            export_mixed.compile_mixed(result.manifest_path, path)
            loaded = vtm_ref.read_vtm(path)
            kinds = {
                tensor.name: tensor.kind_name
                for tensor in loaded.tensors
            }
            self.assertEqual(kinds["l0.q_proj.w"], "fp16")
            self.assertEqual(kinds["l0.gate_proj.w"], "trits")

    def test_shim_restores_compiler(self):
        original = vtm_compile.matrix_tensor
        with export_mixed.allow_dense_linear_weights():
            self.assertIsNot(vtm_compile.matrix_tensor, original)
        self.assertIs(vtm_compile.matrix_tensor, original)

    def test_compiler_still_rejects_dense_without_shim(self):
        """Без патча vtm_compile плотную матрицу Linear не примет — это и есть
        причина патча; проверяем, что мы его действительно включаем."""
        with tempfile.TemporaryDirectory() as tmp:
            result, _policy = export_tiny(tmp, "attn=fp16,ffn=trit")
            export = vtm_compile.load_export(result.manifest_path)
            with self.assertRaises(ValueError) as ctx:
                vtm_compile.compile_model(export.cfg, export.weights, export.meta)
            self.assertIn("триты", str(ctx.exception))

    def test_untied_lm_head_dense(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, _policy = export_tiny(
                tmp, "embed=int8,attn=fp16,ffn=trit", tie=False
            )
            head = next(
                entry for entry in result.manifest["non_ternary"]["tensors"]
                if entry["role"] == "lm_head"
            )
            self.assertEqual(head["dtype"], "float32")
            self.assertNotIn("tied_to", head)
            path = os.path.join(tmp, "model.vtm")
            export_mixed.compile_mixed(result.manifest_path, path)
            out = vtm_ref.execute(vtm_ref.read_vtm(path), [1, 2], 0)
            self.assertTrue(all(math.isfinite(value) for value in out.logits))


# ---------------------------------------------------------------------------
# Отчёт
# ---------------------------------------------------------------------------


class TestReport(unittest.TestCase):
    def _record(self, name, ppl, size, answer="ответ"):
        return {
            "name": name,
            "policy_text": "embed=int8,attn=fp16,ffn=trit",
            "container": {"size_bytes": size},
            "ppl": ppl,
            "bf16_ppl": 16.13,
            "answers": [{"answer": answer}],
        }

    def test_table_has_ratio_and_size(self):
        text = export_mixed.format_table(
            [self._record("a", 16.2, 200 * 2**20)],
            bf16_ppl=16.13,
        )
        self.assertIn("| a |", text)
        self.assertIn("200 МиБ", text)
        self.assertIn("×1", text)
        self.assertIn("ответ", text)

    def test_missing_measurements_are_dashes(self):
        text = export_mixed.format_table(
            [{"name": "b", "policy_text": "attn=trit", "container": {}, "answers": []}]
        )
        self.assertIn("| — | — |", text)

    def test_ratio_helper(self):
        self.assertAlmostEqual(export_mixed.ratio_to_bf16(32.26, 16.13), 2.0)
        self.assertIsNone(export_mixed.ratio_to_bf16(None, 16.13))
        self.assertIsNone(export_mixed.ratio_to_bf16(1.0, None))


class TestCli(unittest.TestCase):
    def test_parser_suite_and_policy(self):
        parser = export_mixed.build_parser()
        args = parser.parse_args(["--model", "m", "--suite"])
        self.assertTrue(args.suite)
        args = parser.parse_args(
            ["--model", "m", "--name", "x", "--policy", "attn=fp16"]
        )
        self.assertEqual(args.name, "x")
        self.assertEqual(export_mixed.Policy.parse(args.policy).attn, "fp16")

    def test_policy_without_name_is_error(self):
        # --policy без --name не должен доходить до экспорта
        code = export_mixed.main(["--model", "m", "--policy", "attn=fp16"])
        self.assertEqual(code, 2)

    def test_missing_mode_is_error(self):
        code = export_mixed.main(["--model", "m"])
        self.assertEqual(code, 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
