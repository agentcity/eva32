"""Тесты tools/parity_real.py без torch: парсер, команды, разбор, сводка.

Запуск:
    python3 -m unittest tests.test_parity_real -v
    python3 -m unittest discover -s tests

Здесь нет ни одного импорта torch/transformers: проверяется ровно то, что
можно проверить на машине без GPU-стека. C-проба проверяется на синтаксис,
если есть cc и src/vtm/vtm.h.
"""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools import parity_real  # noqa: E402


# ---------------------------------------------------------------------------
# Парсер
# ---------------------------------------------------------------------------


class TestParser(unittest.TestCase):
    def test_defaults(self):
        args = parity_real.parse_args([])
        self.assertEqual(args.model, parity_real.DEFAULT_MODEL)
        self.assertEqual(args.prompt, parity_real.DEFAULT_PROMPT)
        self.assertEqual(args.limit_tokens, parity_real.DEFAULT_LIMIT_TOKENS)
        self.assertEqual(args.quantize_embeddings, "none")
        self.assertEqual(args.scale_dtype, "float16")
        self.assertEqual(args.torch_dtype, "float32")
        self.assertEqual(args.device, "auto")
        self.assertFalse(args.dry_run)

    def test_all_requested_flags(self):
        args = parity_real.parse_args([
            "--model", "/models/qwen",
            "--prompt", "hello",
            "--workdir", "/tmp/parity",
            "--limit-tokens", "7",
            "--quantize-embeddings", "trit",
            "--dry-run",
        ])
        self.assertEqual(args.model, "/models/qwen")
        self.assertEqual(args.prompt, "hello")
        self.assertEqual(args.workdir, "/tmp/parity")
        self.assertEqual(args.limit_tokens, 7)
        self.assertEqual(args.quantize_embeddings, "trit")
        self.assertTrue(args.dry_run)

    def test_limit_must_be_positive(self):
        for value in ("0", "-1", "x"):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    parity_real.parse_args(["--limit-tokens", value])
                self.assertEqual(caught.exception.code, 2)

    def test_embedding_mode_choices(self):
        for mode in parity_real.EMBED_MODES:
            self.assertEqual(
                parity_real.parse_args(["--quantize-embeddings", mode])
                .quantize_embeddings,
                mode,
            )
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parity_real.parse_args(["--quantize-embeddings", "fp4"])

    def test_positive_int(self):
        self.assertEqual(parity_real.positive_int("3"), 3)
        with self.assertRaises(Exception):
            parity_real.positive_int("0")


# ---------------------------------------------------------------------------
# Сборка команд
# ---------------------------------------------------------------------------


class TestCommands(unittest.TestCase):
    def test_token_ids_arg(self):
        self.assertEqual(parity_real.token_ids_arg([1, 22, 333]), "1,22,333")
        with self.assertRaises(parity_real.ParityError):
            parity_real.token_ids_arg([])

    def test_export_command(self):
        cmd = parity_real.build_export_command(
            "python3",
            Path("/w/qat_run"),
            Path("/w/export"),
            context_len=8,
            scale_dtype="float16",
        )
        self.assertEqual(cmd[:3], ["python3", "-m", "qat.export"])
        self.assertIn("--run", cmd)
        self.assertEqual(cmd[cmd.index("--run") + 1], "/w/qat_run")
        self.assertEqual(cmd[cmd.index("--out") + 1], "/w/export")
        self.assertEqual(cmd[cmd.index("--step") + 1], "0")
        self.assertEqual(cmd[cmd.index("--context-len") + 1], "8")
        self.assertEqual(cmd[cmd.index("--scale-dtype") + 1], "float16")
        self.assertEqual(cmd[cmd.index("--tie-embeddings") + 1], "auto")
        self.assertIn("--force", cmd)
        # qat.export не принимает этот флаг: режим лежит в run.json.
        self.assertNotIn("--quantize-embeddings", cmd)

    def test_compile_command(self):
        cmd = parity_real.build_compile_command(
            "python3", Path("/w/export/manifest.json"), Path("/w/model.vtm")
        )
        self.assertEqual(cmd[0], "python3")
        self.assertTrue(cmd[1].endswith("tools/vtm_compile.py"))
        self.assertEqual(cmd[cmd.index("--manifest") + 1], "/w/export/manifest.json")
        self.assertEqual(cmd[cmd.index("--out") + 1], "/w/model.vtm")

    def test_run_command(self):
        cmd = parity_real.build_run_command(
            Path("/repo/src/vtm/vtm"), Path("/w/model.vtm"), [5, 6, 7]
        )
        self.assertEqual(cmd[0], "/repo/src/vtm/vtm")
        self.assertEqual(cmd[1], "run")
        self.assertEqual(cmd[2], "/w/model.vtm")
        self.assertEqual(cmd[cmd.index("--tokens") + 1], "5,6,7")
        self.assertEqual(cmd[cmd.index("--max-new") + 1], "1")

    def test_probe_compile_command(self):
        cmd = parity_real.build_probe_compile_command(
            "/usr/bin/cc", Path("/w")
        )
        self.assertEqual(cmd[0], "/usr/bin/cc")
        self.assertIn("-std=c99", cmd)
        self.assertIn("-I", cmd)
        self.assertEqual(cmd[cmd.index("-o") + 1], "/w/parity_probe")
        self.assertIn("/w/parity_probe.c", cmd)
        self.assertTrue(cmd[-1] == "-lm")
        self.assertTrue(any(part.endswith("libvtm.a") for part in cmd))


# ---------------------------------------------------------------------------
# Разбор вывода vtm run / C-пробы
# ---------------------------------------------------------------------------


class TestParseOutput(unittest.TestCase):
    def test_vtm_run_output(self):
        stdout = (
            "модель: model.vtm\n"
            "тензоров: 12, инструкций: 40, бэкенд: scalar-lut243\n"
            "prompt: 1 2 3\n"
            "вывод:  29\n"
            "время: 1.234 мс, 1 токен(ов), 810.0 ток/с\n"
        )
        parsed = parity_real.parse_vtm_run_output(stdout)
        self.assertEqual(parsed["prompt"], [1, 2, 3])
        self.assertEqual(parsed["generated"], [29])
        self.assertEqual(parsed["tensor_count"], 12)
        self.assertEqual(parsed["instr_count"], 40)
        self.assertEqual(parsed["backend"], "scalar-lut243")

    def test_empty_generation(self):
        parsed = parity_real.parse_vtm_run_output("prompt: 4\nвывод:  (пусто)\n")
        self.assertEqual(parsed["prompt"], [4])
        self.assertEqual(parsed["generated"], [])

    def test_bad_token_list(self):
        with self.assertRaises(parity_real.ParityError):
            parity_real.parse_vtm_run_output("prompt: 1 x 3\n")

    def test_probe_output(self):
        stdout = (
            "# meta_present 1\n"
            "# meta_rope_theta 1000000\n"
            "# meta_rope_layout 0\n"
            "# meta_vocab 3\n"
            "# meta_tied 1\n"
            "# count 3\n"
            "-1.5\n"
            "0.25\n"
            "2\n"
        )
        parsed = parity_real.parse_probe_output(stdout)
        self.assertEqual(parsed.logits, [-1.5, 0.25, 2.0])
        self.assertEqual(parsed.meta["meta_rope_theta"], 1000000.0)
        self.assertEqual(parsed.meta["meta_tied"], 1.0)

    def test_probe_count_mismatch(self):
        with self.assertRaises(parity_real.ParityError):
            parity_real.parse_probe_output("# count 3\n1\n2\n")

    def test_probe_missing_count(self):
        with self.assertRaises(parity_real.ParityError):
            parity_real.parse_probe_output("1\n2\n")

    def test_probe_bad_number(self):
        with self.assertRaises(parity_real.ParityError):
            parity_real.parse_probe_output("# count 1\nabc\n")

    def test_probe_empty(self):
        with self.assertRaises(parity_real.ParityError):
            parity_real.parse_probe_output("# count 0\n")


# ---------------------------------------------------------------------------
# Сравнение логитов
# ---------------------------------------------------------------------------


class TestCompare(unittest.TestCase):
    def test_identical_passes(self):
        values = [0.1, 0.5, -0.2, 0.9]
        result = parity_real.compare_logits(values, values)
        self.assertEqual(result["max_abs_delta"], 0.0)
        self.assertTrue(result["argmax_match"])
        self.assertEqual(result["argmax_reference"], 3)
        self.assertEqual(result["top5_overlap"], 4)
        self.assertTrue(result["passed"])

    def test_small_delta_passes(self):
        result = parity_real.compare_logits(
            [0.0, 1.0, 2.0], [0.001, 1.001, 2.001], threshold=1e-2
        )
        self.assertAlmostEqual(result["max_abs_delta"], 0.001)
        self.assertTrue(result["within_threshold"])
        self.assertTrue(result["passed"])

    def test_threshold_is_strict(self):
        result = parity_real.compare_logits(
            [0.0, 1.0], [0.01, 1.0], threshold=1e-2
        )
        self.assertFalse(result["within_threshold"])
        self.assertFalse(result["passed"])

    def test_argmax_mismatch_fails(self):
        result = parity_real.compare_logits(
            [0.0, 1.0], [2.0, 0.0], threshold=1e-2
        )
        self.assertFalse(result["argmax_match"])
        self.assertFalse(result["passed"])

    def test_length_mismatch_raises(self):
        with self.assertRaises(parity_real.ParityError):
            parity_real.compare_logits([1.0], [1.0, 2.0])

    def test_top_indices_ties_use_index(self):
        self.assertEqual(parity_real.top_indices([1.0, 1.0, 1.0], 2), [0, 1])
        self.assertEqual(parity_real.top_indices([], 5), [])

    def test_argmax_first_max(self):
        self.assertEqual(parity_real.argmax([1.0, 3.0, 3.0, 2.0]), 1)
        with self.assertRaises(ValueError):
            parity_real.argmax([])

    def test_summarize_logits(self):
        summary = parity_real.summarize_logits([0.0, 2.0, 1.0])
        self.assertEqual(summary["argmax"], 1)
        self.assertEqual(summary["top5"], [1, 2, 0])
        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["first8"], [0.0, 2.0, 1.0])
        self.assertAlmostEqual(summary["sum"], 3.0)


# ---------------------------------------------------------------------------
# Архитектура / HF-конфиг / run.json
# ---------------------------------------------------------------------------


def _hf_config(**overrides):
    base = dict(
        hidden_size=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        intermediate_size=64,
        vocab_size=32,
        num_hidden_layers=1,
        max_position_embeddings=128,
        rope_theta=1_000_000.0,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        hidden_act="silu",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class TestQatExportContract(unittest.TestCase):
    """run.json.architecture должен быть тем HF-видом, который ждёт qat.export."""

    def test_hf_config_shape_feeds_qat_export(self):
        from qat import export as qat_export

        cfg = _hf_config(
            hidden_size=1536,
            num_attention_heads=12,
            num_key_value_heads=2,
            head_dim=128,
            intermediate_size=8960,
            vocab_size=151936,
            num_hidden_layers=28,
            max_position_embeddings=32768,
            rope_theta=1_000_000.0,
            tie_word_embeddings=True,
            rope_scaling=None,
            use_sliding_window=False,
            partial_rotary_factor=1.0,
        )
        hf_config = parity_real.hf_config_dict(cfg)
        shapes = {
            "model.embed_tokens.weight": (151936, 1536),
            "model.norm.weight": (1536,),
            "lm_head.weight": (151936, 1536),
        }
        layer = {
            "self_attn.q_proj.weight": (1536, 1536),
            "self_attn.k_proj.weight": (256, 1536),
            "self_attn.v_proj.weight": (256, 1536),
            "self_attn.o_proj.weight": (1536, 1536),
            "mlp.gate_proj.weight": (8960, 1536),
            "mlp.up_proj.weight": (8960, 1536),
            "mlp.down_proj.weight": (1536, 8960),
            "input_layernorm.weight": (1536,),
            "post_attention_layernorm.weight": (1536,),
            "self_attn.q_proj.bias": (1536,),
            "self_attn.k_proj.bias": (256,),
            "self_attn.v_proj.bias": (256,),
        }
        for index in range(28):
            for key, shape in layer.items():
                shapes[f"model.layers.{index}.{key}"] = shape
        arch = qat_export.infer_architecture(
            shapes,
            hf_config,
            tie_hint=True,
            overrides={"context_len": 8},
            embed_mode="none",
        )
        self.assertEqual(arch["hidden"], 1536)
        self.assertEqual(arch["n_heads"], 12)
        self.assertEqual(arch["n_kv_heads"], 2)
        self.assertEqual(arch["head_dim"], 128)
        self.assertEqual(arch["intermediate"], 8960)
        self.assertEqual(arch["vocab"], 151936)
        self.assertEqual(arch["n_layers"], 28)
        self.assertEqual(arch["context_len"], 8)
        self.assertEqual(arch["rope_theta"], 1_000_000.0)
        self.assertTrue(arch["tie_word_embeddings"])
        self.assertEqual(arch["embed"], "none")

    def test_unsupported_rope_scaling_is_caught_before_export(self):
        with self.assertRaisesRegex(parity_real.ParityError, "rope_scaling"):
            parity_real.check_model_supported(
                _hf_config(rope_scaling={"type": "yarn", "factor": 4.0})
            )


class TestArchitecture(unittest.TestCase):
    def test_build_architecture(self):
        arch = parity_real.build_architecture(_hf_config(), context_len=8)
        self.assertEqual(arch["hidden"], 32)
        self.assertEqual(arch["n_heads"], 2)
        self.assertEqual(arch["n_kv_heads"], 1)
        self.assertEqual(arch["head_dim"], 16)
        self.assertEqual(arch["intermediate"], 64)
        self.assertEqual(arch["vocab"], 32)
        self.assertEqual(arch["n_layers"], 1)
        self.assertEqual(arch["context_len"], 8)
        self.assertEqual(arch["rope_theta"], 1_000_000.0)
        self.assertTrue(arch["tie_word_embeddings"])
        self.assertEqual(arch["rope_layout"], "half")

    def test_head_dim_is_computed(self):
        cfg = _hf_config(head_dim=None)
        arch = parity_real.build_architecture(cfg, context_len=8)
        self.assertEqual(arch["head_dim"], 16)

    def test_supported_model_passes(self):
        parity_real.check_model_supported(_hf_config())

    def test_unsupported_rope_scaling(self):
        with self.assertRaisesRegex(parity_real.ParityError, "rope_scaling"):
            parity_real.check_model_supported(
                _hf_config(rope_scaling={"type": "linear", "factor": 2.0})
            )

    def test_unsupported_partial_rotary(self):
        with self.assertRaisesRegex(parity_real.ParityError, "partial_rotary_factor"):
            parity_real.check_model_supported(_hf_config(partial_rotary_factor=0.5))

    def test_unsupported_sliding_window(self):
        with self.assertRaisesRegex(parity_real.ParityError, "sliding"):
            parity_real.check_model_supported(_hf_config(use_sliding_window=True))

    def test_sliding_window_beyond_prompt_is_allowed(self):
        # Окно 32768 больше прогона 8: маска ни на что не влияет.
        parity_real.check_model_supported(
            _hf_config(use_sliding_window=True, sliding_window=32768), 8
        )

    def test_small_sliding_window_is_rejected(self):
        with self.assertRaisesRegex(parity_real.ParityError, "sliding"):
            parity_real.check_model_supported(
                _hf_config(use_sliding_window=True, sliding_window=4), 8
            )

    def test_unsupported_activation(self):
        with self.assertRaisesRegex(parity_real.ParityError, "hidden_act"):
            parity_real.check_model_supported(_hf_config(hidden_act="gelu"))

    def test_bad_kv_heads(self):
        with self.assertRaisesRegex(parity_real.ParityError, "делится"):
            parity_real.check_model_supported(
                _hf_config(num_attention_heads=3, num_key_value_heads=2)
            )

    def test_retie_quantized_lm_head(self):
        class Tensor:
            def detach(self):
                return self

            def clone(self):
                return self

        class Module:
            def __init__(self, weight):
                self.weight = weight

        class Model:
            def __init__(self):
                self.embed = Module(Tensor())
                self.head = Module(object())

            def get_input_embeddings(self):
                return self.embed

            def get_output_embeddings(self):
                return self.head

        fake_torch = SimpleNamespace(
            nn=SimpleNamespace(Parameter=lambda value: ("param", value))
        )
        model = Model()
        self.assertTrue(parity_real.retie_quantized_lm_head(fake_torch, model))
        self.assertEqual(model.head.weight, ("param", model.embed.weight))

    def test_manifest_architecture_contract(self):
        arch = parity_real.build_architecture(_hf_config(), context_len=8)
        manifest = {
            "hidden": 32,
            "n_heads": 2,
            "n_kv_heads": 1,
            "head_dim": 16,
            "intermediate": 64,
            "vocab": 32,
            "n_layers": 1,
            "context_len": 8,
            "tie_word_embeddings": True,
            "embed": "none",
            "rope_theta": 1_000_000.0,
            "rms_norm_eps": 1e-6,
        }
        self.assertEqual(
            parity_real.check_manifest_architecture(
                manifest, arch, context_len=8, embed_mode="none"
            ),
            [],
        )
        broken = dict(manifest, rope_theta=10000.0, embed="trit")
        problems = parity_real.check_manifest_architecture(
            broken, arch, context_len=8, embed_mode="none"
        )
        self.assertTrue(any("rope_theta" in item for item in problems))
        self.assertTrue(any("embed" in item for item in problems))

    def test_tie_check(self):
        class FakeWeight:
            def __init__(self, ptr):
                self._ptr = ptr

            def data_ptr(self):
                return self._ptr

        class FakeModule:
            def __init__(self, ptr):
                self.weight = FakeWeight(ptr)

        class FakeModel:
            def __init__(self, tie, embed_ptr, head_ptr):
                self.config = SimpleNamespace(tie_word_embeddings=tie)
                self._embed = FakeModule(embed_ptr)
                self._head = FakeModule(head_ptr)

            def get_input_embeddings(self):
                return self._embed

            def get_output_embeddings(self):
                return self._head

        self.assertFalse(parity_real.hf_tie_broken(FakeModel(True, 1, 1)))
        self.assertTrue(parity_real.hf_tie_broken(FakeModel(True, 1, 2)))
        self.assertFalse(parity_real.hf_tie_broken(FakeModel(False, 1, 2)))

    def test_hf_config_dict_is_jsonable(self):
        cfg = _hf_config(rope_scaling=None)
        data = parity_real.hf_config_dict(cfg)
        self.assertEqual(data["hidden_size"], 32)
        self.assertIsNone(data["rope_scaling"])
        json.dumps(data)

    def test_build_run_config(self):
        args = parity_real.parse_args(["--limit-tokens", "9"])
        quant = SimpleNamespace(
            block_size=128,
            target_suffixes=("q_proj",),
            exclude_suffixes=("lm_head",),
            quantize_embeddings="none",
        )
        config = parity_real.build_run_config(args, quant)
        self.assertEqual(config["quant"]["quantize_embeddings"], "none")
        self.assertEqual(config["quant"]["scale_dtype"], "float16")
        self.assertEqual(config["quant"]["target_suffixes"], ["q_proj"])
        self.assertEqual(config["data"]["max_seq_len"], 9)
        self.assertEqual(config["train"]["max_steps"], 0)

    def test_build_run_meta(self):
        config = {
            "quant": {
                "block_size": 128,
                "scale_dtype": "float16",
                "target_suffixes": ["q_proj"],
            }
        }
        meta = parity_real.build_run_meta(
            model_name="m",
            config=config,
            hf_architecture={"hidden_size": 32},
            vocab=32,
            num_params=100,
            ternary_params=50,
            ternary_report=[{"name": "q_proj", "params": 50}],
            device="cpu",
        )
        self.assertEqual(meta["format"], "eva32.qat.run")
        self.assertEqual(meta["model"]["vocab_size"], 32)
        self.assertEqual(meta["quant"]["modules_count"], 1)
        self.assertEqual(meta["architecture"]["hidden_size"], 32)
        json.dumps(meta)


# ---------------------------------------------------------------------------
# Сводка и JSON
# ---------------------------------------------------------------------------


def _result(**overrides):
    comparison = {
        "method": "c-probe",
        "max_abs_delta": 0.001,
        "argmax_match": True,
        "argmax_reference": 5,
        "argmax_runtime": 5,
        "top5_reference": [5, 4, 3, 2, 1],
        "top5_runtime": [5, 4, 3, 2, 1],
        "top5_overlap": 5,
        "top5_size": 5,
        "threshold": 1e-2,
        "within_threshold": True,
        "passed": True,
        "meta_ok": True,
        "cli_argmax_match": True,
    }
    base = dict(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        prompt="The capital of France is",
        token_ids=[1, 2, 3],
        limit_tokens=16,
        quantize_embeddings="none",
        scale_dtype="float16",
        device="cpu",
        torch_dtype="float32",
        context_len=8,
        workdir=Path("/tmp/parity"),
        artifacts={"json": "/tmp/parity/parity.json", "vtm": "/tmp/parity/model.vtm"},
        reference_quantized={
            "argmax": 5,
            "top5": [5, 4, 3, 2, 1],
            "count": 10,
            "first8": [0.0] * 8,
            "sum": 0.0,
        },
        reference_full={
            "argmax": 4,
            "top5": [4, 5, 3, 2, 1],
            "count": 10,
            "first8": [0.0] * 8,
            "sum": 0.0,
        },
        runtime={"kind": "vtm-c", "argmax": 5, "top5": [5, 4, 3, 2, 1],
                 "count": 10, "first8": [0.0] * 8, "sum": 0.0},
        comparison=comparison,
        full_precision={
            "max_abs_delta": 1.2,
            "argmax_match": True,
            "argmax_reference": 4,
            "argmax_runtime": 5,
            "top5_overlap": 3,
            "top5_size": 5,
            "passed": False,
            "informational": True,
        },
        quantization_effect={
            "max_abs_delta": 1.1,
            "argmax_match": False,
            "top5_overlap": 2,
            "top5_size": 5,
            "informational": True,
        },
        warnings=["test warning"],
        timings={"total": 1.25},
        status="passed",
        error=None,
        argv=["tools/parity_real.py"],
    )
    base.update(overrides)
    return parity_real.make_result(**base)


class TestSummary(unittest.TestCase):
    def test_passed_summary(self):
        text = parity_real.format_summary(_result())
        self.assertIn("parity_real", text)
        self.assertIn("max|Δ|", text)
        self.assertIn("argmax", text)
        self.assertIn("top-5: 5/5", text)
        self.assertIn("порог", text)
        self.assertIn("ПРОЙДЕНО", text)
        self.assertIn("справочно", text)
        self.assertIn("test warning", text)
        self.assertIn("/tmp/parity/parity.json", text)

    def test_failed_summary(self):
        result = _result(status="failed")
        result["comparison"] = dict(result["comparison"], passed=False,
                                    argmax_match=False, max_abs_delta=0.5)
        text = parity_real.format_summary(result)
        self.assertIn("ПРОВАЛЕНО", text)
        self.assertIn("НЕ совпал", text)

    def test_partial_summary(self):
        result = _result(status="partial")
        result["comparison"] = {
            "method": "cli-only",
            "max_abs_delta": None,
            "argmax_match": True,
            "argmax_reference": 5,
            "argmax_runtime": 5,
            "top5_reference": [5],
            "top5_runtime": [],
            "top5_overlap": None,
            "top5_size": 5,
            "threshold": 1e-2,
            "within_threshold": None,
            "passed": False,
            "meta_ok": None,
            "cli_argmax_match": True,
        }
        result["full_precision"] = None
        text = parity_real.format_summary(result)
        self.assertIn("ЧАСТИЧНО", text)
        self.assertIn("не измерен", text)
        self.assertIn("C-логиты: не измерены", text)

    def test_result_json_is_strict(self):
        result = _result()
        encoded = json.dumps(parity_real._jsonable(result), allow_nan=False)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["format"], "eva32.parity.real")
        self.assertEqual(decoded["status"], "passed")
        self.assertTrue(decoded["passed"])

    def test_jsonable_nonfinite(self):
        data = parity_real._jsonable({"x": float("inf"), "y": float("nan")})
        self.assertEqual(data["x"], "inf")
        self.assertEqual(data["y"], "nan")
        json.dumps(data, allow_nan=False)

    def test_write_json_atomic(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "parity.json"
            parity_real.write_json_atomic(path, {"a": 1, "b": Path("/x")})
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded, {"a": 1, "b": "/x"})

    def test_write_failure_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "parity.json"
            args = parity_real.parse_args([])
            parity_real.write_failure_json(
                path,
                args=args,
                token_ids=[1, 2],
                workdir=Path(tmp),
                context_len=8,
                stage="qat.export",
                error="boom",
                returncode=2,
                stderr="line1\nline2",
                warnings=["w"],
                timings={"export": 1.5},
            )
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["format"], "eva32.parity.real")
            self.assertEqual(loaded["status"], "failed")
            self.assertFalse(loaded["passed"])
            self.assertEqual(loaded["token_ids"], [1, 2])
            self.assertEqual(loaded["stage"], "qat.export")
            self.assertEqual(loaded["error"], "boom")
            self.assertIn("line2", loaded["stderr_tail"])
            self.assertEqual(loaded["timings"], {"export": 1.5})


# ---------------------------------------------------------------------------
# C-проба: исходник и синтаксис
# ---------------------------------------------------------------------------


class TestProbeSource(unittest.TestCase):
    def test_source_contract(self):
        source = parity_real.probe_source()
        self.assertIn('vtm_tensor_by_name(machine, "logits")', source)
        self.assertIn("vtm_run(machine", source)
        self.assertIn("vtm_meta_rope_theta", source)
        self.assertIn("# count", source)
        self.assertIn("%.9g", source)

    def test_source_compiles_if_toolchain_available(self):
        cc = parity_real.find_c_compiler()
        if not cc:
            self.skipTest("C-компилятор не найден")
        if not (parity_real.VTM_INCLUDE / "vtm.h").is_file():
            self.skipTest("нет src/vtm/vtm.h")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "parity_probe.c"
            path.write_text(parity_real.probe_source(), encoding="utf-8")
            proc = subprocess.run(
                [
                    cc,
                    "-std=c99",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-I",
                    str(parity_real.VTM_INCLUDE),
                    "-fsyntax-only",
                    str(path),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)


# ---------------------------------------------------------------------------
# dry-run и понятные ошибки
# ---------------------------------------------------------------------------


class TestDryRunAndErrors(unittest.TestCase):
    def test_dry_run_exit_zero(self):
        out = io.StringIO()
        err = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = parity_real.main(["--dry-run"])
        self.assertEqual(code, 0, err.getvalue())
        text = out.getvalue()
        self.assertIn("dry-run", text)
        self.assertIn("план", text)
        self.assertIn("порог", text)
        self.assertIn("команды, которые будут выполнены", text)
        self.assertNotIn("Traceback", text + err.getvalue())

    def test_dry_run_int8_is_plan_only(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = parity_real.main(["--dry-run", "--quantize-embeddings", "int8"])
        self.assertEqual(code, 0)
        self.assertIn("int8", out.getvalue())

    def test_no_torch_exit_2(self):
        def fake_available(name):
            return name != "torch"

        err = io.StringIO()
        with mock.patch.object(
            parity_real, "module_available", side_effect=fake_available
        ), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            code = parity_real.main([])
        self.assertEqual(code, 2)
        self.assertIn("нет torch", err.getvalue())
        self.assertIn("--dry-run", err.getvalue())

    def test_missing_c_runtime_mentions_make(self):
        env = parity_real.Environment(
            torch=True,
            transformers=True,
            numpy=True,
            compiler_script=True,
            export_module=True,
            vtm_bin=False,
            vtm_lib=False,
            cc="/usr/bin/cc",
            model_state="hub",
            c_include=True,
        )
        text = parity_real._missing_c_runtime_message(env)
        self.assertIn("make -C src/vtm", text)
        self.assertIn("src/vtm/vtm", text)
        self.assertIn("libvtm.a", text)

    def test_check_environment_does_not_import_torch(self):
        # find_spec не должен тянуть torch; в тестовом окружении его нет.
        env = parity_real.check_environment(parity_real.DEFAULT_MODEL)
        self.assertIsInstance(env.torch, bool)
        self.assertIn(env.model_state, ("local", "hub", "missing"))

    def test_local_model_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(parity_real.local_model_state(tmp), "local")
        self.assertEqual(
            parity_real.local_model_state("Qwen/Qwen2.5-1.5B-Instruct"), "hub"
        )
        self.assertEqual(
            parity_real.local_model_state("/nonexistent/model/path"), "missing"
        )


if __name__ == "__main__":
    unittest.main()
