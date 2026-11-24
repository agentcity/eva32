"""Холодный экспорт HF-весов в .trit (задача 0134).

Что проверяется без torch (этот файл не требует GPU-стека):

* правило квантизации на синтетике: absmax на блок 128, полу-от-нуля,
  нулевой блок -> нулевые триты и нулевой масштаб;
* round-trip формата: ``cold_export`` пишет ``ternary.trit`` /
  ``non_ternary.bin`` / ``manifest.json``, ``tools/vtm_compile.py`` их
  читает и собирает полный граф, Python-эталон VTM его исполняет;
* манифест в формате компилятора: архитектура, ``context_window``,
  версия, sha256 файлов, смещения сегментов;
* tied-эмбеддинги и режим ``--embed trit`` (таблица словаря сегментом в
  ``ternary.trit``, ссылка ``tied_to`` без fp32-копии);
* ``--limit-layers``: манифест описывает частичную модель (N слоёв);
* отказ при неподдерживаемом конфиге: hidden не кратен блоку, rope_scaling,
  не-SiLU активация, block_size != 128, неизвестный режим эмбеддингов.

С torch дополнительно сверяется кусковой кодировщик больших тензоров:
его сегмент обязан побайтово совпасть с ``qat.export.encode_trit_segment``
(иначе формат «раздвоился»).

Запуск:
    python3 -m unittest tests.test_cold_export -v
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
from tools import cold_export  # noqa: E402
from tools import vtm_compile, vtm_ref  # noqa: E402


def has_torch() -> bool:
    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):
        return False


HAS_TORCH = has_torch()


# ---------------------------------------------------------------------------
# Крошечная Qwen-подобная модель: fake-тензоры без torch
# ---------------------------------------------------------------------------


class FakeTensor:
    """Тензор без torch: shape, numel и итератор значений.

    ``qat.export.encode_trit_segment`` принимает такой объект по «чистому»
    пути (``encode_trit_table``), ``cold_export._write_fp32`` — тоже.
    """

    def __init__(self, shape, values, name=""):
        self.shape = tuple(int(x) for x in shape)
        prod = 1
        for dim in self.shape:
            prod *= dim
        values = list(values)
        if len(values) != prod:
            raise AssertionError(f"{name}: {len(values)} значений на shape {self.shape}")
        self._values = values
        self.name = name

    def numel(self):
        return len(self._values)

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)


def _floats(n, seed=0):
    """Детерминированные значения в [-1, 1] без random (чтобы тест был точным)."""
    return [(((i * 37 + seed * 101) % 200) - 100) / 100.0 for i in range(n)]


def _norms(n, seed=0):
    return [0.5 + (i * 7 + seed) % 50 / 100.0 for i in range(n)]


TINY = {
    "hidden": 128,
    "n_heads": 2,
    "n_kv_heads": 1,
    "head_dim": 64,
    "intermediate": 256,
    "vocab": 128,
    "layers": 2,
    "eps": 1e-6,
    "rope_theta": 1_000_000.0,
}


def tiny_hf_config(**overrides):
    spec = dict(TINY)
    spec.update(overrides)
    return {
        "model_type": "qwen2",
        "architectures": ["Qwen2ForCausalLM"],
        "hidden_size": spec["hidden"],
        "num_attention_heads": spec["n_heads"],
        "num_key_value_heads": spec["n_kv_heads"],
        "head_dim": spec["head_dim"],
        "intermediate_size": spec["intermediate"],
        "vocab_size": spec["vocab"],
        "num_hidden_layers": spec["layers"],
        "rms_norm_eps": spec["eps"],
        "rope_theta": spec["rope_theta"],
        "hidden_act": "silu",
        "tie_word_embeddings": spec.get("tie", True),
        "max_position_embeddings": 32768,
        "use_sliding_window": False,
        "sliding_window": 32768,
    }


def tiny_model(*, tie=True, layers=None, biases=("q", "k", "v")):
    """state_dict в HF-именах: словарь fake-тензоров."""
    hidden = TINY["hidden"]
    q_dim = TINY["n_heads"] * TINY["head_dim"]
    kv_dim = TINY["n_kv_heads"] * TINY["head_dim"]
    inter = TINY["intermediate"]
    vocab = TINY["vocab"]
    n_layers = TINY["layers"] if layers is None else layers
    state = {
        "model.embed_tokens.weight": FakeTensor(
            (vocab, hidden), _floats(vocab * hidden, 1), "embed"
        ),
        "model.norm.weight": FakeTensor((hidden,), _norms(hidden, 2), "final_norm"),
    }
    for layer in range(n_layers):
        prefix = f"model.layers.{layer}"
        seed = 10 + layer
        state[f"{prefix}.self_attn.q_proj.weight"] = FakeTensor(
            (q_dim, hidden), _floats(q_dim * hidden, seed), "q_proj"
        )
        state[f"{prefix}.self_attn.k_proj.weight"] = FakeTensor(
            (kv_dim, hidden), _floats(kv_dim * hidden, seed + 1), "k_proj"
        )
        state[f"{prefix}.self_attn.v_proj.weight"] = FakeTensor(
            (kv_dim, hidden), _floats(kv_dim * hidden, seed + 2), "v_proj"
        )
        state[f"{prefix}.self_attn.o_proj.weight"] = FakeTensor(
            (hidden, q_dim), _floats(hidden * q_dim, seed + 3), "o_proj"
        )
        state[f"{prefix}.mlp.gate_proj.weight"] = FakeTensor(
            (inter, hidden), _floats(inter * hidden, seed + 4), "gate_proj"
        )
        state[f"{prefix}.mlp.up_proj.weight"] = FakeTensor(
            (inter, hidden), _floats(inter * hidden, seed + 5), "up_proj"
        )
        state[f"{prefix}.mlp.down_proj.weight"] = FakeTensor(
            (hidden, inter), _floats(hidden * inter, seed + 6), "down_proj"
        )
        state[f"{prefix}.input_layernorm.weight"] = FakeTensor(
            (hidden,), _norms(hidden, seed), "norm1"
        )
        state[f"{prefix}.post_attention_layernorm.weight"] = FakeTensor(
            (hidden,), _norms(hidden, seed + 1), "norm2"
        )
        for tag in biases:
            dim, key = {
                "q": (q_dim, "q_proj"), "k": (kv_dim, "k_proj"),
                "v": (kv_dim, "v_proj"), "o": (hidden, "o_proj"),
                "gate": (inter, "gate_proj"), "up": (inter, "up_proj"),
                "down": (hidden, "down_proj"),
            }[tag]
            state[f"{prefix}.{_bias_path(tag)}"] = FakeTensor(
                (dim,), _floats(dim, seed + 7), f"{key}.bias"
            )
    if not tie:
        state["lm_head.weight"] = FakeTensor(
            (vocab, hidden), _floats(vocab * hidden, 99), "lm_head"
        )
    return state


def _bias_path(tag: str) -> str:
    if tag in ("q", "k", "v", "o"):
        return f"self_attn.{tag}_proj.bias"
    return f"mlp.{tag}_proj.bias"


def export_tiny(directory, *, tie=True, embed="none", layers=None, config=None,
                force=True, **kwargs):
    state = tiny_model(tie=tie, layers=layers)
    source = cold_export.DictSource(state, label="tiny")
    cfg = config if config is not None else tiny_hf_config(tie=tie, layers=layers)
    result = cold_export.export_model(
        source, cfg, directory,
        model_name="tiny/qwen", embed_mode=embed, force=force, **kwargs,
    )
    return result, state


# ---------------------------------------------------------------------------
# Квантизация на синтетике
# ---------------------------------------------------------------------------


class TestQuantization(unittest.TestCase):
    def test_absmax_blocks_and_half_away(self):
        # Два блока по 128: первый с absmax 2.0, второй нулевой.
        values = [0.0] * 256
        values[0] = 2.0
        values[1] = 1.0          # 1.0/2.0 = 0.5 -> +1 (полу-от-нуля)
        values[2] = -1.0         # -0.5 -> -1
        values[3] = 0.4          # 0.2 -> 0
        tensor = FakeTensor((2, 128), values, "w")
        data, meta = cold_export.encode_trit_segment(tensor, "w")
        trits, scales = ternary.decode_segment(data, ternary.BLOCK_SIZE)
        self.assertEqual(len(scales), 2)
        self.assertEqual(scales[0], 2.0)
        self.assertEqual(scales[1], 0.0)
        self.assertEqual(trits[0], 1)
        self.assertEqual(trits[1], 1)
        self.assertEqual(trits[2], -1)
        self.assertEqual(trits[3], 0)
        # Нулевой блок: все триты нули (деления на ноль нет).
        self.assertEqual(set(trits[128:]), {0})
        self.assertEqual(meta["n_scale_blocks"], 2)
        self.assertEqual(meta["n_trits"], 256)
        self.assertEqual(meta["shape"], [2, 128])

    def test_matches_qat_rule(self):
        """Триты/масштабы совпадают с qat.ternary.quantize на тех же числах."""
        rows, cols = 3, 256
        values = _floats(rows * cols, 5)
        data, _meta = cold_export.encode_trit_segment(
            FakeTensor((rows, cols), values, "w"), "w"
        )
        trits, scales = ternary.decode_segment(data, ternary.BLOCK_SIZE)
        expected_trits, expected_scales = ternary.quantize(values, ternary.BLOCK_SIZE)
        self.assertEqual(len(scales), len(expected_scales))
        for got, want in zip(scales, expected_scales):
            # Масштабы в файле fp32, эталон — double: сравниваем после
            # округления эталона до float32.
            self.assertAlmostEqual(got, float(struct.unpack("<f", struct.pack("<f", want))[0]), places=9)
        self.assertEqual(trits, expected_trits)

    def test_rejects_non_multiple_input_dim(self):
        tensor = FakeTensor((2, 100), [0.1] * 200, "w")
        with self.assertRaises(RuntimeError) as ctx:
            cold_export.encode_trit_segment(tensor, "w")
        self.assertIn("128", str(ctx.exception))

    def test_block_size_must_be_128(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = tiny_model()
            with self.assertRaises(cold_export.ColdExportError) as ctx:
                cold_export.export_model(
                    cold_export.DictSource(state), tiny_hf_config(), tmp,
                    block_size=160,
                )
            self.assertIn("128", str(ctx.exception))

    def test_rejects_embed_mode(self):
        # int8 перестал быть «неизвестным режимом» в 0143: его варианты
        # проверяет tests/test_embed_variants.py. Здесь — по-прежнему отказ на
        # опечатке в имени режима.
        with tempfile.TemporaryDirectory() as tmp:
            state = tiny_model()
            with self.assertRaises(cold_export.ColdExportError):
                cold_export.export_model(
                    cold_export.DictSource(state), tiny_hf_config(), tmp,
                    embed_mode="int4",
                )


# ---------------------------------------------------------------------------
# Поддерживаемость конфига
# ---------------------------------------------------------------------------


class TestUnsupportedConfig(unittest.TestCase):
    def test_rope_scaling_rejected(self):
        cfg = tiny_hf_config()
        cfg["rope_scaling"] = {"type": "linear", "factor": 2.0}
        with self.assertRaises(cold_export.ColdExportError) as ctx:
            cold_export.check_supported_config(cfg)
        self.assertIn("rope_scaling", str(ctx.exception))

    def test_non_silu_rejected(self):
        cfg = tiny_hf_config()
        cfg["hidden_act"] = "gelu"
        with self.assertRaises(cold_export.ColdExportError) as ctx:
            cold_export.check_supported_config(cfg)
        self.assertIn("SiLU", str(ctx.exception))

    def test_foreign_model_type_rejected(self):
        cfg = tiny_hf_config()
        cfg["model_type"] = "phi3"
        with self.assertRaises(cold_export.ColdExportError):
            cold_export.check_supported_config(cfg)

    def test_sliding_window_only_when_enabled(self):
        cfg = tiny_hf_config()
        cfg["sliding_window"] = 4096
        cold_export.check_supported_config(cfg)  # выключено — не мешает
        cfg["use_sliding_window"] = True
        with self.assertRaises(cold_export.ColdExportError):
            cold_export.check_supported_config(cfg)

    def test_hidden_not_multiple_of_block(self):
        architecture = {
            "hidden": 100, "n_heads": 2, "head_dim": 50,
            "n_kv_heads": 1, "intermediate": 256, "vocab": 100,
        }
        with self.assertRaises(cold_export.ColdExportError) as ctx:
            cold_export.check_block_alignment(architecture, 128)
        self.assertIn("не делится на block_size=128", str(ctx.exception))

    def test_qwen2_5_sliding_window_default_passes(self):
        # У Qwen2.5 в config.json sliding_window=32768, но окно выключено:
        # отказывать на этом нельзя (проверено на реальной 0.5B).
        cfg = tiny_hf_config()
        cfg["max_window_layers"] = 21
        cold_export.check_supported_config(cfg)


# ---------------------------------------------------------------------------
# Экспорт -> компиляция -> исполнение
# ---------------------------------------------------------------------------


class TestExportRoundTrip(unittest.TestCase):
    def test_manifest_compiles_and_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, _state = export_tiny(tmp)
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["format"], vtm_compile.MANIFEST_FORMAT)
            self.assertEqual(manifest["format_version"], vtm_compile.MANIFEST_VERSION)
            self.assertEqual(manifest["context_window"], cold_export.DEFAULT_CONTEXT_WINDOW)
            self.assertEqual(manifest["architecture"]["context_len"],
                             cold_export.DEFAULT_CONTEXT_WINDOW)
            self.assertEqual(manifest["architecture"]["n_layers"], TINY["layers"])
            self.assertEqual(manifest["architecture"]["hidden"], TINY["hidden"])
            self.assertEqual(manifest["training"]["trained"], False)
            self.assertEqual(manifest["training"]["step"], None)
            self.assertIn("hf_config", manifest)
            self.assertIn("cold_export", manifest)

            # sha256 в манифесте — от фактических файлов.
            for name in ("ternary.trit", "non_ternary.bin"):
                path = os.path.join(tmp, name)
                actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
                self.assertEqual(manifest["files"][name]["sha256"], actual)

            export = vtm_compile.load_export(result.manifest_path)
            self.assertEqual(export.cfg.n_layers, TINY["layers"])
            self.assertEqual(export.cfg.context_len, cold_export.DEFAULT_CONTEXT_WINDOW)
            self.assertTrue(export.tied)
            self.assertGreater(export.trit_params, 0)
            self.assertGreater(export.fp_params, 0)

            vtm = vtm_compile.compile_model(export.cfg, export.weights, export.meta)
            out = vtm_ref.execute(vtm, [1, 2, 3, 4], 0)
            self.assertIsNotNone(out.logits)
            self.assertEqual(len(out.logits), TINY["vocab"])
            self.assertTrue(all(math.isfinite(x) for x in out.logits))
            self.assertGreater(max(out.logits) - min(out.logits), 1e-6)

            # Контейнер пишется и читается (round-trip .vtm).
            vtm_path = os.path.join(tmp, "model.vtm")
            vtm_ref.write_vtm(vtm_path, vtm.tensors, vtm.program, vtm.meta,
                              integrity=True)
            loaded = vtm_ref.read_vtm(vtm_path)
            again = vtm_ref.execute(loaded, [1, 2, 3, 4], 0)
            self.assertEqual(out.logits, again.logits)

    def test_trit_round_trip_is_exact(self):
        """Файл сегмента декодируется в те же триты/масштабы, что посчитаны."""
        with tempfile.TemporaryDirectory() as tmp:
            result, _state = export_tiny(tmp)
            blob = result.trit_path.read_bytes()
            entries = result.manifest["tensors"]
            self.assertEqual(len(entries), TINY["layers"] * 7)
            for entry in entries:
                offset = entry["trit_offset"]
                length = entry["trit_length"]
                trits, scales = ternary.decode_segment(
                    blob[offset:offset + length], ternary.BLOCK_SIZE
                )
                self.assertEqual(len(trits), entry["n_trits"])
                self.assertEqual(len(scales), entry["n_scale_blocks"])
                self.assertLessEqual(max(abs(t) for t in trits), 1)
                self.assertTrue(all(math.isfinite(s) for s in scales))
                self.assertGreaterEqual(min(scales), 0.0)

    def test_non_ternary_weights_keep_values(self):
        """Нормы и bias лежат fp32 без округлений; смещения из манифеста верны."""
        with tempfile.TemporaryDirectory() as tmp:
            result, state = export_tiny(tmp)
            blob = result.non_ternary_path.read_bytes()
            checked = 0
            for entry in result.manifest["non_ternary"]["tensors"]:
                if entry.get("tied_to"):
                    continue
                payload = blob[entry["offset"]:entry["offset"] + entry["length"]]
                self.assertEqual(len(payload), entry["params"] * 4)
                source = state[entry["name"]]
                expected = list(source)
                got = struct.unpack(f"<{len(expected)}f", payload)
                for a, b in zip(expected, got):
                    self.assertAlmostEqual(a, b, places=6)
                checked += 1
            self.assertGreater(checked, 0)

    def test_context_window_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, _state = export_tiny(tmp, context_window=512)
            self.assertEqual(result.manifest["context_window"], 512)
            self.assertEqual(result.architecture["context_len"], 512)
            export = vtm_compile.load_export(result.manifest_path)
            self.assertEqual(export.cfg.context_len, 512)

    def test_limit_layers(self):
        """--limit-layers обрезает и шарды, и конфиг: манифест описывает N слоёв."""
        with tempfile.TemporaryDirectory() as tmp:
            result, _state = export_tiny(tmp, limit_layers=1)
            self.assertEqual(result.architecture["n_layers"], 1)
            self.assertEqual(result.manifest["cold_export"]["limit_layers"], 1)
            self.assertEqual(
                result.manifest["cold_export"]["n_layers_exported"], 1
            )
            self.assertEqual(len(result.manifest["tensors"]), 7)
            export = vtm_compile.load_export(result.manifest_path)
            self.assertEqual(export.cfg.n_layers, 1)
            vtm = vtm_compile.compile_model(export.cfg, export.weights, export.meta)
            out = vtm_ref.execute(vtm, [1, 2], 0)
            self.assertEqual(len(out.logits), TINY["vocab"])

    def test_existing_export_requires_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            export_tiny(tmp)
            with self.assertRaises(cold_export.ColdExportError) as ctx:
                export_tiny(tmp, force=False)
            self.assertIn("--force", str(ctx.exception))

    def test_memory_measurement_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, _state = export_tiny(tmp)
            memory = result.manifest["cold_export"]["memory"]
            self.assertGreater(memory["peak_rss_bytes"], 0)
            self.assertIn("limit", memory)

    def test_empty_source_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(cold_export.ColdExportError):
                cold_export.export_model(
                    cold_export.DictSource({}, "empty"), tiny_hf_config(), tmp,
                )


class TestTiedAndEmbed(unittest.TestCase):
    def test_tied_lm_head_is_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, _state = export_tiny(tmp, tie=True)
            entries = result.manifest["non_ternary"]["tensors"]
            heads = [e for e in entries if e.get("name") == "lm_head.weight"]
            self.assertEqual(len(heads), 1)
            self.assertEqual(heads[0]["tied_to"], "model.embed_tokens.weight")
            self.assertEqual(heads[0]["dtype"], "float32")
            self.assertNotIn("offset", heads[0])
            embed = [e for e in entries if e.get("name") == "model.embed_tokens.weight"]
            self.assertEqual(len(embed), 1)
            self.assertEqual(embed[0]["role"], "embed")
            self.assertEqual(result.manifest["architecture"]["lm_head"], "mm")

    def test_untied_lm_head_written_fp32(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, _state = export_tiny(tmp, tie=False)
            entries = result.manifest["non_ternary"]["tensors"]
            head = [e for e in entries if e.get("name") == "lm_head.weight"][0]
            self.assertNotIn("tied_to", head)
            self.assertIn("offset", head)
            self.assertEqual(head["shape"], [TINY["vocab"], TINY["hidden"]])
            export = vtm_compile.load_export(result.manifest_path)
            self.assertFalse(export.tied)
            self.assertIn("lm_head", export.weights)

    def test_embed_trit_goes_to_ternary_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, _state = export_tiny(tmp, embed="trit")
            names = [entry["name"] for entry in result.manifest["tensors"]]
            self.assertEqual(names[0], qat_export.EMBED_TENSOR_NAME)
            embed = result.manifest["tensors"][0]
            self.assertEqual(embed["role"], "embed")
            self.assertEqual(embed["trit_offset"], 0)
            # fp32-копии таблицы нет ни в non_ternary, ни в его записях.
            non_names = [
                entry["name"] for entry in result.manifest["non_ternary"]["tensors"]
            ]
            self.assertNotIn(qat_export.EMBED_TENSOR_NAME, non_names)
            tied = [
                entry for entry in result.manifest["non_ternary"]["tensors"]
                if entry.get("name") == "lm_head.weight"
            ][0]
            self.assertEqual(tied["dtype"], "trit")
            self.assertEqual(tied["tied_to"], qat_export.EMBED_TENSOR_NAME)
            self.assertEqual(result.manifest["embeddings"]["mode"], "trit")
            self.assertEqual(
                result.manifest["embeddings"]["storage"], "ternary.trit"
            )
            export = vtm_compile.load_export(result.manifest_path)
            self.assertEqual(export.cfg.embed_quant, "trit")
            self.assertEqual(export.weights["embed"].kind, vtm_ref.KIND_TRITS)
            vtm = vtm_compile.compile_model(export.cfg, export.weights, export.meta)
            out = vtm_ref.execute(vtm, [1, 2], 0)
            self.assertEqual(len(out.logits), TINY["vocab"])

    def test_embed_none_keeps_fp32_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, _state = export_tiny(tmp, embed="none")
            names = [entry["name"] for entry in result.manifest["tensors"]]
            self.assertNotIn(qat_export.EMBED_TENSOR_NAME, names)
            self.assertEqual(result.manifest["embeddings"]["mode"], "none")
            self.assertEqual(result.manifest["embeddings"]["storage"],
                             "non_ternary.bin")


class TestStreamingSource(unittest.TestCase):
    def test_each_tensor_read_once_and_released(self):
        """Экспорт не «загружает модель»: каждый ключ читается ровно один раз."""
        reads: list[str] = []
        releases: list[str] = []

        class CountingSource(cold_export.DictSource):
            def tensor(self, key):
                reads.append(key)
                return super().tensor(key)

            def release(self, key):
                releases.append(key)

        with tempfile.TemporaryDirectory() as tmp:
            state = tiny_model()
            cold_export.export_model(
                CountingSource(state), tiny_hf_config(), tmp, force=True,
            )
            self.assertEqual(len(reads), len(set(reads)))
            self.assertGreater(len(reads), 0)
            self.assertEqual(len(releases), len(reads))


@unittest.skipUnless(HAS_TORCH, "torch не установлен — кусковой путь проверяется "
                                "в окружении с .pylibs")
class TestChunkedEncoder(unittest.TestCase):
    """Кусковой кодировщик больших тензоров = тот же сегмент, что в qat.export."""

    def _tensor(self, rows, cols):
        import torch

        values = torch.arange(rows * cols, dtype=torch.float32)
        return torch.sin(values / 7.0).reshape(rows, cols)

    def test_chunked_matches_single_shot(self):
        import torch  # noqa: F401

        rows, cols = 37, 256  # 37 строк: куски не делят тензор нацело
        tensor = self._tensor(rows, cols)
        expected, expected_meta, _payload = qat_export.encode_trit_segment(
            tensor, "w", block_size=ternary.BLOCK_SIZE, scale_dtype="float32",
        )
        original = cold_export.CHUNK_MIN_ELEMENTS
        cold_export.CHUNK_MIN_ELEMENTS = 1  # заставить кусковой путь
        try:
            data, meta = cold_export.encode_trit_segment(
                tensor, "w", block_size=ternary.BLOCK_SIZE, scale_dtype="float32",
            )
        finally:
            cold_export.CHUNK_MIN_ELEMENTS = original
        self.assertEqual(len(data), len(expected))
        self.assertEqual(data, expected)
        for key in ("n_trits", "n_scale_blocks", "shape", "zero_fraction"):
            self.assertEqual(meta[key], expected_meta[key], key)

    def test_chunk_rows_keeps_word_alignment(self):
        for cols in (128, 896, 2048, 4864, 11008, 151936):
            step = cold_export._chunk_rows(100000, cols)
            self.assertGreater(step, 0)
            self.assertEqual((step * cols) % ternary.TRITS_PER_STORAGE_BLOCK, 0)


if __name__ == "__main__":
    unittest.main()
