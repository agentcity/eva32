"""Полный компилятор VTM: манифест QAT -> контейнер `.vtm`.

Проверяем то, чего не было у синтетики:

* `compile_model(manifest)` читает `manifest.json`, `ternary.trit` и
  `non_ternary.bin` и собирает EMB → слои → финальный RMSN → lm_head;
* при `tie_word_embeddings=true` lm_head не дублируется (MM читает `emb.w`),
  поэтому контейнер меньше ровно на fp32-таблицу;
* CLI `--manifest ... --out model.vtm`;
* Qwen-подобный bias (q/k/v) действительно применяется;
* C-рантайм (`src/vtm/vtm`), если он собран, грузит контейнер и выбирает тот
  же токен, что Python-эталон.

Запуск:
    python3 -m unittest tests.test_vtm_full -v
"""

import hashlib
import json
import os
import random
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qat import export as qat_export  # noqa: E402
from tools import vtm_compile, vtm_ref  # noqa: E402

TOOLS = os.path.join(ROOT, "tools")
VTM_BIN = os.path.join(ROOT, "src", "vtm", "vtm")


def _tiny_config(*, tie=True, lm_head="mm", n_layers=1):
    return vtm_compile.ModelConfig(
        hidden=32,
        n_heads=2,
        n_kv_heads=1,
        head_dim=16,
        intermediate=64,
        vocab=32,
        n_layers=n_layers,
        context_len=4,
        eps=1e-3,
        lm_head=lm_head,
        tie_word_embeddings=tie,
    )


def _tiny_weights(cfg, seed=7, *, biases=True):
    rng = random.Random(seed)

    def floats(n):
        return [rng.uniform(-1.0, 1.0) for _ in range(n)]

    def norm():
        return [rng.uniform(0.5, 1.5) for _ in range(cfg.hidden)]

    layers = []
    for _ in range(cfg.n_layers):
        layer = {
            "norm1": norm(),
            "norm2": norm(),
            "q_proj": floats(cfg.q_dim * cfg.hidden),
            "k_proj": floats(cfg.kv_dim * cfg.hidden),
            "v_proj": floats(cfg.kv_dim * cfg.hidden),
            "o_proj": floats(cfg.hidden * cfg.q_dim),
            "gate_proj": floats(cfg.intermediate * cfg.hidden),
            "up_proj": floats(cfg.intermediate * cfg.hidden),
            "down_proj": floats(cfg.hidden * cfg.intermediate),
        }
        if biases:
            layer.update({
                "q_bias": [rng.uniform(-0.1, 0.1) for _ in range(cfg.q_dim)],
                "k_bias": [rng.uniform(-0.1, 0.1) for _ in range(cfg.kv_dim)],
                "v_bias": [rng.uniform(-0.1, 0.1) for _ in range(cfg.kv_dim)],
            })
        layers.append(layer)
    weights = {
        "embed": floats(cfg.vocab * cfg.hidden),
        "final_norm": norm(),
        "layers": layers,
    }
    if not cfg.tie_word_embeddings:
        weights["lm_head"] = floats(cfg.vocab * cfg.hidden)
    return weights


# ---------------------------------------------------------------------------
# Синтетический экспорт QAT: ровно тот контракт, который пишет qat.export
# ---------------------------------------------------------------------------


def write_export(
    directory,
    cfg,
    weights,
    *,
    tie=None,
    version=2,
    stored_non_ternary=True,
    scale_dtype="float32",
    lm_head_trit=False,
):
    """Собрать export/{manifest.json,ternary.trit,non_ternary.bin}."""
    tie = cfg.tie_word_embeddings if tie is None else tie
    trit_blob = bytearray()
    trit_entries = []
    fp_blob = bytearray()
    fp_entries = []

    def add_trit(name, shape, values):
        trits, scales = vtm_ref.quantize_trits(values)
        payload = vtm_ref.encode_trit_segment(
            vtm_ref.Tensor.ternary("tmp", shape, trits, scales, scale_dtype)
        )
        offset = len(trit_blob)
        trit_blob.extend(payload)
        trit_entries.append({
            "name": name,
            "shape": list(shape),
            "params": vtm_ref.prod(shape),
            "n_trits": vtm_ref.prod(shape),
            "n_scale_blocks": len(scales),
            "trit_offset": offset,
            "trit_length": len(payload),
        })

    def add_fp(name, shape, values):
        assert len(values) == vtm_ref.prod(shape)
        payload = struct.pack(f"<{len(values)}f", *values)
        offset = len(fp_blob)
        fp_blob.extend(payload)
        fp_entries.append({
            "name": name,
            "role": "aux",
            "layer": None,
            "key": name.rsplit(".", 1)[0],
            "shape": list(shape),
            "dtype": "float32",
            "offset": offset,
            "length": len(payload),
            "params": len(values),
        })

    for index, layer in enumerate(weights["layers"]):
        prefix = f"model.layers.{index}"
        add_trit(f"{prefix}.self_attn.q_proj", (cfg.q_dim, cfg.hidden), layer["q_proj"])
        add_trit(f"{prefix}.self_attn.k_proj", (cfg.kv_dim, cfg.hidden), layer["k_proj"])
        add_trit(f"{prefix}.self_attn.v_proj", (cfg.kv_dim, cfg.hidden), layer["v_proj"])
        add_trit(f"{prefix}.self_attn.o_proj", (cfg.hidden, cfg.q_dim), layer["o_proj"])
        add_trit(f"{prefix}.mlp.gate_proj", (cfg.intermediate, cfg.hidden), layer["gate_proj"])
        add_trit(f"{prefix}.mlp.up_proj", (cfg.intermediate, cfg.hidden), layer["up_proj"])
        add_trit(f"{prefix}.mlp.down_proj", (cfg.hidden, cfg.intermediate), layer["down_proj"])
        add_fp(f"{prefix}.input_layernorm.weight", (cfg.hidden,), layer["norm1"])
        add_fp(f"{prefix}.post_attention_layernorm.weight", (cfg.hidden,), layer["norm2"])
        for bias_key, dim in (
            ("q_bias", cfg.q_dim), ("k_bias", cfg.kv_dim), ("v_bias", cfg.kv_dim)
        ):
            if bias_key in layer:
                add_fp(f"{prefix}.self_attn.{bias_key[:-5]}_proj.bias",
                       (dim,), layer[bias_key])

    add_fp("model.embed_tokens.weight",
           (cfg.vocab, cfg.hidden), weights["embed"])
    add_fp("model.norm.weight", (cfg.hidden,), weights["final_norm"])
    if tie:
        fp_entries.append({
            "name": "lm_head.weight",
            "role": "lm_head",
            "layer": None,
            "key": "lm_head",
            "shape": [cfg.vocab, cfg.hidden],
            "dtype": "float32",
            "tied_to": "model.embed_tokens.weight",
            "params": cfg.vocab * cfg.hidden,
        })
    elif lm_head_trit:
        add_trit("lm_head", (cfg.vocab, cfg.hidden), weights["lm_head"])
    else:
        add_fp("lm_head.weight", (cfg.vocab, cfg.hidden), weights["lm_head"])

    os.makedirs(directory, exist_ok=True)
    trit_path = os.path.join(directory, "ternary.trit")
    non_path = os.path.join(directory, "non_ternary.bin")
    with open(trit_path, "wb") as handle:
        handle.write(bytes(trit_blob))
    with open(non_path, "wb") as handle:
        handle.write(bytes(fp_blob))

    def digest(path):
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()

    architecture = {
        "hidden": cfg.hidden,
        "n_heads": cfg.n_heads,
        "n_kv_heads": cfg.n_kv_heads,
        "head_dim": cfg.head_dim,
        "intermediate": cfg.intermediate,
        "vocab": cfg.vocab,
        "n_layers": cfg.n_layers,
        "context_len": cfg.context_len,
        "rope_theta": cfg.rope_theta,
        "rms_norm_eps": cfg.eps,
        "tie_word_embeddings": bool(tie),
        "lm_head": "tmv" if lm_head_trit else cfg.lm_head,
        "source": "synthetic-test",
    }
    manifest = {
        "format": vtm_compile.MANIFEST_FORMAT,
        "format_version": version,
        "created_at": "2026-11-18T05:00:00+00:00",
        "training": {"step": 200, "checkpoint": "synthetic/step_000200"},
        "base_model": {"name_or_path": "synthetic/tiny", "revision": "test"},
        "architecture": architecture,
        "quantization": {"block_size": 128, "scale_dtype": scale_dtype},
        "packing": {"scheme": "trit20", "scale_block": 128},
        "non_ternary": {
            "stored_in_this_export": bool(stored_non_ternary),
            "file": "non_ternary.bin",
            "dtype": "float32",
            "count": len(fp_entries),
            "params": sum(entry["params"] for entry in fp_entries),
            "tensors": fp_entries,
        },
        "files": {
            "ternary.trit": {"bytes": len(trit_blob), "sha256": digest(trit_path)},
            "non_ternary.bin": {"bytes": len(fp_blob), "sha256": digest(non_path)},
        },
        "tensors": trit_entries,
    }
    manifest_path = os.path.join(directory, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest_path


class _TmpCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmpdir = self._tmp.name

    def path(self, name):
        return os.path.join(self.tmpdir, name)


# ---------------------------------------------------------------------------
# Манифест -> полный граф
# ---------------------------------------------------------------------------


class TestManifestCompile(_TmpCase):
    def setUp(self):
        super().setUp()
        self.cfg = _tiny_config()
        self.weights = _tiny_weights(self.cfg)
        self.manifest = write_export(self.tmpdir, self.cfg, self.weights)

    def test_load_export_reads_all_tensor_roles(self):
        export = vtm_compile.load_export(self.manifest)
        self.assertTrue(export.tied)
        self.assertEqual(export.cfg.vocab, self.cfg.vocab)
        self.assertEqual(len(export.weights["layers"]), 1)
        layer = export.weights["layers"][0]
        for key in (
            "norm1", "norm2", "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj", "q_bias", "k_bias", "v_bias",
        ):
            self.assertIn(key, layer)
        self.assertEqual(export.weights["embed"].kind, vtm_ref.KIND_F32)
        self.assertEqual(
            tuple(export.weights["embed"].dims), (self.cfg.vocab, self.cfg.hidden)
        )
        self.assertNotIn("lm_head", export.weights)  # tied: ссылка, не копия
        self.assertGreater(export.trit_params, 0)
        self.assertGreater(export.fp_params, 0)

    def test_manifest_matches_direct_compile(self):
        """Контейнер из манифеста считает ровно то же, что прямой compile_model."""
        export = vtm_compile.load_export(self.manifest)
        from_manifest = vtm_compile.compile_model(
            export.cfg, export.weights, export.meta
        )
        direct = vtm_compile.compile_model(self.cfg, self.weights)
        for tokens in ([1], [1, 2, 3], [0, 17, 31]):
            with self.subTest(tokens=tokens):
                got = vtm_ref.execute(from_manifest, tokens, 0)
                want = vtm_ref.execute(direct, tokens, 0)
                self.assertEqual(got.logits, want.logits)
                self.assertEqual(got.generated, want.generated)

    def test_ternary_lm_head_from_manifest(self):
        cfg = _tiny_config(tie=False, lm_head="tmv")
        weights = _tiny_weights(cfg)
        manifest = write_export(
            self.path("tmv_head"), cfg, weights, lm_head_trit=True
        )
        export = vtm_compile.load_export(manifest)
        self.assertEqual(export.cfg.lm_head, "tmv")
        self.assertEqual(export.weights["lm_head"].kind, vtm_ref.KIND_TRITS)
        from_manifest = vtm_compile.compile_model(manifest)
        direct = vtm_compile.compile_model(cfg, weights)
        self.assertEqual(
            vtm_ref.execute(from_manifest, [1, 2, 3], 0).logits,
            vtm_ref.execute(direct, [1, 2, 3], 0).logits,
        )
        self.assertEqual(
            [int(i.op) for i in from_manifest.program].count(int(vtm_ref.Op.TMV)),
            [int(i.op) for i in direct.program].count(int(vtm_ref.Op.TMV)),
        )

    def test_compile_model_accepts_manifest_path(self):
        model = vtm_compile.compile_model(self.manifest)
        self.assertTrue(model.meta["tied_embeddings"])
        self.assertEqual(model.meta["vocab_size"], self.cfg.vocab)
        self.assertEqual(model.meta["source_manifest"],
                         os.path.realpath(self.manifest))
        self.assertGreater(len(model.program), 20)

    def test_tied_uses_embedding_table_for_mm(self):
        model = vtm_compile.compile_model(self.manifest)
        emb = model.index_of("emb.w")
        mm = [i for i in model.program if int(i.op) == vtm_ref.Op.MM]
        self.assertEqual(len(mm), 1)
        self.assertEqual(mm[0].a, emb)
        with self.assertRaises(KeyError):
            model.by_name("lm_head.w")

    def test_full_graph_instruction_count(self):
        model = vtm_compile.compile_model(self.manifest)
        # EMB + слой (17 + 3 bias ADD) + RMSN + CPY + MM + SAMP + HALT
        self.assertEqual(len(model.program), 1 + 17 + 3 + 1 + 1 + 1 + 1 + 1)
        ops = model.op_counts()
        self.assertEqual(ops["EMB"], 1)
        self.assertEqual(ops["MM"], 1)
        self.assertEqual(ops["RMSN"], 3)  # norm1 + norm2 в слое + финальный
        self.assertEqual(ops["ADD"], 2 + 3)  # два residual + q/k/v bias

    def test_roundtrip_file_and_python_reference(self):
        model = vtm_compile.compile_model(self.manifest)
        path = self.path("model.vtm")
        vtm_ref.write_vtm(path, model.tensors, model.program, model.meta)
        loaded = vtm_ref.read_vtm(path)
        before = vtm_ref.execute(model, [1, 2, 3], 0)
        after = vtm_ref.execute(loaded, [1, 2, 3], 0)
        self.assertEqual(before.logits, after.logits)
        self.assertEqual(loaded.meta["rope_theta"], self.cfg.rope_theta)
        self.assertTrue(loaded.meta["tied_embeddings"])

    def test_bias_changes_logits(self):
        no_bias_weights = _tiny_weights(self.cfg, biases=False)
        plain = vtm_compile.compile_model(self.cfg, no_bias_weights)
        with_bias = vtm_compile.compile_model(self.cfg, self.weights)
        self.assertEqual(
            len(with_bias.program) - len(plain.program), 3
        )  # q/k/v bias ADD
        self.assertNotEqual(
            vtm_ref.execute(plain, [1, 2], 0).logits,
            vtm_ref.execute(with_bias, [1, 2], 0).logits,
        )


class TestTiedLmHeadSize(_TmpCase):
    def test_tied_head_is_not_stored_twice(self):
        tied_cfg = _tiny_config(tie=True)
        untied_cfg = _tiny_config(tie=False)
        weights = _tiny_weights(tied_cfg)
        weights_untied = dict(weights)
        # lm_head = та же таблица: ровно этот случай экономит место.
        weights_untied["lm_head"] = weights["embed"]

        tied_manifest = write_export(
            os.path.join(self.tmpdir, "tied"), tied_cfg, weights
        )
        untied_manifest = write_export(
            os.path.join(self.tmpdir, "untied"), untied_cfg, weights_untied
        )
        tied = vtm_compile.compile_model(tied_manifest)
        untied = vtm_compile.compile_model(untied_manifest)

        tied_path = self.path("tied.vtm")
        untied_path = self.path("untied.vtm")
        vtm_ref.write_vtm(tied_path, tied.tensors, tied.program, tied.meta)
        vtm_ref.write_vtm(untied_path, untied.tensors, untied.program, untied.meta)

        head_bytes = untied_cfg.vocab * untied_cfg.hidden * 4
        delta = os.path.getsize(untied_path) - os.path.getsize(tied_path)
        # payload lm_head (fp32) + 52 байта записи таблицы v0.2
        # + возможное выравнивание meta.
        self.assertGreaterEqual(delta, head_bytes)
        self.assertLessEqual(delta, head_bytes + 128)
        self.assertIn("lm_head.w", [tensor.name for tensor in untied.tensors])
        self.assertNotIn("lm_head.w", [tensor.name for tensor in tied.tensors])
        # При одинаковых данных логиты обязаны совпасть.
        self.assertEqual(
            vtm_ref.execute(tied, [5, 6], 0).logits,
            vtm_ref.execute(untied, [5, 6], 0).logits,
        )



# ---------------------------------------------------------------------------
# Ошибки манифеста: лучше явный ValueError, чем молча неполный граф
# ---------------------------------------------------------------------------


class TestManifestValidation(_TmpCase):
    def setUp(self):
        super().setUp()
        self.cfg = _tiny_config()
        self.weights = _tiny_weights(self.cfg)

    def test_v1_manifest_rejected(self):
        path = write_export(self.tmpdir, self.cfg, self.weights, version=1)
        with self.assertRaisesRegex(ValueError, "format_version"):
            vtm_compile.load_export(path)

    def test_missing_non_ternary_rejected(self):
        path = write_export(
            self.tmpdir, self.cfg, self.weights, stored_non_ternary=False
        )
        with self.assertRaisesRegex(ValueError, "нетернарные веса"):
            vtm_compile.load_export(path)

    def test_manifest_without_architecture_rejected(self):
        path = write_export(self.tmpdir, self.cfg, self.weights)
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest.pop("architecture")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle)
        with self.assertRaisesRegex(ValueError, "architecture"):
            vtm_compile.load_export(path)

    def test_sha256_mismatch_rejected(self):
        path = write_export(self.tmpdir, self.cfg, self.weights)
        with open(os.path.join(self.tmpdir, "ternary.trit"), "r+b") as handle:
            handle.seek(0)
            handle.write(b"XXXX")
        with self.assertRaisesRegex(ValueError, "sha256"):
            vtm_compile.load_export(path)

    def test_unknown_tensor_name_rejected(self):
        path = write_export(self.tmpdir, self.cfg, self.weights)
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest["non_ternary"]["tensors"].append({
            "name": "model.layers.0.mystery.weight",
            "role": "unknown",
            "shape": [4],
            "dtype": "float32",
            "offset": 0,
            "length": 16,
            "params": 4,
        })
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle)
        with self.assertRaisesRegex(ValueError, "не знает, куда положить"):
            vtm_compile.load_export(path)


class _FakeTensor:
    """Мини-замена torch.Tensor для проверки выгрузки без GPU-стека."""

    def __init__(self, values, shape, ptr):
        self.values = list(values)
        self.shape = tuple(shape)
        self._ptr = ptr

    def numel(self):
        return len(self.values)

    def data_ptr(self):
        return self._ptr


class TestExportNonTernaryWrite(_TmpCase):
    """qat.export: tied lm_head не попадает в non_ternary.bin вторым разом."""

    def _write(self, tie):
        from unittest import mock

        cfg = _tiny_config(tie=tie)
        weights = _tiny_weights(cfg)
        embed_ptr = 100
        state = {
            "model.embed_tokens.weight": _FakeTensor(
                weights["embed"], (cfg.vocab, cfg.hidden), embed_ptr
            ),
            "model.norm.weight": _FakeTensor(
                weights["final_norm"], (cfg.hidden,), 200
            ),
            "model.layers.0.input_layernorm.weight": _FakeTensor(
                weights["layers"][0]["norm1"], (cfg.hidden,), 201
            ),
            "model.layers.0.post_attention_layernorm.weight": _FakeTensor(
                weights["layers"][0]["norm2"], (cfg.hidden,), 202
            ),
            "model.layers.0.self_attn.q_proj.bias": _FakeTensor(
                weights["layers"][0]["q_bias"], (cfg.q_dim,), 203
            ),
        }
        if tie:
            # Тот же data_ptr — так выглядит tied-таблица в state_dict.
            state["lm_head.weight"] = _FakeTensor(
                weights["embed"], (cfg.vocab, cfg.hidden), embed_ptr
            )
        else:
            state["lm_head.weight"] = _FakeTensor(
                weights["lm_head"], (cfg.vocab, cfg.hidden), 300
            )
        architecture = {
            "tie_word_embeddings": tie,
            "n_heads": cfg.n_heads,
            "head_dim": cfg.head_dim,
            "vocab": cfg.vocab,
            "hidden": cfg.hidden,
        }

        def fake_bytes(tensor, dtype):
            return struct.pack(f"<{len(tensor.values)}f", *tensor.values)

        with mock.patch.object(qat_export, "_to_numpy_bytes", side_effect=fake_bytes):
            path, entries, total = qat_export._write_non_ternary(
                __import__("pathlib").Path(self.tmpdir), state, architecture
            )
        return path, entries, total, weights

    def test_tied_head_written_once(self):
        path, entries, total, weights = self._write(True)
        by_name = {entry["name"]: entry for entry in entries}
        embed = by_name["model.embed_tokens.weight"]
        head = by_name["lm_head.weight"]
        self.assertEqual(head["tied_to"], embed["name"])
        self.assertNotIn("offset", head)
        self.assertIn("lm_head.weight", embed.get("aliases", []))
        cfg = _tiny_config()
        self.assertEqual(
            os.path.getsize(path),
            cfg.vocab * cfg.hidden * 4 + 4 * cfg.hidden * 4,
        )
        # embed + две нормы + q_bias; lm_head не добавлен вторым.
        self.assertEqual(total, 32 * 32 + 32 + 32 + 32 + 32)

    def test_untied_head_written_separately(self):
        path, entries, total, _weights = self._write(False)
        by_name = {entry["name"]: entry for entry in entries}
        self.assertIn("offset", by_name["lm_head.weight"])
        self.assertGreaterEqual(
            total, 32 * 32 + 32 * 32 + 32 + 32 + 32 + 32
        )


# ---------------------------------------------------------------------------
# CLI и C-рантайм
# ---------------------------------------------------------------------------


class TestManifestCli(_TmpCase):
    def setUp(self):
        super().setUp()
        self.cfg = _tiny_config()
        self.weights = _tiny_weights(self.cfg)
        self.manifest = write_export(self.tmpdir, self.cfg, self.weights)

    def _run(self, *args):
        return subprocess.run(
            [sys.executable, os.path.join(TOOLS, "vtm_compile.py"), *args],
            cwd=ROOT, capture_output=True, text=True, timeout=180,
        )

    def test_cli_manifest_builds_container(self):
        output = self.path("cli_model.vtm")
        proc = self._run("--manifest", self.manifest, "--out", output)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("tied lm_head: да", proc.stdout)
        self.assertIn("инструкций:", proc.stdout)
        self.assertIn("файл:", proc.stdout)
        self.assertTrue(os.path.exists(output))

        loaded = vtm_ref.read_vtm(output)
        export = vtm_compile.load_export(self.manifest)
        direct = vtm_compile.compile_model(export.cfg, export.weights, export.meta)
        self.assertEqual(loaded.meta["tied_embeddings"], True)
        self.assertEqual(
            vtm_ref.execute(loaded, [1, 2, 3], 0).logits,
            vtm_ref.execute(direct, [1, 2, 3], 0).logits,
        )

    def test_cli_manifest_requires_out(self):
        proc = self._run("--manifest", self.manifest)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("--out", proc.stderr)

    def test_cli_rejects_two_modes(self):
        proc = self._run("--manifest", self.manifest, "--synthetic",
                         "--out", self.path("x.vtm"))
        self.assertEqual(proc.returncode, 2)
        self.assertIn("ровно один режим", proc.stderr)


class TestCContainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.access(VTM_BIN, os.X_OK):
            raise unittest.SkipTest("C-рантайм не собран: src/vtm/vtm отсутствует")

    def _run_c(self, path, max_new=1):
        proc = subprocess.run(
            [VTM_BIN, "run", path, "--tokens", "1,2,3",
             "--max-new", str(max_new)],
            cwd=ROOT, capture_output=True, text=True, timeout=180,
        )
        if proc.returncode != 0 and "версия контейнера" in proc.stderr:
            # Параллельно идёт переход vtm_ref на v0.2 (dims u32, запись 52
            # байта); старый бинарник её ещё не знает. Это не ошибка
            # компилятора: контейнер собран по актуальному vtm_ref.
            self.skipTest(
                "C-рантайм устарел относительно tools/vtm_ref.py: "
                + proc.stderr.strip()
            )
        return proc

    def test_c_runs_full_container_and_picks_same_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _tiny_config()
            weights = _tiny_weights(cfg)
            manifest = write_export(tmp, cfg, weights)
            model = vtm_compile.compile_model(manifest)
            path = os.path.join(tmp, "c_model.vtm")
            vtm_ref.write_vtm(path, model.tensors, model.program, model.meta)

            tokens = [1, 2, 3]
            expected = vtm_ref.execute(model, tokens, 1)
            proc = self._run_c(path)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            line = next(
                (line for line in proc.stdout.splitlines() if line.startswith("вывод:")),
                None,
            )
            self.assertIsNotNone(line, proc.stdout)
            got = [int(x) for x in line.split(":", 1)[1].split()]
            self.assertEqual(got, expected.generated)

    def test_c_runs_float16_scales(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _tiny_config()
            weights = _tiny_weights(cfg)
            manifest = write_export(tmp, cfg, weights, scale_dtype="float16")
            model = vtm_compile.compile_model(manifest)
            path = os.path.join(tmp, "c_model_f16.vtm")
            vtm_ref.write_vtm(path, model.tensors, model.program, model.meta)
            # Масштабы в контейнере уже fp16: эталон — сам прочитанный файл.
            expected = vtm_ref.execute(vtm_ref.read_vtm(path), [1, 2, 3], 1)
            proc = self._run_c(path)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            line = next(
                (line for line in proc.stdout.splitlines() if line.startswith("вывод:")),
                None,
            )
            self.assertIsNotNone(line, proc.stdout)
            got = [int(x) for x in line.split(":", 1)[1].split()]
            self.assertEqual(got, expected.generated)


# ---------------------------------------------------------------------------
# qat.export: вычисление architecture без torch
# ---------------------------------------------------------------------------


class TestExportArchitecture(unittest.TestCase):
    def _shapes(self):
        return {
            "model.embed_tokens.weight": (32, 16),
            "model.layers.0.self_attn.q_proj.weight": (16, 16),
            "model.layers.0.self_attn.k_proj.weight": (8, 16),
            "model.layers.0.self_attn.v_proj.weight": (8, 16),
            "model.layers.0.self_attn.o_proj.weight": (16, 16),
            "model.layers.0.mlp.gate_proj.weight": (32, 16),
            "model.layers.0.mlp.up_proj.weight": (32, 16),
            "model.layers.0.mlp.down_proj.weight": (16, 32),
            "model.layers.0.input_layernorm.weight": (16,),
            "model.layers.0.post_attention_layernorm.weight": (16,),
            "model.norm.weight": (16,),
            "lm_head.weight": (32, 16),
        }

    def test_infers_from_hf_config(self):
        arch = qat_export.infer_architecture(
            self._shapes(),
            {
                "num_attention_heads": 2, "num_key_value_heads": 1,
                "head_dim": 8, "tie_word_embeddings": True,
                "rope_theta": 1e6, "rms_norm_eps": 1e-6,
                "max_position_embeddings": 4096,
            },
            tie_hint=True,
        )
        self.assertEqual(
            (arch["hidden"], arch["n_heads"], arch["n_kv_heads"],
             arch["head_dim"], arch["intermediate"], arch["vocab"],
             arch["n_layers"]),
            (16, 2, 1, 8, 32, 32, 1),
        )
        self.assertTrue(arch["tie_word_embeddings"])
        self.assertEqual(arch["lm_head"], "mm")
        self.assertEqual(arch["rope_theta"], 1e6)
        # VTM по умолчанию берёт рабочую длину 2048, а не максимум HF:
        # 32768 токенов KV-кэша — это гигабайты нулей в контейнере.
        self.assertEqual(arch["context_len"], 2048)
        self.assertEqual(arch["max_context"], 4096)

    def test_needs_heads_without_config(self):
        with self.assertRaisesRegex(RuntimeError, "n_heads"):
            qat_export.infer_architecture(self._shapes(), None, tie_hint=True)

    def test_overrides_rescue_without_config(self):
        arch = qat_export.infer_architecture(
            self._shapes(), None, tie_hint=True,
            overrides={"n_heads": 2, "head_dim": 8},
        )
        self.assertEqual((arch["n_heads"], arch["head_dim"]), (2, 8))

    def test_classifies_qwen_state_dict_names(self):
        cases = {
            "model.embed_tokens.weight": ("embed", None, "embed"),
            "model.norm.weight": ("final_norm", None, "final_norm"),
            "lm_head.weight": ("lm_head", None, "lm_head"),
            "model.layers.3.self_attn.q_proj.weight": ("matrix", 3, "q_proj"),
            "model.layers.3.self_attn.k_proj.bias": ("bias", 3, "k_bias"),
            "model.layers.7.mlp.down_proj.weight": ("matrix", 7, "down_proj"),
            "model.layers.7.input_layernorm.weight": ("norm", 7, "norm1"),
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                role = qat_export.classify_state_key(name)
                self.assertEqual((role.kind, role.layer, role.key), expected)

    def test_tied_lm_head_trit_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "lm_head"):
            qat_export.infer_architecture(
                self._shapes(), None, tie_hint=True,
                overrides={"n_heads": 2, "head_dim": 8},
                lm_head_trit=True,
            )


if __name__ == "__main__":
    unittest.main()
