"""kind 6/7 (Q4_K/Q6_K) в `.vtm`: кодек, контейнер, загрузчик и паритет C↔Python.

Задача 0165, ADR 0018. Проверяются независимые слои:

1. декодер блоков `vtm_ref.q4k_block_values`/`q6k_block_values` совпадает с
   `gguf_import._dequant_q4_k`/`_dequant_q6_k` (две независимые реализации),
   `q4k_scale_min` — с `_scale_min_k4`;
2. сегмент: заголовок 18 байт, блоки байт-в-байт, round-trip, отказы на
   чужом magic/версии/reserved/числе блоков и обрезанном сегменте;
3. контейнер: Q4_K/Q6_K остаются kind 6/7, `read_vtm` даёт окно блоков,
   `dense()` декодирует, размер файла = `estimated_file_size()`;
4. C-загрузчик: kind, `n_qblocks`, указатель на блоки, `f32 == NULL`;
5. нативный импорт GGUF (`--q4-kind q4_k`): записи `kind: q4_k|q6_k`,
   sha256 блоков сегмента = sha256 блоков GGUF (rel-RMS 0 по построению);
6. паритет C↔Python на собранной модели ≤1e-6.

Запуск:
    python3 -m unittest tests.test_q4k_native -v
"""

from __future__ import annotations

import hashlib
import math
import os
import random
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools import gguf_import as gi  # noqa: E402
from tools import parity_real, vtm_compile, vtm_ref  # noqa: E402

SRC_VTM = os.path.join(ROOT, "src", "vtm")
VTM_LIB = os.path.join(SRC_VTM, "libvtm.a")

Q4K = vtm_ref.KIND_Q4K
Q6K = vtm_ref.KIND_Q6K


# ---------------------------------------------------------------------------
# Синтетические блоки GGML
# ---------------------------------------------------------------------------


def q4k_blocks(n_values: int, rng: random.Random, d: float = 2e-4) -> bytes:
    """Случайные валидные блоки Q4_K (d/dmin конечны, sc/mn любые 6-битные)."""
    out = bytearray()
    for _ in range(n_values // vtm_ref.Q4K_BLOCK_VALUES):
        out += struct.pack("<ee", d, d * 0.5)
        out += bytes(rng.randrange(256) for _ in range(12))
        out += bytes(rng.randrange(256) for _ in range(128))
    return bytes(out)


def q6k_blocks(n_values: int, rng: random.Random, d: float = 5e-5) -> bytes:
    """Случайные валидные блоки Q6_K."""
    out = bytearray()
    for _ in range(n_values // vtm_ref.Q4K_BLOCK_VALUES):
        out += bytes(rng.randrange(256) for _ in range(128))
        out += bytes(rng.randrange(256) for _ in range(64))
        out += struct.pack(
            "<16b", *[rng.randrange(-128, 128) for _ in range(16)]
        )
        out += struct.pack("<e", d)
    return bytes(out)


class _TmpCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="q4k-native-")
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()


# ---------------------------------------------------------------------------
# Кодек
# ---------------------------------------------------------------------------


class TestQ4KCodec(unittest.TestCase):
    def test_scale_min_matches_gguf_import(self):
        rng = random.Random(3)
        scales = bytes(rng.randrange(256) for _ in range(12))
        for j in range(8):
            self.assertEqual(
                vtm_ref.q4k_scale_min(j, scales), gi._scale_min_k4(j, scales)
            )

    def test_block_decode_matches_import_dequant(self):
        rng = random.Random(7)
        for kind, spec, decode in (
            ("q4_k", gi.GGUF_TYPES[12], gi._dequant_q4_k),
            ("q6_k", gi.GGUF_TYPES[14], gi._dequant_q6_k),
        ):
            raw = q4k_blocks(256, rng) if kind == "q4_k" else q6k_blocks(256, rng)
            reference = decode(raw, spec)
            if kind == "q4_k":
                decoded = vtm_ref.q4k_block_values(raw)
            else:
                decoded = vtm_ref.q6k_block_values(raw)
            self.assertEqual(len(decoded), 256)
            self.assertEqual(decoded, list(reference))

    def test_segment_roundtrip_and_tail(self):
        rng = random.Random(11)
        for kind in (Q4K, Q6K):
            # 300 значений — 2 блока, второй с хвостом: читатель обрезает.
            blocks = (
                q4k_blocks(512, rng) if kind == Q4K else q6k_blocks(512, rng)
            )
            tensor = vtm_ref.Tensor.q4k("w", (300,), blocks, kind=kind)
            segment = vtm_ref.encode_q4k_segment(tensor)
            self.assertEqual(segment[:4], b"Q4K " if kind == Q4K else b"Q6K ")
            self.assertEqual(segment[4], 1)
            self.assertEqual(segment[5], 0)
            n_values, n_blocks = struct.unpack_from("<QI", segment, 6)
            self.assertEqual((n_values, n_blocks), (300, 2))
            window, decoded_kind = vtm_ref.decode_q4k_segment_ex(segment)
            self.assertEqual(decoded_kind, kind)
            self.assertEqual(len(window), 300)
            self.assertEqual(list(window), tensor.dense()[:300])
            # Блоки — те же байты, что на входе (без перепаковки).
            self.assertEqual(
                window.to_packed(), blocks[: 2 * vtm_ref.Q4K_BLOCK_BYTES]
                if kind == Q4K else blocks[: 2 * vtm_ref.Q6K_BLOCK_BYTES]
            )

    def test_segment_rejects_broken_headers(self):
        rng = random.Random(13)
        blocks = q4k_blocks(256, rng)
        good = vtm_ref.encode_q4k_segment(
            vtm_ref.Tensor.q4k("w", (256,), blocks)
        )
        bad_cases = {}
        bad = bytearray(good)
        bad[0:4] = b"Q6K "  # чужой magic для kind 6
        bad_cases["magic"] = bytes(bad)
        bad = bytearray(good)
        bad[4] = 2
        bad_cases["version"] = bytes(bad)
        bad = bytearray(good)
        bad[5] = 1
        bad_cases["reserved"] = bytes(bad)
        bad = bytearray(good)
        struct.pack_into("<I", bad, 14, 7)
        bad_cases["blocks"] = bytes(bad)
        bad_cases["truncated"] = good[:-1]
        for name, data in bad_cases.items():
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    vtm_ref.decode_q4k_segment_ex(data)
        with self.assertRaises(ValueError):
            vtm_ref.decode_q4k_segment_ex(good, vtm_ref.KIND_Q6K)

    def test_tensor_validates_shape(self):
        rng = random.Random(17)
        blocks = q4k_blocks(256, rng)
        with self.assertRaises(ValueError):
            vtm_ref.Tensor.q4k("w", (512,), blocks)  # нужно 2 блока
        with self.assertRaises(ValueError):
            vtm_ref.Tensor.q4k("w", (256,), blocks, kind=vtm_ref.KIND_Q4)


# ---------------------------------------------------------------------------
# Контейнер
# ---------------------------------------------------------------------------


class TestQ4KContainer(_TmpCase):
    def _model(self, kind: int, *, integrity: bool = False):
        rng = random.Random(19)
        rows, cols = 16, 256
        blocks = (
            q4k_blocks(rows * cols, rng) if kind == Q4K
            else q6k_blocks(rows * cols, rng)
        )
        w = vtm_ref.Tensor.q4k("w", (rows, cols), blocks, kind=kind)
        x = vtm_ref.Tensor.f32("x", (cols,), [0.5] * cols)
        y = vtm_ref.Tensor.f32("y", (rows,), [0.0] * rows)
        program = [
            vtm_ref.Instr(op=int(vtm_ref.Op.TMV), dst=2, a=0, b=1),
            vtm_ref.Instr(op=int(vtm_ref.Op.HALT)),
        ]
        path = self.tmp / f"m{kind}.vtm"
        model = vtm_ref.write_vtm(path, [w, x, y], program, {"kind": kind},
                                  integrity=integrity)
        return path, model, w

    def test_roundtrip_and_dense(self):
        for kind in (Q4K, Q6K):
            with self.subTest(kind=kind):
                path, model, w = self._model(kind)
                self.assertEqual(os.path.getsize(path), model.estimated_file_size())
                back = vtm_ref.read_vtm(path)
                got = back.by_name("w")
                self.assertEqual(got.kind, kind)
                self.assertEqual(got.kind_name, "q4_k" if kind == Q4K else "q6_k")
                self.assertEqual(got.block_size, vtm_ref.Q4K_BLOCK_VALUES)
                self.assertEqual(got.scale_offset, vtm_ref.NO_OFFSET)
                self.assertEqual(got.qdata.to_packed(), w.qdata.to_packed())
                self.assertEqual(got.dense(), w.dense())

    def test_streamed_write_is_byte_identical(self):
        """Потоковая запись (PackedQ4K) даёт те же байты, что encode-сегмент."""
        path, _model, w = self._model(Q4K)
        raw = vtm_ref.encode_q4k_segment(w)
        streamed = b"".join(vtm_ref.iter_q4k_payload(w))
        self.assertEqual(
            hashlib.sha256(raw).hexdigest(),
            hashlib.sha256(streamed).hexdigest(),
        )

    def test_python_execute_tmv(self):
        path, _model, w = self._model(Q6K)
        model = vtm_ref.read_vtm(path)
        result = vtm_ref.execute(model, [], 0)
        rows, cols = w.dims
        dense = w.dense()
        x = model.by_name("x").dense()
        want = [
            sum(dense[r * cols + c] * x[c] for c in range(cols))
            for r in range(rows)
        ]
        # execute() пишет результат операции во float32 (как арена C), поэтому
        # сравниваем с округлением эталона, а не с double-суммой.
        for got, expected in zip(result.get("y"), want):
            self.assertEqual(got, vtm_ref._f32(expected))


HARNESS_SOURCE = r"""
#include "vtm.h"

#include <stdio.h>

int main(int argc, char **argv) {
    vtm_t *m;
    const vtm_tensor *t;
    const uint8_t *p;
    uint32_t i, n;
    if (argc != 3) return 2;
    m = vtm_load(argv[1]);
    if (!m) {
        fprintf(stderr, "load: %s\n", vtm_last_error());
        return 1;
    }
    t = vtm_tensor_by_name(m, argv[2]);
    if (!t) {
        fprintf(stderr, "нет тензора %s\n", argv[2]);
        vtm_free(m);
        return 3;
    }
    printf("kind %u\n", (unsigned)t->kind);
    printf("sblock %u\n", (unsigned)t->scale_block);
    printf("f32 %u\n", t->f32 != NULL ? 1u : 0u);
    printf("nq %u\n", (unsigned)t->n_qblocks);
    n = t->n_qblocks * (t->kind == VTM_TENSOR_Q4K
                        ? VTM_Q4K_BLOCK_BYTES : VTM_Q6K_BLOCK_BYTES);
    p = t->qblocks;
    printf("blocks ");
    for (i = 0; i < n; i++) printf("%02x", p[i]);
    printf("\n");
    vtm_free(m);
    return 0;
}
"""


class TestQ4KCLoader(_TmpCase):
    """Python пишет `.vtm`, C-загрузчик читает блоки без fp32-копии."""

    @classmethod
    def setUpClass(cls):
        if not os.path.isdir(SRC_VTM):
            raise unittest.SkipTest("нет src/vtm")
        try:
            build = subprocess.run(
                ["make", "-C", SRC_VTM, "libvtm.a"],
                capture_output=True, text=True, timeout=300,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise unittest.SkipTest(f"make/libvtm.a недоступны: {exc}")
        if build.returncode != 0:
            raise unittest.SkipTest(
                "libvtm.a не собралась: " + build.stderr.strip()[:300]
            )

    def _probe(self, model_path: Path, name: str) -> dict[str, object]:
        source = self.tmp / "probe.c"
        source.write_text(HARNESS_SOURCE, encoding="utf-8")
        binary = self.tmp / "probe"
        compile_cmd = [
            os.environ.get("CC", "cc"), "-std=c99", "-O2", "-I", SRC_VTM,
            "-o", str(binary), str(source), VTM_LIB, "-lm", "-lpthread",
        ]
        proc = subprocess.run(compile_cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise unittest.SkipTest("C-проба не собралась: " + proc.stderr[:300])
        out = subprocess.run(
            [str(binary), str(model_path), name],
            capture_output=True, text=True,
        )
        self.assertEqual(out.returncode, 0, out.stderr)
        parsed: dict[str, object] = {}
        for line in out.stdout.splitlines():
            key, _, value = line.partition(" ")
            parsed[key] = value
        return parsed

    def test_loader_sees_blocks_without_fp32(self):
        rng = random.Random(23)
        for kind in (Q4K, Q6K):
            with self.subTest(kind=kind):
                rows, cols = 8, 256
                blocks = (
                    q4k_blocks(rows * cols, rng) if kind == Q4K
                    else q6k_blocks(rows * cols, rng)
                )
                w = vtm_ref.Tensor.q4k("w", (rows, cols), blocks, kind=kind)
                path = self.tmp / f"load{kind}.vtm"
                vtm_ref.write_vtm(
                    path, [w, vtm_ref.Tensor.f32("x", (cols,), [0.0] * cols)],
                    [vtm_ref.Instr(op=int(vtm_ref.Op.HALT))],
                )
                parsed = self._probe(path, "w")
                self.assertEqual(int(parsed["kind"]), kind)
                self.assertEqual(int(parsed["sblock"]), 256)
                self.assertEqual(int(parsed["f32"]), 0)
                self.assertEqual(int(parsed["nq"]), rows * cols // 256)
                self.assertEqual(
                    parsed["blocks"], blocks.hex()
                )


# ---------------------------------------------------------------------------
# Нативный импорт GGUF
# ---------------------------------------------------------------------------

TINY = {
    "hidden": 256,
    "intermediate": 512,
    "vocab": 256,
    "layers": 1,
    "context": 32,
    "heads": 2,
    "head_dim": 128,
    "kv_heads": 2,
}


def build_tiny_q4k_model(path: Path, *, seed: int = 31) -> Path:
    """Минимальный GGUF Q4_K_M-подобный: Linear — Q4_K, v_proj — Q6_K.

    Все размерности кратны 256, чтобы блоки Q4_K/Q6_K копировались целиком.
    """
    from tests.test_gguf_import import GgufBuilder

    rng = random.Random(seed)
    builder = GgufBuilder()
    hidden = TINY["hidden"]
    inter = TINY["intermediate"]
    vocab = TINY["vocab"]
    q_dim = TINY["heads"] * TINY["head_dim"]
    kv_dim = TINY["kv_heads"] * TINY["head_dim"]
    builder.add_metadata("general.architecture", "qwen2")
    builder.add_metadata("general.name", "Tiny Q4_K test")
    builder.add_metadata("qwen2.block_count", TINY["layers"])
    builder.add_metadata("qwen2.embedding_length", hidden)
    builder.add_metadata("qwen2.attention.head_count", TINY["heads"])
    builder.add_metadata("qwen2.attention.head_count_kv", TINY["kv_heads"])
    builder.add_metadata("qwen2.feed_forward_length", inter)
    builder.add_metadata("qwen2.context_length", TINY["context"])
    builder.add_metadata("qwen2.rope.freq_base", 10000.0)
    builder.add_metadata("qwen2.attention.layer_norm_rms_epsilon", 0.001)
    builder.add_metadata_strings(
        "tokenizer.ggml.tokens", [f"t{index}" for index in range(vocab)]
    )
    builder.add_tensor(
        "token_embd.weight", (hidden, vocab), 14, q6k_blocks(hidden * vocab, rng)
    )
    builder.add_f32("output_norm.weight", (hidden,), [1.0] * hidden)
    for layer in range(TINY["layers"]):
        prefix = f"blk.{layer}."
        builder.add_f32(f"{prefix}attn_norm.weight", (hidden,), [1.0] * hidden)
        builder.add_f32(f"{prefix}ffn_norm.weight", (hidden,), [1.0] * hidden)
        for name, dim in (
            ("attn_q.weight", q_dim),
            ("attn_k.weight", kv_dim),
            ("attn_output.weight", q_dim),
            ("ffn_gate.weight", inter),
            ("ffn_up.weight", inter),
        ):
            builder.add_tensor(
                f"{prefix}{name}", (hidden, dim), 12,
                q4k_blocks(hidden * dim, rng),
            )
        # Q4_K_M держит часть тензоров в Q6_K — проверяем kind 7.
        builder.add_tensor(
            f"{prefix}attn_v.weight", (hidden, kv_dim), 14,
            q6k_blocks(hidden * kv_dim, rng),
        )
        builder.add_tensor(
            f"{prefix}ffn_down.weight", (inter, hidden), 12,
            q4k_blocks(inter * hidden, rng),
        )
    return builder.write(path)


class TestNativeImport(_TmpCase):
    def _import(self, **kwargs):
        source = build_tiny_q4k_model(self.tmp / "tiny.gguf")
        out = self.tmp / "export"
        result = gi.import_gguf(
            source, out, fmt="q4", q4_kind="q4_k", embed_mode="int8", **kwargs
        )
        return source, out, result

    def test_manifest_is_native(self):
        source, out, result = self._import()
        q4 = result.manifest["q4"]
        self.assertTrue(q4["native"])
        self.assertEqual(q4["kind"], 6)
        self.assertEqual(q4["kinds"], {"q4_k": 6, "q6_k": 7})
        kinds = {
            entry["kind"] for entry in result.manifest["tensors"]
            if entry["kind"] in ("q4_k", "q6_k")
        }
        self.assertEqual(kinds, {"q4_k", "q6_k"})
        self.assertGreater(q4["q4k_tensors"], 0)
        self.assertGreater(q4["q6k_tensors"], 0)

    def test_blocks_are_byte_identical_and_decode_to_gguf(self):
        """Сегмент = байты GGUF; декодер контейнера = декодер импортёра."""
        source, out, result = self._import()
        gguf = gi.read_gguf(source)
        q4_raw = (out / "q4.bin").read_bytes()
        checked = 0
        for entry in result.manifest["tensors"]:
            if entry["kind"] not in ("q4_k", "q6_k"):
                continue
            offset = entry["q4_offset"]
            length = entry["q4_length"]
            segment = q4_raw[offset:offset + length]
            info = gguf.tensor(entry["source_tensor"])
            raw = gguf.raw_tensor(info)
            self.assertEqual(
                hashlib.sha256(segment[18:]).hexdigest(),
                hashlib.sha256(raw).hexdigest(),
                entry["name"],
            )
            window, kind = vtm_ref.decode_q4k_segment_ex(segment)
            dense = list(window)
            reference = gi._dequantize(info.spec, raw, info.n_elements)
            self.assertEqual(dense, list(reference), entry["name"])
            checked += 1
        self.assertGreater(checked, 5)

    def test_auto_picks_native_when_source_is_q4k(self):
        source = build_tiny_q4k_model(self.tmp / "auto.gguf")
        out = self.tmp / "auto"
        result = gi.import_gguf(source, out, fmt="q4", q4_kind="auto")
        self.assertEqual(result.manifest["quantization"]["q4_kind"], "q4_k")
        self.assertTrue(result.manifest["q4"]["native"])

    def test_explicit_q4_k_refuses_non_k_source(self):
        """Явный q4_k на источнике Q4_0 — отказ с именем тензора, не тихий q4_1."""
        from tests.test_gguf_import import build_tiny_model

        source = build_tiny_model(self.tmp / "q40.gguf")
        out = self.tmp / "q40"
        with self.assertRaises(gi.GgufError) as ctx:
            gi.import_gguf(source, out, fmt="q4", q4_kind="q4_k")
        self.assertIn("не все Linear/lm_head", str(ctx.exception))

    def test_compile_and_python_execute(self):
        _source, out, _result = self._import()
        export = vtm_compile.load_export(out / "manifest.json")
        model = vtm_compile.compile_model(
            export.cfg, export.weights, export.meta
        )
        path = self.tmp / "tiny.vtm"
        vtm_ref.write_vtm(
            path, model.tensors, model.program, model.meta
        )
        back = vtm_ref.read_vtm(path)
        kinds = {tensor.kind for tensor in back.tensors}
        self.assertIn(Q4K, kinds)
        self.assertIn(Q6K, kinds)
        result = vtm_ref.execute(path, tokens=[1], max_new=0)
        self.assertEqual(len(result.logits), export.cfg.vocab)


class TestQ4KParity(unittest.TestCase):
    """C↔Python на собранной модели: ≤1e-6 (задача 0165)."""

    @classmethod
    def setUpClass(cls):
        if not os.path.isdir(SRC_VTM):
            raise unittest.SkipTest("нет src/vtm")
        build = subprocess.run(
            ["make", "-C", SRC_VTM, "libvtm.a"],
            capture_output=True, text=True, timeout=300,
        )
        if build.returncode != 0:
            raise unittest.SkipTest("libvtm.a не собралась")
        cls.cc = parity_real.find_c_compiler()
        if cls.cc is None:
            raise unittest.SkipTest("нет C-компилятора")

    def test_c_python_logits(self):
        with tempfile.TemporaryDirectory(prefix="q4k-parity-") as tmp:
            tmp_path = Path(tmp)
            source = build_tiny_q4k_model(tmp_path / "tiny.gguf")
            out = tmp_path / "export"
            gi.import_gguf(source, out, fmt="q4", q4_kind="q4_k",
                           embed_mode="int8")
            export = vtm_compile.load_export(out / "manifest.json")
            model = vtm_compile.compile_model(
                export.cfg, export.weights, export.meta
            )
            path = tmp_path / "tiny.vtm"
            vtm_ref.write_vtm(path, model.tensors, model.program, model.meta)
            tokens = [1, 2]
            expected = vtm_ref.execute(str(path), tokens=tokens, max_new=0)
            probe = parity_real.run_c_probe(tmp_path, path, tokens, self.cc)
            self.assertEqual(len(probe.logits), len(expected.logits))
            delta = max(
                abs(float(left) - float(right))
                for left, right in zip(expected.logits, probe.logits)
            )
            self.assertLessEqual(delta, 1e-6, f"max|Δ логитов| = {delta}")


if __name__ == "__main__":
    unittest.main()
