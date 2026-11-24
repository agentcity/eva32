"""Импорт GGUF в VTM (задача 0153).

Проверяется без сети и без внешних зависимостей — тест сам пишет GGUF:

* заголовок GGUF v2/v3, метаданные всех типов (скаляры + пропуск массивов),
  таблица тензоров (порядок GGUF: `ne0` — младшая), выравнивание секции
  данных;
* деквантование F32/F16/BF16/Q4_0/Q4_1/Q8_0/Q4_K/Q6_K на эталонных блоках,
  собранных руками по спецификации ggml (а не тем же кодеком, что читает);
* понятный отказ на Q5_K/Q3_K/неизвестный тип с именем типа и тензора;
* отображение имён `blk.N.attn_q.weight` -> `model.layers.N.self_attn.q_proj.weight`
  и извлечение конфига Qwen (включая vocab из `token_embd` и запрет
  `rope.scaling`);
* импорт крошечной Qwen-модели: манифест, q4.bin (kind 5, сегменты
  разбираются `tools/vtm_ref.py`), non_ternary.bin (int8-эмбеддинг kind 4);
* сквозной путь `--format trit`: импорт -> `vtm_compile.compile_model` ->
  `.vtm` -> 1 токен Python-эталоном (и C-рантаймом, если он собран);
  TRT1-сегменты сверяются с `vtm_ref.decode_trit_segment`.

Запуск:
    python3 -m unittest tests.test_gguf_import -v
"""
from __future__ import annotations

import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from array import array
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from tools import gguf_import as gi  # noqa: E402
from tools import vtm_compile, vtm_ref  # noqa: E402

VTM_BIN = os.path.join(ROOT, "src", "vtm", "vtm")
HAS_Q4_REF = all(
    hasattr(vtm_ref, name)
    for name in ("quantize_q4", "pack_q4", "decode_q4_segment_ex")
)

REAL_BLOB = Path.home() / ".ollama/models/blobs" / (
    "sha256-5ee4f07cdb9beadbbb293e85803c569b01bd37ed059d2715faa7bb405f31caa6"
)


# ---------------------------------------------------------------------------
# Писатель GGUF для тестов (независимый от читателя)
# ---------------------------------------------------------------------------


def _round_half_away(value: float) -> int:
    if value >= 0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def _encode_metadata_value(value: object) -> tuple[int, bytes]:
    if isinstance(value, bool):
        return 7, struct.pack("<?", value)
    if isinstance(value, int):
        if 0 <= value < 1 << 32:
            return 4, struct.pack("<I", value)
        if value < 0:
            return 5, struct.pack("<i", value)
        return 10, struct.pack("<Q", value)
    if isinstance(value, float):
        return 6, struct.pack("<f", value)
    if isinstance(value, str):
        data = value.encode("utf-8")
        return 8, struct.pack("<Q", len(data)) + data
    raise TypeError(f"метаданные: не умею {type(value).__name__}")


class GgufBuilder:
    """Минимальный писатель GGUF v2/v3: ровно то, что нужно тестам."""

    def __init__(self, version: int = 3, alignment: int = 32):
        self.version = version
        self.alignment = alignment
        self.metadata: list[tuple[str, int, bytes]] = []
        self.tensors: list[tuple[str, tuple[int, ...], int, bytes]] = []
        if alignment != 32:
            self.add_metadata("general.alignment", alignment)

    def add_metadata(self, key: str, value: object) -> "GgufBuilder":
        if isinstance(value, (list, tuple)):
            if all(isinstance(item, str) for item in value):
                return self.add_metadata_strings(key, list(value))
            items = [int(item) for item in value]
            payload = struct.pack(f"<{len(items)}i", *items)
            return self.add_metadata_array(key, 5, payload, len(items))
        vtype, payload = _encode_metadata_value(value)
        self.metadata.append((key, vtype, payload))
        return self

    def add_metadata_array(
        self, key: str, elem_type: int, payload: bytes, count: int
    ) -> "GgufBuilder":
        self.metadata.append((
            key,
            9,
            struct.pack("<IQ", elem_type, count) + payload,
        ))
        return self

    def add_metadata_strings(self, key: str, values: list[str]) -> "GgufBuilder":
        payload = b"".join(
            struct.pack("<Q", len(text.encode())) + text.encode()
            for text in values
        )
        return self.add_metadata_array(key, 8, payload, len(values))

    def add_tensor(
        self, name: str, dims: tuple[int, ...], type_id: int, raw: bytes
    ) -> "GgufBuilder":
        spec = gi.GGUF_TYPES.get(type_id)
        if spec is not None:
            n_values = 1
            for dim in dims:
                n_values *= dim
            expected = n_values // spec.block * spec.block_bytes
            if len(raw) != expected:
                raise AssertionError(
                    f"{name}: {len(raw)} байт, а {spec.name} ждёт {expected}"
                )
        self.tensors.append((name, tuple(dims), type_id, raw))
        return self

    def add_f32(self, name: str, dims: tuple[int, ...], values) -> "GgufBuilder":
        return self.add_tensor(
            name, dims, 0, struct.pack(f"<{len(values)}f", *values)
        )

    def add_q4_0(self, name: str, dims: tuple[int, ...], values) -> "GgufBuilder":
        return self.add_tensor(name, dims, 2, q4_0_raw(values))

    def build(self) -> bytes:
        out = bytearray(b"GGUF")
        out += struct.pack("<I", self.version)
        out += struct.pack("<QQ", len(self.tensors), len(self.metadata))
        for key, vtype, payload in self.metadata:
            data = key.encode("utf-8")
            out += struct.pack("<Q", len(data)) + data
            out += struct.pack("<I", vtype) + payload
        offsets: list[int] = []
        data = bytearray()
        for name, dims, type_id, raw in self.tensors:
            while len(data) % 32:
                data += b"\x00"
            offsets.append(len(data))
            data += raw
        for (name, dims, type_id, _raw), offset in zip(self.tensors, offsets):
            encoded = name.encode("utf-8")
            out += struct.pack("<Q", len(encoded)) + encoded
            out += struct.pack("<I", len(dims))
            out += struct.pack(f"<{len(dims)}Q", *dims)
            out += struct.pack("<IQ", type_id, offset)
        while len(out) % self.alignment:
            out += b"\x00"
        return bytes(out + data)

    def write(self, path: Path) -> Path:
        path.write_bytes(self.build())
        return path


def q4_0_raw(values) -> bytes:
    """Q4_0-блоки в раскладке ggml: byte j = value j (low), j+16 (high)."""
    values = [float(value) for value in values]
    out = bytearray()
    for start in range(0, len(values), 32):
        block = values[start:start + 32]
        absmax = max(max(block), -min(block))
        scale = absmax / 7.0 if absmax else 0.0
        out += struct.pack("<e", scale)
        if scale == 0.0:
            out += bytes([0x88]) * 16
            continue
        low = bytes(
            _round_half_away(value / scale) + 8 for value in block[:16]
        )
        high = bytes(
            _round_half_away(value / scale) + 8 for value in block[16:]
        )
        out += (
            int.from_bytes(low, "little") | (int.from_bytes(high, "little") << 4)
        ).to_bytes(16, "little")
    return bytes(out)


TINY = {
    "hidden": 32,
    "heads": 4,
    "kv_heads": 2,
    "head_dim": 8,
    "intermediate": 64,
    "vocab": 64,
    "layers": 2,
    "context": 64,
}


def build_tiny_model(
    path: Path,
    *,
    version: int = 3,
    hidden: int = TINY["hidden"],
    heads: int = TINY["heads"],
    kv_heads: int = TINY["kv_heads"],
    head_dim: int = TINY["head_dim"],
    intermediate: int = TINY["intermediate"],
    vocab: int = TINY["vocab"],
    layers: int = TINY["layers"],
    context: int = TINY["context"],
    seed: int = 11,
    tie: bool = True,
    with_bias: bool = True,
) -> Path:
    import random

    rng = random.Random(seed)
    builder = GgufBuilder(version=version)
    builder.add_metadata("general.architecture", "qwen2")
    builder.add_metadata("general.name", "Tiny Qwen test")
    builder.add_metadata("qwen2.block_count", layers)
    builder.add_metadata("qwen2.embedding_length", hidden)
    builder.add_metadata("qwen2.attention.head_count", heads)
    builder.add_metadata("qwen2.attention.head_count_kv", kv_heads)
    builder.add_metadata("qwen2.feed_forward_length", intermediate)
    builder.add_metadata("qwen2.context_length", context)
    builder.add_metadata("qwen2.rope.freq_base", 10000.0)
    builder.add_metadata("qwen2.attention.layer_norm_rms_epsilon", 0.001)
    builder.add_metadata_strings(
        "tokenizer.ggml.tokens", [f"t{index}" for index in range(vocab)]
    )

    def rand(n: int, scale: float = 0.05):
        return [rng.uniform(-scale, scale) for _ in range(n)]

    q_dim = heads * head_dim
    kv_dim = kv_heads * head_dim
    builder.add_f32("token_embd.weight", (hidden, vocab), rand(hidden * vocab))
    builder.add_f32("output_norm.weight", (hidden,), [1.0] * hidden)
    if not tie:
        builder.add_q4_0(
            "output.weight", (hidden, vocab), rand(vocab * hidden)
        )
    for layer in range(layers):
        prefix = f"blk.{layer}."
        builder.add_f32(f"{prefix}attn_norm.weight", (hidden,), [1.0] * hidden)
        builder.add_f32(f"{prefix}ffn_norm.weight", (hidden,), [1.0] * hidden)
        builder.add_q4_0(
            f"{prefix}attn_q.weight", (hidden, q_dim), rand(q_dim * hidden)
        )
        builder.add_q4_0(
            f"{prefix}attn_k.weight", (hidden, kv_dim), rand(kv_dim * hidden)
        )
        builder.add_q4_0(
            f"{prefix}attn_v.weight", (hidden, kv_dim), rand(kv_dim * hidden)
        )
        builder.add_q4_0(
            f"{prefix}attn_output.weight", (q_dim, hidden), rand(hidden * q_dim)
        )
        builder.add_q4_0(
            f"{prefix}ffn_gate.weight", (hidden, intermediate),
            rand(intermediate * hidden),
        )
        builder.add_q4_0(
            f"{prefix}ffn_up.weight", (hidden, intermediate),
            rand(intermediate * hidden),
        )
        builder.add_q4_0(
            f"{prefix}ffn_down.weight", (intermediate, hidden),
            rand(hidden * intermediate),
        )
        if with_bias:
            builder.add_f32(f"{prefix}attn_q.bias", (q_dim,), [0.0] * q_dim)
            builder.add_f32(f"{prefix}attn_k.bias", (kv_dim,), [0.0] * kv_dim)
            builder.add_f32(f"{prefix}attn_v.bias", (kv_dim,), [0.0] * kv_dim)
    return builder.write(path)


class GgufTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="gguf-import-")
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def tiny(self, **kwargs) -> Path:
        return build_tiny_model(self.tmp / "tiny.gguf", **kwargs)


# ---------------------------------------------------------------------------
# Заголовок, метаданные, таблица тензоров
# ---------------------------------------------------------------------------


class GgufHeaderTests(GgufTestBase):
    def test_reads_v3_metadata_and_tensor_table(self) -> None:
        builder = GgufBuilder(version=3)
        builder.add_metadata("general.architecture", "qwen2")
        builder.add_metadata("general.name", "unit")
        builder.add_metadata("qwen2.block_count", 3)
        builder.add_metadata("qwen2.attention.head_count", [4, 4, 4])

        builder.add_metadata("qwen2.rope.freq_base", 1e6)
        builder.add_metadata("qwen2.attention.layer_norm_rms_epsilon", 1e-6)
        builder.add_metadata_strings("tokenizer.ggml.tokens", ["a", "b", "c"])
        builder.add_f32("blk.0.attn_norm.weight", (8,), [1.0] * 8)
        builder.add_tensor("blk.0.attn_q.weight", (8, 8), 2, bytes(36))
        path = builder.write(self.tmp / "meta.gguf")

        gguf = gi.read_gguf(path)
        self.assertEqual(gguf.version, 3)
        self.assertEqual(gguf.file_size, path.stat().st_size)
        self.assertEqual(gguf.alignment, 32)
        self.assertEqual(gguf.metadata["general.architecture"], "qwen2")
        self.assertEqual(gguf.metadata["qwen2.block_count"], 3)
        self.assertEqual(gguf.array_lengths["tokenizer.ggml.tokens"], 3)
        self.assertIsNone(gguf.metadata["tokenizer.ggml.tokens"])
        self.assertEqual(len(gguf.tensors), 2)
        info = gguf.tensor("blk.0.attn_q.weight")
        self.assertEqual(info.dims, (8, 8))
        self.assertEqual(info.hf_shape, (8, 8))
        self.assertEqual(info.type_id, 2)
        self.assertEqual(info.n_elements, 64)
        self.assertEqual(info.n_bytes, 64 // 32 * 18)
        self.assertEqual(gguf.data_offset % 32, 0)

    def test_reads_v2(self) -> None:
        path = build_tiny_model(self.tmp / "v2.gguf", version=2)
        gguf = gi.read_gguf(path)
        self.assertEqual(gguf.version, 2)
        self.assertEqual(
            gguf.metadata["qwen2.embedding_length"], TINY["hidden"]
        )

    def test_rejects_bad_magic_and_version(self) -> None:
        path = self.tmp / "bad.gguf"
        path.write_bytes(b"NOPE" + bytes(28))
        with self.assertRaises(gi.GgufError) as ctx:
            gi.read_gguf(path)
        self.assertIn("не GGUF", str(ctx.exception))

        path.write_bytes(b"GGUF" + struct.pack("<I", 1) + bytes(28))
        with self.assertRaises(gi.GgufError) as ctx:
            gi.read_gguf(path)
        self.assertIn("v1", str(ctx.exception))

        path.write_bytes(b"GGUF" + struct.pack("<I", 7) + bytes(28))
        with self.assertRaises(gi.GgufError) as ctx:
            gi.read_gguf(path)
        self.assertIn("версия GGUF 7", str(ctx.exception))

    def test_rejects_truncated_file(self) -> None:
        path = build_tiny_model(self.tmp / "tiny.gguf", layers=1)
        data = path.read_bytes()
        path.write_bytes(data[:-100])
        with self.assertRaises(gi.GgufError):
            gi.read_gguf(path)

    def test_streaming_chunks_cover_large_tensor(self) -> None:
        count = gi.CHUNK_VALUES + 12345
        values = [((index * 37) % 1000) / 1000.0 for index in range(count)]
        builder = GgufBuilder()
        builder.add_metadata("general.architecture", "qwen2")
        builder.add_f32("big.weight", (count,), values)
        path = builder.write(self.tmp / "big.gguf")
        gguf = gi.read_gguf(path)
        info = gguf.tensor("big.weight")
        chunks = list(gguf.iter_tensor_chunks(info))
        self.assertGreater(len(chunks), 1)
        total = sum(len(chunk) for chunk in chunks)
        self.assertEqual(total, count)
        got = [value for chunk in chunks for value in chunk]
        for index in (0, 1, 123, count - 1):
            self.assertAlmostEqual(got[index], values[index], places=6)


# ---------------------------------------------------------------------------
# Деквантование: эталонные блоки
# ---------------------------------------------------------------------------


class DequantTests(GgufTestBase):
    def _one_tensor(self, type_id: int, dims, raw: bytes) -> gi.GgufFile:
        builder = GgufBuilder()
        builder.add_tensor("t", dims, type_id, raw)
        path = builder.write(self.tmp / "t.gguf")
        return gi.read_gguf(path)

    def _values(self, type_id: int, dims, raw: bytes) -> list[float]:
        gguf = self._one_tensor(type_id, dims, raw)
        info = gguf.tensor("t")
        return [
            value
            for chunk in gguf.iter_tensor_chunks(info)
            for value in chunk
        ]

    def test_q4_0_layout(self) -> None:
        # d = 1: low j -> j-8, high j -> (15-j)-8.
        group = bytes(
            (index & 0x0F) | (((15 - index) & 0x0F) << 4) for index in range(16)
        )
        raw = struct.pack("<e", 1.0) + group
        values = self._values(2, (32,), raw)
        expected = [float(index - 8) for index in range(16)]
        expected += [float((15 - index) - 8) for index in range(16)]
        self.assertEqual(values, expected)

    def test_q4_1_layout(self) -> None:
        # d = 0.5, m = -1: low j -> 0.5j - 1, high j -> 0.5(15-j) - 1.
        group = bytes(
            (index & 0x0F) | (((15 - index) & 0x0F) << 4) for index in range(16)
        )
        raw = struct.pack("<ee", 0.5, -1.0) + group
        values = self._values(3, (32,), raw)
        expected = [0.5 * index - 1.0 for index in range(16)]
        expected += [0.5 * (15 - index) - 1.0 for index in range(16)]
        self.assertEqual(values, expected)

    def test_q8_0_layout(self) -> None:
        qs = list(range(-16, 16))
        raw = struct.pack("<e", 0.5) + struct.pack("<32b", *qs)
        values = self._values(8, (32,), raw)
        self.assertEqual(values, [0.5 * q for q in qs])

    def test_q4_k_layout_flat_scales(self) -> None:
        # d=1, dmin=0, все 8 под-блоков scale=1, min=0.
        scales = bytes([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1])
        qs = bytes(
            (index & 0x0F) | (((index + 1) & 0x0F) << 4)
            for index in range(128)
        )
        raw = struct.pack("<ee", 1.0, 0.0) + scales + qs
        values = self._values(12, (256,), raw)
        expected: list[float] = []
        for chunk in range(4):
            for within in range(32):
                expected.append(float(qs[chunk * 32 + within] & 0x0F))
            for within in range(32):
                expected.append(float(qs[chunk * 32 + within] >> 4))
        self.assertEqual(values, expected)

    def test_q4_k_min_and_scale(self) -> None:
        # d=2, dmin=0.5; под-блок 0: scale=3, min=4; у остальных 0/0.
        # get_scale_min_k4: j=0 -> q[0]&63, q[4]&63.
        scales = bytearray(12)
        scales[0] = 3
        scales[4] = 4
        qs = bytes([0x10] * 128)  # низ=0, верх=1
        raw = struct.pack("<ee", 2.0, 0.5) + bytes(scales) + qs
        values = self._values(12, (256,), raw)
        # Под-блок 0 (первые 32 значения): 2*3*0 - 0.5*4 = -2.
        self.assertEqual(values[:32], [-2.0] * 32)
        # Под-блок 1..7: scale/min 0 -> 0.
        self.assertEqual(values[32:], [0.0] * 224)

    def test_q6_k_layout(self) -> None:
        # d=1, все 16 масштабов 1. ql=0x21, qh=0xE4:
        # q1 = 1 | (0<<4) = 1, q2 = 1 | (1<<4) = 17,
        # q3 = 2 | (2<<4) = 34, q4 = 2 | (3<<4) = 50.
        ql = bytes([0x21] * 128)
        qh = bytes([0xE4] * 64)
        scales = struct.pack("<16b", *([1] * 16))
        raw = ql + qh + scales + struct.pack("<e", 1.0)
        values = self._values(14, (256,), raw)
        group = (
            [float(1 - 32)] * 32
            + [float(17 - 32)] * 32
            + [float(34 - 32)] * 32
            + [float(50 - 32)] * 32
        )
        self.assertEqual(values, group + group)

    def test_f16_and_bf16(self) -> None:
        f16 = struct.pack("<4e", 0.5, -1.0, 2.0, 0.25)
        self.assertEqual(self._values(1, (4,), f16), [0.5, -1.0, 2.0, 0.25])
        words = [0x3F80, 0xBF80, 0x4000, 0x0000]
        raw = struct.pack("<4H", *words)
        self.assertEqual(self._values(30, (4,), raw), [1.0, -1.0, 2.0, 0.0])


# ---------------------------------------------------------------------------
# Типы, имена, конфиг
# ---------------------------------------------------------------------------


class ValidationTests(GgufTestBase):
    def test_unsupported_type_is_named(self) -> None:
        builder = GgufBuilder()
        builder.add_metadata("general.architecture", "qwen2")
        builder.add_metadata("qwen2.block_count", 1)
        builder.add_metadata("qwen2.embedding_length", 32)
        builder.add_metadata("qwen2.attention.head_count", 4)
        builder.add_metadata("qwen2.attention.head_count_kv", 2)
        builder.add_metadata("qwen2.feed_forward_length", 64)
        builder.add_metadata("qwen2.attention.key_length", 8)
        builder.add_f32("token_embd.weight", (32, 64), [0.0] * (32 * 64))
        builder.add_tensor("blk.0.ffn_up.weight", (32, 32), 13, bytes(576))
        path = builder.write(self.tmp / "q5.gguf")
        with self.assertRaises(gi.GgufError) as ctx:
            gi.read_gguf(path).tensor("blk.0.ffn_up.weight").n_bytes
        self.assertIn("Q5_K (13)", str(ctx.exception))

        out = self.tmp / "out"
        with self.assertRaises(gi.GgufError) as ctx:
            gi.import_gguf(path, out)
        message = str(ctx.exception)
        self.assertIn("Q5_K (13)", message)
        self.assertIn("blk.0.ffn_up.weight", message)
        self.assertIn("неподдерживаемые типы", message)
        self.assertFalse((out / "manifest.json").exists())

    def test_unknown_tensor_name_is_reported(self) -> None:
        builder = GgufBuilder()
        builder.add_metadata("general.architecture", "qwen2")
        builder.add_metadata("qwen2.block_count", 1)
        builder.add_metadata("qwen2.embedding_length", 32)
        builder.add_metadata("qwen2.attention.head_count", 4)
        builder.add_metadata("qwen2.attention.head_count_kv", 2)
        builder.add_metadata("qwen2.feed_forward_length", 64)
        builder.add_metadata("qwen2.attention.key_length", 8)
        builder.add_f32("token_embd.weight", (32, 64), [0.0] * (32 * 64))
        builder.add_f32("output_norm.weight", (32,), [1.0] * 32)
        builder.add_f32("blk.0.attn_q_norm.weight", (8,), [1.0] * 8)
        path = builder.write(self.tmp / "qwen3.gguf")
        with self.assertRaises(gi.GgufError) as ctx:
            gi.import_gguf(path, self.tmp / "out")
        self.assertIn("attn_q_norm", str(ctx.exception))
        self.assertIn("неизвестные тензоры", str(ctx.exception))

    def test_name_mapping(self) -> None:
        cases = {
            "token_embd.weight": (
                "model.embed_tokens.weight", "embed", None, "embed"
            ),
            "output_norm.weight": ("model.norm.weight", "final_norm", None,
                                   "final_norm"),
            "output.weight": ("lm_head.weight", "lm_head", None, "lm_head"),
            "blk.3.attn_q.weight": (
                "model.layers.3.self_attn.q_proj.weight", "matrix", 3, "q_proj"
            ),
            "blk.12.ffn_down.weight": (
                "model.layers.12.mlp.down_proj.weight", "matrix", 12,
                "down_proj"
            ),
            "blk.0.attn_norm.weight": (
                "model.layers.0.input_layernorm.weight", "norm", 0, "norm1"
            ),
            "blk.1.ffn_norm.weight": (
                "model.layers.1.post_attention_layernorm.weight", "norm", 1,
                "norm2"
            ),
            "blk.2.attn_q.bias": (
                "model.layers.2.self_attn.q_proj.bias", "bias", 2, "q_bias"
            ),
        }
        for gguf_name, expected in cases.items():
            self.assertEqual(gi.map_tensor_name(gguf_name), expected)
        with self.assertRaises(gi.GgufError):
            gi.map_tensor_name("blk.0.attn_q_norm.weight")
        with self.assertRaises(gi.GgufError):
            gi.map_tensor_name("class_embedding")

    def test_config_qwen(self) -> None:
        path = self.tiny()
        gguf = gi.read_gguf(path)
        cfg = gi.infer_config(gguf)
        self.assertEqual(cfg.architecture, "qwen2")
        self.assertEqual(cfg.hidden, TINY["hidden"])
        self.assertEqual(cfg.n_layers, TINY["layers"])
        self.assertEqual(cfg.n_heads, TINY["heads"])
        self.assertEqual(cfg.n_kv_heads, TINY["kv_heads"])
        self.assertEqual(cfg.head_dim, TINY["head_dim"])
        self.assertEqual(cfg.intermediate, TINY["intermediate"])
        self.assertEqual(cfg.vocab, TINY["vocab"])
        self.assertEqual(cfg.rope_theta, 10000.0)
        self.assertAlmostEqual(cfg.rms_norm_eps, 0.001, places=7)
        self.assertTrue(cfg.tie_word_embeddings)

    def test_config_rejects_rope_scaling(self) -> None:
        builder = GgufBuilder()
        builder.add_metadata("general.architecture", "qwen2")
        builder.add_metadata("qwen2.block_count", 1)
        builder.add_metadata("qwen2.embedding_length", 32)
        builder.add_metadata("qwen2.attention.head_count", 4)
        builder.add_metadata("qwen2.attention.head_count_kv", 2)
        builder.add_metadata("qwen2.feed_forward_length", 64)
        builder.add_metadata("qwen2.rope.scaling.type", "yarn")
        builder.add_f32("token_embd.weight", (32, 64), [0.0] * (32 * 64))
        path = builder.write(self.tmp / "yarn.gguf")
        with self.assertRaises(gi.GgufError) as ctx:
            gi.infer_config(gi.read_gguf(path))
        self.assertIn("rope.scaling", str(ctx.exception))

    def test_config_rejects_per_layer_heads(self) -> None:
        builder = GgufBuilder()
        builder.add_metadata("general.architecture", "qwen2")
        builder.add_metadata("qwen2.block_count", 2)
        builder.add_metadata("qwen2.embedding_length", 32)
        builder.add_metadata("qwen2.attention.head_count", [4, 8])
        builder.add_metadata("qwen2.attention.head_count_kv", 2)
        builder.add_metadata("qwen2.feed_forward_length", 64)
        builder.add_f32("token_embd.weight", (32, 64), [0.0] * (32 * 64))
        path = builder.write(self.tmp / "heads.gguf")
        with self.assertRaises(gi.GgufError) as ctx:
            gi.infer_config(gi.read_gguf(path))
        self.assertIn("поэлементные", str(ctx.exception))


# ---------------------------------------------------------------------------
# Импорт: q4-манифест
# ---------------------------------------------------------------------------


class Q4ImportTests(GgufTestBase):
    def test_q4_manifest_and_segments(self) -> None:
        path = self.tiny()
        out = self.tmp / "q4-out"
        result = gi.import_gguf(path, out, fmt="q4", stats=True)
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(manifest["format"], "eva32.ternary.manifest")
        self.assertEqual(manifest["format_version"], 2)
        self.assertTrue(manifest["non_ternary"]["stored_in_this_export"])
        # 0154 научил компилятор kind 5 — манифест q4 собирается как есть.
        self.assertTrue(manifest["compiler"]["ready"])
        self.assertIn("lm_head", manifest["architecture"])
        self.assertEqual(manifest["architecture"]["lm_head"], "q4")
        self.assertEqual(manifest["architecture"]["embed"], "int8")
        self.assertEqual(manifest["quantization"]["block_size"], 128)
        self.assertEqual(manifest["q4"]["block_size"], 32)
        self.assertEqual(manifest["q4"]["kind"], 5)
        self.assertTrue(manifest["partial"] is False)
        self.assertTrue(manifest["files"]["ternary.trit"]["bytes"] == 0)

        self.assertIsNotNone(result.q4_path)
        self.assertTrue(result.q4_path.is_file())
        self.assertGreater(result.q4_params, 0)

        # Каждый Linear слоя 0 попал в q4.bin как сегмент Q4_0.
        names = {entry["name"] for entry in manifest["tensors"]}
        for key in ("q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"):
            self.assertIn(
                f"model.layers.0.self_attn.{key}.weight"
                if key in ("q_proj", "k_proj", "v_proj", "o_proj")
                else f"model.layers.0.mlp.{key}.weight",
                names,
            )
        embed = next(
            entry for entry in manifest["non_ternary"]["tensors"]
            if entry["role"] == "embed"
        )
        self.assertEqual(embed["dtype"], "int8")
        tied = next(
            entry for entry in manifest["non_ternary"]["tensors"]
            if entry["role"] == "lm_head"
        )
        self.assertEqual(tied["tied_to"], embed["name"])
        self.assertNotIn("offset", tied)

        gguf = gi.read_gguf(path)
        raw_q4 = result.q4_path.read_bytes()
        checked = 0
        for entry in manifest["tensors"]:
            offset = entry["q4_offset"]
            self.assertEqual(offset % 32, 0)
            segment = raw_q4[offset: offset + entry["q4_length"]]
            # 0160: импорт пишет Q4_1 (asym) — заголовок 22 байта с n_mins.
            magic, version, flags, n_values, n_groups = struct.unpack(
                "<4sBBQI", segment[:18]
            )
            (n_mins,) = struct.unpack("<I", segment[18:22])
            self.assertEqual(magic, b"Q4_1")
            self.assertEqual(version, 1)
            self.assertEqual(flags, 0x01)
            self.assertEqual(n_values, entry["params"])
            self.assertEqual(
                n_groups, (entry["params"] + 31) // 32
            )
            self.assertEqual(n_mins, n_groups)
            self.assertEqual(entry["dtype"], "q4_1")
            if HAS_Q4_REF:
                qdata, scales, scale_dtype, asymmetric, mins = (
                    vtm_ref.decode_q4_segment_ex(segment)
                )
                self.assertEqual(scale_dtype, "float32")
                self.assertTrue(asymmetric)
                self.assertIsNotNone(mins)
                self.assertEqual(len(qdata), entry["params"])
                self.assertEqual(len(scales), n_groups)
                self.assertEqual(len(mins), n_groups)
                self.assertTrue(all(scale >= 0 for scale in scales))
                # Сверка с эталонным квантователем 0151 на тех же значениях:
                # асимметричное правило обязано совпасть побитово.
                info = gguf.tensor(entry["source_tensor"])
                values = [
                    value
                    for chunk in gguf.iter_tensor_chunks(info, entry["params"])
                    for value in chunk
                ]
                ref_q, ref_scales, ref_mins = vtm_ref.quantize_q4(
                    values, symmetric=False
                )
                self.assertEqual(ref_q, list(qdata))
                for left, right in zip(ref_scales, scales):
                    self.assertAlmostEqual(left, right, places=6)
                for left, right in zip(ref_mins, mins):
                    self.assertAlmostEqual(left, right, places=6)
                checked += 1
                if checked >= 3:
                    break

        # int8-эмбеддинг разбирается эталонным декодером kind 4.
        raw_non = result.non_ternary_path.read_bytes()
        embed_segment = raw_non[embed["offset"]: embed["offset"] + embed["length"]]
        qdata, scales, scale_dtype = vtm_ref.decode_int8_segment(embed_segment)
        self.assertEqual(len(qdata), TINY["vocab"] * TINY["hidden"])
        self.assertEqual(scale_dtype, "float32")
        self.assertEqual(len(scales), math.ceil(len(qdata) / 128))

        # Текущий компилятор обязан сказать про q4, а не упасть неясно.
        # Если в компилятор добавят kind 5 — тест не должен ломаться: тогда
        # проверяем, что манифест собирается и файл пишется.
        try:
            export = vtm_compile.load_export(str(result.manifest_path))
        except ValueError as exc:
            self.assertIn("q4", str(exc).lower())
        else:
            vtm = vtm_compile.compile_model(export.cfg, export.weights)
            vtm_ref.write_vtm(
                str(self.tmp / "q4-model.vtm"), vtm.tensors, vtm.program,
                vtm.meta,
            )

    def test_q4_quantizer_matches_reference(self) -> None:
        if not HAS_Q4_REF:
            self.skipTest("в tools/vtm_ref.py ещё нет Q4 (задача 0151)")
        import random

        rng = random.Random(7)
        for _ in range(200):
            values = array(
                "f",
                (round(rng.uniform(-1.0, 1.0), 4) for _ in range(32)),
            )
            ref_q, ref_scales = vtm_ref.quantize_q4(values)
            ref = vtm_ref.pack_q4(ref_q)
            scale, packed = gi._q4_encode_block(values)
            self.assertAlmostEqual(scale, ref_scales[0], places=6)
            self.assertEqual(packed, ref)


# ---------------------------------------------------------------------------
# Сквозной путь: импорт -> компиляция -> 1 токен
# ---------------------------------------------------------------------------


class EndToEndTests(GgufTestBase):
    def _import_and_compile(self, source: Path, out: Path, **kwargs):
        result = gi.import_gguf(source, out, fmt="trit", **kwargs)
        vtm = vtm_compile.compile_model(str(result.manifest_path))
        target = out / "model.vtm"
        vtm_ref.write_vtm(str(target), vtm.tensors, vtm.program, vtm.meta)
        return result, target

    def test_trit_import_compiles_and_runs_one_token(self) -> None:
        source = self.tiny()
        out = self.tmp / "trit"
        result, model_path = self._import_and_compile(source, out)
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        self.assertTrue(manifest["compiler"]["ready"])
        self.assertEqual(manifest["architecture"]["embed"], "int8")
        self.assertEqual(manifest["architecture"]["lm_head"], "mm")
        self.assertTrue(manifest["architecture"]["tie_word_embeddings"])
        self.assertGreater(model_path.stat().st_size, 0)
        self.assertGreater(manifest["files"]["ternary.trit"]["bytes"], 0)
        self.assertIsNone(manifest["q4"])
        self.assertIsNotNone(manifest["trit"])
        self.assertEqual(manifest["trit"]["params"], result.trit_params)

        # TRT1-сегмент первого слоя разбирается эталоном и даёт ровно столько
        # тритов/масштабов, сколько записано в манифесте.
        entry = next(
            item for item in manifest["tensors"]
            if item["name"] == "model.layers.0.self_attn.q_proj.weight"
        )
        raw_trit = (out / "ternary.trit").read_bytes()
        segment = raw_trit[entry["trit_offset"]:
                           entry["trit_offset"] + entry["trit_length"]]
        magic, version, flags, n_values, n_scale = struct.unpack(
            "<4sBBQI", segment[:18]
        )
        self.assertEqual(magic, b"TRT1")
        self.assertEqual(version, 1)
        self.assertEqual(flags, 0x01)
        self.assertEqual(n_values, entry["params"])
        self.assertEqual(n_scale, math.ceil(entry["params"] / 128))
        trits, scales, scale_dtype = vtm_ref.decode_trit_segment(segment)
        self.assertEqual(scale_dtype, "float32")
        self.assertEqual(len(trits), entry["params"])
        self.assertEqual(len(scales), n_scale)

        executed = vtm_ref.execute(str(model_path), tokens=[1], max_new=1)
        logits = executed.logits
        self.assertIsNotNone(logits)
        self.assertEqual(len(logits), TINY["vocab"])
        self.assertTrue(all(math.isfinite(value) for value in logits))
        self.assertEqual(len(executed.tokens), 2)
        self.assertGreaterEqual(executed.tokens[-1], 0)
        self.assertLess(executed.tokens[-1], TINY["vocab"])

        if os.access(VTM_BIN, os.X_OK):
            proc = subprocess.run(
                [VTM_BIN, "run", str(model_path), "--tokens", "1",
                 "--max-new", "1"],
                capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_limit_layers_and_vocab_partial(self) -> None:
        source = self.tiny(layers=2)
        out = self.tmp / "slice"
        result, model_path = self._import_and_compile(
            source, out, limit_layers=1, limit_vocab=32
        )
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["architecture"]["n_layers"], 1)
        self.assertEqual(manifest["architecture"]["vocab"], 32)
        self.assertTrue(manifest["partial"])
        embedded = next(
            entry for entry in manifest["non_ternary"]["tensors"]
            if entry["role"] == "embed"
        )
        self.assertEqual(embedded["shape"], [32, TINY["hidden"]])
        executed = vtm_ref.execute(str(model_path), tokens=[3], max_new=1)
        self.assertEqual(len(executed.logits or []), 32)

    def test_trit_without_tie_uses_lm_head(self) -> None:
        source = self.tiny(tie=False)
        out = self.tmp / "notie"
        result, model_path = self._import_and_compile(source, out)
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        self.assertFalse(manifest["architecture"]["tie_word_embeddings"])
        self.assertEqual(manifest["architecture"]["lm_head"], "tmv")
        entry = next(
            entry for entry in manifest["tensors"]
            if entry["role"] == "lm_head"
        )
        self.assertEqual(entry["kind"], "trit")
        self.assertNotIn("tied_to", entry)
        executed = vtm_ref.execute(str(model_path), tokens=[2], max_new=1)
        self.assertEqual(len(executed.logits or []), TINY["vocab"])

    def test_import_refuses_nonempty_out_without_force(self) -> None:
        source = self.tiny()
        out = self.tmp / "occupied"
        out.mkdir()
        (out / "keep.txt").write_text("x", encoding="utf-8")
        with self.assertRaises(gi.GgufError) as ctx:
            gi.import_gguf(source, out)
        self.assertIn("--force", str(ctx.exception))
        (out / "keep.txt").unlink()
        gi.import_gguf(source, out)
        self.assertTrue((out / "manifest.json").is_file())


# ---------------------------------------------------------------------------
# kind 5 в контейнере: loader/ядра 0151-0152
# ---------------------------------------------------------------------------


def _kinds5_supported() -> bool:
    return (
        hasattr(vtm_ref, "KIND_Q4")
        and hasattr(vtm_ref, "decode_q4_segment_ex")
        and hasattr(vtm_ref.Tensor, "q4")
    )


class Kind5ContainerTests(GgufTestBase):
    """q4.bin из импорта -> Tensor(kind 5) -> .vtm -> 1 токен.

    Компилятор (`tools/vtm_compile.py`) в этом спринте не меняется и q4.bin не
    читает. Здесь веса собираются вручную из манифеста импорта, а
    `matrix_tensor` подменяется в памяти — так проверяется именно контейнер с
    kind 5 (загрузчик и ядра 0151/0152), а не компилятор. Когда компилятор
    научится q4, этот тест можно заменить вызовом `load_export`.
    """

    def setUp(self) -> None:
        super().setUp()
        if not _kinds5_supported():
            self.skipTest("kind 5 ещё нет в tools/vtm_ref.py (задача 0151)")

    def _build_container(self, out: Path) -> Path:
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        arch = manifest["architecture"]
        cfg = vtm_compile.ModelConfig(
            hidden=arch["hidden"], n_heads=arch["n_heads"],
            n_kv_heads=arch["n_kv_heads"], head_dim=arch["head_dim"],
            intermediate=arch["intermediate"], vocab=arch["vocab"],
            n_layers=arch["n_layers"], context_len=arch["context_len"],
            eps=arch["rms_norm_eps"], rope_theta=arch["rope_theta"],
            lm_head="mm", tie_word_embeddings=arch["tie_word_embeddings"],
            embed_quant=arch["embed"],
        )
        weights: dict[str, object] = {
            "layers": [{} for _ in range(cfg.n_layers)]
        }
        non = (out / "non_ternary.bin").read_bytes()
        for entry in manifest["non_ternary"]["tensors"]:
            if entry.get("tied_to"):
                continue
            payload = non[entry["offset"]: entry["offset"] + entry["length"]]
            if entry["dtype"] == "int8":
                qdata, scales, scale_dtype = vtm_ref.decode_int8_segment(payload)
                weights["embed"] = vtm_ref.Tensor.int8(
                    "emb.w", entry["shape"], qdata, scales, scale_dtype
                )
            elif entry["dtype"] == "float32":
                values = list(struct.unpack(f"<{len(payload) // 4}f", payload))
                tensor = vtm_ref.Tensor.f32(entry["key"], entry["shape"], values)
                if entry["role"] == "final_norm":
                    weights["final_norm"] = tensor
                else:
                    weights["layers"][entry["layer"]][entry["key"]] = tensor
            else:  # pragma: no cover - импорт пишет только f32/int8
                raise AssertionError(entry["dtype"])
        q4 = (out / "q4.bin").read_bytes()
        for entry in manifest["tensors"]:
            segment = q4[entry["q4_offset"]: entry["q4_offset"] + entry["q4_length"]]
            qdata, scales, scale_dtype, asym, mins = (
                vtm_ref.decode_q4_segment_ex(segment)
            )
            weights["layers"][entry["layer"]][entry["key"]] = vtm_ref.Tensor.q4(
                entry["key"], entry["shape"], qdata, scales, scale_dtype,
                asymmetric=asym, mins=mins,
            )
        original = vtm_compile.matrix_tensor

        def patched(graph, name, rows, cols, spec, dense=False):
            if isinstance(spec, vtm_ref.Tensor) and spec.kind == vtm_ref.KIND_Q4:
                if tuple(spec.dims) != (rows, cols):
                    raise ValueError(f"{name}: {spec.dims} != {(rows, cols)}")
                spec.name = name
                return graph.add(spec)
            return original(graph, name, rows, cols, spec, dense=dense)

        vtm_compile.matrix_tensor = patched
        try:
            vtm = vtm_compile.compile_model(cfg, weights)
        finally:
            vtm_compile.matrix_tensor = original
        target = out / "model-q4.vtm"
        vtm_ref.write_vtm(str(target), vtm.tensors, vtm.program, vtm.meta)
        return target

    def test_kind5_container_loads_and_runs_one_token(self) -> None:
        source = self.tiny()
        out = self.tmp / "k5"
        result = gi.import_gguf(source, out, fmt="q4")
        self.assertIsNotNone(result.q4_path)
        model_path = self._build_container(out)

        # Python-эталон: kind 5 читается из контейнера и считается.
        executed = vtm_ref.execute(str(model_path), tokens=[1], max_new=1)
        self.assertEqual(len(executed.logits or []), TINY["vocab"])
        self.assertTrue(all(math.isfinite(v) for v in executed.logits or []))
        loaded = vtm_ref.read_vtm(str(model_path))
        self.assertTrue(
            any(t.kind == vtm_ref.KIND_Q4 for t in loaded.tensors),
            "в контейнере нет ни одного kind 5",
        )

        # C-рантайм: только если бинарь не старше kind 5 в loader.c.
        loader = Path(ROOT) / "src/vtm/loader.c"
        binary = Path(VTM_BIN)
        if not (os.access(VTM_BIN, os.X_OK)
                and loader.is_file()
                and binary.stat().st_mtime >= loader.stat().st_mtime):
            self.skipTest("C-рантайм старше kind 5 — пересоберите src/vtm/vtm")
        proc = subprocess.run(
            [VTM_BIN, "run", str(model_path), "--tokens", "1", "--max-new", "1"],
            capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("вывод", proc.stdout)


# ---------------------------------------------------------------------------
# Реальная модель из ollama (по желанию: VTM_GGUF_REAL=1)
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    os.environ.get("VTM_GGUF_REAL") and REAL_BLOB.is_file(),
    "реальный GGUF не проверяется (нет блоба или VTM_GGUF_REAL)",
)
class RealOllamaTests(unittest.TestCase):
    def test_qwen25_3b_config(self) -> None:
        gguf = gi.read_gguf(REAL_BLOB)
        cfg = gi.infer_config(gguf)
        self.assertEqual(cfg.architecture, "qwen2")
        self.assertEqual(cfg.hidden, 2048)
        self.assertEqual(cfg.n_layers, 36)
        self.assertEqual(cfg.n_heads, 16)
        self.assertEqual(cfg.n_kv_heads, 2)
        self.assertEqual(cfg.head_dim, 128)
        self.assertEqual(cfg.intermediate, 11008)
        self.assertEqual(cfg.vocab_source, 151936)
        self.assertEqual(cfg.context_len, 32768)
        self.assertAlmostEqual(cfg.rope_theta, 1e6, places=3)
        self.assertAlmostEqual(cfg.rms_norm_eps, 1e-6, places=9)
        self.assertTrue(cfg.tie_word_embeddings)
        types = {info.type_id for info in gguf.tensors}
        self.assertEqual(types, {gi.GGUF_TYPES[0].type_id,
                                 gi.GGUF_TYPES[12].type_id,
                                 gi.GGUF_TYPES[14].type_id})


if __name__ == "__main__":
    unittest.main(verbosity=2)
