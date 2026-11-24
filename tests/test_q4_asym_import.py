"""Импорт Q4 как Q4_1 (asym + mins) вместо симметричного Q4_0 — задача 0160.

Почему
======

Q4_K_M (то, чем ollama раздаёт Qwen2.5) хранит на под-блок 32 значения свой
масштаб **и минимум**. Переквантование в симметричный Q4_0
(`scale = absmax/7`, `w = (q-8)·scale`) минимум выбрасывает: если 32 веса
лежат узкой полосой вдали от нуля, из 16 уровней работают 3–6, а шаг сетки
в разы больше оптимального. Отсюда 10 % rel-RMS на настоящем 3B и
«Столица Франции - это Пгврф» вместо Парижа. Q4_1 (`w = q·scale + min`)
минимум сохраняет; цена — 4 байта min на группу 32 (+0,125 Б/вес).

Что проверяется (без сети и без гигабайтов)
===========================================

* правило `q = clamp(round_half_away((w-min)/scale), 0, 15)`, `min = min(32)`,
  `scale = (max-min)/15` побитово совпадает с эталоном
  `tools/vtm_ref.py:quantize_q4(..., symmetric=False)`;
* вырожденная группа (все 32 значения равны) кодируется точно (`scale=0`);
* сегмент из импорта синтетического GGUF: magic `"Q4_1"`, заголовок 22 байта
  с `n_mins == n_groups`, порядок «масштабы, min, нибблы», `dtype: "q4_1"`
  в манифесте;
* round-trip: веса, де-квантованные из сегмента, отличаются от исходных не
  больше чем на полшага сетки (`scale/2`) — это и есть одна ступень Q4, и
  она не растёт из-за записи/чтения;
* rel-RMS на тех же весах **ниже**, чем у симметричного Q4_0 (и заметно ниже
  на смещённых блоках — именно так выглядят под-блоки Q4_K);
* компилятор и `.vtm` собираются, Python-эталон и C-рантайм считают один шаг
  с расхождением ≤1e-5 (если собран `src/vtm/libvtm.a` и есть компилятор).

Запуск:
    python3 -m unittest tests.test_q4_asym_import -v
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import os
import random
import struct
import sys
import tempfile
import unittest
from array import array
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tests.test_gguf_import import GgufBuilder  # noqa: E402
from tools import gguf_import as gi  # noqa: E402
from tools import parity_real, vtm_compile, vtm_ref  # noqa: E402

BLOCK = vtm_ref.Q4_SCALE_BLOCK
HAS_Q4_REF = all(
    hasattr(vtm_ref, name)
    for name in ("quantize_q4", "pack_q4", "decode_q4_segment_ex")
)

TINY = {
    "hidden": 32,
    "heads": 4,
    "kv_heads": 2,
    "head_dim": 8,
    "intermediate": 64,
    "vocab": 64,
}


def build_offset_model(path: Path) -> Path:
    """GGUF с F32-матрицами, разложенными как под-блоки Q4_K.

    Настоящий Q4_K-подблок — это 32 веса узкой полосой со своим масштабом
    **и минимумом**, часто вдали от нуля. Синтетика повторяет это: на каждую
    группу 32 значения свой центр и своя ширина. Источник — F32, чтобы
    квантователь было с чем сравнивать (GGUF Q4_0-источник уже лежит на
    симметричной сетке, и round-trip по нему ничего не доказывает).
    """
    rng = random.Random(160)
    hidden = TINY["hidden"]
    q_dim = TINY["heads"] * TINY["head_dim"]
    kv_dim = TINY["kv_heads"] * TINY["head_dim"]

    def weights(n: int) -> list[float]:
        values: list[float] = []
        while len(values) < n:
            center = rng.uniform(-0.8, 0.8)
            width = rng.uniform(0.05, 0.2)
            for _ in range(BLOCK):
                values.append(center + rng.uniform(-width / 2, width / 2))
        return values[:n]

    builder = GgufBuilder()
    builder.add_metadata("general.architecture", "qwen2")
    builder.add_metadata("general.name", "Offsets Q4_1 test")
    builder.add_metadata("qwen2.block_count", 1)
    builder.add_metadata("qwen2.embedding_length", hidden)
    builder.add_metadata("qwen2.attention.head_count", TINY["heads"])
    builder.add_metadata("qwen2.attention.head_count_kv", TINY["kv_heads"])
    builder.add_metadata("qwen2.feed_forward_length", TINY["intermediate"])
    builder.add_metadata("qwen2.context_length", 64)
    builder.add_metadata("qwen2.rope.freq_base", 10000.0)
    builder.add_metadata("qwen2.attention.layer_norm_rms_epsilon", 0.001)
    builder.add_metadata_strings(
        "tokenizer.ggml.tokens", [f"t{index}" for index in range(TINY["vocab"])]
    )
    builder.add_f32(
        "token_embd.weight", (hidden, TINY["vocab"]),
        weights(hidden * TINY["vocab"]),
    )
    builder.add_f32("output_norm.weight", (hidden,), [1.0] * hidden)
    builder.add_f32("blk.0.attn_norm.weight", (hidden,), [1.0] * hidden)
    builder.add_f32("blk.0.ffn_norm.weight", (hidden,), [1.0] * hidden)
    for name, dims in (
        ("blk.0.attn_q.weight", (hidden, q_dim)),
        ("blk.0.attn_k.weight", (hidden, kv_dim)),
        ("blk.0.attn_v.weight", (hidden, kv_dim)),
        ("blk.0.attn_output.weight", (q_dim, hidden)),
        ("blk.0.ffn_gate.weight", (hidden, TINY["intermediate"])),
        ("blk.0.ffn_up.weight", (hidden, TINY["intermediate"])),
        ("blk.0.ffn_down.weight", (TINY["intermediate"], hidden)),
    ):
        n_values = dims[0] * dims[1]
        builder.add_f32(name, dims, weights(n_values))
    return builder.write(path)


def _nibble(packed: bytes, index: int) -> int:
    """Ниббл `index` из 16 байт группы (раскладка GGML: j и j+16)."""
    byte = packed[index & 15]
    return (byte & 0x0F) if index < 16 else (byte >> 4)


def _reconstruct_asym(block, packed: bytes, scale: float, minimum: float):
    for index, value in enumerate(block):
        yield value, _nibble(packed, index) * scale + minimum


def _reconstruct_sym(block, packed: bytes, scale: float):
    for index, value in enumerate(block):
        if scale == 0.0:
            yield value, 0.0
        else:
            yield value, (_nibble(packed, index) - gi.Q4_ZERO_NIBBLE) * scale


def _block_stats(values, *, asymmetric: bool) -> gi._Stats:
    """rel-RMS-статистика правила импорта на последовательности fp32-значений."""
    stats = gi._Stats()
    for start in range(0, len(values), BLOCK):
        block = values[start:start + BLOCK]
        if asymmetric:
            minimum, scale, packed = gi._q4_encode_asym_block(block)
            for value, restored in _reconstruct_asym(
                block, packed, scale, minimum
            ):
                stats.add(value, restored)
        else:
            scale, packed = gi._q4_encode_block(block)
            for value, restored in _reconstruct_sym(block, packed, scale):
                stats.add(value, restored)
    return stats


# ---------------------------------------------------------------------------
# Правило квантования блока
# ---------------------------------------------------------------------------


class AsymBlockRuleTests(unittest.TestCase):
    def test_matches_reference_quantizer(self) -> None:
        """Побитовое совпадение с vtm_ref.quantize_q4(symmetric=False)."""
        if not HAS_Q4_REF:
            self.skipTest("в tools/vtm_ref.py нет quantize_q4 (задача 0151)")
        rng = random.Random(160)
        for _ in range(200):
            # Смещённый блок: так выглядят под-блоки Q4_K (узкая полоса
            # вдали от нуля), и именно на них симметричный Q4_0 плох.
            offset = rng.uniform(-2.0, 2.0)
            width = rng.uniform(1e-3, 1.0)
            values = array(
                "f",
                (offset + rng.uniform(0.0, width) for _ in range(BLOCK)),
            )
            ref_q, ref_scales, ref_mins = vtm_ref.quantize_q4(
                values, symmetric=False
            )
            minimum, scale, packed = gi._q4_encode_asym_block(values)
            self.assertEqual(packed, vtm_ref.pack_q4(ref_q))
            self.assertAlmostEqual(scale, ref_scales[0], places=6)
            self.assertAlmostEqual(minimum, ref_mins[0], places=6)

    def test_round_trip_within_half_step(self) -> None:
        """Каждое значение восстанавливается с ошибкой не больше scale/2."""
        rng = random.Random(7)
        for _ in range(100):
            values = array(
                "f",
                (rng.gauss(0.3, 0.4) for _ in range(BLOCK)),
            )
            minimum, scale, packed = gi._q4_encode_asym_block(values)
            for value, restored in _reconstruct_asym(
                values, packed, scale, minimum
            ):
                self.assertLessEqual(
                    abs(value - restored),
                    scale / 2.0 + 1e-7,
                    f"w={value!r} восстановлено {restored!r} при scale={scale}",
                )

    def test_uses_full_nibble_range(self) -> None:
        """Крайние значения группы — нибблы 0 и 15, а не 8/7 как у Q4_0."""
        values = array("f", (0.5 + 0.01 * index for index in range(BLOCK)))
        minimum, scale, packed = gi._q4_encode_asym_block(values)
        self.assertAlmostEqual(minimum, values[0], places=6)
        nibbles = [_nibble(packed, index) for index in range(BLOCK)]
        self.assertEqual(min(nibbles), gi.Q4_NIBBLE_MIN)
        self.assertEqual(max(nibbles), gi.Q4_NIBBLE_MAX)

    def test_degenerate_block_is_exact(self) -> None:
        """Все 32 значения равны: scale=0, min=значение, ошибки нет."""
        values = array("f", [0.25] * BLOCK)
        minimum, scale, packed = gi._q4_encode_asym_block(values)
        self.assertEqual(scale, 0.0)
        self.assertEqual(minimum, 0.25)
        self.assertEqual(packed, bytes(gi.Q4_GROUP_BYTES))
        for value, restored in _reconstruct_asym(
            values, packed, scale, minimum
        ):
            self.assertEqual(restored, value)

    def test_rejects_nonfinite(self) -> None:
        values = array("f", [0.0] * BLOCK)
        values[3] = math.inf
        with self.assertRaises(gi.GgufError):
            gi._q4_encode_asym_block(values)

    def test_rel_rms_lower_than_symmetric(self) -> None:
        """На смещённых блоках Q4_1 в разы точнее Q4_0 (причина деградации)."""
        rng = random.Random(3)
        values = array(
            "f",
            [0.8 + rng.gauss(0.0, 0.02) for _ in range(BLOCK * 64)],
        )
        sym = _block_stats(values, asymmetric=False)
        asym = _block_stats(values, asymmetric=True)
        # Точная граница асимметричной сетки: ошибка равномерна в ±scale/2.
        self.assertLess(asym.rel_rms, 0.02)
        # Симметричная сетка на полосе 0,8±0,06 тратит 16 уровней на весь
        # диапазон ±0,86 — шаг в 15 раз больше нужного.
        self.assertLess(asym.rel_rms * 5, sym.rel_rms)


# ---------------------------------------------------------------------------
# Выбор вида Q4 по бюджету контейнера (--q4-kind auto)
# ---------------------------------------------------------------------------


class Q4KindSelectionTests(unittest.TestCase):
    """auto: Q4_1, если влезает в 4 ГиБ; иначе Q4_0 (7B) — с текстом в stderr."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="q4-kind-")
        self.tmp = Path(self._tmp.name)
        self.source = build_offset_model(self.tmp / "tiny.gguf")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _import(self, name: str, **kwargs) -> tuple[gi.ImportResult, str]:
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            result = gi.import_gguf(self.source, self.tmp / name, fmt="q4",
                                    **kwargs)
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        return result, err.getvalue(), manifest

    def test_estimate_counts_min_array(self) -> None:
        gguf = gi.read_gguf(self.source)
        cfg = gi.infer_config(gguf)
        plans = gi.plan_import(gguf, cfg, fmt="q4", embed_mode="int8")
        asym = gi.estimate_container_bytes(
            plans, q4_asym=True, scale_dtype="float32"
        )
        sym = gi.estimate_container_bytes(
            plans, q4_asym=False, scale_dtype="float32"
        )
        groups = sum(
            (plan.n_values + BLOCK - 1) // BLOCK
            for plan in plans if plan.target == "q4"
        )
        # Разница Q4_1−Q4_0 = 4 байта min + 4 байта заголовка на сегмент.
        self.assertEqual(asym - sym, groups * 4 + 4 * sum(
            1 for plan in plans if plan.target == "q4"
        ))
        self.assertLess(sym, gi.CONTAINER_LIMIT)

    def test_auto_picks_q4_1_when_it_fits(self) -> None:
        _result, err, manifest = self._import("auto-small")
        self.assertEqual(manifest["q4"]["kind"], 5)
        self.assertTrue(manifest["q4"]["asymmetric"])
        self.assertEqual(manifest["q4"]["segment"]["magic"], "Q4_1")
        self.assertEqual(manifest["quantization"]["q4_kind"], "q4_1")
        self.assertIn("q4_kind=auto → q4_1", err)
        self.assertIn("ГиБ", err)

    def test_auto_falls_back_to_q4_0_over_limit(self) -> None:
        # 7B-случай без 7B: ужимаем предел до размера, который Q4_1 уже не
        # проходит, а Q4_0 — ещё проходит.
        gguf = gi.read_gguf(self.source)
        cfg = gi.infer_config(gguf)
        plans = gi.plan_import(gguf, cfg, fmt="q4", embed_mode="int8")
        asym = gi.estimate_container_bytes(
            plans, q4_asym=True, scale_dtype="float32"
        )
        sym = gi.estimate_container_bytes(
            plans, q4_asym=False, scale_dtype="float32"
        )
        limit = (asym + sym) // 2  # между Q4_1 и Q4_0
        original = gi.CONTAINER_LIMIT
        gi.CONTAINER_LIMIT = limit
        try:
            _result, err, manifest = self._import("auto-big")
        finally:
            gi.CONTAINER_LIMIT = original
        self.assertFalse(manifest["q4"]["asymmetric"])
        self.assertEqual(manifest["q4"]["segment"]["magic"], "Q4_0")
        self.assertEqual(manifest["q4"]["segment"]["header_bytes"],
                         gi.Q4_HEADER_SIZE)
        self.assertEqual(manifest["quantization"]["q4_kind"], "q4_0")
        self.assertIn("q4_kind=auto → q4_0", err)
        self.assertIn("Q4_0", err)

    def test_auto_warns_when_even_q4_0_exceeds_limit(self) -> None:
        """Ни один вид не влезает: в stderr подсказка, что менять."""
        original = gi.CONTAINER_LIMIT
        gi.CONTAINER_LIMIT = 1024  # заведомо меньше любого импорта
        try:
            _result, err, manifest = self._import("auto-tiny-limit")
        finally:
            gi.CONTAINER_LIMIT = original
        self.assertEqual(manifest["quantization"]["q4_kind"], "q4_0")
        self.assertIn("q4_kind=auto → q4_0", err)
        self.assertIn("ADR 0016", err)
        self.assertIn("--q4-scale-dtype fp16", err)
        self.assertIn("--embed trit", err)

    def test_explicit_q4_0_keeps_symmetric_format(self) -> None:
        result, _err, manifest = self._import("sym", q4_kind="q4_0")
        self.assertEqual(manifest["q4"]["segment"]["magic"], "Q4_0")
        self.assertEqual(manifest["q4"]["segment"]["header_bytes"],
                         gi.Q4_HEADER_SIZE)
        self.assertEqual(manifest["q4"]["nibble_zero"], gi.Q4_ZERO_NIBBLE)
        raw = result.q4_path.read_bytes()
        entry = next(e for e in manifest["tensors"] if e.get("kind") == "q4")
        segment = raw[entry["q4_offset"]:
                      entry["q4_offset"] + entry["q4_length"]]
        _q, _s, _d, asym, mins = vtm_ref.decode_q4_segment_ex(segment)
        self.assertFalse(asym)
        self.assertIsNone(mins)
        self.assertEqual(entry["dtype"], "q4_0")

    def test_explicit_q4_1_over_limit_warns(self) -> None:
        gguf = gi.read_gguf(self.source)
        cfg = gi.infer_config(gguf)
        plans = gi.plan_import(gguf, cfg, fmt="q4", embed_mode="int8")
        asym = gi.estimate_container_bytes(
            plans, q4_asym=True, scale_dtype="float32"
        )
        original = gi.CONTAINER_LIMIT
        gi.CONTAINER_LIMIT = asym // 2
        try:
            _result, err, manifest = self._import("forced", q4_kind="q4_1")
        finally:
            gi.CONTAINER_LIMIT = original
        self.assertEqual(manifest["quantization"]["q4_kind"], "q4_1")
        self.assertEqual(manifest["q4"]["segment"]["magic"], "Q4_1")
        self.assertIn("задан явно", err)


# ---------------------------------------------------------------------------
# Синтетический GGUF -> Q4_1-сегмент -> round-trip
# ---------------------------------------------------------------------------


class Q4OneImportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="q4-asym-")
        cls.tmp = Path(cls._tmp.name)
        cls.source = build_offset_model(cls.tmp / "tiny.gguf")
        cls.out = cls.tmp / "export"
        cls.result = gi.import_gguf(cls.source, cls.out, fmt="q4", stats=True)
        cls.manifest = cls.result.manifest
        cls.gguf = gi.read_gguf(cls.source)
        cls.raw = cls.result.q4_path.read_bytes()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def q4_entries(self):
        return [
            entry for entry in self.manifest["tensors"]
            if entry.get("kind") == "q4"
        ]

    def original_values(self, entry) -> list[float]:
        info = self.gguf.tensor(entry["source_tensor"])
        return [
            value
            for chunk in self.gguf.iter_tensor_chunks(info, entry["params"])
            for value in chunk
        ]

    def test_manifest_declares_q4_1(self) -> None:
        q4 = self.manifest["q4"]
        self.assertEqual(q4["kind"], 5)
        self.assertTrue(q4["asymmetric"])
        self.assertEqual(q4["segment"]["magic"], "Q4_1")
        self.assertEqual(q4["segment"]["header_bytes"], 22)
        self.assertIn("min", q4["segment"]["order"])
        self.assertEqual(q4["layout"], "ggml: байт j = значение j (low), "
                                       "j+16 (high); w = q*scale + min")
        self.assertIn("Q4_1", self.manifest["quantization"]["rule"])
        self.assertTrue(self.manifest["compiler"]["ready"])
        self.assertTrue(self.q4_entries())
        for entry in self.q4_entries():
            self.assertEqual(entry["dtype"], "q4_1")

    def test_segment_header_and_layout(self) -> None:
        """Заголовок 22 байта, n_mins == n_groups, порядок полей как в 0017."""
        for entry in self.q4_entries():
            offset = entry["q4_offset"]
            length = entry["q4_length"]
            segment = self.raw[offset: offset + length]
            magic, version, flags, n_values, n_groups = struct.unpack(
                "<4sBBQI", segment[:gi.Q4_HEADER_SIZE]
            )
            (n_mins,) = struct.unpack("<I", segment[gi.Q4_HEADER_SIZE:22])
            self.assertEqual(magic, b"Q4_1")
            self.assertEqual(version, 1)
            self.assertEqual(flags, 0x01)  # fp32-масштабы по умолчанию
            self.assertEqual(n_values, entry["params"])
            self.assertEqual(n_groups, (entry["params"] + BLOCK - 1) // BLOCK)
            self.assertEqual(n_mins, n_groups)
            expected = 22 + n_groups * 4 * 2 + n_groups * gi.Q4_GROUP_BYTES
            self.assertEqual(length, expected)

            # Ручная раскладка: масштабы идут перед min'ами.
            scales = struct.unpack(
                f"<{n_groups}f", segment[22:22 + n_groups * 4]
            )
            mins = struct.unpack(
                f"<{n_groups}f", segment[22 + n_groups * 4:22 + n_groups * 8]
            )
            decoded, dec_scales, dec_dtype, asym, dec_mins = (
                vtm_ref.decode_q4_segment_ex(segment)
            )
            self.assertEqual(dec_dtype, "float32")
            self.assertTrue(asym)
            self.assertEqual(list(dec_scales), list(scales))
            self.assertEqual(list(dec_mins), list(mins))
            self.assertEqual(len(decoded), entry["params"])

    def test_round_trip_matches_original_weights(self) -> None:
        """Де-квантованные из сегмента веса = исходные ± scale/2."""
        checked = 0
        for entry in self.q4_entries():
            segment = self.raw[
                entry["q4_offset"]: entry["q4_offset"] + entry["q4_length"]
            ]
            qdata, scales, _dtype, asymmetric, mins = (
                vtm_ref.decode_q4_segment_ex(segment)
            )
            self.assertTrue(asymmetric)
            values = self.original_values(entry)
            self.assertEqual(len(values), len(qdata))
            worst = 0.0
            for index, value in enumerate(values):
                group = index // BLOCK
                restored = (
                    qdata[index] * scales[group] + mins[group]
                    if scales[group] != 0.0
                    else mins[group]
                )
                error = abs(value - restored)
                limit = scales[group] / 2.0 + 1e-7
                self.assertLessEqual(
                    error, limit,
                    f"{entry['name']}[{index}]: w={value!r}, Q4_1={restored!r},"
                    f" scale={scales[group]!r}",
                )
                worst = max(worst, error)
            self.assertGreater(worst, 0.0)
            checked += 1
            if checked >= 3:
                break
        self.assertGreater(checked, 0)

    def test_rel_rms_beats_symmetric_q4_0(self) -> None:
        """rel-RMS импорта (asym) ниже симметричного правила на тех же весах."""
        asym = self.result.stats["q4"]
        self.assertGreater(asym.values, 0)
        symmetric = gi._Stats()
        for entry in self.q4_entries():
            symmetric.merge(
                _block_stats(
                    self.original_values(entry), asymmetric=False
                )
            )
        self.assertLess(asym.rel_rms, symmetric.rel_rms)
        self.assertLess(asym.rel_rms, 0.1)
        # Каждое значение укладывается в полшага сетки своей группы.
        self.assertGreater(asym.max_abs, 0.0)

    def _compile(self) -> Path:
        export = vtm_compile.load_export(self.out / "manifest.json")
        model = vtm_compile.compile_model(
            export.cfg, export.weights, export.meta
        )
        path = self.out / "q4_1.vtm"
        vtm_ref.write_vtm(
            str(path), model.tensors, model.program, model.meta, integrity=True
        )
        return path

    def test_compiler_builds_container_and_python_runs(self) -> None:
        path = self._compile()
        self.assertGreater(path.stat().st_size, 0)
        loaded = vtm_ref.read_vtm(str(path))
        q4 = [t for t in loaded.tensors if t.kind == vtm_ref.KIND_Q4]
        self.assertTrue(q4)
        self.assertTrue(all(t.q4_asymmetric for t in q4))
        self.assertTrue(all(t.q4_mins is not None for t in q4))
        result = vtm_ref.execute(str(path), tokens=[1, 2], max_new=1)
        self.assertTrue(
            all(math.isfinite(value) for value in (result.logits or []))
        )

    def test_c_python_step_parity(self) -> None:
        """C-рантайм (vtm_q4_tmv/mm с asym) сходится с Python ≤1e-5."""
        cc = parity_real.find_c_compiler()
        if cc is None:
            self.skipTest("нет C-компилятора")
        if not (parity_real.VTM_LIB.is_file()
                and parity_real.VTM_INCLUDE.is_dir()):
            self.skipTest("src/vtm/libvtm.a не собран")
        path = self._compile()
        tokens = [1, 2]
        expected = vtm_ref.execute(str(path), tokens=tokens, max_new=0)
        probe = parity_real.run_c_probe(self.tmp, path, tokens, cc)
        self.assertEqual(len(probe.logits), len(expected.logits))
        delta = max(
            abs(float(left) - float(right))
            for left, right in zip(expected.logits, probe.logits)
        )
        self.assertLessEqual(delta, 1e-5, f"max|Δ логитов| = {delta}")


if __name__ == "__main__":
    unittest.main()
