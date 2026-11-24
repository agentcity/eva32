"""7B Q4 в пределе 4 ГиБ: fp16-масштабы и тритовый эмбеддинг (0156).

Проблема, которую фиксируют тесты: предел контейнера — u32-смещения, 4 ГиБ
(ADR 0016). 7B Qwen2.5 Q4_K_M — 7,07 млрд весов в Linear+lm_head плюс
0,545 млрд в таблице словаря:

* fp32-масштабы Q4 + int8-эмбеддинг — 116 % предела (было единственное);
* fp16-масштабы Q4 + int8-эмбеддинг — 105,7 %;
* fp16-масштабы Q4 + тритовый эмбеддинг — ~95 %.

Формат (Q4-сегмент kind 5 с флагом fp16-масштабов, TRT1-таблица словаря)
это умел; импортёр не умел. Здесь проверяется, что `--q4-scale-dtype fp16`
и `--embed trit` дают байты, которые читает `tools/vtm_ref.py`, манифест,
который собирает `tools/vtm_compile.py`, и что сборка считает токен с
паритетом C↔Python ≤1e-5.

Реальный 7B здесь не импортируется (это ~90 минут и 4,4 ГБ блоба): тест
проверяет контракт формата и бюджет размеров, а полный прогон — в отчёте
задачи. Полный 7B можно включить переменной `VTM_GGUF_7B=1`, если блоб
ollama на месте.
"""
from __future__ import annotations

import json
import math
import os
import struct
import sys
import tempfile
import unittest
from array import array
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from tests.test_gguf_import import build_tiny_model  # noqa: E402
from tools import gguf_import as gi  # noqa: E402
from tools import parity_real, vtm_compile, vtm_ref  # noqa: E402

VTM_BIN = os.path.join(ROOT, "src", "vtm", "vtm")
REAL_7B_BLOB = Path.home() / ".ollama/models/blobs" / (
    "sha256-2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730"
)

#: Qwen2.5-7B-Instruct: hidden 3584, 28 слоёв, intermediate 18944, vocab 152064.
QWEN25_7B = {
    "hidden": 3584,
    "layers": 28,
    "intermediate": 18944,
    "vocab": 152064,
    "q_heads": 28,
    "kv_heads": 4,
    "head_dim": 128,
}
GIB = 1 << 30


def _q4_expected_bytes(
    params: int, scale_bytes: int, *, asym: bool = True
) -> int:
    """Размер Q4-сегментов без выравнивания: заголовки + масштабы [+ min] + нибблы.

    `asym=True` — Q4_1 (импорт по умолчанию с 0160): после масштабов лежит
    ещё массив min того же типа, то есть 2*scale_bytes на группу.
    """
    groups = (params + gi.Q4_SCALE_BLOCK - 1) // gi.Q4_SCALE_BLOCK
    aux = scale_bytes * (2 if asym else 1)
    header = gi.Q4_ASYM_HEADER_SIZE if asym else gi.Q4_HEADER_SIZE
    return header + groups * (aux + gi.Q4_GROUP_BYTES)


def _trit_expected_bytes(params: int) -> int:
    """Размер TRT1-сегментов: заголовок + fp32-масштабы + слова pack20."""
    scales = (params + gi.TRIT_SCALE_BLOCK - 1) // gi.TRIT_SCALE_BLOCK
    words = (params + gi.TRITS_PER_STORAGE_BLOCK - 1) // gi.TRITS_PER_STORAGE_BLOCK * 8
    return gi.TRT1_HEADER_SIZE + scales * 4 + words * 4


def qwen25_7b_weights() -> int:
    """Веса 7B вне таблицы словаря: Linear всех слоёв + lm_head."""
    cfg = QWEN25_7B
    hidden, inter = cfg["hidden"], cfg["intermediate"]
    q_dim = cfg["q_heads"] * cfg["head_dim"]
    kv_dim = cfg["kv_heads"] * cfg["head_dim"]
    per_layer = (
        q_dim * hidden + kv_dim * hidden + kv_dim * hidden + hidden * q_dim
        + inter * hidden + inter * hidden + hidden * inter
    )
    return cfg["layers"] * per_layer + cfg["vocab"] * hidden


def qwen25_7b_embed() -> int:
    return QWEN25_7B["vocab"] * QWEN25_7B["hidden"]


def qwen25_7b_total(
    scale_bytes: int, embed: str, *, asym: bool = True
) -> int:
    """Оценка размера контейнера 7B для выбранных схем квантования."""
    weights = qwen25_7b_weights()
    embed_params = qwen25_7b_embed()
    q4 = _q4_expected_bytes(weights, scale_bytes, asym=asym)
    if embed == "int8":
        groups = (embed_params + gi.INT8_SCALE_BLOCK - 1) // gi.INT8_SCALE_BLOCK
        payload = gi.Q4_HEADER_SIZE + groups * 4 + embed_params
    elif embed == "trit":
        payload = _trit_expected_bytes(embed_params)
    else:  # pragma: no cover - тест не про плотные таблицы
        payload = embed_params * 4
    # Нормы/bias: 141 F32-тензор Qwen2.5-7B, ~0,3 МЭ → 1,3 МиБ; плюс
    # выравнивание сегментов по 32 байта на ~200 тензоров. Это < 2 МиБ.
    return q4 + payload + (2 << 20)


class BudgetTests(unittest.TestCase):
    """Арифметика бюджета: какие схемы влезают в 4 ГиБ, а какие нет."""

    def test_weights_match_reported_7b(self) -> None:
        # В отчёте задачи — «7,07 млрд весов»; проверяем, что формула та же.
        self.assertAlmostEqual(
            qwen25_7b_weights() / 1e9, 7.071, places=2
        )
        self.assertAlmostEqual(
            qwen25_7b_embed() / 1e6, 545.0, places=1
        )

    def test_only_fp16_scales_with_trit_embed_fit(self) -> None:
        """Симметричный Q4_0 (--q4-kind q4_0) — единственная схема для 7B."""
        fp32_int8 = qwen25_7b_total(4, "int8", asym=False)
        fp16_int8 = qwen25_7b_total(2, "int8", asym=False)
        fp16_trit = qwen25_7b_total(2, "trit", asym=False)
        self.assertGreater(fp32_int8, 4 * GIB)
        self.assertGreater(fp16_int8, 4 * GIB)
        self.assertLessEqual(fp16_trit, 4 * GIB)
        # Числа из отчёта задачи 0156 (в пределах выравнивания и округления).
        self.assertAlmostEqual(fp32_int8 / 4_982_296_768, 1.0, places=2)
        self.assertAlmostEqual(fp16_int8 / 4_540_403_904, 1.0, places=2)
        self.assertLess(fp16_trit / (4 * GIB), 0.97,
                        "по задаче — 95 % от 4 ГиБ")

    def test_q4_1_does_not_fit_7b(self) -> None:
        """Цена Q4_1 (0160): 7B даже с fp16-масштабами и тритами > 4 ГиБ.

        min на группу 32 при fp16-масштабах — это +2 байта на 32 значения,
        +0,0625 Б/вес; у 7,07 млрд весов ~0,44 ГБ: 4,10 → 4,54 ГБ, то есть
        106 % предела u32-смещений (ADR 0016). Поэтому импорт 7B остаётся на
        `--q4-kind q4_0`, а Q4_1 — умолчание для моделей, которые влезают
        (3B: ~2,4 ГБ).
        """
        fp16_trit_asym = qwen25_7b_total(2, "trit", asym=True)
        fp16_trit_sym = qwen25_7b_total(2, "trit", asym=False)
        self.assertGreater(fp16_trit_asym, 4 * GIB)
        self.assertGreater(
            fp16_trit_asym - fp16_trit_sym,
            qwen25_7b_weights() // 16,  # 2 байта min на группу 32, без заголовков
        )
        # 3B Q4_1 с fp32-масштабами и int8-эмбеддингом — с запасом.
        three_b = _q4_expected_bytes(2_359_300_000, 4, asym=True) + (750 << 20)
        self.assertLess(three_b, 3 * GIB)


class Fp16ScaleTests(unittest.TestCase):
    """Q4-сегмент с fp16-масштабами: flags, байты, разбор эталоном."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="gguf-fit-")
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _import(self, name: str = "export", **kwargs) -> gi.ImportResult:
        source = build_tiny_model(self.tmp / "tiny.gguf", tie=False, layers=1)
        return gi.import_gguf(source, self.tmp / name, fmt="q4", **kwargs)

    def test_fp16_segments_decode_and_match_reference_bytes(self) -> None:
        """Оба вида Q4: байты сегмента = эталонный писатель 0151.

        `q4_1` — умолчание (0160), `q4_0` — режим 7B (бюджет 4 ГиБ).
        """
        for kind, magic, asym_expected in (
            ("q4_1", b"Q4_1", True),
            ("q4_0", b"Q4_0", False),
        ):
            with self.subTest(kind=kind):
                self._check_fp16_segments(kind, magic, asym_expected)

    def _check_fp16_segments(
        self, kind: str, magic_expected: bytes, asym_expected: bool
    ) -> None:
        result = self._import(f"fp16-{kind}", q4_scale_dtype="fp16",
                              q4_kind=kind)
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["q4"]["scale_dtype"], "float16")
        self.assertEqual(manifest["q4"]["flags"], 0x00)
        self.assertEqual(manifest["q4"]["asymmetric"], asym_expected)
        self.assertEqual(manifest["quantization"]["q4_scale_dtype"], "float16")
        self.assertEqual(manifest["quantization"]["q4_kind"], kind)
        self.assertEqual(manifest["compiler"]["ready"], True)

        header_size = (
            gi.Q4_ASYM_HEADER_SIZE if asym_expected else gi.Q4_HEADER_SIZE
        )
        raw = result.q4_path.read_bytes()
        checked = 0
        for entry in manifest["tensors"]:
            if entry.get("kind") != "q4":
                continue
            self.assertEqual(entry["dtype"], kind)
            segment = raw[entry["q4_offset"]:
                          entry["q4_offset"] + entry["q4_length"]]
            magic, version, flags, n_values, n_groups = struct.unpack(
                "<4sBBQI", segment[:gi.Q4_HEADER_SIZE]
            )
            self.assertEqual(magic, magic_expected)
            self.assertEqual(version, gi.Q4_VERSION)
            self.assertEqual(flags, 0x00)  # бит 0 снят → fp16
            self.assertEqual(n_values, entry["params"])
            self.assertEqual(
                n_groups,
                (entry["params"] + gi.Q4_SCALE_BLOCK - 1) // gi.Q4_SCALE_BLOCK,
            )
            if asym_expected:
                (n_mins,) = struct.unpack(
                    "<I", segment[gi.Q4_HEADER_SIZE:gi.Q4_ASYM_HEADER_SIZE]
                )
                self.assertEqual(n_mins, n_groups)
            qdata, scales, scale_dtype, asym, mins = (
                vtm_ref.decode_q4_segment_ex(segment)
            )
            self.assertEqual(scale_dtype, "float16")
            self.assertEqual(asym, asym_expected)
            if asym_expected:
                self.assertEqual(len(mins), n_groups)
                for minimum in mins:
                    self.assertEqual(
                        struct.unpack("<e", struct.pack("<e", minimum))[0],
                        minimum,
                    )
            else:
                self.assertIsNone(mins)
            self.assertEqual(len(qdata), entry["params"])
            self.assertEqual(len(scales), n_groups)
            # Каждый масштаб — точное fp16: округление сделано до квантования
            # нибблов, второго округления при записи быть не должно.
            for scale in scales:
                self.assertGreaterEqual(scale, 0.0)
                self.assertEqual(
                    struct.unpack("<e", struct.pack("<e", scale))[0], scale
                )
            # Байты сегмента совпадают с эталонным писателем 0151 на тех же
            # значениях: это и есть контракт формата (флаг + fp16-масштабы).
            expected_len = (
                header_size
                + n_groups * gi.Q4_SCALE_BYTES["float16"]
                * (2 if asym_expected else 1)
                + n_groups * gi.Q4_GROUP_BYTES
            )
            self.assertEqual(entry["q4_length"], expected_len)
            tensor = vtm_ref.Tensor.q4(
                "w", (entry["params"],), qdata, scales, "float16",
                asymmetric=asym_expected, mins=mins,
            )
            self.assertEqual(
                vtm_ref.encode_q4_segment(tensor),
                segment[:expected_len],
            )
            checked += 1
            if checked >= 3:
                break
        self.assertGreater(checked, 0)

    def test_fp32_is_still_default(self) -> None:
        result = self._import("default")
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["q4"]["scale_dtype"], "float32")
        self.assertEqual(manifest["q4"]["flags"], 0x01)
        first = next(
            entry for entry in manifest["tensors"] if entry.get("kind") == "q4"
        )
        raw = result.q4_path.read_bytes()
        segment = raw[first["q4_offset"]:
                      first["q4_offset"] + first["q4_length"]]
        _q, _s, scale_dtype, _a, _m = vtm_ref.decode_q4_segment_ex(segment)
        self.assertEqual(scale_dtype, "float32")

    def test_fp16_costs_half_of_fp32(self) -> None:
        fp32 = self._import("fp32", q4_scale_dtype="fp32")
        fp16 = self._import("fp16", q4_scale_dtype="fp16")
        self.assertLess(fp16.q4_bytes, fp32.q4_bytes)
        params = fp32.q4_params
        groups = math.ceil(params / gi.Q4_SCALE_BLOCK)
        # Q4_1 (умолчание) пишет и масштабы, и min — экономия fp16 удвоена.
        self.assertEqual(
            fp32.q4_bytes - fp16.q4_bytes,
            groups * (4 - 2) * 2,
        )

    def test_scale_overflow_in_fp16_is_refused(self) -> None:
        # absmax/7 > 65504 → в fp16 это inf; молча писать нельзя.
        block = array("f", [1.0e6] * 32)
        with self.assertRaises(gi.GgufError) as ctx:
            gi._q4_encode_block(block, "float16")
        self.assertIn("float16", str(ctx.exception))
        scale, packed = gi._q4_encode_block(block, "float32")
        self.assertGreater(scale, 0.0)
        self.assertEqual(len(packed), gi.Q4_GROUP_BYTES)

    def test_unknown_scale_dtype_is_refused(self) -> None:
        with self.assertRaises(gi.GgufError) as ctx:
            gi.import_gguf(
                build_tiny_model(self.tmp / "tiny.gguf", tie=False, layers=1),
                self.tmp / "bad",
                fmt="q4",
                q4_scale_dtype="bf16",
            )
        self.assertIn("q4_scale_dtype", str(ctx.exception))


class TritEmbedTests(unittest.TestCase):
    """Тритовый эмбеддинг: сегмент в ternary.trit, манифест, потеря."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="gguf-trit-emb-")
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _import(self, name: str, **kwargs) -> tuple[gi.ImportResult, dict]:
        source = build_tiny_model(
            self.tmp / "tiny.gguf", tie=False, layers=1
        )
        result = gi.import_gguf(
            source, self.tmp / name, fmt="q4", stats=True, **kwargs
        )
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        return result, manifest

    def test_trit_embed_manifest_and_segment(self) -> None:
        result, manifest = self._import(
            "trit", embed_mode="trit", q4_scale_dtype="fp16"
        )
        self.assertEqual(manifest["architecture"]["embed"], "trit")
        self.assertEqual(manifest["embeddings"]["mode"], "trit")
        self.assertEqual(manifest["embeddings"]["dtype"], "trit")
        self.assertEqual(manifest["embeddings"]["storage"], "ternary.trit")
        self.assertEqual(manifest["embeddings"]["block_size"],
                         gi.TRIT_SCALE_BLOCK)
        self.assertTrue(manifest["trit"]["contains_embed"])
        self.assertGreater(manifest["files"]["ternary.trit"]["bytes"], 0)
        self.assertIsNotNone(manifest["q4"])

        # Запись эмбеддинга в non_ternary.bin быть не должно.
        non_entries = manifest["non_ternary"]["tensors"]
        self.assertFalse(
            any(entry["role"] == "embed" for entry in non_entries)
        )
        # tied lm_head нет: модель untied, lm_head — Q4-сегмент.
        self.assertFalse(
            any(entry.get("tied_to") for entry in non_entries)
        )

        embed = next(
            entry for entry in manifest["tensors"] if entry["role"] == "embed"
        )
        self.assertEqual(embed["dtype"], "trit")
        self.assertEqual(embed["params"], 64 * 32)

        raw = (self.tmp / "trit" / "ternary.trit").read_bytes()
        segment = raw[embed["trit_offset"]:
                      embed["trit_offset"] + embed["trit_length"]]
        magic, version, flags, n_values, n_scale = struct.unpack(
            "<4sBBQI", segment[:gi.TRT1_HEADER_SIZE]
        )
        self.assertEqual(magic, b"TRT1")
        self.assertEqual(version, gi.TRT1_VERSION)
        self.assertEqual(flags, 0x01)  # fp32-масштабы TRT1
        self.assertEqual(n_values, embed["params"])
        self.assertEqual(n_scale, math.ceil(embed["params"] / 128))
        trits, scales, scale_dtype = vtm_ref.decode_trit_segment(segment)
        self.assertEqual(scale_dtype, "float32")
        self.assertEqual(len(trits), embed["params"])
        self.assertTrue(all(trit in (-1, 0, 1) for trit in trits))
        self.assertEqual(len(scales), n_scale)

    def test_trit_embed_loss_is_measured_against_int8(self) -> None:
        int8_result, _ = self._import("int8", embed_mode="int8")
        trit_result, manifest = self._import("trit", embed_mode="trit")
        self.assertIn("int8", int8_result.stats)
        self.assertIn("embed(trit)", trit_result.stats)
        int8_loss = int8_result.stats["int8"].rel_rms
        trit_loss = trit_result.stats["embed(trit)"].rel_rms
        self.assertGreater(int8_loss, 0.0)
        self.assertGreater(trit_loss, 0.0)
        # Триты грубее int8: потеря таблицы против исходника обязана быть
        # больше — иначе замер бессмыслен.
        self.assertGreater(trit_loss, int8_loss)
        # Для Гауссовых весов трит (±scale/0) даёт ~40–55 % rel-RMS: это
        # ожидаемая цена, а не ошибка замера. Верхняя граница — чтобы тест
        # ловил сломанный квантователь, а не фиксировал конкретное число.
        self.assertLess(trit_loss, 0.6)
        # И та же ступень видна в размере: триты примерно вчетверо дешевле.
        self.assertLess(
            manifest["embeddings"]["bytes"],
            int8_result.manifest["embeddings"]["bytes"] / 3.5,
        )

    def test_trit_embed_scale_rule_is_twn(self) -> None:
        # Эмбеддинг квантуется правилом 1,5·mean(|w|), а не absmax: на
        # Q6_K-таблице 7B это 49 % rel-RMS вместо 77 % при тех же байтах.
        values = array("f", [1.0, -0.5, 0.25, -0.1] * 32)
        mean_encoder = gi._TritEncoder(scale_rule="mean")
        mean_encoder.add_values(values)
        expected = 1.5 * sum(abs(v) for v in values) / len(values)
        self.assertAlmostEqual(mean_encoder.scales[0], expected, places=5)

        absmax_encoder = gi._TritEncoder(scale_rule="absmax")
        absmax_encoder.add_values(values)
        self.assertAlmostEqual(absmax_encoder.scales[0], 1.0, places=6)

        # И на «тяжёлых хвостах» mean-правило действительно точнее.
        import random

        rng = random.Random(17)
        sample = array(
            "f",
            (rng.gauss(0.0, 1.0) for _ in range(4096)),
        )
        losses = {}
        for rule in ("absmax", "mean"):
            encoder = gi._TritEncoder(collect_stats=True, scale_rule=rule)
            encoder.add_values(sample)
            losses[rule] = encoder.stats.rel_rms
        self.assertLess(losses["mean"], losses["absmax"])

    def test_unknown_embed_mode_is_refused(self) -> None:
        with self.assertRaises(gi.GgufError) as ctx:
            gi.import_gguf(
                build_tiny_model(self.tmp / "tiny.gguf", tie=False, layers=1),
                self.tmp / "bad",
                fmt="q4",
                embed_mode="q8",
            )
        self.assertIn("embed", str(ctx.exception))


class EndToEndTritEmbedTests(unittest.TestCase):
    """fp16-масштабы + тритовый эмбеддинг: импорт → компиляция → 1 токен."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="gguf-fit-e2e-")
        cls.tmp = Path(cls._tmp.name)
        source = build_tiny_model(cls.tmp / "tiny.gguf", tie=False, layers=1)
        cls.export = cls.tmp / "export"
        gi.import_gguf(
            source, cls.export, fmt="q4",
            q4_scale_dtype="fp16", embed_mode="trit",
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _compile(self) -> Path:
        export = vtm_compile.load_export(self.export / "manifest.json")
        model = vtm_compile.compile_model(
            export.cfg, export.weights, export.meta
        )
        path = self.export / "model.vtm"
        vtm_ref.write_vtm(
            str(path), model.tensors, model.program, model.meta,
            integrity=True,
        )
        return path

    def test_container_reads_back_and_python_runs(self) -> None:
        path = self._compile()
        loaded = vtm_ref.read_vtm(str(path))
        embed = next(
            tensor for tensor in loaded.tensors
            if tensor.kind == vtm_ref.KIND_TRITS
        )
        self.assertEqual(tuple(embed.dims), (64, 32))
        self.assertEqual(loaded.meta["embed"], "trit")
        q4 = [t for t in loaded.tensors if t.kind == vtm_ref.KIND_Q4]
        self.assertTrue(q4)
        self.assertTrue(all(t.scale_dtype == "float16" for t in q4))
        result = vtm_ref.execute(str(path), tokens=[1, 2], max_new=1)
        self.assertTrue(all(math.isfinite(v) for v in (result.logits or [])))

    def test_c_python_step_parity(self) -> None:
        cc = parity_real.find_c_compiler()
        if cc is None:
            self.skipTest("нет C-компилятора")
        if not (parity_real.VTM_LIB.is_file()
                and parity_real.VTM_INCLUDE.is_dir()):
            self.skipTest("src/vtm/libvtm.a не собран")
        if not Path(VTM_BIN).is_file():
            self.skipTest("CLI src/vtm/vtm не собран")
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
        # argmax тоже обязан совпасть: это уже про «токен», а не про допуск.
        self.assertEqual(
            max(range(len(probe.logits)), key=lambda i: probe.logits[i]),
            max(range(len(expected.logits)), key=lambda i: expected.logits[i]),
        )


@unittest.skipUnless(
    os.environ.get("VTM_GGUF_7B") == "1",
    "полный 7B: ~90 минут импорта; включается VTM_GGUF_7B=1",
)
class Real7BTests(unittest.TestCase):
    """Полный 7B Q4: размер контейнера и проходимость компиляции.

    По умолчанию пропущен: это не юнит-тест, а приёмочный прогон отчёта
    (в отчёте 0156 приведены числа и команды).
    """

    @classmethod
    def setUpClass(cls) -> None:
        if not REAL_7B_BLOB.is_file():
            raise unittest.SkipTest(f"нет блоба {REAL_7B_BLOB}")
        cls._tmp = tempfile.TemporaryDirectory(prefix="gguf-7b-")
        cls.tmp = Path(cls._tmp.name)
        cls.export = cls.tmp / "export"
        cls.result = gi.import_gguf(
            REAL_7B_BLOB, cls.export, fmt="q4",
            q4_scale_dtype="fp16", embed_mode="trit",
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_payload_files_fit_4gib(self) -> None:
        # Файлы весов (q4.bin + ternary.trit + non_ternary.bin) — верхняя
        # оценка: контейнер добавляет только заголовок и таблицу тензоров.
        manifest = self.result.manifest
        total = sum(record["bytes"] for record in manifest["files"].values())
        self.assertLessEqual(total, 4 * GIB, f"файлы весов {total} байт")
        self.assertEqual(manifest["architecture"]["embed"], "trit")
        self.assertTrue(manifest["compiler"]["ready"])

    def test_full_compile_and_one_token(self) -> None:
        export = vtm_compile.load_export(self.export / "manifest.json")
        model = vtm_compile.compile_model(
            export.cfg, export.weights, export.meta
        )
        path = self.tmp / "7b.vtm"
        vtm_ref.write_vtm(
            str(path), model.tensors, model.program, model.meta,
            integrity=True,
        )
        self.assertLessEqual(path.stat().st_size, 4 * GIB)
        if not Path(VTM_BIN).is_file():
            self.skipTest("CLI src/vtm/vtm не собран")
        proc = __import__("subprocess").run(
            [VTM_BIN, "run", str(path), "--tokens", "1,2", "--max-new", "1"],
            capture_output=True, text=True, timeout=600,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])


if __name__ == "__main__":
    unittest.main()
