"""Потоковая запись Q4 (kind 5) — задача 0155.

Проблема, которую эти тесты фиксируют: `decode_q4_segment_ex` отдавал
`list[int]` на весь тензор, а `write_vtm` поверх него строил второй список
через `pack_q4`. На полном 3B Q4 это 2,77 млрд значений: ~22 ГиБ списка
нибблов плюс ещё столько же при упаковке, то есть контейнер физически не
собирался на машине с 16 ГиБ.

Проверяется:

* `PackedQ4`/`PackedF32` — окна в байты сегмента: `len`, индексация, срезы,
  итерация и сравнение со списком работают как у списка (мелкие тесты не
  меняются), а `unpack_q4`/`pack_q4` на весь тензор не вызываются;
* `decode_q4_segment_ex` больше не материализует список — это проверяется
  подменой `unpack_q4` на «взрыв»;
* `iter_q4_payload` и `encode_q4_segment` дают одинаковые байты, а
  `write_vtm` пишет Q4 потоком; контейнер байт-в-байт равен эталонной
  сборке через `encode_tensor_payload` (sha256), с `integrity` и без;
* пик памяти писателя не зависит от размера тензора (tracemalloc);
* паритет C/Python на одном шаге для Q4-контейнера (≤1e-5), если собран
  `src/vtm/libvtm.a` и есть C-компилятор.

Реальный 3B здесь не собирается (это делает отчёт задачи): тесты держат
инварианты, а не гигабайты.
"""
from __future__ import annotations

import hashlib
import math
import os
import struct
import sys
import tempfile
import tracemalloc
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tests.test_gguf_import import build_tiny_model  # noqa: E402
from tools import gguf_import as gi  # noqa: E402
from tools import parity_real, vtm_compile, vtm_ref  # noqa: E402


def _q4values(n: int, seed: int = 5) -> list[int]:
    state = seed
    values = []
    for _ in range(n):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        values.append((state >> 16) & 0x0F)
    return values


def _scales(n: int) -> list[float]:
    n_groups = vtm_ref.n_scale_blocks(n, vtm_ref.Q4_SCALE_BLOCK)
    return [0.5 + 0.25 * i for i in range(n_groups)]


class PackedQ4CodecTests(unittest.TestCase):
    """Ленивые окна: те же значения, что у списка, без списка."""

    def test_packed_view_equals_list(self):
        for n in (0, 1, 15, 16, 31, 32, 33, 37, 64, 96):
            with self.subTest(n=n):
                values = _q4values(n)
                packed = vtm_ref.pack_q4(values)
                view = vtm_ref.PackedQ4(packed, n)
                self.assertEqual(len(view), n)
                self.assertEqual(list(view), values)
                self.assertEqual(view, values)
                self.assertEqual(values, list(view))
                for index in range(n):
                    self.assertEqual(view[index], values[index])
                if n:
                    self.assertEqual(view[-1], values[-1])
                self.assertEqual(view[:7], values[:7])
                with self.assertRaises(IndexError):
                    view[n]

    def test_partial_group_padding_is_q4_zero(self):
        # `pack_q4` добивает хвост нибблом 8; окно на тех же байтах обязано
        # отдать те же байты (иначе контейнер разошёлся бы с текущим путём).
        values = _q4values(37)
        packed = vtm_ref.pack_q4(values)
        view = vtm_ref.PackedQ4(packed, 37)
        self.assertEqual(view.to_packed(), packed)
        self.assertEqual(vtm_ref.pack_q4(view), packed)
        self.assertEqual(b"".join(view.iter_packed()), packed)
        last = packed[-vtm_ref.Q4_GROUP_BYTES:]
        self.assertEqual(last[5], 0x88)

    def test_iter_packed_chunks_cover_payload(self):
        values = _q4values(200)
        packed = vtm_ref.pack_q4(values)
        view = vtm_ref.PackedQ4(packed, 200)
        for chunk in (1, 7, 16, 64, 1 << 20):
            with self.subTest(chunk=chunk):
                pieces = list(view.iter_packed(chunk))
                self.assertEqual(b"".join(pieces), packed)
                # Куски — не гигантские: последний не больше chunk.
                for piece in pieces[:-1]:
                    self.assertLessEqual(len(piece), max(chunk, 16))

    def test_decode_returns_windows_not_lists(self):
        n = 96
        q = _q4values(n, seed=9)
        scales = _scales(n)
        tensor = vtm_ref.Tensor.q4("w", (n,), q, scales)
        segment = vtm_ref.encode_q4_segment(tensor)

        original = vtm_ref.unpack_q4

        def boom(*args, **kwargs):  # pragma: no cover - сюда попадать нельзя
            raise AssertionError("decode_q4_segment_ex материализует список")

        vtm_ref.unpack_q4 = boom
        try:
            qdata, got_scales, dtype, asym, mins = (
                vtm_ref.decode_q4_segment_ex(segment)
            )
        finally:
            vtm_ref.unpack_q4 = original
        self.assertIsInstance(qdata, vtm_ref.PackedQ4)
        self.assertIsInstance(got_scales, vtm_ref.PackedF32)
        self.assertEqual(qdata, q)
        self.assertEqual(got_scales, scales)
        self.assertEqual(dtype, "float32")
        self.assertFalse(asym)
        self.assertIsNone(mins)

    def test_legacy_list_wrapper(self):
        n = 64
        q = _q4values(n, seed=3)
        tensor = vtm_ref.Tensor.q4("w", (n,), q, _scales(n))
        segment = vtm_ref.encode_q4_segment(tensor)
        qdata, scales, dtype, asym, mins = vtm_ref.decode_q4_segment_values(
            segment
        )
        self.assertIsInstance(qdata, list)
        self.assertIsInstance(scales, list)
        self.assertEqual(qdata, q)
        self.assertEqual(dtype, "float32")
        self.assertFalse(asym)
        self.assertIsNone(mins)

    def test_tensor_keeps_windows(self):
        n = 64
        q = _q4values(n)
        scales = _scales(n)
        packed = vtm_ref.pack_q4(q)
        view = vtm_ref.PackedQ4(packed, n)
        scale_view = vtm_ref.PackedF32(
            struct.pack(f"<{len(scales)}f", *scales), 0, len(scales)
        )
        tensor = vtm_ref.Tensor.q4("w", (n,), view, scale_view)
        self.assertIs(tensor.qdata, view)
        self.assertIs(tensor.scales, scale_view)
        self.assertEqual(tensor.dense()[0], (q[0] - 8) * scales[0])

    def test_tensor_rejects_short_window(self):
        view = vtm_ref.PackedQ4(b"\x00" * 16, 32)
        with self.assertRaisesRegex(ValueError, "значений"):
            vtm_ref.Tensor.q4("w", (64,), view, [1.0, 1.0])


class PackedInt8Tests(unittest.TestCase):
    """Таблица словаря 3B — 311 млн int8: окно вместо `list[int]`."""

    def test_int8_decode_is_window_and_matches_list(self):
        n = 640
        q = [((i * 37) % 254) - 127 for i in range(n)]
        scales = [0.5 + 0.125 * i
                  for i in range(vtm_ref.n_scale_blocks(n, vtm_ref.INT8_SCALE_BLOCK))]
        tensor = vtm_ref.Tensor.int8("emb.w", (n,), q, scales)
        segment = vtm_ref.encode_int8_segment(tensor)

        qdata, got_scales, dtype = vtm_ref.decode_int8_segment(segment)
        self.assertIsInstance(qdata, vtm_ref.PackedInt8)
        self.assertIsInstance(got_scales, vtm_ref.PackedF32)
        self.assertEqual(qdata, q)
        self.assertEqual(got_scales, scales)
        self.assertEqual(dtype, "float32")
        self.assertEqual(qdata[3], q[3])
        self.assertEqual(qdata[-1], q[-1])
        self.assertEqual(sorted(set(qdata[:5])), sorted(set(q[:5])))

        # Байты сегмента не зависят от представления.
        self.assertEqual(vtm_ref.encode_int8_segment(tensor),
                         vtm_ref.encode_int8_segment(
                             vtm_ref.Tensor.int8("emb.w", (n,), q, scales)))
        self.assertEqual(b"".join(vtm_ref.iter_int8_payload(tensor)),
                         vtm_ref.encode_int8_segment(tensor))

    def test_int8_tensor_keeps_window(self):
        n = 256
        q = [1] * n
        scales = [0.125] * vtm_ref.n_scale_blocks(n, vtm_ref.INT8_SCALE_BLOCK)
        packed = bytes(value & 0xFF for value in q)
        view = vtm_ref.PackedInt8(packed, 0, n)
        scale_view = vtm_ref.PackedF32(
            struct.pack(f"<{len(scales)}f", *scales), 0, len(scales)
        )
        tensor = vtm_ref.Tensor.int8("emb.w", (n,), view, scale_view)
        self.assertIs(tensor.qdata, view)
        self.assertIs(tensor.scales, scale_view)
        self.assertEqual(tensor.dense()[0], 0.125)

    def test_int8_segment_rejects_q_below_minus_127(self):
        n = 256
        q = [0] * n
        scales = [1.0] * vtm_ref.n_scale_blocks(n, vtm_ref.INT8_SCALE_BLOCK)
        segment = bytearray(vtm_ref.encode_int8_segment(
            vtm_ref.Tensor.int8("emb.w", (n,), q, scales)
        ))
        segment[18 + 4 * len(scales) + 5] = 0x80  # -128
        with self.assertRaisesRegex(ValueError, r"q #5 = -128"):
            vtm_ref.decode_int8_segment(bytes(segment))


class StreamPayloadTests(unittest.TestCase):
    """`iter_q4_payload` — те же байты, что у `encode_q4_segment`."""

    def _segment_tensor(self, n=200, *, asym=False, dtype="float32"):
        q = _q4values(n, seed=13)
        scales = _scales(n)
        mins = [(-0.25 * i) for i in range(len(scales))] if asym else None
        packed = vtm_ref.pack_q4(q)
        # Тензор с ленивыми окнами: так его отдаёт decode_q4_segment_ex.
        tensor = vtm_ref.Tensor.q4(
            "w", (n,), vtm_ref.PackedQ4(packed, n), scales,
            scale_dtype=dtype, asymmetric=asym, mins=mins,
        )
        return tensor, q, scales, packed

    def test_stream_equals_encoded_segment(self):
        for asym in (False, True):
            for dtype in ("float32", "float16"):
                with self.subTest(asym=asym, dtype=dtype):
                    tensor, _q, _s, _p = self._segment_tensor(
                        200, asym=asym, dtype=dtype
                    )
                    streamed = b"".join(vtm_ref.iter_q4_payload(tensor, chunk=7))
                    self.assertEqual(streamed, vtm_ref.encode_q4_segment(tensor))
                    self.assertEqual(
                        len(streamed), vtm_ref.tensor_payload_size(tensor)
                    )

    def test_list_and_window_payloads_match(self):
        tensor, q, scales, _p = self._segment_tensor()
        listed = vtm_ref.Tensor.q4("w", (200,), q, scales)
        self.assertEqual(
            vtm_ref.encode_q4_segment(tensor),
            vtm_ref.encode_q4_segment(listed),
        )


class StreamWriterTests(unittest.TestCase):
    """`write_vtm` пишет Q4 потоком, но контейнер побайтово прежний."""

    def _tensors(self, n: int = 4096):
        q = _q4values(n, seed=21)
        scales = _scales(n)
        packed = vtm_ref.pack_q4(q)
        lazy = vtm_ref.Tensor.q4(
            "q4.w", (n,), vtm_ref.PackedQ4(packed, n), scales
        )
        listed = vtm_ref.Tensor.q4("q4.w", (n,), q, scales)
        return lazy, listed

    def _extra(self, lazy_int8: bool):
        # int8-таблица — тоже ленивая: у 3B это 311 млн значений, список на
        # них стоил бы ~2,5 ГиБ ещё до записи контейнера.
        q = [((i * 91) % 254) - 127 for i in range(256)]
        scales = [0.25] * vtm_ref.n_scale_blocks(256, vtm_ref.INT8_SCALE_BLOCK)
        if lazy_int8:
            view = vtm_ref.PackedInt8(bytes(value & 0xFF for value in q), 0, 256)
            embed = vtm_ref.Tensor.int8("emb.w", (256,), view, scales)
        else:
            embed = vtm_ref.Tensor.int8("emb.w", (256,), q, scales)
        return [
            vtm_ref.Tensor.ternary("t.w", (64,), [0, 1, -1] * 21 + [0]),
            vtm_ref.Tensor.f32("x", (4,), [0.5, -0.25, 0.0, 1.0]),
            vtm_ref.Tensor.kv_cache("kv.k", (4, 4)),
            embed,
        ]

    def _program(self, tensors):
        return [
            vtm_ref.Instr(op=int(vtm_ref.Op.EMB), dst=1, a=0, imm=1),
            vtm_ref.Instr(op=int(vtm_ref.Op.HALT)),
        ]

    def test_container_bytes_match_reference(self):
        lazy, listed = self._tensors()
        program = self._program(None)
        for integrity in (False, True):
            with self.subTest(integrity=integrity):
                with tempfile.TemporaryDirectory(prefix="q4-stream-") as tmp:
                    path_a = Path(tmp) / "lazy.vtm"
                    path_b = Path(tmp) / "listed.vtm"
                    meta = {"model": "test", "context_len": 4}
                    vtm_ref.write_vtm(
                        path_a, [lazy] + self._extra(lazy_int8=True), program,
                        meta, integrity=integrity,
                    )
                    vtm_ref.write_vtm(
                        path_b, [listed] + self._extra(lazy_int8=False), program,
                        meta, integrity=integrity,
                    )
                    bytes_a = path_a.read_bytes()
                    bytes_b = path_b.read_bytes()
                    self.assertEqual(len(bytes_a), len(bytes_b))
                    self.assertEqual(
                        hashlib.sha256(bytes_a).hexdigest(),
                        hashlib.sha256(bytes_b).hexdigest(),
                    )
                    loaded = vtm_ref.read_vtm(str(path_a))
                    q4 = next(
                        tensor for tensor in loaded.tensors
                        if tensor.kind == vtm_ref.KIND_Q4
                    )
                    self.assertEqual(list(q4.qdata), list(lazy.qdata))

    def test_writer_streams_instead_of_materializing(self):
        # 2M значений — 1 МиБ упакованных нибблов; список Python на них занял
        # бы ~16 МиБ + упаковка. Пик должен остаться заметно меньше.
        n = 2_000_000
        q = _q4values(n, seed=33)
        packed = vtm_ref.pack_q4(q)
        del q
        scales = _scales(n)
        lazy = vtm_ref.Tensor.q4(
            "q4.w", (n,), vtm_ref.PackedQ4(packed, n), scales
        )
        program = [vtm_ref.Instr(op=int(vtm_ref.Op.HALT))]
        original = vtm_ref.encode_q4_segment

        def boom(tensor):  # pragma: no cover - потоковый путь его не зовёт
            raise AssertionError("write_vtm материализует Q4-сегмент")

        with tempfile.TemporaryDirectory(prefix="q4-stream-big-") as tmp:
            path = Path(tmp) / "big.vtm"
            vtm_ref.encode_q4_segment = boom
            try:
                tracemalloc.start()
                vtm_ref.write_vtm(str(path), [lazy], program, integrity=False)
                _current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
            finally:
                vtm_ref.encode_q4_segment = original
            self.assertLess(peak, 8 << 20, f"пик {peak} байт на потоковом Q4")
            payload = vtm_ref.tensor_payload_size(lazy)
            expected = (
                vtm_ref.align_up(
                    vtm_ref.HEADER_SIZE + vtm_ref.TENSOR_ENTRY_SIZE
                )
                + vtm_ref.align_up(payload)
                + len(vtm_ref.Instr(op=int(vtm_ref.Op.HALT)).pack())
            )
            self.assertEqual(path.stat().st_size, expected)
            vtm_ref.read_vtm(str(path))


class _Q4SliceFixture(unittest.TestCase):
    """Крошечный Q4-экспорт импортом GGUF — общий для паритета."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="q4-parity-")
        cls.tmp = Path(cls._tmp.name)
        source = build_tiny_model(cls.tmp / "tiny.gguf", tie=False, layers=1)
        cls.export = cls.tmp / "export"
        gi.import_gguf(source, cls.export, fmt="q4")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def compile_slice(self, name: str = "model.vtm") -> Path:
        export = vtm_compile.load_export(self.export / "manifest.json")
        vtm = vtm_compile.compile_model(export.cfg, export.weights, export.meta)
        path = self.export / name
        vtm_ref.write_vtm(str(path), vtm.tensors, vtm.program, vtm.meta,
                          integrity=True)
        return path


class SliceParityTests(_Q4SliceFixture):
    """Паритет C/Python на одном шаге и читаемость нового контейнера."""

    def test_python_reads_back_streamed_q4(self):
        path = self.compile_slice("readback.vtm")
        loaded = vtm_ref.read_vtm(str(path))
        q4 = [t for t in loaded.tensors if t.kind == vtm_ref.KIND_Q4]
        self.assertTrue(q4)
        result = vtm_ref.execute(str(path), tokens=[1, 2], max_new=1)
        self.assertTrue(all(math.isfinite(v) for v in (result.logits or [])))

    def test_c_python_step_parity(self):
        cc = parity_real.find_c_compiler()
        if cc is None:
            self.skipTest("нет C-компилятора")
        if not (parity_real.VTM_LIB.is_file()
                and parity_real.VTM_INCLUDE.is_dir()):
            self.skipTest("src/vtm/libvtm.a не собран")
        binary = Path(ROOT, "src", "vtm", "vtm")
        if not binary.is_file():
            self.skipTest("CLI src/vtm/vtm не собран")
        path = self.compile_slice("parity.vtm")
        tokens = [1, 2]
        # max_new=0: логиты обязаны быть с последней позиции промпта —
        # ровно их читает C-проба после прогона тех же токенов. С max_new=1
        # Python вернул бы логиты уже после сгенерированного токена.
        expected = vtm_ref.execute(str(path), tokens=tokens, max_new=0)
        probe = parity_real.run_c_probe(
            self.tmp, path, tokens, cc
        )
        self.assertEqual(len(probe.logits), len(expected.logits))
        delta = max(
            abs(float(left) - float(right))
            for left, right in zip(expected.logits, probe.logits)
        )
        self.assertLessEqual(delta, 1e-5, f"max|Δ логитов| = {delta}")


if __name__ == "__main__":
    unittest.main()
