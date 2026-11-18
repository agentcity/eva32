"""Тесты Python-эталона VTM: контейнер, кодек, операции, синтетика.

Запуск:
    python3 -m unittest tests.test_vtm_ref -v
    python3 -m unittest discover -s tests

Без torch и numpy: только stdlib. Контрольные числа синтетики зафиксированы
из `tools/vtm_compile.py --synthetic` (seed=1, size=128) — именно с ними
сверяется C-интерпретатор.
"""

import math
import os
import random
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools import trit_codec, vtm_compile, vtm_ref  # noqa: E402

TOOLS = os.path.join(ROOT, "tools")
QWEN_CONFIG = os.path.join(ROOT, "qat", "configs", "qwen1.5b.yaml")


def _run(tensors, program, meta=None, tokens=(), max_new=0):
    """Собрать VTM в памяти и исполнить — без файла."""
    vtm = vtm_ref.VTM(tensors=list(tensors), program=list(program), meta=dict(meta or {}))
    return vtm_ref.execute(vtm, tokens, max_new)


def _f32(value):
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


class _TmpCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmpdir = self._tmp.name

    def path(self, name):
        return os.path.join(self.tmpdir, name)


# ---------------------------------------------------------------------------
# Кодек тритов: контрольные векторы docs/trit-format.md §6
# ---------------------------------------------------------------------------


class TestTritCodecControlVectors(unittest.TestCase):
    def test_control_vectors(self):
        # 20 тритов -> одно слово uint32 little-endian.
        cases = [
            ([-1] * 20, 0, "00000000"),
            ([0] * 20, 1_743_392_200, "c80dea67"),
            ([1] * 20, 3_486_784_400, "901bd4cf"),
            ([1, -1] * 10, 871_696_100, "e406f533"),
        ]
        for trits, value, hex_bytes in cases:
            with self.subTest(trits=trits[:4]):
                packed = vtm_ref.pack20(trits)
                self.assertEqual(len(packed), 4)
                self.assertEqual(int.from_bytes(packed, "little"), value)
                self.assertEqual(packed.hex(), hex_bytes)

    def test_pack20_is_trit_codec(self):
        # vtm_ref не копирует кодек, а импортирует его.
        self.assertIs(vtm_ref.pack20, trit_codec.pack20)
        self.assertIs(vtm_ref.unpack20, trit_codec.unpack20)

    def test_160_trits_are_32_bytes(self):
        self.assertEqual(len(vtm_ref.pack20([1] * 160)), 32)
        self.assertEqual(vtm_ref.unpack20(vtm_ref.pack20([-1, 0, 1] * 60), 180),
                         [-1, 0, 1] * 60)

    def test_absmax_scales_by_128(self):
        values = [0.5] * 64 + [-4.0] + [0.1] * 63
        trits, scales = vtm_ref.quantize_trits(values)
        self.assertEqual(len(scales), 1)
        self.assertEqual(scales[0], 4.0)
        self.assertEqual(trits[64], -1)
        self.assertTrue(set(trits) <= {-1, 0, 1})
        self.assertEqual(vtm_ref.quantize_trits([0.0] * 128), ([0] * 128, [0.0]))


# ---------------------------------------------------------------------------
# TRT1-сегмент
# ---------------------------------------------------------------------------


class TestTritSegment(unittest.TestCase):
    def test_segment_header_and_size(self):
        tensor = vtm_ref.Tensor.ternary("w", (3, 50), [1, 0, -1] * 50)
        payload = vtm_ref.encode_trit_segment(tensor)
        magic, version, flags, n_trits, n_scales = struct.unpack("<4sBBQI", payload[:18])
        self.assertEqual(magic, b"TRT1")
        self.assertEqual(version, 1)
        self.assertEqual(flags, 1)  # float32-масштабы
        self.assertEqual(n_trits, 150)
        self.assertEqual(n_scales, 2)  # ceil(150/128)
        # 18 + 2*4 масштаба + 8 слов (150 -> один 160-тритовый блок)
        self.assertEqual(len(payload), 18 + 2 * 4 + 8 * 4)
        self.assertEqual(len(payload), vtm_ref.tensor_payload_size(tensor))

    def test_segment_roundtrip_with_padding(self):
        trits = [1, -1, 0, 1] * 33  # 132 трита -> 2 блока масштаба, 1 блок слов
        scales = [0.5, 0.25]
        tensor = vtm_ref.Tensor.ternary("w", (132,), trits, scales)
        payload = vtm_ref.decode_trit_segment(vtm_ref.encode_trit_segment(tensor))
        self.assertEqual(payload[0], trits)
        self.assertEqual(payload[1], scales)
        self.assertEqual(payload[2], "float32")

    def test_segment_truncated(self):
        tensor = vtm_ref.Tensor.ternary("w", (4, 40), [1] * 160)
        payload = vtm_ref.encode_trit_segment(tensor)
        with self.assertRaises(ValueError):
            vtm_ref.decode_trit_segment(payload[:-1])

    def test_segment_corrupt_word(self):
        tensor = vtm_ref.Tensor.ternary("w", (1, 160), [1] * 160)
        payload = bytearray(vtm_ref.encode_trit_segment(tensor))
        # 160 тритов -> 2 блока масштаба по 4 байта, слова начинаются с 18+8
        payload[18 + 8: 18 + 12] = b"\xff\xff\xff\xff"  # слово вне 3^20
        with self.assertRaises(ValueError):
            vtm_ref.decode_trit_segment(bytes(payload))

    def test_segment_bad_magic(self):
        tensor = vtm_ref.Tensor.ternary("w", (4, 4), [1] * 16)
        payload = bytearray(vtm_ref.encode_trit_segment(tensor))
        payload[0:4] = b"XTX1"
        with self.assertRaises(ValueError):
            vtm_ref.decode_trit_segment(bytes(payload))


# ---------------------------------------------------------------------------
# Round-trip контейнера
# ---------------------------------------------------------------------------


class TestContainerRoundTrip(_TmpCase):
    def _model(self):
        trits = ([1, -1, 0, 1] * 40)[:150]
        tensors = [
            vtm_ref.Tensor.ternary("blk.0.w", (3, 50), trits, [0.5, 0.25]),
            vtm_ref.Tensor.f32("x", (4,), [1.5, -2.0, 0.25, 0.0]),
            vtm_ref.Tensor.fp16("h", (2, 3), [0.5, -1.25, 2.0, 0.0, 1.0, -0.5]),
            vtm_ref.Tensor.bf16("b", (2,), [1.0, -2.0]),
            vtm_ref.Tensor.f32("y", (3,), [0.0, 0.0, 0.0]),
        ]
        program = [
            vtm_ref.Instr(vtm_ref.Op.TMV, dst=4, a=0, b=1),
            vtm_ref.Instr(vtm_ref.Op.SILU, dst=4, a=4),
            vtm_ref.Instr(vtm_ref.Op.SCAL, dst=4, a=4, imm=500),
            vtm_ref.Instr(vtm_ref.Op.HALT, imm=-1),
        ]
        meta = {"note": "привет, VTM", "rope_theta": 10_000.0, "nested": {"a": [1, 2, 3]}}
        return tensors, program, meta

    def test_roundtrip_tensors_program_meta(self):
        tensors, program, meta = self._model()
        path = self.path("model.vtm")
        written = vtm_ref.write_vtm(path, tensors, program, meta)
        self.assertEqual(os.path.getsize(path), written.estimated_file_size())
        loaded = vtm_ref.read_vtm(path)

        self.assertEqual(len(loaded.tensors), len(tensors))
        for original, restored in zip(tensors, loaded.tensors):
            self.assertEqual(original.name, restored.name)
            self.assertEqual(original.kind, restored.kind)
            self.assertEqual(original.dims, restored.dims)
            self.assertEqual(original.dense(), restored.dense())
        # триты и масштабы — побитово
        self.assertEqual(loaded.tensors[0].trits, tensors[0].trits)
        self.assertEqual(loaded.tensors[0].scales, tensors[0].scales)
        self.assertEqual(loaded.tensors[2].data, tensors[2].data)  # fp16 точен
        self.assertEqual(loaded.program, program)
        self.assertEqual(loaded.meta, meta)

    def test_layout_offsets_and_header(self):
        tensors, program, meta = self._model()
        path = self.path("layout.vtm")
        vtm_ref.write_vtm(path, tensors, program, meta)
        with open(path, "rb") as handle:
            data = handle.read()

        magic, version, flags, n_instr, n_tensors, n_program = struct.unpack("<4sHHIII", data[:20])
        self.assertEqual(magic, b"VTM1")
        self.assertEqual(version, 1)
        self.assertIn(flags & vtm_ref.VFLAG_CAUSAL, (0, vtm_ref.VFLAG_CAUSAL))
        self.assertEqual(n_instr, len(program))
        self.assertEqual(n_tensors, len(tensors))
        self.assertEqual(n_program, len(program) * 16)

        program_offset, meta_offset, meta_len = struct.unpack("<III", data[20:32])
        self.assertEqual(program_offset, len(data) - len(program) * 16)
        # meta лежит между данными и программой, JSON читается
        self.assertEqual(meta_len, len(vtm_ref._encode_meta(meta)))
        self.assertEqual(program_offset, vtm_ref.align_up(meta_offset + meta_len))

        # data_start = align32(32 + 44*n), offset'ы тензоров от него и по 32.
        table_end = 32 + 44 * n_tensors
        data_start = vtm_ref.align_up(table_end)
        data_size = 0
        for tensor in tensors:
            data_size = vtm_ref.align_up(
                data_size + vtm_ref.tensor_payload_size(tensor)
            )
        self.assertEqual(meta_offset, data_start + data_size)
        loaded = vtm_ref.read_vtm(path)
        self.assertEqual(loaded.data_offset, data_start)
        self.assertEqual(loaded.data_offset % vtm_ref.ALIGNMENT, 0)
        for tensor in loaded.tensors:
            self.assertEqual(tensor.offset % vtm_ref.ALIGNMENT, 0)
            if tensor.kind == vtm_ref.KIND_TRITS:
                self.assertEqual(tensor.scale_offset, vtm_ref.NO_OFFSET)

    def test_no_meta(self):
        tensors = [vtm_ref.Tensor.f32("x", (2,), [1.0, 2.0])]
        program = [vtm_ref.Instr(vtm_ref.Op.HALT)]
        path = self.path("nometa.vtm")
        vtm_ref.write_vtm(path, tensors, program, {})
        loaded = vtm_ref.read_vtm(path)
        self.assertEqual(loaded.meta, {})
        self.assertEqual(loaded.meta_len, 0)
        # заголовок + запись таблицы + выравнивание данных + 8 байт fp32 + программа
        self.assertEqual(
            os.path.getsize(path), vtm_ref.align_up(32 + 44) + 32 + 16
        )

    def test_execute_after_roundtrip_is_identical(self):
        vtm = vtm_compile.make_synthetic(seed=3, size=128, mode="mlp")
        path = self.path("synth.vtm")
        vtm_ref.write_vtm(path, vtm.tensors, vtm.program, vtm.meta)
        before = vtm_ref.execute(vtm, [], 0)
        after = vtm_ref.execute(vtm_ref.read_vtm(path), [], 0)
        self.assertEqual(before.logits, after.logits)
        self.assertEqual(before.workspace, after.workspace)


class TestInstructionEncoding(unittest.TestCase):
    def test_layout_matches_c(self):
        # C читает инструкцию как op|flags|dst|a|b|c|pad(2)|imm,
        # то есть imm по смещению 12 (src/vtm/interp.c).
        instr = vtm_ref.Instr(
            vtm_ref.Op.RMSN, flags=1, dst=2, a=3, b=4, c=5, imm=-7
        )
        raw = instr.pack()
        self.assertEqual(len(raw), 16)
        self.assertEqual(raw[0], int(vtm_ref.Op.RMSN))
        self.assertEqual(raw[1], 1)
        self.assertEqual(struct.unpack_from("<H", raw, 2)[0], 2)
        self.assertEqual(struct.unpack_from("<H", raw, 4)[0], 3)
        self.assertEqual(struct.unpack_from("<H", raw, 6)[0], 4)
        self.assertEqual(struct.unpack_from("<H", raw, 8)[0], 5)
        self.assertEqual(raw[10:12], b"\x00\x00")
        self.assertEqual(struct.unpack_from("<i", raw, 12)[0], -7)
        self.assertEqual(vtm_ref.Instr.unpack_from(raw), instr)

    def test_rope_b_is_head_count(self):
        program = [vtm_ref.Instr(vtm_ref.Op.ROPE, dst=1, a=0, b=4, imm=0)]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                vtm_ref.write_vtm(
                    os.path.join(tmp, "rope.vtm"),
                    [vtm_ref.Tensor.f32("x", (4, 4), [0.0] * 16),
                     vtm_ref.Tensor.f32("y", (4, 4), [0.0] * 16)],
                    program,
                    {},
                )


# ---------------------------------------------------------------------------
# Битые файлы: read_vtm обязан бросать ValueError
# ---------------------------------------------------------------------------


class TestCorruptFiles(_TmpCase):
    def _good_file(self, name="good.vtm"):
        tensors = [
            vtm_ref.Tensor.ternary("w", (2, 8), [1, -1, 0, 1] * 4, [1.0]),
            vtm_ref.Tensor.f32("x", (8,), [float(i) for i in range(8)]),
            vtm_ref.Tensor.f32("y", (2,), [0.0, 0.0]),
        ]
        program = [
            vtm_ref.Instr(vtm_ref.Op.TMV, dst=2, a=0, b=1),
            vtm_ref.Instr(vtm_ref.Op.HALT),
        ]
        path = self.path(name)
        vtm_ref.write_vtm(path, tensors, program, {"k": "v"})
        with open(path, "rb") as handle:
            return path, handle.read()

    def _patched(self, mutate, name):
        _path, data = self._good_file()
        mutated = bytearray(data)
        mutate(mutated)
        path = self.path(name)
        with open(path, "wb") as handle:
            handle.write(bytes(mutated))
        return path

    def test_truncated_file(self):
        path, data = self._good_file()
        truncated = self.path("trunc.vtm")
        with open(truncated, "wb") as handle:
            handle.write(data[:-1])
        with self.assertRaises(ValueError):
            vtm_ref.read_vtm(truncated)

    def test_header_only(self):
        path = self.path("short.vtm")
        with open(path, "wb") as handle:
            handle.write(b"VTM1" + b"\x00" * 27)
        with self.assertRaises(ValueError):
            vtm_ref.read_vtm(path)

    def test_bad_magic(self):
        path = self._patched(lambda data: data.__setitem__(slice(0, 4), b"VTM2"), "magic.vtm")
        with self.assertRaises(ValueError):
            vtm_ref.read_vtm(path)

    def test_bad_version(self):
        path = self._patched(lambda data: data.__setitem__(slice(4, 6), b"\x09\x00"), "ver.vtm")
        with self.assertRaises(ValueError):
            vtm_ref.read_vtm(path)

    def test_bad_n_tensors(self):
        path = self._patched(
            lambda data: data.__setitem__(slice(12, 16), struct.pack("<I", 1000)), "nt.vtm"
        )
        with self.assertRaises(ValueError):
            vtm_ref.read_vtm(path)

    def test_bad_n_program(self):
        path = self._patched(
            lambda data: data.__setitem__(slice(16, 20), struct.pack("<I", 7)), "np.vtm"
        )
        with self.assertRaises(ValueError):
            vtm_ref.read_vtm(path)

    def test_offset_out_of_bounds(self):
        # offset — u32 по смещению 34 от начала записи таблицы.
        def mutate(data):
            data[32 + 34:32 + 38] = struct.pack("<I", 0xFFFF00)
        path = self._patched(mutate, "offset.vtm")
        with self.assertRaises(ValueError):
            vtm_ref.read_vtm(path)

    def test_unknown_opcode(self):
        def mutate(data):
            program_offset = struct.unpack("<I", data[20:24])[0]
            data[program_offset] = 0x7F
        path = self._patched(mutate, "opcode.vtm")
        with self.assertRaises(ValueError):
            vtm_ref.read_vtm(path)

    def test_index_out_of_table(self):
        def mutate(data):
            program_offset = struct.unpack("<I", data[20:24])[0]
            data[program_offset + 2:program_offset + 4] = struct.pack("<H", 0xFFFF)
        path = self._patched(mutate, "index.vtm")
        with self.assertRaises(ValueError):
            vtm_ref.read_vtm(path)

    def test_corrupt_trit_segment(self):
        def mutate(data):
            # данные первого тензора начинаются с data_start + offset
            n_tensors = struct.unpack("<I", data[12:16])[0]
            data_start = vtm_ref.align_up(32 + 44 * n_tensors)
            offset = struct.unpack("<I", data[32 + 34:32 + 38])[0]
            data[data_start + offset: data_start + offset + 4] = b"XXXX"
        path = self._patched(mutate, "seg.vtm")
        with self.assertRaises(ValueError):
            vtm_ref.read_vtm(path)

    def test_bad_meta_json(self):
        def mutate(data):
            meta_offset = struct.unpack("<I", data[24:28])[0]
            meta_len = struct.unpack("<I", data[28:32])[0]
            data[meta_offset:meta_offset + meta_len] = b"#" * meta_len
        path = self._patched(mutate, "meta.vtm")
        with self.assertRaises(ValueError):
            vtm_ref.read_vtm(path)

    def test_write_rejects_unbound_weights(self):
        tensors = [vtm_ref.Tensor.placeholder("w", (2, 2), vtm_ref.KIND_TRITS)]
        program = [vtm_ref.Instr(vtm_ref.Op.HALT)]
        with self.assertRaises(ValueError):
            vtm_ref.write_vtm(self.path("unbound.vtm"), tensors, program, {})

    def test_write_rejects_duplicate_names(self):
        tensors = [
            vtm_ref.Tensor.f32("x", (2,), [1.0, 2.0]),
            vtm_ref.Tensor.f32("x", (2,), [3.0, 4.0]),
        ]
        with self.assertRaises(ValueError):
            vtm_ref.write_vtm(
                self.path("dup.vtm"), tensors, [vtm_ref.Instr(vtm_ref.Op.HALT)], {}
            )


# ---------------------------------------------------------------------------
# Операции на числах, посчитанных на бумаге
# ---------------------------------------------------------------------------


class TestHandComputedOps(unittest.TestCase):
    def test_tmv_3x4(self):
        # W = [[1,0,-1,1],[0,1,0,-1],[-1,0,1,1]], x = [2,3,-1,4]
        trits = [1, 0, -1, 1, 0, 1, 0, -1, -1, 0, 1, 1]
        w = vtm_ref.Tensor.ternary("w", (3, 4), trits, [1.0])
        x = vtm_ref.Tensor.f32("x", (4,), [2.0, 3.0, -1.0, 4.0])
        y = vtm_ref.Tensor.f32("y", (3,), [0.0, 0.0, 0.0])
        program = [
            vtm_ref.Instr(vtm_ref.Op.TMV, dst=2, a=0, b=1),
            vtm_ref.Instr(vtm_ref.Op.HALT),
        ]
        self.assertEqual(_run([w, x, y], program).workspace["y"], [7.0, -1.0, 1.0])

    def test_tmv_scale_applies_to_partial_sum_not_weights(self):
        # Один блок 128 на строку: y = scale * (trit·x), а не (trit*scale)·x.
        w = vtm_ref.Tensor.ternary(
            "w", (2, 128), [1] * 256, scales=[0.5, 2.0]
        )
        x = vtm_ref.Tensor.f32("x", (128,), [1.0] * 128)
        y = vtm_ref.Tensor.f32("y", (2,), [0.0, 0.0])
        program = [
            vtm_ref.Instr(vtm_ref.Op.TMV, dst=2, a=0, b=1),
            vtm_ref.Instr(vtm_ref.Op.HALT),
        ]
        self.assertEqual(_run([w, x, y], program).workspace["y"], [64.0, 256.0])

    def test_tmv_block_crossing_rows(self):
        # 2 строки по 160 столбцов: блок 128 пересекает границу строк.
        # row0: 128*s0 + 32*s1; row1: 96*s1 + 64*s2 при всех тритах +1 и x=1.
        w = vtm_ref.Tensor.ternary(
            "w", (2, 160), [1] * 320, scales=[1.0, 2.0, 4.0]
        )
        x = vtm_ref.Tensor.f32("x", (160,), [1.0] * 160)
        y = vtm_ref.Tensor.f32("y", (2,), [0.0, 0.0])
        program = [
            vtm_ref.Instr(vtm_ref.Op.TMV, dst=2, a=0, b=1),
            vtm_ref.Instr(vtm_ref.Op.HALT),
        ]
        self.assertEqual(
            _run([w, x, y], program).workspace["y"], [192.0, 448.0]
        )

    def test_rmsnorm(self):
        # x=[3,4], mean(x^2)=12.5, rms=sqrt(12.5)
        x = vtm_ref.Tensor.f32("x", (2,), [3.0, 4.0])
        w = vtm_ref.Tensor.f32("w", (2,), [1.0, 1.0])
        y = vtm_ref.Tensor.f32("y", (2,), [0.0, 0.0])
        program = [
            vtm_ref.Instr(vtm_ref.Op.RMSN, dst=2, a=0, b=1, imm=0),
            vtm_ref.Instr(vtm_ref.Op.HALT),
        ]
        expected = [3.0 / math.sqrt(12.5), 4.0 / math.sqrt(12.5)]
        for got, want in zip(_run([x, w, y], program).workspace["y"], expected):
            self.assertAlmostEqual(got, want, places=6)

        # w=[1,2] домножает поэлементно
        w2 = vtm_ref.Tensor.f32("w", (2,), [1.0, 2.0])
        got = _run([x, w2, y], program).workspace["y"]
        self.assertAlmostEqual(got[0], 3.0 / math.sqrt(12.5), places=6)
        self.assertAlmostEqual(got[1], 8.0 / math.sqrt(12.5), places=6)

    def test_rmsnorm_eps_is_thousandths(self):
        # imm=1000 -> eps=1.0; x=[3,4]: sqrt(12.5+1)
        x = vtm_ref.Tensor.f32("x", (2,), [3.0, 4.0])
        w = vtm_ref.Tensor.f32("w", (2,), [1.0, 1.0])
        y = vtm_ref.Tensor.f32("y", (2,), [0.0, 0.0])
        program = [
            vtm_ref.Instr(vtm_ref.Op.RMSN, dst=2, a=0, b=1, imm=1000),
            vtm_ref.Instr(vtm_ref.Op.HALT),
        ]
        got = _run([x, w, y], program).workspace["y"]
        self.assertAlmostEqual(got[0], 3.0 / math.sqrt(13.5), places=6)
        self.assertAlmostEqual(got[1], 4.0 / math.sqrt(13.5), places=6)

    def test_rmsnorm_zero_vector_with_zero_eps(self):
        x = vtm_ref.Tensor.f32("x", (2,), [0.0, 0.0])
        w = vtm_ref.Tensor.f32("w", (2,), [1.0, 1.0])
        y = vtm_ref.Tensor.f32("y", (2,), [7.0, 7.0])
        program = [
            vtm_ref.Instr(vtm_ref.Op.RMSN, dst=2, a=0, b=1, imm=0),
            vtm_ref.Instr(vtm_ref.Op.HALT),
        ]
        self.assertEqual(_run([x, w, y], program).workspace["y"], [0.0, 0.0])

    def test_softmax_vector(self):
        # softmax([0,1,2]) = [1, e, e^2] / (1+e+e^2)
        x = vtm_ref.Tensor.f32("x", (3,), [0.0, 1.0, 2.0])
        y = vtm_ref.Tensor.f32("y", (3,), [0.0, 0.0, 0.0])
        program = [
            vtm_ref.Instr(vtm_ref.Op.SM, dst=1, a=0),
            vtm_ref.Instr(vtm_ref.Op.HALT),
        ]
        total = 1.0 + math.e + math.e ** 2
        expected = [1.0 / total, math.e / total, math.e ** 2 / total]
        for got, want in zip(_run([x, y], program).workspace["y"], expected):
            self.assertAlmostEqual(got, want, places=6)

    def test_softmax_causal_mask(self):
        # Нулевые логиты, CAUSAL: каждая строка видит только себя и прошлое.
        x = vtm_ref.Tensor.f32("x", (3, 3), [0.0] * 9)
        y = vtm_ref.Tensor.f32("y", (3, 3), [0.0] * 9)
        program = [
            vtm_ref.Instr(vtm_ref.Op.SM, dst=1, a=0, flags=vtm_ref.IFLAG_CAUSAL),
            vtm_ref.Instr(vtm_ref.Op.HALT),
        ]
        got = _run([x, y], program).workspace["y"]
        for value, want in zip(
            got, [1.0, 0.0, 0.0, 0.5, 0.5, 0.0, 1 / 3, 1 / 3, 1 / 3]
        ):
            self.assertAlmostEqual(value, want, places=6)

    def test_softmax_causal_nonuniform(self):
        # x = [[0,5,5],[0,0,5],[0,0,0]] -> [[1,0,0],[.5,.5,0],[1/3,1/3,1/3]]
        x = vtm_ref.Tensor.f32("x", (3, 3), [0, 5, 5, 0, 0, 5, 0, 0, 0])
        y = vtm_ref.Tensor.f32("y", (3, 3), [0.0] * 9)
        program = [
            vtm_ref.Instr(vtm_ref.Op.SM, dst=1, a=0, flags=vtm_ref.IFLAG_CAUSAL),
            vtm_ref.Instr(vtm_ref.Op.HALT),
        ]
        got = _run([x, y], program).workspace["y"]
        expected = [1.0, 0.0, 0.0, 0.5, 0.5, 0.0, 1 / 3, 1 / 3, 1 / 3]
        for value, want in zip(got, expected):
            self.assertAlmostEqual(value, want, places=6)

    def test_rope_half_layout(self):
        # d=4, pos=1, theta=10000: inv_freq = [1, 0.01]; x=[1,0,0,0]
        # пары half: (0,2) -> [cos1, 0, sin1, 0]
        x = vtm_ref.Tensor.f32("x", (1, 4), [1.0, 0.0, 0.0, 0.0])
        y = vtm_ref.Tensor.f32("y", (1, 4), [0.0] * 4)
        program = [
            vtm_ref.Instr(vtm_ref.Op.ROPE, dst=1, a=0, b=1, imm=1),
            vtm_ref.Instr(vtm_ref.Op.HALT),
        ]
        got = _run([x, y], program).workspace["y"]
        expected = [math.cos(1.0), 0.0, math.sin(1.0), 0.0]
        for value, want in zip(got, expected):
            self.assertAlmostEqual(value, want, places=6)

    def test_rope_interleaved_layout(self):
        # pairs (2k, 2k+1): x=[1,0,0,0] -> [cos1, sin1, 0, 0]
        x = vtm_ref.Tensor.f32("x", (1, 4), [1.0, 0.0, 0.0, 0.0])
        y = vtm_ref.Tensor.f32("y", (1, 4), [0.0] * 4)
        expected = [math.cos(1.0), math.sin(1.0), 0.0, 0.0]
        # флагом (как компилятор и C)
        program = [
            vtm_ref.Instr(
                vtm_ref.Op.ROPE, dst=1, a=0,
                flags=vtm_ref.IFLAG_ROPE_INTERLEAVED, b=1, imm=1,
            ),
            vtm_ref.Instr(vtm_ref.Op.HALT),
        ]
        got = _run([x, y], program).workspace["y"]
        for value, want in zip(got, expected):
            self.assertAlmostEqual(value, want, places=6)
        # meta — запасной путь для некомпилированных программ
        program = [
            vtm_ref.Instr(vtm_ref.Op.ROPE, dst=1, a=0, b=1, imm=1),
            vtm_ref.Instr(vtm_ref.Op.HALT),
        ]
        got = _run([x, y], program, meta={"rope_layout": "interleaved"}).workspace["y"]
        for value, want in zip(got, expected):
            self.assertAlmostEqual(value, want, places=6)

    def test_rope_theta_from_meta(self):
        # theta=100: inv_freq[0] = 1, angle = pos; theta влияет только на k>=1.
        x = vtm_ref.Tensor.f32("x", (1, 4), [0.0, 1.0, 0.0, 0.0])
        y = vtm_ref.Tensor.f32("y", (1, 4), [0.0] * 4)
        program = [
            vtm_ref.Instr(vtm_ref.Op.ROPE, dst=1, a=0, b=1, imm=1),
            vtm_ref.Instr(vtm_ref.Op.HALT),
        ]
        got = _run([x, y], program, meta={"rope_theta": 100.0}).workspace["y"]
        angle = 1.0 * 100.0 ** (-2.0 / 4.0)  # inv_freq[1], пара (1,3)
        expected = [0.0, math.cos(angle), 0.0, math.sin(angle)]
        for value, want in zip(got, expected):
            self.assertAlmostEqual(value, want, places=6)

    def test_silu(self):
        x = vtm_ref.Tensor.f32("x", (2,), [1.0, -1.0])
        y = vtm_ref.Tensor.f32("y", (2,), [0.0, 0.0])
        program = [
            vtm_ref.Instr(vtm_ref.Op.SILU, dst=1, a=0),
            vtm_ref.Instr(vtm_ref.Op.HALT),
        ]
        got = _run([x, y], program).workspace["y"]
        self.assertAlmostEqual(got[0], 1.0 / (1.0 + math.exp(-1.0)), places=6)
        self.assertAlmostEqual(got[1], -1.0 / (1.0 + math.exp(1.0)), places=6)

    def test_add_mul_scal(self):
        a = vtm_ref.Tensor.f32("a", (2,), [1.0, 2.0])
        b = vtm_ref.Tensor.f32("b", (2,), [3.0, 4.0])
        out = vtm_ref.Tensor.f32("out", (2,), [0.0, 0.0])
        program = [
            vtm_ref.Instr(vtm_ref.Op.ADD, dst=2, a=0, b=1),
            vtm_ref.Instr(vtm_ref.Op.SCAL, dst=2, a=2, imm=500),
            vtm_ref.Instr(vtm_ref.Op.HALT),
        ]
        self.assertEqual(_run([a, b, out], program).workspace["out"], [2.0, 3.0])

        program = [
            vtm_ref.Instr(vtm_ref.Op.MUL, dst=2, a=0, b=1),
            vtm_ref.Instr(vtm_ref.Op.HALT),
        ]
        self.assertEqual(_run([a, b, out], program).workspace["out"], [3.0, 8.0])

    def test_emb_static_and_host_token(self):
        emb = vtm_ref.Tensor.f32("emb", (3, 2), [10.0, 11.0, 20.0, 21.0, 30.0, 31.0])
        dst = vtm_ref.Tensor.f32("dst", (2,), [0.0, 0.0])
        # imm — статический id
        program = [
            vtm_ref.Instr(vtm_ref.Op.EMB, dst=1, a=0, imm=2),
            vtm_ref.Instr(vtm_ref.Op.HALT),
        ]
        self.assertEqual(_run([emb, dst], program).workspace["dst"], [30.0, 31.0])
        # флаг EMB_INPUT — токен прохода (как vtm_set_token в C)
        program = [
            vtm_ref.Instr(
                vtm_ref.Op.EMB, dst=1, a=0, flags=vtm_ref.IFLAG_EMB_INPUT, imm=0
            ),
            vtm_ref.Instr(vtm_ref.Op.HALT),
        ]
        self.assertEqual(_run([emb, dst], program, tokens=[1]).workspace["dst"],
                         [20.0, 21.0])

    def test_attention_single_query_with_cache(self):
        # ctx=3, 1 голова, head_dim=2. K/V — строки E по токену прохода.
        # E=[[0,0],[1,0],[0,1]], tokens=[1,2] -> k0=v0=[1,0], k1=v1=[0,1].
        # q=[1,0]; на позиции 1 scores=[1,0]/sqrt(2), softmax [p, 1-p],
        # out = p*[1,0] + (1-p)*[0,1].
        e = vtm_ref.Tensor.f32("e", (3, 2), [0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
        q = vtm_ref.Tensor.f32("q", (1, 1, 2), [1.0, 0.0])
        k = vtm_ref.Tensor.f32("k", (1, 1, 2), [0.0, 0.0])
        v = vtm_ref.Tensor.f32("v", (1, 1, 2), [0.0, 0.0])
        k_cache = vtm_ref.Tensor.f32("kc", (3, 2), [0.0] * 6)
        v_cache = vtm_ref.Tensor.f32("vc", (3, 2), [0.0] * 6)
        out = vtm_ref.Tensor.f32("out", (1, 1, 2), [0.0, 0.0])
        program = [
            vtm_ref.Instr(vtm_ref.Op.EMB, dst=2, a=0,
                          flags=vtm_ref.IFLAG_EMB_INPUT),
            vtm_ref.Instr(vtm_ref.Op.EMB, dst=3, a=0,
                          flags=vtm_ref.IFLAG_EMB_INPUT),
            vtm_ref.Instr(vtm_ref.Op.KVW, dst=4, a=2, b=5, c=3, imm=0),
            vtm_ref.Instr(vtm_ref.Op.ATTN, dst=6, a=1, b=4, c=5,
                          flags=vtm_ref.IFLAG_CAUSAL, imm=0),
            vtm_ref.Instr(vtm_ref.Op.HALT),
        ]
        got = _run([e, q, k, v, k_cache, v_cache, out], program,
                   tokens=[1, 2]).workspace["out"]
        score = 1.0 / math.sqrt(2.0)
        p0 = math.exp(score) / (math.exp(score) + 1.0)
        self.assertAlmostEqual(got[0], p0, places=6)
        self.assertAlmostEqual(got[1], 1.0 - p0, places=6)

    def test_cpy_copies_whole_tensor(self):
        source = vtm_ref.Tensor.f32("src", (1, 3), [1.0, 2.0, 3.0])
        dst = vtm_ref.Tensor.f32("dst", (3,), [0.0, 0.0, 0.0])
        program = [
            vtm_ref.Instr(vtm_ref.Op.CPY, dst=1, a=0),
            vtm_ref.Instr(vtm_ref.Op.HALT),
        ]
        self.assertEqual(
            _run([source, dst], program).workspace["dst"], [1.0, 2.0, 3.0]
        )


# ---------------------------------------------------------------------------
# Синтетика: детерминизм и зафиксированные эталоны
# ---------------------------------------------------------------------------


class TestSyntheticModel(unittest.TestCase):
    #: Зафиксировано `python3 tools/vtm_compile.py --synthetic --mode mlp`
    MLP_EXPECTED = {
        "sum": 154.98280582018197,
        "argmax": 66,
        "first8": [
            -2.248037338256836, 0.24765707552433014, 7.291206359863281,
            1.7859314680099487, 6.423129081726074, 9.081073760986328,
            -0.5535104870796204, -13.504440307617188,
        ],
    }
    #: Зафиксировано `python3 tools/vtm_compile.py --synthetic`
    ATTN_EXPECTED = {
        "sum": -201.42793148756027,
        "argmax": 48,
        "first8": [
            -13.945331573486328, 9.275696754455566, 4.834317684173584,
            0.9197925329208374, 1.9832044839859009, -5.496201515197754,
            2.3927385807037354, -0.09410431981086731,
        ],
    }

    @classmethod
    def setUpClass(cls):
        cls.mlp = vtm_compile.make_synthetic(seed=1, size=128, mode="mlp")
        cls.attn = vtm_compile.make_synthetic(seed=1, size=128, mode="attn")

    def _check(self, vtm, tokens, expected):
        result = vtm_ref.execute(vtm, tokens, 0)
        self.assertIsNotNone(result.logits)
        self.assertEqual(len(result.logits), 128)
        self.assertAlmostEqual(sum(result.logits), expected["sum"], places=6)
        self.assertEqual(vtm_ref._argmax(result.logits), expected["argmax"])
        for got, want in zip(result.logits[:8], expected["first8"]):
            self.assertAlmostEqual(got, want, places=6)
        return result

    def test_mlp_reference_numbers(self):
        self.assertEqual(self.mlp.meta["synthetic"]["mode"], "mlp")
        self.assertEqual(self.mlp.meta["synthetic"]["tokens"], [])
        self._check(self.mlp, [], self.MLP_EXPECTED)

    def test_attn_reference_numbers(self):
        self.assertEqual(self.attn.meta["synthetic"]["mode"], "attn")
        self.assertEqual(self.attn.meta["synthetic"]["tokens"], [0, 1, 2, 3])
        result = self._check(self.attn, [0, 1, 2, 3], self.ATTN_EXPECTED)
        self.assertEqual(result.steps, 4)

    def test_meta_expected_matches_execute(self):
        for vtm, tokens in ((self.mlp, []), (self.attn, [0, 1, 2, 3])):
            result = vtm_ref.execute(vtm, tokens, 0)
            expected = vtm.meta["expected"]
            self.assertEqual(result.logits, expected["logits"])
            self.assertEqual(sum(result.logits), expected["sum"])
            self.assertEqual(vtm_ref._argmax(result.logits), expected["argmax"])

    def test_execute_is_deterministic(self):
        for vtm, tokens in ((self.mlp, []), (self.attn, [0, 1, 2, 3])):
            first = vtm_ref.execute(vtm, tokens, 0)
            second = vtm_ref.execute(vtm, tokens, 0)
            self.assertEqual(first.logits, second.logits)
            self.assertEqual(first.workspace, second.workspace)

    def test_kv_cache_persists_between_steps(self):
        one_step = vtm_ref.execute(self.attn, [0], 0)
        four_steps = vtm_ref.execute(self.attn, [0, 1, 2, 3], 0)
        self.assertNotEqual(one_step.logits, four_steps.logits)
        k_cache = four_steps.get("l0.kcache")
        v_cache = four_steps.get("l0.vcache")
        self.assertIsNotNone(k_cache)
        self.assertIsNotNone(v_cache)
        self.assertTrue(any(value != 0.0 for value in k_cache))
        self.assertTrue(any(value != 0.0 for value in v_cache))
        # Кэши — единственные тензоры, которые не сбрасываются между
        # проходами; позиций записано ровно 4 (последние остальные нули).
        self.assertEqual(len(k_cache), 8 * 32)
        self.assertLess(
            sum(1 for value in k_cache if value == 0.0), len(k_cache)
        )

    def test_synthetic_is_reproducible_and_seed_changes_it(self):
        again = vtm_compile.make_synthetic(seed=1, size=128, mode="mlp")
        other = vtm_compile.make_synthetic(seed=2, size=128, mode="mlp")
        self.assertEqual(self.mlp.meta["expected"], again.meta["expected"])
        self.assertNotEqual(self.mlp.meta["expected"], other.meta["expected"])


# ---------------------------------------------------------------------------
# Компилятор графа
# ---------------------------------------------------------------------------


def _tiny_config(lm_head="mm"):
    return vtm_compile.ModelConfig(
        hidden=32,
        n_heads=2,
        n_kv_heads=1,
        head_dim=16,
        intermediate=64,
        vocab=32,
        n_layers=1,
        context_len=4,
        eps=1e-3,
        lm_head=lm_head,
    )


def _tiny_weights(cfg, seed=7):
    rng = random.Random(seed)

    def floats(n):
        return [rng.uniform(-1.0, 1.0) for _ in range(n)]

    def matrix(rows, cols):
        return floats(rows * cols)

    layer = {
        "norm1": [rng.uniform(0.5, 1.5) for _ in range(cfg.hidden)],
        "norm2": [rng.uniform(0.5, 1.5) for _ in range(cfg.hidden)],
        "q_proj": matrix(cfg.q_dim, cfg.hidden),
        "k_proj": matrix(cfg.kv_dim, cfg.hidden),
        "v_proj": matrix(cfg.kv_dim, cfg.hidden),
        "o_proj": matrix(cfg.hidden, cfg.q_dim),
        "gate_proj": matrix(cfg.intermediate, cfg.hidden),
        "up_proj": matrix(cfg.intermediate, cfg.hidden),
        "down_proj": matrix(cfg.hidden, cfg.intermediate),
    }
    return {
        "embed": floats(cfg.vocab * cfg.hidden),
        "final_norm": [rng.uniform(0.5, 1.5) for _ in range(cfg.hidden)],
        "lm_head": floats(cfg.vocab * cfg.hidden),
        "layers": [layer],
    }


class TestCompileLayer(unittest.TestCase):
    EXPECTED_OPS = [
        "RMSN",
        "TMV", "TMV", "TMV",
        "ROPE", "ROPE",
        "KVW", "ATTN",
        "TMV", "ADD",
        "RMSN",
        "TMV", "TMV", "SILU", "MUL",
        "TMV", "ADD",
    ]

    def test_op_sequence(self):
        cfg = _tiny_config()
        graph, x = vtm_compile.compile_layer(cfg)
        names = [vtm_ref.OP_NAMES[int(instr.op)] for instr in graph.program]
        self.assertEqual(names, self.EXPECTED_OPS)
        self.assertEqual(x, graph._names["l0.x"])
        for name in ("l0.h1", "l0.q", "l0.k", "l0.v", "l0.kcache", "l0.vcache",
                     "l0.attn", "l0.o", "l0.h2", "l0.gate", "l0.up", "l0.mul",
                     "l0.down"):
            self.assertIn(name, graph._names)
        # ROPE: b — число голов, imm = 0 (позицию задаёт хост), флаг — раскладка
        rope = [i for i in graph.program if int(i.op) == vtm_ref.Op.ROPE]
        self.assertEqual([i.b for i in rope], [cfg.n_heads, cfg.n_kv_heads])
        self.assertEqual([i.imm for i in rope], [0, 0])
        self.assertEqual([i.flags for i in rope], [0, 0])
        # Q/K/V — 3-D (heads, 1, head_dim), как ждёт ATTN
        self.assertEqual(graph.tensors[graph._names["l0.q"]].dims,
                         (cfg.n_heads, 1, cfg.head_dim))

    def test_tmv_shapes_consistent(self):
        cfg = _tiny_config()
        graph, _x = vtm_compile.compile_layer(cfg, weights=_tiny_weights(cfg)["layers"][0])
        for instr in graph.program:
            if int(instr.op) != vtm_ref.Op.TMV:
                continue
            weight = graph.tensors[instr.a]
            source = graph.tensors[instr.b]
            target = graph.tensors[instr.dst]
            self.assertEqual(weight.kind, vtm_ref.KIND_TRITS)
            self.assertEqual(len(weight.dims), 2)
            self.assertEqual(weight.dims[0], target.n_elements)
            self.assertEqual(weight.dims[1], source.n_elements)

    def test_layer_shares_residual_tensor(self):
        cfg = _tiny_config()
        graph = vtm_compile.Graph(cfg)
        x = graph.act("x", (cfg.hidden,))
        vtm_compile.compile_layer(cfg, graph=graph, prefix="a", x_index=x)
        vtm_compile.compile_layer(cfg, graph=graph, prefix="b", x_index=x)
        self.assertEqual(graph._names["x"], x)
        names = [vtm_ref.OP_NAMES[int(instruction.op)] for instruction in graph.program]
        self.assertEqual(names.count("ADD"), 4)  # по два остатка на слой


class TestCompileModel(unittest.TestCase):
    def test_forward_and_generation(self):
        cfg = _tiny_config(lm_head="mm")
        model = vtm_compile.compile_model(cfg, _tiny_weights(cfg))
        result = vtm_ref.execute(model, [1, 2], 2)
        self.assertEqual(result.prompt, [1, 2])
        self.assertEqual(len(result.generated), 2)
        self.assertEqual(result.steps, 4)
        self.assertTrue(all(0 <= token < cfg.vocab for token in result.generated))
        self.assertEqual(result.tokens, [1, 2] + result.generated)
        self.assertIsNotNone(result.logits)
        self.assertEqual(len(result.logits), cfg.vocab)
        # детерминизм
        again = vtm_ref.execute(model, [1, 2], 2)
        self.assertEqual(result.generated, again.generated)
        self.assertEqual(result.logits, again.logits)

    def test_lm_head_tmv_matches_mm_shape(self):
        for mode in ("tmv", "mm"):
            with self.subTest(mode=mode):
                cfg = _tiny_config(lm_head=mode)
                model = vtm_compile.compile_model(cfg, _tiny_weights(cfg))
                result = vtm_ref.execute(model, [3], 1)
                self.assertEqual(len(result.logits), cfg.vocab)
                self.assertEqual(len(result.generated), 1)
                head = model.by_name("lm_head.w")
                expected_kind = (
                    vtm_ref.KIND_TRITS if mode == "tmv" else vtm_ref.KIND_F32
                )
                self.assertEqual(head.kind, expected_kind)

    def test_skeleton_cannot_be_written(self):
        cfg = _tiny_config()
        model = vtm_compile.compile_model(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                vtm_ref.write_vtm(os.path.join(tmp, "skeleton.vtm"),
                                  model.tensors, model.program, model.meta)
        self.assertGreater(model.estimated_file_size(), 0)

    def test_meta_has_rope_settings(self):
        cfg = _tiny_config()
        model = vtm_compile.compile_model(cfg)
        self.assertEqual(model.meta["rope_theta"], cfg.rope_theta)
        self.assertEqual(model.meta["rope_layout"], cfg.rope_layout)
        self.assertEqual(model.meta["n_layers"], cfg.n_layers)


class TestCompilerCli(unittest.TestCase):
    def test_cli_synthetic(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = os.path.join(tmp, "cli.vtm")
            proc = subprocess.run(
                [sys.executable, os.path.join(TOOLS, "vtm_compile.py"),
                 "--synthetic", "--out", output],
                cwd=ROOT, capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("round-trip", proc.stdout)
            self.assertIn("max|Δ логитов| = 0.000e+00", proc.stdout)
            self.assertTrue(os.path.exists(output))

    def test_cli_synthetic_mlp(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = os.path.join(tmp, "cli_mlp.vtm")
            proc = subprocess.run(
                [sys.executable, os.path.join(TOOLS, "vtm_compile.py"),
                 "--synthetic", "--mode", "mlp", "--size", "128", "--seed", "1",
                 "--out", output],
                cwd=ROOT, capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("mode=mlp", proc.stdout)
            self.assertTrue(os.path.exists(output))

    def test_cli_qwen_plan(self):
        proc = subprocess.run(
            [sys.executable, os.path.join(TOOLS, "vtm_compile.py"),
             "--qwen-config", QWEN_CONFIG],
            cwd=ROOT, capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Qwen/Qwen2.5-1.5B-Instruct", proc.stdout)
        self.assertIn("файл не записывается", proc.stdout)
        self.assertIn("тензоров:", proc.stdout)


if __name__ == "__main__":
    unittest.main()
