"""kind 5 (Q4) в `.vtm`: кодек, контейнер и загрузчик C (задача 0151).

Проверяются три независимых слоя:

1. упаковка нибблов и сегмент `Q4_0`/`Q4_1` в `tools/vtm_ref.py`
   (2 значения в байт, группа 32, байт j = значения j и j+16);
2. контейнер: Q4-тензор пишется kind 5, читается без fp32-материализации,
   `dense()` даёт `(q-8)*scale` / `q*scale+min`;
3. C-загрузчик (`src/vtm/loader.c`) на том же файле: kind 5, `n_scales`,
   `scale_block=32`, упакованные байты в `words`, масштабы в `scales`,
   Q4_1 — в `q4_mins`, `f32 == NULL`. C-драйвер собирается поверх
   `libvtm.a`; без компилятора тест скипается.

Запуск:
    python3 -m unittest tests.test_vtm_q4 -v
"""

from __future__ import annotations

import math
import os
import random
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools import vtm_ref  # noqa: E402

SRC_VTM = os.path.join(ROOT, "src", "vtm")
VTM_LIB = os.path.join(SRC_VTM, "libvtm.a")


def _q4values(n: int, seed: int = 7) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(0, 16) for _ in range(n)]


def _scales(n_values: int, dtype: str = "float32") -> list[float]:
    n_groups = vtm_ref.n_scale_blocks(n_values, vtm_ref.Q4_SCALE_BLOCK)
    return [0.5 + 0.25 * i for i in range(n_groups)]


class TestQ4Codec(unittest.TestCase):
    """Упаковка, сегмент, валидация — без файлов и C."""

    def test_pack_group_layout_and_tail(self):
        values = [i % 16 for i in range(37)]
        packed = vtm_ref.pack_q4(values)
        self.assertEqual(len(packed), 2 * vtm_ref.Q4_GROUP_BYTES)
        # байт j группы: значение j (low) + значение j+16 (high).
        self.assertEqual(packed[0], values[0] | (values[16] << 4))
        self.assertEqual(packed[1], values[1] | (values[17] << 4))
        self.assertEqual(packed[15], values[15] | (values[31] << 4))
        # вторая группа: 5 значений, остальное — padding-ниббл 8.
        self.assertEqual(packed[16], values[32] | (vtm_ref.Q4_ZERO_NIBBLE << 4))
        self.assertEqual(packed[16 + 4], values[36] | (vtm_ref.Q4_ZERO_NIBBLE << 4))
        self.assertEqual(packed[16 + 5], 0x88)
        self.assertEqual(vtm_ref.unpack_q4(packed, 37), values)
        self.assertEqual(vtm_ref.unpack_q4(packed, 32), values[:32])

    def test_pack_rejects_out_of_range(self):
        for bad in (-1, 16, 255):
            with self.subTest(value=bad):
                with self.assertRaisesRegex(ValueError, "вне 0..15"):
                    vtm_ref.pack_q4([0, bad])
        with self.assertRaisesRegex(ValueError, "padding"):
            vtm_ref.pack_q4([0], pad=16)

    def test_unpack_truncated(self):
        with self.assertRaisesRegex(ValueError, "обрезаны"):
            vtm_ref.unpack_q4(b"\x00" * 15, 32)

    def test_segment_symmetric_roundtrip(self):
        n = 37
        q = _q4values(n)
        scales = _scales(n)
        tensor = vtm_ref.Tensor.q4("w", (n,), q, scales)
        segment = vtm_ref.encode_q4_segment(tensor)
        self.assertEqual(segment[:4], vtm_ref.Q4_MAGIC)
        self.assertEqual(segment[4], vtm_ref.Q4_VERSION)
        self.assertEqual(segment[5], vtm_ref.Q4_FLAG_F32_SCALES)
        self.assertEqual(len(segment), vtm_ref.tensor_payload_size(tensor))
        got_q, got_scales, dtype = vtm_ref.decode_q4_segment(segment)
        self.assertEqual(got_q, q)
        self.assertEqual(got_scales, scales)
        self.assertEqual(dtype, "float32")
        got_q, got_scales, dtype, asym, mins = vtm_ref.decode_q4_segment_ex(
            segment
        )
        self.assertEqual(got_q, q)
        self.assertEqual(got_scales, scales)
        self.assertFalse(asym)
        self.assertIsNone(mins)
        self.assertEqual(tensor.dense(), [
            (value - vtm_ref.Q4_ZERO_NIBBLE) * scales[i // 32]
            for i, value in enumerate(q)
        ])

    def test_segment_asymmetric_roundtrip_fp16(self):
        n = 64
        q = _q4values(n, seed=11)
        scales = _scales(n)
        mins = [-1.5, 2.25]
        tensor = vtm_ref.Tensor.q4(
            "w", (n,), q, scales, scale_dtype="float16",
            asymmetric=True, mins=mins,
        )
        segment = vtm_ref.encode_q4_segment(tensor)
        self.assertEqual(segment[:4], vtm_ref.Q4_MAGIC_ASYM)
        self.assertEqual(segment[4], vtm_ref.Q4_VERSION)
        self.assertEqual(segment[5], 0)  # fp16-масштабы
        # Q4_1 несёт явное n_mins после n_groups.
        import struct
        n_mins = struct.unpack_from("<I", segment, vtm_ref.Q4_HEADER_SIZE)[0]
        self.assertEqual(n_mins, 2)
        self.assertEqual(len(segment), vtm_ref.tensor_payload_size(tensor))
        got_q, got_scales, dtype, asym, got_mins = (
            vtm_ref.decode_q4_segment_ex(segment)
        )
        self.assertEqual(got_q, q)
        self.assertEqual(got_scales, scales)
        self.assertEqual(dtype, "float16")
        self.assertTrue(asym)
        self.assertEqual(got_mins, mins)
        dense = tensor.dense()
        self.assertEqual(dense[0], (q[0] * scales[0] + mins[0]))
        self.assertEqual(dense[32], (q[32] * scales[1] + mins[1]))

    def test_segment_validation(self):
        n = 64
        q = _q4values(n)
        scales = _scales(n)
        tensor = vtm_ref.Tensor.q4("w", (n,), q, scales)
        segment = vtm_ref.encode_q4_segment(tensor)

        def broken(mutator, regex):
            data = bytearray(segment)
            mutator(data)
            with self.assertRaisesRegex(ValueError, regex):
                vtm_ref.decode_q4_segment(bytes(data))

        broken(lambda d: d.__setitem__(slice(0, 4), b"Q4_9"), "magic")
        broken(lambda d: d.__setitem__(4, 2), "версия")
        broken(lambda d: d.__setitem__(5, 0x02), "флаги")
        broken(lambda d: d.__setitem__(slice(14, 18), b"\x03\x00\x00\x00"),
               "групп")
        # Отрицательный/NaN масштаб.
        import struct
        bad = bytearray(segment)
        bad[18:22] = struct.pack("<f", -1.0)
        with self.assertRaisesRegex(ValueError, "конечным и неотрицательным"):
            vtm_ref.decode_q4_segment(bytes(bad))
        # Обрезанный сегмент.
        with self.assertRaisesRegex(ValueError, "обрезан"):
            vtm_ref.decode_q4_segment(segment[:-1])
        # Q4_1: n_mins — явное число min, обязано совпасть с числом групп.
        asym = bytearray(vtm_ref.encode_q4_segment(
            vtm_ref.Tensor.q4(
                "w", (n,), q, scales, asymmetric=True, mins=[-1.0, 2.0]
            )
        ))
        asym[vtm_ref.Q4_HEADER_SIZE:vtm_ref.Q4_ASYM_HEADER_SIZE] = (
            struct.pack("<I", 3)
        )
        with self.assertRaisesRegex(ValueError, "n_mins=3"):
            vtm_ref.decode_q4_segment_ex(bytes(asym))

    def test_tensor_validation(self):
        with self.assertRaisesRegex(ValueError, "вне 0..15"):
            vtm_ref.Tensor.q4("w", (1,), [16], [0.5])
        with self.assertRaisesRegex(ValueError, "масштабов"):
            vtm_ref.Tensor.q4("w", (33,), _q4values(33), [0.5])
        with self.assertRaisesRegex(ValueError, "отрицательным"):
            vtm_ref.Tensor.q4("w", (32,), _q4values(32), [-0.5])
        with self.assertRaisesRegex(ValueError, "неконечное"):
            vtm_ref.Tensor.q4("w", (32,), _q4values(32), [math.nan])
        with self.assertRaisesRegex(ValueError, "Q4_1 без min"):
            vtm_ref.Tensor.q4(
                "w", (32,), _q4values(32), [0.5], asymmetric=True
            )
        with self.assertRaisesRegex(ValueError, "Q4_0 не имеет min"):
            vtm_ref.Tensor.q4(
                "w", (32,), _q4values(32), [0.5], mins=[0.0]
            )

    def test_quantize_symmetric_and_asymmetric(self):
        values = [-1.0, -0.5, 0.0, 0.5, 1.0] * 8
        q, scales = vtm_ref.quantize_q4(values)
        self.assertEqual(len(q), len(values))
        self.assertTrue(all(0 <= value <= 15 for value in q))
        self.assertAlmostEqual(max(abs(v) for v in scales), 1.0 / 7.0, places=6)
        _q, _scales, mins = vtm_ref.quantize_q4(values, symmetric=False)
        self.assertEqual(len(mins), 2)
        self.assertAlmostEqual(mins[0], -1.0, places=6)
        self.assertAlmostEqual(mins[1], -1.0, places=6)
        # Круговой рейс квантования: нулевой блок не ломается.
        zero_q, zero_scales = vtm_ref.quantize_q4([0.0] * 32)
        self.assertEqual(zero_scales, [0.0])
        self.assertEqual(set(zero_q), {vtm_ref.Q4_ZERO_NIBBLE})


class _TmpCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def path(self, *parts):
        path = os.path.join(self._tmp.name, *parts)
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        return path


class TestQ4Container(_TmpCase):
    """Контейнер `.vtm`: kind 5 остаётся kind 5, dense() считает Q4."""

    def _container(self, *, asymmetric=False, scale_dtype="float32", n=37):
        q = _q4values(n)
        scales = _scales(n)
        mins = [-1.0, 2.0, -0.25][:len(scales)] if asymmetric else None
        tensor = vtm_ref.Tensor.q4(
            "w", (n,), q, scales, scale_dtype=scale_dtype,
            asymmetric=asymmetric, mins=mins,
        )
        x = vtm_ref.Tensor.f32("x", (1,), [0.0])
        program = [vtm_ref.Instr(op=int(vtm_ref.Op.HALT))]
        path = self.path("q4.vtm")
        model = vtm_ref.write_vtm(path, [tensor, x], program,
                                  {"format": "q4", "q4_1": asymmetric})
        return path, model, tensor, q, scales, mins

    def test_roundtrip_symmetric(self):
        path, model, tensor, q, scales, _mins = self._container()
        self.assertEqual(os.path.getsize(path), model.estimated_file_size())
        back = vtm_ref.read_vtm(path)
        got = back.by_name("w")
        self.assertEqual(got.kind, vtm_ref.KIND_Q4)
        self.assertEqual(got.kind_name, "q4")
        self.assertEqual(got.qdata, q)
        self.assertEqual(got.scales, scales)
        self.assertEqual(got.scale_dtype, "float32")
        self.assertEqual(got.scale_offset, vtm_ref.NO_OFFSET)
        self.assertEqual(got.block_size, vtm_ref.Q4_SCALE_BLOCK)
        self.assertFalse(got.q4_asymmetric)
        self.assertIsNone(got.q4_mins)
        self.assertEqual(got.dense(), tensor.dense())

    def test_roundtrip_asymmetric(self):
        path, _model, tensor, q, scales, mins = self._container(
            asymmetric=True, scale_dtype="float16", n=64
        )
        back = vtm_ref.read_vtm(path)
        got = back.by_name("w")
        self.assertEqual(got.kind, vtm_ref.KIND_Q4)
        self.assertTrue(got.q4_asymmetric)
        self.assertEqual(got.scale_dtype, "float16")
        self.assertEqual(got.qdata, q)
        self.assertEqual(got.scales, scales)
        self.assertEqual(got.q4_mins, mins)
        self.assertEqual(got.dense(), tensor.dense())

    def test_python_execute_tmv(self):
        rows, cols = 2, 4
        n = rows * cols
        q = _q4values(n, seed=3)
        tensor = vtm_ref.Tensor.q4("w", (rows, cols), q, [0.5])
        x = vtm_ref.Tensor.f32("x", (cols,), [1.0, -2.0, 0.5, 3.0])
        y = vtm_ref.Tensor.f32("y", (rows,), [0.0] * rows)
        program = [
            vtm_ref.Instr(op=int(vtm_ref.Op.TMV), dst=2, a=0, b=1),
            vtm_ref.Instr(op=int(vtm_ref.Op.HALT)),
        ]
        path = self.path("tmv.vtm")
        model = vtm_ref.write_vtm(path, [tensor, x, y], program)
        result = vtm_ref.execute(model, [], 0)
        got = result.get("y")
        dense = tensor.dense()
        rows, cols = tensor.dims
        want = [
            sum(dense[r * cols + c] * x.data[c] for c in range(cols))
            for r in range(rows)
        ]
        self.assertEqual(len(got), rows)
        for a, b in zip(got, want):
            self.assertLessEqual(abs(a - b), 1e-6)


HARNESS_SOURCE = r"""
#include "vtm.h"

#include <stdint.h>
#include <stdio.h>

int main(int argc, char **argv) {
    vtm_t *m;
    const vtm_tensor *t;
    const uint8_t *q;
    uint32_t i, g;
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
    q = (const uint8_t *)(const void *)t->words;
    printf("kind %u\n", (unsigned)t->kind);
    printf("nelem %u\n", (unsigned)t->n_elements);
    printf("nscales %u\n", (unsigned)t->n_scales);
    printf("sblock %u\n", (unsigned)t->scale_block);
    printf("f32 %u\n", t->f32 != NULL ? 1u : 0u);
    printf("asym %u\n", (unsigned)t->q4_is_asym);
    printf("scales");
    for (g = 0; g < t->n_scales; g++)
        printf(" %.9g", (double)t->scales[g]);
    printf("\n");
    if (t->q4_is_asym) {
        printf("mins");
        for (g = 0; g < t->n_scales; g++)
            printf(" %.9g", (double)t->q4_mins[g]);
        printf("\n");
    }
    printf("q ");
    for (i = 0; i < t->n_scales * VTM_Q4_GROUP_BYTES; i++)
        printf("%02x", q[i]);
    printf("\n");
    vtm_free(m);
    return 0;
}
"""


class TestQ4CLoader(_TmpCase):
    """Python пишет `.vtm`, C-загрузчик читает те же байты без fp32."""

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
        cls._tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls._tmp.cleanup)
        source = os.path.join(cls._tmp.name, "dump_q4.c")
        cls.binary = os.path.join(cls._tmp.name, "dump_q4")
        with open(source, "w", encoding="utf-8") as handle:
            handle.write(HARNESS_SOURCE)
        try:
            built = subprocess.run(
                [os.environ.get("CC", "cc"), "-std=c99", "-O2",
                 "-I", SRC_VTM, "-o", cls.binary, source, VTM_LIB,
                 "-lm", "-lpthread"],
                capture_output=True, text=True, timeout=300,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise unittest.SkipTest(f"C-драйвер не собрался: {exc}")
        if built.returncode != 0:
            raise unittest.SkipTest(
                "C-драйвер не собрался: " + built.stderr.strip()[:300]
            )

    def dump(self, path, name="w"):
        proc = subprocess.run(
            [self.binary, path, name], capture_output=True, text=True,
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        result = {}
        for line in proc.stdout.splitlines():
            parts = line.split()
            if parts:
                result[parts[0]] = parts[1:]
        return result

    def _write(self, *, asymmetric=False, scale_dtype="float32", n=37):
        q = _q4values(n, seed=17)
        scales = _scales(n)
        mins = [-1.0, 2.0][:len(scales)] if asymmetric else None
        tensor = vtm_ref.Tensor.q4(
            "w", (n,), q, scales, scale_dtype=scale_dtype,
            asymmetric=asymmetric, mins=mins,
        )
        x = vtm_ref.Tensor.f32("x", (1,), [0.0])
        path = self.path("q4.vtm")
        vtm_ref.write_vtm(
            path, [tensor, x],
            [vtm_ref.Instr(op=int(vtm_ref.Op.HALT))],
            {"format": "q4"},
        )
        return path, tensor, q, scales, mins

    def test_c_reads_symmetric_packed(self):
        path, tensor, q, scales, _mins = self._write()
        got = self.dump(path)
        self.assertEqual(got["kind"], ["5"])
        self.assertEqual(got["nelem"], [str(tensor.n_elements)])
        self.assertEqual(got["nscales"], [str(len(scales))])
        self.assertEqual(got["sblock"], ["32"])
        self.assertEqual(got["f32"], ["0"])       # никакой fp32-материализации
        self.assertEqual(got["asym"], ["0"])
        self.assertEqual(
            [float(x) for x in got["scales"]],
            scales,
        )
        packed = vtm_ref.pack_q4(q)
        self.assertEqual(got["q"][0], packed.hex())

    def test_c_reads_asymmetric_mins_fp16(self):
        path, tensor, q, scales, mins = self._write(
            asymmetric=True, scale_dtype="float16", n=64
        )
        got = self.dump(path)
        self.assertEqual(got["kind"], ["5"])
        self.assertEqual(got["nscales"], [str(len(scales))])
        self.assertEqual(got["f32"], ["0"])
        self.assertEqual(got["asym"], ["1"])
        self.assertEqual([float(x) for x in got["scales"]], scales)
        self.assertEqual([float(x) for x in got["mins"]], mins)
        self.assertEqual(got["q"][0], vtm_ref.pack_q4(q).hex())

    def test_c_rejects_missing_q4_1_mins(self):
        # Сегмент Q4_1 без n_mins: C обязан отказать (VTM_E_FORMAT), а не
        # прочитать его как Q4_0. Пишем контейнер с намеренно испорченным
        # сегментом: убираем 4 байта n_mins из payload'а (offset'ы при этом
        # остаются согласованными, потому что writer считает их по длине).
        from unittest import mock

        q = _q4values(64, seed=17)
        scales = _scales(64)
        mins = [-1.0, 2.0]
        tensor = vtm_ref.Tensor.q4(
            "w", (64,), q, scales, asymmetric=True, mins=mins,
        )
        x = vtm_ref.Tensor.f32("x", (1,), [0.0])
        original = vtm_ref.encode_q4_segment

        def broken(item):
            segment = original(item)
            return segment[:vtm_ref.Q4_HEADER_SIZE] + segment[
                vtm_ref.Q4_ASYM_HEADER_SIZE:
            ]

        path = self.path("bad_q4_1.vtm")
        with mock.patch.object(vtm_ref, "encode_q4_segment", side_effect=broken):
            vtm_ref.write_vtm(
                path, [tensor, x],
                [vtm_ref.Instr(op=int(vtm_ref.Op.HALT))],
            )
        proc = subprocess.run(
            [self.binary, path, "w"], capture_output=True, text=True,
            timeout=120,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Q4_1", proc.stderr)


if __name__ == "__main__":
    unittest.main()
