"""Проверки тернарного квантования и формата тритов — без torch.

Запуск: python3 -m unittest discover -s tests -v
Контрольные векторы взяты из docs/trit-format.md §6.
"""
import math
import os
import struct
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qat import ternary  # noqa: E402


class TestRounding(unittest.TestCase):
    def test_half_away_from_zero(self):
        cases = {
            0.5: 1, -0.5: -1,
            1.5: 2, -1.5: -2,
            2.5: 3, -2.5: -3,
            0.499: 0, -0.499: 0,
            0.0: 0,
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(ternary.round_half_away(value), expected)

    def test_non_finite_rejected(self):
        for value in (math.inf, -math.inf, math.nan):
            with self.assertRaises(ternary.TritError):
                ternary.round_half_away(value)


class TestQuantization(unittest.TestCase):
    def test_default_block_is_128(self):
        self.assertEqual(ternary.BLOCK_SIZE, 128)
        # 257 значений = 128 + 128 + 1 -> три группы масштаба.
        self.assertEqual(len(ternary.block_scales([1.0] * 257)), 3)

    def test_scale_is_absmax_of_block(self):
        values = [0.5] * 64 + [-4.0] + [0.1] * 63
        trits, scales = ternary.quantize(values)
        self.assertEqual(len(scales), 1)
        self.assertAlmostEqual(scales[0], 4.0)
        self.assertEqual(trits[64], -1)
        self.assertIn(trits[0], (-1, 0, 1))

    def test_trits_are_ternary(self):
        values = [math.sin(i * 0.37) * (i % 5) for i in range(500)]
        trits, scales = ternary.quantize(values)
        self.assertTrue(set(trits) <= {-1, 0, 1})
        self.assertEqual(len(trits), len(values))
        self.assertEqual(len(scales), ternary.n_blocks(len(values)))

    def test_reconstruction_is_trit_times_scale(self):
        values = [((i * 37) % 101 - 50) / 7.0 for i in range(300)]
        trits, scales = ternary.quantize(values)
        restored = ternary.dequantize(trits, scales)
        self.assertEqual(len(restored), len(values))
        for i, value in enumerate(restored):
            block = i // ternary.BLOCK_SIZE
            self.assertEqual(value, trits[i] * scales[block])

    def test_zero_block(self):
        trits, scales = ternary.quantize([0.0] * 128)
        self.assertEqual(scales, [0.0])
        self.assertEqual(trits, [0] * 128)
        self.assertEqual(ternary.dequantize(trits, scales), [0.0] * 128)

    def test_blocks_do_not_bleed(self):
        values = [1.0] * 128 + [0.001] * 128
        trits, scales = ternary.quantize(values)
        self.assertEqual(len(scales), 2)
        self.assertAlmostEqual(scales[0], 1.0)
        self.assertAlmostEqual(scales[1], 0.001)
        self.assertEqual(trits[128:], [1] * 128)
        # Изменение первого блока не трогает второй.
        changed, changed_scales = ternary.quantize([2.0] + [0.0] * 127 + [0.001] * 128)
        self.assertAlmostEqual(changed_scales[1], 0.001)
        self.assertEqual(changed[128:], [1] * 128)

    def test_partial_last_block(self):
        values = [1.0] * 129
        trits, scales = ternary.quantize(values)
        self.assertEqual(len(scales), 2)
        self.assertEqual(len(trits), 129)
        self.assertAlmostEqual(scales[1], 1.0)

    def test_boundaries_1_5_scale(self):
        # absmax внутри блока не превышает scale, поэтому границу 1.5*scale
        # проверяем через явно заданные масштабы.
        scale = 2.0
        values = [3.0, -3.0, 2.98, -2.98, 1.0, -1.0, 0.98, -0.98, 4.0, -4.0, 0.0]
        trits = ternary.quantize_with_scale(values, [scale], block_size=16)
        self.assertEqual(trits, [1, -1, 1, -1, 1, -1, 0, 0, 1, -1, 0])

    def test_half_scale_rounds_away_from_zero(self):
        trits = ternary.quantize_with_scale([0.5, -0.5], [1.0], block_size=8)
        self.assertEqual(trits, [1, -1])

    def test_scale_zero_means_zero(self):
        trits = ternary.quantize_with_scale([1.0, -2.0, 0.0], [0.0], block_size=8)
        self.assertEqual(trits, [0, 0, 0])

    def test_scale_count_is_checked(self):
        with self.assertRaises(ternary.TritError):
            ternary.quantize_with_scale([1.0] * 129, [1.0], block_size=128)
        with self.assertRaises(ternary.TritError):
            ternary.dequantize([0] * 129, [1.0], block_size=128)

    def test_negative_or_infinite_scale_rejected(self):
        with self.assertRaises(ternary.TritError):
            ternary.dequantize([0], [-1.0])
        with self.assertRaises(ternary.TritError):
            ternary.dequantize([0], [math.inf])

    def test_non_finite_value_rejected(self):
        with self.assertRaises(ternary.TritError):
            ternary.quantize([1.0, math.nan])

    def test_invalid_trit_rejected(self):
        with self.assertRaises(ternary.TritError):
            ternary.dequantize([2], [1.0])


class TestPacking20(unittest.TestCase):
    def test_control_vectors_from_docs(self):
        # docs/trit-format.md §6: значение слова и байты little-endian.
        cases = [
            ([-1] * 20, 0, "00000000"),
            ([0] * 20, 1_743_392_200, "c80dea67"),
            ([1] * 20, 3_486_784_400, "901bd4cf"),
            ([1, -1] * 10, 871_696_100, "e406f533"),
        ]
        for trits, value, hex_bytes in cases:
            with self.subTest(trits=trits[:3]):
                packed = ternary.pack20(trits)
                self.assertEqual(len(packed), 4)
                self.assertEqual(int.from_bytes(packed, "little"), value)
                self.assertEqual(packed.hex(), hex_bytes)

    def test_max_word_fits_uint32(self):
        self.assertEqual(ternary.MAX_WORD, 3 ** 20 - 1)
        self.assertLess(ternary.MAX_WORD, 2 ** 32)
        self.assertEqual(
            int.from_bytes(ternary.pack20([1] * 20), "little"),
            ternary.MAX_WORD,
        )

    def test_roundtrip(self):
        trits = [1, 0, -1, 1, 1, 0, -1, 0, 1, -1, 0, 0, 1, 1, -1, 0, 1, 0, -1, 1]
        self.assertEqual(ternary.unpack20(ternary.pack20(trits), len(trits)), trits)

    def test_partial_word_is_zero_padded(self):
        trits = [1, -1, 1]
        data = ternary.pack20(trits)
        self.assertEqual(len(data), 4)
        self.assertEqual(ternary.unpack20(data, 3), trits)
        self.assertEqual(ternary.unpack20(data, 20), [1, -1, 1] + [0] * 17)

    def test_empty(self):
        self.assertEqual(ternary.pack20([]), b"")
        self.assertEqual(ternary.unpack20(b"", 0), [])

    def test_160_trits_are_32_bytes(self):
        self.assertEqual(len(ternary.pack20([1] * 160)), 32)
        self.assertEqual(len(ternary.words20([1] * 160)), 8)

    def test_bad_length(self):
        with self.assertRaises(ternary.TritError):
            ternary.unpack20(b"\x00", 0)

    def test_corrupted_word(self):
        with self.assertRaises(ternary.TritError):
            ternary.unpack20(b"\xff\xff\xff\xff", 20)

    def test_count_larger_than_data(self):
        with self.assertRaises(ternary.TritError):
            ternary.unpack20(ternary.pack20([1] * 20), 21)

    def test_invalid_trit(self):
        with self.assertRaises(ternary.TritError):
            ternary.pack20([2])


class TestTritFormat(unittest.TestCase):
    def test_segment_roundtrip(self):
        trits = [1, 0, -1, -1, 1, 0] * 50  # 300 тритов -> 3 группы масштаба
        scales = [0.5, 0.25, 0.125]
        data = ternary.encode_segment(trits, scales)
        self.assertEqual(data[:4], b"TRT1")
        decoded, decoded_scales = ternary.decode_segment(data)
        self.assertEqual(decoded, trits)
        for got, expected in zip(decoded_scales, scales):
            self.assertAlmostEqual(got, expected, places=3)

    def test_header_fields(self):
        data = ternary.encode_segment([1] * 160, [1.0, 2.0])
        magic, version, flags, n_trits, n_blocks = struct.unpack("<4sBBQI", data[:18])
        self.assertEqual(magic, b"TRT1")
        self.assertEqual(version, 1)
        self.assertEqual(flags, 0)
        self.assertEqual(n_trits, 160)
        self.assertEqual(n_blocks, 2)
        # 18 байт заголовка + 2 scale (fp16) + 8 слов (32 байта).
        self.assertEqual(len(data), 18 + 2 * 2 + 32)

    def test_storage_alignment(self):
        # 300 тритов: 15 слов данных, но блоки по 8 слов -> 16 слов.
        data = ternary.encode_segment([1] * 300, [1.0, 1.0, 1.0])
        self.assertEqual(len(data), 18 + 3 * 2 + 16 * 4)

    def test_float32_scales(self):
        data = ternary.encode_segment([1, -1] * 80, [0.5, 0.25], scale_dtype="float32")
        _, _, flags, _, _ = struct.unpack("<4sBBQI", data[:18])
        self.assertEqual(flags, ternary.FLAG_SCALE_F32)
        _, scales = ternary.decode_segment(data)
        self.assertEqual(scales, [0.5, 0.25])

    def test_info(self):
        data = ternary.encode_segment([1] * 160, [1.0, 1.0])
        info = ternary.segment_info(data)
        self.assertEqual(info.n_trits, 160)
        self.assertEqual(info.n_scale_blocks, 2)
        self.assertEqual(info.n_words, 8)
        self.assertEqual(info.n_storage_blocks, 1)
        self.assertEqual(info.scale_dtype, "float16")

    def test_bad_magic(self):
        data = bytearray(ternary.encode_segment([1] * 160, [1.0, 1.0]))
        data[0] = ord("X")
        with self.assertRaises(ternary.TritError):
            ternary.decode_segment(bytes(data))

    def test_bad_version(self):
        data = bytearray(ternary.encode_segment([1] * 160, [1.0, 1.0]))
        data[4] = 9
        with self.assertRaises(ternary.TritError):
            ternary.decode_segment(bytes(data))

    def test_corrupted_word(self):
        data = bytearray(ternary.encode_segment([1] * 160, [1.0, 1.0]))
        offset = 18 + 2 * 2
        data[offset:offset + 4] = b"\xff\xff\xff\xff"
        with self.assertRaises(ternary.TritError):
            ternary.decode_segment(bytes(data))

    def test_truncated(self):
        data = ternary.encode_segment([1] * 160, [1.0, 1.0])
        with self.assertRaises(ternary.TritError):
            ternary.decode_segment(data[:-1])

    def test_scale_overflow_to_float16(self):
        with self.assertRaises(ternary.TritError):
            ternary.encode_segment([1], [1e6], scale_dtype="float16")

    def test_scale_count_mismatch(self):
        with self.assertRaises(ternary.TritError):
            ternary.encode_segment([1] * 300, [1.0, 1.0])

    def test_empty_segment(self):
        data = ternary.encode_segment([], [])
        self.assertEqual(len(data), 18)
        self.assertEqual(ternary.decode_segment(data), ([], []))


class TestLazyTorch(unittest.TestCase):
    def test_import_does_not_pull_torch_or_yaml(self):
        code = (
            "import sys\n"
            "import qat.ternary, qat.config\n"
            "assert 'torch' not in sys.modules, 'torch импортирован на уровне модуля'\n"
            "assert 'yaml' not in sys.modules, 'yaml импортирован на уровне модуля'\n"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
        subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    unittest.main()
