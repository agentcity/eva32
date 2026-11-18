"""Флаг квантования эмбеддингов (задача 0027): конфиг и ленивая работа без torch."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qat.config import QuantConfig  # noqa: E402


class TestConfigFlag(unittest.TestCase):
    def test_default_is_none(self):
        self.assertEqual(QuantConfig().quantize_embeddings, "none")

    def test_accepts_known_modes(self):
        for mode in ("none", "int8", "trit"):
            self.assertEqual(QuantConfig(quantize_embeddings=mode).quantize_embeddings, mode)

    def test_mode_is_readable_from_config(self):
        cfg = QuantConfig(quantize_embeddings="trit")
        self.assertEqual(cfg.quantize_embeddings, "trit")
        self.assertEqual(cfg.block_size, 128)


class TestTorchPath(unittest.TestCase):
    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch не установлен — GPU-путь проверяется на аренде")

    def test_unknown_mode_rejected(self):
        from qat.model import apply_embedding_quant
        with self.assertRaises(ValueError):
            apply_embedding_quant(object(), QuantConfig(quantize_embeddings="fp4"))

    def test_none_is_noop(self):
        from qat.model import apply_embedding_quant
        self.assertIsNone(apply_embedding_quant(object(), QuantConfig()))


if __name__ == "__main__":
    unittest.main()
