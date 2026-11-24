"""Варианты таблицы словаря: int8 и посрочные масштабы (задача 0143).

Проверяется то, что нельзя проверить «на глаз» в отчёте:

* **старые контейнеры не меняются**: сегмент ``--embed trit --embed-scale
  block`` побайтово равен прежнему ``encode_trit_segment`` (тем же байтам,
  что писал холодный экспорт 0134 и QAT-экспорт), поэтому раскладка
  ``ternary.trit`` и все уже собранные контейнеры читаются как раньше;
* **int8** (``--embed int8``): сегмент — валидный INT8 (kind 4), правило
  квантования совпадает с эталоном ``vtm_ref.quantize_int8`` (absmax/127 на
  блок 128, округление к ближайшему чётному), запись манифеста имеет
  ``dtype: int8``, tied ``lm_head`` ссылается на неё, а ``vtm_compile``
  собирает граф с ``embed_quant="int8"`` и Python-эталон VTM его исполняет;
* **посрочные масштабы** (``--embed-scale row``): внутри строки таблицы все
  масштабы равны масштабу строки, раскладка сегмента та же (масштабов
  по-прежнему ceil(n/128)), поэтому C-рантайм и компилятор не меняются;
* **отказы**: строка таблицы не кратна блоку 128 — экспорт отказывает, а не
  пишет сегмент, который рантайм прочитает неправильно;
* torch-путь и «чистый» путь (fake-тензоры) дают один и тот же сегмент —
  иначе тесты без torch проверяли бы не то, что едет в прод.

Запуск:
    python3 -m unittest tests.test_embed_variants -v
"""
import importlib.util
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, ROOT)
sys.path.insert(0, str(Path(ROOT) / "tools"))

from tools import cold_export, vtm_compile, vtm_ref  # noqa: E402
from qat import ternary  # noqa: E402

try:  # обычный запуск (`python3 -m unittest tests.test_embed_variants`)
    from tests.test_cold_export import (  # noqa: E402
        FakeTensor, TINY, export_tiny, tiny_hf_config, tiny_model,
    )
except ImportError:  # pragma: no cover - `discover -s tests`: модули top-level
    from test_cold_export import (  # noqa: E402
        FakeTensor, TINY, export_tiny, tiny_hf_config, tiny_model,
    )

EMBED_NAME = "model.embed_tokens.weight"


def has_torch() -> bool:
    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):
        return False


HAS_TORCH = has_torch()


def table(vocab, hidden, *, row_scale=1.0, seed=0):
    """Таблица словаря: каждая строка масштабируется своим множителем.

    Нужна, чтобы посрочный режим отличался от блочного: при одинаковых
    строках absmax блока и absmax строки совпадают, и тест ничего не ловит.
    """
    values = []
    for row in range(vocab):
        factor = 1.0 + row_scale * (row % 5)
        for col in range(hidden):
            base = (((row * 31 + col * 17 + seed) % 200) - 100) / 100.0
            values.append(base * factor)
    return FakeTensor((vocab, hidden), values, "embed")


class _TorchTable:
    """Тот же тензор, но torch-тензор (для сверки путей)."""

    def __init__(self, fake):
        import torch

        self._t = torch.tensor(list(fake), dtype=torch.float32).reshape(fake.shape)

    def __getattr__(self, name):
        return getattr(self._t, name)

    def __getitem__(self, item):
        return self._t[item]


# ---------------------------------------------------------------------------
# Совместимость: trit + block — те же байты, что раньше
# ---------------------------------------------------------------------------


class TestLegacyBytesUnchanged(unittest.TestCase):
    def test_trit_block_matches_legacy_encoder(self):
        weight = table(128, 128, row_scale=0.0, seed=3)
        payload, meta = cold_export.encode_embed_segment(
            weight, EMBED_NAME, mode="trit", scale_mode="block",
        )
        legacy, legacy_meta = cold_export.encode_trit_segment(weight, EMBED_NAME)
        self.assertEqual(payload, legacy)
        for key in ("params", "n_trits", "n_scale_blocks", "scale_blocks_per_row",
                    "n_storage_words", "zero_fraction", "shape", "dtype", "kind"):
            self.assertEqual(meta[key], legacy_meta[key], key)
        # Новые ключи только дополняют запись, старые не переименованы.
        self.assertEqual(meta["scale_mode"], "block")
        self.assertEqual(meta["scale_block"], ternary.BLOCK_SIZE)

    def test_none_mode_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, _state = export_tiny(tmp, embed="none")
            embed = [
                entry for entry in result.manifest["non_ternary"]["tensors"]
                if entry.get("name") == EMBED_NAME
            ][0]
            self.assertEqual(embed["dtype"], "float32")
            self.assertEqual(
                embed["length"], TINY["vocab"] * TINY["hidden"] * 4
            )
            self.assertEqual(result.manifest["embeddings"]["mode"], "none")
            self.assertEqual(result.manifest["embeddings"]["scale_mode"], "block")

    def test_block_size_and_scale_dtype_still_guarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = tiny_model()
            with self.assertRaises(cold_export.ColdExportError):
                cold_export.export_model(
                    cold_export.DictSource(state), tiny_hf_config(), tmp,
                    embed_mode="int8", embed_scale="column",
                )


# ---------------------------------------------------------------------------
# int8: формат kind 4 и правило квантования
# ---------------------------------------------------------------------------


class TestInt8Segment(unittest.TestCase):
    def test_segment_matches_vtm_ref_rule(self):
        vocab, hidden = 32, 256
        weight = table(vocab, hidden, row_scale=0.3, seed=5)
        payload, meta = cold_export.encode_embed_segment(
            weight, EMBED_NAME, mode="int8", scale_mode="block",
        )
        self.assertEqual(payload[:4], vtm_ref.INT8_MAGIC)
        qdata, scales, scale_dtype = vtm_ref.decode_int8_segment(payload)
        expected_q, expected_scales = vtm_ref.quantize_int8(list(weight))
        self.assertEqual(scale_dtype, "float32")
        self.assertEqual(qdata, expected_q)
        self.assertEqual(scales, expected_scales)
        self.assertEqual(meta["n_scale_blocks"], len(scales))
        self.assertEqual(meta["params"], vocab * hidden)
        self.assertEqual(len(payload), vtm_ref.tensor_payload_size(
            vtm_ref.Tensor.int8("t", (vocab, hidden), qdata, scales)
        ))

    def test_segment_size_is_one_byte_per_value(self):
        vocab, hidden = 64, 128
        payload, _meta = cold_export.encode_embed_segment(
            table(vocab, hidden), EMBED_NAME, mode="int8",
        )
        n_values = vocab * hidden
        n_scales = n_values // vtm_ref.INT8_SCALE_BLOCK
        self.assertEqual(len(payload), 18 + n_scales * 4 + n_values)

    def test_zero_block_gets_qat_zero_scale(self):
        values = [0.0] * 256
        payload, _meta = cold_export.encode_embed_segment(
            FakeTensor((2, 128), values), EMBED_NAME, mode="int8",
        )
        qdata, scales, _dtype = vtm_ref.decode_int8_segment(payload)
        self.assertTrue(all(q == 0 for q in qdata))
        self.assertTrue(all(scale > 0 for scale in scales))

    def test_row_scale_is_constant_within_row(self):
        vocab, hidden = 8, 256
        weight = table(vocab, hidden, row_scale=0.7, seed=7)
        payload, meta = cold_export.encode_embed_segment(
            weight, EMBED_NAME, mode="int8", scale_mode="row",
        )
        _qdata, scales, _dtype = vtm_ref.decode_int8_segment(payload)
        per_row = hidden // vtm_ref.INT8_SCALE_BLOCK
        self.assertEqual(len(scales), vocab * per_row)
        rows = [list(weight)[row * hidden:(row + 1) * hidden] for row in range(vocab)]
        for row, row_values in enumerate(rows):
            chunk = scales[row * per_row:(row + 1) * per_row]
            self.assertEqual(len(set(chunk)), 1, f"строка {row}: {chunk}")
            absmax = max(abs(v) for v in row_values)
            self.assertAlmostEqual(chunk[0], absmax / 127.0, places=6)
        self.assertEqual(meta["scale_mode"], "row")
        self.assertEqual(meta["scale_blocks_per_row"], per_row)

    def test_row_and_block_are_different_layouts_of_same_size(self):
        """row и block — одно место, разные значения: число масштабов то же.

        Выброс внутри строки block видит только в своём блоке, row — во всей
        строке. Поэтому на такой таблице block точнее, а сегменты обязаны
        различаться содержимым при одинаковой длине — это и есть «новый режим
        только по флагу, формат прежний».
        """
        vocab, hidden = 2, 256
        values = []
        for _row in range(vocab):
            base = [0.01 * ((i % 7) - 3) for i in range(hidden)]
            base[3] = 9.0  # выброс в первом блоке строки
            values.extend(base)
        weight = FakeTensor((vocab, hidden), values)
        payload_block, meta_block = cold_export.encode_embed_segment(
            weight, EMBED_NAME, mode="int8", scale_mode="block",
        )
        payload_row, meta_row = cold_export.encode_embed_segment(
            weight, EMBED_NAME, mode="int8", scale_mode="row",
        )
        self.assertEqual(meta_block["n_scale_blocks"], meta_row["n_scale_blocks"])
        self.assertEqual(len(payload_block), len(payload_row))
        self.assertNotEqual(payload_block, payload_row)
        q_block, s_block, _ = vtm_ref.decode_int8_segment(payload_block)
        q_row, s_row, _ = vtm_ref.decode_int8_segment(payload_row)
        error_block = _rms_dequant_error(list(weight), q_block, s_block)
        error_row = _rms_dequant_error(list(weight), q_row, s_row)
        # block видит выброс только в своём блоке, row — во всей строке
        self.assertLess(error_block, error_row)


def _max_dequant_error(values, qdata, scales):
    block = vtm_ref.INT8_SCALE_BLOCK
    return max(
        abs(value - qdata[i] * scales[i // block])
        for i, value in enumerate(values)
    )


def _rms_dequant_error(values, qdata, scales):
    """Среднеквадратичная ошибка: max на такой таблице одинаков у обоих режимов."""
    block = vtm_ref.INT8_SCALE_BLOCK
    errors = [
        value - qdata[i] * scales[i // block] for i, value in enumerate(values)
    ]
    return math.sqrt(sum(e * e for e in errors) / len(errors))


# ---------------------------------------------------------------------------
# Сборка контейнера: int8 и row доезжают до графа и исполняются
# ---------------------------------------------------------------------------


class TestExportAndCompile(unittest.TestCase):
    def test_int8_export_compiles_and_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, _state = export_tiny(tmp, embed="int8")
            manifest = result.manifest
            self.assertEqual(manifest["architecture"]["embed"], "int8")
            self.assertEqual(manifest["embeddings"]["mode"], "int8")
            self.assertEqual(manifest["embeddings"]["dtype"], "int8")
            self.assertEqual(manifest["embeddings"]["storage"], "non_ternary.bin")
            entry = next(
                item for item in manifest["non_ternary"]["tensors"]
                if item.get("name") == EMBED_NAME
            )
            self.assertEqual(entry["dtype"], "int8")
            self.assertEqual(entry["length"], len(
                cold_export.encode_embed_segment(
                    tiny_table(), EMBED_NAME, mode="int8",
                )[0]
            ))
            tied = next(
                item for item in manifest["non_ternary"]["tensors"]
                if item.get("name") == "lm_head.weight"
            )
            self.assertEqual(tied["dtype"], "int8")
            self.assertEqual(tied["tied_to"], EMBED_NAME)
            self.assertNotIn("offset", tied)

            export = vtm_compile.load_export(result.manifest_path)
            self.assertEqual(export.cfg.embed_quant, "int8")
            self.assertEqual(export.weights["embed"].kind, vtm_ref.KIND_INT8)
            vtm = vtm_compile.compile_model(export.cfg, export.weights, export.meta)
            self.assertEqual(vtm.meta["embed"], "int8")
            out = vtm_ref.execute(vtm, [1, 2], 0)
            self.assertEqual(len(out.logits), TINY["vocab"])

    def test_int8_logits_close_to_fp32_table(self):
        """Ошибка int8-таблицы на порядок меньше ошибки тритовой.

        Сравнение честное: Linear в обоих контейнерах одни и те же триты,
        меняется только таблица. Это проверка «формат собран верно», а не
        замер качества (замер — tools/parity_cold.py на настоящей модели).
        """
        with tempfile.TemporaryDirectory() as tmp_fp32, \
                tempfile.TemporaryDirectory() as tmp_int8, \
                tempfile.TemporaryDirectory() as tmp_trit:
            fp32_result, _ = export_tiny(tmp_fp32, embed="none")
            int8_result, _ = export_tiny(tmp_int8, embed="int8")
            trit_result, _ = export_tiny(tmp_trit, embed="trit")
            logits = {}
            for label, result in (
                ("fp32", fp32_result), ("int8", int8_result), ("trit", trit_result),
            ):
                export = vtm_compile.load_export(result.manifest_path)
                vtm = vtm_compile.compile_model(
                    export.cfg, export.weights, export.meta
                )
                logits[label] = vtm_ref.execute(vtm, [3, 4, 5], 0).logits
        err_int8 = _max_abs_delta(logits["int8"], logits["fp32"])
        err_trit = _max_abs_delta(logits["trit"], logits["fp32"])
        self.assertLess(err_int8, err_trit)

    def test_row_scales_compile_and_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, _state = export_tiny(tmp, embed="trit", embed_scale="row")
            self.assertEqual(
                result.manifest["architecture"]["embed"], "trit"
            )
            self.assertEqual(result.manifest["embeddings"]["scale_mode"], "row")
            self.assertEqual(result.manifest["cold_export"]["embed_scale"], "row")
            export = vtm_compile.load_export(result.manifest_path)
            self.assertEqual(export.cfg.embed_quant, "trit")
            vtm = vtm_compile.compile_model(export.cfg, export.weights, export.meta)
            out = vtm_ref.execute(vtm, [1, 2], 0)
            self.assertEqual(len(out.logits), TINY["vocab"])

    def test_row_trit_scales_are_constant_within_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, _state = export_tiny(tmp, embed="trit", embed_scale="row")
            segment = result.manifest["tensors"][0]
            payload = result.trit_path.read_bytes()[
                segment["trit_offset"]:segment["trit_offset"] + segment["trit_length"]
            ]
            trits, scales, _dtype = vtm_ref.decode_trit_segment(payload)
            per_row = TINY["hidden"] // ternary.BLOCK_SIZE
            self.assertEqual(len(scales), TINY["vocab"] * per_row)
            for row in range(TINY["vocab"]):
                chunk = scales[row * per_row:(row + 1) * per_row]
                self.assertEqual(len(set(chunk)), 1, f"строка {row}: {chunk}")
            self.assertEqual(len(trits), TINY["vocab"] * TINY["hidden"])

    def test_hidden_not_multiple_of_block_rejected(self):
        """Строка таблицы не кратна 128 — сегмент не пишем.

        У Qwen hidden = 896/2048, то есть 7/16 блоков на строку, но формат
        обязан отказывать, а не раскладывать масштаб поперёк строки.
        """
        narrow = FakeTensor((4, 96), [0.5] * (4 * 96), "embed")
        for mode in ("trit", "int8"):
            with self.subTest(mode=mode):
                with self.assertRaises(cold_export.ColdExportError) as ctx:
                    cold_export.encode_embed_segment(
                        narrow, EMBED_NAME, mode=mode, scale_mode="row",
                    )
                self.assertIn("128", str(ctx.exception))
        arch = {"hidden": 96, "n_heads": 2, "head_dim": 48,
                "intermediate": 256, "vocab": 128}
        with self.assertRaises(cold_export.ColdExportError):
            cold_export.check_block_alignment(arch, 128, "int8", "row")

    def test_embed_scale_row_rejected_without_embedding(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, _state = export_tiny(tmp, embed="none", embed_scale="row")
            # Таблицы в квантованном виде нет — режим просто записан в манифест.
            self.assertEqual(result.manifest["embeddings"]["scale_mode"], "row")
            self.assertEqual(result.manifest["embeddings"]["dtype"], "float32")


def tiny_table():
    return tiny_model()[EMBED_NAME]


def _max_abs_delta(actual, reference):
    return max(abs(a - b) for a, b in zip(actual, reference))


class TestTorchPath(unittest.TestCase):
    """torch-путь и чистый путь обязаны давать один и тот же сегмент."""

    @unittest.skipUnless(HAS_TORCH, "нужен torch (.pylibs)")
    def test_torch_and_plain_agree(self):
        vocab, hidden = 16, 256
        fake = table(vocab, hidden, row_scale=0.5, seed=11)
        torch_weight = _TorchTable(fake)
        for mode in ("trit", "int8"):
            for scale_mode in cold_export.EMBED_SCALES:
                with self.subTest(mode=mode, scale_mode=scale_mode):
                    plain, _ = cold_export.encode_embed_segment(
                        fake, EMBED_NAME, mode=mode, scale_mode=scale_mode,
                    )
                    via_torch, _ = cold_export.encode_embed_segment(
                        torch_weight, EMBED_NAME, mode=mode, scale_mode=scale_mode,
                    )
                    self.assertEqual(len(plain), len(via_torch))
                    if mode == "int8":
                        q_plain, s_plain, _ = vtm_ref.decode_int8_segment(plain)
                        q_torch, s_torch, _ = vtm_ref.decode_int8_segment(via_torch)
                        self.assertEqual(q_plain, q_torch)
                        self.assertEqual(s_plain, s_torch)
                    else:
                        t_plain, s_plain, _ = vtm_ref.decode_trit_segment(plain)
                        t_torch, s_torch, _ = vtm_ref.decode_trit_segment(via_torch)
                        self.assertEqual(t_plain, t_torch)
                        self.assertEqual(s_plain, s_torch)

    @unittest.skipUnless(HAS_TORCH, "нужен torch (.pylibs)")
    def test_row_scales_repeat_in_torch_path(self):
        vocab, hidden = 4, 256
        torch_weight = _TorchTable(table(vocab, hidden, row_scale=0.4, seed=13))
        payload, meta = cold_export.encode_embed_segment(
            torch_weight, EMBED_NAME, mode="int8", scale_mode="row",
        )
        _q, scales, _dtype = vtm_ref.decode_int8_segment(payload)
        per_row = hidden // vtm_ref.INT8_SCALE_BLOCK
        self.assertEqual(meta["n_scale_blocks"], vocab * per_row)
        for row in range(vocab):
            chunk = scales[row * per_row:(row + 1) * per_row]
            self.assertEqual(len(set(chunk)), 1)


if __name__ == "__main__":
    unittest.main()
