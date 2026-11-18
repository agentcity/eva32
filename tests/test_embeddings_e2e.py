"""Эмбеддинги в тритах сквозь весь путь (задача 0043).

Проверяем не отдельные функции, а маршрут целиком:

1. `qat.export.write_weight_files(..., quantize_embeddings="trit")` квантует
   таблицу словаря тем же правилом QAT (absmax по блоку 128) и кладёт её
   сегментом `model.embed_tokens.weight` в `ternary.trit`; в `non_ternary.bin`
   таблицы нет, в манифесте — запись `kind: trit`, `role: embed`;
2. `tools/vtm_compile.compile_model` собирает EMB по тритовому тензору, а
   tied lm_head — MM с A=emb.w (тритовая матрица);
3. C-интерпретатор читает тритовый EMB через LUT243: значения строки
   совпадают с расквантовкой в Python до 1e-6, а логиты tied lm_head —
   с Python-эталоном;
4. `int8`-эмбеддинги честно отложены: экспорт падает с объяснением, а не
   пишет fp32 молча.

torch на машине нет, поэтому чекпоинт — fake-тензоры (как в 0029): экспорт
идёт тем же кодом, что на GPU-машине, только по чистому пути без torch.
C-часть собирается из `src/vtm/libvtm.a`; если компилятора нет — skip.

Запуск:
    python3 -m unittest tests.test_embeddings_e2e -v
"""

import json
import os
import random
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qat import export as qat_export  # noqa: E402
from qat import ternary  # noqa: E402
from tools import vtm_compile, vtm_ref  # noqa: E402

SRC_VTM = os.path.join(ROOT, "src", "vtm")
VTM_LIB = os.path.join(SRC_VTM, "libvtm.a")

#: Параметризованные веса QAT лежат под этим суффиксом.
PARAM = qat_export.PARAM_SUFFIX
HIDDEN = 128
VOCAB = 128
INTERMEDIATE = 256
Q_DIM = 128
KV_DIM = 32
HEAD_DIM = 32
CTX = 4


class FakeTensor:
    """Мини-замена torch.Tensor: shape/numel/data_ptr + значения.

    `data_ptr` нужен экспорту для дедупликации tied-весов; общего указателя
    нет — поэтому у связанных таблиц он один и тот же по построению теста.
    """

    def __init__(self, values, shape, ptr):
        self.values = list(values)
        self.shape = tuple(shape)
        self._ptr = ptr

    def numel(self):
        return len(self.values)

    def data_ptr(self):
        return self._ptr

    def __iter__(self):
        return iter(self.values)


def _architecture(mode):
    """architecture из манифеста — как её пишет qat.export."""
    return {
        "hidden": HIDDEN,
        "n_heads": 4,
        "n_kv_heads": 1,
        "head_dim": HEAD_DIM,
        "intermediate": INTERMEDIATE,
        "vocab": VOCAB,
        "n_layers": 1,
        "context_len": CTX,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-3,
        "tie_word_embeddings": True,
        "lm_head": "mm",
        "embed": mode,
    }


def _state_dict(*, trit_embed):
    """state_dict QAT-подобной модели на fake-тензорах.

    При `trit_embed=True` эмбеддинг параметризован (так его видит чекпоинт
    после `apply_embedding_quant`), при `False` — обычный `...weight`.
    """
    rng = random.Random(11)

    def floats(n):
        return [rng.uniform(-1.0, 1.0) for _ in range(n)]

    def norm():
        return [rng.uniform(0.5, 1.5) for _ in range(HIDDEN)]

    def matrix(rows, cols, ptr):
        return FakeTensor(floats(rows * cols), (rows, cols), ptr)

    state = {}
    if trit_embed:
        state[f"model.embed_tokens{PARAM}"] = FakeTensor(
            floats(VOCAB * HIDDEN), (VOCAB, HIDDEN), 10
        )
    else:
        state["model.embed_tokens.weight"] = FakeTensor(
            floats(VOCAB * HIDDEN), (VOCAB, HIDDEN), 10
        )
    state[f"model.layers.0.self_attn.q_proj{PARAM}"] = matrix(Q_DIM, HIDDEN, 11)
    state[f"model.layers.0.self_attn.k_proj{PARAM}"] = matrix(KV_DIM, HIDDEN, 12)
    state[f"model.layers.0.self_attn.v_proj{PARAM}"] = matrix(KV_DIM, HIDDEN, 13)
    state[f"model.layers.0.self_attn.o_proj{PARAM}"] = matrix(HIDDEN, Q_DIM, 14)
    state[f"model.layers.0.mlp.gate_proj{PARAM}"] = matrix(INTERMEDIATE, HIDDEN, 15)
    state[f"model.layers.0.mlp.up_proj{PARAM}"] = matrix(INTERMEDIATE, HIDDEN, 16)
    state[f"model.layers.0.mlp.down_proj{PARAM}"] = matrix(HIDDEN, INTERMEDIATE, 17)
    state["model.layers.0.input_layernorm.weight"] = FakeTensor(norm(), (HIDDEN,), 18)
    state["model.layers.0.post_attention_layernorm.weight"] = FakeTensor(
        norm(), (HIDDEN,), 19
    )
    state["model.norm.weight"] = FakeTensor(norm(), (HIDDEN,), 20)
    return state


def write_export(directory, mode):
    """Собрать экспорт `mode` (none|trit) настоящим кодом qat.export.

    Возвращает `(путь к manifest.json, разобранный манифест, размеры файлов)`.
    """
    os.makedirs(directory, exist_ok=True)
    architecture = _architecture(mode)
    files = qat_export.write_weight_files(
        directory,
        _state_dict(trit_embed=(mode == "trit")),
        architecture,
        block_size=ternary.BLOCK_SIZE,
        scale_dtype="float32",
        quantize_embeddings=mode,
    )
    manifest = qat_export.build_manifest(
        files=files,
        architecture=architecture,
        config={"train": {"max_steps": 3}},
        quant_cfg={"block_size": ternary.BLOCK_SIZE, "scale_dtype": "float32"},
        block_size=ternary.BLOCK_SIZE,
        scale_dtype="float32",
        run_dir="/fake/run",
        ckpt="/fake/run/checkpoints/step_000001",
        step=1,
        samples_seen=10,
        model_cfg={"name_or_path": "fake/model", "revision": "test"},
        file_stats={},
    )
    manifest_path = os.path.join(directory, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    sizes = {
        name: os.path.getsize(os.path.join(directory, name))
        for name in ("ternary.trit", "non_ternary.bin")
    }
    return manifest_path, manifest, sizes


def _trit_entries(manifest):
    return {entry["name"]: entry for entry in manifest["tensors"]}


class _TmpCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmpdir = self._tmp.name

    def path(self, *parts):
        return os.path.join(self.tmpdir, *parts)


# ---------------------------------------------------------------------------
# Экспорт: таблица словаря в ternary.trit
# ---------------------------------------------------------------------------


class TestTritEmbeddingExport(_TmpCase):
    def setUp(self):
        super().setUp()
        self.trit_dir = self.path("trit")
        self.none_dir = self.path("none")
        self.manifest_path, self.manifest, self.sizes = write_export(
            self.trit_dir, "trit"
        )
        _, _, self.none_sizes = write_export(self.none_dir, "none")

    def _trit_bytes(self):
        with open(os.path.join(self.trit_dir, "ternary.trit"), "rb") as handle:
            return handle.read()

    def test_table_is_a_trit_segment(self):
        entry = _trit_entries(self.manifest)["model.embed_tokens.weight"]
        self.assertEqual(entry["kind"], "trit")
        self.assertEqual(entry["role"], "embed")
        self.assertEqual(entry["dtype"], "trit")
        self.assertEqual(entry["shape"], [VOCAB, HIDDEN])
        self.assertEqual(entry["n_trits"], VOCAB * HIDDEN)
        self.assertEqual(entry["n_scale_blocks"], VOCAB * HIDDEN // ternary.BLOCK_SIZE)
        # Сегмент лежит в файле по offset из манифеста и читается кодеком.
        blob = self._trit_bytes()[entry["trit_offset"]:
                                  entry["trit_offset"] + entry["trit_length"]]
        trits, scales = ternary.decode_segment(blob, ternary.BLOCK_SIZE)
        self.assertEqual(len(trits), VOCAB * HIDDEN)
        self.assertEqual(len(scales), entry["n_scale_blocks"])
        self.assertTrue(all(t in (-1, 0, 1) for t in trits))
        self.assertTrue(all(s >= 0 for s in scales))
        self.assertEqual(self.manifest["embeddings"]["mode"], "trit")
        self.assertEqual(self.manifest["embeddings"]["storage"], "ternary.trit")
        self.assertEqual(self.manifest["architecture"]["embed"], "trit")

    def test_embedding_not_duplicated_in_non_ternary(self):
        section = self.manifest["non_ternary"]
        names = [entry["name"] for entry in section["tensors"]]
        self.assertNotIn("model.embed_tokens.weight", names)
        by_name = {entry["name"]: entry for entry in section["tensors"]}
        head = by_name["lm_head.weight"]
        self.assertEqual(head["tied_to"], "model.embed_tokens.weight")
        self.assertEqual(head["dtype"], "trit")
        self.assertNotIn("offset", head)  # байт у ссылки нет
        # В non_ternary остались только три нормы: 3 × hidden × 4 байта.
        self.assertEqual(
            self.sizes["non_ternary.bin"], 3 * HIDDEN * 4
        )

    def test_none_keeps_fp32_embedding(self):
        with open(os.path.join(self.none_dir, "manifest.json"), encoding="utf-8") as handle:
            none_manifest = json.load(handle)
        self.assertEqual(none_manifest["embeddings"]["mode"], "none")
        self.assertEqual(none_manifest["embeddings"]["storage"], "non_ternary.bin")
        self.assertNotIn(
            "model.embed_tokens.weight",
            [entry["name"] for entry in none_manifest["tensors"]],
        )
        self.assertIn(
            "model.embed_tokens.weight",
            [entry["name"] for entry in none_manifest["non_ternary"]["tensors"]],
        )
        self.assertEqual(
            self.none_sizes["non_ternary.bin"],
            3 * HIDDEN * 4 + VOCAB * HIDDEN * 4,
        )

    def test_export_size_smaller_than_none(self):
        trit_total = self.sizes["ternary.trit"] + self.sizes["non_ternary.bin"]
        none_total = (self.none_sizes["ternary.trit"]
                      + self.none_sizes["non_ternary.bin"])
        embed_fp32 = VOCAB * HIDDEN * 4
        embed_trit = self.manifest["embeddings"]["bytes"]
        print(
            f"\n[0043] синтетика {VOCAB}×{HIDDEN}: trit export "
            f"{trit_total} Б (trit {self.sizes['ternary.trit']} + "
            f"non_ternary {self.sizes['non_ternary.bin']}, таблица "
            f"{embed_trit} Б) против none {none_total} Б "
            f"(тр. {self.none_sizes['ternary.trit']} + "
            f"non_ternary {self.none_sizes['non_ternary.bin']}, таблица "
            f"{embed_fp32} Б): выигрыш {none_total - trit_total} Б "
            f"({100.0 * (none_total - trit_total) / none_total:.1f}%)"
        )
        self.assertLess(embed_trit, embed_fp32)
        self.assertLess(trit_total, none_total)
        self.assertEqual(none_total - trit_total, embed_fp32 - embed_trit)

    def test_int8_is_deferred_with_reason(self):
        with self.assertRaisesRegex(RuntimeError, "отложен"):
            qat_export.write_weight_files(
                self.path("int8"), _state_dict(trit_embed=True),
                _architecture("none"),
                quantize_embeddings="int8",
            )
        self.assertIn("kind 4", qat_export.EMBED_INT8_DEFERRED)
        self.assertIn("loader.c", qat_export.EMBED_INT8_DEFERRED)
        with self.assertRaisesRegex(ValueError, "int8"):
            vtm_compile.ModelConfig(embed_quant="int8")


# ---------------------------------------------------------------------------
# Компилятор: EMB и tied lm_head по тритовой таблице
# ---------------------------------------------------------------------------


class TestTritEmbeddingCompile(_TmpCase):
    def setUp(self):
        super().setUp()
        self.manifest_path, self.manifest, self.sizes = write_export(
            self.path("trit"), "trit"
        )
        self.export = vtm_compile.load_export(self.manifest_path)
        self.model = vtm_compile.compile_model(self.manifest_path)

    def test_load_export_reads_trit_table(self):
        embed = self.export.weights["embed"]
        self.assertEqual(embed.kind, vtm_ref.KIND_TRITS)
        self.assertEqual(tuple(embed.dims), (VOCAB, HIDDEN))
        self.assertEqual(self.export.cfg.embed_quant, "trit")
        self.assertTrue(self.export.tied)
        self.assertNotIn("lm_head", self.export.weights)  # tied: ссылка

    def test_emb_and_tied_mm_use_trit_table(self):
        embed_index = next(
            index for index, tensor in enumerate(self.model.tensors)
            if tensor.name == "emb.w"
        )
        self.assertEqual(self.model.tensors[embed_index].kind, vtm_ref.KIND_TRITS)
        emb = [i for i in self.model.program if int(i.op) == int(vtm_ref.Op.EMB)]
        mm = [i for i in self.model.program if int(i.op) == int(vtm_ref.Op.MM)]
        self.assertEqual(len(emb), 1)
        self.assertEqual(emb[0].a, embed_index)
        self.assertEqual(len(mm), 1, "tied lm_head обязан идти через MM")
        self.assertEqual(mm[0].a, embed_index)
        self.assertEqual(self.model.meta["embed"], "trit")
        self.assertEqual(self.model.meta["quantize_embeddings"], "trit")

    def test_trit_table_matches_fp32_equivalent(self):
        """Тритовая таблица в графе = та же таблица, развёрнутая в fp32."""
        weights = self.export.weights
        trits, scales = weights["embed"].trits, weights["embed"].scales
        dense = ternary.dequantize(trits, scales, ternary.BLOCK_SIZE)
        fp32_weights = dict(weights)
        fp32_weights["embed"] = vtm_ref.Tensor.f32(
            "emb.w", (VOCAB, HIDDEN), dense
        )
        cfg = vtm_compile.ModelConfig(
            hidden=HIDDEN, n_heads=4, n_kv_heads=1, head_dim=HEAD_DIM,
            intermediate=INTERMEDIATE, vocab=VOCAB, n_layers=1,
            context_len=CTX, eps=1e-3, tie_word_embeddings=True,
            lm_head="mm", embed_quant="none",
        )
        fp32_model = vtm_compile.compile_model(cfg, fp32_weights)
        for tokens in ([0], [7, 13]):
            with self.subTest(tokens=tokens):
                got = vtm_ref.execute(self.model, tokens, 0).logits
                want = vtm_ref.execute(fp32_model, tokens, 0).logits
                self.assertLess(
                    max(abs(a - b) for a, b in zip(got, want)), 1e-5
                )


# ---------------------------------------------------------------------------
# C-рантайм: EMB из тритов = расквантование в Python
# ---------------------------------------------------------------------------


HARNESS_SOURCE = r"""
#include <stdio.h>
#include <stdlib.h>
#include "vtm.h"

int main(int argc, char **argv) {
    vtm_t *m;
    const vtm_tensor *t;
    vtm_status st;
    int token, i;

    if (argc != 4) {
        fprintf(stderr, "usage: dump_emb <file.vtm> <tensor> <token>\n");
        return 2;
    }
    token = atoi(argv[3]);
    m = vtm_load(argv[1]);
    if (!m) {
        fprintf(stderr, "load: %s\n", vtm_last_error());
        return 1;
    }
    vtm_set_position(m, 0);
    vtm_set_token(m, token);
    st = vtm_run(m, NULL, NULL);
    if (st != VTM_OK) {
        fprintf(stderr, "run: %s\n", vtm_error_message(m));
        vtm_free(m);
        return 1;
    }
    t = vtm_tensor_by_name(m, argv[2]);
    if (!t) {
        fprintf(stderr, "нет тензора %s\n", argv[2]);
        vtm_free(m);
        return 3;
    }
    if (!t->f32) {
        fprintf(stderr, "тензор %s без fp32-значений (kind %u)\n",
                argv[2], (unsigned)t->kind);
        vtm_free(m);
        return 4;
    }
    printf("%u\n", (unsigned)t->n_elements);
    for (i = 0; i < (int)t->n_elements; i++)
        printf("%.9e\n", (double)t->f32[i]);
    vtm_free(m);
    return 0;
}
"""


class _CHarness(unittest.TestCase):
    """Сборка маленького C-драйвера поверх публичного API libvtm.a."""

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
                "libvtm.a не собралась (нет компилятора?): "
                + build.stderr.strip()[:300]
            )
        cls._tmp = tempfile.TemporaryDirectory()
        src = os.path.join(cls._tmp.name, "dump_values.c")
        cls.binary = os.path.join(cls._tmp.name, "dump_values")
        with open(src, "w", encoding="utf-8") as handle:
            handle.write(HARNESS_SOURCE)
        compile_cmd = [
            os.environ.get("CC", "cc"), "-std=c99", "-O2",
            "-I", SRC_VTM, "-o", cls.binary, src, VTM_LIB, "-lm",
        ]
        try:
            built = subprocess.run(
                compile_cmd, capture_output=True, text=True, timeout=300,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise unittest.SkipTest(f"C-драйвер не собрался: {exc}")
        if built.returncode != 0:
            raise unittest.SkipTest(
                "C-драйвер не собрался: " + built.stderr.strip()[:300]
            )
        cls.addClassCleanup(cls._tmp.cleanup)

    def dump(self, path, tensor, token):
        proc = subprocess.run(
            [self.binary, path, tensor, str(token)],
            capture_output=True, text=True, timeout=300,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        lines = proc.stdout.split()
        count = int(lines[0])
        values = [float(x) for x in lines[1:]]
        self.assertEqual(len(values), count)
        return values


class TestEmbTritInC(_CHarness, _TmpCase):
    def _prepared(self):
        manifest_path, manifest, _sizes = write_export(self.path("trit"), "trit")
        model = vtm_compile.compile_model(manifest_path)
        model_path = self.path("trit", "model.vtm")
        vtm_ref.write_vtm(model_path, model.tensors, model.program, model.meta)
        entry = _trit_entries(manifest)["model.embed_tokens.weight"]
        with open(self.path("trit", "ternary.trit"), "rb") as handle:
            blob = handle.read()[entry["trit_offset"]:
                                 entry["trit_offset"] + entry["trit_length"]]
        trits, scales, scale_dtype = vtm_ref.decode_trit_segment(blob)
        # Отдельный контейнер только с EMB: в полной модели x — residual-поток,
        # слои его меняют, поэтому выход эмбеддинга оттуда не прочитать.
        emb = vtm_ref.Tensor(
            name="emb.w", kind=vtm_ref.KIND_TRITS, dims=(VOCAB, HIDDEN),
            trits=trits, scales=scales, scale_dtype=scale_dtype,
        )
        x = vtm_ref.Tensor.f32("x", (HIDDEN,), [0.0] * HIDDEN)
        program = [
            vtm_ref.Instr(
                op=int(vtm_ref.Op.EMB), flags=vtm_ref.IFLAG_EMB_INPUT,
                dst=1, a=0, b=0, c=0, imm=0,
            ),
            vtm_ref.Instr(op=int(vtm_ref.Op.HALT)),
        ]
        emb_path = self.path("trit", "emb_only.vtm")
        vtm_ref.write_vtm(emb_path, [emb, x], program, {"embed": "trit"})
        return model, model_path, emb_path, trits, scales

    def test_emb_from_trit_equals_python_dequant(self):
        """EMB из тритового тензора = расквантование в Python (1e-6)."""
        _model, _model_path, emb_path, trits, scales = self._prepared()
        dense = ternary.dequantize(trits, scales, ternary.BLOCK_SIZE)
        for token in (0, 7, 42):
            with self.subTest(token=token):
                got = self.dump(emb_path, "x", token)
                want = dense[token * HIDDEN:(token + 1) * HIDDEN]
                self.assertEqual(len(got), HIDDEN)
                self.assertLessEqual(
                    max(abs(a - b) for a, b in zip(got, want)), 1e-6
                )

    def test_c_logits_match_python_for_tied_trit_head(self):
        """MM с A=emb.w (тритовая таблица) в C = Python-эталон."""
        model, model_path, _emb_path, _trits, _scales = self._prepared()
        tokens = [7]
        want = vtm_ref.execute(model, tokens, 0).logits
        got = self.dump(model_path, "logits", tokens[-1])
        self.assertEqual(len(got), VOCAB)
        self.assertLessEqual(max(abs(a - b) for a, b in zip(got, want)), 1e-4)


if __name__ == "__main__":
    unittest.main()
