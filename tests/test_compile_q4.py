"""Q4 (kind 5) в компиляторе VTM — задача 0154.

Проверяется путь импорта GGUF (0153) -> `tools/vtm_compile.py` ->
контейнер `.vtm` -> Python-эталон/ C-рантайм:

* `decode_q4_segment_ex` из `q4.bin` даёт `Tensor` kind 5, `matrix_tensor`
  принимает его в TMV (и не принимает как плотный);
* `lm_head='q4'` (импорт без tied-эмбеддингов) считает логиты TMV по kind 5;
* tied-модель (так импортируется Qwen2.5-3B) с `lm_head='q4'` по-прежнему
  считает `MM` по таблице словаря — отдельной Q4-матрицы нет;
* транспонированный Q4 отвергается внятной ошибкой (`matrix_tensor` и
  проверка программы), а не тихо пишется в контейнер;
* Q4 в норме/bias и Q4-lm_head при `architecture.lm_head != 'q4'`
  отвергаются с текстом, а не падают где-то в загрузчике.

Крошечная модель пишется тестом (`tests.test_gguf_import.build_tiny_model`),
поэтому сеть и реальный GGUF не нужны. C-рантайм проверяется, только если
бинарь `src/vtm/vtm` уже собран.
"""
from __future__ import annotations

import json
import math
import os
import re
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tests.test_gguf_import import build_tiny_model  # noqa: E402
from tools import gguf_import as gi  # noqa: E402
from tools import vtm_compile, vtm_ref  # noqa: E402

VTM_BIN = os.path.join(ROOT, "src", "vtm", "vtm")
HAS_KIND5 = (
    hasattr(vtm_ref, "KIND_Q4")
    and hasattr(vtm_ref, "decode_q4_segment_ex")
    and hasattr(vtm_ref.Tensor, "q4")
)


class CompileQ4Base(unittest.TestCase):
    """Общая обвязка: временный каталог и импорт крошечной модели в Q4."""

    def setUp(self) -> None:
        if not HAS_KIND5:
            self.skipTest("kind 5 ещё нет в tools/vtm_ref.py (задача 0151)")
        self._tmp = tempfile.TemporaryDirectory(prefix="compile-q4-")
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def import_tiny(self, *, tie: bool = False, **kwargs) -> tuple[Path, Path]:
        source = build_tiny_model(self.tmp / "tiny.gguf", tie=tie, **kwargs)
        out = self.tmp / ("export-tied" if tie else "export")
        gi.import_gguf(source, out, fmt="q4")
        return source, out

    def compile_export(self, out: Path, name: str = "model.vtm") -> Path:
        export = vtm_compile.load_export(out / "manifest.json")
        vtm = vtm_compile.compile_model(export.cfg, export.weights, export.meta)
        path = out / name
        vtm_ref.write_vtm(str(path), vtm.tensors, vtm.program, vtm.meta)
        return path

    def c_binary_ready(self) -> bool:
        if not (os.access(VTM_BIN, os.X_OK) and Path(VTM_BIN).is_file()):
            return False
        # Бинарь должен быть не старше исходников ядер/загрузчика: иначе он
        # может не знать kind 5 и тест упадёт не по делу.
        binary = Path(VTM_BIN).stat().st_mtime
        sources = [Path(ROOT, "src", "vtm", name) for name in
                   ("loader.c", "interp.c", "backend_scalar.c", "backend_neon.c")]
        return all(not path.is_file() or path.stat().st_mtime <= binary
                   for path in sources)

    def run_c(self, path: Path, tokens: str = "1,2") -> subprocess.CompletedProcess:
        return subprocess.run(
            [VTM_BIN, "run", str(path), "--tokens", tokens, "--max-new", "1"],
            capture_output=True, text=True, timeout=300,
        )


class Q4CompileTests(CompileQ4Base):
    def test_non_tied_manifest_compiles_with_q4_lm_head(self) -> None:
        _, out = self.import_tiny(tie=False)
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["architecture"]["lm_head"], "q4")
        self.assertFalse(manifest["architecture"]["tie_word_embeddings"])
        # 0160: импорт пишет Q4_1 (asym) — dtype в манифесте "q4_1".
        self.assertTrue(any(str(entry.get("dtype", "")).startswith("q4")
                            for entry in manifest["tensors"]))

        export = vtm_compile.load_export(out / "manifest.json")
        self.assertEqual(export.cfg.lm_head, "q4")
        self.assertGreater(export.q4_params, 0)
        self.assertEqual(export.trit_params, 0)
        self.assertEqual(export.weights["lm_head"].kind, vtm_ref.KIND_Q4)
        self.assertEqual(export.weights["lm_head"].dims, (export.cfg.vocab,
                                                          export.cfg.hidden))
        for key in ("q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"):
            self.assertEqual(export.weights["layers"][0][key].kind,
                             vtm_ref.KIND_Q4, key)

        vtm = vtm_compile.compile_model(export.cfg, export.weights, export.meta)
        # Q4-матрица lm_head идёт в TMV, не в MM (никакой fp32-копии).
        logits_index = vtm.index_of("logits")
        lm_head_index = vtm.index_of("lm_head.w")
        tmv = [instr for instr in vtm.program
               if int(instr.op) == vtm_ref.Op.TMV
               and int(instr.dst) == logits_index]
        self.assertEqual(len(tmv), 1)
        self.assertEqual(int(tmv[0].a), lm_head_index)
        self.assertEqual(int(tmv[0].flags) & vtm_compile.TMV_FLAG_TRANSPOSED, 0)

        path = out / "model.vtm"
        vtm_ref.write_vtm(str(path), vtm.tensors, vtm.program, vtm.meta)
        loaded = vtm_ref.read_vtm(str(path))
        self.assertTrue(any(t.kind == vtm_ref.KIND_Q4 for t in loaded.tensors))
        self.assertIn("q4", vtm_ref.summary(loaded)["kinds"])

        result = vtm_ref.execute(str(path), tokens=[1, 2], max_new=1)
        self.assertEqual(len(result.logits or []), export.cfg.vocab)
        self.assertTrue(all(math.isfinite(value) for value in result.logits or []))
        self.assertGreaterEqual(len(result.generated), 1)

    def test_tied_model_uses_mm_despite_lm_head_q4(self) -> None:
        # Так импортируется Qwen2.5-3B: отдельного lm_head нет, в манифесте
        # всё равно architecture.lm_head='q4' — компилятор обязан считать MM
        # по emb.w, а не искать несуществующий Q4-тензор.
        _, out = self.import_tiny(tie=True)
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["architecture"]["lm_head"], "q4")
        self.assertTrue(manifest["architecture"]["tie_word_embeddings"])
        self.assertFalse(any(entry.get("role") == "lm_head"
                             for entry in manifest["tensors"]))

        export = vtm_compile.load_export(out / "manifest.json")
        self.assertTrue(export.tied)
        self.assertNotIn("lm_head", export.weights)
        vtm = vtm_compile.compile_model(export.cfg, export.weights, export.meta)
        logits_index = vtm.index_of("logits")
        mm = [instr for instr in vtm.program
              if int(instr.op) == vtm_ref.Op.MM
              and int(instr.dst) == logits_index]
        self.assertEqual(len(mm), 1)
        self.assertEqual(int(mm[0].a), vtm.index_of("emb.w"))

        path = out / "model.vtm"
        vtm_ref.write_vtm(str(path), vtm.tensors, vtm.program, vtm.meta)
        result = vtm_ref.execute(str(path), tokens=[1], max_new=1)
        self.assertEqual(len(result.logits or []), export.cfg.vocab)
        self.assertTrue(all(math.isfinite(value) for value in result.logits or []))

    def test_c_runtime_generates_same_token(self) -> None:
        _, out = self.import_tiny(tie=False, layers=1)
        path = self.compile_export(out)
        expected = vtm_ref.execute(str(path), tokens=[1, 2], max_new=1).generated
        self.assertTrue(expected)
        if not self.c_binary_ready():
            self.skipTest("C-рантайм не собран или старше исходников kind 5")

        proc = self.run_c(path)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        match = re.search(r"вывод:\s*([0-9, ]+)", proc.stdout)
        self.assertIsNotNone(match, proc.stdout)
        c_tokens = [int(value) for value in match.group(1).split()]
        self.assertEqual(c_tokens, expected)

    def test_kv_cache_and_q4_coexist(self) -> None:
        # KV-кэши создаёт читатель, Q4-матрицы лежат в файле: оба вида должны
        # уживаться в одной таблице тензоров и в одной арене.
        _, out = self.import_tiny(tie=False, layers=1)
        path = self.compile_export(out)
        loaded = vtm_ref.read_vtm(str(path))
        self.assertTrue(any(t.is_kv_cache for t in loaded.tensors))
        self.assertTrue(any(t.kind == vtm_ref.KIND_Q4 for t in loaded.tensors))


class Q4RefusalTests(CompileQ4Base):
    def test_transposed_q4_matrix_tensor_refused(self) -> None:
        q4 = vtm_ref.Tensor.q4(
            "w", (4, 8), [8] * 32, [1.0],
        )
        graph = vtm_compile.Graph(
            vtm_compile.ModelConfig(hidden=8, n_heads=1, n_kv_heads=1,
                                    head_dim=8, intermediate=8, vocab=8,
                                    n_layers=1, context_len=4)
        )
        with self.assertRaisesRegex(ValueError, "(?i)транспонир"):
            vtm_compile.matrix_tensor(graph, "w", 4, 8, q4, transposed=True)
        self.assertFalse(graph.tensors, "тензор не должен попасть в граф")

    def test_transposed_q4_program_refused(self) -> None:
        _, out = self.import_tiny(tie=False, layers=1)
        export = vtm_compile.load_export(out / "manifest.json")
        vtm = vtm_compile.compile_model(export.cfg, export.weights, export.meta)
        # Instr в Python-эталоне отвергает неизвестные флаги (0x04 — C-шный
        # VTM_FLAG_TMV_TRANSPOSED), поэтому граф «извне» описываем лёгкой
        # записью: компилятор обязан отсечь его до записи контейнера.
        tmv = next(instr for instr in vtm.program
                   if int(instr.op) == vtm_ref.Op.TMV)
        program = [types.SimpleNamespace(
            op=tmv.op, flags=tmv.flags | vtm_compile.TMV_FLAG_TRANSPOSED,
            a=tmv.a, dst=tmv.dst,
        )]
        with self.assertRaisesRegex(ValueError, "Q4.*TRANSPOSED"):
            vtm_compile.check_q4_tmv_support(vtm.tensors, program)

    def test_q4_dense_matrix_refused(self) -> None:
        q4 = vtm_ref.Tensor.q4("w", (4, 8), [8] * 32, [1.0])
        graph = vtm_compile.Graph(vtm_compile.ModelConfig())
        with self.assertRaisesRegex(ValueError, "плотный"):
            vtm_compile.matrix_tensor(graph, "w", 4, 8, q4, dense=True)

    def test_q4_norm_entry_refused(self) -> None:
        _, out = self.import_tiny(tie=False, layers=1)
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        entry = dict(next(item for item in manifest["tensors"]
                          if str(item.get("dtype", "")).startswith("q4")))
        entry["name"] = "model.layers.0.input_layernorm.weight"
        manifest["tensors"].append(entry)
        path = out / "manifest-norm.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        # Файлы q4.bin/non_ternary.bin ищутся рядом с манифестом — каталог
        # тот же, поэтому достаточно нового JSON.
        with self.assertRaisesRegex(ValueError, "Q4 допустим только"):
            vtm_compile.load_export(path)

    def test_q4_lm_head_requires_architecture_q4(self) -> None:
        _, out = self.import_tiny(tie=False, layers=1)
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        manifest["architecture"]["lm_head"] = "mm"
        path = out / "manifest-mm.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "lm_head='mm'.*Q4"):
            vtm_compile.load_export(path)

    def test_dense_lm_head_with_architecture_q4_refused(self) -> None:
        # Обратная несовместимость: architecture.lm_head='q4', а в
        # non_ternary.bin лежит плотный lm_head (например, манифест
        # пересобран без --format q4).
        _, out = self.import_tiny(tie=False, layers=1)
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        manifest["tensors"] = [entry for entry in manifest["tensors"]
                               if entry.get("role") != "lm_head"]
        arch = manifest["architecture"]
        n_values = int(arch["vocab"]) * int(arch["hidden"])
        non_path = out / "non_ternary.bin"
        data = bytearray(non_path.read_bytes())
        offset = len(data)
        data += struct.pack(f"<{n_values}f", *([0.0] * n_values))
        non_path.write_bytes(bytes(data))
        manifest["files"]["non_ternary.bin"].pop("sha256", None)
        manifest["files"]["non_ternary.bin"]["bytes"] = len(data)
        manifest["non_ternary"]["tensors"].append({
            "name": "lm_head.weight",
            "role": "lm_head",
            "layer": None,
            "key": "lm_head",
            "shape": [int(arch["vocab"]), int(arch["hidden"])],
            "dtype": "float32",
            "offset": offset,
            "length": n_values * 4,
        })
        path = out / "manifest-dense.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "lm_head='q4'.*плотный"):
            vtm_compile.load_export(path)

    def test_q4_block_size_mismatch_in_tensor(self) -> None:
        # Q4 хранит масштаб только на 32 значения; чужой блок должен падать
        # в Tensor, а не в ядре.
        with self.assertRaisesRegex(ValueError, "Q4 хранит масштаб"):
            vtm_ref.Tensor(
                name="w", kind=vtm_ref.KIND_Q4, dims=(4, 8),
                qdata=[8] * 32, scales=[1.0, 1.0], block_size=64,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
