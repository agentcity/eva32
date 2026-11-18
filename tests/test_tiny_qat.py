"""Крошечная Qwen2: контракты сквозной цепочки QAT → .trit → .vtm → C.

Задача 0095. Тесты намеренно не импортируют torch/transformers: обучение
идёт отдельно (``scripts/tiny_qat.py``), а здесь проверяется то, что можно
проверить без GPU-стека:

* ``architecture_dict()`` принимается ``qat.export.infer_architecture`` и даёт
  ровно те размерности, с которыми обучался чекпоинт (4 009 984 параметра);
* обрезка пары до 512 токенов сохраняет хвост ответа и eos, а метки промпта
  остаются -100 (на этом держится loss);
* разбор вывода ``vtm run`` и сверка логитов считают те же числа, что
  попадают в ``runs/tiny/parity.json``;
* если артефакты прогона на месте (``runs/tiny/`` в git не входит), то
  C-проба на ``src/vtm/libvtm.a`` заново снимает логиты, а тест сверяет их
  с HF-эталоном из ``parity.json`` и с порогом 1e-4. Без артефактов эти
  проверки пропускаются, а не падают.

Запуск::

    python3 -m unittest tests.test_tiny_qat -v
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPT = ROOT / "scripts" / "tiny_qat.py"
VTM_BIN = ROOT / "src" / "vtm" / "vtm"
VTM_LIB = ROOT / "src" / "vtm" / "libvtm.a"
OUT = ROOT / "runs" / "tiny"


def _load_tiny():
    """Загрузить scripts/tiny_qat.py как модуль (scripts/ — не пакет)."""
    spec = importlib.util.spec_from_file_location("tiny_qat_script", SCRIPT)
    if spec is None or spec.loader is None:  # pragma: no cover - файл на месте
        raise RuntimeError(f"не загружается {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tiny = _load_tiny()


class FakeTokenizer:
    """Байтовый токенизатор-заглушка: один символ → один id (в словаре)."""

    def __call__(self, text, add_special_tokens=False):  # noqa: ARG002 - API HF
        return SimpleNamespace(
            input_ids=[ord(char) % tiny.VOCAB_SIZE for char in str(text)]
        )


def _state_shapes(arch):
    """Формы state_dict крошечной Qwen2 по HF-подобному конфигу."""
    hidden = arch["hidden_size"]
    vocab = arch["vocab_size"]
    intermediate = arch["intermediate_size"]
    q_dim = arch["num_attention_heads"] * arch["head_dim"]
    kv_dim = arch["num_key_value_heads"] * arch["head_dim"]
    shapes = {
        "model.embed_tokens.weight": (vocab, hidden),
        "lm_head.weight": (vocab, hidden),
        "model.norm.weight": (hidden,),
    }
    for layer in range(arch["num_hidden_layers"]):
        prefix = f"model.layers.{layer}"
        shapes.update({
            f"{prefix}.self_attn.q_proj.weight": (q_dim, hidden),
            f"{prefix}.self_attn.k_proj.weight": (kv_dim, hidden),
            f"{prefix}.self_attn.v_proj.weight": (kv_dim, hidden),
            f"{prefix}.self_attn.o_proj.weight": (hidden, q_dim),
            f"{prefix}.mlp.gate_proj.weight": (intermediate, hidden),
            f"{prefix}.mlp.up_proj.weight": (intermediate, hidden),
            f"{prefix}.mlp.down_proj.weight": (hidden, intermediate),
            f"{prefix}.input_layernorm.weight": (hidden,),
            f"{prefix}.post_attention_layernorm.weight": (hidden,),
            f"{prefix}.self_attn.q_proj.bias": (q_dim,),
            f"{prefix}.self_attn.k_proj.bias": (kv_dim,),
            f"{prefix}.self_attn.v_proj.bias": (kv_dim,),
        })
    return shapes


def _count_params(arch):
    """Параметры модели по конфигу — без torch (сверка с задачей 0095)."""
    hidden = arch["hidden_size"]
    vocab = arch["vocab_size"]
    intermediate = arch["intermediate_size"]
    q_dim = arch["num_attention_heads"] * arch["head_dim"]
    kv_dim = arch["num_key_value_heads"] * arch["head_dim"]
    per_layer = (
        q_dim * hidden + q_dim            # q_proj + bias
        + kv_dim * hidden + kv_dim        # k_proj + bias
        + kv_dim * hidden + kv_dim        # v_proj + bias
        + hidden * q_dim                  # o_proj
        + 3 * intermediate * hidden       # gate/up/down
        + 2 * hidden                      # norm1, norm2
    )
    return vocab * hidden + arch["num_hidden_layers"] * per_layer + hidden


class ArchitectureContractTest(unittest.TestCase):
    """run.json["architecture"] — мост между обучением и qat.export."""

    def test_export_infers_architecture_without_transformers(self):
        from qat.export import infer_architecture

        arch = tiny.architecture_dict()
        result = infer_architecture(_state_shapes(arch), arch)
        self.assertEqual(result["hidden"], 256)
        self.assertEqual(result["n_heads"], 4)
        self.assertEqual(result["n_kv_heads"], 2)
        self.assertEqual(result["head_dim"], 64)
        self.assertEqual(result["intermediate"], 1024)
        self.assertEqual(result["vocab"], 287)
        self.assertEqual(result["n_layers"], 4)
        self.assertEqual(result["context_len"], 1024)
        self.assertTrue(result["tie_word_embeddings"])
        self.assertEqual(result["lm_head"], "mm")
        self.assertEqual(result["embed"], "none")
        self.assertAlmostEqual(result["rope_theta"], 10000.0)
        self.assertAlmostEqual(result["rms_norm_eps"], 1e-6)

    def test_architecture_matches_task_param_count(self):
        arch = tiny.architecture_dict()
        self.assertEqual(tiny.EXPECTED_PARAMS, 4_009_984)
        self.assertEqual(_count_params(arch), tiny.EXPECTED_PARAMS)

    def test_vocab_and_special_ids_are_inside_the_table(self):
        self.assertEqual(tiny.VOCAB_SIZE, 287)
        for token_id in (tiny.PAD_ID, tiny.EOS_ID):
            self.assertGreaterEqual(token_id, 0)
            self.assertLess(token_id, tiny.VOCAB_SIZE)


class EncodePairTest(unittest.TestCase):
    """Обрезка 512 токенов: ответ должен выжить, промпт — в -100."""

    def setUp(self):
        self.tokenizer = FakeTokenizer()

    def test_truncates_to_max_seq_len_and_keeps_answer_tail(self):
        sample = tiny.encode_pair(
            self.tokenizer, "q" * 400, "a" * 5000,
            max_seq_len=512, min_answer_tokens=32,
        )
        self.assertIsNotNone(sample)
        ids = sample["input_ids"]
        labels = sample["labels"]
        self.assertEqual(len(ids), 512)
        self.assertEqual(len(labels), len(ids))
        self.assertEqual(ids[-1], tiny.EOS_ID)
        answer_labels = [label for label in labels if label != -100]
        self.assertEqual(len(answer_labels), sample["answer_tokens"])
        self.assertGreaterEqual(sample["answer_tokens"], 32)
        # Промпт — строго начало, ответ — строго хвост.
        prompt_length = len(ids) - sample["answer_tokens"]
        self.assertTrue(all(label == -100 for label in labels[:prompt_length]))
        self.assertTrue(all(label != -100 for label in labels[prompt_length:]))
        self.assertLess(max(ids), tiny.VOCAB_SIZE)

    def test_short_answer_is_dropped(self):
        # Промпт съедает бюджет, на ответ остаётся меньше минимума.
        self.assertIsNone(
            tiny.encode_pair(
                self.tokenizer, "q" * 400, "a",
                max_seq_len=512, min_answer_tokens=32,
            )
        )
        self.assertIsNone(
            tiny.encode_pair(self.tokenizer, "q", "", max_seq_len=512)
        )

    def test_prompt_contains_question_text(self):
        self.assertIn("Вопрос:", tiny.prompt_text("Что такое VTM?"))
        self.assertIn("Ответ:", tiny.prompt_text("Что такое VTM?"))


class CompareLogitsTest(unittest.TestCase):
    """max|Δ| и argmax считаются так же, как в отчёте."""

    def test_identical_vectors(self):
        result = tiny.compare_logits([0.5, -1.0, 2.0], [0.5, -1.0, 2.0])
        self.assertEqual(result["max_abs_delta"], 0.0)
        self.assertTrue(result["argmax_match"])
        self.assertEqual(result["argmax_hf"], 2)
        self.assertEqual(result["top5_overlap"], 3)

    def test_known_delta_and_top5(self):
        result = tiny.compare_logits([1.0, 2.0, 3.0], [1.0, 2.0, 2.5])
        self.assertAlmostEqual(result["max_abs_delta"], 0.5)
        self.assertAlmostEqual(result["mean_abs_delta"], 0.5 / 3)
        self.assertEqual(result["argmax_hf"], 2)
        self.assertEqual(result["top5_overlap"], 3)

    def test_length_mismatch_is_an_error(self):
        with self.assertRaises(tiny.TinyError):
            tiny.compare_logits([1.0, 2.0], [1.0])


class ParseVtmRunTest(unittest.TestCase):
    """Разбор штатного вывода ``vtm run`` (тот же, что в отчёте)."""

    SAMPLE = (
        "модель: runs/tiny/model.vtm\n"
        "тензоров: 107, инструкций: 86, бэкенд: simd-neon\n"
        "prompt: 208 176 32\n"
        "вывод:  1 2 3\n"
        "время: 12.5 мс, 3 токен(ов), 240.0 ток/с\n"
    )

    def test_parses_prompt_generated_and_counters(self):
        parsed = tiny.parse_vtm_run_output(self.SAMPLE)
        self.assertEqual(parsed["prompt"], [208, 176, 32])
        self.assertEqual(parsed["generated"], [1, 2, 3])
        self.assertEqual(parsed["tensors"], 107)
        self.assertEqual(parsed["instructions"], 86)
        self.assertEqual(parsed["backend"], "simd-neon")

    def test_empty_generation_is_allowed(self):
        parsed = tiny.parse_vtm_run_output("вывод:  (пусто)\n")
        self.assertEqual(parsed["generated"], [])


class DataFilesTest(unittest.TestCase):
    """Пара question/answer из data/ читается без torch."""

    def test_read_pairs_on_val_file(self):
        rows, stats = tiny.read_pairs(tiny.VAL_PATH, 3)
        self.assertEqual(len(rows), 3)
        self.assertEqual(stats["pairs"], 3)
        for row in rows:
            self.assertTrue(row["question"])
            self.assertTrue(row["answer"])


@unittest.skipUnless(
    (OUT / "parity.json").is_file() and (OUT / "model.vtm").is_file(),
    "нет артефактов runs/tiny (обучение не прогонялось в этом чекауте)",
)
class ArtifactParityTest(unittest.TestCase):
    """C-проба заново снимает логиты и сверяется с HF-эталоном из parity.json."""

    @classmethod
    def setUpClass(cls):
        cls.parity = json.loads((OUT / "parity.json").read_text(encoding="utf-8"))

    def test_recorded_numbers_are_consistent(self):
        parity = self.parity
        self.assertEqual(len(parity["hf_logits"]), tiny.VOCAB_SIZE)
        self.assertEqual(len(parity["c_logits"]), tiny.VOCAB_SIZE)
        comparison = tiny.compare_logits(parity["hf_logits"], parity["c_logits"])
        self.assertAlmostEqual(
            comparison["max_abs_delta"], parity["max_abs_delta"], places=12
        )
        self.assertEqual(comparison["argmax_hf"], parity["hf_argmax"])
        self.assertEqual(comparison["argmax_c"], parity["c_argmax"])
        self.assertTrue(parity["argmax_match"])
        self.assertLessEqual(parity["max_abs_delta"], parity["threshold"])

    def test_c_probe_reproduces_hf_logits(self):
        if not VTM_LIB.is_file():
            self.skipTest("src/vtm/libvtm.a не собран")
        from tools import parity_real  # noqa: PLC0415 - импорт внутри теста

        compiler = parity_real.find_c_compiler()
        if compiler is None:
            self.skipTest("нет C-компилятора (cc/clang/gcc)")
        probe = parity_real.run_c_probe(
            OUT, OUT / "model.vtm", self.parity["token_ids"], compiler
        )
        self.assertEqual(len(probe.logits), tiny.VOCAB_SIZE)
        comparison = tiny.compare_logits(self.parity["hf_logits"], probe.logits)
        self.assertTrue(
            comparison["argmax_match"],
            f"argmax разошёлся: HF {comparison['argmax_hf']}, "
            f"C {comparison['argmax_c']}",
        )
        self.assertLessEqual(
            comparison["max_abs_delta"], self.parity["threshold"],
            f"max|Δ| {comparison['max_abs_delta']:.3e} больше порога "
            f"{self.parity['threshold']:g}",
        )

    def test_c_cli_loads_container_and_generates(self):
        if not VTM_BIN.is_file() or not os.access(VTM_BIN, os.X_OK):
            self.skipTest("src/vtm/vtm не собран")
        tokens = ",".join(str(token) for token in self.parity["token_ids"][:8])
        proc = subprocess.run(
            [str(VTM_BIN), "run", str(OUT / "model.vtm"),
             "--tokens", tokens, "--max-new", "2"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        parsed = tiny.parse_vtm_run_output(proc.stdout)
        self.assertEqual(parsed["prompt"], self.parity["token_ids"][:8])
        self.assertEqual(len(parsed["generated"]), 2)
        self.assertIsNotNone(parsed["tensors"])


@unittest.skipUnless(
    (OUT / "report.json").is_file(),
    "нет runs/tiny/report.json (обучение не прогонялось в этом чекауте)",
)
class ArtifactReportTest(unittest.TestCase):
    """Сводка прогона: loss упал, .vtm собран, max|Δ| измерен."""

    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))

    def test_loss_went_down(self):
        summary = self.report["summary"]
        self.assertIn("val_loss_before", summary)
        self.assertLess(summary["val_loss_after"], summary["val_loss_before"])
        self.assertGreater(summary.get("steps", 0), 0)

    def test_vtm_built_and_parity_measured(self):
        summary = self.report["summary"]
        self.assertGreater(summary.get("vtm_bytes", 0), 0)
        self.assertIn("max_abs_delta", summary)
        self.assertTrue(summary.get("parity_passed"))

    def test_train_metrics_match_run_json(self):
        metrics = json.loads((OUT / "train_metrics.json").read_text(encoding="utf-8"))
        run_meta = json.loads((OUT / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics["params"], tiny.EXPECTED_PARAMS)
        self.assertEqual(run_meta["model"]["vocab_size"], tiny.VOCAB_SIZE)
        self.assertEqual(run_meta["architecture"]["num_hidden_layers"], 4)
        self.assertEqual(metrics["ternary_modules"], 28)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
