"""Оценка памяти QAT (задача 0047): формула, таблица и --dry-run без torch.

Числа проверяются точно (байты), а не «примерно»: тест ловит случайное
изменение любого компонента формулы. Модуль не импортирует torch и не
требует pyyaml: конфиг читается встроенным парсером, если pyyaml нет.
"""
import importlib.util
import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qat.config import (  # noqa: E402
    Config,
    ConfigError,
    DataConfig,
    ModelConfig,
    QuantConfig,
    TrainConfig,
    _parse_simple_yaml,
    estimate_memory,
    format_memory_matrix,
    format_memory_table,
    load_config,
    memory_matrix,
    validate_config,
)

MODEL_1_5B = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_3B = "Qwen/Qwen2.5-3B-Instruct"


def make_config(
    name: str = MODEL_3B,
    master: str = "bf16",
    optimizer: str = "adam8bit",
    batch: int = 2,
    seq_len: int = 2048,
    grad_dtype: str = "bf16",
    checkpointing: bool = True,
) -> Config:
    return Config(
        model=ModelConfig(name_or_path=name, master_dtype=master),
        quant=QuantConfig(),
        data=DataConfig(max_seq_len=seq_len),
        train=TrainConfig(
            micro_batch_size=batch,
            grad_accum_steps=16,
            optimizer=optimizer,
            grad_dtype=grad_dtype,
            gradient_checkpointing=checkpointing,
        ),
    )


#: Три конфигурации с посчитанными вручную байтами. Это контрольные числа:
#: 3B/bf16/Adam8bit — целевой режим (влезает), 3B/fp32 — исходная проблема,
#: 1.5B/fp32/fp32-градиенты — конфиг qwen1.5b.yaml.
CASES = (
    (
        "3B bf16 adam8bit (qwen3b.yaml)",
        make_config(),
        {
            "master": 6_171_873_280,
            "working_copy": 0,
            "optimizer": 6_171_873_280,
            "gradients": 6_171_873_280,
            "activations": 2_884_632_576,
        },
        21_400_252_416,
        19.931,
        True,
    ),
    (
        "3B fp32 adam8bit (без bf16-мастера)",
        make_config(master="fp32"),
        {
            "master": 12_343_746_560,
            "working_copy": 6_171_873_280,
            "optimizer": 6_171_873_280,
            "gradients": 6_171_873_280,
            "activations": 2_884_632_576,
        },
        33_743_998_976,
        31.427,
        False,
    ),
    (
        "1.5B fp32 adam8bit (qwen1.5b.yaml, batch 4)",
        make_config(
            name=MODEL_1_5B, master="fp32", batch=4, grad_dtype="fp32"
        ),
        {
            "master": 6_174_851_072,
            "working_copy": 3_087_425_536,
            "optimizer": 3_087_425_536,
            "gradients": 6_174_851_072,
            "activations": 4_368_367_616,
        },
        22_892_920_832,
        21.321,
        True,
    ),
)


class TestFormulaOnKnownConfigs(unittest.TestCase):
    def test_components_are_exact(self):
        for label, cfg, components, total, total_gib, fits in CASES:
            with self.subTest(label=label):
                estimate = estimate_memory(cfg)
                self.assertEqual(estimate["components"], components)
                self.assertEqual(sum(components.values()), total)
                self.assertEqual(estimate["total_bytes"], total)
                self.assertEqual(estimate["total_gib"], total_gib)
                self.assertEqual(estimate["fits_24gb"], fits)

    def test_3b_bf16_adam8bit_fits_24_gib(self):
        estimate = estimate_memory(make_config())
        self.assertLessEqual(estimate["total_bytes"], 24 * 1024**3)
        self.assertEqual(estimate["total_gib"], 19.931)
        self.assertEqual(estimate["margin_gib"], 4.069)

    def test_3b_fp32_does_not_fit(self):
        self.assertFalse(estimate_memory(make_config(master="fp32"))["fits_24gb"])

    def test_master_dtype_saves_master_and_copy(self):
        bf16 = estimate_memory(make_config(master="bf16"))
        fp32 = estimate_memory(make_config(master="fp32"))
        # 3B: 2 vs 4 байта на параметр плюс отдельная bf16-копия.
        self.assertEqual(fp32["components"]["master"] - bf16["components"]["master"],
                         3_085_936_640 * 2)
        self.assertEqual(fp32["components"]["working_copy"], 3_085_936_640 * 2)
        self.assertEqual(bf16["components"]["working_copy"], 0)
        self.assertEqual(fp32["total_bytes"] - bf16["total_bytes"],
                         3_085_936_640 * 4)  # 2 байта мастера + 2 байта копии

    def test_grad_dtype_halves_gradients(self):
        fp32 = estimate_memory(make_config(grad_dtype="fp32"))
        bf16 = estimate_memory(make_config(grad_dtype="bf16"))
        self.assertEqual(fp32["components"]["gradients"],
                         3_085_936_640 * 4)
        self.assertEqual(bf16["components"]["gradients"],
                         3_085_936_640 * 2)

    def test_adamw_uses_fp32_states(self):
        adamw = estimate_memory(make_config(optimizer="adamw"))
        self.assertEqual(adamw["components"]["optimizer"], 3_085_936_640 * 8)
        self.assertFalse(adamw["fits_24gb"])

    def test_checkpointing_is_required_for_3b_batch4(self):
        # При batch 2 запас есть и без checkpointing; при batch 4 — уже нет.
        with_ckpt = estimate_memory(make_config(batch=4, checkpointing=True))
        without = estimate_memory(make_config(batch=4, checkpointing=False))
        self.assertTrue(with_ckpt["fits_24gb"])
        self.assertFalse(without["fits_24gb"])
        self.assertGreater(without["components"]["activations"],
                           with_ckpt["components"]["activations"])


class TestConfigFiles(unittest.TestCase):
    def test_simple_yaml_fallback_reads_qwen3b(self):
        # --dry-run должен работать без pyyaml: проверяем парсер напрямую,
        # иначе на машине с pyyaml эта ветка не исполняется.
        path = os.path.join(ROOT, "qat", "configs", "qwen3b.yaml")
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        raw = _parse_simple_yaml(text, "qwen3b.yaml")
        self.assertEqual(raw["model"]["master_dtype"], "bf16")
        self.assertEqual(raw["model"]["num_params"], 3_085_936_640)
        self.assertEqual(raw["train"]["optimizer"], "adam8bit")
        self.assertEqual(raw["train"]["grad_dtype"], "bf16")
        self.assertEqual(raw["train"]["betas"], (0.9, 0.95))
        self.assertEqual(raw["quant"]["target_suffixes"][0], "q_proj")
        self.assertEqual(len(raw["quant"]["target_suffixes"]), 7)

    def test_qwen3b_yaml_is_bf16_8bit(self):
        cfg = load_config(os.path.join(ROOT, "qat", "configs", "qwen3b.yaml"))
        self.assertEqual(cfg.model.name_or_path, MODEL_3B)
        self.assertEqual(cfg.model.master_dtype, "bf16")
        self.assertEqual(cfg.model.torch_dtype, "bfloat16")
        self.assertEqual(cfg.train.optimizer, "adam8bit")
        self.assertEqual(cfg.train.grad_dtype, "bf16")
        self.assertTrue(cfg.train.gradient_checkpointing)
        estimate = estimate_memory(cfg)
        self.assertTrue(estimate["fits_24gb"])
        self.assertLessEqual(estimate["total_gib"], 24.0)

    def test_qwen1_5b_yaml_still_fp32(self):
        cfg = load_config(os.path.join(ROOT, "qat", "configs", "qwen1.5b.yaml"))
        self.assertEqual(cfg.model.master_dtype, "fp32")
        self.assertEqual(cfg.model.torch_dtype, "float32")
        self.assertEqual(cfg.train.grad_dtype, "fp32")

    def test_master_dtype_validation(self):
        cfg = Config(model=ModelConfig(master_dtype="bf16"))
        validate_config(cfg)
        self.assertEqual(cfg.model.torch_dtype, "bfloat16")
        with self.assertRaises(ConfigError):
            validate_config(Config(model=ModelConfig(master_dtype="int4")))
        with self.assertRaises(ConfigError):
            validate_config(Config(train=TrainConfig(grad_dtype="int8")))


class TestTable(unittest.TestCase):
    def test_matrix_has_all_combinations(self):
        rows = memory_matrix(make_config())
        self.assertEqual(len(rows), 8)  # 2 модели × 2 мастера × 2 оптимизатора
        combos = {(row["label"], row["master_dtype"], row["optimizer"]) for row in rows}
        self.assertEqual(len(combos), 8)
        self.assertIn(("3B", "bf16", "adam8bit"), combos)
        self.assertIn(("1.5B", "fp32", "adamw"), combos)

    def test_matrix_text_marks_fit(self):
        text = format_memory_matrix(make_config())
        self.assertIn("матрица памяти", text)
        self.assertIn("3B", text)
        self.assertIn("adam8bit", text)
        self.assertIn("укладываются в 24 ГиБ", text)
        # Именно целевая строка должна быть помечена «да».
        target = [line for line in text.splitlines()
                  if line.startswith("3B") and "bf16" in line and "adam8bit" in line]
        self.assertEqual(len(target), 1)
        self.assertIn("да", target[0])

    def test_single_table_has_all_components(self):
        estimate = estimate_memory(make_config())
        text = format_memory_table(estimate)
        self.assertEqual(estimate["table"], text)
        for label in ("мастер", "копия", "опт.сост", "градиенты", "активации", "итого"):
            self.assertIn(label, text)
        self.assertIn("19.93", text)
        self.assertIn("влезает", text)


def _module_missing(name):
    try:
        return importlib.util.find_spec(name) is None
    except (ImportError, ValueError):
        return True


class TestDryRunWithoutTorch(unittest.TestCase):
    @unittest.skipUnless(
        _module_missing("torch") or _module_missing("transformers"),
        "на машине с GPU-стеком --dry-run дополнительно делает 3 шага",
    )
    def test_cli_prints_table_and_exits_zero(self):
        for config in ("qwen3b.yaml", "qwen1.5b.yaml"):
            with self.subTest(config=config):
                proc = subprocess.run(
                    [sys.executable, "-m", "qat.train",
                     "--config", os.path.join("qat", "configs", config),
                     "--dry-run"],
                    cwd=ROOT, capture_output=True, text=True, timeout=120,
                )
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertNotIn("Traceback", proc.stderr)
                self.assertIn("матрица памяти", proc.stdout)
                self.assertIn("3B", proc.stdout)
                self.assertIn("1.5B", proc.stdout)
                self.assertIn("adam8bit", proc.stdout)
                self.assertIn("adamw", proc.stdout)
                self.assertIn("bf16", proc.stdout)
                self.assertIn("fp32", proc.stdout)


if __name__ == "__main__":
    unittest.main()
