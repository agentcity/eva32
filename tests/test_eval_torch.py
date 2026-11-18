"""Проверки ``tools/eval_torch.py`` без torch.

Покрыто: парсер аргументов, чтение и валидация golden, отложенные пары,
промпты (как в ``eval/run_eval.py`` и ``qat/data.py``), сводка, отчёт,
dry-run, понятная ошибка без torch и запись JSON. Тяжёлый прогон модели
подменяется заглушкой: ни torch, ни сети, ни GPU в тестах нет.

Запуск::

    python3 -m unittest tests.test_eval_torch -v
"""
from __future__ import annotations

import importlib.util
import io
import json
import math
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GOLDEN = ROOT / "eval" / "golden.jsonl"
TOOL = ROOT / "tools" / "eval_torch.py"


def _load_tool():
    """Загрузить tools/eval_torch.py по пути (tools не пакет)."""
    spec = importlib.util.spec_from_file_location("eval_torch_tool", TOOL)
    if spec is None or spec.loader is None:  # pragma: no cover - защита от кривого пути
        raise RuntimeError(f"не удалось загрузить {TOOL}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


eval_torch = _load_tool()


# ---------------------------------------------------------------------------
# заготовки без torch
# ---------------------------------------------------------------------------


class FakeTokenizer:
    """Минимальный токенизатор: chat_template и токены по заготовке."""

    def __init__(self, *, chat_template="template", prompt_ids=(1, 2, 3),
                 answer_ids=(5, 6), eos_token_id=9, pad_token_id=0):
        self.chat_template = chat_template
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.prompt_ids = list(prompt_ids)
        self.answer_ids = list(answer_ids)
        self.padding_side = "right"
        self.truncation_side = "right"
        self.calls = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt,
                            return_dict=False):
        self.calls.append({"messages": messages, "tokenize": tokenize})
        if tokenize:
            return list(self.prompt_ids)
        rendered = " | ".join(f"{m['role']}: {m['content']}" for m in messages)
        return f"<|prompt|>{rendered}<|assistant|>"

    def __call__(self, text, add_special_tokens=False, **kwargs):
        return SimpleNamespace(input_ids=list(self.answer_ids))


class NoTemplateTokenizer(FakeTokenizer):
    """Базовая модель без chat-шаблона: проверяем запасной текст."""

    def __init__(self):
        super().__init__(chat_template=None)


def _write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _valid_row(row_id="q1", **overrides):
    row = {
        "id": row_id,
        "question": "Сколько стоит тариф «Старт»?",
        "checks": {"must": ["990"]},
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestParser(unittest.TestCase):
    def test_defaults(self):
        args = eval_torch.build_parser().parse_args([])
        self.assertEqual(args.model, eval_torch.DEFAULT_MODEL)
        self.assertEqual(args.golden, str(eval_torch.DEFAULT_GOLDEN))
        self.assertIsNone(args.ppl_data)
        self.assertEqual(args.out, "runs/eval.json")
        self.assertEqual(args.limit, 0)
        self.assertEqual(args.ppl_limit, eval_torch.DEFAULT_PPL_LIMIT)
        self.assertEqual(args.max_new_tokens, eval_torch.DEFAULT_MAX_NEW_TOKENS)
        self.assertEqual(args.batch_size, eval_torch.DEFAULT_BATCH_SIZE)
        self.assertEqual(args.device, "auto")
        self.assertFalse(args.dry_run)
        self.assertFalse(args.local_files_only)

    def test_custom_flags(self):
        args = eval_torch.build_parser().parse_args([
            "--model", "runs/pilot1",
            "--golden", "x.jsonl",
            "--ppl-data", "pairs.jsonl",
            "--out", "runs/e.json",
            "--limit", "5",
            "--ppl-limit", "50",
            "--max-new-tokens", "64",
            "--batch-size", "2",
            "--max-seq-len", "512",
            "--device", "cuda:0",
            "--local-files-only",
            "--dry-run",
        ])
        self.assertEqual(args.model, "runs/pilot1")
        self.assertEqual(args.limit, 5)
        self.assertEqual(args.ppl_limit, 50)
        self.assertEqual(args.max_new_tokens, 64)
        self.assertEqual(args.batch_size, 2)
        self.assertEqual(args.max_seq_len, 512)
        self.assertEqual(args.device, "cuda:0")
        self.assertTrue(args.local_files_only)
        self.assertTrue(args.dry_run)

    def test_rejects_non_positive_and_negative(self):
        with redirect_stderr(io.StringIO()):
            for value in ("0", "-1", "abc"):
                with self.subTest(flag="--max-new-tokens", value=value):
                    with self.assertRaises(SystemExit):
                        eval_torch.build_parser().parse_args(["--max-new-tokens", value])
                with self.subTest(flag="--batch-size", value=value):
                    with self.assertRaises(SystemExit):
                        eval_torch.build_parser().parse_args(["--batch-size", value])
            for value in ("-1", "abc"):
                with self.subTest(flag="--limit", value=value):
                    with self.assertRaises(SystemExit):
                        eval_torch.build_parser().parse_args(["--limit", value])

    def test_ppl_limit_zero_means_all(self):
        args = eval_torch.build_parser().parse_args(["--ppl-limit", "0"])
        self.assertEqual(args.ppl_limit, 0)


# ---------------------------------------------------------------------------
# golden
# ---------------------------------------------------------------------------


class TestGoldenSet(unittest.TestCase):
    def test_real_golden_loads(self):
        rows = eval_torch.load_golden(GOLDEN)
        self.assertEqual(len(rows), 36)
        ids = [row["id"] for row in rows]
        self.assertEqual(len(ids), len(set(ids)))
        for row in rows:
            self.assertTrue(row["question"].strip())
            self.assertIsInstance(row["checks"], dict)
            self.assertTrue(row["checks"])
        self.assertIn("greet", ids)
        self.assertIn("price-start", ids)
        self.assertEqual(eval_torch.unknown_check_keys(rows), [])

    def test_rule_counts(self):
        rows = eval_torch.load_golden(GOLDEN)
        counts = eval_torch.check_rule_counts(rows)
        self.assertEqual(set(counts), set(eval_torch.KNOWN_CHECKS))
        self.assertGreater(counts["must"], 0)
        self.assertGreater(counts["no_prompt_shingle"], 0)

    def test_comments_and_blank_lines_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "golden.jsonl"
            path.write_text(
                "# комментарий\n\n" + json.dumps(_valid_row(), ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            rows = eval_torch.load_golden(path)
            self.assertEqual(len(rows), 1)

    def test_unknown_check_key_is_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "golden.jsonl"
            _write_jsonl(path, [_valid_row(checks={"new_rule": True})])
            rows = eval_torch.load_golden(path)
            self.assertEqual(eval_torch.unknown_check_keys(rows), ["new_rule"])

    def test_rejects_broken_json_with_line_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "golden.jsonl"
            path.write_text("{не json}\n", encoding="utf-8")
            with self.assertRaises(eval_torch.EvalError) as ctx:
                eval_torch.load_golden(path)
            self.assertIn(":1:", str(ctx.exception))
            self.assertIn("JSON", str(ctx.exception))

    def test_rejects_bad_rows(self):
        cases = {
            "пустой id": _valid_row(id=""),
            "пустой question": _valid_row(question="  "),
            "context не строка": _valid_row(context=123),
            "пустые checks": _valid_row(checks={}),
            "must не список": _valid_row(checks={"must": "990"}),
            "пустой must": _valid_row(checks={"must": []}),
            "must не строки": _valid_row(checks={"must": [990]}),
            "плохой regex": _valid_row(checks={"regex": ["("]}),
            "no_prompt_shingle=1": _valid_row(checks={"no_prompt_shingle": 1}),
            "no_prompt_shingle строка": _valid_row(checks={"no_prompt_shingle": "8"}),
        }
        for name, row in cases.items():
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "golden.jsonl"
                    _write_jsonl(path, [row])
                    with self.assertRaises(eval_torch.EvalError):
                        eval_torch.load_golden(path)

    def test_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "golden.jsonl"
            _write_jsonl(path, [_valid_row("same"), _valid_row("same")])
            with self.assertRaises(eval_torch.EvalError) as ctx:
                eval_torch.load_golden(path)
            self.assertIn("повтор id", str(ctx.exception))

    def test_rejects_empty_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.jsonl"
            path.write_text("\n# только комментарий\n", encoding="utf-8")
            with self.assertRaises(eval_torch.EvalError):
                eval_torch.load_golden(path)
            with self.assertRaises(eval_torch.EvalError):
                eval_torch.load_golden(Path(tmp) / "nope.jsonl")


# ---------------------------------------------------------------------------
# отложенные пары
# ---------------------------------------------------------------------------


class TestPairs(unittest.TestCase):
    def _write(self, path: Path) -> None:
        rows = [
            {"question": f"q{index}", "answer": f"a{index}"} for index in range(6)
        ]
        rows.append({"question": "без ответа", "answer": "  "})
        _write_jsonl(path, rows)

    def test_reads_and_skips_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.jsonl"
            self._write(path)
            meta = eval_torch.load_pairs(path, 0)
            self.assertEqual(meta["total"], 6)
            self.assertEqual(meta["skipped"], 1)
            self.assertEqual([row["question"] for row in meta["rows"]],
                             [f"q{index}" for index in range(6)])

    def test_limited_sample_is_even_and_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.jsonl"
            self._write(path)
            first = eval_torch.load_pairs(path, 3)["rows"]
            second = eval_torch.load_pairs(path, 3)["rows"]
            self.assertEqual(
                [row["question"] for row in first], ["q0", "q2", "q5"]
            )
            self.assertEqual(first, second)

    def test_bad_json_and_empty_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.jsonl"
            bad.write_text("[]\n", encoding="utf-8")
            with self.assertRaises(eval_torch.EvalError):
                eval_torch.load_pairs(bad)
            broken = Path(tmp) / "broken.jsonl"
            broken.write_text("{oops\n", encoding="utf-8")
            with self.assertRaises(eval_torch.EvalError) as ctx:
                eval_torch.load_pairs(broken)
            self.assertIn(":1:", str(ctx.exception))
            empty = Path(tmp) / "empty.jsonl"
            empty.write_text("\n", encoding="utf-8")
            with self.assertRaises(eval_torch.EvalError):
                eval_torch.load_pairs(empty)

    def test_sample_evenly(self):
        values = list(range(20))
        self.assertEqual(eval_torch.sample_evenly(values, 0), values)
        self.assertEqual(eval_torch.sample_evenly(values, 20), values)
        self.assertEqual(eval_torch.sample_evenly(values, 100), values)
        self.assertEqual(eval_torch.sample_evenly(values, 1), [0])
        self.assertEqual(eval_torch.sample_evenly(values, 2), [0, 19])
        sample = eval_torch.sample_evenly(values, 5)
        self.assertEqual(len(sample), 5)
        self.assertEqual(sample[0], 0)
        self.assertEqual(sample[-1], 19)


# ---------------------------------------------------------------------------
# промпты и токенизация пар
# ---------------------------------------------------------------------------


class TestPrompts(unittest.TestCase):
    def test_row_content_matches_run_eval(self):
        self.assertEqual(eval_torch.row_content({"question": "вопрос"}), "вопрос")
        self.assertEqual(
            eval_torch.row_content({"question": "вопрос", "context": "контекст"}),
            "Контекст:\nконтекст\n\nВопрос: вопрос",
        )

    def test_build_messages_has_system_and_user(self):
        messages = eval_torch.build_messages("система", {"question": "вопрос"})
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(messages[0]["content"], "система")
        self.assertEqual(messages[1]["content"], "вопрос")

    def test_render_prompt_uses_chat_template(self):
        tokenizer = FakeTokenizer()
        prompt = eval_torch.render_prompt(tokenizer, "система", {"question": "вопрос"})
        self.assertIn("system: система", prompt)
        self.assertIn("user: вопрос", prompt)
        self.assertTrue(tokenizer.calls[0]["messages"][0]["content"] == "система")

    def test_render_prompt_fallback_without_template(self):
        prompt = eval_torch.render_prompt(
            NoTemplateTokenizer(), "система", {"question": "вопрос"}
        )
        self.assertEqual(
            prompt, "Система: система\n\nПользователь: вопрос\n\nАссистент:"
        )

    def test_pair_prompt_matches_qat_data_format(self):
        tokenizer = FakeTokenizer(prompt_ids=(1, 2, 3))
        self.assertEqual(eval_torch.pair_prompt_ids(tokenizer, "вопрос"), [1, 2, 3])
        messages = tokenizer.calls[0]["messages"]
        self.assertEqual(messages, [{"role": "user", "content": "вопрос"}])
        self.assertTrue(tokenizer.calls[0]["tokenize"])

        fallback = eval_torch.pair_prompt_ids(NoTemplateTokenizer(), "вопрос")
        self.assertEqual(fallback, [5, 6])  # FakeTokenizer.__call__

    def test_pair_answer_appends_eos_once(self):
        tokenizer = FakeTokenizer(answer_ids=(5, 6), eos_token_id=9)
        self.assertEqual(eval_torch.pair_answer_ids(tokenizer, "ответ"), [5, 6, 9])
        tokenizer.answer_ids = [5, 9]
        self.assertEqual(eval_torch.pair_answer_ids(tokenizer, "ответ"), [5, 9])

    def test_tokenize_pair_masks_prompt(self):
        tokenizer = FakeTokenizer(prompt_ids=(1, 2, 3), answer_ids=(5, 6), eos_token_id=9)
        sample = eval_torch.tokenize_pair(tokenizer, {"question": "в", "answer": "о"}, 20)
        self.assertEqual(sample["input_ids"], [1, 2, 3, 5, 6, 9])
        self.assertEqual(sample["labels"], [-100, -100, -100, 5, 6, 9])
        self.assertEqual(sample["answer_tokens"], 3)

    def test_tokenize_pair_skips_too_long(self):
        tokenizer = FakeTokenizer(prompt_ids=(1, 2, 3), answer_ids=(5, 6), eos_token_id=9)
        self.assertIsNone(
            eval_torch.tokenize_pair(tokenizer, {"question": "в", "answer": "о"}, 5)
        )


# ---------------------------------------------------------------------------
# метрики и формат
# ---------------------------------------------------------------------------


class TestMetrics(unittest.TestCase):
    def test_summarize_accuracy_and_length(self):
        rows = [
            {"ok": True, "answer_tokens": 10, "answer_chars": 40},
            {"ok": False, "answer_tokens": 4, "answer_chars": 12},
        ]
        result = eval_torch.summarize(
            rows, perplexity=4.0, ppl_stats={"pairs": 2, "tokens": 10, "skipped": 1}
        )
        self.assertEqual(result["passed"], 1)
        self.assertEqual(result["total"], 2)
        self.assertAlmostEqual(result["accuracy"], 0.5)
        self.assertEqual(result["perplexity"], 4.0)
        self.assertAlmostEqual(result["avg_len"], 7.0)
        self.assertAlmostEqual(result["avg_len_chars"], 26.0)
        self.assertEqual(result["ppl_pairs"], 2)
        self.assertEqual(result["ppl_tokens"], 10)
        self.assertEqual(result["ppl_skipped"], 1)

    def test_summarize_empty(self):
        result = eval_torch.summarize([], perplexity=None)
        self.assertEqual(result["accuracy"], 0.0)
        self.assertEqual(result["passed"], 0)
        self.assertEqual(result["total"], 0)
        self.assertEqual(result["avg_len"], 0.0)
        self.assertIsNone(result["perplexity"])

    def test_compute_perplexity(self):
        self.assertAlmostEqual(
            eval_torch.compute_perplexity(math.log(10.0), 1), 10.0
        )
        self.assertAlmostEqual(
            eval_torch.compute_perplexity(2.0 * math.log(5.0), 2), 5.0
        )
        self.assertIsNone(eval_torch.compute_perplexity(1.0, 0))
        self.assertIsNone(eval_torch.compute_perplexity(math.nan, 10))
        self.assertIsNone(eval_torch.compute_perplexity(math.inf, 10))

    def test_format_report_lists_failures_and_summary(self):
        rows = [
            {"id": "good", "ok": True, "problems": []},
            {"id": "bad", "ok": False, "problems": ["нет «990»"]},
        ]
        result = eval_torch.summarize(
            rows, perplexity=3.5, ppl_stats={"pairs": 2, "tokens": 20, "skipped": 0}
        )
        result.update({"model": "Qwen/Qwen2.5-1.5B-Instruct", "out": "runs/eval.json",
                       "rows": rows})
        text = eval_torch.format_report(result)
        self.assertIn("FAIL bad: нет «990»", text)
        self.assertIn("accuracy 1/2 = 50.0%", text)
        self.assertIn("perplexity 3.500 (2 пар, 20 токенов)", text)
        self.assertIn("avg_len", text)
        self.assertIn("model Qwen/Qwen2.5-1.5B-Instruct", text)
        self.assertNotIn("FAIL good", text)

    def test_format_report_without_perplexity(self):
        result = eval_torch.summarize(
            [{"id": "a", "ok": True, "answer_tokens": 1}], perplexity=None
        )
        result.update({"model": "m", "out": "o"})
        text = eval_torch.format_report(result)
        self.assertIn("perplexity n/a", text)
        self.assertIn("accuracy 1/1 = 100.0%", text)

    def test_generated_token_count(self):
        self.assertEqual(
            eval_torch._generated_token_count([1, 3, 2, 2, 0], {2}, 0), 2
        )
        self.assertEqual(
            eval_torch._generated_token_count([0, 0, 5, 6], {2}, 0), 2
        )
        self.assertEqual(
            eval_torch._generated_token_count([2, 5], {2}, 0), 0
        )

    def test_ppl_batches_respects_token_budget(self):
        samples = [{"input_ids": [1] * length} for length in (3, 3, 3, 3, 3, 3, 50)]
        batches = list(eval_torch._ppl_batches(samples, 8, 10))
        for batch in batches:
            width = max(len(sample["input_ids"]) for sample in batch)
            if len(batch) > 1:
                self.assertLessEqual(len(batch) * width, 10)
        self.assertEqual(sum(len(batch) for batch in batches), len(samples))
        self.assertEqual([len(batch) for batch in batches], [3, 3, 1])


# ---------------------------------------------------------------------------
# модель: что передали в --model
# ---------------------------------------------------------------------------


class TestResolveModel(unittest.TestCase):
    def test_hf_id_is_not_touched(self):
        spec = eval_torch.resolve_model("Qwen/Qwen2.5-1.5B-Instruct")
        self.assertEqual(spec["kind"], "hf")
        self.assertFalse(spec["local"])

    def test_local_hf_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "config.json").write_text("{}", encoding="utf-8")
            spec = eval_torch.resolve_model(tmp)
            self.assertEqual(spec["kind"], "hf")
            self.assertTrue(spec["local"])

    def test_run_directory_uses_latest_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "pilot1"
            (run / "checkpoints" / "step_000010").mkdir(parents=True)
            latest = run / "checkpoints" / "step_000020"
            latest.mkdir()
            (latest / "model.pt").write_bytes(b"")
            (latest / "trainer_state.pt").write_bytes(b"")
            spec = eval_torch.resolve_model(run)
            self.assertEqual(spec["kind"], "checkpoint")
            self.assertEqual(Path(spec["checkpoint"]).name, "step_000020")

    def test_checkpoint_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "step_000005"
            ckpt.mkdir()
            (ckpt / "model.pt").write_bytes(b"")
            (ckpt / "trainer_state.pt").write_bytes(b"")
            spec = eval_torch.resolve_model(ckpt)
            self.assertEqual(spec["kind"], "checkpoint")
            self.assertEqual(Path(spec["checkpoint"]).resolve(), ckpt.resolve())

    def test_bad_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "empty"
            empty.mkdir()
            with self.assertRaises(eval_torch.EvalError):
                eval_torch.resolve_model(empty)
            run = Path(tmp) / "run"
            (run / "checkpoints").mkdir(parents=True)
            with self.assertRaises(eval_torch.EvalError):
                eval_torch.resolve_model(run)
            only_model = Path(tmp) / "only"
            only_model.mkdir()
            (only_model / "model.pt").write_bytes(b"")
            with self.assertRaises(eval_torch.EvalError) as ctx:
                eval_torch.resolve_model(only_model)
            self.assertIn("trainer_state", str(ctx.exception))
            file_path = Path(tmp) / "weights.bin"
            file_path.write_bytes(b"")
            with self.assertRaises(eval_torch.EvalError):
                eval_torch.resolve_model(file_path)


# ---------------------------------------------------------------------------
# запись JSON и проверки из eval/run_eval.py
# ---------------------------------------------------------------------------


class TestQuantConfigFromState(unittest.TestCase):
    def test_reads_block_size_and_embeddings(self):
        quant, info = eval_torch._quant_config_from_state({
            "config": {"quant": {
                "block_size": 64,
                "scale_dtype": "float16",
                "target_suffixes": ["q_proj", "v_proj"],
                "quantize_embeddings": "trit",
            }}
        })
        self.assertEqual(quant.block_size, 64)
        self.assertEqual(quant.target_suffixes, ("q_proj", "v_proj"))
        self.assertEqual(quant.quantize_embeddings, "trit")
        self.assertEqual(info, {"block_size": 64, "quantize_embeddings": "trit"})

    def test_defaults_for_old_state(self):
        quant, info = eval_torch._quant_config_from_state({"config": {}})
        self.assertEqual(quant.block_size, 128)
        self.assertEqual(quant.quantize_embeddings, "none")
        self.assertEqual(info, {"block_size": 128, "quantize_embeddings": "none"})


class TestResolveDevice(unittest.TestCase):
    def _torch(self, cuda=False, mps=False):
        device = SimpleNamespace(type="cuda" if cuda else "cpu")
        return SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: cuda),
            backends=SimpleNamespace(
                mps=SimpleNamespace(is_available=lambda: mps) if mps is not None else None
            ),
            device=lambda name: SimpleNamespace(type=name.split(":")[0]),
        )

    def test_auto_prefers_cuda_then_cpu(self):
        self.assertEqual(eval_torch.resolve_device(self._torch(cuda=True), "auto").type,
                         "cuda")
        self.assertEqual(eval_torch.resolve_device(self._torch(), "auto").type, "cpu")

    def test_explicit_cuda_requires_availability(self):
        with self.assertRaises(eval_torch.EvalError):
            eval_torch.resolve_device(self._torch(cuda=False), "cuda")
        self.assertEqual(
            eval_torch.resolve_device(self._torch(cuda=True), "cuda:0").type, "cuda"
        )

    def test_unknown_device_is_reported(self):
        def fail(name):
            raise RuntimeError("bad device")

        torch = self._torch()
        torch.device = fail
        with self.assertRaises(eval_torch.EvalError):
            eval_torch.resolve_device(torch, "npu")


class TestWriteAndValidators(unittest.TestCase):
    def test_write_json_creates_directory_and_is_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "eval.json"
            written = eval_torch.write_json(path, {"accuracy": 1.0, "note": "тест"})
            self.assertEqual(written, path)
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["accuracy"], 1.0)
            self.assertEqual(loaded["note"], "тест")
            self.assertFalse(path.with_name(path.name + ".tmp").exists())

    def test_validators_reuse_run_eval_check(self):
        validators = eval_torch.load_validators()
        ok, problems = validators.check(
            "Я EVA32, работаю на базовой модели DeepSeek V4-Flash.",
            {"must": ["EVA32"], "any": ["V4", "Flash"], "forbid": ["Qwen"]},
        )
        self.assertTrue(ok, problems)
        ok, problems = validators.check(
            "На мне Qwen и OpenAI.", {"forbid": ["Qwen", "OpenAI"]}
        )
        self.assertFalse(ok)
        self.assertTrue(any("запрещённое" in problem for problem in problems))

    def test_read_system_prompt(self):
        text = eval_torch.read_system_prompt()
        self.assertIn("EVA32", text)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(eval_torch.EvalError):
                eval_torch.read_system_prompt(Path(tmp) / "missing.md")


# ---------------------------------------------------------------------------
# dry-run и обычный запуск
# ---------------------------------------------------------------------------


class TestDryRun(unittest.TestCase):
    def test_dry_run_returns_zero_without_torch(self):
        out, err = io.StringIO(), io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "eval.json"
            # Если dry-run полезет проверять torch, тест упадёт.
            with mock.patch.object(
                eval_torch, "missing_modules", side_effect=AssertionError("torch не нужен")
            ), redirect_stdout(out), redirect_stderr(err):
                code = eval_torch.main([
                    "--dry-run", "--golden", str(GOLDEN), "--out", str(out_path),
                ])
            self.assertEqual(code, 0, err.getvalue())
            text = out.getvalue()
            self.assertIn("dry-run", text)
            self.assertIn("36 вопросов", text)
            self.assertIn("torch не импортируется", text)
            self.assertIn("perplexity: не будет посчитана", text)
            self.assertIn("проверки считаются", text)
            self.assertFalse(out_path.exists())

    def test_dry_run_with_limit_and_pairs(self):
        out, err = io.StringIO(), io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            pairs = Path(tmp) / "pairs.jsonl"
            _write_jsonl(
                pairs,
                [{"question": f"q{index}", "answer": f"a{index}"} for index in range(4)],
            )
            with redirect_stdout(out), redirect_stderr(err):
                code = eval_torch.main([
                    "--dry-run",
                    "--golden", str(GOLDEN),
                    "--limit", "3",
                    "--ppl-data", str(pairs),
                    "--ppl-limit", "2",
                    "--out", str(Path(tmp) / "eval.json"),
                ])
            self.assertEqual(code, 0, err.getvalue())
            text = out.getvalue()
            self.assertIn("3 вопросов (--limit: первые 3 из 36)", text)
            self.assertIn("2 пар из 4", text)
            self.assertIn("2 пар perplexity", text)

    def test_dry_run_bad_golden_returns_2(self):
        out, err = io.StringIO(), io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            golden = Path(tmp) / "golden.jsonl"
            golden.write_text("{плохо}\n", encoding="utf-8")
            with redirect_stdout(out), redirect_stderr(err):
                code = eval_torch.main(["--dry-run", "--golden", str(golden)])
            self.assertEqual(code, 2)
            self.assertIn("невалидный JSON", err.getvalue())
            self.assertIn(":1:", err.getvalue())

    def test_dry_run_bad_pairs_returns_2(self):
        out, err = io.StringIO(), io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            pairs = Path(tmp) / "pairs.jsonl"
            pairs.write_text("", encoding="utf-8")
            with redirect_stdout(out), redirect_stderr(err):
                code = eval_torch.main([
                    "--dry-run", "--golden", str(GOLDEN), "--ppl-data", str(pairs),
                ])
            self.assertEqual(code, 2)
            self.assertIn("нет пригодных пар", err.getvalue())


class TestMainWithoutTorch(unittest.TestCase):
    def test_normal_run_without_torch_returns_2(self):
        out, err = io.StringIO(), io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "eval.json"
            with mock.patch.object(
                eval_torch, "missing_modules", return_value=["torch", "transformers"]
            ), mock.patch.object(
                eval_torch, "run_evaluation", side_effect=AssertionError("не должен вызываться")
            ), redirect_stdout(out), redirect_stderr(err):
                code = eval_torch.main([
                    "--model", "Qwen/Qwen2.5-1.5B-Instruct",
                    "--golden", str(GOLDEN),
                    "--out", str(out_path),
                ])
            self.assertEqual(code, 2)
            text = err.getvalue()
            self.assertIn("не хватает зависимостей", text)
            self.assertIn("torch", text)
            self.assertIn("pip install -r qat/requirements-qat.txt", text)
            self.assertFalse(out_path.exists())


class TestMainWithStubModel(unittest.TestCase):
    """Оркестрация main: JSON и человеческая строка без torch."""

    def test_success_writes_required_fields(self):
        stub_rows = [
            {"id": "good", "ok": True, "problems": [], "answer": "990",
             "answer_tokens": 2, "answer_chars": 3},
            {"id": "bad", "ok": False, "problems": ["нет «[1]»"],
             "answer": "text", "answer_tokens": 4, "answer_chars": 4},
        ]
        summary = eval_torch.summarize(
            stub_rows, perplexity=5.0, ppl_stats={"pairs": 3, "tokens": 30, "skipped": 0}
        )
        summary.update({"model": "stub-model", "base_model": "stub-model",
                        "checkpoint": None, "step": None, "quant": None,
                        "device": "cpu", "max_new_tokens": 16, "batch_size": 2,
                        "max_seq_len": 128, "rows": stub_rows})
        out, err = io.StringIO(), io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "runs" / "eval.json"
            with mock.patch.object(eval_torch, "missing_modules", return_value=[]), \
                 mock.patch.object(eval_torch, "read_system_prompt", return_value="sys"), \
                 mock.patch.object(eval_torch, "load_validators", return_value=object()), \
                 mock.patch.object(
                     eval_torch, "run_evaluation", return_value=summary
                 ) as run_mock, \
                 redirect_stdout(out), redirect_stderr(err):
                code = eval_torch.main([
                    "--model", "stub-model",
                    "--golden", str(GOLDEN),
                    "--out", str(out_path),
                ])
            self.assertEqual(code, 0, err.getvalue())
            self.assertTrue(run_mock.called)
            self.assertTrue(out_path.exists())
            result = json.loads(out_path.read_text(encoding="utf-8"))
            for key in ("accuracy", "passed", "total", "perplexity", "avg_len"):
                self.assertIn(key, result)
            self.assertAlmostEqual(result["accuracy"], 0.5)
            self.assertEqual(result["passed"], 1)
            self.assertEqual(result["total"], 2)
            self.assertEqual(result["perplexity"], 5.0)
            self.assertAlmostEqual(result["avg_len"], 3.0)
            self.assertEqual(result["golden"], str(GOLDEN))
            self.assertEqual(result["out"], str(out_path))
            text = out.getvalue()
            self.assertIn("FAIL bad: нет «[1]»", text)
            self.assertIn("accuracy 1/2 = 50.0%", text)
            self.assertIn("perplexity 5.000", text)


if __name__ == "__main__":
    unittest.main()
