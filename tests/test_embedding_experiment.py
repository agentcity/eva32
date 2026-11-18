"""Раннер эксперимента 0027: парсинг, команды, таблица — без torch и сети.

Проверяется то, что можно проверить на машине разработчика: разбор YAML без
pyyaml, подстановка `quantize_embeddings`, сборка команд `qat.train`/`qat.export`,
формат итоговой таблицы, чтение loss/eval и поведение `--dry-run`/отказа без
torch. Реальное обучение здесь не запускается: его место — GPU-машина.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "embedding_experiment.py"
sys.path.insert(0, str(ROOT / "tools"))

import embedding_experiment as ee  # noqa: E402

QAT_CONFIG = ROOT / "qat" / "configs" / "qwen1.5b.yaml"
QAT_DATA = ROOT / "distill" / "out" / "pairs.jsonl"


def run_tool(*args: str, timeout: float = 120.0) -> subprocess.CompletedProcess:
    """Запустить раннер как пользователь: отдельным процессом, без импорта."""
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )


class TestConfigScalars(unittest.TestCase):
    def test_reads_qat_config_without_yaml(self):
        values = ee.config_values(QAT_CONFIG)
        self.assertEqual(values["micro_batch_size"], 4)
        self.assertEqual(values["grad_accum_steps"], 8)
        self.assertEqual(values["max_steps"], 2000)
        self.assertEqual(values["save_every"], 100)

    def test_comments_quotes_and_duplicates(self):
        text = (
            "train:\n"
            "  micro_batch_size: 2  # меньше — если OOM\n"
            "  grad_accum_steps: 4\n"
            "other:\n"
            "  micro_batch_size: 99  # чужой ключ не должен перебить\n"
            "  unknown_key: 7\n"
        )
        values = ee.parse_config_scalars(text)
        self.assertEqual(values["micro_batch_size"], 2)
        self.assertEqual(values["grad_accum_steps"], 4)
        self.assertNotIn("unknown_key", values)

    def test_missing_keys_absent(self):
        self.assertEqual(ee.parse_config_scalars("model:\n  revision: main\n"), {})

    def test_garbage_does_not_crash(self):
        self.assertEqual(ee.parse_config_scalars("::: не yaml :::\n- 1\n"), {})


class TestSteps(unittest.TestCase):
    def test_pairs_times_epochs_over_batch(self):
        # 10000 пар × 3 эпохи / (4 × 8) = 937.5 -> 938 шагов
        self.assertEqual(ee.steps_for_run(10000, 3, 4, 8), 938)

    def test_max_steps_wins(self):
        self.assertEqual(ee.steps_for_run(10000, 3, 4, 8, max_steps=2000), 2000)

    def test_never_zero(self):
        self.assertEqual(ee.steps_for_run(0, 3, 4, 8), 1)
        self.assertEqual(ee.steps_for_run(1, 1, 4, 8), 1)

    def test_single_sample_batch(self):
        self.assertEqual(ee.steps_for_run(7, 2, 1, 1), 14)


class TestPatchConfig(unittest.TestCase):
    def setUp(self):
        self.text = QAT_CONFIG.read_text(encoding="utf-8")

    def test_replaces_existing_key_for_each_mode(self):
        for mode in ee.MODES:
            patched = ee.patch_config_text(self.text, mode)
            self.assertIn(f"quantize_embeddings: {mode}", patched)
            self.assertEqual(patched.count("quantize_embeddings:"), 1)
            self.assertEqual(len(patched.splitlines()), len(self.text.splitlines()))

    def test_other_lines_untouched(self):
        patched = ee.patch_config_text(self.text, "trit")
        for old, new in zip(self.text.splitlines(), patched.splitlines()):
            if "quantize_embeddings:" in old:
                continue
            self.assertEqual(old, new)

    def test_insert_into_quant_section_when_key_absent(self):
        text = (
            "model:\n  revision: main\n"
            "quant:\n  block_size: 128\n  scale_dtype: float16\n"
            "\ndata:\n  path: distill/out/pairs.jsonl\n"
        )
        patched = ee.patch_config_text(text, "int8")
        self.assertIn("quantize_embeddings: int8", patched)
        self.assertLess(
            patched.index("quantize_embeddings: int8"),
            patched.index("data:"),
            "ключ должен остаться внутри секции quant, а не после data",
        )
        self.assertIn("  block_size: 128", patched)

    def test_appends_section_when_missing(self):
        patched = ee.patch_config_text("model:\n  revision: main\n", "trit")
        self.assertTrue(patched.rstrip().endswith("quant:\n  quantize_embeddings: trit"))
        self.assertIn("model:\n  revision: main", patched)

    def test_keeps_comment_line_above_key(self):
        text = "quant:\n  # почему эмбеддинги отдельно\n  quantize_embeddings: none\n"
        patched = ee.patch_config_text(text, "trit")
        self.assertIn("  # почему эмбеддинги отдельно\n  quantize_embeddings: trit", patched)

    def test_keeps_inline_comment_after_value(self):
        text = "quant:\n  quantize_embeddings: none  # старый режим\n"
        patched = ee.patch_config_text(text, "int8")
        self.assertIn("  quantize_embeddings: int8  # старый режим", patched)

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            ee.patch_config_text(self.text, "fp4")


class TestParser(unittest.TestCase):
    def test_defaults(self):
        args = ee.build_parser().parse_args([])
        self.assertEqual(args.data, str(ee.DEFAULT_DATA))
        self.assertEqual(args.config, str(ee.DEFAULT_CONFIG))
        self.assertEqual(args.out, str(ee.DEFAULT_OUT))
        self.assertEqual(args.epochs, ee.DEFAULT_EPOCHS)
        self.assertIsNone(args.max_steps)
        self.assertFalse(args.dry_run)
        self.assertEqual(args.train_arg, [])

    def test_documented_cli(self):
        args = ee.build_parser().parse_args([
            "--data", "distill/out/pairs.jsonl", "--epochs", "1",
            "--max-steps", "2000", "--out", "runs/emb", "--dry-run",
        ])
        self.assertEqual(args.data, "distill/out/pairs.jsonl")
        self.assertEqual(args.epochs, 1)
        self.assertEqual(args.max_steps, 2000)
        self.assertEqual(args.out, "runs/emb")
        self.assertTrue(args.dry_run)

    def test_train_arg_repeatable(self):
        args = ee.build_parser().parse_args([
            "--train-arg=--no-save", "--train-arg=--allow-data-change",
        ])
        self.assertEqual(args.train_arg, ["--no-save", "--allow-data-change"])


class TestCommands(unittest.TestCase):
    def test_train_command_calls_existing_module(self):
        command = ee.build_train_command(
            config=Path("/tmp/emb/_configs/trit.yaml"),
            data=Path("/data/pairs.jsonl"),
            out=Path("/tmp/emb/trit"),
            steps=2000,
        )
        self.assertEqual(command[:3], [sys.executable, "-m", "qat.train"])
        for flag, value in (
            ("--config", "/tmp/emb/_configs/trit.yaml"),
            ("--data", "/data/pairs.jsonl"),
            ("--out", "/tmp/emb/trit"),
            ("--max-steps", "2000"),
        ):
            self.assertIn(flag, command)
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertNotIn("--resume", command)

    def test_resume_and_extra_args(self):
        command = ee.build_train_command(
            config=Path("/c.yaml"),
            data=Path("/d.jsonl"),
            out=Path("/o/none"),
            steps=7,
            resume=True,
            extra=["--no-save"],
        )
        self.assertIn("--resume", command)
        self.assertEqual(command[-1], "--no-save")
        self.assertLess(command.index("--max-steps"), command.index("--resume"))

    def test_export_command(self):
        command = ee.build_export_command(
            run=Path("/tmp/emb/int8"), export=Path("/tmp/emb/int8/export"))
        self.assertEqual(command[:3], [sys.executable, "-m", "qat.export"])
        self.assertEqual(command[command.index("--run") + 1], "/tmp/emb/int8")
        self.assertEqual(command[command.index("--out") + 1], "/tmp/emb/int8/export")

    def test_three_mode_commands_differ_only_in_paths(self):
        commands = {
            mode: ee.build_train_command(
                config=Path(f"/out/_configs/{mode}.yaml"),
                data=Path("/data/pairs.jsonl"),
                out=Path(f"/out/{mode}"),
                steps=2000,
            )
            for mode in ee.MODES
        }
        configs = [command[command.index("--config") + 1]
                   for command in commands.values()]
        self.assertEqual(len(set(configs)), 3, "у каждого режима свой конфиг")
        self.assertTrue(all(len(command) == len(commands["none"])
                            for command in commands.values()),
                        "набор флагов одинаковый — режим живёт только в конфиге")


class TestTable(unittest.TestCase):
    def _results(self):
        return [
            ee.RunResult("none", Path("/r/none"), status="done", export_bytes=1024,
                         loss=2.5, metric=(3, 4), hours=6.5),
            ee.RunResult("int8", Path("/r/int8"), status="done", export_bytes=2048,
                         loss=2.25, metric=(3, 4), hours=7.0),
            ee.RunResult("trit", Path("/r/trit"), status="failed"),
        ]

    def test_metric_format(self):
        self.assertEqual(ee.format_metric(3, 4), "3/4 (75%)")
        self.assertEqual(ee.format_metric(7, 8), "7/8 (88%)")
        self.assertEqual(ee.format_metric(0, 0), "—")

    def test_header_is_the_required_six_columns(self):
        table = ee.format_table(self._results(), 100.0)
        header = [cell.strip() for cell in table.splitlines()[0].split("|")]
        self.assertEqual(
            header,
            ["режим", "размер экспорта", "loss", "метрика eval", "часы", "₽"],
        )

    def test_row_values(self):
        table = ee.format_table(self._results(), 100.0)
        none_row = next(line for line in table.splitlines() if line.startswith("none"))
        self.assertIn("1.0 КиБ", none_row)
        self.assertIn("2.5000", none_row)
        self.assertIn("3/4 (75%)", none_row)
        self.assertIn("6.50", none_row)
        self.assertIn("650", none_row)

    def test_missing_values_are_dashes(self):
        table = ee.format_table(self._results(), 100.0)
        trit_row = next(line for line in table.splitlines() if line.startswith("trit"))
        self.assertEqual(trit_row.count("—"), 5)
        self.assertNotIn("None", table)

    def test_total_row_sums_hours_and_money(self):
        table = ee.format_table(self._results(), 120.0)
        total = next(line for line in table.splitlines() if line.startswith("ИТОГО"))
        self.assertIn("13.50", total)
        self.assertIn("1620", total)
        self.assertIn("3.0 КиБ", total)

    def test_plan_table(self):
        plan = ee.make_plan(samples=10000, epochs=3, micro_batch=4, grad_accum=8,
                            max_steps=None, sec_per_step=10.8, with_export=True)
        self.assertEqual([row.mode for row in plan], list(ee.MODES))
        self.assertTrue(all(row.steps == 938 for row in plan))
        table = ee.format_plan(plan, 100.0)
        header = [cell.strip() for cell in table.splitlines()[0].split("|")]
        self.assertEqual(header, ["режим", "шагов", "обучение, ч", "экспорт, ч",
                                  "всего, ч", "₽"])
        total = next(line for line in table.splitlines() if line.startswith("ИТОГО"))
        self.assertIn("2814", total)   # 938 шагов × 3 прогона
        self.assertIn("8.44", total)   # 2.814 ч обучения × 3
        self.assertIn("9.94", total)   # + 0.5 ч экспорта × 3
        self.assertIn("994", total)

    def test_plan_money_matches_hours(self):
        plan = ee.make_plan(samples=206, epochs=3, micro_batch=4, grad_accum=8,
                            max_steps=None, sec_per_step=10.8, with_export=True)
        for row in plan:
            self.assertAlmostEqual(row.rub(100.0), row.hours * 100.0)


class TestComparison(unittest.TestCase):
    def test_deltas_to_baseline(self):
        results = [
            ee.RunResult("none", Path("/r/none"), metric=(10, 10)),
            ee.RunResult("int8", Path("/r/int8"), metric=(10, 10)),
            ee.RunResult("trit", Path("/r/trit"), metric=(8, 10)),
        ]
        notes = "\n".join(ee.comparison_notes(results))
        self.assertIn("int8: 10/10 (100%) (+0.0 п.п. к none)", notes)
        self.assertIn("trit: 8/10 (80%) (-20.0 п.п. к none)", notes)
        self.assertIn("вне порога", notes)

    def test_no_baseline_no_notes(self):
        results = [ee.RunResult("int8", Path("/r/int8"), metric=(1, 2))]
        self.assertEqual(ee.comparison_notes(results), [])

    def test_different_totals_are_not_compared(self):
        results = [
            ee.RunResult("none", Path("/r/none"), metric=(8, 8)),
            ee.RunResult("trit", Path("/r/trit"), metric=(4, 5)),
        ]
        notes = "\n".join(ee.comparison_notes(results))
        self.assertIn("другой размер набора", notes)


class TestReads(unittest.TestCase):
    def test_last_loss_skips_broken_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "log.jsonl"
            log.write_text(
                '{"step": 1, "loss": 3.0}\n'
                "не json\n"
                '{"step": 2, "loss": 2.5}\n'
                '{"step": 3, "loss": null}\n',
                encoding="utf-8",
            )
            self.assertEqual(ee.read_last_loss(log), 2.5)

    def test_last_loss_missing_file(self):
        self.assertIsNone(ee.read_last_loss(Path("/nonexistent/log.jsonl")))

    def test_read_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp) / "eval-trit.json"
            good.write_text(json.dumps({"passed": 7, "total": 8}), encoding="utf-8")
            self.assertEqual(ee.read_metric(good), (7, 8))

            bad = Path(tmp) / "bad.json"
            bad.write_text(json.dumps({"passed": 9, "total": 8}), encoding="utf-8")
            self.assertIsNone(ee.read_metric(bad))

            broken = Path(tmp) / "broken.json"
            broken.write_text("{", encoding="utf-8")
            self.assertIsNone(ee.read_metric(broken))

            self.assertIsNone(ee.read_metric(Path(tmp) / "absent.json"))

    def test_directory_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sub").mkdir()
            (root / "a.bin").write_bytes(b"x" * 100)
            (root / "sub" / "b.bin").write_bytes(b"y" * 23)
            self.assertEqual(ee.directory_size(root), 123)
            self.assertIsNone(ee.directory_size(root / "nope"))

    def test_count_and_inspect_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp) / "pairs.jsonl"
            good.write_text(
                '{"question": "q", "answer": "a"}\n\n{"question": "q2", "answer": "a2"}\n',
                encoding="utf-8",
            )
            self.assertEqual(ee.count_samples(good), 2)
            info = ee.inspect_data(good)
            self.assertTrue(info.exists)
            self.assertEqual(info.samples, 2)
            self.assertIsNone(info.problem)

            empty = Path(tmp) / "empty.jsonl"
            empty.write_text("\n\n", encoding="utf-8")
            self.assertEqual(ee.inspect_data(empty).samples, 0)
            self.assertIsNotNone(ee.inspect_data(empty).problem)

            wrong = Path(tmp) / "wrong.jsonl"
            wrong.write_text('{"q": 1}\n', encoding="utf-8")
            self.assertIn("question/answer", ee.inspect_data(wrong).problem or "")

            missing = ee.inspect_data(Path(tmp) / "missing.jsonl")
            self.assertFalse(missing.exists)
            self.assertIn("нет", missing.problem or "")


class TestDryRun(unittest.TestCase):
    def test_plan_without_torch_exit_zero_and_no_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "runs" / "emb"
            proc = run_tool("--dry-run", "--out", str(out),
                            "--data", str(QAT_DATA), "--max-steps", "2000")
            self.assertEqual(proc.returncode, 0, proc.stderr)
            text = proc.stdout
            self.assertIn("ПЛАН", text)
            for mode in ee.MODES:
                self.assertIn(mode, text)
            self.assertIn("₽", text)
            self.assertIn("ИТОГО", text)
            self.assertIn("2000", text)
            self.assertFalse(out.exists(), "--dry-run не должен создавать каталоги")

    def test_plan_uses_config_batch_and_data_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = tmp_path / "tiny.yaml"
            config.write_text(
                "train:\n  micro_batch_size: 2\n  grad_accum_steps: 4\n"
                "quant:\n  block_size: 8\n",
                encoding="utf-8",
            )
            data = tmp_path / "pairs.jsonl"
            data.write_text(
                "".join(
                    json.dumps({"question": f"q{i}", "answer": "a"}) + "\n"
                    for i in range(10)
                ),
                encoding="utf-8",
            )
            out = tmp_path / "out"
            proc = run_tool("--dry-run", "--config", str(config), "--data", str(data),
                            "--out", str(out), "--epochs", "2", "--rub-per-hour", "120")
            self.assertEqual(proc.returncode, 0, proc.stderr)
            # 10 пар × 2 эпохи / (2 × 4) = 2.5 -> 3 шага
            self.assertIn("3 (батч 2×4 = 8", proc.stdout)
            self.assertIn("120 ₽/ч", proc.stdout)
            self.assertFalse(out.exists())

    def test_missing_data_falls_back_to_samples_and_exit_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            proc = run_tool("--dry-run", "--data", str(Path(tmp) / "no.jsonl"),
                            "--out", str(out), "--samples", "4321", "--epochs", "1")
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("оценка 4321 пар", proc.stdout)
            self.assertIn("нет", proc.stdout)
            self.assertFalse(out.exists())

    def test_bad_epochs_is_usage_error(self):
        proc = run_tool("--dry-run", "--epochs", "0")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("--epochs", proc.stderr)

    def test_missing_config_is_error(self):
        proc = run_tool("--dry-run", "--config", "/nonexistent/config.yaml")
        self.assertEqual(proc.returncode, 2)
        self.assertIn("конфиг", proc.stderr)


class TestSequentialRun(unittest.TestCase):
    """Оркестрация без обучения: qat.train/qat.export подменены заглушкой."""

    def _args(self, tmp: Path, data: Path, **overrides):
        parser = ee.build_parser()
        args = parser.parse_args(
            ["--out", str(tmp / "emb"), "--data", str(data), "--max-steps", "4"])
        args.config = QAT_CONFIG
        args.data = data
        args.out_dir = tmp / "emb"
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def _data(self, tmp: Path) -> Path:
        data = tmp / "pairs.jsonl"
        data.write_text(
            "".join(
                json.dumps({"question": f"q{i}", "answer": "a"}) + "\n"
                for i in range(8)
            ),
            encoding="utf-8",
        )
        return data

    def _fake(self, calls, fail_prefix=None):
        def fake_run(command, *, log_path, prefix, shell=False):
            self.assertFalse(shell, "обучение и экспорт не идут через shell")
            command = list(command)
            out = Path(command[command.index("--out") + 1])
            calls.append((prefix, command))
            if prefix == fail_prefix:
                return 1
            out.mkdir(parents=True, exist_ok=True)
            if "qat.train" in command:
                (out / "log.jsonl").write_text(
                    json.dumps({"step": 4, "loss": 1.5}) + "\n", encoding="utf-8")
                (out / "DONE").write_text("step=4\n", encoding="utf-8")
            else:
                (out / "manifest.json").write_text("{}", encoding="utf-8")
                (out / "ternary.pt").write_bytes(b"x" * 10)
            return 0
        return fake_run

    def test_three_modes_in_order_with_their_own_configs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            args = self._args(tmp_path, self._data(tmp_path))
            calls: list[tuple[str, list[str]]] = []
            stdout = StringIO()
            with mock.patch.object(ee, "missing_modules", return_value=[]), \
                    mock.patch.object(ee, "run_streamed", side_effect=self._fake(calls)), \
                    redirect_stdout(stdout):
                code = ee.run_experiment(args)
            self.assertEqual(code, 0)
            self.assertEqual(
                [prefix for prefix, _ in calls if prefix in ee.MODES],
                ["none", "int8", "trit"],
            )
            # int8-экспорт отложен (нужен kind 4 в контейнере, задача 0044),
            # поэтому экспорт запускается только для none и trit.
            self.assertEqual(
                [prefix for prefix, _ in calls if prefix.endswith("/export")],
                ["none/export", "trit/export"],
            )
            for mode in ee.MODES:
                config = (tmp_path / "emb" / "_configs" / f"{mode}.yaml").read_text(
                    encoding="utf-8")
                self.assertIn(f"quantize_embeddings: {mode}", config)
            summary = json.loads((tmp_path / "emb" / "summary.json").read_text(
                encoding="utf-8"))
            self.assertEqual([run["mode"] for run in summary["runs"]], list(ee.MODES))
            self.assertTrue(all(run["status"] == "done" for run in summary["runs"]))
            self.assertTrue(all(run["loss"] == 1.5 for run in summary["runs"]))
            by_mode = {run["mode"]: run for run in summary["runs"]}
            self.assertEqual(by_mode["none"]["export_bytes"], 12)
            self.assertEqual(by_mode["trit"]["export_bytes"], 12)
            self.assertFalse(by_mode["int8"].get("export_bytes"))
            self.assertIn("отложен", by_mode["int8"].get("note") or "")
            table = stdout.getvalue()
            header = next(line for line in table.splitlines() if "метрика eval" in line)
            self.assertEqual(
                [cell.strip() for cell in header.split("|")],
                ["режим", "размер экспорта", "loss", "метрика eval", "часы", "₽"],
            )
            self.assertIn("ИТОГО", table)
            # одинаковый размер экспорта — известное ограничение qat.export
            self.assertIn("размер экспорта одинаков", table)

    def test_eval_cmd_fills_metric_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            args = self._args(
                tmp_path, self._data(tmp_path),
                eval_cmd="python3 eval/run_eval.py --out {eval_json}",
            )
            calls: list[tuple[str, list[str]]] = []
            base_fake = self._fake(calls)

            def fake_run(command, *, log_path, prefix, shell=False):
                if prefix.endswith("/eval"):
                    self.assertTrue(shell, "eval-cmd идёт через shell")
                    target = Path(command.split("--out ", 1)[1].split()[0])
                    target.write_text(
                        json.dumps({"passed": 3, "total": 4}), encoding="utf-8")
                    calls.append((prefix, [command]))
                    return 0
                return base_fake(command, log_path=log_path, prefix=prefix, shell=shell)

            stdout = StringIO()
            with mock.patch.object(ee, "missing_modules", return_value=[]), \
                    mock.patch.object(ee, "run_streamed", side_effect=fake_run), \
                    redirect_stdout(stdout):
                code = ee.run_experiment(args)
            self.assertEqual(code, 0)
            self.assertEqual(
                [prefix for prefix, _ in calls if prefix.endswith("/eval")],
                ["none/eval", "int8/eval", "trit/eval"],
            )
            summary = json.loads((tmp_path / "emb" / "summary.json").read_text(
                encoding="utf-8"))
            self.assertTrue(all(run["metric"] == {"passed": 3, "total": 4}
                                for run in summary["runs"]))
            self.assertIn("3/4 (75%)", stdout.getvalue())

    def test_failure_stops_and_reports_code_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            args = self._args(tmp_path, self._data(tmp_path))
            calls: list[tuple[str, list[str]]] = []
            stdout = StringIO()
            stderr = StringIO()
            with mock.patch.object(ee, "missing_modules", return_value=[]), \
                    mock.patch.object(ee, "run_streamed",
                                      side_effect=self._fake(calls, fail_prefix="int8")), \
                    redirect_stdout(stdout), redirect_stderr(stderr):
                code = ee.run_experiment(args)
            self.assertEqual(code, 1)
            self.assertIn("qat.train вернул код 1", stderr.getvalue())
            self.assertEqual(
                [prefix for prefix, _ in calls if prefix in ee.MODES],
                ["none", "int8"],
            )
            summary = json.loads((tmp_path / "emb" / "summary.json").read_text(
                encoding="utf-8"))
            self.assertEqual([run["status"] for run in summary["runs"]],
                             ["done", "failed"])

    def test_resume_skips_done_run_and_keeps_its_hours(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            out = tmp_path / "emb"
            done = out / "none"
            done.mkdir(parents=True)
            (done / "log.jsonl").write_text(
                json.dumps({"loss": 2.0}) + "\n", encoding="utf-8")
            (done / "DONE").write_text("step=4\n", encoding="utf-8")
            (out / "summary.json").write_text(
                json.dumps({"runs": [{"mode": "none", "hours": 5.0}]}),
                encoding="utf-8",
            )
            args = self._args(tmp_path, self._data(tmp_path), resume=True)
            calls: list[tuple[str, list[str]]] = []
            stdout = StringIO()
            with mock.patch.object(ee, "missing_modules", return_value=[]), \
                    mock.patch.object(ee, "run_streamed", side_effect=self._fake(calls)), \
                    redirect_stdout(stdout):
                code = ee.run_experiment(args)
            self.assertEqual(code, 0)
            self.assertEqual(
                [prefix for prefix, _ in calls if prefix in ee.MODES],
                ["int8", "trit"],
            )
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            none_run = next(run for run in summary["runs"] if run["mode"] == "none")
            self.assertEqual(none_run["hours"], 5.0)


class TestRealRunWithoutTorch(unittest.TestCase):
    def test_clear_error_exit_two(self):
        import importlib.util
        if importlib.util.find_spec("torch") is not None:
            self.skipTest("torch установлен — отказ без GPU-стека не проверить")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "emb"
            proc = run_tool("--out", str(out), "--data", str(QAT_DATA),
                            "--max-steps", "3")
            self.assertEqual(proc.returncode, 2, proc.stdout)
            combined = proc.stdout + proc.stderr
            self.assertIn("torch", combined)
            self.assertIn("pip install -r qat/requirements-qat.txt", combined)
            self.assertIn("--dry-run", combined)
            self.assertFalse(out.exists(), "до проверки зависимостей каталоги не создаются")


if __name__ == "__main__":
    unittest.main()
