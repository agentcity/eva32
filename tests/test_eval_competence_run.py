"""Раннер golden: мок-прогон, ретраи, сохранение ответов, таблица и провалы.

Тесты не ходят в сеть: живые бэкенды подменяются моком или заглушкой транспорта.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tools.eval_competence as ec  # noqa: E402
import tools.eval_competence_run as run  # noqa: E402


def _call_main(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = run.main(argv)
    return code, out.getvalue(), err.getvalue()


class TestMockRun(unittest.TestCase):
    def test_mock_passes_all_rows_and_writes_answers(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "answers.jsonl"
            code, text, err = _call_main([
                "--backend", "mock", "--out", str(out), "--workers", "2", "--quiet",
            ])
            self.assertEqual(code, 0, err)
            self.assertIn("72", text)
            self.assertIn("100.0%", text)
            self.assertIn("провалов нет", text)
            self.assertIn("мок-бэкенд", text)

            rows = ec.load_golden()
            records = [
                json.loads(line)
                for line in out.read_text(encoding="utf-8").splitlines() if line.strip()
            ]
            self.assertEqual(len(records), len(rows))
            for record in records:
                with self.subTest(category=record["category"]):
                    self.assertTrue(record["ok"], record["failures"])
                    self.assertEqual(record["error"], "")
                    self.assertIn("answer", record)
                    self.assertTrue(record["answer"])
            # порядок строк — как в golden
            self.assertEqual([r["question"] for r in records],
                             [row["question"] for row in rows])

            # файл ответов читается штатным оценщиком без правок
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(ec.main(["--answers", str(out)]), 0)

    def test_mock_serves_every_category(self):
        rows = ec.load_golden()
        numbers = ec.catalog_numbers()
        records = run.run_rows(
            rows, lambda row: (run.mock_answer(row), 1), workers=1, numbers=numbers,
            backend="mock", model="mock",
        )
        report = ec.evaluate(rows, {r["question"]: r["answer"] for r in records})
        self.assertEqual(set(report["per_category"]), {row["category"] for row in rows})
        for category, stat in report["per_category"].items():
            with self.subTest(category=category):
                self.assertEqual(stat["прошли"], stat["всего"])
        self.assertEqual(report["failures"], [])

    def test_limit_and_record_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "answers.jsonl"
            code, _, err = _call_main([
                "--backend", "mock", "--limit", "5", "--out", str(out), "--quiet",
            ])
            self.assertEqual(code, 0, err)
            records = [json.loads(line) for line in out.read_text().splitlines()]
            self.assertEqual(len(records), 5)
            self.assertEqual(set(records[0]), set(run.RECORD_FIELDS))
            self.assertEqual(records[0]["backend"], "mock")

    def test_strict_returns_one_on_failure(self):
        rows = [{"question": "q", "category": "clarify",
                 "checks": {"asks_question": True}}]
        with tempfile.TemporaryDirectory() as tmp:
            golden = pathlib.Path(tmp) / "golden.jsonl"
            golden.write_text(json.dumps(rows[0], ensure_ascii=False) + "\n",
                              encoding="utf-8")
            out = pathlib.Path(tmp) / "answers.jsonl"
            with mock.patch.object(run, "build_ask",
                                   return_value=(lambda row: ("да", 1), lambda: None,
                                                 "x", "")):
                code, text, _ = _call_main([
                    "--backend", "mock", "--golden", str(golden), "--out", str(out),
                    "--strict",
                ])
            self.assertEqual(code, 1)
            self.assertIn("провалы (1)", text)
            self.assertIn("не задан уточняющий вопрос", text)


class TestTable(unittest.TestCase):
    def test_table_has_categories_and_total(self):
        report = {"per_category": {
            "reasoning": {"прошли": 8, "всего": 8},
            "clarify": {"прошли": 2, "всего": 8},
        }, "failures": []}
        table = run.render_table(report)
        self.assertIn("reasoning", table)
        self.assertIn("100.0%", table)
        self.assertIn("25.0%", table)
        self.assertIn("итого", table)
        self.assertIn("10", table)

    def test_table_survives_empty_report(self):
        self.assertIn("итого", run.render_table({"per_category": {}}))

    def test_failures_list_has_reasons(self):
        records = [{
            "question": "Напиши статью", "category": "clarify", "ok": False,
            "failures": ["не задан уточняющий вопрос"],
        }]
        text = run.render_failures(records)
        self.assertIn("[clarify]", text)
        self.assertIn("Напиши статью", text)
        self.assertIn("не задан уточняющий вопрос", text)
        self.assertEqual(run.render_failures([]), "провалов нет")


class TestRetries(unittest.TestCase):
    def test_transient_error_is_retried(self):
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise run.TransientError("таймаут")
            return "готово"

        text, attempts = run._retry(flaky, retries=2, sleep=lambda _: None)
        self.assertEqual(text, "готово")
        self.assertEqual(attempts, 3)

    def test_gives_up_with_clear_message(self):
        def always():
            raise run.TransientError("таймаут")

        with self.assertRaises(run.BackendError) as ctx:
            run._retry(always, retries=1, sleep=lambda _: None, label="ollama:x")
        self.assertIn("ollama:x", str(ctx.exception))
        self.assertIn("2 попыт", str(ctx.exception))
        self.assertIn("таймаут", str(ctx.exception))

    def test_permanent_error_is_not_retried(self):
        calls = {"n": 0}

        def missing_model():
            calls["n"] += 1
            raise run.BackendError("HTTP 404: model not found")

        with self.assertRaises(run.BackendError):
            run._retry(missing_model, retries=3, sleep=lambda _: None)
        self.assertEqual(calls["n"], 1)


class TestBackendErrors(unittest.TestCase):
    def test_ollama_down_is_clear_not_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(run, "get_json",
                                   side_effect=run.TransientError("нет соединения")):
                code, _, err = _call_main([
                    "--backend", "ollama", "--model", "qwen2.5:7b",
                    "--out", str(pathlib.Path(tmp) / "a.jsonl"),
                ])
        self.assertEqual(code, 2)
        self.assertIn("ollama serve", err)
        self.assertIn("Ollama не отвечает", err)
        self.assertNotIn("Traceback", err)

    def test_ollama_model_absent_lists_available(self):
        with mock.patch.object(
            run, "get_json",
            return_value={"models": [{"name": "llama3:8b"}]},
        ):
            with self.assertRaises(run.BackendError) as ctx:
                run.preflight_ollama("http://127.0.0.1:11434", "qwen2.5:7b")
        message = str(ctx.exception)
        self.assertIn("qwen2.5:7b", message)
        self.assertIn("llama3:8b", message)
        self.assertIn("ollama pull", message)

    def test_openai_down_is_clear(self):
        with mock.patch.object(run, "get_json_with_headers",
                               side_effect=run.TransientError("нет соединения")):
            with self.assertRaises(run.BackendError) as ctx:
                run.preflight_openai("http://127.0.0.1:8000", "eva32")
        self.assertIn("OpenAI-эндпоинт не отвечает", str(ctx.exception))

    def test_openai_without_models_route_still_ok(self):
        with mock.patch.object(run, "get_json_with_headers",
                               side_effect=run.BackendError("HTTP 404", status=404)):
            run.preflight_openai("http://127.0.0.1:8000", "eva32")  # не падает

    def test_vtm_without_tokenizer_is_clear(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = pathlib.Path(tmp) / "model.vtm"
            model.write_bytes(b"not a real model")
            with mock.patch.dict(os.environ, {
                "EVA_VTM_MODEL": str(model), "EVA_VTM_TOKENIZER": "",
            }):
                with self.assertRaises(run.BackendError) as ctx:
                    run.preflight_vtm(None, 8)
        self.assertIn("tokenizer", str(ctx.exception).lower())
        self.assertIn("EVA_VTM_TOKENIZER", str(ctx.exception))

    def test_backend_error_on_a_row_is_recorded_not_raised(self):
        rows = [{"question": "q", "category": "reasoning", "checks": {}}]

        def ask(row):
            raise run.BackendError("HTTP 500: сервер упал")

        records = run.run_rows(rows, ask, workers=1, numbers=set(),
                               backend="openai", model="x")
        self.assertEqual(len(records), 1)
        self.assertFalse(records[0]["ok"])
        self.assertIn("HTTP 500", records[0]["error"])
        self.assertEqual(records[0]["answer"], "")
        self.assertTrue(any("ошибка бэкенда" in f for f in records[0]["failures"]))

    def test_all_requests_failed_exits_two(self):
        rows = [{"question": "q", "category": "reasoning", "checks": {}}]

        def ask(row):
            raise run.BackendError("нет модели")

        with tempfile.TemporaryDirectory() as tmp:
            golden = pathlib.Path(tmp) / "golden.jsonl"
            golden.write_text(json.dumps(rows[0], ensure_ascii=False) + "\n",
                              encoding="utf-8")
            with mock.patch.object(run, "build_ask",
                                   return_value=(ask, lambda: None, "x", "")):
                code, text, err = _call_main([
                    "--backend", "mock", "--golden", str(golden), "--quiet",
                    "--out", str(pathlib.Path(tmp) / "a.jsonl"),
                ])
        self.assertEqual(code, 2)
        self.assertIn("ни один запрос не удался", err)
        self.assertIn("нет модели", text)


class TestOutPath(unittest.TestCase):
    def test_default_out_is_safe_and_timestamped(self):
        path = run.default_out("ollama", "qwen2.5:7b")
        self.assertEqual(path.parent, run.OUT_DIR)
        self.assertRegex(path.name, r"^ollama-qwen2\.5-7b-\d{8}-\d{6}\.jsonl$")

    def test_unknown_backend_choice_is_rejected(self):
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                run.parse_args(["--backend", "http"])


if __name__ == "__main__":
    unittest.main()
