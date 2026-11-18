"""Проверки живого замера: формат набора, сравнение фактов, отчёт без сети.

Запуск::

    python3 -m unittest tests.test_eval_live -v

Никакой сети и никаких денег: в тестах клиент большой модели подменяется
заготовкой, а ``--dry-run`` вообще не создаёт клиентов. Живой прогон
запускается только вручную без ``--dry-run``.
"""
from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

GOLDEN = os.path.join(ROOT, "eval", "live_golden.jsonl")
TOOL = os.path.join(ROOT, "tools", "eval_live.py")
ENV_FILE = os.path.join(ROOT, ".env")


def _load_tool():
    """Загрузить tools/eval_live.py по пути (tools не пакет)."""
    spec = importlib.util.spec_from_file_location("eval_live_tool", TOOL)
    if spec is None or spec.loader is None:  # pragma: no cover - защита от кривого пути
        raise RuntimeError(f"не удалось загрузить {TOOL}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


eval_live = _load_tool()


class FakeInnerClient:
    """Клиент большой модели без сети: отдаёт заготовленные ответы."""

    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = []

    def chat(self, messages, max_tokens=512):
        self.calls.append({"messages": list(messages), "max_tokens": max_tokens})
        if not self.texts:
            raise AssertionError("заготовленные ответы кончились")
        text = self.texts.pop(0)
        return {
            "text": text,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost_rub": 0.001},
            "raw": {
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "cost_rub": 0.001,
                }
            },
        }


class FailingInnerClient:
    """Клиент, который всегда падает: проверяем аварийную остановку."""

    def __init__(self):
        self.calls = []

    def chat(self, messages, max_tokens=512):
        self.calls.append({"messages": list(messages), "max_tokens": max_tokens})
        raise RuntimeError("сеть недоступна (заготовка)")


def _write_lines(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


class TestGoldenSet(unittest.TestCase):
    """Набор: ровно 20 вопросов, 10 каталожных и 10 общих."""

    @classmethod
    def setUpClass(cls):
        cls.rows = eval_live.load_golden(GOLDEN)

    def test_size_is_20(self):
        self.assertEqual(len(self.rows), 20)
        self.assertEqual(eval_live.set_counts(self.rows), {"catalog": 10, "general": 10})

    def test_fields_are_complete(self):
        for row in self.rows:
            self.assertTrue(row["id"])
            self.assertTrue(row["question"].strip())
            self.assertIn(row["kind"], eval_live.KINDS)
            self.assertEqual(row["expect_route"], eval_live.EXPECTED_ROUTE[row["kind"]])
            self.assertIsInstance(row["facts"], list)
            self.assertTrue(row["facts"], f"нет фактов у {row['id']}")

    def test_ids_and_questions_are_unique(self):
        ids = [row["id"] for row in self.rows]
        questions = [row["question"] for row in self.rows]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(questions), len(set(questions)))

    def test_catalog_rows_go_to_skill(self):
        catalog = [row for row in self.rows if row["kind"] == "catalog"]
        self.assertEqual(len(catalog), 10)
        self.assertTrue(all(row["expect_route"] == "skill" for row in catalog))

    def test_general_rows_go_to_big(self):
        general = [row for row in self.rows if row["kind"] == "general"]
        self.assertEqual(len(general), 10)
        self.assertTrue(all(row["expect_route"] == "big" for row in general))

    def test_facts_are_not_echoed_in_questions(self):
        # Иначе «попадание» ничего не проверяет: ответ уже есть в вопросе.
        for row in self.rows:
            self.assertEqual(
                eval_live.facts_echoed_in_question(row["question"], row["facts"]),
                [],
                f"{row['id']}: эталон встречается в вопросе",
            )


class TestFactMatching(unittest.TestCase):
    """Попадание факта: регистр, ё, пробелы и границы чисел."""

    def test_case_yo_and_spaces(self):
        self.assertTrue(eval_live.fact_in("Тариф «Старт» — 2 490 ₽/мес", "2490"))
        self.assertTrue(eval_live.fact_in("ВОЗВРАТ — 14 ДНЕЙ", "14"))
        self.assertTrue(eval_live.fact_in("Всё ещё работает", "все еще"))
        self.assertTrue(eval_live.fact_in("Офис: с 9:00 до 19:00", "9:00"))

    def test_numbers_respect_digit_boundaries(self):
        self.assertFalse(eval_live.fact_in("в 1990 году", "990"))
        self.assertFalse(eval_live.fact_in("9900 рублей", "990"))
        self.assertFalse(eval_live.fact_in("с 19:00 до 20:00", "9:00"))
        self.assertTrue(eval_live.fact_in("цена 990 ₽", "990"))
        self.assertTrue(eval_live.fact_in("за 120-32-32", "120-32-32"))

    def test_any_fact_is_enough(self):
        facts = ["атомарн", "неделим", "acid"]
        self.assertTrue(eval_live.fact_hit("Транзакция атомарна.", facts))
        self.assertTrue(eval_live.fact_hit("Это неделимая операция.", facts))
        self.assertFalse(eval_live.fact_hit("Транзакция — это запрос.", facts))

    def test_empty_text_or_facts(self):
        self.assertFalse(eval_live.fact_hit("", ["990"]))
        self.assertFalse(eval_live.fact_hit("990", []))
        self.assertFalse(eval_live.fact_hit("990", [""]))

    def test_echo_detection(self):
        self.assertEqual(
            eval_live.facts_echoed_in_question("Сколько стоит Старт за 990?", ["990"]),
            ["990"],
        )
        self.assertEqual(
            eval_live.facts_echoed_in_question("Сколько стоит Старт?", ["990"]), []
        )


class TestLoaderValidation(unittest.TestCase):
    """Ошибки формата должны называть строку, а не молча пропускаться."""

    def _load(self, rows, strict=True):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "live.jsonl")
            _write_lines(path, rows)
            return eval_live.load_golden(path, strict_counts=strict)

    def test_valid_one_row_without_strict(self):
        rows = self._load(
            [
                {
                    "id": "one",
                    "kind": "general",
                    "question": "Почему небо голубое?",
                    "expect_route": "big",
                    "facts": ["рассе"],
                }
            ],
            strict=False,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["expect_route"], "big")

    def test_strict_counts_requires_20(self):
        with self.assertRaises(ValueError) as ctx:
            self._load(
                [
                    {
                        "id": "one",
                        "kind": "catalog",
                        "question": "Сколько стоит тариф «Старт»?",
                        "expect_route": "skill",
                        "facts": ["990"],
                    }
                ]
            )
        self.assertIn("20", str(ctx.exception))

    def test_missing_facts_is_error(self):
        with self.assertRaises(ValueError) as ctx:
            self._load(
                [
                    {
                        "id": "x",
                        "kind": "catalog",
                        "question": "Сколько?",
                        "expect_route": "skill",
                    }
                ],
                strict=False,
            )
        self.assertIn("facts", str(ctx.exception))

    def test_wrong_route_for_kind_is_error(self):
        with self.assertRaises(ValueError) as ctx:
            self._load(
                [
                    {
                        "id": "x",
                        "kind": "catalog",
                        "question": "Сколько?",
                        "expect_route": "big",
                        "facts": ["990"],
                    }
                ],
                strict=False,
            )
        self.assertIn("expect_route", str(ctx.exception))

    def test_duplicate_question_is_error(self):
        row = {
            "id": "x",
            "kind": "catalog",
            "question": "Сколько?",
            "expect_route": "skill",
            "facts": ["990"],
        }
        other = dict(row, id="y")
        with self.assertRaises(ValueError) as ctx:
            self._load([row, other], strict=False)
        self.assertIn("повтор вопроса", str(ctx.exception))

    def test_bad_json_reports_lineno(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "live.jsonl")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("{не json}\n")
            with self.assertRaises(ValueError) as ctx:
                eval_live.load_golden(path, strict_counts=False)
            self.assertIn(":1:", str(ctx.exception))


class TestLiveConfig(unittest.TestCase):
    """TEACHER_* превращаются в большую модель; ключ никуда не печатается."""

    ENV = {
        "TEACHER_BASE": "https://polza.ai/api",
        "TEACHER_KEY": "super-secret-key-123",
        "TEACHER_MODEL": "deepseek-v4.1-flash",
    }

    def test_teacher_maps_to_big(self):
        cfg = eval_live.build_config(self.ENV, dotenv_path=None)
        self.assertEqual(cfg.big_base, "https://polza.ai/api")
        self.assertEqual(cfg.big_key, "super-secret-key-123")
        self.assertEqual(cfg.big_model, "deepseek-v4.1-flash")
        self.assertEqual(eval_live.config_summary(cfg)["provider"], "polza.ai")

    def test_model_has_default(self):
        env = dict(self.ENV)
        env.pop("TEACHER_MODEL")
        cfg = eval_live.build_config(env, dotenv_path=None)
        self.assertEqual(cfg.big_model, eval_live.DEFAULT_MODEL)

    def test_validate_requires_key_and_base(self):
        cfg = eval_live.build_config({}, dotenv_path=None)
        with self.assertRaises(eval_live.EvalError) as ctx:
            eval_live.validate_live_config(cfg)
        self.assertIn("адрес", str(ctx.exception).lower())

        cfg = eval_live.build_config(
            {"TEACHER_BASE": "https://polza.ai/api"}, dotenv_path=None
        )
        with self.assertRaises(eval_live.EvalError) as ctx:
            eval_live.validate_live_config(cfg)
        self.assertIn("ключ", str(ctx.exception).lower())

    def test_summary_has_no_key(self):
        cfg = eval_live.build_config(self.ENV, dotenv_path=None)
        summary = eval_live.config_summary(cfg)
        self.assertNotIn("super-secret-key-123", json.dumps(summary, ensure_ascii=False))

    def test_read_env_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, ".env")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("# comment\nA=1\nexport B='two'\nC=\"three\"\n")
            data = eval_live.read_env_file(path)
        self.assertEqual(data, {"A": "1", "B": "two", "C": "three"})


class TestOfflineLiveRun(unittest.TestCase):
    """Живой путь через настоящий router.answer, но с подменённым клиентом."""

    CFG_ENV = {
        "TEACHER_BASE": "https://polza.ai/api",
        "TEACHER_KEY": "super-secret-key-123",
        "TEACHER_MODEL": "deepseek-v4.1-flash",
    }

    def _cfg(self):
        return eval_live.build_config(self.CFG_ENV, dotenv_path=None)

    def test_catalog_and_general_rows(self):
        rows = [
            {
                "id": "c1",
                "kind": "catalog",
                "question": "Сколько стоит тариф «Старт»?",
                "expect_route": "skill",
                "facts": ["990"],
            },
            {
                "id": "c2",
                "kind": "catalog",
                "question": "Какие часы работы офиса?",
                "expect_route": "skill",
                "facts": ["9:00"],
            },
            {
                "id": "g1",
                "kind": "general",
                "question": "Почему небо голубое?",
                "expect_route": "big",
                "facts": ["рассе"],
            },
            {
                "id": "g2",
                "kind": "general",
                "question": "Посчитай 17% от 4800.",
                "expect_route": "big",
                "facts": ["816"],
            },
        ]
        inner = FakeInnerClient(["Рассеяние света в атмосфере.", "816."])
        big = eval_live.MeteredBigClient(inner)
        metrics = eval_live.run_live(
            rows, self._cfg(), big_client=big, system_text="", max_model_errors=2
        )

        self.assertEqual(metrics["total"], 4)
        self.assertEqual(metrics["route_ok"], 4)
        self.assertEqual(metrics["fact_ok"], 4)
        self.assertEqual(metrics["both_ok"], 4)
        self.assertEqual(metrics["by_kind"]["catalog"]["fact_ok"], 2)
        self.assertEqual(metrics["by_kind"]["general"]["fact_ok"], 2)
        self.assertEqual(metrics["errors"], 0)
        self.assertEqual(metrics["billed_calls"], 2)
        self.assertEqual(metrics["api_calls"], 2)
        self.assertAlmostEqual(metrics["cost_rub"], 0.002, places=9)
        self.assertAlmostEqual(metrics["api_cost_rub"], 0.002, places=9)
        self.assertEqual(metrics["route_counts"]["skill"], 2)
        self.assertEqual(metrics["route_counts"]["big"], 2)
        self.assertEqual(metrics["small_unavailable"], 0)
        # Каталожные ответы моделью не считаются: два HTTP-вызова, не четыре.
        self.assertEqual(len(inner.calls), 2)
        self.assertEqual(inner.calls[0]["messages"][-1].content, "Почему небо голубое?")

    def test_small_route_is_marked_unavailable(self):
        rows = [
            {
                "id": "s1",
                "kind": "general",
                "question": "Привет!",
                "expect_route": "big",
                "facts": ["привет"],
            }
        ]
        inner = FakeInnerClient(["Привет! Чем помочь?"])
        big = eval_live.MeteredBigClient(inner)
        metrics = eval_live.run_live(
            rows, self._cfg(), big_client=big, system_text="", max_model_errors=1
        )

        record = metrics["records"][0]
        self.assertEqual(record["planned_route"], "small")
        self.assertEqual(record["route"], "big")  # фолбэк small→big
        self.assertTrue(record["small_unavailable"])
        self.assertIn("малая недоступна", record["reason"])
        self.assertEqual(metrics["small_unavailable"], 1)
        self.assertEqual(metrics["small_calls"], 1)
        self.assertEqual(metrics["route_ok"], 1)
        self.assertEqual(metrics["fact_ok"], 1)

    def test_model_errors_stop_run(self):
        rows = [
            {
                "id": "g1",
                "kind": "general",
                "question": "Почему небо голубое?",
                "expect_route": "big",
                "facts": ["рассе"],
            }
        ]
        big = eval_live.MeteredBigClient(FailingInnerClient())
        with self.assertRaises(eval_live.EvalError) as ctx:
            eval_live.run_live(
                rows, self._cfg(), big_client=big, system_text="", max_model_errors=1
            )
        self.assertIn("остановлен", str(ctx.exception))
        self.assertIn("потрачено", str(ctx.exception))

    def test_report_has_cost_and_no_key(self):
        rows = [
            {
                "id": "g1",
                "kind": "general",
                "question": "Почему небо голубое?",
                "expect_route": "big",
                "facts": ["рассе"],
            }
        ]
        inner = FakeInnerClient(["Рассеяние света."])
        metrics = eval_live.run_live(
            rows,
            self._cfg(),
            big_client=eval_live.MeteredBigClient(inner),
            system_text="",
        )
        metrics["provider"] = "polza.ai"
        metrics["model"] = "deepseek-v4.1-flash"
        report = eval_live.format_report(metrics)
        self.assertIn("потрачено:", report)
        self.assertIn("0.0010", report)
        self.assertIn("попадание факта: 1/1", report)
        self.assertNotIn("super-secret-key-123", report)

    def test_fact_miss_is_reported(self):
        rows = [
            {
                "id": "g1",
                "kind": "general",
                "question": "Почему небо голубое?",
                "expect_route": "big",
                "facts": ["рассе"],
            }
        ]
        inner = FakeInnerClient(["Потому что так красиво."])
        metrics = eval_live.run_live(
            rows,
            self._cfg(),
            big_client=eval_live.MeteredBigClient(inner),
            system_text="",
        )
        report = eval_live.format_report(metrics)
        self.assertEqual(metrics["fact_ok"], 0)
        self.assertIn("промахи факта", report)
        self.assertIn("рассе", report)


class TestDryRunCLI(unittest.TestCase):
    """--dry-run: формат и офлайн-предпросмотр, без сети и без клиентов."""

    def test_dry_run_ok(self):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = eval_live.main(["--dry-run"])
        text = out.getvalue()
        self.assertEqual(code, 0, err.getvalue())
        self.assertIn("dry-run", text)
        self.assertIn("20 вопросов", text)
        self.assertIn("каталожных 10, общих 10", text)
        self.assertIn("совпало 20/20", text)
        self.assertIn("сеть не используется", text)
        self.assertEqual(err.getvalue(), "")

    def test_dry_run_does_not_leak_key(self):
        secret = eval_live.read_env_file(ENV_FILE).get("TEACHER_KEY", "")
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            code = eval_live.main(["--dry-run"])
        self.assertEqual(code, 0)
        if secret:
            self.assertNotIn(secret, out.getvalue())

    def test_dry_run_with_limit_keeps_full_format_check(self):
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            code = eval_live.main(["--dry-run", "--limit", "2"])
        text = out.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("набор", text)
        self.assertIn("первые 2 вопросов", text)

    def test_dry_run_no_preview(self):
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            code = eval_live.main(["--dry-run", "--no-preview"])
        self.assertEqual(code, 0)
        self.assertIn("предпросмотр маршрутов: пропущен", out.getvalue())


if __name__ == "__main__":
    unittest.main()
