"""Проверки бенчмарка гибрида: перцентили, разбор, offline-прогон, отчёт.

Запуск::

    python3 -m unittest tests.test_bench_eva -v

Никакой сети: offline-режим подменяет клиентов моделей заглушками, а HTTP-часть
проверяется на чистых функциях (``chat_url``, ``parse_http_payload``). Прогон
``run_offline`` пропускается с внятной причиной, если модули ``eva`` ещё не
написаны (их пишут параллельно); остальные проверки работают всегда.

Числа перцентилей сверяются с ``eva.telemetry.percentile``, чтобы p95 в
бенчмарке и в ``tools/usage_report.py`` считались одинаково.
"""
from __future__ import annotations

import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

TOOL = os.path.join(ROOT, "tools", "bench_eva.py")
QUESTIONS = os.path.join(ROOT, "eval", "hybrid_golden.jsonl")


def _load_tool():
    """Загрузить tools/bench_eva.py по пути (tools — не пакет)."""
    spec = importlib.util.spec_from_file_location("bench_eva_tool", TOOL)
    if spec is None or spec.loader is None:  # pragma: no cover - защита от кривого пути
        raise RuntimeError(f"не удалось загрузить {TOOL}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench = _load_tool()


def _eva_ready() -> tuple[bool, str]:
    """Есть ли всё для offline-прогона: модули роутера, конфиг, набор."""
    missing = []
    for name in ("eva.types", "eva.config", "eva.router"):
        try:
            __import__(name)
        except Exception as exc:  # noqa: BLE001 — причина попадёт в skip
            return False, f"{name}: {type(exc).__name__}: {exc}"
        if name not in sys.modules:
            missing.append(name)
    if not os.path.exists(QUESTIONS):
        return False, f"нет набора вопросов {QUESTIONS}"
    return True, ""


READY, SKIP_REASON = _eva_ready()

PRICES = {"skill": (0.0, 0.0), "small": (0.0, 0.0), "big": (0.4, 1.2)}
COMPLETION = {"skill": 0, "small": 120, "big": 220}


def _record(
    route: str | None = "small",
    latency: float = 1.0,
    *,
    ok: bool = True,
    timeout: bool = False,
    error: str | None = None,
    cost: float | None = None,
    question: str = "вопрос",
    server_latency: int | None = None,
) -> dict:
    return {
        "question": question,
        "route": route,
        "latency_ms": latency,
        "server_latency_ms": server_latency,
        "cost_rub": cost,
        "ok": ok,
        "timeout": timeout,
        "error": error,
    }


def _metrics(records, **overrides):
    params = {
        "mode": "offline",
        "n": len(records),
        "concurrency": 1,
        "wall_s": 0.5,
        "prices": PRICES,
        "system_tokens": 0,
        "completion_tokens": COMPLETION,
        "warmup": 1,
    }
    params.update(overrides)
    return bench.summarize(records, **params)


class TestParser(unittest.TestCase):
    """Аргументы: режим обязателен, N и параллельность — положительные."""

    def test_offline_defaults(self):
        args = bench.build_parser().parse_args(["--offline"])
        self.assertTrue(args.offline)
        self.assertEqual(args.n, 100)
        self.assertEqual(args.concurrency, 4)
        self.assertEqual(args.warmup, 1)
        self.assertEqual(args.timeout, 30.0)
        self.assertFalse(args.json)
        self.assertEqual(args.questions, QUESTIONS)
        self.assertEqual(args.model, "eva-hybrid")

    def test_http_url(self):
        args = bench.build_parser().parse_args(["--http", "http://127.0.0.1:8010"])
        self.assertEqual(args.http, "http://127.0.0.1:8010")
        self.assertFalse(args.offline)

    def test_mode_is_required_and_exclusive(self):
        with self.assertRaises(SystemExit):
            bench.build_parser().parse_args([])
        with self.assertRaises(SystemExit):
            bench.build_parser().parse_args(
                ["--offline", "--http", "http://127.0.0.1:8010"]
            )

    def test_n_must_be_positive(self):
        for value in ("0", "-3", "abc"):
            with self.subTest(value=value):
                with self.assertRaises(SystemExit):
                    bench.build_parser().parse_args(["--offline", "--n", value])

    def test_concurrency_must_be_positive(self):
        with self.assertRaises(SystemExit):
            bench.build_parser().parse_args(["--offline", "--concurrency", "0"])

    def test_price_overrides(self):
        args = bench.build_parser().parse_args(
            ["--offline", "--big-price-in", "0.2", "--big-price-out", "0.6"]
        )
        prices = bench.resolve_prices(PRICES, args)
        self.assertEqual(prices["big"], (0.2, 0.6))
        self.assertEqual(prices["small"], (0.0, 0.0))


class TestPercentile(unittest.TestCase):
    """Метод R-7: линейная интерполяция, как в eva.telemetry."""

    def test_known_values(self):
        self.assertEqual(bench.percentile([1, 2, 3, 4], 50), 2.5)
        self.assertEqual(bench.percentile([1, 2, 3, 4], 0), 1)
        self.assertEqual(bench.percentile([1, 2, 3, 4], 100), 4)
        self.assertAlmostEqual(bench.percentile(list(range(1, 101)), 95), 95.05)
        self.assertAlmostEqual(bench.percentile(list(range(1, 101)), 50), 50.5)

    def test_empty_and_single(self):
        self.assertIsNone(bench.percentile([], 95))
        self.assertEqual(bench.percentile([7], 95), 7)

    def test_unsorted_input(self):
        self.assertEqual(bench.percentile([5, 1, 4, 2, 3], 50), 3)

    def test_out_of_range(self):
        with self.assertRaises(ValueError):
            bench.percentile([1, 2], -1)
        with self.assertRaises(ValueError):
            bench.percentile([1, 2], 101)

    def test_matches_telemetry(self):
        try:
            from eva.telemetry import percentile as telemetry_percentile
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"eva.telemetry недоступен: {type(exc).__name__}: {exc}")
        values = [3.0, 1.0, 4.0, 1.5, 9.0, 2.0]
        for q in (0, 10, 50, 90, 95, 99, 100):
            with self.subTest(q=q):
                self.assertAlmostEqual(
                    bench.percentile(values, q), telemetry_percentile(values, q), places=9
                )


class TestLatencyStats(unittest.TestCase):
    def test_all_fields(self):
        stats = bench.latency_stats([3.0, 1.0, 2.0])
        self.assertEqual(stats["count"], 3)
        self.assertEqual(stats["p50"], 2.0)
        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["max"], 3.0)
        self.assertIn("p95", stats)
        self.assertIn("p99", stats)

    def test_empty_is_none(self):
        stats = bench.latency_stats([])
        self.assertEqual(stats["count"], 0)
        self.assertIsNone(stats["p50"])
        self.assertIsNone(stats["p99"])


class TestQuestions(unittest.TestCase):
    def test_golden_set_loads(self):
        questions = bench.load_questions(QUESTIONS)
        self.assertGreaterEqual(len(questions), 40)
        self.assertTrue(all(item.strip() for item in questions))

    def test_golden_questions_are_usable(self):
        # Разметка здесь не проверяется, но каждый вопрос должен быть строкой.
        for question in bench.load_questions(QUESTIONS)[:10]:
            self.assertIsInstance(question, str)

    def test_pick_is_evenly_spread(self):
        questions = [f"q{i}" for i in range(60)]
        picked = bench.pick_questions(questions, 5)
        self.assertEqual(picked, ["q0", "q12", "q24", "q36", "q48"])

    def test_pick_wraps_when_n_larger(self):
        picked = bench.pick_questions(["a", "b", "c"], 7)
        self.assertEqual(len(picked), 7)
        self.assertTrue(set(picked) <= {"a", "b", "c"})

    def test_pick_rejects_bad_n(self):
        with self.assertRaises(ValueError):
            bench.pick_questions(["a"], 0)
        with self.assertRaises(ValueError):
            bench.pick_questions([], 3)

    def test_loader_allows_plain_strings_and_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "q.jsonl")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write('{"question": "Сколько стоит тариф Старт?"}\n')
                fh.write("# комментарий\n")
                fh.write('"просто строка"\n')
                fh.write("\n")
            self.assertEqual(
                bench.load_questions(path),
                ["Сколько стоит тариф Старт?", "просто строка"],
            )

            with open(path, "w", encoding="utf-8") as fh:
                fh.write("{битый json}\n")
            with self.assertRaises(ValueError):
                bench.load_questions(path)

            with open(path, "w", encoding="utf-8") as fh:
                fh.write('{"question": "   "}\n')
            with self.assertRaises(ValueError):
                bench.load_questions(path)

            with open(path, "w", encoding="utf-8") as fh:
                fh.write("# только комментарий\n")
            with self.assertRaises(ValueError):
                bench.load_questions(path)


class TestPriceModel(unittest.TestCase):
    """Формула цены на 1000 — та же, что в tools/eval_hybrid.py."""

    def test_skill_and_small_are_free(self):
        self.assertEqual(bench.request_cost("skill", 10_000, PRICES), 0.0)
        self.assertEqual(bench.request_cost("small", 10_000, PRICES), 0.0)

    def test_big_request_cost(self):
        cost = bench.request_cost("big", 700, PRICES, COMPLETION)
        self.assertAlmostEqual(cost, 0.7 * 0.4 + 0.22 * 1.2, places=9)

    def test_1000_requests_mix(self):
        actual = bench.cost_per_1000(
            ["skill", "small", "big"], ["", "", "x" * 300], PRICES, 600, COMPLETION
        )
        self.assertAlmostEqual(actual, 181.3333333333, places=6)

    def test_unknown_route_priced_as_big(self):
        unknown = bench.cost_per_1000(["unknown"], ["x" * 300], PRICES, 600, COMPLETION)
        big = bench.cost_per_1000(["big"], ["x" * 300], PRICES, 600, COMPLETION)
        self.assertEqual(unknown, big)

    def test_estimate_tokens(self):
        self.assertEqual(bench.estimate_tokens(""), 0)
        self.assertEqual(bench.estimate_tokens("   "), 0)
        self.assertEqual(bench.estimate_tokens("да"), 1)
        self.assertEqual(bench.estimate_tokens("x" * 300), 100)


class TestHttpHelpers(unittest.TestCase):
    def test_chat_url_variants(self):
        base = "http://127.0.0.1:8010"
        expected = base + "/v1/chat/completions"
        self.assertEqual(bench.chat_url(base), expected)
        self.assertEqual(bench.chat_url(base + "/"), expected)
        self.assertEqual(bench.chat_url(base + "/v1"), expected)
        self.assertEqual(bench.chat_url(expected), expected)
        self.assertEqual(
            bench.chat_url("https://example.test/eva"), "https://example.test/eva/v1/chat/completions"
        )

    def test_chat_url_rejects_bad_input(self):
        for value in ("", "127.0.0.1:8010", "ftp://host"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    bench.chat_url(value)

    def test_parse_payload_with_eva(self):
        payload = {
            "choices": [{"message": {"role": "assistant", "content": "990 ₽"}}],
            "eva": {"route": "Skill", "cost_rub": 0.0, "latency_ms": 7},
        }
        route, cost, server_latency, text = bench.parse_http_payload(payload)
        self.assertEqual(route, "skill")
        self.assertEqual(cost, 0.0)
        self.assertEqual(server_latency, 7)
        self.assertEqual(text, "990 ₽")

    def test_parse_payload_without_eva(self):
        payload = {"choices": [{"message": {"content": "ответ"}}]}
        route, cost, server_latency, text = bench.parse_http_payload(payload)
        self.assertEqual(route, "unknown")
        self.assertIsNone(cost)
        self.assertIsNone(server_latency)
        self.assertEqual(text, "ответ")

    def test_parse_payload_rejects_garbage(self):
        with self.assertRaises(ValueError):
            bench.parse_http_payload(["не объект"])
        with self.assertRaises(ValueError):
            bench.parse_http_payload({"choices": []})
        with self.assertRaises(ValueError):
            bench.parse_http_payload({"choices": [{"message": {"content": "  "}}]})


class TestBudget(unittest.TestCase):
    def test_budget_from_docs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "hybrid.md")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("# Гибрид\n\nПросто упоминание p95: 100 мс, но не порог.\n")
                fh.write("Бюджет латентности: p95 ≤ 250 мс.\n")
            self.assertEqual(bench.budget_from_docs(path), 250.0)

            with open(path, "w", encoding="utf-8") as fh:
                fh.write("# Гибрид\n\nВ тексте p95 упомянут, но бюджета нет.\n")
            self.assertIsNone(bench.budget_from_docs(path))

            with open(path, "w", encoding="utf-8") as fh:
                fh.write("Бюджет: p95 ≤ abc мс\n")
            self.assertIsNone(bench.budget_from_docs(path))

        self.assertIsNone(bench.budget_from_docs(os.path.join(ROOT, "нет-такого.md")))

    def test_default_budget_is_none(self):
        # В docs/hybrid.md бюджета p95 нет; если появится — подхватится из файла.
        self.assertIsNone(bench.DEFAULT_P95_BUDGET_MS)


class TestSummarize(unittest.TestCase):
    """Сводка: доли, перцентили по успешным, ошибки и таймауты отдельно."""

    def setUp(self):
        self.records = [
            _record("skill", 1.0, cost=0.0),
            _record("small", 10.0, cost=0.0),
            _record("big", 30.0, cost=0.001),
            _record(None, 20.0, ok=False, error="RuntimeError: бум"),
            _record(None, 200.0, ok=False, timeout=True, error="таймаут"),
        ]
        self.metrics = _metrics(self.records, wall_s=0.5, n=5)

    def test_counts(self):
        self.assertEqual(self.metrics["ok"], 3)
        self.assertEqual(self.metrics["errors"], 1)
        self.assertEqual(self.metrics["timeouts"], 1)
        self.assertEqual(self.metrics["total"], 5)

    def test_latency_uses_only_successful(self):
        lat = self.metrics["latency_ms"]
        self.assertEqual(lat["count"], 3)
        self.assertEqual(lat["p50"], 10.0)
        self.assertAlmostEqual(lat["p95"], 28.0)
        self.assertAlmostEqual(lat["p99"], 29.6)
        self.assertEqual(lat["min"], 1.0)
        self.assertEqual(lat["max"], 30.0)

    def test_route_shares_over_successful(self):
        self.assertEqual(self.metrics["routes"]["skill"], 1)
        self.assertEqual(self.metrics["routes"]["small"], 1)
        self.assertEqual(self.metrics["routes"]["big"], 1)
        self.assertEqual(self.metrics["routes"]["other"], 0)
        self.assertAlmostEqual(self.metrics["shares"]["skill"], 1 / 3)

    def test_throughput(self):
        self.assertEqual(self.metrics["throughput_rps"], 10.0)
        self.assertEqual(self.metrics["ok_throughput_rps"], 6.0)

    def test_cost_and_reported(self):
        self.assertGreater(self.metrics["cost_per_1000_rub"], 0.0)
        # Средняя по ответам: (0 + 0 + 0.001) / 3 * 1000.
        self.assertAlmostEqual(self.metrics["reported_cost_per_1000_rub"], 1 / 3, places=4)

    def test_budget_verdicts(self):
        below = _metrics(self.records, p95_budget_ms=5.0)
        above = _metrics(self.records, p95_budget_ms=100.0)
        self.assertFalse(below["budget_ok"])
        self.assertTrue(above["budget_ok"])
        self.assertIsNone(self.metrics["budget_ok"])
        self.assertIsNone(self.metrics["p95_budget_ms"])

    def test_all_failed_is_not_a_crash(self):
        records = [
            _record(None, 5.0, ok=False, error="URLError"),
            _record(None, 6.0, ok=False, timeout=True),
        ]
        metrics = _metrics(records)
        self.assertEqual(metrics["ok"], 0)
        self.assertEqual(metrics["errors"], 1)
        self.assertEqual(metrics["timeouts"], 1)
        self.assertIsNone(metrics["latency_ms"]["p50"])
        self.assertEqual(metrics["throughput_rps"], 4.0)

    def test_latency_by_route(self):
        by_route = self.metrics["latency_by_route_ms"]
        self.assertEqual(by_route["skill"]["p50"], 1.0)
        self.assertEqual(by_route["big"]["p50"], 30.0)


class TestReport(unittest.TestCase):
    def setUp(self):
        records = [
            _record("skill", 1.0, cost=0.0),
            _record("small", 10.0, cost=0.0),
            _record("big", 30.0, cost=0.001),
            _record(None, 50.0, ok=False, error="URLError: refused"),
        ]
        self.metrics = _metrics(records, wall_s=0.25, n=4)

    def test_table_has_percentiles_and_cost(self):
        report = bench.format_report(self.metrics)
        self.assertIn("offline", report)
        self.assertIn("p50", report)
        self.assertIn("p95", report)
        self.assertIn("p99", report)
        self.assertIn("₽ за 1000", report)
        self.assertIn("маршруты:", report)
        self.assertIn("ошибки 1", report)
        self.assertIn("бюджет p95: не задан", report)

    def test_table_with_budget(self):
        metrics = _metrics(self.metrics["records"], p95_budget_ms=5.0)
        report = bench.format_report(metrics)
        self.assertIn("ПРЕВЫШЕН", report)
        metrics = _metrics(self.metrics["records"], p95_budget_ms=1000.0)
        self.assertIn("— ok", bench.format_report(metrics))

    def test_json_is_machine_readable(self):
        parsed = json.loads(bench.format_json(self.metrics))
        self.assertEqual(parsed["ok"], 3)
        self.assertIn("p95", parsed["latency_ms"])
        self.assertEqual(len(parsed["records"]), 4)
        self.assertIsInstance(parsed["prices_rub_per_1k"]["big"], list)


@unittest.skipUnless(READY, SKIP_REASON)
class TestOfflineRun(unittest.TestCase):
    """Offline-прогон на заглушках: без сети, быстро и без падений."""

    def test_small_run(self):
        questions = bench.load_questions(QUESTIONS)
        metrics = bench.run_offline(questions, 5, 2, warmup=0)
        self.assertEqual(metrics["ok"], 5)
        self.assertEqual(metrics["errors"], 0)
        self.assertEqual(metrics["timeouts"], 0)
        self.assertEqual(metrics["total"], 5)
        self.assertIsNotNone(metrics["latency_ms"]["p50"])
        self.assertGreater(metrics["wall_s"], 0.0)
        self.assertGreater(metrics["cost_per_1000_rub"], 0.0)
        self.assertEqual(
            sum(metrics["routes"][kind] for kind in ("skill", "small", "big", "other")),
            5,
        )

    def test_route_mix_is_not_single_route(self):
        # Равномерный выбор вопросов должен попасть хотя бы в два маршрута.
        questions = bench.load_questions(QUESTIONS)
        metrics = bench.run_offline(questions, 10, 1, warmup=1)
        seen = [kind for kind in ("skill", "small", "big") if metrics["routes"][kind]]
        self.assertGreaterEqual(len(seen), 2, metrics["routes"])

    def test_timeouts_are_counted_separately(self):
        # Заглушка спит 20 мс, таймаут 1 мс: вызов завершится, но опоздает.
        questions = ["Почему небо голубое? Объясни подробно"]
        metrics = bench.run_offline(
            questions, 2, 2, warmup=0, timeout=0.001, stub_latency_ms=20.0
        )
        self.assertEqual(metrics["ok"], 0)
        self.assertEqual(metrics["timeouts"], 2)
        self.assertEqual(metrics["errors"], 0)
        self.assertIsNone(metrics["latency_ms"]["p50"])

    def test_json_through_main(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = bench.main(
                ["--offline", "--n", "3", "--concurrency", "2", "--warmup", "0", "--json"]
            )
        self.assertEqual(code, 0)
        parsed = json.loads(out.getvalue())
        self.assertEqual(parsed["mode"], "offline")
        self.assertEqual(parsed["ok"], 3)

    def test_cli_subprocess(self):
        result = subprocess.run(
            [
                sys.executable,
                TOOL,
                "--offline",
                "--n",
                "5",
                "--concurrency",
                "2",
                "--warmup",
                "0",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("p50", result.stdout)
        self.assertIn("p95", result.stdout)
        self.assertIn("p99", result.stdout)
        self.assertIn("₽ за 1000", result.stdout)
        self.assertIn("маршруты:", result.stdout)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
