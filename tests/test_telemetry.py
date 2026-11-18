"""Тесты телеметрии гибрида: запись JSONL, агрегат, отчёт.

Запуск::

    python3 -m unittest tests.test_telemetry -v

Ручная сверка 100 записей (test_100_records_totals_and_percentiles):

    маршрут  записей  цена/шт  латентности   сумма ₽  p50      p95
    skill         20   0.0000   1..20          0.00   10.50    19.05
    small         40   0.0010   100..139        0.04  119.50   137.05
    big           40   0.0200   500..539        0.80  519.50   537.05
    ИТОГО        100             1..20,100..139,500..539
                                            0.84 ₽  = 8.40 ₽ / 1000
    сумма латентностей: 210 + 4 780 + 20 780 = 25 770 мс; средняя 257.70 мс
    общий p50: позиция (100-1)*0.5 = 49.5 → (129+130)/2 = 129.5
    общий p95: позиция (100-1)*0.95 = 94.05 → 534 + 0.05 = 534.05
    токены: prompt 4 400, completion 2 900, total 7 300.
"""

import json
import subprocess
import sys
import tempfile
import threading
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eva import telemetry  # noqa: E402
from eva.telemetry import log, percentile, summary  # noqa: E402

TOOL = ROOT / "tools" / "usage_report.py"
DAY = "2026-11-18"
TS = f"{DAY}T12:00:00Z"


def sample_records() -> list[dict]:
    """100 записей для ручной сверки: 20 skill + 40 small + 40 big."""
    records = []
    for latency in range(1, 21):
        records.append({
            "ts": TS, "route": "skill", "skill": "pricing",
            "prompt_tokens": 0, "completion_tokens": 5,
            "cost_rub": 0.0, "latency_ms": latency,
        })
    for latency in range(100, 140):
        records.append({
            "ts": TS, "route": "small", "skill": None,
            "prompt_tokens": 10, "completion_tokens": 20,
            "cost_rub": 0.001, "latency_ms": latency,
        })
    for latency in range(500, 540):
        records.append({
            "ts": TS, "route": "big", "skill": None,
            "prompt_tokens": 100, "completion_tokens": 50,
            "cost_rub": 0.02, "latency_ms": latency,
        })
    return records


def write_records(path: Path, records: list[dict]) -> None:
    for record in records:
        log(record, path)


class TestPercentile(unittest.TestCase):
    def test_manual_small(self):
        self.assertEqual(percentile([1, 2, 3, 4], 50), 2.5)
        self.assertEqual(percentile([1, 2, 3, 4], 0), 1)
        self.assertEqual(percentile([1, 2, 3, 4], 100), 4)

    def test_manual_100(self):
        values = list(range(1, 101))
        self.assertAlmostEqual(percentile(values, 50), 50.5)
        self.assertAlmostEqual(percentile(values, 95), 95.05)

    def test_single_and_empty(self):
        self.assertEqual(percentile([7], 95), 7)
        self.assertIsNone(percentile([], 50))

    def test_invalid_q(self):
        with self.assertRaises(ValueError):
            percentile([1, 2], -1)
        with self.assertRaises(ValueError):
            percentile([1, 2], 101)

    def test_order_independent(self):
        self.assertEqual(percentile([5, 1, 4, 2, 3], 50), 3)


class TestLog(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "eva.jsonl"

    def test_one_line_per_record_and_fields(self):
        record = log({"route": "small", "cost_rub": 0.01, "latency_ms": 12,
                      "usage": {"prompt_tokens": 3, "completion_tokens": 4}}, self.path)
        log({"route": "big", "skill": "pricing", "cost_rub": 0.5, "latency_ms": 900}, self.path)

        lines = self.path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        first = json.loads(lines[0])
        self.assertEqual(first["route"], "small")
        self.assertEqual(first["skill"], None)
        self.assertEqual(first["prompt_tokens"], 3)
        self.assertEqual(first["completion_tokens"], 4)
        self.assertEqual(first["total_tokens"], 7)
        self.assertAlmostEqual(first["cost_rub"], 0.01)
        self.assertEqual(first["latency_ms"], 12)
        self.assertEqual(record["route"], "small")
        self.assertIn("ts", first)
        self.assertTrue(first["ts"].endswith("Z"))

    def test_ts_preserved_and_generated(self):
        log({"ts": TS, "route": "big", "latency_ms": 1}, self.path)
        log({"route": "big", "latency_ms": 2}, self.path)
        first, second = [json.loads(line) for line in
                         self.path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(first["ts"], TS)
        self.assertTrue(second["ts"].endswith("Z"))
        moment = datetime.fromisoformat(second["ts"].replace("Z", "+00:00"))
        self.assertIsNotNone(moment.tzinfo)

    def test_datetime_ts(self):
        moment = datetime(2026, 11, 18, 12, 0, tzinfo=timezone.utc)
        record = log({"ts": moment, "route": "skill", "latency_ms": 1}, self.path)
        self.assertEqual(record["ts"], "2026-11-18T12:00:00.000Z")

    def test_answer_like_object(self):
        class Route:
            kind = "small"
            reason = "короткий вопрос"
            confidence = 0.9

        class Answer:
            route = Route()
            skill = None
            usage = {"prompt_tokens": 2, "completion_tokens": 3, "cost_rub": 0.03}
            cost_rub = 0.03
            latency_ms = 42

        record = log(Answer(), self.path)
        parsed = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(parsed["route"], "small")
        self.assertEqual(parsed["reason"], "короткий вопрос")
        self.assertAlmostEqual(parsed["confidence"], 0.9)
        self.assertEqual(parsed["prompt_tokens"], 2)
        self.assertEqual(parsed["total_tokens"], 5)
        self.assertEqual(parsed["latency_ms"], 42)
        self.assertEqual(record["route"], "small")

    def test_cyrillic_not_escaped(self):
        log({"route": "skill", "skill": "прайс", "user_id": "юзер", "latency_ms": 1},
            self.path)
        raw = self.path.read_text(encoding="utf-8")
        self.assertIn("прайс", raw)
        self.assertNotIn("\\u", raw)

    def test_creates_only_its_parents_and_file(self):
        nested = Path(self.tmp.name) / "logs" / "eva.jsonl"
        log({"route": "big", "latency_ms": 1}, nested)
        files = sorted(
            str(p.relative_to(self.tmp.name))
            for p in Path(self.tmp.name).rglob("*") if p.is_file())
        self.assertEqual(files, ["logs/eva.jsonl"])

    def test_rejects_unsupported_record(self):
        with self.assertRaises(TypeError):
            log(42, self.path)  # type: ignore[arg-type]
        self.assertFalse(self.path.exists())

    def test_empty_path(self):
        with self.assertRaises(ValueError):
            log({"route": "big"}, "")

    def test_concurrent_appends_stay_one_line_each(self):
        def writer(index: int) -> None:
            for number in range(25):
                log({"route": "small", "latency_ms": index * 100 + number,
                     "cost_rub": 0.001}, self.path)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        lines = self.path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 200)
        for line in lines:
            self.assertIsInstance(json.loads(line), dict)
        report = summary(self.path)
        self.assertEqual(report["records"], 200)
        self.assertEqual(report["skipped"], 0)

    def test_negative_values_clamped(self):
        record = log({"route": "big", "cost_rub": -5, "latency_ms": -1}, self.path)
        self.assertEqual(record["cost_rub"], 0.0)
        self.assertEqual(record["latency_ms"], 0.0)

    def test_extra_fields_kept(self):
        log({"route": "big", "latency_ms": 5, "user_id": "u-1", "channel": "web",
             "text": "ответ"}, self.path)
        parsed = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(parsed["user_id"], "u-1")
        self.assertEqual(parsed["channel"], "web")
        self.assertEqual(parsed["text"], "ответ")

    def test_non_finite_extras_become_null(self):
        log({"route": "big", "latency_ms": 1, "score": float("nan"),
             "nested": {"values": [float("inf"), 1.0]}}, self.path)
        parsed = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertIsNone(parsed["score"])
        self.assertEqual(parsed["nested"], {"values": [None, 1.0]})


class TestSummary(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "eva.jsonl"

    def test_empty_file(self):
        self.path.write_text("", encoding="utf-8")
        report = summary(self.path)
        self.assertEqual(report["records"], 0)
        self.assertEqual(report["skipped"], 0)
        self.assertEqual(report["routes"], {})
        self.assertEqual(report["days"], {})
        self.assertEqual(report["total"]["requests"], 0)
        self.assertIsNone(report["total"]["latency_ms_p50"])
        self.assertIsNone(report["total"]["latency_ms_p95"])
        self.assertEqual(report["total"]["cost_rub_per_1000"], 0.0)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            summary(Path(self.tmp.name) / "нет.jsonl")

    def test_100_records_totals_and_percentiles(self):
        records = sample_records()
        self.assertEqual(len(records), 100)
        write_records(self.path, records)
        report = summary(self.path)

        self.assertEqual(report["records"], 100)
        self.assertEqual(report["skipped"], 0)
        self.assertEqual(sorted(report["routes"]), ["big", "skill", "small"])

        total = report["total"]
        self.assertEqual(total["requests"], 100)
        self.assertAlmostEqual(total["cost_rub"], 0.84, places=9)
        self.assertAlmostEqual(total["cost_rub_avg"], 0.0084, places=9)
        self.assertAlmostEqual(total["cost_rub_per_1000"], 8.40, places=9)
        self.assertEqual(total["prompt_tokens"], 4400)
        self.assertEqual(total["completion_tokens"], 2900)
        self.assertEqual(total["total_tokens"], 7300)
        self.assertAlmostEqual(total["latency_ms_sum"], 25770, places=6)
        self.assertAlmostEqual(total["latency_ms_avg"], 257.70, places=6)
        self.assertAlmostEqual(total["latency_ms_p50"], 129.5, places=9)
        self.assertAlmostEqual(total["latency_ms_p95"], 534.05, places=9)
        self.assertEqual(total["skills"], {"pricing": 20})

        skill = report["routes"]["skill"]
        self.assertEqual(skill["requests"], 20)
        self.assertAlmostEqual(skill["cost_rub"], 0.0, places=9)
        self.assertAlmostEqual(skill["cost_rub_per_1000"], 0.0, places=9)
        self.assertAlmostEqual(skill["latency_ms_avg"], 10.5, places=9)
        self.assertAlmostEqual(skill["latency_ms_p50"], 10.5, places=9)
        self.assertAlmostEqual(skill["latency_ms_p95"], 19.05, places=9)
        self.assertEqual(skill["skills"], {"pricing": 20})

        small = report["routes"]["small"]
        self.assertEqual(small["requests"], 40)
        self.assertAlmostEqual(small["cost_rub"], 0.04, places=9)
        self.assertAlmostEqual(small["cost_rub_avg"], 0.001, places=9)
        self.assertAlmostEqual(small["cost_rub_per_1000"], 1.0, places=9)
        self.assertAlmostEqual(small["latency_ms_avg"], 119.5, places=9)
        self.assertAlmostEqual(small["latency_ms_p50"], 119.5, places=9)
        self.assertAlmostEqual(small["latency_ms_p95"], 137.05, places=9)
        self.assertEqual(small["skills"], {})

        big = report["routes"]["big"]
        self.assertEqual(big["requests"], 40)
        self.assertAlmostEqual(big["cost_rub"], 0.80, places=9)
        self.assertAlmostEqual(big["cost_rub_avg"], 0.02, places=9)
        self.assertAlmostEqual(big["cost_rub_per_1000"], 20.0, places=9)
        self.assertAlmostEqual(big["latency_ms_avg"], 519.5, places=9)
        self.assertAlmostEqual(big["latency_ms_p50"], 519.5, places=9)
        self.assertAlmostEqual(big["latency_ms_p95"], 537.05, places=9)

        self.assertEqual(list(report["days"]), [DAY])
        day = report["days"][DAY]
        self.assertEqual(day["requests"], 100)
        self.assertAlmostEqual(day["cost_rub"], 0.84, places=9)
        self.assertAlmostEqual(day["cost_rub_per_1000"], 8.40, places=9)

    def test_broken_lines_skipped_with_counter(self):
        good_small = json.dumps({"ts": TS, "route": "small", "latency_ms": 10})
        good_big = json.dumps({"ts": TS, "route": "big", "cost_rub": 0.5, "latency_ms": 20})
        self.path.write_text(
            good_small + "\n"
            + '{"route": "broken"\n'
            + "[1, 2, 3]\n"
            + "\n"
            + good_big + "\n",
            encoding="utf-8")

        report = summary(self.path)
        self.assertEqual(report["records"], 2)
        self.assertEqual(report["skipped"], 2)
        self.assertEqual([error["line"] for error in report["errors"]], [2, 3])
        self.assertEqual(sorted(report["routes"]), ["big", "small"])
        self.assertAlmostEqual(report["total"]["cost_rub"], 0.5)

    def test_final_line_without_newline(self):
        self.path.write_text(json.dumps({"route": "big", "latency_ms": 3}),
                             encoding="utf-8")
        report = summary(self.path)
        self.assertEqual(report["records"], 1)
        self.assertEqual(report["skipped"], 0)

    def test_bom_tolerated(self):
        self.path.write_text(
            "\ufeff" + json.dumps({"route": "big", "latency_ms": 1}) + "\n",
            encoding="utf-8")
        report = summary(self.path)
        self.assertEqual(report["records"], 1)
        self.assertEqual(report["skipped"], 0)

    def test_missing_fields_default_to_zero(self):
        self.path.write_text(
            json.dumps({"route": "big", "cost_rub": 0.1}) + "\n"
            + json.dumps({"cost_rub": 0.2}) + "\n",
            encoding="utf-8")
        report = summary(self.path)
        self.assertEqual(report["records"], 2)
        self.assertEqual(sorted(report["routes"]), ["big", "unknown"])
        self.assertEqual(report["routes"]["big"]["latency_ms_p50"], 0.0)
        self.assertEqual(report["routes"]["big"]["latency_ms_p95"], 0.0)
        self.assertEqual(list(report["days"]), ["unknown"])
        self.assertAlmostEqual(report["total"]["cost_rub"], 0.3)

    def test_usage_mapping_and_string_numbers(self):
        self.path.write_text(json.dumps({
            "ts": TS, "route": "big",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                      "total_tokens": 15, "cost_rub": 0.25},
            "latency_ms": "100",
        }) + "\n", encoding="utf-8")
        report = summary(self.path)
        total = report["total"]
        self.assertEqual(total["prompt_tokens"], 10)
        self.assertEqual(total["completion_tokens"], 5)
        self.assertEqual(total["total_tokens"], 15)
        self.assertAlmostEqual(total["cost_rub"], 0.25)
        self.assertAlmostEqual(total["latency_ms_avg"], 100.0)

    def test_summary_does_not_modify_log(self):
        write_records(self.path, sample_records())
        before = self.path.read_bytes()
        summary(self.path)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(sorted(p.name for p in Path(self.tmp.name).iterdir()),
                         ["eva.jsonl"])

    def test_usage_report_alias(self):
        self.assertIs(telemetry.usage_report, telemetry.summary)


class TestUsageReportTool(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "eva.jsonl"

    def run_tool(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(TOOL), *args],
            cwd=self.tmp.name,                # отчёт не должен зависеть от cwd
            capture_output=True, text=True, timeout=60,
        )

    def test_report_on_100_records(self):
        write_records(self.path, sample_records())
        before = self.path.read_bytes()
        result = self.run_tool("--path", str(self.path))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        out = result.stdout

        self.assertIn("записей: 100", out)
        self.assertIn("битых строк пропущено: 0", out)
        self.assertIn("На 1000, ₽", out)
        self.assertIn("8.40", out)          # 0.84 ₽ * 1000 / 100
        self.assertIn("129.50", out)        # общий p50
        self.assertIn("534.05", out)        # общий p95
        big_line = next(line for line in out.splitlines() if line.startswith("big"))
        self.assertIn("40", big_line)
        self.assertIn("20.00", big_line)    # 0.02 ₽ * 1000
        self.assertIn("519.50", big_line)
        self.assertIn("537.05", big_line)
        self.assertIn("ИТОГО", out)
        self.assertEqual(self.path.read_bytes(), before)

    def test_broken_line_does_not_break_report(self):
        write_records(self.path, sample_records())
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write("{это не json}\n")
        result = self.run_tool("--path", str(self.path))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("битых строк пропущено: 1", result.stdout)
        self.assertIn("Битые строки", result.stdout)
        self.assertIn("строка 101", result.stdout)

    def test_json_mode(self):
        write_records(self.path, sample_records())
        result = self.run_tool("--path", str(self.path), "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["records"], 100)
        self.assertEqual(report["routes"]["big"]["requests"], 40)
        self.assertAlmostEqual(report["total"]["cost_rub"], 0.84, places=9)

    def test_empty_log(self):
        self.path.write_text("", encoding="utf-8")
        result = self.run_tool("--path", str(self.path))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("записей нет", result.stdout)

    def test_missing_log_is_an_error(self):
        result = self.run_tool("--path", str(Path(self.tmp.name) / "нет.jsonl"))
        self.assertEqual(result.returncode, 1)
        self.assertIn("лог не найден", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_default_path_is_logs_eva_jsonl(self):
        logs = Path(self.tmp.name) / "logs"
        logs.mkdir()
        write_records(logs / "eva.jsonl", sample_records()[:3])
        result = self.run_tool()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("записей: 3", result.stdout)


if __name__ == "__main__":
    unittest.main()
