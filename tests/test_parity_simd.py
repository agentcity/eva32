"""Тесты tools/parity_simd.py и режима сравнения бэкендов в parity_random.py.

Запуск:
    python3 -m unittest tests.test_parity_simd -v
    python3 -m unittest discover -s tests

Проверяется то, что не требует `src/vtm/vtm` и `libvtm.a`: статистика
распределения, разбор `bench_backend --check`, сводка/отчёт/JSON, решения по
argmax и отрыву top1-top2, CLI-парсер, а также низкоуровневые куски
parity_random, на которых стоит инструмент (проба с флагами санитайзеров,
сравнение логитов двух бэкендов, поля `backend` в результате/сводке).
"""
import contextlib
import io
import json
import math
import os
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools import parity_random  # noqa: E402
from tools import parity_simd  # noqa: E402


# ---------------------------------------------------------------------------
# Вспомогательное: результаты без C
# ---------------------------------------------------------------------------


def backend_comparison(
    *,
    delta: float = 6e-6,
    margin: float = 0.5,
    switch: bool = False,
    baseline: str = "scalar",
    candidate: str = "auto",
) -> parity_random.BackendComparison:
    return parity_random.BackendComparison(
        baseline=baseline,
        candidate=candidate,
        max_abs_delta=delta,
        argmax_baseline=3,
        argmax_candidate=4 if switch else 3,
        top1_margin=margin,
        delta_at_baseline_argmax=delta,
        delta_at_candidate_argmax=delta,
    )


def simd_result(
    index: int = 0,
    *,
    delta: float = 6e-6,
    margin: float = 0.5,
    switch: bool = False,
    python_delta: float = 0.0,
    python_match: bool = True,
    top5_overlap: int = 5,
    error: str | None = None,
) -> parity_simd.SimdResult:
    config = parity_random.generate_config(0, index)
    if error is not None:
        return parity_simd.SimdResult(config=config, error=error)
    result = parity_simd.SimdResult(config=config)
    result.backend = backend_comparison(delta=delta, margin=margin, switch=switch)
    result.python_max_abs_delta = python_delta
    result.python_argmax = 3
    result.python_argmax_match = bool(python_match)
    result.top1_token = 3
    result.top2_token = 4
    result.top1_value = 1.0 + margin
    result.top2_value = 1.0
    result.delta_at_top2 = delta
    result.top5_overlap = top5_overlap
    return result


def summarize(results, **overrides):
    kwargs = {
        "seed": 0,
        "count": len(results),
        "baseline": "scalar",
        "candidate": "auto",
        "tolerance": 1e-5,
        "results": results,
        "workdir": "/tmp/parity_simd",
    }
    kwargs.update(overrides)
    return parity_simd.summarize(**kwargs)


# ---------------------------------------------------------------------------
# Статистика
# ---------------------------------------------------------------------------


class TestPercentile(unittest.TestCase):
    def test_median_of_odd(self):
        self.assertAlmostEqual(parity_simd.percentile([3.0, 1.0, 2.0], 50), 2.0)

    def test_median_interpolates_even(self):
        self.assertAlmostEqual(parity_simd.percentile([1.0, 2.0], 50), 1.5)

    def test_p95_interpolates(self):
        values = [float(i) for i in range(1, 21)]  # 1..20
        # position = (20-1)*0.95 = 18.05 -> 19 + 0.05*(20-19)
        self.assertAlmostEqual(parity_simd.percentile(values, 95), 19.05)

    def test_extremes(self):
        values = [5.0, 1.0, 3.0]
        self.assertEqual(parity_simd.percentile(values, 0), 1.0)
        self.assertEqual(parity_simd.percentile(values, 100), 5.0)

    def test_single_value(self):
        self.assertEqual(parity_simd.percentile([7.5], 95), 7.5)

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            parity_simd.percentile([], 50)

    def test_rejects_out_of_range(self):
        for q in (-1, 101):
            with self.subTest(q=q):
                with self.assertRaises(ValueError):
                    parity_simd.percentile([1.0], q)


class TestDistribution(unittest.TestCase):
    def test_counts_and_stats(self):
        values = [0.0, 2e-5, 5e-5, 1.5e-4]
        stats = parity_simd.distribution(values, 1e-5)
        self.assertEqual(stats["count"], 4)
        self.assertEqual(stats["min"], 0.0)
        self.assertEqual(stats["max"], 1.5e-4)
        self.assertEqual(stats["bit_exact"], 1)
        self.assertEqual(stats["outside_tolerance"], 3)
        self.assertEqual(stats["at_least_1e4"], 1)
        self.assertAlmostEqual(stats["mean"], sum(values) / 4)
        self.assertAlmostEqual(stats["median"], (2e-5 + 5e-5) / 2)
        self.assertEqual(stats["tolerance"], 1e-5)

    def test_tolerance_is_inclusive(self):
        stats = parity_simd.distribution([1e-5], 1e-5)
        self.assertEqual(stats["outside_tolerance"], 1)

    def test_empty(self):
        self.assertEqual(parity_simd.distribution([], 1e-5), {"count": 0})


# ---------------------------------------------------------------------------
# Один результат и его классификация
# ---------------------------------------------------------------------------


class TestSimdResult(unittest.TestCase):
    def test_pass_when_argmax_and_python_match(self):
        result = simd_result(margin=0.5)
        self.assertTrue(result.passed)
        self.assertTrue(result.argmax_match)
        self.assertAlmostEqual(result.max_abs_delta, 6e-6)
        self.assertEqual(result.margin, 0.5)

    def test_switch_fails(self):
        result = simd_result(switch=True, margin=1e-6)
        self.assertFalse(result.passed)
        self.assertFalse(result.argmax_match)
        self.assertEqual(result.failure_reasons()[0][:8], "argmax с")
        self.assertIn("отрыв", result.failure_reasons()[0])

    def test_python_mismatch_fails(self):
        result = simd_result(python_delta=1e-9, python_match=False)
        self.assertFalse(result.passed)
        reasons = result.failure_reasons()
        self.assertTrue(any("Python" in reason for reason in reasons))

    def test_error_has_priority(self):
        result = simd_result(error="ParitySimdError: нет пробы")
        self.assertFalse(result.passed)
        self.assertEqual(
            result.failure_reasons(), ["ошибка прогона: ParitySimdError: нет пробы"]
        )

    def test_to_dict_is_json_safe(self):
        result = simd_result()
        data = result.to_dict()
        json.dumps(data, allow_nan=False)
        self.assertEqual(data["argmax_baseline"], 3)
        self.assertEqual(data["argmax_candidate"], 3)
        self.assertTrue(data["argmax_match"])
        self.assertEqual(data["config"]["seed"], 0)

    def test_margin_to_delta(self):
        result = simd_result(delta=2e-4, margin=0.4)
        self.assertAlmostEqual(result.margin_to_delta, 2000.0)

    def test_margin_to_delta_zero_delta(self):
        result = simd_result(delta=0.0, margin=0.4)
        self.assertEqual(result.margin_to_delta, math.inf)


# ---------------------------------------------------------------------------
# Разбор bench_backend --check
# ---------------------------------------------------------------------------

BENCH_SAMPLE = """\
SIMD: neon, доступен: да
  TMV 1x1 блок 128 scale 0.02    max|diff|=0.000e+00  max|ref|=5.770e-03  ok
  TMV direct 1x1 блок 128 scale 0.02 max|diff|=9.000e-01  max|ref|=5.770e-03  ok
  TMV 12x1000 блок 160 scale 1.00 max|diff|=1.907e-06  max|ref|=3.234e+01  ok
  TMV 128x128 блок 128 scale 1.00 max|diff|=9.537e-07  max|ref|=1.431e+01  ok
  MM trit 8x16x8 блок 128 scale 0.02 max|diff|=1.118e-08  max|ref|=9.970e-02  ok
  MM f32 3x1000x17                   max|diff|=1.907e-06  max|ref|=3.589e+01  ok
  RMSN n=1000                        max|diff|=1.192e-07  max|ref|=1.749e+00  ok
  мусорная строка без числа
"""


class TestParsePerOp(unittest.TestCase):
    def test_groups(self):
        groups = parity_simd.parse_per_op_output(BENCH_SAMPLE)
        self.assertEqual(groups["tmv"]["count"], 3)
        self.assertAlmostEqual(groups["tmv"]["max"], 1.907e-06)
        self.assertEqual(groups["mm_trit"]["count"], 1)
        self.assertAlmostEqual(groups["mm_trit"]["max"], 1.118e-08)
        self.assertEqual(groups["mm_f32"]["count"], 1)
        self.assertAlmostEqual(groups["mm_f32"]["max"], 1.907e-06)
        self.assertEqual(groups["rmsn"]["count"], 1)
        self.assertAlmostEqual(groups["rmsn"]["max"], 1.192e-07)

    def test_direct_is_excluded(self):
        groups = parity_simd.parse_per_op_output(BENCH_SAMPLE)
        # «TMV direct … 9.000e-01» — сравнение скалярных путей, не SIMD-вклад.
        self.assertLess(groups["tmv"]["max"], 1e-5)

    def test_empty(self):
        groups = parity_simd.parse_per_op_output("")
        for key in ("tmv", "mm_trit", "mm_f32", "rmsn"):
            self.assertEqual(groups[key]["count"], 0)
            self.assertIsNone(groups[key]["max"])

    def test_unknown_lines_ignored(self):
        groups = parity_simd.parse_per_op_output(
            "  SILU 1x1 max|diff|=1.0e-03\n  что-то ещё\n"
        )
        self.assertEqual(sum(g["count"] for g in groups.values()), 0)


# ---------------------------------------------------------------------------
# Сводка
# ---------------------------------------------------------------------------


class TestSummarize(unittest.TestCase):
    def test_no_switches_passes(self):
        summary = summarize(
            [simd_result(0), simd_result(1, delta=1e-4, margin=0.25)]
        )
        self.assertTrue(summary["ok"])
        self.assertEqual(summary["argmax"]["switches"], 0)
        self.assertEqual(summary["delta"]["count"], 2)
        self.assertEqual(summary["delta"]["outside_tolerance"], 1)
        self.assertEqual(summary["python_vs_baseline"]["bit_exact"], 2)
        self.assertEqual(summary["failed"], 0)

    def test_switch_fails_and_is_listed(self):
        summary = summarize(
            [
                simd_result(0),
                simd_result(1, switch=True, delta=5e-5, margin=2e-5),
            ]
        )
        self.assertFalse(summary["ok"])
        self.assertEqual(summary["argmax"]["switches"], 1)
        switch = summary["argmax"]["switch_configs"][0]
        self.assertEqual(switch["index"], 1)
        self.assertEqual(switch["argmax_baseline"], 3)
        self.assertEqual(switch["argmax_candidate"], 4)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["failures"][0]["index"], 1)

    def test_borderline_below_two_delta(self):
        summary = summarize(
            [
                simd_result(0, delta=1e-5, margin=1.5e-5),  # < 2d
                simd_result(1, delta=1e-5, margin=1.0),     # >> 2d
            ]
        )
        self.assertEqual(summary["argmax"]["borderline_below_2delta"], 1)
        self.assertEqual(summary["argmax"]["borderline"][0]["index"], 0)

    def test_narrowest_and_tightest(self):
        summary = summarize(
            [
                simd_result(0, delta=1e-5, margin=0.5),
                simd_result(1, delta=1e-6, margin=0.01),
            ]
        )
        self.assertEqual(summary["argmax"]["narrowest_margin"]["index"], 1)
        # отношение отрыв/Δ: idx0 = 5e4, idx1 = 1e4 -> узкий по отношению idx1
        self.assertEqual(summary["argmax"]["tightest_ratio"]["index"], 1)

    def test_worst_is_max_delta(self):
        summary = summarize(
            [
                simd_result(0, delta=6e-6),
                simd_result(1, delta=1.05e-4, margin=0.25),
                simd_result(2, delta=2e-5),
            ]
        )
        self.assertEqual(summary["worst"]["index"], 1)
        self.assertAlmostEqual(summary["worst"]["max_abs_delta"], 1.05e-4)
        self.assertIn("config", summary["worst"])

    def test_error_result(self):
        summary = summarize([simd_result(0), simd_result(1, error="boom")])
        self.assertEqual(summary["errors"], 1)
        self.assertFalse(summary["ok"])
        self.assertEqual(summary["delta"]["count"], 1)
        self.assertEqual(summary["failed"], 1)

    def test_all_errors(self):
        summary = summarize([simd_result(0, error="boom")])
        self.assertEqual(summary["delta"], {"count": 0})
        self.assertIsNone(summary["worst"])
        self.assertFalse(summary["ok"])

    def test_python_mismatch_breaks_ok(self):
        summary = summarize([simd_result(0, python_delta=1e-9, python_match=False)])
        self.assertFalse(summary["ok"])
        self.assertEqual(summary["python_vs_baseline"]["argmax_mismatches"], 1)
        self.assertEqual(summary["python_vs_baseline"]["bit_exact"], 0)

    def test_top5_changed_counted(self):
        summary = summarize(
            [simd_result(0), simd_result(1, top5_overlap=4)]
        )
        self.assertEqual(summary["top5"]["changed"], 1)

    def test_json_roundtrip(self):
        summary = summarize([simd_result(0), simd_result(1, switch=True)])
        text = parity_simd.format_json(summary)
        restored = json.loads(text)
        self.assertEqual(restored["argmax"]["switches"], 1)
        self.assertEqual(len(restored["configs"]), 2)

    def test_values_survive_json(self):
        summary = summarize([simd_result(0, delta=6.109e-6, margin=8.693e-3)])
        restored = json.loads(parity_simd.format_json(summary))
        self.assertAlmostEqual(restored["delta"]["median"], 6.109e-6)
        self.assertAlmostEqual(
            restored["argmax"]["narrowest_margin"]["top1_margin"], 8.693e-3
        )


# ---------------------------------------------------------------------------
# Отчёт
# ---------------------------------------------------------------------------


class TestFormatReport(unittest.TestCase):
    def test_report_mentions_distribution_and_verdict(self):
        summary = summarize(
            [
                simd_result(0, delta=6e-6, margin=0.5),
                simd_result(1, delta=1.05e-4, margin=0.25),
            ]
        )
        text = parity_simd.format_report(summary)
        self.assertIn("распределение max|Δ|", text)
        self.assertIn("медиана", text)
        self.assertIn("p95", text)
        self.assertIn("смен argmax: 0/2", text)
        self.assertIn("контроль: Python ↔ scalar", text)
        self.assertIn("бит-в-бит: 2/2", text)
        self.assertIn("максимум: 1.050e-04", text)
        self.assertIn("--only 1", text)
        self.assertTrue(text.rstrip().endswith("PASS"))
        self.assertIn("на ответы не влияет", text)

    def test_report_for_switch(self):
        summary = summarize(
            [simd_result(0, switch=True, delta=5e-5, margin=2e-5)]
        )
        text = parity_simd.format_report(summary)
        self.assertIn("СМЕНА idx=0", text)
        self.assertIn("argmax изменился", text)
        self.assertTrue(text.rstrip().endswith("FAIL"))

    def test_report_without_successful_runs(self):
        summary = summarize([simd_result(0, error="boom")])
        text = parity_simd.format_report(summary)
        self.assertIn("нет ни одного успешного прогона", text)
        self.assertIn("FAIL", text)

    def test_report_per_op(self):
        summary = summarize([simd_result(0)])
        summary["per_op"] = {
            "available": True,
            "groups": parity_simd.parse_per_op_output(BENCH_SAMPLE),
        }
        text = parity_simd.format_report(summary)
        self.assertIn("вклад операций", text)
        self.assertIn("TMV", text)
        self.assertIn("RMSN", text)

    def test_report_per_op_unavailable(self):
        summary = summarize([simd_result(0)])
        summary["per_op"] = {"available": False, "reason": "нет bench_backend"}
        text = parity_simd.format_report(summary)
        self.assertIn("пропущено: нет bench_backend", text)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli(unittest.TestCase):
    def test_defaults(self):
        args = parity_simd.parse_args([])
        self.assertEqual(args.count, 50)
        self.assertEqual(args.seed, 0)
        self.assertEqual(args.baseline, "scalar")
        self.assertEqual(args.candidate, "auto")
        self.assertEqual(args.tolerance, 1e-5)
        self.assertFalse(args.json)
        self.assertFalse(args.per_op)
        self.assertFalse(args.keep)
        self.assertIsNone(args.only)

    def test_flags(self):
        args = parity_simd.parse_args(
            [
                "--count", "3",
                "--seed", "7",
                "--only", "1,2",
                "--baseline", "simd",
                "--candidate", "scalar",
                "--json",
                "--per-op",
                "--keep",
            ]
        )
        self.assertEqual(args.count, 3)
        self.assertEqual(args.seed, 7)
        self.assertEqual(args.only, [1, 2])
        self.assertEqual(args.baseline, "simd")
        self.assertEqual(args.candidate, "scalar")
        self.assertTrue(args.json)
        self.assertTrue(args.per_op)
        self.assertTrue(args.keep)

    def test_bad_backend_rejected(self):
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                parity_simd.parse_args(["--baseline", "cuda"])

    def test_bad_count_rejected(self):
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                parity_simd.parse_args(["--count", "0"])

    def test_same_backend_is_error_without_touching_c(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = parity_simd.main(["--baseline", "scalar", "--candidate", "scalar"])
        self.assertEqual(code, 2)
        self.assertIn("совпадают", stderr.getvalue())

    def test_missing_runtime_returns_2(self):
        # Без C-рантайма инструмент обязан сказать кодом 2, а не падать.
        env = parity_random.Environment(
            vtm_bin=Path("/nonexistent/vtm"),
            vtm_lib=Path("/nonexistent/libvtm.a"),
            include_dir=Path("/nonexistent"),
            cc=None,
        )
        stderr = io.StringIO()
        with unittest.mock.patch.object(
            parity_random, "check_environment", return_value=env
        ):
            with contextlib.redirect_stderr(stderr):
                code = parity_simd.main(["--count", "1"])
        self.assertEqual(code, 2)
        self.assertIn("не хватает", stderr.getvalue())
        self.assertIn("make -C src/vtm", stderr.getvalue())

    def test_parser_epilog_mentions_argmax(self):
        text = parity_simd.build_parser().format_help()
        self.assertIn("argmax", text)
        self.assertIn("parity_simd", text)


# ---------------------------------------------------------------------------
# parity_random: режим сравнения бэкендов
# ---------------------------------------------------------------------------


class TestParityRandomBackendSupport(unittest.TestCase):
    def test_sanitizer_flags_plain_library(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib = Path(tmp) / "libvtm.a"
            lib.write_bytes(b"!<arch>\n" + b"vtm_run\0" * 4)
            self.assertEqual(parity_random.sanitizer_flags(lib), [])

    def test_sanitizer_flags_detects_asan_and_ubsan(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib = Path(tmp) / "libvtm.a"
            lib.write_bytes(b"!<arch>\n___asan_init\0___ubsan_handle_type_mismatch\0")
            self.assertEqual(
                parity_random.sanitizer_flags(lib),
                ["-fsanitize=address", "-fsanitize=undefined"],
            )

    def test_sanitizer_flags_requires_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib = Path(tmp) / "not-a-lib.a"
            lib.write_bytes(b"___asan_init")
            self.assertEqual(parity_random.sanitizer_flags(lib), [])

    def test_sanitizer_flags_missing_file(self):
        self.assertEqual(
            parity_random.sanitizer_flags(Path("/nonexistent/libvtm.a")), []
        )

    def test_probe_compile_command_includes_extra_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            env = parity_random.Environment(
                vtm_bin=workdir / "vtm",
                vtm_lib=workdir / "libvtm.a",
                include_dir=workdir,
                cc="cc",
            )
            env.vtm_lib.write_bytes(b"!<arch>\n___asan_init\0")
            cmd = parity_random.probe_compile_command(env, workdir)
            self.assertIn("-fsanitize=address", cmd)
            # Флаги идут до -I и до входных файлов: порядок важен для clang.
            self.assertLess(cmd.index("-fsanitize=address"), cmd.index("-I"))

    def test_probe_compile_command_can_disable_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            env = parity_random.Environment(
                vtm_bin=workdir / "vtm",
                vtm_lib=workdir / "libvtm.a",
                include_dir=workdir,
                cc="cc",
            )
            env.vtm_lib.write_bytes(b"!<arch>\n___asan_init\0")
            cmd = parity_random.probe_compile_command(
                env, workdir, extra_flags=[]
            )
            self.assertFalse(
                any(part.startswith("-fsanitize=") for part in cmd)
            )

    def test_compare_backend_logits_identical(self):
        logits = [0.5, 2.0, 1.0, 1.9]
        comparison = parity_random.compare_backend_logits(logits, list(logits))
        self.assertEqual(comparison.max_abs_delta, 0.0)
        self.assertTrue(comparison.argmax_match)
        self.assertEqual(comparison.argmax_baseline, 1)
        self.assertAlmostEqual(comparison.top1_margin, 0.1)
        self.assertAlmostEqual(comparison.delta_at_baseline_argmax, 0.0)
        self.assertEqual(comparison.margin_to_delta, math.inf)

    def test_compare_backend_logits_switch(self):
        # Отрыв 2e-4, расхождение 1.5e-4: лидер и второй меняются местами,
        # потому что 2*Δ = 3e-4 > отрыва.
        baseline = [2.0, 1.9998, 0.1]
        candidate = [1.99985, 1.99995, 0.1]
        comparison = parity_random.compare_backend_logits(
            baseline, candidate, baseline="scalar", candidate="auto"
        )
        self.assertFalse(comparison.argmax_match)
        self.assertEqual(comparison.argmax_baseline, 0)
        self.assertEqual(comparison.argmax_candidate, 1)
        self.assertAlmostEqual(comparison.top1_margin, 2e-4, places=15)
        self.assertAlmostEqual(comparison.max_abs_delta, 1.5e-4, places=15)
        self.assertAlmostEqual(comparison.margin_to_delta, 4.0 / 3.0, places=9)

    def test_compare_backend_logits_delta_at_positions(self):
        baseline = [1.0, 2.0, 3.0]
        candidate = [1.5, 2.05, 2.0]
        comparison = parity_random.compare_backend_logits(baseline, candidate)
        self.assertEqual(comparison.argmax_baseline, 2)
        self.assertEqual(comparison.argmax_candidate, 1)
        self.assertAlmostEqual(comparison.delta_at_baseline_argmax, 1.0)
        self.assertAlmostEqual(comparison.delta_at_candidate_argmax, 0.05)

    def test_compare_backend_logits_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            parity_random.compare_backend_logits([1.0], [1.0, 2.0])
        with self.assertRaises(ValueError):
            parity_random.compare_backend_logits([], [])

    def test_top1_margin(self):
        self.assertAlmostEqual(parity_random.top1_margin([1.0, 3.0, 2.0]), 1.0)
        self.assertEqual(parity_random.top1_margin([1.0]), 0.0)
        self.assertEqual(parity_random.top1_margin([2.0, 2.0]), 0.0)

    def test_backend_comparison_dict(self):
        data = backend_comparison(switch=True).to_dict()
        json.dumps(data, allow_nan=False)
        self.assertEqual(data["baseline"], "scalar")
        self.assertEqual(data["candidate"], "auto")
        self.assertFalse(data["argmax_match"])

    def test_config_result_with_backend(self):
        config = parity_random.generate_config(0, 0)
        result = parity_random.ConfigResult(
            config=config,
            max_abs_delta=0.0,
            argmax_reference=5,
            argmax_runtime=5,
            argmax_match=True,
            within_tolerance=True,
            cli_argmax=5,
            cli_match=True,
            backend=backend_comparison(delta=1e-4, margin=0.2),
        )
        self.assertTrue(result.passed)
        data = result.to_dict()
        json.dumps(data, allow_nan=False)
        self.assertEqual(data["backend"]["max_abs_delta"], 1e-4)

    def test_config_result_fails_on_backend_switch(self):
        config = parity_random.generate_config(0, 0)
        result = parity_random.ConfigResult(
            config=config,
            max_abs_delta=0.0,
            argmax_reference=5,
            argmax_runtime=5,
            argmax_match=True,
            within_tolerance=True,
            cli_argmax=5,
            cli_match=True,
            backend=backend_comparison(switch=True, margin=1e-7),
        )
        self.assertFalse(result.passed)
        reasons = result.failure_reasons()
        self.assertTrue(any("между бэкендами" in reason for reason in reasons))

    def test_summary_without_backend_compare(self):
        config = parity_random.generate_config(0, 0)
        result = parity_random.ConfigResult(
            config=config,
            max_abs_delta=0.0,
            argmax_reference=1,
            argmax_runtime=1,
            argmax_match=True,
            within_tolerance=True,
            cli_argmax=1,
            cli_match=True,
        )
        summary = parity_random.summarize(
            seed=0, count=1, tolerance=1e-5, results=[result]
        )
        self.assertFalse(summary["backend_compare"]["enabled"])
        self.assertEqual(summary["backend_compare"]["argmax_switches"], 0)
        self.assertIsNone(summary["backend_compare"]["worst"])

    def test_summary_with_backend_compare(self):
        configs = [parity_random.generate_config(0, index) for index in (0, 1)]
        results = [
            parity_random.ConfigResult(
                config=config,
                max_abs_delta=0.0,
                argmax_reference=1,
                argmax_runtime=1,
                argmax_match=True,
                within_tolerance=True,
                cli_argmax=1,
                cli_match=True,
                backend=backend_comparison(
                    delta=delta, margin=0.3, switch=switch
                ),
            )
            for config, delta, switch in (
                (configs[0], 6e-6, False),
                (configs[1], 1e-4, True),
            )
        ]
        summary = parity_random.summarize(
            seed=0, count=2, tolerance=1e-5, results=results
        )
        backend = summary["backend_compare"]
        self.assertTrue(backend["enabled"])
        self.assertEqual(backend["checked"], 2)
        self.assertEqual(backend["argmax_switches"], 1)
        self.assertEqual(backend["switches"][0]["index"], 1)
        self.assertEqual(backend["worst"]["index"], 1)
        self.assertFalse(backend["ok"])
        self.assertFalse(summary["ok"])

    def test_parse_backend(self):
        for name in parity_random.BACKEND_NAMES:
            self.assertEqual(parity_random.parse_backend(name), name)
        with self.assertRaises(Exception):
            parity_random.parse_backend("cuda")

    def test_cli_flags(self):
        args = parity_random.parse_args([])
        self.assertIsNone(args.backend)
        self.assertFalse(args.compare_backends)
        self.assertEqual(args.compare_backend, "auto")
        args = parity_random.parse_args(
            ["--backend", "auto", "--compare-backends", "--compare-backend", "simd"]
        )
        self.assertEqual(args.backend, "auto")
        self.assertTrue(args.compare_backends)
        self.assertEqual(args.compare_backend, "simd")

    def test_report_contains_backend_section(self):
        summary = summarize([simd_result(0, delta=6e-6, margin=0.4)])
        # Сводка parity_simd — другой формат; секцию бэкендов проверяем на
        # сводке parity_random.
        config = parity_random.generate_config(0, 0)
        result = parity_random.ConfigResult(
            config=config,
            max_abs_delta=0.0,
            argmax_reference=1,
            argmax_runtime=1,
            argmax_match=True,
            within_tolerance=True,
            cli_argmax=1,
            cli_match=True,
            backend=backend_comparison(delta=1.05e-4, margin=0.25),
        )
        pr_summary = parity_random.summarize(
            seed=0, count=1, tolerance=1e-5, results=[result]
        )
        text = parity_random.format_report(pr_summary, verbose_configs=False)
        self.assertIn("сверка бэкендов", text)
        self.assertIn("смен argmax: 0", text)
        self.assertIn("1.050e-04", text)
        self.assertIn("top1-top2", text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
