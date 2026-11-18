"""Perf smoke в tools/ci.py (задача 0117): эталон, допуск и скипы.

Что проверяем:

* файл `tests/fixtures/perf_baseline.json` валиден и хранит дату, машину,
  версию бенчмарка, sha256 модели и протокол замера;
* на чужой машине (другая ОС/архитектура/CPU), с другой моделью или с другой
  версией бенчмарка шаг скипается с предупреждением, а не краснеет;
* при искусственном замедлении (подмена медианы в эталоне) шаг краснеет;
* при ускорении больше допуска шаг тоже краснеет и просит обновить эталон —
  иначе эталон тихо устаревает;
* реальный замер (`vtm run` на demo-модели) тоже ловит подменённый эталон, но
  этот тест скипается, если C-часть не собрана (чистый клон, `make clean`).

Тесты не требуют C-сборки, кроме `TestRealMeasurement`: для остальных замер
подменяется функцией, а модель — временным файлом.
"""
from __future__ import annotations

import json
import statistics
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import ci  # noqa: E402

BASELINE = ROOT / "tests" / "fixtures" / "perf_baseline.json"


def make_baseline(
    median: float,
    *,
    model: Path,
    machine: dict | None = None,
    tol: float | None = None,
) -> dict:
    """Эталон в формате `--record-perf-baseline`, но с заданной медианой."""
    data = ci.perf_baseline_data(
        [median * 0.99, median, median * 1.01],
        machine=machine or ci.perf_machine(),
        model=model,
    )
    data["metric"]["median"] = float(median)
    if tol is not None:
        data["tolerance_pct"] = tol
    return data


class PerfTestCase(unittest.TestCase):
    """Общая заготовка: временный каталог и «модель» на 4 байта."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.model = self.tmp / "model.vtm"
        self.model.write_bytes(b"demo-model-for-sha")

    def write_baseline(self, data: dict, name: str = "baseline.json") -> Path:
        path = self.tmp / name
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return path


class TestBaselineFixture(unittest.TestCase):
    """Репозиторный эталон: формат, метаданные и согласованность с кодом."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.data = json.loads(BASELINE.read_text(encoding="utf-8"))

    def test_exists_and_is_object(self):
        self.assertIsInstance(self.data, dict)

    def test_format_and_benchmark_version(self):
        self.assertEqual(self.data["format"], ci.PERF_BASELINE_FORMAT)
        self.assertEqual(
            self.data["benchmark_version"], ci.PERF_BENCHMARK_VERSION
        )

    def test_has_date(self):
        recorded = self.data.get("recorded_at")
        self.assertIsInstance(recorded, str)
        # ISO-8601 в UTC с «Z» — чтобы дату читал человек и парсила машина.
        parsed = datetime.strptime(recorded, "%Y-%m-%dT%H:%M:%SZ")
        self.assertGreaterEqual(parsed.year, 2026)

    def test_has_machine(self):
        machine = self.data["machine"]
        for key in ("os", "arch", "cpu"):
            self.assertIsInstance(machine.get(key), str, key)
            self.assertTrue(machine[key].strip(), key)
        self.assertGreaterEqual(machine.get("cores", 0), 1)

    def test_has_model_sha256(self):
        model = self.data["model"]
        self.assertEqual(model["path"], "demo/model.vtm")
        self.assertRegex(model["sha256"], r"^[0-9a-f]{64}$")
        self.assertGreater(model["bytes"], 0)

    def test_protocol_matches_code_constants(self):
        params = self.data["params"]
        self.assertEqual(params["tokens"], ci.PERF_TOKENS)
        self.assertEqual(params["max_new"], ci.PERF_MAX_NEW)
        self.assertEqual(params["repeats"], ci.PERF_REPEATS)
        self.assertEqual(params["backend"], "auto")

    def test_metric_is_sane(self):
        metric = self.data["metric"]
        self.assertEqual(metric["name"], "tokens_per_second")
        self.assertGreater(metric["median"], 0)
        self.assertEqual(len(metric["samples"]), ci.PERF_REPEATS)
        # Медиана в файле — действительно медиана сохранённых замеров.
        self.assertAlmostEqual(
            metric["median"],
            statistics.median(metric["samples"]),
            delta=1.0,
        )
        # Записанный разброс должен быть заметно меньше допуска: иначе эталон
        # шумный и будет ложно краснеть.
        self.assertLess(metric["spread_pct"], ci.PERF_TOLERANCE_PCT)

    def test_tolerance_is_25_percent(self):
        self.assertEqual(self.data["tolerance_pct"], 25.0)

    def test_machine_is_current(self):
        """Эталон в репозитории — для этой машины (иначе он тут бесполезен).

        На чужой машине тест скипается, а не падает: там шаг CI законно
        скипается с предупреждением.
        """
        machine = ci.perf_machine()
        saved = self.data["machine"]
        if (saved["os"], saved["arch"], saved["cpu"]) != (
            machine["os"],
            machine["arch"],
            machine["cpu"],
        ):
            self.skipTest(
                f"эталон снят на другой машине: {saved['os']}/{saved['arch']}/"
                f"{saved['cpu']}"
            )

    def test_matches_demo_model(self):
        if not ci.PERF_MODEL.is_file():
            self.skipTest("нет demo/model.vtm (соберите make -C src/vtm test)")
        actual = ci.sha256_file(ci.PERF_MODEL)
        if actual != self.data["model"]["sha256"]:
            # Модель пересобрали — эталон устарел, но это не повод красить
            # чужую задачу: шаг CI в этом случае скипается с предупреждением.
            self.skipTest(
                "demo-модель изменилась — эталон устарел, перезапишите: "
                "python3 tools/ci.py --record-perf-baseline"
            )

    def test_step_is_green_with_real_baseline(self):
        """Без реального замера: метаданные эталона принимаются этой машиной."""
        machine = ci.perf_machine()
        saved = self.data["machine"]
        if (saved["os"], saved["arch"], saved["cpu"]) != (
            machine["os"],
            machine["arch"],
            machine["cpu"],
        ):
            self.skipTest("эталон снят на другой машине")
        if not ci.PERF_MODEL.is_file():
            self.skipTest("нет demo/model.vtm")
        self.assertIsNone(
            ci.perf_skip_reason(
                self.data, machine=machine, model=ci.PERF_MODEL
            )
        )


class TestSkipReasons(PerfTestCase):
    """Чужая машина/модель/версия — предупреждение, а не падение."""

    def test_same_machine_no_skip(self):
        baseline = make_baseline(27000.0, model=self.model)
        self.assertIsNone(
            ci.perf_skip_reason(
                baseline, machine=ci.perf_machine(), model=self.model
            )
        )

    def test_other_os_skips(self):
        machine = ci.perf_machine()
        baseline = make_baseline(
            27000.0,
            model=self.model,
            machine={**machine, "os": "Plan9"},
        )
        reason = ci.perf_skip_reason(
            baseline, machine=machine, model=self.model
        )
        self.assertIsNotNone(reason)
        self.assertIn("другой машине", reason)
        self.assertIn("ОС", reason)

    def test_other_arch_skips(self):
        machine = ci.perf_machine()
        baseline = make_baseline(
            27000.0,
            model=self.model,
            machine={**machine, "arch": "riscv64"},
        )
        reason = ci.perf_skip_reason(
            baseline, machine=machine, model=self.model
        )
        self.assertIsNotNone(reason)
        self.assertIn("архитектура", reason)

    def test_other_cpu_skips(self):
        machine = {**ci.perf_machine(), "cpu": "Apple M1 Pro"}
        baseline = make_baseline(
            27000.0,
            model=self.model,
            machine={**machine, "cpu": "Apple M4 Max"},
        )
        reason = ci.perf_skip_reason(
            baseline, machine=machine, model=self.model
        )
        self.assertIsNotNone(reason)
        self.assertIn("CPU", reason)

    def test_unknown_cpu_does_not_skip(self):
        """Если CPU не определился, решает только ОС/архитектура."""
        machine = {**ci.perf_machine(), "cpu": ""}
        baseline = make_baseline(27000.0, model=self.model, machine=machine)
        self.assertIsNone(
            ci.perf_skip_reason(
                baseline, machine=machine, model=self.model
            )
        )

    def test_other_benchmark_version_skips(self):
        baseline = make_baseline(27000.0, model=self.model)
        baseline["benchmark_version"] = ci.PERF_BENCHMARK_VERSION + 1
        reason = ci.perf_skip_reason(
            baseline, machine=ci.perf_machine(), model=self.model
        )
        self.assertIsNotNone(reason)
        self.assertIn("версией бенчмарка", reason)

    def test_other_format_skips(self):
        baseline = make_baseline(27000.0, model=self.model)
        baseline["format"] = ci.PERF_BASELINE_FORMAT + 1
        reason = ci.perf_skip_reason(
            baseline, machine=ci.perf_machine(), model=self.model
        )
        self.assertIsNotNone(reason)
        self.assertIn("формата", reason)

    def test_other_model_skips(self):
        baseline = make_baseline(27000.0, model=self.model)
        other = self.tmp / "other.vtm"
        other.write_bytes(b"other model")
        reason = ci.perf_skip_reason(
            baseline, machine=ci.perf_machine(), model=other
        )
        self.assertIsNotNone(reason)
        self.assertIn("demo-модели", reason)

    def test_other_protocol_skips(self):
        baseline = make_baseline(27000.0, model=self.model)
        baseline["params"] = {
            **baseline["params"],
            "max_new": ci.PERF_MAX_NEW * 2,
        }
        reason = ci.perf_skip_reason(
            baseline, machine=ci.perf_machine(), model=self.model
        )
        self.assertIsNotNone(reason)
        self.assertIn("протоколу", reason)


class TestCompare(unittest.TestCase):
    """Допуск ±25 %: внутри — зелено, за допуском — красно в обе стороны."""

    def test_within_tolerance_is_ok(self):
        status, detail = ci.compare_perf(27000.0, 26000.0, 25.0)
        self.assertEqual(status, "ok")
        self.assertIn("эталон 26000", detail)

    def test_exact_boundary_is_ok(self):
        # Ровно 25 % — ещё в допуске («больше чем на 25 %» — это регрессия).
        status, _ = ci.compare_perf(750.0, 1000.0, 25.0)
        self.assertEqual(status, "ok")

    def test_slowdown_over_tolerance_fails(self):
        status, detail = ci.compare_perf(700.0, 1000.0, 25.0)
        self.assertEqual(status, "fail")
        self.assertIn("медленнее эталона на 30.0%", detail)
        self.assertIn("допуск ±25%", detail)

    def test_speedup_over_tolerance_fails_and_asks_update(self):
        status, detail = ci.compare_perf(1400.0, 1000.0, 25.0)
        self.assertEqual(status, "fail")
        self.assertIn("быстрее эталона на 40.0%", detail)
        self.assertIn("--record-perf-baseline", detail)

    def test_zero_speed_is_a_regression(self):
        status, _ = ci.compare_perf(0.0, 27000.0, 25.0)
        self.assertEqual(status, "fail")


class TestStepPerf(PerfTestCase):
    """Шаг целиком: скипы на отсутствии входа, красный на замедлении."""

    def test_missing_baseline_skips_with_warning(self):
        outcome = ci.step_perf(
            baseline_path=self.tmp / "нет.json",
            model=self.model,
            measure=lambda: [27000.0],
        )
        self.assertEqual(outcome.status, "skip")
        self.assertIn("предупреждение", outcome.detail)
        self.assertIn("нет эталона", outcome.detail)

    def test_missing_binary_skips(self):
        outcome = ci.step_perf(
            baseline_path=self.tmp / "нет.json",
            binary=self.tmp / "нет-vtm",
            model=self.model,
        )
        self.assertEqual(outcome.status, "skip")
        self.assertIn("C-часть не собрана", outcome.detail)

    def test_missing_model_skips(self):
        outcome = ci.step_perf(
            baseline_path=self.tmp / "нет.json",
            binary=self.tmp / "нет-vtm",
            model=self.tmp / "нет.vtm",
        )
        self.assertEqual(outcome.status, "skip")
        self.assertIn("соберите", outcome.detail)

    def test_corrupt_baseline_fails(self):
        path = self.tmp / "broken.json"
        path.write_text("{ это не json", encoding="utf-8")
        outcome = ci.step_perf(
            baseline_path=path, model=self.model, measure=lambda: [27000.0]
        )
        self.assertEqual(outcome.status, "fail")
        self.assertIn("повреждён", outcome.detail)

    def test_baseline_without_metric_fails(self):
        baseline = make_baseline(27000.0, model=self.model)
        del baseline["metric"]
        path = self.write_baseline(baseline)
        outcome = ci.step_perf(
            baseline_path=path, model=self.model, measure=lambda: [27000.0]
        )
        self.assertEqual(outcome.status, "fail")
        self.assertIn("metric", outcome.detail)

    def test_artificial_slowdown_reds(self):
        """Подмена эталона: медиана в 2 раза выше реальной → -50 % → красный."""
        path = self.write_baseline(
            make_baseline(54000.0, model=self.model)
        )
        outcome = ci.step_perf(
            baseline_path=path,
            model=self.model,
            measure=lambda: [27000.0, 26900.0, 27100.0],
        )
        self.assertEqual(outcome.status, "fail")
        self.assertIn("медленнее эталона", outcome.detail)
        self.assertIn("допуск ±25%", outcome.detail)

    def test_within_tolerance_is_ok(self):
        path = self.write_baseline(
            make_baseline(27000.0, model=self.model)
        )
        outcome = ci.step_perf(
            baseline_path=path,
            model=self.model,
            measure=lambda: [26400.0, 27000.0, 27600.0],
        )
        self.assertEqual(outcome.status, "ok", outcome.detail)
        self.assertIn("отклонение", outcome.detail)

    def test_median_ignores_single_outlier(self):
        """Медиана, а не среднее: один провал не должен красить шаг."""
        path = self.write_baseline(
            make_baseline(27000.0, model=self.model)
        )
        outcome = ci.step_perf(
            baseline_path=path,
            model=self.model,
            measure=lambda: [1000.0, 27000.0, 27000.0, 27000.0, 1000.0],
        )
        self.assertEqual(outcome.status, "ok", outcome.detail)

    def test_foreign_machine_skips_even_with_bad_speed(self):
        """На чужой машине даже «медленный» замер — не повод краснеть."""
        machine = ci.perf_machine()
        path = self.write_baseline(
            make_baseline(
                54000.0,
                model=self.model,
                machine={**machine, "os": "Plan9"},
            )
        )
        outcome = ci.step_perf(
            baseline_path=path,
            model=self.model,
            measure=lambda: [27000.0],
        )
        self.assertEqual(outcome.status, "skip")
        self.assertIn("предупреждение", outcome.detail)

    def test_measurement_error_fails(self):
        path = self.write_baseline(
            make_baseline(27000.0, model=self.model)
        )

        def broken() -> list[float]:
            raise ci.PerfError("vtm run упал")

        outcome = ci.step_perf(
            baseline_path=path, model=self.model, measure=broken
        )
        self.assertEqual(outcome.status, "fail")
        self.assertIn("замер не удался", outcome.detail)

    def test_empty_measurement_fails(self):
        path = self.write_baseline(
            make_baseline(27000.0, model=self.model)
        )
        outcome = ci.step_perf(
            baseline_path=path, model=self.model, measure=lambda: []
        )
        self.assertEqual(outcome.status, "fail")
        self.assertIn("ни одного значения", outcome.detail)


class TestRecord(PerfTestCase):
    """Запись эталона: дата, машина, версия, протокол и медиана на месте."""

    def test_record_writes_valid_baseline(self):
        target = self.tmp / "new-baseline.json"
        outcome = ci.step_perf_record(
            baseline_path=target,
            model=self.model,
            machine=ci.perf_machine(),
            measure=lambda: [27000.0, 27200.0, 26800.0],
        )
        self.assertEqual(outcome.status, "ok", outcome.detail)
        self.assertTrue(target.is_file())
        self.assertIn("медиана 27000", outcome.detail)

        data = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(data["format"], ci.PERF_BASELINE_FORMAT)
        self.assertEqual(data["benchmark_version"], ci.PERF_BENCHMARK_VERSION)
        self.assertRegex(data["recorded_at"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
        self.assertEqual(data["machine"]["os"], ci.perf_machine()["os"])
        self.assertEqual(data["machine"]["arch"], ci.perf_machine()["arch"])
        self.assertEqual(data["model"]["sha256"], ci.sha256_file(self.model))
        self.assertEqual(data["params"]["max_new"], ci.PERF_MAX_NEW)
        self.assertEqual(data["metric"]["median"], 27000.0)
        self.assertEqual(len(data["metric"]["samples"]), 3)
        self.assertEqual(data["tolerance_pct"], ci.PERF_TOLERANCE_PCT)

    def test_recorded_baseline_is_accepted_by_check(self):
        target = self.tmp / "roundtrip.json"
        samples = [27000.0, 27200.0, 26800.0]
        ci.step_perf_record(
            baseline_path=target,
            model=self.model,
            machine=ci.perf_machine(),
            measure=lambda: list(samples),
        )
        outcome = ci.step_perf(
            baseline_path=target, model=self.model, measure=lambda: list(samples)
        )
        self.assertEqual(outcome.status, "ok", outcome.detail)

    def test_record_without_binary_skips(self):
        outcome = ci.step_perf_record(
            baseline_path=self.tmp / "x.json",
            binary=self.tmp / "нет-vtm",
            model=self.model,
        )
        self.assertEqual(outcome.status, "skip")
        self.assertIn("C-часть не собрана", outcome.detail)


class TestCli(unittest.TestCase):
    """Флаги: шаг можно выключить и эталон — перезаписать."""

    def test_perf_is_on_by_default(self):
        args = ci.parse_args([])
        self.assertFalse(args.skip_perf)
        self.assertFalse(args.record_perf_baseline)

    def test_skip_perf_flag(self):
        self.assertTrue(ci.parse_args(["--skip-perf"]).skip_perf)

    def test_record_baseline_flag(self):
        self.assertTrue(
            ci.parse_args(["--record-perf-baseline"]).record_perf_baseline
        )


class TestRealMeasurement(unittest.TestCase):
    """Настоящий `vtm run`: подменённый эталон обязан покраснеть.

    Скипается на чистом клоне: после `make clean` бинаря нет, и шаг CI тоже
    скипается — тест не должен требовать артефактов.
    """

    @classmethod
    def setUpClass(cls) -> None:
        if not (ci.PERF_VTM_BIN.is_file() and ci.PERF_MODEL.is_file()):
            raise unittest.SkipTest(
                "нет src/vtm/vtm или demo/model.vtm — C-часть не собрана"
            )
        cls.samples = ci.measure_perf(repeats=2)

    def test_real_measurement_returns_plausible_speed(self):
        self.assertGreaterEqual(len(self.samples), 2)
        for sample in self.samples:
            self.assertGreater(sample, 0.0)
        # Demo-модель крошечная: меньше 1 тыс. ток/с означало бы, что меряем
        # не то (например, в сборке включили санитайзеры).
        self.assertGreater(statistics.median(self.samples), 1000.0)

    def test_real_slowdown_by_substituted_baseline_reds(self):
        machine = ci.perf_machine()
        saved = json.loads(BASELINE.read_text(encoding="utf-8"))["machine"]
        if (saved["os"], saved["arch"], saved["cpu"]) != (
            machine["os"],
            machine["arch"],
            machine["cpu"],
        ):
            self.skipTest("эталон снят на другой машине")
        real = statistics.median(self.samples)
        data = json.loads(BASELINE.read_text(encoding="utf-8"))
        data["metric"]["median"] = real * 2.0
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "slow.json"
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            outcome = ci.step_perf(
                baseline_path=path,
                measure=lambda: list(self.samples),
            )
        self.assertEqual(outcome.status, "fail")
        self.assertIn("медленнее эталона", outcome.detail)


if __name__ == "__main__":
    unittest.main()
