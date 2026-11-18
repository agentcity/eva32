"""Проверки разбивки ``tools/eval_hybrid.py`` по классам ответа.

Запуск::

    python3 -m unittest tests.test_skills_eval -v

Замер идёт с заглушками моделей: ни сети, ни ключей. Если модулей ``eva``
ещё нет, метрическая часть пропускается с причиной, как в
``tests/test_eval_hybrid.py``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

TOOL = os.path.join(ROOT, "tools", "eval_hybrid.py")


def _load_tool():
    spec = importlib.util.spec_from_file_location("eval_hybrid_classes", TOOL)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"не удалось загрузить {TOOL}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


eval_hybrid = _load_tool()


def _ready() -> tuple[bool, str]:
    try:
        missing = eval_hybrid.missing_for_metrics()
    except Exception as exc:  # noqa: BLE001 — причина в skip
        return False, f"модули eva сломаны: {type(exc).__name__}: {exc}"
    if missing:
        return False, "модули ещё не написаны: " + ", ".join(missing)
    return True, ""


READY, SKIP_REASON = _ready()


class TestClassHelpers(unittest.TestCase):
    """Класс ответа и имена скиллов — без запуска всего замера."""

    def test_skill_class(self):
        self.assertEqual(
            eval_hybrid._answer_class(
                {"actual_route": "skill", "skill": "pricing", "error": None}
            ),
            "skill:pricing",
        )

    def test_skill_without_name(self):
        self.assertEqual(
            eval_hybrid._answer_class({"actual_route": "skill", "skill": None, "error": None}),
            "skill:?",
        )

    def test_model_classes(self):
        self.assertEqual(
            eval_hybrid._answer_class({"actual_route": "small", "skill": None, "error": None}),
            "small",
        )
        self.assertEqual(
            eval_hybrid._answer_class({"actual_route": "big", "skill": None, "error": None}),
            "big",
        )

    def test_error_class_wins(self):
        self.assertEqual(
            eval_hybrid._answer_class(
                {"actual_route": "big", "skill": None, "error": "RuntimeError: x"}
            ),
            "error",
        )

    def test_registered_skill_names(self):
        names = eval_hybrid.registered_skill_names()
        for expected in ("pricing", "hours", "refunds", "contacts", "time", "units", "math", "currency"):
            with self.subTest(expected=expected):
                self.assertIn(expected, names)

    def test_class_title(self):
        self.assertEqual(eval_hybrid._class_title("skill:pricing"), "скилл pricing")
        self.assertEqual(eval_hybrid._class_title("big"), "модель большая")


@unittest.skipUnless(READY, SKIP_REASON)
class TestEvalClock(unittest.TestCase):
    """Замер детерминирован: скилл времени получает фиксированное «сейчас»."""

    def test_requests_carry_fixed_now(self):
        types_module = __import__("eva.types", fromlist=["Request"])
        request = eval_hybrid._make_request(types_module, "Который час?")
        self.assertEqual(request.meta.get("now"), eval_hybrid.EVAL_NOW)

    def test_time_skill_is_reproducible(self):
        types_module = __import__("eva.types", fromlist=["Request"])
        router = __import__("eva.router", fromlist=["answer"])
        request = eval_hybrid._make_request(types_module, "Который час?")
        cfg = eval_hybrid.load_eval_config()
        first = router.answer(request, cfg, {"skills": lambda text, ctx: __import__(
            "eva.skills", fromlist=["try_skills"]
        ).try_skills(text, ctx)})
        self.assertEqual(first.skill, "time")
        self.assertIn("08:54", first.text)


@unittest.skipUnless(READY, SKIP_REASON)
class TestClassBreakdownMetrics(unittest.TestCase):
    """Отчёт показывает, какой скилл поймал, что ушло в модель и промахи."""

    @classmethod
    def setUpClass(cls):
        cls.rows = eval_hybrid.load_golden()
        cls.cfg = eval_hybrid.load_eval_config()
        cls.metrics = eval_hybrid.evaluate(cls.rows, cls.cfg)

    def test_classes_cover_all_rows(self):
        classes = self.metrics["classes"]
        self.assertTrue(classes)
        self.assertEqual(
            sum(entry["count"] for entry in classes.values()),
            self.metrics["total"],
        )

    def test_catalog_skills_are_counted(self):
        classes = self.metrics["classes"]
        for name in ("skill:pricing", "skill:hours", "skill:refunds", "skill:contacts"):
            with self.subTest(name=name):
                self.assertIn(name, classes)
                self.assertGreater(classes[name]["count"], 0)

    def test_model_rows_are_counted(self):
        # Golden каталога — это регрессия скиллов/RAG, поэтому вопросов «малой модели»
        # там может не быть. Проверяем механику на синтетических строках: отчёт обязан
        # посчитать модельные строки, если они есть.
        import json
        import tempfile

        rows = [
            {"question": "Привет!", "expect_route": "small"},
            {"question": "Сравни PostgreSQL и SQLite.", "expect_route": "big"},
            {"question": "Сколько стоит тариф «Старт»?", "expect_route": "skill"},
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False,
                                         encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            path = fh.name
        metrics = eval_hybrid.evaluate(eval_hybrid.load_golden(path),
                                       eval_hybrid.load_eval_config())
        classes = metrics["classes"]
        self.assertIn("small", classes)
        self.assertIn("big", classes)
        self.assertEqual(classes["small"]["count"], len(metrics["model_questions"]["small"]))
        self.assertEqual(classes["big"]["count"], len(metrics["model_questions"]["big"]))

    def test_skill_hits_and_unused(self):
        hits = self.metrics["skill_hits"]
        for name in eval_hybrid.registered_skill_names():
            self.assertIn(name, hits)
        # 0105: набор вырос до 80 строк, у новых классов появились вопросы —
        # пустых скиллов в замере больше нет.
        self.assertEqual(self.metrics["unused_skills"], [])
        for name in ("time", "units", "math", "currency"):
            with self.subTest(name=name):
                self.assertGreater(
                    hits[name], 0, f"у скилла {name} нет попаданий в наборе"
                )

    def test_no_content_misses_on_golden(self):
        self.assertEqual(self.metrics["content_misses"], [])

    def test_report_has_class_section(self):
        report = eval_hybrid.format_report(self.metrics)
        self.assertIn("разбивка по классам", report)
        self.assertIn("скилл pricing", report)
        # 0105: все четыре новых класса отвечают, поэтому строки «без попаданий»
        # в отчёте нет — она печатается только для пустых классов.
        self.assertNotIn("без попаданий", report)
        for name in ("time", "units", "math", "currency"):
            with self.subTest(name=name):
                self.assertIn(f"скилл {name}", report)
        self.assertIn("промахов содержания: 0", report)

    def test_routes_still_complete(self):
        self.assertEqual(self.metrics["correct"], self.metrics["total"])
        self.assertGreaterEqual(self.metrics["skill_share"], 0.40)


if __name__ == "__main__":
    unittest.main()
