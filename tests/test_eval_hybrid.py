"""Проверки замера гибрида: формат набора, модель цены и текущие метрики.

Запуск::

    python3 -m unittest tests.test_eval_hybrid -v

Никакой сети: ``tools/eval_hybrid.py`` подменяет клиентов моделей заглушками,
``eva.router.answer`` вызывается только локально. Если модули ``eva`` ещё не
написаны (их пишут параллельно), тест метрик **пропускается** с внятной
причиной; проверки формата набора и формулы цены работают всегда.

Метрики зафиксированы в ``BASELINE``: это пороги «не хуже, чем в прогоне,
которым они сняты». Улучшение (выше точность, больше скиллов, ниже цена,
больше проверенного содержания) тест не ломает — регрессией считается только
ухудшение. Отдельно проверяется, что не появилось **новых** промахов
маршрутизации: набор ``BASELINE_MISROUTES`` можно только сокращать.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

GOLDEN = os.path.join(ROOT, "eval", "hybrid_golden.jsonl")
TOOL = os.path.join(ROOT, "tools", "eval_hybrid.py")


def _load_tool():
    """Загрузить tools/eval_hybrid.py по пути (tools не пакет)."""
    spec = importlib.util.spec_from_file_location("eval_hybrid_tool", TOOL)
    if spec is None or spec.loader is None:  # pragma: no cover - защита от кривого пути
        raise RuntimeError(f"не удалось загрузить {TOOL}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


eval_hybrid = _load_tool()

#: Зафиксированные метрики прогона. Числа — минимум/максимум, а не «ровно»:
#: улучшение допустимо, ухудшение видно. ``measured_at`` — дата прогона.
BASELINE = {
    # Перебазировка после задачи 0038: промпт получил правило о переводе, вход
    # подорожал (188,20 → 230,15). Переснято на том же наборе, маршруты те же 60/60.
    "measured_at": "2026-11-18",
    "total": 60,
    "accuracy": 1.00,
    "skill_share": 0.40,
    "cost_per_1000_rub": 230.15,
    "content_checked": 24,
    "content_ok": 24,
}

#: Допуск на цену: правки в промпте/тексте ответа меняют оценку токенов,
#: поэтому цена сравнивается с запасом, а не до копейки.
COST_TOLERANCE = 0.05

#: Вопросы, которые на момент фиксации уходили не туда. Список может только
#: сокращаться: новый промах — регрессия, даже если общая точность не упала.
BASELINE_MISROUTES: set[str] = set()  # промахов нет: 60/60 после задачи 0023


def _hybrid_ready() -> tuple[bool, str]:
    """Есть ли всё для честного замера: модули, каталог и набор."""
    try:
        missing = eval_hybrid.missing_core()
    except Exception as exc:  # noqa: BLE001 — причина попадёт в skip, а не в трейсбек
        return False, f"модули eva сломаны: {type(exc).__name__}: {exc}"
    if missing:
        names = ", ".join(missing)
        return False, f"модули ещё не написаны: {names}"
    if not os.path.exists(eval_hybrid.CATALOG):
        return False, "нет data/catalog.json — скиллам не из чего отвечать"
    return True, ""


READY, SKIP_REASON = _hybrid_ready()


class TestGoldenSet(unittest.TestCase):
    """Формат набора: его можно менять, но 40–60 строк и три маршрута — нет."""

    def setUp(self):
        self.rows = eval_hybrid.load_golden(GOLDEN)

    def test_size_is_40_to_60(self):
        self.assertGreaterEqual(len(self.rows), 40)
        self.assertLessEqual(len(self.rows), 60)

    def test_every_row_has_question_and_route(self):
        for row in self.rows:
            self.assertTrue(row["question"].strip())
            self.assertIn(row["expect_route"], eval_hybrid.ROUTE_KINDS)
            self.assertIsInstance(row["expect_contains"], list)

    def test_questions_are_unique(self):
        questions = [row["question"] for row in self.rows]
        self.assertEqual(len(questions), len(set(questions)))

    def test_all_three_routes_are_covered(self):
        for kind in eval_hybrid.ROUTE_KINDS:
            count = sum(1 for row in self.rows if row["expect_route"] == kind)
            self.assertGreaterEqual(count, 10, f"маршрут {kind}: всего {count} строк")

    def test_skill_rows_have_content_checks(self):
        # Скилл — единственная детерминированная часть, где содержание вообще
        # можно проверить без модели. Пустой expect_contains у скилла — дыра.
        skill_rows = [row for row in self.rows if row["expect_route"] == "skill"]
        checked = [row for row in skill_rows if row["expect_contains"]]
        self.assertGreaterEqual(len(checked), 18)
        self.assertEqual(len(checked), len(skill_rows))


class TestPriceModel(unittest.TestCase):
    """Формула цены на 1000 запросов — отдельно от eva, чтобы не было сети."""

    prices = {"skill": (0.0, 0.0), "small": (0.0, 0.0), "big": (0.4, 1.2)}
    completion = {"skill": 0, "small": 120, "big": 220}

    def test_skill_and_small_are_free(self):
        self.assertEqual(eval_hybrid.request_cost("skill", 10_000, self.prices), 0.0)
        self.assertEqual(eval_hybrid.request_cost("small", 10_000, self.prices), 0.0)

    def test_big_request_cost(self):
        # 700 prompt-токенов по 0.4 ₽/1k + 220 completion по 1.2 ₽/1k.
        cost = eval_hybrid.request_cost("big", 700, self.prices, self.completion)
        self.assertAlmostEqual(cost, 0.7 * 0.4 + 0.22 * 1.2, places=9)

    def test_1000_requests_mix(self):
        kinds = ["skill", "small", "big"]
        questions = ["", "", "x" * 300]
        # prompt большой: 300/3 = 100 + 600 системных = 700 токенов.
        expected = (0.0 + 0.0 + (0.7 * 0.4 + 0.22 * 1.2)) / 3 * 1000
        actual = eval_hybrid.cost_per_1000(
            kinds, questions, self.prices, system_tokens=600, completion_tokens=self.completion
        )
        self.assertAlmostEqual(actual, expected, places=9)
        self.assertAlmostEqual(actual, 181.3333333333, places=6)

    def test_empty_set_is_zero(self):
        self.assertEqual(
            eval_hybrid.cost_per_1000([], [], self.prices, system_tokens=600), 0.0
        )

    def test_contents_ignore_case_spaces_and_yo(self):
        text = "Тариф «Старт» — 2 490 ₽/мес [1]. Возврат — 14 дней."
        self.assertTrue(eval_hybrid.contains_all(text, ["2490", "[1]"]))
        self.assertTrue(eval_hybrid.contains_all(text, ["возврат", "14"]))
        self.assertFalse(eval_hybrid.contains_all(text, ["990"]))
        self.assertTrue(eval_hybrid.contains_all("Доставка за МКАД — 790 ₽", ["790 ₽"]))


class TestEstimateTokens(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(eval_hybrid.estimate_tokens(""), 0)
        self.assertEqual(eval_hybrid.estimate_tokens("   "), 0)

    def test_short_text_is_one_token(self):
        self.assertEqual(eval_hybrid.estimate_tokens("да"), 1)

    def test_rough_scale(self):
        self.assertEqual(eval_hybrid.estimate_tokens("x" * 300), 100)


class TestScriptWithoutEva(unittest.TestCase):
    """Скрипт без модулей eva обязан сказать «не найден», а не упасть трейсбеком."""

    def test_missing_module_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "tools"))
            os.makedirs(os.path.join(tmp, "eval"))
            shutil.copy(TOOL, os.path.join(tmp, "tools", "eval_hybrid.py"))
            shutil.copy(GOLDEN, os.path.join(tmp, "eval", "hybrid_golden.jsonl"))
            env = dict(os.environ)
            env.pop("PYTHONPATH", None)  # иначе eva может приехать из репозитория
            result = subprocess.run(
                [sys.executable, os.path.join("tools", "eval_hybrid.py")],
                cwd=tmp,
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("модуль eva.router не найден", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertIn("tools/eval_hybrid.py", result.stderr)


@unittest.skipUnless(READY, SKIP_REASON)
class TestHybridMetrics(unittest.TestCase):
    """Текущие метрики гибрида. Пропуск — только пока модулей нет."""

    @classmethod
    def setUpClass(cls):
        cls.rows = eval_hybrid.load_golden(GOLDEN)
        cls.cfg = eval_hybrid.load_eval_config()
        cls.prices = eval_hybrid.prices_from_config(cls.cfg)
        cls.metrics = eval_hybrid.evaluate(cls.rows, cls.cfg, prices=cls.prices)
        print(
            "\nметрики гибрида: "
            f"маршруты {cls.metrics['correct']}/{cls.metrics['total']} "
            f"({cls.metrics['accuracy'] * 100:.1f}%), "
            f"скиллы {cls.metrics['skill_share'] * 100:.1f}%, "
            f"{cls.metrics['cost_per_1000_rub']:.2f} ₽/1000, "
            f"содержание {cls.metrics['content_ok']}/{cls.metrics['content_checked']}"
        )
        if os.environ.get("EVA_EVAL_VERBOSE"):
            print(eval_hybrid.format_report(cls.metrics))

    def test_total_matches_dataset(self):
        self.assertEqual(self.metrics["total"], len(self.rows))
        self.assertEqual(self.metrics["total"], BASELINE["total"])

    def test_no_answer_errors(self):
        self.assertEqual(
            self.metrics["errors"],
            0,
            f"ошибки answer(): {self.metrics['error_rows']}",
        )

    def test_accuracy_not_below_baseline(self):
        self.assertGreaterEqual(
            self.metrics["accuracy"],
            BASELINE["accuracy"],
            "точность маршрутов упала: "
            f"{self.metrics['accuracy']:.3f} < {BASELINE['accuracy']:.3f}",
        )

    def test_skill_share_not_below_baseline(self):
        self.assertGreaterEqual(
            self.metrics["skill_share"],
            BASELINE["skill_share"],
            "доля скиллов упала: "
            f"{self.metrics['skill_share']:.3f} < {BASELINE['skill_share']:.3f}",
        )

    def test_cost_not_above_baseline(self):
        # Сравниваем не только с текущей ценой, но и с ценой идеального
        # маршрута по разметке: исправление классификатора законно увеличивает
        # долю big (сейчас 3 вопроса ошибочно уходят в малую), и это не
        # регрессия. Потолок — идеальный микс + допуск; «всё в big» (513 ₽)
        # и любая деградация к нему не пройдут.
        ceiling = max(
            BASELINE["cost_per_1000_rub"], self.metrics["ideal_cost_per_1000_rub"]
        ) * (1 + COST_TOLERANCE)
        self.assertLessEqual(
            self.metrics["cost_per_1000_rub"],
            ceiling,
            "цена на 1000 запросов выросла сверх идеального микса: "
            f"{self.metrics['cost_per_1000_rub']:.2f} ₽ > {ceiling:.2f} ₽",
        )

    def test_content_not_worse(self):
        self.assertGreaterEqual(
            self.metrics["content_checked"],
            BASELINE["content_checked"],
            "стало меньше проверяемого содержания",
        )
        self.assertGreaterEqual(
            self.metrics["content_ok"],
            BASELINE["content_ok"],
            "ответы скиллов перестали содержать ожидаемые факты",
        )
        self.assertEqual(
            [
                record["question"]
                for record in self.metrics["records"]
                if record["actual_route"] == "skill" and not record["text"].strip()
            ],
            [],
            "маршрут skill с пустым ответом: скилл не сработал, а роутер его пропустил",
        )

    def test_no_new_misroutes(self):
        actual = {
            item["question"]
            for item in self.metrics["mismatches"]
            if item["actual_route"] != item["expect_route"]
        }
        new = sorted(actual - BASELINE_MISROUTES)
        self.assertEqual(new, [], f"новые промахи маршрутизации: {new}")



if __name__ == "__main__":
    unittest.main()
