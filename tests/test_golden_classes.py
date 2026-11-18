"""Golden set: классы новых скиллов (задача 0105).

Задача 0105 расширила ``eval/hybrid_golden.jsonl``: к историческим 60 строкам
добавлены вопросы времени, единиц, математики и валюты. Смысл файла — держать
три вещи, ради которых набор расширен:

* **исторические 60 не тронуты** — их содержимое зафиксировано хешем, а
  маршруты, доля скиллов и цена сверяются с числами до расширения;
* **новые классы действительно ловятся скиллами** — не уходят в модель, и
  ``eval_hybrid`` видит их отдельными строками разбивки;
* **время воспроизводимо** — эталоны вопросов времени привязаны к
  фиксированному ``EVAL_NOW`` замера, а не к реальным часам.

Запуск::

    python3 -m unittest tests.test_golden_classes -v
"""

# 0105: набор вырос до 80 строк — 60 исторических + 20 вопросов новых классов.
from __future__ import annotations

import hashlib
import json
import os
import sys
import unittest
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

GOLDEN = os.path.join(ROOT, "eval", "hybrid_golden.jsonl")
TOOL = os.path.join(ROOT, "tools", "eval_hybrid.py")


def _load_tool():
    """Загрузить ``tools/eval_hybrid.py`` по пути (tools не пакет)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("eval_hybrid_golden_classes", TOOL)
    if spec is None or spec.loader is None:  # pragma: no cover - защита от кривого пути
        raise RuntimeError(f"не удалось загрузить {TOOL}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


eval_hybrid = _load_tool()

#: Сколько первых строк набора — историческое ядро (задачи 0019/0023/0038).
HISTORICAL_ROWS = 60
#: sha256 канонического JSON этих 60 строк на момент задачи 0105. Меняется
#: только вместе с историческими числами — иначе «поедут» прошлые метрики.
HISTORICAL_SHA256 = "50915c6a3dcea5e28d7952caabcc161c51552c3d46a7c6ed82e062e9bb56ccbd"
#: Минимум новых вопросов по требованию задачи и границы на один класс.
MIN_NEW_ROWS = 16
MIN_PER_CLASS, MAX_PER_CLASS = 4, 6
#: Новые скиллы задачи 0091, чьи классы должны быть в наборе.
NEW_SKILLS = ("time", "units", "math", "currency")
#: Фиксированное «сейчас» замера: эталоны времени обязаны с ним совпадать.
EVAL_NOW_LITERAL = "2026-11-18T08:54:00+03:00"
#: Русские названия для сверки эталонов времени — независимо от скилла.
MONTHS = (
    "", "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
)
WEEKDAYS = (
    "понедельник", "вторник", "среда", "четверг", "пятница", "суббота",
    "воскресенье",
)
#: Исторические числа по первым 60 строкам — сняты до задачи 0105.
#: 0119: 230,15 → 171,92. Цена зависит от длины промпта (входные токены):
#: промпт сжат 2691→1498 символов (897→500 токенов), маршруты и содержание
#: те же, поэтому пересчитан только вход.
HISTORICAL = {
    "total": 60,
    "correct": 60,
    "skill": 24,
    "skill_share": 0.40,
    "cost_per_1000_rub": 171.92,
    "content_checked": 24,
    "content_ok": 24,
}
#: Допуск на цену: оценка токенов по символам, а не BPE. 0119: сжатие промпта
#: законно снижает цену, поэтому сверху ограничиваем рост, а не точное число.
COST_TOLERANCE = 0.05
#: Новые базовые числа набора из 80 строк.
#: 0119: 172,61 → 128,94 по той же причине — короче промпт, дешевле вход.
EXTENDED = {
    "total": 80,
    "skill_share": 0.55,
    "cost_per_1000_rub": 128.94,
}


def _ready() -> tuple[bool, str]:
    """Есть ли всё для честного замера (модули eva и набор)."""
    try:
        missing = eval_hybrid.missing_for_metrics()
    except Exception as exc:  # noqa: BLE001 — причина попадёт в skip
        return False, f"модули eva сломаны: {type(exc).__name__}: {exc}"
    if missing:
        return False, "модули ещё не написаны: " + ", ".join(missing)
    return True, ""


READY, SKIP_REASON = _ready()


class TestGoldenLayout(unittest.TestCase):
    """Состав набора: файл, формат, неприкосновенность исторических строк."""

    @classmethod
    def setUpClass(cls):
        cls.rows = eval_hybrid.load_golden(GOLDEN)
        with open(GOLDEN, encoding="utf-8") as handle:
            cls.lines = [line for line in handle if line.strip()]
        cls.historical = cls.rows[:HISTORICAL_ROWS]
        cls.new_rows = cls.rows[HISTORICAL_ROWS:]
        cls.historical_digest = hashlib.sha256(
            json.dumps(cls.historical, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def test_rows_match_file_lines(self):
        # test_skills_probe читает файл через json.loads на каждой строке,
        # поэтому комментарии и пустые объекты недопустимы.
        self.assertEqual(len(self.rows), len(self.lines))
        self.assertEqual(len(self.rows), EXTENDED["total"])

    def test_every_row_has_expected_format(self):
        for row in self.rows:
            self.assertTrue(row["question"].strip())
            self.assertIn(row["expect_route"], eval_hybrid.ROUTE_KINDS)
            self.assertIsInstance(row["expect_contains"], list)
        for line in self.lines:
            raw = json.loads(line)
            self.assertLessEqual(
                set(raw),
                {"question", "expect_route", "expect_contains"},
                "в строке набора появилось поле вне формата eval_hybrid",
            )

    def test_questions_are_unique(self):
        questions = [row["question"] for row in self.rows]
        self.assertEqual(len(questions), len(set(questions)))

    def test_historical_rows_unchanged(self):
        # 0105: исторические 60 меняются только вместе с прошлыми метриками.
        self.assertEqual(len(self.historical), HISTORICAL["total"])
        self.assertEqual(
            self.historical_digest,
            HISTORICAL_SHA256,
            "первые 60 вопросов набора изменились: исторические числа поедут",
        )

    def test_new_class_rows_are_present(self):
        self.assertGreaterEqual(len(self.new_rows), MIN_NEW_ROWS)
        self.assertEqual(
            sum(1 for row in self.new_rows if row["expect_route"] == "skill"),
            len(self.new_rows),
            "вопросы новых классов должны ожидать маршрут skill",
        )
        for row in self.new_rows:
            self.assertTrue(
                row["expect_contains"],
                f"у вопроса нового класса нет эталона: {row['question']}",
            )


@unittest.skipUnless(READY, SKIP_REASON)
class TestNewClassesAreCaught(unittest.TestCase):
    """Новые классы считаются скиллами и видны в разбивке eval_hybrid."""

    @classmethod
    def setUpClass(cls):
        cls.rows = eval_hybrid.load_golden(GOLDEN)
        cls.cfg = eval_hybrid.load_eval_config()
        cls.prices = eval_hybrid.prices_from_config(cls.cfg)
        cls.metrics = eval_hybrid.evaluate(cls.rows, cls.cfg, prices=cls.prices)
        cls.historical_records = cls.metrics["records"][:HISTORICAL_ROWS]
        cls.new_records = cls.metrics["records"][HISTORICAL_ROWS:]

    def test_new_rows_are_answered_by_skills_not_models(self):
        for record in self.new_records:
            with self.subTest(question=record["question"]):
                self.assertEqual(record["actual_route"], "skill")
                self.assertIn(record["skill"], NEW_SKILLS)
                self.assertIsNone(record["error"])

    def test_each_new_class_has_four_to_six_questions(self):
        counts = {name: 0 for name in NEW_SKILLS}
        for record in self.new_records:
            counts[record["skill"]] = counts.get(record["skill"], 0) + 1
        for name, count in counts.items():
            with self.subTest(skill=name):
                self.assertGreaterEqual(
                    count, MIN_PER_CLASS, f"класс {name}: всего {count} вопросов"
                )
                self.assertLessEqual(
                    count, MAX_PER_CLASS, f"класс {name}: всего {count} вопросов"
                )

    def test_new_content_is_checked_and_ok(self):
        for record in self.new_records:
            with self.subTest(question=record["question"]):
                self.assertTrue(record["content_checked"])
                self.assertTrue(record["content_ok"], record["text"])
        new_misses = [
            item
            for item in self.metrics["content_misses"]
            if item["question"] in {record["question"] for record in self.new_records}
        ]
        self.assertEqual(new_misses, [])

    def test_class_breakdown_counts_new_skills(self):
        classes = self.metrics["classes"]
        for name in NEW_SKILLS:
            key = f"skill:{name}"
            with self.subTest(skill=name):
                self.assertIn(key, self.metrics["class_order"])
                entry = classes[key]
                self.assertGreaterEqual(entry["count"], MIN_PER_CLASS)
                self.assertLessEqual(entry["count"], MAX_PER_CLASS)
                self.assertEqual(entry["content_ok"], entry["content_checked"])

    def test_no_skill_is_unused(self):
        # До 0105 здесь были time/units/math/currency: в наборе не было их
        # вопросов. Теперь попадание есть у каждого класса.
        self.assertEqual(self.metrics["unused_skills"], [])
        for name in NEW_SKILLS:
            with self.subTest(skill=name):
                self.assertGreater(self.metrics["skill_hits"][name], 0)

    def test_extended_metrics_are_baseline(self):
        self.assertEqual(self.metrics["total"], EXTENDED["total"])
        self.assertEqual(self.metrics["errors"], 0)
        self.assertEqual(self.metrics["correct"], self.metrics["total"])
        self.assertEqual(self.metrics["mismatches"], [])
        self.assertAlmostEqual(
            self.metrics["skill_share"], EXTENDED["skill_share"], places=6
        )
        self.assertAlmostEqual(
            self.metrics["expected_skill_share"],
            EXTENDED["skill_share"],
            places=6,
        )
        # 0119: цена — производная от длины промпта, её проверяем мягко:
        # положительна и не выросла больше допуска. Маршруты и доли
        # выше проверены точно — именно они и есть поведение.
        self.assertGreater(self.metrics["cost_per_1000_rub"], 0.0)
        self.assertLessEqual(
            self.metrics["cost_per_1000_rub"],
            EXTENDED["cost_per_1000_rub"] * (1 + COST_TOLERANCE),
            "цена 80 строк выросла: "
            f"{self.metrics['cost_per_1000_rub']:.2f} ₽ > "
            f"{EXTENDED['cost_per_1000_rub'] * (1 + COST_TOLERANCE):.2f} ₽",
        )

    def test_historical_rows_keep_old_metrics(self):
        # 0105: добавление строк не должно было изменить первые 60.
        historical = eval_hybrid.evaluate(
            self.rows[:HISTORICAL_ROWS], self.cfg, prices=self.prices
        )
        self.assertEqual(historical["total"], HISTORICAL["total"])
        self.assertEqual(historical["correct"], HISTORICAL["correct"])
        self.assertEqual(historical["actual_counts"]["skill"], HISTORICAL["skill"])
        self.assertAlmostEqual(
            historical["skill_share"], HISTORICAL["skill_share"], places=6
        )
        # 0119: как и выше — цена мягко (зависит от промпта), поведение точно.
        self.assertGreater(historical["cost_per_1000_rub"], 0.0)
        self.assertLessEqual(
            historical["cost_per_1000_rub"],
            HISTORICAL["cost_per_1000_rub"] * (1 + COST_TOLERANCE),
            "цена исторических 60 выросла: "
            f"{historical['cost_per_1000_rub']:.2f} ₽ > "
            f"{HISTORICAL['cost_per_1000_rub'] * (1 + COST_TOLERANCE):.2f} ₽",
        )
        self.assertEqual(historical["content_checked"], HISTORICAL["content_checked"])
        self.assertEqual(historical["content_ok"], HISTORICAL["content_ok"])
        self.assertEqual(historical["mismatches"], [])


@unittest.skipUnless(READY, SKIP_REASON)
class TestTimeQuestionsAreReproducible(unittest.TestCase):
    """Время в наборе привязано к фиксированному ``EVAL_NOW`` замера."""

    @classmethod
    def setUpClass(cls):
        cls.rows = eval_hybrid.load_golden(GOLDEN)
        cls.cfg = eval_hybrid.load_eval_config()
        cls.new_records = eval_hybrid.evaluate(cls.rows, cls.cfg)["records"][
            HISTORICAL_ROWS:
        ]
        cls.time_records = [
            record for record in cls.new_records if record["skill"] == "time"
        ]
        cls.moment = datetime.fromisoformat(EVAL_NOW_LITERAL)

    def test_eval_now_is_fixed(self):
        self.assertEqual(eval_hybrid.EVAL_NOW, EVAL_NOW_LITERAL)
        self.assertEqual(
            (self.moment.year, self.moment.month, self.moment.day), (2026, 11, 18)
        )
        self.assertEqual((self.moment.hour, self.moment.minute), (8, 54))

    def test_time_answers_match_eval_now(self):
        expected_time = f"{self.moment.hour:02d}:{self.moment.minute:02d}"
        expected_date = (
            f"{self.moment.day} {MONTHS[self.moment.month]} {self.moment.year}"
        )
        expected_weekday = WEEKDAYS[self.moment.weekday()]
        self.assertEqual(expected_time, "08:54")
        self.assertEqual(expected_date, "18 ноября 2026")
        self.assertEqual(expected_weekday, "среда")

        texts = [record["text"] for record in self.time_records]
        self.assertEqual(len(texts), 5)
        self.assertTrue(
            any(expected_time in text for text in texts),
            "ни один вопрос о времени не ответил временем EVAL_NOW",
        )
        self.assertTrue(
            any(expected_date in text for text in texts),
            "ни один вопрос о дате не ответил датой EVAL_NOW",
        )
        self.assertTrue(
            any(expected_weekday in text for text in texts),
            "вопрос о дне недели не ответил днём недели EVAL_NOW",
        )

    def test_time_answers_do_not_depend_on_wall_clock(self):
        # Второй прогон с тем же EVAL_NOW обязан дать те же тексты: иначе
        # набор невоспроизводим и исторические числа бессмысленны.
        first = [record["text"] for record in self.time_records]
        second = [
            record["text"]
            for record in eval_hybrid.evaluate(self.rows, self.cfg)["records"][
                HISTORICAL_ROWS:
            ]
            if record["skill"] == "time"
        ]
        self.assertEqual(first, second)
        self.assertTrue(all(text for text in first))


if __name__ == "__main__":
    unittest.main()
