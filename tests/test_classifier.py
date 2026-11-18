"""Проверки правилового классификатора EVA32.

Запуск: ``python3 -m unittest tests.test_classifier -v``

20 размеченных примеров (по 5 на категорию: скилл-подобные, простые, сложные,
мусор) и проверки того, что пороги берутся из конфига, а не из кода.
"""

import ast
import copy
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eva import classifier  # noqa: E402
from eva.types import Route  # noqa: E402

#: Пороги как их отдаёт eva/config.py (свойство Config.classifier).
CFG = {
    "classifier": {
        "long_chars": 120,
        "latin_ratio": 0.35,
        "reason_markers": classifier.DEFAULT_REASON_MARKERS,
    }
}

#: (текст, ожидаемый kind, категория)
CASES = [
    # --- скилл-подобные: факт из каталога, скилл отвечает детерминированно ---
    ("Сколько стоит тариф «Про»?", "skill", "skill"),
    ("Какая цена подписки на месяц?", "skill", "skill"),
    ("Во сколько вы открываетесь в субботу?", "skill", "skill"),
    ("Как вернуть деньги за подписку?", "skill", "skill"),
    ("Дайте телефон поддержки", "skill", "skill"),
    # --- простые: малая модель справляется сама ---
    ("Привет! Как дела?", "small", "simple"),
    ("Как тебя зовут?", "small", "simple"),
    ("Спасибо!", "small", "simple"),
    ("Расскажи коротко, что ты умеешь", "small", "simple"),
    ("Перефразируй: добрый день, коллеги", "small", "simple"),
    # --- сложные: рассуждение, код, перевод, счёт — нужна большая модель ---
    ("Объясни, почему троичная логика экономичнее бинарной", "big", "complex"),
    ("Сравни QAT и дистилляцию: что даёт больше для качества?", "big", "complex"),
    ("Напиши код на Python для парсинга JSON", "big", "complex"),
    ("Переведи этот абзац на английский", "big", "complex"),
    ("Посчитай, сколько запросов влезет в бюджет 3000 ₽ при цене 0.4 ₽", "big", "complex"),
    # --- мусор: не падаем, отвечаем малой моделью с низкой уверенностью ---
    ("", "small", "garbage"),
    ("   ", "small", "garbage"),
    ("asdf qwer zxcv", "small", "garbage"),
    ("1234567890 !!!", "small", "garbage"),
    ("фывапролдж", "small", "garbage"),
]

LONG_TEXT = (
    "Нужно переписать вступление к отчёту о пилоте: убрать повторы, сократить "
    "первый абзац и добавить короткий вывод о том, что эксперимент прошёл по "
    "плану и серьёзных замечаний нет."
)


class TestExamples(unittest.TestCase):
    """20 размеченных примеров: минимум 17 совпадений с ожидаемым маршрутом."""

    def test_examples_match_expected_route(self):
        for text, expected, category in CASES:
            with self.subTest(category=category, text=text):
                route = classifier.classify(text, CFG)
                self.assertEqual(
                    route.kind,
                    expected,
                    msg=f"{text!r}: ждали {expected}, получили {route.kind} ({route.reason})",
                )

    def test_accuracy_at_least_17_of_20(self):
        matches = 0
        mismatches = []
        for text, expected, category in CASES:
            route = classifier.classify(text, CFG)
            if route.kind == expected:
                matches += 1
            else:
                mismatches.append((category, text, expected, route.kind, route.reason))
        self.assertGreaterEqual(
            matches,
            17,
            msg=f"совпало {matches}/20; расхождения: {mismatches}",
        )
        self.assertEqual(len(CASES), 20)


class TestRouteContract(unittest.TestCase):
    """Route: kind из контракта, непустой reason, confidence в [0, 1]."""

    def test_fields(self):
        texts = [
            "Сколько стоит тариф?",
            "Привет!",
            "Объясни, почему небо голубое",
            "",
            "asdf",
            "Что такое троичная логика?",
            "quick brown fox",
        ]
        for text in texts:
            with self.subTest(text=text):
                route = classifier.classify(text, CFG)
                self.assertIsInstance(route, Route)
                self.assertIn(route.kind, classifier.KINDS)
                self.assertIsInstance(route.reason, str)
                self.assertTrue(route.reason.strip())
                self.assertGreaterEqual(route.confidence, 0.0)
                self.assertLessEqual(route.confidence, 1.0)

    def test_empty_and_garbage_do_not_crash(self):
        for value in (None, "", "   ", "\n\t", "!!!???", "1234567890", "😀😀"):
            with self.subTest(value=value):
                route = classifier.classify(value, CFG)
                self.assertEqual(route.kind, "small")
                self.assertTrue(route.reason.startswith(("empty_text", "garbage_text")))

    def test_non_string_input_does_not_crash(self):
        for value in (12345, b"hi", [1, 2, 3]):
            with self.subTest(value=value):
                route = classifier.classify(value, CFG)
                self.assertIn(route.kind, classifier.KINDS)
                self.assertTrue(route.reason)

    def test_greeting_variants_are_small(self):
        for text in ("Привет!", "Здравствуйте, как дела?", "Как у тебя дела?", "hello", "Спасибо!"):
            with self.subTest(text=text):
                route = classifier.classify(text, CFG)
                self.assertEqual(route.kind, "small")
                self.assertEqual(route.reason, "greeting")

    def test_bad_cfg_does_not_crash(self):
        for cfg in (42, "oops", object(), {"classifier": 7}):
            with self.subTest(cfg=type(cfg).__name__):
                route = classifier.classify("Объясни, почему это так", cfg)
                self.assertIn(route.kind, classifier.KINDS)
                self.assertTrue(route.reason)

    def test_garbage_confidence_is_low(self):
        empty = classifier.classify("", CFG)
        loud = classifier.classify("Спасибо!", CFG)
        self.assertLess(empty.confidence, loud.confidence)

    def test_deterministic(self):
        text = "Сравни два подхода"
        self.assertEqual(
            classifier.classify(text, CFG).to_dict(),
            classifier.classify(text, CFG).to_dict(),
        )


class TestThresholdsFromConfig(unittest.TestCase):
    """Пороги приходят аргументом cfg, а не зашиты в classify."""

    def test_long_chars_changes_route(self):
        self.assertGreater(len(LONG_TEXT), 120)
        self.assertEqual(classifier.classify(LONG_TEXT, CFG).kind, "big")
        self.assertEqual(classifier.classify(LONG_TEXT, None).kind, "big")

        relaxed = {"classifier": {"long_chars": 500}}
        route = classifier.classify(LONG_TEXT, relaxed)
        self.assertEqual(route.kind, "small")
        self.assertIn("short_simple", route.reason)

    def test_latin_ratio_threshold(self):
        text = "quick brown fox"
        self.assertEqual(classifier.classify(text, CFG).kind, "big")

        disabled = {"classifier": {"latin_ratio": 1.1}}
        self.assertEqual(classifier.classify(text, disabled).kind, "small")

        another = {"classifier": {"latin_min_letters": 100}}
        self.assertEqual(classifier.classify(text, another).kind, "small")

    def test_reason_markers_from_config(self):
        text = "Объясни, почему это так"
        self.assertEqual(classifier.classify(text, CFG).kind, "big")

        custom = {"classifier": {"reason_markers": ["докажи"]}}
        self.assertEqual(classifier.classify(text, custom).kind, "small")

    def test_catalog_fact_can_be_disabled(self):
        text = "Сколько стоит тариф «Про»?"
        self.assertEqual(classifier.classify(text, CFG).kind, "skill")

        disabled = {"classifier": {"catalog_fact_to_skill": False}}
        self.assertEqual(classifier.classify(text, disabled).kind, "small")

    def test_flat_config_is_accepted(self):
        flat = {"long_chars": 500, "latin_ratio": 1.1}
        self.assertEqual(classifier.classify(LONG_TEXT, flat).kind, "small")

    def test_broken_values_fall_back_to_defaults(self):
        broken = {"classifier": {"long_chars": "abc", "latin_ratio": None, "reason_markers": []}}
        self.assertEqual(classifier.classify("Сравни два подхода", broken).kind, "big")

    def test_config_object_from_eva_config(self):
        try:
            from eva import config as eva_config
        except Exception as exc:  # pragma: no cover - конфиг ещё не готов
            self.skipTest(f"eva.config недоступен: {exc}")
        cfg = eva_config.Config()
        self.assertEqual(classifier.classify("Сравни два подхода", cfg).kind, "big")
        self.assertEqual(classifier.classify("Привет!", cfg).kind, "small")
        self.assertIn(classifier.classify("", cfg).kind, classifier.KINDS)

    def test_cfg_is_not_mutated(self):
        cfg = copy.deepcopy(CFG)
        classifier.classify("Объясни, почему это так", cfg)
        self.assertEqual(cfg, CFG)

    def test_structural_task_beats_missing_config_marker(self):
        """«Напиши функцию» — тот же класс задач, что «напиши код»."""
        custom = {"classifier": {"reason_markers": ["докажи"]}}
        route = classifier.classify("Напиши функцию на Python", custom)
        self.assertEqual(route.kind, "big")
        self.assertTrue(route.reason.startswith("reasoning"))

    def test_math_without_config_marker(self):
        custom = {"classifier": {"reason_markers": ["докажи"]}}
        self.assertEqual(
            classifier.classify("Реши уравнение x^2 = 4", custom).kind,
            "big",
        )

    def test_contacts_synonyms_route_to_skill(self):
        self.assertEqual(
            classifier.classify("Как связаться с поддержкой?", CFG).kind,
            "skill",
        )

    def test_comparison_routes_to_big(self):
        route = classifier.classify("В чём разница между TCP и UDP?", CFG)
        self.assertEqual(route.kind, "big")
        self.assertTrue(route.reason.startswith("comparison"))


class TestCreativeAndKnowledgeGaps(unittest.TestCase):
    """Промахи из замера гибрида 0023: big→small на творчестве и общих знаниях."""

    #: Творческие задачи: малая модель их не тянет, нужна большая.
    CREATIVE = [
        "Придумай план запуска продукта за 30 дней.",
        "Составь план продвижения на месяц.",
        "Спланируй бюджет рекламной кампании на квартал.",
        "Придумай слоган для тарифа «Старт».",
        "Составь таблицу сравнения тарифов.",
    ]

    #: Структурные признаки: SQL/запрос, формула, схема — тоже большая.
    STRUCTURAL = [
        "Напиши SQL-запрос: топ-10 клиентов по выручке.",
        "Напиши запрос: клиенты с активными подписками за последний месяц.",
        "Сделай формулу для расчёта скидки.",
        "Нарисуй схему процесса согласования.",
    ]

    #: Вопросы на общие знания: каталога и скилла там нет — большая модель.
    KNOWLEDGE = [
        "Какая столица Австралии?",
        "Сколько будет 15% от 800?",
        "Кто такой Алан Тьюринг?",
        "В каком году вышел Python 3.0?",
        "Переведи на французский: добрый вечер.",
    ]

    def test_creative_tasks_are_big(self):
        for text in self.CREATIVE:
            with self.subTest(text=text):
                route = classifier.classify(text, CFG)
                self.assertEqual(
                    route.kind,
                    "big",
                    msg=f"{text!r}: {route.kind} ({route.reason})",
                )
                self.assertTrue(
                    route.reason.startswith(("task: ", "reasoning: ")),
                    msg=route.reason,
                )

    def test_structural_tasks_are_big(self):
        for text in self.STRUCTURAL:
            with self.subTest(text=text):
                route = classifier.classify(text, CFG)
                self.assertEqual(
                    route.kind,
                    "big",
                    msg=f"{text!r}: {route.kind} ({route.reason})",
                )

    def test_general_knowledge_is_big(self):
        for text in self.KNOWLEDGE:
            with self.subTest(text=text):
                route = classifier.classify(text, CFG)
                self.assertEqual(
                    route.kind,
                    "big",
                    msg=f"{text!r}: {route.kind} ({route.reason})",
                )

    def test_catalog_still_wins_over_general_knowledge(self):
        """«Сколько будет стоить тариф» — факт из каталога, а не общие знания."""
        for text in (
            "Сколько стоит тариф «Старт»?",
            "Сколько будет стоить тариф «Старт» при оплате за год?",
            "Сколько будет стоить доставка за МКАД?",
            "Во сколько вы открываетесь в субботу?",
            "Как вернуть деньги за подписку?",
            "Дайте телефон поддержки",
        ):
            with self.subTest(text=text):
                self.assertEqual(classifier.classify(text, CFG).kind, "skill")

    def test_catalog_still_wins_over_structural_marker(self):
        """«сколько запросов входит в тариф» — лимит из каталога, а не SQL-задача."""
        text = "Сколько запросов входит в тариф «Старт»?"
        self.assertEqual(classifier.classify(text, CFG).kind, "skill")

    def test_task_markers_come_from_config(self):
        text = "Придумай план запуска продукта"
        self.assertEqual(classifier.classify(text, CFG).kind, "big")

        custom = {"classifier": {"task_markers": ["набросай"]}}
        self.assertEqual(classifier.classify(text, custom).kind, "small")
        self.assertEqual(
            classifier.classify("Набросай план запуска", custom).kind,
            "big",
        )

    def test_structural_markers_come_from_config(self):
        text = "Напиши SQL-запрос по клиентам"
        self.assertEqual(classifier.classify(text, CFG).kind, "big")

        custom = {"classifier": {"structural_markers": ["набросай"]}}
        self.assertEqual(classifier.classify(text, custom).kind, "small")

    def test_knowledge_markers_come_from_config(self):
        text = "Какая столица Австралии?"
        self.assertEqual(classifier.classify(text, CFG).kind, "big")

        custom = {"classifier": {"knowledge_markers": ["былин"]}}
        self.assertEqual(classifier.classify(text, custom).kind, "small")

    def test_markers_accept_comma_separated_string(self):
        custom = {"classifier": {"task_markers": "придумай, составь"}}
        self.assertEqual(
            classifier.classify("Составь план на неделю", custom).kind,
            "big",
        )

    def test_reasons_have_distinct_prefixes(self):
        """Причина маршрута остаётся читаемой: разные классы — разные префиксы."""
        self.assertTrue(
            classifier.classify("Придумай план запуска продукта", CFG).reason.startswith("task: ")
        )
        self.assertTrue(
            classifier.classify("Напиши SQL-запрос по клиентам", CFG).reason.startswith("structural: ")
        )
        self.assertTrue(
            classifier.classify("Какая столица Австралии?", CFG).reason.startswith("general_knowledge: ")
        )
        self.assertTrue(
            classifier.classify("Объясни, почему небо голубое", CFG).reason.startswith("reasoning: ")
        )


class TestNoNetwork(unittest.TestCase):
    """Классификатор не имеет права ходить в сеть (правило 2 из docs/hybrid.md)."""

    def test_no_network_imports(self):
        source = Path(classifier.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        forbidden = {"socket", "ssl", "http", "urllib", "requests", "httpx", "aiohttp", "asyncio"}
        self.assertEqual(imported & forbidden, set(), msg=f"сетевые импорты: {imported & forbidden}")


if __name__ == "__main__":
    unittest.main()
