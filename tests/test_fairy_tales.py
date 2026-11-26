"""Плагин сказок (задача 0218): распознавание, выбор, озвучка, реестр.

Тесты не ходят в сеть: корпус лежит в репозитории, а «пустой» и «числовой»
корпуса создаются во временных каталогах. Живой Викитека-источник проверяется
один раз при подготовке текстов, а не в наборе.
"""

from __future__ import annotations

import importlib.util
import random
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eva.plugins import (  # noqa: E402
    Permissions,
    PluginContext,
    default_registry,
    load_manifest,
)

PLUGIN_DIR = ROOT / "plugins" / "fairy-tales"


def _load_plugin_module():
    """Загрузить ``plugins/fairy-tales/plugin.py`` по пути (id с дефисом)."""
    path = PLUGIN_DIR / "plugin.py"
    name = "_eva_plugin_fairy_tales_test"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fairy = _load_plugin_module()


def make_ctx(plugin_dir: Path = PLUGIN_DIR) -> PluginContext:
    return PluginContext(
        "fairy-tales",
        plugin_dir=plugin_dir,
        permissions=Permissions(filesystem=("tales",)),
    )


def title_of(answer: str | None) -> str | None:
    if not answer:
        return None
    match = re.search(r"«([^»]+)»", answer)
    return match.group(1) if match else None


class TestRecognition(unittest.TestCase):
    """10+ живых формулировок, которые обязаны включать сказку."""

    PHRASES = (
        "Расскажи сказку",
        "Расскажи мне сказку, пожалуйста",
        "Сказку про лису",
        "Расскажи сказку про колобка",
        "Сказка на ночь",
        "Почитай сказку перед сном",
        "Расскажи что-нибудь интересное ребёнку",
        "Хочу сказку",
        "Про репку",
        "Про медведя",
        "Расскажи сказку про зиму",
        "Сказку про кота",
        "Расскажи историю про петуха",
        "Про вершки и корешки",
    )

    def setUp(self) -> None:
        self.plugin = fairy.FairyTalesPlugin(rng=random.Random(7))
        self.ctx = make_ctx()

    def test_phrases_are_recognized(self) -> None:
        for phrase in self.PHRASES:
            with self.subTest(phrase=phrase):
                answer = self.plugin.answer(self.ctx, phrase)
                self.assertIsInstance(answer, str)
                assert answer is not None
                self.assertIn("Слушай сказку", answer)
                self.assertTrue(
                    "Сказке конец" in answer or "Сладких снов" in answer
                )
                self.assertTrue(title_of(answer))

    def test_not_our_topic_returns_none(self) -> None:
        for phrase in (
            "Который час",
            "Включи музыку",
            "Расскажи про погоду",
            "Сколько стоит доллар",
            "Расскажи анекдот",
            "Как дела",
            "Следующая песня",
            "",
        ):
            with self.subTest(phrase=phrase):
                self.assertIsNone(self.plugin.answer(self.ctx, phrase))

    def test_empty_question_returns_none_without_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tales").mkdir()
            ctx = make_ctx(root)
            plugin = fairy.FairyTalesPlugin(tales_dir=root / "tales")
            self.assertIsNone(plugin.answer(ctx, "Расскажи сказку"))
            self.assertIsNone(plugin.answer(ctx, "Следующая сказка"))
            self.assertIsNone(plugin.answer(ctx, "Про колобка"))


class TestSelection(unittest.TestCase):
    """Выбор по названию/тегу, «следующая» и состояние."""

    def setUp(self) -> None:
        self.plugin = fairy.FairyTalesPlugin(rng=random.Random(11))
        self.ctx = make_ctx()

    def test_tag_and_title_selection(self) -> None:
        cases = {
            "Расскажи сказку про лису": "Лиса и журавль",
            "Сказку про зиму": "Зимовье зверей",
            "Про колобка": "Колобок",
            "Про репку": "Репка",
            "Про курочку": "Курочка Ряба",
            "Про теремок": "Теремок",
            "Про вершки": "Вершки и корешки",
            "Про журавля": "Лиса и журавль",
        }
        for phrase, expected in cases.items():
            with self.subTest(phrase=phrase):
                self.assertEqual(title_of(self.plugin.answer(self.ctx, phrase)), expected)

    def test_bear_matches_known_bear_tale(self) -> None:
        answer = self.plugin.answer(self.ctx, "Расскажи сказку про медведя")
        self.assertIn(
            title_of(answer),
            {"Три медведя", "Вершки и корешки", "Зимовье зверей", "Теремок"},
        )

    def test_unknown_target_returns_none(self) -> None:
        self.assertIsNone(
            self.plugin.answer(self.ctx, "Расскажи сказку про слона")
        )

    def test_next_goes_in_order_and_wraps(self) -> None:
        tales = self.plugin.load_tales(self.ctx)
        self.assertEqual(len(tales), 9)
        first = self.plugin.answer(self.ctx, "Про колобка")
        self.assertEqual(title_of(first), "Колобок")
        self.assertEqual(self.plugin.last_index, 0)
        second = self.plugin.answer(self.ctx, "Следующая сказка")
        self.assertEqual(title_of(second), tales[1].title)
        third = self.plugin.answer(self.ctx, "Ещё сказку")
        self.assertEqual(title_of(third), tales[2].title)
        self.assertEqual(self.plugin.last_index, 2)

        self.plugin.answer(self.ctx, "Про зимовье")
        self.assertEqual(self.plugin.last_index, len(tales) - 1)
        wrapped = self.plugin.answer(self.ctx, "Следующая сказка")
        self.assertEqual(title_of(wrapped), tales[0].title)

    def test_night_ending_survives_next(self) -> None:
        first = self.plugin.answer(self.ctx, "Сказка на ночь")
        assert first is not None
        self.assertIn("Слушай сказку на ночь", first)
        self.assertIn("Сладких снов", first)

        second = self.plugin.answer(self.ctx, "Следующая сказка")
        assert second is not None
        self.assertIn("Сладких снов", second)
        self.assertNotIn("кто слушал — молодец", second)

    def test_generic_request_is_short(self) -> None:
        answer = self.plugin.answer(self.ctx, "Расскажи сказку")
        title = title_of(answer)
        tales = {tale.title: tale for tale in self.plugin.load_tales(self.ctx)}
        assert title is not None
        self.assertTrue(tales[title].is_short)

    def test_next_without_state_returns_none(self) -> None:
        self.assertIsNone(self.plugin.answer(self.ctx, "Продолжи"))


class TestSpeechText(unittest.TestCase):
    """Ответ пригоден для Silero: без markdown, цифр и длинных абзацев."""

    def setUp(self) -> None:
        self.plugin = fairy.FairyTalesPlugin(rng=random.Random(3))
        self.ctx = make_ctx()
        self.tales = self.plugin.load_tales(self.ctx)

    def test_all_tales_are_clean(self) -> None:
        self.assertGreaterEqual(len(self.tales), 6)
        for tale in self.tales:
            for night in (False, True):
                with self.subTest(tale=tale.slug, night=night):
                    text = self.plugin.render(tale, night=night)
                    self.assertTrue(text.startswith("Слушай сказку"))
                    self.assertIsNone(re.search(r"\d", text))
                    self.assertIsNone(re.search(r"[#*`\[\]]", text))
                    self.assertNotIn("http", text)
                    self.assertNotIn("Примечания", text)
                    if night:
                        self.assertIn("Сладких снов", text)
                    else:
                        self.assertIn("Сказке конец, а кто слушал — молодец.", text)

    def test_paragraphs_are_short(self) -> None:
        for tale in self.tales:
            text = self.plugin.render(tale)
            paragraphs = [part for part in text.split("\n\n") if part]
            self.assertGreaterEqual(len(paragraphs), 3)
            self.assertLessEqual(max(len(part) for part in paragraphs), 600)

    def test_numbers_become_words(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tales_dir = root / "tales"
            tales_dir.mkdir()
            (tales_dir / "numbers.md").write_text(
                "---\n"
                "title: Числовая сказка\n"
                "tags: числа, короткая\n"
                "age: 3+\n"
                "source: локальный тест\n"
                "---\n\n"
                "У бабушки 3 яблока. Она дала внуку 1500 конфет.\n",
                encoding="utf-8",
            )
            plugin = fairy.FairyTalesPlugin(tales_dir=tales_dir)
            ctx = make_ctx(root)
            answer = plugin.answer(ctx, "Расскажи сказку")
            assert answer is not None
            self.assertIn("три яблока", answer)
            self.assertIn("тысяча пятьсот конфет", answer)
            self.assertIsNone(re.search(r"\d", answer))


class TestManifestAndRegistry(unittest.TestCase):
    """Манифест и подключение к реестру: capability видна снаружи."""

    def test_manifest_has_answer_and_no_network(self) -> None:
        manifest = load_manifest(PLUGIN_DIR)
        self.assertEqual(manifest.id, "fairy-tales")
        self.assertEqual(manifest.capabilities, ("answer",))
        self.assertEqual(manifest.permissions.network, ())
        self.assertIn("tales", manifest.permissions.filesystem)

    def test_registry_answers_contains_fairy_tales(self) -> None:
        registry = default_registry(reload=True)
        bindings = registry.answers()
        self.assertIn(
            "fairy-tales",
            [binding.plugin_id for binding in bindings],
            "fairy-tales должен быть включён в config/plugins.json",
        )
        text = registry.run_answer("Расскажи сказку про лису")
        self.assertIsInstance(text, str)
        assert text is not None
        self.assertEqual(title_of(text), "Лиса и журавль")


if __name__ == "__main__":
    unittest.main()
