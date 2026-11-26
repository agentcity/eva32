"""Плагин сказок (задачи 0218/0220): корпус, распознавание, выбор, озвучка.

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
TALES_DIR = PLUGIN_DIR / "tales"

FOX_TITLES = {
    "Лиса и рак",
    "Лиса и журавль",
    "Кот и лиса",
    "Кот, петух и лиса",
    "Лиса, заяц и петух",
    "Снегурушка и лиса",
}


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


def answer_text(answer: object) -> str | None:
    """Сервер принимает и строку, и словарь ``{"text", "images"}``."""
    if isinstance(answer, dict):
        value = answer.get("text")
        return value if isinstance(value, str) else None
    return answer if isinstance(answer, str) else None


def title_of(answer: object) -> str | None:
    text = answer_text(answer)
    if not text:
        return None
    match = re.search(r"«([^»]+)»", text)
    return match.group(1) if match else None


def split_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    assert raw.startswith("---\n")
    end = raw.find("\n---", 3)
    assert end >= 0
    header = raw[4:end]
    body = raw[end + 4 :].lstrip("\n")
    data: dict[str, str] = {}
    for line in header.splitlines():
        key, separator, value = line.partition(":")
        assert separator, f"битая строка шапки: {line!r}"
        data[key.strip()] = value.strip()
    return data, body


class TestRecognition(unittest.TestCase):
    """Живые формулировки, которые обязаны включать сказку."""

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
        "Расскажи волшебную сказку",
        "Сказку про бабу-ягу",
        "Расскажи короткую сказку",
        "Сказку для малыша",
        "Про снегурочку",
        "Про зайца",
    )

    def setUp(self) -> None:
        self.plugin = fairy.FairyTalesPlugin(rng=random.Random(7))
        self.ctx = make_ctx()

    def test_phrases_are_recognized(self) -> None:
        for phrase in self.PHRASES:
            with self.subTest(phrase=phrase):
                answer = self.plugin.answer(self.ctx, phrase)
                text = answer_text(answer)
                assert text is not None
                self.assertIn("Слушай сказку", text)
                self.assertTrue(
                    "Сказке конец" in text or "Сладких снов" in text
                )
                self.assertTrue(title_of(answer))

    def test_count_phrase(self) -> None:
        tales = self.plugin.load_tales(self.ctx)
        for phrase in (
            "Сколько у тебя сказок?",
            "Сколько сказок у тебя есть?",
            "Много ли у тебя сказок?",
        ):
            with self.subTest(phrase=phrase):
                answer = self.plugin.answer(self.ctx, phrase)
                assert answer is not None
                self.assertIn(f"{fairy.numbers_to_words(str(len(tales)))}", answer)
                self.assertIn("сказ", answer)
                self.assertIsNone(re.search(r"\d", answer))
                self.assertGreaterEqual(answer.count("«"), 2)

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
            self.assertIsNone(plugin.answer(ctx, "Сколько у тебя сказок?"))


class TestSelection(unittest.TestCase):
    """Выбор по названию/тегу, «следующая» и состояние."""

    def setUp(self) -> None:
        self.plugin = fairy.FairyTalesPlugin(rng=random.Random(11))
        self.ctx = make_ctx()
        self.tales = self.plugin.load_tales(self.ctx)

    def _tale(self, title: str) -> fairy.Tale:
        return next(tale for tale in self.tales if tale.title == title)

    def test_tag_and_title_selection(self) -> None:
        cases = {
            "Сказку про зиму": "Зимовье зверей",
            "Про колобка": "Колобок",
            "Про репку": "Репка",
            "Про курочку": "Курочка Ряба",
            "Про теремок": "Теремок",
            "Про вершки": "Вершки и корешки",
            "Про журавля": "Лиса и журавль",
            "Про зайца": "Лиса, заяц и петух",
        }
        for phrase, expected in cases.items():
            with self.subTest(phrase=phrase):
                self.assertEqual(title_of(self.plugin.answer(self.ctx, phrase)), expected)

    def test_fox_request_returns_fox_tale(self) -> None:
        title = title_of(self.plugin.answer(self.ctx, "Расскажи сказку про лису"))
        self.assertIn(title, FOX_TITLES)
        tale = self._tale(title)
        self.assertTrue("лиса" in tale.tags or "лиса" in tale.title.lower())

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
        self.assertGreaterEqual(len(self.tales), 25)
        first = self.plugin.answer(self.ctx, "Про колобка")
        self.assertEqual(title_of(first), "Колобок")
        kolobok_index = next(
            index for index, tale in enumerate(self.tales) if tale.title == "Колобок"
        )
        self.assertEqual(self.plugin.last_index, kolobok_index)

        second = self.plugin.answer(self.ctx, "Следующая сказка")
        self.assertEqual(title_of(second), self.tales[kolobok_index + 1].title)
        third = self.plugin.answer(self.ctx, "Ещё сказку")
        self.assertEqual(title_of(third), self.tales[kolobok_index + 2].title)

        first_safe = next(
            tale.title for tale in self.tales if not tale.is_scary
        )
        self.plugin.answer(self.ctx, "Про золотую рыбку")
        self.assertEqual(self.plugin.last_index, len(self.tales) - 1)
        wrapped = self.plugin.answer(self.ctx, "Следующая сказка")
        self.assertEqual(title_of(wrapped), first_safe)

    def test_night_ending_survives_next(self) -> None:
        first = answer_text(self.plugin.answer(self.ctx, "Сказка на ночь"))
        assert first is not None
        self.assertIn("Слушай сказку на ночь", first)
        self.assertIn("Сладких снов", first)

        second = answer_text(self.plugin.answer(self.ctx, "Следующая сказка"))
        assert second is not None
        self.assertIn("Сладких снов", second)
        self.assertNotIn("кто слушал — молодец", second)

    def test_generic_request_is_not_very_long(self) -> None:
        answer = self.plugin.answer(self.ctx, "Расскажи сказку")
        title = title_of(answer)
        tale = self._tale(title)
        self.assertLessEqual(len(tale.body), fairy.DEFAULT_LIMIT)

    def test_next_without_state_returns_none(self) -> None:
        self.assertIsNone(self.plugin.answer(self.ctx, "Продолжи"))


class TestNewPhrases(unittest.TestCase):
    """«Волшебная», «баба-яга», «короткая», «для малыша», «страшная»."""

    def setUp(self) -> None:
        self.plugin = fairy.FairyTalesPlugin(rng=random.Random(3))
        self.ctx = make_ctx()
        self.tales = self.plugin.load_tales(self.ctx)
        self.by_title = {tale.title: tale for tale in self.tales}

    def _answer_tale(self, phrase: str) -> fairy.Tale:
        answer = self.plugin.answer(self.ctx, phrase)
        assert answer is not None
        title = title_of(answer)
        assert title is not None
        return self.by_title[title]

    def test_magic(self) -> None:
        tale = self._answer_tale("Расскажи волшебную сказку")
        self.assertIn("волшебная", tale.tags)
        self.assertFalse(tale.is_scary)

    def test_baba_yaga(self) -> None:
        tale = self._answer_tale("Расскажи сказку про бабу-ягу")
        self.assertIn("баба-яга", tale.tags)
        self.assertFalse(tale.is_scary)

    def test_short(self) -> None:
        tale = self._answer_tale("Расскажи короткую сказку")
        self.assertLessEqual(len(tale.body), 800)

    def test_toddler(self) -> None:
        tale = self._answer_tale("Сказку для малыша")
        self.assertEqual(tale.age, "3+")

    def test_scary_only_on_explicit_request(self) -> None:
        tale = self._answer_tale("Расскажи страшную сказку")
        self.assertIn("страшная", tale.tags)

    def test_scary_never_in_random_choice(self) -> None:
        phrases = (
            "Расскажи сказку",
            "Расскажи волшебную сказку",
            "Расскажи короткую сказку",
            "Сказку для малыша",
            "Сказку про бабу-ягу",
        )
        for seed in range(40):
            plugin = fairy.FairyTalesPlugin(rng=random.Random(seed))
            for phrase in phrases:
                with self.subTest(seed=seed, phrase=phrase):
                    answer = plugin.answer(self.ctx, phrase)
                    assert answer is not None
                    tale = self.by_title[title_of(answer)]
                    self.assertNotIn("страшная", tale.tags)


class TestSpeechText(unittest.TestCase):
    """Ответ пригоден для Silero: без markdown, цифр и длинных абзацев."""

    def setUp(self) -> None:
        self.plugin = fairy.FairyTalesPlugin(rng=random.Random(3))
        self.ctx = make_ctx()
        self.tales = self.plugin.load_tales(self.ctx)

    def test_all_tales_are_clean(self) -> None:
        self.assertGreaterEqual(len(self.tales), 25)
        for tale in self.tales:
            for night in (False, True):
                with self.subTest(tale=tale.slug, night=night):
                    text = self.plugin.render(tale, night=night)
                    self.assertTrue(text.startswith("Слушай сказку"))
                    self.assertIsNone(re.search(r"\d", text))
                    self.assertIsNone(re.search(r"[#*`\[\]]", text))
                    self.assertIsNone(re.search(r"[A-Za-z]", text))
                    self.assertNotIn("http", text)
                    self.assertNotIn("Примечания", text)
                    self.assertNotIn("вариант", text.lower())
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


class TestCorpus(unittest.TestCase):
    """Корпус: количество, шапка, источник, возраст, объём."""

    def setUp(self) -> None:
        self.plugin = fairy.FairyTalesPlugin()
        self.ctx = make_ctx()
        self.tales = self.plugin.load_tales(self.ctx)

    def test_at_least_25_tales(self) -> None:
        self.assertGreaterEqual(len(self.tales), 25)
        self.assertEqual(len({tale.slug for tale in self.tales}), len(self.tales))

    def test_frontmatter_is_complete(self) -> None:
        for path in sorted(TALES_DIR.glob("*.md")):
            with self.subTest(path=path.name):
                data, body = split_frontmatter(path.read_text(encoding="utf-8"))
                self.assertTrue(data.get("title"))
                self.assertTrue(data.get("tags"))
                self.assertIn(data.get("age"), {"3+", "5+", "7+"})
                self.assertTrue(
                    data.get("source", "").startswith(
                        "https://ru.wikisource.org/wiki/"
                    )
                )
                self.assertGreater(len(body.strip()), 200)

    def test_corpus_size(self) -> None:
        total = sum(path.stat().st_size for path in TALES_DIR.glob("*.md"))
        self.assertLessEqual(total, 120 * 1024, f"корпус {total} байт")

    def test_scary_tales_are_marked(self) -> None:
        scary = [tale for tale in self.tales if tale.is_scary]
        self.assertGreaterEqual(len(scary), 1)
        for tale in scary:
            self.assertIn(tale.age, {"7+"})


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
        self.assertIn(title_of(text), FOX_TITLES)


if __name__ == "__main__":
    unittest.main()
