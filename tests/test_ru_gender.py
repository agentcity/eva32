"""Женский род Евы в первом лице: ``eva/ru_gender.py`` (задача 0207).

Запуск::

    python3 -m unittest tests.test_ru_gender -v

Ни сети, ни модели: проверяется чистая функция и потоковый стример.

Что проверяется:

1. **Пары** — 40+ мужских форм первого лица превращаются в женские: прошедшее
   время (-л → -ла), краткие прилагательные (готов → готова, уверен → уверена,
   должен → должна, обязан → обязана) и возвратные глаголы (-лся → -лась);
2. **Только первое лицо** — «он понял», «ты готов», «вы готовы», «мы сделали»
   не трогаются: правило работает от «я», а не от списка слов;
3. **Кавычки и код** — «…», „…“, "…" и бэктики/``` ``` ``` — чужой текст, его
   не переписываем; «я» из цитаты не перекрашивает слово снаружи;
4. **Идемпотентность** — второй прогон ничего не меняет, уже женские формы
   остаются как есть;
5. **Стриминг по словам** — куски любой нарезки (включая разрез посреди слова)
   дают тот же текст, что и :func:`feminize` по целому, а недописанное слово
   не уходит в поток;
6. **Подключение** — REPL ``tools/chat.py`` (общий путь для ``--once`` и REPL)
   печатает ответ с женскими формами и кладёт его же в историю.
"""

from __future__ import annotations

import io
import os
import re
import sys
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from eva.ru_gender import GenderStreamer, feminize  # noqa: E402
from tools import chat as chat_cli  # noqa: E402

#: Пары «мужская форма → женская»: 40+ штук, все из списка задачи 0207.
PAIRS = [
    ("понял", "поняла"),
    ("сделал", "сделала"),
    ("смог", "смогла"),
    ("успел", "успела"),
    ("готов", "готова"),
    ("уверен", "уверена"),
    ("рад", "рада"),
    ("должен", "должна"),
    ("обязан", "обязана"),
    ("нашёл", "нашла"),
    ("нашел", "нашла"),
    ("посмотрел", "посмотрела"),
    ("проверил", "проверила"),
    ("подумал", "подумала"),
    ("ответил", "ответила"),
    ("сказал", "сказала"),
    ("хотел", "хотела"),
    ("решил", "решила"),
    ("начал", "начала"),
    ("закончил", "закончила"),
    ("забыл", "забыла"),
    ("помнил", "помнила"),
    ("знал", "знала"),
    ("видел", "видела"),
    ("услышал", "услышала"),
    ("пришёл", "пришла"),
    ("пришел", "пришла"),
    ("ушёл", "ушла"),
    ("ушел", "ушла"),
    ("взял", "взяла"),
    ("дал", "дала"),
    ("ждал", "ждала"),
    ("исправил", "исправила"),
    ("добавил", "добавила"),
    ("убрал", "убрала"),
    ("ошибся", "ошиблась"),
    ("забылся", "забылась"),
    # Возвратные глаголы по общему правилу -лся → -лась.
    ("обрадовался", "обрадовалась"),
    ("старался", "старалась"),
    ("удивился", "удивилась"),
    ("согласился", "согласилась"),
    ("задержался", "задержалась"),
    ("проснулся", "проснулась"),
    ("расслабился", "расслабилась"),
    ("отправился", "отправилась"),
    ("постарался", "постаралась"),
]

#: Уже женские формы: стример не должен их «исправлять» повторно.
FEMININE_ALREADY = (
    "Я поняла.",
    "Я готова помочь.",
    "Я уже всё сделала.",
    "Я уверена в ответе.",
    "Я должна проверить.",
    "Я обрадовалась.",
)


class TestFeminizePairs(unittest.TestCase):
    """Мужская форма после «я» → женская."""

    def test_pair_count(self):
        self.assertGreaterEqual(len(PAIRS), 40)

    def test_pairs_in_first_person(self):
        for masculine, feminine in PAIRS:
            with self.subTest(word=masculine):
                self.assertEqual(
                    feminize(f"Я {masculine}."),
                    f"Я {feminine}.",
                    f"{masculine} должно стать {feminine} после «я»",
                )

    def test_short_adjectives(self):
        self.assertEqual(feminize("Я готов."), "Я готова.")
        self.assertEqual(feminize("Я уверен."), "Я уверена.")
        self.assertEqual(feminize("Я рад."), "Я рада.")
        self.assertEqual(feminize("Я должен."), "Я должна.")
        self.assertEqual(feminize("Я обязан."), "Я обязана.")

    def test_reflexive_verbs(self):
        for masculine, feminine in (
            ("обрадовался", "обрадовалась"),
            ("старался", "старалась"),
            ("удивился", "удивилась"),
            ("согласился", "согласилась"),
        ):
            with self.subTest(word=masculine):
                self.assertEqual(feminize(f"Я {masculine}."), f"Я {feminine}.")

    def test_particles_between_ya_and_form(self):
        cases = {
            "Я уже понял.": "Я уже поняла.",
            "Я всё проверил.": "Я всё проверила.",
            "Я не готов.": "Я не готова.",
            "Я бы сказал.": "Я бы сказала.",
            "Я же понял.": "Я же поняла.",
            "Я так решил.": "Я так решила.",
            # Две частицы — тоже первое лицо («я всё же проверил»).
            "Я всё же проверил.": "Я всё же проверила.",
            "Я уже всё проверил.": "Я уже всё проверила.",
        }
        for source, expected in cases.items():
            with self.subTest(text=source):
                self.assertEqual(feminize(source), expected)

    def test_free_word_order_with_dash(self):
        # Тире не разрывает связь «я → форма».
        self.assertEqual(feminize("Я — понял."), "Я — поняла.")

    def test_case_is_preserved(self):
        self.assertEqual(feminize("Я Понял."), "Я Поняла.")
        self.assertEqual(feminize("я ПОНЯЛ."), "я ПОНЯЛА.")
        self.assertEqual(feminize("Я понял!"), "Я поняла!")

    def test_markdown_emphasis_does_not_block(self):
        self.assertEqual(feminize("**Я понял** вопрос."), "**Я поняла** вопрос.")


class TestFeminizeNegatives(unittest.TestCase):
    """Чужой род, цитаты и код не трогаются."""

    def test_other_persons_are_untouched(self):
        for text in (
            "Он понял.",
            "Она поняла.",  # уже женское, но и не первое лицо Евы
            "Ты готов?",
            "Вы готовы!",
            "Они сделали.",
            "Мы сделали.",
            "Он уже всё проверил.",
            "Он сказал, что готов.",
        ):
            with self.subTest(text=text):
                self.assertEqual(feminize(text), text)

    def test_other_subject_after_ya_resets_chain(self):
        # «я» относится к «не он», а не к «понял».
        for text in (
            "Я не он, он понял.",
            "Я и он пришёл.",
            "Я знаю: ты готов.",
        ):
            with self.subTest(text=text):
                self.assertEqual(feminize(text), text)
        # Первое лицо меняется, чужое — нет, даже в одном предложении.
        self.assertEqual(
            feminize("Я сказал, что он должен."),
            "Я сказала, что он должен.",
        )

    def test_quoted_text_is_untouched(self):
        for text in (
            "Он сказал: «я понял».",
            'Он написал "я готов" и ушёл.',
            "Она ответила: „я не уверен“.",
            "«Я должен проверить», — сказал он.",
        ):
            with self.subTest(text=text):
                self.assertEqual(feminize(text), text)

    def test_quoted_ya_does_not_leak(self):
        # «я» внутри цитаты — не подлежащее следующего слова снаружи.
        self.assertEqual(feminize("«я» сказал он."), "«я» сказал он.")

    def test_code_is_untouched(self):
        for text in (
            "`я понял` — это код.",
            "```\nя не готов\n```",
            "Смотри: `if я понял:` и всё.",
        ):
            with self.subTest(text=text):
                self.assertEqual(feminize(text), text)

    def test_quote_around_verb_keeps_verb(self):
        # Меняется только первое лицо снаружи цитаты.
        self.assertEqual(
            feminize("Я сказал: «он готов»."),
            "Я сказала: «он готов».",
        )

    def test_words_outside_the_list_are_untouched(self):
        for text in ("Я думаю.", "Я иду.", "Я устал.", "Я программист."):
            with self.subTest(text=text):
                self.assertEqual(feminize(text), text)


class TestFeminizeIdempotence(unittest.TestCase):
    def test_second_pass_changes_nothing(self):
        texts = [
            "Я уже всё проверил и нашёл ошибку.",
            "Я не готов, но я уверен.",
            "Я обрадовался и сказал.",
            "Он понял, а я ответил.",
            "`я понял` и «я готов».",
            "«Я должен», — сказала она.",
        ]
        for text in texts:
            with self.subTest(text=text):
                once = feminize(text)
                self.assertEqual(feminize(once), once)

    def test_feminine_forms_are_kept(self):
        for text in FEMININE_ALREADY:
            with self.subTest(text=text):
                self.assertEqual(feminize(text), text)

    def test_empty_and_none(self):
        self.assertEqual(feminize(""), "")
        self.assertEqual(feminize(None), "")


class TestStreaming(unittest.TestCase):
    """Поток: слово целиком, дельты не рвутся."""

    def feed_all(self, chunks):
        streamer = GenderStreamer()
        pieces = [streamer.feed(chunk) for chunk in chunks]
        pieces.append(streamer.flush())
        return pieces

    def test_word_split_across_chunks(self):
        pieces = self.feed_all(["Я всё сдел", "ал, я ", "готов", "."])
        self.assertEqual("".join(pieces), "Я всё сделала, я готова.")
        # Недописанное слово не утекло в поток: «сдел» и «готов» ждали границы.
        self.assertEqual(pieces[0], "Я всё ")
        self.assertEqual(pieces[1], "сделала, я ")
        self.assertEqual(pieces[2], "")

    def test_no_piece_ends_with_a_half_word(self):
        text = "Я уже всё проверил и нашёл ошибку."
        for split in range(1, len(text)):
            pieces = self.feed_all([text[:split], text[split:]])
            for piece in pieces:
                if piece:
                    self.assertIsNone(
                        re.search(r"[А-Яа-яЁё]$", piece),
                        f"кусок {piece!r} кончился посреди слова "
                        f"(разрез {split})",
                    )

    def test_any_split_gives_same_text(self):
        text = (
            "Я уже всё проверил и нашёл ошибку. Он сказал: «я готов», "
            "а я ответил."
        )
        expected = feminize(text)
        self.assertEqual(expected, (
            "Я уже всё проверила и нашёл ошибку. Он сказал: «я готов», "
            "а я ответила."
        ))
        for split in range(len(text) + 1):
            with self.subTest(split=split):
                joined = "".join(self.feed_all([text[:split], text[split:]]))
                self.assertEqual(joined, expected)

    def test_byte_by_byte(self):
        text = "Я не уверен: `я готов` и «я должен»."
        streamer = GenderStreamer()
        got = "".join(streamer.feed(char) for char in text) + streamer.flush()
        self.assertEqual(got, "Я не уверена: `я готов` и «я должен».")

    def test_code_fence_split_between_chunks(self):
        streamer = GenderStreamer()
        got = (
            streamer.feed("Текст ```")
            + streamer.feed("\nя понял\n")
            + streamer.feed("``` и я понял.")
            + streamer.flush()
        )
        self.assertEqual(got, "Текст ```\nя понял\n``` и я поняла.")

    def test_flush_releases_the_last_word(self):
        streamer = GenderStreamer()
        self.assertEqual(streamer.feed("Я понял"), "Я ")
        self.assertEqual(streamer.feed(""), "")
        self.assertEqual(streamer.flush(), "поняла")
        self.assertEqual(streamer.flush(), "")

    def test_reset_drops_quote_state(self):
        streamer = GenderStreamer()
        streamer.feed("«незакрытая цитата")
        streamer.reset()
        self.assertEqual(streamer.feed("я понял.") + streamer.flush(), "я поняла.")


class PassThroughPlugins:
    """Плагинов нет: ядро обязано применить род само (задача 0207)."""

    class _Pipeline:
        def feed(self, text):
            return text

        def flush(self):
            return ""

        def __bool__(self):
            return False

    def stream_filter(self, *, scope="chat"):
        return self._Pipeline()

    def filter_text(self, text, *, scope="chat"):
        return text


class FeminineBackend(chat_cli.ChatBackend):
    """Фейковый бэкенд чата: отдаёт ответ заранее нарезанными кусками."""

    name = "fake"
    model = "mock"

    def __init__(self, pieces):
        self.pieces = list(pieces)
        self.counter = chat_cli.TokenCounter(exact=True, counter=len)
        self.calls = []
        self.closed = False

    def stream(self, messages, *, max_tokens):
        self.calls.append([dict(message) for message in messages])
        for text in self.pieces:
            yield chat_cli.StreamEvent(text=text)
        yield chat_cli.StreamEvent(
            done=True,
            finish_reason="stop",
            usage={"prompt_tokens": 1, "completion_tokens": 2},
        )

    def close(self):
        self.closed = True


class TestChatCliIntegration(unittest.TestCase):
    """``tools/chat.py``: и ``--once``, и REPL идут через ``ChatRepl.turn``."""

    def run_turn(self, pieces, question="что ты сделал?", plugins=None):
        backend = FeminineBackend(pieces)
        session = chat_cli.ChatSession(
            ctx=256, max_tokens=32, counter=backend.counter
        )
        out = io.StringIO()
        kwargs = {} if plugins is None else {"plugins": plugins}
        repl = chat_cli.ChatRepl(
            backend,
            session,
            stdin=io.StringIO(""),
            stdout=out,
            stderr=io.StringIO(),
            **kwargs,
        )
        stats = repl.turn(question, prefix=False)  # так же, как --once
        return stats, out.getvalue(), session

    def test_answer_is_feminized_in_stream(self):
        stats, printed, session = self.run_turn(
            ["Я всё сдел", "ал, я ", "готов", " помочь."]
        )
        self.assertIsNotNone(stats)
        self.assertIn("Я всё сделала, я готова помочь.", printed)
        self.assertNotRegex(printed, r"сделал(?!а)")

    def test_history_keeps_feminized_answer(self):
        _, _, session = self.run_turn(["Я всё проверил."])
        self.assertEqual(
            session.history[-1],
            {"role": "assistant", "content": "Я всё проверила."},
        )

    def test_core_feminizes_without_plugins(self):
        # Плагин ru-gender не найден/выключен — работает страховка ядра.
        _, printed, session = self.run_turn(
            ["Я пон", "ял."], plugins=PassThroughPlugins()
        )
        self.assertIn("Я поняла.", printed)
        self.assertEqual(
            session.history[-1],
            {"role": "assistant", "content": "Я поняла."},
        )

    def test_other_person_is_not_touched_in_cli(self):
        _, printed, _ = self.run_turn(["Он понял."])
        self.assertIn("Он понял.", printed)
        self.assertNotIn("поняла", printed)


class TestPromptRule(unittest.TestCase):
    """Первый слой — правило в промптах: оно есть во всех ``system*.md``."""

    PROMPTS = os.path.join(ROOT, "prompts")

    def prompt_files(self):
        return sorted(
            os.path.join(self.PROMPTS, name)
            for name in os.listdir(self.PROMPTS)
            if name.startswith("system") and name.endswith(".md")
        )

    def test_every_system_prompt_has_the_rule(self):
        files = self.prompt_files()
        self.assertGreaterEqual(len(files), 5)
        for path in files:
            with self.subTest(path=os.path.basename(path)):
                text = Path(path).read_text(encoding="utf-8")
                self.assertIn("женском роде", text.lower())
                self.assertIn("поняла", text.lower())
                self.assertIn("готова", text.lower())
                self.assertIn("сделала", text.lower())

    def test_main_prompt_names_eva_and_gender(self):
        text = Path(self.PROMPTS, "system.ru.md").read_text(encoding="utf-8")
        self.assertIn("Ты Ева, женщина", text)
        self.assertIn("краткие прилагательные", text)

    def test_compact_prompt_rule_fits_two_lines(self):
        text = Path(self.PROMPTS, "system.compact.md").read_text(encoding="utf-8")
        lines = [line for line in text.splitlines() if line.strip()]
        gender_lines = [line for line in lines if "женском роде" in line]
        self.assertEqual(len(gender_lines), 1, "правило о роде — одной строкой")
        self.assertLessEqual(len(gender_lines[0]), 160)
        # Компактный промпт остаётся компактным: правило не раздуло его втрое.
        self.assertLessEqual(len(lines), 4)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
