"""Задача 0212: справка в промпте, чистка служебного, страж противоречия.

Живой симптом: модель получала факт, но отвечала «не могу предоставить курс»
и печатала служебную строку ``[Факт: 1 USD = 82,3130 ₽ (источник: …)]``.
Здесь проверяется вся цепочка:

1. блок в промпте — секция ``СПРАВКА`` с явной командой (а не ``[Факт]``);
2. фильтр-плагин ``plugins/ru-facts`` вырезает служебные пометки, в том числе
   разрезанные между кусками потока и обрезанный ``[Факт``;
3. страж: отказ + факт → короткая фраза из справки, отказ без факта остаётся;
4. это работает в веб-чате (``POST /api/chat``) и в ``tools/chat.py``.

Сеть не нужна: факты подменяются плагином-заглушкой, модели — моками.
"""

from __future__ import annotations

import http.client
import importlib.util
import io
import json
import os
import pathlib
import sys
import tempfile
import threading
import unittest
from datetime import datetime

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eva.plugins import PluginRegistry  # noqa: E402
from eva.skills.clock import MOSCOW_TZ  # noqa: E402
from eva.skills.intent import (  # noqa: E402
    FACT_SECTION_HINT,
    FACT_SECTION_TITLE,
    FactGuardStream,
    fact_block,
    fact_entries,
    fact_line,
    fact_phrase,
    guard_answer,
    is_refusal,
    with_fact,
)
from tools import chat as chat_cli  # noqa: E402
from tools import chat_server  # noqa: E402

NOW = datetime(2026, 11, 26, 15, 30, tzinfo=MOSCOW_TZ)

#: Справка, которую «модель» должна была использовать.
USD_ENTRY = "1 USD = 82,3130 ₽ (ЦБ РФ на 26.11.2026)"
USD_BLOCK = fact_block([USD_ENTRY])

#: Живой ответ, который видел пользователь (симптом задачи).
LEAKY_REFUSAL = (
    "К сожалению, я не могу предоставить актуальный курс доллара без "
    "возможности получения этой информации в реальном времени. "
    "[Факт: 1 USD = 82,3130 ₽ (источник: ЦБ РФ на 26.11.2026)]"
)

REFUSAL = "К сожалению, я не могу предоставить актуальный курс доллара."

#: Тот же ответ без служебной пометки — столько должен оставить фильтр.
LEAKY_STRIPPED = (
    "К сожалению, я не могу предоставить актуальный курс доллара без "
    "возможности получения этой информации в реальном времени."
)


# --------------------------------------------------------------------------- #
# Плагин ru-facts: загрузка как в рантайме
# --------------------------------------------------------------------------- #


def load_facts_plugin():
    """Загрузить ``plugins/ru-facts/plugin.py`` так же, как это делает реестр."""
    path = ROOT / "plugins" / "ru-facts" / "plugin.py"
    spec = importlib.util.spec_from_file_location("_test_ru_facts", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestSectionInPrompt(unittest.TestCase):
    """Требование 1: в промпт уходит «СПРАВКА», а не «[Факт: …]»."""

    def test_fact_line_is_section_with_instruction(self):
        line = fact_line("Который час?", now=NOW)
        self.assertIsNotNone(line)
        assert line is not None
        self.assertEqual(line.splitlines()[0], FACT_SECTION_HINT)
        self.assertIn("используй эти данные дословно", line)
        self.assertIn("не говори, что данных нет", line)
        self.assertIn("не цитируй слово «СПРАВКА»", line)
        self.assertNotIn("[Факт", line)

    def test_with_fact_puts_section_next_to_data(self):
        messages = [
            {"role": "system", "content": "Ты — Ева."},
            {"role": "user", "content": "Курс доллара?"},
        ]
        result = with_fact(messages, USD_BLOCK)
        self.assertEqual(result[0]["content"], "Ты — Ева.\n\n" + USD_BLOCK)
        self.assertEqual(result[1], messages[1])

    def test_prompts_mention_section_and_forbid_refusal(self):
        for name in ("system.compact.md", "system.assistant.md"):
            with self.subTest(prompt=name):
                text = (ROOT / "prompts" / name).read_text(encoding="utf-8")
                low = text.lower()
                self.assertIn("СПРАВКА", text)
                self.assertNotIn("[Факт", text)
                # «не цитируй слово СПРАВКА» одной из двух формулировок.
                self.assertTrue(
                    "цитируй" in low or "не переноси" in low, text
                )


# --------------------------------------------------------------------------- #
# Пост-фильтр: служебное не должно доходить до пользователя
# --------------------------------------------------------------------------- #


class TestStripService(unittest.TestCase):
    """Требование 2: ``[Факт: …]``, «СПРАВКА:», «(источник: …)» — вырезать."""

    @classmethod
    def setUpClass(cls):
        cls.plugin = load_facts_plugin()

    def run_filter(self, text: str) -> str:
        return self.plugin.FactsFilter().run(text)

    def test_bracket_fact_with_source(self):
        self.assertEqual(
            self.run_filter(
                "Курс 82,3130 ₽ [Факт: 1 USD = 82,3130 ₽ "
                "(источник: ЦБ РФ на 26.11.2026)]"
            ),
            "Курс 82,3130 ₽",
        )

    def test_dangling_truncated_fact(self):
        # Модель оборвала строку — «[Факт: 1 USD = 82» тоже служебное.
        self.assertEqual(
            self.run_filter("Курс 82,3130 ₽ [Факт: 1 USD = 82"),
            "Курс 82,3130 ₽",
        )
        self.assertEqual(self.run_filter("Курс 82,3130 ₽ [Факт"), "Курс 82,3130 ₽")

    def test_source_without_brackets(self):
        self.assertEqual(
            self.run_filter("Курс 82,3130 ₽ (источник: ЦБ РФ на 26.11.2026)"),
            "Курс 82,3130 ₽",
        )

    def test_section_header_and_token(self):
        self.assertEqual(
            self.run_filter(USD_BLOCK), USD_ENTRY
        )
        self.assertEqual(
            self.run_filter("СПРАВКА: 1 USD = 82,3130 ₽"), "1 USD = 82,3130 ₽"
        )
        self.assertEqual(
            self.run_filter("Справка: курс 82,3130 ₽"), "курс 82,3130 ₽"
        )

    def test_plain_answer_is_untouched(self):
        text = "Курс доллара 82,3130 ₽. Это данные ЦБ РФ на сегодня."
        self.assertEqual(self.run_filter(text), text)
        self.assertEqual(self.run_filter(""), "")

    def test_plain_word_spravka_is_kept(self):
        # Без двоеточия «справка» — обычное слово, а не служебная метка:
        # фильтр не имеет права съесть его из нормального ответа.
        for text in (
            "Справка о доходах — это документ.",
            "Справка (о доходах) — это документ.",
            "Нужна справка о доходах.",
        ):
            with self.subTest(text=text):
                self.assertEqual(self.run_filter(text), text)

    def test_generic_section_header_is_stripped(self):
        # «СПРАВКА (…):» с произвольной подписью, но только с двоеточием.
        self.assertEqual(
            self.run_filter("СПРАВКА (данные ЦБ):\n1 USD = 82,3130 ₽"),
            "1 USD = 82,3130 ₽",
        )

    def test_stream_keeps_plain_word_spravka(self):
        text = "Справка о доходах — это документ."
        stream = self.plugin.FactsFilter().new_stream()
        out = "".join(
            stream.feed(text[index : index + 2])
            for index in range(0, len(text), 2)
        )
        out += stream.flush()
        self.assertEqual(out, text)

    def test_stream_strips_token_split_before_colon(self):
        # «СПРАВКА» и «:» пришли разными кусками — двоеточие не должно
        # остаться сиротой, а слово — утечь.
        stream = self.plugin.FactsFilter().new_stream()
        out = "".join(stream.feed(chunk) for chunk in ("СПРАВКА", ": ", "82,3130 ₽"))
        out += stream.flush()
        self.assertEqual(out, "82,3130 ₽")

    def test_stream_by_characters(self):
        # Маркер режется между кусками: фильтр обязан не выпустить его
        # половинки в поток.
        stream = self.plugin.FactsFilter().new_stream()
        out = "".join(stream.feed(char) for char in USD_BLOCK) + stream.flush()
        self.assertEqual(out, USD_ENTRY)

    def test_stream_never_leaks_marker_for_any_split(self):
        for size in range(1, 12):
            with self.subTest(chunk=size):
                stream = self.plugin.FactsFilter().new_stream()
                chunks = [
                    USD_BLOCK[index : index + size]
                    for index in range(0, len(USD_BLOCK), size)
                ]
                out = "".join(stream.feed(chunk) for chunk in chunks)
                out += stream.flush()
                self.assertEqual(out, USD_ENTRY)

    def test_stream_keeps_normal_answer(self):
        text = "Сейчас 15:30 MSK. Курс доллара 82,3130 ₽."
        stream = self.plugin.FactsFilter().new_stream()
        out = "".join(stream.feed(text[index : index + 2])
                      for index in range(0, len(text), 2))
        out += stream.flush()
        self.assertEqual(out, text)

    def test_registry_runs_plugin_in_chat_and_tts(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = pathlib.Path(tmp) / "plugins.json"
            config.write_text(
                json.dumps({"enabled": ["ru-facts"]}), encoding="utf-8"
            )
            registry = PluginRegistry([ROOT / "plugins"], config)
            self.assertEqual(registry.errors, [])
            for scope in ("chat", "tts"):
                with self.subTest(scope=scope):
                    self.assertEqual(
                        registry.filter_text(LEAKY_REFUSAL, scope=scope),
                        LEAKY_STRIPPED,
                    )


# --------------------------------------------------------------------------- #
# Страж противоречия
# --------------------------------------------------------------------------- #


class TestGuardAnswer(unittest.TestCase):
    """Требование 3: отказ при наличии справки заменяется фактом."""

    def test_refusal_with_fact_becomes_fact(self):
        answer = guard_answer(REFUSAL, fact_entries(USD_BLOCK))
        self.assertEqual(answer, USD_ENTRY + ".")
        self.assertNotIn("не могу", answer)

    def test_refusal_without_fact_stays(self):
        self.assertEqual(guard_answer(REFUSAL, []), REFUSAL)

    def test_other_refusals_are_caught(self):
        entries = fact_entries(USD_BLOCK)
        for phrase in (
            "Нет доступа к актуальным данным.",
            "Я не имею информации о курсе.",
            "Не располагаю данными о курсе.",
            "У меня нет доступа к интернету.",
        ):
            with self.subTest(phrase=phrase):
                self.assertTrue(is_refusal(phrase))
                self.assertEqual(guard_answer(phrase, entries), USD_ENTRY + ".")

    def test_normal_answer_is_not_replaced(self):
        text = "Курс доллара — 82,3130 ₽. Что-то ещё?"
        self.assertFalse(is_refusal(text))
        self.assertEqual(guard_answer(text, fact_entries(USD_BLOCK)), text)

    def test_degraded_fact_is_not_a_substitute(self):
        # «Не удалось получить» — не данные: подменять отказ нечем.
        degraded = fact_block(["не удалось получить курс USD — нет ответа от ЦБ РФ"])
        self.assertIsNone(fact_phrase(fact_entries(degraded)))
        self.assertEqual(
            guard_answer(REFUSAL, fact_entries(degraded)), REFUSAL
        )

    def test_phrase_from_several_entries(self):
        block = fact_block([USD_ENTRY, "1 EUR = 90,1000 ₽ (ЦБ РФ)"])
        self.assertEqual(
            guard_answer(REFUSAL, fact_entries(block)),
            USD_ENTRY + "; 1 EUR = 90,1000 ₽ (ЦБ РФ).",
        )


class TestGuardStream(unittest.TestCase):
    """Требование 4: страж работает в потоке, не ломая обычные ответы."""

    def stream(self, chunks, entries):
        guard = FactGuardStream(entries)
        out = "".join(guard.feed(chunk) for chunk in chunks)
        return out + guard.flush()

    def test_refusal_by_characters(self):
        out = self.stream(list(REFUSAL), fact_entries(USD_BLOCK))
        self.assertEqual(out, USD_ENTRY + ".")
        self.assertNotIn("не могу", out)

    def test_refusal_head_is_dropped_before_deltas(self):
        # Отказ не должен «мелькнуть» в потоке даже частично.
        guard = FactGuardStream(fact_entries(USD_BLOCK))
        sent = []
        for chunk in ("К сожалению, ", "я не могу ", "предоставить курс."):
            sent.append(guard.feed(chunk))
        sent.append(guard.flush())
        streamed = "".join(sent)
        self.assertNotIn("К сожалению", streamed)
        self.assertNotIn("не могу", streamed)
        self.assertIn("82,3130", streamed)

    def test_normal_answer_passes_through(self):
        text = "Сейчас 15:30 MSK. Курс доллара 82,3130 ₽."
        out = self.stream([text[index : index + 3]
                          for index in range(0, len(text), 3)],
                          fact_entries(USD_BLOCK))
        self.assertEqual(out, text)

    def test_short_answer_is_flushed(self):
        # Ответ короче предложения: буфер отдаётся на flush, не теряется.
        out = self.stream(["Привет", ", как дела"], fact_entries(USD_BLOCK))
        self.assertEqual(out, "Привет, как дела")

    def test_refusal_without_fact_passes_through(self):
        out = self.stream(list(REFUSAL), [])
        self.assertEqual(out, REFUSAL)

    def test_refusal_after_released_head_passes_through(self):
        # Страж держит только начало ответа: после первого предложения он
        # ничего не переписывает (иначе страдал бы обычный длинный ответ).
        guard = FactGuardStream(fact_entries(USD_BLOCK))
        self.assertEqual(guard.feed("Курс 82,3130 ₽. "), "Курс 82,3130 ₽. ")
        tail = "Не могу проверить другое."
        self.assertEqual(guard.feed(tail), tail)
        self.assertEqual(guard.flush(), "")


# --------------------------------------------------------------------------- #
# Живой сервер: то же самое через POST /api/chat
# --------------------------------------------------------------------------- #


REGISTRY = (
    chat_server.BackendSpec(
        id="vtm",
        label="ВТМ (локальная)",
        kind="vtm",
        models=(chat_server.ModelSpec(id="mock-vtm", label="Мок ВТМ"),),
    ),
    chat_server.BackendSpec(
        id="ollama",
        label="Ollama (Qwen)",
        kind="ollama",
        models=(chat_server.ModelSpec(id="mock-qwen", label="Мок Qwen"),),
    ),
)

FACT_PLUGIN = """
from eva.plugins import Fact


def fetch(ctx, question):
    return Fact("1 USD = 82,3130 ₽", source="ЦБ РФ на 26.11.2026",
                kind="currency")
"""


class MockAdapter:
    """Мок модели: отдаёт заданные куски ответа."""

    def __init__(self, pieces) -> None:
        self.pieces = list(pieces)
        self.calls: list[dict] = []

    def stream(self, messages, *, max_tokens):
        self.calls.append(
            {
                "messages": [dict(message) for message in messages],
                "max_tokens": max_tokens,
            }
        )
        for text in self.pieces:
            yield chat_server.StreamPiece(text=text)
        yield chat_server.StreamPiece(
            done=True,
            usage={
                "prompt_tokens": 8,
                "completion_tokens": len(self.pieces),
                "total_tokens": 8 + len(self.pieces),
            },
            finish_reason="stop",
        )


def parse_sse(raw: bytes):
    events = []
    done = False
    for block in raw.decode("utf-8").split("\n\n"):
        for line in block.splitlines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload:
                continue
            if payload == "[DONE]":
                done = True
            else:
                events.append(json.loads(payload))
    return events, done


def payload(question="курс доллара"):
    return {
        "question": question,
        "use_system": True,
        "sides": [
            {"backend": "vtm", "model": "mock-vtm"},
            {"backend": "ollama", "model": "mock-qwen"},
        ],
    }


class TestServerWiring(unittest.TestCase):
    """Требование 5: веб-чат не показывает отказ и служебное при справке."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        plugins_dir = pathlib.Path(self.tmp.name) / "plugins"
        plugin = plugins_dir / "fact"
        plugin.mkdir(parents=True)
        (plugin / "plugin.json").write_text(
            json.dumps(
                {
                    "id": "fact",
                    "name": "Факт",
                    "version": "1.0.0",
                    "api": 1,
                    "entry": "plugin.py",
                    "capabilities": ["skill.fetch"],
                    "permissions": {"network": [], "filesystem": []},
                    "settings": {},
                    "description": "тестовый факт",
                    "owner": "test",
                    "priority": 10,
                    "scope": "chat",
                    "stateful": False,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (plugin / "plugin.py").write_text(FACT_PLUGIN, encoding="utf-8")
        config = pathlib.Path(self.tmp.name) / "plugins.json"
        config.write_text(
            json.dumps({"enabled": ["fact", "ru-facts"]}), encoding="utf-8"
        )
        self.plugins = PluginRegistry([plugins_dir, ROOT / "plugins"], config)
        self.assertEqual(self.plugins.errors, [])

    def start(self, pieces, *, facts=True):
        left, right = MockAdapter(pieces), MockAdapter(pieces)
        server = chat_server.create_server(
            port=0,
            registry=REGISTRY,
            adapters={("vtm", "mock-vtm"): left, ("ollama", "mock-qwen"): right},
            system_prompt="СИСТЕМА",
            facts=facts,
            plugins=self.plugins,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def stop():
            server.shutdown()
            server.server_close()

        self.addCleanup(stop)
        return server, left, right

    def chat(self, server):
        connection = http.client.HTTPConnection(
            "127.0.0.1", server.server_address[1], timeout=10
        )
        try:
            connection.request(
                "POST",
                "/api/chat",
                body=json.dumps(payload()),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            raw = response.read()
            self.assertEqual(response.status, 200)
        finally:
            connection.close()
        events, done = parse_sse(raw)
        self.assertTrue(done)
        return events

    def deltas(self, events, side=0):
        return "".join(
            event["text"]
            for event in events
            if event["type"] == "delta" and event["side"] == side
        )

    def test_refusal_is_replaced_by_fact_in_stream(self):
        server, left, right = self.start(
            ("К сожалению, ", "я не могу ", "предоставить курс.", " [Факт: 1 USD]")
        )
        events = self.chat(server)
        for adapter in (left, right):
            with self.subTest(backend=adapter is left):
                system = adapter.calls[0]["messages"][0]["content"]
                self.assertIn(FACT_SECTION_TITLE, system)
                self.assertIn(USD_ENTRY, system)
                self.assertNotIn("[Факт", system)
        answer = self.deltas(events)
        self.assertIn("82,3130", answer)
        self.assertNotIn("не могу", answer)
        self.assertNotIn("[Факт", answer)

    def test_no_fact_keeps_refusal(self):
        server, _left, _right = self.start(
            ("К сожалению, ", "я не могу предоставить курс."), facts=False
        )
        answer = self.deltas(self.chat(server))
        self.assertIn("не могу", answer)

    def test_service_marker_is_stripped_without_refusal(self):
        server, _left, _right = self.start(
            ("Курс 82,3130 ₽. ", "[Факт: 1 USD = 82,3130 ₽ ",
             "(источник: ЦБ РФ на 26.11.2026)]")
        )
        answer = self.deltas(self.chat(server))
        self.assertIn("82,3130", answer)
        self.assertNotIn("[Факт", answer)
        self.assertNotIn("источник", answer)


# --------------------------------------------------------------------------- #
# tools/chat.py: REPL и --once идут тем же путём
# --------------------------------------------------------------------------- #


class RefusalBackend(chat_cli.ChatBackend):
    """Бэкенд REPL: потоковый отказ со служебной строкой в конце."""

    name = "mock"
    model = "mock"

    def __init__(self, pieces) -> None:
        self.pieces = list(pieces)
        self.calls: list[list[dict[str, str]]] = []

    def stream(self, messages, *, max_tokens):
        self.calls.append([dict(message) for message in messages])
        for text in self.pieces:
            yield chat_cli.StreamEvent(text=text)
        yield chat_cli.StreamEvent(
            done=True,
            finish_reason="stop",
            usage={
                "prompt_tokens": 8,
                "completion_tokens": len(self.pieces),
                "cost_rub": 0.0,
                "cost_known": True,
            },
        )


class TestReplWiring(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        plugins_dir = pathlib.Path(self.tmp.name) / "plugins"
        plugin = plugins_dir / "fact"
        plugin.mkdir(parents=True)
        (plugin / "plugin.json").write_text(
            json.dumps(
                {
                    "id": "fact",
                    "name": "Факт",
                    "version": "1.0.0",
                    "api": 1,
                    "entry": "plugin.py",
                    "capabilities": ["skill.fetch"],
                    "permissions": {"network": [], "filesystem": []},
                    "settings": {},
                    "description": "тестовый факт",
                    "owner": "test",
                    "priority": 10,
                    "scope": "chat",
                    "stateful": False,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (plugin / "plugin.py").write_text(FACT_PLUGIN, encoding="utf-8")
        config = pathlib.Path(self.tmp.name) / "plugins.json"
        config.write_text(
            json.dumps({"enabled": ["fact", "ru-facts"]}), encoding="utf-8"
        )
        self.plugins = PluginRegistry([plugins_dir, ROOT / "plugins"], config)

    def repl(self, backend, *, facts=True):
        session = chat_cli.ChatSession(
            ctx=256,
            max_tokens=32,
            counter=chat_cli.TokenCounter(exact=True, counter=len),
        )
        out, err = io.StringIO(), io.StringIO()
        repl = chat_cli.ChatRepl(
            backend,
            session,
            stdin=io.StringIO(""),
            stdout=out,
            stderr=err,
            facts=facts,
            plugins=self.plugins,
        )
        return repl, out, err

    def test_repl_replaces_refusal_with_fact(self):
        backend = RefusalBackend(
            ("К сожалению, ", "я не могу предоставить курс.",
             " [Факт: 1 USD = 82,3130 ₽ (источник: ЦБ РФ)]")
        )
        repl, out, _err = self.repl(backend)
        repl.turn("курс доллара")
        answer = out.getvalue()
        self.assertIn("82,3130", answer)
        self.assertNotIn("не могу", answer)
        self.assertNotIn("[Факт", answer)
        self.assertIn(FACT_SECTION_TITLE, backend.calls[0][0]["content"])

    def test_repl_without_facts_keeps_refusal(self):
        backend = RefusalBackend(
            ("К сожалению, ", "я не могу предоставить курс.")
        )
        repl, out, _err = self.repl(backend, facts=False)
        repl.turn("курс доллара")
        self.assertIn("не могу", out.getvalue())

    def test_repl_normal_answer_is_untouched(self):
        backend = RefusalBackend(("Сейчас ", "15:30 MSK."))
        repl, out, _err = self.repl(backend)
        repl.turn("который час")
        self.assertIn("Сейчас 15:30 MSK.", out.getvalue())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
