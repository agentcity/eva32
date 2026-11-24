"""Тесты ``tools/chat.py``: история, обрезка, команды, стриминг на моках.

Запуск::

    python3 -m unittest tests.test_chat_cli -v

Сети и GPU нет: HTTP-транспорты подменяются функциями, отдающими заранее
собранные SSE/NDJSON-строки, ВТМ — фейковым ``inner`` (без ``ctypes`` и
``libvtm``). Единственная интеграционная проверка настоящей ВТМ помечена
``skipUnless``: на машине без ``libvtm.a``/модели она скипается, а не краснеет.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import chat as chat_cli  # noqa: E402

TOKENIZER = ROOT / "tests" / "vtm" / "fixtures" / "tokenizer.json"
TINY_MODEL = ROOT / "runs" / "tiny" / "model.vtm"


def _counter() -> chat_cli.TokenCounter:
    """Детерминированный «токенизатор»: 1 символ = 1 токен."""
    return chat_cli.TokenCounter(exact=True, counter=len)


def _sse(events: list[dict], *, done: bool = True) -> bytes:
    text = "".join(
        f"data: {json.dumps(event, ensure_ascii=False)}\n\n" for event in events
    )
    if done:
        text += "data: [DONE]\n\n"
    return text.encode("utf-8")


def _ndjson(events: list[dict]) -> bytes:
    return (
        "\n".join(json.dumps(event, ensure_ascii=False) for event in events) + "\n"
    ).encode("utf-8")


def _stream_response(body: bytes, status: int = 200) -> chat_cli.StreamResponse:
    return chat_cli.StreamResponse(status=status, body=io.BytesIO(body), headers={})


# --------------------------------------------------------------------------- #
# фейковые бэкенды
# --------------------------------------------------------------------------- #

class FakeDecoder:
    """Декодер фейковой ВТМ: id -> готовый текст (12 — неполный UTF-8)."""

    TEXT = {10: "при", 11: "вет", 12: ""}

    def push(self, token: int) -> str:
        return self.TEXT.get(int(token), "")

    def finish(self) -> str:
        return "!"


class FakeTokenizer:
    def __init__(self, eos_id: int = -1) -> None:
        self.eos_id = eos_id

    def stream_decoder(self) -> FakeDecoder:
        return FakeDecoder()


class FakeVtmInner:
    """Минимальный ``VtmBackend``: encode/prompt_builder/stream_ids/close."""

    prompt_builder = staticmethod(
        lambda messages: "\n".join(
            f"{m['role']}: {m['content']}" for m in messages
        ) + "\nassistant:"
    )

    def __init__(self, ids: list[int] | None = None, eos_id: int = -1) -> None:
        self.ids = list(ids if ids is not None else [10, 11, 12])
        self.tokenizer = FakeTokenizer(eos_id)
        self.calls: list[list[int]] = []
        self.closed = False
        self.cancelled = False
        self.encode_error: Exception | None = None

    def encode(self, text: str) -> list[int]:
        if self.encode_error is not None:
            raise self.encode_error
        return [1, 2, 3, 4]

    def stream_ids(self, prompt_ids, *, max_new_tokens=None):
        self.calls.append(list(prompt_ids))
        for token in self.ids:
            yield token

    def cancel(self) -> None:
        self.cancelled = True

    def close(self) -> None:
        self.closed = True


class FakeBackend(chat_cli.ChatBackend):
    """Бэкенд REPL: детерминированный ответ «эхо: <вопрос>»."""

    name = "fake"
    model = "mock"

    def __init__(self, *, fail: bool = False, cost_known: bool = True) -> None:
        self.counter = _counter()
        self.fail = fail
        self.cost_known = cost_known
        self.calls: list[list[dict[str, str]]] = []
        self.closed = False

    def stream(self, messages, *, max_tokens):
        self.calls.append([dict(message) for message in messages])
        if self.fail:
            raise chat_cli.ChatBackendError("мок: сеть недоступна")
        question = messages[-1]["content"] if messages else ""
        answer = f"эхо: {question}"
        for piece in (answer[:2], answer[2:]):
            yield chat_cli.StreamEvent(text=piece)
        yield chat_cli.StreamEvent(
            done=True,
            finish_reason="stop",
            usage={
                "prompt_tokens": sum(len(m["content"]) for m in messages),
                "completion_tokens": 2,
                "cost_rub": 0.0,
                "cost_known": self.cost_known,
            },
        )

    def close(self) -> None:
        self.closed = True


# --------------------------------------------------------------------------- #
# токены, история, обрезка
# --------------------------------------------------------------------------- #

class TestEstimateTokens(unittest.TestCase):
    def test_empty(self) -> None:
        self.assertEqual(chat_cli.estimate_tokens(""), 0)

    def test_grows_with_text(self) -> None:
        self.assertGreater(chat_cli.estimate_tokens("привет мир"), 0)
        self.assertGreater(
            chat_cli.estimate_tokens("привет мир, как дела?"),
            chat_cli.estimate_tokens("привет"),
        )

    def test_describe(self) -> None:
        self.assertIn("точно", chat_cli.TokenCounter(exact=True).describe())
        self.assertIn("оценка", chat_cli.TokenCounter(exact=False).describe())


class TestTrimHistory(unittest.TestCase):
    def test_fits_without_trimming(self) -> None:
        history = [
            {"role": "user", "content": "aaaa"},
            {"role": "assistant", "content": "bb"},
        ]
        result = chat_cli.trim_history(
            history, system="sys", ctx=100, max_tokens=10, counter=_counter()
        )
        self.assertFalse(result.trimmed)
        self.assertEqual(result.dropped, 0)
        self.assertEqual(
            result.messages,
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "aaaa"},
                {"role": "assistant", "content": "bb"},
            ],
        )

    def test_old_messages_dropped_system_and_question_kept(self) -> None:
        history = [
            {"role": "user", "content": "a" * 40},
            {"role": "assistant", "content": "b" * 40},
            {"role": "user", "content": "cc"},
        ]
        result = chat_cli.trim_history(
            history, system="sys", ctx=40, max_tokens=8, counter=_counter()
        )
        self.assertTrue(result.trimmed)
        self.assertEqual(result.messages[0], {"role": "system", "content": "sys"})
        self.assertEqual(result.messages[-1], {"role": "user", "content": "cc"})
        self.assertEqual(result.dropped, 2)
        self.assertLess(result.used_tokens, 40)

    def test_leading_assistant_dropped(self) -> None:
        # Бюджет вмещает только ответ и последний вопрос: вопрос-источник
        # выброшен, значит ответ без вопроса тоже уходит.
        history = [
            {"role": "user", "content": "a" * 30},
            {"role": "assistant", "content": "bb"},
            {"role": "user", "content": "cc"},
        ]
        result = chat_cli.trim_history(
            history, system="", ctx=24, max_tokens=8, counter=_counter()
        )
        self.assertEqual([m["role"] for m in result.messages], ["user"])
        self.assertEqual(result.messages[0]["content"], "cc")
        self.assertEqual(result.dropped, 2)

    def test_last_message_survives_tiny_budget(self) -> None:
        result = chat_cli.trim_history(
            [{"role": "user", "content": "очень длинный вопрос"}],
            system="",
            ctx=10,
            max_tokens=8,
            counter=_counter(),
        )
        self.assertEqual(len(result.messages), 1)
        self.assertEqual(result.messages[0]["content"], "очень длинный вопрос")
        self.assertEqual(result.dropped, 0)

    def test_system_survives_tiny_budget(self) -> None:
        result = chat_cli.trim_history(
            [{"role": "user", "content": "вопрос"}],
            system="системный промпт",
            ctx=10,
            max_tokens=8,
            counter=_counter(),
        )
        self.assertEqual(result.messages[0]["role"], "system")
        self.assertEqual(result.messages[0]["content"], "системный промпт")

    def test_empty_history_keeps_system(self) -> None:
        result = chat_cli.trim_history(
            [], system="sys", ctx=100, max_tokens=10, counter=_counter()
        )
        self.assertEqual(result.messages, [{"role": "system", "content": "sys"}])
        self.assertEqual(result.dropped, 0)


class TestChatSession(unittest.TestCase):
    def make(self, **kwargs) -> chat_cli.ChatSession:
        params = dict(system="", ctx=256, max_tokens=32, counter=_counter())
        params.update(kwargs)
        return chat_cli.ChatSession(**params)

    def test_add_and_reset_keep_system(self) -> None:
        session = self.make(system="ты кот")
        session.add_user("привет")
        session.add_assistant("мур")
        self.assertEqual(len(session.history), 2)
        session.reset()
        self.assertEqual(session.history, [])
        self.assertEqual(session.turns, [])
        self.assertEqual(session.system, "ты кот")

    def test_drop_last_only_matching_role(self) -> None:
        session = self.make()
        session.add_user("вопрос")
        session.drop_last("assistant")
        self.assertEqual(len(session.history), 1)
        session.drop_last("user")
        self.assertEqual(session.history, [])

    def test_build_messages_keeps_order_and_system(self) -> None:
        session = self.make(system="sys")
        session.add_user("u1")
        session.add_assistant("a1")
        session.add_user("u2")
        messages, trim = session.build_messages()
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])
        self.assertIs(session.last_trim, trim)
        self.assertEqual(trim.dropped, 0)

    def test_finish_turn_estimates_missing_usage(self) -> None:
        session = self.make()
        messages = [{"role": "user", "content": "привет"}]
        stats = session.finish_turn(
            "ответ", messages, {}, ttft_s=0.25, elapsed_s=1.25, finish_reason="stop"
        )
        self.assertEqual(stats.prompt_tokens, len("привет") + chat_cli.MESSAGE_OVERHEAD)
        self.assertEqual(stats.completion_tokens, len("ответ"))
        self.assertEqual(stats.chars, len("ответ"))
        self.assertEqual(stats.ttft_ms, 250.0)
        self.assertEqual(stats.finish_reason, "stop")

    def test_finish_turn_uses_provider_usage_and_cost(self) -> None:
        session = self.make()
        stats = session.finish_turn(
            "ответ",
            [{"role": "user", "content": "x"}],
            {
                "prompt_tokens": 11,
                "completion_tokens": 3,
                "cost_rub": 0.0042,
                "cost_known": True,
            },
            ttft_s=0.1,
            elapsed_s=0.6,
        )
        self.assertEqual(stats.total_tokens, 14)
        self.assertEqual(stats.cost_rub, 0.0042)
        self.assertTrue(stats.cost_known)
        self.assertEqual(len(session.turns), 1)

    def test_totals_and_average_ttft(self) -> None:
        session = self.make()
        for ttft, completion in ((0.1, 4), (0.3, 6)):
            session.finish_turn(
                "x" * completion,
                [{"role": "user", "content": "q"}],
                {"prompt_tokens": 5, "completion_tokens": completion,
                 "cost_rub": 0.0, "cost_known": True},
                ttft_s=ttft,
                elapsed_s=1.0,
            )
        totals = session.totals()
        self.assertEqual(totals["turns"], 2)
        self.assertEqual(totals["prompt_tokens"], 10)
        self.assertEqual(totals["completion_tokens"], 10)
        self.assertAlmostEqual(totals["avg_ttft_ms"], 200.0)


class TestTurnStats(unittest.TestCase):
    def test_tps_counts_decode_time_after_ttft(self) -> None:
        stats = chat_cli.TurnStats(
            prompt_tokens=10, completion_tokens=8, chars=40,
            ttft_s=0.5, elapsed_s=1.5, cost_known=True,
        )
        self.assertAlmostEqual(stats.tps, 8.0)
        self.assertAlmostEqual(stats.chars_per_token, 5.0)
        self.assertEqual(stats.total_tokens, 18)

    def test_tps_single_token_and_empty(self) -> None:
        single = chat_cli.TurnStats(completion_tokens=1, elapsed_s=0.5, ttft_s=0.5)
        self.assertAlmostEqual(single.tps, 2.0)
        empty = chat_cli.TurnStats()
        self.assertEqual(empty.tps, 0.0)
        self.assertEqual(empty.chars_per_token, 0.0)


class TestStatsFormatting(unittest.TestCase):
    def make_session(self) -> chat_cli.ChatSession:
        session = chat_cli.ChatSession(system="sys", ctx=512, max_tokens=32, counter=_counter())
        session.add_user("привет")
        session.add_assistant("ответ")
        session.finish_turn(
            "ответ",
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "привет"}],
            {"prompt_tokens": 12, "completion_tokens": 5, "cost_rub": 0.001,
             "cost_known": True},
            ttft_s=0.12,
            elapsed_s=0.5,
            finish_reason="stop",
        )
        return session

    def test_format_stats_shows_required_metrics(self) -> None:
        text = chat_cli.format_stats(self.make_session(), backend_name="fake", model="mock")
        self.assertIn("TTFT", text)
        self.assertIn("120 мс", text)
        self.assertIn("ток/с", text)
        self.assertIn("символов", text)
        self.assertIn("0.0010 ₽", text)
        self.assertIn("контекст: 12/512", text)
        self.assertIn("fake/mock", text)

    def test_format_stats_without_turns(self) -> None:
        session = chat_cli.ChatSession(ctx=128, max_tokens=16, counter=_counter())
        text = chat_cli.format_stats(session, backend_name="vtm", model="m")
        self.assertIn("ходов: 0", text)

    def test_unknown_cost_is_not_zero_rubles(self) -> None:
        stats = chat_cli.TurnStats(completion_tokens=3, cost_known=False)
        text = chat_cli.format_turn_brief(stats)
        self.assertIn("тариф не задан", text)
        self.assertNotIn("0.0000 ₽", text)

    def test_turn_brief(self) -> None:
        stats = chat_cli.TurnStats(
            completion_tokens=20, chars=80, ttft_s=0.2, elapsed_s=1.2,
            cost_known=True,
        )
        text = chat_cli.format_turn_brief(stats)
        self.assertIn("20 ток", text)
        self.assertIn("TTFT 200 мс", text)
        self.assertIn("80 симв", text)


# --------------------------------------------------------------------------- #
# разбор SSE / NDJSON
# --------------------------------------------------------------------------- #

def _events_from(body: bytes) -> list[dict]:
    return list(chat_cli.parse_stream_payloads(io.BytesIO(body)))


class TestParseStreamPayloads(unittest.TestCase):
    def test_sse_events_and_done(self) -> None:
        events = _events_from(
            b'data: {"choices":[{"delta":{"content":"\xd0\xbf"}}]}\n\n'
            b": keep-alive\n"
            b"event: message\n"
            b'data: {"choices":[{"delta":{"content":"\xd1\x80"},"finish_reason":"stop"}]}\n\n'
            b"data: [DONE]\n\n"
            b'data: {"choices":[{"delta":{"content":"\xd0\xbf\xd0\xbe\xd1\x81\xd0\xbb\xd0\xb5 DONE"}}]}\n\n'
        )
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["choices"][0]["delta"]["content"], "п")
        self.assertEqual(events[1]["choices"][0]["finish_reason"], "stop")

    def test_ndjson_ollama(self) -> None:
        body = _ndjson(
            [
                {"message": {"role": "assistant", "content": "при"}, "done": False},
                {"message": {"role": "assistant", "content": "вет"}, "done": False},
                {"done": True, "eval_count": 2, "prompt_eval_count": 9},
            ]
        )
        events = _events_from(body)
        self.assertEqual(len(events), 3)
        self.assertTrue(events[-1]["done"])
        self.assertEqual(events[-1]["eval_count"], 2)

    def test_blank_lines_and_crlf_are_ignored(self) -> None:
        events = _events_from(b"\r\n{\"a\": 1}\r\n\r\n")
        self.assertEqual(events, [{"a": 1}])

    def test_invalid_json_is_backend_error(self) -> None:
        with self.assertRaises(chat_cli.ChatBackendError) as ctx:
            _events_from("data: {не json}\n\n".encode("utf-8"))
        self.assertIn("некорректный JSON", str(ctx.exception))

    def test_non_object_event_is_backend_error(self) -> None:
        with self.assertRaises(chat_cli.ChatBackendError) as ctx:
            _events_from(b"data: [1, 2, 3]\n\n")
        self.assertIn("не объект JSON", str(ctx.exception))

    def test_provider_error_object(self) -> None:
        with self.assertRaises(chat_cli.ChatBackendError) as ctx:
            _events_from(b'data: {"error": {"message": "model not found"}}\n\n')
        self.assertIn("model not found", str(ctx.exception))


# --------------------------------------------------------------------------- #
# ollama
# --------------------------------------------------------------------------- #

class FakeTransport:
    """Транспорт с заранее заданным ответом; запоминает запрос."""

    def __init__(self, response: chat_cli.StreamResponse) -> None:
        self.response = response
        self.calls: list[dict] = []

    def __call__(self, url, headers, payload, timeout):
        self.calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "payload": json.loads(payload.decode("utf-8")),
                "timeout": timeout,
            }
        )
        return self.response


class TestOllamaBackend(unittest.TestCase):
    def test_stream_text_and_usage(self) -> None:
        transport = FakeTransport(
            _stream_response(
                _ndjson(
                    [
                        {"message": {"role": "assistant", "content": "при"}, "done": False},
                        {"message": {"role": "assistant", "content": "вет"}, "done": False},
                        {"done": True, "done_reason": "stop",
                         "prompt_eval_count": 9, "eval_count": 2},
                    ]
                )
            )
        )
        backend = chat_cli.OllamaChatBackend(
            model="qwen2.5:3b", transport=transport
        )
        events = list(backend.stream([{"role": "user", "content": "привет"}], max_tokens=7))
        self.assertEqual("".join(e.text for e in events), "привет")
        done = events[-1]
        self.assertTrue(done.done)
        self.assertEqual(done.finish_reason, "stop")
        self.assertEqual(done.usage["prompt_tokens"], 9)
        self.assertEqual(done.usage["completion_tokens"], 2)
        self.assertTrue(done.usage["cost_known"])
        self.assertEqual(done.usage["cost_rub"], 0.0)

        call = transport.calls[0]
        self.assertEqual(call["url"], "http://127.0.0.1:11434/api/chat")
        self.assertEqual(call["payload"]["model"], "qwen2.5:3b")
        self.assertTrue(call["payload"]["stream"])
        self.assertEqual(call["payload"]["options"]["num_predict"], 7)
        self.assertEqual(call["payload"]["messages"], [{"role": "user", "content": "привет"}])

    def test_base_without_scheme_is_normalized(self) -> None:
        backend = chat_cli.OllamaChatBackend(
            base="127.0.0.1:11434", model="m", transport=FakeTransport(_stream_response(b""))
        )
        self.assertEqual(backend.endpoint, "http://127.0.0.1:11434/api/chat")

    def test_http_404_is_human_error_with_pull_hint(self) -> None:
        transport = FakeTransport(
            _stream_response(b'{"error":"model \'qwen2.5:3b\' not found"}', status=404)
        )
        backend = chat_cli.OllamaChatBackend(model="qwen2.5:3b", transport=transport)
        with self.assertRaises(chat_cli.ChatBackendError) as ctx:
            list(backend.stream([{"role": "user", "content": "x"}], max_tokens=4))
        message = str(ctx.exception)
        self.assertIn("HTTP 404", message)
        self.assertIn("ollama pull qwen2.5:3b", message)

    def test_stream_without_done_is_error(self) -> None:
        transport = FakeTransport(
            _stream_response(_ndjson([{"message": {"content": "при"}, "done": False}]))
        )
        backend = chat_cli.OllamaChatBackend(model="m", transport=transport)
        with self.assertRaises(chat_cli.ChatBackendError) as ctx:
            list(backend.stream([{"role": "user", "content": "x"}], max_tokens=4))
        self.assertIn("оборвался", str(ctx.exception))

    def test_network_error_has_hint(self) -> None:
        def broken(url, headers, payload, timeout):
            raise OSError("Connection refused")

        backend = chat_cli.OllamaChatBackend(model="m", transport=broken)
        with self.assertRaises(chat_cli.ChatBackendError) as ctx:
            list(backend.stream([{"role": "user", "content": "x"}], max_tokens=4))
        self.assertIn("make dev-serve", str(ctx.exception))


# --------------------------------------------------------------------------- #
# openai (SSE через eva.clients)
# --------------------------------------------------------------------------- #

class TestOpenAIBackend(unittest.TestCase):
    def test_sse_stream_text_usage_and_cost(self) -> None:
        body = _sse(
            [
                {"choices": [{"delta": {"content": "при"}}]},
                {
                    "choices": [{"delta": {"content": "вет"}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": 7,
                        "completion_tokens": 2,
                        "total_tokens": 9,
                        "cost_rub": 0.0012,
                    },
                },
            ]
        )
        transport = FakeTransport(_stream_response(body))
        backend = chat_cli.OpenAIChatBackend(
            base="https://api.example.com",
            model="gpt-test",
            key="secret",
            stream_transport=transport,
        )
        events = list(backend.stream([{"role": "user", "content": "привет"}], max_tokens=16))
        self.assertEqual("".join(e.text for e in events), "привет")
        done = events[-1]
        self.assertTrue(done.done)
        self.assertEqual(done.finish_reason, "stop")
        self.assertEqual(done.usage["prompt_tokens"], 7)
        self.assertEqual(done.usage["completion_tokens"], 2)
        self.assertEqual(done.usage["cost_rub"], 0.0012)
        self.assertTrue(done.usage["cost_known"])
        call = transport.calls[0]
        self.assertEqual(call["url"], "https://api.example.com/v1/chat/completions")
        self.assertTrue(call["payload"]["stream"])
        self.assertEqual(call["payload"]["model"], "gpt-test")

    def test_local_prices_make_cost_known(self) -> None:
        body = _sse(
            [
                {
                    "choices": [{"delta": {"content": "ок"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1000, "completion_tokens": 1000,
                              "total_tokens": 2000},
                }
            ]
        )
        transport = FakeTransport(_stream_response(body))
        backend = chat_cli.OpenAIChatBackend(
            base="https://api.example.com",
            model="m",
            price_prompt_rub=0.4,
            price_completion_rub=1.2,
            stream_transport=transport,
        )
        events = list(backend.stream([{"role": "user", "content": "x"}], max_tokens=4))
        done = events[-1]
        self.assertTrue(done.usage["cost_known"])
        self.assertAlmostEqual(done.usage["cost_rub"], 1.6, places=6)

    def test_stream_without_usage_leaves_estimation_to_session(self) -> None:
        body = _sse(
            [{"choices": [{"delta": {"content": "ок"}, "finish_reason": "stop"}]}]
        )
        backend = chat_cli.OpenAIChatBackend(
            base="https://api.example.com",
            model="m",
            stream_transport=FakeTransport(_stream_response(body)),
        )
        events = list(backend.stream([{"role": "user", "content": "привет"}], max_tokens=4))
        done = events[-1]
        self.assertEqual(dict(done.usage), {"cost_rub": 0.0, "cost_known": False})
        session = chat_cli.ChatSession(ctx=128, max_tokens=8, counter=_counter())
        stats = session.finish_turn(
            "".join(e.text for e in events),
            [{"role": "user", "content": "привет"}],
            done.usage,
            ttft_s=0.1,
            elapsed_s=0.2,
        )
        self.assertEqual(stats.completion_tokens, len("ок"))
        self.assertFalse(stats.cost_known)

    def test_missing_base_or_model_is_config_error(self) -> None:
        with self.assertRaises(chat_cli.ChatConfigError):
            chat_cli.OpenAIChatBackend(base="", model="m")
        with self.assertRaises(chat_cli.ChatConfigError):
            chat_cli.OpenAIChatBackend(base="http://127.0.0.1:8000", model="")

    def test_provider_error_is_chat_backend_error(self) -> None:
        transport = FakeTransport(
            _stream_response(b'{"error":{"message":"bad key"}}', status=401)
        )
        backend = chat_cli.OpenAIChatBackend(
            base="https://api.example.com", model="m", stream_transport=transport
        )
        with self.assertRaises(chat_cli.ChatBackendError) as ctx:
            list(backend.stream([{"role": "user", "content": "x"}], max_tokens=4))
        self.assertIn("401", str(ctx.exception))


# --------------------------------------------------------------------------- #
# vtm (фейковый inner, настоящий C-ABI не нужен)
# --------------------------------------------------------------------------- #

class TestVtmBackend(unittest.TestCase):
    def test_stream_tokens_and_exact_usage(self) -> None:
        inner = FakeVtmInner(ids=[10, 11, 12])
        backend = chat_cli.VtmChatBackend(model="fake.vtm", inner=inner)
        events = list(backend.stream([{"role": "user", "content": "привет"}], max_tokens=8))
        text = "".join(e.text for e in events)
        self.assertEqual(text, "привет!")
        done = events[-1]
        self.assertTrue(done.done)
        self.assertEqual(done.usage["prompt_tokens"], 4)
        self.assertEqual(done.usage["completion_tokens"], 3)
        self.assertTrue(done.usage["cost_known"])
        self.assertEqual(done.finish_reason, "stop")
        self.assertEqual(inner.calls, [[1, 2, 3, 4]])

    def test_eos_not_counted(self) -> None:
        inner = FakeVtmInner(ids=[10, 11, 20, 12], eos_id=20)
        backend = chat_cli.VtmChatBackend(model="fake.vtm", inner=inner)
        events = list(backend.stream([{"role": "user", "content": "q"}], max_tokens=8))
        done = events[-1]
        self.assertEqual(done.usage["completion_tokens"], 2)
        self.assertEqual(done.finish_reason, "stop")

    def test_length_when_limit_reached(self) -> None:
        inner = FakeVtmInner(ids=[10, 11, 10, 11])
        backend = chat_cli.VtmChatBackend(model="fake.vtm", inner=inner)
        events = list(backend.stream([{"role": "user", "content": "q"}], max_tokens=4))
        self.assertEqual(events[-1].finish_reason, "length")
        self.assertEqual(events[-1].usage["completion_tokens"], 4)

    def test_eos_at_token_limit_is_stop_not_length(self) -> None:
        # 1 токен и EOS на границе лимита: причина — stop, а не length.
        inner = FakeVtmInner(ids=[10, 20], eos_id=20)
        backend = chat_cli.VtmChatBackend(model="fake.vtm", inner=inner)
        events = list(backend.stream([{"role": "user", "content": "q"}], max_tokens=1))
        self.assertEqual(events[-1].usage["completion_tokens"], 1)
        self.assertEqual(events[-1].finish_reason, "stop")

    def test_vocab_error_gets_demo_hint(self) -> None:
        inner = FakeVtmInner()
        inner.encode_error = RuntimeError(
            "vtm_api_generate: token=276 вне vocab=128"
        )
        backend = chat_cli.VtmChatBackend(model="demo/model.vtm", inner=inner)
        with self.assertRaises(chat_cli.ChatBackendError) as ctx:
            list(backend.stream([{"role": "user", "content": "q"}], max_tokens=4))
        self.assertIn("demo/model.vtm", str(ctx.exception))
        self.assertIn("runs/tiny", str(ctx.exception))

    def test_close_delegates_to_inner(self) -> None:
        inner = FakeVtmInner()
        backend = chat_cli.VtmChatBackend(model="fake.vtm", inner=inner)
        backend.close()
        self.assertTrue(inner.closed)

    def test_cancel_delegates_to_inner(self) -> None:
        inner = FakeVtmInner()
        backend = chat_cli.VtmChatBackend(model="fake.vtm", inner=inner)
        backend.cancel()
        self.assertTrue(inner.cancelled)

    def test_counter_is_exact(self) -> None:
        backend = chat_cli.VtmChatBackend(model="fake.vtm", inner=FakeVtmInner())
        self.assertTrue(backend.counter.exact)
        self.assertEqual(backend.counter("привет"), 4)  # encode всегда [1,2,3,4]


# --------------------------------------------------------------------------- #
# сборка бэкенда из флагов/окружения
# --------------------------------------------------------------------------- #

class TestBuildBackend(unittest.TestCase):
    def test_unknown_backend(self) -> None:
        with self.assertRaises(chat_cli.ChatConfigError) as ctx:
            chat_cli.build_backend(backend="llama", environ={})
        self.assertIn("--backend", str(ctx.exception))

    def test_ollama_defaults_and_env(self) -> None:
        backend = chat_cli.build_backend(backend="ollama", environ={})
        self.assertEqual(backend.name, "ollama")
        self.assertEqual(backend.model, chat_cli.DEFAULT_OLLAMA_MODEL)
        self.assertEqual(backend.base, chat_cli.DEFAULT_OLLAMA_BASE)

        backend = chat_cli.build_backend(
            backend="ollama",
            environ={"OLLAMA_MODEL": "qwen2.5:7b", "OLLAMA_HOST": "localhost:1234"},
        )
        self.assertEqual(backend.model, "qwen2.5:7b")
        self.assertEqual(backend.base, "http://localhost:1234")

    def test_openai_requires_base_and_model(self) -> None:
        with self.assertRaises(chat_cli.ChatConfigError) as ctx:
            chat_cli.build_backend(backend="openai", environ={})
        self.assertIn("base", str(ctx.exception))

        with self.assertRaises(chat_cli.ChatConfigError) as ctx:
            chat_cli.build_backend(
                backend="openai", environ={"EVA_BIG_BASE": "https://api.example.com"}
            )
        self.assertIn("модель", str(ctx.exception))

    def test_openai_from_env(self) -> None:
        backend = chat_cli.build_backend(
            backend="openai",
            environ={
                "EVA_BIG_BASE": "https://api.example.com/v1",
                "EVA_BIG_MODEL": "big-model",
                "EVA_BIG_KEY": "k-123",
                "EVA_BIG_PRICE_PROMPT": "0.4",
                "EVA_BIG_PRICE_COMPLETION": "1.2",
            },
        )
        self.assertEqual(backend.name, "openai")
        self.assertEqual(backend.model, "big-model")
        self.assertEqual(backend.base, "https://api.example.com/v1")
        self.assertEqual(backend._prices, (0.4, 1.2))

    def test_openai_bad_price(self) -> None:
        with self.assertRaises(chat_cli.ChatConfigError):
            chat_cli.build_backend(
                backend="openai",
                environ={
                    "EVA_BIG_BASE": "https://api.example.com",
                    "EVA_BIG_MODEL": "m",
                    "EVA_BIG_PRICE_PROMPT": "дорого",
                },
            )

    def test_vtm_missing_model_file(self) -> None:
        with self.assertRaises(chat_cli.ChatConfigError) as ctx:
            chat_cli.build_backend(backend="vtm", model="/нет/такой.vtm", environ={})
        self.assertIn("файла модели нет", str(ctx.exception))

    def test_vtm_requires_tokenizer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.vtm"
            model.write_bytes(b"VTM1")
            with self.assertRaises(chat_cli.ChatConfigError) as ctx:
                chat_cli.build_backend(backend="vtm", model=str(model), environ={})
            self.assertIn("tokenizer", str(ctx.exception))

    def test_vtm_demo_gets_fixture_tokenizer(self) -> None:
        demo = ROOT / "demo" / "model.vtm"
        if not demo.is_file():
            self.skipTest("нет demo/model.vtm")
        backend = chat_cli.build_backend(
            backend="vtm", model=str(demo), environ={}
        )
        self.assertEqual(backend.name, "vtm")
        self.assertEqual(backend.model, str(demo))
        self.assertTrue(str(backend.tokenizer_path).endswith("tokenizer.json"))

    def test_vtm_cache_prefix_on_by_default_and_env_off(self) -> None:
        demo = ROOT / "demo" / "model.vtm"
        if not demo.is_file():
            self.skipTest("нет demo/model.vtm")
        default = chat_cli.build_backend(backend="vtm", model=str(demo), environ={})
        self.assertTrue(default.cache_prefix)
        disabled = chat_cli.build_backend(
            backend="vtm", model=str(demo), environ={"EVA_VTM_CACHE": "0"}
        )
        self.assertFalse(disabled.cache_prefix)


# --------------------------------------------------------------------------- #
# системный промпт
# --------------------------------------------------------------------------- #

class TestResolveSystem(unittest.TestCase):
    def test_literal_and_empty(self) -> None:
        self.assertEqual(chat_cli.resolve_system("ты кот"), "ты кот")
        self.assertEqual(chat_cli.resolve_system(""), "")
        self.assertEqual(chat_cli.resolve_system(None), "")
        self.assertEqual(chat_cli.resolve_system("-"), "")

    def test_file_and_at_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "system.md"
            path.write_text("  Ты — EVA.\n", encoding="utf-8")
            self.assertEqual(chat_cli.resolve_system(str(path)), "Ты — EVA.")
            self.assertEqual(chat_cli.resolve_system(f"@{path}"), "Ты — EVA.")

    def test_missing_at_file_is_error(self) -> None:
        with self.assertRaises(chat_cli.ChatConfigError):
            chat_cli.resolve_system("@/нет/такого/файла.md")

    def test_repo_prompt_file(self) -> None:
        prompt = ROOT / "prompts" / "system.assistant.md"
        if not prompt.is_file():
            self.skipTest("нет prompts/system.assistant.md")
        self.assertIn("EVA32", chat_cli.resolve_system(str(prompt)))


# --------------------------------------------------------------------------- #
# REPL и команды
# --------------------------------------------------------------------------- #

class TestChatRepl(unittest.TestCase):
    def run_repl(
        self,
        script: str,
        *,
        backend: FakeBackend | None = None,
        session: chat_cli.ChatSession | None = None,
        show_stats: bool = False,
    ) -> tuple[int, str, str, FakeBackend, chat_cli.ChatSession]:
        backend = backend or FakeBackend()
        session = session or chat_cli.ChatSession(
            ctx=256, max_tokens=32, counter=_counter()
        )
        out, err = io.StringIO(), io.StringIO()
        repl = chat_cli.ChatRepl(
            backend,
            session,
            stdin=io.StringIO(script),
            stdout=out,
            stderr=err,
            show_stats=show_stats,
        )
        code = repl.run()
        return code, out.getvalue(), err.getvalue(), backend, session

    def test_answer_history_and_stats(self) -> None:
        code, out, err, backend, session = self.run_repl(
            "привет\n/stats\n/quit\n"
        )
        self.assertEqual(code, 0)
        self.assertIn("assistant> ", out)
        self.assertIn("эхо: привет", out)
        self.assertIn("TTFT", out)
        self.assertIn("ток/с", out)
        self.assertIn("стоимость", out)
        self.assertEqual(
            session.history,
            [
                {"role": "user", "content": "привет"},
                {"role": "assistant", "content": "эхо: привет"},
            ],
        )
        self.assertEqual(backend.calls[0], [{"role": "user", "content": "привет"}])

    def test_system_command_affects_next_request_and_survives_reset(self) -> None:
        code, out, err, backend, session = self.run_repl(
            "/system Ты — кот\nсистема\n/reset\n/stats\n/quit\n"
        )
        self.assertEqual(code, 0)
        self.assertEqual(session.system, "Ты — кот")
        self.assertEqual(backend.calls[0][0], {"role": "system", "content": "Ты — кот"})
        self.assertEqual(backend.calls[0][1], {"role": "user", "content": "система"})
        self.assertIn("история очищена", out)
        self.assertIn("ходов: 0", out)
        self.assertIn("системный промпт: 8 символов", out)

    def test_system_command_show_and_clear(self) -> None:
        code, out, err, backend, session = self.run_repl(
            "/system\n/system Привет\n/system\n/system -\n/system\n/quit\n"
        )
        self.assertEqual(code, 0)
        self.assertIn("system: (нет)", out)
        self.assertIn("system: Привет", out)
        self.assertIn("системный промпт убран", out)
        self.assertEqual(session.system, "")

    def test_system_from_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sys.md"
            path.write_text("из файла", encoding="utf-8")
            code, out, err, backend, session = self.run_repl(
                f"/system @{path}\nпривет\n/quit\n"
            )
        self.assertEqual(code, 0)
        self.assertEqual(session.system, "из файла")
        self.assertEqual(backend.calls[0][0]["content"], "из файла")

    def test_unknown_command_does_not_reach_model(self) -> None:
        code, out, err, backend, session = self.run_repl("/nope\n/quit\n")
        self.assertEqual(code, 0)
        self.assertIn("неизвестная команда", err)
        self.assertEqual(backend.calls, [])

    def test_eof_exits_cleanly(self) -> None:
        code, out, err, backend, session = self.run_repl("")
        self.assertEqual(code, 0)
        self.assertIn("chat · backend=fake", out)

    def test_empty_lines_ignored(self) -> None:
        code, out, err, backend, session = self.run_repl("\n   \n/quit\n")
        self.assertEqual(code, 0)
        self.assertEqual(backend.calls, [])

    def test_backend_error_rolls_back_user_message(self) -> None:
        backend = FakeBackend(fail=True)
        code, out, err, backend, session = self.run_repl(
            "вопрос\n/stats\n/quit\n", backend=backend
        )
        self.assertEqual(code, 0)
        self.assertIn("ошибка", err)
        self.assertIn("мок: сеть недоступна", err)
        self.assertEqual(session.history, [])
        self.assertIn("ходов: 0", out)

    def test_keyboard_interrupt_cancels_and_rolls_back(self) -> None:
        class InterruptingBackend(FakeBackend):
            def __init__(self) -> None:
                super().__init__()
                self.cancelled = False

            def stream(self, messages, *, max_tokens):
                self.calls.append([dict(m) for m in messages])
                raise KeyboardInterrupt

            def cancel(self) -> None:
                self.cancelled = True

        backend = InterruptingBackend()
        code, out, err, backend, session = self.run_repl(
            "вопрос\n/quit\n", backend=backend
        )
        self.assertEqual(code, 0)
        self.assertTrue(backend.cancelled)
        self.assertIn("прервано", err)
        self.assertEqual(session.history, [])

    def test_show_stats_prints_brief(self) -> None:
        code, out, err, backend, session = self.run_repl(
            "привет\n/quit\n", show_stats=True
        )
        self.assertEqual(code, 0)
        self.assertIn("ток/с", err)
        self.assertIn("TTFT", err)

    def test_history_trimmed_in_repl(self) -> None:
        session = chat_cli.ChatSession(
            system="sys", ctx=40, max_tokens=8, counter=_counter()
        )
        backend = FakeBackend()
        script = "".join(f"вопрос номер {i}\n" for i in range(6)) + "/quit\n"
        code, out, err, backend, session = self.run_repl(
            script, backend=backend, session=session
        )
        self.assertEqual(code, 0)
        self.assertIn("история обрезана", err)
        last_call = backend.calls[-1]
        self.assertEqual(last_call[0], {"role": "system", "content": "sys"})
        self.assertEqual(last_call[-1], {"role": "user", "content": "вопрос номер 5"})
        self.assertLessEqual(len(last_call), 4)

    def test_slash_quit_variants(self) -> None:
        for command in ("/quit", "/exit", "/q"):
            code, out, err, backend, session = self.run_repl(f"{command}\n")
            self.assertEqual(code, 0, command)

    def test_ctrl_c_at_prompt_returns_130(self) -> None:
        class InterruptingInput(io.StringIO):
            def readline(self, *args, **kwargs):
                raise KeyboardInterrupt

        out, err = io.StringIO(), io.StringIO()
        repl = chat_cli.ChatRepl(
            FakeBackend(),
            chat_cli.ChatSession(ctx=128, max_tokens=8, counter=_counter()),
            stdin=InterruptingInput(""),
            stdout=out,
            stderr=err,
        )
        self.assertEqual(repl.run(), 130)
        self.assertIn("прервано", err.getvalue())

    def test_help_lists_commands(self) -> None:
        code, out, err, backend, session = self.run_repl("/help\n/quit\n")
        self.assertEqual(code, 0)
        for command in ("/reset", "/system", "/stats", "/quit"):
            self.assertIn(command, out)


# --------------------------------------------------------------------------- #
# CLI: парсер и main
# --------------------------------------------------------------------------- #

class TestParser(unittest.TestCase):
    def test_defaults(self) -> None:
        args = chat_cli.build_parser().parse_args([])
        self.assertEqual(args.backend, "vtm")
        self.assertEqual(args.max_tokens, chat_cli.DEFAULT_MAX_TOKENS)
        self.assertEqual(args.timeout, chat_cli.DEFAULT_TIMEOUT)
        self.assertIsNone(args.ctx)
        self.assertIsNone(args.once)
        self.assertFalse(args.stats)
        self.assertFalse(args.no_cache_prefix)

    def test_values(self) -> None:
        args = chat_cli.build_parser().parse_args(
            [
                "--backend", "ollama", "--model", "qwen2.5:3b",
                "--system", "ты кот", "--ctx", "2048", "--timeout", "5",
                "--max-tokens", "16", "--stats", "--once", "привет",
            ]
        )
        self.assertEqual(args.backend, "ollama")
        self.assertEqual(args.model, "qwen2.5:3b")
        self.assertEqual(args.ctx, 2048)
        self.assertEqual(args.timeout, 5.0)
        self.assertEqual(args.max_tokens, 16)
        self.assertTrue(args.stats)
        self.assertEqual(args.once, "привет")

    def test_invalid_values(self) -> None:
        for argv in (
            ["--backend", "llama"],
            ["--ctx", "0"],
            ["--ctx", "много"],
            ["--max-tokens", "0"],
            ["--timeout", "0"],
            ["--price-prompt", "-1"],
        ):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                with contextlib.redirect_stderr(io.StringIO()):
                    chat_cli.build_parser().parse_args(argv)

    def test_help(self) -> None:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), self.assertRaises(SystemExit) as ctx:
            chat_cli.build_parser().parse_args(["--help"])
        self.assertEqual(ctx.exception.code, 0)
        text = buffer.getvalue()
        for flag in ("--backend", "--model", "--system", "--ctx",
                     "--timeout", "--max-tokens"):
            self.assertIn(flag, text)


class TestMain(unittest.TestCase):
    def test_once_prints_answer_without_prompt(self) -> None:
        backend = FakeBackend()
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(chat_cli, "_backend_from_args", return_value=backend):
            code = chat_cli.main(
                ["--backend", "vtm", "--model", "m", "--tokenizer", "t",
                 "--max-tokens", "8", "--once", "привет", "--stats"],
                stdout=out,
                stderr=err,
                environ={},
            )
        self.assertEqual(code, 0)
        self.assertEqual(out.getvalue(), "эхо: привет\n")
        self.assertIn("ток/с", err.getvalue())
        self.assertEqual(len(backend.calls), 1)

    def test_once_failure_returns_1(self) -> None:
        backend = FakeBackend(fail=True)
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(chat_cli, "_backend_from_args", return_value=backend):
            code = chat_cli.main(
                ["--backend", "vtm", "--model", "m", "--tokenizer", "t",
                 "--once", "привет"],
                stdout=out,
                stderr=err,
                environ={},
            )
        self.assertEqual(code, 1)
        self.assertIn("ошибка", err.getvalue())

    def test_once_empty_returns_2_without_model_call(self) -> None:
        backend = FakeBackend()
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(chat_cli, "_backend_from_args", return_value=backend):
            code = chat_cli.main(
                ["--backend", "vtm", "--model", "m", "--tokenizer", "t", "--once", "  "],
                stdout=out,
                stderr=err,
                environ={},
            )
        self.assertEqual(code, 2)
        self.assertIn("нечего отправлять", err.getvalue())
        self.assertEqual(backend.calls, [])

    def test_config_error_returns_2(self) -> None:
        out, err = io.StringIO(), io.StringIO()
        code = chat_cli.main(
            ["--backend", "openai", "--once", "привет"],
            stdout=out,
            stderr=err,
            environ={},
        )
        self.assertEqual(code, 2)
        self.assertIn("base", err.getvalue())

    def test_repl_via_main(self) -> None:
        backend = FakeBackend()
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(chat_cli, "_backend_from_args", return_value=backend):
            code = chat_cli.main(
                ["--backend", "vtm", "--model", "m", "--tokenizer", "t"],
                stdin=io.StringIO("привет\n/quit\n"),
                stdout=out,
                stderr=err,
                environ={},
            )
        self.assertEqual(code, 0)
        self.assertIn("chat · backend=fake", out.getvalue())
        self.assertIn("эхо: привет", out.getvalue())
        self.assertTrue(backend.closed)

    def test_ctx_default_per_backend(self) -> None:
        for backend_name, expected in (
            ("vtm", 512),
            ("ollama", 4096),
            ("openai", 8192),
        ):
            args = chat_cli.build_parser().parse_args(["--backend", backend_name])
            self.assertEqual(chat_cli._resolve_ctx(args), expected)


# --------------------------------------------------------------------------- #
# интеграция: настоящая ВТМ (скип, если libvtm/модели нет)
# --------------------------------------------------------------------------- #

class TestRealVtmStream(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not TINY_MODEL.is_file() or not TOKENIZER.is_file():
            raise unittest.SkipTest(
                f"нет {TINY_MODEL} или {TOKENIZER}: реальный стриминг не проверить"
            )
        from eva.backends import VtmUnavailable, ensure_library

        try:
            ensure_library()
        except VtmUnavailable as exc:
            raise unittest.SkipTest(f"libvtm недоступна: {exc}") from None

    def test_real_stream_gives_tokens_and_usage(self) -> None:
        backend = chat_cli.VtmChatBackend(
            model=str(TINY_MODEL),
            tokenizer=str(TOKENIZER),
            max_tokens=4,
            cache_prefix=True,
        )
        try:
            events = list(
                backend.stream(
                    [{"role": "user", "content": "Сколько стоит тариф Старт?"}],
                    max_tokens=4,
                )
            )
        finally:
            backend.close()
        self.assertTrue(events)
        done = events[-1]
        self.assertTrue(done.done)
        self.assertGreater(done.usage["prompt_tokens"], 0)
        self.assertGreaterEqual(done.usage["completion_tokens"], 0)
        self.assertLessEqual(done.usage["completion_tokens"], 4)
        self.assertEqual(done.finish_reason in ("stop", "length"), True)
        self.assertIsInstance("".join(e.text for e in events), str)


if __name__ == "__main__":
    unittest.main()
