"""Проверки клиента к OpenAI-совместимым моделям.

Запуск: python3 -m unittest tests.test_clients -v

Сети нет: транспорт и пауза — подменяемые сечения
(`eva.clients.openai_compat._post_json` и `._sleep`), в тестах используются
фейки. Реальный `urlopen` под патчем там, где проверяется, что он не зовётся.
"""

import json
import logging
import os
import socket
import sys
import unittest
import urllib.error
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eva.clients import (  # noqa: E402
    ClientError,
    ClientResult,
    HTTPResponse,
    OpenAIClient,
    openai_compat,
)

# Клиент честно логирует ретраи; в прогоне тестов это шум. Отдельный тест
# проверяет логи через assertLogs, поэтому здесь только глушим вывод.
logging.getLogger("eva.clients").addHandler(logging.NullHandler())
logging.getLogger("eva.clients").propagate = False


# --------------------------------------------------------------------------- #
# Фейки: сеть и пауза
# --------------------------------------------------------------------------- #

def ok_response(text="привет", *, prompt=10, completion=5, cost=None, extra=None):
    """Успешный ответ в формате OpenAI."""
    body = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "test-model",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text},
             "finish_reason": "stop"},
        ],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                  "total_tokens": prompt + completion},
    }
    if cost is not None:
        body["usage"]["cost_rub"] = cost
    if extra:
        body.update(extra)
    return HTTPResponse(status=200, body=json.dumps(body, ensure_ascii=False).encode("utf-8"))


def error_response(status, detail=None, *, headers=None, raw=None):
    """Ответ с HTTP-ошибкой; тело как у провайдера."""
    if raw is not None:
        body = raw
    else:
        payload = {"error": {"message": detail or "boom", "type": "test_error"}}
        body = json.dumps(payload).encode("utf-8")
    return HTTPResponse(status=status, body=body, headers=headers or {})


class FakeTransport:
    """Подмена сети: очередь ответов/исключений + запись запросов."""

    def __init__(self, *responses):
        if not responses:
            raise AssertionError("нужен хотя бы один ответ")
        self.responses = list(responses)
        self.calls = []

    def __call__(self, url, headers, body, timeout):
        self.calls.append(
            SimpleNamespace(
                url=url,
                headers=dict(headers),
                raw_body=body,
                payload=json.loads(body.decode("utf-8")),
                timeout=timeout,
            )
        )
        # Последний ответ повторяется, если очередь кончилась.
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, BaseException):
            raise item
        return item

    @property
    def attempts(self):
        return len(self.calls)


class SleepRecorder:
    """Подмена `time.sleep`: запоминает паузы, не спит."""

    def __init__(self):
        self.delays = []

    def __call__(self, seconds):
        self.delays.append(seconds)


def make_client(transport=None, sleep=None, **kwargs):
    params = {
        "base": "http://127.0.0.1:11434",
        "key": "sk-test",
        "model": "eva32-dev",
        "timeout": 5,
        "retries": 2,
    }
    params.update(kwargs)
    return OpenAIClient(transport=transport, sleep=sleep, **params)


# --------------------------------------------------------------------------- #
# Адрес эндпоинта
# --------------------------------------------------------------------------- #

class TestEndpoint(unittest.TestCase):
    def test_bare_host_gets_v1(self):
        client = make_client(base="http://127.0.0.1:11434")
        self.assertEqual(client.endpoint, "http://127.0.0.1:11434/v1/chat/completions")

    def test_base_with_path_gets_v1(self):
        client = make_client(base="https://polza.ai/api")
        self.assertEqual(client.endpoint, "https://polza.ai/api/v1/chat/completions")

    def test_v1_not_doubled(self):
        for base in ("https://api.example.com/v1", "https://api.example.com/v1/"):
            with self.subTest(base=base):
                client = make_client(base=base)
                self.assertEqual(
                    client.endpoint, "https://api.example.com/v1/chat/completions"
                )

    def test_full_path_kept(self):
        base = "https://api.example.com/v1/chat/completions"
        self.assertEqual(make_client(base=base).endpoint, base)

    def test_trailing_slashes(self):
        client = make_client(base="http://127.0.0.1:11434///")
        self.assertEqual(client.endpoint, "http://127.0.0.1:11434/v1/chat/completions")


# --------------------------------------------------------------------------- #
# Форма запроса
# --------------------------------------------------------------------------- #

class TestRequest(unittest.TestCase):
    def test_payload_and_headers(self):
        transport = FakeTransport(ok_response())
        client = make_client(transport=transport, key="sk-secret")
        client.chat([{"role": "user", "content": "привет"}])

        call = transport.calls[0]
        self.assertEqual(call.url, "http://127.0.0.1:11434/v1/chat/completions")
        self.assertEqual(call.headers["Authorization"], "Bearer sk-secret")
        self.assertEqual(call.headers["Content-Type"], "application/json")
        self.assertEqual(call.payload["model"], "eva32-dev")
        self.assertEqual(call.payload["messages"], [{"role": "user", "content": "привет"}])
        self.assertEqual(call.payload["max_tokens"], 512)
        self.assertEqual(call.timeout, 5)

    def test_empty_key_no_authorization_header(self):
        transport = FakeTransport(ok_response())
        make_client(transport=transport, key="").chat("привет")
        self.assertNotIn("Authorization", transport.calls[0].headers)

    def test_max_tokens_passed(self):
        transport = FakeTransport(ok_response())
        make_client(transport=transport).chat("привет", max_tokens=64)
        self.assertEqual(transport.calls[0].payload["max_tokens"], 64)

    def test_message_objects_and_extra_keys(self):
        # Message-объект из eva/types.py (атрибуты role/content) и словарь с meta.
        system = SimpleNamespace(role="system", content="ты вежлив")
        transport = FakeTransport(ok_response())
        make_client(transport=transport).chat(
            [system, {"role": "user", "content": "привет", "meta": {"user_id": "u1"}}]
        )
        self.assertEqual(
            transport.calls[0].payload["messages"],
            [
                {"role": "system", "content": "ты вежлив"},
                {"role": "user", "content": "привет"},
            ],
        )

    def test_single_string_becomes_user_message(self):
        transport = FakeTransport(ok_response())
        make_client(transport=transport).chat("привет")
        self.assertEqual(
            transport.calls[0].payload["messages"],
            [{"role": "user", "content": "привет"}],
        )


# --------------------------------------------------------------------------- #
# Результат и usage
# --------------------------------------------------------------------------- #

class TestResult(unittest.TestCase):
    def test_text_usage_raw(self):
        transport = FakeTransport(ok_response("  готово  ", prompt=12, completion=7))
        result = make_client(transport=transport).chat("привет")

        self.assertIsInstance(result, ClientResult)
        self.assertEqual(result.text, "готово")
        self.assertEqual(result.usage["prompt_tokens"], 12)
        self.assertEqual(result.usage["completion_tokens"], 7)
        self.assertEqual(result.usage["total_tokens"], 19)
        self.assertEqual(result.usage["cost_rub"], 0.0)
        self.assertEqual(result.raw["model"], "test-model")
        self.assertIsInstance(result.raw, dict)

    def test_dict_access_like_contract(self):
        transport = FakeTransport(ok_response("ответ"))
        result = make_client(transport=transport).chat("привет")

        self.assertEqual(result["text"], "ответ")
        self.assertIn("usage", result)
        self.assertEqual(set(dict(result)), {"text", "usage", "raw"})
        self.assertEqual(result.get("usage"), result.usage)
        self.assertEqual(result.as_dict()["text"], "ответ")
        with self.assertRaises(KeyError):
            _ = result["nope"]

    def test_content_parts_joined(self):
        body = {
            "choices": [{"message": {"role": "assistant", "content": [
                {"type": "text", "text": "при"},
                {"type": "text", "text": "вет"},
            ]}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        }
        transport = FakeTransport(
            HTTPResponse(status=200, body=json.dumps(body).encode("utf-8"))
        )
        result = make_client(transport=transport).chat("привет")
        self.assertEqual(result.text, "привет")

    def test_missing_usage_gives_zeros(self):
        body = {"choices": [{"message": {"content": "ок"}}]}
        transport = FakeTransport(
            HTTPResponse(status=200, body=json.dumps(body).encode("utf-8"))
        )
        usage = make_client(transport=transport).chat("привет").usage
        self.assertEqual(usage["prompt_tokens"], 0)
        self.assertEqual(usage["completion_tokens"], 0)
        self.assertEqual(usage["total_tokens"], 0)
        self.assertEqual(usage["cost_rub"], 0.0)

    def test_http_response_header_is_case_insensitive(self):
        response = HTTPResponse(status=429, headers={"retry-after": "2"})
        self.assertEqual(response.header("Retry-After"), "2")
        self.assertIsNone(response.header("X-Missing"))


# --------------------------------------------------------------------------- #
# Стоимость
# --------------------------------------------------------------------------- #

class TestCost(unittest.TestCase):
    def test_cost_rub_from_usage(self):
        transport = FakeTransport(ok_response(cost=0.42))
        usage = make_client(transport=transport).chat("привет").usage
        self.assertEqual(usage["cost_rub"], 0.42)

    def test_cost_rub_string_from_usage(self):
        body = ok_response()
        payload = json.loads(body.body)
        payload["usage"]["cost_rub"] = "0.125"
        transport = FakeTransport(
            HTTPResponse(status=200, body=json.dumps(payload).encode("utf-8"))
        )
        usage = make_client(transport=transport).chat("привет").usage
        self.assertEqual(usage["cost_rub"], 0.125)

    def test_cost_rub_top_level(self):
        transport = FakeTransport(ok_response(extra={"cost_rub": "1.25"}))
        usage = make_client(transport=transport).chat("привет").usage
        self.assertEqual(usage["cost_rub"], 1.25)

    def test_cost_rub_computed_from_tariff(self):
        transport = FakeTransport(ok_response(prompt=100, completion=50))
        client = make_client(
            transport=transport,
            price_prompt_rub=10.0,
            price_completion_rub=20.0,
        )
        usage = client.chat("привет").usage
        # 100/1000*10 + 50/1000*20 = 1.0 + 1.0
        self.assertEqual(usage["cost_rub"], 2.0)

    def test_provider_cost_wins_over_tariff(self):
        transport = FakeTransport(ok_response(prompt=100, completion=50, cost=0.3))
        client = make_client(
            transport=transport, price_prompt_rub=10.0, price_completion_rub=20.0
        )
        self.assertEqual(client.chat("привет").usage["cost_rub"], 0.3)

    def test_bad_cost_ignored(self):
        body = ok_response()
        payload = json.loads(body.body)
        payload["usage"]["cost_rub"] = "дорого"
        transport = FakeTransport(
            HTTPResponse(status=200, body=json.dumps(payload).encode("utf-8"))
        )
        usage = make_client(transport=transport).chat("привет").usage
        self.assertEqual(usage["cost_rub"], 0.0)


# --------------------------------------------------------------------------- #
# Ретраи: 500, таймаут, 429
# --------------------------------------------------------------------------- #

class TestRetries(unittest.TestCase):
    def test_500_retries_then_client_error(self):
        transport = FakeTransport(*[error_response(503, "upstream overloaded") for _ in range(3)])
        sleep = SleepRecorder()
        client = make_client(transport=transport, sleep=sleep, retries=2)

        with self.assertRaises(ClientError) as ctx:
            client.chat("привет")

        error = ctx.exception
        self.assertEqual(transport.attempts, 3)  # 1 + retries
        self.assertEqual(sleep.delays, [0.5, 1.0])  # backoff 0.5 * 2**n
        self.assertEqual(error.attempts, 3)
        self.assertEqual(error.status, 503)
        self.assertEqual(error.kind, "http")
        self.assertTrue(error.retryable)
        self.assertIn("503", str(error))
        self.assertIn("upstream overloaded", str(error))

    def test_500_then_success(self):
        transport = FakeTransport(error_response(500), ok_response("со второй попытки"))
        sleep = SleepRecorder()
        result = make_client(transport=transport, sleep=sleep).chat("привет")

        self.assertEqual(result.text, "со второй попытки")
        self.assertEqual(transport.attempts, 2)
        self.assertEqual(sleep.delays, [0.5])

    def test_retries_zero_means_single_attempt(self):
        transport = FakeTransport(error_response(500))
        sleep = SleepRecorder()
        client = make_client(transport=transport, sleep=sleep, retries=0)
        with self.assertRaises(ClientError) as ctx:
            client.chat("привет")
        self.assertEqual(transport.attempts, 1)
        self.assertEqual(sleep.delays, [])
        self.assertEqual(ctx.exception.attempts, 1)

    def test_timeout_retries_then_client_error(self):
        transport = FakeTransport(*[socket.timeout("timed out") for _ in range(3)])
        sleep = SleepRecorder()
        client = make_client(transport=transport, sleep=sleep, retries=2)

        with self.assertRaises(ClientError) as ctx:
            client.chat("привет")

        error = ctx.exception
        self.assertEqual(transport.attempts, 3)
        self.assertEqual(sleep.delays, [0.5, 1.0])
        self.assertEqual(error.kind, "timeout")
        self.assertTrue(error.retryable)
        self.assertIsNone(error.status)
        self.assertIn("таймаут", str(error))
        self.assertIn("5", str(error))

    def test_timeout_then_success(self):
        transport = FakeTransport(socket.timeout("timed out"), ok_response("успел"))
        sleep = SleepRecorder()
        result = make_client(transport=transport, sleep=sleep).chat("привет")
        self.assertEqual(result.text, "успел")
        self.assertEqual(transport.attempts, 2)
        self.assertEqual(sleep.delays, [0.5])

    def test_urlerror_timeout_is_retryable_timeout(self):
        transport = FakeTransport(urllib.error.URLError(socket.timeout("timed out")), ok_response())
        client = make_client(transport=transport, sleep=SleepRecorder())
        client.chat("привет")
        self.assertEqual(transport.attempts, 2)

    def test_connection_refused_is_network_error(self):
        transport = FakeTransport(
            *[urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))] * 3
        )
        with self.assertRaises(ClientError) as ctx:
            make_client(transport=transport, sleep=SleepRecorder()).chat("привет")
        self.assertEqual(ctx.exception.kind, "network")
        self.assertTrue(ctx.exception.retryable)
        self.assertIn("сеть", str(ctx.exception))
        self.assertEqual(transport.attempts, 3)

    def test_dns_error_is_named(self):
        transport = FakeTransport(*[urllib.error.URLError(socket.gaierror(-2, "nodename nor servname"))] * 3)
        with self.assertRaises(ClientError) as ctx:
            make_client(transport=transport, sleep=SleepRecorder()).chat("привет")
        self.assertIn("DNS", str(ctx.exception))

    def test_429_retries_with_backoff(self):
        transport = FakeTransport(error_response(429, "rate limit"), ok_response())
        sleep = SleepRecorder()
        result = make_client(transport=transport, sleep=sleep, retries=2).chat("привет")
        self.assertEqual(result.text, "привет")
        self.assertEqual(transport.attempts, 2)
        self.assertEqual(sleep.delays, [0.5])

    def test_429_all_attempts_then_client_error(self):
        transport = FakeTransport(*[error_response(429, "rate limit") for _ in range(3)])
        sleep = SleepRecorder()
        client = make_client(transport=transport, sleep=sleep, retries=2)
        with self.assertRaises(ClientError) as ctx:
            client.chat("привет")
        error = ctx.exception
        self.assertEqual(transport.attempts, 3)
        self.assertEqual(sleep.delays, [0.5, 1.0])
        self.assertEqual(error.status, 429)
        self.assertTrue(error.retryable)
        self.assertIn("429", str(error))
        self.assertIn("rate limit", str(error))

    def test_429_retry_after_header(self):
        transport = FakeTransport(
            error_response(429, headers={"Retry-After": "1.5"}), ok_response()
        )
        sleep = SleepRecorder()
        make_client(transport=transport, sleep=sleep).chat("привет")
        self.assertEqual(sleep.delays, [1.5])

    def test_429_retry_after_capped(self):
        transport = FakeTransport(
            error_response(429, headers={"Retry-After": "999"}), ok_response()
        )
        sleep = SleepRecorder()
        make_client(transport=transport, sleep=sleep, max_backoff=8).chat("привет")
        self.assertEqual(sleep.delays, [8.0])

    def test_429_retry_after_http_date_ignored(self):
        transport = FakeTransport(
            error_response(
                429, headers={"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"}
            ),
            ok_response(),
        )
        sleep = SleepRecorder()
        make_client(transport=transport, sleep=sleep).chat("привет")
        self.assertEqual(sleep.delays, [0.5])

    def test_backoff_never_exceeds_max(self):
        transport = FakeTransport(*[error_response(500) for _ in range(3)])
        sleep = SleepRecorder()
        client = make_client(
            transport=transport, sleep=sleep, retries=2, backoff=5.0, max_backoff=6.0
        )
        with self.assertRaises(ClientError):
            client.chat("привет")
        self.assertEqual(sleep.delays, [5.0, 6.0])

    def test_default_sleep_hook_is_patched(self):
        # Проверяем именно модульное сечение: тест не должен ждать 0.5 с.
        transport = FakeTransport(error_response(429), ok_response())
        with mock.patch.object(openai_compat, "_sleep") as slept:
            make_client(transport=transport).chat("привет")
        slept.assert_called_once_with(0.5)

    def test_retry_is_logged_without_key(self):
        transport = FakeTransport(error_response(500, "upstream overloaded"), ok_response())
        with self.assertLogs("eva.clients", level="WARNING") as captured:
            make_client(transport=transport, sleep=SleepRecorder(), key="sk-secret").chat("привет")
        logged = "\n".join(captured.output)
        self.assertIn("повтор 1/2", logged)
        self.assertIn("500", logged)
        self.assertNotIn("sk-secret", logged)


# --------------------------------------------------------------------------- #
# Неретраибельные ошибки и разбор ответа
# --------------------------------------------------------------------------- #

class TestErrors(unittest.TestCase):
    def test_400_not_retried(self):
        transport = FakeTransport(error_response(400, "invalid model"), ok_response())
        sleep = SleepRecorder()
        client = make_client(transport=transport, sleep=sleep)

        with self.assertRaises(ClientError) as ctx:
            client.chat("привет")

        error = ctx.exception
        self.assertEqual(transport.attempts, 1)
        self.assertEqual(sleep.delays, [])
        self.assertEqual(error.status, 400)
        self.assertEqual(error.kind, "http")
        self.assertFalse(error.retryable)
        self.assertEqual(error.attempts, 1)
        self.assertIn("invalid model", str(error))
        self.assertIn("попыток: 1 из 3", str(error))

    def test_401_message_has_no_key(self):
        transport = FakeTransport(error_response(401, "incorrect api key provided"))
        client = make_client(transport=transport, key="sk-secret")

        with self.assertRaises(ClientError) as ctx:
            client.chat("привет")

        self.assertNotIn("sk-secret", str(ctx.exception))
        self.assertNotIn("sk-secret", repr(ctx.exception))
        self.assertIn("401", str(ctx.exception))

    def test_error_echoing_key_is_redacted(self):
        transport = FakeTransport(error_response(400, "bad key sk-secret"))
        client = make_client(transport=transport, key="sk-secret", retries=0)
        with self.assertRaises(ClientError) as ctx:
            client.chat("привет")
        self.assertNotIn("sk-secret", str(ctx.exception))
        self.assertNotIn("sk-secret", ctx.exception.detail or "")
        self.assertIn("***", str(ctx.exception))

    def test_repr_has_no_key(self):
        client = make_client(key="sk-secret")
        self.assertNotIn("sk-secret", repr(client))
        self.assertIn("eva32-dev", repr(client))

    def test_html_instead_of_json(self):
        transport = FakeTransport(
            HTTPResponse(status=200, body=b"<html>bad gateway</html>")
        )
        with self.assertRaises(ClientError) as ctx:
            make_client(transport=transport).chat("привет")
        error = ctx.exception
        self.assertEqual(error.kind, "response")
        self.assertFalse(error.retryable)
        self.assertIn("не JSON", str(error))
        self.assertIn("bad gateway", error.detail or "")
        self.assertEqual(transport.attempts, 1)

    def test_200_without_choices(self):
        transport = FakeTransport(
            HTTPResponse(status=200, body=b'{"object": "chat.completion"}')
        )
        with self.assertRaises(ClientError) as ctx:
            make_client(transport=transport).chat("привет")
        self.assertIn("choices", str(ctx.exception))
        self.assertEqual(ctx.exception.kind, "response")

    def test_error_body_with_plain_text(self):
        transport = FakeTransport(
            HTTPResponse(status=502, body=b"Bad Gateway: upstream died")
        )
        with self.assertRaises(ClientError) as ctx:
            make_client(transport=transport, sleep=SleepRecorder(), retries=0).chat("привет")
        self.assertIn("upstream died", str(ctx.exception))

    def test_empty_error_body(self):
        transport = FakeTransport(HTTPResponse(status=503, body=b""))
        with self.assertRaises(ClientError) as ctx:
            make_client(transport=transport, sleep=SleepRecorder(), retries=0).chat("привет")
        self.assertEqual(ctx.exception.status, 503)
        self.assertIn("503", str(ctx.exception))


# --------------------------------------------------------------------------- #
# Конфигурация и валидация входа
# --------------------------------------------------------------------------- #

class TestConfig(unittest.TestCase):
    def test_defaults_match_contract(self):
        client = OpenAIClient("http://127.0.0.1:11434", "", "eva32-dev")
        self.assertEqual(client.timeout, 60.0)
        self.assertEqual(client.retries, 2)
        self.assertEqual(
            client.endpoint, "http://127.0.0.1:11434/v1/chat/completions"
        )

    def test_bad_base(self):
        for base in ("", "localhost:8000", "ftp://example.com", "http://"):
            with self.subTest(base=base):
                with self.assertRaises((ValueError, TypeError)):
                    OpenAIClient(base, "", "m")

    def test_bad_timeout_and_retries(self):
        with self.assertRaises(ValueError):
            OpenAIClient("http://h:1", "", "m", timeout=0)
        with self.assertRaises(TypeError):
            OpenAIClient("http://h:1", "", "m", timeout="60")
        with self.assertRaises(ValueError):
            OpenAIClient("http://h:1", "", "m", retries=-1)
        with self.assertRaises(TypeError):
            OpenAIClient("http://h:1", "", "m", retries=2.5)

    def test_bad_model(self):
        with self.assertRaises(ValueError):
            OpenAIClient("http://h:1", "", "")

    def test_bad_max_tokens(self):
        client = make_client(transport=FakeTransport(ok_response()))
        with self.assertRaises(ValueError):
            client.chat("привет", max_tokens=0)
        with self.assertRaises(TypeError):
            client.chat("привет", max_tokens="512")

    def test_bad_messages(self):
        client = make_client(transport=FakeTransport(ok_response()))
        with self.assertRaises(ValueError):
            client.chat([])
        with self.assertRaises(TypeError):
            client.chat(None)
        with self.assertRaises(ValueError):
            client.chat([{"content": "без роли"}])
        with self.assertRaises(ValueError):
            client.chat([{"role": "user", "content": 123}])

    def test_bad_prices(self):
        with self.assertRaises(ValueError):
            OpenAIClient("http://h:1", "", "m", price_prompt_rub=-1)


# --------------------------------------------------------------------------- #
# Гарантии отсутствия сети
# --------------------------------------------------------------------------- #

class TestNoNetwork(unittest.TestCase):
    def test_injected_transport_never_touches_urlopen(self):
        boom = mock.Mock(side_effect=AssertionError("сеть в тестах запрещена"))
        with mock.patch("eva.clients.openai_compat.urllib.request.urlopen", boom):
            result = make_client(transport=FakeTransport(ok_response())).chat("привет")
        self.assertEqual(result.text, "привет")
        boom.assert_not_called()

    def test_module_transport_hook_is_used(self):
        transport = FakeTransport(ok_response("через подмену"))
        with mock.patch.object(openai_compat, "_post_json", transport):
            result = OpenAIClient("http://127.0.0.1:11434", "", "eva32-dev").chat("привет")
        self.assertEqual(result.text, "через подмену")
        self.assertEqual(transport.attempts, 1)

    def test_transport_gets_bytes_body(self):
        transport = FakeTransport(ok_response())
        make_client(transport=transport).chat("привет")
        self.assertIsInstance(transport.calls[0].raw_body, bytes)


if __name__ == "__main__":
    unittest.main()
