#!/usr/bin/env python3
"""Smoke-проверка живого EVA32 со стороны orioncode.

Проверяется ровно тот минимум, которым пользуется OpenAI-совместимый клиент:

* ``GET /health`` — процесс жив и отвечает как EVA32;
* ``GET /v1/models`` — модель ``eva-hybrid`` заявлена;
* ``POST /v1/chat/completions`` без стрима — ответ, поле ``eva``, маршрут и цена;
* ``POST /v1/chat/completions`` со стримом — SSE-чанки и ``[DONE]``.

Стрим — задача 0025. Пока сервер отвечает на ``stream=true`` ошибкой
400 ``unsupported_stream``, проверка печатает SKIP с пояснением и **не**
считается провалом: иначе smoke падал бы на каждом сервере до влития 0025.
После влития та же проверка начнёт проверять чанки сама, без правок скрипта.

Только стандартная библиотека. Сеть нужна по назначению: скрипт проверяет
*живой* сервер, а не подменяет его заглушкой.

Пример::

    python3 scripts/integration_smoke.py --url http://127.0.0.1:8010

Код возврата: 0 — всё прошло (пропуск стрима допустим), 1 — есть провал.
"""

from __future__ import annotations

import argparse
import json
import sys
from urllib import error, parse, request

DEFAULT_URL = "http://127.0.0.1:8010"
DEFAULT_MODEL = "eva-hybrid"
DEFAULT_PROMPT = "Сколько стоит тариф Старт?"
DEFAULT_KEY = "eva-local-key"

#: Маршруты, которые контракт (docs/hybrid.md) разрешает в поле ``eva.route``.
ALLOWED_ROUTES = ("skill", "small", "big")

#: Защита от сервера, который льёт SSE без конца.
MAX_STREAM_LINES = 10000


class SmokeFailure(RuntimeError):
    """Проверка не прошла; текст — для человека."""


class CheckSkipped(RuntimeError):
    """Проверка неприменима к этой версии сервера (не провал)."""


def split_url(raw: str) -> tuple[str, str]:
    """Возвращает ``(origin, api_base)`` из URL сервера.

    ``--url http://127.0.0.1:8010`` и ``--url http://127.0.0.1:8010/v1``
    дают один и тот же API base; ``/health`` всегда живёт в origin.
    """
    parts = parse.urlsplit(raw.strip())
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise SmokeFailure(
            "URL должен быть вида http://ХОСТ:ПОРТ[/v1], получено {!r}".format(raw)
        )
    origin = "{}://{}".format(parts.scheme, parts.netloc)
    path = parts.path.rstrip("/")
    api_path = path if path.endswith("/v1") else path + "/v1"
    return origin, origin + api_path


def _snippet(body: bytes, limit: int = 300) -> str:
    text = body.decode("utf-8", "replace")
    text = " ".join(text.split())
    if len(text) > limit:
        text = text[:limit] + "…"
    return text or "<пустое тело>"


def _parse_json(body: bytes, what: str) -> dict:
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SmokeFailure(
            "{}: ответ не JSON ({}): {}".format(what, exc, _snippet(body))
        ) from exc
    if not isinstance(value, dict):
        raise SmokeFailure("{}: ожидался JSON-объект, получено {}".format(what, type(value).__name__))
    return value


def _error_code(body: bytes) -> str:
    """Код ошибки из тела ``{"error": {"code": ...}}``; пустая строка, если нет."""
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    error_obj = payload.get("error")
    if not isinstance(error_obj, dict):
        return ""
    code = error_obj.get("code")
    return code if isinstance(code, str) else ""


def _fmt_rub(value: float) -> str:
    text = "{:.4f}".format(value).rstrip("0").rstrip(".")
    return text or "0"


class Smoke:
    def __init__(self, origin: str, api_base: str, args: argparse.Namespace) -> None:
        self.origin = origin
        self.api_base = api_base
        self.model = args.model
        self.key = args.key
        self.prompt = args.prompt
        self.timeout = args.timeout
        self.total = 4
        self.step = 0
        self.passed = 0
        self.skipped = 0
        self.failed = 0

    # -- HTTP ----------------------------------------------------------------

    def _headers(self, *, accept: str = "application/json", json_body: bool = False) -> dict:
        headers = {"Accept": accept}
        if self.key:
            headers["Authorization"] = "Bearer {}".format(self.key)
        if json_body:
            headers["Content-Type"] = "application/json"
        return headers

    def _body(self) -> bytes:
        return json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": self.prompt}],
            },
            ensure_ascii=False,
        ).encode("utf-8")

    def _request(self, method: str, url: str) -> tuple[int, bytes]:
        req = request.Request(
            url, data=None, headers=self._headers(), method=method
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                return resp.status, resp.read()
        except error.HTTPError as exc:
            return exc.code, exc.read()
        except error.URLError as exc:
            raise SmokeFailure(
                "нет соединения с {}: {}. Поднимите сервер: "
                "python3 -m eva --port 8010".format(url, exc.reason)
            ) from exc
        except OSError as exc:
            raise SmokeFailure(
                "нет соединения с {}: {}. Поднимите сервер: "
                "python3 -m eva --port 8010".format(url, exc)
            ) from exc

    def _post_chat(self) -> tuple[int, bytes]:
        req = request.Request(
            self.api_base + "/chat/completions",
            data=self._body(),
            headers=self._headers(json_body=True),
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                return resp.status, resp.read()
        except error.HTTPError as exc:
            return exc.code, exc.read()
        except error.URLError as exc:
            raise SmokeFailure(
                "POST {}: нет соединения: {}. Поднимите сервер: "
                "python3 -m eva --port 8010".format(
                    self.api_base + "/chat/completions", exc.reason
                )
            ) from exc
        except OSError as exc:
            raise SmokeFailure(
                "POST {}: нет соединения: {}".format(
                    self.api_base + "/chat/completions", exc
                )
            ) from exc

    # -- вывод ---------------------------------------------------------------

    def _head(self, name: str) -> str:
        self.step += 1
        return "[{}/{}] {}".format(self.step, self.total, name).ljust(46)

    def _run(self, name: str, check) -> None:
        head = self._head(name)
        try:
            detail, extra = check()
        except CheckSkipped as exc:
            self.skipped += 1
            print("{} SKIP  {}".format(head, exc))
            return
        except SmokeFailure as exc:
            self.failed += 1
            print("{} FAIL  {}".format(head, exc))
            return
        except Exception as exc:  # noqa: BLE001 - smoke не должен падать трейсбеком
            self.failed += 1
            print("{} FAIL  неожиданная ошибка: {!r}".format(head, exc))
            return

        self.passed += 1
        print("{} PASS  {}".format(head, detail))
        for line in extra:
            print("      {}".format(line))

    # -- проверки ------------------------------------------------------------

    def check_health(self) -> tuple[str, list]:
        status, body = self._request("GET", self.origin + "/health")
        if status != 200:
            raise SmokeFailure(
                "HTTP {} вместо 200: {}".format(status, _snippet(body))
            )
        data = _parse_json(body, "GET /health")
        if data.get("status") != "ok":
            raise SmokeFailure(
                "в /health status={!r}, ожидалось 'ok'".format(data.get("status"))
            )
        detail = "HTTP 200, status=ok, service={}, version={}".format(
            data.get("service", "?"), data.get("version", "?")
        )
        extra = [
            "model={}, uptime={} с".format(
                data.get("model", "?"), data.get("uptime_s", "?")
            )
        ]
        return detail, extra

    def check_models(self) -> tuple[str, list]:
        status, body = self._request("GET", self.api_base + "/models")
        if status != 200:
            raise SmokeFailure(
                "HTTP {} вместо 200: {}".format(status, _snippet(body))
            )
        data = _parse_json(body, "GET /v1/models")
        items = data.get("data")
        if not isinstance(items, list) or not items:
            raise SmokeFailure("в ответе нет непустого списка data")
        ids = [
            item.get("id")
            for item in items
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        ]
        if not ids:
            raise SmokeFailure("в data нет ни одного id модели")
        if self.model not in ids and DEFAULT_MODEL not in ids:
            raise SmokeFailure(
                "ни {!r}, ни {!r} нет среди моделей: {}".format(
                    self.model, DEFAULT_MODEL, ", ".join(ids)
                )
            )
        detail = "HTTP 200, модели: {}".format(", ".join(ids))
        extra = []
        if self.model not in ids:
            extra.append(
                "запрошенная модель {!r} не заявлена; сервер принимает любое имя "
                "и возвращает его в поле model".format(self.model)
            )
        return detail, extra

    def check_chat(self) -> tuple[str, list]:
        status, body = self._post_chat()
        if status != 200:
            raise SmokeFailure(
                "HTTP {} вместо 200: {}".format(status, _snippet(body))
            )
        data = _parse_json(body, "POST /v1/chat/completions")

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise SmokeFailure("в ответе нет непустого choices")
        first = choices[0]
        message = first.get("message") if isinstance(first, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise SmokeFailure("choices[0].message.content пуст или не строка")

        eva = data.get("eva")
        if not isinstance(eva, dict):
            raise SmokeFailure(
                "в ответе нет поля eva — сервер не той версии или это не EVA32"
            )
        route = eva.get("route")
        if route not in ALLOWED_ROUTES:
            raise SmokeFailure(
                "eva.route={!r}, ожидалось одно из {}".format(
                    route, "/".join(ALLOWED_ROUTES)
                )
            )
        cost = eva.get("cost_rub")
        if isinstance(cost, bool) or not isinstance(cost, (int, float)) or cost < 0:
            raise SmokeFailure("eva.cost_rub={!r}, ожидалось число >= 0".format(cost))
        if route == "skill" and cost > 0:
            raise SmokeFailure(
                "маршрут skill должен стоить 0 ₽, получено {}".format(cost)
            )

        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        prompt_tokens = usage.get("prompt_tokens", "?")
        completion_tokens = usage.get("completion_tokens", "?")
        total_tokens = usage.get("total_tokens", "?")
        skill = eva.get("skill")
        confidence = eva.get("confidence", "?")
        latency = eva.get("latency_ms", "?")
        reason = eva.get("reason") or "—"

        detail = "HTTP 200, route={}, cost={} ₽".format(route, _fmt_rub(cost))
        extra = [
            "reason: {}".format(reason),
            "skill={}, confidence={}, latency={} мс".format(
                skill if skill else "—", confidence, latency
            ),
            "tokens: prompt={}, completion={}, total={}".format(
                prompt_tokens, completion_tokens, total_tokens
            ),
            "answer: {}".format(" ".join(content.split())[:240]),
        ]
        return detail, extra

    def check_stream(self) -> tuple[str, list]:
        req = request.Request(
            self.api_base + "/chat/completions",
            data=json.dumps(
                {
                    "model": self.model,
                    "messages": [{"role": "user", "content": self.prompt}],
                    "stream": True,
                },
                ensure_ascii=False,
            ).encode("utf-8"),
            headers=self._headers(accept="text/event-stream", json_body=True),
            method="POST",
        )
        try:
            resp = request.urlopen(req, timeout=self.timeout)
        except error.HTTPError as exc:
            body = exc.read()
            code = _error_code(body)
            if exc.code == 400 and code == "unsupported_stream":
                raise CheckSkipped(
                    "HTTP 400 unsupported_stream — SSE (задача 0025) ещё не "
                    "влит; стрим не проверен, это не ошибка"
                ) from exc
            raise SmokeFailure(
                "HTTP {} вместо 200: {}".format(exc.code, _snippet(body))
            ) from exc
        except error.URLError as exc:
            raise SmokeFailure(
                "POST {}: нет соединения: {}".format(
                    self.api_base + "/chat/completions", exc.reason
                )
            ) from exc
        except OSError as exc:
            raise SmokeFailure(
                "POST {}: нет соединения: {}".format(
                    self.api_base + "/chat/completions", exc
                )
            ) from exc

        with resp:
            status = resp.status
            content_type = (resp.headers.get("Content-Type") or "").lower()
            if status != 200:
                raise SmokeFailure(
                    "HTTP {} вместо 200: {}".format(status, _snippet(resp.read()))
                )
            if not content_type.startswith("text/event-stream"):
                raise SmokeFailure(
                    "HTTP 200, но Content-Type={!r}, ожидался text/event-stream; "
                    "тело: {}".format(content_type, _snippet(resp.read(300)))
                )

            chunks = 0
            data_lines = 0
            finish_reasons = []
            pieces = []
            stream_eva = None
            done = False
            lines = 0

            while True:
                raw = resp.readline()
                if not raw:
                    break
                lines += 1
                if lines > MAX_STREAM_LINES:
                    raise SmokeFailure(
                        "SSE длиннее {} строк — похоже, поток не закрывается".format(
                            MAX_STREAM_LINES
                        )
                    )
                line = raw.decode("utf-8", "replace").strip()
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    done = True
                    break
                data_lines += 1
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError as exc:
                    raise SmokeFailure(
                        "SSE-событие {} не JSON: {} ({})".format(
                            data_lines, exc, _snippet(payload.encode("utf-8"))
                        )
                    ) from exc
                if not isinstance(event, dict):
                    raise SmokeFailure(
                        "SSE-событие {}: ожидался объект".format(data_lines)
                    )
                if event.get("error"):
                    raise SmokeFailure(
                        "SSE-событие {} содержит ошибку: {}".format(
                            data_lines, json.dumps(event["error"], ensure_ascii=False)
                        )
                    )
                choices = event.get("choices")
                if not isinstance(choices, list):
                    raise SmokeFailure(
                        "SSE-событие {}: choices не список".format(data_lines)
                    )
                if not choices:
                    # OpenAI при stream_options.include_usage шлёт событие без
                    # choices, только с usage. Сервер вправе делать так же.
                    if isinstance(event.get("usage"), dict):
                        continue
                    raise SmokeFailure(
                        "SSE-событие {}: пустой choices без usage".format(data_lines)
                    )
                if isinstance(event.get("eva"), dict):
                    stream_eva = event["eva"]
                first = choices[0]
                delta = first.get("delta") if isinstance(first, dict) else None
                if isinstance(delta, dict):
                    piece = delta.get("content")
                    if isinstance(piece, str) and piece:
                        pieces.append(piece)
                        chunks += 1
                finish = first.get("finish_reason") if isinstance(first, dict) else None
                if finish:
                    finish_reasons.append(str(finish))

        if not done:
            raise SmokeFailure(
                "SSE оборвался без data: [DONE] (событий: {})".format(data_lines)
            )
        if data_lines == 0:
            raise SmokeFailure("SSE: до [DONE] не пришло ни одного события")
        text = "".join(pieces)
        if not text.strip():
            raise SmokeFailure("SSE: в delta.content не пришло ни одного символа")

        detail = "HTTP 200, text/event-stream, чанков с текстом={}, [DONE]".format(
            chunks
        )
        extra = [
            "событий={}, finish_reason={}".format(
                data_lines, ", ".join(finish_reasons) if finish_reasons else "—"
            ),
            "content: {}".format(" ".join(text.split())[:240]),
        ]
        if stream_eva is not None:
            route = stream_eva.get("route", "?")
            skill = stream_eva.get("skill") or "—"
            cost = stream_eva.get("cost_rub")
            cost_text = (
                _fmt_rub(cost)
                if isinstance(cost, (int, float)) and not isinstance(cost, bool)
                else "?"
            )
            extra.append(
                "eva: route={}, cost={} ₽, skill={}".format(
                    route, cost_text, skill
                )
            )
        return detail, extra

    # -- итог ----------------------------------------------------------------

    def run(self) -> int:
        print(
            "EVA32 integration smoke: {} (API {})".format(self.origin, self.api_base)
        )
        print(
            "model={}, key={}, prompt={!r}".format(
                self.model,
                "задан" if self.key else "пустой",
                self.prompt,
            )
        )
        print()
        self._run("GET /health", self.check_health)
        self._run("GET /v1/models", self.check_models)
        self._run("POST /v1/chat/completions", self.check_chat)
        self._run(
            "POST /v1/chat/completions (stream)",
            self.check_stream,
        )
        print()
        print(
            "ИТОГ: {} PASS, {} SKIP, {} FAIL".format(
                self.passed, self.skipped, self.failed
            )
        )
        if self.failed:
            print("Есть провалы — сервер не готов для orioncode (строки FAIL выше).")
            return 1
        print("Сервер отвечает по контракту OpenAI-совместимого API EVA32.")
        if self.skipped:
            print("Пропущенные проверки не считаются провалом (см. SKIP выше).")
        return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="integration_smoke.py",
        description="Smoke-проверка живого EVA32 для подключения orioncode.",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="адрес сервера, с /v1 или без (по умолчанию {})".format(DEFAULT_URL),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="имя модели в запросе (по умолчанию {})".format(DEFAULT_MODEL),
    )
    parser.add_argument(
        "--key",
        default=DEFAULT_KEY,
        help="API key: любой непустой, сервер его не проверяет",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="вопрос для чата; по умолчанию берётся скилл pricing, без модели",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=90.0,
        help="таймаут одного HTTP-запроса, сек (по умолчанию 90)",
    )
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout должен быть больше нуля")
    if not args.prompt.strip():
        parser.error("--prompt не должен быть пустым")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        origin, api_base = split_url(args.url)
    except SmokeFailure as exc:
        print("ОШИБКА: {}".format(exc), file=sys.stderr)
        return 1
    return Smoke(origin, api_base, args).run()


if __name__ == "__main__":
    sys.exit(main())
