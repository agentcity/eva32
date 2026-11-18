#!/usr/bin/env python3
"""Бенчмарк гибрида: латентность под нагрузкой, маршруты и цена.

Один замер средней латентности ничего не говорит о хвостах: сервер, который
«в среднем быстрый», может на каждом сотом запросе думать втрое дольше — и
именно поэтому нельзя решать, выставлять ли его наружу, по среднему. Этот
скрипт гоняет N запросов (по умолчанию 100), в том числе параллельно, и
печатает p50/p95/p99, пропускную способность, долю маршрутов и цену за 1000
запросов.

Два режима:

``--offline``
    Без сети и без ключей: ``eva.router.answer`` вызывается с настоящими
    скиллами и классификатором, но клиенты моделей подменены заглушками
    (``StubClient``). Цифры — накладные расходы роутера и заглушек, а **не**
    латентность моделей; режим нужен для регрессии кода и проверки, что
    параллельность не ломает маршрутизацию. Искусственную задержку заглушки
    можно задать ``--stub-latency-ms`` — тогда видно, как растёт throughput с
    ``--concurrency``.

``--http URL``
    Против поднятого сервера (``python3 -m eva --port 8010``): настоящий HTTP,
    ``POST /v1/chat/completions`` без SSE. Латентность — со стороны клиента,
    включая сеть и очередь сервера; рядом в JSON есть ``server_latency_ms`` из
    поля ``eva`` (сколько запрос обрабатывал сам сервер).

Ни один запрос не роняет прогон: ошибки и таймауты считаются отдельно и
попадают в отчёт, а не в трейсбек. Таймаут в HTTP-режиме — реальный
(``urllib`` с ``timeout``); в offline-режиме прервать уже начатый вызов нельзя,
поэтому запрос, который шёл дольше ``--timeout``, помечается таймаутом после
возврата и в перцентили не входит.

Вопросы берутся из ``eval/hybrid_golden.jsonl`` (та же разметка, что у
``tools/eval_hybrid.py``), но разметка маршрутов здесь не проверяется — важен
набор текстов. Из набора детерминированно выбирается N вопросов с равномерным
шагом, чтобы даже ``--n 10`` попал в разные маршруты.

Перцентили — метод R-7 (линейная интерполяция между соседними значениями, он
же по умолчанию в numpy); так же считает ``eva.telemetry.percentile``, поэтому
цифры бенчмарка и ``tools/usage_report.py`` сопоставимы. Для ``[1, 2, 3, 4]``
p50 = 2.5, для латентностей 1..100 мс p95 = 95.05 мс.

Цена за 1000 — та же модель, что у ``tools/eval_hybrid.py``: тариф ₽ за 1000
токенов из конфига, оценка prompt по символам (``CHARS_PER_TOKEN``) плюс
системный промпт, ожидаемая длина ответа — константы по маршрутам. Успешные
ответы с неизвестным маршрутом оцениваются как ``big`` (консервативно: лучше
завысить, чем решить, что сервер бесплатный). Таймауты и ошибки в цену не
входят — их число печатается отдельно.

Бюджет латентности. В ``docs/hybrid.md`` бюджета p95 на 2026-11-18 нет, поэтому
по умолчанию строка бюджета сообщает об этом и ничего не сравнивает. Если
бюджет там появится (строка со словами «бюджет» и ``p95 ≤ N мс``), он будет
подхвачен; задать вручную — ``--p95-budget-ms N``. Превышение бюджета — код
возврата 1.

Запуск::

    python3 tools/bench_eva.py --offline --n 100 --concurrency 4
    python3 tools/bench_eva.py --offline --n 200 --concurrency 8 --stub-latency-ms 50
    python3 tools/bench_eva.py --http http://127.0.0.1:8010 --n 100 --concurrency 4
    python3 tools/bench_eva.py --http http://127.0.0.1:8010 --n 100 --json
    python3 tools/bench_eva.py --http http://127.0.0.1:8010 --token "$EVA_AUTH_TOKEN"

Код возврата: 0 — метрики посчитаны и бюджет не превышен; 2 — нет модулей
``eva`` или набор вопросов не читается (считать нечем); 1 — ни один запрос не
удался либо превышен бюджет p95.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import re
import socket
import sys
import time
import traceback
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import is_dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Скрипт лежит в tools/, пакет eva — в корне: без этого запуск
# `python3 tools/bench_eva.py` из другого каталога не увидит eva.
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

QUESTIONS = os.path.join(ROOT, "eval", "hybrid_golden.jsonl")
SYSTEM_PROMPT = os.path.join(ROOT, "prompts", "system.ru.md")
HYBRID_DOC = os.path.join(ROOT, "docs", "hybrid.md")

#: Маршруты контракта ``docs/hybrid.md``. Всё прочее — ``other``.
ROUTE_KINDS = ("skill", "small", "big")

DEFAULT_N = 100
DEFAULT_CONCURRENCY = 4
DEFAULT_TIMEOUT = 30.0
DEFAULT_MODEL = "eva-hybrid"
DEFAULT_URL = "http://127.0.0.1:8010"

#: Резервный тариф, ₽ за 1000 токенов: (prompt, completion).
#: Совпадает с умолчаниями ``eva/config.py`` (малая — своё железо, 0/0).
DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    "skill": (0.0, 0.0),
    "small": (0.0, 0.0),
    "big": (0.40, 1.20),
}

#: Символов на токен — та же грубая оценка, что в ``tools/eval_hybrid.py``.
CHARS_PER_TOKEN = 3.0

#: Ожидаемая длина ответа, токены: скилл модель не зовёт, малая короче большой.
COMPLETION_TOKENS: dict[str, int] = {"skill": 0, "small": 120, "big": 220}

#: Стенд: обе модели «настроены» адресами-заглушками. Клиенты всё равно не
#: создаются — ``answer`` получает stub-клиенты через ``deps``. Пустые
#: ``EVA_BIG_BASE``/``EVA_BIG_MODEL`` заставили бы роутер считать большую
#: модель недоступной и исказили бы маршруты.
STUB_ENV = {
    "EVA_SMALL_BASE": "http://127.0.0.1:9/v1",
    "EVA_SMALL_KEY": "bench-stub",
    "EVA_SMALL_MODEL": "bench-stub-small",
    "EVA_SMALL_TIMEOUT": "1",
    "EVA_SMALL_RETRIES": "0",
    "EVA_SMALL_PRICE_PROMPT": "0",
    "EVA_SMALL_PRICE_COMPLETION": "0",
    "EVA_BIG_BASE": "http://127.0.0.1:9/v1",
    "EVA_BIG_KEY": "bench-stub",
    "EVA_BIG_MODEL": "bench-stub-big",
    "EVA_BIG_TIMEOUT": "1",
    "EVA_BIG_RETRIES": "0",
    # Без лимита: замеряем маршрут и задержку, а не срабатывание бюджета.
    "EVA_MAX_COST_RUB": "none",
    "EVA_MAX_TOKENS": "512",
    "EVA_AUTH_TOKEN": "",
    "EVA_RATE_LIMIT_PER_MIN": "0",
}

#: Текст-маркер ответа заглушки; содержание в метрики не входит.
STUB_PREFIX = "[заглушка "

#: Бюджет p95 по умолчанию. ``None`` — потому что в docs/hybrid.md его нет;
#: если появится, :func:`budget_from_docs` подхватит.
DEFAULT_P95_BUDGET_MS: float | None = None

#: Строка бюджета: «бюджет ... p95 ≤ 250 мс». Ищем в docs/hybrid.md.
_BUDGET_RE = re.compile(
    r"p95[^\d\n]{0,40}?(\d+(?:[.,]\d+)?)\s*(?:мс|ms)\b", re.IGNORECASE
)


class BenchUnavailable(RuntimeError):
    """Модулей ``eva`` нет — offline-замер делать не на чем."""


# --------------------------------------------------------------------------- #
# перцентили и статистика
# --------------------------------------------------------------------------- #
def percentile(values: Iterable[float], q: float) -> float | None:
    """Перцентиль ``q`` (0..100) линейной интерполяцией; ``None`` для пустого.

    Метод R-7: позиция ``(n - 1) * q / 100``, значение между соседними
    элементами берётся пропорционально. Он же используется в
    ``eva.telemetry.percentile``, поэтому p50/p95 бенчмарка и отчёта по логу
    считаются одинаково.
    """
    if not 0 <= q <= 100:
        raise ValueError("перцентиль должен быть в диапазоне 0..100")
    data = sorted(float(v) for v in values)
    if not data:
        return None
    if len(data) == 1:
        return data[0]
    pos = (len(data) - 1) * (q / 100.0)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return data[int(pos)]
    return data[lo] + (data[hi] - data[lo]) * (pos - lo)


def latency_stats(values: Iterable[float] | None) -> dict[str, float | int | None]:
    """Сводка латентности: count, p50, p95, p99, min, mean, max (мс)."""
    data = [float(v) for v in (values or []) if v is not None]
    if not data:
        return {
            "count": 0,
            "p50": None,
            "p95": None,
            "p99": None,
            "min": None,
            "mean": None,
            "max": None,
        }
    return {
        "count": len(data),
        "p50": round(percentile(data, 50), 3),
        "p95": round(percentile(data, 95), 3),
        "p99": round(percentile(data, 99), 3),
        "min": round(min(data), 3),
        "mean": round(sum(data) / len(data), 3),
        "max": round(max(data), 3),
    }


# --------------------------------------------------------------------------- #
# вопросы
# --------------------------------------------------------------------------- #
def load_questions(path: str = QUESTIONS) -> list[str]:
    """Прочитать вопросы из JSONL (``{"question": ...}`` или голая строка).

    Разметка маршрутов не нужна: бенчмарк проверяет код под нагрузкой, а не
    качество классификатора (для этого есть ``tools/eval_hybrid.py``).
    """
    rows: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: невалидный JSON: {exc}") from exc
            if isinstance(row, Mapping):
                question = row.get("question")
            elif isinstance(row, str):
                question = row
            else:
                question = None
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"{path}:{lineno}: пустой question")
            rows.append(question.strip())
    if not rows:
        raise ValueError(f"{path}: пустой набор вопросов")
    return rows


def pick_questions(questions: Sequence[str], n: int) -> list[str]:
    """Выбрать ``n`` вопросов с равномерным шагом по набору.

    Так ``--n 5`` попадает в разные маршруты, а не в первые пять (в наборе
    первые строки — сплошь скиллы). При ``n`` больше набора вопросы
    повторяются по кругу, но каждый раз с новым шагом.
    """
    if n <= 0:
        raise ValueError("N должно быть положительным")
    total = len(questions)
    if total == 0:
        raise ValueError("набор вопросов пуст")
    return [questions[(i * total) // n % total] for i in range(n)]


# --------------------------------------------------------------------------- #
# цена
# --------------------------------------------------------------------------- #
def estimate_tokens(text: str) -> int:
    """Грубая оценка токенов по символам; не меньше 1 для непустого текста."""
    text = str(text or "")
    if not text.strip():
        return 0
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN))


def request_cost(
    kind: str,
    prompt_tokens: int,
    prices: Mapping[str, tuple[float, float]],
    completion_tokens: Mapping[str, int] | None = None,
) -> float:
    """Стоимость одного запроса, ₽. Скилл модель не зовёт и стоит 0."""
    if kind == "skill":
        return 0.0
    prompt_price, completion_price = prices.get(kind, (0.0, 0.0))
    completion = int((completion_tokens or COMPLETION_TOKENS).get(kind, 0))
    return (
        max(0, int(prompt_tokens)) / 1000.0 * float(prompt_price)
        + max(0, completion) / 1000.0 * float(completion_price)
    )


def cost_per_1000(
    kinds: Iterable[str],
    questions: Iterable[str],
    prices: Mapping[str, tuple[float, float]],
    system_tokens: int = 0,
    completion_tokens: Mapping[str, int] | None = None,
) -> float:
    """Ожидаемая цена 1000 запросов при данном распределении маршрутов, ₽."""
    total = 0.0
    count = 0
    for kind, question in zip(kinds, questions):
        # Неизвестный маршрут оцениваем как big: завысить безопаснее, чем
        # решить, что сервер, не отдающий поле eva, бесплатный.
        priced = kind if kind in ROUTE_KINDS else "big"
        prompt_tokens = estimate_tokens(question) + (system_tokens if priced != "skill" else 0)
        total += request_cost(priced, prompt_tokens, prices, completion_tokens)
        count += 1
    if not count:
        return 0.0
    return total / count * 1000.0


def system_prompt_tokens(path: str = SYSTEM_PROMPT) -> int:
    """Оценка токенов системного промпта (прод шлёт его с каждым запросом)."""
    try:
        with open(path, encoding="utf-8") as fh:
            return estimate_tokens(fh.read())
    except OSError:
        return 0


def prices_from_config(cfg: Any) -> dict[str, tuple[float, float]]:
    """Тариф (₽/1k prompt, ₽/1k completion) по маршрутам из конфига."""
    prices = dict(DEFAULT_PRICES)
    method = getattr(cfg, "prices", None)
    if callable(method):
        for kind in ROUTE_KINDS:
            try:
                pair = method(kind)
                if pair and len(pair) == 2:
                    prices[kind] = (float(pair[0]), float(pair[1]))
            except Exception:  # noqa: BLE001 — остаётся резервный тариф
                pass
    return prices


def resolve_prices(
    prices: Mapping[str, tuple[float, float]], args: argparse.Namespace
) -> dict[str, tuple[float, float]]:
    """Наложить тариф из ``--*-price-in/out`` на базовый."""
    result = {kind: (float(v[0]), float(v[1])) for kind, v in prices.items()}
    overrides = {
        "small": (args.small_price_in, args.small_price_out),
        "big": (args.big_price_in, args.big_price_out),
    }
    for kind, (prompt_price, completion_price) in overrides.items():
        current = list(result.get(kind, (0.0, 0.0)))
        if prompt_price is not None:
            current[0] = prompt_price
        if completion_price is not None:
            current[1] = completion_price
        result[kind] = (current[0], current[1])
    return result


# --------------------------------------------------------------------------- #
# бюджет из документации
# --------------------------------------------------------------------------- #
def budget_from_docs(path: str = HYBRID_DOC) -> float | None:
    """Бюджет p95 из ``docs/hybrid.md``; ``None`` — его там нет.

    Ищем строку, где есть и слово «бюджет», и ``p95 ≤ N мс``. Так случайное
    упоминание p95 в тексте не станет порогом, а явная строка бюджета — станет.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return None
    for line in text.splitlines():
        if "бюджет" not in line.lower():
            continue
        match = _BUDGET_RE.search(line)
        if not match:
            continue
        try:
            return float(match.group(1).replace(",", "."))
        except ValueError:  # pragma: no cover — регэксп уже гарантирует число
            continue
    return None


# --------------------------------------------------------------------------- #
# offline: конфиг и заглушки
# --------------------------------------------------------------------------- #
def _with_stub_models(cfg: Any) -> Any:
    """Гарантировать, что обе модели «настроены» (нужно для честных маршрутов)."""
    wanted = {
        "small_base": STUB_ENV["EVA_SMALL_BASE"],
        "small_model": STUB_ENV["EVA_SMALL_MODEL"],
        "big_base": STUB_ENV["EVA_BIG_BASE"],
        "big_model": STUB_ENV["EVA_BIG_MODEL"],
        "big_key": STUB_ENV["EVA_BIG_KEY"],
    }
    if not hasattr(cfg, "big_base"):
        return cfg
    changes = {
        key: value
        for key, value in wanted.items()
        if hasattr(cfg, key) and not getattr(cfg, key)
    }
    if not changes:
        return cfg
    if is_dataclass(cfg):
        try:
            return replace(cfg, **changes)
        except Exception:  # noqa: BLE001 — не frozen/не dataclass-поля
            pass
    for key, value in changes.items():
        try:
            setattr(cfg, key, value)
        except Exception:  # noqa: BLE001
            pass
    return cfg


def load_offline_config() -> Any:
    """Конфиг стенда из ``STUB_ENV``, без чтения реального ``.env``.

    Реальный ``.env`` мог бы содержать настоящие базы моделей и бюджет, и
    offline-замер молча ушёл бы в сеть или срезал маршрут бюджетом. Поэтому
    окружение подменяется целиком, а клиенты всё равно приходят через ``deps``.
    """
    try:
        config_module = importlib.import_module("eva.config")
    except ModuleNotFoundError as exc:
        raise BenchUnavailable(
            "модуль eva.config не найден — offline-замер делать не на чем "
            "(контракт docs/hybrid.md)"
        ) from exc

    env = dict(STUB_ENV)
    cfg: Any = None
    for name in ("load", "from_env"):
        factory = getattr(config_module, name, None)
        if not callable(factory):
            continue
        try:
            cfg = factory(env)
        except TypeError:
            try:
                cfg = factory()
            except TypeError:
                continue
        if cfg is not None:
            break
    if cfg is None:
        cls = getattr(config_module, "Config", None)
        if cls is not None and hasattr(cls, "from_env"):
            cfg = cls.from_env(env)
    if cfg is None:
        raise BenchUnavailable(
            "eva.config есть, но конфиг не собрать: нет load()/from_env()/"
            "Config.from_env() (контракт docs/hybrid.md)"
        )
    return _with_stub_models(cfg)


class StubClient:
    """Клиент модели, который никуда не ходит и отвечает маркером.

    ``delay_ms`` — искусственная задержка (``time.sleep`` отпускает GIL), чтобы
    проверить, как ``--concurrency`` поднимает throughput. По умолчанию 0:
    offline меряет накладные расходы роутера, а не модели.

    Объект неизменяемый, поэтому один экземпляр безопасно живёт в пуле потоков.
    """

    def __init__(self, kind: str, delay_ms: float = 0.0) -> None:
        self.kind = kind
        self.delay_ms = max(0.0, float(delay_ms))

    def chat(self, messages: list[Any], max_tokens: int = 512) -> dict[str, Any]:
        if self.delay_ms:
            time.sleep(self.delay_ms / 1000.0)
        text = ""
        if messages:
            last = messages[-1]
            if isinstance(last, Mapping):
                text = str(last.get("content") or "")
            else:
                text = str(getattr(last, "content", "") or "")
        return {
            "text": f"{STUB_PREFIX}{self.kind}] {text[:60]}",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "cost_rub": 0.0},
            "raw": {"stub": True, "kind": self.kind},
        }


def _make_request(types_module: Any, text: str) -> Any:
    request_cls = getattr(types_module, "Request")
    message = types_module.Message(role="user", content=text)
    return request_cls(
        messages=[message], meta={"user_id": "bench", "channel": "bench"}
    )


def _stub_deps(small: StubClient, big: StubClient) -> dict[str, Any]:
    """Все имена, которые понимает ``router._resolve_client``.

    Если роутер переименует ключ, замер не должен уйти в настоящий клиент (а
    значит, в сеть): лучше явная ошибка, чем тихий HTTP-запрос.
    """
    return {
        "small": small,
        "big": big,
        "small_client": small,
        "big_client": big,
        "client_small": small,
        "client_big": big,
        "clients": {"small": small, "big": big},
    }


# --------------------------------------------------------------------------- #
# один запрос: offline и HTTP
# --------------------------------------------------------------------------- #
def _new_record(question: str) -> dict[str, Any]:
    return {
        "question": question,
        "route": None,
        "latency_ms": 0.0,
        "server_latency_ms": None,
        "cost_rub": None,
        "ok": False,
        "timeout": False,
        "error": None,
    }


def _offline_once(
    router: Any,
    types_module: Any,
    question: str,
    cfg: Any,
    deps: Mapping[str, Any],
    timeout: float,
) -> dict[str, Any]:
    """Один вызов ``router.answer`` с заглушками; исключения — в запись."""
    record = _new_record(question)
    started = time.perf_counter()
    try:
        answer = router.answer(_make_request(types_module, question), cfg, deps)
        route = getattr(answer, "route", None)
        kind = str(getattr(route, "kind", "") or "").strip().lower()
        record["route"] = kind or "unknown"
        cost = getattr(answer, "cost_rub", None)
        record["cost_rub"] = float(cost) if cost is not None else None
        record["ok"] = True
    except Exception as exc:  # noqa: BLE001 — одна ошибка не роняет прогон
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
    if record["ok"] and timeout > 0 and record["latency_ms"] > timeout * 1000.0:
        # Поток нельзя прервать снаружи: помечаем постфактум и исключаем из
        # перцентилей, но в счётчике таймаутов запрос остаётся.
        record["ok"] = False
        record["timeout"] = True
        record["error"] = (
            f"таймаут: {record['latency_ms']:.1f} мс > {timeout * 1000.0:.0f} мс"
        )
    return record


def _as_float(value: Any, default: float | None = 0.0) -> float | None:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return int(value)
    return default


def parse_http_payload(payload: Any) -> tuple[str, float | None, int | None, str]:
    """Разобрать ответ ``/v1/chat/completions`` -> (маршрут, цена, латентность, текст).

    Маршрут берётся из дополнительного поля ``eva`` (``docs/hybrid.md``); его
    отсутствие — не ошибка формата (ответ мог дать другой OpenAI-совместимый
    сервер), но маршрут тогда ``unknown`` и в цену пойдёт как ``big``.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("ответ сервера — не JSON-объект")
    route = "unknown"
    cost: float | None = None
    server_latency: int | None = None
    eva = payload.get("eva")
    if isinstance(eva, Mapping):
        kind = eva.get("route")
        if isinstance(kind, str) and kind.strip().lower() in ROUTE_KINDS:
            route = kind.strip().lower()
        cost = _as_float(eva.get("cost_rub"), None)
        server_latency = _as_int(eva.get("latency_ms"), 0) if "latency_ms" in eva else None

    text = ""
    choices = payload.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        message = choices[0].get("message")
        if isinstance(message, Mapping) and isinstance(message.get("content"), str):
            text = message["content"]
    if not text.strip():
        raise ValueError("в ответе нет непустого choices[0].message.content")
    return route, cost, server_latency, text


def _http_error_text(exc: urllib.error.HTTPError) -> str:
    """Код и сообщение из JSON-ошибки сервера; без JSON — короткий текст."""
    try:
        raw = exc.read()
    except Exception:  # noqa: BLE001 — тело могло не прийти
        return str(exc.reason or "")
    try:
        text = raw.decode("utf-8", "replace")
    except Exception:  # noqa: BLE001
        return str(exc.reason or "")
    try:
        payload = json.loads(text)
    except ValueError:
        return text.strip()[:200]
    if isinstance(payload, Mapping):
        error = payload.get("error")
        if isinstance(error, Mapping):
            code = error.get("code")
            message = str(error.get("message") or "").strip()
            return f"{code}: {message}" if code else message
        if isinstance(error, str):
            return error[:200]
    return text.strip()[:200]


def _http_once(
    endpoint: str,
    question: str,
    token: str,
    model: str,
    timeout: float,
) -> dict[str, Any]:
    """Один HTTP-запрос; сетевые ошибки — в запись, а не исключением."""
    record = _new_record(question)
    body = json.dumps(
        {"model": model, "messages": [{"role": "user", "content": question}]},
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
        payload = json.loads(raw.decode("utf-8"))
        route, cost, server_latency, _text = parse_http_payload(payload)
        record["route"] = route
        record["cost_rub"] = cost
        record["server_latency_ms"] = server_latency
        record["ok"] = True
    except urllib.error.HTTPError as exc:
        record["error"] = f"HTTP {exc.code}: {_http_error_text(exc)}"
    except (TimeoutError, socket.timeout):
        record["timeout"] = True
        record["error"] = f"таймаут {timeout:g} с"
    except urllib.error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, (TimeoutError, socket.timeout)):
            record["timeout"] = True
            record["error"] = f"таймаут {timeout:g} с"
        else:
            record["error"] = f"{type(reason).__name__}: {reason}"
    except (ValueError, UnicodeDecodeError) as exc:
        record["error"] = f"некорректный ответ сервера: {type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001 — прогон не должен падать
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
    return record


# --------------------------------------------------------------------------- #
# сводка
# --------------------------------------------------------------------------- #
def summarize(
    records: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    n: int,
    concurrency: int,
    wall_s: float,
    prices: Mapping[str, tuple[float, float]],
    system_tokens: int,
    completion_tokens: Mapping[str, int] | None = None,
    warmup: int = 0,
    url: str | None = None,
    p95_budget_ms: float | None = None,
) -> dict[str, Any]:
    """Свести записи прогона в метрики (без печати — её тест не парсит)."""
    data = list(records)
    total = len(data)
    ok = [r for r in data if r.get("ok")]
    timeouts = [r for r in data if r.get("timeout")]
    errors = [r for r in data if not r.get("ok") and not r.get("timeout")]

    stats = latency_stats([r.get("latency_ms") for r in ok])
    server_stats = latency_stats(
        [r.get("server_latency_ms") for r in ok if r.get("server_latency_ms") is not None]
    )

    routes = {kind: sum(1 for r in ok if r.get("route") == kind) for kind in ROUTE_KINDS}
    routes["other"] = sum(1 for r in ok if r.get("route") not in ROUTE_KINDS)
    shares = {
        kind: (count / len(ok) if ok else 0.0) for kind, count in routes.items()
    }

    # Латентность по маршрутам — видно, что тянут хвост именно большие ответы.
    by_route: dict[str, dict[str, Any]] = {}
    for kind in (*ROUTE_KINDS, "other"):
        subset = [r for r in ok if (r.get("route") or "other") == kind]
        if subset:
            by_route[kind] = latency_stats([r.get("latency_ms") for r in subset])

    kinds = [str(r.get("route") or "other") for r in ok]
    questions = [str(r.get("question") or "") for r in ok]
    estimated = cost_per_1000(
        kinds, questions, prices, system_tokens, completion_tokens
    )
    reported_values = [
        float(r["cost_rub"]) for r in ok if r.get("cost_rub") is not None
    ]
    reported = (
        sum(reported_values) / len(reported_values) * 1000.0
        if reported_values
        else None
    )

    p95 = stats["p95"]
    budget_ok: bool | None = None
    if p95_budget_ms is not None and p95 is not None:
        budget_ok = p95 <= float(p95_budget_ms)

    return {
        "mode": mode,
        "url": url,
        "n": n,
        "concurrency": concurrency,
        "warmup": warmup,
        "total": total,
        "ok": len(ok),
        "errors": len(errors),
        "timeouts": len(timeouts),
        "error_samples": [r.get("error") for r in (errors + timeouts)[:5]],
        "wall_s": round(float(wall_s), 6),
        "throughput_rps": (round(total / wall_s, 3) if wall_s > 0 else None),
        "ok_throughput_rps": (round(len(ok) / wall_s, 3) if wall_s > 0 else None),
        "latency_ms": stats,
        "latency_by_route_ms": by_route,
        "server_latency_ms": server_stats if server_stats["count"] else None,
        "routes": routes,
        "shares": shares,
        "prices_rub_per_1k": {kind: [p[0], p[1]] for kind, p in prices.items()},
        "system_tokens": system_tokens,
        "completion_tokens": dict(completion_tokens or COMPLETION_TOKENS),
        "cost_per_1000_rub": round(estimated, 4),
        "reported_cost_per_1000_rub": (
            round(reported, 4) if reported is not None else None
        ),
        "p95_budget_ms": p95_budget_ms,
        "budget_ok": budget_ok,
        "records": data,
    }


# --------------------------------------------------------------------------- #
# прогоны
# --------------------------------------------------------------------------- #
def _warm_up(
    pool: ThreadPoolExecutor,
    task: Any,
    picked: Sequence[str],
    warmup: int,
    concurrency: int,
) -> int:
    """Прогреть пул: не меньше потоков, иначе часть workers стартует в замере.

    Первый вызов в новом потоке заметно дороже следующих, а первый запрос в
    процессе тянет ленивые импорты скиллов и классификатора. Оба эффекта не
    должны попадать в перцентили, поэтому прогрев идёт через тот же пул и
    захватывает все его потоки. Возвращает число прогревочных запросов.
    """
    count = max(0, int(warmup))
    if not count:
        return 0
    requests = max(count, max(1, int(concurrency)))
    jobs = [pool.submit(task, picked[index % len(picked)]) for index in range(requests)]
    for job in jobs:
        try:
            job.result()
        except Exception:  # noqa: BLE001 — сбой прогрева покажет сам прогон
            pass
    return requests


def run_offline(
    questions: Sequence[str],
    n: int = DEFAULT_N,
    concurrency: int = DEFAULT_CONCURRENCY,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    warmup: int = 1,
    stub_latency_ms: float = 0.0,
    cfg: Any = None,
    prices: Mapping[str, tuple[float, float]] | None = None,
    system_tokens: int | None = None,
    completion_tokens: Mapping[str, int] | None = None,
    p95_budget_ms: float | None = None,
) -> dict[str, Any]:
    """Прогнать N запросов через ``eva.router.answer`` с заглушками моделей."""
    try:
        router = importlib.import_module("eva.router")
        types_module = importlib.import_module("eva.types")
    except ModuleNotFoundError as exc:
        raise BenchUnavailable(
            f"модуль {exc.name or 'eva'} не найден — offline-замер делать не на чем "
            "(контракт docs/hybrid.md)"
        ) from exc

    if cfg is None:
        cfg = load_offline_config()
    if prices is None:
        prices = prices_from_config(cfg)
    if system_tokens is None:
        system_tokens = system_prompt_tokens()

    small = StubClient("small", stub_latency_ms)
    big = StubClient("big", stub_latency_ms)
    deps = _stub_deps(small, big)
    picked = pick_questions(questions, n)

    def task(question: str) -> dict[str, Any]:
        return _offline_once(router, types_module, question, cfg, deps, timeout)

    with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as pool:
        # Прогрев идёт через тот же пул: первый вызов тянет ленивые импорты
        # скиллов/классификатора, а первый вызов в потоке — ещё и его запуск.
        # Без этого p99 мерил бы импорт и старт потоков, а не работу роутера.
        warmed = _warm_up(pool, task, picked, warmup, concurrency)
        started = time.perf_counter()
        records = list(pool.map(task, picked))
        wall_s = time.perf_counter() - started

    return summarize(
        records,
        mode="offline",
        n=n,
        concurrency=concurrency,
        wall_s=wall_s,
        prices=prices,
        system_tokens=system_tokens,
        completion_tokens=completion_tokens,
        warmup=warmed,
        p95_budget_ms=p95_budget_ms,
    )


def chat_url(base: str) -> str:
    """Привести базу сервера к ``POST /v1/chat/completions``.

    Принимает ``http://host:port``, ``http://host:port/v1`` и полный путь —
    так стенд можно запускать и «как в runbook», и точным адресом эндпоинта.
    """
    base = str(base or "").strip()
    if not base:
        raise ValueError("пустой URL")
    if not re.match(r"^https?://", base, re.IGNORECASE):
        raise ValueError(f"URL должен начинаться с http:// или https://: {base!r}")
    trimmed = base.rstrip("/")
    if trimmed.endswith("/v1/chat/completions"):
        return trimmed
    if trimmed.endswith("/v1"):
        return trimmed + "/chat/completions"
    return trimmed + "/v1/chat/completions"


def run_http(
    url: str,
    questions: Sequence[str],
    n: int = DEFAULT_N,
    concurrency: int = DEFAULT_CONCURRENCY,
    *,
    token: str = "",
    model: str = DEFAULT_MODEL,
    timeout: float = DEFAULT_TIMEOUT,
    warmup: int = 1,
    prices: Mapping[str, tuple[float, float]] | None = None,
    system_tokens: int | None = None,
    completion_tokens: Mapping[str, int] | None = None,
    p95_budget_ms: float | None = None,
) -> dict[str, Any]:
    """Прогнать N запросов против поднятого сервера по HTTP (без SSE)."""
    endpoint = chat_url(url)
    if prices is None:
        prices = dict(DEFAULT_PRICES)
    if system_tokens is None:
        system_tokens = system_prompt_tokens()

    picked = pick_questions(questions, n)

    def task(question: str) -> dict[str, Any]:
        return _http_once(endpoint, question, token, model, timeout)

    with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as pool:
        # Ошибку прогрева не считаем: если сервер лежит, её покажет прогон.
        warmed = _warm_up(pool, task, picked, warmup, concurrency)
        started = time.perf_counter()
        records = list(pool.map(task, picked))
        wall_s = time.perf_counter() - started

    return summarize(
        records,
        mode="http",
        n=n,
        concurrency=concurrency,
        wall_s=wall_s,
        prices=prices,
        system_tokens=system_tokens,
        completion_tokens=completion_tokens,
        warmup=warmed,
        url=endpoint,
        p95_budget_ms=p95_budget_ms,
    )


# --------------------------------------------------------------------------- #
# печать
# --------------------------------------------------------------------------- #
def _ms(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def _share(count: int, total: int) -> str:
    return f"{count} ({count / total * 100:.1f}%)" if total else f"{count} (—)"


def format_report(metrics: Mapping[str, Any]) -> str:
    """Человекочитаемая таблица по метрикам прогона."""
    mode = metrics.get("mode")
    lines: list[str] = []
    if mode == "offline":
        lines.append("бенчмарк гибрида: offline (заглушки моделей, сеть не используется)")
    else:
        lines.append(f"бенчмарк гибрида: http {metrics.get('url')}")
    lines.append(
        f"запросов {metrics['n']}, параллельно {metrics['concurrency']}, "
        f"прогрев {metrics.get('warmup', 0)}"
    )
    throughput = metrics.get("throughput_rps")
    lines.append(
        f"время {metrics['wall_s']:.3f} с, пропускная способность "
        + (f"{throughput:.1f} запросов/с" if throughput is not None else "—")
    )

    lat = metrics["latency_ms"]
    lines.append(
        f"латентность, мс: p50 {_ms(lat['p50'])} | p95 {_ms(lat['p95'])} | "
        f"p99 {_ms(lat['p99'])}"
    )
    lines.append(
        f"                  min {_ms(lat['min'])} | mean {_ms(lat['mean'])} | "
        f"max {_ms(lat['max'])} | успешных {lat['count']}"
    )

    routes = metrics["routes"]
    shares = metrics["shares"]
    route_parts = [
        f"{title} {_share(routes.get(kind, 0), metrics['ok'])}"
        for kind, title in (("skill", "skill"), ("small", "small"), ("big", "big"))
    ]
    if routes.get("other"):
        route_parts.append(f"other {_share(routes['other'], metrics['ok'])}")
    lines.append("маршруты: " + " | ".join(route_parts))

    lines.append(f"ошибки {metrics['errors']}, таймауты {metrics['timeouts']}")
    samples = [s for s in (metrics.get("error_samples") or []) if s]
    if samples:
        lines.append(f"  первый отказ: {samples[0]}")

    prices = metrics["prices_rub_per_1k"]
    lines.append(
        f"цена: {metrics['cost_per_1000_rub']:.2f} ₽ за 1000 запросов "
        "(оценка по маршрутам и токенам)"
    )
    lines.append(
        "  тариф: большая {:.2f}/{:.2f}, малая {:.2f}/{:.2f} ₽ за 1000 "
        "prompt/completion-токенов".format(
            prices["big"][0],
            prices["big"][1],
            prices["small"][0],
            prices["small"][1],
        )
    )
    reported = metrics.get("reported_cost_per_1000_rub")
    if mode == "http" and reported is not None:
        lines.append(f"  по ответам сервера: {reported:.2f} ₽ за 1000")

    budget = metrics.get("p95_budget_ms")
    if budget is None:
        lines.append(
            "бюджет p95: не задан — в docs/hybrid.md бюджета латентности нет; "
            "задать: --p95-budget-ms N"
        )
    else:
        verdict = "ok" if metrics.get("budget_ok") else "ПРЕВЫШЕН"
        lines.append(
            f"бюджет p95: {_ms(lat['p95'])} мс ≤ {float(budget):.2f} мс — {verdict}"
        )

    if mode == "offline":
        lines.append(
            "offline: цифры — накладные расходы роутера и заглушек, "
            "а не латентность моделей; при --concurrency > 1 в латентность "
            "входит ожидание GIL"
        )
    return "\n".join(lines)


def format_json(metrics: Mapping[str, Any]) -> str:
    """Машинный вывод: те же метрики плюс записи по каждому запросу."""
    return json.dumps(metrics, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"ожидалось целое число, получено {value!r}") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError(f"нужно положительное число, получено {number}")
    return number


def _non_negative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"ожидалось целое число, получено {value!r}") from exc
    if number < 0:
        raise argparse.ArgumentTypeError(f"нужно неотрицательное число, получено {number}")
    return number


def _positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"ожидалось число, получено {value!r}") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError(f"нужно положительное число, получено {number}")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Бенчмарк гибрида: p50/p95/p99, throughput, маршруты и цена за 1000 "
            "запросов. --offline — заглушки без сети, --http URL — живой сервер."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--offline",
        action="store_true",
        help="без сети: eva.router.answer с заглушками моделей",
    )
    mode.add_argument(
        "--http",
        metavar="URL",
        help=f"живой сервер, например {DEFAULT_URL}",
    )
    parser.add_argument(
        "--n", type=_positive_int, default=DEFAULT_N, help=f"число запросов (по умолчанию {DEFAULT_N})"
    )
    parser.add_argument(
        "--concurrency",
        type=_positive_int,
        default=DEFAULT_CONCURRENCY,
        help=f"параллельных запросов (по умолчанию {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument(
        "--timeout",
        type=_positive_float,
        default=DEFAULT_TIMEOUT,
        help=f"таймаут запроса, сек (по умолчанию {DEFAULT_TIMEOUT:g})",
    )
    parser.add_argument(
        "--warmup",
        type=_non_negative_int,
        default=1,
        help=(
            "прогревочных запросов, не меньше числа потоков, в метрики не "
            "входят (по умолчанию 1; 0 — выключить)"
        ),
    )
    parser.add_argument(
        "--questions", default=QUESTIONS, help="JSONL с вопросами (по умолчанию eval/hybrid_golden.jsonl)"
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("EVA_AUTH_TOKEN", ""),
        help="Bearer-токен сервера (по умолчанию из EVA_AUTH_TOKEN)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="имя модели в запросе")
    parser.add_argument("--json", action="store_true", help="машинный вывод вместо таблицы")
    parser.add_argument(
        "--p95-budget-ms",
        type=_positive_float,
        default=None,
        help="бюджет p95, мс (по умолчанию — из docs/hybrid.md, если там есть)",
    )
    parser.add_argument(
        "--stub-latency-ms",
        type=float,
        default=0.0,
        help="offline: искусственная задержка заглушки, мс (по умолчанию 0)",
    )
    parser.add_argument("--big-price-in", type=float, default=None, help="большая: ₽/1k prompt")
    parser.add_argument("--big-price-out", type=float, default=None, help="большая: ₽/1k completion")
    parser.add_argument("--small-price-in", type=float, default=None, help="малая: ₽/1k prompt")
    parser.add_argument("--small-price-out", type=float, default=None, help="малая: ₽/1k completion")
    parser.add_argument(
        "--system-tokens", type=int, default=None, help="токенов системного промпта (по умолчанию — оценка)"
    )
    parser.add_argument("--traceback", action="store_true", help="показать трейсбек при сбое")
    return parser


def _budget_for(args: argparse.Namespace) -> tuple[float | None, bool]:
    """(бюджет p95, взят ли из документации)."""
    if args.p95_budget_ms is not None:
        return float(args.p95_budget_ms), False
    documented = budget_from_docs()
    if documented is not None:
        return documented, True
    return DEFAULT_P95_BUDGET_MS, False


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        questions = load_questions(args.questions)
    except (OSError, ValueError) as exc:
        print(f"набор вопросов не читается: {exc}", file=sys.stderr)
        return 2

    budget, budget_from_file = _budget_for(args)
    if budget_from_file:
        print(
            f"бюджет p95 из docs/hybrid.md: {budget:.2f} мс",
            file=sys.stderr,
        )

    kwargs = {
        "timeout": args.timeout,
        "warmup": args.warmup,
        "system_tokens": args.system_tokens,
        "p95_budget_ms": budget,
    }
    try:
        if args.offline:
            cfg = load_offline_config()
            prices = resolve_prices(prices_from_config(cfg), args)
            metrics = run_offline(
                questions,
                args.n,
                args.concurrency,
                stub_latency_ms=args.stub_latency_ms,
                cfg=cfg,
                prices=prices,
                **kwargs,
            )
        else:
            prices = resolve_prices(dict(DEFAULT_PRICES), args)
            metrics = run_http(
                args.http,
                questions,
                args.n,
                args.concurrency,
                token=args.token or "",
                model=args.model,
                prices=prices,
                **kwargs,
            )
    except BenchUnavailable as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"ошибка аргументов: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 — CLI не должен пугать трейсбеком
        print(f"ошибка замера: {type(exc).__name__}: {exc}", file=sys.stderr)
        if args.traceback:
            traceback.print_exc()
        return 1

    print(format_json(metrics) if args.json else format_report(metrics))

    if metrics["ok"] == 0:
        print(
            "ни один запрос не удался: смотрите ошибки выше"
            + ("" if args.json else " и host/port сервера"),
            file=sys.stderr,
        )
        return 1
    if metrics.get("budget_ok") is False:
        print(
            f"бюджет p95 превышен: {metrics['latency_ms']['p95']:.2f} мс > "
            f"{float(metrics['p95_budget_ms']):.2f} мс",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
