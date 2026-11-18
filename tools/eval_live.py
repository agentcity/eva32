#!/usr/bin/env python3
"""Живой замер гибрида: 10 каталожных и 10 общих вопросов, настоящие деньги.

В отличие от ``tools/eval_hybrid.py`` здесь **нет заглушек**: каталожные
вопросы идут через скиллы (0 ₽), общие — через большую модель по ключу из
``.env`` (``TEACHER_BASE`` / ``TEACHER_KEY`` / ``TEACHER_MODEL``, polza.ai).
Поэтому сеть разрешена только при явном запуске без ``--dry-run``:

    python3 tools/eval_live.py --dry-run   # формат набора, сеть запрещена
    python3 tools/eval_live.py             # живой прогон, реальные деньги
    python3 tools/eval_live.py --json      # те же метрики машинно

Набор — ``eval/live_golden.jsonl``, одна JSON-строка на вопрос::

    {"id": "cat-01", "kind": "catalog"|"general",
     "question": "...", "expect_route": "skill"|"big",
     "facts": ["990", "2 490"]}

``facts`` — эталонные факты: попаданием считается **хотя бы один** из них.
Сравнение не зависит от регистра, ``ё`` и пробелов, а число не должно быть
частью другого числа (``990`` не совпадёт с ``1990`` и ``9900``).

Малая модель в замере не запускается: маршрут ``small``, если его выберет
классификатор, помечается «малая недоступна», ответ даёт большая через
фолбэк. Поэтому замер всегда честно показывает, сколько запросов **хотело**
в малую, и сколько из них ушло в платную.

Считается:

* попадание факта (по эталону);
* маршрут (решение классификатора и фактический ответивший);
* потраченные рубли — ``cost_rub`` из ответа API, иначе оценка по тарифу;
* латентность каждого ответа и сводка по маршрутам.

Ключ никогда не печатается. В отчёте только хост провайдера и имя модели.
Код возврата: 0 — прогон состоялся; 1 — ошибка/порог; 2 — плохой набор.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import sys
import traceback
import urllib.parse
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

GOLDEN = os.path.join(ROOT, "eval", "live_golden.jsonl")
ENV_FILE = os.path.join(ROOT, ".env")
SYSTEM_PROMPT = os.path.join(ROOT, "prompts", "system.ru.md")

KINDS = ("catalog", "general")
ROUTE_KINDS = ("skill", "small", "big")
EXPECTED_ROUTE = {"catalog": "skill", "general": "big"}
EXPECTED_COUNTS = {"catalog": 10, "general": 10}
EXPECTED_TOTAL = sum(EXPECTED_COUNTS.values())

#: Модель учителя по умолчанию — та же, что в ``distill/teacher.py``.
#: ``.env`` может задать ``TEACHER_MODEL`` или ``EVA_BIG_MODEL``.
DEFAULT_MODEL = "deepseek-v4.1-flash"

#: Ориентир для предупреждения. Это не лимит: бюджет на запрос задаёт
#: ``EVA_MAX_COST_RUB`` (по умолчанию 5 ₽), а суммарный расход считается по факту.
EXPECTED_MAX_RUN_RUB = 5.0

SMALL_UNAVAILABLE = "малая недоступна (ollama в живом замере не запускается)"


class EvalError(RuntimeError):
    """Живой прогон невозможен или прерван."""


class SmallUnavailable(RuntimeError):
    """Клиент малой модели: сеть не трогаем, честно сообщаем о недоступности."""


# --------------------------------------------------------------------------- #
# сравнение с эталонными фактами
# --------------------------------------------------------------------------- #
def normalize(text: Any) -> str:
    """Нижний регистр, ``ё``→``е``, без пробелов (включая неразрывные)."""
    return "".join(str(text or "").lower().replace("ё", "е").split())


def fact_in(text: Any, fact: Any) -> bool:
    """Есть ли факт в ответе; число — только как отдельное число.

    ``990`` не должно находиться внутри ``1990``/``9900``, ``9:00`` — внутри
    ``19:00``. Для нечисловых фактов работает обычная подстрока.
    """
    haystack = normalize(text)
    needle = normalize(fact)
    if not haystack or not needle:
        return False
    pattern = re.escape(needle)
    if needle[0].isdigit():
        pattern = r"(?<!\d)" + pattern
    if needle[-1].isdigit():
        pattern = pattern + r"(?!\d)"
    return re.search(pattern, haystack) is not None


def fact_hit(text: Any, facts: Iterable[str]) -> bool:
    """Попадание факта: достаточно любого эталона из списка."""
    return any(fact_in(text, fact) for fact in facts if str(fact or "").strip())


def facts_echoed_in_question(question: str, facts: Iterable[str]) -> list[str]:
    """Эталоны, которые уже есть в вопросе: такой «факт» ничего не проверяет."""
    return [fact for fact in facts if fact and fact_in(question, fact)]


# --------------------------------------------------------------------------- #
# набор
# --------------------------------------------------------------------------- #
def load_golden(path: str = GOLDEN, *, strict_counts: bool = True) -> list[dict]:
    """Прочитать и проверить набор. Ошибки формата — с номером строки."""
    rows: list[dict] = []
    seen_ids: dict[str, int] = {}
    seen_questions: dict[str, int] = {}
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: невалидный JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{lineno}: ожидался объект")
            question = row.get("question")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"{path}:{lineno}: пустой question")
            question = question.strip()
            kind = row.get("kind")
            if kind not in KINDS:
                raise ValueError(
                    f"{path}:{lineno}: kind={kind!r}, ожидался один из {', '.join(KINDS)}"
                )
            expect_route = row.get("expect_route")
            if expect_route not in ROUTE_KINDS:
                raise ValueError(
                    f"{path}:{lineno}: expect_route={expect_route!r}, "
                    f"ожидался один из {', '.join(ROUTE_KINDS)}"
                )
            required_route = EXPECTED_ROUTE[kind]
            if expect_route != required_route:
                raise ValueError(
                    f"{path}:{lineno}: для kind={kind} ожидается "
                    f"expect_route={required_route!r}, получено {expect_route!r}"
                )
            facts = row.get("facts")
            if not isinstance(facts, list) or not facts:
                raise ValueError(f"{path}:{lineno}: facts должен быть непустым списком")
            clean_facts = [str(item).strip() for item in facts if str(item or "").strip()]
            if len(clean_facts) != len(facts):
                raise ValueError(f"{path}:{lineno}: пустой эталонный факт в facts")
            row_id = str(row.get("id") or "").strip()
            if not row_id:
                row_id = f"line-{lineno}"
            if row_id in seen_ids:
                raise ValueError(
                    f"{path}:{lineno}: повтор id {row_id!r} (уже в строке {seen_ids[row_id]})"
                )
            if question in seen_questions:
                raise ValueError(
                    f"{path}:{lineno}: повтор вопроса (уже в строке "
                    f"{seen_questions[question]})"
                )
            seen_ids[row_id] = lineno
            seen_questions[question] = lineno
            rows.append(
                {
                    "id": row_id,
                    "kind": kind,
                    "question": question,
                    "expect_route": expect_route,
                    "facts": clean_facts,
                }
            )
    if not rows:
        raise ValueError(f"{path}: пустой набор")

    counts = Counter(row["kind"] for row in rows)
    if strict_counts:
        if len(rows) != EXPECTED_TOTAL:
            raise ValueError(
                f"{path}: ожидалось {EXPECTED_TOTAL} вопросов "
                f"({EXPECTED_COUNTS['catalog']} каталожных + "
                f"{EXPECTED_COUNTS['general']} общих), получено {len(rows)}"
            )
        for kind in KINDS:
            if counts[kind] != EXPECTED_COUNTS[kind]:
                raise ValueError(
                    f"{path}: {kind}: ожидалось {EXPECTED_COUNTS[kind]}, "
                    f"получено {counts[kind]}"
                )
    return rows


def set_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row.get("kind") or "") for row in rows)
    return {kind: counts.get(kind, 0) for kind in KINDS}


# --------------------------------------------------------------------------- #
# конфиг: .env → eva.config
# --------------------------------------------------------------------------- #
def read_env_file(path: str = ENV_FILE) -> dict[str, str]:
    """Прочитать ``.env`` как словарь. Ключи не печатаются и не логируются."""
    data: dict[str, str] = {}
    try:
        fh = open(path, encoding="utf-8")
    except OSError:
        return data
    with fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export ") :]
            key, value = line.split("=", 1)
            key = key.strip()
            if not key:
                continue
            data[key] = value.strip().strip("\"'")
    return data


def build_config(env: Mapping[str, Any] | None = None, dotenv_path: str | None = ENV_FILE) -> Any:
    """Собрать ``eva.config.Config``, связав ``TEACHER_*`` с большой моделью.

    Приоритет: явное окружение/аргументы → ``.env`` → умолчания ``eva.config``.
    ``TEACHER_MODEL`` необязателен: без него берётся ``DEFAULT_MODEL``.
    """
    from eva.config import Config

    if env is None:
        source: dict[str, str] = {str(k): str(v) for k, v in os.environ.items()}
    else:
        source = {str(k): str(v) for k, v in env.items()}
    if dotenv_path:
        for key, value in read_env_file(dotenv_path).items():
            source.setdefault(key, value)

    if not source.get("EVA_BIG_BASE"):
        source["EVA_BIG_BASE"] = source.get("TEACHER_BASE", "")
    if not source.get("EVA_BIG_KEY"):
        source["EVA_BIG_KEY"] = source.get("TEACHER_KEY", "")
    if not source.get("EVA_BIG_MODEL"):
        source["EVA_BIG_MODEL"] = source.get("TEACHER_MODEL") or DEFAULT_MODEL

    # Разумные умолчания живого замера: один повтор и запас по времени.
    source.setdefault("EVA_BIG_TIMEOUT", "90")
    source.setdefault("EVA_BIG_RETRIES", "1")
    source.setdefault("EVA_MAX_TOKENS", "512")
    return Config.from_env(source)


def config_summary(cfg: Any) -> dict[str, str]:
    """Только безопасные поля конфига: провайдер и модель, без ключа."""
    base = str(getattr(cfg, "big_base", "") or "")
    parsed = urllib.parse.urlparse(base)
    return {
        "provider": parsed.netloc or base,
        "model": str(getattr(cfg, "big_model", "") or ""),
    }


def validate_live_config(cfg: Any) -> dict[str, str]:
    """Проверить, что живой прогон вообще возможен. Ключ не возвращается."""
    base = str(getattr(cfg, "big_base", "") or "").strip()
    model = str(getattr(cfg, "big_model", "") or "").strip()
    key = str(getattr(cfg, "big_key", "") or "").strip()
    if not base:
        raise EvalError(
            "не задан адрес большой модели: TEACHER_BASE или EVA_BIG_BASE в .env"
        )
    if not model:
        raise EvalError("не задано имя большой модели: TEACHER_MODEL или EVA_BIG_MODEL")
    if not key:
        raise EvalError("не задан ключ: TEACHER_KEY или EVA_BIG_KEY в .env")
    parsed = urllib.parse.urlparse(base)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise EvalError(f"TEACHER_BASE/EVA_BIG_BASE: ожидался http(s)://хост, получено {base!r}")
    return config_summary(cfg)


# --------------------------------------------------------------------------- #
# клиенты: большая — настоящая, малая — честно недоступна
# --------------------------------------------------------------------------- #
def _as_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _result_field(result: Any, name: str) -> Any:
    """Поле результата клиента: ``ClientResult`` или обычный словарь."""
    if isinstance(result, Mapping):
        return result.get(name)
    return getattr(result, name, None)


def _raw_cost(raw: Any) -> float | None:
    """``cost_rub`` из сырого ответа провайдера, если он его отдал."""
    data = _as_mapping(raw)
    for source in (_as_mapping(data.get("usage")), data):
        value = source.get("cost_rub")
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return None


class MeteredBigClient:
    """Обёртка вокруг клиента большой модели: запоминает usage/cost каждого вызова."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages: Any, max_tokens: int = 512) -> Any:
        result = self.inner.chat(messages, max_tokens=max_tokens)
        usage = _as_mapping(_result_field(result, "usage"))
        raw = _as_mapping(_result_field(result, "raw"))
        self.calls.append(
            {
                "usage": usage,
                "api_cost": _raw_cost(raw),
                "raw": raw,
            }
        )
        return result


class SmallUnavailableClient:
    """Клиент малой модели без сети: сразу говорит, что она недоступна."""

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages: Any, max_tokens: int = 512) -> Any:
        self.calls += 1
        raise SmallUnavailable(SMALL_UNAVAILABLE)


def make_big_client(cfg: Any) -> MeteredBigClient:
    """Настоящий OpenAI-совместимый клиент большой модели (polza.ai)."""
    from eva.clients.openai_compat import OpenAIClient

    inner = OpenAIClient(
        cfg.big_base,
        cfg.big_key,
        cfg.big_model,
        timeout=float(cfg.big_timeout),
        retries=int(cfg.big_retries),
        price_prompt_rub=float(cfg.big_price_prompt_rub_per_1k),
        price_completion_rub=float(cfg.big_price_completion_rub_per_1k),
    )
    return MeteredBigClient(inner)


# --------------------------------------------------------------------------- #
# живой прогон
# --------------------------------------------------------------------------- #
def _call_count(client: Any) -> int:
    calls = getattr(client, "calls", None)
    try:
        return len(calls or [])
    except TypeError:
        return 0


def _new_calls(client: Any, before: int) -> list[dict[str, Any]]:
    calls = getattr(client, "calls", None) or []
    try:
        return list(calls[before:])
    except (TypeError, KeyError):
        return []


def _read_text(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return ""


def run_live(
    rows: list[dict],
    cfg: Any,
    *,
    big_client: Any | None = None,
    system_text: str | None = None,
    max_tokens: int | None = None,
    progress: Any | None = None,
    max_model_errors: int = 3,
) -> dict:
    """Прогнать набор через настоящий ``eva.router.answer``.

    ``big_client`` можно подменить в тестах — тогда сети нет. По умолчанию
    создаётся клиент polza.ai по ключу из конфига. Малая модель всегда
    помечена недоступной.
    """
    from eva import router
    from eva.types import Message, Request

    if max_tokens is not None and int(max_tokens) != int(cfg.max_tokens):
        cfg = dataclasses.replace(cfg, max_tokens=int(max_tokens))
    big = big_client if big_client is not None else make_big_client(cfg)
    small = SmallUnavailableClient()
    deps = {"big": big, "small": small}
    if system_text is None:
        system_text = _read_text(SYSTEM_PROMPT)

    records: list[dict[str, Any]] = []
    big_ok = 0
    model_errors = 0
    first_error: str | None = None
    spent = 0.0

    for index, row in enumerate(rows, 1):
        messages = []
        if system_text and system_text.strip():
            messages.append(Message(role="system", content=system_text))
        messages.append(Message(role="user", content=row["question"]))
        request = Request(
            messages=messages,
            meta={"user_id": "eval-live", "channel": "eval-live"},
        )
        planned = ""
        try:
            planned = str(router.decide(request, cfg).kind or "")
        except Exception:  # noqa: BLE001 — предпросмотр не должен ронять прогон
            planned = ""

        before = _call_count(big)
        record: dict[str, Any] = {
            "id": row.get("id") or f"row-{index}",
            "kind": row["kind"],
            "question": row["question"],
            "expect_route": row["expect_route"],
            "facts": list(row["facts"]),
            "planned_route": planned,
            "route": "",
            "reason": "",
            "skill": None,
            "text": "",
            "usage": {},
            "cost_rub": 0.0,
            "cost_source": "none",
            "latency_ms": 0,
            "fact_ok": False,
            "route_ok": False,
            "small_unavailable": planned == "small",
            "error": None,
        }
        try:
            answer = router.answer(request, cfg, deps)
            route = getattr(answer, "route", None)
            record["route"] = str(getattr(route, "kind", "") or "").lower()
            record["reason"] = str(getattr(route, "reason", "") or "")
            record["skill"] = getattr(answer, "skill", None)
            record["text"] = str(getattr(answer, "text", "") or "")
            record["usage"] = _as_mapping(getattr(answer, "usage", None))
            record["cost_rub"] = max(0.0, float(getattr(answer, "cost_rub", 0.0) or 0.0))
            record["latency_ms"] = max(0, int(getattr(answer, "latency_ms", 0) or 0))
        except Exception as exc:  # noqa: BLE001 — одна строка не роняет отчёт
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["route"] = planned

        calls = _new_calls(big, before)
        if calls:
            last = calls[-1]
            api_cost = last.get("api_cost")
            if api_cost is not None:
                record["cost_source"] = "api"
            elif record["cost_rub"] > 0:
                record["cost_source"] = "tariff"
            if not record["cost_rub"]:
                usage_cost = _as_mapping(last.get("usage")).get("cost_rub")
                try:
                    record["cost_rub"] = max(0.0, float(usage_cost or 0.0))
                except (TypeError, ValueError):
                    record["cost_rub"] = 0.0
        elif record["route"] == "skill":
            record["cost_source"] = "skill"

        if record["route"] == "small":
            record["small_unavailable"] = True
        record["route_ok"] = record["route"] == record["expect_route"]
        record["fact_ok"] = bool(record["text"].strip()) and fact_hit(
            record["text"], record["facts"]
        )
        spent += record["cost_rub"]

        if record["error"]:
            model_errors += 1
            first_error = first_error or record["error"]
            if big_ok == 0 and model_errors >= max_model_errors:
                raise EvalError(
                    f"живой прогон остановлен: {model_errors} ошибок подряд, "
                    f"потрачено {spent:.4f} ₽; первая ошибка: {first_error}"
                )
        elif record["route"] == "big":
            big_ok += 1
            model_errors = 0

        records.append(record)
        if progress is not None:
            progress(index, len(rows), record)

    metrics = summarize(records)
    metrics["small_calls"] = int(getattr(small, "calls", 0))
    return metrics


def summarize(records: list[dict], *, provider: str = "", model: str = "") -> dict:
    """Посчитать метрики по записям (чистая функция, без сети)."""
    total = len(records)
    by_kind: dict[str, dict[str, int]] = {}
    for kind in KINDS:
        subset = [record for record in records if record.get("kind") == kind]
        count = len(subset)
        by_kind[kind] = {
            "total": count,
            "route_ok": sum(1 for record in subset if record.get("route_ok")),
            "fact_ok": sum(1 for record in subset if record.get("fact_ok")),
            "both": sum(
                1 for record in subset if record.get("route_ok") and record.get("fact_ok")
            ),
        }

    route_counts = Counter(str(record.get("route") or "") for record in records)
    costs = [float(record.get("cost_rub") or 0.0) for record in records]
    api_cost = sum(
        float(record.get("cost_rub") or 0.0)
        for record in records
        if record.get("cost_source") == "api"
    )
    tariff_cost = sum(
        float(record.get("cost_rub") or 0.0)
        for record in records
        if record.get("cost_source") == "tariff"
    )
    api_calls = sum(1 for record in records if record.get("cost_source") == "api")
    billed_calls = sum(
        1 for record in records if record.get("cost_source") in ("api", "tariff")
    )
    latencies = [int(record.get("latency_ms") or 0) for record in records]
    latency_by_route: dict[str, dict[str, float]] = {}
    for kind in ROUTE_KINDS:
        values = [
            int(record.get("latency_ms") or 0)
            for record in records
            if record.get("route") == kind
        ]
        if values:
            latency_by_route[kind] = {
                "count": len(values),
                "avg": sum(values) / len(values),
                "max": max(values),
            }

    return {
        "total": total,
        "by_kind": by_kind,
        "route_ok": sum(1 for record in records if record.get("route_ok")),
        "fact_ok": sum(1 for record in records if record.get("fact_ok")),
        "both_ok": sum(
            1 for record in records if record.get("route_ok") and record.get("fact_ok")
        ),
        "route_counts": {kind: route_counts.get(kind, 0) for kind in ROUTE_KINDS},
        "route_other": sum(
            count for kind, count in route_counts.items() if kind not in ROUTE_KINDS
        ),
        "errors": sum(1 for record in records if record.get("error")),
        "error_rows": [
            {"question": record.get("question"), "error": record.get("error")}
            for record in records
            if record.get("error")
        ],
        "cost_rub": round(sum(costs), 6),
        "api_cost_rub": round(api_cost, 6),
        "tariff_cost_rub": round(tariff_cost, 6),
        "api_calls": api_calls,
        "billed_calls": billed_calls,
        "latency_ms_avg": (sum(latencies) / total) if total else 0.0,
        "latency_ms_median": _median(latencies),
        "latency_ms_max": max(latencies) if latencies else 0,
        "latency_by_route": latency_by_route,
        "small_unavailable": sum(
            1 for record in records if record.get("small_unavailable")
        ),
        "provider": provider,
        "model": model,
        "records": records,
    }


def _median(values: list[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0


# --------------------------------------------------------------------------- #
# предпросмотр маршрутов офлайн (без моделей и сети)
# --------------------------------------------------------------------------- #
def preview_routes(rows: list[dict]) -> tuple[list[dict] | None, str]:
    """Решения классификатора без вызова моделей. Возвращает (строки, ошибка)."""
    try:
        from eva import router
        from eva.config import Config
        from eva.types import Message, Request
    except Exception as exc:  # noqa: BLE001 — dry-run не должен падать из-за eva
        return None, f"модули eva недоступны: {type(exc).__name__}: {exc}"
    cfg = Config.from_env({})
    preview: list[dict] = []
    for row in rows:
        request = Request(
            messages=[Message(role="user", content=row["question"])],
            meta={"user_id": "eval-live-dry", "channel": "eval-live-dry"},
        )
        try:
            kind = str(router.decide(request, cfg).kind or "")
        except Exception as exc:  # noqa: BLE001
            kind = ""
            error = f"{type(exc).__name__}: {exc}"
        else:
            error = ""
        preview.append(
            {
                "id": row.get("id"),
                "kind": row["kind"],
                "question": row["question"],
                "expect_route": row["expect_route"],
                "planned_route": kind,
                "error": error,
            }
        )
    return preview, ""


# --------------------------------------------------------------------------- #
# печать
# --------------------------------------------------------------------------- #
def _share(count: int, total: int) -> str:
    return f"{count}/{total}" + (f" = {count / total * 100:.0f}%" if total else "")


def format_dry_run(
    rows: list[dict],
    path: str = GOLDEN,
    *,
    preview: list[dict] | None = None,
    preview_error: str = "",
    limit: int = 0,
) -> str:
    counts = set_counts(rows)
    lines = [
        "dry-run: сеть не используется, модели не вызываются, деньги не тратятся.",
        f"набор: {path} — {len(rows)} вопросов "
        f"(каталожных {counts['catalog']}, общих {counts['general']})",
        "формат: ок (question, kind, expect_route, facts; каталожные → skill, общие → big)",
    ]
    if limit:
        lines.append(f"limit: в живой прогон пойдут первые {limit} вопросов")
    echoed = [
        (row["id"], fact)
        for row in rows
        for fact in facts_echoed_in_question(row["question"], row["facts"])
    ]
    if echoed:
        lines.append("внимание: эталон встречается в самом вопросе (проверка вырождена):")
        lines.extend(f"  - {row_id}: {fact}" for row_id, fact in echoed)
    else:
        lines.append(
            "эталоны: ни один факт не встречается в тексте вопроса — проверка не вырождена"
        )
    lines.append(
        "малая модель: не проверяется и не запускается; "
        "если выпадет small — в отчёте будет «малая недоступна»"
    )
    lines.append("")
    if preview is None:
        lines.append("предпросмотр маршрутов: пропущен")
        if preview_error:
            lines.append(f"  причина: {preview_error}")
    elif preview_error:
        lines.append(f"предпросмотр маршрутов: недоступен — {preview_error}")
    else:
        total = len(preview)
        route_ok = sum(1 for row in preview if row["planned_route"] == row["expect_route"])
        by_kind = {
            kind: sum(1 for row in preview if row["kind"] == kind)
            for kind in KINDS
        }
        by_kind_ok = {
            kind: sum(
                1
                for row in preview
                if row["kind"] == kind and row["planned_route"] == row["expect_route"]
            )
            for kind in KINDS
        }
        lines.append("предпросмотр маршрутов (офлайн, без моделей и сети):")
        lines.append(
            f"  совпало {route_ok}/{total}; "
            f"каталог {by_kind_ok['catalog']}/{by_kind['catalog']}, "
            f"общие {by_kind_ok['general']}/{by_kind['general']}"
        )
        for row in preview:
            if row["planned_route"] == row["expect_route"]:
                continue
            got = row["planned_route"] or row["error"] or "error"
            lines.append(
                f"  - {row['id']}: ожидался {row['expect_route']}, "
                f"классификатор выбрал {got} — «{row['question']}»"
            )
    lines.append("")
    lines.append("живой прогон (только вручную, без --dry-run):")
    lines.append("  python3 tools/eval_live.py")
    lines.append(
        f"  потратит реальные деньги: ожидаемо <{EXPECTED_MAX_RUN_RUB:.0f} ₽ "
        f"за {EXPECTED_TOTAL} вопросов; ключ берётся из .env (TEACHER_KEY) и не печатается"
    )
    return "\n".join(lines)


def _truncate(text: Any, limit: int = 60) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"


def format_report(metrics: Mapping[str, Any], *, verbose: bool = False) -> str:
    total = int(metrics.get("total") or 0)
    by_kind = metrics.get("by_kind") or {}
    catalog = by_kind.get("catalog") or {}
    general = by_kind.get("general") or {}
    provider = metrics.get("provider") or "—"
    model = metrics.get("model") or "—"
    lines: list[str] = []
    lines.append(
        f"живой замер гибрида: {total} вопросов "
        f"(каталог {int(catalog.get('total') or 0)}, "
        f"общие {int(general.get('total') or 0)})"
    )
    lines.append(f"провайдер: {provider} · модель {model} · ключ не печатается")
    lines.append(
        "малая модель: не запускалась; при выборе small — пометка «малая недоступна», "
        "отвечает big"
    )
    lines.append(
        "факт: достаточно одного эталона; регистр, «ё» и пробелы не важны, "
        "число — только целиком (990 ≠ 1990)"
    )
    lines.append("")
    lines.append(" №  маршрут  факт   ₽        мс   вопрос")
    for index, record in enumerate(metrics.get("records") or [], 1):
        route = str(record.get("route") or "—")
        fact = "✓" if record.get("fact_ok") else "✗"
        cost = float(record.get("cost_rub") or 0.0)
        latency = int(record.get("latency_ms") or 0)
        note = ""
        if record.get("small_unavailable"):
            note = "  ← малая недоступна"
        if record.get("error"):
            note += "  ← ошибка"
        lines.append(
            f"{index:>2}. {route:<7} {fact:<4} {cost:>7.4f}  {latency:>5}   "
            f"{_truncate(record.get('question'), 54)}{note}"
        )
    lines.append("")

    misses_fact = [
        record
        for record in (metrics.get("records") or [])
        if not record.get("fact_ok") and not record.get("error")
    ]
    misses_route = [
        record
        for record in (metrics.get("records") or [])
        if not record.get("route_ok") and not record.get("error")
    ]
    errors = [record for record in (metrics.get("records") or []) if record.get("error")]
    if misses_fact:
        lines.append("промахи факта:")
        for record in misses_fact:
            lines.append(
                f"  - «{_truncate(record.get('question'), 70)}»: "
                f"ждали {record.get('facts')}; ответ: "
                f"{_truncate(record.get('text'), 180) or '(пусто)'}"
            )
    if misses_route:
        lines.append("промахи маршрута:")
        for record in misses_route:
            reason = record.get("reason") or ""
            lines.append(
                f"  - «{_truncate(record.get('question'), 70)}»: "
                f"ожидался {record.get('expect_route')}, ответил "
                f"{record.get('route') or '—'}"
                + (f" — {_truncate(reason, 110)}" if reason else "")
            )
    if errors:
        lines.append("ошибки:")
        for record in errors:
            lines.append(
                f"  - «{_truncate(record.get('question'), 70)}»: {record.get('error')}"
            )
    if not (misses_fact or misses_route or errors):
        lines.append("промахов нет: маршрут и факт совпали на всех вопросах.")
    if verbose:
        lines.append("")
        lines.append("ответы целиком:")
        for index, record in enumerate(metrics.get("records") or [], 1):
            lines.append(f"{index}. {record.get('question')}")
            lines.append(f"   {record.get('text') or '(пусто)'}")
    lines.append("")

    lines.append("итог:")
    lines.append(
        f"  маршрут совпал:  {_share(int(metrics.get('route_ok') or 0), total)}"
        f"  (каталог {_share(int(catalog.get('route_ok') or 0), int(catalog.get('total') or 0))},"
        f" общие {_share(int(general.get('route_ok') or 0), int(general.get('total') or 0))})"
    )
    lines.append(
        f"  попадание факта: {_share(int(metrics.get('fact_ok') or 0), total)}"
        f"  (каталог {_share(int(catalog.get('fact_ok') or 0), int(catalog.get('total') or 0))},"
        f" общие {_share(int(general.get('fact_ok') or 0), int(general.get('total') or 0))})"
    )
    lines.append(
        f"  и маршрут, и факт: {_share(int(metrics.get('both_ok') or 0), total)}"
    )
    cost = float(metrics.get("cost_rub") or 0.0)
    api_cost = float(metrics.get("api_cost_rub") or 0.0)
    tariff_cost = float(metrics.get("tariff_cost_rub") or 0.0)
    api_calls = int(metrics.get("api_calls") or 0)
    billed = int(metrics.get("billed_calls") or 0)
    lines.append(
        f"  потрачено: {cost:.4f} ₽  "
        f"(cost_rub из ответов API: {api_cost:.4f} ₽, {api_calls}/{billed} платных вызовов; "
        f"оценка по тарифу: {tariff_cost:.4f} ₽)"
    )
    lines.append(
        f"  латентность: средняя {float(metrics.get('latency_ms_avg') or 0):.0f} мс, "
        f"медиана {float(metrics.get('latency_ms_median') or 0):.0f} мс, "
        f"максимум {int(metrics.get('latency_ms_max') or 0)} мс"
    )
    for kind in ROUTE_KINDS:
        stats = (metrics.get("latency_by_route") or {}).get(kind)
        if not stats:
            continue
        lines.append(
            f"  {kind:<7} {int(stats['count'])} ответов, "
            f"средняя {stats['avg']:.0f} мс, максимум {int(stats['max'])} мс"
        )
    small = int(metrics.get("small_unavailable") or 0)
    if small:
        lines.append(f"  small выпадал: {small} — помечено «малая недоступна»")
    else:
        lines.append("  small не выпадал: малая модель в замере не участвовала")
    errors_count = int(metrics.get("errors") or 0)
    lines.append(f"  ошибок: {errors_count}")
    lines.append("")
    lines.append(
        "деньги списаны провайдером по факту; сумма выше — из cost_rub каждого ответа."
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _progress(index: int, total: int, record: Mapping[str, Any]) -> None:
    route = str(record.get("route") or "err")
    mark = "✓" if record.get("fact_ok") else "✗"
    note = ""
    if record.get("small_unavailable"):
        note = " (малая недоступна)"
    print(
        f"[{index:>2}/{total}] {route:<5} {mark} "
        f"{float(record.get('cost_rub') or 0):>7.4f} ₽ "
        f"{int(record.get('latency_ms') or 0):>5} мс  "
        f"{_truncate(record.get('question'), 48)}{note}",
        file=sys.stderr,
        flush=True,
    )


def _live_banner() -> str:
    return "\n".join(
        [
            "=" * 74,
            "ВНИМАНИЕ: ЖИВОЙ ПРОГОН — сеть разрешена, тратятся реальные деньги.",
            f"Ожидаемо: <{EXPECTED_MAX_RUN_RUB:.0f} ₽ за {EXPECTED_TOTAL} вопросов "
            "(ориентир 1–2 ₽).",
            "Ключ читается из .env (TEACHER_KEY) и никогда не печатается.",
            "Малая модель (ollama) не запускается: маршрут small помечается "
            "«малая недоступна».",
            "=" * 74,
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Живой замер гибрида: каталог через скиллы, общие вопросы через big "
            "(polza.ai). --dry-run не ходит в сеть."
        )
    )
    parser.add_argument("--golden", default=GOLDEN, help="путь к live_golden.jsonl")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="только проверить формат набора; сеть и модели не используются",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="взять первые N вопросов (для отладки)"
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="не требовать ровно 20 вопросов (10 каталожных + 10 общих)",
    )
    parser.add_argument("--json", action="store_true", help="метрики в JSON")
    parser.add_argument("--quiet", action="store_true", help="только итоговая строка")
    parser.add_argument("--verbose", action="store_true", help="показать ответы целиком")
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="в dry-run не показывать офлайн-предпросмотр маршрутов",
    )
    parser.add_argument(
        "--no-system",
        action="store_true",
        help="не отправлять prompts/system.ru.md (для сравнения)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None, help="лимит completion-токенов на ответ"
    )
    parser.add_argument("--base", default=None, help="переопределить адрес большой модели")
    parser.add_argument("--model", default=None, help="переопределить имя большой модели")
    parser.add_argument("--timeout", type=float, default=None, help="таймаут вызова, сек")
    parser.add_argument(
        "--max-model-errors",
        type=int,
        default=3,
        help="остановиться после N ошибок больших моделей, если успехов ещё не было",
    )
    parser.add_argument(
        "--min-fact", type=float, default=None, help="порог доли попаданий факта (0..1)"
    )
    parser.add_argument("--traceback", action="store_true", help="показать трейсбек при сбое")
    args = parser.parse_args(argv)

    try:
        rows = load_golden(args.golden, strict_counts=not args.allow_partial)
    except OSError as exc:
        print(f"не прочитать набор {args.golden}: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"ошибка набора: {exc}", file=sys.stderr)
        return 2

    selected = rows[: args.limit] if args.limit and args.limit > 0 else rows

    if args.dry_run:
        preview: list[dict] | None = None
        preview_error = ""
        if not args.no_preview:
            preview, preview_error = preview_routes(selected)
        print(
            format_dry_run(
                rows,
                args.golden,
                preview=preview,
                preview_error=preview_error,
                limit=len(selected) if len(selected) != len(rows) else 0,
            )
        )
        return 0

    try:
        cfg = build_config()
        if args.base:
            cfg = dataclasses.replace(cfg, big_base=args.base)
        if args.model:
            cfg = dataclasses.replace(cfg, big_model=args.model)
        if args.timeout is not None:
            cfg = dataclasses.replace(cfg, big_timeout=float(args.timeout))
        info = validate_live_config(cfg)
        print(_live_banner(), flush=True)
        print(
            f"провайдер: {info['provider']} · модель {info['model']} · "
            f"вопросов: {len(selected)}",
            flush=True,
        )
        print("", flush=True)
        metrics = run_live(
            selected,
            cfg,
            system_text=None if not args.no_system else "",
            max_tokens=args.max_tokens,
            progress=_progress,
            max_model_errors=max(1, int(args.max_model_errors)),
        )
    except EvalError as exc:
        print(f"живой прогон не выполнен: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 — CLI не должен пугать трейсбеком
        print(f"ошибка живого прогона: {type(exc).__name__}: {exc}", file=sys.stderr)
        if args.traceback:
            traceback.print_exc()
        return 1

    metrics["provider"] = info["provider"]
    metrics["model"] = info["model"]
    printable = dict(metrics)
    if args.json:
        print(json.dumps(printable, ensure_ascii=False, indent=2, default=str))
    elif args.quiet:
        total = metrics["total"]
        avg = float(metrics.get("latency_ms_avg") or 0)
        print(
            f"{metrics['route_ok']}/{total} маршрут, "
            f"{metrics['fact_ok']}/{total} факт, "
            f"{float(metrics['cost_rub']):.4f} ₽, "
            f"средняя {avg:.0f} мс"
        )
    else:
        print("")
        print(format_report(metrics, verbose=args.verbose))

    failed = False
    if metrics.get("errors"):
        print(
            f"в прогоне {metrics['errors']} ошибок — см. отчёт",
            file=sys.stderr,
        )
        failed = True
    if args.min_fact is not None and metrics["total"]:
        rate = metrics["fact_ok"] / metrics["total"]
        if rate < args.min_fact:
            print(
                f"порог попадания факта не пройден: {rate:.3f} < {args.min_fact:.3f}",
                file=sys.stderr,
            )
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
