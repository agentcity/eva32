#!/usr/bin/env python3
"""Замер гибридного маршрутизатора без сети, ключей и денег.

Прогоняет ``eval/hybrid_golden.jsonl`` через настоящий ``eva.router.answer``,
подменив клиентов моделей заглушками: ни один HTTP-запрос не выполняется.
Считает три вещи, ради которых гибрид и нужен:

* **точность маршрутов** — совпал ли ``Answer.route.kind`` с ``expect_route``;
* **долю скиллов** — какая часть запросов обошлась без модели (0 ₽);
* **разбивку по классам** — какой скилл поймал вопрос, что ушло в модель и
  какие промахи содержания остались (видно, куда расти: скилл без попаданий
  значит, что в наборе нет вопросов его класса);
* **ожидаемую цену на 1000 запросов** — по тарифу моделей и оценке токенов.

Формат строки ``eval/hybrid_golden.jsonl``::

    {"question": "...", "expect_route": "skill"|"small"|"big",
     "expect_contains": ["990", "[1]"]}

``expect_contains`` — подстроки, которые обязана содержать детерминированная
часть ответа. Проверяются только строки, где ответ даёт скилл: текст малой и
большой модели в замере — заглушка, судить по нему нечего. Для маршрутов
``small``/``big`` список обычно пуст. Регистр, ``ё`` и пробелы (включая
неразрывные) при сравнении не важны — «2 490» и «2490» считаются одним числом.

Время. Скилл ``time`` получает фиксированное «сейчас» (``EVAL_NOW``) через
``meta`` запроса: иначе ответ «который час» менялся бы от прогона к прогону и
метрики не воспроизводились бы.

Цена. Тариф задаётся в ₽ за **1000 токенов** и берётся из ``eva.config``
(``Config.prices``), а если модуля нет — из констант ``DEFAULT_PRICES_RUB_PER_1K``
с комментарием. Переопределить тариф можно аргументами ``--big-price-in``,
``--big-price-out``, ``--small-price-in``, ``--small-price-out``.

Длина prompt — оценка по числу символов (``CHARS_PER_TOKEN``), а не настоящий
BPE-токенизатор: внешних зависимостей нет, а для цены на 1000 запросов нужен
порядок величины, а не токен в токен. К prompt каждого модельного ответа
добавляется системный промпт (``prompts/system.ru.md``), потому что прод его
тоже отправляет. Длина ответа — константы ``COMPLETION_TOKENS`` по маршрутам.

Запуск::

    python3 tools/eval_hybrid.py
    python3 tools/eval_hybrid.py --json
    python3 tools/eval_hybrid.py --big-price-in 0.2 --big-price-out 0.6

Код возврата: 0 — метрики посчитаны; 2 — модулей ``eva`` ещё нет (это не
падение, а честное «считать нечем»); 1 — ошибка замера или не пройден порог
``--min-accuracy`` / ``--max-cost-per-1000``.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import traceback
from dataclasses import is_dataclass, replace
from typing import Any, Iterable, Mapping

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Скрипт лежит в tools/, а пакет eva — в корне: без этого `python3
# tools/eval_hybrid.py` не увидит eva, хотя из теста (cwd=корень) всё работает.
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

GOLDEN = os.path.join(ROOT, "eval", "hybrid_golden.jsonl")
SYSTEM_PROMPT = os.path.join(ROOT, "prompts", "system.ru.md")
CATALOG = os.path.join(ROOT, "data", "catalog.json")

ROUTE_KINDS = ("skill", "small", "big")

#: Модули, без которых маршрут не посчитать (контракт docs/hybrid.md).
CORE_MODULES = ("eva.types", "eva.config", "eva.router")
#: Модули, без которых метрики неполные (маршрут пойдёт по запасной эвристике
#: или скиллы промолчат). ``eva.skills.registry`` — точка входа скиллов.
OPTIONAL_MODULES = ("eva.skills.registry", "eva.classifier")

#: Всё, что нужно для честного замера: ядро + скиллы + классификатор.
READY_MODULES = CORE_MODULES + OPTIONAL_MODULES

#: Резервный тариф, ₽ за 1000 токенов: (prompt, completion).
#: Значения совпадают с умолчаниями ``eva/config.py``: большая модель —
#: 0.40 / 1.20 ₽ за 1k, малая — своё железо, 0/0. Это не тариф провайдера,
#: а консервативная оценка; реальный прайс передаётся аргументами.
DEFAULT_PRICES_RUB_PER_1K: dict[str, tuple[float, float]] = {
    "skill": (0.0, 0.0),
    "small": (0.0, 0.0),
    "big": (0.40, 1.20),
}

#: Символов на токен. Для русского текста BPE даёт ~2.5–3.5 символа на токен;
#: 3.0 — середина. Ошибка оценки систематическая и одинаковая для вариантов,
#: поэтому сравнение «гибрид vs всегда big» от неё не зависит.
CHARS_PER_TOKEN = 3.0

#: Ожидаемая длина ответа, токены. Скилл модель не зовёт (0), малая отвечает
#: коротко, большая — развёрнуто. Переопределяется аргументами.
COMPLETION_TOKENS: dict[str, int] = {"skill": 0, "small": 120, "big": 220}

#: Стенд: обе модели «настроены» адресами-заглушками. Клиенты всё равно не
#: создаются — ``answer`` получает stub-клиенты через ``deps``. Пустые
#: ``EVA_BIG_BASE``/``EVA_BIG_MODEL`` заставили бы роутер считать большую
#: модель недоступной и исказили бы маршруты.
EVAL_ENV = {
    "EVA_SMALL_BASE": "http://127.0.0.1:9/v1",
    "EVA_SMALL_KEY": "eval-stub",
    "EVA_SMALL_MODEL": "eval-stub-small",
    "EVA_SMALL_TIMEOUT": "1",
    "EVA_SMALL_RETRIES": "0",
    "EVA_SMALL_PRICE_PROMPT": "0",
    "EVA_SMALL_PRICE_COMPLETION": "0",
    "EVA_BIG_BASE": "http://127.0.0.1:9/v1",
    "EVA_BIG_KEY": "eval-stub",
    "EVA_BIG_MODEL": "eval-stub-big",
    "EVA_BIG_TIMEOUT": "1",
    "EVA_BIG_RETRIES": "0",
    # Без лимита: замеряем выбор маршрута, а не срабатывание бюджета.
    # Иначе высокий тариф в --big-price-* мог бы молча понизить big до small.
    "EVA_MAX_COST_RUB": "none",
    "EVA_MAX_TOKENS": "512",
}

#: Текст-маркер ответа заглушки; по нему видно, что модель «вызвана», а не
#: скилл. В метрики содержание модельных ответов не входит.
STUB_PREFIX = "[заглушка "

#: Фиксированное «сейчас» для скилла времени. Без него вопрос «который час»
#: в замере отвечал бы реальными часами, и метрики не воспроизводились бы.
EVAL_NOW = "2026-11-18T08:54:00+03:00"


class EvalUnavailable(RuntimeError):
    """Модулей ``eva`` ещё нет — метрики считать не на чем."""

    def __init__(self, missing: Iterable[str]):
        self.missing = tuple(dict.fromkeys(missing))
        super().__init__("; ".join(f"модуль {name} не найден" for name in self.missing))


# --------------------------------------------------------------------------- #
# загрузка набора
# --------------------------------------------------------------------------- #
def load_golden(path: str = GOLDEN) -> list[dict]:
    """Прочитать и проверить набор. Ошибки формата — с номером строки."""
    rows: list[dict] = []
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
            expect_route = row.get("expect_route")
            if expect_route not in ROUTE_KINDS:
                raise ValueError(
                    f"{path}:{lineno}: expect_route={expect_route!r}, "
                    f"ожидался один из {', '.join(ROUTE_KINDS)}"
                )
            expect_contains = row.get("expect_contains", [])
            if not isinstance(expect_contains, list) or not all(
                isinstance(item, str) for item in expect_contains
            ):
                raise ValueError(f"{path}:{lineno}: expect_contains должен быть списком строк")
            rows.append(
                {
                    "question": question.strip(),
                    "expect_route": expect_route,
                    "expect_contains": [item for item in expect_contains if item],
                }
            )
    if not rows:
        raise ValueError(f"{path}: пустой набор")
    return rows


# --------------------------------------------------------------------------- #
# доступность eva и конфиг
# --------------------------------------------------------------------------- #
def module_status(names: Iterable[str] = CORE_MODULES) -> tuple[list[str], dict[str, str]]:
    """Проверить импортируемость модулей без сети.

    Возвращает (отсутствующие, ошибки). ``ModuleNotFoundError`` про сам модуль
    или его зависимость считается отсутствием; синтаксическая ошибка и прочий
    сбой импорта — отдельной ошибкой, чтобы не прятать баг за «нет модуля».
    """
    missing: list[str] = []
    errors: dict[str, str] = {}
    for name in names:
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as exc:
            absent = exc.name or name
            # Если нет всего пакета ``eva``, Python скажет «eva», а не
            # «eva.router». Для сообщения важнее запрошенный модуль.
            if name == absent or name.startswith(absent + ".") or absent == "eva":
                missing.append(name)
            else:
                # Сломанная зависимость внутри модуля (например, eva.skills):
                # называем её, чтобы причина была видна.
                missing.append(absent)
        except Exception as exc:  # noqa: BLE001 — показать, а не проглотить
            errors[name] = f"{type(exc).__name__}: {exc}"
    return list(dict.fromkeys(missing)), errors


def missing_for_metrics() -> list[str]:
    """Список отсутствующих модулей для замера метрик (skip в тесте).

    ``eva.router`` умеет работать без скиллов и классификатора (запасная
    эвристика), но метрики тогда неполные, поэтому для теста ждём весь набор.
    """
    missing, errors = module_status(READY_MODULES)
    if errors:
        # Модуль есть, но сломан: это не «нет модулей», а ошибка. Тест должен
        # упасть с внятным текстом, CLI — показать её.
        first = next(iter(errors.items()))
        raise RuntimeError(f"{first[0]}: {first[1]}")
    return missing


#: Совместимый псевдоним: раньше проверялось только ядро.
missing_core = missing_for_metrics


def _with_stub_models(cfg: Any) -> Any:
    """Гарантировать, что обе модели «настроены» (нужно для честных маршрутов)."""
    wanted = {
        "small_base": EVAL_ENV["EVA_SMALL_BASE"],
        "small_model": EVAL_ENV["EVA_SMALL_MODEL"],
        "big_base": EVAL_ENV["EVA_BIG_BASE"],
        "big_model": EVAL_ENV["EVA_BIG_MODEL"],
        "big_key": EVAL_ENV["EVA_BIG_KEY"],
    }
    if isinstance(cfg, dict):
        result = dict(cfg)
        for key, value in wanted.items():
            if not result.get(key):
                result[key] = value
        return result
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


def load_eval_config() -> Any:
    """Конфиг стенда из переменных-заглушек, без чтения реального ``.env``."""
    config = importlib.import_module("eva.config")
    env = dict(EVAL_ENV)
    for name in ("load", "from_env", "load_config", "make_config", "build_config"):
        factory = getattr(config, name, None)
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
            return _with_stub_models(cfg)
    cls = getattr(config, "Config", None)
    if cls is not None and hasattr(cls, "from_env"):
        return _with_stub_models(cls.from_env(env))
    raise RuntimeError(
        "eva.config найден, но не удалось собрать конфиг: "
        "нет load()/from_env()/Config.from_env()"
    )


def prices_from_config(cfg: Any) -> dict[str, tuple[float, float]]:
    """Тариф (₽/1k prompt, ₽/1k completion) по маршрутам, если конфиг его знает."""
    prices = dict(DEFAULT_PRICES_RUB_PER_1K)
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
    if isinstance(cfg, dict):
        pairs = {
            "skill": (0.0, 0.0),
            "small": (
                cfg.get("small_price_prompt_rub_per_1k", cfg.get("small_price_prompt", 0.0)),
                cfg.get("small_price_completion_rub_per_1k", cfg.get("small_price_completion", 0.0)),
            ),
            "big": (
                cfg.get("big_price_prompt_rub_per_1k", cfg.get("big_price_prompt", 0.40)),
                cfg.get("big_price_completion_rub_per_1k", cfg.get("big_price_completion", 1.20)),
            ),
        }
        for kind, pair in pairs.items():
            prices[kind] = (float(pair[0] or 0.0), float(pair[1] or 0.0))
    return prices


def system_prompt_tokens(path: str = SYSTEM_PROMPT) -> int:
    """Оценка токенов системного промпта (прод шлёт его с каждым запросом)."""
    try:
        with open(path, encoding="utf-8") as fh:
            return estimate_tokens(fh.read())
    except OSError:
        return 0


# --------------------------------------------------------------------------- #
# модель цены
# --------------------------------------------------------------------------- #
def estimate_tokens(text: str) -> int:
    """Грубая оценка числа токенов по символам, не меньше 1 для непустого текста."""
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
    """Стоимость одного запроса, ₽. Скилл не зовёт модель и стоит 0."""
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
    """Ожидаемая цена 1000 запросов с данным распределением маршрутов, ₽."""
    total = 0.0
    count = 0
    for kind, question in zip(kinds, questions):
        prompt_tokens = estimate_tokens(question) + (system_tokens if kind != "skill" else 0)
        total += request_cost(kind, prompt_tokens, prices, completion_tokens)
        count += 1
    if not count:
        return 0.0
    return total / count * 1000.0


# --------------------------------------------------------------------------- #
# заглушки клиентов
# --------------------------------------------------------------------------- #
class StubClient:
    """Клиент модели, который ничего не отправляет и возвращает маркерный текст.

    Контракт ``OpenAIClient.chat``: принимает сообщения, возвращает словарь с
    ``text`` и ``usage``. Счётчик вызовов нужен, чтобы отличить «ответ дал
    скилл» от «ответ дала модель».
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.calls = 0

    def chat(self, messages: list[Any], max_tokens: int = 512) -> dict[str, Any]:
        self.calls += 1
        text = ""
        if messages:
            last = messages[-1]
            text = str(getattr(last, "content", None) or (last.get("content") if isinstance(last, dict) else "") or "")
        return {
            "text": f"{STUB_PREFIX}{self.kind}] {text[:60]}",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "cost_rub": 0.0},
            "raw": {"stub": True, "kind": self.kind},
        }


def _make_request(types_module: Any, text: str) -> Any:
    message: Any
    if hasattr(types_module, "Message"):
        try:
            message = types_module.Message(role="user", content=text)
        except TypeError:
            message = {"role": "user", "content": text}
    else:
        message = {"role": "user", "content": text}
    request_cls = getattr(types_module, "Request")
    try:
        return request_cls(
            messages=[message],
            meta={"user_id": "eval", "channel": "eval", "now": EVAL_NOW},
        )
    except TypeError:
        return request_cls(messages=[message])


def _route_field(route: Any, name: str, default: Any = None) -> Any:
    if isinstance(route, Mapping):
        return route.get(name, default)
    return getattr(route, name, default)


def _normalize(text: str) -> str:
    """Сравнение без регистра, ``ё`` и пробелов: «2 490» == «2490»."""
    return "".join(str(text or "").lower().replace("ё", "е").split())


def contains_all(answer: str, needles: Iterable[str]) -> bool:
    normalized = _normalize(answer)
    return all(_normalize(needle) in normalized for needle in needles if needle)


def _answer_class(record: Mapping[str, Any]) -> str:
    """Класс ответа: ``skill:<имя>``, ``small``, ``big`` или ``error``.

    Именно класс, а не маршрут: двадцать вопросов, пойманных ``pricing``, и
    два, пойманных ``hours``, — это разные строки отчёта, хотя маршрут один.
    """
    if record.get("error"):
        return "error"
    route = str(record.get("actual_route") or "").strip().lower()
    if route == "skill":
        name = str(record.get("skill") or "").strip()
        return f"skill:{name}" if name else "skill:?"
    return route or "error"


def _class_breakdown(
    records: list[dict],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Счётчики по классам ответа: сколько, верно, что с содержанием."""
    classes: dict[str, dict[str, Any]] = {}
    for record in records:
        name = record.get("class") or "error"
        entry = classes.setdefault(
            name,
            {
                "count": 0,
                "correct": 0,
                "content_checked": 0,
                "content_ok": 0,
                "questions": [],
            },
        )
        entry["count"] += 1
        if record.get("actual_route") == record.get("expect_route"):
            entry["correct"] += 1
        if record.get("content_checked"):
            entry["content_checked"] += 1
            if record.get("content_ok"):
                entry["content_ok"] += 1
        entry["questions"].append(record.get("question", ""))
    return classes, list(classes)


def _class_sort_key(name: str, order_index: Mapping[str, int]) -> tuple:
    """Порядок строк отчёта: скиллы в порядке реестра, потом модели, потом ошибки."""
    if name.startswith("skill:"):
        skill = name.split(":", 1)[1]
        return (0, order_index.get(skill, 10_000), skill)
    rank = {"small": 1, "big": 2, "error": 3}.get(name, 4)
    return (rank, 0, name)


def _class_title(name: str) -> str:
    if name.startswith("skill:"):
        return f"скилл {name.split(':', 1)[1]}"
    return {"small": "модель малая", "big": "модель большая", "error": "ошибки"}.get(
        name, name
    )


def registered_skill_names() -> list[str]:
    """Имена скиллов из реестра; пустой список — реестр недоступен.

    Нужны, чтобы отчёт показал и скиллы **без попаданий**: это не ошибка, а
    пустой класс в наборе — сигнал, какие вопросы стоит добавить в eval.
    """
    try:
        module = importlib.import_module("eva.skills.registry")
    except Exception:  # noqa: BLE001 — без реестра отчёт просто неполный
        return []
    entries = getattr(module, "skills", None)
    if callable(entries):
        try:
            entries = entries()
        except Exception:  # noqa: BLE001
            return []
    if entries is None:
        return []
    try:
        items = list(entries)
    except TypeError:
        return []
    names: list[str] = []
    for entry in items:
        name = str(getattr(entry, "name", "") or type(entry).__name__).strip()
        if name and name not in names:
            names.append(name)
    return names


# --------------------------------------------------------------------------- #
# замер
# --------------------------------------------------------------------------- #
def evaluate(
    rows: list[dict],
    cfg: Any | None = None,
    *,
    prices: Mapping[str, tuple[float, float]] | None = None,
    system_tokens: int | None = None,
    completion_tokens: Mapping[str, int] | None = None,
) -> dict:
    """Прогнать набор через ``eva.router.answer`` с заглушками моделей.

    Возвращает словарь метрик; печать — отдельно, чтобы тест мог проверять
    числа, а не парсить stdout.
    """
    router = importlib.import_module("eva.router")
    types_module = importlib.import_module("eva.types")
    if cfg is None:
        cfg = load_eval_config()
    if prices is None:
        prices = prices_from_config(cfg)
    if system_tokens is None:
        system_tokens = system_prompt_tokens()
    completion = dict(COMPLETION_TOKENS)
    if completion_tokens:
        completion.update(completion_tokens)

    small = StubClient("small")
    big = StubClient("big")
    # Все имена, которые понимает ``router._resolve_client``: если роутер
    # переименует ключ, замер не должен уйти в настоящий клиент (а значит,
    # в сеть). Клиенты подменяются, а не «настраиваются на localhost».
    deps = {
        "small": small,
        "big": big,
        "small_client": small,
        "big_client": big,
        "client_small": small,
        "client_big": big,
        "clients": {"small": small, "big": big},
    }

    records: list[dict] = []
    for row in rows:
        question = row["question"]
        expected = row["expect_route"]
        record = {
            "question": question,
            "expect_route": expected,
            "expect_contains": list(row.get("expect_contains") or []),
            "actual_route": None,
            "reason": "",
            "skill": None,
            "text": "",
            "content_checked": False,
            "content_ok": None,
            "error": None,
        }
        try:
            answer = router.answer(_make_request(types_module, question), cfg, deps)
            route = _route_field(answer, "route", answer)
            record["actual_route"] = str(_route_field(route, "kind", "") or "").lower()
            record["reason"] = str(_route_field(route, "reason", "") or "")
            record["skill"] = _route_field(answer, "skill")
            record["text"] = str(_route_field(answer, "text", "") or "")
        except Exception as exc:  # noqa: BLE001 — одна строка не роняет замер
            record["error"] = f"{type(exc).__name__}: {exc}"

        if record["actual_route"] == "skill":
            # Скилл обязан дать непустой детерминированный ответ. Пустой текст
            # при маршруте skill — сломанный скилл, а не «нечего проверять».
            record["content_checked"] = True
            record["content_ok"] = bool(record["text"].strip()) and contains_all(
                record["text"], record["expect_contains"]
            )
        elif record["expect_contains"]:
            # Ожидался факт из каталога, а скилл не сработал: это промах
            # содержания, а не «нечего проверять».
            record["content_checked"] = True
            record["content_ok"] = False
        record["class"] = _answer_class(record)
        records.append(record)

    total = len(records)
    correct = sum(1 for r in records if r["actual_route"] == r["expect_route"])
    actual_counts = {kind: sum(1 for r in records if r["actual_route"] == kind) for kind in ROUTE_KINDS}
    expected_counts = {
        kind: sum(1 for r in records if r["expect_route"] == kind) for kind in ROUTE_KINDS
    }
    errors = [r for r in records if r["error"]]
    empty_answers = [
        r
        for r in records
        if r["actual_route"] in ROUTE_KINDS and not r["text"].strip()
    ]
    content_checked = [r for r in records if r["content_checked"]]
    content_ok = [r for r in content_checked if r["content_ok"]]
    mismatches = [
        {
            "question": r["question"],
            "expect_route": r["expect_route"],
            "actual_route": r["actual_route"],
            "reason": r["reason"],
            "error": r["error"],
        }
        for r in records
        if r["actual_route"] != r["expect_route"] or r["error"]
    ]
    confusion: dict[str, dict[str, int]] = {
        expected: {actual: 0 for actual in ROUTE_KINDS} for expected in ROUTE_KINDS
    }
    for r in records:
        actual = r["actual_route"] if r["actual_route"] in ROUTE_KINDS else "error"
        confusion[r["expect_route"]][actual] = confusion[r["expect_route"]].get(actual, 0) + 1

    classes, class_order = _class_breakdown(records)
    registered = registered_skill_names()
    order_index = {name: index for index, name in enumerate(registered)}
    class_order.sort(key=lambda name: _class_sort_key(name, order_index))
    skill_hits: dict[str, int] = {
        name: classes.get(f"skill:{name}", {}).get("count", 0) for name in registered
    }
    for name in classes:
        if name.startswith("skill:"):
            skill_hits.setdefault(name.split(":", 1)[1], classes[name]["count"])
    unused_skills = [name for name, count in skill_hits.items() if not count]
    content_misses = [
        {
            "question": r["question"],
            "actual_route": r["actual_route"],
            "skill": r["skill"],
            "expect_contains": list(r["expect_contains"]),
            "text": r["text"][:160],
        }
        for r in records
        if r["content_checked"] and not r["content_ok"]
    ]
    model_questions: dict[str, list[str]] = {"small": [], "big": []}
    for r in records:
        if r["actual_route"] in model_questions:
            model_questions[r["actual_route"]].append(r["question"])

    questions = [r["question"] for r in records]
    actual_kinds = [r["actual_route"] for r in records]
    expected_kinds = [r["expect_route"] for r in records]
    actual_cost = cost_per_1000(actual_kinds, questions, prices, system_tokens, completion)
    ideal_cost = cost_per_1000(expected_kinds, questions, prices, system_tokens, completion)
    always_big_cost = cost_per_1000(
        ["big"] * total, questions, prices, system_tokens, completion
    )

    missing, errors_import = module_status(READY_MODULES)
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "actual_counts": actual_counts,
        "expected_counts": expected_counts,
        "skill_share": actual_counts["skill"] / total if total else 0.0,
        "expected_skill_share": expected_counts["skill"] / total if total else 0.0,
        "confusion": confusion,
        "mismatches": mismatches,
        "classes": classes,
        "class_order": class_order,
        "skill_hits": skill_hits,
        "unused_skills": unused_skills,
        "content_misses": content_misses,
        "model_questions": model_questions,
        "content_checked": len(content_checked),
        "content_ok": len(content_ok),
        "content_accuracy": (len(content_ok) / len(content_checked)) if content_checked else 0.0,
        "errors": len(errors),
        "error_rows": errors,
        "empty_answers": len(empty_answers),
        "empty_questions": [r["question"] for r in empty_answers],
        "prices_rub_per_1k": {kind: list(pair) for kind, pair in prices.items()},
        "system_tokens": system_tokens,
        "completion_tokens": dict(completion),
        "cost_per_request_rub": actual_cost / 1000.0,
        "cost_per_1000_rub": actual_cost,
        "ideal_cost_per_1000_rub": ideal_cost,
        "always_big_cost_per_1000_rub": always_big_cost,
        "overpay_per_1000_rub": actual_cost - ideal_cost,
        "savings_vs_always_big": (
            (1.0 - actual_cost / always_big_cost) if always_big_cost > 0 else 0.0
        ),
        "skill_calls": small.calls + big.calls,
        "stub_calls": {"small": small.calls, "big": big.calls},
        "missing_modules": missing,
        "module_errors": errors_import,
        "records": records,
    }


# --------------------------------------------------------------------------- #
# печать
# --------------------------------------------------------------------------- #
def _share(count: int, total: int) -> str:
    return f"{count}/{total} = {count / total * 100:.1f}%" if total else "0/0"


def format_report(metrics: Mapping[str, Any]) -> str:
    total = int(metrics["total"])
    lines: list[str] = []
    lines.append(f"гибрид: {total} вопросов, модели — заглушки (сеть не используется)")
    missing = metrics.get("missing_modules") or []
    if missing:
        lines.append("  внимание: " + ", ".join(f"модуль {name} не найден" for name in missing))
        lines.append("  маршрут считает запасная эвристика/скиллы молчат — метрики неполные")
    for name, error in (metrics.get("module_errors") or {}).items():
        lines.append(f"  ошибка импорта {name}: {error}")
    lines.append("")
    lines.append("маршруты:")
    lines.append(
        f"  точность        {_share(metrics['correct'], total)}"
    )
    for kind, title in (("skill", "скиллы"), ("small", "малая"), ("big", "большая")):
        actual = metrics["actual_counts"][kind]
        expected = metrics["expected_counts"][kind]
        lines.append(
            f"  {title:<14} {_share(actual, total)}   (в наборе {expected})"
        )
    if metrics["errors"]:
        lines.append(f"  ошибок ответа: {metrics['errors']}")
    mismatches = metrics.get("mismatches") or []
    if mismatches:
        lines.append("  промахи:")
        for item in mismatches[:20]:
            arrow = f"{item['expect_route']}→{item['actual_route'] or 'error'}"
            reason = item.get("reason") or item.get("error") or ""
            lines.append(f"    - [{arrow}] «{item['question']}»  {reason[:90]}")
        if len(mismatches) > 20:
            lines.append(f"    ... ещё {len(mismatches) - 20}")

    classes = metrics.get("classes") or {}
    if classes:
        lines.append("")
        lines.append("разбивка по классам (кто ответил, что проверено):")
        for name in metrics.get("class_order") or sorted(classes):
            entry = classes[name]
            tail = ""
            if entry.get("content_checked"):
                tail = (
                    f"   содержание {entry['content_ok']}/"
                    f"{entry['content_checked']}"
                )
            lines.append(f"  {_class_title(name):<20}{_share(entry['count'], total)}{tail}")
        unused = metrics.get("unused_skills") or []
        if unused:
            lines.append(
                "  без попаданий: " + ", ".join(unused)
                + "  (в наборе нет вопросов этих классов — или скилл не срабатывает)"
            )
        misses = metrics.get("content_misses") or []
        lines.append(f"  промахов содержания: {len(misses)}")
        for item in misses[:10]:
            expected = ", ".join(item.get("expect_contains") or [])
            lines.append(
                f"    - «{item['question']}» → {item['actual_route'] or 'error'}"
                f" (скилл {item.get('skill') or '—'}), ждали: {expected}"
            )

    prices = metrics["prices_rub_per_1k"]
    lines.append("")
    lines.append("цена (₽ за 1000 запросов):")
    lines.append(
        f"  прайс: большая {prices['big'][0]:.2f}/{prices['big'][1]:.2f}, "
        f"малая {prices['small'][0]:.2f}/{prices['small'][1]:.2f} "
        f"(prompt/completion за 1k токенов);"
    )
    lines.append(
        f"  системный промпт ≈ {metrics['system_tokens']} токенов, "
        f"ответ ≈ {metrics['completion_tokens']['small']}/"
        f"{metrics['completion_tokens']['big']} токенов (малая/большая)"
    )
    lines.append(f"  гибрид                {metrics['cost_per_1000_rub']:.2f}")
    lines.append(f"  идеальный маршрут     {metrics['ideal_cost_per_1000_rub']:.2f}")
    lines.append(f"  всегда большая        {metrics['always_big_cost_per_1000_rub']:.2f}")
    lines.append(
        f"  экономия vs big       {metrics['savings_vs_always_big'] * 100:.1f}%"
    )
    lines.append(
        f"  разница с идеалом    {metrics['overpay_per_1000_rub']:+.2f}"
        "  (минус — промахи увели в дешёвый маршрут)"
    )
    if metrics.get("empty_answers"):
        lines.append(f"  пустых ответов        {metrics['empty_answers']}")
    lines.append("")
    lines.append(
        f"содержание: проверено {metrics['content_checked']}, "
        f"ok {metrics['content_ok']}"
        + (
            f" ({metrics['content_accuracy'] * 100:.1f}%)"
            if metrics["content_checked"]
            else ""
        )
    )
    return "\n".join(lines)


def _resolve_prices(args: argparse.Namespace, cfg: Any) -> dict[str, tuple[float, float]]:
    prices = prices_from_config(cfg)
    overrides = {
        "small": (args.small_price_in, args.small_price_out),
        "big": (args.big_price_in, args.big_price_out),
    }
    for kind, (prompt_price, completion_price) in overrides.items():
        current = list(prices[kind])
        if prompt_price is not None:
            current[0] = prompt_price
        if completion_price is not None:
            current[1] = completion_price
        prices[kind] = (current[0], current[1])
    return prices


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Замер гибрида без сети: маршруты, скиллы, цена на 1000 запросов."
    )
    parser.add_argument("--golden", default=GOLDEN, help="путь к hybrid_golden.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="взять первые N строк")
    parser.add_argument("--json", action="store_true", help="напечатать метрики в JSON")
    parser.add_argument("--quiet", action="store_true", help="только итоговая строка")
    parser.add_argument(
        "--big-price-in", type=float, default=None, help="большая: ₽/1k prompt-токенов"
    )
    parser.add_argument(
        "--big-price-out", type=float, default=None, help="большая: ₽/1k completion-токенов"
    )
    parser.add_argument(
        "--small-price-in", type=float, default=None, help="малая: ₽/1k prompt-токенов"
    )
    parser.add_argument(
        "--small-price-out", type=float, default=None, help="малая: ₽/1k completion-токенов"
    )
    parser.add_argument(
        "--system-tokens", type=int, default=None, help="токенов системного промпта (по умолчанию — оценка)"
    )
    parser.add_argument(
        "--small-completion-tokens", type=int, default=None, help="ожидаемая длина ответа малой, токенов"
    )
    parser.add_argument(
        "--big-completion-tokens", type=int, default=None, help="ожидаемая длина ответа большой, токенов"
    )
    parser.add_argument(
        "--min-accuracy", type=float, default=None, help="порог точности маршрутов (0..1)"
    )
    parser.add_argument(
        "--max-cost-per-1000", type=float, default=None, help="порог цены на 1000 запросов, ₽"
    )
    parser.add_argument(
        "--min-skill-share", type=float, default=None, help="минимальная доля скиллов (0..1)"
    )
    parser.add_argument("--traceback", action="store_true", help="показать трейсбек при сбое")
    args = parser.parse_args(argv)

    missing, errors = module_status(CORE_MODULES)
    if errors:
        for name, error in errors.items():
            print(f"ошибка импорта {name}: {error}", file=sys.stderr)
        return 2
    if missing:
        for name in missing:
            print(f"модуль {name} не найден", file=sys.stderr)
        print(
            "метрики гибрида считать не на чем: нужны eva/types.py, eva/config.py, "
            "eva/router.py (контракт docs/hybrid.md).\n"
            "когда модули появятся, повторите: python3 tools/eval_hybrid.py",
            file=sys.stderr,
        )
        return 2

    try:
        rows = load_golden(args.golden)
        if args.limit:
            rows = rows[: args.limit]
        cfg = load_eval_config()
        prices = _resolve_prices(args, cfg)
        completion = {
            "small": args.small_completion_tokens,
            "big": args.big_completion_tokens,
        }
        completion = {k: v for k, v in completion.items() if v is not None}
        metrics = evaluate(
            rows,
            cfg,
            prices=prices,
            system_tokens=args.system_tokens,
            completion_tokens=completion,
        )
    except EvalUnavailable as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 — CLI не должен пугать трейсбеком
        print(f"ошибка замера: {type(exc).__name__}: {exc}", file=sys.stderr)
        if args.traceback:
            traceback.print_exc()
        return 1

    if args.json:
        printable = {key: value for key, value in metrics.items() if key != "records"}
        printable["records"] = [
            {key: value for key, value in record.items() if key != "text"}
            for record in metrics["records"]
        ]
        print(json.dumps(printable, ensure_ascii=False, indent=2))
    elif args.quiet:
        print(
            f"{metrics['correct']}/{metrics['total']} маршрутов, "
            f"скиллы {metrics['skill_share'] * 100:.1f}%, "
            f"{metrics['cost_per_1000_rub']:.2f} ₽/1000"
        )
    else:
        print(format_report(metrics))

    failed = False
    if args.min_accuracy is not None and metrics["accuracy"] < args.min_accuracy:
        print(
            f"порог точности не пройден: {metrics['accuracy']:.3f} < {args.min_accuracy:.3f}",
            file=sys.stderr,
        )
        failed = True
    if (
        args.min_skill_share is not None
        and metrics["skill_share"] < args.min_skill_share
    ):
        print(
            f"порог доли скиллов не пройден: {metrics['skill_share']:.3f} < "
            f"{args.min_skill_share:.3f}",
            file=sys.stderr,
        )
        failed = True
    if (
        args.max_cost_per_1000 is not None
        and metrics["cost_per_1000_rub"] > args.max_cost_per_1000
    ):
        print(
            f"порог цены не пройден: {metrics['cost_per_1000_rub']:.2f} ₽ > "
            f"{args.max_cost_per_1000:.2f} ₽",
            file=sys.stderr,
        )
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
