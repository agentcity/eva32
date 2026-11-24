#!/usr/bin/env python3
"""Прогон golden компетенций через живой бэкенд: ответы и проценты по категориям.

`tools/eval_competence.py` умеет оценивать готовые ответы, но не умеет их получать.
Этот раннер — вторая половина: спрашивает модель, складывает ответы в JSONL и сразу
печатает таблицу по 9 категориям и список провалов с причинами.

    python3 tools/eval_competence_run.py --backend ollama --model qwen2.5:3b
    python3 tools/eval_competence_run.py --backend ollama --model qwen2.5:7b --workers 4
    python3 tools/eval_competence_run.py --backend openai --base http://127.0.0.1:8000
    python3 tools/eval_competence_run.py --backend vtm --model demo/model.vtm
    python3 tools/eval_competence_run.py --backend mock        # проверка самого раннера

Бэкенды:

* ``ollama`` — нативный `POST /api/chat` на ``http://127.0.0.1:11434`` (по умолчанию);
* ``openai`` — OpenAI-совместимый `POST /v1/chat/completions` (vLLM, eva-сервер);
* ``vtm``    — локальная ВТМ через ``eva.backends.VtmBackend`` (нужен ``tokenizer.json``);
* ``mock``   — синтетические ответы без сети и без модели. Нужен только для проверки
  самого раннера: числа ``mock`` не характеризуют ни одну модель.

Ответы пишутся в ``runs/eval/<бэкенд>-<модель>-<время>.jsonl``. Каждая строка —
``question``, ``category``, ``answer``, ``ok``, ``failures``, ``error``; файл совместим
с ``tools/eval_competence.py --answers``, поэтому оценку можно перепроверить, а провалы —
прочитать глазами.

Ошибки бэкенда: отсутствие сервера/модели проверяется до прогона и печатается одной
строкой (без трейсбеков), код возврата 2. Транзиентные ошибки (таймаут, обрыв, 429, 5xx)
повторяются с экспоненциальной паузой ``--retries`` раз, у каждой строки свой
``--timeout``. Код возврата 0 — прогон состоялся (даже если модель провалила проверки);
1 — только с ``--strict``, если есть провалы.

Зависимостей кроме стандартной библиотеки нет; чужие проверки не дублируются —
используются ``load_golden``/``check_row``/``evaluate`` из ``tools/eval_competence.py``.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pathlib
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any, Callable

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tools.eval_competence as ec  # noqa: E402

GOLDEN = ROOT / "eval" / "golden_competence.jsonl"
SYSTEM_PROMPT = ROOT / "prompts" / "system.assistant.md"
OUT_DIR = ROOT / "runs" / "eval"

DEFAULT_OLLAMA_BASE = os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434"
DEFAULT_OPENAI_BASE = os.environ.get("EVA_BASE") or "http://127.0.0.1:8000"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"
DEFAULT_OPENAI_MODEL = os.environ.get("EVA_MODEL") or "eva32"
DEFAULT_TIMEOUT = 180.0
DEFAULT_MAX_TOKENS = 1024
DEFAULT_WORKERS = 4
DEFAULT_RETRIES = 2
BACKOFF = 0.5
MAX_BACKOFF = 8.0

#: Поля строки ответа — фиксированный порядок, чтобы диффы были читаемыми.
RECORD_FIELDS = (
    "backend", "model", "question", "category", "checks", "note",
    "answer", "ok", "failures", "error", "attempts", "elapsed_ms",
)


class BackendError(RuntimeError):
    """Понятная ошибка бэкенда: печатается без трейсбека.

    ``status`` — HTTP-код, если он был; ``transient`` — стоит ли повторять запрос.
    """

    def __init__(self, message: str, *, status: int | None = None,
                 transient: bool = False) -> None:
        super().__init__(message)
        self.status = status
        self.transient = transient


class TransientError(BackendError):
    """Временная ошибка (таймаут, обрыв, 429, 5xx): запрос имеет смысл повторить."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message, status=status, transient=True)


# --- HTTP -------------------------------------------------------------------

def _error_detail(exc: urllib.error.HTTPError) -> str:
    """Человеческий текст ошибки: Ollama/vLLM кладут причину в JSON."""
    try:
        raw = exc.read()
    except Exception:  # noqa: BLE001 — тело уже могло быть прочитано
        return ""
    text = raw[:500].decode("utf-8", "replace").strip()
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        obj = None
    if isinstance(obj, dict):
        for key in ("error", "message", "detail"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, dict) and isinstance(value.get("message"), str):
                return value["message"].strip()
    return " ".join(text.split())[:300]


def _request(url: str, *, payload: dict | None = None, timeout: float,
             headers: dict[str, str] | None = None) -> dict:
    """JSON-запрос; транспортные ошибки → TransientError, HTTP → BackendError."""
    data = None
    all_headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        all_headers["Content-Type"] = "application/json"
    if headers:
        all_headers.update(headers)
    request = urllib.request.Request(url, data=data, headers=all_headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = _error_detail(exc)
        message = f"HTTP {exc.code} на {url}"
        if detail:
            message += f": {detail}"
        if exc.code == 429 or 500 <= exc.code < 600:
            raise TransientError(message, status=exc.code) from None
        raise BackendError(message, status=exc.code) from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise TransientError(f"нет соединения с {url} ({reason})") from None
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise TransientError(f"{url}: ответ не JSON (возможно, обрезан)") from None
    if not isinstance(parsed, dict):
        raise BackendError(f"{url}: ожидался JSON-объект, пришло {type(parsed).__name__}")
    return parsed


def get_json(url: str, timeout: float) -> dict:
    return _request(url, timeout=timeout)


def post_json(url: str, payload: dict, timeout: float,
              headers: dict[str, str] | None = None) -> dict:
    return _request(url, payload=payload, timeout=timeout, headers=headers)


def _retry(call: Callable[[], str], *, retries: int = DEFAULT_RETRIES,
           backoff: float = BACKOFF, sleep: Callable[[float], None] = time.sleep,
           label: str = "запрос") -> tuple[str, int]:
    """Выполнить ``call`` с повторами только на временных ошибках.

    Возвращает ``(ответ, число попыток)``. Постоянные ошибки (404 «нет модели»,
    400 «плохой запрос») не повторяются: повтор их не исправит.
    """
    attempts = max(1, retries + 1)
    last: TransientError | None = None
    for attempt in range(1, attempts + 1):
        try:
            return call(), attempt
        except TransientError as exc:
            last = exc
            if attempt >= attempts:
                break
            sleep(min(backoff * 2 ** (attempt - 1), MAX_BACKOFF))
    raise BackendError(
        f"{label}: не удалось за {attempts} попыт(ки): {last}",
        status=getattr(last, "status", None),
    )


# --- бэкенды ----------------------------------------------------------------

def ollama_answer(base: str, model: str, system: str, question: str, *,
                  max_tokens: int, temperature: float, timeout: float) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    data = post_json(base.rstrip("/") + "/api/chat", payload, timeout)
    content = data.get("message", {}).get("content") if isinstance(data.get("message"), dict) else None
    if not isinstance(content, str):
        raise BackendError(f"ollama: в ответе нет message.content: {str(data)[:200]}")
    return content.strip()


def openai_answer(base: str, model: str, system: str, question: str, *,
                  max_tokens: int, temperature: float, timeout: float,
                  key: str = "") -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {key}"} if key else None
    data = post_json(base.rstrip("/") + "/v1/chat/completions", payload, timeout, headers)
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise BackendError(f"openai: в ответе нет choices[0].message.content: {str(data)[:200]}")
    if not isinstance(content, str):
        raise BackendError("openai: content ответа — не строка")
    return content.strip()


def preflight_ollama(base: str, model: str, *, timeout: float = 5.0) -> None:
    """Проверить, что Ollama запущена и модель скачана. Иначе — понятная ошибка."""
    url = base.rstrip("/") + "/api/tags"
    try:
        data = get_json(url, min(timeout, 10.0))
    except BackendError as exc:
        raise BackendError(
            f"Ollama не отвечает на {base}: {exc}. Запустите сервер командой "
            f"`ollama serve` и повторите; адрес можно задать через --base "
            f"или OLLAMA_HOST."
        ) from None
    names = [
        str(item.get("name", ""))
        for item in data.get("models", []) or []
        if isinstance(item, dict) and item.get("name")
    ]
    if not names:
        raise BackendError(
            f"Ollama запущена на {base}, но моделей нет. Скачайте модель: "
            f"`ollama pull {model}`."
        )
    if model not in names and f"{model}:latest" not in names:
        shown = ", ".join(sorted(names)[:10])
        raise BackendError(
            f"модель «{model}» не найдена в Ollama. Доступны: {shown}. "
            f"Скачайте её: `ollama pull {model}` или выберите --model из списка."
        )


def preflight_openai(base: str, model: str, *, timeout: float = 5.0,
                     key: str = "") -> None:
    """Проверить доступность OpenAI-совместимого эндпоинта до прогона."""
    url = base.rstrip("/") + "/v1/models"
    headers = {"Authorization": f"Bearer {key}"} if key else None
    try:
        get_json_with_headers(url, min(timeout, 10.0), headers)
    except BackendError as exc:
        if exc.status == 404:
            # Не все совместимые серверы отдают /v1/models — решает первый запрос.
            return
        if exc.status in (401, 403):
            raise BackendError(
                f"{url}: эндпоинт требует ключ (HTTP {exc.status}). Задайте --key "
                f"или переменную окружения."
            ) from None
        raise BackendError(
            f"OpenAI-эндпоинт не отвечает на {base}: {exc}. Запустите сервер "
            f"(vLLM/eva) или задайте --base; модель — {model}."
        ) from None


def get_json_with_headers(url: str, timeout: float,
                          headers: dict[str, str] | None = None) -> dict:
    return _request(url, timeout=timeout, headers=headers)


def _vtm_module():
    """Импорт ВТМ-бэкенда: пакет ленивый, тяжёлое грузится при первом открытии."""
    try:
        from eva import backends  # noqa: PLC0415 — импорт по требованию
    except Exception as exc:  # noqa: BLE001 — eva может быть недоступна
        raise BackendError(f"vtm: не импортируется eva.backends: {exc}") from None
    return backends


def preflight_vtm(model: str | None, max_tokens: int):
    """Создать бэкенд ВТМ и проверить модель/токенизатор/библиотеку заранее."""
    backends = _vtm_module()
    try:
        backend = backends.VtmBackend(model=model or None, max_tokens=max_tokens)
    except (backends.VtmError, ValueError) as exc:
        raise BackendError(f"vtm: не удалось собрать конфиг: {exc}") from None
    path = backend.config.model
    if not path:
        raise BackendError(
            "vtm: модель не задана. Укажите --model путь к .vtm или EVA_VTM_MODEL "
            "(по умолчанию demo/model.vtm)."
        )
    if not pathlib.Path(path).expanduser().is_file():
        raise BackendError(f"vtm: файл модели не найден: {path}")
    if not backend.config.tokenizer:
        raise BackendError(
            "vtm: нужен tokenizer.json — без него ВТМ не умеет текст, только id. "
            "Задайте EVA_VTM_TOKENIZER или положите tokenizer.json рядом с моделью; "
            "демо-модель предназначена для проверки обвязки, а не для golden."
        )
    if not backends.available():
        raise BackendError(
            "vtm: libvtm не найдена и не собирается. Соберите src/vtm (`make -C "
            "src/vtm`) и задайте EVA_VTM_LIB."
        )
    return backend


def vtm_answer(backend: Any, system: str, question: str, *,
               max_tokens: int, timeout: float) -> str:
    """Ответ ВТМ. Таймаут обеспечивается отменой из отдельного потока."""
    backends = _vtm_module()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    fired = threading.Event()

    def on_timeout() -> None:
        fired.set()
        backend.cancel()

    timer = threading.Timer(timeout, on_timeout)
    timer.daemon = True
    timer.start()
    try:
        result = backend.chat(messages, max_tokens=max_tokens)
    except backends.VtmUnavailable as exc:
        raise BackendError(f"vtm: локальная модель недоступна: {exc}") from None
    except backends.VtmBusy as exc:
        raise TransientError(f"vtm занята: {exc}") from None
    except backends.VtmCancelled:
        raise TransientError(f"vtm: генерация отменена (таймаут {timeout:g} с)") from None
    except backends.VtmError as exc:
        raise BackendError(f"vtm: ошибка генерации: {exc}") from None
    finally:
        timer.cancel()
    if fired.is_set():
        raise TransientError(f"vtm: таймаут {timeout:g} с — генерация отменена")
    text = result.get("text", "") if isinstance(result, dict) else ""
    return str(text).strip()


# --- мок-бэкенд -------------------------------------------------------------

#: Текст без цифр, цен и слов из запретов golden — чтобы мок проходил проверки.
_MOCK_BASE = (
    "Это синтетический ответ мок-бэкенда, он нужен только для проверки раннера "
    "и ничего не говорит о модели."
)
_MOCK_FILLER = (
    "дальше", "идёт", "ровный", "текст", "для", "нужной", "длины", "без", "цифр",
    "и", "выдуманных", "фактов", "примерно", "так",
)


def mock_answer(row: dict) -> str:
    """Синтетический ответ, который заведомо проходит проверки строки.

    Это не «умная модель», а эталон для проверки самого раннера: если мок не
    набирает 72/72, сломаны проверки, сохранение ответов или подсчёт.
    """
    checks = row.get("checks", {})
    head = _MOCK_BASE
    tail: list[str] = []

    for token in checks.get("must_contain", []):
        if token == ",":
            continue  # запятые уже есть в базовой фразе
        if token == "?":
            continue  # вопрос добавит блок asks_question или отдельная фраза ниже
        if token == "1":
            tail.append("Пункт 1.")
            continue
        tail.append(f"Отмечу: {token}.")
    if checks.get("asks_question"):
        tail.append("Уточните, пожалуйста: для какой аудитории и какая цель?")
    elif "?" in checks.get("must_contain", []):
        tail.append("Что именно нужно?")

    required = len((head + " " + " ".join(tail)).split())
    min_words = int(checks.get("min_words", 0) or 0)
    filler: list[str] = []
    index = 0
    while required + len(filler) < min_words:
        filler.append(_MOCK_FILLER[index % len(_MOCK_FILLER)])
        index += 1
    return " ".join([head, *filler, *tail]).strip()


# --- прогон -----------------------------------------------------------------

def _short(text: str, limit: int = 70) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _one(row: dict, ask: Callable[[dict], tuple[str, int]], numbers: set[str],
         backend: str, model: str) -> dict:
    """Один вопрос: получить ответ, проверить его, вернуть запись для JSONL."""
    started = time.monotonic()
    answer = ""
    error = ""
    attempts = 0
    try:
        answer, attempts = ask(row)
    except BackendError as exc:
        error = str(exc)
    except Exception as exc:  # noqa: BLE001 — трейсбек рабочему не показываем
        error = f"неожиданная ошибка {type(exc).__name__}: {exc}"
    elapsed_ms = round((time.monotonic() - started) * 1000, 1)

    if error:
        failures = [f"ошибка бэкенда: {error}"]
        ok = False
    else:
        failures = ec.check_row(row, answer, numbers=numbers)
        ok = not failures
    return {
        "backend": backend,
        "model": model,
        "question": row["question"],
        "category": row["category"],
        "checks": row.get("checks", {}),
        "note": row.get("note", ""),
        "answer": answer,
        "ok": ok,
        "failures": failures,
        "error": error,
        "attempts": attempts,
        "elapsed_ms": elapsed_ms,
    }


def run_rows(rows: list[dict], ask: Callable[[dict], tuple[str, int]], *,
             workers: int = 1, numbers: set[str] | None = None,
             progress: Callable[[int, int, dict], None] | None = None,
             backend: str = "", model: str = "") -> list[dict]:
    """Спросить все строки (параллельно) и вернуть записи в порядке golden."""
    numbers = ec.catalog_numbers() if numbers is None else numbers
    total = len(rows)
    results: list[dict | None] = [None] * total

    def work(index: int) -> tuple[int, dict]:
        return index, _one(rows[index], ask, numbers, backend, model)

    if workers <= 1:
        for index in range(total):
            position, record = work(index)
            results[position] = record
            if progress is not None:
                progress(position + 1, total, record)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(work, index) for index in range(total)]
            done = 0
            for future in concurrent.futures.as_completed(futures):
                position, record = future.result()
                results[position] = record
                done += 1
                if progress is not None:
                    progress(done, total, record)
    return [record for record in results if record is not None]


# --- вывод ------------------------------------------------------------------

def render_table(report: dict) -> str:
    """Таблица «категория | прошло | всего | %» + итоговая строка."""
    header = f"{'категория':<18}{'прошло':>7}{'всего':>7}{'%':>8}"
    lines = [header, "-" * len(header)]
    passed_total = 0
    rows_total = 0
    for category, stat in report["per_category"].items():
        passed = int(stat["прошли"])
        total = int(stat["всего"])
        passed_total += passed
        rows_total += total
        share = passed / total * 100 if total else 0.0
        lines.append(f"{category:<18}{passed:>7}{total:>7}{share:>7.1f}%")
    lines.append("-" * len(header))
    share = passed_total / rows_total * 100 if rows_total else 0.0
    lines.append(f"{'итого':<18}{passed_total:>7}{rows_total:>7}{share:>7.1f}%")
    return "\n".join(lines)


def render_failures(records: list[dict]) -> str:
    """Список провалов с причинами; пусто — честное «провалов нет»."""
    failed = [record for record in records if not record["ok"]]
    if not failed:
        return "провалов нет"
    lines = [f"провалы ({len(failed)}):"]
    for record in failed:
        lines.append(f"  [{record['category']}] {_short(record['question'])}")
        lines.append(f"      {', '.join(record['failures']) or 'без причины'}")
    return "\n".join(lines)


def _display_path(path: pathlib.Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def write_records(path: pathlib.Path, records: list[dict]) -> None:
    """Сохранить ответы: по JSON на строку, поля в фиксированном порядке."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({field: record[field] for field in RECORD_FIELDS}, ensure_ascii=False)
        for record in records
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def default_out(backend: str, model: str) -> pathlib.Path:
    label = pathlib.Path(model).name if ("/" in model or "\\" in model) else model
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", f"{backend}-{label}").strip("-") or backend
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return OUT_DIR / f"{safe}-{stamp}.jsonl"


def read_system(path: pathlib.Path | None = None) -> str:
    target = path or SYSTEM_PROMPT
    try:
        return target.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise BackendError(f"не читается системный промпт {target}: {exc}") from None


def resolve_model(args: argparse.Namespace) -> str:
    if args.model:
        return str(args.model)
    if args.backend == "ollama":
        return DEFAULT_OLLAMA_MODEL
    if args.backend == "openai":
        return DEFAULT_OPENAI_MODEL
    if args.backend == "vtm":
        try:
            from eva.backends import default_model_path  # noqa: PLC0415
        except Exception:  # noqa: BLE001
            return "demo/model.vtm"
        found = default_model_path()
        return str(found) if found else "demo/model.vtm"
    return "synthetic"


def resolve_base(args: argparse.Namespace) -> str:
    if args.base:
        return str(args.base)
    return DEFAULT_OLLAMA_BASE if args.backend == "ollama" else DEFAULT_OPENAI_BASE


def build_ask(args: argparse.Namespace, system: str) -> tuple[
        Callable[[dict], tuple[str, int]], Callable[[], None], str, str]:
    """Собрать функцию «строка golden → (ответ, попытки)», очистку и подписи."""
    model = resolve_model(args)
    base = resolve_base(args)
    cleanup: Callable[[], None] = lambda: None

    if args.backend == "mock":
        return (lambda row: (mock_answer(row), 1)), cleanup, model, base

    if args.backend == "ollama":
        preflight_ollama(base, model, timeout=min(args.timeout, 10.0))

        def ask_ollama(row: dict) -> tuple[str, int]:
            return _retry(
                lambda: ollama_answer(
                    base, model, system, row["question"],
                    max_tokens=args.max_tokens, temperature=args.temperature,
                    timeout=args.timeout,
                ),
                retries=args.retries,
                label=f"ollama:{model}",
            )

        return ask_ollama, cleanup, model, base

    if args.backend == "openai":
        preflight_openai(base, model, timeout=min(args.timeout, 10.0), key=args.key)

        def ask_openai(row: dict) -> tuple[str, int]:
            return _retry(
                lambda: openai_answer(
                    base, model, system, row["question"],
                    max_tokens=args.max_tokens, temperature=args.temperature,
                    timeout=args.timeout, key=args.key,
                ),
                retries=args.retries,
                label=f"openai:{model}",
            )

        return ask_openai, cleanup, model, base

    # vtm
    backend = preflight_vtm(model, args.max_tokens)

    def ask_vtm(row: dict) -> tuple[str, int]:
        return _retry(
            lambda: vtm_answer(
                backend, system, row["question"],
                max_tokens=args.max_tokens, timeout=args.timeout,
            ),
            retries=args.retries,
            label="vtm",
        )

    def close_vtm() -> None:
        try:
            backend.close(timeout=5.0)
        except Exception:  # noqa: BLE001 — закрытие не должно ронять отчёт
            pass

    return ask_vtm, close_vtm, model, base


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"ожидалось целое число, получено {value!r}")
    if number < 1:
        raise argparse.ArgumentTypeError(f"ожидалось >= 1, получено {number}")
    return number


def _non_negative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"ожидалось целое число, получено {value!r}")
    if number < 0:
        raise argparse.ArgumentTypeError(f"ожидалось >= 0, получено {number}")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Прогнать golden компетенций через бэкенд и напечатать проценты "
                    "по категориям.",
        epilog="Примеры: --backend ollama --model qwen2.5:3b; --backend openai "
               "--base http://127.0.0.1:8000; --backend mock (без сети).",
    )
    ap.add_argument("--backend", choices=("ollama", "vtm", "openai", "mock"),
                    default="ollama", help="откуда брать ответы (по умолчанию ollama)")
    ap.add_argument("--model", default=None,
                    help="имя модели; по умолчанию qwen2.5:7b для ollama, "
                         "EVA_MODEL/eva32 для openai, demo/model.vtm для vtm")
    ap.add_argument("--golden", type=pathlib.Path, default=GOLDEN,
                    help="файл golden (по умолчанию eval/golden_competence.jsonl)")
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help="куда писать ответы (по умолчанию "
                         "runs/eval/<бэкенд>-<модель>-<время>.jsonl)")
    ap.add_argument("--base", default=None,
                    help="корень API: для ollama http://127.0.0.1:11434, "
                         "для openai http://127.0.0.1:8000")
    ap.add_argument("--key", default=os.environ.get("VLLM_API_KEY", ""),
                    help="ключ для openai-бэкенда (по умолчанию VLLM_API_KEY)")
    ap.add_argument("--workers", type=_positive_int, default=DEFAULT_WORKERS,
                    help=f"параллельных запросов (по умолчанию {DEFAULT_WORKERS}; "
                         "для vtm всегда 1)")
    ap.add_argument("--max-tokens", type=_positive_int, default=DEFAULT_MAX_TOKENS,
                    help=f"потолок новых токенов на ответ (по умолчанию "
                         f"{DEFAULT_MAX_TOKENS})")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT,
                    help=f"таймаут одного запроса, с (по умолчанию {DEFAULT_TIMEOUT:g})")
    ap.add_argument("--retries", type=_non_negative_int, default=DEFAULT_RETRIES,
                    help=f"повторов на временную ошибку (по умолчанию {DEFAULT_RETRIES})")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="температура генерации (по умолчанию 0.0 — повторяемость)")
    ap.add_argument("--system", type=pathlib.Path, default=None,
                    help="свой системный промпт (по умолчанию prompts/system.assistant.md)")
    ap.add_argument("--limit", type=_non_negative_int, default=0,
                    help="взять первые N строк golden (0 — все)")
    ap.add_argument("--strict", action="store_true",
                    help="вернуть код 1, если есть провалы (по умолчанию прогон — 0)")
    ap.add_argument("--quiet", action="store_true",
                    help="не печатать прогресс по строкам")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.golden.is_file():
        print(f"ошибка: golden не найден: {args.golden}", file=sys.stderr)
        return 2
    try:
        rows = ec.load_golden(args.golden)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ошибка: не читается golden {args.golden}: {exc}", file=sys.stderr)
        return 2
    unknown = {key for row in rows for key in row.get("checks", {})} - ec.KNOWN_CHECKS
    if unknown:
        print(f"ошибка: неизвестные проверки в golden: {sorted(unknown)}", file=sys.stderr)
        return 2
    if not rows:
        print(f"ошибка: golden пуст: {args.golden}", file=sys.stderr)
        return 2
    if args.limit:
        rows = rows[: args.limit]

    if args.timeout <= 0:
        print("ошибка: --timeout должен быть > 0", file=sys.stderr)
        return 2
    if args.max_tokens < 1:
        print("ошибка: --max-tokens должен быть >= 1", file=sys.stderr)
        return 2

    try:
        system = read_system(args.system)
        ask, cleanup, model, base = build_ask(args, system)
    except BackendError as exc:
        print(f"ошибка: {exc}", file=sys.stderr)
        return 2

    workers = args.workers
    if args.backend == "vtm" and workers > 1:
        print("предупреждение: у vtm один контекст — воркеров 1", file=sys.stderr)
        workers = 1

    def progress(done: int, total: int, record: dict) -> None:
        mark = "ok  " if record["ok"] else "FAIL"
        print(f"[{done:>3}/{total}] {mark} {record['category']:<16} "
              f"{_short(record['question'], 44)}", file=sys.stderr)

    started = time.monotonic()
    try:
        records = run_rows(
            rows, ask, workers=workers, progress=None if args.quiet else progress,
            backend=args.backend, model=model,
        )
    finally:
        cleanup()
    elapsed = time.monotonic() - started

    out = args.out or default_out(args.backend, model)
    if out.exists() and out.is_dir():
        out = out / default_out(args.backend, model).name
    try:
        write_records(out, records)
    except OSError as exc:
        print(f"ошибка: не записываются ответы в {out}: {exc}", file=sys.stderr)
        return 2

    answers = {
        record["question"]: record["answer"]
        for record in records
        if not record["error"]
    }
    report = ec.evaluate(rows, answers)

    print(f"бэкенд   {args.backend}" + (f" ({base})" if args.backend != "mock" else ""))
    print(f"модель   {model}")
    print(f"строк    {len(rows)}  воркеров {workers}  таймаут {args.timeout:g} с  "
          f"повторов {args.retries}")
    print(f"промпт   {_display_path(args.system or SYSTEM_PROMPT)}")
    print(f"ответы   {_display_path(out)}")
    print(f"время    {elapsed:.1f} с")
    print()
    if args.backend == "mock":
        print("ВНИМАНИЕ: мок-бэкенд. Ответы синтетические, числа ниже проверяют")
        print("раннер и спецификацию golden, а не модель. Живых чисел здесь нет.")
        print()
    print(render_table(report))
    print()
    print(render_failures(records))
    print()
    print(f"перепроверить ответы: python3 tools/eval_competence.py --answers "
          f"{_display_path(out)}")

    if records and all(record["error"] for record in records):
        print("\nошибка: ни один запрос не удался — бэкенд не отвечает.", file=sys.stderr)
        return 2
    if args.strict and any(not record["ok"] for record in records):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
