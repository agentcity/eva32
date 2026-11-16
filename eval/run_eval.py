#!/usr/bin/env python3
"""Прогон золотого набора против OpenAI-совместимого эндпоинта EVA32.

Отправляет ровно ту же форму запроса, что и прод: system-промпт из
prompts/system.ru.md, параметры из config/generation.yaml, контекст строкой.

Работает и с vLLM (http://127.0.0.1:8000), и с Ollama (http://127.0.0.1:11434).

    python3 eval/run_eval.py --base http://127.0.0.1:11434 --model eva32-dev
    python3 eval/run_eval.py --base http://127.0.0.1:8000 --model eva32

Проверки в golden.jsonl описывают поведение, а не формулировку:

    must                — все подстроки должны встретиться (без учёта регистра);
    any                 — хотя бы одна из подстрок;
    forbid              — ни одной из подстрок;
    regex               — все регулярные выражения должны найти совпадение;
    forbid_regex        — ни одно регулярное выражение не должно совпасть;
    no_prompt_shingle   — в ответе нет дословного фрагмента системного промпта
                          (8 слов подряд; можно задать другое число числом).

Комментарии и сообщения — по-русски, зависимостей кроме stdlib нет.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD = os.path.join(ROOT, "eval", "golden.jsonl")
PROMPT = os.path.join(ROOT, "prompts", "system.ru.md")
CONFIG = os.path.join(ROOT, "config", "generation.yaml")
RESULTS = os.path.join(ROOT, "eval", "results")

# Длина «шингла» для проверки утечки системного промпта.
PROMPT_SHINGLE_WORDS = 8
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_REGEX_FLAGS = re.IGNORECASE | re.UNICODE


# --- минимальный парсер нашего плоского yaml (без внешних зависимостей) ---
def load_config(path: str) -> dict:
    cfg: dict = {}
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            value = value.strip()
            if value.startswith("[") and value.endswith("]"):
                cfg[key.strip()] = [
                    v.strip().strip("\"'") for v in value[1:-1].split(",") if v.strip()
                ]
            elif value.lower() in ("true", "false"):
                cfg[key.strip()] = value.lower() == "true"
            else:
                try:
                    cfg[key.strip()] = float(value) if "." in value else int(value)
                except ValueError:
                    cfg[key.strip()] = value.strip("\"'")
    return cfg


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_system_prompt() -> str:
    """Текущий системный промпт бандла (нужен и для шинглов утечки)."""
    with open(PROMPT, encoding="utf-8") as fh:
        return fh.read().strip()


def build_payload(model: str, system: str, row: dict, cfg: dict) -> dict:
    """Тело запроса ровно в том виде, в каком его обязано слать приложение.

    Источник правды для docs/contract.md: контракт и эта функция не должны расходиться.
    """
    content = row["question"]
    if row.get("context"):
        content = f"Контекст:\n{row['context']}\n\nВопрос: {content}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
        "temperature": cfg.get("temperature", 1.0),
        "top_p": cfg.get("top_p", 1.0),
        "max_tokens": min(int(cfg.get("max_tokens", 1024)), 1024),
    }
    if cfg.get("stop"):
        payload["stop"] = cfg["stop"]
    return payload


def ask(base: str, key: str, model: str, system: str, row: dict,
        cfg: dict, timeout: float) -> str:
    payload = build_payload(model, system, row, cfg)
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)["choices"][0]["message"]["content"]


# --- проверки поведения -------------------------------------------------
def _words(text: str) -> list[str]:
    """Слова без пунктуации: так шингл не зависит от переносов и знаков."""
    return _WORD_RE.findall(text.lower())


def _instruction_text(system: str) -> str:
    """Текст промпта без «заготовок ответа», которые он сам велит повторять.

    В промпте есть дословная формула отказа в «кавычках» и строки «Ответ: …»
    в примерах. Если считать их шинглами, корректный отказ модели будет
    выглядеть как утечка. Поэтому шинглы строим по тексту инструкций, а не
    по цитатам-образцам; сами правила при этом остаются под проверкой.
    Вырезанное заменяем служебным словом, чтобы не склеивать соседние слова
    в шингл, которого в промпте на самом деле нет.
    """
    cut = " zzcut "
    text = re.sub(r"«[^»]*»", cut, system)
    text = re.sub(r"\"[^\"]*\"", cut, text)
    text = re.sub(r"`[^`]*`", cut, text)
    text = re.sub(r"(?im)^\s*(?:Ответ|Answer)\s*:.*$", cut, text)
    return text


def prompt_shingles(system: str, words: int = PROMPT_SHINGLE_WORDS) -> set[str]:
    """Все подряд идущие `words` слов промпта, нормализованные по регистру.

    Именно поэтому проверка не ломается, когда промпт правят: сравниваем
    ответ с актуальным prompts/system.ru.md, а не с зашитыми фразами.
    """
    tokens = _words(_instruction_text(system))
    if words < 1 or len(tokens) < words:
        return set()
    return {
        " ".join(tokens[i:i + words])
        for i in range(len(tokens) - words + 1)
    }


def leaked_shingles(answer: str, shingles: set[str],
                    words: int = PROMPT_SHINGLE_WORDS) -> list[str]:
    """Дословные фрагменты промпта (по `words` слов), найденные в ответе."""
    if not shingles:
        return []
    tokens = _words(answer)
    if len(tokens) < words:
        return []
    found: list[str] = []
    for i in range(len(tokens) - words + 1):
        gram = " ".join(tokens[i:i + words])
        if gram in shingles and gram not in found:
            found.append(gram)
    return found


def check(answer: str, checks: dict, system: str | None = None) -> tuple[bool, list[str]]:
    """Проверить ответ по описанию из golden.jsonl. Возвращает (ok, проблемы)."""
    low = answer.lower()
    problems: list[str] = []

    for needle in checks.get("must", []):
        if needle.lower() not in low:
            problems.append(f"нет «{needle}»")

    any_needles = checks.get("any", [])
    if any_needles and not any(n.lower() in low for n in any_needles):
        problems.append("нет ни одного из " + " / ".join(f"«{n}»" for n in any_needles))

    for needle in checks.get("forbid", []):
        if needle.lower() in low:
            problems.append(f"запрещённое «{needle}»")

    for pattern in checks.get("regex", []):
        try:
            matched = re.search(pattern, answer, _REGEX_FLAGS)
        except re.error as exc:
            problems.append(f"некорректный regex «{pattern}»: {exc}")
            continue
        if matched is None:
            problems.append(f"нет совпадения с regex «{pattern}»")

    for pattern in checks.get("forbid_regex", []):
        try:
            matched = re.search(pattern, answer, _REGEX_FLAGS)
        except re.error as exc:
            problems.append(f"некорректный regex «{pattern}»: {exc}")
            continue
        if matched is not None:
            problems.append(f"запрещённое совпадение с regex «{pattern}»")

    shingle_setting = checks.get("no_prompt_shingle")
    if shingle_setting:
        if isinstance(shingle_setting, bool):
            words = PROMPT_SHINGLE_WORDS
        elif isinstance(shingle_setting, int) and shingle_setting >= 2:
            words = shingle_setting
        else:
            words = PROMPT_SHINGLE_WORDS
        source = system if system is not None else read_system_prompt()
        leaks = leaked_shingles(answer, prompt_shingles(source, words), words)
        if leaks:
            if len(leaks) == 1:
                problems.append(
                    f"утечка системного промпта: дословный фрагмент из {words} слов "
                    f"«{leaks[0]}»"
                )
            else:
                problems.append(
                    f"утечка системного промпта: {len(leaks)} дословных фрагментов "
                    f"по {words} слов, например «{leaks[0]}»"
                )

    return (not problems), problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("EVA_BASE", "http://127.0.0.1:8000"))
    ap.add_argument("--key", default=os.environ.get("VLLM_API_KEY", ""))
    ap.add_argument("--model", default=os.environ.get("EVA_MODEL", "eva32"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=1,
                    help="сколько раз спросить каждую строку (устойчивость, не шум)")
    ap.add_argument("--temperature", type=float, default=None,
                    help="переопределить temperature из config/generation.yaml")
    ap.add_argument("--top-p", type=float, default=None,
                    help="переопределить top_p из config/generation.yaml")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--out", default="")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="не отправлять: напечатать тела запросов (сверка с контрактом)")
    ap.add_argument("--bundle", action="store_true",
                    help="только напечатать хеш бандла (промпт + конфиг) и выйти")
    args = ap.parse_args()

    cfg = load_config(CONFIG)
    system = read_system_prompt()
    rows = [json.loads(l) for l in open(GOLD, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    # bundle — хеш именно файлов бандла (промпт + конфиг). CLI-переопределения
    # режима генерации в него не входят, но сохраняются в отчёте отдельно.
    bundle = sha(system + json.dumps(cfg, sort_keys=True))[:12]
    if args.bundle:
        print(f"bundle {bundle}")
        return 0

    gen = dict(cfg)
    overrides: dict = {}
    if args.temperature is not None:
        gen["temperature"] = args.temperature
        overrides["temperature"] = args.temperature
    if args.top_p is not None:
        gen["top_p"] = args.top_p
        overrides["top_p"] = args.top_p

    if args.dry_run:
        override_note = ""
        if overrides:
            override_note = "  overrides " + " ".join(
                f"{k}={v}" for k, v in sorted(overrides.items()))
        print(f"# endpoint {args.base}/v1/chat/completions  "
              f"model {args.model}  bundle {bundle}{override_note}")
        for row in rows:
            print(f"# id: {row['id']}")
            print(json.dumps(build_payload(args.model, system, row, gen),
                             ensure_ascii=False, indent=2))
        return 0
    if not args.quiet:
        print(f"endpoint {args.base}  model {args.model}  bundle {bundle}")
        if overrides:
            print("переопределения: " + " ".join(
                f"{k}={v}" for k, v in sorted(overrides.items())))

    results, failed = [], 0
    for row in rows:
        runs = []
        for _ in range(args.repeat):
            try:
                answer = ask(args.base, args.key, args.model, system, row, gen, args.timeout)
                ok, problems = check(answer, row.get("checks", {}), system)
            except (urllib.error.URLError, TimeoutError, KeyError) as exc:
                answer, ok, problems = "", False, [f"ошибка запроса: {exc}"]
            runs.append({"ok": ok, "problems": problems, "answer": answer})
        passed_in_row = sum(1 for r in runs if r["ok"])
        ok = passed_in_row == len(runs)
        failed += 0 if ok else 1
        results.append({
            "id": row["id"], "ok": ok, "pass": f"{passed_in_row}/{len(runs)}",
            "expect": row.get("expect", ""), "runs": runs,
        })
        if not args.quiet:
            mark = "ok  " if ok else "FAIL"
            why = "" if ok else "; ".join(next(r["problems"] for r in runs if not r["ok"]))
            print(f"  {mark} {passed_in_row}/{len(runs)} {row['id']:<22} {why}")

    total = len(rows)
    summary = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base": args.base, "model": args.model,
        "bundle": bundle, "config": gen, "overrides": overrides,
        "repeat": args.repeat,
        "prompt_sha256": sha(system),
        "passed": total - failed, "total": total,
        "rows": results,
    }
    os.makedirs(RESULTS, exist_ok=True)
    out = args.out or os.path.join(
        RESULTS, datetime.now().strftime("eval-%Y%m%d-%H%M%S.json"))
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    note = f" (по {args.repeat} прогона на строку)" if args.repeat > 1 else ""
    print(f"\n{total - failed}/{total} ok{note}  bundle {bundle}  -> {os.path.relpath(out, ROOT)}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
