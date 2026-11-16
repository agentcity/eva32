#!/usr/bin/env python3
"""Прогон золотого набора против OpenAI-совместимого эндпоинта EVA32.

Отправляет ровно ту же форму запроса, что и прод: system-промпт из
prompts/system.ru.md, параметры из config/generation.yaml, контекст строкой.

Работает и с vLLM (http://127.0.0.1:8000), и с Ollama (http://127.0.0.1:11434).

    python3 eval/run_eval.py --base http://127.0.0.1:11434 --model eva32-dev
    python3 eval/run_eval.py --base http://127.0.0.1:8000 --model eva32
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD = os.path.join(ROOT, "eval", "golden.jsonl")
PROMPT = os.path.join(ROOT, "prompts", "system.ru.md")
CONFIG = os.path.join(ROOT, "config", "generation.yaml")
RESULTS = os.path.join(ROOT, "eval", "results")


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


def ask(base: str, key: str, model: str, system: str, row: dict,
        cfg: dict, timeout: float) -> str:
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

    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)["choices"][0]["message"]["content"]


def check(answer: str, checks: dict) -> tuple[bool, list[str]]:
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
    return (not problems), problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("EVA_BASE", "http://127.0.0.1:8000"))
    ap.add_argument("--key", default=os.environ.get("VLLM_API_KEY", ""))
    ap.add_argument("--model", default=os.environ.get("EVA_MODEL", "eva32"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=1,
                    help="сколько раз спросить каждую строку (устойчивость, не шум)")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--out", default="")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--bundle", action="store_true",
                    help="только напечатать хеш бандла (промпт + конфиг) и выйти")
    args = ap.parse_args()

    cfg = load_config(CONFIG)
    with open(PROMPT, encoding="utf-8") as fh:
        system = fh.read().strip()
    rows = [json.loads(l) for l in open(GOLD, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    bundle = sha(system + json.dumps(cfg, sort_keys=True))[:12]
    if args.bundle:
        print(f"bundle {bundle}")
        return 0
    if not args.quiet:
        print(f"endpoint {args.base}  model {args.model}  bundle {bundle}")

    results, failed = [], 0
    for row in rows:
        runs = []
        for _ in range(args.repeat):
            try:
                answer = ask(args.base, args.key, args.model, system, row, cfg, args.timeout)
                ok, problems = check(answer, row.get("checks", {}))
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
            print(f"  {mark} {passed_in_row}/{len(runs)} {row['id']:<18} {why}")

    total = len(rows)
    summary = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base": args.base, "model": args.model,
        "bundle": bundle, "config": cfg, "repeat": args.repeat,
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
