#!/usr/bin/env python3
"""Сбор пар «вопрос → ответ» для дистилляции.

    python3 distill/run.py --mock --limit 10          # проверка конвейера без сети
    TEACHER_BASE=... TEACHER_KEY=... python3 distill/run.py \\
        --questions distill/questions.txt --out distill/out/pairs.jsonl

Пайплайн: вопросы → учитель → фильтры → дедуп → JSONL + отчёт.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distill import filters                       # noqa: E402
from distill.teacher import Teacher               # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD = os.path.join(ROOT, "eval", "golden.jsonl")
SYSTEM = os.path.join(ROOT, "prompts", "system.ru.md")


def load_questions(path: str | None) -> list[dict]:
    """Вопросы с необязательным контекстом.

    .jsonl — строки {"question": ..., "context": ...}; .txt — по вопросу в строке.
    Золотой набор (eval/golden.jsonl) уже содержит context у части строк.
    """
    src = path or GOLD
    items: list[dict] = []
    with open(src, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("{"):
                row = json.loads(line)
                items.append({"question": row["question"], "context": row.get("context")})
            else:
                items.append({"question": line, "context": None})
    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=None, help="файл с вопросами (по одному в строке)")
    ap.add_argument("--out", default=os.path.join(ROOT, "distill", "out", "pairs.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mock", action="store_true", help="без сети: учитель-заглушка")
    ap.add_argument("--no-system", action="store_true",
                    help="не передавать prompts/system.ru.md (учим свободному стилю, не нашему)")
    args = ap.parse_args()

    questions = load_questions(args.questions)
    if args.limit:
        questions = questions[: args.limit]
    teacher = Teacher(mock=args.mock)
    system = None
    if not args.no_system:
        with open(SYSTEM, encoding="utf-8") as fh:
            system = fh.read().strip()

    pairs, rejected = [], []
    tok_in = tok_out = 0
    cost_rub = 0.0
    for item in questions:
        q = item["question"]
        answer = teacher.ask(q, item.get("context"), system)
        tok_in += int(teacher.last_usage.get("prompt_tokens") or 0)
        tok_out += int(teacher.last_usage.get("completion_tokens") or 0)
        cost_rub += float(teacher.last_usage.get("cost_rub") or 0)
        problems = filters.check(q, answer, item.get("context"))
        row = {
            "question": q,
            "context": item.get("context"),
            "answer": answer,
            "meta": {
                "model": teacher.model if not args.mock else "mock",
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "flags": problems,
            },
        }
        (rejected if problems else pairs).append(row)

    pairs = filters.dedupe(pairs)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    by_reason: dict[str, int] = {}
    for r in rejected:
        for f in r["meta"]["flags"]:
            by_reason[f] = by_reason.get(f, 0) + 1

    total = len(questions)
    print(f"вопросов: {total}")
    print(f"принято:  {len(pairs)}")
    print(f"отброшено: {len(rejected)}  " + (", ".join(f"{k}={v}" for k, v in by_reason.items()) or ""))
    if tok_in or tok_out:
        print(f"токены: вход {tok_in}, выход {tok_out}")
    if cost_rub:
        print(f"потрачено: {cost_rub:.2f} ₽  (за запрос: {cost_rub / max(total, 1):.4f} ₽)")
    print(f"-> {os.path.relpath(args.out, ROOT)}")
    return 0 if pairs else 1


if __name__ == "__main__":
    sys.exit(main())
