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


def load_questions(path: str | None) -> list[str]:
    if path:
        with open(path, encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    out: list[str] = []
    with open(GOLD, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line)["question"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=None, help="файл с вопросами (по одному в строке)")
    ap.add_argument("--out", default=os.path.join(ROOT, "distill", "out", "pairs.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mock", action="store_true", help="без сети: учитель-заглушка")
    args = ap.parse_args()

    questions = load_questions(args.questions)
    if args.limit:
        questions = questions[: args.limit]
    teacher = Teacher(mock=args.mock)

    pairs, rejected = [], []
    for q in questions:
        answer = teacher.ask(q)
        problems = filters.check(q, answer)
        row = {
            "question": q,
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
    print(f"-> {os.path.relpath(args.out, ROOT)}")
    return 0 if pairs else 1


if __name__ == "__main__":
    sys.exit(main())
