#!/usr/bin/env python3
"""Сгенерировать банк вопросов из контекстов — чтобы был объём для сбора пар.

    python3 distill/gen_questions.py --per-context 30 --out distill/questions.jsonl

Берёт контексты из eval/golden.jsonl (строки с полем context), просит учителя
придумать по ним вопросы и сохраняет пары {question, context}.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distill import filters                       # noqa: E402
from distill.teacher import Teacher               # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD = os.path.join(ROOT, "eval", "golden.jsonl")

PROMPT = """По фрагменту данных о компании придумай {k} разных вопросов клиента,
на которые можно ответить только по этому фрагменту. Вопросы должны быть разными
по формулировке: с опечатками не надо, но перефразируй свободно.
Выведи только вопросы, по одному в строке, без нумерации и пояснений.

Фрагмент:
{context}"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-context", type=int, default=30)
    ap.add_argument("--out", default=os.path.join(ROOT, "distill", "questions.jsonl"))
    args = ap.parse_args()

    contexts = []
    with open(GOLD, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("context"):
                contexts.append(row["context"])
    if not contexts:
        print("в золотом наборе нет строк с контекстом", file=sys.stderr)
        return 1

    teacher = Teacher()
    items: list[dict] = []
    for ctx in contexts:
        raw = teacher.ask(PROMPT.format(k=args.per_context, context=ctx), max_tokens=2500)
        for line in raw.splitlines():
            q = re.sub(r"^\s*[-*\d.)]+\s*", "", line).strip()
            if 8 <= len(q) <= 300 and "?" in q:
                items.append({"question": q, "context": ctx})

    items = filters.dedupe(items)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"контекстов: {len(contexts)}, вопросов: {len(items)} -> {os.path.relpath(args.out, ROOT)}")
    return 0 if items else 1


if __name__ == "__main__":
    sys.exit(main())
