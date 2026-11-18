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
import threading
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


def dedupe_file(path: str) -> None:
    """Убрать дубли по вопросу, сохранив порядок; запись атомарная (ADR 0008)."""
    seen: set[str] = set()
    kept: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = row.get("question")
            if q in seen:
                continue
            seen.add(q)
            kept.append(json.dumps(row, ensure_ascii=False))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for line in kept:
            fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    print(f"файл: {len(kept)} уникальных пар", flush=True)


def _read_questions(path: str) -> tuple[set[str], int]:
    """Прочитать готовые вопросы, отбросив оборванный хвост журнала.

    Возвращает (множество вопросов, сколько строк пришлось отбросить).
    """
    done: set[str] = set()
    bad = 0
    with open(path, "rb") as fh:
        for raw in fh:
            try:
                row = json.loads(raw.decode("utf-8"))
                done.add(row["question"])
            except (UnicodeDecodeError, json.JSONDecodeError, KeyError):
                bad += 1
    return done, bad


def _truncate_to_last_good(path: str, bad_lines: int) -> None:
    """Переписать файл, оставив только целые строки (хвост отбрасывается)."""
    kept: list[bytes] = []
    with open(path, "rb") as fh:
        for raw in fh:
            try:
                json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            kept.append(raw if raw.endswith(b"\n") else raw + b"\n")
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.writelines(kept)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    print(f"журнал: {path} переписан ({len(kept)} строк, отброшено {bad_lines})", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=None, help="файл с вопросами (по одному в строке)")
    ap.add_argument("--out", default=os.path.join(ROOT, "distill", "out", "pairs.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mock", action="store_true", help="без сети: учитель-заглушка")
    ap.add_argument("--no-system", action="store_true",
                    help="не передавать prompts/system.ru.md (учим свободному стилю, не нашему)")
    ap.add_argument("--system", default=None,
                    help="путь к системному промпту (по умолчанию prompts/system.ru.md)")
    ap.add_argument("--profile", choices=("prose", "code"), default="prose",
                    help="набор проверок ответа: проза (каталог) или код")
    ap.add_argument("--workers", type=int, default=1,
                    help="параллельных запросов к учителю (по умолчанию 1)")
    ap.add_argument("--incremental", action="store_true",
                    help="писать каждую принятую пару сразу (переживает обрыв)")
    ap.add_argument("--resume", action="store_true",
                    help="пропустить вопросы, на которые уже есть ответы в --out")
    args = ap.parse_args()

    questions = load_questions(args.questions)
    if args.limit:
        questions = questions[: args.limit]
    teacher = Teacher(mock=args.mock)
    system_path = args.system or SYSTEM
    system = None
    if not args.no_system:
        with open(system_path, encoding="utf-8") as fh:
            system = fh.read().strip()

    sink = None
    if args.incremental:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        mode = "a" if args.resume else "w"
        sink = open(args.out, mode, encoding="utf-8")

    def emit(row: dict) -> None:
        if sink is None:
            return
        sink.write(json.dumps(row, ensure_ascii=False) + "\n")
        sink.flush()

    if args.resume and os.path.exists(args.out):
        done, torn = _read_questions(args.out)
        if torn:
            # ADR 0008: журнал дописывания читается с обрезкой оборванного хвоста.
            # Обрыв на середине строки (например, кончилось место) — не повод
            # терять весь файл: отбрасываем хвост и продолжаем.
            _truncate_to_last_good(args.out, torn)
            print(f"журнал: отброшен оборванный хвост ({torn} строк)", flush=True)
        before = len(questions)
        questions = [q for q in questions if q["question"] not in done]
        print(f"возобновление: уже готово {before - len(questions)}, осталось {len(questions)}")

    pairs, rejected = [], []
    tok_in = tok_out = 0
    cost_rub = 0.0
    workers = max(1, args.workers)
    local = threading.local()

    def worker(item: dict) -> dict:
        t = getattr(local, "teacher", None)
        if t is None:
            t = Teacher(mock=args.mock)
            local.teacher = t
        q = item["question"]
        answer = t.ask(q, item.get("context"), system)
        return {
            "item": item,
            "answer": answer,
            "usage": dict(getattr(t, "last_usage", {}) or {}),
            "model": t.model if not args.mock else "mock",
        }

    def to_row(result: dict) -> dict:
        item = result["item"]
        return {
            "question": item["question"],
            "context": item.get("context"),
            "answer": result["answer"],
            "meta": {
                "model": result["model"],
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "flags": filters.check(item["question"], result["answer"],
                                       item.get("context"), profile=args.profile),
                "gen": (item.get("meta") or {}),
            },
        }

    if workers == 1:
        results_iter = map(worker, questions)
        for idx, result in enumerate(results_iter, 1):
            usage = result["usage"]
            tok_in += int(usage.get("prompt_tokens") or 0)
            tok_out += int(usage.get("completion_tokens") or 0)
            cost_rub += float(usage.get("cost_rub") or 0)
            row = to_row(result)
            (rejected if row["meta"]["flags"] else pairs).append(row)
            if not row["meta"]["flags"]:
                emit(row)
            if idx % 50 == 0:
                print(f"  {idx}/{len(questions)}: принято {len(pairs)}, "
                      f"потрачено {cost_rub:.2f} ₽", flush=True)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(worker, item) for item in questions]
            for idx, fut in enumerate(as_completed(futures), 1):
                result = fut.result()
                usage = result["usage"]
                tok_in += int(usage.get("prompt_tokens") or 0)
                tok_out += int(usage.get("completion_tokens") or 0)
                cost_rub += float(usage.get("cost_rub") or 0)
                row = to_row(result)
                (rejected if row["meta"]["flags"] else pairs).append(row)
                if not row["meta"]["flags"]:
                    emit(row)
                if idx % 100 == 0:
                    print(f"  {idx}/{len(questions)}: принято {len(pairs)}, "
                          f"потрачено {cost_rub:.2f} ₽", flush=True)

    if sink is not None:
        sink.close()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.incremental:
        # Пары уже дописаны в файл (в том числе прошлыми запусками). Финальная
        # запись из памяти стёрла бы их — дедуплицируем сам файл на месте.
        dedupe_file(args.out)
    else:
        pairs = filters.dedupe(pairs)
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
