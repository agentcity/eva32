#!/usr/bin/env python3
"""Перевести английские пары «вопрос → ответ» на русский учителем.

Зачем: готовых русских код-датасетов мало, а английских — много. Перевод дешевле
генерации и сохраняет разнообразие задач, которого у нас нет. Учимся на ответах
**одного** учителя, поэтому стиль не разъезжается на несколько моделей.

    PYTHONPATH=.pylibs python3.13 tools/translate_pairs.py \\
        --in distill/pulled/codealpaca.jsonl --out distill/out/codealpaca_ru.jsonl --limit 50

Что сохраняется: блоки кода, имена функций, API, команды, числа. Что переводится:
весь естественный язык. Проверки после перевода — в `_check`.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from distill.teacher import Teacher  # noqa: E402
from distill import quality  # noqa: E402

SYSTEM = """Ты переводчик технических текстов на русский язык.

Правила:
1. Переводи ТОЛЬКО естественный язык. Блоки кода (```...```), имена функций, переменных,
   команд, флагов, API, пути, числа и названия технологий оставляй без изменений.
2. Ничего не объясняй, не добавляй и не убирай по смыслу — это перевод, а не ответ.
3. Термины: «функция», «массив», «цикл», «строка» и т. п. — по-русски; названия
   библиотек и синтаксис — как есть.
4. Формат ответа — ровно два блока:
ВОПРОС: <перевод вопроса>
ОТВЕТ: <перевод ответа>
"""

_CYR = re.compile(r"[а-яА-ЯёЁ]")
_FENCE = re.compile(r"```")


def _check(question_en: str, answer_en: str, question_ru: str, answer_ru: str) -> str | None:
    """Вернуть причину отказа или None, если перевод годен."""
    if not _CYR.search(question_ru) or len(_CYR.findall(question_ru)) < 10:
        return "вопрос не переведён"
    # Ответ может быть почти целиком кодом: требуем либо 15 русских букв, либо
    # доли кириллицы среди букв — код без слов не должен считаться непереведённым.
    cyr = len(_CYR.findall(answer_ru))
    letters = len(re.findall(r"[A-Za-zА-Яа-яЁё]", answer_ru))
    if cyr < 15 and (letters == 0 or cyr / letters < 0.25):
        return "ответ не переведён"
    if _FENCE.findall(answer_en) and len(_FENCE.findall(answer_en)) != len(_FENCE.findall(answer_ru)):
        return "потеряны блоки кода"
    # Перевод не должен быть вдвое длиннее оригинала: это признак объяснений.
    if len(answer_ru) > 2.2 * max(1, len(answer_en)) + 200:
        return "перевод раздут"
    if re.search(r"\b(вот перевод|перевод:|вот переведённ)", answer_ru, re.I):
        return "болтовня вокруг перевода"
    return None


def _parse(text: str) -> tuple[str, str] | None:
    m = re.search(r"ВОПРОС:\s*(.*?)\s*ОТВЕТ:\s*(.*)", text, re.S)
    if not m:
        return None
    return m.group(1).strip(), m.group(2).strip()


def translate_one(teacher: Teacher, question: str, answer: str,
                  max_tokens: int = 3000) -> tuple[str, str] | None:
    prompt = (
        "Переведи на русский пару «вопрос → ответ».\n\n"
        f"ВОПРОС (en):\n{question}\n\nОТВЕТ (en):\n{answer}"
    )
    raw = teacher.ask(prompt, system=SYSTEM, max_tokens=max_tokens)
    parsed = _parse(raw)
    if parsed is None:
        return None
    return parsed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)

    src = pathlib.Path(args.src)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    if args.resume and out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["question"])
            except (json.JSONDecodeError, KeyError):
                continue

    items: list[dict] = []
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        q = str(row.get("question") or "").strip()
        a = str(row.get("answer") or "").strip()
        if not q or not a or q in done:
            continue
        items.append({"question": q, "answer": a})
        if args.limit and len(items) >= args.limit:
            break

    if not items:
        print("нечего переводить", file=sys.stderr)
        return 1

    teacher = Teacher()
    lock = threading.Lock()
    stats = {"принято": 0, "отброшено": 0, "причины": {}}
    cost = 0.0
    sink = out.open("a" if args.resume else "w", encoding="utf-8")

    def work(item: dict) -> None:
        nonlocal cost
        try:
            result = translate_one(teacher, item["question"], item["answer"])
        except Exception as exc:  # noqa: BLE001 — один сбой не должен ронять прогон
            with lock:
                stats["причины"][f"ошибка: {type(exc).__name__}"] = (
                    stats["причины"].get(f"ошибка: {type(exc).__name__}", 0) + 1
                )
            return
        if result is None:
            reason = "не разобран ответ учителя"
        else:
            question_ru, answer_ru = result
            reason = _check(item["question"], item["answer"], question_ru, answer_ru)
        with lock:
            if reason:
                stats["отброшено"] += 1
                stats["причины"][reason] = stats["причины"].get(reason, 0) + 1
                return
            if not quality.accept_pair(question_ru, answer_ru):
                stats["отброшено"] += 1
                stats["причины"]["фильтр качества"] = (
                    stats["причины"].get("фильтр качества", 0) + 1
                )
                return
            sink.write(json.dumps({
                "question": question_ru,
                "answer": answer_ru,
                "meta": {"source": src.stem, "field": "translated", "language": "ru"},
            }, ensure_ascii=False) + "\n")
            sink.flush()
            stats["принято"] += 1
        usage = teacher.last_usage or {}
        with lock:
            cost += 0.0  # цена считается учителем отдельно; здесь только токены

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        list(pool.map(work, items))

    sink.close()
    print(json.dumps({"принято": stats["принято"], "отброшено": stats["отброшено"],
                      "причины": stats["причины"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
