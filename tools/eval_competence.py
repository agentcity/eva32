#!/usr/bin/env python3
"""Оценка компетенций ассистента: golden — про умение, а не про прайс.

Прайс и каталог живут в `eval/golden_catalog.jsonl` и проверяются регрессией RAG/скиллов.
Здесь — то, что должно работать у **любого** бизнеса: рассуждение, уточняющие вопросы,
маркетинговые тексты, общение с клиентом, правила русского языка, поведение и границы.

Проверки структурные, поэтому не требуют модели: файл ответов можно получить от живого
бэкенда позже, а пока оценка работает как исполняемая спецификация.

    python3 tools/eval_competence.py --list                  # что проверяем
    python3 tools/eval_competence.py --answers a.jsonl       # оценить ответы
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "eval" / "golden_competence.jsonl"
CATALOG = ROOT / "data" / "catalog.json"

_CYR = re.compile(r"[а-яА-ЯёЁ]")
_NUM = re.compile(r"\d[\d\s]{0,9}")
#: Уточнение не обязано заканчиваться «?»: «Уточните тему и объём» — тоже вопрос.
_QUESTION = re.compile(
    r"\?|\b(?:уточнит[еь]|уточни|сообщит[еь]|сообщи|скажит[еь]|скажи|"
    r"подскажит[еь]|подскажи|назови|назовите|опиши|опишите|"
    r"какой|какая|какие|каком|что именно|"
    r"для какой|на какую|сколько|где|когда|кто|зачем|почему)\b",
    re.I,
)

#: Как сравнивать «обязательное слово»: по основе, а не буквально — иначе
#: «вакансия» не совпадает с «вакансии», а «приказ» с «приказу».
_STEM_MIN = 4


def _stem(text: str) -> str:
    return text[: max(_STEM_MIN, len(text) - 2)]

#: Допустимые названия проверок — опечатка в golden должна падать, а не молча игнорироваться.
KNOWN_CHECKS = {
    "russian", "min_words", "max_words", "asks_question",
    "must_contain", "must_contain_any", "must_not_contain", "no_invented_prices",
    "forbid_numbers_from_catalog", "forbid",
}


def load_golden(path: pathlib.Path = GOLDEN) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def catalog_numbers() -> set[str]:
    if not CATALOG.exists():
        return set()
    data = json.loads(CATALOG.read_text(encoding="utf-8"))
    return {str(item["price_rub"]) for item in data if item.get("price_rub")}


def check_row(row: dict, answer: str, *, numbers: set[str] | None = None) -> list[str]:
    """Вернуть список проваленных проверок (пусто — строка прошла)."""
    numbers = catalog_numbers() if numbers is None else numbers
    checks = row.get("checks", {})
    failures: list[str] = []
    words = len(answer.split())
    low = answer.lower()

    if checks.get("russian") and len(_CYR.findall(answer)) < 10:
        failures.append("не по-русски")
    if "min_words" in checks and words < checks["min_words"]:
        failures.append(f"коротко ({words} < {checks['min_words']})")
    if "max_words" in checks and words > checks["max_words"]:
        failures.append(f"длинно ({words} > {checks['max_words']})")
    if checks.get("asks_question") and not _QUESTION.search(answer):
        failures.append("не задан уточняющий вопрос")
    for token in checks.get("must_contain", []):
        if not re.search(rf"\b{re.escape(_stem(token.lower()))}", low):
            failures.append(f"нет «{token}»")
    # Синонимы: «врач» или «доктор» или «специалист» — смысл один, требуем любой.
    for group in checks.get("must_contain_any", []):
        if not any(re.search(rf"\b{re.escape(_stem(alt.lower()))}", low) for alt in group):
            failures.append("нет ни одного из: " + ", ".join(group))
    for token in checks.get("must_not_contain", []):
        if token.lower() in low:
            failures.append(f"есть запрещённое «{token}»")
    for token in checks.get("forbid", []):
        if token.lower() in low:
            failures.append(f"есть запрещённое «{token}»")
    if checks.get("forbid_numbers_from_catalog"):
        found = {n for n in numbers if re.search(rf"(?<!\d){n}(?!\d)", answer)}
        if found:
            failures.append(f"цены из каталога: {sorted(found)}")
    if checks.get("no_invented_prices"):
        # Любая цена в рублях без контекста — выдумка: в этом наборе его нет.
        if re.search(r"\d[\d\s]{0,9}\s*(?:₽|руб)", low):
            failures.append("выдуманная цена")
    return failures


def evaluate(rows: list[dict], answers: dict[str, str]) -> dict:
    numbers = catalog_numbers()
    per_cat: dict[str, dict[str, int]] = {}
    failures: list[dict] = []
    for row in rows:
        cat = row["category"]
        per_cat.setdefault(cat, {"всего": 0, "прошли": 0})
        per_cat[cat]["всего"] += 1
        answer = answers.get(row["question"])
        if answer is None:
            failures.append({"question": row["question"], "category": cat,
                             "failures": ["нет ответа"]})
            continue
        bad = check_row(row, answer, numbers=numbers)
        if bad:
            failures.append({"question": row["question"], "category": cat, "failures": bad})
        else:
            per_cat[cat]["прошли"] += 1
    return {"per_category": per_cat, "failures": failures}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--answers", type=pathlib.Path)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    rows = load_golden()
    unknown = {k for row in rows for k in row.get("checks", {})} - KNOWN_CHECKS
    if unknown:
        print(f"неизвестные проверки в golden: {sorted(unknown)}", file=sys.stderr)
        return 2

    if args.list or not args.answers:
        cats: dict[str, list[str]] = {}
        for row in rows:
            cats.setdefault(row["category"], []).append(row["question"])
        for cat, questions in cats.items():
            print(f"\n{cat} ({len(questions)}):")
            for q in questions:
                print("  -", q)
        print(f"\nвсего строк: {len(rows)}")
        return 0

    answers: dict[str, str] = {}
    for line in args.answers.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        answers[row["question"]] = row.get("answer", "")

    report = evaluate(rows, answers)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for cat, stat in report["per_category"].items():
            share = stat["прошли"] / stat["всего"] * 100 if stat["всего"] else 0
            print(f"{cat:12s} {stat['прошли']:3d}/{stat['всего']:<3d} = {share:5.1f}%")
        if report["failures"]:
            print("\nпровалы:")
            for row in report["failures"][:20]:
                print(f"  [{row['category']}] {row['question'][:60]} — {', '.join(row['failures'])}")
    return 0 if not report["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
