#!/usr/bin/env python3
"""Свежесть документации и целостность ссылок.

Что проверяется:

* в каждом документе (`docs/**/*.md`, кроме `docs/adr/`, и корневой
  `README.md`) есть строка вида «Актуально на ГГГГ-ММ-ДД»;
* дата не в будущем и не старше `--max-age` дней (по умолчанию 30);
* markdown-ссылки и пути репозитория в обратных кавычках указывают на
  существующие файлы (`--no-links` отключает).

ADR исключены из проверки свежести осознанно: у них дата — это дата решения,
а не «актуально на». Поэтому же они не переписываются задним числом.

Скрипт работает без сети и без зависимостей, только stdlib. При любой
проблеме — exit 1; список проблем печатается целиком, чтобы чинить за один
заход.

Запуск::

    python3 tools/check_docs_dates.py
    python3 tools/check_docs_dates.py --today 2026-11-24      # для тестов
    python3 tools/check_docs_dates.py --max-age 60 --no-links
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Строка «Актуально на 2026-11-24», в том числе **жирным** и в цитате:
#   > Актуально на 2026-11-24.
#   **Актуально на 2026-11-24.**
ACTUAL_RE = re.compile(
    r"Актуально\s+на\s+\**\s*(\d{4})-(\d{2})-(\d{2})", re.IGNORECASE
)

MD_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
TICK_RE = re.compile(r"`([^`\n]+)`")

# Пути в обратных кавычках проверяем только от известных корней репозитория:
# имена функций, системные заголовки (sys/mman.h) и артефакты времени
# выполнения (`runs/…`, `data/…` — gitignored) — не ссылки документации.
REPO_ROOTS = {
    "src", "docs", "tools", "tests", "qat", "distill", "eva", "scripts",
    "prompts", "ai", "demo", "config",
}
LINK_EXTS = {
    ".md", ".c", ".h", ".py", ".json", ".vtm", ".trit", ".bin", ".sh",
    ".txt", ".toml", ".yml", ".yaml", ".jsonl", ".png",
}


class Problem(Exception):
    """Одна проблема документа; собирается в общий список."""


def docs_files(root: Path) -> list[Path]:
    """Документы, для которых требуется строка «Актуально на».

    `docs/adr/` исключён: дата ADR — дата решения, а не свежесть. Корневой
    README входит в проверку: он тоже обещает «состояние на сегодня».
    """
    files: list[Path] = []
    readme = root / "README.md"
    if readme.is_file():
        files.append(readme)
    docs = root / "docs"
    if docs.is_dir():
        for path in sorted(docs.rglob("*.md")):
            rel = path.relative_to(docs)
            if rel.parts and rel.parts[0] == "adr":
                continue
            files.append(path)
    return files


def parse_actual_dates(text: str) -> list[date]:
    """Все даты из строк «Актуально на …»; некорректная дата — ValueError."""
    dates: list[date] = []
    for year, month, day in ACTUAL_RE.findall(text):
        try:
            dates.append(date(int(year), int(month), int(day)))
        except ValueError as exc:
            raise ValueError(
                f"некорректная дата {year}-{month}-{day}: {exc}"
            ) from exc
    return dates


def actual_date(text: str) -> date:
    """Ровно одна дата документа; иначе Problem."""
    try:
        dates = parse_actual_dates(text)
    except ValueError as exc:
        raise Problem(str(exc)) from exc
    if not dates:
        raise Problem("нет строки «Актуально на ГГГГ-ММ-ДД»")
    unique = set(dates)
    if len(unique) > 1:
        shown = ", ".join(sorted(d.isoformat() for d in unique))
        raise Problem(f"несколько разных дат «Актуально на»: {shown}")
    return dates[0]


def check_date(
    path: Path, text: str, today: date, max_age: int
) -> tuple[date, list[str]]:
    """Проблемы даты одного документа; возраст — в днях."""
    found = actual_date(text)
    problems: list[str] = []
    age = (today - found).days
    if age < 0:
        problems.append(
            f"дата {found.isoformat()} в будущем (сегодня {today.isoformat()})"
        )
    elif age > max_age:
        problems.append(
            f"документ устарел: {found.isoformat()}, {age} дн. "
            f"> {max_age}"
        )
    return found, problems


def _iter_link_targets(text: str) -> list[str]:
    """Цели ссылок: markdown-ссылки и пути репозитория в обратных кавычках."""
    # Ссылки внутри блоков кода — это примеры, а не ссылки.
    plain = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    targets = list(MD_LINK_RE.findall(plain))
    for token in TICK_RE.findall(text):
        token = token.strip()
        token = token.split("::", 1)[0]
        if "/" not in token or " " in token or "*" in token:
            continue
        if token.startswith(("-", "|", "$", ".", "~", "/", "http://", "https://")):
            continue
        root_component = token.split("/", 1)[0]
        if root_component not in REPO_ROOTS:
            continue
        if Path(token).suffix.lower() not in LINK_EXTS:
            continue
        targets.append(token)
    return targets


def _resolve(doc: Path, target: str, root: Path) -> Path | None:
    """Существующий путь или None. Цели вне корня репозитория не проверяем."""
    path = (doc.parent / target).resolve()
    if path.exists():
        return path
    path = (root / target).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return None
    return path if path.exists() else None


def check_links(doc: Path, text: str, root: Path) -> tuple[int, list[str]]:
    """Проверенные ссылки и проблемы. Внешние ссылки и якоря пропускаются."""
    checked = 0
    problems: list[str] = []
    for target in _iter_link_targets(text):
        target = target.strip()
        if not target or target.startswith(("#", "http://", "https://", "mailto:")):
            continue
        target = target.split("#", 1)[0]
        if not target:
            continue
        if target.startswith("/"):
            problems.append(f"абсолютный путь {target!r} не переносим")
            continue
        checked += 1
        if _resolve(doc, target, root) is None:
            problems.append(f"битая ссылка {target!r}")
    return checked, problems


def check(
    root: Path, today: date, max_age: int = 30, links: bool = True
) -> tuple[int, int, list[str]]:
    """(документов, проверенных ссылок, проблем) для дерева `root`."""
    documents = docs_files(root)
    problems: list[str] = []
    if not documents:
        return 0, 0, [f"не найдено ни одного документа в {root / 'docs'}"]

    checked_links = 0
    for path in documents:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            problems.append(f"{path}: не читается ({exc})")
            continue
        label = path.relative_to(root)
        try:
            _, date_problems = check_date(path, text, today, max_age)
        except Problem as exc:
            problems.append(f"{label}: {exc}")
        else:
            problems.extend(f"{label}: {p}" for p in date_problems)
        if links:
            checked, link_problems = check_links(path, text, root)
            checked_links += checked
            problems.extend(f"{label}: {p}" for p in link_problems)
    return len(documents), checked_links, problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Проверить свежесть docs/ и отсутствие битых ссылок."
    )
    parser.add_argument(
        "--root", type=Path, default=ROOT, help="корень репозитория"
    )
    parser.add_argument(
        "--today",
        type=date.fromisoformat,
        default=None,
        help="сегодняшняя дата ГГГГ-ММ-ДД (по умолчанию — системная)",
    )
    parser.add_argument(
        "--max-age", type=int, default=30, help="предельный возраст, дней"
    )
    parser.add_argument(
        "--no-links", action="store_true", help="не проверять ссылки"
    )
    args = parser.parse_args(argv)

    today = args.today or date.today()
    documents, checked_links, problems = check(
        args.root, today, args.max_age, links=not args.no_links
    )

    print(
        f"docs freshness: {documents} документов, max-age {args.max_age} дн., "
        f"сегодня {today.isoformat()}"
    )
    if problems:
        for problem in problems:
            print(f"ERROR {problem}", file=sys.stderr)
        print(f"FAIL: {len(problems)} проблем(ы)", file=sys.stderr)
        return 1
    if not args.no_links:
        print(f"ссылки: проверено {checked_links}, все существуют")
    print("OK: даты и ссылки в порядке")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
