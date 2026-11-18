#!/usr/bin/env python3
"""Проверка реестра задач `ai/tasks/index.json` и файлов задач.

Что проверяется:
  * `index.json` — валидный JSON-массив записей;
  * у каждой записи есть `id`, `title`, `status`, `file`;
  * `status` — один из статусов, описанных в `ai/README.md`;
  * `file` существует в `ai/tasks/`, во фронтматтере есть `task_id`;
  * `task_id` в файле совпадает с `id` записи, номера не повторяются;
  * в файле есть разделы «## Критерии приёмки» и «## Отчёт исполнителя».

Только стандартная библиотека: YAML-парсер не нужен, фронтматтер разбирается
построчно (нам нужен один скаляр `task_id`).

Запуск:
    python3 tools/check_tasks.py

Код выхода: 0 — реестр согласован, 1 — есть ошибки (список печатается).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = ROOT / "ai" / "tasks"
REGISTRY_PATH = TASKS_DIR / "index.json"

REQUIRED_FIELDS = ("id", "title", "status", "file")
VALID_STATUSES = ("pending", "in_progress", "review", "done", "blocked")
REQUIRED_SECTIONS = ("Критерии приёмки", "Отчёт исполнителя")
ID_RE = re.compile(r"^\d{4}$")

_FRONTMATTER_SCALAR = re.compile(
    r"^\s*(?:\"([^\"]*)\"|'([^']*)'|([^#]*?))\s*(?:#.*)?$"
)


def display(path: Path) -> str:
    """Путь для сообщений: относительно корня репозитория, если возможно."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")


def load_registry(path: Path, errors: list[str]) -> list | None:
    """Читает реестр. Возвращает список записей или None (ошибка уже записана)."""
    try:
        raw = read_text(path)
    except FileNotFoundError:
        errors.append(f"нет файла реестра: {display(path)}")
        return None
    except OSError as exc:
        errors.append(f"не читается {display(path)}: {exc}")
        return None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        errors.append(
            f"{display(path)}: невалидный JSON — {exc.msg} "
            f"(строка {exc.lineno}, колонка {exc.colno})"
        )
        return None

    if not isinstance(data, list):
        errors.append(
            f"{display(path)}: ожидался JSON-массив записей, "
            f"получено {type(data).__name__}"
        )
        return None
    return data


def split_frontmatter(text: str) -> tuple[list[str] | None, str | None]:
    """Возвращает (строки фронтматтера, None) или (None, причина)."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None, "файл не начинается с `---` — фронтматтера нет"
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return lines[1:i], None
    return None, "фронтматтер не закрыт строкой `---`"


def frontmatter_scalar(frontmatter: list[str], key: str) -> str | None:
    """Достаёт скаляр верхнего уровня из строк фронтматтера."""
    pattern = re.compile(r"^" + re.escape(key) + r"\s*:\s*(.*)$")
    for line in frontmatter:
        if line[:1].isspace():  # вложенные ключи (scope:) не наши
            continue
        match = pattern.match(line)
        if not match:
            continue
        value = match.group(1)
        scalar = _FRONTMATTER_SCALAR.match(value)
        if not scalar:
            return ""
        return next(
            (group for group in scalar.groups() if group is not None), ""
        ).strip()
    return None


def has_section(text: str, title: str) -> bool:
    pattern = re.compile(
        r"^##[ \t]+" + re.escape(title) + r"[ \t]*$", re.MULTILINE
    )
    return pattern.search(text) is not None


def plural(count: int, one: str, few: str, many: str) -> str:
    """Русская форма слова для числа: 1 ошибка, 2 ошибки, 5 ошибок."""
    if count % 10 == 1 and count % 100 != 11:
        return one
    if 2 <= count % 10 <= 4 and not 12 <= count % 100 <= 14:
        return few
    return many


def check_entry(
    index: int,
    entry: object,
    seen_ids: dict[str, int],
    seen_files: dict[str, int],
    errors: list[str],
) -> None:
    where = f"запись #{index}"
    if not isinstance(entry, dict):
        errors.append(
            f"{where}: ожидался объект, получено {type(entry).__name__}"
        )
        return

    task_id = entry.get("id")
    if isinstance(task_id, str) and task_id.strip():
        where = f"задача {task_id.strip()}"

    for field in REQUIRED_FIELDS:
        value = entry.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{where}: нет непустого поля `{field}`")

    if not isinstance(task_id, str) or not task_id.strip():
        return  # без id дальше проверять нечего

    task_id = task_id.strip()
    if not ID_RE.match(task_id):
        errors.append(
            f"{where}: id `{task_id}` не в формате NNNN (четыре цифры)"
        )

    if task_id in seen_ids:
        errors.append(
            f"номера задач не уникальны: id `{task_id}` уже встречался "
            f"в записи #{seen_ids[task_id]}"
        )
    else:
        seen_ids[task_id] = index

    status = entry.get("status")
    if isinstance(status, str) and status.strip():
        if status.strip() not in VALID_STATUSES:
            errors.append(
                f"{where}: статус `{status.strip()}` не из списка "
                f"{', '.join(VALID_STATUSES)}"
            )

    file_name = entry.get("file")
    if not isinstance(file_name, str) or not file_name.strip():
        return

    task_file = Path(file_name.strip())
    if not task_file.is_absolute():
        task_file = TASKS_DIR / task_file
    if not task_file.exists():
        errors.append(f"{where}: файл задачи не найден — {display(task_file)}")
        return
    if not task_file.is_file():
        errors.append(f"{where}: путь задачи не файл — {display(task_file)}")
        return

    file_key = str(task_file.resolve())
    if file_key in seen_files:
        errors.append(
            f"{where}: файл {display(task_file)} уже указан в "
            f"записи #{seen_files[file_key]}"
        )
    else:
        seen_files[file_key] = index

    try:
        text = read_text(task_file)
    except OSError as exc:
        errors.append(f"{where}: не читается {display(task_file)}: {exc}")
        return

    frontmatter, problem = split_frontmatter(text)
    if frontmatter is None:
        errors.append(f"{where}: {display(task_file)} — {problem}")
    else:
        file_task_id = frontmatter_scalar(frontmatter, "task_id")
        if file_task_id is None or file_task_id == "":
            errors.append(
                f"{where}: во фронтматтере {display(task_file)} нет `task_id`"
            )
        elif file_task_id != task_id:
            errors.append(
                f"{where}: `task_id: {file_task_id}` в "
                f"{display(task_file)} не совпадает с id записи `{task_id}`"
            )

    for title in REQUIRED_SECTIONS:
        if not has_section(text, title):
            errors.append(
                f"{where}: в {display(task_file)} нет раздела `## {title}`"
            )


def main() -> int:
    errors: list[str] = []
    print(f"== check_tasks: {display(REGISTRY_PATH)} ==")

    registry = load_registry(REGISTRY_PATH, errors)
    if registry is None:
        for error in errors:
            print(f"✗ {error}")
        print("\nFAIL: реестр задач не прочитан.")
        return 1

    seen_ids: dict[str, int] = {}
    seen_files: dict[str, int] = {}
    for index, entry in enumerate(registry, start=1):
        check_entry(index, entry, seen_ids, seen_files, errors)

    if errors:
        for error in errors:
            print(f"✗ {error}")
        word = plural(len(errors), "ошибка", "ошибки", "ошибок")
        print(f"\nFAIL: {len(errors)} {word}, реестр задач не согласован.")
        return 1

    print(f"OK: {len(registry)} задач, id уникальны, файлы и разделы на месте.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
