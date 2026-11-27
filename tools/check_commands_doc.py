#!/usr/bin/env python3
"""Сверка справочника команд Евы с реестром плагинов.

`docs/commands.md` — единственная страница, где пользователь видит, что Ева
умеет, на какие фразы и как отвечает. Страница быстро врёт, если её не
проверять: появился плагин — фразы в справочнике нет, и «источник правды»
снова только код.

Что проверяется:

* `docs/commands.md` существует;
* в нём есть строка «Актуально на ГГГГ-ММ-ДД» (правило общее с
  `tools/check_docs_dates.py`: у документа ровно одна такая дата);
* упомянуты все обязательные id плагинов и все id, включённые в
  `config/plugins.json`: новый включённый плагин без раздела в справочнике —
  красный CI, а не тихая дыра в документации;
* упомянуты ключевые фразы (`стоп`, `сказк`, `курс`, `акци`, `музык`,
  `радио`, `громче`, `номер`, `индекс`) — по ним видно, что разделы про
  сказки, музыку и данные не выпали целиком;
* есть обязательные разделы (быстрый старт, сказки, музыка, данные,
  обычный разговор, чего не умеет, деградация, настройки);
* в документе есть таблицы «фраза → что произойдёт → как ответит», а не
  только проза;
* страница есть в индексе `docs/README.md` ссылкой, а не упоминанием в
  тексте.

Скрипт работает без сети и без зависимостей, только stdlib. При любой
проблеме — exit 1; список проблем печатается целиком, чтобы чинить за один
заход.

Запуск::

    python3 tools/check_commands_doc.py
    python3 tools/check_commands_doc.py --doc docs/commands.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOC = Path("docs/commands.md")
INDEX_PATH = Path("docs/README.md")
CONFIG_PATH = Path("config/plugins.json")

#: Id плагинов, которые справочник обязан упоминать всегда. Это список из
#: задачи 0233: три источника данных, сказки, музыка и два фильтра речи.
REQUIRED_PLUGIN_IDS = (
    "time",
    "currency",
    "stocks",
    "fairy-tales",
    "music",
    "ru-numbers",
    "ru-gender",
)

#: Фразы-маркеры разделов: без них страница перестала быть справочником
#: команд, даже если формально существует.
KEY_PHRASES = (
    "стоп",
    "сказк",
    "курс",
    "акци",
    "музык",
    "радио",
    "громче",
    "номер",
    "индекс",
)

#: Разделы из задачи. Ищем подстрокой в заголовках `##`/`###`: «Чего Ева не
#: умеет» может быть и «Чего Ева не умеет», и «Чего не умеет» — но не
#: «Чего-то ещё».
REQUIRED_SECTIONS = (
    "Быстрый старт",
    "Сказки",
    "Музыка",
    "Данные",
    "Обычный разговор",
    "не умеет",
    "Деградация",
    "Настройки",
)

#: Строка «Актуально на 2026-11-27», в том числе **жирным** и в цитате.
ACTUAL_RE = re.compile(
    r"Актуально\s+на\s+\**\s*(\d{4})-(\d{2})-(\d{2})", re.IGNORECASE
)

#: Заголовок таблицы с ответом: «| Фраза (примеры) | Что произойдёт | Как
#: ответит |». Нужен, чтобы справочник оставался табличным.
ANSWER_HEADER_RE = re.compile(r"^\|.*\|.*(?:как ответит|что достаёт)", re.IGNORECASE)

#: Ссылка индекса именно на страницу, а не упоминание имени в тексте.
INDEX_LINK_RE = re.compile(r"\[[^\]]*\]\(\s*(?:\./)?commands\.md\s*\)")


class CheckError(Exception):
    """Одна проблема документа; собирается в общий список."""


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")


def enabled_plugin_ids(config_path: Path) -> list[str]:
    """Id включённых плагинов из `config/plugins.json`; нет файла — []."""
    try:
        payload = json.loads(read_text(config_path))
    except FileNotFoundError:
        return []
    except (OSError, ValueError):
        return []
    if not isinstance(payload, dict):
        return []
    raw = payload.get("enabled")
    if not isinstance(raw, list):
        return []
    return [str(item).strip() for item in raw if str(item).strip()]


def mentioned_ids(text: str, ids: list[str] | tuple[str, ...]) -> set[str]:
    """Id из перечня, которые встречаются в тексте как отдельные слова.

    Сравнение без регистра и с границами: `time` не должен находиться
    внутри `timeout`, а `stocks` — внутри `stocksx`.
    """
    return {
        plugin_id
        for plugin_id in ids
        if re.search(
            r"(?<![\w-])" + re.escape(plugin_id) + r"(?![\w-])",
            text,
            re.IGNORECASE,
        )
    }


def headings(text: str) -> list[str]:
    """Заголовки документа без решёток, в порядке появления."""
    result: list[str] = []
    for line in text.splitlines():
        hit = re.match(r"^#{1,6}\s+(.*\S)\s*$", line)
        if hit:
            result.append(hit.group(1).strip())
    return result


def check(
    doc_path: Path = DEFAULT_DOC,
    *,
    index_path: Path = INDEX_PATH,
    config_path: Path = CONFIG_PATH,
    root: Path = ROOT,
) -> dict[str, object]:
    """Проверить страницу; при проблеме — CheckError со списком."""
    path = doc_path if doc_path.is_absolute() else root / doc_path
    problems: list[str] = []

    try:
        text = read_text(path)
    except FileNotFoundError:
        raise CheckError(f"нет файла {doc_path}") from None
    except OSError as exc:
        raise CheckError(f"{doc_path}: не читается ({exc})") from exc

    if not text.strip():
        problems.append("документ пуст")

    # 1. Дата актуальности — ровно одна, как у остальных документов.
    dates = {match for match in ACTUAL_RE.findall(text)}
    if not dates:
        problems.append("нет строки «Актуально на ГГГГ-ММ-ДД»")
    elif len(dates) > 1:
        problems.append(
            "несколько разных дат «Актуально на»: "
            + ", ".join("-".join(parts) for parts in sorted(dates))
        )

    # 2. Id плагинов: обязательные + всё, что включает config/plugins.json.
    required = list(REQUIRED_PLUGIN_IDS)
    for plugin_id in enabled_plugin_ids(config_path):
        if plugin_id not in required:
            required.append(plugin_id)
    found = mentioned_ids(text, required)
    missing = [plugin_id for plugin_id in required if plugin_id not in found]
    if missing:
        problems.append(
            "не упомянуты плагины: " + ", ".join(missing)
        )

    # 3. Ключевые фразы разделов.
    lowered = text.lower()
    missing_phrases = [phrase for phrase in KEY_PHRASES if phrase not in lowered]
    if missing_phrases:
        problems.append(
            "нет ключевых фраз: " + ", ".join(f"«{p}»" for p in missing_phrases)
        )

    # 4. Обязательные разделы.
    titles = headings(text)
    missing_sections = [
        section
        for section in REQUIRED_SECTIONS
        if not any(section.lower() in title.lower() for title in titles)
    ]
    if missing_sections:
        problems.append(
            "нет разделов: " + ", ".join(missing_sections)
        )

    # 5. Таблицы «фраза → ответ», а не только проза.
    tables = sum(1 for line in text.splitlines() if ANSWER_HEADER_RE.match(line))
    if tables < 6:
        problems.append(
            f"таблиц «фраза → ответ» меньше шести: найдено {tables}"
        )

    # 6. Индекс документации ссылается на страницу.
    index = index_path if index_path.is_absolute() else root / index_path
    try:
        index_text = read_text(index)
    except FileNotFoundError:
        problems.append(f"нет индекса {index_path}")
    except OSError as exc:
        problems.append(f"{index_path}: не читается ({exc})")
    else:
        if not INDEX_LINK_RE.search(index_text):
            problems.append(
                f"в {index_path} нет ссылки на docs/commands.md"
            )

    if problems:
        raise CheckError("\n".join(f"- {problem}" for problem in problems))

    return {
        "doc": str(doc_path),
        "date": "-".join(sorted(dates)[0]) if dates else "",
        "plugins": required,
        "phrases": list(KEY_PHRASES),
        "sections": len(titles),
        "tables": tables,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Проверить docs/commands.md: дата, плагины, фразы, разделы, индекс."
    )
    parser.add_argument(
        "--doc",
        default=str(DEFAULT_DOC),
        help=f"страница справочника (по умолчанию {DEFAULT_DOC})",
    )
    parser.add_argument(
        "--index",
        default=str(INDEX_PATH),
        help=f"индекс документации (по умолчанию {INDEX_PATH})",
    )
    parser.add_argument(
        "--config",
        default=str(CONFIG_PATH),
        help=f"конфиг плагинов (по умолчанию {CONFIG_PATH})",
    )
    args = parser.parse_args(argv)

    try:
        report = check(Path(args.doc), index_path=Path(args.index), config_path=Path(args.config))
    except CheckError as exc:
        print(f"ОШИБКА в {args.doc}:", file=sys.stderr)
        for line in str(exc).splitlines():
            print(line, file=sys.stderr)
        print("FAIL: справочник команд не согласован с кодом", file=sys.stderr)
        return 1

    print(f"справочник команд: {report['doc']} (актуально на {report['date']})")
    print(f"плагины: {len(report['plugins'])} — {', '.join(report['plugins'])}")
    print(f"ключевые фразы: {len(report['phrases'])} — все на месте")
    print(
        f"разделы: {report['sections']}, таблиц «фраза → ответ»: {report['tables']}"
    )
    print(f"индекс: {args.index} ссылается на страницу")
    print("OK: справочник команд полон")
    return 0


if __name__ == "__main__":
    sys.exit(main())
