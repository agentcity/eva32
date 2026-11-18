#!/usr/bin/env python3
"""Сверка документации VTM с кодом.

Что проверяется:

* таблица «Набор операций» в `docs/vtm.md` — количество опкодов, их коды и
  имена — против enum `vtm_opcode` в `src/vtm/vtm.h`, таблицы имён
  `vtm_op_name()` в `src/vtm/loader.c` и (если есть) enum `Op` /
  `OP_NAMES` в `tools/vtm_ref.py`;
* внутренние ссылки в `docs/vtm.md`: markdown-ссылки и пути в обратных
  кавычках, которые указывают на файлы репозитория.

Скрипт работает без сети, только stdlib. При любом расхождении — exit 1.

Запуск:
    python3 tools/check_vtm_docs.py
    python3 tools/check_vtm_docs.py --no-links
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOC = Path("docs/vtm.md")

# Опкоды в docs/vtm.md: | 0x01 | `TMV` | ...
_DOC_OP_RE = re.compile(
    r"^\|\s*(0x[0-9A-Fa-f]{2})\s*\|\s*`([A-Za-z][A-Za-z0-9_]*)`\s*\|"
)
# enum vtm_opcode в src/vtm/vtm.h: VTM_OP_TMV = 0x01,
_ENUM_OP_RE = re.compile(
    r"\bVTM_OP_([A-Z][A-Z0-9_]*)\s*=\s*(0x[0-9A-Fa-f]+)"
)
# vtm_op_name() в src/vtm/loader.c: case VTM_OP_TMV: return "TMV";
_NAME_OP_RE = re.compile(
    r"case\s+VTM_OP_([A-Z][A-Z0-9_]*)\s*:\s*return\s+\"([^\"]+)\"\s*;"
)
# enum Op(IntEnum) в tools/vtm_ref.py: TMV = 0x01
_PY_OP_RE = re.compile(
    r"^[ \t]+([A-Z][A-Z0-9_]*)[ \t]*=[ \t]*(0x[0-9A-Fa-f]+)[ \t]*,?[ \t]*$",
    re.MULTILINE,
)
_MD_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_TICK_RE = re.compile(r"`([^`\n]+)`")
_LINK_EXTS = {
    ".md",
    ".c",
    ".h",
    ".py",
    ".json",
    ".vtm",
    ".trit",
    ".bin",
    ".sh",
    ".txt",
    ".toml",
    ".yml",
    ".yaml",
}


class CheckError(Exception):
    """Расхождение, из-за которого скрипт обязан вернуть 1."""


def _parse_docs_ops(text: str) -> dict[int, str]:
    ops: dict[int, str] = {}
    for line in text.splitlines():
        match = _DOC_OP_RE.match(line)
        if not match:
            continue
        code = int(match.group(1), 16)
        name = match.group(2)
        if code in ops:
            raise CheckError(
                f"docs: код 0x{code:02X} встречается дважды "
                f"({ops[code]} и {name})"
            )
        if name in ops.values():
            raise CheckError(f"docs: имя {name} встречается дважды")
        ops[code] = name
    return ops


def _parse_c_enum(text: str) -> dict[int, str]:
    ops: dict[int, str] = {}
    for name, code_text in _ENUM_OP_RE.findall(text):
        code = int(code_text, 16)
        if code in ops:
            raise CheckError(
                f"vtm.h: код 0x{code:02X} встречается дважды "
                f"({ops[code]} и {name})"
            )
        ops[code] = name
    return ops


def _parse_c_names(text: str) -> dict[int, str]:
    # Сопоставляем имя из switch с кодом из enum; сюда передаётся уже
    # разобранный enum, чтобы не дублировать таблицу кодов.
    names = {name for name, _ in _NAME_OP_RE.findall(text)}
    return {name: name for name in names}


def _parse_python_ops(text: str) -> dict[int, str] | None:
    start = text.find("class Op(")
    if start < 0:
        return None
    end = text.find("OP_NAMES", start)
    if end < 0:
        end = len(text)
    ops: dict[int, str] = {}
    for name, code_text in _PY_OP_RE.findall(text[start:end]):
        ops[int(code_text, 16)] = name
    return ops or None


def _require_repo_file(rel: str, context: str) -> Path | None:
    path = (ROOT / rel).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError:
        return None
    if not path.is_file():
        raise CheckError(f"{context}: файл {rel!r} не существует")
    return path


def _check_links(doc: Path, text: str) -> int:
    checked: set[str] = set()

    def add_target(target: str) -> None:
        target = target.strip()
        if not target or target.startswith(("#", "http://", "https://", "mailto:")):
            return
        target = target.split("#", 1)[0]
        if not target:
            return
        if target.startswith("/"):
            raise CheckError(f"ссылки: абсолютный путь {target!r} не переносим")
        candidates = [(doc.parent / target), (ROOT / target)]
        for candidate in candidates:
            resolved = candidate.resolve()
            try:
                resolved.relative_to(ROOT)
            except ValueError:
                continue
            if resolved.exists():
                checked.add(target)
                return
        raise CheckError(f"ссылки: цель {target!r} не существует")

    for target in _MD_LINK_RE.findall(text):
        add_target(target)

    for token in _TICK_RE.findall(text):
        token = token.strip()
        if "/" not in token or " " in token or "*" in token:
            continue
        # Пути вида ./tokenizer.json — файлы времени выполнения, их в
        # репозитории может не быть; проверяем только ссылки от корня.
        if token.startswith(("-", "|", "$", ".", "http://", "https://")):
            continue
        if Path(token).suffix.lower() not in _LINK_EXTS:
            continue
        add_target(token)

    return len(checked)


def _format_ops(ops: dict[int, str]) -> str:
    return ", ".join(f"0x{code:02X} {ops[code]}" for code in sorted(ops))


def check(doc_path: Path, links: bool) -> None:
    if not doc_path.is_absolute():
        doc_path = ROOT / doc_path
    if not doc_path.is_file():
        raise CheckError(f"нет документа {doc_path}")

    doc_text = doc_path.read_text(encoding="utf-8")
    docs_ops = _parse_docs_ops(doc_text)
    if not docs_ops:
        raise CheckError(
            f"{doc_path}: не найдена таблица опкодов "
            "(строки вида «| 0x01 | `TMV` | ...»)"
        )

    header = _require_repo_file("src/vtm/vtm.h", "vtm.h")
    assert header is not None
    c_ops = _parse_c_enum(header.read_text(encoding="utf-8"))
    if not c_ops:
        raise CheckError("vtm.h: не найден enum vtm_opcode (VTM_OP_*)")

    loader = _require_repo_file("src/vtm/loader.c", "loader.c")
    assert loader is not None
    loader_text = loader.read_text(encoding="utf-8")
    c_names = _parse_c_names(loader_text)
    if not c_names:
        raise CheckError("loader.c: не найдена таблица имён vtm_op_name()")

    python_ops = None
    ref_path = ROOT / "tools/vtm_ref.py"
    if ref_path.is_file():
        python_ops = _parse_python_ops(ref_path.read_text(encoding="utf-8"))

    # --- коды и количество ------------------------------------------------
    if len(docs_ops) != len(c_ops):
        raise CheckError(
            f"количество опкодов: docs={len(docs_ops)}, vtm.h={len(c_ops)}"
        )
    if set(docs_ops) != set(c_ops):
        only_docs = sorted(set(docs_ops) - set(c_ops))
        only_c = sorted(set(c_ops) - set(docs_ops))
        raise CheckError(
            "набор кодов различается: "
            f"только в docs={[hex(c) for c in only_docs]}, "
            f"только в vtm.h={[hex(c) for c in only_c]}"
        )

    # --- имена ------------------------------------------------------------
    for code in sorted(docs_ops):
        doc_name = docs_ops[code]
        c_name = c_ops[code]
        if doc_name != c_name:
            raise CheckError(
                f"0x{code:02X}: docs={doc_name}, vtm.h={c_name}"
            )
        if doc_name not in c_names:
            raise CheckError(
                f"0x{code:02X} {doc_name}: нет в vtm_op_name() (loader.c)"
            )
        if python_ops is not None and code not in python_ops:
            raise CheckError(
                f"0x{code:02X} {doc_name}: нет в enum Op (tools/vtm_ref.py)"
            )
        if python_ops is not None and python_ops[code] != doc_name:
            raise CheckError(
                f"0x{code:02X}: docs={doc_name}, "
                f"tools/vtm_ref.py={python_ops[code]}"
            )

    if loader_text.count("VTM_OP_") == 0:  # pragma: no cover - страховка
        raise CheckError("loader.c: таблица имён пуста")

    print(f"{doc_path.relative_to(ROOT)}: опкодов {len(docs_ops)}")
    print(f"src/vtm/vtm.h: {len(c_ops)}")
    print(f"src/vtm/loader.c (vtm_op_name): {len(c_names)}")
    if python_ops is not None:
        print(f"tools/vtm_ref.py (enum Op): {len(python_ops)}")
    print(f"коды и имена: {_format_ops(docs_ops)}")

    if links:
        count = _check_links(doc_path, doc_text)
        print(f"внутренние ссылки: проверено {count}, все существуют")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Сверить опкоды и ссылки docs/vtm.md с кодом VTM."
    )
    parser.add_argument(
        "--doc",
        default=str(DEFAULT_DOC),
        help=f"документ для сверки (по умолчанию {DEFAULT_DOC})",
    )
    parser.add_argument(
        "--no-links",
        action="store_true",
        help="не проверять внутренние ссылки документа",
    )
    args = parser.parse_args(argv)

    try:
        check(Path(args.doc), links=not args.no_links)
    except CheckError as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"ОШИБКА: {exc}", file=sys.stderr)
        return 1
    print("OK: документация совпадает с кодом")
    return 0


if __name__ == "__main__":
    sys.exit(main())
