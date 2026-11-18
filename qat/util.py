"""Мелкие утилиты QAT: атомарная запись, хеши, форматирование.

Модуль намеренно без torch/transformers: его можно импортировать где угодно,
в том числе на машине разработчика, где GPU-стека нет.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    """Время в UTC, ISO 8601 до секунд — как в distill/run.py."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_text_atomic(path: str | Path, text: str) -> None:
    """Записать текст атомарно: сначала temp в том же каталоге, потом rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def write_json_atomic(path: str | Path, obj: Any) -> None:
    """Записать JSON атомарно (run.json, manifest.json, latest-указатели)."""
    write_text_atomic(path, json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 файла потоком: хешировать модель целиком в память не нужно."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def file_fingerprint(path: str | Path) -> dict[str, Any]:
    """Отпечаток файла данных: путь, размер, sha256."""
    path = Path(path)
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "sha256": sha256_file(path)}


def human_bytes(n: int | float) -> str:
    """Размер по-человечески: 1536 -> '1.5 КиБ'."""
    value = float(n)
    for unit in ("Б", "КиБ", "МиБ", "ГиБ", "ТиБ"):
        if abs(value) < 1024.0 or unit == "ТиБ":
            return f"{value:.1f} {unit}" if unit != "Б" else f"{int(value)} {unit}"
        value /= 1024.0
    return f"{value:.1f} ТиБ"


def human_count(n: int) -> str:
    """Число параметров: 1536000000 -> '1.54B'."""
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)
