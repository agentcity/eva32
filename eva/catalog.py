"""Каталог услуг и тарифов EVA32.

Контракт — ``docs/hybrid.md``, задача ``ai/tasks/0015-eva-skills.md``.

Модуль отвечает только за данные и их поиск:

    load(path) -> Catalog
    Catalog.get(name) -> dict | None        # имя или синоним
    Catalog.all() -> list[dict]
    Catalog.search(text) -> list[dict]      # грубый поиск по словам

Никакой сети и никаких зависимостей, кроме стандартной библиотеки.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "Catalog",
    "DEFAULT_PATH",
    "alias_in_tokens",
    "load",
    "normalize",
    "stem",
    "token_match",
    "tokens",
]

#: Каталог лежит в ``data/catalog.json`` рядом с пакетом ``eva``.
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "catalog.json"

#: Поля, которые нужны скиллам. ``id`` синтезируется, если его нет в файле.
_REQUIRED_FIELDS = ("kind", "name", "price_rub", "sources")
_KINDS = ("tariff", "service")

_QUOTES_RE = re.compile(r"[«»\"'`]")
_DASHES_RE = re.compile(r"[–—−]")
_SPACES_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[0-9a-zа-я]+")
_SOURCE_RE = re.compile(r"^\[\d+\]$")

#: Порядок важен: длинные окончания срезаются раньше коротких.
_SUFFIXES = (
    "иями",
    "ями",
    "ами",
    "ого",
    "его",
    "ому",
    "ему",
    "ыми",
    "ими",
    "ей",
    "ой",
    "ый",
    "ий",
    "ая",
    "яя",
    "ое",
    "ее",
    "ые",
    "ие",
    "ых",
    "их",
    "ую",
    "юю",
    "ом",
    "ем",
    "ам",
    "ям",
    "ах",
    "ях",
    "ов",
    "ев",
    "ью",
    "ья",
    "ье",
    "ы",
    "и",
    "а",
    "я",
    "у",
    "ю",
    "е",
    "о",
    "ь",
)


def normalize(text: Any) -> str:
    """Привести строку к нижнему регистру без кавычек и лишних пробелов."""
    value = str(text or "").lower().replace("ё", "е")
    value = _QUOTES_RE.sub("", value)
    value = _DASHES_RE.sub("-", value)
    return _SPACES_RE.sub(" ", value).strip()


def tokens(text: Any) -> list[str]:
    """Разбить текст на слова: строчные буквы/цифры, без пунктуации."""
    return _TOKEN_RE.findall(normalize(text))


def stem(word: str) -> str:
    """Очень грубый стеммер: срезать одно типичное русское окончание."""
    value = str(word or "").lower().replace("ё", "е")
    for suffix in _SUFFIXES:
        if value.endswith(suffix) and len(value) - len(suffix) >= 3:
            return value[: -len(suffix)]
    return value


def token_match(left: str, right: str) -> bool:
    """Совпадают ли слова с точностью до грубой морфологии.

    ``Старт``/``старте``, ``аудит``/``аудита``, ``МКАД``/``МКАДом`` — да;
    ``аудит``/``аудитория`` — нет (основы разные).
    """
    return bool(left) and bool(right) and stem(left) == stem(right)


def alias_in_tokens(query_tokens: Iterable[str], alias: str) -> bool:
    """Встречается ли фраза-синоним в токенах запроса подряд."""
    query = list(query_tokens)
    expected = tokens(alias)
    if not query or not expected:
        return False
    width = len(expected)
    for start in range(len(query) - width + 1):
        if all(token_match(query[start + i], expected[i]) for i in range(width)):
            return True
    return False


def _aliases(item: dict[str, Any]) -> list[str]:
    """Имя и все синонимы позиции каталога."""
    values: list[str] = [str(item.get("name") or "")]
    synonyms = item.get("synonyms") or []
    if isinstance(synonyms, str):
        synonyms = [synonyms]
    values.extend(str(value) for value in synonyms)
    return [value for value in values if value.strip()]


def _alias_token_set(item: dict[str, Any]) -> set[str]:
    result: set[str] = set()
    for alias in _aliases(item):
        result.update(tokens(alias))
    return result


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    return []


def _copy(item: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(item)


class Catalog:
    """Неизменяемый каталог с lookup по имени/синониму и грубым поиском."""

    def __init__(self, items: Iterable[dict[str, Any]], path: str | Path | None = None):
        self._items: list[dict[str, Any]] = [_copy(item) for item in items]
        self.path = str(path) if path is not None else None
        self._by_key: dict[str, dict[str, Any]] = {}
        for item in self._items:
            for alias in _aliases(item):
                self._by_key.setdefault(normalize(alias), item)

    # ------------------------------------------------------------------ #
    # контракт docs/hybrid.md
    # ------------------------------------------------------------------ #
    def get(self, name: str) -> dict[str, Any] | None:
        """Позиция по имени или синониму (регистр, кавычки, ё/е — не важны)."""
        key = normalize(name)
        if not key:
            return None
        item = self._by_key.get(key)
        if item is None:
            for prefix in ("тариф ", "услуга ", "услугу "):
                if key.startswith(prefix):
                    item = self._by_key.get(key[len(prefix) :].strip())
                    if item is not None:
                        break
        return _copy(item) if item is not None else None

    def all(self) -> list[dict[str, Any]]:
        """Все позиции каталога (копии: менять внутренние данные нельзя)."""
        return [_copy(item) for item in self._items]

    def search(self, text: str) -> list[dict[str, Any]]:
        """Грубый поиск по словам: имя/синонимы весят больше заметок."""
        query = set(tokens(text))
        if not query:
            return []

        scored: list[tuple[int, int, dict[str, Any]]] = []
        for index, item in enumerate(self._items):
            name_tokens = {
                token
                for alias in _aliases(item)
                for token in tokens(alias)
            }
            other_tokens = {
                token
                for field in ("limits", "notes")
                for token in tokens(" ".join(_as_list(item.get(field))))
            }
            score = 0
            for word in query:
                if any(token_match(word, known) for known in name_tokens):
                    score += 3
                elif any(token_match(word, known) for known in other_tokens):
                    score += 1
            if score:
                scored.append((score, index, item))

        scored.sort(key=lambda row: (-row[0], row[1]))
        return [_copy(item) for _, _, item in scored]

    def mentioned(self, text: str) -> list[dict[str, Any]]:
        """Позиции, чьё имя или синоним встречаются в тексте подряд."""
        query = tokens(text)
        if not query:
            return []
        return [
            _copy(item)
            for item in self._items
            if any(alias_in_tokens(query, alias) for alias in _aliases(item))
        ]

    # ------------------------------------------------------------------ #
    # удобства
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self.all())

    def __repr__(self) -> str:
        return f"Catalog(items={len(self._items)}, path={self.path!r})"


def _validate_item(raw: Any, index: int) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"catalog[{index}]: ожидался объект, получено {type(raw).__name__}")

    item = dict(raw)
    for field in _REQUIRED_FIELDS:
        if field not in item:
            raise ValueError(f"catalog[{index}]: нет обязательного поля {field!r}")

    kind = str(item.get("kind") or "").strip().lower()
    if kind not in _KINDS:
        raise ValueError(f"catalog[{index}]: kind={kind!r}, ожидалось tariff или service")
    item["kind"] = kind

    name = str(item.get("name") or "").strip()
    if not name:
        raise ValueError(f"catalog[{index}]: пустое имя")
    item["name"] = name

    price = item.get("price_rub")
    if isinstance(price, bool) or not isinstance(price, (int, float)):
        raise ValueError(f"catalog[{index}]: price_rub должен быть числом")
    if price < 0:
        raise ValueError(f"catalog[{index}]: price_rub не может быть отрицательным")
    item["price_rub"] = int(price) if float(price).is_integer() else float(price)

    sources = _as_list(item.get("sources"))
    if not sources:
        raise ValueError(f"catalog[{index}]: нужен хотя бы один источник вида [1]")
    for source in sources:
        if not _SOURCE_RE.match(source):
            raise ValueError(f"catalog[{index}]: источник {source!r} не похож на [n]")
    item["sources"] = sources

    for field in ("synonyms", "limits", "notes"):
        item[field] = _as_list(item.get(field))

    item_id = str(item.get("id") or "").strip()
    if not item_id:
        item_id = f"item-{index + 1}"
    item["id"] = item_id
    return item


def load(path: str | Path | None = None) -> Catalog:
    """Прочитать каталог из JSON. Ошибка данных — ``ValueError``."""
    source = Path(path) if path is not None else DEFAULT_PATH
    with source.open("r", encoding="utf-8") as handle:
        try:
            raw = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source}: некорректный JSON: {exc}") from exc
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{source}: ожидался непустой список позиций каталога")

    items = [_validate_item(item, index) for index, item in enumerate(raw)]
    seen: set[str] = set()
    for item in items:
        if item["id"] in seen:
            raise ValueError(f"{source}: повторяющийся id {item['id']!r}")
        seen.add(item["id"])
    return Catalog(items, path=source)
