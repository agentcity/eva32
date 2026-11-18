"""Фильтры качества пар «вопрос → ответ» для дистилляции.

Задача: не тащить в обучение ответы, которые учат модель врать. Всё проверяется
на строках, поэтому фильтры тестируются без модели и без сети.
"""
from __future__ import annotations

import re
from typing import Iterable

CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
LATIN_RUN = re.compile(r"\b[A-Za-z]{3,}(?:\s+[A-Za-z]{2,}){4,}\b")
PRICE = re.compile(r"(?:\d[\d\s]{0,9}\s*(?:₽|руб|рублей|р\.)\b)|\b(?:цена|стоимость)\s*[:—-]?\s*\d", re.I)
PRICE_ANY = re.compile(r"\d[\d\s.,]{0,9}\s*(?:₽|руб(?:лей|\.)?)", re.I)
VENDORS = re.compile(r"\b(OpenAI|ChatGPT|Claude|Anthropic|Gemini|Google|Alibaba|Qwen)\b", re.I)
REFUSAL = re.compile(r"\b(не могу|не нашла|не нашел|не знаю|уточните|нет данных)\b", re.I)

MIN_CHARS, MAX_CHARS = 40, 4000


def has_foreign(text: str) -> bool:
    """Иероглифы/хангыль или длинная латинская фраза в русском ответе."""
    return bool(CJK.search(text) or LATIN_RUN.search(text))


def _numbers(text: str) -> set[str]:
    """Все числа в тексте без пробелов между разрядами: «1 500» → «1500»."""
    return {m.replace(" ", "") for m in re.findall(r"\d[\d\s]{0,12}", text)}


def has_invented_price(answer: str, context: str | None = None) -> bool:
    """Цена в ответе, которой нет в контексте.

    Если контекст передан и цена в нём есть — это цитата, а не выдумка. Если контекста
    нет, любая конкретная цена — фантазия: учителю неоткуда её взять.
    """
    if not (PRICE_ANY.search(answer) or PRICE.search(answer)):
        return False
    if not context:
        return True
    known = _numbers(context)
    return not _numbers(answer).issubset(known)


def is_refusal(answer: str) -> bool:
    return bool(REFUSAL.search(answer))


def has_other_vendor(answer: str) -> bool:
    """Ответ приписывает происхождение чужой модели — такую пару нельзя учить.

    На прогоне учитель на вопрос «на какой модели работаешь» выдал «модель OpenAI»:
    это учение о том, что Ева — не Ева.
    """
    return bool(VENDORS.search(answer))


def length_ok(answer: str) -> bool:
    return MIN_CHARS <= len(answer.strip()) <= MAX_CHARS


def norm_question(text: str) -> str:
    t = re.sub(r"\s+", " ", text.strip().lower())
    return re.sub(r"[\s?!.]+$", "", t)


def check(question: str, answer: str, context: str | None = None,
          profile: str = "prose") -> list[str]:
    """Вернуть список проблем; пустой список — пара годится для обучения.

    ``profile="code"`` — ответы по программированию: латиница и длинные листинги
    нормальны, поэтому проверки «чужой язык» и «длина» заменены на «есть русское
    объяснение» и «нет внешних ссылок».
    """
    if profile == "code":
        return _check_code(question, answer)
    problems: list[str] = []
    if not length_ok(answer):
        problems.append("length")
    if has_foreign(answer):
        problems.append("foreign")
    if has_invented_price(answer, context):
        problems.append("invented_price")
    if not question.strip() or len(question) > 2000:
        problems.append("bad_question")
    if has_other_vendor(answer):
        problems.append("other_vendor")
    return problems


_CODE_FENCE = re.compile(r"```")


def _check_code(question: str, answer: str) -> list[str]:
    problems: list[str] = []
    if not question.strip() or len(question) > 2000:
        problems.append("bad_question")
    if not answer.strip():
        problems.append("empty")
        return problems
    # Без кода ответ должен быть короче, иначе это простыня.
    if "```" not in answer and len(answer) > 4000:
        problems.append("length")
    # Должна остаться русская речь: код сам по себе модель не научит объяснять.
    cyrillic = sum(1 for ch in answer if "а" <= ch.lower() <= "я" or ch.lower() == "ё")
    if cyrillic < 40:
        problems.append("no_russian")
    if re.search(r"https?://", answer):
        problems.append("link")
    return problems


def dedupe(pairs: Iterable[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for p in pairs:
        key = norm_question(p.get("question", ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out
