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


def has_invented_price(answer: str, question: str) -> bool:
    """Цена в ответе, хотя никакого каталога не передавали.

    На этапе сбора учителю не дают контекста, значит любая конкретная цена — выдумка.
    """
    del question  # интерфейс единый: фильтр может понадобиться и с контекстом
    return bool(PRICE_ANY.search(answer) or PRICE.search(answer))


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


def check(question: str, answer: str) -> list[str]:
    """Вернуть список проблем; пустой список — пара годится для обучения."""
    problems: list[str] = []
    if not length_ok(answer):
        problems.append("length")
    if has_foreign(answer):
        problems.append("foreign")
    if has_invented_price(answer, question):
        problems.append("invented_price")
    if not question.strip() or len(question) > 2000:
        problems.append("bad_question")
    if has_other_vendor(answer):
        problems.append("other_vendor")
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
