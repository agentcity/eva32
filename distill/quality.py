"""Фильтр качества для чужих датасетов.

Публичные наборы приходят с мусором: политика, религия, мат, медицинские советы,
фактические ошибки, чужие URL. Учить этому ассистента для бизнеса нельзя — он должен
быть нейтральным и не спорить о политике с клиентом.

Правило: лучше потерять 40% строк, чем научить модель ругаться в чате поддержки.
"""
from __future__ import annotations

import re

#: Темы, на которые ассистент не должен иметь «мнения».
BANNED_TOPICS = re.compile(
    r"\b("
    r"трамп|байден|обама|путин|навальн|оппозиц|протест|митинг|санкц|кремл|"
    r"демократ|республикан|конгресс|сенат|парламент|выборы|политик|"
    r"ислам|христиан|православ|иуда|мусульман|религи|бог|библи|коран|"
    r"аборт|эвтаназ|гей|лгбт|"
    r"игил|теракт|война|армия|военн"
    r")\b",
    re.I,
)

#: Мат и грубость — модель не должна этому подражать.
PROFANITY = re.compile(
    r"\b(хуй|хуя|пизд|бляд|блят|ебал|ёбал|ебат|ёбат|сука|мудак|дерьм|говн|жоп|"
    r"fuck|shit|bitch)\w*",
    re.I,
)

#: Признаки машинного перевода, который сломал смысл.
BROKEN = re.compile(
    r"(URL_0|URL_\d+|\[URL|перевод|импортировать|определение\s+\w+\s*\(|"
    r"возвращаться:|:значение параметра:|:возвращаться:)",
    re.I,
)

#: Медицина, право, налоги — советы в этих темах опасны без источника.
RISKY_ADVICE = re.compile(
    r"\b(лекарств|дозировк|диагноз|симптом|болезн|депресс|рак\b|терапи|"
    r"налог|штраф|суд\b|иск\b|юрист|адвокат|закон\b)\w*",
    re.I,
)

MIN_CYR = 20


def has_cyrillic(text: str, minimum: int = MIN_CYR) -> bool:
    return len(re.findall(r"[а-яА-ЯёЁ]", text)) >= minimum


def is_neutral(text: str) -> bool:
    return not BANNED_TOPICS.search(text)


def is_clean_language(text: str) -> bool:
    return not PROFANITY.search(text)


def is_not_broken(text: str) -> bool:
    return not BROKEN.search(text)


def is_safe_advice(text: str) -> bool:
    return not RISKY_ADVICE.search(text)


def accept_answer(text: str) -> bool:
    """Полный прогон по фильтрам для ответа."""
    return (
        has_cyrillic(text)
        and is_neutral(text)
        and is_clean_language(text)
        and is_not_broken(text)
        and is_safe_advice(text)
    )


def accept_pair(question: str, answer: str) -> bool:
    return has_cyrillic(question, 10) and accept_answer(answer)


def report(rows: list[tuple[str, str]]) -> dict[str, int]:
    """Сколько строк отсекает каждый фильтр — чтобы цифры были в отчёте."""
    out = {"всего": len(rows), "прошли": 0}
    for question, answer in rows:
        if accept_pair(question, answer):
            out["прошли"] += 1
            continue
        for name, check in (
            ("политика/религия", lambda: not is_neutral(answer)),
            ("мат", lambda: not is_clean_language(answer)),
            ("сломанный перевод", lambda: not is_not_broken(answer)),
            ("медицина/право/налоги", lambda: not is_safe_advice(answer)),
            ("не русский", lambda: not has_cyrillic(answer)),
        ):
            if check():
                out[name] = out.get(name, 0) + 1
    return out
