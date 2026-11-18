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

#: Мат → литературный оборот. Заменяем корень, сохраняя окончание: тогда
#: «охуенная идея» → «прекрасная идея», а «дерьмовый код» → «негодный код».
#: Порядок важен: длинные корни раньше коротких.
PROFANITY_MAP: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bблядск", re.I), "неприемлем"),
    (re.compile(r"\bбляд", re.I), "неприемлем"),
    (re.compile(r"\bбля(?:т|дь)?\b", re.I), ""),
    (re.compile(r"\bпиздец\b", re.I), "неприятность"),
    (re.compile(r"\bпиздеца\b", re.I), "неприятности"),
    (re.compile(r"\bпизд", re.I), "ерунд"),
    (re.compile(r"\bхуйня\b", re.I), "ерунда"),
    (re.compile(r"\bхуйни\b", re.I), "ерунды"),
    (re.compile(r"\bхуйню\b", re.I), "ерунду"),
    (re.compile(r"\bхуйнёй\b|\bхуйней\b", re.I), "ерундой"),
    (re.compile(r"\bхуйн", re.I), "ерунд"),
    (re.compile(r"\bхуйня\b", re.I), "ерунда"),
    (re.compile(r"\bхуйни\b", re.I), "ерунды"),
    (re.compile(r"\bхуйню\b", re.I), "ерунду"),
    (re.compile(r"\bхуйнёй\b|\bхуйней\b", re.I), "ерундой"),
    (re.compile(r"\bхуйн", re.I), "ерунд"),
    (re.compile(r"\bху[йяёе]в?", re.I), "ерунд"),
    (re.compile(r"\bохуенн|\bахуенн|\bохуит", re.I), "прекрасн"),
    (re.compile(r"\bохуе|\bахуе", re.I), "удив"),
    (re.compile(r"\bебат|\bёбат", re.I), "удивля"),
    (re.compile(r"\bебал|\bёбал", re.I), "надоеда"),
    (re.compile(r"\bебуч|\bёбуч", re.I), "неудачн"),
    (re.compile(r"\bсука\b", re.I), "вот досада"),
    (re.compile(r"\bмудак", re.I), "глупец"),
    (re.compile(r"\bдерьмов", re.I), "негодн"),
    (re.compile(r"\bдерьм(?:о|а|у|ом|е)\b", re.I), "нечто негодное"),
    (re.compile(r"\bговн", re.I), "ерунд"),
    (re.compile(r"\bжоп", re.I), "задн"),
    (re.compile(r"\b(?:fuck|shit|bitch)\w*", re.I), "ерунда"),
]

MIN_CYR = 20


def has_cyrillic(text: str, minimum: int = MIN_CYR) -> bool:
    return len(re.findall(r"[а-яА-ЯёЁ]", text)) >= minimum


def is_neutral(text: str) -> bool:
    return not BANNED_TOPICS.search(text)


def is_clean_language(text: str) -> bool:
    return not PROFANITY.search(text)


def soften(text: str) -> str:
    """Заменить мат литературными оборотами, не трогая всё остальное.

    Важно: если замен не было — текст возвращается **байт в байт**. Иначе чистка
    нормализует код: схлопывает отступы, поднимает регистр первой буквы — и ломает
    примеры программ. Этот баг уже испортил 2433 пары, поэтому правило жёсткое:
    нет мата — нет изменений.
    """
    out = text
    changed = False
    for pattern, replacement in PROFANITY_MAP:
        if not pattern.search(out):
            continue
        changed = True

        def _sub(match: re.Match[str], repl: str = replacement) -> str:
            if not repl:
                return ""
            return repl[:1].upper() + repl[1:] if match.group(0)[:1].isupper() else repl

        out = pattern.sub(_sub, out)
    if not changed:
        return text
    return out


def soften_pair(question: str, answer: str) -> tuple[str, str]:
    """Почистить оба поля от мата (для датасетов, где ругань — часть стиля)."""
    return soften(question), soften(answer)


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
