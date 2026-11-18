"""Правиловый классификатор маршрута для гибридного ассистента EVA32.

Без сети и без модели: по тексту запроса решает, кто должен отвечать —
скилл (``skill``), малая троичная модель (``small``) или большая по API (``big``).

Контракт: ``classify(text, cfg) -> Route`` (см. ``docs/hybrid.md``).

Признаки (в порядке приоритета):

1. пустой текст / мусор — малая модель, низкая уверенность, причина в ``reason``;
2. приветствие/болтовня — малая модель;
3. маркеры рассуждения (почему, объясни, сравни, напиши код, переведи, посчитай)
   и структурные задачи (код, математика, перевод, доказательство) — большая;
4. творческие задачи (придумай, составь, спланируй) — большая: это порождение
   текста, а не факт, поэтому проверяются до каталога;
5. факт из каталога (цена, часы, возврат, контакты) — скилл;
6. структурные признаки (sql, запрос, формул, таблиц, схем) — большая; идут
   после каталога, чтобы «сколько запросов входит в тариф» осталось скиллом;
7. вопрос на общие знания (столица, сколько будет, кто такой, в каком году,
   переведи на) — большая: каталога и скилла там нет;
8. сравнение (разница/отличается) — большая;
9. вопрос «как»/«что такое» — большая;
10. высокая доля латиницы — большая;
11. длинный текст — большая;
12. остальное — малая.

Каталог проверяется после явных маркеров (просьба «посчитай» важнее упоминания
цены) и до сравнения/«как» («разница в цене» ещё может быть ответом скилла).
Признаки общих знаний тоже идут после каталога: «сколько стоит тариф» — это
факт из каталога, а «сколько будет 15% от 800» — общие знания.

Пороги не захардкожены в ``classify``: они берутся из ``cfg``. Ожидается либо
объект ``eva.config.Config`` (у него есть свойство ``classifier``), либо словарь
``{"classifier": {...}}`` / ``{"long_chars": ..., ...}``, либо объект с полями.
Имена, которые понимает классификатор (алиасы — для совместимости):

======================== ==============================================
``long_chars``           текст длиннее — маршрут ``big`` (по умолчанию 120)
``latin_ratio``          доля латиницы, выше которой — ``big`` (0.35)
``reason_markers``       маркеры рассуждения (список/кортеж/строка через запятую)
``task_markers``         творческие задачи: придумай, составь, спланируй
``structural_markers``   структурные признаки: sql, запрос, формул, таблиц, схем
``knowledge_markers``    маркеры вопросов на общие знания (список/строка)
``latin_min_letters``    минимум букв, чтобы считать долю латиницы (12)
``letters_ratio_min``    минимум доли букв, иначе текст считается мусором (0.30)
``catalog_fact_to_skill````False`` отключает маршрут ``skill`` из классификатора
======================== ==============================================

Если ``cfg`` не передан, используются ``DEFAULT_CLASSIFIER_THRESHOLDS`` — только
чтобы модуль работал сам по себе и в тестах; в рабочем контуре пороги приходят
из конфига.

Мусор и пустой текст не считаются ошибкой: возвращается ``Route`` с ``kind="small"``
и низкой ``confidence`` — это сигнал policy при желании эскалировать запрос.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

try:  # общий тип из eva/types.py (зона eva-core-router)
    from .types import Route
except ImportError:  # pragma: no cover - чтобы модуль работал и до появления types.py
    @dataclass(frozen=True)
    class Route:  # type: ignore[no-redef]
        """Резервная копия контракта Route (поля те же, что в eva/types.py)."""

        kind: str
        reason: str
        confidence: float

__all__ = [
    "classify",
    "DEFAULT_CLASSIFIER_THRESHOLDS",
    "DEFAULT_REASON_MARKERS",
    "DEFAULT_TASK_MARKERS",
    "DEFAULT_STRUCTURAL_MARKERS",
    "DEFAULT_KNOWLEDGE_MARKERS",
    "SKILL",
    "SMALL",
    "BIG",
]

SKILL = "skill"
SMALL = "small"
BIG = "big"
KINDS = (SKILL, SMALL, BIG)

#: Маркеры рассуждения по умолчанию. Совпадают с DEFAULT_REASON_MARKERS в
#: eva/config.py; рабочий контур передаёт свой список через cfg.
DEFAULT_REASON_MARKERS = (
    "почему",
    "зачем",
    "объясни",
    "объясните",
    "сравни",
    "сравните",
    "напиши код",
    "напишите код",
    "переведи",
    "переведите",
    "посчитай",
    "посчитайте",
    "докажи",
    "проанализируй",
    "как работает",
    "что такое",
)

#: Творческие задачи по умолчанию. Малая модель здесь плоха: нужно породить
#: текст (план, сценарий), а не вспомнить факт. Проверяются до каталога.
DEFAULT_TASK_MARKERS = (
    "придумай",
    "придумайте",
    "составь",
    "составьте",
    "спланируй",
    "спланируйте",
)

#: Структурные признаки по умолчанию: SQL/запрос, формула, таблица, схема.
#: Проверяются после каталога, чтобы «сколько запросов входит в тариф»
#: оставалось фактом из каталога, а не уходило в big.
DEFAULT_STRUCTURAL_MARKERS = (
    "sql",
    "запрос",
    "формул",
    "таблиц",
    "схем",
)

#: Маркеры вопросов на общие знания по умолчанию. Скиллов и каталога там нет,
#: поэтому отвечает большая модель. Проверяются после каталога: «сколько стоит
#: тариф» — факт из каталога, «сколько будет 15% от 800» — общие знания.
DEFAULT_KNOWLEDGE_MARKERS = (
    "столиц",
    "сколько будет",
    "кто такой",
    "кто такая",
    "кто такие",
    "в каком году",
    "переведи на",
    "переведите на",
)

#: Пороги по умолчанию. Любой ключ переопределяется через cfg.
DEFAULT_CLASSIFIER_THRESHOLDS: dict[str, Any] = {
    "long_chars": 120,
    "latin_ratio": 0.35,
    "reason_markers": DEFAULT_REASON_MARKERS,
    "task_markers": DEFAULT_TASK_MARKERS,
    "structural_markers": DEFAULT_STRUCTURAL_MARKERS,
    "knowledge_markers": DEFAULT_KNOWLEDGE_MARKERS,
    "latin_min_letters": 12,
    "letters_ratio_min": 0.30,
    "catalog_fact_to_skill": True,
}

#: Алиасы: конфиг может отдать пороги и «плоско», и через поля Config.
_THRESHOLD_ALIASES: dict[str, tuple[str, ...]] = {
    "long_chars": ("long_chars", "long_text_chars", "classifier_long_chars", "max_chars"),
    "latin_ratio": ("latin_ratio", "latin_ratio_threshold", "classifier_latin_ratio"),
    "latin_min_letters": ("latin_min_letters", "latin_min_chars"),
    "letters_ratio_min": ("letters_ratio_min", "garbage_letters_ratio", "min_letters_ratio"),
    "catalog_fact_to_skill": ("catalog_fact_to_skill", "catalog_to_skill"),
}

#: Структурные признаки задачи, которой малой модели мало: код, математика,
#: перевод, доказательство. Дополняют маркеры из конфига (там есть «напиши
#: код», но не «напиши функцию»), поэтому проверяются отдельно.
_EXPLICIT_TASK_RE = re.compile(
    r"```"
    r"|\bdef\s+\w+\s*\("
    r"|\bclass\s+\w+\s*[\(:]"
    r"|\b(?:напиши|напишите|сгенерируй|создай)\b[^.!?]{0,40}?"
    r"\b(?:код|функци\w*|скрипт\w*|программ\w*|класс\w*)\b"
    r"|\b(?:write|generate|create)\s+(?:code|a\s+function|a\s+script)\b"
    r"|\bреши(?:те)?\b|\bуравнени\w*|\bинтеграл\w*|\bпроизводн\w*"
    r"|\bдоказательств\w*|\bпоясни\w*"
    r"|\bперевод\w*|\btranslate\b"
    r"|\bcalculate\b|\bcompute\b|\bsolve\b",
    re.IGNORECASE,
)

#: Сравнение — вопрос про общие знания, поэтому большая модель. Проверяется
#: после каталога: «разница в цене» ещё может быть ответом скилла.
_COMPARISON_RE = re.compile(
    r"\bразниц\w*|\bотлича\w*|\bdifference\b|\bversus\b",
    re.IGNORECASE,
)

#: Вопросы про факт из каталога: цена/часы/возврат/контакты.
_CATALOG_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "pricing",
        re.compile(
            r"цен[аыуеой]"
            r"|стоимост\w*"
            r"|сколько\s+(?:будет\s+)?стоит"
            r"|\bпрайс\w*"
            r"|\bтариф\w*"
            r"|\bprice\b|\bpricing\b|\bcost\b|\bhow\s+much\b",
            re.IGNORECASE,
        ),
    ),
    (
        "hours",
        re.compile(
            r"часы\s+работы"
            r"|режим\s+работы"
            r"|график\s+работы"
            r"|время\s+работы"
            r"|во\s+сколько\s+(?:вы\s+)?(?:откры|закры|работа)"
            r"|\bworking\s+hours\b"
            r"|when\s+(?:do\s+you\s+)?open",
            re.IGNORECASE,
        ),
    ),
    (
        "refunds",
        re.compile(
            r"\bвозврат\w*"
            r"|\bвернуть\b|\bверните\b|\bвернёте\b|\bвернете\b"
            r"|\brefund\w*|\bmoney\s+back\b",
            re.IGNORECASE,
        ),
    ),
    (
        "contacts",
        re.compile(
            r"контакт\w*"
            r"|\bтелефон\w*"
            r"|\bпочт[аыуео]\w*"
            r"|эл\.?\s*почт\w*"
            r"|\bадрес\w*"
            r"|\bсвязаться\b"
            r"|\bподдержк\w*"
            r"|\bemail\b|e-mail"
            r"|\bphone\b|\baddress\w*\b"
            r"|\bsupport@",
            re.IGNORECASE,
        ),
    ),
)

#: «Как» / «что такое» — вопросы про общие знания, малой модели обычно мало.
_CONCEPTUAL_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "что такое",
        re.compile(
            r"\bчто\s+такое\b|\bчто\s+значит\b|\bкто\s+так(?:ой|ая|ие)\b"
            r"|\bwhat\s+(?:is|are|does)\b|\bdefine\b|\bdefinition\b",
            re.IGNORECASE,
        ),
    ),
    (
        "как",
        re.compile(r"(?<![\w-])как\b|\bhow\b", re.IGNORECASE),
    ),
)

#: Приветствия/болтовня: малая модель справляется, ответ дешёвый.
_GREETING_RE = re.compile(
    r"^(?:"
    r"привет\w*|здравствуй(?:те)?|добрый\s+(?:день|вечер)|доброе\s+утро|доброй\s+ночи|"
    r"хай|hello|hi|hey|спасибо|благодарю|thanks|thank\s+you|пока|до\s+свидания|"
    r"как\s+(?:дела|ты|вы|настроение|поживаешь|у\s+(?:тебя|вас)\s+дела)|"
    r"как\s+(?:тебя|вас)\s+зовут|"
    r"кто\s+ты|что\s+ты\s+умеешь|how\s+are\s+you)"
    r"(?:[\s!.,?…)]*(?:как\s+(?:дела|ты|вы|настроение|поживаешь|у\s+(?:тебя|вас)\s+дела))?"
    r"[\s!.,?…)]*)$",
    re.IGNORECASE,
)

#: Клавиатурный «мусор»: такие токены — не запрос, а случайный набор.
_MASH_TOKENS = frozenset(
    {
        "asdf",
        "asdfg",
        "asdfgh",
        "asdfghjkl",
        "qwer",
        "qwerty",
        "qwertyuiop",
        "zxcv",
        "zxcvbn",
        "zxcvbnm",
        "фыва",
        "фывап",
        "фывапр",
        "фывапролдж",
        "йцукен",
        "йцукенгшщзхъ",
        "ячсмить",
        "ячсмитьбю",
    }
)

#: Любая буква (Unicode), кроме цифр и подчёркивания.
_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)


def classify(text: str, cfg: Any = None) -> Route:
    """Определить маршрут запроса: ``skill``, ``small`` или ``big``.

    Не падает на пустом тексте, ``None`` и мусоре. Сетевых вызовов нет.
    """
    raw = "" if text is None else text
    if not isinstance(raw, str):
        raw = str(raw)
    thresholds = load_thresholds(cfg)
    stripped = raw.strip()

    if not stripped:
        return _route(SMALL, "empty_text", 0.20)

    garbage = _garbage_reason(stripped, thresholds)
    if garbage:
        return _route(SMALL, garbage, 0.25)

    if _GREETING_RE.match(stripped):
        return _route(SMALL, "greeting", 0.90)

    reasoning = _reasoning_hits(stripped, thresholds)
    explicit = _EXPLICIT_TASK_RE.search(stripped)
    if reasoning or explicit:
        labels = list(reasoning)
        if explicit:
            snippet = " ".join(explicit.group(0).split())[:32]
            if snippet.lower() not in {label.lower() for label in labels}:
                labels.append(f"«{snippet}»")
        confidence = 0.90 if len(labels) > 1 else 0.85
        return _route(BIG, "reasoning: " + ", ".join(labels[:3]), confidence)

    task = _marker_hits(stripped, thresholds["task_markers"])
    if task:
        labels = [f"«{marker}»" for marker in task[:3]]
        confidence = 0.90 if len(task) > 1 else 0.85
        return _route(BIG, "task: " + ", ".join(labels), confidence)

    catalog = _catalog_hit(stripped)
    if catalog and thresholds["catalog_fact_to_skill"]:
        confidence = 0.90 if len(stripped) <= thresholds["long_chars"] else 0.75
        return _route(SKILL, "catalog_fact: " + catalog, confidence)

    structural = _marker_hits(stripped, thresholds["structural_markers"])
    if structural:
        labels = [f"«{marker}»" for marker in structural[:3]]
        return _route(BIG, "structural: " + ", ".join(labels), 0.80)

    knowledge = _marker_hits(stripped, thresholds["knowledge_markers"])
    if knowledge:
        labels = [f"«{marker}»" for marker in knowledge[:3]]
        return _route(BIG, "general_knowledge: " + ", ".join(labels), 0.75)

    comparison = _COMPARISON_RE.search(stripped)
    if comparison:
        return _route(BIG, "comparison: " + " ".join(comparison.group(0).split()), 0.75)

    conceptual = _conceptual_hit(stripped)
    if conceptual:
        return _route(BIG, "conceptual: " + conceptual, 0.70)

    letters = _LETTER_RE.findall(stripped)
    latin = _latin_ratio(stripped)
    if len(letters) >= thresholds["latin_min_letters"] and latin > thresholds["latin_ratio"]:
        return _route(
            BIG,
            f"latin_ratio: {latin:.2f} > {thresholds['latin_ratio']:.2f}",
            0.75,
        )

    length = len(stripped)
    if length > thresholds["long_chars"]:
        return _route(BIG, f"long_text: {length} > {thresholds['long_chars']}", 0.70)

    return _route(SMALL, f"short_simple: {length} <= {thresholds['long_chars']}", 0.85)


def load_thresholds(cfg: Any = None) -> dict[str, Any]:
    """Собрать пороги из ``cfg``; отсутствующие/битые значения — по умолчанию."""
    section = _config_section(cfg)
    thresholds: dict[str, Any] = dict(DEFAULT_CLASSIFIER_THRESHOLDS)

    for key, aliases in _THRESHOLD_ALIASES.items():
        default = DEFAULT_CLASSIFIER_THRESHOLDS[key]
        for alias in aliases:
            value = _get(section, alias)
            if value is None and section is not cfg:
                value = _get(cfg, alias)
            if value is None:
                continue
            if isinstance(default, bool):
                thresholds[key] = bool(value)
            elif isinstance(default, int):
                thresholds[key] = _as_int(value, default)
            else:
                thresholds[key] = _as_float(value, default)
            break

    markers = _get(section, "reason_markers")
    if markers is None and section is not cfg:
        markers = _get(cfg, "reason_markers")
    if markers is None:
        markers = _get(section, "classifier_reason_markers")
    thresholds["reason_markers"] = _normalize_markers(markers)

    thresholds["task_markers"] = _markers_from(
        section,
        cfg,
        ("task_markers", "classifier_task_markers"),
        DEFAULT_TASK_MARKERS,
    )
    thresholds["structural_markers"] = _markers_from(
        section,
        cfg,
        ("structural_markers", "classifier_structural_markers"),
        DEFAULT_STRUCTURAL_MARKERS,
    )
    thresholds["knowledge_markers"] = _markers_from(
        section,
        cfg,
        ("knowledge_markers", "general_knowledge_markers", "classifier_knowledge_markers"),
        DEFAULT_KNOWLEDGE_MARKERS,
    )

    return thresholds


def _route(kind: str, reason: str, confidence: float) -> Route:
    return Route(kind=kind, reason=reason, confidence=confidence)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Достать значение из словаря или объекта (включая uppercase-константы)."""
    if obj is None:
        return default
    for name in (key, key.upper()):
        if isinstance(obj, Mapping):
            if name in obj:
                return obj[name]
        elif hasattr(obj, name):
            return getattr(obj, name)
    return default


def _config_section(cfg: Any) -> Any:
    """Найти секцию порогов: ``cfg.classifier`` → ``thresholds`` → сам cfg."""
    if cfg is None:
        return {}
    for name in ("classifier", "classifier_thresholds", "thresholds"):
        section = _get(cfg, name)
        if section is None:
            continue
        if isinstance(section, Mapping) and not section:
            continue
        return section
    return cfg


def _as_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return default
    return number if number >= 0 else default


def _as_float(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number or number < 0:  # NaN или отрицательное
        return default
    return number


def _markers_from(
    section: Any,
    cfg: Any,
    keys: tuple[str, ...],
    default: tuple[str, ...],
) -> tuple[str, ...]:
    """Взять список маркеров из секции конфига (или из ``cfg``) по первому ключу."""
    for key in keys:
        value = _get(section, key)
        if value is None and section is not cfg:
            value = _get(cfg, key)
        if value is not None:
            return _normalize_markers(value, default)
    return default


def _normalize_markers(
    value: Any, default: tuple[str, ...] = DEFAULT_REASON_MARKERS
) -> tuple[str, ...]:
    """Привести маркеры к кортежу строк; пустое значение — умолчания."""
    if value is None:
        return default
    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
    else:
        return default
    cleaned = tuple(str(item).strip().lower() for item in items if str(item).strip())
    return cleaned or default


def _reasoning_hits(text: str, thresholds: Mapping[str, Any]) -> list[str]:
    return _marker_hits(text, thresholds["reason_markers"])


def _marker_hits(text: str, markers: Any) -> list[str]:
    """Маркеры из списка, которые встречаются в тексте (подстрока, без регистра)."""
    haystack = text.lower()
    return [marker for marker in markers if marker and marker in haystack]


def _catalog_hit(text: str) -> str | None:
    for name, pattern in _CATALOG_RULES:
        if pattern.search(text):
            return name
    return None


def _conceptual_hit(text: str) -> str | None:
    for name, pattern in _CONCEPTUAL_RULES:
        if pattern.search(text):
            return name
    return None


def _latin_ratio(text: str) -> float:
    letters = _LETTER_RE.findall(text)
    if not letters:
        return 0.0
    latin = sum(1 for char in letters if char.isascii() and char.isalpha())
    return latin / len(letters)


def _garbage_reason(text: str, thresholds: Mapping[str, Any]) -> str | None:
    """Причина считать текст мусором или ``None``, если это нормальный запрос."""
    compact = re.sub(r"\s+", "", text)
    letters = _LETTER_RE.findall(text)
    if not letters:
        return "garbage_text: no_letters"
    if len(letters) / len(compact) < thresholds["letters_ratio_min"]:
        return "garbage_text: few_letters"
    if len(compact) >= 4 and len(set(compact)) <= 1:
        return "garbage_text: repeated_chars"
    tokens = re.findall(r"[^\W\d_]+", text.lower(), re.UNICODE)
    if tokens and all(token in _MASH_TOKENS for token in tokens):
        return "garbage_text: keyboard_mash"
    return None
