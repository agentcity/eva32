"""RAG по каталогу EVA32: контекст для модели без сети и эмбеддингов.

Контракт — ``docs/hybrid.md``, решение — ``docs/adr/0009-hybrid-router.md``,
задача ``ai/tasks/0024-eva-rag.md``.

Зачем модуль: скиллы отвечают на прямые вопросы, но на «а что входит в
Профи» модель должна видеть каталог. Скилл тут молчит (вопрос не про цену),
а модель без фактов начинает сочинять. Поэтому перед вызовом модели
маршрутизатор достаёт из каталога 1–3 релевантные позиции и подмешивает их
в сообщения как блок «Контекст».

Поиск намеренно простой и полностью локальный:

* нормализация — регистр, ``ё``/``е``, кавычки, дефисы (``catalog.normalize``);
* слова — ``catalog.tokens``, грубая морфология — ``catalog.stem``;
* синонимы берутся из карточки каталога и индексируются наравне с именем;
* счёт — пересечение слов запроса с полями карточки, вес поля:
  ``name`` > ``synonyms`` > ``limits`` > ``notes``;
* редкие слова весят больше частых (простой IDF по каталогу);
* точное вхождение имени/синонима фразой в запрос даёт бонус;
* мусор отсекается покрытием запроса (доля найденных значимых слов) и
  отсечкой по относительному счёту — «погода в Москве» не тянет доставку
  только из-за слова «Москве».

Формат результата (``RagContext``)::

    Контекст (каталог EVA32; ссылайся на факты как [n]):

    [1] Тариф «Профи» — 2 490 ₽/мес
        лимиты: до 10 сотрудников; 500 услуг в каталоге
        заметки: при оплате за год скидка 20%; API не предусмотрен

    Источники: [1]

``sources`` — источники позиций (``[1]`` из карточки), ``ids`` — id позиций
каталога. Нумерация ``[n]`` — номер позиции в блоке: именно на него модель
ссылается в ответе (правило 2 системного промпта).

Только стандартная библиотека, никакой сети.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .catalog import Catalog, alias_in_tokens, load, stem, tokens

__all__ = [
    "COVERAGE_MIN",
    "DEFAULT_K",
    "FIELD_WEIGHTS",
    "HEADER",
    "MIN_SCORE",
    "RagContext",
    "retrieve",
]

#: Сколько позиций отдавать по умолчанию: больше трёх уже размывает промпт.
DEFAULT_K = 3

#: Вес поля карточки при совпадении слова. Порядок — из задачи:
#: имя важнее синонимов, синонимы важнее лимитов и заметок.
FIELD_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("name", 10.0),
    ("synonyms", 6.0),
    ("limits", 2.0),
    ("notes", 1.5),
)

#: Заголовок блока: системный промпт ищет в сообщениях именно «Контекст».
HEADER = "Контекст (каталог EVA32; ссылайся на факты как [n]):"

#: Минимальный итоговый счёт позиции. Ниже — это уже случайное совпадение
#: служебного слова, а не факт из каталога; порог низкий, потому что IDF
#: зависит от размера каталога и на маленьком каталоге счёт меньше.
MIN_SCORE = 1.0

#: Насколько позиция может отставать от лучшей (доля счёта), чтобы попасть
#: в контекст: «тариф Профи» не должен тянуть «Старт» и «Бизнес».
RELATIVE_CUTOFF = 0.5

#: Какую долю значимых слов запроса позиция обязана покрыть. 1/2 = 0.5 —
#: мало: «погода в Москве» не про доставку. 2/3 и выше — уже факт.
COVERAGE_MIN = 0.6

#: Бонус за точное вхождение имени/синонима фразой («тариф Профи»).
PHRASE_BONUS = 5.0

#: Минимальная длина основы для нечёткого совпадения: основа одной формы
#: может быть началом другой («аудит»/«аудитор»), а общее начало не короче
#: ``PREFIX_MIN_LEN + 1`` ловит «настроит»/«настройк».
PREFIX_MIN_LEN = 5

#: Слова, которые не несут факта: вопрос, вежливость, связки, просьбы.
#: Сравниваются по основе, поэтому здесь достаточно одной формы.
_STOPWORDS = frozenset(
    stem(word)
    for word in (
        # вопрос и служебное
        "что", "чего", "чему", "чем", "кто", "кого", "кому", "где", "когда",
        "куда", "откуда", "почему", "зачем", "как", "какой", "какая", "какое",
        "какие", "каких", "каким", "который", "которая", "которое", "которые",
        "которых", "сколько", "столько", "ли", "же", "бы", "не", "ни", "и",
        "а", "но", "или", "либо", "в", "во", "на", "за", "по", "с", "со",
        "для", "от", "до", "из", "у", "к", "ко", "о", "об", "обо", "при",
        "про", "без", "над", "под", "через", "это", "этот", "эта", "эти",
        "тот", "та", "те", "такой", "такая", "такое", "такие", "там", "тут",
        "здесь", "сейчас", "вообще", "тоже", "также", "еще", "ещё", "уже",
        "только", "просто", "можно", "нельзя", "нужно", "нужен", "нужна",
        "нужны", "надо", "есть", "был", "была", "было", "были", "будет",
        "будут", "хочу", "хотим", "интересует",
        # вежливость и просьбы
        "пожалуйста", "спасибо", "здравствуйте", "привет", "добрый", "доброе",
        "день", "утро", "вечер", "подскажите", "подскажи", "скажите",
        "скажи", "расскажите", "расскажи", "покажите", "покажи", "дайте",
        "опишите", "опиши", "объясните", "объясни", "напишите", "напиши",
        # местоимения и притяжательные
        "мне", "меня", "мой", "моя", "мои", "наш", "наша", "наши", "ваш",
        "ваша", "ваши", "вас", "вам", "вы", "мы", "нас", "нам", "я", "ты",
        "он", "она", "оно", "они", "их", "его", "ее", "её", "ним", "них",
        "него", "нее", "всё", "все", "вся", "всех", "всем",
        # общие глаголы и оценки, не несущие факта из карточки
        "входит", "входят", "включает", "включают", "включен", "включена",
        "включено", "стоит", "стоят", "стоить", "стоимость", "цена", "цены",
        "цену", "прайс", "почем", "работает", "работают", "работать",
        "быстро", "срочно", "сегодня", "завтра", "хорошо", "плохо",
    )
)


@dataclass
class RagContext:
    """Контекст каталога для промпта. Пустой ``text`` = контекста нет."""

    text: str = ""
    sources: list[str] = field(default_factory=list)
    ids: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.text.strip())

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "sources": list(self.sources), "ids": list(self.ids)}


# --------------------------------------------------------------------------- #
# публичный API
# --------------------------------------------------------------------------- #
def retrieve(text: str, catalog: Any | None = None, k: int = DEFAULT_K) -> RagContext:
    """Найти в каталоге до ``k`` позиций по запросу и собрать контекст.

    ``catalog`` — ``Catalog``, путь к JSON, список позиций или ``None``
    (тогда ``data/catalog.json``). Ошибка данных не роняет запрос: контекст
    просто пустой — по ``docs/hybrid.md`` модель обязана работать и без него.

    Возвращает :class:`RagContext`; ``text`` пустой, если ничего значимого
    не нашлось (вопрос вне каталога, пустой запрос, битый каталог).
    """
    query = str(text or "")
    if not query.strip():
        return RagContext()

    try:
        k = int(k)
    except (TypeError, ValueError):
        k = DEFAULT_K
    if k <= 0:
        return RagContext()

    try:
        items = _as_items(catalog)
    except (OSError, ValueError, TypeError):
        return RagContext()
    if not items:
        return RagContext()

    query_tokens = tokens(query)
    terms = _query_terms(query_tokens)
    if not terms:
        return RagContext()

    indexed = [_index_item(item) for item in items]
    weights = _term_weights(terms, indexed, len(items))
    selected = _select(query_tokens, terms, indexed, weights, k)
    if not selected:
        return RagContext()
    return _build_context(selected)


# --------------------------------------------------------------------------- #
# поиск
# --------------------------------------------------------------------------- #
def _query_terms(query_tokens: Iterable[str]) -> list[str]:
    """Значимые основы запроса без повторов и служебных слов."""
    result: list[str] = []
    seen: set[str] = set()
    for token in query_tokens:
        base = stem(token)
        if not base or base in _STOPWORDS or base in seen:
            continue
        seen.add(base)
        result.append(base)
    return result


def _term_weights(terms: list[str], indexed: list[dict[str, Any]], size: int) -> dict[str, float]:
    """Простой IDF: слово, встречающееся во всех карточках, почти не весит."""
    weights: dict[str, float] = {}
    for term in terms:
        df = sum(1 for entry in indexed if _entry_has(entry, term))
        weights[term] = math.log(1.0 + size / max(1, df))
    return weights


def _select(
    query_tokens: list[str],
    terms: list[str],
    indexed: list[dict[str, Any]],
    weights: Mapping[str, float],
    k: int,
) -> list[dict[str, Any]]:
    """Оценить позиции, отсеять мусор и вернуть лучшие ``k``."""
    ranked: list[tuple[float, int, dict[str, Any]]] = []
    for position, entry in enumerate(indexed):
        score, matched = _score(entry, terms, weights)
        if score <= 0.0:
            continue
        coverage = len(matched) / len(terms)
        if coverage < COVERAGE_MIN:
            continue
        if _phrase_match(query_tokens, entry["aliases"]):
            score += PHRASE_BONUS
        # Позиция, покрывающая запрос целиком, важнее позиции, зацепившей
        # одно слово: «доставка документов по Москве» — это Москва, а не
        # «за МКАД» и не «по России».
        final = score * (0.5 + 0.5 * coverage)
        if final >= MIN_SCORE:
            ranked.append((final, position, entry))

    if not ranked:
        return []
    ranked.sort(key=lambda row: (-row[0], row[1]))
    best = ranked[0][0]
    cutoff = max(MIN_SCORE, best * RELATIVE_CUTOFF)
    selected = [entry for row_score, _, entry in ranked if row_score >= cutoff][:k]
    return selected


def _score(
    entry: dict[str, Any],
    terms: list[str],
    weights: Mapping[str, float],
) -> tuple[float, set[str]]:
    """Счёт позиции: сумма весов полей по совпавшим значимым словам."""
    score = 0.0
    matched: set[str] = set()
    for term in terms:
        best_weight = 0.0
        for name, weight in FIELD_WEIGHTS:
            if _term_in(term, entry["fields"][name]):
                best_weight = max(best_weight, weight)
        if best_weight > 0.0:
            score += best_weight * weights.get(term, 0.0)
            matched.add(term)
    return score, matched


def _term_in(term: str, known: set[str]) -> bool:
    """Совпадает ли основа запроса с основой слова карточки."""
    if term in known:
        return True
    if len(term) < PREFIX_MIN_LEN:
        return False
    for base in known:
        if len(base) < PREFIX_MIN_LEN:
            continue
        if term.startswith(base) or base.startswith(term):
            return True
        if _common_prefix_len(term, base) >= PREFIX_MIN_LEN + 1:
            return True
    return False


def _common_prefix_len(left: str, right: str) -> int:
    """Длина общего начала двух основ: «настроит»/«настройк» → 6."""
    length = 0
    for first, second in zip(left, right):
        if first != second:
            break
        length += 1
    return length


def _entry_has(entry: dict[str, Any], term: str) -> bool:
    return any(_term_in(term, entry["fields"][name]) for name, _ in FIELD_WEIGHTS)


def _phrase_match(query_tokens: list[str], aliases: list[str]) -> bool:
    """Есть ли имя/синоним из нескольких слов в запросе подряд."""
    for alias in aliases:
        alias_tokens = tokens(alias)
        if len(alias_tokens) >= 2 and alias_in_tokens(query_tokens, alias):
            return True
    return False


# --------------------------------------------------------------------------- #
# каталог
# --------------------------------------------------------------------------- #
_DEFAULT_CATALOG: Catalog | None = None


def _default_catalog() -> Catalog:
    """Общий каталог ``data/catalog.json`` (кэш на процесс)."""
    global _DEFAULT_CATALOG
    if _DEFAULT_CATALOG is None:
        _DEFAULT_CATALOG = load()
    return _DEFAULT_CATALOG


def _as_items(catalog: Catalog | Any | None) -> list[dict[str, Any]]:
    """Привести каталог к списку позиций: объект, путь, список или ``None``."""
    if catalog is None:
        catalog = _default_catalog()
    if isinstance(catalog, (str, Path)):
        catalog = load(catalog)
    if isinstance(catalog, (bytes, bytearray)):
        raise TypeError("catalog: ожидался Catalog, путь или список позиций")
    if isinstance(catalog, Mapping):
        raise TypeError("catalog: ожидался Catalog, путь или список позиций")
    if isinstance(catalog, (list, tuple)):
        return [dict(item) for item in catalog if isinstance(item, Mapping)]
    all_items = getattr(catalog, "all", None)
    if callable(all_items):
        return [dict(item) for item in all_items() if isinstance(item, Mapping)]
    raise TypeError("catalog: ожидался Catalog, путь или список позиций")


def _index_item(item: dict[str, Any]) -> dict[str, Any]:
    """Поля карточки, разложенные по основам слов, плюс фразы для бонуса."""
    name = str(item.get("name") or "")
    synonyms = _as_list(item.get("synonyms"))
    fields: dict[str, set[str]] = {}
    for field_name, _ in FIELD_WEIGHTS:
        if field_name == "name":
            values = [name]
        elif field_name == "synonyms":
            values = synonyms
        else:
            values = _as_list(item.get(field_name))
        bases: set[str] = set()
        for value in values:
            for token in tokens(value):
                base = stem(token)
                if base:
                    bases.add(base)
        fields[field_name] = bases
    return {
        "item": item,
        "fields": fields,
        "aliases": [value for value in [name, *synonyms] if value.strip()],
    }


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    return []


# --------------------------------------------------------------------------- #
# форматирование
# --------------------------------------------------------------------------- #
def _build_context(selected: list[dict[str, Any]]) -> RagContext:
    """Собрать текст блока, источники и id позиций с нумерацией с единицы."""
    lines: list[str] = [HEADER, ""]
    sources: list[str] = []
    ids: list[str] = []
    for number, entry in enumerate(selected, 1):
        item = entry["item"]
        lines.append(_item_line(number, item))
        for label, field_name in (("лимиты", "limits"), ("заметки", "notes")):
            values = _as_list(item.get(field_name))
            if values:
                lines.append(f"    {label}: {'; '.join(values)}")
        for source in _as_list(item.get("sources")):
            if source not in sources:
                sources.append(source)
        item_id = str(item.get("id") or "").strip() or f"item-{number}"
        if item_id not in ids:
            ids.append(item_id)
    if sources:
        lines.extend(["", f"Источники: {', '.join(sources)}"])
    return RagContext(text="\n".join(lines), sources=sources, ids=ids)


def _item_line(number: int, item: Mapping[str, Any]) -> str:
    label = "Тариф" if str(item.get("kind") or "").lower() == "tariff" else "Услуга"
    name = _quote(str(item.get("name") or "").strip())
    price = _format_price(item)
    return f"[{number}] {label} {name}" + (f" — {price}" if price else "")


def _quote(name: str) -> str:
    return f"«{name}»" if name else "«»"


def _format_price(item: Mapping[str, Any]) -> str:
    amount = item.get("price_rub")
    if isinstance(amount, bool) or amount is None:
        return ""
    try:
        value = float(amount)
    except (TypeError, ValueError):
        return ""
    if value.is_integer():
        number = f"{int(value):,}".replace(",", " ")
    else:
        number = f"{value:.2f}".rstrip("0").rstrip(".").replace(".", ",")
    suffix = {
        "month": "/мес",
        "order": " за заказ",
        "once": "",
    }.get(str(item.get("period") or ""), "")
    prefix = str(item.get("price_prefix") or "").strip()
    price = f"{number} ₽{suffix}"
    return f"{prefix} {price}" if prefix else price
