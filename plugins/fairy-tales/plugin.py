"""Плагин ``fairy-tales``: Ева рассказывает русские народные сказки.

Способность — ``answer``: сервер зовёт :func:`answer` **до** модели и, если
плагин вернул текст, не генерирует ответ вовсе. Поэтому здесь нет обращения
к модели: только распознавание просьбы, выбор сказки из офлайн-корпуса и
подготовка текста для чтения вслух.

Корпус лежит рядом, в ``tales/*.md``: шапка (``title``, ``tags``, ``age``,
``source``) и plain-text сказка. Сеть не нужна, в манифесте
``permissions.network: []``; файлы читаются через :class:`PluginContext` в
пределах ``permissions.filesystem: ["tales"]``.

Что важно для озвучки:

* никакого markdown — только слова и знаки препинания;
* числа заменяются словами (:func:`eva.ru_numbers.numbers_to_words`);
* абзацы короткие (≤ 480 символов), чтобы сервер резал текст на куски ≤ 600
  символов без разрыва посреди мысли;
* начало и концовка — мягкие, «сказочные», а не «вот ответ».

Состояние — последняя рассказанная сказка: «следующая сказка» идёт по
отсортированному списку и заворачивается на начало.
"""

from __future__ import annotations

import random
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from eva.plugins.context import PluginContext, PluginPermissionError
from eva.ru_numbers import numbers_to_words

__all__ = [
    "FairyTalesPlugin",
    "Tale",
    "answer",
    "register",
]

#: Длина абзаца: сервер режет озвучку на куски по 600 символов, поэтому
#: оставляем запас и не разрываем мысль на середине.
PARAGRAPH_LIMIT = 480

#: Сказка-концовка для обычного дня и для «на ночь».
ENDING_DAY = "Сказке конец, а кто слушал — молодец."
ENDING_NIGHT = (
    "Вот и сказке конец. Закрывай глазки и засыпай. Сладких снов."
)

#: Слова, которые не помогают выбрать сказку: служебные и «обёртки» просьбы.
_STOP_WORDS = frozenset(
    {
        "а",
        "без",
        "бы",
        "в",
        "вам",
        "вас",
        "вот",
        "все",
        "вы",
        "где",
        "да",
        "давай",
        "давайте",
        "для",
        "до",
        "другую",
        "если",
        "есть",
        "еще",
        "ещё",
        "же",
        "за",
        "и",
        "из",
        "или",
        "интересное",
        "интересную",
        "историю",
        "история",
        "как",
        "какую",
        "ко",
        "когда",
        "кто",
        "ли",
        "мне",
        "мог",
        "можешь",
        "можно",
        "мы",
        "на",
        "нам",
        "нас",
        "нашу",
        "не",
        "него",
        "нее",
        "нет",
        "ни",
        "нибудь",
        "но",
        "ночь",
        "ночи",
        "ночью",
        "ну",
        "о",
        "об",
        "одну",
        "он",
        "она",
        "они",
        "от",
        "по",
        "пожалуйста",
        "почитай",
        "почитать",
        "про",
        "просто",
        "прочти",
        "расскажи",
        "расскажите",
        "рассказывай",
        "с",
        "сам",
        "свою",
        "себе",
        "сейчас",
        "сказка",
        "сказки",
        "сказку",
        "сказочку",
        "следующая",
        "следующую",
        "сном",
        "со",
        "так",
        "тебе",
        "тебя",
        "то",
        "только",
        "ты",
        "у",
        "хочу",
        "что",
        "чтобы",
        "это",
        "я",
    }
)

#: Просьба именно сказки/истории, а не «расскажи про погоду».
_STORY_WORDS_RE = re.compile(r"сказк|сказоч|истори[юяи]|почита|прочти")
_READ_RE = re.compile(r"расскаж|почита|прочти|рассказыва")
_CHILD_RE = re.compile(r"ребенк|ребёнк|малыш|дочк|сын|внук|внучк|дет[яи]м|детск")
_NIGHT_RE = re.compile(
    r"\bна ночь\b|\bночь\b|\bночи\b|\bночью\b|перед сном|засыпа|колыбельн"
)
_NEXT_RE = re.compile(
    r"следующ|ещё сказк|еще сказк|другую сказк|другую истори"
    r"|продолж(?:и|ай)|давай ещё|давай еще|расскажи ещё|расскажи еще"
)
_TARGET_RE = re.compile(r"\b(?:про|об|о)\s+([^,.!?;:]{3,80})", re.I)

#: Окончания, которые можно снять без словаря: нужен не морфологический
#: разбор, а «лису» → «лис», «медведя» → «медвед».
_ENDINGS = tuple(
    sorted(
        {
            "ами",
            "ями",
            "ого",
            "его",
            "ому",
            "ему",
            "ыми",
            "ими",
            "ая",
            "яя",
            "ое",
            "ее",
            "ые",
            "ие",
            "ой",
            "ей",
            "ий",
            "ый",
            "ую",
            "юю",
            "ам",
            "ям",
            "ах",
            "ях",
            "ом",
            "ем",
            "ов",
            "ев",
            "ия",
            "ию",
            "а",
            "я",
            "о",
            "е",
            "ы",
            "и",
            "у",
            "ю",
            "ь",
        },
        key=len,
        reverse=True,
    )
)


@dataclass(frozen=True)
class Tale:
    """Одна сказка корпуса: шапка + уже очищенный текст."""

    slug: str
    title: str
    tags: tuple[str, ...]
    age: str
    source: str
    body: str

    @property
    def is_short(self) -> bool:
        return "короткая" in self.tags or len(self.body) <= 3000


def _normalize(text: str) -> str:
    """Нижний регистр, ё→е, одинарные пробелы: для распознавания."""
    value = str(text or "").lower().replace("ё", "е")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _words(text: str) -> list[str]:
    return re.findall(r"[а-яa-z0-9]+", _normalize(text))


def _stem(word: str) -> str:
    for ending in _ENDINGS:
        if len(word) - len(ending) >= 3 and word.endswith(ending):
            return word[: -len(ending)]
    return word


def _common_prefix(left: str, right: str) -> int:
    size = 0
    for first, second in zip(left, right):
        if first != second:
            break
        size += 1
    return size


def _same_root(left: str, right: str) -> bool:
    """Грубое сравнение слов: «лису»/«лиса», «колобка»/«колобок»."""
    if left == right:
        return True
    first, second = _stem(left), _stem(right)
    if first == second:
        return True
    prefix = _common_prefix(first, second)
    return prefix >= 4 and prefix >= min(len(first), len(second)) - 1


def _keywords(text: str) -> list[str]:
    return [word for word in _words(text) if word not in _STOP_WORDS]


def _target_words(text: str) -> list[str]:
    """Слова после «про»/«о»: «сказку про лису» → ``["лису"]``."""
    match = _TARGET_RE.search(_normalize(text))
    if not match:
        return []
    return [word for word in _words(match.group(1)) if word not in _STOP_WORDS]


def _is_story_request(text: str) -> bool:
    """Похоже ли на «расскажи сказку/историю», а не на любой вопрос."""
    value = _normalize(text)
    if _STORY_WORDS_RE.search(value):
        return True
    if _READ_RE.search(value) and _CHILD_RE.search(value):
        return True
    if _READ_RE.search(value) and re.search(r"что[- ]нибудь|интересн", value):
        return True
    return False


def _parse_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    """Разобрать простую YAML-шапку без зависимостей."""
    text = raw.lstrip("\ufeff")
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end < 0:
        return {}, text
    header = text[3:end]
    body = text[end + 4 :].lstrip("\n")
    data: dict[str, str] = {}
    for line in header.splitlines():
        key, separator, value = line.partition(":")
        if not separator:
            continue
        name = key.strip().lower()
        if name:
            data[name] = value.strip()
    return data, body


def _clean_body(body: str) -> str:
    """Убрать остатки разметки и собрать короткие абзацы."""
    value = body
    value = re.sub(r"<[^>]+>", "", value)
    value = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", value)
    value = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", value)
    value = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", value)
    value = value.replace("**", "").replace("__", "")
    value = value.replace("`", "")
    value = re.sub(r"[*_]{1,2}([^*_]+)[*_]{1,2}", r"\1", value)
    value = value.replace("\u00a0", " ").replace("\u202f", " ")
    value = value.replace("\u0301", "")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    value = "\n".join(line.strip() for line in value.splitlines())
    return _paragraphs(value)


def _paragraphs(text: str) -> str:
    """Короткие абзацы: склеить строки, длинное разрезать по предложениям."""
    blocks = re.split(r"\n\s*\n", text)
    result: list[str] = []
    for block in blocks:
        merged = " ".join(
            line.strip() for line in block.splitlines() if line.strip()
        )
        merged = re.sub(r"\s+", " ", merged).strip()
        if not merged:
            continue
        if len(merged) <= PARAGRAPH_LIMIT:
            result.append(merged)
            continue
        for sentence in re.split(r"(?<=[.!?…])\s+", merged):
            if len(sentence) <= PARAGRAPH_LIMIT:
                result.append(sentence)
                continue
            words = sentence.split()
            part = ""
            for word in words:
                if part and len(part) + 1 + len(word) > PARAGRAPH_LIMIT:
                    result.append(part)
                    part = word
                else:
                    part = f"{part} {word}".strip()
            if part:
                result.append(part)
    return "\n\n".join(result)


def _clean_speech(text: str) -> str:
    """Финальная подготовка ответа: без markdown и цифр, ровные пробелы."""
    value = numbers_to_words(str(text or ""))
    value = value.replace("**", "").replace("__", "")
    value = value.replace("`", "")
    value = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", value)
    value = re.sub(r"<[^>]+>", "", value)
    value = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", value)
    value = re.sub(r"[*_]{1,2}([^*_]+)[*_]{1,2}", r"\1", value)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r" +([,.;:!?])", r"\1", value)
    value = re.sub(r"([»!?…])[ \t]*—", r"\1 —", value)
    value = re.sub(r"—(?=\S)", "— ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return "\n".join(line.rstrip() for line in value.splitlines()).strip()


class FairyTalesPlugin:
    """Корпус сказок + выбор по просьбе + состояние «следующая»."""

    name = "fairy-tales"
    scope = "chat"

    def __init__(
        self,
        tales_dir: Path | str | None = None,
        *,
        rng: random.Random | None = None,
    ) -> None:
        self._tales_dir = Path(tales_dir) if tales_dir is not None else None
        self._rng = rng or random.Random()
        self._lock = threading.RLock()
        self._cache_key: str | None = None
        self._cache: tuple[Tale, ...] = ()
        self._last_index: int | None = None
        self._last_night = False

    # -- корпус --------------------------------------------------------------

    @property
    def last_index(self) -> int | None:
        """Индекс последней рассказанной сказки (для тестов и ``next``)."""
        with self._lock:
            return self._last_index

    def load_tales(self, ctx: PluginContext | None = None) -> tuple[Tale, ...]:
        """Прочитать корпус; битые файлы пропускаются, пустой каталог — []."""
        directory = self._directory(ctx)
        key = str(directory.resolve(strict=False))
        with self._lock:
            if key == self._cache_key:
                return self._cache
        tales: list[Tale] = []
        try:
            paths = sorted(directory.glob("*.md"))
        except OSError as exc:  # pragma: no cover - защита от странной ФС
            self._log(ctx, "корпус не прочитан: %s", exc)
            paths = []
        for path in paths:
            try:
                raw = self._read(ctx, path)
            except (OSError, PluginPermissionError) as exc:
                self._log(ctx, "сказка %s не прочитана: %s", path.name, exc)
                continue
            tale = self._parse(raw, path)
            if tale is not None:
                tales.append(tale)
        result = tuple(tales)
        with self._lock:
            self._cache_key = key
            self._cache = result
            if self._last_index is not None and self._last_index >= len(result):
                self._last_index = None
        return result

    def _directory(self, ctx: PluginContext | None) -> Path:
        if self._tales_dir is not None:
            return self._tales_dir
        if ctx is None:
            return Path(__file__).resolve().parent / "tales"
        return Path(ctx.plugin_dir) / "tales"

    def _read(self, ctx: PluginContext | None, path: Path) -> str:
        if self._tales_dir is None and ctx is not None:
            # Штатный путь: контекст проверяет permissions.filesystem.
            return ctx.read_text(Path("tales") / path.name, encoding="utf-8")
        return path.read_text(encoding="utf-8")

    def _parse(self, raw: str, path: Path) -> Tale | None:
        data, body = _parse_frontmatter(raw)
        title = data.get("title", "").strip()
        if not title:
            return None
        tags = tuple(
            tag.strip().lower()
            for tag in re.split(r"[,;]", data.get("tags", ""))
            if tag.strip()
        )
        cleaned = _clean_body(body)
        if not cleaned:
            return None
        return Tale(
            slug=path.stem,
            title=title,
            tags=tags,
            age=data.get("age", "").strip(),
            source=data.get("source", "").strip(),
            body=cleaned,
        )

    # -- выбор ---------------------------------------------------------------

    def _score(self, tale: Tale, keywords: Sequence[str]) -> int:
        title_words = _words(tale.title)
        tag_words = [word for tag in tale.tags for word in _words(tag)]
        score = 0
        for keyword in keywords:
            if any(_same_root(keyword, word) for word in title_words):
                score += 3
            elif any(_same_root(keyword, word) for word in tag_words):
                score += 2
        return score

    def _ranked(self, tales: Sequence[Tale], keywords: Sequence[str]) -> list[Tale]:
        scored = []
        if keywords:
            for tale in tales:
                score = self._score(tale, keywords)
                if score:
                    scored.append((score, tale))
        # Выше балл; при равенстве — короче название (и текст): «про медведя»
        # скорее «Три медведя», чем длинное «Мужик, медведь и лиса».
        scored.sort(
            key=lambda item: (
                -item[0],
                len(item[1].title),
                len(item[1].body),
                item[1].slug,
            )
        )
        return [tale for _score, tale in scored]

    def _pick_short(self, tales: Sequence[Tale], last: int | None) -> int:
        short = [index for index, tale in enumerate(tales) if tale.is_short]
        candidates = short or list(range(len(tales)))
        if len(candidates) > 1 and last in candidates:
            candidates = [index for index in candidates if index != last]
        return self._rng.choice(candidates)

    # -- ответ ---------------------------------------------------------------

    def answer(self, ctx: PluginContext, question: str) -> str | None:
        """Текст сказки для чтения вслух или ``None`` — пусть отвечает модель."""
        raw = str(question or "").strip()
        if not raw:
            return None
        text = _normalize(raw)
        next_request = bool(_NEXT_RE.search(text))
        tales = self.load_tales(ctx)
        if not tales:
            return None

        with self._lock:
            last = self._last_index
            if last is None or last >= len(tales):
                last = None
            night = bool(_NIGHT_RE.search(text)) or (
                next_request and self._last_night
            )

            if next_request and last is not None:
                index = (last + 1) % len(tales)
            else:
                keywords = _keywords(text)
                targets = _target_words(text)
                ranked = self._ranked(tales, keywords)
                if targets:
                    if not ranked:
                        # «сказку про слона»: такой сказки нет — не подменяем
                        # её случайной, пусть модель ответит честно.
                        return None
                    index = tales.index(ranked[0])
                elif _is_story_request(text):
                    index = self._pick_short(tales, last)
                else:
                    return None

            self._last_index = index
            self._last_night = night
            return self.render(tales[index], night=night)

    def render(self, tale: Tale, *, night: bool = False) -> str:
        """Собрать ответ: «Слушай сказку…», текст и мягкая концовка."""
        intro = "Слушай сказку"
        if night:
            intro += " на ночь"
        intro += f"… «{tale.title}»."
        ending = ENDING_NIGHT if night else ENDING_DAY
        text = f"{intro}\n\n{tale.body}\n\n{ending}"
        return _clean_speech(text)

    @staticmethod
    def _log(ctx: PluginContext | None, message: str, *args: object) -> None:
        log = getattr(ctx, "log", None)
        if log is None:
            return
        try:
            log.warning(message, *args)
        except Exception:  # noqa: BLE001 - лог не должен ронять ответ
            pass


_PLUGIN = FairyTalesPlugin()


def answer(ctx: PluginContext, question: str) -> str | None:
    """Точка входа способности ``answer``: сказка или ``None``."""
    return _PLUGIN.answer(ctx, question)


def register(ctx: PluginContext) -> FairyTalesPlugin:
    """Вернуть объект с ``answer(ctx, question)`` (реестр зовёт его)."""
    ctx.log.debug("fairy-tales: корпус готов, сеть не нужна")
    return _PLUGIN
