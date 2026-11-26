"""Плагин ``fairy-tales``: Ева рассказывает русские народные сказки.

Способность — ``answer``: сервер зовёт :func:`answer` **до** модели и, если
плагин вернул текст, не генерирует ответ вовсе. Поэтому здесь нет обращения
к модели: только распознавание просьбы, выбор сказки из офлайн-корпуса и
подготовка текста для чтения вслух.

Корпус лежит рядом, в ``tales/*.md``: шапка (``title``, ``tags``, ``age``,
``source``) и plain-text сказка. Иллюстрации — в ``images/<slug>/NN.jpg``
(задача 0221). Сеть не нужна, в манифесте ``permissions.network: []``; файлы
читаются через :class:`PluginContext` в пределах
``permissions.filesystem: ["tales", "images"]``.

Что важно для озвучки:

* никакого markdown — только слова и знаки препинания;
* числа заменяются словами (:func:`eva.ru_numbers.numbers_to_words`);
* абзацы короткие (≤ 480 символов), чтобы сервер резал текст на куски ≤ 600
  символов без разрыва посреди мысли;
* начало и концовка — мягкие, «сказочные», а не «вот ответ».

Состояние — последняя рассказанная сказка: «следующая сказка» идёт по
отсортированному списку и заворачивается на начало.

Готовая запись (задача 0223) — отдельная ветка: «включи запись сказки»
запускает локальный файл ``audio/<slug>.mp3`` через ``afplay`` и возвращает
``{"text": ..., "silent": true}``, чтобы сервер не озвучивал подтверждение
синтезом поверх записи. «Выключи»/«стоп сказку» завершают процесс.
"""

from __future__ import annotations

import random
import re
import signal
import subprocess
import threading
import time
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

#: «Короткая сказка» для явной просьбы: не длиннее этого числа символов.
SHORT_LIMIT = 800

#: Верхняя граница для случайной сказки «на каждый день»: длинные варианты
#: остаются в корпусе, но не выпадают первыми без просьбы.
DEFAULT_LIMIT = 3500

#: Сказка-концовка для обычного дня и для «на ночь».
ENDING_DAY = "Сказке конец, а кто слушал — молодец."
ENDING_NIGHT = (
    "Вот и сказке конец. Закрывай глазки и засыпай. Сладких снов."
)

#: Каталог иллюстраций внутри плагина: ``images/<slug>/NN.jpg``.
IMAGES_DIR = "images"
#: Сколько картинок максимум показываем за одну сказку (задача 0228: 6–10).
MAX_IMAGES = 10
_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})

#: Готовые записи сказок (задача 0223): ``audio/<slug>.mp3|ogg``.
AUDIO_DIR = "audio"
AUDIO_SUFFIXES = (".mp3", ".ogg", ".m4a", ".wav")
#: Потолок файла записи: 3 МБ, чтобы плагин оставался лёгким.
AUDIO_MAX_BYTES = 3_000_000
#: Плеер macOS; вызов — только списком argv, без shell.
AFPLAY = "/usr/bin/afplay"
OSASCRIPT = "/usr/bin/osascript"
#: Ниже этой громкости системы запись почти не слышно — поднимаем при первом
#: включении и говорим об этом, чтобы не было «включила, но тихо».
QUIET_THRESHOLD = 25
DEFAULT_LOUD_VOLUME = 65

#: Просьба именно записи, а не пересказа: «включи запись сказки»,
#: «поставь аудиосказку», «сказку в исполнении». Проверяется до выбора.
_AUDIO_RE = re.compile(
    r"аудио\s*сказк"
    r"|сказк\w*\s+в\s+(?:исполнени|записи|аудио)"
    r"|(?:включи|поставь|запусти|вруби|включай|давай)\s+"
    r"(?:мне\s+|нам\s+|пожалуйста\s+)*"
    r"(?:запись|записи|аудиозапись|аудио)\s+"
    r"(?:сказк|про\s+)"
    r"|(?:включи|поставь|запусти|вруби)\s+(?:мне\s+|нам\s+)?"
    r"(?:сказк\w*\s+)?(?:в\s+)?(?:записи|исполнении)"
)
#: «Стоп сказку», «выключи запись» — только про сказку/запись, не про музыку.
_AUDIO_STOP_RE = re.compile(
    r"^(?:выключи|отключи|останови|стоп|прекрати|хватит)\s+"
    r"(?:пожалуйста\s+)?(?:эту\s+|это\s+|всё\s+|все\s+)?"
    r"(?:сказк\w*|запис\w*|аудио\w*|воспроизведени\w*|чтени\w*)"
    r"(?:\s+\S+){0,3}$"
)
#: Голое «выключи»/«стоп» останавливает запись, только если она играет:
#: иначе команду должна получить музыка (у неё свой плагин).
_AUDIO_STOP_BARE_RE = re.compile(
    r"^(?:выключи|стоп|останови|прекрати|тишина)$"
)
#: Слова-обёртки аудио-просьбы, чтобы они не влияли на выбор сказки.
_AUDIO_FILLERS = frozenset(
    {
        "аудио",
        "аудиосказка",
        "аудиосказку",
        "аудиосказки",
        "включи",
        "включай",
        "вруби",
        "давай",
        "запись",
        "записи",
        "исполнении",
        "исполнение",
        "поставь",
        "послушать",
        "послушаем",
        "запусти",
    }
)

#: Слаг каталога картинок по названию сказки. Нужен, потому что корпус
#: (задача 0220) может назвать файл иначе, а картинки привязаны к сюжету.
_IMAGE_ALIASES: tuple[tuple[str, str], ...] = (
    ("колобок", "kolobok"),
    ("репка", "repka"),
    ("теремок", "teremok"),
    ("лисаижуравль", "lisa-i-zhuravl"),
    ("тримедведя", "tri-medvedya"),
    ("гусилебеди", "gusi-lebedi"),
    ("царевналягушка", "tsarevna-lyagushka"),
    ("бабаяга", "baba-yaga"),
    ("сивкабурка", "sivka-burka"),
    ("сивкобурко", "sivka-burka"),
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
_MAGIC_RE = re.compile(r"волшебн")
_TODDLER_RE = re.compile(r"малыш|маленьк|для маленьких|ребенк|ребёнк")
_SHORT_RE = re.compile(r"коротк")
_SCARY_RE = re.compile(r"страшн|жутк|ужасн")
#: «Громче/тише» и «громкость 70» — управление системным звуком Mac: запись
#: играет afplay, а его громкость пользователю не видна (жалоба: «очень тихо,
#: непонятно как регулировать»).
_LOUDER_RE = re.compile(r"громче|погромче|прибавь\s+звук|сделай\s+громче")
_QUIETER_RE = re.compile(r"тише|потише|убавь\s+звук|сделай\s+тише")
_VOLUME_RE = re.compile(r"громкость\s*(?:на\s*)?(\d{1,3})|звук\s*(?:на\s*)?(\d{1,3})")
_MUTE_RE = re.compile(r"выключи\s+звук|без\s+звука|молчком|mute")
#: «Какие сказки есть», «список сказок» — перечень, чтобы можно было выбрать.
_LIST_RE = re.compile(
    r"(?:какие|какой|что\s+за)\s+(?:у\s+тебя\s+)?(?:есть\s+)?сказк"
    r"|список\s+сказок|перечисли\s+сказк|покажи\s+сказк"
)
#: «Сказка номер 3», «третью сказку» — выбор по номеру из списка.
_NUMBER_RE = re.compile(r"(?:номер|№)\s*(\d{1,2})|сказк[ау]?\s+(\d{1,2})")

_COUNT_RE = re.compile(
    r"сколько\s+(?:у\s+тебя\s+)?(?:есть\s+)?сказ"
    r"|сколько\s+сказок"
    r"|много\s+ли\s+(?:у\s+тебя\s+)?сказ"
)

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
        """Явно «короткая» сказка: помещается в один-два куска озвучки."""
        return len(self.body) <= SHORT_LIMIT

    @property
    def is_magic(self) -> bool:
        return "волшебная" in self.tags

    @property
    def is_scary(self) -> bool:
        """Страшную сказку не выбираем случайно и без явной просьбы."""
        return "страшная" in self.tags


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


def _age_number(age: str) -> int:
    """«3+» → 3; пустое или странное значение — 99 (не «малыш»)."""
    match = re.match(r"\s*(\d+)", str(age or ""))
    return int(match.group(1)) if match else 99


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


def _title_key(text: str) -> str:
    """Ключ названия без регистра, ё и разделителей: «Царевна-лягушка»."""
    return re.sub(r"[^а-яa-z0-9]+", "", _normalize(text))


def _paragraph_ends(text: str) -> list[int]:
    """Концы абзацев ``text``: позиции сразу после ``\\n\\n``.

    Последний абзац (концовка) не входит: картинка после него оказалась бы
    за пределами текста. Все значения строго внутри ``0 < position < len``.
    """
    ends: list[int] = []
    offset = 0
    for part in text.split("\n\n")[:-1]:
        offset += len(part) + 2
        if 0 < offset < len(text):
            ends.append(offset)
    return ends


def _sentence_ends(part: str) -> list[int]:
    """Концы предложений внутри абзаца, относительно его начала."""
    return [
        match.end()
        for match in re.finditer(r"[.!?…]+(?=\s)", part)
        if 0 < match.end() < len(part)
    ]


def _image_candidates(text: str) -> list[int]:
    """Границы абзацев плюс концы предложений в длинных абзацах.

    Картинок может быть больше, чем абзацев (сказка в диалогах: «— Кто, кто
    в терему?»). Тогда длинный абзац получает вторую картинку на границе
    предложения — мысль всё равно не рвётся посередине.
    """
    candidates = _paragraph_ends(text)
    offset = 0
    for part in text.split("\n\n")[:-1]:
        start = offset
        offset += len(part) + 2
        if len(part) < 160:
            continue
        for end in _sentence_ends(part):
            position = start + end
            if 0 < position < len(text):
                candidates.append(position)
    return sorted(set(candidates))


def _image_positions(text: str, count: int) -> list[int]:
    """Где показать ``count`` картинок: ровно по границам абзацев.

    ``after_chars`` считается по символам уже очищенного текста (того, что
    уходит в озвучку). Позиции строго возрастают и остаются внутри текста;
    при нехватке абзацев берутся концы предложений внутри длинных абзацев.
    """
    length = len(text)
    if count <= 0 or length <= 0:
        return []
    boundaries = _paragraph_ends(text)
    candidates = (
        boundaries if count <= len(boundaries) else _image_candidates(text)
    )
    if not candidates:
        return [length // 2] if length > 1 else []
    if count >= len(candidates):
        # Картинок больше, чем границ: повторы недопустимы, поэтому берём
        # все границы (лишние картинки просто не покажутся).
        return candidates[:count]
    if count == 1:
        return [candidates[(len(candidates) - 1) // 2]]
    last = len(candidates) - 1
    indices = [round(index * last / (count - 1)) for index in range(count)]
    return [candidates[index] for index in indices]


class FairyTalesPlugin:
    """Корпус сказок + выбор по просьбе + состояние «следующая»."""

    name = "fairy-tales"
    scope = "chat"

    def __init__(
        self,
        tales_dir: Path | str | None = None,
        *,
        audio_dir: Path | str | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._tales_dir = Path(tales_dir) if tales_dir is not None else None
        self._audio_dir = Path(audio_dir) if audio_dir is not None else None
        self._rng = rng or random.Random()
        self._lock = threading.RLock()
        self._cache_key: str | None = None
        self._cache: tuple[Tale, ...] = ()
        self._last_index: int | None = None
        self._last_night = False
        # Запись сказки (задача 0223): процесс afplay и слаг играющего файла.
        self._audio_lock = threading.RLock()
        self._audio_process: subprocess.Popen[bytes] | None = None
        self._audio_slug: str | None = None
        # Момент последней остановки: сервер зовёт answer для двух колонок,
        # и вторая должна получить тот же ответ, а не уйти к модели.
        self._audio_stopped_at: float | None = None

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

    def _pool(
        self,
        tales: Sequence[Tale],
        text: str,
        *,
        allow_scary: bool,
    ) -> list[Tale]:
        """Отфильтровать корпус под просьбу: страшное, волшебное, малышу, короткое."""
        pool = list(tales)
        if allow_scary:
            # Явная просьба: только страшные. Если их нет — не подменяем
            # случайной доброй сказкой, пусть ответит модель.
            pool = [tale for tale in pool if tale.is_scary]
        else:
            pool = [tale for tale in pool if not tale.is_scary]
        if _MAGIC_RE.search(text):
            pool = [tale for tale in pool if tale.is_magic]
        if _TODDLER_RE.search(text):
            pool = [tale for tale in pool if _age_number(tale.age) <= 3]
        if _SHORT_RE.search(text):
            pool = [tale for tale in pool if tale.is_short]
        return pool

    def _pick_random(
        self,
        tales: Sequence[Tale],
        pool: Sequence[Tale],
        last: int | None,
        *,
        prefer_short: bool = False,
    ) -> int:
        candidates = list(pool)
        if prefer_short:
            short = [tale for tale in candidates if len(tale.body) <= DEFAULT_LIMIT]
            candidates = short or candidates
        indices = [tales.index(tale) for tale in candidates]
        if len(indices) > 1 and last in indices:
            indices = [index for index in indices if index != last]
        return self._rng.choice(indices)

    def _next_index(
        self,
        tales: Sequence[Tale],
        pool: Sequence[Tale],
        last: int,
    ) -> int:
        indices = sorted(tales.index(tale) for tale in pool)
        for index in indices:
            if index > last:
                return index
        return indices[0]

    @staticmethod
    def _tale_word(count: int) -> str:
        if count % 10 == 1 and count % 100 != 11:
            return "сказка"
        if 2 <= count % 10 <= 4 and not 12 <= count % 100 <= 14:
            return "сказки"
        return "сказок"

    def render_list(self, tales: Sequence[Tale], limit: int = 10) -> str:
        """Перечень сказок с номерами: ребёнку проще выбрать «номер 3».

        Показываем названия, а не слаги; страшные — в перечне, но с пометкой:
        выбор по номеру уважает явный запрос, случайный — нет.
        """
        safe = [tale for tale in tales if not tale.is_scary]
        shown = safe[:limit]
        parts = [f"{index + 1}. {tale.title}" for index, tale in enumerate(shown)]
        tail = ""
        if len(safe) > limit:
            tail = f" И ещё {len(safe) - limit} сказок — скажи «следующая»."
        return "У меня есть сказки: " + "; ".join(parts) + "." + tail

    def _answer_by_number(self, tales: Sequence[Tale], hit: "re.Match[str]") -> str | None:
        """«Сказка номер 3» — выбрать по номеру из списка (без страшных)."""
        raw = hit.group(1) or hit.group(2)
        try:
            number = int(raw)
        except (TypeError, ValueError):
            return None
        safe = [tale for tale in tales if not tale.is_scary]
        if not 1 <= number <= len(safe):
            return None
        tale = safe[number - 1]
        with self._lock:
            self._last_index = tales.index(tale)
            self._last_night = False
        return self.render(tale)

    def _system_volume(self) -> int | None:
        """Текущая громкость выхода в процентах или ``None``."""
        try:
            out = subprocess.run(
                [OSASCRIPT, "-e", "output volume of (get volume settings)"],
                capture_output=True, timeout=5, check=False,
            )
            value = out.stdout.decode("utf-8", "replace").strip()
            return int(value) if value.isdigit() else None
        except (OSError, subprocess.SubprocessError, ValueError):
            return None

    def _set_system_volume(self, percent: int) -> bool:
        """Поставить громкость выхода (0–100) через osascript."""
        value = max(0, min(100, int(percent)))
        try:
            out = subprocess.run(
                [OSASCRIPT, "-e", f"set volume output volume {value}"],
                capture_output=True, timeout=5, check=False,
            )
            return out.returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False

    def _answer_volume(self, text: str) -> str | None:
        """«Громче/тише/громкость 70/выключи звук» — управление звуком Mac."""
        if _MUTE_RE.search(text):
            if self._set_system_volume(0):
                return "Выключила звук. Скажи «громче», чтобы вернуть."
            return None
        hit = _VOLUME_RE.search(text)
        if hit is not None:
            target = int(hit.group(1) or hit.group(2))
            if self._set_system_volume(target):
                return f"Громкость — {max(0, min(100, target))} процентов."
            return None
        if _LOUDER_RE.search(text) or _QUIETER_RE.search(text):
            current = self._system_volume()
            if current is None:
                return None
            step = 15 if _LOUDER_RE.search(text) else -15
            target = max(0, min(100, current + step))
            if self._set_system_volume(target):
                return f"Громкость — {target} процентов."
        return None

    def render_count(self, tales: Sequence[Tale]) -> str:
        """Ответ на «сколько у тебя сказок»: число словами и пара примеров."""
        count = len(tales)
        safe = [tale for tale in tales if not tale.is_scary] or list(tales)
        by_title = {tale.title: tale for tale in safe}
        examples: list[str] = []
        for title in ("Колобок", "Гуси-лебеди", "Царевна-лягушка"):
            if title in by_title and title not in examples:
                examples.append(title)
        if len(examples) < 2:
            positions = (0, len(safe) // 2, len(safe) - 1)
            for position in positions:
                title = safe[position].title
                if title not in examples:
                    examples.append(title)
        examples = examples[:3]
        names = ", ".join(f"«{title}»" for title in examples)
        tail = f" Например, {names}." if names else ""
        return _clean_speech(
            f"У меня {count} {self._tale_word(count)}.{tail}"
        )

    # -- ответ ---------------------------------------------------------------

    def answer(
        self, ctx: PluginContext, question: str
    ) -> str | dict[str, object] | None:
        """Текст сказки, запись, картинки или ``None`` — пусть отвечает модель.

        Со словарём ``{"text", "images"}`` сервер показывает иллюстрации по
        ходу рассказа (задача 0221); ``{"text", "silent": true}`` — включает
        готовую запись и просит не озвучивать подтверждение (задача 0223).
        Если ни картинок, ни записи нет, возвращается прежняя строка: старые
        вызовы не ломаются.
        """
        raw = str(question or "").strip()
        if not raw:
            return None
        text = _normalize(raw)
        if self._is_audio_stop(text):
            return self._audio_stop_answer(text)
        tales = self.load_tales(ctx)
        if not tales:
            return None
        if _AUDIO_RE.search(text):
            return self._answer_audio(ctx, text, tales)

        next_request = bool(_NEXT_RE.search(text))
        if _COUNT_RE.search(text):
            return self.render_count(tales)
        if _LIST_RE.search(text):
            return self.render_list(tales)
        number_hit = _NUMBER_RE.search(text)
        if number_hit:
            return self._answer_by_number(tales, number_hit)
        volume_answer = self._answer_volume(text)
        if volume_answer is not None:
            return volume_answer

        with self._lock:
            last = self._last_index
            if last is None or last >= len(tales):
                last = None
            night = bool(_NIGHT_RE.search(text)) or (
                next_request and self._last_night
            )
            allow_scary = bool(_SCARY_RE.search(text))
            pool = self._pool(tales, text, allow_scary=allow_scary)
            if not pool:
                return None

            filtered = bool(
                _MAGIC_RE.search(text)
                or _TODDLER_RE.search(text)
                or _SHORT_RE.search(text)
            )
            if next_request and last is not None:
                index = self._next_index(tales, pool, last)
            else:
                keywords = _keywords(text)
                targets = _target_words(text)
                ranked = self._ranked(pool, keywords)
                if targets:
                    if not ranked:
                        # «сказку про слона»: такой сказки нет — не подменяем
                        # её случайной, пусть модель ответит честно.
                        return None
                    index = tales.index(ranked[0])
                elif _is_story_request(text):
                    index = self._pick_random(
                        tales,
                        pool,
                        last,
                        prefer_short=not filtered,
                    )
                else:
                    return None

            self._last_index = index
            self._last_night = night
            rendered = self.render(tales[index], night=night)
            images = self.images_for(tales[index], rendered, ctx)
            if images:
                return {"text": rendered, "images": images}
            return rendered

    def render(self, tale: Tale, *, night: bool = False) -> str:
        """Собрать ответ: «Слушай сказку…», текст и мягкая концовка."""
        intro = "Слушай сказку"
        if night:
            intro += " на ночь"
        intro += f"… «{tale.title}»."
        ending = ENDING_NIGHT if night else ENDING_DAY
        text = f"{intro}\n\n{tale.body}\n\n{ending}"
        return _clean_speech(text)

    # -- записи сказок (afplay) ----------------------------------------------

    def _base_dir(self, ctx: PluginContext | None) -> Path:
        """Каталог плагина: контекст сервера или файл рядом с этим модулем."""
        if ctx is not None:
            plugin_dir = getattr(ctx, "plugin_dir", None)
            if plugin_dir:
                return Path(plugin_dir)
        return Path(__file__).resolve().parent

    def _audio_file(
        self, ctx: PluginContext | None, tale: Tale
    ) -> Path | None:
        """Локальная запись сказки ``audio/<slug>.*`` или ``None``.

        Файл больше 3 МБ не отдаём: это уже не «короткая запись», а повод
        пересмотреть комплект. Чтение идёт напрямую из каталога плагина, как
        и у иллюстраций: запись не должна зависеть от прав ``PluginContext``.
        """
        root = (
            self._audio_dir
            if self._audio_dir is not None
            else self._base_dir(ctx) / AUDIO_DIR
        )
        for suffix in AUDIO_SUFFIXES:
            path = root / f"{tale.slug}{suffix}"
            try:
                if not path.is_file():
                    continue
                if path.stat().st_size > AUDIO_MAX_BYTES:
                    self._log(
                        ctx,
                        "запись %s больше %d байт — пропускаю",
                        path.name,
                        AUDIO_MAX_BYTES,
                    )
                    return None
            except OSError:
                continue
            return path
        return None

    def _audio_tales(
        self, ctx: PluginContext | None, tales: Sequence[Tale]
    ) -> list[Tale]:
        return [tale for tale in tales if self._audio_file(ctx, tale) is not None]

    def _answer_audio(
        self, ctx: PluginContext, text: str, tales: Sequence[Tale]
    ) -> str | dict[str, object] | None:
        """Подобрать сказку с записью и включить её; без записи — честно."""
        keywords = [word for word in _keywords(text) if word not in _AUDIO_FILLERS]
        targets = _target_words(text)
        if targets or keywords:
            ranked = self._ranked(tales, targets or keywords)
            if not ranked:
                # «сказку про слона»: такой сказки нет — пусть отвечает модель.
                return None
            # Из подходящих предпочитаем ту, для которой есть запись: «про
            # кота» должно включить «Кот, петух и лиса», а не «Кот и лиса».
            with_audio = self._audio_tales(ctx, ranked)
            tale = with_audio[0] if with_audio else ranked[0]
        else:
            tale = self._default_audio_tale(ctx, tales)
            if tale is None:
                return self._no_recording()
        path = self._audio_file(ctx, tale)
        if path is None:
            return self._no_recording(tale)
        return self._play_audio(ctx, path, tale)

    def _default_audio_tale(
        self, ctx: PluginContext | None, tales: Sequence[Tale]
    ) -> Tale | None:
        """Без названия: последняя сказка, если у неё есть запись, иначе любая."""
        with_audio = self._audio_tales(ctx, tales)
        if not with_audio:
            return None
        with self._lock:
            last = self._last_index
        if last is not None and 0 <= last < len(tales):
            candidate = tales[last]
            if candidate in with_audio:
                return candidate
        return self._rng.choice(with_audio)

    def _play_audio(
        self, ctx: PluginContext | None, path: Path, tale: Tale
    ) -> str | dict[str, object]:
        """Запустить ``afplay`` списком argv, без shell, и подтвердить.

        Сервер зовёт ``answer`` для каждой колонки, поэтому две колонки могут
        одновременно попросить одну и ту же запись. Проверка и запуск идут
        под одним замком: второй вызов видит уже играющий файл и не плодит
        второй ``afplay`` (иначе «выключи» убил бы только один процесс).
        """
        # «Включила, но очень тихо»: если выход почти выключен, поднимаем до
        # слышимого уровня и честно сообщаем об этом. Пользователь не видит
        # громкость afplay, поэтому молчание выглядит как поломка.
        volume_note = ""
        current_volume = self._system_volume()
        if current_volume is not None and current_volume < QUIET_THRESHOLD:
            if self._set_system_volume(DEFAULT_LOUD_VOLUME):
                volume_note = (
                    f" Сделала громче — было {current_volume} процентов,"
                    f" стало {DEFAULT_LOUD_VOLUME}."
                )
        with self._audio_lock:
            current = self._audio_process
            if (
                current is not None
                and self._audio_slug == tale.slug
                and not self._audio_exited(current)
            ):
                return {"text": f"Включаю запись «{tale.title}».{volume_note}", "silent": True}
            self._stop_audio()
            try:
                process = subprocess.Popen(
                    [AFPLAY, str(path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            except FileNotFoundError:
                return "Не получилось включить запись: на этом Mac нет afplay."
            except OSError:
                return "Не получилось включить запись: afplay не запустился."
            self._audio_process = process
            self._audio_slug = tale.slug
            self._audio_stopped_at = None
        # afplay падает сразу, если файл не открылся (битый MP3, нет прав).
        # Даём мгновение, чтобы не обещать запись в пустоту.
        time.sleep(0.3)
        if self._audio_exited(process):
            with self._audio_lock:
                if self._audio_process is process:
                    self._audio_process = None
                    self._audio_slug = None
            return "Не получилось включить запись: файл не открылся."
        return {"text": f"Включаю запись «{tale.title}».{volume_note}", "silent": True}

    @staticmethod
    def _audio_exited(process: object) -> bool:
        poll = getattr(process, "poll", None)
        if not callable(poll):
            return False
        try:
            return poll() is not None
        except Exception:  # noqa: BLE001 - странный Popen не должен ломать ответ
            return False

    def _audio_active(self) -> bool:
        with self._audio_lock:
            process = self._audio_process
        if process is None:
            return False
        if self._audio_exited(process):
            with self._audio_lock:
                if self._audio_process is process:
                    self._audio_process = None
                    self._audio_slug = None
            return False
        return True

    def _recently_stopped(self) -> bool:
        """Запись остановили только что: вторая колонка должна ответить так же."""
        at = self._audio_stopped_at
        return at is not None and (time.monotonic() - at) <= 2.0

    def _is_audio_stop(self, text: str) -> bool:
        if _AUDIO_STOP_RE.match(text):
            return True
        if not _AUDIO_STOP_BARE_RE.match(text):
            return False
        return self._audio_active() or self._recently_stopped()

    def _audio_stop_answer(self, text: str) -> str | None:
        stopped = self._stop_audio()
        if stopped:
            return "Выключила запись."
        if self._recently_stopped():
            return "Выключила запись."
        if _AUDIO_STOP_RE.match(text):
            return "Запись сейчас не играет."
        return None

    def _stop_audio(self) -> bool:
        """Остановить afplay: SIGCONT, terminate, затем kill по таймауту.

        Метка остановки ставится под тем же замком, что и очистка процесса:
        вторая колонка, пришедшая одновременно, не должна увидеть «записи
        нет» и уйти к модели.
        """
        with self._audio_lock:
            process = self._audio_process
            if process is not None:
                self._audio_stopped_at = time.monotonic()
            self._audio_process = None
            self._audio_slug = None
        if process is None:
            return False
        try:
            if not self._audio_exited(process):
                try:
                    process.send_signal(signal.SIGCONT)
                except (ProcessLookupError, OSError):
                    pass
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
        except (ProcessLookupError, OSError):
            pass
        return True

    @staticmethod
    def _no_recording(tale: Tale | None = None) -> str:
        if tale is None:
            return "Записей сказок у меня нет, могу рассказать сама."
        return f"Записи сказки «{tale.title}» нет, могу рассказать сама."

    # -- иллюстрации ---------------------------------------------------------

    def images_for(
        self, tale: Tale, text: str, ctx: PluginContext | None = None
    ) -> list[dict[str, object]]:
        """Иллюстрации сказки из ``images/<slug>/`` по порядку имён файлов.

        Пути — относительно каталога плагина; ``after_chars`` привязан к
        границам абзацев готового текста (задача 0228: до 10 картинок на
        сказку, по одной на абзац). Нет каталога/файлов — пустой список.
        """
        directory, slug = self._image_directory(tale, ctx)
        if directory is None:
            return []
        try:
            names = sorted(
                path.name
                for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
            )
        except OSError as exc:  # pragma: no cover - странная ФС
            self._log(ctx, "иллюстрации %s не прочитаны: %s", tale.slug, exc)
            return []
        names = names[:MAX_IMAGES]
        if not names:
            return []
        positions = _image_positions(text, len(names))
        images: list[dict[str, object]] = []
        for order, (name, after) in enumerate(zip(names, positions), start=1):
            images.append(
                {
                    "path": f"{IMAGES_DIR}/{slug}/{name}",
                    "alt": f"Иллюстрация {order} к сказке «{tale.title}»",
                    "after_chars": after,
                }
            )
        return images

    def _image_directory(
        self, tale: Tale, ctx: PluginContext | None
    ) -> tuple[Path | None, str]:
        """Каталог картинок: сначала слаг сказки, затем слаг по названию."""
        base = Path(__file__).resolve().parent
        if ctx is not None:
            plugin_dir = getattr(ctx, "plugin_dir", None)
            if plugin_dir:
                base = Path(plugin_dir)
        root = base / IMAGES_DIR
        candidates: list[str] = [tale.slug]
        key = _title_key(tale.title)
        for alias, slug in _IMAGE_ALIASES:
            if alias in key and slug not in candidates:
                candidates.append(slug)
        for slug in candidates:
            directory = root / slug
            if directory.is_dir():
                return directory, slug
        return None, candidates[0]

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


def answer(
    ctx: PluginContext, question: str
) -> str | dict[str, object] | None:
    """Точка входа способности ``answer``: сказка, словарь с картинками или None."""
    return _PLUGIN.answer(ctx, question)


def register(ctx: PluginContext) -> FairyTalesPlugin:
    """Вернуть объект с ``answer(ctx, question)`` (реестр зовёт его)."""
    ctx.log.debug("fairy-tales: корпус готов, сеть не нужна")
    return _PLUGIN
