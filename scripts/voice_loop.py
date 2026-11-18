#!/usr/bin/env python3
"""Голосовой прототип EVA32: WAV → whisper.cpp → гибрид → say → звук.

Конвейер (ADR 0010)::

    WAV → ASR (whisper.cpp) → текст → eva.router (скилл) → текст → TTS
                                                        (Piper → say/espeak)

Что важно знать до запуска:

* **Модели не вызываются.** Прототип проверяет только детерминированную часть
  гибрида — скиллы из ``eva.skills``. Если вопрос не покрыт скиллом, скрипт
  честно печатает маршрут ``small``/``big`` и не идёт в сеть/ollama.
* **ASR опционален.** Нет собранного whisper.cpp — скрипт не падает, а печатает
  команды сборки и предлагает ``--text`` для отладки цепочки без ASR.
* **TTS — адаптер ``eva.tts``.** По умолчанию Piper (локальный нейро-TTS),
  если есть бинарь ``piper`` и модель ``.onnx`` (``PIPER_BIN``/``PIPER_MODEL``);
  иначе системный ``say`` (macOS, голос Milena) или ``espeak`` (Linux).
  ``--tts piper|system|null`` перекрывает автовыбор. Синтез идёт в WAV
  (16 кГц, моно, 16 бит для say), затем файл проигрывается ``afplay``. Так
  можно измерить и первый фрагмент синтеза, и реальный старт звука.
* **Замеры.** ASR, маршрут, синтез TTS, первый фрагмент TTS, старт
  воспроизведения и итог до первого звука — в миллисекундах.
* **Потоковая речь (по умолчанию).** Ответ режется на предложения; первое
  синтезируется и сразу играет, остальные синтезируются параллельно, пока
  звучит предыдущее. Задержка до первого звука считается по первому
  предложению, а не по всему ответу. ``--no-stream`` возвращает старое
  поведение (один WAV на весь ответ) — для сравнения.

Примеры::

    python3 scripts/voice_loop.py --text "Сколько стоит тариф Старт"
    python3 scripts/voice_loop.py --text "Сколько стоит тариф Старт" --no-play
    python3 scripts/voice_loop.py --text "Сколько стоит тариф Старт" --no-stream
    python3 scripts/voice_loop.py --text "..." --tts system
    python3 scripts/voice_loop.py --text "..." --tts piper --piper-model voice.onnx
    python3 scripts/voice_loop.py --text "..." --out /tmp/reply.wav
    python3 scripts/voice_loop.py --wav request.wav \
        --whisper-bin ./whisper.cpp/build/bin/whisper-cli \
        --model ./whisper.cpp/models/ggml-small.bin

Только стандартная библиотека. Сеть не используется.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

# Скрипт запускают из корня репозитория и из любого каталога: eva лежит рядом
# с scripts/, а sys.path[0] при запуске файла — сам scripts/.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eva.config import load as load_config  # noqa: E402
from eva.router import answer as router_answer  # noqa: E402
from eva.router import decide as router_decide  # noqa: E402
from eva.tts import DEFAULT_SAY_VOICE  # noqa: E402
from eva.tts import ENGINE_CHOICES  # noqa: E402
from eva.tts import NullTTS  # noqa: E402
from eva.tts import SynthesisResult  # noqa: E402
from eva.tts import TTSAdapter  # noqa: E402
from eva.tts import TTSError as TtsError  # noqa: E402
from eva.tts import TTSUnavailable as TtsUnavailable  # noqa: E402
from eva.tts import build_say_command as _build_say_command  # noqa: E402
from eva.tts import describe_engine  # noqa: E402
from eva.tts import find_say_binary  # noqa: E402
from eva.tts import select_engine  # noqa: E402
from eva.types import Message, Request  # noqa: E402

try:  # скиллы — отдельный пакет; без них прототип остаётся работоспособным
    from eva.skills import try_skills as _try_skills  # noqa: E402
except Exception:  # pragma: no cover — зависит от состояния соседних задач
    _try_skills = None  # type: ignore[assignment]


DEFAULT_VOICE = os.environ.get("EVA_VOICE", DEFAULT_SAY_VOICE).strip() or DEFAULT_SAY_VOICE
_ENGINE_ENV = (os.environ.get("EVA_TTS") or "auto").strip().lower()
#: Движок TTS по умолчанию: EVA_TTS, если значение корректно, иначе auto
#: (Piper → system → null).
DEFAULT_TTS_ENGINE = _ENGINE_ENV if _ENGINE_ENV in ENGINE_CHOICES else "auto"
DEFAULT_LANG = os.environ.get("EVA_WHISPER_LANG", "ru")
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_TTS_TIMEOUT = 120.0
DEFAULT_ASR_TIMEOUT = 180.0
DEFAULT_PLAY_TIMEOUT = 180.0

#: Сколько предложений синтезировать вперёд, пока звучит текущее. Два потока
#: закрывают паузу, если первое предложение окажется совсем коротким.
STREAM_WORKERS = 2

#: Максимальная длина чанка, который уходит в один синтез. Длинное предложение
#: без точек режется по пробелам/запятым, иначе потоковая речь теряет смысл.
MAX_CHUNK_CHARS = 220

#: Имена бинарника whisper.cpp: новое (whisper-cli) и старое (main в build/bin).
WHISPER_BINARY_NAMES = ("whisper-cli", "whisper-cpp")

#: Модели в порядке предпочтения для русского на Apple Silicon.
WHISPER_MODEL_NAMES = (
    "ggml-small.bin",
    "ggml-base.bin",
    "ggml-medium.bin",
    "ggml-small-q5_1.bin",
    "ggml-base-q5_1.bin",
)

#: Типовые места сборки whisper.cpp относительно репозитория и домашнего каталога.
WHISPER_BINARY_CANDIDATES = (
    "whisper.cpp/build/bin/whisper-cli",
    "third_party/whisper.cpp/build/bin/whisper-cli",
    "tools/whisper.cpp/build/bin/whisper-cli",
    "vendor/whisper.cpp/build/bin/whisper-cli",
    "build/bin/whisper-cli",
)

#: Типовые места моделей.
WHISPER_MODEL_DIRS = (
    "models",
    "whisper.cpp/models",
    "third_party/whisper.cpp/models",
    "tools/whisper.cpp/models",
    "vendor/whisper.cpp/models",
)


# --------------------------------------------------------------------------- #
# ошибки и результат
# --------------------------------------------------------------------------- #
class VoiceError(RuntimeError):
    """Ошибка конвейера, которую можно показать человеку без traceback."""


class AsrUnavailable(VoiceError):
    """ASR не настроен: нет whisper.cpp или модели."""


class AsrError(VoiceError):
    """whisper.cpp запустился, но распознать не смог."""


@dataclass
class RouteResult:
    """Ответ гибрида, подготовленный к озвучке."""

    question: str
    answer_text: str
    spoken_text: str
    route_kind: str
    route_reason: str
    skill: str | None
    latency_ms: float
    answered: bool


@dataclass
class PlayResult:
    """Результат воспроизведения WAV через afplay."""

    start_ms: float
    total_ms: float


@dataclass
class ChunkTiming:
    """Замеры одного чанка (предложения) в потоковой озвучке."""

    index: int
    text: str
    chars: int
    synth_ms: float | None = None
    first_fragment_ms: float | None = None
    synth_done_ms: float | None = None
    wait_ms: float | None = None
    play_start_ms: float | None = None
    play_total_ms: float | None = None


@dataclass
class StreamResult:
    """Итог потоковой озвучки: чанки, первый звук и суммарный синтез.

    ``synth_total_ms`` — сумма синтеза всех чанков (работа, а не задержка):
    чанк N+1 синтезируется, пока звучит чанк N. ``first_sound_ms`` — время от
    начала потоковой озвучки до запуска afplay с первым предложением.
    """

    chunks: list[ChunkTiming]
    parts: list[Path]
    first_sound_ms: float | None
    first_play_offset_ms: float | None
    synth_total_ms: float
    play_span_ms: float | None
    wall_ms: float


@dataclass
class Timings:
    """Замеры этапов в миллисекундах."""

    asr_ms: float | None = None
    route_ms: float = 0.0
    tts_synth_ms: float | None = None
    tts_first_fragment_ms: float | None = None
    play_start_ms: float | None = None
    play_total_ms: float | None = None
    stream: bool = False
    chunks: int = 0
    tts_total_synth_ms: float | None = None
    chunk_stats: list[ChunkTiming] = field(default_factory=list)

    @property
    def first_sound_ms(self) -> float | None:
        """Время от старта конвейера до первого звука (файл уже синтезирован).

        В непоточном режиме ``say`` сначала синтезирует весь ответ в WAV,
        поэтому в путь до первого звука входит полный синтез. В потоковом
        режиме ``tts_synth_ms`` — синтез только первого предложения, а
        остальные синтезируются, пока звучит речь.
        """
        if self.play_start_ms is None or self.tts_synth_ms is None:
            return None
        return (self.asr_ms or 0.0) + self.route_ms + self.tts_synth_ms + self.play_start_ms

    @property
    def first_sound_estimate_ms(self) -> float | None:
        """Оценка до первого звука, когда воспроизведение отключено (``--no-play``).

        В этой схеме afplay запускается, когда WAV уже синтезирован целиком,
        поэтому оценка = ASR + маршрут + синтез того, что играем: всего ответа
        в непоточном режиме и первого предложения в потоковом. Разница между
        режимами — ровно выигрыш от потоковой речи.
        """
        synth = self.tts_synth_ms
        if synth is None:
            synth = self.tts_first_fragment_ms
        if synth is None:
            return None
        return (self.asr_ms or 0.0) + self.route_ms + synth

    @property
    def live_estimate_ms(self) -> float | None:
        """Ориентир для live say: ASR + маршрут + первый фрагмент синтеза."""
        fragment = self.tts_first_fragment_ms
        if fragment is None:
            return None
        return (self.asr_ms or 0.0) + self.route_ms + fragment

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["first_sound_ms"] = self.first_sound_ms
        data["first_sound_estimate_ms"] = self.first_sound_estimate_ms
        data["live_estimate_ms"] = self.live_estimate_ms
        return data


# --------------------------------------------------------------------------- #
# аргументы
# --------------------------------------------------------------------------- #
def positive_float(raw: str) -> float:
    """argparse: положительное число секунд."""
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"ожидалось число, получено {raw!r}") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("значение должно быть больше нуля")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="voice_loop.py",
        description=(
            "Прототип голосового конвейера EVA32: WAV → whisper.cpp → eva → "
            "TTS (Piper, если установлен, иначе say/espeak). "
            "Если whisper.cpp нет, используйте --text."
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--wav",
        type=Path,
        metavar="FILE",
        help="входной WAV (16 кГц, моно, 16-bit PCM) для распознавания",
    )
    source.add_argument(
        "--text",
        metavar="TEXT",
        help="готовый текст вопроса: ASR пропускается, отлаживается остальная цепочка",
    )
    parser.add_argument(
        "--out",
        type=Path,
        metavar="FILE",
        help="сохранить синтезированный WAV по этому пути (по умолчанию временный файл)",
    )
    parser.add_argument(
        "--voice",
        default=DEFAULT_VOICE,
        help=f"голос say/espeak (по умолчанию {DEFAULT_VOICE})",
    )
    parser.add_argument(
        "--lang",
        default=DEFAULT_LANG,
        help=f"язык whisper.cpp (по умолчанию {DEFAULT_LANG})",
    )
    parser.add_argument(
        "--whisper-bin",
        default=None,
        metavar="FILE",
        help="путь к whisper-cli/whisper-cpp (иначе PATH и типовые каталоги)",
    )
    parser.add_argument(
        "--model",
        dest="whisper_model",
        default=None,
        metavar="FILE",
        help="путь к модели ggml-*.bin (иначе WHISPER_MODEL и типовые каталоги)",
    )
    parser.add_argument(
        "--tts",
        dest="tts_engine",
        choices=ENGINE_CHOICES,
        default=DEFAULT_TTS_ENGINE,
        help=(
            "TTS-движок: auto — Piper, если доступен, иначе say/espeak "
            f"(по умолчанию {DEFAULT_TTS_ENGINE}); piper; system; null"
        ),
    )
    parser.add_argument(
        "--piper-bin",
        default=None,
        metavar="FILE",
        help="путь к piper (иначе PIPER_BIN, PATH и типовые каталоги)",
    )
    parser.add_argument(
        "--piper-model",
        default=None,
        metavar="FILE",
        help="модель Piper .onnx (иначе PIPER_MODEL и типовые каталоги)",
    )
    parser.add_argument(
        "--say-bin",
        default=None,
        metavar="FILE",
        help="путь к say для системного TTS (по умолчанию из PATH)",
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=None,
        help="скорость речи say (-r) / espeak (-s) / piper (length_scale), слов в минуту",
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="синтезировать WAV, но не воспроизводить (для тестов и замеров)",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help=(
            "старое поведение: синтезировать весь ответ до воспроизведения; "
            "по умолчанию речь идёт по предложениям (первое — сразу, остальные — параллельно)"
        ),
    )
    parser.add_argument(
        "--asr-timeout",
        type=positive_float,
        default=DEFAULT_ASR_TIMEOUT,
        help=f"таймаут whisper.cpp, сек (по умолчанию {DEFAULT_ASR_TIMEOUT:g})",
    )
    parser.add_argument(
        "--tts-timeout",
        type=positive_float,
        default=DEFAULT_TTS_TIMEOUT,
        help=f"таймаут синтеза TTS, сек (по умолчанию {DEFAULT_TTS_TIMEOUT:g})",
    )
    parser.add_argument(
        "--play-timeout",
        type=positive_float,
        default=DEFAULT_PLAY_TIMEOUT,
        help=f"таймаут afplay, сек (по умолчанию {DEFAULT_PLAY_TIMEOUT:g})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="напечатать результат и тайминги JSON-объектом (для машинной проверки)",
    )
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# подготовка текста к озвучке
# --------------------------------------------------------------------------- #
_RUBLE_RE = re.compile(r"(\d[\d\s\u00a0]*)\s*₽(?:\s*/\s*мес)?")
_CITATION_RE = re.compile(r"\[\s*\d+(?:\s*[,;]\s*\d+)*\s*\]")
_TIMESTAMP_RE = re.compile(
    r"\[\s*\d{1,2}:\d{2}(?::\d{2})?[.,]\d{1,3}\s*-->\s*[^\]]+\]\s*"
)
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_MD_MARKS_RE = re.compile(r"[*_`#>]+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")


def _plural(number: int, one: str, few: str, many: str) -> str:
    """Русская форма слова для числа (1 рубль, 2 рубля, 5 рублей)."""
    value = abs(number) % 100
    if 11 <= value <= 14:
        return many
    value %= 10
    if value == 1:
        return one
    if 2 <= value <= 4:
        return few
    return many


def _replace_rubles(match: re.Match[str]) -> str:
    digits = re.sub(r"\D", "", match.group(1))
    word = _plural(int(digits or 0), "рубль", "рубля", "рублей")
    suffix = " в месяц" if "мес" in match.group(0).lower() else ""
    return f"{match.group(1).strip()} {word}{suffix}"


def to_speech(text: str) -> str:
    """Привести текст ответа к виду, удобному для ``say``.

    Убираются ссылки на источники (``[1]``), markdown-разметка; ``₽`` и
    ``₽/мес`` заменяются на «рубль/рубля/рублей» и «в месяц». Остальная
    пунктуация остаётся — say сам расставляет паузы.
    """
    speech = str(text or "")
    if not speech.strip():
        return ""

    speech = _MD_LINK_RE.sub(r"\1", speech)
    speech = _CITATION_RE.sub("", speech)
    speech = _RUBLE_RE.sub(_replace_rubles, speech)
    speech = speech.replace("₽/мес", " рублей в месяц")
    speech = speech.replace("₽", " рублей")
    speech = re.sub(r"\s*/\s*мес\b", " в месяц", speech)
    speech = _MD_MARKS_RE.sub("", speech)
    speech = speech.replace("\u00a0", " ")
    speech = re.sub(r"\s+", " ", speech)
    speech = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", speech)
    return speech.strip()


# --------------------------------------------------------------------------- #
# разбиение ответа на предложения
# --------------------------------------------------------------------------- #
#: Символы, которые могут заканчивать предложение. Точка — с оговорками:
#: она же стоит в числах (1.5), сокращениях (т.е.) и инициалах (А. С. Пушкин).
_SENTENCE_END_CHARS = ".!?…"

#: Парные кавычки: внутри них точка/многоточие не заканчивают предложение.
_QUOTE_PAIRS = {"«": "»", "„": "“", "“": "”"}
_OPEN_QUOTES = "".join(_QUOTE_PAIRS)
_CLOSE_QUOTES = "".join(_QUOTE_PAIRS.values())

#: Сокращения, точка которых не заканчивает предложение (в нижнем регистре).
ABBREVIATIONS = frozenset(
    """
    т.е. т.д. т.п. т.к. т.н. др. г. гг. в. вв. руб. коп. млн. млрд. тыс.
    стр. рис. табл. см. ср. напр. им. проф. акад. доц. п. пп. ул. д. кв.
    обл. гор. ред. изд. прим. e.g. i.e. etc.
    """.split()
)

#: После этих символов точку/многоточие нельзя считать концом предложения.
_NO_SENTENCE_START = ",;:)]—–"


def _last_token(text: str, end: int) -> str:
    """Последнее слово перед позицией ``end`` (вместе с точками сокращений)."""
    head = text[:end]
    cut = max(head.rfind(" "), head.rfind("\t"), head.rfind("\n"))
    return head[cut + 1 :]


def _is_abbreviation(text: str, end: int) -> bool:
    """Точка на позиции ``end - 1`` — часть сокращения, а не конец фразы.

    Учитываются «т.е.», «др.», «т.д.», «г.» и т.п., а также инициалы
    («А. С. Пушкин»): одиночная заглавная буква с точкой перед словом
    с большой буквы.
    """
    token = _last_token(text, end)
    if not token:
        return False
    if token.lower() in ABBREVIATIONS:
        return True
    if re.fullmatch(r"[А-ЯЁA-Z]\.", token):
        following = text[end:].lstrip()
        return bool(following) and following[0].isupper()
    return False


def _is_sentence_end(text: str, start: int, end: int, after: int) -> bool:
    """Проверить, что терминаторы ``text[start:end]`` кончают предложение.

    ``after`` — позиция сразу за терминаторами и закрывающими кавычками:
    именно оттуда смотрим, начинается ли новое предложение.
    """
    run = text[start:end]
    if run == ".":
        before = text[start - 1] if start > 0 else ""
        next_char = text[after] if after < len(text) else ""
        if before.isdigit() and next_char.isdigit():
            return False  # 1.5, 1.2.3 — точка внутри числа
        if _is_abbreviation(text, end):
            return False  # «т.е.», «др.», «А. С.»
    following = text[after:].lstrip()
    if not following:
        return True
    if following[0].islower():
        return False  # продолжение фразы (в том числе после закрывающей кавычки)
    if following[0] in _NO_SENTENCE_START:
        return False
    return True


def _soft_break(text: str, max_chars: int) -> int:
    """Позиция мягкого разрыва длинного предложения (не больше ``max_chars``).

    Режем по пробелу вне кавычек; не рвём «2 490 ₽» (пробел внутри числа или
    перед валютой) и не оставляем «₽/мес» в начале следующего чанка.
    Если пробела вне кавычек нет, берём пробел внутри цитаты, чтобы чанк
    не оказался гигантским; совсем без пробелов — жёсткий разрез.
    """

    def find(ignore_quotes: bool) -> int:
        limit = min(len(text), max_chars)
        depth = 0
        best = -1
        for index in range(limit):
            char = text[index]
            if char in _OPEN_QUOTES:
                depth += 1
                continue
            if char in _CLOSE_QUOTES:
                depth = max(0, depth - 1)
                continue
            if char != " " or (depth > 0 and not ignore_quotes):
                continue
            before = text[index - 1] if index > 0 else ""
            after = text[index + 1] if index + 1 < len(text) else ""
            if before.isdigit() and (after.isdigit() or after == "₽"):
                continue  # пробел внутри числа или между числом и «₽»
            tail = text[index + 1 :].lstrip()
            if tail[:1] in ("₽", "%"):
                continue
            best = index + 1
        return best

    position = find(ignore_quotes=False)
    if position > 0:
        return position
    position = find(ignore_quotes=True)
    if position > 0:
        return position
    space = text.rfind(" ", 0, min(len(text), max_chars))
    if space > 0:
        return space + 1
    return 0


def _wrap_long(chunk: str, *, max_chars: int) -> list[str]:
    """Порезать слишком длинное «предложение» на удобные для синтеза куски."""
    if max_chars <= 0 or len(chunk) <= max_chars:
        return [chunk]
    parts: list[str] = []
    rest = chunk
    while len(rest) > max_chars:
        cut = _soft_break(rest, max_chars)
        if cut <= 0:
            break
        head = rest[:cut].strip()
        if head:
            parts.append(head)
        rest = rest[cut:].strip()
    if rest:
        parts.append(rest)
    return parts


def _split_paragraph(paragraph: str, *, max_chars: int) -> list[str]:
    text = paragraph.strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    depth = 0
    index = 0
    length = len(text)

    while index < length:
        char = text[index]
        if char in _OPEN_QUOTES:
            depth += 1
            index += 1
            continue
        if char in _CLOSE_QUOTES:
            depth = max(0, depth - 1)
            index += 1
            continue
        if char not in _SENTENCE_END_CHARS:
            index += 1
            continue

        end = index
        while end < length and text[end] in _SENTENCE_END_CHARS:
            end += 1

        stop = end
        while stop < length and text[stop] in _CLOSE_QUOTES + ")":
            stop += 1

        # Внутри «ёлочек» режем, только если терминатор закрывает цитату:
        # точка посреди цитаты («Всё хорошо. Завтра продолжим.») её не рвёт.
        if depth > 0 and stop == end:
            index = end
            continue
        if not _is_sentence_end(text, index, end, stop):
            index = end
            continue

        chunk = text[start:stop].strip()
        if chunk:
            chunks.append(chunk)
        for position in range(end, stop):
            if text[position] in _CLOSE_QUOTES:
                depth = max(0, depth - 1)
        start = stop
        index = stop

    tail = text[start:].strip()
    if tail:
        chunks.append(tail)
    return chunks


def split_sentences(text: str, *, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Разбить ответ на предложения для потокового синтеза.

    Правила:

    * конец предложения — ``. ! ? …`` (в том числе ``...``), после которого
      не идёт строчная буква, запятая, закрывающая скобка или тире;
    * точка в числе (``1.5``), в сокращении (``т.е.``, ``др.``, ``г.``) и в
      инициалах (``А. С.``) предложение не заканчивает;
    * внутри «ёлочек» (и „лапок“) не режем, пока впереди есть закрывающая
      кавычка — цитата остаётся целиком;
    * «₽» и «₽/мес» остаются при своём числе: пробел внутри числа и перед
      валютой не считается разрывом;
    * предложение длиннее ``max_chars`` дорезается по пробелам (вне кавычек),
      чтобы в синтез не уходил кусок на несколько абзацев.

    Возвращает список непустых чанков в исходном порядке. Если пунктуации
    нет, вернётся один чанк (или несколько — после дорезки по ``max_chars``).
    """
    raw = str(text or "")
    if not raw.strip():
        return []

    chunks: list[str] = []
    for paragraph in raw.splitlines():
        if not paragraph.strip():
            continue
        for chunk in _split_paragraph(paragraph, max_chars=max_chars):
            chunks.extend(_wrap_long(chunk, max_chars=max_chars))
    return [chunk for chunk in chunks if chunk]


# --------------------------------------------------------------------------- #
# маршрут через eva.router (только скиллы, без моделей)
# --------------------------------------------------------------------------- #
def _skill_name(skill_result: Any) -> str | None:
    if isinstance(skill_result, dict):
        name = skill_result.get("skill")
    else:
        name = getattr(skill_result, "skill", None)
    name = str(name or "").strip()
    return name or None


def ask_hybrid(
    question: str,
    cfg: Any | None = None,
    *,
    skills_fn: Callable[..., Any] | None = None,
    answer_fn: Callable[..., Any] | None = None,
    decide_fn: Callable[..., Any] | None = None,
) -> RouteResult:
    """Ответить на вопрос детерминированной частью гибрида.

    Скиллы вызываются первыми. Если ни один не взял вопрос, маршрут только
    *вычисляется* (``eva.router.decide``), но модель не вызывается: прототип
    не запускает ollama и не ходит в сеть. Возвращается ``answered=False``.
    """
    text = str(question or "").strip()
    if not text:
        raise VoiceError("пустой текст вопроса")

    cfg = cfg if cfg is not None else load_config()
    skills_fn = _try_skills if skills_fn is None else skills_fn
    answer_fn = router_answer if answer_fn is None else answer_fn
    decide_fn = router_decide if decide_fn is None else decide_fn

    request = Request(
        messages=[Message(role="user", content=text)],
        meta={"channel": "voice"},
    )
    started = time.perf_counter()

    matched: Any | None = None
    if skills_fn is not None:
        try:
            matched = skills_fn(text, {})
        except Exception:
            matched = None

    if matched is None:
        try:
            route = decide_fn(request, cfg)
        except Exception as exc:  # noqa: BLE001 — маршрут обязан быть, но не падать
            raise VoiceError(f"маршрутизатор не смог выбрать маршрут: {exc}") from exc
        return RouteResult(
            question=text,
            answer_text="",
            spoken_text="",
            route_kind=str(getattr(route, "kind", "unknown")),
            route_reason=str(getattr(route, "reason", "") or ""),
            skill=None,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            answered=False,
        )

    # Скилл уже совпал, поэтому answer не дойдёт до моделей. deps передаём
    # явно, чтобы прототип не зависел от глобального состояния router.
    try:
        result = answer_fn(request, cfg, deps={"skills": skills_fn})
    except Exception as exc:  # noqa: BLE001
        raise VoiceError(f"eva.router не смог ответить: {exc}") from exc

    latency_ms = (time.perf_counter() - started) * 1000.0
    route = getattr(result, "route", None)
    route_kind = str(getattr(route, "kind", "") or "")
    skill = _skill_name(result) or _skill_name(matched)
    answer_text = str(getattr(result, "text", "") or "")

    if route_kind != "skill" or not answer_text.strip():
        # Защита от неожиданного поведения router: без ответа скилла не
        # озвучиваем и не притворяемся, что модель отработала.
        return RouteResult(
            question=text,
            answer_text=answer_text,
            spoken_text="",
            route_kind=route_kind or "unknown",
            route_reason=str(getattr(route, "reason", "") or ""),
            skill=skill,
            latency_ms=latency_ms,
            answered=False,
        )

    spoken = to_speech(answer_text)
    if not spoken:
        raise VoiceError("скилл вернул пустой текст после подготовки к озвучке")

    return RouteResult(
        question=text,
        answer_text=answer_text,
        spoken_text=spoken,
        route_kind=route_kind,
        route_reason=str(getattr(route, "reason", "") or ""),
        skill=skill,
        latency_ms=latency_ms,
        answered=True,
    )


# --------------------------------------------------------------------------- #
# WAV и whisper.cpp
# --------------------------------------------------------------------------- #
def wav_info(path: Path) -> dict[str, Any]:
    """Проверить, что на входе WAV/PCM, и вернуть его параметры."""
    file = Path(path).expanduser()
    if not file.exists():
        raise AsrUnavailable(f"WAV не найден: {file}")
    if not file.is_file():
        raise AsrUnavailable(f"WAV: ожидался файл, получено {file}")
    try:
        with wave.open(str(file), "rb") as audio:
            channels = audio.getnchannels()
            sample_width = audio.getsampwidth()
            rate = audio.getframerate()
            frames = audio.getnframes()
            comptype = audio.getcomptype()
    except (wave.Error, EOFError, OSError) as exc:
        raise AsrUnavailable(
            f"не удалось прочитать WAV {file}: {exc}. Нужен несжатый PCM WAV "
            "(16 кГц, моно, 16 бит)."
        ) from exc

    if comptype != "NONE" or sample_width != 2:
        raise AsrUnavailable(
            f"WAV {file}: {sample_width * 8}-bit {comptype}. whisper.cpp ждёт "
            "16-bit PCM. Перекодируйте, например: afconvert -f WAVE -d LEI16@16000 in out.wav"
        )

    duration = frames / float(rate) if rate else 0.0
    return {
        "path": str(file),
        "channels": channels,
        "sample_width": sample_width,
        "rate": rate,
        "frames": frames,
        "duration_s": round(duration, 3),
    }


def find_whisper_binary(explicit: str | Path | None = None) -> Path | None:
    """Найти whisper-cli/whisper-cpp: аргумент → WHISPER_BIN → PATH → типовые места."""
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise AsrUnavailable(
                f"whisper.cpp: {path} не найден или не исполняемый. "
                "Укажите --whisper-bin или соберите whisper.cpp (см. docs/voice-prototype.md)."
            )
        return path

    env_binary = os.environ.get("WHISPER_BIN", "").strip()
    if env_binary:
        path = Path(env_binary).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return path

    for name in WHISPER_BINARY_NAMES:
        found = shutil.which(name)
        if found:
            return Path(found)

    for relative in WHISPER_BINARY_CANDIDATES:
        path = ROOT / relative
        if path.is_file() and os.access(path, os.X_OK):
            return path
        home_path = Path.home() / relative
        if home_path.is_file() and os.access(home_path, os.X_OK):
            return home_path

    return None


def find_whisper_model(
    explicit: str | Path | None = None,
    binary: Path | None = None,
) -> Path | None:
    """Найти модель ggml-*.bin рядом с whisper.cpp или в типовых каталогах."""
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise AsrUnavailable(
                f"модель whisper.cpp не найдена: {path}. "
                "Скачайте: bash models/download-ggml-model.sh small"
            )
        return path

    env_model = os.environ.get("WHISPER_MODEL", "").strip()
    if env_model:
        path = Path(env_model).expanduser()
        if path.is_file():
            return path

    roots: list[Path] = []
    for relative in WHISPER_MODEL_DIRS:
        roots.append(ROOT / relative)
        roots.append(Path.home() / relative)
    # Бинарник обычно лежит в whisper.cpp/build/bin/ — модели на два уровня выше.
    if binary is not None:
        roots.extend(
            [
                binary.parent / "models",
                binary.parents[1] / "models" if len(binary.parents) > 1 else Path(),
                binary.parents[2] / "models" if len(binary.parents) > 2 else Path(),
            ]
        )

    seen: set[Path] = set()
    for root in roots:
        if not root or root in seen:
            continue
        seen.add(root)
        for name in WHISPER_MODEL_NAMES:
            path = root / name
            if path.is_file():
                return path

    for directory in (
        Path.home() / ".cache" / "whisper.cpp",
        Path.home() / ".local" / "share" / "whisper.cpp",
    ):
        for name in WHISPER_MODEL_NAMES:
            path = directory / name
            if path.is_file():
                return path

    return None


def asr_help(
    binary: Path | None,
    model: Path | None,
    explicit_model: str | None = None,
) -> str:
    """Человеческое сообщение: чего не хватает и как это исправить."""
    lines: list[str] = []
    if binary is None:
        lines.append("ASR недоступен: whisper.cpp не найден ни в PATH, ни в типовых каталогах.")
    else:
        lines.append(f"ASR недоступен: whisper.cpp найден ({binary}), но модель не найдена.")
        lines.append(
            "Скачайте модель: bash models/download-ggml-model.sh small "
            "(или укажите --model / WHISPER_MODEL)."
        )
        if explicit_model:
            lines.append(f"Переданный --model: {explicit_model}")

    lines.extend(
        [
            "",
            "Как собрать whisper.cpp (macOS, Apple Silicon, Metal):",
            "  git clone https://github.com/ggml-org/whisper.cpp",
            "  cd whisper.cpp",
            "  cmake -B build -DGGML_METAL=ON",
            "  cmake --build build -j --config Release",
            "  bash models/download-ggml-model.sh small",
            "",
            "Готовый бинарник: whisper.cpp/build/bin/whisper-cli",
            "Модель:          whisper.cpp/models/ggml-small.bin",
            "",
            "Затем:",
            "  python3 scripts/voice_loop.py --wav request.wav \\",
            "      --whisper-bin ./whisper.cpp/build/bin/whisper-cli \\",
            "      --model ./whisper.cpp/models/ggml-small.bin",
            "",
            "Без ASR (отладка всей остальной цепочки):",
            '  python3 scripts/voice_loop.py --text "Сколько стоит тариф Старт"',
        ]
    )
    return "\n".join(lines)


def parse_whisper_stdout(raw: str) -> str:
    """Достать текст из stdout whisper.cpp, если .txt рядом не появился."""
    lines: list[str] = []
    for line in str(raw or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(("whisper_", "ggml_", "main:", "system_info", "error:")):
            continue
        line = _TIMESTAMP_RE.sub("", line).strip()
        if line:
            lines.append(line)
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def transcribe_wav(
    wav_path: Path,
    binary: Path,
    model: Path,
    *,
    lang: str = DEFAULT_LANG,
    timeout: float = DEFAULT_ASR_TIMEOUT,
) -> tuple[str, float]:
    """Распознать WAV через whisper.cpp. Возвращает (текст, мс)."""
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="eva-asr-") as tmp:
        base = Path(tmp) / "transcript"
        command = [
            str(binary),
            "-m",
            str(model),
            "-f",
            str(Path(wav_path).expanduser()),
            "-l",
            lang,
            "-nt",  # без таймкодов
            "-np",  # без служебных распечаток
            "-otxt",  # писать .txt
            "-of",
            str(base),
        ]
        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise AsrError(
                f"whisper.cpp не уложился в {timeout:g} с. "
                "Уменьшите WAV или увеличьте --asr-timeout."
            ) from exc
        except OSError as exc:
            raise AsrError(f"whisper.cpp не запустился: {exc}") from exc

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip().splitlines()
            tail = " | ".join(detail[-3:]) if detail else "без вывода"
            raise AsrError(f"whisper.cpp завершился с кодом {proc.returncode}: {tail}")

        text_file = Path(str(base) + ".txt")
        if text_file.is_file():
            text = text_file.read_text(encoding="utf-8", errors="replace")
        else:
            text = parse_whisper_stdout(proc.stdout or "")

        text = _TIMESTAMP_RE.sub(" ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            raise AsrError(
                "whisper.cpp вернул пустой текст. Проверьте, что в WAV есть речь, "
                "модель и язык (--lang ru)."
            )
        return text, elapsed_ms


# --------------------------------------------------------------------------- #
# TTS-адаптер (eva.tts: Piper → say/espeak → null) и afplay
# --------------------------------------------------------------------------- #
def find_say(explicit: str | Path | None = None) -> Path:
    """Найти ``say`` для системного TTS (совместимость со старым API)."""
    say = find_say_binary(explicit)
    if say is None:
        raise TtsUnavailable(
            "TTS недоступен: say не найден. На macOS проверьте голос "
            "`say -v '?' | grep Milena`; на Linux используйте espeak "
            "(--tts system)."
        )
    return say


def build_say_command(
    say_bin: str | Path,
    voice: str,
    out_path: str | Path,
    text: str,
    *,
    rate: int | None = None,
) -> list[str]:
    """Команда say, которая пишет WAV 16 кГц/моно/16 бит (см. eva.tts)."""
    return _build_say_command(say_bin, voice, out_path, text, rate=rate)


def synthesize(
    text: str,
    out_path: Path,
    *,
    voice: str = DEFAULT_VOICE,
    say_bin: str | Path | None = None,
    rate: int | None = None,
    timeout: float = DEFAULT_TTS_TIMEOUT,
    engine: TTSAdapter | None = None,
    tts_engine: str | None = None,
    piper_bin: str | Path | None = None,
    piper_model: str | Path | None = None,
) -> SynthesisResult:
    """Синтезировать текст в WAV через выбранный TTS-движок и замерить его.

    ``engine`` — уже выбранный адаптер (в потоковой озвучке он один на все
    предложения). Без него движок выбирается заново: Piper, если есть бинарь
    и модель, иначе системный say/espeak (``tts_engine``/``EVA_TTS``
    перекрывает автовыбор).

    ``first_fragment_ms`` — время до появления первых байт аудио в файле:
    это нижняя оценка того, когда заговорил бы живой движок.
    """
    chosen = engine
    if chosen is None:
        chosen = select_engine(
            tts_engine,
            piper_bin=piper_bin,
            piper_model=piper_model,
            say_bin=say_bin,
            voice=voice,
        )
    if isinstance(chosen, NullTTS) or chosen.name == "null":
        raise TtsUnavailable(
            "TTS недоступен: не найден ни Piper (PIPER_BIN/PIPER_MODEL), ни "
            "системный say/espeak. Установите Piper или включите --tts system."
        )

    target = Path(out_path).expanduser()
    if target.parent and not target.parent.exists():
        raise TtsError(f"каталог для --out не существует: {target.parent}")
    return chosen.synthesize_timed(
        text, target, rate=rate, voice=voice, timeout=timeout
    )


def play_audio(path: Path, *, timeout: float = DEFAULT_PLAY_TIMEOUT) -> PlayResult:
    """Проиграть WAV через afplay, замерив старт и полное время."""
    player = shutil.which("afplay")
    if not player:
        raise TtsUnavailable("afplay не найден: нечем воспроизвести WAV (macOS).")

    started = time.perf_counter()
    try:
        proc = subprocess.Popen(
            [player, str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as exc:
        raise TtsUnavailable(f"afplay не запустился: {exc}") from exc
    start_ms = (time.perf_counter() - started) * 1000.0

    try:
        _, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        proc.kill()
        proc.communicate()
        raise TtsError(f"afplay не уложился в {timeout:g} с.") from exc
    total_ms = (time.perf_counter() - started) * 1000.0

    if proc.returncode != 0:
        detail = (stderr or "").strip().splitlines()
        tail = " | ".join(detail[-3:]) if detail else "без вывода"
        raise TtsError(f"afplay завершился с кодом {proc.returncode}: {tail}")

    return PlayResult(start_ms=start_ms, total_ms=total_ms)


def concat_wavs(paths: Sequence[Path], out_path: Path) -> Path:
    """Склеить части потоковой озвучки в один WAV для ``--out``.

    Все части пишет один и тот же TTS-движок с одинаковыми параметрами,
    поэтому формат проверяется и должен совпасть; иначе — понятная ошибка.
    """
    parts = [Path(path) for path in paths]
    if not parts:
        raise TtsError("нечего объединять: нет частей WAV")

    target = Path(out_path).expanduser()
    if target.parent and not target.parent.exists():
        raise TtsError(f"каталог для --out не существует: {target.parent}")

    params: tuple[int, int, int] | None = None
    try:
        with wave.open(str(target), "wb") as destination:
            for part in parts:
                with wave.open(str(part), "rb") as source:
                    current = (
                        source.getnchannels(),
                        source.getsampwidth(),
                        source.getframerate(),
                    )
                    if params is None:
                        params = current
                        destination.setnchannels(current[0])
                        destination.setsampwidth(current[1])
                        destination.setframerate(current[2])
                    elif current != params:
                        raise TtsError(
                            f"части WAV разного формата: {part} {current}, ожидалось {params}"
                        )
                    destination.writeframes(source.readframes(source.getnframes()))
    except (wave.Error, EOFError, OSError) as exc:
        raise TtsError(f"не удалось собрать {target} из частей: {exc}") from exc

    return target


def speak_stream(
    sentences: Sequence[str],
    out_dir: Path,
    *,
    voice: str = DEFAULT_VOICE,
    say_bin: str | Path | None = None,
    rate: int | None = None,
    tts_timeout: float = DEFAULT_TTS_TIMEOUT,
    play_timeout: float = DEFAULT_PLAY_TIMEOUT,
    play: bool = True,
    workers: int = STREAM_WORKERS,
    engine: TTSAdapter | None = None,
    tts_engine: str | None = None,
    piper_bin: str | Path | None = None,
    piper_model: str | Path | None = None,
) -> StreamResult:
    """Озвучить ответ по предложениям, не дожидаясь синтеза всего текста.

    Первое предложение синтезируется синхронно и сразу отдаётся в ``afplay``.
    Остальные синтезируются в пуле (``workers`` потоков), пока звучит
    предыдущее; воспроизведение идёт строго по порядку. Если следующий чанк не
    успел — речь делает паузу, и это видно в ``wait_ms``.

    ``engine`` — адаптер из ``eva.tts``; если не передан, он выбирается один
    раз на всю озвучку (Piper → system → null), чтобы все предложения шли
    через один и тот же движок.

    ``first_sound_ms`` — от старта этой функции до запуска afplay с первым
    предложением; вызывающий код прибавляет к нему ASR и маршрут.
    """
    texts = [str(item).strip() for item in sentences]
    texts = [item for item in texts if item]
    if not texts:
        raise TtsError("нечего синтезировать: список предложений пуст")

    directory = Path(out_dir).expanduser()
    if not directory.is_dir():
        raise TtsError(f"каталог для частей WAV не существует: {directory}")

    chosen = engine
    if chosen is None:
        chosen = select_engine(
            tts_engine,
            piper_bin=piper_bin,
            piper_model=piper_model,
            say_bin=say_bin,
            voice=voice,
        )

    started = time.perf_counter()

    def now_ms() -> float:
        return (time.perf_counter() - started) * 1000.0

    chunks = [
        ChunkTiming(index=index, text=text, chars=len(text))
        for index, text in enumerate(texts)
    ]
    parts = [directory / f"reply.part{index:02d}.wav" for index in range(len(texts))]

    def synth_chunk(index: int) -> SynthesisResult:
        return synthesize(
            texts[index],
            parts[index],
            voice=voice,
            say_bin=say_bin,
            rate=rate,
            timeout=tts_timeout,
            engine=chosen,
        )

    first_sound_ms: float | None = None
    first_play_offset_ms: float | None = None
    play_span_ms: float | None = None
    first_play_abs_ms: float | None = None
    last_play_end_ms: float | None = None

    # Первое предложение — синхронно: именно оно определяет первый звук.
    first_result = synth_chunk(0)
    first = chunks[0]
    first.synth_ms = first_result.synth_ms
    first.first_fragment_ms = first_result.first_fragment_ms
    first.synth_done_ms = now_ms()

    executor = ThreadPoolExecutor(max_workers=max(1, int(workers)))
    try:
        futures = {
            index: executor.submit(synth_chunk, index) for index in range(1, len(texts))
        }

        if play:
            before = time.perf_counter()
            playback = play_audio(first_result.path, timeout=play_timeout)
            first_play_offset_ms = playback.start_ms
            first_play_abs_ms = (before - started) * 1000.0 + playback.start_ms
            first_sound_ms = first_play_abs_ms
            last_play_end_ms = first_play_abs_ms + (playback.total_ms - playback.start_ms)
            first.play_start_ms = first_play_abs_ms
            first.play_total_ms = playback.total_ms

        for index in range(1, len(texts)):
            wanted = time.perf_counter()
            result = futures[index].result()
            chunk = chunks[index]
            chunk.synth_ms = result.synth_ms
            chunk.first_fragment_ms = result.first_fragment_ms
            chunk.synth_done_ms = now_ms()
            # Пауза в речи: сколько ждали готовый чанк после конца предыдущего.
            chunk.wait_ms = (time.perf_counter() - wanted) * 1000.0 if play else None

            if play:
                before = time.perf_counter()
                playback = play_audio(result.path, timeout=play_timeout)
                start_abs = (before - started) * 1000.0 + playback.start_ms
                chunk.play_start_ms = start_abs
                chunk.play_total_ms = playback.total_ms
                last_play_end_ms = start_abs + (playback.total_ms - playback.start_ms)
    finally:
        executor.shutdown(wait=True)

    if play and first_play_abs_ms is not None and last_play_end_ms is not None:
        play_span_ms = last_play_end_ms - first_play_abs_ms

    return StreamResult(
        chunks=chunks,
        parts=parts,
        first_sound_ms=first_sound_ms,
        first_play_offset_ms=first_play_offset_ms,
        synth_total_ms=sum(chunk.synth_ms or 0.0 for chunk in chunks),
        play_span_ms=play_span_ms,
        wall_ms=now_ms(),
    )


# --------------------------------------------------------------------------- #
# вывод
# --------------------------------------------------------------------------- #
def _ms(value: float | None, *, width: int = 8) -> str:
    if value is None:
        return "—".rjust(width)
    return f"{value:{width}.1f}"


def format_timings(
    timings: Timings,
    *,
    no_play: bool = False,
    no_tts: bool = False,
) -> str:
    """Человекочитаемая таблица замеров."""
    lines = ["Тайминги, мс:"]
    asr_note = "  (--text: ASR пропущен)" if timings.asr_ms is None else ""
    lines.append(f"  ASR                {_ms(timings.asr_ms)}{asr_note}")
    lines.append(f"  Маршрут (eva)      {_ms(timings.route_ms)}")

    if no_tts:
        lines.append(f"  TTS / звук         {_ms(None)}  (нет ответа скилла — озвучивать нечего)")
        return "\n".join(lines)

    tts_note = ""
    if timings.tts_first_fragment_ms is not None:
        tts_note = f"  (первый фрагмент {timings.tts_first_fragment_ms:.1f})"
    if timings.stream:
        lines.append(f"  TTS, 1-й чанк      {_ms(timings.tts_synth_ms)}{tts_note}")
        lines.append(
            f"  TTS, все чанки     {_ms(timings.tts_total_synth_ms)}"
            "  (сумма синтеза; идёт во время речи)"
        )
        lines.append(f"  Чанков            {timings.chunks:8d}  (поток по предложениям)")
    else:
        lines.append(f"  TTS, синтез WAV    {_ms(timings.tts_synth_ms)}{tts_note}")

    if no_play:
        lines.append(f"  Воспроизведение    {_ms(None)}  (--no-play)")
    else:
        lines.append(f"  Старт afplay       {_ms(timings.play_start_ms)}")
        lines.append(f"  Воспроизведение    {_ms(timings.play_total_ms)}  (полное)")

    lines.append("  " + "-" * 46)
    first = timings.first_sound_ms
    if first is None:
        reason = "воспроизведение отключено" if no_play else "звук не воспроизводился"
        lines.append(f"  До первого звука   {_ms(None)}  ({reason})")
        estimate = timings.first_sound_estimate_ms
        if estimate is not None:
            lines.append(
                f"  Оценка до звука    {_ms(estimate)}"
                "  (ASR + маршрут + синтез того, что играем)"
            )
    elif timings.stream:
        lines.append(
            f"  До первого звука   {_ms(first)}"
            "  (ASR + маршрут + синтез 1-го чанка + старт afplay)"
        )
    else:
        lines.append(f"  До первого звука   {_ms(first)}  (ASR + маршрут + синтез + старт afplay)")
    live = timings.live_estimate_ms
    if live is not None:
        lines.append(f"  Ориентир live say  {_ms(live)}  (ASR + маршрут + первый фрагмент TTS)")
    return "\n".join(lines)


def _json_payload(
    *,
    ok: bool,
    question: str,
    input_kind: str,
    route: RouteResult | None,
    timings: Timings,
    audio_path: Path | None,
    audio_saved: bool,
    played: bool,
    audio_parts: Sequence[Path] | None = None,
    tts: dict[str, Any] | None = None,
    error: str | None = None,
    stage: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": ok,
        "input": {"kind": input_kind, "question": question},
        "route": None
        if route is None
        else {
            "kind": route.route_kind,
            "reason": route.route_reason,
            "skill": route.skill,
            "latency_ms": round(route.latency_ms, 1),
            "answered": route.answered,
        },
        "answer": None if route is None else route.answer_text,
        "spoken_text": None if route is None else route.spoken_text,
        "timings": timings.to_dict(),
        "audio": {
            "path": str(audio_path) if audio_path else None,
            "saved": audio_saved,
            "played": played,
            "parts": [str(part) for part in audio_parts] if audio_parts else [],
        },
    }
    if tts is not None:
        payload["tts"] = tts
    if error is not None:
        payload["error"] = error
        payload["stage"] = stage
    return payload


def _fail(
    args: argparse.Namespace,
    stage: str,
    message: str,
    code: int,
    *,
    route: RouteResult | None = None,
    timings: Timings | None = None,
    question: str = "",
) -> int:
    print(f"voice_loop: {message}", file=sys.stderr)
    if args.json:
        print(
            json.dumps(
                _json_payload(
                    ok=False,
                    question=question,
                    input_kind="text" if args.text is not None else "wav",
                    route=route,
                    timings=timings if timings is not None else Timings(),
                    audio_path=None,
                    audio_saved=False,
                    played=False,
                    error=message,
                    stage=stage,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
    return code


# --------------------------------------------------------------------------- #
# основной сценарий
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace, *, cfg: Any | None = None) -> int:
    """Выполнить конвейер, напечатать результат и вернуть код выхода."""
    cfg = cfg if cfg is not None else load_config()
    input_kind = "text" if args.text is not None else "wav"
    timings = Timings()
    question = ""
    route: RouteResult | None = None
    audio_path: Path | None = None
    audio_saved = False
    played = False

    try:
        if args.text is not None:
            question = str(args.text).strip()
            if not question:
                raise VoiceError("--text пустой")
        else:
            wav_path = Path(args.wav)
            info = wav_info(wav_path)
            if info["channels"] != 1 or info["rate"] != DEFAULT_SAMPLE_RATE:
                print(
                    "voice_loop: предупреждение: WAV {rate} Гц, {channels} кан. "
                    "whisper.cpp ждёт 16 кГц моно (16-bit PCM).".format(**info),
                    file=sys.stderr,
                )
            binary = find_whisper_binary(args.whisper_bin)
            model = find_whisper_model(args.whisper_model, binary)
            if binary is None or model is None:
                raise AsrUnavailable(asr_help(binary, model, args.whisper_model))
            question, timings.asr_ms = transcribe_wav(
                wav_path,
                binary,
                model,
                lang=args.lang,
                timeout=args.asr_timeout,
            )

        route = ask_hybrid(question, cfg)
        timings.route_ms = route.latency_ms

        if not args.json:
            source = "--text" if args.text is not None else str(args.wav)
            print(f"Вход: {source}")
            print(f"Вопрос: {question}")
            print(f"Маршрут: {route.route_kind}" + (f" ({route.skill})" if route.skill else ""))
            print(f"Причина: {route.route_reason or '—'}")

        if not route.answered:
            note = (
                "Модель не вызывалась: прототип озвучивает только ответы скиллов. "
                "Вопрос требует маршрута small/big; ollama и модели не запускаются."
            )
            if args.json:
                print(
                    json.dumps(
                        _json_payload(
                            ok=True,
                            question=question,
                            input_kind=input_kind,
                            route=route,
                            timings=timings,
                            audio_path=None,
                            audio_saved=False,
                            played=False,
                            error=note,
                            stage="model_required",
                        ),
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            else:
                print("Ответ: — (модель не вызывалась)")
                print(f"Озвучка: пропущена. {note}")
                print()
                print(format_timings(timings, no_tts=True))
            return 0

        if not args.json:
            print(f"Ответ: {route.answer_text}")
            print(f"Озвучка: {route.spoken_text}")

        # Движок выбирается один раз на весь ответ: Piper, если есть бинарь и
        # модель, иначе системный say/espeak. Явный --tts перекрывает автовыбор.
        engine = select_engine(
            args.tts_engine,
            piper_bin=args.piper_bin,
            piper_model=args.piper_model,
            say_bin=args.say_bin,
            voice=args.voice,
        )
        if not args.json:
            print(f"TTS-движок: {describe_engine(engine)}")

        sentences = split_sentences(route.spoken_text)
        if not sentences:
            raise VoiceError("не удалось разбить ответ на предложения для озвучки")

        with tempfile.TemporaryDirectory(prefix="eva-voice-") as tmp:
            work_dir = Path(tmp)
            out_path: Path | None = None
            if args.out is not None:
                out_path = Path(args.out).expanduser()
                if out_path.parent and not out_path.parent.exists():
                    raise TtsError(f"каталог для --out не существует: {out_path.parent}")
                audio_saved = True

            audio_parts: list[Path] = []

            if args.no_stream:
                # Старое поведение: один WAV на весь ответ, затем воспроизведение.
                target = out_path if out_path is not None else work_dir / "reply.wav"
                synthesis = synthesize(
                    route.spoken_text,
                    target,
                    voice=args.voice,
                    say_bin=args.say_bin,
                    rate=args.rate,
                    timeout=args.tts_timeout,
                    engine=engine,
                )
                timings.tts_synth_ms = synthesis.synth_ms
                timings.tts_first_fragment_ms = synthesis.first_fragment_ms
                audio_path = synthesis.path

                if not args.no_play:
                    playback = play_audio(synthesis.path, timeout=args.play_timeout)
                    timings.play_start_ms = playback.start_ms
                    timings.play_total_ms = playback.total_ms
                    played = True
            else:
                # Поток: говорим первое предложение, остальные синтезируем рядом.
                stream = speak_stream(
                    sentences,
                    work_dir,
                    voice=args.voice,
                    say_bin=args.say_bin,
                    rate=args.rate,
                    tts_timeout=args.tts_timeout,
                    play_timeout=args.play_timeout,
                    play=not args.no_play,
                    engine=engine,
                )
                audio_parts = list(stream.parts)
                first_chunk = stream.chunks[0]
                timings.stream = True
                timings.chunks = len(stream.chunks)
                timings.chunk_stats = stream.chunks
                timings.tts_synth_ms = first_chunk.synth_ms
                timings.tts_first_fragment_ms = first_chunk.first_fragment_ms
                timings.tts_total_synth_ms = stream.synth_total_ms
                timings.play_start_ms = stream.first_play_offset_ms
                timings.play_total_ms = stream.play_span_ms
                played = not args.no_play

                if out_path is not None:
                    audio_path = concat_wavs(stream.parts, out_path)
                else:
                    audio_path = stream.parts[0]

            if args.json:
                print(
                    json.dumps(
                        _json_payload(
                            ok=True,
                            question=question,
                            input_kind=input_kind,
                            route=route,
                            timings=timings,
                            audio_path=audio_path,
                            audio_saved=audio_saved,
                            played=played,
                            audio_parts=audio_parts,
                            tts={
                                "engine": engine.name,
                                "description": describe_engine(engine),
                            },
                        ),
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            else:
                print()
                print(format_timings(timings, no_play=args.no_play))
                if audio_saved:
                    print(f"Аудио: {audio_path}")
                else:
                    print("Аудио: временный файл (укажите --out, чтобы сохранить)")

        return 0

    except AsrUnavailable as exc:
        return _fail(args, "asr", str(exc), 2, route=route, timings=timings, question=question)
    except AsrError as exc:
        return _fail(args, "asr", str(exc), 2, route=route, timings=timings, question=question)
    except TtsUnavailable as exc:
        return _fail(args, "tts", str(exc), 3, route=route, timings=timings, question=question)
    except TtsError as exc:
        return _fail(args, "tts", str(exc), 3, route=route, timings=timings, question=question)
    except VoiceError as exc:
        return _fail(args, "input", str(exc), 1, route=route, timings=timings, question=question)
    except KeyboardInterrupt:
        print("voice_loop: остановлен по Ctrl+C", file=sys.stderr)
        return 130


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
