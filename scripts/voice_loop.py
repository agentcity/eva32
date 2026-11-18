#!/usr/bin/env python3
"""Голосовой прототип EVA32: WAV → whisper.cpp → гибрид → say → звук.

Конвейер (ADR 0010)::

    WAV → ASR (whisper.cpp) → текст → eva.router (скилл) → текст → TTS (say -v Milena)

Что важно знать до запуска:

* **Модели не вызываются.** Прототип проверяет только детерминированную часть
  гибрида — скиллы из ``eva.skills``. Если вопрос не покрыт скиллом, скрипт
  честно печатает маршрут ``small``/``big`` и не идёт в сеть/ollama.
* **ASR опционален.** Нет собранного whisper.cpp — скрипт не падает, а печатает
  команды сборки и предлагает ``--text`` для отладки цепочки без ASR.
* **TTS — macOS ``say`` с голосом Milena (ru_RU).** Синтез идёт в WAV
  (16 кГц, моно, 16 бит), затем файл проигрывается ``afplay``. Так можно
  измерить и первый фрагмент синтеза, и реальный старт звука.
* **Замеры.** ASR, маршрут, синтез TTS, первый фрагмент TTS, старт
  воспроизведения и итог до первого звука — в миллисекундах.

Примеры::

    python3 scripts/voice_loop.py --text "Сколько стоит тариф Старт"
    python3 scripts/voice_loop.py --text "Сколько стоит тариф Старт" --no-play
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
from dataclasses import asdict, dataclass
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
from eva.types import Message, Request  # noqa: E402

try:  # скиллы — отдельный пакет; без них прототип остаётся работоспособным
    from eva.skills import try_skills as _try_skills  # noqa: E402
except Exception:  # pragma: no cover — зависит от состояния соседних задач
    _try_skills = None  # type: ignore[assignment]


DEFAULT_VOICE = os.environ.get("EVA_VOICE", "Milena")
DEFAULT_LANG = os.environ.get("EVA_WHISPER_LANG", "ru")
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_TTS_TIMEOUT = 120.0
DEFAULT_ASR_TIMEOUT = 180.0
DEFAULT_PLAY_TIMEOUT = 180.0

#: say пишет WAV с выровненным 4-КиБ заголовком; всё, что больше, — аудиоданные.
#: Порог нужен, чтобы «первый фрагмент» не оказывался моментом записи заголовка.
WAV_HEADER_BYTES = 4096

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


class TtsUnavailable(VoiceError):
    """TTS недоступен (нет say/afplay или голоса)."""


class TtsError(VoiceError):
    """TTS запустился, но синтез/воспроизведение не удались."""


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
class SynthesisResult:
    """Результат синтеза WAV через say."""

    path: Path
    synth_ms: float
    first_fragment_ms: float | None


@dataclass
class PlayResult:
    """Результат воспроизведения WAV через afplay."""

    start_ms: float
    total_ms: float


@dataclass
class Timings:
    """Замеры этапов в миллисекундах."""

    asr_ms: float | None = None
    route_ms: float = 0.0
    tts_synth_ms: float | None = None
    tts_first_fragment_ms: float | None = None
    play_start_ms: float | None = None
    play_total_ms: float | None = None

    @property
    def first_sound_ms(self) -> float | None:
        """Время от старта конвейера до первого звука (файл уже синтезирован).

        В текущем прототипе say сначала синтезирует весь ответ в WAV, поэтому
        в путь до первого звука входит полный синтез. Для длинных ответов это
        заметно; живой ``say`` без файла заговорил бы раньше (см.
        ``live_estimate_ms``).
        """
        if self.play_start_ms is None or self.tts_synth_ms is None:
            return None
        return (self.asr_ms or 0.0) + self.route_ms + self.tts_synth_ms + self.play_start_ms

    @property
    def live_estimate_ms(self) -> float | None:
        """Ориентир для live say: ASR + маршрут + первый фрагмент синтеза."""
        if self.tts_first_fragment_ms is None:
            return None
        return (self.asr_ms or 0.0) + self.route_ms + self.tts_first_fragment_ms

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["first_sound_ms"] = self.first_sound_ms
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
            "say -v Milena. Если whisper.cpp нет, используйте --text."
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
        help=f"голос say (по умолчанию {DEFAULT_VOICE})",
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
        "--say-bin",
        default=None,
        metavar="FILE",
        help="путь к say (по умолчанию из PATH)",
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=None,
        help="скорость речи say (-r), слов в минуту",
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="синтезировать WAV, но не воспроизводить (для тестов и замеров)",
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
        help=f"таймаут синтеза say, сек (по умолчанию {DEFAULT_TTS_TIMEOUT:g})",
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
# say / afplay
# --------------------------------------------------------------------------- #
def find_say(explicit: str | Path | None = None) -> Path:
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise TtsUnavailable(f"say не найден: {path}")
        return path
    found = shutil.which("say")
    if not found:
        raise TtsUnavailable(
            "TTS недоступен: say не найден. Прототип рассчитан на macOS с голосом Milena."
        )
    return Path(found)


def build_say_command(
    say_bin: str | Path,
    voice: str,
    out_path: str | Path,
    text: str,
    *,
    rate: int | None = None,
) -> list[str]:
    """Команда say, которая пишет WAV 16 кГц/моно/16 бит."""
    command = [
        str(say_bin),
        "-v",
        voice,
        "-o",
        str(out_path),
        "--file-format=WAVE",
        "--data-format=LEI16@16000",
    ]
    if rate is not None:
        command.extend(["-r", str(int(rate))])
    command.append(text)
    return command


def synthesize(
    text: str,
    out_path: Path,
    *,
    voice: str = DEFAULT_VOICE,
    say_bin: str | Path | None = None,
    rate: int | None = None,
    timeout: float = DEFAULT_TTS_TIMEOUT,
) -> SynthesisResult:
    """Синтезировать текст в WAV и замерить первый фрагмент.

    ``first_fragment_ms`` — время до появления первых байт аудио в файле:
    это нижняя оценка того, когда заговорил бы живой say.
    """
    say = find_say(say_bin)
    target = Path(out_path).expanduser()
    if target.parent and not target.parent.exists():
        raise TtsError(f"каталог для --out не существует: {target.parent}")
    try:
        target.unlink(missing_ok=True)
    except OSError as exc:
        raise TtsError(f"не удалось подготовить {target}: {exc}") from exc

    command = build_say_command(say, voice, target, text, rate=rate)
    started = time.perf_counter()
    try:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as exc:
        raise TtsUnavailable(f"say не запустился: {exc}") from exc

    first_fragment_ms: float | None = None
    while proc.poll() is None:
        if first_fragment_ms is None:
            try:
                if target.stat().st_size > WAV_HEADER_BYTES:  # уже аудиоданные
                    first_fragment_ms = (time.perf_counter() - started) * 1000.0
            except OSError:
                pass
        if time.perf_counter() - started > timeout:
            proc.kill()
            proc.communicate()
            raise TtsError(
                f"say не уложился в {timeout:g} с. Увеличьте --tts-timeout "
                "или сократите ответ."
            )
        time.sleep(0.005)

    stdout, stderr = proc.communicate()
    synth_ms = (time.perf_counter() - started) * 1000.0
    if first_fragment_ms is None and target.exists() and target.stat().st_size > WAV_HEADER_BYTES:
        first_fragment_ms = synth_ms

    if proc.returncode != 0:
        detail = (stderr or stdout or "").strip().splitlines()
        tail = " | ".join(detail[-3:]) if detail else "без вывода"
        raise TtsError(f"say завершился с кодом {proc.returncode}: {tail}")

    if not target.is_file() or target.stat().st_size <= 44:
        raise TtsError(
            "say не создал аудио. Проверьте голос: say -v '?' | grep Milena "
            f"(запрошен {voice!r})."
        )

    return SynthesisResult(
        path=target,
        synth_ms=synth_ms,
        first_fragment_ms=first_fragment_ms,
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
        },
    }
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

        with tempfile.TemporaryDirectory(prefix="eva-voice-") as tmp:
            if args.out is not None:
                out_path = Path(args.out).expanduser()
                audio_saved = True
            else:
                out_path = Path(tmp) / "reply.wav"

            synthesis = synthesize(
                route.spoken_text,
                out_path,
                voice=args.voice,
                say_bin=args.say_bin,
                rate=args.rate,
                timeout=args.tts_timeout,
            )
            timings.tts_synth_ms = synthesis.synth_ms
            timings.tts_first_fragment_ms = synthesis.first_fragment_ms
            audio_path = synthesis.path

            if not args.no_play:
                playback = play_audio(synthesis.path, timeout=args.play_timeout)
                timings.play_start_ms = playback.start_ms
                timings.play_total_ms = playback.total_ms
                played = True

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
