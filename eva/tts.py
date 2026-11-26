"""TTS-адаптер EVA32: ElevenLabs → Piper → системный say/espeak → null.

Задача — [`ai/tasks/0191-elevenlabs-tts.md`](../ai/tasks/0191-elevenlabs-tts.md)
(провайдер), [`ai/tasks/0093-eva-tts-piper.md`](../ai/tasks/0093-eva-tts-piper.md)
(локальный Piper). Весь синтез идёт через один интерфейс, а движок выбирается
по доступности:

============== ==========================================================
``elevenlabs`` облачный нейро-TTS: ``POST /v1/text-to-speech/{voice_id}``,
               ключ ``ELEVENLABS_API_KEY``. Возвращает MP3 (по умолчанию
               ``mp3_44100_128``). Текст уходит в облако ElevenLabs.
``piper``      локальный нейро-TTS: ``piper --model MODEL --output_file OUT``,
               текст — в stdin. Пути из ``PIPER_BIN``/``PIPER_MODEL``
               (или ``--piper-bin``/``--piper-model`` в ``voice_loop.py``).
``system``     macOS ``say`` (голос Milena) или Linux ``espeak``/``espeak-ng``.
``null``       заглушка: синтеза нет, ничего не пишет, ничего не играет.
============== ==========================================================

Автовыбор: **ElevenLabs, если задан ключ**, иначе **Piper, если доступен**
(есть бинарь и модель), иначе **SystemTTS**, иначе **NullTTS**. Явный выбор
(``EVA_TTS=elevenlabs|piper|system|null`` / ``prefer=``) перекрывает
автовыбор — даже если движок недоступен: тогда при синтезе будет понятная
:class:`TTSUnavailable`, а не молчаливая подмена. Исключение — ElevenLabs:
если облако ответило ошибкой (401/429/сеть/пусто), модуль логирует причину
(без ключа!) и **сам откатывается на Piper/system**, чтобы пользователь не
остался без голоса.

Контракт синтеза — ``synthesize(text) -> bytes | Path | (bytes, content_type)``::

    from eva.tts import synthesize, synthesize_audio

    wav_bytes = synthesize("Сколько стоит тариф Старт")        # bytes (Piper)
    mp3, mime = synthesize_audio("Привет")                      # ElevenLabs
    path = synthesize("Привет", "/tmp/reply.wav")               # Path

:class:`ElevenLabsTTS` дополнительно отдаёт пару ``(bytes, content_type)``
через :meth:`ElevenLabsTTS.synthesize_audio`; серверу это нужно, чтобы
выставить ``audio/mpeg`` вместо жёсткого ``audio/wav``.

Адаптеры умеют также :meth:`TTSAdapter.synthesize_timed` — с замерами
``synth_ms`` и ``first_fragment_ms`` для голосового цикла.

Только стандартная библиотека. Сеть используется только ElevenLabs (и только
при заданном ключе); модели не скачиваются, реальный звук не воспроизводится
(это делает ``voice_loop.py`` через afplay).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ = [
    "DEFAULT_ELEVENLABS_FORMAT",
    "DEFAULT_ELEVENLABS_MODEL",
    "DEFAULT_ELEVENLABS_TIMEOUT",
    "DEFAULT_ELEVENLABS_VOICE_ID",
    "DEFAULT_SAY_VOICE",
    "DEFAULT_TIMEOUT",
    "DEFAULT_TTS_LANG",
    "ELEVENLABS_API_URL",
    "ELEVENLABS_CACHE_SIZE",
    "ElevenLabsTTS",
    "ENGINE_CHOICES",
    "NullTTS",
    "PiperTTS",
    "SystemTTS",
    "SynthesisResult",
    "TTSAdapter",
    "TTSError",
    "TTSUnavailable",
    "build_say_command",
    "clear_elevenlabs_cache",
    "describe_engine",
    "elevenlabs_cache_info",
    "find_espeak_binary",
    "find_piper_binary",
    "find_piper_model",
    "find_say_binary",
    "piper_help",
    "select_engine",
    "synthesize",
    "synthesize_audio",
    "system_help",
]


#: Корень репозитория: ``eva/tts.py`` → ``parents[1]``. Нужен для поиска
#: бинаря и моделей в типовых каталогах (``piper/piper``, ``models/``).
ROOT = Path(__file__).resolve().parents[1]

#: Таймаут одного синтеза по умолчанию, сек.
DEFAULT_TIMEOUT = 120.0

#: Голос macOS ``say`` по умолчанию (ru_RU).
DEFAULT_SAY_VOICE = os.environ.get("EVA_VOICE", "Milena").strip() or "Milena"

#: Язык системного TTS (espeak) и подсказка для модели Piper.
DEFAULT_TTS_LANG = os.environ.get("EVA_TTS_LANG", "ru").strip() or "ru"

#: Допустимые значения ``--tts``/``EVA_TTS``. Порядок — приоритет автовыбора
#: (``auto`` не движок, а «выбери сам»).
ENGINE_CHOICES = ("auto", "elevenlabs", "piper", "system", "null")

# --------------------------------------------------------------------------- #
# ElevenLabs (задача 0191)
# --------------------------------------------------------------------------- #
#: Базовый URL синтеза; голос добавляется как ``/{voice_id}``.
ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1/text-to-speech"

#: Голос по умолчанию — публичный голос из библиотеки ElevenLabs. Модель
#: ``eleven_multilingual_v2`` говорит на русском, но акцент зависит от голоса:
#: если не нравится, поставьте свой ``ELEVENLABS_VOICE_ID`` (ID виден в
#: библиотеке голосов или в ответе ``GET /v1/voices``).
DEFAULT_ELEVENLABS_VOICE_ID = "EXAVITQu4vr4xnSDxMaL"

#: Модель синтеза: мультиязычная, поддерживает русский.
DEFAULT_ELEVENLABS_MODEL = "eleven_multilingual_v2"

#: Формат ответа: MP3 44,1 кГц/128 кбит — его играют и Web Audio, и AVPlayer.
DEFAULT_ELEVENLABS_FORMAT = "mp3_44100_128"

#: Таймаут одного HTTP-запроса к ElevenLabs, сек. Облако отвечает быстрее
#: локального Piper, но сеть бывает медленной; 30 с — с большим запасом.
DEFAULT_ELEVENLABS_TIMEOUT = 30.0

#: Размер LRU-кэша успешных ответов (одна фраза = одна запись): экономит
#: квоту на повторных «стоп»/«повтори» и одинаковых коротких репликах.
ELEVENLABS_CACHE_SIZE = 50

#: Голосовые настройки запроса. Стабильность ниже — живее интонация; similarity
#: держит тембр выбранного голоса; style=0 — без актёрской подачи (для Евы
#: ровный разговорный тон важнее драматизма).
ELEVENLABS_VOICE_SETTINGS: dict[str, float] = {
    "stability": 0.4,
    "similarity_boost": 0.75,
    "style": 0.0,
}

#: Content-Type по префиксу ``output_format``. Браузеру нужен точный MIME:
#: MP3 — ``audio/mpeg``, WAV/PCM — ``audio/wav``, Opus — ``audio/ogg``.
ELEVENLABS_CONTENT_TYPES = {
    "mp3": "audio/mpeg",
    "pcm": "audio/wav",
    "opus": "audio/ogg",
    "ulaw": "audio/basic",
}

#: Общий LRU-кэш для всех экземпляров: ``select_engine`` создаёт адаптер на
#: каждый запрос, поэтому кэш должен жить на уровне модуля, иначе он бесполезен.
_ELEVENLABS_CACHE: "OrderedDict[tuple[str, str, str, str], tuple[bytes, str]]" = (
    OrderedDict()
)
_ELEVENLABS_CACHE_LOCK = threading.Lock()
_ELEVENLABS_CACHE_HITS = 0
_ELEVENLABS_CACHE_MISSES = 0

#: ``say`` пишет WAV с выровненным 4-КиБ заголовком: всё, что больше, —
#: аудиоданные. Порог важен, чтобы «первый фрагмент» не оказывался моментом
#: записи заголовка.
WAV_HEADER_BYTES = 4096

#: Минимальный валидный WAV (заголовок). Меньше — синтез не удался.
MIN_WAV_BYTES = 44

#: Имена бинарника Piper в PATH.
PIPER_BINARY_NAMES = ("piper",)

#: Типовые места Piper относительно репозитория и дома.
PIPER_BINARY_CANDIDATES = (
    "piper/piper",
    "bin/piper",
    "tools/piper/piper",
    "third_party/piper/piper",
    "vendor/piper/piper",
)

#: Каталоги, где ищем модели Piper (`.onnx`), в порядке предпочтения.
PIPER_MODEL_DIRS = (
    "models",
    "piper/models",
    "third_party/piper/models",
    "vendor/piper/models",
)

#: Имена бинарника espeak (Linux).
ESPEAK_BINARY_NAMES = ("espeak-ng", "espeak")


# --------------------------------------------------------------------------- #
# ошибки и результат
# --------------------------------------------------------------------------- #
class TTSError(RuntimeError):
    """Синтез не удался: процесс упал, таймаут, пустой файл."""


class TTSUnavailable(TTSError):
    """Движок не настроен: нет бинаря/модели/голоса."""


@dataclass
class SynthesisResult:
    """Результат синтеза: путь к WAV и замеры для голосового цикла."""

    path: Path
    synth_ms: float
    first_fragment_ms: float | None
    engine: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "engine": self.engine,
            "synth_ms": round(self.synth_ms, 1),
            "first_fragment_ms": round(self.first_fragment_ms, 1)
            if self.first_fragment_ms is not None
            else None,
        }


# --------------------------------------------------------------------------- #
# вспомогательное: env, файлы, процесс
# --------------------------------------------------------------------------- #
def _env(env: Mapping[str, str] | None, *names: str) -> str | None:
    """Первое непустое значение из переменных окружения (последние — фолбэк)."""
    source: Mapping[str, str] = os.environ if env is None else env
    for name in names:
        raw = source.get(name)
        text = str(raw).strip() if raw is not None else ""
        if text:
            return text
    return None


def _is_executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def _clean_text(text: object) -> str:
    value = str(text or "").strip()
    if not value:
        raise TTSError("пустой текст для синтеза")
    return value


def _redact(value: object, secret: str | None) -> str:
    """Текст ошибки без секрета: ключ не должен попасть ни в лог, ни в 503."""
    text = str(value)
    if secret:
        text = text.replace(secret, "***")
    return text


def _env_flag(env: Mapping[str, str] | None, name: str, default: bool = True) -> bool:
    raw = _env(env, name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "нет")


def _content_type_for_format(output_format: str) -> str:
    """MIME по префиксу формата ElevenLabs (``mp3_44100_128`` → audio/mpeg)."""
    prefix = (output_format or "").strip().lower().split("_", 1)[0]
    return ELEVENLABS_CONTENT_TYPES.get(prefix, "audio/mpeg")


def clear_elevenlabs_cache() -> None:
    """Очистить общий LRU-кэш ElevenLabs (тесты и смена ключа/голоса)."""
    global _ELEVENLABS_CACHE_HITS, _ELEVENLABS_CACHE_MISSES
    with _ELEVENLABS_CACHE_LOCK:
        _ELEVENLABS_CACHE.clear()
        _ELEVENLABS_CACHE_HITS = 0
        _ELEVENLABS_CACHE_MISSES = 0


def elevenlabs_cache_info() -> dict[str, int]:
    """Состояние кэша: размер, попадания и промахи (диагностика/тесты)."""
    with _ELEVENLABS_CACHE_LOCK:
        return {
            "size": len(_ELEVENLABS_CACHE),
            "hits": _ELEVENLABS_CACHE_HITS,
            "misses": _ELEVENLABS_CACHE_MISSES,
            "max_size": ELEVENLABS_CACHE_SIZE,
        }


def _cache_get(key: tuple[str, str, str, str]) -> tuple[bytes, str] | None:
    global _ELEVENLABS_CACHE_HITS, _ELEVENLABS_CACHE_MISSES
    with _ELEVENLABS_CACHE_LOCK:
        value = _ELEVENLABS_CACHE.get(key)
        if value is None:
            _ELEVENLABS_CACHE_MISSES += 1
            return None
        _ELEVENLABS_CACHE.move_to_end(key)
        _ELEVENLABS_CACHE_HITS += 1
        return value


def _cache_put(key: tuple[str, str, str, str], value: tuple[bytes, str]) -> None:
    with _ELEVENLABS_CACHE_LOCK:
        _ELEVENLABS_CACHE[key] = value
        _ELEVENLABS_CACHE.move_to_end(key)
        while len(_ELEVENLABS_CACHE) > ELEVENLABS_CACHE_SIZE:
            _ELEVENLABS_CACHE.popitem(last=False)


def _prepare_target(out_path: str | Path | None) -> Path:
    """Проверить каталог и убрать старый файл, чтобы не читать его как новый."""
    if out_path is None:
        raise TTSError("не указан путь для синтезированного WAV")
    target = Path(out_path).expanduser()
    if target.parent and not target.parent.exists():
        raise TTSError(f"каталог для WAV не существует: {target.parent}")
    try:
        target.unlink(missing_ok=True)
    except OSError as exc:
        raise TTSError(f"не удалось подготовить {target}: {exc}") from exc
    return target


def _tail(stdout: str | None, stderr: str | None) -> str:
    detail = (stderr or stdout or "").strip().splitlines()
    return " | ".join(detail[-3:]) if detail else "без вывода"


def _run_to_file(
    command: Sequence[str],
    stdin_text: str | None,
    out_path: Path,
    *,
    timeout: float,
    label: str,
    fragment_threshold: int,
) -> tuple[float, float | None]:
    """Запустить TTS-процесс, который пишет WAV в ``out_path``.

    Пока процесс жив, размер файла опрашивается: ``first_fragment_ms`` — время
    до появления первых аудиоданных (после ``fragment_threshold`` байт). Текст
    для stdin отдаётся через временный файл, а не через PIPE: так дочерний
    процесс не может заблокироваться на переполненном канале.
    """
    target = _prepare_target(out_path)
    started = time.perf_counter()
    stdin_file = None
    if stdin_text is not None:
        stdin_file = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
        stdin_file.write(stdin_text)
        stdin_file.flush()
        stdin_file.seek(0)

    try:
        try:
            proc = subprocess.Popen(
                list(command),
                stdin=stdin_file if stdin_file is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except OSError as exc:
            raise TTSUnavailable(f"{label} не запустился: {exc}") from exc

        first_fragment_ms: float | None = None
        while proc.poll() is None:
            if first_fragment_ms is None:
                try:
                    if target.stat().st_size > fragment_threshold:
                        first_fragment_ms = (time.perf_counter() - started) * 1000.0
                except OSError:
                    pass
            if time.perf_counter() - started > timeout:
                proc.kill()
                proc.communicate()
                raise TTSError(
                    f"{label} не уложился в {timeout:g} с. "
                    "Увеличьте --tts-timeout или сократите ответ."
                )
            time.sleep(0.005)

        stdout, stderr = proc.communicate()
    finally:
        if stdin_file is not None:
            stdin_file.close()

    synth_ms = (time.perf_counter() - started) * 1000.0
    if proc.returncode != 0:
        raise TTSError(
            f"{label} завершился с кодом {proc.returncode}: {_tail(stdout, stderr)}"
        )

    try:
        size = target.stat().st_size
    except OSError:
        size = 0
    if not target.is_file() or size <= MIN_WAV_BYTES:
        raise TTSError(
            f"{label} не создал аудио ({target}). "
            "Проверьте голос/модель и права на запись."
        )
    if first_fragment_ms is None and size > fragment_threshold:
        first_fragment_ms = synth_ms
    return synth_ms, first_fragment_ms


# --------------------------------------------------------------------------- #
# общий интерфейс адаптера
# --------------------------------------------------------------------------- #
class TTSAdapter:
    """Общий интерфейс TTS-движка.

    Наследник задаёт ``name``, :meth:`build_command` и (при необходимости)
    ``fragment_threshold``. :meth:`synthesize` и :meth:`synthesize_timed`
    реализованы здесь, чтобы все движки мерились одинаково.
    """

    name = "base"
    fragment_threshold = MIN_WAV_BYTES

    def available(self) -> bool:
        """Движок можно использовать прямо сейчас (бинарь/модель на месте)."""
        return True

    def describe(self) -> str:
        """Строка для отчёта: движок и чем он представлен."""
        return self.name

    def build_command(
        self,
        text: str,
        out_path: str | Path,
        *,
        rate: int | None = None,
        voice: str | None = None,
    ) -> tuple[list[str], str | None]:
        """Собрать argv и текст для stdin (``None`` — текст в аргументах)."""
        raise NotImplementedError

    def synthesize(
        self,
        text: str,
        out_path: str | Path | None = None,
        *,
        rate: int | None = None,
        voice: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> bytes | Path:
        """Синтезировать текст.

        Без ``out_path`` возвращает WAV-байты, с ``out_path`` — путь к файлу
        (контракт ``synthesize(text) -> bytes|Path``).
        """
        clean = _clean_text(text)
        if out_path is not None:
            target = _prepare_target(out_path)
            self.synthesize_timed(clean, target, rate=rate, voice=voice, timeout=timeout)
            return target
        with tempfile.TemporaryDirectory(prefix="eva-tts-") as tmp:
            target = Path(tmp) / "reply.wav"
            self.synthesize_timed(clean, target, rate=rate, voice=voice, timeout=timeout)
            return target.read_bytes()

    def synthesize_timed(
        self,
        text: str,
        out_path: str | Path,
        *,
        rate: int | None = None,
        voice: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> SynthesisResult:
        """Синтезировать в WAV и вернуть замеры (синтез + первый фрагмент)."""
        clean = _clean_text(text)
        target = Path(out_path).expanduser()
        command, stdin_text = self.build_command(
            clean, target, rate=rate, voice=voice
        )
        synth_ms, first_fragment_ms = _run_to_file(
            command,
            stdin_text,
            target,
            timeout=timeout,
            label=self.name,
            fragment_threshold=self.fragment_threshold,
        )
        return SynthesisResult(
            path=target,
            synth_ms=synth_ms,
            first_fragment_ms=first_fragment_ms,
            engine=self.name,
        )


# --------------------------------------------------------------------------- #
# Piper
# --------------------------------------------------------------------------- #
def find_piper_binary(
    explicit: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> Path | None:
    """Найти бинарь Piper: аргумент → ``PIPER_BIN`` → PATH → типовые места.

    Явно переданный ``explicit`` путь обязан существовать и быть исполняемым —
    иначе :class:`TTSUnavailable` (опечатка в конфиге не должна тихо менять
    движок). Кривой ``PIPER_BIN`` из окружения игнорируется: автовыбор должен
    спокойно уходить на системный TTS.
    """
    if explicit:
        path = Path(explicit).expanduser()
        if not _is_executable(path):
            raise TTSUnavailable(
                f"piper: {path} не найден или не исполняемый. "
                "Укажите PIPER_BIN или --piper-bin."
            )
        return path

    raw = _env(env, "PIPER_BIN", "EVA_PIPER_BIN")
    if raw:
        path = Path(raw).expanduser()
        if _is_executable(path):
            return path

    for name in PIPER_BINARY_NAMES:
        found = shutil.which(name)
        if found:
            return Path(found)

    for relative in PIPER_BINARY_CANDIDATES:
        for base in (ROOT, Path.home()):
            path = base / relative
            if _is_executable(path):
                return path
    return None


def _piper_model_rank(path: Path) -> tuple[int, int, str]:
    """Предпочтение моделей: русские → medium → имя (для стабильности)."""
    name = path.name.lower()
    return (0 if "ru" in name else 1, 0 if "medium" in name else 1, name)


def find_piper_model(
    explicit: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
    binary: Path | None = None,
) -> Path | None:
    """Найти модель Piper (``*.onnx``): аргумент → ``PIPER_MODEL`` → каталоги.

    Модель не скачивается. Если ``PIPER_MODEL`` задан, но файла нет, — это
    ошибка конфигурации (:class:`TTSUnavailable`), а не повод молча выбрать
    другую модель. Без явного пути каталоги только просматриваются.
    """
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise TTSUnavailable(
                f"модель Piper не найдена: {path}. Укажите PIPER_MODEL или "
                "--piper-model (файл .onnx; скачивание — отдельная задача)."
            )
        return path

    raw = _env(env, "PIPER_MODEL", "EVA_PIPER_MODEL")
    if raw:
        path = Path(raw).expanduser()
        if path.is_file():
            return path
        raise TTSUnavailable(
            f"PIPER_MODEL={path}: файл не найден. Исправьте переменную или "
            "укажите --piper-model."
        )

    roots: list[Path] = []
    model_dir = _env(env, "PIPER_MODEL_DIR", "EVA_PIPER_MODEL_DIR")
    if model_dir:
        roots.append(Path(model_dir).expanduser())
    for relative in PIPER_MODEL_DIRS:
        roots.append(ROOT / relative)
        roots.append(Path.home() / relative)
    roots.extend(
        [
            Path.home() / ".local" / "share" / "piper",
            Path.home() / ".cache" / "piper",
            Path("/usr/share/piper"),
            Path("/usr/local/share/piper"),
        ]
    )
    if binary is not None:
        roots.extend([binary.parent, binary.parent.parent / "models"])

    seen: set[Path] = set()
    candidates: list[Path] = []
    for root in roots:
        if not root or root in seen or not root.is_dir():
            continue
        seen.add(root)
        try:
            candidates.extend(path for path in root.glob("*.onnx") if path.is_file())
        except OSError:
            continue
    if not candidates:
        return None
    return sorted(candidates, key=_piper_model_rank)[0]


def piper_help(binary: Path | None, model: Path | None) -> str:
    """Человеческое сообщение: чего не хватает и как это исправить."""
    lines: list[str] = []
    if binary is None:
        lines.append(
            "Piper недоступен: бинарь piper не найден ни в PATH, ни в типовых "
            "каталогах."
        )
    else:
        lines.append(f"Piper найден ({binary}), но модель .onnx не найдена.")
    lines.extend(
        [
            "Укажите пути явно:",
            "  export PIPER_BIN=/path/to/piper",
            "  export PIPER_MODEL=/path/to/ru_RU-...-medium.onnx",
            "или флаги: python3 scripts/voice_loop.py --text ... "
            "--tts piper --piper-bin ... --piper-model ...",
            "",
            "Без Piper работает системный TTS: --tts system "
            "(macOS say / Linux espeak).",
        ]
    )
    return "\n".join(lines)


class PiperTTS(TTSAdapter):
    """Локальный нейро-TTS Piper: ``piper --model M --output_file OUT``.

    Текст уходит в stdin, WAV пишется в ``--output_file``. Пути берутся из
    аргументов, затем из ``PIPER_BIN``/``PIPER_MODEL``, затем из PATH и
    типовых каталогов. Модель не скачивается.

    Используются длинные флаги ``piper`` (пакет ``piper-tts``); старые сборки,
    знающие только ``-m``/``-f``, не поддерживаются. Голос задаётся самой
    моделью (``*.onnx``), поэтому ``voice`` здесь не используется.
    """

    name = "piper"
    fragment_threshold = MIN_WAV_BYTES

    def __init__(
        self,
        binary: str | Path | None = None,
        model: str | Path | None = None,
        *,
        length_scale: float | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._binary = binary
        self._model = model
        self._length_scale = length_scale
        self._env = env
        self._resolved: tuple[Path, Path] | None = None

    def resolve(self) -> tuple[Path, Path]:
        """Найти (бинарь, модель) или поднять :class:`TTSUnavailable`."""
        if self._resolved is not None:
            return self._resolved
        binary = find_piper_binary(self._binary, env=self._env)
        model = find_piper_model(self._model, env=self._env, binary=binary)
        if binary is None or model is None:
            raise TTSUnavailable(piper_help(binary, model))
        self._resolved = (binary, model)
        return self._resolved

    def available(self) -> bool:
        try:
            self.resolve()
        except TTSUnavailable:
            return False
        return True

    def describe(self) -> str:
        try:
            binary, model = self.resolve()
        except TTSUnavailable:
            return "piper (недоступен: нет бинаря или модели .onnx)"
        return f"piper ({binary.name}, модель {model.name})"

    def build_command(
        self,
        text: str,
        out_path: str | Path,
        *,
        rate: int | None = None,
        voice: str | None = None,
    ) -> tuple[list[str], str | None]:
        binary, model = self.resolve()
        command = [
            str(binary),
            "--model",
            str(model),
            "--output_file",
            str(Path(out_path).expanduser()),
        ]
        length_scale = self._length_scale
        if length_scale is None and rate:
            # Piper задаёт темп через length_scale (>1 — медленнее). 180 wpm
            # у say считаем нейтральным темпом; ограничиваем, чтобы не выйти
            # в невнятную крайность.
            length_scale = max(0.5, min(2.0, 180.0 / float(rate)))
        if length_scale is not None:
            command.extend(["--length_scale", f"{float(length_scale):.3f}"])
        return command, text + "\n"


# --------------------------------------------------------------------------- #
# ElevenLabs (облачный нейро-TTS)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _ElevenLabsConfig:
    """Разрешённые настройки запроса. ``repr`` не показывает ключ."""

    api_key: str
    voice_id: str
    model_id: str
    output_format: str
    timeout: float

    def __repr__(self) -> str:  # pragma: no cover - защита от случайного лога
        return (
            f"_ElevenLabsConfig(api_key='***', voice_id={self.voice_id!r}, "
            f"model_id={self.model_id!r}, output_format={self.output_format!r}, "
            f"timeout={self.timeout!r})"
        )


class ElevenLabsTTS(TTSAdapter):
    """Облачный TTS ElevenLabs: MP3 через ``POST /v1/text-to-speech/{voice_id}``.

    Настройки — из аргументов, затем из окружения:

    ========================= ==============================================
    ``ELEVENLABS_API_KEY``    обязательный ключ (заголовок ``xi-api-key``)
    ``ELEVENLABS_VOICE_ID``   голос, по умолчанию публичный русскоязычный
    ``ELEVENLABS_MODEL``      модель, по умолчанию ``eleven_multilingual_v2``
    ``ELEVENLABS_FORMAT``     формат, по умолчанию ``mp3_44100_128``
    ``ELEVENLABS_TIMEOUT``    таймаут запроса, сек (по умолчанию 30)
    ``ELEVENLABS_CACHE``      ``0`` отключает LRU-кэш на 50 фраз
    ========================= ==============================================

    Ключ не попадает ни в ``describe()``, ни в тексты ошибок: провайдер
    возвращает только понятную причину (401/429/сеть/пустой ответ). Откат на
    Piper делает модульный :func:`synthesize`, а не сам адаптер.

    ``synthesize`` возвращает ``(bytes, content_type)`` — сервер выставляет
    ``audio/mpeg`` для MP3 и ``audio/wav`` для WAV-движков. С ``out_path``
    возвращается путь (байты пишутся в файл).
    """

    name = "elevenlabs"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        voice_id: str | None = None,
        model: str | None = None,
        output_format: str | None = None,
        timeout: float | None = None,
        env: Mapping[str, str] | None = None,
        opener=None,
    ) -> None:
        self._api_key = api_key
        self._voice_id = voice_id
        self._model = model
        self._output_format = output_format
        self._timeout = timeout
        self._env = env
        self._opener = opener
        self._resolved: _ElevenLabsConfig | None = None
        self._use_cache = _env_flag(env, "ELEVENLABS_CACHE", True)
        self.last_content_type = "audio/mpeg"

    # -- настройки -----------------------------------------------------------

    def resolve(self) -> _ElevenLabsConfig:
        """Собрать настройки или поднять :class:`TTSUnavailable` без ключа.

        Ключ обязателен; остальное имеет разумные значения по умолчанию.
        Сообщение об отсутствии ключа подсказывает переменную, но не содержит
        никакого секрета.
        """
        if self._resolved is not None:
            return self._resolved
        source_env = os.environ if self._env is None else self._env
        api_key = (self._api_key or _env(source_env, "ELEVENLABS_API_KEY") or "").strip()
        if not api_key:
            raise TTSUnavailable(
                "ElevenLabs недоступен: не задан ELEVENLABS_API_KEY. "
                "Получите ключ в кабинете ElevenLabs и экспортируйте его; "
                "без ключа работает Piper/system."
            )
        voice_id = (
            self._voice_id
            or _env(source_env, "ELEVENLABS_VOICE_ID")
            or DEFAULT_ELEVENLABS_VOICE_ID
        ).strip()
        model_id = (
            self._model
            or _env(source_env, "ELEVENLABS_MODEL")
            or DEFAULT_ELEVENLABS_MODEL
        ).strip()
        output_format = (
            self._output_format
            or _env(source_env, "ELEVENLABS_FORMAT")
            or DEFAULT_ELEVENLABS_FORMAT
        ).strip()
        timeout = self._timeout
        if timeout is None:
            raw_timeout = _env(source_env, "ELEVENLABS_TIMEOUT")
            try:
                timeout = float(raw_timeout) if raw_timeout else None
            except (TypeError, ValueError):
                timeout = None
        if timeout is None or timeout <= 0:
            timeout = DEFAULT_ELEVENLABS_TIMEOUT
        self._resolved = _ElevenLabsConfig(
            api_key=api_key,
            voice_id=voice_id,
            model_id=model_id,
            output_format=output_format,
            timeout=float(timeout),
        )
        return self._resolved

    def available(self) -> bool:
        try:
            self.resolve()
        except TTSUnavailable:
            return False
        return True

    def describe(self) -> str:
        try:
            config = self.resolve()
        except TTSUnavailable:
            return "elevenlabs (недоступен: нет ELEVENLABS_API_KEY)"
        return (
            f"elevenlabs (voice {config.voice_id}, model {config.model_id}, "
            f"{config.output_format})"
        )

    def _voice_for(self, requested: str | None) -> str:
        """Голос запроса, если это похоже на ID ElevenLabs, иначе настроенный.

        Страница чата присылает пресеты Piper (``irina``/``soft``), а не ID
        ElevenLabs; такой ``voice`` игнорируется. ID из библиотеки — длинная
        буквенно-цифровая строка (обычно 20 символов).
        """
        config = self.resolve()
        candidate = (requested or "").strip()
        if len(candidate) >= 16 and candidate.isalnum():
            return candidate
        return config.voice_id

    def _effective_timeout(self, timeout: float | None) -> float:
        config = self.resolve()
        # Базовый контракт передаёт DEFAULT_TIMEOUT (120 с) даже когда значение
        # не задавали. Для облака это слишком много: считаем такое значение
        # «по умолчанию» и берём ELEVENLABS_TIMEOUT.
        if timeout is None or float(timeout) >= DEFAULT_TIMEOUT:
            return config.timeout
        return max(1.0, float(timeout))

    # -- HTTP ---------------------------------------------------------------

    @staticmethod
    def _headers(config: _ElevenLabsConfig) -> dict[str, str]:
        return {
            "xi-api-key": config.api_key,
            "Content-Type": "application/json",
            "Accept": _content_type_for_format(config.output_format),
            "User-Agent": "eva32-tts/1.0",
        }

    def build_request(
        self, text: str, *, voice: str | None = None
    ) -> tuple[urllib.request.Request, _ElevenLabsConfig, float]:
        """Собрать HTTP-запрос (вынесено для тестов и диагностики)."""
        config = self.resolve()
        voice_id = self._voice_for(voice)
        query = urllib.parse.urlencode({"output_format": config.output_format})
        url = (
            f"{ELEVENLABS_API_URL}/{urllib.parse.quote(voice_id, safe='')}?"
            f"{query}"
        )
        payload = {
            "text": text,
            "model_id": config.model_id,
            "voice_settings": dict(ELEVENLABS_VOICE_SETTINGS),
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url, data=body, headers=self._headers(config), method="POST"
        )
        return request, config, self._effective_timeout(None)

    def _open(
        self, request: urllib.request.Request, *, timeout: float
    ):
        opener = self._opener or urllib.request.urlopen
        return opener(request, timeout=timeout)

    @staticmethod
    def _response_status(response) -> int:
        status = getattr(response, "status", None)
        if status is None:
            status = getattr(response, "code", None)
        try:
            return int(status) if status is not None else 200
        except (TypeError, ValueError):
            return 200

    @staticmethod
    def _response_content_type(response) -> str:
        headers = getattr(response, "headers", None)
        if headers is None:
            return ""
        try:
            value = headers.get("Content-Type") or headers.get("content-type") or ""
        except (AttributeError, TypeError):
            return ""
        return str(value).split(";", 1)[0].strip().lower()

    def _fetch_audio(
        self, text: str, *, voice: str | None, timeout: float | None
    ) -> tuple[bytes, str]:
        config = self.resolve()
        request, _, _ = self.build_request(text, voice=voice)
        effective_timeout = self._effective_timeout(timeout)
        try:
            response = self._open(request, timeout=effective_timeout)
        except urllib.error.HTTPError as exc:
            code = getattr(exc, "code", None)
            reason = getattr(exc, "reason", "") or ""
            # HTTPError — это ещё и открытый файл-ответ; закрываем, чтобы не
            # тянуть ресурс до сборки мусора (и не шуметь ResourceWarning).
            close = getattr(exc, "close", None)
            if callable(close):
                close()
            if code == 401:
                raise TTSError(
                    "ElevenLabs отклонил ключ (HTTP 401): проверьте "
                    "ELEVENLABS_API_KEY и права ключа."
                ) from None
            if code == 429:
                raise TTSError(
                    "ElevenLabs: превышен лимит/квота (HTTP 429). "
                    "Подождите или проверьте тарифный план."
                ) from None
            if code == 403:
                raise TTSError(
                    "ElevenLabs: доступ запрещён (HTTP 403) — ключу не хватает "
                    "прав или голос недоступен на этом плане."
                ) from None
            raise TTSError(
                f"ElevenLabs: HTTP {code or '?'} {reason}".strip()
            ) from None
        except urllib.error.URLError as exc:
            reason = _redact(getattr(exc, "reason", exc), config.api_key)
            raise TTSError(f"ElevenLabs: сеть недоступна: {reason}") from None
        except (TimeoutError, OSError) as exc:
            message = _redact(exc, config.api_key)
            raise TTSError(
                f"ElevenLabs: таймаут/сеть за {effective_timeout:g} с: {message}"
            ) from None

        try:
            status = self._response_status(response)
            if status >= 400:
                raise TTSError(f"ElevenLabs: HTTP {status}.")
            try:
                data = response.read()
            except OSError as exc:
                raise TTSError(
                    "ElevenLabs: обрыв сети при чтении аудио: "
                    f"{_redact(exc, config.api_key)}"
                ) from None
            header_type = self._response_content_type(response)
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()

        if not data:
            raise TTSError("ElevenLabs вернул пустой аудиоответ.")
        content_type = (
            header_type
            if header_type.startswith("audio/")
            else _content_type_for_format(config.output_format)
        )
        self.last_content_type = content_type
        return bytes(data), content_type

    # -- синтез -------------------------------------------------------------

    def synthesize_audio(
        self,
        text: str,
        *,
        voice: str | None = None,
        timeout: float | None = None,
    ) -> tuple[bytes, str]:
        """Синтезировать фразу в память: ``(аудиобайты, content_type)``.

        Успешный ответ кладётся в общий LRU-кэш (если ``ELEVENLABS_CACHE!=0``):
        повторная фраза не тратит квоту и не идёт в сеть.
        """
        clean = _clean_text(text)
        config = self.resolve()
        voice_id = self._voice_for(voice)
        key = (clean, voice_id, config.model_id, config.output_format)
        if self._use_cache:
            cached = _cache_get(key)
            if cached is not None:
                self.last_content_type = cached[1]
                return cached
        data, content_type = self._fetch_audio(
            clean, voice=voice_id, timeout=timeout
        )
        if self._use_cache:
            _cache_put(key, (data, content_type))
        return data, content_type

    def synthesize(
        self,
        text: str,
        out_path: str | Path | None = None,
        *,
        rate: int | None = None,
        voice: str | None = None,
        timeout: float | None = None,
    ) -> tuple[bytes, str] | Path:
        """Синтезировать: без пути — ``(bytes, content_type)``, с путём — ``Path``."""
        data, content_type = self.synthesize_audio(
            text, voice=voice, timeout=timeout
        )
        if out_path is None:
            return data, content_type
        target = _prepare_target(out_path)
        try:
            target.write_bytes(data)
        except OSError as exc:
            raise TTSError(f"не удалось записать аудио в {target}: {exc}") from exc
        return target

    def synthesize_timed(
        self,
        text: str,
        out_path: str | Path,
        *,
        rate: int | None = None,
        voice: str | None = None,
        timeout: float | None = None,
    ) -> SynthesisResult:
        """Синтезировать в файл и вернуть замеры (для ``voice_loop.py``).

        В файл пишется MP3 (имя может быть ``*.wav`` — формат определяется по
        содержимому). ``first_fragment_ms`` равен полному времени: облако
        отдаёт ответ целиком, без потоковой отдачи первых байт.
        """
        started = time.perf_counter()
        target = self.synthesize(text, out_path, voice=voice, timeout=timeout)
        synth_ms = (time.perf_counter() - started) * 1000.0
        assert isinstance(target, Path)
        return SynthesisResult(
            path=target,
            synth_ms=synth_ms,
            first_fragment_ms=synth_ms,
            engine=self.name,
        )

    def build_command(
        self,
        text: str,
        out_path: str | Path,
        *,
        rate: int | None = None,
        voice: str | None = None,
    ) -> tuple[list[str], str | None]:
        """У ElevenLabs нет локального процесса: команда не нужна."""
        raise TTSUnavailable(
            "ElevenLabs работает через HTTP, а не через локальный процесс."
        )


# --------------------------------------------------------------------------- #
# системный say / espeak
# --------------------------------------------------------------------------- #
def find_say_binary(
    explicit: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> Path | None:
    """Найти ``say`` (macOS): аргумент → ``EVA_SAY_BIN``/``SAY_BIN`` → PATH."""
    if explicit:
        path = Path(explicit).expanduser()
        if not _is_executable(path):
            raise TTSUnavailable(
                f"say не найден: {path}. Укажите --say-bin или уберите флаг."
            )
        return path
    raw = _env(env, "EVA_SAY_BIN", "SAY_BIN")
    if raw:
        path = Path(raw).expanduser()
        if _is_executable(path):
            return path
    found = shutil.which("say")
    return Path(found) if found else None


def find_espeak_binary(
    explicit: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> Path | None:
    """Найти ``espeak-ng``/``espeak`` (Linux): аргумент → env → PATH."""
    if explicit:
        path = Path(explicit).expanduser()
        if not _is_executable(path):
            raise TTSUnavailable(
                f"espeak не найден: {path}. Уберите --say-bin или укажите "
                "рабочий бинарь."
            )
        return path
    raw = _env(env, "EVA_ESPEAK_BIN", "ESPEAK_BIN")
    if raw:
        path = Path(raw).expanduser()
        if _is_executable(path):
            return path
    for name in ESPEAK_BINARY_NAMES:
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


def build_say_command(
    say_bin: str | Path,
    voice: str,
    out_path: str | Path,
    text: str,
    *,
    rate: int | None = None,
) -> list[str]:
    """Команда ``say``, которая пишет WAV 16 кГц/моно/16 бит."""
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


def system_help(say: Path | None, espeak: Path | None) -> str:
    """Человеческое сообщение: системный TTS не найден."""
    lines = [
        "Системный TTS недоступен: say (macOS) и espeak/espeak-ng (Linux) "
        "не найдены.",
        "macOS: say есть в системе, но может отсутствовать голос Milena — "
        "проверьте `say -v '?' | grep Milena`.",
        "Linux: поставьте espeak-ng и укажите EVA_ESPEAK_BIN при "
        "нестандартном пути.",
    ]
    if say is not None:
        lines.append(f"say: {say}")
    if espeak is not None:
        lines.append(f"espeak: {espeak}")
    return "\n".join(lines)


def _looks_like_lang(voice: str) -> bool:
    """Похоже ли имя голоса на языковой код espeak (``ru``, ``en-us``, ``ru+f3``)."""
    head = voice.split("+", 1)[0]
    if not 2 <= len(head) <= 8:
        return False
    if not head.replace("-", "").isascii() or not head.replace("-", "").isalpha():
        return False
    return head.islower() or head.isupper()


class SystemTTS(TTSAdapter):
    """Системный TTS: macOS ``say`` или Linux ``espeak``/``espeak-ng``.

    ``say`` пишет WAV 16 кГц/моно/16 бит через ``-o`` и ``--data-format``;
    ``espeak`` — через ``-w`` и ``--stdin``. Оба варианта синтезируют в файл,
    поэтому замер первого фрагмента одинаковый.
    """

    name = "system"

    def __init__(
        self,
        binary: str | Path | None = None,
        *,
        kind: str | None = None,
        voice: str | None = None,
        lang: str = DEFAULT_TTS_LANG,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._binary = binary
        self._kind = kind
        self._voice = voice
        self._lang = lang or DEFAULT_TTS_LANG
        self._env = env
        self._resolved: tuple[Path, str] | None = None

    @staticmethod
    def _infer_kind(binary: Path) -> str:
        return "say" if "say" in binary.name.lower() else "espeak"

    def resolve(self) -> tuple[Path, str]:
        """Найти (бинарь, вид) или поднять :class:`TTSUnavailable`."""
        if self._resolved is not None:
            return self._resolved
        if self._binary:
            path = Path(self._binary).expanduser()
            if not _is_executable(path):
                raise TTSUnavailable(f"системный TTS не найден: {path}")
            kind = self._kind or self._infer_kind(path)
        else:
            say: Path | None = None
            espeak: Path | None = None
            if self._kind in (None, "say", "auto"):
                say = find_say_binary(None, env=self._env)
            if self._kind in (None, "espeak", "auto"):
                espeak = find_espeak_binary(None, env=self._env)
            if self._kind is None:
                # На macOS предпочитаем say, на остальных — espeak.
                if sys.platform == "darwin":
                    path, kind = (say, "say") if say else (espeak, "espeak")
                else:
                    path, kind = (espeak, "espeak") if espeak else (say, "say")
            elif self._kind == "say":
                path, kind = say, "say"
            else:
                path, kind = espeak, "espeak"
            if path is None:
                raise TTSUnavailable(system_help(say, espeak))
        self._resolved = (path, kind)
        return self._resolved

    def available(self) -> bool:
        try:
            self.resolve()
        except TTSUnavailable:
            return False
        return True

    def describe(self) -> str:
        try:
            binary, kind = self.resolve()
        except TTSUnavailable:
            return "system (недоступен: нет say/espeak)"
        return f"system ({kind}: {binary.name})"

    @property
    def fragment_threshold(self) -> int:  # type: ignore[override]
        """``say`` пишет 4-КиБ заголовок, espeak — обычные 44 байта."""
        try:
            _, kind = self.resolve()
        except TTSUnavailable:
            return MIN_WAV_BYTES
        return WAV_HEADER_BYTES if kind == "say" else MIN_WAV_BYTES

    def build_command(
        self,
        text: str,
        out_path: str | Path,
        *,
        rate: int | None = None,
        voice: str | None = None,
    ) -> tuple[list[str], str | None]:
        binary, kind = self.resolve()
        target = Path(out_path).expanduser()
        if kind == "say":
            spoken = voice or self._voice or DEFAULT_SAY_VOICE
            return (
                build_say_command(binary, spoken, target, text, rate=rate),
                None,
            )

        # espeak/espeak-ng: текст — в stdin, WAV — в -w.
        spoken = voice or self._voice
        if not spoken or not _looks_like_lang(spoken):
            spoken = self._lang
        command = [str(binary), "-w", str(target), "-v", spoken]
        if rate is not None:
            command.extend(["-s", str(int(rate))])
        command.append("--stdin")
        return command, text + "\n"


# --------------------------------------------------------------------------- #
# null
# --------------------------------------------------------------------------- #
class NullTTS(TTSAdapter):
    """Заглушка для тестов: синтеза нет, ничего не пишет и не запускает.

    ``synthesize`` возвращает пустые байты. В голосовом цикле null означает
    «озвучивать нечем»: ``voice_loop.py`` такое состояние показывает как
    ошибку TTS, а не как тишину.
    """

    name = "null"

    def available(self) -> bool:
        return False

    def describe(self) -> str:
        return "null (синтез отключён)"

    def build_command(
        self,
        text: str,
        out_path: str | Path,
        *,
        rate: int | None = None,
        voice: str | None = None,
    ) -> tuple[list[str], str | None]:
        return [], None

    def synthesize(
        self,
        text: str,
        out_path: str | Path | None = None,
        *,
        rate: int | None = None,
        voice: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> bytes | Path:
        _clean_text(text)
        if out_path is not None:
            return Path(out_path).expanduser()
        return b""

    def synthesize_timed(
        self,
        text: str,
        out_path: str | Path,
        *,
        rate: int | None = None,
        voice: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> SynthesisResult:
        _clean_text(text)
        return SynthesisResult(
            path=Path(out_path).expanduser(),
            synth_ms=0.0,
            first_fragment_ms=None,
            engine=self.name,
        )


# --------------------------------------------------------------------------- #
# выбор движка
# --------------------------------------------------------------------------- #
def select_engine(
    prefer: str | None = None,
    *,
    piper_bin: str | Path | None = None,
    piper_model: str | Path | None = None,
    say_bin: str | Path | None = None,
    voice: str | None = None,
    lang: str = DEFAULT_TTS_LANG,
    env: Mapping[str, str] | None = None,
) -> TTSAdapter:
    """Выбрать TTS-движок.

    ``prefer``/``EVA_TTS``: ``auto`` (по умолчанию), ``elevenlabs``,
    ``piper``, ``system``, ``null``. Явный выбор перекрывает автовыбор и не
    подменяется: если выбранный движок не настроен,
    :meth:`TTSAdapter.available` вернёт False, а синтез поднимет
    :class:`TTSUnavailable` с подсказкой.

    Авто: ElevenLabs (если задан ``ELEVENLABS_API_KEY``), иначе Piper (если
    есть бинарь **и** модель), иначе SystemTTS, иначе NullTTS. Отсутствие
    ключа или Piper — не ошибка, а причина фолбэка.
    """
    env_source = os.environ if env is None else env
    choice = _env(env_source, "EVA_TTS", "EVA_TTS_ENGINE")
    if prefer is not None and str(prefer).strip():
        choice = str(prefer).strip()
    choice = (choice or "auto").lower()
    if choice not in ENGINE_CHOICES:
        raise TTSError(
            f"неизвестный TTS-движок {choice!r}: допустимо "
            + ", ".join(ENGINE_CHOICES)
        )

    elevenlabs = ElevenLabsTTS(env=env)
    piper = PiperTTS(piper_bin, piper_model, env=env)
    if choice == "elevenlabs":
        return elevenlabs
    if choice == "piper":
        return piper
    if choice == "system":
        return SystemTTS(say_bin, voice=voice, lang=lang, env=env)
    if choice == "null":
        return NullTTS()

    if elevenlabs.available():
        return elevenlabs
    if piper.available():
        return piper
    system = SystemTTS(say_bin, voice=voice, lang=lang, env=env)
    if system.available():
        return system
    return NullTTS()


def describe_engine(engine: TTSAdapter | None) -> str:
    """Строка вида ``piper (бинарь, модель …)`` для отчёта о выборе движка."""
    if engine is None:
        return "none"
    try:
        return engine.describe()
    except Exception:  # noqa: BLE001 — описание не должно ронять вывод
        return engine.name


def _fallback_engine(
    *,
    piper_bin: str | Path | None = None,
    piper_model: str | Path | None = None,
    say_bin: str | Path | None = None,
    voice: str | None = None,
    lang: str = DEFAULT_TTS_LANG,
    env: Mapping[str, str] | None = None,
) -> TTSAdapter | None:
    """Локальный запасной движок после сбоя ElevenLabs: Piper, затем system.

    Возвращает ``None``, если ни Piper, ни say/espeak недоступны: тогда
    вызывающий поднимает исходную ошибку ElevenLabs — молчаливой тишины нет.
    """
    piper = PiperTTS(piper_bin, piper_model, env=env)
    if piper.available():
        return piper
    system = SystemTTS(say_bin, voice=voice, lang=lang, env=env)
    if system.available():
        return system
    return None


def _synthesize_with_fallback(
    text: str,
    out_path: str | Path | None,
    *,
    engine: TTSAdapter | None,
    prefer: str | None,
    piper_bin: str | Path | None,
    piper_model: str | Path | None,
    say_bin: str | Path | None,
    voice: str | None,
    lang: str,
    rate: int | None,
    timeout: float,
    env: Mapping[str, str] | None,
) -> tuple[object, TTSAdapter]:
    """Синтез с откатом ElevenLabs → Piper/system; возвращает результат и движок."""
    env_source = os.environ if env is None else env
    chosen = engine or select_engine(
        prefer,
        piper_bin=piper_bin,
        piper_model=piper_model,
        say_bin=say_bin,
        voice=voice,
        lang=lang,
        env=env,
    )
    try:
        return (
            chosen.synthesize(
                text, out_path, rate=rate, voice=voice, timeout=timeout
            ),
            chosen,
        )
    except TTSError as exc:
        if chosen.name != "elevenlabs":
            raise
        api_key = _env(env_source, "ELEVENLABS_API_KEY")
        fallback = _fallback_engine(
            piper_bin=piper_bin,
            piper_model=piper_model,
            say_bin=say_bin,
            voice=voice,
            lang=lang,
            env=env,
        )
        if fallback is None:
            # Нечего озвучивать локально: показываем исходную причину, а не
            # «тишину». Ключ в текст не попадает (redact на всякий случай).
            raise TTSError(
                f"ElevenLabs: {_redact(exc, api_key)}; запасной Piper/system "
                "тоже недоступен."
            ) from None
        logger.warning(
            "озвучка: ElevenLabs не сработал (%s), откат на %s",
            _redact(exc, api_key),
            fallback.name,
        )
        return (
            fallback.synthesize(
                text, out_path, rate=rate, voice=voice, timeout=timeout
            ),
            fallback,
        )


def synthesize(
    text: str,
    out_path: str | Path | None = None,
    *,
    engine: TTSAdapter | None = None,
    prefer: str | None = None,
    piper_bin: str | Path | None = None,
    piper_model: str | Path | None = None,
    say_bin: str | Path | None = None,
    voice: str | None = None,
    lang: str = DEFAULT_TTS_LANG,
    rate: int | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    env: Mapping[str, str] | None = None,
) -> bytes | Path | tuple[bytes, str]:
    """Синтезировать текст выбранным (или переданным) движком.

    Без ``out_path`` — аудиобайты (Piper/system — WAV, ElevenLabs — пара
    ``(mp3, "audio/mpeg")``); с ``out_path`` — путь. Если Piper нет,
    автовыбор уходит на системный TTS, а при полном отсутствии TTS вернёт
    пустые байты (NullTTS), не падая. Сбой ElevenLabs (401/429/сеть/пусто)
    логируется без ключа и откатывается на Piper/system.
    """
    result, _ = _synthesize_with_fallback(
        text,
        out_path,
        engine=engine,
        prefer=prefer,
        piper_bin=piper_bin,
        piper_model=piper_model,
        say_bin=say_bin,
        voice=voice,
        lang=lang,
        rate=rate,
        timeout=timeout,
        env=env,
    )
    return result  # type: ignore[return-value]


def synthesize_audio(
    text: str,
    *,
    engine: TTSAdapter | None = None,
    prefer: str | None = None,
    piper_bin: str | Path | None = None,
    piper_model: str | Path | None = None,
    say_bin: str | Path | None = None,
    voice: str | None = None,
    lang: str = DEFAULT_TTS_LANG,
    rate: int | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    env: Mapping[str, str] | None = None,
) -> tuple[bytes, str]:
    """Синтезировать в память и вернуть ``(bytes, content_type)``.

    Обёртка над :func:`synthesize` для HTTP-слоя: ElevenLabs отдаёт MP3 и
    ``audio/mpeg``, локальные движки — WAV и ``audio/wav``. Тот же откат:
    при сбое ElevenLabs возвращается аудио Piper, а не ошибка.
    """
    result, chosen = _synthesize_with_fallback(
        text,
        None,
        engine=engine,
        prefer=prefer,
        piper_bin=piper_bin,
        piper_model=piper_model,
        say_bin=say_bin,
        voice=voice,
        lang=lang,
        rate=rate,
        timeout=timeout,
        env=env,
    )
    if isinstance(result, tuple):
        data, content_type = result
        return bytes(data), str(content_type)
    if isinstance(result, (bytes, bytearray)):
        return bytes(result), "audio/wav"
    # Path без out_path не возвращается, но контракт защищаем.
    return b"", "audio/wav"
