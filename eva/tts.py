"""TTS-адаптер EVA32: ElevenLabs → Silero → системный say/espeak → null.

Задачи — [`ai/tasks/0191-elevenlabs-tts.md`](../ai/tasks/0191-elevenlabs-tts.md)
(провайдер), [`ai/tasks/0193-silero-tts.md`](../ai/tasks/0193-silero-tts.md)
(локальный русский нейроголос Silero),
[`ai/tasks/0197-switch-to-silero-only.md`](../ai/tasks/0197-switch-to-silero-only.md)
(единственный локальный голос — Silero). Весь синтез идёт через один
интерфейс, а движок выбирается по доступности:

============== ==========================================================
``elevenlabs`` облачный нейро-TTS: ``POST /v1/text-to-speech/{voice_id}``,
               ключ ``ELEVENLABS_API_KEY``. Возвращает MP3 (по умолчанию
               ``mp3_44100_128``). Текст уходит в облако ElevenLabs.
``silero``     локальный нейро-TTS: отдельный процесс ``python3.13`` с
               ``vendor/torch-py`` (torch) и моделью v4_ru в
               ``vendor/silero``; WAV 48 кГц. Ставится
               ``tools/install_silero.sh``. Голос — ``SILERO_SPEAKER``
               (``baya`` по умолчанию).
``system``     macOS ``say`` (голос Milena) или Linux ``espeak``/``espeak-ng``.
``null``       заглушка: синтеза нет, ничего не пишет, ничего не играет.
============== ==========================================================

Автовыбор: **ElevenLabs, если задан ключ**, иначе **Silero, если установлен**
(есть python3.13, vendor/torch-py и модель), иначе **SystemTTS**, иначе
**NullTTS**. Piper из автовыбора убран (задача 0197): адаптер
:class:`PiperTTS` остался только для явного ``EVA_TTS=piper`` и в инструкциях
не упоминается. Явный выбор (``EVA_TTS=elevenlabs|silero|system|null`` /
``prefer=``) перекрывает автовыбор — даже если движок недоступен: тогда при
синтезе будет понятная :class:`TTSUnavailable`, а не молчаливая подмена.
Исключение — облачный ElevenLabs и локальный Silero: если движок не сработал
(401/429/сеть/пусто или упал процесс/нет torch), модуль логирует причину
(без ключа!) и **сам откатывается на следующий локальный движок**
(Silero → system), чтобы пользователь не остался без голоса.

Silero работает отдельным **долгоживущим** процессом
``tools/silero_worker.py`` (задача 0195): torch импортируется только внутри
него, поэтому память чат-сервера не растёт на ~300 МБ, а модель грузится
**один раз** — при первой фразе, а не на каждую (как было в 0193 с
одноразовым ``tools/silero_speak.py``). Worker поднимается лениво, живёт до
конца процесса чата (keep-alive, ``atexit``), запросы сериализуются lock'ом,
при падении перезапускается не больше одного раза, при таймауте (20 с)
убивается и уступает место системному TTS. Повторные одинаковые фразы
по-прежнему отдаются из LRU-кэша адаптера. Замеры — в
:meth:`SileroTTS.describe` и :func:`silero_worker_info`.

:func:`prewarm_silero_worker` поднимает worker заранее (фоном): ленивый старт
стоит ~1,5–2 с на импорт torch и модель, и первая фраза после старта чата
укладывается в 2 с только с prewarm.

Контракт синтеза — ``synthesize(text) -> bytes | Path | (bytes, content_type)``::

    from eva.tts import synthesize, synthesize_audio

    wav_bytes = synthesize("Сколько стоит тариф Старт")        # bytes (Silero)
    mp3, mime = synthesize_audio("Привет")                      # ElevenLabs
    path = synthesize("Привет", "/tmp/reply.wav")               # Path

:class:`ElevenLabsTTS` дополнительно отдаёт пару ``(bytes, content_type)``
через :meth:`ElevenLabsTTS.synthesize_audio`; серверу это нужно, чтобы
выставить ``audio/mpeg`` вместо жёсткого ``audio/wav``.

Перед синтезом текст проговаривается словами: числа — :mod:`eva.ru_numbers`
(задача 0205), знаки, сокращения, телефоны и email — :mod:`eva.ru_speech`
(задача 0206). Silero v4_ru не произносит токены-цифры и молча пропускает
«×», «€», «км²», «#», а ``info@example.com`` выпадает целиком; «Цена 1500
рублей» звучала втрое короче словесной формы. Нормализация общая для всех
движков и делается до выбора движка, поэтому ElevenLabs и ``say`` тоже
получают слова.

Для Silero к этому добавляются **ударения** (задача 0222): модель понимает
знак ``+`` перед ударной гласной («Колоб+ок»), но по умолчанию ошибается на
сказочных словах и именах. :meth:`SileroTTS._prepare_text` применяет
:func:`eva.ru_stress.apply_stress` только на пути в движок — текст ответа
пользователю остаётся без плюсов. Откат — ``EVA_STRESS=0``; словарь можно
расширять в ``data/stress.txt`` (``слово<TAB>разметка``).

Адаптеры умеют также :meth:`TTSAdapter.synthesize_timed` — с замерами
``synth_ms`` и ``first_fragment_ms`` для голосового цикла.

Сеть используется только ElevenLabs (и только при заданном ключе); ничего не
скачивается во время синтеза. Сам модуль — только стандартная библиотека:
torch живёт в отдельном процессе Silero, реальный звук не воспроизводится
(это делает ``voice_loop.py`` через afplay).
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import queue
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
from typing import Any, Mapping, Sequence

from .ru_speech import speech_to_words
from .ru_stress import apply_stress

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ = [
    "DEFAULT_ELEVENLABS_FORMAT",
    "DEFAULT_ELEVENLABS_MODEL",
    "DEFAULT_ELEVENLABS_TIMEOUT",
    "DEFAULT_ELEVENLABS_VOICE_ID",
    "DEFAULT_SAY_VOICE",
    "DEFAULT_SILERO_SPEAKER",
    "DEFAULT_SILERO_WORKER_START_TIMEOUT",
    "DEFAULT_SILERO_WORKER_TIMEOUT",
    "DEFAULT_TIMEOUT",
    "DEFAULT_TTS_LANG",
    "ELEVENLABS_API_URL",
    "ELEVENLABS_CACHE_SIZE",
    "ElevenLabsTTS",
    "ENGINE_CHOICES",
    "NullTTS",
    "PiperTTS",
    "SileroTTS",
    "SileroWorker",
    "SystemTTS",
    "SynthesisResult",
    "TTSAdapter",
    "TTSError",
    "TTSUnavailable",
    "build_say_command",
    "clear_elevenlabs_cache",
    "clear_silero_cache",
    "describe_engine",
    "elevenlabs_cache_info",
    "find_espeak_binary",
    "find_piper_binary",
    "find_piper_model",
    "find_say_binary",
    "find_silero_model_dir",
    "find_silero_python",
    "find_silero_torch_dir",
    "find_silero_worker",
    "normalize_tts_text",
    "piper_help",
    "prewarm_silero_worker",
    "select_engine",
    "silero_cache_info",
    "silero_help",
    "silero_worker_info",
    "stop_silero_worker",
    "synthesize",
    "synthesize_audio",
    "system_help",
]


#: Корень репозитория: ``eva/tts.py`` → ``parents[1]``. Нужен для поиска
#: вендорного torch и модели Silero (``vendor/``).
ROOT = Path(__file__).resolve().parents[1]

#: Таймаут одного синтеза по умолчанию, сек.
DEFAULT_TIMEOUT = 120.0

#: Голос macOS ``say`` по умолчанию (ru_RU).
DEFAULT_SAY_VOICE = os.environ.get("EVA_VOICE", "Milena").strip() or "Milena"

#: Язык системного TTS (espeak).
DEFAULT_TTS_LANG = os.environ.get("EVA_TTS_LANG", "ru").strip() or "ru"

#: Допустимые значения ``--tts``/``EVA_TTS`` (``auto`` не движок, а «выбери
#: сам»). ``piper`` оставлен только для явного выбора устаревшего адаптера: в
#: автовыборе его нет и в инструкциях он не упоминается (задача 0197).
ENGINE_CHOICES = ("auto", "elevenlabs", "silero", "piper", "system", "null")

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
#: локального Silero, но сеть бывает медленной; 30 с — с большим запасом.
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
# Silero (задача 0193)
# --------------------------------------------------------------------------- #
#: Голос Silero v4_ru по умолчанию: женский, ближе всего к EVA. Другие —
#: ``kseniya``, ``xenia``, ``aidar`` (мужской), ``eugene``, ``random``.
DEFAULT_SILERO_SPEAKER = os.environ.get("SILERO_SPEAKER", "baya").strip() or "baya"

#: Голоса v4_ru, которые понимает модель. Страница чата и EVA.app дают
#: выбрать ``baya``/``kseniya``/``xenia``/``aidar``; остальные доступны через
#: ``SILERO_SPEAKER``.
SILERO_SPEAKERS = ("aidar", "baya", "eugene", "kseniya", "xenia", "random")

#: Частота Silero v4_ru: 48 кГц — лучшее качество модели; Web Audio и AVPlayer
#: играют такой WAV без пересэмплинга.
SILERO_SAMPLE_RATE = 48000

#: Каталог вендорного torch (``tools/install_silero.sh``). В ``.pylibs``
#: лежит сломанный source-checkout, поэтому подставляем только этот путь.
SILERO_TORCH_DIR = "vendor/torch-py"

#: Кэш модели и репозитория torch.hub: ``TORCH_HOME=vendor/silero``.
SILERO_MODEL_DIR = "vendor/silero"

#: Скрипт одноразового синтеза: читает текст из stdin, пишет WAV в argv[1].
#: Больше не используется по умолчанию (его место занял worker), но остаётся
#: для ``SILERO_WORKER=0`` и ручных прогонов ``tools/install_silero.sh``.
SILERO_SCRIPT = "tools/silero_speak.py"

#: Долгоживущий worker (задача 0195): модель грузится один раз, дальше — фразы.
SILERO_WORKER_SCRIPT = "tools/silero_worker.py"

#: Таймаут одного запроса к worker'у, сек. Дольше — считаем, что он завис:
#: фраза ждать не может, worker убивается, синтез уходит на системный TTS.
DEFAULT_SILERO_WORKER_TIMEOUT = 20.0

#: Таймаут старта worker'а, сек (импорт torch + загрузка модели v4_ru).
#: Холодный старт ≈1,5–2,5 с; 30 с — с запасом на медленный диск.
DEFAULT_SILERO_WORKER_START_TIMEOUT = 30.0

#: Сколько раз перезапускать worker в пределах одной фразы при падении.
#: 1 — «не более одного раза» из задачи 0195: второй сбой = откат на system.
SILERO_WORKER_RESTARTS = 1

#: Python для Silero. 3.13 — единственный с колёсами torch на arm64 macOS;
#: python3.14 их пока не имеет.
SILERO_PYTHON_NAMES = ("python3.13", "/opt/homebrew/bin/python3.13")

#: Сколько последних фраз держать в LRU-кэше адаптера. Одна фраза — WAV
#: 48 кГц/16 бит (~190 КБ/с); 16 записей — несколько мегабайт, зато повторные
#: «стоп»/«повтори» не платят за холодный процесс.
SILERO_CACHE_SIZE = 16


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


def normalize_tts_text(text: object) -> str:
    """Текст для озвучки: знаки и числа — русскими словами (0205/0206).

    Silero v4_ru пропускает токены-цифры («Цена 1500 рублей» звучала втрое
    короче словесной формы) и молча выкидывает знаки («2 × 3», «50 €»,
    «info@example.com»). Поэтому нормализация общая для всех движков: она
    делается **до** выбора движка, в :func:`_synthesize_with_fallback`, и
    повторно в :func:`_clean_text` — на случай прямого вызова
    ``adapter.synthesize()``. Ступени: :mod:`eva.ru_speech` (символы,
    сокращения, телефоны, email) и :mod:`eva.ru_numbers` (числа). Функция
    идемпотентна: слова повторно не меняются.
    """
    return speech_to_words(str(text or ""))


def _clean_text(text: object) -> str:
    value = normalize_tts_text(text).strip()
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


def _env_float(
    env: Mapping[str, str] | None, name: str, default: float
) -> float:
    """Число из окружения (``SILERO_WORKER_TIMEOUT``); мусор → ``default``."""
    raw = _env(env, name)
    if raw is None:
        return float(default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float(default)
    return value if value > 0 else float(default)


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
    env: Mapping[str, str] | None = None,
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
                env=dict(env) if env is not None else None,
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

    def subprocess_env(self) -> dict[str, str] | None:
        """Окружение дочернего процесса (``None`` — унаследовать текущее).

        Нужно Silero: torch и модель должны искаться в ``vendor/``, а не в
        случайном ``.pylibs``/site-packages.
        """
        return None

    def _prepare_text(self, text: object) -> str:
        """Текст, который уходит в движок: чистка + правки конкретного движка.

        Базовая реализация — только :func:`_clean_text` (слова вместо знаков и
        чисел). :class:`SileroTTS` переопределяет метод и добавляет ударения:
        ``+`` понимает лишь Silero, остальным движкам он был бы мусором.
        """
        return _clean_text(text)

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

        Чистка и правки движка делаются ровно один раз — в
        :meth:`synthesize_timed`: повторная нормализация уже размеченного
        текста испортила бы знак ударения (``ru_speech`` читает ``+`` как
        «плюс»).
        """
        if out_path is not None:
            target = _prepare_target(out_path)
            self.synthesize_timed(text, target, rate=rate, voice=voice, timeout=timeout)
            return target
        with tempfile.TemporaryDirectory(prefix="eva-tts-") as tmp:
            target = Path(tmp) / "reply.wav"
            self.synthesize_timed(text, target, rate=rate, voice=voice, timeout=timeout)
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
        clean = self._prepare_text(text)
        return self._synthesize_prepared(
            clean, out_path, rate=rate, voice=voice, timeout=timeout
        )

    def _synthesize_prepared(
        self,
        text: str,
        out_path: str | Path,
        *,
        rate: int | None = None,
        voice: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> SynthesisResult:
        """Синтезировать **уже подготовленный** текст (без чистки и разметки).

        Отделено от :meth:`synthesize_timed`, чтобы Silero мог пройти путь
        «чистка → ударения» ровно один раз: ``+`` перед синтезом уже стоит, а
        повторная нормализация прочитала бы его как слово «плюс».
        """
        target = Path(out_path).expanduser()
        command, stdin_text = self.build_command(
            text, target, rate=rate, voice=voice
        )
        synth_ms, first_fragment_ms = _run_to_file(
            command,
            stdin_text,
            target,
            timeout=timeout,
            label=self.name,
            fragment_threshold=self.fragment_threshold,
            env=self.subprocess_env(),
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
# Silero (локальный русский нейро-TTS)
# --------------------------------------------------------------------------- #
def _silero_autodiscovery_allowed(env: Mapping[str, str] | None) -> bool:
    """Искать ли вендорный Silero, если окружение передано явно.

    ``env=None``/``os.environ`` — боевой путь: ``vendor/torch-py`` и
    ``vendor/silero`` подхватываются сами. Изолированный словарь без
    ``SILERO_*`` (тесты, встраивание) считается явной конфигурацией: чужой
    установленный Silero в неё не подмешивается.
    """
    if env is None or env is os.environ:
        return True
    if not isinstance(env, dict):
        return True
    return any(str(key).upper().startswith("SILERO_") for key in env)


def find_silero_python(
    explicit: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> Path | None:
    """Найти python3.13 для Silero: аргумент → ``SILERO_PYTHON`` → PATH.

    Явный путь обязан существовать: опечатка в конфиге не должна молча
    менять движок. Если автопоиск не разрешён (изолированный ``env``),
    возвращается ``None``.
    """
    if explicit:
        path = Path(explicit).expanduser()
        if not _is_executable(path):
            raise TTSUnavailable(
                f"Silero: python не найден или не исполняемый: {path}. "
                "Укажите SILERO_PYTHON=/opt/homebrew/bin/python3.13."
            )
        return path

    raw = _env(env, "SILERO_PYTHON")
    if raw:
        path = Path(raw).expanduser()
        if _is_executable(path):
            return path
        raise TTSUnavailable(
            f"SILERO_PYTHON={path}: файл не найден или не исполняемый. "
            "Нужен python3.13 (у python3.14 нет колёс torch)."
        )

    if not _silero_autodiscovery_allowed(env):
        return None
    for name in SILERO_PYTHON_NAMES:
        if os.sep in name:
            path = Path(name).expanduser()
            if _is_executable(path):
                return path
        else:
            found = shutil.which(name)
            if found:
                return Path(found)
    return None


def _has_vendored_torch(path: Path) -> bool:
    """Есть ли по пути рабочий (распакованный) пакет torch."""
    return (path / "torch" / "__init__.py").is_file()


def find_silero_torch_dir(
    explicit: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> Path | None:
    """Найти вендорный torch: аргумент → ``SILERO_TORCH_DIR`` → vendor/torch-py."""
    if explicit:
        path = Path(explicit).expanduser()
        if not _has_vendored_torch(path):
            raise TTSUnavailable(
                f"Silero: в {path} нет пакета torch. Запустите "
                "sh tools/install_silero.sh."
            )
        return path

    raw = _env(env, "SILERO_TORCH_DIR")
    if raw:
        path = Path(raw).expanduser()
        if _has_vendored_torch(path):
            return path
        raise TTSUnavailable(
            f"SILERO_TORCH_DIR={path}: torch не найден. Запустите "
            "sh tools/install_silero.sh."
        )

    if not _silero_autodiscovery_allowed(env):
        return None
    path = ROOT / SILERO_TORCH_DIR
    return path if _has_vendored_torch(path) else None


def _silero_weights(model_dir: Path) -> Path | None:
    """Файл весов v4_ru в кэше torch.hub (или ``None``).

    ``torch.hub`` кладёт пакет модели в
    ``$TORCH_HOME/hub/snakers4_silero-models_master/src/silero/model``;
    старые версии — в ``hub/checkpoints``. Проверяем оба места, чтобы
    «папка есть, а модели нет» не считалась установкой.
    """
    candidates: list[Path] = []
    package = (
        model_dir
        / "hub"
        / "snakers4_silero-models_master"
        / "src"
        / "silero"
        / "model"
    )
    try:
        candidates.extend(sorted(package.glob("v4_ru*.pt")))
        candidates.extend(sorted((model_dir / "hub" / "checkpoints").glob("*.pt")))
    except OSError:
        return None
    for path in candidates:
        try:
            if path.is_file() and path.stat().st_size > 1_000_000:
                return path
        except OSError:
            continue
    return None


def find_silero_model_dir(
    explicit: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> Path | None:
    """Найти кэш модели: аргумент → ``SILERO_MODEL_DIR`` → vendor/silero.

    Каталог считается готовым, только если в нём есть веса v4_ru: пустой
    ``vendor/silero`` — это «не установлено», а не рабочий движок.
    """
    if explicit:
        path = Path(explicit).expanduser()
        if _silero_weights(path) is None:
            raise TTSUnavailable(
                f"Silero: модель v4_ru не найдена в {path}. Запустите "
                "sh tools/install_silero.sh (первый раз нужна сеть)."
            )
        return path

    raw = _env(env, "SILERO_MODEL_DIR")
    if raw:
        path = Path(raw).expanduser()
        if _silero_weights(path) is not None:
            return path
        raise TTSUnavailable(
            f"SILERO_MODEL_DIR={path}: модель v4_ru не найдена. Запустите "
            "sh tools/install_silero.sh."
        )

    if not _silero_autodiscovery_allowed(env):
        return None
    path = ROOT / SILERO_MODEL_DIR
    return path if _silero_weights(path) is not None else None


def silero_help(
    python: Path | None = None,
    torch_dir: Path | None = None,
    model_dir: Path | None = None,
) -> str:
    """Человеческое сообщение: чего не хватает и как поставить Silero."""
    missing: list[str] = []
    if python is None:
        missing.append("python3.13")
    if torch_dir is None:
        missing.append("torch в vendor/torch-py")
    if model_dir is None:
        missing.append("модель v4_ru в vendor/silero")
    lines = [
        "Silero недоступен: нет " + ", ".join(missing) + ".",
        "Установите (один раз, нужна сеть):",
        "  sh tools/install_silero.sh",
        "Без Silero работает системный say/espeak; принудительно: EVA_TTS=system.",
    ]
    if python is not None:
        lines.append(f"python: {python}")
    if torch_dir is not None:
        lines.append(f"torch: {torch_dir}")
    if model_dir is not None:
        lines.append(f"модель: {model_dir}")
    return "\n".join(lines)


@dataclass(frozen=True)
class _SileroRuntime:
    """Разрешённые пути и голос для запуска синтезатора."""

    python: Path
    torch_dir: Path
    model_dir: Path
    script: Path
    speaker: str
    weights: Path


#: LRU-кэш WAV по ``(текст, голос, частота)``: повторная фраза не платит за
#: новый процесс torch. Живёт на уровне модуля, потому что ``select_engine``
#: создаёт адаптер на каждый запрос.
_SILERO_CACHE: "OrderedDict[tuple[str, str, int], bytes]" = OrderedDict()
_SILERO_CACHE_LOCK = threading.Lock()
_SILERO_CACHE_HITS = 0
_SILERO_CACHE_MISSES = 0

#: Время первого реального синтеза (torch + модель), мс. Показывается в
#: ``describe()`` как честная цена холодного старта.
_SILERO_WARMUP_MS: float | None = None
def clear_silero_cache() -> None:
    """Очистить LRU-кэш Silero (тесты и смена голоса)."""
    global _SILERO_CACHE_HITS, _SILERO_CACHE_MISSES
    with _SILERO_CACHE_LOCK:
        _SILERO_CACHE.clear()
        _SILERO_CACHE_HITS = 0
        _SILERO_CACHE_MISSES = 0


def silero_cache_info() -> dict[str, int]:
    """Состояние кэша Silero: размер, попадания/промахи (диагностика)."""
    with _SILERO_CACHE_LOCK:
        return {
            "size": len(_SILERO_CACHE),
            "hits": _SILERO_CACHE_HITS,
            "misses": _SILERO_CACHE_MISSES,
            "max_size": SILERO_CACHE_SIZE,
        }


def _silero_cache_get(key: tuple[str, str, int]) -> bytes | None:
    global _SILERO_CACHE_HITS, _SILERO_CACHE_MISSES
    with _SILERO_CACHE_LOCK:
        value = _SILERO_CACHE.get(key)
        if value is None:
            _SILERO_CACHE_MISSES += 1
            return None
        _SILERO_CACHE.move_to_end(key)
        _SILERO_CACHE_HITS += 1
        return value


def _silero_cache_put(key: tuple[str, str, int], value: bytes) -> None:
    with _SILERO_CACHE_LOCK:
        _SILERO_CACHE[key] = value
        _SILERO_CACHE.move_to_end(key)
        while len(_SILERO_CACHE) > SILERO_CACHE_SIZE:
            _SILERO_CACHE.popitem(last=False)


# --------------------------------------------------------------------------- #
# Silero worker: долгоживущий процесс (задача 0195)
# --------------------------------------------------------------------------- #
class _WorkerFailure(TTSError):
    """Транспорт worker'а сломался: не ответил, умер, не поднялся."""


class _WorkerGone(_WorkerFailure):
    """Worker закрыл stdout/умер между запросами — помогает перезапуск."""


class _WorkerTimeout(_WorkerFailure):
    """Worker не ответил за отведённое время — его надо убить и начать заново."""


def _worker_reader(stream, sink: "queue.Queue[dict[str, Any]]") -> None:
    """Читать stdout worker'а построчно и складывать JSON в очередь.

    Отдельный поток нужен из-за таймаута: ``readline`` на пайпе блокирующий,
    а ждать ответа надо с ограничением по времени. Поток живёт столько же,
    сколько процесс: на EOF кладёт сентинел ``{"eof": True}``.
    """
    try:
        for line in stream:
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                # Не наш вывод (например, чужое предупреждение в stdout).
                continue
            if isinstance(payload, dict):
                sink.put(payload)
    except (OSError, ValueError):
        pass
    finally:
        sink.put({"ok": False, "eof": True})


class SileroWorker:
    """Один процесс ``tools/silero_worker.py``: модель загружена один раз.

    Протокол — JSON-строка на строку (stdin → stdout), см. докстринг
    ``tools/silero_worker.py``. Запросы обязаны сериализоваться
    (:data:`_SILERO_WORKER_LOCK`): «запрос → ответ» не помечен id, и два
    параллельных запроса перепутали бы ответы.

    Класс ничего не знает про кэш и откаты: это делает :class:`SileroTTS`.
    """

    name = "silero"

    def __init__(
        self,
        python: str | Path,
        script: str | Path,
        *,
        speaker: str = DEFAULT_SILERO_SPEAKER,
        sample_rate: int = SILERO_SAMPLE_RATE,
        env: Mapping[str, str] | None = None,
        timeout: float = DEFAULT_SILERO_WORKER_TIMEOUT,
        start_timeout: float = DEFAULT_SILERO_WORKER_START_TIMEOUT,
    ) -> None:
        self._python = Path(python)
        self._script = Path(script)
        self._speaker = _silero_safe_speaker(speaker)
        self._sample_rate = int(sample_rate)
        self._env = dict(env) if env is not None else None
        self.timeout = float(timeout)
        self.start_timeout = float(start_timeout)
        self._proc: subprocess.Popen | None = None
        self._sink: "queue.Queue[dict[str, Any]]" | None = None
        self._stderr = None
        self._reader: threading.Thread | None = None
        self._load_ms: float | None = None
        self._requests = 0

    # -- состояние ----------------------------------------------------------

    @property
    def pid(self) -> int | None:
        """PID живого worker'а (``None`` — не запущен/уже умер)."""
        proc = self._proc
        if proc is None:
            return None
        return proc.pid if proc.poll() is None else None

    def alive(self) -> bool:
        return self.pid is not None

    @property
    def load_ms(self) -> float | None:
        """Сколько заняла загрузка модели в worker'е (мс), если известна."""
        return self._load_ms

    @property
    def requests(self) -> int:
        """Сколько фраз синтезировал этот worker."""
        return self._requests

    def stderr_tail(self, *, lines: int = 3) -> str:
        """Последние строки stderr worker'а — причина падения без догадок."""
        handle = self._stderr
        if handle is None:
            return ""
        try:
            handle.flush()
            handle.seek(0)
            data = handle.read()
        except (OSError, ValueError):
            return ""
        if isinstance(data, bytes):
            data = data.decode("utf-8", "replace")
        rows = [row.strip() for row in str(data).strip().splitlines() if row.strip()]
        return " | ".join(rows[-lines:]) if rows else ""

    # -- жизненный цикл -----------------------------------------------------

    def start(self) -> None:
        """Запустить процесс и дождаться ``{"ready": true}`` от модели."""
        if self.alive():
            return
        self.stop()
        command = [
            str(self._python),
            str(self._script),
            "--speaker",
            _silero_safe_speaker(self._speaker),
            "--sample-rate",
            str(self._sample_rate),
        ]
        stderr_file = tempfile.TemporaryFile()
        try:
            proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                # stderr — в файл, не в PIPE: torch пишет предупреждения, и
                # незабранный PIPE рано или поздно заблокировал бы worker.
                stderr=stderr_file,
                text=True,
                encoding="utf-8",
                bufsize=1,
                env=dict(self._env) if self._env is not None else None,
            )
        except OSError as exc:
            stderr_file.close()
            raise TTSUnavailable(
                f"silero: worker не запустился ({command[0]}): {exc}"
            ) from None

        sink: "queue.Queue[dict[str, Any]]" = queue.Queue()
        reader = threading.Thread(
            target=_worker_reader,
            args=(proc.stdout, sink),
            name=f"silero-worker-{proc.pid}",
            daemon=True,
        )
        reader.start()
        self._proc = proc
        self._sink = sink
        self._stderr = stderr_file
        self._reader = reader
        self._load_ms = None
        self._requests = 0

        try:
            ready = self._await(self.start_timeout, what="готовности")
        except _WorkerFailure:
            self.stop()
            raise
        if not ready.get("ready"):
            self.stop()
            raise _WorkerFailure(
                "silero: worker не подтвердил готовность: "
                f"{ready.get('error') or ready!r}"
            )
        try:
            self._load_ms = float(ready.get("load_ms") or 0.0) or None
        except (TypeError, ValueError):
            self._load_ms = None
        logger.info(
            "silero: worker поднят (pid %s, torch %s: импорт %.0f мс, "
            "модель %.0f мс, всего %.0f мс)",
            proc.pid,
            ready.get("torch"),
            float(ready.get("import_ms") or 0.0),
            float(ready.get("model_ms") or 0.0),
            self._load_ms or 0.0,
        )

    def stop(self, *, grace: float = 1.0) -> None:
        """Попросить worker закрыться (``shutdown`` → SIGTERM → SIGKILL)."""
        proc = self._proc
        self._proc = None
        self._sink = None
        if proc is not None:
            try:
                if proc.poll() is None:
                    try:
                        if proc.stdin is not None:
                            proc.stdin.write('{"cmd": "shutdown"}\n')
                            proc.stdin.flush()
                    except (BrokenPipeError, OSError, ValueError):
                        pass
                    try:
                        proc.wait(timeout=grace)
                    except subprocess.TimeoutExpired:
                        proc.terminate()
                        try:
                            proc.wait(timeout=grace)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            try:
                                proc.wait(timeout=grace)
                            except subprocess.TimeoutExpired:
                                pass
            finally:
                for stream in (proc.stdin, proc.stdout):
                    try:
                        if stream is not None:
                            stream.close()
                    except (OSError, ValueError):
                        pass
                handle, self._stderr = self._stderr, None
                if handle is not None:
                    try:
                        handle.close()
                    except OSError:
                        pass

    # -- запросы ------------------------------------------------------------

    def _await(self, limit: float, *, what: str = "ответа") -> dict[str, Any]:
        """Взять один ответ из очереди с таймаутом."""
        sink = self._sink
        if sink is None:
            raise _WorkerGone("silero: worker не запущен")
        try:
            reply = sink.get(timeout=max(0.001, float(limit)))
        except queue.Empty:
            raise _WorkerTimeout(
                f"worker не ответил ({what}) за {float(limit):g} с"
            ) from None
        if not isinstance(reply, dict):
            raise _WorkerFailure("silero: worker вернул не JSON-объект")
        if reply.get("eof"):
            tail = self.stderr_tail()
            detail = f": {tail}" if tail else ""
            raise _WorkerGone(f"worker завершился (EOF){detail}")
        return reply

    def request(
        self, payload: Mapping[str, Any], *, timeout: float | None = None
    ) -> dict[str, Any]:
        """Отправить один запрос и дождаться одного ответа."""
        proc = self._proc
        if proc is None or proc.poll() is not None:
            raise _WorkerGone("silero: worker не запущен")
        line = json.dumps(dict(payload), ensure_ascii=False) + "\n"
        try:
            if proc.stdin is None:
                raise _WorkerGone("silero: у worker'а нет stdin")
            proc.stdin.write(line)
            proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise _WorkerGone(f"worker не принял запрос: {exc}") from None
        limit = self.timeout if timeout is None else float(timeout)
        return self._await(limit)

    def ping(self, *, timeout: float = 5.0) -> dict[str, Any]:
        """Проверка живости на старте: worker отвечает на ``{"cmd": "ping"}``."""
        reply = self.request({"cmd": "ping"}, timeout=timeout)
        if not reply.get("pong"):
            raise _WorkerFailure(
                f"silero: worker не ответил на ping: {reply.get('error') or reply!r}"
            )
        return reply

    def synthesize(
        self,
        text: str,
        speaker: str,
        out_path: str | Path,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Синтезировать фразу в ``out_path``; вернуть ответ worker'а."""
        reply = self.request(
            {"text": text, "speaker": _silero_safe_speaker(speaker), "out": str(out_path)},
            timeout=timeout,
        )
        if not reply.get("ok"):
            raise TTSError(
                f"silero: {reply.get('error') or 'worker вернул ошибку'}"
            )
        self._requests += 1
        return reply


#: Единственный worker на процесс чата: ``select_engine`` создаёт адаптер на
#: каждый запрос, поэтому процесс должен жить на уровне модуля (иначе каждая
#: фраза снова платила бы за torch и модель).
_SILERO_WORKER: "SileroWorker | None" = None
_SILERO_WORKER_KEY: tuple[str, str] | None = None

#: Lock сериализует запросы к worker'у (один запрос за раз) и защищает
#: создание/остановку процесса. RLock: ``prewarm``/``stop`` вызываются
#: вложенно из тех же путей.
_SILERO_WORKER_LOCK = threading.RLock()


def find_silero_worker(
    explicit: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> Path:
    """Найти скрипт worker'а: аргумент → ``SILERO_WORKER_SCRIPT`` → tools/.

    Отсутствие файла — ошибка конфигурации, а не молчаливая подмена движка:
    worker лежит в репозитории, его пропажа означает сломанную установку.
    """
    raw = explicit or _env(env, "SILERO_WORKER_SCRIPT")
    path = Path(raw).expanduser() if raw else ROOT / SILERO_WORKER_SCRIPT
    if not path.is_file():
        raise TTSUnavailable(
            f"Silero: worker не найден: {path}. Ожидается "
            f"{SILERO_WORKER_SCRIPT} в репозитории (путь переопределяет "
            "SILERO_WORKER_SCRIPT)."
        )
    return path


def silero_worker_info() -> dict[str, object]:
    """Состояние worker'а: жив ли, pid, число фраз, время загрузки модели."""
    worker = _SILERO_WORKER
    pid = worker.pid if worker is not None else None
    return {
        "running": pid is not None,
        "pid": pid,
        "requests": worker.requests if worker is not None else 0,
        "load_ms": round(worker.load_ms, 1)
        if worker is not None and worker.load_ms
        else None,
        "timeout": worker.timeout if worker is not None else None,
    }


def _silero_worker_ensure(
    runtime: _SileroRuntime,
    *,
    script: Path,
    speaker: str,
    timeout: float,
    start_timeout: float,
    env: Mapping[str, str] | None,
) -> SileroWorker:
    """Вернуть живой worker, подняв его при необходимости (lock снаружи)."""
    global _SILERO_WORKER, _SILERO_WORKER_KEY
    key = (str(runtime.python), str(script))
    worker = _SILERO_WORKER
    if worker is not None and _SILERO_WORKER_KEY != key:
        logger.info("silero: конфигурация изменилась — перезапускаю worker")
        worker.stop()
        worker = None
    if worker is not None and not worker.alive():
        worker.stop()
        worker = None
    if worker is not None:
        worker.timeout = max(1.0, float(timeout))
        return worker

    candidate = SileroWorker(
        runtime.python,
        script,
        speaker=speaker,
        env=env,
        timeout=timeout,
        start_timeout=start_timeout,
    )
    try:
        candidate.start()
        candidate.ping()
    except TTSError:
        candidate.stop()
        _SILERO_WORKER = None
        _SILERO_WORKER_KEY = None
        raise
    _SILERO_WORKER = candidate
    _SILERO_WORKER_KEY = key
    return candidate


def _drop_silero_worker() -> None:
    """Убить worker и забыть о нём (следующий вызов поднимет заново)."""
    global _SILERO_WORKER, _SILERO_WORKER_KEY
    worker = _SILERO_WORKER
    _SILERO_WORKER = None
    _SILERO_WORKER_KEY = None
    if worker is not None:
        worker.stop()


def stop_silero_worker(*, timeout: float = 1.0) -> None:
    """Остановить worker (``atexit`` и явная уборка); без worker'а — no-op."""
    global _SILERO_WORKER, _SILERO_WORKER_KEY
    acquired = _SILERO_WORKER_LOCK.acquire(timeout=max(0.0, float(timeout)))
    try:
        worker = _SILERO_WORKER
        _SILERO_WORKER = None
        _SILERO_WORKER_KEY = None
        if worker is not None:
            logger.info("silero: останавливаю worker (pid %s)", worker.pid)
            worker.stop()
    finally:
        if acquired:
            _SILERO_WORKER_LOCK.release()


#: Процесс чата не должен оставлять после себя 550 МБ torch: worker получает
#: SIGTERM при выходе, даже если вызывающий код забыл про уборку.
atexit.register(stop_silero_worker)


def prewarm_silero_worker(
    engine: "SileroTTS | None" = None,
    *,
    env: Mapping[str, str] | None = None,
) -> threading.Thread | None:
    """Поднять worker заранее, фоном: первая фраза станет ~0,05–0,2 с.

    Ленивый старт (первый синтез) платит ~1,5–2,5 с за импорт torch и модель,
    поэтому первая фраза после старта чата дольше последующих. Если сервер
    (или голосовой цикл) вызовет эту функцию на старте, к приходу первой
    фразы worker уже готов. Это не ошибка и не обязательство: Silero может
    быть не установлен, тогда просто возвращается ``None``.
    """
    adapter = engine if engine is not None else SileroTTS(env=env)
    try:
        if not adapter.available():
            return None
    except TTSError:
        return None
    if silero_worker_info()["running"]:
        return None

    def _warm() -> None:
        try:
            adapter.prewarm()
        except TTSError as exc:
            logger.warning(
                "silero: prewarm не удался (%s) — останется ленивый старт", exc
            )
        except Exception as exc:  # noqa: BLE001 — фон не должен шуметь трейсом
            logger.warning("silero: prewarm упал (%s)", exc)

    thread = threading.Thread(target=_warm, name="silero-prewarm", daemon=True)
    thread.start()
    return thread


def _silero_safe_speaker(speaker: str | None) -> str:
    """Вернуть допустимый голос Silero: неизвестный → baya.

    Клиент (приложение) мог сохранить старое значение из эпохи Piper
    (`young`, `soft`, `irina`). Раньше это давало ValueError «доступные голоса…»
    и откат на системный `say` — то есть Ева замолкала голосом Евы.
    Лучше сказать голосом по умолчанию, чем не сказать вовсе.
    """
    known = {"aidar", "baya", "kseniya", "xenia", "eugene", "random"}
    value = (speaker or "").strip().lower()
    return value if value in known else "baya"


class SileroTTS(TTSAdapter):
    """Локальный Silero v4_ru в долгоживущем процессе python3.13 (worker).

    ``tools/silero_worker.py`` грузит модель **один раз** и дальше принимает
    фразы JSON-строками (stdin → stdout)::

        python3.13 tools/silero_worker.py --speaker baya --sample-rate 48000
          → {"text": "...", "speaker": "baya", "out": "/tmp/reply.wav"}
          ← {"ok": true, "ms": 41.2, "duration_s": 1.19, ...}

    Старт ленивый (при первой фразе), дальше процесс живёт до конца процесса
    чата: ``torch`` и модель не грузятся повторно. ``PYTHONPATH`` ровно на
    ``vendor/torch-py``, ``TORCH_HOME=vendor/silero``. Один worker на процесс
    (``select_engine`` создаёт адаптер на запрос, поэтому процесс — модульный
    синглтон), запросы по одному (lock), падение — один перезапуск, таймаут
    (20 с) — убить и уступить системному TTS, ``atexit`` — SIGTERM. Повторные
    одинаковые фразы отдаёт LRU-кэш и не трогает worker.

    ========================== ============================================
    ``SILERO_PYTHON``          python3.13 (по умолчанию из PATH или Homebrew)
    ``SILERO_TORCH_DIR``       каталог вендорного torch (``vendor/torch-py``)
    ``SILERO_MODEL_DIR``       кэш модели (``vendor/silero``)
    ``SILERO_SCRIPT``          одноразовый синтезатор-фолбэк (0193)
    ``SILERO_WORKER_SCRIPT``   скрипт worker'а (``tools/silero_worker.py``)
    ``SILERO_WORKER``          ``0`` — не использовать worker (одноразовый путь)
    ``SILERO_WORKER_TIMEOUT``  таймаут запроса, сек (по умолчанию 20)
    ``SILERO_WORKER_START_TIMEOUT`` таймаут старта, сек (по умолчанию 30)
    ``SILERO_SPEAKER``         голос: ``baya``/``kseniya``/``xenia``/``aidar``/…
    ``SILERO_CACHE``           ``0`` отключает LRU-кэш повторных фраз
    ``EVA_STRESS``             ``0`` отключает расстановку ударений (0222)
    ``EVA_STRESS_FILE``        свой словарь ударений (по умолчанию
                               ``data/stress.txt``)
    ========================== ============================================

    Голос задаётся моделью, а не пост-обработкой: ``voice``, совпавший с
    именем голоса Silero (``baya``/``kseniya``/``xenia``/``aidar`` и др.),
    уважается — так страница чата и EVA.app выбирают голос явно. Никакого
    питч-шифта или сглаживания WAV нет (задача 0197).
    """

    name = "silero"
    fragment_threshold = MIN_WAV_BYTES

    def __init__(
        self,
        python: str | Path | None = None,
        *,
        torch_dir: str | Path | None = None,
        model_dir: str | Path | None = None,
        script: str | Path | None = None,
        speaker: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._python = python
        self._torch_dir = torch_dir
        self._model_dir = model_dir
        self._script = script
        self._speaker = speaker
        self._env = env
        self._resolved: _SileroRuntime | None = None

    def _configured_speaker(self) -> str:
        raw = (
            self._speaker
            or _env(self._env, "SILERO_SPEAKER")
            or DEFAULT_SILERO_SPEAKER
        )
        return (raw or DEFAULT_SILERO_SPEAKER).strip().lower()

    def resolve(self) -> _SileroRuntime:
        """Найти python/torch/модель/скрипт или поднять :class:`TTSUnavailable`."""
        if self._resolved is not None:
            return self._resolved
        python = find_silero_python(self._python, env=self._env)
        torch_dir = find_silero_torch_dir(self._torch_dir, env=self._env)
        model_dir = find_silero_model_dir(self._model_dir, env=self._env)
        raw_script = self._script or _env(self._env, "SILERO_SCRIPT")
        script = (
            Path(raw_script).expanduser()
            if raw_script
            else ROOT / SILERO_SCRIPT
        )
        if not script.is_file():
            raise TTSUnavailable(
                f"Silero: скрипт-синтезатор не найден: {script}. "
                "Ожидается tools/silero_speak.py в репозитории."
            )
        if _env_flag(self._env, "SILERO_WORKER", True):
            # Worker — основной путь: сломанная установка (нет скрипта) должна
            # быть видна в available(), а не выясняться на первой фразе.
            find_silero_worker(None, env=self._env)
        speaker = self._configured_speaker()
        if speaker not in SILERO_SPEAKERS:
            raise TTSUnavailable(
                f"SILERO_SPEAKER={speaker!r}: неизвестный голос. Допустимо: "
                + ", ".join(SILERO_SPEAKERS)
            )
        if python is None or torch_dir is None or model_dir is None:
            raise TTSUnavailable(silero_help(python, torch_dir, model_dir))
        weights = _silero_weights(model_dir)
        if weights is None:  # защита от гонки: каталог опустел
            raise TTSUnavailable(silero_help(python, torch_dir, model_dir))
        self._resolved = _SileroRuntime(
            python=python,
            torch_dir=torch_dir,
            model_dir=model_dir,
            script=script,
            speaker=speaker,
            weights=weights,
        )
        return self._resolved

    def available(self) -> bool:
        try:
            self.resolve()
        except TTSUnavailable:
            return False
        return True

    def describe(self) -> str:
        """Строка для отчёта: ``silero (worker, pid N; …)`` или «не запущен»."""
        try:
            runtime = self.resolve()
        except TTSUnavailable:
            return (
                "silero (недоступен: не установлен — sh tools/install_silero.sh)"
            )
        info = silero_worker_info()
        if info["running"]:
            head = f"silero (worker, pid {info['pid']}"
        else:
            head = "silero (worker не запущен"
        text = (
            f"{head}; голос {runtime.speaker}, {SILERO_SAMPLE_RATE} Гц, "
            f"{runtime.python.name}, модель {runtime.weights.name}"
        )
        if info["load_ms"]:
            text += f", модель грузилась {float(info['load_ms']):.0f} мс"
        if _SILERO_WARMUP_MS is not None:
            text += f", первая фраза {_SILERO_WARMUP_MS:.0f} мс"
        return text + ")"

    def _speaker_for(self, requested: str | None) -> str:
        """Голос запроса, если это голос Silero, иначе настроенный.

        Страница чата и EVA.app присылают ``baya``/``kseniya``/``xenia``;
        любое чужое имя (или пустое) означает «голос по умолчанию» из
        ``SILERO_SPEAKER``, а не подмену на случайный тембр.
        """
        candidate = (requested or "").strip().lower()
        if candidate in SILERO_SPEAKERS:
            return candidate
        return self._configured_speaker()

    def _prepare_text(self, text: object) -> str:
        """Текст для Silero: чистка + знак ударения ``+`` перед гласной.

        Silero v4_ru понимает ``+`` («Колоб+ок» → Колобо́к), но сам ошибается
        на сказочных словах; :func:`eva.ru_stress.apply_stress` расставляет
        знаки по словарю. Это единственный движок, которому уходит разметка:
        в тексте ответа пользователю плюсов нет, ``EVA_STRESS=0`` отключает.
        """
        return apply_stress(_clean_text(text), env=self._env)

    def build_command(
        self,
        text: str,
        out_path: str | Path,
        *,
        rate: int | None = None,
        voice: str | None = None,
    ) -> tuple[list[str], str | None]:
        """Команда одноразового скрипта (путь 0193, ``SILERO_WORKER=0``)."""
        runtime = self.resolve()
        speaker = self._speaker_for(voice)
        command = [
            str(runtime.python),
            str(runtime.script),
            str(Path(out_path).expanduser()),
            "--speaker",
            speaker,
            "--sample-rate",
            str(SILERO_SAMPLE_RATE),
        ]
        # Прямой вызов build_command тоже должен получить разметку: обычно
        # текст уже подготовлен _prepare_text, а apply_stress идемпотентна.
        return command, apply_stress(text, env=self._env) + "\n"

    def subprocess_env(self) -> dict[str, str] | None:
        """Окружение дочернего процесса: только вендорный torch и кэш модели."""
        runtime = self.resolve()
        env = dict(os.environ)
        # Ровно vendor/torch-py: иначе сломанный source-checkout из .pylibs
        # (или чужой torch из site-packages) может оказаться первым в sys.path.
        env["PYTHONPATH"] = str(runtime.torch_dir)
        env["TORCH_HOME"] = str(runtime.model_dir)
        env["PYTHONNOUSERSITE"] = "1"
        threads = str(env.get("SILERO_THREADS", "4")).strip() or "4"
        env["SILERO_THREADS"] = threads
        env.setdefault("OMP_NUM_THREADS", threads)
        env["SILERO_SPEAKER"] = runtime.speaker
        return env

    def _worker_timeout(self, timeout: float | None) -> float:
        """Таймаут запроса к worker'у: ``SILERO_WORKER_TIMEOUT`` или явный меньше."""
        base = _env_float(
            self._env, "SILERO_WORKER_TIMEOUT", DEFAULT_SILERO_WORKER_TIMEOUT
        )
        # Базовый контракт передаёт DEFAULT_TIMEOUT (120 с), даже когда
        # значение не задавали: для фразы это бессмысленно долго.
        if timeout is None or float(timeout) >= DEFAULT_TIMEOUT:
            return base
        return max(1.0, min(base, float(timeout)))

    def _worker_start_timeout(self) -> float:
        return _env_float(
            self._env,
            "SILERO_WORKER_START_TIMEOUT",
            DEFAULT_SILERO_WORKER_START_TIMEOUT,
        )

    def prewarm(self) -> int | None:
        """Поднять worker заранее (и прогреть синтез), вернуть его pid.

        Нужен серверу/голосовому циклу, если первая фраза не должна платить
        ~1,5–2,5 с за torch и модель: вызовите до первого запроса синтеза.
        Короткий пробный синтез во временный файл догревает TorchScript
        (первый ``apply_tts`` на ~1,2 с дороже остальных), поэтому после
        prewarm первая настоящая фраза идёт за ~0,05–0,2 с.
        """
        runtime = self.resolve()
        script = find_silero_worker(None, env=self._env)
        timeout = self._worker_timeout(None)
        with _SILERO_WORKER_LOCK:
            worker = _silero_worker_ensure(
                runtime,
                script=script,
                speaker=runtime.speaker,
                timeout=timeout,
                start_timeout=self._worker_start_timeout(),
                env=self.subprocess_env(),
            )
            with tempfile.TemporaryDirectory(prefix="eva-silero-warmup-") as tmp:
                try:
                    worker.synthesize(
                        "Э.",
                        runtime.speaker,
                        Path(tmp) / "warmup.wav",
                        timeout=min(timeout, 10.0),
                    )
                except TTSError as exc:
                    # Прогрев не обязателен: фраза всё равно синтезируется,
                    # просто первый вызов будет дороже.
                    logger.debug("silero: прогрев синтеза не удался (%s)", exc)
        return worker.pid

    def _synthesize_with_worker(
        self,
        text: str,
        target: Path,
        speaker: str,
        runtime: _SileroRuntime,
        timeout: float,
    ) -> SynthesisResult:
        """Фраза через worker: lock, один перезапуск при падении, таймаут."""
        script = find_silero_worker(None, env=self._env)
        worker_timeout = self._worker_timeout(timeout)
        start_timeout = self._worker_start_timeout()
        env = self.subprocess_env()
        target = _prepare_target(target)
        started = time.perf_counter()

        with _SILERO_WORKER_LOCK:
            worker = _silero_worker_ensure(
                runtime,
                script=script,
                speaker=speaker,
                timeout=worker_timeout,
                start_timeout=start_timeout,
                env=env,
            )
            attempt = 0
            while True:
                try:
                    worker.synthesize(
                        text, speaker, target, timeout=worker_timeout
                    )
                    break
                except _WorkerTimeout as exc:
                    # Зависший процесс бесполезен: убиваем. Следующий вызов
                    # поднимет worker заново (это и есть «перезапуск»), а
                    # текущая фраза уходит на system — ждать 2×20 с нельзя.
                    _drop_silero_worker()
                    raise TTSError(
                        f"silero: {exc}; worker остановлен, "
                        "следующий вызов поднимет его заново"
                    ) from None
                except _WorkerGone as exc:
                    _drop_silero_worker()
                    if attempt >= SILERO_WORKER_RESTARTS:
                        raise TTSError(
                            f"silero: worker падает повторно ({exc})"
                        ) from None
                    attempt += 1
                    logger.warning(
                        "silero: worker упал (%s) — перезапускаю (попытка %d из %d)",
                        exc,
                        attempt,
                        SILERO_WORKER_RESTARTS,
                    )
                    worker = _silero_worker_ensure(
                        runtime,
                        script=script,
                        speaker=speaker,
                        timeout=worker_timeout,
                        start_timeout=start_timeout,
                        env=env,
                    )

        synth_ms = (time.perf_counter() - started) * 1000.0
        try:
            size = target.stat().st_size if target.is_file() else 0
        except OSError:
            size = 0
        if size <= MIN_WAV_BYTES:
            raise TTSError(
                f"silero: worker не создал аудио ({target}). "
                "Проверьте голос/модель и права на запись."
            )
        # Worker отвечает целиком после синтеза, поэтому «первый фрагмент» =
        # полное время (как у ElevenLabs: потоковой отдачи байт нет).
        return SynthesisResult(
            path=target,
            synth_ms=synth_ms,
            first_fragment_ms=synth_ms,
            engine=self.name,
        )

    def synthesize_timed(
        self,
        text: str,
        out_path: str | Path,
        *,
        rate: int | None = None,
        voice: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> SynthesisResult:
        """Синтез через worker с LRU-кэшем и замером первой фразы.

        ``SILERO_WORKER=0`` возвращает одноразовый путь 0193
        (``tools/silero_speak.py``): он нужен для отладки и как аварийный
        обход, если worker на машине не работает.
        """
        global _SILERO_WARMUP_MS
        clean = self._prepare_text(text)
        target = Path(out_path).expanduser()
        runtime = self.resolve()
        speaker = self._speaker_for(voice)
        key = (clean, speaker, SILERO_SAMPLE_RATE)
        use_cache = _env_flag(self._env, "SILERO_CACHE", True)
        if use_cache:
            cached = _silero_cache_get(key)
            if cached is not None:
                target = _prepare_target(target)
                started = time.perf_counter()
                try:
                    target.write_bytes(cached)
                except OSError as exc:
                    raise TTSError(
                        f"не удалось записать аудио в {target}: {exc}"
                    ) from exc
                synth_ms = (time.perf_counter() - started) * 1000.0
                return SynthesisResult(
                    path=target,
                    synth_ms=synth_ms,
                    first_fragment_ms=synth_ms,
                    engine=self.name,
                )

        if _env_flag(self._env, "SILERO_WORKER", True):
            result = self._synthesize_with_worker(
                clean, target, speaker, runtime, timeout
            )
        else:
            result = self._synthesize_prepared(
                clean, target, rate=rate, voice=voice, timeout=timeout
            )

        if use_cache:
            try:
                data = target.read_bytes()
            except OSError:
                data = b""
            if data:
                _silero_cache_put(key, data)
        if _SILERO_WARMUP_MS is None:
            _SILERO_WARMUP_MS = result.synth_ms
            logger.info(
                "silero: первая фраза %.0f мс (старт worker'а: torch + модель "
                "v4_ru; дальше модель уже загружена)",
                result.synth_ms,
            )
        return result


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
    Silero/system делает модульный :func:`synthesize`, а не сам адаптер.

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
                "без ключа работает Silero/system."
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

        Страница чата и EVA.app присылают имя голоса Silero (``baya`` и т.п.),
        а не ID ElevenLabs; такое короткое имя игнорируется. ID из библиотеки —
        длинная буквенно-цифровая строка (обычно 20 символов).
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
    поэтому замер первого фрагмента одинаковый. Голос Silero (``baya`` и др.)
    для системного TTS не годится: при откате используется ``Milena``, чтобы
    озвучка не падала из-за неизвестного имени голоса.
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
            # Страница и EVA.app всегда присылают голос Silero; при откате на
            # system он не годится для ``say -v`` (такого голоса нет), поэтому
            # берём Milena — «не падать» важнее, чем угадать чужой тембр.
            if str(spoken).strip().lower() in SILERO_SPEAKERS:
                spoken = DEFAULT_SAY_VOICE
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
    silero_python: str | Path | None = None,
    silero_model_dir: str | Path | None = None,
    silero_speaker: str | None = None,
    say_bin: str | Path | None = None,
    voice: str | None = None,
    lang: str = DEFAULT_TTS_LANG,
    env: Mapping[str, str] | None = None,
) -> TTSAdapter:
    """Выбрать TTS-движок.

    ``prefer``/``EVA_TTS``: ``auto`` (по умолчанию), ``elevenlabs``,
    ``silero``, ``system``, ``null`` (``piper`` — устаревший явный выбор).
    Явный выбор перекрывает автовыбор и не подменяется: если выбранный
    движок не настроен, :meth:`TTSAdapter.available` вернёт False, а синтез
    поднимет :class:`TTSUnavailable` с подсказкой.

    Авто: ElevenLabs (если задан ``ELEVENLABS_API_KEY``), иначе Silero (если
    установлен ``tools/install_silero.sh``: python3.13 + vendor/torch-py +
    модель), иначе SystemTTS, иначе NullTTS. Отсутствие ключа и Silero — не
    ошибка, а причина фолбэка: локальное по умолчанию лучше облака. Piper в
    автовыборе не участвует (задача 0197).
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
    silero = SileroTTS(
        silero_python,
        model_dir=silero_model_dir,
        speaker=silero_speaker,
        env=env,
    )
    piper = PiperTTS(piper_bin, piper_model, env=env)
    if choice == "elevenlabs":
        return elevenlabs
    if choice == "silero":
        return silero
    if choice == "piper":
        return piper
    if choice == "system":
        return SystemTTS(say_bin, voice=voice, lang=lang, env=env)
    if choice == "null":
        return NullTTS()

    if elevenlabs.available():
        return elevenlabs
    if silero.available():
        return silero
    system = SystemTTS(say_bin, voice=voice, lang=lang, env=env)
    if system.available():
        return system
    return NullTTS()


def describe_engine(engine: TTSAdapter | None) -> str:
    """Строка вида ``silero (worker, …)`` для отчёта о выборе движка."""
    if engine is None:
        return "none"
    try:
        return engine.describe()
    except Exception:  # noqa: BLE001 — описание не должно ронять вывод
        return engine.name


def _fallback_engines(
    *,
    skip: Sequence[str] = (),
    piper_bin: str | Path | None = None,
    piper_model: str | Path | None = None,
    say_bin: str | Path | None = None,
    voice: str | None = None,
    lang: str = DEFAULT_TTS_LANG,
    env: Mapping[str, str] | None = None,
) -> list[TTSAdapter]:
    """Локальные запасные движки в порядке предпочтения: Silero → system.

    Движки из ``skip`` не предлагаются (например, Silero, который только что
    упал). Возвращается список доступных прямо сейчас; пустой список означает
    «локального голоса нет» — вызывающий поднимает исходную ошибку. Параметры
    ``piper_bin``/``piper_model`` остались ради совместимости сигнатуры: Piper
    в откате больше не участвует (задача 0197).
    """
    engines: list[TTSAdapter] = []
    if "silero" not in skip:
        silero = SileroTTS(env=env)
        if silero.available():
            engines.append(silero)
    if "system" not in skip:
        system = SystemTTS(say_bin, voice=voice, lang=lang, env=env)
        if system.available():
            engines.append(system)
    return engines


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
    """Синтез с откатом ElevenLabs/Silero → локальные движки.

    Облако может ответить 401/429/сетью, а Silero — упасть из-за
    отсутствующего/битого torch. В обоих случаях пробуем следующий локальный
    движок (Silero, затем system), логируя причину. Если не осталось ни
    одного — поднимаем исходную ошибку, а не отдаём тишину.
    """
    env_source = os.environ if env is None else env
    # Числа — словами до выбора движка: иначе Silero молча пропускает цифры.
    # Адаптеры нормализуют повторно в ``_clean_text``, но переопределённый
    # ``synthesize`` (тесты, сторонние движки) получает уже готовый текст.
    text = normalize_tts_text(text)
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
        if chosen.name not in ("elevenlabs", "silero"):
            raise
        api_key = _env(env_source, "ELEVENLABS_API_KEY")
        fallbacks = _fallback_engines(
            skip=(chosen.name,),
            piper_bin=piper_bin,
            piper_model=piper_model,
            say_bin=say_bin,
            voice=voice,
            lang=lang,
            env=env,
        )
        last_error: TTSError = exc
        for fallback in fallbacks:
            try:
                result = fallback.synthesize(
                    text, out_path, rate=rate, voice=voice, timeout=timeout
                )
            except TTSError as fallback_exc:
                # Запасной тоже упал (например, битый Silero): пробуем
                # следующий, а не сдаёмся после первого.
                last_error = fallback_exc
                logger.warning(
                    "озвучка: %s не сработал (%s), пробую следующий движок",
                    fallback.name,
                    fallback_exc,
                )
                continue
            logger.warning(
                "озвучка: %s не сработал (%s), откат на %s",
                chosen.name,
                _redact(exc, api_key),
                fallback.name,
            )
            return result, fallback

        # Нечего озвучивать локально: показываем исходную причину, а не
        # «тишину». Ключ в текст не попадает (redact на всякий случай).
        if chosen.name == "elevenlabs":
            raise TTSError(
                f"ElevenLabs: {_redact(exc, api_key)}; запасной Silero/system "
                "тоже недоступен."
            ) from None
        raise TTSError(
            f"Silero: {last_error}; запасной system тоже недоступен."
        ) from None


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

    Без ``out_path`` — аудиобайты (Silero/system — WAV, ElevenLabs — пара
    ``(mp3, "audio/mpeg")``); с ``out_path`` — путь. Если Silero нет,
    автовыбор уходит на системный TTS, а при полном отсутствии TTS вернёт
    пустые байты (NullTTS), не падая. Сбой ElevenLabs (401/429/сеть/пусто) или
    Silero логируется без ключа и откатывается на локальный движок
    (Silero/system).
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
    ``audio/mpeg``, локальные движки (Silero/system) — WAV и ``audio/wav``.
    Тот же откат: при сбое ElevenLabs или Silero возвращается аудио локального
    движка, а не ошибка.
    """
    result, _chosen = _synthesize_with_fallback(
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
        # WAV от Silero/system: голос задаёт движок, пост-обработки нет
        # (задача 0197). HTTP-слой всё равно определяет тип по сигнатуре RIFF.
        return bytes(result), "audio/wav"
    # Path без out_path не возвращается, но контракт защищаем.
    return b"", "audio/wav"
