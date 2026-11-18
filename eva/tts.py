"""TTS-адаптер EVA32: локальный нейро-TTS Piper → системный say/espeak → null.

Задача — [`ai/tasks/0093-eva-tts-piper.md`](../ai/tasks/0093-eva-tts-piper.md).
Для «замены Алисы» нужен локальный нейро-голос, но Piper ставится отдельно
(бинарь + модель ``.onnx``), и его может не быть. Поэтому весь синтез идёт
через один интерфейс, а движок выбирается по доступности:

============ ==========================================================
``piper``    локальный нейро-TTS: ``piper --model MODEL --output_file OUT``,
             текст — в stdin. Пути из ``PIPER_BIN``/``PIPER_MODEL``
             (или ``--piper-bin``/``--piper-model`` в ``voice_loop.py``).
``system``   macOS ``say`` (голос Milena) или Linux ``espeak``/``espeak-ng``.
``null``     заглушка: синтеза нет, ничего не пишет, ничего не играет.
============ ==========================================================

Автовыбор: **Piper, если доступен** (есть бинарь и модель), иначе
**SystemTTS**, иначе **NullTTS**. Явный выбор (``EVA_TTS``/``prefer=``)
перекрывает автовыбор — даже если движок недоступен: тогда при синтезе будет
понятная :class:`TTSUnavailable`, а не молчаливая подмена.

Контракт синтеза — ``synthesize(text) -> bytes | Path``::

    from eva.tts import synthesize

    wav_bytes = synthesize("Сколько стоит тариф Старт")       # bytes
    wav_path = synthesize("Привет", "/tmp/reply.wav")          # Path

Адаптеры (:class:`PiperTTS`, :class:`SystemTTS`, :class:`NullTTS`) умеют то же
самое и дополнительно :meth:`TTSAdapter.synthesize_timed` — с замерами
``synth_ms`` и ``first_fragment_ms`` для голосового цикла.

Только стандартная библиотека. Сеть не используется, модели не скачиваются,
реальный звук не воспроизводится (это делает ``voice_loop.py`` через afplay).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

__all__ = [
    "DEFAULT_SAY_VOICE",
    "DEFAULT_TIMEOUT",
    "DEFAULT_TTS_LANG",
    "ENGINE_CHOICES",
    "NullTTS",
    "PiperTTS",
    "SystemTTS",
    "SynthesisResult",
    "TTSAdapter",
    "TTSError",
    "TTSUnavailable",
    "build_say_command",
    "describe_engine",
    "find_espeak_binary",
    "find_piper_binary",
    "find_piper_model",
    "find_say_binary",
    "piper_help",
    "select_engine",
    "synthesize",
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

#: Допустимые значения ``--tts``/``EVA_TTS``.
ENGINE_CHOICES = ("auto", "piper", "system", "null")

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

    ``prefer``/``EVA_TTS``: ``auto`` (по умолчанию), ``piper``, ``system``,
    ``null``. Явный выбор перекрывает автовыбор и не подменяется: если
    выбранный движок не настроен, :meth:`TTSAdapter.available` вернёт False, а
    синтез поднимет :class:`TTSUnavailable` с подсказкой.

    Авто: Piper (если есть бинарь **и** модель), иначе SystemTTS, иначе
    NullTTS. Отсутствие Piper — не ошибка, а причина фолбэка.
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

    piper = PiperTTS(piper_bin, piper_model, env=env)
    if choice == "piper":
        return piper
    if choice == "system":
        return SystemTTS(say_bin, voice=voice, lang=lang, env=env)
    if choice == "null":
        return NullTTS()

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
) -> bytes | Path:
    """Синтезировать текст выбранным (или переданным) движком.

    Без ``out_path`` — WAV-байты; с ``out_path`` — путь. Если Piper нет,
    автовыбор уходит на системный TTS, а при полном отсутствии TTS вернёт
    пустые байты (NullTTS), не падая.
    """
    chosen = engine or select_engine(
        prefer,
        piper_bin=piper_bin,
        piper_model=piper_model,
        say_bin=say_bin,
        voice=voice,
        lang=lang,
        env=env,
    )
    return chosen.synthesize(
        text, out_path, rate=rate, voice=voice, timeout=timeout
    )
