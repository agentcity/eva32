"""Плагин ``music``: короткие команды управления Music.app и радио.

Способность — ``answer``: сервер зовёт :func:`answer` **до** модели и, если
плагин вернул текст, не генерирует ответ вовсе. Поэтому здесь только
распознавание фразы и один AppleScript-вызов; никаких длинных рассуждений.

Почему ``osascript``, а не API: Music.app на macOS не имеет публичного
локального HTTP-API, зато умеет AppleScript. Вызов идёт списком аргументов
(``["/usr/bin/osascript", "-e", script]``) без ``shell=True``; пользовательская
строка попадает в скрипт только строковым литералом AppleScript с
экранированием ``\\`` и ``"``. Никакой конкатенации с shell-командами нет.

Ограничение среды: первый запрос к Music.app может ждать TCC-разрешение
«терминал хочет управлять Music». В headless-сессии диалог никто не
подтверждает, поэтому ``subprocess.run`` всегда с ``timeout``: плагин отвечает
честным «не удалось», а не вешает чат. Радио идёт через ``afplay`` и TCC не
требует.
"""

from __future__ import annotations

import re
import signal
import subprocess
import time
from typing import Any, NamedTuple

from eva.plugins.context import PluginContext

__all__ = [
    "DEFAULT_TIMEOUT",
    "MusicPlugin",
    "MusicUnavailable",
    "answer",
    "register",
]

#: Сколько ждать AppleScript: дольше чат ждать музыкальную команду не должен.
DEFAULT_TIMEOUT = 5.0

OSASCRIPT = "/usr/bin/osascript"
AFPLAY = "/usr/bin/afplay"


#: Публичные прямые потоки без ключей и плейлистов. Проверяются afplay'ем.
class RadioStation(NamedTuple):
    """Зашитая публичная станция: имя для ответа, URL и синонимы команды."""

    key: str
    name: str
    url: str
    aliases: tuple[str, ...]


_RADIO_STATIONS = (
    RadioStation(
        key="europa",
        name="Европа Плюс",
        url="http://ep128.hostingradio.ru:8030/ep128",
        aliases=(
            "европа",
            "европу",
            "европа плюс",
            "европу плюс",
            "europa",
            "europa plus",
            "еп",
        ),
    ),
    RadioStation(
        key="dfm",
        name="DFM",
        url="https://dfm.hostingradio.ru/dfm96.aacp",
        aliases=("dfm", "дфм", "дфм радио"),
    ),
    RadioStation(
        key="nashe",
        name="Наше Радио",
        url="https://nashe1.hostingradio.ru/nashe-128.mp3",
        aliases=("наше", "наше радио", "nashe", "nashe radio"),
    ),
)

#: Эти названия — станции и без слова «радио»: «включи европу плюс».
_STATION_ONLY_ALIASES = frozenset(
    {
        "европа плюс",
        "европу плюс",
        "europa plus",
        "dfm",
        "дфм",
        "наше радио",
    }
)


class MusicUnavailable(RuntimeError):
    """Music.app/osascript не сработали; ``kind`` — причина для честного текста."""

    def __init__(self, message: str, kind: str = "error") -> None:
        self.kind = kind
        super().__init__(message)


# --------------------------------------------------------------------------- #
# Распознавание фраз
# --------------------------------------------------------------------------- #

#: Слова-заполнители перед названием: «включи мне песню Кино» → «Кино».
_FILLERS = frozenset(
    {
        "пожалуйста",
        "мне",
        "нам",
        "песню",
        "песня",
        "трек",
        "композицию",
        "композиция",
        "группу",
        "группа",
        "исполнителя",
        "исполнитель",
        "альбом",
        "плейлист",
        "музыку",
        "музыка",
        "что-нибудь",
        "что нибудь",
        "какую-нибудь",
        "какую нибудь",
        "любую",
        "любимую",
        "послушать",
        "послушаем",
        "из",
        "сейчас",
        "немного",
    }
)

#: «Включи свет» — не наша зона: пусть отвечает модель, а не «не нашла трек».
_NON_MUSIC = frozenset(
    {
        "свет",
        "лампу",
        "лампочку",
        "люстру",
        "чайник",
        "телевизор",
        "тв",
        "телик",
        "кондиционер",
        "вентилятор",
        "компьютер",
        "ноутбук",
        "монитор",
        "проектор",
        "окно",
        "дверь",
        "wifi",
        "wi-fi",
        "интернет",
        "блютуз",
        "bluetooth",
        "будильник",
        "таймер",
        "кофеварку",
        "микроволновку",
        "плиту",
        "духовку",
        "утюг",
        "розетку",
        "пылесос",
        "стиралку",
        "воду",
        "газ",
        "отопление",
        "камеру",
        "запись",
        "уведомления",
    }
)

_PAUSE_RE = re.compile(
    r"^(?:пауза|паузу|на паузу|поставь(?: музыку)? на паузу"
    r"|приостанови(?: (?:музыку|воспроизведение))?|pause)$"
)
_RESUME_RE = re.compile(
    r"^(?:продолжи(?:ть)?(?: (?:музыку|воспроизведение|играть))?"
    r"|продолжай(?: музыку)?|сними с паузы|играй дальше|запусти снова"
    r"|возобнови(?: музыку)?|resume)$"
)
_NEXT_RE = re.compile(
    r"^(?:(?:следующая|следующий|следующую|следующее|следующ\w+)"
    r"(?: (?:трек|песня|песню|композици\w+))?"
    r"|дальше|переключи(?: (?:на|вперёд|вперед))?"
    r"(?: (?:следующую|следующий|следующее|трек|песню))?"
    r"|включи (?:следующую|следующий)(?: (?:трек|песню))?"
    r"|следующий трек|next)$"
)
_PREV_RE = re.compile(
    r"^(?:(?:предыдущая|предыдущий|предыдущую|предыдущее|прошл\w+)"
    r"(?: (?:трек|песня|песню|композици\w+))?"
    r"|назад|верни(?:сь)?(?: назад)?(?: (?:предыдущ|прошл)\w+)?(?: (?:трек|песню))?"
    r"|переключи(?: (?:назад|на предыдущ\w+))?(?: (?:трек|песню))?"
    r"|включи (?:предыдущую|предыдущий|прошл\w+)(?: (?:трек|песню))?"
    r"|previous|prev)$"
)
_STOP_RE = re.compile(
    r"^(?:выключи (?:музыку|плеер|радио|трек|песню|воспроизведение)"
    r"|останови(?: (?:музыку|плеер|радио|воспроизведение))?"
    r"|стоп(?: (?:музыку|радио))?|хватит музыки|stop)$"
)
_LOUDER_RE = re.compile(
    r"^(?:громче|погромче|сделай(?: музыку)? (?:громче|погромче)"
    r"|сделай (?:звук|громкость) (?:громче|больше)"
    r"|прибавь(?: (?:звук|громкость))?|увеличь(?: (?:звук|громкость))?"
    r"|включи (?:громче|погромче)|louder)$"
)
_QUIETER_RE = re.compile(
    r"^(?:тише|потише|сделай(?: музыку)? (?:тише|потише)"
    r"|сделай (?:звук|громкость) (?:тише|меньше)"
    r"|убавь(?: (?:звук|громкость))?|уменьши(?: (?:звук|громкость))?"
    r"|включи (?:тише|потише)|quieter)$"
)
_MUTE_RE = re.compile(r"^(?:заглуши(?: музыку)?|выключи звук|без звука|mute)$")
_UNMUTE_RE = re.compile(r"^(?:включи звук|верни звук|сними mute|unmute)$")
_NOW_RE = re.compile(
    r"^(?:что (?:(?:сейчас|у нас) )?играет(?: сейчас)?"
    r"|что за (?:музыка|песня|трек|композиция)"
    r"|какая (?:сейчас )?(?:песня|музыка|композиция)(?: играет)?(?: сейчас)?"
    r"|какой (?:сейчас )?(?:трек|исполнитель)(?: играет)?(?: сейчас)?"
    r"|now playing)$"
)
_RADIO_RE = re.compile(
    r"^(?:включи|поставь|вруби|запусти|включай)"
    r" (?:интернет[- ])?радио(?: (?P<station>.+))?$"
)
_PLAY_RE = re.compile(
    r"^(?:включи|поставь|вруби|запусти|включай)"
    r"(?: (?:мне|нам|пожалуйста))*"
    r" (?:музыку|музыка|плеер|трек|песню|что-нибудь|что нибудь"
    r"|какую-нибудь|какую нибудь|любую музыку|любимую музыку|музон)"
    r"(?: (?:пожалуйста|сейчас|немного))*$"
)
_SEARCH_RE = re.compile(
    r"^(?:включи|поставь|вруби|запусти|включай|хочу послушать"
    r"|давай послушаем|поставь послушать)"
    r"(?: (?:мне|нам|пожалуйста))* (?P<query>.+)$"
)


def _normalize(question: str) -> str:
    """Нижний регистр, без кавычек/пунктуации, одиночные пробелы."""
    text = str(question or "").strip().lower().replace("ё", "е")
    text = re.sub(r"[\"'«»“”„]", " ", text)
    text = re.sub(r"[,.;:!?]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _display(text: str) -> str:
    """Вернуть название с заглавной буквы (запрос нормализован в нижний)."""
    value = str(text or "").strip()
    return value[:1].upper() + value[1:] if value else value


def _clean_query(raw: str) -> str:
    """Снять «песню/группу/мне/из» с начала и «пожалуйста» с конца."""
    tokens = raw.strip().split()
    while tokens and tokens[0] in _FILLERS:
        tokens.pop(0)
    while tokens and tokens[-1] in _FILLERS:
        tokens.pop()
    return " ".join(tokens)


def _station_by_alias(text: str) -> RadioStation | None:
    """Станция по точному названию без слова «радио» (или ``None``)."""
    value = _normalize(text)
    if value not in _STATION_ONLY_ALIASES:
        return None
    for station in _RADIO_STATIONS:
        if value in station.aliases:
            return station
    return None


def _intent(text: str) -> tuple[str, str] | None:
    """Разобрать нормализованную фразу в ``(вид, значение)`` или ``None``."""
    if not text or len(text) > 160:
        return None
    match = _RADIO_RE.match(text)
    if match is not None:
        station = (match.group("station") or "").strip()
        return ("radio", station)
    # «включи европу плюс» — это станция, а не поиск трека «Европа Плюс».
    match = _SEARCH_RE.match(text)
    if match is not None and _station_by_alias(match.group("query")) is not None:
        return ("radio", match.group("query"))
    if _PAUSE_RE.match(text):
        return ("pause", "")
    if _RESUME_RE.match(text):
        return ("resume", "")
    if _NEXT_RE.match(text):
        return ("next", "")
    if _PREV_RE.match(text):
        return ("prev", "")
    if _STOP_RE.match(text):
        return ("stop", "")
    if _LOUDER_RE.match(text):
        return ("louder", "")
    if _QUIETER_RE.match(text):
        return ("quieter", "")
    if _MUTE_RE.match(text):
        return ("mute", "")
    if _UNMUTE_RE.match(text):
        return ("unmute", "")
    if _NOW_RE.match(text):
        return ("now", "")
    if _PLAY_RE.match(text):
        return ("play", "")
    match = _SEARCH_RE.match(text)
    if match is not None:
        query = _clean_query(match.group("query"))
        if not query or query in _NON_MUSIC:
            return None
        if query.split()[0] in _NON_MUSIC:
            return None
        return ("search", query)
    return None


# --------------------------------------------------------------------------- #
# AppleScript
# --------------------------------------------------------------------------- #

_PLAY_SCRIPT = 'tell application "Music" to play'
_PAUSE_SCRIPT = 'tell application "Music" to pause'
_NEXT_SCRIPT = 'tell application "Music" to next track'
_PREV_SCRIPT = 'tell application "Music" to previous track'
_STOP_SCRIPT = 'tell application "Music" to stop'

_VOLUME_UP_SCRIPT = """tell application "Music"
    set currentVolume to sound volume
    set newVolume to currentVolume + 10
    if newVolume > 100 then set newVolume to 100
    set sound volume to newVolume
    return newVolume
end tell"""

_VOLUME_DOWN_SCRIPT = """tell application "Music"
    set currentVolume to sound volume
    set newVolume to currentVolume - 10
    if newVolume < 0 then set newVolume to 0
    set sound volume to newVolume
    return newVolume
end tell"""

_MUTE_SCRIPT = """tell application "Music"
    set sound volume to 0
    return 0
end tell"""

_UNMUTE_SCRIPT = """tell application "Music"
    set sound volume to 50
    return 50
end tell"""

_NOW_SCRIPT = """tell application "Music"
    set stateText to (player state as text)
    if stateText is "stopped" then return "stopped"
    try
        set currentTrack to current track
    on error
        return stateText & linefeed
    end try
    return stateText & linefeed & (artist of currentTrack) & " — " & (name of currentTrack)
end tell"""


def _applescript_literal(value: str) -> str:
    """Строковый литерал AppleScript: экранировать ``\\`` и ``"``.

    Переводы строк и NUL заменяются пробелом, чтобы пользовательская строка
    не могла закрыть литерал и дописать свой оператор.
    """
    cleaned = "".join(
        " " if char in "\r\n\t\x00" else char for char in str(value)
    )
    escaped = cleaned.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _search_script(query: str) -> str:
    """Плейлист по имени → трек по имени → трек по исполнителю → not-found."""
    literal = _applescript_literal(query)
    return f"""set queryText to {literal}
tell application "Music"
    set matchedPlaylists to (every playlist whose name contains queryText)
    if (count of matchedPlaylists) > 0 then
        set chosen to item 1 of matchedPlaylists
        play chosen
        return "playlist" & linefeed & (name of chosen)
    end if
    set matchedTracks to (every track whose name contains queryText)
    if (count of matchedTracks) > 0 then
        set chosen to item 1 of matchedTracks
        play chosen
        return "track" & linefeed & (name of chosen)
    end if
    set matchedArtists to (every track whose artist contains queryText)
    if (count of matchedArtists) > 0 then
        set chosen to item 1 of matchedArtists
        play chosen
        return "artist" & linefeed & (artist of chosen) & " — " & (name of chosen)
    end if
end tell
return "not-found\""""


#: Подсказки, по которым отличаем TCC-отказ от «объект не найден».
_PERMISSION_MARKERS = (
    "-1743",
    "-10004",
    "not authorized",
    "not allowed",
    "нарушение прав",
    "not permitted",
    "doesn't have permission",
)
_OBJECT_MARKERS = (
    "-1728",
    "can’t get",
    "can't get",
    "object not found",
    "не удалось найти",
)
_TIMEOUT_TEXT = (
    "Music.app не отвечает: macOS, похоже, ждёт разрешения на автоматизацию."
)
#: Текст отказа TCC — с конкретным путём, а не «включите автоматизацию».
#: Пользователь не должен угадывать, где это: Системные настройки →
#: Конфиденциальность и безопасность → Автоматизация → <приложение> → Music.
_PERMISSION_TEXT = (
    "macOS не разрешила управлять Music.app. Откройте Системные настройки → "
    "Конфиденциальность и безопасность → Автоматизация и включите Music для "
    "того приложения, откуда запущен сервер (обычно Терминал). Если пункта "
    "нет — выполните в Терминале tccutil reset AppleEvents и повторите "
    "команду: macOS спросит разрешение."
)
_OBJECT_TEXT = "Music.app не нашла нужный объект — возможно, он удалён из библиотеки."
_ERROR_TEXT = "Не удалось управлять Music.app: приложение закрыто или вернуло ошибку."
_MISSING_TEXT = "Не могу управлять музыкой: на этом Mac нет osascript."


def _friendly_error(exc: MusicUnavailable) -> str:
    if exc.kind == "missing":
        return _MISSING_TEXT
    if exc.kind == "timeout":
        return _TIMEOUT_TEXT
    if exc.kind == "permission":
        return _PERMISSION_TEXT
    if exc.kind == "object":
        return _OBJECT_TEXT
    return _ERROR_TEXT


class MusicPlugin:
    """Фразы → AppleScript / afplay; ответ — одно короткое предложение."""

    name = "music"

    def __init__(self) -> None:
        self._radio: Any = None
        self._radio_station: RadioStation | None = None
        self._radio_paused = False

    # -- публичная точка входа ----------------------------------------------

    def answer(self, ctx: PluginContext, question: str) -> str | None:
        """Ответить на музыкальную команду; ``None`` — это не наша тема."""
        try:
            return self._answer(ctx, question)
        except Exception as exc:  # noqa: BLE001 - плагин не должен ронять чат
            self._log(ctx, "непредвиденная ошибка: %s", exc)
            return "Не получилось выполнить музыкальную команду."

    def _answer(self, ctx: PluginContext, question: str) -> str | None:
        text = _normalize(question)
        intent = _intent(text)
        if intent is None:
            return None
        kind, value = intent

        if kind == "radio":
            return self._start_radio(ctx, value)
        if kind == "now":
            if self._radio_active():
                station = self._radio_station
                return f"Играет радио {station.name}." if station else "Играет радио."
            return self._now_playing(ctx)
        if kind == "pause":
            if self._radio_active():
                return "Пауза." if self._pause_radio() else "Не удалось поставить радио на паузу."
            return self._simple(ctx, "pause", "Пауза.")
        if kind == "resume":
            if self._radio_active():
                return "Продолжаю." if self._resume_radio() else "Не удалось продолжить радио."
            return self._simple(ctx, "resume", "Продолжаю.")
        if kind == "next":
            if self._radio_active():
                return "В радио нет следующего трека."
            return self._simple(ctx, "next", "Следующая.")
        if kind == "prev":
            if self._radio_active():
                return "В радио нет предыдущего трека."
            return self._simple(ctx, "prev", "Предыдущая.")
        if kind == "stop":
            if self._radio_active():
                self._stop_radio()
                return "Выключила радио."
            return self._simple(ctx, "stop", "Выключила музыку.")
        if kind == "louder":
            if self._radio_active():
                return "Громкость радио регулируется системным звуком."
            return self._volume(ctx, up=True)
        if kind == "quieter":
            if self._radio_active():
                return "Громкость радио регулируется системным звуком."
            return self._volume(ctx, up=False)
        if kind == "mute":
            if self._radio_active():
                return "У радио нет отдельной громкости — используйте системный звук."
            return self._set_volume(ctx, _MUTE_SCRIPT, "Звук выключен.")
        if kind == "unmute":
            if self._radio_active():
                return "У радио нет отдельной громкости — используйте системный звук."
            return self._set_volume(ctx, _UNMUTE_SCRIPT, "Звук включён.")

        if kind in ("play", "search"):
            # Перед Music.app гасим радио: иначе польются два потока сразу.
            self._stop_radio()
            if kind == "play":
                answer = self._simple(ctx, "play", "Включаю музыку.")
                if answer == _PERMISSION_TEXT or answer == _TIMEOUT_TEXT:
                    # Нет прав автоматизации: включаем радио — ему TCC не нужен.
                    # Пустая строка — станция по умолчанию (Европа Плюс):
                    # в «включи музыку» названия станции нет.
                    radio = self._start_radio(ctx, "")
                    if radio and "Не получилось" not in radio:
                        return (
                            "Music.app не разрешён (нужна автоматизация в "
                            "Системных настройках), поэтому включила радио. "
                            + radio
                        )
                return answer
            return self._search(ctx, value)
        return None

    # -- Music.app -----------------------------------------------------------

    def _simple(self, ctx: PluginContext, action: str, text: str) -> str:
        script = {
            "play": _PLAY_SCRIPT,
            "pause": _PAUSE_SCRIPT,
            "resume": _PLAY_SCRIPT,
            "next": _NEXT_SCRIPT,
            "prev": _PREV_SCRIPT,
            "stop": _STOP_SCRIPT,
        }[action]
        try:
            self._osascript(ctx, script)
        except MusicUnavailable as exc:
            return _friendly_error(exc)
        return text

    def _volume(self, ctx: PluginContext, *, up: bool) -> str:
        script = _VOLUME_UP_SCRIPT if up else _VOLUME_DOWN_SCRIPT
        try:
            self._osascript(ctx, script)
        except MusicUnavailable as exc:
            return _friendly_error(exc)
        return "Сделала громче." if up else "Сделала тише."

    def _set_volume(self, ctx: PluginContext, script: str, text: str) -> str:
        try:
            self._osascript(ctx, script)
        except MusicUnavailable as exc:
            return _friendly_error(exc)
        return text

    def _now_playing(self, ctx: PluginContext) -> str:
        try:
            output = self._osascript(ctx, _NOW_SCRIPT)
        except MusicUnavailable as exc:
            return _friendly_error(exc)
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        if not lines:
            return "Сейчас ничего не играет."
        state = lines[0].lower()
        if state == "stopped":
            return "Сейчас ничего не играет."
        detail = lines[1] if len(lines) > 1 else ""
        if not detail:
            return "Сейчас ничего не играет."
        prefix = "На паузе: " if state == "paused" else "Играет: "
        return f"{prefix}{detail}."

    def _search(self, ctx: PluginContext, query: str) -> str:
        try:
            output = self._osascript(ctx, _search_script(query))
        except MusicUnavailable as exc:
            return _friendly_error(exc)
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        marker = lines[0].lower() if lines else ""
        name = lines[1] if len(lines) > 1 else ""
        label = _display(query)
        if marker == "not-found":
            return f"Не нашла в библиотеке «{label}»."
        if marker == "playlist":
            return f"Включаю плейлист «{name or label}»."
        if marker in ("track", "artist"):
            return f"Включаю «{name or label}»."
        return f"Включаю «{label}»."

    def _osascript(self, ctx: PluginContext, script: str) -> str:
        """Запустить ``osascript`` без shell; ошибки — :class:`MusicUnavailable`."""
        try:
            completed = subprocess.run(
                [OSASCRIPT, "-e", script],
                capture_output=True,
                text=True,
                timeout=self._timeout(ctx),
                check=False,
            )
        except FileNotFoundError as exc:
            raise MusicUnavailable(str(exc), kind="missing") from exc
        except subprocess.TimeoutExpired as exc:
            raise MusicUnavailable("osascript timeout", kind="timeout") from exc
        except OSError as exc:
            raise MusicUnavailable(str(exc), kind="error") from exc
        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            if self._is_permission_error(stderr):
                kind = "permission"
            elif self._is_object_error(stderr):
                kind = "object"
            else:
                kind = "error"
            raise MusicUnavailable(stderr or f"osascript exit {completed.returncode}", kind=kind)
        return (completed.stdout or "").strip()

    @staticmethod
    def _is_permission_error(stderr: str) -> bool:
        low = stderr.lower()
        return any(marker in low for marker in _PERMISSION_MARKERS)

    @staticmethod
    def _is_object_error(stderr: str) -> bool:
        low = stderr.lower()
        return any(marker in low for marker in _OBJECT_MARKERS)

    def _timeout(self, ctx: PluginContext) -> float:
        raw: Any = DEFAULT_TIMEOUT
        getter = getattr(ctx, "setting", None)
        if callable(getter):
            try:
                raw = getter("timeout", DEFAULT_TIMEOUT)
            except Exception:  # noqa: BLE001 - настройка не должна ломать ответ
                raw = DEFAULT_TIMEOUT
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return DEFAULT_TIMEOUT
        return max(1.0, min(30.0, value))

    # -- радио (afplay) ------------------------------------------------------

    def _start_radio(self, ctx: PluginContext, station_text: str) -> str:
        station = self._find_station(ctx, station_text)
        if station is None:
            names = ", ".join(item.name for item in _RADIO_STATIONS)
            return f"Такой радиостанции у меня нет — есть {names}."
        self._stop_radio()
        try:
            process = subprocess.Popen(
                [AFPLAY, station.url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except FileNotFoundError:
            return "Не получилось включить радио: на этом Mac нет afplay."
        except OSError:
            return "Не получилось включить радио: afplay не запустился."
        self._radio = process
        self._radio_station = station
        self._radio_paused = False
        # afplay падает сразу, если не смог открыть поток (битый URL, нет
        # сети, AudioFileOpen). Даём ему мгновение, чтобы не обещать музыку
        # в пустоту; нормальный поток продолжает играть и poll() = None.
        time.sleep(0.3)
        try:
            exited = process.poll() is not None
        except Exception:  # noqa: BLE001 - странный Popen не должен ломать ответ
            exited = False
        if exited:
            self._radio = None
            self._radio_station = None
            return "Не получилось включить радио: поток не открылся."
        return f"Включаю радио {station.name}."

    def _find_station(
        self, ctx: PluginContext, station_text: str
    ) -> RadioStation | None:
        text = _normalize(station_text)
        if not text:
            key = "europa"
            getter = getattr(ctx, "setting", None)
            if callable(getter):
                try:
                    key = str(getter("default_radio", "europa"))
                except Exception:  # noqa: BLE001
                    key = "europa"
            for station in _RADIO_STATIONS:
                if station.key == key:
                    return station
            return _RADIO_STATIONS[0]
        for station in _RADIO_STATIONS:
            if any(alias in text or text in alias for alias in station.aliases):
                return station
        return None

    def _radio_active(self) -> bool:
        process = self._radio
        if process is None:
            return False
        try:
            running = process.poll() is None
        except Exception:  # noqa: BLE001
            running = False
        if not running:
            self._radio = None
            self._radio_station = None
            self._radio_paused = False
        return bool(running)

    def _pause_radio(self) -> bool:
        process = self._radio
        if process is None:
            return False
        try:
            process.send_signal(signal.SIGSTOP)
        except (ProcessLookupError, OSError):
            return False
        self._radio_paused = True
        return True

    def _resume_radio(self) -> bool:
        process = self._radio
        if process is None:
            return False
        try:
            process.send_signal(signal.SIGCONT)
        except (ProcessLookupError, OSError):
            return False
        self._radio_paused = False
        return True

    def _stop_radio(self) -> bool:
        process = self._radio
        self._radio = None
        self._radio_station = None
        self._radio_paused = False
        if process is None:
            return False
        try:
            if process.poll() is None:
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
    def _log(ctx: PluginContext, message: str, *args: Any) -> None:
        log = getattr(ctx, "log", None)
        if log is not None:
            try:
                log.debug(message, *args)
            except Exception:  # noqa: BLE001
                pass


_PLUGIN = MusicPlugin()


def answer(ctx: PluginContext, question: str) -> str | None:
    """Точка входа способности ``answer``: короткий текст или ``None``."""
    return _PLUGIN.answer(ctx, question)


def register(ctx: PluginContext) -> Any:
    """Вернуть объект с ``answer(ctx, question)`` (реестр зовёт его)."""
    ctx.log.debug("music: готов, озвучка команд без модели")
    return _PLUGIN
