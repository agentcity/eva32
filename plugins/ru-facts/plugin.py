"""Плагин ``ru-facts``: убирает служебные пометки фактов из ответа.

Задача 0212. Модель повторяет в ответе служебное — ``[Факт: …]``,
``СПРАВКА:``, ``(источник: …)`` — и пользователь видит внутреннюю кухню.
Плагин вырезает эти пометки из готового текста (``run``) и из потока
(``feed``/``flush``). Причина, почему это плагин, а не правка промпта:
инструкция «не цитируй» снижает частоту, но не даёт гарантии, а фильтр даёт
её и работает одинаково в веб-чате и в ``tools/chat.py`` (оба зовут
``PluginRegistry.stream_filter(scope="chat")``).

Поток — не построчный: маркер может разрезаться между кусками модели
(«[Фа», «кт: 1 USD»). Поэтому хвост, который может оказаться началом
маркера, придерживается до следующего куска (``_hold_index``), а незакрытые
``[Факт`` / ``(источник`` — до закрывающей скобки. Буфер ограничен: лучше
выпустить редкий обрезанный хвост, чем задержать весь ответ.

``scope: all``: служебное не должно попадать ни в чат, ни в озвучку.
"""

from __future__ import annotations

import re
from typing import Any

from eva.plugins.context import PluginContext

__all__ = ["FactsFilter", "register", "strip_service"]

#: Заголовок секции с подписью-инструкцией. Двоеточие обязательно: без него
#: «Справка о доходах — это документ» — обычный ответ, а не служебная метка.
#: Снимаем только заголовок, а не строку целиком: данные могут стоять на той же
#: строке. Перевод строки убираем, если заголовок занимал её один.
_HEADER_LINE_RE = re.compile(
    r"(?im)^[ \t]*СПРАВКА[ \t]*\([^\n)]*\)[ \t]*:[ \t]*\n?"
)

#: ``[Факт: …]``: закрытая скобка (внутри может быть «(источник: …)»).
_CLOSED_FACT_RE = re.compile(r"[ \t]*\[Факт[^\]\n]*\][ \t]*", re.IGNORECASE)
#: ``[Факт: …`` без закрывающей скобки (модель оборвала строку).
_DANGLING_FACT_RE = re.compile(r"[ \t]*\[Факт[^\]\n]*", re.IGNORECASE)
#: ``(источник: …)`` — источник без квадратных скобок.
_CLOSED_SOURCE_RE = re.compile(
    r"[ \t]*\(\s*источник[^)\n]*\)[ \t]*", re.IGNORECASE
)
_DANGLING_SOURCE_RE = re.compile(r"[ \t]*\(\s*источник[^)\n]*", re.IGNORECASE)
#: Слово «СПРАВКА» с двоеточием (и, возможно, подписью в скобках) — в любом
#: месте строки. Без двоеточия это обычное слово: «справка о доходах» не
#: служебная метка и должна остаться в ответе.
_SECTION_TOKEN_RE = re.compile(
    r"СПРАВКА[ \t]*(?:\([^)\n]*\))?[ \t]*:[ \t]*", re.IGNORECASE
)

#: Начала маркеров: по ним решается, что хвост потока надо придержать.
_MARKER_STARTS = ("[факт", "справка", "(источник")

#: Потолок удержания хвоста: длиннее — уже не маркер, а текст ответа.
_MAX_HOLD = 600


def _remove_markers(text: str) -> str:
    """Снять служебные пометки, не трогая остальной текст."""
    value = _HEADER_LINE_RE.sub("", str(text or ""))
    value = _CLOSED_FACT_RE.sub("", value)
    value = _CLOSED_SOURCE_RE.sub("", value)
    value = _SECTION_TOKEN_RE.sub("", value)
    return value


def strip_service(text: str) -> str:
    """Убрать служебные пометки из целого ответа и прибрать пустые строки."""
    value = _remove_markers(text)
    if not value:
        return value
    # Незакрытые пометки: в целом тексте продолжения уже не будет.
    value = _DANGLING_FACT_RE.sub("", value)
    value = _DANGLING_SOURCE_RE.sub("", value)
    # Остаток заголовка без данных: модель иногда отдаёт одно слово
    # «СПРАВКА» (поток оборвался до двоеточия). Пользователю это видеть нельзя.
    value = re.sub(
        r"(?im)^[ \t]*СПРАВКА(?:[ \t]*\([^)\n]*\))?[ \t]*:?[ \t]*$",
        "",
        value,
    )
    value = re.sub(r"[ \t]+\n", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _hold_index(text: str) -> int:
    """Сколько символов хвоста придержать до следующего куска потока."""
    lower = text.lower()
    hold = 0
    # 1. Незакрытая пометка: ждём закрывающую скобку.
    for start, closing in (("[факт", "]"), ("(источник", ")")):
        position = lower.rfind(start)
        if position >= 0 and closing not in text[position:]:
            hold = max(hold, len(text) - position)
    # 2. Секция «СПРАВКА»: держим всю строку до закрывающей «):» или конца
    #    строки. Проверяем именно начало текущей строки: слово «СПРАВКА»
    #    встречается ещё и внутри подписи («не цитируй слово «СПРАВКА»»), и
    #    искать последнее вхождение — значит выпустить полстроки в поток.
    line_start = text.rfind("\n") + 1
    line = text[line_start:].lstrip()
    if line.lower().startswith("справка"):
        after = line[len("справка") :].lstrip()
        # Держим только возможный заголовок ``СПРАВКА (…):``. Голое слово
        # «справка» — обычный текст; его удерживает только общий случай ниже
        # (когда кусок оборвался прямо на слове).
        if after.startswith("(") and "):" not in line:
            hold = max(hold, len(text) - line_start)
    # Слово «СПРАВКА», оборванное в конце куска: следующим может прийти «:».
    # Без удержания «СПРАВКА» уже ушла бы в поток, а «:» осталась бы сиротой.
    word = re.search(r"(?<![а-яёa-z])справка\s*$", text, re.IGNORECASE)
    if word is not None:
        hold = max(hold, len(text) - word.start())
    if hold > _MAX_HOLD:
        hold = 0
    # 3. Маркер разрезан между кусками: держим его начало.
    for marker in _MARKER_STARTS:
        for size in range(min(len(marker) - 1, len(text)), 0, -1):
            if lower.endswith(marker[:size]):
                hold = max(hold, size)
                break
    return min(hold, len(text))


class _FactsStream:
    """Одна потоковая стадия: снимает пометки, держа разрезанный маркер."""

    def __init__(self) -> None:
        self._buffer = ""
        self._started = False

    def feed(self, text: str) -> str:
        if not isinstance(text, str) or not text:
            return ""
        self._buffer += text
        return self._drain(final=False)

    def flush(self) -> str:
        return self._drain(final=True)

    def _drain(self, *, final: bool) -> str:
        buffered = self._buffer
        if not buffered:
            return ""
        if final:
            self._buffer = ""
            return strip_service(buffered)
        hold = _hold_index(buffered)
        cut = len(buffered) - hold
        self._buffer = buffered[cut:]
        emitted = _remove_markers(buffered[:cut])
        if emitted and not self._started:
            # Снятый заголовок оставляет пустую строку в начале ответа —
            # пользователю она не нужна.
            emitted = emitted.lstrip()
            self._started = bool(emitted)
        return emitted


class FactsFilter:
    """Фильтр ``filter.run`` для целого текста + фабрика потоковых стадий."""

    scope = "all"
    priority = 40

    def run(self, text: str) -> str:
        if not isinstance(text, str) or not text:
            return text
        return strip_service(text)

    def new_stream(self) -> _FactsStream:
        return _FactsStream()


_PLUGIN = FactsFilter()


def register(ctx: PluginContext) -> Any:
    ctx.log.debug("ru-facts: фильтр служебных пометок готов")
    return _PLUGIN
