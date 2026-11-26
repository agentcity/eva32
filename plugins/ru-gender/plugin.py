"""Плагин ``ru-gender``: женский род Евы в первом лице.

Обёртка над :mod:`eva.ru_gender` (задача 0207): ``feminize`` для целого
текста и :class:`eva.ru_gender.GenderStreamer` для потока. Здесь нет своей
таблицы форм — только контракт рантайма.

``new_stream()`` обязателен: у стримера есть состояние (недописанное слово,
хвост бэктиков), и два одновременных ответа не должны делить одно состояние.
"""

from __future__ import annotations

from typing import Any

from eva.plugins.context import PluginContext
from eva.ru_gender import GenderStreamer, feminize

__all__ = ["GenderFilter", "register"]


class _GenderStream:
    """Одна потоковая стадия: обёртка над ``GenderStreamer``."""

    def __init__(self) -> None:
        self._streamer = GenderStreamer()

    def feed(self, text: str) -> str:
        if not isinstance(text, str) or not text:
            return ""
        return self._streamer.feed(text)

    def flush(self) -> str:
        return self._streamer.flush()


class GenderFilter:
    """Фильтр ``filter.run`` для целого текста + фабрика потоковых стадий."""

    scope = "all"
    priority = 30

    def run(self, text: str) -> str:
        if not isinstance(text, str) or not text:
            return text
        return feminize(text)

    def new_stream(self) -> _GenderStream:
        return _GenderStream()


_PLUGIN = GenderFilter()


def register(ctx: PluginContext) -> Any:
    ctx.log.debug("ru-gender: фильтр готов")
    return _PLUGIN
