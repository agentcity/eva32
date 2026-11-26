"""Плагин ``ru-numbers``: числа в русские слова **перед озвучкой**.

Это обёртка над :func:`eva.ru_numbers.numbers_to_words` (задача 0205), а не
вторая реализация. ``scope: tts`` — в чате цифры остаются цифрами (курс
«78,12 ₽» читать словами в тексте не нужно), а Silero/ElevenLabs/say
получают слова: цифры они не произносят.

Рантайм зовёт ``run(text) -> text`` для целой фразы. Потоковый ``feed`` не
нужен: TTS синтезирует фразу целиком.
"""

from __future__ import annotations

from typing import Any

from eva.plugins.context import PluginContext
from eva.ru_numbers import numbers_to_words

__all__ = ["NumbersFilter", "register"]


class NumbersFilter:
    """Фильтр ``filter.run``: цифры → слова. Идемпотентен."""

    scope = "tts"
    priority = 20

    def run(self, text: str) -> str:
        if not isinstance(text, str) or not text:
            return text
        return numbers_to_words(text)


_PLUGIN = NumbersFilter()


def register(ctx: PluginContext) -> Any:
    ctx.log.debug("ru-numbers: фильтр озвучки готов")
    return _PLUGIN
