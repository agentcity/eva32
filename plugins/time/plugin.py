"""Плагин ``time``: «который час» → реальное время сервера.

Оборачивает уже готовые модули ядра, а не дублирует их:

* :class:`eva.skills.clock.TimeSkill` — распознавание вопроса и запасной
  формат ответа (если 0208 ещё не влит);
* :func:`eva.skills.fresh.now_info` — формат факта «26 ноября 2026, 15:30 MSK»
  с источником «часы сервера» (0208).

Сети плагин не требует вовсе — в манифесте ``permissions.network: []``.
"""

from __future__ import annotations

from typing import Any

from eva.plugins.context import Fact, PluginContext
from eva.skills.clock import TimeSkill

try:  # 0208 может быть ещё не влит — тогда работаем на TimeSkill
    from eva.skills import fresh as _fresh
    from eva.skills import intent as _intent
except Exception:  # pragma: no cover - до загрузки 0208
    _fresh = None  # type: ignore[assignment]
    _intent = None  # type: ignore[assignment]

__all__ = ["TimePlugin", "register"]


class TimePlugin:
    """Ответ на вопрос о времени/дате одним фактом для системного блока."""

    name = "time"

    def __init__(self) -> None:
        self._skill = TimeSkill()

    def fetch(self, ctx: PluginContext, question: str) -> Fact | None:
        text = str(question or "").strip()
        if not text:
            return None
        # Намерение — из роутера 0208 (он знает больше формулировок); если
        # его нет, работает TimeSkill. Лишний раз не дублируем регулярки.
        with_weekday = False
        if _intent is not None:
            detected = _intent.detect(text)
            if detected is None or detected.kind != "time":
                return None
            with_weekday = bool(detected.asks_weekday)
        elif not self._skill.match(text):
            return None
        else:
            # «Какое сегодня число / день недели» — показываем день недели.
            with_weekday = any(
                word in text.lower()
                for word in ("день", "дата", "число", "недели")
            )
        if _fresh is not None:
            try:
                fact = _fresh.now_info(with_weekday=with_weekday)
                return Fact(
                    text=str(fact.data),
                    source=str(fact.source or "часы сервера"),
                    kind="time",
                )
            except Exception as exc:  # noqa: BLE001 - запасной путь ниже
                ctx.log.warning("fresh.now_info не сработал: %s", exc)
        result = self._skill.run(text, {})
        if not result:
            return None
        return Fact(
            text=str(result.get("text") or ""),
            source="часы сервера",
            kind="time",
        )


_PLUGIN = TimePlugin()


def register(ctx: PluginContext) -> Any:
    """Точка входа рантайма: вернуть объект с ``fetch(ctx, question)``."""
    ctx.log.debug("time: готов, сеть не нужна")
    return _PLUGIN
