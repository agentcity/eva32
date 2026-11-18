"""Скилл ``time``: текущее время и дата без модели и без сети.

Время берётся из системных часов, по умолчанию — московское (``Europe/Moscow``,
без перехода на летнее время): так ответ одинаков на любом хосте, а в
``route.reason`` видно, что модель не вызывалась.

«Сейчас» подменяется через ``ctx["now"]`` (``datetime``, ``date`` или
ISO-строка) — это нужно тестам, стенду и повторяемым прогонам. Без подмены
скилл использует ``datetime.now``.

Ответ не берётся из каталога, поэтому ``sources`` пустой: приписывать
вычисленному времени источник ``[1]`` было бы ложью.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone
from typing import Any

from eva.catalog import normalize
from eva.skills.base import Skill, SkillResult, make_result

__all__ = ["MONTHS", "MOSCOW_TZ", "WEEKDAYS", "TimeSkill"]

try:  # zoneinfo есть в stdlib 3.9+, но tzdata может отсутствовать
    from zoneinfo import ZoneInfo

    MOSCOW_TZ: Any = ZoneInfo("Europe/Moscow")
except Exception:  # pragma: no cover — среда без tzdata
    # Москва живёт в UTC+3 круглый год: запасной вариант не врёт.
    MOSCOW_TZ = timezone(timedelta(hours=3))

#: Месяцы в родительном падеже: «18 ноября 2026 года».
MONTHS = (
    "",
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
)

#: Дни недели, понедельник = 0.
WEEKDAYS = (
    "понедельник",
    "вторник",
    "среда",
    "четверг",
    "пятница",
    "суббота",
    "воскресенье",
)

#: «Который час», «сколько сейчас времени», «подскажи время».
_TIME_RE = re.compile(
    r"(?:"
    r"котор\w*\s+(?:сейчас\s+)?час\w*"
    r"|сколько\s+(?:сейчас\s+)?врем\w*"
    r"|(?:текущ\w+|точн\w+|местн\w+|московск\w+)\s+врем\w*"
    r"|врем\w*\s+(?:сейчас|у\s+вас)"
    r"|подскаж\w*\s+(?:сейчас\s+)?(?:точн\w+\s+)?врем\w*"
    r")",
    re.IGNORECASE,
)

#: «Какое сегодня число», «какой день недели», «какая дата».
_DATE_RE = re.compile(
    r"(?:"
    r"как(?:ое|ой|ая)\s+(?:сегодня\s+)?(?:число|дата|день(?:\s+недели)?)"
    r"|(?:сегодняшн\w+|текущ\w+)\s+(?:дата|число|день(?:\s+недели)?)"
    r"|какой\s+сегодня\s+день(?:\s+недели)?"
    r"|какое\s+число"
    r")",
    re.IGNORECASE,
)

#: «Сколько времени» — не всегда про часы: чаще это длительность.
#: Такие вопросы остаются модели (или скиллу срока из каталога).
_DURATION_RE = re.compile(
    r"(?:"
    r"занима\w*|займ\w*|займу\w*|длит\w*|длят\w*|потребу\w*"
    r"|остал\w*|ждать|ждем|ждём|пути|ехать|лететь"
    r"|работа\w*|офис\w*|поддержк\w*|техподдержк\w*"
    r"|тариф\w*|доставк\w*|возврат\w*|подписк\w*"
    r")",
    re.IGNORECASE,
)


class TimeSkill(Skill):
    """Отвечает на «который час» и «какое сегодня число»."""

    name = "time"

    def match(self, text: str, ctx: dict[str, Any] | None = None) -> bool:
        if not text or not text.strip():
            return False
        value = normalize(text)
        if _DURATION_RE.search(value):
            return False
        return bool(_TIME_RE.search(value) or _DATE_RE.search(value))

    def run(self, text: str, ctx: dict[str, Any]) -> SkillResult | None:
        if not self.match(text, ctx):
            return None
        value = normalize(text)
        asked_time = bool(_TIME_RE.search(value))
        asked_date = bool(_DATE_RE.search(value))
        now = _now(ctx)

        parts: list[str] = []
        if asked_date:
            parts.append(
                f"Сегодня {WEEKDAYS[now.weekday()]}, "
                f"{now.day} {MONTHS[now.month]} {now.year} года."
            )
        if asked_time:
            parts.append(f"Сейчас {now.hour:02d}:{now.minute:02d} по московскому времени.")
        if not parts:  # pragma: no cover — match() уже гарантировал вопрос
            return None
        return make_result(self.name, " ".join(parts), sources=[])


def _now(ctx: dict[str, Any] | None) -> datetime:
    """Текущее время: ``ctx["now"]``, ``ctx["meta"]["now"]`` или системные часы.

    Первый способ — для тестов и прямых вызовов, второй — для замера
    (``tools/eval_hybrid.py`` кладёт фиксированное «сейчас» в meta запроса,
    иначе вопросы о времени в eval были бы невоспроизводимы).
    """
    context = ctx or {}
    raw = context.get("now")
    if raw is None and isinstance(context.get("meta"), dict):
        raw = context["meta"].get("now")
    moment: datetime | None = None
    if isinstance(raw, datetime):
        moment = raw
    elif isinstance(raw, date):
        moment = datetime(raw.year, raw.month, raw.day)
    elif isinstance(raw, str) and raw.strip():
        try:
            moment = datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
        except ValueError:
            moment = None
    if moment is None:
        return datetime.now(MOSCOW_TZ)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=MOSCOW_TZ)
    return moment.astimezone(MOSCOW_TZ)
