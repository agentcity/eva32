"""Плагин ``currency``: курс валют ЦБ РФ в системный блок.

Логика получения и парсинга — в :mod:`eva.skills.fresh` (задача 0208), здесь
только распознавание вопроса и **сеть через** :meth:`PluginContext.fetch`:
так работает allowlist ``permissions.network`` и видно ошибку плагина, а не
тихий поход в ``urllib``.

Если 0208 ещё не влит, плагин отвечает по встроенному справочнику
:class:`eva.skills.currency.CurrencySkill` и честно ставит дату среза.
"""

from __future__ import annotations

import re
from typing import Any

from eva.plugins.context import Fact, PluginContext, PluginPermissionError
from eva.skills.currency import CurrencySkill

try:  # 0208: реальные курсы с кэшем; без него — статический срез
    from eva.skills import fresh as _fresh
    from eva.skills import intent as _intent
except Exception:  # pragma: no cover - до загрузки 0208
    _fresh = None  # type: ignore[assignment]
    _intent = None  # type: ignore[assignment]

__all__ = ["CurrencyPlugin", "register"]

#: Вопрос про курс/обмен, а не про цену услуги из каталога.
_INTENT_RE = re.compile(
    r"(?:"
    r"курс\w*|котировк\w*|валют\w*"
    r"|сколько\s+(?:стоит|стоят|дают|дадут|будет|будет\s+в)"
    r"|по\s+ч[её]м|обмен\w*|перевед\w*|конверт\w*|кросс-курс\w*"
    r"|\brub\b"
    r")",
    re.IGNORECASE,
)

#: Каталожные вопросы — не наша зона (как в CurrencySkill).
_GUARD_RE = re.compile(
    r"(?:"
    r"тариф\w*|каталог\w*|доставк\w*|возврат\w*|подписк\w*|скидк\w*"
    r"|офис\w*|поддержк\w*|техподдержк\w*"
    r")",
    re.IGNORECASE,
)

#: «Курс валют» без конкретной валюты — показываем основные.
_DEFAULT_CODES = ("USD", "EUR", "CNY")


class CurrencyPlugin:
    """Курс одной-двух валют одним фактом; при сбое — честная деградация."""

    name = "currency"

    def __init__(self) -> None:
        self._fallback = CurrencySkill()

    def fetch(self, ctx: PluginContext, question: str) -> Fact | None:
        text = str(question or "").strip()
        if not text:
            return None
        # Сначала роутер 0208: он уже отличил курс от каталога и акции.
        if _intent is not None:
            detected = _intent.detect(text)
            if detected is None or detected.kind != "currency":
                return None
            codes = list(detected.params[:2]) or list(_DEFAULT_CODES)
        else:  # pragma: no cover - только без 0208
            if _GUARD_RE.search(text) or not _INTENT_RE.search(text):
                return None
            codes = self._codes(text)
        if not codes:
            return None
        if _fresh is None:
            return self._fallback_fact(text)
        return self._live_fact(ctx, codes)

    def _codes(self, text: str) -> list[str]:
        if _fresh is not None:
            found = [
                code
                for code in _fresh.currency_mentions(text)
                if code != "RUB"
            ]
        else:  # pragma: no cover - только без 0208
            code = _fresh_code(text)
            found = [code] if code and code != "RUB" else []
        if not found and re.search(r"курс\w*\s+валют", text, re.IGNORECASE):
            found = list(_DEFAULT_CODES)
        # В одном факте — не больше двух валют: остальное съест контекст.
        return found[:2]

    def _live_fact(self, ctx: PluginContext, codes: list[str]) -> Fact:
        assert _fresh is not None
        # Права проверяются до сети: если хост ЦБ не объявлен в манифесте,
        # это явная ошибка плагина, а не тихая деградация внутри fresh.
        ctx.check_url(_fresh.CBR_URL)
        lines: list[str] = []
        sources: list[str] = []
        failed: list[str] = []
        for code in codes:
            try:
                fact = _fresh.currency_rate(
                    code,
                    fetch=lambda url, timeout: ctx.fetch(
                        url, timeout=float(timeout)
                    ),
                )
            except PluginPermissionError:
                raise
            except Exception as exc:  # noqa: BLE001 - сеть/парсинг деградируют
                ctx.log.warning("курс %s не получен: %s", code, exc)
                fact = None
            if fact is None:
                failed.append(code)
                continue
            data = str(getattr(fact, "data", "") or "").strip()
            source = str(getattr(fact, "source", "") or "").strip()
            if data:
                lines.append(data)
            if source and source not in sources:
                sources.append(source)
        if not lines:
            return Fact(
                text="не удалось получить курс {} — нет ответа от ЦБ РФ".format(
                    ", ".join(failed or codes)
                ),
                source=None,
                kind="currency",
            )
        text = "; ".join(lines)
        if failed:
            text += "; курс {} получить не удалось".format(", ".join(failed))
        return Fact(
            text=text,
            source=", ".join(sources) if sources else "ЦБ РФ",
            kind="currency",
        )

    def _fallback_fact(self, text: str) -> Fact | None:  # pragma: no cover
        result = self._fallback.run(text, {})
        if not result:
            return None
        return Fact(
            text=str(result.get("text") or ""),
            source="встроенный справочник (без сети)",
            kind="currency",
        )


def _fresh_code(text: str) -> str | None:  # pragma: no cover - без 0208
    from eva.skills.currency import _mentions  # noqa: PLC0415

    mentions = _mentions(text)
    for _start, _end, code in mentions:
        if code != "RUB":
            return code
    return None


_PLUGIN = CurrencyPlugin()


def register(ctx: PluginContext) -> Any:
    ctx.log.debug("currency: сеть разрешена только к %s", ctx.permissions.network)
    return _PLUGIN
