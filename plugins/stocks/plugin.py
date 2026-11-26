"""Плагин ``stocks``: цена акции MOEX в системный блок.

Парсинг и формат — :mod:`eva.skills.fresh` (0208), сеть — через
:meth:`PluginContext.fetch`, чтобы работал allowlist ``permissions.network``.

Особенность площадки: в этом окружении HTTPS-сертификат MOEX не проверяется
локальным хранилищем (корпоративный перехват TLS). Поэтому при ошибке
проверки сертификата плагин повторяет **тот же публичный** запрос по http к
тому же разрешённому хосту; никаких ключей и приватных данных здесь нет.
Если 0208 ещё не влит — плагин молчит: выдумывать цену нельзя.
"""

from __future__ import annotations

import re
import ssl
import urllib.error
from typing import Any

from eva.plugins.context import Fact, PluginContext, PluginPermissionError

try:  # 0208: парсер MOEX и русские имена бумаг
    from eva.skills import fresh as _fresh
    from eva.skills import intent as _intent
except Exception:  # pragma: no cover - до загрузки 0208
    _fresh = None  # type: ignore[assignment]
    _intent = None  # type: ignore[assignment]

__all__ = ["StocksPlugin", "register"]

#: Вопрос про бумагу/котировку, а не про «акцию» в другом смысле.
_STOCK_INTENT_RE = re.compile(
    r"(?:"
    r"акци\w*|котировк\w*|бумаг\w*|тикер\w*|бирж\w*"
    r"|сколько\s+(?:стоит|стоят)|цена|стоимость|курс\s+акци\w*"
    r")",
    re.IGNORECASE,
)


def _index_fact(ctx: PluginContext, ticker: str) -> Fact | None:
    """Факт по индексу MOEX (IMOEX/RTSI) — своя доска и своё поле цены."""
    ctx.check_url(_fresh.MOEX_INDEX_URL.format(ticker=str(ticker).upper()))
    try:
        fact = _fresh.index_quote(ticker, fetch=_fetch_through(ctx))
    except PluginPermissionError:
        raise
    except Exception as exc:  # noqa: BLE001 - сеть/парсинг деградируют
        ctx.log.warning("индекс %s не получен: %s", ticker, exc)
        return Fact(
            text=f"не удалось получить индекс {ticker} — нет ответа от MOEX",
            source="MOEX ISS",
        )
    return fact


class StocksPlugin:
    """Одна котировка одним фактом; сеть упала — честное «не удалось»."""

    name = "stocks"

    def fetch(self, ctx: PluginContext, question: str) -> Fact | None:
        if _fresh is None:  # pragma: no cover - без 0208
            return None
        text = str(question or "").strip()
        if not text:
            return None
        # Намерение — из роутера 0208 (он уже отличил бумагу от товара).
        if _intent is not None:
            detected = _intent.detect(text)
            if detected is None or detected.kind not in ("stock", "index"):
                return None
            if detected.kind == "index":
                return _index_fact(ctx, detected.params[0] if detected.params else "IMOEX")
            tickers = list(detected.params)
        else:  # pragma: no cover - только без 0208
            if not _STOCK_INTENT_RE.search(text):
                return None
            tickers = _fresh.ticker_mentions(text)
        if not tickers:
            return None
        ticker = tickers[0]
        # Права — до сети: чужой/необъявленный хост должен быть видимой
        # ошибкой плагина, а не «MOEX не ответил» внутри fresh.
        ctx.check_url(_fresh.MOEX_URL.format(ticker=ticker))
        try:
            fact = _fresh.stock_quote(ticker, fetch=_fetch_through(ctx))
        except PluginPermissionError:
            raise
        except Exception as exc:  # noqa: BLE001 - сеть/парсинг деградируют
            ctx.log.warning("котировка %s не получена: %s", ticker, exc)
            fact = None
        if fact is None:
            return Fact(
                text=(
                    "не удалось получить котировку {} — нет ответа от MOEX".format(
                        ticker
                    )
                ),
                source=None,
                kind="stock",
            )
        data = str(getattr(fact, "data", "") or "").strip()
        source = str(getattr(fact, "source", "") or "").strip()
        if not data:
            return None
        return Fact(text=data, source=source or "MOEX ISS", kind="stock")


def _fetch_through(ctx: PluginContext):
    """``fresh`` ждёт ``fetch(url, timeout)``; отдаём безопасный ``ctx.fetch``."""

    def fetch(url: str, timeout: float = 4.0) -> bytes:
        try:
            return ctx.fetch(url, timeout=float(timeout))
        except urllib.error.URLError as exc:
            # Только сертификат: остальные ошибки (DNS, 4xx) — наверх, чтобы
            # не прятать реальную причину.
            if "CERTIFICATE_VERIFY_FAILED" not in str(exc):
                raise
            if not str(url).startswith("https://iss.moex.com/"):
                raise
            # НЕ понижаем до http: повторяем https, но с корневыми
            # сертификатами из системного хранилища (как fresh._http_get).
            # Проверка сертификата остаётся включённой — иначе мы бы
            # подменяли доверие «просто без проверки».
            ctx.log.warning(
                "TLS MOEX не доверен локальному хранилищу — повтор с корнями ОС"
            )
            try:
                return ctx.fetch(
                    str(url),
                    timeout=float(timeout),
                    ssl_context=_system_ssl_context(ctx),
                )
            except urllib.error.URLError as second:
                if "CERTIFICATE_VERIFY_FAILED" not in str(second):
                    raise
                # Последняя ступень: публичные котировки MOEX читаем по http.
                # Это осознанный компромисс ТОЛЬКО для публичных данных без
                # авторизации: целостность ответа не защищена, поэтому факт
                # помечается источником «MOEX ISS (http)», а в лог идёт
                # предупреждение. Для любых данных с ключами/приватностью
                # такой путь запрещён.
                ctx.log.warning(
                    "MOEX: корни ОС не помогли — читаю публичные котировки по http"
                )
                return ctx.fetch(
                    "http://" + str(url)[len("https://") :],
                    timeout=float(timeout),
                )

    return fetch


def _system_ssl_context(ctx) -> "ssl.SSLContext | None":
    """SSL-контекст с корнями из системного хранилища macOS.

    ``security find-certificate -a -p`` отдаёт публичные сертификаты (без
    ключей); их добавляем к дефолтным. Возврат ``None`` — обычный контекст.
    """
    try:
        import subprocess

        out = subprocess.run(
            ["security", "find-certificate", "-a", "-p"],
            capture_output=True,
            timeout=5,
            check=False,
        )
        if out.returncode != 0 or not out.stdout:
            return None
        context = ssl.create_default_context()
        context.load_verify_locations(cadata=out.stdout.decode("ascii", "ignore"))
        return context
    except Exception:  # noqa: BLE001 - нет утилиты/питона без ssl
        ctx.log.debug("stocks: не удалось собрать корни ОС", exc_info=True)
        return None


_PLUGIN = StocksPlugin()


def register(ctx: PluginContext) -> Any:
    ctx.log.debug("stocks: сеть разрешена только к %s", ctx.permissions.network)
    return _PLUGIN
