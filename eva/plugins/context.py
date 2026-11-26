"""Контекст плагина: настройки, лог и безопасные сеть/файлы.

Плагин получает :class:`PluginContext` при регистрации и **не должен** ходить
в сеть или в файлы напрямую: только через ``ctx.fetch`` / ``ctx.read_text`` и
только в пределах ``permissions`` из ``plugin.json``. Это best-effort
изоляция (без песочницы процесса), но она закрывает главный риск — плагин,
который незаметно лезет на чужой хост или читает чужой файл.

Контракт ошибок: выход за разрешения — не «тихий None», а
:class:`PluginPermissionError` с именем плагина и запрещённым хостом/путём.
Ядро ловит её, пишет в лог и показывает в ``last_error`` плагина; чат при
этом продолжает работать.
"""

from __future__ import annotations

import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

__all__ = [
    "MAX_FETCH_BYTES",
    "Fact",
    "PermissionDenied",
    "Permissions",
    "PluginContext",
    "PluginPermissionError",
]

#: Потолок ответа сети: плагину хватает курса/котировки/погоды, а гигабайты
#: в память чата не нужны.
MAX_FETCH_BYTES = 2 * 1024 * 1024

#: Таймаут сети: не больше, чем чат готов ждать факт. Верхняя граница — от
#: бесконечного зависания, нижняя — чтобы совсем не отключить запрос.
DEFAULT_TIMEOUT = 5.0
MAX_TIMEOUT = 30.0


class PluginPermissionError(PermissionError):
    """Плагин вышел за свои permissions.

    Сообщение всегда называет плагин и объект отказа: «плагин time: хост
    evil.example не разрешён (permissions.network)».
    """

    def __init__(self, plugin_id: str, message: str) -> None:
        self.plugin_id = plugin_id
        super().__init__(f"плагин {plugin_id}: {message}")


#: Короткий алиас для кода ядра и тестов.
PermissionDenied = PluginPermissionError


@dataclass(frozen=True)
class Fact:
    """Факт, который ядро подкладывает модели секцией ``СПРАВКА``.

    ``text`` — данные без обёртки; ``source`` — откуда они (ЦБ РФ, MOEX,
    часы сервера). ``source=None`` — источник не называется (например,
    деградация «не удалось получить»). ``plugin`` заполняет реестр, плагину
    его указывать не нужно.

    ``line()`` отдаёт **строку данных**, а не готовый блок: заголовок с
    командой модели печатается один раз на секцию
    (:func:`eva.skills.intent.fact_block`). Так строка годится и для системного
    блока, и для фактической подстановки стражем (задача 0212).
    """

    text: str
    source: str | None = None
    plugin: str = ""
    kind: str = ""

    def line(self) -> str:
        """Строка данных: ``1 USD = 82,3130 ₽ (ЦБ РФ на 26.11.2026)``."""
        text = str(self.text or "").strip()
        # Старый формат ``[Факт: …]`` из чужого плагина — снимаем обёртку,
        # чтобы внутри секции не оказалось второй служебной строки.
        if text.lower().startswith("[факт:"):
            text = text[len("[факт:") :].rstrip("]").strip()
        source = str(self.source or "").strip()
        if source:
            return f"{text} ({source})"
        return text


@dataclass(frozen=True)
class Permissions:
    """Разрешённые плагину хосты и пути из ``plugin.json``.

    ``network`` — имена хостов без схемы и порта: ``example.com`` или
    ``*.example.com`` (поддомены; сам ``example.com`` надо перечислить
    отдельно, если он тоже нужен).
    ``filesystem`` — пути: относительные считаются от каталога плагина,
    абсолютные — как есть. Всё, что за их пределами, запрещено.
    """

    network: tuple[str, ...] = ()
    filesystem: tuple[str, ...] = ()

    def allows_host(self, host: str) -> bool:
        value = str(host or "").strip().lower().rstrip(".")
        if not value:
            return False
        for pattern in self.network:
            allowed = pattern.strip().lower().rstrip(".")
            if not allowed:
                continue
            if allowed.startswith("*."):
                suffix = allowed[1:]  # ".example.com"
                if value.endswith(suffix) and value != suffix[1:]:
                    return True
            elif value == allowed:
                return True
        return False

    def paths_for(self, plugin_dir: Path) -> tuple[Path, ...]:
        """Разрешённые корни в абсолютном виде (относительно каталога плагина)."""
        roots: list[Path] = []
        base = Path(plugin_dir)
        for raw in self.filesystem:
            candidate = Path(raw)
            if not candidate.is_absolute():
                candidate = base / candidate
            roots.append(candidate.expanduser().resolve(strict=False))
        return tuple(roots)

    def allows_path(self, plugin_dir: Path, path: str | os.PathLike[str]) -> bool:
        target = Path(path).expanduser()
        if not target.is_absolute():
            target = Path(plugin_dir) / target
        try:
            resolved = target.resolve(strict=False)
        except OSError:  # pragma: no cover - защита от странной ФС
            return False
        for root in self.paths_for(plugin_dir):
            if resolved == root or root in resolved.parents:
                return True
        return False


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Редирект разрешён, только если новый хост тоже в allowlist."""

    def __init__(self, context: "PluginContext") -> None:
        self.context = context

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        self.context.check_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class PluginContext:
    """То, что плагин видит от ядра: настройки, лог, сеть и файлы.

    Плагин не импортирует ``urllib`` сам (в примерах документации — только
    ``ctx.fetch``): так права проверяются в одной точке. Если очень нужно
    что-то своё, это остаётся на совести автора плагина — изоляция
    best-effort, как и записано в ADR 0022.
    """

    def __init__(
        self,
        plugin_id: str,
        *,
        plugin_dir: Path,
        permissions: Permissions | None = None,
        settings: Mapping[str, Any] | None = None,
        logger: logging.Logger | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.plugin_id = str(plugin_id)
        self.plugin_dir = Path(plugin_dir)
        self.permissions = permissions or Permissions()
        self.settings: dict[str, Any] = dict(settings or {})
        self.log = logger or logging.getLogger(f"eva.plugins.{self.plugin_id}")
        self.timeout = float(timeout)

    # -- настройки -----------------------------------------------------------

    def setting(self, name: str, default: Any = None) -> Any:
        return self.settings.get(name, default)

    # -- сеть ----------------------------------------------------------------

    def check_url(self, url: str) -> urllib.parse.SplitResult:
        """Проверить URL до запроса/редиректа; вернуть разобранный URL."""
        if not isinstance(url, str) or not url.strip():
            raise PluginPermissionError(
                self.plugin_id, "пустой URL: сеть плагину недоступна"
            )
        parsed = urllib.parse.urlsplit(url.strip())
        if parsed.scheme not in ("http", "https"):
            raise PluginPermissionError(
                self.plugin_id,
                "схема {!r} запрещена (разрешены http/https)".format(parsed.scheme),
            )
        if parsed.username or parsed.password:
            raise PluginPermissionError(
                self.plugin_id, "URL с логином/паролем запрещён"
            )
        host = (parsed.hostname or "").strip().lower()
        if not host:
            raise PluginPermissionError(
                self.plugin_id, "в URL нет хоста: {!r}".format(url)
            )
        if not self.permissions.allows_host(host):
            allowed = ", ".join(self.permissions.network) or "сеть не разрешена"
            raise PluginPermissionError(
                self.plugin_id,
                "хост {} не разрешён (permissions.network: {})".format(
                    host, allowed
                ),
            )
        return parsed

    def fetch(
        self,
        url: str,
        *,
        timeout: float | None = None,
        headers: Mapping[str, str] | None = None,
        max_bytes: int = MAX_FETCH_BYTES,
        ssl_context: Any | None = None,
    ) -> bytes:
        """GET с проверкой хоста (и каждого редиректа) — байты ответа.

        Ошибки сети (URLError, HTTPError) пробрасываются как есть: плагин
        сам решает, деградировать или вернуть ``None``. Нарушение
        permissions — всегда :class:`PluginPermissionError`.
        """
        parsed = self.check_url(url)
        limit = self.timeout if timeout is None else float(timeout)
        limit = max(0.1, min(MAX_TIMEOUT, limit))
        headers_out = {
            "User-Agent": f"eva32-plugin/{self.plugin_id}",
            "Accept": "application/json, application/xml, text/html;q=0.9, */*;q=0.1",
        }
        if headers:
            headers_out.update({str(k): str(v) for k, v in headers.items()})
        request = urllib.request.Request(
            urllib.parse.urlunsplit(parsed), headers=headers_out
        )
        opener = urllib.request.build_opener(_SafeRedirectHandler(self))
        if ssl_context is not None:
            # Замена контекста проверки сертификата (например, корни из
            # системного хранилища). Проверка НЕ отключается: плагин передаёт
            # собранный контекст, а не «verify=False».
            opener.add_handler(urllib.request.HTTPSHandler(context=ssl_context))
        with opener.open(request, timeout=limit) as response:
            data = response.read(max(1, int(max_bytes)) + 1)
        if len(data) > max(1, int(max_bytes)):
            raise urllib.error.URLError(
                f"ответ {parsed.hostname} больше {max_bytes} байт"
            )
        return data

    # -- файлы ---------------------------------------------------------------

    def check_path(self, path: str | os.PathLike[str]) -> Path:
        if not self.permissions.allows_path(self.plugin_dir, path):
            allowed = ", ".join(self.permissions.filesystem) or "файлы не разрешены"
            raise PluginPermissionError(
                self.plugin_id,
                "путь {!r} вне permissions.filesystem ({})".format(
                    str(path), allowed
                ),
            )
        target = Path(path).expanduser()
        if not target.is_absolute():
            target = self.plugin_dir / target
        return target

    def read_bytes(self, path: str | os.PathLike[str]) -> bytes:
        return self.check_path(path).read_bytes()

    def read_text(
        self, path: str | os.PathLike[str], *, encoding: str = "utf-8"
    ) -> str:
        return self.check_path(path).read_text(encoding=encoding)

    def write_text(
        self,
        path: str | os.PathLike[str],
        text: str,
        *,
        encoding: str = "utf-8",
    ) -> None:
        target = self.check_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding=encoding)

    def __repr__(self) -> str:  # pragma: no cover - удобство логов
        return (
            f"PluginContext({self.plugin_id!r}, "
            f"network={list(self.permissions.network)!r})"
        )
