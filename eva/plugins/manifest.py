"""Манифест плагина ``plugin.json``: чтение, проверка, версия API.

Манифест — единственное, что ядро знает о плагине до загрузки кода. Поэтому
он проверяется строго: без ``id``/``name``/``version``/``entry``/``capabilities``
плагин не загрузится, а ``api`` с чужой версией отклоняется с понятным
текстом («ядро поддерживает api=1»). Код плагина при этом даже не
исполняется.

Формат::

    {
      "id": "weather",
      "name": "Погода",
      "version": "1.0.0",
      "api": 1,
      "entry": "plugin.py",
      "capabilities": ["skill.fetch"],
      "permissions": {"network": ["wttr.in"], "filesystem": []},
      "settings": {"city": "Москва"},
      "description": "Текущая погода через wttr.in",
      "owner": "eva32",
      "priority": 40,
      "scope": "chat",
      "stateful": false
    }
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from eva.plugins.context import Permissions

__all__ = [
    "API_VERSION",
    "CAPABILITIES",
    "MANIFEST_NAME",
    "Manifest",
    "ManifestError",
    "load_manifest",
    "parse_manifest",
]

#: Версия контракта плагинов. Меняется только вместе с ADR и ядром.
API_VERSION = 1

#: Имя манифеста в каталоге плагина.
MANIFEST_NAME = "plugin.json"

#: Точки расширения, которые понимает ядро api=1. Неизвестная возможность —
#: ошибка манифеста: плагин, который «умеет» несуществующее, не должен
#: тихо грузиться и не делать ничего.
CAPABILITIES = frozenset(
    {"skill.fetch", "filter.run", "tts.engine", "command"}
)

#: Области действия фильтра: чат, озвучка или оба.
SCOPES = frozenset({"chat", "tts", "all"})

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_HOST_RE = re.compile(r"^(?:\*\.)?[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?$", re.I)


class ManifestError(ValueError):
    """Манифест невалиден; текст — для человека, который пишет плагин."""


@dataclass(frozen=True)
class Manifest:
    """Проверенный ``plugin.json`` плюс каталог, из которого он прочитан."""

    id: str
    name: str
    version: str
    api: int
    entry: str
    capabilities: tuple[str, ...]
    permissions: Permissions
    description: str = ""
    owner: str = ""
    settings: Mapping[str, Any] = field(default_factory=dict)
    #: Порядок вызова: меньше — раньше (факты и фильтры). По умолчанию 100.
    priority: int = 100
    #: Где применять фильтр: ``chat``/``tts``/``all`` (по умолчанию all).
    scope: str = "all"
    #: Плагин держит состояние и не умеет выключаться на лету.
    stateful: bool = False
    path: Path | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "api": self.api,
            "entry": self.entry,
            "capabilities": list(self.capabilities),
            "permissions": {
                "network": list(self.permissions.network),
                "filesystem": list(self.permissions.filesystem),
            },
            "description": self.description,
            "owner": self.owner,
            "settings": dict(self.settings),
            "priority": self.priority,
            "scope": self.scope,
            "stateful": self.stateful,
        }


def _text(value: Any, field_name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ManifestError(f"поле {field_name}: ожидалась строка")
    result = value.strip()
    if not result and not allow_empty:
        raise ManifestError(f"поле {field_name}: пустая строка")
    return result


def _hosts(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, (list, tuple)):
        raise ManifestError("permissions.network: ожидался список хостов")
    hosts: list[str] = []
    for item in raw:
        host = _text(item, "permissions.network[]").lower().rstrip(".")
        if "://" in host or "/" in host or ":" in host or " " in host:
            raise ManifestError(
                "permissions.network: {!r} — нужен только хост без схемы, "
                "порта и пути (например, example.com или *.example.com)".format(
                    item
                )
            )
        if not _HOST_RE.match(host):
            raise ManifestError(
                f"permissions.network: {item!r} — неверное имя хоста"
            )
        if host not in hosts:
            hosts.append(host)
    return tuple(hosts)


def _paths(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, (list, tuple)):
        raise ManifestError("permissions.filesystem: ожидался список путей")
    paths: list[str] = []
    for item in raw:
        path = _text(item, "permissions.filesystem[]")
        if "\x00" in path:
            raise ManifestError("permissions.filesystem: NUL в пути")
        if path not in paths:
            paths.append(path)
    return tuple(paths)


def _permissions(raw: Any) -> Permissions:
    if raw is None:
        return Permissions()
    if not isinstance(raw, Mapping):
        raise ManifestError("permissions: ожидался объект")
    unknown = sorted(set(raw) - {"network", "filesystem"})
    if unknown:
        raise ManifestError(
            "permissions: неизвестные ключи {}".format(", ".join(unknown))
        )
    return Permissions(network=_hosts(raw.get("network")), filesystem=_paths(raw.get("filesystem")))


def _capabilities(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ManifestError(
            "capabilities: нужен непустой список возможностей ("
            + ", ".join(sorted(CAPABILITIES))
            + ")"
        )
    result: list[str] = []
    for item in raw:
        name = _text(item, "capabilities[]")
        if name not in CAPABILITIES:
            raise ManifestError(
                "capabilities: неизвестная возможность {!r}; ядро api={} "
                "понимает: {}".format(
                    name, API_VERSION, ", ".join(sorted(CAPABILITIES))
                )
            )
        if name not in result:
            result.append(name)
    return tuple(result)


def _settings(raw: Any) -> Mapping[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ManifestError("settings: ожидался объект со значениями по умолчанию")
    return dict(raw)


def parse_manifest(payload: Any, *, path: Path | None = None) -> Manifest:
    """Проверить уже разобранный JSON и вернуть :class:`Manifest`."""
    if not isinstance(payload, Mapping):
        raise ManifestError("plugin.json: ожидался JSON-объект")
    raw_id = payload.get("id")
    plugin_id = _text(raw_id, "id")
    if not _ID_RE.match(plugin_id):
        raise ManifestError(
            "id: {!r} — допустимы строчные латинские буквы, цифры, '-' и '_' "
            "(до 64 символов, начинается с буквы/цифры)".format(raw_id)
        )
    name = _text(payload.get("name"), "name")
    version = _text(payload.get("version"), "version")

    api = payload.get("api")
    if isinstance(api, bool) or not isinstance(api, int):
        raise ManifestError("api: ожидалось целое число (сейчас 1)")
    if api != API_VERSION:
        raise ManifestError(
            "плагин {}: api={}, а ядро поддерживает api={}".format(
                plugin_id, api, API_VERSION
            )
        )

    entry = payload.get("entry", "plugin.py")
    entry = _text(entry, "entry")
    entry_path = Path(entry)
    if entry_path.is_absolute() or ".." in entry_path.parts:
        raise ManifestError(
            "entry: {!r} — нужен относительный путь внутри каталога плагина".format(
                entry
            )
        )

    capabilities = _capabilities(payload.get("capabilities"))
    permissions = _permissions(payload.get("permissions"))
    description = _text(
        payload.get("description", ""), "description", allow_empty=True
    )
    owner = _text(payload.get("owner", ""), "owner", allow_empty=True)

    raw_priority = payload.get("priority", 100)
    if isinstance(raw_priority, bool) or not isinstance(raw_priority, int):
        raise ManifestError("priority: ожидалось целое число")
    priority = int(raw_priority)

    scope = _text(payload.get("scope", "all"), "scope").lower()
    if scope not in SCOPES:
        raise ManifestError(
            "scope: {!r} — допустимо: {}".format(scope, ", ".join(sorted(SCOPES)))
        )

    stateful = payload.get("stateful", False)
    if not isinstance(stateful, bool):
        raise ManifestError("stateful: ожидалось true или false")

    unknown = sorted(
        set(payload)
        - {
            "id",
            "name",
            "version",
            "api",
            "entry",
            "capabilities",
            "permissions",
            "description",
            "owner",
            "settings",
            "priority",
            "scope",
            "stateful",
        }
    )
    # Неизвестные поля не ломают загрузку: их может писать более новое
    # поколение плагина, а ядро api=1 просто не знает, что с ними делать.
    # Но в описании манифеста они видны как есть (settings), а не молча
    # теряются в коде.
    _ = unknown

    return Manifest(
        id=plugin_id,
        name=name,
        version=version,
        api=api,
        entry=entry,
        capabilities=capabilities,
        permissions=permissions,
        description=description,
        owner=owner,
        settings=_settings(payload.get("settings")),
        priority=priority,
        scope=scope,
        stateful=stateful,
        path=path,
    )


def load_manifest(plugin_dir: Path | str) -> Manifest:
    """Прочитать ``<plugin_dir>/plugin.json`` и проверить его."""
    directory = Path(plugin_dir)
    path = directory / MANIFEST_NAME
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ManifestError(f"нет {MANIFEST_NAME} в {directory}: {exc}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ManifestError(
            "{}: не JSON ({}), строка {}".format(
                path, exc.msg, exc.lineno
            )
        ) from exc
    return parse_manifest(payload, path=directory)
