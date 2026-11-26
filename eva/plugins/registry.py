"""Реестр плагинов: discovery, загрузка, включение, способности.

Жизненный цикл (ADR 0022):

1. discovery — обойти ``plugins/<id>/`` (и ``EVA_PLUGINS_DIR``), прочитать
   ``plugin.json``, отбросить битые манифесты с понятной ошибкой;
2. загрузка — только включённых по ``config/plugins.json``; код плагина
   исполняется один раз, контекст (:class:`~eva.plugins.context.PluginContext`)
   живёт вместе с плагином;
3. регистрация способностей — ``skill.fetch``, ``filter.run``,
   ``tts.engine``, ``command``;
4. вызовы — :meth:`PluginRegistry.collect_facts` для системного блока и
   :meth:`PluginRegistry.filter_text` / :meth:`stream_filter` для ответа;
5. включение/выключение — :meth:`set_enabled` без перезапуска, если плагин
   не объявил ``"stateful": true``.

Реестр не знает, что делает плагин: он только проверяет права, зовёт
функции и складывает ошибки в ``last_error``. Ошибка одного плагина не
останавливает чат.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Mapping, Sequence

from eva.plugins.context import (
    Fact,
    Permissions,
    PluginContext,
    PluginPermissionError,
)
from eva.plugins.manifest import (
    API_VERSION,
    Manifest,
    ManifestError,
    load_manifest,
)

__all__ = [
    "DEFAULT_FACT_LIMIT",
    "CommandBinding",
    "FilterBinding",
    "FilterPipeline",
    "PluginError",
    "PluginNotFound",
    "PluginRecord",
    "PluginRegistry",
    "PluginRestartRequired",
    "SkillBinding",
    "TtsBinding",
    "default_registry",
    "default_plugin_dirs",
]

logger = logging.getLogger("eva.plugins")

#: Сколько фактов максимум подкладывается в один запрос: больше — это уже
#: не «факт», а простыня, и она съедает контекст модели.
DEFAULT_FACT_LIMIT = 3

#: Имя пакета плагинов в ``sys.modules`` — с префиксом, чтобы не столкнуться
#: с обычными модулями.
_MODULE_PREFIX = "_eva_plugin_"


class PluginError(RuntimeError):
    """Плагин не загрузился или не может выполнить операцию."""

    def __init__(self, plugin_id: str, message: str) -> None:
        self.plugin_id = plugin_id
        super().__init__(f"плагин {plugin_id}: {message}")


class PluginNotFound(PluginError):
    """Плагина с таким ``id`` нет среди найденных."""

    def __init__(self, plugin_id: str) -> None:
        super().__init__(plugin_id, "не найден")


class PluginRestartRequired(PluginError):
    """Плагин держит состояние: выключение на лету невозможно."""

    def __init__(self, plugin_id: str) -> None:
        super().__init__(
            plugin_id,
            "держит состояние (stateful) и не выключается без перезапуска "
            "сервера",
        )


@dataclass
class PluginRecord:
    """Один найденный плагин: манифест, каталог, код и состояние."""

    manifest: Manifest
    path: Path
    enabled: bool = False
    loaded: bool = False
    module: ModuleType | None = None
    obj: Any = None
    ctx: PluginContext | None = None
    error: str = ""
    last_error: str = ""
    scope: str = "all"
    priority: int = 100

    @property
    def id(self) -> str:
        return self.manifest.id

    def info(self) -> dict[str, Any]:
        """Публичное описание для ``GET /api/plugins`` (без путей и кода)."""
        return {
            "id": self.id,
            "name": self.manifest.name,
            "version": self.manifest.version,
            "api": self.manifest.api,
            "enabled": self.enabled,
            "loaded": self.loaded,
            "capabilities": list(self.manifest.capabilities),
            "permissions": {
                "network": list(self.manifest.permissions.network),
                "filesystem": list(self.manifest.permissions.filesystem),
            },
            "description": self.manifest.description,
            "owner": self.manifest.owner,
            "priority": self.priority,
            "scope": self.scope,
            "stateful": self.manifest.stateful,
            "error": self.error or None,
            "last_error": self.last_error or None,
        }


@dataclass(frozen=True)
class SkillBinding:
    """Способность ``skill.fetch`` одного плагина."""

    plugin_id: str
    fetch: Callable[[PluginContext, str], Any]
    ctx: PluginContext
    priority: int = 100


@dataclass(frozen=True)
class FilterBinding:
    """Способность ``filter.run`` одного плагина."""

    plugin_id: str
    run: Callable[[str], str]
    obj: Any
    priority: int = 100
    scope: str = "all"
    new_stream: Callable[[], Any] | None = None


@dataclass(frozen=True)
class TtsBinding:
    """Способность ``tts.engine`` одного плагина (каркас)."""

    plugin_id: str
    synthesize: Callable[..., Any]
    ctx: PluginContext


@dataclass(frozen=True)
class CommandBinding:
    """Способность ``command`` одного плагина (каркас)."""

    plugin_id: str
    run: Callable[..., Any]
    ctx: PluginContext


def _safe_module_name(plugin_id: str, path: Path) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_]", "_", plugin_id)
    return f"{_MODULE_PREFIX}{slug}_{abs(hash(str(path))) & 0xFFFFFFF:x}"


def _callable_of(obj: Any, *names: str, plugin_id: str, capability: str) -> Callable:
    for name in names:
        value = getattr(obj, name, None)
        if callable(value):
            return value
    raise PluginError(
        plugin_id,
        "возможность {} объявлена, но у entry нет {}".format(
            capability, " или ".join(names)
        ),
    )


class FilterPipeline:
    """Потоковый прогон ответа через включённые фильтры.

    Фильтр участвует в потоке, если у него есть ``feed(text)`` (и, обычно,
    ``flush()``). Фильтры только с ``run(text)`` применяются к целому тексту
    (озвучка, история) — на куски потока они не натягиваются, иначе число
    «1500» могло бы разрезаться на «15» и «00».
    """

    def __init__(self, bindings: Sequence[FilterBinding]) -> None:
        self.stages: list[tuple[str, Any]] = []
        for binding in bindings:
            instance = binding.obj
            factory = binding.new_stream
            if callable(factory):
                try:
                    instance = factory()
                except Exception as exc:  # noqa: BLE001 - плагин не должен рвать поток
                    logger.warning(
                        "плагин %s: new_stream не сработал: %s",
                        binding.plugin_id,
                        exc,
                    )
                    continue
            feed = getattr(instance, "feed", None)
            if not callable(feed):
                continue
            self.stages.append((binding.plugin_id, instance))
        self._broken: set[int] = set()

    def __bool__(self) -> bool:
        return bool(self.stages)

    def feed(self, text: str) -> str:
        """Прогнать кусок через стадии; ошибка стадии не рвёт поток."""
        value = text
        for index, (plugin_id, instance) in enumerate(self.stages):
            if index in self._broken:
                continue
            try:
                value = instance.feed(value)
            except Exception as exc:  # noqa: BLE001
                self._broken.add(index)
                logger.warning("плагин %s: фильтр упал: %s", plugin_id, exc)
        return value

    def flush(self) -> str:
        """Дозаписать хвосты стадий в правильном порядке."""
        parts: list[str] = []
        for index, (plugin_id, instance) in enumerate(self.stages):
            if index in self._broken:
                continue
            flush = getattr(instance, "flush", None)
            if not callable(flush):
                continue
            try:
                tail = flush()
            except Exception as exc:  # noqa: BLE001
                self._broken.add(index)
                logger.warning("плагин %s: flush упал: %s", plugin_id, exc)
                continue
            if not tail:
                continue
            # Хвост стадии должен пройти через последующие стадии.
            for later_index in range(index + 1, len(self.stages)):
                if later_index in self._broken:
                    continue
                _later_id, later = self.stages[later_index]
                later_feed = getattr(later, "feed", None)
                if callable(later_feed):
                    try:
                        tail = later_feed(tail)
                    except Exception as exc:  # noqa: BLE001
                        self._broken.add(later_index)
                        logger.warning(
                            "плагин %s: фильтр упал: %s", _later_id, exc
                        )
            parts.append(tail)
        return "".join(parts)


class PluginRegistry:
    """Набор плагинов + их состояние. Потокобезопасен по операциям реестра."""

    def __init__(
        self,
        plugin_dirs: Sequence[Path | str] | None = None,
        config_path: Path | str | None = None,
        *,
        include_env: bool = True,
    ) -> None:
        self._lock = threading.RLock()
        self.plugin_dirs = (
            [Path(item) for item in plugin_dirs]
            if plugin_dirs is not None
            else default_plugin_dirs(include_env=include_env)
        )
        self.config_path = (
            Path(config_path)
            if config_path is not None
            else default_config_path()
        )
        self.records: dict[str, PluginRecord] = {}
        self.errors: list[dict[str, str]] = []
        self.warnings: list[str] = []
        self._config: dict[str, Any] = {}
        self._config_exists = False
        self._config_enabled: set[str] | None = None
        self._discover()
        self._load_config()
        self._load_enabled()

    # -- discovery -----------------------------------------------------------

    def _discover(self) -> None:
        seen: dict[str, Path] = {}
        for directory in self.plugin_dirs:
            base = Path(directory)
            if not base.is_dir():
                if base.exists():
                    self.errors.append(
                        {"path": str(base), "error": "не каталог плагинов"}
                    )
                continue
            try:
                children = sorted(base.iterdir(), key=lambda item: item.name)
            except OSError as exc:
                self.errors.append({"path": str(base), "error": str(exc)})
                continue
            for child in children:
                if not child.is_dir() or child.name.startswith("."):
                    continue
                if not (child / "plugin.json").is_file():
                    continue
                try:
                    manifest = load_manifest(child)
                except ManifestError as exc:
                    self.errors.append(
                        {"path": str(child), "error": str(exc)}
                    )
                    continue
                if manifest.id in seen:
                    self.errors.append(
                        {
                            "path": str(child),
                            "error": "дубликат id {!r}: уже есть в {}".format(
                                manifest.id, seen[manifest.id]
                            ),
                        }
                    )
                    continue
                seen[manifest.id] = child
                scope = manifest.scope
                self.records[manifest.id] = PluginRecord(
                    manifest=manifest,
                    path=child,
                    scope=scope,
                    priority=manifest.priority,
                )
        # Порядок вызова — priority, затем id: воспроизводимо и не зависит
        # от порядка файлов в каталоге.
        self.records = dict(
            sorted(
                self.records.items(),
                key=lambda item: (item[1].priority, item[0]),
            )
        )

    # -- конфигурация --------------------------------------------------------

    def _load_config(self) -> None:
        path = self.config_path
        if not path.is_file():
            self._config_exists = False
            self._config_enabled = None  # нет файла — включены все найденные
            return
        self._config_exists = True
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            self.errors.append(
                {"path": str(path), "error": "config: {}".format(exc)}
            )
            self._config_enabled = set()
            return
        if not isinstance(payload, Mapping):
            self.errors.append(
                {"path": str(path), "error": "config: ожидался JSON-объект"}
            )
            self._config_enabled = set()
            return
        self._config = dict(payload)
        raw = payload.get("enabled", None)
        if raw is None:
            self._config_enabled = None
            return
        if not isinstance(raw, (list, tuple)):
            self.errors.append(
                {
                    "path": str(path),
                    "error": "config.enabled: ожидался список id плагинов",
                }
            )
            self._config_enabled = set()
            return
        enabled: set[str] = set()
        for item in raw:
            if not isinstance(item, str) or not item.strip():
                self.errors.append(
                    {
                        "path": str(path),
                        "error": "config.enabled[]: ожидалась строка id",
                    }
                )
                continue
            enabled.add(item.strip())
        self._config_enabled = enabled
        unknown = sorted(enabled - set(self.records))
        for plugin_id in unknown:
            self.warnings.append(
                "config/plugins.json: плагин {!r} не найден".format(plugin_id)
            )

    def _save_config(self) -> None:
        """Записать включённые плагины атомарно (tmp + replace)."""
        payload = dict(self._config)
        payload["enabled"] = [
            plugin_id for plugin_id, record in self.records.items() if record.enabled
        ]
        if "settings" not in payload:
            payload["settings"] = {}
        path = self.config_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, path)

    def config(self) -> dict[str, Any]:
        payload = dict(self._config)
        payload["enabled"] = [
            plugin_id for plugin_id, record in self.records.items() if record.enabled
        ]
        return payload

    # -- загрузка ------------------------------------------------------------

    def _load_enabled(self) -> None:
        enabled = self._config_enabled
        for record in self.records.values():
            record.enabled = enabled is None or record.id in enabled
        for record in self.records.values():
            if not record.enabled:
                continue
            try:
                self._load_record(record)
            except PluginError as exc:
                record.error = str(exc)
                self.errors.append({"path": str(record.path), "error": str(exc)})
                logger.warning("%s", exc)

    def _entry_file(self, record: PluginRecord) -> Path:
        entry = record.manifest.entry
        candidate = record.path / entry
        if candidate.is_file():
            return candidate
        dotted = record.path / (entry.replace(".", "/") + ".py")
        if dotted.is_file():
            return dotted
        with_py = record.path / (entry + ".py")
        if with_py.is_file():
            return with_py
        raise PluginError(
            record.id,
            "entry {!r} не найден в {}".format(entry, record.path),
        )

    def _load_record(self, record: PluginRecord) -> None:
        if record.loaded and record.error == "":
            return
        entry_file = self._entry_file(record)
        module_name = _safe_module_name(record.id, entry_file)
        spec = importlib.util.spec_from_file_location(module_name, entry_file)
        if spec is None or spec.loader is None:
            raise PluginError(
                record.id, "не удалось загрузить {}".format(entry_file)
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        # Соседние файлы плагина должны импортироваться как локальные
        # модули, но только на время exec_module: чужой sys.path не правим.
        sys.path.insert(0, str(record.path))
        try:
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001 - текст ошибки важнее трейсбека
            sys.modules.pop(module_name, None)
            raise PluginError(
                record.id,
                "ошибка загрузки entry {}: {}: {}".format(
                    entry_file.name, type(exc).__name__, exc
                ),
            ) from exc
        finally:
            try:
                sys.path.remove(str(record.path))
            except ValueError:  # pragma: no cover - кто-то уже убрал
                pass
        ctx = PluginContext(
            record.id,
            plugin_dir=record.path,
            permissions=record.manifest.permissions,
            settings=self._settings_for(record.id, record.manifest.settings),
        )
        register = getattr(module, "register", None)
        try:
            obj = register(ctx) if callable(register) else module
            if obj is None:
                raise PluginError(record.id, "register() вернул None")
            self._validate_capabilities(record.manifest, obj)
        except PluginError:
            sys.modules.pop(module_name, None)
            raise
        except Exception as exc:  # noqa: BLE001
            sys.modules.pop(module_name, None)
            raise PluginError(
                record.id,
                "register() упал: {}: {}".format(type(exc).__name__, exc),
            ) from exc
        record.module = module
        record.obj = obj
        record.ctx = ctx
        record.loaded = True
        record.error = ""
        record.scope = self._scope_of(record)
        record.priority = self._priority_of(record)

    def _settings_for(self, plugin_id: str, defaults: Mapping[str, Any]) -> dict[str, Any]:
        settings: dict[str, Any] = dict(defaults)
        configured = self._config.get("settings")
        if isinstance(configured, Mapping):
            extra = configured.get(plugin_id)
            if isinstance(extra, Mapping):
                settings.update(dict(extra))
        return settings

    def _validate_capabilities(self, manifest: Manifest, obj: Any) -> None:
        plugin_id = manifest.id
        for capability in manifest.capabilities:
            if capability == "skill.fetch":
                _callable_of(
                    obj,
                    "fetch",
                    plugin_id=plugin_id,
                    capability=capability,
                )
            elif capability == "filter.run":
                _callable_of(
                    obj,
                    "run",
                    plugin_id=plugin_id,
                    capability=capability,
                )
            elif capability == "tts.engine":
                _callable_of(
                    obj,
                    "synthesize",
                    "engine",
                    plugin_id=plugin_id,
                    capability=capability,
                )
            elif capability == "command":
                _callable_of(
                    obj,
                    "run_command",
                    "command",
                    plugin_id=plugin_id,
                    capability=capability,
                )

    def _scope_of(self, record: PluginRecord) -> str:
        value = getattr(record.obj, "scope", record.manifest.scope)
        value = str(value or record.manifest.scope).lower()
        return value if value in ("chat", "tts", "all") else record.manifest.scope

    def _priority_of(self, record: PluginRecord) -> int:
        value = getattr(record.obj, "priority", record.manifest.priority)
        if isinstance(value, bool) or not isinstance(value, int):
            return record.manifest.priority
        return value

    # -- список и переключение ----------------------------------------------

    def get(self, plugin_id: str) -> PluginRecord:
        record = self.records.get(plugin_id)
        if record is None:
            raise PluginNotFound(plugin_id)
        return record

    def enabled(self) -> list[PluginRecord]:
        return [record for record in self.records.values() if record.enabled]

    def list_plugins(self) -> list[dict[str, Any]]:
        return [record.info() for record in self.records.values()]

    def as_dict(self) -> dict[str, Any]:
        return {
            "api": API_VERSION,
            "plugins": self.list_plugins(),
            "enabled": [
                record.id
                for record in self.records.values()
                if record.enabled
            ],
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }

    def set_enabled(self, plugin_id: str, enabled: bool) -> PluginRecord:
        """Включить/выключить плагин; для stateful — отказ с текстом."""
        if not isinstance(enabled, bool):
            raise PluginError(plugin_id, "enabled: ожидалось true или false")
        with self._lock:
            record = self.get(plugin_id)
            if record.enabled == enabled and (not enabled or record.loaded):
                return record
            if not enabled and record.manifest.stateful:
                raise PluginRestartRequired(plugin_id)
            if enabled and not record.loaded:
                self._load_record(record)
            old = record.enabled
            record.enabled = enabled
            try:
                self._save_config()
            except OSError as exc:
                record.enabled = old
                raise PluginError(
                    plugin_id,
                    "не удалось сохранить config/plugins.json: {}".format(exc),
                ) from exc
            return record

    def reload(self) -> "PluginRegistry":
        """Перечитать каталог и конфиг (для тестов и ручного обновления)."""
        self.records = {}
        self.errors = []
        self.warnings = []
        self._config = {}
        self._config_exists = False
        self._config_enabled = None
        self._discover()
        self._load_config()
        self._load_enabled()
        return self

    # -- способности ---------------------------------------------------------

    def skills(self) -> list[SkillBinding]:
        bindings: list[SkillBinding] = []
        for record in self.enabled():
            if not record.loaded or "skill.fetch" not in record.manifest.capabilities:
                continue
            assert record.ctx is not None
            try:
                fetch = _callable_of(
                    record.obj,
                    "fetch",
                    plugin_id=record.id,
                    capability="skill.fetch",
                )
            except PluginError as exc:
                record.last_error = str(exc)
                continue
            bindings.append(
                SkillBinding(
                    plugin_id=record.id,
                    fetch=fetch,
                    ctx=record.ctx,
                    priority=record.priority,
                )
            )
        bindings.sort(key=lambda item: (item.priority, item.plugin_id))
        return bindings

    def filters(self, scope: str = "chat") -> list[FilterBinding]:
        requested = str(scope or "all").lower()
        bindings: list[FilterBinding] = []
        for record in self.enabled():
            if not record.loaded or "filter.run" not in record.manifest.capabilities:
                continue
            if record.scope not in ("all", requested):
                continue
            try:
                run = _callable_of(
                    record.obj,
                    "run",
                    plugin_id=record.id,
                    capability="filter.run",
                )
            except PluginError as exc:
                record.last_error = str(exc)
                continue
            factory = getattr(record.obj, "new_stream", None)
            bindings.append(
                FilterBinding(
                    plugin_id=record.id,
                    run=run,
                    obj=record.obj,
                    priority=record.priority,
                    scope=record.scope,
                    new_stream=factory if callable(factory) else None,
                )
            )
        bindings.sort(key=lambda item: (item.priority, item.plugin_id))
        return bindings

    def collect_facts(
        self, question: str, *, limit: int = DEFAULT_FACT_LIMIT
    ) -> list[Fact]:
        """Спросить все включённые skill-плагины и вернуть не больше ``limit``."""
        text = str(question or "").strip()
        if not text:
            return []
        maximum = max(0, int(limit))
        facts: list[Fact] = []
        for binding in self.skills():
            if len(facts) >= maximum:
                break
            record = self.records.get(binding.plugin_id)
            if record is not None:
                record.last_error = ""
            try:
                value = binding.fetch(binding.ctx, text)
            except PluginPermissionError as exc:
                self._note_error(binding.plugin_id, exc)
                logger.warning("%s", exc)
                continue
            except Exception as exc:  # noqa: BLE001 - один плагин не рвёт чат
                self._note_error(
                    binding.plugin_id,
                    f"{type(exc).__name__}: {exc}",
                )
                logger.warning(
                    "плагин %s: skill.fetch упал: %s: %s",
                    binding.plugin_id,
                    type(exc).__name__,
                    exc,
                )
                continue
            for fact in _coerce_facts(value, binding.plugin_id):
                facts.append(fact)
                if len(facts) >= maximum:
                    break
        return facts[:maximum]

    def filter_text(self, text: str, *, scope: str = "chat") -> str:
        """Прогнать целый текст через фильтры (без потокового состояния)."""
        value = str(text or "")
        for binding in self.filters(scope):
            try:
                result = binding.run(value)
            except Exception as exc:  # noqa: BLE001
                self._note_error(
                    binding.plugin_id, f"{type(exc).__name__}: {exc}"
                )
                logger.warning(
                    "плагин %s: filter.run упал: %s", binding.plugin_id, exc
                )
                continue
            if isinstance(result, str):
                value = result
        return value

    def stream_filter(self, *, scope: str = "chat") -> FilterPipeline:
        """Потоковый фильтр ответа: только фильтры с ``feed``/``flush``."""
        return FilterPipeline(self.filters(scope))

    def tts_engines(self) -> list[TtsBinding]:
        """Зарегистрированные движки синтеза (каркас: ядро пока на eva/tts.py)."""
        bindings: list[TtsBinding] = []
        for record in self.enabled():
            if not record.loaded or "tts.engine" not in record.manifest.capabilities:
                continue
            assert record.ctx is not None
            try:
                synthesize = _callable_of(
                    record.obj,
                    "synthesize",
                    "engine",
                    plugin_id=record.id,
                    capability="tts.engine",
                )
            except PluginError as exc:
                record.last_error = str(exc)
                continue
            bindings.append(
                TtsBinding(
                    plugin_id=record.id,
                    synthesize=synthesize,
                    ctx=record.ctx,
                )
            )
        bindings.sort(key=lambda item: (self.records[item.plugin_id].priority, item.plugin_id))
        return bindings

    def commands(self) -> list[CommandBinding]:
        """Зарегистрированные слэш-команды (каркас, чат их пока не зовёт)."""
        bindings: list[CommandBinding] = []
        for record in self.enabled():
            if not record.loaded or "command" not in record.manifest.capabilities:
                continue
            assert record.ctx is not None
            try:
                run = _callable_of(
                    record.obj,
                    "run_command",
                    "command",
                    plugin_id=record.id,
                    capability="command",
                )
            except PluginError as exc:
                record.last_error = str(exc)
                continue
            bindings.append(
                CommandBinding(plugin_id=record.id, run=run, ctx=record.ctx)
            )
        return bindings

    def run_command(self, line: str) -> str | None:
        """Первый плагин, который взял слэш-команду; ``None`` — никто."""
        text = str(line or "").strip()
        if not text.startswith("/"):
            return None
        for binding in self.commands():
            try:
                result = binding.run(binding.ctx, text)
            except Exception as exc:  # noqa: BLE001
                self._note_error(
                    binding.plugin_id, f"{type(exc).__name__}: {exc}"
                )
                logger.warning(
                    "плагин %s: command упал: %s", binding.plugin_id, exc
                )
                continue
            if isinstance(result, str) and result.strip():
                return result
        return None

    # -- журнал --------------------------------------------------------------

    def _note_error(self, plugin_id: str, error: object) -> None:
        record = self.records.get(plugin_id)
        if record is not None:
            record.last_error = str(error)


def _coerce_facts(value: Any, plugin_id: str) -> list[Fact]:
    """Привести ответ плагина к Fact: свой, ``fresh.Fact`` или строка.

    ``fresh.Fact`` (0208) — чужой dataclass с полями ``data``/``source``;
    импортировать его сюда нельзя (плагины могут быть без 0208), поэтому
    читаем duck-typing.
    """
    if value is None:
        return []
    if isinstance(value, Fact):
        return [
            Fact(
                text=value.text,
                source=value.source,
                plugin=value.plugin or plugin_id,
                kind=value.kind,
            )
        ]
    if isinstance(value, (list, tuple)):
        result: list[Fact] = []
        for item in value:
            result.extend(_coerce_facts(item, plugin_id))
        return result
    data = getattr(value, "data", None)
    if isinstance(data, str):
        source = getattr(value, "source", None)
        kind = getattr(value, "kind", "")
        return [
            Fact(
                text=data,
                source=source if isinstance(source, str) else None,
                plugin=plugin_id,
                kind=str(kind or ""),
            )
        ]
    text = getattr(value, "text", None)
    if isinstance(text, str):
        source = getattr(value, "source", None)
        return [
            Fact(
                text=text,
                source=source if isinstance(source, str) else None,
                plugin=plugin_id,
            )
        ]
    if isinstance(value, str):
        return [Fact(text=value, plugin=plugin_id)]
    return []


def default_plugin_dirs(*, include_env: bool = True) -> list[Path]:
    """Каталоги discovery: ``EVA_PLUGINS_DIR`` (через ``os.pathsep``), затем ``plugins/``."""
    dirs: list[Path] = []
    if include_env:
        raw = os.environ.get("EVA_PLUGINS_DIR", "")
        for item in raw.split(os.pathsep):
            item = item.strip()
            if item:
                dirs.append(Path(item).expanduser())
    repo = Path(__file__).resolve().parents[2]
    dirs.append(repo / "plugins")
    unique: list[Path] = []
    for directory in dirs:
        resolved = directory.resolve(strict=False)
        if resolved not in unique:
            unique.append(resolved)
    return unique


def default_config_path() -> Path:
    """``EVA_PLUGINS_CONFIG`` или ``config/plugins.json`` репозитория."""
    raw = os.environ.get("EVA_PLUGINS_CONFIG", "").strip()
    if raw:
        return Path(raw).expanduser()
    repo = Path(__file__).resolve().parents[2]
    return repo / "config" / "plugins.json"


_DEFAULT_LOCK = threading.Lock()
_DEFAULT: PluginRegistry | None = None


def default_registry(*, reload: bool = False) -> PluginRegistry:
    """Общий реестр процесса: один discovery на сервер/CLI, без перезагрузки."""
    global _DEFAULT
    with _DEFAULT_LOCK:
        if _DEFAULT is None or reload:
            _DEFAULT = PluginRegistry()
        return _DEFAULT
