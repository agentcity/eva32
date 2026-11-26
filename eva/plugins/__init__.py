"""Рантайм плагинов EVA32 (ADR 0022).

Плагин — каталог ``plugins/<id>/`` с ``plugin.json`` и python-entry. Ядро
знает только контракт: :class:`~eva.plugins.registry.PluginRegistry` находит
плагины, проверяет манифест и права, зовёт ``skill.fetch`` / ``filter.run``.

Быстрый старт для ядра::

    from eva.plugins import default_registry

    registry = default_registry()
    facts = registry.collect_facts("который час")
    text = registry.filter_text("Я понял", scope="chat")
"""

from __future__ import annotations

from eva.plugins.context import (
    Fact,
    PermissionDenied,
    Permissions,
    PluginContext,
    PluginPermissionError,
)
from eva.plugins.manifest import (
    API_VERSION,
    CAPABILITIES,
    Manifest,
    ManifestError,
    load_manifest,
    parse_manifest,
)
from eva.plugins.registry import (
    DEFAULT_FACT_LIMIT,
    CommandBinding,
    FilterBinding,
    FilterPipeline,
    PluginError,
    PluginNotFound,
    PluginRecord,
    PluginRegistry,
    PluginRestartRequired,
    SkillBinding,
    TtsBinding,
    default_plugin_dirs,
    default_registry,
)

__all__ = [
    "API_VERSION",
    "CAPABILITIES",
    "CommandBinding",
    "DEFAULT_FACT_LIMIT",
    "Fact",
    "FilterBinding",
    "FilterPipeline",
    "Manifest",
    "ManifestError",
    "PermissionDenied",
    "Permissions",
    "PluginContext",
    "PluginError",
    "PluginNotFound",
    "PluginPermissionError",
    "PluginRecord",
    "PluginRegistry",
    "PluginRestartRequired",
    "SkillBinding",
    "TtsBinding",
    "default_plugin_dirs",
    "default_registry",
    "load_manifest",
    "parse_manifest",
]
