"""Бэкенды «маленькой модели» для гибрида.

Сейчас один бэкенд — локальная ВТМ через C-ABI (``vtm_backend``). Пакет
импортируется без загрузки ``libvtm``: всё ленивое, поэтому ``import
eva.backends`` не падает и не требует собранной библиотеки.

    from eva.backends import VtmBackend

    with VtmBackend() as backend:
        print(backend.generate([1, 2, 3]))

    from eva.backends import VTMConfig
    cfg = VTMConfig.from_env()      # EVA_VTM_MODEL/LIB/MAX_TOKENS/...
"""

from __future__ import annotations

from .tokenizer import BPETokenizer, StreamDecoder, TokenizerError
from .vtm_backend import (
    DEFAULT_MAX_TOKENS,
    VTM_API_CANCELLED,
    VTM_API_EOS,
    VTM_API_OK,
    VTM_API_STOPPED,
    VTMConfig,
    VtmBackend,
    VtmBusy,
    VtmCancelled,
    VtmError,
    VtmGeneration,
    VtmUnavailable,
    available,
    build_shared_library,
    default_model_path,
    ensure_library,
    format_chat_prompt,
    load_library,
)

__all__ = [
    "BPETokenizer",
    "DEFAULT_MAX_TOKENS",
    "StreamDecoder",
    "TokenizerError",
    "VTMConfig",
    "VTM_API_CANCELLED",
    "VTM_API_EOS",
    "VTM_API_OK",
    "VTM_API_STOPPED",
    "VtmBackend",
    "VtmBusy",
    "VtmCancelled",
    "VtmError",
    "VtmGeneration",
    "VtmUnavailable",
    "available",
    "build_shared_library",
    "default_model_path",
    "ensure_library",
    "format_chat_prompt",
    "load_library",
]
