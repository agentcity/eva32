"""Клиенты EVA32 к моделям.

Единственный клиент — `OpenAIClient`, OpenAI-совместимый (`/v1/chat/completions`).
Он же обслуживает малую локальную модель, и большую облачную: разница только в
`base`, `key` и `model`.

    from eva.clients import OpenAIClient, ClientError, ClientResult

Контракт: `docs/hybrid.md`. Ретраи — в `eva.clients.openai_compat`.
"""

from .openai_compat import (
    DEFAULT_BACKOFF,
    DEFAULT_RETRIES,
    DEFAULT_TIMEOUT,
    MAX_BACKOFF,
    ClientError,
    ClientResult,
    HTTPResponse,
    OpenAIClient,
)

__all__ = [
    "ClientError",
    "ClientResult",
    "HTTPResponse",
    "OpenAIClient",
    "DEFAULT_BACKOFF",
    "DEFAULT_RETRIES",
    "DEFAULT_TIMEOUT",
    "MAX_BACKOFF",
]
