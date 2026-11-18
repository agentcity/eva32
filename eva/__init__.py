"""EVA32 — гибридный ассистент: скиллы → малая модель → большая модель.

Контракт — ``docs/hybrid.md``, решение — ``docs/adr/0009-hybrid-router.md``.

Публичный API::

    from eva import Config, Message, Request, Route, Answer
    from eva import decide, answer

    cfg = Config.from_env()
    req = Request(messages=[Message("user", "привет")], meta={"user_id": "u1"})
    route = decide(req, cfg)      # skill | small | big + reason
    ans = answer(req, cfg, deps)  # текст, маршрут, цена, латентность

``deps`` (необязательный) — словарь подмен для тестов/сервера:
``skills`` (try_skills), ``classifier`` (classify), ``small`` и ``big``
(клиенты с ``.chat``). Без ``deps`` зависимости импортируются лениво,
поэтому пакет работает, даже когда скиллы/клиенты/классификатор ещё
не написаны: ``decide`` использует встроенную эвристику.

Ошибки политики: :class:`BudgetExceeded` (бюджет на запрос не позволяет
вызов) и :class:`ModelUnavailable` (модель не настроена и фолбэк недоступен).
"""

from __future__ import annotations

from .config import Config, load as load_config
from .policy import Budget, BudgetExceeded, ModelUnavailable, PolicyError
from .router import answer, decide
from .types import Answer, Message, Request, Route

__version__ = "0.1.0"

__all__ = [
    "Answer",
    "Budget",
    "BudgetExceeded",
    "Config",
    "Message",
    "ModelUnavailable",
    "PolicyError",
    "Request",
    "Route",
    "answer",
    "decide",
    "load_config",
    "__version__",
]
