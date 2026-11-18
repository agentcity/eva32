"""Маршрутизатор гибридного ассистента: скиллы → классификатор → модель.

Публичный контракт (``docs/hybrid.md``)::

    decide(req, cfg) -> Route
    answer(req, cfg, deps) -> Answer

``deps`` — необязательные подмены зависимостей (в тестах и на сервере)::

    {
      "skills":      callable(text, ctx) -> SkillResult|None,
      "classifier":  callable(text, cfg) -> Route,
      "small":       клиент с .chat(messages, max_tokens=...),
      "big":         клиент с .chat(messages, max_tokens=...),
      # алиасы: try_skills / classify / small_client / big_client,
      #         либо "clients": {"small": ..., "big": ...}
      # контекст каталога (RAG): callable(text, catalog) -> RagContext|str|None
      # и/или "catalog": Catalog | путь | список позиций,
      # алиасы: "rag" / "retrieve"
    }

Контекст каталога (``eva.rag``) подмешивается в сообщения для маршрутов
``small`` и ``big`` отдельным system-сообщением после уже имеющихся system;
сам запрос пользователя не меняется. Контекста нет — сообщения как раньше.

Без ``deps`` модули ``eva.skills``, ``eva.classifier`` и ``eva.clients``
импортируются **лениво** — пакет работает и тогда, когда их ещё нет.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Mapping
from typing import Any

from . import policy
from .config import Config
from .types import Answer, Message, Request, Route

__all__ = ["answer", "decide"]

_MODEL_KINDS = ("small", "big")
_SKILL_ALIASES = ("skills", "try_skills", "skill")
_CLASSIFIER_ALIASES = ("classifier", "classify")
_SMALL_ALIASES = ("small", "small_client", "client_small")
_BIG_ALIASES = ("big", "big_client", "client_big")
_RAG_ALIASES = ("rag", "retrieve")

# Кэш только успешной загрузки: если модуля ещё нет, пробуем импорт при
# следующем вызове (другие агенты кладут файлы параллельно).
_SKILLS_FN: Any | None = None
_CLASSIFIER_FN: Any | None = None
_RAG_FN: Any | None = None


# --------------------------------------------------------------------------- #
# публичный API
# --------------------------------------------------------------------------- #
def decide(req: Request, cfg: Config) -> Route:
    """Решить, кто отвечает на запрос: ``skill`` | ``small`` | ``big``.

    Причина решения всегда непустая (``Route.reason``).
    """
    route, _ = _decide(req, cfg, deps=None)
    return route


def answer(req: Request, cfg: Config, deps: Mapping[str, Any] | Any | None = None) -> Answer:
    """Ответить на запрос по выбранному маршруту, с бюджетом и фолбэками."""
    started = time.perf_counter()

    route, skill_result = _decide(req, cfg, deps)
    if route.kind == "skill":
        return Answer(
            text=_skill_text(skill_result),
            route=route,
            skill=_skill_name(skill_result),
            usage={"prompt_tokens": 0, "completion_tokens": 0, "cost_rub": 0.0},
            cost_rub=0.0,
            latency_ms=_elapsed_ms(started),
        )

    return _answer_with_models(req, cfg, deps, route, started)


# --------------------------------------------------------------------------- #
# решение о маршруте
# --------------------------------------------------------------------------- #
def _decide(
    req: Request,
    cfg: Config,
    deps: Mapping[str, Any] | Any | None,
) -> tuple[Route, Any | None]:
    text = _request_text(req)

    skills = _skills_provider(deps)
    if skills is not None:
        try:
            skill_result = _call_skills(skills, text, _skill_context(req, text))
        except Exception:
            # Скилл не имеет права ронять маршрут: молча идём дальше.
            skill_result = None
        if skill_result:
            return _skill_route(skill_result), skill_result

    classifier = _classifier_provider(deps)
    if classifier is not None:
        try:
            route = _coerce_route(_call_classifier(classifier, text, cfg))
        except Exception:
            route = None
        if route is not None:
            if route.kind == "skill":
                # Скиллы уже пробовали и промолчали: маршрут skill без ответа
                # скилла недопустим (пустой Answer). По ADR 0009 там, где скилл
                # молчит, отвечает большая модель.
                route = Route(
                    kind="big",
                    reason=(
                        f"{route.reason}; скилл промолчал — по ADR 0009 "
                        "отвечает большая модель"
                    ),
                    confidence=min(route.confidence, 0.6),
                )
            return route, None

    return _heuristic_route(text, cfg), None


def _skill_route(skill_result: Any) -> Route:
    name = _skill_name(skill_result)
    if name:
        reason = f"skill: {name} — детерминированный ответ из каталога, модель не вызывалась"
    else:
        reason = "skill: детерминированный ответ из каталога, модель не вызывалась"
    return Route(kind="skill", reason=reason, confidence=1.0)


def _coerce_route(value: Any) -> Route | None:
    """Привести результат классификатора к Route; мусор — None."""
    if isinstance(value, Route):
        kind, reason, confidence = value.kind, value.reason, value.confidence
    elif isinstance(value, Mapping):
        kind = value.get("kind")
        reason = value.get("reason") or value.get("why") or ""
        confidence = value.get("confidence", 0.5)
    else:
        kind = getattr(value, "kind", None)
        reason = getattr(value, "reason", "") or ""
        confidence = getattr(value, "confidence", 0.5)

    kind = str(kind or "").strip().lower()
    if kind not in ("skill", "small", "big"):
        return None

    reason = str(reason or "").strip()
    if not reason:
        reason = f"classifier: выбран маршрут {kind} (причина не указана)"
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        confidence_value = 0.5
    return Route(kind=kind, reason=reason, confidence=confidence_value)


def _heuristic_route(text: str, cfg: Config) -> Route:
    """Запасное правило на случай, если ``eva.classifier`` ещё не написан."""
    text = (text or "").strip()
    if not text:
        return Route(
            kind="small",
            reason="эвристика (classifier недоступен): пустой запрос — отвечает малая модель",
            confidence=0.2,
        )

    lowered = text.lower()
    for marker in tuple(getattr(cfg, "classifier_reason_markers", ()) or ()):
        marker = str(marker).strip().lower()
        if marker and marker in lowered:
            return Route(
                kind="big",
                reason=(
                    "эвристика (classifier недоступен): маркер рассуждения "
                    f"«{marker}» — нужна большая модель"
                ),
                confidence=0.6,
            )

    long_chars = int(getattr(cfg, "classifier_long_chars", 120) or 0)
    if long_chars and len(text) > long_chars:
        return Route(
            kind="big",
            reason=(
                "эвристика (classifier недоступен): "
                f"запрос длиной {len(text)} > {long_chars} символов — нужна большая модель"
            ),
            confidence=0.55,
        )

    latin_ratio = _latin_ratio(text)
    threshold = float(getattr(cfg, "classifier_latin_ratio", 0.35) or 0.0)
    if threshold and latin_ratio >= threshold:
        return Route(
            kind="big",
            reason=(
                "эвристика (classifier недоступен): "
                f"доля латиницы {latin_ratio:.0%} — нужна большая модель"
            ),
            confidence=0.5,
        )

    return Route(
        kind="small",
        reason=(
            "эвристика (classifier недоступен): короткий запрос "
            "без маркеров рассуждения — хватит малой модели"
        ),
        confidence=0.5,
    )


# --------------------------------------------------------------------------- #
# ответ через модели
# --------------------------------------------------------------------------- #
def _answer_with_models(
    req: Request,
    cfg: Config,
    deps: Mapping[str, Any] | Any | None,
    route: Route,
    started: float,
) -> Answer:
    budget = policy.Budget.from_config(cfg)
    # Контекст каталога — в сообщения, а не в текст запроса: пользовательский
    # ввод остаётся прежним. Оценка цены считается по тем же сообщениям,
    # иначе добавленный контекст не попал бы в бюджет.
    messages = _messages_with_context(req, deps)
    prompt_text = _prompt_text(req, messages)
    clients: dict[str, Any | None] = {}

    def client_for(kind: str) -> Any | None:
        if kind not in clients:
            clients[kind] = _resolve_client(kind, cfg, deps)
        return clients[kind]

    kind = route.kind
    attempted: list[str] = []

    while kind in _MODEL_KINDS:
        attempted.append(kind)
        client = client_for(kind)

        if client is None:
            target = _fallback_if_available(
                target=policy.fallback_target(kind, policy.ERROR),
                budget=budget,
                cfg=cfg,
                prompt_text=prompt_text,
                attempted=attempted,
                client_for=client_for,
            )
            if target is not None:
                route = Route(
                    kind=target,
                    reason=policy.fallback_reason(
                        kind, target, policy.ERROR, f"клиент {kind} не настроен"
                    ),
                    confidence=0.5,
                )
                kind = target
                continue
            raise policy.ModelUnavailable(
                f"маршрут {route.kind}: модель не настроена и фолбэк недоступен",
                route_kind=route.kind,
            )

        estimate = policy.estimate_request_cost(cfg, kind, prompt_text, cfg.max_tokens_for(kind))
        if not budget.can_afford(estimate):
            # Бюджет проверяется ДО вызова: платить нечем — не звоним.
            target = None
            if kind == "big" and "small" not in attempted and client_for("small") is not None:
                small_estimate = policy.estimate_request_cost(
                    cfg, "small", prompt_text, cfg.max_tokens_for("small")
                )
                if budget.can_afford(small_estimate):
                    target = "small"
            if target is not None:
                route = Route(
                    kind=target,
                    reason=policy.budget_downgrade_reason(route.reason, estimate, cfg.max_cost_rub),
                    confidence=0.5,
                )
                kind = target
                continue
            budget.check(estimate, route_kind=kind)  # всегда бросает BudgetExceeded

        try:
            result = client.chat(list(messages), max_tokens=cfg.max_tokens_for(kind))
        except Exception as exc:  # noqa: BLE001 — политика решает, что делать
            cause = policy.error_kind(exc)
            target = _fallback_if_available(
                target=policy.fallback_target(kind, cause),
                budget=budget,
                cfg=cfg,
                prompt_text=prompt_text,
                attempted=attempted,
                client_for=client_for,
            )
            if target is not None:
                route = Route(
                    kind=target,
                    reason=(
                        f"{route.reason}; "
                        f"{policy.fallback_reason(kind, target, cause, str(exc))}"
                        if route.reason
                        else policy.fallback_reason(kind, target, cause, str(exc))
                    ),
                    confidence=0.5,
                )
                kind = target
                continue
            raise

        text, usage, cost = _unpack_result(result, cfg, kind)
        budget.charge(cost)
        reason = route.reason
        if budget.limited and cost > float(budget.max_cost_rub or 0.0) + 1e-9:
            reason += (
                f"; фактическая цена {cost:.4f} ₽ превысила бюджет "
                f"{float(budget.max_cost_rub or 0.0):.4f} ₽ (оценка была ниже)"
            )
        return Answer(
            text=text,
            route=Route(kind=kind, reason=reason, confidence=route.confidence),
            skill=None,
            usage=usage,
            cost_rub=cost,
            latency_ms=_elapsed_ms(started),
        )

    raise policy.ModelUnavailable(
        f"неизвестный маршрут {route.kind!r}: ожидался skill/small/big",
        route_kind=route.kind,
    )


def _fallback_if_available(
    *,
    target: str | None,
    budget: policy.Budget,
    cfg: Config,
    prompt_text: str,
    attempted: list[str],
    client_for: Any,
) -> str | None:
    if target is None or target in attempted or target not in _MODEL_KINDS:
        return None
    if client_for(target) is None:
        return None
    estimate = policy.estimate_request_cost(cfg, target, prompt_text, cfg.max_tokens_for(target))
    if not budget.can_afford(estimate):
        return None
    return target


# --------------------------------------------------------------------------- #
# зависимости: скиллы, классификатор, клиенты
# --------------------------------------------------------------------------- #
def _load_skills() -> Any | None:
    """Ленивая загрузка реестра скиллов (его пишет другой агент)."""
    global _SKILLS_FN
    if _SKILLS_FN is not None:
        return _SKILLS_FN
    try:
        from eva.skills.registry import try_skills
    except Exception:
        return None
    _SKILLS_FN = try_skills
    return _SKILLS_FN


def _load_classifier() -> Any | None:
    """Ленивая загрузка классификатора (его пишет другой агент)."""
    global _CLASSIFIER_FN
    if _CLASSIFIER_FN is not None:
        return _CLASSIFIER_FN
    try:
        from eva.classifier import classify
    except Exception:
        return None
    _CLASSIFIER_FN = classify
    return _CLASSIFIER_FN


def _load_rag() -> Any | None:
    """Ленивая загрузка RAG по каталогу (``eva.rag``)."""
    global _RAG_FN
    if _RAG_FN is not None:
        return _RAG_FN
    try:
        from eva.rag import retrieve
    except Exception:
        return None
    _RAG_FN = retrieve
    return _RAG_FN


def _make_client(kind: str, cfg: Config) -> Any | None:
    """Ленивая сборка клиента модели; ``None`` — модель не настроена."""
    if kind == "big" and not cfg.big_available:
        return None
    if kind == "small" and not cfg.small_available:
        return None
    try:
        from eva.clients.openai_compat import OpenAIClient
    except Exception:
        return None

    if kind == "big":
        return OpenAIClient(
            cfg.big_base,
            cfg.big_key,
            cfg.big_model,
            timeout=cfg.big_timeout,
            retries=cfg.big_retries,
        )
    return OpenAIClient(
        cfg.small_base,
        cfg.small_key,
        cfg.small_model,
        timeout=cfg.small_timeout,
        retries=cfg.small_retries,
    )


def _resolve_client(
    kind: str,
    cfg: Config,
    deps: Mapping[str, Any] | Any | None,
) -> Any | None:
    aliases = _BIG_ALIASES if kind == "big" else _SMALL_ALIASES
    injected = _lookup(deps, aliases)
    if injected is None:
        container = _lookup(deps, ("clients",))
        if isinstance(container, Mapping):
            for alias in aliases:
                if container.get(alias) is not None:
                    injected = container[alias]
                    break
    if injected is not None:
        return injected
    return _make_client(kind, cfg)


def _skills_provider(deps: Mapping[str, Any] | Any | None) -> Any | None:
    provider = _lookup(deps, _SKILL_ALIASES)
    if provider is not None:
        return provider
    return _load_skills()


def _classifier_provider(deps: Mapping[str, Any] | Any | None) -> Any | None:
    provider = _lookup(deps, _CLASSIFIER_ALIASES)
    if provider is not None:
        return provider
    return _load_classifier()


def _lookup(deps: Mapping[str, Any] | Any | None, names: tuple[str, ...]) -> Any | None:
    if deps is None:
        return None
    if isinstance(deps, Mapping):
        for name in names:
            if deps.get(name) is not None:
                return deps[name]
        return None
    for name in names:
        value = getattr(deps, name, None)
        if value is not None:
            return value
    return None


def _call_skills(provider: Any, text: str, ctx: dict[str, Any]) -> Any:
    if hasattr(provider, "try_skills") and callable(provider.try_skills):
        return provider.try_skills(text, ctx)
    if callable(provider):
        return provider(text, ctx)
    raise TypeError("skills: ожидался try_skills(text, ctx) или объект с .try_skills")


def _call_classifier(provider: Any, text: str, cfg: Config) -> Any:
    if hasattr(provider, "classify") and callable(provider.classify):
        return provider.classify(text, cfg)
    if callable(provider):
        return provider(text, cfg)
    raise TypeError("classifier: ожидался classify(text, cfg)")


# --------------------------------------------------------------------------- #
# контекст каталога (RAG)
# --------------------------------------------------------------------------- #
def _messages_with_context(
    req: Request,
    deps: Mapping[str, Any] | Any | None,
) -> list[Message]:
    """Сообщения для модели, при наличии контекста — с блоком каталога.

    Вставка не меняет исходные сообщения: это новый список, пользовательский
    текст остаётся байт в байт. Нет контекста — возвращаются прежние сообщения.
    """
    messages = list(req.messages)
    try:
        context = _rag_context(req, deps)
    except Exception:
        # RAG не имеет права ронять ответ: как и скилл, молчим.
        return messages
    text = _context_text(context)
    if not text.strip():
        return messages
    return _insert_context(messages, text)


def _rag_context(
    req: Request,
    deps: Mapping[str, Any] | Any | None,
) -> Any | None:
    text = _request_text(req)
    if not text.strip():
        return None
    provider = _rag_provider(deps)
    if provider is None:
        return None
    return _call_rag(provider, text, _lookup(deps, ("catalog",)))


def _rag_provider(deps: Mapping[str, Any] | Any | None) -> Any | None:
    provider = _lookup(deps, _RAG_ALIASES)
    if provider is not None:
        return provider
    return _load_rag()


def _call_rag(provider: Any, text: str, catalog: Any) -> Any:
    """Позвать RAG; подмене в тестах достаточно и одного аргумента."""
    target = getattr(provider, "retrieve", None)
    if not callable(target):
        target = provider
    if not callable(target):
        raise TypeError("rag: ожидался retrieve(text, catalog) или объект с .retrieve")
    try:
        inspect.signature(target).bind(text, catalog)
    except (TypeError, ValueError):
        # retrieve(text) — контракт задачи; каталог тогда берётся по умолчанию.
        return target(text)
    return target(text, catalog)


def _context_text(context: Any) -> str:
    """Достать текст контекста из ``RagContext``/словаря/строки."""
    if context is None:
        return ""
    if isinstance(context, str):
        return context
    if isinstance(context, Mapping):
        value = context.get("text")
    else:
        value = getattr(context, "text", None)
    return str(value or "")


def _insert_context(messages: list[Message], text: str) -> list[Message]:
    """Вставить контекст system-сообщением после уже имеющихся system."""
    context_message = Message(role="system", content=text)
    if not messages:
        return [context_message]
    index = 0
    while index < len(messages) and messages[index].role == "system":
        index += 1
    return [*messages[:index], context_message, *messages[index:]]


# --------------------------------------------------------------------------- #
# разбор результатов
# --------------------------------------------------------------------------- #
def _skill_context(req: Request, text: str) -> dict[str, Any]:
    meta = dict(req.meta or {})
    return {
        "text": text,
        "messages": [m.to_dict() for m in req.messages],
        "meta": meta,
        "user_id": meta.get("user_id"),
        "channel": meta.get("channel"),
        "request": req,
    }


def _skill_name(skill_result: Any) -> str | None:
    value: Any = None
    if isinstance(skill_result, Mapping):
        value = skill_result.get("skill") or skill_result.get("name")
    else:
        value = getattr(skill_result, "skill", None) or getattr(skill_result, "name", None)
    value = str(value).strip() if value is not None else ""
    return value or None


def _skill_text(skill_result: Any) -> str:
    if isinstance(skill_result, Mapping):
        value = skill_result.get("text") or ""
    else:
        value = getattr(skill_result, "text", "") or ""
    return str(value)


def _unpack_result(result: Any, cfg: Config, kind: str) -> tuple[str, dict[str, Any], float]:
    if isinstance(result, str):
        return result, {"prompt_tokens": 0, "completion_tokens": 0, "cost_rub": 0.0}, 0.0

    if isinstance(result, Mapping):
        data: dict[str, Any] = dict(result)
    else:
        data = {
            name: getattr(result, name)
            for name in ("text", "usage", "raw")
            if hasattr(result, name)
        }

    text = str(data.get("text") or "")
    usage = _normalize_usage(data.get("usage"), cfg, kind)
    return text, usage, float(usage["cost_rub"])


def _normalize_usage(raw: Any, cfg: Config, kind: str) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        data = dict(raw)
    else:
        data = {
            name: getattr(raw, name)
            for name in ("prompt_tokens", "completion_tokens", "cost_rub")
            if raw is not None and hasattr(raw, name)
        }

    prompt_tokens = _as_int(data.get("prompt_tokens"))
    completion_tokens = _as_int(data.get("completion_tokens"))
    cost = data.get("cost_rub")
    if cost is None:
        cost_value = cfg.estimate_cost(kind, prompt_tokens, completion_tokens)
    else:
        try:
            cost_value = float(cost)
        except (TypeError, ValueError):
            cost_value = cfg.estimate_cost(kind, prompt_tokens, completion_tokens)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_rub": round(float(cost_value), 6),
    }


def _as_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


# --------------------------------------------------------------------------- #
# мелочи
# --------------------------------------------------------------------------- #
def _request_text(req: Request) -> str:
    try:
        return req.text
    except Exception:
        return ""


def _prompt_text(req: Request, messages: list[Message] | None = None) -> str:
    """Склеенный prompt — для оценки токенов и бюджета.

    ``messages`` — то, что реально уйдёт модели (с контекстом каталога);
    без аргумента — исходные сообщения запроса, как раньше.
    """
    source = req.messages if messages is None else messages
    parts = [f"{m.role}: {m.content}" for m in source if (m.content or "").strip()]
    if not parts:
        return ""
    parts.append("assistant:")
    return "\n".join(parts)


def _latin_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    latin = sum(1 for ch in letters if ("a" <= ch.lower() <= "z"))
    return latin / len(letters)


def _elapsed_ms(started: float) -> int:
    return max(0, int(round((time.perf_counter() - started) * 1000)))
