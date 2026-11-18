"""Маршрутизатор гибридного ассистента: скиллы → классификатор → модель.

Публичный контракт (``docs/hybrid.md``)::

    decide(req, cfg) -> Route
    answer(req, cfg, deps) -> Answer
    stream(req, cfg, deps) -> Iterator[StreamChunk]

``deps`` — необязательные подмены зависимостей (в тестах и на сервере)::

    {
      "skills":      callable(text, ctx) -> SkillResult|None,
      "classifier":  callable(text, cfg) -> Route,
      "small":       клиент с .chat(messages, max_tokens=...)
                     и необязательным .stream_chat(messages, max_tokens=...);
                     это может быть и локальная ВТМ (eva.backends.VtmBackend),
      "big":         то же для большой модели,
      # алиасы: try_skills / classify / small_client / big_client,
      #         либо "clients": {"small": ..., "big": ...}
      # контекст каталога (RAG): callable(text, catalog) -> RagContext|str|None
      # и/или "catalog": Catalog | путь | список позиций,
      # алиасы: "rag" / "retrieve"
    }

Локальная модель и порядок деградации (задача 0104)
----------------------------------------------------

«Маленькой моделью» может быть не только HTTP-клиент, но и своя ВТМ
(``eva.backends.VtmBackend``): у неё та же форма ``chat``/``stream_chat``.
Маршрутизатор распознаёт её :func:`is_local_model` (явный флаг
``is_local``/``local``, модуль ``eva.backends`` или duck-typing
``model_path``/``generate`` без HTTP-полей ``base``/``endpoint``) и берёт в
работу как есть — без сети, ключей и демона.

Источник «малой» выбирается так:

1. ``deps["small"]``, если передан (локальная ВТМ или HTTP-клиент);
2. иначе — свой ``VtmBackend`` по ``cfg.local_model`` (``EVA_LOCAL_MODEL``);
3. иначе — HTTP-клиент малой модели по ``cfg.small_*``.

Деградация при отказе идёт **по порядку**, и первый удачный шаг завершает
ответ; каждый шаг заменяет только «малую» модель, маршрут остаётся ``small``:

1. локальная ВТМ (если она выбрана первой — ``EVA_LOCAL_MODEL_PRIORITY=1``);
2. HTTP-клиент малой модели (``eva.clients`` по ``cfg.small_*``);
3. большая модель — по общей политике :mod:`eva.policy` (таймаут, 429,
   бюджет: ``small`` → ``big``);
4. если не осталось никого — исходная ошибка наружу (``VtmUnavailable`` и
   прочие ошибки локальной модели не глотаются, а служат причиной отката).

Приоритет выключается ``EVA_LOCAL_MODEL_PRIORITY=0``: тогда первым идёт
HTTP-клиент, а локальная ВТМ — запасной путь для него. Каждый ответ маршрута
``small`` помечается признаком ``local`` (``True`` — ответила ВТМ, ``False`` —
HTTP), чтобы телеметрия могла их различить: признак лежит в
``Answer.usage["local"]`` и в атрибуте ``Route.local``; для чтения из чужого
кода есть :func:`route_is_local`.

Контекст каталога (``eva.rag``) подмешивается в сообщения для маршрутов
``small`` и ``big`` отдельным system-сообщением после уже имеющихся system;
сам запрос пользователя не меняется. Контекста нет — сообщения как раньше.

Без ``deps`` модули ``eva.skills``, ``eva.classifier`` и ``eva.clients``
импортируются **лениво** — пакет работает и тогда, когда их ещё нет.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from . import policy
from .config import Config
from .types import Answer, Message, Request, Route

__all__ = [
    "StreamChunk",
    "answer",
    "decide",
    "is_local_model",
    "route_is_local",
    "stream",
]

_MODEL_KINDS = ("small", "big")
_SKILL_ALIASES = ("skills", "try_skills", "skill")
_CLASSIFIER_ALIASES = ("classifier", "classify")
_SMALL_ALIASES = ("small", "small_client", "client_small")
_BIG_ALIASES = ("big", "big_client", "client_big")
_RAG_ALIASES = ("rag", "retrieve")

#: Явные пометки «это локальная модель» у клиента (bool или вызываемое).
_LOCAL_FLAG_ATTRS = ("is_local", "local")
#: Пакет, чьи объекты — заведомо локальная ВТМ, а не HTTP-клиент.
_LOCAL_MODULE_PREFIX = "eva.backends"
#: Пакет HTTP-клиентов: даже с ``stream_chat`` это не локальная модель.
_HTTP_MODULE_PREFIXES = ("eva.clients",)

# Кэш только успешной загрузки: если модуля ещё нет, пробуем импорт при
# следующем вызове (другие агенты кладут файлы параллельно).
_SKILLS_FN: Any | None = None
_CLASSIFIER_FN: Any | None = None
_RAG_FN: Any | None = None


@dataclass(frozen=True)
class StreamChunk(Mapping[str, Any]):
    """Один кусок потока ``router.stream``.

    * ``done=False`` — текст ответа (для модели — дельта от клиента;
      ``route`` пуст, потому что маршрут ещё не объявлен клиенту);
    * ``done=True`` — финальный чанк: ``route``, ``skill``, ``usage``,
      ``cost_rub`` и ``latency_ms``. У скилла это единственный чанк: ответ
      детерминирован, дробить его незачем.

    Форма совместима с сервером: у чанка есть ``text`` и (у финального)
    ``route`` — по ним ``eva.server`` собирает ``delta.content`` и ``eva``.
    Читается и как объект (``chunk.text``), и как словарь (``chunk["text"]``).
    """

    text: str = ""
    route: Route | None = None
    skill: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    cost_rub: float = 0.0
    latency_ms: int = 0
    done: bool = False

    _FIELDS = ("text", "route", "skill", "usage", "cost_rub", "latency_ms", "done")

    def __getitem__(self, key: str) -> Any:
        if key in self._FIELDS:
            return getattr(self, key)
        raise KeyError(key)

    def __iter__(self):
        return iter(self._FIELDS)

    def __len__(self) -> int:
        return len(self._FIELDS)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "route": self.route.to_dict() if self.route is not None else None,
            "skill": self.skill,
            "usage": dict(self.usage),
            "cost_rub": self.cost_rub,
            "latency_ms": self.latency_ms,
            "done": self.done,
        }


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


def stream(
    req: Request,
    cfg: Config,
    deps: Mapping[str, Any] | Any | None = None,
) -> Iterator[StreamChunk]:
    """Ответить на запрос потоком кусков :class:`StreamChunk`.

    Порядок: решение о маршруте → текст (скилл отдаёт один кусок целиком,
    модель — куски из ``client.stream_chat``) → финальный чанк с ``route``,
    ``usage`` и ``cost_rub``. Для скилла единственный кусок уже финальный:
    цена 0, модели не вызываются.

    Ошибка модели/клиента после первого отданного куска пробрасывается
    вызывающему (сервер превращает её в событие ``error``): повторять поток
    нельзя, клиент получил бы дубли текста. До первого куска работают те же
    фолбэки, что и в :func:`answer` (small→big / big→small на 429 и бюджет).
    """
    started = time.perf_counter()

    route, skill_result = _decide(req, cfg, deps)
    if route.kind == "skill":
        yield StreamChunk(
            text=_skill_text(skill_result),
            route=route,
            skill=_skill_name(skill_result),
            usage={"prompt_tokens": 0, "completion_tokens": 0, "cost_rub": 0.0},
            cost_rub=0.0,
            latency_ms=_elapsed_ms(started),
            done=True,
        )
        return

    yield from _stream_with_models(req, cfg, deps, route, started)


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
        note = _local_fallback_note(client)
        if note:
            reason = f"{reason}; {note}" if reason else note
        if budget.limited and cost > float(budget.max_cost_rub or 0.0) + 1e-9:
            reason += (
                f"; фактическая цена {cost:.4f} ₽ превысила бюджет "
                f"{float(budget.max_cost_rub or 0.0):.4f} ₽ (оценка была ниже)"
            )
        local = _local_marker(client, kind)
        answer_route = Route(kind=kind, reason=reason, confidence=route.confidence)
        _attach_local(answer_route, local)
        return Answer(
            text=text,
            route=answer_route,
            skill=None,
            usage=_usage_with_local(usage, local),
            cost_rub=cost,
            latency_ms=_elapsed_ms(started),
        )

    raise policy.ModelUnavailable(
        f"неизвестный маршрут {route.kind!r}: ожидался skill/small/big",
        route_kind=route.kind,
    )


# --------------------------------------------------------------------------- #
# поток через модели
# --------------------------------------------------------------------------- #
def _stream_with_models(
    req: Request,
    cfg: Config,
    deps: Mapping[str, Any] | Any | None,
    route: Route,
    started: float,
) -> Iterator[StreamChunk]:
    """Генератор кусков модели: бюджет, фолбэки и цена — как в :func:`answer`.

    Фолбэк допустим только пока не отдан ни один кусок текста: после первого
    куска повтор другой моделью дал бы клиенту два разных ответа подряд.
    """
    budget = policy.Budget.from_config(cfg)
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

        produced = 0
        completion_parts: list[str] = []
        usage_raw: Any = None

        try:
            for piece in _client_stream(client, messages, cfg.max_tokens_for(kind)):
                piece_usage = _stream_piece_usage(piece)
                if piece_usage is not None:
                    usage_raw = piece_usage
                text = _stream_piece_text(piece)
                if not text:
                    continue
                produced += 1
                completion_parts.append(text)
                yield StreamChunk(text=text)
        except Exception as exc:  # noqa: BLE001 — политика решает, что делать
            if produced:
                # Часть ответа уже у клиента: откат невозможен, только ошибка.
                raise
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
                        f"{route.reason}; {policy.fallback_reason(kind, target, cause, str(exc))}"
                        if route.reason
                        else policy.fallback_reason(kind, target, cause, str(exc))
                    ),
                    confidence=0.5,
                )
                kind = target
                continue
            raise

        completion_text = "".join(completion_parts)
        usage = _stream_usage(usage_raw, cfg, kind, prompt_text, completion_text)
        cost = float(usage["cost_rub"])
        budget.charge(cost)
        reason = route.reason
        note = _local_fallback_note(client)
        if note:
            reason = f"{reason}; {note}" if reason else note
        if budget.limited and cost > float(budget.max_cost_rub or 0.0) + 1e-9:
            reason += (
                f"; фактическая цена {cost:.4f} ₽ превысила бюджет "
                f"{float(budget.max_cost_rub or 0.0):.4f} ₽ (оценка была ниже)"
            )
        local = _local_marker(client, kind)
        final_route = Route(kind=kind, reason=reason, confidence=route.confidence)
        _attach_local(final_route, local)
        yield StreamChunk(
            text="",
            route=final_route,
            usage=_usage_with_local(usage, local),
            cost_rub=cost,
            latency_ms=_elapsed_ms(started),
            done=True,
        )
        return

    raise policy.ModelUnavailable(
        f"неизвестный маршрут {route.kind!r}: ожидался skill/small/big",
        route_kind=route.kind,
    )


def _client_stream(client: Any, messages: list[Message], max_tokens: int):
    """Поток от клиента: ``stream_chat``, иначе один кусок из ``chat``.

    Запасной путь нужен подменам в тестах и старым клиентам: серверу всё
    равно, одним куском ответили или многими, а формат событий тот же.
    """
    stream_fn = getattr(client, "stream_chat", None)
    if callable(stream_fn):
        return stream_fn(list(messages), max_tokens=max_tokens)
    return iter((client.chat(list(messages), max_tokens=max_tokens),))


def _stream_piece_text(piece: Any) -> str:
    """Текст куска клиента: строка, объект/словарь ``text`` или ``delta``."""
    if isinstance(piece, str):
        return piece
    if isinstance(piece, bytes):
        return piece.decode("utf-8", "replace")
    if isinstance(piece, Mapping):
        text = piece.get("text")
        if isinstance(text, str):
            return text
        delta = piece.get("delta")
        if isinstance(delta, str):
            return delta
        if isinstance(delta, Mapping) and isinstance(delta.get("content"), str):
            return str(delta["content"])
        content = piece.get("content")
        if isinstance(content, str):
            return content
        return ""
    text = getattr(piece, "text", None)
    if isinstance(text, str):
        return text
    delta = getattr(piece, "delta", None)
    if isinstance(delta, str):
        return delta
    return ""


def _stream_piece_usage(piece: Any) -> dict[str, Any] | None:
    """``usage`` куска клиента; пустой словарь — «провайдер промолчал»."""
    if isinstance(piece, Mapping):
        usage = piece.get("usage")
    else:
        usage = getattr(piece, "usage", None)
    if isinstance(usage, Mapping) and usage:
        return dict(usage)
    return None


def _stream_usage(
    raw: Any,
    cfg: Config,
    kind: str,
    prompt_text: str,
    completion_text: str,
) -> dict[str, Any]:
    """Итоговый ``usage`` потока: данные провайдера, иначе локальная оценка.

    Провайдер отдал токены/цену — используем их (цена провайдера важнее
    тарифа). Не отдал ничего — оцениваем токены по длине prompt и уже
    сгенерированного текста и считаем цену по тарифу ``cfg``. Так в потоке
    цена есть всегда, а не только у провайдеров с ``include_usage``.
    """
    if isinstance(raw, Mapping) and raw:
        usage = _normalize_usage(raw, cfg, kind)
        if not usage["prompt_tokens"] and not usage["completion_tokens"] and not usage["cost_rub"]:
            return _estimated_stream_usage(cfg, kind, prompt_text, completion_text)
        return usage
    return _estimated_stream_usage(cfg, kind, prompt_text, completion_text)


def _estimated_stream_usage(
    cfg: Config, kind: str, prompt_text: str, completion_text: str
) -> dict[str, Any]:
    prompt_tokens = policy.estimate_tokens(prompt_text) if (prompt_text or "").strip() else 0
    completion_tokens = (
        policy.estimate_tokens(completion_text) if (completion_text or "").strip() else 0
    )
    cost = cfg.estimate_cost(kind, prompt_tokens, completion_tokens)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_rub": round(float(cost), 6),
    }


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


def _local_flag(client: Any) -> bool | None:
    """Явная пометка локальности: ``is_local``/``local`` (bool или вызов).

    Не-булево значение (например, ``Mock``) пометкой не считается: иначе
    подмена в тестах случайно выглядела бы локальной моделью.
    """
    for name in _LOCAL_FLAG_ATTRS:
        try:
            value = getattr(client, name)
        except Exception:  # noqa: BLE001 — свойство чужого класса может падать
            continue
        if isinstance(value, bool):
            return value
        if callable(value):
            try:
                result = value()
            except Exception:  # noqa: BLE001
                continue
            if isinstance(result, bool):
                return result
    return None


def is_local_model(client: Any) -> bool:
    """Похож ли объект на локальную модель (ВТМ), а не на HTTP-клиент.

    Порядок проверок:

    1. явный флаг ``is_local``/``local`` — он важнее эвристик;
    2. модуль класса из ``eva.backends`` — да, из ``eva.clients`` — нет;
    3. duck-typing: есть ``chat``, нет HTTP-полей ``base``/``endpoint`` и есть
       ``model_path``/``generate`` (форма :class:`eva.backends.VtmBackend`).

    ``False`` — обычный HTTP-клиент (``OpenAIClient``), ``True`` — локальная
    модель, которую можно звать без сети.
    """
    if client is None:
        return False
    flag = _local_flag(client)
    if flag is not None:
        return flag
    module = str(getattr(type(client), "__module__", "") or "")
    if module.startswith(_HTTP_MODULE_PREFIXES):
        return False
    if module.startswith(_LOCAL_MODULE_PREFIX):
        return True
    if not callable(getattr(client, "chat", None)):
        return False
    if any(hasattr(client, attr) for attr in ("base", "endpoint")):
        return False
    return callable(getattr(client, "generate", None)) or hasattr(client, "model_path")


def route_is_local(route: Any) -> bool | None:
    """Признак ``local`` (``True``/``False``/``None``) у маршрута ``small``.

    Маркер ставит :func:`answer`/:func:`stream` на объект ``Route`` (атрибут
    ``Route.local``): ``True`` — ответила локальная ВТМ, ``False`` — HTTP-клиент.
    ``None`` — маршрут не про малую модель (``skill``/``big``) или старый ответ.
    """
    if route is None:
        return None
    value = getattr(route, "local", None)
    return value if isinstance(value, bool) else None


def _local_marker(client: Any, kind: str) -> bool | None:
    """Ответила ли локальная модель для этого маршрута; ``None`` — не про неё."""
    if kind != "small":
        return None
    used = getattr(client, "last_local", None)
    if isinstance(used, bool):
        return used
    return is_local_model(client)


def _usage_with_local(usage: dict[str, Any], local: bool | None) -> dict[str, Any]:
    """Тот же ``usage`` плюс маркер ``local`` (счётчики токенов не трогаем)."""
    if local is None:
        return usage
    return {**usage, "local": bool(local)}


def _attach_local(route: Route, local: bool | None) -> Route:
    """Пометить маршрут признаком ``local`` (``None`` — маршрут не про малую).

    Поле не добавляется в :class:`eva.types.Route` (это общий контракт, его
    правят отдельной задачей): маркер — динамический атрибут ``route.local``,
    читается через :func:`route_is_local` и не влияет на ``to_dict()``.
    """
    if local is not None:
        try:
            route.local = bool(local)
        except Exception:  # noqa: BLE001 — замороженный Route не должен ронять ответ
            pass
    return route


def _local_fallback_note(client: Any) -> str:
    """Пояснение к причине, если внутри ``small`` был откат между моделями."""
    local = getattr(client, "last_local", None)
    error = str(getattr(client, "last_error", "") or "").strip()
    if not isinstance(local, bool) or not error:
        return ""
    if local:
        return f"HTTP-клиент малой модели не ответил ({error}) — ответила локальная ВТМ"
    return f"локальная модель не ответила ({error}) — откат на HTTP-клиент малой модели"


class _SmallWithFallback:
    """«Малая модель» из двух источников: первичный и запасной.

    Нужна для локальной ВТМ: при :class:`~eva.backends.VtmUnavailable` (или
    любой другой ошибке генерации) она молча уступает HTTP-клиенту, а маршрут
    остаётся ``small``. В потоке откат возможен только пока не отдан ни один
    кусок текста — после первого куска повтор дал бы клиенту два ответа.

    Поля ``last_local``/``last_error``/``fallbacks`` — для маршрутизатора и
    тестов: кто в итоге ответил и почему пришлось переключаться.
    """

    def __init__(
        self,
        primary: Any,
        secondary: Any | None = None,
        *,
        primary_local: bool = False,
        secondary_local: bool = False,
    ) -> None:
        self._primary = primary
        self._secondary = secondary
        self._primary_local = bool(primary_local)
        self._secondary_local = bool(secondary_local)
        self.last_local: bool | None = None
        self.last_error: str = ""
        self.fallbacks = 0

    @property
    def local(self) -> bool:
        """Локальная ли модель выбрана первой (для :func:`is_local_model`)."""
        return self._primary_local

    def _failed(self, exc: BaseException) -> str:
        return f"{type(exc).__name__}: {exc}"

    def chat(self, messages: Any, max_tokens: int = 512) -> Any:
        try:
            result = self._primary.chat(list(messages), max_tokens=max_tokens)
        except Exception as exc:  # noqa: BLE001 — решает наличие запасного клиента
            self.last_error = self._failed(exc)
            if self._secondary is None:
                self.last_local = None
                raise
            self.fallbacks += 1
            result = self._secondary.chat(list(messages), max_tokens=max_tokens)
            self.last_local = self._secondary_local
            return result
        self.last_local = self._primary_local
        return result

    def stream_chat(self, messages: Any, max_tokens: int = 512):
        produced = False
        try:
            for piece in _client_stream(self._primary, messages, max_tokens):
                produced = True
                yield piece
        except Exception as exc:  # noqa: BLE001 — как в chat(), но до первого куска
            self.last_error = self._failed(exc)
            if produced or self._secondary is None:
                # Часть ответа уже у клиента: откат дал бы дубли текста.
                self.last_local = None if produced else self.last_local
                raise
            self.fallbacks += 1
            self.last_local = self._secondary_local
            yield from _client_stream(self._secondary, messages, max_tokens)
            return
        self.last_local = self._primary_local


def _make_local_client(cfg: Config) -> Any | None:
    """Собрать локальную ВТМ по ``cfg.local_model``; ``None`` — не вышло.

    Импорт и сборка ленивые: без ``EVA_LOCAL_MODEL`` ничего не грузится, а
    ``libvtm`` ищется/собирается только при первом ``chat``. Ошибка сборки
    конфига (например, мусор в ``EVA_VTM_BACKEND``) не роняет маршрутизатор:
    локальной модели просто нет, работает HTTP.
    """
    if not getattr(cfg, "local_available", False):
        return None
    try:
        from eva.backends.vtm_backend import VTMConfig, VtmBackend
    except Exception:  # noqa: BLE001 — пакета/модуля может ещё не быть
        return None
    try:
        return VtmBackend(VTMConfig.from_env(model=cfg.local_model_path))
    except Exception:  # noqa: BLE001 — путь/бэкенд негодны: деградируем на HTTP
        return None


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
    if kind == "small":
        return _resolve_small(cfg, injected)
    if injected is not None:
        return injected
    return _make_client(kind, cfg)


def _http_small_client(cfg: Config) -> Any | None:
    """HTTP-клиент малой модели; ``None`` — не настроен или конфиг негоден.

    Кривой ``EVA_SMALL_BASE`` не должен отменять работу локальной ВТМ: при
    локальном приоритете HTTP — только запасной путь, а не обязательный.
    """
    try:
        return _make_client("small", cfg)
    except Exception:  # noqa: BLE001 — деградируем, а не падаем на старте
        return None


def _resolve_small(cfg: Config, injected: Any | None) -> Any | None:
    """«Малая модель»: внедрённая, локальная ВТМ и/или HTTP — по порядку.

    Внедрённый HTTP-клиент не переопределяется: ``deps`` — контракт сервера и
    тестов. Локальная ВТМ из ``deps["small"]`` оборачивается в откат на HTTP,
    HTTP из конфига — в откат на локальную, если приоритет выключен.
    """
    if injected is not None:
        if not is_local_model(injected):
            return injected
        http_small = _http_small_client(cfg)
        if http_small is None:
            return injected
        return _SmallWithFallback(injected, http_small, primary_local=True)

    http_small = _http_small_client(cfg)
    local = _make_local_client(cfg)
    if local is None:
        return http_small
    if not getattr(cfg, "local_first", True):
        # Приоритет выключен: решает HTTP, локальная ВТМ — запасной путь.
        if http_small is None:
            return local
        return _SmallWithFallback(http_small, local, secondary_local=True)
    return _SmallWithFallback(local, http_small, primary_local=True)


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
