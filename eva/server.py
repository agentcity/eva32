"""OpenAI-совместимый HTTP-сервер EVA32 на стандартной библиотеке.

Контракт: ``docs/hybrid.md``, задача ``ai/tasks/0017-eva-server.md``.

Сервер — только транспорт. Он не решает, кто отвечает (скилл, малая модель или
большая), а принимает маршрутизатор снаружи: ``create_server(router=...)``.
В тестах вместо боевого маршрутизатора подставляется заглушка, поэтому тесты
не ходят в сеть и не зависят от моделей.

Ответ ``POST /v1/chat/completions`` повторяет формат OpenAI и дополнен полем
``eva`` (маршрут, скилл, цена, латентность): без него нельзя посчитать
себестоимость запроса.

С ``"stream": true`` сервер отдаёт ``text/event-stream``: чанки
``data: {...}\\n\\n`` с ``choices[0].delta.content`` и финальный ``data: [DONE]``.
Если у инжектированного маршрутизатора есть метод ``stream(request)`` — ответ
уходит по частям по мере генерации; если нет — готовый ответ приходит одним
чанком (без имитации стриминга). Боевая связка (``router=None``) стримит через
``eva.router.stream(req, cfg, deps)``: скилл отдаёт один кусок, модель — поток
от клиента, финальный кусок несёт маршрут и цену. Ошибки до первого чанка —
обычный JSON; обрыв потока после первого чанка — событие ``error`` и
``[DONE]``, сервер продолжает работать.

Каждый запрос, дошедший до маршрутизатора, учитывается в ``server.metrics``
(``eva.telemetry.Metrics``): TTFT, полное время, токены, код завершения
(``ok``/``stop``/``cancel``/``error``) и маршрут. Если клиент оборвал SSE, сервер
не продолжает генерацию в пустоту: поток роутера закрывается, запрос
учитывается как ``cancel``. Сводка отдаётся ``GET /metrics`` (JSON, с
перцентилями p50/p95/p99 по TTFT и полному времени).

Прогрев модели (задача 0115): при старте, если задан ``EVA_VTM_MODEL`` или
передан локальный клиент (``create_server(local_model=...)``), модель
загружается и получает короткий prefill — иначе первый пользователь платит за
mmap модели, загрузку ``libvtm`` и счёт промпта по токену. Время прогрева
пишется в ``GET /metrics`` (``warmup_ms``), а его ошибка — только предупреждение:
сервер стартует и принимает запросы. TTFT в метриках разделён на prefill
(``prefill_ms``) и остаток (``ttft_after_prefill_ms``), если маршрутизатор
сообщает границу prefill; ``prefill_tokens`` считается по длине промпта и
отдаётся заголовком ``X-Prefill-Tokens``.

Ошибки клиента — 4xx с человеческим текстом, ошибка маршрутизатора — 500 с
человеческим текстом и идентификатором запроса; трейсбек уходит в лог сервера
и наружу не попадает.

Пределы запроса: промпт длиннее ``EVA_MAX_PROMPT_TOKENS`` (оценка ~4 символа
на токен) — 413 ``prompt_too_large`` до вызова моделей; лимит запросов
считается по маршруту (пути), сверх него — 429 ``rate_limited`` с
``Retry-After``; генерация дольше ``EVA_REQUEST_TIMEOUT`` — 504
``request_timeout`` с отменой потока. Если большая модель недоступна,
:class:`DegradationChain` пробует локальную модель, затем скилл, и только
потом отдаёт 503 ``service_unavailable``. Отказы и деградация видны в
``GET /metrics`` (``limits`` и ``degradation``).
"""

from __future__ import annotations

import hmac
import inspect
import json
import logging
import os
import queue
import threading
import time
import uuid
from collections import deque
from collections.abc import Iterable, Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from math import ceil, isfinite
from typing import Any, Callable
from urllib.parse import urlsplit

from .telemetry import Metrics

__all__ = [
    "AccessGuard",
    "AUTH_HEADER",
    "DEFAULT_MAX_PROMPT_TOKENS",
    "DEFAULT_MODEL",
    "DEFAULT_REQUEST_TIMEOUT_S",
    "DegradationChain",
    "EvaHTTPServer",
    "EvaRequestHandler",
    "ServerCounters",
    "ServiceUnavailable",
    "as_router",
    "as_stream_router",
    "build_request",
    "chat_chunk",
    "create_server",
    "default_router",
    "default_stream_router",
    "estimate_prompt_tokens",
    "make_handler",
    "reply_to_openai",
    "version",
    "warmup_local_model",
]

logger = logging.getLogger("eva.server")
# Библиотека не настраивает логирование: обработчик добавляет приложение
# (eva/__main__.py). NullHandler убирает только последний рубеж logging —
# вывод WARNING в stderr у процесса без настроенного журнала.
logger.addHandler(logging.NullHandler())

SERVICE_NAME = "eva-server"
#: Имя по умолчанию для ответа, если клиент не передал model.
DEFAULT_MODEL = "eva-hybrid"
#: Больше мегабайта тело чата не бывает: защита от случайного мусора.
MAX_BODY_BYTES = 1024 * 1024
#: Заголовок с ключом доступа, как у OpenAI-совместимых клиентов.
AUTH_HEADER = "Authorization"
#: Схема авторизации из контракта; сравнивается без учёта регистра.
_BEARER = "bearer"
#: Окно лимита: запросов в минуту на ключ.
RATE_LIMIT_WINDOW_S = 60.0
#: Больше стольких ключей лимита в памяти не держим: устаревшие вычищаются.
_MAX_RATE_KEYS = 4096
#: Лимит промпта по умолчанию: ~4000 токенов (оценка — 4 символа на токен).
DEFAULT_MAX_PROMPT_TOKENS = 4000
#: Таймаут генерации на запрос по умолчанию, секунд. ``0`` выключает предел.
DEFAULT_REQUEST_TIMEOUT_S = 60.0
#: Короткий промпт прогрева: prefill целиком, но без ожидания генерации.
WARMUP_PROMPT = "Привет"
#: Сколько новых токенов генерирует прогрев: один шаг декода после prefill.
WARMUP_MAX_NEW_TOKENS = 1
#: Переменная бэкенда ВТМ: задаёт модель, которую надо прогреть при старте.
_ENV_VTM_MODEL = "EVA_VTM_MODEL"

#: Сентинел «поток не дал ни одного куска» для ``next(..., _STREAM_END)``.
_STREAM_END = object()
#: Ошибки записи, означающие уход клиента; для SSE это отмена генерации.
_CLIENT_GONE = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)

_VERSION_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "VERSION"
)


def version() -> str:
    """Версия бандла из файла ``VERSION``; ``0.0.0``, если файла нет."""
    try:
        with open(_VERSION_FILE, encoding="utf-8") as handle:
            value = handle.read().strip()
    except OSError:
        return "0.0.0"
    return value or "0.0.0"


# --------------------------------------------------------------------------- #
# Доступ: токен и лимит запросов
# --------------------------------------------------------------------------- #


class AccessGuard:
    """Аутентификация по Bearer-токену и лимит запросов на ключ.

    Потокобезопасен: ``ThreadingHTTPServer`` обслуживает запросы параллельно,
    поэтому счётчики закрыты одним замком. Ключ лимита — предъявленный токен,
    а в режиме без токена — адрес клиента: разные ключи друг другу не мешают,
    а локальный «умный» клиент всё равно не выжжет бюджет.

    Окно скользящее (``window_s``, по умолчанию минута): хранятся отметки
    времени запросов, попавших в окно; ``Retry-After`` — сколько осталось до
    освобождения слота, но не меньше секунды. ``rate_limit_per_min <= 0``
    выключает лимит.
    """

    def __init__(
        self,
        token: str = "",
        rate_limit_per_min: int = 60,
        *,
        window_s: float = RATE_LIMIT_WINDOW_S,
    ) -> None:
        self.token = token.strip()
        self.rate_limit_per_min = max(0, int(rate_limit_per_min))
        self.window_s = max(1.0, float(window_s))
        self._lock = threading.Lock()
        self._hits: dict[str, deque[float]] = {}

    @property
    def enabled(self) -> bool:
        """Включена ли проверка токена."""
        return bool(self.token)

    def authorized(self, presented: str | None) -> bool:
        """Совпадает ли предъявленный токен с настроенным.

        Сравнение постоянного времени: иначе по задержке ответа токен можно
        подбирать посимвольно. Без настроенного токена пропускается любой.
        """
        if not self.token:
            return True
        if not presented:
            return False
        return hmac.compare_digest(
            self.token.encode("utf-8"), presented.strip().encode("utf-8")
        )

    def key_for(self, presented: str | None, client_ip: str = "") -> str:
        """Ключ лимита: токен, если он настроен, иначе адрес клиента."""
        if self.token:
            return presented or ""
        return client_ip or "unknown"

    def check(self, key: str, *, now: float | None = None) -> float:
        """Учитывает запрос: ``0.0`` — можно, иначе ``Retry-After``, сек.

        ``now`` — для тестов: подменяет ``time.monotonic()``, чтобы проверить
        скольжение окна без реального ожидания минуты.
        """
        if self.rate_limit_per_min <= 0:
            return 0.0
        moment = time.monotonic() if now is None else float(now)
        cutoff = moment - self.window_s
        with self._lock:
            self._prune(cutoff)
            hits = self._hits.setdefault(key, deque())
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= self.rate_limit_per_min:
                return max(1.0, hits[0] + self.window_s - moment)
            hits.append(moment)
            return 0.0

    def _prune(self, cutoff: float) -> None:
        """Выкидывает ключи, у которых все отметки старше окна (под замком)."""
        if len(self._hits) <= _MAX_RATE_KEYS:
            return
        stale = [
            key for key, hits in self._hits.items() if not hits or hits[-1] <= cutoff
        ]
        for key in stale:
            del self._hits[key]


class ServerCounters:
    """Потокобезопасные счётчики отказов сервера для ``GET /metrics``.

    ``eva.telemetry.Metrics`` считает только дошедшие до маршрутизатора
    запросы, а 413/429/503/504 отклоняются раньше. Их нельзя потерять: по
    этим числам видно, что сервер отказывает, и почему. ``degraded`` —
    сколько запросов ответил не big-маршрут после отказа большого клиента.
    """

    #: Имена счётчиков, которые всегда есть в снимке (нули не пропадают).
    NAMES = (
        "prompt_too_large",
        "rate_limited",
        "degraded",
        "unavailable",
        "timeout",
    )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts = {name: 0 for name in self.NAMES}
        self._degraded_by_stage: dict[str, int] = {}

    def bump(self, name: str, *, stage: str | None = None) -> None:
        """Увеличить счётчик; неизвестное имя игнорируется (не 500 из метрики)."""
        if name not in self._counts:
            return
        with self._lock:
            self._counts[name] += 1
            if stage:
                self._degraded_by_stage[str(stage)] = (
                    self._degraded_by_stage.get(str(stage), 0) + 1
                )

    def snapshot(self) -> dict[str, Any]:
        """Срез счётчиков: все имена, даже нулевые, плюс разбивка деградации."""
        with self._lock:
            return {
                **self._counts,
                "degraded_by_stage": dict(sorted(self._degraded_by_stage.items())),
            }


# --------------------------------------------------------------------------- #
# Типы запроса
# --------------------------------------------------------------------------- #


def build_request(pairs: list[tuple[str, str]], meta: dict[str, Any]) -> Any:
    """Собирает ``eva.types.Request`` из пар (role, content) и метаданных."""
    from .types import Message, Request

    return Request(
        messages=[Message(role=role, content=content) for role, content in pairs],
        meta=meta,
    )


def estimate_prompt_tokens(pairs: Iterable[tuple[str, str]]) -> int:
    """Оценка токенов промпта: ~4 символа на токен, минимум 1 на сообщение.

    Точного токенизатора у HTTP-сервера нет (он не должен тянуть модель ради
    проверки длины), поэтому берётся та же дешёвая оценка, что и в
    ``eva.policy.estimate_tokens``. Пустые сообщения не считаются: они не
    уходят модели как содержательный промпт.
    """
    total = 0
    for _role, content in pairs:
        if isinstance(content, str) and content.strip():
            total += max(1, ceil(len(content) / 4))
    return total


# --------------------------------------------------------------------------- #
# Прогрев локальной модели при старте
# --------------------------------------------------------------------------- #

#: Минимум prefill-токенов, если у клиента нет токенизатора.
_WARMUP_IDS = (1, 2, 3)


def _now(clock: Callable[[], float] | None) -> float:
    """Текущий момент для замера прогрева (часы инжектируются в тестах)."""
    return time.monotonic() if clock is None else float(clock())


def _elapsed_seconds(started: float, clock: Callable[[], float] | None) -> float:
    """Секунды с ``started``; отрицательный результат (часы пошли назад) — 0."""
    return max(0.0, _now(clock) - started)


def _model_name(backend: Any) -> str:
    """Человеческое имя модели для лога и метрик (путь, если есть)."""
    for name in ("model_path", "model"):
        value = getattr(backend, name, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return type(backend).__name__


def _make_warmup_backend(model: str) -> Any:
    """Локальная ВТМ по пути; импорт ленивый — без модели ничего не грузим.

    Ошибка сборки конфига (мусор в ``EVA_VTM_BACKEND``) поднимается наверх:
    вызывающий считает её неудачным прогревом и стартует без него.
    """
    from .backends.vtm_backend import VTMConfig, VtmBackend

    return VtmBackend(VTMConfig.from_env(model=model))


def _warmup_prompt_ids(backend: Any, prompt: str) -> list[int]:
    """id промпта прогрева: через токенизатор клиента, иначе короткий список.

    Если токенизатора нет (рядом с моделью нет ``tokenizer.json``), генерация
    по id всё равно прогревает prefill и арену — прогрев не отменяется.
    """
    encode = getattr(backend, "encode", None)
    if callable(encode):
        try:
            ids = [int(token) for token in encode(prompt)]
        except Exception:  # noqa: BLE001 - прогрев важнее конкретного токенизатора
            logger.debug("прогрев: encode недоступен, работаем по id", exc_info=True)
        else:
            if ids:
                return ids
    return list(_WARMUP_IDS)


def _close_quietly(backend: Any) -> None:
    """Закрыть временный бэкенд прогрева, не портя старт его ошибкой."""
    closer = getattr(backend, "close", None)
    if not callable(closer):
        return
    try:
        closer()
    except Exception:  # noqa: BLE001 - закрытие не должно ронять сервер
        logger.debug("прогрев: временный бэкенд не закрылся", exc_info=True)


def warmup_local_model(
    backend: Any = None,
    *,
    metrics: Any = None,
    prompt: str = WARMUP_PROMPT,
    max_new_tokens: int = WARMUP_MAX_NEW_TOKENS,
    clock: Callable[[], float] | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any] | None:
    """Загрузить локальную модель и прогнать короткий prefill.

    ``backend`` — локальный клиент (``VtmBackend`` или совместимый объект с
    ``open()``/``encode()``/``generate()``); ``None`` — взять модель из
    ``EVA_VTM_MODEL``. Если ни клиента, ни переменной нет, прогревать нечего:
    возвращается ``None``.

    Прогрев намеренно не поднимает исключений: первый запрос не должен
    приносить в жертву старт сервера. Ошибка остаётся предупреждением в
    журнале и записью ``ok: false`` в метриках (``warmup_ms``, причина).

    Возвращает запись прогрева::

        {"ok": True, "warmup_ms": 41.2, "prefill_tokens": 3,
         "model": "demo/model.vtm", "error": None}

    Временный бэкенд, собранный из окружения, закрывается после прогрева:
    страницы mmap, динамическая библиотека и токенизатор остаются горячими,
    а второй контекст не держит лишнюю память. Переданный клиент — чужой:
    его не закрывают.
    """
    source = os.environ if environ is None else environ
    model = ""
    if backend is None:
        model = (source.get(_ENV_VTM_MODEL) or "").strip()
        if not model:
            return None  # локальная модель не задана — прогрев не нужен
    else:
        model = _model_name(backend)

    def remember(*, ok: bool, ms: float, tokens: int, error: str | None) -> dict:
        record = {
            "ok": ok,
            "warmup_ms": round(ms, 3),
            "prefill_tokens": int(tokens),
            "model": model,
            "error": error,
        }
        if metrics is not None:
            try:
                metrics.record_warmup(
                    record["warmup_ms"],
                    ok=ok,
                    prefill_tokens=tokens,
                    error=error,
                    model=model,
                )
            except Exception:  # noqa: BLE001 - метрика не важнее старта
                logger.exception("не удалось записать метрику прогрева")
        return record

    started = _now(clock)
    own_backend = False
    ids: list[int] = []
    try:
        if backend is None:
            backend = _make_warmup_backend(model)
            own_backend = True
        opener = getattr(backend, "open", None)
        if callable(opener):
            opener()
        ids = _warmup_prompt_ids(backend, prompt)
        generate = getattr(backend, "generate", None)
        if not callable(generate):
            raise TypeError("у локального клиента нет метода generate(prompt_ids)")
        generate(ids, max_new_tokens=max(1, int(max_new_tokens)))
    except Exception as exc:  # noqa: BLE001 - сервер стартует и без прогрева
        elapsed_ms = _elapsed_seconds(started, clock) * 1000.0
        reason = str(exc).strip() or type(exc).__name__
        logger.warning(
            "прогрев модели не удался (%s): %s; сервер стартует без прогрева",
            model,
            reason,
        )
        return remember(ok=False, ms=elapsed_ms, tokens=0, error=reason)
    finally:
        if own_backend and backend is not None:
            _close_quietly(backend)

    elapsed_ms = _elapsed_seconds(started, clock) * 1000.0
    logger.info(
        "прогрев модели: %.1f мс, prefill %d токенов (%s)",
        elapsed_ms,
        len(ids),
        model,
    )
    return remember(ok=True, ms=elapsed_ms, tokens=len(ids), error=None)


# --------------------------------------------------------------------------- #
# Маршрутизатор
# --------------------------------------------------------------------------- #

#: Маршрутизатор — функция ``request -> Answer`` (Answer из eva.types).
Router = Callable[[Any], Any]
#: Потоковый маршрутизатор — функция ``request -> iterable`` кусков ответа.
StreamRouter = Callable[[Any], Any]


def as_router(router: Any) -> Router:
    """Приводит инжектированный маршрутизатор к виду ``router(request)``.

    Принимает функцию или объект с методом ``answer(request)``. Модуль
    ``eva.router`` с трёхаргументным ``answer(req, cfg, deps)`` напрямую не
    подходит — для него есть :func:`default_router`.
    """
    if router is None:
        return default_router
    if callable(router):
        return router
    answer = getattr(router, "answer", None)
    if callable(answer):
        return answer
    raise TypeError(
        "router должен быть функцией router(request) -> Answer "
        "или объектом с методом answer(request)"
    )


def as_stream_router(router: Any) -> StreamRouter | None:
    """Достаёт у маршрутизатора необязательный метод ``stream(request)``.

    Маршрутизатор объявляет умение отдавать ответ по частям явно: атрибутом
    ``stream``, который вызывается как ``stream(request)`` и возвращает
    итератор кусков. Куском считается строка (текст дельты), объект/словарь с
    полем ``text`` (тогда текст уходит в ``delta.content``, а ``route``/``usage``
    запоминаются для финального чанка) или словарь с ``delta``. Нет такого
    метода — сервер отдаёт готовый ответ одним чанком.

    Отдельная функция, а не поле в :func:`as_router`: объект с ``answer`` и
    ``stream`` после ``as_router`` превращается в связанный метод и второй
    атрибут теряет.
    """
    if router is None:
        return None
    stream = getattr(router, "stream", None)
    if callable(stream):
        return stream
    return None


def _load_config() -> Any:
    """Конфиг из env: ``config.load()``, если он есть, иначе сам модуль."""
    from . import config as config_module

    load = getattr(config_module, "load", None)
    if callable(load):
        return load()
    return config_module


def default_router(request: Any) -> Any:
    """Боевая связка ``eva.router.answer(req, cfg, deps)``, собранная лениво.

    Лениво — потому что ``/health`` должен отвечать даже тогда, когда
    маршрутизатор ещё не собран; при ошибке клиент получает 500 с человеческим
    текстом, а не трейсбек.
    """
    try:
        from .router import answer
    except ImportError as exc:
        raise RuntimeError(
            "маршрутизатор eva.router не собран: запустите сервер с "
            "router=... или --router МОДУЛЬ:ФАБРИКА"
        ) from exc

    if not callable(answer):
        raise RuntimeError("в eva.router нет функции answer(req, cfg, deps)")
    return answer(request, _load_config(), None)


def default_stream_router(request: Any) -> Any:
    """Боевая потоковая связка ``eva.router.stream(req, cfg, deps)``.

    Отдельная функция, а не атрибут :func:`default_router`: ``as_router``
    превращает объект с ``answer``/``stream`` в связанный метод и теряет
    второй атрибут. Собирается лениво, как и ``default_router``: ``/health``
    и обычный ответ не должны зависеть от готовности потока.
    """
    try:
        from .router import stream
    except ImportError as exc:
        raise RuntimeError(
            "потоковый маршрутизатор eva.router не собран: запустите сервер с "
            "router=... или --router МОДУЛЬ:ФАБРИКА"
        ) from exc

    if not callable(stream):
        raise RuntimeError("в eva.router нет функции stream(req, cfg, deps)")
    return stream(request, _load_config(), None)


# --------------------------------------------------------------------------- #
# Перевод Answer в ответ OpenAI
# --------------------------------------------------------------------------- #


def _field(obj: Any, name: str, default: Any = None) -> Any:
    """Поле объекта или ключ словаря — маршрутизатор может вернуть и то и то."""
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return default


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _elapsed_ms(started: float, now: float | None = None) -> int:
    """Миллисекунды с момента ``started`` (``time.monotonic()``).

    ``now`` подменяет текущий момент: сервер считает время по своим часам
    (``server.clock``), а тесты — по фейковым, чтобы TTFT был детерминирован.
    """
    moment = time.monotonic() if now is None else float(now)
    return int((moment - started) * 1000)


def _route_summary(route: Any) -> dict[str, Any]:
    if route is None:
        return {"route": "unknown", "reason": "", "confidence": 0.0}
    if isinstance(route, str):
        kind, reason, confidence = route, "", 0.0
    else:
        kind = _field(route, "kind", "unknown")
        reason = _field(route, "reason", "")
        confidence = _field(route, "confidence", 0.0)

    if not isinstance(kind, str) or not kind.strip():
        kind = "unknown"
    if not isinstance(reason, str):
        reason = ""
    confidence = max(0.0, min(1.0, _as_float(confidence)))

    return {
        "route": kind.strip(),
        "reason": reason.strip(),
        "confidence": round(confidence, 4),
    }


def _usage_summary(answer: Any) -> dict[str, int]:
    """``usage`` в форме OpenAI из ответа маршрутизатора."""
    raw_usage = _field(answer, "usage") or {}
    if not isinstance(raw_usage, Mapping):
        raw_usage = {}
    prompt_tokens = max(0, _as_int(raw_usage.get("prompt_tokens")))
    completion_tokens = max(0, _as_int(raw_usage.get("completion_tokens")))
    total_tokens = _as_int(raw_usage.get("total_tokens"))
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _prefill_ms_of(source: Any) -> float | None:
    """Prefill-часть ответа, если маршрутизатор её сообщил.

    Мягкий контракт (``eva/router.py`` — чужая зона): ``prefill_ms`` в самом
    ответе или куске меты потока, либо в его ``usage``/``timings``. Нет поля —
    ``None``: границу prefill сервер не угадывает и в метрики не пишет.
    """
    if source is None:
        return None
    raw = _field(source, "prefill_ms")
    if raw is None:
        for name in ("usage", "timings"):
            nested = _field(source, name)
            if isinstance(nested, Mapping):
                raw = nested.get("prefill_ms")
                if raw is not None:
                    break
    if isinstance(raw, bool):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not isfinite(value) or value < 0:
        return None
    return value


def _eva_summary(answer: Any, latency_ms: int) -> dict[str, Any]:
    """Поле ``eva``: маршрут, скилл, цена и латентность ответа."""
    skill = _field(answer, "skill")
    if not isinstance(skill, str) or not skill.strip():
        skill = None
    cost_rub = max(0.0, _as_float(_field(answer, "cost_rub")))

    eva = _route_summary(_field(answer, "route"))
    eva["skill"] = skill
    eva["cost_rub"] = round(cost_rub, 4)
    measured = max(0, _as_int(latency_ms))
    eva["latency_ms"] = max(0, _as_int(_field(answer, "latency_ms"), measured))
    return eva


# --------------------------------------------------------------------------- #
# Мягкая деградация: big → локальная модель → скилл → 503
# --------------------------------------------------------------------------- #


class ServiceUnavailable(RuntimeError):
    """Ни один источник ответа не сработал: сервер отвечает 503.

    ``stages`` — порядок, в котором пробовали источники (``big``, ``local``,
    ``skill``); ``errors`` — короткие причины отказа по стадиям для лога и
    тестов. ``public_message`` — текст для клиента: без трейсбеков и
    внутренних подробностей.
    """

    def __init__(
        self,
        message: str,
        *,
        public_message: str | None = None,
        stages: Iterable[str] | None = None,
        errors: Iterable[tuple[str, str]] | None = None,
    ) -> None:
        super().__init__(message)
        self.public_message = public_message or (
            "Все источники ответа недоступны: большая модель, локальная модель "
            "и скилл. Повторите запрос позже."
        )
        self.stages = list(stages or ())
        self.errors = list(errors or ())


class _RequestTimeout(Exception):
    """Генерация не уложилась в ``EVA_REQUEST_TIMEOUT``."""

    def __init__(self, timeout: float) -> None:
        super().__init__("генерация не уложилась в {:g} с".format(timeout))
        self.timeout = float(timeout)


class DegradationStats:
    """Счётчики одной цепочки деградации (потокобезопасно)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._attempts = {stage: 0 for stage in ("big", "local", "skill")}
        self._failures = {stage: 0 for stage in ("big", "local", "skill")}
        self._served = {stage: 0 for stage in ("big", "local", "skill")}
        self._degraded = 0
        self._unavailable = 0

    def attempt(self, stage: str) -> None:
        with self._lock:
            self._attempts[stage] += 1

    def failure(self, stage: str) -> None:
        with self._lock:
            self._failures[stage] += 1

    def served(self, stage: str, *, degraded: bool) -> None:
        with self._lock:
            self._served[stage] += 1
            if degraded:
                self._degraded += 1

    def unavailable(self) -> None:
        with self._lock:
            self._unavailable += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "attempts": dict(self._attempts),
                "failures": dict(self._failures),
                "served": dict(self._served),
                "degraded": self._degraded,
                "unavailable": self._unavailable,
            }


class DegradationChain:
    """Маршрутизатор с явным порядком: big → локальная модель → скилл.

    ``big`` и ``local`` — клиенты с ``chat(messages, max_tokens=...)`` (или
    вызываемые объекты с той же сигнатурой). ``skills`` — ``try_skills(text,
    ctx)``, ``callable(text, ctx)`` или объект с методом ``try_skills``.
    Отказ любого источника (таймаут, ``ClientError``, исключение) не роняет
    запрос: сервер пробует следующий. Когда не осталось никого, бросается
    :class:`ServiceUnavailable` — обработчик превращает его в 503.

    Цепочка сама считает попытки/отказы (:attr:`stats`), поэтому отметка
    ``degraded`` попадает в ``GET /metrics`` даже без ``eva.telemetry``.
    """

    STAGES = ("big", "local", "skill")

    def __init__(
        self,
        *,
        big: Any = None,
        local: Any = None,
        skills: Any = None,
        max_tokens: int = 512,
    ) -> None:
        self.big = big
        self.local = local
        self.skills = skills
        self.max_tokens = max(1, int(max_tokens))
        self.stats = DegradationStats()

    def __call__(self, request: Any) -> Any:
        return self.answer(request)

    def answer(self, request: Any, *, skip: Iterable[str] = ()) -> Any:
        """Ответить первым доступным источником или бросить 503-исключение.

        ``skip`` — стадии, которые уже пробовал вызывающий (например, сервер
        после падения своего основного маршрутизатора пропускает ``big``).
        """
        skipped = {str(stage) for stage in skip}
        started = time.monotonic()
        messages = list(getattr(request, "messages", ()) or ())
        text = _request_text(request)
        failures: list[tuple[str, str]] = []

        for stage, client in (("big", self.big), ("local", self.local)):
            if stage in skipped:
                continue
            if client is None:
                failures.append((stage, "не настроен"))
                continue
            self.stats.attempt(stage)
            try:
                result = _call_client_chat(client, messages, self.max_tokens)
                answer = _client_answer(
                    result,
                    stage,
                    failures,
                    latency_ms=_elapsed_ms(started),
                )
            except Exception as exc:  # noqa: BLE001 - отказ ведёт к следующей стадии
                reason = _short_error(exc)
                self.stats.failure(stage)
                failures.append((stage, reason))
                logger.warning("деградация: %s недоступен: %s", stage, reason)
                continue
            self.stats.served(stage, degraded=stage != "big")
            return answer

        if "skill" not in skipped and self.skills is not None:
            self.stats.attempt("skill")
            try:
                result = _call_skills_provider(
                    self.skills, text, _skill_context(request)
                )
                answer = _skill_answer(
                    result,
                    failures,
                    latency_ms=_elapsed_ms(started),
                )
            except Exception as exc:  # noqa: BLE001 - скилл не обязан отвечать
                reason = _short_error(exc)
                self.stats.failure("skill")
                failures.append(("skill", reason))
                logger.warning("деградация: скилл не сработал: %s", reason)
                answer = None
            if answer is not None:
                self.stats.served("skill", degraded=True)
                return answer
            if not any(stage == "skill" for stage, _ in failures):
                failures.append(("skill", "не дал ответа"))

        self.stats.unavailable()
        details = ", ".join("{}: {}".format(stage, reason) for stage, reason in failures)
        raise ServiceUnavailable(
            "все источники ответа недоступны ({})".format(details or "нет настроенных"),
            stages=[stage for stage, _ in failures],
            errors=failures,
        )


def _request_text(request: Any) -> str:
    """Текст последнего пользовательского сообщения (для скилла)."""
    try:
        return request.text
    except Exception:
        return ""


def _skill_context(request: Any) -> dict[str, Any]:
    messages = list(getattr(request, "messages", ()) or ())
    meta = dict(getattr(request, "meta", {}) or {})
    return {
        "text": _request_text(request),
        "messages": [
            message.to_dict() if hasattr(message, "to_dict") else dict(message)
            for message in messages
        ],
        "meta": meta,
        "user_id": meta.get("user_id"),
        "channel": meta.get("channel"),
        "request": request,
    }


def _short_error(exc: BaseException, limit: int = 200) -> str:
    text = str(exc).strip() or type(exc).__name__
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _format_seconds(value: float) -> str:
    """Секунды для человеческого текста: ``0.5``, ``60`` — без хвостовых нулей."""
    return "{:g}".format(float(value))


def _call_client_chat(client: Any, messages: list[Any], max_tokens: int) -> Any:
    """Позвать клиент модели; подписи ``chat(messages)`` и ``chat(m, max_tokens)``."""
    chat = getattr(client, "chat", None)
    target = chat if callable(chat) else client
    if not callable(target):
        raise TypeError(
            "клиент модели должен иметь chat(messages, max_tokens=...) или быть "
            "вызываемым"
        )
    try:
        inspect.signature(target).bind(messages, max_tokens=max_tokens)
    except (TypeError, ValueError):
        # Старый/упрощённый клиент без max_tokens: зовём как умеет.
        return target(messages)
    return target(messages, max_tokens=max_tokens)


def _call_skills_provider(provider: Any, text: str, ctx: dict[str, Any]) -> Any:
    method = getattr(provider, "try_skills", None)
    if callable(method):
        return method(text, ctx)
    if callable(provider):
        return provider(text, ctx)
    raise TypeError("skills: ожидался try_skills(text, ctx) или вызываемый объект")


def _client_answer(
    result: Any,
    stage: str,
    failures: list[tuple[str, str]],
    *,
    latency_ms: int,
) -> Any:
    """Ответ клиента модели -> ``eva.types.Answer`` с причиной деградации."""
    from .types import Answer, Route

    if isinstance(result, str):
        text = result
        usage: Any = {}
        cost_rub = 0.0
    else:
        text = _field(result, "text")
        if text is None:
            text = _field(result, "content")
        usage = _field(result, "usage") or {}
        cost_rub = _as_float(_field(result, "cost_rub"))

    if not isinstance(text, str) or not text:
        raise TypeError("клиент {} не вернул текст ответа".format(stage))

    usage_map = usage if isinstance(usage, Mapping) else {}
    prompt_tokens = max(0, _as_int(usage_map.get("prompt_tokens")))
    completion_tokens = max(0, _as_int(usage_map.get("completion_tokens")))
    cost = cost_rub or max(0.0, _as_float(usage_map.get("cost_rub")))

    kind = "big" if stage == "big" else "small"
    if stage == "big":
        reason = "ответила большая модель"
    else:
        failed = "; ".join("{}: {}".format(name, why) for name, why in failures)
        reason = (
            "деградация: большая модель недоступна ({}) — ответила локальная "
            "модель".format(failed or "причина неизвестна")
        )

    return Answer(
        text=text,
        route=Route(kind=kind, reason=reason, confidence=0.5 if kind == "small" else 0.9),
        skill=None,
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_rub": round(float(cost), 6),
        },
        cost_rub=round(float(cost), 6),
        latency_ms=max(0, int(latency_ms)),
    )


def _skill_answer(
    result: Any,
    failures: list[tuple[str, str]],
    *,
    latency_ms: int,
) -> Any:
    """Ответ скилла -> ``eva.types.Answer``; пустой результат — ``None``."""
    from .types import Answer, Route

    if result is None or result is False:
        return None
    if isinstance(result, str):
        text = result
        name = None
    elif isinstance(result, Mapping):
        text = result.get("text")
        name = result.get("skill") or result.get("name")
    else:
        text = getattr(result, "text", None)
        name = getattr(result, "skill", None) or getattr(result, "name", None)

    if not isinstance(text, str) or not text:
        return None
    skill = str(name).strip() if name else None
    failed = "; ".join("{}: {}".format(stage, why) for stage, why in failures)
    reason = (
        "деградация: большая и локальная модели недоступны ({}) — ответил "
        "скилл{}".format(
            failed or "причина неизвестна", " «{}»".format(skill) if skill else ""
        )
    )
    return Answer(
        text=text,
        route=Route(kind="skill", reason=reason, confidence=1.0),
        skill=skill,
        usage={"prompt_tokens": 0, "completion_tokens": 0, "cost_rub": 0.0},
        cost_rub=0.0,
        latency_ms=max(0, int(latency_ms)),
    )


def chat_chunk(
    *,
    chunk_id: str,
    created: int,
    model: str,
    delta: Mapping[str, Any],
    finish_reason: str | None = None,
    eva: Mapping[str, Any] | None = None,
    usage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Один SSE-чанк в формате OpenAI (``chat.completion.chunk``).

    ``delta`` всегда содержит ``content`` (пустая строка, если дельты нет):
    клиент не должен проверять наличие ключа. ``finish_reason`` не ``None``
    только в последнем чанке ответа.
    """
    payload: dict[str, Any] = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {"index": 0, "delta": dict(delta), "finish_reason": finish_reason}
        ],
    }
    if usage is not None:
        payload["usage"] = dict(usage)
    if eva is not None:
        payload["eva"] = dict(eva)
    return payload


def reply_to_openai(
    answer: Any,
    *,
    model: str,
    request_id: str,
    latency_ms: int,
) -> dict[str, Any]:
    """Строит тело ответа OpenAI из ``eva.types.Answer`` (или словаря)."""
    text = _field(answer, "text")
    if not isinstance(text, str):
        raise TypeError("маршрутизатор вернул ответ без текста")

    return {
        "id": "chatcmpl-{}".format(request_id[:24]),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": _usage_summary(answer),
        "eva": _eva_summary(answer, latency_ms),
    }


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #


class _ClientError(Exception):
    """Ошибка клиента: статус, код и текст для ответа."""

    def __init__(
        self,
        message: str,
        *,
        status: int = 400,
        code: str = "invalid_request",
        headers: Mapping[str, str] | None = None,
    ):
        super().__init__(message)
        self.status = status
        self.code = code
        self.headers = dict(headers or {})


def _content_to_text(content: Any) -> str | None:
    """Текст сообщения: строка или список частей ``{"type": "text", "text": ...}``."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if not isinstance(part, Mapping):
                return None
            text = part.get("text")
            if not isinstance(text, str):
                return None
            parts.append(text)
        return "".join(parts)
    return None


def _coerce_stream_piece(item: Any) -> tuple[str, Any]:
    """Кусок потока -> ``(текст дельты, Answer-метаданные | None)``.

    Понимает строку, объект/словарь с ``text`` (Answer-подобный: у него есть
    ``route``, метаданные пойдут в финальный чанк), ``delta`` строкой и
    ``{"delta": {"content": ...}}``. Всё прочее — ошибка маршрутизатора.
    """
    if isinstance(item, str):
        return item, None
    if isinstance(item, bytes):
        try:
            return item.decode("utf-8"), None
        except UnicodeDecodeError as exc:
            raise TypeError("кусок потока не в UTF-8") from exc

    text = _field(item, "text")
    if isinstance(text, str):
        meta = item if _field(item, "route") is not None else None
        return text, meta

    delta = _field(item, "delta")
    if isinstance(delta, str):
        return delta, None
    if isinstance(delta, Mapping):
        content = delta.get("content")
        if isinstance(content, str):
            return content, None

    content = _field(item, "content")
    if isinstance(content, str):
        return content, None

    raise TypeError(
        "кусок потока должен быть строкой или объектом с текстом: {!r}".format(item)
    )


def _close_iterator(iterator: Any) -> None:
    """Закрыть поток роутера, если он это умеет; ошибки закрытия не важны."""
    close = getattr(iterator, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # noqa: BLE001 - закрытие не должно ронять сервер
            logger.debug("ошибка закрытия потока", exc_info=True)


class _StreamPump:
    """Тянет куски генератора в отдельном потоке, чтобы работал таймаут.

    ``next(generator)`` нельзя прервать снаружи: пока провайдер «думает»,
    поток сервера висит. Насос вызывает генератор сам, а обработчик забирает
    готовые куски из очереди с дедлайном. Очередь на один элемент сохраняет
    обратное давление: генератор не убегает вперёд, пока сервер не отправил
    предыдущий кусок. ``cancel()`` просит насос остановиться и закрывает
    генератор; если тот уже выполняется, закрытие отложится до его паузы.
    """

    def __init__(self, iterator: Any) -> None:
        self._iterator = iterator
        self._queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)
        self._cancel = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="eva-stream-pump", daemon=True
        )

    def start(self) -> "_StreamPump":
        self._thread.start()
        return self

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def _run(self) -> None:
        try:
            for item in self._iterator:
                if self._cancel.is_set():
                    return
                while not self._cancel.is_set():
                    try:
                        self._queue.put(("item", item), timeout=0.05)
                        break
                    except queue.Full:
                        continue
            if not self._cancel.is_set():
                try:
                    self._queue.put(("end", None), timeout=0.05)
                except queue.Full:
                    pass
        except Exception as exc:  # noqa: BLE001 - ошибку отдаём обработчику
            if not self._cancel.is_set():
                try:
                    self._queue.put(("error", exc), timeout=0.05)
                except queue.Full:
                    pass
        finally:
            _close_iterator(self._iterator)

    def next_item(self, deadline: float | None) -> Any:
        """Следующий кусок, ``_STREAM_END`` или :class:`_RequestTimeout`."""
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            self.cancel()
            raise _RequestTimeout(0.0)
        try:
            kind, payload = self._queue.get(timeout=remaining)
        except queue.Empty:
            self.cancel()
            raise _RequestTimeout(0.0) from None
        if kind == "item":
            return payload
        if kind == "end":
            return _STREAM_END
        if kind == "error":
            raise payload
        return _STREAM_END

    def cancel(self) -> None:
        """Остановить насос и закрыть генератор (насколько это возможно)."""
        self._cancel.set()
        _close_iterator(self._iterator)


def _call_with_timeout(function: Callable[[Any], Any], request: Any, timeout: float) -> Any:
    """Позвать ``function(request)`` и не ждать дольше ``timeout`` секунд.

    Поток-исполнитель объявлен демоном: по таймауту сервер возвращает 504 и
    перестаёт ждать. Убить чужой синхронный вызов из Python нельзя, поэтому
    «отмена» здесь — прекращение ожидания и закрытие потокового генератора
    там, где он есть; фоновый вызов доживёт до своего ответа и будет отброшен.
    """
    done = threading.Event()
    box: list[tuple[bool, Any]] = []

    def run() -> None:
        try:
            box.append((True, function(request)))
        except Exception as exc:  # noqa: BLE001 - ошибку отдаём вызывающему
            box.append((False, exc))
        finally:
            done.set()

    worker = threading.Thread(target=run, name="eva-request", daemon=True)
    worker.start()
    if not done.wait(timeout):
        raise _RequestTimeout(timeout)
    if not box:
        raise RuntimeError("вызов маршрутизатора завершился без результата")
    ok, value = box[0]
    if ok:
        return value
    raise value


class EvaRequestHandler(BaseHTTPRequestHandler):
    """Один HTTP-запрос. Маршрутизатор задаётся подклассом из :func:`make_handler`."""

    server_version = "EVA32/1.0"
    protocol_version = "HTTP/1.1"
    _router: Router = staticmethod(default_router)
    #: Необязательный ``stream(request)``; ``None`` — стриминга у роутера нет.
    _stream_router: StreamRouter | None = None
    #: Тело SSE пишется chunked (HTTP/1.1) или до закрытия соединения (1.0).
    _sse_chunked = True

    # -- маршрутизация методов ------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - имя задано BaseHTTPRequestHandler
        if not self._check_access():
            return
        path = self._path()
        if path == "/health":
            self._send_health()
        elif path == "/metrics":
            self._send_metrics()
        elif path == "/v1/models":
            self._send_models()
        elif path == "/v1/chat/completions":
            self._send_method_not_allowed("POST")
        else:
            self._send_not_found(path)

    def do_POST(self) -> None:  # noqa: N802
        if not self._check_access():
            return
        path = self._path()
        if path == "/v1/chat/completions":
            self._handle_chat()
        elif path in ("/health", "/v1/models", "/metrics"):
            self._send_method_not_allowed("GET")
        else:
            self._send_not_found(path)

    def do_PUT(self) -> None:  # noqa: N802
        if self._check_access():
            self._handle_unsupported_method()

    def do_PATCH(self) -> None:  # noqa: N802
        if self._check_access():
            self._handle_unsupported_method()

    def do_DELETE(self) -> None:  # noqa: N802
        if self._check_access():
            self._handle_unsupported_method()

    def do_OPTIONS(self) -> None:  # noqa: N802
        if not self._check_access():
            return
        path = self._path()
        allow = self._allowed_methods(path)
        if allow is None:
            self._send_not_found(path)
        else:
            self._send_no_content(headers={"Allow": allow})

    # -- доступ ---------------------------------------------------------------

    def _bearer_token(self) -> str | None:
        """Токен из ``Authorization: Bearer ...``; ``None`` — заголовка нет."""
        raw = self.headers.get(AUTH_HEADER)
        if not isinstance(raw, str):
            return None
        scheme, separator, value = raw.strip().partition(" ")
        if not separator or scheme.lower() != _BEARER:
            return None
        token = value.strip()
        return token or None

    def _check_access(self) -> bool:
        """Аутентификация и лимит запросов; ``False`` — ответ уже отправлен.

        ``GET /health`` открыт намеренно: по нему проверяют живость процесса,
        и он ничего не стоит. Всё остальное (включая ``/v1/models``,
        ``/metrics`` и незнакомые пути) при заданном ``EVA_AUTH_TOKEN`` требует
        Bearer-токен.
        Проверка идёт до чтения тела: без токена байты не читаем и 413 не
        выдаём — сначала 401.

        Лимит считается **по маршруту** (пути): исчерпанный бюджет чата не
        мешает читать ``/metrics``, а разные эндпоинты не отнимают квоту друг
        у друга. Ключ в ``AccessGuard`` — путь + токен/адрес клиента.
        """
        if self._path() == "/health":
            return True
        guard: AccessGuard | None = getattr(self.server, "access_guard", None)
        if guard is None:
            return True

        presented = self._bearer_token()
        if not guard.authorized(presented):
            logger.warning(
                "%s: отказ в доступе: нет валидного заголовка %s: Bearer",
                self.address_string(),
                AUTH_HEADER,
            )
            self._send_error(
                401,
                "Нужен заголовок Authorization: Bearer <токен>. Токен задаётся "
                "на сервере переменной EVA_AUTH_TOKEN.",
                "unauthorized",
                headers={"WWW-Authenticate": 'Bearer realm="eva"'},
            )
            return False

        route = self._path()
        key = "{}|{}".format(route, guard.key_for(presented, self.client_address[0]))
        retry_after = guard.check(key)
        if retry_after <= 0.0:
            return True

        seconds = max(1, int(ceil(retry_after)))
        self._count("rate_limited")
        logger.warning(
            "%s: превышен лимит %s запросов в минуту на %s, повтор через %s с",
            self.address_string(),
            guard.rate_limit_per_min,
            route,
            seconds,
        )
        self._send_error(
            429,
            "Превышен лимит запросов к {}: {} в минуту на ключ. "
            "Повторите через {} с.".format(route, guard.rate_limit_per_min, seconds),
            "rate_limited",
            headers={
                "Retry-After": str(seconds),
                "X-RateLimit-Limit": str(guard.rate_limit_per_min),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Route": route,
            },
        )
        return False

    # -- эндпоинты ------------------------------------------------------------

    def _send_health(self) -> None:
        started_at = getattr(self.server, "started_at", None)
        uptime = round(time.monotonic() - started_at, 3) if started_at else 0.0
        self._send_json(
            200,
            {
                "status": "ok",
                "service": SERVICE_NAME,
                "version": version(),
                "model": DEFAULT_MODEL,
                "router": getattr(self.server, "router_name", "unknown"),
                "uptime_s": uptime,
            },
        )

    def _send_models(self) -> None:
        self._send_json(
            200,
            {
                "object": "list",
                "data": [
                    {
                        "id": DEFAULT_MODEL,
                        "object": "model",
                        "created": 0,
                        "owned_by": "eva32",
                    }
                ],
            },
        )

    def _send_metrics(self) -> None:
        """``GET /metrics``: счётчики и перцентили работающего сервера."""
        metrics = getattr(self.server, "metrics", None)
        snapshot = (
            metrics.snapshot() if metrics is not None else Metrics().snapshot()
        )
        started_at = getattr(self.server, "started_at", None)
        payload: dict[str, Any] = {
            "service": SERVICE_NAME,
            "version": version(),
            "uptime_s": (
                round(self._clock() - started_at, 3)
                if started_at is not None
                else 0.0
            ),
        }
        payload.update(snapshot)
        counters = getattr(self.server, "counters", None)
        if counters is not None:
            limits = counters.snapshot()
            payload["limits"] = limits
            # Короткая отметка для дашборда: был ли хоть один degraded-ответ.
            payload["degraded"] = limits["degraded"] > 0
        chain = getattr(self.server, "degradation", None)
        if chain is not None and hasattr(chain, "stats"):
            payload["degradation"] = chain.stats.snapshot()
        self._send_json(200, payload)

    # -- метрики запроса ------------------------------------------------------

    def _clock(self) -> float:
        """Текущий момент по часам сервера; в тестах — фейковые часы."""
        clock = getattr(self.server, "clock", None)
        return time.monotonic() if clock is None else float(clock())

    def _observe(self, **fields: Any) -> None:
        """Учесть запрос в метриках; сбой учёта не должен ронять ответ."""
        metrics = getattr(self.server, "metrics", None)
        if metrics is None:
            return
        try:
            metrics.observe(**fields)
        except Exception:  # noqa: BLE001 - метрика не важнее ответа клиенту
            logger.exception("не удалось записать метрику запроса")

    def _count(self, name: str, *, stage: str | None = None) -> None:
        """Учесть отказ сервера (413/429/503/504) в ``GET /metrics``."""
        counters = getattr(self.server, "counters", None)
        if counters is None:
            return
        try:
            counters.bump(name, stage=stage)
        except Exception:  # noqa: BLE001 - счётчик не важнее ответа
            logger.exception("не удалось записать счётчик отказа")

    def _max_prompt_tokens(self) -> int:
        try:
            return max(0, int(getattr(self.server, "max_prompt_tokens", 0) or 0))
        except (TypeError, ValueError):
            return 0

    def _request_timeout(self) -> float:
        try:
            return max(0.0, float(getattr(self.server, "request_timeout", 0.0) or 0.0))
        except (TypeError, ValueError):
            return 0.0

    def _deadline(self) -> float | None:
        """Монотонный дедлайн запроса; ``None`` — таймаут выключен."""
        timeout = self._request_timeout()
        return None if timeout <= 0 else time.monotonic() + timeout

    def _check_prompt_limit(self, pairs: list[tuple[str, str]]) -> None:
        """413, если оценка промпта больше ``EVA_MAX_PROMPT_TOKENS``."""
        limit = self._max_prompt_tokens()
        if limit <= 0:
            return
        tokens = estimate_prompt_tokens(pairs)
        if tokens <= limit:
            return
        self._count("prompt_too_large")
        raise _ClientError(
            "Промпт слишком длинный: примерно {} токенов при лимите {}. "
            "Сократите сообщения или поднимите EVA_MAX_PROMPT_TOKENS на "
            "сервере.".format(tokens, limit),
            status=413,
            code="prompt_too_large",
            headers={"X-Max-Prompt-Tokens": str(limit)},
        )

    def _answer_with_timeout(self, request: Any, deadline: float | None = None) -> Any:
        """Ответ маршрутизатора с таймаутом ``EVA_REQUEST_TIMEOUT``."""
        timeout = self._request_timeout()
        if timeout <= 0:
            return self._router(request)
        if deadline is not None:
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                raise _RequestTimeout(self._request_timeout())
        return _call_with_timeout(self._router, request, timeout)

    def _fallback_answer(
        self,
        request: Any,
        chain: DegradationChain,
        deadline: float | None,
    ) -> Any:
        """Запасная цепочка (local → skill) с остатком общего дедлайна."""
        if deadline is None:
            return chain.answer(request, skip=("big",))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _RequestTimeout(self._request_timeout())
        return _call_with_timeout(
            lambda req: chain.answer(req, skip=("big",)), request, remaining
        )

    def _count_degraded(self, answer: Any) -> None:
        """Отметить ответ, пришедший не от big-маршрута (после отказа big)."""
        kind = _route_summary(_field(answer, "route"))["route"]
        if kind in ("small", "skill"):
            self._count("degraded", stage=kind)

    def _bump_degraded(self, answer: Any) -> None:
        """Отметить деградацию, если цепочка сама была основным маршрутизатором.

        Обычный ответ скилла от боевого ``eva.router`` деградацией не является:
        скилл мог сработать первым. Поэтому считаем только тогда, когда
        ``DegradationChain`` стоит на месте ``_router`` и уже прошла стадию big.
        """
        chain = getattr(self.server, "degradation", None)
        if chain is None:
            return
        router = self._router
        if router is not chain and getattr(router, "__self__", None) is not chain:
            return
        self._count_degraded(answer)

    def _handle_chat(self) -> None:
        request_id = uuid.uuid4().hex
        started = self._clock()

        try:
            payload = self._read_json_body()
            pairs, meta, model, stream, include_usage = self._parse_chat(payload)
            self._check_prompt_limit(pairs)
        except _ClientError as exc:
            logger.info("запрос %s отклонён (%s): %s", request_id, exc.code, exc)
            self._send_error(
                exc.status, str(exc), exc.code, headers=exc.headers or None
            )
            return

        # Токены промпта считает сервер: при обрыве потока usage неизвестен, а
        # длина промпта известна — по ней и пополняется счётчик prefill_tokens.
        prompt_tokens = estimate_prompt_tokens(pairs)

        if stream:
            self._handle_chat_stream(
                pairs,
                meta,
                model=model,
                include_usage=include_usage,
                request_id=request_id,
                started=started,
                prompt_tokens=prompt_tokens,
            )
            return

        request: Any = None
        deadline = self._deadline()
        try:
            request = build_request(pairs, meta)
            answer = self._answer_with_timeout(request, deadline)
        except _RequestTimeout:
            self._count("timeout")
            logger.warning(
                "запрос %s: генерация не уложилась в %s с",
                request_id,
                self._request_timeout(),
            )
            self._observe(
                route="unknown",
                finish="error",
                latency_ms=_elapsed_ms(started, self._clock()),
                prefill_tokens=prompt_tokens,
                stream=False,
            )
            self._send_timeout(request_id)
            return
        except ServiceUnavailable as exc:
            self._count("unavailable")
            logger.warning("запрос %s: все источники недоступны: %s", request_id, exc)
            self._observe(
                route="unknown",
                finish="error",
                latency_ms=_elapsed_ms(started, self._clock()),
                prefill_tokens=prompt_tokens,
                stream=False,
            )
            self._send_unavailable(request_id, exc)
            return
        except Exception:  # noqa: BLE001 - наружу только текст, трейсбек в лог
            logger.exception("ошибка обработки запроса %s", request_id)
            chain = getattr(self.server, "degradation", None)
            if chain is not None and request is not None:
                # Основной маршрутизатор (big-путь) упал — пробуем локальную
                # модель и скилл; big уже пробовали и не повторяем.
                try:
                    answer = self._fallback_answer(request, chain, deadline)
                except _RequestTimeout:
                    self._count("timeout")
                    logger.warning(
                        "запрос %s: деградация не уложилась в %s с",
                        request_id,
                        self._request_timeout(),
                    )
                    self._observe(
                        route="unknown",
                        finish="error",
                        latency_ms=_elapsed_ms(started, self._clock()),
                        prefill_tokens=prompt_tokens,
                        stream=False,
                    )
                    self._send_timeout(request_id)
                    return
                except ServiceUnavailable as exc:
                    self._count("unavailable")
                    logger.warning(
                        "запрос %s: деградация не помогла: %s", request_id, exc
                    )
                    self._observe(
                        route="unknown",
                        finish="error",
                        latency_ms=_elapsed_ms(started, self._clock()),
                        prefill_tokens=prompt_tokens,
                        stream=False,
                    )
                    self._send_unavailable(request_id, exc)
                    return
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "ошибка деградации запроса %s", request_id
                    )
                    self._observe(
                        route="unknown",
                        finish="error",
                        latency_ms=_elapsed_ms(started, self._clock()),
                        prefill_tokens=prompt_tokens,
                        stream=False,
                    )
                    self._send_internal_error(request_id)
                    return
                self._count_degraded(answer)
            else:
                self._observe(
                    route="unknown",
                    finish="error",
                    latency_ms=_elapsed_ms(started, self._clock()),
                    prefill_tokens=prompt_tokens,
                    stream=False,
                )
                self._send_internal_error(request_id)
                return

        self._bump_degraded(answer)
        latency_ms = _elapsed_ms(started, self._clock())
        try:
            body = reply_to_openai(
                answer,
                model=model,
                request_id=request_id,
                latency_ms=latency_ms,
            )
        except Exception:  # noqa: BLE001 - маршрутизатор вернул мусор
            logger.exception("ошибка обработки запроса %s", request_id)
            self._observe(
                route="unknown",
                finish="error",
                latency_ms=_elapsed_ms(started, self._clock()),
                prefill_tokens=prompt_tokens,
                stream=False,
            )
            self._send_internal_error(request_id)
            return

        eva = body.get("eva") or {}
        usage = body.get("usage") or {}
        self._observe(
            route=eva.get("route", "unknown"),
            finish="ok",
            ttft_ms=None,  # у не потокового ответа нет «первого чанка»
            latency_ms=latency_ms,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens"),
            prefill_ms=_prefill_ms_of(answer),
            prefill_tokens=prompt_tokens,
            stream=False,
        )
        self._send_json(
            200,
            body,
            headers={
                "X-Request-Id": request_id,
                "X-Prefill-Tokens": str(prompt_tokens),
            },
        )

    # -- потоковая выдача -----------------------------------------------------

    def _handle_chat_stream(
        self,
        pairs: list[tuple[str, str]],
        meta: dict[str, Any],
        *,
        model: str,
        include_usage: bool,
        request_id: str,
        started: float,
        prompt_tokens: int = 0,
    ) -> None:
        """SSE: куски из ``router.stream(request)`` или готовый ответ одним чанком.

        Всё, что может упасть до первого чанка (сборка запроса, первый кусок),
        отдаётся обычным JSON-ответом: 500 — ошибка, 503 — все источники
        недоступны, 504 — не уложились в ``EVA_REQUEST_TIMEOUT``. Ошибка после
        первого чанка уже не может стать JSON — уходит событием
        ``data: {"error": ...}`` и ``[DONE]``.

        Таймаут действует и на поток: куски вытягиваются насосом в отдельном
        потоке, а обработчик ждёт их с дедлайном. По истечении дедлайна
        генератор закрывается, сервер отвечает 504 (до первого чанка) или
        событием ошибки (после).

        Обрыв SSE клиентом — не ошибка и не повод продолжать генерацию: первая
        же неудачная запись в сокет помечает запрос ``cancel``, поток роутера
        закрывается (генератор получает ``GeneratorExit``), лишние куски не
        запрашиваются. Запрос учитывается в метриках ровно один раз — в
        ``finally``, чем бы поток ни закончился.
        """
        body: dict[str, Any] | None = None
        iterator: Any = None
        pump: _StreamPump | None = None
        answer_meta: Any = None
        request: Any = None
        route = "unknown"
        finish = "error"
        ttft_ms: int | None = None
        # Граница prefill: её сообщает маршрутизатор (``prefill_ms`` в мете или
        # ответе). Без цифры TTFT не делим и остаток не выдумываем.
        prefill_ms: float | None = None
        deadline = self._deadline()

        def mark_ttft() -> None:
            """TTFT — от начала запроса до первого успешно отданного чанка."""
            nonlocal ttft_ms
            if ttft_ms is None:
                ttft_ms = _elapsed_ms(started, self._clock())

        try:
            try:
                request = build_request(pairs, meta)
                stream_fn = self._stream_router
                if stream_fn is None:
                    # Роутер не умеет по частям — отдаём готовый ответ одним чанком
                    # (честно: это не поток, а один кусок уже сгенерированного текста).
                    answer = self._answer_with_timeout(request, deadline)
                    body = reply_to_openai(
                        answer,
                        model=model,
                        request_id=request_id,
                        latency_ms=_elapsed_ms(started, self._clock()),
                    )
                    first_text = body["choices"][0]["message"]["content"]
                    first_meta = None
                    route = body["eva"]["route"]
                    prefill_ms = _prefill_ms_of(answer)
                    self._bump_degraded(answer)
                else:
                    if deadline is not None and not inspect.isgeneratorfunction(
                        stream_fn
                    ):
                        # Фабрика потока (не генератор) может зависнуть ещё до
                        # первого куска: её тоже ограничиваем дедлайном.
                        remaining = deadline - time.monotonic()
                        iterator = _call_with_timeout(
                            lambda req: iter(stream_fn(req)), request, remaining
                        )
                    else:
                        iterator = iter(stream_fn(request))
                    if deadline is not None:
                        # Без таймаута оставляем прежний путь: генератор тянет
                        # сам обработчик, обратное давление не меняется.
                        pump = _StreamPump(iterator).start()
                        first = pump.next_item(deadline)
                    else:
                        first = next(iterator, _STREAM_END)
                    if first is _STREAM_END:
                        first_text, first_meta = "", None
                    else:
                        first_text, first_meta = _coerce_stream_piece(first)
                        if first_meta is not None:
                            route = _route_summary(
                                _field(first_meta, "route")
                            )["route"]
                            prefill_ms = _prefill_ms_of(first_meta)
            except (_RequestTimeout, ServiceUnavailable):
                raise
            except Exception:  # noqa: BLE001 - до первого чанка ещё можно ответить JSON
                logger.exception("ошибка обработки запроса %s", request_id)
                chain = getattr(self.server, "degradation", None)
                if chain is None or request is None:
                    _close_iterator(iterator)
                    self._send_internal_error(request_id)
                    return
                # Основной (big) путь упал: пробуем локальную модель и скилл.
                if pump is not None:
                    pump.cancel()
                _close_iterator(iterator)
                try:
                    answer = self._fallback_answer(request, chain, deadline)
                except (_RequestTimeout, ServiceUnavailable):
                    raise
                except Exception:  # noqa: BLE001
                    logger.exception("ошибка деградации запроса %s", request_id)
                    self._send_internal_error(request_id)
                    return
                self._count_degraded(answer)
                body = reply_to_openai(
                    answer,
                    model=model,
                    request_id=request_id,
                    latency_ms=_elapsed_ms(started, self._clock()),
                )
                first_text = body["choices"][0]["message"]["content"]
                first_meta = None
                route = body["eva"]["route"]
                prefill_ms = _prefill_ms_of(answer)

            try:
                self._open_sse(request_id, prefill_tokens=prompt_tokens)
            except _CLIENT_GONE:
                # Клиент ушёл до заголовков: генерацию не начинаем.
                logger.debug("клиент закрыл соединение до начала стрима")
                self.close_connection = True
                finish = "cancel"
                return
            created = int(time.time())
            chunk_id = "chatcmpl-{}".format(request_id[:24])

            if body is not None:
                if not self._send_sse_data(
                    chat_chunk(
                        chunk_id=chunk_id,
                        created=created,
                        model=model,
                        delta={"role": "assistant", "content": first_text},
                        finish_reason="stop",
                        eva=body["eva"],
                    )
                ):
                    finish = "cancel"
                    return
                mark_ttft()
                if include_usage and not self._send_sse_usage(
                    chunk_id, created, model, body["usage"]
                ):
                    finish = "cancel"
                    return
                if not self._send_sse_done():
                    finish = "cancel"
                    return
                finish = "stop"
                return

            answer_meta = first_meta
            sent_any = False
            try:
                if first_text:
                    if not self._send_sse_data(
                        chat_chunk(
                            chunk_id=chunk_id,
                            created=created,
                            model=model,
                            delta={"role": "assistant", "content": first_text},
                        )
                    ):
                        finish = "cancel"
                        return
                    sent_any = True
                    mark_ttft()

                while True:
                    item = (
                        pump.next_item(deadline)
                        if pump is not None
                        else next(iterator, _STREAM_END)
                    )
                    if item is _STREAM_END:
                        break
                    text, item_meta = _coerce_stream_piece(item)
                    if item_meta is not None:
                        answer_meta = item_meta
                        route = _route_summary(_field(item_meta, "route"))["route"]
                        reported = _prefill_ms_of(item_meta)
                        if reported is not None:
                            prefill_ms = reported
                    if not text:
                        continue
                    delta: dict[str, Any] = {"content": text}
                    if not sent_any:
                        delta["role"] = "assistant"
                    if not self._send_sse_data(
                        chat_chunk(
                            chunk_id=chunk_id,
                            created=created,
                            model=model,
                            delta=delta,
                        )
                    ):
                        finish = "cancel"
                        return
                    sent_any = True
                    mark_ttft()

                eva = (
                    _eva_summary(answer_meta, _elapsed_ms(started, self._clock()))
                    if answer_meta is not None
                    else None
                )
                if not self._send_sse_data(
                    chat_chunk(
                        chunk_id=chunk_id,
                        created=created,
                        model=model,
                        delta={"content": ""},
                        finish_reason="stop",
                        eva=eva,
                    )
                ):
                    finish = "cancel"
                    return
                mark_ttft()
                if include_usage and answer_meta is not None:
                    if not self._send_sse_usage(
                        chunk_id, created, model, _usage_summary(answer_meta)
                    ):
                        finish = "cancel"
                        return
                if not self._send_sse_done():
                    finish = "cancel"
                    return
                finish = "stop"
            except _RequestTimeout:
                self._count("timeout")
                logger.warning("поток %s: таймаут генерации", request_id)
                if self._send_sse_error(
                    request_id,
                    code="request_timeout",
                    message=(
                        "Генерация не уложилась в {} с и была отменена. "
                        "Сократите запрос или увеличьте EVA_REQUEST_TIMEOUT на "
                        "сервере.".format(_format_seconds(self._request_timeout()))
                    ),
                ):
                    self._send_sse_done()
                finish = "error"
            except Exception:  # noqa: BLE001 - заголовки уже отправлены, только SSE
                logger.exception("ошибка потока %s", request_id)
                if self._send_sse_error(request_id):
                    self._send_sse_done()
                finish = "error"
            finally:
                # Клиент мог уйти или поток оборваться: закрываем генератор
                # роутера, чтобы клиент провайдера не висел до сборки мусора.
                if pump is not None:
                    pump.cancel()
                _close_iterator(iterator)
        except _RequestTimeout:
            self._count("timeout")
            logger.warning(
                "поток %s: генерация не уложилась в %s с",
                request_id,
                self._request_timeout(),
            )
            self._send_timeout(request_id)
        except ServiceUnavailable as exc:
            self._count("unavailable")
            logger.warning("поток %s: все источники недоступны: %s", request_id, exc)
            self._send_unavailable(request_id, exc)
        finally:
            if pump is not None:
                pump.cancel()
            if answer_meta is not None:
                usage = _usage_summary(answer_meta)
            elif body is not None:
                usage = body["usage"]
            else:
                usage = {}
            self._observe(
                route=route,
                finish=finish,
                ttft_ms=ttft_ms,
                latency_ms=_elapsed_ms(started, self._clock()),
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens"),
                prefill_ms=prefill_ms,
                prefill_tokens=prompt_tokens,
                stream=True,
            )

    # -- SSE ------------------------------------------------------------------

    def _open_sse(self, request_id: str, *, prefill_tokens: int = 0) -> None:
        """Заголовки SSE. HTTP/1.1 — chunked (keep-alive), HTTP/1.0 — до закрытия.

        ``X-Prefill-Tokens`` — сколько токенов промпта посчитано: по этому
        заголовку видно, что prefill доминирует в TTFT (задача 0115).
        """
        self._sse_chunked = self.request_version != "HTTP/1.0"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Request-Id", request_id)
        self.send_header("X-Prefill-Tokens", str(max(0, int(prefill_tokens))))
        # Прокси не должны копить поток в буфере.
        self.send_header("X-Accel-Buffering", "no")
        if self._sse_chunked:
            self.send_header("Connection", "keep-alive")
            self.send_header("Transfer-Encoding", "chunked")
        else:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()

    def _write_sse(self, payload: bytes) -> bool:
        """Пишет одно событие ``data: ...\\n\\n``; ``False`` — клиент ушёл."""
        event = b"data: " + payload + b"\n\n"
        try:
            if self._sse_chunked:
                self.wfile.write(b"%x\r\n" % len(event) + event + b"\r\n")
            else:
                self.wfile.write(event)
            self.wfile.flush()
            return True
        except _CLIENT_GONE:
            logger.debug("клиент закрыл соединение во время стрима")
            self.close_connection = True
            return False

    def _send_sse_data(self, payload: Mapping[str, Any]) -> bool:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return self._write_sse(body.encode("utf-8"))

    def _send_sse_usage(
        self, chunk_id: str, created: int, model: str, usage: Mapping[str, Any]
    ) -> bool:
        """Отдельный чанк с ``usage`` (``stream_options.include_usage``)."""
        return self._send_sse_data(
            {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [],
                "usage": dict(usage),
            }
        )

    def _send_sse_error(
        self,
        request_id: str,
        *,
        code: str = "stream_error",
        message: str | None = None,
    ) -> bool:
        """Ошибка после первого чанка: JSON уже не отправить, шлём событие."""
        return self._send_sse_data(
            {
                "error": {
                    "message": message
                    or (
                        "Внутренняя ошибка сервера. Повторите запрос позже; "
                        "если не поможет, сообщите код {}.".format(request_id)
                    ),
                    "type": "server_error",
                    "code": code,
                    "param": None,
                }
            }
        )

    def _send_sse_done(self) -> bool:
        """Терминальное событие ``data: [DONE]`` и конец chunked-тела.

        ``False`` — клиент ушёл до ``[DONE]`` (запрос считается отменённым).
        """
        if not self._write_sse(b"[DONE]"):
            return False
        if self._sse_chunked:
            try:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except _CLIENT_GONE:
                logger.debug("клиент закрыл соединение до конца стрима")
                self.close_connection = True
                return False
        return True

    # -- разбор запроса -------------------------------------------------------

    def _read_json_body(self) -> Any:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise _ClientError(
                "Не указан заголовок Content-Length: сервер не знает, сколько "
                "читать.",
                status=411,
                code="length_required",
            )
        try:
            length = int(raw_length)
        except (TypeError, ValueError):
            raise _ClientError(
                "Некорректный заголовок Content-Length: ожидалось целое число.",
                code="invalid_content_length",
            ) from None

        if length <= 0:
            raise _ClientError("Пустое тело запроса.", code="empty_body")
        if length > MAX_BODY_BYTES:
            raise _ClientError(
                "Тело запроса больше {} байт — такой чат не поддерживается.".format(
                    MAX_BODY_BYTES
                ),
                status=413,
                code="payload_too_large",
            )

        body = self.rfile.read(length)
        if not body:
            raise _ClientError("Пустое тело запроса.", code="empty_body")

        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError:
            raise _ClientError(
                "Тело запроса должно быть в кодировке UTF-8.",
                code="invalid_encoding",
            ) from None

        if not text.strip():
            raise _ClientError("Пустое тело запроса.", code="empty_body")

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise _ClientError(
                "Некорректный JSON: {} (строка {}, позиция {}).".format(
                    exc.msg, exc.lineno, exc.colno
                ),
                code="invalid_json",
            ) from None

    def _parse_chat(
        self, payload: Any
    ) -> tuple[list[tuple[str, str]], dict, str, bool, bool]:
        """Разбор тела чата -> (pairs, meta, model, stream, include_usage)."""
        if not isinstance(payload, Mapping):
            raise _ClientError(
                "Тело запроса должно быть JSON-объектом.", code="invalid_body"
            )

        stream_raw = payload.get("stream", False)
        if not isinstance(stream_raw, bool):
            raise _ClientError(
                "Поле stream должно быть true или false.",
                code="invalid_stream",
            )
        stream = stream_raw

        include_usage = False
        stream_options = payload.get("stream_options")
        if stream_options is not None:
            if not isinstance(stream_options, Mapping):
                raise _ClientError(
                    "Поле stream_options должно быть объектом.",
                    code="invalid_stream_options",
                )
            include_usage = stream_options.get("include_usage") is True

        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise _ClientError(
                "Поле messages обязательно и должно быть непустым списком.",
                code="invalid_messages",
            )

        pairs: list[tuple[str, str]] = []
        for index, item in enumerate(messages):
            if not isinstance(item, Mapping):
                raise _ClientError(
                    "messages[{}] должен быть объектом с полями role и content.".format(
                        index
                    ),
                    code="invalid_messages",
                )
            role = item.get("role")
            if not isinstance(role, str) or not role.strip():
                raise _ClientError(
                    "messages[{}].role должен быть непустой строкой.".format(index),
                    code="invalid_messages",
                )
            content = _content_to_text(item.get("content"))
            if content is None:
                raise _ClientError(
                    "messages[{}].content должен быть строкой или списком частей "
                    "с полем text.".format(index),
                    code="invalid_messages",
                )
            pairs.append((role.strip(), content))

        meta: dict[str, Any] = {"channel": "http"}
        user = payload.get("user")
        if isinstance(user, str) and user.strip():
            meta["user_id"] = user.strip()

        requested_model = payload.get("model")
        model = (
            requested_model.strip()
            if isinstance(requested_model, str) and requested_model.strip()
            else DEFAULT_MODEL
        )
        return pairs, meta, model, stream, include_usage

    # -- ответы ---------------------------------------------------------------

    def _path(self) -> str:
        return urlsplit(self.path).path or "/"

    def _allowed_methods(self, path: str) -> str | None:
        if path in ("/health", "/v1/models", "/metrics"):
            return "GET"
        if path == "/v1/chat/completions":
            return "POST"
        return None

    def _handle_unsupported_method(self) -> None:
        path = self._path()
        allow = self._allowed_methods(path)
        if allow is None:
            self._send_not_found(path)
        else:
            self._send_method_not_allowed(allow)

    def _send_not_found(self, path: str) -> None:
        self._send_error(
            404,
            "Эндпоинт {} не найден. Доступны: GET /health, GET /metrics, "
            "GET /v1/models, POST /v1/chat/completions.".format(path),
            "not_found",
        )

    def _send_method_not_allowed(self, allow: str) -> None:
        self._send_error(
            405,
            "Метод {} для {} не разрешён. Разрешено: {}.".format(
                self.command, self._path(), allow
            ),
            "method_not_allowed",
            headers={"Allow": allow},
        )

    def _send_json(
        self,
        status: int,
        payload: Any,
        *,
        headers: Mapping[str, str] | None = None,
        close: bool = False,
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        if close:
            # Тело могли не дочитать — keep-alive после ошибки небезопасен.
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        try:
            self.wfile.write(body)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("клиент закрыл соединение до отправки ответа")

    def _send_no_content(self, *, headers: Mapping[str, str] | None = None) -> None:
        """204: у ответа нет тела, поэтому нет и Content-Length (RFC 9110)."""
        self.send_response(204)
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()

    def _send_error(
        self,
        status: int,
        message: str,
        code: str,
        *,
        error_type: str = "invalid_request_error",
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self._send_json(
            status,
            {
                "error": {
                    "message": message,
                    "type": error_type,
                    "code": code,
                    "param": None,
                }
            },
            headers=headers,
            close=True,
        )

    def _send_internal_error(self, request_id: str) -> None:
        """500 с идентификатором запроса; трейсбек остаётся в логе."""
        self._send_error(
            500,
            "Внутренняя ошибка сервера. Повторите запрос позже; "
            "если не поможет, сообщите код {}.".format(request_id),
            "internal_error",
            error_type="server_error",
        )

    def _send_timeout(self, request_id: str) -> None:
        """504: генерация не уложилась в ``EVA_REQUEST_TIMEOUT``."""
        timeout = self._request_timeout()
        self._send_error(
            504,
            "Генерация не уложилась в {} с и была отменена. Сократите запрос "
            "или увеличьте EVA_REQUEST_TIMEOUT на сервере. Код запроса: "
            "{}.".format(_format_seconds(timeout), request_id),
            "request_timeout",
            error_type="server_error",
            headers={"Retry-After": "1"},
        )

    def _send_unavailable(self, request_id: str, exc: ServiceUnavailable) -> None:
        """503: большая модель, локальная модель и скилл не дали ответа."""
        message = exc.public_message
        if request_id:
            message += " Код запроса: {}.".format(request_id)
        self._send_error(
            503,
            message,
            "service_unavailable",
            error_type="server_error",
            headers={"Retry-After": "5"},
        )

    # -- журнал ---------------------------------------------------------------

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        logger.info("%s %s", self.address_string(), format % args)

    def log_error(self, format: str, *args: Any) -> None:  # noqa: A002
        logger.warning("%s %s", self.address_string(), format % args)


def make_handler(router: Any) -> type[EvaRequestHandler]:
    """Подкласс обработчика с вшитым маршрутизатором (без глобального состояния).

    ``router`` передаётся как есть: из него берётся и ``answer`` (через
    :func:`as_router`), и необязательный ``stream`` (через
    :func:`as_stream_router`). ``router=None`` — боевая связка: ``answer`` и
    ``stream`` берутся из ``eva.router`` раздельно, потому что это две
    функции модуля, а не методы одного объекта.
    """
    if router is None:
        router_impl: Any = default_router
        stream_impl: Any = default_stream_router
    else:
        router_impl = as_router(router)
        stream_impl = as_stream_router(router)

    class BoundHandler(EvaRequestHandler):
        _router = staticmethod(router_impl)

    BoundHandler._stream_router = (
        staticmethod(stream_impl) if stream_impl is not None else None
    )
    BoundHandler.__name__ = "EvaRequestHandler[{}]".format(
        getattr(router_impl, "__name__", type(router_impl).__name__)
    )
    return BoundHandler


class EvaHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer с отметкой старта для ``/health`` и метриками.

    ``clock`` — источник времени для латентности и TTFT (по умолчанию
    ``time.monotonic``); ``metrics`` — накопитель ``eva.telemetry.Metrics``
    (по умолчанию новый). Оба параметра инжектируются в тестах: фейковые часы
    дают детерминированный TTFT, а свой накопитель — изолированные счётчики.

    ``local_model`` — локальный клиент ВТМ для прогрева (``None`` — взять
    ``EVA_VTM_MODEL``); ``warmup`` — ``False`` выключает прогрев, ``None`` —
    прогреть, если модель задана. Результат лежит в ``self.warmup`` (``None``,
    если прогрева не было), а ошибка прогрева старт не отменяет.
    """

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        handler: type[EvaRequestHandler],
        *,
        router_name: str = "injected",
        access_guard: AccessGuard | None = None,
        metrics: Any = None,
        clock: Callable[[], float] | None = None,
        max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT_S,
        degradation: DegradationChain | None = None,
        local_model: Any = None,
        warmup: bool | None = None,
    ) -> None:
        self.router_name = router_name
        self.access_guard = access_guard
        self.clock = clock if callable(clock) else time.monotonic
        self.metrics = metrics if metrics is not None else Metrics()
        self.max_prompt_tokens = max(0, int(max_prompt_tokens))
        self.request_timeout = max(0.0, float(request_timeout))
        self.degradation = degradation
        self.local_model = local_model
        self.warmup: dict[str, Any] | None = None
        self.counters = ServerCounters()
        self.started_at = self.clock()
        super().__init__(address, handler)
        if warmup is not False:
            # Прогрев идёт после bind: даже зависшая/упавшая модель не мешает
            # сокету и не отменяет старт — ошибка останется предупреждением.
            self.warmup = warmup_local_model(local_model, metrics=self.metrics)


def _auth_settings() -> tuple[str, int]:
    """(токен, лимит) из ``eva.config`` — только эти две настройки."""
    from . import config as config_module

    return config_module.auth_settings()


def _limit_settings() -> tuple[int, float]:
    """(лимит промпта, таймаут запроса) из окружения.

    Некорректное число — ``ValueError`` до открытия порта: лучше не подняться
    вовсе, чем молча работать без обещанного лимита. ``0`` выключает
    соответствующий предел.
    """
    prompt = _env_number("EVA_MAX_PROMPT_TOKENS", DEFAULT_MAX_PROMPT_TOKENS, int)
    timeout = _env_number("EVA_REQUEST_TIMEOUT", DEFAULT_REQUEST_TIMEOUT_S, float)
    return max(0, int(prompt)), max(0.0, float(timeout))


def _env_number(name: str, default: float, cast: Callable[[float], Any]) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return cast(float(raw.strip().replace(",", ".")))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "{}: ожидалось число, получено {!r}".format(name, raw)
        ) from exc


def create_server(
    host: str = "127.0.0.1",
    port: int = 0,
    router: Any = None,
    *,
    auth_token: str | None = None,
    rate_limit_per_min: int | None = None,
    metrics: Any = None,
    clock: Callable[[], float] | None = None,
    max_prompt_tokens: int | None = None,
    request_timeout: float | None = None,
    degradation: DegradationChain | None = None,
    local_model: Any = None,
    warmup: bool | None = None,
) -> EvaHTTPServer:
    """Создаёт (но не запускает) сервер; ``port=0`` — свободный порт.

    ``router`` — функция ``router(request) -> Answer`` или объект с методом
    ``answer(request)``. Если у него есть ещё и ``stream(request)``,
    ``stream=true`` отдаёт ответ по частям. Без ``router`` используется боевая
    связка :func:`default_router`, которая собирается лениво.

    Доступ: ``auth_token`` и ``rate_limit_per_min`` по умолчанию берутся из
    окружения (``EVA_AUTH_TOKEN``, ``EVA_RATE_LIMIT_PER_MIN``). ``None`` —
    «взять из окружения», пустая строка и ``0`` — явно выключить проверку
    токена и лимит соответственно. Заданный токен включает требование
    ``Authorization: Bearer`` на всех путях, кроме ``GET /health``.

    ``metrics`` — накопитель ``eva.telemetry.Metrics`` (по умолчанию новый);
    ``clock`` — часы для латентности и TTFT (по умолчанию ``time.monotonic``).
    Оба нужны тестам: подмена часов делает замеры детерминированными.

    ``max_prompt_tokens`` и ``request_timeout`` по умолчанию берутся из
    окружения (``EVA_MAX_PROMPT_TOKENS``, ``EVA_REQUEST_TIMEOUT``); ``0``
    выключает соответствующий предел.

    ``degradation`` — :class:`DegradationChain` (big → локальная модель →
    скилл → 503). Если ``router`` не задан, цепочка становится основным
    маршрутизатором; если задан — она используется как запасная после падения
    основного (стадия ``big`` уже пробована и не повторяется).

    ``local_model`` и ``warmup`` — прогрев при старте (задача 0115). Модель для
    прогрева берётся из ``local_model`` или ``EVA_VTM_MODEL``; прогрев грузит
    её и считает короткий prefill, а результат (``warmup_ms``, успех, причина
    отказа) попадает в ``GET /metrics``. Ошибка прогрева не мешает серверу
    подняться: в журнал уходит предупреждение, сервер принимает запросы.
    ``warmup=False`` выключает прогрев совсем.
    """
    env_token, env_limit = _auth_settings()
    token = env_token if auth_token is None else auth_token.strip()
    limit = (
        env_limit
        if rate_limit_per_min is None
        else max(0, int(rate_limit_per_min))
    )
    env_prompt, env_timeout = _limit_settings()
    prompt_limit = (
        env_prompt if max_prompt_tokens is None else max(0, int(max_prompt_tokens))
    )
    timeout = (
        env_timeout if request_timeout is None else max(0.0, float(request_timeout))
    )
    if degradation is None and isinstance(router, DegradationChain):
        degradation = router
    if degradation is not None and router is None:
        router = degradation
    if isinstance(router, DegradationChain):
        router_name = "degradation"
    elif router is None:
        router_name = "default"
    else:
        router_name = "injected"

    server = EvaHTTPServer(
        (host, port),
        make_handler(router),
        router_name=router_name,
        access_guard=AccessGuard(token=token, rate_limit_per_min=limit),
        metrics=metrics,
        clock=clock,
        max_prompt_tokens=prompt_limit,
        request_timeout=timeout,
        degradation=degradation,
        local_model=local_model,
        warmup=warmup,
    )
    bound_host, bound_port = server.server_address[:2]
    logger.info("eva-server слушает http://%s:%s", bound_host, bound_port)
    if token:
        logger.info(
            "аутентификация включена (Authorization: Bearer), лимит %s "
            "запросов в минуту на ключ",
            limit,
        )
    else:
        logger.warning(
            "EVA_AUTH_TOKEN не задан: сервер работает без аутентификации. "
            "Кто дотянется до %s:%s, тот и тратит бюджет на большие модели — "
            "наружу не выставлять. Лимит: %s запросов в минуту на адрес клиента.",
            bound_host,
            bound_port,
            limit,
        )
    return server
