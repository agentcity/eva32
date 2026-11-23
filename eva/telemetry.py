"""Телеметрия гибридного ассистента: один ответ — одна строка JSONL.

Формат записи (одна строка, UTF-8, без завершающей запятой)::

    {
      "ts": "2026-11-18T04:40:00.123Z",   # ISO 8601, UTC
      "route": "skill" | "small" | "big" | "unknown",
      "skill": "pricing" | null,
      "prompt_tokens": 12,
      "completion_tokens": 34,
      "total_tokens": 46,
      "cost_rub": 0.0021,
      "latency_ms": 420.0,
      "reason": "короткий вопрос",       # необязательно
      "confidence": 0.82,                # необязательно
      "local": true,                     # ответила своя модель (0109)
      "source": "local",                 # skill | local | api
      "prefill_ms": 180.0,               # граница prefill, если известна (0120)
      "user_id": "u-1", "channel": "web" # прочие поля сохраняются
    }

``local`` — маркер **своей** модели: ``true`` — ответила локальная ВТМ,
``false`` — HTTP/API. У маршрута ``skill`` маркера нет (не применимо), у
``big`` он тоже ``None``: там ``source`` = ``api``. ``source`` — грубая
категория источника: ``skill`` (детерминированный ответ), ``local`` (своя
модель), ``api`` (внешний провайдер). ``prefill_ms`` пишется, только если
маршрутизатор сообщил границу prefill; иначе поля нет — сервер её не угадывает.

Прочие поля (``user_id``, ``channel`` и т. п.) не выбрасываются: по ним видно
канал и пользователя. Секреты в запись не передавайте — значения попадают в
файл как есть.

Читающая сторона снисходительна: отсутствующие числа считаются нулём,
отсутствующий маршрут — ``"unknown"``, а строка, которую не удалось разобрать,
пропускается со счётчиком. Так отчёт переживает записи другой версии кода и
оборванный хвост файла.

Перцентили считаются линейной интерполяцией по позиции ``(n - 1) * q / 100``
(метод R-7, он же по умолчанию в numpy): для ``[1, 2, 3, 4]`` p50 = 2.5,
для латентностей 1..100 мс p95 = 95.05 мс.

Кроме лога в файле живёт :class:`Metrics` — счётчики работающего сервера:
TTFT, полное время, токены, код завершения (``ok``/``stop``/``cancel``/``error``)
и маршрут (``skill``/``small``/``big``). Их отдаёт ``GET /metrics``. Это другой
источник: файл — бухгалтерия по завершённым ответам, ``Metrics`` — состояние
процесса «здесь и сейчас», включая отменённые и упавшие запросы.

Prefill и прогрев (задача 0115). Первый запрос к локальной ВТМ платит за
загрузку модели и за prefill — счёт промпта по токену. Поэтому у запроса
TTFT разделён на две части, когда маршрутизатор сообщает границу prefill:

* ``prefill_ms`` — сколько заняла prefill-часть (нет данных — ``None``);
* ``ttft_after_prefill_ms`` — остаток TTFT после prefill (``ttft_ms - prefill_ms``,
  только если обе цифры известны и prefill не больше TTFT);
* ``ttft_ms`` — по-прежнему полное время до первого отданного чанка;
* ``prefill_tokens`` — накопительный счётчик посчитанных токенов промпта.

Отдельно :meth:`Metrics.record_warmup` пишет прогрев при старте: последнее
время прогрева (``warmup_ms``), его успех и причину отказа. Прогрев не
считается запросом: ``requests``, ``latency_ms`` и ``ttft_ms`` он не трогает.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
from collections import Counter, deque
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

__all__ = [
    "DEFAULT_MAX_SAMPLES",
    "DEFAULT_PATH",
    "FINISH_CODES",
    "Metrics",
    "ROUTE_KINDS",
    "SOURCE_KINDS",
    "iter_records",
    "log",
    "normalize_record",
    "percentile",
    "summary",
    "usage_report",
]

DEFAULT_PATH = "logs/eva.jsonl"

_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")
_NUMBER_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens", "cost_rub", "latency_ms")
_CONSUMED = {
    "ts",
    "route",
    "skill",
    "usage",
    *_NUMBER_FIELDS,
    "reason",
    "confidence",
    "local",
    "source",
    "prefill_ms",
}

#: Категории источника ответа (задача 0109): скилл, своя модель, API.
SOURCE_KINDS = ("skill", "local", "api")


def percentile(values: list[float] | tuple[float, ...], q: float) -> float | None:
    """Перцентиль ``q`` (0..100) линейной интерполяцией; ``None`` для пустого входа."""
    if not 0 <= q <= 100:
        raise ValueError("перцентиль должен быть в диапазоне 0..100")
    data = sorted(float(v) for v in values)
    if not data:
        return None
    if len(data) == 1:
        return data[0]
    pos = (len(data) - 1) * (q / 100.0)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return data[int(pos)]
    return data[lo] + (data[hi] - data[lo]) * (pos - lo)


def _to_float(value: Any) -> float | None:
    """Число из входа; ``None``, если значения нет или оно не конечное."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip().replace(",", "."))
        except ValueError:
            return None
    else:
        return None
    if not math.isfinite(number):
        return None
    return number


def _to_int(value: Any) -> int | None:
    number = _to_float(value)
    return None if number is None else int(round(number))


def _flag(value: Any) -> bool | None:
    """Булев маркер из значения; всё, что не ``bool``, — ``None``.

    Строки и 0/1 намеренно не разбираются: маркер ``local`` ставит
    маршрутизатор полем ``bool``, а «похожее на истину» из чужого кода
    считать своим ответом нельзя.
    """
    return value if isinstance(value, bool) else None


def _local_of(route: Any, usage: Any = None, explicit: Any = None) -> bool | None:
    """Маркер «ответила своя модель» из записи, маршрута или ``usage``."""
    marker = _flag(explicit)
    if marker is not None:
        return marker
    if route is not None and not isinstance(route, str):
        if isinstance(route, Mapping):
            marker = _flag(route.get("local"))
        else:
            marker = _flag(getattr(route, "local", None))
        if marker is not None:
            return marker
    if isinstance(usage, Mapping):
        return _flag(usage.get("local"))
    return None


def _source_of(route: str, local: bool | None, explicit: Any = None) -> str | None:
    """``skill`` | ``local`` | ``api`` по маршруту и маркеру ``local``."""
    if isinstance(explicit, str) and explicit.strip().lower() in SOURCE_KINDS:
        return explicit.strip().lower()
    if route == "skill":
        return "skill"
    if local is True:
        return "local"
    if route == "big":
        return "api"
    if route == "small" and local is False:
        return "api"
    return None


def _prefill_of(data: Mapping[str, Any]) -> float | None:
    """Граница prefill из записи: поле или вложенные ``timings``/``usage``."""
    raw = data.get("prefill_ms")
    if raw is None:
        for name in ("timings", "usage"):
            nested = data.get(name)
            if isinstance(nested, Mapping):
                raw = nested.get("prefill_ms")
                if raw is not None:
                    break
    number = _to_float(raw)
    if number is None or number < 0:
        return None
    return round(number, 3)


def _route_info(value: Any) -> tuple[str, str | None, float | None]:
    """Маршрут, причина и уверенность из строки, dict или объекта ``Route``."""
    if value is None:
        return "unknown", None, None
    if isinstance(value, Mapping):
        kind = value.get("kind") or value.get("route")
        reason = value.get("reason")
        confidence = value.get("confidence")
    elif isinstance(value, str):
        return (value.strip().lower() or "unknown"), None, None
    else:
        kind = getattr(value, "kind", None) or getattr(value, "route", None)
        reason = getattr(value, "reason", None)
        confidence = getattr(value, "confidence", None)
    kind_text = str(kind).strip().lower() if kind is not None else ""
    return (kind_text or "unknown",
            None if reason is None else str(reason),
            _to_float(confidence))


def _skill_name(value: Any) -> str | None:
    """Имя скилла из строки, dict или объекта ``Skill``; пустое — ``None``."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        value = value.get("name") or value.get("skill")
    elif not isinstance(value, str):
        value = getattr(value, "name", None)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _iso_utc(value: Any, default: str | None) -> str | None:
    """Время в ISO 8601 UTC; непонятное значение заменяется ``default``."""
    if value is None:
        return default
    if isinstance(value, datetime):
        moment = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return moment.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace(
            "+00:00", "Z")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            moment = datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return default
        return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    if isinstance(value, str):
        text = value.strip()
        return text or default
    return default


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _day_key(ts: Any) -> str:
    if isinstance(ts, str) and _TS_RE.match(ts):
        return ts[:10]
    return "unknown"


def _as_mapping(record: Any) -> Mapping[str, Any]:
    """Запись из dict или объекта ``Answer`` (duck typing, без импорта eva.types)."""
    if hasattr(record, "to_record") and callable(record.to_record):
        record = record.to_record()
    if isinstance(record, Mapping):
        return record
    fields = (
        "ts", "route", "skill", "usage", "cost_rub", "latency_ms",
        "prompt_tokens", "completion_tokens", "total_tokens",
        "reason", "confidence", "local", "source", "prefill_ms",
        "user_id", "channel",
    )
    data = {}
    for name in fields:
        value = getattr(record, name, None)
        if value is not None:
            data[name] = value
    if not data:
        raise TypeError(
            "запись телеметрии должна быть Mapping или объектом с полями "
            "route/usage/cost_rub/latency_ms")
    return data


def _json_safe(value: Any) -> Any:
    """Заменить NaN/бесконечности в прочих полях на ``null``.

    Телеметрия не имеет права уронить обслуживание запроса: строгий JSON не
    знает NaN, а необязательное поле из чужого кода может его содержать.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def normalize_record(record: Any, *, default_ts: str | None = None) -> dict[str, Any]:
    """Привести запись к формату лога, не меняя входной объект.

    ``default_ts`` подставляется, когда времени нет: ``log()`` передаёт текущее,
    а разбор лога — ``None`` (тогда день попадает в ``"unknown"`` и отчёт
    остаётся детерминированным).
    """
    source = _as_mapping(record)
    data = dict(source)

    usage = data.get("usage")
    if isinstance(usage, Mapping):
        for key in ("prompt_tokens", "completion_tokens", "total_tokens", "cost_rub"):
            if data.get(key) is None and usage.get(key) is not None:
                data[key] = usage[key]

    route, reason, confidence = _route_info(data.get("route"))
    if reason is None and data.get("reason") is not None:
        reason = str(data["reason"])
    if confidence is None:
        confidence = _to_float(data.get("confidence"))

    local = _local_of(data.get("route"), usage, data.get("local"))
    source_key = _source_of(route, local, data.get("source"))
    prefill_ms = _prefill_of(data)

    prompt_tokens = _to_int(data.get("prompt_tokens")) or 0
    completion_tokens = _to_int(data.get("completion_tokens")) or 0
    total_tokens = _to_int(data.get("total_tokens"))
    if total_tokens is None:
        total_tokens = prompt_tokens + completion_tokens

    cost_rub = _to_float(data.get("cost_rub")) or 0.0
    latency_ms = _to_float(data.get("latency_ms")) or 0.0

    out: dict[str, Any] = {
        "ts": _iso_utc(data.get("ts"), default_ts),
        "route": route,
        "skill": _skill_name(data.get("skill")),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        # Отрицательные цена и латентность — заведомо ошибка: не даём им
        # портить суммы и перцентили, приводим к нулю.
        "cost_rub": max(cost_rub, 0.0),
        "latency_ms": max(latency_ms, 0.0),
    }
    if reason is not None:
        out["reason"] = reason
    if confidence is not None:
        out["confidence"] = confidence
    # Маркер локальной модели: у скилла и big его нет — «не применимо».
    if local is not None:
        out["local"] = local
    if source_key is not None:
        out["source"] = source_key
    if prefill_ms is not None:
        out["prefill_ms"] = prefill_ms

    for key, value in source.items():
        if key not in _CONSUMED and value is not None:
            out[key] = _json_safe(value)
    return out


def _json_line(record: dict[str, Any]) -> bytes:
    line = json.dumps(
        record,
        ensure_ascii=False,      # лог читает человек: кириллица без \uXXXX
        allow_nan=False,         # NaN в JSONL ломает любой парсер
        separators=(",", ":"),
        default=str,             # datetime/UUID в прочих полях не роняют запись
    )
    return (line + "\n").encode("utf-8")


def log(record: Any, path: str | os.PathLike[str] = DEFAULT_PATH, *,
        mkdirs: bool = True, sync: bool = False) -> dict[str, Any]:
    """Дописать запись в JSONL и вернуть её нормализованный вид.

    Строка пишется одним ``os.write`` в дескриптор с ``O_APPEND``: два процесса
    не перемешают половины строк, а читатель видит либо всю строку, либо ничего.
    ``fsync`` по умолчанию не делаем — для учёта стоимости достаточно видимости
    в файле; ``sync=True`` добавляет сброс на диск для бухгалтерских выгрузок.
    ``mkdirs`` создаёт только родительские каталоги указанного пути.
    """
    target = os.fspath(path)
    if not target:
        raise ValueError("путь к логу не может быть пустым")
    normalized = normalize_record(record, default_ts=_now_iso())
    data = _json_line(normalized)

    if mkdirs:
        parent = os.path.dirname(os.path.abspath(target))
        if parent:
            os.makedirs(parent, exist_ok=True)

    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            if written <= 0:  # pragma: no cover - защита от зависания в цикле
                raise OSError("не удалось записать строку телеметрии")
            view = view[written:]
        if sync:
            os.fsync(fd)
    finally:
        os.close(fd)
    return normalized


def iter_records(path: str | os.PathLike[str], *,
                 skipped: list[dict[str, Any]] | None = None):
    """Отдавать нормализованные записи; битые строки — в ``skipped``.

    Пустые строки (хвост файла, случайный перевод строки) не считаются ошибкой:
    это не запись. Ошибкой считается непустая строка, которая не является
    JSON-объектом; такая строка пропускается, а её номер и причина попадают
    в ``skipped``.
    """
    target = os.fspath(path)
    # utf-8-sig: лог, отредактированный на Windows, может начинаться с BOM.
    with open(target, "r", encoding="utf-8-sig", errors="replace") as handle:
        for lineno, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except ValueError as exc:
                if skipped is not None:
                    skipped.append({"line": lineno, "error": f"JSON: {exc}"})
                continue
            if not isinstance(item, dict):
                if skipped is not None:
                    skipped.append({"line": lineno, "error": "не JSON-объект"})
                continue
            yield normalize_record(item)


def _new_stats() -> dict[str, Any]:
    return {
        "requests": 0,
        "cost_rub": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "latency": [],
        "skills": Counter(),
        "local": 0,
    }


def _add(stats: dict[str, Any], record: Mapping[str, Any]) -> None:
    stats["requests"] += 1
    stats["cost_rub"] += float(record["cost_rub"])
    stats["prompt_tokens"] += int(record["prompt_tokens"])
    stats["completion_tokens"] += int(record["completion_tokens"])
    stats["total_tokens"] += int(record["total_tokens"])
    stats["latency"].append(float(record["latency_ms"]))
    skill = record.get("skill")
    if skill:
        stats["skills"][str(skill)] += 1
    if record.get("local") is True:
        stats["local"] += 1


def _finish(stats: dict[str, Any]) -> dict[str, Any]:
    requests = stats["requests"]
    latencies = stats["latency"]
    cost = stats["cost_rub"]
    local = stats["local"]
    return {
        "requests": requests,
        "cost_rub": cost,
        "cost_rub_avg": cost / requests if requests else 0.0,
        "cost_rub_per_1000": cost / requests * 1000.0 if requests else 0.0,
        "prompt_tokens": stats["prompt_tokens"],
        "completion_tokens": stats["completion_tokens"],
        "total_tokens": stats["total_tokens"],
        "latency_ms_sum": math.fsum(latencies),
        "latency_ms_avg": math.fsum(latencies) / requests if requests else 0.0,
        "latency_ms_p50": percentile(latencies, 50),
        "latency_ms_p95": percentile(latencies, 95),
        "skills": dict(stats["skills"].most_common()),
        # Доля ответов своей модели в агрегате лога (задача 0109).
        "local_requests": local,
        "local_share": local / requests if requests else 0.0,
    }


def summary(path: str | os.PathLike[str] = DEFAULT_PATH) -> dict[str, Any]:
    """Агрегат лога: итог, по маршрутам и по дням.

    Возвращает::

        {
          "path": str, "records": int, "skipped": int,
          "errors": [{"line": int, "error": str}, ...],  # первые 20
          "total": AGG, "routes": {route: AGG}, "days": {YYYY-MM-DD: AGG},
        }

    AGG: ``requests``, ``cost_rub`` (сумма), ``cost_rub_avg``,
    ``cost_rub_per_1000``, ``prompt_tokens``, ``completion_tokens``,
    ``total_tokens``, ``latency_ms_sum``, ``latency_ms_avg``,
    ``latency_ms_p50``, ``latency_ms_p95``, ``skills`` (счётчик по именам),
    ``local_requests`` и ``local_share`` (доля ответов своей модели, 0109).

    Битые строки не прерывают разбор: они попадают в ``skipped`` (число) и
    ``errors`` (первые 20 с номерами строк).
    """
    target = os.fspath(path)
    skipped_rows: list[dict[str, Any]] = []
    totals = _new_stats()
    routes: dict[str, dict[str, Any]] = {}
    days: dict[str, dict[str, Any]] = {}

    for record in iter_records(target, skipped=skipped_rows):
        _add(totals, record)
        _add(routes.setdefault(record["route"], _new_stats()), record)
        _add(days.setdefault(_day_key(record.get("ts")), _new_stats()), record)

    return {
        "path": target,
        "records": totals["requests"],
        "skipped": len(skipped_rows),
        "errors": skipped_rows[:20],
        "total": _finish(totals),
        "routes": {name: _finish(routes[name]) for name in sorted(routes)},
        "days": {name: _finish(days[name]) for name in sorted(days)},
    }


# Контракт docs/hybrid.md называет агрегат usage_report(); в задаче — summary().
# Одна реализация под двумя именами, чтобы не разъезжались цифры.
usage_report = summary


# --------------------------------------------------------------------------- #
# Счётчики работающего сервера (GET /metrics)
# --------------------------------------------------------------------------- #

#: Коды завершения запроса, которые различает сервер.
#:
#: ``ok`` — обычный (не потоковый) ответ отдан целиком; ``stop`` — поток
#: доехал до терминального чанка; ``cancel`` — клиент оборвал SSE, генерация
#: отменена; ``error`` — маршрутизатор или поток упал.
FINISH_CODES = ("ok", "stop", "cancel", "error")

#: Маршруты гибрида; всё, что не попало сюда, считается ``unknown``.
ROUTE_KINDS = ("skill", "small", "big")

#: Сколько последних замеров хранится для перцентилей.
#:
#: Счётчики (запросы, токены, коды завершения) накопительные и не ограничены;
#: перцентили считаются по «скользящему окну» последних замеров, чтобы
#: долгоживущий процесс не рос в памяти. Окно честно отражено в ``count``
#: соответствующего блока.
DEFAULT_MAX_SAMPLES = 4096

_ROUTE_KEYS = ROUTE_KINDS + ("unknown",)


def _route_key(value: Any) -> str:
    """Маршрут к одному из известных видов; прочее — ``unknown``."""
    text = "" if value is None else str(value).strip().lower()
    return text if text in _ROUTE_KEYS else "unknown"


def _finish_key(value: Any) -> str:
    """Код завершения к одному из :data:`FINISH_CODES`; прочее — ``error``."""
    text = "" if value is None else str(value).strip().lower()
    return text if text in FINISH_CODES else "error"


def _ms_or_none(value: Any) -> float | None:
    """Миллисекунды замера; ``None`` для отсутствующего или негодного значения."""
    number = _to_float(value)
    if number is None or number < 0:
        return None
    return number


def _count(value: Any) -> int:
    """Неотрицательное целое из значения; мусор — ноль."""
    number = _to_float(value)
    if number is None or number <= 0:
        return 0
    return int(round(number))


def _percentile_value(data: list[float], q: float) -> float | None:
    """Перцентиль, округлённый для JSON; ``None`` остаётся ``None``."""
    value = percentile(data, q)
    return None if value is None else round(value, 3)


def _stats(samples: Any) -> dict[str, Any]:
    """Перцентили p50/p95/p99 по замерам; пустой вход — ``None`` и ``count: 0``."""
    data = [float(value) for value in samples]
    if not data:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "avg": None,
            "p50": None,
            "p95": None,
            "p99": None,
        }
    return {
        "count": len(data),
        "min": round(min(data), 3),
        "max": round(max(data), 3),
        "avg": round(math.fsum(data) / len(data), 3),
        "p50": _percentile_value(data, 50),
        "p95": _percentile_value(data, 95),
        "p99": _percentile_value(data, 99),
    }


def _warmup_block(last: dict[str, Any] | None, count: int) -> dict[str, Any]:
    """Блок прогрева в ``/metrics``: последний замер и число прогревов."""
    block: dict[str, Any] = {
        "count": count,
        "warmup_ms": None,
        "ok": None,
        "prefill_tokens": 0,
        "error": None,
        "model": None,
    }
    if last is not None:
        block.update(last)
    return block


class Metrics:
    """Потокобезопасные счётчики запросов сервера и перцентили задержек.

    Один экземпляр живёт на сервер (``EvaHTTPServer.metrics``) и наполняется
    через :meth:`observe` — по одному вызову на запрос. ``snapshot()`` отдаёт
    готовый к JSON словарь для ``GET /metrics``::

        {
          "requests": 12, "streams": 9,
          "finish": {"ok": 2, "stop": 8, "cancel": 1, "error": 1},
          "cancelled": 1, "errors": 1,
          "routes": {"skill": 4, "small": 6, "big": 1, "unknown": 1},
          "tokens": {"prompt": 120, "completion": 340, "total": 460},
          "prefill_tokens": 120,
          "ttft_ms": {"count": 8, "min": ..., "max": ..., "avg": ...,
                      "p50": ..., "p95": ..., "p99": ...},
          "prefill_ms": {...},              # только по замерам, где есть данные
          "ttft_after_prefill_ms": {...},   # TTFT минус prefill
          "latency_ms": {...},
          "small_local": 4, "small_http": 2,   # своя модель vs HTTP (0109)
          "local_requests": 4,
          "local_share": 0.666...,          # доля своей модели среди small
          "local_share_all": 0.333...,      # доля своей модели среди запросов
          "source": {"skill": 1, "local": 4, "api": 3},
          "warmup_ms": 41.2, "warmup": {"count": 1, "ok": true, "warmup_ms": 41.2,
                                        "prefill_tokens": 3, "error": null,
                                        "model": "demo/model.vtm"},
          "by_route": {"small": {...}, ...},
        }

    ``ttft_ms`` — время до первого отданного чанка; у не потокового ответа
    первого чанка нет, поэтому такой запрос попадает в ``latency_ms``, но не
    в ``ttft_ms`` (``ttft_ms: null`` при вызове). Токены для оборванного
    потока часто неизвестны: маршрутизатор сообщает ``usage`` в финальном
    куске, которого при отмене нет, — такие запросы дают нули, а не выдумку.

    ``prefill_ms`` и ``ttft_after_prefill_ms`` заполняются только когда
    маршрутизатор передал ``prefill_ms``: без этой цифры сервер не угадывает
    границу prefill и оставляет ``None``. ``prefill_tokens`` сервер считает
    сам по длине промпта, поэтому счётчик есть у всех запросов.

    ``local`` (0109) приходит из ответа маршрутизатора: ``True`` — своя ВТМ,
    ``False`` — HTTP/API, ``None`` — неизвестно. Доля считается по размеченным
    ответам ``small``: ответы без маркера в знаменатель не попадают, чтобы не
    выдавать HTTP за свою модель.

    Замеры для перцентилей ограничены окном ``max_samples``; счётчики — нет.
    """

    def __init__(self, *, max_samples: int = DEFAULT_MAX_SAMPLES) -> None:
        self.max_samples = max(1, int(max_samples))
        self._lock = threading.Lock()
        self._requests = 0
        self._streams = 0
        self._tokens = {"prompt": 0, "completion": 0, "total": 0}
        self._prefill_tokens = 0
        self._warmup_count = 0
        self._warmup: dict[str, Any] | None = None
        # Ответы своей модели (задача 0109): по малой модели и по всем сразу.
        self._small_local = 0
        self._small_http = 0
        self._local_requests = 0
        self._source = {name: 0 for name in SOURCE_KINDS}
        self._finish = {code: 0 for code in FINISH_CODES}
        self._routes = {kind: 0 for kind in _ROUTE_KEYS}
        maxlen = self.max_samples
        self._ttft: deque[float] = deque(maxlen=maxlen)
        self._prefill: deque[float] = deque(maxlen=maxlen)
        self._ttft_after: deque[float] = deque(maxlen=maxlen)
        self._latency: deque[float] = deque(maxlen=maxlen)
        self._by_route: dict[str, dict[str, Any]] = {
            kind: {
                "requests": 0,
                "tokens": 0,
                "prefill_tokens": 0,
                "local": 0,
                "finish": {code: 0 for code in FINISH_CODES},
                "ttft": deque(maxlen=maxlen),
                "prefill": deque(maxlen=maxlen),
                "ttft_after": deque(maxlen=maxlen),
                "latency": deque(maxlen=maxlen),
            }
            for kind in _ROUTE_KEYS
        }

    # -- запись ---------------------------------------------------------------

    def observe(
        self,
        *,
        route: Any = "unknown",
        finish: Any = "error",
        ttft_ms: Any = None,
        latency_ms: Any = 0.0,
        prompt_tokens: Any = 0,
        completion_tokens: Any = 0,
        total_tokens: Any = None,
        stream: bool = False,
        prefill_ms: Any = None,
        prefill_tokens: Any = None,
        local: Any = None,
    ) -> dict[str, Any]:
        """Учесть один запрос и вернуть его запись (для тестов и логов).

        Значения не доверенные: маршрут и код завершения сводятся к известным
        наборам, отрицательные и нечисловые задержки/токены — к нулю (а TTFT
        без значения не попадает в перцентили вовсе).

        ``prefill_ms`` — prefill-часть TTFT, если маршрутизатор её сообщил;
        ``prefill_tokens`` — посчитанные токены промпта (по умолчанию совпадают
        с ``prompt_tokens``: без usage промпт всё равно посчитан). Отдельный
        параметр нужен потому, что при обрыве потока usage неизвестен, а
        длина промпта известна.

        ``local`` — ответила ли своя модель (0109): ``True``/``False``;
        ``None`` — неизвестно (маршрут не про малую модель). По нему считаются
        ``small_local``/``small_http``, доля локальных ответов и счётчики
        ``source`` (``skill``/``local``/``api``).
        """
        route_key = _route_key(route)
        finish_key = _finish_key(finish)
        ttft = _ms_or_none(ttft_ms)
        latency = _ms_or_none(latency_ms) or 0.0
        prompt = _count(prompt_tokens)
        completion = _count(completion_tokens)
        total = _count(total_tokens) if total_tokens is not None else 0
        if total <= 0:
            # Так же, как в normalize_record: без явной суммы берём p + c.
            total = prompt + completion
        prefill = _ms_or_none(prefill_ms)
        counted = prompt if prefill_tokens is None else _count(prefill_tokens)
        # Prefill больше TTFT — данные противоречат друг другу: остаток не
        # выдумываем, оставляем None (в перцентили такой замер не попадёт).
        ttft_after = (
            ttft - prefill
            if ttft is not None and prefill is not None and prefill <= ttft
            else None
        )
        local_flag = _flag(local)
        source = _source_of(route_key, local_flag)

        record = {
            "route": route_key,
            "finish": finish_key,
            "stream": bool(stream),
            "ttft_ms": ttft,
            "prefill_ms": None if prefill is None else round(prefill, 3),
            "ttft_after_prefill_ms": (
                None if ttft_after is None else round(ttft_after, 3)
            ),
            "latency_ms": latency,
            "prompt_tokens": prompt,
            "prefill_tokens": counted,
            "completion_tokens": completion,
            "total_tokens": total,
            "local": local_flag,
            "source": source,
        }

        with self._lock:
            self._requests += 1
            if stream:
                self._streams += 1
            self._finish[finish_key] += 1
            self._routes[route_key] += 1
            self._tokens["prompt"] += prompt
            self._tokens["completion"] += completion
            self._tokens["total"] += total
            self._prefill_tokens += counted
            if route_key == "small":
                if local_flag is True:
                    self._small_local += 1
                elif local_flag is False:
                    self._small_http += 1
            if local_flag is True:
                self._local_requests += 1
            if source is not None:
                self._source[source] += 1
            if ttft is not None:
                self._ttft.append(ttft)
            if prefill is not None:
                self._prefill.append(prefill)
            if ttft_after is not None:
                self._ttft_after.append(ttft_after)
            self._latency.append(latency)

            row = self._by_route[route_key]
            row["requests"] += 1
            row["tokens"] += total
            row["prefill_tokens"] += counted
            if local_flag is True:
                row["local"] += 1
            row["finish"][finish_key] += 1
            if ttft is not None:
                row["ttft"].append(ttft)
            if prefill is not None:
                row["prefill"].append(prefill)
            if ttft_after is not None:
                row["ttft_after"].append(ttft_after)
            row["latency"].append(latency)
        return record

    def record_warmup(
        self,
        ms: Any,
        *,
        ok: bool = True,
        prefill_tokens: Any = 0,
        error: Any = None,
        model: Any = None,
    ) -> dict[str, Any]:
        """Записать результат прогрева модели; запросом прогрев не считается.

        Хранится только последний прогрев (их и бывает один на процесс), но
        счётчик ``count`` накопительный: перезапуск прогрева в тестах не
        теряется. Ошибка прогрева — штатное значение (``ok: false`` плюс
        текст причины), а не исключение.
        """
        duration = _ms_or_none(ms)
        record = {
            "warmup_ms": None if duration is None else round(duration, 3),
            "ok": bool(ok),
            "prefill_tokens": _count(prefill_tokens),
            "error": None if ok else (str(error) if error else "прогрев не удался"),
            "model": None if model is None else str(model),
        }
        with self._lock:
            self._warmup_count += 1
            self._warmup = record
        return record

    # -- чтение ---------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Срез счётчиков и перцентилей; словарь сериализуется в JSON как есть.

        Блок локальной модели (задача 0109): ``small_local`` — ответы своей
        ВТМ на маршруте ``small``, ``small_http`` — HTTP-клиент на том же
        маршруте, ``local_share`` — доля своей модели среди размеченных
        ответов ``small`` (``None``, пока их нет), ``local_share_all`` — доля
        своей модели среди всех запросов. ``source`` — счётчики по категориям
        ``skill``/``local``/``api``.
        """
        with self._lock:
            requests = self._requests
            streams = self._streams
            finish = dict(self._finish)
            routes = dict(self._routes)
            tokens = dict(self._tokens)
            prefill_tokens = self._prefill_tokens
            small_local = self._small_local
            small_http = self._small_http
            local_requests = self._local_requests
            source = dict(self._source)
            ttft = _stats(self._ttft)
            prefill = _stats(self._prefill)
            ttft_after = _stats(self._ttft_after)
            latency = _stats(self._latency)
            warmup = _warmup_block(self._warmup, self._warmup_count)
            by_route = {
                name: {
                    "requests": row["requests"],
                    "tokens": row["tokens"],
                    "prefill_tokens": row["prefill_tokens"],
                    "local": row["local"],
                    "finish": dict(row["finish"]),
                    "ttft_ms": _stats(row["ttft"]),
                    "prefill_ms": _stats(row["prefill"]),
                    "ttft_after_prefill_ms": _stats(row["ttft_after"]),
                    "latency_ms": _stats(row["latency"]),
                }
                for name, row in self._by_route.items()
            }
        small_marked = small_local + small_http
        return {
            "requests": requests,
            "streams": streams,
            "finish": finish,
            "cancelled": finish["cancel"],
            "errors": finish["error"],
            "routes": routes,
            "tokens": tokens,
            "prefill_tokens": prefill_tokens,
            "ttft_ms": ttft,
            "prefill_ms": prefill,
            "ttft_after_prefill_ms": ttft_after,
            "latency_ms": latency,
            # Своя модель vs API (0109): счётчики и доли.
            "small_local": small_local,
            "small_http": small_http,
            "local_requests": local_requests,
            "local_share": small_local / small_marked if small_marked else None,
            "local_share_all": (
                local_requests / requests if requests else None
            ),
            "source": source,
            "warmup_ms": warmup["warmup_ms"],
            "warmup": warmup,
            "by_route": by_route,
        }

    def reset(self) -> None:
        """Обнулить счётчики (для тестов и перезапуска замера)."""
        with self._lock:
            self._requests = 0
            self._streams = 0
            self._prefill_tokens = 0
            self._warmup_count = 0
            self._warmup = None
            self._small_local = 0
            self._small_http = 0
            self._local_requests = 0
            for name in self._source:
                self._source[name] = 0
            for key in self._tokens:
                self._tokens[key] = 0
            for code in self._finish:
                self._finish[code] = 0
            for kind in self._routes:
                self._routes[kind] = 0
            self._ttft.clear()
            self._prefill.clear()
            self._ttft_after.clear()
            self._latency.clear()
            for row in self._by_route.values():
                row["requests"] = 0
                row["tokens"] = 0
                row["prefill_tokens"] = 0
                row["local"] = 0
                row["finish"] = {code: 0 for code in FINISH_CODES}
                row["ttft"].clear()
                row["prefill"].clear()
                row["ttft_after"].clear()
                row["latency"].clear()
