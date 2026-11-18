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
      "user_id": "u-1", "channel": "web" # прочие поля сохраняются
    }

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
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

__all__ = [
    "DEFAULT_PATH",
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
}


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
        "reason", "confidence", "user_id", "channel",
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


def _finish(stats: dict[str, Any]) -> dict[str, Any]:
    requests = stats["requests"]
    latencies = stats["latency"]
    cost = stats["cost_rub"]
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
    ``latency_ms_p50``, ``latency_ms_p95``, ``skills`` (счётчик по именам).

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
