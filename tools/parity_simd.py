#!/usr/bin/env python3
"""Паритет бэкендов VTM: скаляр против SIMD — расхождение и его цена для argmax.

Зачем
=====

`tools/parity_random.py` проверяет C против Python-эталона с порогом
`max|Δ| < 1e-5`. Пока `VTM_BACKEND=scalar`, это честно: расхождение ровно
0, логиты совпадают побитово. С `VTM_BACKEND=auto` (то есть с SIMD, когда он
собран) порог перестаёт проходить: NEON/AVX2-ядра переупорядочивают
сложения, и логиты расходятся на единицы 1e-5…1e-4. Вопрос не в пороге, а в
том, **меняется ли ответ модели** — то есть argmax.

Инструмент делает ровно это измерение:

1. собирает ту же случайную конфигурацию, что `parity_random.py`
   (`generate_config`/`build_model`), пишет контейнер один раз;
2. прогоняет C-пробу дважды на одном контейнере: `VTM_BACKEND=scalar`
   (baseline) и `VTM_BACKEND=auto` (кандидат, то есть SIMD);
3. для каждой конфигурации считает `max|Δ|` логитов, argmax обоих прогонов,
   отрыв top-1 от top-2 и потерю в top-5;
4. по 50 конфигурациям строит распределение `max|Δ|` (медиана, p95, максимум)
   и отвечает: сколько конфигураций меняют argmax и при каком отрыве
   top-1/top-2 это происходит.

Почему отрыв top-1/top-2 — главное число
========================================

Смена argmax при расхождении `d` возможна, только если лидер и второй
кандидат стоят ближе, чем примерно `2d`. Если отрыв (margin) в тысячи раз
больше `d`, смена невозможна арифметически, и расхождение `1e-4` — шум, а не
изменение ответа. Поэтому для каждой конфигурации печатается
`margin / max|Δ|`; смены argmax при большом отношении означали бы ошибку в
ядрах, а не «допуск».

Контроль
========

Тот же прогон проверяет и обратное: `scalar` против Python-эталона
(`tools/vtm_ref.py`) должен давать `max|Δ| = 0` и тот же argmax. Если это не
так, измерение расхождения бэкендов бессмысленно — сначала сломан скаляр.

Вклад операций
==============

`--per-op` дополнительно разбирает `src/vtm/bench_backend --check` (если
бинарник собран `make -C src/vtm bench-backend`): тот же контейнер не нужен,
сравниваются ядра на синтетических формах — TMV, MM, RMSN. В модели
`interp.c` MM пока считается скалярно (см. `src/vtm/vtm_backend.h`), поэтому
вклад в логиты дают TMV и RMSN; MM в отчёте — справочно.

Запуск
======

    python3 tools/parity_simd.py                     # 50 конфигураций, seed 0
    python3 tools/parity_simd.py --count 10 --seed 7
    python3 tools/parity_simd.py --json
    python3 tools/parity_simd.py --only 28           # воспроизвести худшую
    python3 tools/parity_simd.py --per-op            # + вклад операций
    python3 tools/parity_simd.py --baseline scalar --candidate simd

Коды возврата
=============

* 0 — argmax нигде не сменился (расхождение есть, ответы те же);
* 1 — argmax сменился хотя бы в одной конфигурации (это баг, а не допуск)
  либо были ошибки прогона;
* 2 — нет C-рантайма/библиотеки/компилятора (сначала `make -C src/vtm`).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
for _path in (str(ROOT), str(_HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from tools import parity_random  # noqa: E402
from tools import parity_real  # noqa: E402
from tools import vtm_ref  # noqa: E402

TOOL_NAME = "parity_simd"
DEFAULT_COUNT = parity_random.DEFAULT_COUNT
DEFAULT_SEED = parity_random.DEFAULT_SEED
DEFAULT_TOLERANCE = parity_random.DEFAULT_TOLERANCE
DEFAULT_WORKDIR = ROOT / "data" / "parity_simd"
DEFAULT_BASELINE = "scalar"
DEFAULT_CANDIDATE = "auto"
#: Бинарник проверки ядер: `make -C src/vtm bench-backend`.
BENCH_BIN = ROOT / "src" / "vtm" / "bench_backend"


class ParitySimdError(RuntimeError):
    """Ошибка окружения/подготовки: код возврата 2."""


# ---------------------------------------------------------------------------
# Статистика
# ---------------------------------------------------------------------------


def percentile(values: Sequence[float], q: float) -> float:
    """Перцентиль с линейной интерполяцией (как numpy по умолчанию).

    `q` в [0, 100]. Для 50 значений p95 — это 0.95 * (n-1) = 46.55, то есть
    между 47-м и 48-м по возрастанию; именно так считает numpy, и это не
    «наибольшее», а честная оценка хвоста.
    """
    data = sorted(float(value) for value in values)
    if not data:
        raise ValueError("percentile: пустой список")
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"percentile: q={q} вне 0..100")
    if len(data) == 1:
        return data[0]
    position = (len(data) - 1) * (float(q) / 100.0)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return data[low]
    weight = position - low
    return data[low] * (1.0 - weight) + data[high] * weight


def distribution(values: Sequence[float], tolerance: float) -> dict[str, Any]:
    """Распределение max|Δ|: n, min/median/p95/max/mean и счётчики хвостов."""
    data = [float(value) for value in values]
    if not data:
        return {"count": 0}
    return {
        "count": len(data),
        "min": min(data),
        "median": percentile(data, 50.0),
        "p95": percentile(data, 95.0),
        "max": max(data),
        "mean": math.fsum(data) / len(data),
        "bit_exact": sum(1 for value in data if value == 0.0),
        "outside_tolerance": sum(1 for value in data if value >= tolerance),
        "at_least_1e4": sum(1 for value in data if value >= 1e-4),
        "tolerance": float(tolerance),
    }


# ---------------------------------------------------------------------------
# Один прогон конфигурации
# ---------------------------------------------------------------------------


@dataclass
class SimdResult:
    """Расхождение бэкендов на одной конфигурации."""

    config: parity_random.RandomConfig
    backend: parity_random.BackendComparison | None = None
    python_max_abs_delta: float | None = None
    python_argmax: int | None = None
    python_argmax_match: bool = False
    top1_token: int | None = None
    top2_token: int | None = None
    top1_value: float | None = None
    top2_value: float | None = None
    delta_at_top2: float | None = None
    top5_overlap: int | None = None
    elapsed_ms: float = 0.0
    error: str | None = None

    @property
    def max_abs_delta(self) -> float | None:
        return None if self.backend is None else self.backend.max_abs_delta

    @property
    def argmax_match(self) -> bool:
        return bool(self.backend is not None and self.backend.argmax_match)

    @property
    def margin(self) -> float | None:
        return None if self.backend is None else self.backend.top1_margin

    @property
    def margin_to_delta(self) -> float | None:
        if self.backend is None:
            return None
        return self.backend.margin_to_delta

    @property
    def passed(self) -> bool:
        return bool(
            self.error is None
            and self.backend is not None
            and self.backend.argmax_match
            and self.python_argmax_match
            and self.python_max_abs_delta == 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        backend = None if self.backend is None else self.backend.to_dict()
        return {
            "config": self.config.to_dict(),
            "backend": backend,
            "max_abs_delta": None if backend is None else backend["max_abs_delta"],
            "argmax_baseline": (
                None if backend is None else backend["argmax_baseline"]
            ),
            "argmax_candidate": (
                None if backend is None else backend["argmax_candidate"]
            ),
            "argmax_match": self.argmax_match,
            "top1_margin": None if backend is None else backend["top1_margin"],
            "margin_to_delta": (
                None if backend is None else backend["margin_to_delta"]
            ),
            "top1_token": self.top1_token,
            "top2_token": self.top2_token,
            "top1_value": _json_float(self.top1_value),
            "top2_value": _json_float(self.top2_value),
            "delta_at_top2": _json_float(self.delta_at_top2),
            "top5_overlap": self.top5_overlap,
            "python_max_abs_delta": _json_float(self.python_max_abs_delta),
            "python_argmax": self.python_argmax,
            "python_argmax_match": self.python_argmax_match,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "error": self.error,
            "passed": self.passed,
        }

    def failure_reasons(self) -> list[str]:
        if self.error is not None:
            return [f"ошибка прогона: {self.error}"]
        reasons: list[str] = []
        if self.backend is not None and not self.backend.argmax_match:
            reasons.append(
                f"argmax сменился: {self.backend.baseline} "
                f"{self.backend.argmax_baseline} -> {self.backend.candidate} "
                f"{self.backend.argmax_candidate} (max|Δ| = "
                f"{_fmt(self.backend.max_abs_delta)}, отрыв top1-top2 = "
                f"{_fmt(self.backend.top1_margin)})"
            )
        if not self.python_argmax_match:
            reasons.append(
                f"argmax baseline ({self.backend.baseline if self.backend else '?'}) "
                f"{self.backend.argmax_baseline if self.backend else '?'} != "
                f"Python {self.python_argmax}"
            )
        if self.python_max_abs_delta:
            reasons.append(
                f"Python != baseline-бэкенд: max|Δ| = "
                f"{_fmt(self.python_max_abs_delta)} (ожидалось 0)"
            )
        return reasons


def run_config(
    config: parity_random.RandomConfig,
    *,
    workdir: Path,
    probe: Path,
    baseline: str,
    candidate: str,
    tolerance: float = DEFAULT_TOLERANCE,
    keep: bool = False,
) -> SimdResult:
    """Один контейнер, два бэкенда, Python-эталон — без CLI (он не нужен)."""
    started = time.perf_counter()
    result = SimdResult(config=config)
    container = workdir / f"simd_{config.index:03d}.vtm"
    try:
        vtm = parity_random.build_model(config)
        parity_random.write_container(vtm, container, config)

        reference = vtm_ref.execute(vtm, list(config.tokens), 0)
        if reference.logits is None:  # pragma: no cover - защита от графа
            raise ParitySimdError("Python-эталон не дал логитов")
        python_logits = [float(value) for value in reference.logits]

        base = parity_random.run_probe(
            probe, container, config.tokens, backend=baseline
        )
        cand = parity_random.run_probe(
            probe, container, config.tokens, backend=candidate
        )
        comparison = parity_random.compare_backend_logits(
            base.logits, cand.logits, baseline=baseline, candidate=candidate
        )
        result.backend = comparison

        # Контроль: baseline обязан совпасть с Python побитово.
        control = parity_random.compare_logits(
            python_logits, base.logits, tolerance
        )
        result.python_max_abs_delta = control.max_abs_delta
        result.python_argmax = control.argmax_reference
        result.python_argmax_match = control.argmax_match

        # Отрыв top-1/top-2 считаем по baseline-логитам: это «настоящий»
        # ответ, а кандидат от него лишь шумит на 1e-4.
        order = parity_real.top_indices(base.logits, 2)
        if len(order) == 2:
            result.top1_token, result.top2_token = order
            result.top1_value = float(base.logits[order[0]])
            result.top2_value = float(base.logits[order[1]])
            result.delta_at_top2 = abs(
                float(base.logits[order[1]]) - float(cand.logits[order[1]])
            )
        top5_base = parity_real.top_indices(base.logits, 5)
        top5_cand = parity_real.top_indices(cand.logits, 5)
        result.top5_overlap = len(set(top5_base) & set(top5_cand))
    except Exception as exc:  # noqa: BLE001 - отчёт должен пережить любую ошибку
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        result.elapsed_ms = (time.perf_counter() - started) * 1000.0
        if result.error is None and not keep:
            container.unlink(missing_ok=True)
    return result


# ---------------------------------------------------------------------------
# Вклад операций (bench_backend --check)
# ---------------------------------------------------------------------------

_DIFF_RE = re.compile(r"max\|diff\|\s*=\s*([0-9.eE+-]+)")


def parse_per_op_output(text: str) -> dict[str, dict[str, Any]]:
    """Разобрать `bench_backend --check`: max|diff| по семействам ядер.

    Строки вида ``  TMV 12x1000 блок 128 scale 1.00 max|diff|=1.907e-06  …``
    или ``  MM trit 8x16x8 блок 128 …`` / ``  RMSN n=1000 …``. `TMV direct`
    — сравнение скалярного direct-пути со scalar-lut, это не SIMD-вклад:
    такие строки отбрасываются.
    """
    groups: dict[str, dict[str, Any]] = {
        "tmv": {"max": None, "count": 0},
        "mm_trit": {"max": None, "count": 0},
        "mm_f32": {"max": None, "count": 0},
        "rmsn": {"max": None, "count": 0},
    }
    for raw in text.splitlines():
        line = raw.strip()
        match = _DIFF_RE.search(line)
        if not match or not line:
            continue
        try:
            value = float(match.group(1))
        except ValueError:
            continue
        if line.upper().startswith("TMV"):
            if " direct " in line:
                continue
            key = "tmv"
        elif line.upper().startswith("MM TRIT"):
            key = "mm_trit"
        elif line.upper().startswith("MM F32"):
            key = "mm_f32"
        elif line.upper().startswith("RMSN"):
            key = "rmsn"
        else:
            continue
        entry = groups[key]
        entry["count"] += 1
        entry["max"] = value if entry["max"] is None else max(entry["max"], value)
    return groups


def per_op_report() -> dict[str, Any]:
    """Прогнать `bench_backend --check` (если собран) и разобрать числа."""
    if not (BENCH_BIN.is_file() and os.access(BENCH_BIN, os.X_OK)):
        return {
            "available": False,
            "reason": (
                f"нет {BENCH_BIN.name}: собрать `make -C src/vtm bench-backend`"
            ),
        }
    # backend="auto": bench_backend берёт кандидата из VTM_BACKEND, а
    # run_command по умолчанию ставит scalar — тогда --check сравнил бы
    # скаляр сам с собой и показал нули.
    proc = parity_random.run_command(
        [str(BENCH_BIN), "--check"], quiet=True, backend="auto"
    )
    if proc.returncode != 0:
        return {
            "available": False,
            "reason": (
                f"bench_backend --check вернул {proc.returncode}: "
                f"{parity_random._tail(proc.stderr, 5)}"
            ),
        }
    return {"available": True, "groups": parse_per_op_output(proc.stdout)}


# ---------------------------------------------------------------------------
# Сводка
# ---------------------------------------------------------------------------


def summarize(
    *,
    seed: int,
    count: int,
    baseline: str,
    candidate: str,
    tolerance: float,
    results: Sequence[SimdResult],
    workdir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Собрать JSON-сводку: распределение Δ, argmax, отрывы, контроль Python."""
    ok = [item for item in results if item.error is None]
    errors = [item for item in results if item.error is not None]
    deltas = [item.max_abs_delta for item in ok if item.max_abs_delta is not None]
    switches = [item for item in ok if not item.argmax_match]

    # Отрыв top-1/top-2: самый узкий и самый узкий относительно Δ.
    with_margin = [item for item in ok if item.backend is not None]
    narrowest = min(
        with_margin, key=lambda item: item.backend.top1_margin, default=None
    )
    tightest = min(
        with_margin,
        key=lambda item: item.backend.margin_to_delta,
        default=None,
    )
    # «Могло перевернуться» — отрыв не больше 2*max|Δ| (граница, за которой
    # смена argmax арифметически невозможна).
    borderline = [
        item
        for item in with_margin
        if item.backend.top1_margin < 2.0 * item.backend.max_abs_delta
    ]
    worst = max(
        with_margin, key=lambda item: item.backend.max_abs_delta, default=None
    )
    python_mismatch = [item for item in ok if not item.python_argmax_match]
    python_max = max(
        (item.python_max_abs_delta or 0.0 for item in ok), default=0.0
    )
    failed = [item for item in results if not item.passed]

    delta_stats = distribution(deltas, tolerance)
    summary = {
        "tool": TOOL_NAME,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "count": int(count),
        "baseline": str(baseline),
        "candidate": str(candidate),
        "tolerance": float(tolerance),
        "workdir": None if workdir is None else str(workdir),
        "total": len(results),
        "errors": len(errors),
        "failed": len(failed),
        "delta": delta_stats,
        "argmax": {
            "switches": len(switches),
            "checked": len(with_margin),
            "switch_configs": [
                {
                    "index": item.config.index,
                    "seed": item.config.seed,
                    "config": item.config.to_dict(),
                    "argmax_baseline": item.backend.argmax_baseline,
                    "argmax_candidate": item.backend.argmax_candidate,
                    "max_abs_delta": _json_float(item.backend.max_abs_delta),
                    "top1_margin": _json_float(item.backend.top1_margin),
                    "margin_to_delta": _json_float(item.backend.margin_to_delta),
                }
                for item in switches
            ],
            "narrowest_margin": _margin_brief(narrowest),
            "tightest_ratio": _margin_brief(tightest),
            "borderline_below_2delta": len(borderline),
            "borderline": [_margin_brief(item) for item in borderline],
        },
        "python_vs_baseline": {
            "max_abs_delta": _json_float(python_max),
            "bit_exact": sum(
                1 for item in ok if (item.python_max_abs_delta or 0.0) == 0.0
            ),
            "argmax_mismatches": len(python_mismatch),
            "checked": len(ok),
        },
        "worst": _margin_brief(worst, with_delta=True),
        "top5": {
            "checked": len(with_margin),
            "changed": sum(
                1
                for item in with_margin
                if item.top5_overlap is not None
                and item.top5_overlap < min(5, item.config.vocab)
            ),
        },
        "configs": [item.to_dict() for item in results],
        "failures": [
            {
                "index": item.config.index,
                "seed": item.config.seed,
                "reasons": item.failure_reasons(),
                "config": item.config.to_dict(),
            }
            for item in failed
        ],
        "ok": not switches and not errors and not python_mismatch,
    }
    return summary


def _margin_brief(
    item: SimdResult | None, *, with_delta: bool = False
) -> dict[str, Any] | None:
    if item is None or item.backend is None:
        return None
    brief: dict[str, Any] = {
        "index": item.config.index,
        "seed": item.config.seed,
        "top1_margin": _json_float(item.backend.top1_margin),
        "max_abs_delta": _json_float(item.backend.max_abs_delta),
        "margin_to_delta": _json_float(item.backend.margin_to_delta),
        "argmax_baseline": item.backend.argmax_baseline,
        "argmax_candidate": item.backend.argmax_candidate,
    }
    if with_delta:
        brief["config"] = item.config.to_dict()
        brief["delta_at_top2"] = _json_float(item.delta_at_top2)
        brief["top5_overlap"] = item.top5_overlap
    return brief


def _json_float(value: float | None) -> Any:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else str(value)


def _fmt(value: float | None) -> str:
    if value is None:
        return "—"
    if not math.isfinite(float(value)):
        return str(value)
    return f"{float(value):.3e}"


def _fmt_ratio(value: float | None) -> str:
    if value is None:
        return "—"
    if not math.isfinite(float(value)):
        return "∞" if value > 0 else str(value)
    return f"{float(value):.1f}"


def format_report(summary: dict[str, Any]) -> str:
    """Человекочитаемая сводка (stdout без `--json`)."""
    delta = summary.get("delta") or {}
    argmax = summary.get("argmax") or {}
    python = summary.get("python_vs_baseline") or {}
    lines: list[str] = []
    lines.append("=== parity_simd: паритет бэкендов VTM ===")
    lines.append(
        f"seed: {summary['seed']}, конфигураций: {summary['total']}, "
        f"порог: {summary['tolerance']:g}"
    )
    lines.append(
        f"baseline: {summary['baseline']}, кандидат: {summary['candidate']}"
    )
    if summary.get("workdir"):
        lines.append(f"workdir: {summary['workdir']}")
    lines.append("")
    if not delta.get("count"):
        lines.append("нет ни одного успешного прогона — сравнивать нечего")
    else:
        lines.append(
            f"=== распределение max|Δ| логитов ({summary['baseline']} → "
            f"{summary['candidate']}) ==="
        )
        lines.append(
            f"конфигураций: {delta['count']}, побитово равны: "
            f"{delta['bit_exact']}/{delta['count']}"
        )
        lines.append(f"медиана:  {_fmt(delta['median'])}")
        lines.append(f"p95:      {_fmt(delta['p95'])}")
        lines.append(f"максимум: {_fmt(delta['max'])}")
        lines.append(f"среднее:  {_fmt(delta['mean'])}")
        lines.append(
            f"вне порога {summary['tolerance']:g}: "
            f"{delta['outside_tolerance']}/{delta['count']}; "
            f">= 1e-04: {delta['at_least_1e4']}/{delta['count']}"
        )
        lines.append("")
        lines.append("=== argmax ===")
        lines.append(
            f"смен argmax: {argmax['switches']}/{argmax['checked']}"
        )
        narrowest = argmax.get("narrowest_margin")
        if narrowest is not None:
            lines.append(
                f"самый узкий отрыв top1-top2: "
                f"{_fmt(narrowest['top1_margin'])} "
                f"(индекс {narrowest['index']}, max|Δ| там "
                f"{_fmt(narrowest['max_abs_delta'])})"
            )
        tightest = argmax.get("tightest_ratio")
        if tightest is not None:
            lines.append(
                f"минимальное отношение отрыв/Δ: "
                f"{_fmt_ratio(tightest['margin_to_delta'])} "
                f"(индекс {tightest['index']}: отрыв "
                f"{_fmt(tightest['top1_margin'])} при Δ "
                f"{_fmt(tightest['max_abs_delta'])})"
            )
        lines.append(
            "конфигураций с отрывом < 2*max|Δ| (смена возможна): "
            f"{argmax['borderline_below_2delta']}"
        )
        for item in argmax.get("switch_configs") or []:
            lines.append(
                f"  СМЕНА idx={item['index']}: {item['argmax_baseline']} -> "
                f"{item['argmax_candidate']}, max|Δ| = "
                f"{_fmt(_as_float(item['max_abs_delta']))}, отрыв top1-top2 = "
                f"{_fmt(_as_float(item['top1_margin']))}"
            )
        top5 = summary.get("top5") or {}
        if top5.get("checked"):
            lines.append(
                f"набор top-5 изменился: {top5['changed']}/{top5['checked']}"
            )
        lines.append("")
        lines.append(
            f"=== контроль: Python ↔ {summary['baseline']} ==="
        )
        lines.append(
            f"max|Δ| = {_fmt(_as_float(python.get('max_abs_delta')))} "
            f"(бит-в-бит: {python.get('bit_exact')}/{python.get('checked')}), "
            f"argmax совпал: "
            f"{python.get('checked', 0) - python.get('argmax_mismatches', 0)}"
            f"/{python.get('checked')}"
        )
        worst = summary.get("worst")
        if worst is not None:
            lines.append("")
            lines.append("=== худшая конфигурация по max|Δ| ===")
            config = worst.get("config") or {}
            lines.append(
                f"idx={worst['index']} seed={worst['seed']}: "
                f"{parity_random.format_config_from_dict(config)}"
            )
            lines.append(
                f"max|Δ| = {_fmt(_as_float(worst['max_abs_delta']))}, "
                f"отрыв top1-top2 = {_fmt(_as_float(worst['top1_margin']))}, "
                f"отрыв/Δ = {_fmt_ratio(_as_float(worst['margin_to_delta']))}, "
                f"Δ на втором кандидате = {_fmt(_as_float(worst['delta_at_top2']))}"
            )
            lines.append(
                "воспроизвести: "
                f"python3 tools/parity_simd.py --seed {worst['seed']} "
                f"--only {worst['index']}"
            )
    per_op = summary.get("per_op")
    if per_op is not None:
        lines.append("")
        lines.append("=== вклад операций (bench_backend --check) ===")
        if not per_op.get("available"):
            lines.append(f"пропущено: {per_op.get('reason')}")
        else:
            groups = per_op.get("groups") or {}
            for key, title in (
                ("tmv", "TMV (триты × f32)"),
                ("mm_trit", "MM trit"),
                ("mm_f32", "MM f32"),
                ("rmsn", "RMSN"),
            ):
                entry = groups.get(key) or {}
                if not entry.get("count"):
                    continue
                lines.append(
                    f"{title:22s} max|diff| = {_fmt(entry['max'])} "
                    f"({entry['count']} форм)"
                )
            lines.append(
                "в interp.c MM считается скалярно (vtm_backend.h) — "
                "в логиты MM-вклад SIMD не идёт"
            )
    if summary.get("failures"):
        lines.append("")
        lines.append("=== провалы ===")
        for failure in summary["failures"]:
            lines.append(
                f"idx={failure['index']} seed={failure['seed']}: "
                f"{parity_random.format_config_from_dict(failure['config'])}"
            )
            for reason in failure["reasons"]:
                lines.append(f"  - {reason}")
    lines.append("")
    lines.append("=== вердикт ===")
    if summary["ok"]:
        lines.append(
            f"argmax не изменился ни в одной из {summary['total']} конфигураций: "
            f"расхождение {summary['baseline']} ↔ {summary['candidate']} "
            f"(до {_fmt(_as_float(delta.get('max')))} при медиане "
            f"{_fmt(_as_float(delta.get('median')))}) на ответы не влияет."
        )
    else:
        lines.append(
            "argmax изменился или прогон сломан — это баг, а не допуск "
            f"(смен: {argmax.get('switches', 0)}, ошибок: {summary['errors']})."
        )
    lines.append("PASS" if summary["ok"] else "FAIL")
    return "\n".join(lines)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def format_json(summary: dict[str, Any]) -> str:
    return json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 tools/parity_simd.py",
        description=(
            "Паритет бэкендов VTM: скаляр против SIMD (VTM_BACKEND=auto) на "
            "случайных конфигурациях — распределение max|Δ| и смены argmax."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "примеры:\n"
            "  python3 tools/parity_simd.py\n"
            "  python3 tools/parity_simd.py --count 10 --seed 7\n"
            "  python3 tools/parity_simd.py --json\n"
            "  python3 tools/parity_simd.py --only 28\n"
            "  python3 tools/parity_simd.py --per-op\n\n"
            "Вердикт PASS — argmax нигде не сменился; расхождение логитов "
            "на 1e-5..1e-4 при этом нормально и побитового паритета не даёт.\n"
        ),
    )
    parser.add_argument(
        "--count", type=parity_random._positive_int,
        default=DEFAULT_COUNT,
        help=f"сколько конфигураций прогнать (по умолчанию {DEFAULT_COUNT})",
    )
    parser.add_argument(
        "--seed", type=parity_random._non_negative_int, default=DEFAULT_SEED,
        help=f"фиксированное зерно генератора (по умолчанию {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--only", type=parity_random.parse_only, default=None,
        help="индексы конфигураций (например 28 или 3,7); --count игнорируется",
    )
    parser.add_argument(
        "--tolerance", type=parity_random._positive_float,
        default=DEFAULT_TOLERANCE,
        help=(
            "порог «вне допуска» для max|Δ| при подсчёте хвоста "
            f"(по умолчанию {DEFAULT_TOLERANCE:g}); на вердикт не влияет"
        ),
    )
    parser.add_argument(
        "--baseline", type=parity_random.parse_backend,
        default=DEFAULT_BASELINE, metavar="ИМЯ",
        help=f"бэкенд-эталон (по умолчанию {DEFAULT_BASELINE})",
    )
    parser.add_argument(
        "--candidate", type=parity_random.parse_backend,
        default=DEFAULT_CANDIDATE, metavar="ИМЯ",
        help=(
            "бэкенд-кандидат (по умолчанию {0}: SIMD там, где он есть)".format(
                DEFAULT_CANDIDATE
            )
        ),
    )
    parser.add_argument(
        "--workdir", default=str(DEFAULT_WORKDIR),
        help=f"каталог контейнеров и C-пробы (по умолчанию {DEFAULT_WORKDIR})",
    )
    parser.add_argument(
        "--keep", action="store_true",
        help="не удалять контейнеры успешных прогонов",
    )
    parser.add_argument(
        "--per-op", action="store_true",
        help="добавить вклад операций из `bench_backend --check` (если собран)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="печатать JSON-сводку в stdout (прогресс — в stderr)",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def run(args: argparse.Namespace) -> int:
    if args.baseline == args.candidate:
        raise ParitySimdError(
            f"--baseline и --candidate совпадают ({args.baseline}): "
            "сравнивать бэкенд сам с собой нечего"
        )
    env = parity_random.check_environment()
    missing = env.missing()
    if missing:
        raise ParitySimdError(
            "для прогона не хватает: "
            + ", ".join(missing)
            + "\nC-рантайм и библиотека собираются: make -C src/vtm"
        )

    indices = (
        list(args.only)
        if args.only is not None
        else list(range(int(args.count)))
    )
    workdir = Path(args.workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    probe = parity_random.ensure_probe(workdir, env)

    print(
        f"[{TOOL_NAME}] seed={args.seed}, конфигураций: {len(indices)}, "
        f"{args.baseline} vs {args.candidate}, workdir: {workdir}",
        file=sys.stderr,
        flush=True,
    )
    print(f"[{TOOL_NAME}] C-проба: {probe}", file=sys.stderr, flush=True)

    results: list[SimdResult] = []
    started = time.perf_counter()
    for position, index in enumerate(indices, start=1):
        config = parity_random.generate_config(args.seed, index)
        problems = parity_random.validate_config(config)
        if problems:  # pragma: no cover - генератор обязан давать валидные
            raise ParitySimdError(
                f"невалидная конфигурация idx={index}: " + "; ".join(problems)
            )
        result = run_config(
            config,
            workdir=workdir,
            probe=probe,
            baseline=args.baseline,
            candidate=args.candidate,
            tolerance=args.tolerance,
            keep=bool(args.keep),
        )
        results.append(result)
        status = "ok  " if result.passed else "FAIL"
        print(
            f"[{TOOL_NAME}] [{position:>3}/{len(indices)}] {status} "
            f"idx={index:<3} {parity_random.format_config(config)} "
            f"max|Δ|={_fmt(result.max_abs_delta)} "
            f"argmax {result.backend.argmax_baseline if result.backend else '?'}"
            f"{'=' if result.argmax_match else '->'}"
            f"{result.backend.argmax_candidate if result.backend else '?'} "
            f"отрыв={_fmt(result.margin)} ({result.elapsed_ms:.0f} мс)",
            file=sys.stderr,
            flush=True,
        )
        if result.error is not None:
            print(
                f"[{TOOL_NAME}]   ошибка: {result.error}",
                file=sys.stderr,
                flush=True,
            )

    summary = summarize(
        seed=args.seed,
        count=len(indices),
        baseline=args.baseline,
        candidate=args.candidate,
        tolerance=args.tolerance,
        results=results,
        workdir=workdir,
    )
    summary["total_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
    if args.per_op:
        summary["per_op"] = per_op_report()

    if args.json:
        print(format_json(summary))
    else:
        print(format_report(summary))
    print(
        f"[{TOOL_NAME}] готово за {summary['total_ms'] / 1000.0:.1f} с: "
        f"смен argmax {summary['argmax']['switches']}, "
        f"ошибок {summary['errors']}",
        file=sys.stderr,
        flush=True,
    )
    return 0 if summary["ok"] else 1


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
    except SystemExit as exc:  # argparse уже напечатал ошибку
        return int(exc.code) if isinstance(exc.code, int) else 2
    try:
        return run(args)
    except ParitySimdError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except parity_random.ParityRandomError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(f"\n[{TOOL_NAME}] прервано", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
