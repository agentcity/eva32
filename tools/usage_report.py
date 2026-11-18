#!/usr/bin/env python3
"""Человекочитаемый отчёт по логу телеметрии гибрида.

Запуск::

    python3 tools/usage_report.py --path logs/eva.jsonl
    python3 tools/usage_report.py --path logs/eva.jsonl --json

Отчёт считает по маршрутам и дням: число запросов, среднюю и суммарную цену,
цену на 1000 запросов, p50/p95 латентности. Битые строки не роняют отчёт:
они пропускаются, а их число и номера печатаются в конце. Код возврата 0,
даже если часть строк повреждена; 1 — только если файл не читается.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from eva.telemetry import summary  # noqa: E402

_HEADERS = {
    "route": "Маршрут",
    "req": "Запросов",
    "share": "Доля",
    "avg": "Средняя, ₽",
    "total": "Итого, ₽",
    "per1000": "На 1000, ₽",
    "p50": "p50, мс",
    "p95": "p95, мс",
}

_TEMPLATE = ("{route:<{w}}  {req:>8}  {share:>5}  {avg:>11}  {total:>9}  "
             "{per1000:>10}  {p50:>8}  {p95:>8}")


def _ms(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def _plural(number: int, one: str, few: str, many: str) -> str:
    """Русская форма слова для числа: 1 запрос, 2 запроса, 5 запросов."""
    tail = number % 100
    if 11 <= tail <= 14:
        return many
    tail = number % 10
    if tail == 1:
        return one
    if 2 <= tail <= 4:
        return few
    return many


def _row(route: str, stats: dict, requests: int, width: int) -> str:
    share = f"{100.0 * stats['requests'] / requests:.0f}%" if requests else "—"
    return _TEMPLATE.format(
        route=route,
        w=width,
        req=str(stats["requests"]),
        share=share,
        avg=f"{stats['cost_rub_avg']:.4f}",
        total=f"{stats['cost_rub']:.2f}",
        per1000=f"{stats['cost_rub_per_1000']:.2f}",
        p50=_ms(stats["latency_ms_p50"]),
        p95=_ms(stats["latency_ms_p95"]),
    )


def _table(report: dict, width: int) -> list[str]:
    header = _TEMPLATE.format(
        route=_HEADERS["route"], w=width,
        req=_HEADERS["req"], share=_HEADERS["share"], avg=_HEADERS["avg"],
        total=_HEADERS["total"], per1000=_HEADERS["per1000"],
        p50=_HEADERS["p50"], p95=_HEADERS["p95"],
    )
    lines = [header, "-" * len(header)]
    for name, route_stats in report["routes"].items():
        lines.append(_row(name, route_stats, report["records"], width))
    lines.append("-" * len(header))
    lines.append(_row("ИТОГО", report["total"], report["records"], width))
    return lines


def render(report: dict) -> str:
    """Собрать текстовый отчёт (без печати — удобно тестировать)."""
    width = max([len(_HEADERS["route"])] + [len(name) for name in report["routes"]])
    lines = [
        "EVA32 · телеметрия гибрида",
        f"файл:    {report['path']}",
        f"записей: {report['records']}   битых строк пропущено: {report['skipped']}",
        "",
    ]
    if not report["records"]:
        lines.append("записей нет — лог пуст")
    else:
        lines.extend(_table(report, width))
        lines.extend([
            "",
            "Перцентили — линейная интерполяция (R-7); цена на 1000 запросов = "
            "средняя цена × 1000.",
        ])
        skills = report["total"]["skills"]
        if skills:
            lines.extend(["", "Скиллы:"])
            for name, count in skills.items():
                lines.append(f"  {name} — {count}")
        lines.extend(["", "По дням:"])
        for day, day_stats in report["days"].items():
            count = day_stats["requests"]
            noun = _plural(count, "запрос", "запроса", "запросов")
            lines.append(
                f"  {day}: {count} {noun}, "
                f"{day_stats['cost_rub']:.2f} ₽, "
                f"{day_stats['cost_rub_per_1000']:.2f} ₽ / 1000, "
                f"p50 {_ms(day_stats['latency_ms_p50'])} мс, "
                f"p95 {_ms(day_stats['latency_ms_p95'])} мс")
    if report["skipped"]:
        lines.extend(["", f"Битые строки (пропущены): {report['skipped']}"])
        for error in report["errors"]:
            lines.append(f"  строка {error['line']}: {error['error']}")
        hidden = report["skipped"] - len(report["errors"])
        if hidden > 0:
            lines.append(f"  … и ещё {hidden}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Отчёт по логу телеметрии гибрида (JSONL): маршруты, цена, латентность.",
    )
    parser.add_argument(
        "--path", default="logs/eva.jsonl",
        help="путь к JSONL-логу (по умолчанию logs/eva.jsonl)")
    parser.add_argument(
        "--json", action="store_true",
        help="напечатать агрегат в JSON вместо человекочитаемой таблицы")
    args = parser.parse_args(argv)

    try:
        report = summary(args.path)
    except FileNotFoundError:
        print(f"лог не найден: {args.path}", file=sys.stderr)
        print("подсказка: путь задаётся через --path, лог пишет eva.telemetry.log()",
              file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"не удалось прочитать {args.path}: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
