---
task_id: "0021"
type: feature
title: Гибрид — eva-telemetry
created_at: 2026-11-18T04:40:00Z
author: yuri
assignee: agent:eva-telemetry
status: in_progress
scope:
  docs: false
  code: true
  data: false
---

# Гибрид: eva-telemetry

Контракт — `docs/hybrid.md`, решение — `docs/adr/0009-hybrid-router.md`.
Читать оба до начала работы.

## Проблема

Учёт каждого ответа: маршрут, цена, латентность — чтобы себестоимость не была загадкой.

## Требования

- `telemetry.log(record, path)` — append JSONL, атомарно на строку.
- `summary(path)` → по маршрутам: число запросов, средняя цена, p50/p95 латентности.
- `tools/usage_report.py --path logs/eva.jsonl` — человекочитаемый отчёт, в т.ч. цена на 1000 запросов.
- Ничего не пишет за пределы указанного пути.

## Критерии приёмки

[ ] `python3 -m unittest tests.test_telemetry -v` зелёный
[ ] отчёт на 100 записях считает суммы/перцентили, сверено вручную в тесте
[ ] битая строка в логе не роняет отчёт (пропускается со счётчиком)

## Контекст и ограничения

- Зона: eva/telemetry.py, tools/usage_report.py, tests/test_telemetry.py
- Другие файлы не трогать: над репозиторием работают восемь агентов.
- Только stdlib Python, без сети в тестах, без установки пакетов.
- Не коммитить: сдавать правки и отчёт.

## Отчёт исполнителя

_Заполняется по завершении._

## Ссылки

- `docs/hybrid.md`
- `docs/adr/0009-hybrid-router.md`
