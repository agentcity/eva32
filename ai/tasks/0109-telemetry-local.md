---
task_id: "0109"
type: feature
title: Телеметрия — маркер локальной модели в Route и JSONL
created_at: 2026-11-18T17:00:00Z
author: yuri
assignee: ""
status: pending
scope:
  docs: false
  code: true
  data: false
---

# Локальная модель в телеметрии

## Проблема

Задача 0104 добавила маркер `Answer.usage["local"]` и `router.route_is_local(route)`, но:

- `Route.local` — динамический атрибут, а не поле `eva.types.Route`;
- `Route.to_dict()` его не содержит;
- `telemetry.normalize_record` пишет только `kind/reason/confidence`.

Поэтому в JSONL нельзя посчитать, сколько запросов реально ответила **своя** модель, а
сколько ушло в API. Это ключевая метрика для решения «хватает ли 1.5B».

## Требования

- Поле `local: bool | None` в `eva.types.Route` (None — не применимо, например скилл).
- `to_dict()` включает его; `normalize_record` пишет `local` в запись.
- Сводка `/metrics`: счётчики `small_local` / `small_http` и доля локальных ответов.
- Тест: маршрут через локальную ВТМ → в записи `local=true`; через HTTP → `false`;
  скилл → блока нет.

## Критерии приёмки

- [ ] `local` в JSONL по каждому запросу
- [ ] `/metrics` считает долю локальных ответов
- [ ] Python-набор зелёный

## Отчёт исполнителя

_Заполняется по завершении._

## Ссылки

- `ai/tasks/0104-eva-router-vtm.md`, `ai/tasks/0108-server-limits.md`
