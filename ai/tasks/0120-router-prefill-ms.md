---
task_id: "0120"
type: feature
title: Роутер отдаёт границу prefill — метрики TTFT делятся на prefill и генерацию
created_at: 2026-11-18T18:45:00Z
author: yuri
assignee: ""
status: pending
scope:
  docs: false
  code: true
  data: false
---

# Граница prefill в Answer

## Проблема

Задача 0115 подготовила метрики `prefill_ms` и `ttft_after_prefill_ms`, но они всегда
`None`: сервер ждёт от роутера поле с моментом окончания prefill, а `eva/router.py` его
не отдаёт. В итоге видно только полный TTFT (в котором доминирует prefill), и нельзя
понять, сколько времени уходит на сам промпт, а сколько на первый токен.

## Требования

- `Answer` (или `usage`/`timings`) получает `prefill_ms`; потоковый путь — в куске-мете.
- ВТМ-бэкенд знает границу: prefill-часть = прогон промпта, генерация начинается с
  первого сэмплированного токена.
- Тест: метрики заполняются (`prefill_ms` не None), `ttft_after_prefill_ms` = TTFT − prefill.

## Критерии приёмки

- [ ] `prefill_ms` появляется в `/metrics` по каждому запросу
- [ ] потоковый путь тоже отдаёт границу
- [ ] Python-набор зелёный

## Отчёт исполнителя

_Заполняется по завершении._

## Ссылки

- `ai/tasks/0115-server-ttft.md`
