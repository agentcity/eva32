---
task_id: "0127"
type: feature
title: C-ABI отдаёт prefill_ms — TTFT делится на боевом локальном пути
created_at: 2026-11-23T14:55:00Z
author: yuri
assignee: ""
status: pending
scope:
  docs: false
  code: true
  data: false
---

# Граница prefill из C-ABI

## Проблема

Задача 0120 довела границу prefill до метрик: `prefill_ms`, `ttft_after_prefill_ms`.
Но источник данных — бэкенд, а `vtm_api_generate` время фаз не отдаёт, поэтому на
боевом локальном пути `prefill_ms = None` и TTFT не делится.

## Требования

- C-ABI: после генерации доступно время prefill — либо в `vtm_api_options` (поле из
  `reserved`, sizeof не менять), либо геттером `vtm_api_last_timings(handle, &out)`.
- Внутри: prefill = прогон промпта (включая `vtm_run_batch`), генерация начинается
  с первого сэмплированного токена; при `cache_prefix` попадании prefill = 0.
- Python: `eva/backends/vtm_backend.py` выставляет `last_prefill_ms`, роутер уже умеет
  его читать (0120).
- Тест: на demo `prefill_ms > 0` и `<=` общего времени; при кэше префикса второй
  прогон даёт prefill_ms ≈ 0.
- `sizeof(vtm_api_options)` не меняется; старые хосты работают.

## Критерии приёмки

- [ ] `/metrics` показывает `prefill_ms` на локальном пути (не None)
- [ ] второй запрос с кэшем префикса: prefill_ms ≈ 0
- [ ] ABI: sizeof и смещения прежних полей не изменились (проверка программой)
- [ ] `make test` и Python-набор зелёные

## Отчёт исполнителя

_Заполняется по завершении._

## Ссылки

- `ai/tasks/0120-router-prefill-ms.md`, `ai/tasks/0115-server-ttft.md`
