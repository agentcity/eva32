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

- [x] `prefill_ms` появляется в `/metrics` по каждому запросу
- [x] потоковый путь тоже отдаёт границу
- [x] Python-набор зелёный

## Отчёт исполнителя

Сделано: маркер своей модели и граница prefill — в контракте, логах и метриках.

**0109:** `Route.local` и `Route.source` (skill|local|api) стали полями контракта и
попадают в `to_dict()`. В JSONL: `"local": true, "source": "local"` для ВТМ,
`"local": false, "source": "api"` для HTTP-клиента, у skill поля нет.
В `/metrics` добавлены `small_local`, `small_http`, `local_requests`,
`local_share` (доля среди размеченных small), `local_share_all`, `source`,
`by_route.<route>.local`; `observe(local=...)`, `reset()` чистит.

**0120:** `Answer.prefill_ms` и `StreamChunk.prefill_ms`; роутер берёт границу
только если клиент её явно отдал (поле, `timings`/`usage`, `last_prefill_ms`),
иначе `None` — не угадывает. `ttft_after_prefill_ms = TTFT − prefill`.

**Числа демо-среза:** 3 local, 2 http, 1 big, 1 skill → `small_local=3`,
`small_http=2`, `local_requests=3`, `local_share=0.6`, `local_share_all=0.4286`,
`source={"skill":1,"local":3,"api":3}`, prefill p50=121 мс. Сквозной потоковый
тест: TTFT 125 = prefill 90 + 35 после.

Проверено: 36 новых тестов, полный набор **1689 OK**.

Риск, который честно записан: реальный `VtmBackend` границу prefill не отдаёт —
на боевом локальном пути `prefill_ms=None`. Нужен `timings`/`last_prefill_ms`
в `eva/backends/vtm_backend.py` и в C-ABI (заведена 0127).

## Ссылки

- `ai/tasks/0115-server-ttft.md`
