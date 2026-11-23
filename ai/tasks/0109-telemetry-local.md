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

- [x] `local` в JSONL по каждому запросу
- [x] `/metrics` считает долю локальных ответов
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

- `ai/tasks/0104-eva-router-vtm.md`, `ai/tasks/0108-server-limits.md`
