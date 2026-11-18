---
task_id: "0026"
type: feature
title: Подключение к orioncode
created_at: 2026-11-18T05:00:00Z
author: yuri
assignee: agent:eva-integration
status: in_progress
scope:
  docs: true
  code: true
  data: false
---

# Подключение к orioncode

Контракт — `docs/hybrid.md`, решение — `docs/adr/0009-hybrid-router.md`.

## Проблема

Сервер OpenAI-совместим, но сторона orioncode его ещё не видела: нет примера конфига,
нет smoke-скрипта, нет списка того, что надо поменять у них.

## Требования

- `docs/integration-orioncode.md`: base URL, ключ, модель, что делать с полем `eva`,
  как читать маршрут и цену, чем отличается от vLLM-провайдера, что делать при 500/429.
- `scripts/integration_smoke.py`: проверяет живой сервер — `/health`, `/v1/models`,
  чат без стрима и со стримом; печатает маршрут и цену; exit 1 при провале.
- `.env.example`: добавить блок `EVA_*` (сейчас там только TEACHER_*/MODEL_REVISION).

## Критерии приёмки

[ ] `python3 scripts/integration_smoke.py --url http://127.0.0.1:8010` проходит на поднятом сервере
[ ] документ содержит рабочий пример конфига провайдера
[ ] `bash -n` на всех блоках, ссылки на ADR 0009 и docs/hybrid.md

## Контекст и ограничения

- Зона: docs/integration-orioncode.md, scripts/integration_smoke.py, .env.example (только блок EVA_*)
- Другие файлы не трогать: параллельно работают другие агенты.
- Только stdlib, без сети в тестах, без установки пакетов.
- Не коммитить: сдавать правки и отчёт.

## Отчёт исполнителя

_Заполняется по завершении._

## Ссылки

- `docs/hybrid.md`
- `docs/adr/0009-hybrid-router.md`
