---
task_id: "0114"
type: feature
title: CLI и C-ABI для батчевого prefill
created_at: 2026-11-18T18:20:00Z
author: yuri
assignee: agent:prefill-cli
status: in_progress
scope:
  docs: false
  code: true
  data: false
---

# CLI и C-ABI для батчевого prefill

## Проблема

Даже когда ядро научится считать промпт пачкой (0112), CLI и C-ABI должны это
использовать: сейчас `vtm run --tokens ...` гоняет по токену, а `vtm_api_generate` — так
же. Без проводки выигрыш останется в тестах.

## Требования

- CLI: `--prefill-batch N` (0 = по токену, как раньше) — промпт считается пачками по N.
- C-ABI: `vtm_api_options.prefill_batch` (занять элемент из `reserved`, sizeof сохранить) —
  тот же смысл; по умолчанию 0, поведение не меняется.
- Использовать `vtm_run_batch` из задачи 0112 (если её ещё нет — сделать заглушку,
  которая зовёт прежний путь, и переключиться, когда появится).
- Замер: TTFT на demo и на 1.5B-форме с --prefill-batch 32/128 против 0.

## Критерии приёмки

[ ] CLI --prefill-batch работает, результаты побитово те же (тест)
[ ] cache_prefix/prefill_batch в api.h без смены sizeof (проверка)
[ ] TTFT до/после замерен
[ ] make test и Python-набор зелёные

## Контекст и ограничения

- Зона: src/vtm/main.c, src/vtm/api.c, src/vtm/api.h, tests/test_vtm_api.c
- Другие файлы НЕ трогать: параллельно работают семь других агентов по своим зонам.
  В частности `interp.c` — 0112, `backend_*.c` — 0113, `main.c/api.c/api.h` — 0114,
  `eva/server.py` — 0115, `docs/` — 0116, `tools/ci.py` — 0117, `loader.c` — 0118,
  `prompts/` — 0119.
- **Запрещено:** pip, сеть, запуск сервера на порту, git-коммиты, скачивание моделей.
- **Отчёт пиши первым блоком ответа** — агенты этой сессии падают на финальном шаге.
- Проверка: `python3 -m unittest discover -s tests` (сейчас 1517 тестов) и, где нужно,
  `cd src/vtm && make test`.
- Если нужен API из чужой зоны (например `vtm_run_batch` из 0112) — опиши контракт в
  отчёте и используй заглушку, не правь чужой файл.

## Отчёт исполнителя

_Заполняется по завершении._
