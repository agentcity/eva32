---
task_id: "0125"
type: bugfix
title: CI на Ubuntu: gcc-предупреждения валили сборку с -Werror
created_at: 2026-11-19T20:55:00Z
author: yuri
assignee: architect
status: done
scope:
  docs: false
  code: true
  data: false
---

# Ubuntu-плечо CI

## Проблема

`make -C src/vtm test` падал на ubuntu-latest при зелёном macos-latest: clang не
предупреждает о части конструкций, а сборка идёт с `-Werror`. Логи job'а без токена
недоступны.

## Что сделано

Ubuntu-плечо CI падало с самого начала (macOS проходил): clang не ловит часть
предупреждений gcc, а сборка идёт с `-Werror`. Логи job'а требуют токена, поэтому
первым делом сделан шаг диагностики: вывод сборки и тестов уходит в `::error::`-
аннотации, а их можно читать через API (`check-runs/{id}/annotations`).

Найдено и исправлено пять классов:

1. **Всегда-истинное сравнение** `(unsigned)rec->opcode < 256` при `uint8_t`
   (gcc `-Wtype-limits`) — проверка убрана, вместо неё compile-time assert.
2. **Пять сравнений `> (uint64_t)SIZE_MAX`** в loader.c и kv_store.c: на 64-битных
   SIZE_MAX == UINT64_MAX, проверка всегда ложна и не ловила переполнение умножения.
   Заменены на `mul_u64`/`add_u64`/`fits_size_t`.
3. **`token` в op_samp** — gcc `-Wmaybe-uninitialized` (запись идёт через указатель);
   инициализирован явно.
4. **Мёртвые переменные** в тестах (`d_kv` ×2, `d_emb`) — gcc
   `-Wunused-but-set-variable`.
5. **format-truncation и multi-line comment** в test_vtm_session: буферы не помогали
   (длина `dir` неизвестна), поэтому пути собираются через `join_path()` с malloc.

Чтобы класс не повторялся, clang-сборка локально включает те же проверки:
`-Wtautological-compare`, `-Wtautological-constant-out-of-range-compare`,
`-Wunused-but-set-variable` (для gcc их заменяет `-Wtype-limits`/`-Wall`).

Итог: run 142 зелёный на ubuntu-latest и macos-latest. Диагностический шаг остался
в workflow (срабатывает только при падении) — следующий раз причина будет видна сразу.

## Критерии приёмки

- [x] оба плеча CI зелёные (run 142)
- [x] причина видна без доступа к логам (аннотации)
- [x] локальная сборка ловит те же классы предупреждений

## Отчёт исполнителя

Ubuntu-плечо CI падало с самого начала (macOS проходил): clang не ловит часть
предупреждений gcc, а сборка идёт с `-Werror`. Логи job'а требуют токена, поэтому
первым делом сделан шаг диагностики: вывод сборки и тестов уходит в `::error::`-
аннотации, а их можно читать через API (`check-runs/{id}/annotations`).

Найдено и исправлено пять классов:

1. **Всегда-истинное сравнение** `(unsigned)rec->opcode < 256` при `uint8_t`
   (gcc `-Wtype-limits`) — проверка убрана, вместо неё compile-time assert.
2. **Пять сравнений `> (uint64_t)SIZE_MAX`** в loader.c и kv_store.c: на 64-битных
   SIZE_MAX == UINT64_MAX, проверка всегда ложна и не ловила переполнение умножения.
   Заменены на `mul_u64`/`add_u64`/`fits_size_t`.
3. **`token` в op_samp** — gcc `-Wmaybe-uninitialized` (запись идёт через указатель);
   инициализирован явно.
4. **Мёртвые переменные** в тестах (`d_kv` ×2, `d_emb`) — gcc
   `-Wunused-but-set-variable`.
5. **format-truncation и multi-line comment** в test_vtm_session: буферы не помогали
   (длина `dir` неизвестна), поэтому пути собираются через `join_path()` с malloc.

Чтобы класс не повторялся, clang-сборка локально включает те же проверки:
`-Wtautological-compare`, `-Wtautological-constant-out-of-range-compare`,
`-Wunused-but-set-variable` (для gcc их заменяет `-Wtype-limits`/`-Wall`).

Итог: run 142 зелёный на ubuntu-latest и macos-latest. Диагностический шаг остался
в workflow (срабатывает только при падении) — следующий раз причина будет видна сразу.

## Ссылки

- `ai/tasks/0102-ci-pipeline.md`, `.github/workflows/ci.yml`
