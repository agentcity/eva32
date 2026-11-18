---
task_id: "0075"
type: fix
title: ATTN — переполнение q_start+q_len (SIGBUS на malformed-контейнере)
created_at: 2026-11-18T10:20:00Z
author: yuri
assignee: saron
status: done
scope:
  docs: false
  code: true
  data: false
---

# Переполнение в ATTN

## Проблема

Найдено фаззингом (задача 0069): контейнер с `imm = INT_MAX` у инструкции ATTN
(2-D K/V-кэш) вызывал `q_start = position + imm`, затем `key_count = q_start + q_len`
— **signed overflow**. Отрицательный `key_count` проходил проверку «больше ctx»,
после чего `malloc((size_t)key_count * sizeof(double))` получал отрицательную длину
и запись шла за пределы буфера: **SIGBUS**, exit 138. То есть испорченный файл
(не обрезанный, а с подменённым `imm`) валил процесс.

Repro: `tests/vtm/fixtures/attn_imm_int_max.vtm`, sha256
`08d0b55dbcd188d9d208e2e24d0e80828baf53a7e93a4115d31fd35393ebad70`.

## Решение

Арифметика в `long long` с проверками **до** приведения к `int`:

```c
long long qs   = (long long)m->position + (long long)in->imm;
long long keys = qs + (long long)q_len;
if (qs < 0 || qs > (long long)INT_MAX) fail(...);
if (keys > (long long)ctx)              fail(...);
```

## Критерии приёмки

- [x] тот же контейнер даёт ошибку, а не SIGBUS
- [x] текст ошибки называет ATTN и величины
- [x] регрессионный тест `test_attn_imm_overflow` в `make test`
- [x] остальные тесты не сломаны

## Отчёт исполнителя

Исправлено архитектором (interp.c — вне зоны фаззинг-задачи). Проверено: `exit 1` и
«ATTN: q_start+q_len=2147483648 больше ctx=8» вместо падения. Добавлен `<limits.h>`,
тест использует фикстуру из находки.

## Ссылки

- `ai/tasks/0069-vtm-fuzz.md`, `fuzz-findings/`
