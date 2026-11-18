---
task_id: "0051"
type: feature
title: Сэмплинг: temperature, top-k, top-p, повтор
created_at: 2026-11-18T07:20:00Z
author: yuri
assignee: agent:vtm-sampling
status: in_progress
scope:
  docs: false
  code: true
  data: false
---

# Сэмплинг: temperature, top-k, top-p, повтор

## Проблема

SAMP умеет один вариант — похоже, argmax. Для ассистента нужны temperature, top-k,
top-p и штраф за повторы, иначе модель будет зацикливаться.

## Требования

- `imm` у SAMP — номер стратегии: 0 argmax, 1 temperature, 2 top-k, 3 top-p.
- Параметры в мете контейнера: temperature, top_k, top_p, repetition_penalty.
- Детерминизм: при temperature 0 и argmax результат побитово одинаков при повторах.
- Штраф за повторы по окну последних N токенов.
- Тесты на каждую стратегию: распределение, границы, детерминизм.

## Критерии приёмки

[ ] argmax детерминирован (3 запуска — один токен)
[ ] temperature 0 == argmax
[ ] top-k 1 == argmax; top-p 1.0 не отсекает ничего
[ ] штраф за повторы меняет выбор зацикленного примера
[ ] `make test` зелёный

## Контекст и ограничения

- Зона: src/vtm/interp.c, src/vtm/vtm.h, tests/vtm/test_vtm.c
- Другие файлы не трогать: параллельно работают другие агенты.
- Сборка: `cd src/vtm && make test`; флаги `-Wall -Wextra -Werror`.
- Не ломать существующие тесты и поведение без флагов.
- Не коммитить: сдавать правки и отчёт.

## Отчёт исполнителя

_Заполняется по завершении._
