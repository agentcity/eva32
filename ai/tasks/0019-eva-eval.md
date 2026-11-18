---
task_id: "0019"
type: feature
title: Гибрид — eva-eval
created_at: 2026-11-18T04:40:00Z
author: yuri
assignee: agent:eva-eval
status: in_progress
scope:
  docs: false
  code: true
  data: true
---

# Гибрид: eva-eval

Контракт — `docs/hybrid.md`, решение — `docs/adr/0009-hybrid-router.md`.
Читать оба до начала работы.

## Проблема

Замер гибрида: правильные ли маршруты и не дороже ли, чем нужно.

## Требования

- `eval/hybrid_golden.jsonl`: 40–60 строк `{question, expect_route, expect_contains}`.
- `tools/eval_hybrid.py`: прогон с заглушками моделей (без сети и ключей), считает
  точность маршрутизации, долю скиллов, ожидаемую цену на 1000 запросов.
- Тест фиксирует текущие метрики, чтобы регрессия была видна.

## Критерии приёмки

[ ] `python3 tools/eval_hybrid.py` печатает точность маршрутов и цену/1000
[ ] `python3 -m unittest tests.test_eval_hybrid -v` зелёный, без сети
[ ] метрики зафиксированы в тесте (регрессия ловится)

## Контекст и ограничения

- Зона: tools/eval_hybrid.py, eval/hybrid_golden.jsonl, tests/test_eval_hybrid.py
- Другие файлы не трогать: над репозиторием работают восемь агентов.
- Только stdlib Python, без сети в тестах, без установки пакетов.
- Не коммитить: сдавать правки и отчёт.

## Отчёт исполнителя

_Заполняется по завершении._

## Ссылки

- `docs/hybrid.md`
- `docs/adr/0009-hybrid-router.md`
