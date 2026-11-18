---
task_id: "0046"
type: feature
title: Замер качества обученной модели (eval_torch)
created_at: 2026-11-18T07:45:00Z
author: yuri
assignee: saron
status: done
scope:
  docs: false
  code: true
  data: false
---

# eval_torch

## Проблема

Сравнение прогонов none/int8/trit по loss ничего не говорит о качестве ответов.
Нужен замер на GPU-машине: golden-проверки, perplexity, длина ответов.

## Требования

- `tools/eval_torch.py --model ... --golden eval/golden.jsonl --out runs/eval.json`.
- `--model` принимает HF id, локальный каталог или каталог прогона (берёт последний чекпоинт).
- Считает accuracy по поведенческим проверкам (`eval/run_eval.py::check`), perplexity
  по `--ppl-data` (опционально), среднюю длину ответа.
- `--dry-run` без torch: план и проверка golden, exit 0; без torch обычный запуск — exit 2.

## Критерии приёмки

- [x] `--dry-run` работает без torch
- [x] 50 тестов зелёные, полный набор не сломан
- [x] JSON содержит accuracy, passed, total, perplexity, avg_len

## Отчёт исполнителя

Агент упал на середине, оставив рабочий `tools/eval_torch.py` (dry-run работает) и
тест-файл с синтаксической ошибкой на месте обрыва. Архитектор исправил одну строку,
прогнал тесты (50 OK) и принял. Запуск на реальной модели — на арендованном GPU.

## Ссылки

- `eval/run_eval.py`, `qat/data.py`
