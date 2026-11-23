---
task_id: "0130"
type: feature
title: qat/export.py: экспорт int8-эмбеддинга end-to-end
created_at: 2026-11-23T15:20:00Z
author: yuri
assignee: ""
status: pending
scope:
  docs: false
  code: true
  data: false
---

# Экспорт int8

## Контекст

`qat/export.py` отказывает `quantize_embeddings: int8` (EMBED_INT8_DEFERRED), поэтому путь «обучение → экспорт → int8-тензор → компиляция» не проверен. Контейнер, компилятор и манифест int8 готовы (0044).

## Требования

- `export.py` пишет int8-секцию и манифест `kind: int8`;
- `parity_real`, `embedding_experiment` перестают писать «int8 отложен»;
- паритет ≤1e-5 на реальной модели.

## Критерии приёмки

- [ ] результат совпадает с текущим путём (допуск в отчёте)
- [ ] числа до/после (память/скорость)
- [ ] `make test` и Python-набор зелёные

## Отчёт исполнителя

_Заполняется по завершении._
