---
task_id: "0044"
type: feature
title: Kind 4 — int8-тензоры в контейнере (для эмбеддингов int8)
created_at: 2026-11-18T07:05:00Z
author: yuri
assignee: ""
status: pending
scope:
  docs: false
  code: true
  data: false
---

# Kind 4: int8-тензоры

## Проблема

Задача 0043 сделала сквозной путь для троичных эмбеддингов, но int8 отложила:
экспорт с `quantize_embeddings: int8` падает с объяснением, потому что в контейнере
нет kind 4 (int8 + масштабы) и его чтения в `src/vtm/loader.c` и `tools/vtm_ref.py`.

Пока этого нет, «план Б» (int8-эмбеддинги, если триты теряют качество) нельзя
довезти до рантайма: качество измерить можно на чекпоинте, а модель запустить — нет.

## Требования

- `kind 4` в контейнере: знаковый int8 + масштабы по блоку 128 (`f16`/`f32` по флагу).
- `tools/vtm_ref.py`: запись и чтение kind 4.
- `src/vtm/loader.c`: загрузка kind 4 в арену как fp32 (или ленивое чтение в EMB).
- `src/vtm/interp.c`: `EMB` из int8-таблицы.
- `qat/export.py` и `tools/vtm_compile.py`: снять запрет на `int8`, тесты.

## Критерии приёмки

- [ ] экспорт `int8` даёт int8-тензор таблицы и манифест с `kind: int8`
- [ ] `EMB` из int8 = расквантование в Python (1e-6)
- [ ] размер экспорта int8 меньше fp16 и больше trit (числа в отчёте)
- [ ] `python3 -m unittest discover -s tests` и `cd src/vtm && make test` зелёные

## Контекст и ограничения

- Делать только если эксперимент 0027 покажет, что триты теряют больше 3–5%.
- Зона: `tools/vtm_ref.py`, `src/vtm/loader.c`, `src/vtm/interp.c`, `src/vtm/vtm.h`,
  `qat/export.py`, `tools/vtm_compile.py`, `docs/vtm.md`, `tests/test_embeddings_e2e.py`.

## Отчёт исполнителя

_Заполняется по завершении._

## Ссылки

- `ai/tasks/0043-trit-embeddings-e2e.md`
- `docs/adr/0011-container-v02.md`
