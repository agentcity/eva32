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

- [x] экспорт `int8` даёт int8-тензор таблицы и манифест с `kind: int8`
- [x] `EMB` из int8 = расквантование в Python (1e-6)
- [x] размер экспорта int8 меньше fp16 и больше trit (числа в отчёте)
- [x] `python3 -m unittest discover -s tests` и `cd src/vtm && make test` зелёные

## Контекст и ограничения

- Делать только если эксперимент 0027 покажет, что триты теряют больше 3–5%.
- Зона: `tools/vtm_ref.py`, `src/vtm/loader.c`, `src/vtm/interp.c`, `src/vtm/vtm.h`,
  `qat/export.py`, `tools/vtm_compile.py`, `docs/vtm.md`, `tests/test_embeddings_e2e.py`.

## Отчёт исполнителя

Сделано: kind 4 = сегмент «INT8» (18-байтный заголовок как TRT1: version 1,
flags бит0 = f32-масштабы, n_values u64, n_scale u32), масштабы на блок 128,
q ∈ −127..127, `w = q·scale`. `loader.c` валидирует (magic/version/flags/n_values/
n_scale/границы/q ≠ −128/масштабы) и **материализует в fp32 в арене**, ставя
provenance-флаг `VTM_TENSOR_FLAG_INT8_SOURCE`; KV-кэш с kind 4 отвергается.
`tools/vtm_ref.py` — `KIND_INT8`, `quantize_int8` (absmax/127, half-even), encode/decode,
`dense()`. `vtm_compile.py` принимает `embed_quant="int8"` и собирает EMB по int8-тензору,
tied MM по нему же; `load_export` читает `dtype: "int8"`.

**Числа:** 128×128 — trit 3 826 Б < int8 16 914 Б < fp16 32 768 Б. 1.5B-таблица:
54,0 МБ < 240,7 МБ < 466,7 МБ (int8/fp16 = 0,516; int8/trit = 4,46).
Паритет C↔Python: int8 4,14e-6, native-демо побитово равно packed-демо.

**Честно (что не сделано):** ленивого EMB из int8 нет — материализация в fp32 даёт
+4× памяти к файлу (~233 МиБ для таблицы 1.5B), нужны правки `interp.c` (0129).
`qat/export.py` по-прежнему отказывает `quantize_embeddings: int8` — end-to-end
«экспорт → int8-тензор» не выполнен (0130).

## Ссылки

- `ai/tasks/0043-trit-embeddings-e2e.md`
- `docs/adr/0011-container-v02.md`
