---
task_id: "0176"
type: feature
title: Gated Delta linear attention — ключ к линейке Qwen3.5
created_at: 2026-11-25T13:05:00Z
author: yuri
assignee: agent:gated-delta
status: in_progress
scope: {docs: true, code: true, data: false}
---

# Gated Delta (linear attention) в ВТМ

## Зачем
Вся линейка Qwen3.5 — гибрид: 3 слоя из 4 на `linear_attention` (Gated Delta Net),
каждый 4-й — обычное внимание. Без этой механики недоступны 9b/27b/35b, а 27b
(17–20 ГБ) всё равно не влезает в 16 ГиБ — значит цель **qwen3.5:9b** (6,6–7,6 ГБ).

## Параметры из config.json Qwen3.5-35B-A3B (аналогично у 9b)
- `layer_types`: 30 × linear_attention + 10 × full_attention (каждый 4-й полный);
- `linear_conv_kernel_dim: 4` — короткая свёртка по q/k/v перед правилом;
- `linear_num_key_heads: 16`, `linear_num_value_heads: 32`,
  `linear_key_head_dim: 128`, `linear_value_head_dim: 128`;
- `mamba_ssm_dtype: float32` — состояние в f32;
- полное внимание: `head_dim 256`, 16 голов, 2 KV, `partial_rotary_factor 0.25`,
  mRoPE interleaved section [11,11,10], theta 1e7;
- MoE: 256 экспертов, top-8, `moe_intermediate 512`, shared expert 512, vocab 248320.

## Требования (этап 1 — эталон и C)
1. **Python-эталон** в `tools/vtm_ref.py`: операция `GDELTA` (обновление состояния по
   правилу gated delta + выходной гейт) и `CONV4` (depthwise kernel 4) — по формуле
   из статьи Gated Delta Networks (используй публичное описание; формулы проверить
   на маленьком синтетическом примере с ручным расчётом).
2. **C**: те же операции в `interp.c` (scalar double) + состояние в отдельном буфере
   контекста (по образцу KV), `--trace-summary` учитывает.
3. **Тест паритета** Python↔C на случайных тензорах ≤1e-6.
4. **Не ломать** существующие модели: новые опкоды, старые программы не меняются.

## Критерии приёмки
- [ ] Python-эталон GDELTA/CONV4 с ручной проверкой на малом примере
- [ ] C-реализация, паритет ≤1e-6
- [ ] состояние не течёт между сессиями (reset работает)
- [ ] `make test`, полный набор, docs зелёные

## Отчёт исполнителя

_Заполняется по завершении._

## Перезапуск 2026-11-25 (этап 1a: только Python)

Первый исполнитель упал, оставив объявления опкодов в `vtm.h` и правки в `interp.c`.
C-часть **откладывается**: сейчас `interp.c` держит агент 0181 (свип MoE-пачки),
и параллельная правка снова сломает сборку.

Этап 1a: только **Python-эталон** `GDELTA`/`CONV4` в `tools/vtm_ref.py`,
проверка формул на ручном примере, `docs/qwen35.md`. C-часть — следующей задачей
после 0181.

## Этап 1b (запуск 2026-11-25): C-реализация по готовому эталону

Python-эталон принят (`tools/vtm_ref.py`, 17 тестов, ручной счёт). Теперь C:
- `interp.c`: `op_conv4`/`op_gdelta` (scalar double, запись f32), состояние в контексте
  рядом с KV; банк 6 тензоров `c..c+5` (state, v, beta, g, z, norm_w);
- `loader.c`: буфер STATE по флагу `TENSOR_FLAG_STATE = 0x0008` (биты 1..2 — хранение
  тритов!), `offset == KV_NO_OFFSET`, `vtm_reset()` обнуляет;
- `docs/vtm.md`: добавить 0x12/0x13 (check_vtm_docs сейчас красный из-за этого);
- тест паритета Python↔C ≤1e-6, состояние округлять до f32 на каждом шаге.

Компилятор/импортёр Qwen3.5 (conv1d, A_log, dt_bias, beta, mRoPE, MoE) — после C.

## Отчёт исполнителя: этап 1b — C-реализация (2026-11-25)

### Что сделано

* `src/vtm/interp.c` — `op_conv4` (0x12) и `op_gdelta` (0x13), scalar double,
  запись в f32; состояние — буфер в арене рядом с KV-кэшем, f32 на каждом шаге.
  GDELTA считает по банку `c..c+5`, группы value-голов идут по key-голове
  (`kh = h / (Hv/Hk)`), `Dk`/`Dv` берутся из форм q/v. Оба опкода добавлены в
  `vtm_run()` и `batch_op_one()`; в батчевом prefill они считаются по токену в
  порядке пачки (состояние сквозное), остальные инструкции — батчем.
* `src/vtm/loader.c` — `vtm_reset()` обнуляет тензоры с флагом
  `VTM_TENSOR_FLAG_STATE` (0x0008); биты 1..2 (хранение тритов) не тронуты.
  STATE идёт тем же путём, что KV: `offset == KV_NO_OFFSET`, payload не
  читается, буфер `n_elements×4` в арене, общий кэп `VTM_MAX_KV_BYTES`.
  CONV4/GDELTA добавлены в tripwire допустимых опкодов загрузчика.
* `src/vtm/vtm.h` — константы ABI `VTM_CONV4_KERNEL=4`, `VTM_GDELTA_BANK_SIZE=6`
  (аддитивно) и уточнение контракта `vtm_run_batch()` про сквозное состояние.
* `src/vtm/interp.c` — кэш префикса (`vtm_cache_prefix()`/`vtm_run_cached()`)
  для графа со STATE отключён (`program_has_state()` в `cache_usable()`):
  снимок 0081 хранит K/V и позицию, но не состояние, и попадание дало бы
  чужое S. Тест проверяет, что `vtm_cache_prefix_len() == 0`, а
  `vtm_run_cached()` всё равно даёт те же y/S, что полный прогон.
* `src/vtm/main.c` — правок не потребовалось: `--trace-summary` уже считает по
  общему callback'у 0..255, а `vtm_op_name()` знает CONV4/GDELTA. Проверено
  запуском CLI (см. ниже).
* `docs/vtm.md` — строки 0x12/0x13 в таблице операций и полей, бит 3 `STATE` в
  описании `flags`, абзац о состояниях в разделе «KV-кэши: не в файле».
  `tools/check_vtm_docs.py` зелёный: docs=19, vtm.h=19.
* `tests/vtm/test_vtm.c` — C-тесты: ручные числа CONV4 (4321/2222 и состояние
  после каждого шага), ручные числа GDELTA (3 шага), две value-головы, L2,
  затухание, q=0, in-place `dst == x`, обрезанный банк, отрицательный eps,
  отказ триту с битом STATE, `vtm_reset()`, батч против поштучного, трасса,
  выключение кэша префикса для графа со STATE.
* `tests/vtm/test_gated_delta_parity.py` — паритет Python↔C: контейнер строит
  `tools/vtm_ref.py`, C-проба линкуется с `libvtm.a` и печатает `y`/состояние
  после каждого шага; порог 1e-6. Добавлен `tests/vtm/__init__.py`, чтобы
  `discover -s tests` видел этот тест.

### Проверки

```
$ cd src/vtm && make test
...
  -- CONV4: ручные числа, состояние f32, reset и трасса
  -- GDELTA: ручные числа, пачка, reset и трасса
  -- GDELTA: две value-головы, L2-нормировка, затухание, q=0
  -- GDELTA: обрезанный банк, eps, бит STATE у тритов
все тесты VTM: ok
./bench_backend --check                 # без расхождений
./bench_backend --threads --iters 3     # потоки: ok

$ python3 -m unittest discover -s tests/vtm -p 'test_gated_delta_parity.py' -v
Ran 5 tests ... OK

$ python3 tools/check_vtm_docs.py
docs/vtm.md: опкодов 19
OK: документация совпадает с кодом

$ ./src/vtm/vtm run /tmp/gd_cli.vtm --tokens 0,1,2 --max-new 0 --trace-summary
trace-summary: инструкций=21, ...
опкод   вызовов     время, мс      доля    выход, КиБ
EMB           18     ...
GDELTA         3     ...
итого         21     ...
```

Паритет: на случайных тензорах CONV4 (width 5) и GDELTA (Hk=2/Hv=4,
Dk=3/Dv=5, 5 токенов, eps 0 и 1e-3), а также на форме голов Qwen3.5
(Hk=16/Hv=32, Dk=Dv=4) `max|Δ| = 0` после f32-округления — C и Python
совпали побитово. Ручные числа GDELTA из `tests/test_gated_delta.py`
(`0.4623620423 …`, `2.5/2.0`, `1.75/1.5`) сходятся в пределах 1e-6.

`python3 -m unittest discover -s tests`: `Ran 2436 tests ... OK (skipped=62)`.
Паритетные тесты входят в общий прогон через `tests/vtm/__init__.py`; прогон
запускался после сборки `make test`, чтобы C-проба линковалась с актуальной
`libvtm.a`.

### Что осталось до импорта Qwen3.5

1. **Компилятор/импортёр** (следующая задача): `linear_conv1d.weight` формы
   `[width, 1, 4]` → `w [width, 4]`; `A_log`/`dt_bias` → `g = −exp(A_log) ·
   softplus(a + dt_bias)`; `beta = sigmoid(b)`; банк шести тензоров; STATE в
   писателе контейнера; `layer_types`, mRoPE, MoE (0172).
2. **Кэш префикса**: для графов со STATE он сейчас **выключен** (иначе
   попадание восстановило бы K/V без состояния — молча неверный хвост).
   Это корректно, но теряет ускорение повторного префикса; полноценный снимок
   состояний (для 9b ~60 МиБ) — отдельная задача. В отчёте это риск, а не
   выполненный критерий.
3. **Скорость**: scalar double, `malloc` под аккумулятор на слой/токен;
   ~47 млн MAC на токен для 30 линейных слоёв. Для инференса нужен SIMD/батч,
   но состояние ограничивает батчирование по токенам.
4. **f32-состояние на длинном контексте**: ошибка накапливается; нужен
   паритет с HF на реальных весах/длинах, а не только с эталоном.
5. **Параметризация `g` и выходной гейт** (silu vs sigmoid) не сверены с
   настоящим Qwen3.5 — нет весов/сети; меняется хвост операции, не
   рекуррентность.
6. `docs/qwen35.md` всё ещё пишет «C-часть отложена» — файл вне зоны 0176,
   обновить в задаче импортёра.
