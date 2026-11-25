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
