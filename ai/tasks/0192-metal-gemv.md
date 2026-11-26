---
task_id: "0192"
type: feature
title: Metal-бэкенд, фаза 1 — GEMV Q4_K/Q6_K и lm_head на GPU
created_at: 2026-11-26T03:45:00Z
author: yuri
assignee: agent:metal
status: in_progress
scope: {docs: true, code: true, data: false}
---

# GPU вместо CPU на тяжёлых матрицах

## Контекст
ADR 0021. Замеры 0189: наш CPU — 18,1 ток/с prefill / 12,8 декод; ollama на
Metal — 182–393 / 56,5. Разрыв — в GPU.

## Фаза 1 (эта задача)
1. **Metal-бэкенд** как отдельный исполнитель рядом с scalar/neon:
   `src/vtm/backend_metal.m` (Objective-C++/Metal) + MSL-ядра в `src/vtm/*.metal`
   (или строками в .m, если сборка проще).
2. **Zero-copy**: `MTLBuffer` через `newBufferWithBytesNoCopy` на mmap-регион
   контейнера (выравнивание страницы); веса НЕ копируются. Проверить
   `MTLDevice.currentAllocatedSize` — не должен расти на размер модели.
3. **Ядра**: GEMV для kind 6 (Q4_K) и kind 7 (Q6_K), int8-таблица lm_head
   (kind 4). Раскладка та же, что в NEON: блок 256 значений, под-блоки, d/dmin fp16,
   6-битные масштабы; для Q6_K — ql/qh.
4. **Диспетчер**: `VTM_BACKEND=metal|simd|scalar` + авто-выбор (Metal есть → metal,
   иначе simd). Откат обязателен: без Metal всё работает как сейчас.
5. **Питон/сборка**: `make` собирает `.metal` в `default.metallib` (xcrun metal)
   или компилирует рантайм-исходник; если Metal SDK недоступен — сборка без бэкенда.

## Критерии приёмки
- [ ] `VTM_BACKEND=metal` на 3B Q4_K: **декод ≥30 ток/с** (сейчас 12,8)
- [ ] argmax совпадает с NEON на 3 промптах, Δlogits ≤1e-2
- [ ] footprint GPU-буферов ≈ 0 (zero-copy подтверждён замером)
- [ ] без Metal — прежние числа, тесты зелёные
- [ ] `make test`, `--check`, полный набор, TSan

## Отчёт исполнителя

_Заполняется по завершении._
