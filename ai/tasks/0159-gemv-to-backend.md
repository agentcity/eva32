---
task_id: "0159"
type: refactor
title: dense_gemv в backend_neon.c + AVX2 + батчевый плотный путь
created_at: 2026-11-24T19:32:00Z
author: yuri
assignee: agent:gemv-backend
status: in_progress
scope:
  docs: false
  code: true
  data: false
---

# Слои: GEMV в бэкенде, а не в интерпретаторе

## Проблема

0158 дала 3B Q4 5,3–7,7 ток/с, но NEON-intrinsics вписаны прямо в `interp.c` — слой
нарушен: `interp.c` не должен знать про NEON. Плюс на x86-64 работает только
переносимый d4 (2,6×), AVX2 для n==1 нет, и батчевый плотный путь prefill
(`batch_mm_dense_rows`) остался скалярным.

## Требования

- Перенести `dense_gemv` в `backend_neon.c` как `vtm_simd_gemv_f32()` (и AVX2-версию в
  `backend_avx2.c`), объявления — в `vtm_backend.h`; `interp.c` зовёт бэкенд.
- Батчевый плотный путь: `batch_mm_dense_rows` через GEMV/MM-векторизацию, не скаляр.
- Точность: прежняя или лучше (≤1e-5 против скаляра, побитово от потоков не зависит).
- Замер: 3B Q4 (`runs/q4/3b.vtm`, wall clock), `bench_backend` (добавить форму f32 n==1,
  чтобы правка была видна в бенче).
- Обновить устаревший комментарий в `vtm_backend.h` про «плотные — скалярный эталон».

## Критерии приёмки

- [ ] в `interp.c` нет платформенных intrinsics
- [ ] 3B Q4 не медленнее 5,3 ток/с (лучше — быстрее)
- [ ] `bench_backend` содержит f32 n==1 и показывает ускорение
- [ ] `make test`, `--check` зелёные

## Отчёт исполнителя

_Заполняется по завершении._
