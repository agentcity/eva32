// VTM — SIMD-бэкенд: объявления (без определений).
//
// Здесь только декларации: реализация ядер — src/vtm/backend_neon.c (NEON,
// ARM64; на других арх — только диспетчер с откатом на scalar) и
// src/vtm/backend_avx2.c (AVX2, x86-64; собирается Makefile по uname -m).
//
// Контракт `vtm_backend` (см. vtm.h/docs/vtm.md): любое поле NULL —
// интерпретатор берёт скалярную реализацию. `vtm_backend_simd` никогда не
// роняет прогон: если SIMD в сборке нет или CPU его не умеет, его ядра сами
// вызывают скалярные (см. vtm_simd_available()).
#ifndef VTM_BACKEND_H
#define VTM_BACKEND_H

#include "vtm.h"

#ifdef __cplusplus
extern "C" {
#endif

// Готовый SIMD-бэкенд. Имя: "simd-neon", "simd-avx2" или "simd" (сборка без
// SIMD; ядра при этом работают как скалярные).
extern const vtm_backend vtm_backend_simd;

// 1 — в сборке есть SIMD-ядра и CPU их поддерживает; 0 — будет откат на
// скалярные ядра (vtm_backend_simd при этом всё равно корректен).
int vtm_simd_available(void);

// "neon" | "avx2" | "none" — что реально используется (для логов/бенчмарка).
const char *vtm_simd_isa(void);

// Выбор бэкенда по имени: "auto" | "simd" | "scalar" | "direct" | "lut"
// (NULL/"" — то же, что VTM_BACKEND из окружения). Неизвестное имя — NULL.
const vtm_backend *vtm_backend_by_name(const char *name);

// Бэкенд из env VTM_BACKEND:
//   не задан        -> scalar-lut243 (поведение по умолчанию не меняется);
//   auto            -> simd, если доступен, иначе scalar;
//   simd            -> simd (с внутренним откатом на scalar, если недоступен);
//   scalar | lut    -> scalar-lut243;
//   direct          -> scalar-direct.
// Неизвестное значение — как «не задан» (scalar) + предупреждение в stderr.
const vtm_backend *vtm_backend_from_env(void);

// --- MM: SIMD-ядра вне `vtm_backend` ---------------------------------------
//
// В `vtm_backend` (vtm.h) поля MM нет, а op_mm() в interp.c сейчас считает
// MM скалярно, поэтому ядра MM доступны отдельными функциями: их можно
// подключить к interp.c одной строкой, когда это будет позволено зоной
// (см. docs/vtm-perf.md, «Что не сделано»).
//
//   y[r*n + c] = sum_t A[r*k + t] * B[t*n + c]
//
// A — тритовая матрица (слова pack20 + масштабы, блок 128/160; точно как у
// TMV) либо плотная fp32. 0 — успех, -1 — неверные аргументы. При отсутствии
// SIMD работает скалярный откат (результат тот же по смыслу, медленнее).
int vtm_simd_mm_trit(float *y, const uint8_t *w, const float *scales,
                     const float *b, int m, int k, int n, int scale_block);
int vtm_simd_mm_f32(float *y, const float *a, const float *b,
                    int m, int k, int n);

#ifdef __cplusplus
}
#endif

#endif // VTM_BACKEND_H
