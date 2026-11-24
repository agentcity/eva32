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
//   не задан        -> auto: simd, если доступен, иначе scalar-lut243;
//   auto            -> simd, если доступен, иначе scalar;
//   simd            -> simd (с внутренним откатом на scalar, если недоступен);
//   scalar | lut    -> scalar-lut243;
//   direct          -> scalar-direct.
// Неизвестное значение -> scalar-lut243 + предупреждение в stderr.
const vtm_backend *vtm_backend_from_env(void);

// --- батчевые ядра TMV/MM (задача 0122) ------------------------------------
//
// N токенов против одной матрицы весов: вес читается и распаковывается один
// раз на блок, поэтому N токенов дешевле N отдельных вызовов ядра (это и
// ускоряет батчевый prefill 0112 на SIMD). Раскладка активаций — как у
// батчевого пути interp.c (токен — младший индекс):
//
//   x[t * N + j] — вход токена j по индексу t (t = 0..cols-1 / 0..k-1);
//   y[r * N + j] — выход токена j по индексу r (r = 0..rows-1 / 0..m-1).
//
// Арифметика — как у батчевого скалярного ядра: сумма по блоку масштаба
// (128/160) копится в double и умножается на масштаб блока, внешние блоки
// складываются в double. Расхождение с батчевым скалярным путём — единицы
// ulp (проверяется tests/vtm/bench_backend.c, порог 1e-5).
//
// 0 — успех, -1 — неверные аргументы (NULL, отрицательные размеры, чужой
// scale_block). Ядра не роняют прогон: при отсутствии SIMD/собранного ядра
// диспетчер backend_neon.c зовёт скалярный откат (тот же результат по смыслу,
// медленнее), при n_tokens == 1 — обычное одноколоночное ядро.
//
// TMV: W[rows, cols] (триты pack20 + масштабы) * X[cols, N] -> Y[rows, N].
int vtm_simd_tmv_blocks_batch(float *y, const uint8_t *w, const float *scales,
                              const float *x, int rows, int cols,
                              int n_tokens, int scale_block);
// MM: A[m, k] (триты pack20 + масштабы) * B[k, N] -> Y[m, N]. Для тритовой A
// это та же арифметика, что у TMV (отдельное имя — ради читаемости вызова).
int vtm_simd_mm_batch(float *y, const uint8_t *w, const float *scales,
                      const float *b, int m, int k, int n_tokens,
                      int scale_block);

// Плотные (fp32) варианты — интерфейс батчевого пути interp.c: он батчит и
// нетернарные TMV/MM (эмбеддинги, tied lm_head). Сейчас это скалярный
// double-эталон без SIMD-ускорения: в графе Qwen-формы плотных матриц нет,
// а трогать горячий путь ради редкого случая — лишний риск.
// transposed != 0 — W хранится как [cols, rows] (VTM_FLAG_TMV_TRANSPOSED).
int vtm_simd_tmv_f32_batch(float *y, const float *w, const float *x,
                           int rows, int cols, int n_tokens, int transposed);
int vtm_simd_mm_f32_batch(float *y, const float *a, const float *b,
                          int m, int k, int n_tokens);

// --- kind 5 (Q4): ленивая деквантизация на лету (задача 0152) --------------
//
// Веса Q4 не материализуются в fp32: ядро читает упакованные байты строки,
// разворачивает их в маленький scratch-буфер (стек, O(группы) на поток) и
// сразу умножает на активации. Пиковая память ядра не зависит от размера
// матрицы: строка разворачивается по одной группе 32 значения в scratch на
// стеке (256 байт double на поток), без malloc на строку/токен.
//
// Контракт представления (согласуется с kind 5 в loader.c, задача 0151):
//   * группа VTM_Q4_GROUP_SIZE (32) значений — 16 байт, раскладка как в
//     GGML Q4_0 (импорт GGUF без перепаковки): байт j группы хранит значение
//     j в младшем ниббле и значение j+16 в старшем; значения 0..15;
//   * группы идут по сплющенной матрице, масштаб fp32 на группу:
//     scales[(r*cols + j) / 32];
//   * asym == 0 (Q4_0): w = (q - 8) * scale;
//   * asym != 0 (Q4_1): w = q * scale + min, mins — fp32 на группу
//     (обязателен при asym != 0).
//
// y[r] = sum_j w[r,j] * x[j] (TMV, x — строка длины cols);
// y[r*n + c] = sum_t w[r,t] * b[t*n + c] (MM).
//
// 0 — успех, -1 — неверные аргументы/NULL/отрицательные размеры. Ядра не
// выделяют памяти под веса и не пишут в арену; внутренний scratch —
// фиксированного размера на стеке.
#define VTM_Q4_GROUP_SIZE 32
int vtm_q4_tmv(float *y, const uint8_t *q, const float *scales,
               const float *mins, const float *x, int rows, int cols,
               int asym);
int vtm_q4_mm(float *y, const uint8_t *q, const float *scales,
              const float *mins, const float *b, int m, int k, int n,
              int asym);
// Развернуть одну строку Q4 (cols значений с глобального индекса base) в y:
// нужно EMB — таблица словаря остаётся упакованной, в арену копируется только
// выбранная строка. 0 — успех, -1 — неверные аргументы.
int vtm_q4_dequant_row(float *y, const uint8_t *q, const float *scales,
                       const float *mins, int64_t base, int cols, int asym);

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

// --- многопоточность: пул pthread (задача 0077) ----------------------------
//
// Ядра TMV/MM/RMSN делят работу по строкам выхода. Пул живёт в
// backend_scalar.c (этот TU собирается всегда), ядра SIMD-бэкендов зовут
// те же vtm_threads()/vtm_parallel_rows().
//
// Число потоков N = VTM_THREADS (1..VTM_THREADS_MAX); переменная не задана —
// min(онлайн-ядер, VTM_THREADS_MAX). N == 1 — ровно прежнее поведение:
// vtm_parallel_rows() вызывает колбэк в текущем потоке, пул не создаётся.
// Значение запоминается при первом обращении (vtm_threads()).
#define VTM_THREADS_MAX 8
// Меньше этой оценки работы (в MAC) параллелить невыгодно: оверхед
// пробуждения потоков больше выигрыша. Решение принимает
// vtm_parallel_rows(), порог — по всей задаче.
#define VTM_PAR_MIN_WORK 131072.0
// RMSN параллелится только на больших n: у него два прохода (сумма
// квадратов и масштабирование), и на n порядка скрытого размера модели
// пробуждение потоков дороже счёта. Порог — 512K элементов: на нём scalar
// получает ~3.5x, SIMD ~2x на 4 потоках, меньшие n считаются как раньше.
#define VTM_PAR_RMSN_MIN 524288

// Эффективное число потоков (1..VTM_THREADS_MAX). VTM_THREADS=1 — без
// потоков; 0/мусор в переменной — значение по умолчанию и предупреждение.
int vtm_threads(void);

// Переопределить число потоков в рантайме (тесты, хост). n <= 0 — вернуть
// значение из env. Возвращает эффективное число потоков. Потоки создаются
// лениво, при первом большом вызове; вызывать до параллельных ядер, не
// одновременно из разных потоков (переопределение — не горячий путь).
int vtm_threads_set(int n);

// Колбэк строчного ядра: посчитать строки [r0, r1) выхода. tid — номер
// логического потока в [0, число потоков), по нему ядро берёт свою
// scratch-память. Диапазоны строк не пересекаются, поэтому результат не
// зависит от разбиения: каждая строка считается тем же кодом в том же
// порядке, что и при одном потоке (побитово).
typedef void (*vtm_rows_fn)(void *ctx, int tid, int r0, int r1);

// Разбить [0, n) на N непрерывных диапазонов и выполнить fn в пуле
// (вызвавший считает диапазон tid 0). Строки ядер стоят одинаково, поэтому
// статического деления достаточно и мьютекс на строку не нужен. Если N < 2,
// n < 2, работа work < VTM_PAR_MIN_WORK или потоки не создались —
// fn(ctx, 0, 0, n) вызывается в текущем потоке (прежнее поведение). Вызовы
// из разных потоков сериализуются.
void vtm_parallel_rows(int n, double work, vtm_rows_fn fn, void *ctx);

#ifdef __cplusplus
}
#endif

#endif // VTM_BACKEND_H
