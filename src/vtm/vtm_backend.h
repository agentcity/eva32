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
// нетернарные TMV/MM (эмбеддинги, tied lm_head). Ядро векторизует по токенам
// пачки: x[t*N + j], y[r*N + j] (токен — младший индекс), вес W[r,t] — скаляр,
// поэтому на один t уходит broadcast и N/4 FMA. По t копится блок 64 и
// сворачивается в double — так же, как у GEMV ниже, чтобы f32-ошибка не росла
// на длинной строке. Это не «скалярный эталон»: у 3B Q4 tied lm_head —
// плотная [151936,2048], и в prefill именно этот путь был самым дорогим
// (задача 0159).
//
// transposed != 0 — W хранится как [cols, rows] (VTM_FLAG_TMV_TRANSPOSED).
// При n_tokens == 1 зовётся vtm_simd_gemv_f32 (та же арифметика).
int vtm_simd_tmv_f32_batch(float *y, const float *w, const float *x,
                           int rows, int cols, int n_tokens, int transposed);
int vtm_simd_mm_f32_batch(float *y, const float *a, const float *b,
                          int m, int k, int n_tokens);

// --- плотный GEMV: один столбец (n == 1), задача 0159 -----------------------
//
//   y[r] = sum_t W[r,t]·x[t]        (transposed == 0, W — [rows, cols])
//   y[r] = sum_t W[t,r]·x[t]        (transposed != 0, W — [cols, rows])
//
// Горячий случай — tied lm_head (3B Q4: 311М MAC, ~треть времени декода):
// у vtm_simd_mm_f32 нет векторизации при n == 1 (цикл по 4 столбца не
// выполняется ни разу), поэтому один столбец считает отдельное ядро.
// Внутри — 4 f32-ленты по блоку 64 элемента, блок сворачивается в double.
// Порядок сложений иной, чем у скалярного double-цикла, поэтому результат не
// побитовый: расхождение на logits 3B Q4 ≤ 2e-6 (порог 1e-5); от числа
// потоков не зависит (строки делятся пулом, каждая считается тем же кодом).
//
// 0 — успех, -1 — неверные аргументы (NULL, отрицательные размеры).
int vtm_simd_gemv_f32(float *y, const float *w, const float *x,
                      int rows, int cols, int transposed);

// Переносимые скалярные откаты тех же ядер (backend_scalar.c; SIMD не нужен).
//
// vtm_scalar_gemv_f32 — откат vtm_simd_gemv_f32, когда SIMD в сборке нет или
// CPU его не умеет: 4 double-аккумулятора вместо одной цепи FMA, поэтому
// расхождение с скалярным double-циклом — единицы ulp.
//
// vtm_scalar_dense_batch — эталон батчевого плотного TMV/MM (double, порядок
// сложений — по t). Его зовёт interp.c для скалярных бэкендов: там побитовость
// с прогоном по токену — контракт (это прежний цикл interp.c, перенесённый в
// бэкенд). 0 — успех, -1 — неверные аргументы.
int vtm_scalar_gemv_f32(float *y, const float *w, const float *x,
                        int rows, int cols, int transposed);
int vtm_scalar_dense_batch(float *y, const float *w, const float *x,
                           int rows, int cols, int n_tokens, int transposed);

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

// Батчевые Q4 TMV/MM (задача 0162): N токенов против одной Q4-матрицы.
//
// Раскладка — как у прочих батчевых ядер (см. vtm_simd_*_batch выше):
//   x[t * N + j] — вход токена j по индексу t (t = 0..cols-1 / 0..k-1);
//   y[r * N + j] — выход токена j по индексу r.
//
// Строка W разворачивается из нибблов один раз на пачку (а не N раз), поэтому
// N токенов prefill'а дешевле N отдельных вызовов vtm_q4_tmv: вес читается из
// памяти один раз. Память ядра — буфер строки на поток (cols f32), без копии
// всей матрицы.
//
// Арифметика — как у NEON-ядра 0157: целые s = q - 8 (в f32 точно), 4
// независимых f32-аккумулятора по t внутри группы 32, свёртка и масштаб
// группы — в double. Расхождение с поштучным vtm_q4_* — единицы ulp на
// группу (на logits 3B Q4 — доли 1e-6, порог 1e-5; проверяет
// tests/vtm/bench_backend.c). При N == 1 уходит в vtm_q4_tmv/vtm_q4_mm.
//
// 0 — успех, -1 — неверные аргументы/NULL/отрицательные размеры/нет памяти.
// На -1 вызывающий обязан откатиться на поштучный путь: так же, как
// vtm_q4_tmv на нехватку памяти не роняет прогон.
int vtm_q4_tmv_batch(float *y, const uint8_t *q, const float *scales,
                     const float *mins, const float *x, int rows, int cols,
                     int n_tokens, int asym);
int vtm_q4_mm_batch(float *y, const uint8_t *q, const float *scales,
                    const float *mins, const float *b, int m, int k,
                    int n_tokens, int asym);

// --- MM: SIMD-ядра вне `vtm_backend` ---------------------------------------
//
// В `vtm_backend` (vtm.h) поля MM нет, поэтому ядра MM — отдельные функции.
// op_mm() в interp.c зовёт их при SIMD-бэкенде: тритовую A — vtm_simd_mm_trit,
// плотную A — vtm_simd_mm_f32, а при n == 1 (один столбец) — vtm_simd_gemv_f32.
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
