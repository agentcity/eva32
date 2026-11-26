// SIMD-ядра VTM: NEON (ARM64) + диспетчер бэкенда и скалярные откаты.
//
// Что здесь:
//   * LUT на 8 тритов (3^8 = 6561 запись по 8 байт = 52 КБ): слово pack20
//     раскладывается двумя делениями на 6561, а не четырьмя на 243, и
//     пишется в int8-буфер тремя 64-битными mov'ами (без веток на элемент);
//   * TMV: 128/160 блоки масштаба, распаковка слова pack20 в int8-буфер,
//     затем NEON считает 16 тритов за шаг (int8 -> int16 -> int32 -> f32 FMA).
//     Блок копится в 4 независимых f32-аккумуляторах, свёртка и масштаб —
//     в double, как в скаляре;
//   * RMSN: сумма квадратов в double (float64x2), поэлементная часть — f32;
//   * MM (trit и f32) — отдельными функциями vtm_simd_mm_*(), потому что в
//     vtm_backend поля MM нет (см. vtm_backend.h);
//   * выбор бэкенда: vtm_backend_by_name()/vtm_backend_from_env().
// На x86-64 ядра живут в backend_avx2.c; здесь остаётся диспетчер и откат.
//
// Почему не «просто распаковать в float и умножить»: распаковка — половина
// времени ядра, а делений на 3 на элемент (direct-путь) хватает, чтобы
// проиграть скаляру. LUT8 + int8-буфер дают ~1.2 инструкции на MAC.
#include "vtm.h"
#include "vtm_backend.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Metal-бэкенд (0192) собирается только на Darwin (backend_metal.m). Без него
// вызовы в выборе бэкенда — no-op, чтобы диспетчер собирался на любой ОС.
#ifndef VTM_WITH_METAL
#define VTM_WITH_METAL 0
#endif
#if !VTM_WITH_METAL
#define vtm_metal_set_active(on) ((void)(on))
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
// FEAT_DotProd (int8-dot, задача 0163): на Apple спрашиваем sysctl, на Linux —
// HWCAP_ASIMDDP. Обе ветки нужны только для рантайм-детекта SDOT.
#if defined(__APPLE__)
// <sys/sysctl.h> под -D_POSIX_C_SOURCE=200809L не отдаёт BSD-типы (u_int),
// поэтому объявляем саму функцию: символ есть в libSystem, нужен один вызов.
extern int sysctlbyname(const char *name, void *oldp, size_t *oldlenp,
                        void *newp, size_t newlen);
#elif defined(__linux__)
#include <sys/auxv.h>
#endif
#define VTM_SIMD_NEON 1
#else
#define VTM_SIMD_NEON 0
#endif

// Ядра AVX2: сборка backend_avx2.c. Если Makefile собрал библиотеку без неё
// (WITH_AVX2=0), он же определяет VTM_NO_AVX2, и здесь остаётся только
// диспетчер с откатом на scalar — без висячих ссылок.
#if defined(__x86_64__) && !defined(VTM_NO_AVX2)
extern int vtm_avx2_available(void);
extern int vtm_avx2_tmv_blocks(float *y, const uint8_t *w,
                               const float *scales, const float *x,
                               int rows, int cols, int scale_block);
extern int vtm_avx2_rmsnorm(float *y, const float *x, const float *w,
                            int n, float eps);
extern int vtm_avx2_mm_trit(float *y, const uint8_t *w, const float *scales,
                            const float *b, int m, int k, int n,
                            int scale_block);
extern int vtm_avx2_mm_f32(float *y, const float *a, const float *b,
                           int m, int k, int n);
extern int vtm_avx2_tmv_blocks_batch(float *y, const uint8_t *w,
                                     const float *scales, const float *x,
                                     int rows, int cols, int n_tokens,
                                     int scale_block);
extern int vtm_avx2_mm_batch(float *y, const uint8_t *w, const float *scales,
                             const float *b, int m, int k, int n_tokens,
                             int scale_block);
extern int vtm_avx2_gemv_f32(float *y, const float *w, const float *x,
                             int rows, int cols, int transposed);
extern int vtm_avx2_tmv_f32_batch(float *y, const float *w, const float *x,
                                  int rows, int cols, int n_tokens,
                                  int transposed);
extern int vtm_avx2_mm_f32_batch(float *y, const float *a, const float *b,
                                 int m, int k, int n_tokens);
// int8-dot VNNI (задача 0163): ядро Q4-батча и рантайм-детект AVX-512 VNNI.
extern int vtm_avx2_q4_batch_vnni(float *y, const uint8_t *q,
                                  const float *scales, const float *mins,
                                  const float *x, int rows, int cols,
                                  int ntok, int asym, int mode);
extern int vtm_avx2_vnni_available(void);
#define VTM_SIMD_X86 1
#else
#define VTM_SIMD_X86 0
#endif

// Слова pack20 и масштабы лежат в TRT1-сегментах контейнера со смещением
// 2 mod 4 (задача 0060): разыменование uint32_t*/float* по такому адресу —
// UB, даже если CPU его терпит. Читаем через memcpy: компилятор сворачивает
// байтовое копирование в одну обычную загрузку.
static inline uint32_t load_u32_unaligned(const uint32_t *words, int64_t i) {
    uint32_t v;
    memcpy(&v, (const uint8_t *)words + (size_t)i * sizeof v, sizeof v);
    return v;
}

static inline float load_f32_unaligned(const float *scales, int64_t i) {
    float v;
    memcpy(&v, (const uint8_t *)scales + (size_t)i * sizeof v, sizeof v);
    return v;
}

// --- скалярные откаты для MM -----------------------------------------------
//
// В interp.c MM считается скалярно и к бэкенду не обращается, поэтому
// скалярное ядро MM держим здесь: vtm_simd_mm_* должны работать всегда, в
// том числе на сборке без SIMD.

typedef struct {
    float *y;
    const uint32_t *words;
    const float *scales;
    const float *b;
    int k;
    int n;
    int scale_block;
} mm_scalar_job;

static void mm_trit_scalar_rows(mm_scalar_job *j, int r0, int r1) {
    const uint32_t *words = j->words;
    const float *scales = j->scales;
    const float *b = j->b;
    const int k = j->k;
    const int n = j->n;
    const int scale_block = j->scale_block;
    float *y = j->y;
    int r, c;

    for (r = r0; r < r1; r++) {
        for (c = 0; c < n; c++) {
            int64_t base = (int64_t)r * (int64_t)k;
            double acc = 0.0;
            double block_acc = 0.0;
            int64_t cur_block = -1;
            int64_t cur_word = -1;
            int8_t trit[20];
            int64_t idx;
            for (idx = 0; idx < k; idx++) {
                int64_t abs_idx = base + idx;
                int64_t blk = abs_idx / (int64_t)scale_block;
                int64_t wi = abs_idx / VTM_TRITS_PER_WORD;
                if (blk != cur_block) {
                    if (cur_block >= 0)
                        acc += block_acc
                             * (double)load_f32_unaligned(scales, cur_block);
                    cur_block = blk;
                    block_acc = 0.0;
                }
                // Строка может начинаться не с границы слова pack20:
                // слово берём по абсолютному индексу и перечитываем при
                // смене слова, а не по idx % 20.
                if (wi != cur_word) {
                    vtm_trit_unpack20(load_u32_unaligned(words, wi), trit);
                    cur_word = wi;
                }
                block_acc += (double)trit[abs_idx % VTM_TRITS_PER_WORD]
                           * (double)b[(size_t)idx * (size_t)n + (size_t)c];
            }
            if (cur_block >= 0)
                acc += block_acc
                     * (double)load_f32_unaligned(scales, cur_block);
            y[(size_t)r * (size_t)n + (size_t)c] = (float)acc;
        }
    }
}

static void mm_trit_scalar_worker(void *p, int tid, int r0, int r1) {
    (void)tid;
    mm_trit_scalar_rows((mm_scalar_job *)p, r0, r1);
}

static int mm_trit_scalar(float *y, const uint8_t *w, const float *scales,
                          const float *b, int m, int k, int n,
                          int scale_block) {
    mm_scalar_job job;
    if (!y || !w || !scales || !b) return -1;
    if (m < 0 || k < 0 || n < 0) return -1;
    if (scale_block != VTM_SCALE_BLOCK_128 &&
        scale_block != VTM_SCALE_BLOCK_160) return -1;
    if (m == 0 || n == 0) return 0;
    if (k == 0) {
        memset(y, 0, (size_t)m * (size_t)n * sizeof(float));
        return 0;
    }
    job.y = y;
    job.words = (const uint32_t *)w;
    job.scales = scales;
    job.b = b;
    job.k = k;
    job.n = n;
    job.scale_block = scale_block;
    vtm_parallel_rows(m, (double)m * (double)k * (double)n,
                      mm_trit_scalar_worker, &job);
    return 0;
}

static void mm_f32_scalar_rows(const float *a, const float *b, float *y,
                               int k, int n, int r0, int r1) {
    int r, c, t;
    for (r = r0; r < r1; r++) {
        for (c = 0; c < n; c++) {
            double acc = 0.0;
            for (t = 0; t < k; t++)
                acc += (double)a[(size_t)r * (size_t)k + (size_t)t]
                     * (double)b[(size_t)t * (size_t)n + (size_t)c];
            y[(size_t)r * (size_t)n + (size_t)c] = (float)acc;
        }
    }
}

typedef struct {
    float *y;
    const float *a;
    const float *b;
    int k;
    int n;
} mm_f32_scalar_job;

static void mm_f32_scalar_worker(void *p, int tid, int r0, int r1) {
    mm_f32_scalar_job *j = (mm_f32_scalar_job *)p;
    (void)tid;
    mm_f32_scalar_rows(j->a, j->b, j->y, j->k, j->n, r0, r1);
}

static int mm_f32_scalar(float *y, const float *a, const float *b,
                         int m, int k, int n) {
    mm_f32_scalar_job job;
    if (!y || !a || !b) return -1;
    if (m < 0 || k < 0 || n < 0) return -1;
    if (m == 0 || n == 0) return 0;
    if (k == 0) {
        memset(y, 0, (size_t)m * (size_t)n * sizeof(float));
        return 0;
    }
    job.y = y;
    job.a = a;
    job.b = b;
    job.k = k;
    job.n = n;
    vtm_parallel_rows(m, (double)m * (double)k * (double)n,
                      mm_f32_scalar_worker, &job);
    return 0;
}

// --- батчевые скалярные откаты TMV/MM (задача 0122) -----------------------
//
// Батчевые ядра вне vtm_backend (см. vtm_backend.h): N токенов против одной
// матрицы весов. Здесь — откат, который работает на любой сборке (в том числе
// без SIMD) и повторяет арифметику батчевого скалярного пути interp.c:
// сумма по блоку масштаба копится в double, затем умножается на масштаб
// блока. Поэтому расхождение с interp.c — единицы ulp, а не «похожий
// результат».
//
// Раскладка активаций: x[t*N + j], y[r*N + j] (токен — младший индекс).
// Строки выхода делятся между потоками пула: каждая строка считается тем же
// кодом в том же порядке, поэтому от числа потоков результат не зависит.

typedef struct {
    float *y;
    const uint32_t *words;
    const float *scales;
    const float *x;
    int cols;
    int ntok;
    int scale_block;
    double *acc_all;   // vt * ntok
} batch_scalar_job;

// Триты одной строки [r0, r1) против ntok столбцов активаций.
static void trit_batch_scalar_rows(batch_scalar_job *j, int tid, int r0,
                                   int r1) {
    const int cols = j->cols;
    const int ntok = j->ntok;
    const int64_t sb = (int64_t)j->scale_block;
    const uint32_t *words = j->words;
    const float *scales = j->scales;
    const float *x = j->x;
    double *acc = j->acc_all + (size_t)tid * (size_t)ntok;
    int8_t buf[VTM_SCALE_BLOCK_160 + 64];
    int8_t *tp = buf + 32;
    int r, jj;

    for (r = r0; r < r1; r++) {
        const int64_t base = (int64_t)r * (int64_t)cols;
        const int64_t end = base + (int64_t)cols;
        int64_t s;

        memset(acc, 0, (size_t)ntok * sizeof(double));
        for (s = (base / sb) * sb; s < end; s += sb) {
            int64_t e = s + sb;
            int64_t st;
            int64_t wi, wi_end;
            int len, t;
            int64_t lo;
            double scale;

            if (e > end) e = end;
            st = s < base ? base : s;
            wi = st / VTM_TRITS_PER_WORD;
            wi_end = (e - 1) / VTM_TRITS_PER_WORD;
            for (; wi <= wi_end; wi++)
                vtm_trit_unpack20(load_u32_unaligned(words, wi),
                                  tp + (ptrdiff_t)(wi * VTM_TRITS_PER_WORD
                                                   - st));
            len = (int)(e - st);
            lo = st - base;
            scale = (double)load_f32_unaligned(scales, s / sb);
            for (jj = 0; jj < ntok; jj++) {
                double part = 0.0;
                for (t = 0; t < len; t++)
                    part += (double)tp[t]
                          * (double)x[(size_t)(lo + t) * (size_t)ntok
                                      + (size_t)jj];
                acc[jj] += part * scale;
            }
        }
        for (jj = 0; jj < ntok; jj++)
            j->y[(size_t)r * (size_t)ntok + (size_t)jj] = (float)acc[jj];
    }
}

static void trit_batch_scalar_worker(void *p, int tid, int r0, int r1) {
    trit_batch_scalar_rows((batch_scalar_job *)p, tid, r0, r1);
}

static int trit_batch_scalar(float *y, const uint8_t *w, const float *scales,
                             const float *x, int rows, int cols, int ntok,
                             int scale_block) {
    batch_scalar_job job;
    int vt;

    if (!y || !w || !scales || !x) return -1;
    if (rows < 0 || cols < 0 || ntok < 0) return -1;
    if (scale_block != (int)VTM_SCALE_BLOCK_128 &&
        scale_block != (int)VTM_SCALE_BLOCK_160) return -1;
    if (rows == 0 || cols == 0 || ntok == 0) return 0;
    vt = vtm_threads();
    job.acc_all = (double *)calloc((size_t)vt * (size_t)ntok,
                                   sizeof(double));
    if (!job.acc_all) return -1;
    job.y = y;
    job.words = (const uint32_t *)w;
    job.scales = scales;
    job.x = x;
    job.cols = cols;
    job.ntok = ntok;
    job.scale_block = scale_block;
    vtm_parallel_rows(rows, (double)rows * (double)cols * (double)ntok,
                      trit_batch_scalar_worker, &job);
    free(job.acc_all);
    return 0;
}

// Плотный батчевый TMV/MM (dense_batch/…) переехал в backend_scalar.c:
// vtm_scalar_dense_batch — эталон, его же зовёт interp.c для скалярных
// бэкендов (задача 0159). SIMD-ядра плотного батча — ниже, в секции NEON, и
// в backend_avx2.c.

// --- NEON ------------------------------------------------------------------

#if VTM_SIMD_NEON

// LUT на 8 тритов: 3^8 = 6561 запись по 8 упакованных int8 = 52 КБ.
// Слово pack20 (20 тритов = 8 + 8 + 4) распаковывается двумя делениями на
// 6561 и тремя 64-битными записями. Для сравнения: LUT243 (5 тритов) требует
// четырёх делений и четырёх записей на слово и даёт ~20% медленнее.
// Таблица инициализируется лениво, как и LUT243 в backend_scalar.c.
static uint64_t g_lut8[6561];
static int g_lut8_ready;

static void lut_init(void) {
    unsigned v;
    int k;
    if (g_lut8_ready) return;
    for (v = 0; v < 6561u; v++) {
        uint64_t packed = 0;
        unsigned c = v;
        for (k = 0; k < 8; k++) {
            unsigned d = c % 3u;   // 0 -> -1, 1 -> 0, 2 -> +1
            int8_t trit = (int8_t)(d == 2u ? 1 : (d == 0u ? -1 : 0));
            packed |= (uint64_t)(uint8_t)trit << (8u * (unsigned)k);
            c /= 3u;
        }
        g_lut8[v] = packed;
    }
    g_lut8_ready = 1;
}

// 20 тритов слова pack20 -> 20 int8. Хвост буфера должен иметь запас:
// последняя запись идёт в out + 16 и пишет 8 байт.
static inline void unpack20(uint32_t v, int8_t *out) {
    uint32_t q0 = v / 6561u;
    uint32_t r0 = v - q0 * 6561u;
    uint32_t q1 = q0 / 6561u;
    uint32_t r1 = q0 - q1 * 6561u;
    uint64_t w0 = g_lut8[r0];
    uint64_t w1 = g_lut8[r1];
    uint64_t w2 = g_lut8[q1];
    memcpy(out + 0, &w0, 8);
    memcpy(out + 8, &w1, 8);
    memcpy(out + 16, &w2, 8);
}

// Сумма 4-ленточного аккумулятора в double (без лишнего округления в f32).
static inline double statvq_acc_f64(float32x4_t v) {
    return vaddvq_f64(vcvt_f64_f32(vget_low_f32(v)))
         + vaddvq_f64(vcvt_f64_f32(vget_high_f32(v)));
}

// Скалярное произведение тритов на f32-вектор. 16 элементов за шаг:
// int8 -> int16 -> int32 -> f32, четыре независимых аккумулятора (меньше
// ошибка округления, чем в один). Свёртка аккумуляторов и хвост — в double,
// как в скаляре: так сумма блока отличается от эталона на единицы ulp.
static inline double dot_i8_f32(const int8_t *t, const float *x, int n) {
    float32x4_t a0 = vdupq_n_f32(0.0f);
    float32x4_t a1 = vdupq_n_f32(0.0f);
    float32x4_t a2 = vdupq_n_f32(0.0f);
    float32x4_t a3 = vdupq_n_f32(0.0f);
    double s;
    int i = 0;

    for (; i + 16 <= n; i += 16) {
        int8x16_t b = vld1q_s8(t + i);
        int16x8_t lo = vmovl_s8(vget_low_s8(b));
        int16x8_t hi = vmovl_s8(vget_high_s8(b));
        a0 = vfmaq_f32(a0, vld1q_f32(x + i),
                       vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))));
        a1 = vfmaq_f32(a1, vld1q_f32(x + i + 4),
                       vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))));
        a2 = vfmaq_f32(a2, vld1q_f32(x + i + 8),
                       vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))));
        a3 = vfmaq_f32(a3, vld1q_f32(x + i + 12),
                       vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))));
    }
    s = statvq_acc_f64(vaddq_f32(a0, a1))
      + statvq_acc_f64(vaddq_f32(a2, a3));
    for (; i < n; i++) s += (double)t[i] * (double)x[i];
    return s;
}

// Dot одной строки тритовой матрицы: разбиваем на блоки масштаба (128/160)
// и внутри блока — на слова по 20 тритов. Масштаб применяется к сумме блока
// в double, как в скалярном LUT243.
static double row_dot_trit(const uint32_t *words, const float *scales,
                           const float *x, int64_t base, int cols,
                           int scale_block) {
    const int64_t sb = (int64_t)scale_block;
    const int64_t end = base + (int64_t)cols;
    int8_t buf[VTM_SCALE_BLOCK_160 + 64];
    int8_t *tp = buf + 32;   // запас под отрицательное смещение первого слова
    double acc = 0.0;
    int64_t s;

    for (s = (base / sb) * sb; s < end; s += sb) {
        int64_t e = s + sb;
        int64_t st;
        int64_t wi, wi_end;
        double sum;
        if (e > end) e = end;
        st = s < base ? base : s;
        wi = st / VTM_TRITS_PER_WORD;
        wi_end = (e - 1) / VTM_TRITS_PER_WORD;
        for (; wi <= wi_end; wi++)
            unpack20(load_u32_unaligned(words, wi),
                     tp + (ptrdiff_t)(wi * VTM_TRITS_PER_WORD - st));
        sum = dot_i8_f32(tp, x + (st - base), (int)(e - st));
        acc += sum * (double)load_f32_unaligned(scales, s / sb);
    }
    return acc;
}

// Строки [r0, r1) TMV: base по-прежнему r*cols, порядок внутри строки тот
// же, что при одном потоке, — поэтому разбиение по строкам побитово
// ничего не меняет.
static void tmv_neon_rows(float *y, const uint32_t *words,
                          const float *scales, const float *x, int cols,
                          int scale_block, int r0, int r1) {
    int r;

    for (r = r0; r < r1; r++) {
        int64_t base = (int64_t)r * (int64_t)cols;
        y[r] = (float)row_dot_trit(words, scales, x, base, cols,
                                   scale_block);
    }
}

typedef struct {
    float *y;
    const uint32_t *words;
    const float *scales;
    const float *x;
    int cols;
    int scale_block;
} tmv_neon_job;

static void tmv_neon_worker(void *p, int tid, int r0, int r1) {
    tmv_neon_job *j = (tmv_neon_job *)p;
    (void)tid;
    tmv_neon_rows(j->y, j->words, j->scales, j->x, j->cols, j->scale_block,
                  r0, r1);
}

static int tmv_neon_blocks(float *y, const uint8_t *w, const float *scales,
                           const float *x, int rows, int cols,
                           int scale_block) {
    tmv_neon_job job;
    if (!y || !w || !scales || !x) return -1;
    if (rows < 0 || cols < 0) return -1;
    if (scale_block != (int)VTM_SCALE_BLOCK_128 &&
        scale_block != (int)VTM_SCALE_BLOCK_160) return -1;
    if (rows == 0 || cols == 0) return 0;
    lut_init();   // до потоков: таблица пишется один раз, не в воркерах
    job.y = y;
    job.words = (const uint32_t *)w;
    job.scales = scales;
    job.x = x;
    job.cols = cols;
    job.scale_block = scale_block;
    vtm_parallel_rows(rows, (double)rows * (double)cols, tmv_neon_worker,
                      &job);
    return 0;
}

// Однопоточный RMSN — прежний код (VTM_THREADS=1 и малые n).
static int rmsnorm_neon_serial(float *y, const float *x, const float *w,
                               int n, float eps) {
    float64x2_t a0 = vdupq_n_f64(0.0);
    float64x2_t a1 = vdupq_n_f64(0.0);
    double sum, mean, rms;
    float inv;
    int i = 0;

    if (!y || !x || !w || n < 0) return -1;
    if (n == 0) return 0;

    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        float64x2_t lo = vcvt_f64_f32(vget_low_f32(v));
        float64x2_t hi = vcvt_f64_f32(vget_high_f32(v));
        a0 = vfmaq_f64(a0, lo, lo);
        a1 = vfmaq_f64(a1, hi, hi);
    }
    sum = vaddvq_f64(vaddq_f64(a0, a1));
    for (; i < n; i++) sum += (double)x[i] * (double)x[i];

    mean = sum / (double)n;
    rms = sqrt(mean + (double)eps);
    if (rms == 0.0) {
        // x == 0 и eps == 0 — тот же предел 0/0 -> 0, что в скаляре.
        memset(y, 0, (size_t)n * sizeof(float));
        return 0;
    }
    inv = (float)(1.0 / rms);

    i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t xv = vld1q_f32(x + i);
        float32x4_t wv = vld1q_f32(w + i);
        vst1q_f32(y + i, vmulq_f32(vmulq_f32(xv, wv), vdupq_n_f32(inv)));
    }
    for (; i < n; i++) y[i] = x[i] * w[i] * inv;
    return 0;
}

// RMSN для больших n: сумма квадратов по фиксированным блокам
// VTM_RMSN_BLOCK (порядок внутри блока прежний, частичные суммы
// складываются по возрастанию номера блока), затем поэлементная часть.
// Как и в скалярном бэкенде: при 2/4/8 потоках результат один и тот же
// (побитово), от VTM_THREADS=1 отличается только порядком double-сложения
// суммы квадратов (~1e-16 относительно; в f32-выходе обычно не видно).
#define VTM_RMSN_BLOCK 4096

typedef struct {
    const float *x;
    const float *w;
    float *y;
    double *sums;
    int n;
    float inv;
} rmsn_neon_job;

static void rmsnorm_neon_sum_worker(void *p, int tid, int b0, int b1) {
    rmsn_neon_job *j = (rmsn_neon_job *)p;
    int b;
    (void)tid;
    for (b = b0; b < b1; b++) {
        int i0 = b * VTM_RMSN_BLOCK;
        int i1 = i0 + VTM_RMSN_BLOCK;
        float64x2_t a0 = vdupq_n_f64(0.0);
        float64x2_t a1 = vdupq_n_f64(0.0);
        double sum;
        int i = i0;
        if (i1 > j->n) i1 = j->n;
        // То же векторное накопление double, что в serial-версии, но блок
        // начинается с нуля: порядок сложения внутри блока прежний, а
        // группировка по блокам даёт ~1e-16 относительной разницы.
        for (; i + 4 <= i1; i += 4) {
            float32x4_t v = vld1q_f32(j->x + i);
            float64x2_t lo = vcvt_f64_f32(vget_low_f32(v));
            float64x2_t hi = vcvt_f64_f32(vget_high_f32(v));
            a0 = vfmaq_f64(a0, lo, lo);
            a1 = vfmaq_f64(a1, hi, hi);
        }
        sum = vaddvq_f64(vaddq_f64(a0, a1));
        for (; i < i1; i++) sum += (double)j->x[i] * (double)j->x[i];
        j->sums[b] = sum;
    }
}

static void rmsnorm_neon_scale_worker(void *p, int tid, int b0, int b1) {
    rmsn_neon_job *j = (rmsn_neon_job *)p;
    const float32x4_t vinv = vdupq_n_f32(j->inv);
    int b;
    (void)tid;
    for (b = b0; b < b1; b++) {
        int i0 = b * VTM_RMSN_BLOCK;
        int i1 = i0 + VTM_RMSN_BLOCK;
        int i = i0;
        if (i1 > j->n) i1 = j->n;
        // Формула та же, что в serial-версии (f32: x*w*inv); каждый элемент
        // независим — совпадение побитовое.
        for (; i + 4 <= i1; i += 4) {
            float32x4_t xv = vld1q_f32(j->x + i);
            float32x4_t wv = vld1q_f32(j->w + i);
            vst1q_f32(j->y + i, vmulq_f32(vmulq_f32(xv, wv), vinv));
        }
        for (; i < i1; i++) j->y[i] = j->x[i] * j->w[i] * j->inv;
    }
}

static int rmsnorm_neon(float *y, const float *x, const float *w,
                        int n, float eps) {
    rmsn_neon_job job;
    double sum = 0.0, mean, rms;
    int nblocks, b;

    if (!y || !x || !w || n < 0) return -1;
    if (n == 0) return 0;
    if (vtm_threads() < 2 || n < VTM_PAR_RMSN_MIN)
        return rmsnorm_neon_serial(y, x, w, n, eps);

    nblocks = (n + VTM_RMSN_BLOCK - 1) / VTM_RMSN_BLOCK;
    job.x = x;
    job.w = w;
    job.y = y;
    job.n = n;
    job.sums = (double *)calloc((size_t)nblocks, sizeof(double));
    if (!job.sums)
        return rmsnorm_neon_serial(y, x, w, n, eps);

    vtm_parallel_rows(nblocks, 2.0 * (double)n, rmsnorm_neon_sum_worker,
                      &job);
    for (b = 0; b < nblocks; b++) sum += job.sums[b];
    free(job.sums);

    mean = sum / (double)n;
    rms = sqrt(mean + (double)eps);
    if (rms == 0.0) {
        memset(y, 0, (size_t)n * sizeof(float));
        return 0;
    }
    job.inv = (float)(1.0 / rms);
    vtm_parallel_rows(nblocks, (double)n, rmsnorm_neon_scale_worker, &job);
    return 0;
}

// --- NEON: Q4 (kind 5, задача 0157) ----------------------------------------
//
// Раскладка — по контракту vtm_backend.h: 32 значения группы в 16 байтах
// (GGML Q4_0: байт j — значение j в младшем ниббле и j+16 в старшем), масштаб
// fp32 на группу, группы идут по сплющенной матрице, группа может пересекать
// границу строк (cols не кратен 32).
//
// Почему не vdotq_s32. vdotq считает в int8 и требует int8-активаций:
// квантование x в int8 даёт ошибку порядка 1/127 на группу, то есть ~1e-2 на
// строку — это на три порядка больше порога приёмки (1e-5 против
// double-эталона). Поэтому вес разворачивается в f32 точно: значение — целое
// s = q - 8, |s| <= 8, представимо в f32 без округления (vcvtq_f32_s32 через
// vmovl_s8/vmovl_s16). Умножение — vfmaq_f32 (vmlaq_f32 в базовом NEON),
// четыре независимых f32-аккумулятора на группу (по 8 слагаемых в каждом),
// свёртка и масштаб группы — в double, как у скаляра. Ошибка группы ~1e-7
// относительно, строки — единицы 1e-7 (бенч --q4 печатает max|diff|).
//
// Q4_1 (asym) сводится к той же форме: q*scale + min = s*scale + (min +
// 8*scale), поэтому нужен ещё Σ x по группе (он считается по тем же
// загруженным векторам), а масштаб и (min + 8*scale) применяются в double.

// Группа (16 байт) -> два вектора по 16 значений s = q - 8 (j и j+16).
static inline void q4_group_split(const uint8_t *gp, int8x16_t *slo,
                                  int8x16_t *shi) {
    uint8x16_t b = vld1q_u8(gp);
    int8x16_t eight = vdupq_n_s8(8);
    *slo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(b, vdupq_n_u8(0x0fu))),
                    eight);
    *shi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(b, 4)), eight);
}

// 16 int8 -> 4 вектора f32 (значения целые, округления нет).
static inline void q4_i8x16_to_f32x4(int8x16_t v, float32x4_t *f) {
    int16x8_t lo = vmovl_s8(vget_low_s8(v));
    int16x8_t hi = vmovl_s8(vget_high_s8(v));
    f[0] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo)));
    f[1] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo)));
    f[2] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi)));
    f[3] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi)));
}

// Полная группа (32 значения, off == 0) на непрерывный x: dot = Σ s_t*x_t и
// xsum = Σ x_t (нужен только для Q4_1). dot — уже double (свёртка f32-лент),
// масштаб группы применяет вызывающий: так ошибка не накапливается по строке.
static inline void q4_group_dot(const uint8_t *gp, const float *xp, int asym,
                                double *dot, double *xsum) {
    int8x16_t sl, sh;
    float32x4_t w0[4], w1[4];
    float32x4_t x0, x1, x2, x3, x4, x5, x6, x7;
    float32x4_t a0, a1, a2, a3;

    q4_group_split(gp, &sl, &sh);
    q4_i8x16_to_f32x4(sl, w0);
    q4_i8x16_to_f32x4(sh, w1);
    x0 = vld1q_f32(xp + 0);
    x1 = vld1q_f32(xp + 4);
    x2 = vld1q_f32(xp + 8);
    x3 = vld1q_f32(xp + 12);
    x4 = vld1q_f32(xp + 16);
    x5 = vld1q_f32(xp + 20);
    x6 = vld1q_f32(xp + 24);
    x7 = vld1q_f32(xp + 28);
    a0 = vmulq_f32(w0[0], x0);
    a1 = vmulq_f32(w0[1], x1);
    a2 = vmulq_f32(w0[2], x2);
    a3 = vmulq_f32(w0[3], x3);
    a0 = vfmaq_f32(a0, w1[0], x4);
    a1 = vfmaq_f32(a1, w1[1], x5);
    a2 = vfmaq_f32(a2, w1[2], x6);
    a3 = vfmaq_f32(a3, w1[3], x7);
    *dot = statvq_acc_f64(vaddq_f32(a0, a1))
         + statvq_acc_f64(vaddq_f32(a2, a3));
    if (asym) {
        float32x4_t s0 = vaddq_f32(vaddq_f32(x0, x1), vaddq_f32(x2, x3));
        float32x4_t s1 = vaddq_f32(vaddq_f32(x4, x5), vaddq_f32(x6, x7));
        *xsum = statvq_acc_f64(vaddq_f32(s0, s1));
    } else {
        *xsum = 0.0;
    }
}

// Dot строки Q4 на непрерывный x. Полные группы — векторами, голова/хвост
// строки (off != 0 или неполная последняя группа, всего < 32 значений на
// строку) — скалярно, ровно как в double-эталоне.
static double q4_row_dot_neon(const uint8_t *q, const float *scales,
                              const float *mins, int64_t base, int k,
                              const float *x, int asym) {
    double acc = 0.0;
    int j0 = 0;

    while (j0 < k) {
        const int64_t idx = base + (int64_t)j0;
        const uint64_t g = (uint64_t)idx >> 5;
        const unsigned off = (unsigned)idx & 31u;
        int n = VTM_Q4_GROUP_SIZE - (int)off;
        if (n > k - j0) n = k - j0;

        if (off == 0 && n == VTM_Q4_GROUP_SIZE) {
            const double scale = (double)load_f32_unaligned(scales, (int64_t)g);
            double dot, xsum;
            q4_group_dot(q + (size_t)g * (VTM_Q4_GROUP_SIZE / 2), x + j0,
                         asym, &dot, &xsum);
            if (asym) {
                const double minv = (double)load_f32_unaligned(mins,
                                                               (int64_t)g);
                acc += dot * scale + xsum * (minv + 8.0 * scale);
            } else {
                acc += dot * scale;
            }
            j0 += VTM_Q4_GROUP_SIZE;
            continue;
        }
        {
            const double scale = (double)load_f32_unaligned(scales, (int64_t)g);
            const double minv = asym
                ? (double)load_f32_unaligned(mins, (int64_t)g) : 0.0;
            const uint8_t *gp = q
                + (size_t)g * (VTM_Q4_GROUP_SIZE / 2);
            int t;
            for (t = 0; t < n; t++) {
                unsigned within = off + (unsigned)t;
                unsigned byte = gp[within & 15u];
                unsigned v = (within < 16u) ? (byte & 0x0fu) : (byte >> 4);
                double wv = asym ? (double)v * scale + minv
                                 : ((double)v - 8.0) * scale;
                acc += wv * (double)x[j0 + t];
            }
        }
        j0 += n;
    }
    return acc;
}

// Строки [r0, r1) TMV по Q4. Разбиение по строкам не меняет результат:
// каждая строка считается одним и тем же кодом в одном и том же порядке.
int vtm_neon_q4_tmv_rows(float *y, const uint8_t *q, const float *scales,
                         const float *mins, const float *x, int r0, int r1,
                         int cols, int asym) {
    int r;

    if (!y || !q || !scales || !x) return -1;
    if (cols < 0 || r0 < 0 || r1 < r0) return -1;
    if (asym && !mins) return -1;
    if (cols == 0) {
        for (r = r0; r < r1; r++) y[r] = 0.0f;
        return 0;
    }
    for (r = r0; r < r1; r++)
        y[r] = (float)q4_row_dot_neon(q, scales, mins,
                                      (int64_t)r * (int64_t)cols, cols, x,
                                      asym);
    return 0;
}

// Строка W -> wrow (k значений s = q - 8 в f32, точно). Хвост/голова группы —
// скалярно. Буфер общий для всех n столбцов: деквантизация строки ровно одна
// на строку, а не на каждый столбец (задача 0157, п.2).
static void q4_row_weights_f32(float *wrow, const uint8_t *q, int64_t base,
                               int k) {
    int j0 = 0;

    while (j0 < k) {
        const int64_t idx = base + (int64_t)j0;
        const uint64_t g = (uint64_t)idx >> 5;
        const unsigned off = (unsigned)idx & 31u;
        int n = VTM_Q4_GROUP_SIZE - (int)off;
        const uint8_t *gp;
        if (n > k - j0) n = k - j0;
        gp = q + (size_t)g * (VTM_Q4_GROUP_SIZE / 2);
        if (off == 0 && n == VTM_Q4_GROUP_SIZE) {
            int8x16_t sl, sh;
            float32x4_t f[4];
            q4_group_split(gp, &sl, &sh);
            q4_i8x16_to_f32x4(sl, f);
            vst1q_f32(wrow + j0 + 0, f[0]);
            vst1q_f32(wrow + j0 + 4, f[1]);
            vst1q_f32(wrow + j0 + 8, f[2]);
            vst1q_f32(wrow + j0 + 12, f[3]);
            q4_i8x16_to_f32x4(sh, f);
            vst1q_f32(wrow + j0 + 16, f[0]);
            vst1q_f32(wrow + j0 + 20, f[1]);
            vst1q_f32(wrow + j0 + 24, f[2]);
            vst1q_f32(wrow + j0 + 28, f[3]);
        } else {
            int t;
            for (t = 0; t < n; t++) {
                unsigned within = off + (unsigned)t;
                unsigned byte = gp[within & 15u];
                unsigned v = (within < 16u) ? (byte & 0x0fu) : (byte >> 4);
                wrow[j0 + t] = (float)((int)v - 8);
            }
        }
        j0 += n;
    }
}

// MM по Q4 для n > 1: wrow (k f32, развёрнутая строка) общий для всех
// столбцов; 4 столбца считаются одним f32-вектором, группа копится в f32 и
// вливается в double-аккумуляторы с масштабом группы. Голова/хвост строки и
// столбцы n % 4 — скалярно (как в double-эталоне).
//
// wrow — scratch этого потока на k значений (вызывающий даёт свой на поток:
// строки делятся между потоками, состояние не разделяется).
int vtm_neon_q4_mm_rows(float *y, const uint8_t *q, const float *scales,
                        const float *mins, const float *b, int r0, int r1,
                        int k, int n, int asym, float *wrow) {
    int r, c;

    if (!y || !q || !scales || !b || !wrow) return -1;
    if (k < 0 || n < 0 || r0 < 0 || r1 < r0) return -1;
    if (asym && !mins) return -1;
    if (k == 0) {
        memset(y, 0, (size_t)(r1 - r0) * (size_t)n * sizeof(float));
        return 0;
    }

    for (r = r0; r < r1; r++) {
        const int64_t base = (int64_t)r * (int64_t)k;
        float *yrow = y + (size_t)r * (size_t)n;

        q4_row_weights_f32(wrow, q, base, k);

        for (c = 0; c + 4 <= n; c += 4) {
            float64x2_t accl = vdupq_n_f64(0.0);
            float64x2_t acch = vdupq_n_f64(0.0);
            int j0 = 0;

            while (j0 < k) {
                const int64_t idx = base + (int64_t)j0;
                const uint64_t g = (uint64_t)idx >> 5;
                const unsigned off = (unsigned)idx & 31u;
                const double scale = (double)load_f32_unaligned(scales,
                                                               (int64_t)g);
                const double minv = asym
                    ? (double)load_f32_unaligned(mins, (int64_t)g) : 0.0;
                const double cg = asym ? minv + 8.0 * scale : 0.0;
                int cnt = VTM_Q4_GROUP_SIZE - (int)off;
                if (cnt > k - j0) cnt = k - j0;

                if (cnt == VTM_Q4_GROUP_SIZE) {
                    const float *bp = b + (size_t)j0 * (size_t)n + (size_t)c;
                    float32x4_t a0 = vdupq_n_f32(0.0f);
                    float32x4_t a1 = vdupq_n_f32(0.0f);
                    float32x4_t a2 = vdupq_n_f32(0.0f);
                    float32x4_t a3 = vdupq_n_f32(0.0f);
                    float32x4_t xs0 = vdupq_n_f32(0.0f);
                    float32x4_t xs1 = vdupq_n_f32(0.0f);
                    float32x4_t s, sv;
                    int t;

                    for (t = 0; t < VTM_Q4_GROUP_SIZE; t += 4) {
                        float32x4_t b0 = vld1q_f32(bp + (size_t)(t + 0)
                                                   * (size_t)n);
                        float32x4_t b1 = vld1q_f32(bp + (size_t)(t + 1)
                                                   * (size_t)n);
                        float32x4_t b2 = vld1q_f32(bp + (size_t)(t + 2)
                                                   * (size_t)n);
                        float32x4_t b3 = vld1q_f32(bp + (size_t)(t + 3)
                                                   * (size_t)n);
                        a0 = vfmaq_f32(a0, b0, vdupq_n_f32(wrow[j0 + t + 0]));
                        a1 = vfmaq_f32(a1, b1, vdupq_n_f32(wrow[j0 + t + 1]));
                        a2 = vfmaq_f32(a2, b2, vdupq_n_f32(wrow[j0 + t + 2]));
                        a3 = vfmaq_f32(a3, b3, vdupq_n_f32(wrow[j0 + t + 3]));
                        if (asym) {
                            xs0 = vaddq_f32(xs0, vaddq_f32(b0, b1));
                            xs1 = vaddq_f32(xs1, vaddq_f32(b2, b3));
                        }
                    }
                    s = vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
                    sv = vdupq_n_f64(scale);
                    accl = vfmaq_f64(accl, vcvt_f64_f32(vget_low_f32(s)), sv);
                    acch = vfmaq_f64(acch, vcvt_f64_f32(vget_high_f32(s)), sv);
                    if (asym) {
                        float32x4_t xs = vaddq_f32(xs0, xs1);
                        float64x2_t cgv = vdupq_n_f64(cg);
                        accl = vfmaq_f64(accl, vcvt_f64_f32(vget_low_f32(xs)),
                                         cgv);
                        acch = vfmaq_f64(acch,
                                         vcvt_f64_f32(vget_high_f32(xs)), cgv);
                    }
                } else {
                    double tmp[4];
                    int i;
                    vst1q_f64(tmp, accl);
                    vst1q_f64(tmp + 2, acch);
                    for (i = 0; i < 4; i++) {
                        double p = 0.0, xs = 0.0;
                        int t;
                        for (t = 0; t < cnt; t++) {
                            const float *bp = b + (size_t)(j0 + t)
                                * (size_t)n + (size_t)c + (size_t)i;
                            double bx = (double)bp[0];
                            p += (double)wrow[j0 + t] * bx;
                            if (asym) xs += bx;
                        }
                        tmp[i] += p * scale + (asym ? xs * cg : 0.0);
                    }
                    accl = vld1q_f64(tmp);
                    acch = vld1q_f64(tmp + 2);
                }
                j0 += cnt;
            }
            vst1q_f32(yrow + c, vcombine_f32(vcvt_f32_f64(accl),
                                             vcvt_f32_f64(acch)));
        }
        // n = 2 или 3 (короткий prompt): те же группы, но по два столбца —
        // иначе такой MM уходил бы в скалярный хвост и терял весь выигрыш.
        for (; c + 2 <= n; c += 2) {
            float64x2_t acc = vdupq_n_f64(0.0);
            int j0 = 0;

            while (j0 < k) {
                const int64_t idx = base + (int64_t)j0;
                const uint64_t g = (uint64_t)idx >> 5;
                const unsigned off = (unsigned)idx & 31u;
                const double scale = (double)load_f32_unaligned(scales,
                                                               (int64_t)g);
                const double minv = asym
                    ? (double)load_f32_unaligned(mins, (int64_t)g) : 0.0;
                const double cg = asym ? minv + 8.0 * scale : 0.0;
                int cnt = VTM_Q4_GROUP_SIZE - (int)off;
                if (cnt > k - j0) cnt = k - j0;

                if (cnt == VTM_Q4_GROUP_SIZE) {
                    const float *bp = b + (size_t)j0 * (size_t)n + (size_t)c;
                    float32x2_t a0 = vdup_n_f32(0.0f);
                    float32x2_t a1 = vdup_n_f32(0.0f);
                    float32x2_t a2 = vdup_n_f32(0.0f);
                    float32x2_t a3 = vdup_n_f32(0.0f);
                    float32x2_t xs0 = vdup_n_f32(0.0f);
                    float32x2_t xs1 = vdup_n_f32(0.0f);
                    int t;

                    for (t = 0; t < VTM_Q4_GROUP_SIZE; t += 4) {
                        float32x2_t b0 = vld1_f32(bp + (size_t)(t + 0)
                                                  * (size_t)n);
                        float32x2_t b1 = vld1_f32(bp + (size_t)(t + 1)
                                                  * (size_t)n);
                        float32x2_t b2 = vld1_f32(bp + (size_t)(t + 2)
                                                  * (size_t)n);
                        float32x2_t b3 = vld1_f32(bp + (size_t)(t + 3)
                                                  * (size_t)n);
                        a0 = vfma_f32(a0, b0, vdup_n_f32(wrow[j0 + t + 0]));
                        a1 = vfma_f32(a1, b1, vdup_n_f32(wrow[j0 + t + 1]));
                        a2 = vfma_f32(a2, b2, vdup_n_f32(wrow[j0 + t + 2]));
                        a3 = vfma_f32(a3, b3, vdup_n_f32(wrow[j0 + t + 3]));
                        if (asym) {
                            xs0 = vadd_f32(xs0, vadd_f32(b0, b1));
                            xs1 = vadd_f32(xs1, vadd_f32(b2, b3));
                        }
                    }
                    acc = vfmaq_f64(acc,
                                    vcvt_f64_f32(vadd_f32(vadd_f32(a0, a1),
                                                          vadd_f32(a2, a3))),
                                    vdupq_n_f64(scale));
                    if (asym)
                        acc = vfmaq_f64(acc,
                                        vcvt_f64_f32(vadd_f32(xs0, xs1)),
                                        vdupq_n_f64(cg));
                } else {
                    double tmp[2];
                    int i;
                    vst1q_f64(tmp, acc);
                    for (i = 0; i < 2; i++) {
                        double p = 0.0, xs = 0.0;
                        int t;
                        for (t = 0; t < cnt; t++) {
                            const float *bp = b + (size_t)(j0 + t)
                                * (size_t)n + (size_t)c + (size_t)i;
                            double bx = (double)bp[0];
                            p += (double)wrow[j0 + t] * bx;
                            if (asym) xs += bx;
                        }
                        tmp[i] += p * scale + (asym ? xs * cg : 0.0);
                    }
                    acc = vld1q_f64(tmp);
                }
                j0 += cnt;
            }
            vst1_f32(yrow + c, vcvt_f32_f64(acc));
        }
        for (; c < n; c++) {
            double acc = 0.0;
            int j0 = 0;
            while (j0 < k) {
                const int64_t idx = base + (int64_t)j0;
                const uint64_t g = (uint64_t)idx >> 5;
                const unsigned off = (unsigned)idx & 31u;
                const double scale = (double)load_f32_unaligned(scales,
                                                               (int64_t)g);
                const double minv = asym
                    ? (double)load_f32_unaligned(mins, (int64_t)g) : 0.0;
                const double cg = asym ? minv + 8.0 * scale : 0.0;
                double p = 0.0, xs = 0.0;
                int cnt = VTM_Q4_GROUP_SIZE - (int)off;
                int t;
                if (cnt > k - j0) cnt = k - j0;
                for (t = 0; t < cnt; t++) {
                    const float *bp = b + (size_t)(j0 + t) * (size_t)n
                        + (size_t)c;
                    double bx = (double)bp[0];
                    p += (double)wrow[j0 + t] * bx;
                    if (asym) xs += bx;
                }
                acc += p * scale + (asym ? xs * cg : 0.0);
                j0 += cnt;
            }
            yrow[c] = (float)acc;
        }
    }
    return 0;
}

// Развернуть одну строку Q4 (EMB) в f32. Как и в эталоне, в y пишется
// (float) от значения, посчитанного в double, — расхождение не больше ulp.
int vtm_neon_q4_dequant_row(float *y, const uint8_t *q, const float *scales,
                            const float *mins, int64_t base, int cols,
                            int asym) {
    int j0 = 0;

    if (!y || !q || !scales || cols < 0 || base < 0) return -1;
    if (asym && !mins) return -1;
    while (j0 < cols) {
        const int64_t idx = base + (int64_t)j0;
        const uint64_t g = (uint64_t)idx >> 5;
        const unsigned off = (unsigned)idx & 31u;
        const double scale = (double)load_f32_unaligned(scales, (int64_t)g);
        const double minv = asym
            ? (double)load_f32_unaligned(mins, (int64_t)g) : 0.0;
        const uint8_t *gp = q + (size_t)g * (VTM_Q4_GROUP_SIZE / 2);
        int n = VTM_Q4_GROUP_SIZE - (int)off;

        if (n > cols - j0) n = cols - j0;
        if (off == 0 && n == VTM_Q4_GROUP_SIZE) {
            int8x16_t sl, sh;
            float32x4_t f[4];
            float32x4_t sc, mn;
            q4_group_split(gp, &sl, &sh);
            sc = vdupq_n_f32((float)scale);
            mn = vdupq_n_f32((float)minv);
            q4_i8x16_to_f32x4(sl, f);
            if (asym) {
                f[0] = vfmaq_f32(mn, f[0], sc);
                f[1] = vfmaq_f32(mn, f[1], sc);
                f[2] = vfmaq_f32(mn, f[2], sc);
                f[3] = vfmaq_f32(mn, f[3], sc);
            } else {
                f[0] = vmulq_f32(f[0], sc);
                f[1] = vmulq_f32(f[1], sc);
                f[2] = vmulq_f32(f[2], sc);
                f[3] = vmulq_f32(f[3], sc);
            }
            vst1q_f32(y + j0 + 0, f[0]);
            vst1q_f32(y + j0 + 4, f[1]);
            vst1q_f32(y + j0 + 8, f[2]);
            vst1q_f32(y + j0 + 12, f[3]);
            q4_i8x16_to_f32x4(sh, f);
            if (asym) {
                f[0] = vfmaq_f32(mn, f[0], sc);
                f[1] = vfmaq_f32(mn, f[1], sc);
                f[2] = vfmaq_f32(mn, f[2], sc);
                f[3] = vfmaq_f32(mn, f[3], sc);
            } else {
                f[0] = vmulq_f32(f[0], sc);
                f[1] = vmulq_f32(f[1], sc);
                f[2] = vmulq_f32(f[2], sc);
                f[3] = vmulq_f32(f[3], sc);
            }
            vst1q_f32(y + j0 + 16, f[0]);
            vst1q_f32(y + j0 + 20, f[1]);
            vst1q_f32(y + j0 + 24, f[2]);
            vst1q_f32(y + j0 + 28, f[3]);
        } else {
            int t;
            for (t = 0; t < n; t++) {
                unsigned within = off + (unsigned)t;
                unsigned byte = gp[within & 15u];
                unsigned v = (within < 16u) ? (byte & 0x0fu) : (byte >> 4);
                y[j0 + t] = (float)(asym ? (double)v * scale + minv
                                         : ((double)v - 8.0) * scale);
            }
        }
        j0 += n;
    }
    return 0;
}

// --- NEON: Q4_K (kind 6, задачи 0165/0167) --------------------------------
//
// Блок GGML — 144 байта на 256 значений: d fp16, dmin fp16, 12 байт
// 6-битных scale/min, 128 байт нибблов. Кусок qs[32k:32k+32] даёт два
// под-блока: младшие нибблы — под-блок 2k (32 значения подряд), старшие —
// 2k+1. В отличие от Q4_0 (байт j = значения j и j+16), здесь значения
// под-блока идут подряд, поэтому одна 16-байтовая загрузка — 16 значений.
//
// Арифметика: q ∈ 0..15 точно представим в f32, поэтому Σ q·x копится в
// f32-лентах (4 независимых аккумулятора); d·sc·Σq·x и −dmin·m·Σx тоже
// считаются в f32, а складываются по под-блокам в double (быстрый путь 0167,
// ошибка строки ~1e-6). Эталон 0165 — всё в double, активации заранее
// приведены к double, произведение vfmaq_f64; включается VTM_Q4K_STRICT=1.
// Для минимума под-блока нужен Σ x — он не зависит от строки и считается
// один раз на вызов. Дальше в этом разделе описаны обе ветки.
//
// Быстрый путь — только при cols, кратном 256 (тогда строка начинается с
// границы блока): у Qwen2.5 3B это все Q4_K-матрицы (2048, 11008, 256).
// Иначе ядро возвращает -1, и вызывающий считает строку скалярным эталоном.

// VTM_Q4K_STRICT=1 — эталонное f64-накопление (как в 0165): арифметика
// совпадает со скалярным эталоном backend_scalar.c (ошибка строки ~1e-9), но
// вдвое медленнее f32. Значение читается один раз и кэшируется: горячий путь
// не должен звать getenv на каждую строку.
static int q4k_strict_mode(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("VTM_Q4K_STRICT");
        cached = (e && *e && *e != '0') ? 1 : 0;
    }
    return cached;
}

static inline float q4k_f16(const uint8_t *p) {
    uint16_t h;
    memcpy(&h, p, sizeof h);
    return vtm_half_to_float(h);
}

// get_scale_min_k4 из ggml: 6-битные (scale, min) под-блока j.
static inline void q4k_scale_min(const uint8_t *scales, unsigned j,
                                 unsigned *sc, unsigned *mn) {
    if (j < 4u) {
        *sc = (unsigned)scales[j] & 0x3Fu;
        *mn = (unsigned)scales[j + 4u] & 0x3Fu;
    } else {
        *sc = ((unsigned)scales[j + 4u] & 0x0Fu)
            | (((unsigned)scales[j - 4u] >> 6) << 4);
        *mn = ((unsigned)scales[j + 4u] >> 4)
            | (((unsigned)scales[j] >> 6) << 4);
    }
}

// 32 значения под-блока (low/high нибблы 32 байт qs) на xp[0..31] в double:
// dot = Σ q·x. q ∈ 0..15 точно представим в f32, но произведение q·x в f32
// округляется (|q·x| до десятков → ошибка ~1e-6 на слагаемое), поэтому
// активации заранее приведены к double, а произведение считается vfmaq_f64:
// ошибка строки — единицы 1e-9, как у скалярного эталона (проверено на срезе
// 3B: 1,07e-6 → 5e-9 по логитам). Минимум под-блока применяется к Σx
// вызывающим — Σx тоже double.
static inline double q4k_dot32(const uint8_t *qs, const double *xp, int high) {
    uint8x16_t b0 = vld1q_u8(qs);
    uint8x16_t b1 = vld1q_u8(qs + 16);
    int8x16_t q0, q1;
    float32x4_t w[4];
    float64x2_t a[8];
    int i;

    if (high) {
        q0 = vreinterpretq_s8_u8(vshrq_n_u8(b0, 4));
        q1 = vreinterpretq_s8_u8(vshrq_n_u8(b1, 4));
    } else {
        uint8x16_t mask = vdupq_n_u8(0x0fu);
        q0 = vreinterpretq_s8_u8(vandq_u8(b0, mask));
        q1 = vreinterpretq_s8_u8(vandq_u8(b1, mask));
    }
    for (i = 0; i < 8; i++) a[i] = vdupq_n_f64(0.0);

    q4_i8x16_to_f32x4(q0, w);
    for (i = 0; i < 4; i++) {
        a[i * 2] = vfmaq_f64(a[i * 2], vcvt_f64_f32(vget_low_f32(w[i])),
                             vld1q_f64(xp + 4 * i));
        a[i * 2 + 1] = vfmaq_f64(a[i * 2 + 1],
                                 vcvt_f64_f32(vget_high_f32(w[i])),
                                 vld1q_f64(xp + 4 * i + 2));
    }
    q4_i8x16_to_f32x4(q1, w);
    for (i = 0; i < 4; i++) {
        a[i * 2] = vfmaq_f64(a[i * 2], vcvt_f64_f32(vget_low_f32(w[i])),
                             vld1q_f64(xp + 16 + 4 * i));
        a[i * 2 + 1] = vfmaq_f64(a[i * 2 + 1],
                                 vcvt_f64_f32(vget_high_f32(w[i])),
                                 vld1q_f64(xp + 16 + 4 * i + 2));
    }
    return vaddvq_f64(a[0]) + vaddvq_f64(a[1]) + vaddvq_f64(a[2])
         + vaddvq_f64(a[3]) + vaddvq_f64(a[4]) + vaddvq_f64(a[5])
         + vaddvq_f64(a[6]) + vaddvq_f64(a[7]);
}

// Σ xp[0..31] в double (нужна для минимума под-блока: w = scale·q + minv).
static inline double q4k_sum32(const double *xp) {
    float64x2_t a0 = vdupq_n_f64(0.0), a1 = a0, a2 = a0, a3 = a0;
    int i;
    for (i = 0; i < 32; i += 8) {
        a0 = vaddq_f64(a0, vld1q_f64(xp + i));
        a1 = vaddq_f64(a1, vld1q_f64(xp + i + 2));
        a2 = vaddq_f64(a2, vld1q_f64(xp + i + 4));
        a3 = vaddq_f64(a3, vld1q_f64(xp + i + 6));
    }
    return vaddvq_f64(vaddq_f64(a0, a1)) + vaddvq_f64(vaddq_f64(a2, a3));
}

// f32-вариант q4k_dot32 (задача 0167): q ∈ 0..15 точно в f32, произведение
// копится в f32 (4 независимых аккумулятора по 4 значения), как у Q4-ядер
// 0157/0162. Ошибка на строку — единицы 1e-6 относительно |Σ|, на логитах 3B
// даёт max|Δ| ≈ 1e-6 (порог 1e-5); f64-эталон включается VTM_Q4K_STRICT=1.
static inline float q4k_dot32_f32(const uint8_t *qs, const float *xp,
                                  int high) {
    uint8x16_t b0 = vld1q_u8(qs);
    uint8x16_t b1 = vld1q_u8(qs + 16);
    int8x16_t q0, q1;
    float32x4_t w[4];
    float32x4_t a0, a1, a2, a3, s;

    if (high) {
        q0 = vreinterpretq_s8_u8(vshrq_n_u8(b0, 4));
        q1 = vreinterpretq_s8_u8(vshrq_n_u8(b1, 4));
    } else {
        uint8x16_t mask = vdupq_n_u8(0x0fu);
        q0 = vreinterpretq_s8_u8(vandq_u8(b0, mask));
        q1 = vreinterpretq_s8_u8(vandq_u8(b1, mask));
    }
    // 4 независимых аккумулятора: по одному на четверть 16-байтовой загрузки
    // (16 значений — 4 вектора), второй кусок qs прибавляется к ним же.
    q4_i8x16_to_f32x4(q0, w);
    a0 = vfmaq_f32(vdupq_n_f32(0.0f), w[0], vld1q_f32(xp + 0));
    a1 = vfmaq_f32(vdupq_n_f32(0.0f), w[1], vld1q_f32(xp + 4));
    a2 = vfmaq_f32(vdupq_n_f32(0.0f), w[2], vld1q_f32(xp + 8));
    a3 = vfmaq_f32(vdupq_n_f32(0.0f), w[3], vld1q_f32(xp + 12));
    q4_i8x16_to_f32x4(q1, w);
    a0 = vfmaq_f32(a0, w[0], vld1q_f32(xp + 16));
    a1 = vfmaq_f32(a1, w[1], vld1q_f32(xp + 20));
    a2 = vfmaq_f32(a2, w[2], vld1q_f32(xp + 24));
    a3 = vfmaq_f32(a3, w[3], vld1q_f32(xp + 28));
    s = vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
    return vaddvq_f32(s);
}

// Σ xp[0..31] в double по float-активациям (минимум под-блока умножается на
// сумму активаций; не зависит от строки, считается один раз на вызов).
static inline double q4k_sum32_f32(const float *xp) {
    float64x2_t a0 = vdupq_n_f64(0.0), a1 = a0;
    int i;
    for (i = 0; i < 32; i += 4) {
        a0 = vaddq_f64(a0, vcvt_f64_f32(vld1_f32(xp + i)));
        a1 = vaddq_f64(a1, vcvt_f64_f32(vld1_f32(xp + i + 2)));
    }
    return vaddvq_f64(vaddq_f64(a0, a1));
}

// Одна строка Q4_K в f32-арифметике (быстрый путь, по умолчанию).
static float q4k_row_dot_f32(const uint8_t *blocks, int64_t base, int cols,
                             const float *x, const double *xsums) {
    double acc = 0.0;
    int64_t block = base / (int64_t)VTM_Q4K_BLOCK_VALUES;
    int nblocks = cols / (int)VTM_Q4K_BLOCK_VALUES;
    int b;

    for (b = 0; b < nblocks; b++) {
        const uint8_t *bp = blocks
            + (size_t)(block + b) * (size_t)VTM_Q4K_BLOCK_BYTES;
        // d, dmin — f16, произведение с 6-битными sc/mn точное в f32.
        float d = q4k_f16(bp);
        float dmin = q4k_f16(bp + 2);
        const float *xp = x + (size_t)b * VTM_Q4K_BLOCK_VALUES;
        unsigned s;
        for (s = 0; s < 8u; s++) {
            unsigned sc, mn;
            float dot;
            q4k_scale_min(bp + 4, s, &sc, &mn);
            dot = q4k_dot32_f32(bp + 16u + (size_t)(s >> 1) * 32u,
                                xp + (size_t)s * 32u, (int)(s & 1u));
            acc += (double)((d * (float)sc) * dot)
                 + (double)(-(dmin * (float)mn))
                   * xsums[(size_t)b * 8u + (size_t)s];
        }
    }
    return (float)acc;
}

// Одна полная строка Q4_K (cols кратен 256) -> скалярное произведение.
// xd — активации, заранее приведённые к double (общие для всех строк).
static double q4k_row_dot_neon(const uint8_t *blocks, int64_t base, int cols,
                               const double *xd, const double *xsums) {
    double acc = 0.0;
    int64_t block = base / (int64_t)VTM_Q4K_BLOCK_VALUES;
    int nblocks = cols / (int)VTM_Q4K_BLOCK_VALUES;
    int b;

    for (b = 0; b < nblocks; b++) {
        const uint8_t *bp = blocks
            + (size_t)(block + b) * (size_t)VTM_Q4K_BLOCK_BYTES;
        double d = (double)q4k_f16(bp);
        double dmin = (double)q4k_f16(bp + 2);
        const double *xp = xd + (size_t)b * VTM_Q4K_BLOCK_VALUES;
        unsigned s;
        for (s = 0; s < 8u; s++) {
            unsigned sc, mn;
            double dot, xsum;
            q4k_scale_min(bp + 4, s, &sc, &mn);
            dot = q4k_dot32(bp + 16u + (size_t)(s >> 1) * 32u,
                            xp + (size_t)s * 32u, (int)(s & 1u));
            // Σx по под-блоку не зависит от весов: посчитана один раз на
            // вызов (см. vtm_neon_q4k_tmv_rows), а не на каждую строку.
            xsum = xsums[(size_t)b * 8u + (size_t)s];
            acc += (d * (double)sc) * dot + (-(dmin * (double)mn)) * xsum;
        }
    }
    return acc;
}

int vtm_neon_q4k_tmv_rows(float *y, const uint8_t *blocks, const float *x,
                          int r0, int r1, int cols) {
    double *xd, *xsums, *buf;
    int r, i;

    if (!y || !blocks || !x || r0 < 0 || r1 < r0 || cols <= 0) return -1;
    if (cols % (int)VTM_Q4K_BLOCK_VALUES != 0) return -1;  // скалярный эталон

    // Быстрый путь (0167): f32-накопление, активации остаются float. Σx по
    // под-блокам нужна только для минимума и не зависит от строки — считается
    // один раз на вызов в double (цена — cols сложений против rows·cols MAC).
    if (!q4k_strict_mode()) {
        xsums = (double *)malloc((size_t)(cols / 32) * sizeof(double));
        if (!xsums) return -1;
        for (i = 0; i < cols / 32; i++)
            xsums[i] = q4k_sum32_f32(x + (size_t)i * 32u);
        for (r = r0; r < r1; r++) {
            int64_t base = (int64_t)r * (int64_t)cols;
            y[r] = q4k_row_dot_f32(blocks, base, cols, x, xsums);
        }
        free(xsums);
        return 0;
    }

    // Эталон (VTM_Q4K_STRICT=1): активации в double, произведение — f64.
    // Активации — одни на все строки: double-копия и Σx по под-блокам
    // считаются один раз на вызов, а не на строку.
    buf = (double *)malloc(((size_t)cols + (size_t)(cols / 32) * 8u)
                           * sizeof(double));
    if (!buf) return -1;
    xd = buf;
    xsums = buf + cols;
    for (i = 0; i < cols; i++) xd[i] = (double)x[i];
    for (i = 0; i < cols / 32; i++)
        xsums[i] = q4k_sum32(xd + (size_t)i * 32u);
    for (r = r0; r < r1; r++) {
        int64_t base = (int64_t)r * (int64_t)cols;
        y[r] = (float)q4k_row_dot_neon(blocks, base, cols, xd, xsums);
    }
    free(buf);
    return 0;
}

// --- NEON: Q6_K (kind 7, задачи 0165/0167) --------------------------------
//
// Блок Q6_K — 210 байт на 256 значений: ql 128, qh 64, scales 16*int8, d fp16.
// 16 под-блоков по 16 значений: для под-блока sub chunk = sub/8,
// quad = (sub%8)/2 (ниббл ql и сдвиг 2 бит из qh), half = sub%2,
// q = (ql-ниббл | (2 бита из qh) << 4) − 32, w = d*sc*q.
//
// Арифметика — как у Q4_K: два варианта. Быстрый (0167, по умолчанию): q
// точно представим в f32, Σ q·x копится в f32, d*sc применяется к сумме в
// f32, по под-блокам — в double. Эталон (VTM_Q4K_STRICT=1): активации в
// double, произведение — vfmaq_f64. Минимума у Q6_K нет, Σx не нужна.
// Быстрый путь — при cols, кратном 256.

static inline double q6k_dot16(const uint8_t *block, unsigned sub,
                               const double *xp) {
    unsigned chunk = sub >> 3, in = sub & 7u;
    unsigned quad = in >> 1, half = in & 1u;
    const uint8_t *ql = block + (size_t)chunk * 64u
                      + (size_t)(quad & 1u) * 32u + (size_t)half * 16u;
    const uint8_t *qh = block + 128u + (size_t)chunk * 32u
                      + (size_t)half * 16u;
    uint8x16_t qlb = vld1q_u8(ql);
    uint8x16_t qhb = vld1q_u8(qh);
    uint8x16_t nib;
    uint8x16_t two;
    int8x16_t q;
    float32x4_t w[4];
    float64x2_t a[8];
    int i;

    if (quad >= 2u) {
        nib = vshrq_n_u8(qlb, 4);
    } else {
        nib = vandq_u8(qlb, vdupq_n_u8(0x0fu));
    }
    // Сдвиг qh — переменный (0/2/4/6), поэтому vshlq_u8 с отрицательным
    // сдвигом, а не vshrq_n_u8 (у того сдвиг — константа компиляции).
    two = vandq_u8(vshlq_u8(qhb, vdupq_n_s8(-(int)(quad * 2u))),
                   vdupq_n_u8(0x03u));
    q = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(nib, vshlq_n_u8(two, 4))),
                 vdupq_n_s8(32));
    q4_i8x16_to_f32x4(q, w);
    for (i = 0; i < 8; i++) a[i] = vdupq_n_f64(0.0);
    for (i = 0; i < 4; i++) {
        a[i * 2] = vfmaq_f64(a[i * 2], vcvt_f64_f32(vget_low_f32(w[i])),
                             vld1q_f64(xp + 4 * i));
        a[i * 2 + 1] = vfmaq_f64(a[i * 2 + 1],
                                 vcvt_f64_f32(vget_high_f32(w[i])),
                                 vld1q_f64(xp + 4 * i + 2));
    }
    return vaddvq_f64(a[0]) + vaddvq_f64(a[1]) + vaddvq_f64(a[2])
         + vaddvq_f64(a[3]) + vaddvq_f64(a[4]) + vaddvq_f64(a[5])
         + vaddvq_f64(a[6]) + vaddvq_f64(a[7]);
}

static double q6k_row_dot_neon(const uint8_t *blocks, int64_t base, int cols,
                               const double *xd) {
    double acc = 0.0;
    int64_t block = base / (int64_t)VTM_Q4K_BLOCK_VALUES;
    int nblocks = cols / (int)VTM_Q4K_BLOCK_VALUES;
    int b;

    for (b = 0; b < nblocks; b++) {
        const uint8_t *bp = blocks
            + (size_t)(block + b) * (size_t)VTM_Q6K_BLOCK_BYTES;
        double d = (double)q4k_f16(bp + 208u);
        const double *xp = xd + (size_t)b * VTM_Q4K_BLOCK_VALUES;
        unsigned s;
        for (s = 0; s < 16u; s++) {
            int sc = (int)(int8_t)bp[192u + s];
            acc += (d * (double)sc) * q6k_dot16(bp, s, xp + (size_t)s * 16u);
        }
    }
    return acc;
}

// f32-вариант q6k_dot16 (задача 0167): 16 значений под-блока, произведение
// копится в f32 (4 аккумулятора). Ошибка — как у Q4_K на f32.
static inline float q6k_dot16_f32(const uint8_t *block, unsigned sub,
                                  const float *xp) {
    unsigned chunk = sub >> 3, in = sub & 7u;
    unsigned quad = in >> 1, half = in & 1u;
    const uint8_t *ql = block + (size_t)chunk * 64u
                      + (size_t)(quad & 1u) * 32u + (size_t)half * 16u;
    const uint8_t *qh = block + 128u + (size_t)chunk * 32u
                      + (size_t)half * 16u;
    uint8x16_t qlb = vld1q_u8(ql);
    uint8x16_t qhb = vld1q_u8(qh);
    uint8x16_t nib;
    uint8x16_t two;
    int8x16_t q;
    float32x4_t w[4];
    float32x4_t a0, a1, a2, a3, s;

    if (quad >= 2u) {
        nib = vshrq_n_u8(qlb, 4);
    } else {
        nib = vandq_u8(qlb, vdupq_n_u8(0x0fu));
    }
    two = vandq_u8(vshlq_u8(qhb, vdupq_n_s8(-(int)(quad * 2u))),
                   vdupq_n_u8(0x03u));
    q = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(nib, vshlq_n_u8(two, 4))),
                 vdupq_n_s8(32));
    q4_i8x16_to_f32x4(q, w);
    a0 = vmulq_f32(w[0], vld1q_f32(xp + 0));
    a1 = vmulq_f32(w[1], vld1q_f32(xp + 4));
    a2 = vmulq_f32(w[2], vld1q_f32(xp + 8));
    a3 = vmulq_f32(w[3], vld1q_f32(xp + 12));
    s = vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
    return vaddvq_f32(s);
}

// Одна строка Q6_K в f32-арифметике (быстрый путь, по умолчанию).
static float q6k_row_dot_f32(const uint8_t *blocks, int64_t base, int cols,
                             const float *x) {
    double acc = 0.0;
    int64_t block = base / (int64_t)VTM_Q4K_BLOCK_VALUES;
    int nblocks = cols / (int)VTM_Q4K_BLOCK_VALUES;
    int b;

    for (b = 0; b < nblocks; b++) {
        const uint8_t *bp = blocks
            + (size_t)(block + b) * (size_t)VTM_Q6K_BLOCK_BYTES;
        float d = q4k_f16(bp + 208u);
        const float *xp = x + (size_t)b * VTM_Q4K_BLOCK_VALUES;
        unsigned s;
        for (s = 0; s < 16u; s++) {
            int sc = (int)(int8_t)bp[192u + s];
            acc += (double)((d * (float)sc)
                            * q6k_dot16_f32(bp, s, xp + (size_t)s * 16u));
        }
    }
    return (float)acc;
}

int vtm_neon_q6k_tmv_rows(float *y, const uint8_t *blocks, const float *x,
                          int r0, int r1, int cols) {
    double *xd;
    int r, i;

    if (!y || !blocks || !x || r0 < 0 || r1 < r0 || cols <= 0) return -1;
    if (cols % (int)VTM_Q4K_BLOCK_VALUES != 0) return -1;  // скалярный эталон
    // Быстрый путь 0167: f32, без копии активаций (Σx у Q6_K не нужна).
    if (!q4k_strict_mode()) {
        for (r = r0; r < r1; r++) {
            int64_t base = (int64_t)r * (int64_t)cols;
            y[r] = q6k_row_dot_f32(blocks, base, cols, x);
        }
        return 0;
    }
    xd = (double *)malloc((size_t)cols * sizeof(double));
    if (!xd) return -1;
    for (i = 0; i < cols; i++) xd[i] = (double)x[i];
    for (r = r0; r < r1; r++) {
        int64_t base = (int64_t)r * (int64_t)cols;
        y[r] = (float)q6k_row_dot_neon(blocks, base, cols, xd);
    }
    free(xd);
    return 0;
}

// --- NEON: батчевый Q4_K/Q6_K (задача 0167) --------------------------------
//
// По образцу 0162: строка W деквантизуется в f32 один раз на пачку
// (q4k_block_to_f32/q6k_block_to_f32 — d·sc и −dmin·m применяются сразу к
// весам), затем 4 токена пачки считаются одним набором f32-FMA: вес —
// broadcast, активации токенов лежат подряд (x[t*ntok + j]).
//
// Арифметика: внутри группы 32 значения — 4 независимых f32-аккумулятора по
// токенам, свёртка в f32 и добавление в double. Масштаб уже в весе, поэтому,
// в отличие от Q4_1 (0162), ни Σx, ни cg-поправки не нужны.
//
// Порядок f32-сложения внутри группы иной, чем у поштучного ядра (там ленты по
// t, здесь — по токенам), поэтому результат не побитовый: на logits 3B
// расхождение — доли 1e-6 (порог 1e-5, проверяет bench_backend --q4k).
// Хвост ntok % 4 (1-3 токена) считается по t в double — как в 0162.

// Блок Q4_K (144 Б) -> 256 весов f32: w = d·sc·q − dmin·m.
static void q4k_block_to_f32(float *dst, const uint8_t *bp) {
    float d = q4k_f16(bp);
    float dmin = q4k_f16(bp + 2);
    unsigned s;

    for (s = 0; s < 8u; s++) {
        unsigned sc, mn;
        float scale, mnv;
        const uint8_t *qs = bp + 16u + (size_t)(s >> 1) * 32u;
        int h;

        q4k_scale_min(bp + 4, s, &sc, &mn);
        // d и sc — f16 и 6 бит: произведение точно в f32 (как в ядре току).
        scale = d * (float)sc;
        mnv = -(dmin * (float)mn);
        for (h = 0; h < 2; h++) {
            uint8x16_t b = vld1q_u8(qs + (size_t)h * 16u);
            uint8x16_t nib = (s & 1u)
                ? vshrq_n_u8(b, 4)
                : vandq_u8(b, vdupq_n_u8(0x0fu));
            float32x4_t w[4];
            float *o = dst + (size_t)s * 32u + (size_t)h * 16u;
            q4_i8x16_to_f32x4(vreinterpretq_s8_u8(nib), w);
            vst1q_f32(o + 0, vfmaq_n_f32(vdupq_n_f32(mnv), w[0], scale));
            vst1q_f32(o + 4, vfmaq_n_f32(vdupq_n_f32(mnv), w[1], scale));
            vst1q_f32(o + 8, vfmaq_n_f32(vdupq_n_f32(mnv), w[2], scale));
            vst1q_f32(o + 12, vfmaq_n_f32(vdupq_n_f32(mnv), w[3], scale));
        }
    }
}

// Блок Q6_K (210 Б) -> 256 весов f32: w = d·sc·(q − 32).
static void q6k_block_to_f32(float *dst, const uint8_t *block) {
    float d = q4k_f16(block + 208u);
    unsigned sub;

    for (sub = 0; sub < 16u; sub++) {
        unsigned chunk = sub >> 3, in = sub & 7u;
        unsigned quad = in >> 1, half = in & 1u;
        const uint8_t *ql = block + (size_t)chunk * 64u
                          + (size_t)(quad & 1u) * 32u + (size_t)half * 16u;
        const uint8_t *qh = block + 128u + (size_t)chunk * 32u
                          + (size_t)half * 16u;
        uint8x16_t qlb = vld1q_u8(ql);
        uint8x16_t qhb = vld1q_u8(qh);
        uint8x16_t nib, two;
        int8x16_t q;
        float32x4_t w[4];
        float mult;
        float *o = dst + (size_t)sub * 16u;

        nib = (quad >= 2u) ? vshrq_n_u8(qlb, 4)
                           : vandq_u8(qlb, vdupq_n_u8(0x0fu));
        two = vandq_u8(vshlq_u8(qhb, vdupq_n_s8(-(int)(quad * 2u))),
                       vdupq_n_u8(0x03u));
        q = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(nib,
                                                  vshlq_n_u8(two, 4))),
                     vdupq_n_s8(32));
        mult = d * (float)(int)(int8_t)block[192u + sub];
        q4_i8x16_to_f32x4(q, w);
        vst1q_f32(o + 0, vmulq_n_f32(w[0], mult));
        vst1q_f32(o + 4, vmulq_n_f32(w[1], mult));
        vst1q_f32(o + 8, vmulq_n_f32(w[2], mult));
        vst1q_f32(o + 12, vmulq_n_f32(w[3], mult));
    }
}

typedef struct {
    float *y;
    const uint8_t *blocks;
    const float *x;
    int rows;
    int cols;
    int ntok;
    int q6;            // 0 — Q4_K, 1 — Q6_K
    float *wrow_all;   // vt * cols: развёрнутая строка на поток
} qk_batch_neon_job;

static void qk_batch_neon_rows(qk_batch_neon_job *j, int tid, int r0, int r1) {
    const int cols = j->cols;
    const int ntok = j->ntok;
    const int nblocks = cols / (int)VTM_Q4K_BLOCK_VALUES;
    float *wrow = j->wrow_all + (size_t)tid * (size_t)cols;
    int r, jj, t;

    for (r = r0; r < r1; r++) {
        const size_t bbase = (size_t)r * (size_t)nblocks;
        float *yrow = j->y + (size_t)r * (size_t)ntok;

        // Строка весов деквантизуется один раз на пачку (главный выигрыш
        // prefill: 256 значений блока дают 16 токенов).
        if (j->q6) {
            int b;
            for (b = 0; b < nblocks; b++)
                q6k_block_to_f32(wrow + (size_t)b * 256u,
                                 j->blocks + (bbase + (size_t)b)
                                     * (size_t)VTM_Q6K_BLOCK_BYTES);
        } else {
            int b;
            for (b = 0; b < nblocks; b++)
                q4k_block_to_f32(wrow + (size_t)b * 256u,
                                 j->blocks + (bbase + (size_t)b)
                                     * (size_t)VTM_Q4K_BLOCK_BYTES);
        }

        // 4 токена пачки за проход: вес — broadcast, токены — дорожки.
        for (jj = 0; jj + 4 <= ntok; jj += 4) {
            double acc[4] = {0.0, 0.0, 0.0, 0.0};
            int j0 = 0;

            for (j0 = 0; j0 < cols; j0 += 32) {
                const float *xp = j->x + (size_t)j0 * (size_t)ntok
                                + (size_t)jj;
                float32x4_t a0 = vdupq_n_f32(0.0f), a1 = a0, a2 = a0, a3 = a0;
                float32x4_t s;

                for (t = 0; t < 32; t += 4) {
                    const float *p = xp + (size_t)t * (size_t)ntok;
                    a0 = vfmaq_n_f32(a0, vld1q_f32(p), wrow[j0 + t + 0]);
                    a1 = vfmaq_n_f32(a1, vld1q_f32(p + ntok),
                                     wrow[j0 + t + 1]);
                    a2 = vfmaq_n_f32(a2, vld1q_f32(p + 2 * ntok),
                                     wrow[j0 + t + 2]);
                    a3 = vfmaq_n_f32(a3, vld1q_f32(p + 3 * ntok),
                                     wrow[j0 + t + 3]);
                }
                s = vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
                acc[0] += (double)vgetq_lane_f32(s, 0);
                acc[1] += (double)vgetq_lane_f32(s, 1);
                acc[2] += (double)vgetq_lane_f32(s, 2);
                acc[3] += (double)vgetq_lane_f32(s, 3);
            }
            yrow[jj + 0] = (float)acc[0];
            yrow[jj + 1] = (float)acc[1];
            yrow[jj + 2] = (float)acc[2];
            yrow[jj + 3] = (float)acc[3];
        }
        // Хвост 1-3 токена: по t в double (как в 0162) — редкий случай.
        for (jj = ntok - ntok % 4; jj < ntok; jj++) {
            double acc = 0.0;
            for (t = 0; t < cols; t++)
                acc += (double)wrow[t]
                     * (double)j->x[(size_t)t * (size_t)ntok + (size_t)jj];
            yrow[jj] = (float)acc;
        }
    }
}

static void qk_batch_neon_worker(void *p, int tid, int r0, int r1) {
    qk_batch_neon_rows((qk_batch_neon_job *)p, tid, r0, r1);
}

static int qk_batch_neon(float *y, const uint8_t *blocks, const float *x,
                         int rows, int cols, int ntok, int q6) {
    qk_batch_neon_job job;
    int vt;

    if (!y || !blocks || !x) return -1;
    if (rows < 0 || cols < 0 || ntok < 0) return -1;
    if (rows == 0 || cols == 0 || ntok == 0) return 0;
    if (cols % (int)VTM_Q4K_BLOCK_VALUES != 0) return -1;
    vt = vtm_threads();
    job.y = y;
    job.blocks = blocks;
    job.x = x;
    job.rows = rows;
    job.cols = cols;
    job.ntok = ntok;
    job.q6 = q6;
    job.wrow_all = (float *)malloc((size_t)vt * (size_t)cols * sizeof(float));
    if (!job.wrow_all) return -1;
    vtm_parallel_rows(rows, (double)rows * (double)cols * (double)ntok,
                      qk_batch_neon_worker, &job);
    free(job.wrow_all);
    return 0;
}

int vtm_neon_q4k_batch(float *y, const uint8_t *blocks, const float *x,
                       int rows, int cols, int ntok) {
    return qk_batch_neon(y, blocks, x, rows, cols, ntok, 0);
}

int vtm_neon_q6k_batch(float *y, const uint8_t *blocks, const float *x,
                       int rows, int cols, int ntok) {
    return qk_batch_neon(y, blocks, x, rows, cols, ntok, 1);
}

// --- NEON: батчевый Q4 TMV/MM (задача 0162) --------------------------------
//
// y[r*N + j] = sum_t W[r,t]*x[t*N + j]. Строка W разворачивается из нибблов
// один раз на пачку (q4_row_weights_f32 — те же точные целые s = q-8, что у
// одноколоночного ядра), а не N раз, и сразу умножается на N токенов: вес —
// скаляр, активации токенов лежат подряд, поэтому на один t уходит один
// broadcast и N/4 FMA. Это и есть выигрыш prefill: вес читается один раз.
//
// Арифметика группы — как у q4_row_dot_neon: 4 независимых f32-аккумулятора
// по t (по 8 слагаемых в каждом), свёртка в f32, затем в double применяется
// масштаб группы и (для Q4_1) cg*Σx. Порядок f32-сложения внутри группы
// иной, чем у поштучного ядра (там ленты — по t, здесь — по токенам), поэтому
// результат не побитовый; на logits 3B Q4 расхождение — доли 1e-6 (порог
// 1e-5, проверяет bench_backend).
//
// Σx по группе НЕ зависит от строки, когда cols кратен 32 (группы идут по
// плоской матрице с одинаковыми границами): он считается один раз на пачку,
// а не на каждую строку. Иначе (cols % 32 != 0) группа пересекает границу
// строк и Σx берётся в строке — в скалярной ветке ниже.
//
// Хвост ntok % 4 (1-3 токена) считается по t в double — как скалярный
// эталон backend_scalar.c: батчевые вызовы с 1-3 токенами редки.

typedef struct {
    float *y;
    const uint8_t *q;
    const float *scales;
    const float *mins;
    const float *x;
    int rows;
    int cols;
    int ntok;
    int asym;
    int aligned;       // cols % VTM_Q4_GROUP_SIZE == 0
    float *wrow_all;   // vt * cols: развёрнутая строка на поток
    double *xsum;      // aligned: (cols/32) * ntok, Σx группы по токенам
} q4_batch_neon_job;

static void q4_batch_neon_rows(q4_batch_neon_job *j, int tid, int r0, int r1) {
    const int ntok = j->ntok;
    const int cols = j->cols;
    float *wrow = j->wrow_all + (size_t)tid * (size_t)cols;
    int r, jj;

    for (r = r0; r < r1; r++) {
        const int64_t base = (int64_t)r * (int64_t)cols;
        float *yrow = j->y + (size_t)r * (size_t)ntok;

        q4_row_weights_f32(wrow, j->q, base, cols);
        for (jj = 0; jj + 4 <= ntok; jj += 4) {
            double acc[4];
            int j0 = 0, g = 0;

            acc[0] = acc[1] = acc[2] = acc[3] = 0.0;
            while (j0 < cols) {
                const int64_t idx = base + (int64_t)j0;
                const unsigned off = (unsigned)idx & 31u;
                // Масштаб/минимум — по ГЛОБАЛЬНОМУ номеру группы в плоской
                // матрице (как у q4_dot_row), а не по локальному: строка r
                // начинается с группы (r*cols)/32.
                const int64_t gidx = idx >> 5;
                const double scale = (double)load_f32_unaligned(j->scales,
                                                               gidx);
                const double minv = j->asym
                    ? (double)load_f32_unaligned(j->mins, gidx) : 0.0;
                const double cg = j->asym ? minv + 8.0 * scale : 0.0;
                int cnt = VTM_Q4_GROUP_SIZE - (int)off;

                if (cnt > cols - j0) cnt = cols - j0;
                if (j->aligned && cnt == VTM_Q4_GROUP_SIZE) {
                    const float *xp = j->x + (size_t)j0 * (size_t)ntok
                                    + (size_t)jj;
                    float32x4_t a0 = vdupq_n_f32(0.0f);
                    float32x4_t a1 = a0, a2 = a0, a3 = a0;
                    float32x4_t s;
                    int t;

                    for (t = 0; t < VTM_Q4_GROUP_SIZE; t += 4) {
                        const float *p = xp + (size_t)t * (size_t)ntok;
                        a0 = vfmaq_n_f32(a0, vld1q_f32(p), wrow[j0 + t + 0]);
                        a1 = vfmaq_n_f32(a1, vld1q_f32(p + ntok),
                                         wrow[j0 + t + 1]);
                        a2 = vfmaq_n_f32(a2, vld1q_f32(p + 2 * ntok),
                                         wrow[j0 + t + 2]);
                        a3 = vfmaq_n_f32(a3, vld1q_f32(p + 3 * ntok),
                                         wrow[j0 + t + 3]);
                    }
                    s = vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3));
                    if (j->asym) {
                        const double *xs = j->xsum
                            + (size_t)g * (size_t)ntok + (size_t)jj;
                        acc[0] += (double)vgetq_lane_f32(s, 0) * scale
                                + xs[0] * cg;
                        acc[1] += (double)vgetq_lane_f32(s, 1) * scale
                                + xs[1] * cg;
                        acc[2] += (double)vgetq_lane_f32(s, 2) * scale
                                + xs[2] * cg;
                        acc[3] += (double)vgetq_lane_f32(s, 3) * scale
                                + xs[3] * cg;
                    } else {
                        acc[0] += (double)vgetq_lane_f32(s, 0) * scale;
                        acc[1] += (double)vgetq_lane_f32(s, 1) * scale;
                        acc[2] += (double)vgetq_lane_f32(s, 2) * scale;
                        acc[3] += (double)vgetq_lane_f32(s, 3) * scale;
                    }
                } else {
                    int k;
                    for (k = 0; k < 4; k++) {
                        double p = 0.0, sx = 0.0;
                        int t;
                        for (t = 0; t < cnt; t++) {
                            const double bx = (double)j->x[
                                (size_t)(j0 + t) * (size_t)ntok
                                + (size_t)jj + (size_t)k];
                            p += (double)wrow[j0 + t] * bx;
                            if (j->asym) sx += bx;
                        }
                        acc[k] += p * scale + (j->asym ? sx * cg : 0.0);
                    }
                }
                j0 += cnt;
                g++;
            }
            yrow[jj + 0] = (float)acc[0];
            yrow[jj + 1] = (float)acc[1];
            yrow[jj + 2] = (float)acc[2];
            yrow[jj + 3] = (float)acc[3];
        }
        // Хвост 1-3 токенов: строка уже развёрнута, считаем по t в double и
        // сворачиваем группы с масштабом — тот же порядок, что у скалярного
        // эталона backend_scalar.c.
        for (jj = ntok - ntok % 4; jj < ntok; jj++) {
            double acc = 0.0;
            int j0 = 0;
            while (j0 < cols) {
                const int64_t idx = base + (int64_t)j0;
                const unsigned off = (unsigned)idx & 31u;
                const int64_t gidx = idx >> 5;
                const double scale = (double)load_f32_unaligned(j->scales,
                                                               gidx);
                const double minv = j->asym
                    ? (double)load_f32_unaligned(j->mins, gidx) : 0.0;
                const double cg = j->asym ? minv + 8.0 * scale : 0.0;
                double p = 0.0, sx = 0.0;
                int cnt = VTM_Q4_GROUP_SIZE - (int)off;
                int t;

                if (cnt > cols - j0) cnt = cols - j0;
                for (t = 0; t < cnt; t++) {
                    const double bx = (double)j->x[
                        (size_t)(j0 + t) * (size_t)ntok + (size_t)jj];
                    p += (double)wrow[j0 + t] * bx;
                    if (j->asym) sx += bx;
                }
                acc += p * scale + (j->asym ? sx * cg : 0.0);
                j0 += cnt;
            }
            yrow[jj] = (float)acc;
        }
    }
}

static void q4_batch_neon_worker(void *p, int tid, int r0, int r1) {
    q4_batch_neon_rows((q4_batch_neon_job *)p, tid, r0, r1);
}

int vtm_neon_q4_batch(float *y, const uint8_t *q, const float *scales,
                      const float *mins, const float *x, int rows, int cols,
                      int ntok, int asym) {
    q4_batch_neon_job job;
    size_t groups;
    int vt, g, jj, t;

    if (!y || !q || !scales || !x) return -1;
    if (rows < 0 || cols < 0 || ntok < 0) return -1;
    if (asym && !mins) return -1;
    if (rows == 0 || cols == 0 || ntok == 0) return 0;
    vt = vtm_threads();
    job.y = y;
    job.q = q;
    job.scales = scales;
    job.mins = mins;
    job.x = x;
    job.rows = rows;
    job.cols = cols;
    job.ntok = ntok;
    job.asym = asym;
    job.aligned = (cols % VTM_Q4_GROUP_SIZE) == 0;
    job.wrow_all = (float *)malloc((size_t)vt * (size_t)cols * sizeof(float));
    if (!job.wrow_all) return -1;
    groups = (size_t)cols / (size_t)VTM_Q4_GROUP_SIZE;
    job.xsum = NULL;
    if (job.aligned && asym) {
        // Σx группы по токенам: одна и та же для всех строк (cols кратен 32).
        job.xsum = (double *)malloc(groups * (size_t)ntok * sizeof(double));
        if (!job.xsum) {
            free(job.wrow_all);
            return -1;
        }
        for (g = 0; g < (int)groups; g++) {
            for (jj = 0; jj < ntok; jj++) {
                double s = 0.0;
                const int64_t row = (int64_t)g * VTM_Q4_GROUP_SIZE;
                for (t = 0; t < VTM_Q4_GROUP_SIZE; t++)
                    s += (double)x[(size_t)(row + t) * (size_t)ntok
                                   + (size_t)jj];
                job.xsum[(size_t)g * (size_t)ntok + (size_t)jj] = s;
            }
        }
    }
    vtm_parallel_rows(rows, (double)rows * (double)cols * (double)ntok,
                      q4_batch_neon_worker, &job);
    free(job.xsum);
    free(job.wrow_all);
    return 0;
}

// --- NEON: Q4 int8-dot (SDOT) для батчевого prefill, задача 0163 -----------
//
// Зачем. Батчевое f32-ядро 0162 убирает повторную распаковку строки, но не
// сами умножения: 4 MAC за инструкцию (4 f32-дорожки). SDOT (ARMv8.2
// FEAT_DotProd) — 16 int8-умножений со сложением в int32 за инструкцию, то
// есть 16 MAC за инструкцию, и вчетверо меньше байт на активации пачки.
//
// Вес Q4 точный: q в 0..15, s = q - 8 целое, |s| <= 8, поэтому int8
// представляет его без ошибки. Ошибка — только от квантования активаций:
//   sx[j] = max_t |x[t*N+j]| / 127,   xq = round(x / sx), |xq| <= 127.
// Квантование — на пачку (по токену), а не на строку: x переиспользуется
// всеми строками матрицы, поэтому это доли процента MAC.
//
// Арифметика (по группе 32 значения, масштаб/минимум группы — как у 0157):
//   w = s*scale + cg,  cg = min + 8*scale (Q4_1; для Q4_0 cg = 0)
//   Σ w*x = scale*Σ s*xq*sx + cg*Σ xq*sx + (остаток квантования)
// Первое слагаемое — SDOT (int32, |Σ| <= 32*8*127 = 32512), второе — Σ xq
// по группе, он НЕ зависит от строки (cols кратен 32), поэтому считается
// один раз в квантовании. Точное значение Σ xq на группу и токен — это
// xsq[g][j] = sx[j]*Σ xq: в строке остаётся vfmaq_n_f32 с cg.
//
// Раскладка под SDOT: дорожка int32 = токен (4 токена за вектор), поэтому
//   b-вектор (16 байт) = [токен 0: 4 веса подряд][токен 1: те же 4 веса]...
//   a-вектор (16 байт) = те же 4 веса, размноженные на 4 дорожки
//                        (vdupq_n_u32 по 4 байтам строки весов);
//   vdotq_s32(acc, a, b)[k] = Σ_{m=0..3} w[u+m]*xq_токена k[u+m],
// то есть одна инструкция считает 4 токена. 32 веса группы = 8 чанков по
// 4 -> 8 SDOT на группу и 4 токена. Веса группы грузятся из int8-строки
// (cols байт на строку, разворачивается один раз на пачку, как у 0162).
//
// Режимы (env VTM_Q4_I8, см. vtm_q4_i8_mode() в backend_scalar.c):
//   1 — один проход int8: относительная ошибка ~1/127 от максимума группы,
//       на логитах 3B Q4 это единицы 1e-2 (замер — в отчёте 0163);
//   2 — два прохода: к остатку r = x - xq*sx применяется ещё один int8
//       с шагом sx/127 (|r| <= sx/2, так что |rq| <= 63). Ошибка падает в
//       ~127 раз (единицы 1e-4), цена — второй SDOT-конвейер на тех же
//       загруженных весах (веса переиспользуются, память не растёт).
// Отказ (-1) — CPU без FEAT_DotProd, cols не кратен 32, нет памяти: тогда
// диспетчер backend_scalar.c идёт в f32-ядро 0162 (и в скалярный эталон).

// sdot без зависимости от -march: на сборке, где заголовок не объявляет
// vdotq_s32 (нет __ARM_FEATURE_DOTPROD), инструкция пишется ассемблером.
// Рантайм-проверка ниже гарантирует, что она не выполнится на CPU без неё.
#if defined(__ARM_FEATURE_DOTPROD)
#define VTM_SDOT(acc, a, b) vdotq_s32((acc), (a), (b))
#else
static inline int32x4_t vtm_sdot_one(int32x4_t acc, int8x16_t a, int8x16_t b) {
    __asm__("sdot %0.4s, %1.16b, %2.16b" : "+w"(acc) : "w"(a), "w"(b));
    return acc;
}
#define VTM_SDOT(acc, a, b) vtm_sdot_one((acc), (a), (b))
#endif

// FEAT_DotProd: 1 — SDOT есть. Кэшируется: вызывается из горячего пути.
// Apple Silicon (любой arm64 macOS) — есть всегда; на Linux спрашиваем
// HWCAP_ASIMDDP. Сборка под другую ОС — доверяем сборке (иначе int8-путь
// там просто не включается).
static int vtm_dotprod_cpu(void) {
    static int cached = -1;
    int v = cached;

    if (v >= 0) return v;
    v = 0;
#if defined(__APPLE__)
    {
        // Ключ есть с macOS 11.2; если его нет (старая ОС), считаем, что SDOT
        // есть: arm64 macOS живёт только на Apple Silicon, где FEAT_DotProd
        // обязателен. Явный 0 от системы — выключаем.
        int x = -1;
        size_t len = sizeof x;
        if (sysctlbyname("hw.optional.arm.FEAT_DotProd", &x, &len, NULL, 0)
            != 0)
            x = -1;
        v = (x != 0) ? 1 : 0;
    }
#elif defined(__linux__)
    {
        unsigned long hw = getauxval(AT_HWCAP);
        if (hw & (1ul << 20)) v = 1;   // HWCAP_ASIMDDP
    }
#else
    v = 1;
#endif
    cached = v;
    return v;
}

// 16 байт Q4 (32 значения GGML Q4_0-раскладки) -> 32 int8 s = q - 8 в
// естественном порядке: байт j — значения j и j+16.
static inline void q4_group_i8(const uint8_t *gp, int8_t *w) {
    uint8x16_t b = vld1q_u8(gp);
    int8x16_t eight = vdupq_n_s8(8);
    int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(b, vdupq_n_u8(0x0fu))),
                            eight);
    int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(b, 4)), eight);
    vst1q_s8(w, lo);
    vst1q_s8(w + 16, hi);
}

// Чтение 4 байт строки весов (выравнивания не требуем, алиасинга нет).
static inline uint32_t q4_i8_load4(const int8_t *p) {
    uint32_t v;
    memcpy(&v, p, sizeof v);
    return v;
}

// 4 int32 (уже в [-127,127]) -> 4 int8 в младших байтах 32-битной дорожки.
static inline int32x2_t q4_i8_pack4(int32x4_t v) {
    int16x4_t w = vqmovn_s32(v);
    return vreinterpret_s32_s8(vqmovn_s16(vcombine_s16(w, w)));
}

typedef struct {
    float *y;
    const uint8_t *q;
    const float *scales;
    const float *mins;
    const float *x;    // активации пачки x[t*N + j]
    int8_t *xq;        // квантованный вход, блоки по 4 токена
    int8_t *xq2;       // остаток квантования (mode 2; иначе NULL)
    float *sx;         // масштаб активаций на токен (stride значений)
    float *xsq;        // sx[j]*Σ xq по локальной группе: [g*stride + j]
    float *xsr;        // sx[j]*Σ xq2 (mode 2)
    int rows;
    int cols;
    int ntok;
    int nblk;          // (ntok + 3)/4
    int stride;        // nblk*4 (память под хвост пачки)
    int ngroups;       // cols/32
    int asym;
    int mode;
    int8_t *wrow_all;  // vt*cols: int8-строка на поток
} q4_i8_job;

// Одна строка: веса Q4 -> int8, затем пачка (16 токенов за проход).
static void q4_i8_row(const q4_i8_job *j, int8_t *w, int r) {
    const int cols = j->cols;
    const int ntok = j->ntok;
    const int nblk = j->nblk;
    const int stride = j->stride;
    const int ngroups = j->ngroups;
    const int asym = j->asym;
    const int mode = j->mode;
    // Проход по строке — 4 блока по 4 токена (16 токенов, как C=16 у графа).
    const int nbmax = 4;
    const uint8_t *gp = j->q + (size_t)r * (size_t)cols / 2;
    const int64_t gbase = (int64_t)r * (int64_t)ngroups;
    float *yrow = j->y + (size_t)r * (size_t)ntok;
    int g, B0, b, k;

    for (g = 0; g < ngroups; g++)
        q4_group_i8(gp + (size_t)g * 16, w + (size_t)g * 32);

    // Внутри группы 8 SDOT идут цепочкой зависимости (каждый следующий ждёт
    // предыдущий). В режиме 1 сумма копится в ДВА независимых int32-
    // аккумулятора (чанки 0-3 и 4-7) и складывается перед cvt — ILP важнее
    // двух лишних vaddq (замер: 0,62-0,72 мс против 0,79 мс у одной цепочки).
    // В режиме 2 плоскостей уже две (a и b), и раздваивать каждую нельзя:
    // 16 int32-аккумуляторов + f32-масштабы не влезают в регистры (замер:
    // сужение прохода до 8 токенов с двумя цепочками давало 1,5 мс против
    // 1,0 мс у одной цепочки на 16 токенов).
    for (B0 = 0; B0 < nblk; B0 += nbmax) {
        const int nb = (nblk - B0) < nbmax ? (nblk - B0) : nbmax;
        int32x4_t a0[4], a1[4], b0[4], b1[4];
        float32x4_t af[4], sxb[4];
        // Блок — 4 токена, то есть 4*cols байт квантованного входа.
        const size_t blk = (size_t)4 * (size_t)cols;
        const int8_t *xp = j->xq + (size_t)B0 * blk;
        const int8_t *xp2 = (mode == 2) ? j->xq2 + (size_t)B0 * blk : NULL;

        for (b = 0; b < nb; b++) {
            af[b] = vdupq_n_f32(0.0f);
            sxb[b] = vld1q_f32(j->sx + (size_t)(B0 + b) * 4);
        }
        for (g = 0; g < ngroups; g++) {
            const float scale = load_f32_unaligned(j->scales, gbase + g);
            const int8_t *wg = w + (size_t)g * 32;
            const int8_t *xg = xp + (size_t)g * 128;
            const int8_t *xg2 = xp2 ? xp2 + (size_t)g * 128 : NULL;
            const float *sq = j->xsq + (size_t)g * (size_t)stride
                            + (size_t)B0 * 4;
            float cg = 0.0f;

            if (asym) {
                cg = load_f32_unaligned(j->mins, gbase + g) + 8.0f * scale;
            }
            for (b = 0; b < nb; b++) {
                a0[b] = vdupq_n_s32(0);
                a1[b] = vdupq_n_s32(0);
                b0[b] = vdupq_n_s32(0);
                b1[b] = vdupq_n_s32(0);
            }
            // ILP: в режиме 1 чанки 0-3 и 4-7 копятся в РАЗНЫЕ аккумуляторы
            // (две независимые цепочки SDOT, каждая по 4), иначе цепочка из
            // 8 зависимых sdot не даёт загрузить оба конвейера. В режиме 2
            // плоскостей уже две (a/b), и раздваивать каждую нельзя: 16
            // int32-аккумуляторов не влезают в регистры (замер — отчёт 0163).
            if (mode == 2) {
                for (k = 0; k < 8; k++) {
                    const int8x16_t av = vreinterpretq_s8_u8(
                        vdupq_n_u32(q4_i8_load4(wg + k * 4)));
                    for (b = 0; b < nb; b++) {
                        a0[b] = VTM_SDOT(a0[b], av,
                                         vld1q_s8(xg + (size_t)b * blk
                                                  + (size_t)k * 16));
                        b0[b] = VTM_SDOT(b0[b], av,
                                         vld1q_s8(xg2 + (size_t)b * blk
                                                  + (size_t)k * 16));
                    }
                }
            } else {
                for (k = 0; k < 4; k++) {
                    const int8x16_t av = vreinterpretq_s8_u8(
                        vdupq_n_u32(q4_i8_load4(wg + k * 4)));
                    for (b = 0; b < nb; b++)
                        a0[b] = VTM_SDOT(a0[b], av,
                                         vld1q_s8(xg + (size_t)b * blk
                                                  + (size_t)k * 16));
                }
                for (k = 4; k < 8; k++) {
                    const int8x16_t av = vreinterpretq_s8_u8(
                        vdupq_n_u32(q4_i8_load4(wg + k * 4)));
                    for (b = 0; b < nb; b++)
                        a1[b] = VTM_SDOT(a1[b], av,
                                         vld1q_s8(xg + (size_t)b * blk
                                                  + (size_t)k * 16));
                }
            }
            for (b = 0; b < nb; b++) {
                af[b] = vfmaq_f32(af[b], vcvtq_f32_s32(vaddq_s32(a0[b], a1[b])),
                                  vmulq_n_f32(sxb[b], scale));
                if (mode == 2) {
                    // sr = sx/127: остаток квантован с шагом в 127 раз мельче.
                    af[b] = vfmaq_f32(af[b],
                                      vcvtq_f32_s32(vaddq_s32(b0[b], b1[b])),
                                      vmulq_n_f32(sxb[b], scale / 127.0f));
                }
                if (asym) {
                    af[b] = vfmaq_n_f32(af[b], vld1q_f32(sq + (size_t)b * 4),
                                        cg);
                    if (mode == 2) {
                        const float *sr = j->xsr
                                        + (size_t)g * (size_t)stride
                                        + (size_t)B0 * 4;
                        af[b] = vfmaq_n_f32(af[b],
                                            vld1q_f32(sr + (size_t)b * 4),
                                            cg / 127.0f);
                    }
                }
            }
        }
        for (b = 0; b < nb; b++) {
            const int j0 = (B0 + b) * 4;
            if (j0 + 4 <= ntok) {
                vst1q_f32(yrow + j0, af[b]);
            } else {
                float tmp[4];
                int q;
                vst1q_f32(tmp, af[b]);
                for (q = 0; j0 + q < ntok; q++) yrow[j0 + q] = tmp[q];
            }
        }
    }
}

static void q4_i8_worker(void *p, int tid, int r0, int r1) {
    q4_i8_job *j = (q4_i8_job *)p;
    int8_t *w = j->wrow_all + (size_t)tid * (size_t)j->cols;
    int r;

    for (r = r0; r < r1; r++) q4_i8_row(j, w, r);
}

// Квантование пачки: sx по токену, xq (шифрованная под SDOT раскладка),
// xsq (Σ xq по локальной группе). mode 2 — ещё остаток xq2/xsr.
static void q4_i8_quant(q4_i8_job *j) {
    const int cols = j->cols;
    const int ntok = j->ntok;
    const int nblk = j->nblk;
    const int stride = j->stride;
    const int ngroups = j->ngroups;
    const int mode = j->mode;
    const float *x = j->x;
    const int32x4_t v127 = vdupq_n_s32(127);
    const int32x4_t vn127 = vdupq_n_s32(-127);
    int b, g, jj, k, t;

    // 1. max|x| по токену (по 4 токена за вектор).
    for (b = 0; b < nblk; b++) {
        const int lo = b * 4;
        float m[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (lo + 4 <= ntok) {
            float32x4_t mv = vdupq_n_f32(0.0f);
            for (t = 0; t < cols; t++)
                mv = vmaxq_f32(mv, vabsq_f32(vld1q_f32(x + (size_t)t
                                                       * (size_t)ntok + lo)));
            vst1q_f32(m, mv);
        } else {
            int q;
            for (t = 0; t < cols; t++) {
                const float *p = x + (size_t)t * (size_t)ntok + lo;
                for (q = 0; lo + q < ntok; q++) {
                    const float a = fabsf(p[q]);
                    if (a > m[q]) m[q] = a;
                }
            }
        }
        for (t = 0; t < 4; t++) {
            const int tok = lo + t;
            // Порог 1e-30: у вырожденных (почти нулевых) токенов шаг
            // квантования был бы денормальным, а 1/sx — inf. Тогда xq = 0,
            // что для x ~ 1e-30 и есть верный ответ с точностью до шума.
            j->sx[tok] = (tok < ntok && m[t] > 1e-30f) ? m[t] / 127.0f : 1.0f;
        }
    }

    // 2. Квантование: по токену, чанками по 4 значения (один store).
    for (jj = 0; jj < ntok; jj++) {
        const float s = j->sx[jj];
        const float inv = 1.0f / s;   // x/s = x*127/max
        const int B = jj >> 2, lane = jj & 3;
        const size_t blk = (size_t)4 * (size_t)cols;
        int8_t *dst = j->xq + (size_t)B * blk + (size_t)lane * 4;
        int8_t *dst2 = j->xq2 ? j->xq2 + (size_t)B * blk + (size_t)lane * 4
                              : NULL;

        for (g = 0; g < ngroups; g++) {
            int32x4_t sq = vdupq_n_s32(0);
            int32x4_t sr = vdupq_n_s32(0);
            int8_t *p = dst + (size_t)g * 128;
            int8_t *p2 = dst2 ? dst2 + (size_t)g * 128 : NULL;

            for (k = 0; k < 8; k++) {
                const int t0 = g * 32 + k * 4;
                const float *xp = x + (size_t)t0 * (size_t)ntok + jj;
                float32x4_t v = vdupq_n_f32(xp[0]);
                int32x4_t qi, ri;

                v = vld1q_lane_f32(xp + (size_t)ntok, v, 1);
                v = vld1q_lane_f32(xp + (size_t)ntok * 2, v, 2);
                v = vld1q_lane_f32(xp + (size_t)ntok * 3, v, 3);
                qi = vcvtnq_s32_f32(vmulq_n_f32(v, inv));
                qi = vmaxq_s32(vminq_s32(qi, v127), vn127);
                vst1_lane_s32((int32_t *)(p + (size_t)k * 16),
                              q4_i8_pack4(qi), 0);
                sq = vaddq_s32(sq, qi);
                if (mode == 2) {
                    // r = x - q*s, |r| <= s/2; шаг остатка s/127, то есть
                    // rq = r*(127/s) = r*inv*127.
                    const float32x4_t r = vfmsq_f32(v, vcvtq_f32_s32(qi),
                                                    vdupq_n_f32(s));
                    ri = vcvtnq_s32_f32(vmulq_n_f32(r, inv * 127.0f));
                    ri = vmaxq_s32(vminq_s32(ri, v127), vn127);
                    vst1_lane_s32((int32_t *)(p2 + (size_t)k * 16),
                                  q4_i8_pack4(ri), 0);
                    sr = vaddq_s32(sr, ri);
                }
            }
            j->xsq[(size_t)g * (size_t)stride + jj] = s * (float)vaddvq_s32(sq);
            if (mode == 2) {
                j->xsr[(size_t)g * (size_t)stride + jj]
                    = s * (float)vaddvq_s32(sr);
            }
        }
    }
}

int vtm_neon_q4_batch_i8(float *y, const uint8_t *q, const float *scales,
                         const float *mins, const float *x, int rows,
                         int cols, int ntok, int asym, int mode) {
    q4_i8_job job;
    size_t xq_sz, xs_sz;
    int vt;

    if (!y || !q || !scales || !x) return -1;
    if (rows < 0 || cols < 0 || ntok < 0) return -1;
    if (asym && !mins) return -1;
    if (rows == 0 || cols == 0 || ntok == 0) return 0;
    if (mode != 2) mode = 1;
    if (ntok < 2) return -1;                 // 1 токен — одноколоночное ядро
    if (cols % VTM_Q4_GROUP_SIZE) return -1; // группа не рвётся по строкам
    if (!vtm_dotprod_cpu()) return -1;       // CPU без FEAT_DotProd

    memset(&job, 0, sizeof job);
    job.y = y;
    job.q = q;
    job.scales = scales;
    job.mins = mins;
    job.rows = rows;
    job.cols = cols;
    job.ntok = ntok;
    job.nblk = (ntok + 3) / 4;
    job.stride = job.nblk * 4;
    job.ngroups = cols / VTM_Q4_GROUP_SIZE;
    job.asym = asym;
    job.mode = mode;
    vt = vtm_threads();
    xq_sz = (size_t)job.stride * (size_t)cols;
    xs_sz = (size_t)job.ngroups * (size_t)job.stride;

    job.xq = (int8_t *)calloc(xq_sz, 1);
    job.sx = (float *)malloc((size_t)job.stride * sizeof(float));
    job.xsq = (float *)malloc(xs_sz * sizeof(float));
    job.wrow_all = (int8_t *)malloc((size_t)vt * (size_t)cols);
    if (mode == 2) {
        job.xq2 = (int8_t *)calloc(xq_sz, 1);
        job.xsr = (float *)malloc(xs_sz * sizeof(float));
    }
    if (!job.xq || !job.sx || !job.xsq || !job.wrow_all
        || (mode == 2 && (!job.xq2 || !job.xsr))) {
        free(job.xq);
        free(job.xq2);
        free(job.sx);
        free(job.xsq);
        free(job.xsr);
        free(job.wrow_all);
        return -1;
    }
    // x нужен и в квантовании; поле в job — const float*, поэтому берём его
    // из аргумента (см. q4_i8_quant).
    job.x = x;
    q4_i8_quant(&job);
    vtm_parallel_rows(rows, (double)rows * (double)cols * (double)ntok,
                      q4_i8_worker, &job);
    free(job.xq);
    free(job.xq2);
    free(job.sx);
    free(job.xsq);
    free(job.xsr);
    free(job.wrow_all);
    return 0;
}

// --- NEON: int8-dot GEMV для декода kind 6/7 (задачи 0169, 0171) -----------
//
// Зачем. Декод — это n == 1: ядро q4k_row_dot_f32 разворачивает каждый
// под-блок 32 значения в f32 (12 инструкций на 16 значений только на
// расширение) и делает 8 FMA, то есть ~30 инструкций на 32 MAC. int8-dot
// заменяет это на 2 SDOT на те же 32 значения: активации квантуются в int8
// один раз на токен, веса Q4_K/Q6_K представимы в int8 ТОЧНО (q ∈ 0..15 для
// Q4_K, q−32 ∈ [−32,31] для Q6_K), поэтому ошибка — только от квантования
// активаций, и её гасят дополнительные плоскости (см. ниже).
//
// Раскладка и масштабы:
//   * активации режутся на под-блоки по 32 значения (как у весов Q4_K);
//     у каждого свой масштаб sx = max|x|/127 и точная Σx (нужна для min):
//     первый шаг — int8-плоскость xq = clamp(round(x/sx)),
//     вторая (planes ≥ 2) — остаток r = x − xq·sx с шагом sx/254
//     (|r| ≤ sx/2 ⇒ |rq| ≤ 127, диапазон int8 исчерпан),
//     третья (planes = 3) — остаток второй плоскости с шагом sx/(254·127);
//   * масштаб под-блока применяется в f32 после cvt: A·sx (Q4_K: A = d·sc,
//     Q6_K: A = d·sc) и A·sx/254 (A·sx/(254·127)) для остатков. 6-битный sc
//     и минимум (mn) — из самого блока, как у f32-ядра;
//   * минимум Q4_K вычитается по ТОЧНОЙ Σx под-блока (не по Σ xq): ошибка
//     квантования на него не влияет;
//   * веса и активации — по 16 байт в SDOT; 32 значения под-блока Q4_K — два
//     SDOT, дорожки int32 — четыре независимых частичных суммы, они копятся в
//     f32-аккумулятор под-блока (у каждого своего — 8 независимых цепочек
//     vfma), а в double сворачиваются раз на 256 значений (как f32-ядро, но с
//     f32 внутри блока).
//
// Стоимость под-блока 32 значения (planes = 3, задача 0171): 2 загрузки
// весов, 2 распаковки нибблов, 6 загрузок активаций, 6 SDOT, 2 vmlaq + 1 add
// (слияние плоскостей в int32), 1 cvt, 1 vfma и 1 выборка масштаба из вектора
// ≈ 23 инструкции; масштабы всех 8 под-блоков блока считаются векторно один
// раз на блок (~4 инструкции на под-блок в пересчёте). Замер
// (bench_backend --q4k-i8, 11008x2048, A/B с версией 0169, min из трёх
// парных прогонов): 66,5 → 80,2 Г MAC/с, planes = 2 — 80,5 → 98,4,
// planes = 1 — 103,4 → 131,8.
//
// Отказ (−1) — CPU без FEAT_DotProd, cols не кратен 256 (строка не
// начинается с границы блока) или нет памяти: диспетчер backend_scalar.c
// считает ту же строку f32-ядром, прогон не падает.

// 32 байта Q4_K (под-блок, low/high нибблы) -> 32 int8 q в естественном
// порядке. q ∈ 0..15, знак не нужен: SDOT знаковый, но 15 < 127.
static inline void q4k_i8_sub(const uint8_t *qs, int high, int8x16_t *w0,
                              int8x16_t *w1) {
    uint8x16_t b0 = vld1q_u8(qs);
    uint8x16_t b1 = vld1q_u8(qs + 16);

    if (high) {
        *w0 = vreinterpretq_s8_u8(vshrq_n_u8(b0, 4));
        *w1 = vreinterpretq_s8_u8(vshrq_n_u8(b1, 4));
    } else {
        uint8x16_t m = vdupq_n_u8(0x0fu);
        *w0 = vreinterpretq_s8_u8(vandq_u8(b0, m));
        *w1 = vreinterpretq_s8_u8(vandq_u8(b1, m));
    }
}

// 16 значений Q6_K под-блока sub -> int8 (q − 32) ∈ [−32,31]: точное
// представление, ошибки от весов нет. Раскладка: ql-ниббл (младший или
// старший — quad >= 2) плюс 2 бита qh со сдвигом quad·2, и всё минус 32.
// Распаковка теперь в теле q6k_i8_row_q6k (там ql/qh грузятся на чанк), а
// формула та же: 16 значений = 2 байта ql-нибблов + 2 бита qh на значение.

typedef struct {
    float *y;
    const uint8_t *blocks;
    const float *x;
    int8_t *xq;        // cols: 1-я int8-плоскость активаций
    int8_t *xr;        // cols: остаток (planes ≥ 2; иначе NULL)
    int8_t *xs;        // cols: остаток второй плоскости (planes = 3)
    float *sx;         // cols/32: масштаб под-блока активаций
    float *xsum;       // cols/32: точная Σx под-блока (нужна Q4_K)
    int rows;
    int cols;
    int q6;            // 0 — Q4_K, 1 — Q6_K
    int planes;        // 1..3
    // 0189: строка весов, уже развёрнутая в int8 (cols байт). Батчевый GEMM
    // разворачивает её ОДИН раз на строку и подставляет сюда, чтобы N токенов
    // пачки не распаковывали нибблы/6-битки заново. NULL (декод) — веса
    // берутся из блоков, как раньше: арифметика и порядок те же, поэтому
    // результат батча побитово равен поштучному ядру.
    const int8_t *wrow;
} qk_i8_job;

// Квантование активаций: один проход по cols на токен (а не на строку).
static void qk_i8_quant(qk_i8_job *j) {
    const int nsub = j->cols / 32;
    const float *x = j->x;
    const int32x4_t v127 = vdupq_n_s32(127);
    const int32x4_t vn127 = vdupq_n_s32(-127);
    int sb, i;

    for (sb = 0; sb < nsub; sb++) {
        const float *p = x + (size_t)sb * 32u;
        int8_t *q1 = j->xq + (size_t)sb * 32u;
        float32x4_t mv = vdupq_n_f32(0.0f);
        float32x4_t sv = vdupq_n_f32(0.0f);
        float m, s, inv;

        for (i = 0; i < 32; i += 4) {
            float32x4_t v = vld1q_f32(p + i);
            mv = vmaxq_f32(mv, vabsq_f32(v));
            sv = vaddq_f32(sv, v);
        }
        m = vmaxvq_f32(mv);
        // Вырожденный (почти нулевой) под-блок: 1/sx был бы inf.
        s = (m > 1e-30f) ? m / 127.0f : 1.0f;
        inv = 1.0f / s;
        j->sx[sb] = s;
        j->xsum[sb] = vaddvq_f32(sv);

        for (i = 0; i < 32; i += 4) {
            float32x4_t v = vld1q_f32(p + i);
            int32x4_t qi = vcvtnq_s32_f32(vmulq_n_f32(v, inv));

            qi = vmaxq_s32(vminq_s32(qi, v127), vn127);
            vst1_lane_s32((int32_t *)(q1 + i), q4_i8_pack4(qi), 0);
            if (j->planes >= 2) {
                float32x4_t r = vfmsq_f32(v, vcvtq_f32_s32(qi),
                                          vdupq_n_f32(s));
                // |r| ≤ sx/2, поэтому шаг остатка sx/254 исчерпывает
                // диапазон int8 (|rq| ≤ 127) ровно вдвое лучше, чем sx/127.
                int32x4_t ri = vcvtnq_s32_f32(vmulq_n_f32(r, inv * 254.0f));

                ri = vmaxq_s32(vminq_s32(ri, v127), vn127);
                vst1_lane_s32((int32_t *)(j->xr + (size_t)sb * 32u + i),
                              q4_i8_pack4(ri), 0);
                if (j->planes >= 3) {
                    float32x4_t r2 = vfmsq_f32(r, vcvtq_f32_s32(ri),
                                               vdupq_n_f32(s * (1.0f / 254.0f)));
                    int32x4_t ri2 = vcvtnq_s32_f32(
                        vmulq_n_f32(r2, inv * (254.0f * 127.0f)));

                    ri2 = vmaxq_s32(vminq_s32(ri2, v127), vn127);
                    vst1_lane_s32(
                        (int32_t *)(j->xs + (size_t)sb * 32u + i),
                        q4_i8_pack4(ri2), 0);
                }
            }
        }
    }
}

// --- int8-декод kind 6/7: масштабы блоков — вектором (задача 0171) -------
//
// Замер аблацией (2048x2048, 1 поток, 3 плоскости): тело без скалярного
// get_scale_min быстрее на 18%, без min/Σx — на 12%, без cvt/vfma — на 14%.
// Значит, из ~32 инструкций на под-блок около 24 уходило на обвязку вокруг
// шести SDOT. Здесь обвязка перенесена в векторы и считается один раз на блок
// (8 под-блоков), а поправки плоскостей складываются в int32 ДО cvt, так что
// на под-блок остаётся один cvt + один vfma вместо трёх.
//
// Точность от слияния плоскостей в int32 не страдает: |i1·32258 + i2·127 + i3|
// <= 246,7 млн у Q4_K и <= 526 млн у Q6_K (int32 держит 2,1 млрд), а ошибка
// округления f32 при cvt — 6e-8 относительно, то есть на уровне f32-суммы.
// Δ против f32-ядра остаётся ~1e-7 (проверяет bench_backend --q4k-i8/--check).

// 4 байта (точные целые 0..63) -> f32x4.
static inline float32x4_t q4k_u8x4_to_f32(uint8x8_t v) {
    return vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(v))));
}

// (sc, mn) восьми под-блоков Q4_K -> четыре f32-вектора. Раскладка 6-битных
// полей — get_scale_min из ggml: sc[0..3]=s[0..3]&0x3f, mn[0..3]=s[4..7]&0x3f,
// sc[4..7]=(s[8..11]&0x0f)|((s[0..3]&0xc0)>>2),
// mn[4..7]=(s[8..11]>>4)|((s[4..7]&0xc0)>>2). Читаются 16 байт с bp+4: 12
// нужных и 4 байта qs того же блока (блок 144 Б), за границу не выходим.
static inline void q4k_scmn_f32(const uint8_t *scales, float32x4_t *sc0,
                                float32x4_t *sc1, float32x4_t *mn0,
                                float32x4_t *mn1) {
    const uint8x16_t a = vld1q_u8(scales);
    const uint8x8_t lo = vget_low_u8(a);       // s0..s7
    const uint8x8_t hi = vget_high_u8(a);      // s8..s15 (нужны 8..11)
    const uint8x8_t m3f = vdup_n_u8(0x3Fu);
    const uint8x8_t m0f = vdup_n_u8(0x0Fu);
    const uint8x8_t mc0 = vdup_n_u8(0xC0u);
    // Байты 0..3 — sc[0..3], байты 4..7 — mn[0..3]: один and на оба.
    const uint8x8_t ns = vand_u8(lo, m3f);
    const uint8x8_t sch = vorr_u8(vand_u8(hi, m0f),
                                  vshr_n_u8(vand_u8(lo, mc0), 2));
    const uint8x8_t mnh = vorr_u8(vshr_n_u8(hi, 4),
                                  vshr_n_u8(vand_u8(vext_u8(lo, lo, 4), mc0),
                                            2));
    const uint8x8_t mnl = vext_u8(ns, vdup_n_u8(0), 4);

    *sc0 = q4k_u8x4_to_f32(ns);
    *sc1 = q4k_u8x4_to_f32(sch);
    *mn0 = q4k_u8x4_to_f32(mnl);
    *mn1 = q4k_u8x4_to_f32(mnh);
}

// Одна строка Q4_K: y = Σ_b Σ_sb (d·sc·Σ q·x − dmin·m·Σx). Под-блоки
// чётные/нечётные копятся в разные f32-аккумуляторы (две независимые цепочки
// vfma), в double сворачиваются раз на 256 значений.
static float qk_i8_row_q4k(const qk_i8_job *j, int r) {
    const int nblocks = j->cols / (int)VTM_Q4K_BLOCK_VALUES;
    const uint8_t *bp = j->blocks
        + (size_t)r * (size_t)nblocks * (size_t)VTM_Q4K_BLOCK_BYTES;
    const int planes = j->planes;
    // Три плоскости складываются в int32 ДО cvt:
    //   it = i1·32258 + i2·127 + i3 (32258 = 254·127, 127 — шаг 2-й плоскости),
    //   y += cvt(it) · d·sc·sx·kf,
    // где kf = 1/32258 (три плоскости), 1/254 (две) или 1 (одна) — ровно
    // i1 + i2/254 + i3/(254·127) из контракта ядра.
    const float kf = (planes >= 3) ? (1.0f / 32258.0f)
                   : (planes == 2) ? (1.0f / 254.0f) : 1.0f;
    double total = 0.0;
    int b, sb;

    // Под-блок 32 значения. S — литерал (0..7): распаковка нибблов и выбор
    // дорожки масштаба/аккумулятора сворачиваются на компиляции, поэтому 8
    // под-блоков тела разворачиваются в один прямой участок без ветвей.
    // Аккумулятор у каждого под-блока свой: 8 независимых цепочек vfma, иначе
    // ILP упирается в латентность fma.
#define Q4K_I8_SUB(S, ACC, AV, LN)                                         \
    do {                                                                   \
        const int8_t *xp = j->xq + (size_t)(sb + (S)) * 32u;               \
        int8x16_t w0, w1;                                                  \
        int32x4_t it;                                                      \
        if (j->wrow) {                                                     \
            const int8_t *ws = j->wrow + (size_t)(sb + (S)) * 32u;         \
            w0 = vld1q_s8(ws);                                             \
            w1 = vld1q_s8(ws + 16u);                                       \
        } else {                                                           \
            q4k_i8_sub(bp + 16u + (size_t)((S) >> 1) * 32u, (S) & 1,       \
                       &w0, &w1);                                          \
        }                                                                  \
        it = VTM_SDOT(VTM_SDOT(vdupq_n_s32(0), w0, vld1q_s8(xp)),          \
                      w1, vld1q_s8(xp + 16));                              \
        if (planes >= 2) {                                                 \
            const int8_t *rp = j->xr + (size_t)(sb + (S)) * 32u;           \
            int32x4_t i2 = VTM_SDOT(VTM_SDOT(vdupq_n_s32(0), w0,           \
                                             vld1q_s8(rp)),                \
                                    w1, vld1q_s8(rp + 16));                \
            if (planes >= 3) {                                             \
                const int8_t *sp = j->xs + (size_t)(sb + (S)) * 32u;       \
                int32x4_t i3 = VTM_SDOT(                                   \
                    VTM_SDOT(vdupq_n_s32(0), w0, vld1q_s8(sp)),            \
                    w1, vld1q_s8(sp + 16));                                \
                it = vmulq_n_s32(it, 32258);                               \
                it = vmlaq_n_s32(it, i2, 127);                             \
                it = vaddq_s32(it, i3);                                    \
            } else {                                                       \
                it = vaddq_s32(vmulq_n_s32(it, 254), i2);                  \
            }                                                              \
        }                                                                  \
        ACC = vfmaq_n_f32(ACC, vcvtq_f32_s32(it),                         \
                          vgetq_lane_f32(AV, LN));                         \
    } while (0)

    for (b = 0; b < nblocks; b++, bp += VTM_Q4K_BLOCK_BYTES) {
        const float d = q4k_f16(bp);
        const float dmin = q4k_f16(bp + 2);
        const float dk = d * kf;

        float32x4_t sc0, sc1, mn0, mn1, aq0, aq1, mq0, mq1;
        float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = acc0, acc2 = acc0;
        float32x4_t acc3 = acc0, acc4 = acc0, acc5 = acc0, acc6 = acc0;
        float32x4_t acc7 = acc0;

        sb = b * 8;
        // Масштабы блока: aq = d·kf·sc·sx (на под-блок), mq = dmin·mn·Σx.
        q4k_scmn_f32(bp + 4, &sc0, &sc1, &mn0, &mn1);
        aq0 = vmulq_n_f32(vmulq_f32(sc0, vld1q_f32(j->sx + sb)), dk);
        aq1 = vmulq_n_f32(vmulq_f32(sc1, vld1q_f32(j->sx + sb + 4)), dk);
        mq0 = vmulq_n_f32(vmulq_f32(mn0, vld1q_f32(j->xsum + sb)), dmin);
        mq1 = vmulq_n_f32(vmulq_f32(mn1, vld1q_f32(j->xsum + sb + 4)), dmin);
        total -= (double)vaddvq_f32(vaddq_f32(mq0, mq1));

        Q4K_I8_SUB(0, acc0, aq0, 0);
        Q4K_I8_SUB(1, acc1, aq0, 1);
        Q4K_I8_SUB(2, acc2, aq0, 2);
        Q4K_I8_SUB(3, acc3, aq0, 3);
        Q4K_I8_SUB(4, acc4, aq1, 0);
        Q4K_I8_SUB(5, acc5, aq1, 1);
        Q4K_I8_SUB(6, acc6, aq1, 2);
        Q4K_I8_SUB(7, acc7, aq1, 3);
#undef Q4K_I8_SUB
        total += (double)vaddvq_f32(
            vaddq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)),
                      vaddq_f32(vaddq_f32(acc4, acc5), vaddq_f32(acc6, acc7))));
    }
    return (float)total;
}

// Одна строка Q6_K: 16 под-блоков по 16 значений, w = d·sc·(q − 32), минимума
// нет. Под-блок 16 значений — ровно один SDOT.
//
// ql и qh переиспользуются: на чанк из 128 значений (8 под-блоков) нужно 4
// загрузки ql (16 байт) и 2 загрузки qh, а не 16 загрузок по 2, как когда
// под-блок достаёт свои байты сам. Сдвиг qh (0/2/4/6) и старший/младший
// ниббл ql определяются номером под-блока — литералы, ветвей нет.
// Масштабы (16 int8) и d·sx — вектором один раз на блок, плоскости
// складываются в int32 до cvt (см. Q4_K выше, там же замер аблацией).
static float qk_i8_row_q6k(const qk_i8_job *j, int r) {
    const int nblocks = j->cols / (int)VTM_Q4K_BLOCK_VALUES;
    const uint8_t *bp = j->blocks
        + (size_t)r * (size_t)nblocks * (size_t)VTM_Q6K_BLOCK_BYTES;
    const int planes = j->planes;
    const float kf = (planes >= 3) ? (1.0f / 32258.0f)
                   : (planes == 2) ? (1.0f / 254.0f) : 1.0f;
    double total = 0.0;
    int b, sb;

    // S — литерал (0..15), QLB/QHB — уже загруженные векторы чанка. SBX/OFF
    // (под-блок активаций 32 значения и его половина), SH (сдвиг qh), HIGH
    // (старший ниббл ql) — выводятся из S на компиляции.
#define Q6K_I8_SUB(S, ACC, AV, LN, QLB, QHB, SH, HIGH)                     \
    do {                                                                   \
        const int sbx = sb + ((S) >> 1);                                   \
        const int off = ((int)(S) & 1) * 16;                               \
        const int8_t *xp = j->xq + (size_t)sbx * 32u + (size_t)off;        \
        int8x16_t w;                                                       \
        if (j->wrow) {                                                     \
            w = vld1q_s8(j->wrow + (size_t)sb * 32u                      \
                                  + (size_t)(S) * 16u);                    \
        } else {                                                           \
            const uint8x16_t nib = (HIGH)                                  \
                ? vshrq_n_u8(QLB, 4)                                       \
                : vandq_u8(QLB, vdupq_n_u8(0x0fu));                        \
            const uint8x16_t two = vandq_u8(                               \
                vshlq_u8(QHB, vdupq_n_s8(-(SH))), vdupq_n_u8(0x03u));      \
            w = vsubq_s8(                                                  \
                vreinterpretq_s8_u8(vorrq_u8(nib, vshlq_n_u8(two, 4))),    \
                vdupq_n_s8(32));                                           \
        }                                                                  \
        int32x4_t it = VTM_SDOT(vdupq_n_s32(0), w, vld1q_s8(xp));          \
        if (planes >= 2) {                                                 \
            const int8_t *rp = j->xr + (size_t)sbx * 32u + (size_t)off;    \
            int32x4_t i2 = VTM_SDOT(vdupq_n_s32(0), w, vld1q_s8(rp));      \
            if (planes >= 3) {                                             \
                const int8_t *sp = j->xs + (size_t)sbx * 32u               \
                                 + (size_t)off;                            \
                int32x4_t i3 = VTM_SDOT(vdupq_n_s32(0), w, vld1q_s8(sp));  \
                it = vmulq_n_s32(it, 32258);                               \
                it = vmlaq_n_s32(it, i2, 127);                             \
                it = vaddq_s32(it, i3);                                    \
            } else {                                                       \
                it = vaddq_s32(vmulq_n_s32(it, 254), i2);                  \
            }                                                              \
        }                                                                  \
        ACC = vfmaq_n_f32(ACC, vcvtq_f32_s32(it),                         \
                          vgetq_lane_f32(AV, LN));                         \
    } while (0)

    for (b = 0; b < nblocks; b++, bp += VTM_Q6K_BLOCK_BYTES) {
        const float dk = q4k_f16(bp + 208u) * kf;
        float32x4_t aq0, aq1, aq2, aq3;
        float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = acc0, acc2 = acc0;
        float32x4_t acc3 = acc0;

        sb = b * 8;
        {
            // 16 int8-масштабов -> 4 f32-вектора; sx под-блока 16 значений
            // совпадает с его половиной в 32-значном под-блоке активаций
            // (S и S+1 делят sx), поэтому sx дублируется попарно (vzip).
            float32x4_t scv[4];
            const float32x4_t lo = vld1q_f32(j->sx + sb);
            const float32x4_t hi = vld1q_f32(j->sx + sb + 4);

            q4_i8x16_to_f32x4(vld1q_s8((const int8_t *)(bp + 192u)), scv);
            aq0 = vmulq_n_f32(vmulq_f32(scv[0], vzip1q_f32(lo, lo)), dk);
            aq1 = vmulq_n_f32(vmulq_f32(scv[1], vzip2q_f32(lo, lo)), dk);
            aq2 = vmulq_n_f32(vmulq_f32(scv[2], vzip1q_f32(hi, hi)), dk);
            aq3 = vmulq_n_f32(vmulq_f32(scv[3], vzip2q_f32(hi, hi)), dk);
        }
        {
            const uint8x16_t ql0 = vld1q_u8(bp + 0u);
            const uint8x16_t ql1 = vld1q_u8(bp + 16u);
            const uint8x16_t ql2 = vld1q_u8(bp + 32u);
            const uint8x16_t ql3 = vld1q_u8(bp + 48u);
            const uint8x16_t ql4 = vld1q_u8(bp + 64u);
            const uint8x16_t ql5 = vld1q_u8(bp + 80u);
            const uint8x16_t ql6 = vld1q_u8(bp + 96u);
            const uint8x16_t ql7 = vld1q_u8(bp + 112u);
            const uint8x16_t qh0 = vld1q_u8(bp + 128u);
            const uint8x16_t qh1 = vld1q_u8(bp + 144u);
            const uint8x16_t qh2 = vld1q_u8(bp + 160u);
            const uint8x16_t qh3 = vld1q_u8(bp + 176u);

            Q6K_I8_SUB(0, acc0, aq0, 0, ql0, qh0, 0, 0);
            Q6K_I8_SUB(1, acc1, aq0, 1, ql1, qh1, 0, 0);
            Q6K_I8_SUB(2, acc2, aq0, 2, ql2, qh0, 2, 0);
            Q6K_I8_SUB(3, acc3, aq0, 3, ql3, qh1, 2, 0);
            Q6K_I8_SUB(4, acc0, aq1, 0, ql0, qh0, 4, 1);
            Q6K_I8_SUB(5, acc1, aq1, 1, ql1, qh1, 4, 1);
            Q6K_I8_SUB(6, acc2, aq1, 2, ql2, qh0, 6, 1);
            Q6K_I8_SUB(7, acc3, aq1, 3, ql3, qh1, 6, 1);
            Q6K_I8_SUB(8, acc0, aq2, 0, ql4, qh2, 0, 0);
            Q6K_I8_SUB(9, acc1, aq2, 1, ql5, qh3, 0, 0);
            Q6K_I8_SUB(10, acc2, aq2, 2, ql6, qh2, 2, 0);
            Q6K_I8_SUB(11, acc3, aq2, 3, ql7, qh3, 2, 0);
            Q6K_I8_SUB(12, acc0, aq3, 0, ql4, qh2, 4, 1);
            Q6K_I8_SUB(13, acc1, aq3, 1, ql5, qh3, 4, 1);
            Q6K_I8_SUB(14, acc2, aq3, 2, ql6, qh2, 6, 1);
            Q6K_I8_SUB(15, acc3, aq3, 3, ql7, qh3, 6, 1);
        }
#undef Q6K_I8_SUB
        total += (double)vaddvq_f32(
            vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
    }
    return (float)total;
}

static void qk_i8_rows(const qk_i8_job *j, int r0, int r1) {
    int r;

    for (r = r0; r < r1; r++)
        j->y[r] = j->q6 ? qk_i8_row_q6k(j, r) : qk_i8_row_q4k(j, r);
}

static void qk_i8_worker(void *p, int tid, int r0, int r1) {
    (void)tid;
    qk_i8_rows((const qk_i8_job *)p, r0, r1);
}

// int8-dot GEMV для декода: y[r] = Σ_j w[r,j]·x[j], w — блоки Q4_K/Q6_K.
// q6 != 0 — Q6_K. planes — число int8-плоскостей активаций (1..3).
// 0 — успех; -1 — отказ (CPU/форма/память), вызывающий идёт в f32-ядро.
int vtm_neon_qk_gemv_i8(float *y, const uint8_t *blocks, const float *x,
                        int rows, int cols, int q6, int planes) {
    qk_i8_job job;

    if (!y || !blocks || !x) return -1;
    if (rows <= 0 || cols <= 0) return -1;
    // Границы блока 256 значений (и, значит, сетки под-блоков по 32) —
    // иначе строка может начаться с середины блока: считает f32-ядро.
    if (cols % (int)VTM_Q4K_BLOCK_VALUES != 0) return -1;
    if (planes < 1) planes = 1;
    if (planes > 3) planes = 3;
    if (!vtm_dotprod_cpu()) return -1;

    memset(&job, 0, sizeof job);
    job.y = y;
    job.blocks = blocks;
    job.x = x;
    job.cols = cols;
    job.q6 = q6 ? 1 : 0;
    job.planes = planes;
    job.xq = (int8_t *)malloc((size_t)cols);
    job.sx = (float *)malloc((size_t)(cols / 32) * sizeof(float));
    job.xsum = (float *)malloc((size_t)(cols / 32) * sizeof(float));
    if (planes >= 2) job.xr = (int8_t *)malloc((size_t)cols);
    if (planes >= 3) job.xs = (int8_t *)malloc((size_t)cols);
    if (!job.xq || !job.sx || !job.xsum
        || (planes >= 2 && !job.xr) || (planes >= 3 && !job.xs)) {
        free(job.xq);
        free(job.xr);
        free(job.xs);
        free(job.sx);
        free(job.xsum);
        return -1;
    }
    qk_i8_quant(&job);
    vtm_parallel_rows(rows, (double)rows * (double)cols, qk_i8_worker, &job);
    free(job.xq);
    free(job.xr);
    free(job.xs);
    free(job.sx);
    free(job.xsum);
    return 0;
}

// --- NEON: int8-dot GEMM для prefill kind 6/7 (задача 0189) ----------------
//
// Зачем. Батчевое f32-ядро 0167 (qk_batch_neon) разворачивает строку весов в
// f32 и на каждый токен пачки делает 2048/4 = 512 FMA на 2048 значений
// (4 MAC за инструкцию). Здесь та же строка разворачивается в int8 (веса
// Q4_K/Q6_K представимы в int8 ТОЧНО) и умножается на 4 токена сразу: SDOT
// даёт 16 MAC за инструкцию, а дорожки int32 — четыре токена. На M1 Pro
// (замер микроядра: 8,2 G SDOT/s против 6,2 G FMLA/s на поток) это ~5x по
// MAC/инструкцию; prefill перестаёт упираться в FMLA.
//
// Точность — ПОБИТОВОЕ равенство с поштучным путём. Это не «приятный бонус»:
// поштучный декод kind 6/7 — тоже int8-dot (0169), и квантование активаций
// дискретно, поэтому расхождение 1e-7 на скрытом состоянии после 36 слоёв
// даёт 1e-4 на логитах (замер: f32-пачка 0167 против поштучного пути —
// max|Δlogits| = 1,15e-4, --prefill-check FAIL). Здесь повторён весь порядок
// операций поштучного ядра (qk_i8_row_q4k/q6k):
//   * та же сетка квантования: под-блок 32 значения, sx = max|x|/127,
//     точная Σx, плоскости sx / sx/254 / sx/(254·127) (режим — как у декода,
//     VTM_Q4_I8: 1/2/3, по умолчанию 3);
//   * те же int32-суммы: it = i1·32258 + i2·127 + i3 (и kf = 1/32258 у d);
//   * тот же порядок f32-сложения: у Q4_K — 8 аккумуляторов (по под-блокам
//     блока) и то же дерево vaddq, у Q6_K — 4 аккумулятора acc[S & 3];
//   * минимум Q4_K вычитается из double до q-слагаемых блока, как в декоде.
// Поэтому y[r*N+j] пачки совпадает с y[r] поштучного ядра бит в бит
// (проверяет bench_backend --q4k / --prefill-check).
//
// Раскладка активаций — как у батчевого Q4 0163 (vtm_neon_q4_batch_i8): блок
// из 4 токенов, под-блок 32 значения = 128 байт = 8 операндов SDOT по 16
// байт, из них дорожка l — 4 значения токена l. Квантование — один проход по
// cols на токен (а не на строку весов), так что его цена не зависит от rows.
//
// Отказ (−1): CPU без FEAT_DotProd, cols не кратен 256, ntok < 2, нет памяти
// или VTM_Q4K_STRICT=1 — диспетчер backend_scalar.c идёт в f32-ядро 0167
// (и дальше в скалярный эталон), прогон не падает.

typedef struct {
    float *y;              // [rows * ntok]
    const uint8_t *blocks;
    int rows, cols, ntok;
    int q6;                // 0 — Q4_K, 1 — Q6_K
    int planes;            // 1..3 — плоскостей int8 у активаций
    int nsub;              // cols/32 — под-блоков сетки квантования
    int ntb;               // (ntok+3)/4 — блоков по 4 токена
    int stride;            // ntb*4 — выравненное число токенов
    int8_t *xq;            // [stride*cols] 1-я плоскость
    int8_t *xr;            // [stride*cols] остаток (planes >= 2)
    int8_t *xs;            // [stride*cols] остаток 2-й плоскости (planes = 3)
    float *sx;             // [nsub*stride] масштаб под-блока по токенам
    float *xsum;           // [nsub*stride] точная Σx под-блока (Q4_K)
    float *xt;             // [ntok*cols] транспонированный вход: [tok*cols+t]
    int8_t *wb_all;        // vt*cols: int8-строка весов на поток
    int8_t *wb4_all;       // vt*cols*4: она же, размноженная по 4 дорожкам
} qk_i8b_job;

// Строка весов -> int8: Q4_K — нибблы 0..15 (как q4k_block_to_f32), Q6_K —
// (q − 32) (как q6k_block_to_f32). Ровно те целые, что раскладывает в f32
// поштучное ядро, поэтому ошибки от весов нет.
static void qk_i8b_expand_row(const uint8_t *blocks, int cols, int q6, int r,
                              int8_t *wb)
{
    const int nblk = cols / (int)VTM_Q4K_BLOCK_VALUES;
    int b, s;

    if (!q6) {
        for (b = 0; b < nblk; b++) {
            const uint8_t *bp = blocks
                + ((size_t)r * (size_t)nblk + (size_t)b)
                  * (size_t)VTM_Q4K_BLOCK_BYTES;
            int8_t *o = wb + (size_t)b * 256u;
            for (s = 0; s < 8; s++) {
                const uint8_t *qs = bp + 16u + (size_t)(s >> 1) * 32u;
                const uint8x16_t m = vdupq_n_u8(0x0fu);
                const uint8x16_t b0 = vld1q_u8(qs);
                const uint8x16_t b1 = vld1q_u8(qs + 16);
                uint8x16_t n0, n1;

                if (s & 1) {
                    n0 = vshrq_n_u8(b0, 4);
                    n1 = vshrq_n_u8(b1, 4);
                } else {
                    n0 = vandq_u8(b0, m);
                    n1 = vandq_u8(b1, m);
                }
                vst1q_s8(o + (size_t)s * 32u, vreinterpretq_s8_u8(n0));
                vst1q_s8(o + (size_t)s * 32u + 16u, vreinterpretq_s8_u8(n1));
            }
        }
        return;
    }
    for (b = 0; b < nblk; b++) {
        const uint8_t *bp = blocks
            + ((size_t)r * (size_t)nblk + (size_t)b)
              * (size_t)VTM_Q6K_BLOCK_BYTES;
        int8_t *o = wb + (size_t)b * 256u;
        for (s = 0; s < 16; s++) {
            const unsigned chunk = (unsigned)s >> 3, in = (unsigned)s & 7u;
            const unsigned quad = in >> 1, half = in & 1u;
            const uint8_t *ql = bp + (size_t)chunk * 64u
                              + (size_t)(quad & 1u) * 32u + (size_t)half * 16u;
            const uint8_t *qh = bp + 128u + (size_t)chunk * 32u
                              + (size_t)half * 16u;
            const uint8x16_t qlb = vld1q_u8(ql);
            const uint8x16_t qhb = vld1q_u8(qh);
            const uint8x16_t nib = (quad >= 2u) ? vshrq_n_u8(qlb, 4)
                                                : vandq_u8(qlb,
                                                           vdupq_n_u8(0x0fu));
            const uint8x16_t two = vandq_u8(
                vshlq_u8(qhb, vdupq_n_s8(-(int)(quad * 2u))),
                vdupq_n_u8(0x03u));
            const int8x16_t q = vsubq_s8(
                vreinterpretq_s8_u8(vorrq_u8(nib, vshlq_n_u8(two, 4))),
                vdupq_n_s8(32));

            vst1q_s8(o + (size_t)s * 16u, q);
        }
    }
}

// 4 значения строки -> 16 байт (четыре дорожки): операнд SDOT, одинаковый для
// всех токенов пачки. Строится один раз на строку (cols/4 итераций), а
// используется ntok/4 раз — иначе на каждый SDOT уходили бы load+dup.
static void qk_i8b_bcast(int8_t *wb4, const int8_t *wb, int cols)
{
    int i;

    for (i = 0; i + 4 <= cols; i += 4) {
        uint32_t v;
        memcpy(&v, wb + i, sizeof v);
        vst1q_s8(wb4 + (size_t)i * 4u, vreinterpretq_s8_u32(vdupq_n_u32(v)));
    }
}

// Квантование активаций пачки: та же арифметика, что qk_i8_quant (декод),
// только раскладка «4 токена в дорожках» и sx/xsum — по токенам.
static void qk_i8b_quant(qk_i8b_job *j)
{
    const int cols = j->cols;
    const int ntok = j->ntok;
    const int nsub = j->nsub;
    const int stride = j->stride;
    const int32x4_t v127 = vdupq_n_s32(127);
    const int32x4_t vn127 = vdupq_n_s32(-127);
    int tok, g, i;

    for (tok = 0; tok < ntok; tok++) {
        const float *p = j->xt + (size_t)tok * (size_t)cols;
        const int tb = tok >> 2, lane = tok & 3;
        int8_t *q1 = j->xq + (size_t)tb * 4u * (size_t)cols
                   + (size_t)lane * 4u;
        int8_t *r1 = j->xr ? j->xr + (size_t)tb * 4u * (size_t)cols
                                  + (size_t)lane * 4u : NULL;
        int8_t *s1 = j->xs ? j->xs + (size_t)tb * 4u * (size_t)cols
                                  + (size_t)lane * 4u : NULL;

        for (g = 0; g < nsub; g++) {
            const float *pp = p + (size_t)g * 32u;
            float32x4_t mv = vdupq_n_f32(0.0f), sv = vdupq_n_f32(0.0f);
            float m, s, inv;
            int8_t *qg = q1 + (size_t)g * 128u;
            int8_t *rg = r1 ? r1 + (size_t)g * 128u : NULL;
            int8_t *sg = s1 ? s1 + (size_t)g * 128u : NULL;

            for (i = 0; i < 32; i += 4) {
                const float32x4_t v = vld1q_f32(pp + i);

                mv = vmaxq_f32(mv, vabsq_f32(v));
                sv = vaddq_f32(sv, v);
            }
            m = vmaxvq_f32(mv);
            // Как в qk_i8_quant: вырожденный под-блок не даёт 1/sx = inf.
            s = (m > 1e-30f) ? m / 127.0f : 1.0f;
            inv = 1.0f / s;
            j->sx[(size_t)g * (size_t)stride + (size_t)tok] = s;
            if (j->xsum)
                j->xsum[(size_t)g * (size_t)stride + (size_t)tok]
                    = vaddvq_f32(sv);

            for (i = 0; i < 32; i += 4) {
                const float32x4_t v = vld1q_f32(pp + i);
                int32x4_t qi = vcvtnq_s32_f32(vmulq_n_f32(v, inv));

                qi = vmaxq_s32(vminq_s32(qi, v127), vn127);
                vst1_lane_s32((int32_t *)(qg + (size_t)(i >> 2) * 16u),
                              q4_i8_pack4(qi), 0);
                if (j->planes >= 2) {
                    const float32x4_t r = vfmsq_f32(v, vcvtq_f32_s32(qi),
                                                    vdupq_n_f32(s));
                    int32x4_t ri = vcvtnq_s32_f32(
                        vmulq_n_f32(r, inv * 254.0f));

                    ri = vmaxq_s32(vminq_s32(ri, v127), vn127);
                    vst1_lane_s32((int32_t *)(rg + (size_t)(i >> 2) * 16u),
                                  q4_i8_pack4(ri), 0);
                    if (j->planes >= 3) {
                        const float32x4_t r2 = vfmsq_f32(
                            r, vcvtq_f32_s32(ri),
                            vdupq_n_f32(s * (1.0f / 254.0f)));
                        int32x4_t ri2 = vcvtnq_s32_f32(
                            vmulq_n_f32(r2, inv * (254.0f * 127.0f)));

                        ri2 = vmaxq_s32(vminq_s32(ri2, v127), vn127);
                        vst1_lane_s32(
                            (int32_t *)(sg + (size_t)(i >> 2) * 16u),
                            q4_i8_pack4(ri2), 0);
                    }
                }
            }
        }
    }
}

// Одна строка: 4 токена пачки за проход (дорожки int32), все под-блоки блока
// в работе — 8 независимых цепочек SDOT. Порядок f32-сложения повторяет
// qk_i8_row_q4k/q6k (см. шапку секции) — результат побитовый.
static void qk_i8b_row(const qk_i8b_job *j, int8_t *wb, int8_t *wb4, int r)
{
    const int cols = j->cols;
    const int q6 = j->q6;
    const int planes = j->planes;
    const int ssz = q6 ? 16 : 32;              // значений в под-блоке весов
    const int nch = ssz / 4;                   // чанков по 4 значения в нём
    const int ngrp = cols / (8 * ssz);         // групп по 8 под-блоков
    // Сетка квантования активаций — под-блоки по 32 значения: у Q4_K это сам
    // под-блок, у Q6_K (16 значений) — его половина, поэтому два соседних
    // под-блока делят sx. Отсюда grp*(8*ssz/32) и s>>q6.
    const int sbase = 8 * ssz / 32;
    const int ntb = j->ntb;
    const int stride = j->stride;
    const int nblk = cols / (int)VTM_Q4K_BLOCK_VALUES;
    const size_t bsz = q6 ? (size_t)VTM_Q6K_BLOCK_BYTES
                          : (size_t)VTM_Q4K_BLOCK_BYTES;
    // kf — как у поштучного ядра: у 3 плоскостей шаг младшей 1/32258.
    const float kf = (planes >= 3) ? (1.0f / 32258.0f)
                   : (planes == 2) ? (1.0f / 254.0f) : 1.0f;
    const uint8_t *bp_row = j->blocks + (size_t)r * (size_t)nblk * bsz;
    float *yrow = j->y + (size_t)r * (size_t)j->ntok;
    int tb, grp, s, k, p;

    qk_i8b_expand_row(j->blocks, cols, q6, r, wb);
    qk_i8b_bcast(wb4, wb, cols);

    for (tb = 0; tb < ntb; tb++) {
        const size_t xoff = (size_t)tb * 4u * (size_t)cols;
        const int8_t *act[3];
        float64x2_t dlo = vdupq_n_f64(0.0), dhi = dlo;
        float32x4_t acc[8];

        act[0] = j->xq + xoff;
        act[1] = j->xr ? j->xr + xoff : j->xq + xoff;
        act[2] = j->xs ? j->xs + xoff : j->xq + xoff;
        for (s = 0; s < 8; s++) acc[s] = vdupq_n_f32(0.0f);

        for (grp = 0; grp < ngrp; grp++) {
            const uint8_t *bp = bp_row + (size_t)(grp >> q6) * bsz;
            const size_t abase = (size_t)grp * (size_t)(8 * ssz / 32) * 128u;
            const size_t wbase = (size_t)grp * (size_t)(8 * ssz);
            float sca[8], mna[8];   // масштаб и минимум 8 под-блоков группы
            float dk, dmin = 0.0f;
            int32x4_t c[8];

            if (!q6) {
                // Q4_K: d, dmin fp16 и 8 (sc, mn) по 6 бит — как в декоде.
                float32x4_t sc0, sc1, mn0, mn1;

                dk = q4k_f16(bp) * kf;
                dmin = q4k_f16(bp + 2);
                q4k_scmn_f32(bp + 4, &sc0, &sc1, &mn0, &mn1);
                vst1q_f32(sca, sc0);
                vst1q_f32(sca + 4, sc1);
                vst1q_f32(mna, mn0);
                vst1q_f32(mna + 4, mn1);
            } else {
                float32x4_t sc6[4];
                // Q6_K: 16 int8-масштабов на блок (по 16 значений), группа —
                // половина блока, то есть 8 масштабов: 0..7 у чётной группы
                // и 8..15 у нечётной.
                const int g2 = (grp & 1) * 2;

                dk = q4k_f16(bp + 208u) * kf;
                q4_i8x16_to_f32x4(vld1q_s8((const int8_t *)(bp + 192u)), sc6);
                vst1q_f32(sca, sc6[g2]);
                vst1q_f32(sca + 4, sc6[g2 + 1]);
            }

            // --- int32-суммы: it = i1·32258 + i2·127 + i3 (как в декоде) ---
            for (p = 0; p < planes; p++) {
                const int8_t *ap = act[p] + abase;
                int32x4_t t[8];

                for (s = 0; s < 8; s++) t[s] = vdupq_n_s32(0);
                for (k = 0; k < nch; k++) {
                    for (s = 0; s < 8; s++) {
                        const size_t wo = (wbase
                            + (size_t)s * (size_t)ssz + (size_t)k * 4u) * 4u;
                        const size_t xo = (size_t)(s >> q6) * 128u
                                        + (size_t)(s & q6) * 64u
                                        + (size_t)k * 16u;

                        t[s] = VTM_SDOT(t[s], vld1q_s8(wb4 + wo),
                                        vld1q_s8(ap + xo));
                    }
                }
                if (p == 0) {
                    for (s = 0; s < 8; s++)
                        c[s] = (planes >= 3) ? vmulq_n_s32(t[s], 32258)
                             : (planes == 2) ? vmulq_n_s32(t[s], 254) : t[s];
                } else if (p == 1) {
                    if (planes >= 3) {
                        for (s = 0; s < 8; s++)
                            c[s] = vmlaq_n_s32(c[s], t[s], 127);
                    } else {
                        for (s = 0; s < 8; s++)
                            c[s] = vaddq_s32(c[s], t[s]);
                    }
                } else {
                    for (s = 0; s < 8; s++)
                        c[s] = vaddq_s32(c[s], t[s]);
                }
            }

            // --- f32: aq = (sc·sx)·dk, затем аккумуляторы блока -----------
            if (!q6) {
                float32x4_t mres[4];
                for (s = 0; s < 4; s++)
                    mres[s] = vdupq_n_f32(0.0f);
                for (s = 0; s < 8; s++) {
                    const float sc = sca[s];
                    const size_t sg = (size_t)(grp * sbase + (s >> q6))
                                    * (size_t)stride + (size_t)tb * 4u;
                    const float32x4_t sxv = vld1q_f32(j->sx + sg);
                    const float32x4_t aqv = vmulq_n_f32(
                        vmulq_n_f32(sxv, sc), dk);
                    // минимум: mq = (mn·Σx)·dmin — как в qk_i8_row_q4k;
                    // mres[i] = mq_i + mq_{i+4} — ровно vaddq(mq0, mq1).
                    const float32x4_t xsv = vld1q_f32(j->xsum + sg);
                    const float32x4_t mv = vmulq_n_f32(
                        vmulq_n_f32(xsv, mna[s]), dmin);

                    acc[s] = vfmaq_f32(acc[s], vcvtq_f32_s32(c[s]), aqv);
                    mres[s & 3] = vaddq_f32(mres[s & 3], mv);
                }
                // total -= (double)Σmin — ДО q-слагаемых блока (как в декоде).
                {
                    float32x4_t ms = vaddq_f32(
                        vaddq_f32(mres[0], mres[1]),
                        vaddq_f32(mres[2], mres[3]));

                    dlo = vsubq_f64(dlo,
                                    vcvt_f64_f32(vget_low_f32(ms)));
                    dhi = vsubq_f64(dhi,
                                    vcvt_f64_f32(vget_high_f32(ms)));
                }
                {
                    float32x4_t qs = vaddq_f32(
                        vaddq_f32(vaddq_f32(acc[0], acc[1]),
                                  vaddq_f32(acc[2], acc[3])),
                        vaddq_f32(vaddq_f32(acc[4], acc[5]),
                                  vaddq_f32(acc[6], acc[7])));

                    dlo = vaddq_f64(dlo, vcvt_f64_f32(vget_low_f32(qs)));
                    dhi = vaddq_f64(dhi, vcvt_f64_f32(vget_high_f32(qs)));
                    for (s = 0; s < 8; s++) acc[s] = vdupq_n_f32(0.0f);
                }
            } else {
                for (s = 0; s < 8; s++) {
                    const float sc = sca[s];
                    const size_t sg = (size_t)(grp * sbase + (s >> q6))
                                    * (size_t)stride + (size_t)tb * 4u;
                    const float32x4_t sxv = vld1q_f32(j->sx + sg);
                    const float32x4_t aqv = vmulq_n_f32(
                        vmulq_n_f32(sxv, sc), dk);

                    acc[s & 3] = vfmaq_f32(acc[s & 3], vcvtq_f32_s32(c[s]),
                                           aqv);
                }
                // Q6_K: блок = 2 группы, double-свёртка по границе блока.
                if (grp & 1) {
                    float32x4_t qs = vaddq_f32(
                        vaddq_f32(acc[0], acc[1]),
                        vaddq_f32(acc[2], acc[3]));

                    dlo = vaddq_f64(dlo, vcvt_f64_f32(vget_low_f32(qs)));
                    dhi = vaddq_f64(dhi, vcvt_f64_f32(vget_high_f32(qs)));
                    for (s = 0; s < 4; s++) acc[s] = vdupq_n_f32(0.0f);
                }
            }
        }

        // y = (float)total: одна свёртка double -> f32 на строку и токен.
        {
            const int n = (j->ntok - tb * 4) < 4 ? (j->ntok - tb * 4) : 4;

            if (n == 4) {
                vst1q_f32(yrow + (size_t)tb * 4u, vcombine_f32(
                    vcvt_f32_f64(dlo), vcvt_f32_f64(dhi)));
            } else {
                float tmp[4];
                int q;

                vst1q_f32(tmp, vcombine_f32(vcvt_f32_f64(dlo),
                                            vcvt_f32_f64(dhi)));
                for (q = 0; q < n; q++) yrow[(size_t)tb * 4u + (size_t)q] = tmp[q];
            }
        }
    }
}

static void qk_i8b_worker(void *p, int tid, int r0, int r1)
{
    const qk_i8b_job *j = (const qk_i8b_job *)p;
    int8_t *wb = j->wb_all + (size_t)tid * (size_t)j->cols;
    int8_t *wb4 = j->wb4_all + (size_t)tid * (size_t)j->cols * 4u;
    int r;

    for (r = r0; r < r1; r++) qk_i8b_row(j, wb, wb4, r);
}

int vtm_neon_qk_batch_i8_fast(float *y, const uint8_t *blocks,
                              const float *x, int rows, int cols,
                              int ntok, int q6, int planes)
{
    qk_i8b_job job;
    size_t xsz, ssz;
    int vt, tok, t;

    if (!y || !blocks || !x) return -1;
    if (rows <= 0 || cols <= 0 || ntok <= 0) return -1;
    if (ntok < 2) return -1;                        // 1 токен — GEMV
    if (cols % (int)VTM_Q4K_BLOCK_VALUES) return -1;  // строка не по блокам
    if (planes < 1) return -1;
    if (planes > 3) planes = 3;
    if (!vtm_dotprod_cpu()) return -1;              // CPU без FEAT_DotProd
    if (q4k_strict_mode()) return -1;               // эталонный режим — f64
    memset(&job, 0, sizeof job);
    job.y = y;
    job.blocks = blocks;
    job.cols = cols;
    job.ntok = ntok;
    job.q6 = q6 ? 1 : 0;
    job.planes = planes;
    job.nsub = cols / 32;
    job.ntb = (ntok + 3) / 4;
    job.stride = job.ntb * 4;
    vt = vtm_threads();
    xsz = (size_t)job.stride * (size_t)cols;
    ssz = (size_t)job.nsub * (size_t)job.stride;

    job.xq = (int8_t *)calloc(xsz, 1);
    if (planes >= 2) job.xr = (int8_t *)calloc(xsz, 1);
    if (planes >= 3) job.xs = (int8_t *)calloc(xsz, 1);
    // calloc: у хвоста пачки (ntok % 4) дорожки за пределами ntok должны быть
    // нулями — их sx иначе была бы мусором, а 0·inf дал бы NaN.
    job.sx = (float *)calloc(ssz, sizeof(float));
    if (!job.q6) job.xsum = (float *)calloc(ssz, sizeof(float));
    job.xt = (float *)malloc((size_t)ntok * (size_t)cols * sizeof(float));
    job.wb_all = (int8_t *)malloc((size_t)vt * (size_t)cols);
    job.wb4_all = (int8_t *)malloc((size_t)vt * (size_t)cols * 4u);
    if (!job.xq || !job.sx || !job.xt || !job.wb_all || !job.wb4_all
        || (planes >= 2 && !job.xr) || (planes >= 3 && !job.xs)
        || (!job.q6 && !job.xsum)) {
        free(job.xq);
        free(job.xr);
        free(job.xs);
        free(job.sx);
        free(job.xsum);
        free(job.xt);
        free(job.wb_all);
        free(job.wb4_all);
        return -1;
    }
    // Транспонирование один раз на вызов: квантование идёт по токену
    // (подряд идущие cols значений), а вход пачки лежит как x[t*ntok + j].
    for (tok = 0; tok < ntok; tok++)
        for (t = 0; t < cols; t++)
            job.xt[(size_t)tok * (size_t)cols + (size_t)t]
                = x[(size_t)t * (size_t)ntok + (size_t)tok];
    qk_i8b_quant(&job);
    vtm_parallel_rows(rows, (double)rows * (double)cols * (double)ntok
                            * (double)planes, qk_i8b_worker, &job);
    free(job.xq);
    free(job.xr);
    free(job.xs);
    free(job.sx);
    free(job.xsum);
    free(job.xt);
    free(job.wb_all);
    free(job.wb4_all);
    return 0;
}

// Точный (побитовый) GEMM для prefill kind 6/7 — второй, основной путь.
//
// «Быстрое» ядро выше считает 4 токена в дорожках int32 и поэтому складывает
// f32-слагаемые в другом порядке, чем поштучный декод: на одну строку это
// 1e-7 отн., но квантование активаций дискретно — после 36 слоёв такое
// расхождение даёт max|Δlogits| ≈ 1,15e-4 (замер --prefill-check на 3B Q4_K;
// ровно столько же давало f32-ядро 0167). Приёмка требует ≤1e-5, поэтому по
// умолчанию работает этот путь: он гоняет ТУ ЖЕ функцию строки, что декод
// (qk_i8_row_q4k/q6k), только веса берёт из строки, развёрнутой в int8 один
// раз на пачку (j->wrow). Арифметика, порядок f32-сложения и double-свёртка
// буквально те же — результат совпадает с поштучным путём бит в бит.
//
// Скорость берётся не из 4-дорожечного SDOT, а из того, что строка весов
// пачки читается и распаковывается ОДИН раз: 64 токена промпта читают 2 ГиБ
// блоков вместо 64×2 ГиБ, а распаковка нибблов идёт не N раз, а один.
// На машинах с холодным page cache это главный выигрыш: поштучный путь
// читает всю матрицу на каждый токен.

typedef struct {
    float *y;                // [rows * ntok]
    const uint8_t *blocks;
    int rows, cols, ntok;
    int q6, planes;
    int nsub;                // cols/32
    qk_i8_job rowj;          // шаблон строки: blocks/cols/planes/q6
    int8_t *xq_all;          // [ntok*cols]
    int8_t *xr_all;          // [ntok*cols] (planes ≥ 2)
    int8_t *xs_all;          // [ntok*cols] (planes = 3)
    float *sx_all;           // [ntok*nsub]
    float *xsum_all;         // [ntok*nsub] (Q4_K)
    float *xt;               // [ntok*cols] транспонированный вход
    int8_t *wrow_all;        // vt*cols: int8-строка весов на поток
} qk_i8p_job;

// Строки делятся между потоками, каждый токен пачки — тем же кодом строки,
// что у декода. Указатели на активации токена подставляются в копию шаблона
// на поток, поэтому гонки за job нет.
static void qk_i8p_worker(void *p, int tid, int r0, int r1)
{
    const qk_i8p_job *pj = (const qk_i8p_job *)p;
    const int cols = pj->cols, ntok = pj->ntok, nsub = pj->nsub;
    qk_i8_job j = pj->rowj;
    int8_t *wrow = pj->wrow_all + (size_t)tid * (size_t)cols;
    int r, tok;

    for (r = r0; r < r1; r++) {
        qk_i8b_expand_row(pj->blocks, cols, pj->q6, r, wrow);
        j.wrow = wrow;
        for (tok = 0; tok < ntok; tok++) {
            const size_t off = (size_t)tok * (size_t)cols;

            j.xq = pj->xq_all + off;
            j.xr = pj->xr_all ? pj->xr_all + off : NULL;
            j.xs = pj->xs_all ? pj->xs_all + off : NULL;
            j.sx = pj->sx_all + (size_t)tok * (size_t)nsub;
            j.xsum = pj->xsum_all
                   ? pj->xsum_all + (size_t)tok * (size_t)nsub : NULL;
            pj->y[(size_t)r * (size_t)ntok + (size_t)tok]
                = pj->q6 ? qk_i8_row_q6k(&j, r) : qk_i8_row_q4k(&j, r);
        }
    }
}

int vtm_neon_qk_batch_i8(float *y, const uint8_t *blocks, const float *x,
                         int rows, int cols, int ntok, int q6, int planes)
{
    qk_i8p_job job;
    qk_i8_job qj;
    size_t xsz, ssz;
    int vt, tok, t;

    if (!y || !blocks || !x) return -1;
    if (rows <= 0 || cols <= 0 || ntok <= 0) return -1;
    if (ntok < 2) return -1;                        // 1 токен — GEMV
    if (cols % (int)VTM_Q4K_BLOCK_VALUES) return -1;  // строка не по блокам
    if (planes < 1) return -1;
    if (planes > 3) planes = 3;
    if (!vtm_dotprod_cpu()) return -1;              // CPU без FEAT_DotProd
    if (q4k_strict_mode()) return -1;               // эталонный режим — f64

    memset(&job, 0, sizeof job);
    job.y = y;
    job.blocks = blocks;
    job.cols = cols;
    job.ntok = ntok;
    job.q6 = q6 ? 1 : 0;
    job.planes = planes;
    job.nsub = cols / 32;
    vt = vtm_threads();
    xsz = (size_t)ntok * (size_t)cols;
    ssz = (size_t)ntok * (size_t)job.nsub;

    job.xq_all = (int8_t *)calloc(xsz, 1);
    if (planes >= 2) job.xr_all = (int8_t *)calloc(xsz, 1);
    if (planes >= 3) job.xs_all = (int8_t *)calloc(xsz, 1);
    job.sx_all = (float *)calloc(ssz, sizeof(float));
    // xsum нужен только Q4_K (минимум), но qk_i8_quant пишет его всегда —
    // поэтому буфер выделяется и для Q6_K (это ntok*nsub float, копейки).
    job.xsum_all = (float *)calloc(ssz, sizeof(float));
    job.xt = (float *)malloc(xsz * sizeof(float));
    job.wrow_all = (int8_t *)malloc((size_t)vt * (size_t)cols);
    if (!job.xq_all || !job.sx_all || !job.xsum_all || !job.xt
        || !job.wrow_all
        || (planes >= 2 && !job.xr_all) || (planes >= 3 && !job.xs_all)) {
        free(job.xq_all);
        free(job.xr_all);
        free(job.xs_all);
        free(job.sx_all);
        free(job.xsum_all);
        free(job.xt);
        free(job.wrow_all);
        return -1;
    }
    // Транспонирование один раз на вызов: квантование идёт по токену.
    for (tok = 0; tok < ntok; tok++)
        for (t = 0; t < cols; t++)
            job.xt[(size_t)tok * (size_t)cols + (size_t)t]
                = x[(size_t)t * (size_t)ntok + (size_t)tok];

    // Квантование — тем же кодом, что у декода (qk_i8_quant), по токенам.
    memset(&qj, 0, sizeof qj);
    qj.cols = cols;
    qj.planes = planes;
    qj.q6 = job.q6;
    for (tok = 0; tok < ntok; tok++) {
        const size_t off = (size_t)tok * (size_t)cols;

        qj.x = job.xt + off;
        qj.xq = job.xq_all + off;
        qj.xr = job.xr_all ? job.xr_all + off : NULL;
        qj.xs = job.xs_all ? job.xs_all + off : NULL;
        qj.sx = job.sx_all + (size_t)tok * (size_t)job.nsub;
        qj.xsum = job.xsum_all
                ? job.xsum_all + (size_t)tok * (size_t)job.nsub : NULL;
        qk_i8_quant(&qj);
    }

    job.rowj.blocks = blocks;
    job.rowj.cols = cols;
    job.rowj.planes = planes;
    job.rowj.q6 = job.q6;
    job.rowj.xsum = job.q6 ? NULL : job.xsum_all;   // Q6_K: минимума нет
    vtm_parallel_rows(rows, (double)rows * (double)cols * (double)ntok
                            * (double)planes, qk_i8p_worker, &job);
    free(job.xq_all);
    free(job.xr_all);
    free(job.xs_all);
    free(job.sx_all);
    free(job.xsum_all);
    free(job.xt);
    free(job.wrow_all);
    return 0;
}

// --- NEON: GEMV по упакованной int8-таблице (kind 4), задача 0170 ----------
//
// Зачем. Таблица словаря 3B (tied lm_head) — [151936, 2048]: в fp32 это
// 1,24 ГиБ на токен декода, 36,6 мс при 34 ГБ/с (упирается в полосу памяти,
// потолок ~27 ток/с). В контейнере она уже лежит int8 (kind 4, 0,31 ГБ):
// загрузчик с 0170 оставляет её упакованной, а этот GEMV считает
//   y[r] = Σ_j q[r,j]·sc[j/128]·x[j],  q ∈ −127..127, sc — fp32 на блок 128.
//
// Точность. Веса int8 представимы в int8 ТОЧНО, поэтому ошибка — только от
// квантования активаций; оно то же, что у декода kind 6/7 (0169): под-блоки
// по 32 значения, у каждого свой sx = max|x|/127, плоскости — int8 (шаг sx)
// и остатки (sx/254, sx/(254·127)). Сетка масштабов таблицы (блок 128) кратна
// под-блокам, поэтому масштаб таблицы применяется к сумме четырёх под-блоков.
//
// Отказ (−1): CPU без FEAT_DotProd, cols не кратен 128 (строка начинается не
// с границы блока масштабов) или нет памяти — диспетчер backend_scalar.c
// (vtm_i8_gemv) идёт переносимым f32-путём, прогон не падает.

typedef struct {
    float *y;
    const int8_t *q;
    const float *scales;
    int8_t *xq;        // cols: 1-я int8-плоскость активаций
    int8_t *xr;        // cols: остаток (planes >= 2; иначе NULL)
    int8_t *xs;        // cols: остаток второй плоскости (planes = 3)
    float *sx;         // cols/32: масштаб под-блока активаций
    int rows;
    int cols;
    int planes;        // 1..3
} i8t_job;

// Одна строка: 128 значений = 4 под-блока по 32, у каждого свой sx; масштаб
// таблицы — один на блок. Тело под-блока развёрнуто литералами (S = 0..3),
// у каждого под-блока свой f32-аккумулятор — 4 независимые цепочки vfma.
static float i8t_row_dot(const i8t_job *j, int r) {
    const int nblk = j->cols / (int)VTM_SCALE_BLOCK_128;
    const int8_t *row = j->q + (size_t)r * (size_t)j->cols;
    const float *sc = j->scales + (size_t)r * (size_t)nblk;
    const float k254 = 1.0f / 254.0f;
    const float k254_127 = 1.0f / (254.0f * 127.0f);
    double total = 0.0;
    int b;

    for (b = 0; b < nblk; b++) {
        const int8_t *bp = row + (size_t)b * VTM_SCALE_BLOCK_128;
        const int sb0 = b * 4;
        float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = acc0;
        float32x4_t acc2 = acc0, acc3 = acc0;

#define I8T_SUB(S, acc)                                                    \
        do {                                                               \
            const int8_t *wp = bp + (size_t)(S) * 32u;                     \
            const int8_t *xp = j->xq + (size_t)(sb0 + (S)) * 32u;          \
            const int8x16_t w0 = vld1q_s8(wp);                             \
            const int8x16_t w1 = vld1q_s8(wp + 16);                        \
            const float sx = j->sx[sb0 + (S)];                             \
            int32x4_t i1 = VTM_SDOT(                                       \
                VTM_SDOT(vdupq_n_s32(0), w0, vld1q_s8(xp)),                \
                w1, vld1q_s8(xp + 16));                                    \
            acc = vfmaq_n_f32(acc, vcvtq_f32_s32(i1), sx);                 \
            if (j->planes >= 2) {                                          \
                const int8_t *rp = j->xr + (size_t)(sb0 + (S)) * 32u;      \
                int32x4_t i2 = VTM_SDOT(                                   \
                    VTM_SDOT(vdupq_n_s32(0), w0, vld1q_s8(rp)),            \
                    w1, vld1q_s8(rp + 16));                                \
                acc = vfmaq_n_f32(acc, vcvtq_f32_s32(i2), sx * k254);      \
                if (j->planes >= 3) {                                      \
                    const int8_t *sp = j->xs + (size_t)(sb0 + (S)) * 32u;  \
                    int32x4_t i3 = VTM_SDOT(                               \
                        VTM_SDOT(vdupq_n_s32(0), w0, vld1q_s8(sp)),        \
                        w1, vld1q_s8(sp + 16));                            \
                    acc = vfmaq_n_f32(acc, vcvtq_f32_s32(i3),              \
                                      sx * k254_127);                      \
                }                                                          \
            }                                                              \
        } while (0)
        I8T_SUB(0, acc0);
        I8T_SUB(1, acc1);
        I8T_SUB(2, acc2);
        I8T_SUB(3, acc3);
#undef I8T_SUB
        total += (double)vaddvq_f32(
                     vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)))
               * (double)sc[b];
    }
    return (float)total;
}

static void i8t_rows(const i8t_job *j, int r0, int r1) {
    int r;

    for (r = r0; r < r1; r++) j->y[r] = i8t_row_dot(j, r);
}

static void i8t_worker(void *p, int tid, int r0, int r1) {
    (void)tid;
    i8t_rows((const i8t_job *)p, r0, r1);
}

// GEMV по упакованной int8-таблице: y[r] = Σ_j q[r,j]·sc[j/128]·x[j].
// planes — число int8-плоскостей активаций (1..3). 0 — успех; -1 — отказ
// (CPU/форма/память), вызывающий обязан пойти переносимым путём.
int vtm_neon_i8_gemv(float *y, const int8_t *q, const float *scales,
                     const float *x, int rows, int cols, int planes) {
    i8t_job job;
    qk_i8_job quant;

    if (!y || !q || !scales || !x) return -1;
    if (rows <= 0 || cols <= 0) return -1;
    if (cols % (int)VTM_SCALE_BLOCK_128 != 0) return -1;
    if (planes < 1) planes = 1;
    if (planes > 3) planes = 3;
    if (!vtm_dotprod_cpu()) return -1;

    memset(&job, 0, sizeof job);
    job.y = y;
    job.q = q;
    job.scales = scales;
    job.cols = cols;
    job.planes = planes;
    job.xq = (int8_t *)malloc((size_t)cols);
    job.sx = (float *)malloc((size_t)(cols / 32) * sizeof(float));
    if (planes >= 2) job.xr = (int8_t *)malloc((size_t)cols);
    if (planes >= 3) job.xs = (int8_t *)malloc((size_t)cols);
    if (!job.xq || !job.sx
        || (planes >= 2 && !job.xr) || (planes >= 3 && !job.xs)) {
        free(job.xq);
        free(job.xr);
        free(job.xs);
        free(job.sx);
        return -1;
    }
    // Активации квантуются тем же кодом, что декод kind 6/7 (0169): сетка
    // под-блоков 32 значения и запасная точная Σx (нужна только Q4_K).
    memset(&quant, 0, sizeof quant);
    quant.x = x;
    quant.cols = cols;
    quant.planes = planes;
    quant.xq = job.xq;
    quant.xr = job.xr;
    quant.xs = job.xs;
    quant.sx = job.sx;
    quant.xsum = (float *)malloc((size_t)(cols / 32) * sizeof(float));
    if (!quant.xsum) {
        free(job.xq);
        free(job.xr);
        free(job.xs);
        free(job.sx);
        return -1;
    }
    qk_i8_quant(&quant);
    free(quant.xsum);

    vtm_parallel_rows(rows, (double)rows * (double)cols, i8t_worker, &job);
    free(job.xq);
    free(job.xr);
    free(job.xs);
    free(job.sx);
    return 0;
}


// MM с тритовой A. n == 1 — это фактически TMV по одной строке (tied
// lm_head: [vocab, hidden] * [hidden, 1]), поэтому идём тем же путём, что
// и TMV. Для n > 1 — rank-1 обновление по столбцам: 16 значений t копим в
// f32, складываем в double-аккумулятор (точность как у скаляра).
//
// Многопоточность: строки m делятся между потоками, у каждого свой
// double-аккумулятор на n столбцов (acc_all + tid*n). Каждая строка
// считается тем же кодом и в том же порядке, поэтому результат от числа
// потоков не зависит.
typedef struct {
    float *y;
    const uint32_t *words;
    const float *scales;
    const float *b;
    int k;
    int n;
    int64_t sb;
    int scale_block;
    double *acc_all;   // vt * n; NULL при n == 1 (аккумулятор не нужен)
} mm_neon_job;

static void mm_trit_neon_rows(mm_neon_job *j, int tid, int r0, int r1) {
    const int k = j->k;
    const int n = j->n;
    const float *b = j->b;
    const float *scales = j->scales;
    const uint32_t *words = j->words;
    const int64_t sb = j->sb;
    const int scale_block = j->scale_block;
    float *y = j->y;
    double *acc;
    int r;

    if (n == 1) {
        for (r = r0; r < r1; r++)
            y[r] = (float)row_dot_trit(words, scales, b, (int64_t)r * k, k,
                                       scale_block);
        return;
    }

    acc = j->acc_all + (size_t)tid * (size_t)n;
    for (r = r0; r < r1; r++) {
        const int64_t base = (int64_t)r * (int64_t)k;
        const int64_t end = base + (int64_t)k;
        int8_t buf[VTM_SCALE_BLOCK_160 + 64];
        int8_t *tp = buf + 32;
        int64_t s;
        int c;

        memset(acc, 0, (size_t)n * sizeof(double));
        for (s = (base / sb) * sb; s < end; s += sb) {
            int64_t e = s + sb;
            int64_t st;
            int64_t wi, wi_end;
            int len, t;
            int64_t lo;
            double scale;
            if (e > end) e = end;
            st = s < base ? base : s;
            wi = st / VTM_TRITS_PER_WORD;
            wi_end = (e - 1) / VTM_TRITS_PER_WORD;
            for (; wi <= wi_end; wi++)
                unpack20(load_u32_unaligned(words, wi),
                         tp + (ptrdiff_t)(wi * VTM_TRITS_PER_WORD - st));
            len = (int)(e - st);
            lo = st - base;
            scale = (double)load_f32_unaligned(scales, s / sb);

            for (c = 0; c + 4 <= n; c += 4) {
                float32x4_t a = vdupq_n_f32(0.0f);
                t = 0;
                for (; t + 16 <= len; t += 16) {
                    float32x4_t part = vdupq_n_f32(0.0f);
                    int tt;
                    for (tt = 0; tt < 16; tt++) {
                        const float *bp = b
                            + (size_t)(lo + t + tt) * (size_t)n + (size_t)c;
                        part = vfmaq_f32(part, vdupq_n_f32((float)tp[t + tt]),
                                         vld1q_f32(bp));
                    }
                    a = vaddq_f32(a, part);
                }
                acc[c + 0] += (double)vgetq_lane_f32(a, 0) * scale;
                acc[c + 1] += (double)vgetq_lane_f32(a, 1) * scale;
                acc[c + 2] += (double)vgetq_lane_f32(a, 2) * scale;
                acc[c + 3] += (double)vgetq_lane_f32(a, 3) * scale;
                for (; t < len; t++) {
                    float tv = (float)tp[t];
                    const float *bp = b
                        + (size_t)(lo + t) * (size_t)n + (size_t)c;
                    acc[c + 0] += (double)(tv * bp[0]) * scale;
                    acc[c + 1] += (double)(tv * bp[1]) * scale;
                    acc[c + 2] += (double)(tv * bp[2]) * scale;
                    acc[c + 3] += (double)(tv * bp[3]) * scale;
                }
            }
            for (; c < n; c++) {
                double part = 0.0;
                for (t = 0; t < len; t++)
                    part += (double)tp[t]
                          * (double)b[(size_t)(lo + t) * (size_t)n
                                      + (size_t)c];
                acc[c] += part * scale;
            }
        }
        for (c = 0; c < n; c++)
            y[(size_t)r * (size_t)n + (size_t)c] = (float)acc[c];
    }
}

static void mm_trit_neon_worker(void *p, int tid, int r0, int r1) {
    mm_trit_neon_rows((mm_neon_job *)p, tid, r0, r1);
}

static int mm_trit_neon(float *y, const uint8_t *w, const float *scales,
                        const float *b, int m, int k, int n,
                        int scale_block) {
    mm_neon_job job;
    int vt;

    if (!y || !w || !scales || !b) return -1;
    if (m < 0 || k < 0 || n < 0) return -1;
    if (scale_block != (int)VTM_SCALE_BLOCK_128 &&
        scale_block != (int)VTM_SCALE_BLOCK_160) return -1;
    if (m == 0 || n == 0) return 0;
    if (k == 0) {
        memset(y, 0, (size_t)m * (size_t)n * sizeof(float));
        return 0;
    }
    lut_init();
    job.y = y;
    job.words = (const uint32_t *)w;
    job.scales = scales;
    job.b = b;
    job.k = k;
    job.n = n;
    job.sb = (int64_t)scale_block;
    job.scale_block = scale_block;
    job.acc_all = NULL;
    // Аккумуляторы по числу потоков; при отказе пула на один поток хватит
    // одного, лишние слоты просто не используются.
    vt = vtm_threads();
    if (n > 1) {
        job.acc_all = (double *)calloc((size_t)vt * (size_t)n,
                                       sizeof(double));
        if (!job.acc_all) return -1;
    }
    vtm_parallel_rows(m, (double)m * (double)k * (double)n,
                      mm_trit_neon_worker, &job);
    free(job.acc_all);
    return 0;
}

typedef struct {
    float *y;
    const float *a;
    const float *b;
    int k;
    int n;
} mm_f32_neon_job;

static void mm_f32_neon_rows(mm_f32_neon_job *j, int tid, int r0, int r1) {
    const int k = j->k;
    const int n = j->n;
    const float *a = j->a;
    const float *b = j->b;
    float *y = j->y;
    int r, c, t;
    (void)tid;

    for (r = r0; r < r1; r++) {
        const float *ar = a + (size_t)r * (size_t)k;
        float *yr = y + (size_t)r * (size_t)n;
        for (c = 0; c + 4 <= n; c += 4) {
            double acc[4] = {0.0, 0.0, 0.0, 0.0};
            t = 0;
            for (; t + 16 <= k; t += 16) {
                float32x4_t part = vdupq_n_f32(0.0f);
                int tt;
                for (tt = 0; tt < 16; tt++)
                    part = vfmaq_n_f32(part,
                                       vld1q_f32(b + (size_t)(t + tt)
                                                   * (size_t)n + (size_t)c),
                                       ar[t + tt]);
                acc[0] += (double)vgetq_lane_f32(part, 0);
                acc[1] += (double)vgetq_lane_f32(part, 1);
                acc[2] += (double)vgetq_lane_f32(part, 2);
                acc[3] += (double)vgetq_lane_f32(part, 3);
            }
            for (; t < k; t++) {
                int cc;
                for (cc = 0; cc < 4; cc++)
                    acc[cc] += (double)ar[t]
                             * (double)b[(size_t)t * (size_t)n
                                         + (size_t)(c + cc)];
            }
            yr[c + 0] = (float)acc[0];
            yr[c + 1] = (float)acc[1];
            yr[c + 2] = (float)acc[2];
            yr[c + 3] = (float)acc[3];
        }
        for (; c < n; c++) {
            double acc = 0.0;
            for (t = 0; t < k; t++)
                acc += (double)ar[t] * (double)b[(size_t)t * (size_t)n
                                                + (size_t)c];
            yr[c] = (float)acc;
        }
    }
}

static void mm_f32_neon_worker(void *p, int tid, int r0, int r1) {
    mm_f32_neon_rows((mm_f32_neon_job *)p, tid, r0, r1);
}

static int mm_f32_neon(float *y, const float *a, const float *b,
                       int m, int k, int n) {
    mm_f32_neon_job job;
    if (!y || !a || !b) return -1;
    if (m < 0 || k < 0 || n < 0) return -1;
    if (m == 0 || n == 0) return 0;
    if (k == 0) {
        memset(y, 0, (size_t)m * (size_t)n * sizeof(float));
        return 0;
    }
    job.y = y;
    job.a = a;
    job.b = b;
    job.k = k;
    job.n = n;
    vtm_parallel_rows(m, (double)m * (double)k * (double)n,
                      mm_f32_neon_worker, &job);
    return 0;
}

// --- NEON: плотный GEMV (n == 1), задача 0159 ------------------------------
//
// Перенесено из interp.c (dense_gemv, задача 0158) без изменения арифметики.
// Внутри блока VTM_GEMV_CHUNK элементов — 4 f32-ленты (4 независимые цепи
// FMA), блок сворачивается в double, чтобы f32-ошибка не копилась на всей
// строке 2048. Порядок сложений иной, чем у скалярного double-цикла, поэтому
// результат не побитовый: расхождение на logits 3B Q4 ≤ 2e-6 (порог 1e-5).
// Строки делятся пулом vtm_parallel_rows: каждая считается тем же кодом,
// поэтому от числа потоков результат не зависит (побитово).
#define VTM_GEMV_CHUNK 64
// Блок батчевого ядра короче: на каждый токен-лейн приходится
// вдвое меньше f32-слагаемых до свёртки в double — запас по
// точности больше (порог 1e-5 на больших формах).
#define VTM_DENSE_BATCH_CHUNK 32

// a·b: внутри блока — f32-ленты (4 аккумулятора), блок в double. Длина блока
// ограничена, чтобы f32-ошибка не накапливалась на всей строке 2048.
static double gemv_dot_f32_neon(const float *a, const float *b, int n) {
    double sum = 0.0;
    int i = 0;

    for (; i + 4 <= n; ) {
        float32x4_t a0 = vdupq_n_f32(0.0f);
        float32x4_t a1 = a0, a2 = a0, a3 = a0;
        int len = n - i;
        int j = 0;

        if (len > VTM_GEMV_CHUNK) len = VTM_GEMV_CHUNK;
        for (; j + 16 <= len; j += 16) {
            a0 = vfmaq_f32(a0, vld1q_f32(a + i + j),
                           vld1q_f32(b + i + j));
            a1 = vfmaq_f32(a1, vld1q_f32(a + i + j + 4),
                           vld1q_f32(b + i + j + 4));
            a2 = vfmaq_f32(a2, vld1q_f32(a + i + j + 8),
                           vld1q_f32(b + i + j + 8));
            a3 = vfmaq_f32(a3, vld1q_f32(a + i + j + 12),
                           vld1q_f32(b + i + j + 12));
        }
        for (; j + 4 <= len; j += 4)
            a0 = vfmaq_f32(a0, vld1q_f32(a + i + j),
                           vld1q_f32(b + i + j));
        {
            float64x2_t lo = vaddq_f64(vcvt_f64_f32(vget_low_f32(a0)),
                                       vcvt_f64_f32(vget_low_f32(a1)));
            float64x2_t hi = vaddq_f64(vcvt_f64_f32(vget_high_f32(a0)),
                                       vcvt_f64_f32(vget_high_f32(a1)));
            float64x2_t lo2 = vaddq_f64(vcvt_f64_f32(vget_low_f32(a2)),
                                        vcvt_f64_f32(vget_low_f32(a3)));
            float64x2_t hi2 = vaddq_f64(vcvt_f64_f32(vget_high_f32(a2)),
                                        vcvt_f64_f32(vget_high_f32(a3)));
            sum += vgetq_lane_f64(lo, 0) + vgetq_lane_f64(lo, 1)
                 + vgetq_lane_f64(hi, 0) + vgetq_lane_f64(hi, 1)
                 + vgetq_lane_f64(lo2, 0) + vgetq_lane_f64(lo2, 1)
                 + vgetq_lane_f64(hi2, 0) + vgetq_lane_f64(hi2, 1);
        }
        for (; j < len; j++)
            sum += (double)a[i + j] * (double)b[i + j];
        i += len;
    }
    for (; i < n; i++)
        sum += (double)a[i] * (double)b[i];
    return sum;
}

// y[r] = sum_t Wᵀ[r,t]·x[t] при W, хранящейся как [cols, rows]: r — младший
// индекс, поэтому строки выхода читаются подряд и складываются в f32-ленты
// (4 выхода за шаг), блок по t — в double.
static void gemv_f32_neon_transposed_rows(const float *w, const float *x,
                                          float *y, int rows, int cols,
                                          int r0, int r1) {
    int r = r0, t;

    for (; r + 4 <= r1; r += 4) {
        double acc[4] = {0.0, 0.0, 0.0, 0.0};
        int c;

        for (c = 0; c < cols; c += VTM_GEMV_CHUNK) {
            float32x4_t part = vdupq_n_f32(0.0f);
            int end = c + VTM_GEMV_CHUNK;

            if (end > cols) end = cols;
            for (t = c; t < end; t++)
                part = vfmaq_n_f32(part,
                                   vld1q_f32(w + (size_t)t * (size_t)rows
                                             + (size_t)r),
                                   x[t]);
            {
                float64x2_t lo = vcvt_f64_f32(vget_low_f32(part));
                float64x2_t hi = vcvt_f64_f32(vget_high_f32(part));
                acc[0] += vgetq_lane_f64(lo, 0);
                acc[1] += vgetq_lane_f64(lo, 1);
                acc[2] += vgetq_lane_f64(hi, 0);
                acc[3] += vgetq_lane_f64(hi, 1);
            }
        }
        y[r] = (float)acc[0];
        y[r + 1] = (float)acc[1];
        y[r + 2] = (float)acc[2];
        y[r + 3] = (float)acc[3];
    }
    for (; r < r1; r++) {
        double sum = 0.0;
        for (t = 0; t < cols; t++)
            sum += (double)w[(size_t)t * (size_t)rows + (size_t)r]
                 * (double)x[t];
        y[r] = (float)sum;
    }
}

typedef struct {
    float       *y;
    const float *w;
    const float *x;
    int          rows;
    int          cols;
    int          transposed;
} gemv_f32_neon_job;

static void gemv_f32_neon_rows(gemv_f32_neon_job *j, int tid, int r0, int r1) {
    int r;

    (void)tid;
    if (j->transposed) {
        gemv_f32_neon_transposed_rows(j->w, j->x, j->y, j->rows, j->cols,
                                      r0, r1);
        return;
    }
    for (r = r0; r < r1; r++)
        j->y[r] = (float)gemv_dot_f32_neon(
            j->w + (size_t)r * (size_t)j->cols, j->x, j->cols);
}

static void gemv_f32_neon_worker(void *p, int tid, int r0, int r1) {
    gemv_f32_neon_rows((gemv_f32_neon_job *)p, tid, r0, r1);
}

static int gemv_f32_neon(float *y, const float *w, const float *x,
                         int rows, int cols, int transposed) {
    gemv_f32_neon_job job;

    if (!y || !w || !x) return -1;
    if (rows < 0 || cols < 0) return -1;
    if (rows == 0) return 0;
    if (cols == 0) {
        memset(y, 0, (size_t)rows * sizeof(float));
        return 0;
    }
    job.y = y;
    job.w = w;
    job.x = x;
    job.rows = rows;
    job.cols = cols;
    job.transposed = transposed;
    vtm_parallel_rows(rows, (double)rows * (double)cols,
                      gemv_f32_neon_worker, &job);
    return 0;
}

// --- NEON: батчевый плотный TMV/MM (задача 0159) ---------------------------
//
// y[r*N + j] = sum_t W[r,t] · x[t*N + j] (transposed == 0) или
// W[t,r] (transposed != 0). По вектору — токены пачки: активации лежат
// подряд (x[t*N .. t*N+N)), вес W[r,t] — скаляр, поэтому на один t уходит
// один broadcast и N/4 FMA. Блок VTM_DENSE_BATCH_CHUNK по t копится в
// f32-лентах и сворачивается в double — как в GEMV выше; расхождение с
// батчевым double-эталоном ≤1e-5 (bench_backend --check).
//
// Разбиение по строкам: каждая строка считается тем же кодом, поэтому от
// числа потоков результат не зависит (побитово).
typedef struct {
    float       *y;
    const float *w;
    const float *x;
    int          rows;
    int          cols;
    int          ntok;
    int          transposed;
    double      *acc_all;   // vt * ntok
} dense_batch_neon_job;

static void dense_batch_neon_rows(dense_batch_neon_job *j, int tid, int r0,
                                  int r1) {
    const int ntok = j->ntok;
    const int cols = j->cols;
    const int rows = j->rows;
    const float *w = j->w;
    const float *x = j->x;
    double *acc = j->acc_all + (size_t)tid * (size_t)ntok;
    int r, jj, t, c;

    for (r = r0; r < r1; r++) {
        memset(acc, 0, (size_t)ntok * sizeof(double));
        // 16 токенов за шаг: четыре f32-ленты по 4.
        for (jj = 0; jj + 16 <= ntok; jj += 16) {
            const float *base = x + (size_t)jj;
            for (c = 0; c < cols; c += VTM_DENSE_BATCH_CHUNK) {
                float32x4_t a0 = vdupq_n_f32(0.0f);
                float32x4_t a1 = a0, a2 = a0, a3 = a0;
                int end = c + VTM_DENSE_BATCH_CHUNK;

                if (end > cols) end = cols;
                for (t = c; t < end; t++) {
                    const float *xp = base + (size_t)t * (size_t)ntok;
                    float wv = j->transposed
                             ? w[(size_t)t * (size_t)rows + (size_t)r]
                             : w[(size_t)r * (size_t)cols + (size_t)t];
                    a0 = vfmaq_n_f32(a0, vld1q_f32(xp), wv);
                    a1 = vfmaq_n_f32(a1, vld1q_f32(xp + 4), wv);
                    a2 = vfmaq_n_f32(a2, vld1q_f32(xp + 8), wv);
                    a3 = vfmaq_n_f32(a3, vld1q_f32(xp + 12), wv);
                }
                acc[jj + 0] += (double)vgetq_lane_f32(a0, 0);
                acc[jj + 1] += (double)vgetq_lane_f32(a0, 1);
                acc[jj + 2] += (double)vgetq_lane_f32(a0, 2);
                acc[jj + 3] += (double)vgetq_lane_f32(a0, 3);
                acc[jj + 4] += (double)vgetq_lane_f32(a1, 0);
                acc[jj + 5] += (double)vgetq_lane_f32(a1, 1);
                acc[jj + 6] += (double)vgetq_lane_f32(a1, 2);
                acc[jj + 7] += (double)vgetq_lane_f32(a1, 3);
                acc[jj + 8] += (double)vgetq_lane_f32(a2, 0);
                acc[jj + 9] += (double)vgetq_lane_f32(a2, 1);
                acc[jj + 10] += (double)vgetq_lane_f32(a2, 2);
                acc[jj + 11] += (double)vgetq_lane_f32(a2, 3);
                acc[jj + 12] += (double)vgetq_lane_f32(a3, 0);
                acc[jj + 13] += (double)vgetq_lane_f32(a3, 1);
                acc[jj + 14] += (double)vgetq_lane_f32(a3, 2);
                acc[jj + 15] += (double)vgetq_lane_f32(a3, 3);
            }
        }
        // 8 токенов — две ленты, 4 — одна; хвост 1-3 считаем по столбцам.
        for (; jj + 8 <= ntok; jj += 8) {
            const float *base = x + (size_t)jj;
            for (c = 0; c < cols; c += VTM_DENSE_BATCH_CHUNK) {
                float32x4_t a0 = vdupq_n_f32(0.0f), a1 = a0;
                int end = c + VTM_DENSE_BATCH_CHUNK;

                if (end > cols) end = cols;
                for (t = c; t < end; t++) {
                    const float *xp = base + (size_t)t * (size_t)ntok;
                    float wv = j->transposed
                             ? w[(size_t)t * (size_t)rows + (size_t)r]
                             : w[(size_t)r * (size_t)cols + (size_t)t];
                    a0 = vfmaq_n_f32(a0, vld1q_f32(xp), wv);
                    a1 = vfmaq_n_f32(a1, vld1q_f32(xp + 4), wv);
                }
                acc[jj + 0] += (double)vgetq_lane_f32(a0, 0);
                acc[jj + 1] += (double)vgetq_lane_f32(a0, 1);
                acc[jj + 2] += (double)vgetq_lane_f32(a0, 2);
                acc[jj + 3] += (double)vgetq_lane_f32(a0, 3);
                acc[jj + 4] += (double)vgetq_lane_f32(a1, 0);
                acc[jj + 5] += (double)vgetq_lane_f32(a1, 1);
                acc[jj + 6] += (double)vgetq_lane_f32(a1, 2);
                acc[jj + 7] += (double)vgetq_lane_f32(a1, 3);
            }
        }
        for (; jj + 4 <= ntok; jj += 4) {
            const float *base = x + (size_t)jj;
            for (c = 0; c < cols; c += VTM_DENSE_BATCH_CHUNK) {
                float32x4_t a0 = vdupq_n_f32(0.0f);
                int end = c + VTM_DENSE_BATCH_CHUNK;

                if (end > cols) end = cols;
                for (t = c; t < end; t++) {
                    const float *xp = base + (size_t)t * (size_t)ntok;
                    float wv = j->transposed
                             ? w[(size_t)t * (size_t)rows + (size_t)r]
                             : w[(size_t)r * (size_t)cols + (size_t)t];
                    a0 = vfmaq_n_f32(a0, vld1q_f32(xp), wv);
                }
                acc[jj + 0] += (double)vgetq_lane_f32(a0, 0);
                acc[jj + 1] += (double)vgetq_lane_f32(a0, 1);
                acc[jj + 2] += (double)vgetq_lane_f32(a0, 2);
                acc[jj + 3] += (double)vgetq_lane_f32(a0, 3);
            }
        }
        for (; jj < ntok; jj++) {
            double sum = 0.0;
            for (t = 0; t < cols; t++) {
                double wv = j->transposed
                          ? (double)w[(size_t)t * (size_t)rows + (size_t)r]
                          : (double)w[(size_t)r * (size_t)cols + (size_t)t];
                sum += wv * (double)x[(size_t)t * (size_t)ntok
                                      + (size_t)jj];
            }
            acc[jj] += sum;
        }
        for (jj = 0; jj < ntok; jj++)
            j->y[(size_t)r * (size_t)ntok + (size_t)jj] = (float)acc[jj];
    }
}

static void dense_batch_neon_worker(void *p, int tid, int r0, int r1) {
    dense_batch_neon_rows((dense_batch_neon_job *)p, tid, r0, r1);
}

static int dense_batch_neon(float *y, const float *w, const float *x,
                            int rows, int cols, int ntok, int transposed) {
    dense_batch_neon_job job;
    int vt;

    if (!y || !w || !x) return -1;
    if (rows < 0 || cols < 0 || ntok < 0) return -1;
    if (rows == 0 || cols == 0 || ntok == 0) return 0;
    vt = vtm_threads();
    job.acc_all = (double *)calloc((size_t)vt * (size_t)ntok,
                                   sizeof(double));
    if (!job.acc_all) return -1;
    job.y = y;
    job.w = w;
    job.x = x;
    job.rows = rows;
    job.cols = cols;
    job.ntok = ntok;
    job.transposed = transposed;
    vtm_parallel_rows(rows, (double)rows * (double)cols * (double)ntok,
                      dense_batch_neon_worker, &job);
    free(job.acc_all);
    return 0;
}

// MM-обёртка без transposed — под общий тип указателя в simd_kernels.
static int dense_batch_neon_mm(float *y, const float *a, const float *b,
                               int m, int k, int ntok) {
    return dense_batch_neon(y, a, b, m, k, ntok, 0);
}

// --- NEON: батчевые TMV/MM (задача 0122) -----------------------------------
//
// y[r*N + j] = sum_t trit(W)[r,t] * x[t*N + j]. В отличие от row_dot_trit
// (одна строка × один вектор) здесь по вектору идёт не строка весов, а N
// токенов: трит W[r,t] — скаляр, x[t*N .. t*N+N) — N активаций. Поэтому
// слово pack20 распаковывается и переводится в f32 один раз на строку и
// блок (а не N раз), а веса читаются из памяти один раз на N токенов — это
// и есть выигрыш батчевого prefill.
//
// Арифметика блока — как у скалярного батчевого пути: f32-аккумуляторы
// (4 ленты на 4 токена, 4 независимых цепи по t), затем сумма блока в
// double, умножение на масштаб блока и сложение блоков в double. От
// одноколоночного row_dot_trit отличается только порядком f32-сложения
// внутри блока — расхождение с батчевым скаляром ≤1e-5 (проверка в
// tests/vtm/bench_backend.c).

// len тритов (int8) -> f32. 16 за шаг: int8 -> int16 -> int32 -> f32.
static inline void trits_to_f32(const int8_t *t, float *out, int len) {
    int i = 0;
    for (; i + 16 <= len; i += 16) {
        int8x16_t b = vld1q_s8(t + i);
        int16x8_t lo = vmovl_s8(vget_low_s8(b));
        int16x8_t hi = vmovl_s8(vget_high_s8(b));
        vst1q_f32(out + i + 0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))));
        vst1q_f32(out + i + 4, vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))));
        vst1q_f32(out + i + 8, vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))));
        vst1q_f32(out + i + 12, vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))));
    }
    for (; i < len; i++) out[i] = (float)t[i];
}

// Точечное произведение f32 на f32 (4 ленты, 4 независимых аккумулятора):
// хвост из 1-3 токенов считается по столбцам, а не по токенам. Вход хвоста
// лежит отдельным буфером [ntail][cols] (см. trit_batch_neon), поэтому
// обращение к нему contiguous — как у row_dot_trit.
static inline double dot_f32_f32(const float *a, const float *b, int n) {
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = a0, a2 = a0, a3 = a0;
    double s;
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        a0 = vfmaq_f32(a0, vld1q_f32(a + i), vld1q_f32(b + i));
        a1 = vfmaq_f32(a1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        a2 = vfmaq_f32(a2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
        a3 = vfmaq_f32(a3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    s = statvq_acc_f64(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
    for (; i < n; i++) s += (double)a[i] * (double)b[i];
    return s;
}

// Один блок масштаба: len тритов tp (начало блока — lo во входе x), ntok
// токенов, масштаб scale. Результат копится в acc[0..ntok). xtail — хвост
// 1-3 токенов, лежащий по столбцам (xtail[j*cols + lo ...]).
static inline void trit_batch_neon_block(const int8_t *tp, float *fb, int len,
                                         int64_t lo, const float *x, int ntok,
                                         double *acc, double scale,
                                         const float *xtail, int xtail_cols,
                                         int ntail) {
    int jj;

    trits_to_f32(tp, fb, len);

    // 8 токенов за шаг: две 4-ленточные группы, по 4 независимых
    // аккумулятора на группу (t mod 4) — 8 цепей FMA, хватает, чтобы
    // загрузить FP-конвейер.
    for (jj = 0; jj + 8 <= ntok; jj += 8) {
        float32x4_t a0 = vdupq_n_f32(0.0f), a1 = a0, a2 = a0, a3 = a0;
        float32x4_t b0 = a0, b1 = a0, b2 = a0, b3 = a0;
        int t = 0;
        for (; t + 4 <= len; t += 4) {
            const float *xp = x + (size_t)(lo + t) * (size_t)ntok
                            + (size_t)jj;
            float32x4_t c0 = vdupq_n_f32(fb[t + 0]);
            float32x4_t c1 = vdupq_n_f32(fb[t + 1]);
            float32x4_t c2 = vdupq_n_f32(fb[t + 2]);
            float32x4_t c3 = vdupq_n_f32(fb[t + 3]);
            a0 = vfmaq_f32(a0, vld1q_f32(xp), c0);
            b0 = vfmaq_f32(b0, vld1q_f32(xp + 4), c0);
            a1 = vfmaq_f32(a1, vld1q_f32(xp + ntok), c1);
            b1 = vfmaq_f32(b1, vld1q_f32(xp + ntok + 4), c1);
            a2 = vfmaq_f32(a2, vld1q_f32(xp + 2 * ntok), c2);
            b2 = vfmaq_f32(b2, vld1q_f32(xp + 2 * ntok + 4), c2);
            a3 = vfmaq_f32(a3, vld1q_f32(xp + 3 * ntok), c3);
            b3 = vfmaq_f32(b3, vld1q_f32(xp + 3 * ntok + 4), c3);
        }
        for (; t < len; t++) {
            const float *xp = x + (size_t)(lo + t) * (size_t)ntok
                            + (size_t)jj;
            float tv = fb[t];
            a0 = vfmaq_n_f32(a0, vld1q_f32(xp), tv);
            b0 = vfmaq_n_f32(b0, vld1q_f32(xp + 4), tv);
        }
        {
            float32x4_t sa = vaddq_f32(vaddq_f32(a0, a1),
                                       vaddq_f32(a2, a3));
            float32x4_t sb = vaddq_f32(vaddq_f32(b0, b1),
                                       vaddq_f32(b2, b3));
            acc[jj + 0] += (double)vgetq_lane_f32(sa, 0) * scale;
            acc[jj + 1] += (double)vgetq_lane_f32(sa, 1) * scale;
            acc[jj + 2] += (double)vgetq_lane_f32(sa, 2) * scale;
            acc[jj + 3] += (double)vgetq_lane_f32(sa, 3) * scale;
            acc[jj + 4] += (double)vgetq_lane_f32(sb, 0) * scale;
            acc[jj + 5] += (double)vgetq_lane_f32(sb, 1) * scale;
            acc[jj + 6] += (double)vgetq_lane_f32(sb, 2) * scale;
            acc[jj + 7] += (double)vgetq_lane_f32(sb, 3) * scale;
        }
    }
    for (; jj + 4 <= ntok; jj += 4) {
        float32x4_t a0 = vdupq_n_f32(0.0f), a1 = a0, a2 = a0, a3 = a0;
        int t = 0;
        for (; t + 4 <= len; t += 4) {
            const float *xp = x + (size_t)(lo + t) * (size_t)ntok
                            + (size_t)jj;
            a0 = vfmaq_n_f32(a0, vld1q_f32(xp),
                             fb[t + 0]);
            a1 = vfmaq_n_f32(a1, vld1q_f32(xp + ntok),
                             fb[t + 1]);
            a2 = vfmaq_n_f32(a2, vld1q_f32(xp + 2 * ntok),
                             fb[t + 2]);
            a3 = vfmaq_n_f32(a3, vld1q_f32(xp + 3 * ntok),
                             fb[t + 3]);
        }
        for (; t < len; t++) {
            const float *xp = x + (size_t)(lo + t) * (size_t)ntok
                            + (size_t)jj;
            a0 = vfmaq_n_f32(a0, vld1q_f32(xp), fb[t]);
        }
        {
            float32x4_t a = vaddq_f32(vaddq_f32(a0, a1),
                                      vaddq_f32(a2, a3));
            acc[jj + 0] += (double)vgetq_lane_f32(a, 0) * scale;
            acc[jj + 1] += (double)vgetq_lane_f32(a, 1) * scale;
            acc[jj + 2] += (double)vgetq_lane_f32(a, 2) * scale;
            acc[jj + 3] += (double)vgetq_lane_f32(a, 3) * scale;
        }
    }
    // Хвост 1-3 токенов — по столбцам: fb и xtail contiguous, 4 FMA на 16.
    for (jj = 0; jj < ntail; jj++)
        acc[ntok - ntail + jj] += dot_f32_f32(fb, xtail + (size_t)jj
                                               * (size_t)xtail_cols + lo,
                                               len) * scale;
}

typedef struct {
    float *y;
    const uint32_t *words;
    const float *scales;
    const float *x;
    int rows;
    int cols;
    int ntok;
    int scale_block;
    double *acc_all;     // vt * ntok
    const float *xtail;  // [ntail][cols], хвост ntok % 4 токенов
    int ntail;
} trit_batch_neon_job;

static void trit_batch_neon_rows(trit_batch_neon_job *j, int tid, int r0,
                                 int r1) {
    const int cols = j->cols;
    const int ntok = j->ntok;
    const int ntail = j->ntail;
    const float *xtail = j->xtail;
    const int64_t sb = (int64_t)j->scale_block;
    const uint32_t *words = j->words;
    const float *scales = j->scales;
    const float *x = j->x;
    double *acc = j->acc_all + (size_t)tid * (size_t)ntok;
    int8_t buf[VTM_SCALE_BLOCK_160 + 64];
    int8_t *tp = buf + 32;
    float fb[VTM_SCALE_BLOCK_160];
    int r, jj;

    for (r = r0; r < r1; r++) {
        const int64_t base = (int64_t)r * (int64_t)cols;
        const int64_t end = base + (int64_t)cols;
        int64_t s;

        memset(acc, 0, (size_t)ntok * sizeof(double));
        for (s = (base / sb) * sb; s < end; s += sb) {
            int64_t e = s + sb;
            int64_t st;
            int64_t wi, wi_end;

            if (e > end) e = end;
            st = s < base ? base : s;
            wi = st / VTM_TRITS_PER_WORD;
            wi_end = (e - 1) / VTM_TRITS_PER_WORD;
            for (; wi <= wi_end; wi++)
                unpack20(load_u32_unaligned(words, wi),
                         tp + (ptrdiff_t)(wi * VTM_TRITS_PER_WORD - st));
            trit_batch_neon_block(tp, fb, (int)(e - st), st - base, x, ntok,
                                  acc,
                                  (double)load_f32_unaligned(scales, s / sb),
                                  xtail, cols, ntail);
        }
        for (jj = 0; jj < ntok; jj++)
            j->y[(size_t)r * (size_t)ntok + (size_t)jj] = (float)acc[jj];
    }
}

static void trit_batch_neon_worker(void *p, int tid, int r0, int r1) {
    trit_batch_neon_rows((trit_batch_neon_job *)p, tid, r0, r1);
}

// Общее ядро TMV и MM: у них одна и та же арифметика (триты [rows,cols]
// против N столбцов активаций), различаются только имена аргументов.
static int trit_batch_neon(float *y, const uint8_t *w, const float *scales,
                           const float *x, int rows, int cols, int ntok,
                           int scale_block) {
    trit_batch_neon_job job;
    int vt;

    if (!y || !w || !scales || !x) return -1;
    if (rows < 0 || cols < 0 || ntok < 0) return -1;
    if (scale_block != (int)VTM_SCALE_BLOCK_128 &&
        scale_block != (int)VTM_SCALE_BLOCK_160) return -1;
    if (rows == 0 || cols == 0 || ntok == 0) return 0;
    lut_init();   // до потоков: таблица пишется один раз, не в воркерах
    vt = vtm_threads();
    job.acc_all = (double *)calloc((size_t)vt * (size_t)ntok,
                                   sizeof(double));
    if (!job.acc_all) return -1;
    job.y = y;
    job.words = (const uint32_t *)w;
    job.scales = scales;
    job.x = x;
    job.rows = rows;
    job.cols = cols;
    job.ntok = ntok;
    job.scale_block = scale_block;
    // Хвост 1-3 токенов (ntok % 4) kernel считает по столбцам: раскладываем
    // его в contiguous [ntail][cols] один раз на вызов, а не на строку.
    job.ntail = ntok % 4;
    job.xtail = NULL;
    if (job.ntail > 0) {
        float *xt = (float *)malloc((size_t)job.ntail * (size_t)cols
                                    * sizeof(float));
        int j, t;
        if (!xt) {
            free(job.acc_all);
            return -1;
        }
        for (j = 0; j < job.ntail; j++)
            for (t = 0; t < cols; t++)
                xt[(size_t)j * (size_t)cols + (size_t)t]
                    = x[(size_t)t * (size_t)ntok
                        + (size_t)(ntok - job.ntail + j)];
        job.xtail = xt;
    }
    vtm_parallel_rows(rows, (double)rows * (double)cols * (double)ntok,
                      trit_batch_neon_worker, &job);
    free((void *)job.xtail);
    free(job.acc_all);
    return 0;
}

#endif // VTM_SIMD_NEON

// --- выбор ядер и публичный контракт ---------------------------------------

typedef struct {
    int (*tmv_blocks)(float *, const uint8_t *, const float *, const float *,
                      int, int, int);
    int (*rmsnorm)(float *, const float *, const float *, int, float);
    int (*mm_trit)(float *, const uint8_t *, const float *, const float *,
                   int, int, int, int);
    int (*mm_f32)(float *, const float *, const float *, int, int, int);
    // Батчевые TMV/MM (задача 0122). NULL — публичные vtm_simd_*_batch
    // уходят в скалярный откат (trit_batch_scalar).
    int (*tmv_blocks_batch)(float *, const uint8_t *, const float *,
                            const float *, int, int, int, int);
    int (*mm_batch)(float *, const uint8_t *, const float *, const float *,
                    int, int, int, int);
    // Плотный GEMV (n == 1) и батчевый плотный TMV/MM (задача 0159). NULL —
    // публичные vtm_simd_* уходят в откат backend_scalar.c
    // (vtm_scalar_gemv_f32 / vtm_scalar_dense_batch).
    int (*gemv_f32)(float *, const float *, const float *, int, int, int);
    int (*tmv_f32_batch)(float *, const float *, const float *, int, int,
                         int, int);
    int (*mm_f32_batch)(float *, const float *, const float *, int, int, int);
} simd_kernels;

static const simd_kernels g_simd = {
#if VTM_SIMD_NEON
    tmv_neon_blocks,
    rmsnorm_neon,
    mm_trit_neon,
    mm_f32_neon,
    trit_batch_neon,
    trit_batch_neon,
    gemv_f32_neon,
    dense_batch_neon,
    dense_batch_neon_mm
#elif VTM_SIMD_X86
    vtm_avx2_tmv_blocks,
    vtm_avx2_rmsnorm,
    vtm_avx2_mm_trit,
    vtm_avx2_mm_f32,
    vtm_avx2_tmv_blocks_batch,
    vtm_avx2_mm_batch,
    vtm_avx2_gemv_f32,
    vtm_avx2_tmv_f32_batch,
    vtm_avx2_mm_f32_batch
#else
    NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL
#endif
};

int vtm_simd_available(void) {
    static int cached = -1;
    if (cached < 0) {
#if VTM_SIMD_NEON
        cached = 1;   // NEON — базовая для ARM64, проверка не нужна
#elif VTM_SIMD_X86
        cached = vtm_avx2_available();
#else
        cached = 0;
#endif
    }
    return cached;
}

const char *vtm_simd_isa(void) {
#if VTM_SIMD_NEON
    return "neon";
#elif VTM_SIMD_X86
    return vtm_simd_available() ? "avx2" : "none";
#else
    return "none";
#endif
}

// int8-dot: SDOT на ARM (FEAT_DotProd), VNNI на x86 (AVX-512). Спрашивается
// один раз (кэш внутри детекта): режим выбирает backend_scalar.c по
// VTM_Q4_I8, а само ядро ещё раз проверяет это же перед работой.
int vtm_q4_i8_hw(void) {
#if VTM_SIMD_NEON
    return vtm_dotprod_cpu();
#elif VTM_SIMD_X86
    return vtm_avx2_available() && vtm_avx2_vnni_available();
#else
    return 0;
#endif
}

static int simd_tmv_blocks(float *y, const uint8_t *w, const float *scales,
                           const float *x, int rows, int cols,
                           int scale_block) {
    if (vtm_simd_available() && g_simd.tmv_blocks)
        return g_simd.tmv_blocks(y, w, scales, x, rows, cols, scale_block);
    return vtm_backend_scalar_lut.tmv_blocks(y, w, scales, x, rows, cols,
                                             scale_block);
}

static int simd_tmv(float *y, const uint8_t *w, const float *scales,
                    const float *x, int rows, int cols) {
    return simd_tmv_blocks(y, w, scales, x, rows, cols,
                           (int)VTM_SCALE_BLOCK_160);
}

static int simd_rmsnorm(float *y, const float *x, const float *w,
                        int n, float eps) {
    if (vtm_simd_available() && g_simd.rmsnorm)
        return g_simd.rmsnorm(y, x, w, n, eps);
    return vtm_backend_scalar_lut.rmsnorm(y, x, w, n, eps);
}

int vtm_simd_mm_trit(float *y, const uint8_t *w, const float *scales,
                     const float *b, int m, int k, int n, int scale_block) {
    if (vtm_simd_available() && g_simd.mm_trit)
        return g_simd.mm_trit(y, w, scales, b, m, k, n, scale_block);
    return mm_trit_scalar(y, w, scales, b, m, k, n, scale_block);
}

int vtm_simd_mm_f32(float *y, const float *a, const float *b,
                    int m, int k, int n) {
    // Один столбец — это GEMV: у mm_f32 нет векторизации по столбцам (цикл
    // по 4 столбца не выполняется), а у vtm_simd_gemv_f32 есть (задача 0159).
    if (n == 1) return vtm_simd_gemv_f32(y, a, b, m, k, 0);
    if (vtm_simd_available() && g_simd.mm_f32)
        return g_simd.mm_f32(y, a, b, m, k, n);
    return mm_f32_scalar(y, a, b, m, k, n);
}

// Плотный GEMV (n == 1): SIMD-ядро, если собрано и CPU его умеет, иначе
// переносимый откат backend_scalar.c (4 double-аккумулятора).
int vtm_simd_gemv_f32(float *y, const float *w, const float *x,
                      int rows, int cols, int transposed) {
    if (!y || !w || !x) return -1;
    if (rows < 0 || cols < 0) return -1;
    if (vtm_simd_available() && g_simd.gemv_f32)
        return g_simd.gemv_f32(y, w, x, rows, cols, transposed);
    return vtm_scalar_gemv_f32(y, w, x, rows, cols, transposed);
}

// --- батчевые TMV/MM: публичный контракт (задача 0122) ---------------------
//
// Общий вход в ядра. n_tokens == 1 — это обычный одноколоночный случай:
// отдаём его прежнему ядру, чтобы батчевый путь не менял побитово результат
// декола. Дальше — SIMD-ядро из g_simd, если оно собрано и доступно, иначе
// скалярный откат. Валидация аргументов одна на все ветки.

static int batch_args_ok(const void *y, const void *w, const void *scales,
                         const void *x, int rows, int cols, int ntok,
                         int scale_block) {
    if (!y || !w || !scales || !x) return 0;
    if (rows < 0 || cols < 0 || ntok < 0) return 0;
    if (scale_block != (int)VTM_SCALE_BLOCK_128 &&
        scale_block != (int)VTM_SCALE_BLOCK_160) return 0;
    return 1;
}

int vtm_simd_tmv_blocks_batch(float *y, const uint8_t *w, const float *scales,
                              const float *x, int rows, int cols,
                              int n_tokens, int scale_block) {
    if (!batch_args_ok(y, w, scales, x, rows, cols, n_tokens, scale_block))
        return -1;
    if (rows == 0 || cols == 0 || n_tokens == 0) return 0;
    if (n_tokens == 1)
        return simd_tmv_blocks(y, w, scales, x, rows, cols, scale_block);
    if (vtm_simd_available() && g_simd.tmv_blocks_batch)
        return g_simd.tmv_blocks_batch(y, w, scales, x, rows, cols, n_tokens,
                                       scale_block);
    return trit_batch_scalar(y, w, scales, x, rows, cols, n_tokens,
                             scale_block);
}

int vtm_simd_mm_batch(float *y, const uint8_t *w, const float *scales,
                      const float *b, int m, int k, int n_tokens,
                      int scale_block) {
    // Арифметика тритовой MM и TMV одна и та же (W[m,k] против N столбцов),
    // поэтому и ядро одно; имена аргументов разные ради читаемости вызова.
    if (!batch_args_ok(y, w, scales, b, m, k, n_tokens, scale_block))
        return -1;
    if (m == 0 || k == 0 || n_tokens == 0) return 0;
    if (n_tokens == 1)
        return vtm_simd_mm_trit(y, w, scales, b, m, k, 1, scale_block);
    if (vtm_simd_available() && g_simd.mm_batch)
        return g_simd.mm_batch(y, w, scales, b, m, k, n_tokens, scale_block);
    return trit_batch_scalar(y, w, scales, b, m, k, n_tokens, scale_block);
}

// --- батчевый плотный TMV/MM: публичный контракт (задача 0159) -------------
//
// Общий вход в плотные ядра. n_tokens == 1 — обычный одноколоночный случай:
// отдаём его GEMV, чтобы батчевый путь не расходился с декодом. Дальше —
// SIMD-ядро из g_simd (вектор по токенам пачки), если оно собрано и доступно,
// иначе переносимый откат vtm_scalar_dense_batch (тот же double-порядок, что
// у прежнего батчевого цикла interp.c).
int vtm_simd_tmv_f32_batch(float *y, const float *w, const float *x,
                           int rows, int cols, int n_tokens, int transposed) {
    if (!y || !w || !x) return -1;
    if (rows < 0 || cols < 0 || n_tokens < 0) return -1;
    if (rows == 0 || cols == 0 || n_tokens == 0) return 0;
    if (n_tokens == 1)
        return vtm_simd_gemv_f32(y, w, x, rows, cols, transposed);
    if (vtm_simd_available() && g_simd.tmv_f32_batch)
        return g_simd.tmv_f32_batch(y, w, x, rows, cols, n_tokens,
                                    transposed);
    return vtm_scalar_dense_batch(y, w, x, rows, cols, n_tokens, transposed);
}

int vtm_simd_mm_f32_batch(float *y, const float *a, const float *b,
                          int m, int k, int n_tokens) {
    if (!y || !a || !b) return -1;
    if (m < 0 || k < 0 || n_tokens < 0) return -1;
    if (m == 0 || k == 0 || n_tokens == 0) return 0;
    if (n_tokens == 1) return vtm_simd_gemv_f32(y, a, b, m, k, 0);
    if (vtm_simd_available() && g_simd.mm_f32_batch)
        return g_simd.mm_f32_batch(y, a, b, m, k, n_tokens);
    return vtm_scalar_dense_batch(y, a, b, m, k, n_tokens, 0);
}

#if VTM_SIMD_NEON
#define VTM_SIMD_BACKEND_NAME "simd-neon"
#elif VTM_SIMD_X86
#define VTM_SIMD_BACKEND_NAME "simd-avx2"
#else
#define VTM_SIMD_BACKEND_NAME "simd"
#endif

// Прочие ядра (silu, mul, add, scal, softmax, emb, copy) оставлены NULL:
// интерпретатор возьмёт скалярные. Они не в счётном ядре модели, а лишний
// SIMD-путь — лишний риск расхождения.
const vtm_backend vtm_backend_simd = {
    .name = VTM_SIMD_BACKEND_NAME,
    .tmv = simd_tmv,
    .tmv_blocks = simd_tmv_blocks,
    .rmsnorm = simd_rmsnorm,
    .silu = NULL,
    .mul = NULL,
    .add = NULL,
    .scal = NULL,
    .softmax = NULL,
    .emb = NULL,
    .copy = NULL
};

// auto = hybrid? Решение по парному замеру 0198 (bench_backend --metal на
// 3B Q4_K, одна машина, min из 2, два прогона): prefill 64 токена 19,6 -> 38,4
// ток/с (1,96x, TTFT 3,28 -> 1,67 с), prefill 128 — 1,86x, декод 20,2 -> 19,9
// ток/с (0,98x, шум: в hybrid те же NEON-ядра). Условие задачи выполнено
// (prefill >= 1,3x, декод не хуже simd), поэтому auto выбирает hybrid, когда
// Metal собран и устройство есть. Одна константа, чтобы вернуть simd без
// правок логики: 0 — auto=simd.
// 2026-11-26: auto=simd. Гибрид (0198) даёт верные числа на прямых прогонах
// vtm run, но через C-ABI (vtm_api_generate: кэш префикса + батчевый prefill)
// выдаёт мусор — похоже на гонку асинхронной отправки GPU. До исправления
// гибрид доступен только явно: VTM_BACKEND=hybrid.
#ifndef VTM_AUTO_HYBRID
#define VTM_AUTO_HYBRID 0
#endif

// Гибридный бэкенд (0198): ядра — simd, а батчевые GEMM prefill уходят на GPU
// по vtm_metal_active(). Доступен только в сборке с Metal.
static const vtm_backend *backend_hybrid_or_simd(const char *why) {
#if VTM_WITH_METAL
    if (vtm_metal_compiled() && vtm_metal_available()) {
        vtm_metal_set_mode(VTM_METAL_BATCH);
        return vtm_backend_hybrid();
    }
    fprintf(stderr, "VTM: %s: Metal недоступен — бэкенд simd (откат)\n", why);
#else
    fprintf(stderr, "VTM: %s: сборка без Metal — бэкенд simd (откат)\n", why);
#endif
    vtm_metal_set_active(0);
    return vtm_simd_available() ? &vtm_backend_simd : &vtm_backend_scalar_lut;
}

const vtm_backend *vtm_backend_by_name(const char *name) {
    if (!name || !*name) return vtm_backend_from_env();
    if (strcmp(name, "auto") == 0) {
        // 0198: hybrid (GPU только на prefill GEMM), иначе simd. Устройство
        // проверяется здесь, а шейдеры компилируются лениво при первом GEMM.
        if (!vtm_simd_available()) {
            vtm_metal_set_active(0);
            return &vtm_backend_scalar_lut;
        }
#if VTM_WITH_METAL && VTM_AUTO_HYBRID
        // Устройство, а не только сборка: на машине без GPU auto остаётся simd.
        if (vtm_metal_compiled() && vtm_metal_available()) {
            vtm_metal_set_mode(VTM_METAL_BATCH);
            return vtm_backend_hybrid();
        }
#endif
        vtm_metal_set_active(0);
        return &vtm_backend_simd;
    }
    if (strcmp(name, "hybrid") == 0) {
        return backend_hybrid_or_simd("hybrid");
    }
    if (strcmp(name, "metal") == 0) {
#if VTM_WITH_METAL
        if (vtm_metal_compiled()) {
            vtm_metal_set_active(1);
            return vtm_backend_metal();
        }
        fprintf(stderr, "VTM: Metal недоступен — бэкенд simd (откат)\n");
#else
        fprintf(stderr, "VTM: сборка без Metal — бэкенд simd (откат)\n");
#endif
        vtm_metal_set_active(0);
        return &vtm_backend_simd;
    }
    if (strcmp(name, "simd") == 0 || strcmp(name, "neon") == 0 ||
        strcmp(name, "avx2") == 0 || strcmp(name, "simd-neon") == 0 ||
        strcmp(name, "simd-avx2") == 0) {
        vtm_metal_set_active(0);
        return &vtm_backend_simd;
    }
    if (strcmp(name, "scalar") == 0 || strcmp(name, "lut") == 0 ||
        strcmp(name, "scalar-lut243") == 0) {
        vtm_metal_set_active(0);
        return &vtm_backend_scalar_lut;
    }
    if (strcmp(name, "direct") == 0 || strcmp(name, "scalar-direct") == 0) {
        vtm_metal_set_active(0);
        return &vtm_backend_scalar_direct;
    }
    return NULL;
}

const vtm_backend *vtm_backend_from_env(void) {
    const char *env = getenv("VTM_BACKEND");
    const vtm_backend *backend;

    // Без переменной — auto: SIMD, если он собран и процессор его умеет, иначе
    // скалярный LUT243. Раньше по умолчанию был скаляр, и без флага VTM_BACKEND
    // терялось 5,5x на проходе (замер 0071). auto ничего не ломает: бэкенд сам
    // откатывается на скаляр, если SIMD недоступен.
    if (!env || !*env) return vtm_backend_by_name("auto");
    backend = vtm_backend_by_name(env);
    if (backend) return backend;
    fprintf(stderr, "vtm: неизвестный VTM_BACKEND='%s' "
                    "(auto|hybrid|metal|scalar|simd|direct), беру scalar\n",
            env);
    vtm_metal_set_active(0);
    return &vtm_backend_scalar_lut;
}
