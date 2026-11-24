// AVX2-ядра VTM для x86-64. Makefile собирает файл только на x86-64 (по
// uname -m); на других арх он пустой. AVX2/FMA включаются не флагом TU, а
// __attribute__((target("avx2,fma"))) на каждом ядре — см. ниже.
//
// Устроено как backend_neon.c: LUT на 8 тритов (6561 запись) + int8-буфер,
// затем _mm256_cvtepi8_epi32 (8 int8 -> 8 int32 за одну инструкцию)
// + _mm256_cvtepi32_ps + FMA.
// На бумаге это дешевле NEON (там int8->int16->int32 два шага), но на
// железе x86 здесь не проверялось: только сборка и прогон под Rosetta 2
// (см. docs/vtm-perf.md) — числа с x86 в отчёт не идут.
#include "vtm.h"
#include "vtm_backend.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__)

#include <immintrin.h>

// Файл собирается БЕЗ -mavx2: целевые атрибуты только на ядрах, чтобы
// vtm_avx2_available() и код вне ядер не содержали AVX2-инструкций (иначе
// детект на старом CPU сам бы упал в SIGILL).
#define VTM_AVX2 __attribute__((target("avx2,fma")))

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

// --- ядра ------------------------------------------------------------------

// LUT на 8 тритов (3^8 = 6561 запись по 8 байт) — как в backend_neon.c:
// слово pack20 = 8 + 8 + 4 трита, два деления на 6561 на слово.
static uint64_t g_lut8[6561];
static int g_lut8_ready;

VTM_AVX2 static void lut_init(void) {
    unsigned v;
    int k;
    if (g_lut8_ready) return;
    for (v = 0; v < 6561u; v++) {
        uint64_t packed = 0;
        unsigned c = v;
        for (k = 0; k < 8; k++) {
            unsigned d = c % 3u;
            int8_t trit = (int8_t)(d == 2u ? 1 : (d == 0u ? -1 : 0));
            packed |= (uint64_t)(uint8_t)trit << (8u * (unsigned)k);
            c /= 3u;
        }
        g_lut8[v] = packed;
    }
    g_lut8_ready = 1;
}

VTM_AVX2 static inline void unpack20(uint32_t v, int8_t *out) {
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

// Сумма 8-ленточного аккумулятора в double.
VTM_AVX2 static inline double acc_f64(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m256d d = _mm256_add_pd(_mm256_cvtps_pd(lo), _mm256_cvtps_pd(hi));
    __m128d s2 = _mm_add_pd(_mm256_castpd256_pd128(d),
                            _mm256_extractf128_pd(d, 1));
    return _mm_cvtsd_f64(_mm_add_sd(s2, _mm_unpackhi_pd(s2, s2)));
}

// 16 тритов за шаг: два аккумулятора по 8 f32.
VTM_AVX2 static inline double dot_i8_f32(const int8_t *t, const float *x, int n) {
    __m256 a0 = _mm256_setzero_ps();
    __m256 a1 = _mm256_setzero_ps();
    double s;
    int i = 0;

    for (; i + 16 <= n; i += 16) {
        __m128i b0 = _mm_loadl_epi64((const __m128i *)(const void *)(t + i));
        __m128i b1 = _mm_loadl_epi64((const __m128i *)(const void *)(t + i + 8));
        a0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b0)),
                             _mm256_loadu_ps(x + i), a0);
        a1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b1)),
                             _mm256_loadu_ps(x + i + 8), a1);
    }
    s = acc_f64(_mm256_add_ps(a0, a1));
    for (; i < n; i++) s += (double)t[i] * (double)x[i];
    return s;
}

VTM_AVX2 static double row_dot_trit(const uint32_t *words, const float *scales,
                           const float *x, int64_t base, int cols,
                           int scale_block) {
    const int64_t sb = (int64_t)scale_block;
    const int64_t end = base + (int64_t)cols;
    int8_t buf[VTM_SCALE_BLOCK_160 + 64];
    int8_t *tp = buf + 32;
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
// же, что при одном потоке, — разбиение по строкам побитово ничего не
// меняет.
VTM_AVX2 static void tmv_avx2_rows(float *y, const uint32_t *words,
                                   const float *scales, const float *x,
                                   int cols, int scale_block,
                                   int r0, int r1) {
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
} tmv_avx2_job;

// Колбэк пула — без target-атрибута (его тип общий vtm_rows_fn); работу
// делает AVX2-ядро, вызов к нему прямой и не инлайнится.
static void tmv_avx2_worker(void *p, int tid, int r0, int r1) {
    tmv_avx2_job *j = (tmv_avx2_job *)p;
    (void)tid;
    tmv_avx2_rows(j->y, j->words, j->scales, j->x, j->cols, j->scale_block,
                  r0, r1);
}

VTM_AVX2 int vtm_avx2_tmv_blocks(float *y, const uint8_t *w, const float *scales,
                        const float *x, int rows, int cols,
                        int scale_block) {
    tmv_avx2_job job;
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
    vtm_parallel_rows(rows, (double)rows * (double)cols, tmv_avx2_worker,
                      &job);
    return 0;
}

// Однопоточный RMSN — прежний код (VTM_THREADS=1 и малые n).
VTM_AVX2 static int rmsnorm_avx2_serial(float *y, const float *x,
                                        const float *w, int n, float eps) {
    __m256d a0 = _mm256_setzero_pd();
    __m256d a1 = _mm256_setzero_pd();
    double sum, mean, rms;
    float inv;
    int i = 0;

    if (!y || !x || !w || n < 0) return -1;
    if (n == 0) return 0;

    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        __m128 lo = _mm256_castps256_ps128(v);
        __m128 hi = _mm256_extractf128_ps(v, 1);
        a0 = _mm256_fmadd_pd(_mm256_cvtps_pd(lo),
                             _mm256_cvtps_pd(lo), a0);
        a1 = _mm256_fmadd_pd(_mm256_cvtps_pd(hi),
                             _mm256_cvtps_pd(hi), a1);
    }
    {
        __m256d sum4 = _mm256_add_pd(a0, a1);
        __m128d lo = _mm256_castpd256_pd128(sum4);
        __m128d hi = _mm256_extractf128_pd(sum4, 1);
        __m128d s2 = _mm_add_pd(lo, hi);
        sum = _mm_cvtsd_f64(_mm_add_sd(s2, _mm_unpackhi_pd(s2, s2)));
    }
    for (; i < n; i++) sum += (double)x[i] * (double)x[i];

    mean = sum / (double)n;
    rms = sqrt(mean + (double)eps);
    if (rms == 0.0) {
        memset(y, 0, (size_t)n * sizeof(float));
        return 0;
    }
    inv = (float)(1.0 / rms);

    i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 xv = _mm256_loadu_ps(x + i);
        __m256 wv = _mm256_loadu_ps(w + i);
        _mm256_storeu_ps(y + i,
                         _mm256_mul_ps(_mm256_mul_ps(xv, wv),
                                       _mm256_set1_ps(inv)));
    }
    for (; i < n; i++) y[i] = x[i] * w[i] * inv;
    return 0;
}

// RMSN для больших n: сумма квадратов по фиксированным блокам
// VTM_RMSN_BLOCK, частичные суммы складываются по возрастанию номера блока,
// затем поэлементная часть. Результат один и тот же при 2/4/8 потоках
// (побитово); от VTM_THREADS=1 отличается только порядком double-сложения
// суммы квадратов (~1e-16 относительно).
#define VTM_RMSN_BLOCK 4096

typedef struct {
    const float *x;
    const float *w;
    float *y;
    double *sums;
    int n;
    float inv;
} rmsn_avx2_job;

// Блочные суммы/масштабирование — AVX2-ядром (target-атрибут), а обёртки
// без атрибута — только чтобы тип совпал с vtm_rows_fn: вызов прямой.
VTM_AVX2 static void rmsnorm_avx2_sum_blocks(void *p, int tid, int b0,
                                             int b1) {
    rmsn_avx2_job *j = (rmsn_avx2_job *)p;
    int b;
    (void)tid;
    for (b = b0; b < b1; b++) {
        int i0 = b * VTM_RMSN_BLOCK;
        int i1 = i0 + VTM_RMSN_BLOCK;
        __m256d a0 = _mm256_setzero_pd();
        __m256d a1 = _mm256_setzero_pd();
        __m256d sum4;
        __m128d lo, hi, s2;
        double sum;
        int i = i0;
        if (i1 > j->n) i1 = j->n;
        for (; i + 8 <= i1; i += 8) {
            __m256 v = _mm256_loadu_ps(j->x + i);
            __m128 vlo = _mm256_castps256_ps128(v);
            __m128 vhi = _mm256_extractf128_ps(v, 1);
            a0 = _mm256_fmadd_pd(_mm256_cvtps_pd(vlo),
                                 _mm256_cvtps_pd(vlo), a0);
            a1 = _mm256_fmadd_pd(_mm256_cvtps_pd(vhi),
                                 _mm256_cvtps_pd(vhi), a1);
        }
        sum4 = _mm256_add_pd(a0, a1);
        lo = _mm256_castpd256_pd128(sum4);
        hi = _mm256_extractf128_pd(sum4, 1);
        s2 = _mm_add_pd(lo, hi);
        sum = _mm_cvtsd_f64(_mm_add_sd(s2, _mm_unpackhi_pd(s2, s2)));
        for (; i < i1; i++) sum += (double)j->x[i] * (double)j->x[i];
        j->sums[b] = sum;
    }
}

VTM_AVX2 static void rmsnorm_avx2_scale_blocks(void *p, int tid, int b0,
                                               int b1) {
    rmsn_avx2_job *j = (rmsn_avx2_job *)p;
    const __m256 vinv = _mm256_set1_ps(j->inv);
    int b;
    (void)tid;
    for (b = b0; b < b1; b++) {
        int i0 = b * VTM_RMSN_BLOCK;
        int i1 = i0 + VTM_RMSN_BLOCK;
        int i = i0;
        if (i1 > j->n) i1 = j->n;
        /* Формула та же, что в serial-версии (x*w*inv в f32). */
        for (; i + 8 <= i1; i += 8) {
            __m256 xv = _mm256_loadu_ps(j->x + i);
            __m256 wv = _mm256_loadu_ps(j->w + i);
            _mm256_storeu_ps(j->y + i,
                             _mm256_mul_ps(_mm256_mul_ps(xv, wv), vinv));
        }
        for (; i < i1; i++) j->y[i] = j->x[i] * j->w[i] * j->inv;
    }
}

static void rmsnorm_avx2_sum_worker(void *p, int tid, int b0, int b1) {
    rmsnorm_avx2_sum_blocks(p, tid, b0, b1);
}

static void rmsnorm_avx2_scale_worker(void *p, int tid, int b0, int b1) {
    rmsnorm_avx2_scale_blocks(p, tid, b0, b1);
}

VTM_AVX2 int vtm_avx2_rmsnorm(float *y, const float *x, const float *w,
                     int n, float eps) {
    rmsn_avx2_job job;
    double sum = 0.0, mean, rms;
    int nblocks, b;

    if (!y || !x || !w || n < 0) return -1;
    if (n == 0) return 0;
    if (vtm_threads() < 2 || n < VTM_PAR_RMSN_MIN)
        return rmsnorm_avx2_serial(y, x, w, n, eps);

    nblocks = (n + VTM_RMSN_BLOCK - 1) / VTM_RMSN_BLOCK;
    job.x = x;
    job.w = w;
    job.y = y;
    job.n = n;
    job.sums = (double *)calloc((size_t)nblocks, sizeof(double));
    if (!job.sums)
        return rmsnorm_avx2_serial(y, x, w, n, eps);

    vtm_parallel_rows(nblocks, 2.0 * (double)n, rmsnorm_avx2_sum_worker,
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
    vtm_parallel_rows(nblocks, (double)n, rmsnorm_avx2_scale_worker, &job);
    return 0;
}
typedef struct {
    float *y;
    const uint32_t *words;
    const float *scales;
    const float *b;
    int k;
    int n;
    int64_t sb;
    int scale_block;
    double *acc_all;   // vt * n; NULL при n == 1
} mm_avx2_job;

VTM_AVX2 static void mm_trit_avx2_rows(mm_avx2_job *j, int tid,
                                       int r0, int r1) {
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

            for (c = 0; c + 8 <= n; c += 8) {
                __m256 a = _mm256_setzero_ps();
                t = 0;
                for (; t + 16 <= len; t += 16) {
                    __m256 part = _mm256_setzero_ps();
                    int tt;
                    for (tt = 0; tt < 16; tt++) {
                        const float *bp = b
                            + (size_t)(lo + t + tt) * (size_t)n + (size_t)c;
                        part = _mm256_fmadd_ps(
                            _mm256_set1_ps((float)tp[t + tt]),
                            _mm256_loadu_ps(bp), part);
                    }
                    a = _mm256_add_ps(a, part);
                }
                {
                    __m128 lo = _mm256_castps256_ps128(a);
                    __m128 hi = _mm256_extractf128_ps(a, 1);
                    __m256d d = _mm256_cvtps_pd(lo);
                    __m256d d2 = _mm256_cvtps_pd(hi);
                    _mm256_storeu_pd(acc + c, _mm256_add_pd(
                        _mm256_loadu_pd(acc + c),
                        _mm256_mul_pd(d, _mm256_set1_pd(scale))));
                    _mm256_storeu_pd(acc + c + 4, _mm256_add_pd(
                        _mm256_loadu_pd(acc + c + 4),
                        _mm256_mul_pd(d2, _mm256_set1_pd(scale))));
                }
                for (; t < len; t++) {
                    float tv = (float)tp[t];
                    const float *bp = b
                        + (size_t)(lo + t) * (size_t)n + (size_t)c;
                    int cc;
                    for (cc = 0; cc < 8; cc++)
                        acc[c + cc] += (double)(tv * bp[cc]) * scale;
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

static void mm_trit_avx2_worker(void *p, int tid, int r0, int r1) {
    mm_trit_avx2_rows((mm_avx2_job *)p, tid, r0, r1);
}

VTM_AVX2 int vtm_avx2_mm_trit(float *y, const uint8_t *w, const float *scales,
                     const float *b, int m, int k, int n, int scale_block) {
    mm_avx2_job job;
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
    vt = vtm_threads();
    if (n > 1) {
        job.acc_all = (double *)calloc((size_t)vt * (size_t)n,
                                       sizeof(double));
        if (!job.acc_all) return -1;
    }
    vtm_parallel_rows(m, (double)m * (double)k * (double)n,
                      mm_trit_avx2_worker, &job);
    free(job.acc_all);
    return 0;
}

typedef struct {
    float *y;
    const float *a;
    const float *b;
    int k;
    int n;
} mm_f32_avx2_job;

// 8 столбцов за раз; куски по 16 t копятся в f32, затем в double.
VTM_AVX2 static void mm_f32_avx2_rows(mm_f32_avx2_job *j, int tid,
                                      int r0, int r1) {
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
        for (c = 0; c + 8 <= n; c += 8) {
            double acc[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            t = 0;
            for (; t + 16 <= k; t += 16) {
                __m256 part = _mm256_setzero_ps();
                int tt;
                for (tt = 0; tt < 16; tt++)
                    part = _mm256_fmadd_ps(
                        _mm256_set1_ps(ar[t + tt]),
                        _mm256_loadu_ps(b + (size_t)(t + tt)
                                          * (size_t)n + (size_t)c),
                        part);
                {
                    __m128 lo = _mm256_castps256_ps128(part);
                    __m128 hi = _mm256_extractf128_ps(part, 1);
                    _mm256_storeu_pd(acc + 0, _mm256_add_pd(
                        _mm256_loadu_pd(acc + 0), _mm256_cvtps_pd(lo)));
                    _mm256_storeu_pd(acc + 4, _mm256_add_pd(
                        _mm256_loadu_pd(acc + 4), _mm256_cvtps_pd(hi)));
                }
            }
            for (; t < k; t++) {
                int cc;
                for (cc = 0; cc < 8; cc++)
                    acc[cc] += (double)ar[t]
                             * (double)b[(size_t)t * (size_t)n
                                         + (size_t)(c + cc)];
            }
            yr[c + 0] = (float)acc[0];
            yr[c + 1] = (float)acc[1];
            yr[c + 2] = (float)acc[2];
            yr[c + 3] = (float)acc[3];
            yr[c + 4] = (float)acc[4];
            yr[c + 5] = (float)acc[5];
            yr[c + 6] = (float)acc[6];
            yr[c + 7] = (float)acc[7];
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

static void mm_f32_avx2_worker(void *p, int tid, int r0, int r1) {
    mm_f32_avx2_rows((mm_f32_avx2_job *)p, tid, r0, r1);
}

VTM_AVX2 int vtm_avx2_mm_f32(float *y, const float *a, const float *b,
                    int m, int k, int n) {
    mm_f32_avx2_job job;
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
                      mm_f32_avx2_worker, &job);
    return 0;
}

// --- AVX2: плотный GEMV (n == 1), задача 0159 ------------------------------
//
// Как в backend_neon.c (задача 0158): внутри блока VTM_GEMV_CHUNK элементов —
// 4 f32-аккумулятора по 8 значений, блок сворачивается в double, чтобы
// f32-ошибка не копилась на длинной строке (lm_head 3B Q4: cols = 2048).
// Порядок сложений иной, чем у скалярного double-цикла, — расхождение в
// пределах 1e-5 (bench_backend --check), от числа потоков не зависит.
#define VTM_GEMV_CHUNK 64
// Блок батчевого ядра короче: на каждый токен-лейн приходится
// вдвое меньше f32-слагаемых до свёртки в double — запас по
// точности больше (порог 1e-5 на больших формах).
#define VTM_DENSE_BATCH_CHUNK 32

VTM_AVX2 static double gemv_dot_f32_avx2(const float *a, const float *b,
                                         int n) {
    double sum = 0.0;
    int i = 0;

    for (; i + 8 <= n; ) {
        __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
        int len = n - i;
        int j = 0;

        if (len > VTM_GEMV_CHUNK) len = VTM_GEMV_CHUNK;
        for (; j + 32 <= len; j += 32) {
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + j),
                                 _mm256_loadu_ps(b + i + j), a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + j + 8),
                                 _mm256_loadu_ps(b + i + j + 8), a1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + j + 16),
                                 _mm256_loadu_ps(b + i + j + 16), a2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + j + 24),
                                 _mm256_loadu_ps(b + i + j + 24), a3);
        }
        for (; j + 8 <= len; j += 8)
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + j),
                                 _mm256_loadu_ps(b + i + j), a0);
        sum += acc_f64(_mm256_add_ps(_mm256_add_ps(a0, a1),
                                     _mm256_add_ps(a2, a3)));
        for (; j < len; j++)
            sum += (double)a[i + j] * (double)b[i + j];
        i += len;
    }
    for (; i < n; i++)
        sum += (double)a[i] * (double)b[i];
    return sum;
}

// W — [cols, rows]: 8 выходов за шаг, блок по t — в double.
VTM_AVX2 static void gemv_f32_avx2_transposed_rows(const float *w,
                                                   const float *x, float *y,
                                                   int rows, int cols,
                                                   int r0, int r1) {
    int r = r0, t;

    for (; r + 8 <= r1; r += 8) {
        __m256d acc0 = _mm256_setzero_pd();
        __m256d acc1 = _mm256_setzero_pd();
        int c;

        for (c = 0; c < cols; c += VTM_GEMV_CHUNK) {
            __m256 part = _mm256_setzero_ps();
            int end = c + VTM_GEMV_CHUNK;

            if (end > cols) end = cols;
            for (t = c; t < end; t++)
                part = _mm256_fmadd_ps(
                    _mm256_loadu_ps(w + (size_t)t * (size_t)rows
                                    + (size_t)r),
                    _mm256_set1_ps(x[t]), part);
            acc0 = _mm256_add_pd(acc0,
                                 _mm256_cvtps_pd(_mm256_castps256_ps128(part)));
            acc1 = _mm256_add_pd(
                acc1, _mm256_cvtps_pd(_mm256_extractf128_ps(part, 1)));
        }
        {
            double lo[4], hi[4];
            _mm256_storeu_pd(lo, acc0);
            _mm256_storeu_pd(hi, acc1);
            y[r + 0] = (float)lo[0];
            y[r + 1] = (float)lo[1];
            y[r + 2] = (float)lo[2];
            y[r + 3] = (float)lo[3];
            y[r + 4] = (float)hi[0];
            y[r + 5] = (float)hi[1];
            y[r + 6] = (float)hi[2];
            y[r + 7] = (float)hi[3];
        }
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
} gemv_f32_avx2_job;

VTM_AVX2 static void gemv_f32_avx2_rows(gemv_f32_avx2_job *j, int r0,
                                        int r1) {
    int r;

    if (j->transposed) {
        gemv_f32_avx2_transposed_rows(j->w, j->x, j->y, j->rows, j->cols,
                                      r0, r1);
        return;
    }
    for (r = r0; r < r1; r++)
        j->y[r] = (float)gemv_dot_f32_avx2(
            j->w + (size_t)r * (size_t)j->cols, j->x, j->cols);
}

// Колбэк пула — без target-атрибута (его тип общий vtm_rows_fn).
static void gemv_f32_avx2_worker(void *p, int tid, int r0, int r1) {
    (void)tid;
    gemv_f32_avx2_rows((gemv_f32_avx2_job *)p, r0, r1);
}

VTM_AVX2 int vtm_avx2_gemv_f32(float *y, const float *w, const float *x,
                               int rows, int cols, int transposed) {
    gemv_f32_avx2_job job;

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
                      gemv_f32_avx2_worker, &job);
    return 0;
}

// --- AVX2: батчевый плотный TMV/MM (задача 0159) ---------------------------
//
// y[r*N + j] = sum_t W[r,t] · x[t*N + j] (transposed == 0) / W[t,r].
// По вектору (8 значений) — токены пачки, вес W[r,t] — broadcast; блок
// VTM_DENSE_BATCH_CHUNK по t копится в f32 и сворачивается в double.
// Расхождение с батчевым double-эталоном ≤1e-5 (bench_backend --check).
typedef struct {
    float       *y;
    const float *w;
    const float *x;
    int          rows;
    int          cols;
    int          ntok;
    int          transposed;
    double      *acc_all;   // vt * ntok
} dense_batch_avx2_job;

VTM_AVX2 static void dense_batch_avx2_rows(dense_batch_avx2_job *j, int tid,
                                           int r0, int r1) {
    const int ntok = j->ntok;
    const int cols = j->cols;
    const int rows = j->rows;
    const float *w = j->w;
    const float *x = j->x;
    double *acc = j->acc_all + (size_t)tid * (size_t)ntok;
    int r, jj, t, c;

    for (r = r0; r < r1; r++) {
        memset(acc, 0, (size_t)ntok * sizeof(double));
        // 16 токенов за шаг: два вектора по 8.
        for (jj = 0; jj + 16 <= ntok; jj += 16) {
            const float *base = x + (size_t)jj;
            for (c = 0; c < cols; c += VTM_DENSE_BATCH_CHUNK) {
                __m256 a0 = _mm256_setzero_ps();
                __m256 a1 = _mm256_setzero_ps();
                int end = c + VTM_DENSE_BATCH_CHUNK;

                if (end > cols) end = cols;
                for (t = c; t < end; t++) {
                    const float *xp = base + (size_t)t * (size_t)ntok;
                    float wv = j->transposed
                             ? w[(size_t)t * (size_t)rows + (size_t)r]
                             : w[(size_t)r * (size_t)cols + (size_t)t];
                    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(xp),
                                         _mm256_set1_ps(wv), a0);
                    a1 = _mm256_fmadd_ps(_mm256_loadu_ps(xp + 8),
                                         _mm256_set1_ps(wv), a1);
                }
                {
                    double lo[8], hi[8];
                    int k;
                    _mm256_storeu_pd(lo,
                                     _mm256_cvtps_pd(
                                         _mm256_castps256_ps128(a0)));
                    _mm256_storeu_pd(hi,
                                     _mm256_cvtps_pd(
                                         _mm256_extractf128_ps(a0, 1)));
                    for (k = 0; k < 4; k++) {
                        acc[jj + k] += lo[k];
                        acc[jj + 4 + k] += hi[k];
                    }
                    _mm256_storeu_pd(lo,
                                     _mm256_cvtps_pd(
                                         _mm256_castps256_ps128(a1)));
                    _mm256_storeu_pd(hi,
                                     _mm256_cvtps_pd(
                                         _mm256_extractf128_ps(a1, 1)));
                    for (k = 0; k < 4; k++) {
                        acc[jj + 8 + k] += lo[k];
                        acc[jj + 12 + k] += hi[k];
                    }
                }
            }
        }
        for (; jj + 8 <= ntok; jj += 8) {
            const float *base = x + (size_t)jj;
            for (c = 0; c < cols; c += VTM_DENSE_BATCH_CHUNK) {
                __m256 a0 = _mm256_setzero_ps();
                int end = c + VTM_DENSE_BATCH_CHUNK;

                if (end > cols) end = cols;
                for (t = c; t < end; t++) {
                    const float *xp = base + (size_t)t * (size_t)ntok;
                    float wv = j->transposed
                             ? w[(size_t)t * (size_t)rows + (size_t)r]
                             : w[(size_t)r * (size_t)cols + (size_t)t];
                    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(xp),
                                         _mm256_set1_ps(wv), a0);
                }
                {
                    double lo[4], hi[4];
                    int k;
                    _mm256_storeu_pd(lo,
                                     _mm256_cvtps_pd(
                                         _mm256_castps256_ps128(a0)));
                    _mm256_storeu_pd(hi,
                                     _mm256_cvtps_pd(
                                         _mm256_extractf128_ps(a0, 1)));
                    for (k = 0; k < 4; k++) {
                        acc[jj + k] += lo[k];
                        acc[jj + 4 + k] += hi[k];
                    }
                }
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

static void dense_batch_avx2_worker(void *p, int tid, int r0, int r1) {
    dense_batch_avx2_rows((dense_batch_avx2_job *)p, tid, r0, r1);
}

VTM_AVX2 static int dense_batch_avx2(float *y, const float *w, const float *x,
                                     int rows, int cols, int ntok,
                                     int transposed) {
    dense_batch_avx2_job job;
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
                      dense_batch_avx2_worker, &job);
    free(job.acc_all);
    return 0;
}

VTM_AVX2 int vtm_avx2_tmv_f32_batch(float *y, const float *w, const float *x,
                                    int rows, int cols, int n_tokens,
                                    int transposed) {
    return dense_batch_avx2(y, w, x, rows, cols, n_tokens, transposed);
}

VTM_AVX2 int vtm_avx2_mm_f32_batch(float *y, const float *a, const float *b,
                                   int m, int k, int n_tokens) {
    return dense_batch_avx2(y, a, b, m, k, n_tokens, 0);
}

// --- батчевые TMV/MM (задача 0122) -----------------------------------------
//
// y[r*N + j] = sum_t trit(W)[r,t] * x[t*N + j]. Как в NEON-версии
// (backend_neon.c): по вектору идут N токенов, трит W[r,t] — скаляр, поэтому
// слово pack20 распаковывается и переводится в f32 один раз на строку и
// блок, а веса читаются один раз на N токенов.
//
// Арифметика блока та же, что у батчевого скалярного пути: f32-аккумуляторы
// (8 лент на 8 токенов), затем сумма блока в double, умножение на масштаб
// блока и сложение блоков в double.

VTM_AVX2 static inline void trits_to_f32_avx2(const int8_t *t, float *out,
                                              int len) {
    int i = 0;
    for (; i + 16 <= len; i += 16) {
        __m128i b = _mm_loadu_si128((const __m128i *)(const void *)(t + i));
        __m256i lo = _mm256_cvtepi8_epi32(b);
        __m256i hi = _mm256_cvtepi8_epi32(_mm_srli_si128(b, 8));
        _mm256_storeu_ps(out + i, _mm256_cvtepi32_ps(lo));
        _mm256_storeu_ps(out + i + 8, _mm256_cvtepi32_ps(hi));
    }
    for (; i < len; i++) out[i] = (float)t[i];
}

// Начать/добавить сумму блока: 8 f32-лент s -> acc[jj..jj+8) в double.
VTM_AVX2 static inline void acc_add8(double *acc, int jj, __m256 s,
                                     double scale) {
    __m256d d0 = _mm256_cvtps_pd(_mm256_castps256_ps128(s));
    __m256d d1 = _mm256_cvtps_pd(_mm256_extractf128_ps(s, 1));
    __m256d sc = _mm256_set1_pd(scale);
    _mm256_storeu_pd(acc + jj,
                     _mm256_add_pd(_mm256_loadu_pd(acc + jj),
                                   _mm256_mul_pd(d0, sc)));
    _mm256_storeu_pd(acc + jj + 4,
                     _mm256_add_pd(_mm256_loadu_pd(acc + jj + 4),
                                   _mm256_mul_pd(d1, sc)));
}

VTM_AVX2 static inline void acc_add4(double *acc, int jj, __m128 s,
                                     double scale) {
    __m256d d = _mm256_cvtps_pd(s);
    _mm256_storeu_pd(acc + jj,
                     _mm256_add_pd(_mm256_loadu_pd(acc + jj),
                                   _mm256_mul_pd(d, _mm256_set1_pd(scale))));
}

// Точечное произведение f32 на f32 (8 лент, 4 независимых аккумулятора):
// хвост из 1-3 токенов считается по столбцам (см. dot_f32_f32 в
// backend_neon.c), вход хвоста лежит contiguous [ntail][cols].
VTM_AVX2 static inline double dot_f32_f32_avx2(const float *a, const float *b,
                                               int n) {
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
    double s;
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),
                             _mm256_loadu_ps(b + i), a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8),
                             _mm256_loadu_ps(b + i + 8), a1);
        a2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16),
                             _mm256_loadu_ps(b + i + 16), a2);
        a3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24),
                             _mm256_loadu_ps(b + i + 24), a3);
    }
    s = acc_f64(_mm256_add_ps(_mm256_add_ps(a0, a1),
                              _mm256_add_ps(a2, a3)));
    for (; i < n; i++) s += (double)a[i] * (double)b[i];
    return s;
}

// Один блок масштаба: len тритов tp (начало блока — lo во входе x), ntok
// токенов, масштаб scale. Результат копится в acc[0..ntok). xtail — хвост
// 1-3 токенов, лежащий по столбцам (xtail[j*cols + lo ...]).
VTM_AVX2 static inline void trit_batch_avx2_block(const int8_t *tp, float *fb,
                                                  int len, int64_t lo,
                                                  const float *x, int ntok,
                                                  double *acc, double scale,
                                                  const float *xtail,
                                                  int xtail_cols, int ntail) {
    int jj;

    trits_to_f32_avx2(tp, fb, len);

    // 8 токенов за шаг, 4 независимых аккумулятора (t mod 4).
    for (jj = 0; jj + 8 <= ntok; jj += 8) {
        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = a0, a2 = a0, a3 = a0;
        int t = 0;
        for (; t + 4 <= len; t += 4) {
            const float *xp = x + (size_t)(lo + t) * (size_t)ntok
                            + (size_t)jj;
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(xp),
                                 _mm256_set1_ps(fb[t + 0]), a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(xp + ntok),
                                 _mm256_set1_ps(fb[t + 1]), a1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(xp + 2 * ntok),
                                 _mm256_set1_ps(fb[t + 2]), a2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(xp + 3 * ntok),
                                 _mm256_set1_ps(fb[t + 3]), a3);
        }
        for (; t < len; t++) {
            const float *xp = x + (size_t)(lo + t) * (size_t)ntok
                            + (size_t)jj;
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(xp),
                                 _mm256_set1_ps(fb[t]), a0);
        }
        acc_add8(acc, jj,
                 _mm256_add_ps(_mm256_add_ps(a0, a1),
                               _mm256_add_ps(a2, a3)), scale);
    }
    // Хвост 4..7 токенов — 4-ленточное ядро.
    for (; jj + 4 <= ntok; jj += 4) {
        __m128 a0 = _mm_setzero_ps();
        __m128 a1 = a0, a2 = a0, a3 = a0;
        int t = 0;
        for (; t + 4 <= len; t += 4) {
            const float *xp = x + (size_t)(lo + t) * (size_t)ntok
                            + (size_t)jj;
            a0 = _mm_fmadd_ps(_mm_loadu_ps(xp), _mm_set1_ps(fb[t + 0]), a0);
            a1 = _mm_fmadd_ps(_mm_loadu_ps(xp + ntok),
                              _mm_set1_ps(fb[t + 1]), a1);
            a2 = _mm_fmadd_ps(_mm_loadu_ps(xp + 2 * ntok),
                              _mm_set1_ps(fb[t + 2]), a2);
            a3 = _mm_fmadd_ps(_mm_loadu_ps(xp + 3 * ntok),
                              _mm_set1_ps(fb[t + 3]), a3);
        }
        for (; t < len; t++) {
            const float *xp = x + (size_t)(lo + t) * (size_t)ntok
                            + (size_t)jj;
            a0 = _mm_fmadd_ps(_mm_loadu_ps(xp), _mm_set1_ps(fb[t]), a0);
        }
        acc_add4(acc, jj, _mm_add_ps(_mm_add_ps(a0, a1),
                                     _mm_add_ps(a2, a3)), scale);
    }
    // Хвост 1-3 токенов — по столбцам: fb и xtail contiguous.
    for (jj = 0; jj < ntail; jj++)
        acc[ntok - ntail + jj] += dot_f32_f32_avx2(
            fb, xtail + (size_t)jj * (size_t)xtail_cols + lo, len) * scale;
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
} trit_batch_avx2_job;

VTM_AVX2 static void trit_batch_avx2_rows(trit_batch_avx2_job *j, int tid,
                                          int r0, int r1) {
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
            trit_batch_avx2_block(tp, fb, (int)(e - st), st - base, x, ntok,
                                  acc,
                                  (double)load_f32_unaligned(scales, s / sb),
                                  xtail, cols, ntail);
        }
        for (jj = 0; jj < ntok; jj++)
            j->y[(size_t)r * (size_t)ntok + (size_t)jj] = (float)acc[jj];
    }
}

// Колбэк пула — без target-атрибута (его тип общий vtm_rows_fn).
static void trit_batch_avx2_worker(void *p, int tid, int r0, int r1) {
    trit_batch_avx2_rows((trit_batch_avx2_job *)p, tid, r0, r1);
}

static int trit_batch_avx2(float *y, const uint8_t *w, const float *scales,
                           const float *x, int rows, int cols, int ntok,
                           int scale_block) {
    trit_batch_avx2_job job;
    int vt;

    if (!y || !w || !scales || !x) return -1;
    if (rows < 0 || cols < 0 || ntok < 0) return -1;
    if (scale_block != (int)VTM_SCALE_BLOCK_128 &&
        scale_block != (int)VTM_SCALE_BLOCK_160) return -1;
    if (rows == 0 || cols == 0 || ntok == 0) return 0;
    lut_init();
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
                      trit_batch_avx2_worker, &job);
    free((void *)job.xtail);
    free(job.acc_all);
    return 0;
}

// TMV и MM тритовой матрицы — одна арифметика (различаются имена аргументов).
VTM_AVX2 int vtm_avx2_tmv_blocks_batch(float *y, const uint8_t *w,
                                       const float *scales, const float *x,
                                       int rows, int cols, int n_tokens,
                                       int scale_block) {
    return trit_batch_avx2(y, w, scales, x, rows, cols, n_tokens,
                           scale_block);
}

VTM_AVX2 int vtm_avx2_mm_batch(float *y, const uint8_t *w,
                               const float *scales, const float *b, int m,
                               int k, int n_tokens, int scale_block) {
    return trit_batch_avx2(y, w, scales, b, m, k, n_tokens, scale_block);
}

int vtm_avx2_available(void) {
    static int cached = -1;
    if (cached < 0) {
        __builtin_cpu_init();
        cached = (__builtin_cpu_supports("avx2") &&
                  __builtin_cpu_supports("fma")) ? 1 : 0;
    }
    return cached;
}

#endif // __x86_64__
