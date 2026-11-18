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
            unpack20(words[wi],
                     tp + (ptrdiff_t)(wi * VTM_TRITS_PER_WORD - st));
        sum = dot_i8_f32(tp, x + (st - base), (int)(e - st));
        acc += sum * (double)scales[(size_t)(s / sb)];
    }
    return acc;
}

VTM_AVX2 int vtm_avx2_tmv_blocks(float *y, const uint8_t *w, const float *scales,
                        const float *x, int rows, int cols,
                        int scale_block) {
    const uint32_t *words = (const uint32_t *)w;
    int r;
    if (!y || !w || !scales || !x) return -1;
    if (rows < 0 || cols < 0) return -1;
    if (scale_block != (int)VTM_SCALE_BLOCK_128 &&
        scale_block != (int)VTM_SCALE_BLOCK_160) return -1;
    if (rows == 0 || cols == 0) return 0;
    lut_init();
    for (r = 0; r < rows; r++) {
        int64_t base = (int64_t)r * (int64_t)cols;
        y[r] = (float)row_dot_trit(words, scales, x, base, cols,
                                   scale_block);
    }
    return 0;
}

VTM_AVX2 int vtm_avx2_rmsnorm(float *y, const float *x, const float *w,
                     int n, float eps) {
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

VTM_AVX2 int vtm_avx2_mm_trit(float *y, const uint8_t *w, const float *scales,
                     const float *b, int m, int k, int n, int scale_block) {
    const uint32_t *words = (const uint32_t *)w;
    const int64_t sb = (int64_t)scale_block;
    double *acc;
    int r;

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

    if (n == 1) {
        for (r = 0; r < m; r++) {
            int64_t base = (int64_t)r * (int64_t)k;
            y[r] = (float)row_dot_trit(words, scales, b, base, k,
                                       scale_block);
        }
        return 0;
    }

    acc = (double *)calloc((size_t)n, sizeof(double));
    if (!acc) return -1;
    for (r = 0; r < m; r++) {
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
                unpack20(words[wi],
                         tp + (ptrdiff_t)(wi * VTM_TRITS_PER_WORD - st));
            len = (int)(e - st);
            lo = st - base;
            scale = (double)scales[(size_t)(s / sb)];

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
    free(acc);
    return 0;
}

VTM_AVX2 int vtm_avx2_mm_f32(float *y, const float *a, const float *b,
                    int m, int k, int n) {
    int r, c, t;
    if (!y || !a || !b) return -1;
    if (m < 0 || k < 0 || n < 0) return -1;
    if (m == 0 || n == 0) return 0;
    if (k == 0) {
        memset(y, 0, (size_t)m * (size_t)n * sizeof(float));
        return 0;
    }

    // 8 столбцов за раз; куски по 16 t копятся в f32, затем в double.
    for (r = 0; r < m; r++) {
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
    return 0;
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
