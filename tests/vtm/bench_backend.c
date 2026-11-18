// Бенчмарк и проверка SIMD-бэкенда VTM (src/vtm/backend_neon.c,
// backend_avx2.c). Один бинарник делает три вещи:
//
//   ./bench_backend                 — GFLOP/s: scalar-lut243 / scalar-direct /
//                                     simd (матрица 1024x1024) + MM + RMSN;
//   ./bench_backend --check         — «simd == scalar» на наборе форм (в т.ч.
//                                     нечётные cols, блок 128/160, MM, RMSN);
//                                     вызывается из `make test`, код возврата
//                                     != 0 при расхождении > 1e-5;
//   VTM_BACKEND=simd ./bench_backend --logits ../../demo/model.vtm
//                                   — прогон настоящей модели: логиты
//                                     скалярного и выбранного бэкенда, их
//                                     max|diff| и argmax-токен.
//
// Бэкенд «кандидат» берётся из --backend или VTM_BACKEND (vtm_backend_from_env),
// по умолчанию — simd, если он доступен.
#include "vtm.h"
#include "vtm_backend.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CHECK_TOL 1e-5

// --- утилиты ---------------------------------------------------------------

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

static uint32_t rng_next(uint32_t *s) {
    uint32_t x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    return x;
}

// [-1, 1)
static float rng_float(uint32_t *s) {
    const int32_t half = 1 << 23;
    int32_t v = (int32_t)(rng_next(s) >> 8) - half;
    return (float)v / (float)half;
}

// Детерминированная тритовая матрица: n_trits тритов, упакованных по 20 в
// слово (padding — нулевые триты, как в формате).
static uint32_t *make_words(int n_trits, uint32_t seed) {
    size_t n_words = ((size_t)n_trits + VTM_TRITS_PER_WORD - 1)
                   / VTM_TRITS_PER_WORD;
    uint32_t *words = (uint32_t *)calloc(n_words, sizeof(uint32_t));
    size_t wi;
    if (!words) return NULL;
    for (wi = 0; wi < n_words; wi++) {
        int8_t trits[VTM_TRITS_PER_WORD];
        int t;
        for (t = 0; t < VTM_TRITS_PER_WORD; t++) {
            if ((int)(wi * VTM_TRITS_PER_WORD) + t >= n_trits) {
                trits[t] = 0;
            } else {
                uint32_t r = rng_next(&seed) % 3u;
                trits[t] = (int8_t)(r == 2u ? 1 : (r == 0u ? -1 : 0));
            }
        }
        words[wi] = vtm_trit_pack20(trits);
    }
    return words;
}

// scale > 0 — все масштабы равны scale; scale == 0 — случайные 0.01..0.03
// (типичный порядок для тритовой модели).
static float *make_scales(int n_trits, int scale_block, float scale,
                          uint32_t seed) {
    size_t n_scales = ((size_t)n_trits + (size_t)scale_block - 1)
                    / (size_t)scale_block;
    float *scales = (float *)malloc(n_scales * sizeof(float));
    size_t i;
    if (!scales) return NULL;
    for (i = 0; i < n_scales; i++) {
        if (scale > 0.0f)
            scales[i] = scale;
        else
            scales[i] = 0.01f + 0.02f * (float)(rng_next(&seed) % 1000u) / 999.0f;
    }
    return scales;
}

static float *make_vec(size_t n, uint32_t seed) {
    float *v = (float *)malloc(n * sizeof(float));
    size_t i;
    if (!v) return NULL;
    for (i = 0; i < n; i++) v[i] = rng_float(&seed);
    return v;
}

// max|a-b| и max|b| — чтобы отличить абсолютное расхождение от большого
// масштаба значений.
static double max_diff(const float *a, const float *b, size_t n,
                       double *max_ref) {
    double md = 0.0, mr = 0.0;
    size_t i;
    for (i = 0; i < n; i++) {
        double d = fabs((double)a[i] - (double)b[i]);
        if (d > md) md = d;
        if (fabs((double)b[i]) > mr) mr = fabs((double)b[i]);
    }
    if (max_ref) *max_ref = mr;
    return md;
}

// --- скалярные эталоны MM (в интерпретаторе MM считается здесь же, но
//     interp.c недоступен для правок — зона) --------------------------------

static void mm_trit_ref(float *y, const uint32_t *words, const float *scales,
                        const float *b, int m, int k, int n,
                        int scale_block) {
    int8_t *row = (int8_t *)malloc((size_t)k);
    int r, c, t;
    for (r = 0; r < m; r++) {
        int64_t base = (int64_t)r * (int64_t)k;
        int64_t wi = base / VTM_TRITS_PER_WORD;
        int64_t wi_end = (base + (int64_t)k - 1) / VTM_TRITS_PER_WORD;
        for (; wi <= wi_end; wi++) {
            int8_t trits[VTM_TRITS_PER_WORD];
            int tt;
            vtm_trit_unpack20(words[wi], trits);
            for (tt = 0; tt < VTM_TRITS_PER_WORD; tt++) {
                int64_t idx = wi * VTM_TRITS_PER_WORD + tt - base;
                if (idx >= 0 && idx < (int64_t)k) row[idx] = trits[tt];
            }
        }
        for (c = 0; c < n; c++) {
            double acc = 0.0;
            for (t = 0; t < k;) {
                int64_t blk = (base + t) / (int64_t)scale_block;
                int64_t e = (blk + 1) * (int64_t)scale_block - base;
                double block_acc = 0.0;
                if (e > (int64_t)k) e = (int64_t)k;
                for (; t < (int)e; t++)
                    block_acc += (double)row[t]
                               * (double)b[(size_t)t * (size_t)n + (size_t)c];
                acc += block_acc * (double)scales[(size_t)blk];
            }
            y[(size_t)r * (size_t)n + (size_t)c] = (float)acc;
        }
    }
    free(row);
}

static void mm_f32_ref(float *y, const float *a, const float *b,
                       int m, int k, int n) {
    int r, c, t;
    for (r = 0; r < m; r++)
        for (c = 0; c < n; c++) {
            double acc = 0.0;
            for (t = 0; t < k; t++)
                acc += (double)a[(size_t)r * (size_t)k + (size_t)t]
                     * (double)b[(size_t)t * (size_t)n + (size_t)c];
            y[(size_t)r * (size_t)n + (size_t)c] = (float)acc;
        }
}

// --- проверки --------------------------------------------------------------

static int failures;

static void report(const char *what, double diff, double ref, double tol) {
    int ok = diff <= tol;
    printf("  %-34s max|diff|=%.3e  max|ref|=%.3e  %s\n",
           what, diff, ref, ok ? "ok" : "FAIL");
    if (!ok) failures++;
}

// Возвращает 0, если simd == scalar в пределах tol.
static int check_tmv_case(int rows, int cols, int scale_block, float scale,
                          const vtm_backend *cand) {
    int n_trits = rows * cols;
    uint32_t seed = 0x1234567u + (uint32_t)rows * 131u + (uint32_t)cols;
    uint32_t *words = make_words(n_trits, seed);
    float *scales = make_scales(n_trits, scale_block, scale, seed ^ 0x9e37u);
    float *x = make_vec((size_t)cols, seed ^ 0x51u);
    float *y_ref = (float *)calloc((size_t)rows, sizeof(float));
    float *y_dir = (float *)calloc((size_t)rows, sizeof(float));
    float *y_cand = (float *)calloc((size_t)rows, sizeof(float));
    double ref = 0.0;
    int rc = 0;

    if (!words || !scales || !x || !y_ref || !y_dir || !y_cand) {
        fprintf(stderr, "check_tmv_case: нет памяти\n");
        failures++;
        rc = -1;
        goto done;
    }
    if (vtm_backend_scalar_lut.tmv_blocks(y_ref, (const uint8_t *)words,
                                          scales, x, rows, cols,
                                          scale_block) != 0 ||
        vtm_backend_scalar_direct.tmv_blocks(y_dir, (const uint8_t *)words,
                                             scales, x, rows, cols,
                                             scale_block) != 0 ||
        cand->tmv_blocks(y_cand, (const uint8_t *)words, scales, x, rows,
                         cols, scale_block) != 0) {
        fprintf(stderr, "check_tmv_case: ядро вернуло ошибку\n");
        failures++;
        rc = -1;
        goto done;
    }
    {
        char label[96];
        double d = max_diff(y_cand, y_ref, (size_t)rows, &ref);
        snprintf(label, sizeof(label), "TMV %dx%d блок %d scale %.2f",
                 rows, cols, scale_block, (double)scale);
        report(label, d, ref, CHECK_TOL);
        d = max_diff(y_dir, y_ref, (size_t)rows, NULL);
        snprintf(label, sizeof(label), "TMV direct %dx%d блок %d scale %.2f",
                 rows, cols, scale_block, (double)scale);
        report(label, d, ref, CHECK_TOL);
    }
done:
    free(words);
    free(scales);
    free(x);
    free(y_ref);
    free(y_dir);
    free(y_cand);
    return rc;
}

static void check_mm_case(int m, int k, int n, int scale_block, float scale) {
    int n_trits = m * k;
    uint32_t seed = 0xabcdefu + (uint32_t)m * 17u + (uint32_t)k * 3u
                  + (uint32_t)n;
    uint32_t *words = make_words(n_trits, seed);
    float *scales = make_scales(n_trits, scale_block, scale, seed ^ 0x77u);
    float *a = make_vec((size_t)m * (size_t)k, seed ^ 0x11u);
    float *b = make_vec((size_t)k * (size_t)n, seed ^ 0x22u);
    float *y_ref = (float *)calloc((size_t)m * (size_t)n, sizeof(float));
    float *y_cand = (float *)calloc((size_t)m * (size_t)n, sizeof(float));
    char label[96];
    double ref = 0.0;

    if (!words || !scales || !a || !b || !y_ref || !y_cand) {
        fprintf(stderr, "check_mm_case: нет памяти\n");
        failures++;
        goto done;
    }
    mm_trit_ref(y_ref, words, scales, b, m, k, n, scale_block);
    if (vtm_simd_mm_trit(y_cand, (const uint8_t *)words, scales, b, m, k, n,
                         scale_block) != 0) {
        fprintf(stderr, "check_mm_case: ядро вернуло ошибку\n");
        failures++;
        goto done;
    }
    snprintf(label, sizeof(label), "MM trit %dx%dx%d блок %d scale %.2f",
             m, k, n, scale_block, (double)scale);
    report(label, max_diff(y_cand, y_ref, (size_t)m * (size_t)n, &ref), ref,
           CHECK_TOL);

    {  // f32-эталон: формы маленькие, double-цикл дешёвый
        mm_f32_ref(y_ref, a, b, m, k, n);
        if (vtm_simd_mm_f32(y_cand, a, b, m, k, n) != 0) {
            fprintf(stderr, "check_mm_case: f32 ядро вернуло ошибку\n");
            failures++;
            goto done;
        }
        snprintf(label, sizeof(label), "MM f32 %dx%dx%d", m, k, n);
        report(label, max_diff(y_cand, y_ref, (size_t)m * (size_t)n, &ref),
               ref, CHECK_TOL);
    }
done:
    free(words);
    free(scales);
    free(a);
    free(b);
    free(y_ref);
    free(y_cand);
}

static void check_rmsnorm_case(int n, const vtm_backend *cand) {
    uint32_t seed = 0x31337u + (uint32_t)n;
    float *x = make_vec((size_t)n, seed);
    float *w = make_vec((size_t)n, seed ^ 0x99u);
    float *y_ref = (float *)calloc((size_t)n, sizeof(float));
    float *y_cand = (float *)calloc((size_t)n, sizeof(float));
    char label[64];
    double ref = 0.0;
    float eps = 0.001f;

    if (!x || !w || !y_ref || !y_cand) {
        fprintf(stderr, "check_rmsnorm_case: нет памяти\n");
        failures++;
        goto done;
    }
    vtm_backend_scalar_lut.rmsnorm(y_ref, x, w, n, eps);
    if (cand->rmsnorm(y_cand, x, w, n, eps) != 0) {
        fprintf(stderr, "check_rmsnorm_case: ядро вернуло ошибку\n");
        failures++;
        goto done;
    }
    snprintf(label, sizeof(label), "RMSN n=%d", n);
    report(label, max_diff(y_cand, y_ref, (size_t)n, &ref), ref, CHECK_TOL);
done:
    free(x);
    free(w);
    free(y_ref);
    free(y_cand);
}

static int run_checks(const vtm_backend *cand) {
    static const int shapes[][2] = {
        {1, 1}, {1, 20}, {7, 37}, {5, 53}, {17, 128}, {33, 41}, {64, 160},
        {128, 128}, {12, 1000}
    };
    static const int mmshapes[][3] = {
        {5, 7, 3}, {8, 16, 8}, {3, 1000, 17}, {64, 128, 1}, {16, 32, 64},
        {9, 13, 5}
    };
    size_t i;

    failures = 0;
    printf("SIMD: %s, доступен: %s\n", vtm_simd_isa(),
           vtm_simd_available() ? "да" : "нет (ожидается откат на scalar)");

    for (i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++) {
        check_tmv_case(shapes[i][0], shapes[i][1], 128, 0.02f, cand);
        check_tmv_case(shapes[i][0], shapes[i][1], 160, 0.02f, cand);
        check_tmv_case(shapes[i][0], shapes[i][1], 128, 1.0f, cand);
        check_tmv_case(shapes[i][0], shapes[i][1], 160, 1.0f, cand);
    }
    for (i = 0; i < sizeof(mmshapes) / sizeof(mmshapes[0]); i++) {
        check_mm_case(mmshapes[i][0], mmshapes[i][1], mmshapes[i][2], 128,
                      0.02f);
        check_mm_case(mmshapes[i][0], mmshapes[i][1], mmshapes[i][2], 160,
                      1.0f);
    }
    check_rmsnorm_case(1, cand);
    check_rmsnorm_case(7, cand);
    check_rmsnorm_case(129, cand);
    check_rmsnorm_case(1000, cand);

    if (failures) {
        printf("simd == scalar: FAIL (%d случаев > %.0e)\n", failures,
               CHECK_TOL);
        return 1;
    }
    printf("simd == scalar: ok (в пределах %.0e)\n", CHECK_TOL);
    return 0;
}

// --- бенчмарк --------------------------------------------------------------

static double bench_tmv(const vtm_backend *b, float *y, const uint8_t *words,
                        const float *scales, const float *x, int rows,
                        int cols, int scale_block, int iters) {
    double t0, t1;
    int i;
    if (b->tmv_blocks(y, words, scales, x, rows, cols, scale_block) != 0)
        return -1.0;
    t0 = now_ms();
    for (i = 0; i < iters; i++)
        b->tmv_blocks(y, words, scales, x, rows, cols, scale_block);
    t1 = now_ms();
    return (t1 - t0) / (double)iters;
}

static void bench_tmv_table(int rows, int cols, int scale_block, int iters,
                            const vtm_backend *cand) {
    int n_trits = rows * cols;
    uint32_t seed = 0x5eedu + (uint32_t)rows + (uint32_t)cols;
    uint32_t *words = make_words(n_trits, seed);
    float *scales = make_scales(n_trits, scale_block, 0.0f, seed ^ 0x1234u);
    float *x = make_vec((size_t)cols, seed ^ 0x77u);
    float *y_ref = (float *)calloc((size_t)rows, sizeof(float));
    float *y_cand = (float *)calloc((size_t)rows, sizeof(float));
    const vtm_backend *backends[3];
    double ms[3];
    double base_ms, ref = 0.0;
    double flops = 2.0 * (double)rows * (double)cols;
    int i;

    if (!words || !scales || !x || !y_ref || !y_cand) {
        fprintf(stderr, "bench_tmv_table: нет памяти\n");
        goto done;
    }
    backends[0] = &vtm_backend_scalar_lut;
    backends[1] = &vtm_backend_scalar_direct;
    backends[2] = cand;

    for (i = 0; i < 3; i++)
        ms[i] = bench_tmv(backends[i], y_cand, (const uint8_t *)words, scales,
                          x, rows, cols, scale_block, iters);
    base_ms = ms[0];
    if (vtm_backend_scalar_lut.tmv_blocks(y_ref, (const uint8_t *)words,
                                          scales, x, rows, cols,
                                          scale_block) != 0) {
        fprintf(stderr, "bench_tmv_table: scalar вернул ошибку\n");
        goto done;
    }

    printf("TMV %dx%d, блок масштаба %d, %d итераций\n", rows, cols,
           scale_block, iters);
    printf("  %-16s %10s %10s %10s   %s\n", "бэкенд", "мс/итер", "GFLOP/s",
           "x к scalar", "max|diff| vs scalar");
    for (i = 0; i < 3; i++) {
        double d;
        if (ms[i] < 0.0) {
            printf("  %-16s %10s\n", backends[i]->name, "ошибка ядра");
            continue;
        }
        if (backends[i]->tmv_blocks(y_cand, (const uint8_t *)words, scales,
                                    x, rows, cols, scale_block) != 0)
            d = -1.0;
        else
            d = max_diff(y_cand, y_ref, (size_t)rows, &ref);
        printf("  %-16s %10.4f %10.2f %10.2fx   %.2e\n", backends[i]->name,
               ms[i], flops / (ms[i] * 1e6), base_ms / ms[i], d);
    }
    printf("  (эталон значений — %s, max|ref|=%.2f)\n",
           vtm_backend_scalar_lut.name, ref);
done:
    free(words);
    free(scales);
    free(x);
    free(y_ref);
    free(y_cand);
}

static double bench_mm_trit(const uint8_t *words, const float *scales,
                            const float *b, int m, int k, int n,
                            int scale_block, int iters) {
    float *y = (float *)calloc((size_t)m * (size_t)n, sizeof(float));
    double t0, t1;
    int i;
    if (!y) return -1.0;
    if (vtm_simd_mm_trit(y, words, scales, b, m, k, n, scale_block) != 0) {
        free(y);
        return -1.0;
    }
    t0 = now_ms();
    for (i = 0; i < iters; i++)
        vtm_simd_mm_trit(y, words, scales, b, m, k, n, scale_block);
    t1 = now_ms();
    free(y);
    return (t1 - t0) / (double)iters;
}

static void bench_mm_table(int m, int k, int n, int scale_block, int iters) {
    int n_trits = m * k;
    uint32_t seed = 0x77u + (uint32_t)m + (uint32_t)k + (uint32_t)n;
    uint32_t *words = make_words(n_trits, seed);
    float *scales = make_scales(n_trits, scale_block, 0.0f, seed ^ 0x5u);
    float *b = make_vec((size_t)k * (size_t)n, seed ^ 0x99u);
    float *y_ref = (float *)calloc((size_t)m * (size_t)n, sizeof(float));
    float *y_cand = (float *)calloc((size_t)m * (size_t)n, sizeof(float));
    double ms_ref, ms_cand, flops = 2.0 * (double)m * (double)k * (double)n;
    double t0, t1, ref = 0.0;

    if (!words || !scales || !b || !y_ref || !y_cand) {
        fprintf(stderr, "bench_mm_table: нет памяти\n");
        goto done;
    }
    t0 = now_ms();
    mm_trit_ref(y_ref, words, scales, b, m, k, n, scale_block);
    t1 = now_ms();
    ms_ref = t1 - t0;
    ms_cand = bench_mm_trit((const uint8_t *)words, scales, b, m, k, n,
                            scale_block, iters);
    if (ms_cand < 0.0) {
        printf("MM trit %dx%dx%d: SIMD-ядро вернуло ошибку\n", m, k, n);
        goto done;
    }
    if (vtm_simd_mm_trit(y_cand, (const uint8_t *)words, scales, b, m, k, n,
                         scale_block) != 0)
        goto done;
    printf("MM trit %dx%dx%d, блок %d: scalar %.3f мс (%.2f GFLOP/s), "
           "simd %.3f мс (%.2f GFLOP/s), %.2fx, max|diff|=%.2e, max|ref|=%.2f\n",
           m, k, n, scale_block, ms_ref, flops / (ms_ref * 1e6), ms_cand,
           flops / (ms_cand * 1e6), ms_ref / ms_cand,
           max_diff(y_cand, y_ref, (size_t)m * (size_t)n, &ref), ref);
done:
    free(words);
    free(scales);
    free(b);
    free(y_ref);
    free(y_cand);
}

static void bench_mm_f32_table(int m, int k, int n, int iters) {
    uint32_t seed = 0x1234u + (uint32_t)m + (uint32_t)k;
    float *a = make_vec((size_t)m * (size_t)k, seed);
    float *b = make_vec((size_t)k * (size_t)n, seed ^ 0x42u);
    float *y_ref = (float *)calloc((size_t)m * (size_t)n, sizeof(float));
    float *y_cand = (float *)calloc((size_t)m * (size_t)n, sizeof(float));
    double ms_ref, ms_cand, flops = 2.0 * (double)m * (double)k * (double)n;
    double t0, t1, ref = 0.0;
    int i;

    if (!a || !b || !y_ref || !y_cand) {
        fprintf(stderr, "bench_mm_f32_table: нет памяти\n");
        goto done;
    }
    t0 = now_ms();
    mm_f32_ref(y_ref, a, b, m, k, n);
    t1 = now_ms();
    ms_ref = t1 - t0;

    if (vtm_simd_mm_f32(y_cand, a, b, m, k, n) != 0) goto done;
    t0 = now_ms();
    for (i = 0; i < iters; i++)
        vtm_simd_mm_f32(y_cand, a, b, m, k, n);
    t1 = now_ms();
    ms_cand = (t1 - t0) / (double)iters;
    printf("MM f32 %dx%dx%d: scalar %.3f мс (%.2f GFLOP/s), simd %.3f мс "
           "(%.2f GFLOP/s), %.2fx, max|diff|=%.2e, max|ref|=%.2f\n",
           m, k, n, ms_ref, flops / (ms_ref * 1e6), ms_cand,
           flops / (ms_cand * 1e6), ms_ref / ms_cand,
           max_diff(y_cand, y_ref, (size_t)m * (size_t)n, &ref), ref);
done:
    free(a);
    free(b);
    free(y_ref);
    free(y_cand);
}

// --- многопоточность (задача 0077) -----------------------------------------
//
// Каждое ядро считается при 1/2/4/8 потоках в одном процессе
// (vtm_threads_set — тот же механизм, что VTM_THREADS в env). Проверяется,
// что результат не зависит от числа потоков, и печатается время.
// TMV/MM делят строки выхода, поэтому совпадение с nt=1 побитовое; RMSN
// параллелит сумму квадратов по блокам — расхождение только в порядке
// double-сложения (в f32-выходе обычно не видно), печатается max|diff|.

static const int g_nts[] = {1, 2, 4, 8};
static int g_thr_fail;

static void thr_report(const char *label, int nt, const float *got,
                       const float *ref, size_t n, double ms,
                       double base_ms) {
    size_t i;
    int bitwise = 1;
    double d, mr = 0.0;

    for (i = 0; i < n; i++)
        if (got[i] != ref[i]) {
            bitwise = 0;
            break;
        }
    d = max_diff(got, ref, n, &mr);
    printf("  %-38s nt=%d %9.3f мс %6.2fx  max|diff|=%.2e %s\n", label, nt,
           ms, base_ms > 0.0 ? base_ms / ms : 0.0, d,
           bitwise ? "(побитово)"
                   : (d <= CHECK_TOL ? "(<=1e-5)" : "(FAIL)"));
    if (d > CHECK_TOL) g_thr_fail++;
}

static void thr_tmv(const vtm_backend *b, int rows, int cols,
                    int scale_block, int iters) {
    int n_trits = rows * cols;
    uint32_t seed = 0xb007u + (uint32_t)rows * 3u + (uint32_t)cols;
    uint32_t *words = make_words(n_trits, seed);
    float *scales = make_scales(n_trits, scale_block, 0.0f, seed ^ 0x9e37u);
    float *x = make_vec((size_t)cols, seed ^ 0x51u);
    float *ref = (float *)calloc((size_t)rows, sizeof(float));
    float *got = (float *)calloc((size_t)rows, sizeof(float));
    char label[96];
    double base_ms = 0.0;
    int i;

    if (!words || !scales || !x || !ref || !got) {
        fprintf(stderr, "thr_tmv: нет памяти\n");
        g_thr_fail++;
        goto done;
    }
    snprintf(label, sizeof label, "%s TMV %dx%d", b->name, rows, cols);
    for (i = 0; i < 4; i++) {
        double ms;
        vtm_threads_set(g_nts[i]);
        ms = bench_tmv(b, got, (const uint8_t *)words, scales, x, rows, cols,
                       scale_block, iters);
        if (ms < 0.0) {
            printf("  %-38s nt=%d ошибка ядра\n", label, g_nts[i]);
            g_thr_fail++;
            continue;
        }
        if (i == 0) {
            memcpy(ref, got, (size_t)rows * sizeof(float));
            base_ms = ms;
        }
        thr_report(label, g_nts[i], got, ref, (size_t)rows, ms, base_ms);
    }
done:
    free(words);
    free(scales);
    free(x);
    free(ref);
    free(got);
}

static double thr_time_mm_trit(const uint8_t *words, const float *scales,
                               const float *b, float *y, int m, int k, int n,
                               int scale_block, int iters) {
    double t0, t1;
    int i;
    if (vtm_simd_mm_trit(y, words, scales, b, m, k, n, scale_block) != 0)
        return -1.0;
    t0 = now_ms();
    for (i = 0; i < iters; i++)
        vtm_simd_mm_trit(y, words, scales, b, m, k, n, scale_block);
    t1 = now_ms();
    return (t1 - t0) / (double)iters;
}

static void thr_mm_trit(int m, int k, int n, int scale_block, int iters) {
    int n_trits = m * k;
    uint32_t seed = 0xdd77u + (uint32_t)m * 17u + (uint32_t)k * 3u
                  + (uint32_t)n;
    uint32_t *words = make_words(n_trits, seed);
    float *scales = make_scales(n_trits, scale_block, 0.0f, seed ^ 0x77u);
    float *b = make_vec((size_t)k * (size_t)n, seed ^ 0x99u);
    float *ref = (float *)calloc((size_t)m * (size_t)n, sizeof(float));
    float *got = (float *)calloc((size_t)m * (size_t)n, sizeof(float));
    char label[96];
    double base_ms = 0.0;
    int i;

    if (!words || !scales || !b || !ref || !got) {
        fprintf(stderr, "thr_mm_trit: нет памяти\n");
        g_thr_fail++;
        goto done;
    }
    snprintf(label, sizeof label, "MM trit %dx%dx%d", m, k, n);
    for (i = 0; i < 4; i++) {
        double ms;
        vtm_threads_set(g_nts[i]);
        ms = thr_time_mm_trit((const uint8_t *)words, scales, b, got, m, k,
                              n, scale_block, iters);
        if (ms < 0.0) {
            printf("  %-38s nt=%d ошибка ядра\n", label, g_nts[i]);
            g_thr_fail++;
            continue;
        }
        if (i == 0) {
            memcpy(ref, got, (size_t)m * (size_t)n * sizeof(float));
            base_ms = ms;
        }
        thr_report(label, g_nts[i], got, ref, (size_t)m * (size_t)n, ms,
                   base_ms);
    }
done:
    free(words);
    free(scales);
    free(b);
    free(ref);
    free(got);
}

static void thr_mm_f32(int m, int k, int n, int iters) {
    uint32_t seed = 0x77u + (uint32_t)m + (uint32_t)k + (uint32_t)n;
    float *a = make_vec((size_t)m * (size_t)k, seed);
    float *b = make_vec((size_t)k * (size_t)n, seed ^ 0x42u);
    float *ref = (float *)calloc((size_t)m * (size_t)n, sizeof(float));
    float *got = (float *)calloc((size_t)m * (size_t)n, sizeof(float));
    char label[96];
    double base_ms = 0.0;
    int i;

    if (!a || !b || !ref || !got) {
        fprintf(stderr, "thr_mm_f32: нет памяти\n");
        g_thr_fail++;
        goto done;
    }
    snprintf(label, sizeof label, "MM f32 %dx%dx%d", m, k, n);
    for (i = 0; i < 4; i++) {
        double t0, t1, ms;
        int it;
        vtm_threads_set(g_nts[i]);
        if (vtm_simd_mm_f32(got, a, b, m, k, n) != 0) {
            printf("  %-38s nt=%d ошибка ядра\n", label, g_nts[i]);
            g_thr_fail++;
            continue;
        }
        t0 = now_ms();
        for (it = 0; it < iters; it++)
            vtm_simd_mm_f32(got, a, b, m, k, n);
        t1 = now_ms();
        ms = (t1 - t0) / (double)iters;
        if (i == 0) {
            memcpy(ref, got, (size_t)m * (size_t)n * sizeof(float));
            base_ms = ms;
        }
        thr_report(label, g_nts[i], got, ref, (size_t)m * (size_t)n, ms,
                   base_ms);
    }
done:
    free(a);
    free(b);
    free(ref);
    free(got);
}

static void thr_rmsnorm(const vtm_backend *b, int n, int iters) {
    uint32_t seed = 0x31337u + (uint32_t)n;
    float *x = make_vec((size_t)n, seed);
    float *w = make_vec((size_t)n, seed ^ 0x99u);
    float *ref = (float *)calloc((size_t)n, sizeof(float));
    float *got = (float *)calloc((size_t)n, sizeof(float));
    char label[96];
    double base_ms = 0.0;
    int i;

    if (!x || !w || !ref || !got) {
        fprintf(stderr, "thr_rmsnorm: нет памяти\n");
        g_thr_fail++;
        goto done;
    }
    snprintf(label, sizeof label, "%s RMSN n=%d", b->name, n);
    for (i = 0; i < 4; i++) {
        double t0, t1, ms;
        int it;
        vtm_threads_set(g_nts[i]);
        if (b->rmsnorm(got, x, w, n, 0.001f) != 0) {
            printf("  %-38s nt=%d ошибка ядра\n", label, g_nts[i]);
            g_thr_fail++;
            continue;
        }
        t0 = now_ms();
        for (it = 0; it < iters; it++) b->rmsnorm(got, x, w, n, 0.001f);
        t1 = now_ms();
        ms = (t1 - t0) / (double)iters;
        if (i == 0) {
            memcpy(ref, got, (size_t)n * sizeof(float));
            base_ms = ms;
        }
        thr_report(label, g_nts[i], got, ref, (size_t)n, ms, base_ms);
    }
done:
    free(x);
    free(w);
    free(ref);
    free(got);
}

// Числа и проверка детерминизма 1/2/4/8 потоков. Возврат != 0 — расхождение
// больше 1e-5 (тогда это FAIL в make test).
static int run_threads(int iters) {
    const vtm_backend *simd = &vtm_backend_simd;
    int def = vtm_threads();

    g_thr_fail = 0;
    printf("Многопоточность: VTM_THREADS по умолчанию = %d (ядер ≤ %d), "
           "порог параллели %.0f MAC\n", def, VTM_THREADS_MAX,
           VTM_PAR_MIN_WORK);
    printf("nt=1 — прежнее однопоточное поведение; эталон для сравнения — "
           "результат при nt=1.\n");
    printf("\nTMV 1024x1024 (блок 160):\n");
    thr_tmv(&vtm_backend_scalar_lut, 1024, 1024, 160, iters);
    thr_tmv(simd, 1024, 1024, 160, iters);
    printf("\nTMV 512x512 (блок 160, тоже выше порога):\n");
    thr_tmv(&vtm_backend_scalar_lut, 512, 512, 160, iters);
    thr_tmv(simd, 512, 512, 160, iters);
    printf("\nTMV 128x128 (ниже порога — обязан остаться однопоточным):\n");
    thr_tmv(simd, 128, 128, 160, iters);
    printf("\nMM trit 1024x1024x1 (tied lm_head):\n");
    thr_mm_trit(1024, 1024, 1, 160, iters);
    printf("\nMM trit 128x1024x32:\n");
    thr_mm_trit(128, 1024, 32, 160, iters);
    printf("\nMM f32 256x256x256:\n");
    thr_mm_f32(256, 256, 256, iters > 10 ? 10 : iters);
    printf("\nRMSN n=524288 (порог %d):\n", VTM_PAR_RMSN_MIN);
    thr_rmsnorm(&vtm_backend_scalar_lut, 524288, iters);
    thr_rmsnorm(simd, 524288, iters);

    if (g_thr_fail) {
        printf("\nпотоки: FAIL (%d случаев > %.0e)\n", g_thr_fail, CHECK_TOL);
        return 1;
    }
    printf("\nпотоки: ok (результат не зависит от числа потоков)\n");
    return 0;
}

// --- логиты модели ---------------------------------------------------------

static int emit_first(void *user, int token, int position) {
    int *out = (int *)user;
    (void)position;
    *out = token;
    return 1;  // одного токена достаточно: логиты уже посчитаны
}

static const char *find_model(const char *explicit_path) {
    static const char *candidates[] = {
        "../../demo/model.vtm", "../demo/model.vtm", "demo/model.vtm",
        "../../../demo/model.vtm"
    };
    FILE *f;
    size_t i;
    if (explicit_path) {
        f = fopen(explicit_path, "rb");
        if (!f) return NULL;
        fclose(f);
        return explicit_path;
    }
    for (i = 0; i < sizeof(candidates) / sizeof(candidates[0]); i++) {
        f = fopen(candidates[i], "rb");
        if (f) {
            fclose(f);
            return candidates[i];
        }
    }
    return NULL;
}

// Прогон модели: возвращает logical argmax-токен через *token_out и копирует
// логиты (тензор "logits") в out. 0 — успех.
static int model_logits(const char *path, const vtm_backend *backend,
                        float *out, int cap, int *n_out, int *token_out) {
    vtm_t *m = vtm_load(path);
    const vtm_tensor *logits;
    int token = -1;
    vtm_status st;
    int i;

    if (!m) {
        fprintf(stderr, "vtm_load(%s): %s\n", path, vtm_last_error());
        return -1;
    }
    vtm_set_backend(m, backend);
    vtm_set_token(m, 1);
    vtm_set_position(m, 0);
    st = vtm_run(m, emit_first, &token);
    if (st != VTM_OK) {
        fprintf(stderr, "vtm_run: %s (%s)\n", vtm_strerror(st),
                vtm_error_message(m));
        vtm_free(m);
        return -1;
    }
    logits = vtm_tensor_by_name(m, "logits");
    if (!logits || !logits->f32) {
        fprintf(stderr, "в модели нет тензора logits\n");
        vtm_free(m);
        return -1;
    }
    *n_out = (int)logits->n_elements;
    if (*n_out > cap) *n_out = cap;
    for (i = 0; i < *n_out; i++) out[i] = logits->f32[i];
    *token_out = token;
    vtm_free(m);
    return 0;
}

static int run_logits(const char *explicit_path, const vtm_backend *cand) {
    const char *path = find_model(explicit_path);
    enum { MAX_LOGITS = 4096 };
    static float y_ref[MAX_LOGITS];
    static float y_cand[MAX_LOGITS];
    int n_ref = 0, n_cand = 0, tok_ref = -1, tok_cand = -1, i;
    double d, ref = 0.0;

    if (!path) {
        fprintf(stderr, "модель не найдена (--logits PATH)\n");
        return 2;
    }
    printf("модель: %s\n", path);
    if (model_logits(path, &vtm_backend_scalar_lut, y_ref, MAX_LOGITS,
                     &n_ref, &tok_ref) != 0)
        return 1;
    if (model_logits(path, cand, y_cand, MAX_LOGITS, &n_cand, &tok_cand) != 0)
        return 1;
    if (n_ref != n_cand) {
        fprintf(stderr, "разное число логитов: %d vs %d\n", n_ref, n_cand);
        return 1;
    }
    d = max_diff(y_cand, y_ref, (size_t)n_ref, &ref);
    printf("эталон: %s, кандидат: %s\n", vtm_backend_scalar_lut.name,
           cand->name);
    printf("logits эталон: ");
    for (i = 0; i < n_ref && i < 8; i++) printf("%.6f ", (double)y_ref[i]);
    printf("%s\n", n_ref > 8 ? "..." : "");
    printf("logits кандидат: ");
    for (i = 0; i < n_ref && i < 8; i++) printf("%.6f ", (double)y_cand[i]);
    printf("%s\n", n_ref > 8 ? "..." : "");
    printf("max|logits(кандидат) - logits(эталон)| = %.3e (max|ref|=%.2f)\n",
           d, ref);
    printf("argmax: эталон=%d кандидат=%d %s\n", tok_ref, tok_cand,
           tok_ref == tok_cand ? "(совпал)" : "(РАЗОШЁЛСЯ)");
    if (d > CHECK_TOL || tok_ref != tok_cand) {
        printf("логиты: FAIL (> %.0e или argmax разошёлся)\n", CHECK_TOL);
        return 1;
    }
    printf("логиты: ok (в пределах %.0e)\n", CHECK_TOL);
    return 0;
}

// --- main ------------------------------------------------------------------

static void usage(const char *prog) {
    printf("usage: %s [--rows N] [--cols N] [--iters N] [--scale-block 128|160]\n"
           "          [--backend auto|scalar|simd|direct] [--check] "
           "[--threads] [--logits PATH] [--no-mm]\n"
           "  env VTM_BACKEND — то же, что --backend (по умолчанию simd);\n"
           "  env VTM_THREADS — число потоков ядер (по умолчанию ядра, ≤ 8)\n",
           prog);
}

int main(int argc, char **argv) {
    int rows = 1024, cols = 1024, iters = 50, scale_block = 160;
    int check = 0, no_mm = 0, threads = 0;
    const char *logits_path = NULL;
    const char *backend_name = NULL;
    const vtm_backend *cand;
    int i;

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--rows") == 0 && i + 1 < argc) {
            rows = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--cols") == 0 && i + 1 < argc) {
            cols = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--scale-block") == 0 && i + 1 < argc) {
            scale_block = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            backend_name = argv[++i];
        } else if (strcmp(argv[i], "--check") == 0) {
            check = 1;
        } else if (strcmp(argv[i], "--threads") == 0) {
            threads = 1;
        } else if (strcmp(argv[i], "--no-mm") == 0) {
            no_mm = 1;
        } else if (strcmp(argv[i], "--logits") == 0) {
            logits_path = (i + 1 < argc) ? argv[++i] : "";
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "неизвестный аргумент: %s\n", argv[i]);
            usage(argv[0]);
            return 2;
        }
    }
    if (rows <= 0 || cols <= 0 || iters <= 0 || cols > 32768 || rows > 32768) {
        fprintf(stderr, "плохие размеры/итерации\n");
        return 2;
    }
    if (scale_block != 128 && scale_block != 160) {
        fprintf(stderr, "scale-block должен быть 128 или 160\n");
        return 2;
    }

    if (backend_name)
        cand = vtm_backend_by_name(backend_name);
    else
        cand = vtm_backend_from_env();
    if (!cand) {
        fprintf(stderr, "неизвестный бэкенд: %s\n", backend_name);
        return 2;
    }
    if (!backend_name && (!getenv("VTM_BACKEND") || !*getenv("VTM_BACKEND")))
        cand = &vtm_backend_simd;  // бенчмарк по умолчанию меряет SIMD

    if (threads)
        return run_threads(iters);
    if (logits_path)
        return run_logits(*logits_path ? logits_path : NULL, cand);
    if (check)
        return run_checks(cand);

    printf("VTM SIMD: isa=%s, доступен=%s, кандидат=%s\n", vtm_simd_isa(),
           vtm_simd_available() ? "да" : "нет", cand->name);
    bench_tmv_table(rows, cols, scale_block, iters, cand);
    if (!no_mm) {
        printf("\n");
        bench_mm_table(1024, 1024, 1, scale_block, iters);
        bench_mm_table(128, 1024, 32, scale_block, iters);
        bench_mm_f32_table(256, 256, 256, iters);
    }
    return 0;
}
