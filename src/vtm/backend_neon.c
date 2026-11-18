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

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
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
#define VTM_SIMD_X86 1
#else
#define VTM_SIMD_X86 0
#endif

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
                             * (double)scales[(size_t)cur_block];
                    cur_block = blk;
                    block_acc = 0.0;
                }
                // Строка может начинаться не с границы слова pack20:
                // слово берём по абсолютному индексу и перечитываем при
                // смене слова, а не по idx % 20.
                if (wi != cur_word) {
                    vtm_trit_unpack20(words[wi], trit);
                    cur_word = wi;
                }
                block_acc += (double)trit[abs_idx % VTM_TRITS_PER_WORD]
                           * (double)b[(size_t)idx * (size_t)n + (size_t)c];
            }
            if (cur_block >= 0)
                acc += block_acc * (double)scales[(size_t)cur_block];
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
            unpack20(words[wi],
                     tp + (ptrdiff_t)(wi * VTM_TRITS_PER_WORD - st));
        sum = dot_i8_f32(tp, x + (st - base), (int)(e - st));
        acc += sum * (double)scales[(size_t)(s / sb)];
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
                unpack20(words[wi],
                         tp + (ptrdiff_t)(wi * VTM_TRITS_PER_WORD - st));
            len = (int)(e - st);
            lo = st - base;
            scale = (double)scales[(size_t)(s / sb)];

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

#endif // VTM_SIMD_NEON

// --- выбор ядер и публичный контракт ---------------------------------------

typedef struct {
    int (*tmv_blocks)(float *, const uint8_t *, const float *, const float *,
                      int, int, int);
    int (*rmsnorm)(float *, const float *, const float *, int, float);
    int (*mm_trit)(float *, const uint8_t *, const float *, const float *,
                   int, int, int, int);
    int (*mm_f32)(float *, const float *, const float *, int, int, int);
} simd_kernels;

static const simd_kernels g_simd = {
#if VTM_SIMD_NEON
    tmv_neon_blocks,
    rmsnorm_neon,
    mm_trit_neon,
    mm_f32_neon
#elif VTM_SIMD_X86
    vtm_avx2_tmv_blocks,
    vtm_avx2_rmsnorm,
    vtm_avx2_mm_trit,
    vtm_avx2_mm_f32
#else
    NULL, NULL, NULL, NULL
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
    if (vtm_simd_available() && g_simd.mm_f32)
        return g_simd.mm_f32(y, a, b, m, k, n);
    return mm_f32_scalar(y, a, b, m, k, n);
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

const vtm_backend *vtm_backend_by_name(const char *name) {
    if (!name || !*name) return vtm_backend_from_env();
    if (strcmp(name, "auto") == 0)
        return vtm_simd_available() ? &vtm_backend_simd
                                    : &vtm_backend_scalar_lut;
    if (strcmp(name, "simd") == 0 || strcmp(name, "neon") == 0 ||
        strcmp(name, "avx2") == 0 || strcmp(name, "simd-neon") == 0 ||
        strcmp(name, "simd-avx2") == 0)
        return &vtm_backend_simd;
    if (strcmp(name, "scalar") == 0 || strcmp(name, "lut") == 0 ||
        strcmp(name, "scalar-lut243") == 0)
        return &vtm_backend_scalar_lut;
    if (strcmp(name, "direct") == 0 || strcmp(name, "scalar-direct") == 0)
        return &vtm_backend_scalar_direct;
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
                    "(auto|scalar|simd|direct), беру scalar\n", env);
    return &vtm_backend_scalar_lut;
}
