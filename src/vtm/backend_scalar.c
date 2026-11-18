// Скалярные ядра VTM. Два варианта TMV:
//   * LUT243  — слово pack20 делится на 4 группы по 5 тритов, группа
//               декодируется таблицей на 243 записи (1,2 КБ, живёт в L1);
//   * direct  — триты достаются делением слова на 3.
// Оба варианта считают одно и то же; разница только в скорости. Замер —
// `./vtm bench` и таблица в src/vtm/README.md.
#include "vtm.h"

#include <math.h>
#include <string.h>

// --- LUT на 5 тритов -------------------------------------------------------

static int8_t  LUT243[243][5];
static int     lut_ready;

static void lut_init(void) {
    unsigned v;
    if (lut_ready) return;
    for (v = 0; v < 243; v++) {
        unsigned t = v;
        int k;
        for (k = 0; k < 5; k++) {
            unsigned d = t % 3u;
            t /= 3u;
            LUT243[v][k] = (int8_t)(d == 2u ? 1 : (d == 0u ? -1 : 0));
        }
    }
    lut_ready = 1;
}

// --- TMV -------------------------------------------------------------------

static int tmv_check(const float *y, const uint8_t *w, const float *scales,
                     const float *x, int rows, int cols) {
    if (!y || !w || !scales || !x) return -1;
    if (rows < 0 || cols < 0) return -1;
    return 0;
}

// Прямая распаковка: 20 делений на 3 на слово, зато без таблицы.
// scale_block — 128 (канон Python) или 160 (docs/trit-format). Если блок
// кратен 20, все 20 тритов слова лежат в одном блоке масштаба; иначе
// (128) граница может проходить внутри слова, и масштаб применяется к
// частичной сумме блока поэлементно.
static int tmv_direct_blocks(float *y, const uint8_t *w, const float *scales,
                             const float *x, int rows, int cols,
                             int scale_block) {
    const uint32_t *words = (const uint32_t *)w;
    int r;
    if (tmv_check(y, w, scales, x, rows, cols) != 0) return -1;
    if (scale_block != 128 && scale_block != 160) return -1;
    if (rows == 0 || cols == 0) return 0;

    for (r = 0; r < rows; r++) {
        int64_t base = (int64_t)r * (int64_t)cols;
        int64_t end = base + (int64_t)cols;
        int64_t wi = base / (int64_t)VTM_TRITS_PER_WORD;
        int64_t wi_end = (end + VTM_TRITS_PER_WORD - 1) / VTM_TRITS_PER_WORD;
        double acc = 0.0;

        if (scale_block % VTM_TRITS_PER_WORD == 0) {
            for (; wi < wi_end; wi++) {
                uint32_t v = words[wi];
                int64_t wbase = wi * (int64_t)VTM_TRITS_PER_WORD;
                double block_acc = 0.0;
                int k;
                for (k = 0; k < VTM_TRITS_PER_WORD; k++) {
                    unsigned d = v % 3u;
                    int64_t idx = wbase + k;
                    v /= 3u;
                    if (idx < base) continue;
                    if (idx >= end) break;
                    if (d == 2u)      block_acc += (double)x[idx - base];
                    else if (d == 0u) block_acc -= (double)x[idx - base];
                }
                acc += block_acc
                     * (double)scales[(size_t)((wi * VTM_TRITS_PER_WORD)
                                               / scale_block)];
            }
        } else {
            int64_t cur_block = -1;
            double block_acc = 0.0;
            for (; wi < wi_end; wi++) {
                uint32_t v = words[wi];
                int64_t wbase = wi * (int64_t)VTM_TRITS_PER_WORD;
                int k;
                for (k = 0; k < VTM_TRITS_PER_WORD; k++) {
                    unsigned d = v % 3u;
                    int64_t idx = wbase + k;
                    int64_t blk;
                    v /= 3u;
                    if (idx < base) continue;
                    if (idx >= end) break;
                    blk = idx / (int64_t)scale_block;
                    if (blk != cur_block) {
                        if (cur_block >= 0)
                            acc += block_acc * (double)scales[cur_block];
                        cur_block = blk;
                        block_acc = 0.0;
                    }
                    if (d == 2u)      block_acc += (double)x[idx - base];
                    else if (d == 0u) block_acc -= (double)x[idx - base];
                }
            }
            if (cur_block >= 0)
                acc += block_acc * (double)scales[cur_block];
        }
        y[r] = (float)acc;
    }
    return 0;
}

static int tmv_direct(float *y, const uint8_t *w, const float *scales,
                      const float *x, int rows, int cols) {
    return tmv_direct_blocks(y, w, scales, x, rows, cols,
                             (int)VTM_SCALE_BLOCK_160);
}

// LUT243: 4 выборки из таблицы на слово вместо 20 делений.
static int tmv_lut_blocks(float *y, const uint8_t *w, const float *scales,
                          const float *x, int rows, int cols,
                          int scale_block) {
    const uint32_t *words = (const uint32_t *)w;
    int r;
    if (tmv_check(y, w, scales, x, rows, cols) != 0) return -1;
    if (scale_block != 128 && scale_block != 160) return -1;
    if (rows == 0 || cols == 0) return 0;
    lut_init();

    for (r = 0; r < rows; r++) {
        int64_t base = (int64_t)r * (int64_t)cols;
        int64_t end = base + (int64_t)cols;
        int64_t wi = base / (int64_t)VTM_TRITS_PER_WORD;
        int64_t wi_end = (end + VTM_TRITS_PER_WORD - 1) / VTM_TRITS_PER_WORD;
        double acc = 0.0;

        if (scale_block % VTM_TRITS_PER_WORD == 0) {
            for (; wi < wi_end; wi++) {
                uint32_t v = words[wi];
                int64_t wbase = wi * (int64_t)VTM_TRITS_PER_WORD;
                double block_acc = 0.0;
                int g;
                double scale = (double)scales[(size_t)((wi * VTM_TRITS_PER_WORD)
                                                      / scale_block)];
                for (g = 0; g < 4; g++) {
                    unsigned code = v % 243u;
                    const int8_t *trit = LUT243[code];
                    int64_t idx0 = wbase + (int64_t)g * 5;
                    int k;
                    v /= 243u;
                    for (k = 0; k < 5; k++) {
                        int64_t idx = idx0 + k;
                        if (idx < base) continue;
                        if (idx >= end) break;
                        block_acc += (double)trit[k] * (double)x[idx - base];
                    }
                }
                acc += block_acc * scale;
            }
        } else {
            int64_t cur_block = -1;
            double block_acc = 0.0;
            for (; wi < wi_end; wi++) {
                uint32_t v = words[wi];
                int64_t wbase = wi * (int64_t)VTM_TRITS_PER_WORD;
                int g;
                for (g = 0; g < 4; g++) {
                    unsigned code = v % 243u;
                    const int8_t *trit = LUT243[code];
                    int64_t idx0 = wbase + (int64_t)g * 5;
                    int k;
                    v /= 243u;
                    for (k = 0; k < 5; k++) {
                        int64_t idx = idx0 + k;
                        int64_t blk;
                        if (idx < base) continue;
                        if (idx >= end) break;
                        blk = idx / (int64_t)scale_block;
                        if (blk != cur_block) {
                            if (cur_block >= 0)
                                acc += block_acc * (double)scales[cur_block];
                            cur_block = blk;
                            block_acc = 0.0;
                        }
                        block_acc += (double)trit[k] * (double)x[idx - base];
                    }
                }
            }
            if (cur_block >= 0)
                acc += block_acc * (double)scales[cur_block];
        }
        y[r] = (float)acc;
    }
    return 0;
}

static int tmv_lut(float *y, const uint8_t *w, const float *scales,
                   const float *x, int rows, int cols) {
    return tmv_lut_blocks(y, w, scales, x, rows, cols,
                          (int)VTM_SCALE_BLOCK_160);
}

// --- остальные ядра --------------------------------------------------------

static int rmsnorm_scalar(float *y, const float *x, const float *w,
                          int n, float eps) {
    double sum = 0.0, mean, rms;
    int i;
    if (!y || !x || !w || n < 0) return -1;
    if (n == 0) return 0;
    for (i = 0; i < n; i++) sum += (double)x[i] * (double)x[i];
    mean = sum / (double)n;
    rms = sqrt(mean + (double)eps);
    if (rms == 0.0) {
        // x == 0 и eps == 0: предел 0/0 доопределяем нулём, чтобы не тащить NaN.
        memset(y, 0, (size_t)n * sizeof(float));
        return 0;
    }
    for (i = 0; i < n; i++)
        y[i] = (float)((double)x[i] / rms * (double)w[i]);
    return 0;
}

static int silu_scalar(float *y, const float *x, int n) {
    int i;
    if (!y || !x || n < 0) return -1;
    for (i = 0; i < n; i++) {
        double v = (double)x[i];
        if (v < -700.0) y[i] = 0.0f; // как в Python: без переполнения exp
        else y[i] = (float)(v / (1.0 + exp(-v)));
    }
    return 0;
}

static int mul_scalar(float *y, const float *a, const float *b, int n) {
    int i;
    if (!y || !a || !b || n < 0) return -1;
    for (i = 0; i < n; i++) y[i] = a[i] * b[i];
    return 0;
}

static int add_scalar(float *y, const float *a, const float *b, int n) {
    int i;
    if (!y || !a || !b || n < 0) return -1;
    for (i = 0; i < n; i++) y[i] = a[i] + b[i];
    return 0;
}

static int scal_scalar(float *y, const float *x, float k, int n) {
    int i;
    if (!y || !x || n < 0) return -1;
    for (i = 0; i < n; i++) y[i] = x[i] * k;
    return 0;
}

static int copy_scalar(float *y, const float *x, int n) {
    if (!y || !x || n < 0) return -1;
    if (n == 0) return 0;
    memmove(y, x, (size_t)n * sizeof(float));
    return 0;
}

static int emb_scalar(float *y, const float *e, int dim, int token) {
    if (!y || !e || dim <= 0 || token < 0) return -1;
    memmove(y, e + (size_t)token * (size_t)dim, (size_t)dim * sizeof(float));
    return 0;
}

// offset — сколько позиций уже прошло: маска оставляет j <= r + offset.
static int softmax_scalar(float *y, const float *x, int rows, int cols,
                          int causal, int offset) {
    int r, j;
    if (!y || !x || rows < 0 || cols < 0) return -1;
    if (rows == 0 || cols == 0) return 0;

    for (r = 0; r < rows; r++) {
        const float *xr = x + (size_t)r * (size_t)cols;
        float *yr = y + (size_t)r * (size_t)cols;
        long long limit = (long long)r + (long long)offset;
        double max = -INFINITY;
        double sum = 0.0;
        int any = 0;

        for (j = 0; j < cols; j++) {
            double v;
            if (causal && (long long)j > limit) continue;
            v = (double)xr[j];
            if (isnan(v)) continue;
            any = 1;
            if (v > max) max = v;
        }
        if (!any) {
            for (j = 0; j < cols; j++) yr[j] = 0.0f;
            continue;
        }
        for (j = 0; j < cols; j++) {
            double v;
            if (causal && (long long)j > limit) continue;
            v = (double)xr[j];
            if (isnan(v)) continue;
            sum += exp(v - max);
        }
        if (!(sum > 0.0) || !isfinite(sum)) {
            for (j = 0; j < cols; j++) yr[j] = 0.0f;
            continue;
        }
        for (j = 0; j < cols; j++) {
            double v;
            if (causal && (long long)j > limit) {
                yr[j] = 0.0f;
                continue;
            }
            v = (double)xr[j];
            if (isnan(v)) {
                yr[j] = 0.0f;
                continue;
            }
            yr[j] = (float)(exp(v - max) / sum);
        }
    }
    return 0;
}

// --- публичные описатели бэкендов -----------------------------------------

const vtm_backend vtm_backend_scalar_lut = {
    .name = "scalar-lut243",
    .tmv = tmv_lut,
    .tmv_blocks = tmv_lut_blocks,
    .rmsnorm = rmsnorm_scalar,
    .silu = silu_scalar,
    .mul = mul_scalar,
    .add = add_scalar,
    .scal = scal_scalar,
    .softmax = softmax_scalar,
    .emb = emb_scalar,
    .copy = copy_scalar
};

const vtm_backend vtm_backend_scalar_direct = {
    .name = "scalar-direct",
    .tmv = tmv_direct,
    .tmv_blocks = tmv_direct_blocks,
    .rmsnorm = rmsnorm_scalar,
    .silu = silu_scalar,
    .mul = mul_scalar,
    .add = add_scalar,
    .scal = scal_scalar,
    .softmax = softmax_scalar,
    .emb = emb_scalar,
    .copy = copy_scalar
};
