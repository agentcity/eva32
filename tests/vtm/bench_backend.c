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
//                                     max|diff| и argmax-токен;
//   ./bench_backend --batch         — батчевые TMV/MM (0122) на формах
//                                     1.5B: «по токену» против пачки и
//                                     оценка prefill 32/128.
//   ./bench_backend --q4            — Q4 (kind 5, 0152): скорость и RSS
//                                     против fp32-копии, форма 151936x896
//                                     (lm_head Qwen-0.5B); --q4-rows/cols/iters
//                                     меняют форму и число итераций.
//
// Бэкенд «кандидат» берётся из --backend или VTM_BACKEND (vtm_backend_from_env),
// по умолчанию — simd, если он доступен.
#include "vtm.h"
#include "vtm_backend.h"
// Только тесту: минимальная машина без загрузчика, чтобы проверить проводку
// kind 5 в interp.c (op_tmv/op_mm/op_emb) без сборки контейнера.
#include "vtm_internal.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef __APPLE__
#include <mach/mach.h>
#endif
#if defined(__linux__) || defined(__APPLE__)
#include <unistd.h>
#endif

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

// --- батчевые эталоны TMV/MM (задача 0122) ---------------------------------
//
// Повторяют батчевые ядра interp.c (batch_tmv_trit_rows/batch_mm_trit_rows),
// но независимо от них: сравнение «SIMD-батч == скалярный батч» не должно
// быть сравнением кода с самим собой. Раскладка та же: x[t*N + j], y[r*N + j].

// Распаковать строку тритов base..base+cols-1 в row[0..cols).
static void unpack_row(const uint32_t *words, int64_t base, int cols,
                       int8_t *row) {
    int64_t wi = base / VTM_TRITS_PER_WORD;
    int64_t wi_end = (base + (int64_t)cols - 1) / VTM_TRITS_PER_WORD;
    for (; wi <= wi_end; wi++) {
        int8_t trits[VTM_TRITS_PER_WORD];
        int k;
        vtm_trit_unpack20(words[wi], trits);
        for (k = 0; k < VTM_TRITS_PER_WORD; k++) {
            int64_t idx = wi * VTM_TRITS_PER_WORD + k - base;
            if (idx >= 0 && idx < (int64_t)cols) row[idx] = trits[k];
        }
    }
}

// TMV-эталон: масштаб применяется к сумме блока (double), как в scalar TMV.
static void batch_tmv_ref(float *y, const uint32_t *words, const float *scales,
                          const float *x, int rows, int cols, int ntok,
                          int scale_block) {
    int8_t *row = (int8_t *)malloc((size_t)(cols > 0 ? cols : 1));
    int r, j;
    for (r = 0; r < rows; r++) {
        int64_t base = (int64_t)r * (int64_t)cols;
        int t = 0;
        unpack_row(words, base, cols, row);
        for (j = 0; j < ntok; j++) {
            double acc = 0.0;
            t = 0;
            while (t < cols) {
                int64_t blk = (base + t) / (int64_t)scale_block;
                int64_t e = (blk + 1) * (int64_t)scale_block - base;
                double block_acc = 0.0;
                if (e > (int64_t)cols) e = (int64_t)cols;
                for (; t < (int)e; t++)
                    block_acc += (double)row[t]
                               * (double)x[(size_t)t * (size_t)ntok
                                           + (size_t)j];
                acc += block_acc * (double)scales[(size_t)blk];
            }
            y[(size_t)r * (size_t)ntok + (size_t)j] = (float)acc;
        }
    }
    free(row);
}

// MM-эталон: масштаб у каждого элемента (trit_weight_at), double по t — как в
// batch_mm_trit_rows.
static void batch_mm_ref(float *y, const uint32_t *words, const float *scales,
                         const float *x, int mm, int kk, int ntok,
                         int scale_block) {
    int8_t *row = (int8_t *)malloc((size_t)(kk > 0 ? kk : 1));
    int r, j, t;
    for (r = 0; r < mm; r++) {
        int64_t base = (int64_t)r * (int64_t)kk;
        unpack_row(words, base, kk, row);
        for (j = 0; j < ntok; j++) {
            double acc = 0.0;
            for (t = 0; t < kk; t++) {
                float wv = (float)row[t]
                         * scales[(size_t)((base + t)
                                           / (int64_t)scale_block)];
                acc += (double)wv
                     * (double)x[(size_t)t * (size_t)ntok + (size_t)j];
            }
            y[(size_t)r * (size_t)ntok + (size_t)j] = (float)acc;
        }
    }
    free(row);
}

static void batch_f32_ref(float *y, const float *w, const float *x, int rows,
                          int cols, int ntok, int transposed) {
    int r, t, j;
    for (r = 0; r < rows; r++)
        for (j = 0; j < ntok; j++) {
            double acc = 0.0;
            for (t = 0; t < cols; t++) {
                double wv = transposed
                          ? (double)w[(size_t)t * (size_t)rows + (size_t)r]
                          : (double)w[(size_t)r * (size_t)cols + (size_t)t];
                acc += wv * (double)x[(size_t)t * (size_t)ntok + (size_t)j];
            }
            y[(size_t)r * (size_t)ntok + (size_t)j] = (float)acc;
        }
}

// --- Q4 (kind 5, задача 0152) ----------------------------------------------
//
// Упаковка — как в GGML Q4_0: 32 значения в 16 байтах, байт j хранит
// value[j] (младший ниббл) и value[j+16] (старший). Эталон ниже —
// независимая распаковка в double, без вызова ядра: иначе сравнение было бы
// сравнением кода с самим собой.
static uint8_t *make_q4(int n, uint32_t seed) {
    size_t groups = ((size_t)n + VTM_Q4_GROUP_SIZE - 1)
                  / VTM_Q4_GROUP_SIZE;
    uint8_t *q = (uint8_t *)calloc(groups * (VTM_Q4_GROUP_SIZE / 2), 1u);
    size_t g;

    if (!q) return NULL;
    for (g = 0; g < groups; g++) {
        int j;
        for (j = 0; j < VTM_Q4_GROUP_SIZE / 2; j++) {
            unsigned lo = rng_next(&seed) & 0x0fu;
            unsigned hi = rng_next(&seed) & 0x0fu;
            q[g * (VTM_Q4_GROUP_SIZE / 2) + (size_t)j]
                = (uint8_t)(lo | (hi << 4));
        }
    }
    return q;
}

// scale > 0 — все масштабы равны scale; scale == 0 — 0.005..0.035.
static float *make_q4_scales(int n, float scale, uint32_t seed) {
    size_t groups = ((size_t)n + VTM_Q4_GROUP_SIZE - 1)
                  / VTM_Q4_GROUP_SIZE;
    float *s = (float *)malloc((groups ? groups : 1u) * sizeof(float));
    size_t g;

    if (!s) return NULL;
    for (g = 0; g < groups; g++)
        s[g] = scale > 0.0f
             ? scale
             : 0.005f + 0.03f * (float)(rng_next(&seed) % 1000u) / 999.0f;
    return s;
}

// Q4_1: min на группу, -0.05..0.05.
static float *make_q4_mins(int n, uint32_t seed) {
    size_t groups = ((size_t)n + VTM_Q4_GROUP_SIZE - 1)
                  / VTM_Q4_GROUP_SIZE;
    float *m = (float *)malloc((groups ? groups : 1u) * sizeof(float));
    size_t g;

    if (!m) return NULL;
    for (g = 0; g < groups; g++)
        m[g] = -0.05f + 0.1f * (float)(rng_next(&seed) % 1000u) / 999.0f;
    return m;
}

static double q4_ref_value(const uint8_t *q, int64_t idx, const float *scales,
                           const float *mins, int asym) {
    int64_t g = idx / VTM_Q4_GROUP_SIZE;
    unsigned within = (unsigned)(idx % VTM_Q4_GROUP_SIZE);
    unsigned byte = q[g * (VTM_Q4_GROUP_SIZE / 2) + (within & 15u)];
    unsigned v = (within < 16u) ? (byte & 0x0fu) : (byte >> 4);
    double scale = (double)scales[g];
    return asym ? (double)v * scale + (double)mins[g]
                : ((double)v - 8.0) * scale;
}

static void q4_tmv_ref(float *y, const uint8_t *q, const float *scales,
                       const float *mins, const float *x, int rows, int cols,
                       int asym) {
    int r, j;
    for (r = 0; r < rows; r++) {
        int64_t base = (int64_t)r * (int64_t)cols;
        double acc = 0.0;
        for (j = 0; j < cols; j++)
            acc += q4_ref_value(q, base + (int64_t)j, scales, mins, asym)
                 * (double)x[j];
        y[r] = (float)acc;
    }
}

static void q4_mm_ref(float *y, const uint8_t *q, const float *scales,
                      const float *mins, const float *b, int m, int k, int n,
                      int asym) {
    int r, c, t;
    for (r = 0; r < m; r++) {
        int64_t base = (int64_t)r * (int64_t)k;
        for (c = 0; c < n; c++) {
            double acc = 0.0;
            for (t = 0; t < k; t++)
                acc += q4_ref_value(q, base + (int64_t)t, scales, mins, asym)
                     * (double)b[(size_t)t * (size_t)n + (size_t)c];
            y[(size_t)r * (size_t)n + (size_t)c] = (float)acc;
        }
    }
}

// Резидентная память процесса: нужна, чтобы показать, что Q4-ядро не тянет
// fp32-копию весов. macOS — task_info, Linux — /proc/self/statm; иначе 0.
static size_t rss_bytes(void) {
#if defined(__APPLE__)
    // phys_footprint, а не resident_size: последний не считает сжатые
    // страницы, и на больших матрицах RSS «прыгал» вниз после ядра.
    task_vm_info_data_t info;
    mach_msg_type_number_t cnt = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO,
                  (task_info_t)&info, &cnt) == KERN_SUCCESS)
        return (size_t)info.phys_footprint;
#elif defined(__linux__)
    FILE *f = fopen("/proc/self/statm", "r");
    long total = 0, resident = 0;
    if (f) {
        if (fscanf(f, "%ld %ld", &total, &resident) == 2) {
            long pg = sysconf(_SC_PAGESIZE);
            fclose(f);
            if (pg > 0) return (size_t)resident * (size_t)pg;
            return 0;
        }
        fclose(f);
    }
#endif
    return 0;
}

static double mb_of(size_t bytes) { return (double)bytes / (1024.0 * 1024.0); }

// Плотный fp32 TMV для сравнения скорости: та же арифметика (double по
// строке), то же разбиение по строкам, что у Q4-ядра.
typedef struct {
    float *y;
    const float *w;
    const float *x;
    int cols;
} f32_tmv_job;

static void f32_tmv_rows(void *p, int tid, int r0, int r1) {
    f32_tmv_job *j = (f32_tmv_job *)p;
    int r, t;
    (void)tid;
    for (r = r0; r < r1; r++) {
        double acc = 0.0;
        const float *wr = j->w + (size_t)r * (size_t)j->cols;
        for (t = 0; t < j->cols; t++)
            acc += (double)wr[t] * (double)j->x[t];
        j->y[r] = (float)acc;
    }
}

static void f32_tmv(float *y, const float *w, const float *x, int rows,
                    int cols) {
    f32_tmv_job job;
    job.y = y;
    job.w = w;
    job.x = x;
    job.cols = cols;
    vtm_parallel_rows(rows, (double)rows * (double)cols, f32_tmv_rows, &job);
}

// Минимальная машина VTM без loader.c: три тензора, TMV/MM + HALT. Нужна,
// чтобы проверить именно проводку kind 5 в interp.c (op_tmv/op_mm): поля
// q4_is_asym/q4_mins и ошибку формата, не дожидаясь загрузчика 0151.
static vtm_status q4_interp_tmv(const vtm_tensor *W, const float *x, float *y,
                                int rows, int cols, char *err, size_t errcap) {
    vtm_t m;
    vtm_tensor ts[3];
    vtm_instr ins[2];
    vtm_status st;

    memset(&m, 0, sizeof m);
    memset(ts, 0, sizeof ts);
    memset(ins, 0, sizeof ins);
    ts[0] = *W;
    ts[1].kind = VTM_TENSOR_F32;
    ts[1].ndim = 2;
    ts[1].dims[0] = 1;
    ts[1].dims[1] = (uint32_t)cols;
    ts[1].n_elements = (uint32_t)cols;
    ts[1].f32 = (float *)x;
    ts[2].kind = VTM_TENSOR_F32;
    ts[2].ndim = 2;
    ts[2].dims[0] = 1;
    ts[2].dims[1] = (uint32_t)rows;
    ts[2].n_elements = (uint32_t)rows;
    ts[2].f32 = y;
    ins[0].opcode = VTM_OP_TMV;
    ins[0].a = 0;
    ins[0].b = 1;
    ins[0].dst = 2;
    ins[1].opcode = VTM_OP_HALT;
    m.tensors = ts;
    m.n_tensors = 3;
    m.instr = ins;
    m.n_instr = 2;
    m.backend = &vtm_backend_scalar_lut;
    st = vtm_run(&m, NULL, NULL);
    if (err && errcap) snprintf(err, errcap, "%s", vtm_error_message(&m));
    return st;
}

static vtm_status q4_interp_mm(const vtm_tensor *A, const float *b, float *y,
                               int rows, int k, int n, char *err,
                               size_t errcap) {
    vtm_t m;
    vtm_tensor ts[3];
    vtm_instr ins[2];
    vtm_status st;

    memset(&m, 0, sizeof m);
    memset(ts, 0, sizeof ts);
    memset(ins, 0, sizeof ins);
    ts[0] = *A;
    ts[1].kind = VTM_TENSOR_F32;
    ts[1].ndim = 2;
    ts[1].dims[0] = (uint32_t)k;
    ts[1].dims[1] = (uint32_t)n;
    ts[1].n_elements = (uint32_t)(k * n);
    ts[1].f32 = (float *)b;
    ts[2].kind = VTM_TENSOR_F32;
    ts[2].ndim = 2;
    ts[2].dims[0] = (uint32_t)rows;
    ts[2].dims[1] = (uint32_t)n;
    ts[2].n_elements = (uint32_t)(rows * n);
    ts[2].f32 = y;
    ins[0].opcode = VTM_OP_MM;
    ins[0].a = 0;
    ins[0].b = 1;
    ins[0].dst = 2;
    ins[1].opcode = VTM_OP_HALT;
    m.tensors = ts;
    m.n_tensors = 3;
    m.instr = ins;
    m.n_instr = 2;
    m.backend = &vtm_backend_scalar_lut;
    st = vtm_run(&m, NULL, NULL);
    if (err && errcap) snprintf(err, errcap, "%s", vtm_error_message(&m));
    return st;
}

static vtm_status q4_interp_emb(const vtm_tensor *E, float *y, int token,
                                int dim, char *err, size_t errcap) {
    vtm_t m;
    vtm_tensor ts[2];
    vtm_instr ins[2];
    vtm_status st;

    memset(&m, 0, sizeof m);
    memset(ts, 0, sizeof ts);
    memset(ins, 0, sizeof ins);
    ts[0] = *E;
    ts[1].kind = VTM_TENSOR_F32;
    ts[1].ndim = 1;
    ts[1].dims[0] = (uint32_t)dim;
    ts[1].n_elements = (uint32_t)dim;
    ts[1].f32 = y;
    ins[0].opcode = VTM_OP_EMB;
    ins[0].a = 0;
    ins[0].dst = 1;
    ins[0].imm = token;
    ins[1].opcode = VTM_OP_HALT;
    m.tensors = ts;
    m.n_tensors = 2;
    m.instr = ins;
    m.n_instr = 2;
    m.backend = &vtm_backend_scalar_lut;
    st = vtm_run(&m, NULL, NULL);
    if (err && errcap) snprintf(err, errcap, "%s", vtm_error_message(&m));
    return st;
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
// Q4 TMV: ядро против независимой деквантизации в double. asym=0 — Q4_0,
// asym=1 — Q4_1 (q*scale+min). Формы включают cols не кратные 32, чтобы
// проверить группу, пересекающую границу строк.
static void check_q4_tmv_case(int rows, int cols, int asym, float scale) {
    int n = rows * cols;
    uint32_t seed = 0x4b00u + (uint32_t)rows * 131u + (uint32_t)cols
                  + (uint32_t)asym * 7u;
    uint8_t *q = make_q4(n, seed);
    float *scales = make_q4_scales(n, scale, seed ^ 0x91u);
    float *mins = asym ? make_q4_mins(n, seed ^ 0x33u) : NULL;
    float *x = make_vec((size_t)cols, seed ^ 0x51u);
    float *y_ref = (float *)calloc((size_t)rows, sizeof(float));
    float *y_q4 = (float *)calloc((size_t)rows, sizeof(float));
    char label[96];
    double ref = 0.0;

    if (!q || !scales || (asym && !mins) || !x || !y_ref || !y_q4) {
        fprintf(stderr, "check_q4_tmv_case: нет памяти\n");
        failures++;
        goto done;
    }
    q4_tmv_ref(y_ref, q, scales, mins, x, rows, cols, asym);
    if (vtm_q4_tmv(y_q4, q, scales, mins, x, rows, cols, asym) != 0) {
        fprintf(stderr, "check_q4_tmv_case: ядро вернуло ошибку\n");
        failures++;
        goto done;
    }
    snprintf(label, sizeof label, "Q4%s TMV %dx%d scale %.2f",
             asym ? "1" : "0", rows, cols, (double)scale);
    report(label, max_diff(y_q4, y_ref, (size_t)rows, &ref), ref, CHECK_TOL);
done:
    free(q);
    free(scales);
    free(mins);
    free(x);
    free(y_ref);
    free(y_q4);
}

static void check_q4_mm_case(int m, int k, int n, int asym, float scale) {
    int n_el = m * k;
    uint32_t seed = 0x9d00u + (uint32_t)m * 17u + (uint32_t)k * 3u
                  + (uint32_t)n + (uint32_t)asym * 11u;
    uint8_t *q = make_q4(n_el, seed);
    float *scales = make_q4_scales(n_el, scale, seed ^ 0x77u);
    float *mins = asym ? make_q4_mins(n_el, seed ^ 0x22u) : NULL;
    float *b = make_vec((size_t)k * (size_t)n, seed ^ 0x11u);
    float *y_ref = (float *)calloc((size_t)m * (size_t)n, sizeof(float));
    float *y_q4 = (float *)calloc((size_t)m * (size_t)n, sizeof(float));
    char label[112];
    double ref = 0.0;

    if (!q || !scales || (asym && !mins) || !b || !y_ref || !y_q4) {
        fprintf(stderr, "check_q4_mm_case: нет памяти\n");
        failures++;
        goto done;
    }
    q4_mm_ref(y_ref, q, scales, mins, b, m, k, n, asym);
    if (vtm_q4_mm(y_q4, q, scales, mins, b, m, k, n, asym) != 0) {
        fprintf(stderr, "check_q4_mm_case: ядро вернуло ошибку\n");
        failures++;
        goto done;
    }
    snprintf(label, sizeof label, "Q4%s MM %dx%dx%d scale %.2f",
             asym ? "1" : "0", m, k, n, (double)scale);
    report(label, max_diff(y_q4, y_ref, (size_t)m * (size_t)n, &ref), ref,
           CHECK_TOL);
done:
    free(q);
    free(scales);
    free(mins);
    free(b);
    free(y_ref);
    free(y_q4);
}

// Контракт аргументов: NULL/отрицательные размеры — -1; asym без mins — -1;
// нулевая размерность — 0 и никакого обращения к буферам.
static void check_q4_args(void) {
    float x[2] = {1.0f, 2.0f};
    float y[2] = {0.0f, 0.0f};
    float scales[1] = {1.0f};
    float mins[1] = {0.0f};
    uint8_t q[16] = {0};
    int bad = 0;

    if (vtm_q4_tmv(NULL, q, scales, NULL, x, 1, 2, 0) != -1) bad = 1;
    if (vtm_q4_tmv(y, NULL, scales, NULL, x, 1, 2, 0) != -1) bad = 1;
    if (vtm_q4_tmv(y, q, NULL, NULL, x, 1, 2, 0) != -1) bad = 1;
    if (vtm_q4_tmv(y, q, scales, NULL, NULL, 1, 2, 0) != -1) bad = 1;
    if (vtm_q4_tmv(y, q, scales, NULL, x, -1, 2, 0) != -1) bad = 1;
    if (vtm_q4_tmv(y, q, scales, NULL, x, 1, 2, 1) != -1) bad = 1;
    if (vtm_q4_tmv(y, q, scales, mins, x, 0, 2, 1) != 0) bad = 1;
    if (vtm_q4_mm(NULL, q, scales, NULL, x, 1, 2, 1, 0) != -1) bad = 1;
    if (vtm_q4_mm(y, q, scales, NULL, x, 1, 2, 1, 1) != -1) bad = 1;
    if (vtm_q4_mm(y, q, scales, NULL, x, 1, 2, 0, 0) != 0) bad = 1;
    printf("  %-34s %s\n", "Q4 аргументы/NULL/размеры",
           bad ? "FAIL" : "ok");
    if (bad) failures++;
}

// Проводка kind 5 через interp.c: sym и asym на одних и тех же q/scales дают
// разные результаты (asym учитывает mins), а asym без mins отвергается
// VTM_E_FORMAT, а не считается молча симметрично.
static void check_q4_interp(void) {
    enum { ROWS = 13, COLS = 97, K = 37, MM_ROWS = 7, MM_K = 53 };
    int n = ROWS * COLS;
    int groups = (n + VTM_Q4_GROUP_SIZE - 1) / VTM_Q4_GROUP_SIZE;
    uint8_t *q = make_q4(n, 0x5150u);
    float *scales = make_q4_scales(n, 0.02f, 0x77u);
    float *mins = make_q4_mins(n, 0x33u);
    float *x = make_vec((size_t)COLS, 0x11u);
    float *y_sym = (float *)calloc((size_t)ROWS, sizeof(float));
    float *y_asym = (float *)calloc((size_t)ROWS, sizeof(float));
    float *y_ref = (float *)calloc((size_t)ROWS, sizeof(float));
    float *y_nomins = (float *)calloc((size_t)ROWS, sizeof(float));
    vtm_tensor W;
    char err[256];
    vtm_status st;
    double d, ref = 0.0;

    if (!q || !scales || !mins || !x || !y_sym || !y_asym || !y_ref
        || !y_nomins) {
        fprintf(stderr, "check_q4_interp: нет памяти\n");
        failures++;
        goto done;
    }
    memset(&W, 0, sizeof W);
    W.kind = VTM_TENSOR_Q4;
    W.ndim = 2;
    W.dims[0] = ROWS;
    W.dims[1] = COLS;
    W.n_elements = (uint32_t)n;
    W.words = (const uint32_t *)q;
    W.scales = scales;
    W.n_scales = (uint32_t)groups;
    W.scale_block = VTM_Q4_GROUP_SIZE;
    W.scale_is_f32 = 1;
    W.q4_is_asym = 0;

    err[0] = '\0';
    st = q4_interp_tmv(&W, x, y_sym, ROWS, COLS, err, sizeof err);
    q4_tmv_ref(y_ref, q, scales, NULL, x, ROWS, COLS, 0);
    report("Q4 sym через interp (TMV)",
           max_diff(y_sym, y_ref, (size_t)ROWS, &ref), ref,
           st == VTM_OK ? CHECK_TOL : -1.0);

    W.q4_is_asym = 1;
    W.q4_mins = mins;
    err[0] = '\0';
    st = q4_interp_tmv(&W, x, y_asym, ROWS, COLS, err, sizeof err);
    q4_tmv_ref(y_ref, q, scales, mins, x, ROWS, COLS, 1);
    report("Q4 asym через interp (TMV)",
           max_diff(y_asym, y_ref, (size_t)ROWS, &ref), ref,
           st == VTM_OK ? CHECK_TOL : -1.0);

    d = max_diff(y_sym, y_asym, (size_t)ROWS, NULL);
    printf("  %-34s max|sym-asym|=%.3e  %s\n", "Q4 sym != asym (одни qs)",
           d, d > 1e-4 ? "ok" : "FAIL");
    if (!(d > 1e-4)) failures++;

    // asym без минимумов — ошибка формата с внятным текстом.
    W.q4_mins = NULL;
    err[0] = '\0';
    st = q4_interp_tmv(&W, x, y_nomins, ROWS, COLS, err, sizeof err);
    printf("  %-34s st=%d (%s)  %s\n", "Q4_1 без mins отвергнут", (int)st,
           err, st == VTM_E_FORMAT && strstr(err, "Q4_1") ? "ok" : "FAIL");
    if (st != VTM_E_FORMAT || !strstr(err, "Q4_1")) failures++;

    // MM: та же проводка, n == 1 (tied lm_head).
    {
        int mn = MM_ROWS * MM_K;
        int mgroups = (mn + VTM_Q4_GROUP_SIZE - 1) / VTM_Q4_GROUP_SIZE;
        uint8_t *mq = make_q4(mn, 0x9001u);
        float *mscales = make_q4_scales(mn, 0.02f, 0x9002u);
        float *bm = make_vec((size_t)MM_K, 0x9003u);
        float *ym = (float *)calloc((size_t)MM_ROWS, sizeof(float));
        float *yref = (float *)calloc((size_t)MM_ROWS, sizeof(float));
        vtm_tensor A;
        if (!mq || !mscales || !bm || !ym || !yref) {
            fprintf(stderr, "check_q4_interp: нет памяти на MM\n");
            failures++;
        } else {
            memset(&A, 0, sizeof A);
            A.kind = VTM_TENSOR_Q4;
            A.ndim = 2;
            A.dims[0] = MM_ROWS;
            A.dims[1] = MM_K;
            A.n_elements = (uint32_t)mn;
            A.words = (const uint32_t *)mq;
            A.scales = mscales;
            A.n_scales = (uint32_t)mgroups;
            A.scale_block = VTM_Q4_GROUP_SIZE;
            st = q4_interp_mm(&A, bm, ym, MM_ROWS, MM_K, 1, err, sizeof err);
            q4_mm_ref(yref, mq, mscales, NULL, bm, MM_ROWS, MM_K, 1, 0);
            report("Q4 sym через interp (MM n=1)",
                   max_diff(ym, yref, (size_t)MM_ROWS, &ref), ref,
                   st == VTM_OK ? CHECK_TOL : -1.0);
        }
        free(mq);
        free(mscales);
        free(bm);
        free(ym);
        free(yref);
    }

    // EMB: Q4-таблица словаря — разворачивается только выбранная строка.
    {
        int vocab = 11, dim = 37, en = vocab * dim, etok = vocab - 1;
        int egroups = (en + VTM_Q4_GROUP_SIZE - 1) / VTM_Q4_GROUP_SIZE;
        uint8_t *eq = make_q4(en, 0xa001u);
        float *escales = make_q4_scales(en, 0.02f, 0xa002u);
        float *ey = (float *)calloc((size_t)dim, sizeof(float));
        float *eref = (float *)calloc((size_t)dim, sizeof(float));
        vtm_tensor E;
        if (!eq || !escales || !ey || !eref) {
            fprintf(stderr, "check_q4_interp: нет памяти на EMB\n");
            failures++;
        } else {
            int j;
            memset(&E, 0, sizeof E);
            E.kind = VTM_TENSOR_Q4;
            E.ndim = 2;
            E.dims[0] = (uint32_t)vocab;
            E.dims[1] = (uint32_t)dim;
            E.n_elements = (uint32_t)en;
            E.words = (const uint32_t *)eq;
            E.scales = escales;
            E.n_scales = (uint32_t)egroups;
            E.scale_block = VTM_Q4_GROUP_SIZE;
            st = q4_interp_emb(&E, ey, etok, dim, err, sizeof err);
            for (j = 0; j < dim; j++)
                eref[j] = (float)q4_ref_value(
                    eq, (int64_t)etok * dim + j, escales, NULL, 0);
            report("Q4 sym через interp (EMB)",
                   max_diff(ey, eref, (size_t)dim, &ref), ref,
                   st == VTM_OK ? CHECK_TOL : -1.0);
        }
        free(eq);
        free(escales);
        free(ey);
        free(eref);
    }
done:
    free(q);
    free(scales);
    free(mins);
    free(x);
    free(y_sym);
    free(y_asym);
    free(y_ref);
    free(y_nomins);
}

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

static void check_batch_case(int rows, int cols, int ntok, int scale_block,
                             float scale) {
    int n_trits = rows * cols;
    uint32_t seed = 0x2b47u + (uint32_t)rows * 131u + (uint32_t)cols
                  + (uint32_t)ntok * 7u;
    uint32_t *words = make_words(n_trits, seed);
    float *scales = make_scales(n_trits, scale_block, scale, seed ^ 0x5a5au);
    float *x = make_vec((size_t)cols * (size_t)ntok, seed ^ 0x51u);
    float *y_ref = (float *)calloc((size_t)rows * (size_t)ntok, sizeof(float));
    float *y_cand = (float *)calloc((size_t)rows * (size_t)ntok, sizeof(float));
    char label[128];
    double ref = 0.0;

    if (!words || !scales || !x || !y_ref || !y_cand) {
        fprintf(stderr, "check_batch_case: нет памяти\n");
        failures++;
        goto done;
    }
    batch_tmv_ref(y_ref, words, scales, x, rows, cols, ntok, scale_block);
    if (vtm_simd_tmv_blocks_batch(y_cand, (const uint8_t *)words, scales, x,
                                  rows, cols, ntok, scale_block) != 0) {
        fprintf(stderr, "check_batch_case: TMV-батч вернул ошибку\n");
        failures++;
        goto done;
    }
    snprintf(label, sizeof label, "TMV batch %dx%dxN%d блок %d scale %.2f",
             rows, cols, ntok, scale_block, (double)scale);
    report(label, max_diff(y_cand, y_ref, (size_t)rows * (size_t)ntok, &ref),
           ref, CHECK_TOL);

    batch_mm_ref(y_ref, words, scales, x, rows, cols, ntok, scale_block);
    if (vtm_simd_mm_batch(y_cand, (const uint8_t *)words, scales, x, rows,
                          cols, ntok, scale_block) != 0) {
        fprintf(stderr, "check_batch_case: MM-батч вернул ошибку\n");
        failures++;
        goto done;
    }
    snprintf(label, sizeof label, "MM batch %dx%dxN%d блок %d scale %.2f",
             rows, cols, ntok, scale_block, (double)scale);
    report(label, max_diff(y_cand, y_ref, (size_t)rows * (size_t)ntok, &ref),
           ref, CHECK_TOL);
done:
    free(words);
    free(scales);
    free(x);
    free(y_ref);
    free(y_cand);
}

static void check_batch_f32_case(int rows, int cols, int ntok) {
    uint32_t seed = 0x7f00u + (uint32_t)rows * 31u + (uint32_t)cols
                  + (uint32_t)ntok;
    float *w = make_vec((size_t)rows * (size_t)cols, seed);
    float *wt = make_vec((size_t)rows * (size_t)cols, seed ^ 0x1f1fu);
    float *x = make_vec((size_t)cols * (size_t)ntok, seed ^ 0x51u);
    float *y_ref = (float *)calloc((size_t)rows * (size_t)ntok, sizeof(float));
    float *y_cand = (float *)calloc((size_t)rows * (size_t)ntok, sizeof(float));
    char label[128];
    double ref = 0.0;
    int r, t;

    if (!w || !wt || !x || !y_ref || !y_cand) {
        fprintf(stderr, "check_batch_f32_case: нет памяти\n");
        failures++;
        goto done;
    }
    for (r = 0; r < rows; r++)
        for (t = 0; t < cols; t++)
            wt[(size_t)t * (size_t)rows + (size_t)r]
                = w[(size_t)r * (size_t)cols + (size_t)t];

    batch_f32_ref(y_ref, w, x, rows, cols, ntok, 0);
    if (vtm_simd_tmv_f32_batch(y_cand, w, x, rows, cols, ntok, 0) != 0) {
        fprintf(stderr, "check_batch_f32_case: ядро вернуло ошибку\n");
        failures++;
        goto done;
    }
    snprintf(label, sizeof label, "TMV f32 batch %dx%dxN%d", rows, cols, ntok);
    report(label, max_diff(y_cand, y_ref, (size_t)rows * (size_t)ntok, &ref),
           ref, CHECK_TOL);

    batch_f32_ref(y_ref, wt, x, rows, cols, ntok, 1);
    if (vtm_simd_tmv_f32_batch(y_cand, wt, x, rows, cols, ntok, 1) != 0) {
        fprintf(stderr, "check_batch_f32_case: transposed вернул ошибку\n");
        failures++;
        goto done;
    }
    snprintf(label, sizeof label, "TMV f32 batch T %dx%dxN%d", rows, cols,
             ntok);
    report(label, max_diff(y_cand, y_ref, (size_t)rows * (size_t)ntok, &ref),
           ref, CHECK_TOL);

    // MM f32 батч — тот же dense-путь (a [m,k] * b [k,N]).
    batch_f32_ref(y_ref, w, x, rows, cols, ntok, 0);
    if (vtm_simd_mm_f32_batch(y_cand, w, x, rows, cols, ntok) != 0) {
        fprintf(stderr, "check_batch_f32_case: MM вернул ошибку\n");
        failures++;
        goto done;
    }
    snprintf(label, sizeof label, "MM f32 batch %dx%dxN%d", rows, cols, ntok);
    report(label, max_diff(y_cand, y_ref, (size_t)rows * (size_t)ntok, &ref),
           ref, CHECK_TOL);
done:
    free(w);
    free(wt);
    free(x);
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

    // Батчевые ядра 0122: N токенов против одной матрицы. Малые формы
    // проверяют все ветки (1/4/8/16 токенов, хвосты блоков), большие —
    // реальные формы 1.5B (hidden 1536, intermediate 8960).
    {
        static const int bshapes[][3] = {
            {1, 1, 1}, {7, 37, 5}, {17, 128, 1}, {17, 128, 16},
            {128, 128, 32}, {128, 128, 128}, {64, 160, 3},
            {1536, 1536, 16}, {8960, 1536, 8}, {1536, 8960, 8}
        };
        size_t bi;
        for (bi = 0; bi < sizeof(bshapes) / sizeof(bshapes[0]); bi++) {
            int bn = bshapes[bi][0] * bshapes[bi][1];
            float bs = bn > 65536 ? 0.02f : 1.0f;
            check_batch_case(bshapes[bi][0], bshapes[bi][1], bshapes[bi][2],
                             128, bs);
            check_batch_case(bshapes[bi][0], bshapes[bi][1], bshapes[bi][2],
                             160, bs);
        }
        check_batch_f32_case(7, 13, 5);
        check_batch_f32_case(64, 37, 16);
    }

    // Q4 (kind 5, задача 0152): независимый эталон — деквантизация в double.
    // Проверяем Q4_0 и Q4_1, хвосты групп (cols не кратны 32) и MM n>1.
    {
        static const int q4shapes[][2] = {
            {1, 1}, {1, 31}, {1, 32}, {1, 33}, {7, 37}, {5, 53},
            {17, 128}, {64, 160}, {128, 128}, {3, 1031}
        };
        static const int q4mmshapes[][3] = {
            {5, 7, 3}, {8, 16, 8}, {64, 128, 1}, {16, 32, 64}, {3, 1000, 17}
        };
        size_t qi;
        for (qi = 0; qi < sizeof(q4shapes) / sizeof(q4shapes[0]); qi++) {
            int qn = q4shapes[qi][0] * q4shapes[qi][1];
            float qs = qn > 8192 ? 0.02f : 1.0f;
            check_q4_tmv_case(q4shapes[qi][0], q4shapes[qi][1], 0, qs);
            check_q4_tmv_case(q4shapes[qi][0], q4shapes[qi][1], 1, qs);
        }
        for (qi = 0; qi < sizeof(q4mmshapes) / sizeof(q4mmshapes[0]); qi++) {
            float qs = q4mmshapes[qi][1] > 128 ? 0.02f : 1.0f;
            check_q4_mm_case(q4mmshapes[qi][0], q4mmshapes[qi][1],
                             q4mmshapes[qi][2], 0, qs);
            check_q4_mm_case(q4mmshapes[qi][0], q4mmshapes[qi][1],
                             q4mmshapes[qi][2], 1, qs);
        }
        check_q4_args();
        check_q4_interp();
    }

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

// --- стоимость барьера пула (задача 0113) -----------------------------------
//
// Пустое задание через vtm_parallel_rows() измеряет накладные расходы
// барьера: разбудить воркеров, опубликовать параметры, дождаться нуля в
// workers_left. На 1.5B-форме таких барьеров ~200 на токен, поэтому цена
// одного барьера при 8 потоках была заметной долей времени декода.
// Печатается минимум из 32 замеров (меньше страдают от чужой нагрузки).
static void pool_nop(void *p, int tid, int r0, int r1) {
    (void)p; (void)tid; (void)r0; (void)r1;
}

static int run_pool(int iters) {
    int nt;
    (void)iters;
    printf("Стоимость пустого задания (барьер пула), мкс/вызов:\n");
    for (nt = 1; nt <= VTM_THREADS_MAX; nt++) {
        double best = 0.0;
        int rep, i;
        vtm_threads_set(nt);
        vtm_parallel_rows(8, 1e9, pool_nop, NULL);   // прогрев, создать воркеров
        for (rep = 0; rep < 32; rep++) {
            double t0 = now_ms();
            for (i = 0; i < 20; i++) vtm_parallel_rows(8, 1e9, pool_nop, NULL);
            t0 = (now_ms() - t0) / 20.0;
            if (best <= 0.0 || t0 < best) best = t0;
        }
        printf("  nt=%d %8.2f мкс\n", nt, best * 1000.0);
    }
    printf("порог параллели %.0f MAC; потолок на задание — min(запрошенное, 6)\n",
           VTM_PAR_MIN_WORK);
    return 0;
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

// --- батчевый prefill 1.5B-формы: по токену против пачки (задача 0122) -----
//
// «До» — то, как SIMD-бэкенд считал prefill до 0122: N отдельных вызовов
// одноколоночного tmv_blocks. «После» — vtm_simd_tmv_blocks_batch на те же N
// токенов (вес читается и распаковывается один раз). Формы — реальный граф
// Qwen2.5-1.5B: hidden 1536, intermediate 8960, vocab 151936, 28 слоёв;
// на слой 4 матрицы hidden² и 3 hidden×intermediate, плюс lm_head.
//
// TMV/MM — 98% времени графа 1.5B (docs/vtm-perf.md), поэтому сумма по этим
// формам и есть оценка prefill. Живой замер CLI (`vtm run --prefill-batch`)
// идёт через interp.c поверх этих же ядер — см. отчёт задачи.

#define PREFILL_HIDDEN 1536
#define PREFILL_INTER  8960
#define PREFILL_VOCAB  151936
#define PREFILL_LAYERS 28

typedef struct {
    int rows;
    int cols;
    int count;        // сколько таких матриц в графе (на 1.5B-форму)
    const char *name;
} pref_shape;

// Время одной формы: N одноколоночных вызовов против одного батчевого.
static int bench_batch_one(const pref_shape *s, int ntok, int iters,
                           double *ms_single, double *ms_batch) {
    int n_trits = s->rows * s->cols;
    uint32_t seed = 0x51deu + (uint32_t)s->rows * 13u + (uint32_t)s->cols;
    uint32_t *words = make_words(n_trits, seed);
    float *scales = make_scales(n_trits, 160, 0.02f, seed ^ 0x77u);
    float *xrow = make_vec((size_t)ntok * (size_t)s->cols, seed ^ 0x11u);
    float *xcol = (float *)malloc((size_t)ntok * (size_t)s->cols
                                  * sizeof(float));
    float *y1 = (float *)calloc((size_t)s->rows, sizeof(float));
    float *yb = (float *)calloc((size_t)s->rows * (size_t)ntok,
                                sizeof(float));
    double t0, t1;
    int it, j, t;
    int rc = -1;

    if (!words || !scales || !xrow || !xcol || !y1 || !yb) {
        fprintf(stderr, "bench_batch_one: нет памяти\n");
        goto done;
    }
    for (j = 0; j < ntok; j++)
        for (t = 0; t < s->cols; t++)
            xcol[(size_t)t * (size_t)ntok + (size_t)j]
                = xrow[(size_t)j * (size_t)s->cols + (size_t)t];

    // Прогрев: страницы, LUT, пул потоков.
    if (vtm_backend_simd.tmv_blocks(y1, (const uint8_t *)words, scales, xrow,
                                    s->rows, s->cols,
                                    (int)VTM_SCALE_BLOCK_160) != 0
        || vtm_simd_tmv_blocks_batch(yb, (const uint8_t *)words, scales, xcol,
                                     s->rows, s->cols, ntok,
                                     (int)VTM_SCALE_BLOCK_160) != 0) {
        fprintf(stderr, "bench_batch_one: ядро вернуло ошибку\n");
        goto done;
    }
    t0 = now_ms();
    for (it = 0; it < iters; it++)
        for (j = 0; j < ntok; j++)
            vtm_backend_simd.tmv_blocks(
                y1, (const uint8_t *)words, scales,
                xrow + (size_t)j * (size_t)s->cols, s->rows, s->cols,
                (int)VTM_SCALE_BLOCK_160);
    t1 = now_ms();
    *ms_single = (t1 - t0) / (double)iters;
    t0 = now_ms();
    for (it = 0; it < iters; it++)
        vtm_simd_tmv_blocks_batch(yb, (const uint8_t *)words, scales, xcol,
                                  s->rows, s->cols, ntok,
                                  (int)VTM_SCALE_BLOCK_160);
    t1 = now_ms();
    *ms_batch = (t1 - t0) / (double)iters;
    rc = 0;
done:
    free(words);
    free(scales);
    free(xrow);
    free(xcol);
    free(y1);
    free(yb);
    return rc;
}

static int run_batch_bench(int ntok, int iters) {
    // count — сколько таких матриц во всей 1.5B-форме: на слой 4+2+1, слоёв 28.
    static const pref_shape shapes[] = {
        {PREFILL_HIDDEN, PREFILL_HIDDEN, 4 * PREFILL_LAYERS, "q/k/v/o"},
        {PREFILL_INTER, PREFILL_HIDDEN, 2 * PREFILL_LAYERS, "gate/up"},
        {PREFILL_HIDDEN, PREFILL_INTER, 1 * PREFILL_LAYERS, "down"},
        {PREFILL_VOCAB, PREFILL_HIDDEN, 1, "lm_head"}
    };
    const size_t n_shapes = sizeof(shapes) / sizeof(shapes[0]);
    double single_total = 0.0, batch_total = 0.0, mac_total = 0.0;
    size_t i;

    if (ntok < 1) ntok = 1;
    vtm_threads_set(4);   // цель 0122 — 4 потока; вернём ниже env-дефолт
    printf("Батчевый prefill 1.5B-формы: N=%d токенов на пачку, 4 потока, "
           "блок 160\n", ntok);
    printf("«до» — N вызовов одноколоночного SIMD-ядра (путь до 0122), "
           "«после» — батчевое ядро\n");
    printf("  %-10s %14s %12s %12s %12s   %s\n", "матрица", "MAC/пачку",
           "до, мс", "после, мс", "x", "GFLOP/s после");
    for (i = 0; i < n_shapes; i++) {
        double ms_single = 0.0, ms_batch = 0.0;
        double mac = 2.0 * (double)shapes[i].rows * (double)shapes[i].cols
                   * (double)ntok * (double)shapes[i].count;
        int its = iters;
        if ((double)shapes[i].rows * (double)shapes[i].cols > 4.0e6)
            its = 1;
        if (bench_batch_one(&shapes[i], ntok, its, &ms_single, &ms_batch)
            != 0) {
            vtm_threads_set(0);
            return 1;
        }
        single_total += (double)shapes[i].count * ms_single;
        batch_total += (double)shapes[i].count * ms_batch;
        mac_total += mac;
        printf("  %-10s %14.1f %12.2f %12.2f %11.2fx   %8.1f\n",
               shapes[i].name,
               (double)shapes[i].rows * (double)shapes[i].cols * (double)ntok,
               ms_single, ms_batch, ms_single / ms_batch,
               mac / (ms_batch * (double)shapes[i].count * 1e6));
    }
    printf("\nИтого %d токенов (28 слоёв + lm_head, TMV/MM — 98%% графа):\n",
           ntok);
    printf("  до    %9.1f мс   (%.1f GFLOP/s)\n", single_total,
           mac_total / (single_total * 1e6));
    printf("  после %9.1f мс   (%.1f GFLOP/s)   ускорение %.2fx\n",
           batch_total, mac_total / (batch_total * 1e6),
           single_total / batch_total);

    // Проекция prefill 32/128 токенов: пачки по ntok, «до» — по токену.
    {
        static const int ns[] = {32, 128};
        size_t k;
        printf("\nПроекция prefill (пачки по %d, последняя может быть "
               "неполной):\n", ntok);
        for (k = 0; k < sizeof ns / sizeof ns[0]; k++) {
            double n_chunks = (double)((ns[k] + ntok - 1) / ntok);
            double before = single_total * (double)ns[k] / (double)ntok;
            double after = batch_total * n_chunks;
            printf("  N=%3d: до %8.1f мс, после %8.1f мс, %.2fx\n", ns[k],
                   before, after, before / after);
        }
    }
    printf("\nФизический потолок: вес читается N раз -> 1 раз, но арифметика "
           "та же;\nожидаемо ~2x, не 10x (docs/vtm-perf.md).\n");
    vtm_threads_set(0);
    return 0;
}

// --- логиты модели ---------------------------------------------------------

// --- Q4: скорость и RSS против fp32-пути (задача 0152) --------------------
//
// Форма по умолчанию — lm_head Qwen-0.5B: 151936x896 ≈ 136 М весов (это
// крупнейшая матрица 0.5B-раскладки). В Q4 она занимает 0,5 байта/вес +
// 4 байта на 32 веса (масштабы) ≈ 76 МБ; fp32-копия тех же весов — 544 МБ.
// Замер показывает: (1) во сколько раз Q4-ядро медленнее/быстрее fp32 на
// одной и той же арифметике; (2) на сколько растёт RSS на веса и на само
// ядро (у ядра — 4 КБ scratch на поток, роста быть не должно).
// Минимум по итерациям, а не среднее: на общей машине среднее ловит чужие
// процессы и троттлинг (разброс был 45–365 мс на одном и том же fp32-ядре).
static double bench_q4_tmv(const uint8_t *q, const float *scales,
                           const float *x, float *y, int rows, int cols,
                           int iters) {
    double best = 1e300;
    int i;
    if (vtm_q4_tmv(y, q, scales, NULL, x, rows, cols, 0) != 0) return -1.0;
    for (i = 0; i < iters; i++) {
        double t0 = now_ms();
        vtm_q4_tmv(y, q, scales, NULL, x, rows, cols, 0);
        {
            double dt = now_ms() - t0;
            if (dt < best) best = dt;
        }
    }
    return best;
}

static double bench_f32_tmv(const float *w, const float *x, float *y,
                            int rows, int cols, int iters) {
    double best = 1e300;
    int i;
    f32_tmv(y, w, x, rows, cols);
    for (i = 0; i < iters; i++) {
        double t0 = now_ms();
        f32_tmv(y, w, x, rows, cols);
        {
            double dt = now_ms() - t0;
            if (dt < best) best = dt;
        }
    }
    return best;
}

static int run_q4_bench(int rows, int cols, int iters) {
    int64_t n64 = (int64_t)rows * (int64_t)cols;
    int n = (int)n64;
    size_t groups = ((size_t)n + VTM_Q4_GROUP_SIZE - 1) / VTM_Q4_GROUP_SIZE;
    size_t q_bytes = groups * (VTM_Q4_GROUP_SIZE / 2);
    size_t s_bytes = groups * sizeof(float);
    size_t f32_bytes = (size_t)n * sizeof(float);
    uint8_t *q = NULL;
    float *scales = NULL, *x = NULL, *y = NULL, *y_f32 = NULL, *w = NULL;
    size_t rss0, rss_q4, rss_kernel, rss_f32;
    double t_q4, t_f32;
    double maxd = 0.0, maxr = 0.0;
    int r, s;

    failures = 0;
    if (n64 <= 0 || n64 > (int64_t)0x3fffffff) {
        fprintf(stderr, "Q4: слишком большая форма %dx%d\n", rows, cols);
        return 2;
    }
    x = make_vec((size_t)cols, 0x123u);
    y = (float *)calloc((size_t)rows, sizeof(float));
    y_f32 = (float *)calloc((size_t)rows, sizeof(float));
    if (!x || !y || !y_f32) {
        fprintf(stderr, "Q4: нет памяти на форму %dx%d\n", rows, cols);
        goto done;
    }
    // База — до весов: рост RSS должен быть от упакованных весов + масштабов,
    // а не от fp32-копии.
    rss0 = rss_bytes();
    q = make_q4(n, 0x5150u);
    scales = make_q4_scales(n, 0.0f, 0x77u);
    if (!q || !scales) {
        fprintf(stderr, "Q4: нет памяти на веса %dx%d\n", rows, cols);
        goto done;
    }
    // Ядро Q4 (прогрев, чтобы не мерить первый вызов).
    if (vtm_q4_tmv(y, q, scales, NULL, x, rows, cols, 0) != 0) {
        fprintf(stderr, "Q4: ядро вернуло ошибку\n");
        goto done;
    }
    rss_q4 = rss_bytes();
    t_q4 = bench_q4_tmv(q, scales, x, y, rows, cols, iters);
    rss_kernel = rss_bytes();
    if (t_q4 < 0.0) {
        fprintf(stderr, "Q4: ядро вернуло ошибку на замере\n");
        goto done;
    }

    // Точность на выборке строк против независимой деквантизации в double.
    s = rows < 8 ? rows : 8;
    for (r = 0; r < s; r++) {
        int rr = (int)((int64_t)r * (rows - 1) / (s > 1 ? s - 1 : 1));
        int64_t base = (int64_t)rr * (int64_t)cols;
        double acc = 0.0, d;
        int j;
        for (j = 0; j < cols; j++)
            acc += q4_ref_value(q, base + (int64_t)j, scales, NULL, 0)
                 * (double)x[j];
        d = fabs(acc - (double)y[rr]);
        if (d > maxd) maxd = d;
        if (fabs(acc) > maxr) maxr = fabs(acc);
    }

    // fp32-путь: материализуем те же веса в fp32 (как пришлось бы без Q4).
    w = (float *)malloc(f32_bytes);
    if (!w) {
        fprintf(stderr, "Q4: нет памяти на fp32-копию %.1f МБ\n",
                mb_of(f32_bytes));
        goto done;
    }
    for (r = 0; r < n; r++) w[r] = (float)q4_ref_value(q, r, scales, NULL, 0);
    rss_f32 = rss_bytes();
    f32_tmv(y_f32, w, x, rows, cols);
    t_f32 = bench_f32_tmv(w, x, y_f32, rows, cols, iters);
    if (t_f32 < 0.0) {
        fprintf(stderr, "Q4: fp32-ядро вернуло ошибку\n");
        goto done;
    }

    printf("Q4 TMV: %dx%d = %d весов (1 токен), %d итераций\n", rows, cols, n,
           iters);
    printf("  Q4 упаковано: %8.1f МБ (+масштабы %6.1f МБ) = %.2f Б/вес\n",
           mb_of(q_bytes), mb_of(s_bytes),
           (double)(q_bytes + s_bytes) / (double)n);
    printf("  fp32-копия:   %8.1f МБ = 4.00 Б/вес (в %.1f раза больше)\n",
           mb_of(f32_bytes),
           (double)f32_bytes / (double)(q_bytes + s_bytes));
    if (rss0) {
        printf("  RSS база: %.1f МБ; после Q4-весов: %.1f МБ (%+.1f); "
               "после ядра: %.1f МБ (%+.1f за ядро)\n",
               mb_of(rss0), mb_of(rss_q4),
               mb_of(rss_q4) - mb_of(rss0), mb_of(rss_kernel),
               mb_of(rss_kernel) - mb_of(rss_q4));
        printf("  RSS + fp32-копия: %.1f МБ (%+.1f к базе, %+.1f к Q4)\n",
               mb_of(rss_f32), mb_of(rss_f32) - mb_of(rss0),
               mb_of(rss_f32) - mb_of(rss_q4));
    }
    printf("  Q4  : %8.3f мс/итер (мин)  %7.2f GFLOP/s\n", t_q4,
           2.0 * (double)n / (t_q4 * 1e6));
    printf("  fp32: %8.3f мс/итер (мин)  %7.2f GFLOP/s  (Q4/fp32 = %.2fx)\n",
           t_f32, 2.0 * (double)n / (t_f32 * 1e6), t_f32 / t_q4);
    printf("  точность Q4 vs double-эталон (выборка %d строк): "
           "max|diff|=%.3e (max|ref|=%.3e)\n", s, maxd, maxr);
    if (maxd > CHECK_TOL) {
        printf("  Q4 точность: FAIL (> %.0e)\n", CHECK_TOL);
        failures++;
        goto done;
    }
    printf("  Q4 точность: ok (в пределах %.0e)\n", CHECK_TOL);
done:
    free(q);
    free(scales);
    free(x);
    free(y);
    free(y_f32);
    free(w);
    return failures ? 1 : 0;
}

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
           "[--threads] [--pool] [--logits PATH] [--no-mm]\n"
           "          [--batch] [--batch-tokens N] "
           "[--q4 [--q4-rows N] [--q4-cols N] [--q4-iters N]]\n"
           "  env VTM_BACKEND — то же, что --backend (по умолчанию simd);\n"
           "  env VTM_THREADS — число потоков ядер (по умолчанию ядра, ≤ 8)\n",
           prog);
}

int main(int argc, char **argv) {
    int rows = 1024, cols = 1024, iters = 50, scale_block = 160;
    int check = 0, no_mm = 0, threads = 0, pool = 0, batch = 0;
    int batch_tokens = 16;
    int q4 = 0, q4_rows = 151936, q4_cols = 896, q4_iters = 3;
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
        } else if (strcmp(argv[i], "--q4") == 0) {
            q4 = 1;
        } else if (strcmp(argv[i], "--q4-rows") == 0 && i + 1 < argc) {
            q4_rows = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--q4-cols") == 0 && i + 1 < argc) {
            q4_cols = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--q4-iters") == 0 && i + 1 < argc) {
            q4_iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--scale-block") == 0 && i + 1 < argc) {
            scale_block = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            backend_name = argv[++i];
        } else if (strcmp(argv[i], "--check") == 0) {
            check = 1;
        } else if (strcmp(argv[i], "--threads") == 0) {
            threads = 1;
        } else if (strcmp(argv[i], "--pool") == 0) {
            pool = 1;
        } else if (strcmp(argv[i], "--no-mm") == 0) {
            no_mm = 1;
        } else if (strcmp(argv[i], "--batch") == 0) {
            batch = 1;
        } else if (strcmp(argv[i], "--batch-tokens") == 0 && i + 1 < argc) {
            batch_tokens = atoi(argv[++i]);
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
    if (q4 && (q4_rows <= 0 || q4_cols <= 0 || q4_iters <= 0 ||
               q4_cols > (1 << 20))) {
        fprintf(stderr, "плохие --q4-rows/--q4-cols/--q4-iters\n");
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

    if (q4)
        return run_q4_bench(q4_rows, q4_cols, q4_iters);
    if (pool)
        return run_pool(iters);
    if (batch) {
        if (batch_tokens < 1 || batch_tokens > 128) {
            fprintf(stderr, "--batch-tokens должен быть 1..128\n");
            return 2;
        }
        return run_batch_bench(batch_tokens, iters);
    }
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
