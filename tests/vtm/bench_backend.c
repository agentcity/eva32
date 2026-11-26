// Бенчмарк и проверка SIMD-бэкенда VTM (src/vtm/backend_neon.c,
// backend_avx2.c). Один бинарник делает три вещи:
//
//   ./bench_backend                 — GFLOP/s: scalar-lut243 / scalar-direct /
//                                     simd (матрица 1024x1024) + MM + RMSN;
//                                     в конце — плотный GEMV f32 n == 1
//                                     (tied lm_head, задача 0159), форма
//                                     задаётся --gemv-rows/--gemv-cols;
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
#include "api.h"   // 0199: паритет сгенерированных токенов через C-ABI
// Только тесту: минимальная машина без загрузчика, чтобы проверить проводку
// kind 5 в interp.c (op_tmv/op_mm/op_emb) без сборки контейнера.
#include "vtm_internal.h"
#include "trit.h"

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
// Порог для int8-пути батчевого Q4 (задача 0163). Там ошибка не
// вычислительная, а от квантования активаций в int8, поэтому порог на три
// порядка шире: замер на формах 3B Q4 в режиме 2 (два прохода) — 2048x2048x16
// 5,2e-4, 11008x2048x8 1,6e-3, 2048x11008x4 3,0e-3. Режим 1 (один проход) на
// порядок хуже и в проверку не берётся — его числа в отчёте ai/tasks/0163.
#define CHECK_TOL_I8 1e-2

// --- утилиты ---------------------------------------------------------------

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

// CPU-время процесса (сумма по потокам): на загруженной машине wall
// непригоден для сравнения путей — планировщик отнимает процессор у обоих
// по-разному. CPU-время стабильно и сравнимо, поэтому prefill-замеры ниже
// печатают его рядом с wall (задача 0168).
static double now_cpu_ms(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts) != 0) return -1.0;
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

// int8-таблица (kind 4, задача 0170) против независимого эталона в double:
// w = q·scale, масштаб на блок 128 по сплющенной матрице. Проверяются
//   * vtm_i8_tmv  — TMV/GEMV n == 1 (double, эталон формата);
//   * vtm_i8_gemv — рабочий GEMV декода: режим 0 (точные активации, f32) и
//     режимы 2/3 (int8-dot; порог CHECK_TOL_I8 — ошибка от квантования
//     активаций);
//   * vtm_i8_mm   — общий n > 1;
//   * vtm_i8_dequant_row — строка EMB (точно: там нет сумм).
// cols не кратен 128 — отдельный случай: SIMD-ядро отказывает, работает
// переносимый путь, поэтому проверка не зависит от ISA.
static double i8_ref_dot(const int8_t *q, const float *scales, int64_t base,
                         int cols, const float *v, int stride) {
    double acc = 0.0;
    int j0 = 0;

    while (j0 < cols) {
        int64_t idx = base + (int64_t)j0;
        int blk = (int)(idx / (int64_t)VTM_SCALE_BLOCK_128);
        int m = (int)((int64_t)VTM_SCALE_BLOCK_128
                      - idx % (int64_t)VTM_SCALE_BLOCK_128);
        double sum = 0.0;
        int t;
        if (m > cols - j0) m = cols - j0;
        for (t = 0; t < m; t++)
            sum += (double)q[j0 + t] * (double)v[(size_t)(j0 + t) * (size_t)stride];
        acc += sum * (double)scales[blk];
        j0 += m;
    }
    return acc;
}

static void check_i8_case(int rows, int cols, int ncol_mm) {
    int n = rows * cols;
    int nblk = (n + (int)VTM_SCALE_BLOCK_128 - 1) / (int)VTM_SCALE_BLOCK_128;
    uint32_t seed = 0x1700u + (uint32_t)rows * 17u + (uint32_t)cols;
    int8_t *q = (int8_t *)malloc((size_t)(n ? n : 1));
    float *scales = (float *)malloc((size_t)(nblk ? nblk : 1) * sizeof(float));
    float *x = make_vec((size_t)cols, seed ^ 0x11u);
    float *y_ref = (float *)calloc((size_t)rows, sizeof(float));
    float *y = (float *)calloc((size_t)rows, sizeof(float));
    float *b = make_vec((size_t)cols * (size_t)ncol_mm, seed ^ 0x22u);
    float *ym_ref = (float *)calloc((size_t)rows * (size_t)ncol_mm,
                                    sizeof(float));
    float *ym = (float *)calloc((size_t)rows * (size_t)ncol_mm,
                                sizeof(float));
    float *emb = (float *)calloc((size_t)cols, sizeof(float));
    char label[96];
    int i, r, c;

    if (!q || !scales || !x || !y_ref || !y || !b || !ym_ref || !ym || !emb) {
        fprintf(stderr, "check_i8_case: нет памяти\n");
        failures++;
        goto done;
    }
    for (i = 0; i < n; i++)
        q[i] = (int8_t)((int)(rng_next(&seed) % 255u) - 127);
    for (i = 0; i < nblk; i++)
        scales[i] = 0.0005f + 0.02f * (float)(rng_next(&seed) % 100u) / 99.0f;

    for (r = 0; r < rows; r++) {
        const int8_t *row = q + (size_t)r * (size_t)cols;
        y_ref[r] = (float)i8_ref_dot(row, scales, (int64_t)r * cols, cols,
                                     x, 1);
        for (c = 0; c < ncol_mm; c++)
            ym_ref[(size_t)r * (size_t)ncol_mm + (size_t)c] = (float)
                i8_ref_dot(row, scales, (int64_t)r * cols, cols, b + (size_t)c,
                           ncol_mm);
    }

    {
        double ref = 0.0;
        if (vtm_i8_tmv(y, q, scales, x, rows, cols) != 0) {
            printf("  i8 TMV: ядро вернуло ошибку\n");
            failures++;
        } else {
            double d = max_diff(y, y_ref, (size_t)rows, &ref);
            snprintf(label, sizeof label, "i8 TMV %dx%d (double)", rows, cols);
            report(label, d, ref, 1e-5 * (ref > 1.0 ? ref : 1.0));
        }
    }
    {
        static const int modes[3] = {0, 2, 3};
        int mi;
        for (mi = 0; mi < 3; mi++) {
            const int saved = vtm_q4k_i8_mode();
            const int eff = vtm_q4k_i8_set_mode(modes[mi]);
            double ref = 0.0;

            if (modes[mi] != 0 && eff == 0) {
                vtm_q4k_i8_set_mode(saved);
                continue;   // int8-dot на этой сборке/CPU недоступен
            }
            if (vtm_i8_gemv(y, q, scales, x, rows, cols) != 0) {
                printf("  i8 GEMV: ядро вернуло ошибку\n");
                failures++;
            } else {
                snprintf(label, sizeof label, "i8 GEMV %dx%d (mode %d)",
                         rows, cols, eff);
                report(label, max_diff(y, y_ref, (size_t)rows, &ref), ref,
                       (eff == 0 ? 1e-5 : CHECK_TOL_I8)
                       * (ref > 1.0 ? ref : 1.0));
            }
            vtm_q4k_i8_set_mode(saved);
        }
    }
    if (vtm_i8_mm(ym, q, scales, b, rows, cols, ncol_mm) != 0) {
        printf("  i8 MM: ядро вернуло ошибку\n");
        failures++;
    } else {
        double ref = 0.0;
        snprintf(label, sizeof label, "i8 MM %dx%dx%d", rows, cols, ncol_mm);
        report(label, max_diff(ym, ym_ref, (size_t)rows * (size_t)ncol_mm,
                               &ref), ref, 1e-5 * (ref > 1.0 ? ref : 1.0));
    }
    if (vtm_i8_dequant_row(emb, q, scales, (int64_t)cols, cols) != 0) {
        printf("  i8 dequant_row: ядро вернуло ошибку\n");
        failures++;
    } else {
        double md = 0.0, mr = 0.0;
        for (c = 0; c < cols; c++) {
            double want = (double)q[(size_t)cols + (size_t)c]
                        * (double)scales[((size_t)cols + (size_t)c)
                                         / VTM_SCALE_BLOCK_128];
            double d = fabs((double)emb[c] - want);
            if (d > md) md = d;
            if (fabs(want) > mr) mr = fabs(want);
        }
        snprintf(label, sizeof label, "i8 EMB-строка %d", cols);
        report(label, md, mr, 1e-6 * (mr > 1.0 ? mr : 1.0));
    }
done:
    free(q);
    free(scales);
    free(x);
    free(y_ref);
    free(y);
    free(b);
    free(ym_ref);
    free(ym);
    free(emb);
}

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

// Q4-батч против поштучного пути (задача 0162). Это ровно критерий приёмки:
// y[r*N+j] пачки должен совпасть с N вызовами vtm_q4_tmv/vtm_q4_mm на
// собранный contiguous вход токена j. Отдельно (check_q4_tmv_case выше)
// поштучное ядро уже сверено с независимой double-деквантизацией, поэтому
// здесь проверяется именно батчевая проводка и её арифметика.
static void check_q4_batch_case(int rows, int cols, int ntok, int asym,
                                float scale) {
    int n = rows * cols;
    uint32_t seed = 0x6200u + (uint32_t)rows * 29u + (uint32_t)cols * 7u
                  + (uint32_t)ntok + (uint32_t)asym * 3u;
    uint8_t *q = make_q4(n, seed);
    float *scales = make_q4_scales(n, scale, seed ^ 0x5bu);
    float *mins = asym ? make_q4_mins(n, seed ^ 0x1du) : NULL;
    float *x = make_vec((size_t)cols * (size_t)ntok, seed ^ 0x71u);
    float *y_ref = (float *)calloc((size_t)rows * (size_t)ntok, sizeof(float));
    float *y_bat = (float *)calloc((size_t)rows * (size_t)ntok, sizeof(float));
    float *xt = (float *)malloc((size_t)cols * sizeof(float));
    float *yt = (float *)malloc((size_t)rows * sizeof(float));
    char label[112];
    double ref = 0.0;
    int j, r, t;

    if (!q || !scales || (asym && !mins) || !x || !y_ref || !y_bat || !xt
        || !yt) {
        fprintf(stderr, "check_q4_batch_case: нет памяти\n");
        failures++;
        goto done;
    }
    for (j = 0; j < ntok; j++) {
        for (t = 0; t < cols; t++)
            xt[t] = x[(size_t)t * (size_t)ntok + (size_t)j];
        if (vtm_q4_tmv(yt, q, scales, mins, xt, rows, cols, asym) != 0) {
            fprintf(stderr, "check_q4_batch_case: поштучное ядро вернуло "
                            "ошибку\n");
            failures++;
            goto done;
        }
        for (r = 0; r < rows; r++)
            y_ref[(size_t)r * (size_t)ntok + (size_t)j] = yt[r];
    }
    {
        // f32-путь 0162 проверяется отдельно от int8, даже если окружение
        // включило VTM_Q4_I8: это разные контракты точности.
        const int saved0 = vtm_q4_i8_mode();
        vtm_q4_i8_set_mode(0);
        if (vtm_q4_tmv_batch(y_bat, q, scales, mins, x, rows, cols, ntok,
                             asym) != 0) {
            fprintf(stderr, "check_q4_batch_case: батчевое ядро вернуло "
                            "ошибку\n");
            failures++;
            vtm_q4_i8_set_mode(saved0);
            goto done;
        }
        vtm_q4_i8_set_mode(saved0);
    }
    snprintf(label, sizeof label, "Q4%s TMV-батч %dx%dx%d scale %.2f",
             asym ? "1" : "0", rows, cols, ntok, (double)scale);
    report(label, max_diff(y_bat, y_ref, (size_t)rows * (size_t)ntok, &ref),
           ref, CHECK_TOL);
    // int8-dot (0163): та же пачка, но ядром SDOT/VNNI — активации квантуются
    // в int8. Эталон тот же (поштучный f32-путь), порог — CHECK_TOL_I8.
    // Если int8-ядра на CPU/сборке нет, vtm_q4_i8_set_mode(2) вернёт 0 и
    // проверка пропускается.
    {
        const int saved = vtm_q4_i8_mode();
        if (vtm_q4_i8_set_mode(2) != 0) {
            memset(y_bat, 0, (size_t)rows * (size_t)ntok * sizeof(float));
            if (vtm_q4_tmv_batch(y_bat, q, scales, mins, x, rows, cols, ntok,
                                 asym) != 0) {
                fprintf(stderr, "check_q4_batch_case: int8-ядро вернуло "
                                "ошибку\n");
                failures++;
            } else {
                snprintf(label, sizeof label,
                         "Q4%s TMV-батч int8 %dx%dx%d", asym ? "1" : "0",
                         rows, cols, ntok);
                report(label,
                       max_diff(y_bat, y_ref, (size_t)rows * (size_t)ntok,
                                &ref), ref, CHECK_TOL_I8);
            }
        }
        vtm_q4_i8_set_mode(saved);
    }
done:
    free(q);
    free(scales);
    free(mins);
    free(x);
    free(y_ref);
    free(y_bat);
    free(xt);
    free(yt);
}

static void check_q4_mm_batch_case(int m, int k, int ntok, int asym,
                                   float scale) {
    int n_el = m * k;
    uint32_t seed = 0x6d00u + (uint32_t)m * 13u + (uint32_t)k * 5u
                  + (uint32_t)ntok + (uint32_t)asym * 23u;
    uint8_t *q = make_q4(n_el, seed);
    float *scales = make_q4_scales(n_el, scale, seed ^ 0x37u);
    float *mins = asym ? make_q4_mins(n_el, seed ^ 0x09u) : NULL;
    float *b = make_vec((size_t)k * (size_t)ntok, seed ^ 0x4fu);
    float *y_ref = (float *)calloc((size_t)m * (size_t)ntok, sizeof(float));
    float *y_bat = (float *)calloc((size_t)m * (size_t)ntok, sizeof(float));
    float *bt = (float *)malloc((size_t)k * sizeof(float));
    float *yt = (float *)malloc((size_t)m * sizeof(float));
    char label[112];
    double ref = 0.0;
    int j, r, t;

    if (!q || !scales || (asym && !mins) || !b || !y_ref || !y_bat || !bt
        || !yt) {
        fprintf(stderr, "check_q4_mm_batch_case: нет памяти\n");
        failures++;
        goto done;
    }
    for (j = 0; j < ntok; j++) {
        for (t = 0; t < k; t++)
            bt[t] = b[(size_t)t * (size_t)ntok + (size_t)j];
        // n == 1 у vtm_q4_mm — это одноколоночный путь (vtm_q4_tmv), ровно
        // то, что даёт поштучный prefill на lm_head.
        if (vtm_q4_mm(yt, q, scales, mins, bt, m, k, 1, asym) != 0) {
            fprintf(stderr, "check_q4_mm_batch_case: поштучное ядро вернуло "
                            "ошибку\n");
            failures++;
            goto done;
        }
        for (r = 0; r < m; r++)
            y_ref[(size_t)r * (size_t)ntok + (size_t)j] = yt[r];
    }
    {
        const int saved0 = vtm_q4_i8_mode();
        vtm_q4_i8_set_mode(0);
        if (vtm_q4_mm_batch(y_bat, q, scales, mins, b, m, k, ntok, asym) != 0) {
            fprintf(stderr, "check_q4_mm_batch_case: батчевое ядро вернуло "
                            "ошибку\n");
            failures++;
            vtm_q4_i8_set_mode(saved0);
            goto done;
        }
        vtm_q4_i8_set_mode(saved0);
    }
    snprintf(label, sizeof label, "Q4%s MM-батч %dx%dx%d scale %.2f",
             asym ? "1" : "0", m, k, ntok, (double)scale);
    report(label, max_diff(y_bat, y_ref, (size_t)m * (size_t)ntok, &ref), ref,
           CHECK_TOL);
    {
        const int saved = vtm_q4_i8_mode();
        if (vtm_q4_i8_set_mode(2) != 0) {
            memset(y_bat, 0, (size_t)m * (size_t)ntok * sizeof(float));
            if (vtm_q4_mm_batch(y_bat, q, scales, mins, b, m, k, ntok,
                                asym) != 0) {
                fprintf(stderr, "check_q4_mm_batch_case: int8-ядро вернуло "
                                "ошибку\n");
                failures++;
            } else {
                snprintf(label, sizeof label, "Q4%s MM-батч int8 %dx%dx%d",
                         asym ? "1" : "0", m, k, ntok);
                report(label,
                       max_diff(y_bat, y_ref, (size_t)m * (size_t)ntok, &ref),
                       ref, CHECK_TOL_I8);
            }
        }
        vtm_q4_i8_set_mode(saved);
    }
done:
    free(q);
    free(scales);
    free(mins);
    free(b);
    free(y_ref);
    free(y_bat);
    free(bt);
    free(yt);
}

// Контракт батчевых Q4-ядер: NULL/отрицательные размеры — -1, пустая
// размерность — 0; N == 1 — прежний одноколоночный путь (тот же результат).
static void check_q4_batch_args(void) {
    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float y1[2] = {0.0f, 0.0f};
    float y2[2] = {0.0f, 0.0f};
    float scales[1] = {1.0f};
    uint8_t q[16] = {0};
    int bad = 0;

    if (vtm_q4_tmv_batch(NULL, q, scales, NULL, x, 1, 2, 2, 0) != -1) bad = 1;
    if (vtm_q4_tmv_batch(y1, NULL, scales, NULL, x, 1, 2, 2, 0) != -1) bad = 1;
    if (vtm_q4_tmv_batch(y1, q, NULL, NULL, x, 1, 2, 2, 0) != -1) bad = 1;
    if (vtm_q4_tmv_batch(y1, q, scales, NULL, NULL, 1, 2, 2, 0) != -1) bad = 1;
    if (vtm_q4_tmv_batch(y1, q, scales, NULL, x, -1, 2, 2, 0) != -1) bad = 1;
    if (vtm_q4_tmv_batch(y1, q, scales, NULL, x, 1, 2, 2, 1) != -1) bad = 1;
    if (vtm_q4_tmv_batch(y1, q, scales, NULL, x, 0, 2, 2, 0) != 0) bad = 1;
    if (vtm_q4_mm_batch(NULL, q, scales, NULL, x, 1, 2, 2, 0) != -1) bad = 1;
    if (vtm_q4_mm_batch(y1, q, scales, NULL, x, -1, 2, 2, 0) != -1) bad = 1;
    if (vtm_q4_mm_batch(y1, q, scales, NULL, x, 0, 2, 2, 0) != 0) bad = 1;
    // N == 1 — делегирование одноколоночному ядру, результат тот же.
    if (vtm_q4_tmv(y2, q, scales, NULL, x, 1, 2, 0) != 0) bad = 1;
    if (vtm_q4_tmv_batch(y1, q, scales, NULL, x, 1, 2, 1, 0) != 0) bad = 1;
    if (memcmp(y1, y2, sizeof y1) != 0) bad = 1;
    printf("  %-34s %s\n", "Q4-батч аргументы/N==1", bad ? "FAIL" : "ok");
    if (bad) failures++;
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

// Плотный GEMV (n == 1, задача 0159): эталон — тот же double-цикл по t, что
// у скалярного op_tmv/op_mm (mm_f32_ref с n == 1). Проверяются обе раскладки
// (W и W^T) и хвосты cols, не кратные 4.
static void check_gemv_f32_case(int rows, int cols, int transposed) {
    uint32_t seed = 0x9e00u + (uint32_t)rows * 37u + (uint32_t)cols
                  + (uint32_t)transposed;
    size_t n = (size_t)rows * (size_t)cols;
    float *w = make_vec(n, seed);
    float *wt = transposed ? make_vec(n, seed ^ 0x2b2bu) : NULL;
    float *x = make_vec((size_t)cols, seed ^ 0x71u);
    float *y_ref = (float *)calloc((size_t)rows, sizeof(float));
    float *y_cand = (float *)calloc((size_t)rows, sizeof(float));
    const float *w_use = w;
    char label[128];
    double ref = 0.0;
    int r, t;

    if (!w || !x || !y_ref || !y_cand || (transposed && !wt)) {
        fprintf(stderr, "check_gemv_f32_case: нет памяти\n");
        failures++;
        goto done;
    }
    if (transposed) {
        for (r = 0; r < rows; r++)
            for (t = 0; t < cols; t++)
                wt[(size_t)t * (size_t)rows + (size_t)r]
                    = w[(size_t)r * (size_t)cols + (size_t)t];
        w_use = wt;
        // Эталон W^T: один double-цикл по t (как скалярный op_tmv).
        for (r = 0; r < rows; r++) {
            double sum = 0.0;
            for (t = 0; t < cols; t++)
                sum += (double)wt[(size_t)t * (size_t)rows + (size_t)r]
                     * (double)x[t];
            y_ref[r] = (float)sum;
        }
    } else {
        mm_f32_ref(y_ref, w, x, rows, cols, 1);
    }

    if (vtm_simd_gemv_f32(y_cand, w_use, x, rows, cols, transposed) != 0) {
        fprintf(stderr, "check_gemv_f32_case: ядро вернуло ошибку\n");
        failures++;
        goto done;
    }
    snprintf(label, sizeof label, "GEMV f32 %dx%d%s", rows, cols,
             transposed ? " (W^T)" : "");
    report(label, max_diff(y_cand, y_ref, (size_t)rows, &ref), ref, CHECK_TOL);
done:
    free(w);
    free(wt);
    free(x);
    free(y_ref);
    free(y_cand);
}

// --- Q4_K/Q6_K (kind 6/7, задачи 0165/0167) --------------------------------
//
// Синтетические блоки GGML: d/dmin — fp16 из малых значений, 6-битные
// scale/min, нибблы и ql/qh — случайные. Эталон — независимая деквантизация
// в double по формулам ADR 0018, поэтому проверка не повторяет код ядра
// (как q4_ref_value для kind 5).

static uint8_t *make_q4k(int n_values, uint32_t seed) {
    int64_t nblocks = ((int64_t)n_values + 255) / 256;
    size_t bytes = (size_t)nblocks * VTM_Q4K_BLOCK_BYTES;
    uint8_t *b = (uint8_t *)calloc(bytes ? bytes : 1u, 1);
    int64_t i;
    if (!b) return NULL;
    for (i = 0; i < nblocks; i++) {
        uint8_t *bp = b + (size_t)i * VTM_Q4K_BLOCK_BYTES;
        // d/dmin подобраны так, чтобы |w| было порядка 0,02, как у весов
        // Linear 3B Q4_K: d·sc·q <= 2e-5·63·15 ≈ 0,019. Абсолютный порог
        // CHECK_TOL тогда имеет тот же смысл, что на логитах модели.
        uint16_t d = vtm_float_to_half(1.8e-5f
                                       + (float)(rng_next(&seed) % 40u) * 1e-7f);
        uint16_t dm = vtm_float_to_half(8e-6f
                                        + (float)(rng_next(&seed) % 40u) * 1e-7f);
        int j;
        memcpy(bp, &d, sizeof d);
        memcpy(bp + 2, &dm, sizeof dm);
        for (j = 0; j < 12; j++) bp[4 + j] = (uint8_t)rng_next(&seed);
        for (j = 0; j < 128; j++) bp[16 + j] = (uint8_t)rng_next(&seed);
    }
    return b;
}

static uint8_t *make_q6k(int n_values, uint32_t seed) {
    int64_t nblocks = ((int64_t)n_values + 255) / 256;
    size_t bytes = (size_t)nblocks * VTM_Q6K_BLOCK_BYTES;
    uint8_t *b = (uint8_t *)calloc(bytes ? bytes : 1u, 1);
    int64_t i;
    if (!b) return NULL;
    for (i = 0; i < nblocks; i++) {
        uint8_t *bp = b + (size_t)i * VTM_Q6K_BLOCK_BYTES;
        // |w| = d·sc·(q−32) <= 1e-5·127·32 ≈ 0,04 — как у Q6_K 3B (ffn_down).
        uint16_t d = vtm_float_to_half(8e-6f
                                       + (float)(rng_next(&seed) % 40u) * 1e-7f);
        int j;
        for (j = 0; j < 128; j++) bp[j] = (uint8_t)rng_next(&seed);
        for (j = 0; j < 64; j++) bp[128 + j] = (uint8_t)rng_next(&seed);
        for (j = 0; j < 16; j++) bp[192 + j] = (uint8_t)rng_next(&seed);
        memcpy(bp + 208, &d, sizeof d);
    }
    return b;
}

// get_scale_min_k4 из ggml (независимая копия, как в эталоне ADR).
static void q4k_ref_scmin(const uint8_t *scales, unsigned j, unsigned *sc,
                          unsigned *mn) {
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

// Вес j плоского массива kind 6/7 в double (w = d·sc·q − dmin·m | d·sc·(q−32)).
static double qk_ref_value(const uint8_t *blocks, int64_t idx, int q6) {
    int64_t blk = idx >> 8;
    unsigned within = (unsigned)idx & 255u;
    if (!q6) {
        const uint8_t *bp = blocks + (size_t)blk * VTM_Q4K_BLOCK_BYTES;
        unsigned sub = within / 32u, off = within % 32u;
        uint16_t dh, dmh;
        unsigned sc, mn;
        double d, dmin, q;
        memcpy(&dh, bp, sizeof dh);
        memcpy(&dmh, bp + 2, sizeof dmh);
        d = (double)vtm_half_to_float(dh);
        dmin = (double)vtm_half_to_float(dmh);
        q4k_ref_scmin(bp + 4, sub, &sc, &mn);
        {
            unsigned byte = bp[16u + (size_t)(sub >> 1) * 32u + off];
            q = (double)((sub & 1u) ? (byte >> 4) : (byte & 0x0Fu));
        }
        return d * (double)sc * q - dmin * (double)mn;
    }
    {
        const uint8_t *bp = blocks + (size_t)blk * VTM_Q6K_BLOCK_BYTES;
        unsigned sub = within / 16u, off = within % 16u;
        unsigned chunk = sub >> 3, in = sub & 7u;
        unsigned quad = in >> 1, half = in & 1u;
        const uint8_t *ql = bp + (size_t)chunk * 64u
                          + (size_t)(quad & 1u) * 32u + (size_t)half * 16u;
        const uint8_t *qh = bp + 128u + (size_t)chunk * 32u
                          + (size_t)half * 16u;
        unsigned nib = (quad >= 2u) ? (unsigned)(ql[off] >> 4)
                                    : (unsigned)(ql[off] & 0x0Fu);
        unsigned q = nib | ((((unsigned)qh[off] >> (quad * 2u)) & 3u) << 4);
        double d = (double)vtm_half_to_float(
            (uint16_t)(bp[208] | ((uint16_t)bp[209] << 8)));
        int sc = (int)(int8_t)bp[192u + sub];
        return d * (double)sc * ((double)q - 32.0);
    }
}

static void qk_tmv_ref(float *y, const uint8_t *blocks, const float *x,
                       int rows, int cols, int q6) {
    int r, j;
    for (r = 0; r < rows; r++) {
        double acc = 0.0;
        for (j = 0; j < cols; j++)
            acc += qk_ref_value(blocks, (int64_t)r * cols + j, q6)
                 * (double)x[j];
        y[r] = acc;
    }
}

static void qk_mm_ref(float *y, const uint8_t *blocks, const float *b, int m,
                      int k, int n, int q6) {
    int r, c, t;
    for (r = 0; r < m; r++) {
        for (c = 0; c < n; c++) {
            double acc = 0.0;
            for (t = 0; t < k; t++)
                acc += qk_ref_value(blocks, (int64_t)r * k + t, q6)
                     * (double)b[(size_t)t * n + c];
            y[(size_t)r * n + c] = acc;
        }
    }
}

// Поштучное ядро kind 6/7 против независимой double-деквантизации.
static void check_q4k_tmv_case(int rows, int cols, int q6) {
    int n = rows * cols;
    uint32_t seed = 0x7a00u + (uint32_t)rows * 17u + (uint32_t)cols
                  + (uint32_t)q6 * 5u;
    uint8_t *blocks = q6 ? make_q6k(n, seed) : make_q4k(n, seed);
    float *x = make_vec((size_t)cols, seed ^ 0x2bu);
    float *y_ref = (float *)calloc((size_t)rows, sizeof(float));
    float *y = (float *)calloc((size_t)rows, sizeof(float));
    char label[112];
    double ref = 0.0;
    int rc;

    if (!blocks || !x || !y_ref || !y) {
        fprintf(stderr, "check_q4k_tmv_case: нет памяти\n");
        failures++;
        goto done;
    }
    qk_tmv_ref(y_ref, blocks, x, rows, cols, q6);
    {
        // f32-ядро 0167 проверяется с порогом 1e-5 отдельно от int8-пути
        // (0169): у того ошибка от квантования активаций, свой порог.
        const int saved = vtm_q4k_i8_set_mode(0);
        int pl;

        (void)saved;
        rc = q6 ? vtm_q6k_tmv(y, blocks, x, rows, cols)
                : vtm_q4k_tmv(y, blocks, x, rows, cols);
        if (rc != 0) {
            fprintf(stderr, "check_q4k_tmv_case: ядро вернуло ошибку\n");
            failures++;
            vtm_q4k_i8_set_mode(-1);
            goto done;
        }
        snprintf(label, sizeof label, "%s TMV %dx%d", q6 ? "Q6_K" : "Q4_K",
                 rows, cols);
        report(label, max_diff(y, y_ref, (size_t)rows, &ref), ref, CHECK_TOL);
        // int8-dot GEMV декода: плоскости 1..3. Эталон тот же (f64), порог —
        // CHECK_TOL_I8. Плоскость 1 в проверку не берётся: она для A/B.
        for (pl = 2; pl <= 3; pl++) {
            if (vtm_q4k_i8_set_mode(pl) != pl) break;   // нет int8-dot на CPU
            memset(y, 0, (size_t)rows * sizeof(float));
            rc = q6 ? vtm_q6k_tmv(y, blocks, x, rows, cols)
                    : vtm_q4k_tmv(y, blocks, x, rows, cols);
            if (rc != 0) {
                fprintf(stderr, "check_q4k_tmv_case: int8-ядро вернуло "
                                "ошибку\n");
                failures++;
                break;
            }
            snprintf(label, sizeof label, "%s TMV int8/%d %dx%d",
                     q6 ? "Q6_K" : "Q4_K", pl, rows, cols);
            report(label, max_diff(y, y_ref, (size_t)rows, &ref), ref,
                   CHECK_TOL_I8);
        }
        vtm_q4k_i8_set_mode(-1);
    }
done:
    free(blocks);
    free(x);
    free(y_ref);
    free(y);
}

static void check_q4k_mm_case(int m, int k, int n, int q6) {
    int n_el = m * k;
    uint32_t seed = 0x5c00u + (uint32_t)m * 13u + (uint32_t)k * 3u
                  + (uint32_t)n + (uint32_t)q6 * 7u;
    uint8_t *blocks = q6 ? make_q6k(n_el, seed) : make_q4k(n_el, seed);
    float *b = make_vec((size_t)k * (size_t)n, seed ^ 0x3du);
    float *y_ref = (float *)calloc((size_t)m * (size_t)n, sizeof(float));
    float *y = (float *)calloc((size_t)m * (size_t)n, sizeof(float));
    char label[112];
    double ref = 0.0;
    int rc;

    if (!blocks || !b || !y_ref || !y) {
        fprintf(stderr, "check_q4k_mm_case: нет памяти\n");
        failures++;
        goto done;
    }
    qk_mm_ref(y_ref, blocks, b, m, k, n, q6);
    rc = q6 ? vtm_q6k_mm(y, blocks, b, m, k, n)
            : vtm_q4k_mm(y, blocks, b, m, k, n);
    if (rc != 0) {
        fprintf(stderr, "check_q4k_mm_case: ядро вернуло ошибку\n");
        failures++;
        goto done;
    }
    snprintf(label, sizeof label, "%s MM %dx%dx%d", q6 ? "Q6_K" : "Q4_K",
             m, k, n);
    report(label, max_diff(y, y_ref, (size_t)m * (size_t)n, &ref), ref,
           CHECK_TOL);
done:
    free(blocks);
    free(b);
    free(y_ref);
    free(y);
}

// Батчевое ядро kind 6/7 (задача 0167) против поштучного пути — критерий
// приёмки: y[r*N+j] пачки = N вызовов поштучного ядра на contiguous вход
// токена j. Поштучное ядро уже сверено с double-эталоном выше.
// Ядро батча kind 6/7 — точное (побитовое) или A/B-вариант со своей точностью.
static int q4k_batch_parity_kernel(void) {
    const char *pi8 = getenv("VTM_Q4K_BATCH_I8");
    const char *ker = getenv("VTM_Q4K_BATCH_KERNEL");

    if (pi8 && *pi8) return 0;
    if (ker && strcmp(ker, "fast") == 0) return 0;
    return 1;
}

// При metal-бэкенде (FULL) эталонный поштучный путь идёт на GPU (иной порядок
// суммирования), а батчевый пока остаётся на CPU: побитового паритета тут нет,
// контракт 0192 — Δ ≤ 1e-2 (как у int8-пути). В hybrid поштучный путь — CPU,
// поэтому допуск обычный.
static double qk_batch_tol(void) {
    return vtm_metal_gemv_active() ? 2.0 * CHECK_TOL_I8 : CHECK_TOL;
}

static void check_q4k_batch_case(int rows, int cols, int ntok, int q6) {
    int n = rows * cols;
    uint32_t seed = 0x3e00u + (uint32_t)rows * 29u + (uint32_t)cols * 7u
                  + (uint32_t)ntok + (uint32_t)q6 * 3u;
    uint8_t *blocks = q6 ? make_q6k(n, seed) : make_q4k(n, seed);
    float *x = make_vec((size_t)cols * (size_t)ntok, seed ^ 0x71u);
    float *y_ref = (float *)calloc((size_t)rows * (size_t)ntok, sizeof(float));
    float *y_bat = (float *)calloc((size_t)rows * (size_t)ntok, sizeof(float));
    float *xt = (float *)malloc((size_t)cols * sizeof(float));
    float *yt = (float *)malloc((size_t)rows * sizeof(float));
    char label[128];
    double ref = 0.0;
    int j, r, t, rc, pl;

    if (!blocks || !x || !y_ref || !y_bat || !xt || !yt) {
        fprintf(stderr, "check_q4k_batch_case: нет памяти\n");
        failures++;
        goto done;
    }
    for (j = 0; j < ntok; j++) {
        for (t = 0; t < cols; t++)
            xt[t] = x[(size_t)t * (size_t)ntok + (size_t)j];
        rc = q6 ? vtm_q6k_tmv(yt, blocks, xt, rows, cols)
                : vtm_q4k_tmv(yt, blocks, xt, rows, cols);
        if (rc != 0) {
            fprintf(stderr, "check_q4k_batch_case: поштучное ядро — ошибка\n");
            failures++;
            goto done;
        }
        for (r = 0; r < rows; r++)
            y_ref[(size_t)r * (size_t)ntok + (size_t)j] = yt[r];
    }
    rc = q6 ? vtm_q6k_tmv_batch(y_bat, blocks, x, rows, cols, ntok)
            : vtm_q4k_tmv_batch(y_bat, blocks, x, rows, cols, ntok);
    if (rc != 0) {
        fprintf(stderr, "check_q4k_batch_case: батчевое ядро — ошибка\n");
        failures++;
        goto done;
    }
    snprintf(label, sizeof label, "%s TMV-батч %dx%dx%d",
             q6 ? "Q6_K" : "Q4_K", rows, cols, ntok);
    report(label, max_diff(y_bat, y_ref, (size_t)rows * (size_t)ntok, &ref),
           ref, qk_batch_tol());

    // 0189: батчевое int8-ядро обязано быть ПОБИТОВО равно поштучному при
    // любом числе плоскостей: оно зовёт ту же функцию строки, что декод,
    // меняется только сетка квантования активаций — и она общая у обоих путей.
    // Проверка имеет смысл только у точного ядра: f32-ядро 0167 и быстрое
    // int8-ядро сетку плоскостей не повторяют (VTM_Q4K_BATCH_I8=0 или
    // VTM_Q4K_BATCH_KERNEL=fast — это A/B-режимы со своей точностью).
    for (pl = 1; pl <= 2 && q4k_batch_parity_kernel(); pl++) {
        int k, i;

        if (vtm_q4k_i8_set_mode(pl) != pl) break;
        for (k = 0; k < ntok; k++) {
            for (t = 0; t < cols; t++)
                xt[t] = x[(size_t)t * (size_t)ntok + (size_t)k];
            rc = q6 ? vtm_q6k_tmv(yt, blocks, xt, rows, cols)
                    : vtm_q4k_tmv(yt, blocks, xt, rows, cols);
            if (rc != 0) break;
            for (i = 0; i < rows; i++)
                y_ref[(size_t)i * (size_t)ntok + (size_t)k] = yt[i];
        }
        rc = q6 ? vtm_q6k_tmv_batch(y_bat, blocks, x, rows, cols, ntok)
                : vtm_q4k_tmv_batch(y_bat, blocks, x, rows, cols, ntok);
        snprintf(label, sizeof label, "%s TMV-батч int8/%d %dx%dx%d",
                 q6 ? "Q6_K" : "Q4_K", pl, rows, cols, ntok);
        report(label, rc == 0
                   ? max_diff(y_bat, y_ref,
                              (size_t)rows * (size_t)ntok, &ref)
                   : 1e9,
               ref, qk_batch_tol());
    }
    vtm_q4k_i8_set_mode(-1);
done:
    free(blocks);
    free(x);
    free(y_ref);
    free(y_bat);
    free(xt);
    free(yt);
}

// GPU-батч prefill против CPU-батча (задача 0199).
//
// Длина пачки выбирает вариант ядра (8/16/32 токенов), но последний chunk
// может быть короче выбранного N: 11 -> gemm16, 24 -> gemm32, 44 -> 32+12.
// Именно на таких длинах GPU-путь до 0199 читал staging с неверным шагом
// строки (nt вместо N) и отдавал мусор всему prefill, а с ним и K/V префикса.
// Прежние проверки --check этого не ловили: у них длины были 8/16/32 (равны
// N) или < 8 (GEMM отказывал и работал CPU). Эталон — CPU-батчевое ядро;
// порядок суммирования у GPU другой, поэтому допуск — контракт 0192 (1e-2).
static void check_metal_gemm_case(int rows, int cols, int ntok, int q6) {
    uint32_t seed = 0x1900u + (uint32_t)rows * 31u + (uint32_t)cols * 3u
                  + (uint32_t)ntok * 7u + (uint32_t)q6;
    int n = rows * cols;
    uint8_t *blocks = q6 ? make_q6k(n, seed) : make_q4k(n, seed);
    float *x = make_vec((size_t)cols * (size_t)ntok, seed ^ 0x5au);
    float *y_ref = (float *)calloc((size_t)rows * (size_t)ntok, sizeof(float));
    float *y = (float *)calloc((size_t)rows * (size_t)ntok, sizeof(float));
    char label[128];
    double ref = 0.0;
    int rc;

    if (!blocks || !x || !y_ref || !y) {
        fprintf(stderr, "check_metal_gemm_case: нет памяти\n");
        failures++;
        goto done;
    }
    rc = q6 ? vtm_q6k_tmv_batch(y_ref, blocks, x, rows, cols, ntok)
            : vtm_q4k_tmv_batch(y_ref, blocks, x, rows, cols, ntok);
    if (rc != 0) {
        fprintf(stderr, "check_metal_gemm_case: CPU-батч — ошибка\n");
        failures++;
        goto done;
    }
    rc = vtm_metal_qk_gemm(q6 ? 1 : 0, y, blocks, NULL, x, rows, cols, ntok);
    if (rc != 0) {
        // Устройства нет или MSL не собрался: не ошибка CPU-набора, гибрид
        // в такой сборке всё равно ушёл бы на CPU.
        printf("  %-34s %s\n", q6 ? "Metal GEMM Q6_K" : "Metal GEMM Q4_K",
               "пропущен (Metal не готов)");
        goto done;
    }
    snprintf(label, sizeof label, "Metal GEMM %s %dx%dx%d",
             q6 ? "Q6_K" : "Q4_K", rows, cols, ntok);
    report(label, max_diff(y, y_ref, (size_t)rows * (size_t)ntok, &ref), ref,
           CHECK_TOL_I8);
done:
    free(blocks);
    free(x);
    free(y_ref);
    free(y);
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
        check_batch_f32_case(13, 11, 3);   // хвост 1-3 токена
        check_batch_f32_case(32, 17, 8);   // ровно две 8-ленты (AVX2)
    }

    // Плотный GEMV (n == 1, задача 0159): tied lm_head в декоде. Эталон —
    // double-цикл, как у скалярного пути; обе раскладки W.
    {
        static const int gshapes[][2] = {
            {1, 1}, {1, 20}, {7, 37}, {5, 1031}, {9, 13}, {64, 2048}
        };
        size_t gi;
        for (gi = 0; gi < sizeof(gshapes) / sizeof(gshapes[0]); gi++) {
            check_gemv_f32_case(gshapes[gi][0], gshapes[gi][1], 0);
            check_gemv_f32_case(gshapes[gi][0], gshapes[gi][1], 1);
        }
    }

    // int8-таблица словаря (kind 4, задача 0170): EMB-строка, TMV/GEMV
    // (режимы 0/2/3) и MM n > 1 против double-эталона формата. cols 128 и
    // кратные ему — SIMD-путь; 300 — отказ SIMD и переносимый путь.
    {
        static const int i8shapes[][3] = {
            {1, 128, 1}, {7, 256, 3}, {5, 1024, 2}, {3, 300, 1},
            {17, 128, 5}
        };
        size_t ei;
        for (ei = 0; ei < sizeof(i8shapes) / sizeof(i8shapes[0]); ei++)
            check_i8_case(i8shapes[ei][0], i8shapes[ei][1], i8shapes[ei][2]);
    }

    // Q4 (kind 5, задача 0152): независимый эталон — деквантизация в double.
    // Проверяем Q4_0 и Q4_1, хвосты групп (cols не кратны 32) и MM n>1.
    {
        static const int q4shapes[][2] = {
            {1, 1}, {1, 31}, {1, 32}, {1, 33}, {7, 37}, {5, 53},
            {17, 128}, {64, 160}, {128, 128}, {3, 1031}
        };
        static const int q4mmshapes[][3] = {
            {5, 7, 3}, {8, 16, 8}, {64, 128, 1}, {16, 32, 64}, {3, 1000, 17},
            /* n = 2..3, короткий prompt: у NEON отдельный двухколоночный
               путь (задача 0157), у скаляра — общий буфер строки. */
            {7, 96, 2}, {5, 37, 2}, {9, 64, 3}, {2, 1000, 2}
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

    // Q4-батч (0162): пачка против поштучного пути — критерий приёмки.
    // Формы: реальные (hidden 2048, intermediate 11008/2048), хвост 1-3
    // токена, cols не кратные 32 (группа через границу строк) и N == 1.
    {
        static const int q4bshapes[][3] = {
            {7, 37, 5}, {1, 32, 16}, {1, 33, 3}, {64, 96, 16},
            {128, 128, 2}, {128, 128, 3}, {2048, 2048, 16},
            {11008, 2048, 8}, {2048, 11008, 4}
        };
        static const int q4bmmshapes[][3] = {
            {5, 7, 3}, {16, 32, 16}, {2048, 2048, 16}, {11008, 2048, 8}
        };
        size_t bi;
        for (bi = 0; bi < sizeof(q4bshapes) / sizeof(q4bshapes[0]); bi++) {
            int els = q4bshapes[bi][0] * q4bshapes[bi][1];
            float qs = els > 65536 ? 0.02f : 1.0f;
            check_q4_batch_case(q4bshapes[bi][0], q4bshapes[bi][1],
                                q4bshapes[bi][2], 0, qs);
            check_q4_batch_case(q4bshapes[bi][0], q4bshapes[bi][1],
                                q4bshapes[bi][2], 1, qs);
        }
        for (bi = 0; bi < sizeof(q4bmmshapes) / sizeof(q4bmmshapes[0]); bi++) {
            check_q4_mm_batch_case(q4bmmshapes[bi][0], q4bmmshapes[bi][1],
                                   q4bmmshapes[bi][2], 0, 0.02f);
            check_q4_mm_batch_case(q4bmmshapes[bi][0], q4bmmshapes[bi][1],
                                   q4bmmshapes[bi][2], 1, 0.02f);
        }
        check_q4_batch_args();
    }

    // Q4_K/Q6_K (kind 6/7, задачи 0165/0167): поштучное ядро против
    // независимой double-деквантизации, батчевое — против поштучного пути.
    // cols 256/512/2048/11008 — реальные формы 3B; 320 и 200 — строка не
    // кратна блоку 256 (ядро отказывает, работает поштучный путь).
    {
        static const int qkshapes[][2] = {
            {1, 256}, {7, 256}, {5, 512}, {17, 320}, {64, 2048}, {3, 200}
        };
        static const int qkmshapes[][3] = {
            {5, 7, 3}, {8, 16, 8}, {3, 256, 4}, {16, 32, 16}
        };
        static const int qkbshapes[][3] = {
            {5, 512, 4}, {7, 256, 3}, {1, 256, 16}, {64, 2048, 16},
            {32, 320, 5}, {2048, 2048, 16}, {11008, 2048, 8}, {2048, 11008, 4}
        };
        size_t qi;
        for (qi = 0; qi < sizeof(qkshapes) / sizeof(qkshapes[0]); qi++) {
            check_q4k_tmv_case(qkshapes[qi][0], qkshapes[qi][1], 0);
            check_q4k_tmv_case(qkshapes[qi][0], qkshapes[qi][1], 1);
        }
        for (qi = 0; qi < sizeof(qkmshapes) / sizeof(qkmshapes[0]); qi++) {
            check_q4k_mm_case(qkmshapes[qi][0], qkmshapes[qi][1],
                              qkmshapes[qi][2], 0);
            check_q4k_mm_case(qkmshapes[qi][0], qkmshapes[qi][1],
                              qkmshapes[qi][2], 1);
        }
        for (qi = 0; qi < sizeof(qkbshapes) / sizeof(qkbshapes[0]); qi++) {
            check_q4k_batch_case(qkbshapes[qi][0], qkbshapes[qi][1],
                                 qkbshapes[qi][2], 0);
            check_q4k_batch_case(qkbshapes[qi][0], qkbshapes[qi][1],
                                 qkbshapes[qi][2], 1);
        }
    }

    // GPU-батч prefill (0199): длины chunk'а, не равные варианту ядра
    // (11 -> gemm16, 24/31 -> gemm32, 44 -> 32+12), плюс контроли 8/16/32.
    // Старый код давал здесь ошибку порядка самих сумм, а не допуска.
    if (vtm_metal_available()) {
        static const int mlens[] = {8, 11, 16, 24, 31, 32, 44};
        size_t mi;
        for (mi = 0; mi < sizeof(mlens) / sizeof(mlens[0]); mi++) {
            check_metal_gemm_case(64, 256, mlens[mi], 0);
            check_metal_gemm_case(64, 256, mlens[mi], 1);
        }
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

// --- плотный GEMV f32 n == 1 (задача 0159) ---------------------------------
//
// Форма tied lm_head: [rows, cols] × [cols, 1]. «Скаляр» — прежний путь для
// n == 1 (один double-цикл по t: у mm_f32 нет векторизации по столбцам),
// «simd» — vtm_simd_gemv_f32 (NEON/AVX2, переносимый откат). 3B Q4: emb.w —
// плотная [151936, 2048], это ~311М MAC и около трети времени декода; реальная
// форма задаётся --gemv-rows/--gemv-cols, по умолчанию меньше, чтобы бенч не
// занимал 1,24 ГБ.
static void bench_gemv_f32_table(int rows, int cols, int iters) {
    uint32_t seed = 0x6e6du + (uint32_t)rows + (uint32_t)cols;
    float *w = make_vec((size_t)rows * (size_t)cols, seed);
    float *x = make_vec((size_t)cols, seed ^ 0x33u);
    float *y_ref = (float *)calloc((size_t)rows, sizeof(float));
    float *y_cand = (float *)calloc((size_t)rows, sizeof(float));
    double ms_ref, ms_cand, flops = 2.0 * (double)rows * (double)cols;
    double t0, t1, ref = 0.0;
    int i;

    if (!w || !x || !y_ref || !y_cand) {
        fprintf(stderr, "bench_gemv_f32_table: нет памяти\n");
        goto done;
    }
    // «До»: ровно то, что делал op_mm при n == 1 без GEMV-ядра.
    t0 = now_ms();
    mm_f32_ref(y_ref, w, x, rows, cols, 1);
    t1 = now_ms();
    ms_ref = t1 - t0;

    if (vtm_simd_gemv_f32(y_cand, w, x, rows, cols, 0) != 0) {
        printf("GEMV f32 n==1 %dx%d: ядро вернуло ошибку\n", rows, cols);
        goto done;
    }
    t0 = now_ms();
    for (i = 0; i < iters; i++)
        vtm_simd_gemv_f32(y_cand, w, x, rows, cols, 0);
    t1 = now_ms();
    ms_cand = (t1 - t0) / (double)iters;
    printf("GEMV f32 n==1 %dx%d: scalar %.3f мс (%.2f GFLOP/s), simd "
           "%.3f мс (%.2f GFLOP/s), %.2fx, max|diff|=%.2e, max|ref|=%.2f\n",
           rows, cols, ms_ref, flops / (ms_ref * 1e6), ms_cand,
           flops / (ms_cand * 1e6), ms_ref / ms_cand,
           max_diff(y_cand, y_ref, (size_t)rows, &ref), ref);
done:
    free(w);
    free(x);
    free(y_ref);
    free(y_cand);
}

// Батчевый плотный TMV/MM (f32, задача 0159): N токенов против одной
// матрицы. Эталон — vtm_scalar_dense_batch (прежний double-цикл interp.c),
// кандидат — vtm_simd_mm_f32_batch (вектор по токенам пачки). N = 16 — полная
// пачка VTM_BATCH_TOKENS; форма — tied lm_head в prefill.
static void bench_dense_batch_table(int rows, int cols, int ntok, int iters) {
    uint32_t seed = 0xd3d3u + (uint32_t)rows + (uint32_t)cols
                  + (uint32_t)ntok;
    float *a = make_vec((size_t)rows * (size_t)cols, seed);
    float *b = make_vec((size_t)cols * (size_t)ntok, seed ^ 0x55u);
    float *y_ref = (float *)calloc((size_t)rows * (size_t)ntok, sizeof(float));
    float *y_cand = (float *)calloc((size_t)rows * (size_t)ntok,
                                    sizeof(float));
    double ms_ref, ms_cand, flops = 2.0 * (double)rows * (double)cols
                                          * (double)ntok;
    double t0, ref = 0.0;
    int i;

    if (!a || !b || !y_ref || !y_cand) {
        fprintf(stderr, "bench_dense_batch_table: нет памяти\n");
        goto done;
    }
    t0 = now_ms();
    if (vtm_scalar_dense_batch(y_ref, a, b, rows, cols, ntok, 0) != 0) {
        printf("батч f32 %dx%dxN%d: скаляр вернул ошибку\n", rows, cols, ntok);
        goto done;
    }
    ms_ref = now_ms() - t0;
    if (vtm_simd_mm_f32_batch(y_cand, a, b, rows, cols, ntok) != 0) {
        printf("батч f32 %dx%dxN%d: ядро вернуло ошибку\n", rows, cols, ntok);
        goto done;
    }
    t0 = now_ms();
    for (i = 0; i < iters; i++)
        vtm_simd_mm_f32_batch(y_cand, a, b, rows, cols, ntok);
    ms_cand = (now_ms() - t0) / (double)iters;
    printf("батч f32 %dx%dxN%d: scalar %.3f мс (%.2f GFLOP/s), simd "
           "%.3f мс (%.2f GFLOP/s), %.2fx, max|diff|=%.2e, max|ref|=%.2f\n",
           rows, cols, ntok, ms_ref, flops / (ms_ref * 1e6), ms_cand,
           flops / (ms_cand * 1e6), ms_ref / ms_cand,
           max_diff(y_cand, y_ref, (size_t)rows * (size_t)ntok, &ref), ref);
done:
    free(a);
    free(b);
    free(y_ref);
    free(y_cand);
}

// --- int8-таблица словаря: GEMV декода (задача 0170) ------------------------
//
// Таблица tied lm_head 3B — [151936, 2048]. В контейнере она int8 (kind 4,
// 0,31 ГБ, масштаб на блок 128); до 0170 загрузчик разворачивал её в fp32 в
// арене, и декод читал 1,24 ГБ на каждый токен (36,6 мс при 34 ГБ/с — потолок
// ~27 ток/с). Здесь меряется ядро по упакованной таблице:
//   * vtm_i8_gemv — рабочий GEMV (int8-dot на NEON, режим VTM_Q4_I8);
//   * режим 0 — активации точно в f32 (переносимый путь, точность-эталон
//     по активациям);
//   * --f32-table — тот же GEMV по fp32-копии таблицы (vtm_simd_gemv_f32,
//     прежний путь 0159): это и есть «до» по скорости, но требует ещё 1,24 ГиБ.
//
// Эталон точности — vtm_i8_tmv (double) на первых 64 строках: полный эталон
// 311М MAC в double на загруженной машине меряется минутами.
static void bench_i8_gemv_table(int rows, int cols, int iters, int with_f32) {
    size_t n = (size_t)rows * (size_t)cols;
    size_t nblk = n / (size_t)VTM_SCALE_BLOCK_128;
    int8_t *q = (int8_t *)malloc(n ? n : 1u);
    float *scales = (float *)malloc((nblk ? nblk : 1u) * sizeof(float));
    float *x = make_vec((size_t)cols, 0x1700u);
    float *y = (float *)calloc((size_t)rows, sizeof(float));
    float *y_ref = (float *)calloc((size_t)rows, sizeof(float));
    double flops = 2.0 * (double)rows * (double)cols;
    double gb = (double)n / 1e9;
    int ref_rows = rows > 64 ? 64 : rows;
    int i, m, rc;
    double t0, ms, ref = 0.0;

    if (!q || !scales || !x || !y || !y_ref) {
        fprintf(stderr, "bench_i8_gemv_table: нет памяти\n");
        goto done;
    }
    for (i = 0; i < (int)n; i++) {
        uint32_t s = (uint32_t)i * 2654435761u + 0x9e37u;
        q[i] = (int8_t)((int)((s >> 16) % 255u) - 127);
    }
    for (i = 0; i < (int)nblk; i++)
        scales[i] = 0.0005f + 0.0004f * (float)((i * 37u) % 100u) / 99.0f;

    printf("таблица int8 %dx%d: %.2f ГБ (fp32 была бы %.2f ГБ), масштаб /%d\n",
           rows, cols, gb, gb * 4.0, (int)VTM_SCALE_BLOCK_128);
    if (vtm_i8_tmv(y_ref, q, scales, x, ref_rows, cols) != 0) {
        printf("  эталон vtm_i8_tmv вернул ошибку\n");
        goto done;
    }
    for (m = 0; m < 3; m++) {
        static const int modes[3] = {0, 2, 3};
        int eff = vtm_q4k_i8_set_mode(modes[m]);
        double best = 1e300;

        rc = vtm_i8_gemv(y, q, scales, x, rows, cols);
        if (rc != 0) {
            printf("  режим %d: ядро вернуло ошибку\n", modes[m]);
            continue;
        }
        for (i = 0; i < iters; i++) {
            t0 = now_ms();
            if (vtm_i8_gemv(y, q, scales, x, rows, cols) != 0) break;
            ms = now_ms() - t0;
            if (ms < best) best = ms;
        }
        printf("  режим %d (%s): %7.2f мс  %6.2f ГБ/с  %6.2f GMAC/с  "
               "max|Δ|=%.2e (max|ref|=%.2f, эталон — double, %d строк)\n",
               eff, vtm_q4k_i8_name(), best, gb / (best * 1e-3),
               flops / (best * 1e6) / 2.0,
               max_diff(y, y_ref, (size_t)ref_rows, &ref), ref, ref_rows);
    }
    vtm_q4k_i8_set_mode(-1);
    if (with_f32) {
        // «До» (0159): fp32-копия таблицы и GEMV по ней. Память +4·n байт.
        float *wf = (float *)malloc(n * sizeof(float));
        double best = 1e300;

        if (!wf) {
            printf("  fp32-таблица: нет памяти (%.2f ГБ)\n", gb * 4.0);
            goto done;
        }
        for (i = 0; i < (int)n; i++)
            wf[i] = (float)q[i]
                  * scales[(size_t)i / (size_t)VTM_SCALE_BLOCK_128];
        for (i = 0; i < iters; i++) {
            t0 = now_ms();
            if (vtm_simd_gemv_f32(y_ref, wf, x, rows, cols, 0) != 0) break;
            ms = now_ms() - t0;
            if (ms < best) best = ms;
        }
        printf("  fp32-таблица (0159): %7.2f мс  %6.2f ГБ/с  %6.2f GMAC/с  "
               "max|Δ int8−fp32|=%.2e\n", best, (gb * 4.0) / (best * 1e-3),
               flops / (best * 1e6) / 2.0,
               max_diff(y, y_ref, (size_t)rows, &ref));
        free(wf);
    }
done:
    free(q);
    free(scales);
    free(x);
    free(y);
    free(y_ref);
}

static int run_i8_gemv_bench(int rows, int cols, int iters, int with_f32) {
    bench_i8_gemv_table(rows, cols, iters, with_f32);
    return 0;
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

// Плотный GEMV (n == 1) и батчевый плотный MM (0159): результат обязан быть
// побитово одинаковым при 1/2/4/8 потоках — строки делятся пулом, каждая
// считается тем же кодом.
static void thr_gemv_f32(int rows, int cols, int iters) {
    uint32_t seed = 0x9a9au + (uint32_t)rows + (uint32_t)cols;
    float *w = make_vec((size_t)rows * (size_t)cols, seed);
    float *x = make_vec((size_t)cols, seed ^ 0x42u);
    float *ref = (float *)calloc((size_t)rows, sizeof(float));
    float *got = (float *)calloc((size_t)rows, sizeof(float));
    char label[96];
    double base_ms = 0.0;
    int i;

    if (!w || !x || !ref || !got) {
        fprintf(stderr, "thr_gemv_f32: нет памяти\n");
        g_thr_fail++;
        goto done;
    }
    snprintf(label, sizeof label, "GEMV f32 %dx%d (n=1)", rows, cols);
    for (i = 0; i < 4; i++) {
        double t0, t1, ms;
        int it;
        vtm_threads_set(g_nts[i]);
        if (vtm_simd_gemv_f32(got, w, x, rows, cols, 0) != 0) {
            printf("  %-38s nt=%d ошибка ядра\n", label, g_nts[i]);
            g_thr_fail++;
            continue;
        }
        t0 = now_ms();
        for (it = 0; it < iters; it++)
            vtm_simd_gemv_f32(got, w, x, rows, cols, 0);
        t1 = now_ms();
        ms = (t1 - t0) / (double)iters;
        if (i == 0) {
            memcpy(ref, got, (size_t)rows * sizeof(float));
            base_ms = ms;
        }
        thr_report(label, g_nts[i], got, ref, (size_t)rows, ms, base_ms);
    }
done:
    free(w);
    free(x);
    free(ref);
    free(got);
}

static void thr_dense_batch_f32(int rows, int cols, int ntok, int iters) {
    uint32_t seed = 0x6b6bu + (uint32_t)rows + (uint32_t)cols;
    float *a = make_vec((size_t)rows * (size_t)cols, seed);
    float *b = make_vec((size_t)cols * (size_t)ntok, seed ^ 0x42u);
    size_t n = (size_t)rows * (size_t)ntok;
    float *ref = (float *)calloc(n, sizeof(float));
    float *got = (float *)calloc(n, sizeof(float));
    char label[96];
    double base_ms = 0.0;
    int i;

    if (!a || !b || !ref || !got) {
        fprintf(stderr, "thr_dense_batch_f32: нет памяти\n");
        g_thr_fail++;
        goto done;
    }
    snprintf(label, sizeof label, "батч f32 %dx%dxN%d", rows, cols, ntok);
    for (i = 0; i < 4; i++) {
        double t0, t1, ms;
        int it;
        vtm_threads_set(g_nts[i]);
        if (vtm_simd_mm_f32_batch(got, a, b, rows, cols, ntok) != 0) {
            printf("  %-38s nt=%d ошибка ядра\n", label, g_nts[i]);
            g_thr_fail++;
            continue;
        }
        t0 = now_ms();
        for (it = 0; it < iters; it++)
            vtm_simd_mm_f32_batch(got, a, b, rows, cols, ntok);
        t1 = now_ms();
        ms = (t1 - t0) / (double)iters;
        if (i == 0) {
            memcpy(ref, got, n * sizeof(float));
            base_ms = ms;
        }
        thr_report(label, g_nts[i], got, ref, n, ms, base_ms);
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
    printf("\nGEMV f32 2048x2048 (n == 1, путь tied lm_head):\n");
    thr_gemv_f32(2048, 2048, iters > 10 ? 10 : iters);
    printf("\nбатч f32 512x512xN16 (prefill lm_head):\n");
    thr_dense_batch_f32(512, 512, 16, iters > 10 ? 10 : iters);
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

// Q4-батч prefill (задача 0162): пачка N токенов против N поштучных вызовов
// ядра на той же матрице. Замер — только ядро (активации токенов разложены
// заранее), поэтому это верхняя оценка выигрыша prefill: в графе к нему
// добавляются дешёвые EMB/ROPE/ATTN/RMSN и lm_head.
static int run_q4_batch_bench(int rows, int cols, int ntok, int iters) {
    int n64 = rows * cols;
    int n = n64;
    uint8_t *q = NULL;
    float *scales = NULL, *mins = NULL, *x = NULL, *xtok = NULL;
    float *y_bat = NULL, *y_tok = NULL, *yt = NULL;
    double t_tok = 1e300, t_bat = 1e300, d, ref = 0.0;
    int it, j, r, t;

    failures = 0;
    if (n64 <= 0 || (int64_t)ntok * n64 > (int64_t)8000000000 || ntok < 1) {
        fprintf(stderr, "Q4-батч: слишком большая форма %dx%dx%d\n", rows,
                cols, ntok);
        return 2;
    }
    q = make_q4(n, 0x1620u);
    // Масштабы как у настоящей Q4_1-модели (0.005..0.035), Q4_1 (asym).
    scales = make_q4_scales(n, 0.0f, 0x77u);
    mins = make_q4_mins(n, 0x33u);
    x = make_vec((size_t)cols * (size_t)ntok, 0x5au);
    xtok = (float *)malloc((size_t)cols * (size_t)ntok * sizeof(float));
    y_bat = (float *)calloc((size_t)rows * (size_t)ntok, sizeof(float));
    y_tok = (float *)calloc((size_t)rows * (size_t)ntok, sizeof(float));
    yt = (float *)malloc((size_t)rows * sizeof(float));
    if (!q || !scales || !mins || !x || !xtok || !y_bat || !y_tok || !yt) {
        fprintf(stderr, "Q4-батч: нет памяти на форму %dx%dx%d\n", rows, cols,
                ntok);
        goto done;
    }
    // Раскладка x[t*ntok + j] -> contiguous вход токена j (поштучный путь).
    for (j = 0; j < ntok; j++)
        for (t = 0; t < cols; t++)
            xtok[(size_t)j * (size_t)cols + (size_t)t]
                = x[(size_t)t * (size_t)ntok + (size_t)j];

    for (it = 0; it < iters + 1; it++) {   // первый прогон — прогрев
        double t0 = now_ms(), dt;
        for (j = 0; j < ntok; j++) {
            if (vtm_q4_tmv(yt, q, scales, mins,
                           xtok + (size_t)j * (size_t)cols, rows, cols,
                           1) != 0) {
                fprintf(stderr, "Q4-батч: поштучное ядро вернуло ошибку\n");
                goto done;
            }
            if (it > 0)
                for (r = 0; r < rows; r++)
                    y_tok[(size_t)r * (size_t)ntok + (size_t)j] = yt[r];
        }
        dt = now_ms() - t0;
        if (it > 0 && dt < t_tok) t_tok = dt;
    }
    for (it = 0; it < iters + 1; it++) {
        double t0 = now_ms(), dt;
        if (vtm_q4_tmv_batch(y_bat, q, scales, mins, x, rows, cols, ntok,
                             1) != 0) {
            fprintf(stderr, "Q4-батч: батчевое ядро вернуло ошибку\n");
            goto done;
        }
        dt = now_ms() - t0;
        if (it > 0 && dt < t_bat) t_bat = dt;
    }
    d = max_diff(y_bat, y_tok, (size_t)rows * (size_t)ntok, &ref);

    printf("Q4_1 батч prefill: %dx%d x %d токен(ов), %d итераций (мин)\n",
           rows, cols, ntok, iters);
    printf("  int8-dot: %s (VTM_Q4_I8=%d; 0 — выключен, 1 — один проход, "
           "2 — два)\n", vtm_q4_i8_name(), vtm_q4_i8_mode());
    printf("  поштучно: %9.2f мс  (%7.2f мс/токен, %6.2f GFLOP/s)\n",
           t_tok, t_tok / (double)ntok,
           2.0 * (double)n64 * (double)ntok / (t_tok * 1e6));
    printf("  пачкой:   %9.2f мс  (%7.2f мс/токен, %6.2f GFLOP/s)  "
           "ускорение %.2fx\n", t_bat, t_bat / (double)ntok,
           2.0 * (double)n64 * (double)ntok / (t_bat * 1e6), t_tok / t_bat);
    {
        int ns[] = {64, 256, 512};
        size_t k;
        printf("  проекция prefill (эта матрица, пачки по %d):\n", ntok);
        for (k = 0; k < sizeof ns / sizeof ns[0]; k++) {
            double nchunk = (double)((ns[k] + ntok - 1) / ntok);
            double before = t_tok * (double)ns[k] / (double)ntok;
            double after = t_bat * nchunk;
            printf("    N=%3d: до %9.2f мс, после %9.2f мс, %.2fx\n", ns[k],
                   before, after, before / after);
        }
    }
    {
        // Порог — по активному пути: у int8 (VTM_Q4_I8=1|2) это ошибка
        // квантования активаций, а не арифметики (см. CHECK_TOL_I8).
        const double tol = vtm_q4_i8_mode() ? CHECK_TOL_I8 : CHECK_TOL;
        printf("  точность пачки против поштучного пути: max|diff|=%.3e "
               "(max|ref|=%.3e) %s (порог %.0e, int8-dot %s)\n", d, ref,
               d <= tol ? "ok" : "FAIL", tol, vtm_q4_i8_name());
        if (d > tol) failures++;
    }
done:
    free(q);
    free(scales);
    free(mins);
    free(x);
    free(xtok);
    free(y_bat);
    free(y_tok);
    free(yt);
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

// --- Q4_K/Q6_K: поштучное против батчевого ядра (задача 0167) --------------
//
// Прямой замер на синтетических блоках: N вызовов поштучного ядра (вес
// читается N раз) против батчевого (строка деквантизуется один раз на пачку).
// G MAC/s = rows·cols·ntok / t. Форма cols кратна 256 — работает NEON-ядро;
// иначе оно отказывает и считается скалярный откат (медленнее, но верно).

// --- проекция декода по формам модели (задача 0171) ------------------------
//
// Абсолютный tok/s декода этой машиной померить нельзя: токен читает ~1,7 ГБ
// блоков Q4_K/Q6_K и 0,31 ГБ int8-таблицы, и при свопе/чужой нагрузке время
// уходит в page fault (в сессии 0171 — 1,7-2,8 с/токен при 2,2 ГБ pageins на
// токен, тогда как ядро на тех же формах из кэша даёт 50-80 мс). Поэтому
// проба меряет ядра на РЕАЛЬНЫХ формах тензоров модели (блоки синтетические,
// из памяти) и складывает время токена по составу модели:
//
//   t_токена = Σ_формы(6/7) count·t(форма) + t(kind 4: упакованная таблица)
//
// Это не заменяет замер на живой модели, но даёт воспроизводимую величину,
// по которой видно эффект правок ядер (парный запуск двух бинарей в одном
// окружении: до/после). Разбивка по формам печатается — видно, что дороже.
typedef struct {
    int kind;           // VTM_TENSOR_Q4K / VTM_TENSOR_Q6K
    int rows, cols;
    int count;
    double ms;          // время одного тензора формы
} qk_shape;

static int qk_proj_group(qk_shape *g, int n, int kind, int rows, int cols) {
    int i;

    for (i = 0; i < n; i++)
        if (g[i].kind == kind && g[i].rows == rows && g[i].cols == cols) {
            g[i].count++;
            return n;
        }
    if (n >= 64) return n;
    g[n].kind = kind;
    g[n].rows = rows;
    g[n].cols = cols;
    g[n].count = 1;
    g[n].ms = 0.0;
    return n + 1;
}

static int run_qk_proj(const char *explicit_path, int iters) {
    const char *path = find_model(explicit_path);
    qk_shape shapes[64];
    int nshapes = 0, i, it;
    int i8_rows = 0, i8_cols = 0;
    vtm_t *m;
    double total_ms = 0.0, i8_ms = 0.0;

    if (!path) {
        fprintf(stderr, "модель не найдена (--qk-proj PATH)\n");
        return 2;
    }
    if (iters < 1) iters = 5;
    m = vtm_load(path);
    if (!m) {
        fprintf(stderr, "vtm_load(%s): %s\n", path, vtm_last_error());
        return 1;
    }
    {
        uint32_t t, nt = vtm_tensor_count(m);

        for (t = 0; t < nt; t++) {
            const vtm_tensor *tt = vtm_tensor_by_index(m, t);

            if (!tt) continue;
            if (tt->kind == VTM_TENSOR_Q4K || tt->kind == VTM_TENSOR_Q6K) {
                if (tt->ndim != 2) continue;
                nshapes = qk_proj_group(shapes, nshapes, tt->kind,
                                        (int)tt->dims[0], (int)tt->dims[1]);
            } else if (tt->kind == VTM_TENSOR_INT8 && tt->ndim == 2
                       && (int)tt->dims[0] > i8_rows) {
                i8_rows = (int)tt->dims[0];
                i8_cols = (int)tt->dims[1];
            }
        }
    }
    vtm_free(m);   // 2 ГБ модели не держим: блоки всё равно синтетические

    printf("проекция декода %s\n", path);
    printf("  режим int8: %d плоскостей (%s), потоков %d\n",
           vtm_q4k_i8_mode(), vtm_q4k_i8_name(), vtm_threads());
    printf("  %-5s %5s %6s %6s %8s %10s %9s %6s\n", "kind", "rows", "cols",
           "шт", "G MAC", "мс/шт", "мс/токен", "ток/с");

    for (i = 0; i < nshapes; i++) {
        const int kind = shapes[i].kind, rows = shapes[i].rows;
        const int cols = shapes[i].cols;
        const uint8_t *blocks;
        float *x, *y;
        double best = 1e300;

        if (rows <= 0 || cols <= 0 || cols % 256 != 0) continue;
        blocks = kind == VTM_TENSOR_Q6K
            ? make_q6k(rows * cols, 0x1710u) : make_q4k(rows * cols, 0x1711u);
        x = make_vec((size_t)cols, 0x1712u);
        y = (float *)calloc((size_t)rows, sizeof(float));
        if (!blocks || !x || !y) {
            printf("  %s: нет памяти на форму %dx%d\n",
                   kind == VTM_TENSOR_Q6K ? "Q6_K" : "Q4_K", rows, cols);
            free((void *)blocks);
            free(x);
            free(y);
            continue;
        }
        for (it = 0; it < iters; it++) {
            double t0 = now_ms(), dt;
            int rc = kind == VTM_TENSOR_Q6K
                ? vtm_q6k_tmv(y, blocks, x, rows, cols)
                : vtm_q4k_tmv(y, blocks, x, rows, cols);

            dt = now_ms() - t0;
            if (rc != 0) { best = 1e300; break; }
            if (dt < best) best = dt;
        }
        shapes[i].ms = best;
        total_ms += best * (double)shapes[i].count;
        printf("  %-5s %5d %6d %6d %8.2f %10.3f %9.2f %6.1f\n",
               kind == VTM_TENSOR_Q6K ? "Q6_K" : "Q4_K", rows, cols,
               shapes[i].count,
               (double)rows * (double)cols * (double)shapes[i].count / 1e9,
               best, best * (double)shapes[i].count,
               1000.0 / (best > 0.0 ? best : 1.0));
        free((void *)blocks);
        free(x);
        free(y);
    }

    if (i8_rows > 0 && i8_cols > 0) {
        // Голова/эмбеддинги (kind 4, упакованная int8-таблица): 0,31 ГБ на
        // токен, ядро упирается в полосу памяти, а не в инструкции.
        int8_t *q = (int8_t *)malloc((size_t)i8_rows * (size_t)i8_cols);
        float *scales = (float *)malloc(
            ((size_t)i8_rows * (size_t)i8_cols / VTM_SCALE_BLOCK_128 + 1)
            * sizeof(float));
        float *x = make_vec((size_t)i8_cols, 0x1713u);
        float *y = (float *)calloc((size_t)i8_rows, sizeof(float));

        if (q && scales && x && y) {
            size_t n = (size_t)i8_rows * (size_t)i8_cols;
            size_t nblk = n / VTM_SCALE_BLOCK_128;
            size_t k;
            double best = 1e300;

            for (k = 0; k < n; k++)
                q[k] = (int8_t)((int)((k * 2654435761u >> 16) % 255u) - 127);
            for (k = 0; k < nblk; k++) scales[k] = 0.0007f;
            if (vtm_i8_gemv(y, q, scales, x, i8_rows, i8_cols) == 0) {
                for (it = 0; it < (iters < 3 ? 3 : iters); it++) {
                    double t0 = now_ms(), dt;

                    if (vtm_i8_gemv(y, q, scales, x, i8_rows, i8_cols) != 0)
                        break;
                    dt = now_ms() - t0;
                    if (dt < best) best = dt;
                }
            }
            if (best < 1e299) {
                i8_ms = best;
                total_ms += best;
                printf("  %-5s %5d %6d %6d %8.2f %10.3f %9.2f %6.1f\n",
                       "int8", i8_rows, i8_cols, 1,
                       (double)n / 1e9, best, best, 1000.0 / best);
            } else {
                printf("  int8 %dx%d: ядро вернуло ошибку\n", i8_rows, i8_cols);
            }
        } else {
            printf("  int8 %dx%d: нет памяти на таблицу (%.2f ГБ)\n",
                   i8_rows, i8_cols,
                   (double)i8_rows * (double)i8_cols / 1e9);
        }
        free(q);
        free(scales);
        free(x);
        free(y);
    }

    if (total_ms > 0.0)
        printf("итого проекция: %.2f мс/токен = %.1f ток/с "
               "(ядра kind 6/7 %.2f мс + голова %.2f мс)\n",
               total_ms, 1000.0 / total_ms, total_ms - i8_ms, i8_ms);
    return 0;
}

// --- int8-dot декод kind 6/7: скорость и точность по плоскостям (0169) -----
//
// Одна и та же матрица и один и тот же вектор активаций считаются f32-ядром
// (плоскостей 0) и int8-ядром с 1/2/3 плоскостями. Время — min из iters,
// ошибка — max|Δ| против f32-ядра (это и есть цена квантования активаций:
// веса представимы в int8 точно). Форма cols должна быть кратна 256, иначе
// int8-ядро отказывает и в колонке int8 будет то же время, что у f32.
static int run_q4k_i8_bench(int rows, int cols, int iters) {
    int n = rows * cols;
    uint8_t *blocks[2];
    float *x = NULL, *y_ref = NULL, *y = NULL;
    double mac = (double)rows * (double)cols;
    int kind, pl, i;

    if (rows <= 0 || cols <= 0 || n <= 0 || n > 0x3fffffff || iters <= 0) {
        fprintf(stderr, "q4k-i8: плохая форма %dx%d\n", rows, cols);
        return 2;
    }
    if (cols % 256 != 0) {
        fprintf(stderr, "q4k-i8: cols должно быть кратно 256 (сейчас %d): "
                        "иначе int8-ядро отказывает\n", cols);
        return 2;
    }
    if (!vtm_q4_i8_hw()) {
        printf("q4k-i8: int8-dot на этом CPU/в этой сборке недоступен\n");
        return 0;
    }
    blocks[0] = make_q4k(n, 0x1690u);
    blocks[1] = make_q6k(n, 0x1691u);
    x = make_vec((size_t)cols, 0x1692u);
    y_ref = (float *)calloc((size_t)rows, sizeof(float));
    y = (float *)calloc((size_t)rows, sizeof(float));
    if (!blocks[0] || !blocks[1] || !x || !y_ref || !y) {
        fprintf(stderr, "q4k-i8: нет памяти на форму %dx%d\n", rows, cols);
        goto done;
    }
    printf("int8-dot декод: %dx%d (%.1f МБ блоков), %d итераций\n", rows, cols,
           (double)n * 0.5625 / 1e6, iters);
    for (kind = 0; kind < 2; kind++) {
        const char *nm = kind ? "Q6_K" : "Q4_K";
        double t_f32 = 1e300, ref_abs = 0.0;

        vtm_q4k_i8_set_mode(0);
        for (i = 0; i < iters; i++) {
            double t0 = now_ms(), dt;

            if (kind) vtm_q6k_tmv(y_ref, blocks[kind], x, rows, cols);
            else vtm_q4k_tmv(y_ref, blocks[kind], x, rows, cols);
            dt = now_ms() - t0;
            if (dt < t_f32) t_f32 = dt;
        }
        for (i = 0; i < rows; i++)
            if (fabs((double)y_ref[i]) > ref_abs) ref_abs = fabs((double)y_ref[i]);
        printf("  %s: f32 %7.2f мс (%5.1f G MAC/s, 100%%)", nm, t_f32,
               mac / t_f32 / 1e6);
        for (pl = 1; pl <= 3; pl++) {
            double t = 1e300, d = 0.0, rel;

            vtm_q4k_i8_set_mode(pl);
            if (vtm_q4k_i8_mode() != pl) continue;
            for (i = 0; i < iters; i++) {
                double t0 = now_ms(), dt;

                if (kind) vtm_q6k_tmv(y, blocks[kind], x, rows, cols);
                else vtm_q4k_tmv(y, blocks[kind], x, rows, cols);
                dt = now_ms() - t0;
                if (dt < t) t = dt;
            }
            for (i = 0; i < rows; i++) {
                const double dd = fabs((double)y[i] - (double)y_ref[i]);
                if (dd > d) d = dd;
            }
            rel = ref_abs > 0.0 ? d / ref_abs : 0.0;
            printf(" | int8/%d %6.2f мс (%5.1f G, %.2fx) Δ=%.1e (%.1e отн.)",
                   pl, t, mac / t / 1e6, t_f32 / t, d, rel);
            if (rel > 1e-2) failures++;
        }
        printf("\n");
        vtm_q4k_i8_set_mode(-1);
    }
done:
    free(blocks[0]);
    free(blocks[1]);
    free(x);
    free(y_ref);
    free(y);
    return 0;
}

static int run_q4k_bench(int rows, int cols, int iters, int ntok) {
    int n = rows * cols;
    failures = 0;
    uint8_t *blocks[2];
    float *x = NULL, *y_ref = NULL, *y_bat = NULL, *xt = NULL, *yt = NULL;
    int kind, i, j, t;
    double mac = (double)rows * (double)cols * (double)ntok;

    if (n <= 0 || n > 0x3fffffff || ntok < 1 || ntok > 128) {
        fprintf(stderr, "q4k: плохая форма %dx%d/%d токенов\n", rows, cols,
                ntok);
        return 2;
    }
    blocks[0] = make_q4k(n, 0x5150u);
    blocks[1] = make_q6k(n, 0x6150u);
    x = make_vec((size_t)cols * (size_t)ntok, 0x123u);
    y_ref = (float *)calloc((size_t)rows * (size_t)ntok, sizeof(float));
    y_bat = (float *)calloc((size_t)rows * (size_t)ntok, sizeof(float));
    xt = (float *)malloc((size_t)cols * sizeof(float));
    yt = (float *)malloc((size_t)rows * sizeof(float));
    if (!blocks[0] || !blocks[1] || !x || !y_ref || !y_bat || !xt || !yt) {
        fprintf(stderr, "q4k: нет памяти на форму %dx%d\n", rows, cols);
        goto done;
    }
    printf("Q4_K/Q6_K: %dx%d, %d токенов, %d итераций (порог точности %.0e)\n",
           rows, cols, ntok, iters, CHECK_TOL);
    for (kind = 0; kind < 2; kind++) {
        const char *nm = kind ? "Q6_K" : "Q4_K";
        double t_tok = 1e300, t_bat = 1e300;
        double maxd = 0.0, maxr = 0.0;
        int rc;

        for (j = 0; j < ntok; j++) {
            for (t = 0; t < cols; t++)
                xt[t] = x[(size_t)t * (size_t)ntok + (size_t)j];
            rc = kind ? vtm_q6k_tmv(yt, blocks[kind], xt, rows, cols)
                      : vtm_q4k_tmv(yt, blocks[kind], xt, rows, cols);
            if (rc != 0) {
                fprintf(stderr, "q4k: поштучное ядро вернуло ошибку\n");
                goto done;
            }
            for (i = 0; i < rows; i++)
                y_ref[(size_t)i * (size_t)ntok + (size_t)j] = yt[i];
        }
        for (i = 0; i < iters; i++) {
            double t0 = now_ms(), dt;
            for (j = 0; j < ntok; j++) {
                for (t = 0; t < cols; t++)
                    xt[t] = x[(size_t)t * (size_t)ntok + (size_t)j];
                if (kind)
                    vtm_q6k_tmv(yt, blocks[kind], xt, rows, cols);
                else
                    vtm_q4k_tmv(yt, blocks[kind], xt, rows, cols);
            }
            dt = now_ms() - t0;
            if (dt < t_tok) t_tok = dt;
        }
        rc = kind ? vtm_q6k_tmv_batch(y_bat, blocks[kind], x, rows, cols, ntok)
                  : vtm_q4k_tmv_batch(y_bat, blocks[kind], x, rows, cols, ntok);
        if (rc != 0) {
            fprintf(stderr, "q4k: батчевое ядро вернуло ошибку\n");
            goto done;
        }
        for (i = 0; i < iters; i++) {
            double t0 = now_ms(), dt;
            if (kind)
                vtm_q6k_tmv_batch(y_bat, blocks[kind], x, rows, cols, ntok);
            else
                vtm_q4k_tmv_batch(y_bat, blocks[kind], x, rows, cols, ntok);
            dt = now_ms() - t0;
            if (dt < t_bat) t_bat = dt;
        }
        maxd = max_diff(y_bat, y_ref, (size_t)rows * (size_t)ntok, &maxr);
        printf("  %s: поштучно %8.2f мс (%5.1f G MAC/s) | пачкой %8.2f мс "
               "(%5.1f G MAC/s) | %4.2fx | max|batch-token|=%.2e %s\n",
               nm, t_tok, mac / t_tok / 1e6, t_bat, mac / t_bat / 1e6,
               t_tok / t_bat, maxd, maxd <= (double)CHECK_TOL ? "ok" : "FAIL");
        if (maxd > (double)CHECK_TOL) failures++;
    }
done:
    free(blocks[0]);
    free(blocks[1]);
    free(x);
    free(y_ref);
    free(y_bat);
    free(xt);
    free(yt);
    return failures ? 1 : 0;
}

// --- prefill-проверка на настоящей модели (задача 0162) --------------------
//
// Сквозной критерий приёмки: состояние после одной пачки vtm_run_batch должно
// совпасть с состоянием после N последовательных vtm_run. Сравниваются логиты
// последнего токена (max|diff|) и argmax. В графе 3B Q4 участвуют и Q4
// TMV/MM (батчевое ядро 0162), и плотный lm_head (ядро 0159), поэтому это
// проверка всей проводки, а не одного ядра.
static float *copy_logits(vtm_t *m, int *n_out, int *argmax) {
    const vtm_tensor *lg = vtm_tensor_by_name(m, "logits");
    float *out;
    int i;

    if (!lg || !lg->f32 || lg->n_elements == 0) return NULL;
    out = (float *)malloc((size_t)lg->n_elements * sizeof(float));
    if (!out) return NULL;
    memcpy(out, lg->f32, (size_t)lg->n_elements * sizeof(float));
    *n_out = (int)lg->n_elements;
    *argmax = 0;
    for (i = 1; i < *n_out; i++)
        if (out[i] > out[*argmax]) *argmax = i;
    return out;
}

// --- Metal-бэкенд: точность и скорость simd / metal / hybrid (0192+0198) ---
//
// Один процесс, одна загрузка модели; все числа — парные прогоны в одном окне,
// порядок бэкендов чередуется между повторами (rep 0: simd,metal,hybrid;
// rep 1: hybrid,metal,simd), берётся min из 2. hybrid (0198) — батчевые GEMM
// prefill на GPU, поштучные GEMV декода на NEON.
//
// Точность: три промпта по 5 токенов и 64-токенная пачка против simd —
// argmax и max|Δlogits| ≤ 1e-2 (контракт 0192/0198). В конце — откат:
// VTM_METAL_FORCE_FAIL=1 теряет первый GEMM hybrid, после этого режим Metal
// выключается, и пачка обязана дать логиты чистого CPU-пути (Δ = 0 к simd).
//
// Скорость: декод (мс/токен, ток/с), prefill 64 и 128 (ток/с, TTFT, CPU-время),
// память (RSS по точкам, zero-copy/копии/currentAllocatedSize).
static const char *g_metal_lens;  // --metal-lens L1[,L2...] (0198)

static int metal_step(vtm_t *m, int token, int pos, int *out) {
    *out = -1;
    vtm_set_position(m, pos);
    vtm_set_token(m, token);
    return vtm_run(m, emit_first, out) == VTM_OK ? 0 : -1;
}

static float *metal_bench_logits(vtm_t *m, const vtm_backend *b,
                                 const int *ids, int n, int *n_out,
                                 int *argmax) {
    int i, out = -1;

    vtm_set_backend(m, b);
    vtm_reset(m);
    for (i = 0; i < n; i++)
        if (metal_step(m, ids[i], i, &out) != 0) return NULL;
    return copy_logits(m, n_out, argmax);
}

static int metal_bench_decode(vtm_t *m, const vtm_backend *b,
                              const int *prompt, int np, int ngen,
                              double *prefill_ms, double *ms_per_tok,
                              double *steps) {
    int i, pos, tok = -1, next = -1;
    double t0, t1;

    vtm_set_backend(m, b);
    vtm_reset(m);
    t0 = now_ms();
    for (i = 0; i < np - 1; i++)
        if (metal_step(m, prompt[i], i, &next) != 0) return -1;
    t1 = now_ms();
    *prefill_ms = t1 - t0;
    if (metal_step(m, prompt[np - 1], np - 1, &tok) != 0) return -1;
    pos = np;
    t0 = now_ms();
    for (i = 0; i < ngen; i++) {
        double s0 = now_ms();
        if (metal_step(m, tok, pos, &next) != 0) return -1;
        if (steps) steps[i] = now_ms() - s0;
        tok = next;
        pos++;
    }
    t1 = now_ms();
    *ms_per_tok = (t1 - t0) / (double)ngen;
    return 0;
}

static int run_metal_bench(const char *explicit_path, int ngen) {
    static const int prompts[3][5] = {
        {1, 2, 3, 4, 5},
        {100, 200, 300, 400, 500},
        {1000, 2000, 3000, 4000, 5000}
    };
    static const int lens_def[2] = {64, 128};
    int lens[8];
    int nlens = 2;
    // Порядок повторов: чередование, чтобы дрейф загрузки машины не лег на
    // один бэкенд (эта машина занята, см. 0196/0198).
    static const int order[2][3] = {{0, 1, 2}, {2, 1, 0}};
    const char *path = find_model(explicit_path);
    const vtm_backend *simd;
    const vtm_backend *metal;
    const vtm_backend *hybrid;
    const vtm_backend *bs[3];
    vtm_metal_stats st;
    vtm_t *m;
    size_t rss0, rss_load, rss_pref[3] = {0, 0, 0};
    int mode_seen[3] = {VTM_METAL_OFF, VTM_METAL_OFF, VTM_METAL_OFF};
    int pi, i, li, rc = 0;

    if (!path) {
        fprintf(stderr, "модель не найдена (--metal PATH)\n");
        return 2;
    }
    // Проверка до by_name: тот на сборке без Metal печатает своё
    // предупреждение и отдаёт simd, и замер молча измерил бы CPU как metal.
    if (!vtm_metal_compiled() || !vtm_metal_available()) {
        printf("metal/hybrid: Metal недоступен в этой сборке/на этой машине — "
               "откат проверен, замер пропущен\n");
        return 0;
    }
    simd = vtm_backend_by_name("simd");
    metal = vtm_backend_by_name("metal");
    hybrid = vtm_backend_by_name("hybrid");
    if (!simd || !metal || !metal->name
        || strcmp(metal->name, "metal") != 0
        || !hybrid || !hybrid->name
        || strcmp(hybrid->name, "hybrid") != 0) {
        printf("metal/hybrid: бэкенд недоступен в этой сборке — откат "
               "проверен, замер пропущен\n");
        return 0;
    }
    if (ngen < 1) ngen = 1;
    bs[0] = simd;
    bs[1] = metal;
    bs[2] = hybrid;
    printf("модель: %s\n", path);
    rss0 = rss_bytes();
    m = vtm_load(path);
    if (!m) {
        fprintf(stderr, "vtm_load(%s): %s\n", path, vtm_last_error());
        return 1;
    }
    rss_load = rss_bytes();
    lens[0] = lens_def[0];
    lens[1] = lens_def[1];
    if (g_metal_lens && *g_metal_lens) {
        const char *p = g_metal_lens;
        int k = 0;
        while (*p && k < 8) {
            int v = atoi(p);
            if (v > 0 && v <= 256) lens[k++] = v;
            while (*p && *p != ',') p++;
            if (*p == ',') p++;
        }
        if (k > 0) nlens = k;
    }
    printf("точность (3 промпта по 5 токенов, эталон simd):\n");
    for (pi = 0; pi < 3; pi++) {
        float *ref = NULL;
        int n_ref = 0, tok_ref = -1;

        ref = metal_bench_logits(m, simd, prompts[pi], 5, &n_ref, &tok_ref);
        if (!ref) {
            fprintf(stderr, "metal: эталонные логиты не получены (промпт "
                    "%d)\n", pi + 1);
            rc = 1;
            continue;
        }
        for (i = 1; i < 3; i++) {
            float *got = NULL;
            int n_got = 0, tok_got = -1;
            double d, refmax = 0.0;

            got = metal_bench_logits(m, bs[i], prompts[pi], 5, &n_got,
                                     &tok_got);
            if (!got || n_got != n_ref) {
                fprintf(stderr, "metal: логиты не получены (%s, промпт %d)\n",
                        bs[i]->name, pi + 1);
                free(got);
                rc = 1;
                continue;
            }
            d = max_diff(got, ref, (size_t)n_ref, &refmax);
            printf("  промпт %d: %-9s max|Δlogits|=%.3e (max|ref|=%.2f), "
                   "argmax %d/%d %s\n", pi + 1, bs[i]->name, d, refmax,
                   tok_ref, tok_got,
                   tok_ref == tok_got ? "(совпал)" : "(РАЗОШЁЛСЯ)");
            if (tok_ref != tok_got || d > 1e-2) rc = 1;
            free(got);
        }
        free(ref);
    }

    printf("декод (префикс 5, %d токенов, min из 2, порядок чередуется):\n",
           ngen);
    {
        static const int dlens[3] = {16, 32, 64};
        double best[3] = {1e300, 1e300, 1e300};
        double pre[3] = {1e300, 1e300, 1e300};
        double *steps[3] = {NULL, NULL, NULL};
        double pref_sum[3][3];
        int have[3] = {0, 0, 0};
        int rep, k, li2;

        for (i = 0; i < 3; i++)
            for (li2 = 0; li2 < 3; li2++) pref_sum[i][li2] = 1e300;
        for (i = 0; i < 3; i++) steps[i] = (double *)calloc((size_t)ngen,
                                                            sizeof(double));
        for (rep = 0; rep < 2; rep++) {
            for (k = 0; k < 3; k++) {
                int bi = order[rep][k];
                double prefill = 0.0, ms = 0.0;
                if (metal_bench_decode(m, bs[bi], prompts[0], 5, ngen,
                                       &prefill, &ms, steps[bi]) != 0) {
                    fprintf(stderr, "metal: декод %s — ошибка\n",
                            bs[bi]->name);
                    rc = 1;
                    continue;
                }
                have[bi] = 1;
                if (prefill < pre[bi]) pre[bi] = prefill;
                if (ms < best[bi]) best[bi] = ms;
                // Сумма первых L шагов — скорость на префиксе L. В одном
                // прогоне с парным чередованием бэкендов (0204).
                for (li2 = 0; li2 < 3; li2++) {
                    int L = dlens[li2], jj;
                    double sum = 0.0;
                    if (L > ngen) continue;
                    for (jj = 0; jj < L; jj++) sum += steps[bi][jj];
                    if (sum > 0.0 && sum < pref_sum[bi][li2])
                        pref_sum[bi][li2] = sum;
                }
            }
        }
        for (i = 0; i < 3; i++) {
            if (best[i] < 1e299)
                printf("  %-10s prefill %.1f мс, декод %.2f мс/токен = %.1f "
                       "ток/с\n", bs[i]->name, pre[i], best[i],
                       1000.0 / best[i]);
        }
        if (best[0] < 1e299) {
            for (i = 1; i < 3; i++)
                if (best[i] < 1e299)
                    printf("  %s/simd: %.2fx\n", bs[i]->name,
                           best[0] / best[i]);
        }
        for (li2 = 0; li2 < 3; li2++) {
            int L = dlens[li2], j;

            if (L > ngen) continue;
            for (j = 0; j < 3; j++)
                if (pref_sum[j][li2] < 1e299)
                    break;
            if (j == 3) continue;
            printf("  декод %2d: ", L);
            for (j = 0; j < 3; j++) {
                if (pref_sum[j][li2] >= 1e299) continue;
                printf(" %-6s %.1f ток/с", bs[j]->name,
                       (double)L * 1000.0 / pref_sum[j][li2]);
            }
            if (pref_sum[0][li2] < 1e299 && pref_sum[1][li2] < 1e299)
                printf(" | metal/simd %.2fx",
                       pref_sum[0][li2] / pref_sum[1][li2]);
            printf("\n");
        }
        (void)have;
        for (i = 0; i < 3; i++) free(steps[i]);
    }

    // Prefill пачкой: все токены промпта за один vtm_run_batch (внутри —
    // чанки по VTM_BATCH_TOKENS=32). Парный замер в одном процессе, min из 2:
    // simd / metal / hybrid на одних и тех же страницах весов.
    for (li = 0; li < nlens; li++) {
        int len = lens[li];
        int32_t ids[256];
        double best[3] = {1e300, 1e300, 1e300};
        double cpu_best[3] = {-1.0, -1.0, -1.0};
        float *lg[3] = {NULL, NULL, NULL};
        int nlg[3] = {0, 0, 0}, arg[3] = {0, 0, 0};
        int rep, k;

        for (i = 0; i < len; i++) ids[i] = 1000 + (i * 37) % 150000;
        for (rep = 0; rep < 2; rep++) {
            for (k = 0; k < 3; k++) {
                int bi = order[rep][k];
                double t0, c0, ms, cpu;

                vtm_set_backend(m, bs[bi]);
                vtm_reset(m);
                vtm_set_position(m, 0);
                c0 = now_cpu_ms();
                t0 = now_ms();
                if (vtm_run_batch(m, ids, len) != VTM_OK) {
                    fprintf(stderr, "prefill: %s — ошибка\n", bs[bi]->name);
                    rc = 1;
                    continue;
                }
                ms = now_ms() - t0;
                cpu = c0 >= 0.0 ? now_cpu_ms() - c0 : -1.0;
                if (ms < best[bi]) best[bi] = ms;
                if (cpu >= 0.0 && (cpu_best[bi] < 0.0 || cpu < cpu_best[bi]))
                    cpu_best[bi] = cpu;
                if (rep == 0) {
                    // RSS/логиты — после первого прогона этого бэкенда:
                    // порядок simd->metal->hybrid, поэтому GPU-аллокации
                    // видны на metal и уже не растут на hybrid (zero-copy).
                    rss_pref[bi] = rss_bytes();
                    mode_seen[bi] = vtm_metal_mode();
                    lg[bi] = copy_logits(m, &nlg[bi], &arg[bi]);
                }
            }
        }
        printf("prefill %d токенов (vtm_run_batch, min из 2, порядок "
               "чередуется):\n", len);
        for (i = 0; i < 3; i++) {
            if (best[i] >= 1e299) continue;
            printf("  %-10s %8.1f мс = %6.1f ток/с (TTFT %.2f с), CPU %.2f с, "
                   "режим %s\n", bs[i]->name, best[i],
                   (double)len / (best[i] * 1e-3), best[i] * 1e-3,
                   cpu_best[i] >= 0.0 ? cpu_best[i] * 1e-3 : 0.0,
                   mode_seen[i] == VTM_METAL_FULL ? "full"
                       : mode_seen[i] == VTM_METAL_BATCH ? "batch" : "off");
        }
        if (best[0] < 1e299) {
            for (i = 1; i < 3; i++)
                if (best[i] < 1e299)
                    printf("  ускорение prefill %d: %s/simd %.2fx\n", len,
                           bs[i]->name, best[0] / best[i]);
        }
        // Точность пачки: батчевый GEMM GPU против батчевого CPU-ядра simd.
        if (len == 64 && lg[0]) {
            for (i = 1; i < 3; i++) {
                double d, refmax = 0.0;
                if (!lg[i] || nlg[i] != nlg[0]) continue;
                d = max_diff(lg[i], lg[0], (size_t)nlg[0], &refmax);
                printf("  точность пачки: %-9s max|Δlogits|=%.3e, argmax "
                       "%d/%d %s\n", bs[i]->name, d, arg[0], arg[i],
                       arg[0] == arg[i] ? "(совпал)" : "(РАЗОШЁЛСЯ)");
                if (d > 1e-2 || arg[0] != arg[i]) rc = 1;
            }
        }
        for (i = 0; i < 3; i++) free(lg[i]);
    }

    vtm_metal_stats_get(&st);
    printf("metal stats: init %llu мс (MSL %llu мс), вызовов %llu, отказов "
           "%llu, gemm %llu, отложено %llu, ожиданий %llu, "
           "граф %llu токенов/%llu ops (отказов %llu)\n",
           (unsigned long long)st.init_ms,
           (unsigned long long)st.compile_ms,
           (unsigned long long)st.calls, (unsigned long long)st.failures,
           (unsigned long long)st.gemm_calls,
           (unsigned long long)st.defer_ops,
           (unsigned long long)st.defer_waits,
           (unsigned long long)st.graph_tokens,
           (unsigned long long)st.graph_ops,
           (unsigned long long)st.graph_fail);
    printf("  граф: кодирование CPU %.1f мс на %llu токенов = %.2f мс/токен "
           "(GPU ждёт только кодирование)\n",
           (double)st.graph_encode_ns / 1e6,
           (unsigned long long)st.graph_tokens,
           st.graph_tokens
               ? (double)st.graph_encode_ns / 1e6 / (double)st.graph_tokens
               : 0.0);
    printf("  zero-copy: %.1f МиБ привязано без копии, копий %.1f МиБ; "
           "currentAllocatedSize %.1f МиБ (до init %.1f, после %.1f), "
           "ожидание GPU %.1f мс\n",
           (double)st.nocopy_bytes / (1024.0 * 1024.0),
           (double)st.copy_bytes / (1024.0 * 1024.0),
           (double)st.current_allocated / (1024.0 * 1024.0),
           (double)st.alloc_before / (1024.0 * 1024.0),
           (double)st.alloc_after / (1024.0 * 1024.0),
           (double)st.wait_ns / 1e6);
    printf("память (RSS = phys_footprint): до загрузки %.1f МиБ, после "
           "загрузки %.1f МиБ\n",
           (double)rss0 / (1024.0 * 1024.0),
           (double)rss_load / (1024.0 * 1024.0));
    for (i = 0; i < 3; i++) {
        if (!rss_pref[i]) continue;
        printf("  после prefill %d (%s): %.1f МиБ\n", lens[0], bs[i]->name,
               (double)rss_pref[i] / (1024.0 * 1024.0));
    }
    if (st.last_error[0])
        printf("  последняя ошибка Metal: %s\n", st.last_error);

    // Откат (0198): первая же операция Metal теряет результат — hybrid обязан
    // уйти на чистый CPU. Мы намеренно ставим ошибку перед прогоном пачки:
    // ни один GEMM не считается, значит логиты обязаны совпасть с simd
    // побитово (Δ = 0), а режим Metal — выключиться.
    printf("откат (VTM_METAL_FORCE_FAIL=1, hybrid, пачка 16):\n");
    {
        int32_t ids[16];
        float *ref = NULL, *got = NULL;
        int n_ref = 0, n_got = 0, tok_ref = -1, tok_got = -1;
        double d, refmax = 0.0;

        for (i = 0; i < 16; i++) ids[i] = 7000 + i * 11;
        vtm_set_backend(m, simd);
        vtm_reset(m);
        vtm_set_position(m, 0);
        if (vtm_run_batch(m, ids, 16) == VTM_OK)
            ref = copy_logits(m, &n_ref, &tok_ref);
        setenv("VTM_METAL_FORCE_FAIL", "1", 1);
        vtm_set_backend(m, hybrid);
        vtm_reset(m);
        vtm_set_position(m, 0);
        if (vtm_run_batch(m, ids, 16) == VTM_OK)
            got = copy_logits(m, &n_got, &tok_got);
        unsetenv("VTM_METAL_FORCE_FAIL");
        vtm_metal_stats_get(&st);
        printf("  режим после ошибки: %s, отказов Metal %llu\n",
               vtm_metal_mode() == VTM_METAL_OFF ? "off (чистый CPU)" : "?",
               (unsigned long long)st.failures);
        if (!ref || !got || n_ref != n_got) {
            fprintf(stderr, "откат: логиты не получены\n");
            rc = 1;
        } else {
            d = max_diff(got, ref, (size_t)n_ref, &refmax);
            printf("  hybrid после ошибки: max|Δlogits|=%.3e, argmax %d/%d "
                   "%s\n", d, tok_ref, tok_got,
                   tok_ref == tok_got ? "(совпал)" : "(РАЗОШЁЛСЯ)");
            if (tok_ref != tok_got || d > 1e-2) rc = 1;
        }
        free(ref);
        free(got);
    }

    // Откат графа токена (0204): инъекция отказа в конец КАЖДОГО графа —
    // первый токен теряет результаты, interp.c перезапускает его CPU-путём и
    // выключает Metal; остальные токены обязаны совпасть с simd побитово.
    printf("откат графа (VTM_METAL_FORCE_FAIL=1, metal FULL, 5 токенов):\n");
    {
        static const int ids[5] = {1, 2, 3, 4, 5};
        int n_ref = 0, n_got = 0, tok_ref = -1, tok_got = -1;
        float *ref, *got;
        double d, refmax = 0.0;

        ref = metal_bench_logits(m, simd, ids, 5, &n_ref, &tok_ref);
        setenv("VTM_METAL_FORCE_FAIL_GRAPH", "1", 1);
        got = metal_bench_logits(m, metal, ids, 5, &n_got, &tok_got);
        unsetenv("VTM_METAL_FORCE_FAIL_GRAPH");
        vtm_metal_stats_get(&st);
        printf("  режим после ошибки: %s, отказов Metal %llu, графов "
               "потеряно %llu\n",
               vtm_metal_mode() == VTM_METAL_OFF ? "off (чистый CPU)" : "?",
               (unsigned long long)st.failures,
               (unsigned long long)st.graph_fail);
        if (!ref || !got || n_ref != n_got) {
            fprintf(stderr, "откат графа: логиты не получены\n");
            rc = 1;
        } else {
            d = max_diff(got, ref, (size_t)n_ref, &refmax);
            printf("  metal после ошибки: max|Δlogits|=%.3e, argmax %d/%d "
                   "%s\n", d, tok_ref, tok_got,
                   tok_ref == tok_got ? "(совпал)" : "(РАЗОШЁЛСЯ)");
            if (tok_ref != tok_got || d > 1e-2) rc = 1;
        }
        free(ref);
        free(got);
    }

    vtm_free(m);
    return rc;
}

static int run_prefill_check(const char *explicit_path, int ntok) {
    const char *path = find_model(explicit_path);
    int32_t *ids = NULL;
    float *y_ref = NULL, *y_bat = NULL;
    int n_ref = 0, n_bat = 0, tok_ref = -1, tok_bat = -1;
    vtm_t *m;
    vtm_status st;
    double t0, t_tok, t_bat, d, ref = 0.0;
    int i;

    if (!path) {
        fprintf(stderr, "модель не найдена (--prefill-check PATH)\n");
        return 2;
    }
    if (ntok < 2) ntok = 2;
    if (ntok > 64) ntok = 64;
    printf("модель: %s\n", path);
    printf("prefill-check: %d токенов, эталон — по токену, затем пачка\n",
           ntok);
    printf("  int8-dot: %s (VTM_Q4_I8=%d)\n", vtm_q4_i8_name(),
           vtm_q4_i8_mode());
    m = vtm_load(path);
    if (!m) {
        fprintf(stderr, "vtm_load(%s): %s\n", path, vtm_last_error());
        return 1;
    }
    vtm_set_backend(m, &vtm_backend_simd);
    ids = (int32_t *)malloc((size_t)ntok * sizeof(int32_t));
    if (!ids) {
        vtm_free(m);
        return 1;
    }
    for (i = 0; i < ntok; i++) ids[i] = 1000 + i * 7;

    t0 = now_ms();
    for (i = 0; i < ntok; i++) {
        vtm_set_position(m, i);
        vtm_set_token(m, ids[i]);
        st = vtm_run(m, NULL, NULL);
        if (st != VTM_OK) {
            fprintf(stderr, "vtm_run: %s (%s)\n", vtm_strerror(st),
                    vtm_error_message(m));
            goto fail;
        }
    }
    t_tok = now_ms() - t0;
    y_ref = copy_logits(m, &n_ref, &tok_ref);
    if (!y_ref) {
        fprintf(stderr, "в модели нет тензора logits\n");
        goto fail;
    }

    // Чистое состояние, как у хоста между запросами (api.c): KV-кэш
    // перезаписывается теми же токенами с тех же позиций.
    vtm_reset(m);
    vtm_set_position(m, 0);
    t0 = now_ms();
    st = vtm_run_batch(m, ids, ntok);
    t_bat = now_ms() - t0;
    if (st != VTM_OK) {
        fprintf(stderr, "vtm_run_batch: %s (%s)\n", vtm_strerror(st),
                vtm_error_message(m));
        goto fail;
    }
    y_bat = copy_logits(m, &n_bat, &tok_bat);
    if (!y_bat) {
        fprintf(stderr, "в модели нет тензора logits после пачки\n");
        goto fail;
    }
    if (n_ref != n_bat) {
        fprintf(stderr, "разное число логитов: %d vs %d\n", n_ref, n_bat);
        goto fail;
    }
    d = max_diff(y_bat, y_ref, (size_t)n_ref, &ref);
    printf("  по токену: %9.2f мс (%.2f мс/токен)\n", t_tok,
           t_tok / (double)ntok);
    printf("  пачкой:    %9.2f мс (%.2f мс/токен), ускорение %.2fx\n", t_bat,
           t_bat / (double)ntok, t_tok / t_bat);
    printf("  max|logits(пачка) - logits(по токену)| = %.3e (max|ref|=%.2f) "
           "%s\n", d, ref, d <= CHECK_TOL ? "ok" : "FAIL");
    printf("  argmax: по токену=%d пачка=%d %s\n", tok_ref, tok_bat,
           tok_ref == tok_bat ? "(совпал)" : "(РАЗОШЁЛСЯ)");
    free(ids);
    free(y_ref);
    free(y_bat);
    vtm_free(m);
    if (d > CHECK_TOL || tok_ref != tok_bat) {
        printf("prefill-check: FAIL (> %.0e или argmax разошёлся)\n",
               CHECK_TOL);
        return 1;
    }
    printf("prefill-check: ok (в пределах %.0e)\n", CHECK_TOL);
    return 0;
fail:
    free(ids);
    free(y_ref);
    free(y_bat);
    vtm_free(m);
    return 1;
}

// --- декод kind 6/7: int8-dot против f32 (задача 0169) ---------------------
//
// Один процесс, одна загрузка модели: 3B Q4_K в памяти — 3,2 ГиБ RSS, и
// повторная загрузка на каждый прогон съедает больше, чем сам замер. Поэтому
// оба пути (f32 и int8) считаются подряд на одних и тех же промптах, а
// сравнение идёт по СЫРЫМ логитам каждого шага greedy-декода: это ровно те
// числа, которых требуют критерии 0169 — argmax, top-5 и Δlogits.
//
// Промпты — файл, строка = id через запятую (или пробел). Строки берутся
// подряд, не больше --decode-limit.

typedef struct {
    int *toks;
    int n;
    int cap;
} tok_trace;

static int dec_emit(void *user, int token, int position) {
    tok_trace *t = (tok_trace *)user;
    (void)position;
    if (t->n < t->cap) t->toks[t->n++] = token;
    return 0;
}

// Пять самых больших логитов (id по убыванию значения). n_logits не меньше 5.
static void top5_of(const float *v, int n, int out[5]) {
    int i, k;

    for (k = 0; k < 5; k++) out[k] = -1;
    for (i = 1; i < n; i++) {
        int j, pos = -1;
        for (j = 0; j < 5; j++)
            if (out[j] < 0 || v[i] > v[out[j]]) { pos = j; break; }
        if (pos < 0) continue;
        for (j = 4; j > pos; j--) out[j] = out[j - 1];
        out[pos] = i;
    }
}

static int argmax_of(const float *v, int n) {
    int i, best = 0;

    for (i = 1; i < n; i++)
        if (v[i] > v[best]) best = i;
    return best;
}

// Greedy-декод промпта в текущем режиме int8: как в main.c — все токены
// промпта, кроме последнего, только наполняют KV-кэш; последний даёт первый
// сгенерированный, дальше каждый следующий выводится из предыдущего.
// Сырые логиты после каждого шага кладутся в logits_out (max_new × n_logits).
// Возвращает 0, -1 — ошибка прогона; *t_run_ms — время прогона.
static int decode_run(vtm_t *m, const int32_t *ids, int n_ids, int max_new,
                      float *logits_out, int *n_logits, tok_trace *tr,
                      double *t_run_ms) {
    const vtm_tensor *lg;
    double t0;
    int step, i;

    if (n_ids < 1 || max_new < 1) return -1;
    vtm_reset(m);
    tr->n = 0;
    *n_logits = 0;
    t0 = now_ms();
    for (i = 0; i + 1 < n_ids; i++) {
        vtm_set_position(m, i);
        vtm_set_token(m, ids[i]);
        if (vtm_run(m, NULL, NULL) != VTM_OK) return -1;
    }
    for (step = 0; step < max_new; step++) {
        const int32_t feed = (step == 0) ? ids[n_ids - 1]
                                         : (int32_t)tr->toks[tr->n - 1];

        vtm_set_position(m, n_ids - 1 + step);
        vtm_set_token(m, feed);
        if (vtm_run(m, dec_emit, tr) != VTM_OK) return -1;
        lg = vtm_tensor_by_name(m, "logits");
        if (!lg || !lg->f32 || lg->n_elements == 0) return -1;
        if (*n_logits == 0) *n_logits = (int)lg->n_elements;
        else if (*n_logits != (int)lg->n_elements) return -1;
        memcpy(logits_out + (size_t)step * (size_t)(*n_logits), lg->f32,
               (size_t)(*n_logits) * sizeof(float));
        if (tr->n <= step) return -1;   // SAMP не сработал
    }
    *t_run_ms = now_ms() - t0;
    return 0;
}

static int read_prompt_file(const char *path, int32_t ***out, int *n_out,
                            int **lens_out) {
    FILE *f = fopen(path, "rb");
    int32_t **ids = NULL;
    int *lens = NULL;
    int n = 0, cap = 0;
    char line[8192];

    if (!f) {
        fprintf(stderr, "не открыть промпты: %s\n", path);
        return -1;
    }
    while (fgets(line, sizeof line, f)) {
        int32_t *v = NULL;
        int k = 0, c = 0;
        char *p = line;

        while (*p) {
            long id;
            char *end = NULL;
            if (*p < '0' || *p > '9') { p++; continue; }
            id = strtol(p, &end, 10);
            if (end == p) { p++; continue; }
            if (k == c) {
                int nc = c ? c * 2 : 16;
                int32_t *nv = (int32_t *)realloc(v, (size_t)nc * sizeof *nv);
                if (!nv) { free(v); fclose(f); goto fail; }
                v = nv;
                c = nc;
            }
            v[k++] = (int32_t)id;
            p = end;
        }
        if (k == 0) { free(v); continue; }
        if (n == cap) {
            int nc = cap ? cap * 2 : 8;
            int32_t **ni = (int32_t **)realloc(ids, (size_t)nc * sizeof *ni);
            int *nl = (int *)realloc(lens, (size_t)nc * sizeof *nl);
            if (!ni || !nl) { free(ni ? ni : ids); free(nl ? nl : lens);
                              free(v); fclose(f); goto fail; }
            ids = ni;
            lens = nl;
            cap = nc;
        }
        ids[n] = v;
        lens[n] = k;
        n++;
    }
    fclose(f);
    *out = ids;
    *n_out = n;
    *lens_out = lens;
    return 0;
fail:
    while (n-- > 0) free(ids[n]);
    free(ids);
    free(lens);
    return -1;
}

// Сравнение двух прогонов одного промпта: Δlogits, argmax, top-5 и первый шаг
// расхождения последовательностей.
typedef struct {
    int steps_cmp;          // шагов, где префиксы совпали
    int argmax_ok;          // из них argmax совпал
    double top5_hits;       // сумма |top5 ∩ top5| по шагам
    double max_dlogit;      // max|Δlogits| по шагам
    double max_ref;         // max|logits| эталона (для относительной ошибки)
    int diverge_step;       // шаг, на котором токены разошлись (-1 — нет)
    double t_ref, t_new;    // время прогона, мс
} dec_cmp;

static void decode_compare(const float *l_ref, const float *l_new, int n_logits,
                           const tok_trace *tr_ref, const tok_trace *tr_new,
                           int max_new, dec_cmp *c) {
    int step;

    memset(c, 0, sizeof *c);
    c->diverge_step = -1;
    for (step = 0; step < max_new; step++) {
        const float *a = l_ref + (size_t)step * (size_t)n_logits;
        const float *b = l_new + (size_t)step * (size_t)n_logits;
        int am_a = argmax_of(a, n_logits), am_b = argmax_of(b, n_logits);
        int t5a[5], t5b[5], i, j, hits = 0;
        double d = 0.0, ref = 0.0;

        for (i = 0; i < n_logits; i++) {
            const double dd = fabs((double)b[i] - (double)a[i]);
            if (dd > d) d = dd;
            if (fabs((double)a[i]) > ref) ref = fabs((double)a[i]);
        }
        if (d > c->max_dlogit) c->max_dlogit = d;
        if (ref > c->max_ref) c->max_ref = ref;
        top5_of(a, n_logits, t5a);
        top5_of(b, n_logits, t5b);
        for (i = 0; i < 5; i++)
            for (j = 0; j < 5; j++)
                if (t5a[i] == t5b[j]) { hits++; break; }
        c->steps_cmp++;
        if (am_a == am_b) c->argmax_ok++;
        c->top5_hits += (double)hits / 5.0;
        if (c->diverge_step < 0
            && (tr_ref->toks[step] != tr_new->toks[step]))
            c->diverge_step = step;
    }
}

// --- декод: tok/s f32 против int8 (задача 0169) -----------------------------
//
// Парный замер в одном процессе: один и тот же промпт и число новых токенов,
// пути чередуются (f32 → int8 в одном повторе, int8 → f32 в следующем), берётся
// min по повторам. На общей машине одиночные прогоны расходятся в разы, а
// page cache нагревается первым прогоном, поэтому без чередования сравнение
// бессмысленно.
static int run_decode_bench(const char *explicit_path, const char *prompts_path,
                            int ntok, int reps) {
    const char *path = find_model(explicit_path);
    const char *pp = prompts_path ? prompts_path : NULL;
    int32_t **ids = NULL;
    int *lens = NULL;
    int n_prompts = 0, np;
    tok_trace tr;
    float *lg = NULL;
    int n_logits = 0, n_logits2 = 0;
    vtm_t *m;
    double t0_best = 1e300, t3_best = 1e300, mb0 = 0.0, mb3 = 0.0;
    int i, rep;

    if (!path) {
        fprintf(stderr, "модель не найдена (--decode-bench PATH)\n");
        return 2;
    }
    if (!pp) {
        fprintf(stderr, "--decode-bench требует --prompts FILE\n");
        return 2;
    }
    if (read_prompt_file(pp, &ids, &n_prompts, &lens) != 0) return 2;
    if (n_prompts == 0) return 2;
    if (ntok < 1) ntok = 1;
    if (reps < 1) reps = 1;
    m = vtm_load(path);
    if (!m) {
        fprintf(stderr, "vtm_load(%s): %s\n", path, vtm_last_error());
        return 1;
    }
    vtm_set_backend(m, &vtm_backend_simd);
    {
        const vtm_tensor *lgt = vtm_tensor_by_name(m, "logits");

        if (!lgt || !lgt->f32 || lgt->n_elements == 0) {
            fprintf(stderr, "в модели нет тензора logits\n");
            vtm_free(m);
            return 1;
        }
        n_logits = (int)lgt->n_elements;
    }
    lg = (float *)malloc((size_t)ntok * (size_t)n_logits * sizeof(float));
    tr.toks = (int *)malloc((size_t)ntok * sizeof(int));
    tr.cap = ntok;
    if (!lg || !tr.toks) {
        fprintf(stderr, "decode-bench: нет памяти\n");
        free(lg);
        free(tr.toks);
        vtm_free(m);
        return 1;
    }
    printf("модель: %s\ndecode-bench: %d токенов, %d повторов, чередование путей "
           "(int8-плоскостей %d, %s)\n", path, ntok, reps,
           vtm_q4k_i8_mode(), vtm_q4k_i8_name());
    for (rep = 0; rep < reps; rep++) {
        double t;
        int order;

        for (order = 0; order < 2; order++) {
            const int mode = (order == (rep & 1)) ? 0 : vtm_q4k_i8_mode();
            const int eff = vtm_q4k_i8_set_mode(mode);
            double ms;

            (void)eff;
            // Промпты идут по кругу: на каждый повтор — свой, чтобы кэш не
            // решал за нас (все они короткие и одинаковой формы).
            np = rep % n_prompts;
            n_logits2 = 0;
            if (decode_run(m, ids[np], lens[np], ntok, lg, &n_logits2, &tr,
                           &ms) != 0) {
                fprintf(stderr, "decode-bench: прогон не удался\n");
                goto fail;
            }

            t = ms / (double)ntok;
            if (order == (rep & 1)) {
                if (t < t0_best) { t0_best = t; mb0 = ms; }
            } else if (t < t3_best) {
                t3_best = t;
                mb3 = ms;
            }
            printf("  повтор %d: %s %8.2f мс/токен (%6.1f ток/с)\n", rep + 1,
                   order == (rep & 1) ? "f32  " : "int8 ", t,
                   1000.0 / (t > 0.0 ? t : 1.0));
        }
    }
    vtm_q4k_i8_set_mode(-1);
    printf("итого (min): f32 %.2f мс/токен = %.1f ток/с | int8 %.2f мс/токен "
           "= %.1f ток/с | ускорение %.2fx\n", t0_best, 1000.0 / t0_best,
           t3_best, 1000.0 / t3_best, t0_best / (t3_best > 0.0 ? t3_best : 1.0));
    (void)mb0;
    (void)mb3;
    free(lg);
    free(tr.toks);
    for (i = 0; i < n_prompts; i++) free(ids[i]);
    free(ids);
    free(lens);
    vtm_free(m);
    return 0;
fail:
    free(lg);
    free(tr.toks);
    vtm_free(m);
    return 1;
}

static int run_decode_check(const char *explicit_path, const char *prompts_path,
                            int max_new, int planes, int limit) {
    const char *path = find_model(explicit_path);
    int32_t **ids = NULL;
    int *lens = NULL;
    int n_prompts = 0, np;
    tok_trace tr_ref, tr_new;
    float *l_ref = NULL, *l_new = NULL;
    int n_logits = 0, n_logits2 = 0;
    vtm_t *m;
    int bad_prompts = 0, seq_ok = 0;
    double sum_top5 = 0.0, max_dl = 0.0, max_ref_all = 0.0;
    int steps_all = 0, argmax_all = 0;

    if (!path) {
        fprintf(stderr, "модель не найдена (--decode-check PATH)\n");
        return 2;
    }
    if (!prompts_path) {
        fprintf(stderr, "--decode-check требует --prompts FILE\n");
        return 2;
    }
    if (read_prompt_file(prompts_path, &ids, &n_prompts, &lens) != 0) return 2;
    if (n_prompts == 0) {
        fprintf(stderr, "в %s нет промптов\n", prompts_path);
        return 2;
    }
    if (limit > 0 && limit < n_prompts) n_prompts = limit;
    if (max_new < 1) max_new = 1;
    if (max_new > 32) max_new = 32;

    printf("модель: %s\n", path);
    printf("decode-check: %d промптов × %d токенов; int8-плоскостей: %d; "
           "эталон — f32-путь (режим 0), раньше в этом прогоне — %s\n",
           n_prompts, max_new, planes, vtm_q4k_i8_name());
    m = vtm_load(path);
    if (!m) {
        fprintf(stderr, "vtm_load(%s): %s\n", path, vtm_last_error());
        return 1;
    }
    vtm_set_backend(m, &vtm_backend_simd);
    {
        // Размер логитов известен из тензора модели до прогона: буфер под
        // max_new шагов выделяется один раз на все промпты.
        const vtm_tensor *lg = vtm_tensor_by_name(m, "logits");

        if (!lg || !lg->f32 || lg->n_elements == 0) {
            fprintf(stderr, "в модели нет тензора logits\n");
            vtm_free(m);
            return 1;
        }
        n_logits = (int)lg->n_elements;
    }
    l_ref = (float *)malloc((size_t)max_new * (size_t)n_logits
                            * sizeof(float));
    l_new = (float *)malloc((size_t)max_new * (size_t)n_logits
                            * sizeof(float));
    tr_ref.toks = (int *)malloc((size_t)max_new * sizeof(int));
    tr_new.toks = (int *)malloc((size_t)max_new * sizeof(int));
    tr_ref.cap = tr_new.cap = max_new;
    if (!l_ref || !l_new || !tr_ref.toks || !tr_new.toks) {
        fprintf(stderr, "decode-check: нет памяти (логитов %d × %d)\n",
                max_new, n_logits);
        free(l_ref);
        free(l_new);
        free(tr_ref.toks);
        free(tr_new.toks);
        vtm_free(m);
        return 1;
    }
    printf("  %4s %7s %7s %8s %9s %13s %6s  %s\n", "N", "prompt", "argmax",
           "top-5", "Δlogits", "мс/ток", "ускор.", "расхождение");
    for (np = 0; np < n_prompts; np++) {
        dec_cmp c;
        double t_ref = 0.0, t_new = 0.0;

        vtm_q4k_i8_set_mode(0);
        if (decode_run(m, ids[np], lens[np], max_new, l_ref, &n_logits2,
                       &tr_ref, &t_ref) != 0 || n_logits2 != n_logits) {
            fprintf(stderr, "decode-check: f32-прогон промпта %d не удался\n",
                    np);
            bad_prompts++;
            continue;
        }
        vtm_q4k_i8_set_mode(planes);
        if (decode_run(m, ids[np], lens[np], max_new, l_new, &n_logits2,
                       &tr_new, &t_new) != 0 || n_logits2 != n_logits) {
            fprintf(stderr, "decode-check: int8-прогон промпта %d не удался\n",
                    np);
            bad_prompts++;
            continue;
        }
        vtm_q4k_i8_set_mode(-1);
        decode_compare(l_ref, l_new, n_logits, &tr_ref, &tr_new, max_new, &c);
        // decode_compare заполняет c с нуля, поэтому времена — снаружи.
        c.t_ref = t_ref / (double)max_new;
        c.t_new = t_new / (double)max_new;
        printf("  %4d %7d %5d/%-2d %7.1f%% %9.2e %6.1f/%5.1f %5.2fx  %s\n",
               np + 1, lens[np], c.argmax_ok, c.steps_cmp,
               100.0 * c.top5_hits / (double)c.steps_cmp, c.max_dlogit,
               c.t_ref, c.t_new, c.t_ref / (c.t_new > 0.0 ? c.t_new : 1.0),
               c.diverge_step < 0 ? "нет" : "есть");
        steps_all += c.steps_cmp;
        argmax_all += c.argmax_ok;
        sum_top5 += c.top5_hits;
        if (c.max_dlogit > max_dl) max_dl = c.max_dlogit;
        if (c.max_ref > max_ref_all) max_ref_all = c.max_ref;
        if (c.diverge_step < 0) seq_ok++;
    }
    if (steps_all > 0) {
        printf("итого: последовательность совпала у %d/%d промптов, "
               "argmax %d/%d шагов, top-5 %.2f%%, max|Δlogits| = %.2e "
               "(max|ref|=%.2f, порог 1e-2)\n", seq_ok, n_prompts,
               argmax_all, steps_all,
               100.0 * sum_top5 / (double)steps_all, max_dl, max_ref_all);
        printf("decode-check: %s\n",
               (max_dl <= 1e-2 && argmax_all == steps_all && seq_ok == n_prompts)
                   ? "ok (argmax и последовательности совпали, Δlogits ≤ 1e-2)"
                   : "FAIL (argmax/последовательность/Δlogits вне допуска)");
    }
    free(l_ref);
    free(l_new);
    for (np = 0; np < n_prompts; np++) free(ids[np]);
    free(ids);
    free(lens);
    free(tr_ref.toks);
    free(tr_new.toks);
    vtm_free(m);
    return (bad_prompts || steps_all == 0) ? 1 : 0;
}

// --- упакованная таблица словаря: A/B по сырым логитам (задача 0170) -------
//
// Критерий 0170 — argmax 20/20 и Δlogits ≤1e-2 против прежнего пути, где
// int8-таблица разворачивалась в fp32 в арене. Два режима нельзя держать в
// одном процессе одновременно (1,24 ГиБ fp32-таблицы + упакованная), поэтому
// модель загружается ДВА РАЗА последовательно: сначала с VTM_I8_EMB=0
// (эталон, тензор приходит как kind 1), потом упакованная (kind 4). Пик
// памяти — максимум двух прогонов, а не сумма.
//
// Логиты сравниваются шаг за шагом на тех же промптах (decode_compare):
// argmax, top-5, max|Δlogits|, первый шаг расхождения последовательностей.
// Время на токен печатается тоже, но на загруженной машине абсолютные мс
// непригодны (page cache/подкачка) — решения по скорости принимаются по
// bench_i8_gemv_table и отчёту.
static int run_emb_check(const char *explicit_path, const char *prompts_path,
                         int max_new, int limit) {
    const char *path = find_model(explicit_path);
    int32_t **ids = NULL;
    int *lens = NULL;
    int n_prompts = 0, np, phase;
    float *lg[2] = {NULL, NULL};
    int *toks[2] = {NULL, NULL};
    tok_trace tr;
    double *ms[2] = {NULL, NULL};
    int n_logits = 0, n_logits2 = 0;
    int bad_prompts = 0, seq_ok = 0, steps_all = 0, argmax_all = 0;
    double sum_top5 = 0.0, max_dl = 0.0, max_ref_all = 0.0;

    if (!path) {
        fprintf(stderr, "модель не найдена (--emb-check PATH)\n");
        return 2;
    }
    if (!prompts_path) {
        fprintf(stderr, "--emb-check требует --prompts FILE\n");
        return 2;
    }
    if (read_prompt_file(prompts_path, &ids, &n_prompts, &lens) != 0) return 2;
    if (n_prompts == 0) {
        fprintf(stderr, "в %s нет промптов\n", prompts_path);
        return 2;
    }
    if (limit > 0 && limit < n_prompts) n_prompts = limit;
    if (max_new < 1) max_new = 1;
    if (max_new > 32) max_new = 32;

    printf("модель: %s\n", path);
    printf("emb-check: %d промптов × %d токенов; эталон — fp32-таблица "
           "(VTM_I8_EMB=0), после — упакованная int8 (kind 4)\n",
           n_prompts, max_new);
    for (phase = 0; phase < 2; phase++) {
        vtm_t *m;
        const vtm_tensor *emb;
        double t_all = 0.0;

        /* phase 0 — эталон: fp32-развёртка в арене; 1 — упакованная таблица. */
        vtm_i8_emb_set_mode(phase == 0 ? 0 : 1);
        m = vtm_load(path);
        if (!m) {
            fprintf(stderr, "vtm_load(%s): %s\n", path, vtm_last_error());
            goto fail;
        }
        vtm_set_backend(m, &vtm_backend_simd);
        emb = vtm_tensor_by_name(m, "emb.w");
        {
            const vtm_tensor *lgt = vtm_tensor_by_name(m, "logits");

            if (!lgt || !lgt->f32 || lgt->n_elements == 0) {
                fprintf(stderr, "в модели нет тензора logits\n");
                vtm_free(m);
                goto fail;
            }
            if (n_logits == 0) n_logits = (int)lgt->n_elements;
            else if (n_logits != (int)lgt->n_elements) {
                fprintf(stderr, "emb-check: разные размеры логитов\n");
                vtm_free(m);
                goto fail;
            }
        }
        printf("  фаза %d: emb.w kind=%s, %s\n", phase,
               emb ? (emb->kind == VTM_TENSOR_INT8 ? "int8"
                      : (emb->kind == VTM_TENSOR_F32 ? "f32" : "?")) : "нет",
               emb && emb->i8 ? "байты q в отображении"
                              : "fp32-развёртка в арене");
        lg[phase] = (float *)malloc((size_t)n_prompts * (size_t)max_new
                                    * (size_t)n_logits * sizeof(float));
        ms[phase] = (double *)calloc((size_t)n_prompts, sizeof(double));
        toks[phase] = (int *)malloc((size_t)n_prompts * (size_t)max_new
                                    * sizeof(int));
        if (!lg[phase] || !ms[phase] || !toks[phase]) {
            fprintf(stderr, "emb-check: нет памяти (логитов %d × %d × %d)\n",
                    n_prompts, max_new, n_logits);
            vtm_free(m);
            goto fail;
        }
        for (np = 0; np < n_prompts; np++) {
            double t = 0.0;
            float *dst = lg[phase] + (size_t)np * (size_t)max_new
                       * (size_t)n_logits;

            tr.toks = toks[phase] + (size_t)np * (size_t)max_new;
            tr.cap = max_new;
            tr.n = 0;
            if (decode_run(m, ids[np], lens[np], max_new, dst, &n_logits2,
                           &tr, &t) != 0 || n_logits2 != n_logits) {
                fprintf(stderr, "emb-check: прогон промпта %d (фаза %d) "
                        "не удался\n", np + 1, phase);
                vtm_free(m);
                goto fail;
            }
            ms[phase][np] = t / (double)max_new;
            t_all += ms[phase][np];
        }
        printf("         средней мс/токен: %.2f\n",
               n_prompts ? t_all / (double)n_prompts : 0.0);
        vtm_free(m);
    }
    vtm_i8_emb_set_mode(-1);

    printf("  %4s %7s %7s %8s %9s %13s %6s  %s\n", "N", "prompt", "argmax",
           "top-5", "Δlogits", "мс/ток", "ускор.", "расхождение");
    for (np = 0; np < n_prompts; np++) {
        dec_cmp c;
        size_t off = (size_t)np * (size_t)max_new * (size_t)n_logits;

        tr.toks = toks[0] + (size_t)np * (size_t)max_new;
        tr.cap = max_new;
        tr.n = max_new;
        {
            tok_trace tr_a = tr, tr_b = tr;

            tr_b.toks = toks[1] + (size_t)np * (size_t)max_new;
            decode_compare(lg[0] + off, lg[1] + off, n_logits, &tr_a, &tr_b,
                           max_new, &c);
        }
        c.t_ref = ms[0][np];
        c.t_new = ms[1][np];
        printf("  %4d %7d %5d/%-2d %7.1f%% %9.2e %6.1f/%5.1f %5.2fx  %s\n",
               np + 1, lens[np], c.argmax_ok, c.steps_cmp,
               100.0 * c.top5_hits / (double)c.steps_cmp, c.max_dlogit,
               c.t_ref, c.t_new,
               c.t_ref / (c.t_new > 0.0 ? c.t_new : 1.0),
               c.diverge_step < 0 ? "нет" : "есть");
        steps_all += c.steps_cmp;
        argmax_all += c.argmax_ok;
        sum_top5 += c.top5_hits;
        if (c.max_dlogit > max_dl) max_dl = c.max_dlogit;
        if (c.max_ref > max_ref_all) max_ref_all = c.max_ref;
        if (c.diverge_step < 0) seq_ok++;
    }
    if (steps_all > 0) {
        printf("итого: последовательность совпала у %d/%d промптов, "
               "argmax %d/%d шагов, top-5 %.2f%%, "
               "max|Δlogits| = %.2e (max|ref|=%.2f, порог 1e-2)\n",
               seq_ok, n_prompts, argmax_all, steps_all,
               100.0 * sum_top5 / (double)steps_all, max_dl, max_ref_all);
        printf("emb-check: %s\n",
               (max_dl <= 1e-2 && argmax_all == steps_all)
                   ? "ok (argmax совпал на всех шагах, Δlogits ≤ 1e-2)"
                   : "FAIL (argmax/Δlogits вне допуска)");
    }
    free(lg[0]);
    free(lg[1]);
    free(ms[0]);
    free(ms[1]);
    free(toks[0]);
    free(toks[1]);
    for (np = 0; np < n_prompts; np++) free(ids[np]);
    free(ids);
    free(lens);
    return (bad_prompts || steps_all == 0) ? 1 : 0;
fail:
    vtm_i8_emb_set_mode(-1);
    free(lg[0]);
    free(lg[1]);
    free(ms[0]);
    free(ms[1]);
    free(toks[0]);
    free(toks[1]);
    for (np = 0; np < n_prompts; np++) free(ids[np]);
    free(ids);
    free(lens);
    return 1;
}

// --- TTFT prefill на настоящей модели: по токену против пачки (0162) --------
//
// Один процесс, чередование: для каждой длины промпта сначала один путь,
// потом другой, и порядок меняется от повтора к повтору. Так дрейф нагрузки
// и состояние page cache бьют по обоим путям одинаково — на общей машине
// одиночные прогоны CLI расходятся в 2-3 раза.
//
// Считается только prefill (последний токен промпта включён): ровно то, что
// ждёт хост до первого токена ответа. ids синтетические (одинаковый поток для
// обоих путей), модель настоящая.
static double prefill_token(vtm_t *m, const int32_t *ids, int n, double *cpu) {
    double t0 = now_ms(), c0 = now_cpu_ms();
    int i;
    for (i = 0; i < n; i++) {
        vtm_set_position(m, i);
        vtm_set_token(m, ids[i]);
        if (vtm_run(m, NULL, NULL) != VTM_OK) return -1.0;
    }
    if (cpu) *cpu = now_cpu_ms() - c0;
    return now_ms() - t0;
}

static double prefill_batch(vtm_t *m, const int32_t *ids, int n, double *cpu) {
    double t0, c0;
    vtm_reset(m);
    vtm_set_position(m, 0);
    t0 = now_ms();
    c0 = now_cpu_ms();
    if (vtm_run_batch(m, ids, n) != VTM_OK) return -1.0;
    if (cpu) *cpu = now_cpu_ms() - c0;
    return now_ms() - t0;
}

static const char *g_ttft_lens;   // --prefill-lens
static int g_ttft_reps = 2;       // --prefill-reps
// 0211: --prefill-backend выбирает бэкенд именно для prefill-замера (не
// трогая прежнее поведение «по умолчанию simd»), --prefill-batch-only
// убирает поштучный путь (он читает веса на каждый токен: 384 токена — это
// ~770 ГБ, часы), --prefill-batch N задаёт VTM_BATCH_TOKENS до загрузки.
static const char *g_prefill_backend;
static int g_prefill_batch_only;
static long g_prefill_batch;

static int run_prefill_ttft(const char *explicit_path) {
    static const int lens_def[] = {64, 256, 512};
    const int *lens = lens_def;
    int nlens = (int)(sizeof lens_def / sizeof lens_def[0]);
    const char *path = find_model(explicit_path);
    int32_t ids[512];
    vtm_t *m;
    size_t li;
    int i;

    if (!path) {
        fprintf(stderr, "модель не найдена (--prefill-ttft PATH)\n");
        return 2;
    }
    // --prefill-lens L1[,L2...] — сузить набор длин (на загруженной машине
    // полный прогон 64/256/512x2 занимает часы); --prefill-reps N — число
    // чередующихся повторов (по умолчанию 2, берётся минимум).
    if (g_ttft_lens && *g_ttft_lens) {
        static int custom[16];
        const char *p = g_ttft_lens;
        int k = 0;
        while (*p && k < 16) {
            custom[k++] = atoi(p);
            while (*p && *p != ',') p++;
            if (*p == ',') p++;
        }
        if (k > 0) {
            lens = custom;
            nlens = k;
        }
    }
    m = vtm_load(path);
    if (!m) {
        fprintf(stderr, "vtm_load(%s): %s\n", path, vtm_last_error());
        return 1;
    }
    // 0211: выбор бэкенда для этого замера. По умолчанию simd — прежнее
    // поведение; hybrid/metal нужны, чтобы увидеть, где включается GPU-батч.
    {
        const vtm_backend *be = &vtm_backend_simd;
        if (g_prefill_backend && *g_prefill_backend) {
            be = vtm_backend_by_name(g_prefill_backend);
            if (!be) {
                fprintf(stderr, "неизвестный --prefill-backend: %s\n",
                        g_prefill_backend);
                vtm_free(m);
                return 2;
            }
        }
        vtm_set_backend(m, be);
    }
    if (g_prefill_batch > 0) {
        char buf[32];
        snprintf(buf, sizeof buf, "%ld", g_prefill_batch);
        setenv("VTM_BATCH_TOKENS", buf, 1);
    }
    for (i = 0; i < 512; i++) ids[i] = 1000 + (i * 37) % 150000;
    {
        const char *bt = getenv("VTM_BATCH_TOKENS");
        char batch[32];

        if (g_prefill_batch > 0)
            snprintf(batch, sizeof batch, "%ld", g_prefill_batch);
        else if (bt && *bt)
            snprintf(batch, sizeof batch, "%s", bt);
        else
            snprintf(batch, sizeof batch, "по умолчанию");
        printf("модель: %s\nTTFT prefill (один процесс%s; "
               "VTM_BATCH_TOKENS=%s):\n", path,
               g_prefill_batch_only ? ", только пачка"
                                    : ", по токену против пачки",
               batch);
    }
    printf("int8-dot: %s (VTM_Q4_I8=%d)\n", vtm_q4_i8_name(),
           vtm_q4_i8_mode());
    if (g_prefill_batch_only) {
        printf("  %6s %11s %11s %9s\n", "токенов", "TTFT, мс", "ток/с",
               "CPU, мс");
    } else {
        printf("  %6s %11s %11s %9s %11s %11s %9s\n", "токенов", "токен wall",
               "пачка wall", "ускор.", "токен CPU", "пачка CPU", "ускор.CPU");
    }
    for (li = 0; li < (size_t)nlens; li++) {
        const int n = lens[li];
        double t_tok = 1e300, t_bat = 1e300;
        double c_tok = 1e300, c_bat = 1e300;
        int rep;
        for (rep = 0; rep < g_ttft_reps; rep++) {
            double a, b, ca = -1.0, cb = -1.0;
            if (g_prefill_batch_only) {
                a = 0.0;
                b = prefill_batch(m, ids, n, &cb);
            } else if (rep == 0) {
                a = prefill_token(m, ids, n, &ca);
                b = prefill_batch(m, ids, n, &cb);
            } else {
                b = prefill_batch(m, ids, n, &cb);
                a = prefill_token(m, ids, n, &ca);
            }
            if (a < 0.0 || b < 0.0) {
                fprintf(stderr, "prefill-ttft: прогон вернул ошибку\n");
                vtm_free(m);
                return 1;
            }
            if (a < t_tok) t_tok = a;
            if (b < t_bat) t_bat = b;
            if (ca >= 0.0 && ca < c_tok) c_tok = ca;
            if (cb >= 0.0 && cb < c_bat) c_bat = cb;
        }
        if (g_prefill_batch_only) {
            printf("  %6d %11.2f %11.1f %9.2f\n", n, t_bat,
                   t_bat > 0.0 ? 1e3 * (double)n / t_bat : 0.0, c_bat);
        } else {
            printf("  %6d %11.2f %11.2f %8.2fx %11.2f %11.2f %8.2fx\n", n,
                   t_tok / 1e3, t_bat / 1e3, t_tok / t_bat, c_tok / 1e3,
                   c_bat / 1e3, c_tok / c_bat);
        }
    }
    vtm_free(m);
    return 0;
}

// --- 0199: паритет сгенерированных токенов через C-ABI ----------------------
//
// Путь чата/сервера, которого не было в бенчмарках 0192/0198: vtm_api_generate
// с cache_prefix=1 и prefill_batch=256 на настоящей 3B-модели. Печатает
// строки PROMPT/TOK, чтобы вызывающий сравнил simd и hybrid. Один бэкенд на
// процесс намеренно: режим Metal (VTM_METAL_OFF/FULL/BATCH) — глобальный, и
// два контекста в одном процессе делили бы его (открытие hybrid включило бы
// GPU и для уже открытого simd; ровно это сбило первый замер 0199).
#define API_MAX_NEW 16

typedef struct {
    int32_t ids[API_MAX_NEW];
    size_t n;
} api_bag;

static int api_collect(const int32_t *ids, size_t n, void *user) {
    api_bag *b = (api_bag *)user;
    size_t i;

    for (i = 0; i < n && b->n < API_MAX_NEW; i++) b->ids[b->n++] = ids[i];
    return 0;
}

static int run_api_tokens(const char *path, const char *backend) {
    // Промпты взяты длинами, на которых ломался старый GPU-батч (0199):
    // prefill = len-1 -> 14 (gemm16), 23 (gemm32), 44 (32+12).
    static const int lens[] = {15, 24, 45};
    vtm_api_options opts;
    vtm_api_t *ctx;
    int p, i, rc;
    char what[160];

    // Жадный SAMP: паритет по argmax, а не по ГПСЧ. Переменные читаются при
    // первом SAMP, то есть уже после open(), — поэтому setenv до open.
    setenv("VTM_TEMPERATURE", "0", 1);
    setenv("VTM_TOP_K", "0", 1);
    setenv("VTM_TOP_P", "1", 1);

    memset(&opts, 0, sizeof opts);
    opts.struct_size = sizeof opts;
    opts.backend_name = backend;
    opts.max_new_tokens = 8;
    opts.cache_prefix = 1;
    opts.prefill_batch = 256;
    ctx = vtm_api_open(path, &opts);
    if (!ctx) {
        fprintf(stderr, "api-tokens: open(%s): %s\n", backend,
                vtm_api_last_error(NULL));
        return 1;
    }
    printf("BACKEND %s\n", backend);

    for (p = 0; p < (int)(sizeof lens / sizeof lens[0]); p++) {
        int32_t prompt[128];
        api_bag first, again;

        for (i = 0; i < lens[p]; i++)
            prompt[i] = (int32_t)(100 + (i * 37 + p * 101) % 900);

        memset(&first, 0, sizeof first);
        rc = vtm_api_generate(ctx, prompt, (size_t)lens[p], api_collect,
                              &first);
        snprintf(what, sizeof what, "api-tokens %s prompt %d", backend, p);
        if (rc != 0) {
            fprintf(stderr, "%s: generate -> %d\n", what, rc);
            vtm_api_close(ctx);
            return 1;
        }
        // Повтор того же промпта: теперь префикс берётся из кэша. Если снимок
        // снят с незавершённого GPU-батча, второй ответ разойдётся с первым.
        memset(&again, 0, sizeof again);
        rc = vtm_api_generate(ctx, prompt, (size_t)lens[p], api_collect,
                              &again);
        if (rc != 0) {
            fprintf(stderr, "%s: повтор -> %d\n", what, rc);
            vtm_api_close(ctx);
            return 1;
        }
        printf("PROMPT %d %d\n", p, lens[p]);
        printf("TOK %d 0", p);
        for (i = 0; i < (int)first.n; i++) printf(" %d", (int)first.ids[i]);
        printf("\nTOK %d 1", p);
        for (i = 0; i < (int)again.n; i++) printf(" %d", (int)again.ids[i]);
        printf("\n");
    }
    vtm_api_close(ctx);
    return 0;
}

// --- 0211: изолированный GPU-батч GEMM -------------------------------------
//
// --metal-gemm ROWS,COLS,N,ITERS[,STRIDE[,KIND]]. Меряет батчевый GEMM без
// модели и без перекладки активаций по слоям: видно потолок ядра, а не шум
// загрузки. KIND (0216): 0 — Q4_K, 1 — Q6_K; веса нулевые, важен только
// формат блока (144/210 байт).
// Блоки весов нулевые — арифметика та же, время честное.
//
// STRIDE > 0 — «арена пачки» (0214): вход и выход лежат в одной
// странично-выровненной арене с шагом STRIDE float на токен (так устроен
// b->mem в interp.c), вызов идёт через vtm_metal_qk_gemm_arena +
// vtm_metal_gemm_flush. Это ровно живой путь prefill; STRIDE = cols даёт
// компактную раскладку, STRIDE = размер слота модели (609 КиБ у 3B Q4_K) —
// живую. STRIDE = 0 — прежний упакованный путь vtm_metal_qk_gemm.
//
// Печатает wall (min из ITERS), GPU (min, из vtm_metal_stats.gpu_ns) и
// GMAC/s по wall и по GPU.
static float *bench_page_alloc(size_t bytes) {
    void *p = NULL;
    long page = sysconf(_SC_PAGESIZE);
    size_t rounded;

    if (page <= 0) page = 4096;
    rounded = (bytes + (size_t)page - 1u) & ~((size_t)page - 1u);
    if (rounded == 0) rounded = (size_t)page;
    if (posix_memalign(&p, (size_t)page, rounded) != 0 || !p) return NULL;
    memset(p, 0, rounded);
    return (float *)p;
}

static int run_metal_gemm_bench(const char *spec) {
    int rows = 2048, cols = 2048, ntok = 64, iters = 10;
    long stride = 0;
    int kind = 0;   // 0216: 0 — Q4_K, 1 — Q6_K (формат задаёт только блок)
    unsigned char *blocks;
    float *x = NULL, *y = NULL;
    vtm_metal_stats st0, st1;
    double t0, best = 1e300, best_gpu = 1e300;
    size_t wbytes;
    int it;

    if (spec && *spec)
        sscanf(spec, "%d,%d,%d,%d,%ld,%d", &rows, &cols, &ntok, &iters,
               &stride, &kind);
    if (rows <= 0 || cols <= 0 || ntok <= 0 || iters <= 0
        || cols % 256 != 0 || kind < 0 || kind > 1) {
        fprintf(stderr, "--metal-gemm ROWS,COLS,N,ITERS[,STRIDE[,KIND]] "
                        "(COLS кратен 256, KIND 0 — Q4_K, 1 — Q6_K)\n");
        return 2;
    }
    wbytes = (size_t)rows * (size_t)cols / 256u
           * (kind == 1 ? VTM_Q6K_BLOCK_BYTES : VTM_Q4K_BLOCK_BYTES);
    blocks = (unsigned char *)calloc(wbytes ? wbytes : 1u, 1);
    if (!blocks) {
        fprintf(stderr, "metal-gemm: нет памяти\n");
        return 1;
    }
    if (stride > 0) {
        // Арена: запас 32 токена (ядро варианта NT читает/пишет ровно NT).
        size_t slot = (size_t)stride;
        size_t ntok_pad = (size_t)ntok + 32u;
        size_t abytes;
        float *arena;

        if (slot < (size_t)cols) slot = (size_t)cols;
        abytes = ntok_pad * slot * sizeof(float);
        arena = bench_page_alloc(abytes);
        if (!arena) {
            free(blocks);
            fprintf(stderr, "metal-gemm: нет памяти под арену %zu байт\n",
                    abytes);
            return 1;
        }
        x = arena;                          // вход — с начала слота
        y = arena + (size_t)cols;           // выход — следующий тензор слота
        // Привязка арены — как в живом prefill: одна на прогон, дальше только
        // смещения тензоров. Без неё каждая матрица создаёт no-copy буфер на
        // весь слот (608 КиБ/токен) и хост-время растёт втрое.
        (void)vtm_metal_batch_arena(arena, abytes);
        printf("metal gemm %s %dx%d, %d токенов, %d итераций, арена: шаг %ld "
               "float (%.1f КиБ/токен, %.1f МиБ)\n",
               kind == 1 ? "Q6_K" : "Q4_K", rows, cols, ntok, iters, stride,
               (double)stride * 4.0 / 1024.0,
               (double)abytes / (1024.0 * 1024.0));
        for (it = 0; it < iters; it++) {
            int rc = vtm_metal_qk_gemm_arena(kind, blocks, NULL, rows, cols,
                                             ntok, x, (int)slot, y,
                                             (int)slot);
            if (rc != 0) {
                fprintf(stderr, "metal-gemm: арена отказала (%d)\n", rc);
                free(arena);
                free(blocks);
                return 1;
            }
            (void)vtm_metal_gemm_flush();
            vtm_metal_stats_get(&st0);
            t0 = now_ms();
            rc = vtm_metal_qk_gemm_arena(kind, blocks, NULL, rows, cols, ntok,
                                         x, (int)slot, y, (int)slot);
            if (rc == 0) rc = vtm_metal_gemm_flush();
            t0 = now_ms() - t0;
            vtm_metal_stats_get(&st1);
            if (rc != 0) break;
            if (t0 < best) best = t0;
            {
                double g = (double)(st1.gpu_ns - st0.gpu_ns) / 1e6;
                if (g > 0.0 && g < best_gpu) best_gpu = g;
            }
        }
        vtm_metal_batch_arena_drop(arena);
        free(arena);
    } else {
        x = (float *)calloc((size_t)cols * (size_t)ntok, sizeof *x);
        y = (float *)calloc((size_t)rows * (size_t)ntok, sizeof *y);
        if (!x || !y) {
            free(blocks); free(x); free(y);
            fprintf(stderr, "metal-gemm: нет памяти\n");
            return 1;
        }
        printf("metal gemm %s %dx%d, %d токенов, %d итераций, упакованный "
               "вход\n", kind == 1 ? "Q6_K" : "Q4_K", rows, cols, ntok, iters);
        for (it = 0; it < iters; it++) {
            int rc = vtm_metal_qk_gemm(kind, y, blocks, NULL, x, rows, cols,
                                       ntok);
            if (rc != 0) {
                fprintf(stderr, "metal-gemm: ядро отказало (%d)\n", rc);
                free(blocks); free(x); free(y);
                return 1;
            }
            vtm_metal_stats_get(&st0);
            t0 = now_ms();
            rc = vtm_metal_qk_gemm(kind, y, blocks, NULL, x, rows, cols,
                                   ntok);
            t0 = now_ms() - t0;
            vtm_metal_stats_get(&st1);
            if (rc != 0) break;
            if (t0 < best) best = t0;
            {
                double g = (double)(st1.gpu_ns - st0.gpu_ns) / 1e6;
                if (g > 0.0 && g < best_gpu) best_gpu = g;
            }
        }
    }
    if (best < 1e299) {
        double mac = (double)rows * (double)cols * (double)ntok;
        printf("  wall: %.3f мс = %.1f GMAC/s\n", best, mac / best * 1e-6);
        if (best_gpu < 1e299)
            printf("  GPU:  %.3f мс = %.1f GMAC/s (хост+ожидание %.3f мс)\n",
                   best_gpu, mac / best_gpu * 1e-6,
                   best - best_gpu > 0.0 ? best - best_gpu : 0.0);
    } else {
        printf("  metal недоступен/ядро отказало\n");
    }
    free(blocks); free(x); free(y);
    return 0;
}

// --- 0211: кэш префикса через C-ABI, как в чате -----------------------------
//
// Повторяет путь сервера: один контекст, cache_prefix=1, растущая история.
// Для каждой длины L: запрос на промпте ids[0..L) (первый раз — с хвостом,
// которого не было в кэше), затем ТОТ ЖЕ промпт ещё раз (должен взять весь
// префикс из снимка, без единого нового токена). Печатает TTFT, длину
// снимка и счётчики попаданий/промахов — так видно, что кэш реально
// переиспользует KV, а не просто «стало быстрее».
static int run_api_cache(const char *explicit_path, const char *backend) {
    static const int def_lens[] = {40, 80, 154, 214, 250, 500};
    int lens[16];
    int nlens = (int)(sizeof def_lens / sizeof def_lens[0]);
    int32_t ids[512];
    vtm_api_options opts;
    vtm_api_t *ctx;
    const char *path = find_model(explicit_path);
    size_t li;
    int i;

    if (!path) {
        fprintf(stderr, "модель не найдена (--api-cache PATH)\n");
        return 2;
    }
    if (g_ttft_lens && *g_ttft_lens) {
        const char *p = g_ttft_lens;
        int k = 0;
        while (*p && k < 16) {
            lens[k++] = atoi(p);
            while (*p && *p != ',') p++;
            if (*p == ',') p++;
        }
        if (k > 0) nlens = k;
    }
    for (i = 0; i < 512; i++) ids[i] = 1000 + (i * 37) % 150000;

    setenv("VTM_TEMPERATURE", "0", 1);
    setenv("VTM_TOP_K", "0", 1);
    setenv("VTM_TOP_P", "1", 1);
    memset(&opts, 0, sizeof opts);
    opts.struct_size = sizeof opts;
    opts.backend_name = backend;
    opts.max_new_tokens = 1;      // только TTFT первого токена
    opts.eos_token_id = -1;
    opts.cache_prefix = 1;
    opts.prefill_batch = 256;
    ctx = vtm_api_open(path, &opts);
    if (!ctx) {
        fprintf(stderr, "api-cache: open(%s): %s\n", backend ? backend : "auto",
                vtm_api_last_error(NULL));
        return 1;
    }
    printf("BACKEND %s\n", backend ? backend : "auto");
    printf("  %6s %11s %11s %7s %10s %10s %7s %7s\n", "токенов", "1-й TTFT",
           "повтор TTFT", "хвост", "снимок", "hits", "misses", "ток/с");
    for (li = 0; li < (size_t)nlens; li++) {
        const int L = lens[li];
        double t_first, t_repeat;
        uint64_t h0 = 0, m0 = 0, h1 = 0, m1 = 0;
        int len_before, len_after;
        int rc;

        if (L <= 0 || L > 512) {
            fprintf(stderr, "api-cache: длина %d вне 1..512\n", L);
            vtm_api_close(ctx);
            return 2;
        }
        vtm_api_cache_stats(ctx, &h0, &m0);
        len_before = vtm_api_cache_prefix_len(ctx);
        t_first = now_ms();
        rc = vtm_api_generate(ctx, ids, (size_t)L, NULL, NULL);
        t_first = now_ms() - t_first;
        len_after = vtm_api_cache_prefix_len(ctx);
        if (rc != VTM_API_OK) {
            fprintf(stderr, "api-cache: generate(%d) -> %d: %s\n", L, rc,
                    vtm_api_last_error(ctx));
            vtm_api_close(ctx);
            return 1;
        }
        t_repeat = now_ms();
        rc = vtm_api_generate(ctx, ids, (size_t)L, NULL, NULL);
        t_repeat = now_ms() - t_repeat;
        if (rc != VTM_API_OK) {
            fprintf(stderr, "api-cache: повтор(%d) -> %d: %s\n", L, rc,
                    vtm_api_last_error(ctx));
            vtm_api_close(ctx);
            return 1;
        }
        vtm_api_cache_stats(ctx, &h1, &m1);
        printf("  %6d %11.2f %11.2f %7d %10d %10llu %7llu %7.1f\n", L,
               t_first, t_repeat, L - 1 - len_before, len_after,
               (unsigned long long)(h1 - h0), (unsigned long long)(m1 - m0),
               t_first > 0.0 ? 1e3 * (double)L / t_first : 0.0);
    }
    vtm_api_close(ctx);
    return 0;
}

// --- main ------------------------------------------------------------------

static void usage(const char *prog) {
    printf("usage: %s [--rows N] [--cols N] [--iters N] [--scale-block 128|160]\n"
           "          [--backend auto|scalar|simd|direct] [--check] "
           "[--threads] [--pool] [--logits PATH] [--no-mm]\n"
           "          [--gemv-rows N] [--gemv-cols N]\n"
           "          [--q4k [--rows N] [--cols N] [--iters N] [--batch-tokens N]]\n"
           "          [--q4k-i8 [--rows N] [--cols N] [--iters N]]\n"
           "          [--batch] [--batch-tokens N] "
           "[--q4 [--q4-rows N] [--q4-cols N] [--q4-iters N] "
           "[--q4-batch]]\n"
           "          [--prefill-check PATH] [--prefill-ttft PATH]\n"
           "          [--prefill-lens 64,256,512] [--prefill-reps N]\n"
           "          [--prefill-backend simd|hybrid|metal] [--prefill-batch-only]\n"
           "          [--prefill-batch N]\n"
           "          [--metal PATH] [--metal-new N] [--metal-lens 64,128]\n"
           "          [--decode-bench PATH --prompts FILE [--decode-new N]\n"
           "           [--decode-reps N]]\n"
           "          [--decode-check PATH --prompts FILE [--decode-new N]\n"
           "           [--decode-limit N] [--planes 1|2|3]]\n"
           "          [--emb-check PATH --prompts FILE [--decode-new N]\n"
           "           [--decode-limit N]]\n"
           "          [--i8-gemv [--rows N] [--cols N] [--iters N]\n"
           "           [--f32-table]]\n"
           "          [--qk-proj PATH [--iters N]]\n"
           "          [--api-tokens PATH [--backend simd|hybrid]]\n"
           "          [--api-cache PATH [--backend simd|hybrid|metal]\n"
           "           [--prefill-lens 40,80,154,214,250,500]]\n"
           "          [--metal-gemm ROWS,COLS,N,ITERS[,STRIDE[,KIND]]]\n"
           "  env VTM_BACKEND — то же, что --backend (по умолчанию simd);\n"
           "  env VTM_THREADS — число потоков ядер (по умолчанию ядра, ≤ 8);\n"
           "  env VTM_Q4_I8 — int8-dot (SDOT/VNNI): 0 — выключить; 1/2/3 —\n"
           "    плоскости активаций (декод kind 6/7, задачи 0169/0171; по\n"
           "    умолчанию 3); у батчевого Q4 (0163) 1 — один проход, 2|3 — два;\n"
           "  env VTM_BATCH_RMSH=0 — поштучный RMSH в пачке (A/B к 0215);\n"
           "  env VTM_BATCH_LAST=0 — не считать lm_head только для последнего\n"
           "    токена пачки и исполнять SAMP по всем токенам (A/B к 0215);\n"
           "  env VTM_BATCH_TRACE=1 — разбивка времени пачки по опкодам;\n",
           prog);
}

int main(int argc, char **argv) {
    int rows = 1024, cols = 1024, iters = 50, scale_block = 160;
    int check = 0, no_mm = 0, threads = 0, pool = 0, batch = 0;
    int batch_tokens = 16;
    int q4 = 0, q4_batch = 0, q4_rows = 151936, q4_cols = 896, q4_iters = 3;
    int q4k = 0, q4k_i8 = 0;
    // Форма плотного GEMV (f32 n == 1): по умолчанию — уменьшенный аналог
    // tied lm_head 3B Q4 ([151936, 2048]); реальную задают флагами.
    int gemv_rows = 16384, gemv_cols = 2048;
    const char *logits_path = NULL;
    const char *prefill_check_path = NULL;
    const char *prefill_ttft_path = NULL;
    const char *decode_check_path = NULL;
    const char *decode_bench_path = NULL;
    int decode_reps = 2;
    const char *decode_prompts_path = NULL;
    int decode_new = 8, decode_limit = 0, planes = 3;
    int emb_check_path_set = 0, i8_gemv = 0, i8_f32_table = 0;
    const char *qk_proj_path = NULL;
    const char *emb_check_path = NULL;
    const char *metal_bench_path = NULL;
    int metal_new = 16;
    const char *api_tokens_path = NULL;
    const char *api_cache_path = NULL;
    const char *metal_gemm_spec = NULL;
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
        } else if (strcmp(argv[i], "--gemv-rows") == 0 && i + 1 < argc) {
            gemv_rows = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--gemv-cols") == 0 && i + 1 < argc) {
            gemv_cols = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--q4k") == 0) {
            q4k = 1;
        } else if (strcmp(argv[i], "--q4k-i8") == 0) {
            q4k_i8 = 1;
        } else if (strcmp(argv[i], "--q4") == 0) {
            q4 = 1;
        } else if (strcmp(argv[i], "--q4-rows") == 0 && i + 1 < argc) {
            q4_rows = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--q4-cols") == 0 && i + 1 < argc) {
            q4_cols = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--q4-iters") == 0 && i + 1 < argc) {
            q4_iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--q4-batch") == 0) {
            q4_batch = 1;
        } else if (strcmp(argv[i], "--scale-block") == 0 && i + 1 < argc) {
            scale_block = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--metal") == 0) {
            metal_bench_path = (i + 1 < argc) ? argv[++i] : "";
        } else if (strcmp(argv[i], "--metal-new") == 0 && i + 1 < argc) {
            metal_new = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--metal-lens") == 0 && i + 1 < argc) {
            g_metal_lens = argv[++i];   // длины prefill, по умолчанию 64,128
        } else if (strcmp(argv[i], "--api-tokens") == 0) {
            api_tokens_path = (i + 1 < argc) ? argv[++i] : "";
        } else if (strcmp(argv[i], "--api-cache") == 0) {
            api_cache_path = (i + 1 < argc) ? argv[++i] : "";
        } else if (strcmp(argv[i], "--metal-gemm") == 0 && i + 1 < argc) {
            metal_gemm_spec = argv[++i];
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
        } else if (strcmp(argv[i], "--prefill-check") == 0) {
            prefill_check_path = (i + 1 < argc) ? argv[++i] : "";
        } else if (strcmp(argv[i], "--prefill-ttft") == 0) {
            prefill_ttft_path = (i + 1 < argc) ? argv[++i] : "";
        } else if (strcmp(argv[i], "--prefill-lens") == 0 && i + 1 < argc) {
            g_ttft_lens = argv[++i];
        } else if (strcmp(argv[i], "--prefill-reps") == 0 && i + 1 < argc) {
            g_ttft_reps = atoi(argv[++i]);
            if (g_ttft_reps < 1) g_ttft_reps = 1;
            if (g_ttft_reps > 8) g_ttft_reps = 8;
        } else if (strcmp(argv[i], "--prefill-backend") == 0 && i + 1 < argc) {
            g_prefill_backend = argv[++i];
        } else if (strcmp(argv[i], "--prefill-batch-only") == 0) {
            g_prefill_batch_only = 1;
        } else if (strcmp(argv[i], "--prefill-batch") == 0 && i + 1 < argc) {
            g_prefill_batch = atol(argv[++i]);
            if (g_prefill_batch < 1 || g_prefill_batch > 256) {
                fprintf(stderr, "--prefill-batch должен быть 1..256\n");
                return 2;
            }
        } else if (strcmp(argv[i], "--decode-check") == 0 && i + 1 < argc) {
            decode_check_path = argv[++i];
        } else if (strcmp(argv[i], "--prompts") == 0 && i + 1 < argc) {
            decode_prompts_path = argv[++i];
        } else if (strcmp(argv[i], "--emb-check") == 0 && i + 1 < argc) {
            emb_check_path = argv[++i];
            emb_check_path_set = 1;
        } else if (strcmp(argv[i], "--i8-gemv") == 0) {
            i8_gemv = 1;
        } else if (strcmp(argv[i], "--f32-table") == 0) {
            i8_f32_table = 1;
        } else if (strcmp(argv[i], "--decode-bench") == 0 && i + 1 < argc) {
            decode_bench_path = argv[++i];
        } else if (strcmp(argv[i], "--qk-proj") == 0 && i + 1 < argc) {
            qk_proj_path = argv[++i];
        } else if (strcmp(argv[i], "--decode-reps") == 0 && i + 1 < argc) {
            decode_reps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--decode-new") == 0 && i + 1 < argc) {
            decode_new = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--decode-limit") == 0 && i + 1 < argc) {
            decode_limit = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--planes") == 0 && i + 1 < argc) {
            planes = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "неизвестный аргумент: %s\n", argv[i]);
            usage(argv[0]);
            return 2;
        }
    }
    /* --i8-gemv меряет настоящую форму таблицы словаря (151936×2048), поэтому
       для него предел строк — как у --gemv-rows, а не 32768. */
    if (rows <= 0 || cols <= 0 || iters <= 0 || cols > 32768
        || rows > (i8_gemv ? (1 << 22) : 32768)) {
        fprintf(stderr, "плохие размеры/итерации\n");
        return 2;
    }
    if (gemv_rows <= 0 || gemv_cols <= 0 || gemv_cols > (1 << 20) ||
        gemv_rows > (1 << 22)) {
        fprintf(stderr, "плохие --gemv-rows/--gemv-cols\n");
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
    if (!backend_name && (!getenv("VTM_BACKEND") || !*getenv("VTM_BACKEND"))) {
        cand = &vtm_backend_simd;  // бенчмарк по умолчанию меряет SIMD
        // auto (0198) мог выставить BATCH/устройством — вернуть явный simd,
        // иначе синтетические замеры пойдут с включённым режимом Metal.
        vtm_metal_set_mode(VTM_METAL_OFF);
    }

    // 0199: C-ABI-паритет — один бэкенд на процесс (см. run_api_tokens).
    if (api_tokens_path)
        return run_api_tokens(*api_tokens_path ? api_tokens_path : NULL,
                              cand->name);
    // 0211: кэш префикса через C-ABI (два одинаковых запроса подряд).
    if (api_cache_path)
        return run_api_cache(*api_cache_path ? api_cache_path : NULL,
                             cand->name);
    if (metal_gemm_spec)
        return run_metal_gemm_bench(metal_gemm_spec);
    if (emb_check_path_set)
        return run_emb_check(emb_check_path, decode_prompts_path, decode_new,
                             decode_limit);
    if (i8_gemv)
        return run_i8_gemv_bench(rows, cols, iters, i8_f32_table);
    if (qk_proj_path)
        return run_qk_proj(*qk_proj_path ? qk_proj_path : NULL, iters);
    if (metal_bench_path)
        return run_metal_bench(*metal_bench_path ? metal_bench_path : NULL,
                               metal_new);
    if (decode_bench_path)
        return run_decode_bench(*decode_bench_path ? decode_bench_path : NULL,
                                decode_prompts_path, decode_new, decode_reps);
    if (decode_check_path)
        return run_decode_check(*decode_check_path ? decode_check_path : NULL,
                                decode_prompts_path, decode_new, planes,
                                decode_limit);
    if (prefill_ttft_path)
        return run_prefill_ttft(*prefill_ttft_path ? prefill_ttft_path
                                                   : NULL);
    if (prefill_check_path)
        return run_prefill_check(*prefill_check_path ? prefill_check_path
                                                     : NULL, batch_tokens);
    if (q4k_i8)
        return run_q4k_i8_bench(rows, cols, iters);
    if (q4k)
        return run_q4k_bench(rows, cols, iters, batch_tokens);
    if (q4)
        return q4_batch ? run_q4_batch_bench(q4_rows, q4_cols, batch_tokens,
                                             q4_iters)
                        : run_q4_bench(q4_rows, q4_cols, q4_iters);
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
        int giters = iters;
        printf("\n");
        bench_mm_table(1024, 1024, 1, scale_block, iters);
        bench_mm_table(128, 1024, 32, scale_block, iters);
        bench_mm_f32_table(256, 256, 256, iters);
        // f32 n == 1 — путь tied lm_head в декоде (задача 0159). На форме
        // побольше число итераций меньше: один проход — сотни Мфлоп.
        if ((double)gemv_rows * (double)gemv_cols > 1.0e7 && giters > 5)
            giters = 5;
        bench_gemv_f32_table(gemv_rows, gemv_cols, giters);
        // Батчевый плотный путь — tied lm_head в prefill (N = 16 токенов).
        bench_dense_batch_table(gemv_rows, gemv_cols, 16,
                                giters > 3 ? 3 : giters);
    }
    return 0;
}
