// Исполнитель байткода VTM v0.1.
//
// Позиция. ROPE/KVW/ATTN/SM считают рабочую позицию как
//     pos = m->position + imm,
// где m->position задаёт хост (vtm_set_position), а imm — смещение из
// инструкции. Для одиночного вызова с position == 0 это ровно imm, как в
// спеке. Такая схема позволяет гонять один и тот же байткод по токенам:
// хост перед каждым vtm_run() ставит position, а компилятор пишет imm = 0.
//
// Соглашения по полям инструкции (подробно — src/vtm/README.md):
//   dst — приёмник, a/b/c — источники; imm — параметр операции.
//   MUL/ADD, несмотря на запись «c = a·b» в спеке, пишут в dst: буква c
//   там означает результат, а не поле инструкции.
#include "vtm_internal.h"
#include "vtm_backend.h"
#include "sha256.h"

#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// --- диагностика -----------------------------------------------------------

static vtm_status fail(vtm_t *m, uint32_t idx, vtm_status st,
                       const char *fmt, ...) {
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    vtm_set_error(m, st, "инструкция %u (%s): %s", idx,
                  vtm_op_name(m->instr[idx].opcode), buf);
    return st;
}

static const vtm_tensor *need(vtm_t *m, uint32_t idx, uint32_t ins,
                              const char *role, vtm_status *st) {
    if (idx == VTM_NO_TENSOR) {
        *st = fail(m, ins, VTM_E_TENSOR, "поле %s не задано", role);
        return NULL;
    }
    if (idx >= m->n_tensors) {
        *st = fail(m, ins, VTM_E_TENSOR, "поле %s: индекс %u вне таблицы",
                   role, idx);
        return NULL;
    }
    *st = VTM_OK;
    return &m->tensors[idx];
}

static vtm_status req_f32(vtm_t *m, const vtm_tensor *t, uint32_t ins,
                          const char *role) {
    if (!t || t->kind < VTM_TENSOR_F32 || t->kind > VTM_TENSOR_BF16 || !t->f32)
        return fail(m, ins, VTM_E_TENSOR,
                    "%s: нужен fp32/fp16/bf16-тензор", role);
    return VTM_OK;
}

static vtm_status req_trit(vtm_t *m, const vtm_tensor *t, uint32_t ins,
                           const char *role) {
    if (!t || t->kind != VTM_TENSOR_TRIT || !t->words || !t->scales)
        return fail(m, ins, VTM_E_TENSOR, "%s: нужен тритный тензор", role);
    return VTM_OK;
}

// --- kind 5 (Q4), задача 0152 ----------------------------------------------
//
// Контракт — kind 5 (VTM_TENSOR_Q4), 4 бита, группы 32, масштаб fp32 на
// группу. Раскладка нибблов — GGML Q4_0 (см. vtm_backend.h), указатель на
// байты — t->words; fallback на t->f32 оставлен на случай, если загрузчик
// положит упакованные данные туда.
static const uint8_t *q4_bytes(const vtm_tensor *t) {
    if (t->words) return (const uint8_t *)t->words;
    return (const uint8_t *)t->f32;
}

static vtm_status req_q4(vtm_t *m, const vtm_tensor *t, uint32_t ins,
                         const char *role) {
    uint64_t groups;
    if (!t || t->kind != VTM_TENSOR_Q4 || !t->scales || !q4_bytes(t))
        return fail(m, ins, VTM_E_TENSOR,
                    "%s: нужен Q4-тензор (kind 5): байты и масштабы", role);
    // Q4_1 без минимумов — испорченный формат. Считать его симметрично
    // нельзя: логиты разойдутся молча (задача 0152).
    if (t->q4_is_asym && !t->q4_mins)
        return fail(m, ins, VTM_E_FORMAT,
                    "%s: Q4_1 без минимумов (q4_is_asym=1, q4_mins=NULL)",
                    role);
    if (t->scale_block != 32)
        return fail(m, ins, VTM_E_FORMAT,
                    "%s: Q4: блок масштаба %u, ожидали 32", role,
                    (unsigned)t->scale_block);
    groups = ((uint64_t)t->n_elements + 31u) / 32u;
    if ((uint64_t)t->n_scales < groups)
        return fail(m, ins, VTM_E_FORMAT,
                    "%s: Q4: масштабов %u, нужно %llu (групп по 32)", role,
                    (unsigned)t->n_scales, (unsigned long long)groups);
    return VTM_OK;
}

static vtm_status same_count(vtm_t *m, const vtm_tensor *x, const vtm_tensor *y,
                             uint32_t ins, const char *role) {
    if (x->n_elements != y->n_elements)
        return fail(m, ins, VTM_E_SHAPE,
                    "%s: элементов %u, ожидали %u",
                    role, y->n_elements, x->n_elements);
    return VTM_OK;
}

// --- LUT243: чтение тритов без делений на 3 --------------------------------
//
// Слово pack20 — 20 тритов; оно делится на 4 группы по 5 тритов, каждая
// группа — число 0..242 (база 3). Таблица даёт сразу пять тритов группы.
// Те же 243 записи, что в backend_scalar.c, но там она static и наружу не
// отдаётся, а EMB/MM живут здесь: держим свою копию (1,2 КБ) и инициализируем
// лениво при первом тритовом EMB/MM.
static int8_t g_lut243[243][5];
static int g_lut243_ready;

static void lut243_init(void) {
    unsigned v;
    int k;
    if (g_lut243_ready) return;
    for (v = 0; v < 243u; v++) {
        unsigned x = v;
        for (k = 0; k < 5; k++) {
            unsigned d = x % 3u;  // 0 -> -1, 1 -> 0, 2 -> +1
            g_lut243[v][k] = (int8_t)(d == 2u ? 1 : (d == 0u ? -1 : 0));
            x /= 3u;
        }
    }
    g_lut243_ready = 1;
}

// Трит номер index из потока слов тритового тензора.
static int trit_at(const vtm_tensor *t, size_t index) {
    size_t local = index % (size_t)VTM_TRITS_PER_WORD;
    uint32_t word = t->words[index / (size_t)VTM_TRITS_PER_WORD];
    unsigned group = (unsigned)(local / 5u);
    unsigned g;
    for (g = 0; g < group; g++) word /= 243u;
    return (int)g_lut243[word % 243u][local % 5u];
}

// w = trit * scale блока. 0 — успех, -1 — индекс или масштаб вне тензора.
static int trit_weight_at(const vtm_tensor *t, size_t index, float *out) {
    size_t scale_index;
    if (!t->words || !t->scales || t->scale_block == 0) return -1;
    if (index >= (size_t)t->n_elements) return -1;
    scale_index = index / (size_t)t->scale_block;
    if (scale_index >= (size_t)t->n_scales) return -1;
    *out = (float)trit_at(t, index) * t->scales[scale_index];
    return 0;
}

// --- плотность: плотный GEMV (n == 1) считает бэкенд, задача 0159 --------
//
// Раньше один столбец (tied lm_head: A = emb.w [vocab, hidden], B = [hidden,1];
// на 3B Q4 это 311М MAC и ~треть времени декода) считался здесь, intrinsics
// NEON жили в этом файле — слой был нарушен. Теперь ядро (NEON/AVX2, с
// переносимым откатом в backend_scalar.c) — vtm_simd_gemv_f32() в
// vtm_backend.h: interp.c про ISA не знает.
//
// simd_mm_enabled() — прежнее условие подключения SIMD-ядер: бэкенд `simd` и
// доступный SIMD. На скалярном бэкенде ниже остаётся прежний double-цикл,
// побитовый прогону по токену.
static int simd_mm_enabled(const vtm_t *m) {
    return m && m->backend && m->backend->name
        && strncmp(m->backend->name, "simd", 4) == 0
        && vtm_simd_available();
}


// --- TMV -------------------------------------------------------------------

static vtm_status op_tmv(vtm_t *m, uint32_t ins, const vtm_instr *in) {
    vtm_status st;
    const vtm_tensor *w = need(m, in->a, ins, "a (W)", &st);
    const vtm_tensor *x;
    const vtm_tensor *y;
    int rows, cols, r, t;
    int (*fn)(float *, const uint8_t *, const float *, const float *, int, int);
    int (*fn_blocks)(float *, const uint8_t *, const float *, const float *,
                     int, int, int);
    int rc;

    if (!w) return st;
    x = need(m, in->b, ins, "b (x)", &st);
    if (!x) return st;
    y = need(m, in->dst, ins, "dst (y)", &st);
    if (!y) return st;
    if ((st = req_f32(m, x, ins, "b (x)")) != VTM_OK) return st;
    if ((st = req_f32(m, y, ins, "dst (y)")) != VTM_OK) return st;
    if (w->ndim != 2)
        return fail(m, ins, VTM_E_SHAPE, "W: нужна матрица [rows, cols]");
    if (x == y)
        return fail(m, ins, VTM_E_SHAPE,
                    "dst и x — один тензор; TMV не умеет in-place");

    // Q4 (kind 5): веса остаются упакованными. Ни арены, ни fp32-копии —
    // ядро читает строку по 16 байт на 32 значения, деквантизует в scratch
    // и сразу умножает (задача 0152).
    if (w->kind == VTM_TENSOR_Q4) {
        if ((in->flags & VTM_FLAG_TMV_TRANSPOSED) != 0)
            return fail(m, ins, VTM_E_UNSUPPORTED,
                        "TMV: Q4 с VTM_FLAG_TMV_TRANSPOSED не поддержан");
        if ((st = req_q4(m, w, ins, "a (W)")) != VTM_OK) return st;
        rows = (int)w->dims[0];
        cols = (int)w->dims[1];
        if (x->n_elements != (uint32_t)cols)
            return fail(m, ins, VTM_E_SHAPE,
                        "x: элементов %u, у W cols=%d", x->n_elements, cols);
        if (y->n_elements != (uint32_t)rows)
            return fail(m, ins, VTM_E_SHAPE,
                        "y: элементов %u, у W rows=%d", y->n_elements, rows);
        if (vtm_q4_tmv(y->f32, q4_bytes(w), w->scales,
                       w->q4_is_asym ? w->q4_mins : NULL, x->f32, rows, cols,
                       w->q4_is_asym ? 1 : 0) != 0)
            return fail(m, ins, VTM_E_TENSOR, "ядро Q4 TMV вернуло ошибку");
        return VTM_OK;
    }

    // Плотный TMV: y = W·x. Нужен для нетернарных весов (эмбеддинги,
    // lm_head при tied_embeddings), которым в тритной матрице места нет.
    // VTM_FLAG_TMV_TRANSPOSED читает a как [cols, rows] и считает Wᵀ·x —
    // так tied lm_head может ссылаться на таблицу эмбеддингов в её
    // «выходной» ориентации.
    if (w->kind != VTM_TENSOR_TRIT) {
        int transposed = (in->flags & VTM_FLAG_TMV_TRANSPOSED) != 0;
        if (w->kind < VTM_TENSOR_F32 || w->kind > VTM_TENSOR_BF16 || !w->f32)
            return fail(m, ins, VTM_E_TENSOR,
                        "a (W): нужен тритный или fp32/fp16/bf16-тензор");
        rows = transposed ? (int)w->dims[1] : (int)w->dims[0];
        cols = transposed ? (int)w->dims[0] : (int)w->dims[1];
        if (x->n_elements != (uint32_t)cols)
            return fail(m, ins, VTM_E_SHAPE,
                        "x: элементов %u, у W cols=%d", x->n_elements, cols);
        if (y->n_elements != (uint32_t)rows)
            return fail(m, ins, VTM_E_SHAPE,
                        "y: элементов %u, у W rows=%d", y->n_elements, rows);

        // SIMD-бэкенд: одно-столбцовый GEMV (vtm_simd_gemv_f32, backend_neon.c/
        // backend_avx2.c). Точность — не побитовая, ≤2e-6 на logits 3B Q4; на
        // скалярном бэкенде ниже остаётся прежний double-цикл.
        if (simd_mm_enabled(m) &&
            vtm_simd_gemv_f32(y->f32, w->f32, x->f32, rows, cols,
                              transposed) == 0)
            return VTM_OK;

        for (r = 0; r < rows; r++) {
            double sum = 0.0;
            for (t = 0; t < cols; t++) {
                double wv = transposed
                          ? (double)w->f32[(size_t)t * (size_t)rows + (size_t)r]
                          : (double)w->f32[(size_t)r * (size_t)cols + (size_t)t];
                sum += wv * (double)x->f32[t];
            }
            y->f32[r] = (float)sum;
        }
        return VTM_OK;
    }

    if ((st = req_trit(m, w, ins, "a (W)")) != VTM_OK) return st;
    rows = (int)w->dims[0];
    cols = (int)w->dims[1];
    if (x->n_elements != (uint32_t)cols)
        return fail(m, ins, VTM_E_SHAPE,
                    "x: элементов %u, у W cols=%d", x->n_elements, cols);
    if (y->n_elements != (uint32_t)rows)
        return fail(m, ins, VTM_E_SHAPE,
                    "y: элементов %u, у W rows=%d", y->n_elements, rows);

    if (w->scale_block != VTM_SCALE_BLOCK_128 &&
        w->scale_block != VTM_SCALE_BLOCK_160)
        return fail(m, ins, VTM_E_FORMAT,
                    "TMV: неизвестный блок масштаба %u", w->scale_block);
    fn_blocks = m->backend ? m->backend->tmv_blocks : NULL;
    fn = m->backend ? m->backend->tmv : NULL;
    if (fn_blocks) {
        rc = fn_blocks(y->f32, (const uint8_t *)w->words, w->scales, x->f32,
                       rows, cols, (int)w->scale_block);
    } else if (fn && w->scale_block == VTM_SCALE_BLOCK_160) {
        rc = fn(y->f32, (const uint8_t *)w->words, w->scales, x->f32,
                rows, cols);
    } else {
        // Кастомный бэкенд без tmv_blocks умеет только 160; для 128 берём
        // скалярный LUT, иначе результат молча разойдётся.
        fn_blocks = vtm_backend_scalar_lut.tmv_blocks;
        rc = fn_blocks(y->f32, (const uint8_t *)w->words, w->scales, x->f32,
                       rows, cols, (int)w->scale_block);
    }
    if (rc != 0)
        return fail(m, ins, VTM_E_TENSOR, "ядро TMV вернуло ошибку");
    return VTM_OK;
}

// --- MM --------------------------------------------------------------------
//
// Y = A·B. A может быть тритовой матрицей (tied lm_head: A = emb.w,
// B = [hidden, 1]): триты расквантовываются при чтении (LUT243), копии
// матрицы в fp32 не делаем — у словаря Qwen это 233 М элементов.
// SIMD-ядра MM подключаются, только если выбран SIMD-бэкенд и он доступен:
// результат тот же (порядок сложения иной, расхождение ~1e-7), но в 6–15 раз быстрее.
// Для n == 1 (tied lm_head) зовём vtm_simd_gemv_f32(): у vtm_simd_mm_f32 на
// одном столбце векторизации нет (см. vtm_backend.h).
static vtm_status op_mm(vtm_t *m, uint32_t ins, const vtm_instr *in) {
    vtm_status st;
    const vtm_tensor *a = need(m, in->a, ins, "a (A)", &st);
    const vtm_tensor *b;
    const vtm_tensor *y;
    int mm, kk, nn, r, c, t;
    int a_trit, a_q4;

    if (!a) return st;
    b = need(m, in->b, ins, "b (B)", &st);
    if (!b) return st;
    y = need(m, in->dst, ins, "dst (Y)", &st);
    if (!y) return st;
    a_trit = a->kind == VTM_TENSOR_TRIT;
    a_q4 = a->kind == VTM_TENSOR_Q4;
    if (a_q4) {
        if ((st = req_q4(m, a, ins, "a (A)")) != VTM_OK) return st;
    } else if (a_trit) {
        if (!a->words || !a->scales)
            return fail(m, ins, VTM_E_TENSOR,
                        "a (A): тритовая матрица без слов/масштабов");
        if (a->scale_block != VTM_SCALE_BLOCK_128 &&
            a->scale_block != VTM_SCALE_BLOCK_160)
            return fail(m, ins, VTM_E_FORMAT,
                        "MM: неизвестный блок масштаба %u", a->scale_block);
    } else if ((st = req_f32(m, a, ins, "a (A)")) != VTM_OK) {
        return st;
    }
    if ((st = req_f32(m, b, ins, "b (B)")) != VTM_OK) return st;
    if ((st = req_f32(m, y, ins, "dst (Y)")) != VTM_OK) return st;
    if (a->ndim != 2 || b->ndim != 2 || y->ndim != 2)
        return fail(m, ins, VTM_E_SHAPE, "MM: нужны матрицы 2D");
    mm = (int)a->dims[0];
    kk = (int)a->dims[1];
    nn = (int)b->dims[1];
    if (b->dims[0] != (uint32_t)kk)
        return fail(m, ins, VTM_E_SHAPE,
                    "MM: A=[%d,%d], B=[%u,%u]", mm, kk,
                    (unsigned)b->dims[0], (unsigned)b->dims[1]);
    if (y->dims[0] != (uint32_t)mm || y->dims[1] != (uint32_t)nn)
        return fail(m, ins, VTM_E_SHAPE,
                    "MM: Y=[%u,%u], ожидали [%d,%d]",
                    (unsigned)y->dims[0], (unsigned)y->dims[1], mm, nn);
    if (y == a || y == b)
        return fail(m, ins, VTM_E_SHAPE, "MM: dst совпадает с входом");

    if (a_q4) {
        // Q4 A (kind 5): не разворачиваем матрицу в fp32 — ядро деквантизует
        // строку кусками и сразу считает. n == 1 (tied lm_head) — типовой
        // случай; общий n тоже поддержан (дороже по распаковке).
        if (vtm_q4_mm(y->f32, q4_bytes(a), a->scales,
                      a->q4_is_asym ? a->q4_mins : NULL, b->f32, mm, kk, nn,
                      a->q4_is_asym ? 1 : 0) != 0)
            return fail(m, ins, VTM_E_TENSOR, "ядро Q4 MM вернуло ошибку");
        return VTM_OK;
    }

    if (a_trit) {
        if (simd_mm_enabled(m) &&
            vtm_simd_mm_trit(y->f32, (const uint8_t *)a->words, a->scales,
                             b->f32, mm, kk, nn, (int)a->scale_block) == 0)
            return VTM_OK;
        lut243_init();
        for (r = 0; r < mm; r++) {
            for (c = 0; c < nn; c++) {
                double sum = 0.0;
                for (t = 0; t < kk; t++) {
                    size_t index = (size_t)r * (size_t)kk + (size_t)t;
                    float wv;
                    if (trit_weight_at(a, index, &wv) != 0)
                        return fail(m, ins, VTM_E_BOUNDS,
                                    "MM: элемент %zu матрицы A вне "
                                    "таблицы/масштабов", index);
                    sum += (double)wv
                         * (double)b->f32[(size_t)t * (size_t)nn + (size_t)c];
                }
                y->f32[(size_t)r * (size_t)nn + (size_t)c] = (float)sum;
            }
        }
        return VTM_OK;
    }

    if (simd_mm_enabled(m)) {
        if (nn == 1) {
            // tied lm_head: один столбец — отдельное GEMV-ядро (0159).
            if (vtm_simd_gemv_f32(y->f32, a->f32, b->f32, mm, kk, 0) == 0)
                return VTM_OK;
        } else if (vtm_simd_mm_f32(y->f32, a->f32, b->f32, mm, kk, nn) == 0) {
            return VTM_OK;
        }
    }

    for (r = 0; r < mm; r++) {
        for (c = 0; c < nn; c++) {
            double sum = 0.0;
            for (t = 0; t < kk; t++) {
                sum += (double)a->f32[(size_t)r * (size_t)kk + (size_t)t]
                     * (double)b->f32[(size_t)t * (size_t)nn + (size_t)c];
            }
            y->f32[(size_t)r * (size_t)nn + (size_t)c] = (float)sum;
        }
    }
    return VTM_OK;
}

// --- RMSN ------------------------------------------------------------------

// Число из меты контейнера. Метa — плоский JSON, ключи наших настроек уникальны,
// достаточно первого вхождения. Нужно, чтобы eps нормализации доходил точно:
// imm инструкции хранит eps/1000, и Qwen-овский 1e-6 так округляется в ноль.
static double meta_number(const vtm_t *m, const char *key, double fallback) {
    const char *json = vtm_meta_json(m);
    const char *hit;
    char needle[64];
    size_t klen = strlen(key);

    if (!json || klen + 3u >= sizeof needle) return fallback;
    needle[0] = '"';
    memcpy(needle + 1, key, klen);
    needle[klen + 1u] = '"';
    needle[klen + 2u] = '\0';
    hit = strstr(json, needle);
    if (!hit) return fallback;
    hit += klen + 2u;
    while (*hit == ' ' || *hit == '\t' || *hit == '\n' || *hit == '\r') hit++;
    if (*hit != ':') return fallback;
    hit++;
    while (*hit == ' ' || *hit == '\t' || *hit == '\n' || *hit == '\r') hit++;
    {
        char *end = NULL;
        double value = strtod(hit, &end);
        if (end == hit) return fallback;
        return value;
    }
}

static vtm_status op_rmsn(vtm_t *m, uint32_t ins, const vtm_instr *in) {
    vtm_status st;
    const vtm_tensor *x = need(m, in->a, ins, "a (x)", &st);
    const vtm_tensor *w;
    const vtm_tensor *y;
    int n;
    float eps;
    int (*fn)(float *, const float *, const float *, int, float);

    if (!x) return st;
    w = need(m, in->b, ins, "b (w)", &st);
    if (!w) return st;
    y = need(m, in->dst, ins, "dst (y)", &st);
    if (!y) return st;
    if ((st = req_f32(m, x, ins, "a (x)")) != VTM_OK) return st;
    if ((st = req_f32(m, w, ins, "b (w)")) != VTM_OK) return st;
    if ((st = req_f32(m, y, ins, "dst (y)")) != VTM_OK) return st;
    if ((st = same_count(m, x, w, ins, "b (w)")) != VTM_OK) return st;
    if ((st = same_count(m, x, y, ins, "dst (y)")) != VTM_OK) return st;
    if (in->imm < 0)
        return fail(m, ins, VTM_E_ARG, "RMSN: eps не может быть отрицательным");
    n = (int)x->n_elements;
    // Приоритет у меты: компилятор кладёт туда точный eps (1e-6). imm/1000 —
    // совместимость со старыми контейнерами, где 1e-6 округлялось в 0.
    eps = (float)meta_number(m, "rms_norm_eps", (double)in->imm / 1000.0);
    fn = m->backend && m->backend->rmsnorm ? m->backend->rmsnorm
                                           : vtm_backend_scalar_lut.rmsnorm;
    if (!fn) return fail(m, ins, VTM_E_UNSUPPORTED, "нет ядра RMSN");
    if (fn(y->f32, x->f32, w->f32, n, eps) != 0)
        return fail(m, ins, VTM_E_TENSOR, "ядро RMSN вернуло ошибку");
    return VTM_OK;
}

// --- ROPE ------------------------------------------------------------------
//
// Стандартный RoPE из Llama/HF (rotate_half): для головы h и пары
// (i, i + head_dim/2)
//     theta = pos · base^(-2i/head_dim)
//     y[i]          = x[i]·cos - x[i+half]·sin
//     y[i+half]     = x[i+half]·cos + x[i]·sin.
// base (rope_theta) берётся из меты контейнера: у Qwen2.5 это 1e6, у
// Llama-классики — 1e4. Без меты/с битой метой — 10000, как в v0.1.
// Раскладку задаёт мета `rope_layout` ("half" по умолчанию, "interleaved" —
// GPT-NeoX); флаг VTM_FLAG_ROPE_INTERLEAVED тоже включает интерливинг и
// перекрывает мету для конкретной инструкции.
static vtm_status op_rope(vtm_t *m, uint32_t ins, const vtm_instr *in) {
    vtm_status st;
    const vtm_tensor *x = need(m, in->a, ins, "a (x)", &st);
    const vtm_tensor *y;
    int heads, head_dim, half, h, p;
    long long pos;
    int interleaved;
    double theta;

    if (!x) return st;
    y = need(m, in->dst, ins, "dst (y)", &st);
    if (!y) return st;
    if ((st = req_f32(m, x, ins, "a (x)")) != VTM_OK) return st;
    if ((st = req_f32(m, y, ins, "dst (y)")) != VTM_OK) return st;
    if ((st = same_count(m, x, y, ins, "dst (y)")) != VTM_OK) return st;
    heads = (int)in->b;
    if (heads <= 0)
        return fail(m, ins, VTM_E_ARG, "ROPE: heads (поле b) должно быть > 0");
    if ((x->n_elements % (uint32_t)heads) != 0)
        return fail(m, ins, VTM_E_SHAPE,
                    "ROPE: %u элементов не делятся на %d голов",
                    x->n_elements, heads);
    head_dim = (int)(x->n_elements / (uint32_t)heads);
    if (head_dim < 2 || (head_dim & 1))
        return fail(m, ins, VTM_E_SHAPE,
                    "ROPE: head_dim=%d должен быть чётным и >= 2", head_dim);
    half = head_dim / 2;
    pos = (long long)m->position + (long long)in->imm;
    theta = vtm_meta_rope_theta(m);
    if (!(theta > 0.0) || !isfinite(theta)) theta = VTM_DEFAULT_ROPE_THETA;
    interleaved = ((in->flags & VTM_FLAG_ROPE_INTERLEAVED) != 0)
               || (vtm_meta_rope_layout(m) == VTM_ROPE_LAYOUT_INTERLEAVED);

    for (p = 0; p < half; p++) {
        double inv = pow(theta, -2.0 * (double)p / (double)head_dim);
        double angle = (double)pos * inv;
        double co = cos(angle);
        double si = sin(angle);
        for (h = 0; h < heads; h++) {
            size_t base = (size_t)h * (size_t)head_dim;
            size_t i0 = base + (interleaved ? (size_t)(2 * p) : (size_t)p);
            size_t i1 = base + (interleaved ? (size_t)(2 * p + 1)
                                            : (size_t)(p + half));
            double x0 = (double)x->f32[i0];
            double x1 = (double)x->f32[i1];
            y->f32[i0] = (float)(x0 * co - x1 * si);
            y->f32[i1] = (float)(x1 * co + x0 * si);
        }
    }
    return VTM_OK;
}

// --- SM --------------------------------------------------------------------
//
// Softmax по последней оси. CAUSAL оставляет в строке r элементы
// j <= r + (m->position + imm); остальные обнуляются после softmax.
// Для обычной полной последовательности position=0, imm=0 — маска j <= r.
static vtm_status op_sm(vtm_t *m, uint32_t ins, const vtm_instr *in) {
    vtm_status st;
    const vtm_tensor *x = need(m, in->a, ins, "a (x)", &st);
    const vtm_tensor *y;
    int rows, cols, causal, offset;
    int (*fn)(float *, const float *, int, int, int, int);

    if (!x) return st;
    y = need(m, in->dst, ins, "dst (y)", &st);
    if (!y) return st;
    if ((st = req_f32(m, x, ins, "a (x)")) != VTM_OK) return st;
    if ((st = req_f32(m, y, ins, "dst (y)")) != VTM_OK) return st;
    if ((st = same_count(m, x, y, ins, "dst (y)")) != VTM_OK) return st;
    cols = (int)x->dims[x->ndim - 1];
    if (cols <= 0)
        return fail(m, ins, VTM_E_SHAPE, "SM: последняя ось пустая");
    rows = (int)(x->n_elements / (uint32_t)cols);
    if ((uint32_t)rows * (uint32_t)cols != x->n_elements)
        return fail(m, ins, VTM_E_SHAPE, "SM: размер не делится на cols");
    causal = (in->flags & VTM_FLAG_CAUSAL) != 0;
    offset = m->position + in->imm;
    fn = m->backend && m->backend->softmax ? m->backend->softmax
                                           : vtm_backend_scalar_lut.softmax;
    if (!fn) return fail(m, ins, VTM_E_UNSUPPORTED, "нет ядра SM");
    if (fn(y->f32, x->f32, rows, cols, causal, offset) != 0)
        return fail(m, ins, VTM_E_TENSOR, "ядро SM вернуло ошибку");
    return VTM_OK;
}

// --- поэлементные ----------------------------------------------------------

static vtm_status op_silu(vtm_t *m, uint32_t ins, const vtm_instr *in) {
    vtm_status st;
    const vtm_tensor *x = need(m, in->a, ins, "a (x)", &st);
    const vtm_tensor *y;
    int (*fn)(float *, const float *, int);
    if (!x) return st;
    y = need(m, in->dst, ins, "dst (y)", &st);
    if (!y) return st;
    if ((st = req_f32(m, x, ins, "a (x)")) != VTM_OK) return st;
    if ((st = req_f32(m, y, ins, "dst (y)")) != VTM_OK) return st;
    if ((st = same_count(m, x, y, ins, "dst (y)")) != VTM_OK) return st;
    fn = m->backend && m->backend->silu ? m->backend->silu
                                        : vtm_backend_scalar_lut.silu;
    if (!fn) return fail(m, ins, VTM_E_UNSUPPORTED, "нет ядра SILU");
    if (fn(y->f32, x->f32, (int)x->n_elements) != 0)
        return fail(m, ins, VTM_E_TENSOR, "ядро SILU вернуло ошибку");
    return VTM_OK;
}

static vtm_status op_binary(vtm_t *m, uint32_t ins, const vtm_instr *in,
                            int is_add) {
    vtm_status st;
    const vtm_tensor *a = need(m, in->a, ins, "a", &st);
    const vtm_tensor *b;
    const vtm_tensor *y;
    int (*fn)(float *, const float *, const float *, int);
    if (!a) return st;
    b = need(m, in->b, ins, "b", &st);
    if (!b) return st;
    y = need(m, in->dst, ins, "dst", &st);
    if (!y) return st;
    if ((st = req_f32(m, a, ins, "a")) != VTM_OK) return st;
    if ((st = req_f32(m, b, ins, "b")) != VTM_OK) return st;
    if ((st = req_f32(m, y, ins, "dst")) != VTM_OK) return st;
    if ((st = same_count(m, a, b, ins, "b")) != VTM_OK) return st;
    if ((st = same_count(m, a, y, ins, "dst")) != VTM_OK) return st;
    fn = is_add ? (m->backend && m->backend->add ? m->backend->add
                                                 : vtm_backend_scalar_lut.add)
                : (m->backend && m->backend->mul ? m->backend->mul
                                                 : vtm_backend_scalar_lut.mul);
    if (!fn) return fail(m, ins, VTM_E_UNSUPPORTED, "нет ядра ADD/MUL");
    if (fn(y->f32, a->f32, b->f32, (int)a->n_elements) != 0)
        return fail(m, ins, VTM_E_TENSOR, "ядро ADD/MUL вернуло ошибку");
    return VTM_OK;
}

static vtm_status op_scal(vtm_t *m, uint32_t ins, const vtm_instr *in) {
    vtm_status st;
    const vtm_tensor *x = need(m, in->a, ins, "a (x)", &st);
    const vtm_tensor *y;
    int (*fn)(float *, const float *, float, int);
    if (!x) return st;
    y = need(m, in->dst, ins, "dst (y)", &st);
    if (!y) return st;
    if ((st = req_f32(m, x, ins, "a (x)")) != VTM_OK) return st;
    if ((st = req_f32(m, y, ins, "dst (y)")) != VTM_OK) return st;
    if ((st = same_count(m, x, y, ins, "dst (y)")) != VTM_OK) return st;
    fn = m->backend && m->backend->scal ? m->backend->scal
                                        : vtm_backend_scalar_lut.scal;
    if (!fn) return fail(m, ins, VTM_E_UNSUPPORTED, "нет ядра SCAL");
    if (fn(y->f32, x->f32, (float)in->imm / 1000.0f,
           (int)x->n_elements) != 0)
        return fail(m, ins, VTM_E_TENSOR, "ядро SCAL вернуло ошибку");
    return VTM_OK;
}

static vtm_status op_cpy(vtm_t *m, uint32_t ins, const vtm_instr *in) {
    vtm_status st;
    const vtm_tensor *x = need(m, in->a, ins, "a (x)", &st);
    const vtm_tensor *y;
    int (*fn)(float *, const float *, int);
    if (!x) return st;
    y = need(m, in->dst, ins, "dst (y)", &st);
    if (!y) return st;
    if ((st = req_f32(m, x, ins, "a (x)")) != VTM_OK) return st;
    if ((st = req_f32(m, y, ins, "dst (y)")) != VTM_OK) return st;
    if ((st = same_count(m, x, y, ins, "dst (y)")) != VTM_OK) return st;
    fn = m->backend && m->backend->copy ? m->backend->copy
                                        : vtm_backend_scalar_lut.copy;
    if (!fn) return fail(m, ins, VTM_E_UNSUPPORTED, "нет ядра CPY");
    if (fn(y->f32, x->f32, (int)x->n_elements) != 0)
        return fail(m, ins, VTM_E_TENSOR, "ядро CPY вернуло ошибку");
    return VTM_OK;
}

// --- EMB -------------------------------------------------------------------
//
// imm >= 0 — статический token id из программы. С флагом VTM_FLAG_EMB_INPUT
// токен берётся из vtm_set_token(); это точка входа для хоста.
//
// Таблица может быть тритовой (quantize_embeddings: trit у экспорта):
// строка расквантовывается на месте через LUT243, y[i] = trit * scale.
// В fp32 таблицу не разворачиваем: у Qwen таблица словаря — сотни мегабайт.
static vtm_status op_emb(vtm_t *m, uint32_t ins, const vtm_instr *in) {
    vtm_status st;
    const vtm_tensor *e = need(m, in->a, ins, "a (E)", &st);
    const vtm_tensor *y;
    int token, dim;
    int (*fn)(float *, const float *, int, int);

    if (!e) return st;
    y = need(m, in->dst, ins, "dst (y)", &st);
    if (!y) return st;
    if ((st = req_f32(m, y, ins, "dst (y)")) != VTM_OK) return st;
    if (e->ndim != 2)
        return fail(m, ins, VTM_E_SHAPE, "EMB: E должен быть [vocab, dim]");
    token = (in->flags & VTM_FLAG_EMB_INPUT) ? m->input_token : in->imm;
    if (token < 0)
        return fail(m, ins, VTM_E_ARG,
                    "EMB: входной токен не задан (vtm_set_token)");
    if (token >= (int)e->dims[0])
        return fail(m, ins, VTM_E_SHAPE,
                    "EMB: token=%d вне vocab=%u", token,
                    (unsigned)e->dims[0]);
    dim = (int)e->dims[1];
    if (y->n_elements != (uint32_t)dim)
        return fail(m, ins, VTM_E_SHAPE,
                    "EMB: y=%u элементов, dim=%d", y->n_elements, dim);

    if (e->kind == VTM_TENSOR_Q4) {
        // Q4-таблица словаря остаётся упакованной: разворачиваем одну строку
        // в y (EMB и так копирует строку), в арену fp32-копии нет.
        size_t base;
        if ((st = req_q4(m, e, ins, "a (E)")) != VTM_OK) return st;
        base = (size_t)token * (size_t)dim;
        if (vtm_q4_dequant_row(y->f32, q4_bytes(e), e->scales,
                               e->q4_is_asym ? e->q4_mins : NULL,
                               (int64_t)base, dim,
                               e->q4_is_asym ? 1 : 0) != 0)
            return fail(m, ins, VTM_E_TENSOR,
                        "EMB: ядро Q4 вернуло ошибку");
        return VTM_OK;
    }

    if (e->kind == VTM_TENSOR_TRIT) {
        size_t base, i;
        if (!e->words || !e->scales)
            return fail(m, ins, VTM_E_TENSOR,
                        "EMB: тритовая таблица без слов/масштабов");
        if (e->scale_block != VTM_SCALE_BLOCK_128 &&
            e->scale_block != VTM_SCALE_BLOCK_160)
            return fail(m, ins, VTM_E_FORMAT,
                        "EMB: неизвестный блок масштаба %u", e->scale_block);
        lut243_init();
        base = (size_t)token * (size_t)dim;
        for (i = 0; i < (size_t)dim; i++) {
            float value;
            if (trit_weight_at(e, base + i, &value) != 0)
                return fail(m, ins, VTM_E_BOUNDS,
                            "EMB: элемент %zu вне таблицы/масштабов",
                            base + i);
            y->f32[i] = value;
        }
        return VTM_OK;
    }

    if ((st = req_f32(m, e, ins, "a (E)")) != VTM_OK) return st;
    fn = m->backend && m->backend->emb ? m->backend->emb
                                       : vtm_backend_scalar_lut.emb;
    if (!fn) return fail(m, ins, VTM_E_UNSUPPORTED, "нет ядра EMB");
    if (fn(y->f32, e->f32, dim, token) != 0)
        return fail(m, ins, VTM_E_TENSOR, "ядро EMB вернуло ошибку");
    return VTM_OK;
}

// --- окно KV-кэша ----------------------------------------------------------
//
// Эффективное окно attention — min(длина кэша, meta["context_window"]).
// Ключ context_window читается лениво при первом KVW/ATTN и запоминается в
// машине (m->ctx_window), чтобы не разбирать JSON в горячем цикле. Значение
// битое/<=0/больше длины кэша — окно равно длине кэша (прежнее поведение).
static uint32_t kv_window(vtm_t *m, uint32_t ctx) {
    if (m->ctx_window == 0) {
        double v = meta_number(m, "context_window", 0.0);
        if (isfinite(v) && v >= 1.0 && v <= (double)INT_MAX)
            m->ctx_window = (int)v;
        else
            m->ctx_window = -1;
    }
    if (m->ctx_window > 0 && (uint32_t)m->ctx_window < ctx)
        return (uint32_t)m->ctx_window;
    return ctx;
}

// --- KVW -------------------------------------------------------------------
//
// dst — K-кэш [ctx, kv_dim], a — K-вектор [kv_dim],
// b   — V-кэш [ctx, kv_dim], c — V-вектор [kv_dim], imm — позиция.
// Слой задаётся самими тензорами (у каждого слоя свой кэш), поэтому
// отдельного поля layer нет.
//
// Кэш — кольцо (задача 0076): токен глобальной позиции p пишется в слот
// p % win. Позиция >= win больше не ошибка — она затирает самый старый слот,
// а ATTN читает последние win ключей в хронологическом порядке. ROPE
// применяется к K/V до KVW по глобальной позиции, поэтому ротация его не
// портит: повёрнутые векторы просто переезжают в другой слот.
static vtm_status op_kvw(vtm_t *m, uint32_t ins, const vtm_instr *in) {
    vtm_status st;
    const vtm_tensor *kc = need(m, in->dst, ins, "dst (K-кэш)", &st);
    const vtm_tensor *kv;
    const vtm_tensor *vc;
    const vtm_tensor *vv;
    long long pos;
    size_t slot;
    uint32_t ctx, kv_dim, win;

    if (!kc) return st;
    kv = need(m, in->a, ins, "a (K)", &st);
    if (!kv) return st;
    vc = need(m, in->b, ins, "b (V-кэш)", &st);
    if (!vc) return st;
    vv = need(m, in->c, ins, "c (V)", &st);
    if (!vv) return st;
    if ((st = req_f32(m, kc, ins, "dst (K-кэш)")) != VTM_OK) return st;
    if ((st = req_f32(m, kv, ins, "a (K)")) != VTM_OK) return st;
    if ((st = req_f32(m, vc, ins, "b (V-кэш)")) != VTM_OK) return st;
    if ((st = req_f32(m, vv, ins, "c (V)")) != VTM_OK) return st;
    if (kc->ndim != 2 || vc->ndim != 2)
        return fail(m, ins, VTM_E_SHAPE, "KVW: кэши должны быть [ctx, kv_dim]");
    ctx = kc->dims[0];
    kv_dim = kc->dims[1];
    if (vc->dims[0] != ctx || vc->dims[1] != kv_dim)
        return fail(m, ins, VTM_E_SHAPE, "KVW: K- и V-кэш разной формы");
    if (kv->n_elements != kv_dim || vv->n_elements != kv_dim)
        return fail(m, ins, VTM_E_SHAPE,
                    "KVW: K/V должны быть длины %u", kv_dim);
    win = kv_window(m, ctx);
    pos = (long long)m->position + (long long)in->imm;
    if (pos < 0 || pos > (long long)INT_MAX)
        return fail(m, ins, VTM_E_BOUNDS,
                    "KVW: позиция %lld вне диапазона", pos);
    // Кэш префикса (задача 0081) живёт в этих же тензорах: запись на позицию
    // меньше длины префикса затирает его K/V, а запись за пределами окна
    // запускает кольцо, которое начнёт переиспользовать слоты префикса. В
    // обоих случаях снимок перестаёт описывать KV, и следующий
    // vtm_run_cached() обязан считать заново. Запись позиции >= len(prefix)
    // в пределах окна безопасна: она кладёт K/V продолжения, а нужные ATTN
    // ключи префикса лежат в других слотах (позиция < win не пересекается).
    if (m->cache_valid
        && (pos < (long long)m->cache_len
            || (unsigned long long)pos >= (unsigned long long)win))
        m->cache_valid = 0;
    slot = (size_t)(pos % (long long)win);
    memmove(kc->f32 + slot * kv_dim, kv->f32,
            (size_t)kv_dim * sizeof(float));
    memmove(vc->f32 + slot * kv_dim, vv->f32,
            (size_t)kv_dim * sizeof(float));
    return VTM_OK;
}

// --- ATTN ------------------------------------------------------------------
//
// Два раскладки:
//   1) обычная (b, c — 3D): Q [h, q_len, d], K/V [kv_h, kv_len, d],
//      out [h, q_len, d]; q_start = kv_len - q_len. CAUSAL: j <= q_start+i.
//   2) с кэшем (b, c — 2D): Q/out — 3D, K/V — [ctx, kv_h*d];
//      imm — смещение начала запроса, q_start = m->position + imm,
//      валидны ключи 0..q_start+q_len-1. GQA: голова h идёт в kv-голову
//      h / (h / kv_h).
//
// Окно (задача 0076). 2D-кэш — кольцо: KVW кладёт токен глобальной позиции p
// в слот p % win, а ATTN читает последние win ключей хронологически, начиная
// с max(0, hi - win + 1) для каждого запроса. Поэтому q_start >= ctx — не
// ошибка: диалог длиннее кэша продолжается, старейшие ключи вытесняются.
// win = min(ctx, context_window); без ключа — ctx.
//
// Масштаб — 1/sqrt(d). Softmax — онлайн, блоками по ATTN_KEY_BLOCK ключей:
// running max m, running sum l, накопитель acc[head_dim] в double. Память
// O(head_dim + блок), а не O(ctx): массива scores на весь ключевой диапазон
// нет. exp на ключ считается один раз; при росте максимума acc и l
// домножаются на exp(m_old - m_new).
#define ATTN_KEY_BLOCK 64

static vtm_status op_attn(vtm_t *m, uint32_t ins, const vtm_instr *in) {
    vtm_status st;
    const vtm_tensor *q = need(m, in->a, ins, "a (Q)", &st);
    const vtm_tensor *k;
    const vtm_tensor *v;
    const vtm_tensor *out;
    int plain, n_heads, n_kv_heads, q_len, head_dim, kv_len, ctx;
    int q_start, key_count, h, qi, b, d, dd, rep, causal, kv_dim;
    int win, lo, hi, jb, n;
    double scale, mrun, lrun;
    double sbuf[ATTN_KEY_BLOCK];
    double *scratch, *qv, *acc;
    const float *kbase, *vbase;
    size_t kstride, slot;

    if (!q) return st;
    k = need(m, in->b, ins, "b (K)", &st);
    if (!k) return st;
    v = need(m, in->c, ins, "c (V)", &st);
    if (!v) return st;
    out = need(m, in->dst, ins, "dst (out)", &st);
    if (!out) return st;
    if ((st = req_f32(m, q, ins, "a (Q)")) != VTM_OK) return st;
    if ((st = req_f32(m, k, ins, "b (K)")) != VTM_OK) return st;
    if ((st = req_f32(m, v, ins, "c (V)")) != VTM_OK) return st;
    if ((st = req_f32(m, out, ins, "dst (out)")) != VTM_OK) return st;
    if (q->ndim != 3 || out->ndim != 3)
        return fail(m, ins, VTM_E_SHAPE,
                    "ATTN: Q и out должны быть [heads, len, head_dim]");
    n_heads = (int)q->dims[0];
    q_len = (int)q->dims[1];
    head_dim = (int)q->dims[2];
    if (head_dim <= 0 || q_len <= 0 || n_heads <= 0)
        return fail(m, ins, VTM_E_SHAPE, "ATTN: пустая размерность Q");
    if (out->dims[0] != q->dims[0] || out->dims[1] != q->dims[1] ||
        out->dims[2] != q->dims[2])
        return fail(m, ins, VTM_E_SHAPE,
                    "ATTN: out должен совпадать по форме с Q");

    if (k->ndim == 3 && v->ndim == 3) {
        plain = 1;
        n_kv_heads = (int)k->dims[0];
        kv_len = (int)k->dims[1];
        if ((int)k->dims[2] != head_dim || (int)v->dims[2] != head_dim)
            return fail(m, ins, VTM_E_SHAPE,
                        "ATTN: head_dim K/V не совпадает с Q");
        if ((int)v->dims[0] != n_kv_heads || (int)v->dims[1] != kv_len)
            return fail(m, ins, VTM_E_SHAPE, "ATTN: K и V разной формы");
        if (q_len > kv_len)
            return fail(m, ins, VTM_E_SHAPE,
                        "ATTN: q_len=%d больше kv_len=%d", q_len, kv_len);
        q_start = kv_len - q_len;
        key_count = kv_len;
        kv_dim = n_kv_heads * head_dim;
        ctx = 0;
        win = 0;
    } else if (k->ndim == 2 && v->ndim == 2) {
        plain = 0;
        ctx = (int)k->dims[0];
        kv_dim = (int)k->dims[1];
        if ((int)v->dims[0] != ctx || (int)v->dims[1] != kv_dim)
            return fail(m, ins, VTM_E_SHAPE, "ATTN: K- и V-кэш разной формы");
        if (head_dim <= 0 || kv_dim % head_dim != 0)
            return fail(m, ins, VTM_E_SHAPE,
                        "ATTN: kv_dim=%d не делится на head_dim=%d",
                        kv_dim, head_dim);
        n_kv_heads = kv_dim / head_dim;
        if (n_kv_heads <= 0)
            return fail(m, ins, VTM_E_SHAPE, "ATTN: kv_heads=0");
        // Считаем в long long: malformed-контейнер с imm=INT_MAX раньше давал
        // signed overflow (q_start+q_len < 0), проверка "больше ctx" его пропускала,
        // и malloc получал отрицательную длину — SIGBUS. Фаззинг 0069 нашёл это.
        // Проверка q_start > INT_MAX - q_len закрывает и переполнение, и
        // заведомо битый imm; верхней границы "меньше ctx" больше нет — кэш
        // стал кольцом (0076), длинный диалог законен.
        {
            long long qs = (long long)m->position + (long long)in->imm;
            long long keys;

            if (qs < 0 || qs > (long long)INT_MAX - (long long)q_len)
                return fail(m, ins, VTM_E_BOUNDS,
                            "ATTN: q_start=%lld вне диапазона (q_len=%d)",
                            qs, q_len);
            keys = qs + (long long)q_len;
            q_start = (int)qs;
            key_count = (int)keys;
        }
        kv_len = ctx;
        win = (int)kv_window(m, (uint32_t)ctx);
    } else {
        return fail(m, ins, VTM_E_SHAPE,
                    "ATTN: K/V должны быть оба 3D (без кэша) или оба 2D (кэш)");
    }

    if (n_kv_heads <= 0 || n_heads % n_kv_heads != 0)
        return fail(m, ins, VTM_E_SHAPE,
                    "ATTN: heads=%d не делится на kv_heads=%d",
                    n_heads, n_kv_heads);
    rep = n_heads / n_kv_heads;
    if (out == q || out == k || out == v)
        return fail(m, ins, VTM_E_SHAPE, "ATTN: dst совпадает с входом");

    causal = (in->flags & VTM_FLAG_CAUSAL) != 0;
    scale = 1.0 / sqrt((double)head_dim);
    // Один буфер на вызов: q-строка в double (переиспользуется для всех
    // ключей) и накопитель выхода. Размер зависит от head_dim, не от ctx.
    scratch = (double *)malloc(2u * (size_t)head_dim * sizeof(double));
    if (!scratch)
        return fail(m, ins, VTM_E_NOMEM, "ATTN: нет памяти под аккумулятор");
    qv = scratch;
    acc = scratch + (size_t)head_dim;

    for (h = 0; h < n_heads; h++) {
        int kvh = h / rep;

        // База строк K/V своей kv-головы: при plain-раскладке строки лежат
        // подряд, в 2D-кэше между ними kv_dim чисел.
        if (plain) {
            kbase = k->f32 + (size_t)kvh * (size_t)kv_len * (size_t)head_dim;
            vbase = v->f32 + (size_t)kvh * (size_t)kv_len * (size_t)head_dim;
            kstride = (size_t)head_dim;
        } else {
            kbase = k->f32 + (size_t)kvh * (size_t)head_dim;
            vbase = v->f32 + (size_t)kvh * (size_t)head_dim;
            kstride = (size_t)kv_dim;
        }

        for (qi = 0; qi < q_len; qi++) {
            const float *qrow = q->f32
                + ((size_t)h * (size_t)q_len + (size_t)qi) * (size_t)head_dim;
            float *orow = out->f32
                + ((size_t)h * (size_t)q_len + (size_t)qi) * (size_t)head_dim;

            for (d = 0; d < head_dim; d++) qv[d] = (double)qrow[d];

            // Видимые ключи запроса qi: [lo, hi] в глобальных позициях.
            // plain — прежние границы; 2D-кэш дополнительно обрезается
            // окном win (causal-граница hi = q_start + qi).
            if (plain) {
                lo = 0;
                hi = key_count - 1;
                if (causal && q_start + qi < hi) hi = q_start + qi;
            } else {
                hi = causal ? q_start + qi : key_count - 1;
                if (hi > key_count - 1) hi = key_count - 1;
                lo = hi - win + 1;
                if (lo < 0) lo = 0;
            }
            if (hi < lo) {
                memset(orow, 0, (size_t)head_dim * sizeof(float));
                continue;
            }

            for (d = 0; d < head_dim; d++) acc[d] = 0.0;
            mrun = -INFINITY;
            lrun = 0.0;
            slot = plain ? 0u : (size_t)lo % (size_t)win;

            jb = lo;
            while (jb <= hi) {
                size_t bslot = slot;
                double bmax = -INFINITY;

                n = hi - jb + 1;
                if (n > ATTN_KEY_BLOCK) n = ATTN_KEY_BLOCK;

                // 1) Оценки Q·Kᵀ/√d блока и его максимум. Четыре независимых
                // аккумулятора — короче цепочка зависимостей. Порядок сложения
                // double другой, чем у прежнего скаляра; разница ~1e-16.
                for (b = 0; b < n; b++) {
                    const float *krow = kbase
                        + (size_t)(plain ? jb + b : (int)slot) * kstride;
                    double dot, d0 = 0.0, d1 = 0.0, d2 = 0.0, d3 = 0.0;

                    for (dd = 0; dd + 4 <= head_dim; dd += 4) {
                        d0 += qv[dd] * (double)krow[dd];
                        d1 += qv[dd + 1] * (double)krow[dd + 1];
                        d2 += qv[dd + 2] * (double)krow[dd + 2];
                        d3 += qv[dd + 3] * (double)krow[dd + 3];
                    }
                    dot = (d0 + d1) + (d2 + d3);
                    for (; dd < head_dim; dd++)
                        dot += qv[dd] * (double)krow[dd];
                    sbuf[b] = dot * scale;
                    if (sbuf[b] > bmax) bmax = sbuf[b];
                    if (!plain && ++slot == (size_t)win) slot = 0;
                }

                // 2) Онлайн-softmax: новый максимум блока — тянем acc и l.
                if (bmax > mrun) {
                    double corr = (mrun == -INFINITY) ? 0.0 : exp(mrun - bmax);
                    if (corr != 1.0) {
                        for (d = 0; d < head_dim; d++) acc[d] *= corr;
                        lrun *= corr;
                    }
                    mrun = bmax;
                }

                // 3) Веса блока и накопление V (снова с начала блока).
                if (!plain) slot = bslot;
                for (b = 0; b < n; b++) {
                    double p = exp(sbuf[b] - mrun);
                    const float *vrow = vbase
                        + (size_t)(plain ? jb + b : (int)slot) * kstride;

                    lrun += p;
                    if (p != 0.0) {
                        for (d = 0; d < head_dim; d++)
                            acc[d] += p * (double)vrow[d];
                    }
                    if (!plain && ++slot == (size_t)win) slot = 0;
                }

                jb += n;
            }

            if (!(lrun > 0.0) || !isfinite(lrun)) {
                memset(orow, 0, (size_t)head_dim * sizeof(float));
                continue;
            }
            for (d = 0; d < head_dim; d++) orow[d] = (float)(acc[d] / lrun);
        }
    }
    free(scratch);
    return VTM_OK;
}

// --- SAMP ------------------------------------------------------------------
//
// imm — номер стратегии (см. vtm.h):
//   0 VTM_SAMP_ARGMAX      — жадный argmax, ровно как в v0.1;
//   1 VTM_SAMP_TEMPERATURE — p = softmax(logits / temperature);
//   2 VTM_SAMP_TOPK        — температура, затем top_k старших;
//   3 VTM_SAMP_TOPP        — температура, затем ядро с суммой >= top_p.
// imm вне 0..3 — VTM_E_UNSUPPORTED (конвенция «imm/1000 = температура» из
// старого Python-эталона не поддерживается, см. docs/vtm.md).
//
// Параметры — из меты контейнера: temperature, top_k, top_p,
// repetition_penalty, repetition_window, sample_seed (vtm.h). Штраф за
// повторы действует во всех стратегиях и применяется до температуры: если
// токен встретился среди последних repetition_window сгенерированных, его
// логит делится на penalty (>0) или умножается на неё (<=0); повтор внутри
// окна штрафуется один раз — как в CTRL.
//
// ГПСЧ — xorshift64* с засевом splitmix64. Алгоритм зафиксирован в
// docs/vtm.md и повторён в tools/vtm_ref.py побитово: один сэмпл без
// перехода в argmax потребляет ровно одно 64-битное число, поэтому C и
// Python при одном seed дают одну и ту же последовательность токенов.
// Всё, от чего зависит выбор, считается в double — как в Python-эталоне.

// Кандидат: логит после штрафа и его токен. Порядок — по значению (убывание),
// при равенстве — по индексу (возрастание). Это ровно тот порядок, который
// задавал прежний qsort-компаратор: он виден снаружи, потому что сумма весов
// накапливается по убыванию, а от порядка сложений зависят последние биты
// (паритет C↔Python с tools/vtm_ref.py, задача 0051).
typedef struct {
    double   value;     // логит после штрафа (-0.0 приведён к +0.0)
    uint32_t index;     // токен
} samp_cand;

// x стоит в отсортированном массиве раньше y?
static int samp_before(const samp_cand *x, const samp_cand *y) {
    if (x->value > y->value) return 1;
    if (x->value < y->value) return 0;
    return x->index < y->index;
}

// Ключ сортировки: uint64, числовой порядок double становится лексикографи-
// ческим. Для убывания ключ инвертирован, поэтому обычный LSD-radix даёт
// «по убыванию значения»; равенство значений сохраняет исходный порядок
// индексов (radix устойчив, массив заполняется по возрастанию индекса).
static uint64_t samp_sort_key(double value) {
    uint64_t bits;
    memcpy(&bits, &value, sizeof bits);
    if (bits & 0x8000000000000000ull) bits = ~bits;
    else bits |= 0x8000000000000000ull;
    return ~bits;
}

// Сортировка вставками — для малых n и малого top-k: там она дешевле
// radix (у того на каждый проход приходится обнуление таблицы счётчиков).
static void samp_insertion_sort(samp_cand *a, uint32_t n) {
    uint32_t i;
    for (i = 1; i < n; i++) {
        samp_cand cur = a[i];
        uint32_t j = i;
        while (j > 0 && samp_before(&cur, &a[j - 1])) {
            a[j] = a[j - 1];
            j--;
        }
        a[j] = cur;
    }
}

// LSD-radix: линейное время вместо n·log n у qsort — это и есть основной
// выигрыш задачи 0059 на словаре 150k. Компаратор не нужен вовсе, а
// устойчивость проходов сохраняет порядок индексов внутри равных значений.
// Разрядность — параметр: 16 бит (4 прохода, счётчики 256 КБ из буфера
// машины) для больших словарей и 11 бит (6 проходов, 8 КБ на стеке) для
// маленьких, где 256 КБ на каждую машину были бы дороже самой сортировки.
// by_index: устойчивая сортировка по возрастанию индекса (нужна только
// префиксу top-k, см. samp_radix_sort_prefix); иначе — по значению.
static void samp_radix_sort_bits(samp_cand *a, samp_cand *tmp, uint32_t n,
                                 uint32_t *count, unsigned bits,
                                 int by_index) {
    const uint32_t size = 1u << bits;
    const uint32_t mask = size - 1u;
    const unsigned width = by_index ? 32u : 64u;
    const unsigned passes = (width + bits - 1u) / bits;
    unsigned pass;
    uint32_t i;
    samp_cand *src = a, *dst = tmp;

    for (pass = 0; pass < passes; pass++) {
        unsigned shift = pass * bits;
        memset(count, 0, (size_t)size * sizeof *count);
        for (i = 0; i < n; i++) {
            uint64_t key = by_index ? (uint64_t)src[i].index
                                    : samp_sort_key(src[i].value);
            count[(key >> shift) & mask]++;
        }
        {
            uint32_t sum = 0;
            for (i = 0; i < size; i++) {
                uint32_t c = count[i];
                count[i] = sum;
                sum += c;
            }
        }
        for (i = 0; i < n; i++) {
            uint64_t key = by_index ? (uint64_t)src[i].index
                                    : samp_sort_key(src[i].value);
            dst[count[(key >> shift) & mask]++] = src[i];
        }
        {
            samp_cand *t = src;
            src = dst;
            dst = t;
        }
    }
    if (src != a) memcpy(a, src, (size_t)n * sizeof *a);
}

#define SAMP_RADIX_SMALL_BITS 11u
#define SAMP_RADIX_SMALL_SIZE (1u << SAMP_RADIX_SMALL_BITS)

// Полный порядок: значение по убыванию, равные — по возрастанию индекса.
// Массив заполняется в порядке индексов, поэтому устойчивый radix по
// значению сразу даёт нужный порядок равных.
static void samp_radix_sort(vtm_t *m, samp_cand *a, samp_cand *tmp, uint32_t n) {
    if (n < 2) return;
    if (m->samp_count) {
        samp_radix_sort_bits(a, tmp, n, m->samp_count, 16u, 0);
    } else {
        uint32_t count[SAMP_RADIX_SMALL_SIZE];
        samp_radix_sort_bits(a, tmp, n, count, SAMP_RADIX_SMALL_BITS, 0);
    }
}

// Порядок префикса, отобранного quickselect: тот же (значение по убыванию,
// равные — по возрастанию индекса). Quickselect перемешивает равные
// значения, поэтому сначала устойчиво сортируем по индексу, а затем
// устойчиво по значению — сложение двух устойчивых сортировок даёт
// (значение, индекс). Без первой фазы равные логиты встали бы произвольно.
static void samp_radix_sort_prefix(vtm_t *m, samp_cand *a, samp_cand *tmp,
                                   uint32_t n) {
    if (n < 2) return;
    if (m->samp_count) {
        samp_radix_sort_bits(a, tmp, n, m->samp_count, 16u, 1);
        samp_radix_sort_bits(a, tmp, n, m->samp_count, 16u, 0);
    } else {
        uint32_t count[SAMP_RADIX_SMALL_SIZE];
        samp_radix_sort_bits(a, tmp, n, count, SAMP_RADIX_SMALL_BITS, 1);
        samp_radix_sort_bits(a, tmp, n, count, SAMP_RADIX_SMALL_BITS, 0);
    }
}

// Разбиение Ломуто: a[p] встаёт на своё место в полном порядке, слева —
// меньшие, справа — большие. Медиана из трёх снимает худший случай на
// отсортированных данных.
static uint32_t samp_partition(samp_cand *a, uint32_t lo, uint32_t hi) {
    uint32_t mid = lo + (hi - lo) / 2u, i, p = lo;
    samp_cand pv, t;

    if (samp_before(&a[mid], &a[lo])) { t = a[mid]; a[mid] = a[lo]; a[lo] = t; }
    if (samp_before(&a[hi - 1], &a[mid])) {
        t = a[hi - 1]; a[hi - 1] = a[mid]; a[mid] = t;
    }
    if (samp_before(&a[mid], &a[lo])) { t = a[mid]; a[mid] = a[lo]; a[lo] = t; }
    pv = a[lo];
    for (i = lo + 1; i < hi; i++) {
        if (samp_before(&a[i], &pv)) {
            p++;
            if (p != i) { t = a[p]; a[p] = a[i]; a[i] = t; }
        }
    }
    if (p != lo) { t = a[lo]; a[lo] = a[p]; a[p] = t; }
    return p;
}

// Частичный отбор: в a[0..k) — k старших кандидатов (порядок внутри любой).
// Число раундов ограничено: на патологии уходим в полную radix-сортировку,
// чтобы не поймать квадратичное время.
static void samp_select_prefix(vtm_t *m, samp_cand *a, samp_cand *tmp,
                               uint32_t n, uint32_t k) {
    uint32_t lo = 0, hi = n, rounds = 0, max_rounds = 64;

    while (lo < k && hi - lo > 1) {
        uint32_t p;
        if (++rounds > max_rounds) {
            samp_radix_sort(m, a, tmp, n);
            return;
        }
        p = samp_partition(a, lo, hi);
        if (p >= k) hi = p;
        else lo = p + 1;
    }
}

// splitmix64 — только для засева xorshift64*.
static uint64_t samp_mix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

static uint64_t samp_rng_next(vtm_t *m) {
    uint64_t x = m->samp_rng;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    m->samp_rng = x;
    return x * 0x2545F4914F6CDD1Dull;
}

// Равномерное [0,1): 53 старших бита (как double(random())).
static double samp_uniform(vtm_t *m) {
    return (double)(samp_rng_next(m) >> 11) * (1.0 / 9007199254740992.0);
}

// --- параметры из меты -----------------------------------------------------
//
// Свой маленький сканер, а не загрузчик: loader.c вне зоны задачи, а мета —
// тот же JSON, что уже лежит в машине. Ключ ищется на любом уровне
// вложенности, при нескольких вхождениях побеждает минимальная глубина
// (как у rope_theta в загрузчике). Числа читает strtod; мусор — дефолт.

static size_t samp_skip_ws(const char *s, size_t i, size_t n) {
    while (i < n && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n'
                     || s[i] == '\r'))
        i++;
    return i;
}

static int samp_meta_number(const char *json, const char *key, double *out) {
    size_t n, i = 0, klen = strlen(key);
    int depth = 0, best_depth = 0, found = 0;
    double best = 0.0;

    if (!json || !json[0]) return 0;
    n = strlen(json);
    while (i < n) {
        char c = json[i];
        if (c == '"') {
            size_t start = i + 1;
            int esc = 0, closed = 0;
            i = start;
            while (i < n) {
                if (esc) { esc = 0; i++; continue; }
                if (json[i] == '\\') { esc = 1; i++; continue; }
                if (json[i] == '"') { closed = 1; break; }
                i++;
            }
            if (closed && i - start == klen
                && memcmp(json + start, key, klen) == 0) {
                size_t j = samp_skip_ws(json, i + 1, n);
                if (j < n && json[j] == ':') {
                    char *end = NULL;
                    double v;
                    j = samp_skip_ws(json, j + 1, n);
                    v = strtod(json + j, &end);
                    if (end && end != json + j
                        && (!found || depth < best_depth)) {
                        best = v;
                        best_depth = depth;
                        found = 1;
                    }
                }
            }
            i = closed ? i + 1 : n;
            continue;
        }
        if (c == '{' || c == '[') depth++;
        else if ((c == '}' || c == ']') && depth > 0) depth--;
        i++;
    }
    if (found && out) *out = best;
    return found;
}

static void samp_params_from_meta(const vtm_t *m, vtm_sampling *p) {
    const char *json = vtm_meta_json(m);
    double v;

    p->temperature = VTM_DEFAULT_TEMPERATURE;
    p->top_k = 0;
    p->top_p = VTM_DEFAULT_TOP_P;
    p->repetition_penalty = VTM_DEFAULT_REPETITION_PENALTY;
    p->repetition_window = VTM_DEFAULT_REPETITION_WINDOW;
    p->sample_seed = 0;
    p->has_seed = 0;

    if (samp_meta_number(json, "temperature", &v)) {
        if (isnan(v) || v < 0.0) p->temperature = 0.0;   // мусор/минус — жадно
        else if (v > 1e6) p->temperature = 1e6;
        else p->temperature = v;
    }
    if (samp_meta_number(json, "top_k", &v)) {
        if (!(v > 0.0)) p->top_k = 0;
        else if (v > 1e9) p->top_k = 1000000000;
        else p->top_k = (int)v;
    }
    if (samp_meta_number(json, "top_p", &v)) {
        if (isnan(v)) p->top_p = VTM_DEFAULT_TOP_P;
        else if (v <= 0.0) p->top_p = 0.0;
        else if (v >= 1.0) p->top_p = 1.0;
        else p->top_p = v;
    }
    if (samp_meta_number(json, "repetition_penalty", &v)) {
        if (isnan(v) || v <= 0.0)
            p->repetition_penalty = VTM_DEFAULT_REPETITION_PENALTY;
        else if (v > 1e6) p->repetition_penalty = 1e6;
        else p->repetition_penalty = v;
    }
    if (samp_meta_number(json, "repetition_window", &v)) {
        if (!(v > 0.0)) p->repetition_window = 0;
        else if (v > (double)VTM_MAX_REPETITION_WINDOW)
            p->repetition_window = VTM_MAX_REPETITION_WINDOW;
        else p->repetition_window = (int)v;
    }
    if (samp_meta_number(json, "sample_seed", &v) && isfinite(v)
        && v >= 0.0 && v < 9007199254740992.0) {
        p->sample_seed = (uint64_t)v;
        p->has_seed = 1;
    }
}

int vtm_sampling_params(const vtm_t *m, vtm_sampling *out) {
    vtm_sampling p;
    if (!m) return 0;
    samp_params_from_meta(m, &p);
    if (out) *out = p;
    return vtm_meta_present(m) ? 1 : 0;
}

// --- состояние сэмплера ----------------------------------------------------

static void samp_state_seed(vtm_t *m) {
    m->samp_rng = samp_mix64(m->samp_seed ^ 0x9E3779B97F4A7C15ull);
    if (m->samp_rng == 0) m->samp_rng = 0x2545F4914F6CDD1Dull;
    m->samp_hist_len = 0;
    m->samp_hist_head = 0;
}

// Следующее поколение меток окна повторов. Ноль зарезервирован под «метки
// недействительны» (блок только выделен или освобождён), поэтому живое
// поколение всегда >= 1. Метки с меньшим поколением безопасны: сравнение с
// текущим их не узнаёт. Переполнение счётчика чистит метки — иначе старое
// значение снова стало бы «своим».
static uint32_t samp_stamp_next_gen(vtm_t *m) {
    uint32_t gen = m->samp_stamp_gen + 1u;

    if (gen == 0u || m->samp_stamp_gen == 0u) {
        if (m->samp_stamp)
            memset(m->samp_stamp, 0,
                   (size_t)m->samp_buf_cap * sizeof *m->samp_stamp);
        gen = 1u;
    }
    m->samp_stamp_gen = gen;
    return gen;
}

// Явный сброс состояния SAMP; вызывать из vtm_reset() (loader.c).
// Пока загрузчик этого не делает, op_samp() замечает сброс сам — по
// счётчику vtm_generated(), поэтому поведение одинаково в обоих случаях.
void vtm_samp_reset(vtm_t *m) {
    if (!m) return;
    m->samp_prev_generated = 0;
    if (m->samp_ready) samp_state_seed(m);
    // Метки окна не должны пережить сброс: сдвигаем поколение.
    if (m->samp_stamp) (void)samp_stamp_next_gen(m);
}

// Вызывается в начале каждого SAMP: ленивая инициализация параметров и
// распознавание vtm_reset() по счётчику сгенерированных токенов.
static void samp_state_begin(vtm_t *m) {
    vtm_sampling p;

    if (m->samp_ready) {
        if (m->samp_prev_generated > m->generated) samp_state_seed(m);
        return;
    }
    samp_params_from_meta(m, &p);
    m->samp_temperature = p.temperature;
    m->samp_top_k = p.top_k;
    m->samp_top_p = p.top_p;
    m->samp_repetition_penalty = p.repetition_penalty;
    m->samp_repetition_window = p.repetition_window;
    m->samp_seed = p.sample_seed;
    m->samp_ready = 1;
    samp_state_seed(m);
}

static void samp_hist_push(vtm_t *m, int token) {
    m->samp_hist[m->samp_hist_head] = token;
    m->samp_hist_head = (m->samp_hist_head + 1) % VTM_MAX_REPETITION_WINDOW;
    if (m->samp_hist_len < VTM_MAX_REPETITION_WINDOW) m->samp_hist_len++;
}

static int samp_hist_has(const vtm_t *m, uint32_t token) {
    int k, lim = m->samp_repetition_window;
    if (lim > m->samp_hist_len) lim = m->samp_hist_len;
    for (k = 0; k < lim; k++) {
        int idx = m->samp_hist_head - 1 - k;
        if (idx < 0) idx += VTM_MAX_REPETITION_WINDOW;
        if (m->samp_hist[idx] == (int)token) return 1;
    }
    return 0;
}

static int samp_penalty_active(const vtm_t *m) {
    return m->samp_repetition_penalty != VTM_DEFAULT_REPETITION_PENALTY
        && m->samp_repetition_window > 0
        && m->samp_hist_len > 0;
}

static double samp_penalized(const vtm_t *m, uint32_t token, double value) {
    if (!samp_penalty_active(m) || !samp_hist_has(m, token)) return value;
    return value > 0.0 ? value / m->samp_repetition_penalty
                       : value * m->samp_repetition_penalty;
}

// --- буфер кандидатов (задача 0059) ---------------------------------------
//
// Прежний код на каждый SAMP делал malloc под весь словарь (для Qwen 150k —
// ~3,6 МБ) и free в конце. Теперь блок живёт в машине: первое выделение,
// дальше только переиспользование; растём, если словарь оказался больше.
// Раскладка блока (выравнивание сохраняется, malloc даёт ≥16 байт):
//     [cand n × 16][tmp n × 16][stamp n × 4][счётчики radix 256 КБ]
// tmp нужен radix-сортировке, а после неё играет роль массива весов w
// (8n ≤ 16n), поэтому отдельного массива весов нет.
//
// Счётчики radix для большого словаря 16-битные (256 КБ): на 150k это четыре
// прохода вместо шести и вдвое меньше времени. Мелкому словарю они были бы
// дороже самой сортировки — там radix работает на 11 битах со стеком, а блок
// машины остаётся маленьким.
#define SAMP_BIG_VOCAB 4096u
#define SAMP_RADIX_COUNT 65536u

static int samp_buf_ensure(vtm_t *m, uint32_t n, uint32_t ins, vtm_status *st) {
    size_t base, bytes;
    void *p;

    if (m->samp_buf && m->samp_buf_cap >= n) return 1;
    base = (size_t)n * (2u * sizeof(samp_cand) + sizeof(uint32_t));
    bytes = base + (n >= SAMP_BIG_VOCAB
                    ? (size_t)SAMP_RADIX_COUNT * sizeof(uint32_t) : 0u);
    p = realloc(m->samp_buf, bytes);
    if (!p) {
        *st = fail(m, ins, VTM_E_NOMEM,
                   "SAMP: нет памяти под %u логитов (%lu байт)",
                   (unsigned)n, (unsigned long)bytes);
        return 0;
    }
    m->samp_buf = p;
    m->samp_buf_cap = n;
    m->samp_buf_bytes = bytes;
    m->samp_buf_allocs++;
    m->samp_w = (double *)(void *)((samp_cand *)p + n);
    m->samp_stamp = (uint32_t *)(void *)((uint8_t *)p
                                         + (size_t)n * 2u * sizeof(samp_cand));
    m->samp_count = (n >= SAMP_BIG_VOCAB)
        ? (uint32_t *)(void *)((uint8_t *)p + base) : NULL;
    // Блок приходит из malloc/realloc, а не из calloc, и после vtm_samp_free()
    // (его зовёт vtm_free()) та же память может достаться следующей машине с
    // метками прошлой. Обнуляем метки один раз на выделение: O(n), зато
    // дальше проверка — одно сравнение. Поколение стартует с 1: ноль — это
    // «метки недействительны», иначе нулевые метки выглядели бы как свои.
    memset(m->samp_stamp, 0, (size_t)n * sizeof *m->samp_stamp);
    m->samp_stamp_gen = 1;
    return 1;
}

void vtm_samp_free(vtm_t *m) {
    if (!m) return;
    free(m->samp_buf);
    m->samp_buf = NULL;
    m->samp_w = NULL;
    m->samp_stamp = NULL;
    m->samp_count = NULL;
    m->samp_buf_cap = 0;
    m->samp_buf_bytes = 0;
    m->samp_stamp_gen = 0;
}

size_t vtm_samp_alloc_count(const vtm_t *m) {
    return m ? m->samp_buf_allocs : 0;
}

size_t vtm_samp_buffer_bytes(const vtm_t *m) {
    return m ? m->samp_buf_bytes : 0;
}

// Тест-хук (задача 0051): сымитировать блок, доставшийся от прошлой машины.
// Заполняет все метки значением gen и сбрасывает счётчик поколений в 0 —
// ровно то состояние, в котором оказывается свежая машина на грязной памяти
// (malloc после vtm_samp_free). Норма: первый же SAMP метки чистит, поэтому
// ни один «чужой» токен не получает штраф.
void vtm_samp_poison_stamps(vtm_t *m, uint32_t gen) {
    uint32_t i;

    if (!m || !m->samp_stamp) return;
    for (i = 0; i < m->samp_buf_cap; i++) m->samp_stamp[i] = gen;
    m->samp_stamp_gen = 0;
}

// Метки окна повторов: поколение вместо линейного поиска по кольцу.
// Прежний samp_hist_has() на каждый токен пробегал до 256 записей истории —
// на словаре 150k это 38 млн сравнений на токен. Теперь окно один раз
// помечается в stamp[], а проверка — одно сравнение.
// Возвращает 1, если метки проставлены (штраф активен и буфер есть).
static int samp_stamp_fill(vtm_t *m, uint32_t n) {
    uint32_t gen;
    int k, lim;

    if (!samp_penalty_active(m)) return 0;
    if (!m->samp_buf || m->samp_buf_cap < n || !m->samp_stamp) return 0;
    // Метки обязаны быть чистыми перед первым использованием: блок приходит
    // из malloc/realloc, и в «мусоре» может лежать значение, равное новому
    // поколению, — тогда невинный токен получил бы штраф. samp_buf_ensure()
    // чистит их при выделении, helper страхует состояния «поколение 0»
    // (переиспользованная/испорченная память) и переполнение счётчика.
    gen = samp_stamp_next_gen(m);
    lim = m->samp_repetition_window;
    if (lim > m->samp_hist_len) lim = m->samp_hist_len;
    for (k = 0; k < lim; k++) {
        int idx = m->samp_hist_head - 1 - k;
        int token;
        if (idx < 0) idx += VTM_MAX_REPETITION_WINDOW;
        token = m->samp_hist[idx];
        // Токен из прошлого словаря может быть >= n: он не может совпасть
        // ни с одним j < n, поэтому просто не помечаем.
        if (token >= 0 && (uint32_t)token < n) m->samp_stamp[token] = gen;
    }
    return 1;
}

// Штраф по готовым меткам (samp_stamp_fill); token < n ≤ samp_buf_cap.
static double samp_penalized_stamped(const vtm_t *m, uint32_t token,
                                     double value) {
    if (m->samp_stamp[token] != m->samp_stamp_gen) return value;
    return value > 0.0 ? value / m->samp_repetition_penalty
                       : value * m->samp_repetition_penalty;
}

// Жадный argmax — байт-в-байт как в v0.1, если штраф выключен.
// stamped: метки окна уже проставлены (тогда штраф ищется за O(1)).
static int samp_greedy(const vtm_t *m, const float *v, uint32_t n, int stamped) {
    int best_i = 0;
    double best = (double)v[0];
    uint32_t j;

    if (samp_penalty_active(m)) {
        best = stamped ? samp_penalized_stamped(m, 0, best)
                       : samp_penalized(m, 0, best);
    }
    for (j = 1; j < n; j++) {
        double value = (double)v[j];
        if (samp_penalty_active(m)) {
            value = stamped ? samp_penalized_stamped(m, j, value)
                            : samp_penalized(m, j, value);
        }
        if (value > best) {
            best = value;
            best_i = (int)j;
        }
    }
    return best_i;
}

// Температура/top-k/top-p и одно случайное число.
//
// Порядок кандидатов — по убыванию логита (при равенстве по возрастанию
// индекса), потому что веса накапливаются именно в нём, и от порядка
// сложений зависят последние биты. Полная сортировка нужна только там, где
// сумма идёт по всему словарю (temperature и top-p); для top-k достаточно
// отобрать k старших. Поэтому вместо qsort: quickselect (+ вставки на малом
// k) для top-k и линейная LSD-radix — там, где порядок нужен целиком.
static vtm_status samp_sample(vtm_t *m, uint32_t ins, const float *v, uint32_t n,
                              int strategy, int *token) {
    samp_cand *cand, *tmp;
    double *w;
    uint32_t j, kept = n;
    double top = -INFINITY, total = 0.0, upto, acc, u;
    double temperature = m->samp_temperature;
    int penalty = samp_penalty_active(m);
    int stamped, best_i = 0;
    vtm_status st;

    if (!samp_buf_ensure(m, n, ins, &st)) return st;
    cand = (samp_cand *)m->samp_buf;
    tmp = cand + n;
    w = m->samp_w;
    stamped = samp_stamp_fill(m, n);

    for (j = 0; j < n; j++) {
        double value = (double)v[j];
        if (isnan(value)) value = -INFINITY;
        if (penalty) {
            value = stamped ? samp_penalized_stamped(m, j, value)
                            : samp_penalized(m, j, value);
        }
        // -0.0 и +0.0 численно равны, а ключ radix у них разный: приводим к
        // +0.0, чтобы равные значения упорядочивались по индексу, как в
        // прежнем компараторе.
        if (value == 0.0) value = 0.0;
        cand[j].value = value;
        cand[j].index = j;
        if (value > top) {
            top = value;
            best_i = (int)j;
        }
    }
    *token = best_i;
    if (!isfinite(top)) {          // все логиты -inf/NaN — как argmax по ним
        return VTM_OK;
    }

    if (strategy == VTM_SAMP_TOPK && m->samp_top_k > 0
        && (uint32_t)m->samp_top_k < kept) {
        kept = (uint32_t)m->samp_top_k;
        samp_select_prefix(m, cand, tmp, n, kept);
        if (kept <= 64) samp_insertion_sort(cand, kept);
        else samp_radix_sort_prefix(m, cand, tmp, kept);
    } else if (n <= 64) {
        samp_insertion_sort(cand, n);
    } else {
        samp_radix_sort(m, cand, tmp, n);
    }
    for (j = 0; j < kept; j++) {
        w[j] = exp((cand[j].value - top) / temperature);
        total += w[j];
    }
    if (strategy == VTM_SAMP_TOPP && m->samp_top_p < 1.0 && total > 0.0) {
        double need = m->samp_top_p * total;
        upto = 0.0;
        for (j = 0; j < kept; j++) {
            upto += w[j];
            if (upto >= need) { kept = j + 1; break; }
        }
        total = 0.0;
        for (j = 0; j < kept; j++) total += w[j];
    }
    if (total > 0.0 && isfinite(total)) {
        u = samp_uniform(m) * total;
        acc = 0.0;
        for (j = 0; j < kept; j++) {
            acc += w[j];
            if (acc >= u) { *token = (int)cand[j].index; break; }
        }
        if (j >= kept) *token = (int)cand[kept - 1].index;
    }
    return VTM_OK;
}

static vtm_status op_samp(vtm_t *m, uint32_t ins, const vtm_instr *in,
                          vtm_emit_fn emit, void *user, int *stop) {
    vtm_status st;
    const vtm_tensor *logits;
    const vtm_tensor *tok = NULL;
    uint32_t n;
    // token инициализируем явно: gcc с -Wmaybe-uninitialized не видит запись
    // через указатель внутри samp_sample и валит сборку. Значение перезапишется
    // на любом успешном пути; ноль — безопасный «первый токен» для страховки.
    int strategy, token = 0, greedy;

    *stop = 0;
    logits = need(m, in->a, ins, "a (logits)", &st);
    if (!logits) return st;
    if ((st = req_f32(m, logits, ins, "a (logits)")) != VTM_OK) return st;
    if (in->imm < VTM_SAMP_ARGMAX || in->imm > VTM_SAMP_TOPP)
        return fail(m, ins, VTM_E_UNSUPPORTED,
                    "SAMP: стратегия imm=%d не реализована (0..3)",
                    (int)in->imm);
    n = logits->n_elements;
    if (n == 0)
        return fail(m, ins, VTM_E_SHAPE, "SAMP: пустые логиты");
    if (in->dst != VTM_NO_TENSOR) {
        tok = need(m, in->dst, ins, "dst (token)", &st);
        if (!tok) return st;
        if ((st = req_f32(m, tok, ins, "dst (token)")) != VTM_OK) return st;
        if (tok->n_elements < 1)
            return fail(m, ins, VTM_E_SHAPE,
                        "SAMP: dst должен содержать хотя бы 1 элемент");
    }

    strategy = (int)in->imm;
    samp_state_begin(m);

    // Переходы в детерминированный argmax: стратегия 0, температура 0,
    // top-k == 1, top-p <= 0. Именно они дают побитовый детерминизм.
    greedy = strategy == VTM_SAMP_ARGMAX
          || (strategy == VTM_SAMP_TEMPERATURE && m->samp_temperature <= 0.0)
          || (strategy == VTM_SAMP_TOPK
              && (m->samp_temperature <= 0.0 || m->samp_top_k == 1))
          || (strategy == VTM_SAMP_TOPP
              && (m->samp_temperature <= 0.0 || m->samp_top_p <= 0.0));
    if (greedy) {
        // Метки окна нужны и argmax'у со штрафом: иначе на каждый токен
        // пробегается вся история. Буфер выделяется только при активном
        // штрафе — чистый argmax остаётся без выделений вовсе.
        int stamped = 0;
        if (samp_penalty_active(m)) {
            vtm_status bst = VTM_OK;
            if (samp_buf_ensure(m, n, ins, &bst)) stamped = samp_stamp_fill(m, n);
        }
        token = samp_greedy(m, logits->f32, n, stamped);
    } else if ((st = samp_sample(m, ins, logits->f32, n, strategy,
                                 &token)) != VTM_OK) {
        return st;
    }

    if (tok) tok->f32[0] = (float)token;
    samp_hist_push(m, token);
    m->generated++;
    m->samp_prev_generated = m->generated;
    if (emit && emit(user, token, m->position) != 0) *stop = 1;
    return VTM_OK;
}

// --- кэш префикса (задача 0081) --------------------------------------------
//
// Зачем не копия K/V. K/V префикса и так лежат в KV-тензорах машины (их пишет
// KVW), поэтому копировать их незачем: достаточно знать, что текущее
// состояние соответствует прогону ids[0..len). Всё остальное, что переживает
// vtm_run(), — позиция, счётчик сгенерированных и состояние сэмплера —
// снимается в поля машины и восстанавливается при попадании. Так запрос с тем
// же префиксом (системный промпт) и другим хвостом не пересчитывает префикс,
// а результат остаётся побитово тем же, что у полного прогона.
//
// Почему хеш, а не копия ids. Копия ids потребовала бы либо фиксированного
// предела длины префикса, либо malloc, который vtm_free() (loader.c, вне зоны
// задачи) не освободит. SHA-256 от ids — 32 байта в машине и полная гарантия:
// совпадение двух разных префиксов по 256-битному хешу это 2^-256, а цена
// (один проход по n·4 байтам) на фоне прогона модели не видна.
//
// Совместимость по состоянию. Попадание восстанавливает снимок, поэтому после
// vtm_run_cached() машина находится ровно в том же состоянии, что и после
// полного прогона тех же ids по одному токену: та же позиция, тот же счётчик,
// та же история и ГПСЧ сэмплера. На это опирается тест (побитовое сравнение).
static void cache_store(vtm_t *m, const int32_t *ids, int n) {
    vtm_sha256_ctx ctx;

    if (n <= 0) {
        m->cache_valid = 0;
        m->cache_len = 0;
        return;
    }
    vtm_sha256_init(&ctx);
    vtm_sha256_update(&ctx, ids, (size_t)n * sizeof *ids);
    vtm_sha256_final(&ctx, m->cache_digest);
    m->cache_len = n;
    m->cache_backend = m->backend;
    m->cache_generated = m->generated;
    m->cache_position = m->position;
    m->cache_input = m->input_token;
    m->cache_samp_ready = m->samp_ready;
    m->cache_samp_rng = m->samp_rng;
    m->cache_samp_seed = m->samp_seed;
    m->cache_samp_temperature = m->samp_temperature;
    m->cache_samp_top_k = m->samp_top_k;
    m->cache_samp_top_p = m->samp_top_p;
    m->cache_samp_repetition_penalty = m->samp_repetition_penalty;
    m->cache_samp_repetition_window = m->samp_repetition_window;
    m->cache_samp_prev_generated = m->samp_prev_generated;
    m->cache_samp_hist_len = m->samp_hist_len;
    m->cache_samp_hist_head = m->samp_hist_head;
    memcpy(m->cache_samp_hist, m->samp_hist, sizeof m->samp_hist);
    m->cache_valid = 1;
}

static void cache_restore(vtm_t *m) {
    m->position = m->cache_position;
    m->input_token = m->cache_input;
    m->generated = m->cache_generated;
    m->samp_ready = m->cache_samp_ready;
    m->samp_rng = m->cache_samp_rng;
    m->samp_seed = m->cache_samp_seed;
    m->samp_temperature = m->cache_samp_temperature;
    m->samp_top_k = m->cache_samp_top_k;
    m->samp_top_p = m->cache_samp_top_p;
    m->samp_repetition_penalty = m->cache_samp_repetition_penalty;
    m->samp_repetition_window = m->cache_samp_repetition_window;
    m->samp_prev_generated = m->cache_samp_prev_generated;
    m->samp_hist_len = m->cache_samp_hist_len;
    m->samp_hist_head = m->cache_samp_hist_head;
    memcpy(m->samp_hist, m->cache_samp_hist, sizeof m->samp_hist);
}

// Годен ли снимок для текущего состояния машины.
static int cache_usable(const vtm_t *m) {
    if (!m->cache_valid || m->cache_len <= 0) return 0;
    // Смена бэкенда (vtm_set_backend) — другой численный путь, K/V префикса
    // посчитаны не им. Модель сменить в одной машине нельзя: загрузка даёт
    // новую vtm_t, а у неё кэш пуст.
    if (m->backend != m->cache_backend) return 0;
    // vtm_reset() ставит input_token = -1; после прогона префикса он всегда
    // равен последнему токену (>= 0), поэтому -1 — надёжный признак сброса
    // даже для префикса из одного токена.
    if (m->input_token < 0 && m->cache_input >= 0) return 0;
    // Состояние ушло назад (vtm_reset, vtm_set_position): снимок относится к
    // более поздней точке, и восстанавливать его значило бы проигнорировать
    // сброс.
    if (m->generated < m->cache_generated) return 0;
    if (m->position < m->cache_position) return 0;
    return 1;
}

// Начинается ли ids[0..n) с кэшированного префикса.
static int cache_hit(vtm_t *m, const int32_t *ids, int n) {
    vtm_sha256_ctx ctx;
    uint8_t digest[32];

    if (!cache_usable(m)) return 0;
    if (m->cache_len > n) return 0;
    vtm_sha256_init(&ctx);
    vtm_sha256_update(&ctx, ids, (size_t)m->cache_len * sizeof *ids);
    vtm_sha256_final(&ctx, digest);
    return memcmp(digest, m->cache_digest, sizeof digest) == 0;
}

int vtm_cache_prefix_len(const vtm_t *m) {
    return (m && cache_usable(m)) ? m->cache_len : 0;
}

void vtm_cache_stats(const vtm_t *m, uint64_t *hits, uint64_t *misses) {
    if (hits) *hits = m ? m->cache_hits : 0;
    if (misses) *misses = m ? m->cache_misses : 0;
}

// Кэш префикса (0081) требует, чтобы снимок состояния совпадал с прогоном
// по токену побитово. Скалярные батчевые ядра этому соответствуют, а
// SIMD-ядра 0122 — нет (у них другой порядок f32-сложения, расхождение
// ≤1e-5), поэтому на SIMD кэш идёт прежним путём по токену. Это и обещает
// контракт vtm_run_batch в vtm.h («на SIMD — прежний путь по токену»).
static vtm_status batch_sequential(vtm_t *m, const int32_t *ids, int n,
                                   int pos0);
static int batch_use_simd(const vtm_t *m);

static vtm_status cache_run_tokens(vtm_t *m, const int32_t *ids, int n,
                                   int pos0) {
    if (batch_use_simd(m)) return batch_sequential(m, ids, n, pos0);
    return vtm_run_batch(m, ids, n);
}

vtm_status vtm_cache_prefix(vtm_t *m, const int32_t *ids, int n) {
    vtm_status st;

    if (!m) return VTM_E_ARG;
    if (n < 0 || (n > 0 && !ids)) return VTM_E_ARG;
    if (!m->backend) m->backend = &vtm_backend_scalar_lut;
    // Префикс считается с чистого состояния: vtm_reset() обнуляет позицию,
    // счётчик и сбрасывает сэмплер (лениво, на первом SAMP), поэтому прогон
    // префикса здесь побитово равен прогону тех же ids у хоста.
    vtm_reset(m);
    m->cache_valid = 0;
    // Прогон префикса пачкой (0112): на скалярном бэкенде это в разы
    // быстрее, на SIMD — прежний путь по токену (см. cache_run_tokens).
    st = cache_run_tokens(m, ids, n, 0);
    if (st != VTM_OK) return st;
    cache_store(m, ids, n);
    return VTM_OK;
}

vtm_status vtm_run_cached(vtm_t *m, const int32_t *ids, int n) {
    vtm_status st = VTM_OK;
    int start, hit;

    if (!m) return VTM_E_ARG;
    if (n < 0 || (n > 0 && !ids)) return VTM_E_ARG;
    if (!m->backend) m->backend = &vtm_backend_scalar_lut;

    hit = cache_hit(m, ids, n);
    if (hit) {
        cache_restore(m);
        start = m->cache_len;
        m->cache_hits++;
    } else {
        // Несовпадающий префикс — честный полный прогон с чистого состояния;
        // кэшем становится вся последовательность, чтобы следующий запрос с
        // тем же промптом (и другим продолжением) уже не считал его заново.
        vtm_reset(m);
        m->cache_valid = 0;
        start = 0;
        m->cache_misses++;
    }
    // Хвост после префикса (или весь прогон при промахе) — одной пачкой
    // (0112); на SIMD — по токену, чтобы снимок кэша остался побитовым.
    if (start < n) {
        vtm_set_position(m, start);
        st = cache_run_tokens(m, ids + start, n - start, start);
        if (st != VTM_OK) return st;
    }
    // При попадании кэш остаётся префиксом: запросы с тем же префиксом и
    // разными хвостами переиспользуют его повторно.
    if (!hit) cache_store(m, ids, n);
    return VTM_OK;
}

// --- батчевый prefill (задача 0112) ----------------------------------------
//
// Граф считает один токен за вызов, поэтому промпт из N токенов — это N
// полных проходов по всем слоям, и каждый проход заново читает и
// расквантовывает одни и те же веса. vtm_run_batch() прогоняет пачку из C
// токенов так: инструкция исполняется один раз для всех токенов пачки.
//
// Раскладка. Активации (тензоры, в которые пишет программа) перестают быть
// одним вектором: на время пачки каждому токену даётся своя копия, а
// инструкции видят её через подмену указателя f32 (batch_bind). Веса и
// KV-кэш общие: веса не пишутся, а KV-кэш адресуется глобальной позицией
// (KVW ниже), поэтому копий не требует.
//
// Порядок исполнения — по инструкциям, внутри инструкции по токенам. Это
// допустимо, потому что единственная зависимость между токенами — через
// KV-кэш: KVW пишет слоты всех токенов пачки до ATTN, а ATTN для строки
// позиции p читает ключи 0..p (causal) или 0..p (без маски, key_count =
// q_start+1), т.е. ровно те, что были бы записаны при прогоне по токену.
// Кольцо KV — исключение: если пачка переходит через границу окна, поздние
// KVW затрут слот, нужный ранней строке ATTN. Поэтому пачка режется так,
// чтобы все её позиции были < win (batch_kv_window), а за границей окна
// токены считаются по одному — результат тот же.
//
// Что ускоряется. Только TMV и MM — они и есть 98% времени на форме 1.5B.
// На скалярном бэкенде их батчевые ядра повторяют арифметику скалярных
// (double, накопитель block_acc на блок масштаба), поэтому равенство с n
// прогонами vtm_run побитовое. На SIMD ту же работу делают батчевые ядра
// 0122 (vtm_simd_tmv_blocks_batch/vtm_simd_mm_batch, backend_neon.c/
// backend_avx2.c): вес читается и распаковывается один раз на пачку,
// расхождение с батчевым скаляром ≤1e-5 (проверяет bench_backend --check).
// Плотные (fp32) матрицы идут туда же — в ядра 0159 vtm_simd_*_f32_batch
// (вектор по токенам пачки), а на скалярном бэкенде — в
// vtm_scalar_dense_batch: тот повторяет double-цикл скалярного пути, поэтому
// побитовость там сохраняется.
// Q4 (kind 5) — в батчевое ядро 0162 vtm_q4_tmv_batch/vtm_q4_mm_batch: оно
// разворачивает строку из нибблов один раз на пачку и умножает на N токенов,
// поэтому поштучный prefill 3B Q4 перестаёт читать и распаковывать одни и те
// же 2,2 ГиБ весов на каждый токен промпта. Скалярный эталон ядра битово
// равен N вызовам vtm_q4_*, NEON — доли 1e-6 (проверяет bench_backend
// --check и --prefill-check). Отказ ядра (нет памяти) — на поштучный путь.
// Остальные операции (EMB/ROPE/KVW/ATTN/RMSN/SAMP/...) дешёвые и считаются
// по токену прежними функциями: это сохраняет их семантику (позиция, окно,
// эмбеддинг) без дублирования кода.
//
// Если батчевых ядер 0122 в сборке нет (или бэкенд кастомный), SIMD идёт
// прежним путём по токену: побитово то же состояние, без выигрыша. Так же
// поступает трассировка: записи vtm_set_trace остаются «инструкция на токен».

#define VTM_BATCH_TOKENS        16
// Минимальная пачка для батчевого плотного (fp32) TMV/MM: ядро 0159
// векторизует по токенам, и меньше 4 токенов четвёрка лент не набирается.
// Такие пачки считаются по токену через vtm_simd_gemv_f32 — это быстрее
// скалярного double-цикла и не требует отдельной ветки в ядре.
#define VTM_DENSE_BATCH_MIN     4
#define VTM_BATCH_SCRATCH_BYTES ((size_t)96u << 20)

// Слова/масштабы в TRT1 лежат со смещением 2 mod 4 (см. backend_scalar.c):
// читаем через memcpy, иначе разыменование float* по невыровненному адресу — UB.
static uint32_t batch_load_u32(const uint32_t *words, size_t i) {
    uint32_t v;
    memcpy(&v, (const uint8_t *)words + i * sizeof v, sizeof v);
    return v;
}

static float batch_load_f32(const float *scales, size_t i) {
    float v;
    memcpy(&v, (const uint8_t *)scales + i * sizeof v, sizeof v);
    return v;
}

typedef struct {
    int       C;          // токенов в текущей пачке
    uint32_t  n_tensors;
    uint32_t  n_local;    // сколько тензоров-активаций
    uint8_t  *local;      // [n_tensors] 1 — у токена своя копия
    uint32_t *idx;        // [n_local] индексы этих тензоров
    size_t   *off;        // [n_tensors] смещение в слоте токена, float
    size_t    slot;       // float на токен
    float    *mem;        // C * slot
    float   **real;       // [n_tensors] настоящие указатели f32
    float    *xt;         // [cols * C] транспонированный вход батчевого ядра
    float    *yt;         // [rows * C] транспонированный выход
    size_t    xcap, ycap; // ёмкости xt/yt, float
} batch_scratch;

static void batch_free(batch_scratch *b) {
    free(b->local);
    free(b->idx);
    free(b->off);
    free(b->mem);
    free(b->real);
    free(b->xt);
    free(b->yt);
    memset(b, 0, sizeof *b);
}

// Разметить тензоры: какие получают копию на токена, и сколько float на токен.
// 1 — размечено (n_local > 0), 0 — батчить нечего/нет памяти.
static int batch_classify(vtm_t *m, batch_scratch *b) {
    uint32_t i, n = m->n_tensors;

    memset(b, 0, sizeof *b);
    b->n_tensors = n;
    b->local = (uint8_t *)calloc(n ? n : 1u, 1u);
    b->off = (size_t *)calloc(n ? n : 1u, sizeof *b->off);
    b->idx = (uint32_t *)calloc(n ? n : 1u, sizeof *b->idx);
    b->real = (float **)calloc(n ? n : 1u, sizeof *b->real);
    if (!b->local || !b->off || !b->idx || !b->real) return 0;

    for (i = 0; i < m->n_instr; i++) {
        const vtm_instr *in = &m->instr[i];
        // KV-кэш — общий: его строки адресуются глобальной позицией, а не
        // токеном пачки. KVW пишет в него напрямую (см. op_kvw).
        if (in->opcode == VTM_OP_KVW) {
            if (in->dst != VTM_NO_TENSOR && in->dst < n) b->local[in->dst] = 0;
            if (in->b != VTM_NO_TENSOR && in->b < n) b->local[in->b] = 0;
            continue;
        }
        // У HALT dst — мусор из байткода (как в trace_out_elems): он ничего
        // не пишет, и «приёмником» его считать нельзя, иначе в активации
        // попадёт случайный тензор (в demo это emb.w в read-only mmap).
        if (in->opcode == VTM_OP_HALT) continue;
        if (in->dst == VTM_NO_TENSOR || in->dst >= n) continue;
        if (m->tensors[in->dst].kind == VTM_TENSOR_TRIT) continue;
        if (m->tensors[in->dst].flags & VTM_TENSOR_FLAG_KV_CACHE) continue;
        b->local[in->dst] = 1;
    }

    b->n_local = 0;
    b->slot = 0;
    for (i = 0; i < n; i++) {
        size_t sz;
        if (!b->local[i]) continue;
        sz = (size_t)m->tensors[i].n_elements;
        sz = (sz + 7u) & ~(size_t)7u;   // выравнивание под 4-lane
        if (b->slot > SIZE_MAX - sz) return 0;
        // Дальше слот умножается на C и sizeof(float) — не даём переполнить.
        if (b->slot + sz > VTM_BATCH_SCRATCH_BYTES / sizeof(float)) return 0;
        b->idx[b->n_local++] = i;
        b->off[i] = b->slot;
        b->slot += sz;
    }
    return b->n_local > 0 && b->slot > 0;
}

// Выделить память пачки и разложить по копиям текущее состояние активаций
// (каждый токен начинает проход с того, чем закончил предыдущий).
static int batch_scratch_alloc(vtm_t *m, batch_scratch *b, int C) {
    size_t total = (size_t)C * b->slot;
    uint32_t k;

    b->C = C;
    b->mem = (float *)calloc(total ? total : 1u, sizeof(float));
    if (!b->mem) return 0;
    for (k = 0; k < b->n_local; k++) {
        uint32_t t = b->idx[k];
        b->real[t] = m->tensors[t].f32;
    }
    return 1;
}

// Каждый токен пачки начинает проход с состояния, которое оставил
// предыдущий прогон (так же, как n вызовов vtm_run: активации в машине
// сохраняются между прогонами). После пачки это состояние — значения
// последнего токена (batch_store_last), поэтому одного и того же копирования
// достаточно и на первый, и на все следующие пачки.
static void batch_reinit(vtm_t *m, const batch_scratch *b) {
    uint32_t k;
    int j;
    for (k = 0; k < b->n_local; k++) {
        uint32_t t = b->idx[k];
        const float *src = b->real[t];
        size_t n_el = (size_t)m->tensors[t].n_elements;
        if (!src || n_el == 0) continue;
        for (j = 0; j < b->C; j++)
            memcpy(b->mem + (size_t)j * b->slot + b->off[t], src,
                   n_el * sizeof(float));
    }
}

// Подменить указатели активаций на копию токена j / вернуть настоящие.
// Веса и KV-кэш не трогаются: они общие.
static void batch_bind(vtm_t *m, const batch_scratch *b, int j) {
    uint32_t k;
    for (k = 0; k < b->n_local; k++) {
        uint32_t t = b->idx[k];
        m->tensors[t].f32 = b->mem + (size_t)j * b->slot + b->off[t];
    }
}

static void batch_unbind(vtm_t *m, const batch_scratch *b) {
    uint32_t k;
    for (k = 0; k < b->n_local; k++) {
        uint32_t t = b->idx[k];
        m->tensors[t].f32 = b->real[t];
    }
}

// После пачки активации — это значения последнего токена: ровно то, что
// оставил бы последний vtm_run (по ним читают логиты тесты и хост).
static void batch_store_last(vtm_t *m, const batch_scratch *b) {
    uint32_t k;
    int last = b->C - 1;
    for (k = 0; k < b->n_local; k++) {
        uint32_t t = b->idx[k];
        float *dst = b->real[t];
        if (!dst || m->tensors[t].n_elements == 0) continue;
        memcpy(dst, b->mem + (size_t)last * b->slot + b->off[t],
               (size_t)m->tensors[t].n_elements * sizeof(float));
    }
    (void)m;
}

// Батчевый путь включается для скалярных бэкендов (батчевые ядра interp.c
// повторяют их арифметику побитово) и для SIMD, если собраны батчевые ядра
// 0122 (vtm_simd_tmv_blocks_batch/vtm_simd_mm_batch): там арифметика та же
// по смыслу, расхождение с батчевым скаляром ≤1e-5. Кастомный бэкенд, для
// которого неизвестно, чему равны его ядра, идёт прежним путём по токену —
// это консервативно и ничего не ломает.
static int batch_backend_enabled(const vtm_t *m) {
    const vtm_backend *be = m->backend;
    if (!be) return 1;
    if (be == &vtm_backend_scalar_lut || be == &vtm_backend_scalar_direct)
        return 1;
    if (be->name && strncmp(be->name, "scalar", 6) == 0) return 1;
    if (be->name && strncmp(be->name, "simd", 4) == 0)
        return vtm_simd_available();
    return 0;
}

// 1 — TMV/MM в пачке считает SIMD-ядро 0122 (триты) или 0159 (плотные), 0 —
// батчевые ядра interp.c (скалярный бэкенд) или путь по токену. Решение
// принимается на каждую инструкцию: если тритовое ядро вернёт -1 (неверные
// аргументы), batch_tmv_kernel/batch_mm_kernel тихо уходят в батчевые
// скалярные строки ниже; у плотных отказ невозможен (аргументы проверены в
// ready), поэтому они уходят в скалярный эталон только при use_simd == 0.
static int batch_use_simd(const vtm_t *m) {
    const vtm_backend *be = m->backend;
    return be && be->name && strncmp(be->name, "simd", 4) == 0
        && vtm_simd_available();
}

// Наименьшее окно KV-кэша (INT_MAX — 2-D кэшей нет). Пачка не должна
// перейти через границу окна: слот позиции p занимает p % win, и KVW
// позднего токена пачки затёр бы ключ, который ещё нужен ранней строке ATTN.
static int batch_kv_window(vtm_t *m) {
    int win = INT_MAX;
    uint32_t i;
    for (i = 0; i < m->n_instr; i++) {
        const vtm_instr *in = &m->instr[i];
        const vtm_tensor *kc;
        int w;
        if (in->opcode != VTM_OP_KVW) continue;
        if (in->dst == VTM_NO_TENSOR || in->dst >= m->n_tensors) continue;
        kc = &m->tensors[in->dst];
        if (kc->ndim != 2) continue;
        w = (int)kv_window(m, kc->dims[0]);
        if (w > 0 && w < win) win = w;
    }
    return win;
}

// Батчевый TMV: y[r][j] = sum_t W[r,t] * x[j][t]. Порядок сложений для
// каждого (r,j) — тот же, что у tmv_lut_rows/tmv_direct_rows, поэтому
// результат побитово равен скалярному ядру, вызванному C раз. Строки делятся
// между потоками пулом бэкенда: каждая строка считается тем же кодом и в том
// же порядке, поэтому от числа потоков результат не зависит (побитово).
typedef struct {
    const vtm_tensor *w;
    const float      *xt;
    int               C;
    int               rows;
    int               cols;
    int               transposed;
    float            *yt;
} batch_mm_job;   // общее для TMV (n == C столбцов) и MM

static void batch_tmv_trit_rows(const vtm_tensor *w, const float *xt, int C,
                                int rows, int cols, float *yt,
                                int r0, int r1) {
    const uint32_t *words = w->words;
    const int sb = (int)w->scale_block;
    const int block20 = (sb % VTM_TRITS_PER_WORD) == 0;
    int r, j, g, k;

    lut243_init();   // op_emb зовёт её только для тритовых таблиц; батч — всегда
    (void)rows;

    for (r = r0; r < r1; r++) {
        double acc[VTM_BATCH_TOKENS], ba[VTM_BATCH_TOKENS];
        int64_t base = (int64_t)r * cols;
        int64_t end = base + cols;
        int64_t wi = base / VTM_TRITS_PER_WORD;
        int64_t wi_end = (end + VTM_TRITS_PER_WORD - 1) / VTM_TRITS_PER_WORD;
        int64_t cur = -1;

        for (j = 0; j < C; j++) { acc[j] = 0.0; ba[j] = 0.0; }

        for (; wi < wi_end; wi++) {
            uint32_t v = batch_load_u32(words, (size_t)wi);
            int64_t wbase = wi * VTM_TRITS_PER_WORD;

            for (g = 0; g < 4; g++) {
                unsigned code = v % 243u;
                const int8_t *tr = g_lut243[code];
                int64_t idx0 = wbase + (int64_t)g * 5;

                v /= 243u;
                for (k = 0; k < 5; k++) {
                    int64_t idx = idx0 + k;
                    double tv;
                    const float *xp;

                    if (idx < base) continue;
                    if (idx >= end) break;
                    if (!block20) {
                        int64_t blk = idx / sb;
                        if (blk != cur) {
                            if (cur >= 0) {
                                double sc = (double)batch_load_f32(
                                    w->scales, (size_t)cur);
                                for (j = 0; j < C; j++) {
                                    acc[j] += ba[j] * sc;
                                    ba[j] = 0.0;
                                }
                            }
                            cur = blk;
                        }
                    }
                    tv = (double)tr[k];
                    xp = xt + (size_t)(idx - base) * (size_t)C;
                    for (j = 0; j < C; j++) ba[j] += tv * (double)xp[j];
                }
            }
            if (block20) {
                double sc = (double)batch_load_f32(
                    w->scales, (size_t)(wbase / sb));
                for (j = 0; j < C; j++) { acc[j] += ba[j] * sc; ba[j] = 0.0; }
            }
        }
        if (!block20 && cur >= 0) {
            double sc = (double)batch_load_f32(w->scales, (size_t)cur);
            for (j = 0; j < C; j++) acc[j] += ba[j] * sc;
        }
        for (j = 0; j < C; j++) yt[(size_t)r * C + j] = (float)acc[j];
    }
}

// Батчевый MM: Y[r][j] = sum_t A[r,t] * B[j][t] (n == 1, как в графе
// tied lm_head). Порядок сложений — как у скалярного op_mm.
static void batch_mm_trit_rows(const vtm_tensor *a, const float *bt, int C,
                               int mm, int kk, float *yt, int r0, int r1) {
    int r, t, j;

    lut243_init();
    (void)mm;
    for (r = r0; r < r1; r++) {
        double acc[VTM_BATCH_TOKENS];
        for (j = 0; j < C; j++) acc[j] = 0.0;
        for (t = 0; t < kk; t++) {
            float wv;
            const float *bp;
            if (trit_weight_at(a, (size_t)r * (size_t)kk + (size_t)t, &wv)
                != 0)
                wv = 0.0f;   // недостижимо: границы проверены в ready
            bp = bt + (size_t)t * (size_t)C;
            for (j = 0; j < C; j++) acc[j] += (double)wv * (double)bp[j];
        }
        for (j = 0; j < C; j++) yt[(size_t)r * C + j] = (float)acc[j];
    }
}

static void batch_tmv_worker(void *p, int tid, int r0, int r1) {
    batch_mm_job *j = (batch_mm_job *)p;
    (void)tid;
    batch_tmv_trit_rows(j->w, j->xt, j->C, j->rows, j->cols, j->yt, r0, r1);
}

static void batch_mm_worker(void *p, int tid, int r0, int r1) {
    batch_mm_job *j = (batch_mm_job *)p;
    (void)tid;
    batch_mm_trit_rows(j->w, j->xt, j->C, j->rows, j->cols, j->yt, r0, r1);
}

// Батчевый TMV. Тритовые веса на SIMD — ядро 0122 (вес читается и
// распаковывается один раз на пачку), иначе — батчевые скалярные строки.
// Q4 (kind 5) — ядро 0162 vtm_q4_tmv_batch (строка разворачивается один раз
// на пачку): 0 — посчитано, -1 — нет памяти/форма не та, и вызывающий
// уходит на поштучный путь (batch_token_op → op_tmv). Плотные (fp32) — ядро
// 0159 vtm_simd_tmv_f32_batch (вектор по токенам пачки), а на скалярном
// бэкенде — vtm_scalar_dense_batch: там побитовость с прогоном по токену —
// контракт, и это прежний double-цикл interp.c, перенесённый в бэкенд.
// Аргументы уже проверены batch_tmv_ready(), поэтому отказ ядер тритов/
// плотных здесь — только NULL/отрицательные размеры, которых нет.
static int batch_tmv_kernel(const vtm_tensor *w, const float *xt, int C,
                            int rows, int cols, int transposed, float *yt,
                            int use_simd) {
    batch_mm_job job;

    if (w->kind == VTM_TENSOR_Q4) {
        if (vtm_q4_tmv_batch(yt, q4_bytes(w), w->scales,
                             w->q4_is_asym ? w->q4_mins : NULL, xt, rows, cols,
                             C, w->q4_is_asym ? 1 : 0) != 0)
            return -1;
        return 0;
    }
    if (w->kind == VTM_TENSOR_TRIT) {
        if (use_simd && vtm_simd_tmv_blocks_batch(yt,
                                                  (const uint8_t *)w->words,
                                                  w->scales, xt, rows, cols,
                                                  C, (int)w->scale_block) == 0)
            return 0;
        job.w = w;
        job.xt = xt;
        job.C = C;
        job.rows = rows;
        job.cols = cols;
        job.transposed = transposed;
        job.yt = yt;
        vtm_parallel_rows(rows, (double)rows * (double)cols * (double)C,
                          batch_tmv_worker, &job);
        return 0;
    }
    if (use_simd
        && vtm_simd_tmv_f32_batch(yt, w->f32, xt, rows, cols, C,
                                  transposed) == 0)
        return 0;
    // Сюда попадаем и при нехватке памяти в SIMD-ядре: скалярный эталон
    // даёт тот же результат по смыслу (double-порядок) и доводит пачку.
    (void)vtm_scalar_dense_batch(yt, w->f32, xt, rows, cols, C, transposed);
    return 0;
}

// Батчевый MM: тритовая A — ядро 0122 или батчевые скалярные строки;
// Q4 A — ядро 0162 (0/-1, отказ — на поштучный путь); плотная A — ядро 0159
// (vtm_simd_mm_f32_batch) или скалярный эталон.
static int batch_mm_kernel(const vtm_tensor *a, const float *bt, int C,
                           int mm, int kk, float *yt, int use_simd) {
    batch_mm_job job;

    if (a->kind == VTM_TENSOR_Q4) {
        if (vtm_q4_mm_batch(yt, q4_bytes(a), a->scales,
                            a->q4_is_asym ? a->q4_mins : NULL, bt, mm, kk, C,
                            a->q4_is_asym ? 1 : 0) != 0)
            return -1;
        return 0;
    }
    if (a->kind == VTM_TENSOR_TRIT) {
        if (use_simd && vtm_simd_mm_batch(yt, (const uint8_t *)a->words,
                                          a->scales, bt, mm, kk, C,
                                          (int)a->scale_block) == 0)
            return 0;
        job.w = a;
        job.xt = bt;
        job.C = C;
        job.rows = mm;
        job.cols = kk;
        job.transposed = 0;
        job.yt = yt;
        vtm_parallel_rows(mm, (double)mm * (double)kk * (double)C,
                          batch_mm_worker, &job);
        return 0;
    }
    if (use_simd
        && vtm_simd_mm_f32_batch(yt, a->f32, bt, mm, kk, C) == 0)
        return 0;
    (void)vtm_scalar_dense_batch(yt, a->f32, bt, mm, kk, C, 0);
    return 0;
}

static int batch_buf(float **p, size_t *cap, size_t need) {
    float *q;
    if (*cap >= need) return 1;
    q = (float *)realloc(*p, (need ? need : 1u) * sizeof(float));
    if (!q) return 0;
    *p = q;
    *cap = need;
    return 1;
}

// Указатель на данные тензора для токена j: у активации — своя копия,
// у общего (вес, KV-кэш) — сам тензор.
static const float *batch_src(const vtm_t *m, const batch_scratch *b,
                              uint32_t idx, int j) {
    if (b->local[idx]) return b->mem + (size_t)j * b->slot + b->off[idx];
    return m->tensors[idx].f32;
}

// Готов ли TMV к батчеву ядру. Консервативно: всё нестандартное уходит в
// путь по токену, где сообщение об ошибке даёт сам op_tmv.
static int batch_tmv_ready(vtm_t *m, const vtm_instr *in,
                           const batch_scratch *b, int *rows, int *cols,
                           int *transposed) {
    const vtm_tensor *w, *x, *y;
    int tr = 0;

    if (!batch_backend_enabled(m)) return 0;
    if (in->a == VTM_NO_TENSOR || in->b == VTM_NO_TENSOR
        || in->dst == VTM_NO_TENSOR)
        return 0;
    if (in->a >= m->n_tensors || in->b >= m->n_tensors
        || in->dst >= m->n_tensors)
        return 0;
    if (in->dst == in->b) return 0;           // op_tmv такое отвергает
    if (b->local[in->a] || !b->local[in->dst]) return 0;
    w = &m->tensors[in->a];
    x = &m->tensors[in->b];
    y = &m->tensors[in->dst];
    if (w->ndim != 2) return 0;
    if (!x->f32 || !y->f32) return 0;
    if (w->kind == VTM_TENSOR_Q4) {
        uint64_t ng;
        // Q4 (kind 5) остаётся упакованным и в пачке: ядро 0162 разворачивает
        // строку один раз на пачку. Условия — те же, что у поштучного op_tmv:
        // транспонирование не поддержано, нужны байты, масштабы и (для Q4_1)
        // минимумы; блок масштаба — 32.
        if ((in->flags & VTM_FLAG_TMV_TRANSPOSED) != 0) return 0;
        if (!w->scales || !q4_bytes(w)) return 0;
        if (w->q4_is_asym && !w->q4_mins) return 0;
        if (w->scale_block != VTM_Q4_GROUP_SIZE) return 0;
        ng = ((uint64_t)w->n_elements + VTM_Q4_GROUP_SIZE - 1u)
           / VTM_Q4_GROUP_SIZE;
        if ((uint64_t)w->n_scales < ng) return 0;
        *rows = (int)w->dims[0];
        *cols = (int)w->dims[1];
    } else if (w->kind == VTM_TENSOR_TRIT) {
        if (!w->words || !w->scales) return 0;
        if (w->scale_block != VTM_SCALE_BLOCK_128
            && w->scale_block != VTM_SCALE_BLOCK_160)
            return 0;
        *rows = (int)w->dims[0];
        *cols = (int)w->dims[1];
    } else {
        if (w->kind < VTM_TENSOR_F32 || w->kind > VTM_TENSOR_BF16
            || !w->f32)
            return 0;
        // Мелкая пачка плотных весов: ядро 0159 векторизует по токенам
        // (минимум 4 ленты). На 2-3 токенах выгоднее GEMV по токену —
        // поэтому не берём инструкцию в батч, и batch_token_op посчитает её
        // через op_tmv/vtm_simd_gemv_f32 (вес читается N раз, но SIMD).
        if (b->C < VTM_DENSE_BATCH_MIN) return 0;
        tr = (in->flags & VTM_FLAG_TMV_TRANSPOSED) != 0;
        *rows = tr ? (int)w->dims[1] : (int)w->dims[0];
        *cols = tr ? (int)w->dims[0] : (int)w->dims[1];
    }
    if (*rows <= 0 || *cols <= 0) return 0;
    if (x->n_elements != (uint32_t)*cols || y->n_elements != (uint32_t)*rows)
        return 0;
    *transposed = tr;
    return 1;
}

static vtm_status batch_tmv_apply(vtm_t *m, uint32_t pc,
                                  const vtm_instr *in,
                                  batch_scratch *b, int *handled) {
    const vtm_tensor *w;
    int rows, cols, tr, j, r, t;
    size_t xn, yn;

    *handled = 0;
    if (!batch_tmv_ready(m, in, b, &rows, &cols, &tr)) return VTM_OK;
    w = &m->tensors[in->a];
    xn = (size_t)cols * (size_t)b->C;
    yn = (size_t)rows * (size_t)b->C;
    if (!batch_buf(&b->xt, &b->xcap, xn) || !batch_buf(&b->yt, &b->ycap, yn))
        return fail(m, pc, VTM_E_NOMEM, "батчевый TMV: нет памяти");

    for (j = 0; j < b->C; j++) {
        const float *xs = batch_src(m, b, in->b, j);
        for (t = 0; t < cols; t++) b->xt[(size_t)t * b->C + j] = xs[t];
    }
    if (batch_tmv_kernel(w, b->xt, b->C, rows, cols, tr, b->yt,
                         batch_use_simd(m)) != 0)
        return VTM_OK;   // handled == 0: batch_token_op пойдёт по токену

    for (j = 0; j < b->C; j++) {
        float *ys = b->mem + (size_t)j * b->slot + b->off[in->dst];
        for (r = 0; r < rows; r++)
            ys[r] = b->yt[(size_t)r * b->C + j];
    }
    *handled = 1;
    return VTM_OK;
}

// MM с n == 1 (граф tied lm_head). A общая, B и Y — на токен.
static int batch_mm_ready(vtm_t *m, const vtm_instr *in,
                          const batch_scratch *b, int *mm, int *kk) {
    const vtm_tensor *a, *bv, *y;

    if (!batch_backend_enabled(m)) return 0;
    if (in->a == VTM_NO_TENSOR || in->b == VTM_NO_TENSOR
        || in->dst == VTM_NO_TENSOR)
        return 0;
    if (in->a >= m->n_tensors || in->b >= m->n_tensors
        || in->dst >= m->n_tensors)
        return 0;
    if (b->local[in->a] || !b->local[in->dst]) return 0;
    if (in->dst == in->a || in->dst == in->b) return 0;
    a = &m->tensors[in->a];
    bv = &m->tensors[in->b];
    y = &m->tensors[in->dst];
    if (a->ndim != 2 || bv->ndim != 2 || y->ndim != 2) return 0;
    if (bv->dims[1] != 1 || y->dims[1] != 1) return 0;
    *mm = (int)a->dims[0];
    *kk = (int)a->dims[1];
    if (*mm <= 0 || *kk <= 0) return 0;
    if (bv->dims[0] != (uint32_t)*kk || y->dims[0] != (uint32_t)*mm)
        return 0;
    if (!bv->f32 || !y->f32) return 0;
    if (a->kind == VTM_TENSOR_Q4) {
        uint64_t n = (uint64_t)*mm * (uint64_t)*kk;
        // Q4 A в пачке — ядро 0162; условия те же, что у op_mm: упакованные
        // байты, масштабы, минимумы для Q4_1, блок 32.
        if (!a->scales || !q4_bytes(a)) return 0;
        if (a->q4_is_asym && !a->q4_mins) return 0;
        if (a->scale_block != VTM_Q4_GROUP_SIZE) return 0;
        if ((uint64_t)a->n_elements < n) return 0;
        if ((uint64_t)a->n_scales
            < (n + VTM_Q4_GROUP_SIZE - 1u) / VTM_Q4_GROUP_SIZE) return 0;
    } else if (a->kind == VTM_TENSOR_TRIT) {
        uint64_t n = (uint64_t)*mm * (uint64_t)*kk;
        if (!a->words || !a->scales) return 0;
        if (a->scale_block != VTM_SCALE_BLOCK_128
            && a->scale_block != VTM_SCALE_BLOCK_160)
            return 0;
        if (a->n_elements < n) return 0;
        if ((uint64_t)a->n_scales < (n + a->scale_block - 1u) / a->scale_block)
            return 0;
    } else if (a->kind < VTM_TENSOR_F32 || a->kind > VTM_TENSOR_BF16
               || !a->f32) {
        return 0;
    } else if (b->C < VTM_DENSE_BATCH_MIN) {
        // Мелкая пачка плотной A — по токену через GEMV: батчевое ядро 0159
        // векторизует по токенам (4 ленты), на 2-3 токенах это дороже.
        return 0;
    }
    return 1;
}

static vtm_status batch_mm_apply(vtm_t *m, uint32_t pc,
                                 const vtm_instr *in,
                                 batch_scratch *b, int *handled) {
    const vtm_tensor *a;
    int mm, kk, j, r, t;
    size_t bn, yn;

    *handled = 0;
    if (!batch_mm_ready(m, in, b, &mm, &kk)) return VTM_OK;
    a = &m->tensors[in->a];
    bn = (size_t)kk * (size_t)b->C;
    yn = (size_t)mm * (size_t)b->C;
    if (!batch_buf(&b->xt, &b->xcap, bn) || !batch_buf(&b->yt, &b->ycap, yn))
        return fail(m, pc, VTM_E_NOMEM, "батчевый MM: нет памяти");

    for (j = 0; j < b->C; j++) {
        const float *bs = batch_src(m, b, in->b, j);
        for (t = 0; t < kk; t++) b->xt[(size_t)t * b->C + j] = bs[t];
    }
    if (batch_mm_kernel(a, b->xt, b->C, mm, kk, b->yt,
                        batch_use_simd(m)) != 0)
        return VTM_OK;   // handled == 0: batch_token_op пойдёт по токену

    for (j = 0; j < b->C; j++) {
        float *ys = b->mem + (size_t)j * b->slot + b->off[in->dst];
        for (r = 0; r < mm; r++)
            ys[r] = b->yt[(size_t)r * b->C + j];
    }
    *handled = 1;
    return VTM_OK;
}

// Одна инструкция прежним путём (кроме TMV/MM — их зовёт batch_token_op).
static vtm_status batch_op_one(vtm_t *m, uint32_t i, const vtm_instr *in) {
    switch (in->opcode) {
    case VTM_OP_RMSN:
        return op_rmsn(m, i, in);
    case VTM_OP_ROPE:
        return op_rope(m, i, in);
    case VTM_OP_SM:
        return op_sm(m, i, in);
    case VTM_OP_SILU:
        return op_silu(m, i, in);
    case VTM_OP_MUL:
        return op_binary(m, i, in, 0);
    case VTM_OP_ADD:
        return op_binary(m, i, in, 1);
    case VTM_OP_EMB:
        return op_emb(m, i, in);
    case VTM_OP_KVW:
        return op_kvw(m, i, in);
    case VTM_OP_ATTN:
        return op_attn(m, i, in);
    case VTM_OP_SAMP: {
        int stop = 0;
        return op_samp(m, i, in, NULL, NULL, &stop);
    }
    case VTM_OP_CPY:
        return op_cpy(m, i, in);
    case VTM_OP_SCAL:
        return op_scal(m, i, in);
    default:
        return fail(m, i, VTM_E_OPCODE, "неизвестный opcode 0x%02x",
                    (unsigned)in->opcode);
    }
}

// Инструкция для всех токенов пачки по одному (позиция и входной токен —
// как у хоста). Для TMV/MM это запасной путь, если батчевое ядро неприменимо.
static vtm_status batch_token_op(vtm_t *m, uint32_t pc, const vtm_instr *in,
                                 batch_scratch *b, const int32_t *ids,
                                 int pos0) {
    int j;
    vtm_status st = VTM_OK;

    for (j = 0; j < b->C; j++) {
        m->position = pos0 + j;
        m->input_token = ids[j];
        batch_bind(m, b, j);
        if (in->opcode == VTM_OP_TMV)
            st = op_tmv(m, pc, in);
        else if (in->opcode == VTM_OP_MM)
            st = op_mm(m, pc, in);
        else
            st = batch_op_one(m, pc, in);
        batch_unbind(m, b);
        if (st != VTM_OK) break;
    }
    return st;
}

// Пачка: по инструкциям, внутри инструкции — по токенам.
static vtm_status batch_chunk(vtm_t *m, batch_scratch *b, const int32_t *ids,
                              int C, int pos0) {
    uint32_t pc;
    vtm_status st = VTM_OK;
    int halted = 0;

    b->C = C;
    m->halted = 0;
    m->error[0] = '\0';
    batch_reinit(m, b);

    for (pc = 0; pc < m->n_instr && !halted; pc++) {
        const vtm_instr *in = &m->instr[pc];
        int handled = 0;

        switch (in->opcode) {
        case VTM_OP_TMV:
            st = batch_tmv_apply(m, pc, in, b, &handled);
            if (st == VTM_OK && !handled)
                st = batch_token_op(m, pc, in, b, ids, pos0);
            break;
        case VTM_OP_MM:
            st = batch_mm_apply(m, pc, in, b, &handled);
            if (st == VTM_OK && !handled)
                st = batch_token_op(m, pc, in, b, ids, pos0);
            break;
        case VTM_OP_HALT:
            m->halted = 1;
            halted = 1;
            break;
        default:
            st = batch_token_op(m, pc, in, b, ids, pos0);
            break;
        }
        if (st != VTM_OK) break;
    }
    batch_unbind(m, b);
    if (st == VTM_OK) batch_store_last(m, b);
    m->position = pos0 + C - 1;
    return st;
}

// Путь по токену: прежняя семантика vtm_run на каждый токен (используется,
// когда батчевых ядер нет или пачка упирается в кольцо KV).
static vtm_status batch_sequential(vtm_t *m, const int32_t *ids, int n,
                                   int pos0) {
    int i;
    for (i = 0; i < n; i++) {
        vtm_status st;
        vtm_set_position(m, pos0 + i);
        vtm_set_token(m, ids[i]);
        st = vtm_run(m, NULL, NULL);
        if (st != VTM_OK) return st;
    }
    return VTM_OK;
}

vtm_status vtm_run_batch(vtm_t *m, const int32_t *ids, int n) {
    int64_t start;
    int off = 0, cmax, win;
    batch_scratch b;

    if (!m) return VTM_E_ARG;
    if (n < 0 || (n > 0 && !ids)) return VTM_E_ARG;
    if (!m->backend) m->backend = &vtm_backend_scalar_lut;
    if (n == 0) return VTM_OK;              // no-op: позиция не двигается
    start = (int64_t)m->position;
    if (start + (int64_t)n - 1 > (int64_t)INT_MAX) return VTM_E_BOUNDS;

    // Трассировка (0082) и бэкенды без батчевых ядер идут прежним путём по
    // токену: записи трассы остаются прежними.
    if (m->trace_cb || !batch_backend_enabled(m))
        return batch_sequential(m, ids, n, (int)start);
    if (!batch_classify(m, &b)) {
        batch_free(&b);
        return batch_sequential(m, ids, n, (int)start);
    }
    // Предел пачки — и по памяти (в пачке C копий активаций), и по длине.
    cmax = VTM_BATCH_TOKENS;
    {
        size_t lim = VTM_BATCH_SCRATCH_BYTES / (b.slot * sizeof(float));
        if (lim < 1u) lim = 1u;
        if ((size_t)cmax > lim) cmax = (int)lim;
    }
    if (cmax > n) cmax = n;
    if (cmax < 2) {                         // пачка не помещается/не нужна
        batch_free(&b);
        return batch_sequential(m, ids, n, (int)start);
    }
    if (!batch_scratch_alloc(m, &b, cmax)) {
        batch_free(&b);
        return batch_sequential(m, ids, n, (int)start);
    }

    win = batch_kv_window(m);
    while (off < n) {
        int64_t pos0 = start + off;
        int C = n - off;
        vtm_status st;

        if (C > cmax) C = cmax;
        if (win != INT_MAX) {
            int64_t room = (int64_t)win - pos0;
            if (room < (int64_t)C) C = (int)room;
        }
        if (C > 1) {
            st = batch_chunk(m, &b, ids + off, C, (int)pos0);
            off += C;
        } else {
            // Позиция у границы окна (или за ней): кольцо переиспользует
            // слоты, поэтому токен считается отдельно, как раньше.
            st = batch_sequential(m, ids + off, 1, (int)pos0);
            off += 1;
        }
        if (st != VTM_OK) {
            batch_free(&b);
            return st;
        }
    }
    batch_free(&b);
    m->position = (int)(start + (int64_t)n - 1);
    m->input_token = ids[n - 1];
    return VTM_OK;
}

// --- главный цикл ----------------------------------------------------------

// --- трассировка по инструкциям (задача 0082) ------------------------------
//
// Стоимость хука — два clock_gettime на инструкцию плюс сам callback. Без
// callback (trace_cb == NULL) цикл ниже не читает часы: tracing снимается
// один раз на входе в vtm_run(), поэтому цена выключенной трассировки — одна
// проверка int на инструкцию.

static uint64_t trace_now_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0u;
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

// Размер выхода инструкции: элементов в dst (все dst в VTM — f32-тензоры).
// HALT ничего не пишет, его dst — мусор из байткода, поэтому 0.
static uint32_t trace_out_elems(const vtm_t *m, const vtm_instr *in) {
    if (in->opcode == VTM_OP_HALT) return 0u;
    if (in->dst == VTM_NO_TENSOR || (uint32_t)in->dst >= m->n_tensors) return 0u;
    return m->tensors[in->dst].n_elements;
}

// Отдать запись в callback. 1 — callback просит остановить прогон.
static int trace_emit(vtm_t *m, uint32_t *seq, uint32_t pc,
                      const vtm_instr *in, uint64_t t0) {
    vtm_trace_rec rec;
    uint64_t t1 = trace_now_ns();

    memset(&rec, 0, sizeof rec);
    rec.index = *seq;
    rec.pc = pc;
    rec.opcode = in->opcode;
    rec.flags = in->flags;
    rec.out_elems = trace_out_elems(m, in);
    rec.out_bytes = rec.out_elems * (uint32_t)sizeof(float);
    rec.t_ns = t0;
    rec.dur_ns = (t1 >= t0) ? (t1 - t0) : 0u;
    (*seq)++;
    return m->trace_cb(m->trace_user, &rec) != 0 ? 1 : 0;
}

void vtm_set_trace(vtm_t *m, vtm_trace_fn cb, void *user) {
    if (!m) return;
    m->trace_cb = cb;
    m->trace_user = cb ? user : NULL;
}

vtm_status vtm_run(vtm_t *m, vtm_emit_fn emit, void *user) {
    uint32_t i;
    uint32_t seq = 0;
    int tracing;
    vtm_status st;

    if (!m) return VTM_E_ARG;
    if (!m->backend) m->backend = &vtm_backend_scalar_lut;
    m->halted = 0;
    m->error[0] = '\0';
    tracing = m->trace_cb != NULL;

    for (i = 0; i < m->n_instr; i++) {
        const vtm_instr *in = &m->instr[i];
        uint64_t trace_t0 = tracing ? trace_now_ns() : 0u;
        switch (in->opcode) {
        case VTM_OP_TMV:
            st = op_tmv(m, i, in);
            break;
        case VTM_OP_MM:
            st = op_mm(m, i, in);
            break;
        case VTM_OP_RMSN:
            st = op_rmsn(m, i, in);
            break;
        case VTM_OP_ROPE:
            st = op_rope(m, i, in);
            break;
        case VTM_OP_SM:
            st = op_sm(m, i, in);
            break;
        case VTM_OP_SILU:
            st = op_silu(m, i, in);
            break;
        case VTM_OP_MUL:
            st = op_binary(m, i, in, 0);
            break;
        case VTM_OP_ADD:
            st = op_binary(m, i, in, 1);
            break;
        case VTM_OP_EMB:
            st = op_emb(m, i, in);
            break;
        case VTM_OP_KVW:
            st = op_kvw(m, i, in);
            break;
        case VTM_OP_ATTN:
            st = op_attn(m, i, in);
            break;
        case VTM_OP_SAMP: {
            int stop = 0;
            st = op_samp(m, i, in, emit, user, &stop);
            if (st == VTM_OK && stop) {
                // Запись нужна и на остановке по emit: инструкция исполнена.
                if (tracing) (void)trace_emit(m, &seq, i, in, trace_t0);
                return VTM_OK;
            }
            break;
        }
        case VTM_OP_CPY:
            st = op_cpy(m, i, in);
            break;
        case VTM_OP_SCAL:
            st = op_scal(m, i, in);
            break;
        case VTM_OP_HALT:
            m->halted = 1;
            if (tracing) (void)trace_emit(m, &seq, i, in, trace_t0);
            return VTM_OK;
        default:
            st = fail(m, i, VTM_E_OPCODE, "неизвестный opcode 0x%02x",
                      (unsigned)in->opcode);
            break;
        }
        // Неуспешная инструкция тоже попадает в трассу — по ней видно, где
        // прогон кончился; callback может попросить вежливую остановку.
        if (tracing && trace_emit(m, &seq, i, in, trace_t0) != 0) return VTM_OK;
        if (st != VTM_OK) return st;
    }
    return VTM_OK;
}
