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

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static vtm_status same_count(vtm_t *m, const vtm_tensor *x, const vtm_tensor *y,
                             uint32_t ins, const char *role) {
    if (x->n_elements != y->n_elements)
        return fail(m, ins, VTM_E_SHAPE,
                    "%s: элементов %u, ожидали %u",
                    role, y->n_elements, x->n_elements);
    return VTM_OK;
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

static vtm_status op_mm(vtm_t *m, uint32_t ins, const vtm_instr *in) {
    vtm_status st;
    const vtm_tensor *a = need(m, in->a, ins, "a (A)", &st);
    const vtm_tensor *b;
    const vtm_tensor *y;
    int mm, kk, nn, r, c, t;

    if (!a) return st;
    b = need(m, in->b, ins, "b (B)", &st);
    if (!b) return st;
    y = need(m, in->dst, ins, "dst (Y)", &st);
    if (!y) return st;
    if ((st = req_f32(m, a, ins, "a (A)")) != VTM_OK) return st;
    if ((st = req_f32(m, b, ins, "b (B)")) != VTM_OK) return st;
    if ((st = req_f32(m, y, ins, "dst (Y)")) != VTM_OK) return st;
    if (a->ndim != 2 || b->ndim != 2 || y->ndim != 2)
        return fail(m, ins, VTM_E_SHAPE, "MM: нужны матрицы 2D");
    mm = (int)a->dims[0];
    kk = (int)a->dims[1];
    nn = (int)b->dims[1];
    if (b->dims[0] != (uint16_t)kk)
        return fail(m, ins, VTM_E_SHAPE,
                    "MM: A=[%d,%d], B=[%u,%u]", mm, kk,
                    (unsigned)b->dims[0], (unsigned)b->dims[1]);
    if (y->dims[0] != (uint16_t)mm || y->dims[1] != (uint16_t)nn)
        return fail(m, ins, VTM_E_SHAPE,
                    "MM: Y=[%u,%u], ожидали [%d,%d]",
                    (unsigned)y->dims[0], (unsigned)y->dims[1], mm, nn);
    if (y == a || y == b)
        return fail(m, ins, VTM_E_SHAPE, "MM: dst совпадает с входом");

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
    eps = (float)((double)in->imm / 1000.0);
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
static vtm_status op_emb(vtm_t *m, uint32_t ins, const vtm_instr *in) {
    vtm_status st;
    const vtm_tensor *e = need(m, in->a, ins, "a (E)", &st);
    const vtm_tensor *y;
    int token, dim;
    int (*fn)(float *, const float *, int, int);

    if (!e) return st;
    y = need(m, in->dst, ins, "dst (y)", &st);
    if (!y) return st;
    if ((st = req_f32(m, e, ins, "a (E)")) != VTM_OK) return st;
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
    fn = m->backend && m->backend->emb ? m->backend->emb
                                       : vtm_backend_scalar_lut.emb;
    if (!fn) return fail(m, ins, VTM_E_UNSUPPORTED, "нет ядра EMB");
    if (fn(y->f32, e->f32, dim, token) != 0)
        return fail(m, ins, VTM_E_TENSOR, "ядро EMB вернуло ошибку");
    return VTM_OK;
}

// --- KVW -------------------------------------------------------------------
//
// dst — K-кэш [ctx, kv_dim], a — K-вектор [kv_dim],
// b   — V-кэш [ctx, kv_dim], c — V-вектор [kv_dim], imm — позиция.
// Слой задаётся самими тензорами (у каждого слоя свой кэш), поэтому
// отдельного поля layer нет.
static vtm_status op_kvw(vtm_t *m, uint32_t ins, const vtm_instr *in) {
    vtm_status st;
    const vtm_tensor *kc = need(m, in->dst, ins, "dst (K-кэш)", &st);
    const vtm_tensor *kv;
    const vtm_tensor *vc;
    const vtm_tensor *vv;
    long long pos;
    uint32_t ctx, kv_dim;

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
    pos = (long long)m->position + (long long)in->imm;
    if (pos < 0 || pos >= (long long)ctx)
        return fail(m, ins, VTM_E_BOUNDS,
                    "KVW: позиция %lld вне ctx=%u", pos, ctx);
    memmove(kc->f32 + (size_t)pos * kv_dim, kv->f32,
            (size_t)kv_dim * sizeof(float));
    memmove(vc->f32 + (size_t)pos * kv_dim, vv->f32,
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
// Масштаб — 1/sqrt(d); softmax считается в double с вычитанием максимума.
static vtm_status op_attn(vtm_t *m, uint32_t ins, const vtm_instr *in) {
    vtm_status st;
    const vtm_tensor *q = need(m, in->a, ins, "a (Q)", &st);
    const vtm_tensor *k;
    const vtm_tensor *v;
    const vtm_tensor *out;
    int plain, n_heads, n_kv_heads, q_len, head_dim, kv_len, ctx;
    int q_start, key_count, h, qi, j, d, rep, causal, kv_dim;
    double scale;
    double *scores = NULL;

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
        q_start = m->position + in->imm;
        if (q_start < 0)
            return fail(m, ins, VTM_E_BOUNDS,
                        "ATTN: q_start=%d отрицателен", q_start);
        key_count = q_start + q_len;
        if (key_count > ctx)
            return fail(m, ins, VTM_E_BOUNDS,
                        "ATTN: q_start+q_len=%d больше ctx=%d",
                        key_count, ctx);
        kv_len = ctx;
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
    scores = (double *)malloc((size_t)key_count * sizeof(double));
    if (!scores) return fail(m, ins, VTM_E_NOMEM, "ATTN: нет памяти под scores");

    for (h = 0; h < n_heads; h++) {
        int kvh = h / rep;
        for (qi = 0; qi < q_len; qi++) {
            const float *qrow = q->f32
                + ((size_t)h * (size_t)q_len + (size_t)qi) * (size_t)head_dim;
            float *orow = out->f32
                + ((size_t)h * (size_t)q_len + (size_t)qi) * (size_t)head_dim;
            double max = -INFINITY;
            double sum = 0.0;

            for (j = 0; j < key_count; j++) {
                double dot = 0.0;
                const float *krow;
                if (causal && (long long)j > (long long)q_start + qi) {
                    scores[j] = -INFINITY;
                    continue;
                }
                if (plain) {
                    krow = k->f32
                        + (((size_t)kvh * (size_t)kv_len) + (size_t)j)
                          * (size_t)head_dim;
                } else {
                    krow = k->f32 + (size_t)j * (size_t)kv_dim
                         + (size_t)kvh * (size_t)head_dim;
                }
                for (d = 0; d < head_dim; d++)
                    dot += (double)qrow[d] * (double)krow[d];
                scores[j] = dot * scale;
                if (scores[j] > max) max = scores[j];
            }
            if (max == -INFINITY) {
                memset(orow, 0, (size_t)head_dim * sizeof(float));
                continue;
            }
            for (j = 0; j < key_count; j++) {
                if (scores[j] == -INFINITY) continue;
                sum += exp(scores[j] - max);
            }
            if (!(sum > 0.0) || !isfinite(sum)) {
                memset(orow, 0, (size_t)head_dim * sizeof(float));
                continue;
            }
            // Переиспользуем scores как вероятности, чтобы не считать exp дважды.
            for (j = 0; j < key_count; j++)
                scores[j] = (scores[j] == -INFINITY)
                          ? 0.0 : exp(scores[j] - max) / sum;
            for (d = 0; d < head_dim; d++) {
                double acc = 0.0;
                for (j = 0; j < key_count; j++) {
                    const float *vrow;
                    if (scores[j] == 0.0) continue;
                    if (plain) {
                        vrow = v->f32
                            + (((size_t)kvh * (size_t)kv_len) + (size_t)j)
                              * (size_t)head_dim;
                    } else {
                        vrow = v->f32 + (size_t)j * (size_t)kv_dim
                             + (size_t)kvh * (size_t)head_dim;
                    }
                    acc += scores[j] * (double)vrow[d];
                }
                orow[d] = (float)acc;
            }
        }
    }
    free(scores);
    return VTM_OK;
}

// --- SAMP ------------------------------------------------------------------
//
// v0.1: imm == 0 — жадный argmax (первый максимум; NaN пропускаются,
// если все NaN — токен 0). imm != 0 — зарезервировано под температуру/
// top-k, пока VTM_E_UNSUPPORTED. dst (если задан) — тензор fp32, куда
// кладётся id как float; callback получает id и позицию.
static vtm_status op_samp(vtm_t *m, uint32_t ins, const vtm_instr *in,
                          vtm_emit_fn emit, void *user, int *stop) {
    vtm_status st;
    const vtm_tensor *logits = need(m, in->a, ins, "a (logits)", &st);
    const vtm_tensor *tok = NULL;
    uint32_t n, j;
    int best_i = 0;
    float best;
    int token;

    *stop = 0;
    if (!logits) return st;
    if ((st = req_f32(m, logits, ins, "a (logits)")) != VTM_OK) return st;
    if (in->imm != 0)
        return fail(m, ins, VTM_E_UNSUPPORTED,
                    "SAMP: режим imm=%d не реализован (v0.1: 0 = argmax)",
                    in->imm);
    n = logits->n_elements;
    if (n == 0)
        return fail(m, ins, VTM_E_SHAPE, "SAMP: пустые логиты");
    best = logits->f32[0];
    for (j = 1; j < n; j++) {
        if (logits->f32[j] > best) {
            best = logits->f32[j];
            best_i = (int)j;
        }
    }
    token = best_i;

    if (in->dst != VTM_NO_TENSOR) {
        tok = need(m, in->dst, ins, "dst (token)", &st);
        if (!tok) return st;
        if ((st = req_f32(m, tok, ins, "dst (token)")) != VTM_OK) return st;
        if (tok->n_elements < 1)
            return fail(m, ins, VTM_E_SHAPE,
                        "SAMP: dst должен содержать хотя бы 1 элемент");
        tok->f32[0] = (float)token;
    }

    m->generated++;
    if (emit && emit(user, token, m->position) != 0) *stop = 1;
    return VTM_OK;
}

// --- главный цикл ----------------------------------------------------------

vtm_status vtm_run(vtm_t *m, vtm_emit_fn emit, void *user) {
    uint32_t i;
    vtm_status st;

    if (!m) return VTM_E_ARG;
    if (!m->backend) m->backend = &vtm_backend_scalar_lut;
    m->halted = 0;
    m->error[0] = '\0';

    for (i = 0; i < m->n_instr; i++) {
        const vtm_instr *in = &m->instr[i];
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
            if (st == VTM_OK && stop) return VTM_OK;
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
            return VTM_OK;
        default:
            st = fail(m, i, VTM_E_OPCODE, "неизвестный opcode 0x%02x",
                      (unsigned)in->opcode);
            break;
        }
        if (st != VTM_OK) return st;
    }
    return VTM_OK;
}
