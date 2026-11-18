// Тесты VTM: формат, загрузчик, кодек тритов, все операции v0.1.
//
// Сборка и запуск: cd src/vtm && make test
#define _POSIX_C_SOURCE 200809L

#include "../../src/vtm/vtm.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int g_fail;

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        g_fail++; \
    } \
} while (0)

#define CHECK_NEAR(got, want, tol) do { \
    double _g = (double)(got), _w = (double)(want), _t = (tol); \
    if (!(fabs(_g - _w) <= _t * (1.0 + fabs(_w)))) { \
        fprintf(stderr, "FAIL %s:%d: %.10g != %.10g (tol %.3g)\n", \
                __FILE__, __LINE__, _g, _w, _t); \
        g_fail++; \
    } \
} while (0)

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) {
        fprintf(stderr, "OOM\n");
        exit(2);
    }
    return p;
}

static void *xrealloc(void *p, size_t n) {
    void *q = realloc(p, n ? n : 1);
    if (!q) {
        fprintf(stderr, "OOM\n");
        exit(2);
    }
    return q;
}

// --- динамический буфер ----------------------------------------------------

typedef struct {
    uint8_t *p;
    size_t len, cap;
} buf_t;

static void buf_append(buf_t *b, const void *data, size_t n) {
    if (b->len + n > b->cap) {
        size_t cap = b->cap ? b->cap : 64;
        while (cap < b->len + n) cap *= 2;
        b->p = (uint8_t *)xrealloc(b->p, cap);
        b->cap = cap;
    }
    if (n) memcpy(b->p + b->len, data, n);
    b->len += n;
}

static void buf_u8(buf_t *b, uint8_t v) { buf_append(b, &v, 1); }

static void buf_u16(buf_t *b, uint16_t v) {
    uint8_t x[2];
    x[0] = (uint8_t)(v & 0xFFu);
    x[1] = (uint8_t)(v >> 8);
    buf_append(b, x, 2);
}

static void buf_u32(buf_t *b, uint32_t v) {
    uint8_t x[4];
    x[0] = (uint8_t)(v & 0xFFu);
    x[1] = (uint8_t)((v >> 8) & 0xFFu);
    x[2] = (uint8_t)((v >> 16) & 0xFFu);
    x[3] = (uint8_t)((v >> 24) & 0xFFu);
    buf_append(b, x, 4);
}

static void buf_u64(buf_t *b, uint64_t v) {
    buf_u32(b, (uint32_t)(v & 0xFFFFFFFFu));
    buf_u32(b, (uint32_t)(v >> 32));
}

static void buf_f32(buf_t *b, float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof u);
    buf_u32(b, u);
}

// --- запись в готовый образ ------------------------------------------------

static void put_u16(uint8_t *p, uint16_t v) {
    p[0] = (uint8_t)(v & 0xFFu);
    p[1] = (uint8_t)(v >> 8);
}

static void put_u32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)(v & 0xFFu);
    p[1] = (uint8_t)((v >> 8) & 0xFFu);
    p[2] = (uint8_t)((v >> 16) & 0xFFu);
    p[3] = (uint8_t)((v >> 24) & 0xFFu);
}

static void put_i32(uint8_t *p, int32_t v) {
    uint32_t u;
    memcpy(&u, &v, sizeof u);
    put_u32(p, u);
}

static uint32_t get_u32(const uint8_t *p) {
    return (uint32_t)p[0]
         | ((uint32_t)p[1] << 8)
         | ((uint32_t)p[2] << 16)
         | ((uint32_t)p[3] << 24);
}

// --- модель для тестов -----------------------------------------------------

typedef struct {
    char name[24];
    uint8_t kind, ndim;
    uint32_t dims[4];
    uint8_t *blob;
    size_t blob_len;
    uint8_t *blob2;
    size_t blob2_len;
    int raw;
    uint32_t offset;
    uint32_t scale_offset;
} tensor_def;

typedef struct {
    tensor_def tensors[32];
    int n_tensors;
    vtm_instr prog[64];
    int n_instr;
    uint16_t header_flags;
    char *meta;        // JSON-мета (копия, как в контейнере); NULL — нет меты
    size_t meta_len;
} model_t;

static void model_init(model_t *m) {
    memset(m, 0, sizeof *m);
}

static void model_set_meta(model_t *m, const char *json) {
    size_t len = json ? strlen(json) : 0;
    free(m->meta);
    m->meta = (char *)xmalloc(len);
    if (len) memcpy(m->meta, json, len);
    m->meta_len = len;
}

static size_t align32(size_t v) {
    return (v + 31u) & ~(size_t)31u;
}

static uint32_t dims_product(int ndim, const uint32_t *dims) {
    uint32_t n = 1;
    int i;
    for (i = 0; i < ndim; i++) n *= dims[i];
    return n;
}

static tensor_def *model_slot(model_t *m, const char *name, uint8_t kind,
                              int ndim, const uint32_t *dims) {
    tensor_def *td;
    int i;
    if (m->n_tensors >= 32) {
        fprintf(stderr, "model: слишком много тензоров\n");
        exit(2);
    }
    td = &m->tensors[m->n_tensors++];
    memset(td, 0, sizeof *td);
    snprintf(td->name, sizeof td->name, "%s", name);
    td->kind = kind;
    td->ndim = (uint8_t)ndim;
    for (i = 0; i < 4; i++) td->dims[i] = i < ndim ? dims[i] : 0;
    return td;
}

static void model_add_f32(model_t *m, const char *name, int ndim,
                          const uint32_t *dims, const float *vals) {
    tensor_def *td = model_slot(m, name, VTM_TENSOR_F32, ndim, dims);
    buf_t b = {0};
    uint32_t i, n = dims_product(ndim, dims);
    for (i = 0; i < n; i++) buf_f32(&b, vals[i]);
    td->blob = b.p;
    td->blob_len = b.len;
}

static void model_add_f16(model_t *m, const char *name, int ndim,
                          const uint32_t *dims, const float *vals) {
    tensor_def *td = model_slot(m, name, VTM_TENSOR_F16, ndim, dims);
    buf_t b = {0};
    uint32_t i, n = dims_product(ndim, dims);
    for (i = 0; i < n; i++) buf_u16(&b, vtm_float_to_half(vals[i]));
    td->blob = b.p;
    td->blob_len = b.len;
}

static void model_add_bf16(model_t *m, const char *name, int ndim,
                           const uint32_t *dims, const float *vals) {
    tensor_def *td = model_slot(m, name, VTM_TENSOR_BF16, ndim, dims);
    buf_t b = {0};
    uint32_t i, n = dims_product(ndim, dims);
    for (i = 0; i < n; i++) buf_u16(&b, vtm_float_to_bf16(vals[i]));
    td->blob = b.p;
    td->blob_len = b.len;
}

// TRT1-сегмент: заголовок 18 байт, масштабы, слова по 8 на блок.
static void make_trt1(buf_t *seg, const int8_t *trits, uint32_t n_trits,
                      const float *scales, int fp32_scale, int scale_block) {
    uint32_t n_scales = (n_trits + (uint32_t)scale_block - 1u)
                      / (uint32_t)scale_block;
    uint32_t storage_blocks = (n_trits + VTM_TRITS_PER_BLOCK - 1u)
                            / VTM_TRITS_PER_BLOCK;
    uint32_t b, w, k;
    buf_append(seg, "TRT1", 4);
    buf_u8(seg, 1);
    buf_u8(seg, (uint8_t)(fp32_scale ? 1 : 0));
    buf_u64(seg, n_trits);
    buf_u32(seg, n_scales);
    for (b = 0; b < n_scales; b++) {
        if (fp32_scale) buf_f32(seg, scales[b]);
        else buf_u16(seg, vtm_float_to_half(scales[b]));
    }
    for (b = 0; b < storage_blocks; b++) {
        for (w = 0; w < VTM_WORDS_PER_BLOCK; w++) {
            int8_t tmp[VTM_TRITS_PER_WORD];
            for (k = 0; k < VTM_TRITS_PER_WORD; k++) {
                uint32_t idx = b * VTM_TRITS_PER_BLOCK
                             + w * VTM_TRITS_PER_WORD + k;
                tmp[k] = idx < n_trits ? trits[idx] : 0;
            }
            buf_u32(seg, vtm_trit_pack20(tmp));
        }
    }
}

static void model_add_trit_scaled(model_t *m, const char *name, int ndim,
                                  const uint32_t *dims, const int8_t *trits,
                                  const float *scales, int fp32_scale,
                                  int scale_block) {
    tensor_def *td = model_slot(m, name, VTM_TENSOR_TRIT, ndim, dims);
    buf_t b = {0};
    uint32_t n = dims_product(ndim, dims);
    make_trt1(&b, trits, n, scales, fp32_scale, scale_block);
    td->blob = b.p;
    td->blob_len = b.len;
}

static void model_add_trit(model_t *m, const char *name, int ndim,
                           const uint32_t *dims, const int8_t *trits,
                           const float *scales, int fp32_scale) {
    // Тестовые 160-файлы помечаем флагом, чтобы неоднозначный n_scale
    // (например, 6 тритов) читался именно как 160.
    m->header_flags |= VTM_HEADER_FLAG_SCALE_BLOCK_160;
    model_add_trit_scaled(m, name, ndim, dims, trits, scales, fp32_scale,
                          (int)VTM_SCALE_BLOCK_160);
}

// Тот же тензор, но в «сырой» раскладке: слова в одном месте, масштабы
// в другом (scale_offset), как допускает поле scale_offset в таблице.
static void model_add_raw_trit(model_t *m, const char *name, int ndim,
                               const uint32_t *dims, const int8_t *trits,
                               const float *scales) {
    tensor_def *td = model_slot(m, name, VTM_TENSOR_TRIT, ndim, dims);
    uint32_t n = dims_product(ndim, dims);
    uint32_t n_blocks = (n + VTM_TRITS_PER_BLOCK - 1u) / VTM_TRITS_PER_BLOCK;
    buf_t words = {0}, sc = {0};
    uint32_t w, k, b;
    for (w = 0; w < n_blocks * VTM_WORDS_PER_BLOCK; w++) {
        int8_t tmp[VTM_TRITS_PER_WORD];
        for (k = 0; k < VTM_TRITS_PER_WORD; k++) {
            uint32_t idx = w * VTM_TRITS_PER_WORD + k;
            tmp[k] = idx < n ? trits[idx] : 0;
        }
        buf_u32(&words, vtm_trit_pack20(tmp));
    }
    for (b = 0; b < n_blocks; b++) buf_f32(&sc, scales[b]);
    td->blob = words.p;
    td->blob_len = words.len;
    td->blob2 = sc.p;
    td->blob2_len = sc.len;
    td->raw = 1;
}

static void model_inst(model_t *m, uint8_t opcode, uint8_t flags,
                       uint16_t dst, uint16_t a, uint16_t b, uint16_t c,
                       int32_t imm) {
    vtm_instr *in;
    if (m->n_instr >= 64) {
        fprintf(stderr, "model: слишком много инструкций\n");
        exit(2);
    }
    in = &m->prog[m->n_instr++];
    in->opcode = opcode;
    in->flags = flags;
    in->dst = dst;
    in->a = a;
    in->b = b;
    in->c = c;
    in->imm = imm;
}

static void model_free(model_t *m) {
    int i;
    for (i = 0; i < m->n_tensors; i++) {
        free(m->tensors[i].blob);
        free(m->tensors[i].blob2);
    }
    free(m->meta);
    memset(m, 0, sizeof *m);
}

static int model_build(model_t *m, uint8_t **out, size_t *out_size) {
    size_t table_size = 32u + (size_t)m->n_tensors * VTM_TENSOR_ENTRY_SIZE;
    size_t data_start = align32(table_size);
    size_t cursor = data_start;
    size_t meta_off = 0, program_off, total;
    uint32_t *offsets = (uint32_t *)xmalloc(sizeof(uint32_t)
                                            * (size_t)m->n_tensors);
    uint32_t *soffsets = (uint32_t *)xmalloc(sizeof(uint32_t)
                                             * (size_t)m->n_tensors);
    uint8_t *img;
    int i;

    for (i = 0; i < m->n_tensors; i++) {
        tensor_def *td = &m->tensors[i];
        cursor = align32(cursor);
        offsets[i] = (uint32_t)(cursor - data_start);
        cursor += td->blob_len;
        if (td->raw) {
            cursor = align32(cursor);
            soffsets[i] = (uint32_t)(cursor - data_start);
            cursor += td->blob2_len;
        } else {
            soffsets[i] = VTM_NO_OFFSET;
        }
    }
    program_off = align32(cursor);
    if (m->meta_len) {
        meta_off = program_off;
        program_off = meta_off + align32(m->meta_len);
    }
    total = program_off + (size_t)m->n_instr * VTM_INSTR_SIZE;
    img = (uint8_t *)calloc(1, total ? total : 1u);
    if (!img) {
        fprintf(stderr, "OOM\n");
        exit(2);
    }

    memcpy(img, "VTM1", 4);
    put_u16(img + 4, VTM_FORMAT_VERSION);
    put_u16(img + 6, m->header_flags);
    put_u32(img + 8, (uint32_t)m->n_instr);
    put_u32(img + 12, (uint32_t)m->n_tensors);
    put_u32(img + 16, (uint32_t)m->n_instr * VTM_INSTR_SIZE);
    // Расширение Python-эталона: смещение программы, меты и её длина.
    put_u32(img + 20, (uint32_t)program_off);
    put_u32(img + 24, m->meta_len ? (uint32_t)meta_off : 0);
    put_u32(img + 28, (uint32_t)m->meta_len);

    for (i = 0; i < m->n_tensors; i++) {
        tensor_def *td = &m->tensors[i];
        uint8_t *e = img + 32u + (size_t)i * VTM_TENSOR_ENTRY_SIZE;
        int d;
        memcpy(e, td->name, 24);
        e[24] = td->kind;
        e[25] = td->ndim;
        for (d = 0; d < 4; d++)
            put_u32(e + 26u + (size_t)d * 4u, td->dims[d]);
        put_u32(e + 42u, td->offset = offsets[i]);
        put_u32(e + 46u, td->scale_offset = soffsets[i]);
        memcpy(img + data_start + offsets[i], td->blob, td->blob_len);
        if (td->raw)
            memcpy(img + data_start + soffsets[i], td->blob2, td->blob2_len);
    }
    if (m->meta_len)
        memcpy(img + meta_off, m->meta, m->meta_len);
    for (i = 0; i < m->n_instr; i++) {
        const vtm_instr *in = &m->prog[i];
        uint8_t *p = img + program_off + (size_t)i * VTM_INSTR_SIZE;
        p[0] = in->opcode;
        p[1] = in->flags;
        put_u16(p + 2, in->dst);
        put_u16(p + 4, in->a);
        put_u16(p + 6, in->b);
        put_u16(p + 8, in->c);
        put_i32(p + 12, in->imm);
    }
    free(offsets);
    free(soffsets);
    *out = img;
    *out_size = total;
    return 0;
}

static vtm_t *load_ok(const uint8_t *img, size_t size) {
    vtm_status st = VTM_OK;
    char err[256];
    vtm_t *m = vtm_load_mem(img, size, &st, err, sizeof err);
    if (!m) fprintf(stderr, "load_ok: %s (status %d)\n", err, (int)st);
    return m;
}

static int run_ok(vtm_t *m) {
    vtm_status st = vtm_run(m, NULL, NULL);
    if (st != VTM_OK)
        fprintf(stderr, "run: %s: %s\n", vtm_strerror(st),
                vtm_error_message(m));
    CHECK(st == VTM_OK);
    return st == VTM_OK;
}

// Мини-контейнер (тензор y + HALT) с заданной метой. Всё, кроме машины,
// освобождается здесь: загрузчик держит собственную копию JSON.
static vtm_t *mini_with_meta(const char *meta) {
    model_t md;
    uint32_t dy[1] = {1};
    float yv[1] = {0.0f};
    uint8_t *img;
    size_t size;
    vtm_t *m;

    model_init(&md);
    model_add_f32(&md, "y", 1, dy, yv);
    model_inst(&md, VTM_OP_HALT, 0, VTM_NO_TENSOR, VTM_NO_TENSOR,
               VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    if (meta) model_set_meta(&md, meta);
    model_build(&md, &img, &size);
    m = load_ok(img, size);
    free(img);
    model_free(&md);
    return m;
}

// --- кодек тритов ----------------------------------------------------------

static void test_trit_codec(void) {
    int8_t a[20], back[20];
    uint8_t bytes[4];
    uint32_t w;
    int i;

    memset(a, -1, sizeof a);
    w = vtm_trit_pack20(a);
    CHECK(w == 0x00000000u);
    put_u32(bytes, w);
    CHECK(bytes[0] == 0x00 && bytes[1] == 0x00 && bytes[2] == 0x00 &&
          bytes[3] == 0x00);
    vtm_trit_unpack20(w, back);
    CHECK(memcmp(a, back, sizeof a) == 0);

    memset(a, 0, sizeof a);
    w = vtm_trit_pack20(a);
    CHECK(w == 0x67EA0DC8u);
    put_u32(bytes, w);
    CHECK(bytes[0] == 0xC8 && bytes[1] == 0x0D && bytes[2] == 0xEA &&
          bytes[3] == 0x67);
    vtm_trit_unpack20(w, back);
    CHECK(memcmp(a, back, sizeof a) == 0);

    memset(a, 1, sizeof a);
    w = vtm_trit_pack20(a);
    CHECK(w == 0xCFD41B90u);
    put_u32(bytes, w);
    CHECK(bytes[0] == 0x90 && bytes[1] == 0x1B && bytes[2] == 0xD4 &&
          bytes[3] == 0xCF);
    vtm_trit_unpack20(w, back);
    CHECK(memcmp(a, back, sizeof a) == 0);

    for (i = 0; i < 20; i++) a[i] = (i & 1) ? -1 : 1;
    w = vtm_trit_pack20(a);
    CHECK(w == 0x33F506E4u);
    put_u32(bytes, w);
    CHECK(bytes[0] == 0xE4 && bytes[1] == 0x06 && bytes[2] == 0xF5 &&
          bytes[3] == 0x33);
    vtm_trit_unpack20(w, back);
    CHECK(memcmp(a, back, sizeof a) == 0);

    // Случайный round-trip на 1000 тритов.
    {
        int8_t src[1000], dst[1000];
        uint32_t words[50];
        uint32_t s = 0x12345678u;
        for (i = 0; i < 1000; i++) {
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            src[i] = (int8_t)((int)(s % 3u) - 1);
        }
        for (i = 0; i < 50; i++) words[i] = vtm_trit_pack20(src + i * 20);
        CHECK(vtm_trit_decode_words(words, 50, dst, 1000) == 0);
        CHECK(memcmp(src, dst, sizeof src) == 0);
        CHECK(vtm_trit_decode_words(words, 50, dst, 1001) != 0);
    }

    // Слово вне диапазона 3^20-1 — ошибка.
    {
        uint32_t bad[1] = {VTM_TRIT_MAX_WORD + 1u};
        int8_t out[20];
        CHECK(vtm_trit_decode_words(bad, 1, out, 20) != 0);
    }

    // half / bfloat16.
    CHECK(vtm_half_to_float(0x0000u) == 0.0f);
    CHECK(vtm_half_to_float(0x3C00u) == 1.0f);
    CHECK(vtm_half_to_float(0xC000u) == -2.0f);
    CHECK(vtm_half_to_float(0x7C00u) == INFINITY);
    CHECK_NEAR(vtm_half_to_float(0x0001u), ldexp(1.0, -24), 1e-7);
    CHECK(vtm_float_to_half(0.0f) == 0x0000u);
    CHECK(vtm_float_to_half(1.0f) == 0x3C00u);
    CHECK(vtm_float_to_half(-2.0f) == 0xC000u);
    CHECK_NEAR(vtm_half_to_float(vtm_float_to_half(0.1f)), 0.1, 1e-3);
    CHECK(vtm_bf16_to_float(0x3F80u) == 1.0f);
    CHECK(vtm_float_to_bf16(1.0f) == 0x3F80u);
    CHECK(vtm_float_to_bf16(-2.0f) == 0xC000u);
}

// --- загрузка и TMV --------------------------------------------------------

static void test_load_and_tmv(void) {
    model_t md;
    uint32_t dw[2] = {2, 3};
    uint32_t dx[1] = {3};
    uint32_t dy[1] = {2};
    int8_t wt[6] = {1, 0, -1, -1, 1, 1};
    float scales[1] = {2.0f};
    float xv[3] = {2.0f, 3.0f, 4.0f};
    float yv[2] = {0.0f, 0.0f};
    uint8_t *img;
    size_t size;
    vtm_t *m;
    const vtm_tensor *t;

    model_init(&md);
    model_add_trit(&md, "W", 2, dw, wt, scales, 1);
    model_add_f32(&md, "x", 1, dx, xv);
    model_add_f32(&md, "y", 1, dy, yv);
    model_inst(&md, VTM_OP_TMV, 0, 2, 0, 1, VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_HALT, 0, VTM_NO_TENSOR, VTM_NO_TENSOR,
               VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    model_build(&md, &img, &size);

    m = load_ok(img, size);
    CHECK(m != NULL);
    if (!m) { free(img); model_free(&md); return; }
    CHECK(vtm_tensor_count(m) == 3);
    CHECK(vtm_instr_count(m) == 2);
    CHECK(strcmp(vtm_backend_name(m), "scalar-lut243") == 0);
    t = vtm_tensor_by_name(m, "W");
    CHECK(t != NULL);
    if (t) {
        CHECK(t->kind == VTM_TENSOR_TRIT);
        CHECK(t->ndim == 2);
        CHECK(t->dims[0] == 2 && t->dims[1] == 3);
        CHECK(t->n_elements == 6);
        CHECK(t->n_blocks == 1);
        CHECK(t->n_scales == 1);
        CHECK(t->scale_block == VTM_SCALE_BLOCK_160);
        CHECK(t->scale_is_f32 == 1);
        CHECK(t->scales[0] == 2.0f);
    }
    t = vtm_tensor_by_index(m, 1);
    CHECK(t != NULL && strcmp(t->name, "x") == 0);
    CHECK(vtm_instruction(m, 0)->opcode == VTM_OP_TMV);

    CHECK(run_ok(m));
    t = vtm_tensor_by_name(m, "y");
    CHECK(t != NULL);
    if (t) {
        CHECK_NEAR(t->f32[0], -4.0, 1e-6);
        CHECK_NEAR(t->f32[1], 10.0, 1e-6);
    }
    CHECK(vtm_generated(m) == 0);
    CHECK(vtm_halted(m));

    // Второй прогон другим бэкендом — те же числа.
    t = vtm_tensor_by_name(m, "y");
    t->f32[0] = t->f32[1] = 0.0f;
    vtm_set_backend(m, &vtm_backend_scalar_direct);
    CHECK(strcmp(vtm_backend_name(m), "scalar-direct") == 0);
    CHECK(run_ok(m));
    CHECK_NEAR(t->f32[0], -4.0, 1e-6);
    CHECK_NEAR(t->f32[1], 10.0, 1e-6);

    // Бэкенд без ядер: интерпретатор обязан взять скалярный LUT.
    {
        vtm_backend empty;
        memset(&empty, 0, sizeof empty);
        empty.name = "empty";
        t->f32[0] = t->f32[1] = 0.0f;
        vtm_set_backend(m, &empty);
        CHECK(run_ok(m));
        CHECK_NEAR(t->f32[0], -4.0, 1e-6);
        CHECK_NEAR(t->f32[1], 10.0, 1e-6);
    }

    // vtm_tensor_unpack_trits.
    {
        int8_t out[6];
        t = vtm_tensor_by_name(m, "W");
        CHECK(vtm_tensor_unpack_trits(t, out, 6) == 0);
        CHECK(memcmp(out, wt, sizeof wt) == 0);
    }

    vtm_free(m);
    free(img);
    model_free(&md);
}

// 320 тритов = два блока по 160. n_scale=2, а для 128 было бы 3,
// поэтому загрузчик однозначно узнаёт 160 и проверяет смену масштаба.
static void test_tmv_blocks(void) {
    model_t md;
    uint32_t dw[2] = {1, 320};
    uint32_t dx[1] = {320};
    uint32_t dy[1] = {1};
    int8_t wt[320];
    float scales[2] = {0.5f, 2.0f};
    float xv[320], yv[1] = {0.0f};
    double want = 0.0;
    uint8_t *img;
    size_t size;
    vtm_t *m;
    int i;

    for (i = 0; i < 320; i++) {
        wt[i] = (int8_t)((i % 3) - 1);
        xv[i] = 1.0f + (float)(i % 7) * 0.125f;
        want += (double)wt[i]
              * (double)scales[i / VTM_TRITS_PER_BLOCK] * (double)xv[i];
    }

    model_init(&md);
    model_add_trit(&md, "W", 2, dw, wt, scales, 1);
    model_add_f32(&md, "x", 1, dx, xv);
    model_add_f32(&md, "y", 1, dy, yv);
    model_inst(&md, VTM_OP_TMV, 0, 2, 0, 1, VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_HALT, 0, VTM_NO_TENSOR, VTM_NO_TENSOR,
               VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    model_build(&md, &img, &size);
    m = load_ok(img, size);
    CHECK(m != NULL);
    if (m) {
        const vtm_tensor *w = vtm_tensor_by_name(m, "W");
        const float *y;
        CHECK(w != NULL && w->n_blocks == 2 && w->n_scales == 2);
        CHECK(w != NULL && w->scale_block == VTM_SCALE_BLOCK_160);
        CHECK(run_ok(m));
        y = vtm_tensor_by_name(m, "y")->f32;
        CHECK_NEAR(y[0], want, 1e-5);
        vtm_set_backend(m, &vtm_backend_scalar_direct);
        CHECK(run_ok(m));
        CHECK_NEAR(y[0], want, 1e-5);
        vtm_free(m);
    }
    free(img);
    model_free(&md);
}

// 300 тритов = 2×150: n_scale=3 (128), а 160 дал бы 2 — загрузчик обязан
// выбрать 128. Строка короче блока, поэтому граница 128 пересекает и
// строки, и слова: проверяем глобальный (row-major) индекс блока.
static void test_tmv_scale128(void) {
    model_t md;
    uint32_t dw[2] = {2, 150};
    uint32_t dx[1] = {150};
    uint32_t dy[1] = {2};
    int8_t wt[300];
    float scales[3] = {0.5f, 1.5f, 2.5f};
    float xv[150], yv[2] = {0.0f, 0.0f};
    double want[2] = {0.0, 0.0};
    uint8_t *img;
    size_t size;
    vtm_t *m;
    int i, r, c;

    for (i = 0; i < 300; i++) wt[i] = (int8_t)((i % 3) - 1);
    for (i = 0; i < 150; i++) xv[i] = 1.0f + (float)(i % 5) * 0.25f;
    for (r = 0; r < 2; r++) {
        for (c = 0; c < 150; c++) {
            int64_t idx = (int64_t)r * 150 + c;
            want[r] += (double)wt[idx]
                     * (double)scales[idx / VTM_SCALE_BLOCK_128]
                     * (double)xv[c];
        }
    }

    model_init(&md);
    model_add_trit_scaled(&md, "W", 2, dw, wt, scales, 1,
                          (int)VTM_SCALE_BLOCK_128);
    model_add_f32(&md, "x", 1, dx, xv);
    model_add_f32(&md, "y", 1, dy, yv);
    model_inst(&md, VTM_OP_TMV, 0, 2, 0, 1, VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_HALT, 0, VTM_NO_TENSOR, VTM_NO_TENSOR,
               VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    model_build(&md, &img, &size);
    m = load_ok(img, size);
    CHECK(m != NULL);
    if (m) {
        const vtm_tensor *w = vtm_tensor_by_name(m, "W");
        const float *y;
        CHECK(w != NULL && w->n_blocks == 2 && w->n_scales == 3);
        CHECK(w != NULL && w->scale_block == VTM_SCALE_BLOCK_128);
        CHECK(run_ok(m));
        y = vtm_tensor_by_name(m, "y")->f32;
        CHECK_NEAR(y[0], want[0], 1e-5);
        CHECK_NEAR(y[1], want[1], 1e-5);
        vtm_set_backend(m, &vtm_backend_scalar_direct);
        CHECK(run_ok(m));
        CHECK_NEAR(y[0], want[0], 1e-5);
        CHECK_NEAR(y[1], want[1], 1e-5);
        vtm_free(m);
    }
    free(img);
    model_free(&md);
}

static void test_raw_trit_layout(void) {
    model_t md;
    uint32_t dw[2] = {1, 3};
    uint32_t dx[1] = {3};
    uint32_t dy[1] = {1};
    int8_t wt[3] = {1, -1, 0};
    float scales[1] = {3.0f};
    float xv[3] = {2.0f, 5.0f, 7.0f};
    float yv[1] = {0.0f};
    uint8_t *img;
    size_t size;
    vtm_t *m;

    model_init(&md);
    md.header_flags = VTM_HEADER_FLAG_F32_SCALES;
    model_add_raw_trit(&md, "W", 2, dw, wt, scales);
    model_add_f32(&md, "x", 1, dx, xv);
    model_add_f32(&md, "y", 1, dy, yv);
    model_inst(&md, VTM_OP_TMV, 0, 2, 0, 1, VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_HALT, 0, VTM_NO_TENSOR, VTM_NO_TENSOR,
               VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    model_build(&md, &img, &size);

    m = load_ok(img, size);
    CHECK(m != NULL);
    if (m) {
        const vtm_tensor *w = vtm_tensor_by_name(m, "W");
        CHECK(w != NULL && w->layout == VTM_LAYOUT_RAW);
        CHECK(run_ok(m));
        CHECK_NEAR(vtm_tensor_by_name(m, "y")->f32[0],
                   (1.0 * 2.0 - 1.0 * 5.0) * 3.0, 1e-6);
        vtm_free(m);
    }
    free(img);
    model_free(&md);
}

// --- порча файла -----------------------------------------------------------

static void test_bad_files(void) {
    model_t md;
    uint32_t dw[2] = {1, 1};
    uint32_t dy[1] = {1};
    int8_t wt[1] = {1};
    float scales[1] = {1.0f};
    float yv[1] = {0.0f};
    uint8_t *img;
    size_t size, data_start;
    vtm_status st;
    char err[256];
    vtm_t *m;

    model_init(&md);
    model_add_trit(&md, "W", 2, dw, wt, scales, 1);
    model_add_f32(&md, "y", 1, dy, yv);
    model_inst(&md, VTM_OP_HALT, 0, VTM_NO_TENSOR, VTM_NO_TENSOR,
               VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    model_build(&md, &img, &size);
    data_start = align32(32u + (size_t)md.n_tensors * VTM_TENSOR_ENTRY_SIZE);

    // Обрезка: не должно быть падения, только NULL.
    m = vtm_load_mem(img, size - 1, &st, err, sizeof err);
    CHECK(m == NULL);
    CHECK(st != VTM_OK);
    m = vtm_load_mem(img, 10, &st, err, sizeof err);
    CHECK(m == NULL);
    CHECK(st == VTM_E_TRUNCATED);
    m = vtm_load_mem(img, 0, &st, err, sizeof err);
    CHECK(m == NULL);
    CHECK(st == VTM_E_TRUNCATED);
    CHECK(err[0] != '\0');

    // Чужая магия.
    {
        uint8_t *c = (uint8_t *)xmalloc(size);
        memcpy(c, img, size);
        c[0] = 'X';
        m = vtm_load_mem(c, size, &st, err, sizeof err);
        CHECK(m == NULL && st == VTM_E_MAGIC);
        free(c);
    }
    // Чужая версия.
    {
        uint8_t *c = (uint8_t *)xmalloc(size);
        memcpy(c, img, size);
        put_u16(c + 4, 99);
        m = vtm_load_mem(c, size, &st, err, sizeof err);
        CHECK(m == NULL && st == VTM_E_VERSION);
        free(c);
    }
    // v0.1 отвергается с внятным сообщением (ADR 0011).
    {
        uint8_t *c = (uint8_t *)xmalloc(size);
        memcpy(c, img, size);
        put_u16(c + 4, VTM_FORMAT_VERSION_V01);
        m = vtm_load_mem(c, size, &st, err, sizeof err);
        CHECK(m == NULL && st == VTM_E_VERSION);
        CHECK(strstr(err, "формат v0.1 не поддерживается") != NULL);
        free(c);
    }
    // Абсурдные n_tensors / n_instr.
    {
        uint8_t *c = (uint8_t *)xmalloc(size);
        memcpy(c, img, size);
        put_u32(c + 12, 0xFFFFFFFFu);
        m = vtm_load_mem(c, size, &st, err, sizeof err);
        CHECK(m == NULL);
        memcpy(c, img, size);
        put_u32(c + 8, 0xFFFFFFFFu);
        m = vtm_load_mem(c, size, &st, err, sizeof err);
        CHECK(m == NULL);
        free(c);
    }
    // offset за границей.
    {
        uint8_t *c = (uint8_t *)xmalloc(size);
        memcpy(c, img, size);
        put_u32(c + 32u + 42u, 0xFFFFFFE0u);
        m = vtm_load_mem(c, size, &st, err, sizeof err);
        CHECK(m == NULL);
        CHECK(st == VTM_E_BOUNDS || st == VTM_E_FORMAT);
        free(c);
    }
    // Битое слово тритов (больше 3^20-1).
    {
        uint8_t *c = (uint8_t *)xmalloc(size);
        memcpy(c, img, size);
        // W — первый тензор, fp32-масштаб: слово начинается через 18+4.
        put_u32(c + data_start + 22u, 0xFFFFFFFFu);
        m = vtm_load_mem(c, size, &st, err, sizeof err);
        CHECK(m == NULL);
        CHECK(st == VTM_E_SCALE);
        free(c);
    }
    // n_program не равен ни размеру, ни смещению.
    {
        uint8_t *c = (uint8_t *)xmalloc(size);
        memcpy(c, img, size);
        put_u32(c + 16, 12345u);
        m = vtm_load_mem(c, size, &st, err, sizeof err);
        CHECK(m == NULL);
        CHECK(st == VTM_E_FORMAT);
        free(c);
    }
    // n_program — только размер программы (как в Python-эталоне).
    {
        uint8_t *c = (uint8_t *)xmalloc(size);
        size_t prog_off = size - (size_t)md.n_instr * VTM_INSTR_SIZE;
        memcpy(c, img, size);
        put_u32(c + 16, (uint32_t)prog_off);
        m = vtm_load_mem(c, size, &st, err, sizeof err);
        CHECK(m == NULL);
        CHECK(st == VTM_E_FORMAT);
        free(c);
    }

    free(img);
    model_free(&md);
}

static void test_file_load(void) {
    model_t md;
    uint32_t dx[1] = {1};
    uint32_t dy[1] = {1};
    float xv[1] = {7.0f};
    float yv[1] = {0.0f};
    uint8_t *img;
    size_t size;
    char path1[64], path2[64], path3[64];
    long stamp = (long)time(NULL);
    FILE *f;
    vtm_t *m;

    model_init(&md);
    model_add_f32(&md, "x", 1, dx, xv);
    model_add_f32(&md, "y", 1, dy, yv);
    model_inst(&md, VTM_OP_CPY, 0, 1, 0, VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_HALT, 0, VTM_NO_TENSOR, VTM_NO_TENSOR,
               VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    model_build(&md, &img, &size);

    snprintf(path1, sizeof path1, "/tmp/vtm_test_%ld_1.vtm", stamp);
    snprintf(path2, sizeof path2, "/tmp/vtm_test_%ld_2.vtm", stamp);
    snprintf(path3, sizeof path3, "/tmp/vtm_test_%ld_3.vtm", stamp);
    f = fopen(path1, "wb");
    CHECK(f != NULL);
    if (f) {
        CHECK(fwrite(img, 1, size, f) == size);
        fclose(f);
    }
    f = fopen(path2, "wb");
    CHECK(f != NULL);
    if (f) {
        CHECK(fwrite(img, 1, size - 1, f) == size - 1);
        fclose(f);
    }
    f = fopen(path3, "wb");
    CHECK(f != NULL);
    if (f) fclose(f); // пустой файл
    m = vtm_load(path1);
    CHECK(m != NULL);
    if (m) {
        CHECK(run_ok(m));
        CHECK_NEAR(vtm_tensor_by_name(m, "y")->f32[0], 7.0, 1e-6);
        vtm_free(m);
    }
    m = vtm_load(path2);
    CHECK(m == NULL);
    CHECK(strlen(vtm_last_error()) > 0);
    m = vtm_load(path3);
    CHECK(m == NULL);
    CHECK(strlen(vtm_last_error()) > 0);
    remove(path1);
    remove(path2);
    remove(path3);
    free(img);
    model_free(&md);
}

// --- RMSN, SM --------------------------------------------------------------

static void test_rmsnorm(void) {
    model_t md;
    uint32_t d[2] = {2};
    float xv[2] = {3.0f, 4.0f};
    float wv[2] = {1.0f, 2.0f};
    float yv[2] = {0.0f, 0.0f};
    uint8_t *img;
    size_t size;
    vtm_t *m;

    model_init(&md);
    model_add_f32(&md, "x", 1, d, xv);
    model_add_f32(&md, "w", 1, d, wv);
    model_add_f32(&md, "y", 1, d, yv);
    model_inst(&md, VTM_OP_RMSN, 0, 2, 0, 1, VTM_NO_TENSOR, 1000);
    model_inst(&md, VTM_OP_HALT, 0, VTM_NO_TENSOR, VTM_NO_TENSOR,
               VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    model_build(&md, &img, &size);
    m = load_ok(img, size);
    CHECK(m != NULL);
    if (m) {
        CHECK(run_ok(m));
        CHECK_NEAR(vtm_tensor_by_name(m, "y")->f32[0], 0.816496580927726, 1e-6);
        CHECK_NEAR(vtm_tensor_by_name(m, "y")->f32[1], 2.1773242158072694, 1e-6);
        vtm_free(m);
    }
    free(img);
    model_free(&md);
}

static void test_softmax(void) {
    model_t md;
    uint32_t dx1[1] = {3};
    uint32_t dy1[1] = {3};
    float x1[3] = {1.0f, 2.0f, 3.0f};
    float y1[3] = {0.0f, 0.0f, 0.0f};
    uint32_t dx2[2] = {2, 3};
    uint32_t dy2[2] = {2, 3};
    float x2[6] = {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f};
    float y2[6] = {0.0f};
    uint8_t *img;
    size_t size;
    vtm_t *m;

    model_init(&md);
    model_add_f32(&md, "x1", 1, dx1, x1);
    model_add_f32(&md, "y1", 1, dy1, y1);
    model_add_f32(&md, "x2", 2, dx2, x2);
    model_add_f32(&md, "y2", 2, dy2, y2);
    model_inst(&md, VTM_OP_SM, 0, 1, 0, VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_SM, VTM_FLAG_CAUSAL, 3, 2, VTM_NO_TENSOR,
               VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_HALT, 0, VTM_NO_TENSOR, VTM_NO_TENSOR,
               VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    model_build(&md, &img, &size);
    m = load_ok(img, size);
    CHECK(m != NULL);
    if (m) {
        float *y;
        CHECK(run_ok(m));
        y = vtm_tensor_by_name(m, "y1")->f32;
        CHECK_NEAR(y[0], 0.09003057317038045, 1e-6);
        CHECK_NEAR(y[1], 0.24472847105479764, 1e-6);
        CHECK_NEAR(y[2], 0.6652409557748219, 1e-6);
        y = vtm_tensor_by_name(m, "y2")->f32;
        CHECK_NEAR(y[0], 1.0, 1e-6);
        CHECK_NEAR(y[1], 0.0, 1e-6);
        CHECK_NEAR(y[2], 0.0, 1e-6);
        CHECK_NEAR(y[3], 0.2689414213699951, 1e-6);
        CHECK_NEAR(y[4], 0.7310585786300049, 1e-6);
        CHECK_NEAR(y[5], 0.0, 1e-6);

        // Сдвиг позиции: строка 0 видит j <= 1, строка 1 — все три.
        vtm_set_position(m, 1);
        CHECK(run_ok(m));
        y = vtm_tensor_by_name(m, "y2")->f32;
        CHECK_NEAR(y[0], 0.2689414213699951, 1e-6);
        CHECK_NEAR(y[1], 0.7310585786300049, 1e-6);
        CHECK_NEAR(y[2], 0.0, 1e-6);
        CHECK_NEAR(y[3], 0.09003057317038045, 1e-6);
        CHECK_NEAR(y[4], 0.24472847105479764, 1e-6);
        CHECK_NEAR(y[5], 0.6652409557748219, 1e-6);
        vtm_free(m);
    }
    free(img);
    model_free(&md);
}

// --- ROPE ------------------------------------------------------------------

static void test_rope(void) {
    model_t md;
    uint32_t dx[1] = {4};
    uint32_t dy[1] = {4};
    float xv[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float y0[4] = {0.0f};
    float y1[4] = {0.0f};
    uint8_t *img;
    size_t size;
    vtm_t *m;

    model_init(&md);
    model_add_f32(&md, "x", 1, dx, xv);
    model_add_f32(&md, "y0", 1, dy, y0);
    model_add_f32(&md, "y1", 1, dy, y1);
    model_inst(&md, VTM_OP_ROPE, 0, 1, 0, 1, VTM_NO_TENSOR, 1);
    model_inst(&md, VTM_OP_ROPE, VTM_FLAG_ROPE_INTERLEAVED, 2, 0, 1,
               VTM_NO_TENSOR, 1);
    model_inst(&md, VTM_OP_HALT, 0, VTM_NO_TENSOR, VTM_NO_TENSOR,
               VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    model_build(&md, &img, &size);
    m = load_ok(img, size);
    CHECK(m != NULL);
    if (m) {
        const float *y;
        CHECK(run_ok(m));
        y = vtm_tensor_by_name(m, "y0")->f32;
        CHECK_NEAR(y[0], -1.9841106485555495, 1e-6);
        CHECK_NEAR(y[1], 1.959900667496664, 1e-6);
        CHECK_NEAR(y[2], 2.4623779024123156, 1e-6);
        CHECK_NEAR(y[3], 4.019799668334994, 1e-6);
        y = vtm_tensor_by_name(m, "y1")->f32;
        CHECK_NEAR(y[0], -1.1426396637476532, 1e-6);
        CHECK_NEAR(y[1], 1.922075596544176, 1e-6);
        CHECK_NEAR(y[2], 2.9598506679133294, 1e-6);
        CHECK_NEAR(y[3], 4.029799501669161, 1e-6);
        vtm_free(m);
    }
    free(img);
    model_free(&md);
}

// --- метаданные ------------------------------------------------------------
//
// Мета — JSON между данными и программой (docs/vtm.md). Проверяем: theta и
// раскладка ROPE берутся из меты; fp16-эмбеддинги и tied lm_head (плотный
// TMV, в т.ч. транспонированный); битая мета не роняет загрузку.

// Сборка: x [8] -> ROPE -> y [8] -> TMV(identity) -> logits [8]. Последний
// шаг нужен лишь затем, чтобы «логиты» были настоящими логитами (y·I).
static int run_rope_meta(const char *meta, int pos, float *logits, int n) {
    model_t md;
    uint32_t dx[1] = {8};
    uint32_t di[2] = {8, 8};
    float xv[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float zero[8] = {0.0f};
    float ident[64];
    uint8_t *img;
    size_t size;
    vtm_t *m;
    int i, k, ok = 0;

    for (i = 0; i < 8; i++)
        for (k = 0; k < 8; k++)
            ident[i * 8 + k] = (i == k) ? 1.0f : 0.0f;

    model_init(&md);
    model_add_f32(&md, "x", 1, dx, xv);
    model_add_f32(&md, "y", 1, dx, zero);
    model_add_f32(&md, "I", 2, di, ident);
    model_add_f32(&md, "logits", 1, dx, zero);
    model_inst(&md, VTM_OP_ROPE, 0, 1, 0, 1, VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_TMV, 0, 3, 2, 1, VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_HALT, 0, VTM_NO_TENSOR, VTM_NO_TENSOR,
               VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    if (meta) model_set_meta(&md, meta);
    model_build(&md, &img, &size);
    m = load_ok(img, size);
    if (m) {
        vtm_set_position(m, pos);
        if (run_ok(m)) {
            memcpy(logits, vtm_tensor_by_name(m, "logits")->f32,
                   (size_t)n * sizeof(float));
            ok = 1;
        }
        vtm_free(m);
    }
    free(img);
    model_free(&md);
    return ok;
}

// Эталон RoPE в double — независимо от кода интерпретатора.
static void rope_ref(double *y, const double *x, int heads, int head_dim,
                     long long pos, double theta, int interleaved) {
    int half = head_dim / 2, h, p;
    for (h = 0; h < heads; h++) {
        int base = h * head_dim;
        for (p = 0; p < half; p++) {
            double inv = pow(theta, -2.0 * (double)p / (double)head_dim);
            double ang = (double)pos * inv;
            double co = cos(ang), si = sin(ang);
            int i0 = base + (interleaved ? 2 * p : p);
            int i1 = base + (interleaved ? 2 * p + 1 : p + half);
            double x0 = x[i0], x1 = x[i1];
            y[i0] = x0 * co - x1 * si;
            y[i1] = x1 * co + x0 * si;
        }
    }
}

static double max_abs_diff(const float *a, const float *b, int n) {
    int i;
    double d = 0.0;
    for (i = 0; i < n; i++) {
        double v = fabs((double)a[i] - (double)b[i]);
        if (v > d) d = v;
    }
    return d;
}

static void test_meta_rope(void) {
    float y_none[8], y4[8], y6[8], y6i[8];
    double ref[8], x[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    double d;
    int k;

    CHECK(run_rope_meta(NULL, 13, y_none, 8));
    CHECK(run_rope_meta("{\"rope_theta\": 10000.0}", 13, y4, 8));
    CHECK(run_rope_meta("{\"rope_theta\": 1000000.0}", 13, y6, 8));
    CHECK(run_rope_meta("{\"rope_theta\": 1000000.0, "
                        "\"rope_layout\": \"interleaved\"}", 13, y6i, 8));

    // Без меты — прежние 10000 побитово.
    CHECK(memcmp(y_none, y4, sizeof y4) == 0);

    rope_ref(ref, x, 1, 8, 13, 10000.0, 0);
    for (k = 0; k < 8; k++) CHECK_NEAR(y4[k], ref[k], 1e-6);

    rope_ref(ref, x, 1, 8, 13, 1e6, 0);
    for (k = 0; k < 8; k++) CHECK_NEAR(y6[k], ref[k], 1e-6);

    rope_ref(ref, x, 1, 8, 13, 1e6, 1);
    for (k = 0; k < 8; k++) CHECK_NEAR(y6i[k], ref[k], 1e-6);

    // Главный критерий: theta из меты реально меняет логиты.
    d = max_abs_diff(y4, y6, 8);
    CHECK(d > 1e-3);
    CHECK(max_abs_diff(y6, y6i, 8) > 1e-3);

    printf("мета rope_theta (pos=13, head_dim=8):\n");
    printf("  theta=1e4        logits = [%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f]\n",
           y4[0], y4[1], y4[2], y4[3], y4[4], y4[5], y4[6], y4[7]);
    printf("  theta=1e6        logits = [%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f]\n",
           y6[0], y6[1], y6[2], y6[3], y6[4], y6[5], y6[6], y6[7]);
    printf("  theta=1e6, interleaved = [%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f]\n",
           y6i[0], y6i[1], y6i[2], y6i[3], y6i[4], y6i[5], y6i[6], y6i[7]);
    printf("  max|1e4 - 1e6| = %.6f; max|half - interleaved| = %.6f\n",
           d, max_abs_diff(y6, y6i, 8));

    // Вложенные ключи (meta["model"]) читаются так же, как верхний уровень,
    // а синонимы "vocab"/"tie_word_embeddings" — как "vocab_size"/"tied_embeddings".
    {
        float ynest[8];
        vtm_t *mn;
        CHECK(run_rope_meta("{\"model\": {\"rope_theta\": 1000000.0}}",
                            13, ynest, 8));
        CHECK(memcmp(ynest, y6, sizeof y6) == 0);
        mn = mini_with_meta("{\"model\": {\"rope_theta\": 1000000, "
                            "\"vocab\": 42, \"tie_word_embeddings\": true}}");
        CHECK(mn != NULL);
        if (mn) {
            CHECK(vtm_meta_rope_theta(mn) == 1000000.0);
            CHECK(vtm_meta_vocab_size(mn) == 42u);
            CHECK(vtm_meta_tied_embeddings(mn) == 1);
            vtm_free(mn);
        }
        // Верхний уровень приоритетнее вложенного, даже если вложенный идёт
        // в JSON позже; ключ только во вложенном — тоже читается.
        mn = mini_with_meta("{\"rope_theta\": 1000000, \"vocab_size\": 42, "
                            "\"tied_embeddings\": true, "
                            "\"model\": {\"rope_theta\": 123, "
                            "\"vocab\": 9, \"tied_embeddings\": false, "
                            "\"rope_layout\": \"interleaved\"}}");
        CHECK(mn != NULL);
        if (mn) {
            CHECK(vtm_meta_rope_theta(mn) == 1000000.0);
            CHECK(vtm_meta_vocab_size(mn) == 42u);
            CHECK(vtm_meta_tied_embeddings(mn) == 1);
            CHECK(vtm_meta_rope_layout(mn) == VTM_ROPE_LAYOUT_INTERLEAVED);
            vtm_free(mn);
        }
    }

    // Доступ к мете через API.
    {
        model_t md;
        uint32_t dy[1] = {1};
        float yv[1] = {0.0f};
        uint8_t *img;
        size_t size;
        vtm_t *m;
        model_init(&md);
        model_add_f32(&md, "y", 1, dy, yv);
        model_inst(&md, VTM_OP_HALT, 0, VTM_NO_TENSOR, VTM_NO_TENSOR,
                   VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
        model_set_meta(&md, "{\"format\":\"vtm\",\"rope_theta\":1000000,"
                            "\"rope_layout\":\"interleaved\","
                            "\"vocab_size\":151936,"
                            "\"tied_embeddings\":true}");
        model_build(&md, &img, &size);
        m = load_ok(img, size);
        CHECK(m != NULL);
        if (m) {
            CHECK(vtm_meta_present(m) == 1);
            CHECK(vtm_meta_rope_theta(m) == 1000000.0);
            CHECK(vtm_meta_rope_layout(m) == VTM_ROPE_LAYOUT_INTERLEAVED);
            CHECK(vtm_meta_vocab_size(m) == 151936u);
            CHECK(vtm_meta_tied_embeddings(m) == 1);
            CHECK(strstr(vtm_meta_json(m), "\"tied_embeddings\":true") != NULL);
            vtm_free(m);
        }
        free(img);
        model_free(&md);
    }
}

// fp16-эмбеддинги: EMB читает fp16-таблицу, а tied lm_head — это та же
// таблица как плотная матрица (TMV), в том числе в транспонированном виде.
static void test_meta_fp16_emb_tied(void) {
    model_t md;
    uint32_t de[2] = {3, 4};     // E: [vocab, hidden]
    uint32_t det[2] = {4, 3};    // Et = Eᵀ: [hidden, vocab]
    uint32_t dy[1] = {4};
    uint32_t dl[1] = {3};
    float ev[12] = {0.5f, -1.0f, 2.0f, 3.5f,
                    1.5f, 0.1f, -2.0f, 0.75f,
                    -0.5f, 0.5f, 1.0f, -1.5f};
    float et[12];
    float yh[4] = {0.0f};
    float lg[3] = {0.0f};
    float lgt[3] = {0.0f};
    double want_y[4], want_l[3];
    uint8_t *img;
    size_t size;
    vtm_t *m;
    int i, k;

    for (i = 0; i < 3; i++)
        for (k = 0; k < 4; k++)
            et[k * 3 + i] = ev[i * 4 + k];

    model_init(&md);
    model_add_f16(&md, "E", 2, de, ev);
    model_add_f32(&md, "yh", 1, dy, yh);
    model_add_f32(&md, "logits", 1, dl, lg);
    model_add_f32(&md, "logits_t", 1, dl, lgt);
    model_add_f16(&md, "Et", 2, det, et);
    // yh = E[1]; logits = E·yh; logits_t = Eᵀᵀ·yh (флаг транспонирования).
    model_inst(&md, VTM_OP_EMB, 0, 1, 0, VTM_NO_TENSOR, VTM_NO_TENSOR, 1);
    model_inst(&md, VTM_OP_TMV, 0, 2, 0, 1, VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_TMV, VTM_FLAG_TMV_TRANSPOSED, 3, 4, 1,
               VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_HALT, 0, VTM_NO_TENSOR, VTM_NO_TENSOR,
               VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    model_set_meta(&md, "{\"format\":\"vtm\",\"vocab_size\":3,"
                        "\"tied_embeddings\":true}");
    model_build(&md, &img, &size);
    m = load_ok(img, size);
    CHECK(m != NULL);
    if (m) {
        const vtm_tensor *E;
        CHECK(vtm_meta_present(m) == 1);
        CHECK(vtm_meta_vocab_size(m) == 3u);
        CHECK(vtm_meta_tied_embeddings(m) == 1);
        CHECK(vtm_meta_rope_theta(m) == 10000.0); // нет ключа — дефолт
        E = vtm_tensor_by_name(m, "E");
        CHECK(E != NULL && E->kind == VTM_TENSOR_F16 && E->f32 != NULL);
        // Сырое чтение fp16: значения арены против fp16-эталона, 1e-3.
        for (i = 0; i < 12; i++)
            CHECK_NEAR(E->f32[i],
                       vtm_half_to_float(vtm_float_to_half(ev[i])), 1e-3);
        CHECK(run_ok(m));

        // Эталон — значения после округления до fp16, в double.
        for (k = 0; k < 4; k++)
            want_y[k] = (double)vtm_half_to_float(
                vtm_float_to_half(ev[4 + k]));
        for (i = 0; i < 3; i++) {
            want_l[i] = 0.0;
            for (k = 0; k < 4; k++)
                want_l[i] += (double)vtm_half_to_float(
                    vtm_float_to_half(ev[i * 4 + k])) * want_y[k];
        }
        for (k = 0; k < 4; k++)
            CHECK_NEAR(vtm_tensor_by_name(m, "yh")->f32[k], want_y[k], 1e-3);
        for (i = 0; i < 3; i++) {
            CHECK_NEAR(vtm_tensor_by_name(m, "logits")->f32[i], want_l[i], 1e-3);
            CHECK_NEAR(vtm_tensor_by_name(m, "logits_t")->f32[i], want_l[i],
                       1e-3);
        }
        {
            double max_dev = 0.0;
            for (i = 0; i < 12; i++) {
                double d = fabs((double)E->f32[i] - (double)ev[i]);
                if (d > max_dev) max_dev = d;
            }
            printf("мета fp16+tied: fp16 E: max|E - исходные float| = %.3e; ",
                   max_dev);
        }
        printf("EMB(yh)=[%.6f %.6f %.6f %.6f] "
               "logits=[%.6f %.6f %.6f] (эталон %.6f %.6f %.6f), "
               "транспонированный lm_head совпал\n",
               (double)vtm_tensor_by_name(m, "yh")->f32[0],
               (double)vtm_tensor_by_name(m, "yh")->f32[1],
               (double)vtm_tensor_by_name(m, "yh")->f32[2],
               (double)vtm_tensor_by_name(m, "yh")->f32[3],
               (double)vtm_tensor_by_name(m, "logits")->f32[0],
               (double)vtm_tensor_by_name(m, "logits")->f32[1],
               (double)vtm_tensor_by_name(m, "logits")->f32[2],
               want_l[0], want_l[1], want_l[2]);
        vtm_free(m);
    }
    free(img);
    model_free(&md);
}

// Битая мета: любой дефект — не ошибка загрузки, а откат на дефолты.
static void test_meta_broken(void) {
    static const struct {
        const char *meta;
        int present;
        double theta;
    } cases[] = {
        { "{\"rope_theta\":", 0, VTM_DEFAULT_ROPE_THETA }, // обрезанный JSON
        { "not json", 0, VTM_DEFAULT_ROPE_THETA },          // вообще не JSON
        { "[1, 2, 3]", 0, VTM_DEFAULT_ROPE_THETA },         // не объект
        { "{\"rope_theta\": 1000000} мусор", 0,
          VTM_DEFAULT_ROPE_THETA },                         // мусор после объекта
        { "{\"rope_theta\": 0}", 1, VTM_DEFAULT_ROPE_THETA }, // theta <= 0
        { "{\"rope_theta\": \"abc\", \"rope_layout\": 7, "
          "\"vocab_size\": -5, \"tied_embeddings\": \"yes\"}", 1,
          VTM_DEFAULT_ROPE_THETA },                         // не те типы
        { "{\"rope_theta\": 1000000}", 1, 1000000.0 },      // валидная
    };
    const size_t n_cases = sizeof cases / sizeof cases[0];
    size_t c;
    int ok = 0;

    for (c = 0; c < n_cases; c++) {
        model_t md;
        uint32_t dy[1] = {1};
        float yv[1] = {0.0f};
        uint8_t *img;
        size_t size;
        vtm_t *m;

        model_init(&md);
        model_add_f32(&md, "y", 1, dy, yv);
        model_inst(&md, VTM_OP_HALT, 0, VTM_NO_TENSOR, VTM_NO_TENSOR,
                   VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
        model_set_meta(&md, cases[c].meta);
        model_build(&md, &img, &size);
        m = load_ok(img, size);
        CHECK(m != NULL);
        if (m) {
            memset(vtm_tensor_by_name(m, "y")->f32, 0, sizeof yv);
            CHECK(vtm_meta_present(m) == cases[c].present);
            CHECK(vtm_meta_rope_theta(m) == cases[c].theta);
            CHECK(vtm_meta_rope_layout(m) == VTM_ROPE_LAYOUT_HALF);
            CHECK(vtm_meta_vocab_size(m) == 0u);
            CHECK(vtm_meta_tied_embeddings(m) == 0);
            CHECK(run_ok(m));
            ok++;
            vtm_free(m);
        }
        free(img);
        model_free(&md);
    }
    CHECK(ok == (int)n_cases);

    // Валидная мета читается как надо, но её содержимое испорчено/смещения
    // врут — контейнер всё равно грузится.
    {
        model_t md;
        uint32_t dy[1] = {1};
        float yv[1] = {0.0f};
        uint8_t *img;
        uint8_t *copy;
        size_t size;
        uint32_t meta_off, meta_len;
        vtm_t *m;

        model_init(&md);
        model_add_f32(&md, "y", 1, dy, yv);
        model_inst(&md, VTM_OP_HALT, 0, VTM_NO_TENSOR, VTM_NO_TENSOR,
                   VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
        model_set_meta(&md, "{\"rope_theta\": 1000000, \"vocab_size\": 7}");
        model_build(&md, &img, &size);
        meta_off = get_u32(img + 24);
        meta_len = get_u32(img + 28);
        CHECK(meta_off != 0 && meta_len != 0);

        // 1) мусор вместо JSON по правильному смещению.
        copy = (uint8_t *)xmalloc(size);
        memcpy(copy, img, size);
        memcpy(copy + meta_off, "\x00\xFF\x01", 3);
        m = load_ok(copy, size);
        CHECK(m != NULL);
        if (m) {
            CHECK(vtm_meta_present(m) == 0);
            CHECK(vtm_meta_rope_theta(m) == VTM_DEFAULT_ROPE_THETA);
            CHECK(run_ok(m));
            vtm_free(m);
        }
        free(copy);

        // 2) meta_len больше, чем есть до программы.
        copy = (uint8_t *)xmalloc(size);
        memcpy(copy, img, size);
        put_u32(copy + 28, 0x7FFFFFFFu);
        m = load_ok(copy, size);
        CHECK(m != NULL);
        if (m) {
            CHECK(vtm_meta_present(m) == 0);
            CHECK(run_ok(m));
            vtm_free(m);
        }
        free(copy);

        // 3) meta_offset за концом файла.
        copy = (uint8_t *)xmalloc(size);
        memcpy(copy, img, size);
        put_u32(copy + 24, 0xFFFFFFF0u);
        m = load_ok(copy, size);
        CHECK(m != NULL);
        if (m) {
            CHECK(vtm_meta_present(m) == 0);
            CHECK(run_ok(m));
            vtm_free(m);
        }
        free(copy);

        // 4) meta_offset указывает в таблицу тензоров (перекрытие).
        copy = (uint8_t *)xmalloc(size);
        memcpy(copy, img, size);
        put_u32(copy + 24, 32u);
        m = load_ok(copy, size);
        CHECK(m != NULL);
        if (m) {
            CHECK(vtm_meta_present(m) == 0);
            CHECK(run_ok(m));
            vtm_free(m);
        }
        free(copy);

        free(img);
        model_free(&md);
    }
    printf("битая мета: %zu вариантов JSON + 4 порчи заголовка — "
           "все загрузились на дефолтах\n", n_cases);
}

// --- EMB, MM ---------------------------------------------------------------

static void test_emb(void) {
    model_t md;
    uint32_t de[2] = {2, 2};
    uint32_t dy[1] = {2};
    float ev[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float yv[2] = {0.0f, 0.0f};
    uint8_t *img;
    size_t size;
    vtm_t *m;

    model_init(&md);
    model_add_f32(&md, "E", 2, de, ev);
    model_add_f32(&md, "y", 1, dy, yv);
    model_inst(&md, VTM_OP_EMB, 0, 1, 0, VTM_NO_TENSOR, VTM_NO_TENSOR, 1);
    model_inst(&md, VTM_OP_HALT, 0, VTM_NO_TENSOR, VTM_NO_TENSOR,
               VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    model_build(&md, &img, &size);
    m = load_ok(img, size);
    CHECK(m != NULL);
    if (m) {
        CHECK(run_ok(m));
        CHECK_NEAR(vtm_tensor_by_name(m, "y")->f32[0], 3.0, 1e-6);
        CHECK_NEAR(vtm_tensor_by_name(m, "y")->f32[1], 4.0, 1e-6);

        vtm_free(m);
    }
    free(img);
    model_free(&md);

    // EMB с VTM_FLAG_EMB_INPUT.
    model_init(&md);
    model_add_f32(&md, "E", 2, de, ev);
    model_add_f32(&md, "y", 1, dy, yv);
    model_inst(&md, VTM_OP_EMB, VTM_FLAG_EMB_INPUT, 1, 0, VTM_NO_TENSOR,
               VTM_NO_TENSOR, -1);
    model_inst(&md, VTM_OP_HALT, 0, VTM_NO_TENSOR, VTM_NO_TENSOR,
               VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    model_build(&md, &img, &size);
    m = load_ok(img, size);
    CHECK(m != NULL);
    if (m) {
        vtm_status st;
        CHECK(vtm_run(m, NULL, NULL) == VTM_E_ARG); // токен не задан
        vtm_set_token(m, 1);
        st = vtm_run(m, NULL, NULL);
        CHECK(st == VTM_OK);
        CHECK_NEAR(vtm_tensor_by_name(m, "y")->f32[0], 3.0, 1e-6);
        CHECK_NEAR(vtm_tensor_by_name(m, "y")->f32[1], 4.0, 1e-6);
        vtm_free(m);
    }
    free(img);
    model_free(&md);
}

static void test_mm(void) {
    model_t md;
    uint32_t da[2] = {2, 3};
    uint32_t db[2] = {3, 2};
    uint32_t dy[2] = {2, 2};
    float av[6] = {1, 2, 3, 4, 5, 6};
    float bv[6] = {1, 2, 3, 4, 5, 6};
    float yv[4] = {0};
    uint8_t *img;
    size_t size;
    vtm_t *m;

    model_init(&md);
    model_add_f32(&md, "A", 2, da, av);
    model_add_f32(&md, "B", 2, db, bv);
    model_add_f32(&md, "Y", 2, dy, yv);
    model_inst(&md, VTM_OP_MM, 0, 2, 0, 1, VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_HALT, 0, VTM_NO_TENSOR, VTM_NO_TENSOR,
               VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    model_build(&md, &img, &size);
    m = load_ok(img, size);
    CHECK(m != NULL);
    if (m) {
        const float *y;
        CHECK(run_ok(m));
        y = vtm_tensor_by_name(m, "Y")->f32;
        CHECK_NEAR(y[0], 22.0, 1e-6);
        CHECK_NEAR(y[1], 28.0, 1e-6);
        CHECK_NEAR(y[2], 49.0, 1e-6);
        CHECK_NEAR(y[3], 64.0, 1e-6);
        vtm_free(m);
    }
    free(img);
    model_free(&md);
}

// --- поэлементные операции -------------------------------------------------

static void test_elementwise(void) {
    model_t md;
    uint32_t d2[1] = {2};
    uint32_t d3[1] = {3};
    float av[2] = {1.0f, 2.0f};
    float bv[2] = {3.0f, 4.0f};
    float addv[2] = {0.0f, 0.0f};
    float mulv[2] = {0.0f, 0.0f};
    float xv[2] = {2.0f, -4.0f};
    float scalv[2] = {0.0f, 0.0f};
    float cv[3] = {1.0f, 2.0f, 3.0f};
    float cpyv[3] = {0.0f, 0.0f, 0.0f};
    float silu_x[2] = {0.0f, 1.0f};
    float silu_y[2] = {0.0f, 0.0f};
    uint8_t *img;
    size_t size;
    vtm_t *m;

    model_init(&md);
    model_add_f32(&md, "a", 1, d2, av);
    model_add_f32(&md, "b", 1, d2, bv);
    model_add_f32(&md, "add", 1, d2, addv);
    model_add_f32(&md, "mul", 1, d2, mulv);
    model_add_f32(&md, "x", 1, d2, xv);
    model_add_f32(&md, "scal", 1, d2, scalv);
    model_add_f32(&md, "c", 1, d3, cv);
    model_add_f32(&md, "cpy", 1, d3, cpyv);
    model_add_f32(&md, "silu_x", 1, d2, silu_x);
    model_add_f32(&md, "silu_y", 1, d2, silu_y);
    model_inst(&md, VTM_OP_ADD, 0, 2, 0, 1, VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_MUL, 0, 3, 0, 1, VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_SCAL, 0, 5, 4, VTM_NO_TENSOR, VTM_NO_TENSOR, 500);
    model_inst(&md, VTM_OP_CPY, 0, 7, 6, VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_SILU, 0, 9, 8, VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_HALT, 0, VTM_NO_TENSOR, VTM_NO_TENSOR,
               VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    model_build(&md, &img, &size);
    m = load_ok(img, size);
    CHECK(m != NULL);
    if (m) {
        CHECK(run_ok(m));
        CHECK_NEAR(vtm_tensor_by_name(m, "add")->f32[0], 4.0, 1e-6);
        CHECK_NEAR(vtm_tensor_by_name(m, "add")->f32[1], 6.0, 1e-6);
        CHECK_NEAR(vtm_tensor_by_name(m, "mul")->f32[0], 3.0, 1e-6);
        CHECK_NEAR(vtm_tensor_by_name(m, "mul")->f32[1], 8.0, 1e-6);
        CHECK_NEAR(vtm_tensor_by_name(m, "scal")->f32[0], 1.0, 1e-6);
        CHECK_NEAR(vtm_tensor_by_name(m, "scal")->f32[1], -2.0, 1e-6);
        CHECK_NEAR(vtm_tensor_by_name(m, "cpy")->f32[0], 1.0, 1e-6);
        CHECK_NEAR(vtm_tensor_by_name(m, "cpy")->f32[1], 2.0, 1e-6);
        CHECK_NEAR(vtm_tensor_by_name(m, "cpy")->f32[2], 3.0, 1e-6);
        CHECK_NEAR(vtm_tensor_by_name(m, "silu_y")->f32[0], 0.0, 1e-6);
        CHECK_NEAR(vtm_tensor_by_name(m, "silu_y")->f32[1],
                   0.7310585786300049, 1e-6);
        vtm_free(m);
    }
    free(img);
    model_free(&md);
}

// --- ATTN ------------------------------------------------------------------

static void test_attn_plain(void) {
    model_t md;
    uint32_t dq[3] = {1, 1, 2};
    uint32_t dk[3] = {1, 2, 2};
    uint32_t dv[3] = {1, 2, 2};
    uint32_t dout[3] = {1, 1, 2};
    float qv[2] = {1.0f, 0.0f};
    float kv[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    float vv[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float ov[2] = {0.0f, 0.0f};
    uint8_t *img;
    size_t size;
    vtm_t *m;

    model_init(&md);
    model_add_f32(&md, "Q", 3, dq, qv);
    model_add_f32(&md, "K", 3, dk, kv);
    model_add_f32(&md, "V", 3, dv, vv);
    model_add_f32(&md, "O", 3, dout, ov);
    model_inst(&md, VTM_OP_ATTN, 0, 3, 0, 1, 2, 0);
    model_inst(&md, VTM_OP_HALT, 0, VTM_NO_TENSOR, VTM_NO_TENSOR,
               VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    model_build(&md, &img, &size);
    m = load_ok(img, size);
    CHECK(m != NULL);
    if (m) {
        const float *o;
        CHECK(run_ok(m));
        o = vtm_tensor_by_name(m, "O")->f32;
        CHECK_NEAR(o[0], 1.6604769013466862, 1e-6);
        CHECK_NEAR(o[1], 2.6604769013466862, 1e-6);
        vtm_free(m);
    }
    free(img);
    model_free(&md);
}

static void test_attn_causal(void) {
    model_t md;
    uint32_t dq[3] = {1, 2, 2};
    uint32_t dk[3] = {1, 3, 2};
    uint32_t dv[3] = {1, 3, 2};
    uint32_t dout[3] = {1, 2, 2};
    float qv[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    float kv[6] = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
    float vv[6] = {1.0f, 0.0f, 0.0f, 1.0f, 10.0f, 10.0f};
    float ov[4] = {0.0f};
    uint8_t *img;
    size_t size;
    vtm_t *m;

    model_init(&md);
    model_add_f32(&md, "Q", 3, dq, qv);
    model_add_f32(&md, "K", 3, dk, kv);
    model_add_f32(&md, "V", 3, dv, vv);
    model_add_f32(&md, "O", 3, dout, ov);
    model_inst(&md, VTM_OP_ATTN, VTM_FLAG_CAUSAL, 3, 0, 1, 2, 0);
    model_inst(&md, VTM_OP_HALT, 0, VTM_NO_TENSOR, VTM_NO_TENSOR,
               VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    model_build(&md, &img, &size);
    m = load_ok(img, size);
    CHECK(m != NULL);
    if (m) {
        const float *o;
        CHECK(run_ok(m));
        o = vtm_tensor_by_name(m, "O")->f32;
        CHECK_NEAR(o[0], 0.6697615493266569, 1e-6);
        CHECK_NEAR(o[1], 0.3302384506733431, 1e-6);
        CHECK_NEAR(o[2], 4.208896741438287, 1e-6);
        CHECK_NEAR(o[3], 4.412233019477645, 1e-6);
        vtm_free(m);
    }
    free(img);
    model_free(&md);
}

static void test_kvw_attn_cache(void) {
    model_t md;
    uint32_t dkc[2] = {2, 2};
    uint32_t dvc[2] = {2, 2};
    uint32_t dq[3] = {1, 1, 2};
    uint32_t dout[3] = {1, 1, 2};
    uint32_t dvec[1] = {2};
    float k0[2] = {1.0f, 0.0f};
    float v0[2] = {1.0f, 2.0f};
    float k1[2] = {0.0f, 1.0f};
    float v1[2] = {3.0f, 4.0f};
    float qv[2] = {1.0f, 0.0f};
    float kc[4] = {0.0f};
    float vc[4] = {0.0f};
    float ov[2] = {0.0f, 0.0f};
    uint8_t *img;
    size_t size;
    vtm_t *m;

    model_init(&md);
    model_add_f32(&md, "Kc", 2, dkc, kc);
    model_add_f32(&md, "Vc", 2, dvc, vc);
    model_add_f32(&md, "K0", 1, dvec, k0);
    model_add_f32(&md, "V0", 1, dvec, v0);
    model_add_f32(&md, "K1", 1, dvec, k1);
    model_add_f32(&md, "V1", 1, dvec, v1);
    model_add_f32(&md, "Q", 3, dq, qv);
    model_add_f32(&md, "O", 3, dout, ov);
    model_inst(&md, VTM_OP_KVW, 0, 0, 2, 1, 3, 0);
    model_inst(&md, VTM_OP_KVW, 0, 0, 4, 1, 5, 1);
    // q_start = position (0) + imm (1) = 1; видит ключи 0 и 1.
    model_inst(&md, VTM_OP_ATTN, 0, 7, 6, 0, 1, 1);
    model_inst(&md, VTM_OP_HALT, 0, VTM_NO_TENSOR, VTM_NO_TENSOR,
               VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    model_build(&md, &img, &size);
    m = load_ok(img, size);
    CHECK(m != NULL);
    if (m) {
        const float *o;
        CHECK(run_ok(m));
        o = vtm_tensor_by_name(m, "O")->f32;
        CHECK_NEAR(o[0], 1.6604769013466862, 1e-6);
        CHECK_NEAR(o[1], 2.6604769013466862, 1e-6);
        // Кэш действительно заполнен.
        CHECK_NEAR(vtm_tensor_by_name(m, "Kc")->f32[0], 1.0, 1e-6);
        CHECK_NEAR(vtm_tensor_by_name(m, "Kc")->f32[3], 1.0, 1e-6);
        CHECK_NEAR(vtm_tensor_by_name(m, "Vc")->f32[2], 3.0, 1e-6);
        vtm_free(m);
    }
    free(img);
    model_free(&md);
}

// --- SAMP и f16/bf16 -------------------------------------------------------

typedef struct {
    int token;
    int calls;
    int position;
} cb_state;

static int cb_capture(void *user, int token, int position) {
    cb_state *s = (cb_state *)user;
    s->token = token;
    s->calls++;
    s->position = position;
    return 0;
}

static void test_samp(void) {
    model_t md;
    uint32_t dlog[1] = {3};
    uint32_t dto[1] = {1};
    float logits[3] = {0.5f, 2.0f, 1.0f};
    float tok[1] = {-1.0f};
    uint8_t *img;
    size_t size;
    vtm_t *m;
    cb_state cb;

    model_init(&md);
    model_add_f32(&md, "logits", 1, dlog, logits);
    model_add_f32(&md, "token", 1, dto, tok);
    model_inst(&md, VTM_OP_SAMP, 0, 1, 0, VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_HALT, 0, VTM_NO_TENSOR, VTM_NO_TENSOR,
               VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    model_build(&md, &img, &size);
    m = load_ok(img, size);
    CHECK(m != NULL);
    if (m) {
        memset(&cb, 0, sizeof cb);
        CHECK(vtm_run(m, cb_capture, &cb) == VTM_OK);
        CHECK_NEAR(vtm_tensor_by_name(m, "token")->f32[0], 1.0, 1e-6);
        CHECK(cb.calls == 1);
        CHECK(cb.token == 1);
        CHECK(cb.position == 0);
        CHECK(vtm_generated(m) == 1);
        vtm_set_position(m, 5);
        CHECK(vtm_run(m, cb_capture, &cb) == VTM_OK);
        CHECK(cb.position == 5);
        CHECK(vtm_generated(m) == 2);
        vtm_reset(m);
        CHECK(vtm_generated(m) == 0);
        CHECK(vtm_position(m) == 0);
        CHECK(vtm_input_token(m) == -1);
        vtm_free(m);
    }
    free(img);
    model_free(&md);

    // Неизвестный режим SAMP — внятная ошибка, не падение.
    // imm 0..3 заняты стратегиями (задача 0051), 4 — ещё не реализована.
    model_init(&md);
    model_add_f32(&md, "logits", 1, dlog, logits);
    model_add_f32(&md, "token", 1, dto, tok);
    model_inst(&md, VTM_OP_SAMP, 0, 1, 0, VTM_NO_TENSOR, VTM_NO_TENSOR, 4);
    model_inst(&md, VTM_OP_HALT, 0, VTM_NO_TENSOR, VTM_NO_TENSOR,
               VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    model_build(&md, &img, &size);
    m = load_ok(img, size);
    CHECK(m != NULL);
    if (m) {
        CHECK(vtm_run(m, NULL, NULL) == VTM_E_UNSUPPORTED);
        vtm_free(m);
    }
    free(img);
    model_free(&md);
}

// --- SAMP: стратегии, параметры из меты, воспроизводимость -----------------

// Контейнер «logits -> SAMP(strategy) -> HALT»; dst-тензор "token" уже есть.
static vtm_t *samp_container(const float *logits, uint32_t n, const char *meta,
                             int strategy) {
    model_t md;
    uint32_t dlog[1];
    uint32_t dtok[1] = {1};
    float tokv[1] = {-1.0f};
    uint8_t *img;
    size_t size;
    vtm_t *m;

    model_init(&md);
    dlog[0] = n;
    model_add_f32(&md, "logits", 1, dlog, logits);
    model_add_f32(&md, "token", 1, dtok, tokv);
    model_inst(&md, VTM_OP_SAMP, 0, 1, 0, VTM_NO_TENSOR, VTM_NO_TENSOR,
               strategy);
    model_inst(&md, VTM_OP_HALT, 0, VTM_NO_TENSOR, VTM_NO_TENSOR,
               VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    if (meta) model_set_meta(&md, meta);
    model_build(&md, &img, &size);
    m = load_ok(img, size);
    free(img);
    model_free(&md);
    return m;
}

// Следующий выбранный токен (один прогон программы).
static int samp_next(vtm_t *m) {
    run_ok(m);
    return (int)vtm_tensor_by_name(m, "token")->f32[0];
}

// K прогонов подряд: стратегия + окно/ГПСЧ машины работают как в генерации.
static void samp_seq(vtm_t *m, int *out, int k) {
    int i;
    for (i = 0; i < k; i++) out[i] = samp_next(m);
}

// Гистограмма токенов за k прогонов; counts — массив длины n.
static void samp_counts(vtm_t *m, int *counts, uint32_t n, int k) {
    int i;
    memset(counts, 0, (size_t)n * sizeof(int));
    for (i = 0; i < k; i++) {
        uint32_t token = (uint32_t)samp_next(m);
        CHECK(token < n);
        if (token < n) counts[token]++;
    }
}

static void test_samp_strategies(void) {
    // (1) argmax детерминирован: 3 прогона и 3 отдельные загрузки — токен 1.
    {
        float logits[3] = {0.5f, 2.0f, 1.0f};
        int got[3], i;
        vtm_t *m = samp_container(logits, 3, NULL, VTM_SAMP_ARGMAX);
        CHECK(m != NULL);
        if (m) {
            samp_seq(m, got, 3);
            CHECK(got[0] == 1 && got[1] == 1 && got[2] == 1);
            vtm_free(m);
        }
        for (i = 0; i < 3; i++) {
            m = samp_container(logits, 3, NULL, VTM_SAMP_ARGMAX);
            CHECK(m != NULL);
            if (!m) break;
            CHECK(samp_next(m) == 1);
            vtm_free(m);
        }
    }

    // (2) temperature 0 == argmax: и явный 0, и отрицательная температура.
    {
        float logits[3] = {0.5f, 2.0f, 1.0f};
        static const char *const metas[] = {
            "{\"temperature\": 0.0}",
            "{\"temperature\": 0}",
            "{\"temperature\": -2.0}"
        };
        size_t i;
        for (i = 0; i < sizeof metas / sizeof metas[0]; i++) {
            int got[3];
            vtm_t *m = samp_container(logits, 3, metas[i],
                                      VTM_SAMP_TEMPERATURE);
            CHECK(m != NULL);
            if (!m) continue;
            samp_seq(m, got, 3);
            CHECK(got[0] == 1 && got[1] == 1 && got[2] == 1);
            vtm_free(m);
        }
    }

    // (3) top-k 1 == argmax: и при temperature 0, и при ненулевой.
    {
        float logits[3] = {0.5f, 2.0f, 1.0f};
        static const char *const metas[] = {
            "{\"top_k\": 1}",
            "{\"top_k\": 1, \"temperature\": 0.7}",
            "{\"top_k\": 1, \"temperature\": 2.0, \"top_p\": 0.3}"
        };
        size_t i;
        for (i = 0; i < sizeof metas / sizeof metas[0]; i++) {
            int got[3];
            vtm_t *m = samp_container(logits, 3, metas[i], VTM_SAMP_TOPK);
            CHECK(m != NULL);
            if (!m) continue;
            samp_seq(m, got, 3);
            CHECK(got[0] == 1 && got[1] == 1 && got[2] == 1);
            vtm_free(m);
        }
    }

    // (4) top-p 1.0 ничего не отсекает: последовательность побитово равна
    // стратегии temperature при том же seed (тот же ГПСЧ, тот же один draw).
    {
        float logits[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        const char *meta = "{\"temperature\": 1.0, \"top_p\": 1.0,"
                           " \"sample_seed\": 2024}";
        int temp[64], topp[64];
        vtm_t *m1 = samp_container(logits, 5, meta, VTM_SAMP_TEMPERATURE);
        vtm_t *m3 = samp_container(logits, 5, meta, VTM_SAMP_TOPP);
        CHECK(m1 != NULL && m3 != NULL);
        if (m1 && m3) {
            samp_seq(m1, temp, 64);
            samp_seq(m3, topp, 64);
            CHECK(memcmp(temp, topp, sizeof temp) == 0);
        }
        vtm_free(m1);
        vtm_free(m3);
    }

    // (5) top-p граница: 4 равных логита, p = 0.5 -> ядро ровно из двух,
    // p = 0.6 -> из трёх; токены вне ядра не выбираются.
    {
        float logits[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        int counts[4];
        vtm_t *m = samp_container(logits, 4,
                                  "{\"top_p\": 0.5, \"sample_seed\": 5}",
                                  VTM_SAMP_TOPP);
        CHECK(m != NULL);
        if (m) {
            samp_counts(m, counts, 4, 400);
            CHECK(counts[0] > 0 && counts[1] > 0);
            CHECK(counts[2] == 0 && counts[3] == 0);
            vtm_free(m);
        }
        m = samp_container(logits, 4,
                           "{\"top_p\": 0.6, \"sample_seed\": 5}",
                           VTM_SAMP_TOPP);
        CHECK(m != NULL);
        if (m) {
            samp_counts(m, counts, 4, 400);
            CHECK(counts[0] > 0 && counts[1] > 0 && counts[2] > 0);
            CHECK(counts[3] == 0);
            vtm_free(m);
        }
        // top_p <= 0 — детерминированный argmax (токен 0 при равных логитах).
        m = samp_container(logits, 4, "{\"top_p\": 0.0}", VTM_SAMP_TOPP);
        CHECK(m != NULL);
        if (m) {
            int got[3];
            samp_seq(m, got, 3);
            CHECK(got[0] == 0 && got[1] == 0 && got[2] == 0);
            vtm_free(m);
        }
    }

    // (6) temperature: распределение. logits {0, ln2} -> p = (1/3, 2/3).
    {
        float logits[2] = {0.0f, 0.6931472f};
        int counts[2];
        vtm_t *m = samp_container(logits, 2,
                                  "{\"temperature\": 1.0, \"sample_seed\": 1}",
                                  VTM_SAMP_TEMPERATURE);
        CHECK(m != NULL);
        if (m) {
            samp_counts(m, counts, 2, 3000);
            CHECK(counts[0] + counts[1] == 3000);
            CHECK(counts[1] > 1800 && counts[1] < 2200);   // ~2000 = 2/3
            vtm_free(m);
        }
        // T = 0.5: exp(ln2/0.5) = 4 -> p = (0.2, 0.8).
        m = samp_container(logits, 2,
                           "{\"temperature\": 0.5, \"sample_seed\": 1}",
                           VTM_SAMP_TEMPERATURE);
        CHECK(m != NULL);
        if (m) {
            samp_counts(m, counts, 2, 3000);
            CHECK(counts[1] > 2200 && counts[1] < 2600);   // ~2400 = 0.8
            vtm_free(m);
        }
    }

    // (7) top-k: 5 равных логитов, k = 2 -> ровно два токена, примерно поровну.
    {
        float logits[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        int counts[5];
        vtm_t *m = samp_container(logits, 5,
                                  "{\"top_k\": 2, \"sample_seed\": 3}",
                                  VTM_SAMP_TOPK);
        CHECK(m != NULL);
        if (m) {
            samp_counts(m, counts, 5, 2000);
            CHECK(counts[0] > 0 && counts[1] > 0);
            CHECK(counts[2] == 0 && counts[3] == 0 && counts[4] == 0);
            CHECK(counts[0] > 800 && counts[0] < 1200);    // ~1000
            vtm_free(m);
        }
        // Без top_k (0) стратегия 2 — просто температура: выбираются все 5.
        m = samp_container(logits, 5,
                           "{\"top_k\": 0, \"sample_seed\": 3}",
                           VTM_SAMP_TOPK);
        CHECK(m != NULL);
        if (m) {
            samp_counts(m, counts, 5, 400);
            CHECK(counts[0] > 0 && counts[4] > 0);
            vtm_free(m);
        }
    }

    // (8) штраф за повторы ломает зацикленный пример: без штрафа argmax
    // всегда выбирает токен 2, со штрафом 1.5 следующий выбор — токен 0.
    {
        float logits[4] = {0.9f, 0.0f, 1.0f, 0.0f};
        int got[3];
        vtm_t *m = samp_container(logits, 4, NULL, VTM_SAMP_ARGMAX);
        CHECK(m != NULL);
        if (m) {
            samp_seq(m, got, 3);
            CHECK(got[0] == 2 && got[1] == 2 && got[2] == 2);
            vtm_free(m);
        }
        m = samp_container(logits, 4,
                           "{\"repetition_penalty\": 1.5,"
                           " \"repetition_window\": 8}",
                           VTM_SAMP_ARGMAX);
        CHECK(m != NULL);
        if (m) {
            samp_seq(m, got, 3);
            CHECK(got[0] == 2);
            CHECK(got[1] == 0);          // 1.0 / 1.5 = 0.667 < 0.9
            CHECK(!(got[0] == 2 && got[1] == 2 && got[2] == 2));
            vtm_free(m);
        }
        // Штраф только по окну: window = 0 — выключен, поведение как без него.
        m = samp_container(logits, 4,
                           "{\"repetition_penalty\": 3.0,"
                           " \"repetition_window\": 0}",
                           VTM_SAMP_ARGMAX);
        CHECK(m != NULL);
        if (m) {
            samp_seq(m, got, 3);
            CHECK(got[0] == 2 && got[1] == 2 && got[2] == 2);
            vtm_free(m);
        }
    }

    // (8b) окно ровно N токенов: при penalty 2.0 и логитах {0.5, 0.9, 1.0, 0}
    // window = 1 штрафует только последний токен ([2,1,2]), window = 2 —
    // последние два и перевес уходит токену 0 ([2,1,0]).
    {
        float logits[4] = {0.5f, 0.9f, 1.0f, 0.0f};
        int got1[3], got2[3];
        vtm_t *m = samp_container(logits, 4,
                                  "{\"repetition_penalty\": 2.0,"
                                  " \"repetition_window\": 1}",
                                  VTM_SAMP_ARGMAX);
        CHECK(m != NULL);
        if (m) {
            samp_seq(m, got1, 3);
            CHECK(got1[0] == 2 && got1[1] == 1 && got1[2] == 2);
            vtm_free(m);
        }
        m = samp_container(logits, 4,
                           "{\"repetition_penalty\": 2.0,"
                           " \"repetition_window\": 2}",
                           VTM_SAMP_ARGMAX);
        CHECK(m != NULL);
        if (m) {
            samp_seq(m, got2, 3);
            CHECK(got2[0] == 2 && got2[1] == 1 && got2[2] == 0);
            vtm_free(m);
        }
    }

    // (9) воспроизводимость: vtm_reset() повторяет последовательность, две
    // независимые машины с тем же seed — тоже (ГПСЧ не зависит от адреса).
    {
        float logits[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        const char *meta = "{\"temperature\": 1.0, \"sample_seed\": 7}";
        int first[20], again[20], other[20], diff[20];
        vtm_t *m = samp_container(logits, 5, meta, VTM_SAMP_TEMPERATURE);
        vtm_t *m2 = samp_container(logits, 5, meta, VTM_SAMP_TEMPERATURE);
        vtm_t *m3 = samp_container(logits, 5,
                                   "{\"temperature\": 1.0,"
                                   " \"sample_seed\": 8}",
                                   VTM_SAMP_TEMPERATURE);
        CHECK(m != NULL && m2 != NULL && m3 != NULL);
        if (m && m2 && m3) {
            samp_seq(m, first, 20);
            samp_seq(m2, other, 20);
            samp_seq(m3, diff, 20);
            CHECK(memcmp(first, other, sizeof first) == 0);
            CHECK(memcmp(first, diff, sizeof first) != 0);
            vtm_reset(m);
            samp_seq(m, again, 20);
            CHECK(memcmp(first, again, sizeof first) == 0);
        }
        vtm_free(m);
        vtm_free(m2);
        vtm_free(m3);
    }

    // (10) параметры из меты: доступ через vtm_sampling_params(), вложенный
    // ключ не перебивает верхний.
    {
        vtm_sampling sp;
        vtm_t *m = mini_with_meta("{\"temperature\": 0.25, \"top_k\": 7,"
                                  " \"top_p\": 0.9,"
                                  " \"repetition_penalty\": 1.3,"
                                  " \"repetition_window\": 12,"
                                  " \"sample_seed\": 42,"
                                  " \"model\": {\"temperature\": 0.5}}");
        CHECK(m != NULL);
        if (m) {
            memset(&sp, 0, sizeof sp);
            CHECK(vtm_sampling_params(m, &sp) == 1);
            CHECK_NEAR(sp.temperature, 0.25, 1e-12);
            CHECK(sp.top_k == 7);
            CHECK_NEAR(sp.top_p, 0.9, 1e-12);
            CHECK_NEAR(sp.repetition_penalty, 1.3, 1e-12);
            CHECK(sp.repetition_window == 12);
            CHECK(sp.sample_seed == 42);
            CHECK(sp.has_seed == 1);
            vtm_free(m);
        }
        // Меты нет — дефолты vtm.h.
        m = mini_with_meta(NULL);
        CHECK(m != NULL);
        if (m) {
            memset(&sp, 0xFF, sizeof sp);
            CHECK(vtm_sampling_params(m, &sp) == 0);
            CHECK_NEAR(sp.temperature, VTM_DEFAULT_TEMPERATURE, 1e-12);
            CHECK(sp.top_k == 0);
            CHECK_NEAR(sp.top_p, VTM_DEFAULT_TOP_P, 1e-12);
            CHECK_NEAR(sp.repetition_penalty,
                       VTM_DEFAULT_REPETITION_PENALTY, 1e-12);
            CHECK(sp.repetition_window == VTM_DEFAULT_REPETITION_WINDOW);
            CHECK(sp.sample_seed == 0);
            CHECK(sp.has_seed == 0);
            vtm_free(m);
        }
        // Битая мета — тоже дефолты, без падения.
        m = mini_with_meta("{\"temperature\": ");
        CHECK(m != NULL);
        if (m) {
            memset(&sp, 0, sizeof sp);
            CHECK(vtm_sampling_params(m, &sp) == 0);
            CHECK_NEAR(sp.temperature, VTM_DEFAULT_TEMPERATURE, 1e-12);
            vtm_free(m);
        }
    }

    // (11) стратегия вне 0..3 — VTM_E_UNSUPPORTED, машина не портится.
    {
        float logits[3] = {0.5f, 2.0f, 1.0f};
        vtm_t *m = samp_container(logits, 3, NULL, 9);
        CHECK(m != NULL);
        if (m) {
            CHECK(vtm_run(m, NULL, NULL) == VTM_E_UNSUPPORTED);
            CHECK(strstr(vtm_error_message(m), "SAMP") != NULL);
            CHECK(vtm_generated(m) == 0);
            vtm_free(m);
        }
    }
}

static void test_f16_bf16(void) {
    model_t md;
    uint32_t d[1] = {2};
    float f16v[2] = {1.5f, -2.5f};
    float bf16v[2] = {3.5f, -4.5f};
    float f16y[2] = {0.0f, 0.0f};
    float bf16y[2] = {0.0f, 0.0f};
    uint8_t *img;
    size_t size;
    vtm_t *m;

    model_init(&md);
    model_add_f16(&md, "x16", 1, d, f16v);
    model_add_bf16(&md, "xbf", 1, d, bf16v);
    model_add_f32(&md, "y16", 1, d, f16y);
    model_add_f32(&md, "ybf", 1, d, bf16y);
    model_inst(&md, VTM_OP_CPY, 0, 2, 0, VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_CPY, 0, 3, 1, VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_HALT, 0, VTM_NO_TENSOR, VTM_NO_TENSOR,
               VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    model_build(&md, &img, &size);
    m = load_ok(img, size);
    CHECK(m != NULL);
    if (m) {
        CHECK(run_ok(m));
        CHECK_NEAR(vtm_tensor_by_name(m, "y16")->f32[0], 1.5, 1e-6);
        CHECK_NEAR(vtm_tensor_by_name(m, "y16")->f32[1], -2.5, 1e-6);
        CHECK_NEAR(vtm_tensor_by_name(m, "ybf")->f32[0], 3.5, 1e-6);
        CHECK_NEAR(vtm_tensor_by_name(m, "ybf")->f32[1], -4.5, 1e-6);
        vtm_free(m);
    }
    free(img);
    model_free(&md);
}

// v0.2: словарь Qwen 151936 (> 65535) должен влезать и читаться (ADR 0011).
static void test_large_dims_u32(void) {
    model_t md;
    uint8_t *img = NULL;
    size_t size = 0;
    uint32_t n = 151936u;
    uint32_t dims[1];
    float *vals;
    uint32_t i;
    vtm_t *m;
    const vtm_tensor *t;

    model_init(&md);
    dims[0] = n;
    vals = (float *)xmalloc(sizeof(float) * (size_t)n);
    for (i = 0; i < n; i++) vals[i] = (float)(i % 7u) - 3.0f;
    model_add_f32(&md, "embed", 1, dims, vals);
    model_inst(&md, VTM_OP_HALT, 0, VTM_NO_TENSOR, VTM_NO_TENSOR,
               VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    model_build(&md, &img, &size);
    m = load_ok(img, size);
    CHECK(m != NULL);
    if (m) {
        t = vtm_tensor_by_name(m, "embed");
        CHECK(t != NULL);
        if (t) {
            CHECK(t->ndim == 1);
            CHECK(t->dims[0] == n);
            CHECK(t->n_elements == n);
            CHECK_NEAR(t->f32[151935], (float)(151935u % 7u) - 3.0f, 1e-6);
        }
        vtm_free(m);
    }
    free(vals);
    free(img);
    model_free(&md);
}

// --- CLI: текст -> токены -> модель -> текст --------------------------------
//
// Интеграционный тест: test_vtm.c — C и не линкуется с C++-токенизатором,
// поэтому проверяем собранный CLI снаружи, как пользователь. Модель
// синтетическая: logits[287], максимум на id 33; фикстура tokenizer.json
// декодирует 33 как "!". Проверяем и старый путь --tokens, и ошибку без
// tokenizer.json.

#include <sys/wait.h>
#include <unistd.h>

static int text_file_ok(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    fclose(f);
    return 1;
}

static const char *find_readable(const char *const *paths) {
    int i;
    for (i = 0; paths[i]; i++)
        if (text_file_ok(paths[i])) return paths[i];
    return NULL;
}

static int read_text_file(const char *path, char *buf, size_t cap) {
    FILE *f = fopen(path, "rb");
    size_t n;
    if (!f) return -1;
    n = fread(buf, 1, cap - 1, f);
    fclose(f);
    buf[n] = '\0';
    return 0;
}

static int cli_exit_code(int rc) {
    if (rc == -1) return -1;
    if (WIFEXITED(rc)) return WEXITSTATUS(rc);
    return -1;
}

static void test_cli_text(void) {
    static const char *const cli_paths[] = {
        "./vtm", "src/vtm/vtm", "../../src/vtm/vtm", NULL
    };
    static const char *const tok_paths[] = {
        "../../tests/vtm/fixtures/tokenizer.json",
        "tests/vtm/fixtures/tokenizer.json",
        "../tests/vtm/fixtures/tokenizer.json",
        NULL
    };
    model_t md;
    uint32_t dlog[1] = {287};
    float *logits = (float *)xmalloc(287 * sizeof(float));
    uint8_t *img;
    size_t size;
    char model_path[128], out_path[128], err_path[128], missing_path[128];
    char cmd[2048], out[8192], err[8192];
    const char *cli;
    const char *fixture;
    long stamp = (long)time(NULL);
    FILE *f;
    int rc;

    out[0] = '\0';
    err[0] = '\0';
    memset(logits, 0, 287 * sizeof(float));
    logits[33] = 10.0f;

    model_init(&md);
    model_add_f32(&md, "logits", 1, dlog, logits);
    model_inst(&md, VTM_OP_SAMP, 0, VTM_NO_TENSOR, 0, VTM_NO_TENSOR,
               VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_HALT, 0, VTM_NO_TENSOR, VTM_NO_TENSOR,
               VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    model_build(&md, &img, &size);
    free(logits);

    snprintf(model_path, sizeof model_path, "/tmp/vtm_cli_text_%ld.vtm", stamp);
    snprintf(out_path, sizeof out_path, "/tmp/vtm_cli_text_%ld.out", stamp);
    snprintf(err_path, sizeof err_path, "/tmp/vtm_cli_text_%ld.err", stamp);
    snprintf(missing_path, sizeof missing_path,
             "/tmp/vtm_cli_missing_%ld.json", stamp);
    f = fopen(model_path, "wb");
    CHECK(f != NULL);
    if (!f) {
        free(img);
        model_free(&md);
        return;
    }
    CHECK(fwrite(img, 1, size, f) == size);
    fclose(f);

    cli = find_readable(cli_paths);
    fixture = find_readable(tok_paths);
    CHECK(cli != NULL);
    CHECK(fixture != NULL);
    if (cli && fixture) {
        // Текст -> токены -> модель -> текст: 33 декодируется как "!".
        snprintf(cmd, sizeof cmd,
                 "%s run %s --text hello --tokenizer %s --max-new 2 "
                 "> %s 2> %s",
                 cli, model_path, fixture, out_path, err_path);
        rc = system(cmd);
        CHECK(cli_exit_code(rc) == 0);
        CHECK(read_text_file(out_path, out, sizeof out) == 0);
        CHECK(strstr(out, "текст:   hello") != NULL);
        CHECK(strstr(out, "prompt:") != NULL);
        CHECK(strstr(out, "ответ:   !!") != NULL);

        // Старый путь --tokens не сломан.
        snprintf(cmd, sizeof cmd,
                 "%s run %s --tokens 33 --max-new 1 > %s 2> %s",
                 cli, model_path, out_path, err_path);
        rc = system(cmd);
        CHECK(cli_exit_code(rc) == 0);
        CHECK(read_text_file(out_path, out, sizeof out) == 0);
        CHECK(strstr(out, "prompt: 33") != NULL);
        CHECK(strstr(out, "вывод:  33") != NULL);

        // Нет tokenizer.json — понятная ошибка и код 2, не падение.
        snprintf(cmd, sizeof cmd,
                 "%s run %s --text hello --tokenizer %s > %s 2> %s",
                 cli, model_path, missing_path, out_path, err_path);
        rc = system(cmd);
        CHECK(cli_exit_code(rc) == 2);
        CHECK(read_text_file(err_path, err, sizeof err) == 0);
        CHECK(strstr(err, "токенизатор") != NULL);
    }

    remove(model_path);
    remove(out_path);
    remove(err_path);
    remove(missing_path);
    free(img);
    model_free(&md);
}

// Чат-шаблон Qwen2.5: --chat --system должен дать ровно эталонную строку,
// а без --chat вывод прежний (сырой текст). Спецтокены <|im_start|>/<|im_end|>
// приходят из added_tokens фикстуры (id 500/501), не из кода.
static void test_cli_chat(void) {
    static const char *const cli_paths[] = {
        "./vtm", "src/vtm/vtm", "../../src/vtm/vtm", NULL
    };
    static const char *const tok_paths[] = {
        "../../tests/vtm/fixtures/tokenizer.json",
        "tests/vtm/fixtures/tokenizer.json",
        "../tests/vtm/fixtures/tokenizer.json",
        NULL
    };
    // Эталон: <|im_start|>system\n{system}<|im_end|>\n
    //         <|im_start|>user\n{text}<|im_end|>\n
    //         <|im_start|>assistant\n
    static const char expected[] =
        "шаблон:\n"
        "<|im_start|>system\n"
        "Ты ассистент<|im_end|>\n"
        "<|im_start|>user\n"
        "hello<|im_end|>\n"
        "<|im_start|>assistant\n"
        "конец шаблона\n";
    model_t md;
    uint32_t dlog[1] = {287};
    float *logits = (float *)xmalloc(287 * sizeof(float));
    uint8_t *img;
    size_t size;
    char model_path[128], out_path[128], err_path[128];
    char cmd[2048], out[8192], err[8192];
    const char *cli;
    const char *fixture;
    long stamp = (long)time(NULL);
    FILE *f;
    int rc;

    out[0] = '\0';
    err[0] = '\0';
    memset(logits, 0, 287 * sizeof(float));
    logits[33] = 10.0f;

    model_init(&md);
    model_add_f32(&md, "logits", 1, dlog, logits);
    model_inst(&md, VTM_OP_SAMP, 0, VTM_NO_TENSOR, 0, VTM_NO_TENSOR,
               VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_HALT, 0, VTM_NO_TENSOR, VTM_NO_TENSOR,
               VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    model_build(&md, &img, &size);
    free(logits);

    snprintf(model_path, sizeof model_path, "/tmp/vtm_cli_chat_%ld.vtm", stamp);
    snprintf(out_path, sizeof out_path, "/tmp/vtm_cli_chat_%ld.out", stamp);
    snprintf(err_path, sizeof err_path, "/tmp/vtm_cli_chat_%ld.err", stamp);
    f = fopen(model_path, "wb");
    CHECK(f != NULL);
    if (!f) {
        free(img);
        model_free(&md);
        return;
    }
    CHECK(fwrite(img, 1, size, f) == size);
    fclose(f);

    cli = find_readable(cli_paths);
    fixture = find_readable(tok_paths);
    CHECK(cli != NULL);
    CHECK(fixture != NULL);
    if (cli && fixture) {
        // --chat --system: точное совпадение с эталонным шаблоном.
        snprintf(cmd, sizeof cmd,
                 "%s run %s --text hello --tokenizer %s --chat "
                 "--system \"Ты ассистент\" --max-new 1 > %s 2> %s",
                 cli, model_path, fixture, out_path, err_path);
        rc = system(cmd);
        CHECK(cli_exit_code(rc) == 0);
        CHECK(read_text_file(out_path, out, sizeof out) == 0);
        CHECK(strstr(out, expected) != NULL);
        // Промпт реально закодирован спецтокенами из added_tokens:
        // 500 = <|im_start|>, 501 = <|im_end|>, 259 = "hello".
        CHECK(strstr(out, "prompt: 500 115 121 115 116 101 109 10") != NULL);
        CHECK(strstr(out, "259 501 10 500 97 115 115 105 115 116 97 110 116")
              != NULL);

        // Без --chat — прежнее поведение: сырой текст, без шаблона.
        snprintf(cmd, sizeof cmd,
                 "%s run %s --text hello --tokenizer %s --max-new 1 "
                 "> %s 2> %s",
                 cli, model_path, fixture, out_path, err_path);
        rc = system(cmd);
        CHECK(cli_exit_code(rc) == 0);
        CHECK(read_text_file(out_path, out, sizeof out) == 0);
        CHECK(strstr(out, "шаблон:") == NULL);
        CHECK(strstr(out, "prompt: 259") != NULL);
        CHECK(strstr(out, "текст:   hello") != NULL);

        // --system без --chat — ошибка аргументов, код 2.
        snprintf(cmd, sizeof cmd,
                 "%s run %s --text hello --system \"x\" > %s 2> %s",
                 cli, model_path, out_path, err_path);
        rc = system(cmd);
        CHECK(cli_exit_code(rc) == 2);
        CHECK(read_text_file(err_path, err, sizeof err) == 0);
        CHECK(strstr(err, "--system") != NULL);
    }

    remove(model_path);
    remove(out_path);
    remove(err_path);
    free(img);
    model_free(&md);
}

// --- main ------------------------------------------------------------------

int main(void) {
    test_trit_codec();
    test_load_and_tmv();
    test_tmv_blocks();
    test_tmv_scale128();
    test_raw_trit_layout();
    test_bad_files();
    test_file_load();
    test_rmsnorm();
    test_softmax();
    test_rope();
    test_meta_rope();
    test_meta_fp16_emb_tied();
    test_meta_broken();
    test_emb();
    test_mm();
    test_elementwise();
    test_attn_plain();
    test_attn_causal();
    test_kvw_attn_cache();
    test_samp();
    test_samp_strategies();
    test_f16_bf16();
    test_large_dims_u32();
    test_cli_text();
    test_cli_chat();

    if (g_fail) {
        fprintf(stderr, "\nПРОВАЛЕНО проверок: %d\n", g_fail);
        return 1;
    }
    printf("все тесты VTM: ok\n");
    return 0;
}
