// Тесты VTM: формат, загрузчик, кодек тритов, все операции v0.1.
//
// Сборка и запуск: cd src/vtm && make test
#define _POSIX_C_SOURCE 200809L

#include "../../src/vtm/vtm.h"
#include "../../src/vtm/trit.h"

#include <limits.h>
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

// Хуки pread-кэша экспертов MoE (задача 0175). Объявлены здесь, а не через
// vtm_internal.h, по той же причине, что и хуки загрузчика ниже: тест не
// должен зависеть от внутренней раскладки struct vtm.
void vtm_internal_moe_cache_reset(void);
void vtm_internal_moe_cache_stats(uint64_t *hits, uint64_t *misses,
                                  size_t *bytes, uint64_t *allocs);
// Хуки prefetch-очереди (задача 0177): задания, ожидания, попадания prefetch,
// число I/O-потоков.
void vtm_internal_moe_prefetch_stats(uint64_t *jobs, uint64_t *waits,
                                     uint64_t *useful, int *threads);

// Внутренние точки входа кэша с явной пачкой (задача 0180). В vtm.h не
// выносятся (публичный заголовок ядра не меняется) — тест объявляет их сам,
// как и хуки выше.
vtm_status vtm_cache_prefix_ex(vtm_t *m, const int32_t *ids, int n, int batch);
vtm_status vtm_run_cached_ex(vtm_t *m, const int32_t *ids, int n, int batch);

// --- big-endian: контейнер не читается, загрузчик обязан отказать ----------
//
// Обычная сборка гоняет весь набор ниже. Сборка с -DVTM_HOST_BIG_ENDIAN=1
// эмулирует BE-хост (docs/vtm.md, «Портируемость»): контейнер .vtm
// little-endian, поэтому vtm_load_mem()/vtm_load() обязаны вернуть
// VTM_E_UNSUPPORTED с внятным текстом, а не прочитать u32/f32 как мусор.
// Проверка стоит один раз на загрузку, поэтому остальные тесты в BE-сборке
// смысла не имеют: ни один контейнер не должен загрузиться.
//
//   make clean && make CPPFLAGS="-D_POSIX_C_SOURCE=200809L -DVTM_HOST_BIG_ENDIAN=1" test_vtm
//   ./test_vtm
#if VTM_HOST_BIG_ENDIAN

// Минимальный корректный контейнер: заголовок 32 байта, ни тензоров, ни
// программы. На little-endian он грузится (test_host_endian_guard), значит
// отказ в BE-сборке — именно про хост, а не про кривой файл.
static void minimal_header(uint8_t hdr[VTM_HEADER_SIZE]) {
    memset(hdr, 0, VTM_HEADER_SIZE);
    memcpy(hdr, "VTM1", 4);
    hdr[4] = (uint8_t)(VTM_FORMAT_VERSION & 0xffu);   // версия u16, LE
    hdr[5] = (uint8_t)(VTM_FORMAT_VERSION >> 8);
}

int main(void) {
    uint8_t hdr[VTM_HEADER_SIZE];
    vtm_status st = VTM_OK;
    char err[256];
    char path[64];
    vtm_t *m;
    FILE *f;

    minimal_header(hdr);

    err[0] = '\0';
    m = vtm_load_mem(hdr, sizeof hdr, &st, err, sizeof err);
    CHECK(m == NULL);
    CHECK(st == VTM_E_UNSUPPORTED);
    CHECK(strstr(err, "big-endian") != NULL);
    CHECK(strstr(err, "Портируемость") != NULL);
    CHECK(strstr(vtm_last_error(), "big-endian") != NULL);
    if (m) vtm_free(m);

    // Тот же отказ через файл: vtm_load() идёт через тот же load_core(), в
    // том числе когда mmap-путь доступен.
    snprintf(path, sizeof path, "/tmp/vtm_be_probe_%ld.vtm", (long)time(NULL));
    f = fopen(path, "wb");
    CHECK(f != NULL);
    if (f) {
        CHECK(fwrite(hdr, 1, sizeof hdr, f) == sizeof hdr);
        fclose(f);
        m = vtm_load(path);
        CHECK(m == NULL);
        CHECK(strstr(vtm_last_error(), "big-endian") != NULL);
        if (m) vtm_free(m);
        remove(path);
    }

    if (g_fail) {
        fprintf(stderr, "\nПРОВАЛЕНО проверок: %d\n", g_fail);
        return 1;
    }
    printf("big-endian: загрузчик отказал (VTM_E_UNSUPPORTED), все тесты VTM: ok\n");
    return 0;
}

#else // !VTM_HOST_BIG_ENDIAN

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

static void put_u64(uint8_t *p, uint64_t v) {
    put_u32(p, (uint32_t)(v & 0xFFFFFFFFu));
    put_u32(p + 4, (uint32_t)(v >> 32));
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
    uint16_t flags;          // флаги записи таблицы (KV-кэш/состояние, 0176)
    uint32_t offset;
    uint32_t scale_offset;
} tensor_def;

// Значения из loader.c/vtm.h: в тесте они должны быть литералами, чтобы
// проверять формат, а не повторять приватные макросы загрузчика.
#define TEST_HEADER_FLAG_KV_OUT_OF_FILE 0x0004u
#define TEST_KV_NO_OFFSET 0xFFFFFFFFu

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

// Состояние рекуррентного слоя (CONV4/GDELTA, задача 0176): payload'а нет,
// буфер создаёт загрузчик (нули), offset — маркер KV_NO_OFFSET.
static void model_add_state(model_t *m, const char *name, int ndim,
                            const uint32_t *dims) {
    tensor_def *td = model_slot(m, name, VTM_TENSOR_F32, ndim, dims);
    td->flags = VTM_TENSOR_FLAG_STATE;
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

// Q4-сегмент (задача 0151): 18 байт заголовка, масштабы, [min Q4_1], затем
// n_groups * 16 байт. Байт j группы = значение j (младший ниббл) + j+16
// (старший), как в GGML и ядрах 0152.
static void make_q4(buf_t *seg, const uint8_t *q, uint32_t n_values,
                    const float *scales, const float *mins, int asym,
                    int fp32_scale, int include_mins) {
    uint32_t n_groups = (n_values + VTM_Q4_SCALE_BLOCK - 1u)
                      / VTM_Q4_SCALE_BLOCK;
    uint32_t g, j;
    buf_append(seg, asym ? VTM_Q4_MAGIC_ASYM : VTM_Q4_MAGIC_SYM, 4);
    buf_u8(seg, VTM_Q4_VERSION);
    buf_u8(seg, (uint8_t)(fp32_scale ? VTM_Q4_FLAG_F32 : 0));
    buf_u64(seg, n_values);
    buf_u32(seg, n_groups);
    if (asym && include_mins)
        buf_u32(seg, n_groups);  // n_mins — число min, явная проверка
    for (g = 0; g < n_groups; g++) {
        if (fp32_scale) buf_f32(seg, scales[g]);
        else buf_u16(seg, vtm_float_to_half(scales[g]));
    }
    if (asym && include_mins) {
        for (g = 0; g < n_groups; g++) {
            if (fp32_scale) buf_f32(seg, mins[g]);
            else buf_u16(seg, vtm_float_to_half(mins[g]));
        }
    }
    for (g = 0; g < n_groups; g++) {
        for (j = 0; j < 16u; j++) {
            uint32_t lo_idx = g * VTM_Q4_SCALE_BLOCK + j;
            uint32_t hi_idx = lo_idx + 16u;
            uint8_t lo = lo_idx < n_values ? q[lo_idx] : VTM_Q4_ZERO_NIBBLE;
            uint8_t hi = hi_idx < n_values ? q[hi_idx] : VTM_Q4_ZERO_NIBBLE;
            buf_u8(seg, (uint8_t)((lo & 0x0Fu) | (uint8_t)(hi << 4)));
        }
    }
}

static void model_add_q4(model_t *m, const char *name, int ndim,
                         const uint32_t *dims, const uint8_t *q,
                         const float *scales, const float *mins, int asym,
                         int fp32_scale) {
    tensor_def *td = model_slot(m, name, VTM_TENSOR_Q4, ndim, dims);
    buf_t b = {0};
    make_q4(&b, q, dims_product(ndim, dims), scales, mins, asym, fp32_scale, 1);
    td->blob = b.p;
    td->blob_len = b.len;
}

// INT8-сегмент (kind 4, задача 0044; упакован в памяти с 0170): 18 байт
// заголовка, масштабы (fp32/fp16) на блок 128, затем знаковые байты q.
static void make_int8(buf_t *seg, const int8_t *q, uint32_t n_values,
                      const float *scales, int fp32_scale) {
    uint32_t n_scales = (n_values + VTM_SCALE_BLOCK_128 - 1u)
                      / VTM_SCALE_BLOCK_128;
    uint32_t i;
    buf_append(seg, "INT8", 4);
    buf_u8(seg, 1);
    buf_u8(seg, (uint8_t)(fp32_scale ? 1 : 0));
    buf_u64(seg, n_values);
    buf_u32(seg, n_scales);
    for (i = 0; i < n_scales; i++) {
        if (fp32_scale) buf_f32(seg, scales[i]);
        else buf_u16(seg, vtm_float_to_half(scales[i]));
    }
    for (i = 0; i < n_values; i++) buf_u8(seg, (uint8_t)(int8_t)q[i]);
}

static void model_add_int8(model_t *m, const char *name, int ndim,
                           const uint32_t *dims, const int8_t *q,
                           const float *scales, int fp32_scale) {
    tensor_def *td = model_slot(m, name, VTM_TENSOR_INT8, ndim, dims);
    buf_t b = {0};
    make_int8(&b, q, dims_product(ndim, dims), scales, fp32_scale);
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
        int out_of_file = (td->flags & (VTM_TENSOR_FLAG_KV_CACHE
                                        | VTM_TENSOR_FLAG_STATE)) != 0;
        if (out_of_file) {
            // Буфер вне файла: payload'а нет, offset — маркер. Заголовок
            // обязан нести флаг KV_OUT_OF_FILE (как пишет vtm_ref).
            offsets[i] = TEST_KV_NO_OFFSET;
            soffsets[i] = 0u;
            m->header_flags |= TEST_HEADER_FLAG_KV_OUT_OF_FILE;
            continue;
        }
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
        put_u16(e + 50u, td->flags);
        if (td->blob_len
            && (td->flags & (VTM_TENSOR_FLAG_KV_CACHE
                             | VTM_TENSOR_FLAG_STATE)) == 0)
            memcpy(img + data_start + offsets[i], td->blob, td->blob_len);
        if (td->raw && td->blob2_len)
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

// Та же модель, но контейнер v0.3: запись тензора 60 байт (offset и
// scale_offset — u64), в «резерве» заголовка meta_offset u64 + meta_len u32.
static int model_build_v03(model_t *m, uint8_t **out, size_t *out_size) {
    size_t table_size = 32u + (size_t)m->n_tensors * VTM_TENSOR_ENTRY_SIZE_V03;
    size_t data_start = align32(table_size);
    size_t cursor = data_start;
    size_t meta_off = 0, program_off, total;
    uint64_t *offsets = (uint64_t *)xmalloc(sizeof(uint64_t)
                                            * (size_t)m->n_tensors);
    uint64_t *soffsets = (uint64_t *)xmalloc(sizeof(uint64_t)
                                             * (size_t)m->n_tensors);
    uint8_t *img;
    int i;

    for (i = 0; i < m->n_tensors; i++) {
        tensor_def *td = &m->tensors[i];
        int out_of_file = (td->flags & (VTM_TENSOR_FLAG_KV_CACHE
                                        | VTM_TENSOR_FLAG_STATE)) != 0;
        if (out_of_file) {
            offsets[i] = (uint64_t)TEST_KV_NO_OFFSET;
            soffsets[i] = 0u;
            m->header_flags |= TEST_HEADER_FLAG_KV_OUT_OF_FILE;
            continue;
        }
        cursor = align32(cursor);
        offsets[i] = (uint64_t)(cursor - data_start);
        cursor += td->blob_len;
        if (td->raw) {
            cursor = align32(cursor);
            soffsets[i] = (uint64_t)(cursor - data_start);
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
    put_u16(img + 4, VTM_FORMAT_VERSION_V03);
    put_u16(img + 6, m->header_flags);
    put_u32(img + 8, (uint32_t)m->n_instr);
    put_u32(img + 12, (uint32_t)m->n_tensors);
    put_u32(img + 16, (uint32_t)m->n_instr * VTM_INSTR_SIZE);
    // v0.3: program_offset не хранится; meta_offset u64 в 20..27, meta_len 28.
    put_u64(img + 20, m->meta_len ? (uint64_t)meta_off : 0);
    put_u32(img + 28, (uint32_t)m->meta_len);

    for (i = 0; i < m->n_tensors; i++) {
        tensor_def *td = &m->tensors[i];
        uint8_t *e = img + 32u + (size_t)i * VTM_TENSOR_ENTRY_SIZE_V03;
        int d;
        memcpy(e, td->name, 24);
        e[24] = td->kind;
        e[25] = td->ndim;
        for (d = 0; d < 4; d++)
            put_u32(e + 26u + (size_t)d * 4u, td->dims[d]);
        put_u64(e + 42u, td->offset = offsets[i]);
        put_u64(e + 50u, td->scale_offset = soffsets[i]);
        put_u16(e + 58u, td->flags);
        if (td->blob_len
            && (td->flags & (VTM_TENSOR_FLAG_KV_CACHE
                             | VTM_TENSOR_FLAG_STATE)) == 0)
            memcpy(img + data_start + offsets[i], td->blob, td->blob_len);
        if (td->raw && td->blob2_len)
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
    // По умолчанию теперь auto: metal (0192), иначе simd-*, иначе
    // scalar-lut243. Проверяем именно это, а не конкретное имя.
    CHECK(strcmp(vtm_backend_name(m), "scalar-lut243") == 0 ||
          strcmp(vtm_backend_name(m), "metal") == 0 ||
          strncmp(vtm_backend_name(m), "simd", 4) == 0);
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
                     * (double)scales[idx / (size_t)VTM_SCALE_BLOCK_128]
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

// Контейнер v0.3 (ADR 0019): 60-байтная запись, u64-смещения, meta_offset
// в «резерве» заголовка. Значения и TMV те же, что у v0.2.
static void test_v03_container(void) {
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
    model_set_meta(&md, "{\"rope_theta\": 10000.0}");
    model_add_trit(&md, "W", 2, dw, wt, scales, 1);
    model_add_f32(&md, "x", 1, dx, xv);
    model_add_f32(&md, "y", 1, dy, yv);
    model_inst(&md, VTM_OP_TMV, 0, 2, 0, 1, VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_HALT, 0, VTM_NO_TENSOR, VTM_NO_TENSOR,
               VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    model_build_v03(&md, &img, &size);

    CHECK(get_u32(img + 32u + 42u) == 0u);  // offset u64, младшие 32
    CHECK(get_u32(img + 32u + 46u) == 0u);  // offset u64, старшие 32
    m = load_ok(img, size);
    CHECK(m != NULL);
    if (!m) { free(img); model_free(&md); return; }
    CHECK(vtm_tensor_count(m) == 3);
    t = vtm_tensor_by_name(m, "W");
    CHECK(t != NULL && t->kind == VTM_TENSOR_TRIT);
    CHECK(t != NULL && t->layout == VTM_LAYOUT_TRT1);
    CHECK(run_ok(m));
    t = vtm_tensor_by_name(m, "y");
    CHECK(t != NULL);
    if (t) {
        CHECK_NEAR(t->f32[0], -4.0, 1e-6);
        CHECK_NEAR(t->f32[1], 10.0, 1e-6);
    }
    CHECK(strcmp(vtm_meta_json(m), "{\"rope_theta\": 10000.0}") == 0);
    vtm_free(m);

    // v0.3 с битой версией таблицы не должен читаться как v0.2: это
    // отдельная версия, а не флаг.
    put_u16(img + 4, 99);
    m = vtm_load_mem(img, size, NULL, NULL, 0);
    CHECK(m == NULL);
    CHECK(strstr(vtm_last_error(), "версия контейнера") != NULL);
    free(img);
    model_free(&md);
}

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

// Порядок байт: хост little-endian, как и контейнер, — обычная загрузка
// проходит. Парный тест к BE-ветви в начале файла: там отказ, здесь успех на
// том же минимальном заголовке, значит отказ был именно из-за хоста.
static void test_host_endian_guard(void) {
    uint8_t hdr[VTM_HEADER_SIZE];
    vtm_status st = VTM_OK;
    char err[256];
    vtm_t *m;

    CHECK(VTM_HOST_BIG_ENDIAN == 0);
    memset(hdr, 0, sizeof hdr);
    memcpy(hdr, "VTM1", 4);
    hdr[4] = (uint8_t)(VTM_FORMAT_VERSION & 0xffu);
    hdr[5] = (uint8_t)(VTM_FORMAT_VERSION >> 8);

    err[0] = '\0';
    m = vtm_load_mem(hdr, sizeof hdr, &st, err, sizeof err);
    CHECK(m != NULL);
    CHECK(st == VTM_OK);
    CHECK(err[0] == '\0');
    if (m) {
        CHECK(vtm_tensor_count(m) == 0);
        CHECK(vtm_instr_count(m) == 0);
        vtm_free(m);
    }
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

// kind 4 (int8): таблица словаря остаётся УПАКОВАННОЙ в памяти (0170).
// До 0170 загрузчик разворачивал её в fp32 в арене (1,24 ГиБ у словаря 3B) —
// теперь в арене только fp32-масштабы, байты q читаются из отображения.
// Проверяем: kind остаётся int8, t->f32 == NULL; EMB разворачивает строку, а
// MM (tied lm_head) считает GEMV по упакованной таблице — результат равен
// эталону q·scale·x. VTM_I8_EMB=0 возвращает прежний путь (kind 1) с тем же
// результатом.
static void test_int8_packed(void) {
    static const uint32_t de[2] = {3, 256};
    static const uint32_t dx[2] = {256, 1};
    static const uint32_t dy[2] = {3, 1};
    int fp32_scale;
    int saved_mode = vtm_i8_emb_mode();

    for (fp32_scale = 1; fp32_scale >= 0; fp32_scale--) {
        model_t md;
        int8_t q[3 * 256];
        float scales[3 * 256 / VTM_SCALE_BLOCK_128];
        float xv[256];
        float yv[3] = {0.0f, 0.0f, 0.0f};
        float ev[256];
        uint8_t *img;
        size_t size;
        vtm_t *m;
        uint32_t seed = 0x1700u + (uint32_t)fp32_scale;
        double want[3];
        /* Масштабы могут быть fp16: тогда эталон — округлённый масштаб, а
           допуск сравнения относительный (fp16 даёт ~5e-4). */
        double rel = fp32_scale ? 1e-6 : 2e-3;
        int i, r;

        for (i = 0; i < 3 * 256; i++) {
            seed = seed * 1103515245u + 12345u;
            q[i] = (int8_t)((int)((seed >> 16) % 255u) - 127);
        }
        for (i = 0; i < (int)(3u * 256u / VTM_SCALE_BLOCK_128); i++)
            scales[i] = 0.004f + 0.01f * (float)i;
        for (i = 0; i < 256; i++) {
            seed = seed * 1103515245u + 12345u;
            xv[i] = (float)((int)((seed >> 16) % 200u) - 100) / 100.0f;
        }
        for (i = 0; i < 256; i++) ev[i] = 0.0f;
        for (r = 0; r < 3; r++) {
            double acc = 0.0;
            for (i = 0; i < 256; i++) {
                size_t idx = (size_t)r * 256u + (size_t)i;
                acc += (double)q[idx]
                     * (double)scales[idx / (size_t)VTM_SCALE_BLOCK_128]
                     * (double)xv[i];
            }
            want[r] = acc;
        }

        model_init(&md);
        model_add_int8(&md, "E", 2, de, q, scales, fp32_scale);
        model_add_f32(&md, "x", 2, dx, xv);
        model_add_f32(&md, "logits", 2, dy, yv);
        model_add_f32(&md, "e_out", 2, dx, ev);
        model_inst(&md, VTM_OP_EMB, 0, 3, 0, VTM_NO_TENSOR, VTM_NO_TENSOR, 1);
        model_inst(&md, VTM_OP_MM, 0, 2, 0, 1, VTM_NO_TENSOR, 0);
        model_inst(&md, VTM_OP_HALT, 0, VTM_NO_TENSOR, VTM_NO_TENSOR,
                   VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
        model_build(&md, &img, &size);

        // 1. Упакованный путь (по умолчанию): kind остаётся int8, fp32-копии
        //    таблицы в памяти нет.
        CHECK(vtm_i8_emb_set_mode(1) == 1);
        m = load_ok(img, size);
        CHECK(m != NULL);
        if (m) {
            const vtm_tensor *e = vtm_tensor_by_name(m, "E");

            CHECK(e != NULL);
            if (e) {
                CHECK(e->kind == VTM_TENSOR_INT8);
                CHECK(e->i8 != NULL);
                CHECK(e->f32 == NULL);
                CHECK(e->scales != NULL);
                CHECK(e->scale_block == VTM_SCALE_BLOCK_128);
            }
            CHECK(run_ok(m));
            for (i = 0; i < 256; i++) {
                double want_e = (double)q[256 + i]
                              * (double)scales[(size_t)(256 + i)
                                               / VTM_SCALE_BLOCK_128];
                CHECK_NEAR(vtm_tensor_by_name(m, "e_out")->f32[i], want_e,
                           rel * (want_e > 1.0 ? want_e
                                               : (want_e < -1.0 ? -want_e
                                                                : 1.0)));
            }
            for (r = 0; r < 3; r++)
                CHECK_NEAR(vtm_tensor_by_name(m, "logits")->f32[r], want[r],
                           rel * (want[r] > 1.0 ? want[r]
                                               : (want[r] < -1.0 ? -want[r]
                                                                 : 1.0)));
            vtm_free(m);
        }

        // 2. Прежний путь VTM_I8_EMB=0: fp32-развёртка в арене (kind 1),
        //    результат тот же — эталон считает по тем же q·scale.
        CHECK(vtm_i8_emb_set_mode(0) == 0);
        m = load_ok(img, size);
        CHECK(m != NULL);
        if (m) {
            const vtm_tensor *e = vtm_tensor_by_name(m, "E");

            CHECK(e != NULL);
            if (e) {
                CHECK(e->kind == VTM_TENSOR_F32);
                CHECK(e->f32 != NULL);
                CHECK((e->flags & VTM_TENSOR_FLAG_INT8_SOURCE) != 0);
            }
            CHECK(run_ok(m));
            for (r = 0; r < 3; r++)
                CHECK_NEAR(vtm_tensor_by_name(m, "logits")->f32[r], want[r],
                           rel * (want[r] > 1.0 ? want[r]
                                               : (want[r] < -1.0 ? -want[r]
                                                                 : 1.0)));
            vtm_free(m);
        }
        vtm_i8_emb_set_mode(saved_mode);
        free(img);
        model_free(&md);
    }
    vtm_i8_emb_set_mode(-1);
    printf("int8-таблица: упакована в памяти, EMB-строка и MM по ней — ok\n");
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

// --- ATTN: блочный онлайн-softmax против наивного эталона (задача 0076) ----

// Наивный ATTN, как до 0076: массив scores на весь видимый диапазон, max,
// сумма exp, нормировка, накопление V — всё в double. В plain-раскладке
// ключ j — строка j, в 2D-кэше — строка j % ctx (кольцо). lo/hi задают
// видимый диапазон в глобальных позициях; causal-маска поверх: j > q_start+qi
// скрыт. Нужен, чтобы сравнивать с ним блочный вариант при любых ctx.
static void attn_reference(const float *q, const float *k, const float *v,
                           float *out, int n_heads, int q_len, int head_dim,
                           int n_kv_heads, int kv_len, int ctx, int plain,
                           int q_start, int causal,
                           const int *lo, const int *hi) {
    int rep = n_heads / n_kv_heads;
    int kv_dim = n_kv_heads * head_dim;
    double scale = 1.0 / sqrt((double)head_dim);
    int h, qi, j, d;

    for (h = 0; h < n_heads; h++) {
        int kvh = h / rep;
        for (qi = 0; qi < q_len; qi++) {
            const float *qrow = q + ((size_t)h * (size_t)q_len + (size_t)qi)
                                * (size_t)head_dim;
            float *orow = out + ((size_t)h * (size_t)q_len + (size_t)qi)
                          * (size_t)head_dim;
            int cnt = hi[qi] - lo[qi] + 1;
            double *scores = (double *)xmalloc((size_t)cnt * sizeof(double));
            double mx = -INFINITY, sum = 0.0;

            for (j = lo[qi]; j <= hi[qi]; j++) {
                int b = j - lo[qi];
                const float *krow;
                double dot = 0.0;
                if (causal && j > q_start + qi) {
                    scores[b] = -INFINITY;
                    continue;
                }
                if (plain)
                    krow = k + ((size_t)kvh * (size_t)kv_len + (size_t)j)
                           * (size_t)head_dim;
                else
                    krow = k + (size_t)(j % ctx) * (size_t)kv_dim
                           + (size_t)kvh * (size_t)head_dim;
                for (d = 0; d < head_dim; d++)
                    dot += (double)qrow[d] * (double)krow[d];
                scores[b] = dot * scale;
                if (scores[b] > mx) mx = scores[b];
            }
            if (mx == -INFINITY) {
                memset(orow, 0, (size_t)head_dim * sizeof(float));
                free(scores);
                continue;
            }
            for (j = lo[qi]; j <= hi[qi]; j++) {
                int b = j - lo[qi];
                if (scores[b] == -INFINITY) continue;
                sum += exp(scores[b] - mx);
            }
            if (!(sum > 0.0) || !isfinite(sum)) {
                memset(orow, 0, (size_t)head_dim * sizeof(float));
                free(scores);
                continue;
            }
            for (d = 0; d < head_dim; d++) {
                double acc = 0.0;
                for (j = lo[qi]; j <= hi[qi]; j++) {
                    int b = j - lo[qi];
                    const float *vrow;
                    double p;
                    if (scores[b] == -INFINITY) continue;
                    p = exp(scores[b] - mx) / sum;
                    if (p == 0.0) continue;
                    if (plain)
                        vrow = v + ((size_t)kvh * (size_t)kv_len + (size_t)j)
                               * (size_t)head_dim;
                    else
                        vrow = v + (size_t)(j % ctx) * (size_t)kv_dim
                               + (size_t)kvh * (size_t)head_dim;
                    acc += p * (double)vrow[d];
                }
                orow[d] = (float)acc;
            }
            free(scores);
        }
    }
}

static uint32_t g_attn_rng = 0x12345678u;

static float attn_rand(void) {
    g_attn_rng ^= g_attn_rng << 13;
    g_attn_rng ^= g_attn_rng >> 17;
    g_attn_rng ^= g_attn_rng << 5;
    return (float)((double)(g_attn_rng & 0xFFFFu) / 32768.0 - 1.0);
}

typedef struct {
    const char *name;
    int heads, kv_heads, head_dim, q_len, kv_len, plain, causal, q_start;
} attn_case;

// Одна конфигурация: собираем контейнер, гоняем ATTN и сравниваем с наивным
// эталоном. kv_len — длина 3D K/V (plain) или длина 2D-кэша (ctx).
static void run_attn_case(const attn_case *c) {
    model_t md;
    uint32_t dq[3], dk[3], dv[3], dout[3];
    float *qv, *kv, *vv, *ov, *expect;
    uint8_t *img;
    size_t size;
    vtm_t *m;
    int kv_dim = c->kv_heads * c->head_dim;
    int nq = c->heads * c->q_len * c->head_dim;
    int nk = c->plain ? c->kv_heads * c->kv_len * c->head_dim
                      : c->kv_len * kv_dim;
    int lo_buf[64], hi_buf[64];
    int q_start = c->plain ? c->kv_len - c->q_len : c->q_start;
    int key_count = c->plain ? c->kv_len : c->q_start + c->q_len;
    int qi, i;
    double maxdiff = 0.0;

    qv = (float *)xmalloc((size_t)nq * sizeof(float));
    kv = (float *)xmalloc((size_t)nk * sizeof(float));
    vv = (float *)xmalloc((size_t)nk * sizeof(float));
    ov = (float *)calloc((size_t)nq, sizeof(float));
    expect = (float *)xmalloc((size_t)nq * sizeof(float));
    if (!ov) { fprintf(stderr, "OOM\n"); exit(2); }
    for (i = 0; i < nq; i++) qv[i] = attn_rand();
    for (i = 0; i < nk; i++) kv[i] = attn_rand();
    for (i = 0; i < nk; i++) vv[i] = attn_rand();

    dq[0] = (uint32_t)c->heads; dq[1] = (uint32_t)c->q_len;
    dq[2] = (uint32_t)c->head_dim;
    if (c->plain) {
        dk[0] = (uint32_t)c->kv_heads; dk[1] = (uint32_t)c->kv_len;
        dk[2] = (uint32_t)c->head_dim;
        dv[0] = dk[0]; dv[1] = dk[1]; dv[2] = dk[2];
    } else {
        dk[0] = (uint32_t)c->kv_len; dk[1] = (uint32_t)kv_dim; dk[2] = 0;
        dv[0] = dk[0]; dv[1] = dk[1]; dv[2] = 0;
    }
    dout[0] = dq[0]; dout[1] = dq[1]; dout[2] = dq[2];

    model_init(&md);
    model_add_f32(&md, "Q", 3, dq, qv);
    model_add_f32(&md, "K", c->plain ? 3 : 2, dk, kv);
    model_add_f32(&md, "V", c->plain ? 3 : 2, dv, vv);
    model_add_f32(&md, "O", 3, dout, ov);
    model_inst(&md, VTM_OP_ATTN,
               c->causal ? VTM_FLAG_CAUSAL : 0, 3, 0, 1, 2,
               c->plain ? 0 : c->q_start);
    model_inst(&md, VTM_OP_HALT, 0, VTM_NO_TENSOR, VTM_NO_TENSOR,
               VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    model_build(&md, &img, &size);

    for (qi = 0; qi < c->q_len; qi++) {
        lo_buf[qi] = 0;
        hi_buf[qi] = key_count - 1;
    }
    attn_reference(qv, kv, vv, expect, c->heads, c->q_len, c->head_dim,
                   c->kv_heads, c->kv_len, c->kv_len, c->plain,
                   q_start, c->causal, lo_buf, hi_buf);

    m = load_ok(img, size);
    CHECK(m != NULL);
    if (m) {
        float *got;
        CHECK(run_ok(m));
        got = vtm_tensor_by_name(m, "O")->f32;
        for (i = 0; i < nq; i++) {
            double diff = fabs((double)got[i] - (double)expect[i]);
            if (diff > maxdiff) maxdiff = diff;
            CHECK_NEAR(got[i], expect[i], 1e-5);
        }
        vtm_free(m);
    }
    printf("ATTN %-26s: max|Δ| = %.3g\n", c->name, maxdiff);

    free(qv); free(kv); free(vv); free(ov); free(expect);
    free(img);
    model_free(&md);
}

static void test_attn_tiled_vs_reference(void) {
    static const attn_case cases[] = {
        // 3D без кэша: один блок, каузальная маска.
        {"plain causal q4", 1, 1, 4, 4, 4, 1, 1, 0},
        // 3D без кэша + GQA (4 головы / 2 kv), non-causal, q_len < kv_len.
        {"plain gqa q3 noncausal", 4, 2, 8, 3, 8, 1, 0, 0},
        // 2D-кэш: один блок (13 ключей), как прежний ATTN.
        {"cache causal q1 ctx16", 4, 1, 8, 1, 16, 0, 1, 12},
        // 2D-кэш: 128 ключей = два блока по 64.
        {"cache causal q1 ctx128", 4, 2, 8, 1, 128, 0, 1, 127},
        // 2D-кэш: 106 ключей, q_len 6, non-causal, два неполных блока.
        {"cache noncausal q6 ctx128", 4, 2, 6, 6, 128, 0, 0, 100},
    };
    size_t i;
    for (i = 0; i < sizeof cases / sizeof cases[0]; i++)
        run_attn_case(&cases[i]);
}

// --- ATTN: окно контекста и ротация K/V (задача 0076) ----------------------

enum {
    WIN_CTX = 8,
    WIN_STEPS = 16,
    WIN_H = 2,
    WIN_KVD = 4
};

static float win_k(int pos, int d) {
    return (float)(((pos * 7 + d * 3) % 11) - 5) / 5.0f;
}
static float win_v(int pos, int d) {
    return (float)(((pos * 5 + d * 9) % 13) - 6) / 6.0f;
}
static float win_q(int h, int pos, int d) {
    return (float)(((pos * 3 + h * 5 + d) % 9) - 4) / 4.0f;
}

// ROPE одной строки (half-раскладка, theta=10000 — дефолт без меты). Кэш
// должен хранить поворот по ГЛОБАЛЬНОЙ позиции, а не по номеру слота.
static void rope_ref_row(const float *in, float *out, int pos, int head_dim) {
    int half = head_dim / 2, p;
    for (p = 0; p < half; p++) {
        double inv = pow(10000.0, -2.0 * (double)p / (double)head_dim);
        double angle = (double)pos * inv;
        double co = cos(angle), si = sin(angle);
        out[p] = (float)((double)in[p] * co - (double)in[p + half] * si);
        out[p + half] = (float)((double)in[p + half] * co
                                + (double)in[p] * si);
    }
}

// Прогон steps подряд: каждый шаг пишет K/V позиции pos через KVW (кольцо
// win) и считает ATTN одним запросом. Результат сверяется с наивным эталоном
// по последним win токенам — то есть проверяется и ротация, и то, что свежие
// токены не теряются.
static void run_window_case(int win, int steps) {
    model_t md;
    uint32_t dkc[2] = {WIN_CTX, WIN_KVD}, dvc[2] = {WIN_CTX, WIN_KVD};
    uint32_t dq[3] = {WIN_H, 1, WIN_KVD}, dout[3] = {WIN_H, 1, WIN_KVD};
    uint32_t dvec[1] = {WIN_KVD};
    float zeros[WIN_CTX * WIN_KVD];
    float k0[WIN_KVD], v0[WIN_KVD], q0[WIN_H * WIN_KVD];
    float o0[WIN_H * WIN_KVD];
    float *kcf, *vcf, *qf, *of;
    float expect[WIN_H * WIN_KVD];
    uint8_t *img;
    size_t size;
    vtm_t *m;
    int pos, d, h, lo;
    int lo_buf[1], hi_buf[1];
    char meta[64];

    memset(zeros, 0, sizeof zeros);
    memset(o0, 0, sizeof o0);
    memset(k0, 0, sizeof k0);
    memset(v0, 0, sizeof v0);
    memset(q0, 0, sizeof q0);
    model_init(&md);
    model_add_f32(&md, "Kc", 2, dkc, zeros);
    model_add_f32(&md, "K", 1, dvec, k0);
    model_add_f32(&md, "Vc", 2, dvc, zeros);
    model_add_f32(&md, "V", 1, dvec, v0);
    model_add_f32(&md, "Q", 3, dq, q0);
    model_add_f32(&md, "O", 3, dout, o0);
    if (win != WIN_CTX) {
        snprintf(meta, sizeof meta, "{\"context_window\": %d}", win);
        model_set_meta(&md, meta);
    }
    model_inst(&md, VTM_OP_ROPE, 0, 1, 1, 1, VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_KVW, 0, 0, 1, 2, 3, 0);
    model_inst(&md, VTM_OP_ATTN, VTM_FLAG_CAUSAL, 5, 4, 0, 2, 0);
    model_inst(&md, VTM_OP_HALT, 0, VTM_NO_TENSOR, VTM_NO_TENSOR,
               VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    model_build(&md, &img, &size);
    m = load_ok(img, size);
    CHECK(m != NULL);
    if (!m) {
        free(img);
        model_free(&md);
        return;
    }
    kcf = vtm_tensor_by_name(m, "Kc")->f32;
    vcf = vtm_tensor_by_name(m, "Vc")->f32;
    qf = vtm_tensor_by_name(m, "Q")->f32;
    of = vtm_tensor_by_name(m, "O")->f32;

    for (pos = 0; pos < steps; pos++) {
        int slot = pos % win;
        float knew[WIN_KVD], kexp[WIN_KVD];
        for (d = 0; d < WIN_KVD; d++) {
            knew[d] = win_k(pos, d);
            vtm_tensor_by_name(m, "K")->f32[d] = knew[d];
            vtm_tensor_by_name(m, "V")->f32[d] = win_v(pos, d);
        }
        // ROPE инструкции крутит K по глобальной позиции pos; в кэш должен
        // попасть именно этот вектор, а не поворот по номеру слота.
        rope_ref_row(knew, kexp, pos, WIN_KVD);
        for (h = 0; h < WIN_H; h++)
            for (d = 0; d < WIN_KVD; d++)
                qf[h * WIN_KVD + d] = win_q(h, pos, d);

        vtm_set_position(m, pos);
        CHECK(run_ok(m));

        // KVW положил K/V позиции pos именно в слот pos % win.
        for (d = 0; d < WIN_KVD; d++)
            CHECK_NEAR(kcf[slot * WIN_KVD + d], kexp[d], 1e-6);

        // Эталон видит ровно последние win ключей (win=контекст кэша).
        lo = pos - win + 1;
        if (lo < 0) lo = 0;
        lo_buf[0] = lo;
        hi_buf[0] = pos;
        attn_reference(qf, kcf, vcf, expect, WIN_H, 1, WIN_KVD, 1,
                       WIN_CTX, win, 0, pos, 0, lo_buf, hi_buf);
        for (h = 0; h < WIN_H; h++)
            for (d = 0; d < WIN_KVD; d++)
                CHECK_NEAR(of[h * WIN_KVD + d], expect[h * WIN_KVD + d], 1e-6);
    }
    printf("ATTN окно win=%d (%d шагов, 2×+): ok\n", win, steps);
    vtm_free(m);
    free(img);
    model_free(&md);
}

static void test_attn_window_rotation(void) {
    // win == длина кэша: ротация начинается на позиции ctx.
    run_window_case(WIN_CTX, 2 * WIN_CTX);
    // win < длины кэша: context_window из меты (кольцо меньшего размера).
    run_window_case(4, 12);
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
// --- Q4 (kind 5, задача 0151) ----------------------------------------------
//
// Загрузчик обязан разобрать сегмент и оставить веса УПАКОВАННЫМИ: указатель
// на нибблы — words (16 байт на группу 32), fp32-масштабы — scales, у Q4_1 —
// q4_mins; f32 при этом NULL (никакой материализации в fp32-арену).
static void test_q4_load(void) {
    model_t md;
    uint32_t dims[2] = {2, 19};  // 38 значений, 2 группы по 32 (хвост 6)
    uint8_t q[38];
    float scales[2] = {0.5f, 0.25f};
    float mins[2] = {-1.0f, 2.0f};
    uint8_t *img = NULL;
    size_t size = 0;
    vtm_t *m;
    const vtm_tensor *t;
    const uint8_t *bytes;
    uint32_t i;

    for (i = 0; i < 38u; i++) q[i] = (uint8_t)((i * 7u) % 16u);
    model_init(&md);
    model_add_q4(&md, "w", 2, dims, q, scales, mins, 0, 1);
    model_inst(&md, VTM_OP_HALT, 0, VTM_NO_TENSOR, VTM_NO_TENSOR,
               VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    model_build(&md, &img, &size);
    m = load_ok(img, size);
    CHECK(m != NULL);
    if (m) {
        t = vtm_tensor_by_name(m, "w");
        CHECK(t != NULL);
        if (t) {
            bytes = (const uint8_t *)(const void *)t->words;
            CHECK(t->kind == VTM_TENSOR_Q4);
            CHECK(t->n_elements == 38u);
            CHECK(t->n_scales == 2u);
            CHECK(t->scale_block == VTM_Q4_SCALE_BLOCK);
            CHECK(t->scale_is_f32 == 1u);
            CHECK(t->q4_is_asym == 0u);
            CHECK(t->q4_mins == NULL);
            CHECK(t->f32 == NULL);           // в fp32 не материализован
            CHECK(t->words != NULL);
            CHECK_NEAR(t->scales[0], 0.5f, 1e-7);
            CHECK_NEAR(t->scales[1], 0.25f, 1e-7);
            // Байт j группы: значение j (low) + j+16 (high).
            CHECK(bytes[0] == (uint8_t)(q[0] | (q[16] << 4)));
            CHECK(bytes[1] == (uint8_t)(q[1] | (q[17] << 4)));
            // Хвост второй группы: за пределами n_values — padding 8/8.
            CHECK(bytes[16 + 15] == 0x88u);
        }
        vtm_free(m);
    }
    free(img);
    model_free(&md);
}

static void test_q4_asym_load(void) {
    model_t md;
    uint32_t dims[1] = {64};
    uint8_t q[64];
    float scales[2] = {0.5f, 0.25f};
    float mins[2] = {-1.0f, 2.0f};
    uint8_t *img = NULL;
    size_t size = 0;
    vtm_t *m;
    const vtm_tensor *t = NULL;
    uint32_t i;

    for (i = 0; i < 64u; i++) q[i] = (uint8_t)(i % 16u);
    model_init(&md);
    model_add_q4(&md, "w", 1, dims, q, scales, mins, 1, 1);
    model_inst(&md, VTM_OP_HALT, 0, VTM_NO_TENSOR, VTM_NO_TENSOR,
               VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    model_build(&md, &img, &size);
    m = load_ok(img, size);
    CHECK(m != NULL);
    if (m) {
        t = vtm_tensor_by_name(m, "w");
        CHECK(t != NULL);
        if (t) {
            CHECK(t->kind == VTM_TENSOR_Q4);
            CHECK(t->q4_is_asym == 1u);
            CHECK(t->q4_mins != NULL);
            if (t->q4_mins) {
                CHECK_NEAR(t->q4_mins[0], -1.0f, 1e-7);
                CHECK_NEAR(t->q4_mins[1], 2.0f, 1e-7);
            }
            CHECK(t->f32 == NULL);
        }
        vtm_free(m);
    }
    free(img);
    model_free(&md);
}

static void test_q4_bad(void) {
    model_t md;
    uint32_t dims[1] = {64};
    uint8_t q[64];
    float scales[2] = {0.5f, 0.25f};
    float mins[2] = {-1.0f, 2.0f};
    float neg = -1.0f;
    uint8_t *img = NULL;
    size_t size = 0;
    vtm_status st;
    char err[256];
    uint32_t i;
    tensor_def *td;

    for (i = 0; i < 64u; i++) q[i] = (uint8_t)(i % 16u);
    model_init(&md);
    model_add_q4(&md, "w", 1, dims, q, scales, mins, 0, 1);
    model_inst(&md, VTM_OP_HALT, 0, VTM_NO_TENSOR, VTM_NO_TENSOR,
               VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    td = &md.tensors[0];

    // Чужой magic.
    memcpy(td->blob, "Q4_9", 4);
    model_build(&md, &img, &size);
    st = VTM_OK; err[0] = '\0';
    CHECK(vtm_load_mem(img, size, &st, err, sizeof err) == NULL);
    CHECK(st == VTM_E_FORMAT);
    CHECK(strstr(err, "Q4_0/Q4_1") != NULL);
    free(img); img = NULL;
    memcpy(td->blob, VTM_Q4_MAGIC_SYM, 4);

    // Чужая версия сегмента.
    td->blob[4] = 2;
    model_build(&md, &img, &size);
    st = VTM_OK; err[0] = '\0';
    CHECK(vtm_load_mem(img, size, &st, err, sizeof err) == NULL);
    CHECK(st == VTM_E_VERSION);
    free(img); img = NULL;
    td->blob[4] = VTM_Q4_VERSION;

    // Неизвестный флаг.
    td->blob[5] |= 0x02u;
    model_build(&md, &img, &size);
    st = VTM_OK; err[0] = '\0';
    CHECK(vtm_load_mem(img, size, &st, err, sizeof err) == NULL);
    CHECK(st == VTM_E_FORMAT);
    free(img); img = NULL;
    td->blob[5] = VTM_Q4_FLAG_F32;

    // n_values не совпадает с dims.
    put_u64(td->blob + 6, 63);
    model_build(&md, &img, &size);
    st = VTM_OK; err[0] = '\0';
    CHECK(vtm_load_mem(img, size, &st, err, sizeof err) == NULL);
    CHECK(st == VTM_E_FORMAT);
    free(img); img = NULL;
    put_u64(td->blob + 6, 64);

    // n_groups не ceil(n/32).
    put_u32(td->blob + 14, 3);
    model_build(&md, &img, &size);
    st = VTM_OK; err[0] = '\0';
    CHECK(vtm_load_mem(img, size, &st, err, sizeof err) == NULL);
    CHECK(st == VTM_E_FORMAT);
    free(img); img = NULL;
    put_u32(td->blob + 14, 2);

    // Отрицательный масштаб.
    memcpy(td->blob + VTM_Q4_HEADER_SIZE, &neg, sizeof neg);
    model_build(&md, &img, &size);
    st = VTM_OK; err[0] = '\0';
    CHECK(vtm_load_mem(img, size, &st, err, sizeof err) == NULL);
    CHECK(st == VTM_E_SCALE);
    free(img); img = NULL;
    model_free(&md);

    // Q4_1 с чужим n_mins: явное число min не сходится с n_groups.
    model_init(&md);
    model_add_q4(&md, "w", 1, dims, q, scales, mins, 1, 1);
    td = &md.tensors[0];
    put_u32(td->blob + VTM_Q4_HEADER_SIZE, 3);
    model_inst(&md, VTM_OP_HALT, 0, VTM_NO_TENSOR, VTM_NO_TENSOR,
               VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    model_build(&md, &img, &size);
    st = VTM_OK; err[0] = '\0';
    CHECK(vtm_load_mem(img, size, &st, err, sizeof err) == NULL);
    CHECK(st == VTM_E_FORMAT);
    CHECK(strstr(err, "n_mins=3") != NULL);
    free(img); img = NULL;
    model_free(&md);

    // Q4_1 без min: сегмент короче на n_groups * 4 байт. Это VTM_E_FORMAT,
    // а не тихое чтение как Q4_0 (требование координации с 0152).
    model_init(&md);
    td = model_slot(&md, "w", VTM_TENSOR_Q4, 1, dims);
    {
        buf_t b = {0};
        make_q4(&b, q, 64, scales, mins, 1, 1, 0);
        td->blob = b.p;
        td->blob_len = b.len;
    }
    model_inst(&md, VTM_OP_HALT, 0, VTM_NO_TENSOR, VTM_NO_TENSOR,
               VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    model_build(&md, &img, &size);
    st = VTM_OK; err[0] = '\0';
    CHECK(vtm_load_mem(img, size, &st, err, sizeof err) == NULL);
    CHECK(st == VTM_E_FORMAT);
    CHECK(strstr(err, "Q4_1") != NULL);
    free(img); img = NULL;
    model_free(&md);
}

// Старый бинарь узнавал kind 5/6/7 как «неизвестный kind» и отказывал до
// чтения данных; новый обязан так же отвергать любой неизвестный kind
// (здесь 8), а не угадывать.
static void test_q4_unknown_kind_rejected(void) {
    model_t md;
    uint32_t dims[1] = {1};
    uint8_t *img = NULL;
    size_t size = 0;
    vtm_status st = VTM_OK;
    char err[256];

    model_init(&md);
    model_slot(&md, "bad", 8, 1, dims);
    model_inst(&md, VTM_OP_HALT, 0, VTM_NO_TENSOR, VTM_NO_TENSOR,
               VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    model_build(&md, &img, &size);
    err[0] = '\0';
    CHECK(vtm_load_mem(img, size, &st, err, sizeof err) == NULL);
    CHECK(st == VTM_E_FORMAT);
    CHECK(strstr(err, "неизвестный kind 8") != NULL);
    free(img);
    model_free(&md);
}

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

// --- CLI --stream: безопасный UTF-8 и стоп-токены --------------------------
//
// Синтетическая «модель-цепочка»: EMB с флагом VTM_FLAG_EMB_INPUT берёт
// строку таблицы E по входному токену, SAMP выбирает argmax, CLI подаёт
// выбранный токен на следующем шаге. Так программа выдаёт заранее заданную
// последовательность id — на ней видно, как склеиваются байты, разрезанные
// между токенами.
//
// Фикстура tokenizer.json: 259 = "hello" (промпт), 208 = байт 0xD0,
// 191 = байт 0xBF (вместе — «п»), 65 = "A", 66 = "B", 501 = <|im_end|>
// (model.eos_token — стоп-токен), 503 = <|im|> (спецтокен, но не стоп).

#define STREAM_MAX_ID        504  /* 503 — максимальный id added_tokens;
                                     столько же логитов у фикстуры */
#define STREAM_PROMPT_TOKEN  259  /* "hello" в фикстуре */

// Собрать модель-цепочку, прогнать CLI и вернуть его код возврата.
// out/err — stdout/stderr процесса; stream != 0 — добавить --stream.
static int chain_cli_run(const int *chain, int n_chain, int max_new,
                         int stream, const char *extra, char *out,
                         size_t out_cap, char *err, size_t err_cap) {
    static const char *const cli_paths[] = {
        "./vtm", "src/vtm/vtm", "../../src/vtm/vtm", NULL
    };
    static const char *const tok_paths[] = {
        "../../tests/vtm/fixtures/tokenizer.json",
        "tests/vtm/fixtures/tokenizer.json",
        "../tests/vtm/fixtures/tokenizer.json",
        NULL
    };
    static long counter = 0;
    model_t md;
    uint32_t dE[2] = {STREAM_MAX_ID, STREAM_MAX_ID};
    uint32_t dlog[1] = {STREAM_MAX_ID};
    uint32_t dto[1] = {1};
    float logits[STREAM_MAX_ID];
    float tokv[1] = {-1.0f};
    float *e;
    uint8_t *img = NULL;
    size_t size = 0;
    char model_path[160], out_path[160], err_path[160], cmd[2048];
    char max_arg[64];
    const char *cli;
    const char *fixture;
    long stamp = (long)time(NULL);
    FILE *f;
    int i, rc;

    out[0] = '\0';
    err[0] = '\0';
    cli = find_readable(cli_paths);
    fixture = find_readable(tok_paths);
    if (!cli || !fixture) {
        snprintf(err, err_cap, "нет vtm или tokenizer.json");
        return -1;
    }

    memset(logits, 0, sizeof logits);
    e = (float *)xmalloc((size_t)STREAM_MAX_ID * STREAM_MAX_ID *
                         sizeof(float));
    memset(e, 0, (size_t)STREAM_MAX_ID * STREAM_MAX_ID * sizeof(float));
    {
        int from = STREAM_PROMPT_TOKEN;
        for (i = 0; i < n_chain; i++) {
            if (chain[i] < 0 || chain[i] >= STREAM_MAX_ID) break;
            // Строка from указывает на следующий токен цепочки.
            e[(size_t)from * STREAM_MAX_ID + (size_t)chain[i]] = 10.0f;
            from = chain[i];
        }
    }

    model_init(&md);
    model_add_f32(&md, "logits", 1, dlog, logits);
    model_add_f32(&md, "token", 1, dto, tokv);
    model_add_f32(&md, "E", 2, dE, e);
    free(e);
    model_inst(&md, VTM_OP_EMB, VTM_FLAG_EMB_INPUT, 0, 2,
               VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_SAMP, 0, 1, 0,
               VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_HALT, 0, VTM_NO_TENSOR, VTM_NO_TENSOR,
               VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    model_build(&md, &img, &size);

    counter++;
    snprintf(model_path, sizeof model_path, "/tmp/vtm_cli_stream_%ld_%ld.vtm",
             stamp, counter);
    snprintf(out_path, sizeof out_path, "/tmp/vtm_cli_stream_%ld_%ld.out",
             stamp, counter);
    snprintf(err_path, sizeof err_path, "/tmp/vtm_cli_stream_%ld_%ld.err",
             stamp, counter);

    f = fopen(model_path, "wb");
    if (!f) {
        free(img);
        model_free(&md);
        snprintf(err, err_cap, "не удалось записать модель");
        return -1;
    }
    fwrite(img, 1, size, f);
    fclose(f);

    if (max_new >= 0)
        snprintf(max_arg, sizeof max_arg, "--max-new %d", max_new);
    else
        max_arg[0] = '\0';
    snprintf(cmd, sizeof cmd,
             "%s run %s --text hello --tokenizer %s%s %s %s "
             "> %s 2> %s",
             cli, model_path, fixture, stream ? " --stream" : "", max_arg,
             extra ? extra : "", out_path, err_path);
    rc = system(cmd);
    read_text_file(out_path, out, out_cap);
    read_text_file(err_path, err, err_cap);

    remove(model_path);
    remove(out_path);
    remove(err_path);
    free(img);
    model_free(&md);
    return cli_exit_code(rc);
}

static int stream_cli_run(const int *chain, int n_chain, int max_new,
                          int stream, char *out, size_t out_cap,
                          char *err, size_t err_cap) {
    return chain_cli_run(chain, n_chain, max_new, stream, "", out, out_cap,
                         err, err_cap);
}

// Регрессия из фаззинга (0069): malformed-контейнер с imm=INT_MAX у ATTN давал
// signed overflow q_start+q_len, проверка "больше ctx" его пропускала, и malloc
// получал отрицательную длину — SIGBUS. Теперь q_start считается в long long.
static void test_attn_imm_overflow(void) {
    vtm_t *m = vtm_load("../../tests/vtm/fixtures/attn_imm_int_max.vtm");
    vtm_status st;

    CHECK(m != NULL);
    if (!m) return;
    vtm_set_token(m, 1);
    st = vtm_run(m, NULL, NULL);
    CHECK(st != VTM_OK);
    // Код обёртки зависит от слоя (файл/интерпретатор), поэтому проверяем, что
    // ошибка именно про ATTN, а не падение где-то ещё.
    CHECK(strstr(vtm_last_error(), "ATTN") != NULL);
    vtm_free(m);
}

// Трассировка (0082): один вызов на инструкцию, порядок и размеры сходятся,
// результат с хуком и без — побитово тот же, возврат 1 из колбэка останавливает.
typedef struct {
    int calls;
    uint64_t last_index;
    uint32_t last_pc;
    uint32_t ops_seen[64];  // опкоды 0x01..0x0F
    int bad_order;
    int bad_fields;
    int stop_after;
} trace_probe;

static int trace_cb(void *user, const vtm_trace_rec *rec) {
    trace_probe *p = (trace_probe *)user;

    if (rec == NULL || p == NULL) return 1;
    if (p->calls > 0 && rec->index <= p->last_index) p->bad_order = 1;
    if (rec->out_bytes != rec->out_elems * sizeof(float)) p->bad_fields = 1;
    if (rec->opcode > VTM_OP_HALT) p->bad_fields = 1;
    else p->ops_seen[rec->opcode]++;
    p->last_index = rec->index;
    p->last_pc = rec->pc;
    p->calls++;
    return p->stop_after > 0 && p->calls >= p->stop_after;
}

static void test_trace_hook(void) {
    trace_probe probe;
    vtm_t *m = mini_with_meta(NULL);
    uint32_t before = 0, after = 0;

    CHECK(m != NULL);
    if (!m) return;
    before = vtm_tensor_by_name(m, "y")->f32[0];

    memset(&probe, 0, sizeof probe);
    vtm_set_trace(m, trace_cb, &probe);
    CHECK(run_ok(m));
    after = vtm_tensor_by_name(m, "y")->f32[0];

    CHECK(probe.calls == (int)vtm_instr_count(m));
    CHECK(probe.bad_order == 0);
    CHECK(probe.bad_fields == 0);
    CHECK(probe.ops_seen[VTM_OP_HALT] == 1);
    // Трассировка не меняет результат: тот же байт, что и без хука.
    CHECK(before == after);
    vtm_free(m);

    // Колбэк, вернувший 1, останавливает прогон на первой же инструкции.
    m = mini_with_meta(NULL);
    CHECK(m != NULL);
    if (!m) return;
    memset(&probe, 0, sizeof probe);
    probe.stop_after = 1;
    vtm_set_trace(m, trace_cb, &probe);
    CHECK(run_ok(m));
    CHECK(probe.calls == 1);
    vtm_set_trace(m, NULL, NULL);
    CHECK(run_ok(m));
    vtm_free(m);
}

static void test_cli_stream(void) {
    char out[8192], err[8192];
    int rc;
    int seq_cyr[] = {208, 191};        /* 0xD0 0xBF — «п» */
    int seq_tail[] = {208, 501, 65};   /* обрыв на середине символа */
    int seq_bad[] = {208, 65};         /* 0xD0 + "A": битая пара */
    int seq_special[] = {65, 503, 66}; /* "A", <|im|>, "B" */
    int seq_stop[] = {65, 501, 66};    /* "A", <|im_end|>, "B" */

    // 1. Буква «п» разрезана между двумя токенами (0xD0 и 0xBF): стрим
    //    отдаёт ровно два байта UTF-8, без замен и мусора. Служебные строки
    //    (модель, токены, время) в stdout не попадают — только текст.
    rc = stream_cli_run(seq_cyr, 2, 2, 1, out, sizeof out, err, sizeof err);
    CHECK(rc == 0);
    CHECK(strcmp(out, "\xD0\xBF") == 0);
    CHECK(strstr(out, "модель:") == NULL);
    CHECK(strstr(out, "время:") == NULL);
    CHECK(strstr(err, "модель:") != NULL);
    CHECK(strstr(err, "стоп-токен") == NULL);

    // 2. Хвост не теряется: за 0xD0 сразу идёт стоп-токен. Недописанный
    //    символ отдаётся в конце как U+FFFD (валидный UTF-8), а не остаётся
    //    в буфере. Третий токен "A" не печатается — генерация оборвана.
    rc = stream_cli_run(seq_tail, 3, 3, 1, out, sizeof out, err, sizeof err);
    CHECK(rc == 0);
    CHECK(strcmp(out, "\xEF\xBF\xBD") == 0);
    CHECK(strstr(err, "стоп-токен: 501") != NULL);

    // 3. Битая пара 0xD0 + "A": сырой 0xD0 наружу не выходит, вместо него
    //    U+FFFD, дальше "A". stdout остаётся валидным UTF-8.
    rc = stream_cli_run(seq_bad, 2, 2, 1, out, sizeof out, err, sizeof err);
    CHECK(rc == 0);
    CHECK(strcmp(out, "\xEF\xBF\xBD" "A") == 0);

    // 4. Спецтокен <|im|> (не стоп) в текст не попадает, но генерация
    //    продолжается: следующий токен "B" напечатан.
    rc = stream_cli_run(seq_special, 3, 3, 1, out, sizeof out, err,
                        sizeof err);
    CHECK(rc == 0);
    CHECK(strcmp(out, "AB") == 0);
    CHECK(strstr(out, "<|im|>") == NULL);
    CHECK(strstr(err, "стоп-токен") == NULL);

    // 5. Стоп-токен <|im_end|> (id 501 из model.eos_token): не печатается,
    //    генерация обрывается — следующий токен "B" не появляется.
    rc = stream_cli_run(seq_stop, 3, 3, 1, out, sizeof out, err, sizeof err);
    CHECK(rc == 0);
    CHECK(strcmp(out, "A") == 0);
    CHECK(strstr(err, "стоп-токен: 501") != NULL);
    CHECK(strstr(err, "вывод:  65 501") != NULL);
    CHECK(strstr(out, "B") == NULL);

    // 6. Без --stream — прежнее поведение: служебное в stdout, «ответ:» с
    //    текстом, спецтокены по-прежнему часть ответа, стоп не обрывает.
    rc = stream_cli_run(seq_cyr, 2, 2, 0, out, sizeof out, err, sizeof err);
    CHECK(rc == 0);
    CHECK(strstr(out, "модель:") != NULL);
    CHECK(strstr(out, "prompt: 259") != NULL);
    CHECK(strstr(out, "ответ:   \xD0\xBF") != NULL);

    rc = stream_cli_run(seq_stop, 3, 3, 0, out, sizeof out, err, sizeof err);
    CHECK(rc == 0);
    CHECK(strstr(out, "ответ:   A<|im_end|>B") != NULL);

    // 7. --stream без --text — ошибка аргументов (код 2), а не молчаливый
    //    переход к старому режиму.
    {
        static const char *const cli_paths[] = {
            "./vtm", "src/vtm/vtm", "../../src/vtm/vtm", NULL
        };
        const char *cli = find_readable(cli_paths);
        char cmd[1024];
        long stamp = (long)time(NULL);

        CHECK(cli != NULL);
        if (cli) {
            snprintf(cmd, sizeof cmd,
                     "%s run /tmp/vtm_stream_missing_%ld.vtm --tokens 1 "
                     "--stream > /tmp/vtm_stream_%ld.out 2> /tmp/vtm_stream_%ld.err",
                     cli, stamp, stamp, stamp);
            rc = system(cmd);
            CHECK(cli_exit_code(rc) == 2);
            snprintf(cmd, sizeof cmd, "/tmp/vtm_stream_%ld.err", stamp);
            CHECK(read_text_file(cmd, err, sizeof err) == 0);
            CHECK(strstr(err, "--stream") != NULL);
            snprintf(cmd, sizeof cmd, "/tmp/vtm_stream_%ld.out", stamp);
            remove(cmd);
            snprintf(cmd, sizeof cmd, "/tmp/vtm_stream_%ld.err", stamp);
            remove(cmd);
        }
    }
}

// --- CLI --stop / --timeout / --stats (задача 0068) ------------------------

static void test_cli_limits(void) {
    char out[8192], err[8192];
    int rc;
    int seq_stop[] = {65, 10, 10, 66}; /* "A", "\n", "\n", "B" */
    int seq_a[] = {65};                /* "A" */
    int seq_loop[] = {65, 66, 65};     /* A B A B ... (цикл в таблице E) */

    // 1. --stop "\n\n" без --stream: генерация обрывается на втором переводе
    //    строки, стоп-строка не попадает в «ответ:», следующий токен "B" не
    //    генерируется.
    rc = chain_cli_run(seq_stop, 4, 4, 0, "--stop \"\\n\\n\"", out,
                       sizeof out, err, sizeof err);
    CHECK(rc == 0);
    CHECK(strstr(out, "ответ:   A\n") != NULL);
    CHECK(strstr(out, "ответ:   A\n\n") == NULL);
    CHECK(strstr(out, "вывод:  65 10 10\n") != NULL);
    CHECK(strstr(out, "вывод:  65 10 10 66") == NULL);

    // 2. То же со --stream: stdout — только текст до стопа; стоп-строка и
    //    следующий "B" не печатаются. Стоп приходит по частям (два токена
    //    "\n"), поэтому проверяется накопленный буфер.
    rc = chain_cli_run(seq_stop, 4, 4, 1, "--stop \"\\n\\n\"", out,
                       sizeof out, err, sizeof err);
    CHECK(rc == 0);
    CHECK(strcmp(out, "A") == 0);
    CHECK(strstr(err, "стоп: --stop") != NULL);
    CHECK(strstr(err, "вывод:  65 10 10\n") != NULL);
    CHECK(strstr(err, "вывод:  65 10 10 66") == NULL);

    // 3. --timeout 0.01 на зацикленной цепочке: код 124, уже сгенерированное
    //    напечатано (строка «вывод:» не теряется).
    rc = chain_cli_run(seq_loop, 3, 500000, 0, "--timeout 0.01", out,
                       sizeof out, err, sizeof err);
    CHECK(rc == 124);
    CHECK(strstr(out, "вывод:  65 66") != NULL);

    // 4. --timeout со --stream: код 124, поток текста не потерян.
    rc = chain_cli_run(seq_loop, 3, 500000, 1, "--timeout 0.01", out,
                       sizeof out, err, sizeof err);
    CHECK(rc == 124);
    CHECK(strncmp(out, "ABAB", 4) == 0);
    CHECK(strstr(err, "вывод:  65 66") != NULL);

    // 5. --stats: строка в stderr, stdout/ответ не меняются.
    rc = chain_cli_run(seq_a, 1, 1, 0, "--stats", out, sizeof out, err,
                       sizeof err);
    CHECK(rc == 0);
    CHECK(strstr(out, "ответ:   A\n") != NULL);
    CHECK(strstr(err, "stats:") != NULL);
    CHECK(strstr(err, "токенов=1") != NULL);
    CHECK(strstr(err, "токенов/с=") != NULL);

    // 6. --stats со --stream тоже пишет статистику в stderr, а не в stdout.
    rc = chain_cli_run(seq_a, 1, 1, 1, "--stats", out, sizeof out, err,
                       sizeof err);
    CHECK(rc == 0);
    CHECK(strcmp(out, "A") == 0);
    CHECK(strstr(err, "stats:") != NULL);
    CHECK(strstr(err, "токенов=1") != NULL);

    // 7. --max-tokens — синоним --max-new (в этой команде --max-new нет).
    rc = chain_cli_run(seq_a, 1, -1, 0, "--max-tokens 1", out, sizeof out,
                       err, sizeof err);
    CHECK(rc == 0);
    CHECK(strstr(out, "ответ:   A\n") != NULL);
    CHECK(strstr(out, "вывод:  65\n") != NULL);

    // 8. --stop, который так и не встретился: хвост не теряется ни в стриме,
    //    ни в обычном ответе.
    {
        int seq_ab[] = {65, 66}; /* "AB" */
        rc = chain_cli_run(seq_ab, 2, 2, 1, "--stop ZZ", out, sizeof out,
                           err, sizeof err);
        CHECK(rc == 0);
        CHECK(strcmp(out, "AB") == 0);
        rc = chain_cli_run(seq_ab, 2, 2, 0, "--stop ZZ", out, sizeof out,
                           err, sizeof err);
        CHECK(rc == 0);
        CHECK(strstr(out, "ответ:   AB\n") != NULL);
    }

    // 9. Стоп-строка из кириллицы (2 байта UTF-8): не режем символ посередине.
    //    Токены 208 и 191 дают вместе «п».
    {
        int seq_cyr[] = {208, 191, 65}; /* "п", "A" */
        rc = chain_cli_run(seq_cyr, 3, 3, 1, "--stop \"\xD0\xBF\"", out,
                           sizeof out, err, sizeof err);
        CHECK(rc == 0);
        CHECK(strcmp(out, "") == 0);
        CHECK(strstr(err, "вывод:  208 191\n") != NULL);
    }
}

// --- mmap (задача 0060) ----------------------------------------------------
//
// Внутренние хуки загрузчика (vtm_internal.h): нужны, чтобы отличить
// отображённый файл от фолбэка, не заглядывая в непрозрачную struct vtm.
int vtm_internal_file_is_mapped(const vtm_t *m);
int vtm_internal_tensor_is_mapped(const vtm_t *m, uint32_t index);

static int write_image(const char *path, const uint8_t *img, size_t size) {
    FILE *f = fopen(path, "wb");
    int ok;
    if (!f) return -1;
    ok = (size == 0 || fwrite(img, 1, size, f) == size);
    fclose(f);
    return ok ? 0 : -1;
}

// Ожидается ли отображение в этом запуске: VTM_NO_MMAP=1 (и Windows)
// переводят vtm_load() на чтение в память. Тесты ниже подстраиваются, чтобы
// `VTM_NO_MMAP=1 ./test_vtm` тоже был зелёным.
static int mmap_expected(void) {
#if defined(_WIN32)
    return 0;
#else
    const char *v = getenv("VTM_NO_MMAP");
    return !(v && *v && strcmp(v, "0") != 0);
#endif
}

// Модель для проверки mmap: тритовая матрица, in-place SCAL и KVW (пишет
// не только dst, но и b — V-кэш). Индексы тензоров:
//   0 W(trit) 1 x 2 y 3 act 4 kcache 5 vcache 6 k 7 v
static void mmap_model_init(model_t *md) {
    uint32_t dw[2] = {2, 2};
    uint32_t dv[1] = {2};
    uint32_t dcache[2] = {2, 2};
    int8_t wt[4] = {1, -1, 0, 1};
    float scales[1] = {2.0f};
    float xv[2] = {1.0f, 2.0f};
    float zero2[2] = {0.0f, 0.0f};
    float zero4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float av[2] = {5.0f, 7.0f};
    float kvv[2] = {1.5f, -2.5f};
    float vvv[2] = {3.0f, 4.5f};

    model_init(md);
    model_add_trit(md, "W", 2, dw, wt, scales, 1);
    model_add_f32(md, "x", 1, dv, xv);
    model_add_f32(md, "y", 1, dv, zero2);
    model_add_f32(md, "act", 1, dv, av);
    model_add_f32(md, "kcache", 2, dcache, zero4);
    model_add_f32(md, "vcache", 2, dcache, zero4);
    model_add_f32(md, "k", 1, dv, kvv);
    model_add_f32(md, "v", 1, dv, vvv);
    model_inst(md, VTM_OP_TMV, 0, 2, 0, 1, VTM_NO_TENSOR, 0);
    model_inst(md, VTM_OP_SCAL, 0, 3, 3, VTM_NO_TENSOR, VTM_NO_TENSOR, 2000);
    model_inst(md, VTM_OP_KVW, 0, 4, 6, 5, 7, 0);
    model_inst(md, VTM_OP_HALT, 0, VTM_NO_TENSOR, VTM_NO_TENSOR,
               VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
}

// Проверить результаты mmap_get_model: y, act и оба KV-кэша.
static void check_mmap_model(vtm_t *m) {
    const vtm_tensor *t;
    CHECK(run_ok(m));
    t = vtm_tensor_by_name(m, "y");
    CHECK(t && t->f32);
    if (t && t->f32) {
        CHECK_NEAR(t->f32[0], -2.0, 1e-6);
        CHECK_NEAR(t->f32[1], 4.0, 1e-6);
    }
    t = vtm_tensor_by_name(m, "act");
    CHECK(t && t->f32);
    if (t && t->f32) {
        CHECK_NEAR(t->f32[0], 10.0, 1e-6); // SCAL 2.0 на месте
        CHECK_NEAR(t->f32[1], 14.0, 1e-6);
    }
    t = vtm_tensor_by_name(m, "kcache");
    CHECK(t && t->f32);
    if (t && t->f32) {
        CHECK_NEAR(t->f32[0], 1.5, 1e-6);
        CHECK_NEAR(t->f32[1], -2.5, 1e-6);
    }
    t = vtm_tensor_by_name(m, "vcache");
    CHECK(t && t->f32);
    if (t && t->f32) {
        CHECK_NEAR(t->f32[0], 3.0, 1e-6);
        CHECK_NEAR(t->f32[1], 4.5, 1e-6);
    }
}

static void test_mmap_file_load(void) {
    model_t md;
    uint8_t *img;
    size_t size;
    char path[128];
    long stamp = (long)time(NULL);
    vtm_t *m;

    mmap_model_init(&md);
    model_build(&md, &img, &size);
    snprintf(path, sizeof path, "/tmp/vtm_mmap_%ld.vtm", stamp);
    CHECK(write_image(path, img, size) == 0);

    m = vtm_load(path);
    CHECK(m != NULL);
    if (m) {
        int map = mmap_expected();
        // Файл отображён (или, при VTM_NO_MMAP=1, прочитан в память);
        // fp32 и слова тритов при mmap ссылаются в отображение.
        CHECK(vtm_internal_file_is_mapped(m) == map);
        CHECK(vtm_internal_tensor_is_mapped(m, 0) == map); // W: триты
        CHECK(vtm_internal_tensor_is_mapped(m, 1) == map); // x: fp32-источник
        CHECK(vtm_internal_tensor_is_mapped(m, 2) == 0); // y: dst
        CHECK(vtm_internal_tensor_is_mapped(m, 3) == 0); // act: dst (in-place)
        CHECK(vtm_internal_tensor_is_mapped(m, 4) == 0); // kcache: dst KVW
        CHECK(vtm_internal_tensor_is_mapped(m, 5) == 0); // vcache: b KVW
        CHECK(vtm_internal_tensor_is_mapped(m, 6) == map); // k: только источник
        CHECK(vtm_internal_tensor_is_mapped(m, 7) == map); // v: только источник
        check_mmap_model(m);
        vtm_free(m);
    }

    remove(path);
    free(img);
    model_free(&md);
}

static void test_mmap_fallback(void) {
    model_t md;
    uint8_t *img;
    size_t size;
    char path[128];
    long stamp = (long)time(NULL);
    vtm_t *m;

    mmap_model_init(&md);
    model_build(&md, &img, &size);
    snprintf(path, sizeof path, "/tmp/vtm_mmap_fb_%ld.vtm", stamp);
    CHECK(write_image(path, img, size) == 0);

#if !defined(_WIN32)
    // Чести «mmap не было» добиваемся флагом, а не отсутствием платформы.
    // Прежнее значение сохраняем: `VTM_NO_MMAP=1 ./test_vtm` не должен
    // включать mmap для следующих тестов.
    {
        const char *old = getenv("VTM_NO_MMAP");
        char saved[64];
        int had = 0;
        if (old) {
            snprintf(saved, sizeof saved, "%s", old);
            had = 1;
        }
        CHECK(setenv("VTM_NO_MMAP", "1", 1) == 0);
        m = vtm_load(path);
        if (had) CHECK(setenv("VTM_NO_MMAP", saved, 1) == 0);
        else CHECK(unsetenv("VTM_NO_MMAP") == 0);
    }
#else
    m = vtm_load(path); // на Windows фолбэк всегда
#endif
    CHECK(m != NULL);
    if (m) {
        CHECK(vtm_internal_file_is_mapped(m) == 0);
        CHECK(vtm_internal_tensor_is_mapped(m, 0) == 0);
        check_mmap_model(m);
        vtm_free(m);
    }

    remove(path);
    free(img);
    model_free(&md);
}

// Обрезанный файл не должен ни падать, ни читать за концом отображения:
// загрузчик валидирует все смещения по фактическому размеру до доступа.
static void test_mmap_truncated(void) {
    model_t md;
    uint8_t *img;
    size_t size, cuts[8];
    size_t n_cuts = 0, k;
    char path[128];
    long stamp = (long)time(NULL);

    mmap_model_init(&md);
    model_build(&md, &img, &size);
    snprintf(path, sizeof path, "/tmp/vtm_mmap_tr_%ld.vtm", stamp);

    cuts[n_cuts++] = 0;
    cuts[n_cuts++] = 1;
    cuts[n_cuts++] = VTM_HEADER_SIZE - 1;
    cuts[n_cuts++] = VTM_HEADER_SIZE;
    cuts[n_cuts++] = size / 2;
    cuts[n_cuts++] = size - VTM_INSTR_SIZE;
    cuts[n_cuts++] = size - 1;

    for (k = 0; k < n_cuts; k++) {
        vtm_t *m;
        CHECK(write_image(path, img, cuts[k]) == 0);
        m = vtm_load(path);
        CHECK(m == NULL);
        CHECK(strlen(vtm_last_error()) > 0);
        if (m) vtm_free(m); // не должно случиться, но не течём
    }

    // Контроль: целый файл после всех обрезаний грузится и считает.
    CHECK(write_image(path, img, size) == 0);
    {
        vtm_t *m = vtm_load(path);
        CHECK(m != NULL);
        if (m) {
            check_mmap_model(m);
            vtm_free(m);
        }
    }

    remove(path);
    free(img);
    model_free(&md);
}

// fp16/bf16 интерпретатор читает как float*: их приходится разворачивать в
// fp32 в арене, в отличие от fp32-источников и тритов (те в отображении).
static void test_mmap_f16_bf16(void) {
    model_t md;
    uint32_t de[2] = {2, 2};
    uint32_t dy[1] = {2};
    float ev[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float fv[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float zv[2] = {0.0f, 0.0f};
    uint8_t *img;
    size_t size;
    char path[128];
    long stamp = (long)time(NULL);
    vtm_t *m;

    model_init(&md);
    model_add_f16(&md, "E", 2, de, ev);
    model_add_bf16(&md, "F", 2, de, fv);
    model_add_f32(&md, "y", 1, dy, zv);
    model_add_f32(&md, "z", 1, dy, zv);
    model_inst(&md, VTM_OP_EMB, 0, 2, 0, VTM_NO_TENSOR, VTM_NO_TENSOR, 1);
    model_inst(&md, VTM_OP_EMB, 0, 3, 1, VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_HALT, 0, VTM_NO_TENSOR, VTM_NO_TENSOR,
               VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    model_build(&md, &img, &size);
    snprintf(path, sizeof path, "/tmp/vtm_mmap_f16_%ld.vtm", stamp);
    CHECK(write_image(path, img, size) == 0);

    m = vtm_load(path);
    CHECK(m != NULL);
    if (m) {
        const vtm_tensor *t;
        CHECK(vtm_internal_file_is_mapped(m) == mmap_expected());
        CHECK(vtm_internal_tensor_is_mapped(m, 0) == 0); // fp16 скопирован
        CHECK(vtm_internal_tensor_is_mapped(m, 1) == 0); // bf16 скопирован
        CHECK(run_ok(m));
        t = vtm_tensor_by_name(m, "y");
        CHECK(t && t->f32);
        if (t && t->f32) {
            CHECK_NEAR(t->f32[0], 3.0, 1e-6);
            CHECK_NEAR(t->f32[1], 4.0, 1e-6);
        }
        t = vtm_tensor_by_name(m, "z");
        CHECK(t && t->f32);
        if (t && t->f32) {
            CHECK_NEAR(t->f32[0], 5.0, 1e-6);
            CHECK_NEAR(t->f32[1], 6.0, 1e-6);
        }
        vtm_free(m);
    }

    remove(path);
    free(img);
    model_free(&md);
}

// Тритовые слова в отображении, а fp16-масштабы конвертируются в арену:
// указателя в файл на float* не поставить, но это 1/128 данных тензора.
static void test_mmap_trit_f16_scales(void) {
    model_t md;
    uint32_t dw[2] = {2, 2};
    uint32_t dx[1] = {2};
    int8_t wt[4] = {1, 1, -1, 0};
    float scales[1] = {2.0f};
    float xv[2] = {1.0f, 2.0f};
    float zero2[2] = {0.0f, 0.0f};
    uint8_t *img;
    size_t size;
    char path[128];
    long stamp = (long)time(NULL);
    vtm_t *m;

    model_init(&md);
    model_add_trit_scaled(&md, "W", 2, dw, wt, scales, 0,
                          (int)VTM_SCALE_BLOCK_160);
    md.header_flags |= VTM_HEADER_FLAG_SCALE_BLOCK_160;
    model_add_f32(&md, "x", 1, dx, xv);
    model_add_f32(&md, "y", 1, dx, zero2);
    model_inst(&md, VTM_OP_TMV, 0, 2, 0, 1, VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_HALT, 0, VTM_NO_TENSOR, VTM_NO_TENSOR,
               VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    model_build(&md, &img, &size);
    snprintf(path, sizeof path, "/tmp/vtm_mmap_tf16_%ld.vtm", stamp);
    CHECK(write_image(path, img, size) == 0);

    m = vtm_load(path);
    CHECK(m != NULL);
    if (m) {
        const vtm_tensor *t;
        CHECK(vtm_internal_tensor_is_mapped(m, 0) == mmap_expected()); // слова
        CHECK(run_ok(m));
        t = vtm_tensor_by_name(m, "y");
        CHECK(t && t->f32);
        if (t && t->f32) {
            CHECK_NEAR(t->f32[0], 6.0, 1e-6);
            CHECK_NEAR(t->f32[1], -2.0, 1e-6);
        }
        vtm_free(m);
    }

    remove(path);
    free(img);
    model_free(&md);
}

// --- SAMP: буфер кандидатов и частичный отбор (задача 0059) ----------------
//
// Проверяем то, что задача меняет:
//   1) буфер кандидатов выделяется один раз на машину и переиспользуется —
//      на токен выделений нет (норма — 1 на жизнь машины);
//   2) новый отбор (quickselect для top-k, LSD-radix для полного порядка)
//      даёт те же токены, что прежняя полная сортировка qsort: эталон ниже
//      повторяет код до 0059 дословно, включая порядок накопления суммы;
//   3) на словаре Qwen 150 000 время на токен меряется рядом: прежний путь
//      (malloc + qsort) против нового (буфер + частичный отбор).

// Внутренние тест-хуки буфера (vtm_samp_alloc_count и т.п.). Заголовок
// включает vtm.h сам, повторный include безопасен.
#include "../../src/vtm/vtm_internal.h"

// Эталон: копия samp_sample() до задачи 0059 — malloc на вызов, qsort по
// всему словарю, штраф линейным сканом окна. u передаётся снаружи, чтобы
// тест мог вести ГПСЧ в том же темпе, что машина.
typedef struct {
    double value;
    uint32_t index;
} ref_cand;

static int ref_cand_cmp(const void *pa, const void *pb) {
    const ref_cand *a = (const ref_cand *)pa;
    const ref_cand *b = (const ref_cand *)pb;
    if (a->value > b->value) return -1;
    if (a->value < b->value) return 1;
    if (a->index < b->index) return -1;
    if (a->index > b->index) return 1;
    return 0;
}

// hist/hist_len — уже сгенерированные токены по порядку; окно штрафа — как
// у кольца samp_hist: последние window элементов.
// *used_draw = 0, если прежний код не потреблял случайное число (максимум
// неконечен) — тогда и машина его не потребляет.
static int ref_samp_choose(const float *logits, uint32_t n, int strategy,
                           double temperature, int top_k, double top_p,
                           double penalty, int window,
                           const int *hist, int hist_len, double u,
                           int *used_draw) {
    ref_cand *cand = (ref_cand *)xmalloc((size_t)n * (sizeof *cand
                                                       + sizeof(double)));
    double *w = (double *)(void *)(cand + n);
    uint32_t j, kept = n;
    double top, total = 0.0, upto, acc;
    int token, pen = penalty != 1.0 && window > 0 && hist_len > 0;

    for (j = 0; j < n; j++) {
        double value = (double)logits[j];
        if (isnan(value)) value = -INFINITY;
        if (pen) {
            int k, lim = window < hist_len ? window : hist_len, hit = 0;
            for (k = 0; k < lim; k++) {
                if (hist[hist_len - 1 - k] == (int)j) { hit = 1; break; }
            }
            if (hit) {
                value = value > 0.0 ? value / penalty : value * penalty;
            }
        }
        cand[j].value = value;
        cand[j].index = j;
    }
    qsort(cand, (size_t)n, sizeof *cand, ref_cand_cmp);
    token = (int)cand[0].index;
    top = cand[0].value;
    *used_draw = 0;
    if (!isfinite(top)) {
        free(cand);
        return token;
    }
    if (strategy == VTM_SAMP_TOPK && top_k > 0 && (uint32_t)top_k < kept)
        kept = (uint32_t)top_k;
    for (j = 0; j < kept; j++) {
        w[j] = exp((cand[j].value - top) / temperature);
        total += w[j];
    }
    if (strategy == VTM_SAMP_TOPP && top_p < 1.0 && total > 0.0) {
        double need = top_p * total;
        upto = 0.0;
        for (j = 0; j < kept; j++) {
            upto += w[j];
            if (upto >= need) { kept = j + 1; break; }
        }
        total = 0.0;
        for (j = 0; j < kept; j++) total += w[j];
    }
    if (total > 0.0 && isfinite(total)) {
        double threshold = u * total;
        *used_draw = 1;
        acc = 0.0;
        for (j = 0; j < kept; j++) {
            acc += w[j];
            if (acc >= threshold) { token = (int)cand[j].index; break; }
        }
        if (j >= kept) token = (int)cand[kept - 1].index;
    }
    free(cand);
    return token;
}

// ГПСЧ SAMP из docs/vtm.md — побитово тот же, что в машине.
static uint64_t ref_samp_mix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

static double ref_samp_uniform(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return (double)((x * 0x2545F4914F6CDD1Dull) >> 11)
         * (1.0 / 9007199254740992.0);
}

// Детерминированные логиты для проверок: пила из формулы, поверх неё
// группы равных значений (граница top-k/top-p часто попадает внутрь них —
// именно там расходятся «стабильные» и «нестабильные» сортировки).
static void samp_make_logits(float *v, uint32_t n) {
    uint32_t i;
    for (i = 0; i < n; i++) {
        double x = (double)((i * 2654435761u) % 2000u) / 1000.0 - 1.0;
        if (i % 5 == 0) x = 1.5;      // большая группа равных
        else if (i % 11 == 0) x = -0.25;
        v[i] = (float)x;
    }
}

// Прогнать машину и эталон в одном темпе: u_i из одного ГПСЧ, история
// повторов ведётся тестом (она же уходит в эталон).
static void samp_ref_case(const float *logits, uint32_t n, int strategy,
                          double temperature, int top_k, double top_p,
                          double penalty, int window, uint64_t seed,
                          int steps) {
    char meta[256];
    int hist[512], n_hist = 0, i;
    vtm_t *m;
    uint64_t rng;

    snprintf(meta, sizeof meta,
             "{\"sample_seed\": %lu, \"temperature\": %.17g, \"top_k\": %d,"
             " \"top_p\": %.17g, \"repetition_penalty\": %.17g,"
             " \"repetition_window\": %d}",
             (unsigned long)seed, temperature, top_k, top_p, penalty, window);
    m = samp_container(logits, n, meta, strategy);
    CHECK(m != NULL);
    if (!m) return;
    rng = ref_samp_mix64(seed ^ 0x9E3779B97F4A7C15ull);
    for (i = 0; i < steps; i++) {
        int used = 0, want, got;
        double u = ref_samp_uniform(&rng);
        want = ref_samp_choose(logits, n, strategy, temperature, top_k, top_p,
                               penalty, window, hist, n_hist, u, &used);
        got = samp_next(m);
        CHECK(used == 1);
        if (got != want) {
            fprintf(stderr, "FAIL samp_ref: стратегия %d, шаг %d: %d != %d "
                    "(seed %lu)\n", strategy, i, got, want,
                    (unsigned long)seed);
            g_fail++;
            break;
        }
        CHECK(n_hist < (int)(sizeof hist / sizeof hist[0]));
        if (n_hist < (int)(sizeof hist / sizeof hist[0]))
            hist[n_hist++] = got;
    }
    vtm_samp_free(m);   // в vtm_free() буфер ещё не освобождается (loader.c)
    vtm_free(m);
}

static void test_samp_partial_select(void) {
    uint32_t n = 257, big = 5000, i;
    float *small = (float *)xmalloc((size_t)n * sizeof *small);
    float *wide = (float *)xmalloc((size_t)big * sizeof *wide);

    samp_make_logits(small, n);
    samp_make_logits(wide, big);
    // Маленький словарь (radix на 11 битах) и большой (16 бит): новый отбор
    // обязан совпасть с полной сортировкой в обоих режимах.
    samp_ref_case(small, n, VTM_SAMP_TEMPERATURE, 0.7, 0, 1.0, 1.0, 64,
                  2024, 64);
    samp_ref_case(small, n, VTM_SAMP_TOPK, 0.7, 3, 1.0, 1.0, 64, 1, 64);
    samp_ref_case(small, n, VTM_SAMP_TOPK, 1.3, 17, 1.0, 1.0, 64, 7, 64);
    samp_ref_case(small, n, VTM_SAMP_TOPK, 0.5, 256, 1.0, 1.0, 64, 2024, 64);
    samp_ref_case(small, n, VTM_SAMP_TOPK, 1.0, 300, 1.0, 1.0, 64, 5, 32);
    samp_ref_case(small, n, VTM_SAMP_TOPP, 1.0, 0, 0.5, 1.0, 64, 3, 64);
    samp_ref_case(small, n, VTM_SAMP_TOPP, 0.8, 0, 0.9, 1.0, 64, 11, 64);
    samp_ref_case(small, n, VTM_SAMP_TOPP, 0.8, 0, 1.0, 1.0, 64, 13, 64);
    // Штраф за повторы: метки окна должны работать как линейный скан эталона.
    samp_ref_case(small, n, VTM_SAMP_TEMPERATURE, 0.9, 0, 1.0, 1.7, 8,
                  99, 64);
    samp_ref_case(small, n, VTM_SAMP_TOPK, 0.9, 5, 1.0, 4.0, 2, 99, 64);
    samp_ref_case(small, n, VTM_SAMP_TOPP, 0.6, 0, 0.85, 2.5, 32, 123456, 64);
    // Большой словарь: top-k с отбором префикса и полный порядок для top-p.
    samp_ref_case(wide, big, VTM_SAMP_TOPK, 0.8, 40, 1.0, 1.0, 64, 42, 24);
    samp_ref_case(wide, big, VTM_SAMP_TOPK, 0.8, 1000, 1.0, 1.0, 64, 42, 16);
    samp_ref_case(wide, big, VTM_SAMP_TOPP, 0.8, 0, 0.95, 1.0, 64, 42, 16);
    samp_ref_case(wide, big, VTM_SAMP_TEMPERATURE, 1.1, 0, 1.0, 1.5, 16,
                  77, 16);
    // Края: все логиты равны, один лидер, отрицательные значения.
    for (i = 0; i < big; i++) wide[i] = 0.0f;
    samp_ref_case(wide, big, VTM_SAMP_TOPP, 1.0, 0, 0.75, 1.0, 64, 8, 16);
    wide[big / 2] = 9.0f;
    samp_ref_case(wide, big, VTM_SAMP_TOPK, 1.0, 100, 1.0, 1.0, 64, 8, 16);
    for (i = 0; i < big; i++) wide[i] = -2.0f - (float)(i % 7);
    samp_ref_case(wide, big, VTM_SAMP_TOPP, 0.9, 0, 0.6, 1.0, 64, 8, 16);

    free(small);
    free(wide);
}

static void test_samp_buffer_reuse(void) {
    uint32_t n = 5000, i;
    float *logits = (float *)xmalloc((size_t)n * sizeof *logits);
    vtm_t *m;
    size_t bytes;

    samp_make_logits(logits, n);

    // (1) До первого SAMP выделений нет, после первого — ровно одно.
    m = samp_container(logits, n,
                       "{\"temperature\": 0.9, \"top_k\": 40,"
                       " \"sample_seed\": 42}", VTM_SAMP_TOPK);
    CHECK(m != NULL);
    if (m) {
        CHECK(vtm_samp_alloc_count(m) == 0);
        CHECK(vtm_samp_buffer_bytes(m) == 0);
        (void)samp_next(m);
        bytes = vtm_samp_buffer_bytes(m);
        CHECK(vtm_samp_alloc_count(m) == 1);
        // кандидаты (2×16) + метки (4) на токен + счётчики radix для
        // большого словаря
        CHECK(bytes >= (size_t)n * (2u * 16u + 4u) + 65536u * 4u);
        for (i = 0; i < 100; i++) (void)samp_next(m);
        CHECK(vtm_samp_alloc_count(m) == 1);        // malloc на токен нет
        CHECK(vtm_samp_buffer_bytes(m) == bytes);   // и блок не перевыделялся
        printf("  SAMP буфер: %lu байт, выделений %lu на 101 токен (n=%u)\n",
               (unsigned long)bytes,
               (unsigned long)vtm_samp_alloc_count(m), (unsigned)n);
        vtm_samp_free(m);                           // см. отчёт: vtm_free в loader.c
        CHECK(vtm_samp_buffer_bytes(m) == 0);
        CHECK(vtm_samp_alloc_count(m) == 1);
        vtm_free(m);
    }

    // (2) Чистый argmax буфера не требует вовсе.
    m = samp_container(logits, n, NULL, VTM_SAMP_ARGMAX);
    CHECK(m != NULL);
    if (m) {
        for (i = 0; i < 10; i++) (void)samp_next(m);
        CHECK(vtm_samp_alloc_count(m) == 0);
        CHECK(vtm_samp_buffer_bytes(m) == 0);
        vtm_free(m);
    }

    // (3) argmax со штрафом: буфер нужен под метки окна, но тоже один раз.
    m = samp_container(logits, n,
                       "{\"repetition_penalty\": 3.0,"
                       " \"repetition_window\": 8}", VTM_SAMP_ARGMAX);
    CHECK(m != NULL);
    if (m) {
        for (i = 0; i < 10; i++) (void)samp_next(m);
        CHECK(vtm_samp_alloc_count(m) == 1);
        vtm_samp_free(m);
        vtm_free(m);
    }

    free(logits);
}

// Переиспользованная память не должна штрафовать «чужие» токены.
// vtm_free() вызывает vtm_samp_free(), и следующий malloc может отдать тот же
// блок с метками прошлой машины (на glibc это штатное поведение). Тест-хук
// vtm_samp_poison_stamps() воспроизводит это состояние на любой платформе:
// метки заполнены значением, которое выберет следующее поколение, а счётчик
// поколений обнулён — ровно как у свежей машины на грязном блоке.
static void test_samp_stale_stamps(void) {
    float logits[8] = {0.4f, 0.0f, 1.1f, 0.0f, 0.2f, 0.1f, 0.3f, 0.05f};
    const char *meta = "{\"temperature\": 1.0, \"top_p\": 0.95,"
                       " \"repetition_penalty\": 2.0,"
                       " \"repetition_window\": 4,"
                       " \"sample_seed\": 123456}";
    int clean[24], dirty[24];
    vtm_t *a = samp_container(logits, 8, meta, VTM_SAMP_TOPP);
    vtm_t *b = samp_container(logits, 8, meta, VTM_SAMP_TOPP);

    CHECK(a != NULL && b != NULL);
    if (a && b) {
        // Один шаг: буферы уже выделены, история и состояние ГПСЧ одинаковы.
        (void)samp_next(a);
        (void)samp_next(b);
        // Блоку b «достались» метки поколения 1 от прошлой машины: именно
        // это значение выберет следующий fill, если метки не обнулить, —
        // тогда штраф получили бы все токены словаря, а не окно.
        vtm_samp_poison_stamps(b, 1);
        samp_seq(a, clean, 24);
        samp_seq(b, dirty, 24);
        CHECK(memcmp(clean, dirty, sizeof clean) == 0);
        CHECK(vtm_samp_buffer_bytes(b) > 0);
    }
    vtm_free(a);
    vtm_free(b);

    // Тот же кейс, что ловил падение на glibc: новая машина после free
    // обязана повторять чистую последовательность (метки не переезжают).
    {
        float wide[64];
        char meta2[192];
        int first[16], second[16];
        vtm_t *m1, *m2;
        int i;
        for (i = 0; i < 64; i++) wide[i] = (float)((i * 37) % 29) / 7.0f - 2.0f;
        for (i = 0; i < 4; i++) {
            snprintf(meta2, sizeof meta2,
                     "{\"temperature\": 0.8, \"top_p\": 0.9,"
                     " \"repetition_penalty\": 1.7,"
                     " \"repetition_window\": 8, \"sample_seed\": %d}",
                     100 + i);
            m1 = samp_container(wide, 64, meta2, VTM_SAMP_TOPP);
            CHECK(m1 != NULL);
            if (!m1) continue;
            samp_seq(m1, first, 16);
            vtm_free(m1);                       // освобождает буфер с метками
            m2 = samp_container(wide, 64, meta2, VTM_SAMP_TOPP);
            CHECK(m2 != NULL);
            if (!m2) continue;
            samp_seq(m2, second, 16);
            CHECK(memcmp(first, second, sizeof first) == 0);
            vtm_free(m2);
        }
    }
}

static double bench_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// Замер на словаре Qwen: 150 000 логитов. Число прогонов можно поднять
// через VTM_SAMP_BENCH (по умолчанию 12 — чтобы make test не тормозил).
static void test_samp_bench_150k(void) {
    const uint32_t n = 150000;
    const char *env = getenv("VTM_SAMP_BENCH");
    int iters = env ? atoi(env) : 12;
    float *logits = (float *)xmalloc((size_t)n * sizeof *logits);
    uint32_t i;
    int pass;

    if (iters < 1) iters = 1;
    if (iters > 1000) iters = 1000;
    for (i = 0; i < n; i++) {
        // Логиты «как у модели»: основная масса около нуля, заметный лидер.
        double x = (double)((i * 2654435761u) % 10000u) / 2000.0 - 2.5;
        logits[i] = (float)x;
    }
    logits[n / 3] = 18.0f;

    for (pass = 0; pass < 2; pass++) {
        int strategy = pass == 0 ? VTM_SAMP_TOPK : VTM_SAMP_TEMPERATURE;
        const char *name = pass == 0 ? "top-k 40" : "temperature";
        char meta[160];
        vtm_t *m;
        uint64_t rng;
        int hist[512];
        int n_hist = 0, it, mismatches = 0;
        double t0, old_ms, new_ms;

        snprintf(meta, sizeof meta,
                 "{\"temperature\": 0.8, \"top_k\": 40,"
                 " \"sample_seed\": 7}");
        m = samp_container(logits, n, meta, strategy);
        CHECK(m != NULL);
        if (!m) continue;
        rng = ref_samp_mix64(7 ^ 0x9E3779B97F4A7C15ull);

        (void)samp_next(m);          // первый SAMP выделяет буфер: не в замер
        t0 = bench_now();
        for (it = 0; it < iters; it++) (void)samp_next(m);
        new_ms = (bench_now() - t0) * 1000.0 / (double)iters;

        // Прежний путь на тех же данных: malloc + qsort + обход словаря.
        t0 = bench_now();
        for (it = 0; it < iters; it++) {
            int used = 0, token;
            double u = ref_samp_uniform(&rng);
            token = ref_samp_choose(logits, n, strategy, 0.8, 40, 1.0,
                                    1.0, 64, hist, n_hist, u, &used);
            if (n_hist < (int)(sizeof hist / sizeof hist[0]))
                hist[n_hist++] = token;
        }
        old_ms = (bench_now() - t0) * 1000.0 / (double)iters;

        // И заодно паритет на широком словаре: те же токены, что у эталона.
        n_hist = 0;
        {
            vtm_t *m2 = samp_container(logits, n, meta, strategy);
            uint64_t rng2 = ref_samp_mix64(7 ^ 0x9E3779B97F4A7C15ull);
            CHECK(m2 != NULL);
            if (m2) {
                for (it = 0; it < iters; it++) {
                    int used = 0, want, got;
                    double u = ref_samp_uniform(&rng2);
                    want = ref_samp_choose(logits, n, strategy, 0.8, 40, 1.0,
                                           1.0, 64, hist, n_hist, u, &used);
                    got = samp_next(m2);
                    if (got != want) mismatches++;
                    if (n_hist < (int)(sizeof hist / sizeof hist[0]))
                        hist[n_hist++] = got;
                }
                vtm_samp_free(m2);
                vtm_free(m2);
            }
        }
        CHECK(mismatches == 0);
        printf("  SAMP 150k %-12s старое %7.3f мс/токен, новое %7.3f мс/токен"
               ", ускорение %5.1fx\n", name, old_ms, new_ms,
               new_ms > 0.0 ? old_ms / new_ms : 0.0);
        vtm_samp_free(m);
        vtm_free(m);
    }
    free(logits);
}

// RAW-раскладка (слова и масштабы в разных местах) тоже отображается без
// копирования: offset и scale_offset кратны 32, поэтому указатели выровнены.
static void test_mmap_raw_trit(void) {
    model_t md;
    uint32_t dw[2] = {1, 3};
    uint32_t dx[1] = {3};
    uint32_t dy[1] = {1};
    int8_t wt[3] = {1, -1, 0};
    float scales[1] = {3.0f};
    float xv[3] = {2.0f, 5.0f, 7.0f};
    float zero1[1] = {0.0f};
    uint8_t *img;
    size_t size;
    char path[128];
    long stamp = (long)time(NULL);
    vtm_t *m;

    model_init(&md);
    md.header_flags = VTM_HEADER_FLAG_F32_SCALES;
    model_add_raw_trit(&md, "W", 2, dw, wt, scales);
    model_add_f32(&md, "x", 1, dx, xv);
    model_add_f32(&md, "y", 1, dy, zero1);
    model_inst(&md, VTM_OP_TMV, 0, 2, 0, 1, VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_HALT, 0, VTM_NO_TENSOR, VTM_NO_TENSOR,
               VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    model_build(&md, &img, &size);
    snprintf(path, sizeof path, "/tmp/vtm_mmap_raw_%ld.vtm", stamp);
    CHECK(write_image(path, img, size) == 0);

    m = vtm_load(path);
    CHECK(m != NULL);
    if (m) {
        const vtm_tensor *w = vtm_tensor_by_name(m, "W");
        CHECK(w != NULL && w->layout == VTM_LAYOUT_RAW);
        CHECK(vtm_internal_tensor_is_mapped(m, 0) == mmap_expected());
        CHECK(run_ok(m));
        CHECK_NEAR(vtm_tensor_by_name(m, "y")->f32[0],
                   (1.0 * 2.0 - 1.0 * 5.0) * 3.0, 1e-6);
        vtm_free(m);
    }

    remove(path);
    free(img);
    model_free(&md);
}

// --- кэш префикса (задача 0081) --------------------------------------------
//
// Модель — один Qwen-подобный слой (как demo/model.vtm): EMB → RMSN → q/k/v →
// ROPE → KVW → ATTN → o_proj → MLP → lm_head → SAMP. Размеры задаются
// параметрами, чтобы тот же код собирал маленькую модель для теста и
// 128-скрытую для замера.
//
// Проверяется поведение, а не реализация: после vtm_run_cached() машина
// обязана быть ровно в том же состоянии, что после полного прогона тех же ids
// по одному токену, — побитово (логиты, выбранный токен, K/V, позиция,
// счётчик сгенерированных). Счётчики попаданий/промахов показывают, был ли
// прогон полным; K/V сравниваются по строкам, которые прогон трогал (хвост
// прошлого, более длинного запроса за пределами len не читается и не важен).

enum {
    CT_EMB, CT_X, CT_H1, CT_N1, CT_Q, CT_K, CT_V, CT_QW, CT_KW, CT_VW,
    CT_KC, CT_VC, CT_ATTN, CT_O, CT_OW, CT_H2, CT_N2, CT_GATE, CT_UP,
    CT_GW, CT_UW, CT_MUL, CT_DOWN, CT_DW, CT_NF, CT_HF, CT_LOGITS,
    CT_LMW, CT_TOKEN
};

typedef struct {
    int ctx;        // длина KV-кэша
    int hidden;     // hidden_size
    int vocab;      // словарь (эмбеддинги и lm_head)
    int inter;      // промежуточный размер MLP
    int heads;      // число голов Q
    int head_dim;   // размер головы (heads*head_dim == hidden)
    int kv_heads;   // GQA: kv_dim = kv_heads*head_dim
} cache_cfg;

static void cache_add_f32(model_t *md, const char *name, int ndim,
                          const uint32_t *dims, float base, float amp) {
    uint32_t n = dims_product(ndim, dims), i;
    float *v = (float *)xmalloc((size_t)(n ? n : 1) * sizeof *v);
    for (i = 0; i < n; i++) {
        int d = (int)(((i * 2654435761u) >> 19) % 21u) - 10;
        v[i] = base + amp * (float)d / 10.0f;
    }
    model_add_f32(md, name, ndim, dims, v);
    free(v);
}

// Тритовая матрица [rows, cols] с блоком масштаба 128 (канон Python-эталона).
static void cache_add_trit(model_t *md, const char *name, int rows, int cols) {
    uint32_t dims[2] = {(uint32_t)rows, (uint32_t)cols};
    uint32_t n = (uint32_t)rows * (uint32_t)cols;
    uint32_t n_scales = (n + VTM_SCALE_BLOCK_128 - 1u) / VTM_SCALE_BLOCK_128;
    int8_t *trits = (int8_t *)xmalloc((size_t)(n ? n : 1));
    float *scales = (float *)xmalloc((size_t)(n_scales ? n_scales : 1)
                                     * sizeof *scales);
    uint32_t i;

    for (i = 0; i < n; i++)
        trits[i] = (int8_t)(int)(((i * 2654435761u) >> 9) % 3u) - 1;
    for (i = 0; i < n_scales; i++) scales[i] = 0.05f;
    model_add_trit_scaled(md, name, 2, dims, trits, scales, 1,
                          (int)VTM_SCALE_BLOCK_128);
    free(trits);
    free(scales);
}

static vtm_t *cache_model(const cache_cfg *cfg) {
    model_t md;
    uint32_t d_emb[2], d_vec[1], d_q[3], d_k[3], d_c[2], d_g[1];
    uint32_t d_log[2], d_one[1];
    int kv_dim = cfg->kv_heads * cfg->head_dim;
    uint8_t *img;
    size_t size;
    char meta[192];
    vtm_t *m;

    d_emb[0] = (uint32_t)cfg->vocab;
    d_emb[1] = (uint32_t)cfg->hidden;
    d_vec[0] = (uint32_t)cfg->hidden;
    d_q[0] = (uint32_t)cfg->heads;
    d_q[1] = 1;
    d_q[2] = (uint32_t)cfg->head_dim;
    d_k[0] = (uint32_t)cfg->kv_heads;
    d_k[1] = 1;
    d_k[2] = (uint32_t)cfg->head_dim;
    d_c[0] = (uint32_t)cfg->ctx;
    d_c[1] = (uint32_t)kv_dim;
    d_g[0] = (uint32_t)cfg->inter;
    d_log[0] = (uint32_t)cfg->vocab;
    d_log[1] = 1;
    d_one[0] = 1;

    model_init(&md);
    cache_add_f32(&md, "emb.w", 2, d_emb, 0.0f, 0.05f);
    cache_add_f32(&md, "x", 1, d_vec, 0.0f, 0.0f);
    cache_add_f32(&md, "h1", 1, d_vec, 0.0f, 0.0f);
    cache_add_f32(&md, "norm1.w", 1, d_vec, 1.0f, 0.02f);
    cache_add_f32(&md, "q", 3, d_q, 0.0f, 0.0f);
    cache_add_f32(&md, "k", 3, d_k, 0.0f, 0.0f);
    cache_add_f32(&md, "v", 3, d_k, 0.0f, 0.0f);
    cache_add_trit(&md, "q_proj.w", cfg->hidden, cfg->hidden);
    cache_add_trit(&md, "k_proj.w", kv_dim, cfg->hidden);
    cache_add_trit(&md, "v_proj.w", kv_dim, cfg->hidden);
    cache_add_f32(&md, "kcache", 2, d_c, 0.0f, 0.0f);
    cache_add_f32(&md, "vcache", 2, d_c, 0.0f, 0.0f);
    cache_add_f32(&md, "attn", 3, d_q, 0.0f, 0.0f);
    cache_add_f32(&md, "o", 1, d_vec, 0.0f, 0.0f);
    cache_add_trit(&md, "o_proj.w", cfg->hidden, cfg->hidden);
    cache_add_f32(&md, "h2", 1, d_vec, 0.0f, 0.0f);
    cache_add_f32(&md, "norm2.w", 1, d_vec, 1.0f, 0.02f);
    cache_add_f32(&md, "gate", 1, d_g, 0.0f, 0.0f);
    cache_add_f32(&md, "up", 1, d_g, 0.0f, 0.0f);
    cache_add_trit(&md, "gate_proj.w", cfg->inter, cfg->hidden);
    cache_add_trit(&md, "up_proj.w", cfg->inter, cfg->hidden);
    cache_add_f32(&md, "mul", 1, d_g, 0.0f, 0.0f);
    cache_add_f32(&md, "down", 1, d_vec, 0.0f, 0.0f);
    cache_add_trit(&md, "down_proj.w", cfg->hidden, cfg->inter);
    cache_add_f32(&md, "final_norm.w", 1, d_vec, 1.0f, 0.02f);
    cache_add_f32(&md, "hfinal", 1, d_vec, 0.0f, 0.0f);
    cache_add_f32(&md, "logits", 2, d_log, 0.0f, 0.0f);
    cache_add_trit(&md, "lm_head.w", cfg->vocab, cfg->hidden);
    cache_add_f32(&md, "token", 1, d_one, 0.0f, 0.0f);

    model_inst(&md, VTM_OP_EMB, VTM_FLAG_EMB_INPUT, CT_X, CT_EMB,
               VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_RMSN, 0, CT_H1, CT_X, CT_N1,
               VTM_NO_TENSOR, 1);
    model_inst(&md, VTM_OP_TMV, 0, CT_Q, CT_QW, CT_H1, VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_TMV, 0, CT_K, CT_KW, CT_H1, VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_TMV, 0, CT_V, CT_VW, CT_H1, VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_ROPE, 0, CT_Q, CT_Q, (uint16_t)cfg->heads,
               VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_ROPE, 0, CT_K, CT_K, (uint16_t)cfg->kv_heads,
               VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_KVW, 0, CT_KC, CT_K, CT_VC, CT_V, 0);
    model_inst(&md, VTM_OP_ATTN, VTM_FLAG_CAUSAL, CT_ATTN, CT_Q, CT_KC,
               CT_VC, 0);
    model_inst(&md, VTM_OP_TMV, 0, CT_O, CT_OW, CT_ATTN, VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_ADD, 0, CT_X, CT_X, CT_O, VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_RMSN, 0, CT_H2, CT_X, CT_N2, VTM_NO_TENSOR, 1);
    model_inst(&md, VTM_OP_TMV, 0, CT_GATE, CT_GW, CT_H2, VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_TMV, 0, CT_UP, CT_UW, CT_H2, VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_SILU, 0, CT_GATE, CT_GATE, VTM_NO_TENSOR,
               VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_MUL, 0, CT_MUL, CT_GATE, CT_UP, VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_TMV, 0, CT_DOWN, CT_DW, CT_MUL, VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_ADD, 0, CT_X, CT_X, CT_DOWN, VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_RMSN, 0, CT_HF, CT_X, CT_NF, VTM_NO_TENSOR, 1);
    model_inst(&md, VTM_OP_TMV, 0, CT_LOGITS, CT_LMW, CT_HF,
               VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_SAMP, 0, CT_TOKEN, CT_LOGITS, VTM_NO_TENSOR,
               VTM_NO_TENSOR, VTM_SAMP_TEMPERATURE);
    model_inst(&md, VTM_OP_HALT, 0, VTM_NO_TENSOR, VTM_NO_TENSOR,
               VTM_NO_TENSOR, VTM_NO_TENSOR, 0);

    // Недетерминированный argmax тут не годится: SAMP с температурой тянет
    // ГПСЧ, и совпадение токена проверяет ещё и восстановление состояния
    // сэмплера при попадании.
    snprintf(meta, sizeof meta,
             "{\"temperature\": 0.8, \"sample_seed\": 7,"
             " \"rms_norm_eps\": 1e-06}");
    model_set_meta(&md, meta);
    model_build(&md, &img, &size);
    m = load_ok(img, size);
    free(img);
    model_free(&md);
    return m;
}

#define CACHE_SNAP_VOCAB 32
#define CACHE_SNAP_CTX   64
#define CACHE_SNAP_KV    32

typedef struct {
    float logits[CACHE_SNAP_VOCAB];
    float token;
    float kc[CACHE_SNAP_CTX * CACHE_SNAP_KV];
    float vc[CACHE_SNAP_CTX * CACHE_SNAP_KV];
    int   kvdim;
    int   rows;
    int   position;
    int   generated;
} cache_snap;

static void cache_snap_take(vtm_t *m, cache_snap *s, int rows) {
    const vtm_tensor *lg = vtm_tensor_by_name(m, "logits");
    const vtm_tensor *tk = vtm_tensor_by_name(m, "token");
    const vtm_tensor *kc = vtm_tensor_by_name(m, "kcache");
    const vtm_tensor *vc = vtm_tensor_by_name(m, "vcache");
    uint32_t n_logits, n_kv;

    memset(s, 0, sizeof *s);
    CHECK(lg && tk && kc && vc);
    if (!lg || !tk || !kc || !vc) return;
    CHECK(lg->n_elements <= CACHE_SNAP_VOCAB);
    CHECK(kc->dims[1] <= CACHE_SNAP_KV && rows <= CACHE_SNAP_CTX);
    if (lg->n_elements > CACHE_SNAP_VOCAB || kc->dims[1] > CACHE_SNAP_KV
        || rows > CACHE_SNAP_CTX)
        return;
    s->kvdim = (int)kc->dims[1];
    s->rows = rows;
    s->position = vtm_position(m);
    s->generated = vtm_generated(m);
    n_logits = lg->n_elements;
    n_kv = (uint32_t)rows * (uint32_t)s->kvdim;
    memcpy(s->logits, lg->f32, (size_t)n_logits * sizeof(float));
    s->token = tk->f32[0];
    memcpy(s->kc, kc->f32, (size_t)n_kv * sizeof(float));
    memcpy(s->vc, vc->f32, (size_t)n_kv * sizeof(float));
}

static int cache_snap_equal(const cache_snap *a, const cache_snap *b) {
    size_t kv_bytes;

    if (a->kvdim != b->kvdim || a->rows != b->rows) return 0;
    if (a->position != b->position || a->generated != b->generated) return 0;
    if (memcmp(a->logits, b->logits, sizeof a->logits) != 0) return 0;
    if (memcmp(&a->token, &b->token, sizeof a->token) != 0) return 0;
    kv_bytes = (size_t)a->rows * (size_t)a->kvdim * sizeof(float);
    if (memcmp(a->kc, b->kc, kv_bytes) != 0) return 0;
    if (memcmp(a->vc, b->vc, kv_bytes) != 0) return 0;
    return 1;
}

// Полный прогон ids по одному токену — так это делает хост (CLI, api.c).
static void cache_run_plain(vtm_t *m, const int32_t *ids, int n) {
    int i;
    for (i = 0; i < n; i++) {
        vtm_set_token(m, ids[i]);
        vtm_set_position(m, i);
        CHECK(run_ok(m));
    }
}

static void test_cache_prefix(void) {
    cache_cfg cfg;
    int32_t prefix[8] = {3, 1, 4, 1, 5, 9, 2, 6};
    int32_t ids[16], ids2[16], other[16];
    int i, n, n2;
    uint64_t h0, m0, h1, m1;
    cache_snap want, got;
    vtm_t *base, *mc, *base2;

    cfg.ctx = 32;
    cfg.hidden = 16;
    cfg.vocab = 16;
    cfg.inter = 32;
    cfg.heads = 2;
    cfg.head_dim = 8;
    cfg.kv_heads = 1;

    base = cache_model(&cfg);
    mc = cache_model(&cfg);
    CHECK(base != NULL && mc != NULL);
    if (!base || !mc) {
        vtm_free(base);
        vtm_free(mc);
        return;
    }

    // (1) vtm_cache_prefix() сам по себе — ровно полный прогон префикса.
    CHECK(vtm_cache_prefix(mc, prefix, 8) == VTM_OK);
    CHECK(vtm_cache_prefix_len(mc) == 8);
    cache_run_plain(base, prefix, 8);
    cache_snap_take(base, &want, 8);
    cache_snap_take(mc, &got, 8);
    CHECK(cache_snap_equal(&want, &got));

    // (2) Тот же префикс + новые токены: префикс пропущен, результат побитово
    // тот же, что у полного прогона (логиты, токен, K/V, позиция, счётчик).
    n = 11;
    for (i = 0; i < 8; i++) ids[i] = prefix[i];
    ids[8] = 7;
    ids[9] = 0;
    ids[10] = 13;
    vtm_cache_stats(mc, &h0, &m0);
    vtm_reset(base);
    cache_run_plain(base, ids, n);
    cache_snap_take(base, &want, n);
    CHECK(vtm_run_cached(mc, ids, n) == VTM_OK);
    cache_snap_take(mc, &got, n);
    CHECK(cache_snap_equal(&want, &got));
    vtm_cache_stats(mc, &h1, &m1);
    CHECK(h1 == h0 + 1 && m1 == m0);        // префикс не пересчитывался
    CHECK(vtm_cache_prefix_len(mc) == 8);   // кэш остался префиксом

    // (3) Другой хвост при том же префиксе — снова попадание. Ровно этот
    // случай и нужен: системный промпт один, вопрос каждый раз новый.
    n2 = 10;
    for (i = 0; i < 8; i++) ids2[i] = prefix[i];
    ids2[8] = 12;
    ids2[9] = 2;
    vtm_reset(base);
    cache_run_plain(base, ids2, n2);
    cache_snap_take(base, &want, n2);
    CHECK(vtm_run_cached(mc, ids2, n2) == VTM_OK);
    cache_snap_take(mc, &got, n2);
    CHECK(cache_snap_equal(&want, &got));
    vtm_cache_stats(mc, &h1, &m1);
    CHECK(h1 == h0 + 2 && m1 == m0);
    CHECK(vtm_cache_prefix_len(mc) == 8);

    // (4) Несовпадающий префикс той же длины — полный прогон, правильный
    // результат и кэш на всю последовательность (следующий запрос с тем же
    // промптом и другим продолжением уже попадёт).
    for (i = 0; i < n; i++) other[i] = ids[i];
    other[0] = 8;
    vtm_reset(base);
    cache_run_plain(base, other, n);
    cache_snap_take(base, &want, n);
    vtm_cache_stats(mc, &h0, &m0);
    CHECK(vtm_run_cached(mc, other, n) == VTM_OK);
    cache_snap_take(mc, &got, n);
    CHECK(cache_snap_equal(&want, &got));
    vtm_cache_stats(mc, &h1, &m1);
    CHECK(m1 == m0 + 1 && h1 == h0);
    CHECK(vtm_cache_prefix_len(mc) == n);

    // (5) Та же последовательность ещё раз — попадание без единого прогона:
    // пропуск n токенов не должен менять уже посчитанные тензоры.
    cache_snap_take(mc, &want, n);
    vtm_cache_stats(mc, &h0, &m0);
    CHECK(vtm_run_cached(mc, other, n) == VTM_OK);
    cache_snap_take(mc, &got, n);
    CHECK(cache_snap_equal(&want, &got));
    vtm_cache_stats(mc, &h1, &m1);
    CHECK(h1 == h0 + 1 && m1 == m0);
    CHECK(vtm_generated(mc) == n);

    // (6) vtm_reset() кэш очищает: следующий вызов — полный прогон.
    vtm_reset(mc);
    CHECK(vtm_cache_prefix_len(mc) == 0);
    vtm_cache_stats(mc, &h0, &m0);
    CHECK(vtm_run_cached(mc, other, n) == VTM_OK);
    cache_snap_take(mc, &got, n);
    CHECK(cache_snap_equal(&want, &got));
    vtm_cache_stats(mc, &h1, &m1);
    CHECK(m1 == m0 + 1 && h1 == h0);

    // (7) Смена бэкенда — другой численный путь, кэш негоден.
    base2 = cache_model(&cfg);
    CHECK(base2 != NULL);
    if (base2) {
        CHECK(vtm_cache_prefix(mc, prefix, 8) == VTM_OK);
        CHECK(vtm_cache_prefix_len(mc) == 8);
        vtm_set_backend(mc, &vtm_backend_scalar_direct);
        vtm_set_backend(base2, &vtm_backend_scalar_direct);
        CHECK(vtm_cache_prefix_len(mc) == 0);
        vtm_reset(base2);
        cache_run_plain(base2, ids, n);
        cache_snap_take(base2, &want, n);
        vtm_cache_stats(mc, &h0, &m0);
        CHECK(vtm_run_cached(mc, ids, n) == VTM_OK);
        cache_snap_take(mc, &got, n);
        CHECK(cache_snap_equal(&want, &got));
        vtm_cache_stats(mc, &h1, &m1);
        CHECK(m1 == m0 + 1 && h1 == h0);
        vtm_free(base2);
        // Обратно на бэкенд по умолчанию (тот же, что у base: NULL —
        // vtm_backend_from_env(), а не конкретный скалярный).
        vtm_set_backend(mc, NULL);
    }

    // (8) Чужой прогон, переписывающий позицию префикса (position = 0 без
    // reset), кэш инвалидирует: слоты префикса в KV заменены.
    CHECK(vtm_cache_prefix(mc, prefix, 8) == VTM_OK);
    CHECK(vtm_cache_prefix_len(mc) == 8);
    vtm_set_token(mc, 15);
    vtm_set_position(mc, 0);
    CHECK(run_ok(mc));
    CHECK(vtm_cache_prefix_len(mc) == 0);
    vtm_reset(base);
    cache_run_plain(base, ids, n);
    cache_snap_take(base, &want, n);
    vtm_cache_stats(mc, &h0, &m0);
    CHECK(vtm_run_cached(mc, ids, n) == VTM_OK);
    cache_snap_take(mc, &got, n);
    CHECK(cache_snap_equal(&want, &got));
    vtm_cache_stats(mc, &h1, &m1);
    CHECK(m1 == m0 + 1 && h1 == h0);

    // (9) Генерация через vtm_run() (позиции >= длины префикса) кэш не
    // сбрасывает — так устроен реальный сервер: промпт посчитан, токены
    // досочинены, следующий запрос с тем же системным промптом снова попадает
    // в кэш, а снимок сэмплера восстанавливает историю после префикса.
    CHECK(vtm_cache_prefix(mc, prefix, 8) == VTM_OK);
    vtm_set_token(mc, 11);
    vtm_set_position(mc, 8);
    CHECK(run_ok(mc));
    vtm_set_token(mc, 12);
    vtm_set_position(mc, 9);
    CHECK(run_ok(mc));
    CHECK(vtm_cache_prefix_len(mc) == 8);
    vtm_reset(base);
    cache_run_plain(base, ids, n);
    cache_snap_take(base, &want, n);
    CHECK(vtm_run_cached(mc, ids, n) == VTM_OK);
    cache_snap_take(mc, &got, n);
    CHECK(cache_snap_equal(&want, &got));

    // (10) Последовательность короче кэша — это не префикс, полный прогон.
    CHECK(vtm_cache_prefix(mc, ids, n) == VTM_OK);   // кэш = 11 токенов
    vtm_reset(base);
    cache_run_plain(base, prefix, 8);
    cache_snap_take(base, &want, 8);
    vtm_cache_stats(mc, &h0, &m0);
    CHECK(vtm_run_cached(mc, prefix, 8) == VTM_OK);
    cache_snap_take(mc, &got, 8);
    CHECK(cache_snap_equal(&want, &got));
    vtm_cache_stats(mc, &h1, &m1);
    CHECK(m1 == m0 + 1 && h1 == h0);

    vtm_free(mc);
    vtm_free(base);
    if (!g_fail)
        printf("кэш префикса 0081: побитовое совпадение, попадания/промахи:"
               " ok\n");
}

// Кольцо KV меньше последовательности: K/V префикса уже нельзя восстановить
// после записи за пределы окна, поэтому кэш обязан инвалидироваться, а прогон
// с попаданием — оставаться правильным (ATTN читает только последние win
// ключей, так что переиспользование слота само по себе ошибки не даёт).
static void test_cache_prefix_ring(void) {
    cache_cfg cfg;
    int32_t prefix[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int32_t ids[12];
    cache_snap want, got;
    vtm_t *base, *mc;
    int i;

    cfg.ctx = 8;          // окно меньше префикса (10) и последовательности (12)
    cfg.hidden = 16;
    cfg.vocab = 16;
    cfg.inter = 32;
    cfg.heads = 2;
    cfg.head_dim = 8;
    cfg.kv_heads = 1;

    base = cache_model(&cfg);
    mc = cache_model(&cfg);
    CHECK(base != NULL && mc != NULL);
    if (!base || !mc) {
        vtm_free(base);
        vtm_free(mc);
        return;
    }
    for (i = 0; i < 10; i++) ids[i] = prefix[i];
    ids[10] = 11;
    ids[11] = 12;

    // (1) Префикс длиннее окна: KV уже провернулся, но последние win ключей
    // на месте — ближайшее попадание корректно, а первый же прогон за окно
    // (позиция 10 >= win) кэш инвалидирует: слоты префикса переиспользованы.
    CHECK(vtm_cache_prefix(mc, prefix, 10) == VTM_OK);
    CHECK(vtm_cache_prefix_len(mc) == 10);
    cache_run_plain(base, ids, 12);
    cache_snap_take(base, &want, cfg.ctx);
    CHECK(vtm_run_cached(mc, ids, 12) == VTM_OK);
    cache_snap_take(mc, &got, cfg.ctx);
    CHECK(cache_snap_equal(&want, &got));
    CHECK(vtm_cache_prefix_len(mc) == 0);

    // (2) Кольцо включается и на хите: префикс 6 + 4 новых токена при win=8.
    // K/V всех win слотов после прогона одинаковы у обоих путей.
    {
        int32_t p6[6] = {5, 4, 3, 2, 1, 0};
        int32_t seq[10];
        int j;
        for (j = 0; j < 6; j++) seq[j] = p6[j];
        seq[6] = 13;
        seq[7] = 14;
        seq[8] = 15;
        seq[9] = 3;
        vtm_reset(base);
        cache_run_plain(base, seq, 10);
        cache_snap_take(base, &want, cfg.ctx);
        CHECK(vtm_cache_prefix(mc, p6, 6) == VTM_OK);
        CHECK(vtm_cache_prefix_len(mc) == 6);
        CHECK(vtm_run_cached(mc, seq, 10) == VTM_OK);
        cache_snap_take(mc, &got, cfg.ctx);
        CHECK(cache_snap_equal(&want, &got));
        CHECK(vtm_cache_prefix_len(mc) == 0);
    }

    vtm_free(mc);
    vtm_free(base);
    if (!g_fail) printf("кэш префикса 0081 (кольцо KV): ok\n");
}

// Кэш префикса вместе с пачками (задача 0180). Проверяется проводка, которой
// раньше не было: префикс кэшируется, а некэшированный хвост считается
// пачками через vtm_run_batch(). На SIMD это уже не побитовый путь (батчевые
// ядра дают ≤1e-5), поэтому точность сверяется по argmax/позиции/счётчику, а
// самосогласованность кэша — побитово с полным прогоном теми же пачками.
static int cache_snap_argmax(const cache_snap *s) {
    int best = 0, i;
    for (i = 1; i < CACHE_SNAP_VOCAB; i++)
        if (s->logits[i] > s->logits[best]) best = i;
    return best;
}

static void test_cache_batch(void) {
    cache_cfg cfg;
    int32_t prefix[8] = {3, 1, 4, 1, 5, 9, 2, 6};
    int32_t ids[12];
    cache_snap plain, full_batch, from_cache, again;
    vtm_t *base, *ref, *mc;
    uint64_t h0, m0, h1, m1;
    int i;

    cfg.ctx = 32;
    cfg.hidden = 16;
    cfg.vocab = 16;
    cfg.inter = 32;
    cfg.heads = 2;
    cfg.head_dim = 8;
    cfg.kv_heads = 1;

    base = cache_model(&cfg);   // эталон: по токену
    ref = cache_model(&cfg);    // весь промпт одной и той же пачкой
    mc = cache_model(&cfg);     // кэш префикса + пачка хвоста
    CHECK(base != NULL && ref != NULL && mc != NULL);
    if (!base || !ref || !mc) {
        vtm_free(base);
        vtm_free(ref);
        vtm_free(mc);
        return;
    }
    for (i = 0; i < 8; i++) ids[i] = prefix[i];
    ids[8] = 7;
    ids[9] = 0;
    ids[10] = 13;
    ids[11] = 11;

    // (1) Точность: пачка с кэшем против поштучного пути — argmax, позиция и
    // счётчик сгенерированных совпадают (на скаляре совпало бы и побитово).
    cache_run_plain(base, ids, 12);
    cache_snap_take(base, &plain, 12);
    CHECK(vtm_cache_prefix_ex(ref, ids, 12, 4) == VTM_OK);
    cache_snap_take(ref, &full_batch, 12);
    CHECK(cache_snap_argmax(&plain) == cache_snap_argmax(&full_batch));
    CHECK(plain.position == full_batch.position);
    CHECK(plain.generated == full_batch.generated);

    // (2) Кэш + пачка: префикс 8 считается пачками 4+4, затем попадание и
    // хвост 4 — ещё одна пачка. Это ровно те же пачки в том же порядке, что у
    // полного прогона ref, поэтому состояние обязано совпасть побитово (даже
    // на SIMD: раскладка пачек не меняется).
    CHECK(vtm_cache_prefix_ex(mc, prefix, 8, 4) == VTM_OK);
    CHECK(vtm_cache_prefix_len(mc) == 8);
    vtm_cache_stats(mc, &h0, &m0);
    CHECK(vtm_run_cached_ex(mc, ids, 12, 4) == VTM_OK);
    cache_snap_take(mc, &from_cache, 12);
    CHECK(cache_snap_equal(&full_batch, &from_cache));
    vtm_cache_stats(mc, &h1, &m1);
    CHECK(h1 == h0 + 1 && m1 == m0);          // префикс не пересчитывался
    CHECK(vtm_cache_prefix_len(mc) == 8);     // кэш остался префиксом

    // (3) Повтор того же запроса — полное попадание: состояние не меняется
    // (пачки не запускаются вовсе), счётчик попаданий растёт.
    vtm_cache_stats(mc, &h0, &m0);
    CHECK(vtm_run_cached_ex(mc, ids, 12, 4) == VTM_OK);
    cache_snap_take(mc, &again, 12);
    CHECK(cache_snap_equal(&from_cache, &again));
    vtm_cache_stats(mc, &h1, &m1);
    CHECK(h1 == h0 + 1 && m1 == m0);

    // (4) prefill_batch == 0 — выключатель: прежний путь (на SIMD по токену)
    // даёт тот же argmax и позицию.
    vtm_reset(mc);
    CHECK(vtm_run_cached_ex(mc, ids, 12, 0) == VTM_OK);
    cache_snap_take(mc, &again, 12);
    CHECK(cache_snap_argmax(&plain) == cache_snap_argmax(&again));
    CHECK(plain.position == again.position);
    CHECK(plain.generated == again.generated);

    vtm_free(mc);
    vtm_free(ref);
    vtm_free(base);
    if (!g_fail)
        printf("кэш префикса + пачки 0180: точность и попадание: ok\n");
}


// --- батчевый prefill (задача 0112) ----------------------------------------
//
// Проверяется контракт vtm_run_batch(): после пачки состояние машины —
// логиты последнего токена, K/V, счётчик сгенерированных, сэмплер, halted,
// input_token и позиция — обязано быть побитово тем же, что после n вызовов
// vtm_run() по одному токену. Батчевые ядра есть только у скалярного
// бэкенда (SIMD-ядра — зона 0113), поэтому здесь бэкенд выставляется явно;
// на прочих бэкендах вызов идёт прежним путём по токену, и проверка
// побитовости проходит по построению.
//
// Конфигурации: GQA + causal, MHA (heads == kv_heads) + causal, и кольцо KV
// (context_window меньше длины последовательности): пачка, перешедшая
// границу окна, обязана откатиться на путь по токену и остаться правильной.

typedef struct {
    int ctx;
    int hidden;
    int vocab;
    int inter;
    int heads;
    int head_dim;
    int kv_heads;
    int window;   // 0 — context_window в мете не задан (окно = ctx)
} batch_cfg;

enum {
    BT_EMB, BT_X, BT_H1, BT_N1, BT_Q, BT_K, BT_V, BT_QW, BT_KW, BT_VW,
    BT_KC, BT_VC, BT_ATTN, BT_O, BT_OW, BT_H2, BT_N2, BT_GATE, BT_UP,
    BT_GW, BT_UW, BT_MUL, BT_DOWN, BT_DW, BT_NF, BT_HF, BT_LOGITS,
    BT_LMW, BT_TOKEN
};

// Та же программа, что у cache_model(), плюс context_window в мете.
static vtm_t *batch_model(const batch_cfg *cfg) {
    model_t md;
    uint32_t d_emb[2], d_vec[1], d_q[3], d_k[3], d_c[2], d_g[1];
    uint32_t d_log[2], d_one[1];
    int kv_dim = cfg->kv_heads * cfg->head_dim;
    uint8_t *img;
    size_t size;
    char meta[256];
    vtm_t *m;

    d_emb[0] = (uint32_t)cfg->vocab;
    d_emb[1] = (uint32_t)cfg->hidden;
    d_vec[0] = (uint32_t)cfg->hidden;
    d_q[0] = (uint32_t)cfg->heads;
    d_q[1] = 1;
    d_q[2] = (uint32_t)cfg->head_dim;
    d_k[0] = (uint32_t)cfg->kv_heads;
    d_k[1] = 1;
    d_k[2] = (uint32_t)cfg->head_dim;
    d_c[0] = (uint32_t)cfg->ctx;
    d_c[1] = (uint32_t)kv_dim;
    d_g[0] = (uint32_t)cfg->inter;
    d_log[0] = (uint32_t)cfg->vocab;
    d_log[1] = 1;
    d_one[0] = 1;

    model_init(&md);
    cache_add_f32(&md, "emb.w", 2, d_emb, 0.0f, 0.05f);
    cache_add_f32(&md, "x", 1, d_vec, 0.0f, 0.0f);
    cache_add_f32(&md, "h1", 1, d_vec, 0.0f, 0.0f);
    cache_add_f32(&md, "norm1.w", 1, d_vec, 1.0f, 0.02f);
    cache_add_f32(&md, "q", 3, d_q, 0.0f, 0.0f);
    cache_add_f32(&md, "k", 3, d_k, 0.0f, 0.0f);
    cache_add_f32(&md, "v", 3, d_k, 0.0f, 0.0f);
    cache_add_trit(&md, "q_proj.w", cfg->hidden, cfg->hidden);
    cache_add_trit(&md, "k_proj.w", kv_dim, cfg->hidden);
    cache_add_trit(&md, "v_proj.w", kv_dim, cfg->hidden);
    cache_add_f32(&md, "kcache", 2, d_c, 0.0f, 0.0f);
    cache_add_f32(&md, "vcache", 2, d_c, 0.0f, 0.0f);
    cache_add_f32(&md, "attn", 3, d_q, 0.0f, 0.0f);
    cache_add_f32(&md, "o", 1, d_vec, 0.0f, 0.0f);
    cache_add_trit(&md, "o_proj.w", cfg->hidden, cfg->hidden);
    cache_add_f32(&md, "h2", 1, d_vec, 0.0f, 0.0f);
    cache_add_f32(&md, "norm2.w", 1, d_vec, 1.0f, 0.02f);
    cache_add_f32(&md, "gate", 1, d_g, 0.0f, 0.0f);
    cache_add_f32(&md, "up", 1, d_g, 0.0f, 0.0f);
    cache_add_trit(&md, "gate_proj.w", cfg->inter, cfg->hidden);
    cache_add_trit(&md, "up_proj.w", cfg->inter, cfg->hidden);
    cache_add_f32(&md, "mul", 1, d_g, 0.0f, 0.0f);
    cache_add_f32(&md, "down", 1, d_vec, 0.0f, 0.0f);
    cache_add_trit(&md, "down_proj.w", cfg->hidden, cfg->inter);
    cache_add_f32(&md, "final_norm.w", 1, d_vec, 1.0f, 0.02f);
    cache_add_f32(&md, "hfinal", 1, d_vec, 0.0f, 0.0f);
    cache_add_f32(&md, "logits", 2, d_log, 0.0f, 0.0f);
    cache_add_trit(&md, "lm_head.w", cfg->vocab, cfg->hidden);
    cache_add_f32(&md, "token", 1, d_one, 0.0f, 0.0f);

    model_inst(&md, VTM_OP_EMB, VTM_FLAG_EMB_INPUT, BT_X, BT_EMB,
               VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_RMSN, 0, BT_H1, BT_X, BT_N1, VTM_NO_TENSOR, 1);
    model_inst(&md, VTM_OP_TMV, 0, BT_Q, BT_QW, BT_H1, VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_TMV, 0, BT_K, BT_KW, BT_H1, VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_TMV, 0, BT_V, BT_VW, BT_H1, VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_ROPE, 0, BT_Q, BT_Q, (uint16_t)cfg->heads,
               VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_ROPE, 0, BT_K, BT_K, (uint16_t)cfg->kv_heads,
               VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_KVW, 0, BT_KC, BT_K, BT_VC, BT_V, 0);
    model_inst(&md, VTM_OP_ATTN, VTM_FLAG_CAUSAL, BT_ATTN, BT_Q, BT_KC,
               BT_VC, 0);
    model_inst(&md, VTM_OP_TMV, 0, BT_O, BT_OW, BT_ATTN, VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_ADD, 0, BT_X, BT_X, BT_O, VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_RMSN, 0, BT_H2, BT_X, BT_N2, VTM_NO_TENSOR, 1);
    model_inst(&md, VTM_OP_TMV, 0, BT_GATE, BT_GW, BT_H2, VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_TMV, 0, BT_UP, BT_UW, BT_H2, VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_SILU, 0, BT_GATE, BT_GATE, VTM_NO_TENSOR,
               VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_MUL, 0, BT_MUL, BT_GATE, BT_UP, VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_TMV, 0, BT_DOWN, BT_DW, BT_MUL, VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_ADD, 0, BT_X, BT_X, BT_DOWN, VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_RMSN, 0, BT_HF, BT_X, BT_NF, VTM_NO_TENSOR, 1);
    model_inst(&md, VTM_OP_TMV, 0, BT_LOGITS, BT_LMW, BT_HF,
               VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_SAMP, 0, BT_TOKEN, BT_LOGITS, VTM_NO_TENSOR,
               VTM_NO_TENSOR, VTM_SAMP_TEMPERATURE);
    model_inst(&md, VTM_OP_HALT, 0, VTM_NO_TENSOR, VTM_NO_TENSOR,
               VTM_NO_TENSOR, VTM_NO_TENSOR, 0);

    if (cfg->window > 0)
        snprintf(meta, sizeof meta,
                 "{\"temperature\": 0.8, \"sample_seed\": 7,"
                 " \"rms_norm_eps\": 1e-06, \"context_window\": %d}",
                 cfg->window);
    else
        snprintf(meta, sizeof meta,
                 "{\"temperature\": 0.8, \"sample_seed\": 7,"
                 " \"rms_norm_eps\": 1e-06}");
    model_set_meta(&md, meta);
    model_build(&md, &img, &size);
    m = load_ok(img, size);
    free(img);
    model_free(&md);
    return m;
}


// Сравнение всего наблюдаемого состояния двух машин. tol == 0 — строго
// побитово (скалярный батч обязан совпадать с прогоном по токену точно),
// tol > 0 — допуск на f32-тензоры (SIMD-батч 0122 считает с другим порядком
// f32-сложения: расхождение ≤1e-5, зато выбор токена обязан остаться тем же).
static void batch_check_same(vtm_t *want, vtm_t *got, const char *what,
                             double tol) {
    uint32_t i, n = vtm_tensor_count(want);
    int bad = 0;

    if (vtm_tensor_count(got) != n) {
        fprintf(stderr, "FAIL batch %s: разное число тензоров\n", what);
        g_fail++;
        return;
    }
    if (vtm_position(want) != vtm_position(got)
        || vtm_input_token(want) != vtm_input_token(got)
        || vtm_generated(want) != vtm_generated(got)
        || vtm_halted(want) != vtm_halted(got)) {
        fprintf(stderr, "FAIL batch %s: позиция %d/%d, вход %d/%d, "
                "сгенерировано %d/%d, halted %d/%d\n", what,
                vtm_position(want), vtm_position(got),
                vtm_input_token(want), vtm_input_token(got),
                vtm_generated(want), vtm_generated(got),
                vtm_halted(want), vtm_halted(got));
        g_fail++;
    }
    for (i = 0; i < n; i++) {
        const vtm_tensor *a = vtm_tensor_by_index(want, i);
        const vtm_tensor *b = vtm_tensor_by_index(got, i);
        if (a->n_elements != b->n_elements) {
            fprintf(stderr, "FAIL batch %s: %s: элементов %u/%u\n", what,
                    a->name, a->n_elements, b->n_elements);
            g_fail++;
            continue;
        }
        if (!a->f32 || !b->f32 || a->n_elements == 0) continue;
        if (tol <= 0.0) {
            // Побитово: -0.0/0.0 и NaN тоже считаются расхождением.
            if (memcmp(a->f32, b->f32, (size_t)a->n_elements * 4u) != 0) {
                double d = 0.0;
                uint32_t k;
                for (k = 0; k < a->n_elements; k++) {
                    double v = fabs((double)a->f32[k] - (double)b->f32[k]);
                    if (v > d) d = v;
                }
                fprintf(stderr, "FAIL batch %s: тензор %s (%u эл.) не "
                        "побитово, max|diff|=%.3e\n", what, a->name,
                        a->n_elements, d);
                g_fail++;
                if (++bad >= 5) break;
            }
        } else {
            double d = 0.0;
            uint32_t k;
            for (k = 0; k < a->n_elements; k++) {
                double v = fabs((double)a->f32[k] - (double)b->f32[k]);
                if (v > d) d = v;
            }
            if (d > tol) {
                fprintf(stderr, "FAIL batch %s: тензор %s (%u эл.) "
                        "max|diff|=%.3e > %.0e\n", what, a->name,
                        a->n_elements, d, tol);
                g_fail++;
                if (++bad >= 5) break;
            }
        }
    }
    // Выбор токена важнее чисел: при любом допуске он должен совпасть точно
    // (на этом держится «батч не меняет ответ»).
    {
        const vtm_tensor *ta = vtm_tensor_by_name(want, "token");
        const vtm_tensor *tb = vtm_tensor_by_name(got, "token");
        if (ta && tb && ta->f32 && tb->f32 && ta->n_elements > 0
            && tb->n_elements > 0
            && memcmp(ta->f32, tb->f32, sizeof(float)) != 0) {
            fprintf(stderr, "FAIL batch %s: выбранный токен %.0f != %.0f\n",
                    what, (double)ta->f32[0], (double)tb->f32[0]);
            g_fail++;
        }
    }
}

// Одна конфигурация: n токенов по одному против одной пачки.
static void batch_case(const batch_cfg *cfg, int n, const char *what,
                       int force_scalar) {
    vtm_t *want, *got;
    int32_t ids[128];
    int i;

    CHECK(n <= 128);
    want = batch_model(cfg);
    got = batch_model(cfg);
    CHECK(want != NULL && got != NULL);
    if (!want || !got) {
        vtm_free(want);
        vtm_free(got);
        return;
    }
    if (force_scalar) {
        vtm_set_backend(want, &vtm_backend_scalar_lut);
        vtm_set_backend(got, &vtm_backend_scalar_lut);
    }
    for (i = 0; i < n; i++) ids[i] = (int32_t)((i * 7 + 3) % cfg->vocab);
    cache_run_plain(want, ids, n);
    vtm_set_position(got, 0);
    CHECK(vtm_run_batch(got, ids, n) == VTM_OK);
    CHECK(vtm_position(got) == n - 1);
    CHECK(vtm_input_token(got) == ids[n - 1]);
    // Батч не создаёт снимок кэша префикса: K/V он кладёт, но «это ровно
    // прогон ids[0..n)» машина не запоминает.
    CHECK(vtm_cache_prefix_len(got) == 0);
    // Скалярный батч обязан совпасть побитово; на SIMD (ядра 0122) — 1e-5,
    // выбор токена проверяется точно в batch_check_same.
    batch_check_same(want, got, what, force_scalar ? 0.0 : 1e-5);
    vtm_free(want);
    vtm_free(got);
}

// MM с n == 1 (граф tied lm_head): EMB -> CPY (столбец) -> MM -> SAMP.
enum { BMM_EMB, BMM_X, BMM_XCOL, BMM_LOGITS, BMM_TOKEN };

static vtm_t *batch_mm_model(void) {
    model_t md;
    uint32_t d_x[1], d_col[2], d_log[2], d_one[1];
    uint8_t *img;
    size_t size;
    vtm_t *m;

    d_x[0] = 8;
    d_col[0] = 8; d_col[1] = 1;
    d_log[0] = 16; d_log[1] = 1;
    d_one[0] = 1;

    model_init(&md);
    cache_add_trit(&md, "emb.w", 16, 8);
    cache_add_f32(&md, "x", 1, d_x, 0.0f, 0.05f);
    cache_add_f32(&md, "xcol", 2, d_col, 0.0f, 0.0f);
    cache_add_f32(&md, "logits", 2, d_log, 0.0f, 0.0f);
    cache_add_f32(&md, "token", 1, d_one, 0.0f, 0.0f);

    model_inst(&md, VTM_OP_EMB, VTM_FLAG_EMB_INPUT, BMM_X, BMM_EMB,
               VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_CPY, 0, BMM_XCOL, BMM_X, VTM_NO_TENSOR,
               VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_MM, 0, BMM_LOGITS, BMM_EMB, BMM_XCOL,
               VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_SAMP, 0, BMM_TOKEN, BMM_LOGITS, VTM_NO_TENSOR,
               VTM_NO_TENSOR, VTM_SAMP_ARGMAX);
    model_inst(&md, VTM_OP_HALT, 0, VTM_NO_TENSOR, VTM_NO_TENSOR,
               VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    model_set_meta(&md, "{\"temperature\": 0, \"rms_norm_eps\": 1e-06}");
    model_build(&md, &img, &size);
    m = load_ok(img, size);
    free(img);
    model_free(&md);
    return m;
}

static void batch_mm_case(int n, const char *what) {
    vtm_t *want = batch_mm_model();
    vtm_t *got = batch_mm_model();
    int32_t ids[32];
    int i;

    CHECK(want != NULL && got != NULL);
    if (!want || !got) {
        vtm_free(want);
        vtm_free(got);
        return;
    }
    vtm_set_backend(want, &vtm_backend_scalar_lut);
    vtm_set_backend(got, &vtm_backend_scalar_lut);
    for (i = 0; i < n; i++) ids[i] = (int32_t)((i * 7 + 3) % 16);
    cache_run_plain(want, ids, n);
    vtm_set_position(got, 0);
    CHECK(vtm_run_batch(got, ids, n) == VTM_OK);
    batch_check_same(want, got, what, 0.0);
    vtm_free(want);
    vtm_free(got);
}

static void test_run_batch(void) {
    batch_cfg gqa, mha, ring;
    int32_t ids[16];
    vtm_t *m;
    int i;

    printf("  -- батчевый prefill: vtm_run_batch\n");
    gqa.ctx = 256; gqa.hidden = 32; gqa.vocab = 32; gqa.inter = 64;
    gqa.heads = 4; gqa.kv_heads = 2; gqa.head_dim = 8; gqa.window = 0;
    mha = gqa; mha.kv_heads = 4;
    ring = gqa; ring.window = 32;

    batch_case(&gqa, 8, "GQA/causal 8", 1);
    batch_case(&gqa, 64, "GQA/causal 64", 1);
    batch_case(&gqa, 128, "GQA/causal 128", 1);
    batch_case(&mha, 64, "MHA/causal 64", 1);
    batch_case(&ring, 64, "кольцо KV 64 (окно 32)", 1);
    batch_case(&ring, 128, "кольцо KV 128 (окно 32)", 1);
    // Бэкенд по умолчанию (обычно SIMD): с задачей 0122 там есть батчевые
    // ядра, и порядок f32-сложения у них другой — состояние совпадает в
    // пределах 1e-5, а не побитово (побитовость остаётся на скаляре, её
    // проверяют кейсы выше; выбор токена проверяется точно).
    batch_case(&gqa, 16, "бэкенд по умолчанию 16", 0);

    // MM-путь: tied lm_head (A — тритовая таблица эмбеддингов, B — столбец).
    batch_mm_case(8, "MM 8");
    batch_mm_case(32, "MM 32");

    // Две пачки подряд == один последовательный прогон: стык пачек (позиция,
    // активации последнего токена) не теряется.
    {
        vtm_t *want = batch_model(&gqa);
        vtm_t *got = batch_model(&gqa);
        CHECK(want != NULL && got != NULL);
        if (want && got) {
            vtm_set_backend(want, &vtm_backend_scalar_lut);
            vtm_set_backend(got, &vtm_backend_scalar_lut);
            for (i = 0; i < 16; i++) ids[i] = (int32_t)((i * 5 + 1) % 32);
            cache_run_plain(want, ids, 16);
            vtm_set_position(got, 0);
            CHECK(vtm_run_batch(got, ids, 8) == VTM_OK);
            vtm_set_position(got, 8);
            CHECK(vtm_run_batch(got, ids + 8, 8) == VTM_OK);
            batch_check_same(want, got, "две пачки по 8", 0.0);
            vtm_free(want);
            vtm_free(got);
        }
    }

    // n == 0 — no-op; аргументы; граница int.
    m = batch_model(&gqa);
    CHECK(m != NULL);
    if (m) {
        vtm_set_backend(m, &vtm_backend_scalar_lut);
        for (i = 0; i < 4; i++) ids[i] = (int32_t)i;
        vtm_set_position(m, 5);
        vtm_set_token(m, 3);
        CHECK(vtm_run_batch(m, ids, 0) == VTM_OK);
        CHECK(vtm_position(m) == 5);
        CHECK(vtm_input_token(m) == 3);
        CHECK(vtm_generated(m) == 0);
        CHECK(vtm_run_batch(NULL, ids, 1) == VTM_E_ARG);
        CHECK(vtm_run_batch(m, NULL, 1) == VTM_E_ARG);
        CHECK(vtm_run_batch(m, ids, -1) == VTM_E_ARG);
        vtm_set_position(m, INT_MAX);
        CHECK(vtm_run_batch(m, ids, 2) == VTM_E_BOUNDS);
        vtm_free(m);
    }

    // n == 1 — ровно один прогон.
    {
        vtm_t *want = batch_model(&gqa);
        vtm_t *got = batch_model(&gqa);
        CHECK(want != NULL && got != NULL);
        if (want && got) {
            vtm_set_backend(want, &vtm_backend_scalar_lut);
            vtm_set_backend(got, &vtm_backend_scalar_lut);
            ids[0] = 9;
            cache_run_plain(want, ids, 1);
            vtm_set_position(got, 0);
            CHECK(vtm_run_batch(got, ids, 1) == VTM_OK);
            batch_check_same(want, got, "n=1", 0.0);
            vtm_free(want);
            vtm_free(got);
        }
    }

    if (!g_fail)
        printf("  -- батчевый prefill: побитово (8/64/128, GQA/MHA/кольцо):"
               " ok\n");
}

// --- отмена пачки prefill (задача 0185) -----------------------------------
//
// До 0185 vtm_run_batch() был неделим: «стоп» на промпте ждал всю пачку (до
// 256 токенов), а серверный клиент всё это время не мог занять контекст.
// Теперь интерпретатор опрашивает vtm_cancel_fn перед инструкциями пачки.
// Проверяем три вещи:
//   * хук, который никогда не просит остановки, не меняет результат —
//     побитово то же, что прогон без хука;
//   * хук, который просит остановку, возвращает VTM_E_CANCELLED и правда
//     прерывает пачку (предикат вызван, но инструкции не все);
//   * после отмены полный прогон с нуля даёт побитово эталонный результат:
//     частичное состояние машины не отравляет следующий запрос.
typedef struct {
    int calls;
    int stop_after;
} cancel_probe;

static int cancel_probe_fn(void *user) {
    cancel_probe *p = (cancel_probe *)user;
    p->calls++;
    return p->calls > p->stop_after;
}

static void test_run_batch_cancel(void) {
    batch_cfg gqa;
    int32_t ids[64];
    cancel_probe never = {0, 1 << 30};
    cancel_probe doomed = {0, 3};
    vtm_t *m, *ref;
    int i;

    printf("  -- отмена пачки prefill: vtm_set_cancel\n");
    gqa.ctx = 256; gqa.hidden = 32; gqa.vocab = 32; gqa.inter = 64;
    gqa.heads = 4; gqa.kv_heads = 2; gqa.head_dim = 8; gqa.window = 0;
    for (i = 0; i < 64; i++) ids[i] = (int32_t)((i * 7 + 1) % 32);

    m = batch_model(&gqa);
    ref = batch_model(&gqa);
    CHECK(m != NULL && ref != NULL);
    if (!m || !ref) {
        vtm_free(m);
        vtm_free(ref);
        return;
    }
    vtm_set_backend(m, &vtm_backend_scalar_lut);
    vtm_set_backend(ref, &vtm_backend_scalar_lut);

    // (1) Хук есть, отмены нет — ответ тот же, что без хука.
    vtm_set_cancel(m, cancel_probe_fn, &never);
    vtm_set_position(m, 0);
    CHECK(vtm_run_batch(m, ids, 32) == VTM_OK);
    CHECK(never.calls > 0); // хук действительно опрашивается
    cache_run_plain(ref, ids, 32);
    batch_check_same(ref, m, "хук без отмены == обычный прогон", 0.0);

    // (2) Отмена на третьей проверке: пачка возвращает VTM_E_CANCELLED.
    vtm_set_cancel(m, cancel_probe_fn, &doomed);
    vtm_reset(m);
    vtm_set_position(m, 0);
    CHECK(vtm_run_batch(m, ids, 64) == VTM_E_CANCELLED);
    CHECK(doomed.calls > 3);

    // (3) Хук снят, прогон с нуля — побитово эталон (64 токена).
    vtm_set_cancel(m, NULL, NULL);
    vtm_reset(ref);
    cache_run_plain(ref, ids, 64);
    vtm_reset(m);
    vtm_set_position(m, 0);
    CHECK(vtm_run_batch(m, ids, 64) == VTM_OK);
    batch_check_same(ref, m, "после отмены полный прогон == эталон", 0.0);

    // (4) Поштучный путь пачки (batch_sequential): трассировка выключает
    // батчевые ядра, и vtm_run_batch идёт по токену — отмена тоже видна.
    {
        cancel_probe one = {0, 0};
        trace_probe trace = {0};
        vtm_set_trace(m, trace_cb, &trace);
        vtm_set_cancel(m, cancel_probe_fn, &one);
        vtm_reset(m);
        vtm_set_position(m, 0);
        CHECK(vtm_run_batch(m, ids, 4) == VTM_E_CANCELLED);
        CHECK(one.calls > 0);
        vtm_set_cancel(m, NULL, NULL);
        vtm_set_trace(m, NULL, NULL);
    }

    vtm_free(m);
    vtm_free(ref);
    if (!g_fail)
        printf("  -- отмена пачки prefill (vtm_set_cancel): ok\n");
}

// --- замер кэша префикса (задача 0081) ------------------------------------
//
// Сценарий из задачи: 128-токенный префикс + 16 новых токенов.
// «До» — каждый запрос считает все 144 токена. «После» — префикс считается
// один раз (vtm_cache_prefix), дальше каждый запрос пропускает его и считает
// только 16 токенов. Печатаются мс/запрос и ускорение.
//
//   cd src/vtm && VTM_CACHE_BENCH=1 ./test_vtm
//   VTM_CACHE_BENCH=1 VTM_CACHE_BENCH_MODEL=/tmp/eva/ctx256/model.vtm ./test_vtm
//   VTM_CACHE_BENCH_ITERS=10 ...    (по умолчанию 5)
//
// Демо-контейнер имеет KV на 8 позиций; для 128+16 его надо расширить:
//   python3 -c 'from tools import bench_vtm;
//     print(bench_vtm.prepare_container("demo/model.vtm", 256, "/tmp/eva"))'
// Без VTM_CACHE_BENCH_MODEL строится синтетическая модель той же формы
// (hidden=128, 1 слой), чтобы замер воспроизводился без файлов. Токены
// берутся < vocab, поэтому подходит и демо-словарь на 128.
#define CACHE_BENCH_PREFIX 128
#define CACHE_BENCH_FRESH  16
#define CACHE_BENCH_TOTAL  (CACHE_BENCH_PREFIX + CACHE_BENCH_FRESH)

static void bench_cache_prefix(void) {
    const char *enable = getenv("VTM_CACHE_BENCH");
    const char *path = getenv("VTM_CACHE_BENCH_MODEL");
    const char *iters_env = getenv("VTM_CACHE_BENCH_ITERS");
    int iters = iters_env ? atoi(iters_env) : 5;
    int32_t ids[CACHE_BENCH_TOTAL];
    vtm_t *plain, *cached;
    const vtm_tensor *emb;
    int i, it, vocab = 128;
    double t0, full_ms = 0.0, cached_ms = 0.0, prime_ms = 0.0;
    uint64_t hits = 0, misses = 0;

    if (!enable) return;
    if (iters < 1) iters = 1;
    if (iters > 100) iters = 100;

    if (path) {
        plain = vtm_load(path);
        cached = vtm_load(path);
        CHECK(plain != NULL && cached != NULL);
        if (!plain || !cached) {
            vtm_free(plain);
            vtm_free(cached);
            return;
        }
    } else {
        cache_cfg cfg;
        cfg.ctx = 256;
        cfg.hidden = 128;
        cfg.vocab = 128;
        cfg.inter = 256;
        cfg.heads = 4;
        cfg.head_dim = 32;
        cfg.kv_heads = 1;
        plain = cache_model(&cfg);
        cached = cache_model(&cfg);
    }
    emb = vtm_tensor_by_name(plain, "emb.w");
    if (emb && emb->ndim >= 1 && emb->dims[0] > 0)
        vocab = (int)emb->dims[0];
    for (i = 0; i < CACHE_BENCH_TOTAL; i++)
        ids[i] = (int32_t)((i * 37u + 11u) % (uint32_t)vocab);

    // Прогрев префикса — разовая плата; её показываем отдельно.
    t0 = bench_now();
    CHECK(vtm_cache_prefix(cached, ids, CACHE_BENCH_PREFIX) == VTM_OK);
    prime_ms = (bench_now() - t0) * 1000.0;

    for (it = 0; it < iters; it++) {
        // Разные хвосты у одного префикса — как разные вопросы при одном
        // системном промпте.
        ids[CACHE_BENCH_TOTAL - 1] = (int32_t)((it * 13 + 5) % vocab);
        vtm_reset(plain);
        t0 = bench_now();
        for (i = 0; i < CACHE_BENCH_TOTAL; i++) {
            vtm_set_token(plain, ids[i]);
            vtm_set_position(plain, i);
            CHECK(run_ok(plain));
        }
        full_ms += bench_now() - t0;
        t0 = bench_now();
        CHECK(vtm_run_cached(cached, ids, CACHE_BENCH_TOTAL) == VTM_OK);
        cached_ms += bench_now() - t0;
        {
            const vtm_tensor *a = vtm_tensor_by_name(plain, "logits");
            const vtm_tensor *b = vtm_tensor_by_name(cached, "logits");
            CHECK(a && b && a->n_elements == b->n_elements
                  && memcmp(a->f32, b->f32,
                            (size_t)a->n_elements * sizeof(float)) == 0);
        }
    }
    full_ms = full_ms * 1000.0 / (double)iters;
    cached_ms = cached_ms * 1000.0 / (double)iters;
    vtm_cache_stats(cached, &hits, &misses);

    printf("кэш префикса 0081: %s, префикс %d + %d, запросов %d\n",
           path ? path : "синтетика (1 слой, hidden=128)",
           CACHE_BENCH_PREFIX, CACHE_BENCH_FRESH, iters);
    printf("  до:      %8.3f мс/запрос (полный прогон %d токенов)\n",
           full_ms, CACHE_BENCH_TOTAL);
    printf("  прогрев: %8.3f мс (один раз, %d токенов)\n",
           prime_ms, CACHE_BENCH_PREFIX);
    printf("  после:   %8.3f мс/запрос (пропуск префикса, %d токенов)\n",
           cached_ms, CACHE_BENCH_FRESH);
    if (cached_ms > 0.0) {
        printf("  ускорение: %.2f× без прогрева; с прогревом на %d запросов"
               " — %.2f×\n",
               full_ms / cached_ms, iters,
               (double)iters * full_ms
                   / (prime_ms + (double)iters * cached_ms));
    }
    printf("  попаданий %llu, промахов %llu; логиты совпали побитово\n",
           (unsigned long long)hits, (unsigned long long)misses);
    if (hits != (uint64_t)iters)
        printf("  внимание: попаданий %llu из %d — KV-кэш короче"
               " последовательности или кэш сбрасывается\n",
               (unsigned long long)hits, iters);
    vtm_free(cached);
    vtm_free(plain);
}

// --- MoE: pread-кэш экспертов (задача 0175) --------------------------------

// Банк kind 6: 18-байтовый заголовок сегмента + блоки GGML (144 Б). Значения
// детерминированные, d конечное и маленькое: тесту важна не точность весов, а
// то, что pread отдаёт ровно те же байты, которые дал бы mmap.
static void model_add_q4k_bank_d(model_t *m, const char *name, int ndim,
                                 const uint32_t *dims, uint32_t seed,
                                 float d) {
    tensor_def *td = model_slot(m, name, VTM_TENSOR_Q4K, ndim, dims);
    uint32_t n = dims_product(ndim, dims);
    uint32_t n_blk = (n + VTM_Q4K_BLOCK_VALUES - 1u) / VTM_Q4K_BLOCK_VALUES;
    buf_t b = {0};
    uint32_t k, i;

    buf_append(&b, VTM_Q4K_MAGIC, 4);
    buf_u8(&b, (uint8_t)VTM_Q4K_VERSION);
    buf_u8(&b, 0); // reserved
    buf_u64(&b, n);
    buf_u32(&b, n_blk);
    for (k = 0; k < n_blk; k++) {
        uint8_t block[VTM_Q4K_BLOCK_BYTES];
        for (i = 0; i < sizeof block; i++)
            block[i] = (uint8_t)((i * 37u + k * 11u + seed * 29u) & 0xFFu);
        put_u16(block, vtm_float_to_half(d));        // d
        put_u16(block + 2, vtm_float_to_half(0.0f)); // dmin
        buf_append(&b, block, sizeof block);
    }
    td->blob = b.p;
    td->blob_len = b.len;
}

// Банк kind 7: блок 210 Б, d — последние два байта.
static void model_add_q6k_bank_d(model_t *m, const char *name, int ndim,
                                 const uint32_t *dims, uint32_t seed,
                                 float d) {
    tensor_def *td = model_slot(m, name, VTM_TENSOR_Q6K, ndim, dims);
    uint32_t n = dims_product(ndim, dims);
    uint32_t n_blk = (n + VTM_Q6K_BLOCK_VALUES - 1u) / VTM_Q6K_BLOCK_VALUES;
    buf_t b = {0};
    uint32_t k, i;

    buf_append(&b, VTM_Q6K_MAGIC, 4);
    buf_u8(&b, (uint8_t)VTM_Q4K_VERSION);
    buf_u8(&b, 0);
    buf_u64(&b, n);
    buf_u32(&b, n_blk);
    for (k = 0; k < n_blk; k++) {
        uint8_t block[VTM_Q6K_BLOCK_BYTES];
        for (i = 0; i < sizeof block; i++)
            block[i] = (uint8_t)((i * 53u + k * 17u + seed * 31u) & 0xFFu);
        put_u16(block + VTM_Q6K_BLOCK_BYTES - 2u,
                vtm_float_to_half(d)); // d
        buf_append(&b, block, sizeof block);
    }
    td->blob = b.p;
    td->blob_len = b.len;
}

// Прежние величины d: тестам 0175/0177 важны байты и совпадение путей, а не
// масштаб выхода.
static void model_add_q4k_bank(model_t *m, const char *name, int ndim,
                               const uint32_t *dims, uint32_t seed) {
    model_add_q4k_bank_d(m, name, ndim, dims, seed, 0.01f);
}

static void model_add_q6k_bank(model_t *m, const char *name, int ndim,
                               const uint32_t *dims, uint32_t seed) {
    model_add_q6k_bank_d(m, name, ndim, dims, seed, 0.02f);
}

// Один и тот же файл гоняется дважды: с VTM_MOE_CACHE_MB=0 (mmap + madvise,
// прежний путь) и с кэшем (pread). Выход MOE обязан совпасть побитово, а
// счётчики — доказать, что второй прогон целиком из кэша и что число
// выделений слотов не растёт (нет malloc на токен).
static void test_moe_pread_cache(void) {
    enum { NEXP = 4, HID = 256, INTER = 256, TOPK = 2 };
    model_t md;
    uint32_t rd[2] = {NEXP, HID};
    uint32_t xd[1] = {HID};
    uint32_t yd[1] = {HID};
    uint32_t bd[3] = {NEXP, INTER, HID};
    uint32_t dd[3] = {NEXP, HID, INTER};
    float router[NEXP * HID], xv[HID], yv[HID], base[HID];
    uint8_t *img = NULL;
    size_t size = 0;
    char path[80];
    long stamp = (long)time(NULL);
    FILE *f;
    vtm_t *m;
    const vtm_tensor *t;
    uint64_t hits = 0, misses = 0, allocs = 0;
    size_t bytes = 0;
    int i, bmax = 0, ymax = 0;

    for (i = 0; i < NEXP * HID; i++)
        router[i] = (float)(((i * 13) % 19) - 9) * 0.05f;
    for (i = 0; i < HID; i++) {
        xv[i] = (float)(((i * 7) % 13) - 6) * 0.1f;
        yv[i] = 0.0f;
    }

    model_init(&md);
    model_add_f32(&md, "router", 2, rd, router);
    model_add_f32(&md, "x", 1, xd, xv);
    model_add_f32(&md, "y", 1, yd, yv);
    model_add_q4k_bank(&md, "gate", 3, bd, 1);
    model_add_q4k_bank(&md, "up", 3, bd, 2);
    model_add_q6k_bank(&md, "down", 3, dd, 3);
    // Две одинаковые инструкции подряд: вторая должна попасть в кэш целиком.
    model_inst(&md, VTM_OP_MOE, 0, 2, 0, 1, 3, TOPK);
    model_inst(&md, VTM_OP_MOE, 0, 2, 0, 1, 3, TOPK);
    model_inst(&md, VTM_OP_HALT, 0, VTM_NO_TENSOR, VTM_NO_TENSOR,
               VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    model_build(&md, &img, &size);

    snprintf(path, sizeof path, "/tmp/vtm_moe_cache_%ld.vtm", stamp);
    f = fopen(path, "wb");
    CHECK(f != NULL);
    if (!f) {
        free(img);
        model_free(&md);
        return;
    }
    CHECK(fwrite(img, 1, size, f) == size);
    fclose(f);

    // 1. Эталон: кэш выключен, эксперты читаются из mmap.
    setenv("VTM_MOE_CACHE_MB", "0", 1);
    vtm_internal_moe_cache_reset();
    m = vtm_load(path);
    CHECK(m != NULL);
    if (m) {
        CHECK(run_ok(m));
        vtm_internal_moe_cache_stats(&hits, &misses, &bytes, &allocs);
        CHECK(hits == 0 && misses == 0 && bytes == 0 && allocs == 0);
        t = vtm_tensor_by_name(m, "y");
        CHECK(t != NULL);
        if (t) for (i = 0; i < HID; i++) base[i] = t->f32[i];
        vtm_free(m);
    }

    // 2. Тот же файл через pread-кэш (8 МиБ хватает на 6 слотов с запасом).
    // VTM_MOE_IO_THREADS=0 — проверяем именно последовательный путь: с
    // prefetch-потоками «кто прочитал ключ первым» зависит от планировщика, и
    // точные hits/misses стали бы плавающими. Prefetch проверяется отдельным
    // тестом ниже, где сравнивается выход, а не счётчики.
    setenv("VTM_MOE_IO_THREADS", "0", 1);
    setenv("VTM_MOE_CACHE_MB", "8", 1);
    vtm_internal_moe_cache_reset();
    m = vtm_load(path);
    CHECK(m != NULL);
    if (m) {
        CHECK(run_ok(m));
        vtm_internal_moe_cache_stats(&hits, &misses, &bytes, &allocs);
        CHECK(misses == 6);  // 2 эксперта × 3 банка
        CHECK(hits == 6);    // вторая инструкция целиком из кэша
        CHECK(allocs == 6);  // по слоту на запись, без копий в mmap
        CHECK(bytes == 4u * 36864u + 2u * 57344u); // слоты выровнены
        t = vtm_tensor_by_name(m, "y");
        CHECK(t != NULL);
        if (t) CHECK(memcmp(base, t->f32, sizeof base) == 0);

        // 3. Повторный прогон: все обращения — попадания, выделений больше нет.
        if (t) for (i = 0; i < HID; i++) t->f32[i] = 0.0f;
        CHECK(run_ok(m));
        vtm_internal_moe_cache_stats(&hits, &misses, &bytes, &allocs);
        CHECK(hits == 18);
        CHECK(misses == 6);
        CHECK(allocs == 6);
        t = vtm_tensor_by_name(m, "y");
        CHECK(t != NULL);
        if (t) {
            CHECK(memcmp(base, t->f32, sizeof base) == 0);
            for (i = 1; i < HID; i++)
                if (base[i] > base[bmax]) bmax = i;
            for (i = 1; i < HID; i++)
                if (t->f32[i] > t->f32[ymax]) ymax = i;
            CHECK(ymax == bmax); // argmax выхода MOE совпал с mmap-путём
        }
        vtm_free(m);
    }

    remove(path);
    unsetenv("VTM_MOE_CACHE_MB");
    vtm_internal_moe_cache_reset();
    free(img);
    model_free(&md);
}

// Prefetch экспертов в потоках (задача 0177). Проверяем не счётчики попаданий
// (кто успел первым — дело планировщика), а то, что от этого ничего не
// меняется: выход MOE побитово тот же, что у mmap-пути, слоты не выделяются
// заново, а задания в очередь действительно ставятся.
//
// Инвариант счётчиков, который от гонки не зависит: каждое обращение главного
// потока — либо hit, либо miss, поэтому hits + misses == 6 ключей × 2
// инструкции = 12, сколько бы из них ни успел принести prefetch.
static void test_moe_prefetch(void) {
    enum { NEXP = 4, HID = 256, INTER = 256, TOPK = 2 };
    model_t md;
    uint32_t rd[2] = {NEXP, HID};
    uint32_t xd[1] = {HID};
    uint32_t yd[1] = {HID};
    uint32_t bd[3] = {NEXP, INTER, HID};
    uint32_t dd[3] = {NEXP, HID, INTER};
    float router[NEXP * HID], xv[HID], yv[HID], base[HID];
    uint8_t *img = NULL;
    size_t size = 0;
    char path[80];
    long stamp = (long)time(NULL);
    FILE *f;
    vtm_t *m;
    const vtm_tensor *t;
    uint64_t hits = 0, misses = 0, allocs = 0, jobs = 0, waits = 0, useful = 0;
    size_t bytes = 0;
    int threads = 0, i;

    for (i = 0; i < NEXP * HID; i++)
        router[i] = (float)(((i * 11) % 17) - 8) * 0.05f;
    for (i = 0; i < HID; i++) {
        xv[i] = (float)(((i * 5) % 11) - 5) * 0.1f;
        yv[i] = 0.0f;
    }

    model_init(&md);
    model_add_f32(&md, "router", 2, rd, router);
    model_add_f32(&md, "x", 1, xd, xv);
    model_add_f32(&md, "y", 1, yd, yv);
    model_add_q4k_bank(&md, "gate", 3, bd, 4);
    model_add_q4k_bank(&md, "up", 3, bd, 5);
    model_add_q6k_bank(&md, "down", 3, dd, 6);
    model_inst(&md, VTM_OP_MOE, 0, 2, 0, 1, 3, TOPK);
    model_inst(&md, VTM_OP_MOE, 0, 2, 0, 1, 3, TOPK);
    model_inst(&md, VTM_OP_HALT, 0, VTM_NO_TENSOR, VTM_NO_TENSOR,
               VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    model_build(&md, &img, &size);

    snprintf(path, sizeof path, "/tmp/vtm_moe_pf_%ld.vtm", stamp);
    f = fopen(path, "wb");
    CHECK(f != NULL);
    if (!f) {
        free(img);
        model_free(&md);
        return;
    }
    CHECK(fwrite(img, 1, size, f) == size);
    fclose(f);

    // 1. Эталон: кэш выключен — эксперты из mmap.
    setenv("VTM_MOE_CACHE_MB", "0", 1);
    vtm_internal_moe_cache_reset();
    m = vtm_load(path);
    CHECK(m != NULL);
    if (m) {
        CHECK(run_ok(m));
        t = vtm_tensor_by_name(m, "y");
        CHECK(t != NULL);
        if (t) for (i = 0; i < HID; i++) base[i] = t->f32[i];
        vtm_free(m);
    }

    // 2. Кэш + два I/O-потока.
    setenv("VTM_MOE_CACHE_MB", "8", 1);
    setenv("VTM_MOE_IO_THREADS", "2", 1);
    vtm_internal_moe_cache_reset();
    m = vtm_load(path);
    CHECK(m != NULL);
    if (m) {
        CHECK(run_ok(m));
        vtm_internal_moe_cache_stats(&hits, &misses, &bytes, &allocs);
        vtm_internal_moe_prefetch_stats(&jobs, &waits, &useful, &threads);
        CHECK(threads == 2);
        CHECK(jobs >= 6);          // 6 ключей на инструкцию
        CHECK(useful <= waits);    // «ждали» не может быть меньше успешных
        CHECK(hits + misses == 12); // 2 инструкции × 6 ключей
        CHECK(allocs == 6);
        t = vtm_tensor_by_name(m, "y");
        CHECK(t != NULL);
        if (t) CHECK(memcmp(base, t->f32, sizeof base) == 0);

        // 3. Повтор: всё из кэша, новых слотов нет.
        if (t) for (i = 0; i < HID; i++) t->f32[i] = 0.0f;
        CHECK(run_ok(m));
        vtm_internal_moe_cache_stats(&hits, &misses, &bytes, &allocs);
        vtm_internal_moe_prefetch_stats(&jobs, &waits, &useful, &threads);
        CHECK(hits + misses == 24); // 4 инструкции × 6 ключей за два прогона
        CHECK(allocs == 6);
        t = vtm_tensor_by_name(m, "y");
        CHECK(t != NULL);
        if (t) CHECK(memcmp(base, t->f32, sizeof base) == 0);
        vtm_free(m);
    }

    remove(path);
    unsetenv("VTM_MOE_CACHE_MB");
    unsetenv("VTM_MOE_IO_THREADS");
    vtm_internal_moe_cache_reset();
    free(img);
    model_free(&md);
}

// --- MoE: батчевый prefill (задача 0179) -----------------------------------

// Крошечная MoE-модель той же формы, что у 30B: EMB (x из id токена) ->
// MOE (top-3 из 16 экспертов, Q4_K gate/up + Q6_K down) -> SAMP (argmax).
// Разные id дают разные x, поэтому наборы экспертов у токенов пачки разные и
// пересекаются — ровно то, что группирует батчевый путь.
enum { MB_NEXP = 16, MB_HID = 256, MB_INTER = 256, MB_TOPK = 3, MB_VOCAB = 16 };

static void moe_batch_image(uint8_t **img, size_t *size) {
    model_t md;
    uint32_t rd[2] = {MB_NEXP, MB_HID}, xd[1] = {MB_HID}, yd[1] = {MB_HID};
    uint32_t bd[3] = {MB_NEXP, MB_INTER, MB_HID};
    uint32_t dd[3] = {MB_NEXP, MB_HID, MB_INTER};
    uint32_t ed[2] = {MB_VOCAB, MB_HID}, td[1] = {1};
    float router[MB_NEXP * MB_HID], emb[MB_VOCAB * MB_HID], zero[MB_HID];
    int i;

    for (i = 0; i < MB_NEXP * MB_HID; i++)
        router[i] = (float)(((i * 13) % 19) - 9) * 0.05f;
    for (i = 0; i < MB_VOCAB * MB_HID; i++)
        emb[i] = (float)(((i * 29 + 7) % 23) - 11) * 0.09f;
    for (i = 0; i < MB_HID; i++) zero[i] = 0.0f;

    model_init(&md);
    (void)yd;
    model_init(&md);
    model_add_f32(&md, "router", 2, rd, router);
    model_add_f32(&md, "x", 1, xd, zero);
    // d на два порядка меньше, чем у тестов 0175/0177: у 30B веса обычного
    // масштаба, и выход MOE ~0,05. С «громкими» весами (d=0,01) слагаемые
    // ~1e6 гасят друг друга, и f32-ядро пачки расходится с double-путём на
    // 1e-2 — это свойство синтетики, а не батчевого кода.
    model_add_q4k_bank_d(&md, "gate", 3, bd, 1, 1e-5f);
    model_add_q4k_bank_d(&md, "up", 3, bd, 2, 1e-5f);
    model_add_q6k_bank_d(&md, "down", 3, dd, 3, 2e-5f);
    model_add_f32(&md, "emb", 2, ed, emb);
    model_add_f32(&md, "token", 1, td, zero);
    // dst у MOE — тот же x (резидуальный поток), как у 30B: EMB переписывает
    // его каждый проход, поэтому состояние не копится от токена к токену.
    model_inst(&md, VTM_OP_EMB, VTM_FLAG_EMB_INPUT, 1, 5, VTM_NO_TENSOR,
               VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_MOE, 0, 1, 0, 1, 2, MB_TOPK);
    model_inst(&md, VTM_OP_SAMP, 0, 6, 1, VTM_NO_TENSOR, VTM_NO_TENSOR,
               VTM_SAMP_ARGMAX);
    model_inst(&md, VTM_OP_HALT, 0, VTM_NO_TENSOR, VTM_NO_TENSOR,
               VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    model_set_meta(&md, "{\"temperature\": 0, \"rms_norm_eps\": 1e-06}");
    model_build(&md, img, size);
    model_free(&md);
}

static vtm_t *moe_batch_model(void) {
    uint8_t *img = NULL;
    size_t size = 0;
    vtm_t *m;

    moe_batch_image(&img, &size);
    m = load_ok(img, size);
    free(img);
    return m;
}

// Объединение top-k по пачке — независимым расчётом в тесте, чтобы проверить
// не «сколько получилось», а «каждого эксперта прочитали один раз».
// Роутер: logits[e] = router[e]·x в double; top-k выбирает строго старшего,
// перебором по возрастанию (как moe_route), softmax для порядка не нужен.
static int moe_batch_union(const int32_t *ids, int n) {
    int seen[MB_NEXP];
    int i, e, k, j, ne = 0;
    double *lg = (double *)calloc(MB_NEXP, sizeof *lg);

    memset(seen, 0, sizeof seen);
    for (i = 0; i < n; i++) {
        int token = (int)(ids[i] % MB_VOCAB);
        int chosen[MB_TOPK];
        for (e = 0; e < MB_NEXP; e++) {
            double dot = 0.0;
            for (j = 0; j < MB_HID; j++) {
                double r = (double)(((e * MB_HID + j) * 13 % 19) - 9) * 0.05;
                double x = (double)(((token * MB_HID + j) * 29 + 7) % 23 - 11)
                         * 0.09;
                dot += r * x;
            }
            lg[e] = dot;
        }
        for (k = 0; k < MB_TOPK; k++) {
            int best = -1, used;
            double bp = -1e300;
            for (e = 0; e < MB_NEXP; e++) {
                used = 0;
                for (j = 0; j < k; j++)
                    if (chosen[j] == e) { used = 1; break; }
                if (used) continue;
                if (lg[e] > bp) { bp = lg[e]; best = e; }
            }
            chosen[k] = best < 0 ? 0 : best;
        }
        for (k = 0; k < MB_TOPK; k++) seen[chosen[k]] = 1;
    }
    for (e = 0; e < MB_NEXP; e++) ne += seen[e];
    free(lg);
    return ne;
}

static void test_moe_batch(void) {
    int32_t ids[64];
    vtm_t *want, *got;
    int i;

    printf("  -- MoE: батчевый prefill (0179)\n");
    for (i = 0; i < 64; i++) ids[i] = (int32_t)((i * 5 + 1) % MB_VOCAB);

    // 1. Пачка 12 == 12 прогонов по токену. Здесь явно включено батчевое
    //    ядро kind 6/7 (VTM_MOE_BATCH_KERNEL=1): оно считает ту же сумму в
    //    f32-порядке, поэтому сравнение с допуском 1e-5, а выбранный токен —
    //    точно (расхождение токена это 1, а не 1e-5). Умолчание (поштучные
    //    ядра в группе) проверяется побитово в п. 3b.
    {
        want = moe_batch_model();
        got = moe_batch_model();
        CHECK(want != NULL && got != NULL);
        if (want && got) {
            setenv("VTM_MOE_BATCH_KERNEL", "1", 1);
            cache_run_plain(want, ids, 12);
            vtm_set_position(got, 0);
            CHECK(vtm_run_batch(got, ids, 12) == VTM_OK);
            CHECK(vtm_position(got) == 11);
            batch_check_same(want, got, "MoE пачка 12 (батчевое ядро)", 1e-5);
            unsetenv("VTM_MOE_BATCH_KERNEL");
            vtm_free(want);
            vtm_free(got);
        }
    }

    // 2. Пачка длиннее прежнего потолка 32: у MoE-графа потолок свой
    //    (VTM_MOE_BATCH_TOKENS), иначе 77-токенный промпт на 30B резался бы на
    //    куски и читал одних и тех же экспертов заново.
    {
        want = moe_batch_model();
        got = moe_batch_model();
        CHECK(want != NULL && got != NULL);
        if (want && got) {
            cache_run_plain(want, ids, 40);
            vtm_set_position(got, 0);
            CHECK(vtm_run_batch(got, ids, 40) == VTM_OK);
            batch_check_same(want, got, "MoE пачка 40", 1e-5);
            vtm_free(want);
            vtm_free(got);
        }
    }

    // 3. Один токен; и VTM_MOE_BATCH=0 — прежний поштучный путь целиком.
    {
        vtm_t *a = moe_batch_model();
        vtm_t *b = moe_batch_model();
        CHECK(a != NULL && b != NULL);
        if (a && b) {
            cache_run_plain(a, ids, 1);
            vtm_set_position(b, 0);
            CHECK(vtm_run_batch(b, ids, 1) == VTM_OK);
            batch_check_same(a, b, "MoE пачка 1", 0.0);
            vtm_free(a);
            vtm_free(b);
        }
        setenv("VTM_MOE_BATCH", "0", 1);
        a = moe_batch_model();
        b = moe_batch_model();
        CHECK(a != NULL && b != NULL);
        if (a && b) {
            cache_run_plain(a, ids, 8);
            vtm_set_position(b, 0);
            CHECK(vtm_run_batch(b, ids, 8) == VTM_OK);
            batch_check_same(a, b, "MoE VTM_MOE_BATCH=0", 0.0);
            vtm_free(a);
            vtm_free(b);
        }
        unsetenv("VTM_MOE_BATCH");
    }

    // 3b. VTM_MOE_BATCH_KERNEL=0 — в группе считают поштучные ядра, поэтому
    //     выход MOE обязан совпасть с поштучным путём побитово (порядок
    //     сложений вкладов экспертов тот же: k по возрастанию). Это эталон
    //     группировки: если он расходится, виновата не арифметика ядер.
    {
        vtm_t *a = moe_batch_model();
        vtm_t *b = moe_batch_model();
        CHECK(a != NULL && b != NULL);
        if (a && b) {
            setenv("VTM_MOE_BATCH_KERNEL", "0", 1);
            cache_run_plain(a, ids, 12);
            vtm_set_position(b, 0);
            CHECK(vtm_run_batch(b, ids, 12) == VTM_OK);
            batch_check_same(a, b, "MoE пачка 12 (поштучные ядра)", 0.0);
            unsetenv("VTM_MOE_BATCH_KERNEL");
            vtm_free(a);
            vtm_free(b);
        }
    }

    // 4. Байты: батчевый путь читает каждого эксперта пачки один раз, а не по
    //    разу на каждый выбравший его токен. Счётчики pread детерминированы
    //    при VTM_MOE_IO_THREADS=0; кэш 1 МиБ меньше рабочего набора (16
    //    экспертов × 3 банка = 2 МиБ), поэтому поштучный путь перечитывает.
    {
        uint8_t *img = NULL;
        size_t size = 0;
        char path[80];
        long stamp = (long)time(NULL);
        FILE *f;
        uint64_t hb = 0, mb = 0, ht = 0, mt = 0;
        size_t by = 0, by2 = 0;
        int uni;

        moe_batch_image(&img, &size);
        snprintf(path, sizeof path, "/tmp/vtm_moe_batch_%ld.vtm", stamp);
        f = fopen(path, "wb");
        CHECK(f != NULL);
        if (f) {
            CHECK(fwrite(img, 1, size, f) == size);
            fclose(f);

            setenv("VTM_MOE_IO_THREADS", "0", 1);
            setenv("VTM_MOE_CACHE_MB", "1", 1);

            // Пачка: 12 токенов одной инструкцией.
            vtm_internal_moe_cache_reset();
            want = vtm_load(path);
            CHECK(want != NULL);
            if (want) {
                vtm_set_position(want, 0);
                CHECK(vtm_run_batch(want, ids, 12) == VTM_OK);
                vtm_internal_moe_cache_stats(&hb, &mb, &by, NULL);
                vtm_free(want);
            }
            // Поштучно: те же 12 токенов по одному.
            vtm_internal_moe_cache_reset();
            got = vtm_load(path);
            CHECK(got != NULL);
            if (got) {
                cache_run_plain(got, ids, 12);
                vtm_internal_moe_cache_stats(&ht, &mt, &by2, NULL);
                vtm_free(got);
            }
            uni = moe_batch_union(ids, 12);
            CHECK(uni > 0 && uni < 12 * MB_TOPK); // пересечение есть
            // Батчевый путь: ровно 3 промаха на уникального эксперта и ни
            // одного повторного чтения.
            CHECK(mb == (uint64_t)3 * (uint64_t)uni);
            CHECK(hb == 0);
            // Поштучный: тех же экспертов читают столько раз, сколько токенов
            // их выбрали, поэтому промахов заведомо больше.
            CHECK(mt > mb);
            printf("    экспертов %d: пачка — %llu чтений, поштучно — %llu\n",
                   uni, (unsigned long long)mb, (unsigned long long)mt);

            remove(path);
            unsetenv("VTM_MOE_CACHE_MB");
            unsetenv("VTM_MOE_IO_THREADS");
            vtm_internal_moe_cache_reset();
        }
        free(img);
    }

    // 5. Пачка короче промпта (0181): несколько пачек по 8 и бюджет auto.
    //    Между пачками машина переносит только live-in активации, остальные
    //    пересчитываются — состояние обязано остаться побитово тем же, что у
    //    40 прогонов по токену (это и есть проверка упаковки слота по
    //    интервалам жизни и возврата тензоров в момент последнего касания).
    {
        vtm_t *a = moe_batch_model();
        vtm_t *b = moe_batch_model();
        CHECK(a != NULL && b != NULL);
        if (a && b) {
            setenv("VTM_MOE_BATCH_MB", "auto", 1);
            setenv("VTM_MOE_BATCH_TOKENS", "8", 1);
            cache_run_plain(a, ids, 40);
            vtm_set_position(b, 0);
            CHECK(vtm_run_batch(b, ids, 40) == VTM_OK);
            batch_check_same(a, b, "MoE 40 пачками по 8 (бюджет auto)", 0.0);
            unsetenv("VTM_MOE_BATCH_MB");
            unsetenv("VTM_MOE_BATCH_TOKENS");
            vtm_free(a);
            vtm_free(b);
        }
    }

    // 6. Бюджет auto принимается и на граничных значениях потолка: пачка 40
    //    при явном потолке 40 — одна пачка, результат тот же.
    {
        vtm_t *a = moe_batch_model();
        vtm_t *b = moe_batch_model();
        CHECK(a != NULL && b != NULL);
        if (a && b) {
            setenv("VTM_MOE_BATCH_MB", "auto", 1);
            setenv("VTM_MOE_BATCH_TOKENS", "40", 1);
            cache_run_plain(a, ids, 40);
            vtm_set_position(b, 0);
            CHECK(vtm_run_batch(b, ids, 40) == VTM_OK);
            batch_check_same(a, b, "MoE 40 одной пачкой (бюджет auto)", 0.0);
            unsetenv("VTM_MOE_BATCH_MB");
            unsetenv("VTM_MOE_BATCH_TOKENS");
            vtm_free(a);
            vtm_free(b);
        }
    }

    printf("  -- MoE: батчевый prefill — токены и выход совпали\n");
}


// --- CONV4/GDELTA (задача 0176) --------------------------------------------
//
// Паритет с Python-эталоном проверяется здесь на ручных числах из
// tests/test_gated_delta.py, а случайные тензоры — в
// tests/vtm/test_gated_delta_parity.py (крошечная C-проба против vtm_ref).
// Состояние — f32, поэтому сверяемся с порогом 1e-6, а точные целые числа
// (ручной CONV4) — побитово.

static uint64_t g_gd_trace_count[256];

static int gd_trace_cb(void *user, const vtm_trace_rec *rec) {
    (void)user;
    if (rec) g_gd_trace_count[rec->opcode]++;
    return 0;
}

// Модель `EMB → CONV4` с width = 2 (ручной пример из docs/qwen35.md).
static vtm_t *conv4_load(const float *w, const float *x_table,
                         int inplace) {
    static const float def_w[8] = {
        1.0f, 10.0f, 100.0f, 1000.0f, 2.0f, 20.0f, 200.0f, 2000.0f
    };
    static const float def_x[8] = {
        1.0f, 1.0f, 2.0f, 1.0f, 3.0f, 1.0f, 4.0f, 1.0f
    };
    uint32_t xd[1] = {2}, wd[2] = {2, 4}, sd[2] = {3, 2};
    uint32_t yd[1] = {2}, td[2] = {4, 2};
    float zero2[2] = {0.0f, 0.0f};
    uint8_t *img = NULL;
    size_t size = 0;
    model_t md;
    vtm_t *m;

    model_init(&md);
    model_add_f32(&md, "x", 1, xd, zero2);
    model_add_f32(&md, "w", 2, wd, w ? w : def_w);
    model_add_state(&md, "conv.state", 2, sd);
    model_add_f32(&md, "y", 1, yd, zero2);
    model_add_f32(&md, "x.table", 2, td, x_table ? x_table : def_x);
    model_inst(&md, VTM_OP_EMB, VTM_FLAG_EMB_INPUT, 0, 4,
               VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
    model_inst(&md, VTM_OP_CONV4, 0, inplace ? 0 : 3, 0, 1, 2, 0);
    model_build(&md, &img, &size);
    m = load_ok(img, size);
    free(img);
    model_free(&md);
    return m;
}

// Модель `EMB×6 → GDELTA`. Банк — ровно шесть подряд идущих тензоров:
// состояние, v, beta, g, z, norm_w (индексы 2..7); q/k — 0/1, y — 8.
// Входы задаются таблицами по токенам, поэтому одна машина считает
// последовательность шагов и состояние переживает vtm_run().
static vtm_t *gdelta_load(int nk, int nv, int dk, int dv,
                          const float *qtab, const float *ktab,
                          const float *vtab, const float *btab,
                          const float *gtab, const float *ztab,
                          const float *nw, int ntok, int eps) {
    uint32_t qd[2], vd[2], bd[1], zd[2], nd[1], yd[2], tq[2], tv[2],
             tb[2], tz[2];
    size_t nq = (size_t)nk * (size_t)dk, nv_el = (size_t)nv * (size_t)dv;
    float *zero_q = (float *)xmalloc(nq * sizeof(float));
    float *zero_v = (float *)xmalloc(nv_el * sizeof(float));
    uint8_t *img = NULL;
    size_t size = 0;
    model_t md;
    vtm_t *m;

    memset(zero_q, 0, nq * sizeof(float));
    memset(zero_v, 0, nv_el * sizeof(float));
    qd[0] = (uint32_t)nk; qd[1] = (uint32_t)dk;
    vd[0] = (uint32_t)nv; vd[1] = (uint32_t)dv;
    bd[0] = (uint32_t)nv;
    zd[0] = (uint32_t)nv; zd[1] = (uint32_t)dv;
    nd[0] = (uint32_t)dv;
    yd[0] = (uint32_t)nv; yd[1] = (uint32_t)dv;
    tq[0] = (uint32_t)ntok; tq[1] = (uint32_t)(nk * dk);
    tv[0] = (uint32_t)ntok; tv[1] = (uint32_t)(nv * dv);
    tb[0] = (uint32_t)ntok; tb[1] = (uint32_t)nv;
    tz[0] = (uint32_t)ntok; tz[1] = (uint32_t)(nv * dv);

    model_init(&md);
    model_add_f32(&md, "q", 2, qd, zero_q);
    model_add_f32(&md, "k", 2, qd, zero_q);
    model_add_state(&md, "gd.state", 3, (uint32_t[3]){nv, dk, dv});
    model_add_f32(&md, "v", 2, vd, zero_v);
    model_add_f32(&md, "beta", 1, bd, zero_v);
    model_add_f32(&md, "g", 1, bd, zero_v);
    model_add_f32(&md, "z", 2, zd, zero_v);
    model_add_f32(&md, "norm_w", 1, nd, nw);
    model_add_f32(&md, "y", 2, yd, zero_v);
    model_add_f32(&md, "q.table", 2, tq, qtab);
    model_add_f32(&md, "k.table", 2, tq, ktab);
    model_add_f32(&md, "v.table", 2, tv, vtab);
    model_add_f32(&md, "beta.table", 2, tb, btab);
    model_add_f32(&md, "g.table", 2, tb, gtab);
    model_add_f32(&md, "z.table", 2, tz, ztab);
    model_inst(&md, VTM_OP_EMB, VTM_FLAG_EMB_INPUT, 0, 9, 0, 0, 0);
    model_inst(&md, VTM_OP_EMB, VTM_FLAG_EMB_INPUT, 1, 10, 0, 0, 0);
    model_inst(&md, VTM_OP_EMB, VTM_FLAG_EMB_INPUT, 3, 11, 0, 0, 0);
    model_inst(&md, VTM_OP_EMB, VTM_FLAG_EMB_INPUT, 4, 12, 0, 0, 0);
    model_inst(&md, VTM_OP_EMB, VTM_FLAG_EMB_INPUT, 5, 13, 0, 0, 0);
    model_inst(&md, VTM_OP_EMB, VTM_FLAG_EMB_INPUT, 6, 14, 0, 0, 0);
    model_inst(&md, VTM_OP_GDELTA, 0, 8, 0, 1, 2, eps);
    model_build(&md, &img, &size);
    m = load_ok(img, size);
    free(img);
    model_free(&md);
    free(zero_q);
    free(zero_v);
    return m;
}

static const float GD_Q3[] = {1,0, 1,0, 1,0};
static const float GD_K3[] = {1,0, 1,0, 1,0};
static const float GD_V3[] = {2,4, 4,2, 1,1};
static const float GD_B3[] = {0, 0, 0};
static const float GD_G3[] = {0, 0, 0};
static const float GD_Z3[] = {1,1, 1,1, 1,1};
static const float GD_N2[] = {1,1};

static void test_conv4(void) {
    static const float want_y[4][2] = {
        {1000.0f, 2000.0f}, {2100.0f, 2200.0f},
        {3210.0f, 2220.0f}, {4321.0f, 2222.0f}
    };
    static const float want_state[4][6] = {
        {0,0,0,0,1,1}, {0,0,1,1,2,1}, {1,1,2,1,3,1}, {2,1,3,1,4,1}
    };
    vtm_t *m = conv4_load(NULL, NULL, 0);
    const vtm_tensor *state;
    int step, i;

    CHECK(m != NULL);
    if (!m) return;
    state = vtm_tensor_by_name(m, "conv.state");
    CHECK(state != NULL);
    if (state) {
        CHECK((state->flags & VTM_TENSOR_FLAG_STATE) != 0);
        CHECK(state->offset == TEST_KV_NO_OFFSET);
        CHECK(state->f32 != NULL);
        CHECK(state->n_elements == 6);
        for (i = 0; i < 6; i++) CHECK(state->f32[i] == 0.0f);
    }
    for (step = 0; step < 4; step++) {
        const vtm_tensor *y;
        vtm_set_position(m, step);
        vtm_set_token(m, step);
        CHECK(vtm_run(m, NULL, NULL) == VTM_OK);
        y = vtm_tensor_by_name(m, "y");
        state = vtm_tensor_by_name(m, "conv.state");
        CHECK(y != NULL && state != NULL);
        if (y) {
            CHECK(y->f32[0] == want_y[step][0]);
            CHECK(y->f32[1] == want_y[step][1]);
        }
        if (state)
            for (i = 0; i < 6; i++)
                CHECK(state->f32[i] == want_state[step][i]);
    }
    // Состояние не течёт между «сессиями»: vtm_reset() обнуляет его, и
    // следующий прогон с того же токена даёт те же числа, что первый.
    vtm_reset(m);
    state = vtm_tensor_by_name(m, "conv.state");
    CHECK(state != NULL);
    if (state)
        for (i = 0; i < 6; i++) CHECK(state->f32[i] == 0.0f);
    vtm_set_token(m, 0);
    CHECK(vtm_run(m, NULL, NULL) == VTM_OK);
    {
        const vtm_tensor *y = vtm_tensor_by_name(m, "y");
        CHECK(y != NULL);
        if (y) {
            CHECK(y->f32[0] == 1000.0f);
            CHECK(y->f32[1] == 2000.0f);
        }
    }
    // In-place: dst совпадает с x. Свёртка обязана посчитать y по старым
    // state/x и сдвинуть состояние до записи y (тот же результат, что с
    // отдельным тензором y).
    {
        vtm_t *mi = conv4_load(NULL, NULL, 1);
        CHECK(mi != NULL);
        if (mi) {
            for (step = 0; step < 4; step++) {
                const vtm_tensor *x;
                vtm_set_position(mi, step);
                vtm_set_token(mi, step);
                CHECK(vtm_run(mi, NULL, NULL) == VTM_OK);
                x = vtm_tensor_by_name(mi, "x");
                CHECK(x != NULL);
                if (x) {
                    CHECK(x->f32[0] == want_y[step][0]);
                    CHECK(x->f32[1] == want_y[step][1]);
                }
            }
            vtm_free(mi);
        }
    }
    // --trace-summary считает новые опкоды через общий callback: проверяем,
    // что имена есть, а записи доходят (main.c печатает по opcode).
    memset(g_gd_trace_count, 0, sizeof g_gd_trace_count);
    vtm_reset(m);
    vtm_set_trace(m, gd_trace_cb, NULL);
    vtm_set_token(m, 0);
    CHECK(vtm_run(m, NULL, NULL) == VTM_OK);
    vtm_set_trace(m, NULL, NULL);
    CHECK(g_gd_trace_count[VTM_OP_EMB] == 1);
    CHECK(g_gd_trace_count[VTM_OP_CONV4] == 1);
    CHECK(strcmp(vtm_opcode_name(VTM_OP_CONV4), "CONV4") == 0);
    vtm_free(m);
    printf("  -- CONV4: ручные числа, состояние f32, reset и трасса\n");
}

static void test_gdelta_manual(void) {
    static const float want_y[3][2] = {
        {0.4623620423f, 0.9247240846f},
        {0.8073191449f, 0.6458553159f},
        {0.7849748684f, 0.6728356015f}
    };
    static const float want_state[3][4] = {
        {1,2,0,0}, {2.5f,2,0,0}, {1.75f,1.5f,0,0}
    };
    vtm_t *m = gdelta_load(1, 1, 2, 2, GD_Q3, GD_K3, GD_V3, GD_B3, GD_G3,
                           GD_Z3, GD_N2, 3, 0);
    const vtm_tensor *state;
    float last_y[2] = {0, 0};
    int step, i;

    CHECK(m != NULL);
    if (!m) return;
    state = vtm_tensor_by_name(m, "gd.state");
    CHECK(state != NULL);
    if (state) {
        CHECK((state->flags & VTM_TENSOR_FLAG_STATE) != 0);
        CHECK(state->offset == TEST_KV_NO_OFFSET);
        CHECK(state->ndim == 3 && state->dims[0] == 1 && state->dims[1] == 2
              && state->dims[2] == 2);
    }
    for (step = 0; step < 3; step++) {
        const vtm_tensor *y;
        vtm_set_position(m, step);
        vtm_set_token(m, step);
        CHECK(vtm_run(m, NULL, NULL) == VTM_OK);
        y = vtm_tensor_by_name(m, "y");
        state = vtm_tensor_by_name(m, "gd.state");
        CHECK(y != NULL && state != NULL);
        if (y) {
            CHECK_NEAR(y->f32[0], want_y[step][0], 1e-6);
            CHECK_NEAR(y->f32[1], want_y[step][1], 1e-6);
            last_y[0] = y->f32[0];
            last_y[1] = y->f32[1];
        }
        if (state)
            for (i = 0; i < 4; i++)
                CHECK_NEAR(state->f32[i], want_state[step][i], 1e-6);
    }
    // Батчевый prefill: состояние сквозное, поэтому GDELTA обязан считаться
    // по токенам в порядке пачки. Результат — тот же, что у трёх vtm_run().
    vtm_reset(m);
    {
        int32_t ids[3] = {0, 1, 2};
        CHECK(vtm_run_batch(m, ids, 3) == VTM_OK);
        state = vtm_tensor_by_name(m, "gd.state");
        {
            const vtm_tensor *y = vtm_tensor_by_name(m, "y");
            CHECK(y != NULL && state != NULL);
            if (y) {
                CHECK_NEAR(y->f32[0], last_y[0], 1e-6);
                CHECK_NEAR(y->f32[1], last_y[1], 1e-6);
            }
            if (state)
                for (i = 0; i < 4; i++)
                    CHECK_NEAR(state->f32[i], want_state[2][i], 1e-6);
        }
        CHECK(vtm_position(m) == 2);
    }
    // Кэш префикса (0081) не хранит состояние: для графа со STATE он обязан
    // быть выключен. Иначе попадание восстановило бы K/V, но не S, и хвост
    // считался бы от чужого состояния. Проверяем: длины кэша нет, а
    // vtm_run_cached() даёт те же y/S, что обычный полный прогон.
    {
        int32_t ids[3] = {0, 1, 2};
        const vtm_tensor *y;

        vtm_reset(m);
        CHECK(vtm_cache_prefix(m, ids, 3) == VTM_OK);
        CHECK(vtm_cache_prefix_len(m) == 0);
        CHECK(vtm_run_cached(m, ids, 3) == VTM_OK);
        y = vtm_tensor_by_name(m, "y");
        state = vtm_tensor_by_name(m, "gd.state");
        CHECK(y != NULL && state != NULL);
        if (y) {
            CHECK_NEAR(y->f32[0], last_y[0], 1e-6);
            CHECK_NEAR(y->f32[1], last_y[1], 1e-6);
        }
        if (state)
            for (i = 0; i < 4; i++)
                CHECK_NEAR(state->f32[i], want_state[2][i], 1e-6);
    }
    // Сброс и трасса (main.c --trace-summary печатает эти опкоды по имени).
    vtm_reset(m);
    state = vtm_tensor_by_name(m, "gd.state");
    CHECK(state != NULL);
    if (state)
        for (i = 0; i < 4; i++) CHECK(state->f32[i] == 0.0f);
    memset(g_gd_trace_count, 0, sizeof g_gd_trace_count);
    vtm_set_trace(m, gd_trace_cb, NULL);
    vtm_set_token(m, 0);
    CHECK(vtm_run(m, NULL, NULL) == VTM_OK);
    vtm_set_trace(m, NULL, NULL);
    CHECK(g_gd_trace_count[VTM_OP_EMB] == 6);
    CHECK(g_gd_trace_count[VTM_OP_GDELTA] == 1);
    CHECK(strcmp(vtm_opcode_name(VTM_OP_GDELTA), "GDELTA") == 0);
    vtm_free(m);
    printf("  -- GDELTA: ручные числа, пачка, reset и трасса\n");
}

static void test_gdelta_properties(void) {
    // 1. Две value-головы на одну key-голову: kh = h / 2, у каждой свой
    //    участок состояния [Hv, Dk, Dv].
    {
        static const float q1[] = {1,0};
        static const float k1[] = {1,0};
        static const float v1[] = {2,0, 0,1};
        static const float b1[] = {0,0};
        static const float g1[] = {0,0};
        static const float z1[] = {1,1, 1,1};
        static const float nw1[] = {1,1};
        vtm_t *m = gdelta_load(1, 2, 2, 2, q1, k1, v1, b1, g1, z1, nw1, 1, 0);
        const vtm_tensor *state, *y;
        double root_two = sqrt(2.0), silu1 = 1.0 / (1.0 + exp(-1.0));

        CHECK(m != NULL);
        if (!m) return;
        vtm_set_token(m, 0);
        CHECK(vtm_run(m, NULL, NULL) == VTM_OK);
        state = vtm_tensor_by_name(m, "gd.state");
        y = vtm_tensor_by_name(m, "y");
        CHECK(state != NULL && y != NULL);
        if (state) {
            static const float want[8] = {1,0,0,0, 0,0.5f,0,0};
            int i;
            for (i = 0; i < 8; i++) CHECK_NEAR(state->f32[i], want[i], 1e-6);
        }
        if (y) {
            CHECK_NEAR(y->f32[0], root_two * silu1, 1e-6);
            CHECK_NEAR(y->f32[1], 0.0, 1e-6);
            CHECK_NEAR(y->f32[2], 0.0, 1e-6);
            CHECK_NEAR(y->f32[3], root_two * silu1, 1e-6);
        }
        vtm_free(m);
    }
    // 2. L2-нормировка: q ⊥ k → o = 0, а состояние всё равно обновляется.
    //    Состояние [[0,0],[1,2]] проверяет и ось: k̂ идёт во вторую строку.
    {
        static const float q1[] = {3,0};
        static const float k1[] = {0,5};
        static const float v1[] = {2,4};
        static const float b1[] = {0};
        static const float g1[] = {0};
        static const float z1[] = {1,1};
        vtm_t *m = gdelta_load(1, 1, 2, 2, q1, k1, v1, b1, g1, z1, GD_N2, 1, 0);
        const vtm_tensor *state, *y;
        int i;

        CHECK(m != NULL);
        if (!m) return;
        vtm_set_token(m, 0);
        CHECK(vtm_run(m, NULL, NULL) == VTM_OK);
        state = vtm_tensor_by_name(m, "gd.state");
        y = vtm_tensor_by_name(m, "y");
        CHECK(state != NULL && y != NULL);
        if (state) {
            static const float want[4] = {0,0,1,2};
            for (i = 0; i < 4; i++) CHECK_NEAR(state->f32[i], want[i], 1e-6);
        }
        if (y) {
            CHECK_NEAR(y->f32[0], 0.0, 1e-6);
            CHECK_NEAR(y->f32[1], 0.0, 1e-6);
        }
        vtm_free(m);
    }
    // 3. Затухание: k = v = 0, g = ln 0.5 → состояние ровно вдвое меньше.
    {
        static const float q2[] = {1,0, 0,0};
        static const float k2[] = {1,0, 0,0};
        static const float v2[] = {2,4, 0,0};
        static const float b2[] = {0,0};
        static const float g2[] = {0, (float)-0.6931471805599453};
        static const float z2[] = {1,1, 1,1};
        vtm_t *m = gdelta_load(1, 1, 2, 2, q2, k2, v2, b2, g2, z2, GD_N2, 2, 0);
        const vtm_tensor *state, *y;
        int i;

        CHECK(m != NULL);
        if (!m) return;
        vtm_set_token(m, 0);
        CHECK(vtm_run(m, NULL, NULL) == VTM_OK);
        vtm_set_position(m, 1);
        vtm_set_token(m, 1);
        CHECK(vtm_run(m, NULL, NULL) == VTM_OK);
        state = vtm_tensor_by_name(m, "gd.state");
        y = vtm_tensor_by_name(m, "y");
        CHECK(state != NULL && y != NULL);
        if (state) {
            static const float want[4] = {0.5f,1.0f,0,0};
            for (i = 0; i < 4; i++) CHECK_NEAR(state->f32[i], want[i], 1e-6);
        }
        if (y) {
            CHECK_NEAR(y->f32[0], 0.0, 1e-6);
            CHECK_NEAR(y->f32[1], 0.0, 1e-6);
        }
        vtm_free(m);
    }
    // 4. q = 0 при eps = 0: нулевая строка q̂ остаётся нулевой (а не NaN),
    //    выход нулевой, но состояние обновляется — «выход 0» не значит
    //    «шаг не сделан».
    {
        static const float q1[] = {0,0};
        static const float k1[] = {1,0};
        static const float v1[] = {2,4};
        static const float b1[] = {0};
        static const float g1[] = {0};
        static const float z1[] = {1,1};
        vtm_t *m = gdelta_load(1, 1, 2, 2, q1, k1, v1, b1, g1, z1, GD_N2, 1, 0);
        const vtm_tensor *state, *y;
        int i;

        CHECK(m != NULL);
        if (!m) return;
        vtm_set_token(m, 0);
        CHECK(vtm_run(m, NULL, NULL) == VTM_OK);
        state = vtm_tensor_by_name(m, "gd.state");
        y = vtm_tensor_by_name(m, "y");
        CHECK(state != NULL && y != NULL);
        if (state) {
            static const float want[4] = {1,2,0,0};
            for (i = 0; i < 4; i++) CHECK_NEAR(state->f32[i], want[i], 1e-6);
        }
        if (y) {
            CHECK_NEAR(y->f32[0], 0.0, 1e-6);
            CHECK_NEAR(y->f32[1], 0.0, 1e-6);
        }
        vtm_free(m);
    }
    printf("  -- GDELTA: две value-головы, L2-нормировка, затухание, q=0\n");
}

static void test_gated_delta_errors(void) {
    // 1. Обрезанный банк: c+5 за границей таблицы — отказ до чтения z/norm_w.
    {
        uint32_t qd[2] = {1, 2}, vd[2] = {1, 2}, bd[1] = {1};
        float qv[2] = {1, 0}, kv[2] = {1, 0}, vv[2] = {2, 4};
        float bv[1] = {0}, gv[1] = {0};
        uint8_t *img = NULL;
        size_t size = 0;
        model_t md;
        vtm_t *m;

        model_init(&md);
        model_add_f32(&md, "q", 2, qd, qv);
        model_add_f32(&md, "k", 2, qd, kv);
        model_add_state(&md, "state", 3, (uint32_t[3]){1, 2, 2});
        model_add_f32(&md, "v", 2, vd, vv);
        model_add_f32(&md, "beta", 1, bd, bv);
        model_add_f32(&md, "g", 1, bd, gv);
        model_add_f32(&md, "y", 2, vd, vv);
        model_inst(&md, VTM_OP_GDELTA, 0, 6, 0, 1, 2, 0);
        model_build(&md, &img, &size);
        m = load_ok(img, size);
        free(img);
        model_free(&md);
        CHECK(m != NULL);
        if (m) {
            vtm_status st = vtm_run(m, NULL, NULL);
            CHECK(st == VTM_E_TENSOR);
            CHECK(strstr(vtm_error_message(m), "GDELTA") != NULL);
            vtm_free(m);
        }
    }
    // 2. Отрицательный eps (imm) — отказ, а не тихая нормировка.
    {
        vtm_t *m = gdelta_load(1, 1, 2, 2, GD_Q3, GD_K3, GD_V3, GD_B3, GD_G3,
                               GD_Z3, GD_N2, 1, -1);
        CHECK(m != NULL);
        if (m) {
            vtm_status st;
            vtm_set_token(m, 0);
            st = vtm_run(m, NULL, NULL);
            CHECK(st == VTM_E_ARG);
            CHECK(strstr(vtm_error_message(m), "eps") != NULL);
            vtm_free(m);
        }
    }
    // 3. Бит STATE у тритов — это хранение тритов (биты 1..2), а не буфер:
    //    загрузчик обязан отказать, а не принять трит за состояние.
    {
        uint8_t *img = NULL;
        size_t size = 0;
        model_t md;
        vtm_t *m;
        int8_t trits[4] = {1, -1, 0, 1};
        float scales[1] = {1.0f};
        uint32_t wd[1] = {4};
        float vals[4] = {0, 0, 0, 0};

        model_init(&md);
        model_add_trit_scaled(&md, "w", 1, wd, trits, scales, 1,
                              (int)VTM_SCALE_BLOCK_128);
        md.tensors[0].flags = VTM_TENSOR_FLAG_STATE;
        model_add_f32(&md, "y", 1, wd, vals);
        model_inst(&md, VTM_OP_HALT, 0, VTM_NO_TENSOR, VTM_NO_TENSOR,
                   VTM_NO_TENSOR, VTM_NO_TENSOR, 0);
        model_build(&md, &img, &size);
        m = vtm_load_mem(img, size, NULL, NULL, 0);
        CHECK(m == NULL);
        CHECK(strstr(vtm_last_error(), "fp-тензор") != NULL);
        free(img);
        model_free(&md);
    }
    printf("  -- GDELTA: обрезанный банк, eps, бит STATE у тритов\n");
}

static void test_gated_delta(void) {
    test_conv4();
    test_gdelta_manual();
    test_gdelta_properties();
    test_gated_delta_errors();
}

// --- main ------------------------------------------------------------------

int main(void) {
    test_trit_codec();
    test_load_and_tmv();
    test_v03_container();
    test_tmv_blocks();
    test_tmv_scale128();
    test_raw_trit_layout();
    test_bad_files();
    test_host_endian_guard();
    test_file_load();
    test_rmsnorm();
    test_softmax();
    test_rope();
    test_meta_rope();
    test_meta_fp16_emb_tied();
    test_meta_broken();
    test_emb();
    test_mm();
    test_int8_packed();
    test_elementwise();
    test_attn_plain();
    test_attn_causal();
    test_kvw_attn_cache();
    test_attn_tiled_vs_reference();
    test_attn_window_rotation();
    test_samp();
    test_samp_strategies();
    test_f16_bf16();
    test_q4_load();
    test_q4_asym_load();
    test_q4_bad();
    test_q4_unknown_kind_rejected();
    test_large_dims_u32();
    test_cli_text();
    test_cli_chat();
    test_attn_imm_overflow();
    test_trace_hook();
    test_cli_stream();
    test_cli_limits();
    test_mmap_file_load();
    test_mmap_fallback();
    test_mmap_truncated();
    test_mmap_f16_bf16();
    test_mmap_trit_f16_scales();
    test_mmap_raw_trit();
    test_samp_partial_select();
    test_samp_buffer_reuse();
    test_samp_stale_stamps();
    test_cache_prefix();
    test_cache_prefix_ring();
    test_cache_batch();
    test_run_batch();
    test_run_batch_cancel();
    test_gated_delta();
    test_moe_pread_cache();
    test_moe_prefetch();
    test_moe_batch();
    bench_cache_prefix();
    test_samp_bench_150k();

    if (g_fail) {
        fprintf(stderr, "\nПРОВАЛЕНО проверок: %d\n", g_fail);
        return 1;
    }
    printf("все тесты VTM: ok\n");
    return 0;
}

#endif // !VTM_HOST_BIG_ENDIAN
