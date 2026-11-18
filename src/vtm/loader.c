// Загрузчик .vtm: заголовок 32 байта, таблица тензоров по 52 байта,
// секция данных с выравниванием на 32 и программа в конце файла.
//
// Формат v0.2 (ADR 0011): dims — u32×4. Файлы v0.1 отвергаются: ценных
// артефактов v0.1 нет, синтетика пересобирается.
//
// Все проверки границ делаются до выделения памяти и до исполнения. Кривой
// файл обязан вернуть NULL и внятную причину, а не уронить процесс.
#include "vtm_internal.h"
#include "vtm_backend.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char g_last_error[256] = "нет ошибки";

const char *vtm_last_error(void) {
    return g_last_error;
}

static void set_global_error(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_last_error, sizeof g_last_error, fmt, ap);
    va_end(ap);
}

void vtm_set_error(vtm_t *m, vtm_status st, const char *fmt, ...) {
    char buf[256];
    va_list ap;
    (void)st;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    snprintf(g_last_error, sizeof g_last_error, "%s", buf);
    if (m) snprintf(m->error, sizeof m->error, "%s", buf);
}

const char *vtm_strerror(vtm_status st) {
    switch (st) {
    case VTM_OK:           return "ok";
    case VTM_E_ARG:        return "неверный аргумент";
    case VTM_E_IO:         return "ошибка ввода-вывода";
    case VTM_E_TRUNCATED:  return "файл обрезан";
    case VTM_E_MAGIC:      return "нет магии VTM1";
    case VTM_E_VERSION:    return "неподдерживаемая версия";
    case VTM_E_FORMAT:     return "нарушен формат";
    case VTM_E_BOUNDS:     return "смещение или размер за границей файла";
    case VTM_E_NOMEM:      return "нехватка памяти";
    case VTM_E_TENSOR:     return "нет тензора или неподходящий kind";
    case VTM_E_SHAPE:      return "не сходятся размерности";
    case VTM_E_OPCODE:     return "неизвестный opcode";
    case VTM_E_UNSUPPORTED:return "операция пока не поддерживается";
    case VTM_E_SCALE:      return "испорченные триты или масштаб";
    case VTM_E_CALLBACK:   return "ошибка callback";
    default:               return "неизвестная ошибка";
    }
}

const char *vtm_op_name(uint8_t opcode) {
    switch (opcode) {
    case VTM_OP_TMV:  return "TMV";
    case VTM_OP_MM:   return "MM";
    case VTM_OP_RMSN: return "RMSN";
    case VTM_OP_ROPE: return "ROPE";
    case VTM_OP_SM:   return "SM";
    case VTM_OP_SILU: return "SILU";
    case VTM_OP_MUL:  return "MUL";
    case VTM_OP_ADD:  return "ADD";
    case VTM_OP_EMB:  return "EMB";
    case VTM_OP_KVW:  return "KVW";
    case VTM_OP_ATTN: return "ATTN";
    case VTM_OP_SAMP: return "SAMP";
    case VTM_OP_CPY:  return "CPY";
    case VTM_OP_SCAL: return "SCAL";
    case VTM_OP_HALT: return "HALT";
    default:          return "?";
    }
}

const char *vtm_opcode_name(uint8_t opcode) {
    return vtm_op_name(opcode);
}

// --- little-endian чтение --------------------------------------------------

static uint16_t rd16(const uint8_t *p) {
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

static uint32_t rd32(const uint8_t *p) {
    return (uint32_t)p[0]
         | ((uint32_t)p[1] << 8)
         | ((uint32_t)p[2] << 16)
         | ((uint32_t)p[3] << 24);
}

static uint64_t rd64(const uint8_t *p) {
    return (uint64_t)rd32(p) | ((uint64_t)rd32(p + 4) << 32);
}

static int32_t rd_i32(const uint8_t *p) {
    uint32_t u = rd32(p);
    int32_t v;
    memcpy(&v, &u, sizeof v);
    return v;
}

static float rd_f32(const uint8_t *p) {
    uint32_t u = rd32(p);
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

static int align_up(size_t v, size_t a, size_t *out) {
    size_t add;
    if (a == 0 || (a & (a - 1u)) != 0) return -1;
    add = a - 1u;
    if (v > SIZE_MAX - add) return -1;
    *out = (v + add) & ~add;
    return 0;
}

static int range_ok(uint32_t offset, uint64_t len, size_t data_size) {
    if ((uint64_t)offset > (uint64_t)data_size) return 0;
    return len <= (uint64_t)data_size - (uint64_t)offset;
}

static int validate_words(const uint8_t *p, uint32_t n_words,
                          char *err, size_t errcap) {
    uint32_t i;
    for (i = 0; i < n_words; i++) {
        uint32_t w = rd32(p + (size_t)i * 4u);
        if (w > VTM_TRIT_MAX_WORD) {
            snprintf(err, errcap,
                     "слово тритов #%u = %u больше 3^20-1", i, w);
            return -1;
        }
    }
    return 0;
}

static int tensor_arena_size(const vtm_tensor *t, size_t *out) {
    uint64_t n;
    if (t->kind == VTM_TENSOR_TRIT) {
        n = (uint64_t)t->n_blocks * 32u + (uint64_t)t->n_scales * 4u;
    } else {
        n = (uint64_t)t->n_elements * 4u;
    }
    if (n > (uint64_t)SIZE_MAX) return -1;
    *out = (size_t)n;
    return 0;
}

// Скопировать данные тензора из файла в арену. Для тритов сначала идут слова
// (выровнены на 64), затем масштабы (выровнены на 32, уже fp32).
static void fill_tensor(vtm_tensor *t, const uint8_t *data, uint8_t *dst) {
    if (t->kind == VTM_TENSOR_TRIT) {
        uint32_t nb = t->n_blocks;
        uint32_t ns = t->n_scales;
        uint32_t i;
        const uint8_t *sp;
        const uint8_t *wp;
        uint32_t *words = (uint32_t *)dst;
        float *scales = (float *)(void *)(dst + (size_t)nb * 32u);

        if (t->layout == VTM_LAYOUT_TRT1) {
            const uint8_t *seg = data + t->offset;
            size_t ss = t->scale_is_f32 ? 4u : 2u;
            sp = seg + 18u;
            wp = seg + 18u + (size_t)ns * ss;
        } else {
            sp = data + t->scale_offset;
            wp = data + t->offset;
        }
        for (i = 0; i < ns; i++) {
            scales[i] = t->scale_is_f32 ? rd_f32(sp + (size_t)i * 4u)
                                        : vtm_half_to_float(rd16(sp + (size_t)i * 2u));
        }
        for (i = 0; i < nb; i++) {
            words[i * VTM_WORDS_PER_BLOCK + 0] =
                rd32(wp + ((size_t)i * VTM_WORDS_PER_BLOCK + 0) * 4u);
            words[i * VTM_WORDS_PER_BLOCK + 1] =
                rd32(wp + ((size_t)i * VTM_WORDS_PER_BLOCK + 1) * 4u);
            words[i * VTM_WORDS_PER_BLOCK + 2] =
                rd32(wp + ((size_t)i * VTM_WORDS_PER_BLOCK + 2) * 4u);
            words[i * VTM_WORDS_PER_BLOCK + 3] =
                rd32(wp + ((size_t)i * VTM_WORDS_PER_BLOCK + 3) * 4u);
            words[i * VTM_WORDS_PER_BLOCK + 4] =
                rd32(wp + ((size_t)i * VTM_WORDS_PER_BLOCK + 4) * 4u);
            words[i * VTM_WORDS_PER_BLOCK + 5] =
                rd32(wp + ((size_t)i * VTM_WORDS_PER_BLOCK + 5) * 4u);
            words[i * VTM_WORDS_PER_BLOCK + 6] =
                rd32(wp + ((size_t)i * VTM_WORDS_PER_BLOCK + 6) * 4u);
            words[i * VTM_WORDS_PER_BLOCK + 7] =
                rd32(wp + ((size_t)i * VTM_WORDS_PER_BLOCK + 7) * 4u);
        }
        // const-поля структуры заполняются один раз при загрузке.
        t->words = words;
        t->scales = scales;
    } else if (t->kind == VTM_TENSOR_F32) {
        float *y = (float *)(void *)dst;
        uint32_t i;
        const uint8_t *src = data + t->offset;
        for (i = 0; i < t->n_elements; i++)
            y[i] = rd_f32(src + (size_t)i * 4u);
        t->f32 = y;
    } else if (t->kind == VTM_TENSOR_F16) {
        float *y = (float *)(void *)dst;
        uint32_t i;
        const uint8_t *src = data + t->offset;
        for (i = 0; i < t->n_elements; i++)
            y[i] = vtm_half_to_float(rd16(src + (size_t)i * 2u));
        t->f32 = y;
    } else { // BF16
        float *y = (float *)(void *)dst;
        uint32_t i;
        const uint8_t *src = data + t->offset;
        for (i = 0; i < t->n_elements; i++)
            y[i] = vtm_bf16_to_float(rd16(src + (size_t)i * 2u));
        t->f32 = y;
    }
}

// --- метаданные: минимальный разбор JSON -----------------------------------
//
// Мета — JSON-объект за секцией данных (docs/vtm.md, «Метаданные»). Нужны
// четыре ключа, поэтому дерево не строим: обходим документ по грамматике
// JSON, пропуская значения целиком, и запоминаем только интересующие. Значения
// меты могут быть большими (expected.logits синтетики — мегабайты), поэтому
// вложенные массивы/объекты не материализуются.
//
// Любая порча (нет закрывающей скобки, мусор в конце, глубже
// VTM_JSON_MAX_DEPTH, не тот тип у ключа) — это НЕ ошибка загрузки: мета
// просто не применяется, и остаются дефолты. Так требует задача: битая мета
// не должна ронять контейнер.

#define VTM_JSON_MAX_DEPTH 64

typedef struct {
    double   rope_theta;
    int      rope_layout;
    uint32_t vocab_size;
    int      tied_embeddings;
    // Глубина, на которой ключ найден (-1 = ещё не найден). Верхний уровень
    // приоритетнее вложенного meta["model"]: иначе итог зависел бы от порядка
    // ключей в JSON.
    int      theta_depth, layout_depth, vocab_depth, tied_depth;
} meta_fields;

// Можно ли записать значение с этой глубины: только если ключ ещё не найден
// или найден глубже.
static int meta_take(int have_depth, int depth) {
    return have_depth < 0 || depth < have_depth;
}

typedef struct {
    const char *p;
    size_t      n;
    size_t      i;
    int         depth;
} json_scan;

static void json_ws(json_scan *s) {
    while (s->i < s->n) {
        char c = s->p[s->i];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') s->i++;
        else break;
    }
}

static int json_hex(unsigned char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

// Строка в кавычках. out == NULL — только пропустить. Escape-последовательности
// съедаются, но в out попадает только их ASCII-смысл; \uXXXX даёт '?'
// (в читаемых нами ключах и значениях escape не встречается).
static int json_string(json_scan *s, char *out, size_t cap) {
    size_t o = 0;
    if (s->i >= s->n || s->p[s->i] != '"') return -1;
    s->i++;
    while (s->i < s->n) {
        unsigned char c = (unsigned char)s->p[s->i++];
        if (c == '"') {
            if (out && cap) out[o < cap ? o : cap - 1u] = '\0';
            return 0;
        }
        if (c < 0x20u) return -1;
        if (c == '\\') {
            if (s->i >= s->n) return -1;
            c = (unsigned char)s->p[s->i++];
            switch (c) {
            case '"': case '\\': case '/':
            case 'b': case 'f': case 'n': case 'r': case 't':
                break;
            case 'u': {
                int k;
                for (k = 0; k < 4; k++) {
                    if (s->i >= s->n || json_hex((unsigned char)s->p[s->i]) < 0)
                        return -1;
                    s->i++;
                }
                c = '?';
                break;
            }
            default:
                return -1;
            }
        }
        if (out && o + 1u < cap) out[o++] = (char)c;
    }
    return -1;
}

// Число по грамматике JSON; strtod получает уже проверенный кусок.
static int json_number(json_scan *s, double *out) {
    size_t start = s->i;
    char buf[64];
    size_t len;
    char *end = NULL;
    double v;

    if (s->i < s->n && s->p[s->i] == '-') s->i++;
    if (s->i >= s->n) return -1;
    if (s->p[s->i] == '0') {
        s->i++;
    } else if (s->p[s->i] >= '1' && s->p[s->i] <= '9') {
        while (s->i < s->n && s->p[s->i] >= '0' && s->p[s->i] <= '9') s->i++;
    } else {
        return -1;
    }
    if (s->i < s->n && s->p[s->i] == '.') {
        s->i++;
        if (s->i >= s->n || s->p[s->i] < '0' || s->p[s->i] > '9') return -1;
        while (s->i < s->n && s->p[s->i] >= '0' && s->p[s->i] <= '9') s->i++;
    }
    if (s->i < s->n && (s->p[s->i] == 'e' || s->p[s->i] == 'E')) {
        s->i++;
        if (s->i < s->n && (s->p[s->i] == '+' || s->p[s->i] == '-')) s->i++;
        if (s->i >= s->n || s->p[s->i] < '0' || s->p[s->i] > '9') return -1;
        while (s->i < s->n && s->p[s->i] >= '0' && s->p[s->i] <= '9') s->i++;
    }
    len = s->i - start;
    if (len == 0 || len >= sizeof buf) return -1;
    memcpy(buf, s->p + start, len);
    buf[len] = '\0';
    v = strtod(buf, &end);
    if (end != buf + len || !isfinite(v)) return -1;
    if (out) *out = v;
    return 0;
}

static int json_literal(json_scan *s, const char *lit) {
    size_t len = strlen(lit);
    if (s->i > s->n || len > s->n - s->i) return -1;
    if (memcmp(s->p + s->i, lit, len) != 0) return -1;
    s->i += len;
    return 0;
}

static int json_value(json_scan *s, meta_fields *mf);
static int json_object(json_scan *s, meta_fields *mf);
static int json_array(json_scan *s, meta_fields *mf);

// 1 rope_theta, 2 rope_layout, 3 vocab_size, 4 tied_embeddings, 0 — чужой ключ.
// Ключи ищутся на любом уровне: компилятор кладёт часть конфигурации и в
// meta["model"], и на верхний уровень, а формат это не запрещает.
static int json_key_kind(const char *key) {
    if (strcmp(key, "rope_theta") == 0) return 1;
    if (strcmp(key, "rope_layout") == 0) return 2;
    if (strcmp(key, "vocab_size") == 0 || strcmp(key, "vocab") == 0) return 3;
    if (strcmp(key, "tied_embeddings") == 0 ||
        strcmp(key, "tie_word_embeddings") == 0) return 4;
    return 0;
}

// Разобрать значение известного ключа. Тип не тот — значение пропускается,
// ключ остаётся с дефолтом (битая мета не должна ничего ломать).
static int json_known_value(json_scan *s, meta_fields *mf, int kind) {
    unsigned char c;

    json_ws(s);
    if (s->i >= s->n) return -1;
    c = (unsigned char)s->p[s->i];
    if (c == '"') {
        char val[32];
        if (json_string(s, val, sizeof val) != 0) return -1;
        if (kind == 2 && meta_take(mf->layout_depth, s->depth)) {
            if (strcmp(val, "half") == 0) {
                mf->rope_layout = VTM_ROPE_LAYOUT_HALF;
                mf->layout_depth = s->depth;
            } else if (strcmp(val, "interleaved") == 0) {
                mf->rope_layout = VTM_ROPE_LAYOUT_INTERLEAVED;
                mf->layout_depth = s->depth;
            }
        }
        return 0;
    }
    if (c == '-' || (c >= '0' && c <= '9')) {
        double v;
        if (json_number(s, &v) != 0) return -1;
        if (kind == 1) {
            if (v > 0.0 && v <= 1e15 && meta_take(mf->theta_depth, s->depth)) {
                mf->rope_theta = v;
                mf->theta_depth = s->depth;
            }
        } else if (kind == 3) {
            if (v >= 1.0 && v <= (double)VTM_MAX_ELEMENTS && v == floor(v) &&
                meta_take(mf->vocab_depth, s->depth)) {
                mf->vocab_size = (uint32_t)v;
                mf->vocab_depth = s->depth;
            }
        } else if (kind == 4) {
            if ((v == 0.0 || v == 1.0) && meta_take(mf->tied_depth, s->depth)) {
                mf->tied_embeddings = (v != 0.0);
                mf->tied_depth = s->depth;
            }
        }
        return 0;
    }
    if (json_literal(s, "true") == 0) {
        if (kind == 4 && meta_take(mf->tied_depth, s->depth)) {
            mf->tied_embeddings = 1;
            mf->tied_depth = s->depth;
        }
        return 0;
    }
    if (json_literal(s, "false") == 0) {
        if (kind == 4 && meta_take(mf->tied_depth, s->depth)) {
            mf->tied_embeddings = 0;
            mf->tied_depth = s->depth;
        }
        return 0;
    }
    if (json_literal(s, "null") == 0) return 0;
    // Объект/массив вместо скаляра: пропустить, заодно собрав вложенные ключи.
    return json_value(s, mf);
}

static int json_object(json_scan *s, meta_fields *mf) {
    if (s->depth >= VTM_JSON_MAX_DEPTH) return -1;
    s->depth++;
    s->i++; // '{'
    json_ws(s);
    if (s->i < s->n && s->p[s->i] == '}') {
        s->i++;
        s->depth--;
        return 0;
    }
    for (;;) {
        char key[64];
        int kind;
        if (json_string(s, key, sizeof key) != 0) break;
        json_ws(s);
        if (s->i >= s->n || s->p[s->i] != ':') break;
        s->i++;
        kind = json_key_kind(key);
        if (kind != 0) {
            if (json_known_value(s, mf, kind) != 0) break;
        } else if (json_value(s, mf) != 0) {
            break;
        }
        json_ws(s);
        if (s->i < s->n && s->p[s->i] == ',') {
            s->i++;
            json_ws(s);
            continue;
        }
        if (s->i < s->n && s->p[s->i] == '}') {
            s->i++;
            s->depth--;
            return 0;
        }
        break;
    }
    s->depth--;
    return -1;
}

static int json_array(json_scan *s, meta_fields *mf) {
    if (s->depth >= VTM_JSON_MAX_DEPTH) return -1;
    s->depth++;
    s->i++; // '['
    json_ws(s);
    if (s->i < s->n && s->p[s->i] == ']') {
        s->i++;
        s->depth--;
        return 0;
    }
    for (;;) {
        if (json_value(s, mf) != 0) break;
        json_ws(s);
        if (s->i < s->n && s->p[s->i] == ',') {
            s->i++;
            continue;
        }
        if (s->i < s->n && s->p[s->i] == ']') {
            s->i++;
            s->depth--;
            return 0;
        }
        break;
    }
    s->depth--;
    return -1;
}

static int json_value(json_scan *s, meta_fields *mf) {
    unsigned char c;

    json_ws(s);
    if (s->i >= s->n) return -1;
    c = (unsigned char)s->p[s->i];
    if (c == '{') return json_object(s, mf);
    if (c == '[') return json_array(s, mf);
    if (c == '"') return json_string(s, NULL, 0);
    if (c == '-' || (c >= '0' && c <= '9')) return json_number(s, NULL);
    if (json_literal(s, "true") == 0) return 0;
    if (json_literal(s, "false") == 0) return 0;
    if (json_literal(s, "null") == 0) return 0;
    return -1;
}

// 0 — мета разобрана (объект, без мусора после него), -1 — порча.
// Поля commit'ятся только целиком: недособранный JSON не оставляет полуметы.
static int meta_parse(const char *json, size_t len, meta_fields *out) {
    json_scan s;
    meta_fields f;

    s.p = json;
    s.n = len;
    s.i = 0;
    s.depth = 0;
    f.rope_theta = VTM_DEFAULT_ROPE_THETA;
    f.rope_layout = VTM_ROPE_LAYOUT_HALF;
    f.vocab_size = 0;
    f.tied_embeddings = 0;
    f.theta_depth = f.layout_depth = f.vocab_depth = f.tied_depth = -1;
    json_ws(&s);
    if (s.i >= s.n || s.p[s.i] != '{') return -1;
    if (json_object(&s, &f) != 0) return -1;
    json_ws(&s);
    if (s.i != s.n) return -1;
    *out = f;
    return 0;
}

// --- загрузка из памяти ----------------------------------------------------

vtm_t *vtm_load_mem(const void *data, size_t size,
                    vtm_status *status, char *errbuf, size_t errcap) {
    vtm_status st = VTM_OK;
    vtm_t *m;
    const uint8_t *p = (const uint8_t *)data;
    uint32_t n_instr, n_tensors, n_program;
    size_t table_end, data_start, program_bytes, program_off, data_size;
    uint32_t meta_off = 0, meta_len = 0;
    int meta_valid = 0;
    size_t *sizes = NULL;
    uint32_t i;

    if (status) *status = VTM_OK;
    if (errbuf && errcap) errbuf[0] = '\0';

    if (!p) {
        if (status) *status = VTM_E_ARG;
        set_global_error("vtm_load_mem: data == NULL");
        if (errbuf && errcap) snprintf(errbuf, errcap, "data == NULL");
        return NULL;
    }
    if (size > VTM_MAX_FILE_BYTES) {
        if (status) *status = VTM_E_BOUNDS;
        set_global_error("буфер больше предела %zu байт", VTM_MAX_FILE_BYTES);
        if (errbuf && errcap)
            snprintf(errbuf, errcap, "буфер больше предела %zu байт",
                     VTM_MAX_FILE_BYTES);
        return NULL;
    }
    if (size < VTM_HEADER_SIZE) {
        if (status) *status = VTM_E_TRUNCATED;
        set_global_error("файл короче заголовка: %zu < %u байт",
                         size, (unsigned)VTM_HEADER_SIZE);
        if (errbuf && errcap)
            snprintf(errbuf, errcap, "файл короче заголовка: %zu < %u байт",
                     size, (unsigned)VTM_HEADER_SIZE);
        return NULL;
    }

    m = calloc(1, sizeof *m);
    if (!m) {
        if (status) *status = VTM_E_NOMEM;
        set_global_error("не хватило памяти на vtm_t");
        return NULL;
    }
    m->backend = vtm_backend_from_env();
    m->input_token = -1;
    // Дефолты меты: без меты/с битой метой ROPE работает на 10000, как раньше.
    m->rope_theta = VTM_DEFAULT_ROPE_THETA;
    m->rope_layout = VTM_ROPE_LAYOUT_HALF;

#define FAIL(code, ...) do { \
        st = (code); \
        vtm_set_error(m, st, __VA_ARGS__); \
        goto fail; \
    } while (0)

    if (memcmp(p, "VTM1", 4) != 0)
        FAIL(VTM_E_MAGIC, "нет магии VTM1");
    if (rd16(p + 4) == VTM_FORMAT_VERSION_V01)
        FAIL(VTM_E_VERSION, "формат v0.1 не поддерживается");
    if (rd16(p + 4) != VTM_FORMAT_VERSION)
        FAIL(VTM_E_VERSION, "версия контейнера %u, ожидали %u",
             (unsigned)rd16(p + 4), (unsigned)VTM_FORMAT_VERSION);

    m->header_flags = rd16(p + 6);
    n_instr = rd32(p + 8);
    n_tensors = rd32(p + 12);
    n_program = rd32(p + 16);

    if (n_tensors > VTM_MAX_TENSORS)
        FAIL(VTM_E_FORMAT, "n_tensors=%u больше предела %u",
             n_tensors, (unsigned)VTM_MAX_TENSORS);
    if ((uint64_t)n_tensors * VTM_TENSOR_ENTRY_SIZE > (uint64_t)(size - 32))
        FAIL(VTM_E_TRUNCATED,
             "таблица тензоров (%u × %u) не влезает в файл", n_tensors,
             (unsigned)VTM_TENSOR_ENTRY_SIZE);
    table_end = 32u + (size_t)n_tensors * VTM_TENSOR_ENTRY_SIZE;
    if (align_up(table_end, VTM_DATA_ALIGN, &data_start) != 0 || data_start > size)
        FAIL(VTM_E_TRUNCATED, "не удалось выровнять начало данных");

    if (n_instr > VTM_MAX_INSTR)
        FAIL(VTM_E_FORMAT, "n_instr=%u больше предела %u",
             n_instr, (unsigned)VTM_MAX_INSTR);
    if ((uint64_t)n_instr * VTM_INSTR_SIZE > (uint64_t)(size - data_start))
        FAIL(VTM_E_TRUNCATED, "программа (%u × 16) не влезает в файл", n_instr);
    program_bytes = (size_t)n_instr * VTM_INSTR_SIZE;
    program_off = size - program_bytes;
    if (program_off < data_start)
        FAIL(VTM_E_TRUNCATED, "программа перекрывает секцию данных");
    data_size = program_off - data_start;

    // Метаданные — JSON между секцией данных и программой (docs/vtm.md).
    // Смещение и длину берём из «резерва» заголовка (байты 24 и 28). Если они
    // врут или указывают мимо программы — это битая мета, а не битый файл:
    // мету просто не читаем, данные трактуем как раньше (до программы).
    meta_off = rd32(p + 24);
    meta_len = rd32(p + 28);
    if (meta_off != 0 && meta_len != 0 &&
        (uint64_t)meta_off >= (uint64_t)data_start &&
        (uint64_t)meta_off + (uint64_t)meta_len <= (uint64_t)program_off) {
        meta_valid = 1;
        data_size = (size_t)meta_off - data_start;
    }

    if (n_program != 0) {
        uint64_t as_size = (uint64_t)n_instr * VTM_INSTR_SIZE;
        if ((uint64_t)n_program != as_size)
            FAIL(VTM_E_FORMAT,
                 "n_program=%u не равен размеру программы (%llu)",
                 n_program, (unsigned long long)as_size);
    }

    m->n_tensors = n_tensors;
    if (n_tensors) {
        m->tensors = calloc((size_t)n_tensors, sizeof(vtm_tensor));
        if (!m->tensors)
            FAIL(VTM_E_NOMEM, "не хватило памяти на %u тензоров", n_tensors);
    }

    for (i = 0; i < n_tensors; i++) {
        const uint8_t *e = p + 32u + (size_t)i * VTM_TENSOR_ENTRY_SIZE;
        vtm_tensor *t = &m->tensors[i];
        const uint8_t *nul = (const uint8_t *)memchr(e, 0, 24);
        uint64_t n_el = 1;
        uint32_t d;

        if (!nul)
            FAIL(VTM_E_FORMAT, "тензор %u: имя из 24 байт без NUL", i);
        memcpy(t->name, e, 24);
        t->kind = e[24];
        t->ndim = e[25];
        if (t->kind > VTM_TENSOR_BF16)
            FAIL(VTM_E_FORMAT, "тензор %u (%s): неизвестный kind %u",
                 i, t->name, (unsigned)t->kind);
        if (t->ndim < 1 || t->ndim > VTM_MAX_DIMS)
            FAIL(VTM_E_FORMAT, "тензор %u (%s): ndim=%u вне 1..%u",
                 i, t->name, (unsigned)t->ndim, (unsigned)VTM_MAX_DIMS);

        for (d = 0; d < VTM_MAX_DIMS; d++) {
            uint32_t dim = rd32(e + 26u + (size_t)d * 4u);
            if (d < t->ndim) {
                if (dim == 0)
                    FAIL(VTM_E_FORMAT,
                         "тензор %u (%s): нулевая размерность dims[%u]",
                         i, t->name, d);
                t->dims[d] = dim;
                n_el *= dim;
                if (n_el > (uint64_t)VTM_MAX_ELEMENTS)
                    FAIL(VTM_E_FORMAT,
                         "тензор %u (%s): слишком много элементов", i, t->name);
            } else {
                t->dims[d] = 0;
            }
        }
        t->n_elements = (uint32_t)n_el;
        t->offset = rd32(e + 42u);
        t->scale_offset = rd32(e + 46u);
        t->n_blocks = 0;
        t->n_scales = 0;
        t->scale_block = 0;
        t->scale_is_f32 = 0;
        t->layout = VTM_LAYOUT_TRT1;

        if ((t->offset % VTM_DATA_ALIGN) != 0)
            FAIL(VTM_E_FORMAT, "тензор %u (%s): offset=%u не кратен %u",
                 i, t->name, t->offset, (unsigned)VTM_DATA_ALIGN);
        if ((uint64_t)t->offset >= (uint64_t)data_size)
            FAIL(VTM_E_BOUNDS, "тензор %u (%s): offset=%u за концом данных (%zu)",
                 i, t->name, t->offset, data_size);

        if (t->kind == VTM_TENSOR_TRIT) {
            uint64_t need;
            if (t->scale_offset == VTM_NO_OFFSET) {
                const uint8_t *seg = p + data_start + t->offset;
                uint64_t n_trits;
                uint32_t n_scales, storage_blocks, n128, n160;
                uint32_t scale_block;
                size_t ss, seg_bytes;
                char werr[160];

                if (!range_ok(t->offset, 18u, data_size))
                    FAIL(VTM_E_BOUNDS,
                         "тензор %u (%s): TRT1-заголовок за границей данных",
                         i, t->name);
                if (memcmp(seg, "TRT1", 4) != 0)
                    FAIL(VTM_E_FORMAT,
                         "тензор %u (%s): kind=TRIT без TRT1 и без scale_offset",
                         i, t->name);
                if (seg[4] != 1)
                    FAIL(VTM_E_VERSION,
                         "тензор %u (%s): версия TRT1 %u", i, t->name,
                         (unsigned)seg[4]);
                n_trits = rd64(seg + 6);
                n_scales = rd32(seg + 14);
                if (n_trits != (uint64_t)t->n_elements)
                    FAIL(VTM_E_FORMAT,
                         "тензор %u (%s): TRT1 n_trits=%llu, dims дают %u",
                         i, t->name, (unsigned long long)n_trits,
                         t->n_elements);
                storage_blocks = (uint32_t)((n_trits + VTM_TRITS_PER_BLOCK - 1u)
                                            / VTM_TRITS_PER_BLOCK);
                n160 = storage_blocks;
                n128 = (uint32_t)((n_trits + VTM_SCALE_BLOCK_128 - 1u)
                                  / VTM_SCALE_BLOCK_128);
                if (n_scales == 0)
                    FAIL(VTM_E_FORMAT,
                         "тензор %u (%s): TRT1 без масштабов", i, t->name);
                if ((m->header_flags & VTM_HEADER_FLAG_SCALE_BLOCK_160) != 0) {
                    if (n_scales != n160)
                        FAIL(VTM_E_FORMAT,
                             "тензор %u (%s): флаг 160, но n_scale=%u "
                             "(ожидали %u)",
                             i, t->name, n_scales, n160);
                    scale_block = VTM_SCALE_BLOCK_160;
                } else if (n_scales == n128 && n_scales != n160) {
                    scale_block = VTM_SCALE_BLOCK_128;
                } else if (n_scales == n160 && n_scales != n128) {
                    scale_block = VTM_SCALE_BLOCK_160;
                } else if (n_scales == n128 && n_scales == n160) {
                    // Неоднозначно: VTM-эталон (Python) фиксирует 128.
                    scale_block = VTM_SCALE_BLOCK_128;
                } else {
                    FAIL(VTM_E_FORMAT,
                         "тензор %u (%s): TRT1 n_scale=%u, ожидали %u (128) "
                         "или %u (160)",
                         i, t->name, n_scales, n128, n160);
                }
                ss = (seg[5] & 1u) ? 4u : 2u;
                need = 18u + (uint64_t)n_scales * ss
                     + (uint64_t)storage_blocks * 32u;
                if (need > (uint64_t)SIZE_MAX)
                    FAIL(VTM_E_BOUNDS, "тензор %u (%s): TRT1 слишком велик",
                         i, t->name);
                seg_bytes = (size_t)need;
                if (!range_ok(t->offset, (uint64_t)seg_bytes, data_size))
                    FAIL(VTM_E_BOUNDS,
                         "тензор %u (%s): TRT1-сегмент (%zu байт) за границей",
                         i, t->name, seg_bytes);
                if (validate_words(seg + 18u + (size_t)n_scales * ss,
                                   storage_blocks * VTM_WORDS_PER_BLOCK,
                                   werr, sizeof werr) != 0)
                    FAIL(VTM_E_SCALE, "тензор %u (%s): %s",
                         i, t->name, werr);
                t->n_blocks = storage_blocks;
                t->n_scales = n_scales;
                t->scale_block = scale_block;
                t->scale_is_f32 = (uint8_t)((seg[5] & 1u) ? 1u : 0u);
                t->layout = VTM_LAYOUT_TRT1;
            } else {
                uint32_t n_scales, storage_blocks;
                uint32_t scale_block = VTM_SCALE_BLOCK_160;
                size_t ss;
                char werr[160];
                storage_blocks = (uint32_t)(((uint64_t)t->n_elements
                                             + VTM_TRITS_PER_BLOCK - 1u)
                                            / VTM_TRITS_PER_BLOCK);
                n_scales = (uint32_t)(((uint64_t)t->n_elements
                                       + scale_block - 1u) / scale_block);
                if (storage_blocks == 0 || n_scales == 0)
                    FAIL(VTM_E_FORMAT, "тензор %u (%s): пустой raw-трит",
                         i, t->name);
                if ((t->scale_offset % VTM_DATA_ALIGN) != 0)
                    FAIL(VTM_E_FORMAT,
                         "тензор %u (%s): scale_offset=%u не кратен %u",
                         i, t->name, t->scale_offset, (unsigned)VTM_DATA_ALIGN);
                ss = (m->header_flags & VTM_HEADER_FLAG_F32_SCALES) ? 4u : 2u;
                if (!range_ok(t->offset,
                              (uint64_t)storage_blocks * 32u, data_size))
                    FAIL(VTM_E_BOUNDS,
                         "тензор %u (%s): слова тритов за границей данных",
                         i, t->name);
                if (!range_ok(t->scale_offset,
                              (uint64_t)n_scales * ss, data_size))
                    FAIL(VTM_E_BOUNDS,
                         "тензор %u (%s): масштабы за границей данных",
                         i, t->name);
                if (validate_words(p + data_start + t->offset,
                                   storage_blocks * VTM_WORDS_PER_BLOCK,
                                   werr, sizeof werr) != 0)
                    FAIL(VTM_E_SCALE, "тензор %u (%s): %s",
                         i, t->name, werr);
                t->n_blocks = storage_blocks;
                t->n_scales = n_scales;
                t->scale_block = scale_block;
                t->scale_is_f32 =
                    (uint8_t)((m->header_flags & VTM_HEADER_FLAG_F32_SCALES)
                              ? 1u : 0u);
                t->layout = VTM_LAYOUT_RAW;
            }
        } else {
            size_t esize = (t->kind == VTM_TENSOR_F32) ? 4u : 2u;
            uint64_t bytes = (uint64_t)t->n_elements * esize;
            if (!range_ok(t->offset, bytes, data_size))
                FAIL(VTM_E_BOUNDS,
                     "тензор %u (%s): данные (%llu байт) за границей",
                     i, t->name, (unsigned long long)bytes);
        }
    }

    m->n_instr = n_instr;
    if (n_instr) {
        m->instr = calloc((size_t)n_instr, sizeof(vtm_instr));
        if (!m->instr)
            FAIL(VTM_E_NOMEM, "не хватило памяти на %u инструкций", n_instr);
    }

    for (i = 0; i < n_instr; i++) {
        const uint8_t *ip = p + program_off + (size_t)i * VTM_INSTR_SIZE;
        vtm_instr *in = &m->instr[i];
        in->opcode = ip[0];
        in->flags = ip[1];
        in->dst = rd16(ip + 2);
        in->a = rd16(ip + 4);
        in->b = rd16(ip + 6);
        in->c = rd16(ip + 8);
        // Раскладка 16 байт как в Python-эталоне (<BBHHHH2xi>):
        // байты 10-11 — выравнивание, imm i32 — по смещению 12.
        in->imm = rd_i32(ip + 12);
        if (in->opcode < VTM_OP_TMV || in->opcode > VTM_OP_HALT)
            FAIL(VTM_E_OPCODE, "инструкция %u: неизвестный opcode 0x%02x",
                 i, (unsigned)in->opcode);
        if ((in->dst != VTM_NO_TENSOR && in->dst >= n_tensors) ||
            (in->a != VTM_NO_TENSOR && in->a >= n_tensors) ||
            (in->b != VTM_NO_TENSOR && in->b >= n_tensors) ||
            (in->c != VTM_NO_TENSOR && in->c >= n_tensors))
            FAIL(VTM_E_FORMAT,
                 "инструкция %u: индекс тензора за таблицей (n_tensors=%u)",
                 i, n_tensors);
    }

    // --- метаданные --------------------------------------------------------
    // Разбор терпимый: отсутствие, порча, неверные типы, слишком большая мета
    // — не ошибка загрузки. Поля не сбрасываются, дефолты уже выставлены.
    if (meta_valid) {
        meta_fields mf;
        if ((size_t)meta_len <= VTM_MAX_META_BYTES &&
            meta_parse((const char *)p + meta_off, (size_t)meta_len, &mf) == 0) {
            m->meta_present = 1;
            m->rope_theta = mf.rope_theta;
            m->rope_layout = mf.rope_layout;
            m->vocab_size = mf.vocab_size;
            m->tied_embeddings = mf.tied_embeddings;
            // Копия JSON для хоста; не удалась — не беда, работает мета-API.
            m->meta_json = (char *)malloc((size_t)meta_len + 1u);
            if (m->meta_json) {
                memcpy(m->meta_json, p + meta_off, (size_t)meta_len);
                m->meta_json[meta_len] = '\0';
            }
        }
    }

    // --- арена -------------------------------------------------------------
    sizes = (size_t *)calloc(n_tensors ? (size_t)n_tensors : 1u, sizeof(size_t));
    if (!sizes)
        FAIL(VTM_E_NOMEM, "не хватило памяти на размеры тензоров");
    {
        size_t total = 0;
        for (i = 0; i < n_tensors; i++) {
            size_t sz, aligned;
            if (tensor_arena_size(&m->tensors[i], &sz) != 0)
                FAIL(VTM_E_NOMEM, "тензор %u: размер арены переполнен", i);
            if (align_up(total, VTM_ARENA_ALIGN, &aligned) != 0)
                FAIL(VTM_E_NOMEM, "арена: переполнение выравнивания");
            if (sz > SIZE_MAX - aligned)
                FAIL(VTM_E_NOMEM, "арена: переполнение размера");
            sizes[i] = sz;
            total = aligned + sz;
        }
        m->arena_size = total;
        if (total) {
            uintptr_t base;
            if (total > SIZE_MAX - (VTM_ARENA_ALIGN - 1u))
                FAIL(VTM_E_NOMEM, "арена: переполнение размера");
            m->arena_raw = (uint8_t *)malloc(total + (VTM_ARENA_ALIGN - 1u));
            if (!m->arena_raw)
                FAIL(VTM_E_NOMEM, "арена: не удалось выделить %zu байт",
                     total + (VTM_ARENA_ALIGN - 1u));
            base = (uintptr_t)m->arena_raw + (VTM_ARENA_ALIGN - 1u);
            base &= ~(uintptr_t)(VTM_ARENA_ALIGN - 1u);
            m->arena = (uint8_t *)base;
        }
    }
    {
        size_t used = 0;
        for (i = 0; i < n_tensors; i++) {
            size_t off;
            if (align_up(used, VTM_ARENA_ALIGN, &off) != 0)
                FAIL(VTM_E_NOMEM, "арена: переполнение смещения");
            fill_tensor(&m->tensors[i], p + data_start, m->arena + off);
            used = off + sizes[i];
        }
    }

    free(sizes);
    sizes = NULL;
    if (status) *status = VTM_OK;
    if (errbuf && errcap) errbuf[0] = '\0';
    set_global_error("нет ошибки");
    return m;

fail:
    free(sizes);
    if (status) *status = st;
    if (errbuf && errcap)
        snprintf(errbuf, errcap, "%s", m->error[0] ? m->error : g_last_error);
    vtm_free(m);
    return NULL;
#undef FAIL
}

// --- загрузка из файла -----------------------------------------------------

vtm_t *vtm_load(const char *path) {
    FILE *f;
    long n;
    size_t size;
    uint8_t *buf;
    vtm_status st = VTM_OK;
    char err[256];
    vtm_t *m;

    err[0] = '\0';
    if (!path) {
        set_global_error("vtm_load: path == NULL");
        return NULL;
    }
    f = fopen(path, "rb");
    if (!f) {
        set_global_error("не удалось открыть %s", path);
        return NULL;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        set_global_error("не удалось найти конец %s", path);
        return NULL;
    }
    n = ftell(f);
    if (n < 0) {
        fclose(f);
        set_global_error("не удалось определить размер %s", path);
        return NULL;
    }
    if ((unsigned long)n > (unsigned long)VTM_MAX_FILE_BYTES) {
        fclose(f);
        set_global_error("%s: файл больше предела %zu байт",
                         path, VTM_MAX_FILE_BYTES);
        return NULL;
    }
    if (fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        set_global_error("не удалось перемотать %s", path);
        return NULL;
    }
    size = (size_t)n;
    buf = (uint8_t *)malloc(size ? size : 1u);
    if (!buf) {
        fclose(f);
        set_global_error("%s: не хватило памяти на %zu байт", path, size);
        return NULL;
    }
    if (size && fread(buf, 1, size, f) != size) {
        fclose(f);
        free(buf);
        set_global_error("%s: короткое чтение", path);
        return NULL;
    }
    fclose(f);

    m = vtm_load_mem(buf, size, &st, err, sizeof err);
    free(buf);
    if (!m) {
        if (err[0]) set_global_error("%s: %s", path, err);
        return NULL;
    }
    return m;
}

// --- доступ, состояние, освобождение --------------------------------------

void vtm_free(vtm_t *m) {
    if (!m) return;
    free(m->meta_json);
    free(m->tensors);
    free(m->instr);
    free(m->arena_raw);
    free(m);
}

uint32_t vtm_tensor_count(const vtm_t *m) {
    return m ? m->n_tensors : 0;
}

const vtm_tensor *vtm_tensor_by_index(const vtm_t *m, uint32_t index) {
    if (!m || index >= m->n_tensors) return NULL;
    return &m->tensors[index];
}

const vtm_tensor *vtm_tensor_by_name(const vtm_t *m, const char *name) {
    uint32_t i;
    if (!m || !name) return NULL;
    for (i = 0; i < m->n_tensors; i++) {
        if (strcmp(m->tensors[i].name, name) == 0) return &m->tensors[i];
    }
    return NULL;
}

uint32_t vtm_instr_count(const vtm_t *m) {
    return m ? m->n_instr : 0;
}

const vtm_instr *vtm_instruction(const vtm_t *m, uint32_t index) {
    if (!m || index >= m->n_instr) return NULL;
    return &m->instr[index];
}

void vtm_set_backend(vtm_t *m, const vtm_backend *backend) {
    if (!m) return;
    m->backend = backend ? backend : vtm_backend_from_env();
}

const char *vtm_backend_name(const vtm_t *m) {
    if (!m || !m->backend) return vtm_backend_scalar_lut.name;
    return m->backend->name ? m->backend->name : "(без имени)";
}

void vtm_set_token(vtm_t *m, int token) {
    if (m) m->input_token = token;
}

int vtm_input_token(const vtm_t *m) {
    return m ? m->input_token : -1;
}

void vtm_set_position(vtm_t *m, int position) {
    if (m) m->position = position;
}

int vtm_position(const vtm_t *m) {
    return m ? m->position : 0;
}

int vtm_generated(const vtm_t *m) {
    return m ? m->generated : 0;
}

void vtm_reset(vtm_t *m) {
    if (!m) return;
    m->position = 0;
    m->input_token = -1;
    m->generated = 0;
    m->halted = 0;
    m->error[0] = '\0';
}

const char *vtm_error_message(const vtm_t *m) {
    if (!m) return g_last_error;
    return m->error;
}

int vtm_halted(const vtm_t *m) {
    return m ? m->halted : 0;
}

// --- метаданные ------------------------------------------------------------

int vtm_meta_present(const vtm_t *m) {
    return m ? m->meta_present : 0;
}

double vtm_meta_rope_theta(const vtm_t *m) {
    if (!m || !(m->rope_theta > 0.0) || !isfinite(m->rope_theta))
        return VTM_DEFAULT_ROPE_THETA;
    return m->rope_theta;
}

int vtm_meta_rope_layout(const vtm_t *m) {
    if (!m) return VTM_ROPE_LAYOUT_HALF;
    if (m->rope_layout != VTM_ROPE_LAYOUT_HALF &&
        m->rope_layout != VTM_ROPE_LAYOUT_INTERLEAVED)
        return VTM_ROPE_LAYOUT_HALF;
    return m->rope_layout;
}

uint32_t vtm_meta_vocab_size(const vtm_t *m) {
    if (!m) return 0;
    return m->vocab_size <= VTM_MAX_ELEMENTS ? m->vocab_size : 0;
}

int vtm_meta_tied_embeddings(const vtm_t *m) {
    return m ? (m->tied_embeddings ? 1 : 0) : 0;
}

const char *vtm_meta_json(const vtm_t *m) {
    return (m && m->meta_json) ? m->meta_json : "";
}
