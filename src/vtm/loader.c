// Загрузчик .vtm: заголовок 32 байта, таблица тензоров по 52 байта,
// секция данных с выравниванием на 32 и программа в конце файла.
//
// Формат v0.2 (ADR 0011): dims — u32×4. Файлы v0.1 отвергаются: ценных
// артефактов v0.1 нет, синтетика пересобирается.
//
// Все проверки границ делаются до выделения памяти и до исполнения. Кривой
// файл обязан вернуть NULL и внятную причину, а не уронить процесс.
//
// Платформа: нужен little-endian POSIX. Под Windows этот TU не собирается —
// vtm.h даёт #error (см. «Портируемость» в docs/vtm.md), поэтому ветви
// #if defined(_WIN32) ниже — черновик будущего порта, а не рабочий путь.
// Big-endian проверяется один раз в load_core() (VTM_HOST_BIG_ENDIAN).
#include "vtm_internal.h"
#include "vtm_backend.h"
#include "sha256.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(_WIN32)
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

// TRT1-сегмент устроен как «18 байт заголовка + масштабы + слова», поэтому
// указатели внутри него не выровнены на 4: слово/масштаб идут по смещению
// 2 (mod 4). x86 и ARM допускают такие загрузки; на строгих платформах
// (и для проверки этого пути) можно собрать с -DVTM_UNALIGNED_LOADS=0 —
// тогда такие тензоры копируются в выровненную арену.
#ifndef VTM_UNALIGNED_LOADS
#if defined(__x86_64__) || defined(__i386__) || defined(__aarch64__) || \
    defined(__arm__) || defined(_M_X64) || defined(_M_IX86) || \
    defined(_M_ARM64)
#define VTM_UNALIGNED_LOADS 1
#else
#define VTM_UNALIGNED_LOADS 0
#endif
#endif

// --- KV-кэши вне файла: кэп и признак нового контейнера (задача 0118) ------
//
// 0101 вынесла KV-кэши из файла: у помеченного тензора (VTM_TENSOR_FLAG_KV_CACHE)
// payload'а нет, а загрузчик выделяет буфер n_elements×4 в арене. Файл в
// 192 байта (заголовок + таблица + программа) мог запросить до
// VTM_MAX_ELEMENTS×4 = 4 ГиБ — это вектор амплификации из 0110. Сумма всех
// KV-кэшей ограничена VTM_MAX_KV_BYTES.
//
// Почему 512 МиБ: каноническая 1.5B-раскладка (ctx 2048, kv_dim 256, 28 слоёв)
// — 112 МиБ, то есть запас ×4,5 (тот же профиль при ctx 8192); при этом
// 192-байтный файл просит полгигабайта виртуальной памяти, а не четыре.
// Меньше 256 МиБ нельзя: ctx 4096 у той же формы — уже 224 МиБ. Предел
// переопределяется на сборке: -DVTM_MAX_KV_BYTES=...
#ifndef VTM_MAX_KV_BYTES
#define VTM_MAX_KV_BYTES ((uint64_t)512u << 20)
#endif

// Флаг заголовка «KV-кэши (если есть) лежат вне файла». Бит 2 свободен:
// бит 0 — fp32-масштабы (C и Python), бит 15 — блок 160 (расширение C),
// бит 1 пишет Python (VFLAG_CAUSAL), C его не смотрит. Старые контейнеры
// присылают 0 и читаются прежним путём; контейнер с флагом обязан нести
// хотя бы один KV-кэш.
#define VTM_HEADER_FLAG_KV_OUT_OF_FILE 0x0004u

// offset KV-кэша в контейнере с флагом выше: payload'а нет, поэтому поле —
// не адрес. 0xFFFFFFFF = «смещения нет» (как VTM_NO_OFFSET у scale_offset).
// Это же — отказ старому бинарю: загрузчик до 0101 не знает ни флага, ни
// роли тензора, зато первым делом проверяет `offset % 32 != 0` и возвращает
// VTM_E_FORMAT, не читая данные (подробности — в отчёте задачи 0118).
#define VTM_KV_NO_OFFSET 0xFFFFFFFFu

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

// Флаг из окружения: 1/непусто (кроме "0") — включён. Нужен и на Windows,
// поэтому getenv/stdlib, без unistd.
static int vtm_env_flag(const char *name) {
    const char *v = getenv(name);
    return v && *v && strcmp(v, "0") != 0;
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
    case VTM_E_INTEGRITY:  return "нарушена целостность контейнера (sha256)";
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
//
// Все rd* предполагают little-endian хост: свопинга здесь нет намеренно.
// Контейнер little-endian, и в mmap-режиме часть данных читается из файла
// напрямую как uint32_t*/float*, так что свопинг в этих функциях всё равно не
// спас бы — пришлось бы копировать всё в арену. Поэтому big-endian отсекается
// один раз в load_core() (VTM_HOST_BIG_ENDIAN), до этих циклов.

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

// --- sha256: контроль целостности контейнера (задачи 0066, 0078) ----------
//
// Сам SHA-256 живёт в src/vtm/sha256.c: портативное ядро (голый C99, без
// openssl/libcrypto) и аппаратное (ARMv8 crypto extensions) с самопроверкой и
// откатом на портативное. Загрузчик лишь решает, что хешировать: секцию данных
// и мету, в которой значение meta_sha256 заменено 64 символами '0' — ровно это
// записал писатель (tools/vtm_ref.write_vtm). Замер 1,3 ГБ — в
// tests/vtm/test_sha256.c (аппаратный путь ~0,7 с вместо ~6 с).

// Найти в сыром JSON строковое значение "name": "64 hex"; вернуть смещение
// значения (без кавычек) или (size_t)-1. Сырой JSON, а не разобранный: мета
// может быть частично испорчена, и тогда tolerant-разбор её просто пропустит,
// а хеш обязан заметить подмену.
static size_t json_hex64_offset(const uint8_t *p, size_t n, const char *name) {
    size_t klen = strlen(name);
    size_t i;

    if (n < klen + 4u) return (size_t)-1;
    for (i = 0; i + klen + 2u <= n; i++) {
        size_t j, end, k;
        if (p[i] != '"') continue;
        if (memcmp(p + i + 1u, name, klen) != 0) continue;
        if (p[i + 1u + klen] != '"') continue;
        j = i + klen + 2u;
        while (j < n && (p[j] == ' ' || p[j] == '\t' || p[j] == '\r'
                         || p[j] == '\n'))
            j++;
        if (j >= n || p[j] != ':') continue;
        j++;
        while (j < n && (p[j] == ' ' || p[j] == '\t' || p[j] == '\r'
                         || p[j] == '\n'))
            j++;
        if (j >= n || p[j] != '"') continue;
        j++;
        if (j > n || n - j < 64u) continue;
        for (end = j, k = 0; k < 64u; k++, end++) {
            if (!((p[end] >= '0' && p[end] <= '9')
                  || (p[end] >= 'a' && p[end] <= 'f')))
                break;
        }
        if (k != 64u) continue;
        if (end >= n || p[end] != '"') continue;
        return j;
    }
    return (size_t)-1;
}

// Есть ли в сыром JSON ключ "name" со значением-объектом. Нужно, чтобы
// заметить блок integrity, у которого вырезали оба поля хешей.
static int json_object_key_present(const uint8_t *p, size_t n,
                                   const char *name) {
    size_t klen = strlen(name);
    size_t i;

    if (n < klen + 4u) return 0;
    for (i = 0; i + klen + 2u <= n; i++) {
        size_t j;
        if (p[i] != '"') continue;
        if (memcmp(p + i + 1u, name, klen) != 0) continue;
        if (p[i + 1u + klen] != '"') continue;
        j = i + klen + 2u;
        while (j < n && (p[j] == ' ' || p[j] == '\t' || p[j] == '\r'
                         || p[j] == '\n'))
            j++;
        if (j >= n || p[j] != ':') continue;
        j++;
        while (j < n && (p[j] == ' ' || p[j] == '\t' || p[j] == '\r'
                         || p[j] == '\n'))
            j++;
        if (j < n && p[j] == '{') return 1;
    }
    return 0;
}

// 1 — блок integrity есть и хеши сошлись; 0 — блока нет (старый контейнер);
// -1 — блок есть, но хеш/формат не сошёлся (текст ошибки в err).
static int verify_integrity(const uint8_t *file, size_t data_start,
                            uint32_t meta_off, uint32_t meta_len,
                            char *err, size_t errcap) {
    const uint8_t *meta = file + meta_off;
    size_t data_pos, meta_pos;
    char expected_data[65], expected_meta[65], actual[65];
    vtm_sha256_ctx ctx;
    uint8_t digest[32];
    uint8_t zeros[64];

    data_pos = json_hex64_offset(meta, meta_len, "data_sha256");
    meta_pos = json_hex64_offset(meta, meta_len, "meta_sha256");
    if (data_pos == (size_t)-1 && meta_pos == (size_t)-1) {
        // Блока нет — старый контейнер. Но если объект integrity есть, а
        // поля хешей вырезаны/испорчены, это уже повреждение.
        if (json_object_key_present(meta, meta_len, "integrity")) {
            snprintf(err, errcap,
                     "integrity: блок в мете есть, но data_sha256/"
                     "meta_sha256 в нём отсутствуют (блок повреждён)");
            return -1;
        }
        return 0;
    }
    if (data_pos == (size_t)-1 || meta_pos == (size_t)-1) {
        snprintf(err, errcap,
                 "integrity: в мете нет пары data_sha256/meta_sha256 "
                 "(блок повреждён)");
        return -1;
    }
    memcpy(expected_data, meta + data_pos, 64u);
    expected_data[64] = '\0';
    memcpy(expected_meta, meta + meta_pos, 64u);
    expected_meta[64] = '\0';

    if (vtm_env_flag("VTM_SKIP_HASH")) {
        fprintf(stderr, "VTM: VTM_SKIP_HASH=1 — проверка sha256 контейнера "
                        "пропущена (данные не сверены)\n");
        return 0;
    }

    vtm_sha256_init(&ctx);
    vtm_sha256_update(&ctx, file + data_start, (size_t)meta_off - data_start);
    vtm_sha256_final(&ctx, digest);
    vtm_sha256_hex(digest, actual);
    if (strcmp(actual, expected_data) != 0) {
        snprintf(err, errcap,
                 "sha256 секции данных не совпал: в файле %.16s…, в мете "
                 "%.16s…; контейнер повреждён (VTM_SKIP_HASH=1 — пропустить)",
                 actual, expected_data);
        return -1;
    }

    // Писатель сериализует мету с meta_sha256 = "000…0" (64 символа '0'),
    // поэтому хешируем мету, заменив значение на ASCII-нули ('0'), не 0x00.
    memset(zeros, '0', sizeof zeros);
    vtm_sha256_init(&ctx);
    vtm_sha256_update(&ctx, meta, meta_pos);
    vtm_sha256_update(&ctx, zeros, sizeof zeros);
    vtm_sha256_update(&ctx, meta + meta_pos + 64u, meta_len - meta_pos - 64u);
    vtm_sha256_final(&ctx, digest);
    vtm_sha256_hex(digest, actual);
    if (strcmp(actual, expected_meta) != 0) {
        snprintf(err, errcap,
                 "sha256 метаданных не совпал: в файле %.16s…, в мете "
                 "%.16s…; контейнер повреждён (VTM_SKIP_HASH=1 — пропустить)",
                 actual, expected_meta);
        return -1;
    }
    return 1;
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

// Проверки размера без «всегда ложных» сравнений.
//
// Было `n > (uint64_t)SIZE_MAX`: на 64-битных системах SIZE_MAX == UINT64_MAX,
// сравнение всегда ложно, и gcc честно валил сборку (-Wtype-limits). Заодно
// такие проверки не ловили переполнение умножения. Теперь считаем аккуратно:
// переполнение uint64 и выход за size_t (на 32-битных) проверяются раздельно.
static int mul_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (b != 0 && a > UINT64_MAX / b) return -1;
    *out = a * b;
    return 0;
}

static int add_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (a > UINT64_MAX - b) return -1;
    *out = a + b;
    return 0;
}

static int fits_size_t(uint64_t v, size_t *out) {
#if SIZE_MAX < UINT64_MAX
    if (v > (uint64_t)SIZE_MAX) return -1;
#endif
    *out = (size_t)v;
    return 0;
}

static int tensor_arena_size(const vtm_tensor *t, size_t *out) {
    uint64_t n, a, b;
    if (t->kind == VTM_TENSOR_TRIT) {
        if (mul_u64(t->n_blocks, 32u, &a) != 0) return -1;
        if (mul_u64(t->n_scales, 4u, &b) != 0) return -1;
        if (add_u64(a, b, &n) != 0) return -1;
    } else {
        if (mul_u64(t->n_elements, 4u, &n) != 0) return -1;
    }
    return fits_size_t(n, out);
}

// План загрузки одного тензора. В mmap-режиме (can_reference) неизменяемые
// fp32-тензоры и слова тритов остаются в отображении, поэтому в арене для них
// места нет. Копируются:
//   - всё, что интерпретатор может писать (writable: dst и V-кэш KVW);
//   - fp16/bf16 — их всё равно нужно развернуть в fp32 (t->f32 — float*);
//   - масштабы тритов в fp16: API отдаёт их как float*.
// KV-кэш (flag VTM_TENSOR_FLAG_KV_CACHE) — отдельный случай: содержимого в
// файле нет, поэтому копировать нечего; в арене под него выделяется буфер
// n_elements float (арена выделена calloc'ом и стартует с нулей).
typedef struct {
    size_t arena_size;  // байт в арене
    int    copy_data;   // fp32/fp16/bf16: копировать значения в арену
    int    copy_words;  // трит: копировать слова в арену
    int    copy_scales; // трит: копировать/конвертировать масштабы в арену
} tensor_plan;

static int plan_tensor(const vtm_tensor *t, int writable, int can_reference,
                       tensor_plan *p) {
    size_t sz;
    uint64_t kv_bytes = 0, scale_bytes = 0;

    p->arena_size = 0;
    p->copy_data = 0;
    p->copy_words = 0;
    p->copy_scales = 0;

    if ((t->flags & VTM_TENSOR_FLAG_KV_CACHE) != 0) {
        // writable и can_reference тут ни при чём: файла нет, копировать
        // нечего. Размер считается по dims, а не по payload'у; kind проверен
        // при разборе записи таблицы.
        if (mul_u64(t->n_elements, 4u, &kv_bytes) != 0) return -1;
        if (fits_size_t(kv_bytes, &p->arena_size) != 0) return -1;
        return 0;
    }

    if (!can_reference || writable) {
        if (tensor_arena_size(t, &sz) != 0) return -1;
        p->arena_size = sz;
        p->copy_data = 1;
        if (t->kind == VTM_TENSOR_TRIT) {
            p->copy_words = 1;
            p->copy_scales = 1;
        }
        return 0;
    }

    if (t->kind == VTM_TENSOR_F32) return 0; // отображение читается как float
    if (t->kind == VTM_TENSOR_F16 || t->kind == VTM_TENSOR_BF16) {
        if (tensor_arena_size(t, &sz) != 0) return -1;
        p->arena_size = sz;
        p->copy_data = 1;
        return 0;
    }

    // TRIT: слова — прямо в отображении (главный выигрыш по памяти).
    // Масштабы fp32 тоже читаются из отображения; fp16 приходится развернуть.
#if !VTM_UNALIGNED_LOADS
    // Строгие платформы: TRT1 отдаёт указатели не по границе 4 байт.
    if (t->layout == VTM_LAYOUT_TRT1) {
        if (tensor_arena_size(t, &sz) != 0) return -1;
        p->arena_size = sz;
        p->copy_data = 1;
        p->copy_words = 1;
        p->copy_scales = 1;
        return 0;
    }
#endif
    p->copy_scales = !t->scale_is_f32;
    if (p->copy_scales) {
        if (mul_u64(t->n_scales, 4u, &scale_bytes) != 0) return -1;
        if (fits_size_t(scale_bytes, &p->arena_size) != 0) return -1;
    }
    return 0;
}

static const uint8_t *trit_words_ptr(const vtm_tensor *t, const uint8_t *data) {
    if (t->layout == VTM_LAYOUT_TRT1) {
        size_t ss = t->scale_is_f32 ? 4u : 2u;
        return data + t->offset + 18u + (size_t)t->n_scales * ss;
    }
    return data + t->offset;
}

static const uint8_t *trit_scales_ptr(const vtm_tensor *t, const uint8_t *data) {
    if (t->layout == VTM_LAYOUT_TRT1) return data + t->offset + 18u;
    return data + t->scale_offset;
}

// Разложить тензор по арене (copy_*) или выставить указатели в отображение.
// data — начало секции данных (data_start уже прибавлен вызывающим).
// Для тритов слова идут первыми (выровнены на 64), затем масштабы.
static void fill_tensor_planned(vtm_tensor *t, const uint8_t *data,
                                uint8_t *dst, const tensor_plan *p) {
    if ((t->flags & VTM_TENSOR_FLAG_KV_CACHE) != 0) {
        // Копировать нечего: буфер уже в арене и обнулён calloc'ом (см.
        // выделение арены). Нули — это и есть «пустой кэш»: ровно то же
        // состояние, что давал старый контейнер, где в файле лежали нули.
        t->f32 = (float *)(void *)dst;
        return;
    }
    if (t->kind == VTM_TENSOR_TRIT) {
        uint32_t nb = t->n_blocks;
        uint32_t ns = t->n_scales;
        uint32_t i;
        const uint8_t *sp = trit_scales_ptr(t, data);
        const uint8_t *wp = trit_words_ptr(t, data);

        if (p->copy_words) {
            uint32_t *words = (uint32_t *)(void *)dst;
            for (i = 0; i < nb; i++) {
                uint32_t w;
                for (w = 0; w < VTM_WORDS_PER_BLOCK; w++) {
                    words[i * VTM_WORDS_PER_BLOCK + w] =
                        rd32(wp + ((size_t)i * VTM_WORDS_PER_BLOCK + w) * 4u);
                }
            }
            t->words = words;
        } else {
            t->words = (const uint32_t *)(const void *)wp;
        }

        if (p->copy_scales) {
            float *scales = (float *)(void *)(dst + (p->copy_words
                                                     ? (size_t)nb * 32u : 0u));
            for (i = 0; i < ns; i++) {
                scales[i] = t->scale_is_f32
                          ? rd_f32(sp + (size_t)i * 4u)
                          : vtm_half_to_float(rd16(sp + (size_t)i * 2u));
            }
            t->scales = scales;
        } else {
            t->scales = (const float *)(const void *)sp;
        }
    } else if (t->kind == VTM_TENSOR_F32) {
        if (p->copy_data) {
            float *y = (float *)(void *)dst;
            uint32_t i;
            const uint8_t *src = data + t->offset;
            for (i = 0; i < t->n_elements; i++)
                y[i] = rd_f32(src + (size_t)i * 4u);
            t->f32 = y;
        } else {
            // Немодифицируемый fp32 лежит в отображении как есть: offset
            // кратен 32, отображение выровнено на страницу, float* валиден.
            t->f32 = (float *)(void *)(uintptr_t)(data + t->offset);
        }
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

// Текст отказа на big-endian. Один на оба входа (файл и память): причину
// хочется видеть одинаковую и без «см. исходники». Держим короче 256 байт —
// это ёмкость g_last_error и m->error (тест проверяет, что хвост не срезан).
#define VTM_BIG_ENDIAN_MSG \
    "big-endian хост не поддерживается: контейнер .vtm хранит u32/f32 в " \
    "little-endian, свопинга нет; нужен little-endian хост — " \
    "docs/vtm.md, «Портируемость»"

// Общее ядро загрузки. can_reference == 1 — данные лежат в mmap-отображении,
// которое живёт до vtm_free(): неизменяемые тензоры можно оставить там.
// can_reference == 0 — буфер принадлежит вызывающему (vtm_load_mem) или это
// фолбэк: всё копируется в арену, как в v0.1.
static vtm_t *load_core(const void *data, size_t size, int can_reference,
                        vtm_status *status, char *errbuf, size_t errcap) {
    vtm_status st = VTM_OK;
    vtm_t *m;
    const uint8_t *p = (const uint8_t *)data;
    uint32_t n_instr, n_tensors, n_program;
    size_t table_end, data_start, program_bytes, program_off, data_size;
    uint32_t meta_off = 0, meta_len = 0;
    int meta_valid = 0;
    int kv_flag = 0;
    int has_kv = 0;
    uint64_t kv_bytes = 0;
    tensor_plan *plans = NULL;
    uint8_t *writable = NULL;
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

    // Порядок байт — один раз на загрузку, до разбора и до выделения арены.
    // VTM_HOST_BIG_ENDIAN — константа времени компиляции (vtm.h), поэтому в
    // little-endian сборке эта ветвь стоит ноль: компилятор её выбрасывает.
    // Проверять в циклах чтения (rd32/rd_f32, validate_words, раскладка
    // тензоров) не нужно: на big-endian сюда дело не доходит.
    if (VTM_HOST_BIG_ENDIAN) {
        if (status) *status = VTM_E_UNSUPPORTED;
        set_global_error("%s", VTM_BIG_ENDIAN_MSG);
        if (errbuf && errcap) snprintf(errbuf, errcap, "%s", VTM_BIG_ENDIAN_MSG);
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
    kv_flag = (m->header_flags & VTM_HEADER_FLAG_KV_OUT_OF_FILE) != 0;
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
    m->data_start = data_start;

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

    // Контроль целостности (задача 0066). Проверяем до разбора тензоров:
    // испорченный байт данных должен дать VTM_E_INTEGRITY с внятным текстом,
    // а не случайную ошибку формата. Старые контейнеры без блока integrity
    // читаются как раньше; VTM_SKIP_HASH=1 отключает сверку (с предупреждением).
    if (meta_valid) {
        char ierr[256];
        if (verify_integrity(p, data_start, meta_off, meta_len,
                             ierr, sizeof ierr) < 0)
            FAIL(VTM_E_INTEGRITY, "%s", ierr);
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
        // Резерв таблицы (байты 50-51) — флаги записи; старые контейнеры
        // присылают 0. Неизвестные биты игнорируются: резерв для того и есть.
        t->flags = rd16(e + 50u);
        t->n_blocks = 0;
        t->n_scales = 0;
        t->scale_block = 0;
        t->scale_is_f32 = 0;
        t->layout = VTM_LAYOUT_TRT1;

        // KV-кэш (VTM_TENSOR_FLAG_KV_CACHE): payload'а в файле нет — ни слов,
        // ни масштабов. offset/scale_offset у такой записи адресами не
        // являются, поэтому проверки границ и TRT1 ниже не применяются:
        // буфер создаётся в арене (plan_tensor/fill_tensor_planned). Старые
        // контейнеры флаг не ставят и идут прежним путём, с данными из файла.
        //
        // Новый вид контейнера (0118) отмечается флагом заголовка, и тогда
        // offset KV-кэша — VTM_KV_NO_OFFSET: файл без полезной нагрузки не
        // должен выглядеть как валидный старый контейнер. Без флага (0101 и
        // старее) принимаем прежний offset=0.
        if ((t->flags & VTM_TENSOR_FLAG_KV_CACHE) != 0) {
            uint64_t bytes;
            if (t->kind == VTM_TENSOR_TRIT)
                FAIL(VTM_E_FORMAT,
                     "тензор %u (%s): KV-кэш обязан быть fp-тензором (kind %u)",
                     i, t->name, (unsigned)t->kind);
            if (t->scale_offset != 0)
                FAIL(VTM_E_FORMAT,
                     "тензор %u (%s): у KV-кэша scale_offset=%u, ожидался 0",
                     i, t->name, t->scale_offset);
            if (kv_flag) {
                if (t->offset != VTM_KV_NO_OFFSET)
                    FAIL(VTM_E_FORMAT,
                         "тензор %u (%s): у KV-кэша offset=%u, ожидался 0x%08X "
                         "(payload'а нет; флаг KV_OUT_OF_FILE)",
                         i, t->name, t->offset, (unsigned)VTM_KV_NO_OFFSET);
            } else if (t->offset != 0) {
                FAIL(VTM_E_FORMAT,
                     "тензор %u (%s): KV-кэш без флага KV_OUT_OF_FILE, но "
                     "offset=%u (ожидался 0)",
                     i, t->name, t->offset);
            }
            // Кэп на суммарный размер: один помеченный тензор в 192-байтном
            // файле иначе просит calloc до VTM_MAX_ELEMENTS×4 байт.
            bytes = (uint64_t)t->n_elements * 4u;
            if (bytes > VTM_MAX_KV_BYTES ||
                kv_bytes > VTM_MAX_KV_BYTES - bytes)
                FAIL(VTM_E_FORMAT,
                     "KV-кэши: суммарный размер %llu байт больше предела %llu "
                     "(VTM_MAX_KV_BYTES); тензор %u (%s) — %u элементов",
                     (unsigned long long)(kv_bytes + bytes),
                     (unsigned long long)VTM_MAX_KV_BYTES,
                     i, t->name, t->n_elements);
            kv_bytes += bytes;
            has_kv = 1;
            continue;
        }

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
                {
                    uint64_t part1, part2;
                    if (mul_u64(n_scales, ss, &part1) != 0 ||
                        mul_u64(storage_blocks, 32u, &part2) != 0 ||
                        add_u64(part1, part2, &need) != 0 ||
                        add_u64(need, 18u, &need) != 0 ||
                        fits_size_t(need, &seg_bytes) != 0)
                        FAIL(VTM_E_BOUNDS, "тензор %u (%s): TRT1 слишком велик",
                             i, t->name);
                }
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

    // Флаг обещает новый вид контейнера: в нём обязан быть хотя бы один
    // KV-кэш (иначе это ошибка писателя, а не старый контейнер). Обратное не
    // требуется: старые контейнеры с кэшами внутри и 0101-контейнеры без
    // флага читаются по-прежнему, их KV-записи несут offset=0.
    if (kv_flag && !has_kv)
        FAIL(VTM_E_FORMAT,
             "флаг KV_OUT_OF_FILE поднят, но в таблице нет KV-кэшей");

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

    // --- какие тензоры интерпретатор может писать -------------------------
    // В mmap-режиме отображение PROT_READ, поэтому всё, что может стать
    // приёмником, обязано попасть в арену. KVW пишет не только dst (K-кэш),
    // но и b (V-кэш) — единственная операция с выходом не в dst. HALT dst не
    // использует (в тестах и компиляторе там 0), поэтому его не помечаем:
    // иначе тензор 0 зря копировался бы в арену.
    if (can_reference && n_tensors) {
        writable = (uint8_t *)calloc((size_t)n_tensors, 1u);
        if (!writable)
            FAIL(VTM_E_NOMEM, "не хватило памяти на карту изменяемых тензоров");
        for (i = 0; i < n_instr; i++) {
            const vtm_instr *in = &m->instr[i];
            if (in->opcode != VTM_OP_HALT && in->dst != VTM_NO_TENSOR)
                writable[in->dst] = 1;
            if (in->opcode == VTM_OP_KVW && in->b != VTM_NO_TENSOR)
                writable[in->b] = 1;
        }
    }

    // --- арена -------------------------------------------------------------
    plans = (tensor_plan *)calloc(n_tensors ? (size_t)n_tensors : 1u,
                                  sizeof(tensor_plan));
    if (!plans)
        FAIL(VTM_E_NOMEM, "не хватило памяти на планы тензоров");
    {
        size_t total = 0;
        for (i = 0; i < n_tensors; i++) {
            size_t sz, aligned;
            int w = writable ? writable[i] : 0;
            if (plan_tensor(&m->tensors[i], w, can_reference, &plans[i]) != 0)
                FAIL(VTM_E_NOMEM, "тензор %u: размер арены переполнен", i);
            sz = plans[i].arena_size;
            if (align_up(total, VTM_ARENA_ALIGN, &aligned) != 0)
                FAIL(VTM_E_NOMEM, "арена: переполнение выравнивания");
            if (sz > SIZE_MAX - aligned)
                FAIL(VTM_E_NOMEM, "арена: переполнение размера");
            total = aligned + sz;
        }
        m->arena_size = total;
        if (total) {
            uintptr_t base;
            if (total > SIZE_MAX - (VTM_ARENA_ALIGN - 1u))
                FAIL(VTM_E_NOMEM, "арена: переполнение размера");
            // calloc, а не malloc: буферы KV-кэшей (VTM_TENSOR_FLAG_KV_CACHE)
            // в файле не лежат и не заполняются — они обязаны стартовать с
            // нулей, как кэш из старого контейнера. Большие calloc'и ядро
            // отдаёт ленивыми нулевыми страницами, поэтому 112 МиБ кэша 1.5B
            // не попадают в RSS, пока KVW не начнёт в них писать.
            m->arena_raw = (uint8_t *)calloc(1, total + (VTM_ARENA_ALIGN - 1u));
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
            fill_tensor_planned(&m->tensors[i], p + data_start,
                                m->arena ? m->arena + off : NULL, &plans[i]);
            used = off + plans[i].arena_size;
        }
    }

    free(plans);
    plans = NULL;
    free(writable);
    writable = NULL;
    if (status) *status = VTM_OK;
    if (errbuf && errcap) errbuf[0] = '\0';
    set_global_error("нет ошибки");
    return m;

fail:
    free(plans);
    free(writable);
    if (status) *status = st;
    if (errbuf && errcap)
        snprintf(errbuf, errcap, "%s", m->error[0] ? m->error : g_last_error);
    vtm_free(m);
    return NULL;
#undef FAIL
}

vtm_t *vtm_load_mem(const void *data, size_t size,
                    vtm_status *status, char *errbuf, size_t errcap) {
    return load_core(data, size, 0, status, errbuf, errcap);
}

// --- загрузка из файла -----------------------------------------------------

// Прежний путь: прочитать файл целиком в malloc-буфер и разложить в арену.
// Нужен на Windows, при VTM_NO_MMAP=1 и как фолбэк, если mmap не удался.
static vtm_t *load_fread(const char *path) {
    FILE *f;
    long n;
    size_t size;
    uint8_t *buf;
    vtm_status st = VTM_OK;
    char err[256];
    vtm_t *m;

    err[0] = '\0';
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

    m = load_core(buf, size, 0, &st, err, sizeof err);
    free(buf);
    if (!m) {
        if (err[0]) set_global_error("%s: %s", path, err);
        return NULL;
    }
    return m;
}

#if !defined(_WIN32)
// После конвертации fp16/bf16 в fp32 их страницы в отображении больше не
// нужны. madvise — совет ядру: ошибка не фатальна, но сообщаем о ней, чтобы
// не обещать «память освобождена», когда это не так.
static void map_trim_copied(vtm_t *m) {
#if defined(MADV_DONTNEED)
    long ps = sysconf(_SC_PAGESIZE);
    size_t page = (ps > 0) ? (size_t)ps : 4096u;
    uintptr_t base = (uintptr_t)m->map;
    uintptr_t limit = base + (uintptr_t)m->map_size;
    uint32_t i;
    int warned = 0;

    for (i = 0; i < m->n_tensors; i++) {
        const vtm_tensor *t = &m->tensors[i];
        uintptr_t begin, end, lo, hi;
        if (t->kind != VTM_TENSOR_F16 && t->kind != VTM_TENSOR_BF16) continue;
        // KV-кэша в файле нет (offset — не адрес: 0 или VTM_KV_NO_OFFSET),
        // его страницы в отображении не занимались — трогать их madvise'ом
        // нельзя.
        if ((t->flags & VTM_TENSOR_FLAG_KV_CACHE) != 0) continue;
        begin = base + (uintptr_t)m->data_start + (uintptr_t)t->offset;
        end = begin + (uintptr_t)t->n_elements * 2u;
        lo = begin & ~(uintptr_t)(page - 1u);
        hi = (end + (uintptr_t)page - 1u) & ~(uintptr_t)(page - 1u);
        if (lo < base) lo = base;
        if (hi > limit) hi = limit;
        if (hi > lo &&
            madvise((void *)lo, (size_t)(hi - lo), MADV_DONTNEED) != 0 &&
            !warned) {
            fprintf(stderr, "VTM: madvise(MADV_DONTNEED) не удался: %s "
                            "(не фатально)\n", strerror(errno));
            warned = 1;
        }
    }
#else
    (void)m;
#endif
}

static vtm_t *load_mmap(const char *path) {
    int fd;
    struct stat st, again;
    size_t size;
    void *map;
    vtm_status stt = VTM_OK;
    char err[256];
    vtm_t *m;

    // Big-endian здесь особого случая нет: файл отображается как обычно, а
    // отказ выдаёт load_core() (VTM_BIG_ENDIAN_MSG, VTM_E_UNSUPPORTED) — одна
    // проверка на оба входа, файл и память. Отображение при этом не отдаёт
    // наружу ни одного float: загрузка обрывается до разбора тензоров.

    err[0] = '\0';
    fd = open(path, O_RDONLY);
    if (fd < 0) {
        set_global_error("не удалось открыть %s: %s", path, strerror(errno));
        return NULL;
    }
    if (fstat(fd, &st) != 0) {
        int e = errno;
        close(fd);
        set_global_error("не удалось определить размер %s: %s",
                         path, strerror(e));
        return NULL;
    }
    if (st.st_size < 0 ||
        (uint64_t)st.st_size > (uint64_t)VTM_MAX_FILE_BYTES) {
        close(fd);
        set_global_error("%s: файл больше предела %zu байт",
                         path, VTM_MAX_FILE_BYTES);
        return NULL;
    }
    size = (size_t)st.st_size;
    if (size < VTM_HEADER_SIZE) {
        close(fd);
        set_global_error("%s: файл короче заголовка: %zu < %u байт",
                         path, size, (unsigned)VTM_HEADER_SIZE);
        return NULL;
    }

    map = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) {
        int e = errno;
        close(fd);
        fprintf(stderr, "VTM: mmap(%s) не удался: %s; читаем файл в память\n",
                path, strerror(e));
        return load_fread(path);
    }

    // Файл могли усечь между fstat и mmap. Доступ к странице за новым EOF —
    // это SIGBUS, поэтому сверяем размер до разбора: лучше VTM_E_IO, чем
    // падение. Гонку «усекли во время разбора» ловит проверка ниже; она уже
    // не может предотвратить SIGBUS, но не отдаёт наружу половину модели.
    if (fstat(fd, &again) != 0 || again.st_size != st.st_size) {
        munmap(map, size);
        close(fd);
        set_global_error("%s: файл изменился во время отображения", path);
        return NULL;
    }

    m = load_core(map, size, 1, &stt, err, sizeof err);
    if (!m) {
        munmap(map, size);
        close(fd);
        if (err[0]) set_global_error("%s: %s", path, err);
        return NULL;
    }
    m->map = map;
    m->map_size = size;
    m->mapped = 1;

    if (fstat(fd, &again) != 0 || again.st_size != st.st_size) {
        close(fd);
        vtm_free(m);
        set_global_error("%s: файл изменился во время загрузки", path);
        return NULL;
    }
    close(fd);
    map_trim_copied(m);
    return m;
}
#endif // !_WIN32

vtm_t *vtm_load(const char *path) {
    if (!path) {
        set_global_error("vtm_load: path == NULL");
        return NULL;
    }
#if defined(_WIN32)
    // Недостижимо: под _WIN32 vtm.h даёт #error. Ветвь оставлена черновиком
    // будущего порта (см. docs/vtm.md, «Портируемость»).
    // Windows: mmap нет — честно сообщаем и читаем файл в память, как раньше.
    fprintf(stderr, "VTM: mmap недоступен на Windows — читаем файл в память\n");
    return load_fread(path);
#else
    if (vtm_env_flag("VTM_NO_MMAP")) {
        fprintf(stderr, "VTM: VTM_NO_MMAP=1 — mmap отключён, "
                        "читаем файл в память\n");
        return load_fread(path);
    }
    return load_mmap(path);
#endif
}

// --- доступ, состояние, освобождение --------------------------------------

void vtm_free(vtm_t *m) {
    if (!m) return;
#if !defined(_WIN32)
    if (m->map) {
        munmap(m->map, m->map_size);
        m->map = NULL;
        m->map_size = 0;
        m->mapped = 0;
    }
#endif
    vtm_samp_free(m); // буфер кандидатов SAMP (задача 0059)
    free(m->meta_json);
    free(m->tensors);
    free(m->instr);
    free(m->arena_raw);
    free(m);
}

// --- отладка загрузчика (задача 0060) --------------------------------------

int vtm_internal_file_is_mapped(const vtm_t *m) {
    return (m && m->mapped) ? 1 : 0;
}

int vtm_internal_tensor_is_mapped(const vtm_t *m, uint32_t index) {
    const vtm_tensor *t;
    const uint8_t *p;
    uintptr_t pv, base;
    if (!m || !m->map || index >= m->n_tensors) return 0;
    t = &m->tensors[index];
    if (t->kind == VTM_TENSOR_TRIT) {
        p = (const uint8_t *)(const void *)t->words;
    } else {
        p = (const uint8_t *)(const void *)t->f32;
    }
    if (!p) return 0;
    pv = (uintptr_t)p;
    base = (uintptr_t)m->map;
    return pv >= base && pv < base + (uintptr_t)m->map_size;
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
