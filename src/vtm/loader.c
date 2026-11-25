// Загрузчик .vtm: заголовок 32 байта, таблица тензоров по 52 байта (v0.2)
// или 60 байт (v0.3), секция данных с выравниванием на 32 и программа в конце.
//
// Формат v0.2 (ADR 0011): dims — u32×4, offset/scale_offset — u32.
// Формат v0.3 (ADR 0019, задача 0173): offset/scale_offset — u64, поэтому
// файл может быть >4 ГиБ; meta_offset в заголовке — u64, program_offset
// не хранится (программа последняя). Версия в заголовке выбирает ширину
// записи таблицы; старый бинарь на version=3 отказывает по версии.
// Файлы v0.1 отвергаются: ценных артефактов v0.1 нет, синтетика пересобирается.
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
#include "trit.h"
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

// --- буферы вне файла: кэп и признак нового контейнера ---------------------
//
// 0101 вынесла KV-кэши из файла: у помеченного тензора (VTM_TENSOR_FLAG_KV_CACHE)
// payload'а нет, а загрузчик выделяет буфер n_elements×4 в арене. Файл в
// 192 байта (заголовок + таблица + программа) мог запросить до
// VTM_MAX_ELEMENTS×4 = 4 ГиБ — это вектор амплификации из 0110. Сумма всех
// буферов вне файла ограничена VTM_MAX_KV_BYTES.
//
// 0176 добавила второй вид такого буфера — состояние рекуррентного слоя
// (VTM_TENSOR_FLAG_STATE): короткая свёртка CONV4 и матрица GDELTA. Для
// загрузчика он не отличается от KV-кэша: payload'а нет, буфер в арене,
// offset — маркер. Отличие только в сбросе: vtm_reset() обнуляет STATE, а
// KV-кэш по-прежнему сбрасывает хост.
//
// Почему 512 МиБ: каноническая 1.5B-раскладка (ctx 2048, kv_dim 256, 28 слоёв)
// — 112 МиБ, то есть запас ×4,5 (тот же профиль при ctx 8192); при этом
// 192-байтный файл просит полгигабайта виртуальной памяти, а не четыре.
// Меньше 256 МиБ нельзя: ctx 4096 у той же формы — уже 224 МиБ. Предел
// переопределяется на сборке: -DVTM_MAX_KV_BYTES=...
#ifndef VTM_MAX_KV_BYTES
#define VTM_MAX_KV_BYTES ((uint64_t)512u << 20)
#endif

// Флаг заголовка «буферы (KV-кэши и состояния) лежат вне файла». Бит 2
// свободен: бит 0 — fp32-масштабы (C и Python), бит 15 — блок 160
// (расширение C), бит 1 пишет Python (VFLAG_CAUSAL), C его не смотрит.
// Старые контейнеры присылают 0 и читаются прежним путём; контейнер с флагом
// обязан нести хотя бы один такой буфер.
#define VTM_HEADER_FLAG_KV_OUT_OF_FILE 0x0004u

// offset буфера в контейнере с флагом выше: payload'а нет, поэтому поле —
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

// --- kind 4 (int8) в памяти: упакован или развёрнут (задача 0170) ----------
//
// До 0170 загрузчик разворачивал int8-сегмент в fp32 в арене: таблица словаря
// 3B (151936×2048) занимала 1,24 ГиБ RSS, и каждый токен декода читал её
// целиком (36,6 мс при 34 ГБ/с — потолок ~27 ток/с). Теперь по умолчанию
// матрица остаётся упакованной: байты q — в отображении файла, в арене только
// fp32-масштабы (n_scales·4 = 9,7 МиБ у 3B). EMB разворачивает одну строку,
// lm_head считается GEMV по упакованной таблице (vtm_i8_tmv/vtm_i8_gemv).
//
// VTM_I8_EMB=0 — прежнее поведение (0044): fp32-развёртка в арене. Это и
// эталон для A/B по сырым логитам, и откат, если упакованный путь где-то не
// поддержан: тензор приходит в интерпретатор как kind 1, и весь старый код
// работает без изменений. VTM_I8_EMB=1 — наоборот, заставить упакованный путь
// даже без int8-dot (считает переносимый f32-путь).
//
// Не задано — АВТО: упакованная таблица только если в сборке и на CPU есть
// int8-dot GEMV (vtm_q4k_i8_mode() > 0, то есть NEON + FEAT_DotProd). На x86
// декодного int8-ядра нет, и там упакованная таблица означала бы GEMV
// переносимым f32-путём — это в 2 раза медленнее прежней fp32-таблицы
// (49,9 мс против 24,9 мс на форме 151936×2048). Так x86 остаётся на старом
// пути, а VTM_Q4_I8=0 (он же выключает int8-декод) заодно возвращает fp32.
// Режим фиксируется при загрузке контейнера.
static int g_i8_emb_mode = -1;

int vtm_i8_emb_mode(void) {
    if (g_i8_emb_mode < 0) {
        const char *v = getenv("VTM_I8_EMB");
        if (v && *v) g_i8_emb_mode = (strcmp(v, "0") == 0) ? 0 : 1;
        else g_i8_emb_mode = vtm_q4k_i8_mode() > 0 ? 1 : 0;
    }
    return g_i8_emb_mode;
}

int vtm_i8_emb_set_mode(int mode) {
    if (mode < 0) g_i8_emb_mode = -1;
    else g_i8_emb_mode = mode ? 1 : 0;
    return vtm_i8_emb_mode();
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
    // Задача 0172: имена новых опкодов нужны трассе, сообщениям об ошибках и
    // сверке docs/vtm.md с кодом (tools/check_vtm_docs.py). Сами операции —
    // в interp.c; загрузчик о них по-прежнему ничего не знает.
    case VTM_OP_MOE:  return "MOE";
    case VTM_OP_RMSH: return "RMSH";
    case VTM_OP_CONV4: return "CONV4";
    case VTM_OP_GDELTA: return "GDELTA";
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

static int range_ok(uint64_t offset, uint64_t len, size_t data_size) {
    if (offset > (uint64_t)data_size) return 0;
    return len <= (uint64_t)data_size - offset;
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
                            size_t meta_off, uint32_t meta_len,
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

// Размер блока GGML для kind 6 (Q4_K) / kind 7 (Q6_K): 144 и 210 байт.
static uint64_t q4k_block_bytes(uint8_t kind) {
    return kind == VTM_TENSOR_Q4K ? (uint64_t)VTM_Q4K_BLOCK_BYTES
                                  : (uint64_t)VTM_Q6K_BLOCK_BYTES;
}

static int tensor_arena_size(const vtm_tensor *t, size_t *out) {
    uint64_t n, a, b;
    if (t->kind == VTM_TENSOR_TRIT) {
        if (mul_u64(t->n_blocks, 32u, &a) != 0) return -1;
        if (mul_u64(t->n_scales, 4u, &b) != 0) return -1;
        if (add_u64(a, b, &n) != 0) return -1;
    } else if (t->kind == VTM_TENSOR_INT8) {
        // kind 4 (0170): упакованные байты q (n_elements) + fp32-масштабы
        // по блоку 128 (n_scales·4). fp32-развёртки в арене нет.
        if (mul_u64(t->n_scales, 4u, &a) != 0) return -1;
        if (add_u64(a, t->n_elements, &n) != 0) return -1;
    } else if (t->kind == VTM_TENSOR_Q4K || t->kind == VTM_TENSOR_Q6K) {
        // kind 6/7: блоки GGML как есть (144/210 байт на 256 значений).
        if (mul_u64(t->n_qblocks, q4k_block_bytes(t->kind), &n) != 0) return -1;
    } else if (t->kind == VTM_TENSOR_Q4) {
        // Худший случай: упакованные нибблы + fp32-масштабы (+ min у Q4_1).
        if (mul_u64(t->n_scales, VTM_Q4_GROUP_BYTES, &a) != 0) return -1;
        if (mul_u64(t->n_scales, 4u, &b) != 0) return -1;
        if (add_u64(a, b, &n) != 0) return -1;
        if (t->q4_is_asym != 0 && add_u64(n, b, &n) != 0) return -1;
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
// Буфер вне файла (flag VTM_TENSOR_FLAG_OUT_OF_FILE: KV-кэш или состояние
// рекуррентного слоя) — отдельный случай: содержимого в файле нет, поэтому
// копировать нечего; в арене под него выделяется буфер n_elements float
// (арена выделена calloc'ом и стартует с нулей).
typedef struct {
    size_t arena_size;  // байт в арене
    int    copy_data;   // fp32/fp16/bf16: копировать значения в арену
    int    copy_words;  // трит: копировать слова в арену
    int    copy_scales; // трит: копировать/конвертировать масштабы в арену
    int    copy_q4;     // Q4: копировать упакованные нибблы в арену
    int    copy_q4_min; // Q4_1: копировать/конвертировать min в арену
    int    copy_qblocks; // Q4_K/Q6_K: копировать блоки GGML в арену
    int    copy_i8;     // int8: копировать байты q в арену (в mmap — ссылка)
} tensor_plan;

static int plan_tensor(const vtm_tensor *t, int writable, int can_reference,
                       tensor_plan *p) {
    size_t sz;
    uint64_t kv_bytes = 0, scale_bytes = 0;

    p->arena_size = 0;
    p->copy_data = 0;
    p->copy_words = 0;
    p->copy_scales = 0;
    p->copy_q4 = 0;
    p->copy_q4_min = 0;
    p->copy_qblocks = 0;
    p->copy_i8 = 0;

    if ((t->flags & VTM_TENSOR_FLAG_OUT_OF_FILE) != 0) {
        // writable и can_reference тут ни при чём: файла нет, копировать
        // нечего. Размер считается по dims, а не по payload'у; kind проверен
        // при разборе записи таблицы.
        if (mul_u64(t->n_elements, 4u, &kv_bytes) != 0) return -1;
        if (fits_size_t(kv_bytes, &p->arena_size) != 0) return -1;
        return 0;
    }

    if (t->kind == VTM_TENSOR_INT8) {
        // kind 4 (0170): упакованная таблица вместо fp32-развёртки. Байты q
        // в mmap-режиме остаются в отображении (offsets кратны 32, а сами
        // байты читаются как int8 — выравнивание не нужно); в арену копии
        // под них нет. Масштабы всегда материализуются в fp32 в арене:
        // их 1/128 от числа значений (9,7 МиБ у таблицы 3B), зато ядро
        // логитов читает f32 без конверсии fp16 на каждую группу.
        // VTM_I8_EMB=0 — старый путь: вся матрица в fp32 в арене.
        uint64_t scales_bytes = 0, total = 0;

        if (vtm_i8_emb_mode() == 0) {
            if (mul_u64(t->n_elements, 4u, &total) != 0) return -1;
            p->copy_data = 1;
            return fits_size_t(total, &p->arena_size);
        }
        if (mul_u64(t->n_scales, 4u, &scales_bytes) != 0) return -1;
        p->copy_i8 = !(can_reference && !writable);
        p->copy_scales = 1;
        return fits_size_t((p->copy_i8 ? (uint64_t)t->n_elements : 0u)
                           + scales_bytes, &p->arena_size);
    }

    if (t->kind == VTM_TENSOR_Q4) {
        // Q4 (0151): в fp32 НЕ материализуется. В mmap-режиме упакованные
        // нибблы остаются в отображении (байты, выравнивание не нужно), а
        // fp32-масштабы/min читаются оттуда же только на платформах, где
        // допустимы невыровненные загрузки (заголовок сегмента — 18 байт,
        // поэтому они идут по смещению 2 mod 4, как у TRT1). В остальных
        // случаях в арену копируются только нибблы и/или масштабы — сама
        // матрица по-прежнему упакована, fp32-копии нет.
        uint64_t packed = 0, scales_bytes = 0, total = 0;
        int ref_packed = can_reference && !writable;
        int ref_aux = can_reference && !writable && t->scale_is_f32
                   && VTM_UNALIGNED_LOADS;

        if (mul_u64(t->n_scales, VTM_Q4_GROUP_BYTES, &packed) != 0) return -1;
        if (mul_u64(t->n_scales, 4u, &scales_bytes) != 0) return -1;
        p->copy_q4 = !ref_packed;
        p->copy_scales = !ref_aux;
        p->copy_q4_min = (t->q4_is_asym != 0 && !ref_aux);
        total = p->copy_q4 ? packed : 0;
        if (p->copy_scales) total += scales_bytes;
        if (p->copy_q4_min) total += scales_bytes;
        return fits_size_t(total, &p->arena_size);
    }

    if (t->kind == VTM_TENSOR_Q4K || t->kind == VTM_TENSOR_Q6K) {
        // kind 6/7 (0165): блоки GGML остаются байтами — ровно как в файле.
        // Блоки выровнены на 2 (d fp16), а не на 4/8, поэтому в mmap их
        // можно ссылать только при VTM_UNALIGNED_LOADS или на ARM64 (где
        // memcpy-чтение fp16 компилируется в обычный ldrh); иначе — копия в
        // арену. fp32-материализации нет ни в одном случае.
        uint64_t bytes = 0;
        if (mul_u64(t->n_qblocks, q4k_block_bytes(t->kind), &bytes) != 0)
            return -1;
        p->copy_qblocks = !(can_reference && !writable && VTM_UNALIGNED_LOADS);
        return fits_size_t(p->copy_qblocks ? bytes : 0, &p->arena_size);
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
    // int8 (0044) разворачивается в fp32 при загрузке — mmap не годится.
    if (t->kind == VTM_TENSOR_F16 || t->kind == VTM_TENSOR_BF16 ||
        t->kind == VTM_TENSOR_INT8) {
        if (tensor_arena_size(t, &sz) != 0) return -1;
        p->arena_size = sz;
        p->copy_data = 1;
        return 0;
    }

    // Триты в native-хранении (задача 0057): ядра читают только pack20,
    // поэтому такой тензор всегда переупаковывается в арену. Выигрыша mmap
    // для него нет, зато интерпретатор о хранении не знает.
    if (t->kind == VTM_TENSOR_TRIT &&
        vtm_trit_storage_from_flags(t->flags) != VTM_TRIT_STORAGE_PACKED20) {
        if (tensor_arena_size(t, &sz) != 0) return -1;
        p->arena_size = sz;
        p->copy_data = 1;
        p->copy_words = 1;
        p->copy_scales = 1;
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
        return data + t->offset64 + 18u + (size_t)t->n_scales * ss;
    }
    return data + t->offset64;
}

static const uint8_t *trit_scales_ptr(const vtm_tensor *t, const uint8_t *data) {
    if (t->layout == VTM_LAYOUT_TRT1) return data + t->offset64 + 18u;
    return data + t->scale_offset64;
}

// Разложить тензор по арене (copy_*) или выставить указатели в отображение.
// data — начало секции данных (data_start уже прибавлен вызывающим).
// Для тритов слова идут первыми (выровнены на 64), затем масштабы.
// 0 — успех; -1 — данные не сошлись с проверенным при разборе заголовком
// (native-трит вне -1..1 и т. п.); вызывающий обязан вернуть ошибку, а не
// молча отдать интерпретатору мусор.
static int fill_tensor_planned(vtm_tensor *t, const uint8_t *data,
                               uint8_t *dst, const tensor_plan *p) {
    if ((t->flags & VTM_TENSOR_FLAG_OUT_OF_FILE) != 0) {
        // Копировать нечего: буфер уже в арене и обнулён calloc'ом (см.
        // выделение арены). Нули — это и есть «пустой кэш»: ровно то же
        // состояние, что давал старый контейнер, где в файле лежали нули.
        t->f32 = (float *)(void *)dst;
        return 0;
    }
    if (t->kind == VTM_TENSOR_TRIT) {
        uint32_t nb = t->n_blocks;
        uint32_t ns = t->n_scales;
        uint32_t i;
        unsigned storage = vtm_trit_storage_from_flags(t->flags);
        const uint8_t *sp = trit_scales_ptr(t, data);
        const uint8_t *wp = trit_words_ptr(t, data);

        if (p->copy_words) {
            uint32_t *words = (uint32_t *)(void *)dst;
            if (storage == VTM_TRIT_STORAGE_PACKED20) {
                uint32_t w;
                for (i = 0; i < nb; i++) {
                    for (w = 0; w < VTM_WORDS_PER_BLOCK; w++) {
                        words[i * VTM_WORDS_PER_BLOCK + w] =
                            rd32(wp + ((size_t)i * VTM_WORDS_PER_BLOCK + w) * 4u);
                    }
                }
            } else {
                // native: байты -> слова pack20. Размер выхода — nb блоков
                // по 8 слов, вход — ceil(n/4) слов хранения.
                vtm_trit_store store;
                uint32_t src_words = vtm_trit_storage_words(
                    t->n_elements, VTM_TRIT_STORAGE_NATIVE);
                uint32_t dst_words = nb * VTM_WORDS_PER_BLOCK;
                if (vtm_trit_store_init(&store, storage, wp, src_words,
                                        t->n_elements, NULL, 0, 0) != 0)
                    return -1;
                if (vtm_trit_store_to_packed20(&store, words, dst_words) != 0)
                    return -1;
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
    } else if (t->kind == VTM_TENSOR_Q4) {
        // Q4 (0151): нибблы остаются упакованными; fp32-копии нет. Указатель
        // на байты — words (так его ждут ядра 0152: q[g*16 + (i & 15)]),
        // масштабы — scales, у Q4_1 ещё min'ы q4_mins. В mmap-режиме
        // упакованные байты (и fp32-масштабы/min при VTM_UNALIGNED_LOADS)
        // ссылаются прямо в отображение.
        const uint8_t *seg = data + t->offset64;
        size_t ss = t->scale_is_f32 ? 4u : 2u;
        size_t groups = t->n_scales;
        size_t header_size = t->q4_is_asym
                           ? VTM_Q4_ASYM_HEADER_SIZE : VTM_Q4_HEADER_SIZE;
        const uint8_t *sp = seg + header_size;
        const uint8_t *mp = sp + groups * ss;
        const uint8_t *qp = mp + (t->q4_is_asym ? groups * ss : 0u);
        uint8_t *cur = dst;
        uint32_t i;

        if (p->copy_q4) {
            memcpy(cur, qp, groups * VTM_Q4_GROUP_BYTES);
            t->words = (const uint32_t *)(const void *)cur;
            cur += groups * VTM_Q4_GROUP_BYTES;
        } else {
            t->words = (const uint32_t *)(const void *)qp;
        }
        if (p->copy_scales) {
            float *scales = (float *)(void *)cur;
            for (i = 0; i < t->n_scales; i++) {
                scales[i] = t->scale_is_f32
                          ? rd_f32(sp + (size_t)i * 4u)
                          : vtm_half_to_float(rd16(sp + (size_t)i * 2u));
            }
            t->scales = scales;
            cur += groups * 4u;
        } else {
            t->scales = (const float *)(const void *)sp;
        }
        if (t->q4_is_asym) {
            if (p->copy_q4_min) {
                float *mins = (float *)(void *)cur;
                for (i = 0; i < t->n_scales; i++) {
                    mins[i] = t->scale_is_f32
                            ? rd_f32(mp + (size_t)i * 4u)
                            : vtm_half_to_float(rd16(mp + (size_t)i * 2u));
                }
                t->q4_mins = mins;
            } else {
                t->q4_mins = (const float *)(const void *)mp;
            }
        } else {
            t->q4_mins = NULL;
        }
        t->f32 = NULL;
    } else if (t->kind == VTM_TENSOR_Q4K || t->kind == VTM_TENSOR_Q6K) {
        // kind 6/7 (0165): блоки GGML как есть — ни распаковки, ни fp32-копии.
        // В mmap-режиме и при VTM_UNALIGNED_LOADS указатель смотрит в
        // отображение (блоки читаются через memcpy — выравнивание не нужно),
        // иначе блоки лежат в арене (calloc'нутый буфер уже выровнен).
        const uint8_t *seg = data + t->offset64;
        size_t bytes = (size_t)t->n_qblocks * q4k_block_bytes(t->kind);
        if (p->copy_qblocks) {
            memcpy(dst, seg + VTM_Q4K_HEADER_SIZE, bytes);
            t->qblocks = dst;
        } else {
            t->qblocks = seg + VTM_Q4K_HEADER_SIZE;
        }
        t->f32 = NULL;
    } else if (t->kind == VTM_TENSOR_F32) {
        if (p->copy_data) {
            float *y = (float *)(void *)dst;
            uint32_t i;
            const uint8_t *src = data + t->offset64;
            for (i = 0; i < t->n_elements; i++)
                y[i] = rd_f32(src + (size_t)i * 4u);
            t->f32 = y;
        } else {
            // Немодифицируемый fp32 лежит в отображении как есть: offset
            // кратен 32, отображение выровнено на страницу, float* валиден.
            t->f32 = (float *)(void *)(uintptr_t)(data + t->offset64);
        }
    } else if (t->kind == VTM_TENSOR_F16) {
        float *y = (float *)(void *)dst;
        uint32_t i;
        const uint8_t *src = data + t->offset64;
        for (i = 0; i < t->n_elements; i++)
            y[i] = vtm_half_to_float(rd16(src + (size_t)i * 2u));
        t->f32 = y;
    } else if (t->kind == VTM_TENSOR_INT8) {
        // kind 4 (задача 0044; упаковано с 0170). Два режима:
        //
        //  * по умолчанию таблица остаётся int8: t->i8 указывает на байты q
        //    (в mmap — прямо в отображение), t->scales — на fp32-масштабы в
        //    арене. fp32-копии нет: EMB разворачивает строку, lm_head —
        //    GEMV по упакованной таблице (see op_emb/op_mm в interp.c);
        //  * VTM_I8_EMB=0 — прежний путь: развернуть в fp32 в арене, тензор
        //    становится kind 1, источник помечается
        //    VTM_TENSOR_FLAG_INT8_SOURCE (в файл флаг не пишется).
        const uint8_t *src = data + t->offset64;
        const uint8_t *sp = src + 18u;
        size_t ss = t->scale_is_f32 ? 4u : 2u;
        const uint8_t *qp = sp + (size_t)t->n_scales * ss;
        uint32_t i;

        if (vtm_i8_emb_mode() == 0) {
            float *y = (float *)(void *)dst;
            for (i = 0; i < t->n_elements; i++) {
                uint32_t block = i / t->scale_block;
                float scale = t->scale_is_f32
                            ? rd_f32(sp + (size_t)block * 4u)
                            : vtm_half_to_float(rd16(sp + (size_t)block * 2u));
                y[i] = (float)((int)(int8_t)qp[i] * scale);
            }
            t->f32 = y;
            t->i8 = NULL;
            t->flags |= VTM_TENSOR_FLAG_INT8_SOURCE;
            t->kind = VTM_TENSOR_F32;
            return 0;
        }
        {
            float *scales = (float *)(void *)dst;
            for (i = 0; i < t->n_scales; i++) {
                scales[i] = t->scale_is_f32
                          ? rd_f32(sp + (size_t)i * 4u)
                          : vtm_half_to_float(rd16(sp + (size_t)i * 2u));
            }
            t->scales = scales;
            if (p->copy_i8) {
                int8_t *q = (int8_t *)(void *)(dst + (size_t)t->n_scales * 4u);
                memcpy(q, qp, t->n_elements);
                t->i8 = q;
            } else {
                t->i8 = (const int8_t *)(const void *)qp;
            }
            t->f32 = NULL;
        }
    } else { // BF16
        float *y = (float *)(void *)dst;
        uint32_t i;
        const uint8_t *src = data + t->offset64;
        for (i = 0; i < t->n_elements; i++)
            y[i] = vtm_bf16_to_float(rd16(src + (size_t)i * 2u));
        t->f32 = y;
    }
    return 0;
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
    uint16_t version;
    size_t entry_size, table_end, data_start, program_bytes, program_off, data_size;
    uint64_t meta_off64 = 0;
    size_t meta_off = 0;
    uint32_t meta_len = 0;
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
    // Дескрипторов нет, пока vtm_load() их не отдал (0175). Именно -1, а не
    // 0 из calloc: 0 — валидный fd, и «нет дескриптора» должно отличаться.
    m->fd = -1;
    m->moe_fd = 0;
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
    version = rd16(p + 4);
    if (version == VTM_FORMAT_VERSION_V01)
        FAIL(VTM_E_VERSION, "формат v0.1 не поддерживается");
    if (version != VTM_FORMAT_VERSION_V02 &&
        version != VTM_FORMAT_VERSION_V03)
        FAIL(VTM_E_VERSION,
             "версия контейнера %u, ожидали %u (v0.2) или %u (v0.3)",
             (unsigned)version,
             (unsigned)VTM_FORMAT_VERSION_V02,
             (unsigned)VTM_FORMAT_VERSION_V03);
    // У v0.2 смещения u32 — файл в 4 ГиБ и больше он не представляет.
    // Проверяем после версии: v0.3-файл отвергается не «по размеру», а
    // внятным «версия 3» старым бинарём и грузится новым.
    if (version == VTM_FORMAT_VERSION_V02 &&
        size >= (size_t)VTM_MAX_FILE_BYTES_V02)
        FAIL(VTM_E_BOUNDS,
             "файл v0.2 (%zu байт) больше предела 4 ГиБ: смещения u32; "
             "нужен контейнер v0.3 (u64-смещения, ADR 0019)", size);
    entry_size = (version == VTM_FORMAT_VERSION_V03)
               ? (size_t)VTM_TENSOR_ENTRY_SIZE_V03
               : (size_t)VTM_TENSOR_ENTRY_SIZE;

    m->header_flags = rd16(p + 6);
    kv_flag = (m->header_flags & VTM_HEADER_FLAG_KV_OUT_OF_FILE) != 0;
    n_instr = rd32(p + 8);
    n_tensors = rd32(p + 12);
    n_program = rd32(p + 16);

    if (n_tensors > VTM_MAX_TENSORS)
        FAIL(VTM_E_FORMAT, "n_tensors=%u больше предела %u",
             n_tensors, (unsigned)VTM_MAX_TENSORS);
    if ((uint64_t)n_tensors * (uint64_t)entry_size >
        (uint64_t)(size - 32))
        FAIL(VTM_E_TRUNCATED,
             "таблица тензоров (%u × %u) не влезает в файл", n_tensors,
             (unsigned)entry_size);
    table_end = 32u + (size_t)n_tensors * entry_size;
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
    // v0.2: program_offset u32, meta_offset u32, meta_len u32 (байты 20/24/28).
    // v0.3: program_offset не хранится (он всегда file_size - n_instr*16,
    // программа последняя), зато meta_offset — u64 в байтах 20..27, meta_len —
    // u32 в 28..31. Иначе мета за 4 ГиБ была бы недостижима.
    // Если смещения врут или указывают мимо программы — это битая мета, а не
    // битый файл: мету просто не читаем, данные трактуем до программы.
    if (version == VTM_FORMAT_VERSION_V03) {
        meta_off64 = rd64(p + 20);
        meta_len = rd32(p + 28);
    } else {
        meta_off64 = (uint64_t)rd32(p + 24);
        meta_len = rd32(p + 28);
    }
    if (meta_off64 != 0 && meta_len != 0 &&
        meta_off64 >= (uint64_t)data_start &&
        meta_off64 <= (uint64_t)program_off &&
        (uint64_t)meta_len <= (uint64_t)program_off - meta_off64) {
        meta_valid = 1;
        meta_off = (size_t)meta_off64;
        data_size = meta_off - data_start;
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
        const uint8_t *e = p + 32u + (size_t)i * entry_size;
        vtm_tensor *t = &m->tensors[i];
        const uint8_t *nul = (const uint8_t *)memchr(e, 0, 24);
        uint64_t n_el = 1;
        uint32_t d;

        if (!nul)
            FAIL(VTM_E_FORMAT, "тензор %u: имя из 24 байт без NUL", i);
        memcpy(t->name, e, 24);
        t->kind = e[24];
        t->ndim = e[25];
        // kind 4 (int8 + масштабы, задача 0044), kind 5 (Q4, 0151) и kind 6/7
        // (Q4_K/Q6_K, 0165) — известные виды. Старый загрузчик отвергал бы
        // kind > 4 как «неизвестный kind» ещё до чтения данных (tripwire).
        if (t->kind != VTM_TENSOR_INT8 && t->kind > VTM_TENSOR_Q6K)
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
        // v0.2: offset u32 (42), scale_offset u32 (46), flags u16 (50).
        // v0.3: offset u64 (42..49), scale_offset u64 (50..57), flags u16 (58).
        if (version == VTM_FORMAT_VERSION_V03) {
            t->offset64 = rd64(e + 42u);
            t->scale_offset64 = rd64(e + 50u);
            t->flags = rd16(e + 58u);
        } else {
            t->offset64 = (uint64_t)rd32(e + 42u);
            t->scale_offset64 = (uint64_t)rd32(e + 46u);
            t->flags = rd16(e + 50u);
        }
        // ABI-поля: младшие 32 бита. Для v0.3 полный адрес — только в
        // offset64/scale_offset64; main.c печатает старое поле как раньше.
        t->offset = (uint32_t)t->offset64;
        t->scale_offset = (uint32_t)t->scale_offset64;
        t->n_blocks = 0;
        t->n_scales = 0;
        t->scale_block = 0;
        t->scale_is_f32 = 0;
        t->qblocks = NULL;
        t->i8 = NULL;
        t->n_qblocks = 0;
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
        if ((t->flags & VTM_TENSOR_FLAG_OUT_OF_FILE) != 0) {
            uint64_t bytes;
            if (t->kind == VTM_TENSOR_TRIT || t->kind == VTM_TENSOR_INT8 ||
                t->kind == VTM_TENSOR_Q4 || t->kind == VTM_TENSOR_Q4K ||
                t->kind == VTM_TENSOR_Q6K)
                FAIL(VTM_E_FORMAT,
                     "тензор %u (%s): KV-кэш обязан быть fp-тензором (kind %u)",
                     i, t->name, (unsigned)t->kind);
            if (t->scale_offset64 != 0)
                FAIL(VTM_E_FORMAT,
                     "тензор %u (%s): у KV-кэша scale_offset=%llu, ожидался 0",
                     i, t->name, (unsigned long long)t->scale_offset64);
            if (kv_flag) {
                if (t->offset64 != VTM_KV_NO_OFFSET)
                    FAIL(VTM_E_FORMAT,
                         "тензор %u (%s): у KV-кэша offset=%llu, ожидался 0x%08X "
                         "(payload'а нет; флаг KV_OUT_OF_FILE)",
                         i, t->name, (unsigned long long)t->offset64,
                         (unsigned)VTM_KV_NO_OFFSET);
            } else if (t->offset64 != 0) {
                FAIL(VTM_E_FORMAT,
                     "тензор %u (%s): KV-кэш без флага KV_OUT_OF_FILE, но "
                     "offset=%llu (ожидался 0)",
                     i, t->name, (unsigned long long)t->offset64);
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

        if ((t->offset64 % VTM_DATA_ALIGN) != 0)
            FAIL(VTM_E_FORMAT, "тензор %u (%s): offset=%llu не кратен %u",
                 i, t->name, (unsigned long long)t->offset64,
                 (unsigned)VTM_DATA_ALIGN);
        if ((uint64_t)t->offset64 >= (uint64_t)data_size)
            FAIL(VTM_E_BOUNDS,
                 "тензор %u (%s): offset=%llu за концом данных (%zu)",
                 i, t->name, (unsigned long long)t->offset64, data_size);

        if (t->kind == VTM_TENSOR_TRIT) {
            uint64_t need;
            // Хранение тритов (задача 0057) продублировано в flags записи:
            // биты 1..2. У всех контейнеров до 0057 там 0 — packed20.
            unsigned entry_storage = vtm_trit_storage_from_flags(t->flags);
            if (entry_storage > VTM_TRIT_STORAGE_MAX)
                FAIL(VTM_E_FORMAT,
                     "тензор %u (%s): неизвестное хранение тритов %u в flags",
                     i, t->name, entry_storage);
            if (t->scale_offset64 == VTM_NO_OFFSET) {
                const uint8_t *seg = p + data_start + t->offset64;
                uint64_t n_trits;
                uint32_t n_scales, storage_blocks, n128, n160, n_words;
                uint32_t scale_block;
                unsigned storage = VTM_TRIT_STORAGE_PACKED20;
                size_t ss, seg_bytes;
                char werr[160];

                if (!range_ok(t->offset64, 18u, data_size))
                    FAIL(VTM_E_BOUNDS,
                         "тензор %u (%s): TRT1-заголовок за границей данных",
                         i, t->name);
                if (memcmp(seg, "TRT1", 4) != 0)
                    FAIL(VTM_E_FORMAT,
                         "тензор %u (%s): kind=TRIT без TRT1 и без scale_offset",
                         i, t->name);
                // v1 — только packed20 (так писали все контейнеры до 0057).
                // v2 — хранение в битах 1..2 flags. Отдельная версия нужна,
                // чтобы загрузчик до 0057 отверг native с VTM_E_VERSION, а не
                // прочитал байты как слова pack20 (тихая порча весов).
                if (seg[4] != 1 && seg[4] != 2)
                    FAIL(VTM_E_VERSION,
                         "тензор %u (%s): версия TRT1 %u", i, t->name,
                         (unsigned)seg[4]);
                if (seg[4] == 2) {
                    unsigned bits;
                    if ((seg[5] & (uint8_t)~0x07u) != 0)
                        FAIL(VTM_E_FORMAT,
                             "тензор %u (%s): неизвестные флаги TRT1 0x%02x",
                             i, t->name, (unsigned)seg[5]);
                    bits = (unsigned)((seg[5] >> 1) & 0x03u);
                    if (bits > VTM_TRIT_STORAGE_MAX)
                        FAIL(VTM_E_FORMAT,
                             "тензор %u (%s): неизвестное хранение тритов %u",
                             i, t->name, bits);
                    storage = bits;
                }
                if (entry_storage != storage)
                    FAIL(VTM_E_FORMAT,
                         "тензор %u (%s): хранение в flags (%u) не совпадает "
                         "с TRT1 (%u)", i, t->name, entry_storage, storage);
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
                // Длина массива тритов зависит от хранения: pack20 округляет
                // до 160 тритов (32 байта), native — до 4 тритов (слово).
                n_words = vtm_trit_storage_words((uint32_t)n_trits, storage);
                {
                    uint64_t part1, part2;
                    if (mul_u64(n_scales, ss, &part1) != 0 ||
                        mul_u64(n_words, 4u, &part2) != 0 ||
                        add_u64(part1, part2, &need) != 0 ||
                        add_u64(need, 18u, &need) != 0 ||
                        fits_size_t(need, &seg_bytes) != 0)
                        FAIL(VTM_E_BOUNDS, "тензор %u (%s): TRT1 слишком велик",
                             i, t->name);
                }
                if (!range_ok(t->offset64, (uint64_t)seg_bytes, data_size))
                    FAIL(VTM_E_BOUNDS,
                         "тензор %u (%s): TRT1-сегмент (%zu байт) за границей",
                         i, t->name, seg_bytes);
                if (storage == VTM_TRIT_STORAGE_PACKED20) {
                    if (validate_words(seg + 18u + (size_t)n_scales * ss,
                                       n_words, werr, sizeof werr) != 0)
                        FAIL(VTM_E_SCALE, "тензор %u (%s): %s",
                             i, t->name, werr);
                } else {
                    uint32_t bad = 0;
                    // Проверяем ровно n_trits байт: хвост последнего слова —
                    // выравнивающий padding, ядра его не читают (Python-
                    // эталон проверяет так же).
                    if (vtm_trit_validate_native(
                            seg + 18u + (size_t)n_scales * ss,
                            (uint32_t)n_trits, &bad) != 0)
                        FAIL(VTM_E_SCALE,
                             "тензор %u (%s): native-трит #%u вне -1..1",
                             i, t->name, bad);
                }
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
                if (entry_storage != VTM_TRIT_STORAGE_PACKED20)
                    FAIL(VTM_E_FORMAT,
                         "тензор %u (%s): raw-раскладка несёт только packed20 "
                         "(flags=%u)", i, t->name, entry_storage);
                storage_blocks = (uint32_t)(((uint64_t)t->n_elements
                                             + VTM_TRITS_PER_BLOCK - 1u)
                                            / VTM_TRITS_PER_BLOCK);
                n_scales = (uint32_t)(((uint64_t)t->n_elements
                                       + scale_block - 1u) / scale_block);
                if (storage_blocks == 0 || n_scales == 0)
                    FAIL(VTM_E_FORMAT, "тензор %u (%s): пустой raw-трит",
                         i, t->name);
                if ((t->scale_offset64 % VTM_DATA_ALIGN) != 0)
                    FAIL(VTM_E_FORMAT,
                         "тензор %u (%s): scale_offset=%llu не кратен %u",
                         i, t->name, (unsigned long long)t->scale_offset64,
                         (unsigned)VTM_DATA_ALIGN);
                ss = (m->header_flags & VTM_HEADER_FLAG_F32_SCALES) ? 4u : 2u;
                if (!range_ok(t->offset64,
                              (uint64_t)storage_blocks * 32u, data_size))
                    FAIL(VTM_E_BOUNDS,
                         "тензор %u (%s): слова тритов за границей данных",
                         i, t->name);
                if (!range_ok(t->scale_offset64,
                              (uint64_t)n_scales * ss, data_size))
                    FAIL(VTM_E_BOUNDS,
                         "тензор %u (%s): масштабы за границей данных",
                         i, t->name);
                if (validate_words(p + data_start + t->offset64,
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
        } else if (t->kind == VTM_TENSOR_INT8) {
            // Самодостаточный сегмент kind 4 (задача 0044): 18-байтовый
            // заголовок "INT8" (как у TRT1), масштабы, затем n_elements
            // знаковых байт q. w = q · scale, масштаб на блок 128.
            const uint8_t *seg = p + data_start + t->offset64;
            uint64_t n_values;
            uint32_t n_scales, expected_scales, k;
            size_t ss, seg_bytes;
            if (t->scale_offset64 != VTM_NO_OFFSET)
                FAIL(VTM_E_FORMAT,
                     "тензор %u (%s): kind=int8 требует scale_offset=0x%08X "
                     "(масштабы внутри сегмента), а пришло %llu",
                     i, t->name, (unsigned)VTM_NO_OFFSET,
                     (unsigned long long)t->scale_offset64);
            if (!range_ok(t->offset64, 18u, data_size))
                FAIL(VTM_E_BOUNDS,
                     "тензор %u (%s): INT8-заголовок за границей данных",
                     i, t->name);
            if (memcmp(seg, "INT8", 4) != 0)
                FAIL(VTM_E_FORMAT,
                     "тензор %u (%s): kind=int8 без сегмента INT8", i, t->name);
            if (seg[4] != 1)
                FAIL(VTM_E_VERSION,
                     "тензор %u (%s): версия INT8-сегмента %u", i, t->name,
                     (unsigned)seg[4]);
            if ((seg[5] & (uint8_t)~0x01u) != 0)
                FAIL(VTM_E_FORMAT,
                     "тензор %u (%s): неизвестные флаги INT8 0x%02x", i,
                     t->name, (unsigned)seg[5]);
            n_values = rd64(seg + 6);
            n_scales = rd32(seg + 14);
            if (n_values != (uint64_t)t->n_elements)
                FAIL(VTM_E_FORMAT,
                     "тензор %u (%s): INT8 n_values=%llu, dims дают %u",
                     i, t->name, (unsigned long long)n_values, t->n_elements);
            expected_scales = (uint32_t)((n_values + VTM_SCALE_BLOCK_128 - 1u)
                                         / VTM_SCALE_BLOCK_128);
            if (n_scales != expected_scales)
                FAIL(VTM_E_FORMAT,
                     "тензор %u (%s): INT8 n_scale=%u, ожидали %u (блок %u)",
                     i, t->name, n_scales, expected_scales,
                     (unsigned)VTM_SCALE_BLOCK_128);
            ss = (seg[5] & 1u) ? 4u : 2u;
            {
                uint64_t part1, part2;
                if (mul_u64(n_scales, ss, &part1) != 0 ||
                    mul_u64(n_values, 1u, &part2) != 0 ||
                    add_u64(part1, part2, &part2) != 0 ||
                    add_u64(part2, 18u, &part2) != 0 ||
                    fits_size_t(part2, &seg_bytes) != 0)
                    FAIL(VTM_E_BOUNDS, "тензор %u (%s): INT8 слишком велик",
                         i, t->name);
            }
            if (!range_ok(t->offset64, (uint64_t)seg_bytes, data_size))
                FAIL(VTM_E_BOUNDS,
                     "тензор %u (%s): INT8-сегмент (%zu байт) за границей",
                     i, t->name, seg_bytes);
            for (k = 0; k < n_scales; k++) {
                float scale = (seg[5] & 1u)
                            ? rd_f32(seg + 18u + (size_t)k * 4u)
                            : vtm_half_to_float(rd16(seg + 18u + (size_t)k * 2u));
                if (!isfinite(scale) || scale < 0.0f)
                    FAIL(VTM_E_SCALE,
                         "тензор %u (%s): масштаб #%u не конечный/отрицательный",
                         i, t->name, k);
            }
            {
                const uint8_t *q = seg + 18u + (size_t)n_scales * ss;
                for (k = 0; k < t->n_elements; k++) {
                    if (q[k] == 0x80u)
                        FAIL(VTM_E_SCALE,
                             "тензор %u (%s): int8 q #%u = -128 вне -127..127",
                             i, t->name, k);
                }
            }
            t->n_blocks = 0;
            t->n_scales = n_scales;
            t->scale_block = VTM_SCALE_BLOCK_128;
            t->scale_is_f32 = (uint8_t)((seg[5] & 1u) ? 1u : 0u);
            t->layout = VTM_LAYOUT_RAW;
        } else if (t->kind == VTM_TENSOR_Q4) {
            // Самодостаточный Q4-сегмент kind 5 (задача 0151): заголовок 18 байт
            // (Q4_1 — 22: n_mins), масштабы (fp32/fp16), у Q4_1 ещё min на группу, затем
            // n_groups * 16 упакованных байт (32 значения на группу, GGML:
            // байт j = значение j + значение j+16). В fp32 НЕ расквантовываем:
            // t->words указывает на упакованные байты, t->scales — на fp32-
            // масштабы, t->q4_mins — на min (Q4_1).
            const uint8_t *seg = p + data_start + t->offset64;
            uint64_t n_values, total;
            uint32_t n_groups, expected_groups, n_mins, k;
            size_t header_size, ss, scales_bytes, mins_bytes, packed_bytes;
            int asym;

            if (t->scale_offset64 != VTM_NO_OFFSET)
                FAIL(VTM_E_FORMAT,
                     "тензор %u (%s): kind=Q4 требует scale_offset=0x%08X "
                     "(масштабы внутри сегмента), а пришло %llu",
                     i, t->name, (unsigned)VTM_NO_OFFSET,
                     (unsigned long long)t->scale_offset64);
            if (!range_ok(t->offset64, VTM_Q4_HEADER_SIZE, data_size))
                FAIL(VTM_E_BOUNDS,
                     "тензор %u (%s): Q4-заголовок за границей данных",
                     i, t->name);
            if (memcmp(seg, VTM_Q4_MAGIC_SYM, 4) == 0) {
                asym = 0;
            } else if (memcmp(seg, VTM_Q4_MAGIC_ASYM, 4) == 0) {
                asym = 1;
            } else {
                FAIL(VTM_E_FORMAT,
                     "тензор %u (%s): kind=Q4 без сегмента Q4_0/Q4_1",
                     i, t->name);
            }
            if (seg[4] != VTM_Q4_VERSION)
                FAIL(VTM_E_VERSION,
                     "тензор %u (%s): версия Q4-сегмента %u", i, t->name,
                     (unsigned)seg[4]);
            if ((seg[5] & (uint8_t)~VTM_Q4_FLAG_F32) != 0)
                FAIL(VTM_E_FORMAT,
                     "тензор %u (%s): неизвестные флаги Q4 0x%02x", i,
                     t->name, (unsigned)seg[5]);
            n_values = rd64(seg + 6);
            n_groups = rd32(seg + 14);
            if (n_values != (uint64_t)t->n_elements)
                FAIL(VTM_E_FORMAT,
                     "тензор %u (%s): Q4 n_values=%llu, dims дают %u",
                     i, t->name, (unsigned long long)n_values, t->n_elements);
            expected_groups = (uint32_t)(((uint64_t)t->n_elements
                                          + VTM_Q4_SCALE_BLOCK - 1u)
                                         / VTM_Q4_SCALE_BLOCK);
            if (n_groups != expected_groups)
                FAIL(VTM_E_FORMAT,
                     "тензор %u (%s): Q4 n_groups=%u, ожидали %u (блок %u)",
                     i, t->name, n_groups, expected_groups,
                     (unsigned)VTM_Q4_SCALE_BLOCK);
            header_size = VTM_Q4_HEADER_SIZE;
            if (asym) {
                // Q4_1 несёт явное n_mins: без него отсутствие min в
                // обрезанном сегменте не отличить от выравнивающего padding'а.
                if (!range_ok(t->offset64, VTM_Q4_ASYM_HEADER_SIZE, data_size))
                    FAIL(VTM_E_FORMAT,
                         "тензор %u (%s): Q4_1 без n_mins — тихо считать "
                         "Q4_0 нельзя", i, t->name);
                n_mins = rd32(seg + VTM_Q4_HEADER_SIZE);
                if (n_mins != expected_groups)
                    FAIL(VTM_E_FORMAT,
                         "тензор %u (%s): Q4_1 n_mins=%u, ожидали %u "
                         "(по одному min на группу %u) — тихо считать "
                         "Q4_0 нельзя", i, t->name, n_mins, expected_groups,
                         (unsigned)VTM_Q4_SCALE_BLOCK);
                header_size = VTM_Q4_ASYM_HEADER_SIZE;
            }
            ss = (seg[5] & VTM_Q4_FLAG_F32) ? 4u : 2u;
            if (mul_u64(n_groups, ss, &total) != 0 ||
                fits_size_t(total, &scales_bytes) != 0)
                FAIL(VTM_E_BOUNDS, "тензор %u (%s): Q4 слишком велик",
                     i, t->name);
            mins_bytes = asym ? scales_bytes : 0;
            if (mul_u64(n_groups, VTM_Q4_GROUP_BYTES, &total) != 0 ||
                fits_size_t(total, &packed_bytes) != 0)
                FAIL(VTM_E_BOUNDS, "тензор %u (%s): Q4 слишком велик",
                     i, t->name);
            if (asym &&
                !range_ok(t->offset64,
                          (uint64_t)header_size + scales_bytes
                          + mins_bytes, data_size))
                FAIL(VTM_E_FORMAT,
                     "тензор %u (%s): Q4_1 без min на группу (нужно %u "
                     "значений по %zu байт) — тихо считать Q4_0 нельзя",
                     i, t->name, n_groups, ss);
            if (add_u64((uint64_t)header_size, scales_bytes, &total) != 0 ||
                add_u64(total, mins_bytes, &total) != 0 ||
                add_u64(total, packed_bytes, &total) != 0)
                FAIL(VTM_E_BOUNDS, "тензор %u (%s): Q4 слишком велик",
                     i, t->name);
            if (!range_ok(t->offset64, total, data_size))
                FAIL(VTM_E_BOUNDS,
                     "тензор %u (%s): Q4-сегмент (%llu байт) за границей",
                     i, t->name, (unsigned long long)total);
            for (k = 0; k < n_groups; k++) {
                float scale = (seg[5] & VTM_Q4_FLAG_F32)
                            ? rd_f32(seg + header_size
                                     + (size_t)k * 4u)
                            : vtm_half_to_float(rd16(seg + header_size
                                                     + (size_t)k * 2u));
                if (!isfinite(scale) || scale < 0.0f)
                    FAIL(VTM_E_SCALE,
                         "тензор %u (%s): Q4-масштаб #%u не конечный/"
                         "отрицательный", i, t->name, k);
            }
            if (asym) {
                const uint8_t *mp = seg + header_size + scales_bytes;
                for (k = 0; k < n_groups; k++) {
                    float mn = (seg[5] & VTM_Q4_FLAG_F32)
                             ? rd_f32(mp + (size_t)k * 4u)
                             : vtm_half_to_float(rd16(mp + (size_t)k * 2u));
                    if (!isfinite(mn))
                        FAIL(VTM_E_SCALE,
                             "тензор %u (%s): Q4_1 min #%u не конечный",
                             i, t->name, k);
                }
            }
            t->n_blocks = 0;
            t->n_scales = n_groups;
            t->scale_block = VTM_Q4_SCALE_BLOCK;
            t->scale_is_f32 =
                (uint8_t)((seg[5] & VTM_Q4_FLAG_F32) ? 1u : 0u);
            t->q4_is_asym = (uint8_t)asym;
            t->layout = VTM_LAYOUT_Q4;
        } else if (t->kind == VTM_TENSOR_Q4K ||
                   t->kind == VTM_TENSOR_Q6K) {
            // kind 6/7 (0165): блоки GGML как есть. Заголовок 18 байт:
            // magic | version | reserved(=0) | n_values u64 | n_blocks u32,
            // затем n_blocks блоков по 144 (Q4_K) или 210 (Q6_K) байт. В fp32
            // НЕ расквантовываем: t->qblocks смотрит в отмеченные блоки,
            // t->f32 == NULL. d/dmin каждого блока проверяются на конечность:
            // NaN в контейнере дал бы тихие NaN-логиты.
            const uint8_t *seg = p + data_start + t->offset64;
            const char *magic = t->kind == VTM_TENSOR_Q4K
                              ? VTM_Q4K_MAGIC : VTM_Q6K_MAGIC;
            uint64_t n_values, total, block_bytes;
            uint32_t n_blk, expected_blocks, k;

            if (t->scale_offset64 != VTM_NO_OFFSET)
                FAIL(VTM_E_FORMAT,
                     "тензор %u (%s): kind=Q4_K/Q6_K требует "
                     "scale_offset=0x%08X (масштабы внутри блока), а пришло %llu",
                     i, t->name, (unsigned)VTM_NO_OFFSET,
                     (unsigned long long)t->scale_offset64);
            if (!range_ok(t->offset64, VTM_Q4K_HEADER_SIZE, data_size))
                FAIL(VTM_E_BOUNDS,
                     "тензор %u (%s): заголовок Q4_K/Q6_K за границей данных",
                     i, t->name);
            if (memcmp(seg, magic, 4) != 0)
                FAIL(VTM_E_FORMAT,
                     "тензор %u (%s): kind=%u без сегмента %s", i, t->name,
                     (unsigned)t->kind, magic);
            if (seg[4] != VTM_Q4K_VERSION)
                FAIL(VTM_E_VERSION,
                     "тензор %u (%s): версия Q4_K-сегмента %u", i, t->name,
                     (unsigned)seg[4]);
            if (seg[5] != 0)
                FAIL(VTM_E_FORMAT,
                     "тензор %u (%s): reserved Q4_K-сегмента = %u, ожидался 0 "
                     "(поле зарезервировано, а не маска флагов)", i, t->name,
                     (unsigned)seg[5]);
            n_values = rd64(seg + 6);
            n_blk = rd32(seg + 14);
            if (n_values != (uint64_t)t->n_elements)
                FAIL(VTM_E_FORMAT,
                     "тензор %u (%s): Q4_K n_values=%llu, dims дают %u",
                     i, t->name, (unsigned long long)n_values, t->n_elements);
            expected_blocks = (uint32_t)(((uint64_t)t->n_elements
                                          + VTM_Q4K_BLOCK_VALUES - 1u)
                                         / VTM_Q4K_BLOCK_VALUES);
            if (n_blk != expected_blocks)
                FAIL(VTM_E_FORMAT,
                     "тензор %u (%s): Q4_K n_blocks=%u, ожидали %u (блок %u)",
                     i, t->name, n_blk, expected_blocks,
                     (unsigned)VTM_Q4K_BLOCK_VALUES);
            block_bytes = q4k_block_bytes(t->kind);
            if (mul_u64(n_blk, block_bytes, &total) != 0 ||
                add_u64(total, VTM_Q4K_HEADER_SIZE, &total) != 0)
                FAIL(VTM_E_BOUNDS, "тензор %u (%s): Q4_K слишком велик",
                     i, t->name);
            if (!range_ok(t->offset64, total, data_size))
                FAIL(VTM_E_BOUNDS,
                     "тензор %u (%s): Q4_K-сегмент (%llu байт) за границей",
                     i, t->name, (unsigned long long)total);
            // VTM_Q4K_CHECK=0 (задача 0172) снимает по-блочную проверку
            // d/dmin. Она читает по 2-4 байта из КАЖДОГО блока, то есть
            // трогает все страницы всех kind 6/7 — на 19 ГиБ MoE это 19 ГиБ
            // page faults на загрузке и RSS размером с модель, то есть
            // разреженного чтения экспертов не получается вовсе. Проверки
            // заголовка, magic, версии, размеров и n_blocks остаются: снимается
            // только скан блоков. По умолчанию поведение прежнее (0165).
            if (vtm_env_flag("VTM_Q4K_CHECK_OFF")) {
                static int warned;
                if (!warned) {
                    fprintf(stderr, "VTM: VTM_Q4K_CHECK_OFF=1 — по-блочная "
                                    "проверка d/dmin Q4_K/Q6_K пропущена "
                                    "(заголовки и размеры проверены)\n");
                    warned = 1;
                }
            } else {
                const uint8_t *bp = seg + VTM_Q4K_HEADER_SIZE;
                // Q4_K: d в байтах 0-1, dmin в 2-3. Q6_K: d — последние два
                // байта блока (208-209), dmin у него нет (масштабы — int8).
                unsigned doff = t->kind == VTM_TENSOR_Q4K ? 0u : 208u;
                unsigned moff = t->kind == VTM_TENSOR_Q4K ? 2u : 208u;
                for (k = 0; k < n_blk; k++) {
                    const uint8_t *b = bp + (size_t)k * (size_t)block_bytes;
                    float d = vtm_half_to_float(rd16(b + doff));
                    float mn = vtm_half_to_float(rd16(b + moff));
                    if (!isfinite(d) || !isfinite(mn))
                        FAIL(VTM_E_SCALE,
                             "тензор %u (%s): блок #%u: d/dmin не конечны",
                             i, t->name, k);
                }
            }
            t->n_blocks = 0;
            t->n_scales = 0;
            t->scale_block = (uint32_t)VTM_Q4K_BLOCK_VALUES;
            t->scale_is_f32 = 0;
            t->n_qblocks = n_blk;
            t->layout = t->kind == VTM_TENSOR_Q4K
                      ? VTM_LAYOUT_Q4K : VTM_LAYOUT_Q6K;
        } else {
            size_t esize = (t->kind == VTM_TENSOR_F32) ? 4u : 2u;
            uint64_t bytes = (uint64_t)t->n_elements * esize;
            if (!range_ok(t->offset64, bytes, data_size))
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
        // Проверка «свой/чужой опкод» остаётся tripwire'ом: неизвестный
        // опкод отвергается до чтения данных, поэтому контейнер с новыми
        // операциями не может быть исполнен старым бинарём «почти верно».
        // Новые опкоды (0172) перечислены явно, а не «всё, что <= 0x11».
        if (in->opcode < VTM_OP_TMV
            || (in->opcode > VTM_OP_HALT && in->opcode != VTM_OP_MOE
                && in->opcode != VTM_OP_RMSH))
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
            if (fill_tensor_planned(&m->tensors[i], p + data_start,
                                    m->arena ? m->arena + off : NULL,
                                    &plans[i]) != 0)
                FAIL(VTM_E_FORMAT,
                     "тензор %u (%s): данные не сошлись с заголовком при "
                     "раскладке", i, m->tensors[i].name);
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
        if ((t->flags & VTM_TENSOR_FLAG_OUT_OF_FILE) != 0) continue;
        begin = base + (uintptr_t)m->data_start + (uintptr_t)t->offset64;
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
    // fd переходит машине: pread-кэш экспертов MoE (0175) читает блоки из
    // того же файла, не трогая страницы отображения. Закроет vtm_free().
    m->fd = fd;
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
    if (m->fd >= 0) {
        close(m->fd);
        m->fd = -1;
    }
    if (m->moe_fd > 0) {
        close(m->moe_fd);
        m->moe_fd = 0;
    }
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
    if (t->kind == VTM_TENSOR_TRIT || t->kind == VTM_TENSOR_Q4) {
        p = (const uint8_t *)(const void *)t->words;
    } else if (t->kind == VTM_TENSOR_Q4K || t->kind == VTM_TENSOR_Q6K) {
        p = t->qblocks;
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
