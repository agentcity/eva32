// SHA-256 для проверки целостности .vtm (задачи 0066, 0078).
//
// Два ядра с одинаковым контрактом `blocks(st, p, n)`: обработать n полных
// 64-байтных блоков, p выровнен не обязательно.
//
// * портативное — чистый C99 (то же, что раньше жило в loader.c);
// * аппаратное — ARMv8 crypto extensions (AArch64): sha256h/sha256h2 для
//   раундов и sha256su0/sha256su1 для расписания сообщения. На Apple arm64
//   (M1 и новее) расширения входят в базовый набор, поэтому включены уже на
//   этапе компиляции (__ARM_FEATURE_CRYPTO + __ARM_FEATURE_SHA2).
//
// Аппаратное ядро не включается «по факту сборки»: перед первым использованием
// прогоняется самопроверка на векторе "abc" (см. hw_selftest). Если он не
// совпал с эталоном, ядро навсегда откатывается на портативное. Это стоит
// один блок на процесс и защищает от ошибки в аппаратном пути.
//
// Раскладка состояния под аппаратные инструкции (проверено на M1 Pro
// поштучно по каждой инструкции, см. tests/vtm/test_sha256.c):
//   AB  = (a, b, c, d), EFGH = (e, f, g, h);
//   AB'   = sha256h(AB, EFGH, W+K);
//   EFGH' = sha256h2(EFGH, AB, W+K)   — от СТАРОГО AB;
//   sha256su0(d, n): d[i] += S0((d[1],d[2],d[3],n[0])[i]);
//   sha256su1(d, n, m): d[i] += (n[1],n[2],n[3],m[0])[i] + S1(r[i]),
//     где r[0] = m[2], r[1] = m[3], r[2] и r[3] — уже посчитанные d[0] и d[1]
//     (инструкция последовательная по дорожкам);
//   S0(x) = ror7 ^ ror18 >> 3, S1(x) = ror17 ^ ror19 >> 10.
//
// Глобальное состояние диспетчера (g_impl/g_force) не потокобезопасно — как и
// остальной VTM: одна машина/один хеш на поток.
//
// Порядок байт. Портативное ядро читает вход побайтно (слово собирается
// сдвигами), поэтому от endianness хоста не зависит. Аппаратное ядро ARMv8 —
// зависит: sha256h/sha256su* работают с 32-битными словами регистров, и
// раскладка AB/EFGH ниже верна для little-endian AArch64 (обычный режим ARM,
// BE8 сюда не входит). ВТМ на big-endian всё равно отказывает ещё в
// vtm_load() (VTM_E_UNSUPPORTED, docs/vtm.md «Портируемость»), так что
// BE-путь хеша не исполняется; отдельного BE-ядра у SHA-256 нет.
#include "sha256.h"

#include <stdlib.h>
#include <string.h>

#if defined(__aarch64__) && defined(__ARM_FEATURE_CRYPTO) && \
    defined(__ARM_FEATURE_SHA2) && !defined(VTM_SHA_NO_ARM_CRYPTO)
#define VTM_SHA_HW_ARM 1
#include <arm_neon.h>
#else
#define VTM_SHA_HW_ARM 0
#endif

// --- общие константы и эталон для самопроверки -----------------------------

static const uint32_t sha256_k[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};

#if VTM_SHA_HW_ARM
// SHA-256("abc") — эталон для самопроверки аппаратного ядра.
static const uint8_t sha256_abc[32] = {
    0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea,
    0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22, 0x23,
    0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c,
    0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad
};
#endif /* VTM_SHA_HW_ARM */

static void sha256_store_be(uint32_t st[8], uint8_t out[32]) {
    unsigned j;
    for (j = 0; j < 8u; j++) {
        out[4u * j] = (uint8_t)(st[j] >> 24);
        out[4u * j + 1u] = (uint8_t)(st[j] >> 16);
        out[4u * j + 2u] = (uint8_t)(st[j] >> 8);
        out[4u * j + 3u] = (uint8_t)st[j];
    }
}

static void sha256_set_initial(uint32_t st[8]) {
    st[0] = 0x6a09e667u; st[1] = 0xbb67ae85u;
    st[2] = 0x3c6ef372u; st[3] = 0xa54ff53au;
    st[4] = 0x510e527fu; st[5] = 0x9b05688cu;
    st[6] = 0x1f83d9abu; st[7] = 0x5be0cd19u;
}

// --- портативное ядро ------------------------------------------------------

static uint32_t rotr32(uint32_t x, unsigned n) {
    return (x >> n) | (x << (32u - n));
}

static void sha256_block_portable(uint32_t st[8], const uint8_t *p) {
    uint32_t w[64], a, b, c, d, e, f, g, h;
    unsigned i;

    for (i = 0; i < 16; i++) {
        w[i] = ((uint32_t)p[i * 4u] << 24)
             | ((uint32_t)p[i * 4u + 1u] << 16)
             | ((uint32_t)p[i * 4u + 2u] << 8)
             | (uint32_t)p[i * 4u + 3u];
    }
    for (i = 16; i < 64; i++) {
        uint32_t s0 = rotr32(w[i - 15u], 7) ^ rotr32(w[i - 15u], 18)
                    ^ (w[i - 15u] >> 3);
        uint32_t s1 = rotr32(w[i - 2u], 17) ^ rotr32(w[i - 2u], 19)
                    ^ (w[i - 2u] >> 10);
        w[i] = w[i - 16u] + s0 + w[i - 7u] + s1;
    }
    a = st[0]; b = st[1]; c = st[2]; d = st[3];
    e = st[4]; f = st[5]; g = st[6]; h = st[7];
    for (i = 0; i < 64; i++) {
        uint32_t s1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t t1 = h + s1 + ch + sha256_k[i] + w[i];
        uint32_t s0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = s0 + maj;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    st[0] += a; st[1] += b; st[2] += c; st[3] += d;
    st[4] += e; st[5] += f; st[6] += g; st[7] += h;
}

static void sha256_blocks_portable(uint32_t st[8], const uint8_t *p,
                                   size_t nblocks) {
    while (nblocks--) {
        sha256_block_portable(st, p);
        p += 64u;
    }
}

// --- аппаратное ядро: ARMv8 crypto extensions ------------------------------

#if VTM_SHA_HW_ARM

// Одна группа раундов: 4 раунда над AB/EFGH со словами W+K из *k.
// AB и EFGH — переменные sha256_blocks_arm; k сдвигается на 4 слова.
#define VTM_SHA_ARM_ROUND(msg) do { \
        uint32x4_t wk_ = vaddq_u32((msg), vld1q_u32(k)); \
        uint32x4_t ab_ = ab; \
        ab = vsha256hq_u32(ab_, ef, wk_); \
        ef = vsha256h2q_u32(ef, ab_, wk_); \
        k += 4; \
    } while (0)

// Группа из 4 раундов с обновлением расписания: x — регистр текущих слов
// W[4g..4g+3] (раунды идут первыми), y — следующий регистр (для su0),
// z и w — соседи для su1.
#define VTM_SHA_ARM_GROUP(x, y, z, w) do { \
        VTM_SHA_ARM_ROUND(x); \
        (x) = vsha256su0q_u32((x), (y)); \
        (x) = vsha256su1q_u32((x), (z), (w)); \
    } while (0)

static void sha256_blocks_arm(uint32_t st[8], const uint8_t *p,
                             size_t nblocks) {
    uint32x4_t ab = vld1q_u32(&st[0]);
    uint32x4_t ef = vld1q_u32(&st[4]);

    while (nblocks--) {
        uint32x4_t ab_save = ab, ef_save = ef;
        uint32x4_t m0 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(p)));
        uint32x4_t m1 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(p + 16)));
        uint32x4_t m2 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(p + 32)));
        uint32x4_t m3 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(p + 48)));
        const uint32_t *k = sha256_k;

        // 12 групп обновляют расписание (W[16..63]) и считают раунды 0..47.
        VTM_SHA_ARM_GROUP(m0, m1, m2, m3);
        VTM_SHA_ARM_GROUP(m1, m2, m3, m0);
        VTM_SHA_ARM_GROUP(m2, m3, m0, m1);
        VTM_SHA_ARM_GROUP(m3, m0, m1, m2);

        VTM_SHA_ARM_GROUP(m0, m1, m2, m3);
        VTM_SHA_ARM_GROUP(m1, m2, m3, m0);
        VTM_SHA_ARM_GROUP(m2, m3, m0, m1);
        VTM_SHA_ARM_GROUP(m3, m0, m1, m2);

        VTM_SHA_ARM_GROUP(m0, m1, m2, m3);
        VTM_SHA_ARM_GROUP(m1, m2, m3, m0);
        VTM_SHA_ARM_GROUP(m2, m3, m0, m1);
        VTM_SHA_ARM_GROUP(m3, m0, m1, m2);

        // Раунды 48..63: расписание уже готово, обновлять нечего.
        VTM_SHA_ARM_ROUND(m0);
        VTM_SHA_ARM_ROUND(m1);
        VTM_SHA_ARM_ROUND(m2);
        VTM_SHA_ARM_ROUND(m3);

        // Обратное сложение с состоянием на входе блока (как в обычном
        // SHA-256: раунды работают с ab/ef, а состояние — их сумма).
        ab = vaddq_u32(ab, ab_save);
        ef = vaddq_u32(ef, ef_save);

        p += 64u;
    }
    vst1q_u32(&st[0], ab);
    vst1q_u32(&st[4], ef);
}

#undef VTM_SHA_ARM_ROUND
#undef VTM_SHA_ARM_GROUP

// Самопроверка: один блок "abc" (добитый по правилам SHA-256) через
// аппаратное ядро должен дать эталонный дайджест.
static int sha256_hw_selftest(void) {
    uint32_t st[8];
    uint8_t block[64], out[32];

    memset(block, 0, sizeof block);
    block[0] = 'a';
    block[1] = 'b';
    block[2] = 'c';
    block[3] = 0x80u;
    block[63] = 24u;              // длина в битах, big-endian u64
    sha256_set_initial(st);
    sha256_blocks_arm(st, block, 1);
    sha256_store_be(st, out);
    return memcmp(out, sha256_abc, sizeof out) == 0;
}

#endif /* VTM_SHA_HW_ARM */

// --- выбор ядра ------------------------------------------------------------

enum { VTM_SHA_UNDETECTED = 0, VTM_SHA_PORTABLE = 1, VTM_SHA_HW = 2 };

static int g_impl = VTM_SHA_UNDETECTED;
static int g_force_portable;

// VTM_SHA_PORTABLE=1 — принудительный откат (как VTM_SKIP_HASH для проверки).
static int sha256_env_portable(void) {
    const char *v = getenv("VTM_SHA_PORTABLE");
    return v && *v && strcmp(v, "0") != 0;
}

static void sha256_detect(void) {
    if (g_impl != VTM_SHA_UNDETECTED) return;
    if (g_force_portable || sha256_env_portable()) {
        g_impl = VTM_SHA_PORTABLE;
        return;
    }
#if VTM_SHA_HW_ARM
    g_impl = sha256_hw_selftest() ? VTM_SHA_HW : VTM_SHA_PORTABLE;
#else
    // Аппаратного ядра нет: на x86 SHA-NI намеренно не включён — эту ветку
    // нельзя проверить на машине сборки, а непроверенный хеш хуже медленного.
    // Место для него — sha256_blocks_x86 и hw-проверка здесь.
    g_impl = VTM_SHA_PORTABLE;
#endif
}

static void sha256_blocks(uint32_t st[8], const uint8_t *p, size_t nblocks) {
    sha256_detect();
#if VTM_SHA_HW_ARM
    if (g_impl == VTM_SHA_HW) {
        sha256_blocks_arm(st, p, nblocks);
        return;
    }
#endif
    sha256_blocks_portable(st, p, nblocks);
}

// --- открытый интерфейс ----------------------------------------------------

void vtm_sha256_init(vtm_sha256_ctx *c) {
    sha256_set_initial(c->state);
    c->total = 0;
    c->buflen = 0;
}

void vtm_sha256_update(vtm_sha256_ctx *c, const void *data, size_t n) {
    const uint8_t *p = (const uint8_t *)data;

    c->total += (uint64_t)n;
    if (c->buflen) {
        size_t take = 64u - c->buflen;
        if (take > n) take = n;
        memcpy(c->buf + c->buflen, p, take);
        c->buflen += take;
        p += take;
        n -= take;
        if (c->buflen == 64u) {
            sha256_blocks(c->state, c->buf, 1);
            c->buflen = 0;
        }
    }
    if (n >= 64u) {
        size_t nblocks = n / 64u;
        sha256_blocks(c->state, p, nblocks);
        p += nblocks * 64u;
        n -= nblocks * 64u;
    }
    if (n) {
        memcpy(c->buf, p, n);
        c->buflen = n;
    }
}

void vtm_sha256_final(vtm_sha256_ctx *c, uint8_t out[32]) {
    uint64_t bits = c->total * 8u;
    size_t i = c->buflen;
    unsigned j;

    c->buf[i++] = 0x80u;
    if (i > 56u) {
        memset(c->buf + i, 0, 64u - i);
        sha256_blocks(c->state, c->buf, 1);
        i = 0;
    }
    memset(c->buf + i, 0, 56u - i);
    for (j = 0; j < 8u; j++)
        c->buf[56u + j] = (uint8_t)(bits >> (56u - 8u * j));
    sha256_blocks(c->state, c->buf, 1);
    sha256_store_be(c->state, out);
}

void vtm_sha256_hex(const uint8_t digest[32], char out[65]) {
    static const char hex[] = "0123456789abcdef";
    size_t i;

    for (i = 0; i < 32u; i++) {
        out[i * 2u] = hex[(digest[i] >> 4) & 0x0fu];
        out[i * 2u + 1u] = hex[digest[i] & 0x0fu];
    }
    out[64] = '\0';
}

const char *vtm_sha256_impl(void) {
    sha256_detect();
#if VTM_SHA_HW_ARM
    if (g_impl == VTM_SHA_HW) return "armv8-crypto";
#endif
    return "portable";
}

int vtm_sha256_hw_available(void) {
    sha256_detect();
    return g_impl == VTM_SHA_HW;
}

void vtm_sha256_force_portable(int on) {
    g_force_portable = on ? 1 : 0;
    g_impl = VTM_SHA_UNDETECTED;
}
