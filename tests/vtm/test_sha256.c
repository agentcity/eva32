// Тесты SHA-256 и контроля целостности .vtm (задачи 0066, 0078).
//
// Проверяем:
//   * контрольные векторы (пустая строка, "abc", 448 бит, 1 000 000 'a',
//     1 МиБ псевдослучайных байт — эталон посчитан hashlib);
//   * аппаратное ядро (ARMv8 crypto) побитово совпадает с портативным на
//     случайных размерах, включая хвосты и инкрементальный апдейт;
//   * откат на портативное ядро (vtm_sha256_force_portable) даёт те же хеши;
//   * подмена байта в контейнере по-прежнему даёт VTM_E_INTEGRITY — и в
//     данных, и в мете (сам контейнер собирается здесь же, в памяти);
//   * опциональный замер: VTM_SHA_BENCH_MB=512 ./test_sha256 печатает ГБ/с.
//
// Сборка и запуск: cd src/vtm && make test (или make bench-sha).
#define _POSIX_C_SOURCE 200809L

#include "../../src/vtm/sha256.h"
#include "../../src/vtm/vtm.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if !defined(_WIN32)
#include <unistd.h>
#endif

static int g_fail;

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
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

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}

// --- SHA-256: векторы и паритет ядер ---------------------------------------

static void hex_of(const void *p, size_t n, char out[65]) {
    vtm_sha256_ctx c;
    uint8_t digest[32];

    vtm_sha256_init(&c);
    vtm_sha256_update(&c, p, n);
    vtm_sha256_final(&c, digest);
    vtm_sha256_hex(digest, out);
}

static void expect_hex(const char *what, const void *p, size_t n,
                       const char *want) {
    char got[65];

    hex_of(p, n, got);
    if (strcmp(got, want) != 0) {
        fprintf(stderr, "FAIL %s: sha256 = %s, ждали %s\n", what, got, want);
        g_fail++;
    }
}

static void test_vectors(void) {
    static const char abc448[] =
        "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    char *big;
    size_t i, n = 1u << 20;

    expect_hex("пустая строка", "", 0,
               "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b"
               "7852b855");
    expect_hex("\"abc\"", "abc", 3,
               "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f2"
               "0015ad");
    expect_hex("448 бит", abc448, sizeof abc448 - 1u,
               "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419d"
               "b06c1");

    big = (char *)xmalloc(1000000u);
    memset(big, 'a', 1000000u);
    expect_hex("1 000 000 'a'", big, 1000000u,
               "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc711"
               "2cd0");
    free(big);

    // 1 МиБ псевдослучайных байт — эталон из Python hashlib.
    big = (char *)xmalloc(n);
    for (i = 0; i < n; i++) big[i] = (char)(uint8_t)(i * 31u + 7u);
    expect_hex("1 МиБ", big, n,
               "06b7bbfb7824aa03382051691630eb26de85102d1b08a81e907ec0744cd"
               "8a286");
    free(big);
}

// Один и тот же буфер: аппаратное ядро против портативного на разных
// длинах (включая хвосты 1..63 байта и несколько блоков подряд).
static void test_parity(void) {
    size_t n, i;
    uint8_t *buf = (uint8_t *)xmalloc(4096u + 131u);

    for (i = 0; i < 4096u + 131u; i++) buf[i] = (uint8_t)(i * 37u + 11u);

    for (n = 0; n <= 4096u + 131u; n += (n < 300u ? 1u : 137u)) {
        char hw[65], portable[65];

        hex_of(buf, n, hw);
        vtm_sha256_force_portable(1);
        hex_of(buf, n, portable);
        vtm_sha256_force_portable(0);
        if (strcmp(hw, portable) != 0) {
            fprintf(stderr, "FAIL паритет n=%zu: hw=%s portable=%s\n",
                    n, hw, portable);
            g_fail++;
        }
    }
    free(buf);
}

// Инкрементальный апдейт не должен зависеть от разбиения на куски.
static void test_incremental(void) {
    size_t n, i;
    uint8_t *buf = (uint8_t *)xmalloc(3000u);
    char one[65];
    static const size_t steps[] = { 1, 2, 3, 63, 64, 65, 127, 128, 129, 1000 };

    for (i = 0; i < 3000u; i++) buf[i] = (uint8_t)(i * 29u + 5u);
    for (n = 0; n < 4u; n++) {
        size_t len = n < 2u ? (n ? 1u : 0u) : (n == 2u ? 63u : 3000u);
        size_t s;

        hex_of(buf, len, one);
        for (s = 0; s < sizeof steps / sizeof steps[0]; s++) {
            vtm_sha256_ctx c;
            uint8_t digest[32];
            char got[65];
            size_t pos = 0;

            vtm_sha256_init(&c);
            while (pos < len) {
                size_t take = steps[s];
                if (take > len - pos) take = len - pos;
                vtm_sha256_update(&c, buf + pos, take);
                pos += take;
            }
            vtm_sha256_final(&c, digest);
            vtm_sha256_hex(digest, got);
            if (strcmp(got, one) != 0) {
                fprintf(stderr, "FAIL инкремент len=%zu шаг=%zu: %s != %s\n",
                        len, steps[s], got, one);
                g_fail++;
            }
        }
    }
    free(buf);
}

static void test_impl(void) {
    const char *impl = vtm_sha256_impl();
    int hw = vtm_sha256_hw_available();

    printf("test_sha256: ядро = %s (аппаратное: %s)\n", impl,
           hw ? "да" : "нет, откат на портативное");
    CHECK(impl != NULL);
    CHECK(strcmp(impl, "portable") == 0 || strcmp(impl, "armv8-crypto") == 0);
    CHECK(hw == (strcmp(impl, "portable") != 0));

    // Принудительный откат: имя меняется, хеш — нет.
    vtm_sha256_force_portable(1);
    CHECK(strcmp(vtm_sha256_impl(), "portable") == 0);
    CHECK(!vtm_sha256_hw_available());
    vtm_sha256_force_portable(0);
    CHECK(strcmp(vtm_sha256_impl(), impl) == 0);
    CHECK(vtm_sha256_hw_available() == hw);
}

// --- контроль целостности контейнера .vtm ----------------------------------

static void put16(uint8_t *p, uint16_t v) {
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
}

static void put32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}

static size_t align32(size_t v) {
    return (v + 31u) & ~(size_t)31u;
}

// Контейнер: заголовок 32, один fp32-тензор [4,4] (64 байта) с offset 0,
// мета с блоком integrity, программа HALT (16 байт).
// bad_data != 0 — испортить байт в секции данных уже после того, как хеш
// посчитан по хорошим данным; bad_meta != 0 — то же для меты.
static size_t build_container(uint8_t *buf, size_t cap, int bad_data,
                             int bad_meta) {
    uint8_t data[64];
    char meta[256];
    size_t data_start = align32(32u + 52u);
    size_t meta_off, meta_len, program_off, total, i;
    char data_hex[65], meta_hex[65];
    size_t meta_sha_pos;

    for (i = 0; i < sizeof data; i++) data[i] = (uint8_t)(0x11u + i);
    hex_of(data, sizeof data, data_hex);

    // Первый проход — с нулями на месте meta_sha256: так писатель и хеширует.
    snprintf(meta, sizeof meta,
             "{\"note\":\"0078\",\"integrity\":{\"algo\":\"sha256\","
             "\"data_sha256\":\"%s\",\"meta_sha256\":\"%064d\"}}",
             data_hex, 0);
    meta_len = strlen(meta);
    meta_sha_pos = (size_t)(strstr(meta, "\"meta_sha256\":\"") - meta) + 15u;
    CHECK(meta_sha_pos + 64u < meta_len);

    {
        uint8_t digest[32];
        vtm_sha256_ctx c;
        vtm_sha256_init(&c);
        vtm_sha256_update(&c, meta, meta_sha_pos);
        {
            char zeros[65];
            memset(zeros, '0', 64u);
            vtm_sha256_update(&c, zeros, 64u);
        }
        vtm_sha256_update(&c, meta + meta_sha_pos + 64u,
                          meta_len - meta_sha_pos - 64u);
        vtm_sha256_final(&c, digest);
        vtm_sha256_hex(digest, meta_hex);
    }

    // Второй проход — с настоящим meta_sha256 (длина та же: 64 hex-символа).
    snprintf(meta, sizeof meta,
             "{\"note\":\"0078\",\"integrity\":{\"algo\":\"sha256\","
             "\"data_sha256\":\"%s\",\"meta_sha256\":\"%s\"}}",
             data_hex, meta_hex);
    meta_len = strlen(meta);
    if (bad_meta) {
        // "0078" -> "0079": длина та же, хеш посчитан по исходной мете.
        char *note = strstr(meta, "\"note\":\"");
        CHECK(note != NULL);
        if (note) note[9] = '9';
    }

    meta_off = data_start + sizeof data;              // уже кратно 32
    program_off = meta_off + align32(meta_len);
    total = program_off + 16u;
    CHECK(total <= cap);

    memset(buf, 0, total);
    memcpy(buf, "VTM1", 4);
    put16(buf + 4, 2u);                 // версия формата
    put16(buf + 6, 0u);                 // флаги
    put32(buf + 8, 1u);                 // n_instr
    put32(buf + 12, 1u);                // n_tensors
    put32(buf + 16, 16u);               // n_program (байты)
    put32(buf + 20, (uint32_t)program_off);
    put32(buf + 24, (uint32_t)meta_off);
    put32(buf + 28, (uint32_t)meta_len);

    // Запись тензора: имя[24], kind, ndim, dims[4], offset, scale, резерв.
    memcpy(buf + 32, "w", 1);           // остальные 23 байта — нули
    buf[32 + 24] = 1;                   // kind = F32
    buf[32 + 25] = 2;                   // ndim
    put32(buf + 32 + 26, 4u);
    put32(buf + 32 + 30, 4u);
    put32(buf + 32 + 42, 0u);           // offset
    put32(buf + 32 + 46, 0xFFFFFFFFu);  // scale_offset — не используется

    memcpy(buf + data_start, data, sizeof data);
    if (bad_data) buf[data_start + 5] ^= 0x01u;   // хеш уже посчитан

    memcpy(buf + meta_off, meta, meta_len);
    // Программа: HALT (0x0F) со всеми «нет тензора».
    buf[program_off] = 0x0F;
    put16(buf + program_off + 2, 0xFFFFu);
    put16(buf + program_off + 4, 0xFFFFu);
    put16(buf + program_off + 6, 0xFFFFu);
    put16(buf + program_off + 8, 0xFFFFu);
    return total;
}

#if !defined(_WIN32)

static int write_file(const char *path, const uint8_t *buf, size_t n) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    if (fwrite(buf, 1, n, f) != n) {
        fclose(f);
        return -1;
    }
    return fclose(f) == 0 ? 0 : -1;
}

static void check_rejected(uint8_t *buf, size_t total, const char *path,
                           const char *why) {
    char err[256];
    vtm_status st = VTM_OK;
    vtm_t *m;

    // Файловый путь (mmap) — как у CLI.
    CHECK(write_file(path, buf, total) == 0);
    m = vtm_load(path);
    if (m) {
        fprintf(stderr, "FAIL %s: vtm_load вернул машину\n", why);
        g_fail++;
        vtm_free(m);
    }
    CHECK(strstr(vtm_last_error(), "sha256") != NULL);

    // Тот же разбор из памяти, но с кодом ошибки.
    st = VTM_OK;
    m = vtm_load_mem(buf, total, &st, err, sizeof err);
    if (m) {
        fprintf(stderr, "FAIL %s: vtm_load_mem вернул машину\n", why);
        g_fail++;
        vtm_free(m);
    }
    if (st != VTM_E_INTEGRITY) {
        fprintf(stderr, "FAIL %s: статус %d, ждали VTM_E_INTEGRITY (%d): %s\n",
                why, (int)st, (int)VTM_E_INTEGRITY, err);
        g_fail++;
    }
}

static void test_container_integrity(void) {
    char path[256], err[256];
    const char *tmpdir = getenv("TMPDIR");
    uint8_t *buf;
    size_t total;
    vtm_status st = VTM_OK;
    vtm_t *m;

    buf = (uint8_t *)xmalloc(4096u);
    snprintf(path, sizeof path, "%s/vtm_sha256_XXXXXX",
             tmpdir && *tmpdir ? tmpdir : "/tmp");
    {
        int fd = mkstemp(path);
        if (fd < 0) {
            fprintf(stderr, "FAIL не удалось создать временный файл\n");
            g_fail++;
            free(buf);
            return;
        }
        close(fd);
    }

    // 1. Целый контейнер грузится и из файла, и из памяти.
    total = build_container(buf, 4096u, 0, 0);
    CHECK(write_file(path, buf, total) == 0);
    m = vtm_load(path);
    if (!m) {
        fprintf(stderr, "FAIL целый контейнер не загрузился: %s\n",
                vtm_last_error());
        g_fail++;
    }
    if (m) vtm_free(m);
    m = vtm_load_mem(buf, total, &st, err, sizeof err);
    if (!m) {
        fprintf(stderr, "FAIL vtm_load_mem: статус %d, %s\n", (int)st, err);
        g_fail++;
    } else {
        CHECK(st == VTM_OK);
        vtm_free(m);
    }

    // 2. Подмена одного байта в секции данных — VTM_E_INTEGRITY.
    total = build_container(buf, 4096u, 1, 0);
    check_rejected(buf, total, path, "подмена байта данных");

    // 3. Подмена байта в мете — VTM_E_INTEGRITY по meta_sha256.
    total = build_container(buf, 4096u, 0, 1);
    check_rejected(buf, total, path, "подмена байта меты");

    // 4. Откат на портативное ядро: целый контейнер проверяется так же.
    total = build_container(buf, 4096u, 0, 0);
    setenv("VTM_SHA_PORTABLE", "1", 1);
    vtm_sha256_force_portable(0);          // сбросить детект и перечитать env
    CHECK(strcmp(vtm_sha256_impl(), "portable") == 0);
    st = VTM_OK;
    m = vtm_load_mem(buf, total, &st, err, sizeof err);
    if (!m) {
        fprintf(stderr, "FAIL портативный откат: статус %d, %s\n",
                (int)st, err);
        g_fail++;
    } else {
        CHECK(st == VTM_OK);
        vtm_free(m);
    }
    unsetenv("VTM_SHA_PORTABLE");
    vtm_sha256_force_portable(0);

    unlink(path);
    free(buf);
}

#endif /* !_WIN32 */

// --- замер (не входит в make test) -----------------------------------------

static void bench_if_requested(void) {
    const char *env = getenv("VTM_SHA_BENCH_MB");
    size_t mb, n;
    uint8_t *buf;
    char hw_hex[65], portable_hex[65];
    double t0, t_hw, t_portable;
    vtm_sha256_ctx c;
    uint8_t digest[32];

    if (!env || !*env) return;
    mb = (size_t)strtoul(env, NULL, 10);
    if (mb == 0) return;
    n = mb << 20;
    buf = (uint8_t *)xmalloc(n);
    memset(buf, 0, n);

    vtm_sha256_init(&c);
    t0 = now_s();
    vtm_sha256_update(&c, buf, n);
    vtm_sha256_final(&c, digest);
    t_hw = now_s() - t0;
    vtm_sha256_hex(digest, hw_hex);

    vtm_sha256_force_portable(1);
    vtm_sha256_init(&c);
    t0 = now_s();
    vtm_sha256_update(&c, buf, n);
    vtm_sha256_final(&c, digest);
    t_portable = now_s() - t0;
    vtm_sha256_hex(digest, portable_hex);
    vtm_sha256_force_portable(0);

    printf("sha256 %zu МиБ: аппаратное %.3f с (%.3f ГБ/с), "
           "портативное %.3f с (%.3f ГБ/с)\n",
           mb, t_hw, (double)mb / 1024.0 / t_hw,
           t_portable, (double)mb / 1024.0 / t_portable);
    printf("  хеши совпадают: %s\n",
           strcmp(hw_hex, portable_hex) == 0 ? "да" : "НЕТ");
    CHECK(strcmp(hw_hex, portable_hex) == 0);
    free(buf);
}

int main(void) {
    test_impl();
    test_vectors();
    test_parity();
    test_incremental();
#if !defined(_WIN32)
    test_container_integrity();
#endif
    bench_if_requested();

    if (g_fail) {
        fprintf(stderr, "test_sha256: %d FAIL\n", g_fail);
        return 1;
    }
    printf("test_sha256: OK\n");
    return 0;
}
