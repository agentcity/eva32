// Кодек тритов (формат v0.1, 20 тритов в 32-битном слове) + замеры эмуляции.
//
// Ключевое свойство формата: 3^10 = 59049 < 65536, поэтому слово естественно
// раскладывается на два 16-битных полуслова по 10 тритов. Значит декодирование
// можно делать таблицей на 59049 входов, без деления на 3 в горячем цикле.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#define P10 59049u          // 3^10
#define TRITS_PER_WORD 20

static uint16_t *LUT10;     // LUT10[v][k] = цифра 0..2 для k-го трита
static int8_t   *LUT10S;    // то же в значениях -1..+1
static int8_t    LUT243[243][5];  // 4 группы по 5 тритов на слово: 3^20 = 243^4

static void lut_init(void) {
    LUT10  = malloc(sizeof(uint16_t) * P10 * 10);
    LUT10S = malloc(sizeof(int8_t)   * P10 * 10);
    for (uint32_t v = 0; v < P10; v++) {
        uint32_t t = v;
        for (int k = 0; k < 10; k++) {
            unsigned d = t % 3; t /= 3;
            LUT10[v * 10 + k]  = (uint16_t)d;
            LUT10S[v * 10 + k] = (int8_t)(d == 2 ? 1 : (d == 0 ? -1 : 0));
        }
    }
    for (unsigned v = 0; v < 243; v++) {
        unsigned t = v;
        for (int k = 0; k < 5; k++) {
            unsigned d = t % 3; t /= 3;
            LUT243[v][k] = (int8_t)(d == 2 ? 1 : (d == 0 ? -1 : 0));
        }
    }
}

// --- упаковка 20 тритов (-1/0/+1) в слово
static uint32_t pack20(const int8_t *t) {
    uint32_t v = 0;
    for (int k = TRITS_PER_WORD - 1; k >= 0; k--) {
        unsigned d = t[k] > 0 ? 2u : (t[k] < 0 ? 0u : 1u);
        v = v * 3 + d;
    }
    return v;
}

// --- распаковка слова в 20 тритов
static void unpack20(uint32_t word, int8_t *out) {
    uint32_t lo = word % P10;      // младшие 10 тритов
    uint32_t hi = word / P10;      // старшие 10
    for (int k = 0; k < 10; k++) { out[k] = LUT10S[lo * 10 + k]; }
    for (int k = 0; k < 10; k++) { out[10 + k] = LUT10S[hi * 10 + k]; }
}

static double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// hex в порядке байт little-endian — ровно как в спецификации
static void le_hex(uint32_t v, char *out) {
    snprintf(out, 9, "%02x%02x%02x%02x",
             (unsigned)(v & 0xFF), (unsigned)((v >> 8) & 0xFF),
             (unsigned)((v >> 16) & 0xFF), (unsigned)((v >> 24) & 0xFF));
}

static int selftest(void) {
    struct { int8_t t[20]; const char *hex; } cases[] = {
        {{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1}, "00000000"},
        {{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, "c80dea67"},
        {{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, "901bd4cf"},
        {{ 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1, 1,-1}, "e406f533"},
    };
    int ok = 1;
    for (unsigned i = 0; i < sizeof(cases)/sizeof(cases[0]); i++) {
        uint32_t v = pack20(cases[i].t);
        int8_t back[20]; unpack20(v, back);
        int same = memcmp(back, cases[i].t, 20) == 0;
        char got[9]; le_hex(v, got);
        int hexok = strcmp(got, cases[i].hex) == 0;
        if (!same || !hexok) { ok = 0; }
        printf("  %s: LE=%s ожидали=%s roundtrip=%s %s\n",
               hexok ? "ok  " : "FAIL", got, cases[i].hex,
               same ? "ok" : "FAIL", hexok && same ? "" : "← РАСХОЖДЕНИЕ");
    }
    return ok;
}

// --- GEMV как в реальности: большая матрица W (R x K) и маленький вектор x (K)
#define K 4096
#define R 16384            // 64 М весов: int8 64 МБ, триты 12,8 МБ
static int8_t   *W8;       // R*K байт
static uint32_t *W20;      // R*K/20 слов
static int8_t    xv[K];    // активации, влезают в кэш
static volatile long long sink;

static double gemv_int8(void) {
    long long acc = 0;
    double t0 = now();
    for (int r = 0; r < R; r++) {
        const int8_t *row = &W8[(size_t)r * K];
        long long s = 0;
        for (int k = 0; k < K; k++) s += (int)row[k] * (int)xv[k];
        acc += s;
    }
    sink = acc;
    return now() - t0;
}

static double gemv_trit(void) {
    long long acc = 0;
    double t0 = now();
    for (int r = 0; r < R; r++) {
        const uint32_t *row = &W20[(size_t)r * (K / 20)];
        long long s = 0;
        for (int w = 0; w < K / 20; w++) {
            uint32_t v = row[w];
            uint32_t lo = v % P10, hi = v / P10;
            const int8_t *l = &LUT10S[lo * 10], *h = &LUT10S[hi * 10];
            const int8_t *xr = &xv[w * 20];
            for (int k = 0; k < 10; k++) s += (long long)l[k] * xr[k];
            for (int k = 0; k < 10; k++) s += (long long)h[k] * xr[10 + k];
        }
        acc += s;
    }
    sink = acc;
    return now() - t0;
}

// то же слово, но декодируется четырьмя группами по 5 тритов — таблица 1,2 КБ
static double gemv_trit243(void) {
    long long acc = 0;
    double t0 = now();
    for (int r = 0; r < R; r++) {
        const uint32_t *row = &W20[(size_t)r * (K / 20)];
        long long s = 0;
        for (int w = 0; w < K / 20; w++) {
            uint32_t v = row[w];
            const int8_t *xr = &xv[w * 20];
            for (int g = 0; g < 4; g++) {
                unsigned d = v % 243; v /= 243;
                const int8_t *t = LUT243[d];
                s += (long long)t[0] * xr[g*5] + t[1] * xr[g*5+1] + t[2] * xr[g*5+2]
                   + t[3] * xr[g*5+3] + t[4] * xr[g*5+4];
            }
        }
        acc += s;
    }
    sink = acc;
    return now() - t0;
}

static double unpack_all(void) {
    int8_t tmp[20]; long long acc = 0;
    double t0 = now();
    for (size_t i = 0; i < (size_t)R * K / 20; i++) {
        unpack20(W20[i], tmp);
        acc += tmp[0] + tmp[19];
    }
    sink = acc;
    return now() - t0;
}

int main(void) {
    lut_init();
    printf("кодек: контрольные векторы формата v0.1\n");
    if (!selftest()) { printf("САМОПРОВЕРКА ПРОВАЛЕНА\n"); return 1; }
    printf("  самопроверка: ok\n\n");

    srand(7);
    W8  = malloc((size_t)R * K);
    W20 = malloc((size_t)R * K / 20 * 4);
    for (int k = 0; k < K; k++) xv[k] = (int8_t)(rand() % 20 - 10);
    for (size_t i = 0; i < (size_t)R * K; i++) W8[i] = (int8_t)(rand() % 255 - 127);
    int8_t t[20];
    for (size_t w = 0; w < (size_t)R * K / 20; w++) {
        for (int k = 0; k < 20; k++) { int d = rand() % 3; t[k] = (int8_t)(d == 2 ? 1 : (d == 0 ? -1 : 0)); }
        W20[w] = pack20(t);
    }

    double t8 = gemv_int8(), tt = gemv_trit(), t243 = gemv_trit243(), tu = unpack_all();
    double mb8 = (double)R * K / 1e6, mbt = (double)R * K * 4.0 / 20 / 1e6;
    printf("GEMV: матрица %d x %d (%zu М весов), вне кэша, активации в кэше:\n",
           R, K, (size_t)R * K / 1000000);
    printf("  int8       %7.1f мс   читаем %.1f МБ   %5.2f ГБ/с\n", t8*1e3, mb8, mb8/1e3/t8);
    printf("  триты+LUT10 %7.1f мс   читаем %.1f МБ   %5.2f ГБ/с\n", tt*1e3, mbt, mbt/1e3/tt);
    printf("  триты+LUT5  %7.1f мс   читаем %.1f МБ   %5.2f ГБ/с\n", t243*1e3, mbt, mbt/1e3/t243);
    printf("  → по времени: LUT10 %.2fx, LUT5 %.2fx; по памяти %.1fx\n",
           t8/tt, t8/t243, mb8/mbt);
    printf("\nтолько распаковка кодеком: %.0f М тритов/с, %.2f ГБ/с выхода\n",
           ((double)R*K/1e6)/tu, ((double)R*K/1e9)/tu);
    free(W8); free(W20); free(LUT10); free(LUT10S);
    return 0;
}
