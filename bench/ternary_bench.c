// Сравнение трёх форматов весов на одном и том же скалярном произведении:
//   1) int8:  байт на вес, умножение+сложение
//   2) int4:  2 веса в байте, распаковка + умножение
//   3) триты: 20 тритов в 32-битном слове, только сложения (без умножения)
// Цель — не «кто быстрее в теории», а байты на токен и фактическое время.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#define N 4096            // длина вектора (типичный «срез» активных весов)
#define ITERS 20000

static double now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int8_t  w8[N];
static uint8_t w4[N / 2];
static uint32_t w20[(N + 19) / 20];
static int8_t  x[N];

// --- int8: обычное скалярное произведение
static long long dot8(void) {
    long long acc = 0;
    for (int i = 0; i < N; i++) acc += (int)w8[i] * (int)x[i];
    return acc;
}

// --- int4: два знаковых полубайта в байте, распаковка + умножение
static long long dot4(void) {
    long long acc = 0;
    for (int i = 0; i < N / 2; i++) {
        uint8_t b = w4[i];
        int lo = (int8_t)(b << 4) >> 4;   // знаковое расширение младшего
        int hi = (int8_t)b >> 4;         // знаковое расширение старшего
        acc += lo * (int)x[2 * i] + hi * (int)x[2 * i + 1];
    }
    return acc;
}

// --- триты: 20 тритов в слове, только сложения и вычитания
static long long dot20(void) {
    long long acc = 0;
    for (int w = 0; w < (N + 19) / 20; w++) {
        uint32_t v = w20[w];
        int base = w * 20;
        for (int k = 0; k < 20; k++) {
            int d = v % 3; v /= 3;         // 0→-1, 1→0, 2→+1
            int idx = base + k;
            if (idx >= N) break;
            if (d == 2)      acc += x[idx];  // +1
            else if (d == 0) acc -= x[idx];  // -1
            // d == 1 → вес 0, ничего не читаем из x
        }
    }
    return acc;
}

static double bench(long long (*fn)(void)) {
    volatile long long sink = 0;
    double t0 = now();
    for (int it = 0; it < ITERS; it++) sink += fn();
    double dt = now() - t0;
    (void)sink;
    return dt / ITERS;
}

// --- триты, LUT-декодирование: 5 тритов в байте, таблица на 256 входов
static int8_t LUT5[256][5];
static uint8_t w5[N / 5 * 5 / 5];   // по 5 тритов на байт

static void build_lut5(void) {
    for (int b = 0; b < 256; b++) {
        int v = b % 243;                // 3^5 = 243
        for (int k = 0; k < 5; k++) {   // 0→-1, 1→0, 2→+1
            int d = v % 3; v /= 3;
            LUT5[b][k] = (int8_t)(d == 2 ? 1 : (d == 0 ? -1 : 0));
        }
    }
}

static long long dot5_lut(void) {
    long long acc = 0;
    for (int b = 0; b < N / 5; b++) {
        const int8_t *t = LUT5[w5[b]];
        int base = b * 5;
        acc += (long long)t[0] * x[base] + (long long)t[1] * x[base+1]
             + (long long)t[2] * x[base+2] + (long long)t[3] * x[base+3]
             + (long long)t[4] * x[base+4];
    }
    return acc;
}

// --- большой вектор: 8 МБ весов int8 против 1.6 МБ тритов (стриминг из памяти)
#define BIG (64 << 20)         // 64 М весов = 64 МБ int8, не влезает в кэш
static int8_t  *big8;
static uint8_t *big5;
static int8_t  *bigx;

static volatile long long big_sink;
static double bench_big8(void) {
    long long acc = 0; double t0 = now();
    for (int i = 0; i < BIG; i++) acc += (int)big8[i] * (int)bigx[i];
    big_sink += acc;
    return now() - t0;
}

static double bench_big5(void) {
    long long acc = 0; double t0 = now();
    for (int b = 0; b < BIG / 5; b++) {
        const int8_t *t = LUT5[big5[b]];
        int base = b * 5;
        acc += (long long)t[0]*bigx[base] + t[1]*bigx[base+1] + t[2]*bigx[base+2]
             + t[3]*bigx[base+3] + t[4]*bigx[base+4];
    }
    big_sink += acc;
    return now() - t0;
}

int main(void) {
    srand(42);
    for (int i = 0; i < N; i++) { w8[i] = (int8_t)(rand() % 255 - 127); x[i] = (int8_t)(rand() % 20 - 10); }
    for (int i = 0; i < N / 2; i++) { int lo = rand() % 15 - 7, hi = rand() % 15 - 7;
        w4[i] = (uint8_t)((hi & 0xF) << 4 | (lo & 0xF)); }
    for (int w = 0; w < (N + 19) / 20; w++) {
        uint32_t v = 0, p = 1;
        for (int k = 0; k < 20; k++) { v += (uint32_t)(rand() % 3) * p; p *= 3; }
        w20[w] = v;
    }

    build_lut5();
    for (int i = 0; i < N / 5; i++) w5[i] = (uint8_t)(rand() % 243);

    long long r8 = dot8(), r4 = dot4(), r20 = dot20(), r5 = dot5_lut();
    printf("контроль: int8=%lld int4=%lld триты20=%lld триты5lut=%lld\n", r8, r4, r20, r5);

    double t8 = bench(dot8), t4 = bench(dot4), t20 = bench(dot20), t5 = bench(dot5_lut);
    double b8 = N, b4 = N / 2.0, b20 = ((N + 19) / 20) * 4.0, b5 = N / 5.0;

    printf("\n%-10s %10s %12s %14s %12s\n", "формат", "байт/вес", "байт на вектор", "нс на вектор", "ГБ/с чтения");
    printf("%-10s %10.2f %12.0f %14.1f %12.2f\n", "int8", 1.0, b8, t8 * 1e9, b8 / t8 / 1e9);
    printf("%-10s %10.2f %12.0f %14.1f %12.2f\n", "int4", 0.5, b4, t4 * 1e9, b4 / t4 / 1e9);
    printf("%-10s %10.2f %12.0f %14.1f %12.2f\n", "триты", 0.20, b20, t20 * 1e9, b20 / t20 / 1e9);
    printf("%-10s %10.2f %12.0f %14.1f %12.2f\n", "триты+LUT", 0.20, b5, t5 * 1e9, b5 / t5 / 1e9);
    printf("\nпамять: триты %.1fx меньше int8; счёт: наивно %.2fx, LUT %.2fx (1.0 = как int8)\n",
           b8 / b20, t8 / t20, t8 / t5);

    // стриминг из памяти: 1М весов
    big8 = malloc(BIG); big5 = malloc(BIG / 5); bigx = malloc(BIG);
    for (int i = 0; i < BIG; i++) { big8[i] = (int8_t)(rand()%255-127); bigx[i] = (int8_t)(rand()%20-10); }
    for (int i = 0; i < BIG / 5; i++) big5[i] = (uint8_t)(rand() % 243);
    double tb8 = bench_big8(), tb5 = bench_big5();
    printf("большой вектор (%d М весов, вне кэша):\n", BIG >> 20);
    printf("  int8      %7.1f мс, %5.1f ГБ/с прочитано\n", tb8*1e3, (BIG/1e9)/tb8);
    printf("  триты+LUT %7.1f мс, %5.1f ГБ/с прочитано → %.2fx по времени\n",
           tb5*1e3, (BIG/5.0/1e9)/tb5, tb8/tb5);
    printf("  (проверка, чтобы не выкинули: %lld)\n", big_sink);
    return 0;
}
