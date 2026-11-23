// Стресс-тест пула потоков для ThreadSanitizer (задача 0121).
//
// `make test` гоняет пул только в одном потоке-отправителе (модель считает
// слой за слоем) и не проверяет ни одновременную отправку заданий, ни гонку
// vtm_threads_set/чтения числа потоков, ни заполнение LUT243 из двух потоков.
// Этот тест закрывает ровно эти сценарии; он же — регрессия на найденные
// TSan гонки (см. задачу 0121).
//
// Что проверяется:
//   1) 1..8 потоков: каждая строка [0, n) посчитана ровно один раз, tid в
//      [0, VTM_THREADS_MAX), результат не зависит от числа потоков;
//   2) маленькие n (2, 3) при 8 потоках — лишние потоки не должны ломать
//      раздачу;
//   3) несколько потоков-отправителей одновременно: API обещает
//      сериализацию заданий (vtm_backend.h); до 0121 её не было — TSan
//      показывал гонки на total/chunk/cmd и underflow workers_left;
//   4) vtm_threads_set() параллельно с заданиями: поле переопределения
//      должно быть атомарным;
//   5) пробуждение воркеров, уже уснувших в cond_wait (потерянное
//      пробуждение — один из двух дедлоков 0113);
//   6) два потока одновременно зовут tmv_lut (LUT243): таблица обязана
//      заполняться через pthread_once.
//
// Пул не умеет уничтожаться: в API нет vtm_threads_shutdown(), воркеры
// detached и живут до конца процесса. Сценарий «уничтожение во время работы»
// поэтому не проверяется — его нет в контракте; проверяем лишь, что процесс
// завершается с простаивающими воркерами (main просто return).
#include "vtm_backend.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define ROWS_BIG 257      // больше потоков и не кратно им — раздача порциями
#define WORK_BIG 1e9      // заведомо выше VTM_PAR_MIN_WORK

static int g_fail;

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        g_fail++; \
    } \
} while (0)

// Задание строчного ядра: hits[r]++ — каждая строка должна получить ровно
// одно приращение. calls/seen — по слоту на логический tid, чтобы в самом
// тесте не было гонок (у каждого tid свой индекс).
typedef struct {
    int n;
    int *hits;
    long calls[VTM_THREADS_MAX];
    unsigned char seen[VTM_THREADS_MAX];
    long bad;
} row_job;

static void count_rows(void *p, int tid, int r0, int r1) {
    row_job *j = (row_job *)p;
    int r;

    if (tid < 0 || tid >= VTM_THREADS_MAX) {
        j->bad++;
        return;
    }
    j->seen[tid] = 1;
    j->calls[tid]++;
    if (r0 < 0 || r1 > j->n || r0 > r1) {
        j->bad++;
        return;
    }
    for (r = r0; r < r1; r++) j->hits[r]++;
}

static void job_init(row_job *j, int n) {
    memset(j, 0, sizeof *j);
    j->n = n;
    j->hits = (int *)calloc((size_t)n, sizeof(int));
    if (!j->hits) {
        fprintf(stderr, "tsan_pool: нет памяти на hits[%d]\n", n);
        exit(2);
    }
}

static void job_free(row_job *j) {
    free(j->hits);
    j->hits = NULL;
}

// 1 — раскладка верная; печатает подробности при ошибке.
static int job_ok(const row_job *j, int want_threads, const char *what) {
    long calls = 0;
    int tids = 0, r, bad = 0;

    for (r = 0; r < VTM_THREADS_MAX; r++) {
        calls += j->calls[r];
        if (j->seen[r]) tids++;
    }
    for (r = 0; r < j->n; r++)
        if (j->hits[r] != 1) {
            if (bad < 3)
                fprintf(stderr, "FAIL %s: строка %d посчитана %d раз\n",
                        what, r, j->hits[r]);
            bad++;
        }
    if (j->bad || bad || calls <= 0 || tids <= 0 || tids > want_threads) {
        fprintf(stderr, "FAIL %s: bad=%ld строк=%d вызовов=%ld tid=%d "
                        "(ждали 1..%d)\n",
                what, j->bad, bad, calls, tids, want_threads);
        return 0;
    }
    return 1;
}

static void run_one(int nthreads, int n, int iters, const char *what) {
    row_job j;
    int it;

    job_init(&j, n);
    vtm_threads_set(nthreads);
    for (it = 0; it < iters; it++) {
        memset(j.hits, 0, (size_t)n * sizeof(int));
        memset(j.calls, 0, sizeof j.calls);
        memset(j.seen, 0, sizeof j.seen);
        j.bad = 0;
        vtm_parallel_rows(n, WORK_BIG, count_rows, &j);
        if (!job_ok(&j, nthreads, what)) {
            g_fail++;
            break;
        }
    }
    job_free(&j);
}

// 1) 1..8 потоков.
static void test_thread_sweep(void) {
    int nt;

    printf("  слоты/детерминизм: nt=1..8 по 40 заданий n=%d\n", ROWS_BIG);
    for (nt = 1; nt <= VTM_THREADS_MAX; nt++)
        run_one(nt, ROWS_BIG, 40, "sweep");
    // 2) n меньше числа потоков.
    run_one(VTM_THREADS_MAX, 2, 30, "n=2");
    run_one(VTM_THREADS_MAX, 3, 30, "n=3");
    run_one(VTM_THREADS_MAX, 1, 10, "n=1 (inline)");
}

// 3) Параллельные отправители. До 0121 два отправителя одновременно
// проходили wait_prev_done(), публиковали один gen и портили общий next_row;
// TSan показывал гонки, а прогон зависал.
#define CALLERS 4
#define CALLER_ITERS 60

static void *caller_thread(void *p) {
    row_job *j = (row_job *)p;
    int it;

    for (it = 0; it < CALLER_ITERS; it++) {
        memset(j->hits, 0, (size_t)j->n * sizeof(int));
        memset(j->calls, 0, sizeof j->calls);
        memset(j->seen, 0, sizeof j->seen);
        j->bad = 0;
        vtm_parallel_rows(j->n, WORK_BIG, count_rows, j);
        if (!job_ok(j, VTM_THREADS_MAX, "callers")) {
            j->bad++;
            return NULL;
        }
    }
    return NULL;
}

static void test_parallel_callers(void) {
    pthread_t th[CALLERS];
    row_job jobs[CALLERS];
    int i;

    printf("  параллельные отправители: %d потоков по %d заданий\n",
           CALLERS, CALLER_ITERS);
    vtm_threads_set(VTM_THREADS_MAX);
    for (i = 0; i < CALLERS; i++) {
        job_init(&jobs[i], 128);
        if (pthread_create(&th[i], NULL, caller_thread, &jobs[i]) != 0) {
            fprintf(stderr, "FAIL: pthread_create\n");
            g_fail++;
            job_free(&jobs[i]);
            while (--i >= 0) {
                pthread_join(th[i], NULL);
                if (jobs[i].bad) g_fail++;
                job_free(&jobs[i]);
            }
            return;
        }
    }
    for (i = 0; i < CALLERS; i++) {
        pthread_join(th[i], NULL);
        if (jobs[i].bad) g_fail++;
        job_free(&jobs[i]);
    }
}

// 4) vtm_threads_set() параллельно с заданиями. Контракт просит так не
// делать, но гонка на простом int — это UB; TSan ловил её до 0121.
static void *setter_thread(void *p) {
    int k;
    (void)p;
    for (k = 0; k < 500; k++) vtm_threads_set(1 + (k % VTM_THREADS_MAX));
    return NULL;
}

static void test_threads_set_race(void) {
    pthread_t th[CALLERS + 1];
    row_job jobs[CALLERS];
    int i;

    printf("  vtm_threads_set() параллельно с %d отправителями\n", CALLERS);
    vtm_threads_set(VTM_THREADS_MAX);
    for (i = 0; i < CALLERS; i++) {
        job_init(&jobs[i], 96);
        if (pthread_create(&th[i], NULL, caller_thread, &jobs[i]) != 0) {
            fprintf(stderr, "FAIL: pthread_create\n");
            g_fail++;
            job_free(&jobs[i]);
            while (--i >= 0) {
                pthread_join(th[i], NULL);
                if (jobs[i].bad) g_fail++;
                job_free(&jobs[i]);
            }
            return;
        }
    }
    if (pthread_create(&th[CALLERS], NULL, setter_thread, NULL) != 0) {
        fprintf(stderr, "FAIL: pthread_create setter\n");
        g_fail++;
    } else {
        pthread_join(th[CALLERS], NULL);
    }
    for (i = 0; i < CALLERS; i++) {
        pthread_join(th[i], NULL);
        if (jobs[i].bad) g_fail++;
        job_free(&jobs[i]);
    }
}

// 5) Воркеры успели заснуть: спин 100 мкс, потом cond_wait. Задание после
// паузы проверяет, что пробуждение не теряется (дедлок 0113).
static void test_idle_wake(void) {
    struct timespec ts;
    int it;

    printf("  пробуждение после 200 мс простоя\n");
    run_one(VTM_THREADS_MAX, 64, 20, "прогрев");
    ts.tv_sec = 0;
    ts.tv_nsec = 200L * 1000L * 1000L;
    nanosleep(&ts, NULL);
    for (it = 0; it < 20; it++)
        run_one(VTM_THREADS_MAX, 64, 1, "после простоя");
}

// 6) Два потока одновременно зовут tmv_lut(): LUT243 заполняется через
// pthread_once, иначе TSan видел гонку write/read по таблице.
#define LUT_ROWS 256
#define LUT_COLS 256

typedef struct {
    const uint8_t *words;
    const float *scales;
    const float *x;
    int bad;
} lut_job;

// Флаг старта: barrier не годится — при отказе pthread_create он оставил бы
// уже созданные потоки навсегда ждать недостающего. Атомарный флаг будит
// всех, кто создался; атомики же не дают TSan увидеть гонку в самом тесте.
static int g_lut_start;

static void *lut_thread(void *p) {
    lut_job *j = (lut_job *)p;
    float *y = (float *)calloc(LUT_ROWS, sizeof(float));
    struct timespec pause;
    int k, r;

    if (!y) {
        j->bad = 1;
        return NULL;
    }
    pause.tv_sec = 0;
    pause.tv_nsec = 100000L;   // 0,1 мс
    while (!__atomic_load_n(&g_lut_start, __ATOMIC_ACQUIRE))
        nanosleep(&pause, NULL);
    for (k = 0; k < 10; k++) {
        if (vtm_backend_scalar_lut.tmv(y, j->words, j->scales, j->x,
                                       LUT_ROWS, LUT_COLS) != 0)
            j->bad = 1;
        for (r = 0; r < LUT_ROWS; r++)
            if (y[r] != 0.0f) j->bad = 1;   // нулевые веса → нулевой выход
    }
    free(y);
    return NULL;
}

static void test_concurrent_lut(void) {
    const int n_trits = LUT_ROWS * LUT_COLS;
    const size_t n_words = ((size_t)n_trits + 19u) / 20u;
    const size_t n_scales = ((size_t)n_trits + 159u) / 160u;
    uint32_t *words = (uint32_t *)calloc(n_words, sizeof(uint32_t));
    float *scales = (float *)calloc(n_scales, sizeof(float));
    float *x = (float *)calloc((size_t)LUT_COLS, sizeof(float));
    pthread_t th[CALLERS];
    lut_job jobs[CALLERS];
    int started[CALLERS];
    int i;

    printf("  LUT243: %d потоков одновременно в tmv_lut %dx%d\n",
           CALLERS, LUT_ROWS, LUT_COLS);
    if (!words || !scales || !x) {
        fprintf(stderr, "tsan_pool: нет памяти на LUT-прогон\n");
        g_fail++;
        free(words);
        free(scales);
        free(x);
        return;
    }
    __atomic_store_n(&g_lut_start, 0, __ATOMIC_RELEASE);
    for (i = 0; i < CALLERS; i++) {
        jobs[i].words = (const uint8_t *)words;
        jobs[i].scales = scales;
        jobs[i].x = x;
        jobs[i].bad = 0;
        started[i] = 0;
        if (pthread_create(&th[i], NULL, lut_thread, &jobs[i]) == 0)
            started[i] = 1;
        else {
            fprintf(stderr, "FAIL: pthread_create lut\n");
            g_fail++;
        }
    }
    __atomic_store_n(&g_lut_start, 1, __ATOMIC_RELEASE);
    for (i = 0; i < CALLERS; i++) {
        if (started[i]) pthread_join(th[i], NULL);
        if (jobs[i].bad) {
            fprintf(stderr, "FAIL: tmv_lut в потоке %d\n", i);
            g_fail++;
        }
    }
    free(words);
    free(scales);
    free(x);
}

// Сколько секунд ждать до принудительного выхода: регрессия пула — это
// дедлок, и без сторожа тест повис бы навсегда. Переопределяется при сборке
// (-DTSAN_POOL_ALARM=15), чтобы проверить сам тест на старом коде.
#ifndef TSAN_POOL_ALARM
#define TSAN_POOL_ALARM 240
#endif

int main(void) {
    alarm(TSAN_POOL_ALARM);   // тест не должен висеть вечно при регрессии
    printf("tsan_pool: стресс пула (ThreadSanitizer-регрессия, задача 0121)\n");

    test_thread_sweep();
    test_parallel_callers();
    test_threads_set_race();
    test_idle_wake();
    test_concurrent_lut();

    if (g_fail) {
        printf("tsan_pool: ПРОВАЛ (сбоев %d)\n", g_fail);
        return 1;
    }
    printf("tsan_pool: ok (слоты, 1..8 потоков, параллельные отправители, "
           "vtm_threads_set, пробуждение, LUT243)\n");
    return 0;
}
