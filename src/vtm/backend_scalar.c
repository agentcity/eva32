// Скалярные ядра VTM. Два варианта TMV:
//   * LUT243  — слово pack20 делится на 4 группы по 5 тритов, группа
//               декодируется таблицей на 243 записи (1,2 КБ, живёт в L1);
//   * direct  — триты достаются делением слова на 3.
// Оба варианта считают одно и то же; разница только в скорости. Замер —
// `./vtm bench` и таблица в src/vtm/README.md.
//
// Здесь же — пул потоков (pthread) для строчных ядер TMV/MM/RMSN: этот TU
// собирается в любом варианте сборки, поэтому пул живёт в нём, а ядра
// SIMD-бэкендов зовут те же vtm_threads()/vtm_parallel_rows().
//
// _DARWIN_C_SOURCE нужен до первого include: на macOS sysconf(_SC_NPROCESSORS_ONLN)
// спрятан за __DARWIN_C_FULL, а Makefile задаёт строгий _POSIX_C_SOURCE.
// Константа нужна только для «по умолчанию — число ядер».
#if defined(__APPLE__) && !defined(_DARWIN_C_SOURCE)
#define _DARWIN_C_SOURCE 1
#endif
#include "vtm.h"
#include "vtm_backend.h"

#include <math.h>
#include <string.h>

#if !defined(_WIN32) && !defined(VTM_NO_THREADS)
#define VTM_HAVE_PTHREAD 1
#include <pthread.h>
#include <unistd.h>
#else
#define VTM_HAVE_PTHREAD 0
#endif

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// --- пул потоков (задача 0077) --------------------------------------------
//
// Постоянные воркеры создаются лениво при первом задании выше порога и живут
// до конца процесса (detached). Строки [0, n) делятся на N непрерывных
// диапазонов (статическое деление): строка TMV/MM/RMSN стоит одинаково,
// поэтому балансировка не нужна, а мьютекс на каждую строку не нужен тем
// более. Строки независимы, и каждая считается тем же кодом в том же
// порядке, что и при одном потоке, поэтому результат побитово не зависит ни
// от числа потоков, ни от того, кому достался диапазон.
//
// Вызовы из разных потоков сериализуются: задание одно на пул. Вложенный
// вызов из колбэка (сейчас таких нет) привёл бы к взаимной блокировке —
// поэтому в ядрах ровно один уровень параллелизма.

#if VTM_HAVE_PTHREAD

// Горячий путь задания — без мьютекса. Раньше воркер на каждое задание
// брал мьютекс дважды (прочитать параметры, сообщить о готовности), а
// вызвавший после своего диапазона ждал, пока последний воркер под тем же
// мьютексом выставит job = 0. На мелких заданиях (TMV k_proj: 0,39 MMAC,
// ~55 мкс) это стоило дороже самого счёта: `bench_backend --pool` на
// прежнем пуле даёт 4,95 мкс на задание при 4 потоках и 51,7 мкс при 8
// (теперь 1,1 и 1,4 мкс). На 1.5B-форме это была половина времени 8
// потоков. Теперь параметры задания публикуются одним release-сохранением
// gen, а «задание досчитано» — отдельным счётчиком done, который двигает
// только тот, чей декремент workers_left дал ноль.
//
// Мьютекс остался только на медленном пути — заснуть, когда работы нет, и
// проснуться, когда её опубликовали. В обычном цикле ядер модель зовёт
// TMV/MM подряд, и воркеры успевают поймать gen на спине.
//
// У каждого воркера свой слот команды (g_cmd[tid]) с номером задания.
// Общий на всех слот параметров читать нельзя: воркер, не участвовавший в
// задании (tid >= nthreads), может прочитать поле уже следующего задания и
// лишний раз декрементировать workers_left — счётчик «съедет», и вызвавший
// зависнет в ожидании нуля (такой редкий дедлок ловился на 1.5B-форме).
// Слот свой у каждого, номер задания строго растёт, поэтому совпадение
// g_cmd[tid].gen == gen означает ровно «этот поток — участник этого
// задания», а писатель слота (вызвавший) не тронет его, пока воркер не
// декрементировал workers_left.

typedef struct {
    vtm_rows_fn fn;      // ядро строк
    void *ctx;           // его контекст
    unsigned gen;        // номер задания (0 — не участник)
} vtm_cmd;

typedef struct {
    pthread_mutex_t mu;
    pthread_cond_t work_cv;   // воркеры ждут новое задание
    pthread_cond_t done_cv;   // вызвавший ждёт, пока досчитают диапазоны
    int workers;              // сколько воркеров создано (без вызвавшего)
    int create_failed;        // pthread_create отказал — больше не пробуем
    unsigned gen;             // номер опубликованного задания (атомарно)
    unsigned sub;             // заданий выдано (атомарно)
    unsigned done;            // заданий досчитано (атомарно, монотонно)
    unsigned workers_left;    // сколько потоков ещё не закончили задание
    unsigned sleep_work;      // воркеров в cond_wait(work_cv)
    unsigned sleep_done;      // вызвавших в cond_wait(done_cv)
    unsigned next_row;        // первая невзятая строка (динамическая раздача)
    int total;                // строк в задании
    int chunk;                // строк в одной порции
    vtm_cmd cmd[VTM_THREADS_MAX];   // по слоту на tid
} vtm_pool;

static vtm_pool g_pool = {
    PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER,
    PTHREAD_COND_INITIALIZER, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    {{0, NULL, 0}}
};

// Спин перед сном. Ядра вызываются подряд (слой модели — десятки вызовов),
// а пробуждение спящего на cond_wait потока на macOS стоит десятки
// микросекунд — для матрицы 1024x1024 SIMD это половина выигрыша. Спин
// ограничен VTM_POOL_SPIN_NS, чтобы в простое поток засыпал и не жёг ядро.
#define VTM_POOL_SPIN_NS 100000L   // 100 мкс

#if defined(__GNUC__) || defined(__clang__)
#define VTM_GEN_LOAD() __atomic_load_n(&g_pool.gen, __ATOMIC_ACQUIRE)
#define VTM_GEN_STORE(v) __atomic_store_n(&g_pool.gen, (v), __ATOMIC_RELEASE)
#define VTM_SUB_LOAD() __atomic_load_n(&g_pool.sub, __ATOMIC_ACQUIRE)
#define VTM_SUB_STORE(v) __atomic_store_n(&g_pool.sub, (v), __ATOMIC_RELEASE)
#define VTM_DONE_LOAD() __atomic_load_n(&g_pool.done, __ATOMIC_ACQUIRE)
#define VTM_DONE_STORE(v) \
    __atomic_store_n(&g_pool.done, (unsigned)(v), __ATOMIC_RELEASE)
#define VTM_LEFT_LOAD() __atomic_load_n(&g_pool.workers_left, __ATOMIC_ACQUIRE)
#define VTM_LEFT_STORE(v) \
    __atomic_store_n(&g_pool.workers_left, (unsigned)(v), __ATOMIC_RELEASE)
#define VTM_LEFT_DEC() \
    __atomic_sub_fetch(&g_pool.workers_left, 1u, __ATOMIC_ACQ_REL)
#define VTM_SLEEP_WORK_LOAD() \
    __atomic_load_n(&g_pool.sleep_work, __ATOMIC_ACQUIRE)
#define VTM_SLEEP_WORK_ADD(d) \
    __atomic_add_fetch(&g_pool.sleep_work, (unsigned)(d), __ATOMIC_ACQ_REL)
#define VTM_SLEEP_DONE_LOAD() \
    __atomic_load_n(&g_pool.sleep_done, __ATOMIC_ACQUIRE)
#define VTM_SLEEP_DONE_ADD(d) \
    __atomic_add_fetch(&g_pool.sleep_done, (unsigned)(d), __ATOMIC_ACQ_REL)
#define VTM_NEXT_ROW() \
    __atomic_fetch_add(&g_pool.next_row, (unsigned)g_pool.chunk, \
                       __ATOMIC_RELAXED)
#define VTM_CMD_GEN_LOAD(i) \
    __atomic_load_n(&g_pool.cmd[(i)].gen, __ATOMIC_ACQUIRE)
#define VTM_CMD_GEN_STORE(i, v) \
    __atomic_store_n(&g_pool.cmd[(i)].gen, (v), __ATOMIC_RELEASE)
#else
#define VTM_GEN_LOAD() (g_pool.gen)
#define VTM_GEN_STORE(v) (g_pool.gen = (v))
#define VTM_SUB_LOAD() (g_pool.sub)
#define VTM_SUB_STORE(v) (g_pool.sub = (v))
#define VTM_DONE_LOAD() (g_pool.done)
#define VTM_DONE_STORE(v) (g_pool.done = (unsigned)(v))
#define VTM_LEFT_LOAD() (g_pool.workers_left)
#define VTM_LEFT_STORE(v) (g_pool.workers_left = (unsigned)(v))
#define VTM_LEFT_DEC() (--g_pool.workers_left)
#define VTM_SLEEP_WORK_LOAD() (g_pool.sleep_work)
#define VTM_SLEEP_WORK_ADD(d) (g_pool.sleep_work += (unsigned)(d))
#define VTM_SLEEP_DONE_LOAD() (g_pool.sleep_done)
#define VTM_SLEEP_DONE_ADD(d) (g_pool.sleep_done += (unsigned)(d))
#define VTM_NEXT_ROW() (g_pool.next_row += (unsigned)g_pool.chunk)
#define VTM_CMD_GEN_LOAD(i) (g_pool.cmd[(i)].gen)
#define VTM_CMD_GEN_STORE(i, v) (g_pool.cmd[(i)].gen = (v))
#endif

static void cpu_relax(void) {
#if defined(__aarch64__)
    __asm__ __volatile__("yield");
#elif defined(__x86_64__) || defined(__i386__)
    __asm__ __volatile__("pause");
#else
    __asm__ __volatile__("");
#endif
}

static long ns_between(const struct timespec *a, const struct timespec *b) {
    return (long)(b->tv_sec - a->tv_sec) * 1000000000L
         + (long)(b->tv_nsec - a->tv_nsec);
}

/* Дождаться, пока все диапазоны досчитают, коротким спином по счётчику.
   Пробуждение из cond_wait на критическом пути замера стоит десятки мкс.
   1 — дождались, 0 — истёк спин (дальше — cond_wait). */
static int wait_left_zero(void) {
    struct timespec t0, t;
    long spins = 0;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    while (VTM_LEFT_LOAD() != 0) {
        if ((++spins & 127) == 0) {
            clock_gettime(CLOCK_MONOTONIC, &t);
            if (ns_between(&t0, &t) >= VTM_POOL_SPIN_NS) return 0;
        }
        cpu_relax();
    }
    return 1;
}

/* Задание досчитано: done двигается только вперёд (номера заданий растут),
   поэтому запоздавший воркер не может откатить счётчик следующего задания.
   Если кто-то спит в ожидании done — будим. */
static void set_done(unsigned g) {
    pthread_mutex_lock(&g_pool.mu);
    VTM_DONE_STORE(g);   // done только растёт: g — номер закрытого задания
    if (VTM_SLEEP_DONE_LOAD())
        pthread_cond_broadcast(&g_pool.done_cv);
    pthread_mutex_unlock(&g_pool.mu);
}

// Аргумент воркера: (gen на момент создания) << 8 | tid. Номер задания
// важен: новый воркер создаётся, когда предыдущее задание уже закрыто, и
// без него он принял бы in-flight gen за новое задание и посчитал бы
// чужой диапазон по старым параметрам.
static void *pool_worker(void *arg) {
    const uintptr_t packed = (uintptr_t)arg;
    const int tid = (int)(packed & 0xffu);
    unsigned seen = (unsigned)(packed >> 8);

    for (;;) {
        struct timespec t0;
        long spins = 0;
        unsigned g;

        // Спин на gen: пока задание не опубликовано, не трогаем мьютекс.
        clock_gettime(CLOCK_MONOTONIC, &t0);
        while (VTM_GEN_LOAD() == seen) {
            struct timespec t;
            if ((++spins & 127) == 0) {
                clock_gettime(CLOCK_MONOTONIC, &t);
                if (ns_between(&t0, &t) >= VTM_POOL_SPIN_NS) break;
            }
            cpu_relax();
        }
        g = VTM_GEN_LOAD();
        if (g == seen) {
            // Спин истёк — засыпаем до публикации следующего задания.
            pthread_mutex_lock(&g_pool.mu);
            VTM_SLEEP_WORK_ADD(1);
            while (VTM_GEN_LOAD() == seen)
                pthread_cond_wait(&g_pool.work_cv, &g_pool.mu);
            VTM_SLEEP_WORK_ADD(-1);
            g = VTM_GEN_LOAD();
            pthread_mutex_unlock(&g_pool.mu);
        }
        /* Свой слот: gen в нём равен g только у участника задания. Гонки с
           писателем нет — вызвавший пишет слот, только когда workers_left
           уже ноль, то есть этот воркер давно декрементировал. */
        if (VTM_CMD_GEN_LOAD(tid) == g) {
            vtm_rows_fn fn = g_pool.cmd[tid].fn;
            void *ctx = g_pool.cmd[tid].ctx;
            const int total = g_pool.total;
            const int chunk = g_pool.chunk;
            for (;;) {
                int r0 = (int)VTM_NEXT_ROW();
                int r1;
                if (r0 >= total) break;   // порции кончились
                r1 = r0 + chunk;
                if (r1 > total) r1 = total;
                fn(ctx, tid, r0, r1);
            }
            if (VTM_LEFT_DEC() == 0) set_done(g);
        }
        seen = g;
    }
    return NULL;
}

/* Ждать, пока предыдущее задание закроется. В норме (один отправитель —
   модель считает слой за слоем в одном потоке) done уже равен sub, и это
   одна атомарная загрузка; спин и сон — для чужого отправителя и для
   окна в пару наносекунд между декрементом и set_done. */
static void wait_prev_done(void) {
    unsigned spins = 0;
    while (VTM_DONE_LOAD() != VTM_SUB_LOAD()) {
        if (++spins > 4096) {
            pthread_mutex_lock(&g_pool.mu);
            VTM_SLEEP_DONE_ADD(1);
            while (VTM_DONE_LOAD() != VTM_SUB_LOAD())
                pthread_cond_wait(&g_pool.done_cv, &g_pool.mu);
            VTM_SLEEP_DONE_ADD(-1);
            pthread_mutex_unlock(&g_pool.mu);
            return;
        }
        cpu_relax();
    }
}

// Выполнить задание в пуле: N диапазонов строк, вызвавший — диапазон 0
// (tid 0), остальные — воркеры 1..N-1. Создаёт недостающих воркеров (до
// nthreads-1) под мьютексом; если создать не удалось — считает тем, что
// есть, вплоть до одного потока.
static void pool_run(int n, int nthreads, vtm_rows_fn fn, void *ctx) {
    unsigned mygen;
    int i, chunk;

    wait_prev_done();
    if (!g_pool.create_failed && g_pool.workers < nthreads - 1) {
        pthread_mutex_lock(&g_pool.mu);
        while (!g_pool.create_failed && g_pool.workers < nthreads - 1) {
            pthread_t th;
            uintptr_t arg = ((uintptr_t)VTM_GEN_LOAD() << 8)
                          | (uintptr_t)(g_pool.workers + 1);
            if (pthread_create(&th, NULL, pool_worker, (void *)arg) != 0) {
                g_pool.create_failed = 1;
                break;
            }
            pthread_detach(th);
            g_pool.workers++;
        }
        pthread_mutex_unlock(&g_pool.mu);
    }
    if (g_pool.workers < nthreads - 1) nthreads = g_pool.workers + 1;
    if (nthreads > n) nthreads = n;   // строк меньше потоков — лишние не нужны
    if (nthreads < 2) {
        fn(ctx, 0, 0, n);
        return;
    }
    /* Порция раздачи: каждый поток берёт ~8 порций, чтобы медленный (чужой
       поток планировщика, E-ядро) не задерживал барьер. Строки независимы,
       поэтому порядок и число порций на результат не влияют. */
    chunk = n / (nthreads * 8);
    if (chunk < 1) chunk = 1;
    mygen = VTM_SUB_LOAD() + 1;
    /* Слоты участников пишутся до публикации gen; слоты остальных не
       трогаем — у них там gen прошлых заданий, а номера не повторяются. */
    g_pool.total = n;
    g_pool.chunk = chunk;
    g_pool.next_row = 0;
    for (i = 0; i < nthreads; i++) {
        g_pool.cmd[i].fn = fn;
        g_pool.cmd[i].ctx = ctx;
        VTM_CMD_GEN_STORE(i, mygen);
    }
    VTM_LEFT_STORE(nthreads);
    VTM_SUB_STORE(mygen);
    /* Публикация под мьютексом, хотя быстрые пути его не берут. Иначе
       теряется пробуждение: воркер успевает прочитать старый gen, решить
       «работы нет» и заснуть, а вызвавший в это же время видит
       sleep_work == 0 и не будит — оба спят (дедлок ловился на 1.5B).
       Мьютекс сериализует store gen с проверкой gen в воркере. */
    pthread_mutex_lock(&g_pool.mu);
    VTM_GEN_STORE(mygen);   // release: слоты участников опубликованы
    if (VTM_SLEEP_WORK_LOAD())
        pthread_cond_broadcast(&g_pool.work_cv);
    pthread_mutex_unlock(&g_pool.mu);

    // Вызвавший — тоже участник: берёт порции из общей раздачи.
    for (;;) {
        int r0 = (int)VTM_NEXT_ROW();
        int r1;
        if (r0 >= n) break;
        r1 = r0 + chunk;
        if (r1 > n) r1 = n;
        fn(ctx, 0, r0, r1);
    }

    if (VTM_LEFT_DEC() == 0) {
        set_done(mygen);
        return;
    }
    if (wait_left_zero()) {
        // Диапазоны досчитаны; done выставит тот, чей декремент дал ноль.
        // Если он ещё не успел — следующее задание подождёт его в
        // wait_prev_done (окно в пару наносекунд, без мьютекса).
        return;
    }
    pthread_mutex_lock(&g_pool.mu);
    VTM_SLEEP_DONE_ADD(1);
    while (VTM_DONE_LOAD() < mygen)
        pthread_cond_wait(&g_pool.done_cv, &g_pool.mu);
    VTM_SLEEP_DONE_ADD(-1);
    pthread_mutex_unlock(&g_pool.mu);
}

#endif // VTM_HAVE_PTHREAD

// --- число потоков ---------------------------------------------------------

#if VTM_HAVE_PTHREAD
static pthread_once_t g_cfg_once = PTHREAD_ONCE_INIT;
static int g_cfg_threads = 1;    // из VTM_THREADS или по числу ядер
static int g_cfg_override;       // 0 — переопределения нет (тест-хук)

static void cfg_init(void) {
    const char *env = getenv("VTM_THREADS");
    long n = 0, cores;

    if (env && *env) {
        char *end = NULL;
        n = strtol(env, &end, 10);
        if (!end || *end != '\0' || n <= 0 || n > 1000000L) {
            fprintf(stderr, "vtm: VTM_THREADS='%s' — не число из 1..%d, "
                            "беру по числу ядер\n", env, VTM_THREADS_MAX);
            n = 0;
        }
    }
    if (n <= 0) {
#if defined(_SC_NPROCESSORS_ONLN)
        cores = sysconf(_SC_NPROCESSORS_ONLN);
#else
        cores = 4;   // константа недоступна (не POSIX-система) — дефолт
#endif
        if (cores < 1) cores = 1;
        n = cores;
    }
    if (n > VTM_THREADS_MAX) n = VTM_THREADS_MAX;
    g_cfg_threads = (int)n;
}

int vtm_threads(void) {
    pthread_once(&g_cfg_once, cfg_init);
    return g_cfg_override > 0 ? g_cfg_override : g_cfg_threads;
}

int vtm_threads_set(int n) {
    pthread_once(&g_cfg_once, cfg_init);
    if (n <= 0) n = g_cfg_threads;
    if (n > VTM_THREADS_MAX) n = VTM_THREADS_MAX;
    g_cfg_override = n;
    return n;
}

// Минимум работы на поток. Меньше ~65 тыс. MAC (десяток микросекунд) —
// барьер пула дороже выигрыша от лишнего потока.
#define VTM_PAR_CHUNK 65536.0

// Потолок параллелизма одного задания. Замер на 1.5B-форме (10 ядер,
// задания TMV 0,39–13,8 MMAC, 224 мс/токен при 4 потоках):
//   потоки 4     5     6     7     8
//   ток/с  15,5  17,7  17,9  16,1  14,6
// Дальше шести выигрыша нет: задание короткое (сотни микросекунд), и
// накладные расходы барьера — разбудить воркеров и дождаться самого
// медленного — съедают прибавку, а на занятой машине лишние потоки ещё и
// стоят в очереди планировщика. VTM_THREADS остаётся верхней границей:
// просили меньше — используем меньше.
#define VTM_PAR_THREADS_TOP 6

void vtm_parallel_rows(int n, double work, vtm_rows_fn fn, void *ctx) {
    int vt, want;
    if (!fn || n <= 0) return;
    vt = vtm_threads();
    /* !(work >= порога) — заодно отсекает NaN. */
    if (vt < 2 || n < 2 || !(work >= VTM_PAR_MIN_WORK)) {
        fn(ctx, 0, 0, n);
        return;
    }
    /* Сколько потоков оправдано: не больше запрошенного, не больше
       VTM_PAR_THREADS_TOP и не больше, чем work / VTM_PAR_CHUNK (каждому
       потоку — не меньше порции работы). */
    want = (int)(work / VTM_PAR_CHUNK);
    if (want > VTM_PAR_THREADS_TOP) want = VTM_PAR_THREADS_TOP;
    if (want > vt) want = vt;
    if (want > n) want = n;
    if (want < 2) {
        fn(ctx, 0, 0, n);
        return;
    }
    pool_run(n, want, fn, ctx);
}

#else // !VTM_HAVE_PTHREAD: откат — прежнее однопоточное поведение

int vtm_threads(void) { return 1; }

int vtm_threads_set(int n) {
    (void)n;
    return 1;
}

void vtm_parallel_rows(int n, double work, vtm_rows_fn fn, void *ctx) {
    (void)work;
    if (!fn || n <= 0) return;
    fn(ctx, 0, 0, n);
}

#endif // VTM_HAVE_PTHREAD

// --- LUT на 5 тритов -------------------------------------------------------

static int8_t  LUT243[243][5];
static int     lut_ready;

static void lut_init(void) {
    unsigned v;
    if (lut_ready) return;
    for (v = 0; v < 243; v++) {
        unsigned t = v;
        int k;
        for (k = 0; k < 5; k++) {
            unsigned d = t % 3u;
            t /= 3u;
            LUT243[v][k] = (int8_t)(d == 2u ? 1 : (d == 0u ? -1 : 0));
        }
    }
    lut_ready = 1;
}

// --- TMV -------------------------------------------------------------------

static int tmv_check(const float *y, const uint8_t *w, const float *scales,
                     const float *x, int rows, int cols) {
    if (!y || !w || !scales || !x) return -1;
    if (rows < 0 || cols < 0) return -1;
    return 0;
}

// Слова pack20 и масштабы лежат в TRT1-сегментах контейнера со смещением
// 2 mod 4 (задача 0060): разыменование uint32_t*/float* по такому адресу —
// UB, даже если CPU его терпит. Читаем через memcpy: компилятор сворачивает
// байтовое копирование в одну обычную загрузку.
static inline uint32_t load_u32_unaligned(const uint32_t *words, int64_t i) {
    uint32_t v;
    memcpy(&v, (const uint8_t *)words + (size_t)i * sizeof v, sizeof v);
    return v;
}

static inline float load_f32_unaligned(const float *scales, int64_t i) {
    float v;
    memcpy(&v, (const uint8_t *)scales + (size_t)i * sizeof v, sizeof v);
    return v;
}

// Прямая распаковка: 20 делений на 3 на слово, зато без таблицы.
// scale_block — 128 (канон Python) или 160 (docs/trit-format). Если блок
// кратен 20, все 20 тритов слова лежат в одном блоке масштаба; иначе
// (128) граница может проходить внутри слова, и масштаб применяется к
// частичной сумме блока поэлементно.
// Строки [r0, r1) одного варианта ядра. Разбиение по строкам не меняет
// порядок сложения внутри строки: base по-прежнему r*cols, поэтому
// результат не зависит от числа потоков (побитово).
static void tmv_direct_rows(float *y, const uint8_t *w, const float *scales,
                            const float *x, int cols, int scale_block,
                            int r0, int r1) {
    const uint32_t *words = (const uint32_t *)w;
    int r;

    for (r = r0; r < r1; r++) {
        int64_t base = (int64_t)r * (int64_t)cols;
        int64_t end = base + (int64_t)cols;
        int64_t wi = base / (int64_t)VTM_TRITS_PER_WORD;
        int64_t wi_end = (end + VTM_TRITS_PER_WORD - 1) / VTM_TRITS_PER_WORD;
        double acc = 0.0;

        if (scale_block % VTM_TRITS_PER_WORD == 0) {
            for (; wi < wi_end; wi++) {
                uint32_t v = load_u32_unaligned(words, wi);
                int64_t wbase = wi * (int64_t)VTM_TRITS_PER_WORD;
                double block_acc = 0.0;
                int k;
                for (k = 0; k < VTM_TRITS_PER_WORD; k++) {
                    unsigned d = v % 3u;
                    int64_t idx = wbase + k;
                    v /= 3u;
                    if (idx < base) continue;
                    if (idx >= end) break;
                    if (d == 2u)      block_acc += (double)x[idx - base];
                    else if (d == 0u) block_acc -= (double)x[idx - base];
                }
                acc += block_acc
                     * (double)load_f32_unaligned(
                           scales, (wi * VTM_TRITS_PER_WORD) / scale_block);
            }
        } else {
            int64_t cur_block = -1;
            double block_acc = 0.0;
            for (; wi < wi_end; wi++) {
                uint32_t v = load_u32_unaligned(words, wi);
                int64_t wbase = wi * (int64_t)VTM_TRITS_PER_WORD;
                int k;
                for (k = 0; k < VTM_TRITS_PER_WORD; k++) {
                    unsigned d = v % 3u;
                    int64_t idx = wbase + k;
                    int64_t blk;
                    v /= 3u;
                    if (idx < base) continue;
                    if (idx >= end) break;
                    blk = idx / (int64_t)scale_block;
                    if (blk != cur_block) {
                        if (cur_block >= 0)
                            acc += block_acc
                                 * (double)load_f32_unaligned(scales,
                                                              cur_block);
                        cur_block = blk;
                        block_acc = 0.0;
                    }
                    if (d == 2u)      block_acc += (double)x[idx - base];
                    else if (d == 0u) block_acc -= (double)x[idx - base];
                }
            }
            if (cur_block >= 0)
                acc += block_acc
                     * (double)load_f32_unaligned(scales, cur_block);
        }
        y[r] = (float)acc;
    }
}

// Общий контекст задания TMV: ядро строк и параметры.
typedef struct {
    float *y;
    const uint8_t *w;
    const float *scales;
    const float *x;
    int cols;
    int scale_block;
} tmv_job;

static void tmv_direct_worker(void *p, int tid, int r0, int r1) {
    tmv_job *j = (tmv_job *)p;
    (void)tid;
    tmv_direct_rows(j->y, j->w, j->scales, j->x, j->cols, j->scale_block,
                    r0, r1);
}

static int tmv_direct_blocks(float *y, const uint8_t *w, const float *scales,
                             const float *x, int rows, int cols,
                             int scale_block) {
    tmv_job job;
    if (tmv_check(y, w, scales, x, rows, cols) != 0) return -1;
    if (scale_block != 128 && scale_block != 160) return -1;
    if (rows == 0 || cols == 0) return 0;
    job.y = y;
    job.w = w;
    job.scales = scales;
    job.x = x;
    job.cols = cols;
    job.scale_block = scale_block;
    vtm_parallel_rows(rows, (double)rows * (double)cols, tmv_direct_worker,
                      &job);
    return 0;
}

static int tmv_direct(float *y, const uint8_t *w, const float *scales,
                      const float *x, int rows, int cols) {
    return tmv_direct_blocks(y, w, scales, x, rows, cols,
                             (int)VTM_SCALE_BLOCK_160);
}

// LUT243: 4 выборки из таблицы на слово вместо 20 делений.
static void tmv_lut_rows(float *y, const uint8_t *w, const float *scales,
                         const float *x, int cols, int scale_block,
                         int r0, int r1) {
    const uint32_t *words = (const uint32_t *)w;
    int r;

    for (r = r0; r < r1; r++) {
        int64_t base = (int64_t)r * (int64_t)cols;
        int64_t end = base + (int64_t)cols;
        int64_t wi = base / (int64_t)VTM_TRITS_PER_WORD;
        int64_t wi_end = (end + VTM_TRITS_PER_WORD - 1) / VTM_TRITS_PER_WORD;
        double acc = 0.0;

        if (scale_block % VTM_TRITS_PER_WORD == 0) {
            for (; wi < wi_end; wi++) {
                uint32_t v = load_u32_unaligned(words, wi);
                int64_t wbase = wi * (int64_t)VTM_TRITS_PER_WORD;
                double block_acc = 0.0;
                int g;
                double scale = (double)load_f32_unaligned(
                    scales, (wi * VTM_TRITS_PER_WORD) / scale_block);
                for (g = 0; g < 4; g++) {
                    unsigned code = v % 243u;
                    const int8_t *trit = LUT243[code];
                    int64_t idx0 = wbase + (int64_t)g * 5;
                    int k;
                    v /= 243u;
                    for (k = 0; k < 5; k++) {
                        int64_t idx = idx0 + k;
                        if (idx < base) continue;
                        if (idx >= end) break;
                        block_acc += (double)trit[k] * (double)x[idx - base];
                    }
                }
                acc += block_acc * scale;
            }
        } else {
            int64_t cur_block = -1;
            double block_acc = 0.0;
            for (; wi < wi_end; wi++) {
                uint32_t v = load_u32_unaligned(words, wi);
                int64_t wbase = wi * (int64_t)VTM_TRITS_PER_WORD;
                int g;
                for (g = 0; g < 4; g++) {
                    unsigned code = v % 243u;
                    const int8_t *trit = LUT243[code];
                    int64_t idx0 = wbase + (int64_t)g * 5;
                    int k;
                    v /= 243u;
                    for (k = 0; k < 5; k++) {
                        int64_t idx = idx0 + k;
                        int64_t blk;
                        if (idx < base) continue;
                        if (idx >= end) break;
                        blk = idx / (int64_t)scale_block;
                        if (blk != cur_block) {
                            if (cur_block >= 0)
                                acc += block_acc
                                     * (double)load_f32_unaligned(scales,
                                                                  cur_block);
                            cur_block = blk;
                            block_acc = 0.0;
                        }
                        block_acc += (double)trit[k] * (double)x[idx - base];
                    }
                }
            }
            if (cur_block >= 0)
                acc += block_acc
                     * (double)load_f32_unaligned(scales, cur_block);
        }
        y[r] = (float)acc;
    }
}

static void tmv_lut_worker(void *p, int tid, int r0, int r1) {
    tmv_job *j = (tmv_job *)p;
    (void)tid;
    tmv_lut_rows(j->y, j->w, j->scales, j->x, j->cols, j->scale_block,
                 r0, r1);
}

static int tmv_lut_blocks(float *y, const uint8_t *w, const float *scales,
                          const float *x, int rows, int cols,
                          int scale_block) {
    tmv_job job;
    if (tmv_check(y, w, scales, x, rows, cols) != 0) return -1;
    if (scale_block != 128 && scale_block != 160) return -1;
    if (rows == 0 || cols == 0) return 0;
    lut_init();   // до потоков: таблица пишется один раз, не в воркерах
    job.y = y;
    job.w = w;
    job.scales = scales;
    job.x = x;
    job.cols = cols;
    job.scale_block = scale_block;
    vtm_parallel_rows(rows, (double)rows * (double)cols, tmv_lut_worker,
                      &job);
    return 0;
}

static int tmv_lut(float *y, const uint8_t *w, const float *scales,
                   const float *x, int rows, int cols) {
    return tmv_lut_blocks(y, w, scales, x, rows, cols,
                          (int)VTM_SCALE_BLOCK_160);
}

// --- остальные ядра --------------------------------------------------------

// Однопоточный RMSN — прежний код без изменений: при VTM_THREADS=1 и на
// малых n выполняется ровно он.
static int rmsnorm_scalar_serial(float *y, const float *x, const float *w,
                                 int n, float eps) {
    double sum = 0.0, mean, rms;
    int i;
    if (!y || !x || !w || n < 0) return -1;
    if (n == 0) return 0;
    for (i = 0; i < n; i++) sum += (double)x[i] * (double)x[i];
    mean = sum / (double)n;
    rms = sqrt(mean + (double)eps);
    if (rms == 0.0) {
        // x == 0 и eps == 0: предел 0/0 доопределяем нулём, чтобы не тащить NaN.
        memset(y, 0, (size_t)n * sizeof(float));
        return 0;
    }
    for (i = 0; i < n; i++)
        y[i] = (float)((double)x[i] / rms * (double)w[i]);
    return 0;
}

// --- RMSN по блокам для больших n -----------------------------------------
//
// Сумма квадратов режется на фиксированные блоки VTM_RMSN_BLOCK: каждый блок
// считается одним потоком целиком (порядок внутри блока прежний), затем
// частичные суммы складываются по возрастанию номера блока. Поэтому
// результат один и тот же при 2/4/8 потоках (побитово), а от
// VTM_THREADS=1 отличается только порядком сложения double — на уровне
// ~1e-16 относительной ошибки (в f32-выходе обычно не видно).
//
// Поэлементная часть — та же формула, что в serial-версии, и каждый элемент
// независим, поэтому тут совпадение побитовое при любом числе потоков.

#define VTM_RMSN_BLOCK 4096

typedef struct {
    const float *x;
    const float *w;
    float *y;
    double *sums;      // частичные суммы по блокам
    int n;
    double rms;
} rmsn_job;

static void rmsnorm_sum_worker(void *p, int tid, int b0, int b1) {
    rmsn_job *j = (rmsn_job *)p;
    int b, i;
    (void)tid;
    for (b = b0; b < b1; b++) {
        int i0 = b * VTM_RMSN_BLOCK;
        int i1 = i0 + VTM_RMSN_BLOCK;
        double sum = 0.0;
        if (i1 > j->n) i1 = j->n;
        for (i = i0; i < i1; i++) sum += (double)j->x[i] * (double)j->x[i];
        j->sums[b] = sum;
    }
}

static void rmsnorm_scale_worker(void *p, int tid, int b0, int b1) {
    rmsn_job *j = (rmsn_job *)p;
    int b, i;
    (void)tid;
    for (b = b0; b < b1; b++) {
        int i0 = b * VTM_RMSN_BLOCK;
        int i1 = i0 + VTM_RMSN_BLOCK;
        if (i1 > j->n) i1 = j->n;
        for (i = i0; i < i1; i++)
            j->y[i] = (float)((double)j->x[i] / j->rms * (double)j->w[i]);
    }
}

static int rmsnorm_scalar(float *y, const float *x, const float *w,
                          int n, float eps) {
    rmsn_job job;
    double sum = 0.0, mean, rms;
    int nblocks, b;

    if (!y || !x || !w || n < 0) return -1;
    if (n == 0) return 0;
    if (vtm_threads() < 2 || n < VTM_PAR_RMSN_MIN)
        return rmsnorm_scalar_serial(y, x, w, n, eps);

    nblocks = (n + VTM_RMSN_BLOCK - 1) / VTM_RMSN_BLOCK;
    job.x = x;
    job.w = w;
    job.y = y;
    job.n = n;
    job.sums = (double *)calloc((size_t)nblocks, sizeof(double));
    if (!job.sums)   // нет памяти — считаем как раньше
        return rmsnorm_scalar_serial(y, x, w, n, eps);

    vtm_parallel_rows(nblocks, 2.0 * (double)n, rmsnorm_sum_worker, &job);
    for (b = 0; b < nblocks; b++) sum += job.sums[b];
    free(job.sums);

    mean = sum / (double)n;
    rms = sqrt(mean + (double)eps);
    if (rms == 0.0) {
        memset(y, 0, (size_t)n * sizeof(float));
        return 0;
    }
    job.rms = rms;
    vtm_parallel_rows(nblocks, (double)n, rmsnorm_scale_worker, &job);
    return 0;
}

static int silu_scalar(float *y, const float *x, int n) {
    int i;
    if (!y || !x || n < 0) return -1;
    for (i = 0; i < n; i++) {
        double v = (double)x[i];
        if (v < -700.0) y[i] = 0.0f; // как в Python: без переполнения exp
        else y[i] = (float)(v / (1.0 + exp(-v)));
    }
    return 0;
}

static int mul_scalar(float *y, const float *a, const float *b, int n) {
    int i;
    if (!y || !a || !b || n < 0) return -1;
    for (i = 0; i < n; i++) y[i] = a[i] * b[i];
    return 0;
}

static int add_scalar(float *y, const float *a, const float *b, int n) {
    int i;
    if (!y || !a || !b || n < 0) return -1;
    for (i = 0; i < n; i++) y[i] = a[i] + b[i];
    return 0;
}

static int scal_scalar(float *y, const float *x, float k, int n) {
    int i;
    if (!y || !x || n < 0) return -1;
    for (i = 0; i < n; i++) y[i] = x[i] * k;
    return 0;
}

static int copy_scalar(float *y, const float *x, int n) {
    if (!y || !x || n < 0) return -1;
    if (n == 0) return 0;
    memmove(y, x, (size_t)n * sizeof(float));
    return 0;
}

static int emb_scalar(float *y, const float *e, int dim, int token) {
    if (!y || !e || dim <= 0 || token < 0) return -1;
    memmove(y, e + (size_t)token * (size_t)dim, (size_t)dim * sizeof(float));
    return 0;
}

// offset — сколько позиций уже прошло: маска оставляет j <= r + offset.
static int softmax_scalar(float *y, const float *x, int rows, int cols,
                          int causal, int offset) {
    int r, j;
    if (!y || !x || rows < 0 || cols < 0) return -1;
    if (rows == 0 || cols == 0) return 0;

    for (r = 0; r < rows; r++) {
        const float *xr = x + (size_t)r * (size_t)cols;
        float *yr = y + (size_t)r * (size_t)cols;
        long long limit = (long long)r + (long long)offset;
        double max = -INFINITY;
        double sum = 0.0;
        int any = 0;

        for (j = 0; j < cols; j++) {
            double v;
            if (causal && (long long)j > limit) continue;
            v = (double)xr[j];
            if (isnan(v)) continue;
            any = 1;
            if (v > max) max = v;
        }
        if (!any) {
            for (j = 0; j < cols; j++) yr[j] = 0.0f;
            continue;
        }
        for (j = 0; j < cols; j++) {
            double v;
            if (causal && (long long)j > limit) continue;
            v = (double)xr[j];
            if (isnan(v)) continue;
            sum += exp(v - max);
        }
        if (!(sum > 0.0) || !isfinite(sum)) {
            for (j = 0; j < cols; j++) yr[j] = 0.0f;
            continue;
        }
        for (j = 0; j < cols; j++) {
            double v;
            if (causal && (long long)j > limit) {
                yr[j] = 0.0f;
                continue;
            }
            v = (double)xr[j];
            if (isnan(v)) {
                yr[j] = 0.0f;
                continue;
            }
            yr[j] = (float)(exp(v - max) / sum);
        }
    }
    return 0;
}

// --- публичные описатели бэкендов -----------------------------------------

const vtm_backend vtm_backend_scalar_lut = {
    .name = "scalar-lut243",
    .tmv = tmv_lut,
    .tmv_blocks = tmv_lut_blocks,
    .rmsnorm = rmsnorm_scalar,
    .silu = silu_scalar,
    .mul = mul_scalar,
    .add = add_scalar,
    .scal = scal_scalar,
    .softmax = softmax_scalar,
    .emb = emb_scalar,
    .copy = copy_scalar
};

const vtm_backend vtm_backend_scalar_direct = {
    .name = "scalar-direct",
    .tmv = tmv_direct,
    .tmv_blocks = tmv_direct_blocks,
    .rmsnorm = rmsnorm_scalar,
    .silu = silu_scalar,
    .mul = mul_scalar,
    .add = add_scalar,
    .scal = scal_scalar,
    .softmax = softmax_scalar,
    .emb = emb_scalar,
    .copy = copy_scalar
};
