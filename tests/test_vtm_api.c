// Тест C-ABI ВТМ (задача 0067): колбэк стриминга, вежливая остановка,
// отмена из колбэка, детерминизм повторного запуска. Плюс кэш префикса в
// C-ABI (задача 0087): ответы с opts->cache_prefix == 1 и без него совпадают
// побитово, session_load инвалидирует кэш, а замер показывает время второго
// запроса с общим префиксом с кэшем и без. Плюс батчевый prefill
// (задачи 0112, 0114): opts->prefill_batch = 0/1/2/3/32/128/1e6 и
// --prefill-batch в CLI дают побитово те же ответы, что прежний путь по
// токену, на обоих бэкендах (scalar — настоящий батчевый путь 0112, SIMD идёт
// по токену до 0113); sizeof(vtm_api_options) при этом не меняется.
//
// Модель берётся из аргумента или demo/model.vtm (в нём есть SAMP).
// Замер включается переменной VTM_API_CACHE_MODEL — это должен быть
// контейнер с ctx >= 130 (demo/model.vtm имеет ctx=8):
//
//   python3 -c 'from tools import bench_vtm; print(bench_vtm.prepare_container(
//       "demo/model.vtm", 256, "/tmp/eva"))'
//   VTM_API_CACHE_MODEL=/tmp/eva/ctx256/model.vtm ./test_vtm_api
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "api.h"
#include "vtm.h"  // коды vtm_status (VTM_E_ARG и т.п.)

static int failed = 0;

static void check(int ok, const char *what) {
    printf("  %-46s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) failed++;
}

static int collect_cb(const int32_t *ids, size_t n, void *user) {
    int32_t *out = (int32_t *)user;
    for (size_t i = 0; i < n; i++) out[i] = ids[i];
    return 0;
}

static int stop_cb(const int32_t *ids, size_t n, void *user) {
    (void)ids; (void)n; (void)user;
    return 1; // вежливая остановка
}

static int cancel_cb(const int32_t *ids, size_t n, void *user) {
    (void)ids; (void)n;
    vtm_api_cancel((vtm_api_t *)user); // отмена «извне», флаг атомарный
    return 0;
}

// --- кэш префикса в C-ABI (задача 0087) ------------------------------------

#define BAG_MAX 256

typedef struct {
    int32_t ids[BAG_MAX];
    size_t n;
} token_bag;

static int bag_cb(const int32_t *ids, size_t n, void *user) {
    token_bag *b = (token_bag *)user;
    size_t i;
    for (i = 0; i < n && b->n < BAG_MAX; i++) b->ids[b->n++] = ids[i];
    return 0;
}

static void print_bag(const char *label, const token_bag *b) {
    size_t i;
    printf("  %s:", label);
    for (i = 0; i < b->n; i++) printf(" %d", (int)b->ids[i]);
    printf("  (n=%zu)\n", b->n);
}

static int same_bag(const token_bag *a, const token_bag *b) {
    if (a->n != b->n) return 0;
    return memcmp(a->ids, b->ids, a->n * sizeof *a->ids) == 0;
}

// Открыть контекст с размером пачки prefill (задачи 0112, 0114).
static vtm_api_t *open_opts_batch(const char *path, uint32_t max_new,
                                  uint32_t cache_prefix,
                                  uint32_t prefill_batch) {
    vtm_api_options opts;
    memset(&opts, 0, sizeof opts);
    opts.struct_size = sizeof opts;
    opts.max_new_tokens = max_new;
    opts.cache_prefix = cache_prefix;
    opts.prefill_batch = prefill_batch;
    return vtm_api_open(path, &opts);
}

static vtm_api_t *open_opts(const char *path, uint32_t max_new,
                            uint32_t cache_prefix) {
    return open_opts_batch(path, max_new, cache_prefix, 0);
}

static int gen_bag(vtm_api_t *ctx, const int32_t *prompt, size_t n,
                   token_bag *b) {
    memset(b, 0, sizeof *b);
    return vtm_api_generate(ctx, prompt, n, bag_cb, b);
}

// Совпали ли ответы: код возврата, число токенов и сами токены.
static int same_run(int rc_a, const token_bag *a, int rc_b,
                    const token_bag *b) {
    return rc_a == rc_b && same_bag(a, b);
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

// Ответы с кэшем и без — побитово те же. Модель должна влезать в маленький
// demo-контекст: промпт 3 токена + 3 сгенерированных (ctx=8).
static void test_cache_option(const char *path) {
    const int32_t prompt[3] = {1, 2, 3};
    const int32_t other[3] = {1, 2, 4}; // тот же префикс {1,2}, другой хвост
    token_bag plain_prompt, cache_prompt, plain_other, cache_other;
    token_bag fresh, again;
    vtm_api_t *plain, *cached;
    int rc_plain, rc_cached;
    const int32_t bad_prompt[2] = {1, -1};
    vtm_api_options opts;
    char sess[128];

    printf("  -- кэш префикса в C-ABI (cache_prefix)\n");

    plain = open_opts(path, 3, 0);
    cached = open_opts(path, 3, 1);
    check(plain != NULL && cached != NULL, "открытие с cache_prefix=0 и =1");
    if (!plain || !cached) {
        vtm_api_close(plain);
        vtm_api_close(cached);
        return;
    }

    rc_plain = gen_bag(plain, prompt, 3, &plain_prompt);
    rc_cached = gen_bag(cached, prompt, 3, &cache_prompt);
    check(same_run(rc_plain, &plain_prompt, rc_cached, &cache_prompt),
          "первый запрос: с кэшем == без кэша");
    if (!same_bag(&plain_prompt, &cache_prompt)) {
        print_bag("без кэша", &plain_prompt);
        print_bag("с кэшем", &cache_prompt);
    }

    // Второй запрос с тем же промптом — попадание в кэш префикса.
    rc_plain = gen_bag(plain, prompt, 3, &plain_prompt);
    rc_cached = gen_bag(cached, prompt, 3, &cache_prompt);
    check(same_run(rc_plain, &plain_prompt, rc_cached, &cache_prompt),
          "повторный запрос: с кэшем == без кэша");

    // Тот же префикс {1,2}, другой хвост: попадание обязано остаться
    // корректным, а не отдать хвост прошлого запроса.
    rc_plain = gen_bag(plain, other, 3, &plain_other);
    rc_cached = gen_bag(cached, other, 3, &cache_other);
    check(same_run(rc_plain, &plain_other, rc_cached, &cache_other),
          "другой хвост при том же префиксе совпал");

    // Промпт длиннее прежнего: общий префикс переиспользуется, добавленный
    // токен считается заново (ctx=8: 4 промпта + 3 новых = 7).
    {
        const int32_t longer[4] = {1, 2, 3, 4};
        token_bag pl, ca;
        rc_plain = gen_bag(plain, longer, 4, &pl);
        rc_cached = gen_bag(cached, longer, 4, &ca);
        check(same_run(rc_plain, &pl, rc_cached, &ca),
              "более длинный промпт: с кэшем == без кэша");
    }

    // И снова первый промпт после запроса с другим хвостом: снимок сэмплера
    // восстановлен, ответ прежний.
    rc_plain = gen_bag(plain, prompt, 3, &plain_prompt);
    rc_cached = gen_bag(cached, prompt, 3, &again);
    check(same_run(rc_plain, &plain_prompt, rc_cached, &again)
              && same_bag(&again, &cache_prompt),
          "возврат к прежнему промпту даёт те же токены");

    // Отрицательный id отвергается и с включённой опцией.
    check(vtm_api_generate(cached, bad_prompt, 2, collect_cb, NULL)
              == VTM_E_ARG,
          "отрицательный id -> VTM_E_ARG (кэш включён)");

    // Опция строгая: кроме 0/1 — ошибка открытия.
    memset(&opts, 0, sizeof opts);
    opts.struct_size = sizeof opts;
    opts.cache_prefix = 2;
    check(vtm_api_open(path, &opts) == NULL && *vtm_api_last_error(NULL) != '\0',
          "cache_prefix=2 -> ошибка открытия");

    // Отмена до старта и вежливая остановка колбэком не портят кэш:
    // следующий запрос снова даёт ответ полного прогона.
    vtm_api_cancel(cached);
    check(vtm_api_generate(cached, prompt, 3, collect_cb, NULL)
              == VTM_API_CANCELLED,
          "отмена до старта -> VTM_API_CANCELLED (кэш включён)");
    check(vtm_api_generate(cached, prompt, 3, stop_cb, NULL)
              == VTM_API_STOPPED,
          "остановка колбэком -> VTM_API_STOPPED (кэш включён)");
    rc_plain = gen_bag(plain, prompt, 3, &plain_prompt);
    rc_cached = gen_bag(cached, prompt, 3, &again);
    check(same_run(rc_plain, &plain_prompt, rc_cached, &again),
          "после отмены/остановки ответ прежний");

    // session_load меняет состояние машины — кэш обязан инвалидироваться.
    // Ссылка — тот же промпт на контексте без кэша (полный прогон всегда).
    {
        snprintf(sess, sizeof sess, "/tmp/vtm_api_cache_%ld.vtms",
                 (long)getpid());
        check(vtm_api_session_save(cached, sess) == VTM_OK,
              "session_save после прогрева кэша");
        check(vtm_api_session_load(cached, sess) == VTM_OK,
              "session_load в тот же контекст");
        rc_plain = gen_bag(plain, prompt, 3, &fresh);
        rc_cached = gen_bag(cached, prompt, 3, &again);
        check(same_run(rc_plain, &fresh, rc_cached, &again),
              "после session_load ответ как у полного прогона");
        remove(sess);
    }

    // Опция выключена по умолчанию: свежий контекст с cache_prefix=0 даёт те
    // же токены, что и первый прогон (старое поведение vtm_reset на запрос).
    {
        token_bag a1;
        vtm_api_t *dflt = open_opts(path, 3, 0);
        check(dflt != NULL, "открытие с cache_prefix=0 (дефолт)");
        if (dflt != NULL) {
            int r1 = gen_bag(dflt, prompt, 3, &a1);
            check(r1 == rc_plain && same_bag(&a1, &cache_prompt),
                  "без кэша: свежий контекст -- те же токены");
            vtm_api_close(dflt);
        }
    }

    // NULL opts — дефолты (cache_prefix == 0). generate() не зовём: без
    // max_new_tokens демо-программа не останавливается сама.
    {
        vtm_api_t *dflt = vtm_api_open(path, NULL);
        check(dflt != NULL, "open с NULL opts (дефолты)");
        vtm_api_close(dflt);
    }

    vtm_api_close(plain);
    vtm_api_close(cached);
}

// Замер через C-ABI: два запроса с общим префиксом 128 токенов.
// «Без кэша» — второй запрос считает все 129 токенов; «с кэшем» — только
// хвостовой. Требуется контейнер с ctx >= 130 (иначе KVW упадёт).
static void bench_cache_option(const char *path) {
    const int n_prompt = 129;   // 128 общих + 1 хвост
    const int prefix = 128;
    const char *iters_env = getenv("VTM_API_CACHE_ITERS");
    const char *gen_env = getenv("VTM_API_CACHE_MAX_NEW");
    int iters = iters_env ? atoi(iters_env) : 5;
    int max_new = gen_env ? atoi(gen_env) : 1;
    int32_t prompt[129];
    token_bag plain_out, cache_out;
    vtm_api_t *plain, *cached;
    double t0, prime_plain = 0.0, prime_cache = 0.0;
    double plain_ms = 0.0, cache_ms = 0.0;
    int i, it, rc_plain = 0, rc_cached = 0;

    if (iters < 1) iters = 1;
    if (iters > 100) iters = 100;
    if (max_new < 1) max_new = 1;

    printf("  -- замер: префикс %d токенов + 1 хвост, новых токенов %d,"
           " запросов %d\n", prefix, max_new, iters);
    printf("     модель: %s\n", path);

    for (i = 0; i < n_prompt; i++)
        prompt[i] = (int32_t)((i * 7 + 3) % 97); // id < vocab demo-словаря
    prompt[n_prompt - 1] = 5;

    plain = open_opts(path, (uint32_t)max_new, 0);
    cached = open_opts(path, (uint32_t)max_new, 1);
    check(plain != NULL && cached != NULL, "открытие для замера");
    if (!plain || !cached) {
        vtm_api_close(plain);
        vtm_api_close(cached);
        return;
    }

    // Прогрев: первый запрос. У контекста с кэшем он запоминает префикс.
    t0 = now_ms();
    rc_plain = gen_bag(plain, prompt, (size_t)n_prompt, &plain_out);
    prime_plain = now_ms() - t0;
    t0 = now_ms();
    rc_cached = gen_bag(cached, prompt, (size_t)n_prompt, &cache_out);
    prime_cache = now_ms() - t0;

    // Второй запрос: тот же префикс, другой хвостовой токен.
    prompt[n_prompt - 1] = 7;
    for (it = 0; it < iters; it++) {
        t0 = now_ms();
        rc_plain = gen_bag(plain, prompt, (size_t)n_prompt, &plain_out);
        plain_ms += now_ms() - t0;
        t0 = now_ms();
        rc_cached = gen_bag(cached, prompt, (size_t)n_prompt, &cache_out);
        cache_ms += now_ms() - t0;
    }
    plain_ms /= (double)iters;
    cache_ms /= (double)iters;

    printf("     первый запрос (прогрев): без кэша %.3f мс, с кэшем %.3f мс\n",
           prime_plain, prime_cache);
    printf("     второй запрос: без кэша %.3f мс, с кэшем %.3f мс"
           " (%.2fx)\n", plain_ms, cache_ms,
           cache_ms > 0.0 ? plain_ms / cache_ms : 0.0);

    check(same_run(rc_plain, &plain_out, rc_cached, &cache_out),
          "замер: ответы с кэшем и без совпали");
    check(cache_ms < plain_ms, "замер: второй запрос с кэшем быстрее");

    vtm_api_close(plain);
    vtm_api_close(cached);
}

// --- батчевый prefill в C-ABI и CLI (задачи 0112, 0114) --------------------
//
// prefill_batch — только скорость: сколько токенов промпта ядро считает за
// один проход. Ответы обязаны быть побитово те же, что при 0 (по токену),
// поэтому проверка простая: одинаковые коды возврата и токены при разных N.
// Сейчас (без ядра 0112) api.c/main.c вызывают слабую заглушку «по токену»;
// когда 0112 добавит сильное vtm_run_batch() в interp.c, эти же проверки
// начнут проверять настоящее ядро.

// Раскладка vtm_api_options зафиксирована ABI-версией 1 (api.h) и меняться не
// должна: prefill_batch занял элемент прежнего reserved, sizeof остался 56.
// Если эти typedef'ы перестали компилироваться — сломана ABI, а не тест.
#define VTM_API_OPTIONS_SIZE_V1 56u

typedef char vtm_api_options_size_v1
    [(sizeof(vtm_api_options) == VTM_API_OPTIONS_SIZE_V1) ? 1 : -1];
typedef char vtm_api_options_cache_prefix_off_v1
    [(offsetof(vtm_api_options, cache_prefix) == 32) ? 1 : -1];
typedef char vtm_api_options_prefill_batch_off_v1
    [(offsetof(vtm_api_options, prefill_batch) == 36) ? 1 : -1];
typedef char vtm_api_options_reserved_off_v1
    [(offsetof(vtm_api_options, reserved) == 40) ? 1 : -1];

// Ответ при prefill_batch == N: тот же, что при 0, и повторяемый.
static void test_prefill_batch(const char *path) {
    // demo/model.vtm: ctx=8, поэтому 4 промпта + 2 новых влезают.
    const int32_t prompt[4] = {1, 2, 3, 4};
    const int32_t one[1] = {3};
    const int32_t bad[3] = {1, -1, 2};
    const uint32_t batches[] = {1u, 2u, 3u, 32u, 128u, 1000000u};
    token_bag ref, got, again, one_ref, one_got;
    vtm_api_t *ctx;
    int rc_ref, rc_one_ref;
    size_t i;
    char what[160];

    printf("  -- батчевый prefill в C-ABI (prefill_batch)\n");

    memset(&ref, 0, sizeof ref);
    memset(&one_ref, 0, sizeof one_ref);
    check(sizeof(vtm_api_options) == VTM_API_OPTIONS_SIZE_V1,
          "sizeof(vtm_api_options) == 56 (ABI v1)");

    ctx = open_opts_batch(path, 2, 0, 0);
    check(ctx != NULL, "prefill_batch=0: открытие");
    if (!ctx) return;
    rc_ref = gen_bag(ctx, prompt, 4, &ref);
    rc_one_ref = gen_bag(ctx, one, 1, &one_ref);
    vtm_api_close(ctx);

    for (i = 0; i < sizeof batches / sizeof batches[0]; i++) {
        const unsigned b = (unsigned)batches[i];
        int rc;

        ctx = open_opts_batch(path, 2, 0, batches[i]);
        snprintf(what, sizeof what, "prefill_batch=%u: открытие", b);
        check(ctx != NULL, what);
        if (!ctx) continue;

        rc = gen_bag(ctx, prompt, 4, &got);
        snprintf(what, sizeof what, "prefill_batch=%u: те же токены", b);
        check(same_run(rc_ref, &ref, rc, &got), what);
        if (!same_bag(&ref, &got)) {
            print_bag("batch=0", &ref);
            print_bag("batch=N", &got);
        }

        // Повтор на том же контексте: generate() сам сбрасывает машину.
        rc = gen_bag(ctx, prompt, 4, &again);
        snprintf(what, sizeof what, "prefill_batch=%u: повтор теми же", b);
        check(same_bag(&again, &ref), what);

        // Промпт из одного токена: пачки нет, последний токен — как раньше.
        rc = gen_bag(ctx, one, 1, &one_got);
        snprintf(what, sizeof what, "prefill_batch=%u: промпт 1 токен", b);
        check(same_run(rc_one_ref, &one_ref, rc, &one_got), what);

        // Отрицательный id отвергается до прогона, а не на середине пачки.
        snprintf(what, sizeof what, "prefill_batch=%u: id<0 -> VTM_E_ARG", b);
        check(vtm_api_generate(ctx, bad, 3, collect_cb, NULL) == VTM_E_ARG,
              what);

        vtm_api_close(ctx);
    }

    // Значение больше int не проходит open(): длина пачки уходит в ядро int'ом.
    {
        vtm_api_options opts;
        memset(&opts, 0, sizeof opts);
        opts.struct_size = sizeof opts;
        opts.prefill_batch = 0x80000000u;
        check(vtm_api_open(path, &opts) == NULL
                  && *vtm_api_last_error(NULL) != '\0',
              "prefill_batch > INT_MAX -> ошибка open");
    }

    // cache_prefix=1 (0087) приоритетнее батча: путь прогона префикса свой,
    // ответ всё равно как у полного прогона.
    {
        vtm_api_t *cached = open_opts_batch(path, 2, 1, 32);
        check(cached != NULL, "cache_prefix=1 + prefill_batch=32: открытие");
        if (cached != NULL) {
            token_bag c1, c2;
            int rc1 = gen_bag(cached, prompt, 4, &c1);
            int rc2 = gen_bag(cached, prompt, 4, &c2);
            check(same_run(rc_ref, &ref, rc1, &c1)
                      && same_run(rc_ref, &ref, rc2, &c2),
                  "cache_prefix+prefill_batch: те же токены");
            vtm_api_close(cached);
        }
    }

    // Сессия KV и батч: после session_load() состояние машины заменено, но
    // следующий generate() с батчем всё равно даёт ответ полного прогона.
    {
        char sess[128];
        vtm_api_t *s = open_opts_batch(path, 2, 0, 32);
        check(s != NULL, "prefill_batch=32: открытие под сессию");
        if (s != NULL) {
            token_bag s1;
            int rc1 = gen_bag(s, prompt, 4, &s1);
            snprintf(sess, sizeof sess, "/tmp/vtm_api_batch_%ld.vtms",
                     (long)getpid());
            check(same_run(rc_ref, &ref, rc1, &s1)
                      && vtm_api_session_save(s, sess) == VTM_OK
                      && vtm_api_session_load(s, sess) == VTM_OK,
                  "prefill_batch: session_save/load");
            rc1 = gen_bag(s, prompt, 4, &s1);
            check(same_run(rc_ref, &ref, rc1, &s1),
                  "prefill_batch: после session_load те же токены");
            remove(sess);
            vtm_api_close(s);
        }
    }

    // Отмена до старта не зависит от размера пачки.
    {
        vtm_api_t *c = open_opts_batch(path, 2, 0, 128);
        check(c != NULL, "prefill_batch=128: открытие под отмену");
        if (c != NULL) {
            vtm_api_cancel(c);
            check(vtm_api_generate(c, prompt, 4, collect_cb, NULL)
                      == VTM_API_CANCELLED,
                  "prefill_batch=128: отмена до старта");
            vtm_api_close(c);
        }
    }

    // Скалярный бэкенд: по контракту 0112 батчевые ядра есть только там, на
    // SIMD (0113 ещё не сдана) vtm_run_batch идёт по токену. Значит, именно
    // scalar проверяет настоящий батчевый путь: ответ обязан быть побитово
    // тем же, что у batch=0 на том же (скалярном) бэкенде.
    {
        vtm_api_options opts;
        vtm_api_t *s0 = NULL, *s32 = NULL;
        token_bag a, b;
        int rc0, rc32;

        memset(&opts, 0, sizeof opts);
        opts.struct_size = sizeof opts;
        opts.max_new_tokens = 2;
        opts.backend_name = "scalar";
        s0 = vtm_api_open(path, &opts);
        opts.prefill_batch = 32;
        s32 = vtm_api_open(path, &opts);
        check(s0 != NULL && s32 != NULL, "scalar: batch=0 и batch=32 открылись");
        if (s0 != NULL && s32 != NULL) {
            rc0 = gen_bag(s0, prompt, 4, &a);
            rc32 = gen_bag(s32, prompt, 4, &b);
            check(same_run(rc0, &a, rc32, &b),
                  "scalar: batch=32 == batch=0 побитово");
            if (!same_bag(&a, &b)) {
                print_bag("scalar batch=0", &a);
                print_bag("scalar batch=32", &b);
            }
        }
        vtm_api_close(s0);
        vtm_api_close(s32);
    }

    // NULL opts и struct_size == 0 — «все дефолты» (0180): prefill_batch == 128.
    // Наружу значение не отдаётся, поэтому проверяем открытие; ответы при 128
    // и 0 на demo-модели совпадают — это уже проверено выше для N = 1..1e6.
    {
        vtm_api_options zero;
        vtm_api_t *dflt = vtm_api_open(path, NULL);
        vtm_api_t *zeroed;
        check(dflt != NULL, "open с NULL opts (prefill_batch по умолчанию)");
        vtm_api_close(dflt);
        memset(&zero, 0, sizeof zero);   // struct_size == 0 — все дефолты
        zeroed = vtm_api_open(path, &zero);
        check(zeroed != NULL, "open с struct_size=0 (все дефолты, 0180)");
        vtm_api_close(zeroed);
    }
}

// Строка «вывод: ...» из stdout `vtm run`; 0 — нашли, -1 — нет/ошибка.
static int cli_output_line(const char *cmd, char *out, size_t cap) {
    FILE *p;
    char line[512];
    int found = -1;

    out[0] = '\0';
    p = popen(cmd, "r");
    if (p == NULL) return -1;
    while (fgets(line, sizeof line, p) != NULL) {
        if (strncmp(line, "вывод:", 6) == 0) {
            size_t n = strlen(line);
            if (n >= cap) n = cap - 1;
            memcpy(out, line, n);
            out[n] = '\0';
            found = 0;
            break;
        }
    }
    if (pclose(p) != 0) return -1;
    return found;
}

// CLI: --prefill-batch N не меняет ответа (те же токены, что при 0).
// ./vtm и demo-модель собирает `make test`; без них проверка пропускается,
// чтобы тест можно было гонять и без CLI в рабочем каталоге.
static void test_prefill_batch_cli(const char *path) {
    const int batches[] = {1, 2, 32, 128};
    char cmd[1024], base[256], got[256];
    size_t i;

    printf("  -- CLI: --prefill-batch не меняет ответ\n");
    if (access("./vtm", X_OK) != 0 || access(path, R_OK) != 0) {
        printf("  %-46s skip (нет ./vtm или модели)\n", "--prefill-batch CLI");
        return;
    }
    snprintf(cmd, sizeof cmd,
             "./vtm run '%s' --tokens 1,2,3,4,5 --max-new 3 2>/dev/null", path);
    if (cli_output_line(cmd, base, sizeof base) != 0) {
        printf("  %-46s skip (не разобрал вывод vtm run)\n",
               "--prefill-batch CLI");
        return;
    }
    for (i = 0; i < sizeof batches / sizeof batches[0]; i++) {
        char what[96];
        snprintf(cmd, sizeof cmd,
                 "./vtm run '%s' --tokens 1,2,3,4,5 --max-new 3 "
                 "--prefill-batch %d 2>/dev/null", path, batches[i]);
        snprintf(what, sizeof what, "--prefill-batch %d: ответ как без него",
                 batches[i]);
        if (cli_output_line(cmd, got, sizeof got) != 0) {
            check(0, what);
            continue;
        }
        check(strcmp(base, got) == 0, what);
        if (strcmp(base, got) != 0) {
            printf("     без батча: %s", base);
            printf("     с батчем:  %s", got);
        }
    }

    // Скалярный бэкенд — настоящий батчевый путь ядра 0112, и он же самый
    // чувствительный: ошибка в позициях/KV проявится в токенах.
    {
        char sc_base[256], sc_got[256];
        snprintf(cmd, sizeof cmd,
                 "./vtm run '%s' --tokens 1,2,3,4,5 --max-new 3 "
                 "--backend scalar 2>/dev/null", path);
        if (cli_output_line(cmd, sc_base, sizeof sc_base) == 0) {
            snprintf(cmd, sizeof cmd,
                     "./vtm run '%s' --tokens 1,2,3,4,5 --max-new 3 "
                     "--backend scalar --prefill-batch 32 2>/dev/null", path);
            if (cli_output_line(cmd, sc_got, sizeof sc_got) != 0) {
                check(0, "scalar CLI --prefill-batch 32: ответ");
            } else {
                check(strcmp(sc_base, sc_got) == 0,
                      "scalar CLI --prefill-batch 32: ответ тот же");
                if (strcmp(sc_base, sc_got) != 0) {
                    printf("     без батча: %s", sc_base);
                    printf("     с батчем:  %s", sc_got);
                }
            }
        }
    }
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "../../demo/model.vtm";
    vtm_api_options opts;
    vtm_api_t *ctx;
    int32_t prompt[3] = {1, 2, 3};
    int32_t out[64];
    int rc;

    memset(&opts, 0, sizeof opts);
    opts.struct_size = sizeof opts;
    // demo/model.vtm имеет ctx=8: промпт из 3 токенов оставляет 5 позиций.
    opts.max_new_tokens = 3;

    // Строки теста полезны сразу: если проверка зависнет, видно, на какой.
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("vtm_api: %s\n", path);

    ctx = vtm_api_open(path, &opts);
    check(ctx != NULL, "открытие контейнера");
    if (!ctx) {
        fprintf(stderr, "ошибка: %s\n", vtm_api_last_error(NULL));
        return 1;
    }

    memset(out, 0, sizeof out);
    rc = vtm_api_generate(ctx, prompt, 3, collect_cb, out);
    check(rc == VTM_API_OK || rc == VTM_API_EOS, "generate завершился штатно");
    check(out[0] != 0 || out[1] != 0 || out[2] != 0, "колбэк получил токены");

    rc = vtm_api_generate(ctx, prompt, 3, stop_cb, NULL);
    check(rc == VTM_API_STOPPED, "колбэк вернул 1 -> VTM_API_STOPPED");

    rc = vtm_api_generate(ctx, prompt, 3, cancel_cb, ctx);
    check(rc == VTM_API_CANCELLED, "отмена -> VTM_API_CANCELLED");

    // После отмены контекст должен быть пригоден: флаг сбрасывается на выходе.
    memset(out, 0, sizeof out);
    rc = vtm_api_generate(ctx, prompt, 3, collect_cb, out);
    check(rc == VTM_API_OK || rc == VTM_API_EOS, "после отмены снова работает");

    // Детерминизм: тот же промпт — те же токены (reset внутри generate).
    {
        int32_t a[64], b[64];
        memset(a, 0, sizeof a); memset(b, 0, sizeof b);
        vtm_api_generate(ctx, prompt, 3, collect_cb, a);
        vtm_api_generate(ctx, prompt, 3, collect_cb, b);
        check(memcmp(a, b, sizeof a) == 0, "повторный запуск даёт те же токены");
    }

    rc = vtm_api_generate(ctx, NULL, 0, collect_cb, out);
    check(rc == VTM_E_ARG, "пустой промпт -> VTM_E_ARG");

    vtm_api_close(ctx);

    // Кэш префикса в C-ABI (задача 0087).
    test_cache_option(path);
    // Батчевый prefill (задачи 0112, 0114): C-ABI и CLI.
    test_prefill_batch(path);
    test_prefill_batch_cli(path);
    {
        const char *bench_path = getenv("VTM_API_CACHE_MODEL");
        if (bench_path != NULL && *bench_path != '\0') {
            bench_cache_option(bench_path);
        }
    }

    printf("%s: %s\n", failed ? "ПРОВАЛЫ" : "все проверки прошли",
           failed ? "есть ошибки" : "");
    return failed ? 1 : 0;
}
