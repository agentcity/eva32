// Тест C-ABI ВТМ (задача 0067): колбэк стриминга, вежливая остановка,
// отмена из колбэка, детерминизм повторного запуска. Плюс кэш префикса в
// C-ABI (задача 0087): ответы с opts->cache_prefix == 1 и без него совпадают
// побитово, session_load инвалидирует кэш, а замер показывает время второго
// запроса с общим префиксом с кэшем и без.
//
// Модель берётся из аргумента или demo/model.vtm (в нём есть SAMP).
// Замер включается переменной VTM_API_CACHE_MODEL — это должен быть
// контейнер с ctx >= 130 (demo/model.vtm имеет ctx=8):
//
//   python3 -c 'from tools import bench_vtm; print(bench_vtm.prepare_container(
//       "demo/model.vtm", 256, "/tmp/eva"))'
//   VTM_API_CACHE_MODEL=/tmp/eva/ctx256/model.vtm ./test_vtm_api
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

static vtm_api_t *open_opts(const char *path, uint32_t max_new,
                            uint32_t cache_prefix) {
    vtm_api_options opts;
    memset(&opts, 0, sizeof opts);
    opts.struct_size = sizeof opts;
    opts.max_new_tokens = max_new;
    opts.cache_prefix = cache_prefix;
    return vtm_api_open(path, &opts);
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
