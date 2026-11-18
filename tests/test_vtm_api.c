// Тест C-ABI ВТМ (задача 0067): колбэк стриминга, вежливая остановка,
// отмена из колбэка, детерминизм повторного запуска.
//
// Модель берётся из аргумента или demo/model.vtm (в нём есть SAMP).
#include <stdio.h>
#include <string.h>

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
    printf("%s: %s\n", failed ? "ПРОВАЛЫ" : "все проверки прошли",
           failed ? "есть ошибки" : "");
    return failed ? 1 : 0;
}
