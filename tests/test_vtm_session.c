// Тест KV-сессий (задача 0079): снимок после 3 токенов, загрузка в новую
// машину, продолжение — совпадает с непрерывным прогоном; оборванный и
// повреждённый файл дают ошибку без падения и не портят машину.
//
// Сборка (из корня репозитория):
//   cc -std=c99 -O2 -I src/vtm -o /tmp/t tests/test_vtm_session.c \
//      src/vtm/libvtm.a -lpthread
// Модель по умолчанию — demo/model.vtm (аргументом можно задать другую).
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "api.h"
#include "vtm.h"

#define N_PROMPT 3
#define N_GEN    3
#define N_CONT   (N_GEN * 2)
#define MAX_IDS  64

static int failed = 0;

static void check(int ok, const char *what) {
    printf("  %-56s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) failed++;
}

typedef struct {
    int32_t ids[MAX_IDS];
    size_t n;
} token_bag;

static int bag_cb(const int32_t *ids, size_t n, void *user) {
    token_bag *b = (token_bag *)user;
    size_t i;
    for (i = 0; i < n && b->n < MAX_IDS; i++) b->ids[b->n++] = ids[i];
    return 0;
}

static void print_ids(const char *label, const token_bag *b) {
    size_t i;
    printf("  %s:", label);
    for (i = 0; i < b->n; i++) printf(" %d", (int)b->ids[i]);
    printf("  (n=%zu)\n", b->n);
}

static vtm_api_t *open_ctx(const char *model, uint32_t max_new) {
    vtm_api_options opts;
    memset(&opts, 0, sizeof opts);
    opts.struct_size = sizeof opts;
    opts.max_new_tokens = max_new;
    return vtm_api_open(model, &opts);
}

static int read_file(const char *path, unsigned char **out, size_t *size) {
    FILE *f = fopen(path, "rb");
    long len;
    unsigned char *buf;
    if (f == NULL) return -1;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    len = ftell(f);
    if (len < 0) { fclose(f); return -1; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return -1; }
    buf = (unsigned char *)malloc((size_t)len ? (size_t)len : 1u);
    if (buf == NULL) { fclose(f); return -1; }
    if (len > 0 && fread(buf, 1, (size_t)len, f) != (size_t)len) {
        free(buf);
        fclose(f);
        return -1;
    }
    fclose(f);
    *out = buf;
    *size = (size_t)len;
    return 0;
}

static int write_file(const char *path, const unsigned char *buf, size_t n) {
    FILE *f = fopen(path, "wb");
    if (f == NULL) return -1;
    if (n != 0 && fwrite(buf, 1, n, f) != n) { fclose(f); return -1; }
    return fclose(f) == 0 ? 0 : -1;
}

static uint32_t le32(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8)
         | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint64_t le64(const unsigned char *p) {
    uint64_t v = 0;
    unsigned i;
    for (i = 0; i < 8; i++) v |= (uint64_t)p[i] << (8 * i);
    return v;
}

int main(int argc, char **argv) {
    static const char *candidates[] = {
        "demo/model.vtm", "../../demo/model.vtm", NULL
    };
    const char *model = argc > 1 ? argv[1] : NULL;
    const char *picked = NULL;
    char sess[256], trunc_path[256], bad_path[256], empty_path[256];
    int32_t prompt[N_PROMPT] = {1, 2, 3};
    token_bag cont, first, second;
    vtm_api_t *ctx;
    int rc;
    size_t i;

    for (i = 0; model == NULL && candidates[i] != NULL; i++) {
        if (access(candidates[i], R_OK) == 0) picked = candidates[i];
    }
    if (model == NULL) model = picked != NULL ? picked : candidates[0];

    snprintf(sess, sizeof sess, "/tmp/vtm_session_%ld.vtms", (long)getpid());
    snprintf(trunc_path, sizeof trunc_path, "/tmp/vtm_session_%ld_trunc.vtms",
             (long)getpid());
    snprintf(bad_path, sizeof bad_path, "/tmp/vtm_session_%ld_bad.vtms",
             (long)getpid());
    snprintf(empty_path, sizeof empty_path, "/tmp/vtm_session_%ld_empty.vtms",
             (long)getpid());

    printf("test_vtm_session: %s\n", model);

    // 1. Непрерывный прогон на 6 токенов — эталон.
    memset(&cont, 0, sizeof cont);
    ctx = open_ctx(model, N_CONT);
    check(ctx != NULL, "открытие контейнера (эталон)");
    if (ctx == NULL) {
        fprintf(stderr, "ошибка: %s\n", vtm_api_last_error(NULL));
        return 1;
    }
    rc = vtm_api_generate(ctx, prompt, N_PROMPT, bag_cb, &cont);
    check(rc == VTM_API_OK || rc == VTM_API_EOS, "непрерывный прогон завершился");
    check(cont.n == N_CONT, "эталон: 6 токенов");
    print_ids("непрерывно", &cont);

    // 2. Первые 3 токена, снимок сессии.
    memset(&first, 0, sizeof first);
    ctx = open_ctx(model, N_GEN);
    check(ctx != NULL, "открытие контейнера (снимок)");
    if (ctx == NULL) return 1;
    rc = vtm_api_generate(ctx, prompt, N_PROMPT, bag_cb, &first);
    check(rc == VTM_API_OK, "первый прогон: 3 токена");
    check(first.n == N_GEN, "первый прогон отдал 3 токена");
    print_ids("первые", &first);
    rc = vtm_api_session_save(ctx, sess);
    check(rc == VTM_OK, "session_save -> VTM_OK");
    if (rc != VTM_OK) {
        fprintf(stderr, "  save: %s\n", vtm_api_last_error(ctx));
    }
    {
        char temp_name[300];
        snprintf(temp_name, sizeof temp_name, "%s.tmp.%ld", sess,
                 (long)getpid());
        check(access(temp_name, F_OK) != 0,
              "после успешного save temp-файла не осталось");
    }
    vtm_api_close(ctx);

    // Проверить сам формат: VTMS v1, trailer VTME, размер == 64 + KV + 36.
    {
        unsigned char *buf = NULL;
        size_t size = 0;
        struct stat st;
        int fmt_ok = 0;
        if (stat(sess, &st) == 0 && read_file(sess, &buf, &size) == 0
            && size > 64 + 36) {
            const uint32_t ctx_len = le32(buf + 16);
            const uint32_t kv_dim = le32(buf + 20);
            const uint32_t layers = le32(buf + 24);
            const uint64_t kv_bytes = le64(buf + 40);
            fmt_ok = memcmp(buf, "VTMS", 4) == 0
                  && le32(buf + 4) == 1
                  && ctx_len == 8 && kv_dim == 32 && layers == 1
                  && kv_bytes == (uint64_t)ctx_len * kv_dim * 4 * 2 * layers
                  && (uint64_t)size
                     == (uint64_t)64 + kv_bytes + 36
                  && memcmp(buf + size - 36, "VTME", 4) == 0;
            printf("  формат: размер=%zu, ctx=%u, kv_dim=%u, слоёв=%u\n",
                   size, ctx_len, kv_dim, layers);
        }
        free(buf);
        check(fmt_ok, "заголовок VTMS v1, trailer VTME, размер сходится");
    }

    // 3. Загрузка в НОВУЮ машину и продолжение.
    memset(&second, 0, sizeof second);
    ctx = open_ctx(model, N_GEN);
    check(ctx != NULL, "открытие контейнера (загрузка)");
    if (ctx == NULL) return 1;
    rc = vtm_api_session_load(ctx, sess);
    check(rc == VTM_OK, "session_load -> VTM_OK");
    if (rc != VTM_OK) fprintf(stderr, "  load: %s\n", vtm_api_last_error(ctx));
    rc = vtm_api_session_continue(ctx, bag_cb, &second);
    check(rc == VTM_API_OK, "session_continue -> VTM_OK");
    check(second.n == N_GEN, "продолжение отдало 3 токена");
    print_ids("после загрузки", &second);

    {
        int head_ok = first.n == N_GEN;
        int tail_ok = second.n == N_GEN;
        for (i = 0; i < N_GEN && head_ok; i++)
            if (first.ids[i] != cont.ids[i]) head_ok = 0;
        for (i = 0; i < N_GEN && tail_ok; i++)
            if (second.ids[i] != cont.ids[N_GEN + i]) tail_ok = 0;
        check(head_ok, "первые 3 токена == эталону");
        check(tail_ok, "продолжение == эталонным токенам 4..6");
    }

    // 4. Обрыв файла: нет trailer -> ошибка, машина не испорчена.
    {
        unsigned char *buf = NULL;
        size_t size = 0;
        check(read_file(sess, &buf, &size) == 0, "чтение снимка для обрыва");
        if (buf != NULL) {
            check(write_file(trunc_path, buf, size > 10 ? size - 10 : 0) == 0,
                  "обрезанный файл записан");
            free(buf);
        }
        memset(&first, 0, sizeof first);
        rc = vtm_api_session_load(ctx, trunc_path);
        check(rc == VTM_E_TRUNCATED, "оборванный файл -> VTM_E_TRUNCATED");
        check(strlen(vtm_api_last_error(ctx)) > 0, "у ошибки есть текст");
        printf("  ошибка: %s\n", vtm_api_last_error(ctx));
        // Машина осталась рабочей и не получила половину снимка.
        memset(&first, 0, sizeof first);
        rc = vtm_api_generate(ctx, prompt, N_PROMPT, bag_cb, &first);
        check(rc == VTM_API_OK && first.n == N_GEN,
              "после обрыва машина работает");
        check(first.ids[0] == cont.ids[0] && first.ids[1] == cont.ids[1]
                  && first.ids[2] == cont.ids[2],
              "после обрыва токены те же, что у чистого прогона");
    }

    // 5. Повреждённый байт в KV -> sha256 не сходится.
    {
        unsigned char *buf = NULL;
        size_t size = 0;
        check(read_file(sess, &buf, &size) == 0, "чтение снимка для порчи");
        if (buf != NULL && size > 100) {
            buf[64 + 10] ^= 0xFF;
            check(write_file(bad_path, buf, size) == 0, "порченый файл записан");
            free(buf);
        }
        rc = vtm_api_session_load(ctx, bad_path);
        check(rc == VTM_E_INTEGRITY, "порча KV -> VTM_E_INTEGRITY");
        printf("  ошибка: %s\n", vtm_api_last_error(ctx));
    }

    // 6. Пустой/слишком короткий файл и отсутствующий путь.
    {
        check(write_file(empty_path, (const unsigned char *)"VTMS", 4) == 0,
              "короткий файл записан");
        rc = vtm_api_session_load(ctx, empty_path);
        check(rc == VTM_E_TRUNCATED, "файл из 4 байт -> VTM_E_TRUNCATED");
        rc = vtm_api_session_load(ctx, "/tmp/vtm_session_no_such_file.vtms");
        check(rc == VTM_E_IO, "нет файла -> VTM_E_IO");
    }

    // 7. Ошибка save (каталога нет) не создаёт мусора и не портит снимок.
    rc = vtm_api_session_save(ctx, "/no/such/dir/session.vtms");
    check(rc < 0, "save в несуществующий каталог -> ошибка");
    check(strlen(vtm_api_last_error(ctx)) > 0, "у ошибки save есть текст");
    {
        char temp_name[300];
        snprintf(temp_name, sizeof temp_name,
                 "/no/such/dir/session.vtms.tmp.%ld", (long)getpid());
        check(access(temp_name, F_OK) != 0,
              "после неудачного save мусора нет");
    }
    rc = vtm_api_session_load(ctx, sess);
    check(rc == VTM_OK, "прежний снимок после неудачного save цел");

    // 8. Сессия без «висящего» токена (снимок до генерации): продолжать нечем.
    {
        vtm_api_t *fresh = open_ctx(model, N_GEN);
        rc = vtm_api_session_save(fresh, bad_path);
        check(rc == VTM_OK, "снимок до генерации сохранён");
        vtm_api_close(fresh);
        rc = vtm_api_session_load(ctx, bad_path);
        check(rc == VTM_OK, "снимок до генерации загружен");
        rc = vtm_api_session_continue(ctx, bag_cb, &second);
        check(rc == VTM_E_ARG, "continue без pending -> VTM_E_ARG");
    }

    vtm_api_close(ctx);

    unlink(sess);
    unlink(trunc_path);
    unlink(bad_path);
    unlink(empty_path);

    printf("%s: %s\n", failed ? "ПРОВАЛЫ" : "все проверки прошли",
           failed ? "есть ошибки" : "");
    return failed ? 1 : 0;
}
