// Тест KV-сессий (задачи 0079, 0083): снимок после 3 токенов, загрузка в
// новую машину, продолжение — совпадает с непрерывным прогоном; оборванный и
// повреждённый файл дают ошибку без падения и не портят машину. Отдельно —
// совместимость хеша (trailer == стандартный SHA-256) и лимит каталога с
// вытеснением старых сессий по mtime.
//
// Сборка (из корня репозитория):
//   cc -std=c99 -O2 -I src/vtm -o /tmp/t tests/test_vtm_session.c src/vtm/libvtm.a -lpthread
// Модель по умолчанию — demo/model.vtm (аргументом можно задать другую).
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include "api.h"
#include "sha256.h"
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

// Суммарный размер обычных файлов каталога и их число. Temp-файлы записи
// ("<сессия>.tmp.<pid>") не считаем: их не должно оставаться после save.
static int dir_usage(const char *dir, int *count, unsigned long long *bytes) {
    DIR *d = opendir(dir);
    struct dirent *de;
    int n = 0;
    unsigned long long total = 0;

    if (d == NULL) return -1;
    while ((de = readdir(d)) != NULL) {
        char path[1024];
        struct stat st;

        if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0)
            continue;
        if (strstr(de->d_name, ".tmp.") != NULL) continue;
        snprintf(path, sizeof path, "%s/%s", dir, de->d_name);
        if (stat(path, &st) != 0 || !S_ISREG(st.st_mode)) continue;
        n++;
        total += (unsigned long long)st.st_size;
    }
    (void)closedir(d);
    *count = n;
    *bytes = total;
    return 0;
}

// Явный mtime: лимит вытесняет по времени изменения, а пять save подряд
// укладываются в одну секунду.
static void set_mtime(const char *path, long sec) {
    struct timeval tv[2];

    tv[0].tv_sec = sec;
    tv[0].tv_usec = 0;
    tv[1].tv_sec = sec;
    tv[1].tv_usec = 0;
    (void)utimes(path, tv);
}

static void clear_dir(const char *dir) {
    DIR *d = opendir(dir);
    struct dirent *de;

    if (d == NULL) return;
    while ((de = readdir(d)) != NULL) {
        char path[1024];

        if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0)
            continue;
        snprintf(path, sizeof path, "%s/%s", dir, de->d_name);
        (void)unlink(path);
    }
    (void)closedir(d);
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
        int hash_ok = 0;
        if (stat(sess, &st) == 0 && read_file(sess, &buf, &size) == 0
            && size > 64 + 36) {
            const uint32_t ctx_len = le32(buf + 16);
            const uint32_t kv_dim = le32(buf + 20);
            const uint32_t layers = le32(buf + 24);
            const uint64_t kv_bytes = le64(buf + 40);
            vtm_sha256_ctx h;
            uint8_t digest[32];
            fmt_ok = memcmp(buf, "VTMS", 4) == 0
                  && le32(buf + 4) == 1
                  && ctx_len == 8 && kv_dim == 32 && layers == 1
                  && kv_bytes == (uint64_t)ctx_len * kv_dim * 4 * 2 * layers
                  && (uint64_t)size
                     == (uint64_t)64 + kv_bytes + 36
                  && memcmp(buf + size - 36, "VTME", 4) == 0;
            // Хеш сессии — обычный SHA-256 от заголовка и кадров. Так же
            // считала локальная копия в kv_store.c до 0083, поэтому снимки
            // 0079 читаются без миграции (см. также tests/vtm/test_sha256.c
            // с векторами NIST).
            vtm_sha256_init(&h);
            vtm_sha256_update(&h, buf, size - 36);
            vtm_sha256_final(&h, digest);
            hash_ok = memcmp(buf + size - 36 + 4, digest, sizeof digest) == 0;
            printf("  формат: размер=%zu, ctx=%u, kv_dim=%u, слоёв=%u\n",
                   size, ctx_len, kv_dim, layers);
        }
        free(buf);
        check(fmt_ok, "заголовок VTMS v1, trailer VTME, размер сходится");
        check(hash_ok, "trailer == стандартный sha256(заголовок+кадры)");
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

    // 9. Лимит каталога сессий (0083): 5 снимков по ~2 КБ при лимите 5 КБ ->
    //    остаются 2–3 самые новые, старые вытесняются по mtime; запись новой
    //    не падает, а только что записанный файл не удаляется даже при лимите
    //    меньше его размера.
    {
        char dir[1024];
        char paths[5][1024];
        char probe_path[1024];
        const uint64_t limit = 5u * 1024u;
        unsigned long long bytes = 0;
        long mtime_base = (long)time(NULL) - 3600;
        int made_dir = 0;
        int count = 0;
        int exp_count = 1;
        int exists[5];
        int saves_ok = 0;
        vtm_api_t *lru = NULL;

        // mkdtemp в строгом POSIX C99 на macOS не объявлен, поэтому каталог
        // создаём сами: имя уникально по pid + счётчику.
        for (i = 0; i < 100 && !made_dir; i++) {
            snprintf(dir, sizeof dir, "/tmp/vtm_session_lru_%ld_%zu",
                     (long)getpid(), i);
            if (mkdir(dir, 0700) == 0) made_dir = 1;
            else if (errno != EEXIST) break;
        }
        check(made_dir, "временный каталог под лимит создан");
        if (made_dir) {
            lru = open_ctx(model, N_GEN);
            check(lru != NULL, "контекст для проверки лимита открыт");
        }

        if (lru != NULL) {
            check(vtm_api_sessions_dir(lru, dir) == VTM_OK,
                  "sessions_dir -> VTM_OK");
            check(vtm_api_sessions_limit(lru, limit) == VTM_OK,
                  "sessions_limit 5 КБ -> VTM_OK");

            // Пять снимков; mtime задаём явно, иначе все save попадают в одну
            // секунду и «самый старый» не определить.
            saves_ok = 1;
            for (i = 0; i < 5; i++) {
                snprintf(paths[i], sizeof paths[i], "%s/s%zu.vtms", dir, i);
                if (vtm_api_session_save(lru, paths[i]) != VTM_OK) {
                    saves_ok = 0;
                    break;
                }
                set_mtime(paths[i], mtime_base + (long)i);
            }
            if (i < 5)
                fprintf(stderr, "  save[%zu]: %s\n", i,
                        vtm_api_last_error(lru));
            check(saves_ok, "5 сохранений при лимите 5 КБ -> VTM_OK");
        }

        if (saves_ok) {
            struct stat st;

            // paths[0] уже вытеснен; размер берём у самой новой сессии.
            if (stat(paths[4], &st) == 0 && st.st_size > 0) {
                printf("  размер снимка: %lld байт, лимит: %llu байт\n",
                       (long long)st.st_size, (unsigned long long)limit);
                exp_count = (int)(limit / (uint64_t)st.st_size);
            }
            if (exp_count < 1) exp_count = 1;
            if (exp_count > 5) exp_count = 5;

            count = 0;
            bytes = 0;
            check(dir_usage(dir, &count, &bytes) == 0,
                  "каталог сессий прочитан");
            printf("  осталось сессий: %d, суммарно %llu байт "
                   "(ожидалось %d..%d)\n", count, bytes, exp_count,
                   exp_count + 1);
            check(count >= exp_count && count <= exp_count + 1,
                  "осталось 2–3 самые новые сессии");

            for (i = 0; i < 5; i++)
                exists[i] = access(paths[i], F_OK) == 0;
            check(exists[4], "самая новая сессия на месте");
            {
                int newest_ok = 1;
                for (i = 0; i < 5; i++)
                    if (exists[i] != ((int)i >= 5 - count)) newest_ok = 0;
                check(newest_ok, "уцелели именно самые новые (по mtime)");
            }
            check(count == 1 || bytes <= limit,
                  "суммарный размер уложен в лимит");

            // Только что записанный файл не вытесняется, даже если один
            // больше лимита: иначе save() терял бы свой же результат.
            snprintf(paths[0], sizeof paths[0], "%s/keep.vtms", dir);
            check(vtm_api_sessions_limit(lru, 1) == VTM_OK,
                  "лимит 1 байт принят");
            rc = vtm_api_session_save(lru, paths[0]);
            check(rc == VTM_OK, "save при лимите меньше снимка -> VTM_OK");
            check(access(paths[0], F_OK) == 0,
                  "только что записанный файл не вытеснен");
            count = 0;
            bytes = 0;
            (void)dir_usage(dir, &count, &bytes);
            check(count == 1, "осталась только свежая сессия");

            // Ошибка вытеснения не фатальна: недоступный каталог -> save ОК
            // (предупреждение уходит в stderr).
            check(vtm_api_sessions_dir(lru, "/no/such/vtm_sessions") == VTM_OK,
                  "несуществующий каталог лимита принят");
            snprintf(probe_path, sizeof probe_path,
                     "/tmp/vtm_session_%ld_probe.vtms", (long)getpid());
            rc = vtm_api_session_save(lru, probe_path);
            check(rc == VTM_OK,
                  "save при недоступном каталоге лимита -> VTM_OK");
            unlink(probe_path);

            (void)vtm_api_sessions_limit(lru, 0);
            (void)vtm_api_sessions_dir(lru, NULL);
        }

        if (lru != NULL) vtm_api_close(lru);
        if (made_dir) {
            clear_dir(dir);
            (void)rmdir(dir);
        }
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
