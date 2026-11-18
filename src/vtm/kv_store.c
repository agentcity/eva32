// KV-сессии: снимок и восстановление состояния машины (задача 0079).
//
// Формат файла сессии (все поля заголовка — little-endian):
//
//   [0..63]    заголовок
//     +0  char[4] "VTMS"
//     +4  u32     версия формата (VTM_SESSION_VERSION)
//     +8  u32     размер заголовка (VTM_SESSION_HEADER)
//     +12 u32     flags (0; зарезервировано)
//     +16 u32     ctx_len   — размер кэша по позиции (dims[0])
//     +20 u32     kv_dim    — ширина кэша (dims[1])
//     +24 u32     n_layers  — число пар K/V (по инструкциям KVW)
//     +28 i32     position  — m->position на момент снимка
//     +32 i32     generated — счётчик SAMP (справочно, не восстанавливается)
//     +36 i32     pending_token — токен, ещё не скормленный машине (-1 — нет)
//     +40 u64     kv_bytes  — суммарный размер кадров K/V
//     +48 u8[16]  reserved (нули)
//
//   [64..)     кадры: для каждого слоя (в порядке инструкций KVW) сначала
//              ctx_len*kv_dim f32 K-кэша, затем столько же V-кэша. Байты
//              сырые, host-endian: сессия переносится между процессами на той
//              же машине, но не между архитектурами.
//   [конец)    trailer: char[4] "VTME" + sha256(заголовок + все кадры).
//
// Запись атомарная (ADR 0008, правило 1): temp-файл рядом с целевым, fsync,
// rename, fsync каталога. Чтение только после проверки размера, магии,
// версии и sha256; при любом несовпадении машина остаётся нетронутой.
//
// Обрыв файла (нет trailer) — ошибка VTM_E_TRUNCATED: недописанную сессию
// нельзя принять за целую, а докатывать её до «последнего целого кадра»
// нельзя — состояние из неполного набора слоёв заведомо неверно.
//
// Лимит каталога сессий (задача 0083): vtm_session_prune() считает суммарный
// размер обычных файлов в каталоге и, если он больше лимита, удаляет самые
// старые по mtime. Файл, который прямо сейчас пишется/читается (keep_path),
// и temp-файлы вида "<сессия>.tmp.<pid>" не трогает; ошибки unlink не
// фатальны — предупреждение в stderr и идём дальше. Функцию зовёт api.c после
// save/load, когда заданы vtm_api_sessions_dir() и vtm_api_sessions_limit().
//
// Функции не публичные (api.h остаётся C-ABI): их зовёт api.c. Прототипы
// объявлены там же, чтобы не заводить ещё один заголовок в зоне задачи.
#include "vtm.h"
#include "sha256.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define VTM_SESSION_MAGIC         "VTMS"
#define VTM_SESSION_TRAILER_MAGIC "VTME"
#define VTM_SESSION_VERSION       1u
#define VTM_SESSION_HEADER        64u
#define VTM_SESSION_TRAILER       36u
// Больше слоёв в одной сессии не поддерживаем: массив указателей на стеке
// фиксированного размера, а моделей с тысячей KVW-инструкций не бывает.
#define VTM_SESSION_MAX_LAYERS    1024u
// Защита от абсурдных dims в битом заголовке: 2^20 по каждой оси — с запасом.
#define VTM_SESSION_MAX_DIM       1048576u

// --- little-endian поля и текст ошибки -------------------------------------

static void put_u32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}

static uint32_t get_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8)
         | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static void put_u64(uint8_t *p, uint64_t v) {
    unsigned i;
    for (i = 0; i < 8u; i++) p[i] = (uint8_t)(v >> (8u * i));
}

static uint64_t get_u64(const uint8_t *p) {
    uint64_t v = 0;
    unsigned i;
    for (i = 0; i < 8u; i++) v |= (uint64_t)p[i] << (8u * i);
    return v;
}

static int session_fail(char *err, size_t errcap, int code,
                        const char *fmt, ...) {
    va_list ap;
    if (err != NULL && errcap != 0) {
        va_start(ap, fmt);
        vsnprintf(err, errcap, fmt, ap);
        va_end(ap);
    }
    return code;
}

// --- геометрия и сбор KV-качелей ------------------------------------------

// Собрать пары K/V по инструкциям KVW: dst — K-кэш, b — V-кэш. Так не нужно
// знать имена тензоров и не требуется лезть во внутренности интерпретатора.
static int collect_kv(const vtm_t *m, const vtm_tensor **ks,
                      const vtm_tensor **vs, uint32_t *n_out,
                      char *err, size_t errcap) {
    uint32_t n_instr = vtm_instr_count(m);
    uint32_t i, n = 0;

    for (i = 0; i < n_instr; i++) {
        const vtm_instr *in = vtm_instruction(m, i);
        const vtm_tensor *k, *v;

        if (in == NULL || in->opcode != VTM_OP_KVW) continue;
        if (in->dst == VTM_NO_TENSOR || in->b == VTM_NO_TENSOR)
            return session_fail(err, errcap, VTM_E_FORMAT,
                                "KVW без dst/b: не найти KV-кэш");
        k = vtm_tensor_by_index(m, in->dst);
        v = vtm_tensor_by_index(m, in->b);
        if (k == NULL || v == NULL)
            return session_fail(err, errcap, VTM_E_TENSOR,
                                "KVW ссылается на несуществующий тензор");
        if (k->kind != VTM_TENSOR_F32 || v->kind != VTM_TENSOR_F32
            || k->f32 == NULL || v->f32 == NULL)
            return session_fail(err, errcap, VTM_E_TENSOR,
                                "KV-кэш '%s'/'%s' не f32 в арене",
                                k->name, v->name);
        if (k->ndim != 2 || v->ndim != 2
            || k->dims[0] != v->dims[0] || k->dims[1] != v->dims[1])
            return session_fail(err, errcap, VTM_E_SHAPE,
                                "KV-кэши '%s' и '%s' должны быть [ctx, kv_dim]",
                                k->name, v->name);
        if (n >= VTM_SESSION_MAX_LAYERS)
            return session_fail(err, errcap, VTM_E_UNSUPPORTED,
                                "слоёв с KVW больше %u",
                                (unsigned)VTM_SESSION_MAX_LAYERS);
        if (n > 0
            && (k->dims[0] != ks[0]->dims[0]
                || k->dims[1] != ks[0]->dims[1]))
            return session_fail(err, errcap, VTM_E_UNSUPPORTED,
                                "слои с разной формой KV-кэшей "
                                "(%ux%u и %ux%u) не поддерживаются",
                                (unsigned)ks[0]->dims[0],
                                (unsigned)ks[0]->dims[1],
                                (unsigned)k->dims[0], (unsigned)k->dims[1]);
        ks[n] = k;
        vs[n] = v;
        n++;
    }
    if (n == 0)
        return session_fail(err, errcap, VTM_E_UNSUPPORTED,
                            "в программе нет ни одной инструкции KVW: "
                            "сохранять нечего");
    *n_out = n;
    return VTM_OK;
}

static int kv_geometry(uint32_t ctx_len, uint32_t kv_dim, uint32_t n_layers,
                       uint64_t *frame_bytes, uint64_t *kv_bytes,
                       char *err, size_t errcap) {
    uint64_t one, total;

    if (ctx_len == 0 || kv_dim == 0 || n_layers == 0)
        return session_fail(err, errcap, VTM_E_FORMAT,
                            "нулевая размерность сессии: ctx=%u kv_dim=%u "
                            "layers=%u", (unsigned)ctx_len,
                            (unsigned)kv_dim, (unsigned)n_layers);
    if (ctx_len > VTM_SESSION_MAX_DIM || kv_dim > VTM_SESSION_MAX_DIM)
        return session_fail(err, errcap, VTM_E_FORMAT,
                            "абсурдные размеры KV-кэша: %ux%u",
                            (unsigned)ctx_len, (unsigned)kv_dim);
    one = (uint64_t)ctx_len * (uint64_t)kv_dim * 4u;
    if (one > (uint64_t)SIZE_MAX / 2u)
        return session_fail(err, errcap, VTM_E_NOMEM,
                            "размер KV-кадра не влезает в size_t");
    total = one * 2u * (uint64_t)n_layers;
    if (total > (uint64_t)SIZE_MAX)
        return session_fail(err, errcap, VTM_E_NOMEM,
                            "суммарный размер KV-кадров не влезает в size_t");
    *frame_bytes = one;
    *kv_bytes = total;
    return VTM_OK;
}

// --- запись ----------------------------------------------------------------

static int write_all(FILE *f, const void *p, size_t n,
                     char *err, size_t errcap) {
    if (n != 0 && fwrite(p, 1, n, f) != n)
        return session_fail(err, errcap, VTM_E_IO, "запись сессии: %s",
                            strerror(errno));
    return VTM_OK;
}

// fsync каталога: чтобы после rename запись каталога пережила падение. Ошибку
// игнорируем: на некоторых ФС/платформах fsync каталога не поддерживается, а
// атомарность самого rename от этого не страдает.
static void sync_parent_dir(const char *path) {
    const char *slash = strrchr(path, '/');
    char *dir;
    size_t len;
    int fd;

    if (slash == NULL) {
        dir = (char *)malloc(2);
        if (dir == NULL) return;
        dir[0] = '.';
        dir[1] = '\0';
    } else {
        len = (size_t)(slash - path);
        if (len == 0) len = 1; // путь от корня: "/file"
        dir = (char *)malloc(len + 1);
        if (dir == NULL) return;
        memcpy(dir, path, len);
        dir[len] = '\0';
    }
    fd = open(dir, O_RDONLY);
    if (fd >= 0) {
        (void)fsync(fd);
        (void)close(fd);
    }
    free(dir);
}

int vtm_session_save(const vtm_t *m, int32_t pending_token, const char *path,
                     char *err, size_t errcap) {
    const vtm_tensor *ks[VTM_SESSION_MAX_LAYERS];
    const vtm_tensor *vs[VTM_SESSION_MAX_LAYERS];
    uint8_t hdr[VTM_SESSION_HEADER];
    uint8_t trailer[VTM_SESSION_TRAILER];
    vtm_sha256_ctx sha;
    uint32_t n_layers = 0;
    uint32_t ctx_len, kv_dim;
    uint64_t frame_bytes = 0, kv_bytes = 0;
    char *tmp = NULL;
    size_t plen;
    FILE *f = NULL;
    uint32_t l;
    int rc;

    if (m == NULL || path == NULL || *path == '\0')
        return session_fail(err, errcap, VTM_E_ARG,
                            "session_save: пустой путь или машина");
    rc = collect_kv(m, ks, vs, &n_layers, err, errcap);
    if (rc != VTM_OK) return rc;
    ctx_len = ks[0]->dims[0];
    kv_dim = ks[0]->dims[1];
    rc = kv_geometry(ctx_len, kv_dim, n_layers, &frame_bytes, &kv_bytes,
                     err, errcap);
    if (rc != VTM_OK) return rc;
    if (vtm_position(m) < 0)
        return session_fail(err, errcap, VTM_E_ARG,
                            "session_save: отрицательная позиция %d",
                            vtm_position(m));

    memset(hdr, 0, sizeof hdr);
    memcpy(hdr, VTM_SESSION_MAGIC, 4);
    put_u32(hdr + 4, VTM_SESSION_VERSION);
    put_u32(hdr + 8, VTM_SESSION_HEADER);
    put_u32(hdr + 12, 0);
    put_u32(hdr + 16, ctx_len);
    put_u32(hdr + 20, kv_dim);
    put_u32(hdr + 24, n_layers);
    put_u32(hdr + 28, (uint32_t)vtm_position(m));
    put_u32(hdr + 32, (uint32_t)vtm_generated(m));
    put_u32(hdr + 36, (uint32_t)pending_token);
    put_u64(hdr + 40, kv_bytes);

    // temp-файл рядом с целевым: rename в пределах одного каталога атомарен.
    plen = strlen(path);
    tmp = (char *)malloc(plen + 32u);
    if (tmp == NULL)
        return session_fail(err, errcap, VTM_E_NOMEM,
                            "session_save: нет памяти под имя temp-файла");
    snprintf(tmp, plen + 32u, "%s.tmp.%ld", path, (long)getpid());

    f = fopen(tmp, "wb");
    if (f == NULL) {
        rc = session_fail(err, errcap, VTM_E_IO,
                          "session_save: %s: %s", tmp, strerror(errno));
        free(tmp);
        return rc;
    }

    vtm_sha256_init(&sha);
    vtm_sha256_update(&sha, hdr, sizeof hdr);
    rc = write_all(f, hdr, sizeof hdr, err, errcap);
    if (rc != VTM_OK) goto fail;

    for (l = 0; l < n_layers && rc == VTM_OK; l++) {
        vtm_sha256_update(&sha, ks[l]->f32, (size_t)frame_bytes);
        rc = write_all(f, ks[l]->f32, (size_t)frame_bytes, err, errcap);
        if (rc != VTM_OK) break;
        vtm_sha256_update(&sha, vs[l]->f32, (size_t)frame_bytes);
        rc = write_all(f, vs[l]->f32, (size_t)frame_bytes, err, errcap);
    }
    if (rc != VTM_OK) goto fail;

    memcpy(trailer, VTM_SESSION_TRAILER_MAGIC, 4);
    vtm_sha256_final(&sha, trailer + 4);
    rc = write_all(f, trailer, sizeof trailer, err, errcap);
    if (rc != VTM_OK) goto fail;

    if (fflush(f) != 0) {
        rc = session_fail(err, errcap, VTM_E_IO, "session_save: fflush: %s",
                          strerror(errno));
        goto fail;
    }
    if (fsync(fileno(f)) != 0) {
        rc = session_fail(err, errcap, VTM_E_IO, "session_save: fsync: %s",
                          strerror(errno));
        goto fail;
    }
    if (fclose(f) != 0) {
        f = NULL;
        rc = session_fail(err, errcap, VTM_E_IO, "session_save: fclose: %s",
                          strerror(errno));
        goto fail;
    }
    f = NULL;

    if (rename(tmp, path) != 0) {
        rc = session_fail(err, errcap, VTM_E_IO,
                          "session_save: rename %s -> %s: %s",
                          tmp, path, strerror(errno));
        goto fail;
    }
    sync_parent_dir(path);
    free(tmp);
    return VTM_OK;

fail:
    if (f != NULL) (void)fclose(f);
    if (tmp != NULL) {
        (void)unlink(tmp);
        free(tmp);
    }
    return rc;
}

// --- чтение ----------------------------------------------------------------

int vtm_session_load(vtm_t *m, int32_t *pending_token, const char *path,
                     char *err, size_t errcap) {
    const vtm_tensor *ks[VTM_SESSION_MAX_LAYERS];
    const vtm_tensor *vs[VTM_SESSION_MAX_LAYERS];
    uint8_t hdr[VTM_SESSION_HEADER];
    uint8_t trailer[VTM_SESSION_TRAILER];
    uint8_t digest[32];
    vtm_sha256_ctx sha;
    struct stat st;
    uint32_t n_layers = 0, ctx_len, kv_dim, file_layers;
    uint64_t frame_bytes = 0, kv_bytes = 0, file_kv_bytes, expected;
    uint32_t position, pending_raw;
    uint8_t *buf = NULL;
    FILE *f = NULL;
    uint32_t l;
    size_t off;
    int rc;

    if (m == NULL || path == NULL || *path == '\0')
        return session_fail(err, errcap, VTM_E_ARG,
                            "session_load: пустой путь или машина");
    if (stat(path, &st) != 0)
        return session_fail(err, errcap, VTM_E_IO,
                            "session_load: %s: %s", path, strerror(errno));
    if (st.st_size < (off_t)(VTM_SESSION_HEADER + VTM_SESSION_TRAILER))
        return session_fail(err, errcap, VTM_E_TRUNCATED,
                            "session_load: %s короче заголовка с trailer "
                            "(%lld байт)", path, (long long)st.st_size);

    f = fopen(path, "rb");
    if (f == NULL)
        return session_fail(err, errcap, VTM_E_IO,
                            "session_load: %s: %s", path, strerror(errno));
    if (fread(hdr, 1, sizeof hdr, f) != sizeof hdr) {
        rc = session_fail(err, errcap, VTM_E_TRUNCATED,
                          "session_load: %s: заголовок оборван", path);
        goto fail;
    }
    if (memcmp(hdr, VTM_SESSION_MAGIC, 4) != 0) {
        rc = session_fail(err, errcap, VTM_E_MAGIC,
                          "session_load: %s: нет магии \"VTMS\"", path);
        goto fail;
    }
    if (get_u32(hdr + 4) != VTM_SESSION_VERSION) {
        rc = session_fail(err, errcap, VTM_E_VERSION,
                          "session_load: %s: версия %u, поддерживается %u",
                          path, (unsigned)get_u32(hdr + 4),
                          (unsigned)VTM_SESSION_VERSION);
        goto fail;
    }
    if (get_u32(hdr + 8) != VTM_SESSION_HEADER || get_u32(hdr + 12) != 0) {
        rc = session_fail(err, errcap, VTM_E_FORMAT,
                          "session_load: %s: неверный размер заголовка "
                          "или flags", path);
        goto fail;
    }
    ctx_len = get_u32(hdr + 16);
    kv_dim = get_u32(hdr + 20);
    file_layers = get_u32(hdr + 24);
    position = get_u32(hdr + 28);
    pending_raw = get_u32(hdr + 36);
    file_kv_bytes = get_u64(hdr + 40);

    if ((int32_t)position < 0 || file_layers > VTM_SESSION_MAX_LAYERS) {
        rc = session_fail(err, errcap, VTM_E_FORMAT,
                          "session_load: %s: некорректные position/слои "
                          "в заголовке", path);
        goto fail;
    }
    rc = kv_geometry(ctx_len, kv_dim, file_layers, &frame_bytes, &kv_bytes,
                     err, errcap);
    if (rc != VTM_OK) goto fail;
    if (file_kv_bytes != kv_bytes) {
        rc = session_fail(err, errcap, VTM_E_FORMAT,
                          "session_load: %s: kv_bytes=%llu не сходится с "
                          "геометрией (%llu)", path,
                          (unsigned long long)file_kv_bytes,
                          (unsigned long long)kv_bytes);
        goto fail;
    }
    expected = (uint64_t)VTM_SESSION_HEADER + kv_bytes
             + (uint64_t)VTM_SESSION_TRAILER;
    if ((uint64_t)st.st_size < expected) {
        rc = session_fail(err, errcap, VTM_E_TRUNCATED,
                          "session_load: %s оборван: %lld из %llu байт, "
                          "trailer отсутствует", path,
                          (long long)st.st_size,
                          (unsigned long long)expected);
        goto fail;
    }
    if ((uint64_t)st.st_size > expected) {
        rc = session_fail(err, errcap, VTM_E_FORMAT,
                          "session_load: %s: лишние байты в конце "
                          "(%lld вместо %llu)", path,
                          (long long)st.st_size,
                          (unsigned long long)expected);
        goto fail;
    }

    rc = collect_kv(m, ks, vs, &n_layers, err, errcap);
    if (rc != VTM_OK) goto fail;
    if (n_layers != file_layers || ks[0]->dims[0] != ctx_len
        || ks[0]->dims[1] != kv_dim) {
        rc = session_fail(err, errcap, VTM_E_SHAPE,
                          "session_load: сессия %ux%u×%u слоёв не подходит "
                          "машине %ux%u×%u", (unsigned)ctx_len,
                          (unsigned)kv_dim, (unsigned)file_layers,
                          (unsigned)ks[0]->dims[0],
                          (unsigned)ks[0]->dims[1], (unsigned)n_layers);
        goto fail;
    }

    buf = (uint8_t *)malloc(kv_bytes != 0 ? (size_t)kv_bytes : 1u);
    if (buf == NULL) {
        rc = session_fail(err, errcap, VTM_E_NOMEM,
                          "session_load: нет памяти под %llu байт KV",
                          (unsigned long long)kv_bytes);
        goto fail;
    }
    if (fread(buf, 1, (size_t)kv_bytes, f) != (size_t)kv_bytes) {
        rc = session_fail(err, errcap, VTM_E_TRUNCATED,
                          "session_load: %s: KV-кадры оборваны", path);
        goto fail;
    }
    if (fread(trailer, 1, sizeof trailer, f) != sizeof trailer) {
        rc = session_fail(err, errcap, VTM_E_TRUNCATED,
                          "session_load: %s: trailer оборван", path);
        goto fail;
    }
    if (memcmp(trailer, VTM_SESSION_TRAILER_MAGIC, 4) != 0) {
        rc = session_fail(err, errcap, VTM_E_INTEGRITY,
                          "session_load: %s: нет trailer \"VTME\" — файл "
                          "недописан", path);
        goto fail;
    }
    vtm_sha256_init(&sha);
    vtm_sha256_update(&sha, hdr, sizeof hdr);
    vtm_sha256_update(&sha, buf, (size_t)kv_bytes);
    vtm_sha256_final(&sha, digest);
    if (memcmp(digest, trailer + 4, sizeof digest) != 0) {
        rc = session_fail(err, errcap, VTM_E_INTEGRITY,
                          "session_load: %s: sha256 не совпал — файл "
                          "повреждён или сессия недописана", path);
        goto fail;
    }

    // Всё проверено — только теперь трогаем машину.
    vtm_reset(m);
    off = 0;
    for (l = 0; l < n_layers; l++) {
        memcpy(ks[l]->f32, buf + off, (size_t)frame_bytes);
        off += (size_t)frame_bytes;
        memcpy(vs[l]->f32, buf + off, (size_t)frame_bytes);
        off += (size_t)frame_bytes;
    }
    vtm_set_position(m, (int)position);
    if (pending_token != NULL) {
        const int32_t p = (int32_t)pending_raw;
        *pending_token = (p < -1) ? -1 : p;
    }

    free(buf);
    (void)fclose(f);
    return VTM_OK;

fail:
    free(buf);
    if (f != NULL) (void)fclose(f);
    return rc;
}

// --- каталог сессий: лимит и вытеснение старых (задача 0083) ----------------

// Запись сессии, которую видит вытеснение. Путь свой (malloc), чтобы не
// зависеть от буфера readdir; dev/ino — чтобы не удалить файл, который прямо
// сейчас пишется или читается (его защищает keep_path).
typedef struct {
    char *path;
    uint64_t size;
    time_t mtime_sec;
    long mtime_nsec;
    dev_t dev;
    ino_t ino;
} session_entry;

// Наносекунды mtime: строгий POSIX на macOS не даёт st_mtim, зато даёт
// st_mtimensec; Linux — наоборот.
#if defined(__APPLE__)
#define SESS_MTIME_SEC(st)  ((st).st_mtime)
#define SESS_MTIME_NSEC(st) ((st).st_mtimensec)
#else
#define SESS_MTIME_SEC(st)  ((st).st_mtime)
#define SESS_MTIME_NSEC(st) ((st).st_mtim.tv_nsec)
#endif

// Временный файл записи: "<сессия>.tmp.<pid>". Их не считаем и не удаляем:
// соседний процесс может писать сессию прямо сейчас.
static int sess_name_is_tmp(const char *name) {
    const char *p = strstr(name, ".tmp.");
    if (p == NULL) return 0;
    p += 5; // strlen(".tmp.")
    if (*p == '\0') return 0;
    for (; *p != '\0'; p++)
        if (*p < '0' || *p > '9') return 0;
    return 1;
}

// Старые — первыми: по mtime, при равных mtime — по имени (детерминизм).
static int session_entry_cmp(const void *a, const void *b) {
    const session_entry *x = (const session_entry *)a;
    const session_entry *y = (const session_entry *)b;

    if (x->mtime_sec != y->mtime_sec)
        return x->mtime_sec < y->mtime_sec ? -1 : 1;
    if (x->mtime_nsec != y->mtime_nsec)
        return x->mtime_nsec < y->mtime_nsec ? -1 : 1;
    return strcmp(x->path, y->path);
}

int vtm_session_prune(const char *dir, uint64_t limit_bytes,
                      const char *keep_path) {
    DIR *d;
    struct dirent *de;
    struct stat kst, st;
    session_entry *v = NULL;
    size_t n = 0, cap = 0, i, dlen;
    uint64_t total = 0;
    int have_keep = 0;
    int removed = 0;

    if (dir == NULL || *dir == '\0' || limit_bytes == 0) return 0;
    if (keep_path != NULL && *keep_path != '\0' && stat(keep_path, &kst) == 0)
        have_keep = 1;
    d = opendir(dir);
    if (d == NULL) {
        fprintf(stderr, "vtm: сессии: %s: %s\n", dir, strerror(errno));
        return 0;
    }
    dlen = strlen(dir);
    while ((de = readdir(d)) != NULL) {
        const int slash = (dlen != 0 && dir[dlen - 1] == '/') ? 0 : 1;
        size_t plen;
        char *path;

        if (strcmp(de->d_name, ".") == 0 || strcmp(de->d_name, "..") == 0)
            continue;
        if (sess_name_is_tmp(de->d_name)) continue;
        plen = dlen + (size_t)slash + strlen(de->d_name) + 1u;
        path = (char *)malloc(plen);
        if (path == NULL) break;
        snprintf(path, plen, "%s%s%s", dir, slash ? "/" : "", de->d_name);
        if (stat(path, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size < 0) {
            free(path);
            continue;
        }
        if (n == cap) {
            size_t ncap = cap == 0 ? 16u : cap * 2u;
            session_entry *nv =
                (session_entry *)realloc(v, ncap * sizeof *v);
            if (nv == NULL) {
                free(path);
                break;
            }
            v = nv;
            cap = ncap;
        }
        v[n].path = path;
        v[n].size = (uint64_t)st.st_size;
        v[n].mtime_sec = SESS_MTIME_SEC(st);
        v[n].mtime_nsec = SESS_MTIME_NSEC(st);
        v[n].dev = st.st_dev;
        v[n].ino = st.st_ino;
        total += v[n].size;
        n++;
    }
    (void)closedir(d);

    if (total > limit_bytes && n > 1u)
        qsort(v, n, sizeof *v, session_entry_cmp);

    // Удаляем с самых старых, пока не уложимся в лимит. Последний оставшийся
    // файл не трогаем, даже если он сам больше лимита: иначе снимок, который
    // только что записали, исчез бы. Ошибка unlink не фатальна — идём дальше.
    for (i = 0; i < n && total > limit_bytes; i++) {
        if (have_keep && v[i].dev == kst.st_dev && v[i].ino == kst.st_ino)
            continue;
        if (unlink(v[i].path) != 0) {
            fprintf(stderr, "vtm: сессии: не удалось удалить %s: %s\n",
                    v[i].path, strerror(errno));
            continue;
        }
        total -= v[i].size;
        removed++;
    }
    for (i = 0; i < n; i++) free(v[i].path);
    free(v);
    return removed;
}
