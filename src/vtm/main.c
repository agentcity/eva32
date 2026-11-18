// CLI VTM: run / inspect / bench.
//
//   vtm run model.vtm --tokens 1,2,3 --max-new 8
//   vtm run model.vtm --text "вопрос" --tokenizer tokenizer.json --max-new 8
//   vtm inspect model.vtm
//   vtm bench [--rows 512 --cols 512 --iters 200]
//
// Без аргументов печатается справка и код возврата 2 (как принято для CLI).
//
// Сборка: CLI — C99 + libvtm.a, но режим --text линкуется с BPE-токенизатором.
// tokenizer.hpp объявляет `namespace vtm`, а vtm.h — `struct vtm`; в одной
// единице трансляции C++ они несовместимы. Поэтому Makefile собирает этот же
// файл второй раз как C++17 с -DVTM_TOK_BRIDGE (tok_bridge.o) — получается
// узкий C-ABI без vtm.h, а main.c остаётся обычным C99.

#ifdef VTM_TOK_BRIDGE
// ---------------------------------------------------------------------------
// C++17-мост к tokenizer.hpp. Никакого vtm.h здесь нет.
// ---------------------------------------------------------------------------
#include "tokenizer.hpp"

#include <stddef.h>
#include <stdint.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

struct vtm_tok {
    vtm::Tokenizer tok;
};

static void bridge_error(char *err, size_t errlen, const char *what) {
    if (err == NULL || errlen == 0) return;
    std::snprintf(err, errlen, "%s", what ? what : "неизвестная ошибка");
}

// Как vtm-tok: если путь не открылся относительно cwd, пробуем родителей cwd.
static std::string resolve_path(const std::string &given) {
    FILE *f = NULL;
    if (given.empty()) return given;
    f = std::fopen(given.c_str(), "rb");
    if (f) {
        std::fclose(f);
        return given;
    }
    {
        char cwd[4096];
        if (!getcwd(cwd, sizeof cwd)) return given;
        std::string dir = cwd;
        for (;;) {
            const std::string candidate = dir + "/" + given;
            FILE *g = std::fopen(candidate.c_str(), "rb");
            if (g) {
                std::fclose(g);
                return candidate;
            }
            if (dir == "/" || dir.empty()) break;
            const std::size_t slash = dir.rfind('/');
            if (slash == std::string::npos) break;
            if (slash == 0) dir = "/";
            else dir.resize(slash);
        }
    }
    return given;
}

extern "C" {

const char *vtm_tok_default_path(void) {
    static std::string path;
    try {
        path = vtm::Tokenizer::default_path();
    } catch (...) {
        path.clear();
    }
    return path.c_str();
}

vtm_tok *vtm_tok_open(const char *given, char *err, size_t errlen) {
    if (given == NULL || *given == '\0') {
        bridge_error(err, errlen, "пустой путь к tokenizer.json");
        return NULL;
    }
    try {
        const std::string path = resolve_path(given);
        vtm::Tokenizer loaded = vtm::Tokenizer::load(path);
        vtm_tok *handle = new vtm_tok;
        handle->tok = std::move(loaded);
        return handle;
    } catch (const std::exception &e) {
        bridge_error(err, errlen, e.what());
        return NULL;
    } catch (...) {
        bridge_error(err, errlen, "неизвестная ошибка загрузки tokenizer.json");
        return NULL;
    }
}

void vtm_tok_close(vtm_tok *handle) {
    delete handle;
}

int vtm_tok_encode(vtm_tok *handle, const char *text, int32_t **out_ids,
                   size_t *out_n, char *err, size_t errlen) {
    if (out_ids != NULL) *out_ids = NULL;
    if (out_n != NULL) *out_n = 0;
    if (handle == NULL || out_ids == NULL || out_n == NULL) {
        bridge_error(err, errlen, "мост токенизатора не инициализирован");
        return -1;
    }
    try {
        const std::vector<int32_t> ids = handle->tok.encode(text ? text : "");
        if (ids.empty()) return 0;
        int32_t *copy =
            (int32_t *)std::malloc(ids.size() * sizeof(int32_t));
        if (copy == NULL) {
            bridge_error(err, errlen, "не хватило памяти под токены");
            return -1;
        }
        std::memcpy(copy, ids.data(), ids.size() * sizeof(int32_t));
        *out_ids = copy;
        *out_n = ids.size();
        return 0;
    } catch (const std::exception &e) {
        bridge_error(err, errlen, e.what());
        return -1;
    } catch (...) {
        bridge_error(err, errlen, "неизвестная ошибка токенизации");
        return -1;
    }
}

int vtm_tok_decode(vtm_tok *handle, const int32_t *ids, size_t n,
                   char **out_text, char *err, size_t errlen) {
    if (out_text != NULL) *out_text = NULL;
    if (handle == NULL || out_text == NULL || (ids == NULL && n != 0)) {
        bridge_error(err, errlen, "мост токенизатора не инициализирован");
        return -1;
    }
    try {
        std::vector<int32_t> gen;
        if (n != 0) gen.assign(ids, ids + n);
        const std::string text = handle->tok.decode(gen);
        char *copy = (char *)std::malloc(text.size() + 1);
        if (copy == NULL) {
            bridge_error(err, errlen, "не хватило памяти под текст ответа");
            return -1;
        }
        std::memcpy(copy, text.c_str(), text.size() + 1);
        *out_text = copy;
        return 0;
    } catch (const std::exception &e) {
        bridge_error(err, errlen, e.what());
        return -1;
    } catch (...) {
        bridge_error(err, errlen, "неизвестная ошибка детокенизации");
        return -1;
    }
}

// --- стриминг: инкрементальный декодер по одному токену --------------------
//
// Один объект — одна генерация. Держит недописанные байты UTF-8 между
// токенами (буква может прийти двумя BPE-токенами), не печатает спецтокены и
// сообщает о стоп-токене.

struct vtm_tok_stream {
    vtm::DecoderStream *dec;
};

static char *stream_dup(const std::string &text) {
    char *copy = (char *)std::malloc(text.size() + 1);
    if (copy == NULL) return NULL;
    std::memcpy(copy, text.data(), text.size());
    copy[text.size()] = '\0';
    return copy;
}

vtm_tok_stream *vtm_tok_stream_open(vtm_tok *handle) {
    if (handle == NULL) return NULL;
    try {
        vtm_tok_stream *s = new vtm_tok_stream;
        s->dec = NULL;
        try {
            s->dec = new vtm::DecoderStream(handle->tok);
        } catch (...) {
            delete s;
            throw;
        }
        return s;
    } catch (...) {
        return NULL;
    }
}

void vtm_tok_stream_close(vtm_tok_stream *s) {
    if (s == NULL) return;
    delete s->dec;
    delete s;
}

int vtm_tok_stream_push(vtm_tok_stream *s, int32_t id, char **out_text,
                        int *out_stop, int32_t *out_stop_id, char *err,
                        size_t errlen) {
    if (out_text != NULL) *out_text = NULL;
    if (out_stop != NULL) *out_stop = 0;
    if (out_stop_id != NULL) *out_stop_id = -1;
    if (s == NULL || s->dec == NULL || out_text == NULL || out_stop == NULL) {
        bridge_error(err, errlen, "стриминговый декодер не инициализирован");
        return -1;
    }
    try {
        bool stop = false;
        const std::string piece = s->dec->push(id, &stop);
        char *copy = stream_dup(piece);
        if (copy == NULL) {
            bridge_error(err, errlen, "не хватило памяти под текст токена");
            return -1;
        }
        *out_text = copy;
        *out_stop = stop ? 1 : 0;
        if (out_stop_id != NULL) *out_stop_id = stop ? id : -1;
        return 0;
    } catch (const std::exception &e) {
        bridge_error(err, errlen, e.what());
        return -1;
    } catch (...) {
        bridge_error(err, errlen, "неизвестная ошибка стримингового декодера");
        return -1;
    }
}

int vtm_tok_stream_finish(vtm_tok_stream *s, char **out_text, char *err,
                          size_t errlen) {
    if (out_text != NULL) *out_text = NULL;
    if (s == NULL || s->dec == NULL || out_text == NULL) {
        bridge_error(err, errlen, "стриминговый декодер не инициализирован");
        return -1;
    }
    try {
        const std::string tail = s->dec->finish();
        char *copy = stream_dup(tail);
        if (copy == NULL) {
            bridge_error(err, errlen, "не хватило памяти под хвост ответа");
            return -1;
        }
        *out_text = copy;
        return 0;
    } catch (const std::exception &e) {
        bridge_error(err, errlen, e.what());
        return -1;
    } catch (...) {
        bridge_error(err, errlen, "неизвестная ошибка стримингового декодера");
        return -1;
    }
}

// added_tokens: чат-шаблон берёт из них content/id спецтокенов
// (<|im_start|>, <|im_end|>), поэтому id в C-части не захардкожены.
int vtm_tok_added_count(vtm_tok *handle) {
    if (handle == NULL) return -1;
    try {
        return (int)handle->tok.added_tokens().size();
    } catch (...) {
        return -1;
    }
}

// 1 — content выделен malloc (освобождать vtm_tok_free_text);
// 0 — index за пределами; -1 — ошибка.
int vtm_tok_added_at(vtm_tok *handle, int index, char **out_content,
                     int32_t *out_id, int *out_special) {
    if (out_content != NULL) *out_content = NULL;
    if (handle == NULL || out_content == NULL || index < 0) return -1;
    try {
        const std::vector<vtm::AddedTokenInfo> all = handle->tok.added_tokens();
        if ((std::size_t)index >= all.size()) return 0;
        const vtm::AddedTokenInfo &t = all[(std::size_t)index];
        char *copy = (char *)std::malloc(t.content.size() + 1);
        if (copy == NULL) return -1;
        std::memcpy(copy, t.content.data(), t.content.size());
        copy[t.content.size()] = '\0';
        *out_content = copy;
        if (out_id != NULL) *out_id = t.id;
        if (out_special != NULL) *out_special = t.special ? 1 : 0;
        return 1;
    } catch (...) {
        return -1;
    }
}

void vtm_tok_free_ids(int32_t *ids) {
    std::free(ids);
}

void vtm_tok_free_text(char *text) {
    std::free(text);
}

} // extern "C"

#else  // !VTM_TOK_BRIDGE
// ---------------------------------------------------------------------------
// CLI на C99: только libvtm.a и C-ABI моста токенизатора.
// ---------------------------------------------------------------------------
#include "vtm.h"
#include "vtm_backend.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// C-ABI tok_bridge.o (реализация — в начале этого файла под VTM_TOK_BRIDGE).
typedef struct vtm_tok vtm_tok;

const char *vtm_tok_default_path(void);
vtm_tok *vtm_tok_open(const char *given, char *err, size_t errlen);
void vtm_tok_close(vtm_tok *handle);
int vtm_tok_encode(vtm_tok *handle, const char *text, int32_t **out_ids,
                   size_t *out_n, char *err, size_t errlen);
int vtm_tok_decode(vtm_tok *handle, const int32_t *ids, size_t n,
                   char **out_text, char *err, size_t errlen);
int vtm_tok_added_count(vtm_tok *handle);
int vtm_tok_added_at(vtm_tok *handle, int index, char **out_content,
                     int32_t *out_id, int *out_special);
void vtm_tok_free_ids(int32_t *ids);
void vtm_tok_free_text(char *text);

// Стриминговый декодер (--stream): один объект на генерацию.
typedef struct vtm_tok_stream vtm_tok_stream;
vtm_tok_stream *vtm_tok_stream_open(vtm_tok *handle);
void vtm_tok_stream_close(vtm_tok_stream *s);
// 0 — ок; *out_text — malloc-строка готовых байт (может быть пустой),
// *out_stop = 1 — встретился стоп-токен (*out_stop_id — его id).
int vtm_tok_stream_push(vtm_tok_stream *s, int32_t id, char **out_text,
                        int *out_stop, int32_t *out_stop_id, char *err,
                        size_t errlen);
// Отдать недописанный хвост (битые байты -> U+FFFD).
int vtm_tok_stream_finish(vtm_tok_stream *s, char **out_text, char *err,
                          size_t errlen);

static void usage(FILE *out) {
    fprintf(out,
        "VTM — скалярный интерпретатор виртуальной троичной машины (v0.1)\n"
        "\n"
        "  vtm run model.vtm --tokens 1,2,3 --max-new 8\n"
        "      прогнать программу по токенам и напечатать выбранные токены\n"
        "      --backend auto|simd|scalar|lut|direct  бэкенд (по умолчанию VTM_BACKEND или lut)\n"
        "\n"
        "  vtm run model.vtm --text \"вопрос\" [--tokenizer PATH] [--max-new 8]\n"
        "      токенизировать текст (BPE), прогнать и напечатать ответ текстом\n"
        "      --tokenizer PATH       tokenizer.json; иначе $VTM_TOKENIZER,\n"
        "                             затем tests/vtm/fixtures/tokenizer.json\n"
        "      --chat                 обернуть текст в чат-шаблон Qwen2.5\n"
        "                             (<|im_start|>system/user/assistant <|im_end|>)\n"
        "      --system \"...\"         системное сообщение; только вместе с --chat\n"
        "      --stream               печатать ответ по мере генерации (stdout,\n"
        "                             без буферизации строки; только с --text).\n"
        "                             Декодер ждёт завершения UTF-8 символа и\n"
        "                             не печатает спецтокены; на стоп-токене\n"
        "                             генерация обрывается. Служебные строки —\n"
        "                             в stderr.\n"
        "      --backend auto|simd|scalar|lut|direct  бэкенд (по умолчанию VTM_BACKEND или lut)\n"
        "\n"
        "  vtm inspect model.vtm\n"
        "      показать заголовок, тензоры и инструкции\n"
        "\n"
        "  vtm bench [--rows N] [--cols N] [--iters N] [--scale-block 128|160]\n"
        "      замерить TMV: LUT243 против прямой распаковки\n"
        "\n"
        "  vtm --help\n"
        "\n"
        "Коды возврата: 0 — успех, 1 — ошибка выполнения, 2 — ошибка аргументов\n"
        "или токенизатора.\n");
}

static int parse_int(const char *s, int *out) {
    char *end = NULL;
    long v;
    if (!s || !*s) return -1;
    errno = 0;
    v = strtol(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0') return -1;
    if (v < INT_MIN || v > INT_MAX) return -1;
    *out = (int)v;
    return 0;
}

static int parse_int_list(const char *s, int **out, size_t *out_n) {
    const char *p = s;
    int *arr = NULL;
    size_t n = 0, cap = 0;

    *out = NULL;
    *out_n = 0;
    if (!s || !*s) return -1;

    while (*p) {
        char *end = NULL;
        long v;
        while (*p == ' ' || *p == '\t') p++;
        if (!*p) break;
        errno = 0;
        v = strtol(p, &end, 10);
        if (errno != 0 || end == p || v < 0 || v > INT_MAX) {
            free(arr);
            return -1;
        }
        if (n == cap) {
            size_t ncap = cap ? cap * 2u : 8u;
            int *tmp = (int *)realloc(arr, ncap * sizeof(int));
            if (!tmp) {
                free(arr);
                return -1;
            }
            arr = tmp;
            cap = ncap;
        }
        arr[n++] = (int)v;
        p = end;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == ',') {
            p++;
            continue;
        }
        if (*p != '\0') {
            free(arr);
            return -1;
        }
    }
    if (n == 0) {
        free(arr);
        return -1;
    }
    *out = arr;
    *out_n = n;
    return 0;
}

static const char *kind_name(uint8_t kind) {
    switch (kind) {
    case VTM_TENSOR_TRIT: return "trit";
    case VTM_TENSOR_F32:  return "f32";
    case VTM_TENSOR_F16:  return "f16";
    case VTM_TENSOR_BF16: return "bf16";
    default:              return "?";
    }
}

// --- run -------------------------------------------------------------------

typedef struct {
    int token;
    int have;
    int count;
} emit_state;

static int emit_first(void *user, int token, int position) {
    emit_state *s = (emit_state *)user;
    (void)position;
    if (!s->have) {
        s->token = token;
        s->have = 1;
    }
    s->count++;
    return 0;
}

static void print_tokens(FILE *out, const char *label, const int *v, size_t n) {
    size_t i;
    fprintf(out, "%s", label);
    for (i = 0; i < n; i++) fprintf(out, "%s%d", i ? " " : "", v[i]);
    fprintf(out, "\n");
}

// --- --stream: печать по токену -------------------------------------------
//
// Токен декодируется инкрементально: декодер держит недописанный хвост
// UTF-8, поэтому stdout получает только целые символы. Спецтокены не
// печатаются, стоп-токен обрывает генерацию (см. DecoderStream).
typedef struct {
    vtm_tok_stream *stream;
    int stopped;
    int32_t stop_id;
} stream_state;

// 0 — ок, -1 — ошибка декодера/записи (текст в err).
static int stream_token(stream_state *ss, int token, char *err, size_t errlen) {
    char *piece = NULL;
    int stop = 0;
    int32_t stop_id = -1;
    size_t len = 0;
    int rc;

    if (ss->stream == NULL) return 0;
    rc = vtm_tok_stream_push(ss->stream, (int32_t)token, &piece, &stop,
                             &stop_id, err, errlen);
    if (rc != 0) return -1;
    if (piece != NULL) len = strlen(piece);
    if (len != 0 && fwrite(piece, 1, len, stdout) != len) {
        vtm_tok_free_text(piece);
        snprintf(err, errlen, "не удалось записать текст в stdout");
        return -1;
    }
    if (len != 0) fflush(stdout);
    vtm_tok_free_text(piece);
    if (stop) {
        ss->stopped = 1;
        ss->stop_id = stop_id;
    }
    return 0;
}

// --- чат-шаблон Qwen2.5 ----------------------------------------------------
//
// Шаблон целиком:
//
//   <|im_start|>system\n{system}<|im_end|>\n
//   <|im_start|>user\n{text}<|im_end|>\n
//   <|im_start|>assistant\n
//
// Строки <|im_start|>/<|im_end|> не хардкодятся: они берутся из
// added_tokens tokenizer.json (special == true, content содержит
// "im_start"/"im_end"). Ход system добавляется только при --system — так же
// ведёт себя официальный шаблон Qwen, если системного сообщения нет.

typedef struct {
    char *data;
    size_t len;
    size_t cap;
    int oom;
} chat_buf;

static void chat_buf_free(chat_buf *b) {
    free(b->data);
    b->data = NULL;
    b->len = 0;
    b->cap = 0;
    b->oom = 0;
}

static int chat_buf_add_n(chat_buf *b, const char *s, size_t n) {
    if (b->oom) return -1;
    if (b->len + n + 1 > b->cap) {
        size_t cap = b->cap ? b->cap : 128;
        char *p;
        while (cap < b->len + n + 1) {
            if (cap > (size_t)-1 / 2) {
                b->oom = 1;
                return -1;
            }
            cap *= 2;
        }
        p = (char *)realloc(b->data, cap);
        if (!p) {
            b->oom = 1;
            return -1;
        }
        b->data = p;
        b->cap = cap;
    }
    if (n != 0) memcpy(b->data + b->len, s, n);
    b->len += n;
    b->data[b->len] = '\0';
    return 0;
}

static int chat_buf_add(chat_buf *b, const char *s) {
    return chat_buf_add_n(b, s, strlen(s));
}

// Ищет special added_token, чей content содержит needle, и копирует content
// в out. Так текст спецтокена приходит из tokenizer.json, а не из кода.
static int chat_find_special(vtm_tok *tok, const char *needle, char *out,
                             size_t cap, char *err, size_t errlen) {
    int n, i;
    if (tok == NULL) return -1;
    n = vtm_tok_added_count(tok);
    if (n < 0) {
        snprintf(err, errlen, "не удалось прочитать added_tokens");
        return -1;
    }
    for (i = 0; i < n; i++) {
        char *content = NULL;
        int32_t id = -1;
        int special = 0;
        int rc = vtm_tok_added_at(tok, i, &content, &id, &special);
        if (rc < 0) {
            snprintf(err, errlen, "не удалось прочитать added_tokens[%d]", i);
            return -1;
        }
        if (rc == 0 || !special || content == NULL) {
            vtm_tok_free_text(content);
            continue;
        }
        if (strstr(content, needle) == NULL) {
            vtm_tok_free_text(content);
            continue;
        }
        if (strlen(content) + 1 > cap) {
            snprintf(err, errlen, "added_token \"%s\" не влез в буфер",
                     content);
            vtm_tok_free_text(content);
            return -1;
        }
        memcpy(out, content, strlen(content) + 1);
        vtm_tok_free_text(content);
        return 0;
    }
    snprintf(err, errlen,
             "в tokenizer.json нет special added_token с \"%s\" в content",
             needle);
    return -1;
}

// Собирает промпт. Возвращает 0 и malloc-строку в *out либо -1 с err.
static int chat_build_prompt(vtm_tok *tok, const char *system_msg,
                             const char *user_msg, char **out, char *err,
                             size_t errlen) {
    char start[128];
    char end[128];
    chat_buf b;

    *out = NULL;
    if (chat_find_special(tok, "im_start", start, sizeof start, err, errlen) != 0)
        return -1;
    if (chat_find_special(tok, "im_end", end, sizeof end, err, errlen) != 0)
        return -1;

    memset(&b, 0, sizeof b);
    if (system_msg != NULL) {
        if (chat_buf_add(&b, start) != 0 || chat_buf_add(&b, "system\n") != 0 ||
            chat_buf_add(&b, system_msg) != 0 || chat_buf_add(&b, end) != 0 ||
            chat_buf_add(&b, "\n") != 0)
            goto oom;
    }
    if (chat_buf_add(&b, start) != 0 || chat_buf_add(&b, "user\n") != 0 ||
        chat_buf_add(&b, user_msg) != 0 || chat_buf_add(&b, end) != 0 ||
        chat_buf_add(&b, "\n") != 0 || chat_buf_add(&b, start) != 0 ||
        chat_buf_add(&b, "assistant\n") != 0)
        goto oom;

    *out = b.data;
    return 0;

oom:
    chat_buf_free(&b);
    snprintf(err, errlen, "не хватило памяти под чат-шаблон");
    return -1;
}

static int cmd_run(int argc, char **argv) {
    const char *path = NULL;
    const char *tokens_s = NULL;
    const char *text_s = NULL;
    const char *tokenizer_s = NULL;
    const char *system_s = NULL;
    const char *backend_name = NULL;  /* не задан — решает VTM_BACKEND/загрузчик */
    int chat = 0;
    int stream = 0;
    int max_new = 8;
    int *tokens = NULL;
    size_t n_tokens = 0;
    int *generated = NULL;
    size_t n_generated = 0;
    vtm_tok *tokenizer = NULL;
    vtm_tok_stream *tok_stream = NULL;
    char *prompt = NULL;
    char *answer = NULL;
    char tok_err[512];
    vtm_t *m;
    stream_state ss;
    clock_t c0, c1;
    int i;

    for (i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--tokens") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "vtm run: --tokens без значения\n");
                return 2;
            }
            tokens_s = argv[i];
        } else if (strcmp(argv[i], "--text") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "vtm run: --text без значения\n");
                return 2;
            }
            text_s = argv[i];
        } else if (strcmp(argv[i], "--tokenizer") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "vtm run: --tokenizer без значения\n");
                return 2;
            }
            tokenizer_s = argv[i];
        } else if (strcmp(argv[i], "--chat") == 0) {
            chat = 1;
        } else if (strcmp(argv[i], "--stream") == 0) {
            stream = 1;
        } else if (strcmp(argv[i], "--system") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "vtm run: --system без значения\n");
                return 2;
            }
            system_s = argv[i];
        } else if (strcmp(argv[i], "--max-new") == 0) {
            if (++i >= argc || parse_int(argv[i], &max_new) != 0 ||
                max_new < 0 || max_new > 1000000) {
                fprintf(stderr, "vtm run: плохое значение --max-new\n");
                return 2;
            }
        } else if (strcmp(argv[i], "--backend") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "vtm run: --backend без значения\n");
                return 2;
            }
            backend_name = argv[i];
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "vtm run: неизвестный аргумент %s\n", argv[i]);
            return 2;
        } else if (!path) {
            path = argv[i];
        } else {
            fprintf(stderr, "vtm run: лишний аргумент %s\n", argv[i]);
            return 2;
        }
    }
    if (!path || (!tokens_s && !text_s)) {
        fprintf(stderr, "vtm run: нужны model.vtm и --tokens или --text\n");
        return 2;
    }
    if (tokens_s && text_s) {
        fprintf(stderr, "vtm run: --tokens и --text взаимоисключающие\n");
        return 2;
    }
    if (tokenizer_s && !text_s) {
        fprintf(stderr, "vtm run: --tokenizer имеет смысл только с --text\n");
        return 2;
    }
    if (chat && !text_s) {
        fprintf(stderr, "vtm run: --chat имеет смысл только с --text\n");
        return 2;
    }
    if (stream && !text_s) {
        fprintf(stderr, "vtm run: --stream имеет смысл только с --text\n");
        return 2;
    }
    if (system_s && !chat) {
        fprintf(stderr, "vtm run: --system имеет смысл только с --chat\n");
        return 2;
    }
    if (backend_name && !vtm_backend_by_name(backend_name)) {
        fprintf(stderr, "vtm run: --backend должен быть auto, simd, scalar, "
                        "lut или direct\n");
        return 2;
    }
    if (tokens_s) {
        if (parse_int_list(tokens_s, &tokens, &n_tokens) != 0 ||
            n_tokens > (size_t)INT_MAX) {
            fprintf(stderr,
                    "vtm run: --tokens ожидает список неотрицательных чисел\n");
            return 2;
        }
    } else {
        const char *tok_path = tokenizer_s ? tokenizer_s
                                           : vtm_tok_default_path();
        int32_t *ids = NULL;
        size_t n_ids = 0;

        if (!tok_path || !*tok_path) {
            fprintf(stderr,
                    "vtm run: не найден tokenizer.json: укажите "
                    "--tokenizer PATH или $VTM_TOKENIZER\n");
            return 2;
        }
        tokenizer = vtm_tok_open(tok_path, tok_err, sizeof tok_err);
        if (!tokenizer) {
            fprintf(stderr, "vtm run: токенизатор %s: %s\n", tok_path, tok_err);
            return 2;
        }
        if (chat) {
            if (chat_build_prompt(tokenizer, system_s, text_s, &prompt, tok_err,
                                  sizeof tok_err) != 0) {
                fprintf(stderr, "vtm run: чат-шаблон: %s\n", tok_err);
                vtm_tok_close(tokenizer);
                return 2;
            }
        }
        if (vtm_tok_encode(tokenizer, prompt ? prompt : text_s, &ids, &n_ids,
                           tok_err, sizeof tok_err) != 0) {
            fprintf(stderr, "vtm run: токенизатор %s: %s\n", tok_path, tok_err);
            free(prompt);
            vtm_tok_close(tokenizer);
            return 2;
        }
        if (n_ids == 0) {
            fprintf(stderr,
                    "vtm run: --text \"%s\" дал пустой список токенов "
                    "(проверьте tokenizer.json: %s)\n", text_s, tok_path);
            free(prompt);
            vtm_tok_free_ids(ids);
            vtm_tok_close(tokenizer);
            return 2;
        }
        if (n_ids > (size_t)INT_MAX) {
            fprintf(stderr, "vtm run: --text дал слишком много токенов\n");
            free(prompt);
            vtm_tok_free_ids(ids);
            vtm_tok_close(tokenizer);
            return 2;
        }
        tokens = (int *)malloc(n_ids * sizeof(int));
        if (!tokens) {
            fprintf(stderr, "vtm run: не хватило памяти\n");
            free(prompt);
            vtm_tok_free_ids(ids);
            vtm_tok_close(tokenizer);
            return 1;
        }
        for (i = 0; i < (int)n_ids; i++) tokens[i] = (int)ids[i];
        n_tokens = n_ids;
        vtm_tok_free_ids(ids);
    }

    m = vtm_load(path);
    if (!m) {
        fprintf(stderr, "vtm: %s\n", vtm_last_error());
        free(prompt);
        free(tokens);
        vtm_tok_close(tokenizer);
        return 1;
    }
    if (backend_name)
        vtm_set_backend(m, vtm_backend_by_name(backend_name));

    // В режиме --stream stdout отдан только тексту ответа (его читают голос и
    // SSE), поэтому вся служебная информация уходит в stderr. Без --stream
    // вывод прежний — в stdout.
    {
        FILE *info = stream ? stderr : stdout;
        fprintf(info, "модель: %s\n", path);
        fprintf(info, "тензоров: %u, инструкций: %u, бэкенд: %s\n",
                (unsigned)vtm_tensor_count(m), (unsigned)vtm_instr_count(m),
                vtm_backend_name(m));
        if (system_s) fprintf(info, "система: %s\n", system_s);
        if (text_s) fprintf(info, "текст:   %s\n", text_s);
        if (prompt) {
            fprintf(info, "шаблон:\n%sконец шаблона\n", prompt);
            free(prompt);
            prompt = NULL;
        }
    }

    if (max_new > 0) {
        generated = (int *)calloc((size_t)max_new, sizeof(int));
        if (!generated) {
            fprintf(stderr, "vtm run: не хватило памяти\n");
            vtm_free(m);
            free(tokens);
            vtm_tok_close(tokenizer);
            return 1;
        }
    }

    memset(&ss, 0, sizeof ss);
    ss.stop_id = -1;
    if (stream) {
        tok_stream = vtm_tok_stream_open(tokenizer);
        if (!tok_stream) {
            fprintf(stderr, "vtm run: не удалось создать стриминговый "
                            "декодер\n");
            vtm_free(m);
            free(tokens);
            free(generated);
            vtm_tok_close(tokenizer);
            return 1;
        }
        ss.stream = tok_stream;
    }

    c0 = clock();
    if (max_new == 0) {
        for (i = 0; i < (int)n_tokens; i++) {
            vtm_status st;
            vtm_set_position(m, i);
            vtm_set_token(m, tokens[i]);
            st = vtm_run(m, NULL, NULL);
            if (st != VTM_OK) {
                fprintf(stderr, "vtm run: %s: %s\n", vtm_strerror(st),
                        vtm_error_message(m));
                vtm_tok_stream_close(tok_stream);
                vtm_free(m);
                free(tokens);
                free(generated);
                vtm_tok_close(tokenizer);
                return 1;
            }
        }
    } else {
        vtm_status st;
        emit_state es;
        int last;

        // Все токены промпта, кроме последнего, только наполняют KV-кэш.
        for (i = 0; i + 1 < (int)n_tokens; i++) {
            vtm_set_position(m, i);
            vtm_set_token(m, tokens[i]);
            st = vtm_run(m, NULL, NULL);
            if (st != VTM_OK) {
                fprintf(stderr, "vtm run: %s: %s\n", vtm_strerror(st),
                        vtm_error_message(m));
                vtm_tok_stream_close(tok_stream);
                vtm_free(m);
                free(tokens);
                free(generated);
                vtm_tok_close(tokenizer);
                return 1;
            }
        }
        // Последний токен промпта даёт первый сгенерированный.
        memset(&es, 0, sizeof es);
        vtm_set_position(m, (int)n_tokens - 1);
        vtm_set_token(m, tokens[n_tokens - 1]);
        st = vtm_run(m, emit_first, &es);
        if (st != VTM_OK) {
            fprintf(stderr, "vtm run: %s: %s\n", vtm_strerror(st),
                    vtm_error_message(m));
            vtm_tok_stream_close(tok_stream);
            vtm_free(m);
            free(tokens);
            free(generated);
            vtm_tok_close(tokenizer);
            return 1;
        }
        if (es.count > 1)
            fprintf(stderr, "vtm run: предупреждение: SAMP сработал %d раз, "
                            "CLI берёт первый токен\n", es.count);
        if (es.have) {
            generated[n_generated++] = es.token;
            last = es.token;
            if (stream && stream_token(&ss, es.token, tok_err,
                                       sizeof tok_err) != 0) {
                fprintf(stderr, "vtm run: --stream: %s\n", tok_err);
                vtm_tok_stream_close(tok_stream);
                vtm_free(m);
                free(tokens);
                free(generated);
                vtm_tok_close(tokenizer);
                return 1;
            }
        } else {
            fprintf(stderr, "vtm run: программа не вызвала SAMP; "
                            "генерировать нечего\n");
            vtm_tok_stream_close(tok_stream);
            vtm_free(m);
            free(tokens);
            free(generated);
            vtm_tok_close(tokenizer);
            return 1;
        }

        while (!ss.stopped && n_generated < (size_t)max_new) {
            memset(&es, 0, sizeof es);
            // Входной токен previous стоит на позиции len(prompt)+step-1.
            vtm_set_position(m, (int)(n_tokens + n_generated - 1));
            vtm_set_token(m, last);
            st = vtm_run(m, emit_first, &es);
            if (st != VTM_OK) {
                fprintf(stderr, "vtm run: %s: %s\n", vtm_strerror(st),
                        vtm_error_message(m));
                vtm_tok_stream_close(tok_stream);
                vtm_free(m);
                free(tokens);
                free(generated);
                vtm_tok_close(tokenizer);
                return 1;
            }
            if (!es.have) break;
            if (es.count > 1)
                fprintf(stderr, "vtm run: предупреждение: SAMP сработал %d раз, "
                                "CLI берёт первый токен\n", es.count);
            generated[n_generated++] = es.token;
            last = es.token;
            if (stream && stream_token(&ss, es.token, tok_err,
                                       sizeof tok_err) != 0) {
                fprintf(stderr, "vtm run: --stream: %s\n", tok_err);
                vtm_tok_stream_close(tok_stream);
                vtm_free(m);
                free(tokens);
                free(generated);
                vtm_tok_close(tokenizer);
                return 1;
            }
        }
    }
    c1 = clock();

    // Конец генерации: отдать недописанный хвост символа, если он остался.
    if (stream) {
        char *tail = NULL;
        size_t len = 0;
        if (vtm_tok_stream_finish(tok_stream, &tail, tok_err,
                                  sizeof tok_err) != 0) {
            fprintf(stderr, "vtm run: --stream: %s\n", tok_err);
            vtm_tok_stream_close(tok_stream);
            vtm_free(m);
            free(tokens);
            free(generated);
            vtm_tok_close(tokenizer);
            return 1;
        }
        if (tail != NULL) len = strlen(tail);
        if (len != 0 && fwrite(tail, 1, len, stdout) != len) {
            fprintf(stderr, "vtm run: --stream: не удалось записать текст "
                            "в stdout\n");
            vtm_tok_free_text(tail);
            vtm_tok_stream_close(tok_stream);
            vtm_free(m);
            free(tokens);
            free(generated);
            vtm_tok_close(tokenizer);
            return 1;
        }
        if (len != 0) fflush(stdout);
        vtm_tok_free_text(tail);
        vtm_tok_stream_close(tok_stream);
        tok_stream = NULL;
    }

    if (text_s && !stream) {
        int32_t *gen_ids = NULL;
        if (n_generated != 0) {
            gen_ids = (int32_t *)malloc(n_generated * sizeof(int32_t));
            if (!gen_ids) {
                fprintf(stderr, "vtm run: не хватило памяти\n");
                vtm_free(m);
                free(tokens);
                free(generated);
                vtm_tok_close(tokenizer);
                return 1;
            }
            for (i = 0; i < (int)n_generated; i++)
                gen_ids[i] = (int32_t)generated[i];
        }
        if (vtm_tok_decode(tokenizer, gen_ids, n_generated, &answer, tok_err,
                           sizeof tok_err) != 0) {
            fprintf(stderr, "vtm run: не удалось декодировать вывод: %s\n",
                    tok_err);
            free(gen_ids);
            vtm_free(m);
            free(tokens);
            free(generated);
            vtm_tok_close(tokenizer);
            return 1;
        }
        free(gen_ids);
    }

    {
        double sec = (double)(c1 - c0) / (double)CLOCKS_PER_SEC;
        FILE *info = stream ? stderr : stdout;
        print_tokens(info, "prompt: ", tokens, n_tokens);
        if (n_generated)
            print_tokens(info, "вывод:  ", generated, n_generated);
        else
            fprintf(info, "вывод:  (пусто)\n");
        if (stream) {
            if (ss.stopped)
                fprintf(info, "стоп-токен: %d\n", (int)ss.stop_id);
        } else if (text_s) {
            if (answer && *answer)
                fprintf(info, "ответ:   %s\n", answer);
            else
                fprintf(info, "ответ:   (пусто)\n");
        }
        fprintf(info, "время: %.3f мс", sec * 1000.0);
        if (n_generated)
            fprintf(info, ", %zu токен(ов), %.1f ток/с", n_generated,
                    sec > 0.0 ? (double)n_generated / sec : 0.0);
        fprintf(info, "\n");
    }

    vtm_free(m);
    free(tokens);
    free(generated);
    vtm_tok_close(tokenizer);
    vtm_tok_free_text(answer);
    return 0;
}

// --- inspect ---------------------------------------------------------------

static int cmd_inspect(int argc, char **argv) {
    const char *path;
    vtm_t *m;
    uint32_t i;

    if (argc != 1) {
        fprintf(stderr, "vtm inspect: нужен ровно один файл\n");
        return 2;
    }
    path = argv[0];
    m = vtm_load(path);
    if (!m) {
        fprintf(stderr, "vtm: %s\n", vtm_last_error());
        return 1;
    }
    printf("файл: %s\n", path);
    printf("тензоров: %u, инструкций: %u, бэкенд: %s\n",
           (unsigned)vtm_tensor_count(m), (unsigned)vtm_instr_count(m),
           vtm_backend_name(m));
    for (i = 0; i < vtm_tensor_count(m); i++) {
        const vtm_tensor *t = vtm_tensor_by_index(m, i);
        int d;
        printf("[%u] %-23s kind=%-4s ndim=%u dims=[", i, t->name,
               kind_name(t->kind), (unsigned)t->ndim);
        for (d = 0; d < (int)t->ndim; d++)
            printf("%s%u", d ? "," : "", (unsigned)t->dims[d]);
        printf("] elems=%u offset=%u", t->n_elements, t->offset);
        if (t->kind == VTM_TENSOR_TRIT)
            printf(" blocks=%u scales=%u block=%u scale=%s", t->n_blocks,
                   t->n_scales, t->scale_block,
                   t->scale_is_f32 ? "f32" : "f16");
        printf("\n");
    }
    for (i = 0; i < vtm_instr_count(m); i++) {
        const vtm_instr *in = vtm_instruction(m, i);
        printf("%04u  %-4s flags=0x%02x dst=%u a=%u b=%u c=%u imm=%d\n",
               i, vtm_opcode_name(in->opcode), (unsigned)in->flags,
               (unsigned)in->dst, (unsigned)in->a, (unsigned)in->b,
               (unsigned)in->c, in->imm);
    }
    vtm_free(m);
    return 0;
}

// --- bench -----------------------------------------------------------------

static uint32_t rng_state = 0x9E3779B9u;

static uint32_t rng_next(void) {
    uint32_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng_state = x;
    return x;
}

static int cmd_bench(int argc, char **argv) {
    int rows = 512, cols = 512, iters = 200;
    int scale_block = (int)VTM_SCALE_BLOCK_160;
    int i;
    uint32_t n, storage_blocks, scale_blocks, n_words, w, b;
    int8_t *trits;
    uint32_t *words;
    float *scales, *x, *y_lut, *y_direct;
    clock_t c0, c1;
    double t_lut, t_direct, speedup;
    double max_diff = 0.0;
    uint64_t total_elems;
    volatile float sink = 0.0f;

    for (i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--rows") == 0) {
            if (++i >= argc || parse_int(argv[i], &rows) != 0 || rows <= 0) {
                fprintf(stderr, "vtm bench: плохие --rows\n");
                return 2;
            }
        } else if (strcmp(argv[i], "--cols") == 0) {
            if (++i >= argc || parse_int(argv[i], &cols) != 0 || cols <= 0) {
                fprintf(stderr, "vtm bench: плохие --cols\n");
                return 2;
            }
        } else if (strcmp(argv[i], "--iters") == 0) {
            if (++i >= argc || parse_int(argv[i], &iters) != 0 ||
                iters <= 0 || iters > 100000) {
                fprintf(stderr, "vtm bench: плохие --iters\n");
                return 2;
            }
        } else if (strcmp(argv[i], "--scale-block") == 0) {
            if (++i >= argc || parse_int(argv[i], &scale_block) != 0 ||
                (scale_block != 128 && scale_block != 160)) {
                fprintf(stderr, "vtm bench: --scale-block 128|160\n");
                return 2;
            }
        } else {
            fprintf(stderr, "vtm bench: неизвестный аргумент %s\n", argv[i]);
            return 2;
        }
    }
    total_elems = (uint64_t)(uint32_t)rows * (uint64_t)(uint32_t)cols;
    if (total_elems == 0 || total_elems > VTM_MAX_ELEMENTS) {
        fprintf(stderr, "vtm bench: слишком большая матрица\n");
        return 2;
    }
    n = (uint32_t)total_elems;
    storage_blocks = (n + VTM_TRITS_PER_BLOCK - 1u) / VTM_TRITS_PER_BLOCK;
    scale_blocks = (n + (uint32_t)scale_block - 1u) / (uint32_t)scale_block;
    n_words = storage_blocks * VTM_WORDS_PER_BLOCK;

    trits = (int8_t *)calloc((size_t)storage_blocks * VTM_TRITS_PER_BLOCK, 1);
    words = (uint32_t *)malloc((size_t)n_words * sizeof(uint32_t));
    scales = (float *)malloc((size_t)scale_blocks * sizeof(float));
    x = (float *)malloc((size_t)cols * sizeof(float));
    y_lut = (float *)calloc((size_t)rows, sizeof(float));
    y_direct = (float *)calloc((size_t)rows, sizeof(float));
    if (!trits || !words || !scales || !x || !y_lut || !y_direct) {
        fprintf(stderr, "vtm bench: не хватило памяти\n");
        free(trits); free(words); free(scales); free(x);
        free(y_lut); free(y_direct);
        return 1;
    }
    for (w = 0; w < n; w++)
        trits[w] = (int8_t)((int)(rng_next() % 3u) - 1);
    for (w = 0; w < n_words; w++)
        words[w] = vtm_trit_pack(trits + (size_t)w * VTM_TRITS_PER_WORD,
                                 VTM_TRITS_PER_WORD);
    for (b = 0; b < scale_blocks; b++)
        scales[b] = 0.5f + (float)(rng_next() % 1000u) / 1000.0f;
    for (i = 0; i < cols; i++)
        x[i] = (float)((int)(rng_next() % 2001u) - 1000) / 100.0f;

    // Прогрев и проверка совпадения двух алгоритмов.
    vtm_backend_scalar_lut.tmv_blocks(y_lut, (const uint8_t *)words, scales, x,
                                      rows, cols, scale_block);
    vtm_backend_scalar_direct.tmv_blocks(y_direct, (const uint8_t *)words, scales,
                                         x, rows, cols, scale_block);
    for (i = 0; i < rows; i++) {
        double diff = fabs((double)y_lut[i] - (double)y_direct[i]);
        if (diff > max_diff) max_diff = diff;
    }
    sink += y_lut[0] + y_direct[0];

    c0 = clock();
    for (i = 0; i < iters; i++) {
        vtm_backend_scalar_lut.tmv_blocks(y_lut, (const uint8_t *)words, scales,
                                          x, rows, cols, scale_block);
        sink += y_lut[0];
    }
    c1 = clock();
    t_lut = (double)(c1 - c0) / (double)CLOCKS_PER_SEC;

    c0 = clock();
    for (i = 0; i < iters; i++) {
        vtm_backend_scalar_direct.tmv_blocks(y_direct, (const uint8_t *)words,
                                             scales, x, rows, cols, scale_block);
        sink += y_direct[0];
    }
    c1 = clock();
    t_direct = (double)(c1 - c0) / (double)CLOCKS_PER_SEC;
    (void)sink;

    printf("TMV, матрица %d×%d (%u весов), блок масштаба %d, "
           "%d итераций, скалярный C\n",
           rows, cols, n, scale_block, iters);
    printf("  LUT243: %8.3f мс/итерацию  %8.1f Мтрит/с  %6.2f ГБ/с чтения\n",
           t_lut * 1e3 / (double)iters,
           (double)n * (double)iters / (t_lut > 0.0 ? t_lut : 1.0) / 1e6,
           (double)storage_blocks * 32.0 * (double)iters
               / (t_lut > 0.0 ? t_lut : 1.0) / 1e9);
    printf("  direct: %8.3f мс/итерацию  %8.1f Мтрит/с  %6.2f ГБ/с чтения\n",
           t_direct * 1e3 / (double)iters,
           (double)n * (double)iters / (t_direct > 0.0 ? t_direct : 1.0) / 1e6,
           (double)storage_blocks * 32.0 * (double)iters
               / (t_direct > 0.0 ? t_direct : 1.0) / 1e9);
    if (t_lut <= 0.0 || t_direct <= 0.0) {
        printf("  (время меньше разрешения clock(); увеличьте --iters)\n");
    } else if (t_lut < t_direct) {
        speedup = t_direct / t_lut;
        printf("  быстрее LUT243 в %.2fx\n", speedup);
    } else {
        speedup = t_lut / t_direct;
        printf("  быстрее direct в %.2fx\n", speedup);
    }
    printf("  расхождение LUT/direct: %.3g (должно быть ~0)\n", max_diff);

    free(trits); free(words); free(scales); free(x);
    free(y_lut); free(y_direct);
    return 0;
}

// --- main ------------------------------------------------------------------

int main(int argc, char **argv) {
    if (argc < 2) {
        usage(stdout);
        return 2;
    }
    if (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
        usage(stdout);
        return 0;
    }
    if (strcmp(argv[1], "run") == 0) return cmd_run(argc - 2, argv + 2);
    if (strcmp(argv[1], "inspect") == 0) return cmd_inspect(argc - 2, argv + 2);
    if (strcmp(argv[1], "bench") == 0) return cmd_bench(argc - 2, argv + 2);
    fprintf(stderr, "vtm: неизвестная команда %s\n\n", argv[1]);
    usage(stderr);
    return 2;
}

#endif // VTM_TOK_BRIDGE
