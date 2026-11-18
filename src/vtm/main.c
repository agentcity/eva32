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

// --- батчевый prefill (задачи 0112, 0114) ----------------------------------
//
// vtm_run_batch() — публичный API ядра из задачи 0112 (объявляется в vtm.h).
// Здесь он объявлен и определён слабо: пока ядра нет, работает заглушка «по
// токену» (ровно прежний цикл CLI), а когда interp.c получит сильное
// определение, линковщик возьмёт его вместо заглушки — main.c менять не надо.
//
// Слабое определение, а не только слабое объявление: в Mach-O (macOS)
// неразрешённая слабая ссылка — ошибка линковки, а слабое определение
// линкуется везде, и сильное определение из libvtm.a его перекрывает.
//
// Контракт (его должен реализовать 0112): vtm_run_batch(m, ids, n) прогоняет
// ids[0..n) начиная с текущей позиции машины; токен k считается на позиции
// start + k (start = vtm_position(m) до вызова); emit не вызывается; на
// выходе состояние машины и vtm_position(m) == start + n - 1 побитово как
// после n вызовов vtm_run() по одному. n == 0 — no-op, VTM_OK.
//
// Компилятор без weak-атрибута собирает заглушку как обычную функцию — батч
// тогда просто не ускоряет. Другая сигнатура в 0112 — ошибка компиляции здесь
// (а не молчаливая работа заглушки).
#if defined(__GNUC__) || defined(__clang__)
#define VTM_WEAK __attribute__((weak))
#else
#define VTM_WEAK
#endif

vtm_status vtm_run_batch(vtm_t *m, const int32_t *ids, int n);
VTM_WEAK
vtm_status vtm_run_batch(vtm_t *m, const int32_t *ids, int n) {
    const int64_t start = (int64_t)vtm_position(m);
    int i;

    for (i = 0; i < n; i++) {
        const int64_t pos = start + i;
        vtm_status st;
        if (pos > (int64_t)INT_MAX) return VTM_E_BOUNDS;
        vtm_set_position(m, (int)pos);
        vtm_set_token(m, (int)ids[i]);
        st = vtm_run(m, NULL, NULL);
        if (st != VTM_OK) return st;
    }
    if (n > 0) vtm_set_position(m, (int)(start + n - 1));
    return VTM_OK;
}

static void usage(FILE *out) {
    fprintf(out,
        "VTM — скалярный интерпретатор виртуальной троичной машины (v0.1)\n"
        "\n"
        "  vtm run model.vtm --tokens 1,2,3 --max-new 8\n"
        "      прогнать программу по токенам и напечатать выбранные токены\n"
        "      --backend auto|simd|scalar|lut|direct  бэкенд (по умолчанию VTM_BACKEND или auto)\n"
        "\n"
        "  vtm run model.vtm --text \"вопрос\" [--tokenizer PATH] [--max-new 8]\n"
        "      токенизировать текст (BPE), прогнать и напечатать ответ текстом\n"
        "      --tokenizer PATH       tokenizer.json; иначе $VTM_TOKENIZER,\n"
        "                             затем tests/vtm/fixtures/tokenizer.json\n"
        "      --chat                 обернуть текст в чат-шаблон Qwen2.5\n"
        "                             (<|im_start|>system/user/assistant <|im_end|>)\n"
        "      --system \"...\"         системное сообщение; только вместе с --chat\n"
        "      --stop \"текст\"         остановиться, когда декодированный текст\n"
        "                             кончается на эту строку; сама строка в вывод\n"
        "                             не попадает. Можно повторять; в строке\n"
        "                             понимаются \\n, \\t, \\r и \\\\.\n"
        "      --timeout SEC          прервать генерацию по таймауту: код 124,\n"
        "                             уже сгенерированное печатается.\n"
        "      --stats                статистика в stderr: токены, время, ток/с.\n"
        "      --trace                трасса в stderr: строка на инструкцию\n"
        "                             (pc, opcode, номер, выход, время в нс)\n"
        "      --trace-summary        сводка в stderr: по опкоду вызовы, время,\n"
        "                             доля; время — реальный замер (0082)\n"
        "      --max-tokens N         синоним --max-new.\n"
        "      --prefill-batch N      считать промпт пачками по N токенов:\n"
        "                             быстрее на длинном промпте, ответ тот же;\n"
        "                             0 — по токену, как раньше (по умолчанию 0).\n"
        "      --stream               печатать ответ по мере генерации (stdout,\n"
        "                             без буферизации строки; только с --text).\n"
        "                             Декодер ждёт завершения UTF-8 символа и\n"
        "                             не печатает спецтокены; на стоп-токене\n"
        "                             генерация обрывается. Служебные строки —\n"
        "                             в stderr.\n"
        "      --backend auto|simd|scalar|lut|direct  бэкенд (по умолчанию VTM_BACKEND или auto)\n"
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
        "или токенизатора, 124 — таймаут генерации (--timeout).\n");
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

static int parse_double(const char *s, double *out) {
    char *end = NULL;
    double v;
    if (!s || !*s) return -1;
    errno = 0;
    v = strtod(s, &end);
    if (errno != 0 || end == s || *end != '\0' || !isfinite(v)) return -1;
    *out = v;
    return 0;
}

// --- --stop: текстовые стоп-последовательности -----------------------------
//
// Строка из argv разворачивается как в C-литерале для \n, \t, \r и \\:
// из shell удобнее писать --stop "\n\n" (обратный слэш и n), чем вставлять
// настоящий перевод строки. Неизвестная escape-пара остаётся как есть
// (важно для путей с \).
typedef struct {
    char *text;
    size_t len;
} stop_text;

// malloc-строка; NULL — не хватило памяти.
static char *stop_unescape(const char *src) {
    size_t n = strlen(src);
    char *out = (char *)malloc(n + 1);
    size_t i = 0, o = 0;

    if (!out) return NULL;
    while (i < n) {
        if (src[i] == '\\' && i + 1 < n) {
            char c = src[i + 1];
            if (c == 'n') {
                out[o++] = '\n';
                i += 2;
                continue;
            }
            if (c == 't') {
                out[o++] = '\t';
                i += 2;
                continue;
            }
            if (c == 'r') {
                out[o++] = '\r';
                i += 2;
                continue;
            }
            if (c == '\\') {
                out[o++] = '\\';
                i += 2;
                continue;
            }
        }
        out[o++] = src[i++];
    }
    out[o] = '\0';
    return out;
}

static void stop_texts_free(stop_text *stops, size_t n) {
    size_t i;
    if (!stops) return;
    for (i = 0; i < n; i++) free(stops[i].text);
    free(stops);
}

// 0 — добавлено; -1 — пустая строка; -2 — не хватило памяти.
static int stop_texts_add(stop_text **stops, size_t *n, size_t *max_len,
                          const char *raw) {
    char *text = stop_unescape(raw);
    stop_text *p;

    if (!text) return -2;
    if (text[0] == '\0') {
        free(text);
        return -1;
    }
    p = (stop_text *)realloc(*stops, (*n + 1) * sizeof(stop_text));
    if (!p) {
        free(text);
        return -2;
    }
    *stops = p;
    p[*n].text = text;
    p[*n].len = strlen(text);
    if (p[*n].len > *max_len) *max_len = p[*n].len;
    (*n)++;
    return 0;
}

static double monotonic_now(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0)
        return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

// --- трассировка по инструкциям (--trace / --trace-summary, задача 0082) ---
//
// Трасса — диагностика, поэтому идёт в stderr (как --stats): stdout остаётся
// разбору «prompt:/вывод:/время:». Строка --trace начинается с имени opcode и
// заканчивается длительностью в нс — тот же вид, что у строк --trace-summary,
// поэтому tools/bench_vtm.py разбирает оба режима одним парсером.

#define TRACE_OP_SLOTS 256

typedef struct trace_state {
    int lines;                  // 1 — печатать каждую инструкцию (--trace)
    int summary;                // 1 — сводка в конце (--trace-summary)
    uint64_t events;            // сколько инструкций прошло через callback
    uint64_t events_ns;         // сумма их длительностей
    uint64_t per_op_ns[TRACE_OP_SLOTS];
    uint64_t per_op_count[TRACE_OP_SLOTS];
    uint64_t per_op_bytes[TRACE_OP_SLOTS];
} trace_state;

static int trace_cli_cb(void *user, const vtm_trace_rec *rec) {
    trace_state *ts = (trace_state *)user;

    if (!ts || !rec) return 0;
    ts->events++;
    ts->events_ns += rec->dur_ns;
    if ((unsigned)rec->opcode < TRACE_OP_SLOTS) {
        ts->per_op_ns[rec->opcode] += rec->dur_ns;
        ts->per_op_count[rec->opcode]++;
        ts->per_op_bytes[rec->opcode] += rec->out_bytes;
    }
    if (ts->lines) {
        fprintf(stderr, "%s pc=%u seq=%u out=%u dur=%.1f ns\n",
                vtm_opcode_name(rec->opcode), (unsigned)rec->pc,
                (unsigned)rec->index, (unsigned)rec->out_elems,
                (double)rec->dur_ns);
    }
    return 0;
}

// Сводка: опкод -> вызовов, суммарное время, доля от суммы времён инструкций.
// Строки начинаются с имени opcode (машинный разбор), шапка — кириллицей.
static void trace_print_summary(const trace_state *ts, double run_ms) {
    unsigned op;
    double total_ns = (double)ts->events_ns;

    fprintf(stderr, "trace-summary: инструкций=%llu, время в инструкциях="
                    "%.6f мс",
            (unsigned long long)ts->events, total_ns / 1e6);
    if (run_ms > 0.0 && total_ns > 0.0)
        fprintf(stderr, " (%.1f%% времени прогона %.3f мс по монотонным часам)",
                100.0 * (total_ns / 1e6) / run_ms, run_ms);
    fprintf(stderr, "\n");
    fprintf(stderr, "опкод   вызовов     время, мс      доля    выход, КиБ\n");
    for (op = 0; op < TRACE_OP_SLOTS; op++) {
        if (ts->per_op_count[op] == 0) continue;
        fprintf(stderr, "%-6s %9llu %12.6f ms %7.2f%% %12.1f\n",
                vtm_opcode_name((uint8_t)op),
                (unsigned long long)ts->per_op_count[op],
                (double)ts->per_op_ns[op] / 1e6,
                total_ns > 0.0 ? 100.0 * (double)ts->per_op_ns[op] / total_ns
                               : 0.0,
                (double)ts->per_op_bytes[op] / 1024.0);
    }
    fprintf(stderr, "итого  %9llu %12.6f мс %7.2f%%\n",
            (unsigned long long)ts->events, total_ns / 1e6,
            total_ns > 0.0 ? 100.0 : 0.0);
}

static int timeout_expired(double started, double timeout_sec) {
    return timeout_sec > 0.0 && monotonic_now() - started >= timeout_sec;
}

// Прогнать токены промпта [from, to) — пачками по batch через vtm_run_batch()
// (задачи 0112, 0114) или по токену (batch == 0, прежний путь).
//
// emit здесь не нужен: эти токены только наполняют KV-кэш (последний токен
// промпта прогоняется отдельно, на нём SAMP выбирает первый ответ).
// Пачка неделима, поэтому таймаут проверяется между пачками (как у
// vtm_run_cached() в C-ABI), а не между токенами. *timed_out = 1 — таймаут
// сработал, прогон прекращён; возврат VTM_OK.
// timeout_sec == 0 — таймаута нет (режим --max-new 0 его и раньше не смотрел).
static vtm_status prompt_prefill(vtm_t *m, const int *tokens, int from, int to,
                                 int batch, double t_start, double timeout_sec,
                                 int *timed_out) {
    int i = from;

    while (i < to) {
        vtm_status st;
        int len = to - i;

        if (timeout_expired(t_start, timeout_sec)) {
            *timed_out = 1;
            return VTM_OK;
        }
        if (batch > 0) {
            if (len > batch) len = batch;
            vtm_set_position(m, i);
            // tokens — int: те же 32 бита и представление, что у int32_t
            // (id токена). C-ABI ядра принимает int32_t.
            st = vtm_run_batch(m, (const int32_t *)(const void *)(tokens + i),
                               len);
        } else {
            len = 1;
            vtm_set_position(m, i);
            vtm_set_token(m, tokens[i]);
            st = vtm_run(m, NULL, NULL);
        }
        if (st != VTM_OK) return st;
        i += len;
    }
    return VTM_OK;
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
//
// --stop: декодированный текст проверяется на «кончается на строку» после
// каждого токена. Хвост, который ещё может оказаться началом стоп-строки,
// придерживается в hold и не печатается, пока не станет ясно, что стоп не
// будет собран из нескольких токенов. Так --stop работает и со --stream, и
// без него (тогда hold — это ещё и текст для строки «ответ:»).
typedef struct {
    vtm_tok_stream *stream;
    int stopped;              // остановить генерацию
    int32_t stop_id;          // стоп-токен, если сработал он
    int text_stopped;         // сработала одна из --stop
    int no_print;             // без --stream: hold не печатаем, копим ответ
    const stop_text *stops;
    size_t n_stops;
    size_t max_stop_len;
    char *hold;
    size_t hold_len;
    size_t hold_cap;
} stream_state;

static void stream_state_setup(stream_state *ss, vtm_tok_stream *stream,
                               const stop_text *stops, size_t n_stops,
                               size_t max_stop_len, int no_print) {
    memset(ss, 0, sizeof *ss);
    ss->stream = stream;
    ss->stop_id = -1;
    ss->stops = stops;
    ss->n_stops = n_stops;
    ss->max_stop_len = max_stop_len;
    ss->no_print = no_print;
}

static void stream_state_free(stream_state *ss) {
    free(ss->hold);
    ss->hold = NULL;
    ss->hold_len = 0;
    ss->hold_cap = 0;
}

static int write_stdout_text(const char *text, size_t len, char *err,
                             size_t errlen) {
    if (len != 0 && fwrite(text, 1, len, stdout) != len) {
        snprintf(err, errlen, "не удалось записать текст в stdout");
        return -1;
    }
    if (len != 0) fflush(stdout);
    return 0;
}

static int hold_append(stream_state *ss, const char *text, size_t len,
                       char *err, size_t errlen) {
    if (len == 0) return 0;
    if (ss->hold_len + len + 1 > ss->hold_cap) {
        size_t cap = ss->hold_cap ? ss->hold_cap : 256;
        char *p;
        while (cap < ss->hold_len + len + 1) {
            if (cap > (size_t)-1 / 2) {
                snprintf(err, errlen, "не хватило памяти под --stop");
                return -1;
            }
            cap *= 2;
        }
        p = (char *)realloc(ss->hold, cap);
        if (!p) {
            snprintf(err, errlen, "не хватило памяти под --stop");
            return -1;
        }
        ss->hold = p;
        ss->hold_cap = cap;
    }
    memcpy(ss->hold + ss->hold_len, text, len);
    ss->hold_len += len;
    ss->hold[ss->hold_len] = '\0';
    return 0;
}

// Длина самой длинной стоп-строки, которой кончается hold (0 — нет).
static size_t matching_stop_len(const stream_state *ss) {
    size_t best = 0;
    size_t i;
    for (i = 0; i < ss->n_stops; i++) {
        const size_t len = ss->stops[i].len;
        if (len <= best || len > ss->hold_len) continue;
        if (memcmp(ss->hold + ss->hold_len - len, ss->stops[i].text, len) == 0)
            best = len;
    }
    return best;
}

// 0 — ок, -1 — ошибка записи/памяти.
static int stream_emit(stream_state *ss, const char *text, size_t len,
                       char *err, size_t errlen) {
    if (len == 0 || ss->text_stopped) return 0;
    if (ss->n_stops == 0) return write_stdout_text(text, len, err, errlen);

    if (hold_append(ss, text, len, err, errlen) != 0) return -1;
    {
        const size_t m = matching_stop_len(ss);
        if (m != 0) {
            // Стоп-строка стоит в конце hold; начало hold — обычный текст.
            const size_t cut = ss->hold_len - m;
            ss->text_stopped = 1;
            ss->stopped = 1;
            if (ss->no_print) {
                ss->hold_len = cut;
                ss->hold[cut] = '\0';
            } else {
                if (write_stdout_text(ss->hold, cut, err, errlen) != 0)
                    return -1;
                ss->hold_len = 0;
                ss->hold[0] = '\0';
            }
            return 0;
        }
    }
    if (ss->no_print) return 0;

    // Печатаем всё, кроме хвоста, из которого ещё может вырасти стоп-строка.
    // Начало хвоста сдвигаем назад до границы UTF-8: наружу отдаём только
    // целые символы.
    {
        const size_t keep = ss->max_stop_len > 1 ? ss->max_stop_len - 1 : 0;
        if (ss->hold_len > keep) {
            size_t start = ss->hold_len - keep;
            while (start > 0 &&
                   ((unsigned char)ss->hold[start] & 0xC0) == 0x80)
                start--;
            if (start != 0) {
                if (write_stdout_text(ss->hold, start, err, errlen) != 0)
                    return -1;
                memmove(ss->hold, ss->hold + start, ss->hold_len - start);
                ss->hold_len -= start;
                ss->hold[ss->hold_len] = '\0';
            }
        }
    }
    return 0;
}

// Дописать в stdout хвост, который больше не может быть началом стопа.
static int stream_flush_hold(stream_state *ss, char *err, size_t errlen) {
    if (ss->n_stops == 0 || ss->no_print || ss->text_stopped) return 0;
    if (ss->hold_len != 0) {
        if (write_stdout_text(ss->hold, ss->hold_len, err, errlen) != 0)
            return -1;
        ss->hold_len = 0;
        ss->hold[0] = '\0';
    }
    return 0;
}

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
    rc = stream_emit(ss, piece ? piece : "", len, err, errlen);
    vtm_tok_free_text(piece);
    if (rc != 0) return -1;
    if (stop) {
        ss->stopped = 1;
        ss->stop_id = stop_id;
    }
    return 0;
}

// То же для --stop без --stream: только накопление текста и проверка стопа,
// без печати. Стоп-токен здесь не обрывает генерацию — без новых флагов это
// прежнее поведение, и --stop не должен его менять.
static int stop_feed(stream_state *ss, int token, char *err, size_t errlen) {
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
    rc = stream_emit(ss, piece ? piece : "", len, err, errlen);
    vtm_tok_free_text(piece);
    (void)stop;
    (void)stop_id;
    return rc;
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
    // 0 — промпт по токену (как раньше), N — пачками по N (задачи 0112, 0114).
    int prefill_batch = 0;
    int stats = 0;
    int trace_lines = 0;
    int trace_summary = 0;
    trace_state trace;
    int timed_out = 0;
    double timeout_sec = 0.0;
    double t_start = 0.0, t_end = 0.0;
    stop_text *stops = NULL;
    size_t n_stops = 0;
    size_t max_stop_len = 0;
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
                goto arg_error;
            }
            tokens_s = argv[i];
        } else if (strcmp(argv[i], "--text") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "vtm run: --text без значения\n");
                goto arg_error;
            }
            text_s = argv[i];
        } else if (strcmp(argv[i], "--tokenizer") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "vtm run: --tokenizer без значения\n");
                goto arg_error;
            }
            tokenizer_s = argv[i];
        } else if (strcmp(argv[i], "--chat") == 0) {
            chat = 1;
        } else if (strcmp(argv[i], "--stream") == 0) {
            stream = 1;
        } else if (strcmp(argv[i], "--stats") == 0) {
            stats = 1;
        } else if (strcmp(argv[i], "--trace") == 0) {
            trace_lines = 1;
        } else if (strcmp(argv[i], "--trace-summary") == 0) {
            trace_summary = 1;
        } else if (strcmp(argv[i], "--system") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "vtm run: --system без значения\n");
                goto arg_error;
            }
            system_s = argv[i];
        } else if (strcmp(argv[i], "--stop") == 0) {
            int rc;
            if (++i >= argc) {
                fprintf(stderr, "vtm run: --stop без значения\n");
                goto arg_error;
            }
            rc = stop_texts_add(&stops, &n_stops, &max_stop_len, argv[i]);
            if (rc == -1) {
                fprintf(stderr, "vtm run: пустая строка --stop\n");
                goto arg_error;
            }
            if (rc != 0) {
                fprintf(stderr, "vtm run: не хватило памяти под --stop\n");
                goto arg_error;
            }
        } else if (strcmp(argv[i], "--timeout") == 0) {
            if (++i >= argc || parse_double(argv[i], &timeout_sec) != 0 ||
                timeout_sec <= 0.0) {
                fprintf(stderr,
                        "vtm run: --timeout ожидает положительное число секунд\n");
                goto arg_error;
            }
        } else if (strcmp(argv[i], "--max-new") == 0 ||
                   strcmp(argv[i], "--max-tokens") == 0) {
            if (++i >= argc || parse_int(argv[i], &max_new) != 0 ||
                max_new < 0 || max_new > 1000000) {
                fprintf(stderr, "vtm run: плохое значение --max-new/--max-tokens\n");
                goto arg_error;
            }
        } else if (strcmp(argv[i], "--prefill-batch") == 0) {
            if (++i >= argc || parse_int(argv[i], &prefill_batch) != 0 ||
                prefill_batch < 0 || prefill_batch > 1000000) {
                fprintf(stderr,
                        "vtm run: --prefill-batch ожидает N >= 0 "
                        "(0 — промпт по токену)\n");
                goto arg_error;
            }
        } else if (strcmp(argv[i], "--backend") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "vtm run: --backend без значения\n");
                goto arg_error;
            }
            backend_name = argv[i];
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "vtm run: неизвестный аргумент %s\n", argv[i]);
            goto arg_error;
        } else if (!path) {
            path = argv[i];
        } else {
            fprintf(stderr, "vtm run: лишний аргумент %s\n", argv[i]);
            goto arg_error;
        }
    }
    if (!path || (!tokens_s && !text_s)) {
        fprintf(stderr, "vtm run: нужны model.vtm и --tokens или --text\n");
        goto arg_error;
    }
    if (tokens_s && text_s) {
        fprintf(stderr, "vtm run: --tokens и --text взаимоисключающие\n");
        goto arg_error;
    }
    if (tokenizer_s && !text_s) {
        fprintf(stderr, "vtm run: --tokenizer имеет смысл только с --text\n");
        goto arg_error;
    }
    if (chat && !text_s) {
        fprintf(stderr, "vtm run: --chat имеет смысл только с --text\n");
        goto arg_error;
    }
    if (stream && !text_s) {
        fprintf(stderr, "vtm run: --stream имеет смысл только с --text\n");
        goto arg_error;
    }
    if (n_stops != 0 && !text_s) {
        fprintf(stderr, "vtm run: --stop имеет смысл только с --text\n");
        goto arg_error;
    }
    if (system_s && !chat) {
        fprintf(stderr, "vtm run: --system имеет смысл только с --chat\n");
        goto arg_error;
    }
    if (backend_name && !vtm_backend_by_name(backend_name)) {
        fprintf(stderr, "vtm run: --backend должен быть auto, simd, scalar, "
                        "lut или direct\n");
        goto arg_error;
    }
    goto args_ok;

arg_error:
    stop_texts_free(stops, n_stops);
    return 2;

args_ok:
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
            stop_texts_free(stops, n_stops);
            return 2;
        }
        tokenizer = vtm_tok_open(tok_path, tok_err, sizeof tok_err);
        if (!tokenizer) {
            fprintf(stderr, "vtm run: токенизатор %s: %s\n", tok_path, tok_err);
            stop_texts_free(stops, n_stops);
            return 2;
        }
        if (chat) {
            if (chat_build_prompt(tokenizer, system_s, text_s, &prompt, tok_err,
                                  sizeof tok_err) != 0) {
                fprintf(stderr, "vtm run: чат-шаблон: %s\n", tok_err);
                vtm_tok_close(tokenizer);
                stop_texts_free(stops, n_stops);
                return 2;
            }
        }
        if (vtm_tok_encode(tokenizer, prompt ? prompt : text_s, &ids, &n_ids,
                           tok_err, sizeof tok_err) != 0) {
            fprintf(stderr, "vtm run: токенизатор %s: %s\n", tok_path, tok_err);
            free(prompt);
            vtm_tok_close(tokenizer);
            stop_texts_free(stops, n_stops);
            return 2;
        }
        if (n_ids == 0) {
            fprintf(stderr,
                    "vtm run: --text \"%s\" дал пустой список токенов "
                    "(проверьте tokenizer.json: %s)\n", text_s, tok_path);
            free(prompt);
            vtm_tok_free_ids(ids);
            vtm_tok_close(tokenizer);
            stop_texts_free(stops, n_stops);
            return 2;
        }
        if (n_ids > (size_t)INT_MAX) {
            fprintf(stderr, "vtm run: --text дал слишком много токенов\n");
            free(prompt);
            vtm_tok_free_ids(ids);
            vtm_tok_close(tokenizer);
            stop_texts_free(stops, n_stops);
            return 2;
        }
        tokens = (int *)malloc(n_ids * sizeof(int));
        if (!tokens) {
            fprintf(stderr, "vtm run: не хватило памяти\n");
            free(prompt);
            vtm_tok_free_ids(ids);
            vtm_tok_close(tokenizer);
            stop_texts_free(stops, n_stops);
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
        stop_texts_free(stops, n_stops);
        return 1;
    }
    if (backend_name)
        vtm_set_backend(m, vtm_backend_by_name(backend_name));

    memset(&trace, 0, sizeof trace);
    trace.lines = trace_lines;
    trace.summary = trace_summary;
    if (trace_lines || trace_summary)
        vtm_set_trace(m, trace_cli_cb, &trace);

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
        if (prefill_batch > 0)
            fprintf(info, "prefill: пачками по %d токен(ов)\n", prefill_batch);
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
            stop_texts_free(stops, n_stops);
            return 1;
        }
    }

    stream_state_setup(&ss, NULL, stops, n_stops, max_stop_len, !stream);
    if (stream || n_stops != 0) {
        tok_stream = vtm_tok_stream_open(tokenizer);
        if (!tok_stream) {
            fprintf(stderr, "vtm run: не удалось создать стриминговый "
                            "декодер\n");
            vtm_free(m);
            free(tokens);
            free(generated);
            vtm_tok_close(tokenizer);
            stop_texts_free(stops, n_stops);
            return 1;
        }
        ss.stream = tok_stream;
    }

    c0 = clock();
    t_start = monotonic_now();
    if (max_new == 0) {
        // Здесь таймаут, как и раньше, не проверяется (--timeout действует
        // на генерацию): timeout_sec = 0.
        vtm_status st = prompt_prefill(m, tokens, 0, (int)n_tokens,
                                       prefill_batch, t_start, 0.0, &timed_out);
        if (st != VTM_OK) {
            fprintf(stderr, "vtm run: %s: %s\n", vtm_strerror(st),
                    vtm_error_message(m));
            vtm_tok_stream_close(tok_stream);
            vtm_free(m);
            free(tokens);
            free(generated);
            vtm_tok_close(tokenizer);
            stop_texts_free(stops, n_stops);
            stream_state_free(&ss);
            return 1;
        }
    } else {
        vtm_status st;
        emit_state es;
        int last;

        // Все токены промпта, кроме последнего, только наполняют KV-кэш:
        // пачками по prefill_batch или по токену (prefill_batch == 0).
        st = prompt_prefill(m, tokens, 0, (int)n_tokens - 1, prefill_batch,
                            t_start, timeout_sec, &timed_out);
        if (timed_out) goto generation_done;
        if (st != VTM_OK) {
            fprintf(stderr, "vtm run: %s: %s\n", vtm_strerror(st),
                    vtm_error_message(m));
            vtm_tok_stream_close(tok_stream);
            vtm_free(m);
            free(tokens);
            free(generated);
            vtm_tok_close(tokenizer);
            stop_texts_free(stops, n_stops);
            stream_state_free(&ss);
            return 1;
        }
        if (timeout_expired(t_start, timeout_sec)) {
            timed_out = 1;
            goto generation_done;
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
            stop_texts_free(stops, n_stops);
            stream_state_free(&ss);
            return 1;
        }
        if (es.count > 1)
            fprintf(stderr, "vtm run: предупреждение: SAMP сработал %d раз, "
                            "CLI берёт первый токен\n", es.count);
        if (es.have) {
            generated[n_generated++] = es.token;
            last = es.token;
            if (stream) {
                if (stream_token(&ss, es.token, tok_err,
                                 sizeof tok_err) != 0) {
                    fprintf(stderr, "vtm run: --stream: %s\n", tok_err);
                    vtm_tok_stream_close(tok_stream);
                    vtm_free(m);
                    free(tokens);
                    free(generated);
                    vtm_tok_close(tokenizer);
                    stop_texts_free(stops, n_stops);
                    stream_state_free(&ss);
                    return 1;
                }
            } else if (n_stops != 0) {
                if (stop_feed(&ss, es.token, tok_err, sizeof tok_err) != 0) {
                    fprintf(stderr, "vtm run: --stop: %s\n", tok_err);
                    vtm_tok_stream_close(tok_stream);
                    vtm_free(m);
                    free(tokens);
                    free(generated);
                    vtm_tok_close(tokenizer);
                    stop_texts_free(stops, n_stops);
                    stream_state_free(&ss);
                    return 1;
                }
            }
        } else {
            fprintf(stderr, "vtm run: программа не вызвала SAMP; "
                            "генерировать нечего\n");
            vtm_tok_stream_close(tok_stream);
            vtm_free(m);
            free(tokens);
            free(generated);
            vtm_tok_close(tokenizer);
            stop_texts_free(stops, n_stops);
            stream_state_free(&ss);
            return 1;
        }

        while (!ss.stopped && n_generated < (size_t)max_new) {
            if (timeout_expired(t_start, timeout_sec)) {
                timed_out = 1;
                break;
            }
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
                stop_texts_free(stops, n_stops);
                stream_state_free(&ss);
                return 1;
            }
            if (!es.have) break;
            if (es.count > 1)
                fprintf(stderr, "vtm run: предупреждение: SAMP сработал %d раз, "
                                "CLI берёт первый токен\n", es.count);
            generated[n_generated++] = es.token;
            last = es.token;
            if (stream) {
                if (stream_token(&ss, es.token, tok_err,
                                 sizeof tok_err) != 0) {
                    fprintf(stderr, "vtm run: --stream: %s\n", tok_err);
                    vtm_tok_stream_close(tok_stream);
                    vtm_free(m);
                    free(tokens);
                    free(generated);
                    vtm_tok_close(tokenizer);
                    stop_texts_free(stops, n_stops);
                    stream_state_free(&ss);
                    return 1;
                }
            } else if (n_stops != 0) {
                if (stop_feed(&ss, es.token, tok_err, sizeof tok_err) != 0) {
                    fprintf(stderr, "vtm run: --stop: %s\n", tok_err);
                    vtm_tok_stream_close(tok_stream);
                    vtm_free(m);
                    free(tokens);
                    free(generated);
                    vtm_tok_close(tokenizer);
                    stop_texts_free(stops, n_stops);
                    stream_state_free(&ss);
                    return 1;
                }
            }
        }
    }

generation_done:
    c1 = clock();
    t_end = monotonic_now();

    // Конец генерации: отдать недописанный хвост символа, если он остался.
    // В режиме --stop хвост проходит ту же проверку: стоп-строка может
    // собраться только на finish(), и тогда она тоже не попадёт в stdout.
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
            stop_texts_free(stops, n_stops);
            stream_state_free(&ss);
            return 1;
        }
        if (tail != NULL) len = strlen(tail);
        if (stream_emit(&ss, tail ? tail : "", len, tok_err,
                        sizeof tok_err) != 0) {
            fprintf(stderr, "vtm run: --stream: %s\n", tok_err);
            vtm_tok_free_text(tail);
            vtm_tok_stream_close(tok_stream);
            vtm_free(m);
            free(tokens);
            free(generated);
            vtm_tok_close(tokenizer);
            stop_texts_free(stops, n_stops);
            stream_state_free(&ss);
            return 1;
        }
        vtm_tok_free_text(tail);
        if (stream_flush_hold(&ss, tok_err, sizeof tok_err) != 0) {
            fprintf(stderr, "vtm run: --stream: %s\n", tok_err);
            vtm_tok_stream_close(tok_stream);
            vtm_free(m);
            free(tokens);
            free(generated);
            vtm_tok_close(tokenizer);
            stop_texts_free(stops, n_stops);
            stream_state_free(&ss);
            return 1;
        }
    }
    if (tok_stream != NULL) {
        vtm_tok_stream_close(tok_stream);
        tok_stream = NULL;
    }

    if (text_s && !stream) {
        if (n_stops != 0 && ss.text_stopped) {
            // --stop без --stream: hold уже содержит текст до стоп-строки.
            answer = (char *)malloc(ss.hold_len + 1);
            if (!answer) {
                fprintf(stderr, "vtm run: не хватило памяти\n");
                vtm_free(m);
                free(tokens);
                free(generated);
                vtm_tok_close(tokenizer);
                stop_texts_free(stops, n_stops);
                stream_state_free(&ss);
                return 1;
            }
            if (ss.hold_len != 0) memcpy(answer, ss.hold, ss.hold_len);
            answer[ss.hold_len] = '\0';
        } else {
            int32_t *gen_ids = NULL;
            if (n_generated != 0) {
                gen_ids = (int32_t *)malloc(n_generated * sizeof(int32_t));
                if (!gen_ids) {
                    fprintf(stderr, "vtm run: не хватило памяти\n");
                    vtm_free(m);
                    free(tokens);
                    free(generated);
                    vtm_tok_close(tokenizer);
                    stop_texts_free(stops, n_stops);
                    stream_state_free(&ss);
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
                stop_texts_free(stops, n_stops);
                stream_state_free(&ss);
                return 1;
            }
            free(gen_ids);
        }
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
            if (ss.text_stopped)
                fprintf(info, "стоп: --stop\n");
            else if (ss.stopped)
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

    if (trace_summary)
        trace_print_summary(&trace, (t_end - t_start) * 1000.0);

    if (timed_out)
        fprintf(stderr, "vtm run: таймаут %g с: генерация прервана\n",
                timeout_sec);

    if (stats) {
        const double wall = t_end - t_start;
        fprintf(stderr, "stats: токенов=%zu, время=%.6f с, токенов/с=%.2f\n",
                n_generated, wall,
                wall > 0.0 ? (double)n_generated / wall : 0.0);
    }

    vtm_free(m);
    free(tokens);
    free(generated);
    vtm_tok_close(tokenizer);
    vtm_tok_free_text(answer);
    stop_texts_free(stops, n_stops);
    stream_state_free(&ss);
    return timed_out ? 124 : 0;
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
