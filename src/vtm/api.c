// VTM — C-ABI для хоста: долгоживущий контекст, стриминг, отмена.
//
// Реализация намеренно тонкая: вся семантика исполнения остаётся в
// loader.c/interp.c, здесь только цикл «промпт → токен → колбэк → следующий
// токен» (как в CLI, main.c) плюс атомарный флаг отмены. Никаких потоков
// внутри: vtm_api_cancel() только выставляет флаг, а проверяет его поток
// генерации между вызовами vtm_run().
//
// Сборка: C11 (нужен <stdatomic.h>), но публичный api.h остаётся обычным C99.
#include "api.h"
#include "vtm.h"
#include "vtm_backend.h"

#include <limits.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Внутренние функции kv_store.c (задачи 0079, 0083). Не публичный API:
// объявлены здесь, чтобы не заводить ещё один заголовок в зоне задачи.
int vtm_session_save(const vtm_t *m, int32_t pending_token, const char *path,
                     char *err, size_t errcap);
int vtm_session_load(vtm_t *m, int32_t *pending_token, const char *path,
                     char *err, size_t errcap);
// Уложить каталог сессий в лимит: удалить самые старые по mtime, кроме
// keep_path и незавершённых temp-файлов. Возврат — число удалённых файлов.
int vtm_session_prune(const char *dir, uint64_t limit_bytes,
                      const char *keep_path);

// Внутренние точки входа interp.c: кэш префикса с явным размером пачки
// (задача 0180). В vtm.h не объявлены намеренно — это проводка C-ABI, а
// публичный заголовок ядра задачей не меняется. batch == 0 — прежнее
// поведение (на SIMD побитовый путь по токену), batch > 0 — пачки и на SIMD.
vtm_status vtm_cache_prefix_ex(vtm_t *m, const int32_t *ids, int n, int batch);
vtm_status vtm_run_cached_ex(vtm_t *m, const int32_t *ids, int n, int batch);

// --- батчевый prefill (задачи 0112, 0114) ----------------------------------
//
// vtm_run_batch() — публичный API ядра из задачи 0112 (объявляется в vtm.h).
// Здесь он объявлен и определён слабо: пока 0112 не сдана, работает заглушка
// «по токену» (ровно прежний цикл), а когда interp.c получит сильное
// определение, линковщик возьмёт его вместо заглушки — правок в api.c не
// нужно. Слабое определение, а не только слабое объявление, выбрано
// осознанно: в Mach-O (macOS) неразрешённая слабая ссылка — ошибка линковки,
// а слабое определение собирается и линкуется на всех платформах; сильное
// определение из libvtm.a его перекрывает (проверено: nm показывает
// external, вызов идёт в ядро).
//
// Контракт, на который рассчитан вызов (его и должен реализовать 0112):
//   vtm_run_batch(m, ids, n) прогоняет ids[0..n) начиная с текущей позиции
//   машины: токен k считается на позиции start + k, где start =
//   vtm_position(m) до вызова; emit не вызывается (это prefill, а не
//   генерация); на выходе состояние машины (K/V, счётчик, сэмплер) и
//   vtm_position(m) == start + n - 1 — побитово как после n вызовов
//   vtm_run() по одному токену. n == 0 — no-op, VTM_OK.
//
// Компилятор без weak-атрибута собирает заглушку как обычную функцию — батч
// тогда просто не ускоряет. Другая сигнатура в 0112 — ошибка компиляции здесь
// (а не молчаливая работа заглушки): это лучше, чем незаметно потерять батч.
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

struct vtm_api {
    vtm_t *m;                 // загруженная машина; владеет контекст
    atomic_int cancel;        // 1 — запрошена остановка (пишут другие потоки)
    atomic_int busy;          // 1 — идёт generate() (защита от двух потоков)
    uint32_t max_new_tokens;  // 0 — без лимита
    int32_t eos_token_id;     // < 0 — нет стоп-токена
    int last_status;          // код последнего generate() (для отладки)
    // Кэш префикса промпта между запросами (задача 0087): 0 — каждый
    // generate() начинает с vtm_reset(), 1 — префикс переиспользуется.
    int cache_prefix;
    // Промпт пачками по prefill_batch токенов (задачи 0112, 0114, 0180).
    // Действует и при cache_prefix == 1: кэшированный префикс
    // восстанавливается, некэшированный хвост считается пачками
    // (vtm_run_cached_ex). 0 — выключатель: прежний путь по токену.
    int prefill_batch;
    // 1 — следующий generate() обязан посчитать префикс заново, не доверяя
    // снимку ядра (ставится vtm_api_session_load(): состояние машины —
    // K/V и позиция — заменено данными с диска).
    int cache_invalidate;
    // Сессия KV: токен, уже отданный колбэку, но ещё не скормленный машине.
    // Снимок обязан его запомнить, иначе продолжение потеряет один шаг.
    int32_t session_token;    // значим при session_token_valid
    int session_token_valid;  // 1 — session_token можно подать машине
    // Лимит каталога сессий (задача 0083): sessions_limit == 0 или пустой
    // sessions_dir — вытеснение выключено.
    char sessions_dir[1024];
    uint64_t sessions_limit;
    char error[256];          // последняя ошибка; "" — ошибки не было
};

// Ошибка открытия: контекста ещё нет, поэтому храним отдельно. Как и
// vtm_last_error(), буфер статический и не потокобезопасный — открытие
// контекстов из разных потоков одновременно не поддерживается.
static char g_open_error[256];

static int api_fail(vtm_api_t *ctx, int code, const char *fmt, ...) {
    va_list ap;
    if (ctx != NULL) {
        va_start(ap, fmt);
        vsnprintf(ctx->error, sizeof ctx->error, fmt, ap);
        va_end(ap);
        ctx->last_status = code;
    }
    return code;
}

static int api_stop(vtm_api_t *ctx, int code) {
    // Штатная остановка — не ошибка: last_error остаётся пустым.
    ctx->last_status = code;
    return code;
}

static int open_fail(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_open_error, sizeof g_open_error, fmt, ap);
    va_end(ap);
    return 0;
}

// Признак отмены для ядра (0185, vtm_set_cancel): пачка prefill больше не
// неделима — интерпретатор опрашивает этот callback перед инструкциями
// пачки, поэтому «стоп» не ждёт конца промпта. Только атомарное чтение:
// функция зовётся с потока генерации, а пишет флаг любой другой поток.
static int api_cancel_pred(void *user) {
    const vtm_api_t *ctx = (const vtm_api_t *)user;
    return atomic_load_explicit(&ctx->cancel, memory_order_acquire) != 0;
}

// --- open ------------------------------------------------------------------

vtm_api_t *vtm_api_open(const char *path, const vtm_api_options *opts) {
    vtm_api_t *ctx;
    const vtm_backend *backend = NULL;
    uint32_t max_new = 0;
    uint32_t cache_prefix = 0;
    uint32_t prefill_batch = VTM_API_PREFILL_BATCH_DEFAULT;
    int32_t eos = -1;
    size_t i;

    g_open_error[0] = '\0';

    if (path == NULL || *path == '\0') {
        open_fail("vtm_api_open: пустой путь к .vtm");
        return NULL;
    }
    if (opts != NULL) {
        const uint32_t size = opts->struct_size;
        if (size != 0 && size < (uint32_t)sizeof(vtm_api_options)) {
            open_fail("vtm_api_open: opts->struct_size=%u меньше %u "
                      "(раскладка vtm_api_options)",
                      (unsigned)size, (unsigned)sizeof(vtm_api_options));
            return NULL;
        }
        if (opts->flags != 0) {
            open_fail("vtm_api_open: opts->flags=0x%x не поддерживаются",
                      (unsigned)opts->flags);
            return NULL;
        }
        for (i = 0; i < sizeof opts->reserved / sizeof opts->reserved[0]; i++) {
            if (opts->reserved[i] != 0) {
                open_fail("vtm_api_open: opts->reserved[%u] должно быть 0",
                          (unsigned)i);
                return NULL;
            }
        }
        backend = opts->backend;
        if (backend == NULL && opts->backend_name != NULL
            && *opts->backend_name != '\0') {
            backend = vtm_backend_by_name(opts->backend_name);
            if (backend == NULL) {
                open_fail("vtm_api_open: неизвестный backend '%s' "
                          "(auto|simd|scalar|lut|direct)",
                          opts->backend_name);
                return NULL;
            }
        }
        max_new = opts->max_new_tokens;
        eos = opts->eos_token_id;
        if (opts->cache_prefix > 1u) {
            open_fail("vtm_api_open: opts->cache_prefix=%u (ожидалось 0 или 1)",
                      (unsigned)opts->cache_prefix);
            return NULL;
        }
        cache_prefix = opts->cache_prefix;
        // prefill_batch (0114, 0180). Умолчание — 128 (промпт короче
        // считается по длине); 0 в заданной структуре — явный выключатель,
        // прежний путь по токену. struct_size == 0 по контракту api.h значит
        // «все дефолты», поэтому поле из такой структуры не читается:
        // оставляем 128. Ограничение сверху — int: длина пачки уходит в
        // vtm_run_batch() как int.
        if (size != 0) {
            if (opts->prefill_batch > (uint32_t)INT_MAX) {
                open_fail("vtm_api_open: opts->prefill_batch=%u больше %d",
                          (unsigned)opts->prefill_batch, INT_MAX);
                return NULL;
            }
            prefill_batch = opts->prefill_batch;
        }
    }

    ctx = (vtm_api_t *)calloc(1, sizeof *ctx);
    if (ctx == NULL) {
        open_fail("vtm_api_open: не хватило памяти под контекст");
        return NULL;
    }
    atomic_init(&ctx->cancel, 0);
    atomic_init(&ctx->busy, 0);
    ctx->max_new_tokens = max_new;
    ctx->eos_token_id = eos;
    ctx->cache_prefix = (int)cache_prefix;
    ctx->prefill_batch = (int)prefill_batch;

    ctx->m = vtm_load(path);
    if (ctx->m == NULL) {
        open_fail("vtm_api_open: %s: %s", path, vtm_last_error());
        free(ctx);
        return NULL;
    }
    // Отмена доходит до пачки prefill (0185): ядро опрашивает флаг между
    // инструкциями пачки, а не только между токенами генерации.
    vtm_set_cancel(ctx->m, api_cancel_pred, ctx);
    // vtm_load() уже применил VTM_BACKEND; явная опция его перекрывает.
    if (backend != NULL) vtm_set_backend(ctx->m, backend);
    return ctx;
}

// --- generate --------------------------------------------------------------

// Первый SAMP за прогон — как в CLI: программа может вызвать SAMP несколько
// раз, хост берёт первый токен. Дополнительные вызовы не роняют генерацию.
typedef struct {
    int token;
    int have;
} api_emit;

static int api_emit_cb(void *user, int token, int position) {
    api_emit *e = (api_emit *)user;
    (void)position;
    if (!e->have) {
        e->token = token;
        e->have = 1;
    }
    return 0;
}

static int api_cancelled(const vtm_api_t *ctx) {
    return api_cancel_pred((void *)ctx);
}

// Выдать cap->token хосту и крутить цикл до лимита/eos/остановки.
// Позиция следующего прогона — текущая позиция машины + 1: ко входу сюда
// m->position уже указывает на последний скормленный токен (промпта или
// сгенерированный). Так цикл одинаково работает и после промпта, и после
// восстановленной сессии. what — префикс сообщений об ошибке.
static int api_decode(vtm_api_t *ctx, api_emit *cap, vtm_api_token_cb cb,
                      void *user, const char *what) {
    vtm_t *m = ctx->m;
    int64_t generated = 0;
    int status;

    for (;;) {
        int32_t token32;
        int token;

        if (api_cancelled(ctx)) {
            return api_stop(ctx, VTM_API_CANCELLED);
        }
        token = cap->token;
        token32 = (int32_t)token;

        // Токен ещё не скормлен машине: снимок сессии должен его запомнить.
        ctx->session_token = token32;
        ctx->session_token_valid = 1;

        if (cb != NULL) {
            if (cb(&token32, 1, user) != 0) {
                return api_stop(ctx, VTM_API_STOPPED);
            }
        }
        // Отмена, пришедшая во время колбэка (или уже выставленная им через
        // vtm_api_cancel), важнее eos/лимита.
        if (api_cancelled(ctx)) {
            return api_stop(ctx, VTM_API_CANCELLED);
        }

        generated++;
        if (ctx->eos_token_id >= 0 && token == (int)ctx->eos_token_id) {
            return api_stop(ctx, VTM_API_EOS);
        }
        if (ctx->max_new_tokens != 0
            && (uint64_t)generated >= (uint64_t)ctx->max_new_tokens) {
            return api_stop(ctx, VTM_API_OK);
        }

        {
            const int64_t pos64 = (int64_t)vtm_position(m) + 1;
            if (pos64 > (int64_t)INT_MAX) {
                return api_fail(ctx, VTM_E_BOUNDS,
                                "%s: позиция %lld не влезает в int",
                                what, (long long)pos64);
            }
            vtm_set_position(m, (int)pos64);
        }
        // Токен уходит в машину — «висящего» токена больше нет.
        ctx->session_token_valid = 0;
        vtm_set_token(m, token);
        memset(cap, 0, sizeof *cap);
        status = (int)vtm_run(m, api_emit_cb, cap);
        if (status != VTM_OK) {
            return api_fail(ctx, status, "%s: токен %lld: %s: %s", what,
                            (long long)generated,
                            vtm_strerror((vtm_status)status),
                            vtm_error_message(m));
        }
        if (!cap->have) {
            // Программа дошла до HALT, не выбрав токен: генерация закончена.
            return api_stop(ctx, VTM_API_OK);
        }
    }
}

// Возвращает код vtm_api_generate(); busy/cancel сбрасывает вызывающий.
static int api_run(vtm_api_t *ctx, const int32_t *prompt_ids, size_t n,
                   vtm_api_token_cb cb, void *user) {
    vtm_t *m = ctx->m;
    api_emit cap;
    size_t i;

    // Повторное использование: каждый generate() — с начала. vtm_reset()
    // обнуляет позицию/входной токен/счётчик и (через счётчик) состояние
    // сэмплера; KV-кэш не чистится, но ATTN читает только позиции,
    // перезаписанные текущим промптом.
    vtm_reset(m);
    ctx->session_token_valid = 0;

    // Отрицательные id проверяются до прогона: пачка (vtm_run_batch) неделима
    // и не даёт точки, где цикл поймал бы VTM_E_ARG на середине промпта.
    for (i = 0; i < n; i++) {
        if (prompt_ids[i] < 0) {
            return api_fail(ctx, VTM_E_ARG,
                            "vtm_api_generate: prompt_ids[%zu]=%d отрицателен",
                            i, (int)prompt_ids[i]);
        }
    }

    // Токены промпта, кроме последнего, только наполняют KV-кэш: SAMP на них
    // (если программа его вызывает) исполняется, но токен хосту не отдаётся —
    // ровно как в CLI. prefill_batch > 0 (0114) считает их пачками через
    // vtm_run_batch(); последний токен всё равно идёт отдельным vtm_run() с
    // колбэком, потому что emit нужен только на нём.
    for (i = 0; i + 1 < n;) {
        vtm_status st;

        if (api_cancelled(ctx)) {
            return api_stop(ctx, VTM_API_CANCELLED);
        }
        if (ctx->prefill_batch > 0) {
            size_t len = (n - 1) - i;
            if (len > (size_t)ctx->prefill_batch) {
                len = (size_t)ctx->prefill_batch;
            }
            vtm_set_position(m, (int)i);
            st = vtm_run_batch(m, prompt_ids + i, (int)len);
            if (st == VTM_E_CANCELLED) {
                // 0185: отмена внутри пачки prefill (ядро опросило флаг перед
                // инструкцией). Это не сбой: код штатной остановки.
                return api_stop(ctx, VTM_API_CANCELLED);
            }
            if (st != VTM_OK) {
                return api_fail(ctx, (int)st,
                                "vtm_api_generate: промпт[%zu..%zu): %s: %s",
                                i, i + len, vtm_strerror(st),
                                vtm_error_message(m));
            }
            i += len;
        } else {
            vtm_set_position(m, (int)i);
            vtm_set_token(m, (int)prompt_ids[i]);
            st = vtm_run(m, NULL, NULL);
            if (st != VTM_OK) {
                return api_fail(ctx, (int)st,
                                "vtm_api_generate: промпт[%zu]: %s: %s", i,
                                vtm_strerror(st), vtm_error_message(m));
            }
            i++;
        }
        if (api_cancelled(ctx)) {
            return api_stop(ctx, VTM_API_CANCELLED);
        }
    }

    // Последний токен промпта даёт первый токен ответа (и в режиме пачек).
    memset(&cap, 0, sizeof cap);
    vtm_set_position(m, (int)(n - 1));
    vtm_set_token(m, (int)prompt_ids[n - 1]);
    {
        vtm_status st = vtm_run(m, api_emit_cb, &cap);
        if (st != VTM_OK) {
            return api_fail(ctx, (int)st,
                            "vtm_api_generate: промпт[%zu]: %s: %s", n - 1,
                            vtm_strerror(st), vtm_error_message(m));
        }
    }

    if (!cap.have) {
        return api_fail(ctx, VTM_E_ARG,
                        "vtm_api_generate: программа не вызвала SAMP на "
                        "последнем токене промпта");
    }

    return api_decode(ctx, &cap, cb, user, "vtm_api_generate");
}

// То же, но с кэшем префикса (задача 0087, opts->cache_prefix == 1).
//
// Кэш ядра — это снимок состояния машины после прогона ровно ids[0..len):
// K/V уже лежат в тензорах, поэтому «попадание» не пересчитывает префикс, а
// восстанавливает снимок и продолжает с позиции len. Хост-функция
// vtm_run_cached() не вызывает emit (это прогон промпта, а не генерация), а
// токен, выбранный SAMP на последнем токене промпта, нужен: с него начинается
// api_decode(). Поэтому кэшируются все токены промпта, кроме последнего
// (n-1): последний прогоняется здесь же с колбэком, как в api_run(). Цена —
// один токен на запрос, выигрыш — весь остальной префикс.
//
// vtm_reset() в этом пути не вызывается: на промахе его делает сам
// vtm_run_cached(), на попадании снимок восстанавливает позицию, счётчик и
// состояние сэмплера. Сессия KV меняет состояние машины (K/V и позицию), не
// трогая снимок, поэтому после vtm_api_session_load() ставится
// cache_invalidate: следующий прогон префикса считается заново
// (vtm_cache_prefix — это vtm_reset() + полный прогон + новый снимок).
//
// Пачки (0112/0114/0180) и кэш работают вместе: префикс берётся из кэша,
// некэшированный хвост считается пачками по prefill_batch через
// vtm_run_cached_ex() (включая MoE-пачку 0179); vtm_cache_prefix_ex() — тот
// же батчевый путь при промахе после session_load. При prefill_batch == 0
// вызываются прежние vtm_run_cached()/vtm_cache_prefix(): на SIMD это
// побитовый путь по токену, на скаляре — пачка, как было до 0180. Один
// случай interp.c откатывает на поштучный путь и при N > 0 — плотный граф с
// Q4_K/Q6_K на SIMD (батчевое ядро kind 6/7 там медленнее int8-dot, см.
// cache_batch_worthwhile в interp.c): на 3B пачка 128 давала 11,4 с против
// 9,3 с по токену, а критерий задачи — «на 3B не хуже».
static int api_run_cached(vtm_api_t *ctx, const int32_t *prompt_ids, size_t n,
                          vtm_api_token_cb cb, void *user) {
    vtm_t *m = ctx->m;
    const size_t prefill = n - 1; // n >= 1 проверено vtm_api_generate()
    api_emit cap;
    vtm_status st;
    size_t i;

    ctx->session_token_valid = 0;

    // Отрицательные id проверяются до прогона: vtm_run_cached() неделим и не
    // даёт точки, где api_run() поймал бы VTM_E_ARG на середине промпта.
    for (i = 0; i < n; i++) {
        if (prompt_ids[i] < 0) {
            return api_fail(ctx, VTM_E_ARG,
                            "vtm_api_generate: prompt_ids[%zu]=%d отрицателен",
                            i, (int)prompt_ids[i]);
        }
    }
    if (api_cancelled(ctx)) {
        return api_stop(ctx, VTM_API_CANCELLED);
    }

    if (ctx->cache_invalidate) {
        st = vtm_cache_prefix_ex(m, prompt_ids, (int)prefill,
                                 ctx->prefill_batch);
        ctx->cache_invalidate = 0;
    } else {
        st = vtm_run_cached_ex(m, prompt_ids, (int)prefill,
                               ctx->prefill_batch);
    }
    if (st == VTM_E_CANCELLED) {
        // 0185: отмена внутри префикса (пачка или поштучный путь). Снимок
        // кэша при этом не создаётся (cache_store не дошёл), поэтому
        // следующий запрос пересчитает префикс заново — отравы нет.
        return api_stop(ctx, VTM_API_CANCELLED);
    }
    if (st != VTM_OK) {
        return api_fail(ctx, (int)st,
                        "vtm_api_generate: промпт[0..%zu): %s: %s", prefill,
                        vtm_strerror(st), vtm_error_message(m));
    }
    // Отмена во время префикса, пришедшая уже после прогона: проверяем
    // здесь, чтобы не начинать генерацию.
    if (api_cancelled(ctx)) {
        return api_stop(ctx, VTM_API_CANCELLED);
    }

    // Последний токен промпта — ровно как в api_run(): с emit, потому что
    // SAMP на нём выбирает первый токен ответа.
    memset(&cap, 0, sizeof cap);
    vtm_set_position(m, (int)prefill);
    vtm_set_token(m, (int)prompt_ids[prefill]);
    st = vtm_run(m, api_emit_cb, &cap);
    if (st != VTM_OK) {
        return api_fail(ctx, (int)st, "vtm_api_generate: промпт[%zu]: %s: %s",
                        prefill, vtm_strerror(st), vtm_error_message(m));
    }
    if (api_cancelled(ctx)) {
        return api_stop(ctx, VTM_API_CANCELLED);
    }
    if (!cap.have) {
        return api_fail(ctx, VTM_E_ARG,
                        "vtm_api_generate: программа не вызвала SAMP на "
                        "последнем токене промпта");
    }

    return api_decode(ctx, &cap, cb, user, "vtm_api_generate");
}

int vtm_api_generate(vtm_api_t *ctx, const int32_t *prompt_ids, size_t n,
                     vtm_api_token_cb cb, void *user) {
    int expected = 0;
    int rc;

    if (ctx == NULL) return VTM_E_ARG;
    ctx->error[0] = '\0';
    ctx->last_status = VTM_API_OK;

    if (prompt_ids == NULL) {
        return api_fail(ctx, VTM_E_ARG,
                        "vtm_api_generate: prompt_ids == NULL при n=%zu", n);
    }
    if (n == 0) {
        return api_fail(ctx, VTM_E_ARG,
                        "vtm_api_generate: пустой промпт (n=0)");
    }
    if (n > (size_t)INT_MAX) {
        return api_fail(ctx, VTM_E_ARG,
                        "vtm_api_generate: промпт слишком длинный (%zu токенов)",
                        n);
    }

    if (!atomic_compare_exchange_strong_explicit(
            &ctx->busy, &expected, 1, memory_order_acq_rel,
            memory_order_acquire)) {
        // Чужой generate() уже работает. Не трогаем ctx->error: это была бы
        // гонка с ним; ошибку можно понять по коду VTM_E_ARG.
        return VTM_E_ARG;
    }

    if (api_cancelled(ctx)) {
        rc = api_stop(ctx, VTM_API_CANCELLED);
    } else if (ctx->cache_prefix) {
        rc = api_run_cached(ctx, prompt_ids, n, cb, user);
    } else {
        rc = api_run(ctx, prompt_ids, n, cb, user);
    }

    // Флаг отмены живёт только в пределах одного generate(): иначе отмена,
    // пришедшая в конце, отменила бы следующий запуск.
    atomic_store_explicit(&ctx->cancel, 0, memory_order_release);
    atomic_store_explicit(&ctx->busy, 0, memory_order_release);
    return rc;
}

// --- сессии KV (задачи 0079, 0083) ----------------------------------------

// Лимит каталога: вытеснить старые снимки, не трогая keep_path. Ошибки
// вытеснения не поднимаются наружу: снимок уже записан/прочитан, а
// предупреждение vtm_session_prune печатает в stderr сам.
static void api_sessions_enforce(vtm_api_t *ctx, const char *keep_path) {
    if (ctx->sessions_limit == 0 || ctx->sessions_dir[0] == '\0') return;
    (void)vtm_session_prune(ctx->sessions_dir, ctx->sessions_limit, keep_path);
}

// Занять контекст на время операции: та же защита, что у generate(), без
// гонки с параллельным generate() (вернём VTM_E_ARG, не трогая error).
static int api_try_busy(vtm_api_t *ctx) {
    int expected = 0;
    return atomic_compare_exchange_strong_explicit(
        &ctx->busy, &expected, 1, memory_order_acq_rel, memory_order_acquire);
}

int vtm_api_session_save(vtm_api_t *ctx, const char *path) {
    int rc;

    if (ctx == NULL) return VTM_E_ARG;
    ctx->error[0] = '\0';
    ctx->last_status = VTM_API_OK;
    if (path == NULL || *path == '\0') {
        return api_fail(ctx, VTM_E_ARG, "vtm_api_session_save: пустой путь");
    }
    if (!api_try_busy(ctx)) return VTM_E_ARG;

    rc = vtm_session_save(ctx->m,
                          ctx->session_token_valid ? ctx->session_token : -1,
                          path, ctx->error, sizeof ctx->error);
    if (rc != VTM_OK) {
        ctx->last_status = rc;
        if (ctx->error[0] == '\0') {
            snprintf(ctx->error, sizeof ctx->error, "vtm_api_session_save: %s",
                     vtm_strerror((vtm_status)rc));
        }
    } else {
        // Снимок уже на диске: даже если вытеснение старых не удалось, save()
        // успешен (ошибки prune — только предупреждения в stderr).
        api_sessions_enforce(ctx, path);
    }
    atomic_store_explicit(&ctx->busy, 0, memory_order_release);
    return rc;
}

int vtm_api_session_load(vtm_api_t *ctx, const char *path) {
    int32_t pending = -1;
    int rc;

    if (ctx == NULL) return VTM_E_ARG;
    ctx->error[0] = '\0';
    ctx->last_status = VTM_API_OK;
    if (path == NULL || *path == '\0') {
        return api_fail(ctx, VTM_E_ARG, "vtm_api_session_load: пустой путь");
    }
    if (!api_try_busy(ctx)) return VTM_E_ARG;

    rc = vtm_session_load(ctx->m, &pending, path, ctx->error,
                          sizeof ctx->error);
    if (rc == VTM_OK) {
        // pending < 0 — сессия закончилась HALT'ом, продолжать нечем.
        ctx->session_token = pending;
        ctx->session_token_valid = (pending >= 0);
        // K/V и позиция заменены данными с диска: снимок кэша префикса
        // (задача 0087) описывает уже не это состояние — следующий
        // generate() обязан посчитать префикс заново.
        ctx->cache_invalidate = 1;
    } else {
        ctx->last_status = rc;
        if (ctx->error[0] == '\0') {
            snprintf(ctx->error, sizeof ctx->error, "vtm_api_session_load: %s",
                     vtm_strerror((vtm_status)rc));
        }
    }
    if (rc == VTM_OK) api_sessions_enforce(ctx, path);
    atomic_store_explicit(&ctx->busy, 0, memory_order_release);
    return rc;
}

int vtm_api_sessions_dir(vtm_api_t *ctx, const char *path) {
    size_t n;

    if (ctx == NULL) return VTM_E_ARG;
    ctx->error[0] = '\0';
    if (path == NULL || *path == '\0') {
        ctx->sessions_dir[0] = '\0';
        return VTM_OK;
    }
    n = strlen(path);
    if (n >= sizeof ctx->sessions_dir) {
        return api_fail(ctx, VTM_E_ARG,
                        "vtm_api_sessions_dir: путь длиннее %u байт",
                        (unsigned)(sizeof ctx->sessions_dir - 1u));
    }
    memcpy(ctx->sessions_dir, path, n + 1u);
    return VTM_OK;
}

int vtm_api_sessions_limit(vtm_api_t *ctx, uint64_t bytes) {
    if (ctx == NULL) return VTM_E_ARG;
    ctx->error[0] = '\0';
    // 0 — лимит снят; вытеснение не запускается до следующего save/load.
    ctx->sessions_limit = bytes;
    return VTM_OK;
}

int vtm_api_session_continue(vtm_api_t *ctx, vtm_api_token_cb cb, void *user) {
    vtm_t *m;
    api_emit cap;
    int rc;
    int status;

    if (ctx == NULL) return VTM_E_ARG;
    ctx->error[0] = '\0';
    ctx->last_status = VTM_API_OK;
    if (!ctx->session_token_valid) {
        return api_fail(ctx, VTM_E_ARG,
                        "vtm_api_session_continue: нет загруженной сессии "
                        "(сначала vtm_api_session_load)");
    }
    if (!api_try_busy(ctx)) return VTM_E_ARG;

    m = ctx->m;
    if (api_cancelled(ctx)) {
        rc = api_stop(ctx, VTM_API_CANCELLED);
    } else {
        const int64_t pos64 = (int64_t)vtm_position(m) + 1;
        if (pos64 > (int64_t)INT_MAX) {
            rc = api_fail(ctx, VTM_E_BOUNDS,
                          "vtm_api_session_continue: позиция %lld не влезает "
                          "в int", (long long)pos64);
        } else {
            vtm_set_position(m, (int)pos64);
            vtm_set_token(m, ctx->session_token);
            ctx->session_token_valid = 0;
            memset(&cap, 0, sizeof cap);
            status = (int)vtm_run(m, api_emit_cb, &cap);
            if (status != VTM_OK) {
                rc = api_fail(ctx, status,
                              "vtm_api_session_continue: входной токен: "
                              "%s: %s", vtm_strerror((vtm_status)status),
                              vtm_error_message(m));
            } else if (!cap.have) {
                rc = api_stop(ctx, VTM_API_OK); // HALT: продолжать нечего
            } else {
                rc = api_decode(ctx, &cap, cb, user,
                                "vtm_api_session_continue");
            }
        }
    }

    // Как и в generate(): отмена живёт только в пределах одного вызова.
    atomic_store_explicit(&ctx->cancel, 0, memory_order_release);
    atomic_store_explicit(&ctx->busy, 0, memory_order_release);
    return rc;
}

// --- cancel / close / error ------------------------------------------------

void vtm_api_cancel(vtm_api_t *ctx) {
    if (ctx == NULL) return;
    // Только атомарная запись: никаких блокировок, аллокаций и обращений к
    // машине — функция безопасна из любого потока и в любой момент.
    atomic_store_explicit(&ctx->cancel, 1, memory_order_release);
}

void vtm_api_cancel_clear(vtm_api_t *ctx) {
    if (ctx == NULL) return;
    // Добавлено в 0185 для хостов с асинхронной отменой: если «стоп» пришёл
    // уже после возврата generate(), флаг остался бы и отменил следующий
    // запуск. Хост знает, что генерации нет (его поток вернулся), и снимает
    // флаг сам. Гонки с новым generate() быть не должно: вызывающий не
    // стартует его, пока не снимет флаг.
    atomic_store_explicit(&ctx->cancel, 0, memory_order_release);
}

void vtm_api_close(vtm_api_t *ctx) {
    if (ctx == NULL) return;
    // Контракт: close() не гоняется с generate() на этом контексте.
    vtm_free(ctx->m);
    ctx->m = NULL;
    free(ctx);
}

const char *vtm_api_last_error(const vtm_api_t *ctx) {
    if (ctx == NULL) return g_open_error;
    return ctx->error;
}
