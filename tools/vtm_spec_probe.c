// vtm_spec_probe.c — C-помощник для спекулятивного декода (задача 0203).
//
// Зачем отдельный файл, а не C-ABI src/vtm/api.h: спекулятивному декоду нужны
// две вещи, которых в публичном C-ABI нет:
//
//   1. **argmax на каждой позиции пачки.** vtm_run_batch() считает пачку одним
//      проходом по весам, но emit не вызывает (это prefill, а не генерация), а
//      в машине после пачки остаются логиты только последнего токена. Для
//      проверки K черновых токенов нужен SAMP-токен каждой позиции. Его можно
//      взять из истории повторов машины: op_samp() кладёт выбранный токен в
//      samp_hist на каждом SAMP, в том числе внутри пачки. Для жадного SAMP
//      это ровно argmax позиции.
//   2. **Откат на принятый префикс.** KV-кэш адресуется глобальной позицией,
//      ATTN читает только позиции до текущей, а KVW перезаписывает строку при
//      следующем токене — поэтому «откат» это vtm_set_position() назад, без
//      копирования K/V. То же состояние сэмплера (историю повторов) надо
//      вернуть вручную: пачка успела положить в неё все K проверенных токенов,
//      а в последовательности остались только принятые.
//
// Файл не трогает src/vtm: он подключает vtm_internal.h (как это делают
// tests/vtm/test_vtm.c) и линкуется с готовым libvtm.a. Собирается из Python
// (eva/speculative.py) тем же cc, что и libvtm.dylib в
// eva/backends/vtm_backend.py.
//
// Контракт состояния — как у vtm_api_generate():
//   * prefill(ids, n) сбрасывает машину, считает ids[0..n-1] (первые n-1 —
//     пачкой), возвращает SAMP-токен последнего токена промпта (первый
//     кандидат продолжения); позиция == n-1, в KV — весь промпт;
//   * feed(t) сдвигает позицию на +1, кормит t и возвращает следующий
//     SAMP-токен (кандидат на позицию +2);
//   * batch(ids, K) с позиции p считает ids[0..K-1] на позициях p+1..p+K и
//     возвращает SAMP-токен каждой позиции (out[0..K-1]); позиция == p+K;
//   * rewind(pos) — только vtm_set_position() назад (pos <= текущей).
//
// Сэмплинг принудительно жадный (argmax): мета контейнеров Qwen2.5 3B и 30B
// говорит "sampling": "greedy", и паритет с vtm_api_generate() проверяется
// именно на нём. Сэмплинг с температурой в этой версии не поддерживается.

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vtm.h"
#include "vtm_backend.h"
#include "vtm_internal.h"

typedef struct probe {
    vtm_t *m;
    char err[256];
} probe;

static probe *probe_of(void *h) { return (probe *)h; }

static int probe_fail(probe *p, const char *what) {
    if (p == NULL) return -1;
    snprintf(p->err, sizeof p->err, "%s: %s", what,
             p->m ? vtm_error_message(p->m) : "нет машины");
    return -1;
}

void *vtm_spec_open(const char *path, const char *backend, char *err,
                    int errcap) {
    probe *p;
    vtm_t *m;

    if (err != NULL && errcap > 0) err[0] = '\0';
    m = vtm_load(path);
    if (m == NULL) {
        if (err != NULL && errcap > 0)
            snprintf(err, (size_t)errcap, "vtm_load: %s", vtm_last_error());
        return NULL;
    }
    if (backend != NULL && *backend != '\0') {
        const vtm_backend *b = vtm_backend_by_name(backend);
        if (b == NULL) {
            if (err != NULL && errcap > 0)
                snprintf(err, (size_t)errcap, "неизвестный backend '%s'",
                         backend);
            vtm_free(m);
            return NULL;
        }
        vtm_set_backend(m, b);
    }
    p = (probe *)calloc(1, sizeof *p);
    if (p == NULL) {
        if (err != NULL && errcap > 0)
            snprintf(err, (size_t)errcap, "нет памяти под probe");
        vtm_free(m);
        return NULL;
    }
    p->m = m;
    return p;
}

void vtm_spec_close(void *h) {
    probe *p = probe_of(h);

    if (p == NULL) return;
    vtm_free(p->m);
    free(p);
}

const char *vtm_spec_error(void *h) {
    probe *p = probe_of(h);

    if (p == NULL) return "нет контекста";
    if (p->err[0] != '\0') return p->err;
    return vtm_error_message(p->m);
}

void vtm_spec_clear_error(void *h) {
    probe *p = probe_of(h);

    if (p != NULL) p->err[0] = '\0';
}

const char *vtm_spec_backend(void *h) {
    probe *p = probe_of(h);

    return p == NULL ? "" : vtm_backend_name(p->m);
}

int vtm_spec_sampling(void *h, double *temperature, int *top_k, double *top_p,
                      double *repetition_penalty, int *repetition_window) {
    probe *p = probe_of(h);
    vtm_sampling s;

    if (p == NULL) return -1;
    if (vtm_sampling_params(p->m, &s) != 1) return 0;
    if (temperature != NULL) *temperature = s.temperature;
    if (top_k != NULL) *top_k = s.top_k;
    if (top_p != NULL) *top_p = s.top_p;
    if (repetition_penalty != NULL) *repetition_penalty = s.repetition_penalty;
    if (repetition_window != NULL) *repetition_window = s.repetition_window;
    return 1;
}

// Принудительно жадный SAMP (argmax) без штрафа за повторы.
int vtm_spec_force_greedy(void *h) {
    probe *p = probe_of(h);

    if (p == NULL || p->m == NULL) return -1;
    p->m->samp_ready = 1;
    p->m->samp_temperature = 0.0;
    p->m->samp_top_k = 0;
    p->m->samp_top_p = 1.0;
    p->m->samp_repetition_penalty = 1.0;
    p->m->samp_repetition_window = 0;
    p->m->samp_seed = 0;
    vtm_samp_reset(p->m);   // засеять ГПСЧ и очистить историю
    return 0;
}

int vtm_spec_hist_len(void *h) {
    probe *p = probe_of(h);

    return p == NULL ? -1 : p->m->samp_hist_len;
}

// Сохранить историю повторов в хронологическом порядке (старые → новые).
// Возврат — длина истории; buf может быть NULL (только узнать длину).
int vtm_spec_hist_save(void *h, int32_t *buf, int cap) {
    probe *p = probe_of(h);
    int len, i, start;

    if (p == NULL) return -1;
    len = p->m->samp_hist_len;
    if (buf == NULL) return len;
    if (cap < len) {
        snprintf(p->err, sizeof p->err, "hist_save: cap=%d < len=%d", cap, len);
        return -1;
    }
    start = p->m->samp_hist_head - len;
    while (start < 0) start += VTM_MAX_REPETITION_WINDOW;
    for (i = 0; i < len; i++)
        buf[i] = p->m->samp_hist[(start + i) % VTM_MAX_REPETITION_WINDOW];
    return len;
}

// Восстановить историю (buf — старые → новые, len <= VTM_MAX_REPETITION_WINDOW).
int vtm_spec_hist_restore(void *h, const int32_t *buf, int len) {
    probe *p = probe_of(h);
    int i;

    if (p == NULL || len < 0 || len > VTM_MAX_REPETITION_WINDOW) return -1;
    if (len > 0 && buf == NULL) return -1;
    for (i = 0; i < len; i++) p->m->samp_hist[i] = (int)buf[i];
    p->m->samp_hist_len = len;
    p->m->samp_hist_head = len % VTM_MAX_REPETITION_WINDOW;
    return 0;
}

// Дописать один токен в историю (как samp_hist_push() в interp.c).
int vtm_spec_hist_push(void *h, int32_t token) {
    probe *p = probe_of(h);

    if (p == NULL) return -1;
    p->m->samp_hist[p->m->samp_hist_head] = (int)token;
    p->m->samp_hist_head =
        (p->m->samp_hist_head + 1) % VTM_MAX_REPETITION_WINDOW;
    if (p->m->samp_hist_len < VTM_MAX_REPETITION_WINDOW)
        p->m->samp_hist_len++;
    return 0;
}

// SAMP-токен, выбранный на позиции steps_back шагов назад (0 — последний).
static int hist_back(const vtm_t *m, int steps_back) {
    int idx = m->samp_hist_head - 1 - steps_back;

    while (idx < 0) idx += VTM_MAX_REPETITION_WINDOW;
    return m->samp_hist[idx];
}

static int emit_cb(void *user, int token, int position) {
    (void)position;
    *(int *)user = token;
    return 0;
}

int vtm_spec_reset(void *h) {
    probe *p = probe_of(h);

    if (p == NULL) return -1;
    vtm_reset(p->m);
    p->err[0] = '\0';
    return 0;
}

int vtm_spec_position(void *h) {
    probe *p = probe_of(h);

    return p == NULL ? -1 : p->m->position;
}

int vtm_spec_generated(void *h) {
    probe *p = probe_of(h);

    return p == NULL ? -1 : p->m->generated;
}

// Считать промпт ids[0..n-1] с нуля; вернуть SAMP-токен последнего токена.
int vtm_spec_prefill(void *h, const int32_t *ids, int n, int32_t *out) {
    probe *p = probe_of(h);
    int token = -1;
    vtm_status st;

    if (p == NULL || ids == NULL || n <= 0 || out == NULL) return -1;
    vtm_reset(p->m);
    p->err[0] = '\0';
    if (n > 1) {
        st = vtm_run_batch(p->m, ids, n - 1);
        if (st != VTM_OK) return probe_fail(p, "prefill batch");
        // vtm_run_batch оставляет позицию на последнем токене пачки; он же
        // нужен как позиция последнего токена промпта.
        if (vtm_position(p->m) != n - 2) vtm_set_position(p->m, n - 2);
    }
    vtm_set_position(p->m, n - 1);
    vtm_set_token(p->m, (int)ids[n - 1]);
    st = vtm_run(p->m, emit_cb, &token);
    if (st != VTM_OK) return probe_fail(p, "prefill last");
    if (token < 0) {
        snprintf(p->err, sizeof p->err, "prefill: SAMP не выбрал токен");
        return -1;
    }
    *out = (int32_t)token;
    return 0;
}

// Один шаг: покормить token на следующей позиции, вернуть следующий SAMP-токен.
int vtm_spec_feed(void *h, int32_t token, int32_t *out) {
    probe *p = probe_of(h);
    int tok = -1;
    int64_t pos;
    vtm_status st;

    if (p == NULL || out == NULL) return -1;
    pos = (int64_t)vtm_position(p->m) + 1;
    if (pos > (int64_t)INT_MAX) {
        snprintf(p->err, sizeof p->err, "feed: позиция %lld не влезает в int",
                 (long long)pos);
        return -1;
    }
    vtm_set_position(p->m, (int)pos);
    vtm_set_token(p->m, (int)token);
    st = vtm_run(p->m, emit_cb, &tok);
    if (st != VTM_OK) return probe_fail(p, "feed");
    if (tok < 0) {
        snprintf(p->err, sizeof p->err, "feed: SAMP не выбрал токен");
        return -1;
    }
    *out = (int32_t)tok;
    return 0;
}

// Пачка K токенов СЛЕДУЮЩИХ за текущей позицией p: позиции p+1..p+K.
// out[i] — SAMP-токен позиции p+1+i (для жадного SAMP это argmax на ней).
// vtm_run_batch() сам начинает с текущей позиции (start = vtm_position), т.е.
// первый id лёг бы в p — поэтому позиция сдвигается на p+1 до вызова.
int vtm_spec_batch(void *h, const int32_t *ids, int n, int32_t *out) {
    probe *p = probe_of(h);
    vtm_status st;
    int i, len;
    int64_t pos;

    if (p == NULL || ids == NULL || out == NULL || n <= 0) return -1;
    if (n > VTM_MAX_REPETITION_WINDOW) {
        snprintf(p->err, sizeof p->err, "batch: n=%d больше истории %d", n,
                 VTM_MAX_REPETITION_WINDOW);
        return -1;
    }
    p->err[0] = '\0';
    pos = (int64_t)vtm_position(p->m) + 1;
    if (pos + (int64_t)n - 1 > (int64_t)INT_MAX) {
        snprintf(p->err, sizeof p->err, "batch: позиция %lld+%d не влезает в int",
                 (long long)pos, n);
        return -1;
    }
    vtm_set_position(p->m, (int)pos);
    st = vtm_run_batch(p->m, ids, n);
    if (st != VTM_OK) return probe_fail(p, "batch");
    len = p->m->samp_hist_len;
    if (len < n) {
        snprintf(p->err, sizeof p->err,
                 "batch: в истории %d токенов, ожидалось не меньше %d", len, n);
        return -1;
    }
    for (i = 0; i < n; i++) out[i] = (int32_t)hist_back(p->m, n - 1 - i);
    return 0;
}

int vtm_spec_rewind(void *h, int pos) {
    probe *p = probe_of(h);

    if (p == NULL || pos < 0 || pos > vtm_position(p->m)) {
        if (p != NULL)
            snprintf(p->err, sizeof p->err, "rewind: %d вне 0..%d", pos,
                     p->m->position);
        return -1;
    }
    vtm_set_position(p->m, pos);
    return 0;
}

// Только для отладки/замера: сколько инструкций в графе и версия ABI меты.
int vtm_spec_instr_count(void *h) {
    probe *p = probe_of(h);

    return p == NULL ? -1 : (int)vtm_instr_count(p->m);
}

// Границы KV-окна (контекст) для проверки, что пачка не режется кольцом.
int vtm_spec_context_len(void *h) {
    probe *p = probe_of(h);

    if (p == NULL) return -1;
    return p->m->ctx_window > 0 ? p->m->ctx_window : -1;
}
