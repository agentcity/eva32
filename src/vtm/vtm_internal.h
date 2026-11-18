// Внутреннее устройство машины. Наружу (vtm.h) структура vtm_t непрозрачна.
#ifndef VTM_INTERNAL_H
#define VTM_INTERNAL_H

#include "vtm.h"

// Предел на разбор JSON-метаданных: защита от абсурда, а не от OOM. Мета
// синтетики с полным списком логитов весит единицы мегабайт, так что с
// запасом; больше — считаем мету битой и живём на дефолтах.
#define VTM_MAX_META_BYTES ((size_t)64u << 20)

struct vtm {
    vtm_tensor     *tensors;
    uint32_t        n_tensors;
    vtm_instr      *instr;
    uint32_t        n_instr;

    const vtm_backend *backend;

    uint8_t        *arena;      // выровненное начало арены
    uint8_t        *arena_raw;  // то, что вернул malloc
    size_t          arena_size;

    uint16_t        header_flags;
    int             position;   // база позиции для ROPE/KVW/ATTN/SM
    int             input_token;// токен для EMB с флагом EMB_INPUT (-1 = нет)
    int             generated;  // сколько токенов выбрал SAMP
    int             halted;     // последний прогон дошёл до HALT
    char            error[256];

    // Метаданные (docs/vtm.md, «Метаданные»). Дефолты выставляются в
    // vtm_load_mem() до разбора, чтобы битая мета ничего не ломала.
    int             meta_present;    // JSON разобран как объект
    double          rope_theta;      // VTM_DEFAULT_ROPE_THETA, если не задана
    int             rope_layout;     // vtm_rope_layout
    uint32_t        vocab_size;      // 0 = не задан
    int             tied_embeddings; // 0/1
    char           *meta_json;       // копия JSON (NUL-терминированная) или NULL

    // --- SAMP (задача 0051): параметры и состояние сэмплера ---------------
    //
    // Живут в машине, а не в статике interp.c: иначе новая машина, получившая
    // адрес освобождённой, унаследовала бы чужое окно повторов. calloc в
    // vtm_load_mem() обнуляет поля; параметры разбираются лениво, на первом
    // SAMP (samp_ready), потому что vtm_reset()/vtm_free() здесь не наши
    // файлы — сброс окна замечается по счётчику generated.
    int             samp_ready;          // параметры разобраны, ГПСЧ засеян
    uint64_t        samp_rng;            // состояние xorshift64*
    uint64_t        samp_seed;           // эффективное зерно (мета или 0)
    double          samp_temperature;    // <= 0 — жадно
    int             samp_top_k;          // <= 0 — не отсекать
    double          samp_top_p;          // >= 1 — не отсекать
    double          samp_repetition_penalty;
    int             samp_repetition_window;
    int             samp_prev_generated; // vtm_generated() на прошлом SAMP
    int             samp_hist_len;       // элементов в кольце 0..MAX
    int             samp_hist_head;      // куда писать следующий
    int             samp_hist[VTM_MAX_REPETITION_WINDOW];
};

// Записать ошибку в машину и в глобальный буфер vtm_last_error().
void vtm_set_error(vtm_t *m, vtm_status st, const char *fmt, ...);

const char *vtm_op_name(uint8_t opcode);

// Сбросить состояние сэмплера SAMP (окно повторов и ГПСЧ). Предназначено
// для vtm_reset(); без вызова сброс всё равно распознаётся по vtm_generated().
void vtm_samp_reset(vtm_t *m);

#endif // VTM_INTERNAL_H
