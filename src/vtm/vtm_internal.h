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
};

// Записать ошибку в машину и в глобальный буфер vtm_last_error().
void vtm_set_error(vtm_t *m, vtm_status st, const char *fmt, ...);

const char *vtm_op_name(uint8_t opcode);

#endif // VTM_INTERNAL_H
