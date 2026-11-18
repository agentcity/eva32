// Внутреннее устройство машины. Наружу (vtm.h) структура vtm_t непрозрачна.
#ifndef VTM_INTERNAL_H
#define VTM_INTERNAL_H

#include "vtm.h"

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
};

// Записать ошибку в машину и в глобальный буфер vtm_last_error().
void vtm_set_error(vtm_t *m, vtm_status st, const char *fmt, ...);

const char *vtm_op_name(uint8_t opcode);

#endif // VTM_INTERNAL_H
