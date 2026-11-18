// SHA-256 для контроля целостности контейнера .vtm (задачи 0066, 0078).
//
// Публичный интерфейс небольшой: инкрементальный хеш плюс диагностика,
// какое ядро выбрано (портативное или аппаратное). Реализация —
// src/vtm/sha256.c: портативный C99 и ARMv8 crypto extensions
// (sha256h/sha256h2/sha256su0/sha256su1) с откатом на портативную.
#ifndef VTM_SHA256_H
#define VTM_SHA256_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t state[8];
    uint64_t total;
    uint8_t buf[64];
    size_t buflen;
} vtm_sha256_ctx;

// Инкрементальный хеш. Порядок как у обычного SHA-256:
// init → update (сколько угодно раз) → final.
void vtm_sha256_init(vtm_sha256_ctx *c);
void vtm_sha256_update(vtm_sha256_ctx *c, const void *data, size_t n);
void vtm_sha256_final(vtm_sha256_ctx *c, uint8_t out[32]);

// 64 строчных hex-символа + NUL.
void vtm_sha256_hex(const uint8_t digest[32], char out[65]);

// Имя выбранного ядра: "armv8-crypto" или "portable". Первый вызов запускает
// детект и самопроверку аппаратного ядра.
const char *vtm_sha256_impl(void);

// 1 — аппаратное ядро включено и прошло самопроверку.
int vtm_sha256_hw_available(void);

// Принудительный откат на портативную реализацию (тесты, флаг
// VTM_SHA_PORTABLE=1). on == 0 снимает флаг и сбрасывает результат детекта.
void vtm_sha256_force_portable(int on);

#ifdef __cplusplus
}
#endif

#endif
