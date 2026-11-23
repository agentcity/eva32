// Сменный слой хранения тритов (задача 0057).
//
// Ядра ВТМ (TMV/MM/EMB в interp.c и backend_*.c) работают с потоком слов
// pack20: 20 тритов в u32, цифра -1/0/+1 = 0/1/2, слово = Σ digit_i·3^i.
// Так устроен и TRT1-сегмент контейнера. Это представление удобно для CPU,
// но не единственно возможное: троичная память или другое железо могут
// хранить триты иначе (в задаче — 1 трит на байт, «native»).
//
// Здесь — граница между «массивом тритов» и его физическим хранением:
//
//   vtm_trit_store  — невладеющее представление массива: storage, число
//                     тритов, указатель на данные, масштабы и блок масштаба;
//   fetch(i)        — трит по индексу;
//   count()         — сколько тритов;
//   scale_of(i)     — масштаб блока, в который попадает трит i.
//
// Загрузчик (loader.c) разбирает файл через этот контракт и на границе
// нормализует всё в pack20 в арене, поэтому интерпретатор о хранении не знает
// и правок в interp.c/backend_*.c не требует. Прямое чтение native ядрами
// (без нормализации) — отдельная задача: она меняет горячий цикл.
//
// Формат в контейнере (docs/trit-format.md §7, docs/vtm.md «Раскладки тритов»):
//
//   * packed20 (0, по умолчанию) — TRT1 version 1, слова pack20;
//   * native   (1)               — TRT1 version 2, flags биты 1..2 = storage,
//                                  по 1 знаковому байту на трит (-1/0/+1).
//
// Идентификатор хранения дублируется в flags записи тензора (биты 1..2, бит 0
// занят KV_CACHE). Старые контейнеры присылают 0 — это packed20, поэтому
// чтение старых файлов не меняется. Native-контейнер помечен TRT1 version 2:
// загрузчик до 0057 отвергает его с VTM_E_VERSION, а не читает байты как
// слова pack20 (тихая порча весов была бы хуже отказа).
#ifndef VTM_TRIT_H
#define VTM_TRIT_H

#include <stdint.h>

// --- идентификаторы хранения ----------------------------------------------
#define VTM_TRIT_STORAGE_PACKED20 0u
#define VTM_TRIT_STORAGE_NATIVE   1u
#define VTM_TRIT_STORAGE_MAX      VTM_TRIT_STORAGE_NATIVE

// Биты storage в flags записи тензора (поле 50 таблицы; бит 0 — KV_CACHE).
#define VTM_TENSOR_FLAG_STORAGE_SHIFT 1
#define VTM_TENSOR_FLAG_STORAGE_MASK  0x0006u

// kind 4 (int8 + масштабы, задача 0044) пока не объявлен в публичном
// vtm.h: enum vtm_tensor_kind там заканчивается на BF16. Константа живёт
// здесь, чтобы загрузчик мог разобрать запись, не трогая публичный заголовок.
#define VTM_TENSOR_INT8 4

// Тензор пришёл из int8-сегмента и в памяти материализован как fp32
// (интерпретатор знает только kind 0..3). Бит ставит загрузчик; в файл он не
// пишется — это диагностика для хоста/тестов, а не часть формата.
#define VTM_TENSOR_FLAG_INT8_SOURCE 0x0008u

// Сколько тритов несёт одно слово native-хранения (1 трит = 1 байт).
#define VTM_NATIVE_TRITS_PER_WORD 4u

// Массив тритов в конкретном физическом хранении.
//
// Память не принадлежит структуре. Для packed20 поле words — массив uint32_t
// (n_words слов, little-endian хост: big-endian контейнеры отвергает loader),
// для native — байты (n_words слов = 4·n_words байт, знаковый int8).
// scales — уже приведённые к fp32 масштабы (у загрузчика так всегда: fp16
// конвертируется при разборе), по одному на блок scale_block.
typedef struct vtm_trit_store {
    unsigned         storage;      // VTM_TRIT_STORAGE_*
    uint32_t         n_trits;      // логическое число тритов
    uint32_t         n_words;      // физических слов по 4 байта
    const void      *words;        // uint32_t[] (packed20) или uint8_t[] (native)
    const float     *scales;       // fp32-масштабы, может быть NULL
    uint32_t         n_scales;
    uint32_t         scale_block;  // 128 (канон VTM) или 160 (docs)
} vtm_trit_store;

// Сколько 4-байтовых слов занимает массив из n_trits в хранении storage.
// packed20 округляет до 160-тритовых (32-байтовых) блоков, native — до слова.
uint32_t vtm_trit_storage_words(uint32_t n_trits, unsigned storage);

// То же в байтах.
uint32_t vtm_trit_storage_bytes(uint32_t n_trits, unsigned storage);

// storage из flags записи тензора и обратно (обратно — только биты storage).
unsigned vtm_trit_storage_from_flags(uint16_t flags);
uint16_t vtm_trit_storage_entry_flags(unsigned storage);

// Инициализировать представление. data — слова (packed20) или байты (native),
// n_units — число 4-байтовых слов в data. Проверяет согласованность размеров и
// масштабов; 0 — успех, -1 — аргументы не сходятся.
int vtm_trit_store_init(vtm_trit_store *store, unsigned storage, const void *data,
                        uint32_t n_units, uint32_t n_trits,
                        const float *scales, uint32_t n_scales,
                        uint32_t scale_block);

// Трит по индексу: 0 — успех (значение в *trit), -1 — индекс вне массива или
// испорченное значение native (не -1/0/+1).
int vtm_trit_store_fetch(const vtm_trit_store *store, uint32_t index,
                         int8_t *trit);

// Число тритов; 0 — пустой/неинициализированный store.
int vtm_trit_store_count(const vtm_trit_store *store);

// Масштаб блока, в который попадает трит index: 0 — успех, -1 — индекс вне
// массива или масштаба нет.
int vtm_trit_store_scale_of(const vtm_trit_store *store, uint32_t index,
                            float *scale);

// Развернуть весь массив в pack20 (по 20 тритов на слово, хвост — нули).
// out_words — не меньше vtm_trit_storage_words(n_trits, PACKED20).
int vtm_trit_store_to_packed20(const vtm_trit_store *store, uint32_t *out_words,
                               uint32_t out_words_cap);

// Проверить native-байты: каждый обязан быть -1, 0 или +1.
// 0 — корректно, -1 — порча (позиция в *bad_index, если не NULL).
int vtm_trit_validate_native(const uint8_t *bytes, uint32_t n_bytes,
                             uint32_t *bad_index);

#endif // VTM_TRIT_H
