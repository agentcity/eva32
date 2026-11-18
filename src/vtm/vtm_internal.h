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

    // --- mmap (задача 0060) -------------------------------------------------
    // vtm_load() отображает файл read-only и оставляет неизменяемые тензоры
    // (fp32 и триты) прямо в отображении: копий в арене для них нет. Арена
    // хранит только то, что интерпретатор пишет (dst, V-кэш KVW), и fp16/bf16
    // (их нужно развернуть в fp32 — другого представления API не знает).
    // vtm_load_mem() и фолбэк (VTM_NO_MMAP=1, Windows) отображения не делают:
    // там map == NULL, а все тензоры по-прежнему копируются в арену.
    void           *map;        // база mmap или NULL
    size_t          map_size;   // размер отображения (== размер файла)
    size_t          data_start; // начало секции данных в файле
    int             mapped;     // 1 — файл отображён

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

    // --- окно KV-кэша (задача 0076) ----------------------------------------
    // Эффективное окно attention: min(длина кэша, meta["context_window"]).
    // 0 — ключ ещё не разбирали, <0 — в мете его нет/значение битое (окно =
    // длина кэша). Разбирается лениво в interp.c (kv_window), loader.c не
    // меняется: поле обнуляется calloc'ом при загрузке.
    int             ctx_window;

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

    // --- буфер сэмплера (задача 0059) --------------------------------------
    //
    // Словарь Qwen — 150 000 токенов, поэтому массив кандидатов на каждый
    // SAMP весит мегабайты. Держим его в машине: одно выделение на машину,
    // дальше только переиспользование (рост — если словарь стал больше).
    // Раскладка одного блока: [кандидаты n×16][копия n×16 → веса][метки n×4]
    // и, для словаря ≥ 4096, [счётчики radix 256 КБ].
    void           *samp_buf;         // база блока (malloc/realloc), NULL до 1-го SAMP
    double         *samp_w;           // веса, внутри samp_buf (== копия кандидатов)
    uint32_t       *samp_stamp;       // метки окна повторов, внутри samp_buf
    uint32_t       *samp_count;       // счётчики radix (16 бит), внутри samp_buf
    uint32_t        samp_buf_cap;     // ёмкость в кандидатах (0 — буфера нет)
    uint32_t        samp_stamp_gen;   // поколение меток (0 — метки пусты)
    size_t          samp_buf_bytes;   // сколько байт выделено
    size_t          samp_buf_allocs;  // выделений за жизнь машины (тест-хук)

    // --- кэш префикса (задача 0081) ----------------------------------------
    //
    // K/V префикса — это уже содержимое KV-тензоров (KVW), поэтому кэш хранит
    // не копию K/V, а снимок всего остального состояния: идентификатор
    // префикса (sha256 от ids), позицию и состояние сэмплера. Попадание
    // восстанавливает снимок, и прогон продолжается с позиции cache_len так,
    // как будто префикс только что посчитан.
    //
    // Снимок хранится прямо в машине: он маленький (~1,2 КБ) и фиксированный,
    // а выделять память нельзя — vtm_free() в loader.c (вне зоны задачи) её
    // не освободит. Поля обнуляются calloc'ом в vtm_load_mem(); cache_len ==
    // 0 и cache_valid == 0 означают «кэша нет».
    int             cache_valid;      // 1 — снимок описывает префикс
    int             cache_len;        // длина префикса в токенах
    uint8_t         cache_digest[32]; // sha256 от ids[0..cache_len)
    const vtm_backend *cache_backend; // бэкенд, которым считан префикс
    int             cache_generated;  // m->generated после префикса
    int             cache_position;   // m->position после префикса
    int             cache_input;      // m->input_token после префикса
    int             cache_samp_ready;
    uint64_t        cache_samp_rng;
    uint64_t        cache_samp_seed;
    double          cache_samp_temperature;
    double          cache_samp_top_p;
    double          cache_samp_repetition_penalty;
    int             cache_samp_top_k;
    int             cache_samp_repetition_window;
    int             cache_samp_prev_generated;
    int             cache_samp_hist_len;
    int             cache_samp_hist_head;
    int             cache_samp_hist[VTM_MAX_REPETITION_WINDOW];
    uint64_t        cache_hits;       // попаданий vtm_run_cached()
    uint64_t        cache_misses;     // полных прогонов vtm_run_cached()
};

// Записать ошибку в машину и в глобальный буфер vtm_last_error().
void vtm_set_error(vtm_t *m, vtm_status st, const char *fmt, ...);

const char *vtm_op_name(uint8_t opcode);

// Сбросить состояние сэмплера SAMP (окно повторов и ГПСЧ). Предназначено
// для vtm_reset(); без вызова сброс всё равно распознаётся по vtm_generated().
void vtm_samp_reset(vtm_t *m);

// Освободить буфер кандидатов SAMP (задача 0059). Вызывать из vtm_free()
// (loader.c); файлы зон не пересекаются, поэтому пока это делает только тест.
void vtm_samp_free(vtm_t *m);

// Тест-хуки буфера SAMP: сколько раз выделяли (норма — 1 на машину) и
// сколько байт занимает буфер.
size_t vtm_samp_alloc_count(const vtm_t *m);
size_t vtm_samp_buffer_bytes(const vtm_t *m);

// Тест-хук буфера SAMP: заполнить метки окна значением gen и сбросить
// счётчик поколений в 0 — имитация памяти, переиспользованной после
// vtm_samp_free(). Норма: следующий SAMP метки чистит, штраф идёт только по
// реальному окну (регрессия «новая машина штрафует чужие токены»).
void vtm_samp_poison_stamps(vtm_t *m, uint32_t gen);

// --- отладка загрузчика (задача 0060) --------------------------------------
// Не публичный API: нужны тестам, чтобы отличить mmap-путь от фолбэка, не
// заглядывая в struct vtm (она непрозрачна для пользователя vtm.h).
int vtm_internal_file_is_mapped(const vtm_t *m);
int vtm_internal_tensor_is_mapped(const vtm_t *m, uint32_t index);

#endif // VTM_INTERNAL_H
