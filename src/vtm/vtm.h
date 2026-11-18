// VTM — виртуальная троичная машина, публичный API (v0.2).
//
// Единственный вход для хоста: vtm_load()/vtm_load_mem(), затем vtm_run().
// Формат контейнера описан в docs/vtm.md; решения, которых в спеке не было,
// зафиксированы в src/vtm/README.md (раздел «Соглашения v0.1»).
//
// Всё, что делает интерпретатор, — это читает тензоры и инструкции из файла,
// раскладывает данные в одну арену и исполняет инструкции по одной. Никаких
// сетевых, GPU- и стороннних зависимостей: только C99 + libm.
#ifndef VTM_H
#define VTM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// --- размеры контейнера -----------------------------------------------------
// v0.2: dims — u32×4 (ADR 0011), запись тензора 52 байта вместо 44.
#define VTM_FORMAT_VERSION   2u
#define VTM_FORMAT_VERSION_V01 1u
#define VTM_HEADER_SIZE      32u
#define VTM_TENSOR_ENTRY_SIZE 52u
#define VTM_INSTR_SIZE       16u
#define VTM_DATA_ALIGN       32u
#define VTM_ARENA_ALIGN      64u

// Максимумы нужны, чтобы кривой заголовок не приводил к гигантским calloc.
#if SIZE_MAX > 0xFFFFFFFFu
#define VTM_MAX_FILE_BYTES   ((size_t)4u << 30)   // 4 ГиБ — с запасом для v0.1
#else
#define VTM_MAX_FILE_BYTES   ((size_t)0xFFFFFFFFu)
#endif
#define VTM_MAX_TENSORS      (1u << 20)
#define VTM_MAX_INSTR        (1u << 24)
#define VTM_MAX_ELEMENTS     (1u << 30)
#define VTM_MAX_DIMS         4
#define VTM_SCALE_BLOCK_128  128u
#define VTM_SCALE_BLOCK_160  160u

// Индекс «тензора нет» в полях инструкции.
#define VTM_NO_TENSOR        0xFFFFu
// scale_offset == VTM_NO_OFFSET: масштабы лежат внутри TRT1-сегмента.
#define VTM_NO_OFFSET        0xFFFFFFFFu

// --- троичный формат -------------------------------------------------------
#define VTM_TRITS_PER_WORD   20
#define VTM_WORDS_PER_BLOCK  8
#define VTM_TRITS_PER_BLOCK  160
// 3^20 - 1; слово больше этого значения — порча данных.
#define VTM_TRIT_MAX_WORD    3486784400u

// --- коды ошибок -----------------------------------------------------------
typedef enum {
    VTM_OK = 0,
    VTM_E_ARG = -1,         // неверный аргумент API
    VTM_E_IO = -2,          // файл не открылся/не прочитался
    VTM_E_TRUNCATED = -3,   // файл короче, чем обещает заголовок
    VTM_E_MAGIC = -4,       // нет "VTM1"
    VTM_E_VERSION = -5,     // чужая версия контейнера
    VTM_E_FORMAT = -6,      // нарушен формат (таблица, TRT1, инструкции)
    VTM_E_BOUNDS = -7,      // offset/size выходят за файл
    VTM_E_NOMEM = -8,       // не хватило памяти
    VTM_E_TENSOR = -9,      // нет тензора/неподходящий kind
    VTM_E_SHAPE = -10,      // не сходятся размерности
    VTM_E_OPCODE = -11,     // неизвестный opcode
    VTM_E_UNSUPPORTED = -12,// операция/режим пока не реализованы
    VTM_E_SCALE = -13,      // испорченный масштаб/слово тритов
    VTM_E_CALLBACK = -14    // ошибка внутри callback
} vtm_status;

// Человеческая строка для кода ошибки (русская, статическая).
const char *vtm_strerror(vtm_status st);

// Ошибка последнего vtm_load()/vtm_load_mem() (статический буфер).
const char *vtm_last_error(void);

// --- тензоры ---------------------------------------------------------------
typedef enum {
    VTM_TENSOR_TRIT = 0,    // TRT1-сегмент: 160 тритов на блок + масштаб
    VTM_TENSOR_F32  = 1,
    VTM_TENSOR_F16  = 2,
    VTM_TENSOR_BF16 = 3
    // 4 (int8 + масштабы) зарезервирован, но контейнером пока не читается:
    // int8-эмбеддинги отложены вместе с загрузчиком (задача 0043).
} vtm_tensor_kind;

typedef enum {
    VTM_LAYOUT_TRT1 = 0,    // offset -> "TRT1"..., масштабы внутри сегмента
    VTM_LAYOUT_RAW  = 1     // offset -> слова, scale_offset -> масштабы
} vtm_tensor_layout;

typedef struct {
    char name[24];          // NUL-терминированное имя из таблицы
    uint8_t kind;           // vtm_tensor_kind
    uint8_t ndim;           // 1..4
    uint32_t dims[4];       // dims[0] — старшая ось, dims[ndim-1] — младшая
    uint32_t n_elements;    // произведение dims[0..ndim-1]
    uint32_t offset;        // от начала секции данных, кратен 32
    uint32_t scale_offset;  // VTM_NO_OFFSET для встроенного TRT1
    uint32_t n_blocks;      // kind == TRIT: блоков хранения по 160 тритов
    uint32_t n_scales;      // kind == TRIT: число масштабов
    uint32_t scale_block;   // kind == TRIT: 128 (VTM/Python) или 160 (docs)
    uint8_t scale_is_f32;   // только kind == TRIT (1 = fp32, 0 = fp16)
    uint8_t layout;         // vtm_tensor_layout
    uint8_t reserved[2];
    // Для kind 1..3 — распакованные значения в арене (fp32, host-endian).
    float *f32;
    // Для kind 0 — слова pack20 в арене; масштабы уже приведены к fp32.
    const uint32_t *words;
    const float *scales;
} vtm_tensor;

// --- инструкция ------------------------------------------------------------
//
// EMB (0x09): a — таблица словаря [vocab, dim]. Годится любой kind 0..3:
//   kind 0 (триты) читается через LUT243 (5 тритов на выборку из 243),
//   fp16/bf16 загрузчик уже привёл к fp32 в арене.
// MM (0x02): A может быть тритовой матрицей — так tied lm_head считает
//   логиты по emb.w (A = [vocab, hidden], B = [hidden, 1]); триты
//   расквантовываются на месте (trit * scale), без копии матрицы в fp32.
typedef struct {
    uint8_t opcode;
    uint8_t flags;
    uint16_t dst;
    uint16_t a;
    uint16_t b;
    uint16_t c;
    int32_t imm;
} vtm_instr;

const char *vtm_opcode_name(uint8_t opcode);

typedef enum {
    VTM_OP_TMV   = 0x01,
    VTM_OP_MM    = 0x02,
    VTM_OP_RMSN  = 0x03,
    VTM_OP_ROPE  = 0x04,
    VTM_OP_SM    = 0x05,
    VTM_OP_SILU  = 0x06,
    VTM_OP_MUL   = 0x07,
    VTM_OP_ADD   = 0x08,
    VTM_OP_EMB   = 0x09,
    VTM_OP_KVW   = 0x0A,
    VTM_OP_ATTN  = 0x0B,
    VTM_OP_SAMP  = 0x0C,
    VTM_OP_CPY   = 0x0D,
    VTM_OP_SCAL  = 0x0E,
    VTM_OP_HALT  = 0x0F
} vtm_opcode;

// --- SAMP: стратегии и параметры ------------------------------------------
//
// imm инструкции SAMP — номер стратегии:
//   0 — argmax (жадно; поведение v0.1 не изменилось);
//   1 — temperature: p = softmax(logits / temperature);
//   2 — top-k: температура, затем k старших логитов (top_k <= 0 — не отсекать);
//   3 — top-p: температура, затем минимальное ядро с суммой вероятностей
//       >= top_p (top_p >= 1 — не отсекать, top_p <= 0 — жадно).
// imm вне 0..3 — VTM_E_UNSUPPORTED: старый Python-эталон трактовал imm > 0 как
// temperature = imm/1000, C-интерпретатор эту конвенцию не наследует (задача
// 0051), поэтому байткод с такими imm надо перекомпилировать.
//
// Параметры читаются из JSON-меты контейнера; ключ ищется на любом уровне
// вложенности (как rope_theta), при отсутствии/порче — дефолт:
//   temperature        double, 1.0;  0 и меньше — жадно (детерминизм);
//   top_k              int,    0;    не отсекать (стратегия 2);
//   top_p              double, 1.0;  не отсекать (стратегия 3);
//   repetition_penalty double, 1.0;  1.0 — штраф выключен;
//   repetition_window  int,    64;   окно последних токенов, 0 — выключить,
//                                    максимум VTM_MAX_REPETITION_WINDOW;
//   sample_seed        int,    0;    зерно ГПСЧ (детерминизм/воспроизводимость).
// Штраф за повторы действует во всех стратегиях и применяется до температуры:
// для каждого токена окна, встретившегося среди последних repetition_window
// сгенерированных токенов, логит > 0 делится на penalty, иначе умножается.
#define VTM_SAMP_ARGMAX      0
#define VTM_SAMP_TEMPERATURE 1
#define VTM_SAMP_TOPK        2
#define VTM_SAMP_TOPP        3

#define VTM_DEFAULT_TEMPERATURE        1.0
#define VTM_DEFAULT_TOP_P              1.0
#define VTM_DEFAULT_REPETITION_PENALTY 1.0
#define VTM_DEFAULT_REPETITION_WINDOW  64
#define VTM_MAX_REPETITION_WINDOW      256

// Флаги локальны для opcode (один и тот же бит значит разное).
#define VTM_FLAG_CAUSAL          0x01u  // SM, ATTN
#define VTM_FLAG_INPLACE         0x02u  // подсказка: dst может совпадать с a
#define VTM_FLAG_EMB_INPUT       0x01u  // EMB: взять токен из vtm_set_token()
#define VTM_FLAG_ROPE_INTERLEAVED 0x02u // ROPE: пары (2i, 2i+1), иначе половинки
// TMV: поле a задано в транспонированном виде [cols, rows] (например,
// lm_head = Eᵀ при tied_embeddings хранится как [hidden, vocab]).
#define VTM_FLAG_TMV_TRANSPOSED  0x04u

// Флаги заголовка контейнера.
#define VTM_HEADER_FLAG_F32_SCALES       0x0001u // raw-триты: масштабы fp32
// Расширение C: явно сказать, что неоднозначный n_scale — это блок 160.
// Python-эталон этот бит не ставит (у него канон 128) и не проверяет flags.
#define VTM_HEADER_FLAG_SCALE_BLOCK_160  0x8000u

// --- бэкенд ----------------------------------------------------------------
// Контракт из docs/vtm.md: поле NULL — интерпретатор берёт скалярную версию.
// tmv: w — поток слов pack20 (32 байта = 160 тритов), scales — масштабы,
// x — строка длины cols, y — rows. tmv_blocks — то же, но с явным размером
// блока масштаба (128 — канон Python-эталона, 160 — docs/trit-format).
// Старый tmv считает блок 160; если tmv_blocks == NULL, интерпретатор
// использует tmv только для 160 и падает обратно на скаляр для 128.
typedef struct vtm_backend {
    const char *name;
    int (*tmv)(float *y, const uint8_t *w, const float *scales,
               const float *x, int rows, int cols);
    int (*rmsnorm)(float *y, const float *x, const float *w, int n, float eps);
    int (*silu)(float *y, const float *x, int n);
    int (*mul)(float *y, const float *a, const float *b, int n);
    int (*add)(float *y, const float *a, const float *b, int n);
    int (*scal)(float *y, const float *x, float k, int n);
    int (*softmax)(float *y, const float *x, int rows, int cols,
                   int causal, int offset);
    int (*emb)(float *y, const float *e, int dim, int token);
    int (*copy)(float *y, const float *x, int n);
    // Расширение v0.1: TMV с явным блоком масштаба (128/160). В конце
    // структуры, чтобы позиционные инициализаторы из спеки не сдвигались.
    int (*tmv_blocks)(float *y, const uint8_t *w, const float *scales,
                      const float *x, int rows, int cols, int scale_block);
} vtm_backend;

// Два скалярных варианта TMV: LUT на 5 тритов (243 записи) и прямая
// распаковка делением на 3. Различие только в tmv; остальные ядра общие.
extern const vtm_backend vtm_backend_scalar_lut;
extern const vtm_backend vtm_backend_scalar_direct;

// --- машина ----------------------------------------------------------------
typedef struct vtm vtm_t;

// Эффективные параметры SAMP из меты контейнера (для отладки, тестов и
// логов хоста). Заполняется vtm_sampling_params(); при отсутствии меты —
// дефолты VTM_DEFAULT_*. Разбор — линейный проход по JSON, не для горячего
// цикла; сам интерпретатор держит разобранные параметры в машине.
typedef struct {
    double   temperature;
    int      top_k;
    double   top_p;
    double   repetition_penalty;
    int      repetition_window;   // 0 — штраф выключен
    uint64_t sample_seed;
    int      has_seed;            // 1 — sample_seed задан в мете
} vtm_sampling;

// 1 — мета есть и разобрана; 0 — меты нет, поля заполнены дефолтами.
int vtm_sampling_params(const vtm_t *m, vtm_sampling *out);

// Callback получает выбранный SAMP токен и позицию (m->position).
// Возврат != 0 — вежливая остановка прогона (vtm_run вернёт VTM_OK).
typedef int (*vtm_emit_fn)(void *user, int token, int position);

// Загрузить .vtm из файла. NULL — ошибка; причина в vtm_last_error().
vtm_t *vtm_load(const char *path);

// То же из памяти. data не забирается и может быть освобождён сразу после
// возврата. status/errbuf могут быть NULL.
vtm_t *vtm_load_mem(const void *data, size_t size,
                    vtm_status *status, char *errbuf, size_t errcap);

void vtm_free(vtm_t *m);

uint32_t vtm_tensor_count(const vtm_t *m);
const vtm_tensor *vtm_tensor_by_index(const vtm_t *m, uint32_t index);
const vtm_tensor *vtm_tensor_by_name(const vtm_t *m, const char *name);

uint32_t vtm_instr_count(const vtm_t *m);
const vtm_instr *vtm_instruction(const vtm_t *m, uint32_t index);

// Бэкенд по умолчанию — vtm_backend_scalar_lut; NULL возвращает скаляр.
void vtm_set_backend(vtm_t *m, const vtm_backend *backend);
const char *vtm_backend_name(const vtm_t *m);

// Входной токен для EMB с флагом VTM_FLAG_EMB_INPUT. -1 — не задан.
void vtm_set_token(vtm_t *m, int token);
int vtm_input_token(const vtm_t *m);

// База позиции: ROPE/KVW/ATTN/SM считают позицию как m->position + imm.
// Хост двигает её между вызовами vtm_run(); внутри прогона она не меняется.
void vtm_set_position(vtm_t *m, int position);
int vtm_position(const vtm_t *m);
int vtm_generated(const vtm_t *m);

// Сбросить позицию, входной токен, счётчик сгенерированных, halted и ошибку.
// Тензоры и KV-кэш не обнуляются — это делает вызывающий, если нужно.
void vtm_reset(vtm_t *m);

vtm_status vtm_run(vtm_t *m, vtm_emit_fn emit, void *user);

// Текст ошибки последнего прогона (буфер внутри машины).
const char *vtm_error_message(const vtm_t *m);
int vtm_halted(const vtm_t *m);

// --- метаданные контейнера -------------------------------------------------
// JSON-метаданные лежат за секцией данных, перед программой (docs/vtm.md,
// «Метаданные»). Загрузчик читает их терпимо: нет меты, битый JSON, битые
// смещения — это не ошибка загрузки, а откат на дефолты. Читаются ключи,
// влияющие на исполнение: rope_theta, rope_layout, vocab_size,
// tied_embeddings; ключи ищутся на любом уровне вложенности (в том числе в
// meta["model"]).
typedef enum {
    VTM_ROPE_LAYOUT_HALF = 0,        // пары (i, i + head_dim/2) — Qwen/Llama
    VTM_ROPE_LAYOUT_INTERLEAVED = 1  // пары (2i, 2i+1) — GPT-NeoX
} vtm_rope_layout;

#define VTM_DEFAULT_ROPE_THETA 10000.0

// 1 — JSON-объект разобран; 0 — меты нет или она битая (действуют дефолты).
int vtm_meta_present(const vtm_t *m);
// theta для ROPE; без меты или с некорректным значением — 10000.
double vtm_meta_rope_theta(const vtm_t *m);
// vtm_rope_layout; без меты — VTM_ROPE_LAYOUT_HALF.
int vtm_meta_rope_layout(const vtm_t *m);
// vocab_size из меты; 0 — не задан (размер брать из dims тензора EMB).
uint32_t vtm_meta_vocab_size(const vtm_t *m);
// 1 — lm_head разделяет таблицу с эмбеддингами (tied); 0 — нет/неизвестно.
int vtm_meta_tied_embeddings(const vtm_t *m);
// Сырой JSON (NUL-терминированный) или ""; живёт до vtm_free().
const char *vtm_meta_json(const vtm_t *m);

// --- кодек тритов ----------------------------------------------------------
// Контрольные векторы docs/trit-format.md:
//   все -1 -> 0x00000000, все 0 -> 0x67EA0DC8,
//   все +1 -> 0xCFD41B90, (+1,-1)*10 -> 0x33F506E4.
uint32_t vtm_trit_pack20(const int8_t trits[20]);
uint32_t vtm_trit_pack(const int8_t *trits, int n); // n <= 20, добор нулями
void vtm_trit_unpack20(uint32_t word, int8_t out[20]);

// Развернуть n_trits из потока слов. 0 — успех, -1 — порча/переполнение.
int vtm_trit_decode_words(const uint32_t *words, uint32_t n_words,
                          int8_t *out, uint32_t n_trits);

// Развернуть тензор kind==TRIT (n_trits <= n_elements).
int vtm_tensor_unpack_trits(const vtm_tensor *t, int8_t *out, uint32_t n_trits);

// IEEE 754 binary16 / bfloat16 <-> float. Нужны загрузчику и тестам.
float vtm_half_to_float(uint16_t h);
uint16_t vtm_float_to_half(float f);
float vtm_bf16_to_float(uint16_t b);
uint16_t vtm_float_to_bf16(float f);

#ifdef __cplusplus
}
#endif

#endif // VTM_H
