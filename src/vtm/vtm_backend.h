// VTM — SIMD-бэкенд: объявления (без определений).
//
// Здесь только декларации: реализация ядер — src/vtm/backend_neon.c (NEON,
// ARM64; на других арх — только диспетчер с откатом на scalar) и
// src/vtm/backend_avx2.c (AVX2, x86-64; собирается Makefile по uname -m).
//
// Контракт `vtm_backend` (см. vtm.h/docs/vtm.md): любое поле NULL —
// интерпретатор берёт скалярную реализацию. `vtm_backend_simd` никогда не
// роняет прогон: если SIMD в сборке нет или CPU его не умеет, его ядра сами
// вызывают скалярные (см. vtm_simd_available()).
#ifndef VTM_BACKEND_H
#define VTM_BACKEND_H

#include "vtm.h"

#ifdef __cplusplus
extern "C" {
#endif

// Готовый SIMD-бэкенд. Имя: "simd-neon", "simd-avx2" или "simd" (сборка без
// SIMD; ядра при этом работают как скалярные).
extern const vtm_backend vtm_backend_simd;

// --- Metal-бэкенд, фаза 1 (задача 0192, ADR 0021) --------------------------
//
// Отдельный исполнитель на GPU: GEMV для kind 6 (Q4_K), kind 7 (Q6_K) и
// int8-таблицы словаря (kind 4). Остальные операции (attention, KV, RMSNorm,
// softmax, триты) остаются на CPU — бэкенд отдаёт те же ядра, что simd, и
// подменяет только перечисленные GEMV. Веса не копируются: MTLBuffer
// создаётся на выровненный срез mmap-региона контейнера (zero-copy).
//
// Включение: VTM_BACKEND=metal (всё на GPU) или hybrid (0196/0198: GPU только
// на батчевые GEMM prefill); см. vtm_backend_by_name. auto с 0198 — hybrid,
// если устройство есть и парный замер подтвердил выигрыш (см. docs/vtm.md);
// иначе auto — simd. Одиночные ядра инициализируют GPU лениво, при первом
// вызове, и при отказе прогон уходит на CPU.
// Любая ошибка Metal (нет устройства, не собрался MSL, ошибка командного
// буфера) — лог в stderr и откат на NEON/scalar: ядра возвращают -1,
// вызывающий идёт прежним путём. Побитового паритета с CPU нет (GPU меняет
// порядок суммирования): контракт точности — argmax + Δlogits ≤ 1e-2.
//
// Сборка без Metal (не Darwin или VTM_WITH_METAL=0): backend_metal.m не
// компилируется, а backend_scalar.c даёт заглушки Metal-API (ядра → -1).
//
// Режим (задача 0198). GPU выигрывает только на батчевом GEMM (prefill), а на
// одиночных GEMV декода проигрывает CPU в 2,3 раза (замер 0196). Поэтому у
// Metal три режима:
//   VTM_METAL_OFF   — Metal выключен (simd/scalar);
//   VTM_METAL_FULL  — все GEMV kind 4/6/7 и батчевые GEMM на GPU
//                     (VTM_BACKEND=metal);
//   VTM_METAL_BATCH — только батчевые GEMM prefill на GPU, одиночные GEMV
//                     декода — NEON (VTM_BACKEND=hybrid).
// Развилка живёт в диспетчере, а не в ядре: vtm_metal_active() истинно для
// FULL и BATCH (его спрашивают батчевые пути interp.c), а
// vtm_metal_gemv_active() — только для FULL (поштучные GEMV в
// vtm_q4k_tmv/vtm_q6k_tmv/vtm_i8_gemv и сессия отложенной очереди). Так
// «не умею» не нужно возвращать из ядра: Metal просто не вызывается.
#define VTM_METAL_OFF   0
#define VTM_METAL_FULL  1
#define VTM_METAL_BATCH 2
int vtm_metal_available(void);       // 1 — устройство есть (шейдеры ещё нет)
int vtm_metal_compiled(void);        // 1 — сборка с Metal (без проверки устройства)
int vtm_metal_active(void);          // 1 — режим FULL или BATCH
int vtm_metal_gemv_active(void);     // 1 — только FULL: поштучные GEMV на GPU
int vtm_metal_mode(void);            // VTM_METAL_OFF/FULL/BATCH
void vtm_metal_set_mode(int mode);   // выбрать режим (см. выше)
void vtm_metal_set_active(int on);   // on — FULL, 0 — OFF (прежний контракт)
const vtm_backend *vtm_backend_metal(void); // .name == "metal" + ядра simd
// Гибрид: .name == "hybrid", ядра — simd (поштучные операции на CPU), а
// батчевые GEMM interp.c уходят на GPU по vtm_metal_active(). Тот же ввод,
// что у vtm_backend_metal(), но режим BATCH; без Metal — NULL/откат.
const vtm_backend *vtm_backend_hybrid(void);

// GEMV-ядра. 0 — успех, -1 — отказ (сборка/устройство/форма/память): тогда
// диспетчер backend_scalar.c зовёт CPU-ядро, прогон не падает.
int vtm_metal_q4k_gemv(float *y, const uint8_t *blocks, const float *x,
                       int rows, int cols);
int vtm_metal_q6k_gemv(float *y, const uint8_t *blocks, const float *x,
                       int rows, int cols);
int vtm_metal_i8_gemv(float *y, const int8_t *q, const float *scales,
                      const float *x, int rows, int cols);

// --- фаза 2 (задача 0196, ADR 0021 §4) --------------------------------------
//
// Батчевый GEMM prefill: n_tokens токенов пачки против одной матрицы, один
// dispatch на матрицу (внутри — chunk'и до 32 токенов). Раскладка — как у
// батчевых ядер interp.c:
//   x[t * n_tokens + j] — вход токена j по индексу t;
//   y[r * n_tokens + j] — выход токена j по строке r.
// kind: 0 — Q4_K, 1 — Q6_K, 2 — int8-таблица словаря (kind 4).
// 0 — успех, -1 — отказ (нет Metal/шейдера, форма, VTM_Q4K_STRICT): вызывающий
// обязан пойти прежним CPU-путём, прогон не падает.
int vtm_metal_qk_gemm(int kind, float *y, const void *w, const float *scales,
                      const float *x, int rows, int cols, int n_tokens);

// --- 0214: батчевый GEMM в арену пачки --------------------------------------
//
// Тот же GEMM, но вход и выход — со шагом на токен (x_stride/y_stride в
// float), а не упакованные: interp.c держит активации пачки в одной mmap-арене,
// Metal берёт её zero-copy буфером, поэтому ни транспонирования входа, ни
// разноски выхода на хосте нет. x_stride = 0 — один вектор на все токены.
//
// dispatch'и копятся в ОДНОМ командном буфере (vtm_metal_gemm_flush): подряд
// идущие матрицы слоя (q/k/v, gate/up) считаются без ожидания между ними.
// 0 — поставлено в очередь (числа готовы после flush); 1 — не взялся (старый
// metallib, форма, выравнивание, буфер не zero-copy) — считать CPU-путём;
// -1 — Metal отказал (тоже CPU-путь). Пока очередь не закрыта, арену читать
// нельзя: её пишет GPU.
int vtm_metal_qk_gemm_arena(int kind, const void *w, const float *scales,
                            int rows, int cols, int n_tokens,
                            const float *x, int x_stride,
                            float *y, int y_stride);
// Закрыть группу и дождаться. 0 — всё посчитано (или очередь пуста),
// -1 — командный буфер не завершился (результаты пачки недостоверны).
int vtm_metal_gemm_flush(void);
// 0214: attention пачки — C независимых «декодных» вниманий (у токена i
// видимы ключи до pos0+i включительно). q/out — по слоту на токен (stride),
// K/V — кольцевой кэш модели. 0 — dispatch в той же очереди, что GEMM
// (один командный буфер на группу), 1 — не взялся, -1 — Metal отказал.
int vtm_metal_attn_prefill(float *out, const float *q, const float *kc,
                           const float *vc, int n_heads, int n_kv_heads,
                           int head_dim, int kv_dim, int win, int n_tokens,
                           int pos0, int q_stride, int out_stride);
// Сколько токенов-«пустышек» нужно арене в хвосте: тайловое ядро варианта NT
// читает/пишет ровно NT токенов, даже если в хвосте пачки их меньше.
int vtm_metal_gemm_pad(void);
// Привязать арену пачки к Metal один раз на прогон (zero-copy MTLBuffer поверх
// странично-выровненной памяти). Пока арена привязана, vtm_metal_qk_gemm_arena
// передаёт в ядро только смещение тензора — иначе пришлось бы создавать
// no-copy буфер на каждую матрицу (драйвер отображает тысячи страниц, это
// дороже самого GEMM). 0 — привязана, -1 — не вышло (пачка пойдёт CPU-путём).
int vtm_metal_batch_arena(const void *ptr, size_t bytes);
// Отпустить арену перед free()/munmap её памяти (ptr == NULL — все).
void vtm_metal_batch_arena_drop(const void *ptr);

// Отложенная очередь декода (фаза 2): interp.c собирает подряд идущие GEMV в
// один MTLCommandBuffer и ждёт результат только когда он нужен CPU. Между
// begin/end сессией владеет один поток; вторая попытка begin вернёт -1 и
// вызывающий идёт синхронным путём (одиночный GEMV выше). Сцены две
// (ping-pong): пока GPU считает одну, CPU кодирует другую; ожидание —
// dispatch_semaphore, а не waitUntilCompleted.
//
// vtm_metal_defer_gemv: 0 — dispatch поставлен в очередь; 1 — не отложено
// (форма/порог/нет сессии/strict), считайте CPU-путём; -1 — Metal ошибся
// (тоже CPU-путь). Результаты попадают в y после vtm_metal_defer_wait/end.
// vtm_metal_defer_submit отдаёт накопленную сцену GPU, не дожидаясь её
// (CPU может делать независимую работу); vtm_metal_defer_wait ждёт все
// отданные сцены и разносит результаты по y; -1 — буфер не завершился.
int vtm_metal_defer_begin(void);
int vtm_metal_defer_active(void);
int vtm_metal_defer_gemv(int kind, float *y, const void *w,
                         const float *scales, const float *x, int rows,
                         int cols);
int vtm_metal_defer_submit(void);
int vtm_metal_defer_wait(void);
int vtm_metal_defer_end(void);

// --- фаза 3 (задача 0204): граф одного токена ------------------------------
//
// Весь токен декода кодируется в ОДИН MTLCommandBuffer: GEMV (kind 4/6/7),
// RMSNorm/RMSH, RoPE, SiLU, MUL, ADD, KVW и attention декода — MSL-ядрами.
// CPU ждёт один раз, перед сэмплингом. Один поток-владелец; вторая
// одновременная сессия получает -1 и идёт прежним (defer/CPU) путём.
//
// Все функции возвращают 0 — dispatch поставлен, -1 — граф не может (нет
// ядра/буфера zero-copy/форма): interp.c выбрасывает недоотправленный
// командный буфер (он не закоммичен, GPU ничего не записал) и пересчитывает
// токен CPU-путём. vtm_metal_graph_end после ошибки всегда -1.
int vtm_metal_graph_available(void);
int vtm_metal_graph_active(void);
int vtm_metal_graph_begin(void);
int vtm_metal_graph_end(void);
int vtm_metal_graph_gemv(int kind, float *y, const void *w,
                         const float *scales, const float *x, int rows,
                         int cols);
int vtm_metal_graph_norm(float *y, const float *x, const float *w, int dim,
                         int rows, float eps);
int vtm_metal_graph_rope(float *y, const float *x, int heads, int head_dim,
                         int interleaved, int pos, double theta);
int vtm_metal_graph_silu(float *y, const float *x, int n);
int vtm_metal_graph_copy(float *y, const float *x, int n);
int vtm_metal_graph_mul(float *y, const float *a, const float *b, int n);
int vtm_metal_graph_add(float *y, const float *a, const float *b, int n);
int vtm_metal_graph_kvw(float *kc, const float *k, float *vc, const float *v,
                        int kv_dim, int slot);
int vtm_metal_graph_attn(float *out, const float *q, const float *kc,
                         const float *vc, int n_heads, int n_kv_heads,
                         int head_dim, int kv_dim, int win, int n_keys,
                         int slot0);

// Статистика для бенчмарка/отчёта: время инициализации (в т.ч. компиляции
// MSL), число вызовов/отказов, сколько байт весов привязано zero-copy и
// сколько скопировано, currentAllocatedSize до/после инициализации. Поля
// дополняются нулями; last_error — текст последней ошибки ("" — не было).
typedef struct vtm_metal_stats {
    int      available;
    int      active;
    int      mode;          // VTM_METAL_OFF/FULL/BATCH на момент опроса
    uint64_t init_ms;       // вся инициализация (устройство + MSL + pipeline)
    uint64_t compile_ms;    // только загрузка/компиляция библиотеки шейдеров
    uint64_t calls;         // успешных GEMV
    uint64_t failures;      // отказов Metal (ядро вернуло -1)
    uint64_t buffers;       // записей в кэше MTLBuffer
    uint64_t nocopy_bytes;  // байт весов в zero-copy буферах (не копия!)
    uint64_t copy_bytes;    // байт, скопированных в Metal (fallback/тесты)
    uint64_t alloc_before;  // MTLDevice.currentAllocatedSize до init
    uint64_t alloc_after;   // ... сразу после init
    uint64_t current_allocated; // ... сейчас
    uint64_t gemm_calls;    // батчевых GEMM prefill
    uint64_t defer_ops;     // GEMV, отложенных в общий командный буфер
    uint64_t defer_waits;   // ожиданий отданных буферов
    uint64_t wait_ns;       // суммарное время ожидания GPU
    uint64_t gpu_ns;        // 0214: суммарное GPU-время командных буферов
    uint64_t graph_tokens;  // токенов, посчитанных одним графом (0204)
    uint64_t graph_ops;     // dispatches в графах
    uint64_t graph_fail;    // неудачных графов (откат на CPU)
    uint64_t graph_encode_ns; // время CPU-кодирования графов (не GPU)
    char     last_error[256];
} vtm_metal_stats;

void vtm_metal_stats_get(vtm_metal_stats *out);
// Порог MAC, ниже которого GEMV остаётся на CPU (накладные Metal на мелких
// формах больше выигрыша; ADR 0021 §5). По умолчанию 1M MAC; VTM_METAL_MIN_MAC
// переопределяет (0 — всё на GPU).
double vtm_metal_min_mac(void);
// Освободить no-copy MTLBuffer'ы перед munmap/free модели (vtm_free).
void vtm_metal_forget_all(void);
void vtm_backend_forget_metal(void);


// 1 — в сборке есть SIMD-ядра и CPU их поддерживает; 0 — будет откат на
// скалярные ядра (vtm_backend_simd при этом всё равно корректен).
int vtm_simd_available(void);

// "neon" | "avx2" | "none" — что реально используется (для логов/бенчмарка).
const char *vtm_simd_isa(void);

// Выбор бэкенда по имени: "auto" | "metal" | "hybrid" | "simd" | "scalar" |
// "direct" | "lut" (NULL/"" — то же, что VTM_BACKEND из окружения).
// Неизвестное имя — NULL.
const vtm_backend *vtm_backend_by_name(const char *name);

// Внутренний хук диспетчера: запомнить фактически выбранный бэкенд машины.
// Нужен, потому что vtm_set_backend() может сменить metal на simd/scalar уже
// после выбора из env, а vtm_q4k_*/vtm_i8_* не получают vtm_t.
void vtm_backend_note_active(const vtm_backend *backend);

// Бэкенд из env VTM_BACKEND:
//   не задан        -> auto (ниже);
//   auto            -> hybrid, если Metal собран и устройство есть, иначе
//                      simd, если доступен, иначе scalar. Решение по замеру
//                      0198: prefill 64 токена 19,6 -> 34,1 ток/с (1,74x),
//                      декод 20,4 -> 20,4 (шум), TTFT 3,27 -> 1,88 с;
//   metal           -> всё на GPU (GEMV+GEMM); без Metal — simd;
//   hybrid          -> батчевые GEMM на GPU, одиночные GEMV на NEON; без
//                      Metal — simd с предупреждением;
//   simd            -> simd (с внутренним откатом на scalar, если недоступен);
//   scalar | lut    -> scalar-lut243;
//   direct          -> scalar-direct.
// Неизвестное значение -> scalar-lut243 + предупреждение в stderr.
const vtm_backend *vtm_backend_from_env(void);

// --- батчевые ядра TMV/MM (задача 0122) ------------------------------------
//
// N токенов против одной матрицы весов: вес читается и распаковывается один
// раз на блок, поэтому N токенов дешевле N отдельных вызовов ядра (это и
// ускоряет батчевый prefill 0112 на SIMD). Раскладка активаций — как у
// батчевого пути interp.c (токен — младший индекс):
//
//   x[t * N + j] — вход токена j по индексу t (t = 0..cols-1 / 0..k-1);
//   y[r * N + j] — выход токена j по индексу r (r = 0..rows-1 / 0..m-1).
//
// Арифметика — как у батчевого скалярного ядра: сумма по блоку масштаба
// (128/160) копится в double и умножается на масштаб блока, внешние блоки
// складываются в double. Расхождение с батчевым скалярным путём — единицы
// ulp (проверяется tests/vtm/bench_backend.c, порог 1e-5).
//
// 0 — успех, -1 — неверные аргументы (NULL, отрицательные размеры, чужой
// scale_block). Ядра не роняют прогон: при отсутствии SIMD/собранного ядра
// диспетчер backend_neon.c зовёт скалярный откат (тот же результат по смыслу,
// медленнее), при n_tokens == 1 — обычное одноколоночное ядро.
//
// TMV: W[rows, cols] (триты pack20 + масштабы) * X[cols, N] -> Y[rows, N].
int vtm_simd_tmv_blocks_batch(float *y, const uint8_t *w, const float *scales,
                              const float *x, int rows, int cols,
                              int n_tokens, int scale_block);
// MM: A[m, k] (триты pack20 + масштабы) * B[k, N] -> Y[m, N]. Для тритовой A
// это та же арифметика, что у TMV (отдельное имя — ради читаемости вызова).
int vtm_simd_mm_batch(float *y, const uint8_t *w, const float *scales,
                      const float *b, int m, int k, int n_tokens,
                      int scale_block);

// Плотные (fp32) варианты — интерфейс батчевого пути interp.c: он батчит и
// нетернарные TMV/MM (эмбеддинги, tied lm_head). Ядро векторизует по токенам
// пачки: x[t*N + j], y[r*N + j] (токен — младший индекс), вес W[r,t] — скаляр,
// поэтому на один t уходит broadcast и N/4 FMA. По t копится блок 64 и
// сворачивается в double — так же, как у GEMV ниже, чтобы f32-ошибка не росла
// на длинной строке. Это не «скалярный эталон»: у 3B Q4 tied lm_head —
// плотная [151936,2048], и в prefill именно этот путь был самым дорогим
// (задача 0159).
//
// transposed != 0 — W хранится как [cols, rows] (VTM_FLAG_TMV_TRANSPOSED).
// При n_tokens == 1 зовётся vtm_simd_gemv_f32 (та же арифметика).
int vtm_simd_tmv_f32_batch(float *y, const float *w, const float *x,
                           int rows, int cols, int n_tokens, int transposed);
int vtm_simd_mm_f32_batch(float *y, const float *a, const float *b,
                          int m, int k, int n_tokens);

// --- плотный GEMV: один столбец (n == 1), задача 0159 -----------------------
//
//   y[r] = sum_t W[r,t]·x[t]        (transposed == 0, W — [rows, cols])
//   y[r] = sum_t W[t,r]·x[t]        (transposed != 0, W — [cols, rows])
//
// Горячий случай — tied lm_head (3B Q4: 311М MAC, ~треть времени декода):
// у vtm_simd_mm_f32 нет векторизации при n == 1 (цикл по 4 столбца не
// выполняется ни разу), поэтому один столбец считает отдельное ядро.
// Внутри — 4 f32-ленты по блоку 64 элемента, блок сворачивается в double.
// Порядок сложений иной, чем у скалярного double-цикла, поэтому результат не
// побитовый: расхождение на logits 3B Q4 ≤ 2e-6 (порог 1e-5); от числа
// потоков не зависит (строки делятся пулом, каждая считается тем же кодом).
//
// 0 — успех, -1 — неверные аргументы (NULL, отрицательные размеры).
int vtm_simd_gemv_f32(float *y, const float *w, const float *x,
                      int rows, int cols, int transposed);

// Переносимые скалярные откаты тех же ядер (backend_scalar.c; SIMD не нужен).
//
// vtm_scalar_gemv_f32 — откат vtm_simd_gemv_f32, когда SIMD в сборке нет или
// CPU его не умеет: 4 double-аккумулятора вместо одной цепи FMA, поэтому
// расхождение с скалярным double-циклом — единицы ulp.
//
// vtm_scalar_dense_batch — эталон батчевого плотного TMV/MM (double, порядок
// сложений — по t). Его зовёт interp.c для скалярных бэкендов: там побитовость
// с прогоном по токену — контракт (это прежний цикл interp.c, перенесённый в
// бэкенд). 0 — успех, -1 — неверные аргументы.
int vtm_scalar_gemv_f32(float *y, const float *w, const float *x,
                        int rows, int cols, int transposed);
int vtm_scalar_dense_batch(float *y, const float *w, const float *x,
                           int rows, int cols, int n_tokens, int transposed);

// --- kind 5 (Q4): ленивая деквантизация на лету (задача 0152) --------------
//
// Веса Q4 не материализуются в fp32: ядро читает упакованные байты строки,
// разворачивает их в маленький scratch-буфер (стек, O(группы) на поток) и
// сразу умножает на активации. Пиковая память ядра не зависит от размера
// матрицы: строка разворачивается по одной группе 32 значения в scratch на
// стеке (256 байт double на поток), без malloc на строку/токен.
//
// Контракт представления (согласуется с kind 5 в loader.c, задача 0151):
//   * группа VTM_Q4_GROUP_SIZE (32) значений — 16 байт, раскладка как в
//     GGML Q4_0 (импорт GGUF без перепаковки): байт j группы хранит значение
//     j в младшем ниббле и значение j+16 в старшем; значения 0..15;
//   * группы идут по сплющенной матрице, масштаб fp32 на группу:
//     scales[(r*cols + j) / 32];
//   * asym == 0 (Q4_0): w = (q - 8) * scale;
//   * asym != 0 (Q4_1): w = q * scale + min, mins — fp32 на группу
//     (обязателен при asym != 0).
//
// y[r] = sum_j w[r,j] * x[j] (TMV, x — строка длины cols);
// y[r*n + c] = sum_t w[r,t] * b[t*n + c] (MM).
//
// 0 — успех, -1 — неверные аргументы/NULL/отрицательные размеры. Ядра не
// выделяют памяти под веса и не пишут в арену; внутренний scratch —
// фиксированного размера на стеке.
#define VTM_Q4_GROUP_SIZE 32
int vtm_q4_tmv(float *y, const uint8_t *q, const float *scales,
               const float *mins, const float *x, int rows, int cols,
               int asym);
int vtm_q4_mm(float *y, const uint8_t *q, const float *scales,
              const float *mins, const float *b, int m, int k, int n,
              int asym);
// Развернуть одну строку Q4 (cols значений с глобального индекса base) в y:
// нужно EMB — таблица словаря остаётся упакованной, в арену копируется только
// выбранная строка. 0 — успех, -1 — неверные аргументы.
int vtm_q4_dequant_row(float *y, const uint8_t *q, const float *scales,
                       const float *mins, int64_t base, int cols, int asym);

// --- Q4_K/Q6_K (kind 6/7): блоки GGML как есть (задача 0165, ADR 0018) -----
//
// Сегмент несёт блоки байт-в-байт: Q4_K — 144 байта на 256 значений
// (d fp16, dmin fp16, 12 байт 6-битных scale/min, 128 байт нибблов;
// w = d*sc*q - dmin*m), Q6_K — 210 байт (ql 128, qh 64, scales 16*int8,
// d fp16; w = d*sc*(q-32)). Ядро деквантизует под-блок (32 у Q4_K, 16 у
// Q6_K) в стековый scratch и сразу умножает: fp32-копии матрицы нет.
//
// Те же раскладки аргументов, что у vtm_q4_*: y[r] = sum_j w[r,j]*x[j]
// (TMV), y[r*n+c] = sum_t w[r,t]*b[t*n+c] (MM), dequant_row — одна строка
// таблицы словаря с глобального индекса base (EMB).
//
// Эталон — backend_scalar.c (double); Q4_K на NEON считает быстрее
// (backend_neon.c), расхождение — единицы 1e-7 на строку, как у Q4.
// 0 — успех, -1 — неверные аргументы/NULL/отрицательные размеры.
int vtm_q4k_tmv(float *y, const uint8_t *blocks, const float *x,
                int rows, int cols);
int vtm_q4k_mm(float *y, const uint8_t *blocks, const float *b,
               int m, int k, int n);
int vtm_q4k_dequant_row(float *y, const uint8_t *blocks, int64_t base,
                        int cols);
int vtm_q6k_tmv(float *y, const uint8_t *blocks, const float *x,
                int rows, int cols);
int vtm_q6k_mm(float *y, const uint8_t *blocks, const float *b,
               int m, int k, int n);
int vtm_q6k_dequant_row(float *y, const uint8_t *blocks, int64_t base,
                        int cols);

// Батчевые Q4_K/Q6_K (задача 0167): N токенов против одной матрицы kind 6/7.
//
// Раскладка — как у прочих батчевых ядер: x[t*N + j] (вход токена j по
// индексу t), y[r*N + j] (выход токена j по r). Строка весов деквантизуется
// один раз на пачку, поэтому prefill перестаёт читать блоки N раз.
//
// С 0189 у этих ядер две ступени, и первая — int8-dot GEMM
// (vtm_neon_qk_batch_i8, backend_neon.c): строка весов разворачивается в int8
// ТОЧНО (q ∈ 0..15 у Q4_K, q−32 у Q6_K), активации пачки квантуются теми же
// плоскостями, что у поштучного декода (0169), и всё это перемножается SDOT
// по 16 MAC за инструкцию. Арифметика, порядок f32-сложения и double-свёртка
// буквально те же, что у поштучного ядра строки, поэтому y пачки совпадает с N
// вызовами одноколоночного ядра БИТ В БИТ (не «в пределах 1e-5»): на 3B Q4_K
// --prefill-check даёт max|Δlogits| = 0. Это важно, потому что квантование
// активаций дискретно: расхождение 1e-7 на строку после 36 слоёв давало 1e-4
// на логитах у f32-ядра 0167 (и argmax при этом совпадал).
//
// Вторая ступень — прежнее f32-ядро 0167 (вес в f32, 4 MAC/FMA); она остаётся
// для VTM_Q4K_BATCH_I8=0 и как откат, если int8-ядро отказало (CPU без
// FEAT_DotProd, cols не кратен 256, нет памяти). VTM_Q4K_BATCH_KERNEL=fast —
// вариант int8-ядра с четырьмя токенами в дорожках int32: он в ~2x быстрее при
// одной плоскости, но складывает f32 в другом порядке и на 3B даёт
// max|Δlogits| ≈ 1,15e-4 к поштучному пути (для prefill это допустимо не
// всегда, поэтому по умолчанию он выключен).
//
// Арифметика — f32 внутри группы 32 значения (4 аккумулятора по токенам),
// свёртка группы и сумма по группам — в double; d·sc/−dmin·m применяются к
// развёрнутым весам (f32). Расхождение с поштучным vtm_q4k_tmv/vtm_q6k_tmv —
// доли 1e-6 на logits 3B (порог 1e-5; проверяет bench_backend --q4k).
// Отказ (-1): NULL/отрицательные размеры, cols не кратен 256 (тогда строка
// не начинается с границы блока) или нет памяти — вызывающий обязан пойти
// поштучным путём. При N == 1 уходит в поштучное ядро.
//
// MM: A[m,k] (kind 6/7) * B[k,N] -> Y[m,N].
int vtm_q4k_tmv_batch(float *y, const uint8_t *blocks, const float *x,
                      int rows, int cols, int n_tokens);
int vtm_q4k_mm_batch(float *y, const uint8_t *blocks, const float *b, int m,
                     int k, int n_tokens);
int vtm_q6k_tmv_batch(float *y, const uint8_t *blocks, const float *x,
                      int rows, int cols, int n_tokens);
int vtm_q6k_mm_batch(float *y, const uint8_t *blocks, const float *b, int m,
                     int k, int n_tokens);

// Батчевые Q4 TMV/MM (задача 0162): N токенов против одной Q4-матрицы.
//
// Раскладка — как у прочих батчевых ядер (см. vtm_simd_*_batch выше):
//   x[t * N + j] — вход токена j по индексу t (t = 0..cols-1 / 0..k-1);
//   y[r * N + j] — выход токена j по индексу r.
//
// Строка W разворачивается из нибблов один раз на пачку (а не N раз), поэтому
// N токенов prefill'а дешевле N отдельных вызовов vtm_q4_tmv: вес читается из
// памяти один раз. Память ядра — буфер строки на поток (cols f32), без копии
// всей матрицы.
//
// Арифметика — как у NEON-ядра 0157: целые s = q - 8 (в f32 точно), 4
// независимых f32-аккумулятора по t внутри группы 32, свёртка и масштаб
// группы — в double. Расхождение с поштучным vtm_q4_* — единицы ulp на
// группу (на logits 3B Q4 — доли 1e-6, порог 1e-5; проверяет
// tests/vtm/bench_backend.c). При N == 1 уходит в vtm_q4_tmv/vtm_q4_mm.
//
// 0 — успех, -1 — неверные аргументы/NULL/отрицательные размеры/нет памяти.
// На -1 вызывающий обязан откатиться на поштучный путь: так же, как
// vtm_q4_tmv на нехватку памяти не роняет прогон.
int vtm_q4_tmv_batch(float *y, const uint8_t *q, const float *scales,
                     const float *mins, const float *x, int rows, int cols,
                     int n_tokens, int asym);
int vtm_q4_mm_batch(float *y, const uint8_t *q, const float *scales,
                    const float *mins, const float *b, int m, int k,
                    int n_tokens, int asym);

// --- int8-dot (SDOT / VNNI) для батчевого Q4, задача 0163 -------------------
//
// Батчевое f32-ядро 0162 убирает повторную распаковку строки, но не сами
// умножения: 4 MAC за инструкцию. int8-dot даёт 16 MAC за инструкцию
// (`vdotq_s32`, ARMv8.2 FEAT_DotProd) — активации пачки квантуются в int8 с
// масштабом на токен, 4-битные веса разворачиваются в точный int8 (s = q-8),
// накопление — int32 по группе 32, деквант — f32.
//
// Ошибка зависит от режима (env VTM_Q4_I8):
//   1 — один проход int8 (ошибка ~1/127 от максимума группы);
//   2 — два прохода: остаток квантования x - xq*sx тоже в int8 (шаг sx/127),
//       ошибка ~1/127 от режима 1, цена — второй SDOT на уже загруженных весах.
// Аппаратный откат: CPU без FEAT_DotProd/VNNI, cols не кратен 32 или нет
// памяти — ядро возвращает -1, диспетчер идёт в f32-ядро 0162 (и дальше в
// скалярный эталон). VTM_Q4_I8=0 выключает int8 полностью.
//
// Фактический режим (с учётом CPU, сборки и VTM_Q4_I8): 0 — выключен,
// 1/2 — режим. Имя для логов/бенчмарка: "off", "sdot", "sdot2", "vnni",
// "vnni2".
int vtm_q4_i8_mode(void);
const char *vtm_q4_i8_name(void);
// Переопределить режим в рантайме (тесты, A/B): 0..2 либо -1 — снова env.
// Возвращает действующий режим (0, если CPU/сборка int8-dot не умеют).
// Вызывать до параллельных ядер, не одновременно из разных потоков — как
// vtm_threads_set().
int vtm_q4_i8_set_mode(int mode);

// --- int8-dot GEMV для декода kind 6/7 (Q4_K/Q6_K), задача 0169 ------------
//
// Декод — это n == 1 (одна строка активаций на матрицу), поэтому батчевое
// ядро 0163/0167 здесь не работает; отдельное одноколоночное int8-ядро
// (backend_neon.c, vtm_neon_qk_gemv_i8) считает y[r] = Σ_j w[r,j]·x[j] через
// SDOT: веса Q4_K/Q6_K представимы в int8 ТОЧНО (q и q−32 целые), квантуются
// только активации — по под-блокам 32 значения, с точной Σx для минимума Q4_K.
//
// Точность задаётся числом int8-плоскостей активаций (VTM_Q4_I8):
//   1 — int8 (шаг sx, ошибка ≤ sx/2 на значение; для A/B, argmax не держит);
//   2 — int8 + остаток (шаг sx/127, ошибка ≤ sx/254) — по умолчанию: argmax
//       на golden 20/20, Δlogits в допуске 1e-2 (замер — отчёт 0169);
//   3 — + остаток второй плоскости (шаг sx/127²) — если нужен запас точности.
// Откат: VTM_Q4_I8=0 — прежний f32-путь (q4k_row_dot_f32/q6k_row_dot_f32).
// Отказ ядра (-1: CPU без FEAT_DotProd, cols не кратен 256, нет памяти) —
// тот же f32-путь, прогон не падает; VTM_Q4K_STRICT=1 тоже выключает int8.
// 1 — в сборке есть int8-dot-ядра (SDOT/VNNI) и CPU их умеет; 0 — нет
// (тогда оба режима выше всегда 0). Без учёта VTM_Q4_I8.
int vtm_q4_i8_hw(void);
int vtm_q4k_i8_mode(void);
const char *vtm_q4k_i8_name(void);
// Переопределить режим декода в рантайме (0..3 либо -1 — снова env).
// Возвращает действующий режим. Как vtm_q4_i8_set_mode(): до параллельных
// ядер и не одновременно из разных потоков.
int vtm_q4k_i8_set_mode(int mode);

// --- int8 (kind 4): упакованная таблица словаря, задача 0170 -----------------
//
// kind 4 — самодостаточный INT8-сегмент (18 байт заголовка, fp32/fp16-масштабы
// на блок 128, затем знаковые байты q, w = q·scale). С 0170 загрузчик держит
// такую таблицу упакованной: байты q — в отображении файла, в арене только
// fp32-масштабы (n_scales·4). VTM_I8_EMB=0 возвращает старое поведение
// (fp32-развёртка в арене, тензор приходит как kind 1).
//
// Раскладка: строка r начинается с индекса r·cols, масштаб — общий на блок
// 128 подряд идущих значений (по сплющенной матрице), т.е.
// sc[((r·cols + j) / 128)].
//
// vtm_i8_tmv/vtm_i8_mm — эталон в double: y[r] = Σ_j q[r,j]·sc·x[j],
// y[r·n+c] = Σ_t q[r,t]·sc·b[t·n+c]. vtm_i8_dequant_row — одна строка (EMB).
// vtm_i8_gemv — рабочий GEMV декода (n == 1): NEON int8-dot (SDOT) с
// квантованными активациями (точность как у 0169, режим VTM_Q4_I8), иначе
// переносимый f32-путь по блокам 128. 0 — успех, -1 — неверные аргументы.
int vtm_i8_tmv(float *y, const int8_t *q, const float *scales,
               const float *x, int rows, int cols);
int vtm_i8_mm(float *y, const int8_t *q, const float *scales,
              const float *b, int m, int k, int n);
int vtm_i8_dequant_row(float *y, const int8_t *q, const float *scales,
                       int64_t base, int cols);
int vtm_i8_gemv(float *y, const int8_t *q, const float *scales,
                const float *x, int rows, int cols);

// --- MM: SIMD-ядра вне `vtm_backend` ---------------------------------------
//
// В `vtm_backend` (vtm.h) поля MM нет, поэтому ядра MM — отдельные функции.
// op_mm() в interp.c зовёт их при SIMD-бэкенде: тритовую A — vtm_simd_mm_trit,
// плотную A — vtm_simd_mm_f32, а при n == 1 (один столбец) — vtm_simd_gemv_f32.
//
//   y[r*n + c] = sum_t A[r*k + t] * B[t*n + c]
//
// A — тритовая матрица (слова pack20 + масштабы, блок 128/160; точно как у
// TMV) либо плотная fp32. 0 — успех, -1 — неверные аргументы. При отсутствии
// SIMD работает скалярный откат (результат тот же по смыслу, медленнее).
int vtm_simd_mm_trit(float *y, const uint8_t *w, const float *scales,
                     const float *b, int m, int k, int n, int scale_block);
int vtm_simd_mm_f32(float *y, const float *a, const float *b,
                    int m, int k, int n);

// --- многопоточность: пул pthread (задача 0077) ----------------------------
//
// Ядра TMV/MM/RMSN делят работу по строкам выхода. Пул живёт в
// backend_scalar.c (этот TU собирается всегда), ядра SIMD-бэкендов зовут
// те же vtm_threads()/vtm_parallel_rows().
//
// Число потоков N = VTM_THREADS (1..VTM_THREADS_MAX); переменная не задана —
// min(P-ядер, VTM_THREADS_MAX) на Apple Silicon (там ядра разные, см.
// vtm_cfg_cores в backend_scalar.c), иначе min(онлайн-ядер, VTM_THREADS_MAX).
// N == 1 — ровно прежнее поведение:
// vtm_parallel_rows() вызывает колбэк в текущем потоке, пул не создаётся.
// Значение запоминается при первом обращении (vtm_threads()).
//
// 0201: было 8. GEMV декода kind 6/7 масштабируется по потокам почти линейно
// до числа P-ядер (замер на M1 Pro: 8,8 ГБ/с на поток, 6 потоков — 47 ГБ/с,
// 8 — 54 ГБ/с), поэтому потолок поднят до 10, чтобы 10-ядерные конфигурации
// (и E-ядра через VTM_THREADS=10) не упирались в константу. Пул держит слот
// команды на tid, память под слоты — VTM_THREADS_MAX*sizeof(vtm_cmd), это
// сотни байт.
#define VTM_THREADS_MAX 10
// Меньше этой оценки работы (в MAC) параллелить невыгодно: оверхед
// пробуждения потоков больше выигрыша. Решение принимает
// vtm_parallel_rows(), порог — по всей задаче.
#define VTM_PAR_MIN_WORK 131072.0
// RMSN параллелится только на больших n: у него два прохода (сумма
// квадратов и масштабирование), и на n порядка скрытого размера модели
// пробуждение потоков дороже счёта. Порог — 512K элементов: на нём scalar
// получает ~3.5x, SIMD ~2x на 4 потоках, меньшие n считаются как раньше.
#define VTM_PAR_RMSN_MIN 524288

// Эффективное число потоков (1..VTM_THREADS_MAX). VTM_THREADS=1 — без
// потоков; 0/мусор в переменной — значение по умолчанию и предупреждение.
int vtm_threads(void);

// Переопределить число потоков в рантайме (тесты, хост). n <= 0 — вернуть
// значение из env. Возвращает эффективное число потоков. Потоки создаются
// лениво, при первом большом вызове; вызывать до параллельных ядер, не
// одновременно из разных потоков (переопределение — не горячий путь).
int vtm_threads_set(int n);

// Колбэк строчного ядра: посчитать строки [r0, r1) выхода. tid — номер
// логического потока в [0, число потоков), по нему ядро берёт свою
// scratch-память. Диапазоны строк не пересекаются, поэтому результат не
// зависит от разбиения: каждая строка считается тем же кодом в том же
// порядке, что и при одном потоке (побитово).
typedef void (*vtm_rows_fn)(void *ctx, int tid, int r0, int r1);

// Разбить [0, n) на N непрерывных диапазонов и выполнить fn в пуле
// (вызвавший считает диапазон tid 0). Строки ядер стоят одинаково, поэтому
// статического деления достаточно и мьютекс на строку не нужен. Если N < 2,
// n < 2, работа work < VTM_PAR_MIN_WORK или потоки не создались —
// fn(ctx, 0, 0, n) вызывается в текущем потоке (прежнее поведение). Вызовы
// из разных потоков сериализуются.
void vtm_parallel_rows(int n, double work, vtm_rows_fn fn, void *ctx);

#ifdef __cplusplus
}
#endif

#endif // VTM_BACKEND_H

// Атрибут «может быть не использована»: функция живёт под #if одной
// платформы, а компилятор другой (gcc -Werror) считает её лишней.
#if defined(__GNUC__) || defined(__clang__)
#define VTM_MAYBE_UNUSED __attribute__((unused))
#else
#define VTM_MAYBE_UNUSED
#endif
