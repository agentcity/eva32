// VTM — Metal-бэкенд, фазы 1–2 (задачи 0192/0196, ADR 0021).
//
// Отдельный исполнитель рядом с scalar/neon: GEMV/GEMM для kind 6 (Q4_K),
// kind 7 (Q6_K) и int8-таблицы словаря (kind 4). Attention, KV, RMSNorm,
// softmax и сэмплинг остаются на CPU — они маленькие и латентные (ADR 0021 §2).
//
// Zero-copy. Веса не копируются: MTLBuffer создаётся через
// newBufferWithBytesNoCopy: на выровненный вниз до страницы срез mmap-региона
// контейнера (на Apple Silicon страница 16 КиБ), а смещение тензора внутри
// среза передаётся ядру аргументом w_off. Кэш буферов — (указатель, длина);
// вытеснение LRU. Если no-copy невозможен (память не страничная или Metal
// отказал), маленькие буферы копируются (тесты/арена), большие — отказ:
// диспетчер backend_scalar.c идёт прежним CPU-путём, прогон не падает.
//
// Откат. Любая ошибка Metal (нет устройства, не скомпилировался MSL, ошибка
// командного буфера) логируется один раз и переводит ядро в отказ (-1);
// вызывающий уже умеет откатываться на NEON/scalar. Побитового паритета с CPU
// нет: GPU меняет порядок суммирования (контракт — argmax + Δlogits ≤ 1e-2).
//
// Фаза 2 (0196): декод больше не делает синхронный командный буфер на каждый
// тензор. Интерпретатор (interp.c) собирает подряд идущие GEMV в очередь
// vtm_metal_defer_*: все dispatches уходят в один MTLCommandBuffer, ждём
// только когда CPU действительно нужен результат (attn/RMSN/softmax/сэмплинг).
// Сцены две (ping-pong): пока GPU считает одну, CPU кодирует другую; ожидание —
// dispatch_semaphore, который сигналит completedHandler командного буфера, а
// не waitUntilCompleted. Prefill идёт батчевым GEMM: все токены пачки в одном
// dispatch на матрицу (vtm_metal_qk_gemm, ядра *_gemm в vtm_metal.metal).
//
// MSL. Исходник — src/vtm/vtm_metal.metal; Makefile встраивает его в
// vtm_metal_kernels.h, поэтому по умолчанию шейдеры компилируются в рантайме
// (newLibraryWithSource). Если рядом есть vtm.metallib (его собирает
// `make metallib` при наличии Metal Toolchain, Xcode 26 требует скачать
// компонент) или задан VTM_METAL_LIB, он загружается вместо исходника. Время
// компиляции/загрузки библиотеки доступно в статистике (vtm_metal_stats).
//
// Сборка без Metal: Makefile не добавляет этот TU (VTM_WITH_METAL=0), а
// backend_scalar.c компилирует вызовы metal только под #if VTM_WITH_METAL.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "vtm_backend.h"

#if defined(__has_feature)
#  if __has_feature(thread_sanitizer)
#    define VTM_METAL_TSAN 1
#  endif
#endif
#ifndef VTM_METAL_TSAN
#  define VTM_METAL_TSAN 0
#endif

// Сгенерированный Makefile'ом из vtm_metal.metal C-строковый литерал.
#include "vtm_metal_kernels.h"

// Должно совпадать с GemvArgs в vtm_metal.metal (sizeof == 32).
typedef struct {
    uint32_t rows;
    uint32_t cols;
    uint32_t nblocks;
    uint32_t n_tokens;  // GEMM: число токенов пачки (GEMV игнорирует)
    uint64_t w_off;
    uint64_t sc_off;    // смещение fp32-масштабов в элементах (int8)
} vtm_metal_args;

#define VTM_METAL_MAX_BIND 4096
#define VTM_METAL_COPY_LIMIT (64u * 1024u * 1024u)

// Фаза 2: сцены ping-pong и предел отложенных GEMV на сцену.
#define VTM_METAL_STAGES 2
#define VTM_METAL_MAX_DEFER 64
#define VTM_METAL_GEMM_MAX_N 32
// Потолок staging-буфера результатов одной сцены (больше — сброс и ожидание).
#define VTM_METAL_YBUF_MAX ((size_t)10u << 20)

typedef struct {
    const void *ptr;
    size_t bytes;
    id<MTLBuffer> buf;
    uint64_t off;      // байтовое смещение тензора в buf
    uint64_t tick;
    int copied;        // 1 — буфер создан копией (не zero-copy)
} vtm_metal_bind;

typedef struct {
    float *dst;        // куда вернуть результат (адрес тензора)
    size_t bytes;      // размер результата
    size_t off;        // смещение в y-буфере сцены
} metal_result;

typedef struct {
    id<MTLBuffer> xb;  // staging активаций
    id<MTLBuffer> yb;  // staging результатов
    size_t xcap, ycap, xused, yused;
    id<MTLCommandBuffer> cb;              // удерживается до завершения
    id<MTLComputeCommandEncoder> enc;     // открыт, пока cb не закоммичен
    dispatch_semaphore_t sem;             // сигналит completedHandler
    metal_result res[VTM_METAL_MAX_DEFER];
    int nres;
    int submitted;
} metal_stage;

typedef struct {
    id<MTLCommandBuffer> cb;
    id<MTLComputeCommandEncoder> enc;
    id<MTLBuffer> tmp;      // staging для cos/sin RoPE
    size_t        tmpcap;
    size_t        tmpused;
    uint64_t      rope_pos;
    uint32_t      rope_dim;
    uint32_t      rope_int;
    size_t        rope_off;
    int           rope_valid;
    uint64_t      ops;
    double        t_encode0;
} metal_graph;

static metal_graph g_g;
static int g_graph_on;
static int g_graph_busy;
static int g_graph_failed;
static uintptr_t g_graph_owner;

static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t g_defer_lock = PTHREAD_MUTEX_INITIALIZER;
static int g_avail = -1;            // -1 — не проверяли, 0/1
static int g_init = 0;              // 0 — нет, 1 — готов, -1 — отказ
static int g_mode = VTM_METAL_OFF;  // OFF / FULL / BATCH (задача 0198)
static int g_warned_tsan = 0;

static id<MTLDevice> g_dev;
static id<MTLCommandQueue> g_queue;
static id<MTLLibrary> g_lib;
static id<MTLComputePipelineState> g_ps_q4k;
static id<MTLComputePipelineState> g_ps_q6k;
static id<MTLComputePipelineState> g_ps_i8;
// Фаза 3: R=4 строки на warp (см. vtm_metal.metal). Нет в старом metallib —
// остаются nil, и GEMV идёт прежними однострочными ядрами.
static id<MTLComputePipelineState> g_ps_q4k_r4;
static id<MTLComputePipelineState> g_ps_q6k_r4;
static id<MTLComputePipelineState> g_ps_q4k_g4;
// 0214: тайловые батчевые GEMM (2 ряда на поток): [0] NT=16, [1] NT=32.
static id<MTLComputePipelineState> g_ps_q4k_tile[2];
static id<MTLComputePipelineState> g_ps_q6k_tile[2];
static id<MTLComputePipelineState> g_ps_i8_tile[2];
// Фаза 3: ядра графа токена (RMSNorm/RoPE/SiLU/MUL/ADD/KVW/attention).
static id<MTLComputePipelineState> g_ps_norm;
static id<MTLComputePipelineState> g_ps_rope;
static id<MTLComputePipelineState> g_ps_silu;
static id<MTLComputePipelineState> g_ps_mul;
static id<MTLComputePipelineState> g_ps_add;
static id<MTLComputePipelineState> g_ps_kvw;
static id<MTLComputePipelineState> g_ps_attn;
static id<MTLComputePipelineState> g_ps_attn_batch;  // 0214: attention пачки
static id<MTLComputePipelineState> g_ps_copy;

static id<MTLBuffer> g_xbuf;
static size_t g_xcap;
static id<MTLBuffer> g_ybuf;
static size_t g_ycap;

// Батчевый GEMM (prefill): staging входа (token-major) и выхода (token-major).
// Оба — буферы Metal, а не zero-copy поверх памяти вызывающего: маленькие
// malloc-буферы тестов могут кончаться на границе страницы, и no-copy буфер
// над ними ловит SIGBUS (проверено --check). Живой prefill идёт другим путём —
// vtm_metal_qk_gemm_arena на mmap-арене пачки.
static id<MTLBuffer> g_gxbuf;
static size_t g_gxcap;
static id<MTLBuffer> g_gybuf;
static size_t g_gycap;

// Очередь декода (фаза 2).
static metal_stage g_stage[VTM_METAL_STAGES];
static int g_enc = -1;              // сцена, в которую кодируем, -1 — нет
static int g_flight = -1;           // сцена, отданная GPU, -1 — нет
static int g_defer_on;
static int g_defer_busy;
static int g_defer_failed;  // ошибка Metal в сессии: всё пересчитать на CPU
static uintptr_t g_defer_owner;  // pthread_self() владельца (атомарно)

static vtm_metal_bind g_binds[VTM_METAL_MAX_BIND];
static int g_nbinds;
static uint64_t g_tick;

static char g_err[256];
static uint64_t g_init_ms;
static uint64_t g_compile_ms;
static uint64_t g_calls;
static uint64_t g_fail;
static uint64_t g_nocopy_bytes;
static uint64_t g_copy_bytes;
static uint64_t g_alloc_before;
static uint64_t g_gpu_ns;
static uint64_t g_alloc_after;
static uint64_t g_gemm_calls;
static int g_gq_active;   // 0214: в очереди есть незакрытая группа GEMM
static uint64_t g_defer_ops;
static uint64_t g_defer_waits;
static uint64_t g_graph_tokens;
static uint64_t g_graph_ops;
static uint64_t g_graph_fail;
static uint64_t g_graph_encode_ns;
static uint64_t g_wait_ns;
static int g_atexit_done;

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

static void metal_log(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "VTM: metal: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
}

static void metal_set_err(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_err, sizeof g_err, fmt, ap);
    va_end(ap);
}

// --- статистика -------------------------------------------------------------
static void metal_stats_print(void) {
    if (!getenv("VTM_METAL_STATS")) return;
    fprintf(stderr,
            "VTM: metal stats: init %.1f ms (msl %.1f ms), вызовов %llu, "
            "отказов %llu, буферов %d, no-copy %.1f МиБ, копий %.1f МиБ, "
            "currentAllocatedSize %.1f МиБ (до init %.1f МиБ), GPU %.1f мс, "
            "gemm %llu, отложено %llu, ожиданий %llu (%.1f мс), "
            "граф %llu токенов/%llu ops (отказов %llu, кодирование "
            "%.1f мс)\n",
            (double)g_init_ms, (double)g_compile_ms,
            (unsigned long long)g_calls, (unsigned long long)g_fail, g_nbinds,
            (double)g_nocopy_bytes / (1024.0 * 1024.0),
            (double)g_copy_bytes / (1024.0 * 1024.0),
            g_dev ? (double)[g_dev currentAllocatedSize] / (1024.0 * 1024.0)
                  : 0.0,
            (double)g_alloc_before / (1024.0 * 1024.0),
            (double)g_gpu_ns / 1e6,
            (unsigned long long)g_gemm_calls,
            (unsigned long long)g_defer_ops,
            (unsigned long long)g_defer_waits, (double)g_wait_ns / 1e6,
            (unsigned long long)g_graph_tokens,
            (unsigned long long)g_graph_ops,
            (unsigned long long)g_graph_fail,
            (double)g_graph_encode_ns / 1e6);
}

// --- инициализация ----------------------------------------------------------
//
// Устройство проверяется отдельно от компиляции шейдеров: выбор бэкенда не
// должен платить за MSL, если модель вообще без kind 4/6/7.
static int metal_device(void) {
    if (g_avail >= 0) return g_avail;
    if (VTM_METAL_TSAN) {
        if (!g_warned_tsan) {
            metal_log("TSan-сборка — metal выключен, auto уходит в simd");
            g_warned_tsan = 1;
        }
        g_avail = 0;
        return 0;
    }
    g_dev = MTLCreateSystemDefaultDevice();
    if (!g_dev) {
        metal_log("MTLCreateSystemDefaultDevice вернул nil — откат на CPU");
        g_avail = 0;
        return 0;
    }
    g_avail = 1;
    return 1;
}

static int metal_load_library(void) {
    NSError *err = nil;
    const char *envlib = getenv("VTM_METAL_LIB");
    double t0 = now_ms();

    if (envlib && *envlib) {
        NSString *p = [NSString stringWithUTF8String:envlib];
        if (p) g_lib = [g_dev newLibraryWithURL:[NSURL fileURLWithPath:p]
                                          error:&err];
        if (!g_lib)
            metal_log("не загрузился VTM_METAL_LIB=%s: %s; компилирую MSL",
                      envlib, err ? [[err localizedDescription] UTF8String]
                                  : "?");
    }
    if (!g_lib && access("vtm.metallib", R_OK) == 0) {
        err = nil;
        g_lib = [g_dev newLibraryWithURL:
                 [NSURL fileURLWithPath:@"vtm.metallib"] error:&err];
        if (!g_lib)
            metal_log("не загрузился vtm.metallib: %s; компилирую MSL",
                      err ? [[err localizedDescription] UTF8String] : "?");
    }
    if (!g_lib) {
        NSString *src = [NSString stringWithUTF8String:vtm_metal_msl];
        if (!src) {
            metal_set_err("MSL-исходник не UTF-8");
            return -1;
        }
        g_lib = [g_dev newLibraryWithSource:src options:nil error:&err];
        if (!g_lib) {
            metal_set_err("компиляция MSL: %s",
                          err ? [[err localizedDescription] UTF8String] : "?");
            return -1;
        }
    }
    g_compile_ms = (uint64_t)(now_ms() - t0);
    return 0;
}

static id<MTLComputePipelineState> metal_pipeline(const char *name) {
    NSError *err = nil;
    id<MTLFunction> fn = [g_lib newFunctionWithName:
                          [NSString stringWithUTF8String:name]];
    id<MTLComputePipelineState> ps;

    if (!fn) {
        metal_set_err("нет функции %s в metallib", name);
        return nil;
    }
    ps = [g_dev newComputePipelineStateWithFunction:fn error:&err];
    if (!ps) {
        metal_set_err("pipeline %s: %s", name,
                      err ? [[err localizedDescription] UTF8String] : "?");
    }
    return ps;
}

// Ядро GEMV и число строк на threadgroup. Фаза 3: Q4_K/Q6_K считаются
// вариантом R=4 (16 строк на 4-warp threadgroup, 4 независимые загрузки
// веса на warp); если в metallib его нет — прежнее однострочное ядро.
static int gemv_r4_env(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("VTM_METAL_GEMV_R4");
        cached = (e && *e && *e != '0') ? 1 : 0;
    }
    return cached;
}

static int gemv4_env(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("VTM_METAL_GEMV4");
        cached = (e && *e && *e != '0') ? 1 : 0;
    }
    return cached;
}

static id<MTLComputePipelineState> metal_gemv_ps(int kind, NSUInteger *rpt) {
    // Фаза 3: Q4_K — четыре строки на warp (16 строк на threadgroup). В
    // изолированном --qk-proj выигрыш нестабилен (машина занята), а в полном
    // графе CLI он оказался МЕДЛЕННЕЕ однострочного: 14,6 против 16,7 ток/с
    // (3 пары прогонов, 1,91-2,19 с на 32 токена). Поэтому по умолчанию
    // выключен; включается VTM_METAL_GEMV4=1 для A/B.
    if (kind == 0 && g_ps_q4k_g4 && gemv4_env()) {
        *rpt = 16;
        return g_ps_q4k_g4;
    }
    // Прежний эксперимент R=4 (массив указателей) включается только явно:
    // на этой машине он оказался медленнее однострочного (0.95 против 0.79 мс).
    if (gemv_r4_env()) {
        if (kind == 0 && g_ps_q4k_r4) { *rpt = 16; return g_ps_q4k_r4; }
        if (kind == 1 && g_ps_q6k_r4) { *rpt = 16; return g_ps_q6k_r4; }
    }
    *rpt = 4;
    return kind == 0 ? g_ps_q4k : (kind == 1 ? g_ps_q6k : g_ps_i8);
}

// Вызывается под g_lock.
static int metal_init(void) {
    double t0 = now_ms();

    if (g_init) return g_init == 1 ? 0 : -1;
    if (sizeof(vtm_metal_args) != 32) {
        metal_set_err("sizeof(vtm_metal_args) = %zu, ожидали 32",
                      sizeof(vtm_metal_args));
        g_init = -1;
        return -1;
    }
    if (!metal_device()) {
        metal_set_err("Metal-устройства нет");
        g_init = -1;
        return -1;
    }
    g_alloc_before = (uint64_t)[g_dev currentAllocatedSize];
    g_queue = [g_dev newCommandQueue];
    if (!g_queue) {
        metal_set_err("newCommandQueue вернул nil");
        g_init = -1;
        return -1;
    }
    if (metal_load_library() != 0) {
        metal_log("%s — откат на CPU", g_err);
        g_init = -1;
        return -1;
    }
    g_ps_q4k = metal_pipeline("vtm_q4k_gemv");
    g_ps_q6k = metal_pipeline("vtm_q6k_gemv");
    g_ps_i8 = metal_pipeline("vtm_i8_gemv");
    g_ps_q4k_r4 = metal_pipeline("vtm_q4k_gemv_r4");
    g_ps_q6k_r4 = metal_pipeline("vtm_q6k_gemv_r4");
    g_ps_q4k_g4 = metal_pipeline("vtm_q4k_gemv4");
    // Фаза 3: отсутствие этих ядер в старом metallib не ломает прогон —
    // vtm_metal_graph_available() вернёт 0, и токен пойдёт прежним путём.
    g_ps_norm = metal_pipeline("vtm_rmsnorm");
    g_ps_rope = metal_pipeline("vtm_rope");
    g_ps_silu = metal_pipeline("vtm_silu");
    g_ps_mul = metal_pipeline("vtm_mul");
    g_ps_add = metal_pipeline("vtm_add");
    g_ps_kvw = metal_pipeline("vtm_kvw");
    g_ps_attn = metal_pipeline("vtm_attn_decode");
    g_ps_attn_batch = metal_pipeline("vtm_attn_prefill");
    g_ps_copy = metal_pipeline("vtm_copy");
    if (!g_ps_q4k || !g_ps_q6k || !g_ps_i8) {
        metal_log("%s — откат на CPU", g_err);
        g_init = -1;
        return -1;
    }
    // 0214: тайловые ядра. Их отсутствие (старый metallib) не ломает прогон:
    // арена пачки не берётся, prefill идёт прежним путём.
    {
        static const char *const tnames[3][2] = {
            {"vtm_q4k_gemm_t16", "vtm_q4k_gemm_t32"},
            {"vtm_q6k_gemm_t16", "vtm_q6k_gemm_t32"},
            {"vtm_i8_gemm_t16", "vtm_i8_gemm_t32"}
        };
        int v;
        for (v = 0; v < 2; v++) {
            g_ps_q4k_tile[v] = metal_pipeline(tnames[0][v]);
            g_ps_q6k_tile[v] = metal_pipeline(tnames[1][v]);
            g_ps_i8_tile[v] = metal_pipeline(tnames[2][v]);
        }
        if (!g_ps_q4k_tile[1] || !g_ps_q6k_tile[1] || !g_ps_i8_tile[1])
            metal_log("тайловых GEMM-ядер нет (%s) — prefill пачкой пойдёт "
                      "прежним путём", g_err);
    }

    g_alloc_after = (uint64_t)[g_dev currentAllocatedSize];
    g_init = 1;
    g_init_ms = (uint64_t)(now_ms() - t0);
    if (!g_atexit_done) {
        g_atexit_done = 1;
        atexit(metal_stats_print);
    }
    return 0;
}

// --- staging-буферы ---------------------------------------------------------
static int metal_reserve(id<MTLBuffer> *buf, size_t *cap, size_t need) {
    if (*buf && *cap >= need) return 0;
    [*buf release];
    *buf = [g_dev newBufferWithLength:need
                              options:MTLResourceStorageModeShared];
    if (!*buf) {
        *cap = 0;
        metal_set_err("newBufferWithLength(%zu) вернул nil", need);
        return -1;
    }
    *cap = need;
    return 0;
}

// --- zero-copy буферы весов -------------------------------------------------
//
// Найти/создать MTLBuffer для региона [ptr, ptr+bytes). При успехе *off —
// байтовое смещение ptr в буфере (0 для копии). Под g_lock.
static int metal_bind(const void *ptr, size_t bytes, id<MTLBuffer> *out,
                      uint64_t *off, int *copied) {
    long page = sysconf(_SC_PAGESIZE);
    uintptr_t addr, base;
    size_t boff, len, i;
    int slot;

    if (!ptr || !bytes) return -1;
    if (page <= 0) page = 16384;
    for (i = 0; i < (size_t)g_nbinds; i++) {
        if (g_binds[i].ptr == ptr && g_binds[i].bytes == bytes) {
            g_binds[i].tick = ++g_tick;
            *out = g_binds[i].buf;
            *off = g_binds[i].off;
            if (copied) *copied = g_binds[i].copied;
            return 0;
        }
    }

    addr = (uintptr_t)ptr;
    base = addr & ~(uintptr_t)((uintptr_t)page - 1u);
    boff = (size_t)(addr - base);
    // Длина — до границы страницы. Для mmap-региона это внутри отображения
    // (ядро округляет его до страницы), для malloc — хвост той же страницы;
    // GPU читает только [boff, boff+bytes), лишняя страница не трогается.
    len = (boff + bytes + (size_t)page - 1u) & ~((size_t)page - 1u);
    if (len < boff + bytes) return -1; // переполнение

    if (g_nbinds < VTM_METAL_MAX_BIND) {
        slot = g_nbinds;
    } else {
        // LRU: освобождаем самый старый слот и переиспользуем его как новый.
        slot = 0;
        for (i = 1; i < (size_t)VTM_METAL_MAX_BIND; i++)
            if (g_binds[i].tick < g_binds[slot].tick) slot = (int)i;
        [g_binds[slot].buf release];
        g_binds[slot].buf = nil;
        g_binds[slot] = g_binds[g_nbinds - 1];
        g_nbinds--;
        slot = g_nbinds;
    }

    {
        id<MTLBuffer> b = [g_dev newBufferWithBytesNoCopy:(void *)base
                                                   length:len
                                                  options:MTLResourceStorageModeShared
                                                deallocator:nil];
        if (b) {
            g_binds[slot].ptr = ptr;
            g_binds[slot].bytes = bytes;
            g_binds[slot].buf = b;
            g_binds[slot].off = (uint64_t)boff;
            g_binds[slot].tick = ++g_tick;
            g_binds[slot].copied = 0;
            g_nbinds = slot + 1;
            g_nocopy_bytes += (uint64_t)bytes;
            *out = b;
            *off = (uint64_t)boff;
            if (copied) *copied = 0;
            return 0;
        }
    }

    // No-copy не вышел: для небольших буферов (тесты, арена) копия, для
    // больших — отказ, чтобы не удваивать 2-ГиБ веса в памяти.
    if (bytes > (size_t)VTM_METAL_COPY_LIMIT) {
        metal_set_err("no-copy не удался для %zu байт (страница %ld)",
                      bytes, page);
        return -1;
    }
    {
        id<MTLBuffer> b = [g_dev newBufferWithBytes:ptr
                                             length:bytes
                                            options:MTLResourceStorageModeShared];
        if (!b) {
            metal_set_err("newBufferWithBytes(%zu) вернул nil", bytes);
            return -1;
        }
        g_binds[slot].ptr = ptr;
        g_binds[slot].bytes = bytes;
        g_binds[slot].buf = b;
        g_binds[slot].off = 0;
        g_binds[slot].tick = ++g_tick;
        g_binds[slot].copied = 1;
        g_nbinds = slot + 1;
        g_copy_bytes += (uint64_t)bytes;
        *out = b;
        *off = 0;
        if (copied) *copied = 1;
        return 0;
    }
}

// --- общий GEMV (синхронный путь: одиночный тензор) -------------------------
static int metal_xbuf(size_t bytes) {
    return metal_reserve(&g_xbuf, &g_xcap, bytes);
}

static int metal_ybuf(size_t bytes) {
    return metal_reserve(&g_ybuf, &g_ycap, bytes);
}

static int metal_gemv_body(int kind, float *y, const void *w,
                           const float *scales, const float *x,
                           int rows, int cols) {
    id<MTLBuffer> wb = nil, sb = nil;
    uint64_t woff = 0, scoff = 0;
    vtm_metal_args a;
    id<MTLCommandBuffer> cb;
    id<MTLComputeCommandEncoder> enc;
    id<MTLComputePipelineState> ps;
    size_t wbytes, scbytes, xbytes, ybytes;
    NSUInteger nrows_pt = 4;
    int rc = -1;

    if (!y || !w || !x) return -1;
    if (rows <= 0 || cols <= 0) return -1;

    pthread_mutex_lock(&g_lock);
    if (metal_init() != 0) {
        g_fail++;
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    if (kind == 0 || kind == 1) {
        if (cols % 256 != 0) {
            pthread_mutex_unlock(&g_lock);
            return -1;
        }
        a.nblocks = (uint32_t)(cols / 256);
    } else {
        a.nblocks = 0;
    }
    // Путь kind: 0 — Q4_K, 1 — Q6_K, 2 — int8.
    if (kind == 0)
        wbytes = ((size_t)rows * (size_t)cols / 256u) * 144u;
    else if (kind == 1)
        wbytes = ((size_t)rows * (size_t)cols / 256u) * 210u;
    else
        wbytes = (size_t)rows * (size_t)cols;
    if (metal_bind(w, wbytes, &wb, &woff, NULL) != 0) {
        g_fail++;
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    if (kind == 2) {
        uint64_t n = (uint64_t)rows * (uint64_t)cols;
        scbytes = (size_t)(n / 128u + (n % 128u ? 1u : 0u)) * sizeof(float);
        if (!scales || metal_bind(scales, scbytes, &sb, &scoff, NULL) != 0) {
            g_fail++;
            pthread_mutex_unlock(&g_lock);
            return -1;
        }
        a.sc_off = scoff / sizeof(float);
    } else {
        a.sc_off = 0;
    }
    a.rows = (uint32_t)rows;
    a.cols = (uint32_t)cols;
    a.n_tokens = 1;
    a.w_off = woff;

    xbytes = (size_t)cols * sizeof(float);
    ybytes = (size_t)rows * sizeof(float);
    if (metal_xbuf(xbytes) != 0 || metal_ybuf(ybytes) != 0) {
        metal_set_err("нет Metal-буфера под активации/выход");
        g_fail++;
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    memcpy([g_xbuf contents], x, xbytes);

    {
        NSUInteger rpt = 4;
        ps = metal_gemv_ps(kind, &rpt);
        nrows_pt = rpt;
    }
    cb = [g_queue commandBuffer];
    if (!cb) {
        metal_set_err("commandBuffer вернул nil");
        g_fail++;
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ps];
    [enc setBuffer:wb offset:0 atIndex:0];
    [enc setBuffer:g_xbuf offset:0 atIndex:1];
    [enc setBuffer:g_ybuf offset:0 atIndex:2];
    [enc setBytes:&a length:sizeof a atIndex:3];
    if (kind == 2) [enc setBuffer:sb offset:0 atIndex:4];
    [enc dispatchThreadgroups:
        MTLSizeMake(((NSUInteger)rows + nrows_pt - 1u) / nrows_pt, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    {
        double gpu = (cb.GPUEndTime - cb.GPUStartTime) * 1e9;
        if (gpu > 0.0) g_gpu_ns += (uint64_t)gpu;
        if (getenv("VTM_METAL_TRACE"))
            metal_log("kind %d %dx%d: GPU %.3f мс", kind, rows, cols,
                      (cb.GPUEndTime - cb.GPUStartTime) * 1e3);
    }
    if (cb.status != MTLCommandBufferStatusCompleted || cb.error) {
        if (g_fail == 0)
            metal_log("командный буфер: %s — дальше CPU-откат",
                      cb.error ? [[cb.error localizedDescription] UTF8String]
                               : "status != completed");
        metal_set_err("командный буфер не завершился");
        g_fail++;
    } else {
        memcpy(y, [g_ybuf contents], ybytes);
        g_calls++;
        rc = 0;
    }
    pthread_mutex_unlock(&g_lock);
    return rc;
}

static int metal_gemv(int kind, float *y, const void *w, const float *scales,
                      const float *x, int rows, int cols) {
    // Автоосвобождение: на токен сотни командных буферов; без пула они
    // копились бы до конца процесса.
    @autoreleasepool {
        return metal_gemv_body(kind, y, w, scales, x, rows, cols);
    }
}

// --- фаза 2: отложенная очередь декода --------------------------------------
//
// Одна сцена — один MTLCommandBuffer с серией dispatches. Пока GPU считает
// сцену A, CPU кодирует сцену B (ping-pong). Ожидание — semaphore, который
// сигналит completedHandler; waitUntilCompleted не используется.

// Вызывается под g_defer_lock.
static int stage_wait(int idx);

static int stage_submit(void) {
    metal_stage *s;
    dispatch_semaphore_t sem;

    if (g_enc < 0) return 0;
    s = &g_stage[g_enc];
    if (s->enc) {
        [s->enc endEncoding];
        [s->enc release];
        s->enc = nil;
    }
    if (s->nres == 0) {
        [s->cb release];
        s->cb = nil;
        g_enc = -1;
        return 0;
    }
    // Вторая сцена не может быть отдана, пока не дождались первой: Metal
    // исполняет командные буферы одной очереди по порядку, но staging-
    // результат первой ещё нужен CPU. Ждём здесь, а не в отдельном потоке:
    // перекрытие уже получено тем, что вторая сцена кодировалась, пока
    // первая считалась.
    if (g_flight >= 0) {
        int prev = g_flight;
        g_flight = -1;
        if (stage_wait(prev) != 0) {
            g_defer_failed = 1;
            return -1;
        }
    }
    sem = s->sem;
    [s->cb addCompletedHandler:^(id<MTLCommandBuffer> cb) {
        (void)cb;
        dispatch_semaphore_signal(sem);
    }];
    [s->cb commit];
    s->submitted = 1;
    g_flight = g_enc;
    g_enc = -1;
    return 0;
}

// Вызывается под g_defer_lock.
static int stage_wait(int idx) {
    metal_stage *s = &g_stage[idx];
    double t0;
    int i, rc = 0;

    if (!s->submitted) return 0;
    t0 = now_ms();
    dispatch_semaphore_wait(s->sem, DISPATCH_TIME_FOREVER);
    g_wait_ns += (uint64_t)((now_ms() - t0) * 1e6);
    g_defer_waits++;
    {
        double gpu = (s->cb.GPUEndTime - s->cb.GPUStartTime) * 1e9;
        if (gpu > 0.0) g_gpu_ns += (uint64_t)gpu;
        if (getenv("VTM_METAL_TRACE"))
            metal_log("defer: %d GEMV, GPU %.3f мс", s->nres,
                      (s->cb.GPUEndTime - s->cb.GPUStartTime) * 1e3);
    }
    if (s->cb.status != MTLCommandBufferStatusCompleted || s->cb.error) {
        if (g_fail == 0)
            metal_log("отложенный буфер: %s — дальше CPU-откат",
                      s->cb.error ? [[s->cb.error localizedDescription] UTF8String]
                                  : "status != completed");
        metal_set_err("отложенный командный буфер не завершился");
        g_fail++;
        rc = -1;
    } else {
        for (i = 0; i < s->nres; i++) {
            if (s->res[i].dst && s->res[i].bytes)
                memcpy(s->res[i].dst,
                       (uint8_t *)[s->yb contents] + s->res[i].off,
                       s->res[i].bytes);
        }
        g_calls += (uint64_t)s->nres;
    }
    [s->cb release];
    s->cb = nil;
    s->enc = nil;
    s->submitted = 0;
    s->nres = 0;
    s->xused = 0;
    s->yused = 0;
    return rc;
}

// Инъекция отказа для проверки отката (VTM_METAL_FORCE_FAIL=1): первая же
// синхронизация теряет результаты и возвращает -1 — interp.c обязан
// пересчитать очередь на CPU. Нужна тестам/бенчмарку, в обычном прогоне
// переменной нет и ветка не исполняется.
static int force_fail(void) {
    static int used;
    const char *e = getenv("VTM_METAL_FORCE_FAIL");
    if (used || !e || !*e || *e == '0') return 0;
    used = 1;
    return 1;
}

// Отдельная инъекция для графа токена: тест откатов в bench_backend проверяет
// и гибрид (GEMM), и FULL (граф) в одном процессе, а force_fail() — одноразовый.
static int force_fail_graph(void) {
    static int used;
    const char *e = getenv("VTM_METAL_FORCE_FAIL_GRAPH");
    if (used || !e || !*e || *e == '0') return 0;
    used = 1;
    return 1;
}

// Вызывается под g_defer_lock: открыть сцену для кодирования.
static int stage_open(void) {
    int idx, i;

    if (g_enc >= 0) return 0;
    for (i = 0; i < VTM_METAL_STAGES; i++) {
        if (i == g_flight) continue;
        idx = i;
        if (g_stage[idx].submitted) {
            if (stage_wait(idx) != 0) return -1;
        }
        g_enc = idx;
        return 0;
    }
    return -1;
}

// Закрыть/отдать текущую сцену. Возвращает -1, если дождаться предыдущей не
// удалось (тогда вызывающий обязан уйти на CPU).
static int defer_submit_locked(void) {
    int idx;
    int fail = force_fail();

    if (fail && g_enc >= 0) {
        // Результаты открытой сцены «потеряны»: отпускаем буфер, не отдавая
        // его GPU. Отданную ранее сцену дождёмся ниже, но и её результаты
        // будут перезаписаны CPU-пересчётом.
        metal_stage *s = &g_stage[g_enc];
        if (s->enc) { [s->enc endEncoding]; [s->enc release]; s->enc = nil; }
        [s->cb release];
        s->cb = nil;
        s->nres = 0;
        s->xused = 0;
        s->yused = 0;
        g_enc = -1;
    }
    if (g_enc >= 0) {
        if (stage_submit() != 0) return -1;
    }
    if (g_flight >= 0) {
        idx = g_flight;
        if (stage_wait(idx) != 0) {
            g_flight = -1;
            return -1;
        }
        g_flight = -1;
    }
    if (fail) {
        metal_set_err("VTM_METAL_FORCE_FAIL: инъекция отказа Metal");
        g_fail++;
        g_defer_failed = 1;
        return -1;
    }
    return g_defer_failed ? -1 : 0;
}

static int metal_strict_env(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *st = getenv("VTM_Q4K_STRICT");
        cached = (st && *st && *st != '0') ? 1 : 0;
    }
    return cached;
}

int vtm_metal_defer_begin(void) {
    if (g_init == -1) return -1;
    if (metal_strict_env()) return -1;
    if (!__sync_bool_compare_and_swap(&g_defer_busy, 0, 1)) return -1;
    if (!metal_device()) {
        __sync_lock_release(&g_defer_busy);
        return -1;
    }
    g_defer_failed = 0;
    __atomic_store_n(&g_defer_owner, (uintptr_t)pthread_self(),
                     __ATOMIC_RELAXED);
    __atomic_store_n(&g_defer_on, 1, __ATOMIC_RELEASE);
    g_enc = -1;
    g_flight = -1;
    return 0;
}

int vtm_metal_defer_active(void) {
    if (!__atomic_load_n(&g_defer_on, __ATOMIC_ACQUIRE)) return 0;
    return __atomic_load_n(&g_defer_owner, __ATOMIC_RELAXED)
           == (uintptr_t)pthread_self();
}

// Кэш MTLBuffer (g_binds/g_nbinds/g_tick) и инициализация Metal по контракту
// живут под g_lock: так их держат синхронный GEMV и батчевый GEMM. Очередь
// декода держит свой g_defer_lock и до 0199 звала metal_init()/metal_bind()
// только с ним — при двух контекстах в разных потоках (FULL-декод и BATCH-
// prefill) это была настоящая гонка на LRU-вытеснении и на g_init. Берём
// g_lock на время init/bind; порядок замков — defer -> g, как в
// vtm_metal_forget_all(), обратного порядка нет.
//
// Это не причина мусора 0199 (там детерминированный шаг строки GEMM, см.
// metal_gemm_body), а тот самый разрыв синхронизации, который искали по
// гипотезе: он существует, но в одиночном прогоне не проявлялся.
static int defer_metal_init(void) {
    int rc;

    pthread_mutex_lock(&g_lock);
    rc = metal_init();
    pthread_mutex_unlock(&g_lock);
    return rc;
}

static int defer_metal_bind(const void *ptr, size_t bytes, id<MTLBuffer> *out,
                            uint64_t *off) {
    int rc;

    pthread_mutex_lock(&g_lock);
    rc = metal_bind(ptr, bytes, out, off, NULL);
    pthread_mutex_unlock(&g_lock);
    return rc;
}

int vtm_metal_defer_gemv(int kind, float *y, const void *w,
                         const float *scales, const float *x,
                         int rows, int cols) {
    id<MTLBuffer> wb = nil, sb = nil;
    uint64_t woff = 0, scoff = 0;
    vtm_metal_args a;
    metal_stage *s;
    size_t wbytes, scbytes, xbytes, ybytes, xoff, yoff;
    int rc = 1;
    double gpu;

    if (!vtm_metal_defer_active()) return 1;
    if (!y || !w || !x || rows <= 0 || cols <= 0) return 1;
    if (metal_strict_env()) return 1;
    if ((double)rows * (double)cols < vtm_metal_min_mac()) return 1;
    if (kind == 0 || kind == 1) {
        if (cols % 256 != 0) return 1;
    } else if (kind != 2 || !scales) {
        return 1;
    }

    pthread_mutex_lock(&g_defer_lock);
    if (defer_metal_init() != 0) {
        g_fail++;
        rc = -1;
        goto out;
    }
    if (kind == 2) {
        if (!g_ps_i8) { rc = 1; goto out; }
    } else if ((kind == 0 && !g_ps_q4k) || (kind == 1 && !g_ps_q6k)) {
        rc = 1;
        goto out;
    }

    if (kind == 0) {
        wbytes = ((size_t)rows * (size_t)cols / 256u) * 144u;
        a.nblocks = (uint32_t)(cols / 256);
    } else if (kind == 1) {
        wbytes = ((size_t)rows * (size_t)cols / 256u) * 210u;
        a.nblocks = (uint32_t)(cols / 256);
    } else {
        wbytes = (size_t)rows * (size_t)cols;
        a.nblocks = 0;
    }
    if (defer_metal_bind(w, wbytes, &wb, &woff) != 0) {
        g_fail++;
        rc = 1;
        goto out;
    }
    if (kind == 2) {
        uint64_t n = (uint64_t)rows * (uint64_t)cols;
        scbytes = (size_t)(n / 128u + (n % 128u ? 1u : 0u)) * sizeof(float);
        if (defer_metal_bind(scales, scbytes, &sb, &scoff) != 0) {
            g_fail++;
            rc = 1;
            goto out;
        }
        a.sc_off = scoff / sizeof(float);
    } else {
        a.sc_off = 0;
    }
    a.rows = (uint32_t)rows;
    a.cols = (uint32_t)cols;
    a.n_tokens = 1;
    a.w_off = woff;

    // Сцена, место под очередной dispatch. Переполнение — сброс и ожидание.
    if (stage_open() != 0) { rc = -1; goto out; }
    s = &g_stage[g_enc];
    xbytes = (size_t)cols * sizeof(float);
    ybytes = (size_t)rows * sizeof(float);
    xoff = (s->xused + 255u) & ~(size_t)255u;
    yoff = (s->yused + 255u) & ~(size_t)255u;
    if (s->nres >= VTM_METAL_MAX_DEFER
        || yoff + ybytes > VTM_METAL_YBUF_MAX
        || xoff + xbytes > VTM_METAL_YBUF_MAX) {
        // Не помещается: закрываем сцену, ждём её и открываем заново.
        if (defer_submit_locked() != 0) { rc = -1; goto out; }
        if (stage_open() != 0) { rc = -1; goto out; }
        s = &g_stage[g_enc];
        xoff = 0;
        yoff = 0;
    }
    // Рост staging-буфера при уже закодированных dispatches нельзя делать
    // «на месте»: они ссылаются на старый буфер, а копировать результат мы
    // будем из нового. Поэтому сначала закрываем сцену и ждём её, потом растим.
    if ((!s->xb || s->xcap < xoff + xbytes
         || !s->yb || s->ycap < yoff + ybytes) && s->nres > 0) {
        if (defer_submit_locked() != 0) { rc = -1; goto out; }
        if (stage_open() != 0) { rc = -1; goto out; }
        s = &g_stage[g_enc];
        xoff = 0;
        yoff = 0;
    }
    if (metal_reserve(&s->xb, &s->xcap, xoff + xbytes) != 0
        || metal_reserve(&s->yb, &s->ycap, yoff + ybytes) != 0) {
        g_fail++;
        rc = 1;
        goto out;
    }
    memcpy((uint8_t *)[s->xb contents] + xoff, x, xbytes);
    if (!s->cb) {
        s->cb = [[g_queue commandBuffer] retain];
        if (!s->cb) {
            metal_set_err("commandBuffer вернул nil (defer)");
            g_fail++;
            rc = 1;
            goto out;
        }
        s->enc = [[s->cb computeCommandEncoder] retain];
        if (!s->sem) s->sem = dispatch_semaphore_create(0);
    }
    {
        NSUInteger rpt = 4;
        id<MTLComputePipelineState> ps = metal_gemv_ps(kind, &rpt);

        [s->enc setComputePipelineState:ps];
        [s->enc setBuffer:wb offset:0 atIndex:0];
        [s->enc setBuffer:s->xb offset:xoff atIndex:1];
        [s->enc setBuffer:s->yb offset:yoff atIndex:2];
        [s->enc setBytes:&a length:sizeof a atIndex:3];
        if (kind == 2) [s->enc setBuffer:sb offset:0 atIndex:4];
        [s->enc dispatchThreadgroups:
            MTLSizeMake(((NSUInteger)rows + rpt - 1u) / rpt, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    }
    s->res[s->nres].dst = y;
    s->res[s->nres].bytes = ybytes;
    s->res[s->nres].off = yoff;
    s->nres++;
    s->xused = xoff + xbytes;
    s->yused = yoff + ybytes;
    g_defer_ops++;
    rc = 0;
out:
    pthread_mutex_unlock(&g_defer_lock);
    (void)gpu;
    return rc;
}

int vtm_metal_defer_submit(void) {
    int rc;

    if (!vtm_metal_defer_active()) return 1;
    pthread_mutex_lock(&g_defer_lock);
    rc = stage_submit();
    pthread_mutex_unlock(&g_defer_lock);
    return rc == 0 ? 0 : -1;
}

int vtm_metal_defer_wait(void) {
    int rc;

    if (!__atomic_load_n(&g_defer_on, __ATOMIC_ACQUIRE)) return 0;
    if (__atomic_load_n(&g_defer_owner, __ATOMIC_RELAXED)
        != (uintptr_t)pthread_self())
        return -1;
    pthread_mutex_lock(&g_defer_lock);
    rc = defer_submit_locked();
    pthread_mutex_unlock(&g_defer_lock);
    if (g_defer_failed) rc = -1;
    return rc;
}

int vtm_metal_defer_end(void) {
    int rc = 0;

    if (!__atomic_load_n(&g_defer_on, __ATOMIC_ACQUIRE)
        || __atomic_load_n(&g_defer_owner, __ATOMIC_RELAXED)
           != (uintptr_t)pthread_self())
        return 0;
    pthread_mutex_lock(&g_defer_lock);
    rc = defer_submit_locked();
    if (g_defer_failed) rc = -1;
    __atomic_store_n(&g_defer_on, 0, __ATOMIC_RELEASE);
    __sync_lock_release(&g_defer_busy);
    pthread_mutex_unlock(&g_defer_lock);
    return rc;
}

// --- фаза 2/0214: батчевый GEMM prefill -------------------------------------
//
// 0214. Было (0196/0211): N токенов пачки гонялись через ОДИН dispatch на
// chunk (до 32 токенов), но каждый chunk — отдельный командный буфер с
// ожиданием, а активации хост транспонировал дважды: в b->xt (batch_tmv_apply)
// и в g_gxbuf (ядро). Профиль 0211: GPU 3,73 с, хост-CPU 3,62 с, стена 8,65 с.
//
// Стало:
//   1) ядро читает вход и пишет выход по шагам (x_stride/y_stride) и смещениям
//      (x_off/y_off) — активации пачки лежат в арене interp.c и не копируются
//      ни на хост, ни обратно (zero-copy MTLBuffer поверх mmap-арены);
//   2) все chunk'и матрицы и подряд идущие матрицы слоя (q/k/v, gate/up)
//      кодируются в ОДИН MTLCommandBuffer: CPU ждёт один раз, когда ему правда
//      нужны числа (vtm_metal_gemm_flush из interp.c);
//   3) ядро тайловое (vtm_*_gemm_t16/t32 в vtm_metal.metal): два ряда на поток
//      и переиспользование float4 активаций из threadgroup-памяти.
//
// Отказ (-1/1) на любом шаге — прежний CPU-путь: прогон не падает.

// Должно совпадать с GemmArgs в vtm_metal.metal (sizeof == 56).
typedef struct {
    uint32_t rows;
    uint32_t cols;
    uint32_t nblocks;
    uint32_t n_tokens;
    uint32_t x_off;
    uint32_t y_off;
    uint32_t x_stride;
    uint32_t y_stride;
    uint32_t y_rstride;
    uint32_t pad0;
    uint64_t w_off;
    uint64_t sc_off;
} vtm_metal_gemm_args;

#define VTM_METAL_TILE_TG   256u
#define VTM_METAL_TILE_ROWS 16u
// Хвост пачки: ядро варианта NT считает NT токенов, даже если их меньше.
// Арена пачки выделяется с этим запасом (interp.c), чтобы чтение за последним
// токеном не вышло за отображение.
#define VTM_METAL_GEMM_PAD  32

// Очередь батчевых GEMM: один командный буфер на группу матриц.
typedef struct {
    id<MTLCommandBuffer> cb;
    id<MTLComputeCommandEncoder> enc;
    int nops;
    int failed;
} metal_gemmq;

static metal_gemmq g_gq;

// Арена пачки. Одна на прогон пачки (b->mem в interp.c), привязывается ОДИН
// раз: у матрицы меняется только смещение тензора в ней. Привязка «на каждую
// матрицу» (no-copy MTLBuffer на 38 МиБ над 608-КиБ шагом токенов) стоила
// дороже самого GEMM: драйвер отображает тысячи страниц, и живые 250 токенов
// шли 6,2 с при 1,4 с на 64 (замер 0214 до этой правки).
#define VTM_METAL_ARENA_MAX 4
typedef struct {
    const void *ptr;
    size_t bytes;
    id<MTLBuffer> buf;
} metal_arena;
static metal_arena g_arenas[VTM_METAL_ARENA_MAX];

static int gemm_tile_env(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("VTM_METAL_GEMM_KERNEL");
        cached = (e && *e && strcmp(e, "base") == 0) ? 0 : 1;
    }
    return cached;
}

// Ядро варианта по числу токенов chunk'а. vn — сколько токенов вариант
// покрывает (16 или 32); tokens >= vn идут следующим chunk'ом.
static id<MTLComputePipelineState> gemm_tile_ps(int kind, int nt, int *vn) {
    int v = nt <= 16 ? 0 : 1;

    *vn = v == 0 ? 16 : 32;
    if (kind == 0) return g_ps_q4k_tile[v];
    if (kind == 1) return g_ps_q6k_tile[v];
    return g_ps_i8_tile[v];
}

// Размеры буферов весов/масштабов и число блоков. 0 — форма не та.
static int gemm_wsize(int kind, int rows, int cols, size_t *wbytes,
                      size_t *scbytes, uint32_t *nblocks) {
    if (kind == 0 || kind == 1) {
        if (cols % 256 != 0) return -1;
        *nblocks = (uint32_t)(cols / 256);
        *wbytes = ((size_t)rows * (size_t)cols / 256u)
                * (kind == 0 ? 144u : 210u);
        *scbytes = 0;
        return 0;
    }
    if (kind != 2 || cols % 128 != 0) return -1;
    {
        uint64_t n = (uint64_t)rows * (uint64_t)cols;

        *nblocks = 0;
        *wbytes = (size_t)n;
        *scbytes = (size_t)(n / 128u + (n % 128u ? 1u : 0u)) * sizeof(float);
    }
    return 0;
}

// Привязать арену пачки (zero-copy). Вызывается под g_lock. 0 — готово,
// -1 — не вышло (тогда GEMM пойдёт CPU-путём).
//
// Буфер создаётся СВОЙ, не через metal_bind: кэш весов живёт до vtm_free, а
// арена пачки освобождается в конце каждого vtm_run_batch — no-copy буфер не
// должен переживать своё отображение (иначе «страницы, отданные другому
// mmap»). Освобождение — gemm_arena_drop по указателю.
static int gemm_arena_attach(const void *ptr, size_t bytes) {
    int i, slot = -1;
    long page = sysconf(_SC_PAGESIZE);
    size_t plen;

    if (!ptr || !bytes) return -1;
    if (page <= 0) page = 16384;
    if (((uintptr_t)ptr & (uintptr_t)((uintptr_t)page - 1u)) != 0u) return -1;
    plen = (bytes + (size_t)page - 1u) & ~((size_t)page - 1u);
    for (i = 0; i < VTM_METAL_ARENA_MAX; i++) {
        if (g_arenas[i].buf && g_arenas[i].ptr == ptr
            && g_arenas[i].bytes == plen)
            return 0;
        if (!g_arenas[i].buf && slot < 0) slot = i;
    }
    if (slot < 0) {
        // Все слоты заняты (несколько контекстов): вытесняем первый.
        slot = 0;
        [g_arenas[slot].buf release];
        g_arenas[slot].buf = nil;
    }
    {
        id<MTLBuffer> b = [g_dev newBufferWithBytesNoCopy:(void *)ptr
                                                   length:plen
                                                  options:MTLResourceStorageModeShared
                                                deallocator:nil];
        if (!b) {
            metal_set_err("арена пачки: no-copy buffer не создался (%zu Б)",
                          plen);
            return -1;
        }
        g_arenas[slot].ptr = ptr;
        g_arenas[slot].bytes = plen;
        g_arenas[slot].buf = b;
        g_nocopy_bytes += (uint64_t)plen;
    }
    return 0;
}

// Отпустить арену (ptr == NULL — все). Вызывается под g_lock.
static void gemm_arena_drop(const void *ptr) {
    int i;

    for (i = 0; i < VTM_METAL_ARENA_MAX; i++) {
        if (!g_arenas[i].buf) continue;
        if (ptr && g_arenas[i].ptr != ptr) continue;
        [g_arenas[i].buf release];
        g_arenas[i].buf = nil;
        g_arenas[i].ptr = NULL;
        g_arenas[i].bytes = 0;
    }
}

// Найти арену, в которой лежит [p, p+need). Возвращает буфер и смещение в
// байтах; nil — не в арене (тогда привязываем тензор отдельно).
static id<MTLBuffer> gemm_arena_lookup(const void *p, size_t need,
                                       uint64_t *off_out) {
    int i;
    uintptr_t a = (uintptr_t)p;

    for (i = 0; i < VTM_METAL_ARENA_MAX; i++) {
        uintptr_t base;

        if (!g_arenas[i].buf) continue;
        base = (uintptr_t)g_arenas[i].ptr;
        if (a < base || a - base > g_arenas[i].bytes) continue;
        if (need > g_arenas[i].bytes - (a - base)) continue;
        *off_out = (uint64_t)(a - base);
        return g_arenas[i].buf;
    }
    return nil;
}

// Один dispatch. Возвращает vn (сколько токенов покрыто), -1 — нет ядра.
static int gemm_dispatch(id<MTLComputeCommandEncoder> enc, int kind,
                         id<MTLBuffer> wb, uint64_t woff,
                         id<MTLBuffer> sb, uint64_t scoff,
                         id<MTLBuffer> xb, id<MTLBuffer> yb,
                         int rows, int cols, int nt, long x_off,
                         int x_stride, long y_off, int y_stride,
                         int y_rstride, int tile) {
    int vn = 0;

    if (tile) {
        vtm_metal_gemm_args a;
        id<MTLComputePipelineState> ps = gemm_tile_ps(kind, nt, &vn);

        if (!ps) return -1;
        memset(&a, 0, sizeof a);
        a.rows = (uint32_t)rows;
        a.cols = (uint32_t)cols;
        a.nblocks = kind == 2 ? 0u : (uint32_t)(cols / 256);
        a.n_tokens = (uint32_t)nt;
        a.x_off = (uint32_t)x_off;
        a.y_off = (uint32_t)y_off;
        a.x_stride = (uint32_t)x_stride;
        a.y_stride = (uint32_t)y_stride;
        a.y_rstride = (uint32_t)y_rstride;
        a.w_off = woff;
        a.sc_off = scoff / sizeof(float);
        [enc setComputePipelineState:ps];
        [enc setBuffer:wb offset:0 atIndex:0];
        [enc setBuffer:xb offset:0 atIndex:1];
        [enc setBuffer:yb offset:0 atIndex:2];
        [enc setBytes:&a length:sizeof a atIndex:3];
        if (kind == 2) [enc setBuffer:sb offset:0 atIndex:4];
        [enc dispatchThreadgroups:
            MTLSizeMake(((NSUInteger)rows + VTM_METAL_TILE_ROWS - 1u)
                        / VTM_METAL_TILE_ROWS, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(VTM_METAL_TILE_TG, 1, 1)];
        return vn;
    }
    return -1;
}

// Все chunk'и одной матрицы. Смещения — в float от начала буферов.
static int gemm_encode_matrix(id<MTLComputeCommandEncoder> enc, int kind,
                              id<MTLBuffer> wb, uint64_t woff,
                              id<MTLBuffer> sb, uint64_t scoff,
                              id<MTLBuffer> xb, id<MTLBuffer> yb,
                              int rows, int cols, int n_tokens, long x_off,
                              int x_stride, long y_off, int y_stride,
                              int y_rstride, int tile) {
    int j0 = 0;

    while (j0 < n_tokens) {
        int nt = n_tokens - j0, vn, step;

        if (nt > VTM_METAL_GEMM_MAX_N) nt = VTM_METAL_GEMM_MAX_N;
        vn = gemm_dispatch(enc, kind, wb, woff, sb, scoff, xb, yb, rows, cols,
                           nt, x_off + (long)j0 * x_stride, x_stride,
                           y_off + (long)j0 * y_stride, y_stride, y_rstride,
                           tile);
        if (vn < 0) return -1;
        step = nt < vn ? nt : vn;
        j0 += step;
    }
    return 0;
}

// Открыть/закрыть общий командный буфер группы. Вызывается под g_lock.
static int gemmq_open(void) {
    if (g_gq.enc) return 0;
    if (g_gq.failed) return -1;
    if (!g_gq.cb) g_gq.cb = [[g_queue commandBuffer] retain];
    if (!g_gq.cb) {
        metal_set_err("commandBuffer вернул nil (gemm)");
        return -1;
    }
    g_gq.enc = [[g_gq.cb computeCommandEncoder] retain];
    if (!g_gq.enc) {
        metal_set_err("computeCommandEncoder вернул nil (gemm)");
        [g_gq.cb release];
        g_gq.cb = nil;
        return -1;
    }
    return 0;
}

// Закрыть группу и дождаться. 0 — всё посчитано, -1 — сбой Metal.
static int gemmq_flush_locked(void) {
    double gpu;

    if (!g_gq.enc && !g_gq.cb) return g_gq.failed ? -1 : 0;
    if (g_gq.enc) {
        [g_gq.enc endEncoding];
        [g_gq.enc release];
        g_gq.enc = nil;
    }
    if (!g_gq.cb) return g_gq.failed ? -1 : 0;
    if (g_gq.nops > 0) {
        [g_gq.cb commit];
        [g_gq.cb waitUntilCompleted];
        gpu = (g_gq.cb.GPUEndTime - g_gq.cb.GPUStartTime) * 1e9;
        if (gpu > 0.0) g_gpu_ns += (uint64_t)gpu;
        if (getenv("VTM_METAL_TRACE"))
            metal_log("gemm-группа: %d dispatch, GPU %.3f мс", g_gq.nops,
                      (g_gq.cb.GPUEndTime - g_gq.cb.GPUStartTime) * 1e3);
        if (g_gq.cb.status != MTLCommandBufferStatusCompleted
            || g_gq.cb.error) {
            if (g_fail == 0)
                metal_log("gemm-группа: %s — дальше CPU-откат",
                          g_gq.cb.error
                              ? [[g_gq.cb.error localizedDescription] UTF8String]
                              : "status != completed");
            metal_set_err("командный буфер группы не завершился");
            g_fail++;
            g_gq.failed = 1;
        }
    }
    [g_gq.cb release];
    g_gq.cb = nil;
    g_gq.nops = 0;
    return g_gq.failed ? -1 : 0;
}

int vtm_metal_gemm_flush(void) {
    int rc;

    if (!__atomic_load_n(&g_gq_active, __ATOMIC_ACQUIRE)) return 0;
    pthread_mutex_lock(&g_lock);
    rc = gemmq_flush_locked();
    pthread_mutex_unlock(&g_lock);
    return rc;
}

// 0214: GEMM матрицы со шагами по токенам, прямо в памяти хоста (zero-copy).
// 0 — dispatch'и в очереди (результат появится после vtm_metal_gemm_flush);
// 1 — не взялся (форма/буфер/нет тайлового ядра): считать CPU-путём;
// -1 — Metal отказал (тоже CPU-путь, режим выключается вызывающим).
int vtm_metal_qk_gemm_arena(int kind, const void *w, const float *scales,
                            int rows, int cols, int n_tokens,
                            const float *x, int x_stride,
                            float *y, int y_stride) {
    id<MTLBuffer> wb = nil, sb = nil, xb = nil, yb = nil;
    uint64_t woff = 0, scoff = 0, xoff = 0, yoff = 0;
    size_t wbytes, scbytes, xbytes, ybytes;
    uint32_t nblocks = 0;
    int copied = 0;

    if (!w || !x || !y || rows <= 0 || cols <= 0) return 1;
    if (n_tokens < 1) return 1;
    if (!gemm_tile_env() || metal_strict_env()) return 1;
    if (force_fail()) {
        metal_set_err("VTM_METAL_FORCE_FAIL: инъекция отказа Metal (GEMM)");
        g_fail++;
        vtm_metal_set_mode(VTM_METAL_OFF);
        return -1;
    }
    // float4-загрузки входа: и указатель, и шаг должны быть выровнены.
    if (((uintptr_t)x & 15u) != 0u || (x_stride & 3) != 0) return 1;
    if (gemm_wsize(kind, rows, cols, &wbytes, &scbytes, &nblocks) != 0)
        return 1;
    if (kind == 2 && !scales) return 1;
    xbytes = ((size_t)(n_tokens - 1) * (size_t)x_stride + (size_t)cols)
           * sizeof(float);
    ybytes = ((size_t)(n_tokens - 1) * (size_t)y_stride + (size_t)rows)
           * sizeof(float);

    pthread_mutex_lock(&g_lock);
    if (metal_init() != 0) {
        g_fail++;
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    if (!g_ps_q4k_tile[1]) {
        pthread_mutex_unlock(&g_lock);
        return 1;
    }
    if (metal_bind(w, wbytes, &wb, &woff, &copied) != 0) {
        g_fail++;
        pthread_mutex_unlock(&g_lock);
        return 1;
    }
    if (kind == 2
        && metal_bind(scales, scbytes, &sb, &scoff, &copied) != 0) {
        g_fail++;
        pthread_mutex_unlock(&g_lock);
        return 1;
    }
    // Вход и выход — из привязанной арены пачки (одна привязка на прогон).
    // Не в арене (общая активация, чужой буфер) — своя no-copy привязка.
    xb = gemm_arena_lookup(x, xbytes, &xoff);
    if (!xb && (metal_bind(x, xbytes, &xb, &xoff, &copied) != 0 || copied)) {
        pthread_mutex_unlock(&g_lock);
        return 1;
    }
    yb = gemm_arena_lookup(y, ybytes, &yoff);
    if (!yb && (metal_bind(y, ybytes, &yb, &yoff, &copied) != 0 || copied)) {
        pthread_mutex_unlock(&g_lock);
        return 1;
    }
    if (gemmq_open() != 0) {
        g_fail++;
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    if (gemm_encode_matrix(g_gq.enc, kind, wb, woff, sb, scoff, xb, yb, rows,
                           cols, n_tokens, (long)(xoff / sizeof(float)),
                           x_stride, (long)(yoff / sizeof(float)), y_stride,
                           1, 1) != 0) {
        pthread_mutex_unlock(&g_lock);
        return 1;
    }
    g_gq.nops += (n_tokens + 15) / 16 + 1;
    g_gemm_calls++;
    g_calls++;
    __atomic_store_n(&g_gq_active, 1, __ATOMIC_RELEASE);
    // VTM_METAL_GEMM_GROUP=0 — по командному буферу на матрицу (A/B группировки
    // и разбивка GPU-времени по матрицам в VTM_METAL_TRACE).
    {
        static int group = -1;

        if (group < 0) {
            const char *e = getenv("VTM_METAL_GEMM_GROUP");
            group = (e && *e && *e == '0') ? 0 : 1;
        }
        if (!group && gemmq_flush_locked() != 0) {
            pthread_mutex_unlock(&g_lock);
            return -1;
        }
    }
    pthread_mutex_unlock(&g_lock);
    return 0;
}

// 0214: attention пачки. q/out — в арене пачки (шаг stride на токен), K/V — в
// кольцевом кэше модели (общий, zero-copy). 0 — dispatch в очереди группы,
// 1 — не взялся (форма/буфер), -1 — Metal отказал.
typedef struct {
    uint32_t n_heads;
    uint32_t rep;
    uint32_t head_dim;
    uint32_t kv_dim;
    uint32_t win;
    uint32_t n_tokens;
    uint32_t pos0;
    uint32_t stride;
    uint32_t q_off;
    uint32_t out_off;
    uint32_t pad0;
    uint32_t pad1;
    float    scale;
} vtm_metal_attn_batch_args;

int vtm_metal_attn_prefill(float *out, const float *q, const float *kc,
                           const float *vc, int n_heads, int n_kv_heads,
                           int head_dim, int kv_dim, int win, int n_tokens,
                           int pos0, int q_stride, int out_stride) {
    id<MTLBuffer> qb = nil, ob = nil, kb = nil, vb = nil;
    uint64_t qoff = 0, ooff = 0, koff = 0, voff = 0;
    vtm_metal_attn_batch_args a;
    int copied = 0;

    if (!out || !q || !kc || !vc) return 1;
    if (n_heads <= 0 || n_kv_heads <= 0 || n_heads % n_kv_heads != 0
        || head_dim <= 0 || head_dim > 256 || kv_dim != n_kv_heads * head_dim
        || win <= 0 || n_tokens <= 0 || pos0 < 0)
        return 1;
    if (metal_strict_env()) return 1;
    if (force_fail()) {
        metal_set_err("VTM_METAL_FORCE_FAIL: инъекция отказа Metal (attn)");
        g_fail++;
        vtm_metal_set_mode(VTM_METAL_OFF);
        return -1;
    }
    a.n_heads = (uint32_t)n_heads;
    a.rep = (uint32_t)(n_heads / n_kv_heads);
    a.head_dim = (uint32_t)head_dim;
    a.kv_dim = (uint32_t)kv_dim;
    a.win = (uint32_t)win;
    a.n_tokens = (uint32_t)n_tokens;
    a.pos0 = (uint32_t)pos0;
    a.stride = (uint32_t)q_stride;
    a.q_off = 0;
    a.out_off = 0;
    a.pad0 = a.pad1 = 0;
    a.scale = (float)(1.0 / sqrt((double)head_dim));

    pthread_mutex_lock(&g_lock);
    if (metal_init() != 0) {
        g_fail++;
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    if (!g_ps_attn_batch) {
        pthread_mutex_unlock(&g_lock);
        return 1;
    }
    if (out_stride != q_stride) {
        pthread_mutex_unlock(&g_lock);
        return 1;
    }
    {
        size_t qbytes = ((size_t)(n_tokens - 1) * (size_t)q_stride
                         + (size_t)n_heads * (size_t)head_dim) * sizeof(float);
        size_t ocap = (size_t)win * (size_t)kv_dim * sizeof(float);

        qb = gemm_arena_lookup(q, qbytes, &qoff);
        if (!qb && (metal_bind(q, qbytes, &qb, &qoff, &copied) != 0
                    || copied)) {
            pthread_mutex_unlock(&g_lock);
            return 1;
        }
        ob = gemm_arena_lookup(out, qbytes, &ooff);
        if (!ob && (metal_bind(out, qbytes, &ob, &ooff, &copied) != 0
                    || copied)) {
            pthread_mutex_unlock(&g_lock);
            return 1;
        }
        // Кэш кольца — общий тензор модели: читать можно и из копии, но копия
        // устарела бы (KVW пишет каждый токен), поэтому только zero-copy.
        if (metal_bind(kc, ocap, &kb, &koff, &copied) != 0 || copied
            || metal_bind(vc, ocap, &vb, &voff, &copied) != 0 || copied) {
            pthread_mutex_unlock(&g_lock);
            return 1;
        }
        a.q_off = (uint32_t)(qoff / sizeof(float));
        a.out_off = (uint32_t)(ooff / sizeof(float));
        if (gemmq_open() != 0) {
            g_fail++;
            pthread_mutex_unlock(&g_lock);
            return -1;
        }
        [g_gq.enc setComputePipelineState:g_ps_attn_batch];
        // K/V-кэш — из mmap модели: его смещение внутри страничного no-copy
        // буфера передаём как offset (иначе ядро читало бы начало страницы).
        [g_gq.enc setBuffer:qb offset:0 atIndex:0];
        [g_gq.enc setBuffer:kb offset:(NSUInteger)koff atIndex:1];
        [g_gq.enc setBuffer:vb offset:(NSUInteger)voff atIndex:2];
        [g_gq.enc setBuffer:ob offset:0 atIndex:3];
        [g_gq.enc setBytes:&a length:sizeof a atIndex:4];
        [g_gq.enc dispatchThreadgroups:
            MTLSizeMake((NSUInteger)n_tokens * (NSUInteger)n_heads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }
    g_gq.nops++;
    g_calls++;
    __atomic_store_n(&g_gq_active, 1, __ATOMIC_RELEASE);
    pthread_mutex_unlock(&g_lock);
    return 0;
}

// --- прежний синхронный API (тесты, бенчмарк, откат) ------------------------
//
// Раскладка входа — x[t*n_tokens + j], выхода — y[r*n_tokens + j] (как у
// батчевых ядер interp.c). Вход копируется в staging с нулевым хвостом до 32
// токенов (ядро варианта NT читает ровно NT), выход тайловое ядро пишет прямо
// в y по шагам. --metal-gemm меряет именно этот путь, живой prefill — арену.
static int metal_gemm_body(int kind, float *y, const void *w,
                           const float *scales, const float *x,
                           int rows, int cols, int n_tokens) {
    id<MTLBuffer> wb = nil, sb = nil;
    uint64_t woff = 0, scoff = 0;
    id<MTLCommandBuffer> cb = nil;
    id<MTLComputeCommandEncoder> enc = nil;
    size_t wbytes, scbytes, xbytes, ybytes;
    int rc = -1, copied = 0, tile, j0, vnmax = VTM_METAL_GEMM_MAX_N;
    uint32_t nblocks = 0;

    if (!y || !w || !x || rows <= 0 || cols <= 0 || n_tokens <= 1) return -1;
    // Пачка меньше 8 токенов на GPU проигрывает CPU (замер 0196): ядро всё
    // равно считает 8 (минимум из вариантов), а накладные те же.
    if (n_tokens < 8) return -1;
    if (metal_strict_env()) return -1;
    if (force_fail()) {
        metal_set_err("VTM_METAL_FORCE_FAIL: инъекция отказа Metal (GEMM)");
        g_fail++;
        vtm_metal_set_mode(VTM_METAL_OFF);
        return -1;
    }
    tile = gemm_tile_env();
    if (gemm_wsize(kind, rows, cols, &wbytes, &scbytes, &nblocks) != 0)
        return -1;

    pthread_mutex_lock(&g_lock);
    if (metal_init() != 0) {
        g_fail++;
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    if (!g_ps_q4k_tile[0] && !g_ps_q4k_tile[1]) {
        // Тайловых ядер нет (старый metallib) — батчевый prefill пойдёт CPU.
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    if (metal_bind(w, wbytes, &wb, &woff, &copied) != 0) {
        g_fail++;
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    if (kind == 2) {
        if (!scales || metal_bind(scales, scbytes, &sb, &scoff, &copied) != 0) {
            g_fail++;
            pthread_mutex_unlock(&g_lock);
            return -1;
        }
    }
    // Вход — в staging: все токены пачки, шаг токена cols (ядро читает
    // xt[j*cols + t]), хвост до кратности 32 — нули (ядро варианта NT читает
    // ровно NT токенов). Staging свой, копия безопасна.
    {
        int ntok_pad = (n_tokens + VTM_METAL_GEMM_MAX_N - 1)
                     / VTM_METAL_GEMM_MAX_N * VTM_METAL_GEMM_MAX_N;
        float *xt;
        int t, j;
        const int stride = cols;

        xbytes = (size_t)cols * (size_t)ntok_pad * sizeof(float);
        if (metal_reserve(&g_gxbuf, &g_gxcap, xbytes) != 0) {
            g_fail++;
            pthread_mutex_unlock(&g_lock);
            return -1;
        }
        xt = (float *)[g_gxbuf contents];
        for (t = 0; t < cols; t++) {
            for (j = 0; j < n_tokens; j++)
                xt[(size_t)j * (size_t)stride + (size_t)t] =
                    x[(size_t)t * (size_t)n_tokens + (size_t)j];
            for (j = n_tokens; j < ntok_pad; j++)
                xt[(size_t)j * (size_t)stride + (size_t)t] = 0.0f;
        }
    }
    ybytes = (size_t)rows * (size_t)vnmax * sizeof(float);
    if (metal_reserve(&g_gybuf, &g_gycap, ybytes) != 0) {
        g_fail++;
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    cb = [g_queue commandBuffer];
    if (!cb) {
        metal_set_err("commandBuffer вернул nil (gemm)");
        g_fail++;
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    enc = [cb computeCommandEncoder];
    // Шаг выхода — rows: ядро пишет yt[j*rows + r] (на токен — все строки).
    if (gemm_encode_matrix(enc, kind, wb, woff, sb, scoff, g_gxbuf, g_gybuf,
                           rows, cols, n_tokens, 0, cols, 0, rows, 1,
                           tile) != 0) {
        [enc endEncoding];
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    [enc endEncoding];
    enc = nil;
    [cb commit];
    [cb waitUntilCompleted];
    {
        double gpu = (cb.GPUEndTime - cb.GPUStartTime) * 1e9;
        if (gpu > 0.0) g_gpu_ns += (uint64_t)gpu;
        if (getenv("VTM_METAL_TRACE"))
            metal_log("gemm kind %d %dx%d n=%d: GPU %.3f мс", kind, rows, cols,
                      n_tokens, (cb.GPUEndTime - cb.GPUStartTime) * 1e3);
    }
    if (cb.status != MTLCommandBufferStatusCompleted || cb.error) {
        if (g_fail == 0)
            metal_log("GEMM-буфер: %s — дальше CPU",
                      cb.error ? [[cb.error localizedDescription] UTF8String]
                               : "status != completed");
        metal_set_err("GEMM-буфер не завершился");
        g_fail++;
        vtm_metal_set_mode(VTM_METAL_OFF);
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    {
        // Выход ядра — token-major с шагом vnmax (yt[j*vnmax + r]): разнести
        // по раскладке батчевых ядер y[r*n_tokens + j].
        const float *yt = (const float *)[g_gybuf contents];
        int r;

        for (j0 = 0; j0 < n_tokens; j0 += vnmax) {
            int nt = n_tokens - j0, j;

            if (nt > vnmax) nt = vnmax;
            for (r = 0; r < rows; r++)
                for (j = 0; j < nt; j++)
                    y[(size_t)r * (size_t)n_tokens + (size_t)(j0 + j)] =
                        yt[(size_t)(j0 + j) * (size_t)rows + (size_t)r];
        }
    }
    g_gemm_calls++;
    g_calls++;
    rc = 0;
    pthread_mutex_unlock(&g_lock);
    return rc;
}

int vtm_metal_qk_gemm(int kind, float *y, const void *w, const float *scales,
                      const float *x, int rows, int cols, int n_tokens) {
    @autoreleasepool {
        return metal_gemm_body(kind, y, w, scales, x, rows, cols, n_tokens);
    }
}

int vtm_metal_gemm_pad(void) {
    return VTM_METAL_GEMM_PAD;
}

int vtm_metal_batch_arena(const void *ptr, size_t bytes) {
    int rc;

    if (!ptr || !bytes) return -1;
    if (metal_strict_env()) return -1;
    pthread_mutex_lock(&g_lock);
    if (metal_init() != 0) {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    rc = gemm_arena_attach(ptr, bytes);
    pthread_mutex_unlock(&g_lock);
    return rc;
}

void vtm_metal_batch_arena_drop(const void *ptr) {
    pthread_mutex_lock(&g_lock);
    // Сначала дождаться группы: её dispatch'и читают/пишут эту арену.
    (void)gemmq_flush_locked();
    gemm_arena_drop(ptr);
    pthread_mutex_unlock(&g_lock);
}

// Порог «мелкие формы — на CPU» (ADR 0021 §5): накладные Metal на GEMV
// 256x2048 и меньше больше выигрыша. VTM_METAL_MIN_MAC=N задаёт порог в MAC,
// 0 — отдать GPU всё. Значение читается один раз.
double vtm_metal_min_mac(void) {
    static double cached = -1.0;
    if (cached < 0.0) {
        const char *e = getenv("VTM_METAL_MIN_MAC");
        if (e && *e) {
            char *end = NULL;
            double v = strtod(e, &end);
            cached = (end && *end == '\0' && v >= 0.0) ? v : 1048576.0;
        } else {
            cached = 1048576.0;
        }
    }
    return cached;
}

// Сбросить кэш MTLBuffer: вызывается из vtm_free(), потому что после
// munmap/free память, на которую смотрел no-copy буфер, может быть отдана
// другому отображению (и Metal не обязан видеть новые страницы). Освобождать
// до munmap безопасно: все командные буферы к этому моменту завершены.
void vtm_metal_forget_all(void) {
    int i;

    // Незакоммиченный граф этого потока — выбросить до освобождения буферов:
    // после munmap/освобождения арены no-copy регистрации недействительны.
    if (__atomic_load_n(&g_graph_on, __ATOMIC_ACQUIRE)
        && __atomic_load_n(&g_graph_owner, __ATOMIC_RELAXED)
           == (uintptr_t)pthread_self())
        (void)vtm_metal_graph_end();
    pthread_mutex_lock(&g_defer_lock);
    if (__atomic_load_n(&g_defer_on, __ATOMIC_ACQUIRE)
        && __atomic_load_n(&g_defer_owner, __ATOMIC_RELAXED)
           == (uintptr_t)pthread_self()) {
        (void)defer_submit_locked();
        __atomic_store_n(&g_defer_on, 0, __ATOMIC_RELEASE);
        __sync_lock_release(&g_defer_busy);
    }
    pthread_mutex_lock(&g_lock);
    for (i = 0; i < g_nbinds; i++) {
        [g_binds[i].buf release];
        g_binds[i].buf = nil;
    }
    g_nbinds = 0;
    for (i = 0; i < VTM_METAL_ARENA_MAX; i++) {
        [g_arenas[i].buf release];
        g_arenas[i].buf = nil;
        g_arenas[i].ptr = NULL;
        g_arenas[i].bytes = 0;
    }
    [g_gq.enc release];
    g_gq.enc = nil;
    [g_gq.cb release];
    g_gq.cb = nil;
    g_gq.nops = 0;
    g_gq.failed = 0;
    __atomic_store_n(&g_gq_active, 0, __ATOMIC_RELEASE);
    [g_g.tmp release];
    g_g.tmp = nil;
    g_g.tmpcap = 0;
    g_g.tmpused = 0;
    g_g.rope_valid = 0;
    pthread_mutex_unlock(&g_lock);
    pthread_mutex_unlock(&g_defer_lock);
}


// --- фаза 3: граф одного токена ---------------------------------------------
//
// Задача 0204. Фаза 2 сократила число командных буферов до 4 на слой, но
// декод всё равно ждал CPU на attention/RMSN/SiLU: пока они считались, GPU
// стоял. Здесь RMSNorm/RoPE/SiLU/MUL/ADD/KVW и attention декода считаются
// MSL-ядрами, а ВСЕ операции токена (все слои) кодируются в ОДИН
// MTLCommandBuffer. CPU ждёт один раз — перед сэмплингом; результатом
// командного буфера к этому моменту уже записаны в арену модели (zero-copy
// буферы смотрят прямо в неё, staging-копий нет).
//
// Сессией владеет один поток; вторая одновременная попытка begin вернёт -1, и
// прогон пойдёт прежним путём (defer/CPU). Ошибка Metal на любом шаге —
// vtm_metal_graph_end() возвращает -1, interp.c выключает Metal и пересчитывает
// токен CPU-ядрами (откат, ADR 0021).

#define VTM_GRAPH_TMP_BYTES ((size_t)64u * 1024u)

int vtm_metal_graph_available(void) {
    int rc;

    if (g_init == -1) return 0;
    if (metal_strict_env()) return 0;
    pthread_mutex_lock(&g_lock);
    rc = metal_init();
    pthread_mutex_unlock(&g_lock);
    if (rc != 0) return 0;
    if (getenv("VTM_METAL_TRACE"))
        metal_log("граф: ядра norm %d rope %d silu %d mul %d add %d kvw %d "
                  "attn %d copy %d", g_ps_norm != nil, g_ps_rope != nil,
                  g_ps_silu != nil, g_ps_mul != nil, g_ps_add != nil,
                  g_ps_kvw != nil, g_ps_attn != nil, g_ps_copy != nil);
    return g_ps_norm && g_ps_rope && g_ps_silu && g_ps_mul && g_ps_add
        && g_ps_kvw && g_ps_attn && g_ps_copy;
}

int vtm_metal_graph_active(void) {
    return __atomic_load_n(&g_graph_on, __ATOMIC_RELAXED) != 0;
}

int vtm_metal_graph_begin(void) {
    if (g_init == -1) return -1;
    if (metal_strict_env()) return -1;
    if (!vtm_metal_graph_available()) return -1;
    if (!__sync_bool_compare_and_swap(&g_graph_busy, 0, 1)) return -1;
    if (g_init == -1) {
        __sync_lock_release(&g_graph_busy);
        return -1;
    }
    __atomic_store_n(&g_graph_owner, (uintptr_t)pthread_self(),
                     __ATOMIC_RELAXED);
    g_g.ops = 0;
    g_g.tmpused = 0;
    g_g.rope_valid = 0;
    g_graph_failed = 0;
    @autoreleasepool {
        g_g.cb = [[g_queue commandBuffer] retain];
    }
    if (!g_g.cb) {
        metal_set_err("commandBuffer вернул nil (граф)");
        __atomic_store_n(&g_graph_owner, (uintptr_t)0, __ATOMIC_RELAXED);
        __sync_lock_release(&g_graph_busy);
        return -1;
    }
    @autoreleasepool {
        g_g.enc = [[g_g.cb computeCommandEncoder] retain];
    }
    if (!g_g.enc) {
        [g_g.cb release];
        g_g.cb = nil;
        metal_set_err("computeCommandEncoder вернул nil (граф)");
        __atomic_store_n(&g_graph_owner, (uintptr_t)0, __ATOMIC_RELAXED);
        __sync_lock_release(&g_graph_busy);
        return -1;
    }
    g_g.t_encode0 = now_ms();
    __atomic_store_n(&g_graph_on, 1, __ATOMIC_RELEASE);
    return 0;
}

// Привязать буфер арены/весов. Копия (copied) не годится: GPU пишет в неё, а
// CPU читает арену — результаты разошлись бы молча. Поэтому копия = отказ
// графа, и токен считается прежним путём.
static int graph_bind(const void *ptr, size_t bytes, id<MTLBuffer> *out,
                      uint64_t *off) {
    uint64_t dummy = 0;
    int copied = 1, rc;

    if (!ptr || !bytes) return -1;
    if (!off) off = &dummy;   // metal_bind пишет смещение всегда
    pthread_mutex_lock(&g_lock);
    rc = metal_bind(ptr, bytes, out, off, &copied);
    pthread_mutex_unlock(&g_lock);
    if (rc != 0 || copied) return -1;
    return 0;
}

static int graph_fail(void) {
    g_graph_failed = 1;
    return -1;
}

static void graph_encode_done(void) {
    g_g.ops++;
}

int vtm_metal_graph_gemv(int kind, float *y, const void *w,
                         const float *scales, const float *x, int rows,
                         int cols) {
    id<MTLBuffer> wb = nil, sb = nil, xb = nil, yb = nil;
    uint64_t woff = 0, scoff = 0, xoff = 0, yoff = 0;
    vtm_metal_args a;
    size_t wbytes, scbytes;
    NSUInteger rpt = 4;
    id<MTLComputePipelineState> ps;

    if (!g_graph_on || g_graph_failed) return -1;
    if (!y || !w || !x || rows <= 0 || cols <= 0) return graph_fail();
    if (kind == 0 || kind == 1) {
        if (cols % 256 != 0) return graph_fail();
        wbytes = ((size_t)rows * (size_t)cols / 256u)
               * (kind == 0 ? 144u : 210u);
        a.nblocks = (uint32_t)(cols / 256);
    } else if (kind == 2) {
        uint64_t n = (uint64_t)rows * (uint64_t)cols;
        wbytes = (size_t)n;
        scbytes = (size_t)(n / 128u + (n % 128u ? 1u : 0u)) * sizeof(float);
        a.nblocks = 0;
    } else {
        return graph_fail();
    }
    if (graph_bind(w, wbytes, &wb, &woff) != 0) return graph_fail();
    if (kind == 2) {
        if (!scales || graph_bind(scales, scbytes, &sb, &scoff) != 0)
            return graph_fail();
        a.sc_off = scoff / sizeof(float);
    } else {
        a.sc_off = 0;
    }
    if (graph_bind(x, (size_t)cols * sizeof(float), &xb, &xoff) != 0)
        return graph_fail();
    if (graph_bind(y, (size_t)rows * sizeof(float), &yb, &yoff) != 0)
        return graph_fail();
    a.rows = (uint32_t)rows;
    a.cols = (uint32_t)cols;
    a.n_tokens = 1;
    a.w_off = woff;
    ps = metal_gemv_ps(kind, &rpt);
    [g_g.enc setComputePipelineState:ps];
    [g_g.enc setBuffer:wb offset:0 atIndex:0];
    [g_g.enc setBuffer:xb offset:(NSUInteger)xoff atIndex:1];
    [g_g.enc setBuffer:yb offset:(NSUInteger)yoff atIndex:2];
    [g_g.enc setBytes:&a length:sizeof a atIndex:3];
    if (kind == 2) [g_g.enc setBuffer:sb offset:0 atIndex:4];
    [g_g.enc dispatchThreadgroups:
        MTLSizeMake(((NSUInteger)rows + rpt - 1u) / rpt, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    graph_encode_done();
    return 0;
}

int vtm_metal_graph_norm(float *y, const float *x, const float *w,
                         int dim, int rows, float eps) {
    id<MTLBuffer> xb = nil, wb = nil, yb = nil;
    uint64_t xoff = 0, woff = 0, yoff = 0;
    typedef struct { uint32_t dim, rows; float eps; } Normargs;
    Normargs a;

    if (!g_graph_on || g_graph_failed) return -1;
    if (!y || !x || !w || dim <= 0 || rows <= 0) return graph_fail();
    if (rows > 65536 || dim > 65536) return graph_fail();
    if (graph_bind(x, (size_t)rows * (size_t)dim * sizeof(float), &xb, &xoff)
        != 0
        || graph_bind(w, (size_t)dim * sizeof(float), &wb, &woff) != 0
        || graph_bind(y, (size_t)rows * (size_t)dim * sizeof(float), &yb,
                      &yoff) != 0)
        return graph_fail();
    a.dim = (uint32_t)dim;
    a.rows = (uint32_t)rows;
    a.eps = eps;
    [g_g.enc setComputePipelineState:g_ps_norm];
    [g_g.enc setBuffer:xb offset:(NSUInteger)xoff atIndex:0];
    [g_g.enc setBuffer:wb offset:(NSUInteger)woff atIndex:1];
    [g_g.enc setBuffer:yb offset:(NSUInteger)yoff atIndex:2];
    [g_g.enc setBytes:&a length:sizeof a atIndex:3];
    [g_g.enc dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    graph_encode_done();
    return 0;
}

int vtm_metal_graph_rope(float *y, const float *x, int heads, int head_dim,
                         int interleaved, int pos, double theta) {
    id<MTLBuffer> xb = nil, yb = nil, csb = nil;
    uint64_t xoff = 0, yoff = 0;
    typedef struct { uint32_t heads, half, head_dim, interleaved; } Ropeargs;
    Ropeargs a;
    size_t off = 0;
    int half, ok = 0;

    if (!g_graph_on || g_graph_failed) return -1;
    if (!y || !x || heads <= 0 || head_dim < 2 || (head_dim & 1))
        return graph_fail();
    if (head_dim > 256) return graph_fail();
    half = head_dim / 2;
    if (g_g.rope_valid && g_g.rope_pos == (uint64_t)(uint32_t)pos
        && g_g.rope_dim == (uint32_t)head_dim
        && g_g.rope_int == (uint32_t)interleaved) {
        off = g_g.rope_off;
        ok = 1;
    }
    if (!ok) {
        size_t need = (size_t)head_dim * sizeof(float);
        float *dst;

        if (g_g.tmpused + need > VTM_GRAPH_TMP_BYTES) return graph_fail();
        if (!g_g.tmp || g_g.tmpcap < VTM_GRAPH_TMP_BYTES) {
            [g_g.tmp release];
            g_g.tmp = [g_dev newBufferWithLength:VTM_GRAPH_TMP_BYTES
                                         options:MTLResourceStorageModeShared];
            if (!g_g.tmp) {
                g_g.tmpcap = 0;
                return graph_fail();
            }
            g_g.tmpcap = VTM_GRAPH_TMP_BYTES;
        }
        off = g_g.tmpused;
        dst = (float *)((uint8_t *)[g_g.tmp contents] + off);
        {
            int p;
            for (p = 0; p < half; p++) {
                double inv = pow(theta, -2.0 * (double)p / (double)head_dim);
                double angle = (double)pos * inv;
                dst[p] = (float)cos(angle);
                dst[half + p] = (float)sin(angle);
            }
        }
        g_g.tmpused += need;
        g_g.rope_pos = (uint64_t)(uint32_t)pos;
        g_g.rope_dim = (uint32_t)head_dim;
        g_g.rope_int = (uint32_t)interleaved;
        g_g.rope_off = off;
        g_g.rope_valid = 1;
    }
    if (graph_bind(x, (size_t)heads * (size_t)head_dim * sizeof(float), &xb,
                   &xoff) != 0
        || graph_bind(y, (size_t)heads * (size_t)head_dim * sizeof(float), &yb,
                      &yoff) != 0)
        return graph_fail();
    csb = g_g.tmp;
    a.heads = (uint32_t)heads;
    a.half = (uint32_t)half;
    a.head_dim = (uint32_t)head_dim;
    a.interleaved = (uint32_t)(interleaved ? 1 : 0);
    {
        NSUInteger pairs = (NSUInteger)heads * (NSUInteger)half;
        [g_g.enc setComputePipelineState:g_ps_rope];
        [g_g.enc setBuffer:xb offset:(NSUInteger)xoff atIndex:0];
        [g_g.enc setBuffer:yb offset:(NSUInteger)yoff atIndex:1];
        [g_g.enc setBuffer:csb offset:(NSUInteger)off atIndex:2];
        [g_g.enc setBytes:&a length:sizeof a atIndex:3];
        [g_g.enc dispatchThreadgroups:
            MTLSizeMake((pairs + 255u) / 256u, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    graph_encode_done();
    return 0;
}

static int graph_elt(int which, float *y, const float *a_, const float *b_,
                     int n) {
    id<MTLBuffer> ab = nil, bb = nil, yb = nil;
    uint64_t aoff = 0, boff2 = 0, yoff = 0;
    typedef struct { uint32_t n; } Eltargs;
    Eltargs ea;
    NSUInteger vec;

    if (!g_graph_on || g_graph_failed) return -1;
    if (!y || !a_ || !b_ || n <= 0) return graph_fail();
    if (graph_bind(a_, (size_t)n * sizeof(float), &ab, &aoff) != 0
        || graph_bind(b_, (size_t)n * sizeof(float), &bb, &boff2) != 0
        || graph_bind(y, (size_t)n * sizeof(float), &yb, &yoff) != 0)
        return graph_fail();
    ea.n = (uint32_t)n;
    vec = (NSUInteger)((n + 3) / 4);
    [g_g.enc setComputePipelineState:
        which == 0 ? g_ps_silu : (which == 1 ? g_ps_mul : g_ps_add)];
    [g_g.enc setBuffer:ab offset:(NSUInteger)aoff atIndex:0];
    if (which != 0) [g_g.enc setBuffer:bb offset:(NSUInteger)boff2 atIndex:1];
    [g_g.enc setBuffer:yb offset:(NSUInteger)yoff atIndex:2];
    [g_g.enc setBytes:&ea length:sizeof ea atIndex:3];
    [g_g.enc dispatchThreadgroups:
        MTLSizeMake((vec + 255u) / 256u, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    graph_encode_done();
    return 0;
}

int vtm_metal_graph_silu(float *y, const float *x, int n) {
    if (!g_graph_on || g_graph_failed) return -1;
    if (!y || !x || n <= 0) return graph_fail();
    {
        id<MTLBuffer> xb = nil, yb = nil;
        uint64_t xoff = 0, yoff = 0;
        typedef struct { uint32_t n; } Eltargs;
        Eltargs ea;
        NSUInteger vec;

        if (graph_bind(x, (size_t)n * sizeof(float), &xb, &xoff) != 0
            || graph_bind(y, (size_t)n * sizeof(float), &yb, &yoff) != 0)
            return graph_fail();
        ea.n = (uint32_t)n;
        vec = (NSUInteger)((n + 3) / 4);
        [g_g.enc setComputePipelineState:g_ps_silu];
        [g_g.enc setBuffer:xb offset:(NSUInteger)xoff atIndex:0];
        [g_g.enc setBuffer:yb offset:(NSUInteger)yoff atIndex:1];
        [g_g.enc setBytes:&ea length:sizeof ea atIndex:2];
        [g_g.enc dispatchThreadgroups:
            MTLSizeMake((vec + 255u) / 256u, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        graph_encode_done();
    }
    return 0;
}

int vtm_metal_graph_mul(float *y, const float *a_, const float *b_, int n) {
    return graph_elt(1, y, a_, b_, n);
}

int vtm_metal_graph_add(float *y, const float *a_, const float *b_, int n) {
    return graph_elt(2, y, a_, b_, n);
}

int vtm_metal_graph_kvw(float *kc, const float *k, float *vc, const float *v,
                        int kv_dim, int slot) {
    id<MTLBuffer> kb = nil, vb = nil, kcb = nil, vcb = nil;
    uint64_t koff = 0, voff = 0, kcoff = 0, vcoff = 0;
    typedef struct { uint32_t kv_dim, slot; } Kvwargs;
    Kvwargs a;

    if (!g_graph_on || g_graph_failed) return -1;
    if (!kc || !vc || !k || !v || kv_dim <= 0 || slot < 0) return graph_fail();
    if (graph_bind(k, (size_t)kv_dim * sizeof(float), &kb, &koff) != 0
        || graph_bind(v, (size_t)kv_dim * sizeof(float), &vb, &voff) != 0)
        return graph_fail();
    if (graph_bind(kc, (size_t)(slot + 1) * (size_t)kv_dim * sizeof(float),
                   &kcb, &kcoff) != 0
        || graph_bind(vc, (size_t)(slot + 1) * (size_t)kv_dim * sizeof(float),
                      &vcb, &vcoff) != 0)
        return graph_fail();
    a.kv_dim = (uint32_t)kv_dim;
    a.slot = (uint32_t)slot;
    [g_g.enc setComputePipelineState:g_ps_kvw];
    [g_g.enc setBuffer:kb offset:(NSUInteger)koff atIndex:0];
    [g_g.enc setBuffer:vb offset:(NSUInteger)voff atIndex:1];
    [g_g.enc setBuffer:kcb offset:(NSUInteger)kcoff atIndex:2];
    [g_g.enc setBuffer:vcb offset:(NSUInteger)vcoff atIndex:3];
    [g_g.enc setBytes:&a length:sizeof a atIndex:4];
    [g_g.enc dispatchThreadgroups:
        MTLSizeMake((((NSUInteger)kv_dim + 255u) / 256u), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    graph_encode_done();
    return 0;
}

int vtm_metal_graph_attn(float *out, const float *q, const float *kc,
                         const float *vc, int n_heads, int n_kv_heads,
                         int head_dim, int kv_dim, int win, int n_keys,
                         int slot0) {
    id<MTLBuffer> qb = nil, kb = nil, vb = nil, ob = nil;
    uint64_t qoff = 0, koff = 0, voff = 0, ooff = 0;
    typedef struct {
        uint32_t n_heads, rep, head_dim, kv_dim, win, n_keys, slot0;
        float    scale;
    } Attnargs;
    Attnargs a;

    if (!g_graph_on || g_graph_failed) return -1;
    if (!out || !q || !kc || !vc) return graph_fail();
    if (n_heads <= 0 || n_kv_heads <= 0 || n_heads % n_kv_heads != 0
        || head_dim <= 0 || head_dim > 256 || kv_dim != n_kv_heads * head_dim
        || win <= 0 || n_keys < 0 || slot0 < 0)
        return graph_fail();
    if (graph_bind(q, (size_t)n_heads * (size_t)head_dim * sizeof(float), &qb,
                   &qoff) != 0
        || graph_bind(out, (size_t)n_heads * (size_t)head_dim * sizeof(float),
                      &ob, &ooff) != 0)
        return graph_fail();
    /* Кэш-тензор — [ctx, kv_dim]; регистрируем ровно столько, сколько нужно
       attention'у: win строк (кольцо всё равно не больше win). */
    if (graph_bind(kc, (size_t)win * (size_t)kv_dim * sizeof(float), &kb,
                   &koff) != 0
        || graph_bind(vc, (size_t)win * (size_t)kv_dim * sizeof(float), &vb,
                      &voff) != 0)
        return graph_fail();
    a.n_heads = (uint32_t)n_heads;
    a.rep = (uint32_t)(n_heads / n_kv_heads);
    a.head_dim = (uint32_t)head_dim;
    a.kv_dim = (uint32_t)kv_dim;
    a.win = (uint32_t)win;
    a.n_keys = (uint32_t)n_keys;
    a.slot0 = (uint32_t)slot0;
    a.scale = (float)(1.0 / sqrt((double)head_dim));
    [g_g.enc setComputePipelineState:g_ps_attn];
    [g_g.enc setBuffer:qb offset:(NSUInteger)qoff atIndex:0];
    [g_g.enc setBuffer:kb offset:(NSUInteger)koff atIndex:1];
    [g_g.enc setBuffer:vb offset:(NSUInteger)voff atIndex:2];
    [g_g.enc setBuffer:ob offset:(NSUInteger)ooff atIndex:3];
    [g_g.enc setBytes:&a length:sizeof a atIndex:4];
    [g_g.enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_heads, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    graph_encode_done();
    return 0;
}

int vtm_metal_graph_copy(float *y, const float *x, int n) {
    id<MTLBuffer> xb = nil, yb = nil;
    uint64_t xoff = 0, yoff = 0;
    typedef struct { uint32_t n; } Eltargs;
    Eltargs ea;
    NSUInteger vec;

    if (!g_graph_on || g_graph_failed) return -1;
    if (!y || !x || n <= 0) return graph_fail();
    if (graph_bind(x, (size_t)n * sizeof(float), &xb, &xoff) != 0
        || graph_bind(y, (size_t)n * sizeof(float), &yb, &yoff) != 0)
        return graph_fail();
    ea.n = (uint32_t)n;
    vec = (NSUInteger)((n + 3) / 4);
    [g_g.enc setComputePipelineState:g_ps_copy];
    [g_g.enc setBuffer:xb offset:(NSUInteger)xoff atIndex:0];
    [g_g.enc setBuffer:yb offset:(NSUInteger)yoff atIndex:1];
    [g_g.enc setBytes:&ea length:sizeof ea atIndex:2];
    [g_g.enc dispatchThreadgroups:
        MTLSizeMake((vec + 255u) / 256u, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    graph_encode_done();
    return 0;
}

int vtm_metal_graph_end(void) {
    int rc = 0;
    double gpu;

    if (!__atomic_load_n(&g_graph_on, __ATOMIC_ACQUIRE)) return 0;
    __atomic_store_n(&g_graph_on, 0, __ATOMIC_RELEASE);
    if (g_g.t_encode0 > 0.0) {
        double dt = now_ms() - g_g.t_encode0;
        if (dt > 0.0) g_graph_encode_ns += (uint64_t)(dt * 1e6);
        g_g.t_encode0 = 0.0;
    }
    if (g_g.enc) {
        [g_g.enc endEncoding];
        [g_g.enc release];
        g_g.enc = nil;
    }
    if (force_fail_graph()) {
        metal_set_err("VTM_METAL_FORCE_FAIL_GRAPH: инъекция отказа Metal "
                      "(граф)");
        g_graph_failed = 1;
    }
    if (g_graph_failed || !g_g.cb) {
        [g_g.cb release];
        g_g.cb = nil;
        g_graph_fail += 1;
        g_fail++;
        __atomic_store_n(&g_graph_owner, (uintptr_t)0, __ATOMIC_RELAXED);
        __sync_lock_release(&g_graph_busy);
        return -1;
    }
    @autoreleasepool {
        [g_g.cb commit];
        [g_g.cb waitUntilCompleted];
    }
    gpu = (g_g.cb.GPUEndTime - g_g.cb.GPUStartTime) * 1e9;
    if (gpu > 0.0) g_gpu_ns += (uint64_t)gpu;
    if (g_g.cb.status != MTLCommandBufferStatusCompleted || g_g.cb.error) {
        if (g_fail == 0)
            metal_log("граф токена: %s — дальше CPU-откат",
                      g_g.cb.error
                          ? [[g_g.cb.error localizedDescription] UTF8String]
                          : "status != completed");
        metal_set_err("граф токена не завершился");
        g_fail++;
        rc = -1;
    } else {
        g_calls += g_g.ops;
        g_graph_tokens++;
        g_graph_ops += g_g.ops;
    }
    [g_g.cb release];
    g_g.cb = nil;
    g_g.tmpused = 0;
    g_g.rope_valid = 0;
    __atomic_store_n(&g_graph_owner, (uintptr_t)0, __ATOMIC_RELAXED);
    __sync_lock_release(&g_graph_busy);
    return rc;
}

// --- публичный контракт -----------------------------------------------------
int vtm_metal_available(void) {
    int r;
    pthread_mutex_lock(&g_lock);
    r = metal_device();
    pthread_mutex_unlock(&g_lock);
    return r;
}

int vtm_metal_compiled(void) {
    return 1;
}

int vtm_metal_active(void) {
    // FULL и BATCH: батчевые GEMM prefill работают в обоих режимах.
    return __atomic_load_n(&g_mode, __ATOMIC_RELAXED) != VTM_METAL_OFF;
}

int vtm_metal_gemv_active(void) {
    // Одиночные GEMV — только FULL; в hybrid (BATCH) их считает NEON.
    return __atomic_load_n(&g_mode, __ATOMIC_RELAXED) == VTM_METAL_FULL;
}

int vtm_metal_mode(void) {
    return __atomic_load_n(&g_mode, __ATOMIC_RELAXED);
}

void vtm_metal_set_mode(int mode) {
    if (mode < VTM_METAL_OFF || mode > VTM_METAL_BATCH) mode = VTM_METAL_OFF;
    // GEMV-путь читает флаг из рабочих потоков пула, поэтому запись/чтение
    // атомарны (замок здесь только сериализует выбор бэкенда с инициализацией).
    __atomic_store_n(&g_mode, mode, __ATOMIC_RELAXED);
}

void vtm_metal_set_active(int on) {
    vtm_metal_set_mode(on ? VTM_METAL_FULL : VTM_METAL_OFF);
}

// Оба описателя — копия simd-бэкенда (триты/нормы не трогаем), отличается
// только имя: по нему interp.c/call-сайты понимают, что батч можно отдать GPU.
static const vtm_backend *metal_backend_named(int hybrid) {
    static vtm_backend b[2];
    static int ready[2];

    pthread_mutex_lock(&g_lock);
    if (!ready[hybrid]) {
        b[hybrid] = vtm_backend_simd;
        b[hybrid].name = hybrid ? "hybrid" : "metal";
        ready[hybrid] = 1;
    }
    pthread_mutex_unlock(&g_lock);
    return &b[hybrid];
}

const vtm_backend *vtm_backend_metal(void) {
    return metal_backend_named(0);
}

const vtm_backend *vtm_backend_hybrid(void) {
    return metal_backend_named(1);
}

int vtm_metal_q4k_gemv(float *y, const uint8_t *blocks, const float *x,
                       int rows, int cols) {
    return metal_gemv(0, y, blocks, NULL, x, rows, cols);
}

int vtm_metal_q6k_gemv(float *y, const uint8_t *blocks, const float *x,
                       int rows, int cols) {
    return metal_gemv(1, y, blocks, NULL, x, rows, cols);
}

int vtm_metal_i8_gemv(float *y, const int8_t *q, const float *scales,
                      const float *x, int rows, int cols) {
    return metal_gemv(2, y, q, scales, x, rows, cols);
}

void vtm_metal_stats_get(vtm_metal_stats *out) {
    if (!out) return;
    pthread_mutex_lock(&g_lock);
    out->available = g_avail > 0;
    out->active = vtm_metal_active();
    out->mode = vtm_metal_mode();
    out->init_ms = g_init_ms;
    out->compile_ms = g_compile_ms;
    out->calls = g_calls;
    out->failures = g_fail;
    out->buffers = (uint64_t)g_nbinds;
    out->nocopy_bytes = g_nocopy_bytes;
    out->copy_bytes = g_copy_bytes;
    out->alloc_before = g_alloc_before;
    out->alloc_after = g_alloc_after;
    out->current_allocated = g_dev ? (uint64_t)[g_dev currentAllocatedSize]
                                   : 0;
    out->gemm_calls = g_gemm_calls;
    out->defer_ops = g_defer_ops;
    out->defer_waits = g_defer_waits;
    out->graph_tokens = g_graph_tokens;
    out->graph_ops = g_graph_ops;
    out->graph_fail = g_graph_fail;
    out->graph_encode_ns = g_graph_encode_ns;
    out->wait_ns = g_wait_ns;
    out->gpu_ns = g_gpu_ns;
    snprintf(out->last_error, sizeof out->last_error, "%s", g_err);
    pthread_mutex_unlock(&g_lock);
}
