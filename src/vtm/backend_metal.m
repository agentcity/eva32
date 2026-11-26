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
static id<MTLComputePipelineState> g_ps_q4k_gemm[3]; // 8/16/32 токенов
static id<MTLComputePipelineState> g_ps_q6k_gemm[3];
static id<MTLComputePipelineState> g_ps_i8_gemm[3];
// Фаза 3: ядра графа токена (RMSNorm/RoPE/SiLU/MUL/ADD/KVW/attention).
static id<MTLComputePipelineState> g_ps_norm;
static id<MTLComputePipelineState> g_ps_rope;
static id<MTLComputePipelineState> g_ps_silu;
static id<MTLComputePipelineState> g_ps_mul;
static id<MTLComputePipelineState> g_ps_add;
static id<MTLComputePipelineState> g_ps_kvw;
static id<MTLComputePipelineState> g_ps_attn;
static id<MTLComputePipelineState> g_ps_copy;

static id<MTLBuffer> g_xbuf;
static size_t g_xcap;
static id<MTLBuffer> g_ybuf;
static size_t g_ycap;

// Батчевый GEMM (prefill): token-major активации и token-minor результат.
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
    g_ps_copy = metal_pipeline("vtm_copy");
    if (!g_ps_q4k || !g_ps_q6k || !g_ps_i8) {
        metal_log("%s — откат на CPU", g_err);
        g_init = -1;
        return -1;
    }
    // GEMM-ядра фазы 2: отсутствие в старом metallib не ломает GEMV — просто
    // батчевый prefill останется на CPU (вызывающий проверит возврат -1).
    {
        static const char *const names[3][3] = {
            {"vtm_q4k_gemm8", "vtm_q4k_gemm16", "vtm_q4k_gemm32"},
            {"vtm_q6k_gemm8", "vtm_q6k_gemm16", "vtm_q6k_gemm32"},
            {"vtm_i8_gemm8", "vtm_i8_gemm16", "vtm_i8_gemm32"}
        };
        int n, v;
        for (v = 0; v < 3; v++) {
            g_ps_q4k_gemm[v] = metal_pipeline(names[0][v]);
            g_ps_q6k_gemm[v] = metal_pipeline(names[1][v]);
            g_ps_i8_gemm[v] = metal_pipeline(names[2][v]);
        }
        for (v = 0; v < 3; v++)
            if (!g_ps_q4k_gemm[v] || !g_ps_q6k_gemm[v] || !g_ps_i8_gemm[v])
                n = 0;
        (void)n;
        if (!g_ps_q4k_gemm[2] || !g_ps_q6k_gemm[2] || !g_ps_i8_gemm[2])
            metal_log("GEMM-ядра недоступны (%s) — prefill останется на CPU",
                      g_err);
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

// --- фаза 2: батчевый GEMM prefill ------------------------------------------
//
// N токенов пачки — один dispatch на матрицу (на chunk до 32 токенов).
// Активации транспонируются в token-major staging (xt[j*cols + t]): дорожка
// ядра — колонка, поэтому по warp'у читается 32 подряд идущих значения
// одного токена. Результат — token-minor y[r*N + j], как у батчевых ядер.
static int metal_gemm_body(int kind, float *y, const void *w,
                           const float *scales, const float *x,
                           int rows, int cols, int n_tokens) {
    id<MTLBuffer> wb = nil, sb = nil;
    uint64_t woff = 0, scoff = 0;
    vtm_metal_args a;
    id<MTLCommandBuffer> cb = nil;
    id<MTLComputeCommandEncoder> enc = nil;
    id<MTLComputePipelineState> ps;
    size_t wbytes, scbytes, xbytes, ybytes;
    int j0, nt, rc = -1, vi, vn, rpt;

    if (!y || !w || !x || rows <= 0 || cols <= 0 || n_tokens <= 1) return -1;
    // Пачка меньше 8 токенов на GPU проигрывает CPU (замер 0196): ядро всё
    // равно считает 8 (минимум из вариантов), а накладные те же.
    if (n_tokens < 8) return -1;
    if (metal_strict_env()) return -1;
    // Инъекция отказа (VTM_METAL_FORCE_FAIL) для проверки откала гибрида:
    // первый GEMM «теряет» результат, режим выключается — весь остаток
    // пачки/прогона считается CPU-ядрами, как при любой ошибке Metal.
    if (force_fail()) {
        metal_set_err("VTM_METAL_FORCE_FAIL: инъекция отказа Metal (GEMM)");
        g_fail++;
        vtm_metal_set_mode(VTM_METAL_OFF);
        return -1;
    }

    pthread_mutex_lock(&g_lock);
    if (metal_init() != 0) {
        g_fail++;
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    // 0211: вариант GEMM по числу токенов (8/16/32 аккумуляторов на поток).
    // По умолчанию 32: на этой машине он быстрее 16 на длинных пачках (замер
    // 128 токенов: 32,9 против 21,7 ток/с при 32/16 строках на threadgroup).
    // VTM_METAL_GEMM_N=8|16 — A/B и старые metallib без gemm32.
    {
        static int gemm_n = -1;
        if (gemm_n < 0) {
            const char *e = getenv("VTM_METAL_GEMM_N");
            int v = e ? atoi(e) : 0;
            gemm_n = (v == 8 || v == 16 || v == 32) ? v : 32;
        }
        vn = n_tokens <= 8 ? 8 : (n_tokens <= gemm_n ? gemm_n : 32);
        vi = vn == 8 ? 0 : (vn == 16 ? 1 : 2);
    }
    ps = (kind == 0) ? g_ps_q4k_gemm[vi]
       : (kind == 1) ? g_ps_q6k_gemm[vi] : g_ps_i8_gemm[vi];
    if (!ps) {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    // Ядро всегда берёт 8 строк на threadgroup (256 потоков): столько нужно
    // и коллективной загрузке x в threadgroup-память. Если pipeline не
    // держит 256 потоков — на CPU.
    rpt = (int)(ps.maxTotalThreadsPerThreadgroup / 32u);
    // 0211: строк на threadgroup. Было 8 (256 потоков) — x пачки читался
    // заново на каждые 8 строк, и GEMM упирался в эту полосу. Теперь до 16
    // (512 потоков, 16 аккумуляторов на поток): x читается вдвое реже.
    // VTM_METAL_GEMM_RPT=8 возвращает прежний замер 0196 для A/B.
    {
        static int rmax = -1;
        if (rmax < 0) {
            const char *e = getenv("VTM_METAL_GEMM_RPT");
            rmax = (e && atoi(e) > 0) ? atoi(e) : 16;
            if (rmax < 1) rmax = 1;
            // 32 строки (1024 потока) на этой машине дают неверный результат
            // на NT=32 — --check ловит это на 31/32/44; 16 проверено.
            if (rmax > 16) rmax = 16;
        }
        if (rpt > rmax) rpt = rmax;
    }
    // VTM_GEMM_LOAD_HALF делит 128*NT значений на rpt*32 потоков; при
    // rpt > NT цикл загрузки пуст, поэтому потолок — NT.
    if (rpt > vn) rpt = vn;
    if (rpt < 1) rpt = 1;
    if ((ps.maxTotalThreadsPerThreadgroup < (NSUInteger)rpt * 32u)) {
        pthread_mutex_unlock(&g_lock);
        return -1;
    }
    if (kind == 0) {
        if (cols % 256 != 0) { pthread_mutex_unlock(&g_lock); return -1; }
        wbytes = ((size_t)rows * (size_t)cols / 256u) * 144u;
        a.nblocks = (uint32_t)(cols / 256);
    } else if (kind == 1) {
        if (cols % 256 != 0) { pthread_mutex_unlock(&g_lock); return -1; }
        wbytes = ((size_t)rows * (size_t)cols / 256u) * 210u;
        a.nblocks = (uint32_t)(cols / 256);
    } else {
        wbytes = (size_t)rows * (size_t)cols;
        a.nblocks = 0;
    }
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
    a.w_off = woff;

    xbytes = (size_t)cols * (size_t)VTM_METAL_GEMM_MAX_N * sizeof(float);
    ybytes = (size_t)rows * (size_t)VTM_METAL_GEMM_MAX_N * sizeof(float);
    if (metal_reserve(&g_gxbuf, &g_gxcap, xbytes) != 0
        || metal_reserve(&g_gybuf, &g_gycap, ybytes) != 0) {
        g_fail++;
        pthread_mutex_unlock(&g_lock);
        return -1;
    }

    // Токены пачки — по chunk'ам до 32: в ядре столько аккумуляторов-регистров.
    // На chunk — свой командный буфер: результат копируется в y на CPU сразу
    // после завершения, staging не переиспользуется под живым GPU.
    for (j0 = 0; j0 < n_tokens; j0 += vn) {
        int j, r;
        nt = n_tokens - j0;
        if (nt > vn) nt = vn;
        {
            float *xt = (float *)[g_gxbuf contents];
            int t;
            // x[t*n_tokens + j0 + j] -> xt[j*cols + t]; хвост до vn — нули.
            for (t = 0; t < cols; t++) {
                for (j = 0; j < nt; j++)
                    xt[(size_t)j * (size_t)cols + (size_t)t] =
                        x[(size_t)t * (size_t)n_tokens + (size_t)(j0 + j)];
                for (j = nt; j < vn; j++)
                    xt[(size_t)j * (size_t)cols + (size_t)t] = 0.0f;
            }
        }
        a.n_tokens = (uint32_t)rpt;
        cb = [g_queue commandBuffer];
        if (!cb) {
            metal_set_err("commandBuffer вернул nil (gemm)");
            g_fail++;
            goto out;
        }
        enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:ps];
        [enc setBuffer:wb offset:0 atIndex:0];
        [enc setBuffer:g_gxbuf offset:0 atIndex:1];
        [enc setBuffer:g_gybuf offset:0 atIndex:2];
        [enc setBytes:&a length:sizeof a atIndex:3];
        if (kind == 2) [enc setBuffer:sb offset:0 atIndex:4];
        [enc dispatchThreadgroups:
            MTLSizeMake(((NSUInteger)rows + (NSUInteger)rpt - 1u)
                        / (NSUInteger)rpt, 1, 1)
            threadsPerThreadgroup:MTLSizeMake((NSUInteger)rpt * 32u, 1, 1)];
        [enc endEncoding];
        enc = nil;
        [cb commit];
        [cb waitUntilCompleted];
        {
            double gpu = (cb.GPUEndTime - cb.GPUStartTime) * 1e9;
            if (gpu > 0.0) g_gpu_ns += (uint64_t)gpu;
            if (getenv("VTM_METAL_TRACE"))
                metal_log("gemm kind %d %dx%d n=%d: GPU %.3f мс", kind, rows,
                          cols, nt,
                          (cb.GPUEndTime - cb.GPUStartTime) * 1e3);
        }
        if (cb.status != MTLCommandBufferStatusCompleted || cb.error) {
            if (g_fail == 0)
                metal_log("GEMM-буфер: %s — дальше CPU",
                          cb.error
                              ? [[cb.error localizedDescription] UTF8String]
                              : "status != completed");
            metal_set_err("GEMM-буфер не завершился");
            g_fail++;
            // Ошибка командного буфера — не «форма не та», а сбой GPU: чтобы
            // не повторять её на каждой матрице, режим выключается, и
            // остаток прогона идёт чистым CPU-путём (0198, откат).
            vtm_metal_set_mode(VTM_METAL_OFF);
            goto out;
        }
        {
            const float *yt = (const float *)[g_gybuf contents];
            // Строка результата ядра — ровно N = vn столбцов (вариант
            // gemm8/16/32 выбирается по n_tokens, хвост пачки добит нулями
            // ещё в xt). Последний chunk может быть короче vn (nt < vn), но
            // шаг строки в g_gybuf всё равно vn. Читать его как nt нельзя:
            // 0199 — на nt != vn строки сдвигались, и весь батч (а с ним
            // K/V префикса) превращался в мусор.
            for (r = 0; r < rows; r++)
                for (j = 0; j < nt; j++)
                    y[(size_t)r * (size_t)n_tokens + (size_t)(j0 + j)] =
                        yt[(size_t)r * (size_t)vn + (size_t)j];
        }
        g_gemm_calls++;
        g_calls++;
    }
    rc = 0;
out:
    pthread_mutex_unlock(&g_lock);
    return rc;
}

int vtm_metal_qk_gemm(int kind, float *y, const void *w, const float *scales,
                      const float *x, int rows, int cols, int n_tokens) {
    @autoreleasepool {
        return metal_gemm_body(kind, y, w, scales, x, rows, cols, n_tokens);
    }
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
    snprintf(out->last_error, sizeof out->last_error, "%s", g_err);
    pthread_mutex_unlock(&g_lock);
}
