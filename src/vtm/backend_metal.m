// VTM — Metal-бэкенд, фаза 1 (задача 0192, ADR 0021).
//
// Отдельный исполнитель рядом с scalar/neon: GEMV для kind 6 (Q4_K), kind 7
// (Q6_K) и int8-таблицы словаря (kind 4). Attention, KV, RMSNorm, softmax и
// сэмплинг остаются на CPU — они маленькие и латентные (ADR 0021 §2).
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
    uint32_t reserved;
    uint64_t w_off;
    uint64_t sc_off;   // смещение fp32-масштабов в элементах (int8)
} vtm_metal_args;

#define VTM_METAL_MAX_BIND 1024
#define VTM_METAL_COPY_LIMIT (64u * 1024u * 1024u)

typedef struct {
    const void *ptr;
    size_t bytes;
    id<MTLBuffer> buf;
    uint64_t off;      // байтовое смещение тензора в buf
    uint64_t tick;
    int copied;        // 1 — буфер создан копией (не zero-copy)
} vtm_metal_bind;

static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static int g_avail = -1;            // -1 — не проверяли, 0/1
static int g_init = 0;              // 0 — нет, 1 — готов, -1 — отказ
static int g_active = 0;            // выбран ли metal-бэкенд
static int g_warned_tsan = 0;

static id<MTLDevice> g_dev;
static id<MTLCommandQueue> g_queue;
static id<MTLLibrary> g_lib;
static id<MTLComputePipelineState> g_ps_q4k;
static id<MTLComputePipelineState> g_ps_q6k;
static id<MTLComputePipelineState> g_ps_i8;

static id<MTLBuffer> g_xbuf;
static size_t g_xcap;
static id<MTLBuffer> g_ybuf;
static size_t g_ycap;

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
            "отказов %llu, buferов %d, no-copy %.1f МиБ, копий %.1f МиБ, "
            "currentAllocatedSize %.1f МиБ (до init %.1f МиБ), GPU %.1f мс\n",
            (double)g_init_ms, (double)g_compile_ms,
            (unsigned long long)g_calls, (unsigned long long)g_fail, g_nbinds,
            (double)g_nocopy_bytes / (1024.0 * 1024.0),
            (double)g_copy_bytes / (1024.0 * 1024.0),
            g_dev ? (double)[g_dev currentAllocatedSize] / (1024.0 * 1024.0)
                  : 0.0,
            (double)g_alloc_before / (1024.0 * 1024.0),
            (double)g_gpu_ns / 1e6);
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
    if (!g_ps_q4k || !g_ps_q6k || !g_ps_i8) {
        metal_log("%s — откат на CPU", g_err);
        g_init = -1;
        return -1;
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

// --- общий GEMV -------------------------------------------------------------
static int metal_xbuf(size_t bytes) {
    if (g_xbuf && g_xcap >= bytes) return 0;
    [g_xbuf release];
    g_xbuf = nil;
    g_xcap = 0;
    g_xbuf = [g_dev newBufferWithLength:bytes
                                options:MTLResourceStorageModeShared];
    if (!g_xbuf) return -1;
    g_xcap = bytes;
    return 0;
}

static int metal_ybuf(size_t bytes) {
    if (g_ybuf && g_ycap >= bytes) return 0;
    [g_ybuf release];
    g_ybuf = nil;
    g_ycap = 0;
    g_ybuf = [g_dev newBufferWithLength:bytes
                                options:MTLResourceStorageModeShared];
    if (!g_ybuf) return -1;
    g_ycap = bytes;
    return 0;
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
    a.reserved = 0;
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

    ps = (kind == 0) ? g_ps_q4k : (kind == 1) ? g_ps_q6k : g_ps_i8;
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
    [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)rows + 3u) / 4u, 1, 1)
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
    pthread_mutex_lock(&g_lock);
    for (i = 0; i < g_nbinds; i++) {
        [g_binds[i].buf release];
        g_binds[i].buf = nil;
    }
    g_nbinds = 0;
    pthread_mutex_unlock(&g_lock);
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
    return __atomic_load_n(&g_active, __ATOMIC_RELAXED);
}

void vtm_metal_set_active(int on) {
    // GEMV-путь читает флаг из рабочих потоков пула, поэтому запись/чтение
    // атомарны (замок здесь только сериализует выбор бэкенда с инициализацией).
    __atomic_store_n(&g_active, on ? 1 : 0, __ATOMIC_RELAXED);
}

const vtm_backend *vtm_backend_metal(void) {
    static vtm_backend b;
    static int ready = 0;

    pthread_mutex_lock(&g_lock);
    if (!ready) {
        b = vtm_backend_simd;   // те же ядра для тритов/норм: их не трогаем
        b.name = "metal";
        ready = 1;
    }
    pthread_mutex_unlock(&g_lock);
    return &b;
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
    out->active = __atomic_load_n(&g_active, __ATOMIC_RELAXED);
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
    snprintf(out->last_error, sizeof out->last_error, "%s", g_err);
    pthread_mutex_unlock(&g_lock);
}
