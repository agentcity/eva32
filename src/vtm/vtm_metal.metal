// VTM — Metal-ядра фазы 1 (задача 0192, ADR 0021).
//
// Только GEMV (n == 1): декод. Раскладки весов — те же, что у NEON-ядер
// backend_neon.c (блоки GGML байт-в-байт, задача 0165/ADR 0018):
//
//   Q4_K (kind 6): блок 144 Б на 256 значений: d fp16, dmin fp16,
//                  12 Б 6-битных scale/min, 128 Б нибблов; w = d·sc·q − dmin·m;
//   Q6_K (kind 7): блок 210 Б: ql 128, qh 64, scales 16×int8, d fp16;
//                  w = d·sc·(q − 32);
//   int8 (kind 4): байты q по строкам, масштаб fp32 общий на блок 128
//                  подряд идущих значений сплющенной матрицы; w = q·scale.
//
// Точность: аккумулятор — f32 на дорожку (32 дорожки на строку), свёртка
// simd_sum. Порядок сложений иной, чем у CPU, поэтому побитового паритета нет:
// контракт — argmax + Δlogits ≤ 1e-2 (см. ADR 0021 и задачу 0192).
//
// Все параметры, кроме самих весов, передаются в GemvArgs: базовый буфер
// весов может быть создан на весь mmap-регион, а смещение тензора в нём —
// произвольное (кратно 32, но не странице).

#include <metal_stdlib>
using namespace metal;

// Должно совпадать с vtm_metal_args в backend_metal.m (sizeof == 32).
typedef struct {
    uint rows;
    uint cols;
    uint nblocks;    // cols / 256 для kind 6/7
    uint n_tokens;   // GEMM: токенов пачки; GEMV его игнорирует (1)
    ulong w_off;     // смещение блоков веса в базовом MTLBuffer
    ulong sc_off;    // смещение fp32-масштабов (только int8)
} GemvArgs;

static inline float vtm_f16_to_f32(device const uchar *p) {
    // Побайтово: блоки в контейнере выровнены на 2, но указатель на строку
    // может прийти с произвольным смещением — не полагаемся на выравнивание.
    ushort u = (ushort)p[0] | ((ushort)p[1] << 8);
    return float(as_type<half>(u));
}

// 6-битные (sc, mn) под-блока j из 12 байт scales — формула ggml get_scale_min_k4.
static inline void vtm_q4k_sc_mn(device const uchar *s, uint j,
                                  thread uint &sc, thread uint &mn) {
    if (j < 4u) {
        sc = (uint)(s[j] & 0x3fu);
        mn = (uint)(s[j + 4u] & 0x3fu);
    } else {
        sc = (uint)((s[j + 4u] & 0x0fu) | ((s[j - 4u] >> 6) << 4));
        mn = (uint)((s[j + 4u] >> 4) | ((s[j] >> 6) << 4));
    }
}

// GEMV (декод): 4 строки на threadgroup, simdgroup — строка. Дорожка берёт
// ЧЕТЫРЕ соседних колонки (uchar4 веса + float4 активаций): на одну загрузку
// 4 MAC, и транзакция веса — 128 байт на warp вместо 32. Старая схема
// (байт на дорожку) упиралась в 32-байтовые транзакции: 17–30 ГБ/с на M1 Pro
// (замер 0196).
//
// Q4_K: 32 дорожки закрывают 128 значений (под-блоки 4k..4k+3); блок 256 —
// две итерации. li = lane>>3 — под-блок в половине, l8 = lane&7 — четвёрка.
kernel void vtm_q4k_gemv(device const uchar *w [[buffer(0)]],
                         device const float *x [[buffer(1)]],
                         device float *y [[buffer(2)]],
                         constant GemvArgs &a [[buffer(3)]],
                         uint tg [[threadgroup_position_in_grid]],
                         uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    uint row = tg * 4u + sg;
    uint l8 = lane & 7u;
    uint li = lane >> 3;
    float acc = 0.0f;
    device const uchar *rp;

    if (row >= a.rows) return;
    rp = w + a.w_off + (uint)row * a.nblocks * 144u;

    for (uint b = 0; b < a.nblocks; b++) {
        device const uchar *bp = rp + b * 144u;
        float d = vtm_f16_to_f32(bp);
        float dmin = vtm_f16_to_f32(bp + 2u);
        device const uchar *scp = bp + 4u;
        device const uchar *qs = bp + 16u;

        for (uint k = 0; k < 2u; k++) {
            uint sb = 4u * k + li;
            uint g = sb >> 1;
            uint h = sb & 1u;
            uint scv, mnv;
            uchar4 bytes;
            float w0, w1, w2, w3;
            float4 xv;

            vtm_q4k_sc_mn(scp, sb, scv, mnv);
            bytes = *(device const uchar4 *)(qs + g * 32u + l8 * 4u);
            w0 = d * (float)scv * (float)(h ? (bytes.x >> 4)
                                            : (bytes.x & 0x0fu))
               - dmin * (float)mnv;
            w1 = d * (float)scv * (float)(h ? (bytes.y >> 4)
                                            : (bytes.y & 0x0fu))
               - dmin * (float)mnv;
            w2 = d * (float)scv * (float)(h ? (bytes.z >> 4)
                                            : (bytes.z & 0x0fu))
               - dmin * (float)mnv;
            w3 = d * (float)scv * (float)(h ? (bytes.w >> 4)
                                            : (bytes.w & 0x0fu))
               - dmin * (float)mnv;
            xv = *(device const float4 *)(x + b * 256u + sb * 32u + l8 * 4u);
            acc += w0 * xv.x + w1 * xv.y + w2 * xv.z + w3 * xv.w;
        }
    }

    {
        float total = simd_sum(acc);
        if (lane == 0u) y[row] = total;
    }
}

// Q6_K: под-блок 16 значений, дорожка — четвёрка (l4=lane&3), liq=lane>>2
// (0..7). Половина блока — 8 под-блоков = 128 значений, блок — две итерации.
kernel void vtm_q6k_gemv(device const uchar *w [[buffer(0)]],
                         device const float *x [[buffer(1)]],
                         device float *y [[buffer(2)]],
                         constant GemvArgs &a [[buffer(3)]],
                         uint tg [[threadgroup_position_in_grid]],
                         uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    uint row = tg * 4u + sg;
    uint l4 = lane & 3u;
    uint liq = lane >> 2;
    float acc = 0.0f;
    device const uchar *rp;

    if (row >= a.rows) return;
    rp = w + a.w_off + (uint)row * a.nblocks * 210u;

    for (uint b = 0; b < a.nblocks; b++) {
        device const uchar *bp = rp + b * 210u;
        float d = vtm_f16_to_f32(bp + 208u);
        device const uchar *ql = bp;
        device const uchar *qh = bp + 128u;
        device const char *sc = (device const char *)(bp + 192u);

        for (uint k = 0; k < 2u; k++) {
            uint sb = 8u * k + liq;
            uint jj = l4 * 4u;
            uint chunk = sb >> 3;
            uint in = sb & 7u;
            uint quad = in >> 1;
            uint hs = in & 1u;
            uchar4 qlb = *(device const uchar4 *)(ql + chunk * 64u
                          + (quad & 1u) * 32u + hs * 16u + jj);
            uchar4 qhb = *(device const uchar4 *)(qh + chunk * 32u
                          + hs * 16u + jj);
            float s = d * (float)sc[sb];
            float w0 = s * (float)((int)((quad >= 2u ? (qlb.x >> 4)
                                                   : (qlb.x & 0x0fu))
                            | (((qhb.x >> (quad * 2u)) & 3u) << 4)) - 32);
            float w1 = s * (float)((int)((quad >= 2u ? (qlb.y >> 4)
                                                   : (qlb.y & 0x0fu))
                            | (((qhb.y >> (quad * 2u)) & 3u) << 4)) - 32);
            float w2 = s * (float)((int)((quad >= 2u ? (qlb.z >> 4)
                                                   : (qlb.z & 0x0fu))
                            | (((qhb.z >> (quad * 2u)) & 3u) << 4)) - 32);
            float w3 = s * (float)((int)((quad >= 2u ? (qlb.w >> 4)
                                                   : (qlb.w & 0x0fu))
                            | (((qhb.w >> (quad * 2u)) & 3u) << 4)) - 32);
            float4 xv = *(device const float4 *)(x + b * 256u + sb * 16u + jj);
            acc += w0 * xv.x + w1 * xv.y + w2 * xv.z + w3 * xv.w;
        }
    }

    {
        float total = simd_sum(acc);
        if (lane == 0u) y[row] = total;
    }
}

// int8-таблица словаря (kind 4): дорожка — 4 колонки, масштаб общий на блок
// 128 (четвёрка границу не пересекает: col % 4 == 0).
kernel void vtm_i8_gemv(device const char *q [[buffer(0)]],
                        device const float *x [[buffer(1)]],
                        device float *y [[buffer(2)]],
                        constant GemvArgs &a [[buffer(3)]],
                        device const float *scales [[buffer(4)]],
                        uint tg [[threadgroup_position_in_grid]],
                        uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    uint row = tg * 4u + sg;
    uint base;
    float acc = 0.0f;

    if (row >= a.rows) return;
    base = row * a.cols;
    for (uint col = lane * 4u; col < a.cols; col += 128u) {
        uchar4 qb = *(device const uchar4 *)(q + a.w_off + base + col);
        float sc = scales[a.sc_off + ((base + col) >> 7)];
        float4 xv = *(device const float4 *)(x + col);
        acc += sc * ((float)(char)qb.x * xv.x + (float)(char)qb.y * xv.y
                   + (float)(char)qb.z * xv.z + (float)(char)qb.w * xv.w);
    }

    {
        float total = simd_sum(acc);
        if (lane == 0u) y[row] = total;
    }
}

// ===========================================================================
// Батчевый GEMM (задача 0196, фаза 2): N токенов против одной матрицы.
//
// Вход — token-major: xt[j * cols + t] (хост транспонирует пачку в staging и
// добивает нулями до N). Выход — token-minor, как у батчевых ядер interp.c:
// y[r * N + j].
//
// Схема: threadgroup = 8 строк (8 simdgroups), дорожка берёт 4 подряд идущих
// колонки. Активации половины блока (128 колонок × N токенов) сначала
// коллективно грузятся в threadgroup-память: 256 потоков делают 4 независимые
// float4-загрузки — это даёт нужную параллельность памяти. Без этого каждая
// дорожка тянула float4 из глобальной памяти последовательно, и ядро
// упиралось в задержку (замер 0196: x-трафик — 70 ГБ/с и 140 GMAC/s против
// 1,7 TMAC/s, когда x целиком в кэше).
//
// N — параметр сборки (8/16/32): цикл токенов развёрнут, аккумуляторы в
// регистрах.
// ===========================================================================

#define VTM_GEMM_TG 256u   /* потоков: 8 simdgroup по 32 */

// Общая загрузка половины блока: xs[j*128 + i] = xt[j*cols + colbase + i].
// tid — thread_index_in_threadgroup, NT — токенов, M — загрузок на поток
// (128*NT/1024 = NT/8).
#define VTM_GEMM_LOAD_HALF(NT, COLBASE)                                      \
    do {                                                                     \
        uint m;                                                              \
        _Pragma("unroll")                                                    \
        for (m = 0; m < (NT) / rpt; m++) {                                   \
            uint flat = (tid + rpt * 32u * m) * 4u;                          \
            uint lj = flat >> 7;                                             \
            uint li = flat & 127u;                                           \
            *(threadgroup float4 *)(xs + lj * 128u + li) =                   \
                *(device const float4 *)(xt + (ulong)lj * a.cols            \
                                         + (COLBASE) + li);                  \
        }                                                                    \
        threadgroup_barrier(mem_flags::mem_threadgroup);                     \
    } while (0)

#define VTM_GEMM_KERNEL(NAME, NT, DECODE_RP, DECODE_HALF)              \
kernel void NAME(device const uchar *w [[buffer(0)]],                        \
                 device const float *xt [[buffer(1)]],                       \
                 device float *y [[buffer(2)]],                              \
                 constant GemvArgs &a [[buffer(3)]],                         \
                 uint tg [[threadgroup_position_in_grid]],                   \
                 uint sg [[simdgroup_index_in_threadgroup]],                 \
                 uint lane [[thread_index_in_simdgroup]],                    \
                 uint tid [[thread_index_in_threadgroup]]) {                 \
    const uint N = NT;                                                       \
    threadgroup float xs[128u * NT];                                         \
    uint rpt = a.n_tokens;                                                   \
    uint row = tg * rpt + sg;                                                \
    float acc[NT];                                                           \
    device const uchar *rp;                                                  \
    uint active = row < a.rows;                                              \
    for (uint z = 0; z < NT; z++) acc[z] = 0.0f;                             \
    DECODE_RP;                                                               \
    for (uint b = 0; b < a.nblocks; b++) {                                   \
        VTM_GEMM_LOAD_HALF(NT, b * 256u);                                    \
        if (active) { DECODE_HALF(0); }                                      \
        threadgroup_barrier(mem_flags::mem_threadgroup);                     \
        VTM_GEMM_LOAD_HALF(NT, b * 256u + 128u);                             \
        if (active) { DECODE_HALF(1); }                                      \
        threadgroup_barrier(mem_flags::mem_threadgroup);                     \
    }                                                                        \
    if (active) {                                                            \
        for (uint j = 0; j < NT; j++) {                                      \
            float total = simd_sum(acc[j]);                                  \
            if (lane == 0u) y[(uint)row * N + j] = total;                    \
        }                                                                    \
    }                                                                        \
}

// Q4_K: половина k — под-блоки 4k..4k+3, дорожка li=lane>>3, четвёрка l8.
#define VTM_Q4K_RP                                                           \
    rp = w + a.w_off + (uint)row * a.nblocks * 144u;

#define VTM_Q4K_HALF(K)                                                      \
    do {                                                                     \
        device const uchar *bp = rp + b * 144u;                              \
        float d = vtm_f16_to_f32(bp);                                        \
        float dmin = vtm_f16_to_f32(bp + 2u);                                \
        device const uchar *scp = bp + 4u;                                   \
        device const uchar *qs = bp + 16u;                                   \
        uint l8 = lane & 7u;                                                 \
        uint li = lane >> 3;                                                 \
        uint sb = 4u * (K) + li;                                             \
        uint g = sb >> 1;                                                    \
        uint h = sb & 1u;                                                    \
        uint scv, mnv;                                                       \
        vtm_q4k_sc_mn(scp, sb, scv, mnv);                                    \
        uchar4 bytes = *(device const uchar4 *)(qs + g * 32u + l8 * 4u);    \
        float w0 = d * (float)scv * (float)(h ? (bytes.x >> 4)              \
                                                : (bytes.x & 0x0fu))         \
                 - dmin * (float)mnv;                                        \
        float w1 = d * (float)scv * (float)(h ? (bytes.y >> 4)              \
                                                : (bytes.y & 0x0fu))         \
                 - dmin * (float)mnv;                                        \
        float w2 = d * (float)scv * (float)(h ? (bytes.z >> 4)              \
                                                : (bytes.z & 0x0fu))         \
                 - dmin * (float)mnv;                                        \
        float w3 = d * (float)scv * (float)(h ? (bytes.w >> 4)              \
                                                : (bytes.w & 0x0fu))         \
                 - dmin * (float)mnv;                                        \
        for (uint j = 0; j < N; j++) {                                       \
            float4 xv = *(threadgroup const float4 *)(xs + j * 128u         \
                                                     + li * 32u + l8 * 4u); \
            acc[j] += w0 * xv.x + w1 * xv.y + w2 * xv.z + w3 * xv.w;        \
        }                                                                    \
    } while (0)

// Q6_K: половина k — под-блоки 8k..8k+7, liq=lane>>2, четвёрка l4.
#define VTM_Q6K_RP                                                           \
    rp = w + a.w_off + (uint)row * a.nblocks * 210u;

#define VTM_Q6K_HALF(K)                                                      \
    do {                                                                     \
        device const uchar *bp = rp + b * 210u;                              \
        float d = vtm_f16_to_f32(bp + 208u);                                 \
        device const uchar *ql = bp;                                         \
        device const uchar *qh = bp + 128u;                                  \
        device const char *sc = (device const char *)(bp + 192u);            \
        uint l4 = lane & 3u;                                                 \
        uint liq = lane >> 2;                                                \
        uint sb = 8u * (K) + liq;                                            \
        uint jj = l4 * 4u;                                                   \
        uint chunk = sb >> 3;                                                \
        uint in = sb & 7u;                                                   \
        uint quad = in >> 1;                                                 \
        uint hs = in & 1u;                                                   \
        uchar4 qlb = *(device const uchar4 *)(ql + chunk * 64u              \
                      + (quad & 1u) * 32u + hs * 16u + jj);                 \
        uchar4 qhb = *(device const uchar4 *)(qh + chunk * 32u + hs * 16u   \
                      + jj);                                                 \
        float s = d * (float)sc[sb];                                         \
        float w0 = s * (float)((int)((quad >= 2u ? (qlb.x >> 4)             \
                                                  : (qlb.x & 0x0fu))        \
                        | (((qhb.x >> (quad * 2u)) & 3u) << 4)) - 32);       \
        float w1 = s * (float)((int)((quad >= 2u ? (qlb.y >> 4)             \
                                                  : (qlb.y & 0x0fu))        \
                        | (((qhb.y >> (quad * 2u)) & 3u) << 4)) - 32);       \
        float w2 = s * (float)((int)((quad >= 2u ? (qlb.z >> 4)             \
                                                  : (qlb.z & 0x0fu))        \
                        | (((qhb.z >> (quad * 2u)) & 3u) << 4)) - 32);       \
        float w3 = s * (float)((int)((quad >= 2u ? (qlb.w >> 4)             \
                                                  : (qlb.w & 0x0fu))        \
                        | (((qhb.w >> (quad * 2u)) & 3u) << 4)) - 32);       \
        for (uint j = 0; j < N; j++) {                                       \
            float4 xv = *(threadgroup const float4 *)(xs + j * 128u         \
                                                     + liq * 16u + l4 * 4u);\
            acc[j] += w0 * xv.x + w1 * xv.y + w2 * xv.z + w3 * xv.w;        \
        }                                                                    \
    } while (0)

VTM_GEMM_KERNEL(vtm_q4k_gemm8, 8u, VTM_Q4K_RP, VTM_Q4K_HALF)
VTM_GEMM_KERNEL(vtm_q4k_gemm16, 16u, VTM_Q4K_RP, VTM_Q4K_HALF)
VTM_GEMM_KERNEL(vtm_q4k_gemm32, 32u, VTM_Q4K_RP, VTM_Q4K_HALF)
VTM_GEMM_KERNEL(vtm_q6k_gemm8, 8u, VTM_Q6K_RP, VTM_Q6K_HALF)
VTM_GEMM_KERNEL(vtm_q6k_gemm16, 16u, VTM_Q6K_RP, VTM_Q6K_HALF)
VTM_GEMM_KERNEL(vtm_q6k_gemm32, 32u, VTM_Q6K_RP, VTM_Q6K_HALF)

// int8-таблица словаря (kind 4): дорожка — 4 подряд идущих колонки, масштаб
// общий на блок 128 (четвёрка не пересекает границу).
#define VTM_I8_GEMM_KERNEL(NAME, NT)                                         \
kernel void NAME(device const char *q [[buffer(0)]],                        \
                 device const float *xt [[buffer(1)]],                       \
                 device float *y [[buffer(2)]],                              \
                 constant GemvArgs &a [[buffer(3)]],                         \
                 device const float *scales [[buffer(4)]],                   \
                 uint tg [[threadgroup_position_in_grid]],                   \
                 uint sg [[simdgroup_index_in_threadgroup]],                 \
                 uint lane [[thread_index_in_simdgroup]],                    \
                 uint tid [[thread_index_in_threadgroup]]) {                 \
    const uint N = NT;                                                       \
    threadgroup float xs[128u * NT];                                         \
    uint rpt = a.n_tokens;                                                   \
    uint row = tg * rpt + sg;                                                \
    uint active = row < a.rows;                                              \
    uint base;                                                               \
    float acc[NT];                                                           \
    for (uint z = 0; z < NT; z++) acc[z] = 0.0f;                             \
    base = row * a.cols;                                                     \
    for (uint blk = 0; blk < a.cols; blk += 128u) {                          \
        VTM_GEMM_LOAD_HALF(NT, blk);                                         \
        if (active) {                                                        \
            uint col = blk + lane * 4u;                                      \
            uchar4 qb = *(device const uchar4 *)(q + a.w_off + base + col);  \
            float sc = scales[a.sc_off + ((base + col) >> 7)];               \
            float w0 = sc * (float)(char)qb.x;                               \
            float w1 = sc * (float)(char)qb.y;                               \
            float w2 = sc * (float)(char)qb.z;                               \
            float w3 = sc * (float)(char)qb.w;                               \
            for (uint j = 0; j < N; j++) {                                   \
                float4 xv = *(threadgroup const float4 *)(xs + j * 128u     \
                                                         + lane * 4u);       \
                acc[j] += w0 * xv.x + w1 * xv.y + w2 * xv.z + w3 * xv.w;    \
            }                                                                \
        }                                                                    \
        threadgroup_barrier(mem_flags::mem_threadgroup);                     \
    }                                                                        \
    if (active) {                                                            \
        for (uint j = 0; j < N; j++) {                                       \
            float total = simd_sum(acc[j]);                                  \
            if (lane == 0u) y[(uint)row * N + j] = total;                    \
        }                                                                    \
    }                                                                        \
}

VTM_I8_GEMM_KERNEL(vtm_i8_gemm8, 8u)
VTM_I8_GEMM_KERNEL(vtm_i8_gemm16, 16u)
VTM_I8_GEMM_KERNEL(vtm_i8_gemm32, 32u)
