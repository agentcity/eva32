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

// ===========================================================================
// GEMV декода с несколькими строками на дорожку-группу (фаза 3, 0204).
//
// Диагноз: однострочные ядра выше читают 4 байта веса на дорожку за итерацию
// и имеют ОДНУ независимую цепочку загрузок на warp. На M1 Pro это давало
// 21–28 ГБ/с на Q4_K/Q6_K (замер --qk-proj: 11008x2048 Q4_K — 0.79 мс),
// тогда как int8-GEMV с 128-байтовой транзакцией на warp и длинным циклом
// держит 145 ГБ/с. Причина — не полоса, а задержка: у warp'а в полёте одна
// загрузка, следующая итерация ждёт её.
//
// Здесь каждый warp считает R=4 строки сразу: активации загружаются один раз
// на итерацию и переиспользуются R раз, а четыре загрузки весов независимы.
// Аккумуляторов R (в регистрах), цепочек тоже R — ILP вместо одной.
// Строки последнего threadgroup клампятся к последней существующей: лишние
// считаются вхолостую (не более 4R-1 строк), но цикл остаётся без ветвлений.
// ===========================================================================

#define VTM_GEMV_R 4u

kernel void vtm_q4k_gemv_r4(device const uchar *w [[buffer(0)]],
                            device const float *x [[buffer(1)]],
                            device float *y [[buffer(2)]],
                            constant GemvArgs &a [[buffer(3)]],
                            uint tg [[threadgroup_position_in_grid]],
                            uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
    uint row0 = (tg * 4u + sg) * VTM_GEMV_R;
    uint l8 = lane & 7u;
    uint li = lane >> 3;
    uint last = a.rows - 1u;
    float acc[VTM_GEMV_R];
    device const uchar *rp[VTM_GEMV_R];

    _Pragma("unroll")
    for (uint r = 0; r < VTM_GEMV_R; r++) {
        uint row = row0 + r;
        if (row > last) row = last;
        acc[r] = 0.0f;
        rp[r] = w + a.w_off + (ulong)row * a.nblocks * 144u;
    }

    for (uint b = 0; b < a.nblocks; b++) {
        _Pragma("unroll")
        for (uint k = 0; k < 2u; k++) {
            uint sb = 4u * k + li;
            uint g = sb >> 1u;
            uint h = sb & 1u;
            uint qoff = 16u + g * 32u + l8 * 4u;
            float4 xv = *(device const float4 *)(x + b * 256u
                                                 + sb * 32u + l8 * 4u);
            _Pragma("unroll")
            for (uint r = 0; r < VTM_GEMV_R; r++) {
                device const uchar *bp = rp[r] + b * 144u;
                float d = vtm_f16_to_f32(bp);
                float dmin = vtm_f16_to_f32(bp + 2u);
                uint scv, mnv;
                uchar4 bytes;
                float A, B;

                vtm_q4k_sc_mn(bp + 4u, sb, scv, mnv);
                bytes = *(device const uchar4 *)(bp + qoff);
                A = d * (float)scv;
                B = dmin * (float)mnv;
                {
                    float w0 = A * (float)(h ? (bytes.x >> 4)
                                             : (bytes.x & 0x0fu)) - B;
                    float w1 = A * (float)(h ? (bytes.y >> 4)
                                             : (bytes.y & 0x0fu)) - B;
                    float w2 = A * (float)(h ? (bytes.z >> 4)
                                             : (bytes.z & 0x0fu)) - B;
                    float w3 = A * (float)(h ? (bytes.w >> 4)
                                             : (bytes.w & 0x0fu)) - B;
                    acc[r] += w0 * xv.x + w1 * xv.y + w2 * xv.z + w3 * xv.w;
                }
            }
        }
    }

    _Pragma("unroll")
    for (uint r = 0; r < VTM_GEMV_R; r++) {
        float total = simd_sum(acc[r]);
        if (lane == 0u && row0 + r < a.rows) y[row0 + r] = total;
    }
}

kernel void vtm_q6k_gemv_r4(device const uchar *w [[buffer(0)]],
                            device const float *x [[buffer(1)]],
                            device float *y [[buffer(2)]],
                            constant GemvArgs &a [[buffer(3)]],
                            uint tg [[threadgroup_position_in_grid]],
                            uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
    uint row0 = (tg * 4u + sg) * VTM_GEMV_R;
    uint l4 = lane & 3u;
    uint liq = lane >> 2;
    uint last = a.rows - 1u;
    float acc[VTM_GEMV_R];
    device const uchar *rp[VTM_GEMV_R];

    _Pragma("unroll")
    for (uint r = 0; r < VTM_GEMV_R; r++) {
        uint row = row0 + r;
        if (row > last) row = last;
        acc[r] = 0.0f;
        rp[r] = w + a.w_off + (ulong)row * a.nblocks * 210u;
    }

    for (uint b = 0; b < a.nblocks; b++) {
        _Pragma("unroll")
        for (uint k = 0; k < 2u; k++) {
            uint sb = 8u * k + liq;
            uint jj = l4 * 4u;
            uint chunk = sb >> 3;
            uint in = sb & 7u;
            uint quad = in >> 1;
            uint hs = in & 1u;
            uint qloff = chunk * 64u + (quad & 1u) * 32u + hs * 16u + jj;
            uint qhoff = 128u + chunk * 32u + hs * 16u + jj;
            float4 xv = *(device const float4 *)(x + b * 256u + sb * 16u + jj);

            _Pragma("unroll")
            for (uint r = 0; r < VTM_GEMV_R; r++) {
                device const uchar *bp = rp[r] + b * 210u;
                device const char *sc = (device const char *)(bp + 192u);
                float d = vtm_f16_to_f32(bp + 208u);
                uchar4 qlb = *(device const uchar4 *)(bp + qloff);
                uchar4 qhb = *(device const uchar4 *)(bp + qhoff);
                float s = d * (float)sc[sb];
                float w0 = s * (float)((int)((quad >= 2u ? (qlb.x >> 4)
                                                          : (qlb.x & 0x0fu))
                                 | (((qhb.x >> (quad * 2u)) & 3u) << 4))
                                       - 32);
                float w1 = s * (float)((int)((quad >= 2u ? (qlb.y >> 4)
                                                          : (qlb.y & 0x0fu))
                                 | (((qhb.y >> (quad * 2u)) & 3u) << 4))
                                       - 32);
                float w2 = s * (float)((int)((quad >= 2u ? (qlb.z >> 4)
                                                          : (qlb.z & 0x0fu))
                                 | (((qhb.z >> (quad * 2u)) & 3u) << 4))
                                       - 32);
                float w3 = s * (float)((int)((quad >= 2u ? (qlb.w >> 4)
                                                          : (qlb.w & 0x0fu))
                                 | (((qhb.w >> (quad * 2u)) & 3u) << 4))
                                       - 32);
                acc[r] += w0 * xv.x + w1 * xv.y + w2 * xv.z + w3 * xv.w;
            }
        }
    }

    _Pragma("unroll")
    for (uint r = 0; r < VTM_GEMV_R; r++) {
        float total = simd_sum(acc[r]);
        if (lane == 0u && row0 + r < a.rows) y[row0 + r] = total;
    }
}

// ===========================================================================
// Фаза 3 (0204): RMSNorm/RoPE/SiLU/MUL/ADD/KVW/attention декода на GPU.
//
// Смысл — не «перенести всё», а убрать ожидания CPU между матрицами слоя:
// пока RMSN/ROPE/attention/SiLU считались на CPU, интерпретатор обязан был
// ждать GPU на каждой из них (4 ожидания на слой, 145 командных буферов на
// токен). Здесь весь слой — один MTLCommandBuffer, CPU ждёт один раз за токен
// (перед сэмплингом). Арифметика — f32; эталоны CPU считают в double, поэтому
// побитового паритета нет, контракт — argmax + Δlogits ≤ 1e-2 (ADR 0021).
// ===========================================================================

typedef struct {
    uint  dim;      // длина строки (RMSN: весь тензор; RMSH: последняя ось)
    uint  rows;     // число строк
    float eps;
} NormArgs;

// Один threadgroup на строку, 256 потоков. Сумма квадратов — f32 с
// simd_sum по варпам и добором через threadgroup-память: 2048 значений дают
// относительную ошибку ~1e-6, это на порядки ниже допуска по логитам.
kernel void vtm_rmsnorm(device const float *x [[buffer(0)]],
                        device const float *w [[buffer(1)]],
                        device float *y [[buffer(2)]],
                        constant NormArgs &a [[buffer(3)]],
                        uint tg [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint wid [[simdgroup_index_in_threadgroup]]) {
    const uint NT = 256u;
    threadgroup float red[8];
    device const float *xr = x + (ulong)tg * a.dim;
    device float *yr = y + (ulong)tg * a.dim;
    float s = 0.0f;
    float inv;
    uint i;

    for (i = tid; i < a.dim; i += NT) s = fma(xr[i], xr[i], s);
    s = simd_sum(s);
    if (lane == 0u) red[wid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float t = red[0] + red[1] + red[2] + red[3]
                + red[4] + red[5] + red[6] + red[7];
        red[0] = t / (float)a.dim + a.eps;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    inv = 1.0f / sqrt(red[0]);
    for (i = tid; i < a.dim; i += NT) yr[i] = xr[i] * inv * w[i];
}

typedef struct {
    uint heads;
    uint nhalf;       // head_dim / 2 (half — зарезервированное имя MSL)
    uint head_dim;
    uint interleaved;
} RopeArgs;

// Одна пара (p, p+half) на поток. cos/sin считает CPU в double (как эталон) и
// кладёт в staging: в MSL нет fp64, а считать cos большого угла в f32 значило
// бы терять точность на длинном контексте.
kernel void vtm_rope(device const float *x [[buffer(0)]],
                     device float *y [[buffer(1)]],
                     device const float *cs [[buffer(2)]],
                     constant RopeArgs &a [[buffer(3)]],
                     uint gid [[thread_position_in_grid]]) {
    uint idx = gid;
    uint total = a.heads * a.nhalf;

    if (idx >= total) return;
    {
        uint h = idx / a.nhalf;
        uint p = idx - h * a.nhalf;
        uint i0 = h * a.head_dim + (a.interleaved ? 2u * p : p);
        uint i1 = h * a.head_dim
                + (a.interleaved ? 2u * p + 1u : p + a.nhalf);
        float c = cs[p];
        float s = cs[a.nhalf + p];
        float x0 = x[i0];
        float x1 = x[i1];
        y[i0] = x0 * c - x1 * s;
        y[i1] = x1 * c + x0 * s;
    }
}

typedef struct {
    uint n;
} EltArgs;

kernel void vtm_silu(device const float *x [[buffer(0)]],
                     device float *y [[buffer(1)]],
                     constant EltArgs &a [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    uint i = gid * 4u;
    if (i + 4u <= a.n) {
        float4 v = *(device const float4 *)(x + i);
        float4 r;
        r.x = v.x / (1.0f + exp(-v.x));
        r.y = v.y / (1.0f + exp(-v.y));
        r.z = v.z / (1.0f + exp(-v.z));
        r.w = v.w / (1.0f + exp(-v.w));
        *(device float4 *)(y + i) = r;
    } else {
        for (; i < a.n; i++) y[i] = x[i] / (1.0f + exp(-x[i]));
    }
}

kernel void vtm_mul(device const float *a_ [[buffer(0)]],
                    device const float *b_ [[buffer(1)]],
                    device float *y [[buffer(2)]],
                    constant EltArgs &a [[buffer(3)]],
                    uint gid [[thread_position_in_grid]]) {
    uint i = gid * 4u;
    if (i + 4u <= a.n) {
        float4 u = *(device const float4 *)(a_ + i);
        float4 v = *(device const float4 *)(b_ + i);
        *(device float4 *)(y + i) = u * v;
    } else {
        for (; i < a.n; i++) y[i] = a_[i] * b_[i];
    }
}

kernel void vtm_add(device const float *a_ [[buffer(0)]],
                    device const float *b_ [[buffer(1)]],
                    device float *y [[buffer(2)]],
                    constant EltArgs &a [[buffer(3)]],
                    uint gid [[thread_position_in_grid]]) {
    uint i = gid * 4u;
    if (i + 4u <= a.n) {
        float4 u = *(device const float4 *)(a_ + i);
        float4 v = *(device const float4 *)(b_ + i);
        *(device float4 *)(y + i) = u + v;
    } else {
        for (; i < a.n; i++) y[i] = a_[i] + b_[i];
    }
}

typedef struct {
    uint kv_dim;
    uint slot;
} KvwArgs;

// KVW: K/V новой позиции в слот кольца. Кэш — тот же f32-тензор арены, что у
// CPU-пути; zero-copy буфер смотрит в него же, поэтому после завершения
// командного буфера запись видна и CPU (следующий prefill/сэмплинг).
kernel void vtm_kvw(device const float *k [[buffer(0)]],
                    device const float *v [[buffer(1)]],
                    device float *kc [[buffer(2)]],
                    device float *vc [[buffer(3)]],
                    constant KvwArgs &a [[buffer(4)]],
                    uint gid [[thread_position_in_grid]]) {
    ulong base = (ulong)a.slot * a.kv_dim;
    if (gid < a.kv_dim) {
        kc[base + gid] = k[gid];
        vc[base + gid] = v[gid];
    }
}

typedef struct {
    uint  n_heads;
    uint  rep;        // n_heads / n_kv_heads
    uint  head_dim;
    uint  kv_dim;
    uint  win;
    uint  n_keys;     // видимых ключей (min(key_count, win))
    uint  slot0;      // слот самого старого видимого ключа
    float scale;
} AttnArgs;

// Attention декода (q_len == 1): один warp (threadgroup) на голову.
// Каждая дорожка держит head_dim/32 элементов q и аккумулятора в регистрах;
// Q·K и softmax — online, без threadgroup-барьеров: simd_sum отдаёт сумму
// всем дорожкам, поэтому max/сумма — это uniform-значения по warp'у.
// K/V читаются из кольцевого кэша [ctx, kv_dim] со слотом (slot0+n) % win.
kernel void vtm_attn_decode(device const float *q [[buffer(0)]],
                            device const float *kc [[buffer(1)]],
                            device const float *vc [[buffer(2)]],
                            device float *out [[buffer(3)]],
                            constant AttnArgs &a [[buffer(4)]],
                            uint tg [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]]) {
    uint h = tg;
    uint kvh = h / a.rep;
    device const float *qh = q + (ulong)h * a.head_dim;
    device const float *kb = kc + (ulong)kvh * a.head_dim;
    device const float *vb = vc + (ulong)kvh * a.head_dim;
    float qv[8];
    float acc[8];
    float m = -INFINITY;
    float l = 0.0f;
    uint slot = a.slot0;
    uint i, n;

    _Pragma("unroll")
    for (i = 0; i < 8u; i++) {
        uint d = lane + i * 32u;
        qv[i] = (d < a.head_dim) ? qh[d] : 0.0f;
        acc[i] = 0.0f;
    }

    for (n = 0; n < a.n_keys; n++) {
        device const float *krow = kb + (ulong)slot * a.kv_dim;
        device const float *vrow = vb + (ulong)slot * a.kv_dim;
        float dot = 0.0f;
        float s;
        float p;
        float corr;

        _Pragma("unroll")
        for (i = 0; i < 8u; i++) {
            uint d = lane + i * 32u;
            if (d < a.head_dim) dot = fma(qv[i], krow[d], dot);
        }
        dot = simd_sum(dot) * a.scale;

        // Online-softmax: m — текущий максимум, l — сумма exp, acc — числитель.
        s = (dot > m) ? dot : m;
        corr = exp(m - s);            // m == -INFINITY -> 0
        p = exp(dot - s);
        l = l * corr + p;
        _Pragma("unroll")
        for (i = 0; i < 8u; i++) {
            uint d = lane + i * 32u;
            if (d < a.head_dim)
                acc[i] = fma(p, vrow[d], acc[i] * corr);
        }
        m = s;
        if (++slot == a.win) slot = 0u;
    }

    {
        float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
        device float *orow = out + (ulong)h * a.head_dim;

        _Pragma("unroll")
        for (i = 0; i < 8u; i++) {
            uint d = lane + i * 32u;
            if (d < a.head_dim) orow[d] = acc[i] * inv;
        }
    }
}

// CPY: f32 -> f32 той же длины (финальная норма перед lm_head в графе токена).
kernel void vtm_copy(device const float *x [[buffer(0)]],
                     device float *y [[buffer(1)]],
                     constant EltArgs &a [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    uint i = gid * 4u;
    if (i + 4u <= a.n) {
        *(device float4 *)(y + i) = *(device const float4 *)(x + i);
    } else {
        for (; i < a.n; i++) y[i] = x[i];
    }
}

// --- фаза 3 (0204): GEMV с четырьмя строками на warp, без массивов ---------
//
// В однострочном ядре выше на одну итерацию приходится 64 Б веса и 512 Б
// активаций: каждый warp читает x целиком для СВОЕЙ строки. Для gate/up
// (11008 строк) это 90 МБ чтений x против 12,7 МБ веса — узкое место не
// полоса весов, а повторные чтения x. Здесь warp считает 4 строки: x грузится
// один раз на итерацию и переиспользуется, четыре загрузки веса независимы
// (MLP), аккумуляторов четыре.
//
// Прошлая попытка (r4 в начале раздела) держала указатели в массиве
// `device const uchar *rp[4]` — компилятор выносил его в local-память, и ядро
// было медленнее однострочного (0,95 против 0,79 мс на 11008x2048). Здесь
// четыре явных указателя и четыре явных аккумулятора, без массивов и
// динамической индексации.

kernel void vtm_q4k_gemv4(device const uchar *w [[buffer(0)]],
                          device const float *x [[buffer(1)]],
                          device float *y [[buffer(2)]],
                          constant GemvArgs &a [[buffer(3)]],
                          uint tg [[threadgroup_position_in_grid]],
                          uint sg [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
    uint grow = (tg * 4u + sg) * 4u;
    uint last = a.rows - 1u;
    uint stride = a.nblocks * 144u;
    uint row0 = grow;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    device const uchar *b0, *b1, *b2, *b3;
    uint l8 = lane & 7u;
    uint li = lane >> 3;

    if (a.rows >= 4u) {
        uint max0 = last - 3u;
        if (row0 > max0) row0 = max0;
    }
    b0 = w + a.w_off + (ulong)row0 * stride;
    b1 = b0 + stride;
    b2 = b1 + stride;
    b3 = b2 + stride;

    for (uint blk = 0; blk < a.nblocks; blk++) {
        for (uint k = 0; k < 2u; k++) {
            uint sb = 4u * k + li;
            uint g = sb >> 1u;
            uint h = sb & 1u;
            uint qoff = 16u + g * 32u + l8 * 4u;
            uint boff = blk * 144u;
            float4 xv = *(device const float4 *)(x + blk * 256u
                                                 + sb * 32u + l8 * 4u);
            float d0, d1, d2, d3, m0, m1, m2, m3;
            uint s0, s1, s2, s3, n0, n1, n2, n3;
            uchar4 c0, c1, c2, c3;

            d0 = vtm_f16_to_f32(b0 + boff);
            d1 = vtm_f16_to_f32(b1 + boff);
            d2 = vtm_f16_to_f32(b2 + boff);
            d3 = vtm_f16_to_f32(b3 + boff);
            m0 = vtm_f16_to_f32(b0 + boff + 2u);
            m1 = vtm_f16_to_f32(b1 + boff + 2u);
            m2 = vtm_f16_to_f32(b2 + boff + 2u);
            m3 = vtm_f16_to_f32(b3 + boff + 2u);
            vtm_q4k_sc_mn(b0 + boff + 4u, sb, s0, n0);
            vtm_q4k_sc_mn(b1 + boff + 4u, sb, s1, n1);
            vtm_q4k_sc_mn(b2 + boff + 4u, sb, s2, n2);
            vtm_q4k_sc_mn(b3 + boff + 4u, sb, s3, n3);
            c0 = *(device const uchar4 *)(b0 + boff + qoff);
            c1 = *(device const uchar4 *)(b1 + boff + qoff);
            c2 = *(device const uchar4 *)(b2 + boff + qoff);
            c3 = *(device const uchar4 *)(b3 + boff + qoff);
            {
                float A0 = d0 * (float)s0, B0 = m0 * (float)n0;
                float A1 = d1 * (float)s1, B1 = m1 * (float)n1;
                float A2 = d2 * (float)s2, B2 = m2 * (float)n2;
                float A3 = d3 * (float)s3, B3 = m3 * (float)n3;
                float w0, w1, w2, w3, f0, f1, f2, f3;

                w0 = A0 * (float)(h ? (c0.x >> 4) : (c0.x & 0x0fu)) - B0;
                w1 = A0 * (float)(h ? (c0.y >> 4) : (c0.y & 0x0fu)) - B0;
                w2 = A0 * (float)(h ? (c0.z >> 4) : (c0.z & 0x0fu)) - B0;
                w3 = A0 * (float)(h ? (c0.w >> 4) : (c0.w & 0x0fu)) - B0;
                a0 += w0 * xv.x + w1 * xv.y + w2 * xv.z + w3 * xv.w;

                f0 = A1 * (float)(h ? (c1.x >> 4) : (c1.x & 0x0fu)) - B1;
                f1 = A1 * (float)(h ? (c1.y >> 4) : (c1.y & 0x0fu)) - B1;
                f2 = A1 * (float)(h ? (c1.z >> 4) : (c1.z & 0x0fu)) - B1;
                f3 = A1 * (float)(h ? (c1.w >> 4) : (c1.w & 0x0fu)) - B1;
                a1 += f0 * xv.x + f1 * xv.y + f2 * xv.z + f3 * xv.w;

                w0 = A2 * (float)(h ? (c2.x >> 4) : (c2.x & 0x0fu)) - B2;
                w1 = A2 * (float)(h ? (c2.y >> 4) : (c2.y & 0x0fu)) - B2;
                w2 = A2 * (float)(h ? (c2.z >> 4) : (c2.z & 0x0fu)) - B2;
                w3 = A2 * (float)(h ? (c2.w >> 4) : (c2.w & 0x0fu)) - B2;
                a2 += w0 * xv.x + w1 * xv.y + w2 * xv.z + w3 * xv.w;

                f0 = A3 * (float)(h ? (c3.x >> 4) : (c3.x & 0x0fu)) - B3;
                f1 = A3 * (float)(h ? (c3.y >> 4) : (c3.y & 0x0fu)) - B3;
                f2 = A3 * (float)(h ? (c3.z >> 4) : (c3.z & 0x0fu)) - B3;
                f3 = A3 * (float)(h ? (c3.w >> 4) : (c3.w & 0x0fu)) - B3;
                a3 += f0 * xv.x + f1 * xv.y + f2 * xv.z + f3 * xv.w;
            }
        }
    }

    {
        float t0 = simd_sum(a0), t1 = simd_sum(a1);
        float t2 = simd_sum(a2), t3 = simd_sum(a3);
        if (lane == 0u) {
            if (grow < a.rows) y[grow] = t0;
            if (grow + 1u < a.rows) y[grow + 1u] = t1;
            if (grow + 2u < a.rows) y[grow + 2u] = t2;
            if (grow + 3u < a.rows) y[grow + 3u] = t3;
        }
    }
}
