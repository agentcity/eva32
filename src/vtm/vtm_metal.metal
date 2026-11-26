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
#include <metal_simdgroup_matrix>
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

// ===========================================================================
// 0214: тайловый батчевый GEMM — переиспользование фрагмента B.
//
// Что не так с ядром выше (VTM_GEMM_KERNEL, замер 0211/0214): threadgroup —
// 8–16 строк, по одной строке на warp; на каждую итерацию токена поток читает
// float4 активаций из threadgroup-памяти и делает 4 MAC, то есть 16 Б
// threadgroup-трафика на 4 MAC (4 Б/MAC). Веса при этом читаются один раз на
// строку-блок — узкое место не полоса весов, а LDS: на один FFMA приходится
// 0.25 LDS.128, а LDS.128 на Apple GPU вчетверо дороже FFMA по слотам выдачи.
//
// Здесь каждый warp считает ДВЕ строки (по два аккумулятора на токен), и один
// float4 активаций идёт в 8 MAC вместо 4: 2 Б/MAC. Вес по-прежнему читается
// один раз (строка-блок принадлежит ровно одному threadgroup), декодирование
// Q4_K/Q6_K не удваивается: на 2 строки × 4 колонки — две uchar4-загрузки.
//
// Аргументы (GemmArgs) несут смещения и шаги, поэтому ядро читает вход и
// пишет выход прямо в памяти хоста/арене пачки (interp.c) — без
// транспонирования на хосте. Раскладка входа:
//   x[a.x_off + j*a.x_stride + t]   (j — токен, t — колонка)
//   y[a.y_off + j*a.y_stride + r*a.y_rstride]
// x_stride = 0 — один вектор на все токены (общая активация); y_rstride = 1 —
// токен-мажорный выход арены, y_rstride = n_tokens и y_stride = 1 — прежняя
// раскладка батчевых ядер (y[r*n_tokens + j]).
// ===========================================================================

typedef struct {
    uint rows;
    uint cols;
    uint nblocks;   // cols / 256 для kind 6/7
    uint n_tokens;  // токенов в chunk'е (NT варианта)
    uint x_off;     // смещение входа в x, float
    uint y_off;     // смещение выхода в y, float
    uint x_stride;  // шаг токена во входе, float
    uint y_stride;  // шаг токена в выходе, float
    uint y_rstride; // шаг строки весов в выходе, float
    uint pad0;      // выравнивание до ulong (sizeof == 56)
    ulong w_off;    // смещение блоков веса в базовом MTLBuffer
    ulong sc_off;   // смещение fp32-масштабов (int8), float
} GemmArgs;

#define VTM_TILE_TG    256u   // потоков: 8 warp по 32
#define VTM_TILE_RPT   2u     // строк на warp (== строк на поток)
#define VTM_TILE_ROWS  16u    // строк на threadgroup

// Коллективная загрузка половины блока (128 колонок × NT токенов) в
// threadgroup-память: 256 потоков делают NT/8 float4-загрузок.
#define VTM_TILE_LOAD(NT, COLBASE)                                           \
    do {                                                                     \
        uint m;                                                              \
        _Pragma("unroll")                                                    \
        for (m = 0; m < (NT) / 8u; m++) {                                    \
            uint flat = (tid + VTM_TILE_TG * m) * 4u;                        \
            uint lj = flat >> 7;                                             \
            uint li0 = flat & 127u;                                          \
            *(threadgroup float4 *)(xs + lj * 128u + li0) =                  \
                *(device const float4 *)(xt + (ulong)lj * a.x_stride         \
                                         + a.x_off + (COLBASE) + li0);       \
        }                                                                    \
        threadgroup_barrier(mem_flags::mem_threadgroup);                     \
    } while (0)

// Четыре подряд идущих веса строки — одна uchar4-загрузка.
static inline float4 vtm_q4k_quad(device const uchar *bp, uint sb, uint l8) {
    float d = vtm_f16_to_f32(bp);
    float dmin = vtm_f16_to_f32(bp + 2u);
    uint g = sb >> 1;
    uint h = sb & 1u;
    uint scv, mnv;
    float A, B;
    uchar4 b;

    vtm_q4k_sc_mn(bp + 4u, sb, scv, mnv);
    b = *(device const uchar4 *)(bp + 16u + g * 32u + l8 * 4u);
    A = d * (float)scv;
    B = dmin * (float)mnv;
    return float4(A * (float)(h ? (b.x >> 4) : (b.x & 0x0fu)) - B,
                  A * (float)(h ? (b.y >> 4) : (b.y & 0x0fu)) - B,
                  A * (float)(h ? (b.z >> 4) : (b.z & 0x0fu)) - B,
                  A * (float)(h ? (b.w >> 4) : (b.w & 0x0fu)) - B);
}

static inline float4 vtm_q6k_quad(device const uchar *bp, uint sb, uint l4) {
    device const uchar *ql = bp;
    device const uchar *qh = bp + 128u;
    device const char *sc = (device const char *)(bp + 192u);
    float d = vtm_f16_to_f32(bp + 208u);
    uint jj = l4 * 4u;
    uint chunk = sb >> 3;
    uint in = sb & 7u;
    uint quad = in >> 1;
    uint hs = in & 1u;
    uchar4 qlb = *(device const uchar4 *)(ql + chunk * 64u
                  + (quad & 1u) * 32u + hs * 16u + jj);
    uchar4 qhb = *(device const uchar4 *)(qh + chunk * 32u + hs * 16u + jj);
    float s = d * (float)sc[sb];

    return float4(
        s * (float)((int)((quad >= 2u ? (qlb.x >> 4) : (qlb.x & 0x0fu))
                          | (((qhb.x >> (quad * 2u)) & 3u) << 4)) - 32),
        s * (float)((int)((quad >= 2u ? (qlb.y >> 4) : (qlb.y & 0x0fu))
                          | (((qhb.y >> (quad * 2u)) & 3u) << 4)) - 32),
        s * (float)((int)((quad >= 2u ? (qlb.z >> 4) : (qlb.z & 0x0fu))
                          | (((qhb.z >> (quad * 2u)) & 3u) << 4)) - 32),
        s * (float)((int)((quad >= 2u ? (qlb.w >> 4) : (qlb.w & 0x0fu))
                          | (((qhb.w >> (quad * 2u)) & 3u) << 4)) - 32));
}

static inline float4 vtm_i8_quad(device const char *q, uint base, uint col,
                                 device const float *scales, ulong sc_off) {
    uchar4 b = *(device const uchar4 *)(q + base + col);
    float sc = scales[sc_off + ((base + col) >> 7)];

    return float4(sc * (float)(char)b.x, sc * (float)(char)b.y,
                  sc * (float)(char)b.z, sc * (float)(char)b.w);
}

#define VTM_TILE_DECL(NT)                                                    \
    threadgroup float xs[128u * (NT)];                                       \
    device const uchar *rp0;                                                 \
    device const uchar *rp1;                                                 \
    float acc0[(NT)], acc1[(NT)];                                            \
    uint l8 = lane & 7u;                                                     \
    uint li = lane >> 3;                                                     \
    uint l4 = lane & 3u;                                                     \
    uint liq = lane >> 2;                                                    \
    uint row0 = tg * VTM_TILE_ROWS + sg * VTM_TILE_RPT;                      \
    uint row1 = row0 + 1u;                                                   \
    uint last = a.rows - 1u;                                                 \
    uint act0 = row0 < a.rows;                                               \
    uint act1 = row1 < a.rows;                                               \
    (void)l8; (void)li; (void)l4; (void)liq;                                 \
    if (row0 > last) row0 = last;                                            \
    if (row1 > last) row1 = last;

// Тело: две строки на поток, B-фрагмент (float4) из threadgroup-памяти — на
// 8 MAC. SB и IX — выражения от k/lane конкретного формата весов.
#define VTM_TILE_BODY(NT, BSTEP, KSTEP, QUAD, SB, IX, OFFX)                  \
    for (uint j = 0; j < (NT); j++) { acc0[j] = 0.0f; acc1[j] = 0.0f; }      \
    rp0 = w + a.w_off + (ulong)row0 * a.nblocks * (KSTEP);                   \
    rp1 = w + a.w_off + (ulong)row1 * a.nblocks * (KSTEP);                   \
    for (uint b = 0; b < a.nblocks; b++) {                                   \
        _Pragma("unroll")                                                    \
        for (uint k = 0; k < 2u; k++) {                                      \
            VTM_TILE_LOAD(NT, b * 256u + k * 128u);                          \
            float4 w0 = QUAD(rp0 + b * (BSTEP), (SB), (IX));                 \
            float4 w1 = QUAD(rp1 + b * (BSTEP), (SB), (IX));                 \
            _Pragma("unroll")                                                \
            for (uint j = 0; j < (NT); j++) {                                \
                float4 xv = *(threadgroup const float4 *)(xs + j * 128u      \
                             + (OFFX));                                      \
                acc0[j] += w0.x * xv.x + w0.y * xv.y + w0.z * xv.z           \
                         + w0.w * xv.w;                                      \
                acc1[j] += w1.x * xv.x + w1.y * xv.y + w1.z * xv.z           \
                         + w1.w * xv.w;                                      \
            }                                                                \
            threadgroup_barrier(mem_flags::mem_threadgroup);                 \
        }                                                                    \
    }                                                                        \
    _Pragma("unroll")                                                        \
    for (uint j = 0; j < (NT); j++) {                                        \
        float t0 = simd_sum(acc0[j]);                                        \
        float t1 = simd_sum(acc1[j]);                                        \
        ulong ob = (ulong)a.y_off + (ulong)j * a.y_stride;                   \
        ulong o0 = ob + (ulong)row0 * a.y_rstride;                           \
        ulong o1 = ob + (ulong)row1 * a.y_rstride;                           \
        if (lane == 0u && j < a.n_tokens) {                                  \
            if (act0) y[o0] = t0;                                            \
            if (act1) y[o1] = t1;                                            \
        }                                                                    \
    }

#define VTM_TILE_KERNEL(NAME, NT)                                            \
kernel void NAME(device const uchar *w [[buffer(0)]],                        \
                 device const float *xt [[buffer(1)]],                       \
                 device float *y [[buffer(2)]],                              \
                 constant GemmArgs &a [[buffer(3)]],                         \
                 uint tg [[threadgroup_position_in_grid]],                   \
                 uint sg [[simdgroup_index_in_threadgroup]],                 \
                 uint lane [[thread_index_in_simdgroup]],                    \
                 uint tid [[thread_index_in_threadgroup]]) {                 \
    VTM_TILE_DECL(NT)                                                        \
    (void)tid;                                                               \
    VTM_TILE_BODY(NT, 144u, 144u, vtm_q4k_quad, (4u * k + li), l8,           \
                  (li * 32u + l8 * 4u))                                      \
}

#define VTM_TILE_KERNEL_Q6(NAME, NT)                                         \
kernel void NAME(device const uchar *w [[buffer(0)]],                        \
                 device const float *xt [[buffer(1)]],                       \
                 device float *y [[buffer(2)]],                              \
                 constant GemmArgs &a [[buffer(3)]],                         \
                 uint tg [[threadgroup_position_in_grid]],                   \
                 uint sg [[simdgroup_index_in_threadgroup]],                 \
                 uint lane [[thread_index_in_simdgroup]],                    \
                 uint tid [[thread_index_in_threadgroup]]) {                 \
    VTM_TILE_DECL(NT)                                                        \
    (void)tid;                                                               \
    VTM_TILE_BODY(NT, 210u, 210u, vtm_q6k_quad, (8u * k + liq), l4,          \
                  (liq * 16u + l4 * 4u))                                     \
}

VTM_TILE_KERNEL(vtm_q4k_gemm_t32, 32u)
VTM_TILE_KERNEL(vtm_q4k_gemm_t16, 16u)
VTM_TILE_KERNEL_Q6(vtm_q6k_gemm_t32, 32u)
VTM_TILE_KERNEL_Q6(vtm_q6k_gemm_t16, 16u)

// int8-таблица словаря (kind 4): блок 128 колонок, дорожка — 4 колонки,
// масштаб общий на блок. Две строки на поток, как у Q4_K/Q6_K.
#define VTM_I8_TILE_KERNEL(NAME, NT)                                         \
kernel void NAME(device const char *q [[buffer(0)]],                         \
                 device const float *xt [[buffer(1)]],                       \
                 device float *y [[buffer(2)]],                              \
                 constant GemmArgs &a [[buffer(3)]],                         \
                 device const float *scales [[buffer(4)]],                   \
                 uint tg [[threadgroup_position_in_grid]],                   \
                 uint sg [[simdgroup_index_in_threadgroup]],                 \
                 uint lane [[thread_index_in_simdgroup]],                    \
                 uint tid [[thread_index_in_threadgroup]]) {                 \
    VTM_TILE_DECL(NT)                                                        \
    (void)tid; (void)li; (void)liq;                                          \
    uint base0 = row0 * a.cols;                                              \
    uint base1 = row1 * a.cols;                                              \
    for (uint j = 0; j < (NT); j++) { acc0[j] = 0.0f; acc1[j] = 0.0f; }      \
    for (uint blk = 0; blk < a.cols; blk += 128u) {                          \
        VTM_TILE_LOAD(NT, blk);                                              \
        float4 w0 = vtm_i8_quad(q + a.w_off, base0, blk + lane * 4u,         \
                                scales, a.sc_off);                           \
        float4 w1 = vtm_i8_quad(q + a.w_off, base1, blk + lane * 4u,         \
                                scales, a.sc_off);                           \
        _Pragma("unroll")                                                    \
        for (uint j = 0; j < (NT); j++) {                                    \
            float4 xv = *(threadgroup const float4 *)(xs + j * 128u          \
                         + lane * 4u);                                       \
            acc0[j] += w0.x * xv.x + w0.y * xv.y + w0.z * xv.z               \
                     + w0.w * xv.w;                                          \
            acc1[j] += w1.x * xv.x + w1.y * xv.y + w1.z * xv.z               \
                     + w1.w * xv.w;                                          \
        }                                                                    \
        threadgroup_barrier(mem_flags::mem_threadgroup);                     \
    }                                                                        \
    _Pragma("unroll")                                                        \
    for (uint j = 0; j < (NT); j++) {                                        \
        float t0 = simd_sum(acc0[j]);                                        \
        float t1 = simd_sum(acc1[j]);                                        \
        ulong ob = (ulong)a.y_off + (ulong)j * a.y_stride;                   \
        ulong o0 = ob + (ulong)row0 * a.y_rstride;                           \
        ulong o1 = ob + (ulong)row1 * a.y_rstride;                           \
        if (lane == 0u && j < a.n_tokens) {                                  \
            if (act0) y[o0] = t0;                                            \
            if (act1) y[o1] = t1;                                            \
        }                                                                    \
    }                                                                        \
}

VTM_I8_TILE_KERNEL(vtm_i8_gemm_t32, 32u)
VTM_I8_TILE_KERNEL(vtm_i8_gemm_t16, 16u)

// ===========================================================================
// 0216: f16-GEMM на simdgroup_matrix — матричный блок Apple.
//
// Диагноз 0214: f32-тайловое ядро на 128 MAC (4 веса × 32 токена) тратит
// ~128 FFMA + 32 LDS.128 + ~16 инструкций декванта, то есть упирается в ВЫДАЧУ
// инструкций, а не в АЛУ: 307–325 GMAC/s при пике f32 ≈ 2,6 TMAC/s (12%).
// Матричный блок 8×8 half делает 512 MAC одной инструкцией, поэтому те же
// 128 MAC превращаются из 160 инструкций в 4 MMA + 4 LDS: 32 токена на тайл
// перестают быть «32 отдельных FFMA».
//
// Схема (та же, что у kernel_mul_mm в llama.cpp metal):
//   ws[MT][KT+PAD]  — деквантованные веса: ДЕКВАНТ ОДИН РАЗ на k-tile;
//   xs[NT][KT+PAD]  — активации пачки, f32→half при загрузке из устройства;
//   acc[NT/8]       — simdgroup_float8x8: накопление f32, выдача f32.
// A-фрагмент (8 строк × 8 k) читается из ws, B-фрагмент (8 k × 8 токенов) —
// из xs с transpose: xs лежит по токенам (как активации в памяти хоста), а
// фрагменту B нужна раскладка [k][токен], поэтому транспонирование делает
// сам simdgroup_load, без отдельной перекладки и без второго чтения веса.
//
// Деквантование: 16 дорожек на строку весов, дорожка — 4 подряд идущих
// значения (одна uchar4-загрузка). 64 колонки k-tile — это ровно 2 под-блока
// Q4_K (по 32) или 4 под-блока Q6_K (по 16), поэтому sc/mn читаются один раз
// на дорожку и работают на 4 веса: вес читается ровно один раз за k-tile, как
// и в f32-ядре. Дорожки 0..15 — строка r, 16..31 — строка r+1 (два ряда на
// итерацию, четыре независимые загрузки в полёте).
//
// Точность: веса и активации — half (11 бит мантиссы), накопление — f32.
// Ошибка на вес ~5e-4 относительной, по сумме 2048 слагаемых не растёт
// (независимые округления): контракт f16-пути — argmax + Δlogits ≤ 1e-2.
// Строгий допуск 1e-5 держит только f32-путь (VTM_METAL_GEMM_F16=0).
//
// Кэш: ws и xs вместе 14 КиБ (MT=64, NT=32) — два threadgroup'а на ядро
// (потолок M1 Pro 32 КиБ), поэтому короткие матрицы (2048 строк = 32 TG)
// укладываются в одну волну занятости и не дают «хвоста», как f32-ядро с
// 16 строками на TG.
// ===========================================================================

#define VTM_F16_MT   64u                    // строк весов на threadgroup
#define VTM_F16_KT   64u                    // колонок (k) на k-tile
#define VTM_F16_ST   (VTM_F16_KT + 8u)      // шаг строки: 72 half = 144 Б =
                                            // 36 банков, соседние строки
                                            // сдвинуты на 4 банка

// Деквант quadruple-а в half: (sb, koff) считаются от номера k-tile, поэтому
// формат (Q4_K/Q6_K) задаётся только этими двумя выражениями.
static inline half4 vtm_q4k_quad_h(device const uchar *bp, uint sb, uint koff) {
    float4 f = vtm_q4k_quad(bp, sb, (koff & 31u) >> 2);

    return half4((half)f.x, (half)f.y, (half)f.z, (half)f.w);
}

static inline half4 vtm_q6k_quad_h(device const uchar *bp, uint sb, uint koff) {
    float4 f = vtm_q6k_quad(bp, sb, (koff & 15u) >> 2);

    return half4((half)f.x, (half)f.y, (half)f.z, (half)f.w);
}

// NT — токенов в варианте (16 или 32), BSTEP — байт на блок веса (144/210),
// SB(k34, l16) — под-блок формата, KOFF(l16) — смещение четырёх значений
// внутри k-tile.
#define VTM_F16_KERNEL(NAME, NT, BSTEP, QUADH, SB, KOFF)                     \
kernel void NAME(device const uchar *w [[buffer(0)]],                        \
                 device const float *xt [[buffer(1)]],                       \
                 device float *y [[buffer(2)]],                              \
                 constant GemmArgs &a [[buffer(3)]],                         \
                 uint tg [[threadgroup_position_in_grid]],                   \
                 uint sg [[simdgroup_index_in_threadgroup]],                 \
                 uint lane [[thread_index_in_simdgroup]],                    \
                 uint tid [[thread_index_in_threadgroup]]) {                 \
    threadgroup half ws[VTM_F16_MT * VTM_F16_ST];                            \
    threadgroup half xs[(NT) * VTM_F16_ST];                                  \
    simdgroup_float8x8 acc[(NT) / 8u];                                       \
    uint rows8 = tg * VTM_F16_MT + sg * 8u;                                  \
    uint last8 = a.rows - 8u;                                                \
    uint two = lane >> 4;                                                    \
    uint l16 = lane & 15u;                                                   \
    uint nt;                                                                 \
                                                                              \
    /* Хвост по строкам: тайл, вышедший за матрицу, клампят к последним 8    \
       строкам — считает то же самое (значения те же), лишние строки не      \
       пишутся. Так rows не обязана быть кратной 64, а строки, которые       \
       никто не считает, не появляются. */                                   \
    if (rows8 > last8) rows8 = last8;                                        \
    _Pragma("unroll")                                                        \
    for (nt = 0; nt < (NT) / 8u; nt++)                                       \
        acc[nt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);            \
                                                                              \
    for (uint b = 0; b < a.nblocks; b++) {                                   \
        _Pragma("unroll")                                                    \
        for (uint kt4 = 0; kt4 < 4u; kt4++) {                                \
            uint kt = b * 256u + kt4 * 64u;                                  \
            uint m;                                                          \
                                                                              \
            /* xs: 256 потоков по два float4 (NT=32) — строка токена, k       \
               подряд, поэтому загрузка из устройства слитная. */             \
            _Pragma("unroll")                                                \
            for (m = 0; m < (NT) / 16u; m++) {                               \
                uint fl = (tid + 256u * m) * 4u;                             \
                uint j = fl >> 6;                                            \
                uint k = fl & 63u;                                           \
                float4 xv = *(device const float4 *)(xt                       \
                             + (ulong)j * a.x_stride + a.x_off + kt + k);     \
                *(threadgroup half4 *)(xs + j * VTM_F16_ST + k) =             \
                    half4((half)xv.x, (half)xv.y, (half)xv.z, (half)xv.w);    \
            }                                                                \
            /* ws: 8 строк warpa, две на итерацию (дорожки 0..15 и 16..31).*/ \
            _Pragma("unroll")                                                \
            for (m = 0; m < VTM_F16_MT / 16u; m++) {                         \
                uint r = m * 2u + two;                                       \
                uint koff = KOFF;                                            \
                uint sb = SB;                                                \
                device const uchar *bp = w + a.w_off                         \
                    + (ulong)(rows8 + r) * a.nblocks * (BSTEP) + b * (BSTEP); \
                *(threadgroup half4 *)(ws + (sg * 8u + r) * VTM_F16_ST        \
                                       + koff) = QUADH(bp, sb, koff);         \
            }                                                                \
            threadgroup_barrier(mem_flags::mem_threadgroup);                 \
            _Pragma("unroll")                                                \
            for (uint kk = 0; kk < VTM_F16_KT; kk += 8u) {                   \
                simdgroup_half8x8 fa;                                        \
                                                                              \
                simdgroup_load(fa, ws + (sg * 8u) * VTM_F16_ST + kk,          \
                               VTM_F16_ST);                                  \
                _Pragma("unroll")                                            \
                for (nt = 0; nt < (NT) / 8u; nt++) {                         \
                    simdgroup_half8x8 fb;                                    \
                    simdgroup_load(fb, xs + (nt * 8u) * VTM_F16_ST + kk,      \
                                   VTM_F16_ST, ulong2(0, 0), true);          \
                    simdgroup_multiply_accumulate(acc[nt], fa, fb, acc[nt]);  \
                }                                                            \
            }                                                                \
            threadgroup_barrier(mem_flags::mem_threadgroup);                 \
        }                                                                    \
    }                                                                        \
    /* Выход: y[токен][строка], строки токена подряд (y_rstride == 1) —       \
       поэтому store тоже транспонированный: элементы C[строка][токен] лягут \
       в память как [токен][строка]. */                                       \
    _Pragma("unroll")                                                        \
    for (nt = 0; nt < (NT) / 8u; nt++) {                                     \
        if ((nt * 8u) < a.n_tokens) {                                        \
            device float *op = y + (ulong)a.y_off                            \
                             + (ulong)(nt * 8u) * a.y_stride                 \
                             + (ulong)rows8;                                 \
            simdgroup_store(acc[nt], op, a.y_stride, ulong2(0, 0), true);    \
        }                                                                    \
    }                                                                        \
}

VTM_F16_KERNEL(vtm_q4k_gemm_f16_t32, 32u, 144u, vtm_q4k_quad_h,
               (kt4 * 2u + (l16 >> 3)), ((l16 >> 3) * 32u + (l16 & 7u) * 4u))
VTM_F16_KERNEL(vtm_q4k_gemm_f16_t16, 16u, 144u, vtm_q4k_quad_h,
               (kt4 * 2u + (l16 >> 3)), ((l16 >> 3) * 32u + (l16 & 7u) * 4u))
VTM_F16_KERNEL(vtm_q6k_gemm_f16_t32, 32u, 210u, vtm_q6k_quad_h,
               (kt4 * 4u + ((l16 >> 2) & 3u)),
               (((l16 >> 2) & 3u) * 16u + (l16 & 3u) * 4u))
VTM_F16_KERNEL(vtm_q6k_gemm_f16_t16, 16u, 210u, vtm_q6k_quad_h,
               (kt4 * 4u + ((l16 >> 2) & 3u)),
               (((l16 >> 2) & 3u) * 16u + (l16 & 3u) * 4u))

// ===========================================================================
// 0214: attention ПАЧКИ prefill на GPU.
//
// На 250 токенах CPU-attention стоил 1,4 с из 5,0 (профиль 0214: op=0x0B 25%
// стены), на 500 — уже 5,8 с из 15: это O(C²) в double на одном потоке, пока
// GPU стоит. Здесь каждый токен пачки — та же задача, что у декода (q_len == 1
// против кольцевого KV-кэша), поэтому ядро — копия vtm_attn_decode с двумя
// отличиями: threadgroup'а выбирает (токен пачки, голова), а границы видимых
// ключей считаются от позиции ЭТОГО токена (каузальность бесплатна: у токена i
// свой n_keys = pos0 + i + 1 - start, поэтому "past" и "future" не нужны).
//
// K/V пачки уже лежат в кольце: KVW идёт по всем токенам пачки до ATTN.
// Точность: f32 + online-softmax, как у декода (контракт — argmax + Δ≤1e-2).
// ===========================================================================

typedef struct {
    uint  n_heads;
    uint  rep;        // n_heads / n_kv_heads
    uint  head_dim;
    uint  kv_dim;
    uint  win;        // слотов в кольце
    uint  n_tokens;   // токенов в пачке
    uint  pos0;       // позиция первого токена (+ imm инструкции)
    uint  stride;     // шаг токена пачки в q/out, float
    uint  q_off;      // смещение Q первого токена, float
    uint  out_off;    // смещение out первого токена, float
    uint  pad0;
    uint  pad1;
    float scale;
} AttnBatchArgs;

kernel void vtm_attn_prefill(device const float *q [[buffer(0)]],
                             device const float *kc [[buffer(1)]],
                             device const float *vc [[buffer(2)]],
                             device float *out [[buffer(3)]],
                             constant AttnBatchArgs &a [[buffer(4)]],
                             uint tg [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]]) {
    uint i = tg / a.n_heads;          // токен пачки
    uint h = tg - i * a.n_heads;      // голова
    uint kvh = h / a.rep;
    uint pos = a.pos0 + i;
    uint keys = pos + 1u;
    uint start = keys > a.win ? keys - a.win : 0u;
    uint nk = keys - start;
    uint slot = start % a.win;
    device const float *qh = q + (ulong)i * a.stride + a.q_off
                           + (ulong)h * a.head_dim;
    device const float *kb = kc + (ulong)kvh * a.head_dim;
    device const float *vb = vc + (ulong)kvh * a.head_dim;
    float qv[8];
    float acc[8];
    float m = -INFINITY;
    float l = 0.0f;
    uint d, n;

    _Pragma("unroll")
    for (d = 0; d < 8u; d++) {
        uint j = lane + d * 32u;
        qv[d] = (j < a.head_dim) ? qh[j] : 0.0f;
        acc[d] = 0.0f;
    }

    for (n = 0; n < nk; n++) {
        device const float *krow = kb + (ulong)slot * a.kv_dim;
        device const float *vrow = vb + (ulong)slot * a.kv_dim;
        float dot = 0.0f;
        float s, p, corr;

        _Pragma("unroll")
        for (d = 0; d < 8u; d++) {
            uint j = lane + d * 32u;
            if (j < a.head_dim) dot = fma(qv[d], krow[j], dot);
        }
        dot = simd_sum(dot) * a.scale;
        s = (dot > m) ? dot : m;
        corr = exp(m - s);
        p = exp(dot - s);
        l = l * corr + p;
        _Pragma("unroll")
        for (d = 0; d < 8u; d++) {
            uint j = lane + d * 32u;
            if (j < a.head_dim) acc[d] = fma(p, vrow[j], acc[d] * corr);
        }
        m = s;
        if (++slot == a.win) slot = 0u;
    }

    {
        float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
        device float *orow = out + (ulong)i * a.stride + a.out_off
                           + (ulong)h * a.head_dim;

        _Pragma("unroll")
        for (d = 0; d < 8u; d++) {
            uint j = lane + d * 32u;
            if (j < a.head_dim) orow[j] = acc[d] * inv;
        }
    }
}
