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
    uint reserved;
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

// Q4_K GEMV: 4 строки на threadgroup, simdgroup (32 дорожки) — одна строка.
// Дорожка lane обрабатывает байт lane в каждой 32-байтовой половине блока:
// под-блок 2r + (lane>>4), значения (lane&15) и (lane&15)+16.
// (Схема «64 строки на threadgroup + x в threadgroup-память» проверена
// замером: на M1 Pro она медленнее из-за регистрового давления 8
// аккумуляторов, поэтому x остаётся в глобальной памяти — он кэшируется.)
kernel void vtm_q4k_gemv(device const uchar *w [[buffer(0)]],
                         device const float *x [[buffer(1)]],
                         device float *y [[buffer(2)]],
                         constant GemvArgs &a [[buffer(3)]],
                         uint tg [[threadgroup_position_in_grid]],
                         uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    uint row = tg * 4u + sg;
    if (row >= a.rows) return;

    // Смещение строки — 32 бита: тензор < 2 ГиБ, 64-битная арифметика дороже.
    device const uchar *rp = w + a.w_off + (uint)row * a.nblocks * 144u;
    // (sc,mn) считает дорожка j = lane&7, остальные получают их через
    // simd_shuffle: иначе 32 дорожки делают по 24 скалярных загрузки scales
    // на блок.
    uint j = lane & 7u;
    float acc = 0.0f;

    for (uint b = 0; b < a.nblocks; b++) {
        device const uchar *bp = rp + b * 144u;
        float d = vtm_f16_to_f32(bp);
        float dmin = vtm_f16_to_f32(bp + 2u);
        device const uchar *scp = bp + 4u;
        device const uchar *qs = bp + 16u;
        uint colbase = b * 256u;
        uint scv, mnv;

        vtm_q4k_sc_mn(scp, j, scv, mnv);

        // Группа 32 байт = два под-блока: младшие нибблы — 2g, старшие — 2g+1;
        // каждый под-блок 32 значения, дорожка lane — значение lane. Один
        // байт даёт два значения (низ/верх), 4 группы — 8 значений на блок.
        for (uint g = 0; g < 4u; g++) {
            uchar byte = qs[g * 32u + lane];
            uint sc0 = simd_shuffle(scv, ushort(2u * g));
            uint mn0 = simd_shuffle(mnv, ushort(2u * g));
            uint sc1 = simd_shuffle(scv, ushort(2u * g + 1u));
            uint mn1 = simd_shuffle(mnv, ushort(2u * g + 1u));
            float dl0 = d * (float)sc0;
            float ml0 = dmin * (float)mn0;
            float dl1 = d * (float)sc1;
            float ml1 = dmin * (float)mn1;
            uint col = colbase + (2u * g) * 32u + lane;
            acc += (dl0 * (float)(byte & 0x0fu) - ml0) * x[col]
                 + (dl1 * (float)(byte >> 4) - ml1) * x[col + 32u];
        }
    }

    float total = simd_sum(acc);
    if (lane == 0u) y[row] = total;
}

// Q6_K GEMV: 4 строки на threadgroup, simdgroup — строка. 16 под-блоков по 16
// значений на блок: половина дорожек (0..15) берёт чётный под-блок, половина
// (16..31) — нечётный; адреса ql/qh выводятся из номера под-блока, как в
// qk_i8b_expand_row (backend_neon.c) и dequant_row ggml.
kernel void vtm_q6k_gemv(device const uchar *w [[buffer(0)]],
                         device const float *x [[buffer(1)]],
                         device float *y [[buffer(2)]],
                         constant GemvArgs &a [[buffer(3)]],
                         uint tg [[threadgroup_position_in_grid]],
                         uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    uint row = tg * 4u + sg;
    if (row >= a.rows) return;

    device const uchar *rp = w + a.w_off + (uint)row * a.nblocks * 210u;
    float acc = 0.0f;

    for (uint b = 0; b < a.nblocks; b++) {
        device const uchar *bp = rp + b * 210u;
        float d = vtm_f16_to_f32(bp + 208u);
        device const uchar *ql = bp;
        device const uchar *qh = bp + 128u;
        device const char *sc = (device const char *)(bp + 192u);
        uint colbase = b * 256u;

        for (uint t = 0; t < 8u; t++) {
            uint sb = 2u * t + (lane >> 4);
            uint j = lane & 15u;
            uint chunk = sb >> 3;
            uint in = sb & 7u;
            uint quad = in >> 1;
            uint hs = in & 1u;
            uchar qlb = ql[chunk * 64u + (quad & 1u) * 32u + hs * 16u + j];
            uchar qhb = qh[chunk * 32u + hs * 16u + j];
            uint nib = (quad >= 2u) ? (uint)(qlb >> 4) : (uint)(qlb & 0x0fu);
            uint two = (uint)(qhb >> (quad * 2u)) & 3u;
            int q = (int)(nib | (two << 4)) - 32;
            float s = (float)sc[sb];
            uint col = colbase + sb * 16u + j;
            acc += d * s * (float)q * x[col];
        }
    }

    float total = simd_sum(acc);
    if (lane == 0u) y[row] = total;
}

// int8-таблица словаря (kind 4, tied lm_head): 4 строки на threadgroup,
// simdgroup — строка. Масштаб — общий на блок 128 подряд идущих значений
// сплющенной матрицы; делим индекс на 128 (для cols = 2048 это 16 делений на
// строку, не горячо). x читается всеми строками одна и та же — кэшируется.
kernel void vtm_i8_gemv(device const char *q [[buffer(0)]],
                        device const float *x [[buffer(1)]],
                        device float *y [[buffer(2)]],
                        constant GemvArgs &a [[buffer(3)]],
                        device const float *scales [[buffer(4)]],
                        uint tg [[threadgroup_position_in_grid]],
                        uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    uint row = tg * 4u + sg;
    uint base, col;

    if (row >= a.rows) return;
    base = row * a.cols;
    device const char *qr = q + a.w_off + base;
    float acc = 0.0f;

    for (col = lane; col < a.cols; col += 32u) {
        uint blk = (base + col) >> 7; // /128
        float sc = scales[a.sc_off + blk];

        acc += (float)qr[col] * sc * x[col];
    }

    float total = simd_sum(acc);
    if (lane == 0u) y[row] = total;
}
