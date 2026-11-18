// Кодек тритов и преобразования float16/bfloat16.
//
// Формат v0.1 (docs/trit-format.md): слово u32 little-endian несёт 20 тритов,
// цифра -1 = 0, 0 = 1, +1 = 2, разряды младший-вперёд: word = Σ d_i·3^i.
// Блок — 8 слов = 160 тритов — имеет один общий масштаб.
#include "vtm.h"

#include <string.h>

// --- упаковка/распаковка ---------------------------------------------------

uint32_t vtm_trit_pack(const int8_t *trits, int n) {
    uint32_t word = 0;
    uint32_t p = 1;
    int i;
    if (!trits || n < 0) return 0;
    if (n > VTM_TRITS_PER_WORD) n = VTM_TRITS_PER_WORD;
    for (i = 0; i < VTM_TRITS_PER_WORD; i++) {
        unsigned digit = 1; // 0 тритов -> цифра 0
        if (i < n) digit = trits[i] > 0 ? 2u : (trits[i] < 0 ? 0u : 1u);
        word += digit * p;
        p *= 3u;
    }
    return word;
}

uint32_t vtm_trit_pack20(const int8_t trits[20]) {
    if (!trits) return 0;
    return vtm_trit_pack(trits, 20);
}

void vtm_trit_unpack20(uint32_t word, int8_t out[20]) {
    int i;
    if (!out) return;
    for (i = 0; i < VTM_TRITS_PER_WORD; i++) {
        unsigned d = word % 3u;
        word /= 3u;
        out[i] = (int8_t)(d == 2u ? 1 : (d == 0u ? -1 : 0));
    }
}

int vtm_trit_decode_words(const uint32_t *words, uint32_t n_words,
                          int8_t *out, uint32_t n_trits) {
    uint32_t w;
    uint32_t done = 0;
    if ((!words && n_words) || (!out && n_trits)) return -1;
    if ((uint64_t)n_trits > (uint64_t)n_words * VTM_TRITS_PER_WORD)
        return -1;
    for (w = 0; w < n_words && done < n_trits; w++) {
        int8_t tmp[VTM_TRITS_PER_WORD];
        uint32_t take;
        if (words[w] > VTM_TRIT_MAX_WORD) return -1;
        vtm_trit_unpack20(words[w], tmp);
        take = n_trits - done;
        if (take > VTM_TRITS_PER_WORD) take = VTM_TRITS_PER_WORD;
        memcpy(out + done, tmp, take);
        done += take;
    }
    return done == n_trits ? 0 : -1;
}

int vtm_tensor_unpack_trits(const vtm_tensor *t, int8_t *out, uint32_t n_trits) {
    if (!t || !out || t->kind != VTM_TENSOR_TRIT) return -1;
    if (n_trits > t->n_elements || !t->words) return -1;
    return vtm_trit_decode_words(t->words, t->n_blocks * VTM_WORDS_PER_BLOCK,
                                 out, n_trits);
}

// --- IEEE 754 binary16 -----------------------------------------------------

float vtm_half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x03FFu;
    uint32_t bits;
    float f;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign; // ±0
        } else {
            // Субнормаль: приводим к нормали float сдвигом мантиссы.
            int e = 127 - 15 + 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                e--;
            }
            mant &= 0x03FFu;
            bits = sign | ((uint32_t)e << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13); // ±inf / NaN
    } else {
        bits = sign | ((exp - 15u + 127u) << 23) | (mant << 13);
    }
    memcpy(&f, &bits, sizeof f);
    return f;
}

// Округление v / 2^shift к ближайшему чётному (shift > 0).
static uint32_t round_shift(uint32_t v, int shift) {
    uint32_t q, rem, half, lsb;
    if (shift <= 0) return v;
    if (shift > 31) return 0;
    q = v >> shift;
    rem = v & ((1u << shift) - 1u);
    half = 1u << (shift - 1);
    lsb = q & 1u;
    if (rem > half || (rem == half && lsb)) q++;
    return q;
}

uint16_t vtm_float_to_half(float f) {
    uint32_t x;
    uint16_t sign;
    uint32_t exp, mant;
    int eh;

    memcpy(&x, &f, sizeof x);
    sign = (uint16_t)((x >> 16) & 0x8000u);
    exp = (x >> 23) & 0xFFu;
    mant = x & 0x007FFFFFu;

    if (exp == 0xFFu) {
        if (mant) return (uint16_t)(sign | 0x7E00u); // тихий NaN
        return (uint16_t)(sign | 0x7C00u);           // inf
    }
    eh = (int)exp - 127 + 15;
    if (eh >= 31) return (uint16_t)(sign | 0x7C00u);
    if (eh <= 0) {
        uint32_t m24, q;
        int shift;
        if (eh < -10) return sign; // меньше половины минимальной субнормали
        m24 = mant | 0x00800000u;
        shift = 14 - eh;
        q = round_shift(m24, shift);
        if (q >= 0x0400u) return (uint16_t)(sign | 0x0400u);
        return (uint16_t)(sign | (uint16_t)q);
    }
    {
        uint32_t q = round_shift(mant, 13);
        if (q == 0x0400u) {
            q = 0;
            eh++;
            if (eh >= 31) return (uint16_t)(sign | 0x7C00u);
        }
        return (uint16_t)(sign | ((uint16_t)eh << 10) | (uint16_t)q);
    }
}

// --- bfloat16 --------------------------------------------------------------

float vtm_bf16_to_float(uint16_t b) {
    uint32_t bits = (uint32_t)b << 16;
    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

uint16_t vtm_float_to_bf16(float f) {
    uint32_t x, lsb;
    memcpy(&x, &f, sizeof x);
    // Округление к ближайшему чётному по младшим 16 битам.
    lsb = (x >> 16) & 1u;
    x += 0x7FFFu + lsb;
    return (uint16_t)(x >> 16);
}
