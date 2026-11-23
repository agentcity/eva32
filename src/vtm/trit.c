// Кодек тритов и преобразования float16/bfloat16.
//
// Формат v0.1 (docs/trit-format.md): слово u32 little-endian несёт 20 тритов,
// цифра -1 = 0, 0 = 1, +1 = 2, разряды младший-вперёд: word = Σ d_i·3^i.
// Блок — 8 слов = 160 тритов — имеет один общий масштаб.
//
// Сменное хранение (задача 0057): кроме pack20 (20 тритов в u32) бывает
// native — 1 знаковый байт на трит. Контракт vtm_trit_store (trit.h) прячет
// разницу от загрузчика; ядра интерпретатора по-прежнему получают pack20.
#include "vtm.h"
#include "trit.h"

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

// --- сменное хранение тритов (задача 0057) ---------------------------------

uint32_t vtm_trit_storage_words(uint32_t n_trits, unsigned storage) {
    if (storage == VTM_TRIT_STORAGE_PACKED20) {
        uint32_t blocks = (n_trits + VTM_TRITS_PER_BLOCK - 1u) / VTM_TRITS_PER_BLOCK;
        return blocks * VTM_WORDS_PER_BLOCK;
    }
    if (storage == VTM_TRIT_STORAGE_NATIVE) {
        return (n_trits + VTM_NATIVE_TRITS_PER_WORD - 1u) / VTM_NATIVE_TRITS_PER_WORD;
    }
    return 0;
}

uint32_t vtm_trit_storage_bytes(uint32_t n_trits, unsigned storage) {
    uint32_t words = vtm_trit_storage_words(n_trits, storage);
    // Верхняя граница n_trits — VTM_MAX_ELEMENTS (2^30), поэтому words·4 < 2^32.
    return words * 4u;
}

unsigned vtm_trit_storage_from_flags(uint16_t flags) {
    return (unsigned)((flags & VTM_TENSOR_FLAG_STORAGE_MASK)
                      >> VTM_TENSOR_FLAG_STORAGE_SHIFT);
}

uint16_t vtm_trit_storage_entry_flags(unsigned storage) {
    return (uint16_t)(((storage << VTM_TENSOR_FLAG_STORAGE_SHIFT)
                       & VTM_TENSOR_FLAG_STORAGE_MASK));
}

int vtm_trit_store_init(vtm_trit_store *store, unsigned storage, const void *data,
                        uint32_t n_units, uint32_t n_trits,
                        const float *scales, uint32_t n_scales,
                        uint32_t scale_block) {
    uint32_t need, expected_scales;
    if (!store) return -1;
    memset(store, 0, sizeof *store);
    if (storage > VTM_TRIT_STORAGE_MAX) return -1;
    if (n_trits != 0 && !data) return -1;
    need = vtm_trit_storage_words(n_trits, storage);
    if (n_units < need) return -1;
    if (scales != NULL && n_scales != 0) {
        if (scale_block == 0) return -1;
        expected_scales = (n_trits + scale_block - 1u) / scale_block;
        if (n_scales != expected_scales) return -1;
    } else {
        n_scales = 0;
        scales = NULL;
    }
    store->storage = storage;
    store->n_trits = n_trits;
    store->n_words = need;
    store->words = data;
    store->scales = scales;
    store->n_scales = n_scales;
    store->scale_block = scale_block;
    return 0;
}

int vtm_trit_store_fetch(const vtm_trit_store *store, uint32_t index,
                         int8_t *trit) {
    if (!store || !trit || index >= store->n_trits) return -1;
    if (store->storage == VTM_TRIT_STORAGE_PACKED20) {
        const uint32_t *words = (const uint32_t *)store->words;
        uint32_t word, shift;
        if (!words) return -1;
        word = words[index / VTM_TRITS_PER_WORD];
        for (shift = 0; shift < index % VTM_TRITS_PER_WORD; shift++)
            word /= 3u;
        word %= 3u;
        *trit = (int8_t)(word == 2u ? 1 : (word == 0u ? -1 : 0));
        return 0;
    }
    if (store->storage == VTM_TRIT_STORAGE_NATIVE) {
        const int8_t *bytes = (const int8_t *)store->words;
        int8_t value;
        if (!bytes) return -1;
        value = bytes[index];
        if (value < -1 || value > 1) return -1;
        *trit = value;
        return 0;
    }
    return -1;
}

int vtm_trit_store_count(const vtm_trit_store *store) {
    if (!store) return 0;
    return (int)store->n_trits;
}

int vtm_trit_store_scale_of(const vtm_trit_store *store, uint32_t index,
                            float *scale) {
    uint32_t block;
    if (!store || !scale || index >= store->n_trits) return -1;
    if (!store->scales || store->scale_block == 0 || store->n_scales == 0)
        return -1;
    block = index / store->scale_block;
    if (block >= store->n_scales) return -1;
    *scale = store->scales[block];
    return 0;
}

int vtm_trit_store_to_packed20(const vtm_trit_store *store, uint32_t *out_words,
                               uint32_t out_words_cap) {
    uint32_t need, base, i;
    if (!store || !out_words) return -1;
    need = vtm_trit_storage_words(store->n_trits, VTM_TRIT_STORAGE_PACKED20);
    if (out_words_cap < need) return -1;
    if (need) memset(out_words, 0, (size_t)need * 4u);
    for (base = 0; base < store->n_trits; base += VTM_TRITS_PER_WORD) {
        int8_t block[VTM_TRITS_PER_WORD];
        memset(block, 0, sizeof block);
        for (i = 0; i < VTM_TRITS_PER_WORD && base + i < store->n_trits; i++) {
            if (vtm_trit_store_fetch(store, base + i, &block[i]) != 0) return -1;
        }
        out_words[base / VTM_TRITS_PER_WORD] = vtm_trit_pack20(block);
    }
    return 0;
}

int vtm_trit_validate_native(const uint8_t *bytes, uint32_t n_bytes,
                             uint32_t *bad_index) {
    uint32_t i;
    if (!bytes && n_bytes) return -1;
    for (i = 0; i < n_bytes; i++) {
        int8_t value = (int8_t)bytes[i];
        if (value < -1 || value > 1) {
            if (bad_index) *bad_index = i;
            return -1;
        }
    }
    return 0;
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
