/*
 * fuzz_interp.c — внутрипроцессный фаззер загрузчика/интерпретатора VTM.
 *
 * Задача 0069. В отличие от tools/fuzz_vtm.py, который гоняет CLI в отдельном
 * процессе, этот harness мутирует контейнер в памяти и вызывает
 * vtm_load_mem() + vtm_run() напрямую — так под ASan/UBSan видно не только
 * падение, но и переполнения/UB внутри библиотеки. Один и тот же
 * детерминированный набор мутаций можно воспроизвести: перед каждой загрузкой
 * текущий случай пишется в fuzz-findings/last_interp.vtm и .json.
 *
 * Сборка (санитайзеры — вручную, Makefile не трогаем):
 *   cd src/vtm
 *   make clean && make test \
 *     CFLAGS="-std=c99 -O1 -g -Wall -Wextra -Werror \
 *             -fsanitize=address,undefined -fno-omit-frame-pointer" \
 *     CXXFLAGS="-std=c++17 -O1 -g -fsanitize=address,undefined \
 *               -fno-omit-frame-pointer"
 *   cc -std=c99 -O1 -g -Wall -Wextra -Werror \
 *      -fsanitize=address,undefined -fno-omit-frame-pointer \
 *      -I. -I../.. -o fuzz_interp ../../tests/vtm/fuzz_interp.c libvtm.a -lm
 *
 * Запуск:
 *   ./fuzz_interp ../../demo/model.vtm 5000 1
 *   (аргументы: модель, итераций, seed; по умолчанию 5000 и 1)
 */
#define _POSIX_C_SOURCE 200809L

#include "vtm.h"

#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define HEADER_SIZE       32u
#define TENSOR_ENTRY_SIZE 52u
#define INSTR_SIZE        16u
#define DATA_ALIGN        32u
#define MAX_TRT1          256

typedef struct {
    const char *name;
    size_t start;
    size_t end;
    unsigned weight;
} fuzz_section;

typedef struct {
    size_t offset;
    unsigned size;
    uint32_t old_value;
    uint32_t new_value;
    const char *label;
} fuzz_mutation;

static uint64_t g_rng;
static volatile sig_atomic_t g_iteration;

static uint64_t rnd64(void) {
    uint64_t x = g_rng;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    g_rng = x;
    return x;
}

static uint32_t rnd32(void) {
    return (uint32_t)(rnd64() >> 32);
}

static size_t rnd_range(size_t n) {
    return n ? (size_t)(rnd64() % (uint64_t)n) : 0;
}

static uint32_t pick_u32(const uint32_t *values, size_t n) {
    return values[rnd_range(n)];
}

static uint64_t pick_u64(const uint64_t *values, size_t n) {
    return values[rnd_range(n)];
}

static void put_u32(uint8_t *buf, size_t off, uint32_t value) {
    memcpy(buf + off, &value, sizeof value);
}

static uint32_t get_u32(const uint8_t *buf, size_t off) {
    uint32_t value;
    memcpy(&value, buf + off, sizeof value);
    return value;
}

static void on_alarm(int sig) {
    static const char msg[] =
        "fuzz_interp: таймаут 10 с, текущий случай — "
        "fuzz-findings/last_interp.vtm/.json\n";
    (void)sig;
    (void)write(STDERR_FILENO, msg, sizeof msg - 1u);
    _exit(124);
}

static int read_file(const char *path, uint8_t **out, size_t *size) {
    FILE *f = fopen(path, "rb");
    long length;
    uint8_t *buf;

    if (!f) {
        perror(path);
        return -1;
    }
    if (fseek(f, 0, SEEK_END) != 0 || (length = ftell(f)) < 0 ||
        fseek(f, 0, SEEK_SET) != 0) {
        fprintf(stderr, "%s: не удалось определить размер\n", path);
        fclose(f);
        return -1;
    }
    buf = (uint8_t *)malloc((size_t)length ? (size_t)length : 1u);
    if (!buf) {
        fprintf(stderr, "%s: не хватило памяти\n", path);
        fclose(f);
        return -1;
    }
    if (length > 0 && fread(buf, 1, (size_t)length, f) != (size_t)length) {
        fprintf(stderr, "%s: short read\n", path);
        free(buf);
        fclose(f);
        return -1;
    }
    fclose(f);
    *out = buf;
    *size = (size_t)length;
    return 0;
}

static size_t align_up(size_t value) {
    return (value + DATA_ALIGN - 1u) / DATA_ALIGN * DATA_ALIGN;
}

static int parse_sections(const uint8_t *base, size_t size,
                          fuzz_section *sections, size_t *n_sections,
                          size_t *trt1, size_t *n_trt1,
                          uint32_t *n_tensors, uint32_t *n_instr) {
    uint32_t local_tensors, local_instr;
    size_t table_end, data_start, program_start, data_end;
    uint32_t meta_off, meta_len;
    size_t pos;

    if (size < HEADER_SIZE || memcmp(base, "VTM1", 4) != 0) {
        fprintf(stderr, "не VTM1 или файл короче заголовка\n");
        return -1;
    }
    memcpy(&local_tensors, base + 12, 4);
    memcpy(&local_instr, base + 8, 4);
    memcpy(&meta_off, base + 24, 4);
    memcpy(&meta_len, base + 28, 4);
    if ((uint64_t)local_tensors * TENSOR_ENTRY_SIZE >
        (uint64_t)(size - HEADER_SIZE)) {
        fprintf(stderr, "таблица тензоров не влезает в файл\n");
        return -1;
    }
    table_end = HEADER_SIZE + (size_t)local_tensors * TENSOR_ENTRY_SIZE;
    data_start = align_up(table_end);
    if (data_start > size) {
        fprintf(stderr, "начало данных за файлом\n");
        return -1;
    }
    if ((uint64_t)local_instr * INSTR_SIZE > (uint64_t)(size - data_start)) {
        fprintf(stderr, "программа не влезает в файл\n");
        return -1;
    }
    program_start = size - (size_t)local_instr * INSTR_SIZE;
    data_end = program_start;
    if (meta_off != 0 && meta_len != 0 &&
        (uint64_t)meta_off >= (uint64_t)data_start &&
        (uint64_t)meta_off + (uint64_t)meta_len <= (uint64_t)program_start) {
        data_end = meta_off;
    }

    sections[0] = (fuzz_section){"header", 0, HEADER_SIZE, 18};
    sections[1] = (fuzz_section){"table", HEADER_SIZE, table_end, 22};
    sections[2] = (fuzz_section){"data", data_start, data_end, 25};
    sections[3] = (fuzz_section){"program", program_start, size, 30};
    sections[4] = (fuzz_section){"meta", 0, 0, 5};
    if (data_end != program_start) {
        sections[4].start = data_end;
        sections[4].end = program_start;
    }
    *n_sections = 5;

    *n_trt1 = 0;
    for (pos = data_start; pos + 18 <= data_end; pos++) {
        if ((pos - data_start) % DATA_ALIGN != 0) continue;
        if (memcmp(base + pos, "TRT1", 4) != 0) continue;
        if (*n_trt1 < MAX_TRT1) trt1[(*n_trt1)++] = pos;
    }
    *n_tensors = local_tensors;
    *n_instr = local_instr;
    return 0;
}

static fuzz_section pick_section(const fuzz_section *sections, size_t n) {
    unsigned total = 0, roll;
    size_t i;

    for (i = 0; i < n; i++) total += sections[i].weight;
    roll = rnd_range(total);
    for (i = 0; i < n; i++) {
        if (roll < sections[i].weight) return sections[i];
        roll -= sections[i].weight;
    }
    return sections[n - 1u];
}

static void write_mutation(uint8_t *buf, size_t size, fuzz_mutation *journal,
                           size_t *n_journal, size_t offset, unsigned width,
                           uint32_t value, const char *label) {
    uint32_t old_value;
    if (offset + width > size) return;
    if (width == 1u) {
        old_value = buf[offset];
    } else if (width == 2u) {
        old_value = (uint32_t)buf[offset] | ((uint32_t)buf[offset + 1u] << 8);
    } else {
        old_value = get_u32(buf, offset);
    }
    journal[*n_journal].offset = offset;
    journal[*n_journal].size = width;
    journal[*n_journal].old_value = old_value;
    journal[*n_journal].new_value = value;
    journal[*n_journal].label = label;
    (*n_journal)++;
    if (width == 1u) {
        buf[offset] = (uint8_t)value;
    } else {
        put_u32(buf, offset, value);
    }
}

static void mutate_header(uint8_t *buf, size_t size, fuzz_mutation *journal,
                          size_t *n_journal, const uint8_t *base) {
    static const uint32_t edges[] = {
        0u, 1u, 2u, 3u, 4u, 16u, 32u, 52u, 128u, 256u,
        0x7FFFFFFFu, 0x80000000u, 0xFFFFFFFEu, 0xFFFFFFFFu
    };
    static const struct { size_t off; unsigned width; } fields[] = {
        {4, 2}, {6, 2}, {8, 4}, {12, 4}, {16, 4}, {20, 4}, {24, 4}, {28, 4}
    };
    size_t field = rnd_range(sizeof fields / sizeof fields[0]);
    uint32_t current = (fields[field].width == 2u)
                           ? (uint32_t)(base[fields[field].off] |
                                        (base[fields[field].off + 1] << 8))
                           : get_u32(base, fields[field].off);
    uint32_t value;
    if (rnd_range(10) < 7) {
        value = pick_u32(edges, sizeof edges / sizeof edges[0]);
        if (rnd_range(4) == 0) value = current + 1u;
        if (rnd_range(4) == 0) value = current - 1u;
    } else {
        value = rnd32();
    }
    if (fields[field].width == 2u) value &= 0xFFFFu;
    write_mutation(buf, size, journal, n_journal, fields[field].off,
                   fields[field].width, value, "header.field");
}

static void mutate_table(uint8_t *buf, size_t size, fuzz_mutation *journal,
                         size_t *n_journal, uint32_t n_tensors,
                         const uint8_t *base) {
    static const struct { size_t off; unsigned width; const char *label; }
        fields[] = {
            {0, 1, "table.name"},     {24, 1, "table.kind"},
            {25, 1, "table.ndim"},    {26, 4, "table.dim0"},
            {30, 4, "table.dim1"},    {34, 4, "table.dim2"},
            {38, 4, "table.dim3"},    {42, 4, "table.offset"},
            {46, 4, "table.scale_offset"}, {50, 2, "table.reserved"}
        };
    static const uint32_t edges[] = {
        0u, 1u, 2u, 3u, 4u, 8u, 16u, 32u, 64u, 128u, 256u,
        0x7FFFFFFFu, 0x80000000u, 0xFFFFFFE0u, 0xFFFFFFFEu, 0xFFFFFFFFu
    };
    size_t entry, field;
    size_t off;
    uint32_t value;

    if (!n_tensors) return;
    entry = rnd_range(n_tensors);
    field = rnd_range(sizeof fields / sizeof fields[0]);
    off = HEADER_SIZE + entry * TENSOR_ENTRY_SIZE + fields[field].off;
    value = pick_u32(edges, sizeof edges / sizeof edges[0]);
    if (rnd_range(5) == 0) value = rnd32();
    if (fields[field].width == 2u) value &= 0xFFFFu;
    if (rnd_range(3) == 0) {
        uint32_t current = (fields[field].width == 1u) ? base[off]
                           : get_u32(base, off);
        value = (rnd_range(2) ? current + 1u : current - 1u);
        if (fields[field].width == 2u) value &= 0xFFFFu;
    }
    write_mutation(buf, size, journal, n_journal, off, fields[field].width,
                   value, fields[field].label);
}

static void mutate_program(uint8_t *buf, size_t size, fuzz_mutation *journal,
                           size_t *n_journal, uint32_t n_tensors,
                           uint32_t n_instr, size_t program_start) {
    static const struct { size_t off; unsigned width; const char *label; }
        fields[] = {
            {0, 1, "instr.opcode"}, {1, 1, "instr.flags"},
            {2, 2, "instr.dst"},    {4, 2, "instr.a"},
            {6, 2, "instr.b"},      {8, 2, "instr.c"},
            {12, 4, "instr.imm"}
        };
    static const uint32_t edges[] = {
        0u, 1u, 2u, 3u, 4u, 5u, 8u, 12u, 16u, 127u, 128u, 255u, 256u,
        0x7FFFu, 0x8000u, 0xFFFFu
    };
    size_t instr, field;
    size_t off;
    uint32_t value;

    if (!n_instr) return;
    instr = rnd_range(n_instr);
    field = rnd_range(sizeof fields / sizeof fields[0]);
    off = program_start + instr * INSTR_SIZE + fields[field].off;
    value = pick_u32(edges, sizeof edges / sizeof edges[0]);
    if (strcmp(fields[field].label, "instr.imm") == 0) {
        /* imm: знаковые и «позиции» */
        static const uint32_t imm_edges[] = {
            0u, 1u, 2u, 3u, 4u, 7u, 8u, 16u, 32u, 64u, 100u, 128u, 256u,
            1000u, 0x7FFFFFFFu, 0x80000000u, 0xFFFFFFFFu
        };
        value = pick_u32(imm_edges, sizeof imm_edges / sizeof imm_edges[0]);
        if (rnd_range(4) == 0) value = rnd32();
    } else if (fields[field].width == 2u) {
        /* индексы тензоров: чаще держим их в таблице, иногда — за ней */
        if (rnd_range(3) != 0 && n_tensors) value = rnd_range(n_tensors);
        if (rnd_range(4) == 0) value = n_tensors + rnd_range(3);
        value &= 0xFFFFu;
    } else if (fields[field].width == 1u) {
        value = pick_u32(edges, sizeof edges / sizeof edges[0]);
        if (rnd_range(3) == 0) value = rnd_range(256);
    }
    write_mutation(buf, size, journal, n_journal, off, fields[field].width,
                   value, fields[field].label);
}

static void mutate_data(uint8_t *buf, size_t size, fuzz_mutation *journal,
                        size_t *n_journal, const fuzz_section *data,
                        const size_t *trt1, size_t n_trt1,
                        const uint8_t *base) {
    static const uint32_t f32_edges[] = {
        0x00000000u, 0x80000000u, 0x3F800000u, 0xBF800000u,
        0x7F800000u, 0xFF800000u, 0x7FC00000u, 0x7F7FFFFFu,
        0x00000001u, 0xFFFFFFFFu
    };
    size_t off, width;
    uint32_t value;

    if (data->end <= data->start) return;
    if (n_trt1 && rnd_range(100) < 30) {
        size_t seg = trt1[rnd_range(n_trt1)];
        uint64_t n_trits;
        uint32_t n_scales;
        memcpy(&n_trits, base + seg + 6, 8);
        memcpy(&n_scales, base + seg + 14, 4);
        switch (rnd_range(5)) {
        case 0:
            write_mutation(buf, size, journal, n_journal, seg + rnd_range(4),
                           1, (uint32_t)rnd_range(256), "trt1.magic");
            return;
        case 1:
            write_mutation(buf, size, journal, n_journal, seg + 4, 1,
                           (uint32_t)rnd_range(4), "trt1.version");
            return;
        case 2:
            write_mutation(buf, size, journal, n_journal, seg + 5, 1,
                           (uint32_t)rnd_range(4), "trt1.flags");
            return;
        case 3: {
            static const uint64_t delta[] = {0u, 1u, 159u, 160u, 320u,
                                             0xFFFFFFFFFFFFFFFFull};
            uint64_t v = n_trits + pick_u64(delta,
                                            sizeof delta / sizeof delta[0]);
            if (rnd_range(4) == 0) v = 0u;
            memcpy(buf + seg + 6, &v, 8);
            journal[*n_journal].offset = seg + 6;
            journal[*n_journal].size = 8;
            journal[*n_journal].old_value = (uint32_t)n_trits;
            journal[*n_journal].new_value = (uint32_t)v;
            journal[*n_journal].label = "trt1.n_trits";
            (*n_journal)++;
            return;
        }
        default: {
            static const uint32_t scale_edges[] = {0u, 1u, 2u, 3u,
                                                   0xFFFFFFFFu};
            uint32_t v = n_scales;
            if (rnd_range(2)) v += 1u;
            else v = (v ? v - 1u : 0u);
            if (rnd_range(3) == 0)
                v = pick_u32(scale_edges, sizeof scale_edges /
                                              sizeof scale_edges[0]);
            write_mutation(buf, size, journal, n_journal, seg + 14, 4, v,
                           "trt1.n_scales");
            return;
        }
        }
    }
    off = data->start + rnd_range(data->end - data->start);
    if (rnd_range(100) < 35 && off + 4 <= data->end) {
        width = 4;
        value = pick_u32(f32_edges, sizeof f32_edges / sizeof f32_edges[0]);
    } else {
        width = 1;
        value = rnd_range(256);
    }
    write_mutation(buf, size, journal, n_journal, off, width, value,
                   "data.byte");
}

static void mutate_meta(uint8_t *buf, size_t size, fuzz_mutation *journal,
                        size_t *n_journal, const fuzz_section *meta) {
    static const char json_edges[] = "{}[]\",:0123456789-.eE\\";
    size_t off;
    uint32_t value;

    if (meta->end <= meta->start) return;
    off = meta->start + rnd_range(meta->end - meta->start);
    if (rnd_range(2)) {
        value = (uint32_t)(unsigned char)
            json_edges[rnd_range(sizeof json_edges - 1u)];
    } else {
        value = rnd_range(256);
    }
    write_mutation(buf, size, journal, n_journal, off, 1, value, "meta.byte");
}

static void save_last(const uint8_t *buf, size_t size,
                      const fuzz_mutation *journal, size_t n_journal,
                      uint64_t iteration, uint64_t seed,
                      const fuzz_section *section) {
    FILE *f;
    size_t i;

    f = fopen("fuzz-findings/last_interp.vtm", "wb");
    if (f) {
        if (fwrite(buf, 1, size, f) != size) {
            fprintf(stderr, "не удалось записать last_interp.vtm\n");
        }
        fclose(f);
    }
    f = fopen("fuzz-findings/last_interp.json", "w");
    if (!f) return;
    fprintf(f, "{\n  \"iteration\": %llu,\n  \"seed\": %llu,\n"
               "  \"section\": \"%s\",\n  \"mutations\": [\n",
            (unsigned long long)iteration, (unsigned long long)seed,
            section->name);
    for (i = 0; i < n_journal; i++) {
        fprintf(f, "    {\"offset\": %zu, \"size\": %u, "
                   "\"old\": \"0x%08x\", \"new\": \"0x%08x\", "
                   "\"label\": \"%s\"}%s\n",
                journal[i].offset, journal[i].size,
                journal[i].old_value, journal[i].new_value,
                journal[i].label, (i + 1u < n_journal) ? "," : "");
    }
    fprintf(f, "  ]\n}\n");
    fclose(f);
}

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1] : "../../demo/model.vtm";
    uint64_t iterations = (argc > 2) ? strtoull(argv[2], NULL, 10) : 5000u;
    uint64_t seed = (argc > 3) ? strtoull(argv[3], NULL, 10) : 1u;
    uint8_t *base = NULL, *work = NULL;
    size_t size = 0;
    fuzz_section sections[5];
    size_t n_sections = 0, trt1[MAX_TRT1], n_trt1 = 0;
    uint32_t n_tensors = 0, n_instr = 0;
    uint64_t iteration;
    uint64_t load_ok = 0, load_fail = 0, run_ok = 0, run_err = 0;
    size_t max_journal = 8;
    fuzz_mutation *journal;
    struct timespec t0, t1;
    double elapsed;

    if (read_file(path, &base, &size) != 0) return 2;
    if (parse_sections(base, size, sections, &n_sections, trt1, &n_trt1,
                       &n_tensors, &n_instr) != 0) {
        free(base);
        return 2;
    }
    work = (uint8_t *)malloc(size ? size : 1u);
    journal = (fuzz_mutation *)calloc(max_journal, sizeof *journal);
    if (!work || !journal) {
        fprintf(stderr, "не хватило памяти\n");
        free(base);
        free(work);
        free(journal);
        return 2;
    }
    if (mkdir("fuzz-findings", 0777) != 0 && access("fuzz-findings", W_OK) != 0) {
        perror("fuzz-findings");
        free(base);
        free(work);
        free(journal);
        return 2;
    }
    signal(SIGALRM, on_alarm);
    g_rng = seed ? seed : 0x9E3779B97F4A7C15ull;

    printf("fuzz_interp: %s (%zu байт), тензоров %u, инструкций %u\n",
           path, size, n_tensors, n_instr);
    printf("  секции:");
    {
        size_t i;
        for (i = 0; i < n_sections; i++) {
            if (sections[i].end > sections[i].start)
                printf(" %s[0x%zx..0x%zx]", sections[i].name,
                       sections[i].start, sections[i].end);
        }
        printf("\n");
    }
    printf("  TRT1-сегментов: %zu; итераций: %llu; seed: %llu\n",
           n_trt1, (unsigned long long)iterations,
           (unsigned long long)seed);

    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (iteration = 1; iteration <= iterations; iteration++) {
        size_t n_journal = 0;
        fuzz_section section;
        unsigned count = (rnd_range(100) < 72) ? 1u
                       : (rnd_range(100) < 86) ? 2u : 3u;
        unsigned k;
        vtm_status st = VTM_OK;
        char err[512];
        vtm_t *m;

        g_iteration = (sig_atomic_t)iteration;
        memcpy(work, base, size);
        section = pick_section(sections, n_sections);
        for (k = 0; k < count; k++) {
            fuzz_section pick = pick_section(sections, n_sections);
            if (strcmp(pick.name, "header") == 0) {
                mutate_header(work, size, journal, &n_journal, base);
            } else if (strcmp(pick.name, "table") == 0) {
                mutate_table(work, size, journal, &n_journal, n_tensors, base);
            } else if (strcmp(pick.name, "program") == 0) {
                mutate_program(work, size, journal, &n_journal, n_tensors,
                               n_instr, sections[3].start);
            } else if (strcmp(pick.name, "data") == 0) {
                mutate_data(work, size, journal, &n_journal, &sections[2],
                            trt1, n_trt1, base);
            } else {
                mutate_meta(work, size, journal, &n_journal, &sections[4]);
            }
            section = pick;
        }
        save_last(work, size, journal, n_journal, iteration, seed, &section);

        alarm(10);
        m = vtm_load_mem(work, size, &st, err, sizeof err);
        if (!m) {
            load_fail++;
        } else {
            load_ok++;
            vtm_set_token(m, 1);
            vtm_set_position(m, 0);
            st = vtm_run(m, NULL, NULL);
            if (st == VTM_OK) run_ok++;
            else run_err++;
            vtm_free(m);
        }
        alarm(0);

        if (iteration % 1000u == 0u) {
            printf("  [%llu/%llu]\n", (unsigned long long)iteration,
                   (unsigned long long)iterations);
            fflush(stdout);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    elapsed = (double)(t1.tv_sec - t0.tv_sec) +
              (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;
    printf("итог: %llu итераций за %.2f с\n", (unsigned long long)iterations,
           elapsed);
    printf("  загрузка: ok=%llu, ошибок=%llu; run: ok=%llu, ошибок=%llu\n",
           (unsigned long long)load_ok, (unsigned long long)load_fail,
           (unsigned long long)run_ok, (unsigned long long)run_err);
    printf("  последний случай: fuzz-findings/last_interp.vtm "
           "(итерация %d)\n", (int)g_iteration);
    free(base);
    free(work);
    free(journal);
    return 0;
}
