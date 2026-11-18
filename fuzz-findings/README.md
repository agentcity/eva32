# fuzz-findings

Результаты задачи 0069 (фаззинг VTM). Все смещения — абсолютные, от начала
`.vtm`; файлы-мутации побайтово сравнимы с `demo/model.vtm`.

| Файл | Что это |
|---|---|
| `findings.json` | манифест Python-фаззера, seed 5: 1 находка (SIGBUS) |
| `finding-0001-SIGBUS-02524.*` | случай Python-фаззера, итерация 2524: SIGBUS, sha256 `dcd4ae17…` |
| `minimal-ATTN-imm-INT_MAX.vtm/.json` | минимальный repro: только 4 байта в `instr[8].imm` = INT_MAX, sha256 `08d0b55d…` |
| `interp-seed2-crash.vtm/.json` | падение `tests/vtm/fuzz_interp.c`, seed 2, итерация 4165 (тот же дефект) |
| `last_interp.vtm/.json` | последний случай in-process фаззера (seed 1, чистый прогон) |
| `seed1-5000/` | Python-фаззер, 5000 мутаций, seed 1: 0 сигналов, 0 таймаутов |
| `sanitizer/` | логи ASan+UBSan: `make test`, demo (default и strict), repro, `fuzz_interp` |

## Воспроизведение

```sh
# обычная сборка: Bus error (exit 138)
./src/vtm/vtm run fuzz-findings/minimal-ATTN-imm-INT_MAX.vtm --tokens 1 --max-new 1

# ASan+UBSan: signed integer overflow + allocation-size-too-big
cd src/vtm
make clean && make test \
  CFLAGS="-std=c99 -O1 -g -Wall -Wextra -Werror \
          -fsanitize=address,undefined -fno-omit-frame-pointer" \
  CXXFLAGS="-std=c++17 -O1 -g -Wall -Wextra -Wpedantic -Werror \
            -fsanitize=address,undefined -fno-omit-frame-pointer"
./vtm run /абсолютный/путь/fuzz-findings/minimal-ATTN-imm-INT_MAX.vtm \
    --tokens 1 --max-new 1
```

Причина: `interp.c`, `op_attn`, 2-D K/V-кэш: `q_start = position + imm`
(INT_MAX), затем `key_count = q_start + q_len` — переполнение `int`; отрицательный
`key_count` проходит проверку `key_count > ctx` и уходит в
`malloc((size_t)key_count * sizeof(double))`. Нормальная сборка падает с
SIGBUS; ASan сообщает `allocation-size-too-big`. Исправление — в
`src/vtm/interp.c` (задача архитектора), здесь код не менялся.
