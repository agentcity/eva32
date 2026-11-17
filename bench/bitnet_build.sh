#!/usr/bin/env bash
# Сборка bitnet.cpp (CPU, без GPU/денег) и замер на M1 Pro.
# Всё делается в /tmp, репозиторий не засоряется. Проверено 2026-11-18.
set -euo pipefail

WORK=${WORK:-/tmp/bitnet}
PY=${PY:-/Users/a/.dsh/dsh-runtimes/dsh-primary-runtime/dependencies/python/bin/python3}

# 1. cmake без установки в систему: pip --target + обёртка
"$PY" -m pip install -q --target /tmp/cmakepkg cmake
cat > /tmp/cmake.sh <<'EOF'
#!/bin/bash
export PYTHONPATH=/tmp/cmakepkg
exec /Users/a/.dsh/dsh-runtimes/dsh-primary-runtime/dependencies/python/bin/python3 \
     /tmp/cmakepkg/bin/cmake "$@"
EOF
chmod +x /tmp/cmake.sh

# 2. Клон и сабмодуль (без него падает get_target_property "ggml")
GIT_TERMINAL_PROMPT=0 git clone -q --depth 1 https://github.com/microsoft/BitNet "$WORK"
cd "$WORK"
GIT_TERMINAL_PROMPT=0 git submodule update --init --depth 1 3rdparty/llama.cpp

# 3. Сборка. Ключевые флаги:
#    BUILD_SHARED_LIBS=OFF — в общем режиме линковка падает на i2_s символах
#    LLAMA_BUILD_COMMON/EXAMPLES=ON — иначе не собирается ни один CLI
/tmp/cmake.sh -B build -DCMAKE_BUILD_TYPE=Release \
  -DGGML_BITNET_ARM_TL1=ON -DBUILD_SHARED_LIBS=OFF \
  -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TESTS=OFF
/tmp/cmake.sh --build build -j 8 --target llama-simple

# 4. Модель (1,1 ГБ)
curl -sS -L -o /tmp/bitnet-2b-i2s.gguf \
  "https://huggingface.co/microsoft/BitNet-b1.58-2B-4T-gguf/resolve/main/ggml-model-i2_s.gguf"

# 5. Замер. -ngl 0 обязателен: Metal-путь в форке не имеет ядра kernel_mul_mm_i2_s
./build/bin/llama-simple -m /tmp/bitnet-2b-i2s.gguf -ngl 0 -t 8 \
  -p "Сколько стоит тариф Старт?" -n 48 2>&1 | grep -E "eval time|load time"
