# VTM: скалярный интерпретатор. Внешних зависимостей нет, кроме libm.
# CLI (main.c) — C99; режим --text линкуется с BPE-токенизатором (C++17),
# поэтому tok_bridge.o собирается из того же main.c с -DVTM_TOK_BRIDGE,
# а компилятор и линковщик для него — $(CXX).
CC       ?= cc
CXX      ?= c++
AR       ?= ar
# Clang умеет -Wtautological-*: они ловят сравнения, которые всегда истинны
# (на gcc это -Wtype-limits). Ровно такой баг валил CI на Ubuntu: сравнение
# `opcode < 256` при uint8_t. Включаем их локально, чтобы не узнавать из CI.
CC_IS_CLANG := $(shell $(CC) --version 2>/dev/null | grep -qi clang && echo 1 || echo 0)
ifeq ($(CC_IS_CLANG),1)
CFLAGS   ?= -std=c99 -O2 -Wall -Wextra -Wtautological-compare -Wtautological-constant-out-of-range-compare -Werror
else
CFLAGS   ?= -std=c99 -O2 -Wall -Wextra -Werror
endif
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror
CPPFLAGS ?= -D_POSIX_C_SOURCE=200809L
LDLIBS   ?= -lm
# LDFLAGS — флаги линковщика отдельно от LDLIBS (библиотек): через них
# `make ci` подставляет санитайзер, не трогая обычную сборку.
LDFLAGS  ?=
# Пул потоков ядер (backend_scalar.c) — pthread; на macOS он в libSystem,
# на Linux нужен -lpthread. Отдельная переменная: LDLIBS задают снаружи.
THREAD_LIBS ?= -lpthread

LIB      = libvtm.a
BIN      = vtm
# SIMD-ядра: backend_neon.c собирается всегда (на ARM — NEON, иначе только
# диспетчер с откатом на scalar). AVX2-ядра — отдельный TU с -mavx2, только
# на x86-64; WITH_AVX2=1 можно задать руками при кросс-сборке.
SRC      = trit.c loader.c sha256.c interp.c backend_scalar.c backend_neon.c api.c \
           kv_store.c
UNAME_M := $(shell uname -m)
WITH_AVX2 ?= $(if $(filter x86_64 amd64,$(UNAME_M)),1,0)
ifeq ($(WITH_AVX2),1)
SRC     += backend_avx2.c
else
CPPFLAGS += -DVTM_NO_AVX2   # backend_neon.c не будет ссылаться на AVX2-ядра
endif
OBJ      = $(SRC:.c=.o)
TOK_OBJ  = tokenizer.o
BRIDGE   = tok_bridge.o
TEST_SRC = ../../tests/vtm/test_vtm.c
TEST_BIN = test_vtm
TEST_SHA_SRC = ../../tests/vtm/test_sha256.c
TEST_SHA_BIN = test_sha256
TEST_API_SRC = ../../tests/test_vtm_api.c
TEST_API_BIN = test_vtm_api
TEST_SESSION_SRC = ../../tests/test_vtm_session.c
TEST_SESSION_BIN = test_vtm_session
BENCH_SRC = ../../tests/vtm/bench_backend.c
BENCH_BIN = bench_backend

.PHONY: all test test-all ci bench bench-backend bench-sha clean

all: $(BIN)

$(BIN): main.o $(LIB) $(BRIDGE) $(TOK_OBJ)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ main.o $(LIB) $(BRIDGE) $(TOK_OBJ) $(LDLIBS) $(THREAD_LIBS)

$(LIB): $(OBJ)
	$(AR) rcs $@ $(OBJ)

%.o: %.c vtm.h vtm_internal.h vtm_backend.h sha256.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -I. -c -o $@ $<

# api.c использует <stdatomic.h> (C11) для флага отмены и признака «занято».
# Остальные файлы остаются на C99; -std=c11 идёт после CFLAGS намеренно,
# чтобы перекрыть -std=c99, если его передали снаружи.
api.o: api.c api.h vtm.h vtm_backend.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -std=c11 -I. -c -o $@ $<

# Тот же main.c, но как C++17: C-ABI моста к tokenizer.hpp (см. main.c).
$(BRIDGE): main.c tokenizer.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -DVTM_TOK_BRIDGE -I. -x c++ -c -o $@ main.c

# tokenizer.o собирает Makefile.tok; здесь только зависимость и вызов.
$(TOK_OBJ): tokenizer.cpp tokenizer.hpp
	$(MAKE) -f Makefile.tok $(TOK_OBJ)

$(TEST_BIN): $(TEST_SRC) $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -I. -I../.. -o $@ $(TEST_SRC) $(LIB) \
		$(LDLIBS) $(THREAD_LIBS)

# SHA-256 отдельно от интерпретатора: векторы, аппаратный путь против
# портативного, откат и подмена байта в контейнере (задача 0078).
$(TEST_SHA_BIN): $(TEST_SHA_SRC) $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -I. -I../.. -o $@ $(TEST_SHA_SRC) $(LIB) \
		$(LDLIBS) $(THREAD_LIBS)

# C-ABI: колбэк стриминга и отмена из другого потока (pthread в тесте).
$(TEST_API_BIN): $(TEST_API_SRC) $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -I. -I../.. -o $@ $(TEST_API_SRC) $(LIB) \
		$(LDLIBS) $(THREAD_LIBS)

# Бенчмарк/проверка SIMD (tests/vtm/bench_backend.c). Собирается тем же CC,
# что и библиотека, поэтому видит тот же SIMD.
$(BENCH_BIN): $(BENCH_SRC) $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -I. -I../.. -o $@ $(BENCH_SRC) $(LIB) $(LDLIBS) $(THREAD_LIBS)

# Сессии KV (задача 0079): save/load/continue, обрыв и порча файла.
$(TEST_SESSION_BIN): $(TEST_SESSION_SRC) $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -I. -I../.. -o $@ $(TEST_SESSION_SRC) $(LIB) \
		$(LDLIBS) $(THREAD_LIBS)

# backend_avx2.c собирается без -mavx2: AVX2/FMA включены атрибутами только
# на функциях-ядрах, чтобы детект CPUID не исполнял AVX2-инструкции.

# $(BIN) нужен интеграционному тесту CLI (текст -> токены -> ответ).
# bench_backend --check — тест «simd == scalar» (1e-5) на наборе форм;
# bench_backend --threads — тест «результат не зависит от 1/2/4/8 потоков»
# (задача 0077), с малым числом итераций: печатает и время, но проверяет
# только расхождение.
# Интеграционные тесты ходят в demo/model.vtm, а он генерируется и в git не лежит.
# Правило делает сборку модели зависимостью теста: чистый клон больше не падает.
DEMO_MODEL = ../../demo/model.vtm

$(DEMO_MODEL):
	mkdir -p ../../demo
	python3 ../../scripts/demo_ternary_model.py --out ../../demo

test: $(TEST_BIN) $(TEST_API_BIN) $(TEST_SESSION_BIN) $(TEST_SHA_BIN) $(BIN) \
	$(BENCH_BIN) $(DEMO_MODEL)
	./$(TEST_BIN)
	./$(TEST_API_BIN)
	./$(TEST_SESSION_BIN)
	./$(TEST_SHA_BIN)
	./$(BENCH_BIN) --check
	./$(BENCH_BIN) --threads --iters 3

# Сборка токенизатора живёт отдельно (Makefile.tok), цели объединяются здесь:
# общего Makefile у трёх параллельных задач быть не могло.
test-all: test
	$(MAKE) -f Makefile.tok test-tok

# --- ci: обе проверки одной командой (задача 0102) ------------------------
# Что делает `make ci`:
#   1) обычная сборка и `make test` (как у разработчика);
#   2) тот же `make test` с -fsanitize=undefined: CFLAGS/CXXFLAGS/LDFLAGS
#      передаются переменными, поэтому обычная сборка их не видит.
# Между прогонами обязателен `clean`: объектные файлы не помнят свои флаги,
# и make иначе переиспользовал бы несанитизированные .o.
# Санитайзерный прогон собирается с -fno-sanitize-recover: любое срабатывание
# UBSan — это ненулевой код возврата, а не строчка в логе. В конце дерево
# чистится, чтобы после `make ci` не остался санитизированный бинарник.
UBSAN_FLAGS  ?= -fsanitize=undefined -fno-sanitize-recover=undefined \
                -fno-omit-frame-pointer
CI_CFLAGS    ?= $(CFLAGS) $(UBSAN_FLAGS)
CI_CXXFLAGS  ?= $(CXXFLAGS) $(UBSAN_FLAGS)
# На линковке нужен сам рантайм; остальные флаги — только для компиляции.
CI_LDFLAGS   ?= -fsanitize=undefined
UBSAN_OPTIONS ?= halt_on_error=1

ci:
	@set -e; \
	if [ -z '$(strip $(UBSAN_FLAGS))' ]; then \
	  echo "FAIL: UBSAN_FLAGS пуст — UBSan-прогон обязателен (задача 0102)"; \
	  exit 1; \
	fi; \
	t0=$$(date +%s); \
	echo "== ci[1/2]: обычная сборка + make test =="; \
	$(MAKE) clean >/dev/null; \
	if ! $(MAKE) test; then \
	  echo; echo "FAIL: обычная сборка или тесты"; exit 1; \
	fi; \
	t1=$$(date +%s); \
	echo; echo "OK: обычная сборка и make test — $$((t1 - t0))s"; \
	echo; \
	echo "== ci[2/2]: make test с UBSan ($(UBSAN_FLAGS)) =="; \
	$(MAKE) clean >/dev/null; \
	if ! UBSAN_OPTIONS=$(UBSAN_OPTIONS) $(MAKE) test \
	     CFLAGS='$(CI_CFLAGS)' CXXFLAGS='$(CI_CXXFLAGS)' \
	     LDFLAGS='$(CI_LDFLAGS)'; then \
	  echo; echo "FAIL: UBSan-прогон"; exit 1; \
	fi; \
	if command -v nm >/dev/null 2>&1; then \
	  if ! nm $(TEST_BIN) 2>/dev/null | grep -q ubsan; then \
	    echo; echo "FAIL: UBSan не подключился к $(TEST_BIN) — прогон не санитизирован"; \
	    exit 1; \
	  fi; \
	  echo "UBSan-рантайм в $(TEST_BIN): есть"; \
	fi; \
	t2=$$(date +%s); \
	$(MAKE) clean >/dev/null; \
	echo; echo "OK: UBSan-прогон — $$((t2 - t1))s"; \
	echo; \
	echo "== ci: OK — оба прогона зелёные, $$((t2 - t0))s =="

bench: $(BIN)
	./$(BIN) bench --rows 512 --cols 512 --iters 200

# GFLOP/s: scalar-lut243 / scalar-direct / SIMD + MM + RMSN.
bench-backend: $(BENCH_BIN)
	./$(BENCH_BIN)

# SHA-256: аппаратный и портативный путь в ГБ/с (задача 0078).
bench-sha: $(TEST_SHA_BIN)
	VTM_SHA_BENCH_MB=512 ./$(TEST_SHA_BIN)

# Удаляем все объектные файлы каталога, а не только текущий $(OBJ): на x86-64
# в OBJ попадает backend_avx2.o, на ARM — нет, объекты токенизатора строит
# Makefile.tok. Иначе `make ci` переиспользует .o от сборки с другими флагами:
# make не помнит, с какими флагами собран объектный файл. Бинарники самого
# токенизатора (vtm-tok, test_tokenizer) убирает `make -f Makefile.tok clean-tok`.
clean:
	rm -f *.o $(LIB) $(BIN) $(TEST_BIN) \
		$(TEST_API_BIN) $(TEST_SESSION_BIN) $(TEST_SHA_BIN) $(BENCH_BIN)
