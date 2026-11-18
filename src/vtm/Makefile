# VTM: скалярный интерпретатор. Внешних зависимостей нет, кроме libm.
# CLI (main.c) — C99; режим --text линкуется с BPE-токенизатором (C++17),
# поэтому tok_bridge.o собирается из того же main.c с -DVTM_TOK_BRIDGE,
# а компилятор и линковщик для него — $(CXX).
CC       ?= cc
CXX      ?= c++
AR       ?= ar
CFLAGS   ?= -std=c99 -O2 -Wall -Wextra -Werror
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror
CPPFLAGS ?= -D_POSIX_C_SOURCE=200809L
LDLIBS   ?= -lm
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

.PHONY: all test test-all bench bench-backend bench-sha clean

all: $(BIN)

$(BIN): main.o $(LIB) $(BRIDGE) $(TOK_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ main.o $(LIB) $(BRIDGE) $(TOK_OBJ) $(LDLIBS) $(THREAD_LIBS)

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
	$(CC) $(CPPFLAGS) $(CFLAGS) -I. -I../.. -o $@ $(TEST_SRC) $(LIB) \
		$(LDLIBS) $(THREAD_LIBS)

# SHA-256 отдельно от интерпретатора: векторы, аппаратный путь против
# портативного, откат и подмена байта в контейнере (задача 0078).
$(TEST_SHA_BIN): $(TEST_SHA_SRC) $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I. -I../.. -o $@ $(TEST_SHA_SRC) $(LIB) \
		$(LDLIBS) $(THREAD_LIBS)

# C-ABI: колбэк стриминга и отмена из другого потока (pthread в тесте).
$(TEST_API_BIN): $(TEST_API_SRC) $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I. -I../.. -o $@ $(TEST_API_SRC) $(LIB) \
		$(LDLIBS) $(THREAD_LIBS)

# Бенчмарк/проверка SIMD (tests/vtm/bench_backend.c). Собирается тем же CC,
# что и библиотека, поэтому видит тот же SIMD.
$(BENCH_BIN): $(BENCH_SRC) $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I. -I../.. -o $@ $(BENCH_SRC) $(LIB) $(LDLIBS) $(THREAD_LIBS)

# Сессии KV (задача 0079): save/load/continue, обрыв и порча файла.
$(TEST_SESSION_BIN): $(TEST_SESSION_SRC) $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I. -I../.. -o $@ $(TEST_SESSION_SRC) $(LIB) \
		$(LDLIBS) $(THREAD_LIBS)

# backend_avx2.c собирается без -mavx2: AVX2/FMA включены атрибутами только
# на функциях-ядрах, чтобы детект CPUID не исполнял AVX2-инструкции.

# $(BIN) нужен интеграционному тесту CLI (текст -> токены -> ответ).
# bench_backend --check — тест «simd == scalar» (1e-5) на наборе форм;
# bench_backend --threads — тест «результат не зависит от 1/2/4/8 потоков»
# (задача 0077), с малым числом итераций: печатает и время, но проверяет
# только расхождение.
test: $(TEST_BIN) $(TEST_API_BIN) $(TEST_SESSION_BIN) $(TEST_SHA_BIN) $(BIN) \
	$(BENCH_BIN)
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

bench: $(BIN)
	./$(BIN) bench --rows 512 --cols 512 --iters 200

# GFLOP/s: scalar-lut243 / scalar-direct / SIMD + MM + RMSN.
bench-backend: $(BENCH_BIN)
	./$(BENCH_BIN)

# SHA-256: аппаратный и портативный путь в ГБ/с (задача 0078).
bench-sha: $(TEST_SHA_BIN)
	VTM_SHA_BENCH_MB=512 ./$(TEST_SHA_BIN)

clean:
	rm -f $(OBJ) main.o $(BRIDGE) $(LIB) $(BIN) $(TEST_BIN) $(TEST_API_BIN) \
		$(TEST_SESSION_BIN) $(TEST_SHA_BIN) $(BENCH_BIN)
