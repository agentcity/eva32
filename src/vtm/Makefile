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
CFLAGS   ?= -std=c99 -O2 -Wall -Wextra -Wtautological-compare -Wtautological-constant-out-of-range-compare -Wunused-but-set-variable -Werror
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

# --- Metal-бэкенд (задача 0192, ADR 0021) ---------------------------------
# backend_metal.m — Objective-C: MTLDevice/MTLBuffer и GEMV-ядра. Веса не
# копируются: newBufferWithBytesNoCopy на срез mmap-региона контейнера.
#
# MSL (vtm_metal.metal) по умолчанию встраивается в vtm_metal_kernels.h и
# компилируется в рантайме (newLibraryWithSource): так сборка работает без
# Metal Toolchain (Xcode 26 выносит компилятор metal в скачиваемый компонент,
# `xcrun metal` без него падает). Если Toolchain есть, `make metallib` (или
# `make all` при найденном компиляторе) собирает vtm.metallib, и бэкенд
# загрузит его вместо исходника. Сборка без Metal: WITH_METAL=0 или не-Darwin
# — TU не добавляется, тесты проходят на CPU.
UNAME_S := $(shell uname -s)
WITH_METAL ?= $(if $(filter Darwin,$(UNAME_S)),1,0)
METAL_TC := $(shell xcrun -sdk macosx metal --version >/dev/null 2>&1 && echo 1 || echo 0)
ifeq ($(WITH_METAL),1)
CPPFLAGS += -DVTM_WITH_METAL=1
METAL_OBJ = backend_metal.o
METAL_HDR = vtm_metal_kernels.h
PLATFORM_LIBS += -framework Metal -framework Foundation
else
METAL_OBJ =
METAL_HDR =
PLATFORM_LIBS =
endif

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
# Стресс пула для TSan (задача 0121): одновременные отправители,
# vtm_threads_set, LUT243, пробуждение из cond_wait.
TSAN_SRC = ../../tests/vtm/tsan_pool.c
TSAN_BIN = tsan_pool

.PHONY: all test test-all ci tsan bench bench-backend bench-sha metallib clean

all: $(BIN)
# При установленном Metal Toolchain заодно собираем metallib: бэкенд загрузит
# его вместо рантайм-компиляции MSL. Без Toolchain молча остаётся рантайм.
ifeq ($(METAL_TC),1)
all: metallib
endif

$(BIN): main.o $(LIB) $(BRIDGE) $(TOK_OBJ)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ main.o $(LIB) $(BRIDGE) $(TOK_OBJ) $(LDLIBS) $(THREAD_LIBS) $(PLATFORM_LIBS)

$(LIB): $(OBJ) $(METAL_OBJ)
	$(AR) rcs $@ $(OBJ) $(METAL_OBJ)

%.o: %.c vtm.h vtm_internal.h vtm_backend.h sha256.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -I. -c -o $@ $<

# MSL -> C-строка для рантайм-компиляции. Файл генерируемый, в git не нужен.
$(METAL_HDR): vtm_metal.metal
	@python3 -c 'import sys; s=open(sys.argv[1]).read(); print("/* сгенерировано из vtm_metal.metal, не править */"); print("static const char vtm_metal_msl[] ="); [print("\""+l.replace("\\","\\\\").replace("\"","\\\"")+"\\n\"") for l in s.splitlines()]; print(";")' $< > $@

# -std=c99 не годится для .m; остальные флаги (включая -Werror) сохраняем.
backend_metal.o: backend_metal.m $(METAL_HDR) vtm.h vtm_internal.h vtm_backend.h
	$(CC) $(CPPFLAGS) $(filter-out -std=c99,$(CFLAGS)) -std=gnu11 -I. -c -o $@ $<

# Необязательный metallib: без Metal Toolchain цель честно говорит, что MSL
# будет компилироваться в рантайме, и не валит сборку.
metallib: vtm_metal.metal
	@if xcrun -sdk macosx metal --version >/dev/null 2>&1; then 	  xcrun -sdk macosx metal -c -o vtm_metal.air vtm_metal.metal && 	  xcrun -sdk macosx metallib -o vtm.metallib vtm_metal.air && 	  echo "OK: vtm.metallib"; 	else 	  echo "Metal Toolchain недоступен — MSL компилируется в рантайме"; 	fi

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
		$(LDLIBS) $(THREAD_LIBS) $(PLATFORM_LIBS)

# SHA-256 отдельно от интерпретатора: векторы, аппаратный путь против
# портативного, откат и подмена байта в контейнере (задача 0078).
$(TEST_SHA_BIN): $(TEST_SHA_SRC) $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -I. -I../.. -o $@ $(TEST_SHA_SRC) $(LIB) \
		$(LDLIBS) $(THREAD_LIBS) $(PLATFORM_LIBS)

# C-ABI: колбэк стриминга и отмена из другого потока (pthread в тесте).
$(TEST_API_BIN): $(TEST_API_SRC) $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -I. -I../.. -o $@ $(TEST_API_SRC) $(LIB) \
		$(LDLIBS) $(THREAD_LIBS) $(PLATFORM_LIBS)

# Бенчмарк/проверка SIMD (tests/vtm/bench_backend.c). Собирается тем же CC,
# что и библиотека, поэтому видит тот же SIMD.
$(BENCH_BIN): $(BENCH_SRC) $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -I. -I../.. -o $@ $(BENCH_SRC) $(LIB) $(LDLIBS) $(THREAD_LIBS) $(PLATFORM_LIBS)

# Стресс пула для TSan: обычная сборка тоже проходит, но смысл — в `make tsan`.
$(TSAN_BIN): $(TSAN_SRC) $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -I. -I../.. -o $@ $(TSAN_SRC) $(LIB) $(LDLIBS) $(THREAD_LIBS) $(PLATFORM_LIBS)

# Сессии KV (задача 0079): save/load/continue, обрыв и порча файла.
$(TEST_SESSION_BIN): $(TEST_SESSION_SRC) $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) -I. -I../.. -o $@ $(TEST_SESSION_SRC) $(LIB) \
		$(LDLIBS) $(THREAD_LIBS) $(PLATFORM_LIBS)

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

# --- tsan: ThreadSanitizer по пулу потоков (задача 0121) ------------------
# Пул ядер — lock-free горячий путь (слот команды, release/gen, монотонный
# done); обычные тесты гонки не ловят, а 0113 дважды ловила дедлоки руками.
# `make tsan` собирает с -fsanitize=thread то же, что `make test` (плюс
# tests/vtm/tsan_pool.c) и гоняет те же бинарники: отчёт TSan — это падение,
# а не строчка в логе.
#
# -Werror из обычных CFLAGS снят: сборка под санитайзером проверяет гонки,
#   а предупреждения уже ловит обычная сборка/CI; иначе чужое WIP-предупреждение
#   в тестовом файле валит TSan-прогон не по делу.
# TSAN_RUN — переносимый таймаут: perl's alarm переживает exec, поэтому
# зависший бинарник умирает по SIGALRM через TSAN_TIMEOUT секунд (на macOS
# нет coreutils timeout). `make tsan TSAN_RUN=` отключает таймаут.
# TSAN_OPTIONS можно переопределить, например halt_on_error=0 для сбора всех
# отчётов (тогда падение проверяется по коду возврата).
TSAN_FLAGS    ?= -fsanitize=thread -g -O1
TSAN_CFLAGS   ?= $(filter-out -Werror,$(CFLAGS)) $(TSAN_FLAGS)
TSAN_CXXFLAGS ?= $(filter-out -Werror,$(CXXFLAGS)) $(TSAN_FLAGS)
TSAN_LDFLAGS  ?= -fsanitize=thread
TSAN_TIMEOUT  ?= 600
TSAN_RUN      ?= perl -e 'alarm shift; exec @ARGV' $(TSAN_TIMEOUT)
TSAN_OPTIONS  ?= halt_on_error=1

tsan:
	@set -e; \
	t0=$$(date +%s); \
	echo "== tsan[1/2]: сборка с $(TSAN_FLAGS) =="; \
	$(MAKE) clean >/dev/null; \
	$(MAKE) $(BIN) $(TEST_BIN) $(TEST_API_BIN) $(TEST_SESSION_BIN) \
	    $(TEST_SHA_BIN) $(BENCH_BIN) $(TSAN_BIN) $(DEMO_MODEL) \
	    CFLAGS='$(TSAN_CFLAGS)' CXXFLAGS='$(TSAN_CXXFLAGS)' \
	    LDFLAGS='$(TSAN_LDFLAGS)'; \
	t1=$$(date +%s); \
	echo; echo "OK: TSan-сборка — $$((t1 - t0))s"; \
	echo; \
	echo "== tsan[2/2]: прогон, таймаут $(TSAN_TIMEOUT)с на бинарник =="; \
	TSAN_OPTIONS='$(TSAN_OPTIONS)' $(TSAN_RUN) ./$(TEST_SHA_BIN); \
	TSAN_OPTIONS='$(TSAN_OPTIONS)' $(TSAN_RUN) ./$(TEST_SESSION_BIN); \
	TSAN_OPTIONS='$(TSAN_OPTIONS)' $(TSAN_RUN) ./$(TEST_API_BIN); \
	TSAN_OPTIONS='$(TSAN_OPTIONS)' $(TSAN_RUN) ./$(TEST_BIN); \
	TSAN_OPTIONS='$(TSAN_OPTIONS)' $(TSAN_RUN) ./$(BENCH_BIN) --check; \
	TSAN_OPTIONS='$(TSAN_OPTIONS)' $(TSAN_RUN) ./$(BENCH_BIN) --threads --iters 3; \
	TSAN_OPTIONS='$(TSAN_OPTIONS)' $(TSAN_RUN) ./$(TSAN_BIN); \
	t2=$$(date +%s); \
	$(MAKE) clean >/dev/null; \
	echo; echo "OK: tsan — отчётов нет, $$((t2 - t0))s"

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
		$(TEST_API_BIN) $(TEST_SESSION_BIN) $(TEST_SHA_BIN) $(BENCH_BIN) \
		$(TSAN_BIN) $(METAL_OBJ) $(METAL_HDR) vtm.metallib vtm_metal.air
