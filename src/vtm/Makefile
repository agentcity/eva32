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

LIB      = libvtm.a
BIN      = vtm
SRC      = trit.c loader.c interp.c backend_scalar.c
OBJ      = $(SRC:.c=.o)
TOK_OBJ  = tokenizer.o
BRIDGE   = tok_bridge.o
TEST_SRC = ../../tests/vtm/test_vtm.c
TEST_BIN = test_vtm

.PHONY: all test test-all bench clean

all: $(BIN)

$(BIN): main.o $(LIB) $(BRIDGE) $(TOK_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ main.o $(LIB) $(BRIDGE) $(TOK_OBJ) $(LDLIBS)

$(LIB): $(OBJ)
	$(AR) rcs $@ $(OBJ)

%.o: %.c vtm.h vtm_internal.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -I. -c -o $@ $<

# Тот же main.c, но как C++17: C-ABI моста к tokenizer.hpp (см. main.c).
$(BRIDGE): main.c tokenizer.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -DVTM_TOK_BRIDGE -I. -x c++ -c -o $@ main.c

# tokenizer.o собирает Makefile.tok; здесь только зависимость и вызов.
$(TOK_OBJ): tokenizer.cpp tokenizer.hpp
	$(MAKE) -f Makefile.tok $(TOK_OBJ)

$(TEST_BIN): $(TEST_SRC) $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I. -I../.. -o $@ $(TEST_SRC) $(LIB) $(LDLIBS)

# $(BIN) нужен интеграционному тесту CLI (текст -> токены -> ответ).
test: $(TEST_BIN) $(BIN)
	./$(TEST_BIN)

# Сборка токенизатора живёт отдельно (Makefile.tok), цели объединяются здесь:
# общего Makefile у трёх параллельных задач быть не могло.
test-all: test
	$(MAKE) -f Makefile.tok test-tok

bench: $(BIN)
	./$(BIN) bench --rows 512 --cols 512 --iters 200

clean:
	rm -f $(OBJ) main.o $(BRIDGE) $(LIB) $(BIN) $(TEST_BIN)
