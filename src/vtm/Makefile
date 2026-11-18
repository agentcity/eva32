# VTM: скалярный интерпретатор. Внешних зависимостей нет, только libm.
CC       ?= cc
AR       ?= ar
CFLAGS   ?= -std=c99 -O2 -Wall -Wextra -Werror
CPPFLAGS ?= -D_POSIX_C_SOURCE=200809L
LDLIBS   ?= -lm

LIB      = libvtm.a
BIN      = vtm
SRC      = trit.c loader.c interp.c backend_scalar.c
OBJ      = $(SRC:.c=.o)
TEST_SRC = ../../tests/vtm/test_vtm.c
TEST_BIN = test_vtm

.PHONY: all test test-all bench clean

all: $(BIN)

$(BIN): main.o $(LIB)
	$(CC) $(CFLAGS) -o $@ main.o $(LIB) $(LDLIBS)

$(LIB): $(OBJ)
	$(AR) rcs $@ $(OBJ)

%.o: %.c vtm.h vtm_internal.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -I. -c -o $@ $<

$(TEST_BIN): $(TEST_SRC) $(LIB)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I. -I../.. -o $@ $(TEST_SRC) $(LIB) $(LDLIBS)

test: $(TEST_BIN)
	./$(TEST_BIN)

# Сборка токенизатора живёт отдельно (Makefile.tok), цели объединяются здесь:
# общего Makefile у трёх параллельных задач быть не могло.
test-all: test
	$(MAKE) -f Makefile.tok test-tok

bench: $(BIN)
	./$(BIN) bench --rows 512 --cols 512 --iters 200

clean:
	rm -f $(OBJ) main.o $(LIB) $(BIN) $(TEST_BIN)
