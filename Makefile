CC      = gcc
AR      = ar
CFLAGS  = -std=c11 -O3 -Wall -Wextra -Iinclude -Isrc
LDFLAGS =

ARCH := $(shell uname -m)
ifeq ($(ARCH),aarch64)
    CFLAGS += -march=armv8-a+simd
endif

LIB_SRCS = src/model.c src/runner.c \
           src/ops/gemm.c src/ops/batchmatmul.c \
           src/ops/softmax.c src/ops/layernorm.c \
           src/ops/gelu.c src/ops/conv2d.c \
           src/ops/elemwise.c
LIB_OBJS = $(LIB_SRCS:.c=.o)

.PHONY: all clean

all: libdna.a

libdna.a: $(LIB_OBJS)
	$(AR) rcs $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f libdna.a $(LIB_OBJS)
