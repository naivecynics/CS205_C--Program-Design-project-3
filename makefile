CC = gcc

CFLAGS = -lopenblas -fopenmp -ftree-vectorize -mavx -mfma -march=native -funroll-loops -ftree-loop-vectorize -lm -O4 -msse2 -msse3 -msse4

SRCS = ./src/main.c ./src/mat_mul.c
OBJS = $(SRCS:.c=.o)
MAIN = ./main

.PHONY: clean

$(MAIN): $(OBJS)
    $(CC) -o $(MAIN) $(OBJS) $(CFLAGS)

.c.o:
    $(CC) -c $

run:
	make clean

clean:
	rm
