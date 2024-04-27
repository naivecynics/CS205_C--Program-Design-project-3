gcc ./src/main.c ./src/mat_mul.c -o ./main -Ofast -lopenblas -fopenmp -ftree-vectorize -mavx -mfma -march=native -funroll-loops -ftree-loop-vectorize -g -lm
gdb ./main