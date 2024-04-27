gcc ./src/main.c ./src/mat_mul.c -o ./main -O3 -lopenblas -fopenmp -ftree-vectorize -mavx -mfma -march=native -funroll-loops -ftree-loop-vectorize -lm
./main
