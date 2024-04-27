gcc ./src/main.c ./src/mat_mul.c -o ./main -lopenblas -fopenmp -ftree-vectorize -mavx -mfma -march=native -funroll-loops -ftree-loop-vectorize -lm -O4 -msse2 -msse3 -msse4
./main
