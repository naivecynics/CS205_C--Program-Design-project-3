cd /mnt/c/SUSCode/CS205_C++-Program-Design/project-3
gcc ./src/main.c ./src/mat_mul.c -o ./src/main -Ofast -lopenblas -fopenmp -ftree-vectorize -mavx -mfma -march=native -funroll-loops -ftree-loop-vectorize -g
gdb ./src/main