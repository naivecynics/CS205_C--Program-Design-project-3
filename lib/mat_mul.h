#include <stdlib.h>
#include <stdio.h>

struct mat {
    size_t rows, cols;
    float *data;
};

void matmul_openblas (struct mat *A, struct mat *B, struct mat *C);
void matmul_plain (struct mat *A, struct mat *B, struct mat *C);
void matmul_simd (struct mat *A, struct mat *B, struct mat *C);

void read_csv_to_mat (const char *path, struct mat *A, struct mat *B, struct mat *C);
void clean_mat (struct mat *M);
void print_mat (struct mat *M);
