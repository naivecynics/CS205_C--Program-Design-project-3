#include <stdlib.h>
#include <stdio.h>

#pragma once

#define MIN(i, j) ((i)<(j) ? (i): (j))

#define INVALID_POINTER { \
    fprintf(stderr, "Invalid Pointer\n"); \
    exit(1); \
    }
#define UNMATCHED_MATRIX { \
    fprintf(stderr, "Unmatched Matrix\n"); \
    exit(1); \
    }
#define PACK_UNABLE { \
    fprintf(stderr, "Pack Unable\n"); \
    exit(1); \
    }
#define BLOCK_UNABLE { \
    fprintf(stderr, "Block Unable\n"); \
    exit(1); \
    }

struct mat {
    size_t rows, cols;
    float *data;
};

void matmul_openblas (struct mat *A, struct mat *B, struct mat *C);
void matmul_plain (struct mat *A, struct mat *B, struct mat *C);
void matmul_simd (struct mat *A, struct mat *B, struct mat *C);
void matmul_simd_512 (struct mat *A, struct mat *B, struct mat *C);
void matmul_blocked (struct mat *A, struct mat *B, struct mat *C);
void matmul_unloop (struct mat *A, struct mat *B, struct mat *C);
// void matmul_cuda (struct mat *A, struct mat *B, struct mat *C);

void read_csv_to_mat (const char *path, struct mat *A, struct mat *B, struct mat *C);
void gen_random_mat (size_t scale, struct mat *A, struct mat *B, struct mat *C);
void clean_mat (struct mat *M);
void print_mat (struct mat *M);
