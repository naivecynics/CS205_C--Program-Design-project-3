#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <math.h>

#include "omp.h"
#include "cblas.h"
#include "../lib/mat_mul.h"

#define NUM_LEN 10
#define PACK_SIZE 16
#define BLOCK_SIZE 256
// #define UNLOOP_RATE

void matmul_openblas (struct mat *A, struct mat *B, struct mat *C) {
    if (A == NULL || B == NULL || C == NULL) INVALID_POINTER
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, A->rows, B->rows, A->cols, 1.0, A->data, A->cols, B->data, B->cols, 0.0, C->data, C->cols);
}

void matmul_plain (struct mat *A, struct mat *B, struct mat *C) {
    if (A == NULL || B == NULL || C == NULL) INVALID_POINTER
    for (size_t i = 0; i < A->rows; i++)
        for (size_t j = 0; j < B->rows; j++)
            for (size_t k = 0; k < A->cols; k++)
                C->data[i * C->cols + j] += A->data[i * A->cols + k] * B->data[j * B->cols + k];
}

void matmul_simd (struct mat *A, struct mat *B, struct mat *C) {
    if (A == NULL || B == NULL || C == NULL) INVALID_POINTER
    if (A->cols != B->cols) UNMATCHED_MATRIX
    if (A->cols % 8 || B ->cols % 8) PACK_UNABLE

    // #pragma omp parallel for
    for (size_t i = 0; i < C->rows; i++) {
        for (size_t j = 0; j < C->cols; j++) {
            __m256 sum = _mm256_setzero_ps();
            for (size_t k = 0; k < A->cols; k += 8) {
                __m256 a = _mm256_loadu_ps(&A->data[i * A->cols + k]);
                __m256 b = _mm256_loadu_ps(&B->data[j * B->cols + k]);
                sum = _mm256_fmadd_ps(a, b, sum);
            }
            __m256 hsum = _mm256_hadd_ps(sum, sum);
            __m128 hsum_high = _mm256_extractf128_ps(hsum, 1);
            __m128 dot = _mm_add_ps(_mm256_castps256_ps128(hsum), hsum_high);
            __m128 dot_hadd = _mm_hadd_ps(dot, dot);
            C->data[i * C->cols + j] = _mm_cvtss_f32(dot_hadd);
        }
    }  
}

void matmul_simd_512 (struct mat *A, struct mat *B, struct mat *C) {
    if (A == NULL || B == NULL || C == NULL) INVALID_POINTER
    if (A->cols != B->cols) UNMATCHED_MATRIX
    if (A->cols % 16 || B ->cols % 16) PACK_UNABLE

    for (size_t i = 0; i < C->rows; i++) {
        for (size_t j = 0; j < C->cols; j++) {
            __m512 sum = _mm512_setzero_ps();
            for (size_t k = 0; k < A->cols; k += 16) {
                __m512 a = _mm512_loadu_ps(&A->data[i * A->cols + k]);
                __m512 b = _mm512_loadu_ps(&B->data[j * B->cols + k]);
                sum = _mm512_fmadd_ps(a, b, sum);
            }
            C->data[i * C->cols + j] = _mm512_mask_reduce_add_ps(65535, sum);
        }
    }
}

void matmul_blocked (struct mat *A, struct mat *B, struct mat *C) {
    if (A == NULL || B == NULL || C == NULL) INVALID_POINTER
    if (A->cols != B->cols) UNMATCHED_MATRIX
    // if (A->cols % 16 || B ->cols % 16) PACK_UNABLE

    for (size_t i = 0; i < C->rows; i += BLOCK_SIZE) {
        for (size_t j = 0; j < C->cols; j += BLOCK_SIZE) {
            float *block = (float *)malloc(BLOCK_SIZE * BLOCK_SIZE * sizeof(float));
            for (size_t ii = i; ii < MIN(i + BLOCK_SIZE, C->rows); ii++) {
                for (size_t jj = j; jj < MIN(j + BLOCK_SIZE, C->cols); jj++) {
                    __m512 sum = _mm512_setzero_ps();
                    for (size_t k = 0; k < A->cols; k += 16) {
                        __m512 a = _mm512_loadu_ps(&A->data[ii * A->cols + k]);
                        __m512 b = _mm512_loadu_ps(&B->data[jj * B->cols + k]);
                        sum = _mm512_fmadd_ps(a, b, sum);
                    }
                    block[(ii - i) * BLOCK_SIZE + (jj - j)] = _mm512_mask_reduce_add_ps(65535, sum);
                }
            }
            for (size_t ii = i; ii < MIN(i + BLOCK_SIZE, C->rows); ii++) {
                memcpy(&C->data[ii * C->cols + j], &block[(ii - i) * BLOCK_SIZE], BLOCK_SIZE * sizeof(float));
            }
            free(block);
        }
    }
}

void matmul_unloop (struct mat *A, struct mat *B, struct mat *C) {
        if (A == NULL || B == NULL || C == NULL) INVALID_POINTER
    if (A->cols != B->cols) UNMATCHED_MATRIX
    // if (A->cols % 16 || B ->cols % 16) PACK_UNABLE

    for (size_t i = 0; i < C->rows; i += BLOCK_SIZE) {
        for (size_t j = 0; j < C->cols; j += BLOCK_SIZE) {
            
            for (size_t ii = i; ii < MIN(i + BLOCK_SIZE, C->rows); ii++) {
                for (size_t jj = j; jj < MIN(j + BLOCK_SIZE, C->cols); jj++) {
                    __m512 sum = _mm512_setzero_ps();
                    for (size_t k = 0; k < A->cols; k += 16) {
                    __m512 a = _mm512_loadu_ps(&A->data[ii * A->cols + k]);
                    __m512 b = _mm512_loadu_ps(&B->data[jj * B->cols + k]);
                    sum = _mm512_fmadd_ps(a, b, sum);
                    
                    // a = _mm512_loadu_ps(&A->data[ii * A->cols + k + 16]);
                    // b = _mm512_loadu_ps(&B->data[jj * B->cols + k + 16]);
                    // sum = _mm512_fmadd_ps(a, b, sum);
                    
                    // a = _mm512_loadu_ps(&A->data[ii * A->cols + k + 32]);
                    // b = _mm512_loadu_ps(&B->data[jj * B->cols + k + 32]);
                    // sum = _mm512_fmadd_ps(a, b, sum);
                    
                    // a = _mm512_loadu_ps(&A->data[ii * A->cols + k + 48]);
                    // b = _mm512_loadu_ps(&B->data[jj * B->cols + k + 48]);
                    // sum = _mm512_fmadd_ps(a, b, sum);
                    }
                    C->data[ii * C->cols + jj] = _mm512_mask_reduce_add_ps(65535, sum);
                }
            }
        }
    }
}


void gen_random_mat (size_t scale, struct mat *A, struct mat *B, struct mat *C) {
    if (A == NULL || B == NULL || C == NULL) INVALID_POINTER 

    printf ("Generaing Matrixes...");

    // memory allocation
    size_t size = 2 * pow(8, scale);
    A->rows = A->cols = B->rows = B->cols = C->rows = C->cols = size;
    A->data = (float *)malloc(A->rows * A->cols * sizeof(float));
    B->data = (float *)malloc(B->rows * B->cols * sizeof(float));
    C->data = (float *)malloc(C->rows * C->cols * sizeof(float));
    
    for (size_t i = 0; i < size * size; i++) {
        A->data[i] = 2.0 * (float)drand48() - 1.0;
    }
    for (size_t i = 0; i < size * size; i++) {
        B->data[i] = 2.0 * (float)drand48() - 1.0;
    }

    printf ("Done.\n");
}

void read_csv_to_mat (const char *path, struct mat *A, struct mat *B, struct mat *C) {
    if (A == NULL || B == NULL || C == NULL) INVALID_POINTER 

    FILE *file = fopen(path, "r");
    if (file == NULL) { fprintf(stderr, "File not found\n"); exit(1); }

    printf ("Reading Matrixes...");

    size_t n, m, k;

    if (fscanf(file, "%zu", &n) != 1) {
        fprintf(stderr, "Failed to read A->rows\n");
        exit(1);
    }
    if (fscanf(file, "%zu", &m) != 1) {
        fprintf(stderr, "Failed to read B->rows\n");
        exit(1);
    }
    if (fscanf(file, "%zu", &k) != 1) {
        fprintf(stderr, "Failed to read C->rows\n");
        exit(1);
    }

    A->rows = C->rows = n;
    B->rows = C->cols = m;
    A->cols = B->cols = k;

    A->data = (float *)malloc(A->rows * A->cols * sizeof(float));
    B->data = (float *)malloc(B->rows * B->cols * sizeof(float));
    C->data = (float *)malloc(C->rows * C->cols * sizeof(float));

    size_t size_A = 0, size_B = 0;
    // char line[C->rows * C->cols * NUM_LEN];
    char *line = (char *)malloc(C->rows * C->cols * NUM_LEN * sizeof(char));

    if (fgets(line, sizeof(line), file) == NULL) {
        fprintf(stderr, "Failed to read the first line and discard it\n");
        exit(1);
    }

    if (fgets(line, C->rows * C->cols * NUM_LEN, file) != NULL) {
        char *token = strtok(line, ",");
        while (token != NULL) {
            A->data[size_A++] = atof(token);
            token = strtok(NULL, ",");
        }
    }

    if (fgets(line, C->rows * C->cols * NUM_LEN, file) != NULL) {
        char *token = strtok(line, ",");
        while (token != NULL) {
            B->data[size_B++] = atof(token);
            token = strtok(NULL, ",");
        }
    }

    fclose(file);
    printf ("Done.\n");
}

void clean_mat (struct mat *M) {
    if (M == NULL || M->data == NULL) INVALID_POINTER
    memset(M->data, 0, M->rows * M->cols * sizeof(float));
}

void print_mat (struct mat *M) {
    if (M == NULL) INVALID_POINTER
    for (size_t i = 0; i < M->rows; i++) {
        for (size_t j = 0; j < M->cols; j++) {
            printf("%.2f \t", M->data[i * M->cols + j]);
        }
        printf("\n");
    }
}