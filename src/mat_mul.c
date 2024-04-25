#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

#include "omp.h"
#include "cblas.h"
#include "../lib/mat_mul.h"

#define NUM_LEN 10
#define INVALID_POINTER { \
    fprintf(stderr, "Invalid pointer\n"); \
    exit(1); \
    } \

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
    #pragma omp parallel for
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