#include <iostream>
#include <cstring>
#include <fstream>
#include <sstream>
#include "../lib/mat_mul.hpp"
#include "cblas.h"
using namespace std;

#define INVALID_POINTER { \
    cerr << "Invalid pointer" << std::endl; \
    exit(1); \
    } \

void matmul_openblas (mat *A, mat *B, mat *C) {
    if (A == nullptr || B == nullptr || C == nullptr) INVALID_POINTER
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A->rows, B->cols, A->cols, 1.0, A->data, A->cols, B->data, B->cols, 0.0, C->data, C->cols);
}

void matmul_plain (mat *A, mat *B, mat *C) {
    if (A == nullptr || B == nullptr || C == nullptr) INVALID_POINTER
    for (size_t i = 0; i < A->rows; i++)
        for (size_t j = 0; j < B->cols; j++)
            for (size_t k = 0; k < A->cols; k++)
                C->data[i * C->cols + j] += A->data[i * A->cols + k] * B->data[k * B->cols + j];
}

void read_csv_to_mat (string path, mat *A, mat *B, mat *C) { 
    if (A == nullptr || B == nullptr || C == nullptr) INVALID_POINTER
    ifstream file(path);
    if (!file) { cerr << "File not found" << std::endl; exit(1); }
    string line, cell;
    getline(file, line); A->rows = C->rows = stoi(line);
    getline(file, line); A->cols = B->rows = stoi(line);
    getline(file, line); B->cols = C->cols = stoi(line);
    A->data = new float[A->rows * A->cols];
    B->data = new float[B->rows * B->cols];
    C->data = new float[C->rows * C->cols];
    size_t size_A = 0, size_B = 0;
    getline(file, line); stringstream matAstream(line);
    while (getline(matAstream, cell, ','))
        A->data[size_A++] = stof(cell);
    getline(file, line); stringstream matBstream(line);
    while (getline(matBstream, cell, ','))
        B->data[size_B++] = stof(cell);
}

void clean_mat (mat *M) {
    if (M == nullptr || M->data == nullptr) INVALID_POINTER
    memset(M->data, 0, M->rows * M->cols * sizeof(float));
}

void print_mat (mat *M) {
    if (M == nullptr) INVALID_POINTER
    for (size_t i = 0; i < M->rows; i++) {
        for (size_t j = 0; j < M->cols; j++) {
            cout << M->data[i * M->cols + j] << " ";
        }   
        cout << std::endl;
    }
}
