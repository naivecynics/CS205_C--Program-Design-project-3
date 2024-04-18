#include <string>
#pragma once
using namespace std;

struct mat {
    size_t rows, cols;
    float *data;
};

void matmul_openblas (mat *A, mat *B, mat *C);
void matmul_plain (mat *A, mat *B, mat *C);

void read_csv_to_mat (string path, mat *A, mat *B, mat *C);
void clean_mat (mat *M);
void print_mat (mat *M);