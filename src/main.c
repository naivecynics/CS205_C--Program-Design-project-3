#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../lib/mat_mul.h"

#define TIME_START start=clock();
#define TIME_END(NAME) end=clock(); \
            duration=(double)(end-start) / CLOCKS_PER_SEC * 1000; \
            printf("%s duration: %.2fms\n", (NAME), duration);

int main() {
    size_t scale;
    char path[100];
    printf ("Calculate the Matrix with Size: ");
    scanf ("%zu", &scale) != 1;


    sprintf (path, "/mnt/c/SUSCode/CS205_C++-Program-Design/project-3/data/mat_data_8^%zu.0.csv", scale);
    struct mat matA, matB, matC;

    // data loading includes memory allocation
    read_csv_to_mat(path, &matA, &matB, &matC);

    // print_mat (&matA);
    // printf ("\n");
    // print_mat (&matB);

    clock_t start, end;
    double duration;

    size_t id = 2;

    TIME_START
    matmul_simd(&matA, &matB, &matC);
    TIME_END("simd")
    printf ("matC.data[%zu] = %.2f\n", id, matC.data[id]);
    // print_mat (&matC);
    clean_mat(&matC);

    TIME_START
    matmul_openblas(&matA, &matB, &matC);
    TIME_END("openblas")
    printf ("matC.data[%zu] = %.2f\n", id, matC.data[id]);
    // print_mat (&matC);
    clean_mat(&matC);

    TIME_START
    matmul_plain(&matA, &matB, &matC);
    TIME_END("plain")
    printf ("matC.data[%zu] = %.2f\n", id, matC.data[id]);
    // print_mat (&matC);
    clean_mat(&matC);

    free(matA.data);
    free(matB.data);
    free(matC.data);

    return 0;
}
