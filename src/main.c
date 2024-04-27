#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "../lib/mat_mul.h"

#define TIME_START start=clock();
#define TIME_END(NAME) end=clock(); \
            duration=(double)(end-start) / CLOCKS_PER_SEC * 1000; \
            GflowPerSecond=(double)(2 * pow(2*pow(8,scale),3) * 1.0e-06) / duration; \
            printf ("    %s    duration: %.2fms\n", (NAME), duration); \
            printf ("    %s performance: %.2fGFlow/s\n", (NAME), GflowPerSecond);

int main() {
    size_t SCALE;
    printf ("Calculate the Matrix with Size Under: ");
    scanf ("%zu", &SCALE) != 1;

  for (size_t scale = 1; scale <= SCALE; scale++) {

    printf ("Dueing with Matrix of Size 2*8^%zu\n", scale);

    struct mat matA, matB, matC;

    // data loading includes memory allocation
    // char path[100];
    // sprintf (path, "./data/mat_2*8^%zu.csv", scale);
    // read_csv_to_mat(path, &matA, &matB, &matC);

    gen_random_mat (scale, &matA, &matB, &matC);

    // print_mat (&matA);
    // printf ("\n");
    // print_mat (&matB);

    clock_t start, end;
    double duration, GflowPerSecond;

    size_t id = 2;

    TIME_START
    matmul_simd(&matA, &matB, &matC);
    TIME_END("simd")
    printf ("   matC.data[%zu] = %.2f\n", id, matC.data[id]);
    // print_mat (&matC);
    clean_mat(&matC);

    TIME_START
    matmul_openblas(&matA, &matB, &matC);
    TIME_END("openblas")
    printf ("   matC.data[%zu] = %.2f\n", id, matC.data[id]);
    // print_mat (&matC);
    clean_mat(&matC);

    TIME_START
    matmul_plain(&matA, &matB, &matC);
    TIME_END("plain")
    printf ("   matC.data[%zu] = %.2f\n", id, matC.data[id]);
    // print_mat (&matC);
    clean_mat(&matC);

    free(matA.data);
    free(matB.data);
    free(matC.data);

  }

    return 0;
}
