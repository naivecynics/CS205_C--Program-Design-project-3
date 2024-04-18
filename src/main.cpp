#include <iostream>
#include <chrono>
#include <string>
#include "../lib/mat_mul.hpp"
#include "cblas.h"
using namespace std;

#define TIME_START start=std::chrono::steady_clock::now();
#define TIME_END(NAME) end=std::chrono::steady_clock::now(); \
            duration=std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();\
            cout<<(NAME)<<" duration: "<<duration<<"ms"<<endl;

int main() {
    const string path = "/mnt/c/SUSCode/CS205_C++-Program-Design/project-3/data/mat_data_10^3.0.csv";
    mat matA, matB, matC;
    read_csv_to_mat(path, &matA, &matB, &matC);

    // print_clean_mat(&matA);
    // print_clean_mat(&matB);

    auto start = std::chrono::steady_clock::now();
    auto end = std::chrono::steady_clock::now();
    auto duration = 0L;

    TIME_START
    matmul_openblas(&matA, &matB, &matC);
    TIME_END("openblas")
    clean_mat(&matC);

    TIME_START
    matmul_plain(&matA, &matB, &matC);
    TIME_END("plain")

    clean_mat(&matC);

    delete[] matA.data;
    delete[] matB.data;

    return 0;
}