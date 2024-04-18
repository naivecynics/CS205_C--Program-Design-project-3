#include <iostream>
#include <cstdlib>
#include <chrono>
#include <immintrin.h>
#include "omp.h"
#include "cblas.h"
using namespace std;

#define TIME_START start=std::chrono::steady_clock::now();
#define TIME_END(NAME) end=std::chrono::steady_clock::now(); \
            duration=std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();\
            cout<<(NAME)<<": ps[2]="<<result[2] \
            <<", ps[3]="<<result[3] \
            <<" ps[n-1]="<<result[nSize-1] \
            <<", duration = "<<duration<<"ms"<<endl;

float *add(float *p1, float *p2, size_t n)
{
    float *sum = new float[n]();
    omp_set_num_threads(8);
    // #pragma omp parallel for
    for (size_t i = 0; i < n; i++)
    {
        sum[i] = p1[i] + p2[i];
        // printf("i = %d, I am Thread %d\n", i, omp_get_thread_num());
        // printf ("%d ", omp_get_thread_num());
    }
    return sum;
}

float *add_avx2(const float *p1, const float *p2, size_t n)
{
#ifdef WITH_AVX2
    if (n % 8 != 0)
    {
        std::cerr << "The size n must be a multiple of 8." << std::endl;
        return nullptr;
    }

    float *sum = new float[n]();
    __m256 a, b;
    __m256 c = _mm256_setzero_ps();

    for (size_t i = 0; i < n; i += 8)
    {
        a = _mm256_loadu_ps(p1 + i);
        b = _mm256_loadu_ps(p2 + i);
        c = _mm256_add_ps(a, b);
        _mm256_storeu_ps(sum + i, c);
        // printf ("%d ", omp_get_thread_num());
    }
    return sum;
#else
    std::cerr << "AVX2 is not supported" << std::endl;
    return nullptr;
#endif
}

float *add_avx2_omp(const float *p1, const float *p2, size_t n)
{
#ifdef WITH_AVX2
    if (n % 8 != 0)
    {
        std::cerr << "The size n must be a multiple of 8." << std::endl;
        return nullptr;
    }

    float *sum = new float[n]();
    __m256 a, b;
    __m256 c = _mm256_setzero_ps();

    // omp_set_num_threads(6);

    #pragma omp parallel for
    for (int i = 0; i < n; i += 8)
    {
        a = _mm256_loadu_ps(p1 + i);
        b = _mm256_loadu_ps(p2 + i);
        c = _mm256_add_ps(a, b);
        _mm256_storeu_ps(sum + i, c);
        // printf("i = %d, I am Thread %d\n", i, omp_get_thread_num());
        // printf ("%d ", omp_get_thread_num());
    }
    return sum;
#else
    std::cerr << "AVX2 is not supported" << std::endl;
    return nullptr;
#endif
}

int main()
{
    size_t nSize = 200000000;

    // 256bits aligned, C++17 standard
    float *p1 = static_cast<float *>(aligned_alloc(256, nSize * sizeof(float)));
    float *p2 = static_cast<float *>(aligned_alloc(256, nSize * sizeof(float)));
    float *result = NULL;

    p1[2] = 2.3f;
    p2[2] = 3.0f;
    p1[nSize - 1] = 2.0f;
    p2[nSize - 1] = 1.1f;

    auto start = std::chrono::steady_clock::now();
    auto end = std::chrono::steady_clock::now();
    auto duration = 0L;

    result = add(p1, p2, nSize);
    result = add(p1, p2, nSize);

    TIME_START
    result = add(p1, p2, nSize);
    TIME_END("normal_add")
    free(result);

    TIME_START
    result = add_avx2(p1, p2, nSize);
    TIME_END("SIMD_add  ")
    free(result);

    TIME_START
    result = add_avx2_omp(p1, p2, nSize);
    TIME_END("BOTH_add  ")
    free(result);

    free(p1);
    free(p2);
}