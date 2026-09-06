// matmul_simd.cpp  STAGE 1: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "matmul.h"

static inline float reduce256(__m256 x) {
    __m128 lo = _mm256_castps256_ps128(x);
    __m128 hi = _mm256_extractf128_ps(x, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {
    constexpr int J_TILE = 4;
    constexpr int VEC = 8;
    for (int i = 0; i < M; ++i) {
        int j = 0;
        for (; j + J_TILE <= N; j += J_TILE) {
            const float* a = A + i * lda;
            const float* b0 = B + (j + 0) * ldb;
            const float* b1 = B + (j + 1) * ldb;
            const float* b2 = B + (j + 2) * ldb;
            const float* b3 = B + (j + 3) * ldb;

            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();
            int k = 0;
            for (; k + VEC <= K; k += VEC) {
                __m256 av = _mm256_loadu_ps(a + k);

                acc0 = _mm256_fmadd_ps(
                    av,
                    _mm256_loadu_ps(b0 + k),
                    acc0);

                acc1 = _mm256_fmadd_ps(
                    av,
                    _mm256_loadu_ps(b1 + k),
                    acc1);

                acc2 = _mm256_fmadd_ps(
                    av,
                    _mm256_loadu_ps(b2 + k),
                    acc2);

                acc3 = _mm256_fmadd_ps(
                    av,
                    _mm256_loadu_ps(b3 + k),
                    acc3);
            }

            float sum0 = reduce256(acc0);
            float sum1 = reduce256(acc1);
            float sum2 = reduce256(acc2);
            float sum3 = reduce256(acc3);
            for (; k < K; ++k) {
                const float av = a[k];
                sum0 += av * b0[k];
                sum1 += av * b1[k];
                sum2 += av * b2[k];
                sum3 += av * b3[k];
            }
            C[i * ldc + j + 0] = sum0;
            C[i * ldc + j + 1] = sum1;
            C[i * ldc + j + 2] = sum2;
            C[i * ldc + j + 3] = sum3;
        }

        for (; j < N; ++j) {
            const float* a = A + i * lda;
            const float* b = B + j * ldb;
            __m256 acc = _mm256_setzero_ps();
            int k = 0;

            for (; k + VEC <= K; k += VEC) {
                acc = _mm256_fmadd_ps(
                    _mm256_loadu_ps(a + k),
                    _mm256_loadu_ps(b + k),
                    acc);
            }

            float sum = reduce256(acc);
            for (; k < K; ++k)
                sum += a[k] * b[k];
            C[i * ldc + j] = sum;
        }
    }
}
