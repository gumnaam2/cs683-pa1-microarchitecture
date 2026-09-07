// matmul_simd.cpp  STAGE 1: SIMD with AVX2/SSE intrinsics
#include <immintrin.h>
#include <cstdlib>

#include "matmul.h"

namespace {

int env_int(const char* name, int def) {
    const char* v = std::getenv(name);
    if (!v || *v == '\0') return def;
    char* end = nullptr;
    long parsed = std::strtol(v, &end, 10);
    if (end == v) return def;
    return static_cast<int>(parsed);
}

// SIMD_WIDTH=128 selects the SSE path below; anything else (including
// unset) keeps the original AVX2/256-bit path, so existing invocations
// are unaffected.
const int g_simd_width = env_int("SIMD_WIDTH", 256);

inline float reduce256(__m256 x) {
    __m128 lo = _mm256_castps256_ps128(x);
    __m128 hi = _mm256_extractf128_ps(x, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

inline float reduce128(__m128 x) {
    __m128 s = _mm_hadd_ps(x, x);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

void matmul_simd256(const float* A, const float* B, float* C,
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
                acc0 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b0 + k), acc0);
                acc1 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b1 + k), acc1);
                acc2 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b2 + k), acc2);
                acc3 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b3 + k), acc3);
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
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + k), _mm256_loadu_ps(b + k), acc);
            }
            float sum = reduce256(acc);
            for (; k < K; ++k)
                sum += a[k] * b[k];
            C[i * ldc + j] = sum;
        }
    }
}

// 128-bit (SSE/FMA) mirror of the above -- same structure, __m128/VEC=4
// instead of __m256/VEC=8. J_TILE kept at 4 so the register-blocking
// shape is otherwise identical; only the vector width changes, which is
// exactly the variable Table 2.1 is asking about.
void matmul_simd128(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {
    constexpr int J_TILE = 4;
    constexpr int VEC = 4;
    for (int i = 0; i < M; ++i) {
        int j = 0;
        for (; j + J_TILE <= N; j += J_TILE) {
            const float* a = A + i * lda;
            const float* b0 = B + (j + 0) * ldb;
            const float* b1 = B + (j + 1) * ldb;
            const float* b2 = B + (j + 2) * ldb;
            const float* b3 = B + (j + 3) * ldb;

            __m128 acc0 = _mm_setzero_ps();
            __m128 acc1 = _mm_setzero_ps();
            __m128 acc2 = _mm_setzero_ps();
            __m128 acc3 = _mm_setzero_ps();
            int k = 0;
            for (; k + VEC <= K; k += VEC) {
                __m128 av = _mm_loadu_ps(a + k);
                acc0 = _mm_fmadd_ps(av, _mm_loadu_ps(b0 + k), acc0);
                acc1 = _mm_fmadd_ps(av, _mm_loadu_ps(b1 + k), acc1);
                acc2 = _mm_fmadd_ps(av, _mm_loadu_ps(b2 + k), acc2);
                acc3 = _mm_fmadd_ps(av, _mm_loadu_ps(b3 + k), acc3);
            }

            float sum0 = reduce128(acc0);
            float sum1 = reduce128(acc1);
            float sum2 = reduce128(acc2);
            float sum3 = reduce128(acc3);
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
            __m128 acc = _mm_setzero_ps();
            int k = 0;
            for (; k + VEC <= K; k += VEC) {
                acc = _mm_fmadd_ps(_mm_loadu_ps(a + k), _mm_loadu_ps(b + k), acc);
            }
            float sum = reduce128(acc);
            for (; k < K; ++k)
                sum += a[k] * b[k];
            C[i * ldc + j] = sum;
        }
    }
}

} // namespace

void matmul_simd(const float* A, const float* B, float* C,
                  int M, int N, int K, int lda, int ldb, int ldc) {
    if (g_simd_width == 128) {
        matmul_simd128(A, B, C, M, N, K, lda, ldb, ldc);
    } else {
        matmul_simd256(A, B, C, M, N, K, lda, ldb, ldc);
    }
}

// // matmul_simd.cpp  STAGE 1: SIMD with AVX2 intrinsics
// #include <immintrin.h>

// #include "matmul.h"

// static inline float reduce256(__m256 x) {
//     __m128 lo = _mm256_castps256_ps128(x);
//     __m128 hi = _mm256_extractf128_ps(x, 1);
//     __m128 s = _mm_add_ps(lo, hi);
//     s = _mm_hadd_ps(s, s);
//     s = _mm_hadd_ps(s, s);
//     return _mm_cvtss_f32(s);
// }

// void matmul_simd(const float* A, const float* B, float* C,
//                  int M, int N, int K, int lda, int ldb, int ldc) {
//     constexpr int J_TILE = 4;
//     constexpr int VEC = 8;
//     for (int i = 0; i < M; ++i) {
//         int j = 0;
//         for (; j + J_TILE <= N; j += J_TILE) {
//             const float* a = A + i * lda;
//             const float* b0 = B + (j + 0) * ldb;
//             const float* b1 = B + (j + 1) * ldb;
//             const float* b2 = B + (j + 2) * ldb;
//             const float* b3 = B + (j + 3) * ldb;

//             __m256 acc0 = _mm256_setzero_ps();
//             __m256 acc1 = _mm256_setzero_ps();
//             __m256 acc2 = _mm256_setzero_ps();
//             __m256 acc3 = _mm256_setzero_ps();
//             int k = 0;
//             for (; k + VEC <= K; k += VEC) {
//                 __m256 av = _mm256_loadu_ps(a + k);

//                 acc0 = _mm256_fmadd_ps(
//                     av,
//                     _mm256_loadu_ps(b0 + k),
//                     acc0);

//                 acc1 = _mm256_fmadd_ps(
//                     av,
//                     _mm256_loadu_ps(b1 + k),
//                     acc1);

//                 acc2 = _mm256_fmadd_ps(
//                     av,
//                     _mm256_loadu_ps(b2 + k),
//                     acc2);

//                 acc3 = _mm256_fmadd_ps(
//                     av,
//                     _mm256_loadu_ps(b3 + k),
//                     acc3);
//             }

//             float sum0 = reduce256(acc0);
//             float sum1 = reduce256(acc1);
//             float sum2 = reduce256(acc2);
//             float sum3 = reduce256(acc3);
//             for (; k < K; ++k) {
//                 const float av = a[k];
//                 sum0 += av * b0[k];
//                 sum1 += av * b1[k];
//                 sum2 += av * b2[k];
//                 sum3 += av * b3[k];
//             }
//             C[i * ldc + j + 0] = sum0;
//             C[i * ldc + j + 1] = sum1;
//             C[i * ldc + j + 2] = sum2;
//             C[i * ldc + j + 3] = sum3;
//         }

//         for (; j < N; ++j) {
//             const float* a = A + i * lda;
//             const float* b = B + j * ldb;
//             __m256 acc = _mm256_setzero_ps();
//             int k = 0;

//             for (; k + VEC <= K; k += VEC) {
//                 acc = _mm256_fmadd_ps(
//                     _mm256_loadu_ps(a + k),
//                     _mm256_loadu_ps(b + k),
//                     acc);
//             }

//             float sum = reduce256(acc);
//             for (; k < K; ++k)
//                 sum += a[k] * b[k];
//             C[i * ldc + j] = sum;
//         }
//     }
// }
