#include <immintrin.h>
#include "matmul.h"

namespace {
constexpr int PREFETCH_DISTANCE_x = 64;
inline void prefetch_nta(const float* p) { _mm_prefetch(reinterpret_cast<const char*>(p), _MM_HINT_NTA); }
inline void prefetch_l3(const float* p) { _mm_prefetch(reinterpret_cast<const char*>(p), _MM_HINT_T2); }
inline void prefetch_l2(const float* p) { _mm_prefetch(reinterpret_cast<const char*>(p), _MM_HINT_T1); }
inline void prefetch_l1(const float* p) { _mm_prefetch(reinterpret_cast<const char*>(p), _MM_HINT_T0); }
}


void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            const float* a = A + i * lda;
            const float* b = B + j * ldb;
            float sum = 0.0f;
            prefetch_l2(A + (i+1)*lda);
            prefetch_l2(B + (j+1)*ldb);

            for (int k = 0; k < K; ++k) {
                const int pf = k + PREFETCH_DISTANCE_x;
                if ((k & 15) == 0) {
                    prefetch_l1(a + pf);
                    prefetch_l1(b + pf);
                }

                sum += a[k] * b[k];
            }

            C[i * ldc + j] = sum;
        }
    }
}

void matmul_prefetch2(const float* A, const float* B, float* C,
                     int M, int N, int K,
                     int lda, int ldb, int ldc)
{
    constexpr int CACHE_LINE_FLOATS = 16;
    constexpr int L1_DISTANCE = 64;   // ~4 cache lines ahead, into L1
    constexpr int L2_DISTANCE = 256;  // ~16 cache lines ahead, into L2/L3

    for (int i = 0; i < M; ++i) {
        const float* a = A + i * lda;

        // prefetch the start of the next A row (used next i iteration)
        if (i + 1 < M) {
            prefetch_l3(A + (i + 1) * lda);
        }

        for (int j = 0; j < N; ++j) {
            const float* b = B + j * ldb;

            // prefetch the start of the next B row before we start this one
            if (j + 1 < N) {
                prefetch_l2(B + (j + 1) * ldb);
            }

            float sum = 0.0f;

            for (int k = 0; k < K; ++k) {

                if ((k & (CACHE_LINE_FLOATS - 1)) == 0) {
                    int pf_near = k + L1_DISTANCE;
                    int pf_far  = k + L2_DISTANCE;

                    if (pf_near < K) {
                        prefetch_l1(a + pf_near);
                        prefetch_l1(b + pf_near);
                    }
                    if (pf_far < K) {
                        prefetch_l2(b + pf_far);   // B is the bigger footprint, worth a farther hint
                    }
                }

                sum += a[k] * b[k];
            }

            C[i * ldc + j] = sum;
        }
    }
}

void matmul_prefetch1(const float* A, const float* B, float* C,
                     int M, int N, int K,
                     int lda, int ldb, int ldc)
{
    constexpr int CACHE_LINE_FLOATS = 16;
    constexpr int L1_DISTANCE = 64;   // ~4 cache lines ahead, into L1
    constexpr int L2_DISTANCE = 256;  // ~16 cache lines ahead, into L2/L3

    for (int i = 0; i < M; ++i) {
        const float* a = A + i * lda;

        // prefetch the start of the next A row (used next i iteration)
        if (i + 1 < M) {
            prefetch_l3(A + (i + 1) * lda);
        }

        for (int j = 0; j < N; ++j) {
            const float* b = B + j * ldb;

            // prefetch the start of the next B row before we start this one
            if (j + 1 < N) {
                prefetch_l2(B + (j + 1) * ldb);
            }

            float sum = 0.0f;

            for (int k = 0; k < K; ++k) {

                if ((k & (CACHE_LINE_FLOATS - 1)) == 0) {
                    int pf_near = k + L1_DISTANCE;
                    int pf_far  = k + L2_DISTANCE;

                    if (pf_near < K) {
                        prefetch_l1(a + pf_near);
                        prefetch_l1(b + pf_near);
                    }
                    if (pf_far < K) {
                        prefetch_l2(b + pf_far);   // B is the bigger footprint, worth a farther hint
                    }
                }

                sum += a[k] * b[k];
            }

            C[i * ldc + j] = sum;
        }
    }
}


