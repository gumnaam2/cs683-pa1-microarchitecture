#include <immintrin.h>
#include "matmul.h"

namespace {
constexpr int PREFETCH_DISTANCE = 1024;
constexpr _mm_hint PREFETCH_HINT = _MM_HINT_T1;

inline void prefetch_read(const float* ptr) {
    _mm_prefetch(reinterpret_cast<const char*>(ptr), PREFETCH_HINT);
}
}

void set_prefetch_distance(int) {}

void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            const float* a = A + i * lda;
            const float* b = B + j * ldb;
            float sum = 0.0f;

            for (int k = 0; k < K; ++k) {
                if ((k & 15) == 0) {
                    const int pf = k + PREFETCH_DISTANCE;
                    if (pf < K) {
                        prefetch_read(a + pf);
                        prefetch_read(b + pf);
                    }
                }

                sum += a[k] * b[k];
            }

            C[i * ldc + j] = sum;
        }
    }
}
