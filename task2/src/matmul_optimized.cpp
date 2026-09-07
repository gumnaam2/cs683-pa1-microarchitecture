#include <immintrin.h>
#include "matmul.h"

namespace {
inline float reduce256(__m256 x) {
    __m128 lo = _mm256_castps256_ps128(x);
    __m128 hi = _mm256_extractf128_ps(x, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}
inline void prefetch_nta(const float* p) { _mm_prefetch(reinterpret_cast<const char*>(p), _MM_HINT_NTA); }
inline void prefetch_l3(const float* p) { _mm_prefetch(reinterpret_cast<const char*>(p), _MM_HINT_T2); }
inline void prefetch_l2(const float* p) { _mm_prefetch(reinterpret_cast<const char*>(p), _MM_HINT_T1); }
inline void prefetch_l1(const float* p) { _mm_prefetch(reinterpret_cast<const char*>(p), _MM_HINT_T0); }

}

void matmul_optimized(const float* A, const float* B, float* C,
                       int M, int N, int K, int lda, int ldb, int ldc) {
    matmul_optimized8(A, B, C, M, N, K, lda, ldb, ldc);
}

void matmul_optimized8(const float* A, const float* B, float* C,
                       int M, int N, int K, int lda, int ldb, int ldc) {
    constexpr int VEC = 8;
    constexpr int SIMD_REGS = 8;     // register blocking width (unchanged)
    constexpr int JT = 128;       // columns of B per tile: 128*K*4B = 512KB for K=1024 -> fits L2
    constexpr int IT = 16;         // rows of A per tile: 8*K*4B = 32KB -> fits L1

    constexpr int prefetch_distance = 32;
    for (int jt = 0; jt < N; jt += JT) {
        const int j_hi = jt + JT < N ? jt + JT : N;
        // for (int pj = jt; pj < j_hi && pj < jt + 4; ++pj) {
        prefetch_l2(B);
        prefetch_l2(B + 1*ldb);
        prefetch_l2(B + 2*ldb);
        prefetch_l2(B + 3*ldb);
        prefetch_l2(B + 4*ldb);
        prefetch_l2(B + 5*ldb);
        prefetch_l2(B + 6*ldb);
        prefetch_l2(B + 7*ldb);
        // }
        for (int it = 0; it < M; it += IT) {
            const int i_hi = it + IT < M ? it + IT : M;
            constexpr int L2_PROMOTE_STRIDE = 64; // elements into each row
            // for (int pj = jt; pj < j_hi && pj < jt + 8; ++pj) {
            //     for (int off = 0; off < K; off += L2_PROMOTE_STRIDE) {
            //         prefetch_l2(B + pj * ldb + off);
            //     }
            // }
            for (int i = it; i < i_hi; ++i) {
                const float* a = A + i * lda;
                float* c_row = C + i*ldc;
                int j = jt;

                for (; j + SIMD_REGS <= j_hi; j += SIMD_REGS) {
                    const float* b0 = B + (j + 0) * ldb;
                    const float* b1 = B + (j + 1) * ldb;
                    const float* b2 = B + (j + 2) * ldb;
                    const float* b3 = B + (j + 3) * ldb;
                    const float* b4 = B + (j + 4) * ldb;
                    const float* b5 = B + (j + 5) * ldb;
                    const float* b6 = B + (j + 6) * ldb;
                    const float* b7 = B + (j + 7) * ldb;
                    // if (j + SIMD_REGS + 7 < j_hi) {
                    //     for (int r = 0; r < SIMD_REGS; ++r) {
                    //         prefetch_l2(B + (j + SIMD_REGS + r) * ldb);
                    //     }
                    // }
                    __m256 acc0 = _mm256_setzero_ps();
                    __m256 acc1 = _mm256_setzero_ps();
                    __m256 acc2 = _mm256_setzero_ps();
                    __m256 acc3 = _mm256_setzero_ps();
                    __m256 acc4 = _mm256_setzero_ps();
                    __m256 acc5 = _mm256_setzero_ps();
                    __m256 acc6 = _mm256_setzero_ps();
                    __m256 acc7 = _mm256_setzero_ps();
                    int k = 0;
                    for (; k + VEC <= K; k += VEC) {
                        const int pf = k + prefetch_distance;
                        // if (((k & (prefetch_distance - 1)) == 0)){
                            // prefetch_l1(a  + pf);
                            prefetch_l1(b0 + pf);
                            // prefetch_l1(b1 + pf);
                            // prefetch_l1(b2 + pf);
                            // prefetch_l1(b3 + pf);
                            prefetch_l1(b4 + pf);
                            // prefetch_l1(b5 + pf);
                            // prefetch_l1(b6 + pf);
                            // prefetch_l1(b7 + pf);
                        // }
                        // prefetch_l1(b0 + pf);
                        // prefetch_l1(b1 + pf);
                        __m256 av = _mm256_loadu_ps(a + k);
                        acc0 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b0 + k), acc0);
                        acc1 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b1 + k), acc1);
                        acc2 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b2 + k), acc2);
                        acc3 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b3 + k), acc3);
                        acc4 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b4 + k), acc4);
                        acc5 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b5 + k), acc5);
                        acc6 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b6 + k), acc6);
                        acc7 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b7 + k), acc7);
                    }
                    constexpr int C_TILE_PREFETCH_AHEAD = 3; // tiles, i.e. 2*SIMD_REGS floats
                    const int c_pf_j = j + C_TILE_PREFETCH_AHEAD * SIMD_REGS;
                    // if (c_pf_j < j_hi) {
                    //     prefetch_l1(c_row + c_pf_j);
                    // }

                    float sum0 = reduce256(acc0),
                        sum1 = reduce256(acc1),
                        sum2 = reduce256(acc2),
                        sum3 = reduce256(acc3),
                        sum4 = reduce256(acc4),
                        sum5 = reduce256(acc5),
                        sum6 = reduce256(acc6),
                        sum7 = reduce256(acc7);
                        
                    for (; k < K; ++k) {
                        const float av = a[k];
                        sum0 += av * b0[k]; sum1 += av * b1[k];
                        sum2 += av * b2[k]; sum3 += av * b3[k];
                        sum4 += av * b4[k]; sum5 += av * b5[k];
                        sum6 += av * b6[k]; sum7 += av * b7[k];
                    }
                    C[i * ldc + j + 0] = sum0; C[i * ldc + j + 1] = sum1;
                    C[i * ldc + j + 2] = sum2; C[i * ldc + j + 3] = sum3;
                    C[i * ldc + j + 4] = sum4; C[i * ldc + j + 5] = sum5;
                    C[i * ldc + j + 6] = sum6; C[i * ldc + j + 7] = sum7;
                }

                for (; j < j_hi; ++j) {
                    const float* b = B + j * ldb;
                    __m256 acc = _mm256_setzero_ps();
                    int k = 0;
                    for (; k + VEC <= K; k += VEC) {
                        // if (j + SIMD_REGS + 3 < j_hi) {
                    //     prefetch_l2(B + (j + SIMD_REGS + 0) * ldb);
                    //     prefetch_l2(B + (j + SIMD_REGS + 1) * ldb);
                    //     // prefetch_l2(B + (j + SIMD_REGS + 2) * ldb);
                    //     // prefetch_l2(B + (j + SIMD_REGS + 3) * ldb);
                    // }
                        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + k), _mm256_loadu_ps(b + k), acc);
                    }
                    float sum = reduce256(acc);
                    for (; k < K; ++k) sum += a[k] * b[k];
                    C[i * ldc + j] = sum;
                }
            }
        }
    }
}





















































































void matmul_optimized4(const float* A, const float* B, float* C,
                       int M, int N, int K, int lda, int ldb, int ldc) {
    //take I
    constexpr int VEC = 8;
    constexpr int SIMD_REGS = 4; //no. of SIMD
    constexpr int JT = 128; //cols of B per tile
    constexpr int IT = 8; //rows of A per tile
    
    constexpr int prefetch_distance = 8;
    
    for (int jt = 0; jt < N; jt += JT) {
        const int j_hi = jt + JT < N ? jt + JT : N;
        // for (int pj = jt; pj < j_hi && pj < jt + 4; ++pj) {
        //     prefetch_l1(B + pj * ldb);
        // }
        for (int it = 0; it < M; it += IT) {
            const int i_hi = it + IT < M ? it + IT : M;

            for (int i = it; i < i_hi; ++i) {
                const float* a = A + i * lda;
                float* c_row = C + i*ldc;
                int j = jt;

                for (; j + SIMD_REGS <= j_hi; j += SIMD_REGS) {
                    const float* b0 = B + (j + 0) * ldb;
                    const float* b1 = B + (j + 1) * ldb;
                    const float* b2 = B + (j + 2) * ldb;
                    const float* b3 = B + (j + 3) * ldb;
                    // if (j + SIMD_REGS + 3 < j_hi) {
                    //     prefetch_l2(B + (j + SIMD_REGS + 0) * ldb);
                    //     prefetch_l2(B + (j + SIMD_REGS + 1) * ldb);
                    //     // prefetch_l2(B + (j + SIMD_REGS + 2) * ldb);
                    //     // prefetch_l2(B + (j + SIMD_REGS + 3) * ldb);
                    // }
                    __m256 acc0 = _mm256_setzero_ps();
                    __m256 acc1 = _mm256_setzero_ps();
                    __m256 acc2 = _mm256_setzero_ps();
                    __m256 acc3 = _mm256_setzero_ps();
                    int k = 0;
                    for (; k + VEC <= K; k += VEC) {
                        // if ((k & (prefetch_distance - 1)) == 0){
                        //     const int pf = k + prefetch_distance;
                        //     if (pf + VEC <= K) {
                        //         prefetch_l1(a  + pf);
                        //         prefetch_l1(b0 + pf);
                        //         prefetch_l1(b1 + pf);
                        //         prefetch_l1(b2 + pf);
                        //         prefetch_l1(b3 + pf);
                        //     }
                        // }
                        __m256 av = _mm256_loadu_ps(a + k);
                        acc0 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b0 + k), acc0);
                        acc1 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b1 + k), acc1);
                        acc2 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b2 + k), acc2);
                        acc3 = _mm256_fmadd_ps(av, _mm256_loadu_ps(b3 + k), acc3);
                    }
                    constexpr int C_TILE_PREFETCH_AHEAD = 3; // tiles, i.e. 2*SIMD_REGS floats
                    const int c_pf_j = j + C_TILE_PREFETCH_AHEAD * SIMD_REGS;
                    // if (c_pf_j < j_hi) {
                    //     prefetch_l1(c_row + c_pf_j);
                    // }

                    float sum0 = reduce256(acc0), sum1 = reduce256(acc1);
                    float sum2 = reduce256(acc2), sum3 = reduce256(acc3);
                    for (; k < K; ++k) {
                        const float av = a[k];
                        sum0 += av * b0[k]; sum1 += av * b1[k];
                        sum2 += av * b2[k]; sum3 += av * b3[k];
                    }
                    C[i * ldc + j + 0] = sum0; C[i * ldc + j + 1] = sum1;
                    C[i * ldc + j + 2] = sum2; C[i * ldc + j + 3] = sum3;
                }

                for (; j < j_hi; ++j) {
                    const float* b = B + j * ldb;
                    __m256 acc = _mm256_setzero_ps();
                    int k = 0;
                    for (; k + VEC <= K; k += VEC) {
                        // const int pf = k + prefetch_distance;
                        // if (pf + VEC <= K) {
                        //     prefetch_l1(a + pf);
                        //     prefetch_l1(b + pf);
                        // }
                        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + k), _mm256_loadu_ps(b + k), acc);
                    }
                    float sum = reduce256(acc);
                    for (; k < K; ++k) sum += a[k] * b[k];
                    C[i * ldc + j] = sum;
                }
            }
        }
    }
}
