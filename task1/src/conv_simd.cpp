// conv_simd.cpp  STAGE 4: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "convolution.h"

void conv_simd128(const float* in, float* out, const float* ker,
                  int H, int W, int K);
                  
void conv_simd256(const float* in, float* out, const float* ker,
                  int H, int W, int K);

void conv_simd512(const float* in, float* out, const float* ker,
                  int H, int W, int K);

void conv_simd(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    conv_simd512(in, out, ker, H, W, K);
}

void conv_simd128(const float* in, float* out, const float* ker,
                  int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride
    const int fpi = 4; // floats per instr: 4

    for (int oy = 0; oy < H; ++oy) {
        int ox = 0;
        for (; ox <= W-fpi; ox += fpi) {
            __m128 acc = _mm_setzero_ps();
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    __m128 vi = _mm_loadu_ps(in + (oy+ky) * in_stride + ox + kx);
                    __m128 vk = _mm_set1_ps(ker[ky*K + kx]);

                    acc = _mm_fmadd_ps(vi, vk, acc);
                }
            }
            _mm_storeu_ps(out + oy * W + ox, acc);
        }

        for(; ox < W; ox++) {
            float acc = 0.0f;
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                }
            }
            out[oy * W + ox] = acc;
        }
    }
}

void conv_simd256(const float* in, float* out, const float* ker,
                  int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride
    const int fpi = 8; // floats per instr: 8

    for (int oy = 0; oy < H; ++oy) {
        int ox = 0;
        for (; ox <= W-fpi; ox += fpi) {
            __m256 acc = _mm256_setzero_ps();
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    __m256 vi = _mm256_loadu_ps(in + (oy+ky) * in_stride + ox + kx);
                    __m256 vk = _mm256_set1_ps(ker[ky*K + kx]);

                    acc = _mm256_fmadd_ps(vi, vk, acc);
                }
            }
            _mm256_storeu_ps(out + oy * W + ox, acc);
        }

        if(ox <= W-4) {
            __m128 acc = _mm_setzero_ps();
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    __m128 vi = _mm_loadu_ps(in + (oy+ky) * in_stride + ox + kx);
                    __m128 vk = _mm_set1_ps(ker[ky*K + kx]);

                    acc = _mm_fmadd_ps(vi, vk, acc);
                }
            }
            _mm_storeu_ps(out + oy * W + ox, acc);
            ox += 4;
        }

        for(; ox < W; ox++) {
            float acc = 0.0f;
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                }
            }
            out[oy * W + ox] = acc;
        }
    }
}


void conv_simd512(const float* in, float* out, const float* ker,
                  int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride
    const int fpi = 16; // floats per instr: 16

    for (int oy = 0; oy < H; ++oy) {
        int ox = 0;
        for (; ox <= W-fpi; ox += fpi) {
            __m512 acc = _mm512_setzero_ps();
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    __m512 vi = _mm512_loadu_ps(in + (oy+ky) * in_stride + ox + kx);
                    __m512 vk = _mm512_set1_ps(ker[ky*K + kx]);

                    acc = _mm512_fmadd_ps(vi, vk, acc);
                }
            }
            _mm512_storeu_ps(out + oy * W + ox, acc);
        }

        if (ox <= W-8) {
            __m256 acc = _mm256_setzero_ps();
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    __m256 vi = _mm256_loadu_ps(in + (oy+ky) * in_stride + ox + kx);
                    __m256 vk = _mm256_set1_ps(ker[ky*K + kx]);

                    acc = _mm256_fmadd_ps(vi, vk, acc);
                }
            }
            _mm256_storeu_ps(out + oy * W + ox, acc);
            ox += 8;
        }

        if (ox <= W-4) {
            __m128 acc = _mm_setzero_ps();
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    __m128 vi = _mm_loadu_ps(in + (oy+ky) * in_stride + ox + kx);
                    __m128 vk = _mm_set1_ps(ker[ky*K + kx]);

                    acc = _mm_fmadd_ps(vi, vk, acc);
                }
            }
            _mm_storeu_ps(out + oy * W + ox, acc);
            ox += 4;
        }

        for(; ox < W; ox++) {
            float acc = 0.0f;
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                }
            }
            out[oy * W + ox] = acc;
        }
    }
}