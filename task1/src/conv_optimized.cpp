// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER
// Hint: measure after every change. Not every "optimization" helps  let the numbers,
// not intuition, decide.

#include <immintrin.h>

#include "convolution.h"

void conv_optimized_2(const float* in, float* out, const float* ker,
                          int H, int W, int K);

void conv_optimized_4(const float* in, float* out, const float* ker,
                          int H, int W, int K);

void conv_optimized_8(const float* in, float* out, const float* ker,
                          int H, int W, int K);

void conv_optimized(const float* in, float* out, const float* ker,
                  int H, int W, int K) {
    return conv_optimized_4(in, out, ker, H, W, K);
}

void conv_optimized_2(const float* in, float* out, const float* ker,
                          int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;
    const int fpi = 16;
    const int unroll = 2;

    for (int oy = 0; oy < H; ++oy) {
        float* out_row = out + oy * W;
        int ox = 0;

        // 2 x AVX-512 = 32 output pixels
        for (; ox <= W - unroll * fpi; ox += unroll * fpi) {
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();

            for (int ky = 0; ky < K; ++ky) {
                const float* in_row =
                    in + (oy + ky) * in_stride + ox;
                const float* ker_row =
                    ker + ky * K;

                for (int kx = 0; kx < K; ++kx) {
                    __m512 vk = _mm512_set1_ps(ker_row[kx]);

                    __m512 vi0 =
                        _mm512_loadu_ps(in_row + kx);
                    __m512 vi1 =
                        _mm512_loadu_ps(in_row + kx + 16);

                    acc0 = _mm512_fmadd_ps(vi0, vk, acc0);
                    acc1 = _mm512_fmadd_ps(vi1, vk, acc1);
                }
            }

            _mm512_storeu_ps(out_row + ox,      acc0);
            _mm512_storeu_ps(out_row + ox + 16, acc1);
        }

        // Remaining full 16-float vector
        if (ox <= W - 16) {
            __m512 acc = _mm512_setzero_ps();

            for (int ky = 0; ky < K; ++ky) {
                const float* in_row =
                    in + (oy + ky) * in_stride + ox;
                const float* ker_row =
                    ker + ky * K;

                for (int kx = 0; kx < K; ++kx) {
                    __m512 vi =
                        _mm512_loadu_ps(in_row + kx);
                    __m512 vk =
                        _mm512_set1_ps(ker_row[kx]);

                    acc = _mm512_fmadd_ps(vi, vk, acc);
                }
            }

            _mm512_storeu_ps(out_row + ox, acc);
            ox += 16;
        }

        // Remaining 8 floats
        if (ox <= W - 8) {
            __m256 acc = _mm256_setzero_ps();

            for (int ky = 0; ky < K; ++ky) {
                const float* in_row =
                    in + (oy + ky) * in_stride + ox;
                const float* ker_row =
                    ker + ky * K;

                for (int kx = 0; kx < K; ++kx) {
                    __m256 vi =
                        _mm256_loadu_ps(in_row + kx);
                    __m256 vk =
                        _mm256_set1_ps(ker_row[kx]);

                    acc = _mm256_fmadd_ps(vi, vk, acc);
                }
            }

            _mm256_storeu_ps(out_row + ox, acc);
            ox += 8;
        }

        // Remaining 4 floats
        if (ox <= W - 4) {
            __m128 acc = _mm_setzero_ps();

            for (int ky = 0; ky < K; ++ky) {
                const float* in_row =
                    in + (oy + ky) * in_stride + ox;
                const float* ker_row =
                    ker + ky * K;

                for (int kx = 0; kx < K; ++kx) {
                    __m128 vi =
                        _mm_loadu_ps(in_row + kx);
                    __m128 vk =
                        _mm_set1_ps(ker_row[kx]);

                    acc = _mm_fmadd_ps(vi, vk, acc);
                }
            }

            _mm_storeu_ps(out_row + ox, acc);
            ox += 4;
        }

        // Scalar remainder
        for (; ox < W; ++ox) {
            float acc = 0.0f;

            for (int ky = 0; ky < K; ++ky) {
                const float* in_row =
                    in + (oy + ky) * in_stride + ox;
                const float* ker_row =
                    ker + ky * K;

                for (int kx = 0; kx < K; ++kx) {
                    acc += in_row[kx] * ker_row[kx];
                }
            }

            out_row[ox] = acc;
        }
    }
}

void conv_optimized_4(const float* in, float* out, const float* ker,
                  int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride
    const int fpi = 16;               // floats per AVX-512 instruction
    const int unroll = 4;

    for (int oy = 0; oy < H; ++oy) {
        float* out_row = out + oy * W;
        int ox = 0;

        // AVX-512, unrolled by 4:
        // process 4 * 16 = 64 output pixels at once
        for (; ox <= W - unroll * fpi; ox += unroll * fpi) {
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();

            for (int ky = 0; ky < K; ++ky) {
                const float* in_row =
                    in + (oy + ky) * in_stride + ox;
                const float* ker_row =
                    ker + ky * K;

                for (int kx = 0; kx < K; ++kx) {
                    __m512 vk = _mm512_set1_ps(ker_row[kx]);

                    __m512 vi0 =
                        _mm512_loadu_ps(in_row + kx);
                    __m512 vi1 =
                        _mm512_loadu_ps(in_row + kx + 16);
                    __m512 vi2 =
                        _mm512_loadu_ps(in_row + kx + 32);
                    __m512 vi3 =
                        _mm512_loadu_ps(in_row + kx + 48);

                    acc0 = _mm512_fmadd_ps(vi0, vk, acc0);
                    acc1 = _mm512_fmadd_ps(vi1, vk, acc1);
                    acc2 = _mm512_fmadd_ps(vi2, vk, acc2);
                    acc3 = _mm512_fmadd_ps(vi3, vk, acc3);
                }
            }

            _mm512_storeu_ps(out_row + ox,      acc0);
            _mm512_storeu_ps(out_row + ox + 16, acc1);
            _mm512_storeu_ps(out_row + ox + 32, acc2);
            _mm512_storeu_ps(out_row + ox + 48, acc3);
        }

        // Remaining full AVX-512 vectors
        for (; ox <= W - 16; ox += 16) {
            __m512 acc = _mm512_setzero_ps();

            for (int ky = 0; ky < K; ++ky) {
                const float* in_row =
                    in + (oy + ky) * in_stride + ox;
                const float* ker_row =
                    ker + ky * K;

                for (int kx = 0; kx < K; ++kx) {
                    __m512 vi =
                        _mm512_loadu_ps(in_row + kx);
                    __m512 vk =
                        _mm512_set1_ps(ker_row[kx]);

                    acc = _mm512_fmadd_ps(vi, vk, acc);
                }
            }

            _mm512_storeu_ps(out_row + ox, acc);
        }

        // Remaining 8 pixels
        if (ox <= W - 8) {
            __m256 acc = _mm256_setzero_ps();

            for (int ky = 0; ky < K; ++ky) {
                const float* in_row =
                    in + (oy + ky) * in_stride + ox;
                const float* ker_row =
                    ker + ky * K;

                for (int kx = 0; kx < K; ++kx) {
                    __m256 vi =
                        _mm256_loadu_ps(in_row + kx);
                    __m256 vk =
                        _mm256_set1_ps(ker_row[kx]);

                    acc = _mm256_fmadd_ps(vi, vk, acc);
                }
            }

            _mm256_storeu_ps(out_row + ox, acc);
            ox += 8;
        }

        // Remaining 4 pixels
        if (ox <= W - 4) {
            __m128 acc = _mm_setzero_ps();

            for (int ky = 0; ky < K; ++ky) {
                const float* in_row =
                    in + (oy + ky) * in_stride + ox;
                const float* ker_row =
                    ker + ky * K;

                for (int kx = 0; kx < K; ++kx) {
                    __m128 vi =
                        _mm_loadu_ps(in_row + kx);
                    __m128 vk =
                        _mm_set1_ps(ker_row[kx]);

                    acc = _mm_fmadd_ps(vi, vk, acc);
                }
            }

            _mm_storeu_ps(out_row + ox, acc);
            ox += 4;
        }

        // Scalar remainder
        for (; ox < W; ++ox) {
            float acc = 0.0f;

            for (int ky = 0; ky < K; ++ky) {
                const float* in_row =
                    in + (oy + ky) * in_stride + ox;
                const float* ker_row =
                    ker + ky * K;

                for (int kx = 0; kx < K; ++kx) {
                    acc += in_row[kx] * ker_row[kx];
                }
            }

            out_row[ox] = acc;
        }
    }
}


void conv_optimized_8(const float* in, float* out, const float* ker,
                          int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;
    const int fpi = 16;
    const int unroll = 8;

    for (int oy = 0; oy < H; ++oy) {
        float* out_row = out + oy * W;
        int ox = 0;

        // 8 x AVX-512 = 128 output pixels
        for (; ox <= W - unroll * fpi; ox += unroll * fpi) {
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();
            __m512 acc4 = _mm512_setzero_ps();
            __m512 acc5 = _mm512_setzero_ps();
            __m512 acc6 = _mm512_setzero_ps();
            __m512 acc7 = _mm512_setzero_ps();

            for (int ky = 0; ky < K; ++ky) {
                const float* in_row =
                    in + (oy + ky) * in_stride + ox;
                const float* ker_row =
                    ker + ky * K;

                for (int kx = 0; kx < K; ++kx) {
                    __m512 vk = _mm512_set1_ps(ker_row[kx]);

                    __m512 vi0 =
                        _mm512_loadu_ps(in_row + kx);
                    __m512 vi1 =
                        _mm512_loadu_ps(in_row + kx + 16);
                    __m512 vi2 =
                        _mm512_loadu_ps(in_row + kx + 32);
                    __m512 vi3 =
                        _mm512_loadu_ps(in_row + kx + 48);
                    __m512 vi4 =
                        _mm512_loadu_ps(in_row + kx + 64);
                    __m512 vi5 =
                        _mm512_loadu_ps(in_row + kx + 80);
                    __m512 vi6 =
                        _mm512_loadu_ps(in_row + kx + 96);
                    __m512 vi7 =
                        _mm512_loadu_ps(in_row + kx + 112);

                    acc0 = _mm512_fmadd_ps(vi0, vk, acc0);
                    acc1 = _mm512_fmadd_ps(vi1, vk, acc1);
                    acc2 = _mm512_fmadd_ps(vi2, vk, acc2);
                    acc3 = _mm512_fmadd_ps(vi3, vk, acc3);
                    acc4 = _mm512_fmadd_ps(vi4, vk, acc4);
                    acc5 = _mm512_fmadd_ps(vi5, vk, acc5);
                    acc6 = _mm512_fmadd_ps(vi6, vk, acc6);
                    acc7 = _mm512_fmadd_ps(vi7, vk, acc7);
                }
            }

            _mm512_storeu_ps(out_row + ox,       acc0);
            _mm512_storeu_ps(out_row + ox + 16,  acc1);
            _mm512_storeu_ps(out_row + ox + 32,  acc2);
            _mm512_storeu_ps(out_row + ox + 48,  acc3);
            _mm512_storeu_ps(out_row + ox + 64,  acc4);
            _mm512_storeu_ps(out_row + ox + 80,  acc5);
            _mm512_storeu_ps(out_row + ox + 96,  acc6);
            _mm512_storeu_ps(out_row + ox + 112, acc7);
        }

        // Remaining full 16-float vectors
        for (; ox <= W - 16; ox += 16) {
            __m512 acc = _mm512_setzero_ps();

            for (int ky = 0; ky < K; ++ky) {
                const float* in_row =
                    in + (oy + ky) * in_stride + ox;
                const float* ker_row =
                    ker + ky * K;

                for (int kx = 0; kx < K; ++kx) {
                    __m512 vi =
                        _mm512_loadu_ps(in_row + kx);
                    __m512 vk =
                        _mm512_set1_ps(ker_row[kx]);

                    acc = _mm512_fmadd_ps(vi, vk, acc);
                }
            }

            _mm512_storeu_ps(out_row + ox, acc);
        }

        // Remaining 8 floats
        if (ox <= W - 8) {
            __m256 acc = _mm256_setzero_ps();

            for (int ky = 0; ky < K; ++ky) {
                const float* in_row =
                    in + (oy + ky) * in_stride + ox;
                const float* ker_row =
                    ker + ky * K;

                for (int kx = 0; kx < K; ++kx) {
                    __m256 vi =
                        _mm256_loadu_ps(in_row + kx);
                    __m256 vk =
                        _mm256_set1_ps(ker_row[kx]);

                    acc = _mm256_fmadd_ps(vi, vk, acc);
                }
            }

            _mm256_storeu_ps(out_row + ox, acc);
            ox += 8;
        }

        // Remaining 4 floats
        if (ox <= W - 4) {
            __m128 acc = _mm_setzero_ps();

            for (int ky = 0; ky < K; ++ky) {
                const float* in_row =
                    in + (oy + ky) * in_stride + ox;
                const float* ker_row =
                    ker + ky * K;

                for (int kx = 0; kx < K; ++kx) {
                    __m128 vi =
                        _mm_loadu_ps(in_row + kx);
                    __m128 vk =
                        _mm_set1_ps(ker_row[kx]);

                    acc = _mm_fmadd_ps(vi, vk, acc);
                }
            }

            _mm_storeu_ps(out_row + ox, acc);
            ox += 4;
        }

        // Scalar remainder
        for (; ox < W; ++ox) {
            float acc = 0.0f;

            for (int ky = 0; ky < K; ++ky) {
                const float* in_row =
                    in + (oy + ky) * in_stride + ox;
                const float* ker_row =
                    ker + ky * K;

                for (int kx = 0; kx < K; ++kx) {
                    acc += in_row[kx] * ker_row[kx];
                }
            }

            out_row[ox] = acc;
        }
    }
}