// conv_unroll.cpp  STAGE 2: LOOP UNROLLING
#include "convolution.h"

void conv_unroll(const float* in, float* out, const float* ker,
                 int H, int W, int K) {
    conv_unroll24(in, out, ker, H, W, K);
}

void conv_unroll2(const float* in, float* out, const float* ker,
                         int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oy = 0; oy < H; ++oy) {
        int ox = 0;
        float* out_row = out + oy * W;

        for (; ox + 2 <= W; ox += 2) {
            float p0 = 0.0f, p1 = 0.0f;

            for (int ky = 0; ky < K; ++ky) {
                const float* in_row = in + (oy + ky) * in_stride + ox;
                const float* ker_row = ker + ky * K;

                for (int kx = 0; kx < K; ++kx) {
                    const float kw = ker_row[kx];

                    p0 += in_row[kx]     * kw;
                    p1 += in_row[kx + 1] * kw;
                }
            }

            out_row[ox]     = p0;
            out_row[ox + 1] = p1;
        }

        for (; ox < W; ++ox) {
            float acc = 0.0f;

            for (int ky = 0; ky < K; ++ky) {
                const float* in_row = in + (oy + ky) * in_stride + ox;
                const float* ker_row = ker + ky * K;

                for (int kx = 0; kx < K; ++kx) {
                    acc += in_row[kx] * ker_row[kx];
                }
            }

            out_row[ox] = acc;
        }
    }
}


void conv_unroll4(const float* in, float* out, const float* ker,
                         int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oy = 0; oy < H; ++oy) {
        int ox = 0;
        float* out_row = out + oy * W;

        for (; ox + 4 <= W; ox += 4) {
            float p0 = 0.0f, p1 = 0.0f;
            float p2 = 0.0f, p3 = 0.0f;

            for (int ky = 0; ky < K; ++ky) {
                const float* in_row = in + (oy + ky) * in_stride + ox;
                const float* ker_row = ker + ky * K;

                for (int kx = 0; kx < K; ++kx) {
                    const float kw = ker_row[kx];

                    p0 += in_row[kx]     * kw;
                    p1 += in_row[kx + 1] * kw;
                    p2 += in_row[kx + 2] * kw;
                    p3 += in_row[kx + 3] * kw;
                }
            }

            out_row[ox]     = p0;
            out_row[ox + 1] = p1;
            out_row[ox + 2] = p2;
            out_row[ox + 3] = p3;
        }

        for (; ox < W; ++ox) {
            float acc = 0.0f;

            for (int ky = 0; ky < K; ++ky) {
                const float* in_row = in + (oy + ky) * in_stride + ox;
                const float* ker_row = ker + ky * K;

                for (int kx = 0; kx < K; ++kx) {
                    acc += in_row[kx] * ker_row[kx];
                }
            }

            out_row[ox] = acc;
        }
    }
}


void conv_unroll6(const float* in, float* out, const float* ker,
                         int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oy = 0; oy < H; ++oy) {
        int ox = 0;
        float* out_row = out + oy * W;

        for (; ox + 6 <= W; ox += 6) {
            float p0 = 0.0f, p1 = 0.0f;
            float p2 = 0.0f, p3 = 0.0f;
            float p4 = 0.0f, p5 = 0.0f;

            for (int ky = 0; ky < K; ++ky) {
                const float* in_row = in + (oy + ky) * in_stride + ox;
                const float* ker_row = ker + ky * K;

                for (int kx = 0; kx < K; ++kx) {
                    const float kw = ker_row[kx];

                    p0 += in_row[kx]     * kw;
                    p1 += in_row[kx + 1] * kw;
                    p2 += in_row[kx + 2] * kw;
                    p3 += in_row[kx + 3] * kw;
                    p4 += in_row[kx + 4] * kw;
                    p5 += in_row[kx + 5] * kw;
                }
            }

            out_row[ox]     = p0;
            out_row[ox + 1] = p1;
            out_row[ox + 2] = p2;
            out_row[ox + 3] = p3;
            out_row[ox + 4] = p4;
            out_row[ox + 5] = p5;
        }

        for (; ox < W; ++ox) {
            float acc = 0.0f;

            for (int ky = 0; ky < K; ++ky) {
                const float* in_row = in + (oy + ky) * in_stride + ox;
                const float* ker_row = ker + ky * K;

                for (int kx = 0; kx < K; ++kx) {
                    acc += in_row[kx] * ker_row[kx];
                }
            }

            out_row[ox] = acc;
        }
    }
}


void conv_unroll8(const float* in, float* out, const float* ker,
                         int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oy = 0; oy < H; ++oy) {
        int ox = 0;
        float* out_row = out + oy * W;

        for (; ox + 8 <= W; ox += 8) {
            float p0 = 0.0f, p1 = 0.0f;
            float p2 = 0.0f, p3 = 0.0f;
            float p4 = 0.0f, p5 = 0.0f;
            float p6 = 0.0f, p7 = 0.0f;

            for (int ky = 0; ky < K; ++ky) {
                const float* in_row = in + (oy + ky) * in_stride + ox;
                const float* ker_row = ker + ky * K;

                for (int kx = 0; kx < K; ++kx) {
                    const float kw = ker_row[kx];

                    p0 += in_row[kx]     * kw;
                    p1 += in_row[kx + 1] * kw;
                    p2 += in_row[kx + 2] * kw;
                    p3 += in_row[kx + 3] * kw;
                    p4 += in_row[kx + 4] * kw;
                    p5 += in_row[kx + 5] * kw;
                    p6 += in_row[kx + 6] * kw;
                    p7 += in_row[kx + 7] * kw;
                }
            }

            out_row[ox]     = p0;
            out_row[ox + 1] = p1;
            out_row[ox + 2] = p2;
            out_row[ox + 3] = p3;
            out_row[ox + 4] = p4;
            out_row[ox + 5] = p5;
            out_row[ox + 6] = p6;
            out_row[ox + 7] = p7;
        }

        for (; ox < W; ++ox) {
            float acc = 0.0f;

            for (int ky = 0; ky < K; ++ky) {
                const float* in_row = in + (oy + ky) * in_stride + ox;
                const float* ker_row = ker + ky * K;

                for (int kx = 0; kx < K; ++kx) {
                    acc += in_row[kx] * ker_row[kx];
                }
            }

            out_row[ox] = acc;
        }
    }
}


void conv_unroll12(const float* in, float* out, const float* ker,
                          int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oy = 0; oy < H; ++oy) {
        int ox = 0;
        float* out_row = out + oy * W;

        for (; ox + 12 <= W; ox += 12) {
            float p0  = 0.0f, p1  = 0.0f;
            float p2  = 0.0f, p3  = 0.0f;
            float p4  = 0.0f, p5  = 0.0f;
            float p6  = 0.0f, p7  = 0.0f;
            float p8  = 0.0f, p9  = 0.0f;
            float p10 = 0.0f, p11 = 0.0f;

            for (int ky = 0; ky < K; ++ky) {
                const float* in_row = in + (oy + ky) * in_stride + ox;
                const float* ker_row = ker + ky * K;

                for (int kx = 0; kx < K; ++kx) {
                    const float kw = ker_row[kx];

                    p0  += in_row[kx]      * kw;
                    p1  += in_row[kx + 1]  * kw;
                    p2  += in_row[kx + 2]  * kw;
                    p3  += in_row[kx + 3]  * kw;
                    p4  += in_row[kx + 4]  * kw;
                    p5  += in_row[kx + 5]  * kw;
                    p6  += in_row[kx + 6]  * kw;
                    p7  += in_row[kx + 7]  * kw;
                    p8  += in_row[kx + 8]  * kw;
                    p9  += in_row[kx + 9]  * kw;
                    p10 += in_row[kx + 10] * kw;
                    p11 += in_row[kx + 11] * kw;
                }
            }

            out_row[ox]      = p0;
            out_row[ox + 1]  = p1;
            out_row[ox + 2]  = p2;
            out_row[ox + 3]  = p3;
            out_row[ox + 4]  = p4;
            out_row[ox + 5]  = p5;
            out_row[ox + 6]  = p6;
            out_row[ox + 7]  = p7;
            out_row[ox + 8]  = p8;
            out_row[ox + 9]  = p9;
            out_row[ox + 10] = p10;
            out_row[ox + 11] = p11;
        }

        for (; ox < W; ++ox) {
            float acc = 0.0f;

            for (int ky = 0; ky < K; ++ky) {
                const float* in_row = in + (oy + ky) * in_stride + ox;
                const float* ker_row = ker + ky * K;

                for (int kx = 0; kx < K; ++kx) {
                    acc += in_row[kx] * ker_row[kx];
                }
            }

            out_row[ox] = acc;
        }
    }
}


void conv_unroll14(const float* in, float* out, const float* ker,
                          int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oy = 0; oy < H; ++oy) {
        int ox = 0;
        float* out_row = out + oy * W;

        for (; ox + 14 <= W; ox += 14) {
            float p0  = 0.0f, p1  = 0.0f;
            float p2  = 0.0f, p3  = 0.0f;
            float p4  = 0.0f, p5  = 0.0f;
            float p6  = 0.0f, p7  = 0.0f;
            float p8  = 0.0f, p9  = 0.0f;
            float p10 = 0.0f, p11 = 0.0f;
            float p12 = 0.0f, p13 = 0.0f;

            for (int ky = 0; ky < K; ++ky) {
                const float* in_row = in + (oy + ky) * in_stride + ox;
                const float* ker_row = ker + ky * K;

                for (int kx = 0; kx < K; ++kx) {
                    const float kw = ker_row[kx];

                    p0  += in_row[kx]      * kw;
                    p1  += in_row[kx + 1]  * kw;
                    p2  += in_row[kx + 2]  * kw;
                    p3  += in_row[kx + 3]  * kw;
                    p4  += in_row[kx + 4]  * kw;
                    p5  += in_row[kx + 5]  * kw;
                    p6  += in_row[kx + 6]  * kw;
                    p7  += in_row[kx + 7]  * kw;
                    p8  += in_row[kx + 8]  * kw;
                    p9  += in_row[kx + 9]  * kw;
                    p10 += in_row[kx + 10] * kw;
                    p11 += in_row[kx + 11] * kw;
                    p12 += in_row[kx + 12] * kw;
                    p13 += in_row[kx + 13] * kw;
                }
            }

            out_row[ox]      = p0;
            out_row[ox + 1]  = p1;
            out_row[ox + 2]  = p2;
            out_row[ox + 3]  = p3;
            out_row[ox + 4]  = p4;
            out_row[ox + 5]  = p5;
            out_row[ox + 6]  = p6;
            out_row[ox + 7]  = p7;
            out_row[ox + 8]  = p8;
            out_row[ox + 9]  = p9;
            out_row[ox + 10] = p10;
            out_row[ox + 11] = p11;
            out_row[ox + 12] = p12;
            out_row[ox + 13] = p13;
        }

        for (; ox < W; ++ox) {
            float acc = 0.0f;

            for (int ky = 0; ky < K; ++ky) {
                const float* in_row = in + (oy + ky) * in_stride + ox;
                const float* ker_row = ker + ky * K;

                for (int kx = 0; kx < K; ++kx) {
                    acc += in_row[kx] * ker_row[kx];
                }
            }

            out_row[ox] = acc;
        }
    }
}

void conv_unroll16(const float* in, float* out, const float* ker,
                          int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oy = 0; oy < H; ++oy) {
        int ox = 0;
        float* out_row = out + oy * W;

        for (; ox + 16 <= W; ox += 16) {
            float p0  = 0.0f, p1  = 0.0f;
            float p2  = 0.0f, p3  = 0.0f;
            float p4  = 0.0f, p5  = 0.0f;
            float p6  = 0.0f, p7  = 0.0f;
            float p8  = 0.0f, p9  = 0.0f;
            float p10 = 0.0f, p11 = 0.0f;
            float p12 = 0.0f, p13 = 0.0f;
            float p14 = 0.0f, p15 = 0.0f;

            for (int ky = 0; ky < K; ++ky) {
                const float* in_row = in + (oy + ky) * in_stride + ox;
                const float* ker_row = ker + ky * K;

                for (int kx = 0; kx < K; ++kx) {
                    const float kw = ker_row[kx];

                    p0  += in_row[kx]      * kw;
                    p1  += in_row[kx + 1]  * kw;
                    p2  += in_row[kx + 2]  * kw;
                    p3  += in_row[kx + 3]  * kw;
                    p4  += in_row[kx + 4]  * kw;
                    p5  += in_row[kx + 5]  * kw;
                    p6  += in_row[kx + 6]  * kw;
                    p7  += in_row[kx + 7]  * kw;
                    p8  += in_row[kx + 8]  * kw;
                    p9  += in_row[kx + 9]  * kw;
                    p10 += in_row[kx + 10] * kw;
                    p11 += in_row[kx + 11] * kw;
                    p12 += in_row[kx + 12] * kw;
                    p13 += in_row[kx + 13] * kw;
                    p14 += in_row[kx + 14] * kw;
                    p15 += in_row[kx + 15] * kw;
                }
            }

            out_row[ox]      = p0;
            out_row[ox + 1]  = p1;
            out_row[ox + 2]  = p2;
            out_row[ox + 3]  = p3;
            out_row[ox + 4]  = p4;
            out_row[ox + 5]  = p5;
            out_row[ox + 6]  = p6;
            out_row[ox + 7]  = p7;
            out_row[ox + 8]  = p8;
            out_row[ox + 9]  = p9;
            out_row[ox + 10] = p10;
            out_row[ox + 11] = p11;
            out_row[ox + 12] = p12;
            out_row[ox + 13] = p13;
            out_row[ox + 14] = p14;
            out_row[ox + 15] = p15;
        }

        for (; ox < W; ++ox) {
            float acc = 0.0f;

            for (int ky = 0; ky < K; ++ky) {
                const float* in_row = in + (oy + ky) * in_stride + ox;
                const float* ker_row = ker + ky * K;

                for (int kx = 0; kx < K; ++kx) {
                    acc += in_row[kx] * ker_row[kx];
                }
            }

            out_row[ox] = acc;
        }
    }
}

void conv_unroll20(const float* in, float* out, const float* ker,
                          int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oy = 0; oy < H; ++oy) {
        int ox = 0;
        float* out_row = out + oy * W;

        for (; ox + 20 <= W; ox += 20) {
            float p0  = 0.0f, p1  = 0.0f;
            float p2  = 0.0f, p3  = 0.0f;
            float p4  = 0.0f, p5  = 0.0f;
            float p6  = 0.0f, p7  = 0.0f;
            float p8  = 0.0f, p9  = 0.0f;
            float p10 = 0.0f, p11 = 0.0f;
            float p12 = 0.0f, p13 = 0.0f;
            float p14 = 0.0f, p15 = 0.0f;
            float p16 = 0.0f, p17 = 0.0f;
            float p18 = 0.0f, p19 = 0.0f;

            for (int ky = 0; ky < K; ++ky) {
                const float* in_row =
                    in + (oy + ky) * in_stride + ox;

                const float* ker_row =
                    ker + ky * K;

                for (int kx = 0; kx < K; ++kx) {
                    const float kw = ker_row[kx];

                    p0  += in_row[kx]      * kw;
                    p1  += in_row[kx + 1]  * kw;
                    p2  += in_row[kx + 2]  * kw;
                    p3  += in_row[kx + 3]  * kw;
                    p4  += in_row[kx + 4]  * kw;
                    p5  += in_row[kx + 5]  * kw;
                    p6  += in_row[kx + 6]  * kw;
                    p7  += in_row[kx + 7]  * kw;
                    p8  += in_row[kx + 8]  * kw;
                    p9  += in_row[kx + 9]  * kw;
                    p10 += in_row[kx + 10] * kw;
                    p11 += in_row[kx + 11] * kw;
                    p12 += in_row[kx + 12] * kw;
                    p13 += in_row[kx + 13] * kw;
                    p14 += in_row[kx + 14] * kw;
                    p15 += in_row[kx + 15] * kw;
                    p16 += in_row[kx + 16] * kw;
                    p17 += in_row[kx + 17] * kw;
                    p18 += in_row[kx + 18] * kw;
                    p19 += in_row[kx + 19] * kw;
                }
            }

            out_row[ox]      = p0;
            out_row[ox + 1]  = p1;
            out_row[ox + 2]  = p2;
            out_row[ox + 3]  = p3;
            out_row[ox + 4]  = p4;
            out_row[ox + 5]  = p5;
            out_row[ox + 6]  = p6;
            out_row[ox + 7]  = p7;
            out_row[ox + 8]  = p8;
            out_row[ox + 9]  = p9;
            out_row[ox + 10] = p10;
            out_row[ox + 11] = p11;
            out_row[ox + 12] = p12;
            out_row[ox + 13] = p13;
            out_row[ox + 14] = p14;
            out_row[ox + 15] = p15;
            out_row[ox + 16] = p16;
            out_row[ox + 17] = p17;
            out_row[ox + 18] = p18;
            out_row[ox + 19] = p19;
        }

        // Scalar cleanup for W % 20 remaining outputs
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

void conv_unroll24(const float* in, float* out, const float* ker,
                          int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oy = 0; oy < H; ++oy) {
        int ox = 0;
        float* out_row = out + oy * W;

        for (; ox + 24 <= W; ox += 24) {
            float p0  = 0.0f, p1  = 0.0f;
            float p2  = 0.0f, p3  = 0.0f;
            float p4  = 0.0f, p5  = 0.0f;
            float p6  = 0.0f, p7  = 0.0f;
            float p8  = 0.0f, p9  = 0.0f;
            float p10 = 0.0f, p11 = 0.0f;
            float p12 = 0.0f, p13 = 0.0f;
            float p14 = 0.0f, p15 = 0.0f;
            float p16 = 0.0f, p17 = 0.0f;
            float p18 = 0.0f, p19 = 0.0f;
            float p20 = 0.0f, p21 = 0.0f;
            float p22 = 0.0f, p23 = 0.0f;

            for (int ky = 0; ky < K; ++ky) {
                const float* in_row =
                    in + (oy + ky) * in_stride + ox;

                const float* ker_row =
                    ker + ky * K;

                for (int kx = 0; kx < K; ++kx) {
                    const float kw = ker_row[kx];

                    p0  += in_row[kx]      * kw;
                    p1  += in_row[kx + 1]  * kw;
                    p2  += in_row[kx + 2]  * kw;
                    p3  += in_row[kx + 3]  * kw;
                    p4  += in_row[kx + 4]  * kw;
                    p5  += in_row[kx + 5]  * kw;
                    p6  += in_row[kx + 6]  * kw;
                    p7  += in_row[kx + 7]  * kw;
                    p8  += in_row[kx + 8]  * kw;
                    p9  += in_row[kx + 9]  * kw;
                    p10 += in_row[kx + 10] * kw;
                    p11 += in_row[kx + 11] * kw;
                    p12 += in_row[kx + 12] * kw;
                    p13 += in_row[kx + 13] * kw;
                    p14 += in_row[kx + 14] * kw;
                    p15 += in_row[kx + 15] * kw;
                    p16 += in_row[kx + 16] * kw;
                    p17 += in_row[kx + 17] * kw;
                    p18 += in_row[kx + 18] * kw;
                    p19 += in_row[kx + 19] * kw;
                    p20 += in_row[kx + 20] * kw;
                    p21 += in_row[kx + 21] * kw;
                    p22 += in_row[kx + 22] * kw;
                    p23 += in_row[kx + 23] * kw;
                }
            }

            out_row[ox]      = p0;
            out_row[ox + 1]  = p1;
            out_row[ox + 2]  = p2;
            out_row[ox + 3]  = p3;
            out_row[ox + 4]  = p4;
            out_row[ox + 5]  = p5;
            out_row[ox + 6]  = p6;
            out_row[ox + 7]  = p7;
            out_row[ox + 8]  = p8;
            out_row[ox + 9]  = p9;
            out_row[ox + 10] = p10;
            out_row[ox + 11] = p11;
            out_row[ox + 12] = p12;
            out_row[ox + 13] = p13;
            out_row[ox + 14] = p14;
            out_row[ox + 15] = p15;
            out_row[ox + 16] = p16;
            out_row[ox + 17] = p17;
            out_row[ox + 18] = p18;
            out_row[ox + 19] = p19;
            out_row[ox + 20] = p20;
            out_row[ox + 21] = p21;
            out_row[ox + 22] = p22;
            out_row[ox + 23] = p23;
        }

        // Scalar cleanup for W % 24 remaining outputs
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

void conv_unroll28(const float* in, float* out, const float* ker,
                          int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oy = 0; oy < H; ++oy) {
        int ox = 0;
        float* out_row = out + oy * W;

        for (; ox + 28 <= W; ox += 28) {
            float p0  = 0.0f, p1  = 0.0f;
            float p2  = 0.0f, p3  = 0.0f;
            float p4  = 0.0f, p5  = 0.0f;
            float p6  = 0.0f, p7  = 0.0f;
            float p8  = 0.0f, p9  = 0.0f;
            float p10 = 0.0f, p11 = 0.0f;
            float p12 = 0.0f, p13 = 0.0f;
            float p14 = 0.0f, p15 = 0.0f;
            float p16 = 0.0f, p17 = 0.0f;
            float p18 = 0.0f, p19 = 0.0f;
            float p20 = 0.0f, p21 = 0.0f;
            float p22 = 0.0f, p23 = 0.0f;
            float p24 = 0.0f, p25 = 0.0f;
            float p26 = 0.0f, p27 = 0.0f;

            for (int ky = 0; ky < K; ++ky) {
                const float* in_row =
                    in + (oy + ky) * in_stride + ox;

                const float* ker_row =
                    ker + ky * K;

                for (int kx = 0; kx < K; ++kx) {
                    const float kw = ker_row[kx];

                    p0  += in_row[kx]      * kw;
                    p1  += in_row[kx + 1]  * kw;
                    p2  += in_row[kx + 2]  * kw;
                    p3  += in_row[kx + 3]  * kw;
                    p4  += in_row[kx + 4]  * kw;
                    p5  += in_row[kx + 5]  * kw;
                    p6  += in_row[kx + 6]  * kw;
                    p7  += in_row[kx + 7]  * kw;
                    p8  += in_row[kx + 8]  * kw;
                    p9  += in_row[kx + 9]  * kw;
                    p10 += in_row[kx + 10] * kw;
                    p11 += in_row[kx + 11] * kw;
                    p12 += in_row[kx + 12] * kw;
                    p13 += in_row[kx + 13] * kw;
                    p14 += in_row[kx + 14] * kw;
                    p15 += in_row[kx + 15] * kw;
                    p16 += in_row[kx + 16] * kw;
                    p17 += in_row[kx + 17] * kw;
                    p18 += in_row[kx + 18] * kw;
                    p19 += in_row[kx + 19] * kw;
                    p20 += in_row[kx + 20] * kw;
                    p21 += in_row[kx + 21] * kw;
                    p22 += in_row[kx + 22] * kw;
                    p23 += in_row[kx + 23] * kw;
                    p24 += in_row[kx + 24] * kw;
                    p25 += in_row[kx + 25] * kw;
                    p26 += in_row[kx + 26] * kw;
                    p27 += in_row[kx + 27] * kw;
                }
            }

            out_row[ox]      = p0;
            out_row[ox + 1]  = p1;
            out_row[ox + 2]  = p2;
            out_row[ox + 3]  = p3;
            out_row[ox + 4]  = p4;
            out_row[ox + 5]  = p5;
            out_row[ox + 6]  = p6;
            out_row[ox + 7]  = p7;
            out_row[ox + 8]  = p8;
            out_row[ox + 9]  = p9;
            out_row[ox + 10] = p10;
            out_row[ox + 11] = p11;
            out_row[ox + 12] = p12;
            out_row[ox + 13] = p13;
            out_row[ox + 14] = p14;
            out_row[ox + 15] = p15;
            out_row[ox + 16] = p16;
            out_row[ox + 17] = p17;
            out_row[ox + 18] = p18;
            out_row[ox + 19] = p19;
            out_row[ox + 20] = p20;
            out_row[ox + 21] = p21;
            out_row[ox + 22] = p22;
            out_row[ox + 23] = p23;
            out_row[ox + 24] = p24;
            out_row[ox + 25] = p25;
            out_row[ox + 26] = p26;
            out_row[ox + 27] = p27;
        }

        // Scalar cleanup for W % 28 remaining outputs
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


void conv_unroll32(const float* in, float* out, const float* ker,
                          int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oy = 0; oy < H; ++oy) {
        int ox = 0;
        float* out_row = out + oy * W;

        for (; ox + 32 <= W; ox += 32) {
            float p0  = 0.0f, p1  = 0.0f;
            float p2  = 0.0f, p3  = 0.0f;
            float p4  = 0.0f, p5  = 0.0f;
            float p6  = 0.0f, p7  = 0.0f;
            float p8  = 0.0f, p9  = 0.0f;
            float p10 = 0.0f, p11 = 0.0f;
            float p12 = 0.0f, p13 = 0.0f;
            float p14 = 0.0f, p15 = 0.0f;
            float p16 = 0.0f, p17 = 0.0f;
            float p18 = 0.0f, p19 = 0.0f;
            float p20 = 0.0f, p21 = 0.0f;
            float p22 = 0.0f, p23 = 0.0f;
            float p24 = 0.0f, p25 = 0.0f;
            float p26 = 0.0f, p27 = 0.0f;
            float p28 = 0.0f, p29 = 0.0f;
            float p30 = 0.0f, p31 = 0.0f;

            for (int ky = 0; ky < K; ++ky) {
                const float* in_row =
                    in + (oy + ky) * in_stride + ox;

                const float* ker_row =
                    ker + ky * K;

                for (int kx = 0; kx < K; ++kx) {
                    const float kw = ker_row[kx];

                    p0  += in_row[kx]      * kw;
                    p1  += in_row[kx + 1]  * kw;
                    p2  += in_row[kx + 2]  * kw;
                    p3  += in_row[kx + 3]  * kw;
                    p4  += in_row[kx + 4]  * kw;
                    p5  += in_row[kx + 5]  * kw;
                    p6  += in_row[kx + 6]  * kw;
                    p7  += in_row[kx + 7]  * kw;
                    p8  += in_row[kx + 8]  * kw;
                    p9  += in_row[kx + 9]  * kw;
                    p10 += in_row[kx + 10] * kw;
                    p11 += in_row[kx + 11] * kw;
                    p12 += in_row[kx + 12] * kw;
                    p13 += in_row[kx + 13] * kw;
                    p14 += in_row[kx + 14] * kw;
                    p15 += in_row[kx + 15] * kw;
                    p16 += in_row[kx + 16] * kw;
                    p17 += in_row[kx + 17] * kw;
                    p18 += in_row[kx + 18] * kw;
                    p19 += in_row[kx + 19] * kw;
                    p20 += in_row[kx + 20] * kw;
                    p21 += in_row[kx + 21] * kw;
                    p22 += in_row[kx + 22] * kw;
                    p23 += in_row[kx + 23] * kw;
                    p24 += in_row[kx + 24] * kw;
                    p25 += in_row[kx + 25] * kw;
                    p26 += in_row[kx + 26] * kw;
                    p27 += in_row[kx + 27] * kw;
                    p28 += in_row[kx + 28] * kw;
                    p29 += in_row[kx + 29] * kw;
                    p30 += in_row[kx + 30] * kw;
                    p31 += in_row[kx + 31] * kw;
                }
            }

            out_row[ox]      = p0;
            out_row[ox + 1]  = p1;
            out_row[ox + 2]  = p2;
            out_row[ox + 3]  = p3;
            out_row[ox + 4]  = p4;
            out_row[ox + 5]  = p5;
            out_row[ox + 6]  = p6;
            out_row[ox + 7]  = p7;
            out_row[ox + 8]  = p8;
            out_row[ox + 9]  = p9;
            out_row[ox + 10] = p10;
            out_row[ox + 11] = p11;
            out_row[ox + 12] = p12;
            out_row[ox + 13] = p13;
            out_row[ox + 14] = p14;
            out_row[ox + 15] = p15;
            out_row[ox + 16] = p16;
            out_row[ox + 17] = p17;
            out_row[ox + 18] = p18;
            out_row[ox + 19] = p19;
            out_row[ox + 20] = p20;
            out_row[ox + 21] = p21;
            out_row[ox + 22] = p22;
            out_row[ox + 23] = p23;
            out_row[ox + 24] = p24;
            out_row[ox + 25] = p25;
            out_row[ox + 26] = p26;
            out_row[ox + 27] = p27;
            out_row[ox + 28] = p28;
            out_row[ox + 29] = p29;
            out_row[ox + 30] = p30;
            out_row[ox + 31] = p31;
        }

        // Scalar cleanup for W % 32 remaining outputs
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

