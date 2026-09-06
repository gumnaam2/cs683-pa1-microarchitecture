// conv_unroll.cpp  STAGE 2: LOOP UNROLLING
#include "convolution.h"

void conv_unroll2(const float* in, float* out, const float* ker,
                 int H, int W, int K);
void conv_unroll4(const float* in, float* out, const float* ker,
                 int H, int W, int K);
void conv_unroll8(const float* in, float* out, const float* ker,
                 int H, int W, int K);

void conv_unroll(const float* in, float* out, const float* ker,
                 int H, int W, int K) {
    conv_unroll8(in, out, ker, H, W, K);
}


void conv_unroll4(const float* in, float* out, const float* ker,
                 int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride

    for (int oy = 0; oy < H; ++oy) {
        int ox = 0;
        float* out_row = out + oy * W;

        for (; ox + 4 <= W; ox += 4) {
            float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;

            for (int ky = 0; ky < K; ++ky) {
                const float* in_row = in + (oy + ky) * in_stride + ox;
                const float* ker_row = ker + ky * K;

                for (int kx = 0; kx < K; ++kx) {
                    const float kw = ker_row[kx];

                    p0 += in_row[kx] * kw;
                    p1 += in_row[kx + 1] * kw;
                    p2 += in_row[kx + 2] * kw;
                    p3 += in_row[kx + 3] * kw;
                }
            }

            out_row[ox] = p0;
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

void conv_unroll2(const float* in, float* out, const float* ker,
                  int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride

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

                    p0 += in_row[kx] * kw;
                    p1 += in_row[kx + 1] * kw;
                }
            }

            out_row[ox]     = p0;
            out_row[ox + 1] = p1;
        }

        // Handle remaining element if W is not divisible by 2
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
    const int in_stride = W + 2 * p;  // padded row stride

    for (int oy = 0; oy < H; ++oy) {
        int ox = 0;
        float* out_row = out + oy * W;

        for (; ox + 8 <= W; ox += 8) {
            float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
            float p4 = 0.0f, p5 = 0.0f, p6 = 0.0f, p7 = 0.0f;

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

        // Handle remaining elements if W is not divisible by 8
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

// void conv_unroll(const float* in, float* out, const float* ker,
//                  int H, int W, int K) {
//     const int p = K / 2;
//     const int in_stride = W + 2 * p;  // padded row stride

    
//     for (int oy = 0; oy < H; ++oy) {
//         for (int ox = 0; ox < W; ++ox) {
//             // int ky = 0; int kx = 0;
//             float psum0 = 0.0f, psum1 = 0.0f, psum2 = 0.0f;
//             float acc = 0.0f;
//             // int k = 0;
                        
//             for (int ky = 0; ky < K; ++ky) {
//                 const float* in_row = in + (oy + ky) * in_stride + ox;
//                 const float* ker_row = ker + ky * K;

//                 int kx = 0;
//                 for (; kx + 3 <= K; kx += 3) {
//                     psum0 += in_row[kx] * ker_row[kx];
//                     psum1 += in_row[kx+1] * ker_row[kx+1];
//                     psum2 += in_row[kx+2] * ker_row[kx+2];
//                 }

//                 for (; kx < K; kx += 1){
//                     acc += in_row[kx] * ker_row[kx];
//                 }
//             }
//             out[oy * W + ox] = psum0 + psum1 + psum2 + acc;
//         }
//     }

//     return;
// }

// for (int kx = 0; kx < K; ++kx) {
//     psum0 += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
//     psum1 += in[(oy + ky + 1) * in_stride + (ox + kx)] * ker[(ky+1) * K + kx];
//     psum2 += in[(oy + ky + 2) * in_stride + (ox + kx)] * ker[(ky+2) * K + kx];
// }

// void conv_unroll(const float* in, float* out, const float* ker,
//                  int H, int W, int K) {
//     const int p = K / 2;
//     const int in_stride = W + 2 * p;  // padded row stride

//     for (int oy = 0; oy < H; ++oy) {
//         for (int ox = 0; ox < W; ++ox) {
//             out[oy * W + ox] = 0.0;
//             int k = 0;
//             for (; k + 8 <= K*K; k += 8) {
//                 float psum0 = 0.0f, psum1 = 0.0f, psum2 = 0.0f;
//                 float psum3 = 0.0f, psum4 = 0.0f, psum5 = 0.0f;
//                 float psum6 = 0.0f, psum7 = 0.0f;

//                 psum0 = in[(oy + ky(k, K)) * in_stride + (ox + kx(k, K))] * ker[ky(k, K) * K + kx(k, K)];
//                 psum1 = in[(oy + ky(k + 1, K)) * in_stride + (ox + kx(k + 1, K))] * ker[ky(k + 1, K) * K + kx(k + 1, K)];
//                 psum2 = in[(oy + ky(k + 2, K)) * in_stride + (ox + kx(k + 2, K))] * ker[ky(k + 2, K) * K + kx(k + 2, K)];

//                 psum3 = in[(oy + ky(k + 3, K)) * in_stride + (ox + kx(k + 3, K))] * ker[ky(k + 3, K) * K + kx(k + 3, K)];
//                 psum4 = in[(oy + ky(k + 4, K)) * in_stride + (ox + kx(k + 4, K))] * ker[ky(k + 4, K) * K + kx(k + 4, K)];
//                 psum5 = in[(oy + ky(k + 5, K)) * in_stride + (ox + kx(k + 5, K))] * ker[ky(k + 5, K) * K + kx(k + 5, K)];

//                 psum6 = in[(oy + ky(k + 6, K)) * in_stride + (ox + kx(k + 6, K))] * ker[ky(k + 6, K) * K + kx(k + 6, K)];
//                 psum7 = in[(oy + ky(k + 7, K)) * in_stride + (ox + kx(k + 7, K))] * ker[ky(k + 7, K) * K + kx(k + 7, K)];

//                 out[oy * W + ox] += psum0 + psum1 + psum2 +
//                                 psum3 + psum4 + psum5 +
//                                 psum6 + psum7;
//             }
//             float psum;
//             for (; k < K*K; k += 1){
//                 int ky = k /  K;
//                 int kx = k % K;
//                 psum = in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
//                 out[oy * W + ox] += psum;
//             }
//         }
//     }
// }


// for (int ky = 0; ky < K; ++ky) {
//     int kx = 0;
//     for (; kx + 3 <= K; kx += 3) {
//         psum0 += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
//         psum1 += in[(oy + ky) * in_stride + (ox + kx+1)] * ker[(ky) * K + kx+1];
//         psum2 += in[(oy + ky) * in_stride + (ox + kx+2)] * ker[(ky) * K + kx+2];
//     }
    
//     for (; kx < K; kx += 1){
//         acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
//     }
// }


// int kxarr[K*K];
// int kyarr[K*K];

// for (int ky = 0; ky < K; ++ky) {
//     for (int kx = 0; kx < K; ++kx) {
//         kxarr[ky*K + kx] = kx;
//         kyarr[ky*K + kx] = ky;
//     }
// }

// for (; k + 5 <= K*K; k += 5) {
//                 psum0 += in[(oy + kyarr[k]) * in_stride + (ox + kxarr[k])] * ker[kyarr[k] * K + kxarr[k]];
//                 psum1 += in[(oy + kyarr[k+1]) * in_stride + (ox + kxarr[k+1])] * ker[(kyarr[k+1]) * K + kxarr[k+1]];
//                 psum2 += in[(oy + kyarr[k+2]) * in_stride + (ox + kxarr[k+2])] * ker[(kyarr[k+2]) * K + kxarr[k+2]];
//                 psum3 += in[(oy + kyarr[k+3]) * in_stride + (ox + kxarr[k+3])] * ker[(kyarr[k+3]) * K + kxarr[k+3]];
//                 psum4 += in[(oy + kyarr[k+4]) * in_stride + (ox + kxarr[k+4])] * ker[(kyarr[k+4]) * K + kxarr[k+4]];
//             }
// for (; k < K*K; k++){
//                 acc += in[(oy + kyarr[k]) * in_stride + (ox + kxarr[k])] * ker[(kyarr[k]) * K + kxarr[k]];
//             }


// for (int kx = 0; kx < K; ++kx) {
//     int ky = 0;
//     for (; ky + 3 <= K; ky += 3) {
//         psum0 += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
//         psum1 += in[(oy + ky+1) * in_stride + (ox + kx)] * ker[(ky+1) * K + kx];
//         psum2 += in[(oy + ky+2) * in_stride + (ox + kx)] * ker[(ky+2) * K + kx];
//     }
    
//     for (; ky < K; ky += 1){
//         acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
//     }
// }