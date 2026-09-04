// conv_unroll.cpp  STAGE 2: LOOP UNROLLING
#include "convolution.h"
#define kx(k, K) ((k) % (K))
#define ky(k, K) ((k) / (K))

void conv_unroll(const float* in, float* out, const float* ker,
                 int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride

    
    for (int oy = 0; oy < H; ++oy) {
        for (int ox = 0; ox < W; ++ox) {
            // float acc = 0.0f;
            int ky = 0; int kx = 0;
            float psum0 = 0.0f, psum1 = 0.0f, psum2 = 0.0f, psum3 = 0.0f, psum4 = 0.0f;
            float acc = 0.0f;
            int k = 0;
                        
            for (int ky = 0; ky < K; ++ky) {
                int kx = 0;
                for (; kx + 3 <= K; kx += 3) {
                    psum0 += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                    psum1 += in[(oy + ky) * in_stride + (ox + kx+1)] * ker[(ky) * K + kx+1];
                    psum2 += in[(oy + ky) * in_stride + (ox + kx+2)] * ker[(ky) * K + kx+2];
                }

                for (; kx < K; kx += 1){
                    acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                }
            }
            out[oy * W + ox] = psum0 + psum1 + psum2 + psum3 + psum4 + acc;
        }
    }
}

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