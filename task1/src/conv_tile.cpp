// conv_tile.cpp  STAGE 3: CACHE TILING

#include "convolution.h"
#include <algorithm>

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K, int Tx, int Ty) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride
    float acc;
    int ox_end, oy_end;
    
    for (int ty = 0; ty < H; ty += Ty){
        oy_end = std::min(ty + Ty, H);
        for (int tx = 0; tx < W; tx += Tx){
            ox_end = std::min(tx + Tx, W);
            //store to ofmap[ty -> ty + Ty - 1][tx -> tx + Tx - 1]
            //loads ifmap[ty -> ty + Ty + ky - 1][tx -> tx + Tx + kx - 1]
            //loads entire kernel
            //working set: Ty*Tx + (Ty + k - 1)*(Tx + k - 1) + k^2
            for (int oy = ty; oy < oy_end; ++oy) {
                float* out_row = out + oy * W;
                for (int ox = tx; ox < ox_end; ++ox) {
                    acc = 0.0f;
                    //load ifmap[oy -> (oy + ky)][ox -> ox + kx]
                    //H,W,k = 2048, 2048, 3 : stride = 2050
                    //load entire kernel: K^2 loads
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
    }

    // float kW;
    // for (int ky = 0; ky < K; ++ky) {
    //     for (int kx = 0; kx < K; ++kx) {
    //         kW = ker[ky * K + kx];
    //         for (int oy = ty; oy < ty+Ty; ++oy) {
    //             for (int ox = tx; ox < tx + Tx; ++ox) {
    //                 out[oy * W + ox] += in[(oy + ky) * in_stride + (ox + kx)] * kW;;
    //             }
    //         }
    //     }
    // }


    // ky, kx, oy, ox
    // for (int oy = 0; oy < H; ++oy) {
    //     for (int ox = 0; ox < W; ++ox) {
    //         out[oy * W + ox] = 0;
    //     }
    // }
    // for (int ty = 0; ty < H; ty += T){
    //     for (int tx = 0; tx < W; tx += T){
    //         for (int oy = ty; oy < ty+T; ++oy) {
    //             for (int ox = tx; ox < tx + T; ++ox) {
    //                 float acc = 0.0f;
    //                 for (int ky = 0; ky < K; ++ky) {
    //                     for (int kx = 0; kx < K; ++kx) {
    //                         acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
    //                     }
    //                 }
    //                 out[oy * W + ox] = acc;
    //             }
    //         }
    //     }
    // }
}

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    int default_Tx = 2048;
    int default_Ty = 128;
    conv_tile(in, out, ker, H, W, K, default_Tx, default_Ty);
}
