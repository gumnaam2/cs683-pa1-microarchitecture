// conv_tile.cpp  STAGE 3: CACHE TILING

#include "convolution.h"

//There is notable variation between P-cores and E-cores for this task
//In E-cores, the performance is improved by ~15%. In P-cores, it is worsened by ~20% for the same tile size
//It is likely due to branching overhead and (from data) increased L1-icache misses - not immediately clear why

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride
    const int Tx = 64; //Tile size - x dirn
    const int Ty = 64; //Tile size - y dirn
    float kW;

    // ky, kx, oy, ox
    for (int oy = 0; oy < H; ++oy) {
        for (int ox = 0; ox < W; ++ox) {
            out[oy * W + ox] = 0;
        }
    }

    for (int ty = 0; ty < H; ty += Ty){
        for (int tx = 0; tx < W; tx += Tx){
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    kW = ker[ky * K + kx];
                    for (int oy = ty; oy < ty+Ty; ++oy) {
                        for (int ox = tx; ox < tx + Tx; ++ox) {
                            out[oy * W + ox] += in[(oy + ky) * in_stride + (ox + kx)] * kW;;
                        }
                    }
                }
            }
        }
    }

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