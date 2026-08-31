// conv_unroll.cpp  STAGE 2: LOOP UNROLLING
#include "convolution.h"

void conv_unroll(const float* in, float* out, const float* ker,
                 int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride

    for (int oy = 0; oy < H; ++oy) {
        for (int ox = 0; ox < W; ++ox) {
            //naive implementation for fixed K=3
            float psum00 = 0.0f, psum01 = 0.0f, psum02 = 0.0f;
            float psum10 = 0.0f, psum11 = 0.0f, psum12 = 0.0f;
            float psum20 = 0.0f, psum21 = 0.0f, psum22 = 0.0f;

            int ky = 0;
            int kx = 0;

            psum00 = in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
            psum01 = in[(oy + ky) * in_stride + (ox + kx + 1)] * ker[ky * K + kx + 1];
            psum02 = in[(oy + ky) * in_stride + (ox + kx + 2)] * ker[ky * K + kx + 2];

            psum10 = in[(oy + ky + 1) * in_stride + (ox + kx)] * ker[(ky + 1) * K + kx];
            psum11 = in[(oy + ky + 1) * in_stride + (ox + kx + 1)] * ker[(ky + 1) * K + kx + 1];
            psum12 = in[(oy + ky + 1) * in_stride + (ox + kx + 2)] * ker[(ky + 1) * K + kx + 2];

            psum20 = in[(oy + ky + 2) * in_stride + (ox + kx)] * ker[(ky + 2) * K + kx];
            psum21 = in[(oy + ky + 2) * in_stride + (ox + kx + 1)] * ker[(ky + 2) * K + kx + 1];
            psum22 = in[(oy + ky + 2) * in_stride + (ox + kx + 2)] * ker[(ky + 2) * K + kx + 2];

            out[oy * W + ox] = psum00 + psum01 + psum02 +
                            psum10 + psum11 + psum12 +
                            psum20 + psum21 + psum22;
        }
    }
}
