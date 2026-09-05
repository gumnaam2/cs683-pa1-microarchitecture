// num_instr.cpp
//
// Runs exactly ONE convolution implementation repeatedly so that
// `perf stat` can measure its retired instruction count.
//
// Usage:
//   ./bin/num_instr naive
//   ./bin/num_instr 128
//   ./bin/num_instr 256
//   ./bin/num_instr 512

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "convolution.h"
#include "utils.h"


// If these are not already declared in convolution.h:
void conv_simd128(const float* in, float* out, const float* ker,
                  int H, int W, int K);

void conv_simd256(const float* in, float* out, const float* ker,
                  int H, int W, int K);

void conv_simd512(const float* in, float* out, const float* ker,
                  int H, int W, int K);


using KernelFn =
    void (*)(const float*, float*, const float*, int, int, int);


static constexpr int kDefaultH = 2048;
static constexpr int kDefaultW = 2048;
static constexpr int kDefaultK = 3;
static constexpr unsigned kDefaultSeed = 1234u;

// perf measures all 100 calls together.
// Divide the reported instruction count by 100 afterward.
static constexpr int kReps = 100;


static void usage(const char* prog)
{
    std::printf(
        "Usage:\n"
        "  %s naive|128|256|512\n"
        "  %s naive|128|256|512 H W K [seed]\n",
        prog, prog
    );
}


int main(int argc, char** argv)
{
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    // ---------------------------------------------------------
    // Choose which implementation will be measured
    // ---------------------------------------------------------

    KernelFn kernel = nullptr;

    if (std::strcmp(argv[1], "naive") == 0) {

        kernel = conv_naive;

    } else if (std::strcmp(argv[1], "128") == 0) {

        kernel = conv_simd128;

    } else if (std::strcmp(argv[1], "256") == 0) {

        kernel = conv_simd256;

    } else if (std::strcmp(argv[1], "512") == 0) {

        kernel = conv_simd512;

    } else {

        std::fprintf(stderr,
                     "Error: unknown implementation '%s'\n",
                     argv[1]);

        usage(argv[0]);
        return 1;
    }


    // ---------------------------------------------------------
    // Default workload
    // ---------------------------------------------------------

    int H = kDefaultH;
    int W = kDefaultW;
    int K = kDefaultK;

    unsigned seed = kDefaultSeed;


    float* img =
        pa1::alloc_floats(
            static_cast<std::size_t>(H) * W
        );

    // Kernel: K x K
    float* ker =
        pa1::alloc_floats(
            static_cast<std::size_t>(K) * K
        );

    // Output: H x W
    float* out =
        pa1::alloc_floats(
            static_cast<std::size_t>(H) * W
        );


    // Random image
    pa1::fill_random(
        img,
        static_cast<std::size_t>(H) * W,
        seed
    );


    // Random kernel, using seed + 1 just like main.cpp
    pa1::fill_random(
        ker,
        static_cast<std::size_t>(K) * K,
        seed + 1u
    );

    float* in =
        pa1::make_padded(
            img,
            H,
            W,
            K
        );


    for (int r = 0; r < kReps; ++r) {

        kernel(
            in,
            out,
            ker,
            H,
            W,
            K
        );
    }


    // ---------------------------------------------------------
    // Consume the output
    //
    // Prevent optimizer from deciding that the convolution
    // results are unused.
    // ---------------------------------------------------------

    double checksum = 0.0;

    for (int i = 0; i < H * W; ++i) {
        checksum += out[i];
    }


    std::printf(
        "implementation = %s\n"
        "H = %d, W = %d, K = %d\n"
        "seed = %u\n"
        "repetitions = %d\n"
        "checksum = %.9f\n",
        argv[1],
        H,
        W,
        K,
        seed,
        kReps,
        checksum
    );


    // ---------------------------------------------------------
    // Cleanup
    // ---------------------------------------------------------

    pa1::free_floats(in);
    pa1::free_floats(img);
    pa1::free_floats(ker);
    pa1::free_floats(out);

    return 0;
}