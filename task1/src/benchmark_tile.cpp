#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "convolution.h"
#include "utils.h"

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K, int Tx, int Ty);

namespace {

struct Buffers {
    float* image;
    float* input;
    float* kernel;
    float* output;
    float* reference;
};

[[noreturn]] void usage(const char* program) {
    std::fprintf(stderr,
        "Usage:\n"
        "  %s time H W K Tx Ty runs\n"
        "  %s kernel naive|tile H W K Tx Ty iterations\n",
        program, program);
    std::exit(2);
}

int parse_positive(const char* text, const char* name) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (*text == '\0' || *end != '\0' || value <= 0 ||
        value > std::numeric_limits<int>::max()) {
        std::fprintf(stderr, "Invalid %s: %s\n", name, text);
        std::exit(2);
    }
    return static_cast<int>(value);
}

Buffers allocate_buffers(int H, int W, int K) {
    const std::size_t image_count = static_cast<std::size_t>(H) * W;
    const std::size_t kernel_count = static_cast<std::size_t>(K) * K;

    Buffers b{};
    b.image = pa1::alloc_floats(image_count);
    b.kernel = pa1::alloc_floats(kernel_count);
    b.output = pa1::alloc_floats(image_count);
    b.reference = pa1::alloc_floats(image_count);
    pa1::fill_random(b.image, image_count, 1234);
    pa1::fill_random(b.kernel, kernel_count, 1235);
    b.input = pa1::make_padded(b.image, H, W, K);
    return b;
}

void free_buffers(Buffers& b) {
    pa1::free_floats(b.input);
    pa1::free_floats(b.image);
    pa1::free_floats(b.kernel);
    pa1::free_floats(b.output);
    pa1::free_floats(b.reference);
}

bool correct(const Buffers& b, int H, int W, int K, int Tx, int Ty) {
    const std::size_t count = static_cast<std::size_t>(H) * W;
    std::fill_n(b.output, count, std::numeric_limits<float>::quiet_NaN());
    conv_naive(b.input, b.reference, b.kernel, H, W, K);
    conv_tile(b.input, b.output, b.kernel, H, W, K, Tx, Ty);

    for (std::size_t i = 0; i < count; ++i) {
        const float difference = std::fabs(b.output[i] - b.reference[i]);
        if (!std::isfinite(difference) || difference > 1.0e-3f)
            return false;
    }
    return true;
}

template <class Function>
double measure_once_ms(Function&& function) {
    const auto begin = std::chrono::steady_clock::now();
    function();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

double median(std::vector<double>& values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    if (values.size() % 2 != 0)
        return values[middle];
    return (values[middle - 1] + values[middle]) * 0.5;
}

int run_time_mode(int argc, char** argv) {
    if (argc != 9)
        usage(argv[0]);

    const int H = parse_positive(argv[2], "H");
    const int W = parse_positive(argv[3], "W");
    const int K = parse_positive(argv[4], "K");
    const int Tx = parse_positive(argv[5], "Tx");
    const int Ty = parse_positive(argv[6], "Ty");
    const int runs = parse_positive(argv[7], "runs");
    (void)argv[8];

    if ((K & 1) == 0) {
        std::fprintf(stderr, "K must be odd.\n");
        return 2;
    }

    Buffers b = allocate_buffers(H, W, K);
    if (!correct(b, H, W, K, Tx, Ty)) {
        std::printf("correct=no\n");
        free_buffers(b);
        return 1;
    }

    for (int i = 0; i < 2; ++i) {
        conv_naive(b.input, b.output, b.kernel, H, W, K);
        conv_tile(b.input, b.output, b.kernel, H, W, K, Tx, Ty);
    }

    std::vector<double> naive_times;
    std::vector<double> tile_times;
    naive_times.reserve(runs);
    tile_times.reserve(runs);

    for (int run = 0; run < runs; ++run) {
        if ((run & 1) == 0) {
            naive_times.push_back(measure_once_ms([&] {
                conv_naive(b.input, b.output, b.kernel, H, W, K);
            }));
            tile_times.push_back(measure_once_ms([&] {
                conv_tile(b.input, b.output, b.kernel, H, W, K, Tx, Ty);
            }));
        } else {
            tile_times.push_back(measure_once_ms([&] {
                conv_tile(b.input, b.output, b.kernel, H, W, K, Tx, Ty);
            }));
            naive_times.push_back(measure_once_ms([&] {
                conv_naive(b.input, b.output, b.kernel, H, W, K);
            }));
        }
    }

    const double naive_ms = median(naive_times);
    const double tile_ms = median(tile_times);
    std::printf("correct=yes naive_ms=%.9f tile_ms=%.9f speedup=%.9f\n",
                naive_ms, tile_ms, naive_ms / tile_ms);
    free_buffers(b);
    return 0;
}

int run_kernel_mode(int argc, char** argv) {
    if (argc != 10)
        usage(argv[0]);

    const char* implementation = argv[2];
    const int H = parse_positive(argv[3], "H");
    const int W = parse_positive(argv[4], "W");
    const int K = parse_positive(argv[5], "K");
    const int Tx = parse_positive(argv[6], "Tx");
    const int Ty = parse_positive(argv[7], "Ty");
    const int iterations = parse_positive(argv[8], "iterations");
    (void)argv[9];

    if ((K & 1) == 0) {
        std::fprintf(stderr, "K must be odd.\n");
        return 2;
    }

    const bool use_naive = std::strcmp(implementation, "naive") == 0;
    const bool use_tile = std::strcmp(implementation, "tile") == 0;
    if (!use_naive && !use_tile)
        usage(argv[0]);

    Buffers b = allocate_buffers(H, W, K);
    for (int iteration = 0; iteration < iterations; ++iteration) {
        if (use_naive)
            conv_naive(b.input, b.output, b.kernel, H, W, K);
        else
            conv_tile(b.input, b.output, b.kernel, H, W, K, Tx, Ty);
    }

    double checksum = 0.0;
    const std::size_t count = static_cast<std::size_t>(H) * W;
    for (std::size_t i = 0; i < count; ++i)
        checksum += b.output[i];
    std::printf("checksum=%.9e\n", checksum);
    free_buffers(b);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2)
        usage(argv[0]);
    if (std::strcmp(argv[1], "time") == 0)
        return run_time_mode(argc, argv);
    if (std::strcmp(argv[1], "kernel") == 0)
        return run_kernel_mode(argc, argv);
    usage(argv[0]);
}
