#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

#include "convolution.h"
#include "utils.h"

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
        "  %s time H W K runs\n"
        "  %s kernel naive|reorder H W K iterations\n",
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

bool reorder_is_correct(Buffers& b, int H, int W, int K) {
    const std::size_t count = static_cast<std::size_t>(H) * W;
    conv_naive(b.input, b.reference, b.kernel, H, W, K);
    std::fill_n(b.output, count, std::numeric_limits<float>::quiet_NaN());
    conv_reorder(b.input, b.output, b.kernel, H, W, K);

    for (std::size_t i = 0; i < count; ++i) {
        const float difference = std::fabs(b.output[i] - b.reference[i]);
        if (!std::isfinite(difference) || difference > 1.0e-3f)
            return false;
    }
    return true;
}

template <class Function>
double measure_once_ms(Function&& function) {
    const auto start = std::chrono::steady_clock::now();
    function();
    const auto finish = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(finish - start).count();
}

double mean(const std::vector<double>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    if ((values.size() & 1U) != 0)
        return values[middle];
    return 0.5 * (values[middle - 1] + values[middle]);
}

double coefficient_of_variation(const std::vector<double>& values) {
    const double average = mean(values);
    double squared_difference_sum = 0.0;
    for (double value : values) {
        const double difference = value - average;
        squared_difference_sum += difference * difference;
    }
    const double standard_deviation =
        std::sqrt(squared_difference_sum / values.size());
    return 100.0 * standard_deviation / average;
}

int run_time_mode(int argc, char** argv) {
    if (argc != 7)
        usage(argv[0]);

    const int H = parse_positive(argv[2], "H");
    const int W = parse_positive(argv[3], "W");
    const int K = parse_positive(argv[4], "K");
    const int runs = parse_positive(argv[5], "runs");
    (void)argv[6];

    if ((K & 1) == 0) {
        std::fprintf(stderr, "K must be odd.\n");
        return 2;
    }

    Buffers b = allocate_buffers(H, W, K);
    if (!reorder_is_correct(b, H, W, K)) {
        std::printf("correct=no\n");
        free_buffers(b);
        return 1;
    }

    for (int warmup = 0; warmup < 2; ++warmup) {
        conv_naive(b.input, b.output, b.kernel, H, W, K);
        conv_reorder(b.input, b.output, b.kernel, H, W, K);
    }

    std::vector<double> naive_times;
    std::vector<double> reorder_times;
    naive_times.reserve(runs);
    reorder_times.reserve(runs);

    for (int run = 0; run < runs; ++run) {
        if ((run & 1) == 0) {
            naive_times.push_back(measure_once_ms([&] {
                conv_naive(b.input, b.output, b.kernel, H, W, K);
            }));
            reorder_times.push_back(measure_once_ms([&] {
                conv_reorder(b.input, b.output, b.kernel, H, W, K);
            }));
        } else {
            reorder_times.push_back(measure_once_ms([&] {
                conv_reorder(b.input, b.output, b.kernel, H, W, K);
            }));
            naive_times.push_back(measure_once_ms([&] {
                conv_naive(b.input, b.output, b.kernel, H, W, K);
            }));
        }
    }

    const double naive_mean = mean(naive_times);
    const double reorder_mean = mean(reorder_times);
    std::printf(
        "correct=yes naive_mean_ms=%.9f reorder_mean_ms=%.9f "
        "naive_median_ms=%.9f reorder_median_ms=%.9f "
        "naive_cv=%.6f reorder_cv=%.6f speedup=%.9f\n",
        naive_mean, reorder_mean, median(naive_times), median(reorder_times),
        coefficient_of_variation(naive_times),
        coefficient_of_variation(reorder_times), naive_mean / reorder_mean);

    free_buffers(b);
    return 0;
}

int run_kernel_mode(int argc, char** argv) {
    if (argc != 8)
        usage(argv[0]);

    const char* implementation = argv[2];
    const int H = parse_positive(argv[3], "H");
    const int W = parse_positive(argv[4], "W");
    const int K = parse_positive(argv[5], "K");
    const int iterations = parse_positive(argv[6], "iterations");
    (void)argv[7];

    if ((K & 1) == 0) {
        std::fprintf(stderr, "K must be odd.\n");
        return 2;
    }

    ConvFn function = nullptr;
    if (std::strcmp(implementation, "naive") == 0)
        function = conv_naive;
    else if (std::strcmp(implementation, "reorder") == 0)
        function = conv_reorder;
    else
        usage(argv[0]);

    Buffers b = allocate_buffers(H, W, K);
    for (int iteration = 0; iteration < iterations; ++iteration)
        function(b.input, b.output, b.kernel, H, W, K);

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
