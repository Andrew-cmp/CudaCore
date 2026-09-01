#include <cuda_runtime.h>

#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using GemmLauncher = void (*)(
    int, int, int, float*, float*, float*, cudaStream_t);

void launch_gemm_v0(int, int, int, float*, float*, float*, cudaStream_t);
void launch_gemm_test(int, int, int, float*, float*, float*, cudaStream_t);
void launch_gemm_v1(int, int, int, float*, float*, float*, cudaStream_t);
void launch_gemm_v2(int, int, int, float*, float*, float*, cudaStream_t);
void launch_gemm_v3(int, int, int, float*, float*, float*, cudaStream_t);
void launch_gemm_v4(int, int, int, float*, float*, float*, cudaStream_t);
void launch_gemm_v5(int, int, int, float*, float*, float*, cudaStream_t);
void launch_gemm_v2_warp_tiling(
    int, int, int, float*, float*, float*, cudaStream_t);
void launch_gemm_v2_warp_tiling_swizzle(
    int, int, int, float*, float*, float*, cudaStream_t);
void launch_gemm_v2_warp_tiling_swizzle_rw(
    int, int, int, float*, float*, float*, cudaStream_t);

namespace {

struct KernelEntry {
    const char* name;
    GemmLauncher launcher;
};

constexpr KernelEntry kKernels[] = {
    {"gemm_v0", launch_gemm_v0},
    {"gemm_v1", launch_gemm_v1},
    {"gemm_v2", launch_gemm_v2},
    {"gemm_v3", launch_gemm_v3},
    {"gemm_v4", launch_gemm_v4},
    {"gemm_v5", launch_gemm_v5},
    {"gemm_v2_warp_tiling", launch_gemm_v2_warp_tiling},
    {"gemm_v2_warp_tiling_swizzle", launch_gemm_v2_warp_tiling_swizzle},
    {"gemm_test", launch_gemm_test},
    {"gemm_v2_warp_tiling_swizzle_rw", launch_gemm_v2_warp_tiling_swizzle_rw},
};

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", operation,
                     cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

int parse_integer(const char* value, const char* name, int minimum) {
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed < minimum ||
        parsed > INT_MAX) {
        std::fprintf(stderr, "%s must be an integer >= %d\n", name, minimum);
        std::exit(EXIT_FAILURE);
    }
    return static_cast<int>(parsed);
}

const KernelEntry* find_kernel(const char* name) {
    for (const KernelEntry& kernel : kKernels) {
        if (std::strcmp(kernel.name, name) == 0) {
            return &kernel;
        }
    }
    return nullptr;
}

void print_usage(const char* program) {
    std::fprintf(
        stderr,
        "Usage: %s [kernel M N K warmup repeat]\n"
        "Kernels: gemm_v0 gemm_v1 gemm_v2 gemm_v3 gemm_v4 gemm_v5 "
        "gemm_v2_warp_tiling gemm_v2_warp_tiling_swizzle gemm_test gemm_v2_warp_tiling_swizzle_rw\n",
        program);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 1 && argc != 7) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    const char* kernel_name = argc == 7 ? argv[1] : "gemm_v3";
    const int M = argc == 7 ? parse_integer(argv[2], "M", 1) : 4096;
    const int N = argc == 7 ? parse_integer(argv[3], "N", 1) : 4096;
    const int K = argc == 7 ? parse_integer(argv[4], "K", 1) : 4096;
    const int warmup = argc == 7 ? parse_integer(argv[5], "warmup", 0) : 5;
    const int repeat = argc == 7 ? parse_integer(argv[6], "repeat", 1) : 1;

    const KernelEntry* kernel = find_kernel(kernel_name);
    if (kernel == nullptr) {
        std::fprintf(stderr, "Unknown kernel: %s\n", kernel_name);
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    const size_t a_bytes =
        static_cast<size_t>(M) * static_cast<size_t>(K) * sizeof(float);
    const size_t b_bytes =
        static_cast<size_t>(K) * static_cast<size_t>(N) * sizeof(float);
    const size_t c_bytes =
        static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(float);

    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    check_cuda(cudaMalloc(&A, a_bytes), "cudaMalloc(A)");
    check_cuda(cudaMalloc(&B, b_bytes), "cudaMalloc(B)");
    check_cuda(cudaMalloc(&C, c_bytes), "cudaMalloc(C)");
    check_cuda(cudaMemset(A, 0, a_bytes), "cudaMemset(A)");
    check_cuda(cudaMemset(B, 0, b_bytes), "cudaMemset(B)");
    check_cuda(cudaMemset(C, 0, c_bytes), "cudaMemset(C)");

    for (int iteration = 0; iteration < warmup; ++iteration) {
        kernel->launcher(M, N, K, A, B, C, nullptr);
        check_cuda(cudaGetLastError(), "warmup kernel launch");
    }
    check_cuda(cudaDeviceSynchronize(), "warmup synchronization");

    cudaEvent_t start = nullptr;
    cudaEvent_t end = nullptr;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    check_cuda(cudaEventCreate(&end), "cudaEventCreate(end)");
    check_cuda(cudaEventRecord(start), "cudaEventRecord(start)");
    for (int iteration = 0; iteration < repeat; ++iteration) {
        kernel->launcher(M, N, K, A, B, C, nullptr);
        check_cuda(cudaGetLastError(), "profiled kernel launch");
    }
    check_cuda(cudaEventRecord(end), "cudaEventRecord(end)");
    check_cuda(cudaEventSynchronize(end), "cudaEventSynchronize(end)");

    float total_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&total_ms, start, end),
               "cudaEventElapsedTime");
    const double latency_ms = static_cast<double>(total_ms) / repeat;
    const double tflops =
        2.0 * static_cast<double>(M) * N * K /
        (latency_ms * 1.0e-3) / 1.0e12;

    std::printf("Kernel = %s\n", kernel->name);
    std::printf("M N K = %d %d %d\n", M, N, K);
    std::printf("Warmup = %d, Repeat = %d\n", warmup, repeat);
    std::printf("Latency = %.6f ms, Performance = %.4f TFLOPS\n",
                latency_ms, tflops);

    check_cuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
    check_cuda(cudaEventDestroy(end), "cudaEventDestroy(end)");
    check_cuda(cudaFree(A), "cudaFree(A)");
    check_cuda(cudaFree(B), "cudaFree(B)");
    check_cuda(cudaFree(C), "cudaFree(C)");
    return EXIT_SUCCESS;
}
