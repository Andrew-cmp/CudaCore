#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace {

cublasHandle_t g_handle = nullptr;
int g_device = -1;

cublasStatus_t init_cublas_handle() {
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return CUBLAS_STATUS_INTERNAL_ERROR;
    }
    if (g_handle != nullptr && g_device == device) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (g_handle != nullptr) {
        cublasDestroy(g_handle);
        g_handle = nullptr;
    }
    const cublasStatus_t status = cublasCreate(&g_handle);
    if (status == CUBLAS_STATUS_SUCCESS) {
        g_device = device;
    }
    return status;
}

#ifndef NO_CUBLAS_SGEMM_BIN
void destroy_cublas_handle() {
    if (g_handle != nullptr) {
        cublasDestroy(g_handle);
        g_handle = nullptr;
        g_device = -1;
    }
}
#endif

}  // namespace

// A, B and C are row-major. cuBLAS is column-major, so this computes
// C^T = B^T * A^T by swapping A/B and M/N in the cuBLAS call.
cublasStatus_t launch_cublas_sgemm(
    int M,
    int N,
    int K,
    const float* A,
    const float* B,
    float* C,
    cudaStream_t stream,
    bool allow_tf32) {
    cublasStatus_t status = init_cublas_handle();
    if (status != CUBLAS_STATUS_SUCCESS) {
        return status;
    }
    status = cublasSetStream(g_handle, stream);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return status;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cublasComputeType_t compute_type =
        allow_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    const cublasGemmAlgo_t algorithm =
        allow_tf32 ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT;

    return cublasGemmEx(
        g_handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N,
        M,
        K,
        &alpha,
        B,
        CUDA_R_32F,
        N,
        A,
        CUDA_R_32F,
        K,
        &beta,
        C,
        CUDA_R_32F,
        N,
        compute_type,
        algorithm);
}

#ifndef NO_CUBLAS_SGEMM_BIN

namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", operation,
                     cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

void check_cublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "%s failed with cuBLAS status %d\n", operation,
                     static_cast<int>(status));
        std::exit(EXIT_FAILURE);
    }
}

int parse_positive(const char* value, const char* name) {
    const int parsed = std::atoi(value);
    if (parsed <= 0) {
        std::fprintf(stderr, "%s must be a positive integer\n", name);
        std::exit(EXIT_FAILURE);
    }
    return parsed;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 1 && argc != 6) {
        std::fprintf(stderr, "Usage: %s [M N K warmup repeat]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const int M = argc == 6 ? parse_positive(argv[1], "M") : 4096;
    const int N = argc == 6 ? parse_positive(argv[2], "N") : 4096;
    const int K = argc == 6 ? parse_positive(argv[3], "K") : 4096;
    const int warmup = argc == 6 ? parse_positive(argv[4], "warmup") : 10;
    const int repeat = argc == 6 ? parse_positive(argv[5], "repeat") : 100;

    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    check_cuda(cudaMalloc(&A, static_cast<size_t>(M) * K * sizeof(float)),
               "cudaMalloc(A)");
    check_cuda(cudaMalloc(&B, static_cast<size_t>(K) * N * sizeof(float)),
               "cudaMalloc(B)");
    check_cuda(cudaMalloc(&C, static_cast<size_t>(M) * N * sizeof(float)),
               "cudaMalloc(C)");
    check_cuda(cudaMemset(A, 0, static_cast<size_t>(M) * K * sizeof(float)),
               "cudaMemset(A)");
    check_cuda(cudaMemset(B, 0, static_cast<size_t>(K) * N * sizeof(float)),
               "cudaMemset(B)");

    for (int iteration = 0; iteration < warmup; ++iteration) {
        check_cublas(
            launch_cublas_sgemm(M, N, K, A, B, C, nullptr, false),
            "cuBLAS warmup");
    }
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    cudaEvent_t start;
    cudaEvent_t end;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    check_cuda(cudaEventCreate(&end), "cudaEventCreate(end)");
    check_cuda(cudaEventRecord(start), "cudaEventRecord(start)");
    for (int iteration = 0; iteration < repeat; ++iteration) {
        check_cublas(
            launch_cublas_sgemm(M, N, K, A, B, C, nullptr, false),
            "cuBLAS SGEMM");
    }
    check_cuda(cudaEventRecord(end), "cudaEventRecord(end)");
    check_cuda(cudaEventSynchronize(end), "cudaEventSynchronize(end)");

    float total_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&total_ms, start, end),
               "cudaEventElapsedTime");
    const double latency_ms = total_ms / repeat;
    const double tflops =
        2.0 * static_cast<double>(M) * N * K / (latency_ms * 1e-3) / 1e12;
    std::printf("cuBLAS SGEMM FP32\n");
    std::printf("M N K = %d %d %d\n", M, N, K);
    std::printf("Latency = %.6f ms, Performance = %.4f TFLOPS\n",
                latency_ms, tflops);

    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    destroy_cublas_handle();
    return EXIT_SUCCESS;
}

#endif
