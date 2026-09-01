#include <torch/extension.h>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <limits>

#ifndef TUNED_LAUNCHER
#error "TUNED_LAUNCHER must name the launcher in the generated CUDA source"
#endif

using GemmLauncher = void (*)(
    int, int, int, float*, float*, float*, cudaStream_t);

void TUNED_LAUNCHER(
    int, int, int, float*, float*, float*, cudaStream_t);
cublasStatus_t launch_cublas_sgemm(
    int, int, int, const float*, const float*, float*, cudaStream_t, bool);

namespace {

void check_tensors(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& c) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && c.is_cuda(),
                "A, B and C must be CUDA tensors");
    TORCH_CHECK(a.scalar_type() == torch::kFloat32 &&
                    b.scalar_type() == torch::kFloat32 &&
                    c.scalar_type() == torch::kFloat32,
                "A, B and C must have dtype torch.float32");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2,
                "A, B and C must be two-dimensional matrices");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && c.is_contiguous(),
                "A, B and C must be contiguous");
    TORCH_CHECK(a.device() == b.device() && a.device() == c.device(),
                "A, B and C must be on the same CUDA device");
    TORCH_CHECK(a.size(1) == b.size(0), "incompatible GEMM shapes");
    TORCH_CHECK(c.size(0) == a.size(0) && c.size(1) == b.size(1),
                "C has the wrong shape");
    TORCH_CHECK(a.size(0) <= std::numeric_limits<int>::max() &&
                    a.size(1) <= std::numeric_limits<int>::max() &&
                    b.size(1) <= std::numeric_limits<int>::max(),
                "matrix dimensions exceed the int32 kernel interface");
}

void run_tuned(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& c) {
    check_tensors(a, b, c);
    const c10::cuda::CUDAGuard device_guard(a.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(a.get_device());
    TUNED_LAUNCHER(
        static_cast<int>(a.size(0)),
        static_cast<int>(b.size(1)),
        static_cast<int>(a.size(1)),
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void run_cublas(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& c,
    bool allow_tf32) {
    check_tensors(a, b, c);
    const c10::cuda::CUDAGuard device_guard(a.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(a.get_device());
    const cublasStatus_t status = launch_cublas_sgemm(
        static_cast<int>(a.size(0)),
        static_cast<int>(b.size(1)),
        static_cast<int>(a.size(1)),
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        stream,
        allow_tf32);
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS,
                "cuBLAS SGEMM failed with status ", static_cast<int>(status));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("gemm", &run_tuned);
    module.def("cublas_sgemm", &run_cublas,
               pybind11::arg("a"), pybind11::arg("b"),
               pybind11::arg("c"), pybind11::arg("allow_tf32") = false);
}
