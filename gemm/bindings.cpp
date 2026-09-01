#include <torch/extension.h>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <limits>

using GemmLauncher = void (*)(
    int, int, int, float*, float*, float*, cudaStream_t);

void launch_gemm_v0(int, int, int, float*, float*, float*, cudaStream_t);
void launch_gemm_v1(int, int, int, float*, float*, float*, cudaStream_t);
void launch_gemm_v2(int, int, int, float*, float*, float*, cudaStream_t);
void launch_gemm_v3(int, int, int, float*, float*, float*, cudaStream_t);
void launch_gemm_v4(int, int, int, float*, float*, float*, cudaStream_t);
void launch_gemm_v5(int, int, int, float*, float*, float*, cudaStream_t);
void launch_gemm_v2_warp_tiling(
    int, int, int, float*, float*, float*, cudaStream_t);
void launch_gemm_v2_warp_tiling_swizzle(
    int, int, int, float*, float*, float*, cudaStream_t);
void launch_gemm_test(int, int, int, float*, float*, float*, cudaStream_t);
cublasStatus_t launch_cublas_sgemm(
    int, int, int, const float*, const float*, float*, cudaStream_t, bool);

namespace {

void run_gemm(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& c,
    GemmLauncher launcher) {
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
    TORCH_CHECK(a.size(1) == b.size(0),
                "incompatible GEMM shapes: A is ", a.sizes(),
                " and B is ", b.sizes());
    TORCH_CHECK(c.size(0) == a.size(0) && c.size(1) == b.size(1),
                "C must have shape (", a.size(0), ", ", b.size(1), ")");
    TORCH_CHECK(a.size(0) <= std::numeric_limits<int>::max() &&
                    a.size(1) <= std::numeric_limits<int>::max() &&
                    b.size(1) <= std::numeric_limits<int>::max(),
                "matrix dimensions exceed the int32 kernel interface");

    const c10::cuda::CUDAGuard device_guard(a.device());
    const int m = static_cast<int>(a.size(0));
    const int n = static_cast<int>(b.size(1));
    const int k = static_cast<int>(a.size(1));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(a.get_device());

    launcher(
        m,
        n,
        k,
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void run_cublas_sgemm(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& c,
    bool allow_tf32) {
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
    module.def("cublas_sgemm", &run_cublas_sgemm,
               pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("c"),
               pybind11::arg("allow_tf32") = false);
    module.def("gemm_v0", [](const torch::Tensor& a, const torch::Tensor& b,
                              const torch::Tensor& c) {
        run_gemm(a, b, c, launch_gemm_v0);
    });
    module.def("gemm_v1", [](const torch::Tensor& a, const torch::Tensor& b,
                              const torch::Tensor& c) {
        run_gemm(a, b, c, launch_gemm_v1);
    });
    module.def("gemm_v2", [](const torch::Tensor& a, const torch::Tensor& b,
                              const torch::Tensor& c) {
        run_gemm(a, b, c, launch_gemm_v2);
    });
    module.def("gemm_v3", [](const torch::Tensor& a, const torch::Tensor& b,
                              const torch::Tensor& c) {
        run_gemm(a, b, c, launch_gemm_v3);
    });
    module.def("gemm_v4", [](const torch::Tensor& a, const torch::Tensor& b,
                              const torch::Tensor& c) {
        run_gemm(a, b, c, launch_gemm_v4);
    });
    module.def("gemm_v5", [](const torch::Tensor& a, const torch::Tensor& b,
                              const torch::Tensor& c) {
        run_gemm(a, b, c, launch_gemm_v5);
    });
    module.def("gemm_v2_warp_tiling",
               [](const torch::Tensor& a, const torch::Tensor& b,
                  const torch::Tensor& c) {
                   run_gemm(a, b, c, launch_gemm_v2_warp_tiling);
               });
    module.def("gemm_v2_warp_tiling_swizzle",
               [](const torch::Tensor& a, const torch::Tensor& b,
                  const torch::Tensor& c) {
                   run_gemm(a, b, c, launch_gemm_v2_warp_tiling_swizzle);
               });
    module.def("gemm_test", [](const torch::Tensor& a, const torch::Tensor& b,
                                const torch::Tensor& c) {
        run_gemm(a, b, c, launch_gemm_test);
    });
}
