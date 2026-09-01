import os
import time
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

common_flags = ["-O3", "-std=c++17"]
current_dir = Path(__file__).parent.resolve()

lib = load(
    name="sgemm_lib",
    sources=[
        str(current_dir / "sgemm.cu"),
        str(current_dir / "sgemm_mma.cu"),
    ],
    extra_cuda_cflags=common_flags
    + [
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "-Xptxas -v",
    ],
    extra_cflags=common_flags,
    verbose=True,
)

KERNEL_NAMES = (
    "sgemm_tf32_bt",
    "gemm_v0",
    "gemm_v1",
    "gemm_v2",
    "gemm_v3",
    "gemm_v4",
    "gemm_v5",
    "gemm_v2_warp_tiling",
    "gemm_v2_warp_tiling_swizzle",
    "gemm_v2_warp_tiling_swizzle_rw",
    "gemm_test",
)

baseline = 1e-6


def benchmark(op, a, b, c=None, warmup=0, rep=1, prefix="torch"):
    if c is not None:
        for _ in range(warmup):
            op(a, b, c)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(rep):
            op(a, b, c)
        torch.cuda.synchronize()
    else:
        for _ in range(warmup):
            op(a, b)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(rep):
            op(a, b)
        torch.cuda.synchronize()

    duration = time.perf_counter() - start
    if prefix == "torch":
        global baseline
        baseline = duration
        print(f"{prefix:40s} mean time: {duration / rep * 1000:.6f} ms")
    else:
        speedup = baseline / duration
        print(
            f"{prefix:40s} mean time: {duration / rep * 1000:.6f} ms, "
            f"speedup: {speedup:.2f}"
        )


def run_fp32():
    size = int(os.environ.get("SGEMM_NCU_SIZE", "4096"))
    m = n = k = size
    print("#" * 100)
    print(f"m: {m}, n: {n}, k: {k}")

    a = torch.randn(m, k).float().cuda()
    b = torch.randn(k, n).float().cuda()
    output = torch.zeros(m, n).float().cuda()

    requested_kernel = os.environ.get("SGEMM_NCU_KERNEL")
    kernel_names = KERNEL_NAMES
    if requested_kernel:
        if requested_kernel == "sgemm_cublas":
            kernel_names = ()
        elif requested_kernel in KERNEL_NAMES:
            kernel_names = (requested_kernel,)
        else:
            raise ValueError(
                f"unknown SGEMM_NCU_KERNEL={requested_kernel!r}; "
                f"expected sgemm_cublas or one of {KERNEL_NAMES}"
            )

    if not requested_kernel or requested_kernel == "sgemm_cublas":
        benchmark(lib.sgemm_cublas, a, b, output, prefix="sgemm_cublas")
    for name in kernel_names:
        benchmark(getattr(lib, name), a, b, output, prefix=name)


if __name__ == "__main__":
    run_fp32()
