# CUDA SGEMM

这个目录用于实现、验证和比较不同版本的 FP32 CUDA GEMM kernel。日常编译、正确性验证、性能测试和绘图由一个 Python 入口管理；cuBLAS 基线同时保留独立二进制入口，方便后续接入 Makefile 和 Nsight Compute。

当前正确性参考和性能基线均由项目内的 `cublas/sgemm_cublas.cu` 显式调用 cuBLAS，不再使用 `torch.mm`。

## 目录结构

```text
gemm/
├── README.md
├── gemm.py          # 编译、正确性验证、性能测试和绘图入口
├── bindings.cpp     # PyTorch/pybind 接口与通用 Tensor 检查
├── gemm_v0.cu       # CUDA kernel 与 launcher
├── gemm_v1.cu
├── gemm_v2.cu
├── gemm_v3.cu
├── gemm_v4.cu
├── gemm_v5.cu
├── cublas/
│   └── sgemm_cublas.cu  # Python 扩展与独立二进制共用的 cuBLAS 基线
├── build/           # 自动生成的扩展编译缓存
└── results/         # 自动生成的 CSV 和性能图
```

`gemm_v0.cu` 到 `gemm_v5.cu` 只包含自定义 kernel 及其 launcher。输入生成、正确性验证、批量性能测试和绘图均由 Python 完成。

`cublas/sgemm_cublas.cu` 是一个单文件双用途实现：

- 由 Python 扩展编译时，只提供显式 cuBLAS SGEMM 接口。
- 由 NVCC 直接编译时，提供包含 CUDA Event 计时的独立 `main()`。

## 环境依赖

- Python 3.10 或更高版本
- PyTorch（CUDA 版本）
- CUDA Toolkit 和 NVCC
- Ninja（PyTorch CUDA 扩展使用）
- Matplotlib（仅绘图需要）

可以使用下面的命令安装 Python 侧依赖：

```bash
python3 -m pip install torch ninja matplotlib
```

`gemm.py` 使用 `torch.utils.cpp_extension.load` 编译扩展。首次运行会编译所有 CUDA 文件，之后源码没有变化时会复用 `build/` 中的缓存。

如果系统默认 CUDA Toolkit 与 PyTorch CUDA 的主版本不同，脚本会尝试从 `/usr/local/cuda-*` 中选择主版本一致且版本最接近的 Toolkit。也可以显式指定：

```bash
CUDA_HOME=/usr/local/cuda-12.4 python3 gemm.py
```

## 快速开始

进入本目录后运行：

```bash
python3 gemm.py
```

默认流程为：

1. 编译并加载 CUDA 扩展。
2. 对全部 kernel 进行正确性验证。
3. 测量不同矩阵形状下的延迟和 TFLOPS。
4. 与显式调用的 cuBLAS SGEMM 进行性能对比。
5. 保存 CSV，并在 Matplotlib 可用时绘制性能曲线。

## 常用命令

只进行正确性验证：

```bash
python3 gemm.py --check-only
```

只进行性能测试：

```bash
python3 gemm.py --bench-only
```

选择部分 kernel：

```bash
python3 gemm.py --kernels gemm_v2 gemm_v3 gemm_v5
```

测试方阵：

```bash
python3 gemm.py --shapes 256 512 1024 2048
```

测试一般的 M、N、K 形状：

```bash
python3 gemm.py --shapes 256x512x128 1024x2048x512
```

调整预热和重复次数：

```bash
python3 gemm.py --warmup 10 --repeat 50
```

允许性能测试中的 cuBLAS 使用 TF32：

```bash
python3 gemm.py --allow-tf32
```

跳过绘图：

```bash
python3 gemm.py --no-plot
```

查看全部参数：

```bash
python3 gemm.py --help
```

## 正确性验证

参考结果由 `cublasGemmEx` 的 FP32 模式生成。脚本为每个 kernel 报告：

- PASS 或 FAIL
- 最大绝对误差
- 最大相对误差
- 输出是否包含 NaN 或 Inf

默认容差为：

```text
rtol = 1e-3
atol = 1e-2
```

可以通过命令行修改：

```bash
python3 gemm.py --rtol 1e-4 --atol 1e-3
```

当前测试形状会选用所有已有 kernel 都能正确处理的对齐尺寸。手动传入形状时，应确保形状符合被测 kernel 本身的实现假设。

## 性能测试

脚本使用 CUDA Event 测量 GPU 执行时间。每个测试包含预热，并进行三组计时，最终采用三组结果的中位数。

计算性能使用：

```text
TFLOPS = 2 × M × N × K / time_seconds / 10¹²
```

对比基线在结果中标记为 `cublas`，由项目中的 `cublas/sgemm_cublas.cu` 显式调用 `cublasGemmEx`。

正确性验证始终使用 `CUBLAS_COMPUTE_32F`。性能测试默认也使用 FP32；指定 `--allow-tf32` 后，cuBLAS 性能基线改用 `CUBLAS_COMPUTE_32F_FAST_TF32`，此时它与普通 CUDA Core FP32 kernel 不再是完全相同的计算路径。

## 输出文件

运行性能测试后会生成：

```text
results/performance.csv
results/performance.png
```

CSV 包含：

- kernel 名称
- M、N、K
- 平均单次延迟
- TFLOPS
- 相对 `cublas` 的性能比例

如果没有安装 Matplotlib，CSV 仍会正常生成，绘图步骤会自动跳过。

## 添加新的 kernel

添加新版本时：

1. 新建 `gemm_vN.cu`，实现 CUDA kernel 和 `launch_gemm_vN`。
2. 在 `bindings.cpp` 中声明 launcher，并导出同名 pybind 函数。
3. 在 `gemm.py` 的 `KERNEL_NAMES` 中加入 `gemm_vN`。
4. 运行正确性验证后再进行性能测试。

统一 launcher 接口为：

```cpp
void launch_gemm_vN(
    int M,
    int N,
    int K,
    float* A,
    float* B,
    float* C,
    cudaStream_t stream);
```

launcher 应使用传入的 CUDA stream 启动 kernel，不在 CUDA 文件中分配 Tensor、执行同步、计时或输出日志。

## 独立编译 cuBLAS 基线

`cublas/sgemm_cublas.cu` 采用单文件双用途结构。Python 扩展编译时定义 `NO_CUBLAS_SGEMM_BIN`，只使用其中的 cuBLAS 调用；直接用 NVCC 编译且不定义该宏时，文件会提供独立的性能测试入口。

```bash
CUDA_TOOLKIT=/usr/local/cuda-12.4
"$CUDA_TOOLKIT/bin/nvcc" cublas/sgemm_cublas.cu -O3 \
    -L"$CUDA_TOOLKIT/lib64" -lcublas \
    -Xlinker -rpath -Xlinker "$CUDA_TOOLKIT/lib64" \
    -o cublas_sgemm
./cublas_sgemm
./cublas_sgemm 4096 4096 4096 10 100
```

五个位置参数依次为 `M N K warmup repeat`。该二进制可以直接交给 Nsight Compute：

```bash
ncu ./cublas_sgemm 4096 4096 4096 10 1
```
