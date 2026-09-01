# HGEMM

本目录沿用 `../sgemm` 的组织方式，通过 PyTorch JIT 扩展编译、测试和分析
FP16 GEMM kernel。统一的 Python 调用约定是：

```python
lib.kernel_name(a, b, c)
```

其中 `a`、`b`、`c` 都是连续的 CUDA `torch.float16` 二维 tensor，计算语义为
`C = A @ B`。当前 starter 实现以 FP32 累加，最后写回 FP16。

## 目录结构

```text
hgemm/
├── hgemm.cu          # cuBLAS baseline、starter kernel、launcher 和 pybind module
├── test.py           # 统一编译、动态发现、正确性测试、benchmark、NCU target
├── test_for_ncu.py   # 旧环境变量入口的兼容包装
├── run_ncu.sh        # NCU 参数封装、预编译、报告和 CSV 导出
├── ncu_reports/      # 默认 NCU 报告目录
└── readme.md         # 本文档
```

`test.py` 自动编译本目录下的全部 `.cu` 文件，并动态发现扩展导出的 pybind
callable。新增 `lib.new_kernel(a, b, c)` 后，无需修改 Python kernel 列表，也无需
修改 `run_ncu.sh`。

## 环境要求

- NVIDIA GPU 和可用驱动。
- CUDA Toolkit，包含 `nvcc` 和 `ncu`。
- CUDA 版本的 PyTorch。
- Ninja（PyTorch JIT 扩展构建使用）。

可以先检查：

```bash
nvidia-smi
nvcc --version
ncu --version
python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

如有需要，显式指定目标架构，例如：

```bash
export TORCH_CUDA_ARCH_LIST=8.6
```

以下命令默认在 `hgemm/` 中执行。

## 编译和发现 kernel

仅编译/加载扩展：

```bash
python3 test.py --build-only
```

打印完整 JIT/NVCC 命令：

```bash
python3 test.py --build-only --verbose-build
```

列出所有公开 binding：

```bash
python3 test.py --list-kernels
```

编译并验证指定 binding：

```bash
python3 test.py --build-only --kernel hgemm_v0
```

未知名字会在启动测试或 NCU 前直接报错并列出可用名字。

## 正确性测试与 benchmark

默认测试：

```bash
python3 test.py
```

默认使用 `M=N=K=4096`，对每个算子预热 10 次、计时 200 次，并输出平均延迟、
相对 cuBLAS 的加速比和 TFLOPS。参考结果由 `torch.matmul` 生成；FP16 默认误差为
`atol=0.1, rtol=0.02`。

当前 binding：

- `hgemm_cublas`：FP16 输入/输出、FP32 compute 的 cuBLAS Tensor Core baseline。
- `hgemm_v0`：16×16 shared-memory tiled starter kernel，支持非整除边界。

只测一个或多个 kernel：

```bash
python3 test.py --kernel hgemm_v0 --size 1024
python3 test.py --kernel hgemm_cublas,hgemm_v0 --size 4096
```

自定义非方阵与计时次数：

```bash
python3 test.py \
  --kernel hgemm_v0 \
  --m 1024 --n 2048 --k 512 \
  --warmup 5 --repeat 100
```

运行内置尺寸组合，或跳过/调整正确性校验：

```bash
python3 test.py --all-shapes
python3 test.py --kernel hgemm_v0 --size 4096 --no-check
python3 test.py --kernel hgemm_v0 --atol 0.2 --rtol 0.03
```

生成性能图：

```bash
python3 test.py --plot
```

默认输出 `hgemm_flops.png`，也可用 `--plot-output` 指定 PNG、SVG 或 PDF。

## NCU profiling

默认完整采集：

```bash
bash run_ncu.sh
```

默认 profile `hgemm_v0` 的 `4096³` 输入，使用 `--set full` 并生成：

```text
ncu_reports/hgemm_v0_profile.ncu-rep
```

执行流程与 `sgemm` 相同：

```text
校验参数
  → test.py 预编译并验证 pybind binding
  → NCU 启动 test.py --profile
  → 只执行目标 binding
  → 采集一个匹配的 CUDA kernel launch
  → 持久化 .ncu-rep，并按需导出 CSV/打印 details
```

指定 kernel、尺寸和报告名：

```bash
bash run_ncu.sh \
  --kernel hgemm_v0 \
  --size 4096 \
  --output ncu_reports/hgemm_v0_4096
```

非方阵、快速 section set、覆盖旧报告：

```bash
bash run_ncu.sh --kernel hgemm_v0 --shape 1024,2048,512
bash run_ncu.sh --kernel hgemm_v0 --size 256 --set basic
bash run_ncu.sh --kernel hgemm_v0 --force
```

完整采集并导出 raw CSV，或采集后打印 details：

```bash
bash run_ncu.sh --kernel hgemm_v0 --set full --csv
bash run_ncu.sh --kernel hgemm_v0 --details
```

报告也可以稍后读取，无需重新执行 kernel：

```bash
ncu --import ncu_reports/hgemm_v0_profile.ncu-rep --page details
ncu --import ncu_reports/hgemm_v0_profile.ncu-rep --page raw --csv > hgemm_v0.csv
```

默认用 pybind 名构造 NCU kernel 正则，例如 `hgemm_v0` 对应
`regex:hgemm_v0`。如果 binding 与 CUDA global function 名不同，显式指定：

```bash
bash run_ncu.sh \
  --kernel new_kernel \
  --kernel-filter 'regex:actual_cuda_kernel_name'
```

cuBLAS binding 会自动使用 `regex:.*gemm.*`。如果需要预热，脚本同时传递
`--launch-skip`，确保只采集预热后的一个匹配 launch：

```bash
bash run_ncu.sh --kernel hgemm_v0 --warmup 3
```

## 新 kernel 约定

1. 实现放在 `hgemm/` 下的 `.cu` 文件中。
2. 在唯一的 pybind module（当前为 `hgemm.cu`）中导出
   `lib.new_kernel(a, b, c)`。
3. 输入和输出使用连续 CUDA FP16 tensor；如需不同的输出/累加精度，应同步修改
   测试说明和参考精度。
4. CUDA global function 名最好包含 binding 名，以便 NCU 自动过滤。
5. launcher 应校验自身的尺寸/对齐约束，并使用 PyTorch 当前 CUDA stream。

推荐验证顺序：

```bash
python3 test.py --build-only --kernel new_kernel
python3 test.py --kernel new_kernel --size 256 --warmup 2 --repeat 10
python3 test.py --kernel new_kernel --size 4096 --warmup 10 --repeat 200
bash run_ncu.sh --kernel new_kernel --size 4096 --set full --csv
```

PyTorch JIT 编译产物位于 PyTorch 扩展缓存目录；NCU 报告默认持久化到
`hgemm/ncu_reports/`。
