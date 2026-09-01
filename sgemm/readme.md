# SGEMM

本目录通过 PyTorch JIT 扩展编译和测试 FP32/TF32 CUDA GEMM kernel。CUDA
kernel、launcher 和 pybind binding 仍保留在 `.cu` 文件中，Python 侧统一使用：

```python
lib.kernel_name(a, b, c)
```

新的日常入口是 `test.py`，NCU 入口是 `run_ncu.sh`。`test_for_ncu.py` 仅作为旧流程
的兼容文件保留，新流程不再调用它。

## 文件作用

```text
sgemm/
├── sgemm.cu          # FP32 kernel、cuBLAS launcher 和 pybind module
├── sgemm_mma.cu      # TF32 Tensor Core kernel
├── test.py           # 统一编译、kernel 发现、测试、benchmark、NCU target
├── test_for_ncu.py   # 未修改的旧 NCU 入口
├── run_ncu.sh        # NCU 参数封装、预编译、报告和 CSV 导出
├── ncu_reports/      # 默认报告目录（运行后生成）
└── readme.md         # 本文档
```

`test.py` 会自动编译本目录下的所有 `.cu` 文件，并从加载后的扩展中动态发现公开的
pybind callable。因此，在现有 pybind 方式下新增 `lib.new_kernel(a, b, c)` 后，不需要
再维护 Python kernel 名单，也不需要修改 `run_ncu.sh`。

## 环境要求

- NVIDIA GPU 和可用的驱动。
- CUDA Toolkit，包含 `nvcc` 和 `ncu`。
- CUDA 版本的 PyTorch。
- Ninja，供 PyTorch JIT 扩展构建使用。

检查环境：

```bash
nvidia-smi
nvcc --version
ncu --version
python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

建议设置目标 GPU 架构。例如 compute capability 8.6：

```bash
export TORCH_CUDA_ARCH_LIST=8.6
```

以下命令可以在 `sgemm/` 目录执行；`run_ncu.sh` 也支持从其他目录通过完整路径执行。

## 编译与发现 kernel

仅编译或加载扩展，不分配测试矩阵：

```bash
python3 test.py --build-only
```

第一次执行会调用 NVCC，源文件未变化时复用 PyTorch JIT 缓存。需要查看完整编译命令时：

```bash
python3 test.py --build-only --verbose-build
```

列出扩展当前导出的全部 kernel binding：

```bash
python3 test.py --list-kernels
```

编译并验证一个新 binding 是否已经可用：

```bash
python3 test.py --build-only --kernel gemm_v6
```

未知名字会直接报错，并打印可用 binding，避免启动长测试或 NCU 后才发现参数写错。

## 正确性测试与 benchmark

### 默认测试

```bash
python3 test.py
```

默认行为：

- 使用 `M=N=K=4096`。
- 使用 `torch.matmul` 生成参考结果和性能基线。
- 自动测试所有公开 pybind GEMM binding，包括 cuBLAS 基线和新加入的 kernel。
- 每个算子预热 10 次并计时 200 次。
- 输出平均延迟、相对 `torch.matmul` 的加速比和 TFLOPS。
- FP32 默认使用 `atol=rtol=1e-3`；名字含 `tf32` 的 binding 默认使用 `0.1`。

### 测试一个 kernel

```bash
python3 test.py --kernel gemm_v5 --size 4096
```

一次测试多个 kernel，可以重复参数或使用逗号：

```bash
python3 test.py --kernel gemm_v4 --kernel gemm_v5 --size 4096
python3 test.py --kernel gemm_v4,gemm_v5 --size 4096
```

显式使用全部动态发现的 binding：

```bash
python3 test.py --kernel all
```

### 自定义形状和计时次数

方阵可直接使用 `--size`：

```bash
python3 test.py --kernel gemm_v5 --size 1024 --warmup 5 --repeat 100
```

非方阵分别指定 `M/N/K`：

```bash
python3 test.py --kernel gemm_v5 --m 1024 --n 2048 --k 512
```

运行内置尺寸组合：

```bash
python3 test.py --all-shapes
```

内置尺寸为 `256、512、1024、4096、8192`。该模式包含大矩阵且默认对每个算子
计时 200 次，耗时会明显长于单尺寸测试。

只做性能测量或为近似计算 kernel 指定误差阈值：

```bash
python3 test.py --kernel gemm_v6 --no-check
python3 test.py --kernel gemm_v6 --atol 0.01 --rtol 0.01
```

查看所有测试选项：

```bash
python3 test.py --help
```

## NCU profiling

### 默认完整采集

```bash
bash run_ncu.sh
```

默认分析 `sgemm_tf32_bt` 的 `4096³` 输入，使用 NCU `full` section set，输出：

```text
ncu_reports/sgemm_tf32_bt_profile.ncu-rep
```

脚本按以下顺序执行：

```text
解析并校验参数
  → test.py 预编译扩展并验证 pybind binding
  → NCU 启动 test.py --profile
  → 创建输入并只执行目标 binding
  → 采集一个匹配的 CUDA kernel launch
  → 保存 .ncu-rep，并按需导出 CSV/打印 details
```

预编译避免把 NVCC 构建过程混入 profiler。`--set full` 会对目标 kernel 进行多个
replay pass，采集完整报告可能耗时较长。

### 指定 kernel、尺寸和报告名

```bash
bash run_ncu.sh \
  --kernel gemm_v5 \
  --size 4096 \
  --output ncu_reports/gemm_v5_4096
```

`--output` 可以带或不带 `.ncu-rep`。默认不会覆盖已有报告；确认需要覆盖时加
`--force`：

```bash
bash run_ncu.sh --kernel gemm_v5 --size 4096 --force
```

非方阵使用 `M,N,K` 顺序的 `--shape`：

```bash
bash run_ncu.sh --kernel gemm_v5 --shape 1024,2048,512
```

`--shape` 与 `--size` 互斥，三个维度都必须是正整数。

旧的位置参数形式仍可使用：

```bash
bash run_ncu.sh gemm_v5 ncu_reports/gemm_v5_profile
```

### 快速采集与报告导出

开发期间可使用较小矩阵和较轻的 section set 快速验证流程：

```bash
bash run_ncu.sh --kernel gemm_v5 --size 256 --set basic
```

完整采集并同时导出 raw CSV：

```bash
bash run_ncu.sh --kernel gemm_v5 --set full --csv
```

采集后直接打印 details 页面：

```bash
bash run_ncu.sh --kernel gemm_v5 --details
```

也可以稍后读取已有报告，无需重新运行 kernel：

```bash
ncu --import ncu_reports/gemm_v5_profile.ncu-rep --page details
ncu --import ncu_reports/gemm_v5_profile.ncu-rep --page raw --csv > gemm_v5.csv
```

使用图形界面：

```bash
ncu-ui ncu_reports/gemm_v5_profile.ncu-rep
```

### 新 kernel 的 NCU 过滤

默认情况下，脚本使用 pybind 名作为 NCU 正则过滤条件。例如：

```text
--kernel gemm_v6  →  --kernel-name regex:gemm_v6
```

只要 CUDA 全局函数名包含 binding 名（推荐命名为 `gemm_v6_kernel`），新增 kernel
即可直接 profile：

```bash
bash run_ncu.sh --kernel gemm_v6 --size 4096
```

如果 binding 和 CUDA 全局函数名不同，显式提供 NCU 过滤表达式：

```bash
bash run_ncu.sh \
  --kernel gemm_v6 \
  --kernel-filter 'regex:actual_cuda_kernel_name'
```

`sgemm_cublas*` binding 默认使用 `regex:.*gemm.*` 寻找底层 cuBLAS GEMM kernel。
必要时同样可以用 `--kernel-filter` 收紧匹配。传入 `--kernel-filter none` 会禁用
kernel 名过滤，一般只用于诊断。

有 warmup 需求时，脚本会让 NCU 跳过相同数量的匹配 launch，只采集随后的一个：

```bash
bash run_ncu.sh --kernel gemm_v6 --warmup 3
```

查看所有 NCU 选项：

```bash
bash run_ncu.sh --help
```

## 新 kernel 开发约定

要让新 kernel 自动进入编译、测试和 NCU 流程，需要满足：

1. CUDA 实现位于 `sgemm/` 下的 `.cu` 文件中；统一入口会自动收集这些源文件。
2. 沿用当前 pybind 方式导出一个公开 callable，不改变调用签名：

   ```python
   lib.new_kernel(a, b, c)
   ```

3. CUDA 全局函数名最好包含 binding 名，例如 binding `gemm_v6` 对应
   `gemm_v6_kernel`。
4. 选择满足该 kernel 对齐和分块要求的 `M/N/K`；特殊精度用 `--atol/--rtol`
   明确设置测试阈值。

推荐的新 kernel 验证流程：

```bash
python3 test.py --build-only --kernel gemm_v6
python3 test.py --kernel gemm_v6 --size 256 --warmup 2 --repeat 10
python3 test.py --kernel gemm_v6 --size 4096 --warmup 10 --repeat 200
bash run_ncu.sh --kernel gemm_v6 --size 4096 --set full --csv
```

## 当前 kernel 尺寸约束

- `gemm_v0`：`M`、`N` 必须是 32 的倍数。
- `gemm_v1`：包含边界处理。
- `gemm_v2` 及三个 warp-tiling 版本：`M`、`N` 是 128 的倍数，`K` 是 16 的倍数。
- `gemm_v3`：`M`、`N` 是 64 的倍数，`K` 是 8 的倍数。
- `gemm_v4`：`M`、`N`、`K` 至少满足 `float4` 所需的 4 元素对齐。
- `gemm_v5`：对齐尺寸使用 fast path，其他尺寸使用带边界判断的 fallback。
- `gemm_test`：`M`、`N` 是 128 的倍数，`K` 是 16 的倍数。
- `sgemm_tf32_bt`：`M`、`N` 是 128 的倍数，`K` 是 16 的倍数；GPU compute
  capability 至少为 8.0。

PyTorch JIT 编译产物位于 PyTorch 扩展缓存目录，不写入本目录。NCU 报告默认写入
`sgemm/ncu_reports/`。
