#!/usr/bin/env python3
"""Compile, validate and benchmark the CUDA SGEMM kernels."""

from __future__ import annotations

import argparse
import csv
import os
import statistics
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch


def select_cuda_home() -> None:
    """Prefer an installed CUDA toolkit with the same major version as PyTorch."""
    if not torch.version.cuda:
        return
    torch_major, torch_minor = (int(part) for part in torch.version.cuda.split(".")[:2])
    configured = os.environ.get("CUDA_HOME")
    if configured:
        resolved_name = Path(configured).resolve().name.removeprefix("cuda-")
        try:
            if int(resolved_name.split(".")[0]) == torch_major:
                return
        except ValueError:
            return
    candidates = []
    for path in Path("/usr/local").glob(f"cuda-{torch_major}.*"):
        try:
            _, minor = path.name.removeprefix("cuda-").split(".")[:2]
            candidates.append((abs(int(minor) - torch_minor), path))
        except ValueError:
            continue
    if candidates:
        os.environ["CUDA_HOME"] = str(min(candidates, key=lambda item: item[0])[1])


select_cuda_home()

from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parent
BUILD_DIR = ROOT / "build"
RESULTS_DIR = ROOT / "results"
KERNEL_NAMES = tuple(f"gemm_v{i}" for i in range(6))
DEFAULT_CHECK_SHAPES = ((64, 64, 32), (128, 192, 96), (256, 128, 320))
DEFAULT_BENCH_SHAPES = ((256, 256, 256), (512, 512, 512),
                        (1024, 1024, 1024), (2048, 2048, 2048))


def parse_shape(value: str) -> tuple[int, int, int]:
    normalized = value.lower().replace("×", "x")
    parts = normalized.split("x")
    if len(parts) == 1:
        parts *= 3
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"invalid shape {value!r}; use SIZE or MxNxK")
    try:
        shape = tuple(int(part) for part in parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid shape {value!r}") from error
    if any(dimension <= 0 for dimension in shape):
        raise argparse.ArgumentTypeError("matrix dimensions must be positive")
    return shape  # type: ignore[return-value]


def load_kernels(verbose: bool):
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    sources = [ROOT / "bindings.cpp"] + [
        ROOT / f"{name}.cu" for name in KERNEL_NAMES
    ] + [ROOT / "cublas" / "sgemm_cublas.cu"]
    return load(
        name="gemm_cuda",
        sources=[str(source) for source in sources],
        build_directory=str(BUILD_DIR),
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3", "--use_fast_math", "-lineinfo", "-DNO_CUBLAS_SGEMM_BIN"
        ],
        extra_ldflags=["-lcublas"],
        verbose=verbose,
    )


def make_inputs(shape: tuple[int, int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    m, n, k = shape
    a = torch.empty((m, k), device="cuda", dtype=torch.float32).uniform_(-1, 1)
    b = torch.empty((k, n), device="cuda", dtype=torch.float32).uniform_(-1, 1)
    return a, b


def validate_kernels(
    kernels: dict[str, Callable],
    cublas_sgemm: Callable,
    shapes: Iterable[tuple[int, int, int]],
    rtol: float,
    atol: float,
) -> bool:
    print("\nCorrectness (reference: cuBLAS FP32)")
    all_correct = True
    for shape in shapes:
        a, b = make_inputs(shape)
        reference = torch.empty(
            (shape[0], shape[1]), device="cuda", dtype=torch.float32)
        cublas_sgemm(a, b, reference, False)
        shape_text = "x".join(map(str, shape))
        for name, kernel in kernels.items():
            output = torch.zeros_like(reference)
            kernel(a, b, output)
            difference = (output - reference).abs()
            max_abs = difference.max().item()
            max_rel = (difference / reference.abs().clamp_min(1e-8)).max().item()
            finite = torch.isfinite(output).all().item()
            correct = finite and torch.allclose(output, reference, rtol=rtol, atol=atol)
            all_correct &= correct
            status = "PASS" if correct else "FAIL"
            print(f"  {shape_text:>15}  {name:<8} {status}  "
                  f"max_abs={max_abs:.3e}  max_rel={max_rel:.3e}")
    return all_correct


def measure_ms(
    function: Callable[[], None], warmup: int, repeat: int, samples: int = 3
) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()

    timings = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeat):
            function()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) / repeat)
    return statistics.median(timings)


def benchmark_kernels(
    kernels: dict[str, Callable],
    cublas_sgemm: Callable,
    shapes: Iterable[tuple[int, int, int]],
    warmup: int,
    repeat: int,
    allow_tf32: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    print("\nPerformance (median of 3 measurements)")
    for shape in shapes:
        m, n, k = shape
        a, b = make_inputs(shape)
        outputs = {
            name: torch.zeros((m, n), device="cuda", dtype=torch.float32)
            for name in kernels
        }
        cublas_output = torch.empty((m, n), device="cuda", dtype=torch.float32)
        functions = {
            name: (lambda fn=kernel, out=outputs[name]: fn(a, b, out))
            for name, kernel in kernels.items()
        }
        functions["cublas"] = lambda: cublas_sgemm(
            a, b, cublas_output, allow_tf32)

        shape_text = f"{m}x{n}x{k}"
        print(f"  {shape_text}")
        baseline_tflops = None
        shape_rows = []
        for name, function in functions.items():
            latency_ms = measure_ms(function, warmup, repeat)
            tflops = 2.0 * m * n * k / (latency_ms * 1e-3) / 1e12
            row = {
                "kernel": name,
                "m": m,
                "n": n,
                "k": k,
                "latency_ms": latency_ms,
                "tflops": tflops,
            }
            shape_rows.append(row)
            if name == "cublas":
                baseline_tflops = tflops

        assert baseline_tflops is not None
        for row in shape_rows:
            row["vs_cublas"] = float(row["tflops"]) / baseline_tflops
            print(f"    {str(row['kernel']):<9} {float(row['latency_ms']):9.4f} ms  "
                  f"{float(row['tflops']):8.3f} TFLOPS  "
                  f"{float(row['vs_cublas']):6.2%} of cuBLAS")
        rows.extend(shape_rows)
    return rows


def save_csv(rows: Sequence[dict[str, object]]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "performance.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_results(rows: Sequence[dict[str, object]]) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib is not installed; skipping the performance plot. "
              "Install it with: python3 -m pip install matplotlib")
        return None

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    labels = []
    for row in rows:
        label = f"{row['m']}x{row['n']}x{row['k']}"
        if label not in labels:
            labels.append(label)

    figure, axis = plt.subplots(figsize=(10, 6))
    names = list(dict.fromkeys(str(row["kernel"]) for row in rows))
    for name in names:
        values = [
            float(row["tflops"])
            for label in labels
            for row in rows
            if row["kernel"] == name
            and f"{row['m']}x{row['n']}x{row['k']}" == label
        ]
        axis.plot(labels, values, marker="o", label=name)

    axis.set_xlabel("M x N x K")
    axis.set_ylabel("TFLOPS")
    axis.set_title("SGEMM performance")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    path = RESULTS_DIR / "performance.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernels", nargs="+", choices=KERNEL_NAMES,
                        default=list(KERNEL_NAMES))
    parser.add_argument("--shapes", nargs="+", type=parse_shape,
                        default=list(DEFAULT_BENCH_SHAPES),
                        metavar="SIZE|MxNxK")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--bench-only", action="store_true")
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="draw the performance figure after benchmarking",
    )
    parser.add_argument("--verbose-build", action="store_true")
    args = parser.parse_args()
    if args.check_only and args.bench_only:
        parser.error("--check-only and --bench-only cannot be used together")
    if args.warmup < 0 or args.repeat <= 0:
        parser.error("--warmup must be non-negative and --repeat must be positive")
    return args


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this Python environment")

    torch.manual_seed(0)
    extension = load_kernels(args.verbose_build)
    kernels = {name: getattr(extension, name) for name in args.kernels}

    print(f"Device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, "
          f"cuBLAS mode: {'TF32' if args.allow_tf32 else 'FP32'}")

    correct = True
    if not args.bench_only:
        correct = validate_kernels(kernels, extension.cublas_sgemm,
                                   DEFAULT_CHECK_SHAPES,
                                   args.rtol, args.atol)
    if args.check_only:
        return 0 if correct else 1

    rows = benchmark_kernels(kernels, extension.cublas_sgemm, args.shapes,
                             args.warmup, args.repeat, args.allow_tf32)
    csv_path = save_csv(rows)
    plot_path = plot_results(rows) if args.plot else None
    print(f"\nCSV: {csv_path}")
    if plot_path is not None:
        print(f"Plot: {plot_path}")
    return 0 if correct else 1


if __name__ == "__main__":
    raise SystemExit(main())
