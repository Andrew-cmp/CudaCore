import argparse
import math
import re
import time
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import torch
from torch.utils.cpp_extension import load


torch.set_grad_enabled(False)

CURRENT_DIR = Path(__file__).parent.resolve()
EXTENSION_NAME = "hgemm_lib"
SOURCES = tuple(sorted(CURRENT_DIR.glob("*.cu")))
COMMON_FLAGS = ["-O3", "-std=c++17"]
CUDA_FLAGS = COMMON_FLAGS + [
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--use_fast_math",
    "-Xptxas -v",
]
BASELINE_NAMES = ("hgemm_cublas",)
PRESET_SIZES = (256, 512, 1024, 4096)
PLOT_SIZES = (256, 512, 1024, 2048, 4096)


def load_extension(verbose: bool = False):
    """Build (when needed) and load every CUDA source in this directory."""
    if not SOURCES:
        raise FileNotFoundError(f"no CUDA sources found in {CURRENT_DIR}")
    return load(
        name=EXTENSION_NAME,
        sources=[str(source) for source in SOURCES],
        extra_cuda_cflags=CUDA_FLAGS,
        extra_cflags=COMMON_FLAGS,
        verbose=verbose,
    )


def _natural_key(name: str):
    return tuple(
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", name)
    )


def discover_kernels(lib) -> Tuple[str, ...]:
    names = [
        name
        for name in dir(lib)
        if not name.startswith("_") and callable(getattr(lib, name))
    ]
    return tuple(sorted(names, key=_natural_key))


def default_kernels(lib) -> Tuple[str, ...]:
    available = discover_kernels(lib)
    baselines = [name for name in BASELINE_NAMES if name in available]
    custom = [name for name in available if name not in BASELINE_NAMES]
    return tuple(baselines + custom)


def _flatten_kernel_args(values: Optional[Sequence[str]]) -> Tuple[str, ...]:
    if not values:
        return ()
    return tuple(
        name.strip()
        for value in values
        for name in value.split(",")
        if name.strip()
    )


def select_kernels(lib, requested: Optional[Sequence[str]]) -> Tuple[str, ...]:
    available = discover_kernels(lib)
    names = _flatten_kernel_args(requested)
    if not names or names == ("all",):
        selected = default_kernels(lib)
        if not selected:
            raise RuntimeError("no pybind kernels were discovered")
        return selected
    if "all" in names:
        raise ValueError("'all' cannot be combined with another --kernel value")
    unknown = [name for name in names if name not in available]
    if unknown:
        raise ValueError(
            f"unknown kernel(s): {', '.join(unknown)}; "
            f"available: {', '.join(available)}"
        )
    return tuple(dict.fromkeys(names))


def _invoke(op: Callable, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
    return op(a, b, c)


def benchmark(
    op: Callable,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    warmup: int,
    repeat: int,
) -> Tuple[float, float]:
    for _ in range(warmup):
        _invoke(op, a, b, c)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeat):
        _invoke(op, a, b, c)
    torch.cuda.synchronize()

    mean_seconds = (time.perf_counter() - start) / repeat
    tflops = 2.0 * a.shape[0] * b.shape[0] * b.shape[1] / mean_seconds / 1e12
    return mean_seconds, tflops


def print_result(
    name: str,
    mean_seconds: float,
    tflops: float,
    baseline_seconds: Optional[float] = None,
):
    fields = [f"{name:40s}", f"mean: {mean_seconds * 1e3:9.6f} ms"]
    if baseline_seconds is not None:
        fields.append(f"vs cuBLAS: {baseline_seconds / mean_seconds:6.2f}x")
    fields.append(f"{tflops:8.2f} TFLOPS")
    print(" | ".join(fields))


def check_result(
    reference: torch.Tensor,
    output: torch.Tensor,
    name: str,
    atol: Optional[float],
    rtol: Optional[float],
):
    actual_atol = 0.1 if atol is None else atol
    actual_rtol = 0.02 if rtol is None else rtol
    if torch.allclose(reference, output, atol=actual_atol, rtol=actual_rtol):
        return
    difference = (reference.float() - output.float()).abs()
    raise AssertionError(
        f"{name} result mismatch: mean_abs_diff={difference.mean().item():.6g}, "
        f"max_abs_diff={difference.max().item():.6g}, "
        f"atol={actual_atol}, rtol={actual_rtol}"
    )


def run_benchmark_shape(
    lib,
    kernel_names: Sequence[str],
    m: int,
    n: int,
    k: int,
    warmup: int,
    repeat: int,
    check: bool,
    atol: Optional[float],
    rtol: Optional[float],
    continue_on_check_error: bool = False,
):
    print("=" * 100)
    print(f"shape: M={m}, N={n}, K={k}; dtype=float16; warmup={warmup}, repeat={repeat}")
    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    b = torch.randn(k, n, device="cuda", dtype=torch.float16)
    reference = torch.matmul(a, b)
    results = {}

    if not hasattr(lib, "hgemm_cublas"):
        raise RuntimeError("required benchmark baseline is not bound: hgemm_cublas")
    baseline_output = torch.empty_like(reference)
    baseline_seconds, baseline_tflops = benchmark(
        lib.hgemm_cublas, a, b, baseline_output, warmup, repeat
    )
    print_result("hgemm_cublas", baseline_seconds, baseline_tflops)
    results["hgemm_cublas"] = baseline_tflops
    if check:
        check_result(reference, baseline_output, "hgemm_cublas", atol, rtol)

    for name in kernel_names:
        if name in BASELINE_NAMES:
            continue
        output = torch.empty_like(reference)
        mean_seconds, tflops = benchmark(
            getattr(lib, name), a, b, output, warmup, repeat
        )
        print_result(name, mean_seconds, tflops, baseline_seconds)
        results[name] = tflops
        if check:
            try:
                check_result(reference, output, name, atol, rtol)
            except AssertionError as error:
                if not continue_on_check_error:
                    raise
                results[name] = math.nan
                print(f"WARNING: omitting invalid plot point: {error}")
    return results


def write_flops_figure(
    output_path: Path,
    sizes: Sequence[int],
    series: dict,
    warmup: int,
    repeat: int,
):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("plotting requires Matplotlib") from error

    positions = list(range(len(sizes)))
    figure, axis = plt.subplots(figsize=(12, 7), facecolor="#eef2f7")
    for index, (name, values) in enumerate(series.items()):
        baseline = series["hgemm_cublas"][-1]
        final = values[-1]
        ratio = final / baseline if math.isfinite(final) and baseline > 0 else math.nan
        label = f"{name} · {ratio:.2f}x" if math.isfinite(ratio) else f"{name} · n/a"
        axis.plot(
            positions,
            values,
            marker="o",
            linewidth=3 if name == "hgemm_cublas" else 2,
            linestyle="--" if name == "hgemm_cublas" else "-",
            label=label,
        )
    axis.set_title("HGEMM Kernel Performance Comparison", fontsize=18, fontweight="bold")
    axis.set_xticks(positions, [str(size) for size in sizes])
    axis.set_xlabel("Matrix size (M = N = K)")
    axis.set_ylabel("Throughput (TFLOPS)")
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.35)
    axis.legend(title=f"Speedup vs cuBLAS @ {sizes[-1]}")
    figure.text(
        0.5,
        0.01,
        f"FP16 input/output · FP32 accumulation · warmup={warmup} · repeat={repeat}",
        ha="center",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"FLOPS chart saved to: {output_path}")


def run_plot_benchmarks(
    lib,
    kernel_names: Sequence[str],
    warmup: int,
    repeat: int,
    check: bool,
    atol: Optional[float],
    rtol: Optional[float],
    output_path: Path,
):
    series = {}
    for size in PLOT_SIZES:
        results = run_benchmark_shape(
            lib,
            kernel_names,
            size,
            size,
            size,
            warmup,
            repeat,
            check,
            atol,
            rtol,
            continue_on_check_error=True,
        )
        for name, tflops in results.items():
            series.setdefault(name, []).append(tflops)
    write_flops_figure(output_path, PLOT_SIZES, series, warmup, repeat)


def run_profile(
    lib, kernel_name: str, m: int, n: int, k: int, warmup: int, repeat: int
):
    print(f"NCU target: kernel={kernel_name}, M={m}, N={n}, K={k}, dtype=float16")
    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    b = torch.randn(k, n, device="cuda", dtype=torch.float16)
    output = torch.empty(m, n, device="cuda", dtype=torch.float16)
    mean_seconds, tflops = benchmark(
        getattr(lib, kernel_name), a, b, output, warmup, repeat
    )
    print_result(kernel_name, mean_seconds, tflops)


def preset_shapes():
    for index, m in enumerate(PRESET_SIZES):
        nearby = PRESET_SIZES[max(0, index - 1) : index + 1]
        for k in nearby:
            for n in nearby:
                yield m, n, k


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description="Build, test, benchmark, or profile the HGEMM pybind kernels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--kernel",
        action="append",
        metavar="NAME",
        help="kernel binding to run; repeat or use commas; default/all runs every binding",
    )
    parser.add_argument("--list-kernels", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=CURRENT_DIR / "hgemm_flops.png",
    )
    parser.add_argument("--size", type=positive_int)
    parser.add_argument("--m", type=positive_int)
    parser.add_argument("--n", type=positive_int)
    parser.add_argument("--k", type=positive_int)
    parser.add_argument("--all-shapes", action="store_true")
    parser.add_argument("--warmup", type=non_negative_int)
    parser.add_argument("--repeat", type=positive_int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--atol", type=float)
    parser.add_argument("--rtol", type=float)
    parser.add_argument("--verbose-build", action="store_true")
    args = parser.parse_args(argv)

    explicit_shape = args.size is not None or any(
        value is not None for value in (args.m, args.n, args.k)
    )
    if args.size is not None and any(
        value is not None for value in (args.m, args.n, args.k)
    ):
        parser.error("--size cannot be combined with --m, --n, or --k")
    if args.all_shapes and explicit_shape:
        parser.error("--all-shapes cannot be combined with explicit dimensions")
    if args.profile and args.all_shapes:
        parser.error("--profile cannot be combined with --all-shapes")
    if args.plot and (args.profile or args.all_shapes or explicit_shape):
        parser.error("--plot cannot be combined with profile or explicit shape options")
    if args.plot_output.suffix.lower() not in {".png", ".svg", ".pdf"}:
        parser.error("--plot-output must use a .png, .svg, or .pdf extension")
    return parser, args


def resolve_shape(args) -> Tuple[int, int, int]:
    if args.size is not None:
        return args.size, args.size, args.size
    return args.m or 4096, args.n or 4096, args.k or 4096


def main(argv: Optional[Sequence[str]] = None):
    parser, args = parse_args(argv)
    lib = load_extension(verbose=args.verbose_build)
    available = discover_kernels(lib)

    if args.list_kernels:
        print("\n".join(available))
        if not args.build_only:
            return
    try:
        selected = select_kernels(lib, args.kernel)
    except ValueError as error:
        parser.error(str(error))

    if args.build_only:
        if args.kernel:
            print(f"extension ready; selected kernel(s): {', '.join(selected)}")
        else:
            print(f"extension ready; discovered {len(available)} binding(s)")
        return
    if not torch.cuda.is_available():
        parser.error("CUDA is not available in PyTorch")

    torch.manual_seed(args.seed)
    warmup = args.warmup if args.warmup is not None else (0 if args.profile else 10)
    repeat = args.repeat if args.repeat is not None else (1 if args.profile else 200)

    if args.profile:
        requested = _flatten_kernel_args(args.kernel)
        if not requested or requested == ("all",) or len(selected) != 1:
            parser.error("--profile requires exactly one explicit --kernel NAME")
        run_profile(lib, selected[0], *resolve_shape(args), warmup, repeat)
        return
    if args.plot:
        run_plot_benchmarks(
            lib,
            selected,
            warmup,
            repeat,
            not args.no_check,
            args.atol,
            args.rtol,
            args.plot_output.resolve(),
        )
        return

    shapes = preset_shapes() if args.all_shapes else (resolve_shape(args),)
    for m, n, k in shapes:
        run_benchmark_shape(
            lib,
            selected,
            m,
            n,
            k,
            warmup,
            repeat,
            not args.no_check,
            args.atol,
            args.rtol,
        )


if __name__ == "__main__":
    main()
