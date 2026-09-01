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
EXTENSION_NAME = "sgemm_lib"
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
BASELINE_NAMES = ("sgemm_cublas", "sgemm_cublas_tf32")
PRESET_SIZES = (256, 512, 1024, 4096)
PLOT_SIZES = (256, 512, 1024, 2048, 4096)


def load_extension(verbose: bool = False):
    """Build (when needed) and load the CUDA extension."""
    if not SOURCES:
        raise FileNotFoundError(f"no CUDA sources found in {CURRENT_DIR}")
    missing_sources = [str(source) for source in SOURCES if not source.is_file()]
    if missing_sources:
        raise FileNotFoundError(f"missing extension source(s): {', '.join(missing_sources)}")

    return load(
        name=EXTENSION_NAME,
        sources=[str(source) for source in SOURCES],
        extra_cuda_cflags=CUDA_FLAGS,
        extra_cflags=COMMON_FLAGS,
        verbose=verbose,
    )


def _natural_key(name: str):
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name))


def discover_kernels(lib) -> Tuple[str, ...]:
    """Discover public pybind callables, so newly bound kernels need no Python edit."""
    names = [
        name
        for name in dir(lib)
        if not name.startswith("_") and callable(getattr(lib, name))
    ]
    return tuple(sorted(names, key=_natural_key))


def custom_kernels(lib) -> Tuple[str, ...]:
    return tuple(name for name in discover_kernels(lib) if name not in BASELINE_NAMES)


def default_kernels(lib) -> Tuple[str, ...]:
    available = discover_kernels(lib)
    ordered = []
    if "sgemm_cublas" in available:
        ordered.append("sgemm_cublas")
    ordered.extend(custom_kernels(lib))
    if "sgemm_cublas_tf32" in available:
        ordered.append("sgemm_cublas_tf32")
    return tuple(ordered)


def _flatten_kernel_args(values: Optional[Sequence[str]]) -> Tuple[str, ...]:
    if not values:
        return ()
    return tuple(name.strip() for value in values for name in value.split(",") if name.strip())


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
            f"unknown kernel(s): {', '.join(unknown)}; available: {', '.join(available)}"
        )
    return tuple(dict.fromkeys(names))


def _invoke(op: Callable, a: torch.Tensor, b: torch.Tensor, c: Optional[torch.Tensor]):
    if c is None:
        return op(a, b)
    return op(a, b, c)


def benchmark(
    op: Callable,
    a: torch.Tensor,
    b: torch.Tensor,
    c: Optional[torch.Tensor],
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

    duration = time.perf_counter() - start
    mean_seconds = duration / repeat
    tflops = 2.0 * a.shape[0] * b.shape[0] * b.shape[1] / mean_seconds / 1e12
    return mean_seconds, tflops


def print_result(
    name: str,
    mean_seconds: float,
    tflops: float,
    baselines: Optional[dict] = None,
):
    fields = [f"{name:40s}", f"mean: {mean_seconds * 1e3:9.6f} ms"]
    for baseline_name, baseline_seconds in (baselines or {}).items():
        label = baseline_name.removeprefix("sgemm_")
        fields.append(f"vs {label}: {baseline_seconds / mean_seconds:6.2f}x")
    fields.append(f"{tflops:8.2f} TFLOPS")
    print(" | ".join(fields))


def check_result(
    reference: torch.Tensor,
    output: torch.Tensor,
    name: str,
    atol: Optional[float],
    rtol: Optional[float],
):
    default_tolerance = 0.1 if "tf32" in name.lower() else 1e-3
    actual_atol = default_tolerance if atol is None else atol
    actual_rtol = default_tolerance if rtol is None else rtol
    if torch.allclose(reference, output, atol=actual_atol, rtol=actual_rtol):
        return

    difference = (reference - output).abs()
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
    print(f"shape: M={m}, N={n}, K={k}; warmup={warmup}, repeat={repeat}")

    a = torch.randn(m, k, device="cuda", dtype=torch.float32)
    b = torch.randn(k, n, device="cuda", dtype=torch.float32)
    reference = torch.empty(m, n, device="cuda", dtype=torch.float32)

    torch.matmul(a, b, out=reference)
    results = {}
    baseline_times = {}

    for name in BASELINE_NAMES:
        if not hasattr(lib, name):
            raise RuntimeError(f"required benchmark baseline is not bound: {name}")
        output = torch.empty_like(reference)
        mean_seconds, tflops = benchmark(
            getattr(lib, name), a, b, output, warmup, repeat
        )
        print_result(name, mean_seconds, tflops)
        results[name] = tflops
        baseline_times[name] = mean_seconds
        if check:
            try:
                check_result(reference, output, name, atol, rtol)
            except AssertionError as error:
                if not continue_on_check_error:
                    raise
                results[name] = math.nan
                print(f"WARNING: omitting invalid plot point: {error}")

    for name in kernel_names:
        if name in BASELINE_NAMES:
            continue
        output = torch.empty_like(reference)
        mean_seconds, tflops = benchmark(
            getattr(lib, name), a, b, output, warmup, repeat
        )
        print_result(name, mean_seconds, tflops, baselines=baseline_times)
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


def _plot_kernel_group(
    axis,
    sizes: Sequence[int],
    series: dict,
    names: Sequence[str],
    baseline_name: str,
    title: str,
    subtitle: str,
):
    import matplotlib.pyplot as plt

    x_positions = list(range(len(sizes)))
    custom_names = [name for name in names if name != baseline_name]
    colors = plt.colormaps["turbo"].resampled(max(1, len(custom_names)))
    markers = ("o", "s", "^", "D", "v", "P", "X", "h", "<", ">", "*")
    baseline_final = series[baseline_name][-1]

    def legend_label(name: str) -> str:
        final_value = series[name][-1]
        if (
            not math.isfinite(final_value)
            or not math.isfinite(baseline_final)
            or baseline_final <= 0
        ):
            return f"{name}  ·  n/a"
        return f"{name}  ·  {final_value / baseline_final:.2f}×"

    if baseline_name in series:
        axis.plot(
            x_positions,
            series[baseline_name],
            color="#111827",
            linewidth=3.2,
            linestyle="--",
            marker="o",
            markersize=7,
            markerfacecolor="#ffffff",
            markeredgewidth=2,
            label=legend_label(baseline_name),
            zorder=10,
        )

    for index, name in enumerate(custom_names):
        axis.plot(
            x_positions,
            series[name],
            color=colors(index),
            linewidth=2.25,
            marker=markers[index % len(markers)],
            markersize=6,
            markeredgecolor="#ffffff",
            markeredgewidth=0.8,
            label=legend_label(name),
            alpha=0.96,
        )

    axis.set_title(title, fontsize=17, fontweight="bold", color="#0f172a", pad=25)
    axis.text(
        0.5,
        1.02,
        subtitle,
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=10.5,
        color="#64748b",
    )
    axis.set_xticks(x_positions, [str(size) for size in sizes])
    axis.set_xlabel("Matrix size (M = N = K)", fontsize=11.5, labelpad=10)
    axis.set_ylabel("Throughput (TFLOPS)", fontsize=11.5, labelpad=10)
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", color="#cbd5e1", linewidth=0.9, alpha=0.7)
    axis.grid(axis="x", color="#e2e8f0", linewidth=0.7, alpha=0.55)
    axis.tick_params(axis="both", colors="#334155", labelsize=10.5)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#94a3b8")
    axis.spines["bottom"].set_color("#94a3b8")
    axis.set_facecolor("#f8fafc")
    axis.legend(
        title=f"Speedup vs {baseline_name} @ {sizes[-1]}",
        title_fontsize=9.5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        fontsize=8.5,
        frameon=True,
        fancybox=True,
        framealpha=0.96,
        facecolor="#ffffff",
        edgecolor="#e2e8f0",
        handlelength=2.6,
        columnspacing=1.2,
    )


def write_flops_figure(
    output_path: Path,
    sizes: Sequence[int],
    series: dict,
    warmup: int,
    repeat: int,
):
    """Render FP32 and TF32 kernel comparisons as two Matplotlib subplots."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "plotting requires Matplotlib; install it with "
            "'python3 -m pip install --user matplotlib'"
        ) from error

    fp32_names = [
        name
        for name in series
        if name == "sgemm_cublas"
        or (name not in BASELINE_NAMES and not name.startswith("sgemm_tf32"))
    ]
    tf32_names = [
        name
        for name in series
        if name == "sgemm_cublas_tf32" or name.startswith("sgemm_tf32")
    ]
    if not fp32_names or not tf32_names:
        raise RuntimeError("plot data must contain both FP32 and TF32 kernel groups")

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(20, 9), facecolor="#eef2f7")
    _plot_kernel_group(
        axes[0],
        sizes,
        series,
        fp32_names,
        "sgemm_cublas",
        "CUDA Core SGEMM",
        "sgemm.cu kernels compared with cuBLAS FP32",
    )
    _plot_kernel_group(
        axes[1],
        sizes,
        series,
        tf32_names,
        "sgemm_cublas_tf32",
        "Tensor Core TF32 SGEMM",
        "sgemm_mma.cu kernels compared with cuBLAS TF32",
    )

    figure.suptitle(
        "SGEMM Kernel Performance Comparison",
        fontsize=25,
        fontweight="bold",
        color="#0f172a",
        y=0.975,
    )
    figure.text(
        0.5,
        0.925,
        f"Five square matrix sizes · warmup={warmup} · repeat={repeat} · higher is better",
        ha="center",
        fontsize=11.5,
        color="#475569",
    )
    invalid_names = [
        name
        for name, values in series.items()
        if any(not math.isfinite(value) for value in values)
    ]
    if invalid_names:
        figure.text(
            0.5,
            0.025,
            "Missing points failed correctness checks: " + ", ".join(invalid_names),
            ha="center",
            fontsize=9.5,
            color="#b91c1c",
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": "#fef2f2",
                "edgecolor": "#fecaca",
            },
        )
    figure.subplots_adjust(left=0.065, right=0.985, top=0.84, bottom=0.29, wspace=0.16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
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


def run_profile(lib, kernel_name: str, m: int, n: int, k: int, warmup: int, repeat: int):
    """Minimal launch path intended to run under Nsight Compute."""
    print(f"NCU target: kernel={kernel_name}, M={m}, N={n}, K={k}")
    a = torch.randn(m, k, device="cuda", dtype=torch.float32)
    b = torch.randn(k, n, device="cuda", dtype=torch.float32)
    output = torch.empty(m, n, device="cuda", dtype=torch.float32)
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
        description="Build, test, benchmark, or profile the SGEMM pybind kernels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--kernel",
        action="append",
        metavar="NAME",
        help="kernel binding to run; repeat or use commas; default/all runs every binding",
    )
    parser.add_argument("--list-kernels", action="store_true", help="build and list discovered bindings")
    parser.add_argument("--build-only", action="store_true", help="build/load the extension, then exit")
    parser.add_argument("--profile", action="store_true", help="minimal single-kernel launch for NCU")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="benchmark square sizes and save a two-panel TFLOPS line chart",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=CURRENT_DIR / "sgemm_flops.png",
        help="PNG, SVG, or PDF output path used by --plot",
    )
    parser.add_argument("--size", type=positive_int, help="set M=N=K")
    parser.add_argument("--m", type=positive_int, help="matrix M dimension")
    parser.add_argument("--n", type=positive_int, help="matrix N dimension")
    parser.add_argument("--k", type=positive_int, help="matrix K dimension")
    parser.add_argument("--all-shapes", action="store_true", help="run all preset shape combinations")
    parser.add_argument("--warmup", type=non_negative_int, help="warmup launches")
    parser.add_argument("--repeat", type=positive_int, help="measured launches")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--no-check", action="store_true", help="skip correctness checks")
    parser.add_argument("--atol", type=float, help="override absolute correctness tolerance")
    parser.add_argument("--rtol", type=float, help="override relative correctness tolerance")
    parser.add_argument("--verbose-build", action="store_true", help="show JIT build commands")
    args = parser.parse_args(argv)

    if args.size is not None and any(value is not None for value in (args.m, args.n, args.k)):
        parser.error("--size cannot be combined with --m, --n, or --k")
    if args.all_shapes and (args.size is not None or any(value is not None for value in (args.m, args.n, args.k))):
        parser.error("--all-shapes cannot be combined with explicit dimensions")
    if args.profile and args.all_shapes:
        parser.error("--profile cannot be combined with --all-shapes")
    if args.plot and args.profile:
        parser.error("--plot cannot be combined with --profile")
    if args.plot and args.all_shapes:
        parser.error("--plot cannot be combined with --all-shapes")
    if args.plot and (args.size is not None or any(value is not None for value in (args.m, args.n, args.k))):
        parser.error("--plot uses fixed M=N=K sizes and cannot be combined with explicit dimensions")
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
            check=not args.no_check,
            atol=args.atol,
            rtol=args.rtol,
            output_path=args.plot_output.resolve(),
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
            check=not args.no_check,
            atol=args.atol,
            rtol=args.rtol,
        )


if __name__ == "__main__":
    main()
