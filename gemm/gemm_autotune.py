#!/usr/bin/env python3
"""Autotune CUDA GEMM tile parameters and compare the winners with cuBLAS."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import itertools
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import torch


ROOT = Path(__file__).resolve().parent
DEFAULT_SPACE = ROOT / "autotune_space.json"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "autotune"
BUILD_ROOT = ROOT / "build" / "autotune"
TUNABLE_PARAMETERS = (
    "THREAD_SIZE_M",
    "THREAD_SIZE_N",
    "BLOCK_SIZE_M",
    "BLOCK_SIZE_N",
    "BLOCK_SIZE_K",
)
CSV_FIELDS = (
    "result_id",
    "build_id",
    "kernel",
    *TUNABLE_PARAMETERS,
    "m",
    "n",
    "k",
    "status",
    "correct",
    "max_abs",
    "max_rel",
    "latency_ms",
    "tflops",
    "cublas_latency_ms",
    "cublas_tflops",
    "vs_cublas",
    "error",
)


def select_cuda_home() -> None:
    """Select an installed CUDA toolkit matching PyTorch's CUDA major."""
    if not torch.version.cuda:
        return
    torch_major, torch_minor = (
        int(part) for part in torch.version.cuda.split(".")[:2]
    )
    configured = os.environ.get("CUDA_HOME")
    if configured:
        try:
            major = int(Path(configured).resolve().name.removeprefix("cuda-").split(".")[0])
            if major == torch_major:
                return
        except ValueError:
            return
    candidates: list[tuple[int, Path]] = []
    for path in Path("/usr/local").glob(f"cuda-{torch_major}.*"):
        try:
            minor = int(path.name.removeprefix("cuda-").split(".")[1])
            candidates.append((abs(minor - torch_minor), path))
        except (IndexError, ValueError):
            continue
    if candidates:
        os.environ["CUDA_HOME"] = str(min(candidates, key=lambda item: item[0])[1])


def parse_shape(value: str) -> tuple[int, int, int]:
    parts = value.lower().replace("×", "x").split("x")
    if len(parts) == 1:
        parts *= 3
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"invalid shape {value!r}; use SIZE or MxNxK"
        )
    try:
        shape = tuple(int(part) for part in parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid shape {value!r}") from error
    if any(value <= 0 for value in shape):
        raise argparse.ArgumentTypeError("matrix dimensions must be positive")
    return shape  # type: ignore[return-value]


def discover_kernels() -> tuple[str, ...]:
    def version_key(name: str) -> tuple[int, str]:
        match = re.fullmatch(r"gemm_v(\d+)(.*)", name)
        if match is None:
            return (sys.maxsize, name)
        return (int(match.group(1)), match.group(2))

    versioned = sorted(
        (path.stem for path in ROOT.glob("gemm_v*.cu")),
        key=version_key,
    )
    if (ROOT / "gemm_test.cu").exists():
        versioned.append("gemm_test")
    return tuple(versioned)


def parameter_pattern(name: str) -> re.Pattern[str]:
    return re.compile(
        rf"(?m)^(\s*(?:const|constexpr)\s+int\s+{re.escape(name)}\s*=\s*)"
        rf"([-+]?\d+)(\s*;)"
    )


def extract_parameters(source: str, kernel: str) -> dict[str, int]:
    parameters: dict[str, int] = {}
    for name in TUNABLE_PARAMETERS:
        matches = parameter_pattern(name).findall(source)
        if len(matches) != 1:
            raise ValueError(
                f"{kernel}: expected exactly one integer definition of {name}, "
                f"found {len(matches)}"
            )
        parameters[name] = int(matches[0][1])
    return parameters


def replace_parameters(source: str, parameters: dict[str, int]) -> str:
    generated = source
    for name, value in parameters.items():
        pattern = parameter_pattern(name)
        generated, count = pattern.subn(
            lambda match, replacement=value: (
                f"{match.group(1)}{replacement}{match.group(3)}"
            ),
            generated,
            count=1,
        )
        if count != 1:
            raise ValueError(f"could not replace {name}")
    return generated


def parse_overrides(values: Sequence[str]) -> dict[str, list[int]]:
    overrides: dict[str, list[int]] = {}
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError(
                f"invalid --set {value!r}; use NAME=VALUE1,VALUE2"
            )
        name, raw_values = value.split("=", 1)
        name = name.strip().upper()
        if name not in TUNABLE_PARAMETERS:
            raise argparse.ArgumentTypeError(
                f"unknown parameter {name}; choose from {', '.join(TUNABLE_PARAMETERS)}"
            )
        try:
            candidates = [int(item) for item in raw_values.split(",")]
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"invalid integer list in --set {value!r}"
            ) from error
        if not candidates or any(candidate <= 0 for candidate in candidates):
            raise argparse.ArgumentTypeError(
                f"--set values must be positive integers: {value!r}"
            )
        overrides[name] = list(dict.fromkeys(candidates))
    return overrides


def load_space(path: Path) -> dict[str, list[dict[str, int]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("search-space JSON must contain an object")
    result: dict[str, list[dict[str, int]]] = {}
    for kernel, entries in data.items():
        if not isinstance(entries, list):
            raise ValueError(f"{kernel}: search-space entry must be a list")
        normalized: list[dict[str, int]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f"{kernel}: each candidate must be an object")
            candidate: dict[str, int] = {}
            for name, value in entry.items():
                if name not in TUNABLE_PARAMETERS:
                    raise ValueError(f"{kernel}: unknown parameter {name}")
                if not isinstance(value, int) or value <= 0:
                    raise ValueError(f"{kernel}: {name} must be a positive integer")
                candidate[name] = value
            normalized.append(candidate)
        result[kernel] = normalized
    return result


def make_candidates(
    defaults: dict[str, int],
    space_entries: Sequence[dict[str, int]],
    overrides: dict[str, list[int]],
    limit: int,
) -> list[dict[str, int]]:
    if overrides:
        value_lists = [overrides.get(name, [defaults[name]]) for name in TUNABLE_PARAMETERS]
        candidates = [
            dict(zip(TUNABLE_PARAMETERS, values))
            for values in itertools.product(*value_lists)
        ]
    else:
        entries = space_entries or [{}]
        candidates = [defaults | entry for entry in entries]

    unique: list[dict[str, int]] = []
    seen: set[tuple[int, ...]] = set()
    for candidate in candidates:
        key = tuple(candidate[name] for name in TUNABLE_PARAMETERS)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    if len(unique) > limit:
        print(f"    limiting {len(unique)} candidates to the first {limit}")
    return unique[:limit]


def filter_reason(
    kernel: str,
    parameters: dict[str, int],
    shapes: Iterable[tuple[int, int, int]],
) -> str | None:
    tm = parameters["THREAD_SIZE_M"]
    tn = parameters["THREAD_SIZE_N"]
    bm = parameters["BLOCK_SIZE_M"]
    bn = parameters["BLOCK_SIZE_N"]
    bk = parameters["BLOCK_SIZE_K"]
    if bm % tm or bn % tn:
        return "BLOCK_SIZE_M/N must be divisible by THREAD_SIZE_M/N"

    threads = bm * bn if kernel in {"gemm_v0", "gemm_v1"} else (bm // tm) * (bn // tn)
    if not 1 <= threads <= 1024:
        return f"thread-block size {threads} is outside [1, 1024]"

    if kernel == "gemm_v0":
        for m, n, _ in shapes:
            if m % bm or n % bn:
                return "gemm_v0 has no M/N boundary guard for this shape"
    elif kernel == "gemm_v1":
        if not (bm == bn == bk):
            return "gemm_v1 requires BLOCK_SIZE_M == BLOCK_SIZE_N == BLOCK_SIZE_K"
    elif kernel in {"gemm_v2", "gemm_v3"}:
        if bk % 4 or bn % 4:
            return "vectorized loads require BLOCK_SIZE_K/N divisible by 4"
        a_k_threads = bk // 4
        b_n_threads = bn // 4
        if threads % a_k_threads:
            return "thread count must be divisible by BLOCK_SIZE_K/4"
        a_m_threads = threads // a_k_threads
        if not a_m_threads or bm % a_m_threads:
            return "A tile cannot be evenly distributed over the thread block"
        if threads % b_n_threads:
            return "thread count must be divisible by BLOCK_SIZE_N/4"
        b_k_threads = threads // b_n_threads
        if not b_k_threads or bk % b_k_threads:
            return "B tile cannot be evenly distributed over the thread block"
        if kernel == "gemm_v3":
            for m, n, k in shapes:
                if m % bm or n % bn or k % bk:
                    return "gemm_v3 fast path requires shapes divisible by its tiles"
    elif kernel == "gemm_v4":
        if bk % 4 or bn % 4 or tn % 4:
            return "gemm_v4 vector operations require K/N/thread-N divisible by 4"
        if (bm * bk) % (4 * threads) or (bn * bk) % (4 * threads):
            return "gemm_v4 cooperative loads do not evenly cover the tile"
    elif kernel == "gemm_v5":
        if tm != 8 or tn != 8:
            return "current gemm_v5 fragment layout requires THREAD_SIZE_M=N=8"
        if bk % 4 or bn % 4:
            return "gemm_v5 vectorized loads require BLOCK_SIZE_K/N divisible by 4"
        a_k_threads = bk // 4
        b_n_threads = bn // 4
        if threads % a_k_threads or threads % b_n_threads:
            return "gemm_v5 cooperative-load thread mapping is not integral"
        if bm % (threads // a_k_threads) or bk % (threads // b_n_threads):
            return "gemm_v5 cooperative loads do not evenly cover the tile"
    elif kernel in {"gemm_v2_warp_tiling", "gemm_v2_warp_tiling_swizzle"}:
        if (tm, tn, bm, bn, bk) != (8, 8, 128, 128, 16):
            return (
                f"{kernel} currently requires TM=TN=8, BM=BN=128 and BK=16"
            )
        for m, n, k in shapes:
            if m % bm or n % bn or k % bk:
                return f"{kernel} requires shapes divisible by its tiles"
    return None


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def empty_row(
    result_id: str,
    build_id: str,
    kernel: str,
    parameters: dict[str, int],
    shape: tuple[int, int, int],
) -> dict[str, Any]:
    row: dict[str, Any] = {field: "" for field in CSV_FIELDS}
    row.update(
        {
            "result_id": result_id,
            "build_id": build_id,
            "kernel": kernel,
            "m": shape[0],
            "n": shape[1],
            "k": shape[2],
            **parameters,
        }
    )
    return row


def make_inputs(shape: tuple[int, int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    m, n, k = shape
    a = torch.empty((m, k), device="cuda", dtype=torch.float32).uniform_(-1, 1)
    b = torch.empty((k, n), device="cuda", dtype=torch.float32).uniform_(-1, 1)
    return a, b


def measure_ms(
    function: Callable[[], None], warmup: int, repeat: int, samples: int
) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    timings: list[float] = []
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


def build_extension(request: dict[str, Any]):
    select_cuda_home()
    from torch.utils.cpp_extension import load

    build_id = request["build_id"]
    build_dir = BUILD_ROOT / build_id
    build_dir.mkdir(parents=True, exist_ok=True)
    source_path = build_dir / f"{request['kernel']}.cu"
    if (
        not source_path.exists()
        or source_path.read_text(encoding="utf-8") != request["generated_source"]
    ):
        source_path.write_text(request["generated_source"], encoding="utf-8")
    os.environ.setdefault("MAX_JOBS", str(request["max_jobs"]))
    return load(
        name=f"gemm_tune_{build_id}",
        sources=[
            str(ROOT / "autotune_bindings.cpp"),
            str(source_path),
            str(ROOT / "cublas" / "sgemm_cublas.cu"),
        ],
        build_directory=str(build_dir),
        extra_cflags=[
            "-O3",
            f"-DTUNED_LAUNCHER=launch_{request['kernel']}",
        ],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-lineinfo",
            "-DNO_CUBLAS_SGEMM_BIN",
        ],
        extra_ldflags=["-lcublas"],
        verbose=request["verbose_build"],
    )


def run_worker(request_path: Path, result_path: Path) -> int:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    row = empty_row(
        request["result_id"],
        request["build_id"],
        request["kernel"],
        request["parameters"],
        tuple(request["shape"]),
    )
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in this Python environment")
        torch.manual_seed(request["seed"])
        extension = build_extension(request)

        check_shape = tuple(request["check_shape"])
        a, b = make_inputs(check_shape)
        output = torch.zeros((check_shape[0], check_shape[1]), device="cuda")
        reference = torch.empty_like(output)
        extension.cublas_sgemm(a, b, reference, False)
        extension.gemm(a, b, output)
        difference = (output - reference).abs()
        max_abs = difference.max().item()
        max_rel = (difference / reference.abs().clamp_min(1e-8)).max().item()
        correct = bool(
            torch.isfinite(output).all().item()
            and torch.allclose(
                output,
                reference,
                rtol=request["rtol"],
                atol=request["atol"],
            )
        )
        row.update({"correct": correct, "max_abs": max_abs, "max_rel": max_rel})
        if not correct:
            row.update({"status": "incorrect", "error": "correctness check failed"})
        else:
            del a, b, output, reference, difference
            torch.cuda.empty_cache()
            shape = tuple(request["shape"])
            a, b = make_inputs(shape)
            output = torch.empty((shape[0], shape[1]), device="cuda")
            cublas_output = torch.empty_like(output)
            tuned_ms = measure_ms(
                lambda: extension.gemm(a, b, output),
                request["warmup"],
                request["repeat"],
                request["samples"],
            )
            cublas_ms = measure_ms(
                lambda: extension.cublas_sgemm(
                    a, b, cublas_output, request["allow_tf32"]
                ),
                request["warmup"],
                request["repeat"],
                request["samples"],
            )
            m, n, k = shape
            operations = 2.0 * m * n * k
            tuned_tflops = operations / (tuned_ms * 1e-3) / 1e12
            cublas_tflops = operations / (cublas_ms * 1e-3) / 1e12
            row.update(
                {
                    "status": "ok",
                    "latency_ms": tuned_ms,
                    "tflops": tuned_tflops,
                    "cublas_latency_ms": cublas_ms,
                    "cublas_tflops": cublas_tflops,
                    "vs_cublas": tuned_tflops / cublas_tflops,
                }
            )
    except BaseException as error:  # The worker must report compiler/CUDA failures.
        row.update({"status": "error", "correct": False, "error": str(error)[-6000:]})

    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
    return 0 if row["status"] in {"ok", "incorrect"} else 1


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def best_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    best: list[dict[str, Any]] = []
    kernels = list(dict.fromkeys(str(row["kernel"]) for row in rows))
    for kernel in kernels:
        valid = [
            row for row in rows
            if row["kernel"] == kernel and row["status"] == "ok"
        ]
        if valid:
            best.append(max(valid, key=lambda row: float(row["tflops"])))
    return best


def config_label(row: dict[str, Any]) -> str:
    return " ".join(
        (
            f"TM={row['THREAD_SIZE_M']}",
            f"TN={row['THREAD_SIZE_N']}",
            f"BM={row['BLOCK_SIZE_M']}",
            f"BN={row['BLOCK_SIZE_N']}",
            f"BK={row['BLOCK_SIZE_K']}",
        )
    )


def write_svg(path: Path, winners: Sequence[dict[str, Any]], shape: tuple[int, int, int]) -> None:
    width = 1120
    left = 150
    right = 170
    top = 100
    row_height = 58
    entries: list[tuple[str, float, str, str]] = []
    for row in winners:
        entries.append(
            (
                str(row["kernel"]),
                float(row["tflops"]),
                "#3478c9",
                f"{float(row['vs_cublas']):.1%} cuBLAS | {config_label(row)}",
            )
        )
    if winners:
        cublas_value = statistics.median(float(row["cublas_tflops"]) for row in winners)
        entries.append(("cuBLAS", cublas_value, "#e58b2a", "FP32 baseline"))
    height = max(260, top + row_height * max(1, len(entries)) + 90)
    plot_width = width - left - right
    max_value = max((entry[1] for entry in entries), default=1.0) * 1.12
    title_shape = " × ".join(map(str, shape))

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,"Noto Sans SC",sans-serif;fill:#20242a}.title{font-size:24px;font-weight:700}.subtitle{font-size:14px;fill:#5d6672}.label{font-size:15px;font-weight:600}.value{font-size:14px;font-weight:600}.config{font-size:11px;fill:#5d6672}.tick{font-size:11px;fill:#6d7580}</style>',
        f'<text x="{width / 2}" y="38" text-anchor="middle" class="title">GEMM autotuning vs cuBLAS</text>',
        f'<text x="{width / 2}" y="62" text-anchor="middle" class="subtitle">FP32, M × N × K = {title_shape}; each custom bar is its best valid configuration</text>',
    ]
    if not entries:
        lines.append(
            f'<text x="{width / 2}" y="145" text-anchor="middle" class="subtitle">No correct benchmark result was produced.</text>'
        )
    else:
        for tick_index in range(6):
            value = max_value * tick_index / 5
            x = left + plot_width * tick_index / 5
            lines.extend(
                [
                    f'<line x1="{x:.1f}" y1="{top - 12}" x2="{x:.1f}" y2="{height - 52}" stroke="#e5e9ef"/>',
                    f'<text x="{x:.1f}" y="{height - 30}" text-anchor="middle" class="tick">{value:.1f}</text>',
                ]
            )
        lines.append(
            f'<text x="{left + plot_width / 2}" y="{height - 8}" text-anchor="middle" class="subtitle">TFLOPS</text>'
        )
        for index, (name, value, color, detail) in enumerate(entries):
            y = top + index * row_height
            bar_width = plot_width * value / max_value
            lines.extend(
                [
                    f'<text x="{left - 14}" y="{y + 20}" text-anchor="end" class="label">{html.escape(name)}</text>',
                    f'<rect x="{left}" y="{y}" width="{bar_width:.1f}" height="27" rx="4" fill="{color}"/>',
                    f'<text x="{left + bar_width + 9:.1f}" y="{y + 19}" class="value">{value:.3f}</text>',
                    f'<text x="{left}" y="{y + 43}" class="config">{html.escape(detail)}</text>',
                ]
            )
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_candidate(request: dict[str, Any], output_dir: Path, timeout: int) -> dict[str, Any]:
    work_dir = output_dir / "workers" / request["result_id"]
    work_dir.mkdir(parents=True, exist_ok=True)
    request_path = work_dir / "request.json"
    result_path = work_dir / "result.json"
    request_path.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(request_path), str(result_path)]
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        row = empty_row(
            request["result_id"], request["build_id"], request["kernel"],
            request["parameters"], tuple(request["shape"]),
        )
        row.update({"status": "timeout", "correct": False, "error": str(error)})
        return row
    if result_path.exists():
        row = json.loads(result_path.read_text(encoding="utf-8"))
    else:
        row = empty_row(
            request["result_id"], request["build_id"], request["kernel"],
            request["parameters"], tuple(request["shape"]),
        )
        row.update(
            {
                "status": "error",
                "correct": False,
                "error": (completed.stdout or "worker exited without a result")[-6000:],
            }
        )
    if completed.returncode and not row.get("error"):
        row["error"] = (completed.stdout or f"worker exit code {completed.returncode}")[-6000:]
    return row


def parse_args() -> argparse.Namespace:
    kernels = discover_kernels()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernels", nargs="+", choices=kernels, default=list(kernels))
    parser.add_argument("--shape", type=parse_shape, default=(4096, 4096, 4096), metavar="SIZE|MxNxK")
    parser.add_argument("--check-shape", type=parse_shape, default=(256, 256, 256), metavar="SIZE|MxNxK")
    parser.add_argument("--space", type=Path, default=DEFAULT_SPACE)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=V1,V2",
        help="override JSON and form a Cartesian search space; repeat for more parameters",
    )
    parser.add_argument("--max-configs-per-kernel", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=600, help="seconds allowed for each isolated candidate")
    parser.add_argument("--max-jobs", type=int, default=2, help="Ninja compile parallelism inside each worker")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--verbose-build", action="store_true")
    args = parser.parse_args()
    if args.max_configs_per_kernel <= 0:
        parser.error("--max-configs-per-kernel must be positive")
    if args.warmup < 0 or args.repeat <= 0 or args.samples <= 0:
        parser.error("warmup must be non-negative; repeat and samples must be positive")
    if args.timeout <= 0 or args.max_jobs <= 0:
        parser.error("timeout and max-jobs must be positive")
    try:
        args.overrides = parse_overrides(args.set)
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    return args


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this Python environment")
    space = {} if args.overrides else load_space(args.space)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    started = time.monotonic()

    print(f"Device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"Target shape: {'x'.join(map(str, args.shape))}")
    print(f"Correctness shape: {'x'.join(map(str, args.check_shape))}")
    print(f"cuBLAS mode: {'TF32' if args.allow_tf32 else 'FP32'}")

    for kernel in args.kernels:
        source_path = ROOT / f"{kernel}.cu"
        source = source_path.read_text(encoding="utf-8")
        defaults = extract_parameters(source, kernel)
        candidates = make_candidates(
            defaults,
            space.get(kernel, []),
            args.overrides,
            args.max_configs_per_kernel,
        )
        print(f"\n{kernel}: {len(candidates)} candidate(s)")
        for index, parameters in enumerate(candidates, start=1):
            generated_source = replace_parameters(source, parameters)
            build_id = stable_hash(
                {
                    "kernel": kernel,
                    "source": generated_source,
                    "bindings": (ROOT / "autotune_bindings.cpp").read_text(encoding="utf-8"),
                    "cublas": (ROOT / "cublas" / "sgemm_cublas.cu").read_text(encoding="utf-8"),
                }
            )
            result_id = stable_hash(
                {
                    "build_id": build_id,
                    "shape": args.shape,
                    "check_shape": args.check_shape,
                    "warmup": args.warmup,
                    "repeat": args.repeat,
                    "samples": args.samples,
                    "allow_tf32": args.allow_tf32,
                }
            )
            label = config_label(parameters)
            reason = filter_reason(kernel, parameters, (args.check_shape, args.shape))
            if reason:
                row = empty_row(result_id, build_id, kernel, parameters, args.shape)
                row.update({"status": "filtered", "correct": False, "error": reason})
                rows.append(row)
                print(f"  [{index}/{len(candidates)}] FILTER  {label}: {reason}")
                continue

            print(f"  [{index}/{len(candidates)}] RUN     {label}", flush=True)
            request = {
                "result_id": result_id,
                "build_id": build_id,
                "kernel": kernel,
                "parameters": parameters,
                "generated_source": generated_source,
                "shape": args.shape,
                "check_shape": args.check_shape,
                "warmup": args.warmup,
                "repeat": args.repeat,
                "samples": args.samples,
                "rtol": args.rtol,
                "atol": args.atol,
                "allow_tf32": args.allow_tf32,
                "seed": args.seed,
                "max_jobs": args.max_jobs,
                "verbose_build": args.verbose_build,
            }
            row = run_candidate(request, args.output_dir, args.timeout)
            rows.append(row)
            if row["status"] == "ok":
                print(
                    f"                 PASS  {float(row['tflops']):.3f} TFLOPS, "
                    f"{float(row['vs_cublas']):.1%} of cuBLAS"
                )
            else:
                short_error = str(row.get("error", "")).splitlines()[-1]
                print(f"                 {str(row['status']).upper()}  {short_error}")
            write_csv(args.output_dir / "all_results.csv", rows)

    all_csv = args.output_dir / "all_results.csv"
    write_csv(all_csv, rows)
    winners = best_rows(rows)
    best_csv = args.output_dir / "best_results.csv"
    write_csv(best_csv, winners)
    shape_name = "x".join(map(str, args.shape))
    figure_path = args.output_dir / f"best_vs_cublas_{shape_name}.svg"
    write_svg(figure_path, winners, args.shape)

    print("\nBest configurations")
    for row in winners:
        print(
            f"  {row['kernel']:<10} {float(row['tflops']):8.3f} TFLOPS  "
            f"{float(row['vs_cublas']):6.1%}  {config_label(row)}"
        )
    print(f"\nAll results: {all_csv}")
    print(f"Best results: {best_csv}")
    print(f"Figure: {figure_path}")
    print(f"Elapsed: {time.monotonic() - started:.1f} s")
    return 0 if winners else 1


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        raise SystemExit(run_worker(Path(sys.argv[2]), Path(sys.argv[3])))
    raise SystemExit(main())
