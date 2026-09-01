#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
test_script="${script_dir}/test.py"

kernel_name="hgemm_v0"
matrix_size=4096
matrix_shape=""
size_was_set=0
report_base=""
section_set="full"
kernel_filter="auto"
warmup=0
force=0
export_csv=0
print_details=0
positional_count=0

usage() {
    cat <<'EOF'
Usage:
  bash run_ncu.sh [options]
  bash run_ncu.sh KERNEL [REPORT_BASE]

Options:
  -k, --kernel NAME          pybind kernel name (default: hgemm_v0)
  -s, --size N               set M=N=K (default: 4096)
      --shape M,N,K          profile a non-square GEMM shape
  -o, --output PATH          report base path, with or without .ncu-rep
      --set NAME             NCU section set (default: full)
      --kernel-filter VALUE  NCU filter; auto, none, or regex/exact value
      --warmup N             skip N matching warmup launches (default: 0)
  -f, --force                overwrite an existing report
      --csv                  export raw report data to REPORT_BASE.csv
      --details              print the report details page after profiling
  -h, --help                 show this help

The extension is built and the pybind name is validated before NCU starts.
New lib.kernel(a, b, c) bindings are discovered without editing this script.
If a CUDA global function name does not contain its pybind name, provide an
explicit --kernel-filter value.
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 2
}

require_value() {
    [[ $# -ge 2 ]] || die "$1 requires a value"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -k|--kernel)
            require_value "$@"
            kernel_name="$2"
            shift 2
            ;;
        -s|--size)
            require_value "$@"
            matrix_size="$2"
            size_was_set=1
            shift 2
            ;;
        --shape)
            require_value "$@"
            matrix_shape="$2"
            shift 2
            ;;
        -o|--output)
            require_value "$@"
            report_base="$2"
            shift 2
            ;;
        --set)
            require_value "$@"
            section_set="$2"
            shift 2
            ;;
        --kernel-filter)
            require_value "$@"
            kernel_filter="$2"
            shift 2
            ;;
        --warmup)
            require_value "$@"
            warmup="$2"
            shift 2
            ;;
        -f|--force)
            force=1
            shift
            ;;
        --csv)
            export_csv=1
            shift
            ;;
        --details)
            print_details=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            die "unknown option: $1"
            ;;
        *)
            if [[ ${positional_count} -eq 0 ]]; then
                kernel_name="$1"
            elif [[ ${positional_count} -eq 1 ]]; then
                report_base="$1"
            else
                die "too many positional arguments"
            fi
            positional_count=$((positional_count + 1))
            shift
            ;;
    esac
done

[[ $# -eq 0 ]] || die "unexpected argument(s): $*"
[[ "${matrix_size}" =~ ^[1-9][0-9]*$ ]] || die "--size must be a positive integer"
if [[ -n "${matrix_shape}" && ${size_was_set} -eq 1 ]]; then
    die "--shape cannot be combined with --size"
fi
if [[ -n "${matrix_shape}" ]]; then
    [[ "${matrix_shape}" =~ ^([1-9][0-9]*),([1-9][0-9]*),([1-9][0-9]*)$ ]] || \
        die "--shape must use positive integers in M,N,K form"
    matrix_m="${BASH_REMATCH[1]}"
    matrix_n="${BASH_REMATCH[2]}"
    matrix_k="${BASH_REMATCH[3]}"
else
    matrix_m="${matrix_size}"
    matrix_n="${matrix_size}"
    matrix_k="${matrix_size}"
fi
[[ "${warmup}" =~ ^[0-9]+$ ]] || die "--warmup must be a non-negative integer"
[[ -n "${kernel_name}" ]] || die "kernel name cannot be empty"
[[ -n "${section_set}" ]] || die "NCU section set cannot be empty"

command -v python3 >/dev/null 2>&1 || die "python3 was not found in PATH"
command -v ncu >/dev/null 2>&1 || die "ncu was not found in PATH"
[[ -f "${test_script}" ]] || die "test entry not found: ${test_script}"

if [[ -z "${report_base}" ]]; then
    report_base="${script_dir}/ncu_reports/${kernel_name}_profile"
fi
report_base="${report_base%.ncu-rep}"
report_file="${report_base}.ncu-rep"
csv_file="${report_base}.csv"

mkdir -p "$(dirname -- "${report_base}")"
if [[ -e "${report_file}" && ${force} -ne 1 ]]; then
    die "report already exists: ${report_file} (use --force to overwrite)"
fi

printf 'Building extension and validating binding: %s\n' "${kernel_name}"
python3 "${test_script}" --build-only --kernel "${kernel_name}"

if [[ "${kernel_filter}" == "auto" ]]; then
    if [[ "${kernel_name}" == hgemm_cublas* ]]; then
        kernel_filter='regex:.*gemm.*'
    else
        kernel_filter="regex:${kernel_name}"
    fi
fi

ncu_args=(
    --set "${section_set}"
    --target-processes all
    --launch-count 1
    -o "${report_base}"
)
if [[ "${kernel_filter}" != "none" ]]; then
    ncu_args+=(--kernel-name "${kernel_filter}")
fi
if [[ ${warmup} -gt 0 ]]; then
    ncu_args+=(--launch-skip "${warmup}")
fi
if [[ ${force} -eq 1 ]]; then
    ncu_args+=(--force-overwrite)
fi

printf 'Profiling %s (M=%s, N=%s, K=%s)\n' \
    "${kernel_name}" "${matrix_m}" "${matrix_n}" "${matrix_k}"
printf 'NCU report: %s\n' "${report_file}"
ncu "${ncu_args[@]}" \
    python3 "${test_script}" \
        --profile \
        --kernel "${kernel_name}" \
        --m "${matrix_m}" \
        --n "${matrix_n}" \
        --k "${matrix_k}" \
        --warmup "${warmup}" \
        --repeat 1

if [[ ${export_csv} -eq 1 ]]; then
    ncu --import "${report_file}" --page raw --csv > "${csv_file}"
    printf 'CSV export: %s\n' "${csv_file}"
fi

if [[ ${print_details} -eq 1 ]]; then
    ncu --import "${report_file}" --page details
fi
