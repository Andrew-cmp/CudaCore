#!/usr/bin/env bash
set -euo pipefail

# Build a CUDA source under this directory. Usage: ./build.sh <file|file.cu> [nvcc args...]
cd "$(dirname "$0")"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <cuda_source.cu> [nvcc args...]"
  exit 1
fi

src="$1"
shift || true

if [[ "${src}" != *.cu ]]; then
  echo "Source file must end with .cu: ${src}"
  exit 1
fi

if [[ ! -f "${src}" ]]; then
  echo "Source file not found: ${src}"
  exit 1
fi

out="${src%.cu}"

nvcc -O3 -std=c++14 "${src}" -lnvToolsExt -lcublas -o "${out}" "$@"
