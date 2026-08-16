#!/bin/bash

# ================================
# 用量定义区域 (Configuration)
# ================================
# 不用sudo就需要加上这个
export TMPDIR="/home/houhw/cuda_learning_practice_and_notes/reduce"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <target_binary>"
    exit 1
fi

TARGET_ARG="$1"
shift || true

# CUDA 安装路径（可根据需要切换版本）
CUDA_HOME="/usr"

# NCU 工具路径
NCU_BIN="${CUDA_HOME}/bin/ncu"

# 被测程序路径（支持相对脚本所在目录的路径）
if [[ "$TARGET_ARG" = /* ]]; then
    APP_BIN="$TARGET_ARG"
else
    APP_BIN="${SCRIPT_DIR}/${TARGET_ARG}"
fi

if [[ ! -x "$APP_BIN" ]]; then
    echo "Target binary not found or not executable: $APP_BIN"
    exit 1
fi

# 输出文件名（不含扩展名，ncu 会自动加 .ncu-rep）
OUTPUT_NAME="$(basename "$APP_BIN")"

# NCU 配置参数
NCU_CONFIG=(
    --nvtx --nvtx-include "NCU_GEMM/" 
    --config-file off
    --export "${OUTPUT_NAME}"
    --force-overwrite
    --target-processes all
    --set full
)

# ================================
# 执行命令
# ================================

echo "Starting NCU profiling..."
echo "Target: ${APP_BIN}"
echo "Output: ${OUTPUT_NAME}.ncu-rep"
echo "Command: sudo ${NCU_BIN} ${NCU_CONFIG[*]} ${APP_BIN}"

# 执行分析
"${NCU_BIN}" "${NCU_CONFIG[@]}" "${APP_BIN}"

# 检查执行结果
if [ $? -eq 0 ]; then
    echo "Profiling completed successfully."
else
    echo "Profiling failed!"
    exit 1
fi
