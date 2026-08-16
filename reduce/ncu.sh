#!/bin/bash

# ================================
# 用量定义区域 (Configuration)
# ================================
# 不用sudo就需要加上这个
export TMPDIR="/home/houhw/cuda_learning_practice_and_notes/reduce"
# CUDA 安装路径（可根据需要切换版本）
CUDA_HOME="/usr"

# NCU 工具路径
NCU_BIN="${CUDA_HOME}/bin/ncu"

# 被测程序路径
APP_BIN="/home/houhw/cuda_learning_practice_and_notes/reduce/reduce_v4"

# 输出文件名（不含扩展名，ncu 会自动加 .ncu-rep）
OUTPUT_NAME="reduce_v4"

# NCU 配置参数
NCU_CONFIG=(
    --config-file off
    --export "${OUTPUT_NAME}"
    --force-overwrite
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