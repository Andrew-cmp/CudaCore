#!/bin/bash

echo "=== MHA Kernel 验证测试套件 ==="
echo ""

# 检查CUDA环境
if ! command -v nvcc &> /dev/null; then
    echo "错误: 未找到nvcc编译器"
    exit 1
fi

# 编译程序
echo "编译验证程序..."
make clean
make verify_mha

if [ $? -ne 0 ]; then
    echo "编译失败!"
    exit 1
fi

echo "编译成功!"
echo ""

# 运行验证测试
echo "运行完整验证测试..."
./verify_mha

# 保存退出码
EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "🎉 所有测试通过! MHA kernel实现正确。"
else
    echo "❌ 测试失败，请检查实现。"
fi

echo ""
echo "测试完成。"

exit $EXIT_CODE