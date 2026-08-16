

cmake_minimum_required(VERSION 3.18)
project(GEMM_Project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-O3 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "-O3 ${CMAKE_CUDA_FLAGS}")
set(CMAKE_CUDA_ARCHITECTURES 86)


find_package(CUDAToolkit REQUIRED)


set(CMAKE_BUILD_TYPE Release)
add_executable(gemm_v0 gemm_v0.cu)
add_executable(gemm_v1 gemm_v1.cu)
# add_executable(gemm_v2 gemm_v2.cu)
# add_executable(gemm_v3 gemm_v3.cu)
# add_executable(gemm_v4 gemm_v4.cu)
# add_executable(gemm_v5 gemm_v5.cu)
# add_executable(gemm_v6 gemm_v6.cu)


# 统一链接 cudart 和 cublas
foreach(tgt gemm_v0 gemm_v1)
  target_link_libraries(${tgt} PRIVATE CUDA::cudart CUDA::cublas)
endforeach()