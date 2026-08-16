cmake_minimum_required(VERSION 3.20.0)
project(cuda_learning_practice_and_notes VERSION 0.1.0 LANGUAGES CUDA CXX C)
find_package(CUDAToolkit)
add_subdirectory(reduce)