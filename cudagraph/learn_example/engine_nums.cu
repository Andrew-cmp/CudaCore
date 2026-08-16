#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount returned " << static_cast<int>(error) << ": " << cudaGetErrorString(error) << std::endl;
        return 1;
    }

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        error = cudaGetDeviceProperties(&deviceProp, device);

        if (error != cudaSuccess) {
            std::cerr << "cudaGetDeviceProperties returned " << static_cast<int>(error) << ": " << cudaGetErrorString(error) << std::endl;
            return 1;
        }

        std::cout << "Device " << device << ": " << deviceProp.name << std::endl;
        std::cout << "  asyncEngineCount: " << deviceProp.asyncEngineCount << std::endl;
    }

    return 0;
}