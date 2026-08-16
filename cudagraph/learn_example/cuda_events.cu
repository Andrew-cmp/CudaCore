#include <iostream>
#include <cuda_runtime.h>

// Kernel functions to perform computation
__global__ void kernel1(int64_t *data, int64_t repeat) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for (size_t i = 0; i < repeat; i++)
    {
        data[idx] += 1;
    }
}

__global__ void kernel2(int64_t *data, int64_t repeat) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for (size_t i = 0; i < repeat; i++)
    {
        data[idx] += 2;
    }
}

__global__ void kernel3(int64_t *data, int64_t repeat) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for (size_t i = 0; i < repeat; i++)
    {
        data[idx] -= 1;
    }
}

int main() {
    const int dataSize = 1024;
    const int printSize = 10;
    int64_t *h_data = new int64_t[dataSize]; // Host data
    int64_t *d_data1, *d_data2; // Device data

    // Initialize host data
    for (int i = 0; i < dataSize; i++) {
        h_data[i] = 0;
    }

    // Allocate memory on the device
    cudaMalloc((void**)&d_data1, dataSize * sizeof(int64_t));
    cudaMalloc((void**)&d_data2, dataSize * sizeof(int64_t));

    // Transfer data from host to device
    cudaMemcpy(d_data1, h_data, dataSize * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data2, h_data, dataSize * sizeof(int64_t), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(256);
    dim3 gridDim((dataSize + blockDim.x - 1) / blockDim.x);

    // Create streams and event
    cudaStream_t stream1, stream2;
    cudaEvent_t event1;
    int priorityHigh, priorityLow;
    cudaDeviceGetStreamPriorityRange(&priorityLow, &priorityHigh);
    cudaStreamCreate(&stream1);
    cudaStreamCreateWithPriority(&stream2, cudaStreamDefault, priorityHigh);
    cudaEventCreate(&event1);

    const int64_t repeat = 1000;

    // Execute kernel1 in stream1
    kernel1<<<gridDim, blockDim, 0, stream1>>>(d_data1, repeat);
    cudaEventRecord(event1, stream1); // Record event1 after kernel1 execution in stream1

    // Execute kernel2 in stream2, waiting for event1
    cudaStreamWaitEvent(stream2, event1, 0);

    kernel2<<<gridDim, blockDim, 0, stream2>>>(d_data1, repeat+1000);

    // Execute kernel3 in stream1 on a different array
    kernel3<<<gridDim, blockDim, 0, stream1>>>(d_data2, repeat+1000);
    // Synchronize streams
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // Transfer data back from device to host
    cudaMemcpy(h_data, d_data1, dataSize * sizeof(int64_t), cudaMemcpyDeviceToHost);

    // Display the result for d_data1
    std::cout << "Data after kernel1 and kernel2:" << std::endl;
    for (int i = 0; i < printSize; i++) {
        std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;

    // Transfer data back from device to host for d_data2
    cudaMemcpy(h_data, d_data2, dataSize * sizeof(int64_t), cudaMemcpyDeviceToHost);

    // Display the result for d_data2
    std::cout << "Data after kernel3:" << std::endl;
    for (int i = 0; i < printSize; i++) {
        std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory and destroy streams and event
    cudaFree(d_data1);
    cudaFree(d_data2);
    delete[] h_data;
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaEventDestroy(event1);

    return 0;
}