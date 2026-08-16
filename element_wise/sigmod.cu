#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <chrono>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cmath>
#define FETCH(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
__global__ void sigmoid(float* x, float* y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) y[idx] = 1.0f / (1.0f + expf(-x[idx]));
}

// float4
__global__ void sigmoid_float4(float* x, float* y, int N) {
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    if (idx < N) {
        float4 tmp_x = FETCH(x[idx]);
        float4 tmp_y;
        tmp_y.x = 1.0f / (1.0f + expf(-tmp_x.x));
        tmp_y.y = 1.0f / (1.0f + expf(-tmp_x.y));
        tmp_y.z = 1.0f / (1.0f + expf(-tmp_x.z));
        tmp_y.w = 1.0f / (1.0f + expf(-tmp_x.w));
        FETCH(y[idx]) = tmp_y;
    }
}

int main(){
    int N = 1024 * 1024;
    float *x = (float *)malloc(N * sizeof(float));
    float *y = (float *)malloc(N * sizeof(float));
    for(int i = 0;i < N;i++){
        x[i] = rand() % 100;
        y[i] = 0.0f;
    }
    float *d_x, *d_y;
    cudaMalloc((void**)&d_x, N * sizeof(float));
    cudaMalloc((void**)&d_y, N * sizeof(float));
    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    sigmoid_float4<<<(N+255)/256, 256/4>>>(d_x, d_y, N);



    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time = 0;
    cudaEventElapsedTime(&time, start, stop);
    std::cout << "Time: " << time << " ms" << std::endl;
    cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 0;i < 10;i++){
        std::cout << y[i] << " ";
    }
    free(x);
    free(y);
    cudaFree(d_x);
    cudaFree(d_y);
    return 0;
}