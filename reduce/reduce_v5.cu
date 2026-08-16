#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>  // 用于 CPU 计时
const int BLOCK_SIZE = 1024;
const int N = 1024 * 1024;  // 1M elements
#define FETCH(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
//https://github.com/Tongkaio/CUDA_Kernel_Samples



bool check(float *out, float *res, int n)
{
    for (int i = 0; i < n; i++)
    {
        if (abs(out[i] - res[i]) > 0.005)
            return false;
    }
    return true;
}
template<int NUM>
__device__ float warp_reduce(float sum){
    if(NUM >= 32){
        sum += __shfl_down_sync(0xffffffff,sum,16);
    }
    if(NUM >= 16){
        sum += __shfl_down_sync(0xffffffff,sum,8);
    }
    if(NUM >= 8){
        sum += __shfl_down_sync(0xffffffff,sum,4);
    }
    if(NUM >= 4){
        sum += __shfl_down_sync(0xffffffff,sum,2);
    }
    if(NUM >= 2){
        sum += __shfl_down_sync(0xffffffff,sum,1);
    }
    return sum;
}
template<int BLOCK_SIZE>
__global__ void reduce(float * d_input, float * d_output){
    int tid = threadIdx.x;
    int idx = (blockDim.x * blockIdx.x + threadIdx.x)*4;
    float sum;
    if(idx < N){
        float4 tmp = FETCH(d_input[idx]);
        sum = tmp.x + tmp.y + tmp.z + tmp.w;
    }else{
        sum = 0.0f;
    }

    int lid = tid % 32;
    int wid = tid / 32;
    const int num_warps = BLOCK_SIZE / 32;
    __shared__ float shared[num_warps];
    sum = warp_reduce<BLOCK_SIZE>(sum);
    if(lid == 0)shared[wid] = sum;
    __syncthreads();
    if(wid == 0)sum = shared[lid];

    __syncthreads();
    
    sum = warp_reduce<num_warps>(sum);
    if(tid == 0){
        d_output[blockIdx.x] = sum;
    }


}
// CPU验证函数
float reduce_cpu(float * data) {
    float sum = 0.0f;
    for (int i = 0;i < N;i++) {
      sum += data[i];
    }
    return sum;
  }
  
int main(){
    const int BLOCK_NUM = ((N+BLOCK_SIZE-1)/BLOCK_SIZE)/4;

    float *input = (float *)malloc(N * sizeof(float));
    float *output = (float *)malloc(BLOCK_NUM * sizeof(float));
    float *final_output = (float *)malloc(1* sizeof(float));
    float *d_input;
    float *d_output;
    float *d_final_output;
    for(int i = 0;i < N;i++){
        input[i] = i%10;
    }

    // -------------------------------
    // CPU 计时开始
    auto cpu_start = std::chrono::high_resolution_clock::now();

    float cpu_result = reduce_cpu(input);

    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;
    // CPU 计时结束
    // -------------------------------
    std::cout << "CPU result: " << cpu_result << std::endl;
    std::cout << "CPU time: " << cpu_duration.count() << " ms" << std::endl;


    cudaMalloc((void **)&d_input,N*sizeof(float));
    cudaMalloc((void **)&d_output,BLOCK_NUM*sizeof(float));
    cudaMalloc((void **)&d_final_output,1*sizeof(float));

    cudaMemcpy(d_input, input, N*sizeof(float), cudaMemcpyHostToDevice);      // 修正4
    // -------------------------------
    // GPU 计时开始 (CUDA Events)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    dim3 grid(BLOCK_NUM);
    dim3 block(BLOCK_SIZE);
    reduce<BLOCK_SIZE><<<grid,block>>>(d_input, d_output);
    reduce<BLOCK_NUM><<<1, BLOCK_NUM>>>(d_output, d_final_output);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
  
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    // GPU 计时结束
    // -------------------------------

    cudaMemcpy(final_output, d_final_output, sizeof(float), cudaMemcpyDeviceToHost); // 修正5

    std::cout << "GPU kernel time: " << milliseconds << " ms" << std::endl;

    std::cout << "GPU result: " << *final_output << std::endl;


}

// int main(){
//     // printf("hello reduce\n");
//     const int N = 32 * 1024 * 1024;
//     float *input = (float *)malloc(N * sizeof(float));
//     float *d_input;
//     cudaMalloc((void **)&d_input, N * sizeof(float));

//     int block_num = N / THREAD_PER_BLOCK;
//     float *output = (float *)malloc((N / THREAD_PER_BLOCK) * sizeof(float));
//     float *d_output;
//     cudaMalloc((void **)&d_output, (N / THREAD_PER_BLOCK) * sizeof(float));
//     float *result = (float *)malloc((N / THREAD_PER_BLOCK) * sizeof(float));
//     for (int i = 0; i < N; i++)
//     {
//         input[i] = 2.0 * (float)drand48() - 1.0;
//     }
//     // cpu calc
//     for (int i = 0; i < block_num; i++)
//     {
//         float cur = 0;
//         for (int j = 0; j < THREAD_PER_BLOCK; j++)
//         {
//             cur += input[i * THREAD_PER_BLOCK + j];
//         }
//         result[i] = cur;
//     }

//     cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

//     dim3 Grid(N / THREAD_PER_BLOCK, 1);
//     dim3 Block(THREAD_PER_BLOCK, 1);
//     for (int i = 0; i < 50; i++)
//     reduce<<<Grid, Block>>>(d_input, d_output);
//     cudaMemcpy(output, d_output, block_num * sizeof(float), cudaMemcpyDeviceToHost);

//     if (check(output, result, block_num))
//         printf("the ans is right\n");
//     else
//     {
//         for (int i = 0; i < block_num; i++)
//         {
//             printf("%lf ", output[i]);
//         }
//         printf("\n");
//         printf("the ans is wrong\n");
//     }

//     cudaFree(d_input);
//     cudaFree(d_output);
//     return 0;
// }