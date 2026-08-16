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
#include <cfloat>

#define FETCH(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
__device__ inline float warp_reduce_max_f32(float val){
    #pragma unroll
    for(int i = 16 ;i >= 1;i >>= 1){
        val = fmaxf(val,__shfl_up_sync(0xffffffff,val,i));
    }
    return val;
}
__device__ inline float warp_reduce_sum_f32(float val){   
    #pragma unroll
    for(int i = 16 ;i >= 1;i >>= 1){
        val += __shfl_up_sync(0xffffffff, val, i);
    }
    return val;
}
// NUM_THREAD必须 <= 32*32=1024，因为第一次warp的结果必须小于32个
template <const int NUM_THREADS>
__device__ float block_reduce_max_f32(float val){
    static_assert(NUM_THREADS % 32 == 0, "NUM_THREADS must be a multiple of 32");
    constexpr int NUM_WARPS = NUM_THREADS/32;
    int tid = threadIdx.x;
    int lid = tid / 32;
    int wid = tid % 32;
    __shared__ float shared_max[NUM_WARPS];

    // 每个warp内部归约
    float value = warp_reduce_max_f32(val);
    if(wid == 0)shared[lid] = value;
    //这里不对，不应该只让lid==0的warp进行计算，应该让所有warp计算，因为需要为所有线程都生成最后结果。
    // if(lid == 0){
    //     value = shared[wid];
    // }

    value = (wid < NUM_WARPS) ? shared[wid] : -FLT_MAX;
    __syncthreads();
    value = warp_reduce_max_f32(value);
    ///这里不能只tid==0的线程返回，需要全部线程都返回max value，因为每个线程都需要。
    //if(tid == 0)return value;
    // WRAN: need to broadcast value to all threads within warp
    //每个线程都会拿到warp 内wid = 0的value
    value = __shfl_sync(0xffffffff,value,0,32);
    return value;

}
template <const int NUM_THREADS>
__device__ float block_reduce_sum_f32(float val){
    static_assert(NUM_THREADS % 32 == 0, "NUM_THREADS must be a multiple of 32");
    constexpr int NUM_WARPS = NUM_THREADS/32;
    __shared__ float shared[NUM_WARPS];
    
    int lid = threadIdx.x/32;
    int wid = threadIdx.x%32;
    float value = warp_reduce_sum_f32(val);
    if(wid == 0)shared[lid] = value;
    __syncthreads();
    value = (wid < NUM_WARPS) ? shared[wid] : -FLT_MAX;
    value = warp_reduce_sum_f32(value);

    value = __shfl_sync(0xffffffff,value,0,32);
    return value;

}
template<const int NUM_THREADS>
__global__ void softmax(float* x, float* y, int N){
    int idx = (threadIdx.x+blockDim.x*blockIdx.x)*4;

    float4 tmpx;
    if (idx + 3 < N) {
        tmpx = FETCH(x[idx]);
    } else {
        tmpx.x = (idx     < N) ? x[idx]     : -FLT_MAX;
        tmpx.y = (idx + 1 < N) ? x[idx + 1] : -FLT_MAX;
        tmpx.z = (idx + 2 < N) ? x[idx + 2] : -FLT_MAX;
        tmpx.w = (idx + 3 < N) ? x[idx + 3] : -FLT_MAX;
    }

    // 先做四次maxx，再做block reduce
    float maxx = tmpx.x;
    maxx = fmaxf(tmpx.y,maxx); 
    maxx = fmaxf(tmpx.z,maxx); 
    maxx = fmaxf(tmpx.w,maxx);

    maxx = block_reduce_max_f32<NUM_THREADS>(maxx);

    tmpx.x -= maxx;
    tmpx.y -= maxx;
    tmpx.z -= maxx;
    tmpx.w -= maxx;

    float sum = expf(tmpx.x) + expf(tmpx.y) + expf(tmpx.z) + expf(tmpx.w);
    sum = block_reduce_sum_f32<NUM_THREADS>(sum);
    // e^x_i/sum(e^x_0,...,e^x_n-1)
    if (idx + 3 < N) {
      float4 reg_y;
      reg_y.x = expf(tmpx.x) / (sum);
      reg_y.y = expf(tmpx.y) / (sum);
      reg_y.z = expf(tmpx.z) / (sum);
      reg_y.w = expf(tmpx.w) / (sum);
      FETCH(y[idx]) = reg_y;
    } else {
      if (idx     < N) y[idx]     = expf(tmpx.x) / sum;
      if (idx + 1 < N) y[idx + 1] = expf(tmpx.y) / sum;
      if (idx + 2 < N) y[idx + 2] = expf(tmpx.z) / sum;
      if (idx + 3 < N) y[idx + 3] = expf(tmpx.w) / sum;
    }
}

void softmax_cpu(float* input,float* output,int N){
    float maxx = -FLT_MAX;
    for(int i = 0;i < N;i++){
        maxx = std::max(maxx,input[i]);
    }
    float div = 0;
    for (int i = 0; i < N; i++) {
        output[i] = std::exp(input[i] -maxx);
        div += output[i];
    }
    for (int i = 0; i < N; i++) {
        output[i] /= div;
    }
}


int main(){


    int N = 1024;
    float *x = (float *)malloc(N * sizeof(float));
    float *y = (float *)malloc(N * sizeof(float));
    float *y_cpu = (float*)malloc(N * sizeof(float));
    for(int i = 0;i < N;i++){
        x[i] = rand() % 10;
        y[i] = 0.0f;
        y_cpu[i] = 0.0f;
    }
    float *d_x, *d_y;
    cudaMalloc((void**)&d_x, N * sizeof(float));
    cudaMalloc((void**)&d_y, N * sizeof(float));
    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaEvent_t start, stop;    

    constexpr int block_size = 1024;
    // 每个线程处理4个元素
    int grid_size  = (N + (block_size * 4) - 1) / (block_size * 4);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    softmax<block_size><<<grid_size, block_size>>>(d_x, d_y, N);
    softmax_cpu(x,y_cpu,N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time = 0;
    cudaEventElapsedTime(&time, start, stop);
    std::cout << "Time: " << time << " ms" << std::endl;
    cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 0;i < 10;i++){
        printf("y[%d]: %f\n", i, y[i]);
        printf("y_cpu[%d]: %f\n", i, y_cpu[i]);
    }

    return 0;
}