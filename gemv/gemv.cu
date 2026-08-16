#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h> 

#define FETCH(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
#define CEIL(a,b) ((a)+((b)-1))/(b)
#define checkCudaErrors(func) {                                                   \
    cudaError_t e = (func);                                                       \
    if(e != cudaSuccess)                                                          \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));  \
}
__device__ float warpReduce(float val){

    #pragma unroll
    for(int offset = 16;offset >=1;offset >>= 1){
        val += __shfl_down_sync(0xffffffff,val,offset);
    }
    return val;
}
// dim3 dimGrid(M/128/4);
// dim3 dimBlock(4,32);
// gemv 针对不同的N都有不同的优化方式
__global__ void sgemv_k32(float* A, float* x, float* y, int M, int N) {
    
    // 全局线程 id
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lid = tid % 32;    // warp 内 thread id
    int wid = tid / 32;    // 全局 warp id = 当前处理的行号

    int current_row = wid; // 一 warp 一行
    if (current_row < M) {

        int kIteration = (N / 32) / 4; 
        if (kIteration == 0) kIteration = 1;

        // A 指针跳到当前行
        A = &A[current_row * N];

        float res = 0.0f;

        #pragma unroll
        for (int i = 0; i < kIteration; i++) {
            int current_col = (i * 32 * 4 + lid * 4);
            if (current_col + 3 < N) { // 避免越界
                float4 cur_a = FETCH(A[current_col]);
                float4 cur_x = FETCH(x[current_col]);
                res += cur_a.x * cur_x.x;
                res += cur_a.y * cur_x.y;
                res += cur_a.z * cur_x.z;
                res += cur_a.w * cur_x.w;
            }
        }

        res = warpReduce(res);
        if (lid == 0) y[current_row] = res;
    }
}

int main() {
    size_t M = 1024;
    size_t K = 128;

    size_t bytes_A = sizeof(float) * M * K;
    size_t bytes_x = sizeof(float) * K;
    size_t bytes_y = sizeof(float) * M;
    float* h_A  = (float*)malloc(bytes_A);
    float* h_x  = (float*)malloc(bytes_x);
    float* h_y  = (float*)malloc(bytes_y);
    float* h_y1 = (float*)malloc(bytes_y);

    float* d_A;
    float* d_x;
    float* d_y;

    checkCudaErrors(cudaMalloc(&d_A, bytes_A));
    checkCudaErrors(cudaMalloc(&d_x, bytes_x));
    checkCudaErrors(cudaMalloc(&d_y, bytes_y));

    double duration[2] = {0, 0};
    double GFLOPS[2] = {0, 0};
    double GFLOPs = 2.0 * M * 1 * K;

    // 生成A的数据
    for( int i = 0; i < M * K; i++ ) {
        h_A[i] = (float)i/K;
    }

    // 生成x的数据
    for( int i = 0; i < K; i++ ) {
        h_x[i] = 1;
    }
    memset(h_y,  0, M * sizeof(float));
    memset(h_y1, 0, M * sizeof(float));

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float msecTotal = 0;
    int iteration = 1000;

    checkCudaErrors(cudaMemcpy( d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( d_x, h_x, bytes_x, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( d_y, h_y, bytes_y, cudaMemcpyHostToDevice));
    
    checkCudaErrors(cudaEventRecord(start));

    for (int run = 0 ; run < iteration; run ++ ) {
        dim3 dimGrid((M)/4);
        dim3 dimBlock(128);
        sgemv_k32<<<dimGrid, dimBlock>>>(d_A, d_x, d_y, M, K);
    }

    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
    checkCudaErrors(cudaMemcpy( h_y, d_y, bytes_y, cudaMemcpyDeviceToHost));

    duration[0] = msecTotal / iteration;
    GFLOPS[0] = (GFLOPs * 1.0e-9f) / (duration[0] / 1000.0f);
    printf( "My gemm Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
        GFLOPS[0],
        duration[0],
        GFLOPs);

    // cublas
    cublasHandle_t blas_handle;  
    cublasCreate(&blas_handle);
    float alpha = 1.0;
    float beta = 0;
    checkCudaErrors(cudaMemcpy( d_y, h_y1, bytes_y, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaEventRecord(start));
    for (int run = 0 ; run < iteration; run ++ ) {
        cublasSgemv (blas_handle, CUBLAS_OP_T, 
            K, M, &alpha, 
            d_A, K, d_x, 1, &beta, d_y, 1
        );
    }

    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    checkCudaErrors(cudaMemcpy( h_y1, d_y, bytes_y, cudaMemcpyDeviceToHost));

    duration[1] = msecTotal / iteration;
    GFLOPS[1] = (GFLOPs * 1.0e-9f) / (duration[1] / 1000.0f);
    printf( "CuBlas Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
        GFLOPS[1],
        duration[1],
        GFLOPs);

    cublasDestroy(blas_handle);
    
    double eps = 1.e-6;  // machine zero
    bool correct = true;
    for (int i = 0; i < M; i++) {
        double abs_err = fabs(h_y[i] - h_y1[i]);
        double dot_length = M;
        double abs_val = fabs(h_y[i]);
        double rel_err = abs_err / abs_val / dot_length;
        if (rel_err > eps) {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                    i, h_y[i], h_y1[i], eps);
            correct = false;
            break;
        }
    }

    printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");
    
    // Free Memory
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
    
    free(h_A);
    free(h_x);
    free(h_y);
    free(h_y1);
}