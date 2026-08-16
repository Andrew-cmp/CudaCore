#include <cuda_runtime.h>
#include <stdio.h>
#define ELE_TYPE float
template<int M,int N,int K>
__global__ void gemm_kernel(ELE_TYPE* A, ELE_TYPE* B,ELE_TYPE* C){
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    ELE_TYPE sum = 0;
    if(x < N&&y<M){
        for(int i = 0 ;i < K;i++){
            sum += A[y * K+i]*B[i*N+x];
        }
        C[y*N + x] = sum;
    }
    
}

int main(){

    const int N = 1024;
    const int M = 1024;
    const int K = 1024;
    int size_A = M*K*sizeof(ELE_TYPE);
    int size_B = K*N*sizeof(ELE_TYPE);
    int size_C = M*N*sizeof(ELE_TYPE);
    ELE_TYPE * h_a =(ELE_TYPE*)malloc(size_A);
    ELE_TYPE * h_b =(ELE_TYPE*)malloc(size_B);
    ELE_TYPE * h_c =(ELE_TYPE*)malloc(size_C);
    for (int i = 0; i < M * K; ++i) h_a[i] = 2.0f;
    for (int i = 0; i < K * N; ++i) h_b[i] = 2.0f;
    ELE_TYPE *d_a, *d_b, *d_c;
    cudaMalloc(&d_a,M*K*sizeof(ELE_TYPE));
    cudaMalloc(&d_b,K*N*sizeof(ELE_TYPE));
    cudaMalloc(&d_c,M*N*sizeof(ELE_TYPE));
    
    cudaMemcpy(d_a,h_a,size_A,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,size_B,cudaMemcpyHostToDevice);
    
    // 创建CUDA事件用于计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 记录开始时间
    cudaEventRecord(start);
    
    dim3 blockDim(16,16);
    dim3 gridDim((N+blockDim.x-1)/blockDim.x,
                      (M+blockDim.y-1)/blockDim.y );
    //草，大模型给的代码，下面的GridDim和blockDim位置对调了
    //gemm_kernel<M,N,K><<<blockDim,gridDim>>>(d_a,d_b,d_c);
    gemm_kernel<M,N,K><<<gridDim,blockDim>>>(d_a,d_b,d_c);

    // 记录结束时间
    cudaEventRecord(stop);
    
    // 同步事件，确保GPU计算完成
    cudaEventSynchronize(stop);
    
    // 计算时间差（毫秒）
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaMemcpy(h_c,d_c,size_C,cudaMemcpyDeviceToHost);

    // 验证结果（只打印第一个元素）
    printf("结果矩阵第一个元素: %.1f\n", h_c[0]);
    
    // 输出计算时间
    printf("\nGPU计算时间: %.3f 毫秒\n", milliseconds);
    printf("\nGPU计算速度: %.2f GFLOPS\n", (sizeof(ELE_TYPE) * M * N * K) / (milliseconds / 1000.0f) / 1e9);
    
    // 销毁CUDA事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    
}