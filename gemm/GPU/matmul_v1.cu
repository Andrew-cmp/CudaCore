#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>  // 用于 CPU 计时
#include <cublas_v2.h>
#include <cmath>    // for fabsf
#include <fstream>  // for CSV output
#include <iostream>
#include <vector>
#define TOL 1e-5f
//对sharemem再次进行分块，将sharemem分块到register中。
const int THREAD_SIZE_M=1;//每个线程计算的C中元素的高度
const int THREAD_SIZE_N=1;//每个线程计算的C中元素的宽度
const int BLOCK_SIZE_M=32; ///每个线程块需要处理的M维度数据块大小
const int BLOCK_SIZE_N=32; ///每个线程块需要处理的N维度数据块大小
const int BLOCK_SIZE_K=32;  //每个线程块需要A load into sharemen的宽度
//每个线程块的所包含的线程数量
const int THREAD_SIZE_PER_BLCOK_M=BLOCK_SIZE_M/THREAD_SIZE_M; 
const int THREAD_SIZE_PER_BLCOK_N=BLOCK_SIZE_N/THREAD_SIZE_N;

void checkCudaError(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    std::cerr << msg << " CUDA ERROR: " << cudaGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }
}

void checkCublasError(cublasStatus_t status, const char *msg) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << msg << " CUBLAS ERROR: " << status << std::endl;
    exit(EXIT_FAILURE);
  }
}
///对M、N、K都进行tile，但必须要求BLOCK_SIZE_m\n\K都一致。
__global__ void gemm_v1(float* A,float* B,float* C,int M, int N, int K){
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float sum  = 0;
    __shared__ float A_shared[BLOCK_SIZE_M][BLOCK_SIZE_K];
    __shared__ float B_shared[BLOCK_SIZE_K][BLOCK_SIZE_N];

    const int numTiles = (K + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K;
    for(int num_k = 0;num_k < numTiles;num_k++){
        if(num_k*BLOCK_SIZE_K+tx <K && y < M)
            A_shared[ty][tx] = A[y*K+num_k*BLOCK_SIZE_K+tx];
        else
            A_shared[ty][tx] = 0;
        if(num_k*BLOCK_SIZE_K+ty < K && x<N)
            B_shared[ty][tx] = B[(num_k*BLOCK_SIZE_K+ty)*N + x];
        else
            B_shared[ty][tx] = 0;
        __syncthreads();
        for(int i = 0;i < BLOCK_SIZE_K;i++){
            sum += A_shared[ty][i]*B_shared[i][tx];
        }
        __syncthreads();
    }
    if(y < M && x < N)
        C[y*N+x] = sum;

}
void gemm_cpu(float* A,float* B,float* C,int M,int N,int K){
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            for(int num_k = 0; num_k < K; num_k++){
                C[i*N+j] += A[i*K+num_k] * B[num_k*N+j];
            }
        }
    }
}
int main(){

    const int M = 1024;
    const int N = 1024;
    const int K = 1024;
    float* A = (float * )malloc(sizeof(float)*M*K);
    float* B = (float * )malloc(sizeof(float)*K*N);
    float* C = (float * )malloc(sizeof(float)*M*N);

    float* d_A;
    float* d_B;
    float* d_C;
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            C[i*N+j] = 0;
        }
    }   
    for(int i = 0; i < M; i++){
        for(int j = 0; j < K; j++){
            A[i*K+j] = rand()%10;
        }
    }
    for(int i = 0; i < K; i++){ 
        for(int j = 0; j < N; j++){
            B[i*N+j] = rand()%10;
        }
    }
    auto start = std::chrono::high_resolution_clock::now();
    gemm_cpu(A,B,C,M,N,K);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "total GFLOPs: " << 1.0*2*M*N*K/1e9 << " GFLOPs" << std::endl;
    std::cout << "===================================================" << std::endl;
    std::chrono::duration<double> duration = end - start;
    std::cout << "CPU time: " << duration.count() << " seconds" << std::endl;
    std::cout << "CPU result: " << C[1000] << std::endl;
    std::cout << "CPU GFLOPS: " << 1.0*2*M*N*K/duration.count()/1e9 << std::endl;

    std::cout << "===================================================" << std::endl;

    cudaMalloc((void**)&d_A,sizeof(float)*M*K );
    cudaMalloc((void**)&d_B,sizeof(float)*K*N );
    cudaMalloc((void**)&d_C,sizeof(float)*M*N );
    cudaMemcpy(d_A,A,sizeof(float)*M*K,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B,sizeof(float)*K*N,cudaMemcpyHostToDevice);
    cudaMemset(d_C,0,sizeof(float)*M*N);

    cudaEvent_t start_event,stop_event;
    cudaStream_t stream_gemm;
    cudaStreamCreate(&stream_gemm);
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    cudaEventRecord(start_event,stream_gemm);

    //这里必须是BLOCK_SIZE_N,BLOCK_SIZE_M，否则会导致越界写入0而导致结果偏小。
    dim3 block(BLOCK_SIZE_N,BLOCK_SIZE_M);
    dim3 grid((N+BLOCK_SIZE_N-1)/BLOCK_SIZE_N,(M+BLOCK_SIZE_M-1)/BLOCK_SIZE_M);

    gemm_v1<<<grid,block,0,stream_gemm>>>(d_A,d_B,d_C,M,N,K);

    cudaEventRecord(stop_event,stream_gemm);
    cudaEventSynchronize(stop_event);
    float duration_gpu;
    cudaEventElapsedTime(&duration_gpu,start_event,stop_event);
    std::cout << "GPU time: " << duration_gpu/1000.0f << " seconds" << std::endl;
    cudaMemcpy(C,d_C,sizeof(float)*M*N,cudaMemcpyDeviceToHost);
    std::cout << "GPU result: " << C[1000] << std::endl;
    std::cout << "GPU GFLOPS: " << (1.0*2*M*N*K/1e9)/(duration_gpu/1e3) << std::endl;


}