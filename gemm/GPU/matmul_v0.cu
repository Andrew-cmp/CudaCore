#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>  // 用于 CPU 计时
#include <cublas_v2.h>

__global__ void gemm_v0(float* A,float* B,float* C,int M,int N,int K){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0;
    for(int i = 0; i < K; i++){
        sum += A[row*K+i] * B[i*N+col];
    }
    C[row*N+col] = sum;
}
void gemm_cpu(float* A,float* B,float* C,int M,int N,int K){
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            for(int k = 0; k < K; k++){
                C[i*N+j] += A[i*K+k] * B[k*N+j];
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


    dim3 block(16,16);
    dim3 grid((N+15)/16,(M+15)/16);

    gemm_v0<<<grid,block,0,stream_gemm>>>(d_A,d_B,d_C,M,N,K);

    cudaEventRecord(stop_event,stream_gemm);
    cudaEventSynchronize(stop_event);
    float duration_gpu;
    cudaEventElapsedTime(&duration_gpu,start_event,stop_event);
    std::cout << "GPU time: " << duration_gpu/1000.0f << " seconds" << std::endl;
    cudaMemcpy(C,d_C,sizeof(float)*M*N,cudaMemcpyDeviceToHost);
    std::cout << "GPU result: " << C[1000] << std::endl;
    std::cout << "GPU GFLOPS: " << (1.0*2*M*N*K/1e9)/(duration_gpu/1e3) << std::endl;


}