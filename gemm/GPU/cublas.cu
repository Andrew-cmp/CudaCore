
#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>  // 用于 CPU 计时
#include <cublas_v2.h>


int main(){
    
    cudaEvent_t start_event,stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    std::cout << "===================================================" << std::endl;
    int M = 1024;
    int N = 1024;
    int K = 1024;
    float *d_A,*d_B,*d_C;
    cudaMalloc((void**)&d_A,sizeof(float)*M*K );
    cudaMalloc((void**)&d_B,sizeof(float)*K*N );
    cudaMalloc((void**)&d_C,sizeof(float)*M*N );
    float *A = (float*)malloc(sizeof(float)*M*K);
    float *B = (float*)malloc(sizeof(float)*K*N);
    float *C = (float*)malloc(sizeof(float)*M*N);
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
    cudaMemcpy(d_A,A,sizeof(float)*M*K,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B,sizeof(float)*K*N,cudaMemcpyHostToDevice);
    cudaMemcpy(d_C,C,sizeof(float)*M*N,cudaMemcpyHostToDevice);



    cublasHandle_t handle;
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasStatus_t status = cublasCreate(&handle);
    cudaStream_t stream_cublas;
    cudaStreamCreate(&stream_cublas);
    cudaEventRecord(start_event,stream_cublas);
    cublasSetStream(handle, stream_cublas);
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N, N, M, K,&alpha,d_B, N, d_A, K,&beta, d_C, N);
    cudaEventRecord(stop_event,stream_cublas);
    cudaEventSynchronize(stop_event);
    float duration_cublas;
    cudaEventElapsedTime(&duration_cublas,start_event,stop_event);
    std::cout << "cublas time: " << duration_cublas/1000.0f << " seconds" << std::endl;
    cudaMemcpy(C,d_C,sizeof(float)*M*N,cudaMemcpyDeviceToHost);
    std::cout << "cublas GFLOPS: " << (1.0*2*M*N*K/1e9)/(duration_cublas/1e3) << std::endl;   
}