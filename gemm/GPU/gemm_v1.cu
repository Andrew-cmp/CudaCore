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

__global__ void gemm_v1(int M, int N, int K, float alpha,float* A,float* B, float beta,float* C){
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
    std::vector<int> sizes = {128, 256, 512, 1024, 2048, 4096, 8192};

    // 打开CSV文件
    std::ofstream csv_file("sgemm_benchmark_v1.csv");
    csv_file << "Size,CUBLAS_GFLOPS,MySGEMM_FLOPS,Matched" << std::endl;
    for (int N : sizes){

            size_t size = N * N * sizeof(float);
        float *A = (float *)malloc(size);
        float *B = (float *)malloc(size);
        float *C_cublas = (float *)malloc(size);
        float *C_v1 = (float *)malloc(size);
        float *C = (float *)malloc(size);

        float *d_A, *d_B, *d_C_v1;
        checkCudaError(cudaMalloc(&d_A, size), "cudaMalloc d_A failed");
        checkCudaError(cudaMalloc(&d_B, size), "cudaMalloc d_B failed");
        checkCudaError(cudaMalloc(&d_C_v1, size), "cudaMalloc d_C_v1 failed");
        for (int i = 0; i < N * N; ++i) {
            A[i] = 1.0f;
            B[i] = 2.0f;
        } 

      // 拷贝到设备
      checkCudaError(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice),
                     "cudaMemcpy A to device failed");
      checkCudaError(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice),
                     "cudaMemcpy B to device failed");

      cublasHandle_t handle;
      checkCublasError(cublasCreate(&handle), "cublasCreate failed");

      float alpha = 1.0f;
      float beta = 0.0f;

      cudaEvent_t start, stop;
      checkCudaError(cudaEventCreate(&start), "cudaEventCreate(start) failed");
      checkCudaError(cudaEventCreate(&stop), "cudaEventCreate(stop) failed");

      // warmup
      int warpup_time = 10;  // 热身次数
      for (int i = 0; i < warpup_time; ++i) {
        checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                     &alpha, d_B, N, d_A, N, &beta, d_C_v1, N),
                         "cublasSgemm failed");
      }
      cudaDeviceSynchronize();

      // cuBLAS SGEMM
      int repeat_time = 5;
      checkCudaError(cudaEventRecord(start),
                     "cudaEventRecord(start cublas) failed");
      for (int i = 0; i < repeat_time; ++i) {
        checkCublasError(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                     &alpha, d_B, N, d_A, N, &beta, d_C_v1, N),
                         "cublasSgemm failed");
      }

      checkCudaError(cudaEventRecord(stop),
                     "cudaEventRecord(stop cublas) failed");
      checkCudaError(cudaEventSynchronize(stop),
                     "cudaEventSynchronize cublas failed");

      float cublas_time = 0;
      checkCudaError(cudaEventElapsedTime(&cublas_time, start, stop),
                     "cudaEventElapsedTime cublas failed");

      // 拷贝 cuBLAS 结果
      checkCudaError(cudaMemcpy(C_cublas, d_C_v1, size, cudaMemcpyDeviceToHost),
                     "cudaMemcpy C_cublas failed");
    
    // mysgemm_v1
      checkCudaError(cudaMemset(d_C_v1, 0, size), "cudaMemset d_C_v1 failed");
      dim3 threads(BLOCK_SIZE_M, BLOCK_SIZE_N);
      dim3 blocks((N + threads.x - 1) / threads.x,
                  (N + threads.y - 1) / threads.y);

      for (int i = 0; i < warpup_time; ++i) {
            gemm_v1<<<blocks, threads>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
      }
      cudaDeviceSynchronize();

      checkCudaError(cudaEventRecord(start),
                     "cudaEventRecord(start v1) failed");
      for (int i = 0; i < repeat_time; ++i) {
            gemm_v1<<<blocks, threads>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
      }
      checkCudaError(cudaEventRecord(stop), "cudaEventRecord(stop v1) failed");
      checkCudaError(cudaEventSynchronize(stop),
                     "cudaEventSynchronize v1 failed");

      float v1_time = 0;
      checkCudaError(cudaEventElapsedTime(&v1_time, start, stop),
                     "cudaEventElapsedTime v1 failed");

      // 拷贝手写 kernel 结果
      checkCudaError(cudaMemcpy(C_v1, d_C_v1, size, cudaMemcpyDeviceToHost),
                     "cudaMemcpy C_v1 failed");
      // 结果比较
      int error_count = 0;
      for (int i = 0; i < N * N && error_count < 10; ++i) {
        if (fabsf(C_cublas[i] - C_v1[i]) > TOL) {
          error_count++;
        }
      }

      float cublas_gflops =
          repeat_time * 2.0f * N * N * N / (cublas_time * 1e6f);  // GFlops
      float v1_gflops =
          repeat_time * 2.0f * N * N * N / (v1_time * 1e6f);  // GFlops
      // 写入CSV
      csv_file << N << "," << cublas_gflops << "," << v1_gflops << ","
               << (error_count == 0 ? "1" : "0") << std::endl;

      // 释放资源
      cublasDestroy(handle);
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      cudaFree(d_A);
      cudaFree(d_B);
      cudaFree(d_C_v1);

      free(A);
      free(B);
      free(C_cublas);
      free(C_v1);
    }
    csv_file.close();
    std::cout << "Benchmark completed. Results saved to 'sgemm_benchmark.csv'"
            << std::endl;
    return 0;
}