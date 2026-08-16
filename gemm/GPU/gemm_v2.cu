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
const int THREAD_SIZE_M=8;//每个线程计算的C中元素的高度
const int THREAD_SIZE_N=8;//每个线程计算的C中元素的宽度
const int BLOCK_SIZE_M=64; ///每个线程块需要处理的M维度数据块大小
const int BLOCK_SIZE_N=64; ///每个线程块需要处理的N维度数据块大小
const int BLOCK_SIZE_K=8;  //每个线程块需要A load into sharemen的宽度
//每个线程块的所包含的线程数量
const int THREAD_SIZE_PER_BLCOK_M=BLOCK_SIZE_M/THREAD_SIZE_M;   //8
const int THREAD_SIZE_PER_BLCOK_N=BLOCK_SIZE_N/THREAD_SIZE_N;   //8
const int THREAD_NUM_PER_BLOCK = THREAD_SIZE_M*THREAD_SIZE_N;
__global__ void gemm_v2(int M, int N, int K, float alpha,float* A,float* B, float beta,float* C){
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    // 线性线程索引，
    const int tid = ty * blockDim.x + tx;     // 0..(blockDim.x*blockDim.y-1)
    const int numThreads = blockDim.x * blockDim.y;
    
    //这个thread block所独属于的globalmem 起始地址的计算
    A = &A[by*BLOCK_SIZE_M*K];
    B = &B[bx*BLOCK_SIZE_N];
    C = &C[by*BLOCK_SIZE_M*N+bx*BLOCK_SIZE_N];

    __shared__ float As[BLOCK_SIZE_M*BLOCK_SIZE_K];
    __shared__ float Bs[BLOCK_SIZE_K*BLOCK_SIZE_N];

    //线程块计算和传输是完全没关系的，传输的线程块甚至要“reshape”其线程摆布，不要用原本计算的思维来进行传输。
    //我们用thread_num来考虑搬运。

    //对于A来说，在K轴上安排BK个线程，那么在M轴上则可以排a_tile_m行线程。
    const int a_tile_m = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_K;
    //在M轴上，如果BLOCK_SIZE_M>a_tile_m，那么需要迭代a_iter_num_m次。
    const int a_iter_num_m = BLOCK_SIZE_M / a_tile_m;

    //线程索引，在smem a上的索引。
    int load_smem_a_m = tid / BLOCK_SIZE_K;
    int load_smem_a_k = tid % BLOCK_SIZE_K;

    //对于B来说，在N轴上安排BN个线程，那么在K轴上可以安排b_tile_k行线程。
    const int b_tile_k = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_N;
    //在K轴上，如果BLOCK_SIZE_K>b_tile_k，那么需要迭代b_iter_num_k次。
    const int b_iter_num_k = BLOCK_SIZE_K / b_tile_k;

    //线程索引，在smem b上的索引。
    int load_smem_b_k = tid / BLOCK_SIZE_N;
    int load_smem_b_n = tid % BLOCK_SIZE_N;

    //计算相关：
    float sum[THREAD_SIZE_M][THREAD_SIZE_N] = {0};
    
    //线程负责计算的数据在As和Bs中的起始位置
    // x 对应 N 方向（列），y 对应 M 方向（行）
    int read_smem_m = ty * THREAD_SIZE_M;
    int read_smem_n = tx * THREAD_SIZE_N;

  // K 方向分块迭代
  for (int kb = 0; kb < (int)K; kb += BLOCK_SIZE_K) {
    // 1) 装载 A 子块到共享内存 As[BM][BK]，采用跨 stride 方式覆盖 0..BM-1
    for(int i = 0;i < a_iter_num_m;i++){
        int smem_row_a = load_smem_a_m + i * a_tile_m; // 0..63 覆盖
        As[smem_row_a * BLOCK_SIZE_K + load_smem_a_k] = A[smem_row_a * K + load_smem_a_k];
    }
    // 2) 装载 B 子块到共享内存 Bs[BK][BN]，采用跨 stride 方式覆盖 0..BK-1
    for(int i = 0;i < b_iter_num_k;i++){
        int smem_row_bk = load_smem_b_k + i * b_tile_k; // 0..7 覆盖
        Bs[smem_row_bk * BLOCK_SIZE_N + load_smem_b_n] = B[smem_row_bk * N + load_smem_b_n];
    }
    __syncthreads();
    
    A += BLOCK_SIZE_K;
    B += BLOCK_SIZE_K * N;

    for(int kk = 0;kk < BLOCK_SIZE_K;kk++){
        for(int i = 0;i < THREAD_SIZE_M;i++){
            int a_row = read_smem_m + i; 
            for(int j = 0;j < THREAD_SIZE_N;j++){
                int b_col = read_smem_n + j;
                sum[i][j] += As[a_row * BLOCK_SIZE_K + kk] * Bs[kk * BLOCK_SIZE_N + b_col];
            }
        }
    }
    __syncthreads();
  }
  for(int i = 0;i < THREAD_SIZE_M;i++){
    for(int j = 0;j < THREAD_SIZE_N;j++){
        // 注意：C 指针已偏移到该 block 的起始位置，这里只能写相对坐标
        C[(read_smem_m + i) * N + (read_smem_n + j)] = sum[i][j];
    }
  }
}
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

int main(){
    std::vector<int> sizes = {128, 256, 512, 1024, 2048, 4096, 8192};

    // 打开CSV文件
    std::ofstream csv_file("sgemm_benchmark_v2.csv");
    csv_file << "Size,CUBLAS_GFLOPS,MySGEMM_FLOPS,Matched" << std::endl;
    for (int N : sizes){
        int M = N;
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
    
    // mysgemm_v2
      checkCudaError(cudaMemset(d_C_v1, 0, size), "cudaMemset d_C_v1 failed");
      dim3 threads(THREAD_SIZE_PER_BLCOK_N,THREAD_SIZE_PER_BLCOK_M);
      dim3 blocks((N+BLOCK_SIZE_N-1)/BLOCK_SIZE_N,(M+BLOCK_SIZE_M-1)/BLOCK_SIZE_M);

      for (int i = 0; i < warpup_time; ++i) {
            gemm_v2<<<blocks, threads>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
      }
      cudaDeviceSynchronize();

      checkCudaError(cudaEventRecord(start),
                     "cudaEventRecord(start v1) failed");
      for (int i = 0; i < repeat_time; ++i) {
            gemm_v2<<<blocks, threads>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
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