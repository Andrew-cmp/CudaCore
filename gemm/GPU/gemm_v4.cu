#include <cublas_v2.h>

#include <cmath>    // for fabsf

#include <fstream>  // for CSV output

#include <iostream>
#include <nvToolsExt.h>
#include <vector>

#define TOL 1e-5f

#define OFFSET(row, col, lgd) ((row)*(lgd)+(col))

#define FETCH(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
// As: (BLOCK_SIZE_K, BLOCK_SIZE_M) 行主序
// Bs: (BLOCK_SIZE_K, BLOCK_SIZE_N) 行主序
//对sharemem再次进行分块，将sharemem分块到register中。
const int THREAD_SIZE_M=8;//每个线程计算的C中元素的高度
const int THREAD_SIZE_N=8;//每个线程计算的C中元素的宽度
const int BLOCK_SIZE_M=64; ///每个线程块需要处理的M维度数据块大小
const int BLOCK_SIZE_N=64; ///每个线程块需要处理的N维度数据块大小
const int BLOCK_SIZE_K=8;  //每个线程块需要A load into sharemen的宽度
//每个线程块的所包含的线程数量
const int THREAD_SIZE_PER_BLCOK_M=BLOCK_SIZE_M/THREAD_SIZE_M;   //8
const int THREAD_SIZE_PER_BLCOK_N=BLOCK_SIZE_N/THREAD_SIZE_N;   //8
const int THREAD_NUM_PER_BLOCK = THREAD_SIZE_PER_BLCOK_M*THREAD_SIZE_PER_BLCOK_N;
__device__ __forceinline__ float4 load4(const float* p) {
  // 假设 p 至少 16-byte 对齐；否则有些架构会慢或潜在异常
  return *reinterpret_cast<const float4*>(p);
}
//仅对M、K进行tile，这种情况下K不能太大，否则sharedmem不够用

__global__ void gemm_v4(int M, int N, int K, float alpha,float* A,float* B, float beta,float* C){
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    // 线性线程索引，
    const int tid = ty * blockDim.x + tx;     // 0..(blockDim.x*blockDim.y-1)

    // 在global mem读取A时，为防止后续的bank conflict 将其倒置，As形状为（BK，BM）
    __shared__ float As[2][BLOCK_SIZE_K*BLOCK_SIZE_M];
    __shared__ float Bs[2][BLOCK_SIZE_K*BLOCK_SIZE_N];
    //线程块计算和传输是完全没关系的，传输的线程块甚至要“reshape”其线程摆布，不要用原本计算的思维来进行传输。
    //我们用thread_num来考虑搬运。
    //对于A来说，在K轴上安排load_threads_a_k个线程，那么在M轴上则可以排load_threads_a_m行线程。
    const int load_threads_a_k = BLOCK_SIZE_K / 4;
    const int load_threads_a_m = THREAD_NUM_PER_BLOCK / (load_threads_a_k);
    //在M轴上，如果BLOCK_SIZE_M>load_threads_a_m，那么需要迭代num_row_strides_a次。
    const int num_row_strides_a = BLOCK_SIZE_M / load_threads_a_m;

    //线程索引，在smem a上的索引。
    int load_smem_a_m = tid / load_threads_a_k;
    int load_smem_a_k = tid % (load_threads_a_k) * 4;

    const int load_threads_b_n = BLOCK_SIZE_N / 4;
    //对于B来说，在N轴上安排BLOCK_SIZE_N个线程，那么在K轴上可以安排load_threads_b_k行线程。
    const int load_threads_b_k = THREAD_NUM_PER_BLOCK / load_threads_b_n;
    //在K轴上，如果BLOCK_SIZE_K>load_threads_b_k，那么需要迭代num_k_strides_b次。
    const int num_k_strides_b = BLOCK_SIZE_K / load_threads_b_k;
    //线程索引，在smem b上的索引。

    int load_smem_b_k = tid / load_threads_b_n;
    int load_smem_b_n = tid % (load_threads_b_n)* 4;
    // ldg=load global mem
    // 暂存global mem的值
    // 表示一个线程需要进行load global的float4的指令条数


    const int ldg_a_num = BLOCK_SIZE_K * BLOCK_SIZE_M / THREAD_NUM_PER_BLOCK / 4;
    const int ldg_b_num = BLOCK_SIZE_K * BLOCK_SIZE_N / THREAD_NUM_PER_BLOCK / 4;

    //这个thread block所独属于的globalmem 起始地址的计算
    A = &A[by*BLOCK_SIZE_M*K];
    B = &B[bx*BLOCK_SIZE_N];
    C = &C[by*BLOCK_SIZE_M*N+bx*BLOCK_SIZE_N];


    // x 对应 N 方向（列），y 对应 M 方向（行）
    int comp_m = ty * THREAD_SIZE_M;
    int comp_n = tx * THREAD_SIZE_N;

    float accum[THREAD_SIZE_M][THREAD_SIZE_N] = {0.};


    float a_frag[2][THREAD_SIZE_M];
    float b_frag[2][THREAD_SIZE_N];
    // ldg=load global mem
    // 暂存global mem的值,避免对float数组做未对齐的float4写入
    float ldg_a_reg[4 * ldg_a_num] = {0.};
    float ldg_b_reg[4 * ldg_b_num] = {0.};
    
    // global mem -> shared mem k轮
    #pragma unroll
    for(int i = 0;i < num_row_strides_a;i++){
        int row = load_smem_a_m + i * load_threads_a_m;   // 0..63
        int col = load_smem_a_k;                           // 0 or 4 (within BLOCK_SIZE_K)
        // 安全读取4个连续元素
        float4 va = load4(&A[row * K + col]);              // A is already block-offset, K is global ld
        // 将以上的一行的4个元素，按列排布在共享显存中
        As[0][OFFSET(col + 0, row, BLOCK_SIZE_M)] = va.x;
        As[0][OFFSET(col + 1, row, BLOCK_SIZE_M)] = va.y;
        As[0][OFFSET(col + 2, row, BLOCK_SIZE_M)] = va.z;
        As[0][OFFSET(col + 3, row, BLOCK_SIZE_M)] = va.w;
    }
    // 2) 装载 B 子块到共享内存 Bs[BLOCK_SIZE_K][BLOCK_SIZE_N]，采用跨 stride 方式覆盖 0..BLOCK_SIZE_K-1
    for(int i = 0;i < num_k_strides_b;i++){
        // k in [0, BLOCK_SIZE_K), n in [0, BLOCK_SIZE_N)
        int row = load_smem_b_k + i * load_threads_b_k;  // 0..7
        int col = load_smem_b_n;                         // 0,4,8,...,60
        float4 vb = load4(&B[row * N + col]);              // B is already block-offset, N is global ld
        // Bs is BLOCK_SIZE_K x BLOCK_SIZE_N, ld = BLOCK_SIZE_N
        Bs[0][OFFSET(row, col + 0, BLOCK_SIZE_N)] = vb.x;
        Bs[0][OFFSET(row, col + 1, BLOCK_SIZE_N)] = vb.y;
        Bs[0][OFFSET(row, col + 2, BLOCK_SIZE_N)] = vb.z;
        Bs[0][OFFSET(row, col + 3, BLOCK_SIZE_N)] = vb.w;
    }

    __syncthreads();
    // shared mem -> register k轮数据
    // 每个线程取出THREAD_SIZE_M个数据放入register中
    #pragma unroll
    for (int m = 0; m < THREAD_SIZE_M; m++) {
        (a_frag[0][m]) = (As[0][OFFSET(0, comp_m + m, BLOCK_SIZE_M)]);
    }
    #pragma unroll
    for (int n = 0; n < THREAD_SIZE_N; n++) {
        (b_frag[0][n]) = (Bs[0][OFFSET(0, comp_n + n, BLOCK_SIZE_N)]);
    }
    int write_index = 1;
    int load_index = 0;
    int k = 0;
    do {
        k += BLOCK_SIZE_K;
        //global mem -> shared mem 前置步骤 -> register k+1 轮数据 
        if(k < K){
        #pragma unroll
            for (int i = 0; i < BLOCK_SIZE_M; i += load_threads_a_m) {
              int ldg_index = i / load_threads_a_m * 4;
              FETCH(ldg_a_reg[ldg_index]) =
                  FETCH(A[OFFSET(load_smem_a_m + i, k + load_smem_a_k, K)]);
            }
        #pragma unroll
            for (int i = 0; i < BLOCK_SIZE_K; i += load_threads_b_k) {
              int ldg_index = i / load_threads_b_k * 4;
              FETCH(ldg_b_reg[ldg_index]) =
                  FETCH(B[OFFSET(k + load_smem_b_k + i, load_smem_b_n, N)]);
            }
        }
        load_index = write_index ^ 1;
        // shared mem -> register k+1轮数据
        for(int kk = 0;kk < BLOCK_SIZE_K - 1;kk++){
            #pragma unroll
            for(int i = 0;i < THREAD_SIZE_M; i+=4){
                FETCH(a_frag[(kk+1)%2][i]) = FETCH(As[load_index][OFFSET(kk + 1, comp_m + i, BLOCK_SIZE_M)]);

            }
            #pragma unroll
            for(int j = 0;j < THREAD_SIZE_N; j+=4){
                FETCH(b_frag[(kk+1)%2][j]) = FETCH(Bs[load_index][OFFSET(kk + 1, comp_n + j, BLOCK_SIZE_N)]) ;
            }
            // compute k轮
            #pragma unroll
            for(int i = 0;i < THREAD_SIZE_M;i++){
                #pragma unroll
                for(int j = 0;j < THREAD_SIZE_N;j++){
                    accum[i][j] += a_frag[(kk)%2][i] * b_frag[(kk)%2][j];
                }
            }
        }
        if (k < K) {
            // load global mem ->shared mem 阶段的register to As k+1轮
            #pragma unroll
            for (int i = 0; i < BLOCK_SIZE_M; i += load_threads_a_m) {
                int row = load_smem_a_m + i * load_threads_a_m;   // 0..63
                int col = load_smem_a_k;                           // 0 or 4 (within BLOCK_SIZE_K)
                int ldg_index = i / load_threads_a_m * 4;
                As[write_index][OFFSET(col    , row, BLOCK_SIZE_M)] =
                    ldg_a_reg[ldg_index];
                As[write_index][OFFSET(col + 1, row, BLOCK_SIZE_M)] =
                    ldg_a_reg[ldg_index + 1];
                As[write_index][OFFSET(col + 2, row, BLOCK_SIZE_M)] =
                    ldg_a_reg[ldg_index + 2];
                As[write_index][OFFSET(col + 3, row, BLOCK_SIZE_M)] =
                    ldg_a_reg[ldg_index + 3];
            }
            #pragma unroll
            for (int i = 0; i < BLOCK_SIZE_K; i += load_threads_b_k) {
              int ldg_index = i / load_threads_b_k * 4;
              FETCH(Bs[write_index][OFFSET(load_smem_b_k + i, load_smem_b_n, BLOCK_SIZE_N)]) =
                  FETCH(ldg_b_reg[ldg_index]);
            }
            __syncthreads();
            // shared mem->register k+1轮
            #pragma unroll
            for (int m = 0; m < THREAD_SIZE_M; m += 4) {
                FETCH(a_frag[0][m]) =
                FETCH(As[write_index][OFFSET(0, comp_m + m, BLOCK_SIZE_M)]);
            }
            #pragma unroll
            for (int n = 0; n < THREAD_SIZE_N; n += 4) {
                FETCH(b_frag[0][n]) =
                    FETCH(Bs[write_index][OFFSET(0, comp_n + n, BLOCK_SIZE_N)]);
            }
            write_index ^= 1;
        }
        // 最后一轮的 compute
        #pragma unroll
        for (int m = 0; m < THREAD_SIZE_M; m++) {
            #pragma unroll
            for (int n = 0; n < THREAD_SIZE_N; n++) {
                accum[m][n] += a_frag[(BLOCK_SIZE_K - 1) % 2][m] * b_frag[(BLOCK_SIZE_K - 1) % 2][n];
            }
        }
    }while (k < K);

    #pragma unroll
    for (int m = 0; m < THREAD_SIZE_M; m++) {
    #pragma unroll
      for (int n = 0; n < THREAD_SIZE_N; n += 4) {
        float4 ctmp = FETCH(C[OFFSET(comp_m + m, comp_n + n, N)]);
        ctmp.x = alpha * accum[m][n] + beta * ctmp.x;
        ctmp.y = alpha * accum[m][n + 1] + beta * ctmp.y;
        ctmp.z = alpha * accum[m][n + 2] + beta * ctmp.z;
        ctmp.w = alpha * accum[m][n + 3] + beta * ctmp.w;
        FETCH(C[OFFSET(comp_m + m, comp_n + n, N)]) = ctmp;
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
    std::vector<int> sizes = {128, 256, 512, 1024, 2048, 4096};

    // 打开CSV文件
    std::ofstream csv_file("sgemm_benchmark_v3.csv");
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
    
    nvtxRangePushA("NCU_GEMM");
    // mysgemm_v4
      checkCudaError(cudaMemset(d_C_v1, 0, size), "cudaMemset d_C_v1 failed");
      dim3 threads(THREAD_SIZE_PER_BLCOK_N,THREAD_SIZE_PER_BLCOK_M);
      dim3 blocks((N+BLOCK_SIZE_N-1)/BLOCK_SIZE_N,(M+BLOCK_SIZE_M-1)/BLOCK_SIZE_M);

      for (int i = 0; i < warpup_time; ++i) {
            gemm_v4<<<blocks, threads>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
      }
      cudaDeviceSynchronize();

      nvtxRangePop();
      checkCudaError(cudaEventRecord(start),
                     "cudaEventRecord(start v1) failed");
      for (int i = 0; i < repeat_time; ++i) {
            gemm_v4<<<blocks, threads>>>(N, N, N, alpha, d_A, d_B, beta, d_C_v1);
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
