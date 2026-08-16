#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>  // 用于 CPU 计时
#include <cublas_v2.h>
void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
      std::cerr << msg << " CUDA ERROR: " << cudaGetErrorString(err) << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  
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
template<int M,int N,int K>
__global__ void gemm_v2(float* A,float* B,float* C){
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
    memset(C,sizeof(float)*M*N,0);
    cudaEvent_t start_event,stop_event;
    cudaStream_t stream_gemm;
    cudaStreamCreate(&stream_gemm);
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    cudaEventRecord(start_event,stream_gemm);

    //注意，这两个顺序不能换，是和上面的x、y的顺序有关的。
    //x 轴放 BLOCK_SIZE_N（对应矩阵的列方向 N），y 轴放 BLOCK_SIZE_M（对应行方向 M）。
    //也就是说thread block的维度是BLOCK_SIZE_M*BLOCK_SIZE_N，顺序不一样不要搞混
    dim3 block(THREAD_SIZE_PER_BLCOK_N,THREAD_SIZE_PER_BLCOK_M);
    dim3 grid((N+BLOCK_SIZE_N-1)/BLOCK_SIZE_N,(M+BLOCK_SIZE_M-1)/BLOCK_SIZE_M);

    gemm_v2<M,N,K><<<grid,block,0,stream_gemm>>>(d_A,d_B,d_C);
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "Launch error: " << cudaGetErrorString(err) << std::endl;
    }
    cudaEventRecord(stop_event,stream_gemm);
    cudaEventSynchronize(stop_event);
    float duration_gpu = 0;
    checkCudaError(cudaEventElapsedTime(&duration_gpu,start_event,stop_event),"time error");
    std::cout << "GPU time: " << duration_gpu<< " ms" << std::endl;
    std::cout << "GPU time: " << 1.0*duration_gpu/1000.0f << " seconds" << std::endl;
    cudaMemcpy(C,d_C,sizeof(float)*M*N,cudaMemcpyDeviceToHost);
    std::cout << "GPU result: " << C[1000] << std::endl;
    std::cout << "GPU GFLOPS: " << (1.0f*2*M*N*K/1e9)/(1.0f*duration_gpu/1e3) << std::endl;


}