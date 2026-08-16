#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>  // 用于 CPU 计时
#include <cublas_v2.h>
#define OFFSET(row, col, lgd) ((row)*(lgd)+(col))
#define FETCH(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
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
//仅对M、K进行tile，这种情况下K不能太大，否则sharedmem不够用
template<int M,int N,int K>
__global__ void gemm_v2(float* A,float* B,float* C){
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    // 线性线程索引，
    const int tid = ty * blockDim.x + tx;     // 0..(blockDim.x*blockDim.y-1)
    

    __shared__ float As[BLOCK_SIZE_M*BLOCK_SIZE_K];
    __shared__ float Bs[BLOCK_SIZE_K*BLOCK_SIZE_N];

    //线程块计算和传输是完全没关系的，传输的线程块甚至要“reshape”其线程摆布，不要用原本计算的思维来进行传输。
    //我们用thread_num来考虑搬运。

    //对于A来说，在K轴上安排load_threads_a_k个线程，那么在M轴上则可以排load_threeads_a_m行线程。
    const int load_threads_a_k = BLOCK_SIZE_K / 4;
    const int load_threeads_a_m = THREAD_NUM_PER_BLOCK / (load_threads_a_k);
    //在M轴上，如果BLOCK_SIZE_M>load_threeads_a_m，那么需要迭代num_row_strides_a次。
    const int num_row_strides_a = BLOCK_SIZE_M / load_threeads_a_m;

    //线程索引，在smem a上的索引。
    int load_smem_a_m = tid / load_threads_a_k;
    int load_smem_a_k = tid % (load_threads_a_k) *4;

    const int load_threads_b_n = BLOCK_SIZE_N / 4;
    //对于B来说，在N轴上安排BN个线程，那么在K轴上可以安排load_threads_b_k行线程。
    const int load_threads_b_k = THREAD_NUM_PER_BLOCK / load_threads_b_n;
    //在K轴上，如果BLOCK_SIZE_K>load_threads_b_k，那么需要迭代num_k_strides_b次。
    const int num_k_strides_b = BLOCK_SIZE_K / load_threads_b_k;

    //线程索引，在smem b上的索引。
    int load_smem_b_k = tid / load_threads_b_n;
    int load_smem_b_n = tid % (load_threads_b_n)* 4;

    const int ldg_a_num = BLOCK_SIZE_K * BLOCK_SIZE_M / THREAD_NUM_PER_BLOCK / 4;
    // 使用float4寄存缓存，避免对float数组做未对齐的float4写入
    float4 ldg_a_reg[ldg_a_num];
    //这个thread block所独属于的globalmem 起始地址的计算
    A = &A[by*BLOCK_SIZE_M*K];
    B = &B[bx*BLOCK_SIZE_N];
    C = &C[by*BLOCK_SIZE_M*N+bx*BLOCK_SIZE_N];
    //计算相关：
    float sum[THREAD_SIZE_M][THREAD_SIZE_N] = {0};
    // x 对应 N 方向（列），y 对应 M 方向（行）
    int comp_m = ty * THREAD_SIZE_M;
    int comp_n = tx * THREAD_SIZE_N;
    
    float a_frag[THREAD_SIZE_M];
    float b_frag[THREAD_SIZE_N];
  // K 方向分块迭代
  for (int kb = 0; kb < (int)K; kb += BLOCK_SIZE_K) {
    // 1) 装载 A 子块到共享内存 As[BM][BK]，采用跨 stride 方式覆盖 0..BM-1
    
    for(int i = 0;i < num_row_strides_a;i++){
        int ldg_index = i;
        // 安全读取4个连续元素
        float4 va = load4(&A[OFFSET(load_smem_a_k + i*load_threeads_a_m, load_smem_a_m, BLOCK_SIZE_K)]);
        ldg_a_reg[ldg_index] = va;
        // 将以上的一行的4个元素，按列排布在共享显存中
        As[OFFSET(load_smem_a_m,     i + load_smem_a_k, BLOCK_SIZE_M)] = va.x;
        As[OFFSET(load_smem_a_m + 1, i + load_smem_a_k, BLOCK_SIZE_M)] = va.y;
        As[OFFSET(load_smem_a_m + 2, i + load_smem_a_k, BLOCK_SIZE_M)] = va.z;
        As[OFFSET(load_smem_a_m + 3, i + load_smem_a_k, BLOCK_SIZE_M)] = va.w;
    }
    // 2) 装载 B 子块到共享内存 Bs[BK][BN]，采用跨 stride 方式覆盖 0..BK-1
    for(int i = 0;i < num_k_strides_b;i++){
        float4 vb = load4(&B[OFFSET(load_smem_b_n + i*load_threads_b_k, load_smem_b_k, N)]);
        int base = OFFSET(load_smem_b_n + i*load_threads_b_k, load_smem_b_k, BLOCK_SIZE_N);
        // 将4个标量安全写入共享内存的连续位置
        Bs[base + 0] = vb.x;
        Bs[base + 1] = vb.y;
        Bs[base + 2] = vb.z;
        Bs[base + 3] = vb.w;
    }

    __syncthreads();
    
    A += BLOCK_SIZE_K;
    B += BLOCK_SIZE_K * N;

    for(int kk = 0;kk < BLOCK_SIZE_K;kk++){
        // 从共享内存按每线程子块装入寄存器（标量方式，避免未对齐）
        #pragma unroll
        for(int i = 0;i < THREAD_SIZE_M; ++i){
            a_frag[i] = As[OFFSET(comp_m + i, kk, BLOCK_SIZE_M)];
        }
        #pragma unroll
        for(int j = 0;j < THREAD_SIZE_N; ++j){
            b_frag[j] = Bs[OFFSET(kk, comp_n + j, BLOCK_SIZE_N)];
        }
        #pragma unroll
        for(int i = 0;i < THREAD_SIZE_M;i++){
            #pragma unroll
            for(int j = 0;j < THREAD_SIZE_N;j++){
                sum[i][j] += a_frag[i] * b_frag[j];
            }
        }
    }
    __syncthreads();
  }
  // 将每线程累加结果写回全局内存（标量方式，避免未对齐）
  for(int i = 0;i < THREAD_SIZE_M;i++){
    for(int j = 0;j < THREAD_SIZE_N;j++){
        C[OFFSET(comp_m+i, comp_n+j, N)] += sum[i][j];
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
    memset(C,0,sizeof(float)*M*N);
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