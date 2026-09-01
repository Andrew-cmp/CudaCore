#include <cuda_runtime.h>
#define OFFSET(row, col, lgd) ((row)*(lgd)+(col))
//对sharemem再次进行分块，将sharemem分块到register中。
const int THREAD_SIZE_M=8;//每个线程计算的C中元素的高度
const int THREAD_SIZE_N=8;//每个线程计算的C中元素的宽度
const int BLOCK_SIZE_M=128; ///每个线程块需要处理的M维度数据块大小
const int BLOCK_SIZE_N=128; ///每个线程块需要处理的N维度数据块大小
const int BLOCK_SIZE_K=16;  //每个线程块需要A load into sharemen的宽度
//每个线程块的所包含的线程数量
const int THREAD_SIZE_PER_BLCOK_M=BLOCK_SIZE_M/THREAD_SIZE_M;   //16
const int THREAD_SIZE_PER_BLCOK_N=BLOCK_SIZE_N/THREAD_SIZE_N;   //16
const int THREAD_NUM_PER_BLOCK = THREAD_SIZE_PER_BLCOK_M*THREAD_SIZE_PER_BLCOK_N;
// 4096 * 4096 * 4096下，grid size为(4096/128, 4096/128) = (32, 32)，block size为(16, 16)，
//总线程数为32*32*16*16=262144
//block 总数为32*32=1024，block内线程数为16*16=256，warp数为256/32=8，

//单缓冲下申请的sharemem大小为BLOCK_SIZE_K*BLOCK_SIZE_M + BLOCK_SIZE_K*BLOCK_SIZE_N = 16*128 + 16*128 = 4096，
//单缓冲下申请的sharemem大小为4096 * 4 = 16384 bytes = 16KB
//那么每个SM最多能够容纳的block数为64KB / 16KB = 4个block，可以了

//双缓冲下申请的sharemem大小为2 * (BLOCK_SIZE_K*BLOCK_SIZE_M + BLOCK_SIZE_K*BLOCK_SIZE_N) = 2 * (16*128 + 16*128) = 8192，
//双缓冲下申请的sharemem大小为8192 * 4 = 32768 bytes = 32KB
//那么每个SM最多能够容纳的block数为64KB / 32KB = 2个block，也还可以

//单缓冲下申请的register大小为THREAD_SIZE_M*THREAD_SIZE_N*4 = 16*8*4 = 512 bytes
//那么每个SM最多能够容纳的block数为65536 / 128 = 512个block，可以了

//A6000 峰值计算能力为 38.7 TFLOPS，内存带宽为768 GB/s，计算强度为38.7 * 1024 / 768 = 51.6 FLOPS/byte
//tile后的计算强度为
//2*BLOCK_SIZE_M * BLOCK_SIZE_N * BLOCK_SIZE_K / (4*BLOCK_SIZE_M * BLOCK_SIZE_K + 4*BLOCK_SIZE_K * BLOCK_SIZE_N)
//=2* 128 * 128 * 16 / (4*128 * 16 + 4*16 * 128) = 32 FLOPS/byte
// 32 FLOPS/byte < 51.6 FLOPS/byte，说明该算法是内存带宽受限的。
__device__ __forceinline__ float4 load4(const float* p) {
  // 假设 p 至少 16-byte 对齐；否则有些架构会慢或潜在异常
  return *reinterpret_cast<const float4*>(p);
}
__global__ void gemm_v2(
    int M, int N, int K,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C) {

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // 线性线程索引，
    const int tid = ty * blockDim.x + tx;
    
    __shared__ float As[BLOCK_SIZE_K*BLOCK_SIZE_M];
    __shared__ float Bs[BLOCK_SIZE_K*BLOCK_SIZE_N];

    float reg_a[THREAD_SIZE_M] = {0.0f};
    float reg_b[THREAD_SIZE_N] = {0.0f};

    float reg_c[THREAD_SIZE_M][THREAD_SIZE_N] = {0.0f};
    
    A = &A[by*BLOCK_SIZE_M*K];
    B = &B[bx*BLOCK_SIZE_N];
    for(int kk = 0;kk < K;kk += BLOCK_SIZE_K){
        const int iter_couts = BLOCK_SIZE_M / (THREAD_NUM_PER_BLOCK / (BLOCK_SIZE_K / 4));
        const float *A_ptr = &A[kk];
    #pragma unroll
        for(int i = 0;i < iter_couts;i++){

            int load_smem_a_m = tid / (BLOCK_SIZE_K / 4)+ i*(THREAD_NUM_PER_BLOCK / (BLOCK_SIZE_K / 4));
            int load_smem_a_k = tid % (BLOCK_SIZE_K / 4) * 4;
            float4 a = load4(&A_ptr[OFFSET(load_smem_a_m, load_smem_a_k, K)]);
            As[OFFSET(load_smem_a_k + 0, load_smem_a_m, BLOCK_SIZE_M)] = a.x;
            As[OFFSET(load_smem_a_k + 1, load_smem_a_m, BLOCK_SIZE_M)] = a.y;
            As[OFFSET(load_smem_a_k + 2, load_smem_a_m, BLOCK_SIZE_M)] = a.z;
            As[OFFSET(load_smem_a_k + 3, load_smem_a_m, BLOCK_SIZE_M)] = a.w;
        }
        const int iter_couts_b = BLOCK_SIZE_K / (THREAD_NUM_PER_BLOCK / (BLOCK_SIZE_N / 4));
        const float *B_ptr = &B[kk*N];
    #pragma unroll
        for(int i = 0;i < iter_couts_b;i++){
            int load_smem_b_k = tid / (BLOCK_SIZE_N / 4) + i*(THREAD_NUM_PER_BLOCK / (BLOCK_SIZE_N / 4));
            int load_smem_b_n = tid % (BLOCK_SIZE_N / 4) * 4;
            float4 b = load4(&B_ptr[OFFSET(load_smem_b_k, load_smem_b_n, N)]);
            Bs[OFFSET(load_smem_b_k, load_smem_b_n + 0, BLOCK_SIZE_N)] = b.x;
            Bs[OFFSET(load_smem_b_k, load_smem_b_n + 1, BLOCK_SIZE_N)] = b.y;
            Bs[OFFSET(load_smem_b_k, load_smem_b_n + 2, BLOCK_SIZE_N)] = b.z;
            Bs[OFFSET(load_smem_b_k, load_smem_b_n + 3, BLOCK_SIZE_N)] = b.w;
        }
        __syncthreads();
        for(int bk = 0;bk < BLOCK_SIZE_K;bk++){
            
    #pragma unroll
            for(int i = 0;i < THREAD_SIZE_M;i++){
                reg_a[i] = As[OFFSET(bk, ty*THREAD_SIZE_M+i, BLOCK_SIZE_M)];
            }
            
    #pragma unroll
            for(int j = 0;j < THREAD_SIZE_N;j++){
                reg_b[j] = Bs[OFFSET(bk, tx*THREAD_SIZE_N+j, BLOCK_SIZE_N)];
            }
    #pragma unroll
            for(int i = 0;i < THREAD_SIZE_M;i++){
    #pragma unroll
                for(int j = 0;j < THREAD_SIZE_N;j++){
                    reg_c[i][j] += reg_a[i]*reg_b[j];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for(int i = 0;i < THREAD_SIZE_M;i++){
        
    #pragma unroll
        for(int j = 0;j < THREAD_SIZE_N;j++){
            int global_c_m = by*BLOCK_SIZE_M+i+ty*THREAD_SIZE_M;
            int global_c_n = bx*BLOCK_SIZE_N+j+tx*THREAD_SIZE_N;
            C[OFFSET(global_c_m, global_c_n, N)] = reg_c[i][j];
        }
    }
}

void launch_gemm_v2(
    int M, int N, int K, float* A, float* B, float* C,
    cudaStream_t stream) {
    const dim3 block(THREAD_SIZE_PER_BLCOK_N, THREAD_SIZE_PER_BLCOK_M);
    const dim3 grid((N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N,
                    (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);
    gemm_v2<<<grid, block, 0, stream>>>(M, N, K, A, B, C);
}
