#include <cuda_runtime.h>
#define OFFSET(row, col, lgd) ((row)*(lgd)+(col))
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
const int WARP_SIZE=32;
//对sharemem再次进行分块，将sharemem分块到register中。
const int THREAD_SIZE_M=8;//每个线程计算的C中元素的高度
const int THREAD_SIZE_N=8;//每个线程计算的C中元素的宽度
const int BLOCK_SIZE_M=128; ///每个线程块需要处理的M维度数据块大小
const int BLOCK_SIZE_N=128; ///每个线程块需要处理的N维度数据块大小
const int BLOCK_SIZE_K=16;  //每个线程块需要A load into sharemen的宽度
//每个线程块的所包含的线程数量
const int THREAD_SIZE_PER_BLCOK_M=BLOCK_SIZE_M/THREAD_SIZE_M;   //16
const int THREAD_SIZE_PER_BLCOK_N=BLOCK_SIZE_N/THREAD_SIZE_N;   //16
#define BK BLOCK_SIZE_K
#define BM BLOCK_SIZE_M
#define BN BLOCK_SIZE_N
#define TM THREAD_SIZE_M
#define TN THREAD_SIZE_N
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


const int THREAD_NUM_PER_BLOCK = THREAD_SIZE_PER_BLCOK_M*THREAD_SIZE_PER_BLCOK_N;

__device__ __forceinline__ float4 load4(const float* p) {
  // 假设 p 至少 16-byte 对齐；否则有些架构会慢或潜在异常
  return *reinterpret_cast<const float4*>(p);
}

__global__ void gemm_v2_warp_tiling(
    int M, int N, int K,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C) {

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    int tid = ty * blockDim.x + tx;
    int warp_id = tid /32;
    int lane_id = tid % 32;

    // 一个 block 一次搬运 64x16 个 a， 8x128 个 b， 分两次搬运恰好共 128x16, 16x128
    // 每 4 个线程负责一行 a(16 个元素），每 32 个线程负责一行 b(128 个元素）
    int load_a_row = tid / 4;               // 0~63
    int load_a_col = (tid % 4) * 4;         // 0,4,8,12...
    int load_b_row = tid / WARP_SIZE;       // 0~8
    int load_b_col = (tid % WARP_SIZE) * 4; // 0,4,8,12,16,20,24,28...


    // warp tiling, 每 4 个 warp 负责 c 的上下两部分 64x128，
    int warp_row = warp_id / 4;      // 0, 1
    int warp_col = warp_id % 4;      // 0, 1, 2, 3
    int t_row_in_warp = lane_id / 4; // 0~7
    int t_col_in_warp = lane_id % 4; // 0~3


    // c out 初始坐标， 每个线程负责 8 行 8 列 tile, 共 256 线程，256*64 = 128*128
    int c_row = warp_row * 64 + t_row_in_warp * 8;
    int c_col = warp_col * 32 + t_col_in_warp * 8;


    __shared__ float As[BLOCK_SIZE_K][BLOCK_SIZE_M];
    __shared__ float Bs[BLOCK_SIZE_K][BLOCK_SIZE_N];

    float reg_a[THREAD_SIZE_M] = {0.0f};
    float reg_b[THREAD_SIZE_N] = {0.0f};

    float sum[THREAD_SIZE_M][THREAD_SIZE_N] = {0.0f};

    A = &A[by*BLOCK_SIZE_M*K];
    B = &B[bx*BLOCK_SIZE_N];
    C = &C[by*BLOCK_SIZE_M*N + bx*BLOCK_SIZE_N];
    // warp tile 下就是这样的写的，不用循环
    for (int bk = 0; bk < K; bk += BK) {

        float4 tmp_a0 =
            load4(&A[OFFSET(load_a_row, bk + load_a_col, K)]);
        int store_a_row = load_a_col;
        int store_a_col = load_a_row;
        As[store_a_row + 0][store_a_col] = tmp_a0.x;
        As[store_a_row + 1][store_a_col] = tmp_a0.y;
        As[store_a_row + 2][store_a_col] = tmp_a0.z;
        As[store_a_row + 3][store_a_col] = tmp_a0.w;

        float4 tmp_a1 =
            load4(&A[OFFSET(load_a_row + 64, bk + load_a_col, K)]);
        As[store_a_row + 0][store_a_col + 64] = tmp_a1.x;
        As[store_a_row + 1][store_a_col + 64] = tmp_a1.y;
        As[store_a_row + 2][store_a_col + 64] = tmp_a1.z;
        As[store_a_row + 3][store_a_col + 64] = tmp_a1.w;

        
        FLOAT4(Bs[load_b_row][load_b_col]) =
            load4(&B[OFFSET(bk + load_b_row, load_b_col, N)]);
        FLOAT4(Bs[load_b_row + 8][load_b_col]) =
            load4(&B[OFFSET(bk + load_b_row + 8, load_b_col, N)]);

        __syncthreads();

        // 8x8 循环计算累加乘积和，k 纬度
#pragma unroll
        for (int i = 0; i < BK; i++) {
#pragma unroll
            FLOAT4(reg_a[0]) = FLOAT4(As[i][c_row]);
            FLOAT4(reg_a[4]) = FLOAT4(As[i][c_row + 4]);

            FLOAT4(reg_b[0]) = FLOAT4(Bs[i][c_col]);
            FLOAT4(reg_b[4]) = FLOAT4(Bs[i][c_col + 4]);

            #pragma unroll
            for (int m_idx = 0; m_idx < TM; ++m_idx) {
                #pragma unroll
                for (int n_idx = 0; n_idx < TN; ++n_idx) {
                    sum[m_idx][n_idx] += reg_a[m_idx] * reg_b[n_idx];
                }
            }
        }
        __syncthreads();
    }

        #pragma unroll
        for (int i = 0; i < TM; ++i) {
            FLOAT4(C[OFFSET(c_row + i, c_col, N)]) = FLOAT4(sum[i][0]);
            FLOAT4(C[OFFSET(c_row + i, c_col + 4, N)]) = FLOAT4(sum[i][4]);
        }

}


void launch_gemm_v2_warp_tiling(
    int M, int N, int K, float* A, float* B, float* C,
    cudaStream_t stream) {
    const dim3 block(THREAD_SIZE_PER_BLCOK_N, THREAD_SIZE_PER_BLCOK_M);
    const dim3 grid((N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N,
                    (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);
    gemm_v2_warp_tiling<<<grid, block, 0, stream>>>(M, N, K, A, B, C);
}
