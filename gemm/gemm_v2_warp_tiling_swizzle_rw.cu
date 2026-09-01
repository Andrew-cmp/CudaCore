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

//奇怪为什么性能反而更差

#define SWIZZLE_A(x, y) ((y) ^ ((x >> 2) << 3))
const int THREAD_NUM_PER_BLOCK = THREAD_SIZE_PER_BLCOK_M*THREAD_SIZE_PER_BLCOK_N;

__device__ __forceinline__ float4 load4(const float* p) {
  // 假设 p 至少 16-byte 对齐；否则有些架构会慢或潜在异常
  return *reinterpret_cast<const float4*>(p);
}

__global__ void gemm_v2_warp_tiling_swizzle_rw(
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






    // Thread row offset within warp is still 0 or 8
    int t_row_in_warp = (lane_id / 16) * 8;

    // Each warp covers 16 rows; every 16 threads handle 8 rows × 128 cols, split into two float4 passes.
    // E.g., T0 reads/writes cols 0~3 and 64~67 — every 8 consecutive threads produce 32 contiguous
    // floats (128 bytes) on write-back, achieving perfect transaction coalescing.
    int c_row = warp_id * 16 + t_row_in_warp;
    int c_col_base = (lane_id % 16) * 4;
    int c_col_0 = c_col_base;      // 0~3
    int c_col_1 = c_col_base + 64; // 64~67


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
            
        As[load_a_col + 0][SWIZZLE_A(load_a_col + 0, load_a_row)] = tmp_a0.x;
        As[load_a_col + 1][SWIZZLE_A(load_a_col + 1, load_a_row)] = tmp_a0.y;
        As[load_a_col + 2][SWIZZLE_A(load_a_col + 2, load_a_row)] = tmp_a0.z;
        As[load_a_col + 3][SWIZZLE_A(load_a_col + 3, load_a_row)] = tmp_a0.w;
        float4 tmp_a1 =
            load4(&A[OFFSET(load_a_row + 64, bk + load_a_col, K)]);
        As[load_a_col + 0][SWIZZLE_A(load_a_col + 0, load_a_row + 64)] = tmp_a1.x;
        As[load_a_col + 1][SWIZZLE_A(load_a_col + 1, load_a_row + 64)] = tmp_a1.y;
        As[load_a_col + 2][SWIZZLE_A(load_a_col + 2, load_a_row + 64)] = tmp_a1.z;
        As[load_a_col + 3][SWIZZLE_A(load_a_col + 3, load_a_row + 64)] = tmp_a1.w;
        FLOAT4(Bs[load_b_row][load_b_col]) =
            load4(&B[OFFSET(bk + load_b_row, load_b_col, N)]);
        FLOAT4(Bs[load_b_row + 8][load_b_col]) =
            load4(&B[OFFSET(bk + load_b_row + 8, load_b_col, N)]);

        __syncthreads();

        // 8x8 循环计算累加乘积和，k 纬度
#pragma unroll
        for (int i = 0; i < BK; i++) {
#pragma unroll
            FLOAT4(reg_a[0]) = FLOAT4(As[i][SWIZZLE_A(i, c_row)]);
            FLOAT4(reg_a[4]) = FLOAT4(As[i][SWIZZLE_A(i, c_row + 4)]);

            FLOAT4(reg_b[0]) = FLOAT4(Bs[i][c_col_0]);
            FLOAT4(reg_b[4]) = FLOAT4(Bs[i][c_col_1]);
            

            //就是他算他算，你算你的，拿来数据就能算，下边这部分不会动，算的哪部分你也不用管
            //存的时候自然会存到正确的地方去

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
            FLOAT4(C[OFFSET(c_row + i, c_col_0, N)]) = FLOAT4(sum[i][0]);
            FLOAT4(C[OFFSET(c_row + i, c_col_1, N)]) = FLOAT4(sum[i][4]);
        }

}

void launch_gemm_v2_warp_tiling_swizzle_rw(
    int M, int N, int K, float* A, float* B, float* C,
    cudaStream_t stream) {
    const dim3 block(THREAD_SIZE_PER_BLCOK_N, THREAD_SIZE_PER_BLCOK_M);
    const dim3 grid((N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N,
                    (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);
    gemm_v2_warp_tiling_swizzle_rw<<<grid, block, 0, stream>>>(M, N, K, A, B, C);
}
