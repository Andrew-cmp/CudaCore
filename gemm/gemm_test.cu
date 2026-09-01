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

__global__ void gemm_test(int M, int N, int K, float *A, float *B, float *C) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tid = threadIdx.x; // 0~255; 8 warps
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // Load mapping
    int load_a_row = tid / 4;               // 0~63
    int load_a_col = (tid % 4) * 4;         // 0,4,8,12...
    int load_b_row = tid / WARP_SIZE;       // 0~8
    int load_b_col = (tid % WARP_SIZE) * 4; // 0,4,8,12,16,20,24,28...

    // C compute/read/write mapping (same as above)
    int t_row_in_warp = (lane_id / 16) * 8;
    int c_row = warp_id * 16 + t_row_in_warp;
    int c_col_base = (lane_id % 16) * 4;
    int c_col_0 = c_col_base; // 0~3
    // int c_col_1 = c_col_base + 64; // 64~67

    // double buffer
    __shared__ float AS[2][BK][BM];
    __shared__ float Bs[2][BK][BN];

    float sum[TM][TN] = {0.f};

    // Flat pointers into global memory for easy pipeline advancement
    float *a_ptr = A + (by * BM + load_a_row) * K + load_a_col;
    // float *a_ptr_64 = A + (by * BM + load_a_row + 64) * K + load_a_col;
    float *b_ptr = B + load_b_row * N + bx * BN + load_b_col;
    // float *b_ptr_8 = B + (load_b_row + 8) * N + bx * BN + load_b_col;

    // Prefetch first tile (costs 16 extra regs; commented-out ptrs above keep total under 128 to preserve occupancy)
    float4 tmp_a0 = FLOAT4(a_ptr[0]);
    float4 tmp_a1 = FLOAT4(a_ptr[64 * K]);
    float4 tmp_b0 = FLOAT4(b_ptr[0]);
    float4 tmp_b1 = FLOAT4(b_ptr[8 * N]);

    AS[0][load_a_col + 0][SWIZZLE_A(load_a_col + 0, load_a_row)] = tmp_a0.x;
    AS[0][load_a_col + 1][SWIZZLE_A(load_a_col + 1, load_a_row)] = tmp_a0.y;
    AS[0][load_a_col + 2][SWIZZLE_A(load_a_col + 2, load_a_row)] = tmp_a0.z;
    AS[0][load_a_col + 3][SWIZZLE_A(load_a_col + 3, load_a_row)] = tmp_a0.w;

    AS[0][load_a_col + 0][SWIZZLE_A(load_a_col + 0, load_a_row + 64)] = tmp_a1.x;
    AS[0][load_a_col + 1][SWIZZLE_A(load_a_col + 1, load_a_row + 64)] = tmp_a1.y;
    AS[0][load_a_col + 2][SWIZZLE_A(load_a_col + 2, load_a_row + 64)] = tmp_a1.z;
    AS[0][load_a_col + 3][SWIZZLE_A(load_a_col + 3, load_a_row + 64)] = tmp_a1.w;

    FLOAT4(Bs[0][load_b_row][load_b_col]) = tmp_b0;
    FLOAT4(Bs[0][load_b_row + 8][load_b_col]) = tmp_b1;

    __syncthreads();

    // Double buffer indices
    int write_idx = 1;
    int read_idx = 0;
    // Main loop
    for (int bk = BK; bk < K; bk += BK) {
        // Advance pointers along K dimension
        a_ptr += BK;
        b_ptr += BK * N;

        // Prefetch next tile — after issuing LDG, we immediately compute on the current tile
        tmp_a0 = FLOAT4(a_ptr[0]);
        tmp_a1 = FLOAT4(a_ptr[64 * K]);
        tmp_b0 = FLOAT4(b_ptr[0]);
        tmp_b1 = FLOAT4(b_ptr[8 * N]);

        // Compute logic identical to non-pipelined version
#pragma unroll
        for (int i = 0; i < BK; i++) {
            float reg_a[TM], reg_b[TN];

            FLOAT4(reg_a[0]) = FLOAT4(AS[read_idx][i][SWIZZLE_A(i, c_row)]);
            FLOAT4(reg_a[4]) = FLOAT4(AS[read_idx][i][SWIZZLE_A(i, c_row + 4)]);

            FLOAT4(reg_b[0]) = FLOAT4(Bs[read_idx][i][c_col_0]);
            FLOAT4(reg_b[4]) = FLOAT4(Bs[read_idx][i][c_col_0 + 64]);

#pragma unroll
            for (int m_idx = 0; m_idx < TM; ++m_idx) {
#pragma unroll
                for (int n_idx = 0; n_idx < TN; ++n_idx) {
                    sum[m_idx][n_idx] += reg_a[m_idx] * reg_b[n_idx];
                }
            }
        }

        // Computation done — store prefetched registers into SMEM write buffer
        AS[write_idx][load_a_col + 0][SWIZZLE_A(load_a_col + 0, load_a_row)] = tmp_a0.x;
        AS[write_idx][load_a_col + 1][SWIZZLE_A(load_a_col + 1, load_a_row)] = tmp_a0.y;
        AS[write_idx][load_a_col + 2][SWIZZLE_A(load_a_col + 2, load_a_row)] = tmp_a0.z;
        AS[write_idx][load_a_col + 3][SWIZZLE_A(load_a_col + 3, load_a_row)] = tmp_a0.w;

        AS[write_idx][load_a_col + 0][SWIZZLE_A(load_a_col + 0, load_a_row + 64)] = tmp_a1.x;
        AS[write_idx][load_a_col + 1][SWIZZLE_A(load_a_col + 1, load_a_row + 64)] = tmp_a1.y;
        AS[write_idx][load_a_col + 2][SWIZZLE_A(load_a_col + 2, load_a_row + 64)] = tmp_a1.z;
        AS[write_idx][load_a_col + 3][SWIZZLE_A(load_a_col + 3, load_a_row + 64)] = tmp_a1.w;

        FLOAT4(Bs[write_idx][load_b_row][load_b_col]) = tmp_b0;
        FLOAT4(Bs[write_idx][load_b_row + 8][load_b_col]) = tmp_b1;

        __syncthreads();
        write_idx ^= 1;
        read_idx ^= 1;
    }
    // Process the last prefetched tile
#pragma unroll
    for (int i = 0; i < BK; i++) {
        float reg_a[TM], reg_b[TN];

        FLOAT4(reg_a[0]) = FLOAT4(AS[read_idx][i][SWIZZLE_A(i, c_row)]);
        FLOAT4(reg_a[4]) = FLOAT4(AS[read_idx][i][SWIZZLE_A(i, c_row + 4)]);

        FLOAT4(reg_b[0]) = FLOAT4(Bs[read_idx][i][c_col_0]);
        FLOAT4(reg_b[4]) = FLOAT4(Bs[read_idx][i][c_col_0 + 64]);

#pragma unroll
        for (int m_idx = 0; m_idx < TM; ++m_idx) {
#pragma unroll
            for (int n_idx = 0; n_idx < TN; ++n_idx) {
                sum[m_idx][n_idx] += reg_a[m_idx] * reg_b[n_idx];
            }
        }
    }
    // Pipeline done — write back C
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        FLOAT4(C[(by * BM + c_row + i) * N + bx * BN + c_col_0]) = FLOAT4(sum[i][0]);
        FLOAT4(C[(by * BM + c_row + i) * N + bx * BN + c_col_0 + 64]) = FLOAT4(sum[i][4]);
    }
}

void launch_gemm_test(
    int M, int N, int K, float* A, float* B, float* C,
    cudaStream_t stream) {
    const dim3 block(THREAD_SIZE_PER_BLCOK_N, THREAD_SIZE_PER_BLCOK_M);
    const dim3 grid((N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N,
                    (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);
    gemm_test<<<grid, block, 0, stream>>>(M, N, K, A, B, C);
}
