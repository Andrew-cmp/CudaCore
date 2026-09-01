// Migrated from ../gemm. Kernel computation and memory-access logic are preserved.
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/types.h>

#include <limits>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define SWIZZLE_A(x, y) ((y) ^ (((x) >> 2) << 3))
#define torch_pybinding_func(function) module.def(#function, &function, #function)

constexpr int WARP_SIZE = 32;

// Implemented in sgemm_mma.cu and linked into the same PyTorch extension.
void sgemm_tf32_bt(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void sgemm_tf32_bt_swizzle(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void sgemm_tf32_bt_swizzle_dbf(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void sgemm_tf32_swizzle_bcf(torch::Tensor a, torch::Tensor b, torch::Tensor c);
void sgemm_tf32_swizzle_bcf_dbf(torch::Tensor a, torch::Tensor b, torch::Tensor c);

__device__ __forceinline__ float4 load4(const float* pointer) {
    return *reinterpret_cast<const float4*>(pointer);
}

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void gemm_v0_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0;
    for(int i = 0; i < K; i++){
        sum += A[row*K+i] * B[i*N+col];
    }
    C[row*N+col] = sum;
}

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void gemm_v1_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float sum  = 0;
    __shared__ float A_shared[BM][BK];
    __shared__ float B_shared[BK][BN];

    const int numTiles = (K + BK - 1) / BK;
    for(int num_k = 0;num_k < numTiles;num_k++){
        if(num_k*BK+tx <K && y < M)
            A_shared[ty][tx] = A[y*K+num_k*BK+tx];
        else
            A_shared[ty][tx] = 0;
        if(num_k*BK+ty < K && x<N)
            B_shared[ty][tx] = B[(num_k*BK+ty)*N + x];
        else
            B_shared[ty][tx] = 0;
        __syncthreads();
        for(int i = 0;i < BK;i++){
            sum += A_shared[ty][i]*B_shared[i][tx];
        }
        __syncthreads();
    }
    if(y < M && x < N)
        C[y*N+x] = sum;

}

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void gemm_v2_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K) {
    constexpr int NUM_THREADS = (BM / TM) * (BN / TN);

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    // 线性线程索引，
    const int tid = ty * blockDim.x + tx;
    
    __shared__ float As[BK*BM];
    __shared__ float Bs[BK*BN];

    float reg_a[TM] = {0.0f};
    float reg_b[TN] = {0.0f};

    float reg_c[TM][TN] = {0.0f};
    
    A = &A[by*BM*K];
    B = &B[bx*BN];
    for(int kk = 0;kk < K;kk += BK){
        const int iter_couts = BM / (NUM_THREADS / (BK / 4));
        const float *A_ptr = &A[kk];
    #pragma unroll
        for(int i = 0;i < iter_couts;i++){

            int load_smem_a_m = tid / (BK / 4)+ i*(NUM_THREADS / (BK / 4));
            int load_smem_a_k = tid % (BK / 4) * 4;
            float4 a = load4(&A_ptr[OFFSET(load_smem_a_m, load_smem_a_k, K)]);
            As[OFFSET(load_smem_a_k + 0, load_smem_a_m, BM)] = a.x;
            As[OFFSET(load_smem_a_k + 1, load_smem_a_m, BM)] = a.y;
            As[OFFSET(load_smem_a_k + 2, load_smem_a_m, BM)] = a.z;
            As[OFFSET(load_smem_a_k + 3, load_smem_a_m, BM)] = a.w;
        }
        const int iter_couts_b = BK / (NUM_THREADS / (BN / 4));
        const float *B_ptr = &B[kk*N];
    #pragma unroll
        for(int i = 0;i < iter_couts_b;i++){
            int load_smem_b_k = tid / (BN / 4) + i*(NUM_THREADS / (BN / 4));
            int load_smem_b_n = tid % (BN / 4) * 4;
            float4 b = load4(&B_ptr[OFFSET(load_smem_b_k, load_smem_b_n, N)]);
            Bs[OFFSET(load_smem_b_k, load_smem_b_n + 0, BN)] = b.x;
            Bs[OFFSET(load_smem_b_k, load_smem_b_n + 1, BN)] = b.y;
            Bs[OFFSET(load_smem_b_k, load_smem_b_n + 2, BN)] = b.z;
            Bs[OFFSET(load_smem_b_k, load_smem_b_n + 3, BN)] = b.w;
        }
        __syncthreads();
        for(int bk = 0;bk < BK;bk++){
            
    #pragma unroll
            for(int i = 0;i < TM;i++){
                reg_a[i] = As[OFFSET(bk, ty*TM+i, BM)];
            }
            
    #pragma unroll
            for(int j = 0;j < TN;j++){
                reg_b[j] = Bs[OFFSET(bk, tx*TN+j, BN)];
            }
    #pragma unroll
            for(int i = 0;i < TM;i++){
    #pragma unroll
                for(int j = 0;j < TN;j++){
                    reg_c[i][j] += reg_a[i]*reg_b[j];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for(int i = 0;i < TM;i++){
        
    #pragma unroll
        for(int j = 0;j < TN;j++){
            int global_c_m = by*BM+i+ty*TM;
            int global_c_n = bx*BN+j+tx*TN;
            C[OFFSET(global_c_m, global_c_n, N)] = reg_c[i][j];
        }
    }
}

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void gemm_v2_warp_tiling_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K) {

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


    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];

    float reg_a[TM] = {0.0f};
    float reg_b[TN] = {0.0f};

    float sum[TM][TN] = {0.0f};

    A = &A[by*BM*K];
    B = &B[bx*BN];
    C = &C[by*BM*N + bx*BN];
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

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void gemm_v2_warp_tiling_swizzle_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K) {

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


    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];

    float reg_a[TM] = {0.0f};
    float reg_b[TN] = {0.0f};

    float sum[TM][TN] = {0.0f};

    A = &A[by*BM*K];
    B = &B[bx*BN];
    C = &C[by*BM*N + bx*BN];
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

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void gemm_v2_warp_tiling_swizzle_rw_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K) {

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


    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];

    float reg_a[TM] = {0.0f};
    float reg_b[TN] = {0.0f};

    float sum[TM][TN] = {0.0f};

    A = &A[by*BM*K];
    B = &B[bx*BN];
    C = &C[by*BM*N + bx*BN];
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

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void gemm_v3_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K) {
    constexpr int NUM_THREADS = (BM / TM) * (BN / TN);
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    // 线性线程索引，
    const int tid = ty * blockDim.x + tx;     // 0..(blockDim.x*blockDim.y-1)
    

    __shared__ float As[BK*BM];
    __shared__ float Bs[BK*BN];

    //线程块计算和传输是完全没关系的，传输的线程块甚至要“reshape”其线程摆布，不要用原本计算的思维来进行传输。
    //我们用thread_num来考虑搬运。

    //对于A来说，在K轴上安排load_threads_a_k个线程，那么在M轴上则可以排load_threeads_a_m行线程。
    const int load_threads_a_k = BK / 4;
    const int load_threeads_a_m = NUM_THREADS / (load_threads_a_k);
    //在M轴上，如果BM>load_threeads_a_m，那么需要迭代num_row_strides_a次。
    const int num_row_strides_a = BM / load_threeads_a_m;

    //线程索引，在smem a上的索引。
    int load_smem_a_m = tid / load_threads_a_k;
    int load_smem_a_k = tid % (load_threads_a_k) *4;

    const int load_threads_b_n = BN / 4;
    //对于B来说，在N轴上安排BN个线程，那么在K轴上可以安排load_threads_b_k行线程。
    const int load_threads_b_k = NUM_THREADS / load_threads_b_n;
    //在K轴上，如果BK>load_threads_b_k，那么需要迭代num_k_strides_b次。
    const int num_k_strides_b = BK / load_threads_b_k;

    //线程索引，在smem b上的索引。
    int load_smem_b_k = tid / load_threads_b_n;
    int load_smem_b_n = tid % (load_threads_b_n)* 4;

    //这个thread block所独属于的globalmem 起始地址的计算
    A = &A[by*BM*K];
    B = &B[bx*BN];
    C = &C[by*BM*N+bx*BN];
    //计算相关：
    float sum[TM][TN] = {0};
    // x 对应 N 方向（列），y 对应 M 方向（行）
    int comp_m = ty * TM;
    int comp_n = tx * TN;
    
    float a_frag[TM];
    float b_frag[TN];
  // K 方向分块迭代
  for (int kb = 0; kb < (int)K; kb += BK) {
    // 1) 装载 A 子块到共享内存 As[BM][BK]，采用跨 stride 方式覆盖 0..BM-1
    
    for(int i = 0;i < num_row_strides_a;i++){
        int row = load_smem_a_m + i * load_threeads_a_m;   // 0..63
        int col = load_smem_a_k;                           // 0 or 4 (within BK)
        // 安全读取4个连续元素
        float4 va = load4(&A[row * K + col]);              // A is already block-offset, K is global ld
        // 将以上的一行的4个元素，按列排布在共享显存中
        As[OFFSET(col    ,row , BM)] = va.x;
        As[OFFSET(col + 1,row , BM)] = va.y;
        As[OFFSET(col + 2,row , BM)] = va.z;
        As[OFFSET(col + 3,row , BM)] = va.w;
    }
    // 2) 装载 B 子块到共享内存 Bs[BK][BN]，采用跨 stride 方式覆盖 0..BK-1
    for(int i = 0;i < num_k_strides_b;i++){
        // k in [0, BK), n in [0, BN)
        int k = load_smem_b_k + i * load_threads_b_k;  // 0..7
        int n = load_smem_b_n;                         // 0,4,8,...,60

        float4 vb = load4(&B[k * N + n]);              // B is already block-offset, N is global ld

        // Bs is BK x BN, ld = BN
        int base = k * BN + n;
        Bs[base + 0] = vb.x;
        Bs[base + 1] = vb.y;
        Bs[base + 2] = vb.z;
        Bs[base + 3] = vb.w;

    }

    __syncthreads();
    
    A += BK;
    B += BK * N;

    #pragma unroll
    for(int kk = 0;kk < BK;kk++){
        // 从共享内存按每线程子块装入寄存器（标量方式，避免未对齐）
        #pragma unroll
        for(int i = 0;i < TM; ++i){
              a_frag[i] = As[OFFSET(kk, comp_m + i, BM)];

        }
        #pragma unroll
        for(int j = 0;j < TN; ++j){
            b_frag[j] = Bs[OFFSET(kk, comp_n + j, BN)];
        }
        #pragma unroll
        for(int i = 0;i < TM;i++){
            #pragma unroll
            for(int j = 0;j < TN;j++){
                sum[i][j] += a_frag[i] * b_frag[j];
            }
        }
    }
    __syncthreads();
  }
  // launcher 的语义固定为 C=A*B；每4个连续结果合并成一次float4写回。
  #pragma unroll
  for (int i = 0; i < TM; ++i) {
    #pragma unroll
    for (int j = 0; j < TN; j += 4) {
      FLOAT4(C[OFFSET(comp_m + i, comp_n + j, N)]) =
          make_float4(sum[i][j + 0], sum[i][j + 1],
                      sum[i][j + 2], sum[i][j + 3]);
    }
  }
}

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void gemm_v4_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K) {
    constexpr int NUM_THREADS = (BM / TM) * (BN / TN);
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    // 线性线程索引，
    const int tid = ty * blockDim.x + tx;     // 0..(blockDim.x*blockDim.y-1)

    // 在global mem读取A时，为防止后续的bank conflict 将其倒置，As形状为（BK，BM）
    __shared__ float As[2][BK*BM];
    __shared__ float Bs[2][BK*BN];

    //实际上这个值化简后是4*NUM_THREADS，但有时候可能I不是整除值，因此这里在算一遍
    //表示每个iter中需要搬运的float4的单元数量，也即i_stride。
    const int I_A = BM*BK/(4*NUM_THREADS);
    const int I_A_stride = (BM*BK/(4*I_A));
    A = &A[by*BM*K];
    C = &C[by*BM*N+bx*BN];
    for (int i = 0; i < I_A; i++) {
        int q = i*I_A_stride + tid ;
        int row = q / (BK/4);
        int col = q % (BK/4) * 4;

        float4 a = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (by * BM + row < M && col < K) {
            a = load4(&A[OFFSET(row, col, K)]);
        }
        As[0][OFFSET(col + 0, row, BM)] = a.x;
        As[0][OFFSET(col + 1, row, BM)] = a.y;
        As[0][OFFSET(col + 2, row, BM)] = a.z;
        As[0][OFFSET(col + 3, row, BM)] = a.w;
    }

    const int I_B = BK*BN/(4*NUM_THREADS);
    const int I_B_stride = (BK*BN/(4*I_B));
    B = &B[bx*BN];
    for(int i = 0;i < I_B;i++){
        int q = i*I_B_stride + tid;
        int row = q / (BN/4);
        int col = q % (BN/4) * 4;
        float4 b = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (row < K && bx * BN + col < N) {
            b = load4(&B[OFFSET(row, col, N)]);
        }
        Bs[0][OFFSET(row, col + 0, BN)] = b.x;
        Bs[0][OFFSET(row, col + 1, BN)] = b.y;
        Bs[0][OFFSET(row, col + 2, BN)] = b.z;
        Bs[0][OFFSET(row, col + 3, BN)] = b.w;
    }
    float reg_a[2][TM];
    float reg_b[2][TN];

    float accumalator[TM][TN] = {0.};

    
    __syncthreads();

    // x 对应 N 方向（列），y 对应 M 方向（行）
    int comp_m = ty * TM;
    int comp_n = tx * TN;
    #pragma unroll
    for (int m = 0; m < TM; m++) {
        reg_a[0][m] = As[0][OFFSET(0, comp_m + m, BM)];
    }
    #pragma unroll
    for (int n = 0; n < TN; n++) {
        reg_b[0][n] = Bs[0][OFFSET(0, comp_n + n, BN)];
    }

    int write_index = 1;
    int load_index = 0;
    float ldg_a_reg[4 * BK * BM / NUM_THREADS / 4] = {0.};
    float ldg_b_reg[4 * BK * BN / NUM_THREADS / 4] = {0.};
    const float *A_ptr = A;
    const float *B_ptr = B;

    for (int bk = 0; bk < K; bk += BK) {
        const bool has_next_block = bk + BK < K;

        // 提前把下一个K块从global memory搬到register。
        if (has_next_block) {
            A_ptr = A + bk + BK;
            B_ptr = B + (bk + BK) * N;
            #pragma unroll
            for (int i = 0; i < I_A; i++) {
                int q = i * I_A_stride + tid;
                int row = q / (BK / 4);
                int col = q % (BK / 4) * 4;
                int ldg_index = i * 4;

                float4 a = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (by * BM + row < M &&
                    bk + BK + col < K) {
                    a = load4(&A_ptr[OFFSET(row, col, K)]);
                }
                ldg_a_reg[ldg_index + 0] = a.x;
                ldg_a_reg[ldg_index + 1] = a.y;
                ldg_a_reg[ldg_index + 2] = a.z;
                ldg_a_reg[ldg_index + 3] = a.w;
            }

            #pragma unroll
            for (int i = 0; i < I_B; i++) {
                int q = i * I_B_stride + tid;
                int row = q / (BN / 4);
                int col = q % (BN / 4) * 4;
                int ldg_index = i * 4;

                float4 b = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (bk + BK + row < K &&
                    bx * BN + col < N) {
                    b = load4(&B_ptr[OFFSET(row, col, N)]);
                }
                ldg_b_reg[ldg_index + 0] = b.x;
                ldg_b_reg[ldg_index + 1] = b.y;
                ldg_b_reg[ldg_index + 2] = b.z;
                ldg_b_reg[ldg_index + 3] = b.w;
            }
        }

        // 计算当前shared-memory块。
        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            if (kk < BK - 1) {
                #pragma unroll
                for (int m = 0; m < TM; m++) {
                    reg_a[(kk + 1) % 2][m] =
                        As[load_index][OFFSET(kk + 1, comp_m + m, BM)];
                }
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    reg_b[(kk + 1) % 2][n] =
                        Bs[load_index][OFFSET(kk + 1, comp_n + n, BN)];
                }
            }

            #pragma unroll
            for (int m = 0; m < TM; m++) {
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    accumalator[m][n] += reg_a[kk % 2][m] * reg_b[kk % 2][n];
                }
            }
        }

        if (has_next_block) {
            // 使用与global load相同的q、row、col下标写入另一个shared-memory块。
            #pragma unroll
            for (int i = 0; i < I_A; i++) {
                int q = i * I_A_stride + tid;
                int row = q / (BK / 4);
                int col = q % (BK / 4) * 4;
                int ldg_index = i * 4;

                As[write_index][OFFSET(col + 0, row, BM)] = ldg_a_reg[ldg_index + 0];
                As[write_index][OFFSET(col + 1, row, BM)] = ldg_a_reg[ldg_index + 1];
                As[write_index][OFFSET(col + 2, row, BM)] = ldg_a_reg[ldg_index + 2];
                As[write_index][OFFSET(col + 3, row, BM)] = ldg_a_reg[ldg_index + 3];
            }

            #pragma unroll
            for (int i = 0; i < I_B; i++) {
                int q = i * I_B_stride + tid;
                int row = q / (BN / 4);
                int col = q % (BN / 4) * 4;
                int ldg_index = i * 4;

                Bs[write_index][OFFSET(row, col + 0, BN)] = ldg_b_reg[ldg_index + 0];
                Bs[write_index][OFFSET(row, col + 1, BN)] = ldg_b_reg[ldg_index + 1];
                Bs[write_index][OFFSET(row, col + 2, BN)] = ldg_b_reg[ldg_index + 2];
                Bs[write_index][OFFSET(row, col + 3, BN)] = ldg_b_reg[ldg_index + 3];
            }

            __syncthreads();

            load_index = write_index;
            write_index ^= 1;

            #pragma unroll
            for (int m = 0; m < TM; m++) {
                reg_a[0][m] =
                    As[load_index][OFFSET(0, comp_m + m, BM)];
            }
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                reg_b[0][n] =
                    Bs[load_index][OFFSET(0, comp_n + n, BN)];
            }

        }
    }

    #pragma unroll
    for (int m = 0; m < TM; m++) {
        #pragma unroll
        for (int n = 0; n < TN; n += 4) {
            if (by * BM + comp_m + m < M &&
                bx * BN + comp_n + n < N) {
                float4 c = load4(&C[OFFSET(comp_m + m, comp_n + n, N)]);
                c.x = alpha * accumalator[m][n + 0] + beta * c.x;
                c.y = alpha * accumalator[m][n + 1] + beta * c.y;
                c.z = alpha * accumalator[m][n + 2] + beta * c.z;
                c.w = alpha * accumalator[m][n + 3] + beta * c.w;
                FLOAT4(C[OFFSET(comp_m + m, comp_n + n, N)]) = c;
            }
        }
    }

}

template <bool kAligned, const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void gemm_v5_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K) {
    constexpr int THREADS_M = BM / TM;
    constexpr int THREADS_N = BN / TN;
    constexpr int NUM_THREADS = THREADS_M * THREADS_N;
    constexpr int A_FRAGMENT_STRIDE = TM;
    constexpr int B_FRAGMENT_STRIDE = 12;
    constexpr int A_SMEM_K_STRIDE = THREADS_M * A_FRAGMENT_STRIDE + 4;
    constexpr int B_SMEM_K_STRIDE = THREADS_N * B_FRAGMENT_STRIDE + 8;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tid = ty * blockDim.x + tx;

    // Single buffering: one A tile and one B tile only.
    __shared__ float As[BK * A_SMEM_K_STRIDE];
    __shared__ float Bs[BK * B_SMEM_K_STRIDE];

    // Keep the v3 cooperative-load indexing. A has two float4 chunks per row.
    constexpr int load_threads_a_k = BK / 4;
    constexpr int load_threads_a_m =
        NUM_THREADS / load_threads_a_k;
    constexpr int num_row_strides_a = BM / load_threads_a_m;
    const int load_smem_a_m = tid / load_threads_a_k;
    const int load_smem_a_k = tid % load_threads_a_k * 4;

    constexpr int load_threads_b_n = BN / 4;
    constexpr int load_threads_b_k =
        NUM_THREADS / load_threads_b_n;
    constexpr int num_k_strides_b = BK / load_threads_b_k;
    const int load_smem_b_k = tid / load_threads_b_n;
    const int load_smem_b_n = tid % load_threads_b_n * 4;

    A += by * BM * K;
    B += bx * BN;
    C += by * BM * N + bx * BN;

    float sum[TM][TN] = {0.0f};
    const int comp_m = ty * TM;
    // Unlike v3, comp_n is the lane-local starting column. The eight columns
    // owned by a thread are comp_n + j * THREADS_N.
    const int comp_n = tx;

    float a_frag[TM];
    float b_frag[TN];

    for (int kb = 0; kb < K; kb += BK) {
        #pragma unroll
        for (int i = 0; i < num_row_strides_a; ++i) {
            const int row = load_smem_a_m + i * load_threads_a_m;
            const int col = load_smem_a_k;
            float4 value;
            if constexpr (kAligned) {
                value = load4(&A[OFFSET(row, col, K)]);
            } else {
                value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                const int global_row = by * BM + row;
                if (global_row < M) {
                    if (kb + col + 0 < K) {
                        value.x = A[OFFSET(row, col + 0, K)];
                    }
                    if (kb + col + 1 < K) {
                        value.y = A[OFFSET(row, col + 1, K)];
                    }
                    if (kb + col + 2 < K) {
                        value.z = A[OFFSET(row, col + 2, K)];
                    }
                    if (kb + col + 3 < K) {
                        value.w = A[OFFSET(row, col + 3, K)];
                    }
                }
            }

            const int fragment = row / TM;
            const int element = row % TM;
            const int base = fragment * A_FRAGMENT_STRIDE + element;
            As[(col + 0) * A_SMEM_K_STRIDE + base] = value.x;
            As[(col + 1) * A_SMEM_K_STRIDE + base] = value.y;
            As[(col + 2) * A_SMEM_K_STRIDE + base] = value.z;
            As[(col + 3) * A_SMEM_K_STRIDE + base] = value.w;
        }

        #pragma unroll
        for (int i = 0; i < num_k_strides_b; ++i) {
            const int k = load_smem_b_k + i * load_threads_b_k;
            const int n = load_smem_b_n;
            float4 value;
            if constexpr (kAligned) {
                value = load4(&B[OFFSET(k, n, N)]);
            } else {
                value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                const int global_k = kb + k;
                const int global_n = bx * BN + n;
                if (global_k < K) {
                    if (global_n + 0 < N) {
                        value.x = B[OFFSET(k, n + 0, N)];
                    }
                    if (global_n + 1 < N) {
                        value.y = B[OFFSET(k, n + 1, N)];
                    }
                    if (global_n + 2 < N) {
                        value.z = B[OFFSET(k, n + 2, N)];
                    }
                    if (global_n + 3 < N) {
                        value.w = B[OFFSET(k, n + 3, N)];
                    }
                }
            }

            const int element = n % THREADS_N;
            const int fragment = n / THREADS_N;
            const int base = k * B_SMEM_K_STRIDE;
            Bs[base + (element + 0) * B_FRAGMENT_STRIDE + fragment] =
                value.x;
            Bs[base + (element + 1) * B_FRAGMENT_STRIDE + fragment] =
                value.y;
            Bs[base + (element + 2) * B_FRAGMENT_STRIDE + fragment] =
                value.z;
            Bs[base + (element + 3) * B_FRAGMENT_STRIDE + fragment] =
                value.w;
        }

        __syncthreads();

        A += BK;
        B += BK * N;

        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            const float4 a0 = load4(
                &As[kk * A_SMEM_K_STRIDE + ty * A_FRAGMENT_STRIDE]);
            const float4 a1 = load4(
                &As[kk * A_SMEM_K_STRIDE + ty * A_FRAGMENT_STRIDE + 4]);
            a_frag[0] = a0.x;
            a_frag[1] = a0.y;
            a_frag[2] = a0.z;
            a_frag[3] = a0.w;
            a_frag[4] = a1.x;
            a_frag[5] = a1.y;
            a_frag[6] = a1.z;
            a_frag[7] = a1.w;

            const float4 b0 = load4(
                &Bs[kk * B_SMEM_K_STRIDE + tx * B_FRAGMENT_STRIDE]);
            const float4 b1 = load4(
                &Bs[kk * B_SMEM_K_STRIDE + tx * B_FRAGMENT_STRIDE + 4]);
            b_frag[0] = b0.x;
            b_frag[1] = b0.y;
            b_frag[2] = b0.z;
            b_frag[3] = b0.w;
            b_frag[4] = b1.x;
            b_frag[5] = b1.y;
            b_frag[6] = b1.z;
            b_frag[7] = b1.w;
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    sum[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }

        __syncthreads();
    }

    // For a fixed (i,j), tx=0..7 writes adjacent columns. Stores are therefore
    // coalesced at warp level even though each thread owns interleaved columns.
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int col = comp_n + j * THREADS_N;
            if constexpr (kAligned) {
                C[OFFSET(comp_m + i, col, N)] = sum[i][j];
            } else if (by * BM + comp_m + i < M &&
                       bx * BN + col < N) {
                C[OFFSET(comp_m + i, col, N)] = sum[i][j];
            }
        }
    }
}

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void gemm_test_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K) {
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
    const float *a_ptr = A + (by * BM + load_a_row) * K + load_a_col;
    // float *a_ptr_64 = A + (by * BM + load_a_row + 64) * K + load_a_col;
    const float *b_ptr = B + load_b_row * N + bx * BN + load_b_col;
    // float *b_ptr_8 = B + (load_b_row + 8) * N + bx * BN + load_b_col;

    // Prefetch first tile (costs 16 extra regs; commented-out ptrs above keep total under 128 to preserve occupancy)
    float4 tmp_a0 = load4(&a_ptr[0]);
    float4 tmp_a1 = load4(&a_ptr[64 * K]);
    float4 tmp_b0 = load4(&b_ptr[0]);
    float4 tmp_b1 = load4(&b_ptr[8 * N]);

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
        tmp_a0 = load4(&a_ptr[0]);
        tmp_a1 = load4(&a_ptr[64 * K]);
        tmp_b0 = load4(&b_ptr[0]);
        tmp_b1 = load4(&b_ptr[8 * N]);

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

namespace {

struct GemmShape {
    int m;
    int n;
    int k;
};

GemmShape check_gemm_tensors(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& c) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && c.is_cuda(),
                "A, B and C must be CUDA tensors");
    TORCH_CHECK(a.scalar_type() == torch::kFloat32 &&
                    b.scalar_type() == torch::kFloat32 &&
                    c.scalar_type() == torch::kFloat32,
                "A, B and C must have dtype torch.float32");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2,
                "A, B and C must be two-dimensional");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && c.is_contiguous(),
                "A, B and C must be contiguous");
    TORCH_CHECK(a.device() == b.device() && a.device() == c.device(),
                "A, B and C must be on the same CUDA device");
    TORCH_CHECK(a.size(1) == b.size(0),
                "incompatible GEMM input shapes");
    TORCH_CHECK(c.size(0) == a.size(0) && c.size(1) == b.size(1),
                "C has the wrong shape");
    TORCH_CHECK(a.size(0) <= std::numeric_limits<int>::max() &&
                    a.size(1) <= std::numeric_limits<int>::max() &&
                    b.size(1) <= std::numeric_limits<int>::max(),
                "matrix dimensions exceed the int32 kernel interface");
    return {
        static_cast<int>(a.size(0)),
        static_cast<int>(b.size(1)),
        static_cast<int>(a.size(1)),
    };
}

void cublas_gemm(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& c,
    bool use_tf32) {
    const GemmShape shape = check_gemm_tensors(a, b, c);
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cublasComputeType_t compute_type =
        use_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    const cublasGemmAlgo_t algorithm =
        use_tf32 ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT;
    const cublasStatus_t status = cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        shape.n,
        shape.m,
        shape.k,
        &alpha,
        b.data_ptr<float>(),
        CUDA_R_32F,
        shape.n,
        a.data_ptr<float>(),
        CUDA_R_32F,
        shape.k,
        &beta,
        c.data_ptr<float>(),
        CUDA_R_32F,
        shape.n,
        compute_type,
        algorithm);
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS,
                "cuBLAS GEMM failed with status ",
                static_cast<int>(status));
}

}  // namespace

void sgemm_cublas(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    cublas_gemm(a, b, c, false);
}

void sgemm_cublas_tf32(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    cublas_gemm(a, b, c, true);
}

void gemm_v0(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    const GemmShape shape = check_gemm_tensors(a, b, c);
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int BK = 32;
    constexpr int TM = 1;
    constexpr int TN = 1;
    TORCH_CHECK(shape.m % BM == 0 && shape.n % BN == 0, "gemm_v0 requires shape.m % BM == 0 && shape.n % BN == 0");
    const dim3 threads_per_block(BN / TN, BM / TM);
    const dim3 blocks_per_grid(
        (shape.n + BN - 1) / BN,
        (shape.m + BM - 1) / BM);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    gemm_v0_kernel<BM, BN, BK, TM, TN>
        <<<blocks_per_grid, threads_per_block, 0, stream>>>(
            a.data_ptr<float>(),
            b.data_ptr<float>(),
            c.data_ptr<float>(),
            shape.m,
            shape.n,
            shape.k);
}

void gemm_v1(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    const GemmShape shape = check_gemm_tensors(a, b, c);
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int BK = 32;
    constexpr int TM = 1;
    constexpr int TN = 1;
    TORCH_CHECK(true, "gemm_v1 requires true");
    const dim3 threads_per_block(BN / TN, BM / TM);
    const dim3 blocks_per_grid(
        (shape.n + BN - 1) / BN,
        (shape.m + BM - 1) / BM);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    gemm_v1_kernel<BM, BN, BK, TM, TN>
        <<<blocks_per_grid, threads_per_block, 0, stream>>>(
            a.data_ptr<float>(),
            b.data_ptr<float>(),
            c.data_ptr<float>(),
            shape.m,
            shape.n,
            shape.k);
}

void gemm_v2(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    const GemmShape shape = check_gemm_tensors(a, b, c);
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 16;
    constexpr int TM = 8;
    constexpr int TN = 8;
    TORCH_CHECK(shape.m % BM == 0 && shape.n % BN == 0 && shape.k % BK == 0, "gemm_v2 requires shape.m % BM == 0 && shape.n % BN == 0 && shape.k % BK == 0");
    const dim3 threads_per_block(BN / TN, BM / TM);
    const dim3 blocks_per_grid(
        (shape.n + BN - 1) / BN,
        (shape.m + BM - 1) / BM);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    gemm_v2_kernel<BM, BN, BK, TM, TN>
        <<<blocks_per_grid, threads_per_block, 0, stream>>>(
            a.data_ptr<float>(),
            b.data_ptr<float>(),
            c.data_ptr<float>(),
            shape.m,
            shape.n,
            shape.k);
}

void gemm_v2_warp_tiling(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    const GemmShape shape = check_gemm_tensors(a, b, c);
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 16;
    constexpr int TM = 8;
    constexpr int TN = 8;
    TORCH_CHECK(shape.m % BM == 0 && shape.n % BN == 0 && shape.k % BK == 0, "gemm_v2_warp_tiling requires shape.m % BM == 0 && shape.n % BN == 0 && shape.k % BK == 0");
    const dim3 threads_per_block(BN / TN, BM / TM);
    const dim3 blocks_per_grid(
        (shape.n + BN - 1) / BN,
        (shape.m + BM - 1) / BM);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    gemm_v2_warp_tiling_kernel<BM, BN, BK, TM, TN>
        <<<blocks_per_grid, threads_per_block, 0, stream>>>(
            a.data_ptr<float>(),
            b.data_ptr<float>(),
            c.data_ptr<float>(),
            shape.m,
            shape.n,
            shape.k);
}

void gemm_v2_warp_tiling_swizzle(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    const GemmShape shape = check_gemm_tensors(a, b, c);
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 16;
    constexpr int TM = 8;
    constexpr int TN = 8;
    TORCH_CHECK(shape.m % BM == 0 && shape.n % BN == 0 && shape.k % BK == 0, "gemm_v2_warp_tiling_swizzle requires shape.m % BM == 0 && shape.n % BN == 0 && shape.k % BK == 0");
    const dim3 threads_per_block(BN / TN, BM / TM);
    const dim3 blocks_per_grid(
        (shape.n + BN - 1) / BN,
        (shape.m + BM - 1) / BM);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    gemm_v2_warp_tiling_swizzle_kernel<BM, BN, BK, TM, TN>
        <<<blocks_per_grid, threads_per_block, 0, stream>>>(
            a.data_ptr<float>(),
            b.data_ptr<float>(),
            c.data_ptr<float>(),
            shape.m,
            shape.n,
            shape.k);
}

void gemm_v2_warp_tiling_swizzle_rw(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    const GemmShape shape = check_gemm_tensors(a, b, c);
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 16;
    constexpr int TM = 8;
    constexpr int TN = 8;
    TORCH_CHECK(shape.m % BM == 0 && shape.n % BN == 0 && shape.k % BK == 0, "gemm_v2_warp_tiling_swizzle_rw requires shape.m % BM == 0 && shape.n % BN == 0 && shape.k % BK == 0");
    const dim3 threads_per_block(BN / TN, BM / TM);
    const dim3 blocks_per_grid(
        (shape.n + BN - 1) / BN,
        (shape.m + BM - 1) / BM);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    gemm_v2_warp_tiling_swizzle_rw_kernel<BM, BN, BK, TM, TN>
        <<<blocks_per_grid, threads_per_block, 0, stream>>>(
            a.data_ptr<float>(),
            b.data_ptr<float>(),
            c.data_ptr<float>(),
            shape.m,
            shape.n,
            shape.k);
}

void gemm_v3(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    const GemmShape shape = check_gemm_tensors(a, b, c);
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;
    TORCH_CHECK(shape.m % BM == 0 && shape.n % BN == 0 && shape.k % BK == 0, "gemm_v3 requires shape.m % BM == 0 && shape.n % BN == 0 && shape.k % BK == 0");
    const dim3 threads_per_block(BN / TN, BM / TM);
    const dim3 blocks_per_grid(
        (shape.n + BN - 1) / BN,
        (shape.m + BM - 1) / BM);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    gemm_v3_kernel<BM, BN, BK, TM, TN>
        <<<blocks_per_grid, threads_per_block, 0, stream>>>(
            a.data_ptr<float>(),
            b.data_ptr<float>(),
            c.data_ptr<float>(),
            shape.m,
            shape.n,
            shape.k);
}

void gemm_v4(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    const GemmShape shape = check_gemm_tensors(a, b, c);
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 4;
    TORCH_CHECK(shape.m % 4 == 0 && shape.n % 4 == 0 && shape.k % 4 == 0, "gemm_v4 requires shape.m % 4 == 0 && shape.n % 4 == 0 && shape.k % 4 == 0");
    const dim3 threads_per_block(BN / TN, BM / TM);
    const dim3 blocks_per_grid(
        (shape.n + BN - 1) / BN,
        (shape.m + BM - 1) / BM);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    gemm_v4_kernel<BM, BN, BK, TM, TN>
        <<<blocks_per_grid, threads_per_block, 0, stream>>>(
            a.data_ptr<float>(),
            b.data_ptr<float>(),
            c.data_ptr<float>(),
            shape.m,
            shape.n,
            shape.k);
}

void gemm_v5(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    const GemmShape shape = check_gemm_tensors(a, b, c);
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;
    const dim3 threads_per_block(BN / TN, BM / TM);
    const dim3 blocks_per_grid(
        (shape.n + BN - 1) / BN,
        (shape.m + BM - 1) / BM);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (shape.m % BM == 0 && shape.n % BN == 0 && shape.k % BK == 0) {
        gemm_v5_kernel<true, BM, BN, BK, TM, TN>
            <<<blocks_per_grid, threads_per_block, 0, stream>>>(
                a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
                shape.m, shape.n, shape.k);
    } else {
        gemm_v5_kernel<false, BM, BN, BK, TM, TN>
            <<<blocks_per_grid, threads_per_block, 0, stream>>>(
                a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
                shape.m, shape.n, shape.k);
    }
}

void gemm_test(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    const GemmShape shape = check_gemm_tensors(a, b, c);
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 16;
    constexpr int TM = 8;
    constexpr int TN = 8;
    TORCH_CHECK(shape.m % BM == 0 && shape.n % BN == 0 && shape.k % BK == 0, "gemm_test requires shape.m % BM == 0 && shape.n % BN == 0 && shape.k % BK == 0");
    const dim3 threads_per_block((BM / TM) * (BN / TN));
    const dim3 blocks_per_grid(
        (shape.n + BN - 1) / BN,
        (shape.m + BM - 1) / BM);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    gemm_test_kernel<BM, BN, BK, TM, TN>
        <<<blocks_per_grid, threads_per_block, 0, stream>>>(
            a.data_ptr<float>(),
            b.data_ptr<float>(),
            c.data_ptr<float>(),
            shape.m,
            shape.n,
            shape.k);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    torch_pybinding_func(sgemm_cublas);
    torch_pybinding_func(sgemm_cublas_tf32);
    torch_pybinding_func(sgemm_tf32_bt);
    torch_pybinding_func(sgemm_tf32_bt_swizzle);
    torch_pybinding_func(sgemm_tf32_bt_swizzle_dbf);
    torch_pybinding_func(sgemm_tf32_swizzle_bcf);
    torch_pybinding_func(sgemm_tf32_swizzle_bcf_dbf);
    torch_pybinding_func(gemm_v0);
    torch_pybinding_func(gemm_v1);
    torch_pybinding_func(gemm_v2);
    torch_pybinding_func(gemm_v3);
    torch_pybinding_func(gemm_v4);
    torch_pybinding_func(gemm_v5);
    torch_pybinding_func(gemm_v2_warp_tiling);
    torch_pybinding_func(gemm_v2_warp_tiling_swizzle);
    torch_pybinding_func(gemm_v2_warp_tiling_swizzle_rw);
    torch_pybinding_func(gemm_test);
}
