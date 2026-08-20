#include <cuda_runtime.h>

#define OFFSET(row, col, lgd) ((row)*(lgd)+(col))

#define FETCH(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

const int THREAD_SIZE_M = 8;
const int THREAD_SIZE_N = 8;
const int BLOCK_SIZE_M = 64;
const int BLOCK_SIZE_N = 64;
const int BLOCK_SIZE_K = 32;
const int THREAD_SIZE_PER_BLOCK_M = BLOCK_SIZE_M / THREAD_SIZE_M;
const int THREAD_SIZE_PER_BLOCK_N = BLOCK_SIZE_N / THREAD_SIZE_N;
const int THREAD_SIZE_PER_BLCOK_M=BLOCK_SIZE_M/THREAD_SIZE_M;   //8
const int THREAD_SIZE_PER_BLCOK_N=BLOCK_SIZE_N/THREAD_SIZE_N;   //8
const int THREAD_NUM_PER_BLOCK = THREAD_SIZE_PER_BLOCK_M * THREAD_SIZE_PER_BLOCK_N;

__device__ __forceinline__ float4 load4(const float* p) {
    return *reinterpret_cast<const float4*>(p);
}

__global__ void  __launch_bounds__(256) gemm_v5(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tid = ty * blockDim.x + tx;

    // 共享内存，双缓冲
    __shared__ float As[2][BLOCK_SIZE_K][BLOCK_SIZE_M+4];  // (BK, BM)
    __shared__ float Bs[2][BLOCK_SIZE_K][BLOCK_SIZE_N+4];  // (BK, BN)

    // 加载A的线程配置
    const int load_threads_a_k = BLOCK_SIZE_K / 4;  // 每个线程负责4个K维度元素
    const int load_threads_a_m = THREAD_NUM_PER_BLOCK / load_threads_a_k;
    const int num_row_strides_a = BLOCK_SIZE_M / load_threads_a_m;

    // 当前线程负责加载的A共享内存位置
    int load_smem_a_m = tid / load_threads_a_k;  // M维度索引
    int load_smem_a_k = (tid % load_threads_a_k) * 4;  // K维度起始索引

    // 加载B的线程配置
    const int load_threads_b_n = BLOCK_SIZE_N / 4;
    const int load_threads_b_k = THREAD_NUM_PER_BLOCK / load_threads_b_n;
    const int num_k_strides_b = BLOCK_SIZE_K / load_threads_b_k;

    // 当前线程负责加载的B共享内存位置
    int load_smem_b_k = tid / load_threads_b_n;  // K维度索引
    int load_smem_b_n = (tid % load_threads_b_n) * 4;  // N维度起始索引

    // 每个线程需要加载的float4数量
    const int ldg_a_num = BLOCK_SIZE_K * BLOCK_SIZE_M / THREAD_NUM_PER_BLOCK / 4;
    const int ldg_b_num = BLOCK_SIZE_K * BLOCK_SIZE_N / THREAD_NUM_PER_BLOCK / 4;

    // 计算当前块在全局内存中的起始位置
    A = &A[by * BLOCK_SIZE_M * K];
    B = &B[bx * BLOCK_SIZE_N];
    C = &C[by * BLOCK_SIZE_M * N + bx * BLOCK_SIZE_N];

    // 当前线程计算的C块内起始位置
    int comp_m = ty * THREAD_SIZE_M;  // M维度起始
    int comp_n = tx * THREAD_SIZE_N;  // N维度起始

    // 累加寄存器
    float accum[THREAD_SIZE_M][THREAD_SIZE_N] = {{0.0f}};

    // 寄存器片段缓存（双缓冲）
    float a_frag[2][THREAD_SIZE_M];
    float b_frag[2][THREAD_SIZE_N];

    // 全局内存加载寄存器缓存
    float ldg_a_reg[4 * ldg_a_num];
    float ldg_b_reg[4 * ldg_b_num];

    // --- 第一阶段：加载第一个块到共享内存缓冲区0 ---
    // 加载A到共享内存
    #pragma unroll
    for (int i = 0; i < num_row_strides_a; i++) {
        int row = load_smem_a_m + i * load_threads_a_m;  // 全局M索引
        int col = load_smem_a_k;  // 全局K起始索引
        
        float4 va = load4(&A[row * K + col]);
        
        // 写入到共享内存As[0]，转置存储：As[K][M]
        As[0][col][row] = va.x;
        As[0][col + 1][row] = va.y;
        As[0][col + 2][row] = va.z;
        As[0][col + 3][row] = va.w;
    }

    // 加载B到共享内存
    #pragma unroll
    for (int i = 0; i < num_k_strides_b; i++) {
        int row = load_smem_b_k + i * load_threads_b_k;  // 全局K索引
        int col = load_smem_b_n;  // 全局N起始索引
        
        float4 vb = load4(&B[row * N + col]);
        
        // 写入到共享内存Bs[0]，保持布局：Bs[K][N]
        Bs[0][row][col] = vb.x;
        Bs[0][row][col + 1] = vb.y;
        Bs[0][row][col + 2] = vb.z;
        Bs[0][row][col + 3] = vb.w;
    }

    __syncthreads();

    // 从共享内存加载第一个K=0到寄存器
    #pragma unroll
    for (int m = 0; m < THREAD_SIZE_M; m++) {
        a_frag[0][m] = As[0][0][comp_m + m];
    }
    #pragma unroll
    for (int n = 0; n < THREAD_SIZE_N; n++) {
        b_frag[0][n] = Bs[0][0][comp_n + n];
    }

    // --- 主循环 ---
    int write_stage = 1;  // 下一个要写入的缓冲区
    int read_stage = 0;   // 当前正在读取的缓冲区
    int k = 0;

    do {
        k += BLOCK_SIZE_K;

        // 预取下一个块到寄存器（如果还有）
        if (k < K) {
            // 预取A的下一个块
            #pragma unroll
            for (int i = 0; i < num_row_strides_a; i++) {
                int row = load_smem_a_m + i * load_threads_a_m;
                int col = load_smem_a_k;
                int ldg_idx = i * 4;
                
                float4 va = load4(&A[row * K + k + col]);
                ldg_a_reg[ldg_idx] = va.x;
                ldg_a_reg[ldg_idx + 1] = va.y;
                ldg_a_reg[ldg_idx + 2] = va.z;
                ldg_a_reg[ldg_idx + 3] = va.w;
            }

            // 预取B的下一个块
            #pragma unroll
            for (int i = 0; i < num_k_strides_b; i++) {
                int row = load_smem_b_k + i * load_threads_b_k;
                int col = load_smem_b_n;
                int ldg_idx = i * 4;
                
                float4 vb = load4(&B[(k + row) * N + col]);
                ldg_b_reg[ldg_idx] = vb.x;
                ldg_b_reg[ldg_idx + 1] = vb.y;
                ldg_b_reg[ldg_idx + 2] = vb.z;
                ldg_b_reg[ldg_idx + 3] = vb.w;
            }
        }

        // 计算当前块（使用read_stage缓冲区的数据）
        for (int kk = 0; kk < BLOCK_SIZE_K; kk++) {
            // 预取下一个kk+1（如果是最后一次迭代，从下一个缓冲区预取）
            if (kk < BLOCK_SIZE_K - 1) {
                #pragma unroll
                for (int m = 0; m < THREAD_SIZE_M; m++) {
                    a_frag[(kk + 1) % 2][m] = As[read_stage][kk + 1][comp_m + m];
                }
                #pragma unroll
                for (int n = 0; n < THREAD_SIZE_N; n++) {
                    b_frag[(kk + 1) % 2][n] = Bs[read_stage][kk + 1][comp_n + n];
                }
            }
            
            // 计算当前kk
            #pragma unroll
            for (int m = 0; m < THREAD_SIZE_M; m++) {
                #pragma unroll
                for (int n = 0; n < THREAD_SIZE_N; n++) {
                    accum[m][n] += a_frag[kk % 2][m] * b_frag[kk % 2][n];
                }
            }
        }

        // 如果还有下一个块，将预取的数据写入共享内存
        if (k < K) {
            // 将预取的A写入共享内存write_stage缓冲区
            #pragma unroll
            for (int i = 0; i < num_row_strides_a; i++) {
                int row = load_smem_a_m + i * load_threads_a_m;
                int col = load_smem_a_k;
                int ldg_idx = i * 4;
                
                As[write_stage][col][row] = ldg_a_reg[ldg_idx];
                As[write_stage][col + 1][row] = ldg_a_reg[ldg_idx + 1];
                As[write_stage][col + 2][row] = ldg_a_reg[ldg_idx + 2];
                As[write_stage][col + 3][row] = ldg_a_reg[ldg_idx + 3];
            }

            // 将预取的B写入共享内存write_stage缓冲区
            #pragma unroll
            for (int i = 0; i < num_k_strides_b; i++) {
                int row = load_smem_b_k + i * load_threads_b_k;
                int col = load_smem_b_n;
                int ldg_idx = i * 4;
                
                Bs[write_stage][row][col] = ldg_b_reg[ldg_idx];
                Bs[write_stage][row][col + 1] = ldg_b_reg[ldg_idx + 1];
                Bs[write_stage][row][col + 2] = ldg_b_reg[ldg_idx + 2];
                Bs[write_stage][row][col + 3] = ldg_b_reg[ldg_idx + 3];
            }

            __syncthreads();

            // 交换缓冲区
            read_stage ^= 1;
            write_stage ^= 1;

            // 从新缓冲区加载K=0到寄存器
            #pragma unroll
            for (int m = 0; m < THREAD_SIZE_M; m++) {
                a_frag[0][m] = As[read_stage][0][comp_m + m];
            }
            #pragma unroll
            for (int n = 0; n < THREAD_SIZE_N; n++) {
                b_frag[0][n] = Bs[read_stage][0][comp_n + n];
            }
        }
    } while (k < K);

    // --- 写回结果 ---
    #pragma unroll
    for (int m = 0; m < THREAD_SIZE_M; m++) {
        int row = comp_m + m;
        #pragma unroll
        for (int n = 0; n < THREAD_SIZE_N; n += 4) {
            int col = comp_n + n;
            
            float4 c_val;
            c_val.x = C[row * N + col];
            c_val.y = C[row * N + col + 1];
            c_val.z = C[row * N + col + 2];
            c_val.w = C[row * N + col + 3];
            
            c_val.x = alpha * accum[m][n] + beta * c_val.x;
            c_val.y = alpha * accum[m][n + 1] + beta * c_val.y;
            c_val.z = alpha * accum[m][n + 2] + beta * c_val.z;
            c_val.w = alpha * accum[m][n + 3] + beta * c_val.w;
            
            float4* c_ptr = reinterpret_cast<float4*>(&C[row * N + col]);
            *c_ptr = c_val;
        }
    }
}

void launch_gemm_v5(
    int M, int N, int K, float* A, float* B, float* C,
    cudaStream_t stream) {
    const dim3 block(THREAD_SIZE_PER_BLCOK_N, THREAD_SIZE_PER_BLCOK_M);
    const dim3 grid((N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N,
                    (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);
    gemm_v5<<<grid, block, 0, stream>>>(M, N, K, 1.0f, A, B, 0.0f, C);
}
