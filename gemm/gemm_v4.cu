#include <cuda_runtime.h>

#define OFFSET(row, col, lgd) ((row)*(lgd)+(col))

#define FETCH(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
// As: (BLOCK_SIZE_K, BLOCK_SIZE_M) 行主序
// Bs: (BLOCK_SIZE_K, BLOCK_SIZE_N) 行主序
//对sharemem再次进行分块，将sharemem分块到register中。
const int THREAD_SIZE_M=8;//每个线程计算的C中元素的高度
const int THREAD_SIZE_N=4;//每个线程计算的C中元素的宽度
const int BLOCK_SIZE_M=64; ///每个线程块需要处理的M维度数据块大小
const int BLOCK_SIZE_N=64; ///每个线程块需要处理的N维度数据块大小
const int BLOCK_SIZE_K=8;  //每个线程块需要A load into sharemen的宽度
//每个线程块的所包含的线程数量
const int THREAD_SIZE_PER_BLCOK_M=BLOCK_SIZE_M/THREAD_SIZE_M;   //4
const int THREAD_SIZE_PER_BLCOK_N=BLOCK_SIZE_N/THREAD_SIZE_N;   //4
const int THREAD_NUM_PER_BLOCK = THREAD_SIZE_PER_BLCOK_M*THREAD_SIZE_PER_BLCOK_N;
__device__ __forceinline__ float4 load4(const float* p) {
  // 假设 p 至少 16-byte 对齐；否则有些架构会慢或潜在异常
  return *reinterpret_cast<const float4*>(p);
}
//仅对M、K进行tile，这种情况下K不能太大，否则sharedmem不够用

// 边界处理以M、N、K均能被4整除为前提，从而保证float4访问完整且对齐。
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

    //实际上这个值化简后是4*THREAD_NUM_PER_BLOCK，但有时候可能I不是整除值，因此这里在算一遍
    //表示每个iter中需要搬运的float4的单元数量，也即i_stride。
    const int I_A = BLOCK_SIZE_M*BLOCK_SIZE_K/(4*THREAD_NUM_PER_BLOCK);
    const int I_A_stride = (BLOCK_SIZE_M*BLOCK_SIZE_K/(4*I_A));
    A = &A[by*BLOCK_SIZE_M*K];
    C = &C[by*BLOCK_SIZE_M*N+bx*BLOCK_SIZE_N];
    for (int i = 0; i < I_A; i++) {
        int q = i*I_A_stride + tid ;
        int row = q / (BLOCK_SIZE_K/4);
        int col = q % (BLOCK_SIZE_K/4) * 4;

        float4 a = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (by * BLOCK_SIZE_M + row < M && col < K) {
            a = load4(&A[OFFSET(row, col, K)]);
        }
        As[0][OFFSET(col + 0, row, BLOCK_SIZE_M)] = a.x;
        As[0][OFFSET(col + 1, row, BLOCK_SIZE_M)] = a.y;
        As[0][OFFSET(col + 2, row, BLOCK_SIZE_M)] = a.z;
        As[0][OFFSET(col + 3, row, BLOCK_SIZE_M)] = a.w;
    }

    const int I_B = BLOCK_SIZE_K*BLOCK_SIZE_N/(4*THREAD_NUM_PER_BLOCK);
    const int I_B_stride = (BLOCK_SIZE_K*BLOCK_SIZE_N/(4*I_B));
    B = &B[bx*BLOCK_SIZE_N];
    for(int i = 0;i < I_B;i++){
        int q = i*I_B_stride + tid;
        int row = q / (BLOCK_SIZE_N/4);
        int col = q % (BLOCK_SIZE_N/4) * 4;
        float4 b = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (row < K && bx * BLOCK_SIZE_N + col < N) {
            b = load4(&B[OFFSET(row, col, N)]);
        }
        Bs[0][OFFSET(row, col + 0, BLOCK_SIZE_N)] = b.x;
        Bs[0][OFFSET(row, col + 1, BLOCK_SIZE_N)] = b.y;
        Bs[0][OFFSET(row, col + 2, BLOCK_SIZE_N)] = b.z;
        Bs[0][OFFSET(row, col + 3, BLOCK_SIZE_N)] = b.w;
    }
    float reg_a[2][THREAD_SIZE_M];
    float reg_b[2][THREAD_SIZE_N];

    float accumalator[THREAD_SIZE_M][THREAD_SIZE_N] = {0.};

    
    __syncthreads();

    // x 对应 N 方向（列），y 对应 M 方向（行）
    int comp_m = ty * THREAD_SIZE_M;
    int comp_n = tx * THREAD_SIZE_N;
    #pragma unroll
    for (int m = 0; m < THREAD_SIZE_M; m++) {
        reg_a[0][m] = As[0][OFFSET(0, comp_m + m, BLOCK_SIZE_M)];
    }
    #pragma unroll
    for (int n = 0; n < THREAD_SIZE_N; n++) {
        reg_b[0][n] = Bs[0][OFFSET(0, comp_n + n, BLOCK_SIZE_N)];
    }

    int write_index = 1;
    int load_index = 0;
    float ldg_a_reg[4 * BLOCK_SIZE_K * BLOCK_SIZE_M / THREAD_NUM_PER_BLOCK / 4] = {0.};
    float ldg_b_reg[4 * BLOCK_SIZE_K * BLOCK_SIZE_N / THREAD_NUM_PER_BLOCK / 4] = {0.};
    float *A_ptr = A;
    float *B_ptr = B;

    for (int bk = 0; bk < K; bk += BLOCK_SIZE_K) {
        const bool has_next_block = bk + BLOCK_SIZE_K < K;

        // 提前把下一个K块从global memory搬到register。
        if (has_next_block) {
            A_ptr = A + bk + BLOCK_SIZE_K;
            B_ptr = B + (bk + BLOCK_SIZE_K) * N;
            #pragma unroll
            for (int i = 0; i < I_A; i++) {
                int q = i * I_A_stride + tid;
                int row = q / (BLOCK_SIZE_K / 4);
                int col = q % (BLOCK_SIZE_K / 4) * 4;
                int ldg_index = i * 4;

                float4 a = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (by * BLOCK_SIZE_M + row < M &&
                    bk + BLOCK_SIZE_K + col < K) {
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
                int row = q / (BLOCK_SIZE_N / 4);
                int col = q % (BLOCK_SIZE_N / 4) * 4;
                int ldg_index = i * 4;

                float4 b = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (bk + BLOCK_SIZE_K + row < K &&
                    bx * BLOCK_SIZE_N + col < N) {
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
        for (int kk = 0; kk < BLOCK_SIZE_K; kk++) {
            if (kk < BLOCK_SIZE_K - 1) {
                #pragma unroll
                for (int m = 0; m < THREAD_SIZE_M; m++) {
                    reg_a[(kk + 1) % 2][m] =
                        As[load_index][OFFSET(kk + 1, comp_m + m, BLOCK_SIZE_M)];
                }
                #pragma unroll
                for (int n = 0; n < THREAD_SIZE_N; n++) {
                    reg_b[(kk + 1) % 2][n] =
                        Bs[load_index][OFFSET(kk + 1, comp_n + n, BLOCK_SIZE_N)];
                }
            }

            #pragma unroll
            for (int m = 0; m < THREAD_SIZE_M; m++) {
                #pragma unroll
                for (int n = 0; n < THREAD_SIZE_N; n++) {
                    accumalator[m][n] += reg_a[kk % 2][m] * reg_b[kk % 2][n];
                }
            }
        }

        if (has_next_block) {
            // 使用与global load相同的q、row、col下标写入另一个shared-memory块。
            #pragma unroll
            for (int i = 0; i < I_A; i++) {
                int q = i * I_A_stride + tid;
                int row = q / (BLOCK_SIZE_K / 4);
                int col = q % (BLOCK_SIZE_K / 4) * 4;
                int ldg_index = i * 4;

                As[write_index][OFFSET(col + 0, row, BLOCK_SIZE_M)] = ldg_a_reg[ldg_index + 0];
                As[write_index][OFFSET(col + 1, row, BLOCK_SIZE_M)] = ldg_a_reg[ldg_index + 1];
                As[write_index][OFFSET(col + 2, row, BLOCK_SIZE_M)] = ldg_a_reg[ldg_index + 2];
                As[write_index][OFFSET(col + 3, row, BLOCK_SIZE_M)] = ldg_a_reg[ldg_index + 3];
            }

            #pragma unroll
            for (int i = 0; i < I_B; i++) {
                int q = i * I_B_stride + tid;
                int row = q / (BLOCK_SIZE_N / 4);
                int col = q % (BLOCK_SIZE_N / 4) * 4;
                int ldg_index = i * 4;

                Bs[write_index][OFFSET(row, col + 0, BLOCK_SIZE_N)] = ldg_b_reg[ldg_index + 0];
                Bs[write_index][OFFSET(row, col + 1, BLOCK_SIZE_N)] = ldg_b_reg[ldg_index + 1];
                Bs[write_index][OFFSET(row, col + 2, BLOCK_SIZE_N)] = ldg_b_reg[ldg_index + 2];
                Bs[write_index][OFFSET(row, col + 3, BLOCK_SIZE_N)] = ldg_b_reg[ldg_index + 3];
            }

            __syncthreads();

            load_index = write_index;
            write_index ^= 1;

            #pragma unroll
            for (int m = 0; m < THREAD_SIZE_M; m++) {
                reg_a[0][m] =
                    As[load_index][OFFSET(0, comp_m + m, BLOCK_SIZE_M)];
            }
            #pragma unroll
            for (int n = 0; n < THREAD_SIZE_N; n++) {
                reg_b[0][n] =
                    Bs[load_index][OFFSET(0, comp_n + n, BLOCK_SIZE_N)];
            }

        }
    }

    #pragma unroll
    for (int m = 0; m < THREAD_SIZE_M; m++) {
        #pragma unroll
        for (int n = 0; n < THREAD_SIZE_N; n += 4) {
            if (by * BLOCK_SIZE_M + comp_m + m < M &&
                bx * BLOCK_SIZE_N + comp_n + n < N) {
                float4 c = load4(&C[OFFSET(comp_m + m, comp_n + n, N)]);
                c.x = alpha * accumalator[m][n + 0] + beta * c.x;
                c.y = alpha * accumalator[m][n + 1] + beta * c.y;
                c.z = alpha * accumalator[m][n + 2] + beta * c.z;
                c.w = alpha * accumalator[m][n + 3] + beta * c.w;
                FETCH(C[OFFSET(comp_m + m, comp_n + n, N)]) = c;
            }
        }
    }

}

void launch_gemm_v4(
    int M, int N, int K, float* A, float* B, float* C,
    cudaStream_t stream) {
    if (M <= 0 || N <= 0) {
        return;
    }
    const dim3 block(THREAD_SIZE_PER_BLCOK_N, THREAD_SIZE_PER_BLCOK_M);
    const dim3 grid((N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N,
                    (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);
    gemm_v4<<<grid, block, 0, stream>>>(M, N, K, 1.0f, A, B, 0.0f, C);
}
