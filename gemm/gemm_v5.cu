#include <cuda_runtime.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

namespace {

constexpr int THREAD_SIZE_M = 8;
constexpr int THREAD_SIZE_N = 8;
constexpr int BLOCK_SIZE_M = 64;
constexpr int BLOCK_SIZE_N = 64;
constexpr int BLOCK_SIZE_K = 8;

constexpr int BLOCK_THREADS_M = BLOCK_SIZE_M / THREAD_SIZE_M;
constexpr int BLOCK_THREADS_N = BLOCK_SIZE_N / THREAD_SIZE_N;
constexpr int THREAD_NUM_PER_BLOCK = BLOCK_THREADS_M * BLOCK_THREADS_N;

// Each thread consumes eight consecutive values using two float4 loads.
// A needs no gap between thread fragments; a four-float K-row padding makes
// its transposed staging stores conflict-free. B uses a 12-float fragment
// stride and an eight-float K-row padding so both its loads and stores map to
// disjoint bank groups.
constexpr int A_FRAGMENT_STRIDE = THREAD_SIZE_M;
constexpr int B_FRAGMENT_STRIDE = 12;
constexpr int A_SMEM_K_STRIDE = BLOCK_THREADS_M * A_FRAGMENT_STRIDE + 4;
constexpr int B_SMEM_K_STRIDE = BLOCK_THREADS_N * B_FRAGMENT_STRIDE + 8;

__device__ __forceinline__ float4 load4(const float* pointer) {
    return *reinterpret_cast<const float4*>(pointer);
}

template <bool kAligned>
__global__ void gemm_v5(
    int M,
    int N,
    int K,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tid = ty * blockDim.x + tx;

    // Single buffering: one A tile and one B tile only.
    __shared__ float As[BLOCK_SIZE_K * A_SMEM_K_STRIDE];
    __shared__ float Bs[BLOCK_SIZE_K * B_SMEM_K_STRIDE];

    // Keep the v3 cooperative-load indexing. A has two float4 chunks per row.
    constexpr int load_threads_a_k = BLOCK_SIZE_K / 4;
    constexpr int load_threads_a_m =
        THREAD_NUM_PER_BLOCK / load_threads_a_k;
    constexpr int num_row_strides_a = BLOCK_SIZE_M / load_threads_a_m;
    const int load_smem_a_m = tid / load_threads_a_k;
    const int load_smem_a_k = tid % load_threads_a_k * 4;

    constexpr int load_threads_b_n = BLOCK_SIZE_N / 4;
    constexpr int load_threads_b_k =
        THREAD_NUM_PER_BLOCK / load_threads_b_n;
    constexpr int num_k_strides_b = BLOCK_SIZE_K / load_threads_b_k;
    const int load_smem_b_k = tid / load_threads_b_n;
    const int load_smem_b_n = tid % load_threads_b_n * 4;

    A += by * BLOCK_SIZE_M * K;
    B += bx * BLOCK_SIZE_N;
    C += by * BLOCK_SIZE_M * N + bx * BLOCK_SIZE_N;

    float sum[THREAD_SIZE_M][THREAD_SIZE_N] = {0.0f};
    const int comp_m = ty * THREAD_SIZE_M;
    // Unlike v3, comp_n is the lane-local starting column. The eight columns
    // owned by a thread are comp_n + j * BLOCK_THREADS_N.
    const int comp_n = tx;

    float a_frag[THREAD_SIZE_M];
    float b_frag[THREAD_SIZE_N];

    for (int kb = 0; kb < K; kb += BLOCK_SIZE_K) {
        #pragma unroll
        for (int i = 0; i < num_row_strides_a; ++i) {
            const int row = load_smem_a_m + i * load_threads_a_m;
            const int col = load_smem_a_k;
            float4 value;
            if constexpr (kAligned) {
                value = load4(&A[OFFSET(row, col, K)]);
            } else {
                value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                const int global_row = by * BLOCK_SIZE_M + row;
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

            const int fragment = row / THREAD_SIZE_M;
            const int element = row % THREAD_SIZE_M;
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
                const int global_n = bx * BLOCK_SIZE_N + n;
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

            const int element = n % BLOCK_THREADS_N;
            const int fragment = n / BLOCK_THREADS_N;
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

        A += BLOCK_SIZE_K;
        B += BLOCK_SIZE_K * N;

        #pragma unroll
        for (int kk = 0; kk < BLOCK_SIZE_K; ++kk) {
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
            for (int i = 0; i < THREAD_SIZE_M; ++i) {
                #pragma unroll
                for (int j = 0; j < THREAD_SIZE_N; ++j) {
                    sum[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }

        __syncthreads();
    }

    // For a fixed (i,j), tx=0..7 writes adjacent columns. Stores are therefore
    // coalesced at warp level even though each thread owns interleaved columns.
    #pragma unroll
    for (int i = 0; i < THREAD_SIZE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < THREAD_SIZE_N; ++j) {
            const int col = comp_n + j * BLOCK_THREADS_N;
            if constexpr (kAligned) {
                C[OFFSET(comp_m + i, col, N)] = sum[i][j];
            } else if (by * BLOCK_SIZE_M + comp_m + i < M &&
                       bx * BLOCK_SIZE_N + col < N) {
                C[OFFSET(comp_m + i, col, N)] = sum[i][j];
            }
        }
    }
}

}  // namespace

void launch_gemm_v5(
    int M,
    int N,
    int K,
    float* A,
    float* B,
    float* C,
    cudaStream_t stream) {
    if (M <= 0 || N <= 0) {
        return;
    }

    const dim3 block(BLOCK_THREADS_N, BLOCK_THREADS_M);
    const dim3 grid((N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N,
                    (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);
    if (M % BLOCK_SIZE_M == 0 && N % BLOCK_SIZE_N == 0 &&
        K % BLOCK_SIZE_K == 0) {
        gemm_v5<true><<<grid, block, 0, stream>>>(M, N, K, A, B, C);
    } else {
        gemm_v5<false><<<grid, block, 0, stream>>>(M, N, K, A, B, C);
    }
}
