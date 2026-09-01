#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#define SWIZZLE_A(row, col) ((col) ^ (((row >> 1) & 0x3) << 3))
#define SWIZZLE_C(row, col) ((col) ^ (((row) & 0x7) << 3))

#define SWIZZLE_B(row, col) ((col) ^ (((row) & 0x7) << 3))
#include <limits>
#define FLOAT2(value) (reinterpret_cast<float2 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])

#define torch_pybinding_func(function) module.def(#function, &function, #function)
constexpr int WARP_SIZE = 32;
// ---------------- Inline PTX assembly macros ----------------
// cp.async: async 16-byte copy from gmem (src) to smem (dst_smem_32b)
#define CP_ASYNC_CG(dst_smem_32b, src_global_ptr)                                                                      \
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::"r"(dst_smem_32b), "l"(src_global_ptr))

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_GROUP_0() asm volatile("cp.async.wait_group 0;\n" ::)

// ldmatrix
#define LDMATRIX_X4(R0, R1, R2, R3, PTR)                                                                               \
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"                                    \
                 : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                                                              \
                 : "r"(PTR))

#define LDMATRIX_X2(R0, R1, PTR)                                                                                       \
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];" : "=r"(R0), "=r"(R1) : "r"(PTR))

// Load two 8x8 tiles and transpose
#define LDMATRIX_X2_TRANS(R0, R1, PTR)                                                                                 \
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];" : "=r"(R0), "=r"(R1) : "r"(PTR))

// mma.sync
#define M16N8K16_F16(C0, C1, C2, C3, A0, A1, A2, A3, B0, B1)                                                           \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "                                                  \
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"                                         \
                 : "=f"(C0), "=f"(C1), "=f"(C2), "=f"(C3)                                                              \
                 : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(B0), "r"(B1), "f"(C0), "f"(C1), "f"(C2), "f"(C3))

#define M16N8K16_BF16(C0, C1, C2, C3, A0, A1, A2, A3, B0, B1)                                                          \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "                                                \
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"                                         \
                 : "=f"(C0), "=f"(C1), "=f"(C2), "=f"(C3)                                                              \
                 : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(B0), "r"(B1), "f"(C0), "f"(C1), "f"(C2), "f"(C3))

// ------------------------------------------ ldmatrix + mma  ----------------------------------------------------
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
    TORCH_CHECK(a.scalar_type() == torch::kFloat16 &&
                    b.scalar_type() == torch::kFloat16 &&
                    c.scalar_type() == torch::kFloat16,
                "A, B and C must have dtype torch.float16");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2,
                "A, B and C must be two-dimensional");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && c.is_contiguous(),
                "A, B and C must be contiguous");
    TORCH_CHECK(a.device() == b.device() && a.device() == c.device(),
                "A, B and C must be on the same CUDA device");
    TORCH_CHECK(a.size(1) == b.size(0), "incompatible GEMM input shapes");
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

void check_cuda_launch(const char* kernel_name) {
    const cudaError_t error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess,
                kernel_name,
                " launch failed: ",
                cudaGetErrorString(error));
}
template <int BM = 128, int BN = 128, int BK = 32,typename T = __half>
__global__ void hgemm_v1_kernel(
    T* __restrict__ a,
    T* __restrict__ b,
    T* __restrict__ c,
    int m,
    int n,
    int k) {
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int row = blockIdx.y * BM + ty;
        int col = blockIdx.x * BN + tx;
        int tid = ty * blockDim.x + tx;

        int warp_id = tid / WARP_SIZE;
        int lane_id = tid % WARP_SIZE;
        int warp_m = warp_id/4;
        int warp_n = warp_id%4;

        a = a + blockIdx.y * BM * k;
        b = b + blockIdx.x * BN;
        c = c + blockIdx.y * BM * n + blockIdx.x * BN;

        int load_a_smem_m = tid / 4;
        int load_a_smem_k = (tid % 4) * 8;
        int load_b_smem_k = tid / 16;
        int load_b_smem_n = (tid % 16) * 8;

        __shared__ T As[BM][BK];
        __shared__ T Bs[BK][BN];
        float reg_c[4][4][4] = {0};  
        for(int bk = 0; bk < k; bk += BK) {
            
            uint32_t smem_a0 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[load_a_smem_m][load_a_smem_k]));
            uint32_t smem_a1 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[load_a_smem_m + 64][load_a_smem_k]));

            T * global_a0 = &a[load_a_smem_m * k + bk + load_a_smem_k];
            T * global_a1 = &a[(64+load_a_smem_m) * k + bk + load_a_smem_k];

            CP_ASYNC_CG(smem_a0, global_a0);
            CP_ASYNC_CG(smem_a1, global_a1);

            
            uint32_t smem_b0 = static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[load_b_smem_k][load_b_smem_n]));
            uint32_t smem_b1 = static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[load_b_smem_k + 16][load_b_smem_n]));

            T * global_b0 = &b[(bk + load_b_smem_k) * n + load_b_smem_n];
            T * global_b1 = &b[(bk + load_b_smem_k + 16) * n + load_b_smem_n];

            CP_ASYNC_CG(smem_b0, global_b0);
            CP_ASYNC_CG(smem_b1, global_b1);

            CP_ASYNC_COMMIT_GROUP();
            CP_ASYNC_WAIT_GROUP_0();
            __syncthreads();
            for(int bk_step = 0; bk_step < 2;bk_step++){
                int bk_offset = bk_step * 16;
                int reg_a[4][4];
                int reg_b[4][2];

                for(int m_idx = 0;m_idx < 4;m_idx++){

                    int load_reg_a_n = warp_m*64 + m_idx*16 + (lane_id % 16);
                    int load_reg_a_k = bk_offset + (lane_id / 16) * 8;
                    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&As[load_reg_a_n][load_reg_a_k]));
                    LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);

                }
                for (int n_idx = 0; n_idx < 4; ++n_idx) {
                    // Lane 0~15 的线程恰好覆盖了 16 行 （两块 8x8 的首地址）
                    int b_row = bk_offset + (lane_id % 16);
                    int b_col = warp_n * 32 + n_idx * 8;

                    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[b_row][b_col]));
                    LDMATRIX_X2_TRANS(reg_b[n_idx][0], reg_b[n_idx][1], smem_addr);
                }
                for(int m_idx = 0; m_idx < 4; ++m_idx) {
                    for(int n_idx = 0; n_idx < 4; ++n_idx) {
                        M16N8K16_F16(
                            reg_c[m_idx][n_idx][0],
                            reg_c[m_idx][n_idx][1],
                            reg_c[m_idx][n_idx][2],
                            reg_c[m_idx][n_idx][3],
                            reg_a[m_idx][0],
                            reg_a[m_idx][1],
                            reg_a[m_idx][2],
                            reg_a[m_idx][3],
                            reg_b[n_idx][0],
                            reg_b[n_idx][1]
                        );
                    }
                }
            }
            __syncthreads();
        }

   // ---------------- Store C ----------------
    int t_row = lane_id / 4;       // 0~7
    int t_col = (lane_id % 4) * 2; // 0, 2, 4, 6

#pragma unroll
    for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
        for (int n_idx = 0; n_idx < 4; ++n_idx) {
            int c_base_row = warp_m * 64 + m_idx * 16;
            int c_base_col = warp_n * 32 + n_idx * 8;
            int idx_0 = (c_base_row + t_row) * n + c_base_col + t_col;
            int idx_2 = (c_base_row + t_row + 8) * n + c_base_col + t_col;
            HALF2(c[idx_0]) = __float22half2_rn(FLOAT2(reg_c[m_idx][n_idx][0]));
            HALF2(c[idx_2]) = __float22half2_rn(FLOAT2(reg_c[m_idx][n_idx][2]));
        }
    }
}

template <int BM = 128, int BN = 128, int BK = 32,typename T = __half>
__global__ void hgemm_gridswizzle_kernel(
    T* __restrict__ a,
    T* __restrict__ b,
    T* __restrict__ c,
    int m,
    int n,
    int k) {
        int linear_id = blockIdx.y * gridDim.x + blockIdx.x;
        const int SWIZZLE_W = 8; // 将执行块设置为 8 的宽度
        int bx = (linear_id % SWIZZLE_W) + (linear_id / (SWIZZLE_W * gridDim.y)) * SWIZZLE_W;
        int by = (linear_id / SWIZZLE_W) % gridDim.y;


        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int row = by * BM + ty;
        int col = bx * BN + tx;
        int tid = ty * blockDim.x + tx;
        int warp_id = tid / WARP_SIZE;
        int lane_id = tid % WARP_SIZE;
        int warp_m = warp_id/4;
        int warp_n = warp_id%4;

        a = a + by * BM * k;
        b = b + bx * BN;
        c = c + by * BM * n + bx * BN;

        int load_a_smem_m = tid / 4;
        int load_a_smem_k = (tid % 4) * 8;
        int load_b_smem_k = tid / 16;
        int load_b_smem_n = (tid % 16) * 8;

        __shared__ T As[BM][BK];
        __shared__ T Bs[BK][BN];
        float reg_c[4][4][4] = {0};  
        for(int bk = 0; bk < k; bk += BK) {
            
            uint32_t smem_a0 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[load_a_smem_m][load_a_smem_k]));
            uint32_t smem_a1 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[load_a_smem_m + 64][load_a_smem_k]));

            T * global_a0 = &a[load_a_smem_m * k + bk + load_a_smem_k];
            T * global_a1 = &a[(64+load_a_smem_m) * k + bk + load_a_smem_k];

            CP_ASYNC_CG(smem_a0, global_a0);
            CP_ASYNC_CG(smem_a1, global_a1);

            
            uint32_t smem_b0 = static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[load_b_smem_k][load_b_smem_n]));
            uint32_t smem_b1 = static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[load_b_smem_k + 16][load_b_smem_n]));

            T * global_b0 = &b[(bk + load_b_smem_k) * n + load_b_smem_n];
            T * global_b1 = &b[(bk + load_b_smem_k + 16) * n + load_b_smem_n];

            CP_ASYNC_CG(smem_b0, global_b0);
            CP_ASYNC_CG(smem_b1, global_b1);

            CP_ASYNC_COMMIT_GROUP();
            CP_ASYNC_WAIT_GROUP_0();
            __syncthreads();
            for(int bk_step = 0; bk_step < 2;bk_step++){
                int bk_offset = bk_step * 16;
                int reg_a[4][4];
                int reg_b[4][2];

                for(int m_idx = 0;m_idx < 4;m_idx++){

                    int load_reg_a_n = warp_m*64 + m_idx*16 + (lane_id % 16);
                    int load_reg_a_k = bk_offset + (lane_id / 16) * 8;
                    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&As[load_reg_a_n][load_reg_a_k]));
                    LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);

                }
                for (int n_idx = 0; n_idx < 4; ++n_idx) {
                    // Lane 0~15 的线程恰好覆盖了 16 行 （两块 8x8 的首地址）
                    int b_row = bk_offset + (lane_id % 16);
                    int b_col = warp_n * 32 + n_idx * 8;

                    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[b_row][b_col]));
                    LDMATRIX_X2_TRANS(reg_b[n_idx][0], reg_b[n_idx][1], smem_addr);
                }
                for(int m_idx = 0; m_idx < 4; ++m_idx) {
                    for(int n_idx = 0; n_idx < 4; ++n_idx) {
                        M16N8K16_F16(
                            reg_c[m_idx][n_idx][0],
                            reg_c[m_idx][n_idx][1],
                            reg_c[m_idx][n_idx][2],
                            reg_c[m_idx][n_idx][3],
                            reg_a[m_idx][0],
                            reg_a[m_idx][1],
                            reg_a[m_idx][2],
                            reg_a[m_idx][3],
                            reg_b[n_idx][0],
                            reg_b[n_idx][1]
                        );
                    }
                }
            }
            __syncthreads();
        }

   // ---------------- Store C ----------------
    int t_row = lane_id / 4;       // 0~7
    int t_col = (lane_id % 4) * 2; // 0, 2, 4, 6

#pragma unroll
    for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
        for (int n_idx = 0; n_idx < 4; ++n_idx) {
            int c_base_row = warp_m * 64 + m_idx * 16;
            int c_base_col = warp_n * 32 + n_idx * 8;
            int idx_0 = (c_base_row + t_row) * n + c_base_col + t_col;
            int idx_2 = (c_base_row + t_row + 8) * n + c_base_col + t_col;
            HALF2(c[idx_0]) = __float22half2_rn(FLOAT2(reg_c[m_idx][n_idx][0]));
            HALF2(c[idx_2]) = __float22half2_rn(FLOAT2(reg_c[m_idx][n_idx][2]));
        }
    }
}



template <int BM = 128, int BN = 128, int BK = 32,typename T = __half>
__global__ void hgemm_gridswizzle_smemswizzle_kernel(
    T* __restrict__ a,
    T* __restrict__ b,
    T* __restrict__ c,
    int m,
    int n,
    int k) {
        int linear_id = blockIdx.y * gridDim.x + blockIdx.x;
        const int SWIZZLE_W = 8; // 将执行块设置为 8 的宽度
        int bx = (linear_id % SWIZZLE_W) + (linear_id / (SWIZZLE_W * gridDim.y)) * SWIZZLE_W;
        int by = (linear_id / SWIZZLE_W) % gridDim.y;


        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int row = by * BM + ty;
        int col = bx * BN + tx;
        int tid = ty * blockDim.x + tx;
        int warp_id = tid / WARP_SIZE;
        int lane_id = tid % WARP_SIZE;
        int warp_m = warp_id/4;
        int warp_n = warp_id%4;

        a = a + by * BM * k;
        b = b + bx * BN;
        c = c + by * BM * n + bx * BN;

        int load_a_smem_m = tid / 4;
        int load_a_smem_k = (tid % 4) * 8;
        int load_b_smem_k = tid / 16;
        int load_b_smem_n = (tid % 16) * 8;

        __shared__ T As[BM][BK];
        __shared__ T Bs[BK][BN];
        float reg_c[4][4][4] = {0};  
        for(int bk = 0; bk < k; bk += BK) {
            
            uint32_t smem_a0 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[load_a_smem_m][SWIZZLE_A(load_a_smem_m, load_a_smem_k)]));
            uint32_t smem_a1 = static_cast<uint32_t>(__cvta_generic_to_shared(&As[load_a_smem_m + 64][SWIZZLE_A(load_a_smem_m + 64, load_a_smem_k)]));

            T * global_a0 = &a[load_a_smem_m * k + bk + load_a_smem_k];
            T * global_a1 = &a[(64+load_a_smem_m) * k + bk + load_a_smem_k];

            CP_ASYNC_CG(smem_a0, global_a0);
            CP_ASYNC_CG(smem_a1, global_a1);

            
            uint32_t smem_b0 = static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[load_b_smem_k][SWIZZLE_B(load_b_smem_k, load_b_smem_n)]));
            uint32_t smem_b1 = static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[load_b_smem_k + 16][SWIZZLE_B(load_b_smem_k + 16, load_b_smem_n)]));

            T * global_b0 = &b[(bk + load_b_smem_k) * n + load_b_smem_n];
            T * global_b1 = &b[(bk + load_b_smem_k + 16) * n + load_b_smem_n];

            CP_ASYNC_CG(smem_b0, global_b0);
            CP_ASYNC_CG(smem_b1, global_b1);

            CP_ASYNC_COMMIT_GROUP();
            CP_ASYNC_WAIT_GROUP_0();
            __syncthreads();
            for(int bk_step = 0; bk_step < 2;bk_step++){
                int bk_offset = bk_step * 16;
                int reg_a[4][4];
                int reg_b[4][2];

                for(int m_idx = 0;m_idx < 4;m_idx++){

                    int load_reg_a_n = warp_m*64 + m_idx*16 + (lane_id % 16);
                    int load_reg_a_k = bk_offset + (lane_id / 16) * 8;
                    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&As[load_reg_a_n][SWIZZLE_A(load_reg_a_n, load_reg_a_k)]));
                    LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);

                }
                for (int n_idx = 0; n_idx < 4; ++n_idx) {
                    // Lane 0~15 的线程恰好覆盖了 16 行 （两块 8x8 的首地址）
                    int b_row = bk_offset + (lane_id % 16);
                    int b_col = warp_n * 32 + n_idx * 8;

                    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[b_row][SWIZZLE_B(b_row, b_col)]));
                    LDMATRIX_X2_TRANS(reg_b[n_idx][0], reg_b[n_idx][1], smem_addr);
                }
                for(int m_idx = 0; m_idx < 4; ++m_idx) {
                    for(int n_idx = 0; n_idx < 4; ++n_idx) {
                        M16N8K16_F16(
                            reg_c[m_idx][n_idx][0],
                            reg_c[m_idx][n_idx][1],
                            reg_c[m_idx][n_idx][2],
                            reg_c[m_idx][n_idx][3],
                            reg_a[m_idx][0],
                            reg_a[m_idx][1],
                            reg_a[m_idx][2],
                            reg_a[m_idx][3],
                            reg_b[n_idx][0],
                            reg_b[n_idx][1]
                        );
                    }
                }
            }
            __syncthreads();
        }

   // ---------------- Store C ----------------
    int t_row = lane_id / 4;       // 0~7
    int t_col = (lane_id % 4) * 2; // 0, 2, 4, 6

#pragma unroll
    for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
        for (int n_idx = 0; n_idx < 4; ++n_idx) {
            int c_base_row = warp_m * 64 + m_idx * 16;
            int c_base_col = warp_n * 32 + n_idx * 8;
            int idx_0 = (c_base_row + t_row) * n + c_base_col + t_col;
            int idx_2 = (c_base_row + t_row + 8) * n + c_base_col + t_col;
            HALF2(c[idx_0]) = __float22half2_rn(FLOAT2(reg_c[m_idx][n_idx][0]));
            HALF2(c[idx_2]) = __float22half2_rn(FLOAT2(reg_c[m_idx][n_idx][2]));
        }
    }
}

}

// a block calculate c[128][128]
template <const int BM = 128, const int BN = 128, const int BK = 32, typename T>
__global__ void hgemm_bcf_dbf_kernel(T *a, T *b, T *c, int m, int n, int k) {
    // grid swizzling
    int linear_id = blockIdx.y * gridDim.x + blockIdx.x;
    const int SWIZZLE_W = gridDim.x % 8 == 0 ? 8 : gridDim.x;

    int bx = (linear_id % SWIZZLE_W) + (linear_id / (SWIZZLE_W * gridDim.y)) * SWIZZLE_W;
    int by = (linear_id / SWIZZLE_W) % gridDim.y;

    int tid = threadIdx.x; // 0~255
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // GMEM->SMEM load indexing
    int load_a_row = tid / 4;        // 0~63
    int load_a_col = (tid % 4) * 8;  // 0,8,16,24
    int load_b_row = tid / 16;       // 0..15 (K)
    int load_b_col = (tid % 16) * 8; // 0,8,...,120 (N)

    // Layout works for both A and B (row-major friendly)
    __shared__ T As[2][BM][BK];
    __shared__ T Bs[2][BK][BN];

    // warp tiling
    // Each warp owns a 64x32 C tile
    int warp_id_m = warp_id / 4; // 0, 1
    int warp_id_n = warp_id % 4; // 0, 1, 2, 3

    // Accumulators: 4 M-fragments * 4 N-fragments * 4 float regs = 64
    float sum[4][4][4] = {0.f};

    // ----------------------------- Prologue: prefetch As/Bs once
    // cp.async load A
    uint32_t smem_a0 =
        static_cast<uint32_t>(__cvta_generic_to_shared(&As[0][load_a_row][SWIZZLE_A(load_a_row, load_a_col)]));
    uint32_t smem_a1 = static_cast<uint32_t>(
        __cvta_generic_to_shared(&As[0][load_a_row + 64][SWIZZLE_A(load_a_row + 64, load_a_col)]));

    T *global_a0 = &a[(by * BM + load_a_row) * k + load_a_col];
    T *global_a1 = &a[(by * BM + load_a_row + 64) * k + load_a_col];

    CP_ASYNC_CG(smem_a0, global_a0);
    CP_ASYNC_CG(smem_a1, global_a1);

    // cp.async load B
    uint32_t smem_b0 =
        static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[0][load_b_row][SWIZZLE_B(load_b_row, load_b_col)]));
    uint32_t smem_b1 = static_cast<uint32_t>(
        __cvta_generic_to_shared(&Bs[0][load_b_row + 16][SWIZZLE_B(load_b_row + 16, load_b_col)]));

    T *global_b0 = &b[(load_b_row)*n + bx * BN + load_b_col];
    T *global_b1 = &b[(load_b_row + 16) * n + bx * BN + load_b_col];

    CP_ASYNC_CG(smem_b0, global_b0);
    CP_ASYNC_CG(smem_b1, global_b1);

    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP_0();
    __syncthreads();

    int read_idx = 0;
    int write_idx = 1;

    // Main K loop
    for (int bk = 32; bk < k; bk += BK) {

        // 1. cp.async load A
        smem_a0 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&As[write_idx][load_a_row][SWIZZLE_A(load_a_row, load_a_col)]));
        smem_a1 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&As[write_idx][load_a_row + 64][SWIZZLE_A(load_a_row + 64, load_a_col)]));

        // Strided global pointers to ease register pressure
        global_a0 += BK;
        global_a1 += BK;

        CP_ASYNC_CG(smem_a0, global_a0);
        CP_ASYNC_CG(smem_a1, global_a1);

        // 2. cp.async load B
        smem_b0 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&Bs[write_idx][load_b_row][SWIZZLE_B(load_b_row, load_b_col)]));
        smem_b1 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&Bs[write_idx][load_b_row + 16][SWIZZLE_B(load_b_row + 16, load_b_col)]));

        global_b0 += BK * n;
        global_b1 += BK * n;

        CP_ASYNC_CG(smem_b0, global_b0);
        CP_ASYNC_CG(smem_b1, global_b1);

        CP_ASYNC_COMMIT_GROUP();

        // 3. Tensor Core: two K steps, 16 elements each
#pragma unroll
        for (int k_step = 0; k_step < 2; ++k_step) {
            int k_offset = k_step * 16;

            uint32_t reg_a[4][4];
            uint32_t reg_b[4][2];

            // Four ldmatrix.issue for A (4 * 16 = 64 rows)
#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
                // ldmatrix x4 loads a 16x16 tile
                int a_row = warp_id_m * 64 + m_idx * 16 + (lane_id % 16);
                int a_col = k_offset + (lane_id / 16) * 8;
                uint32_t smem_addr =
                    static_cast<uint32_t>(__cvta_generic_to_shared(&As[read_idx][a_row][SWIZZLE_A(a_row, a_col)]));
                LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);
            }

            // Four ldmatrix.issue for B (4 * 8 = 32 columns)
#pragma unroll
            for (int n_idx = 0; n_idx < 4; ++n_idx) {
                // Lanes 0-15 load 16 rows (bases of two 8x8 tiles)
                int b_row = k_offset + (lane_id % 16);
                int b_col = warp_id_n * 32 + n_idx * 8;

                uint32_t smem_addr =
                    static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[read_idx][b_row][SWIZZLE_B(b_row, b_col)]));
                LDMATRIX_X2_TRANS(reg_b[n_idx][0], reg_b[n_idx][1], smem_addr);
            }

            // MMA body: 4x4 m16n8k16
#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
                for (int n_idx = 0; n_idx < 4; ++n_idx) {
                    if constexpr (std::is_same_v<T, __half>) {
                        M16N8K16_F16(sum[m_idx][n_idx][0],
                                     sum[m_idx][n_idx][1],
                                     sum[m_idx][n_idx][2],
                                     sum[m_idx][n_idx][3],
                                     reg_a[m_idx][0],
                                     reg_a[m_idx][1],
                                     reg_a[m_idx][2],
                                     reg_a[m_idx][3],
                                     reg_b[n_idx][0],
                                     reg_b[n_idx][1]);
                    } else {
                        M16N8K16_BF16(sum[m_idx][n_idx][0],
                                      sum[m_idx][n_idx][1],
                                      sum[m_idx][n_idx][2],
                                      sum[m_idx][n_idx][3],
                                      reg_a[m_idx][0],
                                      reg_a[m_idx][1],
                                      reg_a[m_idx][2],
                                      reg_a[m_idx][3],
                                      reg_b[n_idx][0],
                                      reg_b[n_idx][1]);
                    }
                }
            }
        }
        CP_ASYNC_WAIT_GROUP_0();
        __syncthreads();
        read_idx ^= 1;
        write_idx ^= 1;
    }
    // ------------------- Epilogue: final MMA, then store C
#pragma unroll
    for (int k_step = 0; k_step < 2; ++k_step) {
        int k_offset = k_step * 16;

        uint32_t reg_a[4][4];
        uint32_t reg_b[4][2];

        // Four ldmatrix.issue for A (4 * 16 = 64 rows)
#pragma unroll
        for (int m_idx = 0; m_idx < 4; ++m_idx) {
            // ldmatrix x4 loads a 16x16 tile
            int a_row = warp_id_m * 64 + m_idx * 16 + (lane_id % 16);
            int a_col = k_offset + (lane_id / 16) * 8;
            uint32_t smem_addr =
                static_cast<uint32_t>(__cvta_generic_to_shared(&As[read_idx][a_row][SWIZZLE_A(a_row, a_col)]));
            LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);
        }

        // Four ldmatrix.issue for B (4 * 8 = 32 columns)
#pragma unroll
        for (int n_idx = 0; n_idx < 4; ++n_idx) {
            // Lanes 0-15 load 16 rows (bases of two 8x8 tiles)
            int b_row = k_offset + (lane_id % 16);
            int b_col = warp_id_n * 32 + n_idx * 8;

            uint32_t smem_addr =
                static_cast<uint32_t>(__cvta_generic_to_shared(&Bs[read_idx][b_row][SWIZZLE_B(b_row, b_col)]));
            LDMATRIX_X2_TRANS(reg_b[n_idx][0], reg_b[n_idx][1], smem_addr);
        }

        // MMA body: 4x4 m16n8k16
#pragma unroll
        for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
            for (int n_idx = 0; n_idx < 4; ++n_idx) {
                if constexpr (std::is_same_v<T, __half>) {
                    M16N8K16_F16(sum[m_idx][n_idx][0],
                                 sum[m_idx][n_idx][1],
                                 sum[m_idx][n_idx][2],
                                 sum[m_idx][n_idx][3],
                                 reg_a[m_idx][0],
                                 reg_a[m_idx][1],
                                 reg_a[m_idx][2],
                                 reg_a[m_idx][3],
                                 reg_b[n_idx][0],
                                 reg_b[n_idx][1]);
                } else {
                    M16N8K16_BF16(sum[m_idx][n_idx][0],
                                  sum[m_idx][n_idx][1],
                                  sum[m_idx][n_idx][2],
                                  sum[m_idx][n_idx][3],
                                  reg_a[m_idx][0],
                                  reg_a[m_idx][1],
                                  reg_a[m_idx][2],
                                  reg_a[m_idx][3],
                                  reg_b[n_idx][0],
                                  reg_b[n_idx][1]);
                }
            }
        }
    }

    // ---------------- Store C ----------------
    int t_row = lane_id / 4;       // 0~7
    int t_col = (lane_id % 4) * 2; // 0, 2, 4, 6

#pragma unroll
    for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
        for (int n_idx = 0; n_idx < 4; ++n_idx) {
            int c_base_row = by * BM + warp_id_m * 64 + m_idx * 16;
            int c_base_col = bx * BN + warp_id_n * 32 + n_idx * 8;
            if constexpr (std::is_same_v<T, __half>) {
                HALF2(c[(c_base_row + t_row) * n + c_base_col + t_col]) =
                    __float22half2_rn(FLOAT2(sum[m_idx][n_idx][0]));
                HALF2(c[(c_base_row + t_row + 8) * n + c_base_col + t_col]) =
                    __float22half2_rn(FLOAT2(sum[m_idx][n_idx][2]));
            } else {
                BFLOAT2(c[(c_base_row + t_row) * n + c_base_col + t_col]) =
                    __float22bfloat162_rn(FLOAT2(sum[m_idx][n_idx][0]));
                BFLOAT2(c[(c_base_row + t_row + 8) * n + c_base_col + t_col]) =
                    __float22bfloat162_rn(FLOAT2(sum[m_idx][n_idx][2]));
            }
        }
    }
}
// a block calculate c[128][128]
template <const int BM = 128, const int BN = 128, const int BK = 32, typename T>
__global__ void hgemm_bcf_dbf_rw_kernel(T *a, T *b, T *c, int m, int n, int k) {
    // grid swizzling
    int linear_id = blockIdx.y * gridDim.x + blockIdx.x;
    const int SWIZZLE_W = 8; // swizzle / logical width = 8

    int bx = (linear_id % SWIZZLE_W) + (linear_id / (SWIZZLE_W * gridDim.y)) * SWIZZLE_W;
    int by = (linear_id / SWIZZLE_W) % gridDim.y;

    int tid = threadIdx.x; // 0~255
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // GMEM->SMEM load indexing
    int load_a_row = tid / 4;        // 0~63
    int load_a_col = (tid % 4) * 8;  // 0,8,16,24
    int load_b_row = tid / 16;       // 0..15 (K)
    int load_b_col = (tid % 16) * 8; // 0,8,...,120 (N)

    // Layout works for both A and B; union aliases one smem block for A/B then C
    __shared__ __align__(128) union {
        // First phase: A and B tiles
        struct {
            T As[2][BM][BK];
            T Bs[2][BK][BN];
        };
        // Second phase: C writeback buffer
        T Cs[BM][BN];
    } smem;

    // warp tiling
    // Each warp owns a 64x32 C tile
    int warp_id_m = warp_id / 4; // 0, 1
    int warp_id_n = warp_id % 4; // 0, 1, 2, 3

    // Accumulators: 4 M-fragments * 4 N-fragments * 4 float regs = 64
    float sum[4][4][4] = {0.f};

    // ----------------------------- Prologue: prefetch As/Bs once
    // cp.async load A
    uint32_t smem_a0 =
        static_cast<uint32_t>(__cvta_generic_to_shared(&smem.As[0][load_a_row][SWIZZLE_A(load_a_row, load_a_col)]));
    uint32_t smem_a1 = static_cast<uint32_t>(
        __cvta_generic_to_shared(&smem.As[0][load_a_row + 64][SWIZZLE_A(load_a_row + 64, load_a_col)]));

    T *global_a0 = &a[(by * BM + load_a_row) * k + load_a_col];
    T *global_a1 = &a[(by * BM + load_a_row + 64) * k + load_a_col];

    CP_ASYNC_CG(smem_a0, global_a0);
    CP_ASYNC_CG(smem_a1, global_a1);

    // cp.async load B
    uint32_t smem_b0 =
        static_cast<uint32_t>(__cvta_generic_to_shared(&smem.Bs[0][load_b_row][SWIZZLE_B(load_b_row, load_b_col)]));
    uint32_t smem_b1 = static_cast<uint32_t>(
        __cvta_generic_to_shared(&smem.Bs[0][load_b_row + 16][SWIZZLE_B(load_b_row + 16, load_b_col)]));

    T *global_b0 = &b[(load_b_row)*n + bx * BN + load_b_col];
    T *global_b1 = &b[(load_b_row + 16) * n + bx * BN + load_b_col];

    CP_ASYNC_CG(smem_b0, global_b0);
    CP_ASYNC_CG(smem_b1, global_b1);

    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP_0();
    __syncthreads();

    int read_idx = 0;
    int write_idx = 1;

    // Main K loop
    for (int bk = 32; bk < k; bk += BK) {

        // 1. cp.async load A
        smem_a0 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&smem.As[write_idx][load_a_row][SWIZZLE_A(load_a_row, load_a_col)]));
        smem_a1 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&smem.As[write_idx][load_a_row + 64][SWIZZLE_A(load_a_row + 64, load_a_col)]));

        // Strided global pointers to ease register pressure
        global_a0 += BK;
        global_a1 += BK;

        CP_ASYNC_CG(smem_a0, global_a0);
        CP_ASYNC_CG(smem_a1, global_a1);

        // 2. cp.async load B
        smem_b0 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&smem.Bs[write_idx][load_b_row][SWIZZLE_B(load_b_row, load_b_col)]));
        smem_b1 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&smem.Bs[write_idx][load_b_row + 16][SWIZZLE_B(load_b_row + 16, load_b_col)]));

        global_b0 += BK * n;
        global_b1 += BK * n;

        CP_ASYNC_CG(smem_b0, global_b0);
        CP_ASYNC_CG(smem_b1, global_b1);

        CP_ASYNC_COMMIT_GROUP();

        // 3. Tensor Core: two K steps, 16 elements each
#pragma unroll
        for (int k_step = 0; k_step < 2; ++k_step) {
            int k_offset = k_step * 16;

            uint32_t reg_a[4][4];
            uint32_t reg_b[4][2];

            // Four ldmatrix.issue for A (4 * 16 = 64 rows)
#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
                // ldmatrix x4 loads a 16x16 tile
                int a_row = warp_id_m * 64 + m_idx * 16 + (lane_id % 16);
                int a_col = k_offset + (lane_id / 16) * 8;
                uint32_t smem_addr =
                    static_cast<uint32_t>(__cvta_generic_to_shared(&smem.As[read_idx][a_row][SWIZZLE_A(a_row, a_col)]));
                LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);
            }

            // Four ldmatrix.issue for B (4 * 8 = 32 columns)
#pragma unroll
            for (int n_idx = 0; n_idx < 4; ++n_idx) {
                // Lanes 0-15 load 16 rows (bases of two 8x8 tiles)
                int b_row = k_offset + (lane_id % 16);
                int b_col = warp_id_n * 32 + n_idx * 8;

                uint32_t smem_addr =
                    static_cast<uint32_t>(__cvta_generic_to_shared(&smem.Bs[read_idx][b_row][SWIZZLE_B(b_row, b_col)]));
                LDMATRIX_X2_TRANS(reg_b[n_idx][0], reg_b[n_idx][1], smem_addr);
            }

            // MMA body: 4x4 m16n8k16
#pragma unroll
            for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
                for (int n_idx = 0; n_idx < 4; ++n_idx) {
                    if constexpr (std::is_same_v<T, __half>) {
                        M16N8K16_F16(sum[m_idx][n_idx][0],
                                     sum[m_idx][n_idx][1],
                                     sum[m_idx][n_idx][2],
                                     sum[m_idx][n_idx][3],
                                     reg_a[m_idx][0],
                                     reg_a[m_idx][1],
                                     reg_a[m_idx][2],
                                     reg_a[m_idx][3],
                                     reg_b[n_idx][0],
                                     reg_b[n_idx][1]);
                    } else {
                        M16N8K16_BF16(sum[m_idx][n_idx][0],
                                      sum[m_idx][n_idx][1],
                                      sum[m_idx][n_idx][2],
                                      sum[m_idx][n_idx][3],
                                      reg_a[m_idx][0],
                                      reg_a[m_idx][1],
                                      reg_a[m_idx][2],
                                      reg_a[m_idx][3],
                                      reg_b[n_idx][0],
                                      reg_b[n_idx][1]);
                    }
                }
            }
        }
        CP_ASYNC_WAIT_GROUP_0();
        __syncthreads();
        read_idx ^= 1;
        write_idx ^= 1;
    }
    // ------------------- Epilogue: final MMA, then store C
#pragma unroll
    for (int k_step = 0; k_step < 2; ++k_step) {
        int k_offset = k_step * 16;

        uint32_t reg_a[4][4];
        uint32_t reg_b[4][2];

        // Four ldmatrix.issue for A (4 * 16 = 64 rows)
#pragma unroll
        for (int m_idx = 0; m_idx < 4; ++m_idx) {
            // ldmatrix x4 loads a 16x16 tile
            int a_row = warp_id_m * 64 + m_idx * 16 + (lane_id % 16);
            int a_col = k_offset + (lane_id / 16) * 8;
            uint32_t smem_addr =
                static_cast<uint32_t>(__cvta_generic_to_shared(&smem.As[read_idx][a_row][SWIZZLE_A(a_row, a_col)]));
            LDMATRIX_X4(reg_a[m_idx][0], reg_a[m_idx][1], reg_a[m_idx][2], reg_a[m_idx][3], smem_addr);
        }

        // Four ldmatrix.issue for B (4 * 8 = 32 columns)
#pragma unroll
        for (int n_idx = 0; n_idx < 4; ++n_idx) {
            // Lanes 0-15 load 16 rows (bases of two 8x8 tiles)
            int b_row = k_offset + (lane_id % 16);
            int b_col = warp_id_n * 32 + n_idx * 8;

            uint32_t smem_addr =
                static_cast<uint32_t>(__cvta_generic_to_shared(&smem.Bs[read_idx][b_row][SWIZZLE_B(b_row, b_col)]));
            LDMATRIX_X2_TRANS(reg_b[n_idx][0], reg_b[n_idx][1], smem_addr);
        }

        // MMA body: 4x4 m16n8k16
#pragma unroll
        for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
            for (int n_idx = 0; n_idx < 4; ++n_idx) {
                if constexpr (std::is_same_v<T, __half>) {
                    M16N8K16_F16(sum[m_idx][n_idx][0],
                                 sum[m_idx][n_idx][1],
                                 sum[m_idx][n_idx][2],
                                 sum[m_idx][n_idx][3],
                                 reg_a[m_idx][0],
                                 reg_a[m_idx][1],
                                 reg_a[m_idx][2],
                                 reg_a[m_idx][3],
                                 reg_b[n_idx][0],
                                 reg_b[n_idx][1]);
                } else {
                    M16N8K16_BF16(sum[m_idx][n_idx][0],
                                  sum[m_idx][n_idx][1],
                                  sum[m_idx][n_idx][2],
                                  sum[m_idx][n_idx][3],
                                  reg_a[m_idx][0],
                                  reg_a[m_idx][1],
                                  reg_a[m_idx][2],
                                  reg_a[m_idx][3],
                                  reg_b[n_idx][0],
                                  reg_b[n_idx][1]);
                }
            }
        }
    }

    // ---------------- Store C ----------------
    // Reuse As/Bs smem as staging for C stores
    __syncthreads();

    int t_row = lane_id / 4;       // 0~7
    int t_col = (lane_id % 4) * 2; // 0, 2, 4, 6

    // register to Cs smem
#pragma unroll
    for (int m_idx = 0; m_idx < 4; ++m_idx) {
#pragma unroll
        for (int n_idx = 0; n_idx < 4; ++n_idx) {
            int c_base_row = warp_id_m * 64 + m_idx * 16; // M: 16-row span
            int c_base_col = warp_id_n * 32 + n_idx * 8;  // N: 8-column span

            // Store 16 rows in two 8-row passes
            int c_row_0 = c_base_row + t_row;
            int c_row_2 = c_base_row + t_row + 8;
            int c_col = c_base_col + t_col;

            if constexpr (std::is_same_v<T, __half>) {
                HALF2(smem.Cs[c_row_0][SWIZZLE_C(c_row_0, c_col)]) = __float22half2_rn(FLOAT2(sum[m_idx][n_idx][0]));
                HALF2(smem.Cs[c_row_2][SWIZZLE_C(c_row_2, c_col)]) = __float22half2_rn(FLOAT2(sum[m_idx][n_idx][2]));
            } else {
                BFLOAT2(smem.Cs[c_row_0][SWIZZLE_C(c_row_0, c_col)]) =
                    __float22bfloat162_rn(FLOAT2(sum[m_idx][n_idx][0]));
                BFLOAT2(smem.Cs[c_row_2][SWIZZLE_C(c_row_2, c_col)]) =
                    __float22bfloat162_rn(FLOAT2(sum[m_idx][n_idx][2]));
            }
        }
    }

    __syncthreads();

    // smem to gmem
    // Each thread moves 64 elems (fp16/bf16) == 8 float4; one warp moves 32*4*4 = 512 B; 256 threads -> 4096 B
    T *c_block = &c[by * BM * n + bx * BN];

#pragma unroll
    for (int step = 0; step < 8; ++step) {
        // Keep elem_idx contiguous within the warp (coalesced)
        int elem_idx = (step * 256 + tid) * 8;
        int row = elem_idx / 128;
        int col = elem_idx % 128;

        int s_col = SWIZZLE_C(row, col);

        FLOAT4(c_block[row * n + col]) = FLOAT4(smem.Cs[row][s_col]);
    }
}

template <int TILE>
__global__ void hgemm_v0_kernel(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    __half* __restrict__ c,
    int m,
    int n,
    int k) {
    __shared__ __half a_tile[TILE][TILE];
    __shared__ __half b_tile[TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;
    float accumulator = 0.0f;

    for (int tile_k = 0; tile_k < k; tile_k += TILE) {
        const int a_col = tile_k + threadIdx.x;
        const int b_row = tile_k + threadIdx.y;
        a_tile[threadIdx.y][threadIdx.x] =
            row < m && a_col < k ? a[row * k + a_col] : __float2half(0.0f);
        b_tile[threadIdx.y][threadIdx.x] =
            b_row < k && col < n ? b[b_row * n + col] : __float2half(0.0f);
        __syncthreads();

#pragma unroll
        for (int inner = 0; inner < TILE; ++inner) {
            accumulator = fmaf(
                __half2float(a_tile[threadIdx.y][inner]),
                __half2float(b_tile[inner][threadIdx.x]),
                accumulator);
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * n + col] = __float2half(accumulator);
    }
}

  // namespace

void hgemm_cublas(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    const GemmShape shape = check_gemm_tensors(a, b, c);
    const c10::cuda::CUDAGuard device_guard(a.device());
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // PyTorch tensors are row-major. Swapping A/B and M/N maps the operation
    // onto cuBLAS's column-major interface without materializing transposes.
    const cublasStatus_t status = cublasGemmEx(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        shape.n,
        shape.m,
        shape.k,
        &alpha,
        b.data_ptr<at::Half>(),
        CUDA_R_16F,
        shape.n,
        a.data_ptr<at::Half>(),
        CUDA_R_16F,
        shape.k,
        &beta,
        c.data_ptr<at::Half>(),
        CUDA_R_16F,
        shape.n,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS,
                "cuBLAS HGEMM failed with status ",
                static_cast<int>(status));
}

void hgemm_v0(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    const GemmShape shape = check_gemm_tensors(a, b, c);
    const c10::cuda::CUDAGuard device_guard(a.device());
    constexpr int TILE = 16;
    const dim3 threads(TILE, TILE);
    const dim3 blocks(
        (shape.n + TILE - 1) / TILE,
        (shape.m + TILE - 1) / TILE);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    hgemm_v0_kernel<TILE><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __half*>(a.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(b.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(c.data_ptr<at::Half>()),
        shape.m,
        shape.n,
        shape.k);
    check_cuda_launch("hgemm_v0_kernel");
}

void hgemm_v1(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    const GemmShape shape = check_gemm_tensors(a, b, c);
    const c10::cuda::CUDAGuard device_guard(a.device());
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 32;
    const dim3 threads(32, 8);
    const dim3 blocks(shape.n / BN, shape.m / BM);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    hgemm_v1_kernel<BM, BN, BK><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<__half*>(a.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(b.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(c.data_ptr<at::Half>()),
        shape.m,
        shape.n,
        shape.k);
    check_cuda_launch("hgemm_v1_kernel");
}
void hgemm_gridswizzle(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    const GemmShape shape = check_gemm_tensors(a, b, c);
    const c10::cuda::CUDAGuard device_guard(a.device());
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 32;
    const dim3 threads(32, 8);
    const dim3 blocks(shape.n / BN, shape.m / BM);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    hgemm_gridswizzle_kernel<BM, BN, BK><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<__half*>(a.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(b.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(c.data_ptr<at::Half>()),
        shape.m,
        shape.n,
        shape.k);
    check_cuda_launch("hgemm_gridswizzle_kernel");
}
void hgemm_gridswizzle_smemswizzle(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    const GemmShape shape = check_gemm_tensors(a, b, c);
    const c10::cuda::CUDAGuard device_guard(a.device());
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 32;
    const dim3 threads(32, 8);
    const dim3 blocks(shape.n / BN, shape.m / BM);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    hgemm_gridswizzle_smemswizzle_kernel<BM, BN, BK><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<__half*>(a.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(b.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(c.data_ptr<at::Half>()),
        shape.m,
        shape.n,
        shape.k);
    check_cuda_launch("hgemm_gridswizzle_smemswizzle_kernel");
}
void hgemm_bcf_dbf(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    const GemmShape shape = check_gemm_tensors(a, b, c);
    const c10::cuda::CUDAGuard device_guard(a.device());
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 32;
    const dim3 threads(256);
    const dim3 blocks(shape.n / BN, shape.m / BM);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    hgemm_bcf_dbf_kernel<BM, BN, BK,__half><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<__half*>(a.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(b.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(c.data_ptr<at::Half>()),
        shape.m,
        shape.n,
        shape.k);
    check_cuda_launch("hgemm_bcf_dbf_kernel");
}
void hgemm_bcf_dbf_rw(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    const GemmShape shape = check_gemm_tensors(a, b, c);
    const c10::cuda::CUDAGuard device_guard(a.device());
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 32;
    const dim3 threads(256);
    const dim3 blocks(shape.n / BN, shape.m / BM);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    hgemm_bcf_dbf_rw_kernel<BM, BN, BK,__half><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<__half*>(a.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(b.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(c.data_ptr<at::Half>()),
        shape.m,
        shape.n,
        shape.k);
    check_cuda_launch("hgemm_bcf_dbf_rw_kernel");
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("hgemm_cublas", &hgemm_cublas, "cuBLAS FP16 GEMM baseline");
    module.def("hgemm_v0", &hgemm_v0, "tiled FP16 GEMM starter kernel");
    module.def("hgemm_v1", &hgemm_v1, "Tensor Core FP16 GEMM kernel");
    module.def("hgemm_gridswizzle", &hgemm_gridswizzle, "Grid Swizzle FP16 GEMM kernel");
    module.def("hgemm_gridswizzle_smemswizzle", &hgemm_gridswizzle_smemswizzle, "Grid Swizzle with Shared Memory Swizzle FP16 GEMM kernel");
    module.def("hgemm_bcf_dbf", &hgemm_bcf_dbf, "BCF DBF FP16 GEMM kernel");
    module.def("hgemm_bcf_dbf_rw", &hgemm_bcf_dbf_rw, "BCF DBF RW FP16 GEMM kernel");
}
