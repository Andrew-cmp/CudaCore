#include <cuda_runtime.h>
#define OFFSET(row, col, lgd) ((row)*(lgd)+(col))
#define FETCH(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
//对sharemem再次进行分块，将sharemem分块到register中。
const int THREAD_SIZE_M=8;//每个线程计算的C中元素的高度
const int THREAD_SIZE_N=8;//每个线程计算的C中元素的宽度
const int BLOCK_SIZE_M=64; ///每个线程块需要处理的M维度数据块大小
const int BLOCK_SIZE_N=64; ///每个线程块需要处理的N维度数据块大小
const int BLOCK_SIZE_K=8;  //每个线程块需要A load into sharemen的宽度
//每个线程块的所包含的线程数量
const int THREAD_SIZE_PER_BLCOK_M=BLOCK_SIZE_M/THREAD_SIZE_M;   //8
const int THREAD_SIZE_PER_BLCOK_N=BLOCK_SIZE_N/THREAD_SIZE_N;   //8
const int THREAD_NUM_PER_BLOCK = THREAD_SIZE_PER_BLCOK_M*THREAD_SIZE_PER_BLCOK_N;

__device__ __forceinline__ float4 load4(const float* p) {
  // 假设 p 至少 16-byte 对齐；否则有些架构会慢或潜在异常
  return *reinterpret_cast<const float4*>(p);
}

//仅对M、K进行tile，这种情况下K不能太大，否则sharedmem不够用
__global__ void gemm_v3(int M, int N, int K, float alpha,float* A,float* B, float beta,float* C){
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    // 线性线程索引，
    const int tid = ty * blockDim.x + tx;     // 0..(blockDim.x*blockDim.y-1)
    

    __shared__ float As[BLOCK_SIZE_K*BLOCK_SIZE_M];
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
        int row = load_smem_a_m + i * load_threeads_a_m;   // 0..63
        int col = load_smem_a_k;                           // 0 or 4 (within BK)
        // 安全读取4个连续元素
        float4 va = load4(&A[row * K + col]);              // A is already block-offset, K is global ld
        // 将以上的一行的4个元素，按列排布在共享显存中
        As[OFFSET(col    ,row , BLOCK_SIZE_M)] = va.x;
        As[OFFSET(col + 1,row , BLOCK_SIZE_M)] = va.y;
        As[OFFSET(col + 2,row , BLOCK_SIZE_M)] = va.z;
        As[OFFSET(col + 3,row , BLOCK_SIZE_M)] = va.w;
    }
    // 2) 装载 B 子块到共享内存 Bs[BK][BN]，采用跨 stride 方式覆盖 0..BK-1
    for(int i = 0;i < num_k_strides_b;i++){
        // k in [0, BK), n in [0, BN)
        int k = load_smem_b_k + i * load_threads_b_k;  // 0..7
        int n = load_smem_b_n;                         // 0,4,8,...,60

        float4 vb = load4(&B[k * N + n]);              // B is already block-offset, N is global ld

        // Bs is BK x BN, ld = BN
        int base = k * BLOCK_SIZE_N + n;
        Bs[base + 0] = vb.x;
        Bs[base + 1] = vb.y;
        Bs[base + 2] = vb.z;
        Bs[base + 3] = vb.w;

    }

    __syncthreads();
    
    A += BLOCK_SIZE_K;
    B += BLOCK_SIZE_K * N;

    #pragma unroll
    for(int kk = 0;kk < BLOCK_SIZE_K;kk++){
        // 从共享内存按每线程子块装入寄存器（标量方式，避免未对齐）
        #pragma unroll
        for(int i = 0;i < THREAD_SIZE_M; ++i){
              a_frag[i] = As[OFFSET(kk, comp_m + i, BLOCK_SIZE_M)];

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
      int idx = OFFSET(comp_m+i, comp_n+j, N);
      C[idx] = alpha * sum[i][j] + beta * C[idx];
      
    }
  }
}

void launch_gemm_v3(
    int M, int N, int K, float* A, float* B, float* C,
    cudaStream_t stream) {
    const dim3 block(THREAD_SIZE_PER_BLCOK_N, THREAD_SIZE_PER_BLCOK_M);
    const dim3 grid((N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N,
                    (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);
    gemm_v3<<<grid, block, 0, stream>>>(M, N, K, 1.0f, A, B, 0.0f, C);
}
