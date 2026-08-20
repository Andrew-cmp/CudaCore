#include <cuda_runtime.h>
//对sharemem再次进行分块，将sharemem分块到register中。
const int THREAD_SIZE_M=1;//每个线程计算的C中元素的高度
const int THREAD_SIZE_N=1;//每个线程计算的C中元素的宽度
const int BLOCK_SIZE_M=32; ///每个线程块需要处理的M维度数据块大小
const int BLOCK_SIZE_N=32; ///每个线程块需要处理的N维度数据块大小
const int BLOCK_SIZE_K=32;  //每个线程块需要A load into sharemen的宽度
//每个线程块的所包含的线程数量
const int THREAD_SIZE_PER_BLCOK_M=BLOCK_SIZE_M/THREAD_SIZE_M; 
const int THREAD_SIZE_PER_BLCOK_N=BLOCK_SIZE_N/THREAD_SIZE_N;

///对M、N、K都进行tile，但必须要求BLOCK_SIZE_m\n\K都一致。

__global__ void gemm_v1(int M, int N, int K, float alpha,float* A,float* B, float beta,float* C){
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float sum  = 0;
    __shared__ float A_shared[BLOCK_SIZE_M][BLOCK_SIZE_K];
    __shared__ float B_shared[BLOCK_SIZE_K][BLOCK_SIZE_N];

    const int numTiles = (K + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K;
    for(int num_k = 0;num_k < numTiles;num_k++){
        if(num_k*BLOCK_SIZE_K+tx <K && y < M)
            A_shared[ty][tx] = A[y*K+num_k*BLOCK_SIZE_K+tx];
        else
            A_shared[ty][tx] = 0;
        if(num_k*BLOCK_SIZE_K+ty < K && x<N)
            B_shared[ty][tx] = B[(num_k*BLOCK_SIZE_K+ty)*N + x];
        else
            B_shared[ty][tx] = 0;
        __syncthreads();
        for(int i = 0;i < BLOCK_SIZE_K;i++){
            sum += A_shared[ty][i]*B_shared[i][tx];
        }
        __syncthreads();
    }
    if(y < M && x < N)
        C[y*N+x] = sum;

}

void launch_gemm_v1(
    int M, int N, int K, float* A, float* B, float* C,
    cudaStream_t stream) {
    const dim3 block(BLOCK_SIZE_N, BLOCK_SIZE_M);
    const dim3 grid((N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N,
                    (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);
    gemm_v1<<<grid, block, 0, stream>>>(M, N, K, 1.0f, A, B, 0.0f, C);
}
