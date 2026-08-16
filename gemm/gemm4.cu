#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
//参考 CUDA_Learn_note 
#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])



// SGEMM: Block Tile + Thread Tile + K Tile + Vec4, with smem
// BK:TILE_K=8 BM=BN=128
// TM=TN=8 增加计算密度 BM/TM=16 BN/TN=16
// dim3 blockDim(BN/TN, BM/TM);
// dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM)

#define ELE_TYPE float
//对sharemem再次进行分块，将sharemem分块到register中。
#define THREAD_SIZE_M 8//每个线程计算的C中元素的高度
#define THREAD_SIZE_N 8//每个线程计算的C中元素的宽度
#define BLOCK_SIZE_M 128 ///每个线程块需要处理的M维度数据块大小
#define BLOCK_SIZE_N 128 ///每个线程块需要处理的N维度数据块大小
#define BLOCK_SIZE_K 8  //每个线程块需要A load into sharemen的宽度
    // 每个线程块的所包含的线程数量
#define THREAD_SIZE_PER_BLCOK_M BLOCK_SIZE_M/THREAD_SIZE_M  
#define THREAD_SIZE_PER_BLCOK_N BLOCK_SIZE_N/THREAD_SIZE_N
template<uint32_t M,uint32_t N,uint32_t K>
__global__ void gemm_kernel(ELE_TYPE* A, ELE_TYPE* B,ELE_TYPE* C){
  // [1]  Block Tile: 一个16x16的block处理C上大小为128X128的一个目标块
  // [2] Thread Tile: 每个thread负责计算TM*TN(8*8)个元素，增加计算密度
  // [3]      K Tile: 将K分块，每块BK大小，迭代(K+BK-1/BK)次，
  //                  每次计算TM*TN个元素各自的部分乘累加
  // [4]   Vectorize: 减少load和store指令，使用float4
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx; // tid within the block

    // row指的是行上的坐标，而不是指的是第几行
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    // col指的是列上的坐标，而不是第几列
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    __shared__ ELE_TYPE Sa[BLOCK_SIZE_M][BLOCK_SIZE_K];
    __shared__ ELE_TYPE Sb[BLOCK_SIZE_K][BLOCK_SIZE_N];

    //想清楚一个问题，就是一个线程不一定取自己计算的数据。也就是说取数据和计算可以是两种完全不同的索引计算方式。
    ///下面就是抛弃计算过程，先计算存取smem和gmem的索引
    //0.先计算shared mem中的索引
    //tid先后需要加载的Smen Sa[BM][BN]之间的索引关系 BM=128， BK=8 按行读取
    //对于Sa的8个数据， 每个线程读取4个，需要2个线程；总共128行，需要128x2刚好256线程
    int load_smem_a_m = tid /2;
    int load_smem_a_k = (tid%2) ? 0:4;
    // tid和需要加载的smem s_b[BK][BN] 之间的索引关系 BK=8 BN=128 按行读取 B行主序
    // 对于s_b每行128个数据，每个线程读4个数据，需要32个线程；总共8行，需要32x8=256个线程
    int load_smem_b_k = tid/32;
    int load_smem_b_n = (tid%32)*4;

    // 1. 再计算全局内存中的索引
    // 要加载到s_a中的元素对应到A全局内存中的行数 每个block负责出C中大小为BM*BN的块
    //by*BM:上面部分block共取走了by*BM个，load_smem_a_m即在块内的相对地址。
    int load_gmem_a_m = by*BM + load_smem_a_m;// a矩阵和c矩阵的行号
    int load_gmem_b_n = bx*BN = load_smem_b_n; // b矩阵和c矩阵的列号
    
    float r_c[TM][TN] = {0.0};

    // 2. 先对K进行分块，每块BK大小
    for(int bk = 0;bk<(K+BK-1)/BK;++bk){
        // 加载数据到共享内存smem s_a BM*BK 128*8 vectorize float4
        //a矩阵的列号
        int load_a_gmem_k= bk*BK + load_smem_a_k;
        //a矩阵要加载数据的地址=行号乘行宽+列号
        int load_a_addr =  load_gmem_a_m * K + load_gmem_a_k;
        FLOAT4(Sa[load_smem_a_m][load_smem_a_k])=FLOAT4(load_a_addr)；

        // 加载数据到共享内存smem s_b BK*BN 8*128 vectorize float4
        //b矩阵的行号
        int load_gmem_b_k = bk*BK + load_smem_b_k;
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n; 
        FLOAT4(s_b[load_smem_b_k][load_smem_b_n]) = FLOAT4(b[load_gmem_b_addr]); 
        __syncthreads();
        #pragma unroll
        for(int k = 0;k < BK;k++){
            #pragma unroll
            for(int m = 0;m < TM;m++){
                #pragma unroll
                for(int n = 0;n < TN;n++){
                    // k from 0~7，0 ~ BK, ty and tx range from 0 to 15, 16x8=128
                    int comp_smem_a_m = ty * TM + m;  // 128*8 128/TM(8)=16 M方向 16线程
                    int comp_smem_b_n = tx * TN + n;  // 8*128 128/TN(8)=16 N方向 16线程
                    r_c[m][n] += s_a[comp_smem_a_m][k] * s_b[k][comp_smem_b_n];
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
      int store_gmem_c_m = by * BM + ty * TM + m;
      #pragma unroll
      for (int n = 0; n < TN; n += 4) {
        int store_gmem_c_n = bx * BN + tx * TN + n;
        int store_gmem_c_addr = store_gmem_c_m * N + store_gmem_c_n;
        FLOAT4(c[store_gmem_c_addr]) = FLOAT4(r_c[m][n]);
      }
    }
}

int main(){

    ///这个不加还不能当cuda模板参数
    ///要求必须是编译时期已知且运行时不会变的constant
    constexpr uint32_t N = 512;
    constexpr uint32_t M = 1024;
    constexpr uint32_t K = 8;
    int size_A = M*K*sizeof(ELE_TYPE);
    int size_B = K*N*sizeof(ELE_TYPE);
    int size_C = M*N*sizeof(ELE_TYPE);
    ELE_TYPE * h_a =(ELE_TYPE*)malloc(size_A);
    ELE_TYPE * h_b =(ELE_TYPE*)malloc(size_B);
    ELE_TYPE * h_c =(ELE_TYPE*)malloc(size_C);
    for (int i = 0; i < M * K; ++i) h_a[i] = 2.0f;
    for (int i = 0; i < K * N; ++i) h_b[i] = 2.0f;
    ELE_TYPE *d_a, *d_b, *d_c;
    cudaMalloc(&d_a,M*K*sizeof(ELE_TYPE));
    cudaMalloc(&d_b,K*N*sizeof(ELE_TYPE));
    cudaMalloc(&d_c,M*N*sizeof(ELE_TYPE));
    
    cudaMemcpy(d_a,h_a,size_A,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,size_B,cudaMemcpyHostToDevice);
    dim3 blockDim(BLOCK_SIZE_N,BLOCK_SIZE_M);
    dim3 gridDim((N+blockDim.x-1)/blockDim.x,
                      (M+blockDim.y-1)/blockDim.y );
    //草，大模型给的代码，下面的GridDim和blockDim位置对调了
    //gemm_kernel<M,N,K><<<blockDim,gridDim>>>(d_a,d_b,d_c);
    gemm_kernel<M,N,K><<<gridDim,blockDim>>>(d_a,d_b,d_c);

    cudaMemcpy(h_c,d_c,size_C,cudaMemcpyDeviceToHost);

    for(int i = 0;i < M;++i){
        for(int j = 0;j < N;j++){
            printf("%0.1f ",*(h_c+i*N+j));
        }
        printf("\n");
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    
}