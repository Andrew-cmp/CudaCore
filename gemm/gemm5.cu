#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

//参考CUDA_Learn_note
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

#define OFFSET 4 为防止bank confict所作的pad位数
template<uint32_t M,uint32_t N,uint32_t K>
__global__ void gemm_kernel(ELE_TYPE* A, ELE_TYPE* B,ELE_TYPE* C){
  // [1]  Block Tile: 一个16x16的block处理C上大小为128X128的一个目标块
  // [2] Thread Tile: 每个thread负责计算TM*TN(8*8)个元素，增加计算密度
  // [3]      K Tile: 将K分块，每块BK大小，迭代(K+BK-1/BK)次，
  //                  每次计算TM*TN个元素各自的部分乘累加
  // [4]   Vectorize: 减少load和store指令，使用float4
  // [5]   bank confict advoid: 减少访问smem的访问bank冲突，//在上一个版本里，当从sharemem移动数据到register中时，是“竖着移动的”，也就是说同一个线程访问了bank里的不同层，会导致读取事件的序列化。
  // 此处使用的是float4版本的解决conlict的版本。
 //为什么会产生bank confict，bank confict只会产生在一个warp中，如果一个warp内的32个线程访问不同bank，性能最佳，否则就会产生bank confict；
 //如果每个thread都按索引只读取或存储一个float32，即那永远不会产生bank confict 

 
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx; // tid within the block

    // row指的是行上的坐标，而不是指的是第几行
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    // col指的是列上的坐标，而不是第几列
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    //以下访问会出现bank confict
    // __shared__ ELE_TYPE Sa[BLOCK_SIZE_M][BLOCK_SIZE_K];
    // __shared__ ELE_TYPE Sb[BLOCK_SIZE_K][BLOCK_SIZE_N];
    __shared__ ELE_TYPE Sa[BLOCK_SIZE_K][BLOCK_SIZE_M + OFFSET];
    __shared__ ELE_TYPE Sb[BLOCK_SIZE_K][BLOCK_SIZE_N + OFFSET];

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

    //暂存float4数据，因为在globalmem中是按行读取，写入sharemem的时候是按列写入，肯定会发生bank  conflict。
    //因此先从gmem中存入到register中，然后存入sharemem
    float r_load_a[TM/2]; // 4
    float r_load_b[TN/2]; // 4
    float r_comp_a[TM];
    float r_comp_b[TN];
    float r_c[TM][TN] = {0.0};

    // 2. 先对K进行分块，每块BK大小
    for(int bk = 0;bk<(K+BK-1)/BK;++bk){
        //a矩阵的列号
        int load_a_gmem_k= bk*BK + load_smem_a_k;
        //a矩阵要加载数据的地址=行号乘行宽+列号        
        int load_a_addr =  load_gmem_a_m * K + load_gmem_a_k;

        //b矩阵的行号
        int load_gmem_b_k = bk*BK + load_smem_b_k;
        //b矩阵要加载数据的地址=行号乘行宽+列号        
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n; 
            //先存入register做中转，然后送入sharemem
        FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
        FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);

        //按列送入sharemem
        s_a[load_a_smem_k    ][load_a_smem_m] = r_load_a[0]; // e.g layer_0  b0
        s_a[load_a_smem_k + 1][load_a_smem_m] = r_load_a[1]; // e.g layer_4  b0
        s_a[load_a_smem_k + 2][load_a_smem_m] = r_load_a[2]; // e.g layer_8  b0
        s_a[load_a_smem_k + 3][load_a_smem_m] = r_load_a[3]; // e.g layer_12 b0

        //b不需要转置，b是按行。
        FLOAT4(s_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);

        __syncthreads();
        #pragma unroll
        for(int k = 0;k < BK;k++){
            ///看不懂了
            // bank conflicts analysis, tx/ty 0~15, 0~7 bank 4*8=32 bytes
            // tid 0~15 access bank 0~3,  tid 16~31 access bank 4~7, etc.
            // tid 0,  tk 0 -> ty 0 -> [0][0+0~3],[0][64+0~3] -> bank 0~3(layer_0/2),   
            // tid 0,  tk 7 -> ty 0 -> [7][0+0~3],[0][64+0~3] -> bank 0~3(layer_28/30), 
            // tid 15, tk 0 -> ty 0 -> [0][0+0~3],[0][64+0~3] -> bank 0~3(layer_0/2),   
            // tid 15, tk 7 -> ty 0 -> [7][0+0~3],[0][64+0~3] -> bank 0~3(layer_28/30), 
            // tid 16, tk 0 -> ty 1 -> [0][0+4~7],[0][64+4~7] -> bank 4~7(layer_0/2),   
            // tid 16, tk 7 -> ty 1 -> [7][0+4~7],[0][64+4~7] -> bank 4~7(layer_28/30), 
            // tid 31, tk 0 -> ty 1 -> [0][0+4~7],[0][64+4~7] -> bank 4~7(layer_0/2),   
            // tid 31, tk 7 -> ty 1 -> [7][0+4~7],[0][64+4~7] -> bank 4~7(layer_28/30), 
            // tid 255,tk 0 -> ty 15 -> [0][0+60~63],[0][64+60~63] -> bank 28~31(layer_1/3),   
            // tid 255,tk 7 -> ty 15 -> [7][0+60~63],[0][64+60~63] -> bank 28~31(layer_29/31), 
            FLOAT4(r_comp_a[0]) = FLOAT4(s_a[tk][ty * TM / 2         ]);
            FLOAT4(r_comp_a[4]) = FLOAT4(s_a[tk][ty * TM / 2 + BM / 2]);
            // if (tid == < 32 && bx == 0 && by == 0) {
            //   printf("tid: %d, tx: %d, ty: %d, [%d][%d]\n", tid, tx, ty, tk, ty * TM / 2);
            //   printf("tid: %d, tx: %d, ty: %d, [%d][%d]\n", tid, tx, ty, tk, ty * TM / 2 + BM / 2);
            // }
            // conclusion: still have bank conflicts, need 16 memory issues ?

            // tid 0/8/16/24  access bank 0~3,  tid 1/9/17/25  access bank 4~7, 
            // tid 2/10/18/26 access bank 8~11, tid 7/15/23/31 access bank 28~31, etc.
            // tid 0, tk 0 -> tx 0 -> [0][0+0~3],[0][64+0~3] -> bank 0~3(layer_0/2),    
            // tid 0, tk 7 -> tx 0 -> [7][0+0~3],[0][64+0~3] -> bank 0~3(layer_28/30), 
            // tid 1, tk 0 -> tx 1 -> [0][0+4~7],[0][64+4~7] -> bank 4~7(layer_0/2),    
            // tid 1, tk 7 -> tx 1 -> [7][0+4~7],[0][64+4~7] -> bank 4~7(layer_28/30), 
            FLOAT4(r_comp_b[0]) = FLOAT4(s_b[tk][tx * TN / 2         ]);
            FLOAT4(r_comp_b[4]) = FLOAT4(s_b[tk][tx * TN / 2 + BN / 2]);
            // conclusion: still have some bank conflicts, need 4 memory issues.


            #pragma unroll
            for (int tm = 0; tm < TM; tm++) {
              #pragma unroll
              for (int tn = 0; tn < TN; tn++) {
                // r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
                r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
              }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < TM / 2; i++) {
      int store_c_gmem_m = by * BM + ty * TM / 2 + i;
      int store_c_gmem_n = bx * BN + tx * TN / 2;
      int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
      FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);
      FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i][4]);
    }
    #pragma unroll
    for (int i = 0; i < TM / 2; i++) {
      int store_c_gmem_m = by * BM + BM / 2 + ty * TM / 2 + i;
      int store_c_gmem_n = bx * BN + tx * TN / 2;
      int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
      FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i + TM / 2][0]);
      FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i + TM / 2][4]);
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