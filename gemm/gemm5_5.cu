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
  // [6]   double buffer 版本

 
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
    int load_smem_a_m = tid /2;
    int load_smem_a_k = (tid%2) ? 0:4;
    int load_smem_b_k = tid/32;
    int load_smem_b_n = (tid%32)*4;
    int load_gmem_a_m = by*BM + load_smem_a_m;// a矩阵和c矩阵的行号
    int load_gmem_b_n = bx*BN = load_smem_b_n; // b矩阵和c矩阵的列号

    float r_load_a[TM/2]; // 4
    float r_load_b[TN/2]; // 4
    float r_comp_a[TM];
    float r_comp_b[TN];
    float r_c[TM][TN] = {0.0};


  // 1）主循环从bk = 1 开始，第一次数据加载在主循环之前，最后一次计算在主循环之后，这是pipeline 的特点决定的；
  // 2）由于计算和下一次访存使用的Shared Memory不同，因此主循环中每次循环只需要一次__syncthreads()即可
  // 3）由于GPU不能向CPU那样支持乱序执行，主循环中需要先将下一次循环计算需要的Gloabal Memory中的数据load 
  // 到寄存器，然后进行本次计算，之后再将load到寄存器中的数据写到Shared Memory，这样在LDG指令向Global 
  // Memory做load时，不会影响后续FFMA及其它运算指令的 launch 执行，也就达到了Double Buffering的目的。
  
  // bk = 0 is loading here, buffer 0
    {
        
        //将第1步的数据从gmem移动进入register
        int load_a_gmem_k = load_a_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        int load_b_gmem_k = load_b_smem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
        FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
        FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);
    
        //将第1步的数据从register移动进入gmem
        s_a[0][load_a_smem_k + 0][load_a_smem_m] = r_load_a[0];
        s_a[0][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        s_a[0][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        s_a[0][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
        FLOAT4(s_b[0][load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);
      }
      // Without this synchronization, accuracy may occasionally be abnormal.
      __syncthreads(); 
    
      // bk start from 1，需要注意的是，虽然 bk 从 1 开始，但实际上 bk=1时，使用的是
      // 第0块BK中的数据（已经加载到共享内存s_a[0]和s_b[0]）；也就是说执行的是bk0的计算
      //bk=2时，实际计算的是第1块BK中的数据。其余以此类推，这个循环结束后，剩下最后一块BK大小的数据需要计算。
      for (int bk = 1; bk < (K + BK - 1) / BK; bk++) {
    
        int smem_sel = (bk - 1) & 1;
        int smem_sel_next = bk & 1;
    
        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
        //将第bk步的数据从gmem移动进入register
        FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
        FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);
    

        //执行bk-1步的计算
        #pragma unroll
        for (int tk = 0; tk < BK; tk++) {
          FLOAT4(r_comp_a[0]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2     ]);
          FLOAT4(r_comp_a[4]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2 + BM / 2]);
          FLOAT4(r_comp_b[0]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2     ]);
          FLOAT4(r_comp_b[4]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2 + BN / 2]);
    
          #pragma unroll
          for (int tm = 0; tm < TM; tm++) {
            #pragma unroll
            for (int tn = 0; tn < TN; tn++) {
              // r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
              r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
            }
          }
        }
        
        //将第bk步的数据从gmem移动进入register

        // 对比非double buffers版本，此处不需要__syncthreads()，总共节省了
        // ((K + BK - 1) / BK) - 1 次block内的同步操作。比如，bk=1时，HFMA计算
        // 使用的是s_a[0]和s_b[0]，因此，和s_a[1]和s_b[1]的加载是没有依赖关系的。
        // 从global内存到s_a[1]和s_b[1]和HFMA计算可以并行。s_a[1]和s_b[1]用于
        // 加载下一块BK需要的数据到共享内存。
        s_a[smem_sel_next][load_a_smem_k + 0][load_a_smem_m] = r_load_a[0];
        s_a[smem_sel_next][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        s_a[smem_sel_next][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        s_a[smem_sel_next][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
        FLOAT4(s_b[smem_sel_next][load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);
    
        __syncthreads();
      }
      
      // 计算剩下最后一块BK
      #pragma unroll
      for (int tk = 0; tk < BK; tk++) {
        FLOAT4(r_comp_a[0]) = FLOAT4(s_a[1][tk][ty * TM / 2     ]);
        FLOAT4(r_comp_a[4]) = FLOAT4(s_a[1][tk][ty * TM / 2 + BM / 2]);
        FLOAT4(r_comp_b[0]) = FLOAT4(s_b[1][tk][tx * TN / 2     ]);
        FLOAT4(r_comp_b[4]) = FLOAT4(s_b[1][tk][tx * TN / 2 + BN / 2]);
    
        #pragma unroll
        for (int tm = 0; tm < TM; tm++) {
          #pragma unroll
          for (int tn = 0; tn < TN; tn++) {
            // r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
            r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
          }
        }
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