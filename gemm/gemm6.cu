#include <stdio.h>
#include <stdlib.h>
#include "assert.h" 

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>
//https://zhuanlan.zhihu.com/p/442930482
//row,col 坐标 ld：长方体的长
#define OFFSET(row,col,ld) ((row)*(ld)+(col))
#define FETCH_FLOAT4(pointer) （reinterpret_cast<float4*>(&(pointer)[0])

template<
    const int BLOCK_SIZE_M,         //每个线程块需要计算的C矩阵的高度
    const int BLOCK_SIZE_K,         //每个线程块需要A load into sharemen的宽度
    const int BLOCK_SIZE_N,         //每个线程块需要计算的C矩阵的宽度
    const int THREAD_SIZE_Y,        //每个线程计算的C中元素的高度
    const int THREAD_SIZE_X,        //每个线程计算的C中元素的宽度
    const bool ENABLE_DOUBLE_BUFFER
>
__global__ void Sgemm(
    float * __restrict__ A,
    float * __restrict__ B,
    float * __restrict__ C,
    const int M,
    const int N,
    const int K
)
{
    //线程所在线程块的坐标。bx代表横向的block坐标，by代表竖向的block坐标。
    int bx = blockIdx.x;
    int by = blockIdx.y;
    //线程在线程块中的坐标。tx代表横向的线程坐标，ty代表竖向的线程坐标。
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 每个线程块的所包含的线程数量
    const int THREAD_PER_BLOCK_X = BLOCK_SIZE_N / THREAD_SIZE_X;
    const int THREAD_PER_BLOCK_Y = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int THREAD_NUM_PER_BLOCK = THREAD_X_PER_BLOCK * THREAD_Y_PER_BLOCK;


    //线程在线程块内的编号
    const int tid = ty*THREAD_PER_BLOCK_X + tx;

    __shared__ float As[2][BLOCK_SIZE_K][BLOCK_SIZE_M];
    __shared__ float Bs[2][BLOCK_SIZE_K][BLOCK_SIZE_N];

    float accm[THREAD_SIZE_Y][THREAD_SIZE_X];
    
    #pragma unroll
    for(int i=0; i<THREAD_SIZE_Y; i++){
        #pragma unroll
        for(int j=0; j<THREAD_SIZE_X; j++){
            accum[i][j]=0.0;
        }
    }
    //register分组后，从bm*bk里和bk*bn里每次取出两行数据。两行数据的大小由每个线程处理的元素数量决定。即THREAD_SIZE
    float frag_a[2][THREAD_SIZE_X];
    float frag_b[2][THREAD_SIZE_Y];
    //每个线程需要从global mem搬运到register mem，需要搬运的次数，*4就是搬运的数据量
    const int ldg_num_a = BLOCK_SIZE_M * BLOCK_SIZE_N/(THREAD_NUM_PER_BLOCK * 4);
    const int ldg_num_b = BLOCK_SIZE_N * BLOCK_SIZE_K/(THREAD_NUM_PER_BLOCK * 4);
    //暂时存储
    float ldg_a_reg[4*ldg_num_a];
    float ldg_b_reg[4*ldg_num_b];


    //使用float4取值指令后，将A、B从global_memory读取到sharemen需要的线程数量
    //读A时每次读取BLOCK_SIZE_K*1一行大小的，因此需要如此大小
    const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
    //读B时每次读取BLOCK_SIZE_N*1一行大小的，因此需要如此大小
    const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 4;
    
    //读A时，每个线程所读数据块的竖向坐标；
    const int A_TILE_ROW_START = tid / A_TILE_THREAD_PER_ROW;
    //读A时，每个线程所读数据块的横向坐标，因为每次读4个，因此需要乘以4
    const int A_TILE_COL = tid % A_TILE_THREAD_PER_ROW * 4;

    //B同理
    const int B_TILE_ROW_START = tid / B_TILE_THREAD_PER_ROW;
    const int B_TILE_COL = tid % B_TILE_THREAD_PER_ROW * 4;

    // //没懂
    // const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW;
    // const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / B_TILE_THREAD_PER_ROW;

    //每一行需要BLOCK_SIZE_K/4个线程搬运，一共有BLOCK_SIZE_M行，那么就需要BLOCK_SIZE_K/4*BLOCK_SIZE_M个线程搬运
    //但目前仅有THREAD_NUM_PER_BLOCK个线程，因此每个线程需要多次搬运，搬运次数为BLOCK_SIZE_K/4*BLOCK_SIZE_M/THREAD_NUM_PER_BLOCK
    //因为是按行搬运，因此竖着的就被分为了 搬运次数 个块，因此stride=BLOCK_SIZE_M/搬运次数
    //以下表达式和const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / A_TILE_THREAD_PER_ROW等价
    const int A_TILE_ROW_STRIDE = BLOCK_SIZE_M/(BLOCK_SIZE_K/4*BLOCK_SIZE_M/THREAD_NUM_PER_BLOCK);
    //B的分析同理
    const int B_TILE_ROW_START_STRIDE = BLOCK_SIZE_K/(BLOCK_SIZE_N/4*BLOCK_SIZE_K/THREAD_NUM_PER_BLOCK);


    //只取出和本block有关联的A、B数组大小，括号内的是本block所需元素的首地址大小  
    //注意乘K，BLOCK_SIZE_M*by是第几行，其首地址大小是第几行乘行的大小     
    A = &A[BLOCK_SIZE_M*by*K];
    B = &B[BLOCK_SIZE_N*bx]
    
// 将第一个大迭代的数据从global mem读取到share mem
// 按行读取，所以要按列搬运。
    #pragma unroll
    for(int i = 0;i < BLOCK_SIZE_K;i += ){
        //这个过程也需要经过寄存器，但没必要了
        FETCH_FLOAT4(Bs[0][B_TILE_ROW_START+i][B_TILE_COL])=FETCH_FLOAT4(B[OFFSET(
            B_TILE_ROW_START + i, // row
            B_TILE_COL,
            N
        )])
    }
    __syncthreads();
    ///寄存器ldg_index这一块不太懂，这里的寄存器和tid没有关系，说明block所有的线程都是使用的这几个寄存器，不会产生依赖吗
    #pragma unroll
    for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
        int ldg_index = i / A_TILE_ROW_STRIDE * 4;
        FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(
            BLOCK_SIZE_M * by + A_TILE_ROW_START + i, // row
            A_TILE_COL, // col
            K )]);
        As[0][A_TILE_COL][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index];
        As[0][A_TILE_COL+1][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+1];
        As[0][A_TILE_COL+2][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+2];
        As[0][A_TILE_COL+3][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+3];
    }

//将第一个大迭代中第一个小迭代所需要的数据shared memory中的数据存到寄存器中。
//THREAD_SIZE_Y即rm，每个线程负责的是计算C的rm*rn个数据，对应的A的大小为rm*bk、B的大小为bk*rn。
//每次单个线程仅在As中拿出rm*1个数，在Bs中拿出1*rn个数进行计算，并且以4个数据为单位拿出。
//因此这里一共需要从As中取THREAD_SIZE_Y个数，每次取4个数
//关于As[0][0][THREAD_SIZE_Y * ty + thread_y]，其实就是As[THREAD_SIZE_Y * ty + thread_y]
//一个sharemem整个blcok用，每个block分的大小是THREAD_SIZE_X*THREAD_SIZE_Y
//As的stride=THREAD_SIZE_Y
    #pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
        FETCH_FLOAT4(frag_a[0][thread_y]) = FETCH_FLOAT4(As[0][0][THREAD_SIZE_Y * ty + thread_y]);
    }
    // load B from shared memory to register
    #pragma unroll
    for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
        FETCH_FLOAT4(frag_b[0][thread_x]) = FETCH_FLOAT4(Bs[0][0][THREAD_SIZE_X * tx + thread_x]);
    }
    ///目前是处于写阶段还是读阶段
    int write_stage_inx = 1;
    //大迭代时，所处理的A矩阵的第一列的列号
    int tile_idx = 0;
    ///256次大迭代
    do{
        tile_idx += BLOCK_SIZE_K;
        if(tile_idx< K){
            //如果还有下一个迭代，则将下一个迭代的数据块，搬运到寄存器上，这里面的for循环代表可能需要多次搬运
            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
                int ldg_index = i / A_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(ldg_a_reg[ldg_index]) = FETCH_FLOAT4(A[OFFSET(
                    BLOCK_SIZE_M * by + A_TILE_ROW_START + i, // row
                    A_TILE_COL + tile_idx, // col
                    K )]);
            }
            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
                int ldg_index = i / A_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(ldg_b_reg[ldg_index]) = FETCH_FLOAT4(B[OFFSET(
                    tile_idx + B_TILE_ROW_START + i, // row
                    B_TILE_COL + BLOCK_SIZE_N * bx, // col
                    N )]);
            }

        }
        //load_stage_idx参数代表需要从As的哪个空间（读空间还是写空间）进行读数
        int load_stage_idx = write_stage_idx ^ 1;
        //BLOCK_SIZE_K-1次小迭代，最后一个小迭代放到最后
        #pragma unroll
        for(int j=0; j<BLOCK_SIZE_K-1; ++j){
            //这里已经开始从sharemen中预取下一次小迭代的数据到寄存器了
            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
                FETCH_FLOAT4(frag_a[(j+1)%2][thread_y]) = FETCH_FLOAT4(As[load_stage_idx][j+1][THREAD_SIZE_Y * ty + thread_y]);
            }
            // 同上
            #pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
                FETCH_FLOAT4(frag_b[(j+1)%2][thread_x]) = FETCH_FLOAT4(Bs[load_stage_idx][j+1][THREAD_SIZE_X * tx + thread_x]);
            }
            // compute C THREAD_SIZE_X x THREAD_SIZE_Y
            //我们已经在循环最开始的时候对第一个大迭代的第一个小迭代进行了预取
            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
                #pragma unroll
                for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                    accum[thread_y][thread_x] += frag_a[j%2][thread_y] * frag_b[j%2][thread_x];
                }
            }
        }
    //如果还有下一个迭代，则将刚刚从global mem上搬运到register上的数据在搬运到sharemem上。
        if(tile_idx < K){
            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
                int ldg_index = i / A_TILE_ROW_STRIDE * 4;
                As[write_stage_idx][A_TILE_COL][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index];
                As[write_stage_idx][A_TILE_COL+1][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+1];
                As[write_stage_idx][A_TILE_COL+2][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+2];
                As[write_stage_idx][A_TILE_COL+3][A_TILE_ROW_START + i]=ldg_a_reg[ldg_index+3];
            }
            // load B from global memory to shared memory
            #pragma unroll
            for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
                int ldg_index = i / B_TILE_ROW_STRIDE * 4;
                FETCH_FLOAT4(Bs[write_stage_idx][B_TILE_ROW_START + i][B_TILE_COL]) = FETCH_FLOAT4(ldg_b_reg[ldg_index]);
            }
            // use double buffer, only need one sync
            __syncthreads();
            // switch
            write_stage_idx ^= 1;
        }
        //最后完成下一个大迭代的小迭代中寄存器的预取。有些奇怪的是为什么这里不用判断tile_idx < K
        #pragma unroll
        for (int thread_y = 0; thread_y < THREAD_SIZE_Y; thread_y += 4) {
            FETCH_FLOAT4(frag_a[0][thread_y]) = FETCH_FLOAT4(As[load_stage_idx^1][0][THREAD_SIZE_Y * ty + thread_y]);
        }
        // load B from shared memory to register
        #pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x += 4) {
            FETCH_FLOAT4(frag_b[0][thread_x]) = FETCH_FLOAT4(Bs[load_stage_idx^1][0][THREAD_SIZE_X * tx + thread_x]);
        }
        //计算本次大迭代的最后一个小迭代
        #pragma unroll
        for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
            #pragma unroll
            for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                accum[thread_y][thread_x] += frag_a[1][thread_y] * frag_b[1][thread_x];
            }
        }

    }
    //最后将计算结果写回C
    #pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
        #pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; thread_x+=4) {
            FETCH_FLOAT4(C[OFFSET(
                BLOCK_SIZE_M * by + ty * THREAD_SIZE_Y + thread_y,
                BLOCK_SIZE_N * bx + tx * THREAD_SIZE_X + thread_x,
                N)]) = FETCH_FLOAT4(accum[thread_y][thread_x]);
        }
    }
}
int main(int argc, char ** argv){
    if (argc != 4) {
        printf("usage: ./main [M] [K] [N]\n");
        exit(0);
    }
    size_t M = atoi(argv[1]);
    size_t K = atoi(argv[2]);
    size_t N = atoi(argv[3]);

    assert( M%8 == 0); 
    assert( N%8 == 0); 
    assert( K%8 == 0); 

    size_t bytes_A = sizeof(float) * M * K;
    size_t bytes_B = sizeof(float) * K * N;
    size_t bytes_C = sizeof(float) * M * N;
    float* h_A = (float*)malloc(bytes_A);
    float* h_B = (float*)malloc(bytes_B);
    float* h_C = (float*)malloc(bytes_C);
    float* h_C_blas = (float*)malloc(bytes_C);

    float* d_A;
    float* d_B;
    float* d_C;

    checkCudaErrors(cudaMalloc(&d_A, bytes_A));
    checkCudaErrors(cudaMalloc(&d_B, bytes_B));
    checkCudaErrors(cudaMalloc(&d_C, bytes_C));
    const int BLOCK_SIZE_M = 128;
    const int BLOCK_SIZE_K = 8;
    const int BLOCK_SIZE_N = 128;
    const int THREAD_SIZE_X = 8;
    const int THREAD_SIZE_Y = 8;
    const bool ENABLE_DOUBLE_BUFFER = false;

    double msecPerMatrixMul[2] = {0, 0};
    double gigaFlops[2] = {0, 0};
    double flopsPerMatrixMul = 2.0 * M * N * K;

    // generate A
    for( int i = 0; i < M * K; i++ ){
        h_A[i] = i / 13;
    }

    // generate B
    for( int i = 0; i < K * N; i++ ) {
        h_B[i] = i % 13;
    }

    checkCudaErrors(cudaMemcpy( d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float msecTotal = 0;
    int nIter = 1000;

    checkCudaErrors(cudaMemcpy( d_C, h_C, bytes_C, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaEventRecord(start));
    for (int run = 0 ; run < nIter; run ++ ) {
        dim3 dimBlock(BLOCK_SIZE_N / THREAD_SIZE_X, BLOCK_SIZE_M / THREAD_SIZE_Y);
        dim3 dimGrid(N / BLOCK_SIZE_N, M / BLOCK_SIZE_M);
        Sgemm<BLOCK_SIZE_M, BLOCK_SIZE_K, BLOCK_SIZE_N, THREAD_SIZE_Y, THREAD_SIZE_X, ENABLE_DOUBLE_BUFFER> 
        <<< dimGrid, dimBlock >>>(d_A, d_B, d_C, M, N, K);
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    checkCudaErrors(cudaMemcpy( h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));
    msecPerMatrixMul[0] = msecTotal / nIter;
    gigaFlops[0] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[0] / 1000.0f);
    printf( "My gemm Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
        gigaFlops[0],
        msecPerMatrixMul[0],
        flopsPerMatrixMul);


     // cublas
     cublasHandle_t blas_handle;  
     cublasCreate(&blas_handle);
     float alpha = 1.0;
     float beta = 0;
     checkCudaErrors(cudaMemcpy( d_C, h_C, bytes_C, cudaMemcpyHostToDevice));
     checkCudaErrors(cudaEventRecord(start));
     for (int run = 0 ; run < nIter; run ++ ) {
         cublasSgemm (blas_handle, CUBLAS_OP_T, CUBLAS_OP_T, 
             M, N, K, &alpha, 
             d_A, K, d_B, N, &beta, d_C, N
         );
     }
     checkCudaErrors(cudaEventRecord(stop));
     checkCudaErrors(cudaEventSynchronize(stop));
     checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
 
     checkCudaErrors(cudaMemcpy( h_C_blas, d_C, bytes_C, cudaMemcpyDeviceToHost));
 
     msecPerMatrixMul[1] = msecTotal / nIter;
     gigaFlops[1] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[1] / 1000.0f);
     printf( "CuBlas Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
         gigaFlops[1],
         msecPerMatrixMul[1],
         flopsPerMatrixMul);
 
     cublasDestroy(blas_handle); 
     
     double eps = 1.e-6;  // machine zero
     bool correct = true;
     for (int i = 0; i < M * N; i++) {
         int row = i / N;
         int col = i % N;
         double abs_err = fabs(h_C[i] - h_C_blas[col * M + row]);
         double dot_length = M;
         double abs_val = fabs(h_C[i]);
         double rel_err = abs_err / abs_val / dot_length;
         if (rel_err > eps) {
             printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                     i, h_C[i], h_C_blas[col * M + row], eps);
             correct = false;
             break;
         }
     }
 
     printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");
     printf("ratio= %f\n", gigaFlops[0] / gigaFlops[1]);
     
     // Free Memory
     cudaFree(d_A);
     cudaFree(d_B);
     cudaFree(d_C);
     
     free(h_A);
     free(h_B);
     free(h_C);
     free(h_C_blas);
}