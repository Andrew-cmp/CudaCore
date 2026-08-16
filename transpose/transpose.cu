#include <stdio.h>
#include <stdlib.h>
#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <random>
#define CEIL(a, b) ((a) + (b) - 1) / (b)
#define cudaCheck(err) _cudaCheck(err, __FILE__, __LINE__)
#define TIME_RECORD(N, func)                                                                    \
    [&] {                                                                                       \
        float total_time = 0;                                                                   \
        for (int repeat = 0; repeat <= N; ++repeat) {                                           \
            cudaEvent_t start, stop;                                                            \
            cudaCheck(cudaEventCreate(&start));                                                 \
            cudaCheck(cudaEventCreate(&stop));                                                  \
            cudaCheck(cudaEventRecord(start));                                                  \
            cudaEventQuery(start);                                                              \
            func();                                                                             \
            cudaCheck(cudaEventRecord(stop));                                                   \
            cudaCheck(cudaEventSynchronize(stop));                                              \
            float elapsed_time;                                                                 \
            cudaCheck(cudaEventElapsedTime(&elapsed_time, start, stop));                        \
            if (repeat > 0) total_time += elapsed_time;                                         \
            cudaCheck(cudaEventDestroy(start));                                                 \
            cudaCheck(cudaEventDestroy(stop));                                                  \
        }                                                                                       \
        if (N == 0) return (float)0.0;                                                          \
        return total_time;                                                                      \
    }()

void print_matrix(float* a, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%7.3f", a[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

bool verify_matrix(float *mat1, float *mat2, size_t N) {
    double diff = 0.0;
    int i;
    for (i = 0; mat1 + i && mat2 + i && i < N; i++) {
        diff = fabs((double) mat1[i] - (double) mat2[i]);
        if (diff > 1e-4) {
            printf("Error: mat1[%d]=%5.6f, mat2[%d]=%5.6f, \n", i, mat1[i], i, mat2[i]);
            return false;
        }
    }
    return true;
}
void _cudaCheck(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s(line %d):\n%s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    return;
}

void randomize_matrix(float *mat, int N) {
    std::random_device rd;  
    std::mt19937 gen(rd()); // 使用随机设备初始化生成器  

    // 创建一个在[0, 2000)之间均匀分布的分布对象  
    std::uniform_int_distribution<> dis(0, 2000); 
    for (int i = 0; i < N; i++) {
        // 生成随机数，限制范围在[-1.0,1.0]
        mat[i] = (dis(gen)-1000)/1000.0;  
    }
}
void host_transpose(float* input, int M, int N, float* output) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            output[i * M + j] = input[j * N + i];
        }
    }
}

__global__ void device_transpose_v0(float* input, float* output, int M, int N){
    const int x = threadIdx.x+blockIdx.x*blockDim.x;
    const int y = threadIdx.y+blockIdx.y*blockDim.y;

    if(y < M && x < N){

        output[y+x*M] = input[x+y*N];
    }

}
template<const int TILE_DIM>
__global__ void device_transpose_v2(float* input , float* output ,int M, int N){
    const int x = threadIdx.x+blockIdx.x*blockDim.x;
    const int y = threadIdx.y+blockIdx.y*blockDim.y;

    // assert(TILE_DIM==blockDim.y);
    // assert(TILE_DIM==blockDim.x);

    __shared__  float shared_num[TILE_DIM][TILE_DIM];
    
    if(x < N && y<M)
        shared_num[threadIdx.y][threadIdx.x] = input[y*N+x];
    __syncthreads();

    int xx = blockIdx.x*blockDim.x+threadIdx.y;
    int yy = blockIdx.y*blockDim.y+threadIdx.x;
    if(yy < M && xx < N){
        //下标画个图就明了了。
        //共享内存里的数据要做一次“坐标transpose”，共享内存中“transpose”之后的数据放到output时，output的坐标也要做一次“transpose”
        //共享内存“坐标transpose”也就是[threadIdx.y][threadIdx.x] -> [threadIdx.x][threadIdx.y]的过程
        //关于bank confilct：
        //一种思路：由于 TILE_DIM = 16，访问步长是：
        //对于一个 warp（假设 threadIdx.x 连续变化），col = threadIdx.y 固定，row = threadIdx.x 从 0 到 16
        //地址差 = TILE_DIM * sizeof(float) = 16 × 4B = 64B
        // 当threadId.y固定时，每隔一个就相当于隔了128B，这正好是一行bank的值，说明每隔一个线程就要访问同一个bank。

        //另一种思路：假设BLOCK_DIM = 16,一个warp 32个线程，（假设 threadIdx.x 连续变化），col = threadIdx.y 固定，row = threadIdx.x 从 0 到 16
        //bank_id = ((threadIdx.x * TILE_DIM) + threadIdx.y) mod 32。当 TILE_DIM = 16 且 threadIdx.y 固定时：
        //bank_id = (threadIdx.x * 16 + const) mod 32
        //因为 16 mod 32 = 16，所以：
        // threadIdx.x = 0 → bank_id = const
        // threadIdx.x = 1 → bank_id = const + 16 (mod 32)
        // threadIdx.x = 2 → bank_id = const + 0 (mod 32) ← 重复了！
        // threadIdx.x = 3 → bank_id = const + 16 (mod 32)
        //一个 warp 内的线程访问的 bank pattern 是 两个 bank 来回交替
        //最终造成 32 个线程访问 2 个 bank → 16-way bank conflict（每次 1/16 的 bank 并行度）
        
        output[yy+xx*M] = shared_num[threadIdx.x][threadIdx.y];
    }

}

template<const int TILE_DIM>
__global__ void device_transpose_v3(float* input , float* output ,int M, int N){
    const int x = threadIdx.x+blockIdx.x*blockDim.x;
    const int y = threadIdx.y+blockIdx.y*blockDim.y;

    // assert(TILE_DIM==blockDim.y);
    // assert(TILE_DIM==blockDim.x);

    __shared__  float shared_num[TILE_DIM][TILE_DIM + 1]; // 对共享内存做padding，解决bank conflict
    
    if(x < N && y<M)
        shared_num[threadIdx.y][threadIdx.x] = input[y*N+x];
    __syncthreads();

    int xx = blockIdx.x*blockDim.x+threadIdx.y;
    int yy = blockIdx.y*blockDim.y+threadIdx.x;
    if(yy < M && xx < N){
        //关于bank confilct 是如何通过padding解决的：
        //一种思路：由于 TILE_DIM = 16，但此时第二维+1，因此row_stride=17，
        //对于一个 warp（假设 threadIdx.x 连续变化），col = threadIdx.y 固定，row = threadIdx.x 从 0 到 16
        //地址差 = TILE_DIM * sizeof(float) = 17 × 4B = 68B
        // 当threadId.y固定时，每隔一个就相当于隔了136B，136B%128B=8B，因此现在每隔一个线程，访的bank id也隔了两个，解决的bank冲突。
        
        //另一种思路：row_stride = TILE_DIM + 1
        // 假设 TILE_DIM = 16 → row_stride = 17。
        // 转置读 shared_num[threadIdx.x][threadIdx.y] 时：
        // bank_id：bank_id=(threadIdx.x×17+const)mod32
        // 17 mod 32 = 17，意味着每增加一行，bank id 向前跨 17 个 bank，而不是 16 个：
        // tid.x = 0 → bank_id = const
        // tid.x = 1 → bank_id = const + 17
        // tid.x = 2 → bank_id = const + 34 → (34 mod 32 = const + 2)
        // tid.x = 3 → bank_id = const + 19
        output[yy+xx*M] = shared_num[threadIdx.x][threadIdx.y];
    }

}

template<const int TILE_DIM>
__global__ void device_transpose_v4(float* input , float* output ,int M, int N){
    const int x = threadIdx.x+blockIdx.x*blockDim.x;
    const int y = threadIdx.y+blockIdx.y*blockDim.y;

    // assert(TILE_DIM==blockDim.y);
    // assert(TILE_DIM==blockDim.x);

    __shared__  float shared_num[TILE_DIM][TILE_DIM]; 
    
    if(x < N && y<M)
        shared_num[threadIdx.y][threadIdx.x ^ threadIdx.y] = input[y*N+x];
    __syncthreads();

    int xx = blockIdx.x*blockDim.x+threadIdx.y;
    int yy = blockIdx.y*blockDim.y+threadIdx.x;
    if(yy < M && xx < N){
        // swizzling主要利用了异或运算的以下两个性质来规避bank conflict：
        // 1. 运算的封闭性 
        //对于给定的 y，x ^ y 仍然在合法索引范围内（不会越界）。
        // 2. x1^y!=x2^y当且仅当x1!=x2 
        //
        //意味着 warp 内不同线程的 (threadIdx.x) 经过异或变换后，得到的结果仍然不同。
        // 举例：
        // 第一行的访存位置由0,0,0,0...变为0,1,2,3...
        // 第二行的访存位置由1,1,1,1...变为1,0,3,2...
        // 第三行的访存位置由2,2,2,2...变为2,3,0,1...
        // 第四行的访存位置由3,3,3,3...变为3,2,1,0...
        // 这样既能保证充分利用shared memory的空间（由于性质1和2）
        // 又能保证warp中的各个线程不会访问同一bank（由于性质2）
        output[yy+xx*M] = shared_num[threadIdx.x][threadIdx.x ^ threadIdx.y];
    }
}

int main() {
    // 输入是M行N列，转置后是N行M列
    size_t M = 12800;
    size_t N = 1280;
    constexpr size_t BLOCK_SIZE = 32;
    const int repeat_times = 10;

    // --------------------host 端计算一遍转置, 输出的结果用于后续验证---------------------- //
    float *h_matrix = (float *)malloc(sizeof(float) * M * N);
    float *h_matrix_tr_ref = (float *)malloc(sizeof(float) * N * M);
    randomize_matrix(h_matrix, M * N);
    host_transpose(h_matrix, M, N, h_matrix_tr_ref);
    // printf("init_matrix:\n");
    // print_matrix(h_matrix, M, N);
    // printf("host_transpose:\n");
    // print_matrix(h_matrix_tr_ref, N, M);

    float *d_matrix;
    cudaMalloc((void **) &d_matrix, sizeof(float) * M * N);
    cudaMemcpy(d_matrix, h_matrix, sizeof(float) * M * N, cudaMemcpyHostToDevice);
    free(h_matrix);

    // --------------------------------call transpose_v0--------------------------------- //
    float *d_output0;
    cudaMalloc((void **) &d_output0, sizeof(float) * N * M);                              // device输出内存
    float *h_output0 = (float *)malloc(sizeof(float) * N * M);                            // host内存, 用于保存device输出的结果

    dim3 block_size0(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size0(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE));                            // 根据input的形状(M行N列)进行切块
    float total_time0 = TIME_RECORD(repeat_times, ([&]{device_transpose_v0<<<grid_size0, block_size0>>>(d_matrix, d_output0, M, N);}));
    cudaMemcpy(h_output0, d_output0, sizeof(float) * N * M, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    verify_matrix(h_output0, h_matrix_tr_ref, M * N);                                     // 检查正确性
    printf("[device_transpose_v0] Average time: (%f) ms\n", total_time0 / repeat_times);  // 输出平均耗时

    cudaFree(d_output0);
    free(h_output0);

    // // --------------------------------call transpose_v1--------------------------------- //
    // float *d_output1;
    // cudaMalloc((void **) &d_output1, sizeof(float) * N * M);                              // device输出内存
    // float *h_output1 = (float *)malloc(sizeof(float) * N * M);                            // host内存, 用于保存device输出的结果

    // dim3 block_size1(BLOCK_SIZE, BLOCK_SIZE);
    // dim3 grid_size1(CEIL(M, BLOCK_SIZE), CEIL(N, BLOCK_SIZE));                            // 根据output的形状(N行M列)进行切块
    // float total_time1 = TIME_RECORD(repeat_times, ([&]{device_transpose_v1<<<grid_size1, block_size1>>>(d_matrix, d_output1, M, N);}));
    // cudaMemcpy(h_output1, d_output1, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    // cudaDeviceSynchronize();

    // verify_matrix(h_output1, h_matrix_tr_ref, M * N);                                     // 检查正确性
    // printf("[device_transpose_v1] Average time: (%f) ms\n", total_time1 / repeat_times);  // 输出平均耗时

    // cudaFree(d_output1);
    // free(h_output1);


    // --------------------------------call transpose_v2--------------------------------- //
    float *d_output2;
    cudaMalloc((void **) &d_output2, sizeof(float) * N * M);                              // device输出内存
    float *h_output2 = (float *)malloc(sizeof(float) * N * M);                            // host内存, 用于保存device输出的结果

    dim3 block_size2(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size2(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE));                            // 根据input的形状(M行N列)进行切块
    float total_time2 = TIME_RECORD(repeat_times, ([&]{device_transpose_v2<BLOCK_SIZE> <<<grid_size2, block_size2>>> (d_matrix, d_output2, M, N);}));
    cudaMemcpy(h_output2, d_output2, sizeof(float) * N * M, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    verify_matrix(h_output2, h_matrix_tr_ref, M * N);                                     // 检查正确性
    printf("[device_transpose_v2] Average time: (%f) ms\n", total_time2 / repeat_times);  // 输出平均耗时

    cudaFree(d_output2);
    free(h_output2);


        // --------------------------------call transpose_v3--------------------------------- //
    float *d_output3;
    cudaMalloc((void **) &d_output3, sizeof(float) * N * M);                              // device输出内存
    float *h_output3 = (float *)malloc(sizeof(float) * N * M);                            // host内存, 用于保存device输出的结果

    dim3 block_size3(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size3(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE));                            // 根据input的形状(M行N列)进行切块
    float total_time3 = TIME_RECORD(repeat_times, ([&]{device_transpose_v3<BLOCK_SIZE> <<<grid_size3, block_size3>>> (d_matrix, d_output3, M, N);}));
    cudaMemcpy(h_output3, d_output3, sizeof(float) * N * M, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    verify_matrix(h_output3, h_matrix_tr_ref, M * N);                                     // 检查正确性
    printf("[device_transpose_v3] Average time: (%f) ms\n", total_time3 / repeat_times);  // 输出平均耗时

    cudaFree(d_output3);
    free(h_output3);

    // --------------------------------call transpose_v4--------------------------------- //
    float *d_output4;
    cudaMalloc((void **) &d_output4, sizeof(float) * N * M);                              // device输出内存
    float *h_output4 = (float *)malloc(sizeof(float) * N * M);                            // host内存, 用于保存device输出的结果

    dim3 block_size4(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size4(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE));                            // 根据input的形状(M行N列)进行切块
    float total_time4 = TIME_RECORD(repeat_times, ([&]{device_transpose_v4<BLOCK_SIZE><<<grid_size4, block_size4>>>(d_matrix, d_output4, M, N);}));
    cudaMemcpy(h_output4, d_output4, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    verify_matrix(h_output4, h_matrix_tr_ref, M * N);
    printf("[device_transpose_v4] Average time: (%f) ms\n", total_time4 / repeat_times);

    cudaFree(d_output4);
    free(h_output4);

    // // --------------------------------call transpose_v5--------------------------------- //
    // float *d_output5;
    // cudaMalloc((void **) &d_output5, sizeof(float) * N * M);                              // device输出内存
    // float *h_output5 = (float *)malloc(sizeof(float) * N * M);                            // host内存, 用于保存device输出的结果

    // dim3 block_size5(BLOCK_SIZE, BLOCK_SIZE);
    // dim3 grid_size5(CEIL(N, BLOCK_SIZE), CEIL(M, BLOCK_SIZE));                            // 根据input的形状(M行N列)进行切块
    // float total_time5 = TIME_RECORD(repeat_times, ([&]{device_transpose_v5<BLOCK_SIZE><<<grid_size5, block_size5>>>(d_matrix, d_output5, M, N);}));
    // cudaMemcpy(h_output5, d_output5, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    // cudaDeviceSynchronize();

    // verify_matrix(h_output5, h_matrix_tr_ref, M * N);
    // printf("[device_transpose_v5] Average time: (%f) ms\n", total_time5 / repeat_times);

    // cudaFree(d_output5);
    // free(h_output5);

    // ---------------------------------------------------------------------------------- //
    free(h_matrix_tr_ref);

    return 0;
}