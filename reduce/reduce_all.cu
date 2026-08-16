#include "cuda_runtime.h"
#include <iostream>
template<int BLOCK_SIZE,int N>
__global__ void BlockReduce(int* input){
    int tx = threadIdx.x;
    int tid = blockIdx.x * blockDim.x + tx;
    __shared__ int temp[BLOCK_SIZE];
    if(tx < N)
        temp[tx] = input[tid];
    else
        temp[tx] = 0;
    __syncthreads();
    for(int i = BLOCK_SIZE>>1;i > 0;i >>= 1){
        if(tx < i){
            temp[tx] += temp[tx+i];
        }
        __syncthreads();
    }
    if(tx == 0){
        input[blockIdx.x] = temp[0];
    }
}
//block
template<int BLOCK_SIZE>
__device__ int warReduce(int sum){
    if(BLOCK_SIZE>=32){
        sum += __shfl_down_sync(0xffffffff,sum,16);
    }
    if(BLOCK_SIZE>=16){
        sum += __shfl_down_sync(0xffffffff,sum,8);
    }
    if(BLOCK_SIZE>=8){
        sum += __shfl_down_sync(0xffffffff,sum,4);
    }
    if(BLOCK_SIZE>=4){
        sum += __shfl_down_sync(0xffffffff,sum,2);
    }
    if(BLOCK_SIZE>=2){
        sum += __shfl_down_sync(0xffffffff,sum,1);
    }
    return sum;
}
template<int BLOCK_SIZE>
__global__ void BlockReduceUsingWarp(int* input, int *output){
    int tx = threadIdx.x;
    int tid = blockIdx.x * blockDim.x + tx;
    constexpr int warp_num = BLOCK_SIZE/32;
    __shared__ int temp[warp_num];
    int sum;
    sum = input[tid];
    sum = warReduce<BLOCK_SIZE>(sum);
    int lid = tx%32;
    int wid = tx/32;
    if(lid == 0){
        temp[wid] = sum;
    }
    __syncthreads();
    if(wid == 0){
        sum = temp[lid];
    }
    sum = warReduce<warp_num>(sum);
    if(tx == 0){
        output[blockIdx.x] = sum;
    }
}
template<int BLOCK_SIZE, int N>
void Reduce(int* input){
    const int block_num = (N+BLOCK_SIZE-1)/BLOCK_SIZE;
    dim3 block_size(BLOCK_SIZE);
    ///比如有100000000个元素，那么就有100000000/1024=100000个block。
    dim3 grid_size(block_num);

    //现在先假设一个kernel都能算完吧
    BlockReduce<BLOCK_SIZE,N><<<grid_size,block_size>>>(input);
    // printf("block_num:%d\n",block_num);
    // int * h_buffer = (int*)malloc(sizeof(int)*128);
    // cudaMemcpy(h_buffer,buffer,sizeof(int)*128,cudaMemcpyDeviceToHost);
    // printf("buffer[0] = %d\n",h_buffer[0]);
    // for(int i = 0;i < block_num-10;i++){
    //     printf("buffer[%d] = %d\n",i,buffer[i]);
    // }
    BlockReduce<BLOCK_SIZE,block_num><<<1,block_num>>>(input);
}
template<int BLOCK_SIZE, int N>
void ReduceUsingWarp(int* input, int *output, int* buffer){
    const int block_num = (N+BLOCK_SIZE-1)/BLOCK_SIZE;
    dim3 block_size(BLOCK_SIZE);
    ///比如有100000000个元素，那么就有100000000/1024=100000个block。
    dim3 grid_size(block_num);

    //现在先假设一个kernel都能算完吧
    BlockReduceUsingWarp<BLOCK_SIZE><<<grid_size,block_size>>>(input,buffer);
    // printf("block_num:%d\n",block_num);
    // int * h_buffer = (int*)malloc(sizeof(int)*128);
    // cudaMemcpy(h_buffer,buffer,sizeof(int)*128,cudaMemcpyDeviceToHost);
    // printf("buffer[0] = %d\n",h_buffer[0]);
    // for(int i = 0;i < block_num-10;i++){
    //     printf("buffer[%d] = %d\n",i,buffer[i]);
    // }
    BlockReduceUsingWarp<block_num><<<1,block_num>>>(buffer,output);
}
template<int BLOCK_SIZE, int N>
void ReduceLargeN(int* input){
    const int block_num = (N+BLOCK_SIZE-1)/BLOCK_SIZE;
    dim3 block_size(BLOCK_SIZE);
    ///比如有100000000个元素，那么就有100000000/1024=100000个block。
    dim3 grid_size(block_num);
    BlockReduce<BLOCK_SIZE,N><<<grid_size,block_size>>>(input);
    // printf("block_num:%d\n",block_num);
    // int * h_buffer = (int*)malloc(sizeof(int)*128);
    // cudaMemcpy(h_buffer,input,sizeof(int)*128,cudaMemcpyDeviceToHost);
    // for(int i = 0;i < 128;i++){
    //     printf("buffer[%d] = %d\n",i,h_buffer[i]);
    // }
    if(block_num > 1){
        ReduceLargeN<BLOCK_SIZE,block_num>(input);
    }
}
int main(){
    const int N = 1024*128;
    const int BLOCK_SIZE = 1024;
    //申请host端内存
    int * h_ans = (int*)malloc(sizeof(int));
    int * h_nums =(int*)malloc(N*sizeof(int));
    //host段内存初始化
    for(size_t i = 0;i < N;i++){
        h_nums[i] = 1;
    }
    //申请device端内存
    int *d_nums;
    cudaMalloc(&d_nums,N*sizeof(int));
    int *d_ans;
    cudaMalloc(&d_ans,sizeof(int));
    int* buffer;
    cudaMalloc(&buffer,sizeof(int)*N/BLOCK_SIZE);

    //将host端内存拷贝到device端
    cudaMemcpy(d_nums,h_nums,N*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemset(d_ans,0,sizeof(int));
    cudaMemset(buffer,0,sizeof(int)*N/BLOCK_SIZE);

    ////调用kernel
    //不需要output和buffer，和前缀和不一样，不需要之前的结果
    //Reduce<BLOCK_SIZE,N>(d_nums);

    //使用output和buffer的版本
    //ReduceUsingWarp<BLOCK_SIZE,N>(d_nums,d_ans,buffer);

    //使用递归的版本
    ReduceLargeN<BLOCK_SIZE,N>(d_nums);
    
    cudaMemcpy(h_nums,d_nums,N*sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ans,d_ans,sizeof(int),cudaMemcpyDeviceToHost);
    printf("ans = %d\n",h_nums[0]);
}




