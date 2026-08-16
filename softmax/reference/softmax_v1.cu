#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#include <algorithm>
#include <iostream>
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include <math_constants.h>
using namespace std;
//https://github.com/xgqdut2016/cuda_code/blob/main/softmax/1Dsoftmax/softmax_v4.cu
__global__ softmax_kernel(float * input, int size){
    int tid = thread.x+blockIdx.x*blockDim.x;
    int val = input[tid];
    float max_value = -__FLT_MAX__;
    //这里感觉问题挺大的，一方面跨block通信肯定不快，另一方面存在重复计算的问题，每个block都算了block间的最大值,只开一个block就可以了。 
    for(int i = threadIdx.x; i < size;i+=BLOCK_DIM){
        max_value = device_max(max_value,value[i]);
    }
    typedef BlockReduce<float, BLOCK_DIM> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ flaot max_total;
    float block_max =  BlockReduce(temp_storage).Reduce(max_partial,device_max);
    if(threadIdx.x == 0){
        max_total = block_max;//max_total是share才能保证即使threadIdx.x !=0时，线程也能获得max_total的取值
    }
    __syncthreads();

    float sum_partial = 0.0f;
    for(int i = threadIdx.x; i < size;i+=BLOCK_DIM){
        sum_partial +=  __expf(input[id] - max_total);
    }
    float sum_partial = 0.0f;
    for(int id = threadIdx.x; id < size; id += BLOCK_DIM){
        sum_partial += __expf(input[id] - max_total);//CUDA高精度exp函数，把所有信息集中到一个线程块
    } 
    
    __shared__ float sum_inverse_total;
    float block_sum = BlockReduce(temp_storage).Reduce(sum_partial,cub::Sum());
    if (threadIdx.x == 0){
        sum_inverse_total = __fdividef(1.0F, block_sum);//高精度除法
    }

    __syncthreads();
    input[tid] = __expf(input[tid] - max_total)*sum_inverse_total;
    
}



int main(){


    
}