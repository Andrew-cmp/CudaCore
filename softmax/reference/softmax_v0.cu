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
#define BLOCK_DIM 128
//块级归约是指在 单个 CUDA 线程块（Block）内部，将多个线程的数据通过某种操作（如求和、最大值、最小值等）合并为一个结果的过程。
//所有线程通过 共享内存（Shared Memory） 协作完成归约，结果仅在块内可见。
//wrap规约是先对每32个线程内部进行规约，然后再完成块级规约。
double get_walltime() {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return (double) (tp.tv_sec + tp.tv_usec*1e-6);
}
// 一维block softmax grid(N/BLOCK_DIM) block(BLOCK_DIM)
__global__
void BlockMax(float *input, int size, float *result){
  int tid = blockDim.x*blockIdx.x+threadIdx.x;
  //定义了一个名为 BlockReduce 的类型，它是CUB库中 cub::BlockReduce 模板类的实例化。
  //该类型提供了高效的块内归约操作（如求和、最大值、最小值等）。
  typedef cub::BLockReduce<float,BLOCK_DIM> BlockReduce;
  //在共享内存中分配一块临时存储空间，用于归约操作的中间结果。
  __shared__ typename BlockReduce::TempStorage temp_storage;
  if(tid< size){
    float value = input[tid];
    float block_max = BlockReduce(temp_storage).Reduce(value,cub::Max());
    if(threadIdx.x == 0){
      result[blockIdx.x] = block_max;
    }
  }
}
//和reduce里面第三中方法一样
__global__
void GridMax(float *result, int num_blocks){
  for(int s = num_blocks; s >0;s>>=1 ){
    if(threadIdx.x + stride < num_blocks){
      result[threadIdx.x] = max_function(result[threadIdx.x + stride], result[threadIdx.x]);
    }
    __syncthreads();
  }
}
//不仅做sum操作，还将所有的操作数xi进行标准化和exp
__global__ BlockSum(float *input, int size, float *result){
  int tid = blockDim.x*blockIdx.x+threadIdx.x;
  typedef cub::BLockReduce<float,BLOCK_DIM> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  if(tid < size){
    input[tid] = __expf(input[tid]-result[0]);
    float value = input[tid];
    //这里不用同步吗，因为肯定要等所有的val计算结束啊
    // 执行规约操作
    float block_sum = BlockReduce(temp_storage).Reduce(value,cub::Sum());
    // 只有第一个线程将结果写回到全局内存
    if (threadIdx.x == 0)
    {
        result[blockIdx.x] = block_sum;
    }
  }
}
__global__
void GridSum(float *result, int num_blocks){
  for(int s = num_blocks; s >0;s>>=1 ){
    if(threadIdx.x + stride < num_blocks){
      result[threadIdx.x] += result[threadIdx.x + stride];
    }
    __syncthreads();
  }
}
__global__ void softmax(float *input, float *result, int size){
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  if (tid < size){
      input[tid] /= result[0];
  }
}

void cpu_softmax(float *cpu_input, int size){
  doule st, ela;
  st = get_walltime()
  int num_blocks = ceil(size/(double)BLOCK_DIM);
  float* input, *result, *cpu_result;

  //因为是先进行Blcok reduce，所有所有block reduce的结果都要存到这。
  int mem_size = num_blocks*sizeof(float);
  cudaMalloc((void**)&input, size*sizeof(float));
  cudaMalloc((void **) &result, mem_size);
  cpu_result = (float *)malloc(mem_size);
  cudaMemcpy(input, cpu_input, size*sizeof(float), cudaMemcpyHostToDevice);
  cudaEvent_t start, end;
  float kernel_time=0;
  // 进行一次block max在对block max的结果规约到一个。
  // sum 同理，不过在进行sum的时候有一个elemwise的操作
  // 最后再进一个elemwise的操作。
  cudaEventCreate((&start));
  cudaEventCreate($stop);
  cudaEventRecord(start,0);
  dim3 BlockSize(BLOCK_DIM);
  dim3 GridSize(N/BLOCK_DIM);
  BlockMax<<<GridSize,BlockSize>>>(input,size,result);
  GridMax<<<GridSize,BlockSize>>>(result,num_blocks);
  BlockSum<<<GridSize,BlockSize>>>(input,size,result);
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&kernel_time,start,stop);
  cudaMemcpy(cpu_input, input, size*sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(input);
  cudaFree(result);
  free(cpu_result);
  ela = get_walltime() - st;
  printf("BlockReduce,kernel time:%.4f, use time:%.4f\n", ker_time/1000., ela);
}
int main(){
  float *cpu_input;
  int size = 16;

  cpu_input = (float *)malloc(size*sizeof(float));
  for(int i = 0; i < size; i++){
      cpu_input[i] = i%100;

  }
  cpu_softmax(cpu_input, size);


  for(int i = 0; i < size; i++){

      printf("softmax:%.4e\n",cpu_input[i]);
  }

  free(cpu_input);


  return 0;
}
