
#include <cuda_runtime.h>
#include <stdio.h>
#define N 1024
#define BLOCK_SIZE 256
///https://zhuanlan.zhihu.com/p/688610091
//https://zhuanlan.zhihu.com/p/646998011
//https://github.com/BBuf/how-to-optim-algorithm-in-cuda/blob/master/reduce/README.md
// block reduce 针对block内的所有数据进行规约，block间的数据交给host端
//最简单的树形规约，规约轮次为logBLOCK_SIZE
__global__ void reduce_v0(float *g_idata, float * g_odata){
    __share__ float sdata[BLOCK_SIZE];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __synchreads();
    for(unsigned int s = 1; s<blockDim.x;s *= 2){
        //比如step = 1时，0，2，4（2的倍数）会被挑出来
        //step = 2时，0，4，8（4的倍数）会被挑出来
        if(tid%(s<<1) == 0){
            sdata[tid] += sdata[tid+s];
        }
    }
    __synchreads();
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
//每个线程读取两个数，相邻的两个线程读取的两个数间隔为2*s。
//当s=16时，两个数相隔即为32，之后均为32的倍数，因此会产生warp divergence. 
//（当不同线程访问的索引之间为32的倍数关系时，会产生warp divergence. ）

//这样在满足条件的线程个数小于 32 之前, 都不会有 
//在这之后虽然有 warp divergence, 但实际工作的 warp 只有 1 个, 相比于 kernel 0 从一开始就有的情况会好很多.
__global__ void reduce_v1(float *g_idata, float * g_odata){
    //[avoid warp divergence ]
    __share__ float sdata[BLOCK_SIZE];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __synchreads();
    for(unsigned int s = 1; s<blockDim.x;s *= 2){
        ///thread 负责的第一个元素的索引：
        //其计算方式综合图和 stride得到
        //比如step=1时，0，1，2线程负责的第一个元素分别为0，2，4（tid*s*2）
        //比如step=2时，0，1，2线程负责的第一个元素分别为0，4，8（tid*s*2）
        int index = 2*s*tid;
        for(index < blockDim.x){
            sdata[index] += sdata[index+s]
        }
    }
    __synchreads();
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
__global void ReduceV0_5(float* iData, float *oData){
    __share__ float sdata[BLOCK_SIZE];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __synchreads();
    for(unsigned int s = 1; s<blockDim.x;s *= 2){
        if((tid & (2*s-1)) == 0){
            sdata[tid] += sdata[tid+s];
        }
    }
    __synchreads();
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
//顺序寻址
//同样是对共享内存归约的 for 循环进行了改动, 之所以称之为"顺序寻址", 就是因为相比与 kernel 1, 
//每个线程读取的两个数据仍然相差 s, 但相邻线程之间读取的数据则变成连续的了, 即间隔 1. 
//如下图所示. 这样 warp 内的 32 个线程读取的数据就映射到了不同的 32 个 bank, 从而避免了 bank conflict 的问题, 
//进而提高了性能. 
//(注: 额外一提的是, 每个线程会读取间隔差 s 的两个数据, 当 s>=32 时两个数据同样会位于同一 bank 上,
// 但并不会有 bank conflict, 因为这两个数据是由同一线程读取的, 地址不连续的情况下必然要通过前后两个 load 指令读取, 
//因此不会有 bank conflict; bank conflict 需要是不同线程同时读取同一 bank 才会发生.)
__global__ void reduce_v2(float *g_idata,float *g_odata){
    //[avoid warp divergence ]
    //[avoid bank conflict]
    __shared__ float sdata[BLOCK_SIZE];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __syncthreads();
    index = tid;
    for(unsigned int s = blockDim.x/2;s>0;s>>=1){
        if(tid<s){
            sdata[tid] += sdata[tid+s]
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
__global__ void reduce_v3(float *g_idata,float *g_odata){
    __shared__ float sdata[BLOCK_SIZE];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    //代表访问的global id的编号。
    unsigned int gid = blockIdx.x*blockDim.x*2 + threadIdx.x;
    ///相当于在block与global mem层面做了一次规约，示意图与thread和sharemem一致。
    sdata[tid] = g_odata[gid] + g_odata[gid+blockDim.x]
    __syncthreads();
    index = tid;
    for(unsigned int s = blockDim.x/2;s>0;s>>=1){
        if(tid < s){
            sdata[tid] += sdata[index+s]
        }
    }
    __syncthreads();
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

//很关键的一点原因是 reduce 是一个低算术强度的算子, 即计算量本身就不大.
//指令开销可能成为性能瓶颈. 这里说的指令不是加载、存储或者算术指令; 而是指地址算术指令和循环的开销.

//处理的共享内存数据, 需要加上 volatile 修饰符, 表示该变量是"易变的", 其作用是保证每次对 cache 访问是都重新从共享内存中加载数据.
//原因是编译器可能会对代码进行一些优化, 将需要访问的共享内存数据预先保存在寄存器中, 特别是一些架构的 GPU 共享内存之间的数据拷贝必须要经过寄存器;
//此时去掉 volatile 可能导致线程保存在寄存器中的值并不是刚刚由其他线程计算好的最新的结果值,而导致计算结果错误.

//比如线程 0 在计算 cache[0]+=cache[0+4] 时, 需要读取最新的 cache[4], 这有之前线程 4 的 cache[4]+=cache[4+8] 计算得到;
// 而没有 volatile 时编译器在线程 0 预先加载了 cache[4] 的结果到寄存器, 那么就不能拿到线程 4 更新的结果
__device__ void warpReduce_v1(volatitle float* cache, unsigned int tid){
    //由于在 warp 内的指令满足 SIMT 同步(注: 这个要求算力 7.0 以下的 GPU, 后文会具体说明), 因此无需 __syncthreads()
    //也不再需要 if (tid < s) 的条件判断(因为 warp 内线程都会执行这些指令, 这也是 warp divergence 的原因).
    //也就是说这几条指令因为在一个warp内，所以是**顺序**执行的
    //相当于对上面的函数进行了展开。
    cache[tid] += cache[tid+32];
    cache[tid] += cache[tid+16];
    cache[tid] += cache[tid+8];
    cache[tid] += cache[tid+4];
    cache[tid] += cache[tid+2];
    cache[tid] += cache[tid+1];
}
//zai cc>=7的设备上warp 内的指令满足 SIMT 同步的条件不再满足。新提出的independent thread scheduling
//使同一个warp32个线程共用一个PC和栈，编程了各个线程也有各自的PC和栈了。因此，warp内32个线程并不一定同步执行命令。
//也就是说v1版本的warpReduce在同一个warp下不同线程各行不一定使串行同步执行的了，有快有慢。
//（这里 warp 内的线程仍然满足 SIMT, 即任何时钟周期所有活动线程执行相同的指令, 但可能 warp 中的线程被分为了多个活动的线程组）
//对此的措施是，使用__syncwarp()对线程内的数据进行手动同步。
__device__ void warpReduce_v2(volatile float* cache, unsigned int tid){
    //但这种直接在后面加__syncwarp();的行为是不能达到目的了
    //该代码相当于每完成一次对共享内存的读写操作后再进行 warp 的线程同步. 这样仍然是会存在不同线程间读写共享内存的竞态问题,
    //因为这只能保证调用 __syncwarp() 时的线程同步, 而这之间有读和写内存的两个操作, 无法保证每个线程这两个操作步调都是一致的.
    //仍然是上面的例子, 线程 0 在执行 cache[0]+=cache[0+4] 时需读取 cache[4], 此时线程 4 执行 cache[4]+=cache[4+4],
    //但如果线程 0 在读取之前线程 4 已经完成了对 cache[4] 的写入, 那么结果就会产生错误. 而上述 kernel 4.1 的代码则可以避免此问题.

    //线程 0 在执行 cache[0]+=cache[0+4] 时需读取 cache[4], 此时线程 4 执行 cache[4]+=cache[4+4],
    // 但如果线程 0 在读取之前线程 4 已经完成了对 cache[4] 的写入, 那么结果就会产生错误

    ///这个问题v1版本好像也有，为什么不使用增加int的方式解决呢。
    //确实v1版本是有问题的，详见https://zhuanlan.zhihu.com/p/426978026下的u wen评论
    // cache[tid] += cache[tid+32];__syncwarp();
    // cache[tid] += cache[tid+16];__syncwarp();
    // cache[tid] += cache[tid+8];__syncwarp();
    // cache[tid] += cache[tid+4];__syncwarp();
    // cache[tid] += cache[tid+2];__syncwarp();
    // cache[tid] += cache[tid+1];__syncwarp();
    int v = cache[tid];
    v += cache[tid+32]; 
    __syncwarp();
    cache[tid] = v;     
    __syncwarp();
    v += cache[tid+16]; 
    __syncwarp();
    cache[tid] = v;     
    __syncwarp();
    v += cache[tid+8];  
    __syncwarp();
    cache[tid] = v;     
    __syncwarp();
    v += cache[tid+4];  
    __syncwarp();
    cache[tid] = v;     
    __syncwarp();
    v += cache[tid+2];  
    __syncwarp();
    cache[tid] = v;     
    __syncwarp();
    v += cache[tid+1];  
    __syncwarp();
    cache[tid] = v;

}
//__shfl_down_sync,它正是一个 warp 层次的原语, 用于获取 warp 内其他线程变量的函数, 它的优势在于可以直接在寄存器间进行变量交换而无需通过共享内存,
//而且正如函数名的 sync, 每次函数调用都会进行 warp 内线程的同步, 保证 warp 内线程的步调一致.
#define FULL_MASK 0xffffffff
__device__ void warpReduce_v3(float* cache, unsigned int tid){
    int v = cache[tid] + cache[tid+32];
    v += __shfl_down_sync(FULL_MASK,v,16);
    v += __shfl_down_sync(FULL_MASK,v,8);
    v += __shfl_down_sync(FULL_MASK,v,4);
    v += __shfl_down_sync(FULL_MASK,v,2);
    v += __shfl_down_sync(FULL_MASK,v,1);
    cache[tid] = v;
}

__global__ void reduce_v4(float * g_idata,float *g_odata){
    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x*blockDim.x*2 + threadIdx.x;
    __shared__ sdata[BLOCK_SIZE];
    sdata[tid] = g_idata[gid] + g_idata[gid+blockDim.x];
    __syncthreads();
    ///s这在里是间隔数量，也是活跃线程的数量。
    ///当活跃线程数量小于32，既最后只剩下一个warp执行时，会导致warp divergence情况加剧，因此选择直接展开最后一个warp
    for(int s = blockDim.x/2; s>32;s>>=1){
        if(tid < s){
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if(tid<32)warpReduce_v2(sdata,tid);
    if(tid == 0)g_odata[blockIdx.x] = sdata[0];
}

template<unsigned block_size>
__device__ void warpReduce_v4(float* cache, unsigned int tid){
    int v = cache[tid];
    if(blockSize >= 64) v += __shfl_down_sync(FULL_MASK,v,32);
    if(blockSize >= 32)v += __shfl_down_sync(FULL_MASK,v,16);
    if(blockSize >= 16)v += __shfl_down_sync(FULL_MASK,v,8);
    if(blockSize >= 8)v += __shfl_down_sync(FULL_MASK,v,4);
    if(blockSize >= 4)v += __shfl_down_sync(FULL_MASK,v,2);
    if(blockSize >= 2)v += __shfl_down_sync(FULL_MASK,v,1);
    cache[tid] = v;
}
//针对不同的block_size进行完全展开,根据实际blocksize的大小转换跳过不符合的blocksize条件语句
//减少指令的产生，提高执行精度。
//注意这里的if(block > x)都可以在编译器阶段判断得到。
template <unsigned block_size>
__global__ void reduce_v5(float * g_idata, float * g_odata){

    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x*blockDim.x*2 + threadIdx.x;
    __shared__ sdata[BLOCK_SIZE];
    sdata[tid] = g_idata[gid] + g_idata[gid+blockDim.x];
    __syncthreads();
    //注意下面几个if的顺序是不能颠倒的，也不能换成else if
    //处理完这个if后数据归约到0~512thread上
    if(block_size == 1024){
        if(tid<512) sdata[tid] += sdata[tid+512];
        __syncthreads();
    }
    //处理完这个if后数据归约到0~256thread上
    if(block_size >= 512){
        if(tid<256) sdata[tid] += sdata[tid+256];
        __syncthreads();
    }
    if(block_size >= 256){
        if(tid<128) sdata[tid] += sdata[tid+128];
        __syncthreads();
    }
    if(block_size >= 128){
        if(tid<64) sdata[tid] += sdata[tid+64];
        __syncthreads();
    }
    if(block_size >= 64){
        if(tid<32) sdata[tid] += sdata[tid+32];
        __syncthreads();
    }
    if(tid<32)warpReduce_v4<block_size>(sdata,tid);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

//很重要的一点是: 每个线程更多的工作可以提供更好的延迟隐藏.
//这有点类似于 kernel 3 的做法, kernel 3 只是让空闲的线程多做了 1 次归约计算, 而实际上我们可以做更多次,
//而这样就会导致需要的线程块数 grid_size 成倍减少. 因此, 这两者实际上是一回事.
template <unsigned block_size, unsigned NUM_PER_THREAD>
__global__ void reduce_v6(float * g_idata, float * g_odata){

    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x*blockDim.x*NUM_PER_THREAD + threadIdx.x;
    __shared__ sdata[BLOCK_SIZE];
    sdata[tid] = 0;
    #pragma unroll
    for(int iter = 0; iter < NUM_PER_THREAD; ++iter){
        sdata[tid] += g_idata[i + iter * blockSize];
    }
    __syncthreads();
    //注意下面几个if的顺序是不能颠倒的，也不能换成else if
    //处理完这个if后数据归约到0~512thread上
    if(block_size == 1024){
        if(tid<512) sdata[tid] += sdata[tid+512];
        __syncthreads();
    }
    //处理完这个if后数据归约到0~256thread上
    if(block_size >= 512){
        if(tid<256) sdata[tid] += sdata[tid+256];
        __syncthreads();
    }
    if(block_size >= 256){
        if(tid<128) sdata[tid] += sdata[tid+128];
        __syncthreads();
    }
    if(block_size >= 128){
        if(tid<64) sdata[tid] += sdata[tid+64];
        __syncthreads();
    }
    if(block_size >= 64){
        if(tid<32) sdata[tid] += sdata[tid+32];
        __syncthreads();
    }
    if(tid<32)warpReduce_v4<block_size>(sdata,tid);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
template<unsigned block_size>
__device__ __forceinline__  void warpReduce_v5(float sum){
    if (blockSize >= 32)
        sum += __shfl_down_sync(0xffffffff, sum, 16);  // 0-16, 1-17, 2-18, etc.
    if (blockSize >= 16)
        sum += __shfl_down_sync(0xffffffff, sum, 8);  // 0-8, 1-9, 2-10, etc.
    if (blockSize >= 8)
        sum += __shfl_down_sync(0xffffffff, sum, 4);  // 0-4, 1-5, 2-6, etc.
    if (blockSize >= 4)
        sum += __shfl_down_sync(0xffffffff, sum, 2);  // 0-2, 1-3, 4-6, 5-7, etc.
    if (blockSize >= 2)
        sum += __shfl_down_sync(0xffffffff, sum, 1);  // 0-1, 2-3, 4-5, etc.
    return sum;
}
//这应该是block reduce的概念
//首先要弄清楚lane的概念，lane中文意思为车道，在cuda中表示一个warp中的线程数量。
//1D_block中一个lane内线程索引为lane_index[0,warpsize-1]。一个block中会有多个lane，warp_id=threadIdx.x/warpsize.最多有1024/warpsize=32个lane
//第一次调用warpReduce_v5 分别对各个warp进行计算，非常精巧。最后在每个warp的lid=0的位置得到规约结果。
//第二次调用warpreduce之前，将各个warp的规约值放到第一个warp的线程中，在进行最后一次规约。前面提到过一个block最多只能放下32个lane，因此这个做法是合理的。
//需要注意的是①第二次规约的时候，由于整体可能并没有32个warp，因此不够的值需要补零。

//NUM_PER_THREAD表示kernel从gmem中读取数据读取了几个数据。
//先前的 kernel 都是每个线程读取 1 个数据, 现在每个线程变成了读取NUM_PER_THREAD 个数据, 并进行一次归约操作. 
//Input:N:1024 
//Block_size:B:128
//NUM_PER_THREAD:4
//grid_size:N/B/NUM_PER_THREAD：1024/128/4=2
//具体请看图：reduce从gmem到smem

//这种思想和上面的还不太一样，上面的优化方法只有最后一个warp才使用warpreduce，而这个算法就是调用的两次warp就结束了。
#define WARP_SIZE 32
template <unsigned block_size, unsigned NUM_PER_THREAD>
__global__ void reduce_v7(float * g_idata, float * g_odata){
    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x*blockDim.x*NUM_PER_THREAD + threadIdx.x;
    float sum = 0;
    #pragma unroll
    //这个地方是为了在从g_idata中读取数据时增加计算密度用的。
    //具体可以看kernel3
    //先前的 kernel 都是每个线程读取 1 个数据, 现在每个线程变成了读取NUM_PER_THREAD 个数据, 并进行一次归约操作. 
    for(int iter = 0; iter < NUM_PER_THREAD; ++iter){
        sum += g_idata[gid + iter * blockSize];
    }
    int lid = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;
    int num_warps = block_size/WARP_SIZE;
    sum = warpReduce_v5<block_size>(sum);

    __shared__ float shared[WARP_SIZE];
    if(lid == 0){
        shared[wid] = sum;
    }
    __syncthreads();
    sum = (tid<num_warps)? shared[lid] : 0;
    sum = warpReduce_v5<num_warps>(sum);
    if(tid == 0) g_odata[blockIdx.x] = sum;
}

int main(){


    float A[N], B[N], C[N];
    for(int i = 0; i < N;i++){
        A[i] = i;
        B[i] = i;
        C[i] = 0;
    }
    int * d_A,* d_B,* d_C;
    int size_A = sizeof(float)*N;
    int size_B = sizeof(float)*N;
    int size_C = sizeof(float)*N;
    cudaMalloc(&d_A,sizeof(float)*N);
    cudaMalloc(&d_B,sizeof(float)*N);
    cudaMalloc(&d_C,sizeof(float)*N);

    cudaMemcpy(d_A,A,size_A,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B,size_B,cudaMemcpyHostToDevice);


    dim block(BLOCK_SIZE);
    dim grid(N/BLOCK_SIZE);

}
