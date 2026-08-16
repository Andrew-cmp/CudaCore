#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 128;

// Warp级别的Softmax实现
__device__ void warp_softmax(float* __restrict__ x, int size, int warp_id, int lane_id) {
    // 找最大值
    float max_val = (lane_id < size) ? x[lane_id] : -FLT_MAX;
    for (int i = lane_id + WARP_SIZE; i < size; i += WARP_SIZE) {
        max_val = fmaxf(max_val, x[i]);
    }
    
    // Warp内归约求最大值
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }
    max_val = __shfl_sync(0xffffffff, max_val, 0);
    
    // 计算exp并求和
    float sum = 0.0f;
    for (int i = lane_id; i < size; i += WARP_SIZE) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    
    // Warp内归约求和
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    sum = __shfl_sync(0xffffffff, sum, 0);
    
    // 归一化
    for (int i = lane_id; i < size; i += WARP_SIZE) {
        x[i] /= sum;
    }
}

// Prefill阶段的多头自注意力kernel
// 每个block处理一个头的一个query位置
__global__ void prefill_mha_kernel(
    float* __restrict__ query,      // [seq_len, head_num, head_size]
    float* __restrict__ key,        // [seq_len, head_num, head_size] 
    float* __restrict__ value,      // [seq_len, head_num, head_size]
    float* __restrict__ output,     // [seq_len, head_num, head_size]
    float* __restrict__ score_buf,  // [seq_len, head_num, seq_len] 临时存储attention scores
    int seq_len,
    int head_num, 
    int head_size
) {
    int q_pos = blockIdx.x;           // query位置
    int head = blockIdx.y;            // 注意力头索引
    int tid = threadIdx.x;            // 线程索引
    
    if (q_pos >= seq_len || head >= head_num) return;
    
    extern __shared__ float smem[];
    float* s_query = smem;                              // [head_size]
    float* s_scores = smem + head_size;                 // [seq_len]
    
    float scale = 1.0f / sqrtf(float(head_size));
    
    // 当前query向量的起始地址
    float* query_vec = query + q_pos * head_num * head_size + head * head_size;
    
    // 加载query到共享内存
    for (int i = tid; i < head_size; i += blockDim.x) {
        s_query[i] = query_vec[i];
    }
    __syncthreads();
    
    // 计算attention scores: Q * K^T
    for (int k_pos = tid; k_pos <= q_pos; k_pos += blockDim.x) {  // 因果掩码：只看当前及之前的位置
        float* key_vec = key + k_pos * head_num * head_size + head * head_size;
        
        float score = 0.0f;
        // 向量化点积计算
        for (int i = 0; i < head_size; i += 4) {
            if (i + 3 < head_size) {
                float4 q_val = *reinterpret_cast<float4*>(s_query + i);
                float4 k_val = *reinterpret_cast<float4*>(key_vec + i);
                score += q_val.x * k_val.x + q_val.y * k_val.y + 
                         q_val.z * k_val.z + q_val.w * k_val.w;
            } else {
                // 处理不能被4整除的部分
                for (int j = i; j < head_size; j++) {
                    score += s_query[j] * key_vec[j];
                }
                break;
            }
        }
        
        score *= scale;
        s_scores[k_pos] = score;
        
        // 对于k_pos > q_pos的位置，设置为负无穷（因果掩码）
        if (k_pos > q_pos) {
            s_scores[k_pos] = -FLT_MAX;
        }
    }
    __syncthreads();
    
    // Softmax计算（只对有效长度q_pos+1）
    // 使用warp级别的softmax
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    
    if (warp_id == 0) {  // 只用第一个warp做softmax
        warp_softmax(s_scores, q_pos + 1, warp_id, lane_id);
    }
    __syncthreads();
    
    // 计算加权的value: Attention * V
    float* output_vec = output + q_pos * head_num * head_size + head * head_size;
    
    for (int i = tid; i < head_size; i += blockDim.x) {
        float weighted_value = 0.0f;
        
        for (int k_pos = 0; k_pos <= q_pos; k_pos++) {
            float* value_vec = value + k_pos * head_num * head_size + head * head_size;
            weighted_value += s_scores[k_pos] * value_vec[i];
        }
        
        output_vec[i] = weighted_value;
    }
}

// 批量处理版本：每个block处理多个query位置
__global__ void prefill_mha_batch_kernel(
    float* __restrict__ query,      // [seq_len, head_num, head_size]
    float* __restrict__ key,        // [seq_len, head_num, head_size]
    float* __restrict__ value,      // [seq_len, head_num, head_size]
    float* __restrict__ output,     // [seq_len, head_num, head_size]
    int seq_len,
    int head_num,
    int head_size
) {
    int head = blockIdx.x;           // 注意力头
    int tid = threadIdx.x;           // 线程索引
    
    if (head >= head_num) return;
    
    extern __shared__ float smem[];
    float* s_scores = smem;          // [seq_len] 存储当前处理的scores
    
    float scale = 1.0f / sqrtf(float(head_size));
    
    // 每个线程处理多个query位置
    for (int q_pos = 0; q_pos < seq_len; q_pos++) {
        float* query_vec = query + q_pos * head_num * head_size + head * head_size;
        float* output_vec = output + q_pos * head_num * head_size + head * head_size;
        
        // 计算attention scores
        for (int k_pos = tid; k_pos <= q_pos; k_pos += blockDim.x) {
            float* key_vec = key + k_pos * head_num * head_size + head * head_size;
            
            float score = 0.0f;
            for (int i = 0; i < head_size; i++) {
                score += query_vec[i] * key_vec[i];
            }
            
            s_scores[k_pos] = score * scale;
        }
        
        // 填充掩码位置
        for (int k_pos = q_pos + 1 + tid; k_pos < seq_len; k_pos += blockDim.x) {
            s_scores[k_pos] = -FLT_MAX;
        }
        __syncthreads();
        
        // Softmax (简化版本，使用第一个warp)
        if (tid < WARP_SIZE) {
            warp_softmax(s_scores, q_pos + 1, 0, tid);
        }
        __syncthreads();
        
        // 计算输出
        for (int i = tid; i < head_size; i += blockDim.x) {
            float result = 0.0f;
            for (int k_pos = 0; k_pos <= q_pos; k_pos++) {
                float* value_vec = value + k_pos * head_num * head_size + head * head_size;
                result += s_scores[k_pos] * value_vec[i];
            }
            output_vec[i] = result;
        }
        __syncthreads();
    }
}

// Host端测试函数
void test_prefill_mha() {
    // 测试参数
    const int seq_len = 64;
    const int head_num = 32;
    const int head_size = 128;
    
    // 分配host内存
    size_t qkv_size = seq_len * head_num * head_size * sizeof(float);
    size_t score_size = seq_len * head_num * seq_len * sizeof(float);
    
    std::vector<float> h_query(seq_len * head_num * head_size);
    std::vector<float> h_key(seq_len * head_num * head_size);
    std::vector<float> h_value(seq_len * head_num * head_size);
    std::vector<float> h_output(seq_len * head_num * head_size);
    
    // 随机初始化
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (auto& v : h_query) v = dist(rng);
    for (auto& v : h_key) v = dist(rng);
    for (auto& v : h_value) v = dist(rng);
    
    // 分配device内存
    float *d_query, *d_key, *d_value, *d_output, *d_score_buf;
    cudaMalloc(&d_query, qkv_size);
    cudaMalloc(&d_key, qkv_size);
    cudaMalloc(&d_value, qkv_size);
    cudaMalloc(&d_output, qkv_size);
    cudaMalloc(&d_score_buf, score_size);
    
    // 拷贝数据到GPU
    cudaMemcpy(d_query, h_query.data(), qkv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_key, h_key.data(), qkv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_value, h_value.data(), qkv_size, cudaMemcpyHostToDevice);
    
    // 启动kernel
    dim3 grid(seq_len, head_num);  // 每个block处理一个(q_pos, head)
    dim3 block(BLOCK_SIZE);
    size_t smem_size = (head_size + seq_len) * sizeof(float);
    
    printf("启动Prefill MHA kernel...\n");
    printf("Grid: (%d, %d), Block: %d, Shared Memory: %zu bytes\n", 
           grid.x, grid.y, block.x, smem_size);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    prefill_mha_kernel<<<grid, block, smem_size>>>(
        d_query, d_key, d_value, d_output, d_score_buf,
        seq_len, head_num, head_size
    );
    cudaEventRecord(stop);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
        return;
    }
    
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    // 拷贝结果回host
    cudaMemcpy(h_output.data(), d_output, qkv_size, cudaMemcpyDeviceToHost);
    
    printf("Kernel执行时间: %.3f ms\n", ms);
    printf("前8个输出值: ");
    for (int i = 0; i < 8; i++) {
        printf("%.6f ", h_output[i]);
    }
    printf("\n");
    
    // 清理资源
    cudaFree(d_query);
    cudaFree(d_key);
    cudaFree(d_value);
    cudaFree(d_output);
    cudaFree(d_score_buf);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    printf("=== Prefill阶段多头自注意力测试 ===\n");
    test_prefill_mha();
    return 0;
} 