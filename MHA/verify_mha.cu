#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <cassert>

constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 128;
constexpr float EPSILON = 1e-5f;

// 从prefill_mha.cu复制的kernel代码
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

__global__ void prefill_mha_kernel(
    float* __restrict__ query,
    float* __restrict__ key,
    float* __restrict__ value,
    float* __restrict__ output,
    float* __restrict__ score_buf,
    int seq_len,
    int head_num, 
    int head_size
) {
    int q_pos = blockIdx.x;
    int head = blockIdx.y;
    int tid = threadIdx.x;
    
    if (q_pos >= seq_len || head >= head_num) return;
    
    extern __shared__ float smem[];
    float* s_query = smem;
    float* s_scores = smem + head_size;
    
    float scale = 1.0f / sqrtf(float(head_size));
    
    float* query_vec = query + q_pos * head_num * head_size + head * head_size;
    
    // 加载query到共享内存
    for (int i = tid; i < head_size; i += blockDim.x) {
        s_query[i] = query_vec[i];
    }
    __syncthreads();
    
    // 计算attention scores
    for (int k_pos = tid; k_pos <= q_pos; k_pos += blockDim.x) {
        float* key_vec = key + k_pos * head_num * head_size + head * head_size;
        
        float score = 0.0f;
        for (int i = 0; i < head_size; i += 4) {
            if (i + 3 < head_size) {
                float4 q_val = *reinterpret_cast<float4*>(s_query + i);
                float4 k_val = *reinterpret_cast<float4*>(key_vec + i);
                score += q_val.x * k_val.x + q_val.y * k_val.y + 
                         q_val.z * k_val.z + q_val.w * k_val.w;
            } else {
                for (int j = i; j < head_size; j++) {
                    score += s_query[j] * key_vec[j];
                }
                break;
            }
        }
        
        score *= scale;
        s_scores[k_pos] = score;
    }
    
    // 填充掩码位置
    for (int k_pos = q_pos + 1 + tid; k_pos < seq_len; k_pos += blockDim.x) {
        s_scores[k_pos] = -FLT_MAX;
    }
    __syncthreads();
    
    // Softmax
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    if (warp_id == 0) {
        warp_softmax(s_scores, q_pos + 1, warp_id, lane_id);
    }
    __syncthreads();
    
    // 计算加权的value
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

// CPU参考实现
void cpu_softmax(std::vector<float>& x, int size) {
    // 找最大值
    float max_val = *std::max_element(x.begin(), x.begin() + size);
    
    // 计算exp和sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    
    // 归一化
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void cpu_mha_reference(
    const std::vector<float>& query,
    const std::vector<float>& key,
    const std::vector<float>& value,
    std::vector<float>& output,
    int seq_len,
    int head_num,
    int head_size
) {
    float scale = 1.0f / sqrtf(float(head_size));
    
    for (int head = 0; head < head_num; head++) {
        for (int q_pos = 0; q_pos < seq_len; q_pos++) {
            // 获取当前query向量
            std::vector<float> q_vec(head_size);
            for (int i = 0; i < head_size; i++) {
                q_vec[i] = query[q_pos * head_num * head_size + head * head_size + i];
            }
            
            // 计算attention scores
            std::vector<float> scores(seq_len, -FLT_MAX);
            for (int k_pos = 0; k_pos <= q_pos; k_pos++) {  // 因果掩码
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    float k_val = key[k_pos * head_num * head_size + head * head_size + i];
                    score += q_vec[i] * k_val;
                }
                scores[k_pos] = score * scale;
            }
            
            // Softmax
            cpu_softmax(scores, q_pos + 1);
            
            // 计算加权value
            for (int i = 0; i < head_size; i++) {
                float weighted_value = 0.0f;
                for (int k_pos = 0; k_pos <= q_pos; k_pos++) {
                    float v_val = value[k_pos * head_num * head_size + head * head_size + i];
                    weighted_value += scores[k_pos] * v_val;
                }
                output[q_pos * head_num * head_size + head * head_size + i] = weighted_value;
            }
        }
    }
}

// 比较两个向量的相似度
bool compare_vectors(const std::vector<float>& a, const std::vector<float>& b, 
                    float tolerance = EPSILON) {
    if (a.size() != b.size()) {
        printf("向量大小不匹配: %zu vs %zu\n", a.size(), b.size());
        return false;
    }
    
    float max_diff = 0.0f;
    float max_rel_diff = 0.0f;
    int diff_count = 0;
    
    for (size_t i = 0; i < a.size(); i++) {
        float diff = fabsf(a[i] - b[i]);
        float rel_diff = (fabsf(a[i]) > EPSILON) ? diff / fabsf(a[i]) : diff;
        
        if (diff > tolerance) {
            diff_count++;
            if (diff_count <= 10) {  // 只打印前10个差异
                printf("位置 %zu: CPU=%.6f, GPU=%.6f, diff=%.6f, rel_diff=%.6f\n", 
                       i, a[i], b[i], diff, rel_diff);
            }
        }
        
        max_diff = fmaxf(max_diff, diff);
        max_rel_diff = fmaxf(max_rel_diff, rel_diff);
    }
    
    printf("最大绝对差异: %.6f\n", max_diff);
    printf("最大相对差异: %.6f\n", max_rel_diff);
    printf("超出容差的元素数量: %d / %zu\n", diff_count, a.size());
    
    return diff_count == 0;
}

// 测试用例1：小规模精确测试
bool test_small_case() {
    printf("\n=== 测试用例1: 小规模精确测试 ===\n");
    
    const int seq_len = 4;
    const int head_num = 2;
    const int head_size = 8;  // 必须是4的倍数
    
    // 创建简单的测试数据
    std::vector<float> h_query(seq_len * head_num * head_size);
    std::vector<float> h_key(seq_len * head_num * head_size);
    std::vector<float> h_value(seq_len * head_num * head_size);
    
    // 填充简单的测试数据
    for (int i = 0; i < h_query.size(); i++) {
        h_query[i] = (i % 10) * 0.1f;
        h_key[i] = ((i + 3) % 10) * 0.1f;
        h_value[i] = ((i + 7) % 10) * 0.1f;
    }
    
    // CPU参考计算
    std::vector<float> cpu_output(seq_len * head_num * head_size, 0.0f);
    cpu_mha_reference(h_query, h_key, h_value, cpu_output, seq_len, head_num, head_size);
    
    // GPU计算
    float *d_query, *d_key, *d_value, *d_output, *d_score_buf;
    size_t qkv_size = seq_len * head_num * head_size * sizeof(float);
    size_t score_size = seq_len * head_num * seq_len * sizeof(float);
    
    cudaMalloc(&d_query, qkv_size);
    cudaMalloc(&d_key, qkv_size);
    cudaMalloc(&d_value, qkv_size);
    cudaMalloc(&d_output, qkv_size);
    cudaMalloc(&d_score_buf, score_size);
    
    cudaMemcpy(d_query, h_query.data(), qkv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_key, h_key.data(), qkv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_value, h_value.data(), qkv_size, cudaMemcpyHostToDevice);
    
    dim3 grid(seq_len, head_num);
    dim3 block(BLOCK_SIZE);
    size_t smem_size = (head_size + seq_len) * sizeof(float);
    
    prefill_mha_kernel<<<grid, block, smem_size>>>(
        d_query, d_key, d_value, d_output, d_score_buf,
        seq_len, head_num, head_size
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel错误: %s\n", cudaGetErrorString(err));
        return false;
    }
    
    std::vector<float> gpu_output(seq_len * head_num * head_size);
    cudaMemcpy(gpu_output.data(), d_output, qkv_size, cudaMemcpyDeviceToHost);
    
    bool passed = compare_vectors(cpu_output, gpu_output, 1e-4f);
    
    cudaFree(d_query);
    cudaFree(d_key);
    cudaFree(d_value);
    cudaFree(d_output);
    cudaFree(d_score_buf);
    
    printf("小规模测试: %s\n", passed ? "通过" : "失败");
    return passed;
}

// 测试用例2：随机数据测试
bool test_random_case() {
    printf("\n=== 测试用例2: 随机数据测试 ===\n");
    
    const int seq_len = 16;
    const int head_num = 4;
    const int head_size = 32;
    
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    std::vector<float> h_query(seq_len * head_num * head_size);
    std::vector<float> h_key(seq_len * head_num * head_size);
    std::vector<float> h_value(seq_len * head_num * head_size);
    
    for (auto& v : h_query) v = dist(rng);
    for (auto& v : h_key) v = dist(rng);
    for (auto& v : h_value) v = dist(rng);
    
    // CPU参考计算
    std::vector<float> cpu_output(seq_len * head_num * head_size, 0.0f);
    auto start = std::chrono::high_resolution_clock::now();
    cpu_mha_reference(h_query, h_key, h_value, cpu_output, seq_len, head_num, head_size);
    auto end = std::chrono::high_resolution_clock::now();
    float cpu_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    // GPU计算
    float *d_query, *d_key, *d_value, *d_output, *d_score_buf;
    size_t qkv_size = seq_len * head_num * head_size * sizeof(float);
    size_t score_size = seq_len * head_num * seq_len * sizeof(float);
    
    cudaMalloc(&d_query, qkv_size);
    cudaMalloc(&d_key, qkv_size);
    cudaMalloc(&d_value, qkv_size);
    cudaMalloc(&d_output, qkv_size);
    cudaMalloc(&d_score_buf, score_size);
    
    cudaMemcpy(d_query, h_query.data(), qkv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_key, h_key.data(), qkv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_value, h_value.data(), qkv_size, cudaMemcpyHostToDevice);
    
    dim3 grid(seq_len, head_num);
    dim3 block(BLOCK_SIZE);
    size_t smem_size = (head_size + seq_len) * sizeof(float);
    
    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);
    
    cudaEventRecord(gpu_start);
    prefill_mha_kernel<<<grid, block, smem_size>>>(
        d_query, d_key, d_value, d_output, d_score_buf,
        seq_len, head_num, head_size
    );
    cudaEventRecord(gpu_stop);
    cudaEventSynchronize(gpu_stop);
    
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, gpu_start, gpu_stop);
    
    std::vector<float> gpu_output(seq_len * head_num * head_size);
    cudaMemcpy(gpu_output.data(), d_output, qkv_size, cudaMemcpyDeviceToHost);
    
    bool passed = compare_vectors(cpu_output, gpu_output, 1e-3f);
    
    printf("CPU时间: %.3f ms\n", cpu_time);
    printf("GPU时间: %.3f ms\n", gpu_time);
    printf("加速比: %.2fx\n", cpu_time / gpu_time);
    printf("随机数据测试: %s\n", passed ? "通过" : "失败");
    
    cudaFree(d_query);
    cudaFree(d_key);
    cudaFree(d_value);
    cudaFree(d_output);
    cudaFree(d_score_buf);
    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_stop);
    
    return passed;
}

// 测试用例3：边界情况测试
bool test_edge_cases() {
    printf("\n=== 测试用例3: 边界情况测试 ===\n");
    
    bool all_passed = true;
    
    // 测试1: 序列长度为1
    {
        printf("测试单token序列...\n");
        const int seq_len = 1, head_num = 2, head_size = 16;
        
        std::vector<float> h_query(seq_len * head_num * head_size, 1.0f);
        std::vector<float> h_key(seq_len * head_num * head_size, 1.0f);
        std::vector<float> h_value(seq_len * head_num * head_size, 2.0f);
        
        std::vector<float> cpu_output(seq_len * head_num * head_size);
        cpu_mha_reference(h_query, h_key, h_value, cpu_output, seq_len, head_num, head_size);
        
        // 单token情况下，attention权重应该是1.0，输出应该等于value
        bool single_token_passed = true;
        for (int i = 0; i < head_size; i++) {
            if (fabsf(cpu_output[i] - 2.0f) > EPSILON) {
                single_token_passed = false;
                break;
            }
        }
        
        printf("单token测试: %s\n", single_token_passed ? "通过" : "失败");
        all_passed &= single_token_passed;
    }
    
    // 测试2: 全零输入
    {
        printf("测试全零输入...\n");
        const int seq_len = 4, head_num = 2, head_size = 8;
        
        std::vector<float> h_query(seq_len * head_num * head_size, 0.0f);
        std::vector<float> h_key(seq_len * head_num * head_size, 0.0f);
        std::vector<float> h_value(seq_len * head_num * head_size, 1.0f);
        
        std::vector<float> cpu_output(seq_len * head_num * head_size);
        cpu_mha_reference(h_query, h_key, h_value, cpu_output, seq_len, head_num, head_size);
        
        // 全零Q和K的情况下，attention权重应该是均匀分布
        // 对于位置i，应该有1/(i+1)的权重分布
        bool zero_input_passed = true;
        for (int q_pos = 0; q_pos < seq_len; q_pos++) {
            float expected_value = 1.0f;  // 因为value都是1.0
            float actual_value = cpu_output[q_pos * head_num * head_size];
            if (fabsf(actual_value - expected_value) > 1e-3f) {
                zero_input_passed = false;
                printf("位置%d: 期望%.6f, 实际%.6f\n", q_pos, expected_value, actual_value);
                break;
            }
        }
        
        printf("全零输入测试: %s\n", zero_input_passed ? "通过" : "失败");
        all_passed &= zero_input_passed;
    }
    
    return all_passed;
}

// 性能基准测试
void benchmark_performance() {
    printf("\n=== 性能基准测试 ===\n");
    
    struct TestConfig {
        int seq_len, head_num, head_size;
        const char* name;
    };
    
    TestConfig configs[] = {
        {64, 8, 64, "小规模 (64x8x64)"},
        {128, 12, 64, "中规模 (128x12x64)"},
        {256, 16, 64, "大规模 (256x16x64)"},
        {512, 8, 128, "长序列 (512x8x128)"}
    };
    
    for (const auto& config : configs) {
        printf("\n测试配置: %s\n", config.name);
        
        size_t qkv_size = config.seq_len * config.head_num * config.head_size * sizeof(float);
        size_t score_size = config.seq_len * config.head_num * config.seq_len * sizeof(float);
        
        printf("内存使用: QKV %.2f MB, Score Buffer %.2f MB\n", 
               qkv_size * 3 / 1024.0f / 1024.0f,
               score_size / 1024.0f / 1024.0f);
        
        // 分配GPU内存
        float *d_query, *d_key, *d_value, *d_output, *d_score_buf;
        cudaMalloc(&d_query, qkv_size);
        cudaMalloc(&d_key, qkv_size);
        cudaMalloc(&d_value, qkv_size);
        cudaMalloc(&d_output, qkv_size);
        cudaMalloc(&d_score_buf, score_size);
        
        // 初始化随机数据
        std::vector<float> h_data(config.seq_len * config.head_num * config.head_size);
        std::mt19937 rng(123);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto& v : h_data) v = dist(rng);
        
        cudaMemcpy(d_query, h_data.data(), qkv_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_key, h_data.data(), qkv_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_value, h_data.data(), qkv_size, cudaMemcpyHostToDevice);
        
        dim3 grid(config.seq_len, config.head_num);
        dim3 block(BLOCK_SIZE);
        size_t smem_size = (config.head_size + config.seq_len) * sizeof(float);
        
        // 预热
        for (int i = 0; i < 3; i++) {
            prefill_mha_kernel<<<grid, block, smem_size>>>(
                d_query, d_key, d_value, d_output, d_score_buf,
                config.seq_len, config.head_num, config.head_size
            );
        }
        cudaDeviceSynchronize();
        
        // 性能测试
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        const int num_runs = 100;
        cudaEventRecord(start);
        for (int i = 0; i < num_runs; i++) {
            prefill_mha_kernel<<<grid, block, smem_size>>>(
                d_query, d_key, d_value, d_output, d_score_buf,
                config.seq_len, config.head_num, config.head_size
            );
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float total_time;
        cudaEventElapsedTime(&total_time, start, stop);
        float avg_time = total_time / num_runs;
        
        // 计算FLOPS
        long long ops = (long long)config.seq_len * config.seq_len * config.head_num * config.head_size * 4; // QK^T + softmax + AV
        float gflops = ops / (avg_time * 1e6);
        
        printf("平均执行时间: %.3f ms\n", avg_time);
        printf("估计GFLOPS: %.2f\n", gflops);
        
        cudaFree(d_query);
        cudaFree(d_key);
        cudaFree(d_value);
        cudaFree(d_output);
        cudaFree(d_score_buf);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
}

int main() {
    printf("=== MHA Kernel 正确性验证程序 ===\n");
    
    // 检查CUDA设备
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("错误: 未找到CUDA设备\n");
        return -1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("使用GPU: %s\n", prop.name);
    printf("计算能力: %d.%d\n", prop.major, prop.minor);
    printf("共享内存: %zu KB\n", prop.sharedMemPerBlock / 1024);
    
    bool all_tests_passed = true;
    
    // 运行测试用例
    all_tests_passed &= test_small_case();
    all_tests_passed &= test_random_case();
    all_tests_passed &= test_edge_cases();
    
    // 性能测试
    benchmark_performance();
    
    printf("\n=== 测试总结 ===\n");
    printf("所有测试: %s\n", all_tests_passed ? "通过" : "失败");
    
    if (all_tests_passed) {
        printf("✅ MHA kernel实现正确!\n");
        return 0;
    } else {
        printf("❌ 发现错误，请检查实现\n");
        return -1;
    }
} 