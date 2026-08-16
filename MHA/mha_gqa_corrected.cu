#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>

// 示例kernel函数声明（需要根据你的具体实现调整）
__global__ void gemm_qk_kernel(float* query, float* key, float* scores,
                              int q_len, int heads_per_group, int head_dim, int kv_len);
__global__ void softmax_kernel(float* scores, int q_len, int heads_per_group, int kv_len);
__global__ void gemm_sv_kernel(float* scores, float* value, float* output,
                              int q_len, int heads_per_group, int head_dim, int kv_len);

int main() {
    // 参数设置 - 基于Python代码的逻辑
    int q_len = 1;           // 序列长度
    int kv_len = 1024;       // key/value序列长度
    int num_groups = 4;      // GQA组数
    int hidden_size = 4096;

    int q_head_num = 32;     // query头数
    int head_dim = hidden_size / q_head_num; // 128
    int heads_per_group = q_head_num / num_groups; // 8 (每组的query头数)

    // 在GQA中，KV头数等于组数
    int kv_head_num = num_groups; // 4
    int kv_head_dim = head_dim;   // 128 (与query头维度相同)

    printf("GQA配置参数:\n");
    printf("q_len=%d, kv_len=%d, num_groups=%d\n", q_len, kv_len, num_groups);
    printf("q_head_num=%d, head_dim=%d, heads_per_group=%d\n", q_head_num, head_dim, heads_per_group);
    printf("kv_head_num=%d, kv_head_dim=%d\n", kv_head_num, kv_head_dim);

    // 分配host内存
    // Query: [q_len, q_head_num, head_dim]
    float* query = new float[q_len * q_head_num * head_dim];
    // Key: [kv_len, kv_head_num, kv_head_dim] 
    float* key = new float[kv_len * kv_head_num * kv_head_dim];
    // Value: [kv_len, kv_head_num, kv_head_dim]
    float* value = new float[kv_len * kv_head_num * kv_head_dim];
    // Output: [q_len, q_head_num, head_dim]
    float* output = new float[q_len * q_head_num * head_dim];

    // 初始化数据
    srand(42);
    for (int i = 0; i < q_len * q_head_num * head_dim; i++) {
        query[i] = (rand() / (float)RAND_MAX - 0.5f) * 2.0f; // [-1, 1]
    }
    for (int i = 0; i < kv_len * kv_head_num * kv_head_dim; i++) {
        key[i] = (rand() / (float)RAND_MAX - 0.5f) * 2.0f;
        value[i] = (rand() / (float)RAND_MAX - 0.5f) * 2.0f;
    }

    // 分配device内存
    float *d_query, *d_key, *d_value, *d_output;
    cudaMalloc(&d_query, q_len * q_head_num * head_dim * sizeof(float));
    cudaMalloc(&d_key, kv_len * kv_head_num * kv_head_dim * sizeof(float));
    cudaMalloc(&d_value, kv_len * kv_head_num * kv_head_dim * sizeof(float));
    cudaMalloc(&d_output, q_len * q_head_num * head_dim * sizeof(float));

    // 为每个组分配scores缓冲区
    float* d_scores;
    cudaMalloc(&d_scores, q_len * q_head_num * kv_len * sizeof(float));

    // 拷贝数据到GPU
    cudaMemcpy(d_query, query, q_len * q_head_num * head_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_key, key, kv_len * kv_head_num * kv_head_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_value, value, kv_len * kv_head_num * kv_head_dim * sizeof(float), cudaMemcpyHostToDevice);

    // 创建stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 设置kernel启动参数
    dim3 block(256);
    dim3 grid_qk((q_len * heads_per_group * kv_len + 255) / 256);
    dim3 grid_softmax((q_len * heads_per_group + 255) / 256);
    dim3 grid_sv((q_len * heads_per_group * head_dim + 255) / 256);

    printf("\n开始GQA计算...\n");

    // 按照Python代码的逻辑：对每个组计算注意力
    for (int group = 0; group < num_groups; group++) {
        printf("处理第%d组 (共%d组)...\n", group + 1, num_groups);

        // 计算内存偏移 - 对应Python代码中的索引逻辑
        
        // 获取当前组的queries: group_q = q[:, start_idx:end_idx]
        // start_idx = group * heads_per_group
        // end_idx = (group + 1) * heads_per_group
        // q ->[head_per_group,q_len,head_dim]
        int query_offset = group * heads_per_group * q_len * head_dim;
        float* d_group_query = d_query + query_offset;

        // 获取当前组的key: group_k = k[:, group:group+1]
        // 在GQA中，每个组只有一个KV头
        // k v ->[kv_len,kv_head_dim]
        // 需要将其expand 到[head_per_group,q_len,kv_len]，然后调用batch_gemm_kernel
        int kv_offset = group * kv_len * kv_head_dim;
        float* d_group_key = d_key + kv_offset;
        float* d_group_value = d_value + kv_offset;

        // 获取当前组的输出位置
        // output ->[head_per_group,q_len,head_dim]
        int output_offset = group * heads_per_group * q_len * head_dim;
        float* d_group_output = d_output + output_offset;

        // 获取当前组的scores位置
        // score ->[head_per_group,q_len,kv_len]
        int scores_offset = group * heads_per_group * q_len * kv_len;
        float* d_group_scores = d_scores + scores_offset;

        // 步骤1: 计算注意力分数 Q * K^T
        // scores = torch.matmul(group_q, group_k.transpose(-2, -1)) / math.sqrt(head_dim)
        printf("  计算Q*K^T...\n");
        gemm_qk_kernel<<<grid_qk, block, 0, stream>>>(
            d_group_query,    // [q_len, heads_per_group, head_dim]
            d_group_key,      // [kv_len, kv_head_dim] - 需要在kernel中广播到heads_per_group
            d_group_scores,   // [q_len, heads_per_group, kv_len]
            q_len, heads_per_group, head_dim, kv_len
        );

        // 等待Q*K^T计算完成
        cudaStreamSynchronize(stream);

        // 步骤2: 应用softmax获得注意力权重
        // attn_weights = F.softmax(scores, dim=-1)
        printf("  应用Softmax...\n");
        softmax_kernel<<<grid_softmax, block, 0, stream>>>(
            d_group_scores,   // [q_len, heads_per_group, kv_len]
            q_len, heads_per_group, kv_len
        );

        // 等待Softmax完成
        cudaStreamSynchronize(stream);

        // 步骤3: 应用注意力权重到values
        // out[:, start_idx:end_idx] = torch.matmul(attn_weights, group_v)
        printf("  计算注意力加权输出...\n");
        gemm_sv_kernel<<<grid_sv, block, 0, stream>>>(
            d_group_scores,   // [q_len, heads_per_group, kv_len]
            d_group_value,    // [kv_len, kv_head_dim] - 需要在kernel中广播到heads_per_group
            d_group_output,   // [q_len, heads_per_group, head_dim]
            q_len, heads_per_group, head_dim, kv_len
        );

        // 等待当前组完成
        cudaStreamSynchronize(stream);
        printf("第%d组处理完成\n", group + 1);
    }

    printf("\n所有组处理完成！\n");

    // 拷贝结果回host
    cudaMemcpy(output, d_output, q_len * q_head_num * head_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // 验证结果
    printf("前16个输出值:\n");
    for (int i = 0; i < 16; i++) {
        printf("%.6f ", output[i]);
        if ((i + 1) % 8 == 0) printf("\n");
    }

    // 计算一些统计信息验证正确性
    float sum = 0.0f, max_val = output[0], min_val = output[0];
    for (int i = 0; i < q_len * q_head_num * head_dim; i++) {
        sum += output[i];
        max_val = fmaxf(max_val, output[i]);
        min_val = fminf(min_val, output[i]);
    }
    float mean = sum / (q_len * q_head_num * head_dim);
    
    printf("\n输出统计:\n");
    printf("均值: %.6f\n", mean);
    printf("最大值: %.6f\n", max_val);
    printf("最小值: %.6f\n", min_val);

    // 清理资源
    cudaStreamDestroy(stream);
    
    cudaFree(d_query);
    cudaFree(d_key);
    cudaFree(d_value);
    cudaFree(d_output);
    cudaFree(d_scores);
    
    delete[] query;
    delete[] key;
    delete[] value;
    delete[] output;

    printf("\nGQA计算完成!\n");
    return 0;
}

// 下面是对应的kernel实现示例（需要根据具体需求完善）

__global__ void gemm_qk_kernel(float* query, float* key, float* scores,
                              int q_len, int heads_per_group, int head_dim, int kv_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = q_len * heads_per_group * kv_len;
    
    if (idx >= total_elements) return;
    
    // 解析索引: [q_pos, head, kv_pos]
    int kv_pos = idx % kv_len;
    int head = (idx / kv_len) % heads_per_group;
    int q_pos = idx / (kv_len * heads_per_group);
    
    float scale = 1.0f / sqrtf((float)head_dim);
    
    // 计算点积: query[q_pos, head, :] · key[kv_pos, :]
    float score = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        float q_val = query[q_pos * heads_per_group * head_dim + head * head_dim + d];
        float k_val = key[kv_pos * head_dim + d]; // 注意：key在所有头之间共享
        score += q_val * k_val;
    }
    
    scores[idx] = score * scale;
}

__global__ void softmax_kernel(float* scores, int q_len, int heads_per_group, int kv_len) {
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_rows = q_len * heads_per_group;
    
    if (row_idx >= total_rows) return;
    
    // 对每一行应用softmax
    float* row = scores + row_idx * kv_len;
    
    // 找最大值（数值稳定性）
    float max_val = row[0];
    for (int i = 1; i < kv_len; i++) {
        max_val = fmaxf(max_val, row[i]);
    }
    
    // 计算exp和sum
    float sum = 0.0f;
    for (int i = 0; i < kv_len; i++) {
        row[i] = expf(row[i] - max_val);
        sum += row[i];
    }
    
    // 归一化
    for (int i = 0; i < kv_len; i++) {
        row[i] /= sum;
    }
}

__global__ void gemm_sv_kernel(float* scores, float* value, float* output,
                              int q_len, int heads_per_group, int head_dim, int kv_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = q_len * heads_per_group * head_dim;
    
    if (idx >= total_elements) return;
    
    // 解析索引: [q_pos, head, dim]
    int dim = idx % head_dim;
    int head = (idx / head_dim) % heads_per_group;
    int q_pos = idx / (head_dim * heads_per_group);
    
    // 计算加权和: sum(scores[q_pos, head, kv_pos] * value[kv_pos, dim])
    float result = 0.0f;
    for (int kv_pos = 0; kv_pos < kv_len; kv_pos++) {
        float score = scores[q_pos * heads_per_group * kv_len + head * kv_len + kv_pos];
        float val = value[kv_pos * head_dim + dim]; // 注意：value在所有头之间共享
        result += score * val;
    }
    
    output[idx] = result;
} 