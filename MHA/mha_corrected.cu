#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

// 声明kernel函数（这里只是示例，实际需要你的具体实现）
__global__ void gemm_kernel(float* query, float* key, float* value, float* output, float* score,
                           int q_len, int q_head_num, int head_dim, int kv_len, int kv_head_num, int kv_head_dim);
__global__ void softmax_kernel(float* score, int q_len, int q_head_num, int kv_len);

int main() {
    // 参数设置
    int q_len = 1;
    int kv_len = 1024;
    int num_groups = 4;
    int hidden_size = 4096;

    int q_head_num = 32;
    int head_dim = hidden_size / q_head_num; // 128
    int head_per_group = q_head_num / num_groups; // 8

    int kv_head_num = num_groups; // 4
    int kv_head_dim = head_dim; // 应该是128，不是head_dim * num_groups

    printf("配置参数:\n");
    printf("q_len=%d, kv_len=%d, num_groups=%d\n", q_len, kv_len, num_groups);
    printf("q_head_num=%d, head_dim=%d, head_per_group=%d\n", q_head_num, head_dim, head_per_group);
    printf("kv_head_num=%d, kv_head_dim=%d\n", kv_head_num, kv_head_dim);

    // 分配host内存
    float* query = new float[q_head_num * q_len * head_dim];
    float* key = new float[kv_head_num * kv_len * kv_head_dim];
    float* value = new float[kv_head_num * kv_len * kv_head_dim];
    float* score = new float[q_len * q_head_num * kv_len];
    float* output = new float[q_len * q_head_num * head_dim]; // 只定义一次

    // 初始化数据
    srand(42);
    for (int i = 0; i < q_head_num * q_len * head_dim; i++) {
        query[i] = rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < kv_len * kv_head_num * kv_head_dim; i++) {
        key[i] = rand() / (float)RAND_MAX;
        value[i] = rand() / (float)RAND_MAX;
    }

    // 分配device内存
    float *d_query, *d_key, *d_value, *d_output, *d_score;
    cudaMalloc(&d_query, q_len * q_head_num * head_dim * sizeof(float));
    cudaMalloc(&d_key, kv_len * kv_head_num * kv_head_dim * sizeof(float));
    cudaMalloc(&d_value, kv_len * kv_head_num * kv_head_dim * sizeof(float));
    cudaMalloc(&d_output, q_len * q_head_num * head_dim * sizeof(float));
    cudaMalloc(&d_score, q_len * q_head_num * kv_len * sizeof(float));

    // 拷贝数据到GPU
    cudaMemcpy(d_query, query, q_len * q_head_num * head_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_key, key, kv_len * kv_head_num * kv_head_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_value, value, kv_len * kv_head_num * kv_head_dim * sizeof(float), cudaMemcpyHostToDevice);

    // 创建stream和events用于同步
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // 为每个group创建event用于同步
    cudaEvent_t* gemm_events = new cudaEvent_t[num_groups];
    for (int i = 0; i < num_groups; i++) {
        cudaEventCreate(&gemm_events[i]);
    }

    dim3 block(256, 1, 1); // 调整block大小
    dim3 grid((q_len * head_per_group * head_dim + 256 - 1) / 256, 1, 1);

    printf("\n开始处理各组...\n");
    
    // 方法1：使用cudaStreamSynchronize确保每个group内的同步
    for (int i = 0; i < num_groups; i++) {
        printf("处理第%d组...\n", i);
        
        // 计算正确的内存偏移
        // Query: 每组有head_per_group个头，每个头有q_len*head_dim个元素
        // q ->[head_per_group,q_len,head_dim]
        float* d_query_group = d_query + i * head_per_group * q_len * head_dim;
        
        // Key/Value: 每组有1个KV头，每个头有kv_len*kv_head_dim个元素  
        // k v ->[kv_len,kv_head_dim]
        float* d_key_group = d_key + i * kv_len * kv_head_dim;
        float* d_value_group = d_value + i * kv_len * kv_head_dim;
        
        // Output: 每组有head_per_group个头的输出
        // output ->[head_per_group,q_len,head_dim]
        float* d_output_group = d_output + i * head_per_group * q_len * head_dim;
        
        // Score: 每组有head_per_group个头，每个头有q_len*kv_len个分数
        // score ->[head_per_group,q_len,kv_len]
        float* d_score_group = d_score + i * head_per_group * q_len * kv_len;

        // 启动GEMM kernel
        gemm_kernel<<<grid, block, 0, stream>>>(
            d_query_group, d_key_group, d_value_group, 
            d_output_group, d_score_group,
            q_len, head_per_group, head_dim, kv_len, 1, kv_head_dim
        );
        
        // 记录GEMM完成事件
        cudaEventRecord(gemm_events[i], stream);
        
        // 等待GEMM完成（方法1：简单但效率较低）
        cudaStreamSynchronize(stream);
        
        // 启动Softmax kernel
        softmax_kernel<<<grid, block, 0, stream>>>(
            d_score_group, q_len, head_per_group, kv_len
        );
        
        // 等待Softmax完成
        cudaStreamSynchronize(stream);
        
        printf("第%d组处理完成\n", i);
    }

    printf("\n所有组处理完成，进行最终的输出计算...\n");
    
    // 最终的GEMM操作：score * value -> output
    // 注意：这里需要重新计算grid大小
    dim3 final_grid((q_len * q_head_num * head_dim + 256 - 1) / 256, 1, 1);
    
    // 这个调用看起来有问题，需要根据你的具体kernel实现来调整参数
    // gemm_kernel<<<final_grid, block, 0, stream>>>(
    //     d_score, d_value, d_output, 
    //     q_len, q_head_num, head_dim, kv_len, kv_head_num, kv_head_dim
    // );
    
    cudaStreamSynchronize(stream);

    // 拷贝结果回host
    cudaMemcpy(output, d_output, q_len * q_head_num * head_dim * sizeof(float), cudaMemcpyDeviceToHost);

    printf("计算完成!\n");
    printf("前8个输出值: ");
    for (int i = 0; i < 8; i++) {
        printf("%.6f ", output[i]);
    }
    printf("\n");

    // 清理资源
    for (int i = 0; i < num_groups; i++) {
        cudaEventDestroy(gemm_events[i]);
    }
    delete[] gemm_events;
    
    cudaStreamDestroy(stream);
    
    cudaFree(d_query);
    cudaFree(d_key);
    cudaFree(d_value);
    cudaFree(d_output);
    cudaFree(d_score);
    
    delete[] query;
    delete[] key;
    delete[] value;
    delete[] score;
    delete[] output;

    return 0;
}

// 更高效的同步方法示例
void mha_with_events() {
    // ... 前面的初始化代码相同 ...
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // 创建events用于更精确的同步
    cudaEvent_t* gemm_events = new cudaEvent_t[num_groups];
    cudaEvent_t* softmax_events = new cudaEvent_t[num_groups];
    
    for (int i = 0; i < num_groups; i++) {
        cudaEventCreate(&gemm_events[i]);
        cudaEventCreate(&softmax_events[i]);
    }
    
    // 方法2：使用events进行更精确的同步
    for (int i = 0; i < num_groups; i++) {
        // 计算内存偏移...
        
        // 启动GEMM
        // gemm_kernel<<<grid, block, 0, stream>>>(...);
        cudaEventRecord(gemm_events[i], stream);
        
        // 让stream等待GEMM完成
        cudaStreamWaitEvent(stream, gemm_events[i], 0);
        
        // 启动Softmax
        // softmax_kernel<<<grid, block, 0, stream>>>(...);
        cudaEventRecord(softmax_events[i], stream);
    }
    
    // 等待所有操作完成
    for (int i = 0; i < num_groups; i++) {
        cudaEventSynchronize(softmax_events[i]);
    }
    
    // 清理events
    for (int i = 0; i < num_groups; i++) {
        cudaEventDestroy(gemm_events[i]);
        cudaEventDestroy(softmax_events[i]);
    }
    delete[] gemm_events;
    delete[] softmax_events;
}

// 最高效的方法：使用多个stream
void mha_with_multiple_streams() {
    // ... 初始化代码 ...
    
    // 为每个group创建独立的stream
    cudaStream_t* streams = new cudaStream_t[num_groups];
    cudaEvent_t* events = new cudaEvent_t[num_groups];
    
    for (int i = 0; i < num_groups; i++) {
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
    }
    
    // 并行处理各个group
    for (int i = 0; i < num_groups; i++) {
        // 在独立stream中启动GEMM
        // gemm_kernel<<<grid, block, 0, streams[i]>>>(...);
        
        // 在同一stream中启动Softmax（自动等待GEMM完成）
        // softmax_kernel<<<grid, block, 0, streams[i]>>>(...);
        
        cudaEventRecord(events[i], streams[i]);
    }
    
    // 等待所有stream完成
    for (int i = 0; i < num_groups; i++) {
        cudaEventSynchronize(events[i]);
    }
    
    // 清理资源
    for (int i = 0; i < num_groups; i++) {
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }
    delete[] streams;
    delete[] events;
} 