


int main() {

    int q_len = 1;
    int kv_len = 1024;
    int num_groups = 4;
    int hidden_size = 4096;

    int q_head_num = 32;
    int head_dim = hidden_size / q_head_num; 

    int head_per_group = q_head_num / num_groups;

    int kv_head_num = num_groups ;       
    int kv_head_dim = head_dim * num_groups;

    float * query = new float[q_head_num * q_len * head_dim];
    float * key = new float[kv_head_num * kv_len * kv_head_dim];
    float * value = new float[kv_head_num * kv_len * kv_head_dim];

    float * output = new float[q_head_num * q_len * head_dim];

    for (int i = 0; i < q_head_num * q_len * head_dim; i++) {
        query[i] = rand() / (float)RAND_MAX;    
    }
    for (int i = 0; i < kv_len * kv_head_num * kv_head_dim; i++) {
        key[i] = rand() / (float)RAND_MAX;
        value[i] = rand() / (float)RAND_MAX;
    }

    float * score = new float[q_len * q_head_num * kv_len];
    float * output = new float[q_len * q_head_num * head_dim];

    float * d_query, * d_key, * d_value, * d_output, * d_score;
    cudaMalloc(&d_query, q_len * q_head_num * head_dim * sizeof(float));
    cudaMalloc(&d_key, kv_len * kv_head_num * kv_head_dim * sizeof(float)   );
    cudaMalloc(&d_value, kv_len * kv_head_num * kv_head_dim * sizeof(float));
    cudaMalloc(&d_output, q_len * q_head_num * head_dim * sizeof(float));
    cudaMalloc(&d_score, q_len * q_head_num * kv_len * sizeof(float));


    cudaMemcpy(d_query, query, q_len * q_head_num * head_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_key, key, kv_len * kv_head_num * kv_head_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_value, value, kv_len * kv_head_num * kv_head_dim * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(1024, 1, 1);
    dim3 grid((q_len * q_head_num * head_dim + 1024 - 1) / 1024, 1, 1);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    #pragma unroll
    for(int i = 0; i < num_groups; i++) {
        float * d_query_group = d_query + i * q_head_num * head_dim* head_per_group;        
    
        float * d_key_group = d_key + i * kv_head_num * kv_head_dim * kv_len;
        float * d_value_group = d_value + i * kv_head_num * kv_head_dim * kv_len;
        float * d_output_group = d_output + i * q_head_num * head_dim * q_len;
        float * d_score_group = d_score + i * q_head_num * kv_len * q_len;
        // q ->[head_per_group,q_len,head_dim]
        // k ->[kv_len,kv_head_dim]
        // v ->[kv_len,kv_head_dim]
        // score ->[q_len,kv_len]
        // output ->[q_len,head_dim]

        // q * k^T ->[q_len,kv_len]
        // score ->[q_len,kv_len]

        gemm_kernel<<<grid, block, 0, stream>>>(d_query_group, d_key_group, d_value_group, d_output_group, d_score_group, q_len, q_head_num, head_dim, kv_len, kv_head_num, kv_head_dim);
        softmax_kernel<<<grid, block, 0, stream>>>(d_score_group, q_len, q_head_num, kv_len);
    }
    

    gemm_kernel<<<grid, block, 0, stream>>>(d_score, d_value, d_output, q_len, q_head_num, head_dim, kv_len, kv_head_num, kv_head_dim);
    
    
    cudaMemcpy(output, d_output, q_len * q_head_num * head_dim * sizeof(float), cudaMemcpyDeviceToHost);

    

    return 0;
}