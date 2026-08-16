#include<bits/stdc++.h>
#include "cuda_runtime.h"
//参考：https://zhuanlan.zhihu.com/p/705070986，下面代码可能不对
template<int eps,int B, int T,int C>
void layernorm_cpu(float* input, float* out, float *weight, float* bias){
    for(int i = 0;i < B*T;i++){
        float mean = 0;
        for(j = 0;j < n;j++){
            mean += input[i*C+j];
        }
        mean /= C;
        float var = 0;
        for(int j = 0;j < n;j++){
            float t = x[i*C]-mean;
            var += t*t;
        }
        var = 1.0f/sqrt(var/C+eps);
        for(int j = 0;j < n;j++){
            out[i*C+j] = ((x[i*C+j]-mean)*var+bias[i])*weight[i];
        }

    }
}
template<int eps,int B, int T,int C>
__global__ void layernorm_GPU(float* input, float* out, float *weight, float* bias){
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    float x = input[tid];
    float sum = block_reduce_sum(x);
    float mean = sum/C;
    x -= mean;
    float t = x*x;
    float var = block_reduce_sum(t);
    var /= C;
    t = 1/(sqrt(var/C+eps));
    out[tid] = weight[threadIdx.x]*t*x+bias[threadIdx.x];
}
int main(){

    const int B = 8;
    const int T = 1024;
    const int C = 728;
    const int eps = 1e-8;
    float* input = (float*)malloc(sizeof(float)*B*T*C);
    float* out = (float*)malloc(sizeof(float)*B*T*C);
    float* weight = (float*)malloc(sizeof(float)*C);
    float *bias = (float *)malloc(C * sizeof(float));

    for(int i = 0;i < B*T*C;i++){
        input[i] = i%20;
    }
    layernorm_cpu<eps,B,T,C>(input, out,weight,bias);
    float *inputGPU, *outputGPU, *weightGPU, *biasGPU;
    cudaMalloc(&inputGPU, B * T * C * sizeof(float));
    cudaMemcpy(inputGPU, input, B * T * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&weightGPU, C * sizeof(float));
    cudaMemcpy(weightGPU, weight, C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&biasGPU, C * sizeof(float));
    cudaMemcpy(biasGPU, bias, C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&outputGPU, B * T * C * sizeof(float));
    cudaMemset(outputGPU, 0, B * T * C * sizeof(float));
    dim3 gridsize = B*T;
    dim3 block_size = C;
    layernorm_GPU<eps,B,T,C><<<B*T,C>>>(inputGPU,outputGPU,weightGPU,biasGPU);

}