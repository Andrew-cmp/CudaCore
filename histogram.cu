#include "stdio.h"

const int N = 1024 * 1024 * 100;
const int NUM_REAPEATS = 20;

void bigRandomBlock(unsigned char* data, const int n) {
    srand(1234);
    for (int i=0; i<n; ++i) data[i] = rand();
}

void histCpu(const unsigned char *h_buffer, unsigned int *h_histo, const int N) {
    for (int i=0; i<N; ++i) h_histo[h_buffer[i]]++;
}

__global__ void histKernel(
    const unsigned char *d_buffer, 
    const int N, 
    unsigned int *d_histo) {

    const int n = threadIdx.x + blockDim.x * blockIdx.x;
    if (n < N) atomicAdd(&d_histo[d_buffer[n]], 1);
}

__global__ void histParallelism(
    const unsigned char *d_buffer, 
    const int N, 
    unsigned int *d_histo) {

    int n = threadIdx.x + blockDim.x * blockIdx.x;
    __shared__ unsigned int s_histo[256];
    s_histo[threadIdx.x] = 0;
    __syncthreads();
    const int offset = gridDim.x * blockDim.x;
    for (; n<N; n+=offset) {
        atomicAdd(&s_histo[d_buffer[n]], 1); 
    }
    __syncthreads();
    atomicAdd(&d_histo[threadIdx.x], s_histo[threadIdx.x]);
}

__global__ void histshared(
    const unsigned char *d_buffer, 
    const int N, 
    unsigned int *d_histo) {

    int n = threadIdx.x + blockDim.x * blockIdx.x;
    __shared__ unsigned int s_histo[256];
    s_histo[threadIdx.x] = 0;
    __syncthreads();
    if (n < N) {
        atomicAdd(&s_histo[d_buffer[n]], 1);
        __syncthreads();
        if (s_histo[threadIdx.x]) atomicAdd(&d_histo[threadIdx.x], s_histo[threadIdx.x]);
    }
}

void hist(
    unsigned char *h_buffer,
    unsigned char *d_buffer,
    unsigned int *h_histo,
    unsigned int *d_histo,
    const int N, 
    const int method) {

    switch (method)
    {
    case 0:
        histCpu(h_buffer, h_histo, N);
        break;
    case 1:
        histKernel<<<(N - 1) / 256 + 1, 256>>>(d_buffer, N, d_histo);
        break;
    case 2:
        histParallelism<<<10240, 256>>>(d_buffer, N, d_histo);
        break;
    case 3:
        histshared<<<(N - 1) / 256 + 1, 256>>>(d_buffer, N, d_histo);
        break;
    
    default:
        break;
    }
}

void timing(
    unsigned char *h_buffer,
    unsigned char *d_buffer,
    unsigned int *h_histo,
    unsigned int *d_histo,
    const int hmem,
    const int N, 
    const int method) {

    float tSum = 0.0;
    float t2Sum = 0.0;
    for (int i=0; i<NUM_REAPEATS; ++i) {
        for (int i=0; i<256; i++) h_histo[i] = 0;
        CHECK(cudaMemcpy(d_histo, h_histo, hmem, cudaMemcpyHostToDevice));
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        hist(h_buffer, d_buffer, h_histo, d_histo, N, method);

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsedTime;
        CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
        tSum += elapsedTime;
        t2Sum += elapsedTime * elapsedTime;
        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }
    float tAVG = tSum / NUM_REAPEATS;
    float tERR = sqrt(t2Sum / NUM_REAPEATS - tAVG * tAVG);
    printf("Time = %g +- %g ms.\n", tAVG, tERR);
    if (method > 0) CHECK(cudaMemcpy(h_histo, d_histo, hmem, cudaMemcpyDeviceToHost));
}


int main() {
    unsigned char *h_buffer = new unsigned char[N];
    bigRandomBlock(h_buffer, N);
    unsigned int *h_histo = new unsigned int[256];
    for (int i=0; i<256; i++) h_histo[i] = 0;
    unsigned char *d_buffer;
    unsigned int *d_histo;
    const int bmem = sizeof(unsigned char) * N;
    const int hmem = sizeof(unsigned int) * 256;
    CHECK(cudaMalloc((void **)&d_buffer, bmem));
    CHECK(cudaMalloc((void **)&d_histo, hmem));
    CHECK(cudaMemcpy(d_buffer, h_buffer, bmem, cudaMemcpyHostToDevice));
    printf("Using cpu:\n");
    timing(h_buffer, d_buffer, h_histo, d_histo, hmem, N, 0);
    for (int i=0; i<256; i+=16) printf("%d  ", h_histo[i]);
    printf("\n");

    printf("Using kernel:\n");
    timing(h_buffer, d_buffer, h_histo, d_histo, hmem, N, 1);
    for (int i=0; i<256; i+=16) printf("%d  ", h_histo[i]);
    printf("\n");

    printf("Using kernel Parallelism:\n");
    timing(h_buffer, d_buffer, h_histo, d_histo, hmem, N, 2);
    for (int i=0; i<256; i+=16) printf("%d  ", h_histo[i]);
    printf("\n");

    printf("Using kernel shared:\n");
    timing(h_buffer, d_buffer, h_histo, d_histo, hmem, N, 3);
    for (int i=0; i<256; i+=16) printf("%d  ", h_histo[i]);
    printf("\n");

    delete[] h_buffer;
    delete[] h_histo;
    CHECK(cudaFree(d_buffer));
    CHECK(cudaFree(d_histo));
    
    return 0;
}