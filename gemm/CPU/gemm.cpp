#include <stdio.h>
#include <string.h>
#include <chrono>
#include <stdlib.h>
#include <omp.h>
#include <immintrin.h>  // AVX指令集头文件
// 假定beta和alpha永远为1
// 使用float类型进行计算
// 假设都能整除
typedef float ELE_TYPE;
#define A(i,j,lda) A[(i)*(lda)+(j)]
#define B(i,j,ldb) B[(i)*(ldb)+(j)]
#define C(i,j,ldc) C[(i)*(ldc)+(j)]
// 行主序访问宏定义 
void naive_row_major_sgemm(
    char transa,
    char transb,
    int M, int N, int K, 
    const float alpha,
    const float * src_a, int lda,
    const float * src_b, int ldb, 
    const float beta,
    float * dst, int ldc)
{
    int a_stride_m = transa == 'n' ? 1 : lda;
    int a_stride_k = transa == 'n' ? lda : 1;
    int b_stride_k = transb == 'n' ? 1 : ldb;
    int b_stride_n = transb == 'n' ? ldb : 1;
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            float acc = 0.f;
            
            for(int k = 0; k < K; k++){
                acc += src_a[i * a_stride_m + k * a_stride_k] * src_b[k * b_stride_k + j * b_stride_n];
            }
            dst[i + j * ldc ] = alpha * acc + beta * dst[i + j * ldc];

        }
    }
}
//循环重排并不能解决问题，很奇怪，速度反而满了
void loop_reoder_row_major_sgemm(
    char transa,
    char transb,
    int M, int N, int K, 
    const float alpha,
    const float * src_a, int lda,
    const float * src_b, int ldb, 
    const float beta,
    float * dst, int ldc)
{
    int a_stride_m = transa == 'n' ? 1 : lda;
    int a_stride_k = transa == 'n' ? lda : 1;
    int b_stride_k = transb == 'n' ? 1 : ldb;
    int b_stride_n = transb == 'n' ? ldb : 1;
    for(int i = 0; i < M; i++){
        for(int k = 0; k < K; k++){
            float acc = 0.f;
            for(int j = 0; j < N; j++){
                dst[i + j * ldc ] += src_a[i * a_stride_m + k * a_stride_k] * src_b[k * b_stride_k + j * b_stride_n];
            }
        }
    }
}
void tile_omp_row_major_sgemm(
    char transa,
    char transb,
    int M, int N, int K, 
    const float alpha,
    const float * src_a, int lda,
    const float * src_b, int ldb, 
    const float beta,
    float * dst, int ldc)
{
    int a_stride_m = transa == 'n' ? 1 : lda;
    int a_stride_k = transa == 'n' ? lda : 1;
    int b_stride_k = transb == 'n' ? 1 : ldb;
    int b_stride_n = transb == 'n' ? ldb : 1;
    int tile_size = 16;
    
    #pragma omp parallel for
    for(int i = 0; i < M; i+=tile_size){
        for(int j = 0; j < N; j+=tile_size){
            for(int k = 0; k < K; k+=tile_size){
                // 计算当前块的实际边界
                int i_end = (i + tile_size < M) ? i + tile_size : M;
                int j_end = (j + tile_size < N) ? j + tile_size : N;
                int k_end = (k + tile_size < K) ? k + tile_size : K;
                for(int ii = i; ii < i_end; ii++){  
                    for(int jj = j; jj < j_end; jj++){
                        float acc = 0.f;
                        for(int kk = k; kk < k_end; kk++){
                            acc += src_a[ii * a_stride_m + kk * a_stride_k] * src_b[kk * b_stride_k + jj * b_stride_n];
                        }
                        dst[ii + jj * ldc ] += alpha * acc + beta * dst[ii + jj * ldc];
                    }
                }   
            }
        }
    }
}

void tile_omp_loop_reorder_row_major_sgemm(
    char transa,
    char transb,
    int M, int N, int K, 
    const float alpha,
    const float * src_a, int lda,
    const float * src_b, int ldb, 
    const float beta,
    float * dst, int ldc)
{
    int a_stride_m = transa == 'n' ? 1 : lda;
    int a_stride_k = transa == 'n' ? lda : 1;
    int b_stride_k = transb == 'n' ? 1 : ldb;
    int b_stride_n = transb == 'n' ? ldb : 1;
    int tile_size = 16;
    
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < M; i+=tile_size){
        for(int j = 0; j < N; j+=tile_size){
            for(int k = 0; k < K; k+=tile_size){
                // 计算当前块的实际边界
                int i_end = (i + tile_size < M) ? i + tile_size : M;
                int j_end = (j + tile_size < N) ? j + tile_size : N;
                int k_end = (k + tile_size < K) ? k + tile_size : K;
                for(int ii = i; ii < i_end; ii++){  
                    for(int kk = k; kk < k_end; kk++){
                        for(int jj = j; jj < j_end; jj++){
                            dst[ii + jj * ldc ] += alpha * src_a[a_stride_m*ii+a_stride_k*kk]*src_b[b_stride_k*kk+b_stride_n*j] + beta * dst[ii + jj * ldc];
                        }
                    }
                }   
            }
        }
    }
}
// SIMD + OpenMP 优化版本：向量化load/store + 并行
void tile_omp_simd_sgemm(
    char transa, char transb,
    int M, int N, int K, 
    const float alpha,
    const float * src_a, int lda,
    const float * src_b, int ldb, 
    const float beta,
    float * dst, int ldc)
{
    int a_stride_m = transa == 'n' ? 1 : lda;
    int a_stride_k = transa == 'n' ? lda : 1;
    int b_stride_k = transb == 'n' ? 1 : ldb;
    int b_stride_n = transb == 'n' ? ldb : 1;
    
    const int tile_size = 64;  // 更大的tile适合SIMD
    const int vec_size = 8;    // AVX2处理8个float
    
    __m256 alpha_vec = _mm256_set1_ps(alpha);
    __m256 beta_vec = _mm256_set1_ps(beta);
    
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < M; i += tile_size) {
        for(int j = 0; j < N; j += tile_size) {
            int i_end = (i + tile_size < M) ? i + tile_size : M;
            int j_end = (j + tile_size < N) ? j + tile_size : N;
            for(int k = 0; k < K; k += tile_size) {
                int k_end = (k + tile_size < K) ? k + tile_size : K;
                for(int ii = i; ii < i_end; ii++) {
                    for(int kk = k; kk < k_end; kk++) {
                        // 广播A矩阵元素到向量寄存器
                        __m256 a_vec = _mm256_set1_ps(alpha * src_a[ii * a_stride_m + kk * a_stride_k]);
                        int jj = j;
                        // SIMD向量化内层循环：一次处理8个元素
                        for(; jj <= j_end - vec_size; jj += vec_size) {
                            // 向量化load B矩阵
                            __m256 b_vec = _mm256_loadu_ps(&src_b[kk * b_stride_k + jj * b_stride_n]);
                            
                            // 向量化load C矩阵
                            __m256 c_vec = _mm256_loadu_ps(&dst[ii + jj * ldc]);
                            
                            // 融合乘加运算：c = a * b + c
                            c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                            
                            // 向量化store结果
                            _mm256_storeu_ps(&dst[ii + jj * ldc], c_vec);
                        }
                        
                        // 处理剩余元素（标量计算）
                        float a_scalar = alpha * src_a[ii * a_stride_m + kk * a_stride_k];
                        for(; jj < j_end; jj++) {
                            dst[ii + jj * ldc] += a_scalar * src_b[kk * b_stride_k + jj * b_stride_n];
                        }
                    }
                }
            }
        }
    }
}

// 高级SIMD优化版本：循环展开 + 向量化
void tile_omp_simd_unroll_sgemm(
    char transa, char transb,
    int M, int N, int K, 
    const float alpha,
    const float * src_a, int lda,
    const float * src_b, int ldb, 
    const float beta,
    float * dst, int ldc)
{
    int a_stride_m = transa == 'n' ? 1 : lda;
    int a_stride_k = transa == 'n' ? lda : 1;
    int b_stride_k = transb == 'n' ? 1 : ldb;
    int b_stride_n = transb == 'n' ? ldb : 1;
    
    const int tile_size = 64;
    const int vec_size = 8;
    const int unroll = 4;  // 4倍循环展开
    
    __m256 alpha_vec = _mm256_set1_ps(alpha);
    __m256 beta_vec = _mm256_set1_ps(beta);
    
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < M; i += tile_size) {
        for(int j = 0; j < N; j += tile_size) {
            int i_end = (i + tile_size < M) ? i + tile_size : M;
            int j_end = (j + tile_size < N) ? j + tile_size : N;
            
            // // 向量化初始化C矩阵
            // for(int ii = i; ii < i_end; ii++) {
            //     int jj = j;
            //     for(; jj <= j_end - vec_size; jj += vec_size) {
            //         __m256 c_vec = _mm256_loadu_ps(&dst[ii + jj * ldc]);
            //         c_vec = _mm256_mul_ps(c_vec, beta_vec);
            //         _mm256_storeu_ps(&dst[ii + jj * ldc], c_vec);
            //     }
            //     for(; jj < j_end; jj++) {
            //         dst[ii + jj * ldc] *= beta;
            //     }
            // }
            
            for(int k = 0; k < K; k += tile_size) {
                int k_end = (k + tile_size < K) ? k + tile_size : K;
                
                for(int ii = i; ii < i_end; ii++) {
                    for(int kk = k; kk < k_end; kk++) {
                        __m256 a_vec = _mm256_set1_ps(alpha * src_a[ii * a_stride_m + kk * a_stride_k]);
                        
                        int jj = j;
                        // 4倍循环展开的SIMD计算：一次处理32个元素
                        for(; jj <= j_end - (vec_size * unroll); jj += (vec_size * unroll)) {
                            // 第一组向量
                            __m256 b_vec1 = _mm256_loadu_ps(&src_b[kk * b_stride_k + jj * b_stride_n]);
                            __m256 c_vec1 = _mm256_loadu_ps(&dst[ii + jj * ldc]);
                            c_vec1 = _mm256_fmadd_ps(a_vec, b_vec1, c_vec1);
                            _mm256_storeu_ps(&dst[ii + jj * ldc], c_vec1);
                            
                            // 第二组向量
                            __m256 b_vec2 = _mm256_loadu_ps(&src_b[kk * b_stride_k + (jj + vec_size) * b_stride_n]);
                            __m256 c_vec2 = _mm256_loadu_ps(&dst[ii + (jj + vec_size) * ldc]);
                            c_vec2 = _mm256_fmadd_ps(a_vec, b_vec2, c_vec2);
                            _mm256_storeu_ps(&dst[ii + (jj + vec_size) * ldc], c_vec2);
                            
                            // 第三组向量
                            __m256 b_vec3 = _mm256_loadu_ps(&src_b[kk * b_stride_k + (jj + 2*vec_size) * b_stride_n]);
                            __m256 c_vec3 = _mm256_loadu_ps(&dst[ii + (jj + 2*vec_size) * ldc]);
                            c_vec3 = _mm256_fmadd_ps(a_vec, b_vec3, c_vec3);
                            _mm256_storeu_ps(&dst[ii + (jj + 2*vec_size) * ldc], c_vec3);
                            
                            // 第四组向量
                            __m256 b_vec4 = _mm256_loadu_ps(&src_b[kk * b_stride_k + (jj + 3*vec_size) * b_stride_n]);
                            __m256 c_vec4 = _mm256_loadu_ps(&dst[ii + (jj + 3*vec_size) * ldc]);
                            c_vec4 = _mm256_fmadd_ps(a_vec, b_vec4, c_vec4);
                            _mm256_storeu_ps(&dst[ii + (jj + 3*vec_size) * ldc], c_vec4);
                        }
                        
                        // 处理剩余的向量化部分
                        for(; jj <= j_end - vec_size; jj += vec_size) {
                            __m256 b_vec = _mm256_loadu_ps(&src_b[kk * b_stride_k + jj * b_stride_n]);
                            __m256 c_vec = _mm256_loadu_ps(&dst[ii + jj * ldc]);
                            c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                            _mm256_storeu_ps(&dst[ii + jj * ldc], c_vec);
                        }
                        
                        // 处理剩余标量元素
                        float a_scalar = alpha * src_a[ii * a_stride_m + kk * a_stride_k];
                        for(; jj < j_end; jj++) {
                            dst[ii + jj * ldc] += a_scalar * src_b[kk * b_stride_k + jj * b_stride_n];
                        }
                    }
                }
            }
        }
    }
}

// 预取优化版本：添加软件预取指令
void tile_omp_simd_prefetch_sgemm(
    char transa, char transb,
    int M, int N, int K, 
    const float alpha,
    const float * src_a, int lda,
    const float * src_b, int ldb, 
    const float beta,
    float * dst, int ldc)
{
    int a_stride_m = transa == 'n' ? 1 : lda;
    int a_stride_k = transa == 'n' ? lda : 1;
    int b_stride_k = transb == 'n' ? 1 : ldb;
    int b_stride_n = transb == 'n' ? ldb : 1;
    
    const int tile_size = 64;
    const int vec_size = 8;
    
    __m256 alpha_vec = _mm256_set1_ps(alpha);
    __m256 beta_vec = _mm256_set1_ps(beta);
    
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < M; i += tile_size) {
        for(int j = 0; j < N; j += tile_size) {
            int i_end = (i + tile_size < M) ? i + tile_size : M;
            int j_end = (j + tile_size < N) ? j + tile_size : N;
            
            // 初始化
            for(int ii = i; ii < i_end; ii++) {
                int jj = j;
                for(; jj <= j_end - vec_size; jj += vec_size) {
                    __m256 c_vec = _mm256_loadu_ps(&dst[ii + jj * ldc]);
                    c_vec = _mm256_mul_ps(c_vec, beta_vec);
                    _mm256_storeu_ps(&dst[ii + jj * ldc], c_vec);
                }
                for(; jj < j_end; jj++) {
                    dst[ii + jj * ldc] *= beta;
                }
            }
            
            for(int k = 0; k < K; k += tile_size) {
                int k_end = (k + tile_size < K) ? k + tile_size : K;
                
                for(int ii = i; ii < i_end; ii++) {
                    for(int kk = k; kk < k_end; kk++) {
                        __m256 a_vec = _mm256_set1_ps(alpha * src_a[ii * a_stride_m + kk * a_stride_k]);
                        
                        // 预取下一轮数据
                        if(kk + 1 < k_end) {
                            _mm_prefetch((const char*)&src_a[ii * a_stride_m + (kk + 1) * a_stride_k], _MM_HINT_T0);
                        }
                        
                        int jj = j;
                        for(; jj <= j_end - vec_size; jj += vec_size) {
                            // 预取B矩阵数据到L1缓存
                            _mm_prefetch((const char*)&src_b[kk * b_stride_k + (jj + vec_size) * b_stride_n], _MM_HINT_T0);
                            
                            __m256 b_vec = _mm256_loadu_ps(&src_b[kk * b_stride_k + jj * b_stride_n]);
                            __m256 c_vec = _mm256_loadu_ps(&dst[ii + jj * ldc]);
                            c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                            _mm256_storeu_ps(&dst[ii + jj * ldc], c_vec);
                        }
                        
                        float a_scalar = alpha * src_a[ii * a_stride_m + kk * a_stride_k];
                        for(; jj < j_end; jj++) {
                            dst[ii + jj * ldc] += a_scalar * src_b[kk * b_stride_k + jj * b_stride_n];
                        }
                    }
                }
            }
        }
    }
}
int main(){
    const int M = 1024, N = 1024, K = 1024;
    float *A = (float*)malloc(M * K * sizeof(float));
    float *B = (float*)malloc(K * N * sizeof(float));
    float *C = (float*)malloc(M * N * sizeof(float));
    
    // 检查内存分配是否成功
    if (!A || !B || !C) {
        printf("内存分配失败！\n");
        if (A) free(A);
        if (B) free(B);
        if (C) free(C);
        return -1;
    }
    
    // 初始化矩阵
    for(int i = 0; i < M; i++){
        for(int j = 0; j < K; j++){
            A(i,j,K) = 2.0f;
        }
    }
    
    for(int i = 0; i < K; i++){
        for(int j = 0; j < N; j++){
            B(i,j,N) = 2.0f;
        }
    }
    
    // 初始化C矩阵为0
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            C(i,j,N) = 0.0f;
        }
    }
    
    // 记录开始时间
    auto start = std::chrono::high_resolution_clock::now();
    naive_row_major_sgemm('n','n',M,N,K,1.0f,(ELE_TYPE*)A,K, (ELE_TYPE*)B,N,0.0f,(ELE_TYPE*)C,N);
    auto end = std::chrono::high_resolution_clock::now();
    memset(C,0,M*N*sizeof(float));
    // 计算执行时间
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    printf("native函数执行时间: %.6f 秒\n", duration.count() / 1000000.0);
    printf("C[0][0] = %f\n", C(0,0,N));
    printf("native 计算速度为%f GFLOPS\n", 2.0f * M * N * K / (duration.count()/ 1000000.0) / 1000000000.0);
    printf("--------------------------------\n");
    memset(C,0,M*N*sizeof(float));
    start = std::chrono::high_resolution_clock::now();
    loop_reoder_row_major_sgemm('n','n',M,N,K,1.0f,(ELE_TYPE*)A,K, (ELE_TYPE*)B,N,0.0f,(ELE_TYPE*)C,N);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    printf("loop_reoder函数执行时间: %.6f 秒\n", duration.count() / 1000000.0);
    printf("C[0][0] = %f\n", C(0,0,N));
    printf("loop_reoder 计算速度为%f GFLOPS\n", 2.0f * M * N * K / (duration.count()/ 1000000.0) / 1000000000.0);
    printf("--------------------------------\n");
    memset(C,0,M*N*sizeof(float));
    start = std::chrono::high_resolution_clock::now();
    tile_omp_row_major_sgemm('n','n',M,N,K,1.0f,(ELE_TYPE*)A,K, (ELE_TYPE*)B,N,0.0f,(ELE_TYPE*)C,N);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    printf("tile_omp函数执行时间: %.6f 秒\n", duration.count() / 1000000.0);
    printf("C[0][0] = %f\n", C(0,0,N));
    printf("tile_omp 计算速度为%f GFLOPS\n", 2.0f * M * N * K / (duration.count()/ 1000000.0) / 1000000000.0);
    printf("--------------------------------\n");
    memset(C,0,M*N*sizeof(float));
    start = std::chrono::high_resolution_clock::now();
    tile_omp_loop_reorder_row_major_sgemm('n','n',M,N,K,1.0f,(ELE_TYPE*)A,K, (ELE_TYPE*)B,N,0.0f,(ELE_TYPE*)C,N);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    printf(" tile_omp_loop_reorder 函数执行时间: %.6f 秒\n", duration.count() / 1000000.0);
    printf("C[0][0] = %f\n", C(0,0,N));
    printf(" tile_omp_loop_reorder 计算速度为%f GFLOPS\n", 2.0f * M * N * K / (duration.count()/ 1000000.0) / 1000000000.0);
    printf("--------------------------------\n");
    memset(C,0,M*N*sizeof(float));
    start = std::chrono::high_resolution_clock::now();
    tile_omp_simd_sgemm('n','n',M,N,K,1.0f,(ELE_TYPE*)A,K, (ELE_TYPE*)B,N,0.0f,(ELE_TYPE*)C,N);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    printf("tile_omp_simd函数执行时间: %.6f 秒\n", duration.count() / 1000000.0);
    printf("C[0][0] = %f\n", C(0,0,N));
    printf("tile_omp_simd 计算速度为%f GFLOPS\n", 2.0f * M * N * K / (duration.count()/ 1000000.0) / 1000000000.0);
    printf("--------------------------------\n");
    memset(C,0,M*N*sizeof(float));
    start = std::chrono::high_resolution_clock::now();
    tile_omp_simd_unroll_sgemm('n','n',M,N,K,1.0f,(ELE_TYPE*)A,K, (ELE_TYPE*)B,N,0.0f,(ELE_TYPE*)C,N);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    printf("tile_omp_simd_unroll函数执行时间: %.6f 秒\n", duration.count() / 1000000.0);
    printf("C[0][0] = %f\n", C(0,0,N));
    printf("tile_omp_simd_unroll 计算速度为%f GFLOPS\n", 2.0f * M * N * K / (duration.count()/ 1000000.0) / 1000000000.0);
    printf("--------------------------------\n");
    memset(C,0,M*N*sizeof(float));
    start = std::chrono::high_resolution_clock::now();
    tile_omp_simd_prefetch_sgemm('n','n',M,N,K,1.0f,(ELE_TYPE*)A,K, (ELE_TYPE*)B,N,0.0f,(ELE_TYPE*)C,N);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    printf("tile_omp_simd_prefetch函数执行时间: %.6f 秒\n", duration.count() / 1000000.0);
    printf("C[0][0] = %f\n", C(0,0,N));
    printf("tile_omp_simd_prefetch 计算速度为%f GFLOPS\n", 2.0f * M * N * K / (duration.count()/ 1000000.0) / 1000000000.0);
     
    
    // 释放内存
    free(A);
    free(B);
    free(C);
    
    return 0;
}