#include <stdio.h>
#include <string.h>
#include <chrono>
#include <stdlib.h>
#include <immintrin.h>  // for AVX instructions

typedef float ELE_TYPE;

// 宏定义保持不变
#define A(i,j,lda) A[(i)*(lda)+(j)]
#define B(i,j,ldb) B[(i)*(ldb)+(j)]
#define C(i,j,ldc) C[(i)*(ldc)+(j)]

// 1. 循环重排序优化 - 最内层循环访问连续内存
void optimized_loop_order_sgemm(
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
    
    // 优化的循环顺序：i-k-j，使B矩阵访问连续
    for(int i = 0; i < M; i++) {
        for(int k = 0; k < K; k++) {
            float a_ik = src_a[i * a_stride_m + k * a_stride_k];
            for(int j = 0; j < N; j++) {
                dst[i + j * ldc] += alpha * a_ik * src_b[k * b_stride_k + j * b_stride_n];
            }
        }
        // 处理beta项
        if(beta != 1.0f) {
            for(int j = 0; j < N; j++) {
                dst[i + j * ldc] = dst[i + j * ldc] + beta * dst[i + j * ldc];
            }
        }
    }
}

// 2. 分块优化 - 提高缓存局部性
void cache_blocking_sgemm(
    char transa, char transb,
    int M, int N, int K, 
    const float alpha,
    const float * src_a, int lda,
    const float * src_b, int ldb, 
    const float beta,
    float * dst, int ldc)
{
    const int MC = 256;  // L1缓存块大小
    const int NC = 256;
    const int KC = 128;
    
    int a_stride_m = transa == 'n' ? 1 : lda;
    int a_stride_k = transa == 'n' ? lda : 1;
    int b_stride_k = transb == 'n' ? 1 : ldb;
    int b_stride_n = transb == 'n' ? ldb : 1;
    
    // 三层分块循环
    for(int m_block = 0; m_block < M; m_block += MC) {
        int m_end = (m_block + MC < M) ? m_block + MC : M;
        
        for(int n_block = 0; n_block < N; n_block += NC) {
            int n_end = (n_block + NC < N) ? n_block + NC : N;
            
            // 初始化C块
            for(int i = m_block; i < m_end; i++) {
                for(int j = n_block; j < n_end; j++) {
                    dst[i + j * ldc] *= beta;
                }
            }
            
            for(int k_block = 0; k_block < K; k_block += KC) {
                int k_end = (k_block + KC < K) ? k_block + KC : K;
                
                // 内层微内核
                for(int i = m_block; i < m_end; i++) {
                    for(int k = k_block; k < k_end; k++) {
                        float a_ik = alpha * src_a[i * a_stride_m + k * a_stride_k];
                        for(int j = n_block; j < n_end; j++) {
                            dst[i + j * ldc] += a_ik * src_b[k * b_stride_k + j * b_stride_n];
                        }
                    }
                }
            }
        }
    }
}

// 3. 数据重排列优化 - 改善内存访问模式
void data_packing_sgemm(
    char transa, char transb,
    int M, int N, int K, 
    const float alpha,
    const float * src_a, int lda,
    const float * src_b, int ldb, 
    const float beta,
    float * dst, int ldc)
{
    const int MC = 256;
    const int NC = 256;
    const int KC = 128;
    
    // 分配打包缓冲区
    float *packed_a = (float*)aligned_alloc(32, MC * KC * sizeof(float));
    float *packed_b = (float*)aligned_alloc(32, KC * NC * sizeof(float));
    
    if(!packed_a || !packed_b) {
        printf("内存分配失败\n");
        if(packed_a) free(packed_a);
        if(packed_b) free(packed_b);
        return;
    }
    
    int a_stride_m = transa == 'n' ? 1 : lda;
    int a_stride_k = transa == 'n' ? lda : 1;
    int b_stride_k = transb == 'n' ? 1 : ldb;
    int b_stride_n = transb == 'n' ? ldb : 1;
    
    for(int m_block = 0; m_block < M; m_block += MC) {
        int m_end = (m_block + MC < M) ? m_block + MC : M;
        int m_size = m_end - m_block;
        
        for(int n_block = 0; n_block < N; n_block += NC) {
            int n_end = (n_block + NC < N) ? n_block + NC : N;
            int n_size = n_end - n_block;
            
            // 初始化C块
            for(int i = m_block; i < m_end; i++) {
                for(int j = n_block; j < n_end; j++) {
                    dst[i + j * ldc] *= beta;
                }
            }
            
            for(int k_block = 0; k_block < K; k_block += KC) {
                int k_end = (k_block + KC < K) ? k_block + KC : K;
                int k_size = k_end - k_block;
                
                // 打包A矩阵 - 行主序到列主序
                for(int k = 0; k < k_size; k++) {
                    for(int i = 0; i < m_size; i++) {
                        packed_a[k * m_size + i] = src_a[(m_block + i) * a_stride_m + (k_block + k) * a_stride_k];
                    }
                }
                
                // 打包B矩阵 - 保持行主序但重新排列
                for(int k = 0; k < k_size; k++) {
                    for(int j = 0; j < n_size; j++) {
                        packed_b[k * n_size + j] = src_b[(k_block + k) * b_stride_k + (n_block + j) * b_stride_n];
                    }
                }
                
                // 优化的微内核计算
                for(int k = 0; k < k_size; k++) {
                    for(int i = 0; i < m_size; i++) {
                        float a_val = alpha * packed_a[k * m_size + i];
                        float *b_row = &packed_b[k * n_size];
                        float *c_row = &dst[(m_block + i) + n_block * ldc];
                        
                        // 向量化内层循环
                        for(int j = 0; j < n_size; j++) {
                            c_row[j * ldc] += a_val * b_row[j];
                        }
                    }
                }
            }
        }
    }
    
    free(packed_a);
    free(packed_b);
}

// 4. SIMD向量化优化
void vectorized_sgemm(
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
    
    const int vec_size = 8; // AVX 256位，8个float
    __m256 alpha_vec = _mm256_set1_ps(alpha);
    __m256 beta_vec = _mm256_set1_ps(beta);
    
    for(int i = 0; i < M; i++) {
        for(int k = 0; k < K; k++) {
            __m256 a_ik_vec = _mm256_set1_ps(src_a[i * a_stride_m + k * a_stride_k]);
            a_ik_vec = _mm256_mul_ps(a_ik_vec, alpha_vec);
            
            int j = 0;
            // 向量化处理
            for(; j <= N - vec_size; j += vec_size) {
                // 加载B矩阵数据
                __m256 b_vec = _mm256_loadu_ps(&src_b[k * b_stride_k + j * b_stride_n]);
                
                // 加载C矩阵数据
                __m256 c_vec = _mm256_loadu_ps(&dst[i + j * ldc]);
                
                // 如果k==0，应用beta
                if(k == 0) {
                    c_vec = _mm256_mul_ps(c_vec, beta_vec);
                }
                
                // 计算并累加
                __m256 result = _mm256_fmadd_ps(a_ik_vec, b_vec, c_vec);
                
                // 存储结果
                _mm256_storeu_ps(&dst[i + j * ldc], result);
            }
            
            // 处理剩余元素
            for(; j < N; j++) {
                if(k == 0) {
                    dst[i + j * ldc] *= beta;
                }
                dst[i + j * ldc] += alpha * src_a[i * a_stride_m + k * a_stride_k] * 
                                   src_b[k * b_stride_k + j * b_stride_n];
            }
        }
    }
}

// 5. 多级缓存优化
void multilevel_cache_sgemm(
    char transa, char transb,
    int M, int N, int K, 
    const float alpha,
    const float * src_a, int lda,
    const float * src_b, int ldb, 
    const float beta,
    float * dst, int ldc)
{
    // L3缓存块大小 (通常几MB)
    const int L3_MC = 1024;
    const int L3_NC = 1024;
    const int L3_KC = 512;
    
    // L2缓存块大小 (通常256KB)
    const int L2_MC = 256;
    const int L2_NC = 256;
    const int L2_KC = 128;
    
    // L1缓存块大小 (通常32KB)
    const int L1_MC = 64;
    const int L1_NC = 64;
    const int L1_KC = 32;
    
    int a_stride_m = transa == 'n' ? 1 : lda;
    int a_stride_k = transa == 'n' ? lda : 1;
    int b_stride_k = transb == 'n' ? 1 : ldb;
    int b_stride_n = transb == 'n' ? ldb : 1;
    
    // L3级别分块
    for(int m3 = 0; m3 < M; m3 += L3_MC) {
        int m3_end = (m3 + L3_MC < M) ? m3 + L3_MC : M;
        
        for(int n3 = 0; n3 < N; n3 += L3_NC) {
            int n3_end = (n3 + L3_NC < N) ? n3 + L3_NC : N;
            
            for(int k3 = 0; k3 < K; k3 += L3_KC) {
                int k3_end = (k3 + L3_KC < K) ? k3 + L3_KC : K;
                
                // L2级别分块
                for(int m2 = m3; m2 < m3_end; m2 += L2_MC) {
                    int m2_end = (m2 + L2_MC < m3_end) ? m2 + L2_MC : m3_end;
                    
                    for(int n2 = n3; n2 < n3_end; n2 += L2_NC) {
                        int n2_end = (n2 + L2_NC < n3_end) ? n2 + L2_NC : n3_end;
                        
                        for(int k2 = k3; k2 < k3_end; k2 += L2_KC) {
                            int k2_end = (k2 + L2_KC < k3_end) ? k2 + L2_KC : k3_end;
                            
                            // L1级别分块 - 最内层优化循环
                            for(int m1 = m2; m1 < m2_end; m1 += L1_MC) {
                                int m1_end = (m1 + L1_MC < m2_end) ? m1 + L1_MC : m2_end;
                                
                                for(int n1 = n2; n1 < n2_end; n1 += L1_NC) {
                                    int n1_end = (n1 + L1_NC < n2_end) ? n1 + L1_NC : n2_end;
                                    
                                    for(int k1 = k2; k1 < k2_end; k1 += L1_KC) {
                                        int k1_end = (k1 + L1_KC < k2_end) ? k1 + L1_KC : k2_end;
                                        
                                        // 微内核 - 最优化的内层循环
                                        for(int i = m1; i < m1_end; i++) {
                                            for(int k = k1; k < k1_end; k++) {
                                                float a_ik = src_a[i * a_stride_m + k * a_stride_k];
                                                if(k3 == 0 && k2 == k3 && k1 == k2 && k == k1) {
                                                    // 第一次计算时应用beta
                                                    for(int j = n1; j < n1_end; j++) {
                                                        dst[i + j * ldc] = alpha * a_ik * 
                                                            src_b[k * b_stride_k + j * b_stride_n] + 
                                                            beta * dst[i + j * ldc];
                                                    }
                                                } else {
                                                    for(int j = n1; j < n1_end; j++) {
                                                        dst[i + j * ldc] += alpha * a_ik * 
                                                            src_b[k * b_stride_k + j * b_stride_n];
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// 性能测试函数
void benchmark_gemm_versions() {
    const int M = 1024, N = 1024, K = 1024;
    float *A = (float*)aligned_alloc(32, M * K * sizeof(float));
    float *B = (float*)aligned_alloc(32, K * N * sizeof(float));
    float *C = (float*)aligned_alloc(32, M * N * sizeof(float));
    
    if (!A || !B || !C) {
        printf("内存分配失败！\n");
        return;
    }
    
    // 初始化数据
    for(int i = 0; i < M * K; i++) A[i] = 2.0f;
    for(int i = 0; i < K * N; i++) B[i] = 2.0f;
    for(int i = 0; i < M * N; i++) C[i] = 0.0f;
    
    // 测试各个版本的性能
    struct {
        const char* name;
        void (*func)(char, char, int, int, int, const float, const float*, int, const float*, int, const float, float*, int);
    } versions[] = {
        {"循环重排序优化", optimized_loop_order_sgemm},
        {"缓存分块优化", cache_blocking_sgemm},
        {"数据重排列优化", data_packing_sgemm},
        {"SIMD向量化优化", vectorized_sgemm},
        {"多级缓存优化", multilevel_cache_sgemm}
    };
    
    for(int v = 0; v < 5; v++) {
        // 重置C矩阵
        for(int i = 0; i < M * N; i++) C[i] = 0.0f;
        
        auto start = std::chrono::high_resolution_clock::now();
        versions[v].func('n', 'n', M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double gflops = 2.0 * M * N * K / (duration.count() / 1000000.0) / 1000000000.0;
        
        printf("%s: %.3f ms, %.2f GFLOPS, C[0][0] = %f\n", 
               versions[v].name, duration.count() / 1000.0, gflops, C[0]);
    }
    
    free(A);
    free(B);
    free(C);
}

int main() {
    printf("=== GEMM内存重拍优化测试 ===\n");
    benchmark_gemm_versions();
    return 0;
} 