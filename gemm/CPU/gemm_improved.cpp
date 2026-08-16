#include <stdio.h>
#include <string.h>
#include <chrono>
#include <stdlib.h>
#include <immintrin.h>
#include <omp.h>

typedef float ELE_TYPE;

// 1. 正确的循环重排序 - i-k-j顺序，C矩阵写入连续
void correct_loop_reorder_sgemm(
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
    
    // 初始化C矩阵
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            dst[i + j * ldc] *= beta;
        }
    }
    
    // i-k-j循环顺序：C矩阵写入连续，A元素重复利用
    for(int i = 0; i < M; i++) {
        for(int k = 0; k < K; k++) {
            float a_ik = alpha * src_a[i * a_stride_m + k * a_stride_k];
            for(int j = 0; j < N; j++) {
                dst[i + j * ldc] += a_ik * src_b[k * b_stride_k + j * b_stride_n];
            }
        }
    }
}

// 2. 优化的分块GEMM - 更小的块大小，更好的缓存利用
void optimized_blocking_sgemm(
    char transa, char transb,
    int M, int N, int K, 
    const float alpha,
    const float * src_a, int lda,
    const float * src_b, int ldb, 
    const float beta,
    float * dst, int ldc)
{
    const int BM = 64;   // 针对L1缓存优化的块大小
    const int BN = 64;
    const int BK = 64;
    
    int a_stride_m = transa == 'n' ? 1 : lda;
    int a_stride_k = transa == 'n' ? lda : 1;
    int b_stride_k = transb == 'n' ? 1 : ldb;
    int b_stride_n = transb == 'n' ? ldb : 1;
    
    // 初始化C矩阵
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            dst[i + j * ldc] *= beta;
        }
    }
    
    // 分块计算
    for(int ii = 0; ii < M; ii += BM) {
        int i_end = (ii + BM < M) ? ii + BM : M;
        
        for(int jj = 0; jj < N; jj += BN) {
            int j_end = (jj + BN < N) ? jj + BN : N;
            
            for(int kk = 0; kk < K; kk += BK) {
                int k_end = (kk + BK < K) ? kk + BK : K;
                
                // 微内核 - 优化的内层循环
                for(int i = ii; i < i_end; i++) {
                    for(int k = kk; k < k_end; k++) {
                        float a_ik = alpha * src_a[i * a_stride_m + k * a_stride_k];
                        for(int j = jj; j < j_end; j++) {
                            dst[i + j * ldc] += a_ik * src_b[k * b_stride_k + j * b_stride_n];
                        }
                    }
                }
            }
        }
    }
}

// 3. 结合SIMD和循环展开的优化版本
void simd_unroll_sgemm(
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
    
    const int vec_size = 8;
    __m256 alpha_vec = _mm256_set1_ps(alpha);
    __m256 beta_vec = _mm256_set1_ps(beta);
    
    // 初始化C矩阵
    for(int i = 0; i < M; i++) {
        int j = 0;
        for(; j <= N - vec_size; j += vec_size) {
            __m256 c_vec = _mm256_loadu_ps(&dst[i + j * ldc]);
            c_vec = _mm256_mul_ps(c_vec, beta_vec);
            _mm256_storeu_ps(&dst[i + j * ldc], c_vec);
        }
        for(; j < N; j++) {
            dst[i + j * ldc] *= beta;
        }
    }
    
    // SIMD优化的主循环，增加循环展开
    for(int i = 0; i < M; i++) {
        for(int k = 0; k < K; k++) {
            __m256 a_ik_vec = _mm256_set1_ps(alpha * src_a[i * a_stride_m + k * a_stride_k]);
            
            int j = 0;
            // 循环展开：每次处理32个元素（4个向量）
            for(; j <= N - 32; j += 32) {
                // 第一个向量
                __m256 b_vec1 = _mm256_loadu_ps(&src_b[k * b_stride_k + j * b_stride_n]);
                __m256 c_vec1 = _mm256_loadu_ps(&dst[i + j * ldc]);
                c_vec1 = _mm256_fmadd_ps(a_ik_vec, b_vec1, c_vec1);
                _mm256_storeu_ps(&dst[i + j * ldc], c_vec1);
                
                // 第二个向量
                __m256 b_vec2 = _mm256_loadu_ps(&src_b[k * b_stride_k + (j+8) * b_stride_n]);
                __m256 c_vec2 = _mm256_loadu_ps(&dst[i + (j+8) * ldc]);
                c_vec2 = _mm256_fmadd_ps(a_ik_vec, b_vec2, c_vec2);
                _mm256_storeu_ps(&dst[i + (j+8) * ldc], c_vec2);
                
                // 第三个向量
                __m256 b_vec3 = _mm256_loadu_ps(&src_b[k * b_stride_k + (j+16) * b_stride_n]);
                __m256 c_vec3 = _mm256_loadu_ps(&dst[i + (j+16) * ldc]);
                c_vec3 = _mm256_fmadd_ps(a_ik_vec, b_vec3, c_vec3);
                _mm256_storeu_ps(&dst[i + (j+16) * ldc], c_vec3);
                
                // 第四个向量
                __m256 b_vec4 = _mm256_loadu_ps(&src_b[k * b_stride_k + (j+24) * b_stride_n]);
                __m256 c_vec4 = _mm256_loadu_ps(&dst[i + (j+24) * ldc]);
                c_vec4 = _mm256_fmadd_ps(a_ik_vec, b_vec4, c_vec4);
                _mm256_storeu_ps(&dst[i + (j+24) * ldc], c_vec4);
            }
            
            // 处理剩余的向量化部分
            for(; j <= N - vec_size; j += vec_size) {
                __m256 b_vec = _mm256_loadu_ps(&src_b[k * b_stride_k + j * b_stride_n]);
                __m256 c_vec = _mm256_loadu_ps(&dst[i + j * ldc]);
                c_vec = _mm256_fmadd_ps(a_ik_vec, b_vec, c_vec);
                _mm256_storeu_ps(&dst[i + j * ldc], c_vec);
            }
            
            // 处理剩余元素
            float a_ik = alpha * src_a[i * a_stride_m + k * a_stride_k];
            for(; j < N; j++) {
                dst[i + j * ldc] += a_ik * src_b[k * b_stride_k + j * b_stride_n];
            }
        }
    }
}

// 4. OpenMP并行化版本
void parallel_sgemm(
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
    
    // 初始化C矩阵
    #pragma omp parallel for
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            dst[i + j * ldc] *= beta;
        }
    }
    
    // 并行化的主循环
    #pragma omp parallel for
    for(int i = 0; i < M; i++) {
        for(int k = 0; k < K; k++) {
            float a_ik = alpha * src_a[i * a_stride_m + k * a_stride_k];
            for(int j = 0; j < N; j++) {
                dst[i + j * ldc] += a_ik * src_b[k * b_stride_k + j * b_stride_n];
            }
        }
    }
}

// 5. 最终优化版本：结合所有技术
void ultimate_sgemm(
    char transa, char transb,
    int M, int N, int K, 
    const float alpha,
    const float * src_a, int lda,
    const float * src_b, int ldb, 
    const float beta,
    float * dst, int ldc)
{
    const int BM = 128;  // 分块大小
    const int BN = 128;
    const int BK = 64;
    
    int a_stride_m = transa == 'n' ? 1 : lda;
    int a_stride_k = transa == 'n' ? lda : 1;
    int b_stride_k = transb == 'n' ? 1 : ldb;
    int b_stride_n = transb == 'n' ? ldb : 1;
    
    const int vec_size = 8;
    __m256 alpha_vec = _mm256_set1_ps(alpha);
    __m256 beta_vec = _mm256_set1_ps(beta);
    
    // 并行化的分块GEMM
    #pragma omp parallel for collapse(2)
    for(int ii = 0; ii < M; ii += BM) {
        for(int jj = 0; jj < N; jj += BN) {
            int i_end = (ii + BM < M) ? ii + BM : M;
            int j_end = (jj + BN < N) ? jj + BN : N;
            
            // 初始化当前块的C
            for(int i = ii; i < i_end; i++) {
                int j = jj;
                for(; j <= j_end - vec_size; j += vec_size) {
                    __m256 c_vec = _mm256_loadu_ps(&dst[i + j * ldc]);
                    c_vec = _mm256_mul_ps(c_vec, beta_vec);
                    _mm256_storeu_ps(&dst[i + j * ldc], c_vec);
                }
                for(; j < j_end; j++) {
                    dst[i + j * ldc] *= beta;
                }
            }
            
            // 分块计算
            for(int kk = 0; kk < K; kk += BK) {
                int k_end = (kk + BK < K) ? kk + BK : K;
                
                for(int i = ii; i < i_end; i++) {
                    for(int k = kk; k < k_end; k++) {
                        __m256 a_ik_vec = _mm256_set1_ps(alpha * src_a[i * a_stride_m + k * a_stride_k]);
                        
                        int j = jj;
                        for(; j <= j_end - vec_size; j += vec_size) {
                            __m256 b_vec = _mm256_loadu_ps(&src_b[k * b_stride_k + j * b_stride_n]);
                            __m256 c_vec = _mm256_loadu_ps(&dst[i + j * ldc]);
                            c_vec = _mm256_fmadd_ps(a_ik_vec, b_vec, c_vec);
                            _mm256_storeu_ps(&dst[i + j * ldc], c_vec);
                        }
                        
                        // 处理剩余元素
                        float a_ik = alpha * src_a[i * a_stride_m + k * a_stride_k];
                        for(; j < j_end; j++) {
                            dst[i + j * ldc] += a_ik * src_b[k * b_stride_k + j * b_stride_n];
                        }
                    }
                }
            }
        }
    }
}

// 性能测试函数
void benchmark_improved_versions() {
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
    
    // 测试各个改进版本
    struct {
        const char* name;
        void (*func)(char, char, int, int, int, const float, const float*, int, const float*, int, const float, float*, int);
    } versions[] = {
        {"正确的循环重排序", correct_loop_reorder_sgemm},
        {"优化的分块算法", optimized_blocking_sgemm},
        {"SIMD+循环展开", simd_unroll_sgemm},
        {"OpenMP并行化", parallel_sgemm},
        {"终极优化版本", ultimate_sgemm}
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
    printf("=== 改进的GEMM内存重拍优化测试 ===\n");
    printf("使用OpenMP线程数: %d\n", omp_get_max_threads());
    benchmark_improved_versions();
    return 0;
} 