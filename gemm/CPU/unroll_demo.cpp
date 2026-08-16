#include <stdio.h>
#include <immintrin.h>
#include <chrono>
#include <stdlib.h>
#include <string.h>

// 1. 原始手工展开版本（你当前的代码）
void manual_unroll_version(float* dst, const float* src_b, __m256 a_vec, 
                          int ii, int jj, int j_end, int kk, int ldc, 
                          int b_stride_k, int b_stride_n) {
    const int vec_size = 8;
    const int unroll = 4;
    
    // 手工展开：一次处理32个元素（4个向量）
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
}

// 2. 使用#pragma unroll的版本
void pragma_unroll_version(float* dst, const float* src_b, __m256 a_vec, 
                          int ii, int jj, int j_end, int kk, int ldc, 
                          int b_stride_k, int b_stride_n) {
    const int vec_size = 8;
    const int unroll = 4;
    
    // 改写成循环形式，然后使用#pragma unroll
    #pragma unroll(4)  // 或者 #pragma GCC unroll 4
    for(int u = 0; u < unroll; u++) {
        if(jj + u * vec_size < j_end - vec_size) {
            int offset = u * vec_size;
            __m256 b_vec = _mm256_loadu_ps(&src_b[kk * b_stride_k + (jj + offset) * b_stride_n]);
            __m256 c_vec = _mm256_loadu_ps(&dst[ii + (jj + offset) * ldc]);
            c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
            _mm256_storeu_ps(&dst[ii + (jj + offset) * ldc], c_vec);
        }
    }
}

// 3. 编译器友好的模板版本
template<int UNROLL_FACTOR>
inline void template_unroll_version(float* dst, const float* src_b, __m256 a_vec, 
                                   int ii, int jj, int j_end, int kk, int ldc, 
                                   int b_stride_k, int b_stride_n) {
    const int vec_size = 8;
    
    #pragma unroll(UNROLL_FACTOR)
    for(int u = 0; u < UNROLL_FACTOR; u++) {
        if(jj + u * vec_size < j_end - vec_size) {
            int offset = u * vec_size;
            __m256 b_vec = _mm256_loadu_ps(&src_b[kk * b_stride_k + (jj + offset) * b_stride_n]);
            __m256 c_vec = _mm256_loadu_ps(&dst[ii + (jj + offset) * ldc]);
            c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
            _mm256_storeu_ps(&dst[ii + (jj + offset) * ldc], c_vec);
        }
    }
}

// 4. 最佳实践：结合宏的版本
#define SIMD_UNROLL_KERNEL(offset) do { \
    __m256 b_vec##offset = _mm256_loadu_ps(&src_b[kk * b_stride_k + (jj + offset) * b_stride_n]); \
    __m256 c_vec##offset = _mm256_loadu_ps(&dst[ii + (jj + offset) * ldc]); \
    c_vec##offset = _mm256_fmadd_ps(a_vec, b_vec##offset, c_vec##offset); \
    _mm256_storeu_ps(&dst[ii + (jj + offset) * ldc], c_vec##offset); \
} while(0)

void macro_unroll_version(float* dst, const float* src_b, __m256 a_vec, 
                         int ii, int jj, int j_end, int kk, int ldc, 
                         int b_stride_k, int b_stride_n) {
    const int vec_size = 8;
    
    if(jj + 4 * vec_size <= j_end) {
        SIMD_UNROLL_KERNEL(0);
        SIMD_UNROLL_KERNEL(vec_size);
        SIMD_UNROLL_KERNEL(2 * vec_size);
        SIMD_UNROLL_KERNEL(3 * vec_size);
    }
}

// 5. 更灵活的版本：动态展开因子
void flexible_unroll_version(float* dst, const float* src_b, __m256 a_vec, 
                            int ii, int jj, int j_end, int kk, int ldc, 
                            int b_stride_k, int b_stride_n, int unroll_factor) {
    const int vec_size = 8;
    
    // 使用变长数组（VLA）或者alloca
    __m256 b_vecs[unroll_factor];
    __m256 c_vecs[unroll_factor];
    
    // 加载
    #pragma unroll
    for(int u = 0; u < unroll_factor; u++) {
        if(jj + u * vec_size < j_end - vec_size) {
            int offset = u * vec_size;
            b_vecs[u] = _mm256_loadu_ps(&src_b[kk * b_stride_k + (jj + offset) * b_stride_n]);
            c_vecs[u] = _mm256_loadu_ps(&dst[ii + (jj + offset) * ldc]);
        }
    }
    
    // 计算
    #pragma unroll
    for(int u = 0; u < unroll_factor; u++) {
        if(jj + u * vec_size < j_end - vec_size) {
            c_vecs[u] = _mm256_fmadd_ps(a_vec, b_vecs[u], c_vecs[u]);
        }
    }
    
    // 存储
    #pragma unroll
    for(int u = 0; u < unroll_factor; u++) {
        if(jj + u * vec_size < j_end - vec_size) {
            int offset = u * vec_size;
            _mm256_storeu_ps(&dst[ii + (jj + offset) * ldc], c_vecs[u]);
        }
    }
}

// 性能测试函数
void benchmark_unroll_methods() {
    const int size = 1024;
    float* dst = (float*)aligned_alloc(32, size * sizeof(float));
    float* src_b = (float*)aligned_alloc(32, size * sizeof(float));
    
    // 初始化数据
    for(int i = 0; i < size; i++) {
        dst[i] = 1.0f;
        src_b[i] = 2.0f;
    }
    
    __m256 a_vec = _mm256_set1_ps(3.0f);
    
    const int iterations = 1000000;
    
    // 测试手工展开
    auto start = std::chrono::high_resolution_clock::now();
    for(int iter = 0; iter < iterations; iter++) {
        manual_unroll_version(dst, src_b, a_vec, 0, 0, size-32, 0, size, 1, 1);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // 测试pragma unroll
    start = std::chrono::high_resolution_clock::now();
    for(int iter = 0; iter < iterations; iter++) {
        pragma_unroll_version(dst, src_b, a_vec, 0, 0, size-32, 0, size, 1, 1);
    }
    end = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // 测试模板版本
    start = std::chrono::high_resolution_clock::now();
    for(int iter = 0; iter < iterations; iter++) {
        template_unroll_version<4>(dst, src_b, a_vec, 0, 0, size-32, 0, size, 1, 1);
    }
    end = std::chrono::high_resolution_clock::now();
    auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    printf("手工展开版本:    %.3f ms\n", duration1.count() / 1000.0);
    printf("pragma unroll:   %.3f ms\n", duration2.count() / 1000.0);
    printf("模板版本:        %.3f ms\n", duration3.count() / 1000.0);
    
    free(dst);
    free(src_b);
}

int main() {
    printf("=== 循环展开方法性能对比 ===\n");
    benchmark_unroll_methods();
    
    printf("\n=== 编译器支持情况 ===\n");
    #ifdef __GNUC__
    printf("GCC编译器: 支持 #pragma GCC unroll\n");
    #endif
    
    #ifdef __clang__
    printf("Clang编译器: 支持 #pragma unroll\n");
    #endif
    
    #ifdef __INTEL_COMPILER
    printf("Intel编译器: 支持 #pragma unroll\n");
    #endif
    
    return 0;
} 