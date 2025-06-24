#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <immintrin.h>  // Intel intrinsics for AVX/SSE

// 检查系统是否支持AVX指令集
int check_avx_support() {
    int cpuInfo[4];
    __cpuid(cpuInfo, 1);
    return (cpuInfo[2] & (1 << 28)) != 0;  // Check AVX bit
}

// 使用SSE指令集的矩阵乘法（处理4个float/int元素）
void matrixmultiply_sse(int N, int **matrixA, int **matrixB, int **matrixC) {
    int i, j, k;
    
    // 初始化结果矩阵
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            matrixC[i][j] = 0;
        }
    }
    
    // 确保N是4的倍数，否则处理剩余元素
    int N_simd = (N / 4) * 4;
    
    for (i = 0; i < N; i++) {
        for (k = 0; k < N; k++) {
            // 广播matrixA[i][k]到所有4个位置
            __m128i a_broadcast = _mm_set1_epi32(matrixA[i][k]);
            
            // 使用SIMD处理4个元素
            for (j = 0; j < N_simd; j += 4) {
                // 加载matrixB的4个元素
                __m128i b_vec = _mm_loadu_si128((__m128i*)&matrixB[k][j]);
                
                // 加载matrixC的4个元素
                __m128i c_vec = _mm_loadu_si128((__m128i*)&matrixC[i][j]);
                
                // 执行乘法：a_broadcast * b_vec
                __m128i prod = _mm_mullo_epi32(a_broadcast, b_vec);
                
                // 执行加法：c_vec + prod
                __m128i result = _mm_add_epi32(c_vec, prod);
                
                // 存储结果
                _mm_storeu_si128((__m128i*)&matrixC[i][j], result);
            }
            
            // 处理剩余的元素（非4的倍数部分）
            for (j = N_simd; j < N; j++) {
                matrixC[i][j] += matrixA[i][k] * matrixB[k][j];
            }
        }
    }
}

// 使用AVX2指令集的矩阵乘法（处理8个int元素）
void matrixmultiply_avx2(int N, int **matrixA, int **matrixB, int **matrixC) {
    int i, j, k;
    
    // 初始化结果矩阵
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            matrixC[i][j] = 0;
        }
    }
    
    // 确保N是8的倍数，否则处理剩余元素
    int N_simd = (N / 8) * 8;
    
    for (i = 0; i < N; i++) {
        for (k = 0; k < N; k++) {
            // 广播matrixA[i][k]到所有8个位置
            __m256i a_broadcast = _mm256_set1_epi32(matrixA[i][k]);
            
            // 使用SIMD处理8个元素
            for (j = 0; j < N_simd; j += 8) {
                // 加载matrixB的8个元素
                __m256i b_vec = _mm256_loadu_si256((__m256i*)&matrixB[k][j]);
                
                // 加载matrixC的8个元素
                __m256i c_vec = _mm256_loadu_si256((__m256i*)&matrixC[i][j]);
                
                // 执行乘法：a_broadcast * b_vec
                __m256i prod = _mm256_mullo_epi32(a_broadcast, b_vec);
                
                // 执行加法：c_vec + prod
                __m256i result = _mm256_add_epi32(c_vec, prod);
                
                // 存储结果
                _mm256_storeu_si256((__m256i*)&matrixC[i][j], result);
            }
            
            // 处理剩余的元素（非8的倍数部分）
            for (j = N_simd; j < N; j++) {
                matrixC[i][j] += matrixA[i][k] * matrixB[k][j];
            }
        }
    }
}

// 组合SIMD和分块的优化版本
void matrixmultiply_simd_blocked(int N, int **matrixA, int **matrixB, int **matrixC) {
    int i, j, k, ii, jj, kk;
    const int BLOCK_SIZE = 64;
    
    // 初始化结果矩阵
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            matrixC[i][j] = 0;
        }
    }
    
    // 分块矩阵乘法 + SIMD优化
    for (kk = 0; kk < N; kk += BLOCK_SIZE) {
        for (ii = 0; ii < N; ii += BLOCK_SIZE) {
            for (jj = 0; jj < N; jj += BLOCK_SIZE) {
                // 在每个块内使用SIMD优化
                for (k = kk; k < kk + BLOCK_SIZE && k < N; k++) {
                    for (i = ii; i < ii + BLOCK_SIZE && i < N; i++) {
                        // 广播matrixA[i][k]
                        __m256i a_broadcast = _mm256_set1_epi32(matrixA[i][k]);
                        
                        int j_end = (jj + BLOCK_SIZE < N) ? jj + BLOCK_SIZE : N;
                        int j_simd = jj + ((j_end - jj) / 8) * 8;
                        
                        // SIMD处理
                        for (j = jj; j < j_simd; j += 8) {
                            __m256i b_vec = _mm256_loadu_si256((__m256i*)&matrixB[k][j]);
                            __m256i c_vec = _mm256_loadu_si256((__m256i*)&matrixC[i][j]);
                            __m256i prod = _mm256_mullo_epi32(a_broadcast, b_vec);
                            __m256i result = _mm256_add_epi32(c_vec, prod);
                            _mm256_storeu_si256((__m256i*)&matrixC[i][j], result);
                        }
                        
                        // 处理剩余元素
                        for (j = j_simd; j < j_end; j++) {
                            matrixC[i][j] += matrixA[i][k] * matrixB[k][j];
                        }
                    }
                }
            }
        }
    }
}

// 通用SIMD接口函数
void matrixmultiply_simd(int N, int **matrixA, int **matrixB, int **matrixC) {
    // 检查AVX2支持
    int cpuInfo[4];
    __cpuid(cpuInfo, 7);
    int has_avx2 = (cpuInfo[1] & (1 << 5)) != 0;
    
    if (has_avx2) {
        printf("Using AVX2 instruction set optimization\n");
        matrixmultiply_avx2(N, matrixA, matrixB, matrixC);
    } else {
        printf("Using SSE instruction set optimization\n");
        matrixmultiply_sse(N, matrixA, matrixB, matrixC);
    }
}

// 辅助函数
int** create_matrix(int N) {
    // 使用对齐内存分配以优化SIMD访问
    int **matrix = (int**)malloc(N * sizeof(int*));
    for (int i = 0; i < N; i++) {
        matrix[i] = (int*)_aligned_malloc(N * sizeof(int), 32); // 32字节对齐
    }
    return matrix;
}

void free_matrix(int **matrix, int N) {
    for (int i = 0; i < N; i++) {
        _aligned_free(matrix[i]);
    }
    free(matrix);
}

void init_test_matrices(int N, int **matrixA, int **matrixB) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrixA[i][j] = i + j;
            matrixB[i][j] = i * j + 1;
        }
    }
}

#ifdef STANDALONE_TEST
int main() {
    int N = 1024; // 测试矩阵大小
    printf("测试SIMD优化版本矩阵乘法，矩阵大小: %dx%d\n", N, N);
    
    // 检查CPU支持
    printf("检查CPU指令集支持:\n");
    if (check_avx_support()) {
        printf("- AVX: 支持\n");
    } else {
        printf("- AVX: 不支持\n");
    }
    
    // 创建矩阵
    int **matrixA = create_matrix(N);
    int **matrixB = create_matrix(N);
    int **matrixC = create_matrix(N);
    
    // 初始化测试数据
    init_test_matrices(N, matrixA, matrixB);
    
    // 测试SSE版本
    printf("\n测试SSE版本:\n");
    clock_t start = clock();
    matrixmultiply_sse(N, matrixA, matrixB, matrixC);
    clock_t end = clock();
    double time_sse = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("SSE版本执行时间: %.4f 秒\n", time_sse);
    
    // 重置结果矩阵
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrixC[i][j] = 0;
        }
    }
    
    // 测试AVX2版本
    printf("\n测试AVX2版本:\n");
    start = clock();
    matrixmultiply_avx2(N, matrixA, matrixB, matrixC);
    end = clock();
    double time_avx2 = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("AVX2版本执行时间: %.4f 秒\n", time_avx2);
    printf("AVX2相对SSE加速: %.2fx\n", time_sse/time_avx2);
    
    // 重置结果矩阵
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrixC[i][j] = 0;
        }
    }
    
    // 测试SIMD+分块版本
    printf("\n测试SIMD+分块组合版本:\n");
    start = clock();
    matrixmultiply_simd_blocked(N, matrixA, matrixB, matrixC);
    end = clock();
    double time_combined = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("SIMD+分块版本执行时间: %.4f 秒\n", time_combined);
    printf("组合优化相对AVX2加速: %.2fx\n", time_avx2/time_combined);
    
    // 释放内存
    free_matrix(matrixA, N);
    free_matrix(matrixB, N);
    free_matrix(matrixC, N);
    
    return 0;
}
#endif 