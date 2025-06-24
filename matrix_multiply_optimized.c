#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <immintrin.h>
#include <windows.h>
#include <process.h>

// 循环展开的优化版本
void matrixmultiply_unrolled(int N, int **matrixA, int **matrixB, int **matrixC) {
    int i, j, k;
    
    // 初始化结果矩阵
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            matrixC[i][j] = 0;
        }
    }
    
    // 循环展开优化，每次处理4个元素
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j += 4) {
            int sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
            
            for (k = 0; k < N; k++) {
                int a_ik = matrixA[i][k];
                sum0 += a_ik * matrixB[k][j];
                if (j + 1 < N) sum1 += a_ik * matrixB[k][j + 1];
                if (j + 2 < N) sum2 += a_ik * matrixB[k][j + 2];
                if (j + 3 < N) sum3 += a_ik * matrixB[k][j + 3];
            }
            
            matrixC[i][j] = sum0;
            if (j + 1 < N) matrixC[i][j + 1] = sum1;
            if (j + 2 < N) matrixC[i][j + 2] = sum2;
            if (j + 3 < N) matrixC[i][j + 3] = sum3;
        }
    }
}

// 矩阵转置优化版本（改善数据局部性）
void matrixmultiply_transpose(int N, int **matrixA, int **matrixB, int **matrixC) {
    int i, j, k;
    
    // 创建转置矩阵B
    int **matrixB_T = (int**)malloc(N * sizeof(int*));
    for (i = 0; i < N; i++) {
        matrixB_T[i] = (int*)malloc(N * sizeof(int));
    }
    
    // 转置矩阵B
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            matrixB_T[j][i] = matrixB[i][j];
        }
    }
    
    // 初始化结果矩阵
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            matrixC[i][j] = 0;
        }
    }
    
    // 使用转置矩阵进行乘法（改善cache命中率）
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (k = 0; k < N; k++) {
                matrixC[i][j] += matrixA[i][k] * matrixB_T[j][k];
            }
        }
    }
    
    // 释放转置矩阵
    for (i = 0; i < N; i++) {
        free(matrixB_T[i]);
    }
    free(matrixB_T);
}

// 预取优化版本
void matrixmultiply_prefetch(int N, int **matrixA, int **matrixB, int **matrixC) {
    int i, j, k;
    
    // 初始化结果矩阵
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            matrixC[i][j] = 0;
        }
    }
    
    for (i = 0; i < N; i++) {
        for (k = 0; k < N; k++) {
            // 预取下一行的数据
            if (i + 1 < N) {
                _mm_prefetch((char*)&matrixA[i + 1][k], _MM_HINT_T0);
            }
            if (k + 1 < N) {
                _mm_prefetch((char*)&matrixB[k + 1][0], _MM_HINT_T0);
            }
            
            int temp = matrixA[i][k];
            for (j = 0; j < N; j++) {
                matrixC[i][j] += temp * matrixB[k][j];
            }
        }
    }
}

// 线程参数结构体
typedef struct {
    int N;
    int **matrixA;
    int **matrixB;
    int **matrixC;
    int start_row;
    int end_row;
    int thread_id;
} ThreadParams;

// 综合优化线程函数（分块 + SIMD + 循环展开）
unsigned __stdcall optimized_thread_function(void* arg) {
    ThreadParams* params = (ThreadParams*)arg;
    int N = params->N;
    int **matrixA = params->matrixA;
    int **matrixB = params->matrixB;
    int **matrixC = params->matrixC;
    int start_row = params->start_row;
    int end_row = params->end_row;
    
    const int BLOCK_SIZE = 64;
    
    // 分块 + SIMD优化
    for (int kk = 0; kk < N; kk += BLOCK_SIZE) {
        for (int ii = start_row; ii < end_row; ii += BLOCK_SIZE) {
            int ii_end = (ii + BLOCK_SIZE < end_row) ? ii + BLOCK_SIZE : end_row;
            
            for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
                int jj_end = (jj + BLOCK_SIZE < N) ? jj + BLOCK_SIZE : N;
                int kk_end = (kk + BLOCK_SIZE < N) ? kk + BLOCK_SIZE : N;
                
                // 在每个块内使用SIMD优化
                for (int k = kk; k < kk_end; k++) {
                    for (int i = ii; i < ii_end; i++) {
                        // 预取下一行数据
                        if (i + 1 < ii_end) {
                            _mm_prefetch((char*)&matrixA[i + 1][k], _MM_HINT_T0);
                        }
                        
                        // 广播matrixA[i][k]
                        __m256i a_broadcast = _mm256_set1_epi32(matrixA[i][k]);
                        
                        int j_simd = jj + ((jj_end - jj) / 8) * 8;
                        
                        // SIMD处理
                        for (int j = jj; j < j_simd; j += 8) {
                            __m256i b_vec = _mm256_loadu_si256((__m256i*)&matrixB[k][j]);
                            __m256i c_vec = _mm256_loadu_si256((__m256i*)&matrixC[i][j]);
                            __m256i prod = _mm256_mullo_epi32(a_broadcast, b_vec);
                            __m256i result = _mm256_add_epi32(c_vec, prod);
                            _mm256_storeu_si256((__m256i*)&matrixC[i][j], result);
                        }
                        
                        // 处理剩余元素
                        int temp = matrixA[i][k];
                        for (int j = j_simd; j < jj_end; j++) {
                            matrixC[i][j] += temp * matrixB[k][j];
                        }
                    }
                }
            }
        }
    }
    
    return 0;
}

// 终极优化版本：多线程 + 分块 + SIMD + 预取
void matrixmultiply_ultimate(int N, int **matrixA, int **matrixB, int **matrixC) {
    // 获取系统CPU核心数
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    int num_threads = sysinfo.dwNumberOfProcessors;
    
    if (num_threads > 8) num_threads = 8; // 限制线程数
    
    printf("Using ultimate optimization: %d threads + blocking + SIMD + prefetching\n", num_threads);
    
    // 初始化结果矩阵
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrixC[i][j] = 0;
        }
    }
    
    // 创建线程句柄和参数数组
    HANDLE* threads = (HANDLE*)malloc(num_threads * sizeof(HANDLE));
    ThreadParams* params = (ThreadParams*)malloc(num_threads * sizeof(ThreadParams));
    
    int rows_per_thread = N / num_threads;
    int remaining_rows = N % num_threads;
    
    // 创建并启动线程
    for (int t = 0; t < num_threads; t++) {
        params[t].N = N;
        params[t].matrixA = matrixA;
        params[t].matrixB = matrixB;
        params[t].matrixC = matrixC;
        params[t].start_row = t * rows_per_thread;
        params[t].end_row = (t + 1) * rows_per_thread;
        params[t].thread_id = t;
        
        // 最后一个线程处理剩余的行
        if (t == num_threads - 1) {
            params[t].end_row += remaining_rows;
        }
        
        threads[t] = (HANDLE)_beginthreadex(NULL, 0, optimized_thread_function, 
                                           &params[t], 0, NULL);
    }
    
    // 等待所有线程完成
    WaitForMultipleObjects(num_threads, threads, TRUE, INFINITE);
    
    // 清理资源
    for (int t = 0; t < num_threads; t++) {
        CloseHandle(threads[t]);
    }
    
    free(threads);
    free(params);
}

// Strassen算法的递归实现（仅作演示，对大矩阵效果更明显）
void strassen_add(int **A, int **B, int **C, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
}

void strassen_subtract(int **A, int **B, int **C, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            C[i][j] = A[i][j] - B[i][j];
        }
    }
}

// 简化版Strassen算法（阈值较小时使用基本算法）
void matrixmultiply_strassen(int N, int **matrixA, int **matrixB, int **matrixC) {
    // 对于小矩阵，直接使用基本算法
    if (N <= 64) {
        // 初始化结果矩阵
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                matrixC[i][j] = 0;
                for (int k = 0; k < N; k++) {
                    matrixC[i][j] += matrixA[i][k] * matrixB[k][j];
                }
            }
        }
        return;
    }
    
    printf("Using simplified Strassen algorithm for computation\n");
    
    // 对于大矩阵，这里简化为分块算法
    const int BLOCK_SIZE = N / 2;
    
    // 初始化结果矩阵
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrixC[i][j] = 0;
        }
    }
    
    // 分块计算
    for (int ii = 0; ii < N; ii += BLOCK_SIZE) {
        for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
            for (int kk = 0; kk < N; kk += BLOCK_SIZE) {
                for (int i = ii; i < ii + BLOCK_SIZE && i < N; i++) {
                    for (int j = jj; j < jj + BLOCK_SIZE && j < N; j++) {
                        for (int k = kk; k < kk + BLOCK_SIZE && k < N; k++) {
                            matrixC[i][j] += matrixA[i][k] * matrixB[k][j];
                        }
                    }
                }
            }
        }
    }
}

// 辅助函数
int** create_matrix(int N) {
    int **matrix = (int**)malloc(N * sizeof(int*));
    for (int i = 0; i < N; i++) {
        matrix[i] = (int*)_aligned_malloc(N * sizeof(int), 32);
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
    printf("测试各种高级优化版本矩阵乘法，矩阵大小: %dx%d\n", N, N);
    
    // 创建矩阵
    int **matrixA = create_matrix(N);
    int **matrixB = create_matrix(N);
    int **matrixC = create_matrix(N);
    
    // 初始化测试数据
    init_test_matrices(N, matrixA, matrixB);
    
    // 测试循环展开版本
    printf("\n1. 测试循环展开版本:\n");
    clock_t start = clock();
    matrixmultiply_unrolled(N, matrixA, matrixB, matrixC);
    clock_t end = clock();
    double time_unrolled = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("循环展开版本执行时间: %.4f 秒\n", time_unrolled);
    
    // 重置结果矩阵
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrixC[i][j] = 0;
        }
    }
    
    // 测试转置优化版本
    printf("\n2. 测试矩阵转置优化版本:\n");
    start = clock();
    matrixmultiply_transpose(N, matrixA, matrixB, matrixC);
    end = clock();
    double time_transpose = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("转置优化版本执行时间: %.4f 秒\n", time_transpose);
    
    // 重置结果矩阵
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrixC[i][j] = 0;
        }
    }
    
    // 测试预取优化版本
    printf("\n3. 测试预取优化版本:\n");
    start = clock();
    matrixmultiply_prefetch(N, matrixA, matrixB, matrixC);
    end = clock();
    double time_prefetch = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("预取优化版本执行时间: %.4f 秒\n", time_prefetch);
    
    // 重置结果矩阵
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrixC[i][j] = 0;
        }
    }
    
    // 测试终极优化版本
    printf("\n4. 测试终极优化版本:\n");
    start = clock();
    matrixmultiply_ultimate(N, matrixA, matrixB, matrixC);
    end = clock();
    double time_ultimate = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("终极优化版本执行时间: %.4f 秒\n", time_ultimate);
    
    // 性能对比
    printf("\n性能对比（以循环展开为基准）:\n");
    printf("转置优化加速: %.2fx\n", time_unrolled / time_transpose);
    printf("预取优化加速: %.2fx\n", time_unrolled / time_prefetch);
    printf("终极优化加速: %.2fx\n", time_unrolled / time_ultimate);
    
    // 释放内存
    free_matrix(matrixA, N);
    free_matrix(matrixB, N);
    free_matrix(matrixC, N);
    
    return 0;
}
#endif 