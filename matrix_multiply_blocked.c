#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

// 块大小定义，通常设置为L1 cache的大小，这里使用64
#define BLOCK_SIZE 64

void matrixmultiply_blocked(int N, int **matrixA, int **matrixB, int **matrixC) {
    int i, j, k, ii, jj, kk;
    
    // 初始化结果矩阵
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            matrixC[i][j] = 0;
        }
    }
    
    // 分块矩阵乘法
    for (ii = 0; ii < N; ii += BLOCK_SIZE) {
        for (jj = 0; jj < N; jj += BLOCK_SIZE) {
            for (kk = 0; kk < N; kk += BLOCK_SIZE) {
                // 在每个块内进行矩阵乘法
                for (i = ii; i < ii + BLOCK_SIZE && i < N; i++) {
                    for (j = jj; j < jj + BLOCK_SIZE && j < N; j++) {
                        for (k = kk; k < kk + BLOCK_SIZE && k < N; k++) {
                            matrixC[i][j] += matrixA[i][k] * matrixB[k][j];
                        }
                    }
                }
            }
        }
    }
}

// 改进的分块算法，使用更好的数据局部性
void matrixmultiply_blocked_optimized(int N, int **matrixA, int **matrixB, int **matrixC) {
    int i, j, k, ii, jj, kk;
    
    // 初始化结果矩阵
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            matrixC[i][j] = 0;
        }
    }
    
    // 分块矩阵乘法，改变循环顺序以提高cache命中率
    for (kk = 0; kk < N; kk += BLOCK_SIZE) {
        for (ii = 0; ii < N; ii += BLOCK_SIZE) {
            for (jj = 0; jj < N; jj += BLOCK_SIZE) {
                // 在每个块内进行矩阵乘法
                for (k = kk; k < kk + BLOCK_SIZE && k < N; k++) {
                    for (i = ii; i < ii + BLOCK_SIZE && i < N; i++) {
                        int temp = matrixA[i][k];
                        for (j = jj; j < jj + BLOCK_SIZE && j < N; j++) {
                            matrixC[i][j] += temp * matrixB[k][j];
                        }
                    }
                }
            }
        }
    }
}

// 自适应块大小的版本
void matrixmultiply_blocked_adaptive(int N, int **matrixA, int **matrixB, int **matrixC) {
    int block_size;
    
    // 根据矩阵大小选择合适的块大小
    if (N <= 512) {
        block_size = 32;
    } else if (N <= 1024) {
        block_size = 64;
    } else if (N <= 2048) {
        block_size = 128;
    } else {
        block_size = 256;
    }
    
    printf("Using adaptive block size: %d\n", block_size);
    
    int i, j, k, ii, jj, kk;
    
    // 初始化结果矩阵
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            matrixC[i][j] = 0;
        }
    }
    
    // 分块矩阵乘法
    for (kk = 0; kk < N; kk += block_size) {
        for (ii = 0; ii < N; ii += block_size) {
            for (jj = 0; jj < N; jj += block_size) {
                // 在每个块内进行矩阵乘法
                for (k = kk; k < kk + block_size && k < N; k++) {
                    for (i = ii; i < ii + block_size && i < N; i++) {
                        int temp = matrixA[i][k];
                        for (j = jj; j < jj + block_size && j < N; j++) {
                            matrixC[i][j] += temp * matrixB[k][j];
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
        matrix[i] = (int*)malloc(N * sizeof(int));
    }
    return matrix;
}

void free_matrix(int **matrix, int N) {
    for (int i = 0; i < N; i++) {
        free(matrix[i]);
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
    printf("测试分块优化版本矩阵乘法，矩阵大小: %dx%d\n", N, N);
    
    // 创建矩阵
    int **matrixA = create_matrix(N);
    int **matrixB = create_matrix(N);
    int **matrixC = create_matrix(N);
    
    // 初始化测试数据
    init_test_matrices(N, matrixA, matrixB);
    
    // 测试基本分块版本
    printf("\n测试基本分块版本 (块大小: %d):\n", BLOCK_SIZE);
    clock_t start = clock();
    matrixmultiply_blocked(N, matrixA, matrixB, matrixC);
    clock_t end = clock();
    double time1 = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("基本分块版本执行时间: %.4f 秒\n", time1);
    
    // 重置结果矩阵
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrixC[i][j] = 0;
        }
    }
    
    // 测试优化分块版本
    printf("\n测试优化分块版本:\n");
    start = clock();
    matrixmultiply_blocked_optimized(N, matrixA, matrixB, matrixC);
    end = clock();
    double time2 = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("优化分块版本执行时间: %.4f 秒\n", time2);
    printf("优化提升: %.2fx\n", time1/time2);
    
    // 重置结果矩阵
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrixC[i][j] = 0;
        }
    }
    
    // 测试自适应分块版本
    printf("\n测试自适应分块版本:\n");
    start = clock();
    matrixmultiply_blocked_adaptive(N, matrixA, matrixB, matrixC);
    end = clock();
    double time3 = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("自适应分块版本执行时间: %.4f 秒\n", time3);
    
    // 释放内存
    free_matrix(matrixA, N);
    free_matrix(matrixB, N);
    free_matrix(matrixC, N);
    
    return 0;
}
#endif 