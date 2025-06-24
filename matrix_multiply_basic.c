#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void matrixmultiply_basic(int N, int **matrixA, int **matrixB, int **matrixC) {
    int i, j, k;
    
    // 初始化结果矩阵
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            matrixC[i][j] = 0;
        }
    }
    
    // 基本的三重循环矩阵乘法
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (k = 0; k < N; k++) {
                matrixC[i][j] += matrixA[i][k] * matrixB[k][j];
            }
        }
    }
}

// 创建矩阵的辅助函数
int** create_matrix(int N) {
    int **matrix = (int**)malloc(N * sizeof(int*));
    for (int i = 0; i < N; i++) {
        matrix[i] = (int*)malloc(N * sizeof(int));
    }
    return matrix;
}

// 释放矩阵的辅助函数
void free_matrix(int **matrix, int N) {
    for (int i = 0; i < N; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

// 初始化测试矩阵
void init_test_matrices(int N, int **matrixA, int **matrixB) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrixA[i][j] = i + j;
            matrixB[i][j] = i * j + 1;
        }
    }
}

// 测试矩阵正确性的函数
int verify_result(int N, int **matrixC, int **reference) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (matrixC[i][j] != reference[i][j]) {
                return 0; // 不匹配
            }
        }
    }
    return 1; // 匹配
}

#ifdef STANDALONE_TEST
int main() {
    int N = 512; // 测试矩阵大小
    printf("测试基础C语言版本矩阵乘法，矩阵大小: %dx%d\n", N, N);
    
    // 创建矩阵
    int **matrixA = create_matrix(N);
    int **matrixB = create_matrix(N);
    int **matrixC = create_matrix(N);
    
    // 初始化测试数据
    init_test_matrices(N, matrixA, matrixB);
    
    // 记录开始时间
    clock_t start = clock();
    
    // 执行矩阵乘法
    matrixmultiply_basic(N, matrixA, matrixB, matrixC);
    
    // 记录结束时间
    clock_t end = clock();
    double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    printf("基础C版本执行时间: %.4f 秒\n", cpu_time_used);
    
    // 释放内存
    free_matrix(matrixA, N);
    free_matrix(matrixB, N);
    free_matrix(matrixC, N);
    
    return 0;
}
#endif 