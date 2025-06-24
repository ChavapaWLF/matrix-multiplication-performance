#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <windows.h>
#include <process.h>

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

// 线程函数
unsigned __stdcall matrix_multiply_thread(void* arg) {
    ThreadParams* params = (ThreadParams*)arg;
    int N = params->N;
    int **matrixA = params->matrixA;
    int **matrixB = params->matrixB;
    int **matrixC = params->matrixC;
    int start_row = params->start_row;
    int end_row = params->end_row;
    
    // 计算分配给该线程的行
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < N; j++) {
            matrixC[i][j] = 0;
            for (int k = 0; k < N; k++) {
                matrixC[i][j] += matrixA[i][k] * matrixB[k][j];
            }
        }
    }
    
    return 0;
}

void matrixmultiply_multithread(int N, int **matrixA, int **matrixB, int **matrixC) {
    // 获取系统CPU核心数
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    int num_threads = sysinfo.dwNumberOfProcessors;
    
    // 限制最大线程数
    if (num_threads > 16) num_threads = 16;
    
    printf("Using %d threads for computation\n", num_threads);
    
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
        
        threads[t] = (HANDLE)_beginthreadex(NULL, 0, matrix_multiply_thread, 
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

// 辅助函数（重用之前的）
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
    printf("测试多线程版本矩阵乘法，矩阵大小: %dx%d\n", N, N);
    
    // 创建矩阵
    int **matrixA = create_matrix(N);
    int **matrixB = create_matrix(N);
    int **matrixC = create_matrix(N);
    
    // 初始化测试数据
    init_test_matrices(N, matrixA, matrixB);
    
    // 记录开始时间
    clock_t start = clock();
    
    // 执行多线程矩阵乘法
    matrixmultiply_multithread(N, matrixA, matrixB, matrixC);
    
    // 记录结束时间
    clock_t end = clock();
    double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    printf("多线程版本执行时间: %.4f 秒\n", cpu_time_used);
    
    // 释放内存
    free_matrix(matrixA, N);
    free_matrix(matrixB, N);
    free_matrix(matrixC, N);
    
    return 0;
}
#endif 