import numpy as np
import time
from ctypes import *

def matrixmultiply_python_pure(N, matrixA, matrixB, matrixC):
    """
    纯Python实现的矩阵乘法
    """
    for i in range(N):
        for j in range(N):
            matrixC[i][j] = 0
            for k in range(N):
                matrixC[i][j] += matrixA[i][k] * matrixB[k][j]

def matrixmultiply_python_numpy(N, matrixA, matrixB, matrixC):
    """
    使用NumPy实现的矩阵乘法
    """
    # 将Python列表转换为NumPy数组
    A = np.array(matrixA)
    B = np.array(matrixB)
    
    # 执行矩阵乘法
    C = np.dot(A, B)
    
    # 将结果复制回原数组
    for i in range(N):
        for j in range(N):
            matrixC[i][j] = int(C[i][j])

def create_test_matrices(N):
    """
    创建测试矩阵
    """
    matrixA = [[i + j for j in range(N)] for i in range(N)]
    matrixB = [[i * j + 1 for j in range(N)] for i in range(N)]
    matrixC = [[0 for j in range(N)] for i in range(N)]
    return matrixA, matrixB, matrixC

if __name__ == "__main__":
    N = 128  # 用较小的矩阵测试，因为纯Python很慢
    print(f"测试矩阵大小: {N}x{N}")
    
    # 创建测试矩阵
    matrixA, matrixB, matrixC = create_test_matrices(N)
    
    # 测试纯Python版本
    print("测试纯Python版本...")
    start_time = time.time()
    matrixmultiply_python_pure(N, matrixA, matrixB, matrixC)
    python_time = time.time() - start_time
    print(f"纯Python版本执行时间: {python_time:.4f} 秒")
    
    # 重置结果矩阵
    matrixC = [[0 for j in range(N)] for i in range(N)]
    
    # 测试NumPy版本
    print("测试NumPy版本...")
    start_time = time.time()
    matrixmultiply_python_numpy(N, matrixA, matrixB, matrixC)
    numpy_time = time.time() - start_time
    print(f"NumPy版本执行时间: {numpy_time:.4f} 秒")
    print(f"NumPy加速倍数: {python_time/numpy_time:.2f}x") 