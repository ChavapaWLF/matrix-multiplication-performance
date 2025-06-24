#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
矩阵乘法性能测试主程序
统一测试所有优化版本的性能，并进行对比分析
"""

import os
import sys
import time
import ctypes
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import pandas as pd
from ctypes import c_int, c_void_p, POINTER, Structure

# 设置matplotlib字体
import platform
if platform.system() == 'Windows':
    try:
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    except:
        plt.rcParams['font.family'] = 'DejaVu Sans'
else:
    plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class MatrixMultiplyTester:
    def __init__(self, test_size=1024):
        """
        初始化测试器
        
        Args:
            test_size (int): 测试矩阵的大小
        """
        self.test_size = test_size
        self.results = {}
        self.dlls = {}
        
        # 编译标志
        self.compile_flags = {
            'basic': ['-O2'],
            'multithread': ['-O2', '-fopenmp'],
            'blocked': ['-O2', '-march=native'],
            'simd': ['-O2', '-march=native', '-mavx2'],
            'optimized': ['-O2', '-march=native', '-mavx2', '-fopenmp']
        }
        
        print(f"初始化矩阵乘法性能测试器")
        print(f"测试矩阵大小: {test_size}x{test_size}")
        print(f"矩阵内存大小: {test_size * test_size * 4 / 1024 / 1024:.2f} MB")
        
    def compile_c_libraries(self):
        """编译所有C语言库"""
        print("\n开始编译C语言库...")
        
        c_files = {
            'basic': 'matrix_multiply_basic.c',
            'multithread': 'matrix_multiply_multithread.c', 
            'blocked': 'matrix_multiply_blocked.c',
            'simd': 'matrix_multiply_simd.c',
            'optimized': 'matrix_multiply_optimized.c'
        }
        
        for name, c_file in c_files.items():
            if not os.path.exists(c_file):
                print(f"警告: {c_file} 不存在，跳过编译")
                continue
                
            dll_name = f"matrix_{name}.dll"
            compile_cmd = [
                'gcc', '-shared', '-fPIC', c_file, '-o', dll_name
            ] + self.compile_flags.get(name, ['-O2'])
            
            print(f"编译 {name}: {' '.join(compile_cmd)}")
            
            try:
                result = subprocess.run(compile_cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"✓ {name} 编译成功")
                    # 加载DLL
                    self.dlls[name] = ctypes.CDLL(f"./{dll_name}")
                    self._setup_function_signatures(name)
                else:
                    print(f"✗ {name} 编译失败: {result.stderr}")
            except Exception as e:
                print(f"✗ {name} 编译异常: {e}")
    
    def _setup_function_signatures(self, name):
        """设置函数签名"""
        if name not in self.dlls:
            return
            
        dll = self.dlls[name]
        
        # 设置矩阵乘法函数签名
        func_names = {
            'basic': 'matrixmultiply_basic',
            'multithread': 'matrixmultiply_multithread',
            'blocked': 'matrixmultiply_blocked_adaptive',
            'simd': 'matrixmultiply_simd',
            'optimized': 'matrixmultiply_ultimate'
        }
        
        if name in func_names:
            func = getattr(dll, func_names[name])
            func.argtypes = [c_int, POINTER(POINTER(c_int)), 
                           POINTER(POINTER(c_int)), POINTER(POINTER(c_int))]
            func.restype = None
            
        # 设置辅助函数签名
        if hasattr(dll, 'create_matrix'):
            dll.create_matrix.argtypes = [c_int]
            dll.create_matrix.restype = POINTER(POINTER(c_int))
            
        if hasattr(dll, 'free_matrix'):
            dll.free_matrix.argtypes = [POINTER(POINTER(c_int)), c_int]
            dll.free_matrix.restype = None
            
        if hasattr(dll, 'init_test_matrices'):
            dll.init_test_matrices.argtypes = [c_int, POINTER(POINTER(c_int)), 
                                             POINTER(POINTER(c_int))]
            dll.init_test_matrices.restype = None
    
    def create_test_matrices_python(self):
        """创建Python版本的测试矩阵"""
        print("创建Python测试矩阵...")
        N = self.test_size
        
        # 创建测试矩阵
        matrixA = [[i + j for j in range(N)] for i in range(N)]
        matrixB = [[i * j + 1 for j in range(N)] for i in range(N)]
        matrixC = [[0 for j in range(N)] for i in range(N)]
        
        return matrixA, matrixB, matrixC
    
    def create_test_matrices_c(self, dll):
        """创建C语言版本的测试矩阵"""
        N = self.test_size
        
        # 创建矩阵
        matrixA = dll.create_matrix(N)
        matrixB = dll.create_matrix(N)
        matrixC = dll.create_matrix(N)
        
        # 初始化测试数据
        dll.init_test_matrices(N, matrixA, matrixB)
        
        return matrixA, matrixB, matrixC
    
    def test_python_pure(self):
        """测试纯Python版本"""
        print("\n测试纯Python版本...")
        
        # 使用完整的输入矩阵大小
        test_size = self.test_size
        print(f"使用完整 {test_size}x{test_size} 矩阵进行纯Python测试")
        
        matrixA = [[i + j for j in range(test_size)] for i in range(test_size)]
        matrixB = [[i * j + 1 for j in range(test_size)] for i in range(test_size)]
        matrixC = [[0 for j in range(test_size)] for i in range(test_size)]
        
        start_time = time.time()
        
        # 纯Python矩阵乘法
        for i in range(test_size):
            for j in range(test_size):
                matrixC[i][j] = 0
                for k in range(test_size):
                    matrixC[i][j] += matrixA[i][k] * matrixB[k][j]
        
        elapsed_time = time.time() - start_time
        
        self.results['Python_Pure'] = {
            'time': elapsed_time,
            'estimated_time': elapsed_time,
            'matrix_size': test_size,
            'description': '纯Python实现'
        }
        
        print(f"纯Python版本执行时间: {elapsed_time:.4f} 秒 ({test_size}x{test_size})")
    
    def test_python_numpy(self):
        """测试NumPy版本"""
        print("\n测试NumPy版本...")
        
        # 创建NumPy矩阵
        N = self.test_size
        matrixA = np.random.randint(0, 100, (N, N), dtype=np.int32)
        matrixB = np.random.randint(0, 100, (N, N), dtype=np.int32)
        
        start_time = time.time()
        matrixC = np.dot(matrixA, matrixB)
        elapsed_time = time.time() - start_time
        
        self.results['Python_NumPy'] = {
            'time': elapsed_time,
            'estimated_time': elapsed_time,
            'matrix_size': N,
            'description': 'NumPy实现'
        }
        
        print(f"NumPy版本执行时间: {elapsed_time:.4f} 秒")
    
    def test_c_version(self, name, func_name):
        """测试C语言版本"""
        if name not in self.dlls:
            print(f"跳过 {name}：DLL未加载")
            return
            
        print(f"\n测试 {name} 版本...")
        
        dll = self.dlls[name]
        N = self.test_size
        
        # 创建矩阵
        matrixA, matrixB, matrixC = self.create_test_matrices_c(dll)
        
        # 获取函数
        func = getattr(dll, func_name)
        
        # 执行测试
        start_time = time.time()
        func(N, matrixA, matrixB, matrixC)
        elapsed_time = time.time() - start_time
        
        self.results[name] = {
            'time': elapsed_time,
            'estimated_time': elapsed_time,
            'matrix_size': N,
            'description': f'C语言{name}实现'
        }
        
        print(f"{name} 版本执行时间: {elapsed_time:.4f} 秒")
        
        # 释放内存
        dll.free_matrix(matrixA, N)
        dll.free_matrix(matrixB, N)
        dll.free_matrix(matrixC, N)
    
    def run_all_tests(self):
        """运行所有测试"""
        print("=" * 60)
        print("开始矩阵乘法性能测试")
        print("=" * 60)
        
        # 编译C库
        self.compile_c_libraries()
        
        # 测试Python版本
        self.test_python_pure()
        self.test_python_numpy()
        
        # 测试C语言版本
        c_tests = {
            'basic': 'matrixmultiply_basic',
            'multithread': 'matrixmultiply_multithread',
            'blocked': 'matrixmultiply_blocked_adaptive',
            'simd': 'matrixmultiply_simd',
            'optimized': 'matrixmultiply_ultimate'
        }
        
        for name, func_name in c_tests.items():
            try:
                self.test_c_version(name, func_name)
            except Exception as e:
                print(f"测试 {name} 时出错: {e}")
    
    def analyze_results(self):
        """分析测试结果"""
        print("\n" + "=" * 60)
        print("性能测试结果分析")
        print("=" * 60)
        
        if not self.results:
            print("没有测试结果可分析")
            return
        
        # 创建results目录
        os.makedirs('results', exist_ok=True)
        
        # 创建结果表格
        df_data = []
        baseline_time = None
        
        for name, result in self.results.items():
            time_to_use = result.get('estimated_time', result['time'])
            
            # 设定基准（选择基础C版本或纯Python版本）
            if baseline_time is None:
                if 'basic' in self.results:
                    baseline_time = self.results['basic']['time']
                else:
                    baseline_time = time_to_use
            
            speedup = baseline_time / time_to_use if time_to_use > 0 else 0
            
            df_data.append({
                '实现版本': name,
                '描述': result['description'],
                '执行时间(秒)': f"{time_to_use:.4f}",
                '加速倍数': f"{speedup:.2f}x",
                '矩阵大小': f"{result['matrix_size']}x{result['matrix_size']}"
            })
        
        df = pd.DataFrame(df_data)
        print(df.to_string(index=False))
        
        # 保存结果到CSV
        csv_path = os.path.join('results', 'matrix_multiply_results.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存到 {csv_path}")
        
        return df
    
    def plot_results(self):
        """绘制性能对比图"""
        if not self.results:
            return
            
        print("\n生成性能对比图...")
        
        # 准备数据
        names = []
        times = []
        speedups = []
        
        baseline_time = None
        if 'basic' in self.results:
            baseline_time = self.results['basic']['time']
        elif 'Python_Pure' in self.results:
            baseline_time = self.results['Python_Pure']['estimated_time']
        else:
            baseline_time = min(r.get('estimated_time', r['time']) for r in self.results.values())
        
        for name, result in self.results.items():
            time_to_use = result.get('estimated_time', result['time'])
            speedup = baseline_time / time_to_use if time_to_use > 0 else 0
            
            names.append(name.replace('_', '\n'))
            times.append(time_to_use)
            speedups.append(speedup)
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 执行时间对比
        bars1 = ax1.bar(names, times, color='skyblue', alpha=0.7)
        ax1.set_title('Matrix Multiplication Execution Time Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Execution Time (seconds)', fontsize=12)
        ax1.set_yscale('log')  # 使用对数坐标
        
        # 在柱状图上添加数值标签
        for bar, time_val in zip(bars1, times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time_val:.3f}s', ha='center', va='bottom', fontsize=10)
        
        # 加速倍数对比
        bars2 = ax2.bar(names, speedups, color='lightcoral', alpha=0.7)
        ax2.set_title('Matrix Multiplication Speedup Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Speedup Factor', fontsize=12)
        
        # 在柱状图上添加数值标签
        for bar, speedup in zip(bars2, speedups):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{speedup:.2f}x', ha='center', va='bottom', fontsize=10)
        
        # 调整x轴标签
        for ax in [ax1, ax2]:
            ax.tick_params(axis='x', rotation=45, labelsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 确保results目录存在
        os.makedirs('results', exist_ok=True)
        
        # 保存图表到results目录
        png_path = os.path.join('results', 'matrix_multiply_performance.png')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"性能对比图已保存到 {png_path}")
        
        # 关闭图表以避免程序卡住
        plt.close('all')
    
    def generate_report(self):
        """生成详细的性能报告"""
        print("\n生成详细性能报告...")
        
        report = []
        report.append("# 矩阵乘法性能测试报告\n")
        report.append(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"测试矩阵大小: {self.test_size}x{self.test_size}\n")
        report.append(f"系统平台: {sys.platform}\n\n")
        
        report.append("## 测试结果汇总\n")
        
        if self.results:
            baseline_time = None
            if 'basic' in self.results:
                baseline_time = self.results['basic']['time']
            elif self.results:
                baseline_time = min(r.get('estimated_time', r['time']) for r in self.results.values())
            
            for name, result in self.results.items():
                time_to_use = result.get('estimated_time', result['time'])
                speedup = baseline_time / time_to_use if time_to_use > 0 and baseline_time else 1.0
                
                report.append(f"### {name}\n")
                report.append(f"- 描述: {result['description']}\n")
                report.append(f"- 执行时间: {time_to_use:.4f} 秒\n")
                report.append(f"- 加速倍数: {speedup:.2f}x\n")
                report.append(f"- 矩阵大小: {result['matrix_size']}x{result['matrix_size']}\n\n")
        
        report.append("## 优化技术说明\n")
        report.append("1. **基础C语言版本**: 简单的三重循环实现\n")
        report.append("2. **多线程版本**: 使用Windows线程并行化计算\n")
        report.append("3. **分块优化版本**: 提高cache局部性，减少cache miss\n")
        report.append("4. **SIMD优化版本**: 使用AVX2/SSE指令集并行计算\n")
        report.append("5. **综合优化版本**: 结合多线程、分块、SIMD等多种优化技术\n")
        report.append("6. **NumPy版本**: 使用高度优化的BLAS库\n\n")
        
        report.append("## 结论\n")
        report.append("通过本次测试可以看出，不同优化技术对矩阵乘法性能的提升效果。\n")
        report.append("SIMD指令集和多线程并行化是最有效的优化手段。\n")
        
        # 确保results目录存在
        os.makedirs('results', exist_ok=True)
        
        # 保存报告到results目录
        report_path = os.path.join('results', 'performance_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.writelines(report)
        
        print(f"详细报告已保存到 {report_path}")

def main():
    """主函数"""
    print("矩阵乘法性能测试程序")
    print("Author: ChavapaWLF")
    
    # 根据系统性能调整测试大小
    test_sizes = [512, 1024, 2048]
    print(f"\n可选测试大小: {test_sizes}")
    
    try:
        size_input = input("请输入测试矩阵大小 (默认1024): ").strip()
        test_size = int(size_input) if size_input else 1024
    except ValueError:
        test_size = 1024
    
    print(f"使用测试大小: {test_size}")
    
    # 创建测试器并运行测试
    tester = MatrixMultiplyTester(test_size)
    
    try:
        # 运行所有测试
        tester.run_all_tests()
        
        # 分析结果
        tester.analyze_results()
        
        # 绘制图表
        tester.plot_results()
        
        # 生成报告
        tester.generate_report()
        
        print("\n" + "=" * 60)
        print("所有测试完成！请查看生成的文件：")
        print("- results/matrix_multiply_results.csv: 测试结果表格")
        print("- results/matrix_multiply_performance.png: 性能对比图")
        print("- results/performance_report.md: 详细测试报告")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n用户中断测试")
    except Exception as e:
        print(f"\n测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 