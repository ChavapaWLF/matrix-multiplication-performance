# 矩阵乘法优化性能测试项目

## 项目简介

这是一个《汇编语言》课程的大作业项目，通过实现多种优化技术来比较不同矩阵乘法算法的性能。项目包含了从基础的Python实现到高度优化的C语言版本，展示了各种优化技术对性能的影响。

## 项目结构

```
matrixmultiply/
├── matrix_multiply_python.py      # Python版本实现
├── matrix_multiply_basic.c        # 基础C语言版本
├── matrix_multiply_multithread.c  # 多线程优化版本
├── matrix_multiply_blocked.c      # 分块优化版本
├── matrix_multiply_simd.c         # SIMD向量指令优化版本
├── matrix_multiply_optimized.c    # 综合优化版本
├── performance_test.py            # 统一性能测试主程序
├── Makefile                       # 编译脚本
├── requirements.txt               # Python依赖项
└── README.md                      # 项目说明
```

## 实现的优化技术

### 1. Python版本
- **纯Python实现**: 基础的三重循环矩阵乘法
- **NumPy版本**: 使用高度优化的BLAS库

### 2. C语言基础版本
- 简单的三重循环实现
- 作为其他优化版本的性能基准

### 3. 多线程优化版本
- 使用Windows线程API进行并行化
- 自动检测CPU核心数并分配工作负载
- 支持多核CPU的并行计算

### 4. 分块优化版本 (Cache优化)
- 矩阵分块技术提高Cache局部性
- 自适应块大小选择
- 优化的循环顺序减少Cache miss

### 5. SIMD向量指令优化版本
- 支持SSE和AVX2指令集
- 同时处理多个数据元素
- 自动检测CPU指令集支持

### 6. 综合优化版本
- 结合多线程、分块、SIMD、预取等多种优化技术
- 针对现代CPU架构进行深度优化
- 代表了当前最佳的优化水平

## 环境要求

### 操作系统
- Windows 11 (项目针对Windows环境开发)
- 其他系统可能需要修改部分代码

### 编译工具
- **MinGW-w64**: GCC编译器套件
- **支持的指令集**: AVX2 (用于SIMD优化)
- **OpenMP**: 多线程支持

### Python环境
- Python 3.7 或更高版本
- 安装requirements.txt中的依赖项

## 安装和使用

### 1. 克隆项目
```bash
git clone https://github.com/ChavapaWLF/matrix-multiplication-performance.git
cd matrixmultiply
```

### 2. 安装Python依赖
```bash
pip install -r requirements.txt
```

### 3. 编译C语言库
```bash
# 使用Makefile编译所有版本
make all

# 或者手动编译单个版本
gcc -shared -fPIC -O2 matrix_multiply_basic.c -o matrix_basic.dll
```

### 4. 运行性能测试
```bash
# 运行完整的性能测试对比
python performance_test.py

# 或使用Makefile
make test
```

### 5. 单独测试各个版本
```bash
# 编译并运行单个版本的测试
make test-basic
make test-multithread
make test-blocked
make test-simd
make test-optimized
```

## 测试矩阵大小

默认测试矩阵大小为1024x1024，实际可以根据系统性能调整：

- **512x512**: 适合较低性能的系统
- **1024x1024**: 默认测试大小
- **2048x2048**: 适合高性能系统
- **4096x4096**: 大作业要求的大小

## 输出结果

程序会生成以下文件：

1. **matrix_multiply_results.csv**: 详细的测试结果表格
2. **matrix_multiply_performance.png**: 性能对比图表
3. **performance_report.md**: 详细的测试报告

## 性能优化技术详解

### 分块算法 (Blocking)
通过将大矩阵分解为小块来提高Cache命中率，减少内存访问延迟。

### SIMD向量指令
利用现代CPU的向量处理单元，同时处理多个数据元素：
- SSE: 128位向量，同时处理4个32位整数
- AVX2: 256位向量，同时处理8个32位整数

### 多线程并行化
将矩阵计算任务分配到多个CPU核心，充分利用多核处理器的计算能力。

### 内存预取 (Prefetching)
提前将数据加载到Cache中，减少CPU等待内存的时间。

## 许可证

MIT

## 作者

UCAS-ChavapaWLF