# 快速开始指南

## 1. 环境准备

### Windows 11 系统需要安装：

1. **Python 3.7+**
   ```bash
   # 检查Python版本
   python --version
   ```

2. **MinGW-w64编译器**
   - 下载地址: https://www.mingw-w64.org/downloads/
   - 或使用MSYS2: https://www.msys2.org/
   - 确保gcc在PATH环境变量中

## 2. 项目设置

### 方法一：直接运行
```bash
# 1. 安装Python依赖
pip install numpy pandas matplotlib

# 2. 运行Python测试程序  
python performance_test.py
```

### 方法二：完整编译和测试
```bash
# 1. 安装Python依赖
pip install -r requirements.txt

# 2. 编译所有C语言版本
make all

# 3. 运行完整性能测试
make test
```

## 3. 第一次运行

### 简单测试
```bash
# 测试小矩阵（512x512）
python performance_test.py
# 当提示输入矩阵大小时，输入：512
```

### 快速验证各个版本
```bash
# 单独测试Python版本
python matrix_multiply_python.py

# 如果编译成功，测试C语言版本
make test-basic
make test-multithread
```

## 4. 查看结果

运行完成后会生成：
- `matrix_multiply_results.csv` - 性能数据表格
- `matrix_multiply_performance.png` - 性能对比图
- `performance_report.md` - 详细报告

## 5. 预期结果

### 典型性能提升（相对于基础C版本）：
- Python纯实现：0.01x - 0.1x（非常慢）
- NumPy：0.5x - 10x（优化版）
- 多线程：2x - 10x（取决于CPU核心数）
- 分块优化：1.5x - 8x
- SIMD优化：8x - 20x
- 综合优化：50x - 150x

## 6. 获取帮助

### 命令行帮助
```bash
# 查看Makefile帮助
make help

# 查看Python程序帮助
python performance_test.py --help
```

### 调试模式
```bash
# 编译调试版本
gcc -g -DSTANDALONE_TEST matrix_multiply_basic.c -o debug_basic.exe

# 运行调试版本
./debug_basic.exe
```