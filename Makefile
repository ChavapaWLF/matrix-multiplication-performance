# Makefile for Matrix Multiplication Optimization Project
# Windows环境下使用MinGW-w64编译

CC = gcc
CFLAGS = -O2 -Wall -std=c99
LIBS = 

# 源文件
SOURCES = matrix_multiply_basic.c matrix_multiply_multithread.c matrix_multiply_blocked.c matrix_multiply_simd.c matrix_multiply_optimized.c

# 目标文件
TARGETS = matrix_basic.dll matrix_multithread.dll matrix_blocked.dll matrix_simd.dll matrix_optimized.dll

# 测试可执行文件
TEST_TARGETS = test_basic.exe test_multithread.exe test_blocked.exe test_simd.exe test_optimized.exe

.PHONY: all clean test help dlls tests

# 默认目标
all: dlls

# 编译所有动态链接库
dlls: $(TARGETS)

# 编译所有测试程序
tests: $(TEST_TARGETS)

# 基础版本
matrix_basic.dll: matrix_multiply_basic.c
	$(CC) -shared -fPIC $(CFLAGS) $< -o $@

test_basic.exe: matrix_multiply_basic.c
	$(CC) $(CFLAGS) -DSTANDALONE_TEST $< -o $@

# 多线程版本
matrix_multithread.dll: matrix_multiply_multithread.c
	$(CC) -shared -fPIC $(CFLAGS) -fopenmp $< -o $@

test_multithread.exe: matrix_multiply_multithread.c
	$(CC) $(CFLAGS) -DSTANDALONE_TEST -fopenmp $< -o $@

# 分块优化版本
matrix_blocked.dll: matrix_multiply_blocked.c
	$(CC) -shared -fPIC $(CFLAGS) -march=native $< -o $@

test_blocked.exe: matrix_multiply_blocked.c
	$(CC) $(CFLAGS) -DSTANDALONE_TEST -march=native $< -o $@

# SIMD优化版本
matrix_simd.dll: matrix_multiply_simd.c
	$(CC) -shared -fPIC $(CFLAGS) -march=native -mavx2 $< -o $@

test_simd.exe: matrix_multiply_simd.c
	$(CC) $(CFLAGS) -DSTANDALONE_TEST -march=native -mavx2 $< -o $@

# 综合优化版本
matrix_optimized.dll: matrix_multiply_optimized.c
	$(CC) -shared -fPIC $(CFLAGS) -march=native -mavx2 -fopenmp $< -o $@

test_optimized.exe: matrix_multiply_optimized.c
	$(CC) $(CFLAGS) -DSTANDALONE_TEST -march=native -mavx2 -fopenmp $< -o $@

# 运行性能测试
test: dlls
	python performance_test.py

# 单独测试各个版本
test-basic: test_basic.exe
	./test_basic.exe

test-multithread: test_multithread.exe
	./test_multithread.exe

test-blocked: test_blocked.exe
	./test_blocked.exe

test-simd: test_simd.exe
	./test_simd.exe

test-optimized: test_optimized.exe
	./test_optimized.exe



# 清理编译产生的文件
clean:
	del /Q *.dll *.exe *.o 2>nul || true
	del /Q matrix_multiply_results.csv 2>nul || true
	del /Q matrix_multiply_performance.png 2>nul || true
	del /Q performance_report.md 2>nul || true



# 帮助信息
help:
	@echo "可用的make目标："
	@echo "  all (默认)    - 编译所有动态链接库"
	@echo "  dlls          - 编译所有动态链接库"
	@echo "  tests         - 编译所有测试程序"
	@echo "  test          - 运行Python性能测试"
	@echo "  test-basic    - 运行基础版本测试"
	@echo "  test-multithread - 运行多线程版本测试"
	@echo "  test-blocked  - 运行分块优化版本测试"
	@echo "  test-simd     - 运行SIMD优化版本测试"
	@echo "  test-optimized - 运行综合优化版本测试"
	@echo "  clean         - 清理编译产生的文件"
	@echo "  help          - 显示此帮助信息"
	@echo ""
	@echo "编译要求："
	@echo "  - MinGW-w64 (gcc)"
	@echo "  - 支持AVX2指令集的CPU（用于SIMD版本）"
	@echo "  - OpenMP支持（用于多线程版本）"

# 检查编译环境
check-env:
	@echo "检查编译环境..."
	@gcc --version || echo "错误: 未找到gcc编译器"
	@echo "CPU信息:"
	@wmic cpu get name /format:list | findstr Name=
	@echo "检查AVX2支持:"
	@gcc -march=native -dM -E - < nul | findstr AVX2 || echo "警告: CPU可能不支持AVX2"

.SECONDARY: $(SOURCES) 