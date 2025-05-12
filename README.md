# C++ Mathematics Library / C++数学库

A comprehensive C++ mathematics library that provides various statistical functions, probability distributions, and combinatorial mathematics functions.

一个全面的C++数学库，提供各种统计函数、概率分布和组合数学函数。

## Features / 功能特点

### Statistical Functions / 统计函数
- Basic Statistics / 基础统计
  - Mean, Variance, Standard Deviation / 均值、方差、标准差
  - Median, Mode, Quantiles / 中位数、众数、分位数
  - Covariance and Correlation / 协方差和相关系数
- Advanced Statistics / 高级统计
  - Skewness and Kurtosis / 偏度和峰度
  - Hypothesis Testing (t-test, F-test) / 假设检验（t检验、F检验）
  - Confidence Intervals / 置信区间
- Non-parametric Tests / 非参数检验
  - Wilcoxon Rank-Sum Test / Wilcoxon秩和检验
  - Kruskal-Wallis Test / Kruskal-Wallis检验
  - Friedman Test / 弗里德曼检验
- Correlation Analysis / 相关性分析
  - Pearson Correlation / 皮尔逊相关系数
  - Spearman Rank Correlation / 斯皮尔曼等级相关系数
  - Kendall Rank Correlation / 肯德尔等级相关系数

### Linear Algebra / 线性代数
- Matrix Operations / 矩阵运算
  - Basic Operations (Addition, Subtraction, Multiplication) / 基础运算（加法、减法、乘法）
  - Matrix Transpose and Inverse / 矩阵转置和逆
  - Determinant and Trace / 行列式和迹
  - Eigenvalues and Eigenvectors / 特征值和特征向量
- Vector Operations / 向量运算
  - Dot Product and Cross Product / 点积和叉积
  - Vector Norms / 向量范数
  - Vector Projection / 向量投影
- Decomposition Methods / 分解方法
  - LU Decomposition / LU分解
  - QR Decomposition / QR分解
  - Singular Value Decomposition (SVD) / 奇异值分解
  - Cholesky Decomposition / 乔列斯基分解
- Linear Systems / 线性系统
  - Solving Linear Equations / 求解线性方程组
  - Least Squares Solutions / 最小二乘解
  - Matrix Condition Number / 矩阵条件数

### Probability Distributions / 概率分布
- Continuous Distributions / 连续分布
  - Normal Distribution / 正态分布
  - Exponential Distribution / 指数分布
  - Chi-Squared Distribution / 卡方分布
  - t-Distribution / t分布
  - F-Distribution / F分布
  - Beta Distribution / Beta分布
  - Gamma Distribution / Gamma分布
  - Weibull Distribution / 威布尔分布
  - Log-Normal Distribution / 对数正态分布
- Discrete Distributions / 离散分布
  - Binomial Distribution / 二项分布
  - Poisson Distribution / 泊松分布

### Combinatorial Mathematics / 组合数学
- Basic Combinatorics / 基础组合
  - Factorial, Permutations, Combinations / 阶乘、排列、组合
  - Stirling Numbers / 斯特林数
  - Catalan Numbers / 卡特兰数
  - Bell Numbers / 贝尔数
- Special Functions / 特殊函数
  - Euler and Bernoulli Numbers / 欧拉数和伯努利数
  - Harmonic Numbers / 调和数
  - Polylogarithm / 多对数函数
- Orthogonal Polynomials / 正交多项式
  - Euler Polynomials / 欧拉多项式
  - Bernoulli Polynomials / 伯努利多项式
  - Laguerre Polynomials / 拉盖尔多项式
  - Chebyshev Polynomials / 切比雪夫多项式
  - Hermite Polynomials / 埃尔米特多项式
  - Jacobi Polynomials / 雅可比多项式
  - Hypergeometric Function / 超几何函数

## Requirements / 要求
- C++23 compatible compiler / C++23兼容的编译器
- CMake 3.10 or higher / CMake 3.10或更高版本

## Building / 构建
```bash
mkdir build
cd build
cmake ..
make
```

## Usage / 使用
```cpp
#include "mathlib.hpp"
#include "mathlib/linear_algebra.hpp"

// 统计函数示例 / Statistics example
std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
double mean = mathlib::statistics::mean(data);
double variance = mathlib::statistics::variance(data);

// 概率分布示例 / Probability distribution example
mathlib::probability::NormalDistribution normal(0.0, 1.0);
double pdf = normal.pdf(0.0);
double cdf = normal.cdf(1.0);

// 组合数学示例 / Combinatorics example
unsigned long long comb = mathlib::combinatorics::combination(5, 2);
double euler_poly = mathlib::combinatorics::euler_polynomial(3, 0.5);

// 矩阵运算示例 / Matrix operations example
mathlib::linear_algebra::Matrix<double> A({{1, 2}, {3, 4}});
mathlib::linear_algebra::Matrix<double> B({{5, 6}, {7, 8}});

// 矩阵加法和乘法 / Matrix addition and multiplication
auto C = A + B;
auto D = A * B;

// 矩阵转置 / Matrix transpose
auto A_transpose = A.transpose();

// 向量运算示例 / Vector operations example
mathlib::linear_algebra::Vector<double> v1({1, 2, 3});
mathlib::linear_algebra::Vector<double> v2({4, 5, 6});

// 点积和范数 / Dot product and norm
double dot_product = v1.dot(v2);
double norm = v1.norm();

// 线性方程组求解示例 / Linear system solving example
mathlib::linear_algebra::Matrix<double> A({{2, 1}, {1, 3}});
mathlib::linear_algebra::Vector<double> b({5, 7});
auto x = mathlib::linear_algebra::solve_linear_system(A, b);
```

## Testing / 测试
```bash
cd build
make test
```

The test suite includes comprehensive tests for:
- Matrix operations and error handling
- Vector operations
- Linear system solving
- Edge cases and invalid inputs

测试套件包括：
- 矩阵运算和错误处理
- 向量运算
- 线性方程组求解
- 边界情况和无效输入

## Examples / 示例
The library includes example programs demonstrating various features:
- `statistics_example.cpp`: Statistical functions and tests
- `probability_example.cpp`: Probability distributions
- `combinatorics_example.cpp`: Combinatorial functions and polynomials
- `linear_algebra_example.cpp`: Matrix and vector operations

库包含演示各种功能的示例程序：
- `statistics_example.cpp`：统计函数和检验
- `probability_example.cpp`：概率分布
- `combinatorics_example.cpp`：组合函数和多项式
- `linear_algebra_example.cpp`：矩阵和向量运算

## License / 许可证
MIT License / MIT许可证 