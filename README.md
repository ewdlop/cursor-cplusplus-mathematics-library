# C++ 数学库 / C++ Mathematics Library

## 简介 | Introduction

本库为 C++ 提供了丰富的数学功能，包括统计、概率、组合数学和线性代数。现已将原有的 `mathlib.hpp` 拆分为多个子模块，便于按需引用和维护。

This library provides rich mathematical functionalities for C++, including statistics, probability, combinatorics, and linear algebra. The original `mathlib.hpp` has been split into several submodules for easier and more flexible usage.

---

## 目录结构 | Directory Structure

```
include/
  mathlib/
    statistics.hpp      # 统计相关
    probability.hpp     # 概率相关
    combinatorics.hpp   # 组合数学相关
    linear_algebra.hpp  # 线性代数相关
```

---

## 如何使用 | How to Use

**请注意：自 v2.0 起，`mathlib.hpp` 已被移除。请直接包含所需的子模块头文件。**

**Note:** Since v2.0, `mathlib.hpp` has been removed. Please include the required submodule headers directly.

### 示例 | Example

```cpp
#include "mathlib/statistics.hpp"
#include "mathlib/probability.hpp"
#include "mathlib/combinatorics.hpp"
#include "mathlib/linear_algebra.hpp"

#include <vector>
#include <iostream>

int main() {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::cout << "Mean: " << mathlib::statistics::mean(data) << std::endl;
    std::cout << "Variance: " << mathlib::statistics::variance(data) << std::endl;
    std::cout << "Normal PDF(0): " << mathlib::probability::normal_pdf(0.0) << std::endl;
    std::cout << "5! = " << mathlib::combinatorics::factorial<int>(5) << std::endl;
    mathlib::linear_algebra::Matrix<double> A({{1,2},{3,4}});
    std::cout << "A(0,0): " << A(0,0) << std::endl;
    return 0;
}
```

---

## 主要功能 | Main Features

- **统计 (Statistics):**
  - 均值、方差、标准差、中位数、众数、偏度、峰度、相关性、协方差、分位数、假设检验、置信区间等
- **概率 (Probability):**
  - 常见分布（正态、二项、泊松）、概率密度/分布函数、熵、互信息、马尔可夫链、随机数生成等
- **组合数学 (Combinatorics):**
  - 阶乘、组合数、排列数、多项式、生成函数、斯特林数、贝尔数等
- **线性代数 (Linear Algebra):**
  - 矩阵与向量运算、行列式、迹、LU分解、特征值/特征向量、线性方程组、最小二乘、条件数等

---

## 兼容性与迁移 | Compatibility & Migration

- **如您的代码原本包含 `#include "mathlib.hpp"`，请替换为需要的子模块头文件。**
- **If your code previously included `#include "mathlib.hpp"`, please replace it with the required submodule headers.**

---

## 构建与测试 | Build & Test

请确保您的 include 路径包含 `include/` 目录，并根据需要链接 Google Test 等测试框架。

Make sure your include path contains the `include/` directory, and link with Google Test or other frameworks as needed.

---

## 反馈与贡献 | Feedback & Contribution

欢迎提交 issue 或 pull request，帮助本库完善！

Feel free to submit issues or pull requests to help improve this library!

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