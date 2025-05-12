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
```

## Testing / 测试
```bash
cd build
make test
```

## Examples / 示例
The library includes example programs demonstrating various features:
- `statistics_example.cpp`: Statistical functions and tests
- `probability_example.cpp`: Probability distributions
- `combinatorics_example.cpp`: Combinatorial functions and polynomials

库包含演示各种功能的示例程序：
- `statistics_example.cpp`：统计函数和检验
- `probability_example.cpp`：概率分布
- `combinatorics_example.cpp`：组合函数和多项式

## License / 许可证
MIT License / MIT许可证 