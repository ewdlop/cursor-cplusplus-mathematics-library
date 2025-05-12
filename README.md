# C++ Mathematics Library / C++数学库

A comprehensive C++ mathematics library that provides statistical functions, probability distributions, and combinatorial mathematics functions.

一个全面的C++数学库，提供统计函数、概率分布和组合数学函数。

## Features / 功能特性

### Statistical Functions / 统计函数
- Basic statistics (mean, variance, standard deviation) / 基础统计（均值、方差、标准差）
- Advanced statistics (skewness, kurtosis) / 高级统计（偏度、峰度）
- Hypothesis testing (t-test, F-test) / 假设检验（t检验、F检验）
- Non-parametric tests (Wilcoxon rank-sum test, Kruskal-Wallis test) / 非参数检验（Wilcoxon秩和检验、Kruskal-Wallis检验）
- Confidence intervals and p-values / 置信区间和p值

### Probability Distributions / 概率分布
- Continuous distributions / 连续分布
  - Normal distribution / 正态分布
  - Exponential distribution / 指数分布
  - Chi-squared distribution / 卡方分布
  - t-distribution / t分布
  - F-distribution / F分布
  - Beta distribution / Beta分布
  - Gamma distribution / Gamma分布
- Discrete distributions / 离散分布
  - Binomial distribution / 二项分布
  - Poisson distribution / 泊松分布

### Combinatorial Mathematics / 组合数学
- Basic combinatorics / 基础组合
  - Factorial / 阶乘
  - Combinations / 组合数
  - Permutations / 排列数
- Special numbers / 特殊数
  - Stirling numbers / 斯特林数
  - Catalan numbers / 卡特兰数
  - Bell numbers / 贝尔数
  - Euler numbers / 欧拉数
  - Bernoulli numbers / 伯努利数
- Special functions / 特殊函数
  - Euler-Mascheroni constant / 欧拉-马歇罗尼常数
  - Harmonic numbers / 调和数
  - Polylogarithm / 多重对数函数
- Polynomials / 多项式
  - Euler polynomials / 欧拉多项式
  - Bernoulli polynomials / 伯努利多项式
  - Laguerre polynomials / 拉盖尔多项式
  - Chebyshev polynomials / 切比雪夫多项式

## Requirements / 要求
- C++23 compatible compiler / C++23兼容的编译器
- CMake 3.10 or higher / CMake 3.10或更高版本
- Google Test (for testing) / Google Test（用于测试）

## Building / 构建
```bash
mkdir build
cd build
cmake ..
make
```

## Usage / 使用示例

### Statistical Functions / 统计函数
```cpp
#include "mathlib.hpp"

std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
double m = mathlib::statistics::mean(data);
double v = mathlib::statistics::variance(data);
double s = mathlib::statistics::skewness(data);
```

### Probability Distributions / 概率分布
```cpp
#include "mathlib.hpp"

mathlib::probability::NormalDistribution normal(0.0, 1.0);
double pdf = normal.pdf(0.0);
double cdf = normal.cdf(1.0);
double sample = normal.sample();
```

### Combinatorial Mathematics / 组合数学
```cpp
#include "mathlib.hpp"

unsigned long long c = mathlib::combinatorics::combination(10, 3);
double euler = mathlib::combinatorics::euler_polynomial(3, 0.5);
double chebyshev = mathlib::combinatorics::chebyshev_polynomial(2, 0.5, true);
```

## Testing / 测试
```bash
cd build
make test
```

## License / 许可证
MIT License / MIT许可证 