# C++ Mathematics Library / C++数学库

A C++ mathematics library providing rich mathematical computing capabilities.
这是一个功的C++数学库，提供了丰富的数学计算功能。

## Directory Structure / 目录结构

```
include/
  ├── mathlib/
  │   ├── statistics.hpp    # Statistical functions / 统计函数
  │   ├── probability.hpp   # Probability functions / 概率函数
  │   ├── combinatorics.hpp # Combinatorics / 组合数学
  │   ├── numerical.hpp     # Numerical analysis / 数值分析
  │   ├── optimization.hpp  # Optimization algorithms / 优化算法
  │   └── signal.hpp        # Signal processing / 信号处理
  └── mathlib.hpp          # Main header file / 主头文件
tests/                     # Test files / 测试文件
examples/                  # Example code / 示例代码
```

## Usage / 使用方法

Include the required submodule headers directly:
直接包含需要的子模块头文件：

```cpp
#include "mathlib/statistics.hpp"    // Statistical functions / 统计函数
#include "mathlib/probability.hpp"   // Probability functions / 概率函数
#include "mathlib/combinatorics.hpp" // Combinatorics / 组合数学
#include "mathlib/numerical.hpp"     // Numerical analysis / 数值分析
#include "mathlib/optimization.hpp"  // Optimization algorithms / 优化算法
#include "mathlib/signal.hpp"        // Signal processing / 信号处理
```

## Main Features / 主要功能

### 1. Statistical Functions / 统计函数 (statistics.hpp)
- Descriptive Statistics: mean, variance, standard deviation, median, etc.
  描述性统计：均值、方差、标准差、中位数等
- Hypothesis Testing: t-test, chi-square test, etc.
  假设检验：t检验、卡方检验等
- Confidence Interval Calculation
  置信区间计算
- Correlation Analysis
  相关性分析

### 2. Probability Functions / 概率函数 (probability.hpp)
- Probability Distributions: normal, binomial, Poisson, etc.
  概率分布：正态分布、二项分布、泊松分布等
- Random Number Generators
  随机数生成器
- Probability Theory Tools: entropy, mutual information, etc.
  概率论工具：熵、互信息等
- Markov Chains
  马尔可夫链

### 3. Combinatorics / 组合数学 (combinatorics.hpp)
- Basic Combinatorial Functions: factorial, combinations, permutations
  基本组合函数：阶乘、组合数、排列数
- Polynomials: Laguerre polynomials, Chebyshev polynomials
  多项式：拉盖尔多项式、切比雪夫多项式
- Generating Functions: combination generation, permutation generation
  生成函数：组合生成、排列生成
- Combinatorial Optimization: binomial coefficients, polynomial coefficients
  组合优化：二项式系数、多项式系数
- Combinatorial Identities: Stirling numbers, Bell numbers
  组合恒等式：斯特林数、贝尔数

### 4. Numerical Analysis / 数值分析 (numerical.hpp)
- Numerical Integration / 数值积分:
  - Trapezoidal Rule / 梯形法则
  - Simpson's Rule / 辛普森法则
  - Gaussian Quadrature / 高斯求积
  - Gauss-Legendre Quadrature / 高斯-勒让德求积
  - Adaptive Quadrature / 自适应积分
  - Multiple Integration / 多重积分
- Numerical Differentiation / 数值微分:
  - Central Difference / 中心差分
  - Forward Difference / 前向差分
- Equation Solving / 方程求解:
  - Newton's Method / 牛顿法
  - Bisection Method / 二分法
- Interpolation Methods / 插值方法:
  - Lagrange Interpolation / 拉格朗日插值
- Ordinary Differential Equations / 常微分方程:
  - Euler Method / 欧拉法
  - Runge-Kutta Method / 龙格-库塔法

### 5. Optimization Algorithms / 优化算法 (optimization.hpp)
- Unconstrained Optimization / 无约束优化:
  - Gradient Descent / 梯度下降
  - Conjugate Gradient Method / 共轭梯度法
- Global Optimization / 全局优化:
  - Simulated Annealing / 模拟退火
  - Particle Swarm Optimization / 粒子群优化
  - Genetic Algorithm / 遗传算法
- Linear Programming / 线性规划:
  - Simplex Method / 单纯形法

### 6. Signal Processing / 信号处理 (signal.hpp)
- Fourier Transform / 傅里叶变换:
  - FFT (Fast Fourier Transform) / 快速傅里叶变换
  - IFFT (Inverse Fast Fourier Transform) / 逆快速傅里叶变换
  - Short-Time Fourier Transform / 短时傅里叶变换
- Filters / 滤波器:
  - FIR Filter / FIR滤波器
  - IIR Filter / IIR滤波器
  - Adaptive Filter / 自适应滤波器
- Spectral Analysis / 频谱分析:
  - Power Spectral Density / 功率谱密度
  - Cepstrum Analysis / 倒谱分析
- Signal Analysis / 信号分析:
  - Convolution / 卷积
  - Autocorrelation / 自相关
  - Wavelet Transform / 小波变换
  - Hilbert Transform / 希尔伯特变换

## Example Code / 示例代码

### Numerical Analysis Example / 数值分析示例
```cpp
#include "mathlib/numerical.hpp"

// Using Gauss-Legendre quadrature for integration
// 使用高斯-勒让德求积法计算积分
auto f = [](double x) { return x * x; };
double result = mathlib::numerical::gauss_legendre_quadrature(f, 0.0, 1.0, 4);

// Using adaptive quadrature
// 使用自适应积分法
double result2 = mathlib::numerical::adaptive_quadrature(f, 0.0, 1.0);

// Computing double integral
// 计算二重积分
auto f2 = [](double x, double y) { return x * x + y * y; };
double result3 = mathlib::numerical::double_integral(f2, 0.0, 1.0, 0.0, 1.0);
```

### Optimization Example / 优化算法示例
```cpp
#include "mathlib/optimization.hpp"

// Using Particle Swarm Optimization
// 使用粒子群优化
auto objective = [](const std::vector<double>& x) {
    return x[0] * x[0] + x[1] * x[1];
};
std::vector<double> lower_bounds = {-10.0, -10.0};
std::vector<double> upper_bounds = {10.0, 10.0};
auto result = mathlib::optimization::particle_swarm_optimization(
    objective, lower_bounds, upper_bounds);

// Using Genetic Algorithm
// 使用遗传算法
auto result2 = mathlib::optimization::genetic_algorithm(
    objective, lower_bounds, upper_bounds);
```

### Signal Processing Example / 信号处理示例
```cpp
#include "mathlib/signal.hpp"

// Wavelet Transform
// 小波变换
std::vector<double> signal = {1.0, 2.0, 3.0, 4.0};
std::vector<double> wavelet = {0.5, 0.5};
auto result = mathlib::signal::wavelet_transform(signal, wavelet, 2);

// Short-Time Fourier Transform
// 短时傅里叶变换
auto stft = mathlib::signal::short_time_fourier_transform(signal, 4, 2);

// Using Adaptive Filter
// 使用自适应滤波器
mathlib::signal::AdaptiveFilter filter(4);
std::vector<double> desired = {1.0, 1.5, 2.0, 2.5};
auto filtered = filter.filter(signal, desired);
```

## Building and Testing / 编译和测试

Build the project using CMake:
使用CMake构建项目：

```bash
mkdir build
cd build
cmake ..
make
make test
```

## License / 许可证

MIT License 
