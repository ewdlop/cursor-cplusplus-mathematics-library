# C++ 数学库

这是一个C++数学库，提供了统计、概率和组合数学相关的功能。

## 功能特点

### 统计功能
- 计算均值
- 计算方差
- 计算标准差

### 概率分布
- 正态分布
  - 概率密度函数 (PDF)
  - 累积分布函数 (CDF)
  - 随机采样

### 组合数学
- 阶乘计算
- 组合数计算
- 排列数计算

## 构建要求
- CMake 3.10 或更高版本
- C++17 兼容的编译器

## 构建步骤

```bash
mkdir build
cd build
cmake ..
make
```

## 使用示例

```cpp
#include "mathlib.hpp"
#include <vector>

int main() {
    // 统计示例
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    double mean = mathlib::statistics::mean(data);
    
    // 正态分布示例
    mathlib::probability::NormalDistribution normal(0.0, 1.0);
    double pdf = normal.pdf(0.0);
    
    // 组合数学示例
    unsigned long long comb = mathlib::combinatorics::combination(5, 2);
    
    return 0;
}
```

## 许可证

MIT License 