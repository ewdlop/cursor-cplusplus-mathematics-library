#include <iostream>
#include <vector>
#include "mathlib.hpp"

int main() {
    // 统计示例
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::cout << "统计示例：" << std::endl;
    std::cout << "均值: " << mathlib::statistics::mean(data) << std::endl;
    std::cout << "方差: " << mathlib::statistics::variance(data) << std::endl;
    std::cout << "标准差: " << mathlib::statistics::standard_deviation(data) << std::endl;

    // 概率分布示例
    mathlib::probability::NormalDistribution normal(0.0, 1.0);
    std::cout << "\n正态分布示例：" << std::endl;
    std::cout << "PDF(0): " << normal.pdf(0.0) << std::endl;
    std::cout << "CDF(0): " << normal.cdf(0.0) << std::endl;
    std::cout << "随机样本: " << normal.sample() << std::endl;

    // 组合数学示例
    std::cout << "\n组合数学示例：" << std::endl;
    std::cout << "5! = " << mathlib::combinatorics::factorial(5) << std::endl;
    std::cout << "C(5,2) = " << mathlib::combinatorics::combination(5, 2) << std::endl;
    std::cout << "P(5,2) = " << mathlib::combinatorics::permutation(5, 2) << std::endl;

    return 0;
} 