#include "../include/mathlib.hpp"
#include <iostream>
#include <iomanip>
#include <vector>

int main() {
    // 创建示例数据
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    
    // 基础统计
    std::cout << "Basic Statistics / 基础统计:" << std::endl;
    std::cout << "Mean / 均值: " << mathlib::statistics::mean(data) << std::endl;
    std::cout << "Variance / 方差: " << mathlib::statistics::variance(data) << std::endl;
    std::cout << "Standard Deviation / 标准差: " << mathlib::statistics::standard_deviation(data) << std::endl;
    std::cout << "Median / 中位数: " << mathlib::statistics::median(data) << std::endl;
    
    // 高级统计
    std::cout << "\nAdvanced Statistics / 高级统计:" << std::endl;
    std::cout << "Skewness / 偏度: " << mathlib::statistics::skewness(data) << std::endl;
    std::cout << "Kurtosis / 峰度: " << mathlib::statistics::kurtosis(data) << std::endl;
    
    // 假设检验
    std::cout << "\nHypothesis Testing / 假设检验:" << std::endl;
    double t_stat = mathlib::statistics::t_statistic(data, 5.0);
    std::cout << "t-statistic / t统计量: " << t_stat << std::endl;
    
    // 置信区间
    auto [lower, upper] = mathlib::statistics::confidence_interval(data, 0.95);
    std::cout << "95% Confidence Interval / 95%置信区间: [" 
              << lower << ", " << upper << "]" << std::endl;
    
    // 非参数检验
    std::vector<double> data2 = {2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0};
    double wilcoxon = mathlib::statistics::wilcoxon_rank_sum(data, data2);
    std::cout << "Wilcoxon Rank-Sum Test / Wilcoxon秩和检验: " << wilcoxon << std::endl;
    
    return 0;
} 