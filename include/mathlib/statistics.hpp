#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace mathlib {
namespace statistics {

// 基础统计函数
template<typename T>
T mean(const std::vector<T>& data) {
    if (data.empty()) {
        throw std::invalid_argument("Empty data set");
    }
    return std::accumulate(data.begin(), data.end(), T(0)) / data.size();
}

template<typename T>
T variance(const std::vector<T>& data, bool sample = true) {
    if (data.empty()) {
        throw std::invalid_argument("Empty data set");
    }
    T m = mean(data);
    T sum_sq = std::accumulate(data.begin(), data.end(), T(0),
        [m](T acc, T x) { return acc + (x - m) * (x - m); });
    return sum_sq / (data.size() - (sample ? 1 : 0));
}

template<typename T>
T standard_deviation(const std::vector<T>& data, bool sample = true) {
    return std::sqrt(variance(data, sample));
}

template<typename T>
T median(const std::vector<T>& data) {
    if (data.empty()) {
        throw std::invalid_argument("Empty data set");
    }
    std::vector<T> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());
    size_t n = sorted_data.size();
    if (n % 2 == 0) {
        return (sorted_data[n/2 - 1] + sorted_data[n/2]) / 2;
    } else {
        return sorted_data[n/2];
    }
}

template<typename T>
T mode(const std::vector<T>& data) {
    if (data.empty()) {
        throw std::invalid_argument("Empty data set");
    }
    std::vector<T> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());
    
    T current_value = sorted_data[0];
    T mode_value = current_value;
    size_t current_count = 1;
    size_t max_count = 1;
    
    for (size_t i = 1; i < sorted_data.size(); ++i) {
        if (sorted_data[i] == current_value) {
            ++current_count;
        } else {
            if (current_count > max_count) {
                max_count = current_count;
                mode_value = current_value;
            }
            current_value = sorted_data[i];
            current_count = 1;
        }
    }
    
    if (current_count > max_count) {
        mode_value = current_value;
    }
    
    return mode_value;
}

// 高级统计函数
template<typename T>
T skewness(const std::vector<T>& data) {
    if (data.size() < 3) {
        throw std::invalid_argument("Need at least 3 data points for skewness");
    }
    T m = mean(data);
    T s = standard_deviation(data);
    T sum_cubed = std::accumulate(data.begin(), data.end(), T(0),
        [m, s](T acc, T x) { 
            T z = (x - m) / s;
            return acc + z * z * z;
        });
    return sum_cubed / data.size();
}

template<typename T>
T kurtosis(const std::vector<T>& data) {
    if (data.size() < 4) {
        throw std::invalid_argument("Need at least 4 data points for kurtosis");
    }
    T m = mean(data);
    T s = standard_deviation(data);
    T sum_quartic = std::accumulate(data.begin(), data.end(), T(0),
        [m, s](T acc, T x) { 
            T z = (x - m) / s;
            return acc + z * z * z * z;
        });
    return sum_quartic / data.size() - 3; // 超额峰度
}

// 假设检验
template<typename T>
struct TTestResult {
    T t_statistic;
    T p_value;
    T degrees_of_freedom;
};

template<typename T>
TTestResult<T> t_test(const std::vector<T>& sample1, const std::vector<T>& sample2) {
    if (sample1.empty() || sample2.empty()) {
        throw std::invalid_argument("Empty sample");
    }
    
    T mean1 = mean(sample1);
    T mean2 = mean(sample2);
    T var1 = variance(sample1);
    T var2 = variance(sample2);
    
    T n1 = static_cast<T>(sample1.size());
    T n2 = static_cast<T>(sample2.size());
    
    T pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2);
    T t_stat = (mean1 - mean2) / std::sqrt(pooled_var * (1/n1 + 1/n2));
    T df = n1 + n2 - 2;
    
    // 简化处理，实际应该使用t分布计算p值
    T p_value = 2 * (1 - std::erf(std::abs(t_stat) / std::sqrt(2)));
    
    return {t_stat, p_value, df};
}

// 置信区间
template<typename T>
struct ConfidenceInterval {
    T lower;
    T upper;
    T confidence_level;
};

template<typename T>
ConfidenceInterval<T> confidence_interval(const std::vector<T>& data, T confidence_level = 0.95) {
    if (data.empty()) {
        throw std::invalid_argument("Empty data set");
    }
    if (confidence_level <= 0 || confidence_level >= 1) {
        throw std::invalid_argument("Confidence level must be between 0 and 1");
    }
    
    T m = mean(data);
    T s = standard_deviation(data);
    T n = static_cast<T>(data.size());
    
    // 使用正态分布近似
    T z = std::sqrt(2) * std::erfinv(confidence_level);
    T margin = z * s / std::sqrt(n);
    
    return {m - margin, m + margin, confidence_level};
}

} // namespace statistics
} // namespace mathlib 