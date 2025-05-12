#ifndef MATHLIB_HPP
#define MATHLIB_HPP

#include <vector>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

namespace mathlib {

// 统计函数
namespace statistics {
    // 计算均值
    template<typename T>
    double mean(const std::vector<T>& data) {
        if (data.empty()) {
            throw std::invalid_argument("数据不能为空");
        }
        return std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    }

    // 计算方差
    template<typename T>
    double variance(const std::vector<T>& data) {
        if (data.empty()) {
            throw std::invalid_argument("数据不能为空");
        }
        double m = mean(data);
        double sum_sq_diff = 0.0;
        for (const auto& x : data) {
            sum_sq_diff += (x - m) * (x - m);
        }
        return sum_sq_diff / data.size();
    }

    // 计算标准差
    template<typename T>
    double standard_deviation(const std::vector<T>& data) {
        return std::sqrt(variance(data));
    }
}

// 概率分布
namespace probability {
    class NormalDistribution {
    private:
        double mean_;
        double stddev_;
        std::mt19937 rng_;

    public:
        NormalDistribution(double mean = 0.0, double stddev = 1.0)
            : mean_(mean), stddev_(stddev), rng_(std::random_device{}()) {}

        double pdf(double x) const {
            double z = (x - mean_) / stddev_;
            return std::exp(-0.5 * z * z) / (stddev_ * std::sqrt(2 * M_PI));
        }

        double cdf(double x) const {
            return 0.5 * (1 + std::erf((x - mean_) / (stddev_ * std::sqrt(2))));
        }

        double sample() {
            std::normal_distribution<double> dist(mean_, stddev_);
            return dist(rng_);
        }
    };
}

// 组合数学
namespace combinatorics {
    // 计算阶乘
    unsigned long long factorial(unsigned int n) {
        if (n > 20) {
            throw std::overflow_error("阶乘结果超出范围");
        }
        unsigned long long result = 1;
        for (unsigned int i = 2; i <= n; ++i) {
            result *= i;
        }
        return result;
    }

    // 计算组合数 C(n,k)
    unsigned long long combination(unsigned int n, unsigned int k) {
        if (k > n) {
            throw std::invalid_argument("k 不能大于 n");
        }
        if (k > n - k) {
            k = n - k;
        }
        unsigned long long result = 1;
        for (unsigned int i = 0; i < k; ++i) {
            result *= (n - i);
            result /= (i + 1);
        }
        return result;
    }

    // 计算排列数 P(n,k)
    unsigned long long permutation(unsigned int n, unsigned int k) {
        if (k > n) {
            throw std::invalid_argument("k 不能大于 n");
        }
        unsigned long long result = 1;
        for (unsigned int i = 0; i < k; ++i) {
            result *= (n - i);
        }
        return result;
    }
}

} // namespace mathlib

#endif // MATHLIB_HPP 