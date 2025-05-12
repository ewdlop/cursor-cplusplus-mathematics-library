#ifndef MATHLIB_HPP
#define MATHLIB_HPP

#include <vector>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <algorithm>
#include <map>
#include <unordered_map>

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

    // 计算中位数
    template<typename T>
    double median(std::vector<T> data) {
        if (data.empty()) {
            throw std::invalid_argument("数据不能为空");
        }
        std::sort(data.begin(), data.end());
        size_t n = data.size();
        if (n % 2 == 0) {
            return (data[n/2 - 1] + data[n/2]) / 2.0;
        } else {
            return data[n/2];
        }
    }

    // 计算众数
    template<typename T>
    T mode(const std::vector<T>& data) {
        if (data.empty()) {
            throw std::invalid_argument("数据不能为空");
        }
        std::unordered_map<T, size_t> counts;
        for (const auto& x : data) {
            counts[x]++;
        }
        return std::max_element(counts.begin(), counts.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; })->first;
    }

    // 计算分位数
    template<typename T>
    double quantile(std::vector<T> data, double q) {
        if (data.empty()) {
            throw std::invalid_argument("数据不能为空");
        }
        if (q < 0.0 || q > 1.0) {
            throw std::invalid_argument("分位数必须在0到1之间");
        }
        std::sort(data.begin(), data.end());
        size_t n = data.size();
        double index = q * (n - 1);
        size_t i = static_cast<size_t>(index);
        double fraction = index - i;
        if (i + 1 >= n) {
            return data[n-1];
        }
        return data[i] + fraction * (data[i+1] - data[i]);
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

    // 二项分布
    class BinomialDistribution {
    private:
        int n_;
        double p_;
        std::mt19937 rng_;

    public:
        BinomialDistribution(int n, double p)
            : n_(n), p_(p), rng_(std::random_device{}()) {
            if (p < 0.0 || p > 1.0) {
                throw std::invalid_argument("概率必须在0到1之间");
            }
            if (n < 0) {
                throw std::invalid_argument("试验次数必须非负");
            }
        }

        double pmf(int k) const {
            if (k < 0 || k > n_) return 0.0;
            return mathlib::combinatorics::combination(n_, k) * 
                   std::pow(p_, k) * std::pow(1-p_, n_-k);
        }

        double cdf(int k) const {
            if (k < 0) return 0.0;
            if (k >= n_) return 1.0;
            double sum = 0.0;
            for (int i = 0; i <= k; ++i) {
                sum += pmf(i);
            }
            return sum;
        }

        int sample() {
            std::binomial_distribution<int> dist(n_, p_);
            return dist(rng_);
        }
    };

    // 泊松分布
    class PoissonDistribution {
    private:
        double lambda_;
        std::mt19937 rng_;

    public:
        PoissonDistribution(double lambda)
            : lambda_(lambda), rng_(std::random_device{}()) {
            if (lambda <= 0.0) {
                throw std::invalid_argument("参数必须为正数");
            }
        }

        double pmf(int k) const {
            if (k < 0) return 0.0;
            return std::pow(lambda_, k) * std::exp(-lambda_) / 
                   mathlib::combinatorics::factorial(k);
        }

        double cdf(int k) const {
            if (k < 0) return 0.0;
            double sum = 0.0;
            for (int i = 0; i <= k; ++i) {
                sum += pmf(i);
            }
            return sum;
        }

        int sample() {
            std::poisson_distribution<int> dist(lambda_);
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

    // 计算第一类斯特林数 S(n,k)
    unsigned long long stirling_first(unsigned int n, unsigned int k) {
        if (k > n) return 0;
        if (k == 0) return (n == 0) ? 1 : 0;
        if (k == n) return 1;
        return (n-1) * stirling_first(n-1, k) + stirling_first(n-1, k-1);
    }

    // 计算第二类斯特林数 S(n,k)
    unsigned long long stirling_second(unsigned int n, unsigned int k) {
        if (k > n) return 0;
        if (k == 0) return (n == 0) ? 1 : 0;
        if (k == 1 || k == n) return 1;
        return k * stirling_second(n-1, k) + stirling_second(n-1, k-1);
    }

    // 计算卡特兰数 C(n)
    unsigned long long catalan(unsigned int n) {
        if (n <= 1) return 1;
        unsigned long long result = 0;
        for (unsigned int i = 0; i < n; ++i) {
            result += catalan(i) * catalan(n-1-i);
        }
        return result;
    }

    // 计算贝尔数 B(n)
    unsigned long long bell(unsigned int n) {
        if (n == 0) return 1;
        std::vector<unsigned long long> bell_numbers(n + 1);
        bell_numbers[0] = 1;
        for (unsigned int i = 1; i <= n; ++i) {
            bell_numbers[i] = 0;
            for (unsigned int j = 0; j < i; ++j) {
                bell_numbers[i] += combination(i-1, j) * bell_numbers[j];
            }
        }
        return bell_numbers[n];
    }
}

} // namespace mathlib

#endif // MATHLIB_HPP 