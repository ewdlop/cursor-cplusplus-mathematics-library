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
#include <numbers>
#include <tuple>

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

    // 计算协方差
    template<typename T>
    double covariance(const std::vector<T>& x, const std::vector<T>& y) {
        if (x.empty() || y.empty()) {
            throw std::invalid_argument("数据不能为空");
        }
        if (x.size() != y.size()) {
            throw std::invalid_argument("数据长度必须相同");
        }
        double mean_x = mean(x);
        double mean_y = mean(y);
        double sum = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            sum += (x[i] - mean_x) * (y[i] - mean_y);
        }
        return sum / x.size();
    }

    // 计算相关系数
    template<typename T>
    double correlation(const std::vector<T>& x, const std::vector<T>& y) {
        if (x.empty() || y.empty()) {
            throw std::invalid_argument("数据不能为空");
        }
        if (x.size() != y.size()) {
            throw std::invalid_argument("数据长度必须相同");
        }
        double cov = covariance(x, y);
        double std_x = standard_deviation(x);
        double std_y = standard_deviation(y);
        if (std_x == 0.0 || std_y == 0.0) {
            throw std::runtime_error("标准差不能为零");
        }
        return cov / (std_x * std_y);
    }

    // 计算偏度
    template<typename T>
    double skewness(const std::vector<T>& data) {
        if (data.empty()) {
            throw std::invalid_argument("数据不能为空");
        }
        double m = mean(data);
        double s = standard_deviation(data);
        if (s == 0.0) {
            throw std::runtime_error("标准差不能为零");
        }
        double sum = 0.0;
        for (const auto& x : data) {
            sum += std::pow((x - m) / s, 3);
        }
        return sum / data.size();
    }

    // 计算峰度
    template<typename T>
    double kurtosis(const std::vector<T>& data) {
        if (data.empty()) {
            throw std::invalid_argument("数据不能为空");
        }
        double m = mean(data);
        double s = standard_deviation(data);
        if (s == 0.0) {
            throw std::runtime_error("标准差不能为零");
        }
        double sum = 0.0;
        for (const auto& x : data) {
            sum += std::pow((x - m) / s, 4);
        }
        return sum / data.size() - 3.0; // 超额峰度
    }

    // 计算t检验统计量
    template<typename T>
    double t_statistic(const std::vector<T>& sample, double mu0) {
        if (sample.empty()) {
            throw std::invalid_argument("样本不能为空");
        }
        double m = mean(sample);
        double s = standard_deviation(sample);
        if (s == 0.0) {
            throw std::runtime_error("样本标准差不能为零");
        }
        return (m - mu0) / (s / std::sqrt(sample.size()));
    }

    // 计算F检验统计量
    template<typename T>
    double f_statistic(const std::vector<T>& sample1, const std::vector<T>& sample2) {
        if (sample1.empty() || sample2.empty()) {
            throw std::invalid_argument("样本不能为空");
        }
        double var1 = variance(sample1);
        double var2 = variance(sample2);
        if (var2 == 0.0) {
            throw std::runtime_error("第二个样本的方差不能为零");
        }
        return var1 / var2;
    }

    // 计算置信区间
    template<typename T>
    std::tuple<double, double> confidence_interval(const std::vector<T>& sample, double confidence_level) {
        if (sample.empty()) {
            throw std::invalid_argument("样本不能为空");
        }
        if (confidence_level <= 0.0 || confidence_level >= 1.0) {
            throw std::invalid_argument("置信水平必须在0到1之间");
        }

        double m = mean(sample);
        double s = standard_deviation(sample);
        double n = static_cast<double>(sample.size());
        
        // 使用t分布的分位数
        double alpha = (1.0 - confidence_level) / 2.0;
        double t = std::abs(mathlib::probability::t_distribution_quantile(n - 1, alpha));
        
        double margin = t * s / std::sqrt(n);
        return std::make_tuple(m - margin, m + margin);
    }

    // 计算p值
    template<typename T>
    double p_value(double test_statistic, int degrees_of_freedom, bool two_tailed = true) {
        double p = 1.0 - mathlib::probability::t_distribution_cdf(std::abs(test_statistic), degrees_of_freedom);
        return two_tailed ? 2.0 * p : p;
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

    // 指数分布
    class ExponentialDistribution {
    private:
        double lambda_;
        std::mt19937 rng_;

    public:
        ExponentialDistribution(double lambda)
            : lambda_(lambda), rng_(std::random_device{}()) {
            if (lambda <= 0.0) {
                throw std::invalid_argument("参数必须为正数");
            }
        }

        double pdf(double x) const {
            if (x < 0.0) return 0.0;
            return lambda_ * std::exp(-lambda_ * x);
        }

        double cdf(double x) const {
            if (x < 0.0) return 0.0;
            return 1.0 - std::exp(-lambda_ * x);
        }

        double sample() {
            std::exponential_distribution<double> dist(lambda_);
            return dist(rng_);
        }
    };

    // 卡方分布
    class ChiSquaredDistribution {
    private:
        int df_;
        std::mt19937 rng_;

        // 不完全伽马函数
        double incomplete_gamma(double a, double x) const {
            if (x <= 0.0) return 0.0;
            if (x < a + 1.0) {
                // 使用级数展开
                double sum = 0.0;
                double term = 1.0 / a;
                for (int n = 0; n < 100; ++n) {
                    sum += term;
                    term *= x / (a + n + 1);
                }
                return std::pow(x, a) * std::exp(-x) * sum;
            } else {
                // 使用连分数展开
                double b = x + 1.0 - a;
                double c = 1.0 / std::numeric_limits<double>::min();
                double d = 1.0 / b;
                double h = d;
                for (int i = 1; i <= 100; ++i) {
                    double an = -i * (i - a);
                    b += 2.0;
                    d = an * d + b;
                    if (std::abs(d) < std::numeric_limits<double>::min()) d = std::numeric_limits<double>::min();
                    c = b + an / c;
                    if (std::abs(c) < std::numeric_limits<double>::min()) c = std::numeric_limits<double>::min();
                    d = 1.0 / d;
                    double del = d * c;
                    h *= del;
                    if (std::abs(del - 1.0) < std::numeric_limits<double>::epsilon()) break;
                }
                return 1.0 - std::pow(x, a) * std::exp(-x) * h;
            }
        }

    public:
        ChiSquaredDistribution(int degrees_of_freedom)
            : df_(degrees_of_freedom), rng_(std::random_device{}()) {
            if (degrees_of_freedom <= 0) {
                throw std::invalid_argument("自由度必须为正数");
            }
        }

        double pdf(double x) const {
            if (x < 0.0) return 0.0;
            return std::pow(x, df_/2.0 - 1) * std::exp(-x/2.0) / 
                   (std::pow(2, df_/2.0) * std::tgamma(df_/2.0));
        }

        double cdf(double x) const {
            if (x < 0.0) return 0.0;
            return incomplete_gamma(df_/2.0, x/2.0) / std::tgamma(df_/2.0);
        }

        double sample() {
            std::chi_squared_distribution<double> dist(df_);
            return dist(rng_);
        }
    };

    // t分布
    class TDistribution {
    private:
        int df_;
        std::mt19937 rng_;

    public:
        TDistribution(int degrees_of_freedom)
            : df_(degrees_of_freedom), rng_(std::random_device{}()) {
            if (degrees_of_freedom <= 0) {
                throw std::invalid_argument("自由度必须为正数");
            }
        }

        double pdf(double x) const {
            return std::tgamma((df_ + 1.0) / 2.0) / 
                   (std::sqrt(df_ * M_PI) * std::tgamma(df_ / 2.0)) *
                   std::pow(1.0 + x * x / df_, -(df_ + 1.0) / 2.0);
        }

        double cdf(double x) const {
            if (x == 0.0) return 0.5;
            double z = df_ / (df_ + x * x);
            return 0.5 + 0.5 * std::copysign(1.0, x) * 
                   (1.0 - special::incomplete_beta(z, df_/2.0, 0.5));
        }

        double sample() {
            std::student_t_distribution<double> dist(df_);
            return dist(rng_);
        }

        static double quantile(int df, double p) {
            if (p < 0.0 || p > 1.0) {
                throw std::invalid_argument("概率必须在0到1之间");
            }
            if (p == 0.5) return 0.0;
            
            // 使用近似方法计算分位数
            double sign = (p < 0.5) ? -1.0 : 1.0;
            p = (p < 0.5) ? p : 1.0 - p;
            
            double x = std::sqrt(-2.0 * std::log(2.0 * p));
            double c0 = 2.515517;
            double c1 = 0.802853;
            double c2 = 0.010328;
            double d1 = 1.432788;
            double d2 = 0.189269;
            double d3 = 0.001308;
            
            double t = x - (c0 + c1 * x + c2 * x * x) / 
                      (1.0 + d1 * x + d2 * x * x + d3 * x * x * x);
            
            return sign * t * std::sqrt(df / (df - 2.0));
        }
    };

    // F分布
    class FDistribution {
    private:
        int df1_;
        int df2_;
        std::mt19937 rng_;

    public:
        FDistribution(int df1, int df2)
            : df1_(df1), df2_(df2), rng_(std::random_device{}()) {
            if (df1 <= 0 || df2 <= 0) {
                throw std::invalid_argument("自由度必须为正数");
            }
        }

        double pdf(double x) const {
            if (x < 0.0) return 0.0;
            return std::sqrt(std::pow(df1_ * x, df1_) * std::pow(df2_, df2_) / 
                           std::pow(df1_ * x + df2_, df1_ + df2_)) /
                   (x * special::beta(df1_/2.0, df2_/2.0));
        }

        double cdf(double x) const {
            if (x < 0.0) return 0.0;
            double z = df1_ * x / (df1_ * x + df2_);
            return special::incomplete_beta(z, df1_/2.0, df2_/2.0) / 
                   special::beta(df1_/2.0, df2_/2.0);
        }

        double sample() {
            std::fisher_f_distribution<double> dist(df1_, df2_);
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

    // 计算欧拉数 E(n,k)
    unsigned long long euler(unsigned int n, unsigned int k) {
        if (k >= n) return 0;
        if (k == 0) return 1;
        if (n == 0) return 1;
        return (k + 1) * euler(n-1, k) + (n - k) * euler(n-1, k-1);
    }

    // 计算伯努利数 B(n)
    double bernoulli(unsigned int n) {
        if (n == 0) return 1.0;
        if (n == 1) return -0.5;
        if (n % 2 == 1) return 0.0;

        std::vector<double> b(n + 1);
        b[0] = 1.0;
        b[1] = -0.5;

        for (unsigned int m = 2; m <= n; ++m) {
            b[m] = 0.0;
            for (unsigned int k = 0; k < m; ++k) {
                b[m] -= combination(m + 1, k) * b[k];
            }
            b[m] /= (m + 1);
        }

        return b[n];
    }

    // 计算欧拉-马歇罗尼常数
    double euler_mascheroni() {
        const int n = 1000000;
        double sum = 0.0;
        for (int i = 1; i <= n; ++i) {
            sum += 1.0 / i;
        }
        return sum - std::log(n);
    }

    // 计算调和数 H(n)
    double harmonic(unsigned int n) {
        double sum = 0.0;
        for (unsigned int i = 1; i <= n; ++i) {
            sum += 1.0 / i;
        }
        return sum;
    }

    // 计算欧拉多项式 E_n(x)
    double euler_polynomial(unsigned int n, double x) {
        if (n == 0) return 1.0;
        if (n == 1) return x - 0.5;
        
        std::vector<double> e(n + 1);
        e[0] = 1.0;
        e[1] = x - 0.5;
        
        for (unsigned int i = 2; i <= n; ++i) {
            e[i] = (x - 0.5) * e[i-1] - 0.5 * i * e[i-2];
        }
        
        return e[n];
    }

    // 计算伯努利多项式 B_n(x)
    double bernoulli_polynomial(unsigned int n, double x) {
        if (n == 0) return 1.0;
        if (n == 1) return x - 0.5;
        
        std::vector<double> b(n + 1);
        b[0] = 1.0;
        b[1] = x - 0.5;
        
        for (unsigned int i = 2; i <= n; ++i) {
            b[i] = x * b[i-1];
            for (unsigned int j = 0; j < i; ++j) {
                b[i] -= combination(i, j) * bernoulli(i-j) * b[j];
            }
            b[i] /= i;
        }
        
        return b[n];
    }

    // 计算广义调和数 H(n,r)
    double generalized_harmonic(unsigned int n, double r) {
        if (r <= 0.0) {
            throw std::invalid_argument("阶数必须为正数");
        }
        double sum = 0.0;
        for (unsigned int i = 1; i <= n; ++i) {
            sum += 1.0 / std::pow(i, r);
        }
        return sum;
    }

    // 计算多重对数函数 Li_s(z)
    double polylogarithm(double s, double z) {
        if (std::abs(z) >= 1.0) {
            throw std::invalid_argument("|z|必须小于1");
        }
        if (s <= 0.0) {
            throw std::invalid_argument("s必须为正数");
        }
        
        double sum = 0.0;
        double term = z;
        for (int n = 1; n < 1000; ++n) {
            sum += term / std::pow(n, s);
            term *= z;
            if (std::abs(term) < std::numeric_limits<double>::epsilon()) break;
        }
        return sum;
    }
}

// 特殊函数
namespace special {
    // 计算beta函数 B(a,b)
    double beta(double a, double b) {
        return std::tgamma(a) * std::tgamma(b) / std::tgamma(a + b);
    }

    // 计算不完全beta函数 I_x(a,b)
    double incomplete_beta(double x, double a, double b) {
        if (x < 0.0 || x > 1.0) {
            throw std::invalid_argument("x必须在0到1之间");
        }
        if (a <= 0.0 || b <= 0.0) {
            throw std::invalid_argument("参数必须为正数");
        }

        // 使用连分数展开
        const double eps = std::numeric_limits<double>::epsilon();
        const int max_iter = 1000;

        double c = 1.0;
        double d = 1.0 - (a + b) * x / (a + 1.0);
        if (std::abs(d) < eps) d = eps;
        d = 1.0 / d;
        double h = d;

        for (int i = 1; i <= max_iter; ++i) {
            double m = i / 2.0;
            double an;
            if (i % 2 == 0) {
                an = m * (b - m) * x / ((a + 2.0 * m - 1.0) * (a + 2.0 * m));
            } else {
                an = -(a + m) * (a + b + m) * x / ((a + 2.0 * m) * (a + 2.0 * m + 1.0));
            }

            d = 1.0 + an * d;
            if (std::abs(d) < eps) d = eps;
            c = 1.0 + an / c;
            if (std::abs(c) < eps) c = eps;
            d = 1.0 / d;
            double del = d * c;
            h *= del;
            if (std::abs(del - 1.0) < eps) break;
        }

        return std::pow(x, a) * std::pow(1.0 - x, b) * h / (a * beta(a, b));
    }
}

} // namespace mathlib

#endif // MATHLIB_HPP 