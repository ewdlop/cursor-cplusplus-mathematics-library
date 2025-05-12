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
    double correlation(const std::vector<double>& x, const std::vector<double>& y) {
        if (x.size() != y.size() || x.empty()) {
            throw std::invalid_argument("Invalid input vectors");
        }
        double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0;
        double sum_x2 = 0.0, sum_y2 = 0.0;
        size_t n = x.size();
        
        for (size_t i = 0; i < n; ++i) {
            sum_x += x[i];
            sum_y += y[i];
            sum_xy += x[i] * y[i];
            sum_x2 += x[i] * x[i];
            sum_y2 += y[i] * y[i];
        }
        
        double numerator = n * sum_xy - sum_x * sum_y;
        double denominator = std::sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y));
        
        return numerator / denominator;
    }

    // 计算斯皮尔曼等级相关系数
    double spearman_correlation(const std::vector<double>& x, const std::vector<double>& y) {
        if (x.size() != y.size() || x.empty()) {
            throw std::invalid_argument("Invalid input vectors");
        }
        
        // 计算秩
        auto rank = [](const std::vector<double>& v) {
            std::vector<size_t> indices(v.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::sort(indices.begin(), indices.end(),
                     [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });
            
            std::vector<double> ranks(v.size());
            for (size_t i = 0; i < indices.size(); ++i) {
                ranks[indices[i]] = i + 1;
            }
            return ranks;
        };
        
        std::vector<double> rank_x = rank(x);
        std::vector<double> rank_y = rank(y);
        
        return correlation(rank_x, rank_y);
    }

    // 计算肯德尔等级相关系数
    double kendall_correlation(const std::vector<double>& x, const std::vector<double>& y) {
        if (x.size() != y.size() || x.empty()) {
            throw std::invalid_argument("Invalid input vectors");
        }
        
        size_t n = x.size();
        int concordant = 0, discordant = 0;
        
        for (size_t i = 0; i < n - 1; ++i) {
            for (size_t j = i + 1; j < n; ++j) {
                double dx = x[j] - x[i];
                double dy = y[j] - y[i];
                if (dx * dy > 0) {
                    ++concordant;
                } else if (dx * dy < 0) {
                    ++discordant;
                }
            }
        }
        
        return static_cast<double>(concordant - discordant) / (n * (n - 1) / 2);
    }

    // 计算弗里德曼检验
    double friedman_test(const std::vector<std::vector<double>>& data) {
        if (data.empty() || data[0].empty()) {
            throw std::invalid_argument("Invalid input data");
        }
        
        size_t k = data.size();  // 处理数
        size_t n = data[0].size();  // 区组数
        
        // 计算每个区组中的秩
        std::vector<std::vector<double>> ranks(k, std::vector<double>(n));
        for (size_t j = 0; j < n; ++j) {
            std::vector<size_t> indices(k);
            std::iota(indices.begin(), indices.end(), 0);
            std::sort(indices.begin(), indices.end(),
                     [&data, j](size_t i1, size_t i2) { return data[i1][j] < data[i2][j]; });
            
            for (size_t i = 0; i < k; ++i) {
                ranks[indices[i]][j] = i + 1;
            }
        }
        
        // 计算每个处理的秩和
        std::vector<double> rank_sums(k, 0.0);
        for (size_t i = 0; i < k; ++i) {
            for (size_t j = 0; j < n; ++j) {
                rank_sums[i] += ranks[i][j];
            }
        }
        
        // 计算检验统计量
        double Q = 0.0;
        for (size_t i = 0; i < k; ++i) {
            Q += rank_sums[i] * rank_sums[i];
        }
        Q = 12.0 / (n * k * (k + 1)) * Q - 3 * n * (k + 1);
        
        return Q;
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

    // Wilcoxon秩和检验
    template<typename T>
    double wilcoxon_rank_sum(const std::vector<T>& sample1, const std::vector<T>& sample2) {
        if (sample1.empty() || sample2.empty()) {
            throw std::invalid_argument("样本不能为空");
        }

        // 合并样本并计算秩
        std::vector<std::pair<T, int>> combined;
        for (size_t i = 0; i < sample1.size(); ++i) {
            combined.emplace_back(sample1[i], 1);
        }
        for (size_t i = 0; i < sample2.size(); ++i) {
            combined.emplace_back(sample2[i], 2);
        }

        // 排序
        std::sort(combined.begin(), combined.end());

        // 计算秩和
        double rank_sum = 0.0;
        for (size_t i = 0; i < combined.size(); ++i) {
            if (combined[i].second == 1) {
                rank_sum += i + 1;
            }
        }

        // 计算期望值和标准差
        double n1 = static_cast<double>(sample1.size());
        double n2 = static_cast<double>(sample2.size());
        double expected = n1 * (n1 + n2 + 1) / 2.0;
        double stddev = std::sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0);

        // 计算标准化统计量
        return (rank_sum - expected) / stddev;
    }

    // Kruskal-Wallis检验
    template<typename T>
    double kruskal_wallis(const std::vector<std::vector<T>>& samples) {
        if (samples.empty()) {
            throw std::invalid_argument("样本不能为空");
        }

        // 合并所有样本并计算秩
        std::vector<std::pair<T, int>> combined;
        for (size_t i = 0; i < samples.size(); ++i) {
            for (const auto& x : samples[i]) {
                combined.emplace_back(x, i);
            }
        }

        // 排序
        std::sort(combined.begin(), combined.end());

        // 计算每个样本的秩和
        std::vector<double> rank_sums(samples.size(), 0.0);
        std::vector<double> n(samples.size(), 0.0);
        for (size_t i = 0; i < combined.size(); ++i) {
            rank_sums[combined[i].second] += i + 1;
            n[combined[i].second] += 1.0;
        }

        // 计算H统计量
        double N = static_cast<double>(combined.size());
        double H = 0.0;
        for (size_t i = 0; i < samples.size(); ++i) {
            H += rank_sums[i] * rank_sums[i] / n[i];
        }
        H = 12.0 * H / (N * (N + 1)) - 3.0 * (N + 1);

        return H;
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

    // Beta分布
    class BetaDistribution {
    private:
        double alpha_;
        double beta_;
        std::mt19937 rng_;

    public:
        BetaDistribution(double alpha, double beta)
            : alpha_(alpha), beta_(beta), rng_(std::random_device{}()) {
            if (alpha <= 0.0 || beta <= 0.0) {
                throw std::invalid_argument("参数必须为正数");
            }
        }

        double pdf(double x) const {
            if (x < 0.0 || x > 1.0) return 0.0;
            return std::pow(x, alpha_ - 1.0) * std::pow(1.0 - x, beta_ - 1.0) / 
                   special::beta(alpha_, beta_);
        }

        double cdf(double x) const {
            if (x < 0.0) return 0.0;
            if (x > 1.0) return 1.0;
            return special::incomplete_beta(x, alpha_, beta_);
        }

        double sample() {
            std::gamma_distribution<double> gamma1(alpha_, 1.0);
            std::gamma_distribution<double> gamma2(beta_, 1.0);
            double x = gamma1(rng_);
            double y = gamma2(rng_);
            return x / (x + y);
        }
    };

    // Gamma分布
    class GammaDistribution {
    private:
        double k_;      // 形状参数
        double theta_;  // 尺度参数
        std::mt19937 rng_;

    public:
        GammaDistribution(double k, double theta)
            : k_(k), theta_(theta), rng_(std::random_device{}()) {
            if (k <= 0.0 || theta <= 0.0) {
                throw std::invalid_argument("参数必须为正数");
            }
        }

        double pdf(double x) const {
            if (x < 0.0) return 0.0;
            return std::pow(x, k_ - 1.0) * std::exp(-x / theta_) / 
                   (std::tgamma(k_) * std::pow(theta_, k_));
        }

        double cdf(double x) const {
            if (x < 0.0) return 0.0;
            return special::incomplete_gamma(k_, x / theta_) / std::tgamma(k_);
        }

        double sample() {
            std::gamma_distribution<double> dist(k_, theta_);
            return dist(rng_);
        }
    };

    // 威布尔分布
    class WeibullDistribution {
    private:
        double k_;  // 形状参数
        double lambda_;  // 尺度参数
        
    public:
        WeibullDistribution(double k, double lambda) : k_(k), lambda_(lambda) {
            if (k <= 0 || lambda <= 0) {
                throw std::invalid_argument("Invalid parameters");
            }
        }
        
        double pdf(double x) const {
            if (x < 0) return 0.0;
            return (k_ / lambda_) * std::pow(x / lambda_, k_ - 1) * 
                   std::exp(-std::pow(x / lambda_, k_));
        }
        
        double cdf(double x) const {
            if (x < 0) return 0.0;
            return 1.0 - std::exp(-std::pow(x / lambda_, k_));
        }
        
        double sample() const {
            double u = static_cast<double>(rand()) / RAND_MAX;
            return lambda_ * std::pow(-std::log(1.0 - u), 1.0 / k_);
        }
    };
    
    // 对数正态分布
    class LogNormalDistribution {
    private:
        double mu_;  // 位置参数
        double sigma_;  // 尺度参数
        
    public:
        LogNormalDistribution(double mu, double sigma) : mu_(mu), sigma_(sigma) {
            if (sigma <= 0) {
                throw std::invalid_argument("Invalid parameters");
            }
        }
        
        double pdf(double x) const {
            if (x <= 0) return 0.0;
            double z = (std::log(x) - mu_) / sigma_;
            return 1.0 / (x * sigma_ * std::sqrt(2 * M_PI)) * 
                   std::exp(-0.5 * z * z);
        }
        
        double cdf(double x) const {
            if (x <= 0) return 0.0;
            return 0.5 * (1.0 + std::erf((std::log(x) - mu_) / (sigma_ * std::sqrt(2))));
        }
        
        double sample() const {
            double u = static_cast<double>(rand()) / RAND_MAX;
            double v = static_cast<double>(rand()) / RAND_MAX;
            double z = std::sqrt(-2.0 * std::log(u)) * std::cos(2.0 * M_PI * v);
            return std::exp(mu_ + sigma_ * z);
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

    // 拉盖尔多项式 L_n(x)
    double laguerre_polynomial(unsigned int n, double x) {
        if (n == 0) return 1.0;
        if (n == 1) return 1.0 - x;
        
        double L0 = 1.0;
        double L1 = 1.0 - x;
        
        for (unsigned int i = 2; i <= n; ++i) {
            double L2 = ((2.0 * i - 1.0 - x) * L1 - (i - 1.0) * L0) / i;
            L0 = L1;
            L1 = L2;
        }
        
        return L1;
    }

    // 切比雪夫多项式 T_n(x)
    double chebyshev_polynomial(unsigned int n, double x) {
        if (n == 0) return 1.0;
        if (n == 1) return x;
        
        double T0 = 1.0;
        double T1 = x;
        
        for (unsigned int i = 2; i <= n; ++i) {
            double T2 = 2.0 * x * T1 - T0;
            T0 = T1;
            T1 = T2;
        }
        
        return T1;
    }

    // 第二类切比雪夫多项式 U_n(x)
    double chebyshev_polynomial_second_kind(unsigned int n, double x) {
        if (n == 0) return 1.0;
        if (n == 1) return 2.0 * x;
        
        double U0 = 1.0;
        double U1 = 2.0 * x;
        
        for (unsigned int i = 2; i <= n; ++i) {
            double U2 = 2.0 * x * U1 - U0;
            U0 = U1;
            U1 = U2;
        }
        
        return U1;
    }

    // 埃尔米特多项式
    double hermite_polynomial(int n, double x) {
        if (n < 0) {
            throw std::invalid_argument("Invalid polynomial order");
        }
        
        if (n == 0) return 1.0;
        if (n == 1) return 2.0 * x;
        
        double h0 = 1.0;
        double h1 = 2.0 * x;
        double h2;
        
        for (int i = 2; i <= n; ++i) {
            h2 = 2.0 * x * h1 - 2.0 * (i - 1) * h0;
            h0 = h1;
            h1 = h2;
        }
        
        return h1;
    }
    
    // 雅可比多项式
    double jacobi_polynomial(int n, double alpha, double beta, double x) {
        if (n < 0 || alpha <= -1 || beta <= -1) {
            throw std::invalid_argument("Invalid parameters");
        }
        
        if (n == 0) return 1.0;
        if (n == 1) return 0.5 * ((alpha + beta + 2) * x + (alpha - beta));
        
        double p0 = 1.0;
        double p1 = 0.5 * ((alpha + beta + 2) * x + (alpha - beta));
        double p2;
        
        for (int i = 2; i <= n; ++i) {
            double a1 = 2 * i * (i + alpha + beta) * (2 * i + alpha + beta - 2);
            double a2 = (2 * i + alpha + beta - 1) * (alpha * alpha - beta * beta);
            double a3 = (2 * i + alpha + beta - 2) * (2 * i + alpha + beta - 1) * (2 * i + alpha + beta);
            double a4 = 2 * (i + alpha - 1) * (i + beta - 1) * (2 * i + alpha + beta);
            
            p2 = ((a2 + a3 * x) * p1 - a4 * p0) / a1;
            p0 = p1;
            p1 = p2;
        }
        
        return p1;
    }
    
    // 超几何函数
    double hypergeometric_function(double a, double b, double c, double x) {
        if (std::abs(x) >= 1.0) {
            throw std::invalid_argument("Invalid argument");
        }
        
        double result = 1.0;
        double term = 1.0;
        double n = 1.0;
        
        while (std::abs(term) > 1e-10) {
            term *= (a + n - 1) * (b + n - 1) * x / ((c + n - 1) * n);
            result += term;
            n += 1.0;
        }
        
        return result;
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

    // 计算不完全伽马函数
    double incomplete_gamma(double a, double x) {
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
}

} // namespace mathlib

#endif // MATHLIB_HPP 