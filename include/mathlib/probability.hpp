#pragma once

#include <vector>
#include <cmath>
#include <stdexcept>
#include <random>

namespace mathlib {
namespace probability {

// 概率分布函数
template<typename T>
T normal_pdf(T x, T mu = 0, T sigma = 1) {
    if (sigma <= 0) {
        throw std::invalid_argument("Standard deviation must be positive");
    }
    T z = (x - mu) / sigma;
    return std::exp(-0.5 * z * z) / (sigma * std::sqrt(2 * M_PI));
}

template<typename T>
T normal_cdf(T x, T mu = 0, T sigma = 1) {
    if (sigma <= 0) {
        throw std::invalid_argument("Standard deviation must be positive");
    }
    return 0.5 * (1 + std::erf((x - mu) / (sigma * std::sqrt(2))));
}

template<typename T>
T binomial_pmf(size_t k, size_t n, T p) {
    if (p < 0 || p > 1) {
        throw std::invalid_argument("Probability must be between 0 and 1");
    }
    if (k > n) {
        throw std::invalid_argument("k must be less than or equal to n");
    }
    
    T q = 1 - p;
    T result = 1;
    
    // 计算组合数
    for (size_t i = 0; i < k; ++i) {
        result *= (n - i);
        result /= (i + 1);
    }
    
    return result * std::pow(p, k) * std::pow(q, n - k);
}

template<typename T>
T poisson_pmf(size_t k, T lambda) {
    if (lambda <= 0) {
        throw std::invalid_argument("Lambda must be positive");
    }
    return std::exp(-lambda) * std::pow(lambda, k) / std::tgamma(k + 1);
}

// 随机数生成
template<typename T>
class RandomGenerator {
private:
    std::mt19937 rng;
    std::uniform_real_distribution<T> uniform_dist;
    std::normal_distribution<T> normal_dist;

public:
    RandomGenerator() : rng(std::random_device{}()) {}
    
    T uniform(T min = 0, T max = 1) {
        if (min >= max) {
            throw std::invalid_argument("min must be less than max");
        }
        return min + (max - min) * uniform_dist(rng);
    }
    
    T normal(T mu = 0, T sigma = 1) {
        if (sigma <= 0) {
            throw std::invalid_argument("Standard deviation must be positive");
        }
        return mu + sigma * normal_dist(rng);
    }
    
    std::vector<T> normal_sample(size_t n, T mu = 0, T sigma = 1) {
        std::vector<T> result(n);
        for (size_t i = 0; i < n; ++i) {
            result[i] = normal(mu, sigma);
        }
        return result;
    }
};

// 概率论工具函数
template<typename T>
T entropy(const std::vector<T>& probabilities) {
    T sum = 0;
    for (T p : probabilities) {
        if (p < 0 || p > 1) {
            throw std::invalid_argument("Probabilities must be between 0 and 1");
        }
        if (p > 0) {
            sum -= p * std::log2(p);
        }
    }
    return sum;
}

template<typename T>
T mutual_information(const std::vector<std::vector<T>>& joint_prob,
                    const std::vector<T>& marginal_x,
                    const std::vector<T>& marginal_y) {
    if (joint_prob.empty() || joint_prob[0].empty()) {
        throw std::invalid_argument("Empty probability matrix");
    }
    
    T mi = 0;
    for (size_t i = 0; i < joint_prob.size(); ++i) {
        for (size_t j = 0; j < joint_prob[0].size(); ++j) {
            T p = joint_prob[i][j];
            if (p > 0) {
                mi += p * std::log2(p / (marginal_x[i] * marginal_y[j]));
            }
        }
    }
    return mi;
}

// 马尔可夫链
template<typename T>
class MarkovChain {
private:
    std::vector<std::vector<T>> transition_matrix;
    size_t n_states;

public:
    MarkovChain(const std::vector<std::vector<T>>& matrix) : transition_matrix(matrix) {
        n_states = matrix.size();
        if (n_states == 0 || matrix[0].size() != n_states) {
            throw std::invalid_argument("Invalid transition matrix dimensions");
        }
        
        // 验证转移矩阵的有效性
        for (const auto& row : matrix) {
            T sum = 0;
            for (T p : row) {
                if (p < 0 || p > 1) {
                    throw std::invalid_argument("Probabilities must be between 0 and 1");
                }
                sum += p;
            }
            if (std::abs(sum - 1) > 1e-10) {
                throw std::invalid_argument("Row probabilities must sum to 1");
            }
        }
    }
    
    std::vector<T> stationary_distribution(size_t max_iterations = 1000) {
        std::vector<T> pi(n_states, 1.0 / n_states);
        std::vector<T> new_pi(n_states);
        
        for (size_t iter = 0; iter < max_iterations; ++iter) {
            for (size_t i = 0; i < n_states; ++i) {
                new_pi[i] = 0;
                for (size_t j = 0; j < n_states; ++j) {
                    new_pi[i] += pi[j] * transition_matrix[j][i];
                }
            }
            
            // 检查收敛性
            T max_diff = 0;
            for (size_t i = 0; i < n_states; ++i) {
                max_diff = std::max(max_diff, std::abs(new_pi[i] - pi[i]));
            }
            if (max_diff < 1e-10) {
                return new_pi;
            }
            
            pi = new_pi;
        }
        
        throw std::runtime_error("Failed to converge to stationary distribution");
    }
};

} // namespace probability
} // namespace mathlib 