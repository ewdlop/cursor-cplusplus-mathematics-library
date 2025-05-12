#pragma once

#include <vector>
#include <cmath>
#include <stdexcept>
#include <numeric>

namespace mathlib {
namespace combinatorics {

// 基本组合函数
template<typename T>
T factorial(size_t n) {
    if (n > 170) { // 对于double类型，170!是最大值
        throw std::overflow_error("Factorial too large");
    }
    T result = 1;
    for (size_t i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}

template<typename T>
T combination(size_t n, size_t k) {
    if (k > n) {
        throw std::invalid_argument("k must be less than or equal to n");
    }
    if (k > n/2) {
        k = n - k; // 利用对称性
    }
    
    T result = 1;
    for (size_t i = 0; i < k; ++i) {
        result *= (n - i);
        result /= (i + 1);
    }
    return result;
}

template<typename T>
T permutation(size_t n, size_t k) {
    if (k > n) {
        throw std::invalid_argument("k must be less than or equal to n");
    }
    
    T result = 1;
    for (size_t i = 0; i < k; ++i) {
        result *= (n - i);
    }
    return result;
}

// 多项式函数
template<typename T>
T laguerre_polynomial(size_t n, T x) {
    if (n == 0) return 1;
    if (n == 1) return 1 - x;
    
    T L0 = 1;
    T L1 = 1 - x;
    T L2;
    
    for (size_t i = 2; i <= n; ++i) {
        L2 = ((2*i - 1 - x) * L1 - (i - 1) * L0) / i;
        L0 = L1;
        L1 = L2;
    }
    
    return L1;
}

template<typename T>
T chebyshev_polynomial(size_t n, T x) {
    if (n == 0) return 1;
    if (n == 1) return x;
    
    T T0 = 1;
    T T1 = x;
    T T2;
    
    for (size_t i = 2; i <= n; ++i) {
        T2 = 2 * x * T1 - T0;
        T0 = T1;
        T1 = T2;
    }
    
    return T1;
}

// 生成函数
template<typename T>
std::vector<T> generate_combinations(size_t n, size_t k) {
    if (k > n) {
        throw std::invalid_argument("k must be less than or equal to n");
    }
    
    std::vector<T> result;
    std::vector<bool> mask(n, false);
    std::fill(mask.begin(), mask.begin() + k, true);
    
    do {
        T combination = 0;
        for (size_t i = 0; i < n; ++i) {
            if (mask[i]) {
                combination |= (1 << i);
            }
        }
        result.push_back(combination);
    } while (std::prev_permutation(mask.begin(), mask.end()));
    
    return result;
}

template<typename T>
std::vector<std::vector<T>> generate_permutations(const std::vector<T>& elements) {
    std::vector<std::vector<T>> result;
    std::vector<T> current = elements;
    
    do {
        result.push_back(current);
    } while (std::next_permutation(current.begin(), current.end()));
    
    return result;
}

// 组合优化
template<typename T>
T binomial_coefficient(size_t n, size_t k) {
    if (k > n) return 0;
    if (k > n - k) k = n - k;
    
    T result = 1;
    for (size_t i = 0; i < k; ++i) {
        result *= (n - i);
        result /= (i + 1);
    }
    return result;
}

template<typename T>
T multinomial_coefficient(const std::vector<size_t>& k) {
    size_t n = std::accumulate(k.begin(), k.end(), size_t(0));
    T result = factorial<T>(n);
    
    for (size_t ki : k) {
        result /= factorial<T>(ki);
    }
    
    return result;
}

// 组合恒等式
template<typename T>
T stirling_number_second_kind(size_t n, size_t k) {
    if (k > n) return 0;
    if (k == 0) return (n == 0) ? 1 : 0;
    if (k == 1 || k == n) return 1;
    
    return k * stirling_number_second_kind<T>(n-1, k) + 
           stirling_number_second_kind<T>(n-1, k-1);
}

template<typename T>
T bell_number(size_t n) {
    if (n == 0) return 1;
    
    T result = 0;
    for (size_t k = 0; k <= n; ++k) {
        result += stirling_number_second_kind<T>(n, k);
    }
    return result;
}

} // namespace combinatorics
} // namespace mathlib 