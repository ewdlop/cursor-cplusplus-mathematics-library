#pragma once

#include <vector>
#include <complex>
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace mathlib {
namespace signal {

// FFT实现
template<typename T>
std::vector<std::complex<T>> fft(const std::vector<T>& input) {
    size_t n = input.size();
    if (n == 0) return {};
    
    // 确保输入长度是2的幂
    size_t m = 1;
    while (m < n) m *= 2;
    std::vector<std::complex<T>> x(m);
    for (size_t i = 0; i < n; ++i) {
        x[i] = std::complex<T>(input[i], 0);
    }
    
    // 位反转排序
    for (size_t i = 0; i < m; ++i) {
        size_t j = 0;
        for (size_t k = 0; k < std::log2(m); ++k) {
            j = (j << 1) | (i & 1);
            i >>= 1;
        }
        if (j > i) {
            std::swap(x[i], x[j]);
        }
    }
    
    // FFT计算
    for (size_t s = 1; s <= std::log2(m); ++s) {
        size_t m1 = 1 << s;
        size_t m2 = m1 >> 1;
        std::complex<T> w(1, 0);
        std::complex<T> wm = std::exp(std::complex<T>(0, -2 * M_PI / m1));
        
        for (size_t j = 0; j < m2; ++j) {
            for (size_t k = j; k < m; k += m1) {
                std::complex<T> t = w * x[k + m2];
                std::complex<T> u = x[k];
                x[k] = u + t;
                x[k + m2] = u - t;
            }
            w *= wm;
        }
    }
    
    return x;
}

// 逆FFT
template<typename T>
std::vector<T> ifft(const std::vector<std::complex<T>>& input) {
    size_t n = input.size();
    if (n == 0) return {};
    
    // 共轭
    std::vector<std::complex<T>> x = input;
    for (auto& val : x) {
        val = std::conj(val);
    }
    
    // 正向FFT
    x = fft(std::vector<T>(n));
    
    // 共轭并归一化
    std::vector<T> result(n);
    for (size_t i = 0; i < n; ++i) {
        result[i] = std::real(std::conj(x[i])) / n;
    }
    
    return result;
}

// FIR滤波器
template<typename T>
std::vector<T> fir_filter(const std::vector<T>& input, const std::vector<T>& coefficients) {
    size_t n = input.size();
    size_t m = coefficients.size();
    if (n == 0 || m == 0) return {};
    
    std::vector<T> output(n);
    for (size_t i = 0; i < n; ++i) {
        T sum = 0;
        for (size_t j = 0; j < m; ++j) {
            if (i >= j) {
                sum += coefficients[j] * input[i - j];
            }
        }
        output[i] = sum;
    }
    
    return output;
}

// IIR滤波器
template<typename T>
std::vector<T> iir_filter(
    const std::vector<T>& input,
    const std::vector<T>& a_coeffs,
    const std::vector<T>& b_coeffs
) {
    size_t n = input.size();
    size_t na = a_coeffs.size();
    size_t nb = b_coeffs.size();
    if (n == 0 || na == 0 || nb == 0) return {};
    
    std::vector<T> output(n);
    std::vector<T> x_hist(nb, 0);
    std::vector<T> y_hist(na, 0);
    
    for (size_t i = 0; i < n; ++i) {
        // 更新输入历史
        for (size_t j = nb - 1; j > 0; --j) {
            x_hist[j] = x_hist[j - 1];
        }
        x_hist[0] = input[i];
        
        // 计算输出
        T sum = 0;
        for (size_t j = 0; j < nb; ++j) {
            sum += b_coeffs[j] * x_hist[j];
        }
        for (size_t j = 1; j < na; ++j) {
            sum -= a_coeffs[j] * y_hist[j - 1];
        }
        
        output[i] = sum / a_coeffs[0];
        
        // 更新输出历史
        for (size_t j = na - 1; j > 0; --j) {
            y_hist[j] = y_hist[j - 1];
        }
        y_hist[0] = output[i];
    }
    
    return output;
}

// 功率谱密度
template<typename T>
std::vector<T> power_spectral_density(const std::vector<T>& input, size_t window_size = 1024) {
    size_t n = input.size();
    if (n == 0) return {};
    
    // 应用汉宁窗
    std::vector<T> window(window_size);
    for (size_t i = 0; i < window_size; ++i) {
        window[i] = 0.5 * (1 - std::cos(2 * M_PI * i / (window_size - 1)));
    }
    
    // 计算FFT
    std::vector<std::complex<T>> fft_result = fft(input);
    
    // 计算功率谱
    std::vector<T> psd(fft_result.size() / 2);
    for (size_t i = 0; i < psd.size(); ++i) {
        psd[i] = std::norm(fft_result[i]) / window_size;
    }
    
    return psd;
}

// 卷积
template<typename T>
std::vector<T> convolve(const std::vector<T>& input, const std::vector<T>& kernel) {
    size_t n = input.size();
    size_t m = kernel.size();
    if (n == 0 || m == 0) return {};
    
    std::vector<T> output(n + m - 1, 0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            output[i + j] += input[i] * kernel[j];
        }
    }
    
    return output;
}

// 自相关
template<typename T>
std::vector<T> autocorrelation(const std::vector<T>& input) {
    size_t n = input.size();
    if (n == 0) return {};
    
    std::vector<T> result(n);
    for (size_t lag = 0; lag < n; ++lag) {
        T sum = 0;
        for (size_t i = 0; i < n - lag; ++i) {
            sum += input[i] * input[i + lag];
        }
        result[lag] = sum / (n - lag);
    }
    
    return result;
}

} // namespace signal
} // namespace mathlib 