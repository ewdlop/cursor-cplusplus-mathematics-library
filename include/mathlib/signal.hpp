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

// 小波变换
std::vector<std::vector<double>> wavelet_transform(
    const std::vector<double>& signal,
    const std::vector<double>& wavelet,
    int levels
) {
    if (signal.empty() || wavelet.empty()) {
        throw std::invalid_argument("输入信号或小波函数不能为空");
    }

    int signal_length = signal.size();
    int wavelet_length = wavelet.size();
    std::vector<std::vector<double>> result(levels);

    // 对每一层进行小波变换
    std::vector<double> current_signal = signal;
    for (int level = 0; level < levels; ++level) {
        int current_length = current_signal.size();
        if (current_length < wavelet_length) break;

        // 计算当前层的系数
        std::vector<double> coefficients(current_length / 2);
        for (int i = 0; i < current_length / 2; ++i) {
            double sum = 0.0;
            for (int j = 0; j < wavelet_length; ++j) {
                if (2 * i + j < current_length) {
                    sum += current_signal[2 * i + j] * wavelet[j];
                }
            }
            coefficients[i] = sum;
        }

        result[level] = coefficients;
        current_signal = coefficients;
    }

    return result;
}

// 希尔伯特变换
std::vector<std::complex<double>> hilbert_transform(const std::vector<double>& signal) {
    if (signal.empty()) {
        throw std::invalid_argument("输入信号不能为空");
    }

    // 计算FFT
    auto fft_result = fft(signal);
    int n = fft_result.size();

    // 应用希尔伯特变换
    for (int i = 0; i < n; ++i) {
        if (i == 0 || i == n/2) {
            fft_result[i] *= 0;
        } else if (i < n/2) {
            fft_result[i] *= 2;
        } else {
            fft_result[i] *= 0;
        }
    }

    // 计算IFFT
    return ifft(fft_result);
}

// 短时傅里叶变换
std::vector<std::vector<std::complex<double>>> short_time_fourier_transform(
    const std::vector<double>& signal,
    int window_size,
    int hop_size
) {
    if (signal.empty()) {
        throw std::invalid_argument("输入信号不能为空");
    }

    // 创建汉宁窗
    std::vector<double> window(window_size);
    for (int i = 0; i < window_size; ++i) {
        window[i] = 0.5 * (1 - std::cos(2 * M_PI * i / (window_size - 1)));
    }

    // 计算STFT
    std::vector<std::vector<std::complex<double>>> stft;
    for (int i = 0; i + window_size <= signal.size(); i += hop_size) {
        std::vector<double> windowed_signal(window_size);
        for (int j = 0; j < window_size; ++j) {
            windowed_signal[j] = signal[i + j] * window[j];
        }
        stft.push_back(fft(windowed_signal));
    }

    return stft;
}

// 倒谱分析
std::vector<double> cepstrum(const std::vector<double>& signal) {
    if (signal.empty()) {
        throw std::invalid_argument("输入信号不能为空");
    }

    // 计算功率谱
    auto psd = power_spectral_density(signal);
    
    // 计算对数功率谱
    std::vector<double> log_psd(psd.size());
    for (size_t i = 0; i < psd.size(); ++i) {
        log_psd[i] = std::log(psd[i] + 1e-10);  // 添加小量避免取对数0
    }

    // 计算倒谱
    auto cepstrum_result = fft(log_psd);
    std::vector<double> result(cepstrum_result.size());
    for (size_t i = 0; i < cepstrum_result.size(); ++i) {
        result[i] = std::abs(cepstrum_result[i]);
    }

    return result;
}

// 自适应滤波器
class AdaptiveFilter {
public:
    AdaptiveFilter(int filter_length, double mu = 0.01)
        : filter_length_(filter_length), mu_(mu) {
        weights_.resize(filter_length, 0.0);
    }

    std::vector<double> filter(const std::vector<double>& input, const std::vector<double>& desired) {
        if (input.size() != desired.size()) {
            throw std::invalid_argument("输入信号和期望信号长度不匹配");
        }

        std::vector<double> output(input.size());
        std::vector<double> error(input.size());

        for (size_t i = filter_length_; i < input.size(); ++i) {
            // 计算输出
            double y = 0.0;
            for (int j = 0; j < filter_length_; ++j) {
                y += weights_[j] * input[i - j];
            }
            output[i] = y;

            // 计算误差
            error[i] = desired[i] - y;

            // 更新权重
            for (int j = 0; j < filter_length_; ++j) {
                weights_[j] += mu_ * error[i] * input[i - j];
            }
        }

        return output;
    }

private:
    int filter_length_;
    double mu_;
    std::vector<double> weights_;
};

} // namespace signal
} // namespace mathlib 