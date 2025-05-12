#include "mathlib/signal.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <complex>

using namespace mathlib::signal;

// 辅助函数：检查复数向量近似相等
bool complex_vectors_are_close(
    const std::vector<std::complex<double>>& a,
    const std::vector<std::complex<double>>& b,
    double tol = 1e-6
) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i].real() - b[i].real()) > tol ||
            std::abs(a[i].imag() - b[i].imag()) > tol) {
            return false;
        }
    }
    return true;
}

// 辅助函数：检查实数向量近似相等
bool vectors_are_close(
    const std::vector<double>& a,
    const std::vector<double>& b,
    double tol = 1e-6
) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i] - b[i]) > tol) return false;
    }
    return true;
}

TEST(FFT, BasicFunctionality) {
    // 测试信号：cos(2πt)
    std::vector<double> input(8);
    for (int i = 0; i < 8; ++i) {
        input[i] = std::cos(2 * M_PI * i / 8);
    }
    
    auto result = fft(input);
    EXPECT_EQ(result.size(), 8);
    
    // 检查对称性
    EXPECT_TRUE(std::abs(result[1].real() - 4.0) < 1e-6);
    EXPECT_TRUE(std::abs(result[7].real() - 4.0) < 1e-6);
}

TEST(FFT, InverseFFT) {
    std::vector<double> input = {1, 2, 3, 4};
    auto fft_result = fft(input);
    auto ifft_result = ifft(fft_result);
    EXPECT_TRUE(vectors_are_close(input, ifft_result));
}

TEST(Filters, FIRFilter) {
    std::vector<double> input = {1, 2, 3, 4, 5};
    std::vector<double> coeffs = {0.5, 0.5};  // 简单移动平均
    
    auto result = fir_filter(input, coeffs);
    std::vector<double> expected = {0.5, 1.5, 2.5, 3.5, 4.5};
    EXPECT_TRUE(vectors_are_close(result, expected));
}

TEST(Filters, IIRFilter) {
    std::vector<double> input = {1, 2, 3, 4, 5};
    std::vector<double> a_coeffs = {1.0, -0.5};  // 一阶IIR
    std::vector<double> b_coeffs = {0.5, 0.5};   // 移动平均
    
    auto result = iir_filter(input, a_coeffs, b_coeffs);
    EXPECT_EQ(result.size(), input.size());
}

TEST(SpectralAnalysis, PowerSpectralDensity) {
    // 生成正弦信号
    std::vector<double> input(1024);
    for (int i = 0; i < 1024; ++i) {
        input[i] = std::sin(2 * M_PI * 10 * i / 1024);
    }
    
    auto psd = power_spectral_density(input);
    EXPECT_EQ(psd.size(), 512);
    
    // 检查峰值位置
    int peak_index = 0;
    double max_value = 0;
    for (size_t i = 0; i < psd.size(); ++i) {
        if (psd[i] > max_value) {
            max_value = psd[i];
            peak_index = i;
        }
    }
    EXPECT_NEAR(peak_index, 10, 1);
}

TEST(SignalProcessing, Convolution) {
    std::vector<double> input = {1, 2, 3, 4};
    std::vector<double> kernel = {1, 1};
    
    auto result = convolve(input, kernel);
    std::vector<double> expected = {1, 3, 5, 7, 4};
    EXPECT_TRUE(vectors_are_close(result, expected));
}

TEST(SignalProcessing, Autocorrelation) {
    std::vector<double> input = {1, 2, 3, 4};
    auto result = autocorrelation(input);
    
    // 自相关在lag=0处最大
    EXPECT_TRUE(result[0] >= result[1]);
    EXPECT_TRUE(result[0] >= result[2]);
    EXPECT_TRUE(result[0] >= result[3]);
}

// 边界情况测试
TEST(EdgeCases, EmptyInput) {
    std::vector<double> empty;
    EXPECT_TRUE(fft(empty).empty());
    EXPECT_TRUE(fir_filter(empty, {1}).empty());
    EXPECT_TRUE(iir_filter(empty, {1}, {1}).empty());
    EXPECT_TRUE(power_spectral_density(empty).empty());
    EXPECT_TRUE(convolve(empty, {1}).empty());
    EXPECT_TRUE(autocorrelation(empty).empty());
}

TEST(EdgeCases, InvalidFilterCoefficients) {
    std::vector<double> input = {1, 2, 3};
    std::vector<double> empty_coeffs;
    EXPECT_TRUE(fir_filter(input, empty_coeffs).empty());
    EXPECT_TRUE(iir_filter(input, empty_coeffs, {1}).empty());
    EXPECT_TRUE(iir_filter(input, {1}, empty_coeffs).empty());
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 