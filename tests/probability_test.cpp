#include <gtest/gtest.h>
#include "mathlib.hpp"
#include <cmath>

using namespace mathlib::probability;

TEST(ProbabilityTest, NormalDistributionTest) {
    // 测试标准正态分布
    NormalDistribution standard_normal;
    
    // 测试PDF
    EXPECT_DOUBLE_EQ(standard_normal.pdf(0.0), 1.0 / std::sqrt(2 * M_PI));
    EXPECT_DOUBLE_EQ(standard_normal.pdf(1.0), std::exp(-0.5) / std::sqrt(2 * M_PI));
    
    // 测试CDF
    EXPECT_DOUBLE_EQ(standard_normal.cdf(0.0), 0.5);
    EXPECT_NEAR(standard_normal.cdf(1.0), 0.8413, 0.0001);
    EXPECT_NEAR(standard_normal.cdf(-1.0), 0.1587, 0.0001);

    // 测试自定义参数的正态分布
    NormalDistribution custom_normal(2.0, 3.0);
    
    // 测试PDF
    EXPECT_DOUBLE_EQ(custom_normal.pdf(2.0), 1.0 / (3.0 * std::sqrt(2 * M_PI)));
    EXPECT_DOUBLE_EQ(custom_normal.pdf(5.0), std::exp(-0.5) / (3.0 * std::sqrt(2 * M_PI)));
    
    // 测试CDF
    EXPECT_DOUBLE_EQ(custom_normal.cdf(2.0), 0.5);
    EXPECT_NEAR(custom_normal.cdf(5.0), 0.8413, 0.0001);
    EXPECT_NEAR(custom_normal.cdf(-1.0), 0.1587, 0.0001);

    // 测试随机采样
    // 由于是随机数，我们只测试生成的值是否在合理范围内
    double sample = custom_normal.sample();
    EXPECT_GE(sample, -10.0);
    EXPECT_LE(sample, 14.0);
}

TEST(ProbabilityTest, BinomialDistributionTest) {
    // 测试无效参数
    EXPECT_THROW(BinomialDistribution(-1, 0.5), std::invalid_argument);
    EXPECT_THROW(BinomialDistribution(10, -0.1), std::invalid_argument);
    EXPECT_THROW(BinomialDistribution(10, 1.1), std::invalid_argument);

    // 测试基本二项分布
    BinomialDistribution bin(10, 0.5);
    
    // 测试PMF
    EXPECT_DOUBLE_EQ(bin.pmf(-1), 0.0);
    EXPECT_DOUBLE_EQ(bin.pmf(11), 0.0);
    EXPECT_NEAR(bin.pmf(5), 0.24609375, 0.0001);
    
    // 测试CDF
    EXPECT_DOUBLE_EQ(bin.cdf(-1), 0.0);
    EXPECT_DOUBLE_EQ(bin.cdf(10), 1.0);
    EXPECT_NEAR(bin.cdf(5), 0.623046875, 0.0001);

    // 测试随机采样
    int sample = bin.sample();
    EXPECT_GE(sample, 0);
    EXPECT_LE(sample, 10);
}

TEST(ProbabilityTest, PoissonDistributionTest) {
    // 测试无效参数
    EXPECT_THROW(PoissonDistribution(0), std::invalid_argument);
    EXPECT_THROW(PoissonDistribution(-1), std::invalid_argument);

    // 测试基本泊松分布
    PoissonDistribution pois(2.0);
    
    // 测试PMF
    EXPECT_DOUBLE_EQ(pois.pmf(-1), 0.0);
    EXPECT_NEAR(pois.pmf(0), 0.1353, 0.0001);
    EXPECT_NEAR(pois.pmf(1), 0.2707, 0.0001);
    EXPECT_NEAR(pois.pmf(2), 0.2707, 0.0001);
    
    // 测试CDF
    EXPECT_DOUBLE_EQ(pois.cdf(-1), 0.0);
    EXPECT_NEAR(pois.cdf(0), 0.1353, 0.0001);
    EXPECT_NEAR(pois.cdf(1), 0.4060, 0.0001);
    EXPECT_NEAR(pois.cdf(2), 0.6767, 0.0001);

    // 测试随机采样
    int sample = pois.sample();
    EXPECT_GE(sample, 0);
} 