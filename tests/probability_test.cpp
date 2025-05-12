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

TEST(ProbabilityTest, ExponentialDistributionTest) {
    // 测试无效参数
    EXPECT_THROW(ExponentialDistribution(0), std::invalid_argument);
    EXPECT_THROW(ExponentialDistribution(-1), std::invalid_argument);

    // 测试基本指数分布
    ExponentialDistribution exp(2.0);
    
    // 测试PDF
    EXPECT_DOUBLE_EQ(exp.pdf(-1.0), 0.0);
    EXPECT_DOUBLE_EQ(exp.pdf(0.0), 2.0);
    EXPECT_NEAR(exp.pdf(1.0), 0.2707, 0.0001);
    
    // 测试CDF
    EXPECT_DOUBLE_EQ(exp.cdf(-1.0), 0.0);
    EXPECT_DOUBLE_EQ(exp.cdf(0.0), 0.0);
    EXPECT_NEAR(exp.cdf(1.0), 0.8647, 0.0001);

    // 测试随机采样
    double sample = exp.sample();
    EXPECT_GE(sample, 0.0);
}

TEST(ProbabilityTest, ChiSquaredDistributionTest) {
    // 测试无效参数
    EXPECT_THROW(ChiSquaredDistribution(0), std::invalid_argument);
    EXPECT_THROW(ChiSquaredDistribution(-1), std::invalid_argument);

    // 测试基本卡方分布
    ChiSquaredDistribution chi(3);
    
    // 测试PDF
    EXPECT_DOUBLE_EQ(chi.pdf(-1.0), 0.0);
    EXPECT_NEAR(chi.pdf(1.0), 0.2419, 0.0001);
    EXPECT_NEAR(chi.pdf(2.0), 0.2076, 0.0001);
    
    // 测试CDF
    EXPECT_DOUBLE_EQ(chi.cdf(-1.0), 0.0);
    EXPECT_NEAR(chi.cdf(1.0), 0.1987, 0.0001);
    EXPECT_NEAR(chi.cdf(2.0), 0.4276, 0.0001);

    // 测试随机采样
    double sample = chi.sample();
    EXPECT_GE(sample, 0.0);
}

TEST(ProbabilityTest, TDistributionTest) {
    probability::TDistribution dist(10);
    
    // 测试无效参数
    EXPECT_THROW(probability::TDistribution(0), std::invalid_argument);
    EXPECT_THROW(probability::TDistribution(-1), std::invalid_argument);
    
    // 测试PDF
    EXPECT_GT(dist.pdf(0.0), 0.0);
    EXPECT_NEAR(dist.pdf(0.0), 0.389108, 1e-6);
    EXPECT_GT(dist.pdf(1.0), 0.0);
    EXPECT_GT(dist.pdf(-1.0), 0.0);
    
    // 测试CDF
    EXPECT_NEAR(dist.cdf(0.0), 0.5, 1e-10);
    EXPECT_GT(dist.cdf(1.0), 0.5);
    EXPECT_LT(dist.cdf(-1.0), 0.5);
    
    // 测试分位数
    EXPECT_NEAR(probability::TDistribution::quantile(10, 0.5), 0.0, 1e-10);
    EXPECT_GT(probability::TDistribution::quantile(10, 0.75), 0.0);
    EXPECT_LT(probability::TDistribution::quantile(10, 0.25), 0.0);
    
    // 测试随机采样
    double sample = dist.sample();
    EXPECT_TRUE(std::isfinite(sample));
}

TEST(ProbabilityTest, FDistributionTest) {
    probability::FDistribution dist(5, 10);
    
    // 测试无效参数
    EXPECT_THROW(probability::FDistribution(0, 10), std::invalid_argument);
    EXPECT_THROW(probability::FDistribution(5, 0), std::invalid_argument);
    EXPECT_THROW(probability::FDistribution(-1, 10), std::invalid_argument);
    EXPECT_THROW(probability::FDistribution(5, -1), std::invalid_argument);
    
    // 测试PDF
    EXPECT_GT(dist.pdf(1.0), 0.0);
    EXPECT_NEAR(dist.pdf(1.0), 0.559467, 1e-6);
    EXPECT_EQ(dist.pdf(-1.0), 0.0);
    
    // 测试CDF
    EXPECT_GT(dist.cdf(1.0), 0.0);
    EXPECT_LT(dist.cdf(1.0), 1.0);
    EXPECT_EQ(dist.cdf(-1.0), 0.0);
    
    // 测试随机采样
    double sample = dist.sample();
    EXPECT_TRUE(std::isfinite(sample));
    EXPECT_GE(sample, 0.0);
} 