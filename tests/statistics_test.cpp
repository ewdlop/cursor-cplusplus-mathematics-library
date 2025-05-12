#include <gtest/gtest.h>
#include "mathlib.hpp"
#include <vector>
#include <cmath>

using namespace mathlib::statistics;

TEST(StatisticsTest, MeanTest) {
    // 测试空向量
    std::vector<double> empty;
    EXPECT_THROW(mean(empty), std::invalid_argument);

    // 测试单个元素
    std::vector<double> single = {5.0};
    EXPECT_DOUBLE_EQ(mean(single), 5.0);

    // 测试多个元素
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    EXPECT_DOUBLE_EQ(mean(data), 3.0);

    // 测试整数向量
    std::vector<int> int_data = {1, 2, 3, 4, 5};
    EXPECT_DOUBLE_EQ(mean(int_data), 3.0);
}

TEST(StatisticsTest, VarianceTest) {
    // 测试空向量
    std::vector<double> empty;
    EXPECT_THROW(variance(empty), std::invalid_argument);

    // 测试单个元素
    std::vector<double> single = {5.0};
    EXPECT_DOUBLE_EQ(variance(single), 0.0);

    // 测试多个元素
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    EXPECT_DOUBLE_EQ(variance(data), 2.0);

    // 测试所有元素相同的情况
    std::vector<double> same = {3.0, 3.0, 3.0};
    EXPECT_DOUBLE_EQ(variance(same), 0.0);
}

TEST(StatisticsTest, StandardDeviationTest) {
    // 测试空向量
    std::vector<double> empty;
    EXPECT_THROW(standard_deviation(empty), std::invalid_argument);

    // 测试单个元素
    std::vector<double> single = {5.0};
    EXPECT_DOUBLE_EQ(standard_deviation(single), 0.0);

    // 测试多个元素
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    EXPECT_DOUBLE_EQ(standard_deviation(data), std::sqrt(2.0));

    // 测试所有元素相同的情况
    std::vector<double> same = {3.0, 3.0, 3.0};
    EXPECT_DOUBLE_EQ(standard_deviation(same), 0.0);
}

TEST(StatisticsTest, MedianTest) {
    // 测试空向量
    std::vector<double> empty;
    EXPECT_THROW(median(empty), std::invalid_argument);

    // 测试单个元素
    std::vector<double> single = {5.0};
    EXPECT_DOUBLE_EQ(median(single), 5.0);

    // 测试奇数个元素
    std::vector<double> odd = {1.0, 3.0, 2.0, 5.0, 4.0};
    EXPECT_DOUBLE_EQ(median(odd), 3.0);

    // 测试偶数个元素
    std::vector<double> even = {1.0, 3.0, 2.0, 4.0};
    EXPECT_DOUBLE_EQ(median(even), 2.5);
}

TEST(StatisticsTest, ModeTest) {
    // 测试空向量
    std::vector<int> empty;
    EXPECT_THROW(mode(empty), std::invalid_argument);

    // 测试单个元素
    std::vector<int> single = {5};
    EXPECT_EQ(mode(single), 5);

    // 测试多个元素
    std::vector<int> data = {1, 2, 2, 3, 2, 4, 5};
    EXPECT_EQ(mode(data), 2);

    // 测试所有元素相同
    std::vector<int> same = {3, 3, 3};
    EXPECT_EQ(mode(same), 3);
}

TEST(StatisticsTest, QuantileTest) {
    // 测试空向量
    std::vector<double> empty;
    EXPECT_THROW(quantile(empty, 0.5), std::invalid_argument);

    // 测试无效的分位数
    std::vector<double> data = {1.0, 2.0, 3.0};
    EXPECT_THROW(quantile(data, -0.1), std::invalid_argument);
    EXPECT_THROW(quantile(data, 1.1), std::invalid_argument);

    // 测试中位数（0.5分位数）
    std::vector<double> values = {1.0, 2.0, 3.0, 4.0, 5.0};
    EXPECT_DOUBLE_EQ(quantile(values, 0.5), 3.0);

    // 测试四分位数
    EXPECT_DOUBLE_EQ(quantile(values, 0.25), 2.0);
    EXPECT_DOUBLE_EQ(quantile(values, 0.75), 4.0);

    // 测试边界值
    EXPECT_DOUBLE_EQ(quantile(values, 0.0), 1.0);
    EXPECT_DOUBLE_EQ(quantile(values, 1.0), 5.0);
} 