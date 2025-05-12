#include <gtest/gtest.h>
#include "mathlib.hpp"

using namespace mathlib::combinatorics;

TEST(CombinatoricsTest, FactorialTest) {
    // 测试基本阶乘
    EXPECT_EQ(factorial(0), 1);
    EXPECT_EQ(factorial(1), 1);
    EXPECT_EQ(factorial(5), 120);
    EXPECT_EQ(factorial(10), 3628800);

    // 测试边界情况
    EXPECT_THROW(factorial(21), std::overflow_error);
}

TEST(CombinatoricsTest, CombinationTest) {
    // 测试基本组合数
    EXPECT_EQ(combination(5, 0), 1);
    EXPECT_EQ(combination(5, 1), 5);
    EXPECT_EQ(combination(5, 2), 10);
    EXPECT_EQ(combination(5, 3), 10);
    EXPECT_EQ(combination(5, 4), 5);
    EXPECT_EQ(combination(5, 5), 1);

    // 测试边界情况
    EXPECT_THROW(combination(5, 6), std::invalid_argument);
    EXPECT_THROW(combination(3, 4), std::invalid_argument);

    // 测试对称性
    EXPECT_EQ(combination(10, 3), combination(10, 7));
}

TEST(CombinatoricsTest, PermutationTest) {
    // 测试基本排列数
    EXPECT_EQ(permutation(5, 0), 1);
    EXPECT_EQ(permutation(5, 1), 5);
    EXPECT_EQ(permutation(5, 2), 20);
    EXPECT_EQ(permutation(5, 3), 60);
    EXPECT_EQ(permutation(5, 4), 120);
    EXPECT_EQ(permutation(5, 5), 120);

    // 测试边界情况
    EXPECT_THROW(permutation(5, 6), std::invalid_argument);
    EXPECT_THROW(permutation(3, 4), std::invalid_argument);
}

TEST(CombinatoricsTest, StirlingFirstTest) {
    // 测试基本值
    EXPECT_EQ(stirling_first(0, 0), 1);
    EXPECT_EQ(stirling_first(1, 1), 1);
    EXPECT_EQ(stirling_first(2, 1), 1);
    EXPECT_EQ(stirling_first(2, 2), 1);
    EXPECT_EQ(stirling_first(3, 1), 2);
    EXPECT_EQ(stirling_first(3, 2), 3);
    EXPECT_EQ(stirling_first(3, 3), 1);

    // 测试边界情况
    EXPECT_EQ(stirling_first(5, 0), 0);
    EXPECT_EQ(stirling_first(5, 6), 0);
}

TEST(CombinatoricsTest, StirlingSecondTest) {
    // 测试基本值
    EXPECT_EQ(stirling_second(0, 0), 1);
    EXPECT_EQ(stirling_second(1, 1), 1);
    EXPECT_EQ(stirling_second(2, 1), 1);
    EXPECT_EQ(stirling_second(2, 2), 1);
    EXPECT_EQ(stirling_second(3, 1), 1);
    EXPECT_EQ(stirling_second(3, 2), 3);
    EXPECT_EQ(stirling_second(3, 3), 1);

    // 测试边界情况
    EXPECT_EQ(stirling_second(5, 0), 0);
    EXPECT_EQ(stirling_second(5, 6), 0);
}

TEST(CombinatoricsTest, CatalanTest) {
    // 测试基本值
    EXPECT_EQ(catalan(0), 1);
    EXPECT_EQ(catalan(1), 1);
    EXPECT_EQ(catalan(2), 2);
    EXPECT_EQ(catalan(3), 5);
    EXPECT_EQ(catalan(4), 14);
    EXPECT_EQ(catalan(5), 42);
}

TEST(CombinatoricsTest, BellTest) {
    // 测试基本值
    EXPECT_EQ(bell(0), 1);
    EXPECT_EQ(bell(1), 1);
    EXPECT_EQ(bell(2), 2);
    EXPECT_EQ(bell(3), 5);
    EXPECT_EQ(bell(4), 15);
    EXPECT_EQ(bell(5), 52);
}

TEST(CombinatoricsTest, EulerTest) {
    // 测试基本值
    EXPECT_EQ(euler(0, 0), 1);
    EXPECT_EQ(euler(1, 0), 1);
    EXPECT_EQ(euler(2, 0), 1);
    EXPECT_EQ(euler(2, 1), 1);
    EXPECT_EQ(euler(3, 0), 1);
    EXPECT_EQ(euler(3, 1), 4);
    EXPECT_EQ(euler(3, 2), 1);

    // 测试边界情况
    EXPECT_EQ(euler(5, 5), 0);
    EXPECT_EQ(euler(5, 6), 0);
}

TEST(CombinatoricsTest, BernoulliTest) {
    // 测试基本值
    EXPECT_DOUBLE_EQ(bernoulli(0), 1.0);
    EXPECT_DOUBLE_EQ(bernoulli(1), -0.5);
    EXPECT_DOUBLE_EQ(bernoulli(2), 1.0/6.0);
    EXPECT_DOUBLE_EQ(bernoulli(3), 0.0);
    EXPECT_DOUBLE_EQ(bernoulli(4), -1.0/30.0);
    EXPECT_DOUBLE_EQ(bernoulli(5), 0.0);
    EXPECT_DOUBLE_EQ(bernoulli(6), 1.0/42.0);
}

TEST(CombinatoricsTest, EulerMascheroniTest) {
    // 测试欧拉-马歇罗尼常数
    double gamma = euler_mascheroni();
    EXPECT_NEAR(gamma, 0.5772156649, 0.0001);
}

TEST(CombinatoricsTest, HarmonicTest) {
    // 测试调和数
    EXPECT_DOUBLE_EQ(harmonic(0), 0.0);
    EXPECT_DOUBLE_EQ(harmonic(1), 1.0);
    EXPECT_DOUBLE_EQ(harmonic(2), 1.5);
    EXPECT_DOUBLE_EQ(harmonic(3), 11.0/6.0);
    EXPECT_DOUBLE_EQ(harmonic(4), 25.0/12.0);
    EXPECT_DOUBLE_EQ(harmonic(5), 137.0/60.0);
} 