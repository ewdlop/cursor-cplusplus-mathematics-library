#include <gtest/gtest.h>
#include "../include/mathlib.hpp"
#include <vector>
#include <cmath>

using namespace mathlib;

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

TEST(CombinatoricsTest, EulerPolynomialTest) {
    // 测试基本情况
    EXPECT_NEAR(combinatorics::euler_polynomial(0, 0.5), 1.0, 1e-10);
    EXPECT_NEAR(combinatorics::euler_polynomial(1, 0.5), 0.0, 1e-10);
    EXPECT_NEAR(combinatorics::euler_polynomial(2, 0.5), -0.25, 1e-10);
    
    // 测试对称性
    double x = 0.3;
    EXPECT_NEAR(combinatorics::euler_polynomial(3, x), 
                -combinatorics::euler_polynomial(3, 1.0 - x), 1e-10);
}

TEST(CombinatoricsTest, BernoulliPolynomialTest) {
    // 测试基本情况
    EXPECT_NEAR(combinatorics::bernoulli_polynomial(0, 0.5), 1.0, 1e-10);
    EXPECT_NEAR(combinatorics::bernoulli_polynomial(1, 0.5), 0.0, 1e-10);
    EXPECT_NEAR(combinatorics::bernoulli_polynomial(2, 0.5), -0.083333, 1e-6);
    
    // 测试递推关系
    double x = 0.3;
    double b2 = combinatorics::bernoulli_polynomial(2, x);
    double b3 = combinatorics::bernoulli_polynomial(3, x);
    double b4 = combinatorics::bernoulli_polynomial(4, x);
    EXPECT_NEAR(b4, x * b3 - 0.5 * b2, 1e-10);
}

TEST(CombinatoricsTest, GeneralizedHarmonicTest) {
    // 测试基本情况
    EXPECT_NEAR(combinatorics::generalized_harmonic(1, 1.0), 1.0, 1e-10);
    EXPECT_NEAR(combinatorics::generalized_harmonic(2, 1.0), 1.5, 1e-10);
    EXPECT_NEAR(combinatorics::generalized_harmonic(3, 2.0), 1.361111, 1e-6);
    
    // 测试无效参数
    EXPECT_THROW(combinatorics::generalized_harmonic(5, 0.0), std::invalid_argument);
    EXPECT_THROW(combinatorics::generalized_harmonic(5, -1.0), std::invalid_argument);
}

TEST(CombinatoricsTest, PolylogarithmTest) {
    // 测试基本情况
    EXPECT_NEAR(combinatorics::polylogarithm(1.0, 0.5), 0.693147, 1e-6);
    EXPECT_NEAR(combinatorics::polylogarithm(2.0, 0.5), 0.582241, 1e-6);
    EXPECT_NEAR(combinatorics::polylogarithm(3.0, 0.5), 0.537213, 1e-6);
    
    // 测试无效参数
    EXPECT_THROW(combinatorics::polylogarithm(0.0, 0.5), std::invalid_argument);
    EXPECT_THROW(combinatorics::polylogarithm(-1.0, 0.5), std::invalid_argument);
    EXPECT_THROW(combinatorics::polylogarithm(1.0, 1.0), std::invalid_argument);
    EXPECT_THROW(combinatorics::polylogarithm(1.0, -1.0), std::invalid_argument);
}

TEST(LaguerrePolynomialTest, BasicCases) {
    // 测试L_0(x) = 1
    EXPECT_DOUBLE_EQ(mathlib::combinatorics::laguerre_polynomial(0, 1.0), 1.0);
    
    // 测试L_1(x) = 1 - x
    EXPECT_DOUBLE_EQ(mathlib::combinatorics::laguerre_polynomial(1, 1.0), 0.0);
    
    // 测试L_2(x) = (x^2 - 4x + 2)/2
    EXPECT_DOUBLE_EQ(mathlib::combinatorics::laguerre_polynomial(2, 1.0), -0.5);
}

TEST(LaguerrePolynomialTest, InvalidParameters) {
    EXPECT_THROW(mathlib::combinatorics::laguerre_polynomial(-1, 1.0), std::invalid_argument);
}

TEST(ChebyshevPolynomialTest, FirstKind) {
    // 测试T_0(x) = 1
    EXPECT_DOUBLE_EQ(mathlib::combinatorics::chebyshev_polynomial(0, 0.5, true), 1.0);
    
    // 测试T_1(x) = x
    EXPECT_DOUBLE_EQ(mathlib::combinatorics::chebyshev_polynomial(1, 0.5, true), 0.5);
    
    // 测试T_2(x) = 2x^2 - 1
    EXPECT_DOUBLE_EQ(mathlib::combinatorics::chebyshev_polynomial(2, 0.5, true), -0.5);
}

TEST(ChebyshevPolynomialTest, SecondKind) {
    // 测试U_0(x) = 1
    EXPECT_DOUBLE_EQ(mathlib::combinatorics::chebyshev_polynomial(0, 0.5, false), 1.0);
    
    // 测试U_1(x) = 2x
    EXPECT_DOUBLE_EQ(mathlib::combinatorics::chebyshev_polynomial(1, 0.5, false), 1.0);
    
    // 测试U_2(x) = 4x^2 - 1
    EXPECT_DOUBLE_EQ(mathlib::combinatorics::chebyshev_polynomial(2, 0.5, false), 0.0);
}

TEST(ChebyshevPolynomialTest, InvalidParameters) {
    EXPECT_THROW(mathlib::combinatorics::chebyshev_polynomial(-1, 0.5, true), std::invalid_argument);
    EXPECT_THROW(mathlib::combinatorics::chebyshev_polynomial(-1, 0.5, false), std::invalid_argument);
} 