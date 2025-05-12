#include "mathlib/numerical.hpp"
#include <gtest/gtest.h>
#include <cmath>

using namespace mathlib::numerical;

// 辅助函数：检查浮点数近似相等
bool is_close(double a, double b, double tol = 1e-6) {
    return std::abs(a - b) < tol;
}

// 测试函数
double test_function(double x) {
    return x * x;  // f(x) = x^2
}

double test_function_derivative(double x) {
    return 2 * x;  // f'(x) = 2x
}

double test_ode(double t, double y) {
    return -y;  // dy/dt = -y
}

TEST(NumericalIntegration, TrapezoidalRule) {
    auto f = [](double x) { return x * x; };
    double result = trapezoidal_rule(f, 0.0, 1.0, 100);
    EXPECT_TRUE(is_close(result, 1.0/3.0));
}

TEST(NumericalIntegration, SimpsonRule) {
    auto f = [](double x) { return x * x; };
    double result = simpson_rule(f, 0.0, 1.0, 100);
    EXPECT_TRUE(is_close(result, 1.0/3.0));
}

TEST(NumericalIntegration, GaussQuadrature) {
    auto f = [](double x) { return x * x; };
    double result = gauss_quadrature(f, 0.0, 1.0, 5);
    EXPECT_TRUE(is_close(result, 1.0/3.0));
}

TEST(NumericalDifferentiation, CentralDifference) {
    auto f = [](double x) { return x * x; };
    double result = central_difference(f, 1.0);
    EXPECT_TRUE(is_close(result, 2.0));
}

TEST(NumericalDifferentiation, ForwardDifference) {
    auto f = [](double x) { return x * x; };
    double result = forward_difference(f, 1.0);
    EXPECT_TRUE(is_close(result, 2.0, 1e-4));
}

TEST(EquationSolving, NewtonMethod) {
    auto f = [](double x) { return x * x - 2; };
    auto df = [](double x) { return 2 * x; };
    double result = newton_method(f, df, 1.0);
    EXPECT_TRUE(is_close(result, std::sqrt(2.0)));
}

TEST(EquationSolving, BisectionMethod) {
    auto f = [](double x) { return x * x - 2; };
    double result = bisection_method(f, 0.0, 2.0);
    EXPECT_TRUE(is_close(result, std::sqrt(2.0)));
}

TEST(Interpolation, LagrangeInterpolation) {
    std::vector<double> x = {0, 1, 2, 3};
    std::vector<double> y = {0, 1, 4, 9};
    double result = lagrange_interpolation(x, y, 1.5);
    EXPECT_TRUE(is_close(result, 2.25));
}

TEST(ODESolving, EulerMethod) {
    auto f = [](double t, double y) { return -y; };
    std::vector<double> result = euler_method(f, 0.0, 1.0, 0.1, 10);
    EXPECT_TRUE(is_close(result[10], std::exp(-1.0), 1e-2));
}

TEST(ODESolving, RungeKutta4) {
    auto f = [](double t, double y) { return -y; };
    std::vector<double> result = runge_kutta_4(f, 0.0, 1.0, 0.1, 10);
    EXPECT_TRUE(is_close(result[10], std::exp(-1.0), 1e-4));
}

// 边界情况测试
TEST(EdgeCases, EmptyInput) {
    std::vector<double> x = {0, 1};
    std::vector<double> y = {0, 1};
    EXPECT_THROW(lagrange_interpolation(x, std::vector<double>(), 0.5), std::invalid_argument);
}

TEST(EdgeCases, InvalidIntervals) {
    auto f = [](double x) { return x * x; };
    EXPECT_THROW(trapezoidal_rule(f, 0.0, 1.0, 0), std::invalid_argument);
    EXPECT_THROW(simpson_rule(f, 0.0, 1.0, 1), std::invalid_argument);
}

TEST(EdgeCases, NonConvergence) {
    auto f = [](double x) { return 1.0; };
    auto df = [](double x) { return 0.0; };
    EXPECT_THROW(newton_method(f, df, 1.0), std::runtime_error);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 