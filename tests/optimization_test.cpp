#include "mathlib/optimization.hpp"
#include <gtest/gtest.h>
#include <cmath>

using namespace mathlib::optimization;

// 辅助函数：检查向量近似相等
bool vectors_are_close(const std::vector<double>& a, const std::vector<double>& b, double tol = 1e-6) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i] - b[i]) > tol) return false;
    }
    return true;
}

// 测试函数：二次函数 f(x) = x^2 + y^2
double quadratic_objective(const std::vector<double>& x) {
    return x[0] * x[0] + x[1] * x[1];
}

std::vector<double> quadratic_gradient(const std::vector<double>& x) {
    return {2 * x[0], 2 * x[1]};
}

// 测试函数：Rosenbrock函数
double rosenbrock_objective(const std::vector<double>& x) {
    return 100 * std::pow(x[1] - x[0] * x[0], 2) + std::pow(1 - x[0], 2);
}

std::vector<double> rosenbrock_gradient(const std::vector<double>& x) {
    return {
        -400 * x[0] * (x[1] - x[0] * x[0]) - 2 * (1 - x[0]),
        200 * (x[1] - x[0] * x[0])
    };
}

// 邻域函数：随机扰动
std::vector<double> random_neighbor(const std::vector<double>& x) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 0.1);
    
    std::vector<double> result = x;
    for (auto& val : result) {
        val += dist(gen);
    }
    return result;
}

TEST(GradientDescent, QuadraticFunction) {
    std::vector<double> initial_point = {1.0, 1.0};
    auto result = gradient_descent(quadratic_objective, quadratic_gradient, initial_point);
    std::vector<double> expected = {0.0, 0.0};
    EXPECT_TRUE(vectors_are_close(result, expected));
}

TEST(GradientDescent, RosenbrockFunction) {
    std::vector<double> initial_point = {0.0, 0.0};
    auto result = gradient_descent(rosenbrock_objective, rosenbrock_gradient, initial_point);
    std::vector<double> expected = {1.0, 1.0};
    EXPECT_TRUE(vectors_are_close(result, expected, 1e-2));
}

TEST(ConjugateGradient, QuadraticFunction) {
    std::vector<double> initial_point = {1.0, 1.0};
    auto result = conjugate_gradient(quadratic_objective, quadratic_gradient, initial_point);
    std::vector<double> expected = {0.0, 0.0};
    EXPECT_TRUE(vectors_are_close(result, expected));
}

TEST(SimulatedAnnealing, QuadraticFunction) {
    std::vector<double> initial_point = {1.0, 1.0};
    auto result = simulated_annealing(quadratic_objective, random_neighbor, initial_point);
    std::vector<double> expected = {0.0, 0.0};
    EXPECT_TRUE(vectors_are_close(result, expected, 1e-2));
}

TEST(SimplexMethod, LinearProgramming) {
    // 最大化 3x + 4y
    // 约束条件：
    // x + y <= 4
    // 2x + y <= 5
    // x, y >= 0
    std::vector<std::vector<double>> A = {
        {1, 1},
        {2, 1}
    };
    std::vector<double> b = {4, 5};
    std::vector<double> c = {3, 4};
    
    auto result = simplex_method(A, b, c);
    std::vector<double> expected = {1, 3};
    EXPECT_TRUE(vectors_are_close(result, expected));
}

// 边界情况测试
TEST(EdgeCases, EmptyInput) {
    std::vector<double> empty;
    EXPECT_THROW(gradient_descent(quadratic_objective, quadratic_gradient, empty), std::runtime_error);
}

TEST(EdgeCases, UnboundedSolution) {
    std::vector<std::vector<double>> A = {{1, 1}};
    std::vector<double> b = {1};
    std::vector<double> c = {1, 1};
    EXPECT_THROW(simplex_method(A, b, c), std::runtime_error);
}

TEST(EdgeCases, InvalidDimensions) {
    std::vector<std::vector<double>> A = {{1, 1}, {2, 1, 3}};
    std::vector<double> b = {1, 2};
    std::vector<double> c = {1, 1};
    EXPECT_THROW(simplex_method(A, b, c), std::runtime_error);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 