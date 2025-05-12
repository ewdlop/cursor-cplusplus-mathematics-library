#include "mathlib/linear_algebra.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

using namespace mathlib::linear_algebra;

// 辅助函数：检查两个浮点数是否近似相等
template<typename T>
bool is_close(T a, T b, T tolerance = 1e-6) {
    return std::abs(a - b) < tolerance;
}

// 辅助函数：检查两个向量是否近似相等
template<typename T>
bool vectors_are_close(const Vector<T>& a, const Vector<T>& b, T tolerance = 1e-6) {
    if (a.get_size() != b.get_size()) return false;
    for (size_t i = 0; i < a.get_size(); ++i) {
        if (!is_close(a[i], b[i], tolerance)) return false;
    }
    return true;
}

// 辅助函数：检查两个矩阵是否近似相等
template<typename T>
bool matrices_are_close(const Matrix<T>& a, const Matrix<T>& b, T tolerance = 1e-6) {
    if (a.get_rows() != b.get_rows() || a.get_cols() != b.get_cols()) return false;
    for (size_t i = 0; i < a.get_rows(); ++i) {
        for (size_t j = 0; j < a.get_cols(); ++j) {
            if (!is_close(a(i, j), b(i, j), tolerance)) return false;
        }
    }
    return true;
}

// 矩阵构造函数测试
TEST(MatrixTest, Constructor) {
    // 测试空矩阵
    EXPECT_THROW(Matrix<double>(0, 0), std::invalid_argument);
    
    // 测试正常矩阵
    Matrix<double> m(2, 3);
    EXPECT_EQ(m.get_rows(), 2);
    EXPECT_EQ(m.get_cols(), 3);
    
    // 测试从向量构造
    std::vector<std::vector<double>> data = {{1, 2, 3}, {4, 5, 6}};
    Matrix<double> m2(data);
    EXPECT_EQ(m2.get_rows(), 2);
    EXPECT_EQ(m2.get_cols(), 3);
    EXPECT_EQ(m2(0, 0), 1);
    EXPECT_EQ(m2(1, 2), 6);
    
    // 测试不一致的维度
    std::vector<std::vector<double>> invalid_data = {{1, 2}, {3}};
    EXPECT_THROW(Matrix<double>(invalid_data), std::invalid_argument);
}

// 矩阵运算测试
TEST(MatrixTest, Operations) {
    Matrix<double> m1({{1, 2}, {3, 4}});
    Matrix<double> m2({{5, 6}, {7, 8}});
    
    // 测试加法
    Matrix<double> sum = m1 + m2;
    Matrix<double> expected_sum({{6, 8}, {10, 12}});
    EXPECT_TRUE(matrices_are_close(sum, expected_sum));
    
    // 测试乘法
    Matrix<double> product = m1 * m2;
    Matrix<double> expected_product({{19, 22}, {43, 50}});
    EXPECT_TRUE(matrices_are_close(product, expected_product));
    
    // 测试转置
    Matrix<double> transposed = m1.transpose();
    Matrix<double> expected_transposed({{1, 3}, {2, 4}});
    EXPECT_TRUE(matrices_are_close(transposed, expected_transposed));
}

// 向量构造函数测试
TEST(VectorTest, Constructor) {
    // 测试空向量
    EXPECT_THROW(Vector<double>(0), std::invalid_argument);
    
    // 测试正常向量
    Vector<double> v(3);
    EXPECT_EQ(v.get_size(), 3);
    
    // 测试从向量构造
    std::vector<double> data = {1, 2, 3};
    Vector<double> v2(data);
    EXPECT_EQ(v2.get_size(), 3);
    EXPECT_EQ(v2[0], 1);
    EXPECT_EQ(v2[2], 3);
}

// 向量运算测试
TEST(VectorTest, Operations) {
    Vector<double> v1({1, 2, 3});
    Vector<double> v2({4, 5, 6});
    
    // 测试点积
    double dot_product = v1.dot(v2);
    EXPECT_DOUBLE_EQ(dot_product, 32);  // 1*4 + 2*5 + 3*6 = 32
    
    // 测试范数
    double norm = v1.norm();
    EXPECT_DOUBLE_EQ(norm, std::sqrt(14));  // sqrt(1^2 + 2^2 + 3^2)
}

// 线性方程组求解测试
TEST(LinearSystemTest, Solve) {
    // 测试简单系统
    Matrix<double> A({{2, 1}, {1, 3}});
    Vector<double> b({5, 7});
    Vector<double> x = solve_linear_system(A, b);
    
    // 验证解
    Vector<double> expected_x({1.6, 1.8});
    EXPECT_TRUE(vectors_are_close(x, expected_x));
    
    // 验证 Ax = b
    Vector<double> Ax(b.get_size());
    for (size_t i = 0; i < A.get_rows(); ++i) {
        double sum = 0;
        for (size_t j = 0; j < A.get_cols(); ++j) {
            sum += A(i, j) * x[j];
        }
        Ax[i] = sum;
    }
    EXPECT_TRUE(vectors_are_close(Ax, b));
    
    // 测试奇异矩阵
    Matrix<double> singular({{1, 1}, {1, 1}});
    Vector<double> b2({1, 1});
    EXPECT_THROW(solve_linear_system(singular, b2), std::runtime_error);
}

// 边界情况测试
TEST(LinearSystemTest, EdgeCases) {
    // 测试维度不匹配
    Matrix<double> A(2, 2);
    Vector<double> b(3);
    EXPECT_THROW(solve_linear_system(A, b), std::invalid_argument);
    
    // 测试非方阵
    Matrix<double> non_square(2, 3);
    Vector<double> b2(2);
    EXPECT_THROW(solve_linear_system(non_square, b2), std::invalid_argument);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 