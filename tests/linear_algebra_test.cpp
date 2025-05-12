#include "mathlib/linear_algebra.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <complex>
#include <random>

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

// 矩阵高级运算测试
TEST(MatrixTest, AdvancedOperations) {
    Matrix<double> m({{4, 1}, {1, 3}});
    
    // 测试行列式
    EXPECT_DOUBLE_EQ(m.determinant(), 11);
    
    // 测试迹
    EXPECT_DOUBLE_EQ(m.trace(), 7);
    
    // 测试LU分解
    auto [L, U] = m.lu_decomposition();
    Matrix<double> product = L * U;
    EXPECT_TRUE(matrices_are_close(product, m));
    
    // 测试特征值和特征向量
    auto [eigenvalues, eigenvectors] = m.eigenvalues_eigenvectors();
    EXPECT_EQ(eigenvalues.size(), 2);
    EXPECT_EQ(eigenvectors.size(), 2);
    
    // 验证特征值和特征向量
    for (size_t i = 0; i < eigenvalues.size(); ++i) {
        Vector<double> Av = m * eigenvectors[i];
        Vector<double> lambda_v = eigenvectors[i] * eigenvalues[i].real();
        EXPECT_TRUE(vectors_are_close(Av, lambda_v));
    }
}

// 向量高级运算测试
TEST(VectorTest, AdvancedOperations) {
    Vector<double> v1({1, 0, 0});
    Vector<double> v2({0, 1, 0});
    
    // 测试叉积
    Vector<double> cross_product = v1.cross(v2);
    Vector<double> expected_cross({0, 0, 1});
    EXPECT_TRUE(vectors_are_close(cross_product, expected_cross));
    
    // 测试投影
    Vector<double> v3({1, 1, 0});
    Vector<double> projection = v3.project(v1);
    Vector<double> expected_projection({1, 0, 0});
    EXPECT_TRUE(vectors_are_close(projection, expected_projection));
    
    // 测试夹角
    double angle = v1.angle(v2);
    EXPECT_DOUBLE_EQ(angle, M_PI / 2);
}

// 矩阵条件数测试
TEST(MatrixTest, ConditionNumber) {
    // 测试良态矩阵
    Matrix<double> well_conditioned({{1, 0}, {0, 1}});
    EXPECT_DOUBLE_EQ(condition_number(well_conditioned), 1.0);
    
    // 测试病态矩阵
    Matrix<double> ill_conditioned({{1, 1}, {1, 1.0001}});
    double cond = condition_number(ill_conditioned);
    EXPECT_GT(cond, 1000); // 条件数应该很大
}

// 最小二乘解测试
TEST(LinearSystemTest, LeastSquares) {
    // 超定系统
    Matrix<double> A({{1, 1}, {1, 2}, {1, 3}});
    Vector<double> b({1, 2, 2});
    
    Vector<double> x = least_squares(A, b);
    
    // 验证残差
    Vector<double> residual = A * x - b;
    EXPECT_LT(residual.norm(), 1e-6);
}

// 边界情况测试
TEST(MatrixTest, EdgeCases) {
    // 测试非方阵的行列式
    Matrix<double> non_square(2, 3);
    EXPECT_THROW(non_square.determinant(), std::invalid_argument);
    
    // 测试非方阵的迹
    EXPECT_THROW(non_square.trace(), std::invalid_argument);
    
    // 测试非方阵的LU分解
    EXPECT_THROW(non_square.lu_decomposition(), std::invalid_argument);
    
    // 测试非方阵的特征值计算
    EXPECT_THROW(non_square.eigenvalues_eigenvectors(), std::invalid_argument);
}

TEST(VectorTest, EdgeCases) {
    // 测试非3D向量的叉积
    Vector<double> v1(2);
    Vector<double> v2(2);
    EXPECT_THROW(v1.cross(v2), std::invalid_argument);
    
    // 测试不同维度向量的投影
    Vector<double> v3(3);
    EXPECT_THROW(v1.project(v3), std::invalid_argument);
    
    // 测试不同维度向量的夹角
    EXPECT_THROW(v1.angle(v3), std::invalid_argument);
}

// 生成随机矩阵
template<typename T>
Matrix<T> generate_random_matrix(size_t rows, size_t cols, T min = -1.0, T max = 1.0) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(min, max);
    
    Matrix<T> m(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            m(i, j) = dis(gen);
        }
    }
    return m;
}

// 生成随机向量
template<typename T>
Vector<T> generate_random_vector(size_t size, T min = -1.0, T max = 1.0) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dis(min, max);
    
    Vector<T> v(size);
    for (size_t i = 0; i < size; ++i) {
        v[i] = dis(gen);
    }
    return v;
}

// 矩阵运算的数值稳定性测试
TEST(MatrixTest, NumericalStability) {
    // 测试大矩阵乘法
    Matrix<double> A = generate_random_matrix<double>(100, 100);
    Matrix<double> B = generate_random_matrix<double>(100, 100);
    Matrix<double> C = A * B;
    
    // 验证乘法结合律
    Matrix<double> D = generate_random_matrix<double>(100, 100);
    Matrix<double> left = (A * B) * D;
    Matrix<double> right = A * (B * D);
    EXPECT_TRUE(matrices_are_close(left, right, 1e-10));
    
    // 测试转置的幂等性
    Matrix<double> A_transpose = A.transpose();
    Matrix<double> A_transpose_transpose = A_transpose.transpose();
    EXPECT_TRUE(matrices_are_close(A, A_transpose_transpose));
}

// 行列式计算测试
TEST(MatrixTest, DeterminantProperties) {
    // 测试单位矩阵的行列式
    Matrix<double> I({{1, 0}, {0, 1}});
    EXPECT_DOUBLE_EQ(I.determinant(), 1.0);
    
    // 测试行列式的乘法性质
    Matrix<double> A = generate_random_matrix<double>(3, 3);
    Matrix<double> B = generate_random_matrix<double>(3, 3);
    double det_A = A.determinant();
    double det_B = B.determinant();
    double det_AB = (A * B).determinant();
    EXPECT_TRUE(is_close(det_AB, det_A * det_B, 1e-10));
    
    // 测试奇异矩阵的行列式
    Matrix<double> singular({{1, 1}, {1, 1}});
    EXPECT_DOUBLE_EQ(singular.determinant(), 0.0);
}

// LU分解测试
TEST(MatrixTest, LUDecompositionProperties) {
    Matrix<double> A = generate_random_matrix<double>(5, 5);
    auto [L, U] = A.lu_decomposition();
    
    // 验证L是下三角矩阵
    for (size_t i = 0; i < L.get_rows(); ++i) {
        for (size_t j = i + 1; j < L.get_cols(); ++j) {
            EXPECT_DOUBLE_EQ(L(i, j), 0.0);
        }
    }
    
    // 验证U是上三角矩阵
    for (size_t i = 0; i < U.get_rows(); ++i) {
        for (size_t j = 0; j < i; ++j) {
            EXPECT_DOUBLE_EQ(U(i, j), 0.0);
        }
    }
    
    // 验证L的对角线元素为1
    for (size_t i = 0; i < L.get_rows(); ++i) {
        EXPECT_DOUBLE_EQ(L(i, i), 1.0);
    }
    
    // 验证A = LU
    Matrix<double> reconstructed = L * U;
    EXPECT_TRUE(matrices_are_close(A, reconstructed, 1e-10));
}

// 特征值计算测试
TEST(MatrixTest, EigenvalueProperties) {
    // 测试对称矩阵的特征值
    Matrix<double> A({{2, 1}, {1, 2}});
    auto [eigenvalues, eigenvectors] = A.eigenvalues_eigenvectors();
    
    // 验证特征值是实数
    for (const auto& lambda : eigenvalues) {
        EXPECT_NEAR(lambda.imag(), 0.0, 1e-10);
    }
    
    // 验证特征向量的正交性
    for (size_t i = 0; i < eigenvectors.size(); ++i) {
        for (size_t j = i + 1; j < eigenvectors.size(); ++j) {
            double dot_product = eigenvectors[i].dot(eigenvectors[j]);
            EXPECT_NEAR(dot_product, 0.0, 1e-10);
        }
    }
    
    // 验证特征值方程
    for (size_t i = 0; i < eigenvalues.size(); ++i) {
        Vector<double> Av = A * eigenvectors[i];
        Vector<double> lambda_v = eigenvectors[i] * eigenvalues[i].real();
        EXPECT_TRUE(vectors_are_close(Av, lambda_v, 1e-10));
    }
}

// 向量运算测试
TEST(VectorTest, VectorOperations) {
    // 测试向量的正交性
    Vector<double> v1({1, 0, 0});
    Vector<double> v2({0, 1, 0});
    Vector<double> v3({0, 0, 1});
    
    EXPECT_DOUBLE_EQ(v1.dot(v2), 0.0);
    EXPECT_DOUBLE_EQ(v2.dot(v3), 0.0);
    EXPECT_DOUBLE_EQ(v3.dot(v1), 0.0);
    
    // 测试叉积的性质
    Vector<double> cross = v1.cross(v2);
    EXPECT_TRUE(vectors_are_close(cross, v3));
    
    // 测试投影的性质
    Vector<double> v({1, 1, 1});
    Vector<double> proj = v.project(v1);
    EXPECT_TRUE(vectors_are_close(proj, Vector<double>({1, 0, 0})));
    
    // 验证投影后的残差与投影方向正交
    Vector<double> residual = v - proj;
    EXPECT_NEAR(residual.dot(v1), 0.0, 1e-10);
}

// 最小二乘解测试
TEST(LinearSystemTest, LeastSquaresProperties) {
    // 测试超定系统
    Matrix<double> A({{1, 1}, {1, 2}, {1, 3}});
    Vector<double> b({1, 2, 2});
    
    Vector<double> x = least_squares(A, b);
    
    // 验证法方程
    Matrix<double> ATA = A.transpose() * A;
    Vector<double> ATb = A.transpose() * b;
    Vector<double> ATAx = ATA * x;
    EXPECT_TRUE(vectors_are_close(ATAx, ATb, 1e-10));
    
    // 验证残差与A的列空间正交
    Vector<double> residual = A * x - b;
    for (size_t j = 0; j < A.get_cols(); ++j) {
        Vector<double> col(A.get_rows());
        for (size_t i = 0; i < A.get_rows(); ++i) {
            col[i] = A(i, j);
        }
        EXPECT_NEAR(residual.dot(col), 0.0, 1e-10);
    }
}

// 条件数测试
TEST(MatrixTest, ConditionNumberProperties) {
    // 测试单位矩阵的条件数
    Matrix<double> I({{1, 0}, {0, 1}});
    EXPECT_DOUBLE_EQ(condition_number(I), 1.0);
    
    // 测试条件数的尺度不变性
    Matrix<double> A = generate_random_matrix<double>(3, 3);
    double scale = 2.0;
    Matrix<double> scaled_A = A * scale;
    EXPECT_DOUBLE_EQ(condition_number(scaled_A), condition_number(A));
    
    // 测试病态矩阵
    Matrix<double> ill_conditioned({{1, 1}, {1, 1.0001}});
    double cond = condition_number(ill_conditioned);
    EXPECT_GT(cond, 1000);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 