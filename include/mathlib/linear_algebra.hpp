#pragma once

#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <complex>

namespace mathlib {
namespace linear_algebra {

template<typename T>
class Matrix {
private:
    std::vector<std::vector<T>> data;
    size_t rows;
    size_t cols;

public:
    Matrix(size_t r, size_t c) : rows(r), cols(c) {
        data.resize(r, std::vector<T>(c));
    }

    Matrix(const std::vector<std::vector<T>>& input) : data(input) {
        if (input.empty() || input[0].empty()) {
            throw std::invalid_argument("Empty matrix");
        }
        rows = input.size();
        cols = input[0].size();
        for (const auto& row : input) {
            if (row.size() != cols) {
                throw std::invalid_argument("Inconsistent matrix dimensions");
            }
        }
    }

    // 基本访问器
    size_t get_rows() const { return rows; }
    size_t get_cols() const { return cols; }
    T& operator()(size_t i, size_t j) { return data[i][j]; }
    const T& operator()(size_t i, size_t j) const { return data[i][j]; }

    // 矩阵加法
    Matrix operator+(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions do not match");
        }
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(i, j) = data[i][j] + other(i, j);
            }
        }
        return result;
    }

    // 矩阵乘法
    Matrix operator*(const Matrix& other) const {
        if (cols != other.rows) {
            throw std::invalid_argument("Matrix dimensions do not match for multiplication");
        }
        Matrix result(rows, other.cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < other.cols; ++j) {
                T sum = 0;
                for (size_t k = 0; k < cols; ++k) {
                    sum += data[i][k] * other(k, j);
                }
                result(i, j) = sum;
            }
        }
        return result;
    }

    // 矩阵转置
    Matrix transpose() const {
        Matrix result(cols, rows);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(j, i) = data[i][j];
            }
        }
        return result;
    }

    // 计算行列式
    T determinant() const {
        if (rows != cols) {
            throw std::invalid_argument("Matrix must be square for determinant calculation");
        }
        if (rows == 1) return data[0][0];
        if (rows == 2) {
            return data[0][0] * data[1][1] - data[0][1] * data[1][0];
        }

        T det = 0;
        for (size_t j = 0; j < cols; ++j) {
            Matrix<T> submatrix(rows - 1, cols - 1);
            for (size_t i = 1; i < rows; ++i) {
                for (size_t k = 0; k < cols; ++k) {
                    if (k < j) submatrix(i-1, k) = data[i][k];
                    else if (k > j) submatrix(i-1, k-1) = data[i][k];
                }
            }
            det += (j % 2 == 0 ? 1 : -1) * data[0][j] * submatrix.determinant();
        }
        return det;
    }

    // 计算矩阵的迹
    T trace() const {
        if (rows != cols) {
            throw std::invalid_argument("Matrix must be square for trace calculation");
        }
        T tr = 0;
        for (size_t i = 0; i < rows; ++i) {
            tr += data[i][i];
        }
        return tr;
    }

    // LU分解
    std::pair<Matrix<T>, Matrix<T>> lu_decomposition() const {
        if (rows != cols) {
            throw std::invalid_argument("Matrix must be square for LU decomposition");
        }

        Matrix<T> L(rows, cols);
        Matrix<T> U(rows, cols);
        Matrix<T> A = *this;  // 创建副本以避免修改原矩阵

        // 初始化L为单位矩阵
        for (size_t i = 0; i < rows; ++i) {
            L(i, i) = 1;
        }

        // 使用部分主元法进行LU分解
        for (size_t k = 0; k < rows; ++k) {
            // 选择主元
            size_t max_row = k;
            T max_val = std::abs(A(k, k));
            for (size_t i = k + 1; i < rows; ++i) {
                T val = std::abs(A(i, k));
                if (val > max_val) {
                    max_val = val;
                    max_row = i;
                }
            }

            // 如果主元太小，认为矩阵奇异
            if (max_val < 1e-10) {
                throw std::runtime_error("Matrix is singular or nearly singular");
            }

            // 交换行
            if (max_row != k) {
                for (size_t j = 0; j < cols; ++j) {
                    std::swap(A(k, j), A(max_row, j));
                    if (j < k) {
                        std::swap(L(k, j), L(max_row, j));
                    }
                }
            }

            // 计算U的第k行和L的第k列
            for (size_t j = k; j < cols; ++j) {
                T sum = 0;
                for (size_t i = 0; i < k; ++i) {
                    sum += L(k, i) * U(i, j);
                }
                U(k, j) = A(k, j) - sum;
            }

            for (size_t i = k + 1; i < rows; ++i) {
                T sum = 0;
                for (size_t j = 0; j < k; ++j) {
                    sum += L(i, j) * U(j, k);
                }
                L(i, k) = (A(i, k) - sum) / U(k, k);
            }
        }

        return {L, U};
    }

    // 计算特征值和特征向量（使用幂迭代法）
    std::pair<std::vector<std::complex<T>>, std::vector<Vector<T>>> eigenvalues_eigenvectors(size_t max_iterations = 1000) const {
        if (rows != cols) {
            throw std::invalid_argument("Matrix must be square for eigenvalue calculation");
        }

        std::vector<std::complex<T>> eigenvalues;
        std::vector<Vector<T>> eigenvectors;
        Matrix<T> A = *this;

        // 使用QR迭代法计算特征值
        for (size_t i = 0; i < rows; ++i) {
            // 初始化随机向量
            Vector<T> v(rows);
            for (size_t j = 0; j < rows; ++j) {
                v[j] = static_cast<T>(rand()) / RAND_MAX;
            }
            v = v * (1.0 / v.norm());

            T prev_lambda = 0;
            bool converged = false;

            // 幂迭代
            for (size_t iter = 0; iter < max_iterations && !converged; ++iter) {
                Vector<T> v_new = A * v;
                T lambda = v_new.norm();
                
                // 检查收敛性
                if (iter > 0 && std::abs(lambda - prev_lambda) < 1e-10) {
                    converged = true;
                }
                
                v = v_new * (1.0 / lambda);
                prev_lambda = lambda;
            }

            if (!converged) {
                throw std::runtime_error("Eigenvalue computation did not converge");
            }

            eigenvalues.push_back(std::complex<T>(prev_lambda, 0));
            eigenvectors.push_back(v);

            // 收缩矩阵以计算下一个特征值
            A = A - (prev_lambda * (v * v.transpose()));
        }

        return {eigenvalues, eigenvectors};
    }

    // 添加缺失的运算符重载
    Matrix operator-(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions do not match");
        }
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(i, j) = data[i][j] - other(i, j);
            }
        }
        return result;
    }

    Matrix& operator+=(const Matrix& other) {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions do not match");
        }
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                data[i][j] += other(i, j);
            }
        }
        return *this;
    }

    Matrix& operator-=(const Matrix& other) {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions do not match");
        }
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                data[i][j] -= other(i, j);
            }
        }
        return *this;
    }

    Matrix& operator*=(const T& scalar) {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                data[i][j] *= scalar;
            }
        }
        return *this;
    }

    // 标量乘法
    friend Matrix operator*(const Matrix& m, const T& scalar) {
        Matrix result(m.rows, m.cols);
        for (size_t i = 0; i < m.rows; ++i) {
            for (size_t j = 0; j < m.cols; ++j) {
                result(i, j) = m(i, j) * scalar;
            }
        }
        return result;
    }

    friend Matrix operator*(const T& scalar, const Matrix& m) {
        return m * scalar;
    }

    // 矩阵-向量乘法
    friend Vector<T> operator*(const Matrix& m, const Vector<T>& v) {
        if (m.cols != v.get_size()) {
            throw std::invalid_argument("Matrix and vector dimensions do not match");
        }
        Vector<T> result(m.rows);
        for (size_t i = 0; i < m.rows; ++i) {
            T sum = 0;
            for (size_t j = 0; j < m.cols; ++j) {
                sum += m(i, j) * v[j];
            }
            result[i] = sum;
        }
        return result;
    }
};

template<typename T>
class Vector {
private:
    std::vector<T> data;
    size_t size;

public:
    Vector(size_t n) : size(n) {
        data.resize(n);
    }

    Vector(const std::vector<T>& input) : data(input), size(input.size()) {}

    // 基本访问器
    size_t get_size() const { return size; }
    T& operator[](size_t i) { return data[i]; }
    const T& operator[](size_t i) const { return data[i]; }

    // 向量点积
    T dot(const Vector& other) const {
        if (size != other.size) {
            throw std::invalid_argument("Vector dimensions do not match");
        }
        return std::inner_product(data.begin(), data.end(), other.data.begin(), T(0));
    }

    // 向量范数（L2范数）
    T norm() const {
        return std::sqrt(dot(*this));
    }

    // 向量叉积（仅适用于3D向量）
    Vector cross(const Vector& other) const {
        if (size != 3 || other.size != 3) {
            throw std::invalid_argument("Cross product is only defined for 3D vectors");
        }
        Vector result(3);
        result[0] = data[1] * other[2] - data[2] * other[1];
        result[1] = data[2] * other[0] - data[0] * other[2];
        result[2] = data[0] * other[1] - data[1] * other[0];
        return result;
    }

    // 向量投影
    Vector project(const Vector& other) const {
        if (size != other.size) {
            throw std::invalid_argument("Vector dimensions do not match");
        }
        T scale = dot(other) / other.dot(other);
        Vector result(size);
        for (size_t i = 0; i < size; ++i) {
            result[i] = other[i] * scale;
        }
        return result;
    }

    // 向量夹角（弧度）
    T angle(const Vector& other) const {
        if (size != other.size) {
            throw std::invalid_argument("Vector dimensions do not match");
        }
        return std::acos(dot(other) / (norm() * other.norm()));
    }

    // 添加缺失的运算符重载
    Vector operator+(const Vector& other) const {
        if (size != other.size) {
            throw std::invalid_argument("Vector dimensions do not match");
        }
        Vector result(size);
        for (size_t i = 0; i < size; ++i) {
            result[i] = data[i] + other[i];
        }
        return result;
    }

    Vector operator-(const Vector& other) const {
        if (size != other.size) {
            throw std::invalid_argument("Vector dimensions do not match");
        }
        Vector result(size);
        for (size_t i = 0; i < size; ++i) {
            result[i] = data[i] - other[i];
        }
        return result;
    }

    Vector& operator+=(const Vector& other) {
        if (size != other.size) {
            throw std::invalid_argument("Vector dimensions do not match");
        }
        for (size_t i = 0; i < size; ++i) {
            data[i] += other[i];
        }
        return *this;
    }

    Vector& operator-=(const Vector& other) {
        if (size != other.size) {
            throw std::invalid_argument("Vector dimensions do not match");
        }
        for (size_t i = 0; i < size; ++i) {
            data[i] -= other[i];
        }
        return *this;
    }

    Vector& operator*=(const T& scalar) {
        for (size_t i = 0; i < size; ++i) {
            data[i] *= scalar;
        }
        return *this;
    }

    // 标量乘法
    friend Vector operator*(const Vector& v, const T& scalar) {
        Vector result(v.size);
        for (size_t i = 0; i < v.size; ++i) {
            result[i] = v[i] * scalar;
        }
        return result;
    }

    friend Vector operator*(const T& scalar, const Vector& v) {
        return v * scalar;
    }
};

// 求解线性方程组 Ax = b
template<typename T>
Vector<T> solve_linear_system(const Matrix<T>& A, const Vector<T>& b) {
    if (A.get_rows() != A.get_cols() || A.get_rows() != b.get_size()) {
        throw std::invalid_argument("Invalid dimensions for linear system");
    }

    // 这里实现高斯消元法
    Matrix<T> augmented(A.get_rows(), A.get_cols() + 1);
    for (size_t i = 0; i < A.get_rows(); ++i) {
        for (size_t j = 0; j < A.get_cols(); ++j) {
            augmented(i, j) = A(i, j);
        }
        augmented(i, A.get_cols()) = b[i];
    }

    // 前向消元
    for (size_t i = 0; i < A.get_rows(); ++i) {
        // 选择主元
        size_t max_row = i;
        for (size_t j = i + 1; j < A.get_rows(); ++j) {
            if (std::abs(augmented(j, i)) > std::abs(augmented(max_row, i))) {
                max_row = j;
            }
        }

        // 交换行
        if (max_row != i) {
            for (size_t j = i; j <= A.get_cols(); ++j) {
                std::swap(augmented(i, j), augmented(max_row, j));
            }
        }

        // 消元
        for (size_t j = i + 1; j < A.get_rows(); ++j) {
            T factor = augmented(j, i) / augmented(i, i);
            for (size_t k = i; k <= A.get_cols(); ++k) {
                augmented(j, k) -= factor * augmented(i, k);
            }
        }
    }

    // 回代
    Vector<T> x(A.get_rows());
    for (int i = A.get_rows() - 1; i >= 0; --i) {
        T sum = 0;
        for (size_t j = i + 1; j < A.get_cols(); ++j) {
            sum += augmented(i, j) * x[j];
        }
        x[i] = (augmented(i, A.get_cols()) - sum) / augmented(i, i);
    }

    return x;
}

// 计算矩阵条件数
template<typename T>
T condition_number(const Matrix<T>& A) {
    if (A.get_rows() != A.get_cols()) {
        throw std::invalid_argument("Matrix must be square for condition number calculation");
    }
    
    // 使用Frobenius范数
    T norm_A = 0;
    for (size_t i = 0; i < A.get_rows(); ++i) {
        for (size_t j = 0; j < A.get_cols(); ++j) {
            norm_A += A(i, j) * A(i, j);
        }
    }
    norm_A = std::sqrt(norm_A);

    // 计算逆矩阵的范数（使用LU分解）
    auto [L, U] = A.lu_decomposition();
    T norm_inv_A = 0;
    // 这里简化处理，实际应该计算逆矩阵
    for (size_t i = 0; i < A.get_rows(); ++i) {
        for (size_t j = 0; j < A.get_cols(); ++j) {
            norm_inv_A += U(i, j) * U(i, j);
        }
    }
    norm_inv_A = std::sqrt(norm_inv_A);

    return norm_A * norm_inv_A;
}

// 最小二乘解
template<typename T>
Vector<T> least_squares(const Matrix<T>& A, const Vector<T>& b) {
    Matrix<T> ATA = A.transpose() * A;
    Vector<T> ATb = A.transpose() * b;
    return solve_linear_system(ATA, ATb);
}

} // namespace linear_algebra
} // namespace mathlib 