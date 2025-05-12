#pragma once

#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <numeric>

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

} // namespace linear_algebra
} // namespace mathlib 