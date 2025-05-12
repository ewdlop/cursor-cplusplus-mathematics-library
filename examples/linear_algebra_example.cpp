#include "mathlib/linear_algebra.hpp"
#include <iostream>
#include <iomanip>

int main() {
    try {
        // 创建矩阵
        mathlib::linear_algebra::Matrix<double> A(3, 3);
        A(0, 0) = 2; A(0, 1) = 1; A(0, 2) = -1;
        A(1, 0) = -3; A(1, 1) = -1; A(1, 2) = 2;
        A(2, 0) = -2; A(2, 1) = 1; A(2, 2) = 2;

        // 创建向量
        mathlib::linear_algebra::Vector<double> b({8, -11, -3});

        std::cout << "Solving linear system Ax = b:\n";
        std::cout << "Matrix A:\n";
        for (size_t i = 0; i < A.get_rows(); ++i) {
            for (size_t j = 0; j < A.get_cols(); ++j) {
                std::cout << std::setw(8) << A(i, j);
            }
            std::cout << "\n";
        }

        std::cout << "\nVector b:\n";
        for (size_t i = 0; i < b.get_size(); ++i) {
            std::cout << std::setw(8) << b[i];
        }
        std::cout << "\n\n";

        // 求解线性方程组
        auto x = mathlib::linear_algebra::solve_linear_system(A, b);

        std::cout << "Solution x:\n";
        for (size_t i = 0; i < x.get_size(); ++i) {
            std::cout << std::setw(8) << x[i];
        }
        std::cout << "\n";

        // 验证解
        std::cout << "\nVerifying solution (Ax):\n";
        mathlib::linear_algebra::Vector<double> Ax(b.get_size());
        for (size_t i = 0; i < A.get_rows(); ++i) {
            double sum = 0;
            for (size_t j = 0; j < A.get_cols(); ++j) {
                sum += A(i, j) * x[j];
            }
            Ax[i] = sum;
            std::cout << std::setw(8) << Ax[i];
        }
        std::cout << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
} 