#include "../include/mathlib.hpp"
#include <iostream>
#include <iomanip>
#include <vector>

int main() {
    // 欧拉多项式示例
    std::cout << "Euler Polynomials / 欧拉多项式:" << std::endl;
    for (int n = 0; n <= 3; ++n) {
        std::cout << "E_" << n << "(0.5) = " 
                  << mathlib::combinatorics::euler_polynomial(n, 0.5) << std::endl;
    }
    
    // 伯努利多项式示例
    std::cout << "\nBernoulli Polynomials / 伯努利多项式:" << std::endl;
    for (int n = 0; n <= 3; ++n) {
        std::cout << "B_" << n << "(0.5) = " 
                  << mathlib::combinatorics::bernoulli_polynomial(n, 0.5) << std::endl;
    }
    
    // 广义调和数示例
    std::cout << "\nGeneralized Harmonic Numbers / 广义调和数:" << std::endl;
    for (int n = 1; n <= 5; ++n) {
        std::cout << "H_" << n << "^(2) = " 
                  << mathlib::combinatorics::generalized_harmonic(n, 2) << std::endl;
    }
    
    // 多对数函数示例
    std::cout << "\nPolylogarithm / 多对数函数:" << std::endl;
    for (int n = 1; n <= 3; ++n) {
        std::cout << "Li_" << n << "(0.5) = " 
                  << mathlib::combinatorics::polylogarithm(n, 0.5) << std::endl;
    }
    
    // 拉盖尔多项式示例
    std::cout << "\nLaguerre Polynomials / 拉盖尔多项式:" << std::endl;
    for (int n = 0; n <= 3; ++n) {
        std::cout << "L_" << n << "(1.0) = " 
                  << mathlib::combinatorics::laguerre_polynomial(n, 1.0) << std::endl;
    }
    
    // 切比雪夫多项式示例
    std::cout << "\nChebyshev Polynomials / 切比雪夫多项式:" << std::endl;
    std::cout << "First Kind / 第一类:" << std::endl;
    for (int n = 0; n <= 3; ++n) {
        std::cout << "T_" << n << "(0.5) = " 
                  << mathlib::combinatorics::chebyshev_polynomial(n, 0.5) << std::endl;
    }
    std::cout << "\nSecond Kind / 第二类:" << std::endl;
    for (int n = 0; n <= 3; ++n) {
        std::cout << "U_" << n << "(0.5) = " 
                  << mathlib::combinatorics::chebyshev_polynomial_second_kind(n, 0.5) << std::endl;
    }
    
    return 0;
} 