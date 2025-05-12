#include "../include/mathlib.hpp"
#include <iostream>
#include <iomanip>
#include <vector>

int main() {
    // 正态分布示例
    std::cout << "Normal Distribution / 正态分布:" << std::endl;
    mathlib::probability::NormalDistribution normal(0.0, 1.0);
    std::cout << "PDF at x=0: " << normal.pdf(0.0) << std::endl;
    std::cout << "CDF at x=1: " << normal.cdf(1.0) << std::endl;
    std::cout << "Random sample: " << normal.sample() << std::endl;
    
    // 指数分布示例
    std::cout << "\nExponential Distribution / 指数分布:" << std::endl;
    mathlib::probability::ExponentialDistribution exp(2.0);
    std::cout << "PDF at x=1: " << exp.pdf(1.0) << std::endl;
    std::cout << "CDF at x=1: " << exp.cdf(1.0) << std::endl;
    std::cout << "Random sample: " << exp.sample() << std::endl;
    
    // 卡方分布示例
    std::cout << "\nChi-Squared Distribution / 卡方分布:" << std::endl;
    mathlib::probability::ChiSquaredDistribution chi(3);
    std::cout << "PDF at x=1: " << chi.pdf(1.0) << std::endl;
    std::cout << "CDF at x=1: " << chi.cdf(1.0) << std::endl;
    std::cout << "Random sample: " << chi.sample() << std::endl;
    
    // t分布示例
    std::cout << "\nt-Distribution / t分布:" << std::endl;
    mathlib::probability::TDistribution t(10);
    std::cout << "PDF at x=0: " << t.pdf(0.0) << std::endl;
    std::cout << "CDF at x=1: " << t.cdf(1.0) << std::endl;
    std::cout << "Random sample: " << t.sample() << std::endl;
    
    // Beta分布示例
    std::cout << "\nBeta Distribution / Beta分布:" << std::endl;
    mathlib::probability::BetaDistribution beta(2.0, 3.0);
    std::cout << "PDF at x=0.5: " << beta.pdf(0.5) << std::endl;
    std::cout << "CDF at x=0.5: " << beta.cdf(0.5) << std::endl;
    std::cout << "Random sample: " << beta.sample() << std::endl;
    
    // Gamma分布示例
    std::cout << "\nGamma Distribution / Gamma分布:" << std::endl;
    mathlib::probability::GammaDistribution gamma(2.0, 3.0);
    std::cout << "PDF at x=1: " << gamma.pdf(1.0) << std::endl;
    std::cout << "CDF at x=1: " << gamma.cdf(1.0) << std::endl;
    std::cout << "Random sample: " << gamma.sample() << std::endl;
    
    // 威布尔分布示例
    std::cout << "\nWeibull Distribution / 威布尔分布:" << std::endl;
    mathlib::probability::WeibullDistribution weibull(2.0, 1.0);
    std::cout << "PDF at x=1: " << weibull.pdf(1.0) << std::endl;
    std::cout << "CDF at x=1: " << weibull.cdf(1.0) << std::endl;
    std::cout << "Random sample: " << weibull.sample() << std::endl;
    
    // 对数正态分布示例
    std::cout << "\nLog-normal Distribution / 对数正态分布:" << std::endl;
    mathlib::probability::LogNormalDistribution lognormal(0.0, 1.0);
    std::cout << "PDF at x=1: " << lognormal.pdf(1.0) << std::endl;
    std::cout << "CDF at x=1: " << lognormal.cdf(1.0) << std::endl;
    std::cout << "Random sample: " << lognormal.sample() << std::endl;
    
    return 0;
} 