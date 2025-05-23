#pragma once

#include <vector>
#include <functional>
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace mathlib {
namespace numerical {

// 数值积分
template<typename T>
T trapezoidal_rule(std::function<T(T)> f, T a, T b, int n) {
    if (n <= 0) throw std::invalid_argument("Number of intervals must be positive");
    T h = (b - a) / n;
    T sum = (f(a) + f(b)) / 2.0;
    for (int i = 1; i < n; ++i) {
        sum += f(a + i * h);
    }
    return h * sum;
}

template<typename T>
T simpson_rule(std::function<T(T)> f, T a, T b, int n) {
    if (n <= 0 || n % 2 != 0) throw std::invalid_argument("Number of intervals must be positive and even");
    T h = (b - a) / n;
    T sum = f(a) + f(b);
    for (int i = 1; i < n; ++i) {
        sum += (i % 2 == 0 ? 2 : 4) * f(a + i * h);
    }
    return h * sum / 3.0;
}

// 数值微分
template<typename T>
T central_difference(std::function<T(T)> f, T x, T h = 1e-6) {
    return (f(x + h) - f(x - h)) / (2 * h);
}

template<typename T>
T forward_difference(std::function<T(T)> f, T x, T h = 1e-6) {
    return (f(x + h) - f(x)) / h;
}

// 方程求解
template<typename T>
T newton_method(std::function<T(T)> f, std::function<T(T)> df, T x0, T tol = 1e-6, int max_iter = 100) {
    T x = x0;
    for (int i = 0; i < max_iter; ++i) {
        T fx = f(x);
        if (std::abs(fx) < tol) return x;
        T dfx = df(x);
        if (std::abs(dfx) < tol) throw std::runtime_error("Derivative too close to zero");
        x = x - fx / dfx;
    }
    throw std::runtime_error("Newton method did not converge");
}

template<typename T>
T bisection_method(std::function<T(T)> f, T a, T b, T tol = 1e-6, int max_iter = 100) {
    if (f(a) * f(b) >= 0) throw std::invalid_argument("Function must have opposite signs at endpoints");
    
    T c;
    for (int i = 0; i < max_iter; ++i) {
        c = (a + b) / 2;
        T fc = f(c);
        if (std::abs(fc) < tol) return c;
        if (fc * f(a) < 0) b = c;
        else a = c;
    }
    throw std::runtime_error("Bisection method did not converge");
}

// 插值方法
template<typename T>
T lagrange_interpolation(const std::vector<T>& x, const std::vector<T>& y, T point) {
    if (x.size() != y.size()) throw std::invalid_argument("x and y vectors must have same size");
    
    T result = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        T term = y[i];
        for (size_t j = 0; j < x.size(); ++j) {
            if (i != j) {
                term *= (point - x[j]) / (x[i] - x[j]);
            }
        }
        result += term;
    }
    return result;
}

// 常微分方程求解
template<typename T>
std::vector<T> euler_method(std::function<T(T, T)> f, T t0, T y0, T h, int n) {
    std::vector<T> result(n + 1);
    result[0] = y0;
    T t = t0;
    
    for (int i = 0; i < n; ++i) {
        result[i + 1] = result[i] + h * f(t, result[i]);
        t += h;
    }
    return result;
}

template<typename T>
std::vector<T> runge_kutta_4(std::function<T(T, T)> f, T t0, T y0, T h, int n) {
    std::vector<T> result(n + 1);
    result[0] = y0;
    T t = t0;
    
    for (int i = 0; i < n; ++i) {
        T k1 = f(t, result[i]);
        T k2 = f(t + h/2, result[i] + h*k1/2);
        T k3 = f(t + h/2, result[i] + h*k2/2);
        T k4 = f(t + h, result[i] + h*k3);
        
        result[i + 1] = result[i] + h * (k1 + 2*k2 + 2*k3 + k4) / 6;
        t += h;
    }
    return result;
}

// 高斯求积
template<typename T>
T gauss_quadrature(std::function<T(T)> f, T a, T b, int n) {
    // 高斯-勒让德求积点
    static const std::vector<std::pair<T, T>> points_weights = {
        {0.5773502691896257, 1.0},
        {-0.5773502691896257, 1.0},
        {0.7745966692414834, 0.5555555555555556},
        {0.0, 0.8888888888888888},
        {-0.7745966692414834, 0.5555555555555556}
    };
    
    T sum = 0;
    T scale = (b - a) / 2;
    T shift = (a + b) / 2;
    
    for (const auto& [point, weight] : points_weights) {
        sum += weight * f(scale * point + shift);
    }
    
    return scale * sum;
}

// 高斯-勒让德求积法
template<typename Func>
double gauss_legendre_quadrature(Func f, double a, double b, int n) {
    if (n <= 0) {
        throw std::invalid_argument("阶数必须为正数");
    }

    // 高斯-勒让德求积点（预计算）
    static const std::vector<std::pair<double, double>> points_weights = {
        {0.0, 2.0},  // n=1
        {-0.5773502691896257, 1.0}, {0.5773502691896257, 1.0},  // n=2
        {-0.7745966692414834, 0.5555555555555556},  // n=3
        {0.0, 0.8888888888888888},
        {0.7745966692414834, 0.5555555555555556},
        {-0.8611363115940526, 0.3478548451374538},  // n=4
        {-0.3399810435848563, 0.6521451548625461},
        {0.3399810435848563, 0.6521451548625461},
        {0.8611363115940526, 0.3478548451374538}
    };

    // 计算积分
    double sum = 0.0;
    double scale = (b - a) / 2.0;
    double shift = (a + b) / 2.0;

    for (int i = 0; i < n; ++i) {
        double x = scale * points_weights[n*(n-1)/2 + i].first + shift;
        double w = points_weights[n*(n-1)/2 + i].second;
        sum += w * f(x);
    }

    return scale * sum;
}

// 自适应积分法
template<typename Func>
double adaptive_quadrature(Func f, double a, double b, double tol = 1e-6, int max_depth = 10) {
    if (a >= b) {
        throw std::invalid_argument("积分区间无效");
    }

    auto integrate = [&](auto&& self, double a, double b, double fa, double fb, double fc, double tol, int depth) -> double {
        double h = b - a;
        double c = (a + b) / 2.0;
        double fd = f((a + c) / 2.0);
        double fe = f((c + b) / 2.0);

        double S1 = h * (fa + 4 * fc + fb) / 6.0;
        double S2 = h * (fa + 4 * fd + 2 * fc + 4 * fe + fb) / 12.0;

        if (depth >= max_depth || std::abs(S2 - S1) < 15 * tol) {
            return S2 + (S2 - S1) / 15.0;
        }

        return self(self, a, c, fa, fc, fd, tol/2.0, depth+1) +
               self(self, c, b, fc, fb, fe, tol/2.0, depth+1);
    };

    double fa = f(a);
    double fb = f(b);
    double fc = f((a + b) / 2.0);

    return integrate(integrate, a, b, fa, fb, fc, tol, 0);
}

// 多重积分（二重积分）
template<typename Func>
double double_integral(Func f, double a, double b, double c, double d, int nx = 100, int ny = 100) {
    if (a >= b || c >= d) {
        throw std::invalid_argument("积分区间无效");
    }

    double hx = (b - a) / nx;
    double hy = (d - c) / ny;
    double sum = 0.0;

    for (int i = 0; i < nx; ++i) {
        double x = a + (i + 0.5) * hx;
        for (int j = 0; j < ny; ++j) {
            double y = c + (j + 0.5) * hy;
            sum += f(x, y);
        }
    }

    return hx * hy * sum;
}

} // namespace numerical
} // namespace mathlib 