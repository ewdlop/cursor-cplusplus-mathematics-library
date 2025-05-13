#pragma once

#include <vector>
#include <functional>
#include <cmath>
#include <stdexcept>
#include <random>
#include <algorithm>

namespace mathlib {
namespace optimization {

// 梯度下降
template<typename T>
std::vector<T> gradient_descent(
    std::function<T(const std::vector<T>&)> objective,
    std::function<std::vector<T>(const std::vector<T>&)> gradient,
    const std::vector<T>& initial_point,
    T learning_rate = 0.01,
    T tolerance = 1e-6,
    int max_iterations = 1000
) {
    std::vector<T> current_point = initial_point;
    T prev_value = objective(current_point);
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        std::vector<T> grad = gradient(current_point);
        
        // 更新点
        for (size_t i = 0; i < current_point.size(); ++i) {
            current_point[i] -= learning_rate * grad[i];
        }
        
        T current_value = objective(current_point);
        if (std::abs(current_value - prev_value) < tolerance) {
            break;
        }
        prev_value = current_value;
    }
    
    return current_point;
}

// 共轭梯度法
template<typename T>
std::vector<T> conjugate_gradient(
    std::function<T(const std::vector<T>&)> objective,
    std::function<std::vector<T>(const std::vector<T>&)> gradient,
    const std::vector<T>& initial_point,
    T tolerance = 1e-6,
    int max_iterations = 1000
) {
    std::vector<T> x = initial_point;
    std::vector<T> r = gradient(x);
    std::vector<T> p = r;
    T rsold = 0;
    
    for (size_t i = 0; i < r.size(); ++i) {
        rsold += r[i] * r[i];
    }
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        std::vector<T> Ap = gradient(x);
        T alpha = rsold;
        for (size_t i = 0; i < Ap.size(); ++i) {
            alpha -= p[i] * Ap[i];
        }
        alpha = rsold / alpha;
        
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }
        
        T rsnew = 0;
        for (size_t i = 0; i < r.size(); ++i) {
            rsnew += r[i] * r[i];
        }
        
        if (std::sqrt(rsnew) < tolerance) {
            break;
        }
        
        for (size_t i = 0; i < p.size(); ++i) {
            p[i] = r[i] + (rsnew / rsold) * p[i];
        }
        rsold = rsnew;
    }
    
    return x;
}

// 模拟退火
template<typename T>
std::vector<T> simulated_annealing(
    std::function<T(const std::vector<T>&)> objective,
    std::function<std::vector<T>(const std::vector<T>&)> neighbor,
    const std::vector<T>& initial_point,
    T initial_temp = 100.0,
    T final_temp = 1e-6,
    T cooling_rate = 0.95,
    int iterations_per_temp = 100
) {
    std::vector<T> current_point = initial_point;
    T current_value = objective(current_point);
    std::vector<T> best_point = current_point;
    T best_value = current_value;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dist(0.0, 1.0);
    
    for (T temp = initial_temp; temp > final_temp; temp *= cooling_rate) {
        for (int i = 0; i < iterations_per_temp; ++i) {
            std::vector<T> new_point = neighbor(current_point);
            T new_value = objective(new_point);
            T delta = new_value - current_value;
            
            if (delta < 0 || dist(gen) < std::exp(-delta / temp)) {
                current_point = new_point;
                current_value = new_value;
                
                if (current_value < best_value) {
                    best_point = current_point;
                    best_value = current_value;
                }
            }
        }
    }
    
    return best_point;
}

// 线性规划（单纯形法）
template<typename T>
std::vector<T> simplex_method(
    const std::vector<std::vector<T>>& A,
    const std::vector<T>& b,
    const std::vector<T>& c,
    T tolerance = 1e-6
) {
    int m = A.size();
    int n = A[0].size();
    
    // 构建单纯形表
    std::vector<std::vector<T>> tableau(m + 1, std::vector<T>(n + m + 1));
    
    // 填充约束矩阵
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            tableau[i][j] = A[i][j];
        }
        tableau[i][n + i] = 1;  // 添加松弛变量
        tableau[i][n + m] = b[i];
    }
    
    // 填充目标函数
    for (int j = 0; j < n; ++j) {
        tableau[m][j] = -c[j];
    }
    
    while (true) {
        // 找到进入变量
        int entering = -1;
        T min_val = 0;
        for (int j = 0; j < n + m; ++j) {
            if (tableau[m][j] < min_val) {
                min_val = tableau[m][j];
                entering = j;
            }
        }
        
        if (entering == -1) break;  // 最优解找到
        
        // 找到离开变量
        int leaving = -1;
        T min_ratio = std::numeric_limits<T>::infinity();
        for (int i = 0; i < m; ++i) {
            if (tableau[i][entering] > tolerance) {
                T ratio = tableau[i][n + m] / tableau[i][entering];
                if (ratio < min_ratio) {
                    min_ratio = ratio;
                    leaving = i;
                }
            }
        }
        
        if (leaving == -1) throw std::runtime_error("Unbounded solution");
        
        // 主元消元
        T pivot = tableau[leaving][entering];
        for (int j = 0; j <= n + m; ++j) {
            tableau[leaving][j] /= pivot;
        }
        
        for (int i = 0; i <= m; ++i) {
            if (i != leaving) {
                T factor = tableau[i][entering];
                for (int j = 0; j <= n + m; ++j) {
                    tableau[i][j] -= factor * tableau[leaving][j];
                }
            }
        }
    }
    
    // 提取解
    std::vector<T> solution(n);
    for (int j = 0; j < n; ++j) {
        int basic_row = -1;
        for (int i = 0; i < m; ++i) {
            if (std::abs(tableau[i][j] - 1) < tolerance) {
                if (basic_row == -1) {
                    basic_row = i;
                } else {
                    basic_row = -1;
                    break;
                }
            }
        }
        solution[j] = (basic_row != -1) ? tableau[basic_row][n + m] : 0;
    }
    
    return solution;
}

// 粒子群优化算法
template<typename Func>
std::vector<double> particle_swarm_optimization(
    Func objective,
    const std::vector<double>& lower_bounds,
    const std::vector<double>& upper_bounds,
    int num_particles = 30,
    int max_iterations = 100,
    double w = 0.7,  // 惯性权重
    double c1 = 1.5, // 个体学习因子
    double c2 = 1.5  // 社会学习因子
) {
    if (lower_bounds.size() != upper_bounds.size()) {
        throw std::invalid_argument("边界维度不匹配");
    }

    int dim = lower_bounds.size();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // 初始化粒子
    struct Particle {
        std::vector<double> position;
        std::vector<double> velocity;
        std::vector<double> best_position;
        double best_value;
    };

    std::vector<Particle> particles(num_particles);
    std::vector<double> global_best_position(dim);
    double global_best_value = std::numeric_limits<double>::infinity();

    // 初始化粒子
    for (auto& p : particles) {
        p.position.resize(dim);
        p.velocity.resize(dim);
        p.best_position.resize(dim);
        
        for (int i = 0; i < dim; ++i) {
            p.position[i] = lower_bounds[i] + dist(gen) * (upper_bounds[i] - lower_bounds[i]);
            p.velocity[i] = 0.0;
            p.best_position[i] = p.position[i];
        }
        
        p.best_value = objective(p.position);
        if (p.best_value < global_best_value) {
            global_best_value = p.best_value;
            global_best_position = p.best_position;
        }
    }

    // 主循环
    for (int iter = 0; iter < max_iterations; ++iter) {
        for (auto& p : particles) {
            // 更新速度
            for (int i = 0; i < dim; ++i) {
                double r1 = dist(gen);
                double r2 = dist(gen);
                p.velocity[i] = w * p.velocity[i] +
                               c1 * r1 * (p.best_position[i] - p.position[i]) +
                               c2 * r2 * (global_best_position[i] - p.position[i]);
            }

            // 更新位置
            for (int i = 0; i < dim; ++i) {
                p.position[i] += p.velocity[i];
                p.position[i] = std::max(lower_bounds[i], std::min(upper_bounds[i], p.position[i]));
            }

            // 更新最优解
            double current_value = objective(p.position);
            if (current_value < p.best_value) {
                p.best_value = current_value;
                p.best_position = p.position;
                if (current_value < global_best_value) {
                    global_best_value = current_value;
                    global_best_position = p.position;
                }
            }
        }
    }

    return global_best_position;
}

// 遗传算法
template<typename Func>
std::vector<double> genetic_algorithm(
    Func objective,
    const std::vector<double>& lower_bounds,
    const std::vector<double>& upper_bounds,
    int population_size = 100,
    int max_generations = 100,
    double mutation_rate = 0.1
) {
    if (lower_bounds.size() != upper_bounds.size()) {
        throw std::invalid_argument("边界维度不匹配");
    }

    int dim = lower_bounds.size();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // 初始化种群
    struct Individual {
        std::vector<double> genes;
        double fitness;
    };

    std::vector<Individual> population(population_size);
    std::vector<Individual> new_population(population_size);

    // 初始化种群
    for (auto& ind : population) {
        ind.genes.resize(dim);
        for (int i = 0; i < dim; ++i) {
            ind.genes[i] = lower_bounds[i] + dist(gen) * (upper_bounds[i] - lower_bounds[i]);
        }
        ind.fitness = objective(ind.genes);
    }

    // 主循环
    for (int gen = 0; gen < max_generations; ++gen) {
        // 选择
        std::sort(population.begin(), population.end(),
                 [](const Individual& a, const Individual& b) {
                     return a.fitness < b.fitness;
                 });

        // 精英保留
        for (int i = 0; i < population_size / 10; ++i) {
            new_population[i] = population[i];
        }

        // 交叉和变异
        for (int i = population_size / 10; i < population_size; i += 2) {
            // 选择父代
            int parent1 = std::floor(dist(gen) * population_size);
            int parent2 = std::floor(dist(gen) * population_size);

            // 交叉
            for (int j = 0; j < dim; ++j) {
                if (dist(gen) < 0.5) {
                    new_population[i].genes[j] = population[parent1].genes[j];
                    if (i + 1 < population_size) {
                        new_population[i + 1].genes[j] = population[parent2].genes[j];
                    }
                } else {
                    new_population[i].genes[j] = population[parent2].genes[j];
                    if (i + 1 < population_size) {
                        new_population[i + 1].genes[j] = population[parent1].genes[j];
                    }
                }
            }

            // 变异
            for (int j = 0; j < dim; ++j) {
                if (dist(gen) < mutation_rate) {
                    new_population[i].genes[j] = lower_bounds[j] + dist(gen) * (upper_bounds[j] - lower_bounds[j]);
                }
                if (i + 1 < population_size && dist(gen) < mutation_rate) {
                    new_population[i + 1].genes[j] = lower_bounds[j] + dist(gen) * (upper_bounds[j] - lower_bounds[j]);
                }
            }

            // 计算适应度
            new_population[i].fitness = objective(new_population[i].genes);
            if (i + 1 < population_size) {
                new_population[i + 1].fitness = objective(new_population[i + 1].genes);
            }
        }

        population = new_population;
    }

    // 返回最优解
    auto best = std::min_element(population.begin(), population.end(),
                               [](const Individual& a, const Individual& b) {
                                   return a.fitness < b.fitness;
                               });
    return best->genes;
}

} // namespace optimization
} // namespace mathlib 