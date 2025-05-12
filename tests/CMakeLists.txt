cmake_minimum_required(VERSION 3.10)

# 查找Google Test包
find_package(GTest REQUIRED)

# 添加测试可执行文件
add_executable(mathlib_tests
    statistics_test.cpp
    probability_test.cpp
    combinatorics_test.cpp
)

# 链接Google Test和我们的库
target_link_libraries(mathlib_tests
    GTest::GTest
    GTest::Main
    mathlib
)

# 添加测试
add_test(NAME mathlib_tests COMMAND mathlib_tests) 