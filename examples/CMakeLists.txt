cmake_minimum_required(VERSION 3.10)
project(mathlib_examples)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加示例程序
add_executable(statistics_example statistics_example.cpp)
add_executable(probability_example probability_example.cpp)
add_executable(combinatorics_example combinatorics_example.cpp)

# 链接数学库
target_link_libraries(statistics_example PRIVATE mathlib)
target_link_libraries(probability_example PRIVATE mathlib)
target_link_libraries(combinatorics_example PRIVATE mathlib)

# 设置包含目录
target_include_directories(statistics_example PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(probability_example PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(combinatorics_example PRIVATE ${CMAKE_SOURCE_DIR}/include) 