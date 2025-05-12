cmake_minimum_required(VERSION 3.10)
project(CPlusPlusMathematicsLibrary VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加头文件目录
include_directories(include)

# 添加源文件目录
add_subdirectory(src)

# 添加测试目录
enable_testing()
add_subdirectory(tests)

# 安装规则
install(DIRECTORY include/ DESTINATION include)
install(TARGETS mathlib DESTINATION lib) 