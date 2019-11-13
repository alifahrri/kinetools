project(cppad-exercise)
cmake_minimum_required(VERSION 3.0.0)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(PkgConfig REQUIRED)
pkg_check_modules(cppad REQUIRED)

add_subdirectory(examples)