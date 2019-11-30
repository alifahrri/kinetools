project(kinetools)
cmake_minimum_required(VERSION 3.0.0)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(PkgConfig REQUIRED)
pkg_check_modules(cppad REQUIRED)

include_directories(
    include
)

option(BUILD_TESTS "Build test programs" ON)
option(BUILD_EXAMPLES "Build example programs" ON)

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif(BUILD_EXAMPLES)

if(BUILD_TESTS)
    add_subdirectory(tests)
endif(BUILD_TESTS)