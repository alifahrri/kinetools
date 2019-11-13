add_executable(
    ${PROJECT_NAME}_examples_get_started
    src/get_started.cpp
)

target_link_libraries(
    ${PROJECT_NAME}_examples_get_started
    ${cppad_LIBRARIES}
)

target_include_directories(
    ${PROJECT_NAME}_examples_get_started PUBLIC
    ${cppad_INCLUDE_DIRS}
)

add_executable(
    ${PROJECT_NAME}_examples_eigen_array
    src/eigen_array.cpp
)

target_link_libraries(
    ${PROJECT_NAME}_examples_eigen_array
    ${cppad_LIBRARIES}
    Eigen3::Eigen
)

target_include_directories(
    ${PROJECT_NAME}_examples_eigen_array PUBLIC
    ${cppad_INCLUDE_DIRS}
)