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

add_executable(
    ${PROJECT_NAME}_examples_exp2
    src/cppad_exp2.cpp
)

target_link_libraries(
    ${PROJECT_NAME}_examples_exp2
    ${cppad_LIBRARIES}
)

target_include_directories(
    ${PROJECT_NAME}_examples_exp2 PUBLIC
    ${cppad_INCLUDE_DIRS}
)

add_executable(
    ${PROJECT_NAME}_examples_jacobian
    src/jacobian.cpp
)

target_link_libraries(
    ${PROJECT_NAME}_examples_jacobian
    ${cppad_LIBRARIES}
)

target_include_directories(
    ${PROJECT_NAME}_examples_jacobian PUBLIC
    ${cppad_INCLUDE_DIRS}
)

add_executable(
    ${PROJECT_NAME}_examples_dh_matrices
    src/dh_matrices.cpp
)

target_link_libraries(
    ${PROJECT_NAME}_examples_dh_matrices
    ${cppad_LIBRARIES}
    Eigen3::Eigen
)

target_include_directories(
    ${PROJECT_NAME}_examples_dh_matrices PUBLIC
    ${cppad_INCLUDE_DIRS}
)

target_compile_options(
    ${PROJECT_NAME}_examples_dh_matrices
    PRIVATE
    "-g"
)

set_target_properties(
    ${PROJECT_NAME}_examples_dh_matrices
    PROPERTIES
    CXX_STANDARD 17
)

add_executable(
    ${PROJECT_NAME}_examples_scara_jacobian
    src/scara_jacobian.cpp
)

target_link_libraries(
    ${PROJECT_NAME}_examples_scara_jacobian
    ${cppad_LIBRARIES}
    Eigen3::Eigen
)

target_include_directories(
    ${PROJECT_NAME}_examples_scara_jacobian PUBLIC
    ${cppad_INCLUDE_DIRS}
)

target_compile_options(
    ${PROJECT_NAME}_examples_scara_jacobian
    PRIVATE
    "-g"
)

set_target_properties(
    ${PROJECT_NAME}_examples_scara_jacobian
    PROPERTIES
    CXX_STANDARD 17
)

add_executable(
    ${PROJECT_NAME}_examples_scara
    src/scara.cpp
)

target_link_libraries(
    ${PROJECT_NAME}_examples_scara
    ${cppad_LIBRARIES}
    Eigen3::Eigen
)

target_include_directories(
    ${PROJECT_NAME}_examples_scara PUBLIC
    ${cppad_INCLUDE_DIRS}
)

target_compile_options(
    ${PROJECT_NAME}_examples_scara
    PRIVATE
    "-g"
)

set_target_properties(
    ${PROJECT_NAME}_examples_scara
    PROPERTIES
    CXX_STANDARD 17
)

add_executable(
    ${PROJECT_NAME}_examples_scara_inverse
    src/scara_inverse.cpp
)

target_link_libraries(
    ${PROJECT_NAME}_examples_scara_inverse
    ${cppad_LIBRARIES}
    Eigen3::Eigen
)

target_include_directories(
    ${PROJECT_NAME}_examples_scara_inverse PUBLIC
    ${cppad_INCLUDE_DIRS}
)

target_compile_options(
    ${PROJECT_NAME}_examples_scara_inverse
    PRIVATE
    "-g"
)

set_target_properties(
    ${PROJECT_NAME}_examples_scara_inverse
    PROPERTIES
    CXX_STANDARD 17
)