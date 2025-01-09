cmake_minimum_required(VERSION 3.10)
project(NeuralNetworkLayers)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/Operators/include
    ${CMAKE_SOURCE_DIR}/Test_Operators/include
    ${CMAKE_SOURCE_DIR}/utilities/include
)
# Add main executable
add_executable(main
    src/main.cpp
    Operators/src/conv.cpp
    Operators/src/relu.cpp
    Operators/src/dense.cpp
    Operators/src/flatten.cpp
    Operators/src/max_pool.cpp
    Operators/src/softmax.cpp
    utilities/src/file_utils.cpp
)
# Add test executable
# Add test executable
add_executable(test_operators
    Test_Operators/src/main_test.cpp
    Test_Operators/src/test.cpp
    utilities/src/file_utils.cpp
)

