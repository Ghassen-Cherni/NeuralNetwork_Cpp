cmake_minimum_required(VERSION 3.10)

project(NeuralNetwork)

# Set the output directory for the executable
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

# Add Eigen include directories
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)

# Add your source files
add_executable(NeuralNetwork main.cpp neural_network.cpp utils.cpp)

# Link against Eigen, if necessary
target_link_libraries(NeuralNetwork Eigen3::Eigen)

