cmake_minimum_required(VERSION 3.15)

project(AG VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/W4 /WX-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_library(ag STATIC
    src/value.cpp
    src/tensor/tensor/tensor.cpp
    src/tensor/tensor/transpose.cpp
    src/tensor/ops/reductions.cpp
    src/tensor/ops/elementwise.cpp
    src/tensor/ops/matops.cpp
)

target_include_directories(ag
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_compile_features(ag PUBLIC cxx_std_23)

option(BUILD_TESTS "Build the unit tests" ON)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

option(BUILD_EXAMPLES "Build the example programs" OFF)

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
