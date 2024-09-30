cmake_minimum_required(VERSION 3.10)

set(CMAKE_C_COMPILER C:/dev/msys64/mingw64/bin/gcc.exe)
set(CMAKE_CXX_COMPILER C:/dev/msys64/mingw64/bin/g++.exe)

project(mindspace)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY bin)

set(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")


set (HEADERS
  ${PROJECT_ROOT_DIR}/tensor/inc
  ${PROJECT_ROOT_DIR}/block/inc
  ${PROJECT_ROOT_DIR}/test/unit_test/inc
  ${PROJECT_ROOT_DIR}/test/benchmark/inc
)
include_directories(${HEADERS})

file (GLOB_RECURSE TENSOR_SRC ${PROJECT_ROOT_DIR}/tensor/src/*.cpp)
file (GLOB_RECURSE CPU_ENGINE_SRC ${PROJECT_ROOT_DIR}/engine/cpu/*.cpp)
file (GLOB_RECURSE CUDA_ENGINE_SRC ${PROJECT_ROOT_DIR}/engine/cuda/*.cpp)
file (GLOB_RECURSE BLOCK_SRC ${PROJECT_ROOT_DIR}/block/src/*.cpp)
file (GLOB_RECURSE UNIT_TEST_SRC ${PROJECT_ROOT_DIR}/test/unit_test/src/*.cpp)
file (GLOB_RECURSE BENCHMARK_SRC ${PROJECT_ROOT_DIR}/test/benchmark/src/*.cpp)


set(SOURCES
  ${PROJECT_ROOT_DIR}/main.cpp
  ${TENSOR_SRC}
  ${BLOCK_SRC}
  ${CPU_ENGINE_SRC}
  ${CUDA_ENGINE_SRC}
  ${UNIT_TEST_SRC}
  ${BENCHMARK_SRC}
)

add_executable(main ${SOURCES})


#commands to build this:
# (in cmake folder)
#mkdir build
#cd build
#cmake -G Ninja ..
#ninja
