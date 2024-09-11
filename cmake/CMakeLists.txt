cmake_minimum_required(VERSION 3.10)

set(CMAKE_C_COMPILER C:/dev/msys64/mingw64/bin/gcc.exe)
set(CMAKE_CXX_COMPILER C:/dev/msys64/mingw64/bin/g++.exe)

project(mindspace)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY bin)

set(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
set (INC_ROOT_DIR ${PROJECT_ROOT_DIR}/inc)
set (SRC_ROOT_DIR ${PROJECT_ROOT_DIR}/src)
set (TEST_ROOT_DIR ${PROJECT_ROOT_DIR}/test)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")


set (HEADERS
  ${INC_ROOT_DIR}/tensor
  ${INC_ROOT_DIR}/block
  ${INC_ROOT_DIR}/model
  ${INC_ROOT_DIR}/train
  ${TEST_ROOT_DIR}/inc/unit_test
  ${TEST_ROOT_DIR}/inc/benchmark
)
include_directories(${HEADERS})

file (GLOB_RECURSE SRC ${SRC_ROOT_DIR}/*.cpp)
file (GLOB_RECURSE TEST_SRC % {TEST_ROOT_DIR}/*.cpp)

set(SOURCES
  ${SRC}
  ${TEST_SRC}
)

add_executable(main ${SOURCES})


#commands to build this:
# (in cmake folder)
#mkdir build
#cd build
#cmake -G Ninja ..
#ninja
