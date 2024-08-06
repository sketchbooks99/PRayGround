cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

set(PROJECT_NAME "PRayGround")
project(${PROJECT_NAME} VERSION 0.0.1)

# Set the C++ standard to C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# OptiX SDK path must be parent directory of the SDK folder
option(OptiX_PATH "Path to the OptiX SDK" "")
if (NOT OptiX_PATH)
  message(FATAL_ERROR "Please specify the path to the OptiX SDK using -DOptiX_PATH")
endif()

set(CMAKE_MODULE_PATH "${OptiX_PATH}/SDK/CMake" ${CMAKE_MODULE_PATH})

include(Macros)
include(CompilerInfo)
include(ConfigCompilerFlags)

find_package(CUDA 11.0 REQUIRED)
mark_as_advanced(CLEAR CUDA_64_BIT_DEVICE_CODE)

set(OptiX_INSTALL_DIR "${OptiX_PATH}" CACHE PATH)
find_package(OptiX REQUIRED)

set(CUDA_NVRTC_FLAGS -std=c++17 -arch compute_60 -use_fast_math -lineinfo -default-device -rdc true -D__x86_64 CACHE STRING "Flags for NVRTC" FORCE)
mark_as_advanced(CUDA_NVRTC_FLAGS)

if(USING_WINDOWS_CL)
    list(APPEND CUDA_NVCC_FLAGS --compiler-options /D_USE_MATH_DEFINES)
endif()

set(CUDA_NVRTC_ENABLED ON CACHE BOOL "Use NVRTC for CUDA compilation at run-time (default ON)")

set(PRAYGROUND_PTX_DIR "${CMAKE_BINARY_DIR}/lib/ptx")
set(PRAYGROUND_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

set(CUDA_GENERATED_OUTPUT_DIR ${PRAYGROUND_PTX_DIR})

if (WIN32)
    string(REPLACE "/" "\\\\" PRAYGROUND_PTX_DIR ${PRAYGROUND_PTX_DIR})
endif()

set(PRAYGROUND_EXTERNAL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/prayground/ext")

# NVRTC include paths relative to the prayground path
set(PRAYGROUND_RELATIVE_INCLUDE_DIRS "\\
  \"prayground\", \\
  \"${PRAYGROUND_EXTERNAL_DIR}\", \\
  \"${CMAKE_CURRENT_SOURCE_DIR}\", ")

# NVRTC absolute include paths to the headers 
set(PRAYGROUND_ABSOLUTE_INCLUDE_DIRS "\\
  \"${OptiX_INCLUDE}\", \\
  \"${CUDA_INCLUDE_DIRS}\", ")