cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

set(PROJECT_NAME "PRayGround")
project(${PROJECT_NAME} VERSION 0.0.1)

# Set the C++ standard to C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# OptiX SDK path must be parent directory of the SDK folder
set(OptiX_PATH "" CACHE PATH "Path to the OptiX SDK")
if (NOT OptiX_PATH)
  message(FATAL_ERROR "Please specify the path to the OptiX SDK using -DOptiX_PATH")
endif()

set(CMAKE_MODULE_PATH "${OptiX_PATH}/SDK/CMake" ${CMAKE_MODULE_PATH})

option(BUILD_SHARED_LIBS "Build shared libraries" ON)

# Enable C++11. Needs to be done before the include of ConfigCompilerFlags.cmake below.
set(GCC_LIBSTDCPP11 ON)

# Include all CMake Macros.
include(Macros)
# Determine information about the compiler
include (CompilerInfo)
# Check for specific machine/compiler options.
include (ConfigCompilerFlags)

find_package(CUDA 11.0 REQUIRED)
mark_as_advanced(CLEAR CUDA_64_BIT_DEVICE_CODE)

set(OptiX_INSTALL_DIR "${OptiX_PATH}" CACHE PATH "Path to the OptiX SDK")
find_package(OptiX REQUIRED)

# Set the CUDA flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++17 -arch sm_60 --use_fast_math -lineinfo CACHE STRING "Flags for NVCC" FORCE)
set(CUDA_NVRTC_FLAGS ${PRAYGROUND_NVRTC_CXX} -arch compute_60 --use_fast_math -lineinfo -default-device -rdc true -D__x86_64 CACHE STRING "Semi-colon delimit multiple arguments." FORCE)
mark_as_advanced(CUDA_NVRTC_FLAGS)

if(USING_WINDOWS_CL)
    list(APPEND CUDA_NVCC_FLAGS --compiler-options /D_USE_MATH_DEFINES)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

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

set(CUDA_NVRTC_OPTIONS)
foreach(flag ${CUDA_NVRTC_FLAGS})
  set(CUDA_NVRTC_OPTIONS "${CUDA_NVRTC_OPTIONS} \\\n  \"${flag}\",")
endforeach()
set(CUDA_NVRTC_OPTIONS "${CUDA_NVRTC_OPTIONS}")

configure_file(prayground_config.h.in prayground_config.h @ONLY)

include_directories( ${CMAKE_CURRENT_SOURCE_DIR}
                     "${CMAKE_BINARY_DIR}/include"
                     ${CMAKE_CURRENT_BINARY_DIR}
                     ${CUDA_INCLUDE_DIRS}
                     ${OptiX_INCLUDE}
                     )

function(PRAYGROUND_add_executable target_name_base target_name_var)
  set(target_name ${target_name_base})
  set(${target_name_var} ${target_name} PARENT_SCOPE)

  if (NOT CUDA_NVRTC_ENABLED)
    source_group("PTX Files" REGULAR_EXPRESSION ".+\\.ptx$")
  endif()
  source_group("CUDA Files" REGULAR_EXPRESSION ".+\\.cu$")
  source_group("Shader Files" REGULAR_EXPRESSION ".+\\.(vert|frag|geom|glsl|compute|comp)$")

  CUDA_GET_SOURCES_AND_OPTIONS(source_files cmake_options options ${ARGN})
  if (CUDA_NVRTC_ENABLED)
    set(cu_obj_source_files)
    foreach(file ${source_files})
      get_source_file_property(_cuda_source_format ${file} CUDA_SOURCE_PROPERTY_FORMAT)
      if(${_cuda_source_format} MATCHES "OBJ")
        list(APPEND cu_obj_source_files ${file})
      endif()
    endforeach()
    CUDA_WRAP_SRCS( ${target_name} OBJ generated_files ${cu_obj_source_files} ${cmake_options} OPTIONS ${options} )
  else()
    CUDA_WRAP_SRCS( ${target_name} PTX generated_files ${source_files} ${cmake_options} OPTIONS ${options} )
  endif()

  add_executable(${target_name} ${source_files} ${generated_files} ${cmake_options})

  target_include_directories(${target_name} PRIVATE
    "${PRAYGROUND_DIR}"
    "${CMAKE_BINARY_DIR}/lib"
    "${OptiX_INCLUDE}")
  
  target_link_libraries(${target_name}
    # ${GLFW_LIB_NAME}
    imgui
    nanovdb
    prayground
  )

  if (WIN32)
    target_compile_options(${target_name} PUBLIC "/source-charset:utf-8")
  endif()

  if( UNIX AND NOT APPLE )
    # Force using RPATH instead of RUNPATH on Debian
    target_link_libraries( ${target_name} "-Wl,--disable-new-dtags" )
  endif()

  if(USING_GNU_CXX)
    target_link_libraries( ${target_name} m ) # Explicitly link against math library (C samples don't do that by default)
  endif()
endfunction()

# External libraries
add_subdirectory(prayground/ext)

# PRayGround backend library
add_subdirectory(prayground)

################################################################
# Example apps 
# 
# Please uncomment the target you'd like to build
################################################################
add_subdirectory(examples/single_gas)
add_subdirectory(examples/dynamic_update)
add_subdirectory(examples/motion_blur)
add_subdirectory(examples/pathtracing)
add_subdirectory(examples/objscene)
add_subdirectory(examples/rayTracingNextWeek)
add_subdirectory(examples/rayTracingRestOfYourLife)
add_subdirectory(examples/denoiser)
add_subdirectory(examples/thumbnail)
add_subdirectory(examples/spectrum)
add_subdirectory(examples/pgVolume)
add_subdirectory(examples/curves)
add_subdirectory(examples/opacityMicromap)
add_subdirectory(examples/fluidSimulation)

################################################################
# Applications
# 
# Please add your custom applications to build here
################################################################
# add_subdirectory(apps/empty_app)

################################################################
# Unit test
################################################################
# add_subdirectory(tests/math)
# add_subdirectory(tests/core)
# add_subdirectory(tests/thrust)
# add_subdirectory(tests/primitives)

set(PASSED_FIRST_CONFIGURE ON CACHE INTERNAL "Already Configured once?")