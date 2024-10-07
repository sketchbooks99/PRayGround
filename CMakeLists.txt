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

# NVRTC support
set(CUDA_NVRTC_ENABLED OFF CACHE BOOL "Use NVRTC for CUDA compilation at run-time (default OFF)")
set( CUDA_NVRTC_ENABLED OFF CACHE BOOL "Use NVRTC to compile OPTIXIR/PTX at run-time instead of NVCC at build-time" )
if( NOT ( NOT ( NOT CUDA_NVRTC_ENABLED ) ) EQUAL ( NOT ( NOT CUDA_NVRTC_ENABLED_INTERNAL ) ) )
  message( STATUS "Resetting OPTIXIR/PTX support" )
  unset( PRAYGROUND_INPUT_ENABLE_OPTIXIR_SUPPORT CACHE )
endif()
set( CUDA_NVRTC_ENABLED_INTERNAL ${CUDA_NVRTC_ENABLED} CACHE INTERNAL "Previous configured value (NVRTC)" FORCE )

if( CUDA_NVRTC_ENABLED )
  if( CUDA_VERSION VERSION_LESS 12.0 )
    if( PRAYGROUND_INPUT_ENABLE_OPTIXIR_SUPPORT )
      message( SEND_ERROR "CUDA_NVRTC_ENABLED is not compatible with PRAYGROUND_INPUT_ENABLE_OPTIXIR_SUPPORT in CUDA versions less than 12.0" )
    else()
      option( PRAYGROUND_INPUT_ENABLE_OPTIXIR_SUPPORT "Enable support for generating OptiX-IR targetted input files" OFF )
    endif()
  else()
    option( PRAYGROUND_INPUT_ENABLE_OPTIXIR_SUPPORT "Enable support for generating OptiX-IR targetted input files" ON )
  endif()
else()
  if( CUDA_VERSION VERSION_LESS 11.7 )
    if( PRAYGROUND_INPUT_ENABLE_OPTIXIR_SUPPORT )
      message( SEND_ERROR "PRAYGROUND_INPUT_ENABLE_OPTIXIR_SUPPORT is not supported in CUDA versions less than 11.7" )
    else()
      option( PRAYGROUND_INPUT_ENABLE_OPTIXIR_SUPPORT "Enable support for generating OptiX-IR targetted input files" OFF )
    endif()
  else()
    option( PRAYGROUND_INPUT_ENABLE_OPTIXIR_SUPPORT "Enable support for generating OptiX-IR targetted input files" ON )
  endif()
endif()

# CMake doesn't have an equality operator for boolean types (only integer and string). 
# By doing NOT NOT VAL, you can force the original value into 0
# or 1 and allow the EQUAL operator to function correctly.
if( NOT ( NOT ( NOT PRAYGROUND_INPUT_ENABLE_OPTIXIR_SUPPORT ) ) EQUAL ( NOT ( NOT PRAYGROUND_INPUT_ENABLE_OPTIXIR_SUPPORT_INTERNAL ) ) )
  message( STATUS "Unsetting values associated with OptiX code generation" )
  # This allows us to reset dependent options if you change it.
  unset( PRAYGROUND_INPUT_GENERATE_OPTIXIR CACHE )
  unset( PRAYGROUND_INPUT_GENERATE_PTX CACHE )
  unset( GENERATE_DEBUG_DEVICE_CODE CACHE )
endif()
set(PRAYGROUND_INPUT_ENABLE_OPTIXIR_SUPPORT_INTERNAL ${PRAYGROUND_INPUT_ENABLE_OPTIXIR_SUPPORT} CACHE INTERNAL "Previous configured value (OPTIXIR)" FORCE)

if( PRAYGROUND_INPUT_ENABLE_OPTIXIR_SUPPORT )
  option( PRAYGROUND_INPUT_GENERATE_OPTIXIR "Generate Optix-IR OptiX shaders" ON )
  option( PRAYGROUND_INPUT_GENERATE_PTX "Generate PTX OptiX shaders" OFF )
else()
  option( PRAYGROUND_INPUT_GENERATE_OPTIXIR "Generate Optix-IR OptiX shaders" OFF )
  option( PRAYGROUND_INPUT_GENERATE_PTX "Generate PTX OptiX shaders" ON )
endif()

# Determine if we are going to use the static CRT on windows.
if(WIN32)
  option(RELEASE_USE_STATIC_CRT "Build using the static CRT library" ON)
endif()

mark_as_advanced(CLEAR CUDA_64_BIT_DEVICE_CODE)

set(CUDA_MIN_SM_TARGET sm_86 CACHE STRING "Minimum CUDA SM architecture to use for compilation.")

function(optix_add_cuda_flag_config config flag)
  string(TOUPPER "${config}" config)
  list(FIND CUDA_NVCC_FLAGS${config} ${flag} index)
  if(index EQUAL -1)
    list(APPEND CUDA_NVCC_FLAGS${config} ${flag})
    set(CUDA_NVCC_FLAGS${config} ${CUDA_NVCC_FLAGS${config}} CACHE STRING ${CUDA_NVCC_FLAGS_DESCRIPTION} FORCE)
  endif()
endfunction()

function(optix_add_cuda_flag flag)
  optix_add_cuda_flag_config( "" ${flag} )
endfunction()

# Add some useful default arguments to the NVCC and NVRTC flags.  This is an example of
# how we use PASSED_FIRST_CONFIGURE.  Once you have configured, this variable is TRUE
# and following block of code will not be executed leaving you free to edit the values
# as much as you wish from the GUI or from ccmake.
if( NOT PASSED_FIRST_CONFIGURE )
  set(CUDA_NVCC_FLAGS_DESCRIPTION "Semi-colon delimit multiple arguments.")
  string(REPLACE "sm_" "compute_" CUDA_MIN_SM_COMPUTE_TARGET ${CUDA_MIN_SM_TARGET})

  list(FIND CUDA_NVCC_FLAGS "-arch" index)
  message(STATUS "Index = ${index}")
  if(index EQUAL -1)
    list(APPEND CUDA_NVCC_FLAGS -arch ${CUDA_MIN_SM_TARGET})
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} CACHE STRING "Semi-colon delimit multiple arguments." FORCE)
  endif()

  optix_add_cuda_flag("--use_fast_math")
  optix_add_cuda_flag("-lineinfo")
  if( DEFINED CMAKE_CONFIGURATION_TYPES )
    foreach( config ${CMAKE_CONFIGURATION_TYPES} )
      if( ${config} STREQUAL "Debug" )
        optix_add_cuda_flag_config( _${config} "-G" )
        optix_add_cuda_flag_config( _${config} "-O0" )
      endif()
    endforeach()
  else()
    if( CMAKE_BUILD_TYPE STREQUAL "Debug" )
      optix_add_cuda_flag( "-G" )
      optix_add_cuda_flag( "-O0" )
    endif()
  endif()

  if( CUDA_VERSION VERSION_LESS "3.0" )
    optix_add_cuda_flag("--keep")
  endif()

  # Some CUDA 11.x toolkits erroneously complain about sm_50 being deprecated
  if(CUDA_VERSION VERSION_GREATER "11.0")
    optix_add_cuda_flag("-Wno-deprecated-gpu-targets")
  endif()

  if(CUDA_USE_LOCAL_ENV)
    optix_add_cuda_flag("--use-local-env")
  endif()

  set(PRAYGROUND_NVRTC_CXX "-std=c++17")

  if( NOT DEFINED CMAKE_CONFIGURATION_TYPES )
    if( NOT CMAKE_BUILD_TYPE STREQUAL CMAKE_BUILD_TYPE_PREVIOUS )
      message( STATUS "Resetting CUDA_NVRTC_FLAGS" )
      unset( CUDA_NVRTC_FLAGS CACHE )
    endif()
    set( CMAKE_BUILD_TYPE_PREVIOUS ${CMAKE_BUILD_TYPE} CACHE INTERNAL "Previous configured value (CMAKE_BUILD_TYPE)" FORCE )

    set( configs "Debug" "Release" "RelWithDebInfo" "MinSizeRel" )
    foreach( config ${configs} )
      if( ${config} STREQUAL "Debug" )
        set( PRAYGROUND_NVRTC_DEBUG "-G" )
      else()
        set( PRAYGROUND_NVRTC_DEBUG "-lineinfo" )
      endif()

      string( TOUPPER ${config} config_upper )
      set( CUDA_NVRTC_FLAGS_${config_upper} ${PRAYGROUND_NVRTC_CXX} -arch ${CUDA_MIN_SM_COMPUTE_TARGET} ${PRAYGROUND_NVRTC_DEBUG} -use_fast_math -default-device -rdc true -D__x86_64 CACHE STRING "List of NVRTC options just for the PRAYGROUND" FORCE )

      if( ${config} STREQUAL ${CMAKE_BUILD_TYPE} )
        set( CUDA_NVRTC_FLAGS ${CUDA_NVRTC_FLAGS_${config_upper}} CACHE STRING "List of NVRTC options just for the PRAYGROUND" )
      endif()
    endforeach()
  else()
    set( CUDA_NVRTC_FLAGS ${PRAYGROUND_NVRTC_CXX} -arch ${CUDA_MIN_SM_COMPUTE_TARGET} -lineinfo -use_fast_math -default-device -rdc true -D__x86_64 CACHE STRING "List of NVRTC options just for the PRAYGROUND" FORCE )
    set( CUDA_NVRTC_FLAGS_DEBUG ${PRAYGROUND_NVRTC_CXX} -arch ${CUDA_MIN_SM_COMPUTE_TARGET} -G -use_fast_math -default-device -rdc true -D__x86_64 CACHE STRING "List of NVRTC options just for the PRAYGROUND" FORCE )
  endif()    
endif()

message(STATUS "CUDA_NVCC_FLAGS = ${CUDA_NVCC_FLAGS}")

mark_as_advanced(CUDA_NVRTC_FLAGS)

# This passes a preprocessor definition to cl.exe when processing CUDA code.
if(USING_WINDOWS_CL)
  list(APPEND CUDA_NVCC_FLAGS --compiler-options /D_USE_MATH_DEFINES)
endif()

# Select whether to use NVRTC or NVCC to generate PTX
if( NOT PRAYGROUND_INPUT_ENABLE_OPTIXIR_SUPPORT AND PRAYGROUND_INPUT_GENERATE_OPTIXIR )
  message( SEND_ERROR "Must enable PRAYGROUND_INPUT_ENABLE_OPTIXIR_SUPPORT to enable PRAYGROUND_INPUT_GENERATE_OPTIXIR" )
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set(PRAYGROUND_PTX_DIR "${CMAKE_BINARY_DIR}/bin/Release/ptx")
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

# Build a null-terminated option list for NVRTC
set( config_suffixes "_RELEASE" "_DEBUG" )
foreach( config_suffix ${config_suffixes} )
  # CMake doesn't allow empty strings in lists, so use a dummy suffix
  if( ${config_suffix} STREQUAL "_RELEASE" )
    set( config_suffix "" )
  endif()
  set(CUDA_NVRTC_OPTIONS${config_suffix})
  foreach(flag ${CUDA_NVRTC_FLAGS${config_suffix}})
    set(CUDA_NVRTC_OPTIONS${config_suffix} "${CUDA_NVRTC_OPTIONS${config_suffix}} \\\n  \"${flag}\",")
  endforeach()
  set(CUDA_NVRTC_OPTIONS${config_suffix} "${CUDA_NVRTC_OPTIONS${config_suffix}}")
endforeach()

configure_file(prayground_config.h.in prayground_config.h @ONLY)

include_directories( ${CMAKE_CURRENT_SOURCE_DIR}
                     "${CMAKE_BINARY_DIR}/include"
                     ${CMAKE_CURRENT_BINARY_DIR}
                     ${CUDA_INCLUDE_DIRS}
                     ${OptiX_INCLUDE}
                     )

# Helper macro to generate PTX from the CUDA files in prayground.
macro(OPTIX_prayground_compile_to_optix_input generated_files)
  if( NOT CUDA_NVRTC_ENABLED )
    if( PRAYGROUND_INPUT_GENERATE_OPTIXIR )
      message("prayground OPTIXIR")
      CUDA_WRAP_SRCS( prayground OPTIXIR generated_files2 ${ARGN} )
      list(APPEND ${generated_files} ${generated_files2})
    endif()
    if( PRAYGROUND_INPUT_GENERATE_PTX )
      message("prayground PTX")
      CUDA_WRAP_SRCS( prayground PTX generated_files3 ${ARGN} )
      list(APPEND ${generated_files} ${generated_files3})
    endif()
  endif()
  message("${generated_files} = ${${generated_files}}")
endmacro()

# These calls will group PTX and CUDA files into their own directories in the Visual
# Studio projects.
macro(OPTIX_add_source_groups)
  if( NOT CUDA_NVRTC_ENABLED )
    if( PRAYGROUND_INPUT_GENERATE_PTX )
      source_group("PTX Files"  REGULAR_EXPRESSION ".+\\.ptx$")
    endif()
    if( PRAYGROUND_INPUT_GENERATE_OPTIXIR )
      source_group("OptixIR Files"  REGULAR_EXPRESSION ".+\\.optixir$")
    endif()
  endif()
  source_group("CUDA Files" REGULAR_EXPRESSION ".+\\.cu$")
endmacro()

function(PRAYGROUND_add_executable target_name_base target_name_var)
  set( target_name ${target_name_base} )
  set( ${target_name_var} ${target_name} PARENT_SCOPE )

  OPTIX_add_source_groups()

  # if (NOT CUDA_NVRTC_ENABLED)
  #   source_group("PTX Files" REGULAR_EXPRESSION ".+\\.ptx$")
  # endif()
  # source_group("CUDA Files" REGULAR_EXPRESSION ".+\\.cu$")
  # source_group("Shader Files" REGULAR_EXPRESSION ".+\\.(vert|frag|geom|glsl|compute|comp)$")

  # message(STATUS "${CUDA_NVCC_FLAGS}")
  CUDA_GET_SOURCES_AND_OPTIONS(source_files cmake_options options ${ARGN})

  # Isolate OBJ target files. NVCC should only process these files and leave PTX targets for NVRTC
  set(cu_obj_source_files)
  set(cu_optix_source_files)
  foreach(file ${source_files})
    get_source_file_property(_cuda_source_format ${file} CUDA_SOURCE_PROPERTY_FORMAT)
    if(${_cuda_source_format} MATCHES "OBJ")
      list(APPEND cu_obj_source_files ${file})
    else()
      list(APPEND cu_optix_source_files ${file})
    endif()
  endforeach()

  CUDA_WRAP_SRCS( ${target_name} OBJ generated_files ${cu_obj_source_files} ${cmake_options} OPTIONS ${options} )

  # Create the rules to build the PTX and/or OPTIX files.
  if( PRAYGROUND_INPUT_GENERATE_OPTIXIR )
    CUDA_WRAP_SRCS( ${target_name} OPTIXIR generated_files2 ${cu_optix_source_files} ${cmake_options} OPTIONS ${options} )
    list(APPEND generated_files ${generated_files2})
  endif()
  if( PRAYGROUND_INPUT_GENERATE_PTX AND NOT CUDA_NVRTC_ENABLED)
    CUDA_WRAP_SRCS( ${target_name} PTX generated_files3 ${cu_optix_source_files} ${cmake_options} OPTIONS ${options} )
    list(APPEND generated_files ${generated_files3})
  endif()

  add_executable(${target_name}
    ${source_files}
    ${generated_files}
    ${cmake_options}
    )

  target_include_directories(${target_name} PRIVATE
    "${PRAYGROUND_DIR}"
    "${CMAKE_BINARY_DIR}/bin"
    "${OptiX_INCLUDE}")
  
  target_link_libraries(${target_name}
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
# add_subdirectory(examples/single_gas)
# add_subdirectory(examples/dynamic_update)
# add_subdirectory(examples/motion_blur)
# add_subdirectory(examples/pathtracing)
# add_subdirectory(examples/objscene)
# add_subdirectory(examples/rayTracingNextWeek)
# add_subdirectory(examples/rayTracingRestOfYourLife)
# add_subdirectory(examples/denoiser)
# add_subdirectory(examples/thumbnail)
# add_subdirectory(examples/spectrum)
# add_subdirectory(examples/pgVolume)
# add_subdirectory(examples/curves)
# add_subdirectory(examples/opacityMicromap)
# add_subdirectory(examples/fluidSimulation)

################################################################
# Applications
# 
# Please add your custom applications to build here
################################################################
# add_subdirectory(apps/empty_app)
add_subdirectory(apps/rtcamp10)

################################################################
# Unit test
################################################################
# add_subdirectory(tests/math)
# add_subdirectory(tests/core)
# add_subdirectory(tests/thrust)
# add_subdirectory(tests/primitives)

set(PASSED_FIRST_CONFIGURE ON CACHE INTERNAL "Already Configured once?")