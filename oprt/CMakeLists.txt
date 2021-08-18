#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

set(oprt_target_name "oprt")
set(oprt_sources 
  # Core libraries ==========
  core/aabb.h 
  core/bitmap.cpp 
  core/bitmap.h 
  core/bsdf.h 
  core/camera.h 
  core/color.h 
  core/cudabuffer.h 
  core/emitter.h 
  core/file_util.h 
  core/file_util.cpp 
  core/film.h 
  core/film.cpp
  core/load3d.h 
  core/material.h 
  core/onb.h 
  # core/primitive.h 
  # core/primitive.cpp 
  core/ray.h 
  # core/scene.h 
  # core/scene.cpp 
  core/shape.h 
  core/stream_helpers.h 
  core/texture.h 
  core/transform.h 
  core/util.h
  core/shaders/bitmap.vert
  core/shaders/bitmap.frag

  # Math libraries ==========
  math/matrix.h

  # App libraries ==========
  app/baseapp.h 
  app/baseapp.cpp 
  app/event.h 
  app/window.h
  app/window.cpp
  app/app_runner.h 
  app/app_runner.cpp

  # OpenGL libraries ==========
  gl/shader.h 
  gl/shader.cpp

  # Optix wrapper libraries ==========
  optix/accel.h 
  optix/accel.cpp
  optix/context.h 
  optix/context.cpp 
  optix/helpers.h 
  optix/instance.h 
  optix/instance.cpp
  optix/macros.h 
  optix/module.h 
  optix/module.cpp
  optix/pipeline.h 
  optix/pipeline.cpp
  optix/program.h 
  optix/program.cpp 
  optix/sbt.h 
  optix/util.h 
  optix/cuda/oprt.cu 
  optix/cuda/texture.cuh

  # Shapes ==========
  shape/cylinder.h 
  shape/plane.h 
  shape/sphere.h 
  shape/trianglemesh.h
  shape/trianglemesh.cpp
  shape/cuda/cylinder.cuh 
  shape/cuda/plane.cuh 
  shape/cuda/sphere.cuh 
  shape/cuda/trianglemesh.cuh
 
  # Materials ==========
  material/conductor.h 
  material/dielectric.h 
  material/diffuse.h 
  material/disney.h

  # Textures ==========
  texture/bitmap.h
  texture/bitmap.cpp
  texture/checker.h 
  texture/constant.h  

  # Emitter ==========
  emitter/area.h 
  emitter/area.cpp 
  emitter/envmap.h
  emitter/envmap.cpp
  emitter/cuda/area.cuh 
  emitter/cuda/envmap.cuh
  
  oprt_dev.cpp
  oprt.h 
  params.h

  # OPTIONS -rdc true
)

# -------------------------------------------------------------------------------------
# The following declarations are copied from the function of OPTIX_add_sample_executable 
# defined in the CMakeLists.txt of OptiX/SDK, and modified to suit my oprt library.
# -------------------------------------------------------------------------------------

# These calls will group PTX and CUDA files into their own directories in the Visual
# Studio projects.
if (NOT CUDA_NVRTC_ENABLED)
  source_group("PTX Files"  REGULAR_EXPRESSION ".+\\.ptx$")
endif()
source_group("CUDA Files" REGULAR_EXPRESSION ".+\\.cuh?$")
source_group("Shader Files" REGULAR_EXPRESSION ".+\\.(vert|frag|geom|glsl|compute|comp)$")

# Separate the sources from the CMake and CUDA options fed to the macro.  This code
# comes from the CUDA_COMPILE_PTX macro found in FindCUDA.cmake.  We are copying the
# code here, so that we can use our own name for the target.  target_name is used in the
# creation of the output file names, and we want this to be unique for each target in
# the SDK.
CUDA_GET_SOURCES_AND_OPTIONS(source_files cmake_options options ${oprt_sources})

if (CUDA_NVRTC_ENABLED)

  # Isolate OBJ target files. NVCC should only process these files and leave PTX targets for NVRTC
  set(cu_obj_source_files)
  foreach(file ${source_files})
    get_source_file_property(_cuda_source_format ${file} CUDA_SOURCE_PROPERTY_FORMAT)
    if(${_cuda_source_format} MATCHES "OBJ")
      list(APPEND cu_obj_source_files ${file})
    endif()
  endforeach()

  # Create the rules to build the OBJ from the CUDA files.
  CUDA_WRAP_SRCS( ${oprt_target_name} OBJ generated_files ${cu_obj_source_files} ${cmake_options} OPTIONS ${options} )
else()

  # Create the rules to build the PTX and OBJ from the CUDA files.
  CUDA_WRAP_SRCS( ${oprt_target_name} PTX generated_files ${source_files} ${cmake_options} OPTIONS ${options} )
endif()


# Here is where we create the rule to make the executable.  We define a target name and
# list all the source files used to create the target.  In addition we also pass along
# the cmake_options parsed out of the arguments.
add_executable(${oprt_target_name}
  ${source_files}
  ${generated_files}
  ${cmake_options}
)

# Most of the samples link against the sutil library and the optix library.  Here is the
# rule that specifies this linkage.
target_link_libraries( ${oprt_target_name}
  ${GLFW_LIB_NAME}
  imgui
  sutil_7_sdk
  ${CUDA_LIBRARIES}
)

target_include_directories(${oprt_target_name} PUBLIC "${OPRT_DIR}/oprt")

set_target_properties( ${oprt_target_name} PROPERTIES
  COMPILE_DEFINITIONS
  "OPTIX_SAMPLE_NAME_DEFINE=${target_name};OPTIX_SAMPLE_DIR_DEFINE=${target_name}" )

if( UNIX AND NOT APPLE )
  # Force using RPATH instead of RUNPATH on Debian
  target_link_libraries( ${oprt_target_name} "-Wl,--disable-new-dtags" )
endif()

if(USING_GNU_CXX)
  target_link_libraries( ${oprt_target_name} m ) # Explicitly link against math library (C samples don't do that by default)
endif()

target_compile_definitions( 
  ${oprt_target_name}
  PUBLIC
  OPRT_ROOT_DIR="${PROJECT_SOURCE_DIR}"
)

target_compile_options(${oprt_target_name} PUBLIC "/source-charset:utf-8")
