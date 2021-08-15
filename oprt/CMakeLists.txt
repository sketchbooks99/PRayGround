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

OPRT_add_executable( oprt target_name

  # Core libraries ==========
  core/aabb.h 
  # core/api.h 
  # core/api.cpp 
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
  core/scene.h 
  core/scene.cpp 
  core/shape.h 
  core/stream_helpers.h 
  core/texture.h 
  core/transform.h 
  core/util.h
  core/shaders/bitmap.vert
  core/shaders/bitmap.frag

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
  scene_config.h

  # OPTIONS -rdc true
)

target_compile_definitions( 
  ${target_name} 
  PRIVATE 
  OPRT_ROOT_DIR="${PROJECT_SOURCE_DIR}"
)

target_link_libraries( ${target_name}
  ${CUDA_LIBRARIES}
)
