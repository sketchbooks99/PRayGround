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

# add_subdirectory(include/core)
# add_subdirectory(include/optix)

set_source_files_properties(
  ${CMAKE_CURRENT_SOURCE_DIR}/optix/create_obj.cu 
  PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ
)

OPTIX_add_sample_executable( pathtracer target_name

  # ===== INCLUDE =====
  ../include/core/aabb.h
  ../include/core/bsdf.h
  ../include/core/cudabuffer.h
  ../include/core/material.h
  ../include/core/onb.h
  ../include/core/pathtracer.h
  ../include/core/primitive.h
  ../include/core/scene.h
  ../include/core/shape.h
  ../include/core/stream_helpers.h
  ../include/core/util.h
  ../include/core/transform.h
  # ../include/core/texture.h

  ../include/optix/helpers.h
  ../include/optix/macros.h
  ../include/optix/module.h
  ../include/optix/pipeline.h
  ../include/optix/program.h
  ../include/optix/ray.h
  ../include/optix/sbt.h
  ../include/optix/util.h

  # ===== HEADER =====
  material/conductor.h 
  material/dielectric.h 
  material/diffuse.h
  material/emitter.h

  shape/sphere.h
  shape/trianglemesh.h 

  # ===== SOURCE =====
  # pathtracer.cpp
  
  pathtracer_fullscratch.cpp
  shape/trianglemesh.cpp

  # ===== CUDA =====
  optix/pathtracer.cu
  # optix/bsdf.cuh
  optix/create_obj.cu
  shape/optix/sphere.cuh
  shape/optix/trianglemesh.cuh

  # OPTIONS -rdc true
  )

target_link_libraries( ${target_name}
  ${CUDA_LIBRARIES}
)
