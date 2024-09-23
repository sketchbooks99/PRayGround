set(sources
  # Core libraries ==========
  core/aabb.h 
  core/attribute.h 
  core/attribute.cpp
  core/bitmap.cpp 
  core/bitmap.h 
  core/bsdf.h 
  core/camera.h 
  core/camera.cpp
  core/cexpr_map.h
  core/spectrum.h 
  core/cudabuffer.h 
  core/emitter.h 
  core/file_util.h 
  core/file_util.cpp 
  core/interaction.h
  core/load3d.h 
  core/load3d.cpp
  core/material.h 
  core/onb.h 
  core/ray.h 
  core/sampling.h
  core/sampling.cpp
  core/sampler.h
  core/shape.h 
  core/shape.cpp
  core/scene.h 
  core/stream_helpers.h 
  core/texture.h 
  core/util.h
  core/shaders/bitmap.vert
  core/shaders/bitmap.frag

  # Math libraries ==========
  math/matrix.h
  math/noise.h
  math/vec.h
  math/vec_math.h
  math/util.h
  math/random.h

  # App libraries ==========
  app/baseapp.h 
  app/baseapp.cpp 
  app/event.h
  app/input.h
  app/window.h
  app/window.cpp
  app/app_runner.h 
  app/app_runner.cpp

  # OpenGL libraries ==========
  gl/primitive.h 
  gl/primitive.cpp
  gl/shader.h 
  gl/shader.cpp
  gl/texture.h
  gl/texture.cpp
  gl/util.h

  # Optix wrapper libraries ==========
  optix/context.h 
  optix/context.cpp
  optix/denoiser.h 
  optix/denoiser.cpp
  optix/dmm.h
  optix/dmm.cpp
  optix/geometry_accel.h 
  optix/geometry_accel.cpp
  optix/util.h 
  optix/instance_accel.h 
  optix/instance_accel.cpp
  optix/instance.h 
  optix/instance.cpp
  optix/macros.h 
  optix/module.h 
  optix/module.cpp
  optix/omm.h
  optix/omm.cpp
  optix/pipeline.h 
  optix/pipeline.cpp
  optix/program.h 
  optix/program.cpp 
  optix/transform.h
  optix/transform.cpp
  optix/sbt.h 
  optix/cuda/device_util.cuh
  optix/cuda/omm.cu

  # Shapes ==========
  shape/box.h
  shape/box.cpp
  shape/curves.h 
  shape/curves.cpp
  shape/cylinder.h 
  shape/cylinder.cpp
  shape/gltfmesh.h 
  shape/gltfmesh.cpp
  shape/pcd.h
  shape/pcd.cpp
  shape/plane.h 
  shape/plane.cpp 
  shape/primitivemesh.h 
  shape/primitivemesh.cpp
  shape/shapegroup.h
  shape/sphere.h 
  shape/sphere.cpp
  shape/trianglemesh.h
  shape/trianglemesh.cpp
  shape/cuda/shapes.cuh
 
  # Materials ==========
  material/conductor.h 
  material/conductor.cpp 
  material/custom.h
  material/dielectric.h 
  material/dielectric.cpp 
  material/diffuse.h 
  material/diffuse.cpp
  material/disney.h
  material/disney.cpp
  material/roughconductor.h 
  material/roughconductor.cpp 
  material/roughdielectric.h
  material/roughdielectric.cpp
  material/isotropic.h
  material/thinfilm.h
  material/layered.h
  material/layered.cpp
  material/cuda/materials.cuh

  # Textures ==========
  texture/bitmap.h
  texture/bitmap.cpp
  texture/checker.h 
  texture/constant.h  
  texture/cuda/textures.cuh

  # Medium ==========
  medium/gridmedium.h
  medium/vdbgrid.h 
  medium/atmosphere.h

  # Emitter ==========
  emitter/area.h 
  emitter/area.cpp 
  emitter/envmap.h
  emitter/envmap.cpp
  emitter/point.h
  emitter/point.cpp

  # Physics ==========
  physics/sph.h
  physics/sph.cpp
  physics/cuda/sph.cu
  physics/cuda/sph.cuh

  prayground.h
)

set(pg_target "prayground")
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${sources}) 

# TODO: Link CUDA libraries to the prayground target
CUDA_GET_SOURCES_AND_OPTIONS(${sources} cmake_options options)
set(cuda_sources optix/cuda/omm.cu physics/cuda/sph.cu)
set_source_files_properties(${cuda_sources} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
if (CUDA_NVRTC_ENABLED)
  set(cu_obj_source_files)
  foreach(file ${sources})
    get_source_file_property(_cuda_source_format ${file} CUDA_SOURCE_PROPERTY_FORMAT)
    if (${_cuda_source_format} MATCHES "OBJ")
      list(APPEND cu_obj_source_files ${file})
    endif()
  endforeach()

  CUDA_WRAP_SRCS(${pg_target} OBJ generated_files ${cu_obj_source_files})
endif()

add_library(${pg_target} STATIC ${sources} ${generated_files})

if (WIN32)
  target_compile_definitions(${pg_target} PUBLIC GLAD_GLAPI_EXPORT)
endif()

target_link_libraries(${pg_target} LINK_PRIVATE
  ${GLFW_LIB_NAME}
  glad 
  imgui 
  ${CUDA_LIBRARIES}
)

if(USING_GNU_C OR USING_GNU_CXX)
  if(GCC_LIBSTDCPP_HACK)
    set_target_properties(${pg_target} PROPERTIES LINKER_LANGUAGE "C")
    target_link_libraries(${pg_target} LINK_PRIVATE ${STATIC_LIBSTDCPP})
  endif()
endif()

if(CUDA_NVRTC_ENABLED)
  target_link_libraries(${pg_target} LINK_PRIVATE ${CUDA_nvrtc_LIBRARY})
endif()
if(WIN32)
  target_link_libraries(${pg_target} LINK_PRIVATE winmm.lib)
endif()

if (WIN32)
  target_compile_options(${pg_target} PUBLIC "/source-charset:utf-8")
endif()

target_compile_definitions(${pg_target} PRIVATE PRAYGROUND_DIR="${PRAYGROUND_DIR}")
