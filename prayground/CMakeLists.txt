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
  core/sampler.h
  core/shape.h 
  core/shape.cpp
  core/scene.h 
  core/stream_helpers.h 
  core/texture.h 
  core/util.h
  core/shaders/bitmap.vert
  core/shaders/bitmap.frag
  # core/cuda/spectrum.cu

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
  optix/pipeline.h 
  optix/pipeline.cpp
  optix/program.h 
  optix/program.cpp 
  optix/transform.h
  optix/transform.cpp
  optix/sbt.h 
  optix/cuda/device_util.cuh

  # Shapes ==========
  shape/box.h
  shape/box.cpp
  shape/curves.h 
  shape/curves.cpp
  shape/cylinder.h 
  shape/cylinder.cpp
  shape/gltfmesh.h 
  shape/gltfmesh.cpp
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
  
  prayground.h
)

set(pg_target "prayground")
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${sources})

add_library(${pg_target} STATIC ${sources})

set_property(TARGET ${pg_target} PROPERTY CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (WIN32)
  target_compile_definitions(${pg_target} PUBLIC GLAD_GLAPI_EXPORT)
endif()

target_link_libraries(${pg_target} LINK_PRIVATE
  ${GLFW_LIB_NAME}
  glad
  imgui
  tinyusdz::tinyusdz_static
  ${CUDA_LIBRARIES}
)

target_include_directories(${pg_target} PRIVATE 
  "${PRAYGROUND_EXTERNAL_DIR}"
  "${PRAYGROUND_EXTERNAL_DIR}/tinyusdz/src")

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

set(prayground_sources ${sources} PARENT_SCOPE)
