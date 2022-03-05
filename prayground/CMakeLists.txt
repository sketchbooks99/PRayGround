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
  core/shape.h 
  core/shape.cpp
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
  gl/shader.h 
  gl/shader.cpp

  # Optix wrapper libraries ==========
  optix/context.h 
  optix/context.cpp
  optix/denoiser.h 
  optix/denoiser.cpp
  optix/geometry_accel.h 
  optix/geometry_accel.cpp
  optix/helpers.h 
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
  optix/cuda/texture.cuh
  optix/cuda/device_util.cuh

  # Shapes ==========
  shape/box.h
  shape/box.cpp
  shape/cylinder.h 
  shape/cylinder.cpp 
  shape/plane.h 
  shape/plane.cpp 
  shape/shapegroup.h
  shape/sphere.h 
  shape/sphere.cpp
  shape/trianglemesh.h
  shape/trianglemesh.cpp
  shape/cuda/cylinder.cuh 
  shape/cuda/plane.cuh 
  shape/cuda/sphere.cuh 
  shape/cuda/trianglemesh.cuh
  shape/cuda/box.cuh
 
  # Materials ==========
  material/conductor.h 
  material/conductor.cpp 
  material/dielectric.h 
  material/dielectric.cpp 
  material/diffuse.h 
  material/diffuse.cpp
  material/disney.h
  material/disney.cpp
  material/isotropic.h 
  material/cuda/conductor.cuh
  material/cuda/dielectric.cuh
  material/cuda/diffuse.cuh
  material/cuda/disney.cuh

  # Textures ==========
  texture/bitmap.h
  texture/bitmap.cpp
  texture/checker.h 
  texture/constant.h  

  # Medium ==========
  medium/gridmedium.h 
  medium/atmosphere.h

  # Emitter ==========
  emitter/area.h 
  emitter/area.cpp 
  emitter/envmap.h
  emitter/envmap.cpp
  emitter/cuda/area.cuh 
  emitter/cuda/envmap.cuh
  
  prayground.h

  # OPTIONS -rdc true
)

set(pg_target "prayground")
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${sources})

add_library(${pg_target} STATIC ${sources})
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

set(prayground_sources ${sources} PARENT_SCOPE)
