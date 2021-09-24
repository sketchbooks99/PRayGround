set(sources
  # Core libraries ==========
  core/aabb.h 
  core/bitmap.cpp 
  core/bitmap.h 
  core/bsdf.h 
  core/camera.h 
  core/camera.cpp
  core/color.h 
  core/cudabuffer.h 
  core/emitter.h 
  core/file_util.h 
  core/file_util.cpp 
  core/film.h 
  core/film.cpp
  core/interaction.h
  core/load3d.h 
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

  # Shapes ==========
  shape/cylinder.h 
  shape/cylinder.cpp 
  shape/plane.h 
  shape/plane.cpp 
  shape/sphere.h 
  shape/sphere.cpp
  shape/trianglemesh.h
  shape/trianglemesh.cpp
  shape/cuda/cylinder.cuh 
  shape/cuda/plane.cuh 
  shape/cuda/sphere.cuh 
  shape/cuda/trianglemesh.cuh
 
  # Materials ==========
  material/conductor.h 
  material/conductor.cpp 
  material/dielectric.h 
  material/dielectric.cpp 
  material/diffuse.h 
  material/diffuse.cpp
  material/disney.h
  material/disney.cpp
  material/cuda/conductor.cuh
  material/cuda/dielectric.cuh
  material/cuda/diffuse.cuh
  material/cuda/disney.cuh

  # Textures ==========
  texture/bitmap.h
  texture/bitmap.cpp
  texture/checker.h 
  texture/checker.cpp 
  texture/constant.h  
  texture/constant.cpp

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

set(prayground_sources)
foreach (source ${sources})
  set(prayground_sources ${prayground_sources} "${PRAYGROUND_DIR}/prayground/${source}")
endforeach()

set(prayground_sources ${prayground_sources} ${CMAKE_CURRENT_BINARY_DIR}/../prayground_config.h)

set(prayground_sources ${prayground_sources} PARENT_SCOPE)
