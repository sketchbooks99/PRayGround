set(sources
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
  core/interaction.h
  core/load3d.h 
  core/material.h 
  core/onb.h 
  core/ray.h 
  core/shape.h 
  core/shape.cpp
  core/stream_helpers.h 
  core/texture.h 
  core/transform.h 
  core/util.h
  core/shaders/bitmap.vert
  core/shaders/bitmap.frag

  # Math libraries ==========
  math/matrix.h
  # math/vec.h
  math/util.h

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
  
  prayground.h

  # OPTIONS -rdc true
)

set(prayground_sources)
foreach (source ${sources})
  set(prayground_sources ${prayground_sources} "${PRAYGROUND_DIR}/prayground/${source}")
endforeach()

set(prayground_sources ${prayground_sources} ${CMAKE_CURRENT_BINARY_DIR}/../sampleConfig.h)

set(prayground_sources ${prayground_sources} PARENT_SCOPE)
