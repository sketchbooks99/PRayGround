#pragma once 

// CPU only
#ifndef __CUDACC__

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <optix.h>

#include <prayground_config.h>

// optix utilities
#include "optix/module.h"
#include "optix/pipeline.h"
#include "optix/sbt.h"
#include "optix/program.h"
#include "optix/macros.h"
#include "optix/context.h"
#include "optix/geometry_accel.h"
#include "optix/instance_accel.h"
#include "optix/instance.h"
#include "optix/transform.h"
#include "optix/denoiser.h"

// core utilities
#include "core/util.h"
#include "core/file_util.h"
#include "core/cudabuffer.h"
#include "core/bitmap.h"
#include "core/camera.h"
#include "core/attribute.h"

// application utilities
#include "app/baseapp.h"
#include "app/window.h"
#include "app/app_runner.h"
#include "app/input.h"

#include "gl/shader.h"

#endif

// math utilities
#include "math/util.h"
#include "math/vec_math.h"
#include "math/matrix.h"
#include "math/random.h"
#include "math/noise.h"

// shape include
#include "shape/sphere.h"
#include "shape/trianglemesh.h"
#include "shape/plane.h"
#include "shape/cylinder.h"
#include "shape/box.h"
#include "shape/shapegroup.h"

// material include 
#include "material/conductor.h"
#include "material/dielectric.h"
#include "material/diffuse.h"
#include "material/disney.h"
#include "material/isotropic.h"

// emitter include 
#include "emitter/area.h"
#include "emitter/envmap.h"

// texture include 
#include "texture/constant.h"
#include "texture/checker.h"
#include "texture/bitmap.h"

using namespace prayground;