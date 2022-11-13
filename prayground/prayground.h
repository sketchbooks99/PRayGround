#pragma once 

#ifndef __CUDACC__ // CPU only

	#include <glad/glad.h>
	#include <GLFW/glfw3.h>

	#include <prayground_config.h>

	#include <cuda_gl_interop.h>
	#include <cuda_runtime.h>

	// core utilities
	#include "core/util.h"
	#include "core/file_util.h"
	#include "core/cudabuffer.h"
	#include "core/bitmap.h"
	#include "core/camera.h"
	#include "core/attribute.h"
	#include "core/scene.h"

	// optix utilities
	#include "optix/module.h"
	#include "optix/pipeline.h"
	#include "optix/program.h"
	#include "optix/macros.h"
	#include "optix/context.h"
	#include "optix/geometry_accel.h"
	#include "optix/instance_accel.h"
	#include "optix/instance.h"
	#include "optix/transform.h"
	#include "optix/denoiser.h"

	// application utilities
	#include "app/baseapp.h"
	#include "app/window.h"
	#include "app/app_runner.h"
	#include "app/input.h"

	#include "gl/shader.h"
#endif // __CUDACC__

#include <optix.h>

#include "core/spectrum.h"
#include "core/aabb.h"
#include "core/sampler.h"
#include "core/bsdf.h"
#include "core/interaction.h"
#include "core/onb.h"
#include "core/ray.h"

#include "optix/sbt.h"

// math utilities
#include "math/util.h"
#include "math/matrix.h"
#include "math/random.h"
#include "math/noise.h"
#include "math/vec.h"

// shape include
#include "shape/box.h"
#include "shape/curves.h"
#include "shape/cylinder.h"
#include "shape/plane.h"
#include "shape/primitivemesh.h"
#include "shape/sphere.h"
#include "shape/trianglemesh.h"
#include "shape/shapegroup.h"

// material include 
#include "material/conductor.h"
#include "material/dielectric.h"
#include "material/diffuse.h"
#include "material/disney.h"
#include "material/isotropic.h"
#include "material/custom.h"

// emitter include 
#include "emitter/area.h"
#include "emitter/envmap.h"

// texture include 
#include "texture/constant.h"
#include "texture/checker.h"
#include "texture/bitmap.h"

// Medium include 
#include "medium/atmosphere.h"
#include "medium/gridmedium.h"
#include "medium/vdbgrid.h"

#ifdef __CUDACC__ // GPU only
#include "optix/cuda/device_util.cuh"
#include "math/vec_math.h"
// Contains intersection test programs for custom primitives
// and utility functions for triangle/curves primitives.
#include "shape/cuda/shapes.cuh"

#include "texture/cuda/textures.cuh"
#endif // __CUDACC__

using namespace prayground;