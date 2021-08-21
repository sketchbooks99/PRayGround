//
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
#pragma once 

#ifndef __CUDACC__

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <optix.h>

#include <sampleConfig.h>

#include <sutil/sutil.h>
#include <sutil/vec_math.h>

// optix utilities
#include "optix/module.h"
#include "optix/pipeline.h"
#include "optix/sbt.h"
#include "optix/program.h"
#include "optix/macros.h"
#include "optix/context.h"
#include "optix/accel.h"
#include "optix/instance.h"

// application utilities
#include "core/util.h"
#include "core/file_util.h"
#include "core/cudabuffer.h"
#include "core/bitmap.h"
#include "core/film.h"
#include "core/camera.h"

#include "app/baseapp.h"
#include "app/window.h"
#include "app/app_runner.h"

#include "gl/shader.h"

#endif

// shape include
#include "shape/sphere.h"
#include "shape/trianglemesh.h"
#include "shape/plane.h"
#include "shape/cylinder.h"

// material include 
#include "material/conductor.h"
#include "material/dielectric.h"
#include "material/diffuse.h"
#include "material/disney.h"

// emitter include 
#include "emitter/area.h"
#include "emitter/envmap.h"

// texture include 
#include "texture/constant.h"
#include "texture/checker.h"
#include "texture/bitmap.h"

using namespace oprt;