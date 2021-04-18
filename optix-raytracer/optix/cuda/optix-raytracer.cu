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

#include <optix.h>
#include <cuda/random.h>
#include "../../optix/util.h"
#include "../../optix/sbt.h"

#include "../../shape/optix/sphere.cuh"
#include "../../shape/optix/trianglemesh.cuh"

#include "../../material/conductor.h"
#include "../../material/dielectric.h"
#include "../../material/diffuse.h"
#include "../../material/emitter.h"

#include "../../texture/checker.h"
#include "../../texture/image.h"

// -------------------------------------------------------------------------------
CALLABLE_FUNC void RG_FUNC(raygen)()
{
	const int w = params.width;
	const int h = params.height;
	const float3 eye = params.eye;
	const float3 U = params.U;
	const float3 V = params.V;
	const float3 W = params.W;
	const uint3 idx = optixGetLaunchIndex();
	const int subframe_index = params.subframe_index;

	unsigned int seed = tea<4>(idx.y * w + idx.x, subframe_index);

	float3 result = make_float3(0.0f, 0.0f, 0.0f);
	int i = params.samples_per_launch;

	do
	{
		const float2 subpixel_jitter = make_float2(rnd(seed) - 0.5f, rnd(seed) - 0.5f);

		const float2 d = 2.0f * make_float2(
			(static_cast<float>(idx.x) + subpixel_jitter.x) / static_cast<float>(w),
			(static_cast<float>(idx.y) + subpixel_jitter.y) / static_cast<float>(h)
		) - 1.0f;
		float3 ray_direction = normalize(d.x * U + d.y * V + W);
		float3 ray_origin = eye;

		oprt::SurfaceInteraction si;
		si.seed = seed;
		si.emission = make_float3(0.0f);
		si.radiance = make_float3(0.0f);
		si.attenuation = make_float3(1.0f);
		si.trace_terminate = false;
		si.radiance_evaled = false;

		// float3 radiance = make_float3(1.0f);

		int depth = 0;
		for ( ;; ) {
			trace_radiance(
				params.handle,
				ray_origin, 
				ray_direction, 
				0.01f, 
				1e16f, 
				&si 
			);
			
			if ( !si.radiance_evaled )
				result += si.emission * si.attenuation;
			
			result += si.radiance * si.attenuation;
	
			if ( si.trace_terminate || depth >= params.max_depth ) {
				break;
			}
			
			ray_origin = si.p;
			ray_direction = si.wo;

			++depth;
		}
	} while (--i);

	const uint3 launch_index = optixGetLaunchIndex();
	const unsigned int image_index = launch_index.y * params.width + launch_index.x;
	float3 accum_color = result / static_cast<float>(params.samples_per_launch);

	if (subframe_index > 0)
	{
		const float a = 1.0f / static_cast<float>(subframe_index + 1);
		const float3 accum_color_prev = make_float3(params.accum_buffer[image_index]);
		accum_color = lerp(accum_color_prev, accum_color, a);
	}
	params.accum_buffer[image_index] = make_float4(accum_color, 1.0f);
	params.frame_buffer[image_index] = make_color(accum_color);
}

// -------------------------------------------------------------------------------
CALLABLE_FUNC void MS_FUNC(radiance)()
{
	oprt::MissData* rt_data = reinterpret_cast<oprt::MissData*>(optixGetSbtDataPointer());
	oprt::SurfaceInteraction *si = get_surfaceinteraction();

	// si->radiance = make_float3(rt_data->bg_color);
	si->emission = make_float3(rt_data->bg_color);
	si->trace_terminate = true;
}
