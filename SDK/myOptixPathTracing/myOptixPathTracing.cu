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

#include "myOptixPathTracing.h"
#include "random.h"

#include <sutil/vec_math.h>
#include <cuda/helpers.h>

#define float3_as_args(u) \
	reinterpret_cast<unsigned int&>((u).x), \
	reinterpret_cast<unsigned int&>((u).y), \
	reinterpret_cast<unsigned int&>((u).z)

extern "C" {
__constant__ Params params;
}

//struct RadiancePRD
//{
//	float3 emitted;
//	float3 radiance;
//	float3 attenuation;
//	float3 origin;
//	float3 direction;
//	unsigned int seed;
//	unsigned int depth;
//	int countEmitted;
//	int done;
//	int pad;
//};

struct RadiancePRD
{
	float3 result;
	unsigned int depth;
	unsigned int seed;
};

struct Onb
{
	__forceinline__ __device__ Onb(const float3& normal)
	{
		m_normal = normal;

		if (fabs(m_normal.x) > fabs(m_normal.z))
		{
			m_binormal.x = -m_normal.y;
			m_binormal.y = m_normal.x;
			m_binormal.z = 0;
		}
		else
		{
			m_binormal.x = 0;
			m_binormal.y = -m_normal.z;
			m_binormal.z = -m_normal.y;
		}

		m_binormal = normalize(m_binormal);
		m_tangent = cross(m_binormal, m_normal);
	}

	__forceinline__ __device__ void inverse_transform(float3& p) const
	{
		p = p.x * m_tangent + p.y * m_binormal + p.z * m_normal;
	}

	float3 m_tangent;
	float3 m_binormal;
	float3 m_normal;
};

// -------------------------------------------------------------------------------
static __forceinline__ __device__ void* unpackPointer(unsigned int i0, unsigned int i1)
{
	const unsigned long long uptr = static_cast<unsigned long long>(i0) << 32 | i1;
	void* ptr = reinterpret_cast<void*>(uptr);
	return ptr;
}

// -------------------------------------------------------------------------------
static __forceinline__ __device__ void packPointer(void* ptr, unsigned int& i0, unsigned int& i1)
{
	const unsigned long long uptr = reinterpret_cast<unsigned long long>(ptr);
	i0 = uptr >> 32;
	i1 = uptr & 0x00000000ffffffff;
}

// -------------------------------------------------------------------------------
static __forceinline__ __device__ RadiancePRD getPRD()
{
	RadiancePRD prd;
	prd.result.x = int_as_float(optixGetPayload_0());
	prd.result.y = int_as_float(optixGetPayload_1());
	prd.result.z = int_as_float(optixGetPayload_2());
	prd.depth = optixGetPayload_3();
	prd.seed = optixGetPayload_4();
	return prd;
}

static __forceinline__ __device__ void setPRD(const RadiancePRD& prd)
{
	optixSetPayload_0(float_as_int(prd.result.x));
	optixSetPayload_1(float_as_int(prd.result.y));
	optixSetPayload_2(float_as_int(prd.result.z));
	optixSetPayload_3(float_as_int(prd.depth));
	optixSetPayload_4(float_as_int(prd.seed));
}

// -------------------------------------------------------------------------------
static __forceinline__ __device__ void setPayloadOcclusion(bool occluded)
{
	optixSetPayload_0(static_cast<unsigned int>(occluded));
}

// -------------------------------------------------------------------------------
static __forceinline__ __device__ void cosine_sample_hemisphere(const float u1, const float u2, float3& p)
{
	// Uniformly sample disk.
	const float r = sqrtf(u1);
	const float phi = 2.0f * M_PIf * u2;
	p.x = r * cosf(phi);
	p.y = r * sinf(phi);

	// Project up to hemisphere
	p.z = sqrtf(fmaxf(0.0f, 1.0f - p.x * p.x - p.y * p.y));
}

// -------------------------------------------------------------------------------
static __forceinline__ __device__ float schlick(float cosine, float ior)
{
	float r0 = (1.0f - ior) / (1.0f + ior);
	r0 = r0 * r0;
	return r0 + (1.0f - r0) * powf((1.0f - cosine), 5.0f);
}

// -------------------------------------------------------------------------------
// ref: http://www.sic.shibaura-it.ac.jp/~yaoki/cg/CG2008-10.pdf
static __forceinline__ __device__ float3 refract(float3 r_dir, float3 normal, float etai_over_etat)
{
	float k_f = 1.0f / (sqrtf(pow(etai_over_etat, 2.0f) * powf(length(r_dir), 2.0f) - powf(length(normal + r_dir), 2.0f)));
	return k_f * (normal + r_dir) - normal;
}

 // -------------------------------------------------------------------------------
static __forceinline__ __device__ float3 traceRadiance(
	OptixTraversableHandle handle,
	float3 ray_origin,
	float3 ray_direction,
	unsigned int depth,
	unsigned int seed
)
{
	RadiancePRD prd;
	prd.depth = depth;
	prd.seed = seed;
	optixTrace(
		handle,
		ray_origin,
		ray_direction,
		0.01f,
		1e16f,
		0.0f,
		OptixVisibilityMask(1),
		OPTIX_RAY_FLAG_NONE,
		RAY_TYPE_RADIANCE,
		RAY_TYPE_COUNT,
		RAY_TYPE_RADIANCE,
		float3_as_args(prd.result),
		reinterpret_cast<unsigned int&>(prd.depth),
		reinterpret_cast<unsigned int&>(prd.seed)
	);

	return prd.result;
}

// -------------------------------------------------------------------------------
static __forceinline__ __device__ bool traceOcclusion(
	OptixTraversableHandle handle,
	float3 ray_origin,
	float3 ray_direction,
	float tmin, float tmax)
{
	unsigned int occluded = 0u;
	optixTrace(
		handle,
		ray_origin,
		ray_direction,
		0.01f, // tmin
		1e16f,  // tmax
		0.0f,	// rayTime
		OptixVisibilityMask(1),
		OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
		RAY_TYPE_OCCLUSION,
		RAY_TYPE_COUNT,
		RAY_TYPE_OCCLUSION,
		occluded);
	return occluded;
}

// -------------------------------------------------------------------------------
extern "C" __global__ void __raygen__rg()
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

		RadiancePRD prd;
		prd.result = make_float3(0.0f);
		prd.depth = 0;
		prd.seed = seed;
		/*result += traceRadiance(
			params.handle,
			ray_origin,
			ray_direction,
			prd.depth);*/
		optixTrace(
			params.handle,
			ray_origin,
			ray_direction,
			0.01f,
			1e16f,
			0.0f,
			OptixVisibilityMask(1),
			OPTIX_RAY_FLAG_NONE,
			RAY_TYPE_RADIANCE,
			RAY_TYPE_COUNT,
			RAY_TYPE_RADIANCE,
			float3_as_args(prd.result),
			reinterpret_cast<unsigned int&>(prd.depth),
			reinterpret_cast<unsigned int&>(prd.seed)
		);

		result += prd.result;
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
extern "C" __global__ void __miss__radiance()
{
	MissData* rt_data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
	RadiancePRD prd = getPRD();

	prd.result = make_float3(rt_data->bg_color);
	prd.depth = params.max_depth;
	setPRD(prd);
}

// -------------------------------------------------------------------------------
extern "C" __global__ void __closesthit__occlusion()
{
	setPayloadOcclusion(true);
}

// -------------------------------------------------------------------------------
extern "C" __global__ void __closesthit__radiance__dielectric()
{
	// Get binded data by application.
	HitGroupData* rt_data = (HitGroupData*)optixGetSbtDataPointer();

	const int prim_idx = optixGetPrimitiveIndex();
	const int3 index = rt_data->indices[prim_idx];
	const float3 ray_dir = optixGetWorldRayDirection();
	const float u = optixGetTriangleBarycentrics().x;
	const float v = optixGetTriangleBarycentrics().y;
	const int vert_idx_offset = prim_idx * 3;

	const float3 v0 = make_float3(rt_data->vertices[index.x]);
	const float3 v1 = make_float3(rt_data->vertices[index.y]);
	const float3 v2 = make_float3(rt_data->vertices[index.z]);
	const float3 n0 = normalize(make_float3(rt_data->normals[index.x]));
	const float3 n1 = normalize(make_float3(rt_data->normals[index.y]));
	const float3 n2 = normalize(make_float3(rt_data->normals[index.z]));

	float3 N = normalize((1.0f - u - v) * n0 + u * n1 + v * n2);
	const float3 P = optixGetWorldRayOrigin() + optixGetRayTmax() * ray_dir;
	float3 result = make_float3(0.0f);

	float ior = 1.52f;

	GlassHitType hit_type = (GlassHitType)optixGetHitKind();
	float3 front_hit_point = P, back_hit_point = P;
	if (hit_type & HIT_OUTSIDE_FROM_OUTSIDE || hit_type & HIT_INSIDE_FROM_INSIDE)
	{
		front_hit_point += 0.01f * N;
		back_hit_point -= 0.01f * N;
	}
	else
	{
		front_hit_point -= 0.01f * N;
		back_hit_point += 0.01f * N;
	}

	float etai_over_etat = optixIsTriangleFrontFaceHit() ? (1.0f / ior) : ior;
	//float etai_over_etat = ior;

	RadiancePRD prd = getPRD();

	float3 w_in = normalize(ray_dir);
	float cos_theta = fminf(dot(-w_in, N), 1.0f);
	float reflect_prob = 1.0f;

	//prd->emitted = rt_data->emission_color;
	float3 w_out = make_float3(0.0f);
	unsigned int seed = prd.seed;
	if (prd.depth < params.max_depth)
	{
		// refraction
		{
			w_out = refract(w_in, N, etai_over_etat);

			// I can simply change roughness of glass
			//w_out += make_float3(rnd(seed), rnd(seed), rnd(seed)) * 1.0f - make_float3(0.5f);
			float3 radiance = traceRadiance(params.handle,
				back_hit_point,
				w_out,
				prd.depth + 1,
				seed);
			reflect_prob = schlick(cos_theta, etai_over_etat);
			result += (1.0f - reflect_prob) * radiance;
		}

		// reflection
		{
			w_out = reflect(w_in, N);
			float3 radiance = traceRadiance(params.handle,
				front_hit_point,
				w_out,
				prd.depth + 1,
				seed);
			result += reflect_prob * radiance;
		}
	}
	prd.result = result;
	setPRD(prd);
}

// -------------------------------------------------------------------------------
extern "C" __global__ void __closesthit__radiance()
{
	HitGroupData* rt_data = (HitGroupData*)optixGetSbtDataPointer();

	const int prim_idx = optixGetPrimitiveIndex();
	const int3 index = rt_data->indices[prim_idx];
	const float3 ray_dir = optixGetWorldRayDirection();
	const float u = optixGetTriangleBarycentrics().x;
	const float v = optixGetTriangleBarycentrics().y;
	const int vert_idx_offset = prim_idx * 3;

	const float3 v0 = make_float3(rt_data->vertices[index.x]);
	const float3 v1 = make_float3(rt_data->vertices[index.y]);
	const float3 v2 = make_float3(rt_data->vertices[index.z]);
	const float3 n0 = normalize(make_float3(rt_data->normals[index.x]));
	const float3 n1 = normalize(make_float3(rt_data->normals[index.y]));
	const float3 n2 = normalize(make_float3(rt_data->normals[index.z]));

	float3 N_0 = normalize((1.0f - u - v) * n0 + u * n1 + v * n2);

	const float3 N = faceforward(N_0, -ray_dir, N_0);
	const float3 P = optixGetWorldRayOrigin() + optixGetRayTmax() * ray_dir;

	RadiancePRD prd = getPRD();

	float3 emission = rt_data->emission_color;
	float3 radiance = make_float3(0.0f);
	float3 attenuation = rt_data->diffuse_color;
	float3 w_in = make_float3(0.0f);
	float3 result = make_float3(0.0f);

	unsigned int seed = prd.seed;

	const float z1 = rnd(seed);
	const float z2 = rnd(seed);

	cosine_sample_hemisphere(z1, z2, w_in);
	Onb onb(N);
	onb.inverse_transform(w_in);

	if (prd.depth < params.max_depth)
	{
		float3 light_emission = make_float3(0.0f);
		size_t num_light = sizeof(params) / sizeof(ParallelogramLight);
		float weight = 0.0f;
		for (int i = 0; i < num_light; i++)
		{
			const float z1 = rnd(seed);
			const float z2 = rnd(seed);

			ParallelogramLight light = params.light[i];
			const float3 light_pos = light.corner + light.v1 * z1 + light.v2 * z2;

			// Calculate properties of light sample (for area based pdf)
			const float Ldist = length(light_pos - P);
			const float3 L = normalize(light_pos - P);
			const float nDl = dot(N, L);
			const float LnDl = -dot(light.normal, L);
			if (nDl > 0.0f && LnDl > 0.0f)
			{
				const bool occluded = traceOcclusion(
					params.handle,
					P,
					L,
					0.01f,
					Ldist - 0.01f
				);

				if (!occluded)
				{
					const float A = length(cross(light.v1, light.v2));
					weight += nDl * LnDl * A / (M_PIf * Ldist * Ldist);
				}
			}
			light_emission += light.emission * weight;
		}

		radiance = traceRadiance(
			params.handle,
			P,
			w_in,
			prd.depth + 1,
			seed);
		radiance += light_emission;
		result = emission + radiance * attenuation;
	} 
	prd.result = result;
	setPRD(prd);
}