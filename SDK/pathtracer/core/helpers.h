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

#include <optix.h>
#include <vector_types.h>
#include <sutil/vec_math.h>

__forceinline__ __device__ float3 toSRGB( const float3& c )
{
    float  invGamma = 1.0f / 2.4f;
    float3 powed    = make_float3( powf( c.x, invGamma ), powf( c.y, invGamma ), powf( c.z, invGamma ) );
    return make_float3(
        c.x < 0.0031308f ? 12.92f * c.x : 1.055f * powed.x - 0.055f,
        c.y < 0.0031308f ? 12.92f * c.y : 1.055f * powed.y - 0.055f,
        c.z < 0.0031308f ? 12.92f * c.z : 1.055f * powed.z - 0.055f );
}

//__forceinline__ __device__ float dequantizeUnsigned8Bits( const unsigned char i )
//{
//    enum { N = (1 << 8) - 1 };
//    return min((float)i / (float)N), 1.f)
//}
__forceinline__ __device__ unsigned char quantizeUnsigned8Bits( float x )
{
    x = clamp( x, 0.0f, 1.0f );
    enum { N = (1 << 8) - 1, Np1 = (1 << 8) };
    return (unsigned char)min((unsigned int)(x * (float)Np1), (unsigned int)N);
}

__forceinline__ __device__ uchar4 make_color( const float3& c )
{
    // first apply gamma, then convert to unsigned char
    float3 srgb = toSRGB( clamp( c, 0.0f, 1.0f ) );
    return make_uchar4( quantizeUnsigned8Bits( srgb.x ), quantizeUnsigned8Bits( srgb.y ), quantizeUnsigned8Bits( srgb.z ), 255u );
}
__forceinline__ __device__ uchar4 make_color( const float4& c )
{
    return make_color( make_float3( c.x, c.y, c.z ) );
}

#define float3_as_args(u) \
    reinterpret_cast<unsigned int&>((u).x), \
    reinterpret_cast<unsigned int&>((u).y), \
    reinterpret_cast<unsigned int&>((u).z)

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