#pragma once 


#include "../params.h"

using namespace prayground;

extern "C"
{
	__constant__ LaunchParams params;
}

INLINE DEVICE SurfaceInteraction* getSurfaceInteraction()
{
	const uint32_t u0 = getPayload<0>();
	const uint32_t u1 = getPayload<1>();
	return reinterpret_cast<SurfaceInteraction*>(unpackPointer(u0, u1));
}

INLINE DEVICE void trace(
	OptixTraversableHandle handle,
	const float3& ro, const float3& rd,
	float tmin, float tmax,
	uint32_t ray_type,
	uint32_t ray_count,
	SurfaceInteraction* si
)
{
	uint32_t u0, u1;
	packPointer(si, u0, u1);
	trace(handle, ro, rd, tmin, tmax, ray_type, ray_count, u0, u1);
}