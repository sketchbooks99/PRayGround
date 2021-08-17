#include <optix.h>
#include <cuda/random.h>
#include <sutil/vec_math.h>
#include "../../oprt/core/color.h"
#include "../../oprt/oprt.h"
#include "../../oprt/params.h"

namespace oprt 
{

extern "C"
{
    __constant__ LaunchParams params;
}

static __forceinline__ __device__ cameraUVW(const CameraData& camera, float3& U, float3& V, float3& W)
{
    W = camera.lookat - camera.eye;
    float wlen = length(W);
    U = normalize(cross(W, camera.up));
    V = normalize(cross(U, W));

    float vlen = wlen * tanf(0.5f * camera.fov * M_PIf / 180.0f);
    V *= vlen;
    float ulen = vlen * camera.aspect;
    U *= ulen;
}

extern "C" __device__ __raygen__pinhole()
{
    const RaygenData* raygen = reinterpret_cast<RaygenData*>(optixGetSbtDataPointer());
    float3 U, V, W;
    cameraUVW(raygen.camera, U, V, W);

    const uint3 idx = optixGetLaunchIndex();
    unsigned int seed = tea<4>(idx.y * params.width + idx.x, subframe_index);

    float3 result = make_float3(0.0f, 0.0f, 0.0f);
    int i = params.samples_per_launch;

    do 
    {
        const float2 subpixel_jitter = make_float2(rnd(seed) - 0.5f, rnd(seed) - 0.5f);
    }
}

}