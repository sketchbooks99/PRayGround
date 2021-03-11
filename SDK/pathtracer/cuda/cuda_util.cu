#include <optix.h>

#include <sutil/vec_math.h>
#include <cuda/helpers.h>

#include "../core/core_util.h"

namespace pt {

/** MEMO: 
 * If we need to take into account spectral property (not RGB), we should
 * switch Spectrum representation.
 * 
 * If Spectrum is rgb, rgb is float3? char3? I'm not sure which one is better.
 * 
 * NOTE: Currently, `Spectrum` must be a float3 */
 struct SurfaceInteraction {
    /** position of intersection point in world coordinates. */
    float3 p;

    /** Surface normal of primitive at intersection point. */
    float3 n;

    /** Incident and outgoing directions at a surface. */
    float3 wi;
    float3 wo;

    /** Spectrum information of ray. */
    float3 spectrum;

    /** radiance and attenuation term computed by a material attached with surface. */
    float3 radiance;
    float3 attenuation;
    float3 emission;

    /** UV coordinate at intersection point. */
    float2 uv;

    /** Derivatives on texture coordinates. */
    float dpdu, dpdv;

    /** seed for random */
    unsigned int seed;

    int trace_terminate;
};

INLINE DEVICE void* unpack_pointer( unsigned int i0, unsigned int i1 )
{
    const unsigned long long uptr = static_cast<unsigned long long>( i0 ) << 32 | i1;
    void* ptr = reinterpret_cast<void*>( uptr );
    return ptr;
}

INLINE DEVICE void packPointer(void* ptr, unsigned int& i0, unsigned int i1)
{
    const unsigned long long uptr = reinterpret_cast<unsigned long long>( ptr );
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

INLINE DEVICE SurfaceInteraction* getSurfaceInteraction()
{
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    return reinterpret_cast<SurfaceInteraction*>( unpack_pointer(u0, u1) ); 
}

}
