#pragma once

#include <sutil/vec_math.h>
#include "material.h"

namespace pt {

/** MEMO: 
 * If we need to take into account spectral property (not RGB), we should
 * switch Spectrum representation.
 * 
 * If Spectrum is rgb, rgb is float3? char3? I'm not sure which one is better.
 * 
 * NOTE: Currently, `Spectrum` must be only float3.
 * */
template <typename Spectrum> 
struct SurfaceInteraction {
    /** position of intersection point in world coordinates. */
    float3 p;

    /** Surface normal of primitive at intersection point. */
    float3 n;

    /** UV coordinate at intersection point. */
    float2 uv;

    /** Spectrum information of ray. */
    Spectrum spectrum;

    /** Type of material to identify the shading at intersected point. */
    /** MEMO:
     *  Can this be a pointer such as shared_ptr? Can optixTrace() propagate pointer? */
    MaterialType mattype;       
};

}

