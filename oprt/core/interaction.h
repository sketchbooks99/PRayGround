#pragma once

#include <sutil/vec_math.h>

namespace oprt {

struct SurfaceProperty {
    void* data;
    unsigned int program_id;
};

enum class SurfaceType {
    Material,    // Scene geometry
    AreaEmitter, // Emitter sampling
    Medium       // Meduim --- Future work
};

/// @note Currently \c spectrum is RGB representation, not spectrum. 
struct SurfaceInteraction {
    /** Position of intersection point in world coordinates. */
    float3 p;

    /** Surface normal of primitive at an intersection point. */
    float3 n;

    /** Incident and outgoing directions at a surface. */
    float3 wi;
    float3 wo;

    /** Spectrum information of ray. */
    float3 spectrum;

    /** Radiance and attenuation computed by a material attached with a surface. */
    float3 attenuation;
    float3 emission;

    /** UV coordinate at an intersection point. */
    float2 uv;

    /** Derivatives on texture coordinates. */
    float2 dpdu;    // Tangent vector at a surface.
    float2 dpdv;    // Binormal vector at a surface.

    /** Seed for random */
    unsigned int seed;

    SurfaceProperty surface_property;

    SurfaceType surface_type;

    int trace_terminate;
    int radiance_evaled;
};

}
