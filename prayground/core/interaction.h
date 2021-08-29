#pragma once

#include <prayground/math/vec_math.h>

namespace prayground {

struct SurfaceProperty {
    void* data;
    unsigned int program_id;
};

enum class SurfaceType {
    None,        // None type (specifically, for envmap)
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

    /** Attenuation and self-emission from a surface attached with a shape. */
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
    int radiance_evaled; // For NEE
};

} // ::prayground

