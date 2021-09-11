#pragma once

#include <prayground/math/vec_math.h>
#include <curand.h>
#include <curand_kernel.h>

namespace prayground {

struct SurfaceProperty {
    void* data;
    unsigned int program_id;
};

enum class SurfaceType : unsigned int {
    // None type ( default )
    None            = 0,        
    // Diffuse surface
    Diffuse         = 1u << 0,  
    // Specular surfaces
    Reflection      = 1u << 1,
    Refraction      = 1u << 2,
    // Rough surfaces ( w/ microfacet )
    RoughReflection = 1u << 3,
    RoughRefraction = 1u << 4,
    // Emitter 
    AreaEmitter     = 1u << 5,
    // Material
    Material        = Diffuse | Reflection | Refraction | RoughReflection | RoughRefraction,
    // Medium --- Future work
    Medium 
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

    curandState_t* curand_state;

    SurfaceProperty surface_property;

    SurfaceType surface_type;

    int trace_terminate;
    int radiance_evaled; // For NEE
};

} // ::prayground

