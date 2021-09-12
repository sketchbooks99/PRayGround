#pragma once

#include <prayground/math/vec_math.h>
#include <curand.h>
#include <curand_kernel.h>

namespace prayground {

// namespace builtin { 

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

    // Delta
    Delta = Reflection | Refraction,

    // Rough surface
    Rough = RoughReflection | RoughRefraction,

    // Material
    Material        = Diffuse | Reflection | Refraction | RoughReflection | RoughRefraction,
    // Medium --- Future work
    Medium 
};

struct SurfaceInfo 
{
    // Surfaceのデータ
    void* data;

    // 重点的サンプリングやbsdfの評価用のCallable関数へのID
    unsigned int sample_id;
    unsigned int bsdf_id;
    unsigned int pdf_id;

    SurfaceType type;
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

    /** Partial derivatives on intersection point */
    float3 dpdu;
    float3 dpdv;

    curandState_t* curand_state;

    SurfaceInfo surface_property;

    bool trace_terminate;
    bool radiance_evaled; // For NEE
    bool is_specular;
};

// } // ::builtin

} // ::prayground

