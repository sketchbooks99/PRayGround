#pragma once

#include <prayground/math/vec_math.h>
#include <curand.h>
#include <curand_kernel.h>

namespace prayground {

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

    // Delta
    Delta = Reflection | Refraction,

    // Rough surface
    Rough = RoughReflection | RoughRefraction,

    // Material
    Material        = Diffuse | Reflection | Refraction | RoughReflection | RoughRefraction,

    // Emitter 
    AreaEmitter     = 1u << 5,
};

constexpr SurfaceType  operator|(SurfaceType t1, SurfaceType t2)    { return static_cast<SurfaceType>(  (unsigned int)t1 | (unsigned int)t2 ); }
constexpr SurfaceType  operator|(unsigned int t1, SurfaceType t2)   { return static_cast<SurfaceType>(                t1 | (unsigned int)t2 ); }
constexpr SurfaceType  operator&(SurfaceType t1, SurfaceType t2)    { return static_cast<SurfaceType>(  (unsigned int)t1 & (unsigned int)t2 ); }
constexpr SurfaceType  operator&(unsigned int t1, SurfaceType t2)   { return static_cast<SurfaceType>(                t1 & (unsigned int)t2 ); }
constexpr SurfaceType  operator~(SurfaceType t1)                    { return static_cast<SurfaceType>( ~(unsigned int)t1 ); }
constexpr unsigned int operator+(SurfaceType t1)                    { return static_cast<unsigned int>(t1); }

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
/// @todo template <typename Spectrum>
struct SurfaceInteraction {
    /** Position of intersection point in world coordinates. */
    float3 p;

    /** Incident and outgoing directions at a surface. */
    float3 wi;
    float3 wo;

    /** ray time */
    float t;

    /** Spectrum information of ray. */
    float3 spectrum;

    /** Albedo and self-emission from a surface attached with a shape. */
    float3 albedo;
    float3 emission;

    /** UV coordinate at an intersection point. */
    float2 uv;
 
    struct {
        /** Surface normal */
        float3 n;

        /** Partial derivative on intersection point */
        float3 dpdu, dpdv;

        /** Partial derivative on surface normal */
        float3 dndu, dndv;
    } shading;

    unsigned int seed;

    SurfaceInfo surface_info;

    bool trace_terminate;
    bool radiance_evaled; // For NEE
};

} // ::prayground

