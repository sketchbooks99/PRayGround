#pragma once

#include <curand.h>
#include <curand_kernel.h>

#include <prayground/math/vec.h>
#include <prayground/core/texture.h>

#ifdef __CUDACC__
#include <prayground/optix/cuda/device_util.cuh>
#endif

namespace prayground {

    enum class SurfaceType : uint32_t {
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
        Rough = RoughReflection | RoughRefraction | Diffuse,

        // Material
        Material        = Diffuse | Reflection | Refraction | RoughReflection | RoughRefraction,

        // Refractive material
        Refractive = Refraction | RoughRefraction,

        // Emitter 
        AreaEmitter     = 1u << 5,

        // Medium
        Medium          = 1u << 6, 

        // Layered
        Layered         = 1u << 7
    };

    constexpr HOSTDEVICE SurfaceType  operator|(SurfaceType t1, SurfaceType t2)    { return static_cast<SurfaceType>(  (uint32_t)t1 | (uint32_t)t2 ); }
    constexpr HOSTDEVICE SurfaceType  operator|(uint32_t t1, SurfaceType t2)       { return static_cast<SurfaceType>(            t1 | (uint32_t)t2 ); }
    constexpr HOSTDEVICE SurfaceType  operator&(SurfaceType t1, SurfaceType t2)    { return static_cast<SurfaceType>(  (uint32_t)t1 & (uint32_t)t2 ); }
    constexpr HOSTDEVICE SurfaceType  operator&(uint32_t t1, SurfaceType t2)       { return static_cast<SurfaceType>(            t1 & (uint32_t)t2 ); }
    constexpr HOSTDEVICE SurfaceType  operator~(SurfaceType t1)                    { return static_cast<SurfaceType>( ~(uint32_t)t1 ); }
    constexpr HOSTDEVICE uint32_t     operator+(SurfaceType t1)                    { return static_cast<uint32_t>(t1); }

    struct SurfaceCallableID {
        uint32_t sample;
        uint32_t bsdf;
        uint32_t pdf;
    };

    struct SurfaceInfo 
    {
        // Surfaceのデータ
        void* data;

        // BSDFの重点サンプリングと評価関数用のCallables関数へのID
        SurfaceCallableID callable_id;
    
        SurfaceType type;

        bool use_bumpmap = false;
        Texture::Data bumpmap;
    };
    
    struct MediumInfo {
        void* data;

        uint32_t phase_func_id;
    };

    struct MediumInterface {
        MediumInfo outside;
        MediumInfo inside;
    };

    struct Shading {
        /* Surface normal */
        Vec3f n;
        /* Texture coordinate at an intersection point */
        Vec2f uv;
        /* Partial derivative on intersection point */
        Vec3f dpdu, dpdv;
        /* Partial derivative on surface normal */
        Vec3f dndu, dndv;
    };

    /// @note Currently \c spectrum is RGB representation, not spectrum. 
    /// @todo template <typename Spectrum>
    template <typename Spectrum>
    struct SurfaceInteraction_ {
        /** Position of intersection point in world coordinates. */
        Vec3f p;

        /** ray time */
        float t;

        /** Shading frame */
        Shading shading;

        /** Incident and outgoing directions at a surface. */
        Vec3f wi;
        Vec3f wo;

        /** Albedo and self-emission from a surface attached with a shape. */
        Spectrum albedo;
        Spectrum emission;

        /* For propagating random seed among path */
        uint32_t seed;

        /* IOR of current medium */
        float ior = 1.000293f;

        /* Array of surface information. Multiple informations are typically used for layered material */
        SurfaceInfo* surface_info;

        bool trace_terminate;
    };

    struct MediumInteraction {
        Vec3f p;

        Vec3f wi;
        Vec3f wo;

        float t;

        MediumInterface medium_interface;
    };

#ifndef __CUDACC__
    inline std::ostream& operator<<(std::ostream& out, const SurfaceCallableID& id)
    {
        return out << "SurfaceCallableID{sample=" << id.sample << ", bsdf=" << id.bsdf << ", pdf=" << id.pdf << "}";
    }

    inline std::ostream& operator<<(std::ostream& out, SurfaceType surface_type)
    {
        switch (surface_type)
        {
        case SurfaceType::None:
            return out << "SurfaceType::None";
        case SurfaceType::Diffuse:
            return out << "SurfaceType::Diffuse";
        case SurfaceType::Reflection:
            return out << "SurfaceType::Reflection";
        case SurfaceType::Refraction:
            return out << "SurfaceType::Refraction";
        case SurfaceType::RoughReflection:
            return out << "SurfaceType::RoughReflection";
        case SurfaceType::RoughRefraction:
            return out << "SurfaceType::RoughRefraction";
        case SurfaceType::Delta:
            return out << "SurfaceType::Delta";
        case SurfaceType::Rough:
            return out << "SurfaceType::Rough";
        case SurfaceType::Material:
            return out << "SurfaceType::Material";
        case SurfaceType::AreaEmitter:
            return out << "SurfaceType::AreaEmitter";
        case SurfaceType::Medium:
            return out << "SurfaceType::Medium";
        }
    }
#endif

} // namespace prayground

