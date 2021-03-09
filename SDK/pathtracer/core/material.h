#pragma once

#include <optix.h>
#include <sutil/vec_math.h>
#include "core_util.h"

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

    /** UV coordinate at intersection point. */
    float2 uv;

    /** Spectrum information of ray. */
    float3 spectrum;

    /** Type of material to identify the shading at intersected point. 
     * MEMO:
     * Can this be a pointer such as shared_ptr? Can optixTrace() propagate pointer? */
    
};

enum class MaterialType {
    Diffuse = 1u << 0,
    Metal = 1u << 1,
    Dielectric = 1u << 2,
    Emission = 1u << 3,
    Disney = 1u << 4
};

#if !defined(__CUDACC__)
inline std::ostream& operator<<(std::ostream& out, MaterialType type) {
    switch(type) {
    case MaterialType::Diffuse:
        return out << "MaterialType::Diffuse";
    case MaterialType::Metal:
        return out << "MaterialType::Metal";
    case MaterialType::Dielectric:
        return out << "MaterialType::Sphere";
    case MaterialType::Emission:
        return out << "MaterialType::Emission";
    case MaterialType::Disney:
        return out << "MaterialType::Disney";
    default:
        Throw("This MaterialType is not supported\n");
        return out << "";
    }
}
#endif

// This is abstract class for readability
class Material {
public:    
    virtual MaterialType type() const = 0;
};

// Dielectric material
struct Dielectric : public Material {
public:
    float3 mat_color;
    float ior;

public:
    Dielectric(float3 mat_color = make_float3(0.8f), float ior=1.52f)
    : mat_color(mat_color), ior(ior) {}

    MaterialType type() const override { return MaterialType::Dielectric; }
};

// Metal material
struct Metal : public Material {
public:
    float3 mat_color;
    float reflection;

public:
    Metal(float3 mat_color=make_float3(0.8f), float reflection=1.0f)
    : mat_color(mat_color), reflection(reflection) {}

    MaterialType type() const override { return MaterialType::Metal; }
};

// Diffuse material
class Diffuse : public Material {
public:
    float3 mat_color;
    bool is_normal;

public:
    Diffuse(float3 mat_color=make_float3(0.8f), bool is_normal=false)
    : mat_color(mat_color), is_normal(is_normal) {}
    
    MaterialType type() const override { return MaterialType::Diffuse; }
};

// Emissive material
class Emission : public Material {
public:
    float3 color;

public:
    Emission(float3 color=make_float3(1.0f)) : color(color) {}

    MaterialType type() const override { return MaterialType::Emission; }
};

}