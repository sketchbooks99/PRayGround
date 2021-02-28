#pragma once

#include <optix.h>
#include <sutil/vec_math.h>

struct Material;
using MaterialPtr = Material*;

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
     *  Can this be a pointer such as shared_ptr? Can optixTrace() propagate pointer? */
    MaterialPtr material_ptr;
};

enum class MaterialType {
    Diffuse = 1u << 0,
    Metal = 1u << 1,
    Dielectric = 1u << 2,
    Emission = 1u << 3
};

// This is abstract class for readability
struct Material {
    virtual MaterialType type() const = 0;
    virtual SUTIL_HOSTDEVICE float3 sample() = 0;
};

// Dielectric material
struct Dielectric : public Material {
    Dielectric(float3 mat_color = make_float3(0.8f), float ior=1.52f) : mat_color(mat_color), ior(ior) {}

    SUTIL_HOSTDEVICE float3 sample() override { return make_float3(0.0f); }
    MaterialType type() const override { return MaterialType::Dielectric; }

    float3 mat_color;
    float ior;
};

// Metal material
struct Metal : public Material {
    Metal(float3 mat_color=make_float3(0.8f), float reflection=1.0f) : mat_color(mat_color), reflection(reflection) {}

    SUTIL_HOSTDEVICE float3 sample() override { return make_float3(0.0f); }
    MaterialType type() const override { return MaterialType::Metal; }

    float3 mat_color;
    float reflection;
};

// Diffuse material
struct Diffuse : public Material {
    Diffuse(float3 mat_color=make_float3(0.8f), bool is_normal=false)
    : mat_color(mat_color), is_normal(is_normal) {}
    
    SUTIL_HOSTDEVICE float3 sample() override { return make_float3(0.0f); }
    MaterialType type() const override { return MaterialType::Diffuse; }

    float3 mat_color;
    bool is_normal;
};

// Emissive material
struct Emission : public Material {
    Emission(float3 color=make_float3(1.0f)) : color(color) {}

    SUTIL_HOSTDEVICE float3 sample() override {return make_float3(0.0f); }
    MaterialType type() const override { return MaterialType::Emission; }

    float3 color;
};