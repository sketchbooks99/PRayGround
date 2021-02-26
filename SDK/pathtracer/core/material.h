#pragma once

#include <optix.h>
#include <sutil/vec_math.h>

enum class MaterialType {
    Diffuse = 1u << 0,
    Metal = 1u << 1,
    Dielectric = 1u << 2,
    Emission = 1u << 3
};

// This is abstract class for readability
struct Material {
    virtual MaterialType type() const = 0;
};

// Dielectric material
struct Dielectric : public Material{
    Dielectric(float3 mat_color = make_float3(0.8f), float ior=1.52f) : mat_color(mat_color), ior(ior) {}
    MaterialType type() const override { return MaterialType::Dielectric; }
    float3 mat_color;
    float ior;
};

// Metal material
struct Metal : public Material {
    Metal(float3 mat_color=make_float3(0.8f), float reflection=1.0f) : mat_color(mat_color), reflection(reflection) {}
    MaterialType type() const override { return MaterialType::Metal; }
    float3 mat_color;
    float reflection;
};

// Diffuse material
struct Diffuse : public Material {
    Diffuse(float3 mat_color=make_float3(0.8f), bool is_normal=false)
    : mat_color(mat_color), is_normal(is_normal) {}
    MaterialType type() const override { return MaterialType::Diffuse; }
    float3 mat_color;
    bool is_normal;
};

// Emissive material
struct Emission : public Material {
    Emission(float3 color=make_float3(1.0f)) : color(color) {}
    MaterialType type() const override { return MaterialType::Emission; }
    float3 color;
};