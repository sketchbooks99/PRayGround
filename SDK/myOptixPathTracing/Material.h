#pragma once

#include <optix.h>
#include <sutil/vec_math.h>

enum class MatType {
    Diffuse = 1u << 0,
    Metal = 1u << 1,
    Dielectric = 1u << 2,
    Emission = 1u << 3
};

// This is abstract class for readability
struct Material {
    virtual MatType type() const = 0;
private:
    MatType mattype;
};

// Dielectric material
struct Dielectric : public Material{
    Dielectric(float3 mat_color = make_float3(0.8f), float ior=1.52f) : mat_color(mat_color), ior(ior) {}
    MatType type() const override { return MatType::Dielectric; }
    float3 mat_color;
    float ior;
};

// Metal material
struct Metal : public Material {
    Metal(float3 mat_color=make_float3(0.8f), float reflection=1.0f) : mat_color(mat_color), reflection(reflection) {}
    MatType type() const override { return MatType::Metal; }
    float3 mat_color;
    float reflection;
};

// Diffuse material
struct Diffuse : public Material {
    Diffuse(float3 mat_color=make_float3(0.8f), bool is_normal=false)
    : mat_color(mat_color), is_normal(is_normal) {}
    MatType type() const override { return MatType::Diffuse; }
    float3 mat_color;
    bool is_normal;
};

// Emissive material
struct Emission : public Material {
    Emission(float3 color=make_float3(1.0f)) : color(color) {}
    MatType type() const override { return MatType::Emission; }
    float3 color;
};