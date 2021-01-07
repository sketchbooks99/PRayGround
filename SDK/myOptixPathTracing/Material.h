#pragma once

#include <optix.h>
#include <sutil/vec_math.h>

enum class MatType {
    DIFFUSE = 1u << 0,
    METAL = 1u << 1,
    DIELECTRIC = 1u << 2,
    EMISSION = 1u << 3
};

// This is abstract class for readability
struct Material {
    Material(MatType mattype) : mattype(mattype) {}
    const bool isEqualType(MatType mattype) { 
        return this->mattype == mattype;
    }
private:
    MatType mattype;
};

// Dielectric material
struct Dielectric : public Material{
    Dielectric(float3 mat_color = make_float3(0.8f), float ior=1.52f) : mat_color(mat_color), ior(ior), Material(MatType::DIELECTRIC) {}
    float3 mat_color;
    float ior;
};

// Metal material
struct Metal : public Material {
    Metal(float3 mat_color=make_float3(0.8f), float reflection=1.0f) : mat_color(mat_color), reflection(reflection), Material(MatType::METAL) {}
    float3 mat_color;
    float reflection;
};

// Diffuse material
struct Diffuse : public Material {
    Diffuse(float3 mat_color=make_float3(0.8f), bool is_normal=false)
    : mat_color(mat_color), is_normal(is_normal), Material(MatType::DIFFUSE) {}
    float3 mat_color;
    bool is_normal;
};

// Emissive material
struct Emission : public Material {
    Emission(float3 color=make_float3(1.0f)) : color(color), Material(MatType::EMISSION) {}
    float3 color;
};