#pragma once 

#include <optix.h>
#include <curand.h>
#include <curand_kernel.h>
#include <prayground/math/vec_math.h>
#include <prayground/optix/sbt.h>

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

    curandState_t* state;

    SurfaceProperty surface_property;

    SurfaceType surface_type;

    int trace_terminate;
    int radiance_evaled; // For NEE
};

struct LaunchParams 
{
    unsigned int width, height;
    unsigned int samples_per_launch;
    unsigned int max_depth;
    int subframe_index;
    uchar4* result_buffer;
    float4* accum_buffer;
    OptixTraversableHandle handle;
};

struct CameraData
{
    float3 origin; 
    float3 lookat; 
    float3 up;
    float fov;
    float aspect;
};

struct RaygenData
{
    CameraData camera;
};

struct HitgroupData
{
    void* shape_data;
    void* surface_data;

    unsigned int surface_program_id;   
    SurfaceType surface_type;
};

struct MissData
{
    void* env_data;
};

struct EmptyData
{

};

} // ::prayground