#pragma once 

#include <optix.h>
#include <prayground/math/vec_math.h>
#include <prayground/math/matrix.h>
#include <prayground/optix/sbt.h>
#include <prayground/core/interaction.h>

using namespace prayground;

using Spectrum = SampledSpectrum;

using ConstantTexture = ConstantTexture_<Spectrum>;
using CheckerTexture = CheckerTexture_<Spectrum>;

struct AreaEmitterInfo
{
    void* shape_data;
    Matrix4f objToWorld;
    Matrix4f worldToObj;

    unsigned int sample_id;
    unsigned int pdf_id;
    
    OptixTraversableHandle gas_handle;
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

    AreaEmitterInfo* lights;
    int num_lights;

    SampledSpectrum* white_spd;
    SampledSpectrum* cyan_spd;
    SampledSpectrum* magenta_spd;
    SampledSpectrum* yellow_spd;
    SampledSpectrum* red_spd;
    SampledSpectrum* green_spd;
    SampledSpectrum* blue_spd;

    float white;
};

struct CameraData 
{
    float3 origin; 
    float3 lookat;
    float3 U; 
    float3 V; 
    float3 W;
    float fov;
    float aspect;
    float aperture;
    float focus_distance;
    float farclip;
};

struct RaygenData
{
    CameraData camera;
};

struct HitgroupData
{
    void* shape_data;
    SurfaceInfo surface_info;
};

struct MissData
{
    void* env_data;
};

struct EmptyData
{

};