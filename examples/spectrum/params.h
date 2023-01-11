#pragma once 

#include <optix.h>
#include <prayground/math/vec.h>
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

    uint32_t sample_id;
    
    OptixTraversableHandle gas_handle;
};

struct LaunchParams 
{
    uint32_t width, height;
    uint32_t samples_per_launch;
    uint32_t max_depth;
    int frame;
    Vec4u* result_buffer;
    Vec4f* accum_buffer;
    OptixTraversableHandle handle;

    AreaEmitterInfo* lights;
    int num_lights;

    RGB2Spectrum rgb2spectrum;
};

struct RaygenData
{
    LensCamera::Data camera;
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