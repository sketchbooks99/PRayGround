#pragma once 

#include <optix.h>
#include <prayground/math/vec_math.h>
#include <prayground/optix/sbt.h>

namespace prayground {

    using CheckerTexture = CheckerTexture_<Vec3f>;
    using ConstantTexture = ConstantTexture_<Vec3f>;

    struct LaunchParams 
    {
        uint32_t width;
        uint32_t height;
        uint32_t samples_per_launch;
        uint32_t max_depth;
        int frame;
        Vec4u* result_buffer;
        OptixTraversableHandle handle;
    };

    struct RaygenData
    {
        Camera::Data camera;
    };

    struct HitgroupData
    {
        void* shape_data;
        Texture::Data texture;
    };

    struct MissData
    {
        void* env_data;
    };

    struct EmptyData
    {

    };

} // ::prayground