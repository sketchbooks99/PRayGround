#pragma once 

#include <optix.h>
#include <prayground/math/vec_math.h>
#include <prayground/optix/sbt.h>

namespace prayground {

    using ConstantTexture = ConstantTexture_<Vec3f>;
    using CheckerTexture = CheckerTexture_<Vec3f>;

    struct Light
    {
        Vec3f pos;
    };

    struct LaunchParams
    {
        unsigned int width, height;
        Vec4u* result_buffer;
        Vec3f* normal_buffer;
        Vec3f* albedo_buffer;

        Light light;

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

} // namespace prayground