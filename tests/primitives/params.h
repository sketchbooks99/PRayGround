#pragma once 

#include <optix.h>
#include <prayground/math/vec.h>
#include <prayground/texture/constant.h>
#include <prayground/texture/checker.h>

namespace prayground
{
    using ConstantTexture = ConstantTexture_<Vec3f>;
    using CheckerTexture = CheckerTexture_<Vec3f>;

    struct LaunchParams {
        uint32_t width;
        uint32_t height;
        int32_t frame;

        Vec4u* result_buffer;
        OptixTraversableHandle handle;
    };

    struct PhongData {
        Vec3f emission;
        Vec3f ambient;
        Vec3f diffuse;
        Vec3f specular;
        float shininess;
    };

} // namespace prayground