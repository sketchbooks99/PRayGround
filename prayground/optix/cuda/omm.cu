#pragma once

#include <optix.h>
#include <optix_micromap.h>
#include <vector_types.h>
#include <prayground/math/vec.h>
#include <prayground/optix/util.h>
#include <prayground/optix/omm.h>

namespace prayground {
    DEVICE float evaluateTransparencyInSingleMicroTriangle(Vec2f uv0, Vec2f uv1, Vec2f uv2, OpacityMicromap::MicroBarycentrics bc, cudaTextureObject_t texture)
    {
        return 1.0f;
    }

    GLOBAL void generateOpacityMap(uint16_t* out_omm_data, Vec2i tex_size, Vec2f uv0, Vec2f uv1, Vec2f uv2, cudaTextureObject_t texture)
    {

    }

    extern "C" HOST void generateOpacityMapByTexture(
        uint16_t* out_omm_data, 
        int32_t subdivision_level, 
        OptixOpacityMicromapFormat format,
        Vec2i tex_size, 
        Vec2f uv0, Vec2f uv1, Vec2f uv2, 
        cudaTextureObject_t texture
    ) {
        
    }
}