#pragma once

#include <prayground/math/vec.h>
#include "textures.h"

using namespace prayground;

// Environment map importance sampling data
struct EnvmapSamplingData
{
    // Conditional CDF per row (width * height)
    float* conditional_cdf;
    // Row luminance sum (height)
    float* row_luminances;
    // Marginal CDF (height)
    float* marginal_cdf;
    // Texture sampling callable ID
    uint32_t texture_id;
    // Texture data pointer
    void* texture_data;
    // Resolution
    uint32_t width;
    uint32_t height;
    // Total luminance sum
    float total_luminance;
};

#ifdef __cplusplus
extern "C" {
#endif

using StarNightTexture = StarNightTexture_<Vec4f>;

// Bake procedural texture to device buffer using CUDA kernel
// Returns device pointer to Vec4f array (RGBA data)
Vec4f* bakeProceduralTextureToDevice(
    StarNightTexture::Data star_night_data,
    uint32_t width,
    uint32_t height
);

// Build environment map importance sampling data from device color buffer
// Returns device pointer to EnvmapSamplingData
EnvmapSamplingData buildEnvmapSamplingDataFromDevice(
    Vec4f* d_colors,
    uint32_t width,
    uint32_t height,
    uint32_t texture_id,
    void* texture_device_data
);

#ifdef __cplusplus
}
#endif
