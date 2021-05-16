#pragma once

#include "../core/texture.h"

namespace oprt {

struct ImageTextureData {
    cudaTextureObject_t texture;
};

#ifndef __CUDACC__
class ImageTexture final : public Texture {
public:
    explicit ImageTexture(const std::string& filename);

    ~ImageTexture() noexcept(false) {
        if (d_texture != 0) 
            CUDA_CHECK( cudaDestroyTextureObject( d_texture ) );
        if (d_array != 0)
            CUDA_CHECK( cudaFreeArray( d_array ) );
    }

    void prepareData() override;

    TextureType type() const override { return TextureType::Image; }
private:
    void _initTextureDesc() {
        tex_desc.addressMode[0] = cudaAddressModeClamp;
        tex_desc.addressMode[1] = cudaAddressModeClamp;
        tex_desc.filterMode = cudaFilterModeLinear;
        tex_desc.readMode = cudaReadModeNormalizedFloat;
        tex_desc.normalizedCoords = 1;
        // tex_desc.maxAnisotropy = 1;
        // tex_desc.maxMipmapLevelClamp = 99;
        // tex_desc.minMipmapLevelClamp = 0;
        // tex_desc.mipmapFilterMode = cudaFilterModePoint;
        // tex_desc.borderColor[0] = 0.0f;
        tex_desc.sRGB = 1;
    }

    int m_width, m_height;
    int m_channels;
    uchar4* m_data;

    cudaTextureDesc tex_desc {};
    cudaTextureObject_t d_texture;
    cudaArray_t d_array { nullptr };
};

#else
CALLABLE_FUNC float3 DC_FUNC(eval_image)(SurfaceInteraction* si, void* texdata) {
    const ImageTextureData* image = reinterpret_cast<ImageTextureData*>(texdata);
    float4 c = tex2D<float4>(image->texture, si->uv.x, si->uv.y);
    return make_float3(c.x, c.y, c.z);
}

#endif

}