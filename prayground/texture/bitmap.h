#pragma once

#include <prayground/core/texture.h>

#ifndef __CUDACC__
    #include <prayground/core/bitmap.h>
#endif

namespace prayground {

struct BitmapTextureData {
    cudaTextureObject_t texture;
};

#ifndef __CUDACC__

template <typename PixelType>
class BitmapTexture_ final : public Texture {
public:
    explicit BitmapTexture_(const std::filesystem::path& filename, int prg_id);

    void copyToDevice() override;
    void free() override;

private:
    using Vec_t = std::conditional_t<std::is_same_v<PixelType, float>, float4, uchar4>;

    void _initTextureDesc() {
        m_tex_desc.addressMode[0] = cudaAddressModeClamp;
        m_tex_desc.addressMode[1] = cudaAddressModeClamp;
        m_tex_desc.filterMode = cudaFilterModeLinear;
        m_tex_desc.readMode = cudaReadModeNormalizedFloat;
        m_tex_desc.normalizedCoords = 1;
        // m_tex_desc.maxAnisotropy = 1;
        // m_tex_desc.maxMipmapLevelClamp = 99;
        // m_tex_desc.minMipmapLevelClamp = 0;
        // m_tex_desc.mipmapFilterMode = cudaFilterModePoint;
        // m_tex_desc.borderColor[0] = 0.0f;
        m_tex_desc.sRGB = 1;
    }

    std::shared_ptr<Bitmap_<PixelType>> m_bitmap;

    cudaTextureDesc m_tex_desc {};
    cudaTextureObject_t d_texture;
    cudaArray_t d_array { nullptr };
};

using BitmapTexture = BitmapTexture_<unsigned char>;
using FloatBitmapTexture = BitmapTexture_<float>;

#endif

}