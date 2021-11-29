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
    BitmapTexture_(const std::filesystem::path& filename, int prg_id);
    BitmapTexture_(const std::filesystem::path& filename, cudaTextureDesc desc, int prg_id);

    void copyToDevice() override;
    void free() override;

    int32_t width() const;
    int32_t height() const;

    void setTextureDesc(const cudaTextureDesc& desc);
    cudaTextureDesc textureDesc() const;

private:
    using Vec_t = std::conditional_t<std::is_same_v<PixelType, float>, float4, uchar4>;

    Bitmap_<PixelType> m_bitmap;

    cudaTextureDesc m_tex_desc {};
    cudaTextureObject_t d_texture;
    cudaArray_t d_array { nullptr };
};

using BitmapTexture = BitmapTexture_<unsigned char>;
using FloatBitmapTexture = BitmapTexture_<float>;

#endif

}