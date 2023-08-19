#pragma once

#include <prayground/core/texture.h>
#include <prayground/core/bitmap.h>

namespace prayground {

template <typename PixelT>
class BitmapTexture_ final : public Texture, Bitmap_<PixelT> {
public:
    using ColorType = PixelT;
    struct Data
    {
        cudaTextureObject_t texture;
    };

#ifndef __CUDACC__
    BitmapTexture_(const std::filesystem::path& filename, int prg_id);
    BitmapTexture_(const std::filesystem::path& filename, cudaTextureDesc desc, int prg_id);

    constexpr TextureType type() override;

    void copyToDevice() override;
    void free() override;

    void setTextureDesc(const cudaTextureDesc& desc);
    cudaTextureDesc textureDesc() const;

private:
    using Vec_t = std::conditional_t<std::is_same_v<PixelT, float>, float4, uchar4>;

    /*Bitmap_<PixelT> m_bitmap;*/

    cudaTextureDesc m_tex_desc {};
    cudaTextureObject_t d_texture;
    cudaArray_t d_array { nullptr };
#endif // __CUDACC__
};

using BitmapTexture = BitmapTexture_<unsigned char>;
using FloatBitmapTexture = BitmapTexture_<float>;

}