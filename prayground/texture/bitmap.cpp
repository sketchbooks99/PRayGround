#include "bitmap.h"

#include <prayground/core/file_util.h>

namespace prayground {

// ---------------------------------------------------------------------
template <typename PixelT>
BitmapTexture_<PixelT>::BitmapTexture_(const std::filesystem::path& filename, int prg_id)
: Texture(prg_id)
{
    std::optional<std::filesystem::path> filepath = pgFindDataPath(filename);
    if (!filepath)
    {
        pgLogFatal("The texture file '" + filename.string() + "' is not found.");
        int width = 512;
        int height = 512;
        Vec_t magenta;
        if constexpr (std::is_same_v<PixelT, unsigned char>)
            magenta = make_uchar4(255, 0, 255, 255);
        else 
            magenta = make_float4(1.0f, 0.0f, 1.0f, 1.0f);
        std::vector<Vec_t> pixels(width * height, magenta);
        m_bitmap = Bitmap_<PixelT>(
            PixelFormat::RGBA, width, height, reinterpret_cast<Bitmap_<PixelT>::Type*>(pixels.data()));
    }
    else
    {
        m_bitmap = Bitmap_<PixelT>(filepath.value(), PixelFormat::RGBA);
    }

    // Initialize texture description
    m_tex_desc.addressMode[0] = cudaAddressModeWrap;
    m_tex_desc.addressMode[1] = cudaAddressModeWrap;
    m_tex_desc.filterMode = cudaFilterModeLinear;
    m_tex_desc.normalizedCoords = 1;
    m_tex_desc.sRGB = 1;
    if constexpr (std::is_same_v<PixelT, float>)
        m_tex_desc.readMode = cudaReadModeElementType;
    else
        m_tex_desc.readMode = cudaReadModeNormalizedFloat;
}

template <typename PixelT>
BitmapTexture_<PixelT>::BitmapTexture_(const std::filesystem::path& filename, cudaTextureDesc desc, int prg_id)
: BitmapTexture_<PixelT>(filename, prg_id)
{
    m_tex_desc = desc;
    if constexpr (std::is_same_v<PixelT, float>)
        m_tex_desc.readMode = cudaReadModeElementType;
    else
        m_tex_desc.readMode = cudaReadModeNormalizedFloat;
}

// ---------------------------------------------------------------------
template <typename PixelT>
void BitmapTexture_<PixelT>::copyToDevice()
{
    // Alloc CUDA array in device memory.
    int32_t pitch = m_bitmap.width() * sizeof(Vec_t);

    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<Vec_t>();

    CUDA_CHECK( cudaMallocArray( &d_array, &channel_desc, m_bitmap.width(), m_bitmap.height() ) );
    PixelT* raw_data = m_bitmap.data();
    CUDA_CHECK( cudaMemcpy2DToArray( d_array, 0, 0, raw_data, pitch, pitch, m_bitmap.height(), cudaMemcpyHostToDevice ) );

    // Create texture object.
    cudaResourceDesc res_desc;
    std::memset(&res_desc, 0, sizeof(cudaResourceDesc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = d_array;

    CUDA_CHECK( cudaCreateTextureObject( &d_texture, &res_desc, &m_tex_desc, nullptr ) );
    BitmapTexture::Data texture_data = { 
        .texture = d_texture
    };

    if (!d_data) 
        CUDA_CHECK( cudaMalloc( &d_data, sizeof(BitmapTexture::Data) ) );
    CUDA_CHECK( cudaMemcpy(
        d_data, 
        &texture_data, sizeof(BitmapTexture::Data), 
        cudaMemcpyHostToDevice
    ));
}

// ---------------------------------------------------------------------
template <typename PixelT>
void BitmapTexture_<PixelT>::free()
{
    if (d_texture != 0) 
        CUDA_CHECK( cudaDestroyTextureObject( d_texture ) );

    /// @todo Following function raised cudaErrorContextIsDestroyed when call it from App::close();
    /// if (d_array != 0)
    ///     CUDA_CHECK( cudaFreeArray( d_array ) );

    Texture::free();
}

template<typename PixelT>
int32_t BitmapTexture_<PixelT>::width() const
{
    return m_bitmap.width();
}

template<typename PixelT>
int32_t BitmapTexture_<PixelT>::height() const
{
    return m_bitmap.height();
}

template<typename PixelT>
void BitmapTexture_<PixelT>::setTextureDesc(const cudaTextureDesc& desc)
{
    m_tex_desc = desc;
}

template<typename PixelT>
cudaTextureDesc BitmapTexture_<PixelT>::textureDesc() const
{
    return m_tex_desc;
}

template class BitmapTexture_<float>;
template class BitmapTexture_<unsigned char>;

}