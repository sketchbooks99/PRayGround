#include "bitmap.h"

#include "../ext/stb/stb_image.h"
#include "../core/file_util.h"

namespace oprt {

// ---------------------------------------------------------------------
BitmapTexture::BitmapTexture(const std::filesystem::path& filename)
{
    std::optional<std::filesystem::path> filepath = findDatapath(filename);
    if (!filepath)
    {
        Message(MSG_ERROR, "The texture file '" + filename.string() + "' is not found.");
        int width = 512;
        int height = 512;
        uchar4 magenta = make_uchar4(255, 0, 255, 255);
        std::vector<uchar4> pixels(width * height, magenta);
        m_bitmap = std::make_shared<Bitmap>(Bitmap::Format::RGBA, width, height, reinterpret_cast<Bitmap::Type*>(pixels.data()));
    }
    else
    {
        m_bitmap = std::make_shared<Bitmap>(filepath.value(), Bitmap::Format::RGBA);
    }
    _initTextureDesc();
}

// ---------------------------------------------------------------------
void BitmapTexture::prepareData()
{
    // Alloc CUDA array in device memory.
    int32_t pitch = m_bitmap->width() * 4 * sizeof( unsigned char );
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<uchar4>();

    CUDA_CHECK( cudaMallocArray( &d_array, &channel_desc, m_bitmap->width(), m_bitmap->height() ) );
    CUDA_CHECK( cudaMemcpy2DToArray( d_array, 0, 0, m_bitmap->data(), pitch, pitch, m_bitmap->height(), cudaMemcpyHostToDevice ) );

    // Create texture object.
    cudaResourceDesc res_desc;
    std::memset(&res_desc, 0, sizeof(cudaResourceDesc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = d_array;

    CUDA_CHECK( cudaCreateTextureObject( &d_texture, &res_desc, &m_tex_desc, nullptr ) );
    BitmapTextureData texture_data = { 
        d_texture
    };

    CUDA_CHECK( cudaMalloc( &d_data, sizeof(BitmapTextureData) ) );
    CUDA_CHECK( cudaMemcpy(
        d_data, 
        &texture_data, sizeof(BitmapTextureData), 
        cudaMemcpyHostToDevice
    ));
}

void BitmapTexture::freeData()
{
    if (d_texture != 0) 
        CUDA_CHECK( cudaDestroyTextureObject( d_texture ) );
    if (d_array != 0)
        CUDA_CHECK( cudaFreeArray( d_array ) );
}

}