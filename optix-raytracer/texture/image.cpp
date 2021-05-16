#include "image.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../ext/stb/stb_image.h"
#include "../core/file_util.h"

namespace oprt {

// ---------------------------------------------------------------------
ImageTexture::ImageTexture(const std::string& filename)
{
    std::string filepath = findDatapath(filename).string();
    uint8_t* d = stbi_load( filepath.c_str(), &m_width, &m_height, &m_channels, STBI_rgb_alpha );
    Assert(d, "Failed to load image file'"+filename+"'");
    m_data = new uchar4[m_width*m_height];
    memcpy(m_data, d, m_width*m_height*STBI_rgb_alpha);

    stbi_image_free(d);

    _initTextureDesc();
}

// ---------------------------------------------------------------------
void ImageTexture::prepareData() 
{
    // Alloc CUDA array in device memory.
    int32_t pitch = m_width * 4 * sizeof( unsigned char );
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<uchar4>();

    CUDA_CHECK( cudaMallocArray( &d_array, &channel_desc, m_width, m_height ) );
    CUDA_CHECK( cudaMemcpy2DToArray( d_array, 0, 0, m_data, pitch, pitch, m_height, cudaMemcpyHostToDevice ) );

    // Create texture object.
    cudaResourceDesc res_desc;
    std::memset(&res_desc, 0, sizeof(cudaResourceDesc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = d_array;

    CUDA_CHECK( cudaCreateTextureObject( &d_texture, &res_desc, &tex_desc, nullptr ) );
    ImageTextureData image_texture_data = { 
        d_texture
    };

    CUDA_CHECK( cudaMalloc( &d_data, sizeof(ImageTextureData) ) );
    CUDA_CHECK( cudaMemcpy(
        d_data, 
        &image_texture_data, sizeof(ImageTextureData), 
        cudaMemcpyHostToDevice
    ));
}

}