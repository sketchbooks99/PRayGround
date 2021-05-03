#pragma once

#include "../core/texture.h"
#include "../core/file_util.h"

#ifndef __CUDACC__
    #define STB_IMAGE_IMPLEMENTATION
    #include "../ext/stb/stb_image.h"
#endif

namespace oprt {

struct ImageTextureData {
    cudaTextureObject_t texture;
};

enum ImageFormat {
    UNSIGNED_BYTE4,
    FLOAT4, 
    FLOAT3
};

#ifndef __CUDACC__
class ImageTexture final : public Texture {
public:
    explicit ImageTexture(const std::string& filename)
    {   
        std::string filepath = find_datapath(filename).string();
        uint8_t* d = stbi_load( filepath.c_str(), &width, &height, &channels, STBI_rgb_alpha );
        Assert(d, "Failed to load image file'"+filename+"'");
        data = new uchar4[width*height];
        format = UNSIGNED_BYTE4;
        memcpy(data, d, width*height*STBI_rgb_alpha);

        stbi_image_free(d);

        _init_texture_desc();
    }

    ~ImageTexture() noexcept(false) {
        if (d_texture != 0) 
            CUDA_CHECK( cudaDestroyTextureObject( d_texture ) );
        if (d_array != 0)
            CUDA_CHECK( cudaFreeArray( d_array ) );
    }

    float3 eval(const SurfaceInteraction& si) const override { return make_float3(1.0f); }

    void prepare_data() override
    {
        // Alloc CUDA array in device memory.
        int32_t pitch = width * 4 * sizeof( unsigned char );
        cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<uchar4>();

        CUDA_CHECK( cudaMallocArray( &d_array, &channel_desc, width, height ) );
        CUDA_CHECK( cudaMemcpy2DToArray( d_array, 0, 0, data, pitch, pitch, height, cudaMemcpyHostToDevice ) );

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

    TextureType type() const override { return TextureType::Image; }
private:
    void _init_texture_desc() {
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
        // tex_desc.sRGB = 1;
    }

    int width, height;
    int channels;
    uchar4* data;
    ImageFormat format;

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