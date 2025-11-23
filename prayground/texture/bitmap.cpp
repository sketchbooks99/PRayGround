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
            ColorType magenta;
            if constexpr (std::is_same_v<PixelT, unsigned char>)
                magenta = Vec4u(255, 0, 255, 255);
            else 
                magenta = Vec4f(1.0f, 0.0f, 1.0f, 1.0f);
            std::vector<ColorType> pixels(width * height, magenta);
            Bitmap_<PixelT>::allocate(PixelFormat::RGBA, width, height, reinterpret_cast<PixelT*>(pixels.data()));
        }
        else
        {
            Bitmap_<PixelT>::load(filepath.value(), PixelFormat::RGBA);
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

    template<typename PixelT>
    BitmapTexture_<PixelT>::BitmapTexture_(const std::shared_ptr<Bitmap_<PixelT>>& bitmap, int prg_id)
        : Texture(prg_id)
    {
        // Copy bitmap data to this texture
        int width = bitmap->width();
        int height = bitmap->height();
        int channels = bitmap->channels();

        // Determine pixel format from channels
        PixelFormat format;
        switch (channels) {
        case 1: format = PixelFormat::GRAY; break;
        case 2: format = PixelFormat::GRAY_ALPHA; break;
        case 3: format = PixelFormat::RGB; break;
        case 4: format = PixelFormat::RGBA; break;
        default:
            pgLogFatal("Invalid number of channels: " + std::to_string(channels));
            format = PixelFormat::RGBA;
        }

        // Allocate and copy data
        Bitmap_<PixelT>::allocate(format, width, height, bitmap->data());

        // Initialize texture description with default settings
        m_tex_desc.addressMode[0] = cudaAddressModeWrap;
        m_tex_desc.addressMode[1] = cudaAddressModeWrap;
        m_tex_desc.filterMode = cudaFilterModeLinear;
        m_tex_desc.normalizedCoords = 1;
        m_tex_desc.sRGB = 0;  // Heightmap should not use sRGB

        if constexpr (std::is_same_v<PixelT, float>)
            m_tex_desc.readMode = cudaReadModeElementType;
        else
            m_tex_desc.readMode = cudaReadModeNormalizedFloat;
    }


    template<typename PixelT>
    constexpr TextureType BitmapTexture_<PixelT>::type()
    {
        return TextureType::Bitmap;
    }

    // ---------------------------------------------------------------------
    template <typename PixelT>
    BitmapTexture_<PixelT>::ColorType BitmapTexture_<PixelT>::eval(const Vec2f& texcoord) const
    {
        int32_t x = clamp(texcoord.x(), 0.0f, 0.999f) * Bitmap_<PixelT>::width();
        int32_t y = clamp(texcoord.y(), 0.0f, 0.999f) * Bitmap_<PixelT>::height();
        ColorType pixel = std::get<ColorType>(Bitmap_<PixelT>::at(x, y));
        return pixel;
    }

    template <typename PixelT>
    BitmapTexture_<PixelT>::ColorType BitmapTexture_<PixelT>::eval(const Vec2i& pixel) const
    {
        ColorType color = std::get<ColorType>(Bitmap_<PixelT>::at(pixel.x(), pixel.y()));
        return color;
    }

    // ---------------------------------------------------------------------
    template <typename PixelT>
    void BitmapTexture_<PixelT>::copyToDevice()
    {
        int channels = Bitmap_<PixelT>::channels();
        int width = Bitmap_<PixelT>::width();
        int height = Bitmap_<PixelT>::height();
        PixelT* raw_data = Bitmap_<PixelT>::data();

        // Calculate pitch based on actual channel count
        int32_t element_size = channels * sizeof(PixelT);
        int32_t pitch = width * element_size;

        // Create channel descriptor based on number of channels
        cudaChannelFormatDesc channel_desc;
        if constexpr (std::is_same_v<PixelT, float>) {
            switch (channels) {
            case 1: channel_desc = cudaCreateChannelDesc<float>(); break;
            case 2: channel_desc = cudaCreateChannelDesc<float2>(); break;
            case 4: channel_desc = cudaCreateChannelDesc<float4>(); break;
            default:
                pgLogFatal("Unsupported channel count for float texture: " + std::to_string(channels));
            }
        }
        else if constexpr (std::is_same_v<PixelT, unsigned char>) {
            switch (channels) {
            case 1: channel_desc = cudaCreateChannelDesc<unsigned char>(); break;
            case 2: channel_desc = cudaCreateChannelDesc<uchar2>(); break;
            case 4: channel_desc = cudaCreateChannelDesc<uchar4>(); break;
            default:
                pgLogFatal("Unsupported channel count for uchar texture: " + std::to_string(channels));
            }
        }
        else {
            pgLogFatal("Unsupported pixel type for BitmapTexture");
        }

        if (!d_array)
            CUDA_CHECK(cudaMallocArray(&d_array, &channel_desc, width, height));
        CUDA_CHECK(cudaMemcpy2DToArray(d_array, 0, 0, raw_data, pitch, pitch, height, cudaMemcpyHostToDevice));

        // Create texture object only if not already created
        if (d_texture == 0) {
            cudaResourceDesc res_desc;
            std::memset(&res_desc, 0, sizeof(cudaResourceDesc));
            res_desc.resType = cudaResourceTypeArray;
            res_desc.res.array.array = d_array;

            CUDA_CHECK(cudaCreateTextureObject(&d_texture, &res_desc, &m_tex_desc, nullptr));
        }

        BitmapTexture_<PixelT>::Data texture_data = {
            .texture = d_texture
        };

        if (!d_data)
            CUDA_CHECK(cudaMalloc(&d_data, sizeof(BitmapTexture_<PixelT>::Data)));
        CUDA_CHECK(cudaMemcpy(
            d_data,
            &texture_data, sizeof(BitmapTexture_<PixelT>::Data),
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
    void BitmapTexture_<PixelT>::setTextureDesc(const cudaTextureDesc& desc)
    {
        m_tex_desc = desc;
    }

    template<typename PixelT>
    cudaTextureDesc BitmapTexture_<PixelT>::textureDesc() const
    {
        return m_tex_desc;
    }

    template<typename PixelT>
    cudaTextureObject_t BitmapTexture_<PixelT>::cudaTextureObject() const
    {
        return d_texture;
    }

    template class BitmapTexture_<float>;
    template class BitmapTexture_<unsigned char>;

} // namespace prayground