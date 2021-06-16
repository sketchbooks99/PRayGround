#include "bitmap.h"
#include "color.h"
#include "cudabuffer.h"
#include "file_util.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../ext/stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../ext/stb/stb_image_write.h"

namespace oprt {

/**
 * @todo Update implementation of bitmap class 
 * @note class T = float4, uchar4 or some vector type 
 *    -> typename PixelType = float, unsigned char or other primitive type.
 */

template <typename PixelType>
Bitmap_<PixelType>::Bitmap_() 
{

}

// --------------------------------------------------------------------
template <typename PixelType>
Bitmap_<PixelType>::Bitmap_(PixelType* data, int width, int height, Format format)
{
    allocate(width, height, format);
    memcpy(m_data, data, sizeof(PixelType) * m_width * m_height * m_channels);
}

// --------------------------------------------------------------------
template <typename PixelType>
Bitmap_<PixelType>::Bitmap_(const std::filesystem::path& filepath)
{
    load(filepath);
}

// --------------------------------------------------------------------
template <typename PixelType>
void Bitmap_<PixelType>::allocate(int width, int height, Format format)
{
    Assert(!this->m_data, "Image data in the host side is already allocated. Please use fillData() if you'd like to override bitmap data with your specified range.");

    m_width = width; 
    m_height = height;
    m_format = format;
    switch (m_format)
    {
        case Format::GRAY:
            m_channels = 1;
            break;
        case Format::GRAY_ALPHA:
            m_channels = 2;
            break;
        case Format::RGB:
            m_channels = 3;
            break;
        case Format::RGBA:
            m_channels = 4;
            break;
    }

    // ピクセルデータのゼロ初期化
    std::vector<PixelType> zero_arr(m_channels * m_width * m_height, static_cast<Element_t>(0));
    m_data = new PixelType[m_width * m_height * m_channels];
    memcpy(m_data, zero_arr.data(), zero_arr.size() * sizeof(PixelType));
}

// --------------------------------------------------------------------
/**
 * @todo 
 * - T = float4 でも jpg ファイルを読み込みたい場合に対応する...?
 * - テンプレートの特殊化
 */
template <typename PixelType>
void Bitmap_<PixelType>::load(const std::filesystem::path& filepath) {
    // 画像ファイルが存在するかのチェック
    Assert(std::filesystem::exists(filepath), "The image file '" + filepath + "' is not found.");

    using Loadable_t = std::conditional_t< 
        (std::is_same_v<PixelType, uchar4> || std::is_same_v<PixelType, float4>), 
        uchar4, 
        uchar3
    >;

    std::string file_extension = getExtension(filepath);

    if (file_extension == ".png" || file_extension == ".PNG")
        Assert(sizeof(Loadable_t) == 4, "The type of Bitmap_ must have 4 channels (RGBA) when loading PNG file.");
    else if (file_extension == ".jpg" || file_extension == ".JPG")
        Assert(sizeof(Loadable_t) == 3, "The type of Bitmap_ must have 3 channels (RGB) when loading JPG file.");

    Loadable_t* data = reinterpret_cast<Loadable_t*>(
        stbi_load(filepath.c_str(), &m_width, &m_height, &m_channels, sizeof(Loadable_t)));
    Assert(data, "Failed to load image file'" + filepath + "'");

    m_data = new T[m_width * m_height];
    // Tの型がfloatの場合は中身を unsigned char [0, 255] -> float [0.0f, 1.0f] に変換する
    if constexpr (std::is_same_v<PixelType, float4> || std::is_same_v<PixelType, float3>)
    {
        PixelType* float_data = new T[m_width*m_height];
        for (auto i=0; i<m_width; i++)
        {
            for (auto j=0; j<m_height; j++)
            {
                auto idx = i * m_height + j;
                float_data[idx] = color2float(data[idx]);
            }
        }
        memcpy(m_data, float_data, m_width*m_height*sizeof(PixelType));
        delete[] float_data;
    }
    else
    {
        memcpy(m_data, data, m_width*m_height*sizeof(PixelType));
    }
    stbi_image_free(data);
}

// --------------------------------------------------------------------
/**
 * @todo
 * テンプレートの特殊化
 */
template <typename PixelType>
void Bitmap_<PixelType>::write(const std::filesystem::path& filepath, int quality) const
{
    // Tの方によって出力時の型をuchar4 or uchar3 で切り替える。
    using Writable_t = std::conditional_t< 
        (std::is_same_v<PixelType, uchar4> || std::is_same_v<PixelType, float4>), 
        uchar4, 
        uchar3
    >;

    std::string file_extension = getExtension(filepath);

    Writable_t* data = new Writable_t[m_width*m_height];
    // Tの型がfloatの場合は中身を float [0.0f, 1.0f] -> unsigned char [0, 255] に変換する
    if constexpr (std::is_same_v<PixelType, float4> || std::is_same_v<PixelType, float3>)
    {
        for (int i=0; i<m_width; i++)
        {
            for (int j=0; j<m_height; j++)
            {
                int idx = i * m_height + j;
                data[idx] = make_color(m_data[idx], gamma_enabled);
            }
        }
    }
    else 
    {
        memcpy(data, m_data, m_width*m_height*sizeof(PixelType));
    }
    
    if (file_extension == ".png" || file_extension == ".PNG")
    {
        stbi_flip_vertically_on_write(true);
        stbi_write_png(filepath.c_str(), m_width, m_height, m_channels, data, 0);
    }
    else if (file_extension == ".jpg" || file_extension == ".JPG")
    {
        stbi_write_jpg(filepath.c_str(), m_width, m_height, m_channels, data, quality);
    }

    delete[] data;
}

template <typename PixelType>
void Bitmap_<PixelType>::load(const std::filesystem::path& filepath)
{
    Assert(std::filesystem::exists(filepath), "The image file '" + filepath + "' is not found.");

    // 拡張子の取得
    std::string file_extension = getExtension(filepath);

    
}

// --------------------------------------------------------------------
template <typename PixelType>
void Bitmap_<PixelType>::copyToDevice() {
    // CPU側のデータが準備されているかチェック
    Assert(m_data, "Image data in the host side has been not allocated yet.");

    // GPU上に画像データを準備
    CUDABuffer<T> d_buffer;
    d_buffer.copyToDevice(m_data, m_width*m_height*sizeof(PixelType));
    d_data = d_buffer.deviceData();
}

template class Bitmap_<float>;
template class Bitmap_<unsigned char>;
template class Bitmap_<int>;

}