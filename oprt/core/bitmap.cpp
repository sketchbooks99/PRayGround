#include "bitmap.h"
#include "color.h"
#include "cudabuffer.h"
#include "file_util.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../ext/stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../ext/stb/stb_image_write.h"

namespace oprt {

template <typename PixelType>
Bitmap_<PixelType>::Bitmap_() 
{

}

// --------------------------------------------------------------------
template <typename PixelType>
Bitmap_<PixelType>::Bitmap_(Format format, int width, int height, PixelType* data)
{
    allocate(format, width, height);
    if (data)
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
void Bitmap_<PixelType>::allocate(Format format, int width, int height)
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
    std::vector<PixelType> zero_arr(m_channels * m_width * m_height, static_cast<PixelType>(0));
    m_data = new PixelType[m_width * m_height * m_channels];
    memcpy(m_data, zero_arr.data(), zero_arr.size() * sizeof(PixelType));
}

// --------------------------------------------------------------------
template <typename PixelType>
void Bitmap_<PixelType>::fillData(PixelType* data, int width, int height, int offset_x, int offset_y)
{
    Assert(m_data, "Please allocate the bitmap before filling data with specified range.");

    if (m_width < offset_x + width || m_height < offset_y + height)
    {
        Message(MSG_WARNING, "The range of data to fill which specified by offset and resolution exceeded the dimension of the bitmap.");
        height = m_height - offset_y;
        width = m_height - offset_x;
    }

    for (int y = offset_y; y < offset_y + height; y++)
    {
        int dst_base = (y * m_width + offset_x) * m_channels;
        int src_base = ((y - offset_y) * width) * m_channels;
        memcpy(&m_data[dst_base], &data[src_base], sizeof(PixelType) * width * m_channels);
    }
}

// --------------------------------------------------------------------
template <typename PixelType>
void Bitmap_<PixelType>::load(const std::filesystem::path& filepath) {
    // 画像ファイルが存在するかのチェック
    Assert(std::filesystem::exists(filepath), "The image file '" + filepath.string() + "' is not found.");

    std::string file_extension = getExtension(filepath);

    unsigned char* data = nullptr;
    if (file_extension == ".png" || file_extension == ".PNG") {
        m_format = Format::RGBA;
        data = stbi_load(filepath.string().c_str(), &m_width, &m_height, &m_channels, STBI_rgb_alpha);
    }
    else if (file_extension == ".jpg" || file_extension == ".EXR")
    {
        m_format = Format::RGB;
        data = stbi_load(filepath.string().c_str(), &m_width, &m_height, &m_channels, STBI_rgb);
    }
    else if (file_extension == ".hdr" || file_extension == ".HDR")
    {
        Message(MSG_WARNING, "Sorry! Bitmap doesn't support to load HDR image currently.");
        return;
    }
    else 
    {
        Message(MSG_WARNING, "This format is not loadable with bitmap.");
        return;
    }

    m_data = new PixelType[m_width * m_height * m_channels];

    // unsigned char の場合はそのままコピー
    if constexpr (std::is_same_v<PixelType, unsigned char>)
    {
        memcpy(m_data, data, sizeof(PixelType) * m_width * m_height * m_channels);
    }
    /// PixelTypeが浮動小数点型の場合は [0, 255] -> [0, 1]で正規化する
    /// その他の方の場合は、警告を出し unsigned char -> PixelType で型変換するのみ
    else
    {
        PixelType denom = static_cast<PixelType>(255.0);
        if constexpr (!std::is_same_v<PixelType, float>)
        {
            Message(MSG_WARNING, "This PixelType is not recommended to load image (Recommended ... unsigned char or float).",
                             "It may use too large memory space to store pixel values and may degrade the performance of application.");
            denom = static_cast<PixelType>(1);
        }
        for (int x = 0; x < m_width; x++)
        {
            for (int y = 0; y < m_height; y++)
            {
                for (int c = 0; c < m_channels; c++)
                {
                    int idx = (y * m_width + x) * m_channels + c;
                    m_data[idx] = static_cast<PixelType>(data[idx] / denom);
                }
            }
        }
    }
}

// --------------------------------------------------------------------
template <typename PixelType>
void Bitmap_<PixelType>::write(const std::filesystem::path& filepath, int quality) const
{
    std::string file_extension = getExtension(filepath);

    unsigned char* data = new unsigned char[m_width * m_height * m_channels];
    if constexpr (std::is_same_v<PixelType, unsigned char>)
    {
        memcpy(data, m_data, m_width * m_height * m_channels);
    }
    // Tの型がfloatの場合は中身を float [0.0f, 1.0f] -> unsigned char [0, 255] に変換する
    else 
    {
        for (int i=0; i<m_width; i++)
        {
            for (int j=0; j<m_height; j++)
            {
                if constexpr (std::is_same_v<PixelType, float>)
                {
                    int idx = (i * m_height + j) * m_channels;
                    float4 fcolor{0.0f, 0.0f, 0.0f, 1.0f};
                    memcpy(&fcolor, &m_data[idx], sizeof(float) * m_channels);
                    uchar4 ucolor = make_color(fcolor, false);
                    memcpy(&data[idx], &ucolor, m_channels);
                }
                else
                {
                    for (int c = 0; c < m_channels; c++)
                    {
                        int idx = (i * m_height + j) * m_channels + c;
                        data[idx] = static_cast<unsigned char>(m_data[idx]);
                    }
                }
            }
        }
    }
    
    if (file_extension == ".png" || file_extension == ".PNG")
    {
        stbi_flip_vertically_on_write(true);
        stbi_write_png(filepath.string().c_str(), m_width, m_height, m_channels, data, m_width * m_channels);
        delete[] data;
    }
    else if (file_extension == ".jpg" || file_extension == ".JPG")
    {
        stbi_flip_vertically_on_write(true);
        stbi_write_jpg(filepath.string().c_str(), m_width, m_height, m_channels, data, quality);
        delete[] data;
    }
    else if (file_extension == ".exr" || file_extension == ".EXR")
    {
        Message(MSG_WARNING, "Sorry! Bitmap doesn't support to write out image with .exr format currently.");
        return;
    }
    else 
    {
        Message(MSG_WARNING, "This format is not writable with bitmap.");
        return;
    }
}

// --------------------------------------------------------------------
template <typename PixelType>
void Bitmap_<PixelType>::copyToDevice() 
{
    // CPU側のデータが準備されているかチェック
    Assert(m_data, "Image data in the host side has been not allocated yet.");

    // GPU上に画像データを準備
    CUDABuffer<PixelType> d_buffer;
    d_buffer.copyToDevice(m_data, m_width * m_height * m_channels*sizeof(PixelType));
    d_data = d_buffer.deviceData();
}

// --------------------------------------------------------------------
template <typename PixelType>
void Bitmap_<PixelType>::copyFromDevice()
{
    Assert(d_data, "No data has been allocated on the device yet.");

    CUDA_CHECK(cudaMemcpy(
        m_data,
        d_data, m_width * m_height * m_channels * sizeof(PixelType),
        cudaMemcpyHostToDevice
    ));
}

template class Bitmap_<float>;
template class Bitmap_<double>;
template class Bitmap_<unsigned char>;
template class Bitmap_<char>;
template class Bitmap_<int>;

}