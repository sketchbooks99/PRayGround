#include "bitmap.h"
#include "color.h"
#include "cudabuffer.h"
#include "file_util.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../ext/stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../ext/stb/stb_image_write.h"
#define TINYEXR_IMPLEMENTATION
#include "../ext/tinyexr/tinyexr.h"

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
Bitmap_<PixelType>::Bitmap_(const std::filesystem::path& filename)
{
    load(filename);
}

// --------------------------------------------------------------------
template <typename PixelType>
Bitmap_<PixelType>::Bitmap_(const std::filesystem::path& filename, Format format)
{
    load(filename, format);
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
        case Format::UNKNOWN:
        default:
            m_channels = 4;
            break;
    }

    // Zero-initialization of pixel data
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
void Bitmap_<PixelType>::load(const std::filesystem::path& filename, Format format)
{
    m_format = format;
    load(filename);
}

// --------------------------------------------------------------------
template <typename PixelType>
void Bitmap_<PixelType>::load(const std::filesystem::path& filename)
{
    Message(MSG_WARNING, "oprt::Bitmap_<PixelType>::load(): This function is only implemented for when PixelType is unsigned char or float.");
}

// --------------------------------------------------------------------
template <>
void Bitmap_<unsigned char>::load(const std::filesystem::path& filename)
{
    std::optional<std::filesystem::path> filepath = findDatapath(filename);
    Assert(filepath, "oprt::Bitmap_<unsigned char>::load(): The input file for bitmap '" + filename.string() + "' is not found.");

    auto ext = getExtension(filepath.value());

    if (ext == ".png" || ext == ".PNG")
    {
        if (m_format == Format::UNKNOWN)
            m_format = Format::RGBA;
    }
    else if (ext == ".jpg" || ext == ".JPG")
    {
        if (m_format == Format::UNKNOWN)
            m_format = Format::RGB;
    }
    else if (ext == ".exr" || ext == ".EXR")
    {
        Message(MSG_ERROR, "oprt::Bitmap_<unsigned char>::load(): EXR format can be loaded only in BitmapFloat.");
        return;
    }
    m_data = stbi_load(filepath.value().string().c_str(), &m_width, &m_height, &m_channels, type2channels[m_format]);
    m_channels = type2channels[m_format];
}

// --------------------------------------------------------------------
template <>
void Bitmap_<float>::load(const std::filesystem::path& filename)
{
    std::optional<std::filesystem::path> filepath = findDatapath(filename);
    Assert(filepath, "oprt::Bitmap_<float>::load(): The input file for bitmap '" + filename.string() + "' is not found.");

    auto ext = getExtension(filepath.value());

    // EXR 形式の場合はそのまま読み込む
    if (ext == ".exr" || ext == ".EXR")
    {
        m_format = m_format== Format::UNKNOWN ? Format::RGBA : m_format;

        const char* err = nullptr;
        int result = LoadEXR(&m_data, &m_width, &m_height, filepath.value().string().c_str(), &err);
        if (result != TINYEXR_SUCCESS)
        {
            if (err)
            {
                Message(MSG_ERROR, "oprt::Bitmap_<float>::load():", err);
                FreeEXRErrorMessage(err);
                return;
            }
        }
    }
    // PNG/JPG 形式の場合は uint8_t* [0, 255] -> float* [0, 1] に変換 
    else
    {
        if (ext == ".png" || ext == ".PNG") 
            m_format = m_format == Format::UNKNOWN ? Format::RGBA : m_format;
        else if (ext == ".jpg" || ext == ".JPG")
            m_format = m_format == Format::UNKNOWN ? Format::RGB  : m_format;

        uint8_t* raw_data = stbi_load(filepath.value().string().c_str(), &m_width, &m_height, &m_channels, type2channels[m_format]);
        m_channels = type2channels[m_format];
        m_data = new float[m_width * m_height * m_channels];
        for (int i = 0; i < m_width * m_height * m_channels; i += m_channels)
        {
            for (int c = 0; c < m_channels; c++)
            {
                m_data[i + c] = static_cast<float>(raw_data[i + c]) / 255.0f;
            }
        }
        stbi_image_free(raw_data);
    }
}

// --------------------------------------------------------------------
template <typename PixelType>
void Bitmap_<PixelType>::write(const std::filesystem::path& filepath, int quality) const
{
    std::string ext = getExtension(filepath);

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
    
    if (ext == ".png" || ext == ".PNG")
    {
        stbi_flip_vertically_on_write(true);
        stbi_write_png(filepath.string().c_str(), m_width, m_height, m_channels, data, m_width * m_channels);
        delete[] data;
    }
    else if (ext == ".jpg" || ext == ".JPG")
    {
        stbi_flip_vertically_on_write(true);
        stbi_write_jpg(filepath.string().c_str(), m_width, m_height, m_channels, data, quality);
        delete[] data;
    }
    else if (ext == ".exr" || ext == ".EXR")
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
const std::string vert_source = R"(
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texcoord;
out vec2 vTexCoord;

void main()
{
    gl_Position(position, 1.0);
    vTexCoord = texcoord;
}
)";

const std::string frag_source = R"(
#version 330 core

in vec2 vTexCoord;
out vec4 fragColor;

uniform sampler2D tex;

void main()
{
    fragColor = texture( tex, vTexCoord );
}
)";

template <typename PixelType>
void Bitmap_<PixelType>::draw() const
{
    draw(0, 0, m_width, m_height);
}

template <typename PixelType>
void Bitmap_<PixelType>::draw(int32_t x, int32_t y) const
{
    draw(x, y, m_width, m_height);
}

template <typename PixelType>
void Bitmap_<PixelType>::draw(int32_t x, int32_t y, int32_t width, int32_t height) const
{

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