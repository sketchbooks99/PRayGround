#include "bitmap.h"
#include <prayground/core/color.h>
#include <prayground/core/cudabuffer.h>
#include <prayground/core/file_util.h>
#include <prayground/core/util.h>
#include <prayground/app/app_runner.h>

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#endif
#include <prayground/ext/stb/stb_image.h>

#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#endif
#include <prayground/ext/stb/stb_image_write.h>

#ifndef TINYEXR_IMPLEMENTATION
#define TINYEXR_IMPLEMENTATION
#endif
#include <prayground/ext/tinyexr/tinyexr.h>

namespace prayground {

template <typename PixelType>
Bitmap_<PixelType>::Bitmap_()
{
    
}

// --------------------------------------------------------------------
template <typename PixelType>
Bitmap_<PixelType>::Bitmap_(PixelFormat format, int width, int height, PixelType* data)
{
    allocate(format, width, height);
    if (data)
        memcpy(m_data.get(), data, sizeof(PixelType) * m_width * m_height * m_channels);
}

// --------------------------------------------------------------------
template <typename PixelType>
Bitmap_<PixelType>::Bitmap_(const std::filesystem::path& filename)
{
    load(filename);
}

// --------------------------------------------------------------------
template <typename PixelType>
Bitmap_<PixelType>::Bitmap_(const std::filesystem::path& filename, PixelFormat format)
{
    load(filename, format);
}

// --------------------------------------------------------------------
template <typename PixelType>
void Bitmap_<PixelType>::allocate(PixelFormat format, int width, int height)
{
    m_width = width; 
    m_height = height;
    m_format = format;
    switch (m_format)
    {
        case PixelFormat::GRAY:
            m_channels = 1;
            break;
        case PixelFormat::GRAY_ALPHA:
            m_channels = 2;
            break;
        case PixelFormat::RGB:
            m_channels = 3;
            break;
        case PixelFormat::RGBA:
            m_channels = 4;
            break;
        case PixelFormat::NONE:
        default:
            THROW("Invalid type of allocation");
    }

    // Zero-initialization of pixel data
    std::vector<PixelType> zero_arr(m_channels * m_width * m_height, static_cast<PixelType>(0));
    m_data = std::make_unique<PixelType[]>(m_channels * m_width * m_height);
    memcpy(m_data.get(), zero_arr.data(), m_channels * m_width * m_height * sizeof(PixelType));

    if (pgWindowInitialized())
        prepareGL();
}

// --------------------------------------------------------------------
template <typename PixelType>
void Bitmap_<PixelType>::setData(PixelType* data, int offset_x, int offset_y, int width, int height)
{
    ASSERT(m_data.get(), "Please allocate the bitmap before filling data with specified range.");

    if (m_width < offset_x + width || m_height < offset_y + height)
    {
        pgLogWarn("The range of data to fill which specified by offset and resolution exceeded the dimension of the bitmap.");
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

template <typename PixelType>
void Bitmap_<PixelType>::setData(PixelType* data, const int2& offset, const int2& res)
{
    setData(data, offset.x, offset.y, res.x, res.y); 
}

// --------------------------------------------------------------------
template <typename PixelType>
void Bitmap_<PixelType>::load(const std::filesystem::path& filename, PixelFormat format)
{
    m_format = format;
    load(filename);
}

// --------------------------------------------------------------------
template <typename PixelType>
void Bitmap_<PixelType>::load(const std::filesystem::path& filename)
{
    UNIMPLEMENTED();
}

// --------------------------------------------------------------------
template <>
void Bitmap_<unsigned char>::load(const std::filesystem::path& filename)
{
    std::optional<std::filesystem::path> filepath = pgFindDataPath(filename);
    ASSERT(filepath, "The input file for bitmap '" + filename.string() + "' is not found.");

    auto ext = pgGetExtension(filepath.value());

    if (ext == ".png" || ext == ".PNG")
        pgLog("Loading PNG file '" + filepath.value().string() + "' ...");
    else if (ext == ".jpg" || ext == ".JPG")
        pgLog("Loading JPG file '" + filepath.value().string() + "' ...");
    else if (ext == ".bmp" || ext == ".BMP")
        pgLog("Loading BMP file '" + filepath.value().string() + "' ...");
    else if (ext == ".exr" || ext == ".EXR" || ext == ".hdr" || ext == ".HDR")
    {
        pgLogFatal("EXR format can be loaded only in Bitmap_<float>.");
        return;
    }
    uint8_t* raw_data;
    raw_data = stbi_load(filepath.value().string().c_str(), &m_width, &m_height, &m_channels, static_cast<int>(m_format));
    if (m_format == PixelFormat::NONE) 
        m_format = static_cast<PixelFormat>(m_channels);
    m_channels = static_cast<int>(m_format);
    m_data.reset(raw_data);

    if (pgWindowInitialized())
        prepareGL();
}

// --------------------------------------------------------------------
template <>
void Bitmap_<float>::load(const std::filesystem::path& filename)
{
    std::optional<std::filesystem::path> filepath = pgFindDataPath(filename);
    ASSERT(filepath, "The input file for bitmap '" + filename.string() + "' is not found.");

    auto ext = pgGetExtension(filepath.value());

    // EXR/HDR 形式の場合はそのまま読み込む
    if (ext == ".exr" || ext == ".EXR" || ext == ".hdr" || ext == ".HDR")
    {
        std::string kind = ext == ".exr" || ext == ".EXR" ? "EXR" : "HDR";
        pgLog("Loading " + kind + " file '" + filepath.value().string() + "' ...");
        m_format = m_format == PixelFormat::NONE ? PixelFormat::RGBA : m_format;

        const char* err = nullptr;
        float* raw_data;

        if (kind == "EXR")
        {
            int result = LoadEXR(&raw_data, &m_width, &m_height, filepath.value().string().c_str(), &err);
            m_data.reset(raw_data);
            m_channels = static_cast<int>(m_format);
            if (result != TINYEXR_SUCCESS)
            {
                if (err)
                {
                    pgLogFatal(err);
                    FreeEXRErrorMessage(err);
                    return;
                }
            }
        }
        else if (kind == "HDR")
        {
            raw_data = stbi_loadf(filepath.value().string().c_str(), &m_width, &m_height, &m_channels, static_cast<int>(m_format));
            m_channels = static_cast<int>(m_format);
            m_data.reset(raw_data);
        }
    }
    // PNG/JPG/BMP 形式の場合は uint8_t* [0, 255] -> float* [0, 1] に変換 
    else
    {
        if (ext == ".png" || ext == ".PNG")
            pgLog("Loading PNG file '" + filepath.value().string() + "' ...");
        else if (ext == ".jpg" || ext == ".JPG")
            pgLog("Loading JPG file '" + filepath.value().string() + "' ...");
        else if (ext == ".bmp" || ext == ".BMP")
            pgLog("Loading BMP file '" + filepath.value().string() + "' ...");

        uint8_t* raw_data = stbi_load(filepath.value().string().c_str(), &m_width, &m_height, &m_channels, static_cast<int>(m_format));
        if (m_format == PixelFormat::NONE)
            m_format = static_cast<PixelFormat>(m_channels);
        m_channels = static_cast<int>(m_format);
        m_data = std::make_unique<float[]>(m_width * m_height * m_channels);
        for (int i = 0; i < m_width * m_height * m_channels; i += m_channels)
        {
            for (int c = 0; c < m_channels; c++)
            {
                m_data[i + c] = static_cast<float>(raw_data[i + c]) / 255.0f;
            }
        }
        stbi_image_free(raw_data);
    }

    if (pgWindowInitialized())
        prepareGL();
}

// --------------------------------------------------------------------
template <typename PixelType>
void Bitmap_<PixelType>::write(const std::filesystem::path& filepath, int quality) const 
{
    UNIMPLEMENTED();
}

template <>
void Bitmap_<unsigned char>::write(const std::filesystem::path& filepath, int quality) const 
{
    std::string ext = pgGetExtension(filepath);

    bool supported = ext == ".png" || ext == ".PNG" || ext == ".jpg" || ext == ".JPG" || ext == ".bmp" || ext == ".BMP";
    if (!supported)
    {
        pgLogFatal("This extension '" + ext + "' is not suppoted with Bitmap_<unsigned char>");
        return;
    }

    uint8_t* data = new uint8_t[m_width * m_height * m_channels];
    memcpy(data, m_data.get(), m_width * m_height * m_channels);

    if (ext == ".png" || ext == ".PNG")
    {
        stbi_write_png(filepath.string().c_str(), m_width, m_height, m_channels, data, m_width * m_channels);
        delete[] data;
    }
    else if (ext == ".jpg" || ext == ".JPG")
    {
        stbi_write_jpg(filepath.string().c_str(), m_width, m_height, m_channels, data, quality);
        delete[] data;
    }
    else if (ext == ".bmp" || ext == ".BMP")
    {
        stbi_write_bmp(filepath.string().c_str(), m_width, m_height, m_channels, data);
        delete[] data;
    }

    pgLog("Wrote bitmap to '" + filepath.string() + "'");
}

template <> 
void Bitmap_<float>::write(const std::filesystem::path& filepath, int quality) const 
{
    std::string ext = pgGetExtension(filepath);
    
    bool supported = ext == ".png" || ext == ".PNG" || ext == ".jpg" || ext == ".JPG" || ext == ".bmp" || ext == ".BMP" || 
                     ext == ".exr" || ext == ".EXR" || ext == ".hdr" || ext == ".HDR";

    if (!supported)
    {
        pgLogFatal("This extension '" + ext + "' is not suppoted with Bitmap_<float>");
        return;
    }

    // Copy bitmap data to temporal pointer
    float* data = new float[m_width * m_height * m_channels];
    memcpy(data, m_data.get(), m_width * m_height * m_channels * sizeof(float));

    // PNG/JPG/BMP の場合は float [0, 1] -> uint8_t [0, 255] に変換する
    if (ext == ".png" || ext == ".PNG" || ext == ".jpg" || ext == ".JPG" || ext == ".bmp" || ext == ".BMP") 
    {
        uint8_t* uc_data = new uint8_t[m_width * m_height * m_channels];
        for (int i=0; i<m_width; i++)
        {
            for (int j=0; j<m_height; j++)
            {
                int idx = (i * m_height + j) * m_channels;

                // Convert float data to uint8_t to write image using stb_image
                float4 tmp{0.0f, 0.0f, 0.0f, 1.0f};
                memcpy(&tmp, &data[idx], sizeof(float)*m_channels);
                uchar4 ucolor = make_color(tmp, false);
                memcpy(&uc_data[idx], &ucolor, m_channels);
            }
        }

        if (ext == ".png" || ext == ".PNG")
            stbi_write_png(filepath.string().c_str(), m_width, m_height, m_channels, uc_data, m_width * m_channels);
        else if (ext == ".jpg" || ext == ".JPG")
            stbi_write_jpg(filepath.string().c_str(), m_width, m_height, m_channels, uc_data, quality);
        else if (ext == ".bmp" || ext == ".BMP")
            stbi_write_bmp(filepath.string().c_str(), m_width, m_height, m_channels, uc_data);

        delete[] uc_data;
        delete[] data;
    }
    else // EXR or HDR
    {
        if (ext == ".exr" || ext == ".EXR")
        {
            const char* err;
            int ret = SaveEXR(data, m_width, m_height, m_channels, /* save_as_fp16 = */ 0, filepath.string().c_str(), &err);
            if (ret != TINYEXR_SUCCESS)
            {
                pgLogFatal("Failed to write EXR:", err);
                delete[] data;
                return;
            }
            delete[] data;
        }
        else // HDR 
        {
            stbi_write_hdr(filepath.string().c_str(), m_width, m_height, m_channels, data);
            delete[] data;
        }
    }
    pgLog("Wrote bitmap to '" + filepath.string() + "'");
}

// --------------------------------------------------------------------
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
    // Prepare vertices data
    int window_width = pgGetWidth(), window_height = pgGetHeight();
    GLfloat x0 = -1.0f + (static_cast<float>(x) / window_width) * 2.0f;
    GLfloat x1 = -1.0f + (static_cast<float>(x + width) / window_width) * 2.0f;
    GLfloat y0 =  1.0f - (static_cast<float>(y + height) / window_height) * 2.0f;
    GLfloat y1 =  1.0f - (static_cast<float>(y) / window_height) * 2.0f;
    GLfloat vertices[] = {
        // position     // texcoord (vertically flipped)
        x0, y0, 0.0f,   0.0f, 1.0f,
        x0, y1, 0.0f,   0.0f, 0.0f, 
        x1, y0, 0.0f,   1.0f, 1.0f, 
        x1, y1, 0.0f,   1.0f, 0.0f
    };
    GLuint indices[] = {
        0, 1, 2,
        2, 1, 3
    };

    glBindVertexArray(m_vao);

    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ebo);
    glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, sizeof(indices), indices);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (void*)0);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (void*)(3 * sizeof(GLfloat)));
    glEnableVertexAttribArray(1);

    // Initialize Texture2D 
    bool is_gray = m_format == PixelFormat::GRAY;
    GLenum texture_data_type = GL_UNSIGNED_BYTE;
    GLint internal_format = GL_RGB8;
    GLenum format = GL_RGB;
    if constexpr (std::is_same_v<PixelType, float>) {
        texture_data_type = GL_FLOAT;
    }

    switch(m_format)
    {
    case PixelFormat::GRAY:
        format = GL_RED;
        if constexpr (std::is_same_v<PixelType, float>)
            internal_format = GL_R32F;
        else 
            internal_format = GL_R8;
        break;
    case PixelFormat::GRAY_ALPHA:
        format = GL_RG;
        if constexpr (std::is_same_v<PixelType, float>)
            internal_format = GL_RG32F;
        else 
            internal_format = GL_RG8;
        break;
    case PixelFormat::RGB:
        format = GL_RGB;
        if constexpr (std::is_same_v<PixelType, float>)
            internal_format = GL_RGB32F;
        else 
            internal_format = GL_RGB8;
        break;
    case PixelFormat::RGBA:
        format = GL_RGBA;
        if constexpr (std::is_same_v<PixelType, float>)
            internal_format = GL_RGBA32F;
        else 
            internal_format = GL_RGBA8;
        break;
    case PixelFormat::NONE:
    default:
        return;
    }
    
    glBindTexture(GL_TEXTURE_2D, m_gltex);
    PixelType* raw_data = m_data.get();
    glTexImage2D(GL_TEXTURE_2D, 0, internal_format, m_width, m_height, 0, format, texture_data_type, raw_data);

    m_shader.begin();
    m_shader.setUniform1i("tex", 0);
    m_shader.setUniform1i("is_gray", is_gray);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_gltex);
    glBindVertexArray(m_vao);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

// --------------------------------------------------------------------
template <typename PixelType>
void Bitmap_<PixelType>::allocateDevicePtr()
{
    if (d_data)
        CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_data),
        m_width * m_height * m_channels * sizeof(PixelType)
    ));
}

// --------------------------------------------------------------------
template <typename PixelType>
void Bitmap_<PixelType>::copyToDevice() 
{
    // CPU側のデータが準備されているかチェック
    ASSERT(m_data.get(), "Image data in the host side has been not allocated yet.");

    // GPU上に画像データを準備
    CUDABuffer<PixelType> d_buffer;
    d_buffer.copyToDevice(m_data.get(), m_width * m_height * m_channels*sizeof(PixelType));
    d_data = d_buffer.deviceData();
}

// --------------------------------------------------------------------
template <typename PixelType>
void Bitmap_<PixelType>::copyFromDevice()
{
    ASSERT(d_data, "No data has been allocated on the device yet.");

    PixelType* raw_data = m_data.get();
    CUDA_CHECK(cudaMemcpy(
        raw_data,
        d_data, m_width * m_height * m_channels * sizeof(PixelType),
        cudaMemcpyDeviceToHost
    ));
}

// --------------------------------------------------------------------
template <typename PixelType>
void Bitmap_<PixelType>::prepareGL()
{
    // Prepare vertex array object
    glGenVertexArrays(1, &m_vao);
    glGenBuffers(1, &m_vbo);
    glGenBuffers(1, &m_ebo);

    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*20, nullptr, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint)*6, nullptr, GL_DYNAMIC_DRAW);

    // Preparing texture
    m_gltex = 0;
    glGenTextures(1, &m_gltex);
    glBindTexture(GL_TEXTURE_2D, m_gltex);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    m_shader.load("prayground/core/shaders/bitmap.vert", "prayground/core/shaders/bitmap.frag");
}

template class Bitmap_<float>;
template class Bitmap_<double>;
template class Bitmap_<unsigned char>;
template class Bitmap_<char>;
template class Bitmap_<int>;

}