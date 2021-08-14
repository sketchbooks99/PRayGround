#pragma once

#include "util.h"
#include "../gl/shader.h"
//#include <glad/glad.h>

namespace oprt {

template <typename PixelType>
class Bitmap_ {
public:
    using Type = PixelType;

    enum class Format 
    {
        GRAY,       // 1 channels
        GRAY_ALPHA, // 2 channels
        RGB,        // 3 channels
        RGBA,       // 4 channels
        UNKNOWN
    };

    Bitmap_();
    Bitmap_(Format format, int width, int height, PixelType* data = nullptr);
    explicit Bitmap_(const std::filesystem::path& filename);
    explicit Bitmap_(const std::filesystem::path& filename, Format format);

    void allocate(Format format, int width, int height);
    void setData(PixelType* data, int width, int height, int offset_x, int offset_y);
    void setData(PixelType* data, const int2& res, const int2& offset);

    void load(const std::filesystem::path& filename);
    void load(const std::filesystem::path& filename, Format format);
    void write(const std::filesystem::path& filename, int quality=100) const;

    void draw() const;
    void draw(int32_t x, int32_t y) const;
    void draw(int32_t x, int32_t y, int32_t width, int32_t height) const;

    void copyToDevice();
    void copyFromDevice();

    PixelType* data() const { return m_data; }
    PixelType* devicePtr() const { return d_data; }

    int width() const { return m_width; }
    int height() const { return m_height; }
    int channels() const { return m_channels; }
private:
    std::map<Format, int> type2channels = 
    {
        { Format::GRAY, 1 }, 
        { Format::GRAY_ALPHA, 2}, 
        { Format::RGB, 3 }, 
        { Format::RGBA, 4 }, 
        { Format::UNKNOWN, 0 }
    };

    PixelType* m_data { nullptr };  // CPU側のデータ
    PixelType* d_data { nullptr };  // GPU側のデータ

    Format m_format { Format::UNKNOWN };
    int m_width { 0 };
    int m_height { 0 };
    int m_channels { 0 };

    // Member variables to draw Bitmap on OpenGL context
    GLint m_gltex; 
    // GLuint m_gl_vertex_array;
    gl::Shader m_shader;
};

using Bitmap = Bitmap_<unsigned char>;
using FloatBitmap = Bitmap_<float>;

}
