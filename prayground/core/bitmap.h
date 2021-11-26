#pragma once

#include <filesystem>
#include <prayground/gl/shader.h>
#include <map>

namespace prayground {

/// @todo Casting bitmap to OptixImage2D

enum class PixelFormat : int 
{
    NONE        = 0, 
    GRAY        = 1, 
    GRAY_ALPHA  = 2, 
    RGB         = 3, 
    RGBA        = 4
};

template <typename PixelType>
class Bitmap_ {
public:
    using Type = PixelType;

    Bitmap_();
    Bitmap_(PixelFormat format, int width, int height, PixelType* data = nullptr);
    explicit Bitmap_(const std::filesystem::path& filename);
    explicit Bitmap_(const std::filesystem::path& filename, PixelFormat format);

    void allocate(PixelFormat format, int width, int height);
    void setData(PixelType* data, int offset_x, int offset_y, int width, int height);
    void setData(PixelType* data, const int2& offset, const int2& res);

    void load(const std::filesystem::path& filename);
    void load(const std::filesystem::path& filename, PixelFormat format);
    void write(const std::filesystem::path& filename, int quality=100) const;

    void draw() const;
    void draw(int32_t x, int32_t y) const;
    void draw(int32_t x, int32_t y, int32_t width, int32_t height) const;
    
    void allocateDevicePtr();
    void copyToDevice();
    void copyFromDevice();

    PixelType* data() const { return m_data.get(); }
    PixelType* devicePtr() const { return d_data; }

    int width() const { return m_width; }
    int height() const { return m_height; }
    int channels() const { return m_channels; }
private:
    std::unique_ptr<PixelType[]> m_data;  // CPU側のデータ
    PixelType* d_data { nullptr };        // GPU側のデータ

    PixelFormat m_format { PixelFormat::NONE };
    int m_width { 0 };
    int m_height { 0 };
    int m_channels { 0 };

    // Member variables to draw Bitmap on OpenGL context
    GLint m_gltex; 
    gl::Shader m_shader;
};

using Bitmap = Bitmap_<unsigned char>;
using FloatBitmap = Bitmap_<float>;

}
