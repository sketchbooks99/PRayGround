#pragma once

#include <filesystem>
#include <prayground/gl/shader.h>
#include <map>

namespace prayground {

/// @todo Casting bitmap to OptixImage2D

template <typename PixelType>
class Bitmap_ {
public:
    using Type = PixelType;

    enum class Format : int
    {
        AUTO        = 0,
        GRAY        = 1,       // 1 channels
        GRAY_ALPHA  = 2,       // 2 channels
        RGB         = 3,       // 3 channels
        RGBA        = 4        // 4 channels
    };

    Bitmap_();
    Bitmap_(Format format, int width, int height, PixelType* data = nullptr);
    explicit Bitmap_(const std::filesystem::path& filename);
    explicit Bitmap_(const std::filesystem::path& filename, Format format);

    void allocate(Format format, int width, int height);
    void setData(PixelType* data, int offset_x, int offset_y, int width, int height);
    void setData(PixelType* data, const int2& offset, const int2& res);

    void load(const std::filesystem::path& filename);
    void load(const std::filesystem::path& filename, Format format);
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

    Format m_format { Format::AUTO };
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
