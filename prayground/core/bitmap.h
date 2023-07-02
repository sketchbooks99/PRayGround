#pragma once

#include <filesystem>
#include <map>
#include <prayground/gl/shader.h>
#include <prayground/app/window.h>

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

    template <typename PixelT>
    class Bitmap_ {
    public:
        using Type = PixelT;

        Bitmap_();
        Bitmap_(PixelFormat format, int width, int height, PixelT* data = nullptr);
        explicit Bitmap_(const std::filesystem::path& filename);
        explicit Bitmap_(const std::filesystem::path& filename, PixelFormat format);
        /// @todo: Check if "Disallow the copy-constructor"
        // Bitmap_(const Bitmap_& bmp) = delete;

        void allocate(PixelFormat format, int width, int height);
        void setData(PixelT* data, int offset_x, int offset_y, int width, int height);
        void setData(PixelT* data, const int2& offset, const int2& res);

        void load(const std::filesystem::path& filename);
        void load(const std::filesystem::path& filename, PixelFormat format);
        void write(const std::filesystem::path& filename, int quality=100) const;

        void draw() const;
        void draw(int32_t x, int32_t y) const;
        void draw(int32_t x, int32_t y, int32_t width, int32_t height) const;
    
        void allocateDevicePtr();
        void copyToDevice();
        void copyFromDevice();

        PixelT* data() const { return m_data.get(); }
        PixelT* devicePtr() const { return d_data; }

        OptixImage2D toOptixImage2D() const;

        int width() const { return m_width; }
        int height() const { return m_height; }
        int channels() const { return m_channels; }
    private:
        void prepareGL();

        std::unique_ptr<PixelT[]> m_data;  // Data on CPU
        PixelT* d_data { nullptr };        //      on GPU

        PixelFormat m_format { PixelFormat::NONE };
        int m_width { 0 };
        int m_height { 0 };
        int m_channels { 0 };

        // Member variables to draw Bitmap on OpenGL context
        GLuint m_gltex;
        GLuint m_vbo, m_vao, m_ebo; // vertex buffer object, vertex array object, element buffer object
        gl::Shader m_shader;
    };

    using Bitmap = Bitmap_<unsigned char>;
    using FloatBitmap = Bitmap_<float>;

} // namespace prayground
