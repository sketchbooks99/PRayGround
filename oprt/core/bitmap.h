#pragma once

#include "util.h"

namespace oprt {

/**
 * @note
 * Currently, Bitmap supports following image formats (char4, char3, float4, float3).
 * 
 * @todo
 * Implement manager for HDR format (.hdr, .exr)
 */

template <typename PixelType>
class Bitmap_ {
public:
    using Type = PixelType;

    enum class Format 
    {
        GRAY,       // 1 channels
        GRAY_ALPHA, // 2 channels
        RGB,        // 3 channels
        RGBA        // 4 channels
    };

    Bitmap_();
    Bitmap_(PixelType* data, int width, int height, Format format);
    explicit Bitmap_(const std::filesystem::path& filepath);

    void allocate(int width, int height, Format format);
    void fillData(PixelType* data, int width, int height, int offset_x=0, int offset_y=0);
    void fillData(PixelType* data, int2 res, int2 offset);

    void load(const std::filesystem::path& filepath);
    void write(const std::filesystem::path& filepath, int quality=100) const;

    void copyToDevice();
    void copyFromDevice();

    PixelType* data() const { return m_data; }
    PixelType* devicePtr() const { return d_data; }

    int width() const { return m_width; }
    int height() const { return m_height; }
    int channels() const { return m_channels; }
private:
    PixelType* m_data { nullptr };  // CPU側のデータ
    PixelType* d_data { nullptr };  // GPU側のデータ

    Format m_format { Format::GRAY };
    int m_width { 0 };
    int m_height { 0 };
    int m_channels { 0 };
};

using Bitmap = Bitmap_<unsigned char>;
using BitmapFloat = Bitmap_<float>;
using BitmapDouble = Bitmap_<double>;

}
