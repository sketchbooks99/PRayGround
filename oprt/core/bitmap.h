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

template <typename T, unsigned int N>
class Bitmap {
public:
    Bitmap() { m_channels = N; }
    Bitmap(T* data, int width, int height) : m_data(data), m_width(width), m_height(height) 
    {
    }
    explicit Bitmap(const std::string& filename) 
    {
    }

    void allocate(int width, int height);
    void load(const std::string& filename);
    void write(const std::string& filename, bool gamma_enabled=true, int quality=100) const;

    void copyToDevice();
    void copyFromDevice();

    T* data() const { return m_data; }
    T* devicePtr() const { return d_data; 

    int width() const { return m_width; }
    int height() const { return m_height; }
    int channels() const { return m_channels; }
private:
    T* m_data;  // CPU側のデータ
    T* d_data;  // GPU側のデータ

    int m_width, m_height;
    int m_channels;
};

using BitmapPNG = Bitmap<unsigned char, 4>;
using BitmapJPG = Bitmap<unsigned char, 3>;
using BitmapUchar4 = Bitmap<unsigned char, 4>;
using BitmapUchar3 = Bitmap<unsigned char, 4>;
using BitmapFloat4 = Bitmap<float, 4>;
using BitmapFloat3 = Bitmap<float, 3>;

}
