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

enum BitmapFormat {
    UNSIGNED_BYTE4,
    UNSIGNED_BYTE3,
    FLOAT4, 
    FLOAT3
};

template <class T>
class Bitmap {
public:
    Bitmap() { _detectFormat(); }
    Bitmap(T* data, int width, int height) : m_data(data), m_width(width), m_height(height) 
    {
        _detectFormat();
    }
    explicit Bitmap(const std::string& filename) 
    {
        _detectFormat();
    }

    void allocate(int width, int height);
    void load(const std::string& filename);
    void write(const std::string& filename, bool gamma_enabled=true, int quality=100) const;

    void copyToDevice();
    void copyFromDevice();

    T* data() const { return m_data; }
    T* devicePtr() const { return d_data; }
    ImageFormat format() const { return m_format; }
protected:
    void _detectFormat() {
        if constexpr (std::is_same_v<T, char4>) {
            m_format = UNSIGNED_BYTE4;
            m_channels = 4;
        }
        if constexpr (std::is_same_v<T, char3>) {
            m_format = UNSIGNED_BYTE3;
            m_channels = 3;
        }
        if constexpr (std::is_same_v<T, float4>) {
            m_format = FLOAT4;
            m_channels = 4;
        }
        if constexpr (std::is_same_v<T, float3>) {
            m_format = FLOAT3;
            m_channels = 3;
        }
    }

    BitmapFormat m_format;
    int m_width, m_height;
    int m_channels;

    T* m_data;  // CPU側のデータ
    T* d_data;  // GPU側のデータ
};

using BitmapPNG = Bitmap<uchar4>;
using BitmapJPG = Bitmap<uchar3>;
using BitmapUchar4 = Bitmap<uchar4>;
using BitmapUchar3 = Bitmap<uchar3>;
using BitmapFloat4 = Bitmap<float4>;
using BitmapFloat3 = Bitmap<float3>;

}
