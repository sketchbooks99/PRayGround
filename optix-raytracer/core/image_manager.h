#pragma once

#include "util.h"

namespace oprt {

/**
 * @note
 * Currently, ImageManager supports following image formats (char4, char3, float4, float3).
 * 
 * @todo
 * Implement manager for HDR format (.hdr, .exr)
 */

enum ImageFormat {
    UNSIGNED_BYTE4,
    UNSIGNED_BYTE3,
    FLOAT4, 
    FLOAT3
};

template <class T>
class ImageManager {
public:
    ImageManager(){}

    virtual void load(const std::string& filename) = 0;
    virtual void write(const std::string& filename, bool gamma_enabled) const = 0;

    void copy_to_device();

    T* data() const { return m_data; }
    T* d_ptr() const { return d_data; }
    ImageFormat format() const { return m_format; }

protected:

    void _detect_format() {
        if constexpr (std::is_same_v<T, char4>) 
            m_format = UNSIGNED_BYTE4;
        if constexpr (std::is_same_v<T, char3>) 
            m_format = UNSIGNED_BYTE3;
        if constexpr (std::is_same_v<T, float4>)
            m_format = FLOAT4;
        if constexpr (std::is_same_v<T, float3>)
            m_format = FLOAT3;
    }

    ImageFormat m_format;
    int m_width, m_height;
    int m_channels;

    T* m_data;  // CPU側のデータ
    T* d_data;  // GPU側のデータ
};

template <class T>
class PNGManager final : public ImageManager<T> {
public:
    /// Enable call of a function in the base class \c ImageManager<T> .
    using ImageManager<T>::copy_to_device;

    PNGManager() 
    {
        this->_detect_format();
    }
    explicit PNGManager(const std::string& filename)
    {
        this->_detect_format();
        load(filename);
    }

    void load(const std::string& filename) override;
    void write(const std::string& filename, bool gamma_enabled=true) const override;
};

/**
 * @todo
 * Implement this right now.
 */
// template <class T>
// class JPGManager final : public ImageManager<T> {

// };

}
