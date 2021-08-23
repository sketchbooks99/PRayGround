#pragma once 

#include <oprt/core/bitmap.h>
#include <sutil/vec_math.h>
#include <unordered_map>
#include <memory>

namespace oprt {

class Film {
public:
    Film();
    Film(int32_t width, int32_t height);
    ~Film();

    void addBitmap(const std::string& name, Bitmap::Format format);
    std::shared_ptr<Bitmap> bitmapAt(const std::string& name) const;
    std::vector<std::shared_ptr<Bitmap>> bitmaps() const;
    size_t numBitmaps() const;

    void addFloatBitmap(const std::string& name, FloatBitmap::Format format);
    std::shared_ptr<FloatBitmap> floatBitmapAt(const std::string& name);
    std::vector<std::shared_ptr<FloatBitmap>> floatBitmaps() const;
    size_t numFloatBitmaps() const;

    void setResolution(int32_t width, int32_t height);
    void setResolution(int2 resolution);
    void setWidth(int32_t width);
    void setHeight(int32_t height);
    int32_t width() const;
    int32_t height() const;
private:
    std::unordered_map<std::string, std::shared_ptr<Bitmap>> m_bitmaps;
    std::unordered_map<std::string, std::shared_ptr<FloatBitmap>> m_float_bitmaps;
    int32_t m_width { 0 };
    int32_t m_height { 0 };
};

}