#pragma once 

#include "util.h"
#include "bitmap.h"
#include <unordered_map>

namespace oprt {

template <typename PixelType>
class Film_ {
public:

    Film();
    ~Film();

    void addBitmap(const std::string& name, const std::shared_ptr<Bitmap>& bitmap);
    Bitmap getBitmap(const std::string& name) const;
    std::vector<std::shared_ptr<Bitmap>> bitmaps() const;

    void addBitmapFloat(const std::string& name, const BitmapFloat bitmap);
    Bitmap getBitmapFloat(const std::string& name);
    std::vector<std::shared_ptr<BitmapFloat>> bitmapFloats() const;
private:
    std::unordered_map<std::string, std::shared_ptr<Bitmap>> m_bitmaps;
    std::unordered_map<std::string, std::shared_ptr<BitmapFloat>> m_bitmap_floats;
    float m_gamma;
    float m_exposure;
};

}