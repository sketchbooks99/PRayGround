#pragma once 

#include "util.h"
#include "bitmap.h"

namespace oprt {

template <typename PixelType>
class Film_ {
public:

    Film();
    ~Film();

    /** @brief Apply postprocess to the bitmap. */
    void postProcess();

    void setGamma(const float gamma);
    float gamma() const;

    void setExposure(const float exposure);
    float exposure() const;

    std::shared_ptr<Bitmap_<PixelType>> bitmapPtr();
    Bitmap_<PixelType>> bitmap() const;

private:
    // std::shared_ptr<Bitmap_<PixelType>> m_bitmap;
    std::shared_ptr<Bitmap_<PixelType>> m_bitmaps;
    float m_gamma;
    float m_exposure;
};

}