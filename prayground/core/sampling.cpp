#include "sampling.h"
#include <prayground/core/spectrum.h>

namespace prayground {

    Distribution2D::Distribution2D(const float* data, int width, int height) {
        init(data, width, height);
    }

    Distribution2D::Distribution2D(const FloatBitmap& bitmap) {
        assert(bitmap.channels() == 3 || bitmap.channels() == 4);
        float* data = new float[bitmap.width() * bitmap.height()];
        for (int i = 0; i < bitmap.height(); i++) {
            for (int j = 0; j < bitmap.width(); j++) {
                data[i * bitmap.width() + j] = luminance(std::get<Vec3f>(bitmap.at(i, j)));
            }
        }
        init(data, bitmap.width(), bitmap.height());
    }

    Distribution2D::Distribution2D(const Bitmap& bitmap) {
        assert (bitmap.channels() == 3 || bitmap.channels() == 4);
        float* data = new float[bitmap.width() * bitmap.height()];
        for (int i = 0; i < bitmap.height(); i++) {
            for (int j = 0; j < bitmap.width(); j++) {
                data[i * bitmap.width() + j] = luminance(color2float(std::get<Vec3u>(bitmap.at(i, j))));
            }
        }
        init(data, bitmap.width(), bitmap.height());
    }

    void Distribution2D::init(const float* data, int width, int height) {
        m_conditional = new Distribution1D*[height];
        for (int i = 0; i < height; i++) {
            m_conditional[i] = new Distribution1D(&data[i * width], width);
        }

        std::vector<float> marginal_func;
        marginal_func.reserve(height);
        for (int i = 0; i < height; i++) {
            marginal_func.push_back(m_conditional[i]->funcInt());
        }
        m_marginal = new Distribution1D(marginal_func.data(), height);
    }
} // namespace prayground

