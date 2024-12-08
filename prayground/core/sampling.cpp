#include "sampling.h"
#include <prayground/core/spectrum.h>

namespace prayground {

    // Distribution1D
    Distribution1D::Distribution1D(float* data, uint32_t size) {
        func = new float[size];
        memcpy(func, data, size * sizeof(float));

        cdf = new float[size + 1]; cdf[0] = 0.0f;
        for (int i = 1; i < size + 1; ++i)
            cdf[i] = cdf[i - 1] + func[i - 1] / size;

        func_int = cdf[size];
        if (func_int == 0.0f) {
            for (int i = 1; i < size + 1; ++i)
                cdf[i] = float(i) / float(size);
        }
        else {
            for (int i = 1; i < size + 1; ++i)
                cdf[i] /= func_int;
        }
    }

    // Distribution2D
    Distribution2D::Distribution2D(float* data, uint32_t width, uint32_t height) {
        init(data, width, height);
    }

    Distribution2D::Distribution2D(const FloatBitmap& bitmap) {
        assert(bitmap.channels() == 3 || bitmap.channels() == 4);
        float* data = new float[bitmap.width() * bitmap.height()];
        for (int i = 0; i < bitmap.height(); i++) {
            for (int j = 0; j < bitmap.width(); j++) {
                auto pixel = bitmap.at(i, j);
                if (bitmap.channels() == 3) {
                    data[i * bitmap.width() + j] = luminance(std::get<float3>(pixel));
                } else {
                    data[i * bitmap.width() + j] = luminance(std::get<float4>(pixel));
                }
            }
        }
        init(data, bitmap.width(), bitmap.height());
    }

    Distribution2D::Distribution2D(const Bitmap& bitmap) {
        assert (bitmap.channels() == 3 || bitmap.channels() == 4);
        float* data = new float[bitmap.width() * bitmap.height()];
        for (int i = 0; i < bitmap.height(); i++) {
            for (int j = 0; j < bitmap.width(); j++) {
                if (bitmap.channels() == 3) {
                    data[i * bitmap.width() + j] = luminance(color2float(Vec3u(std::get<uchar3>(bitmap.at(i, j)))));
                } else {
                    data[i * bitmap.width() + j] = luminance(color2float(Vec4u(std::get<uchar4>(bitmap.at(i, j)))));
                }
            }
        }
        init(data, bitmap.width(), bitmap.height());
    }

    void Distribution2D::init(float* data, uint32_t width, uint32_t height) {
        m_conditional = new Distribution1D[height];
        for (int i = 0; i < height; i++) {
            m_conditional[i] = Distribution1D(&data[i * width], width);
        }

        std::vector<float> marginal_func;
        marginal_func.reserve(height);
        for (int i = 0; i < height; i++) {
            marginal_func.push_back(m_conditional[i].func_int);
        }
        m_marginal = Distribution1D(marginal_func.data(), height);
    }
} // namespace prayground

