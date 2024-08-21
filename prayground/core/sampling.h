#pragma once 

#include <prayground/core/bitmap.h>

namespace prayground {

    struct Distribution1D {
        Distribution1D() = default;
    #ifndef __CUDACC__
        Distribution1D(float* data, uint32_t size) {
            func = new float[size];
            memcpy(func, data, size * sizeof(float));

            cdf = new float[size + 1];
            for (int i = 1; i < size + 1; ++i)
                cdf[i] = cdf[i - 1] + func[i - 1] / size;

            func_int = cdf[size];
            if (func_int == 0.0f) {
                for (int i = 1; i < size + 1; ++i)
                    cdf[i] = float(i) / float(size);
            } else {
                for (int i = 1; i < size + 1; ++i)
                    cdf[i] /= func_int;
            }

        }
    #endif

        uint32_t offset(float u) const {
            int first = 0;
            int len = cdf_size;

            while (len > 0) {
                int half = len >> 1;
                int middle = first + half;
                if (cdf[middle] < u) {
                    first = middle + 1;
                    len -= half + 1;
                }
                else {
                    len = half;
                }
            }

            return first;
        }

        float sample(float u, float& out_pdf, uint32_t& out_offset) const {
            out_offset = this->offset(u);
            float du = u - cdf[out_offset];
            if ((cdf[out_offset + 1] - cdf[out_offset]) > 0.0f) {
                du /= (cdf[out_offset + 1] - cdf[out_offset]);
            }

            out_pdf = (func_int > 0) ? func[out_offset] / func_int : 0.0f;

            return (out_offset + du) / size;
        }

        float pdfAt(uint32_t idx) const {
            return func[idx] / func_int * size;
        }

        float* cdf;
        uint32_t cdf_size;
        float* func;
        float func_int;
        uint32_t size;
    };

    class Distribution2D {
    public:
#ifndef __CUDACC__
        /* This constructor consider input data as single-channel bitmap */
        Distribution2D(float* data, uint32_t width, uint32_t height);
        /* Only 32-bit float or 8-bit can be used to initialize Distribution2D */
        Distribution2D(const FloatBitmap& bitmap);
        Distribution2D(const Bitmap& bitmap);
    private:
        void init(float* data, uint32_t width, uint32_t height);
#endif
    public:
        float pdfAt(const Vec2f& p) const {
            uint32_t iu = clamp((uint32_t)(p.x() * m_conditional[0].size), 0, m_conditional[0].size - 1);
            uint32_t iv = clamp((uint32_t)(p.y() * m_marginal.size), 0, m_marginal.size - 1);
            return m_conditional[iv].func[iu] / m_marginal.func_int;
        }
    private:
        Distribution1D* m_conditional;
        Distribution1D m_marginal;
    };

} // namespace prayground