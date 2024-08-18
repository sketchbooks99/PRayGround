#pragma once 

#include <prayground/core/bitmap.h>

namespace prayground {

    class Distribution1D {
    public:
        Distribution1D(float* data, uint32_t size);

        uint32_t offset(float u) const {
            int first = 0;
            int len = m_cdf_size;

            while (len > 0) {
                int half = len >> 1;
                int middle = first + half;
                if (m_cdf[middle] < u) {
                    first = middle + 1;
                    len -= half + 1;
                } else {
                    len = half;
                }
            }

            return first;
        }

        float sample(float u, float& pdf, uint32_t& offset) const {
            offset = this->offset(u);
            float du = u - m_cdf[offset];
            if ((m_cdf[offset + 1] - m_cdf[offset]) > 0.0f) {
                du /= (m_cdf[offset + 1] - m_cdf[offset]);
            }

            pdf = (m_func_int > 0) ? m_func[offset] / m_func_int : 0.0f;

            return (offset + du) / m_size;
        }

        uint32_t size() const {
            return m_size;
        }

        float pdfAt(uint32_t idx) const {
            return m_func[idx] / m_func_int * m_size;
        }
    private:
        float* m_cdf;
        uint32_t m_cdf_size;
        float* m_func;
        float m_func_int;
        uint32_t m_size;
    }

    class Distribution2D {
    public:
    #ifndef __CUDACC__
        /* This constructor consider input data as single-channel bitmap */
        Distribution2D(const float* data, int width, int height);
        /* Only 32-bit float or 8-bit can be used to initialize Distribution2D */
        Distribution2D(const FloatBitmap& bitmap);
        Distribution2D(const Bitmap& bitmap);
    #endif

        float pdfAt(const Vec2f& p) const {
            uint32_t iu = clamp((uint32_t)(p.x() * m_conditional.size()), 0, m_conditional.size() - 1);
            uint32_t iv = clamp((uint32_t)(p.y() * m_marginal.size()), 0, m_marginal.size() - 1);
        }
    private:
        void init(const float* data, int width, int height);
        Distribution1D* m_conditional;
        Distribution1D m_marginal;
    };

} // namespace prayground