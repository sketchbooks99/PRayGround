#pragma once

#include <optix.h>
#include <vector_types.h>
#include <prayground/math/vec.h>
#include <prayground/optix/macros.h>
#include <prayground/optix/util.h>
#include <prayground/core/util.h>

#ifndef __CUDACC__
#include <filesystem>
#include <fstream>
#endif

namespace prayground {

    constexpr int min_lambda = 380;
    constexpr int max_lambda = 720;
    constexpr int nSpectrumSamples = 81;
    constexpr float spectrum_lambda[nSpectrumSamples] = {
        380.00f, 384.25f, 388.50f, 392.75f, 397.00f, 401.25f, 405.50f, 409.75f, 414.00f, 418.25f,
        422.50f, 426.75f, 431.00f, 435.25f, 439.50f, 443.75f, 448.00f, 452.25f, 456.50f, 460.75f,
        465.00f, 469.25f, 473.50f, 477.75f, 482.00f, 486.25f, 490.50f, 494.75f, 499.00f, 503.25f,
        507.50f, 511.75f, 516.00f, 520.25f, 524.50f, 528.75f, 533.00f, 537.25f, 541.50f, 545.75f,
        550.00f, 554.25f, 558.50f, 562.75f, 567.00f, 571.25f, 575.50f, 579.75f, 584.00f, 588.25f,
        592.50f, 596.75f, 601.00f, 605.25f, 609.50f, 613.75f, 618.00f, 622.25f, 626.50f, 630.75f,
        635.00f, 639.25f, 643.50f, 647.75f, 652.00f, 656.25f, 660.50f, 664.75f, 669.00f, 673.25f,
        677.50f, 681.75f, 686.00f, 690.25f, 694.50f, 698.75f, 703.00f, 707.25f, 711.50f, 715.75f,
        720.00f
    };

    constexpr float CIE_Y_integral = 106.911594f;

    /** @ref An RGB to Spectrum Conversion for Reflectances, Smits 2000 */
    static constexpr int nRGB2SpectrumSamples = 10;
    static constexpr float rgb2spectrum_lambda[nRGB2SpectrumSamples] = {
        380.00f, 417.78f, 455.55f, 493.33f, 531.11f,
        568.89f, 606.67f, 644.44f, 682.22f, 720.00f
    };
    static constexpr float rgb2spectrum_white_table[nRGB2SpectrumSamples] = {
        1.0000f, 1.0000f, 0.9999f, 0.9993f, 0.9992f, 0.9998f, 1.0000f, 1.0000f, 1.0000f, 1.0000f
    };
    static constexpr float rgb2spectrum_cyan_table[nRGB2SpectrumSamples] = {
        0.9710f, 0.9426f, 1.0007f, 1.0007f, 1.0007f, 1.0007f, 0.1564f, 0.0000f, 0.0000f, 0.0000f
    };
    static constexpr float rgb2spectrum_magenta_table[nRGB2SpectrumSamples] = {
        1.0000f, 1.0000f, 0.9685f, 0.2229f, 0.0000f, 0.0458f, 0.8369f, 1.0000f, 1.0000f, 0.9959f
    };
    static constexpr float rgb2spectrum_yellow_table[nRGB2SpectrumSamples] = {
        0.0001f, 0.0000f, 0.1088f, 0.6651f, 1.0000f, 1.0000f, 0.9996f, 0.9586f, 0.9685f, 0.9840f
    };
    static constexpr float rgb2spectrum_red_table[nRGB2SpectrumSamples] = {
        0.1012f, 0.0515f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.8325f, 1.0149f, 1.0149f, 1.0149f
    };
    static constexpr float rgb2spectrum_green_table[nRGB2SpectrumSamples] = {
        0.0000f, 0.0000f, 0.0273f, 0.7937f, 1.0000f, 0.9418f, 0.1719f, 0.0000f, 0.0000f, 0.0025f
    };
    static constexpr float rgb2spectrum_blue_table[nRGB2SpectrumSamples] = {
        1.0000f, 1.0000f, 0.8916f, 0.3323f, 0.0000f, 0.0000f, 0.0003f, 0.0369f, 0.0483f, 0.0496f
    };

    // Forward declaration
    class SampledSpectrum;
    HOSTDEVICE Vec3f XYZToSRGB(const Vec3f& xyz);
    HOSTDEVICE void XYZToSRGB(float xyz2rgb[3]);
    HOSTDEVICE Vec3f sRGBToXYZ(const Vec3f& rgb);
    HOSTDEVICE void sRGBToXYZ(float rgb2xyz[3]);
    HOSTDEVICE Vec3f linearToSRGB(const Vec3f& c);
    HOSTDEVICE Vec3f sRGBToLinear(const Vec3f& c);
    HOSTDEVICE Vec4f color2float(const Vec4u& c);
    HOSTDEVICE Vec3f color2float(const Vec3u& c);
    HOSTDEVICE Vec3u make_color(const Vec3f& c, bool gamma_enabled = true);
    HOSTDEVICE Vec4u make_color(const Vec4f& c, bool gamma_enabled = true);
    HOSTDEVICE float luminance(const Vec3f& c);
    static HOSTDEVICE float gauss(const float& x, const float& mu, const float& sigma1, const float& sigma2);
    HOSTDEVICE float CIE_X(const float& lambda);
    HOSTDEVICE float CIE_Y(const float& lambda);
    HOSTDEVICE float CIE_Z(const float& lambda);
    static HOSTDEVICE float averageSpectrumSamples(const float* lambda, const float* v, int n, const float& lambda_start, const float& lambda_end);
    HOSTDEVICE float linearInterpSpectrumSamples(const float* lambda, const float* v, int n, const float& l);
    HOSTDEVICE float Lerp(const float a, const float b, const float t);

     // SampledSpectrum ---------------------------------------------------------------
    class SampledSpectrum {
    public:
        static constexpr int nSamples = nSpectrumSamples;

        SampledSpectrum() = default;

        explicit HOSTDEVICE SampledSpectrum(float t)
        {
            for (int i = 0; i < nSpectrumSamples; i++)
                c[i] = t;
        }
        
        // Copy constructor
        HOSTDEVICE SampledSpectrum(const SampledSpectrum& spd)
        {
            for (int i = 0; i < nSpectrumSamples; i++)
                c[i] = spd.c[i];
        }

#ifndef __CUDACC__ /// @note Not defined on CUDA kernel
        static HOSTDEVICE SampledSpectrum fromSample(const float* lambda, const float* v, int n)
        {
            /// @todo Sort with lambda if the spectrum is randomly stored.

            SampledSpectrum ss;
            for (int i = 0; i < nSpectrumSamples; i++)
            {
                const float start_l = static_cast<float>(min_lambda);
                const float end_l = static_cast<float>(max_lambda);
                const float offset0 = float(i) / nSpectrumSamples;
                const float offset1 = float(i + 1) / nSpectrumSamples;
                float lambda0 = Lerp(start_l, end_l, offset0);
                float lambda1 = Lerp(start_l, end_l, offset1);
                ss.c[i] = averageSpectrumSamples(lambda, v, n, lambda0, lambda1);
            }
            return ss;
        }

        static HOST SampledSpectrum fromFile(const std::filesystem::path& filepath)
        {
            SampledSpectrum ss;
            std::vector<float> lambda;
            std::vector<float> value;

            std::ifstream ifs(filepath, std::ios::in);
            ASSERT(ifs.is_open(), "The SPD file '" + filepath.string() + "' is not found.");

            while (!ifs.eof())
            {
                std::string line;
                if (!std::getline(ifs, line)) continue;

                std::istringstream iss(line);
                float l, v;
                iss >> l >> v;
                lambda.emplace_back(l);
                value.emplace_back(v);
            }

            return fromSample(lambda.data(), value.data(), static_cast<int>(lambda.size()));
        }
#endif

        HOSTDEVICE float& operator[](int i) {
            return c[i];
        }

        HOSTDEVICE const float& operator[](int i) const
        {
            return c[i];
        }

        /* Addition */
        HOSTDEVICE SampledSpectrum& operator+=(const SampledSpectrum& s2)
        {
            for (int i = 0; i < nSpectrumSamples; i++)
                c[i] += s2.c[i];
            return *this;
        }
        HOSTDEVICE SampledSpectrum operator+(const SampledSpectrum& s2) const
        {
            SampledSpectrum ret = *this;
            for (int i = 0; i < nSpectrumSamples; i++)
                ret.c[i] += s2.c[i];
            return ret;
        }

        /* Subtraction */
        HOSTDEVICE SampledSpectrum& operator-=(const SampledSpectrum& s2)
        {
            for (int i = 0; i < nSpectrumSamples; i++)
                c[i] -= s2.c[i];
            return *this;
        }
        HOSTDEVICE SampledSpectrum operator-(const SampledSpectrum& s2) const
        {
            SampledSpectrum ret = *this;
            for (int i = 0; i < nSpectrumSamples; i++)
                ret.c[i] -= s2.c[i];
            return ret;
        }

        /* Multiplication */
        HOSTDEVICE SampledSpectrum& operator*=(const SampledSpectrum& s2)
        {
            for (int i = 0; i < nSpectrumSamples; i++)
                c[i] *= s2.c[i];
            return *this;
        }
        HOSTDEVICE SampledSpectrum operator*(const SampledSpectrum& s2) const
        {
            SampledSpectrum ret = *this;
            for (int i = 0; i < nSpectrumSamples; i++)
                ret.c[i] *= s2.c[i];
            return ret;
        }
        HOSTDEVICE SampledSpectrum& operator*=(const float& t)
        {
            for (int i = 0; i < nSpectrumSamples; i++)
                c[i] *= t;
            return *this;
        }
        HOSTDEVICE SampledSpectrum operator*(const float& t) const
        {
            SampledSpectrum ret = *this;
            for (int i = 0; i < nSpectrumSamples; i++)
                ret.c[i] *= t;
            return ret;
        }
        HOSTDEVICE friend inline SampledSpectrum operator*(const float& t, const SampledSpectrum& s)
        {
            assert(!isnan(t));
            return s * t;
        }

        /* Division */
        HOSTDEVICE SampledSpectrum& operator/=(const SampledSpectrum& s2)
        {
            for (int i = 0; i < nSpectrumSamples; i++)
                c[i] /= s2.c[i] != 0.0f ? s2.c[i] : 1.0f;
            return *this;
        }
        HOSTDEVICE SampledSpectrum operator/(const SampledSpectrum& s2) const
        {
            SampledSpectrum ret = *this;
            for (int i = 0; i < nSpectrumSamples; i++)
                ret.c[i] /= s2.c[i] != 0.0f ? s2.c[i] : 1.0f;
            return ret;
        }
        HOSTDEVICE SampledSpectrum& operator/=(const float& t)
        {
            for (int i = 0; i < nSpectrumSamples; i++)
                c[i] /= t;
            return *this;
        }
        HOSTDEVICE SampledSpectrum operator/(const float& t) const
        {
            SampledSpectrum ret = *this;
            for (int i = 0; i < nSpectrumSamples; i++)
                ret.c[i] /= t;
            return ret;
        }
        HOSTDEVICE friend inline SampledSpectrum operator/(const float& t, const SampledSpectrum& s)
        {
            assert(!isnan(t) && t != 0.0f);
            return s * t;
        }

        HOSTDEVICE bool isBlack() const
        {
            for (int i = 0; i < nSpectrumSamples; i++)
                if (c[i] != 0.0f) return false;
            return true;
        }

        HOSTDEVICE Vec3f toXYZ() const
        {
            Vec3f ret{0.0f, 0.0f, 0.0f};
            for (int i = 0; i < nSpectrumSamples; i++)
            {
                const float lambda = lerp(min_lambda, max_lambda, float(i) / nSpectrumSamples);
                ret[0] += c[i] * CIE_X(lambda);
                ret[1] += c[i] * CIE_Y(lambda);
                ret[2] += c[i] * CIE_Z(lambda);                
            }
            const float scale = float(max_lambda - min_lambda) / (CIE_Y_integral * nSpectrumSamples);

            return ret * scale;
        }

        HOSTDEVICE Vec3f toRGB() const
        {
            Vec3f xyz = toXYZ();
            return XYZToSRGB(xyz);
        }

        HOSTDEVICE float getSpectrumFromWavelength(const float& lambda) const
        {
            return linearInterpSpectrumSamples(spectrum_lambda, c, nSpectrumSamples, lambda);
        }

        HOSTDEVICE float y() const
        {
            float sum = 0.0f;
            for (int i = 0; i < nSpectrumSamples; i++)
            {
                sum += c[i];
            }
            return sum;
        }

        friend SampledSpectrum sqrtf(const SampledSpectrum& s)
        {
            SampledSpectrum ret;
            for (int i = 0; i < nSpectrumSamples; i++)
                ret.c[i] = sqrtf(s.c[i]);
            return ret;
        }

        friend SampledSpectrum expf(const SampledSpectrum& s)
        {
            SampledSpectrum ret;
            for (int i = 0; i < nSpectrumSamples; i++)
                ret.c[i] = expf(s.c[i]);
            return ret;
        }

        friend SampledSpectrum powf(const SampledSpectrum& s, float t)
        {
            SampledSpectrum ret;
            for (int i = 0; i < nSpectrumSamples; i++)
                ret.c[i] = powf(s.c[i], t);
            return ret;
        }
    private:
        float c[nSamples];
    };

    // Conversion from RGB to SampledSpectrum
    class RGB2Spectrum {
    public:
        RGB2Spectrum() = default;

#ifndef __CUDACC__
        void HOST init()
        {
            for (int i = 0; i < nSpectrumSamples; i++)
            {
                const float lambda0 = lerp(min_lambda, max_lambda, float(i) / nSpectrumSamples);
                const float lambda1 = lerp(min_lambda, max_lambda, float(i + 1) / nSpectrumSamples);

                m_white[i] = averageSpectrumSamples(rgb2spectrum_lambda, rgb2spectrum_white_table, nRGB2SpectrumSamples, lambda0, lambda1);
                m_cyan[i] = averageSpectrumSamples(rgb2spectrum_lambda, rgb2spectrum_cyan_table, nRGB2SpectrumSamples, lambda0, lambda1);
                m_magenta[i] = averageSpectrumSamples(rgb2spectrum_lambda, rgb2spectrum_magenta_table, nRGB2SpectrumSamples, lambda0, lambda1);
                m_yellow[i] = averageSpectrumSamples(rgb2spectrum_lambda, rgb2spectrum_yellow_table, nRGB2SpectrumSamples, lambda0, lambda1);
                m_red[i] = averageSpectrumSamples(rgb2spectrum_lambda, rgb2spectrum_red_table, nRGB2SpectrumSamples, lambda0, lambda1);
                m_green[i] = averageSpectrumSamples(rgb2spectrum_lambda, rgb2spectrum_green_table, nRGB2SpectrumSamples, lambda0, lambda1);
                m_blue[i] = averageSpectrumSamples(rgb2spectrum_lambda, rgb2spectrum_blue_table, nRGB2SpectrumSamples, lambda0, lambda1);
            }
        }
#endif // #ifndef __CUDACC__

        HOSTDEVICE INLINE const SampledSpectrum& whiteTable() const { return m_white; }
        HOSTDEVICE INLINE const SampledSpectrum& cyanTable() const { return m_cyan; }
        HOSTDEVICE INLINE const SampledSpectrum& magentaTable() const { return m_magenta; }
        HOSTDEVICE INLINE const SampledSpectrum& yellowTable() const { return m_yellow; }
        HOSTDEVICE INLINE const SampledSpectrum& redTable() const { return m_red; }
        HOSTDEVICE INLINE const SampledSpectrum& greenTable() const { return m_green; }
        HOSTDEVICE INLINE const SampledSpectrum& blueTable() const { return m_blue; }

        HOSTDEVICE INLINE SampledSpectrum getSpectrum(const Vec3f& rgb) const
        {
            SampledSpectrum ret(0);
            const float r = rgb[0];
            const float g = rgb[1];
            const float b = rgb[2];

            if (r <= g && r <= b)
            {
                ret += m_white * r;
                if (g <= b)
                {
                    ret += m_cyan * (g - r);
                    ret += m_blue * (b - g);
                }
                else
                {
                    ret += m_cyan * (b - r);
                    ret += m_green * (g - b);
                }
            }
            else if (g <= r && g <= b)
            {
                ret += m_white * g;
                if (r <= g)
                {
                    ret += m_magenta * (r - g);
                    ret += m_blue * (b - r);
                }
                else
                {
                    ret += m_magenta * (b - g);
                    ret += m_red * (r - b);
                }
            }
            else // blue <= red && blue <= green
            {
                ret += m_white * b;
                if (r <= g)
                {
                    ret += m_yellow * (r - b);
                    ret += m_green * (g - r);
                }
                else
                {
                    ret += m_yellow * (g - b);
                    ret += m_red * (r - g);
                }
            }
            return ret;
        }

    private:
        SampledSpectrum m_white;
        SampledSpectrum m_cyan;
        SampledSpectrum m_magenta;
        SampledSpectrum m_yellow;
        SampledSpectrum m_red;
        SampledSpectrum m_green;
        SampledSpectrum m_blue;
    };

#ifndef __CUDACC__
    /* Stream function of spectrum classes */
    HOST inline std::ostream& operator<<(std::ostream& out, const SampledSpectrum& spd)
    {
        for (int i = 0; i < SampledSpectrum::nSamples; i++)
            out << spd[i] << ' ';
        return out;
    }

    HOST inline std::ostream& operator<<(std::ostream& out, const RGB2Spectrum& r2s)
    {
        out << "RGB2Spectrum::white_table: " << r2s.whiteTable() << std::endl;
        out << "RGB2Spectrum::cyan_table: " << r2s.cyanTable() << std::endl;
        out << "RGB2Spectrum::magenta_table: " << r2s.magentaTable() << std::endl;
        out << "RGB2Spectrum::yellow_table: " << r2s.yellowTable() << std::endl;
        out << "RGB2Spectrum::red_table: " << r2s.redTable() << std::endl;
        out << "RGB2Spectrum::green_table: " << r2s.greenTable() << std::endl;
        out << "RGB2Spectrum::blue_table: " << r2s.blueTable();

        return out;
    }
#endif

    /* Conversion from XYZ to RGB color space, vice versa */
    HOSTDEVICE INLINE Vec3f XYZToSRGB(const Vec3f& xyz)
    {
        return Vec3f(
            3.2410f * xyz[0] - 1.5374f * xyz[1] - 0.4986f * xyz[2],
            -0.9692f * xyz[0] + 1.8760f * xyz[1] + 0.0416f * xyz[2],
            0.0556f * xyz[0] - 0.2040f * xyz[1] + 1.0507f * xyz[2]
        );
    }

    HOSTDEVICE INLINE void XYZToSRGB(float xyz2rgb[3])
    {
        const float x = xyz2rgb[0];
        const float y = xyz2rgb[1];
        const float z = xyz2rgb[2];
        xyz2rgb[0] = 3.2410f * x - 1.5374f * y - 0.4986f * z;
        xyz2rgb[1] = -0.9692f * x + 1.8760f * y + 0.0416f * z;
        xyz2rgb[2] = 0.0556f * x - 0.2040f * y + 1.0507f * z;
    }

    HOSTDEVICE INLINE Vec3f sRGBToXYZ(const Vec3f& rgb)
    {
        return Vec3f(
            0.4124f * rgb[0] + 0.3576f * rgb[1] + 0.1805f * rgb[2],
            0.2126f * rgb[0] + 0.7152f * rgb[1] + 0.0722f * rgb[2],
            0.0193f * rgb[0] + 0.1192f * rgb[1] + 0.9505f * rgb[2]
        );
    }

    HOSTDEVICE INLINE void sRGBToXYZ(float rgb2xyz[3])
    {
        const float r = rgb2xyz[0];
        const float g = rgb2xyz[1];
        const float b = rgb2xyz[2];
        rgb2xyz[0] = 0.4124f * r + 0.3576f * g + 0.1805f * b;
        rgb2xyz[1] = 0.2126f * r + 0.7152f * g + 0.0722f * b;
        rgb2xyz[2] = 0.0193f * r + 0.1192f * g + 0.9505f * b;
    }

    /* Conversion from linear to sRGB color, vice versa */
    HOSTDEVICE INLINE Vec3f linearToSRGB(const Vec3f& c)
    {
        float invGamma = 1.0f / 2.4f;
        Vec3f powed = Vec3f(powf(c.x(), invGamma), powf(c.y(), invGamma), powf(c.z(), invGamma));
        return Vec3f(
            c.x() < 0.0031308f ? 12.92f * c.x() : 1.055f * powed.x() - 0.055f,
            c.y() < 0.0031308f ? 12.92f * c.y() : 1.055f * powed.y() - 0.055f,
            c.z() < 0.0031308f ? 12.92f * c.z() : 1.055f * powed.z() - 0.055f
        );
    }

    HOSTDEVICE INLINE Vec3f sRGBToLinear(const Vec3f& c)
    {
        const float gamma = 2.4f;
        return Vec3f(
            c[0] < 0.0404482f ? c[0] / 12.92 : powf((c[0] + 0.055f) / 1.055f, gamma),
            c[1] < 0.0404482f ? c[1] / 12.92 : powf((c[1] + 0.055f) / 1.055f, gamma),
            c[2] < 0.0404482f ? c[2] / 12.92 : powf((c[2] + 0.055f) / 1.055f, gamma)
        );
    }

    /* 1 bit color to 4 bit float color */
    HOSTDEVICE INLINE Vec4f color2float(const Vec4u& c)
    {
        return Vec4f(
            static_cast<float>(c[0]) / 255.0f,
            static_cast<float>(c[1]) / 255.0f,
            static_cast<float>(c[2]) / 255.0f,
            static_cast<float>(c[3]) / 255.0f
        );
    }

    HOSTDEVICE INLINE Vec3f color2float(const Vec3u& c)
    {
        return Vec3f(
            static_cast<float>(c[0]) / 255.0f,
            static_cast<float>(c[1]) / 255.0f,
            static_cast<float>(c[2]) / 255.0f
        );
    }

    /* Conversion from float to 1 byte color considering gamma correction */
    HOSTDEVICE INLINE Vec3u make_color(const Vec3f& c, bool gamma_enalbed)
    {
        // first apply gamma, then convert to unsigned char
        Vec3f rgb = c;
        if (gamma_enalbed)
            rgb = linearToSRGB(clamp(c, 0.0f, 1.0f));
        return Vec3u(quantizeUnsigned8Bits(rgb.x()), quantizeUnsigned8Bits(rgb.y()), quantizeUnsigned8Bits(rgb.z()));
    }

    HOSTDEVICE INLINE Vec4u make_color(const Vec4f& c, bool gamma_enabled)
    {
        Vec3u rgb = make_color(Vec3f(c[0], c[1], c[2]), gamma_enabled);
        return Vec4u(rgb[0], rgb[1], rgb[2], (unsigned char)(clamp(c[3], 0.0f, 1.0f) * 255.0f));
    }

    /* Luminance of RGB color */
    HOSTDEVICE INLINE float luminance(const Vec3f& c)
    {
        return 0.2126f * c[0] + 0.7152f * c[1] + 0.0722f * c[2];
    }

    /* Approximation of CIE 1931 XYZ curve */
    /**
    * @note
    * I suppose that replacing this gauss function to the analytical approximation method [Wyman et al. 2013]
    * will be good for computation cost.
    * @ref http://cwyman.org/papers/jcgt13_xyzApprox.pdf
    */
    static HOSTDEVICE INLINE float gauss(const float& x, const float& mu, const float& sigma1, const float& sigma2)
    {
        return x < mu ? expf(-0.5f * (x - mu) * (x - mu) / (sigma1 * sigma1)) : expf(-0.5f * (x - mu) * (x - mu) / (sigma2 * sigma2));
    }

    HOSTDEVICE INLINE float CIE_X(const float& lambda)
    {
        return 1.056f * gauss(lambda, 599.8f, 37.9f, 31.0f) + 0.362f * gauss(lambda, 442.0f, 16.0f, 26.7f) - 0.065f * gauss(lambda, 501.1f, 20.4f, 26.2f);
    }

    HOSTDEVICE INLINE float CIE_Y(const float& lambda)
    {
        return 0.821f * gauss(lambda, 568.8f, 46.9f, 40.5f) + 0.286f * gauss(lambda, 530.9f, 16.3f, 31.1f);
    }

    HOSTDEVICE INLINE float CIE_Z(const float& lambda)
    {
        return 1.217f * gauss(lambda, 437.0f, 11.8f, 36.0f) + 0.681f * gauss(lambda, 459.0f, 26.0f, 13.8f);
    }

    static HOSTDEVICE INLINE float averageSpectrumSamples(const float* lambda, const float* v, int n, const float& lambda_start, const float& lambda_end)
    {
        /// @todo Check if input arrays are sorted with lambda 

        if (lambda_end <= lambda[0]) return v[0];
        if (lambda_start >= lambda[n - 1]) return v[n - 1];
        if (n == 1) return v[0];

        float sum = 0.0f;
        if (lambda_start < lambda[0]) sum += v[0] * (lambda[0] - lambda_start);
        if (lambda_end >= lambda[n - 1]) sum += v[n - 1] * (lambda_end - lambda[n - 1]);

        int i = 0;
        while (lambda_start > lambda[i + 1]) i++;

        auto interp = [lambda, v](float w, int i)
        {
            return lerp(v[i], v[i + 1], (w - lambda[i]) / (lambda[i + 1] - lambda[i]));
        };
        for (; i + 1 < n && lambda_end >= lambda[i]; i++)
        {
            float seg_lambda_start = fmaxf(lambda_start, lambda[i]);
            float seg_lambda_end = fminf(lambda_end, lambda[i + 1]);
            sum += 0.5f * (interp(seg_lambda_start, i) + interp(seg_lambda_end, i)) * (seg_lambda_end - seg_lambda_start);
        }

        return sum / (lambda_end - lambda_start);
    }

    /* Linear interpolation of spectrum value by lambda sample 'l' */
    HOSTDEVICE INLINE float linearInterpSpectrumSamples(
        const float* lambda, const float* v, int n, const float& l
    )
    {
        /// @todo Check if the 'lambda' is sorted

        if (l <= lambda[0]) return v[0];
        if (l >= lambda[n - 1]) return v[n - 1];
        int offset = 0;
        for (int i = 0; i < n - 1; i++)
        {
            /// @note Assumption: all lambda values are different
            if (lambda[i] <= l && lambda[i + 1] > l)
            {
                offset = i;
                break;
            }
        }
        const float t = (l - lambda[offset]) / (lambda[offset + 1] - lambda[offset]);
        return lerp(v[offset], v[offset + 1], t);
    }

    HOSTDEVICE INLINE float Lerp(const float a, const float b, const float t)
    {
        return a + t * (b - a);
    }

} // namespace prayground
