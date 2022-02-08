#pragma once

#include <optix.h>
#include <vector_types.h>
#include <prayground/math/vec_math.h>
#include <prayground/optix/macros.h>
#include <prayground/optix/helpers.h>

namespace prayground {

    constexpr int min_lambda = 380;
    constexpr int max_lambda = 780;
    constexpr float CIE_Y_integral = 106.919853f;

    class RGBSpectrum;
    template <int nSpectrumSamples> class SampledSpectrum;

    /* Conversion from XYZ to RGB color space, vice versa */
    HOSTDEVICE INLINE float3 XYZToSRGB(const float3& xyz)
    {
        return make_float3(
             3.2410f * xyz.x - 1.5374f * xyz.y - 0.4986f * xyz.z,
            -0.9692f * xyz.x + 1.8760f * xyz.y + 0.0416f * xyz.z,
             3.2410f * xyz.x - 0.2040f * xyz.y + 1.0507f * xyz.z
        );
    }

    HOSTDEVICE INLINE void XYZToSRGB(float xyz2rgb[3])
    {
        const float x = xyz2rgb[0];
        const float y = xyz2rgb[1];
        const float z = xyz2rgb[2];
        xyz2rgb[0] = 3.2410f * x - 1.5374f * y - 0.4986f * z;
        xyz2rgb[1] = -0.9692f * x + 1.8760f * y + 0.0416f * z;
        xyz2rgb[2] = 3.2410f * x - 0.2040f * y + 1.0507f * z;
    }

    HOSTDEVICE INLINE float3 sRGBToXYZ(const float3& rgb)
    {
        return make_float3(
            0.4124f * rgb.x + 0.3576f * rgb.y + 0.1805f * rgb.z,
            0.2126f * rgb.x + 0.7152f * rgb.y + 0.0722f * rgb.z,
            0.0193f * rgb.x + 0.1192f * rgb.y + 0.9505f * rgb.z
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
    HOSTDEVICE INLINE float3 linearToSRGB(const float3& c)
    {
        float invGamma = 1.0f / 2.4f;
        float3 powed = make_float3(powf(c.x, invGamma), powf(c.y, invGamma), powf(c.z, invGamma));
        return make_float3(
            c.x < 0.0031308f ? 12.92f * c.x : 1.055f * powed.x - 0.055f,
            c.y < 0.0031308f ? 12.92f * c.y : 1.055f * powed.y - 0.055f,
            c.z < 0.0031308f ? 12.92f * c.z : 1.055f * powed.z - 0.055f
        );
    }

    HOSTDEVICE INLINE RGBSpectrum linearToSRGB(const RGBSpectrum& c)
    {
        float  invGamma = 1.0f / 2.4f;
        RGBSpectrum ret;
        RGBSpectrum powed = powf(c, invGamma);
        ret[0] = c[0] < 0.0031308f ? 12.92f * c[0] : 1.055f * powed[0] - 0.055f;
        ret[1] = c[1] < 0.0031308f ? 12.92f * c[1] : 1.055f * powed[1] - 0.055f;
        ret[2] = c[2] < 0.0031308f ? 12.92f * c[2] : 1.055f * powed[2] - 0.055f;
        return ret;
    }

    HOSTDEVICE INLINE float3 sRGBToLinear(const float3& c)
    {
        const float gamma = 2.4f;
        return make_float3(
            c.x < 0.0404482f ? c.x / 12.92 : powf((c.x + 0.055f) / 1.055f, gamma),
            c.y < 0.0404482f ? c.y / 12.92 : powf((c.y + 0.055f) / 1.055f, gamma),
            c.z < 0.0404482f ? c.z / 12.92 : powf((c.z + 0.055f) / 1.055f, gamma)
        );
    }

    HOSTDEVICE INLINE RGBSpectrum sRGBToLinear(const RGBSpectrum& c)
    {
        const float gamma = 2.4f;
        RGBSpectrum ret;
        ret[0] = c[0] < 0.0404482f ? c[0] / 12.92 : powf((c[0] + 0.055f) / 1.055f, gamma);
        ret[1] = c[1] < 0.0404482f ? c[1] / 12.92 : powf((c[1] + 0.055f) / 1.055f, gamma);
        ret[2] = c[2] < 0.0404482f ? c[2] / 12.92 : powf((c[2] + 0.055f) / 1.055f, gamma);
        return ret;
    }

    /* 1 byte color to 4 bit float color */
    HOSTDEVICE INLINE float4 color2float(const uchar4& c)
    {
        return make_float4(
            static_cast<float>(c.x) / 255.0f,
            static_cast<float>(c.y) / 255.0f,
            static_cast<float>(c.z) / 255.0f,
            static_cast<float>(c.w) / 255.0f
        );
    }

    HOSTDEVICE INLINE float3 color2float(const uchar3& c)
    {
        return make_float3(
            static_cast<float>(c.x) / 255.0f,
            static_cast<float>(c.y) / 255.0f,
            static_cast<float>(c.z) / 255.0f
        );
    }

    /* Conversion from float to 1 byte color considering gamma correction */
    HOSTDEVICE INLINE uchar3 make_color(const float3& c, bool gamma_enalbed = true)
    {
        // first apply gamma, then convert to unsigned char
        float3 rgb = c;
        if (gamma_enalbed)
            rgb = linearToSRGB(clamp(c, 0.0f, 1.0f));
        return make_uchar3(quantizeUnsigned8Bits(rgb.x), quantizeUnsigned8Bits(rgb.y), quantizeUnsigned8Bits(rgb.z));
    }

    HOSTDEVICE INLINE uchar4 make_color(const float4& c, bool gamma_enabled = true)
    {
        uchar3 rgb = make_color(make_float3(c.x, c.y, c.z), gamma_enabled);
        return make_uchar4(rgb.x, rgb.y, rgb.z, (unsigned char)(clamp(c.w, 0.0f, 1.0f) * 255.0f));
    }

    /* Luminance of RGB color */
    HOSTDEVICE INLINE float luminance(const float3& c)
    {
        return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
    }

    /* Approximation of CIE 1931 XYZ curve */
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

    static HOSTDEVICE INLINE float averageSpectrumSamples(const float* lambda, const float* v, int n, const float& lambdaStart, const float& lambdaEnd)
    {
        /// @todo Check if input arrays are sorted with lambda 

        if (lambdaEnd <= lambda[0]) return v[0];
        if (lambdaStart >= lambda[n - 1]) return v[n - 1];
        if (n == 1) return v[0];

        float sum = 0.0f;
        if (lambdaStart < lambda[0]) sum += v[0] * (lambda[0] - lambdaStart);
        if (lambdaEnd >= lambda[n - 1]) sum += v[n - 1] * (lambdaEnd - lambda[n - 1]);

        int i = 0; 
        while (lambdaStart > lambda[i + 1]) i++;

        auto interp = [lambda, v] HOSTDEVICE(float w, int i)
        {
            return lerp(v[i], v[i + 1], (w - lambda[i]) / (lambda[i + 1] - lambda[i]));
        };
        for (; i + 1 < n && lambdaEnd >= lambda[i]; i++)
        {
            float segLambdaStart = fmaxf(lambdaStart, lambda[i]);
            float segLambdaEnd = fminf(lambdaEnd, lambda[i + 1]);
            sum += 0.5f * (interp(segLambdaStart, i) + interp(segLambdaEnd, i)) * (segLambdaEnd - segLambdaStart);
        }

        return sum / (lambdaEnd - lambdaStart);
    }
    
    /* Linearly interpolate spectrum value with lambda sample 'l' */
    HOSTDEVICE INLINE float linearInterpSpectrumSamples(
        const float* lambda, const float* v, int n, const float& l
    )
    {
        /// @todo Check if the 'lambda' is sorted

        if (l <= lambda[0]) return v[0];
        if (l >= lambda[n-1]) return v[n-1];
        int offset = 0;
        for (int i = 0; i < n-1; i++)
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

    /**
     * 各関数や実装においてtemplate <Spectrum> とするが、
     * OptiXでは仮想関数呼び出しが禁止されているので、
     * SampledSpectrumとRGBSpectrumのメンバ関数の名前を同じにし、
     * コーディングルールによってコンパイルエラーを防ぐ 
     */

    // RGBSpectrum ---------------------------------------------------------------
    class RGBSpectrum {
    public:
        static constexpr int nSamples = 3;

        RGBSpectrum(float t = 0.0f)
        {
            c[0] = t; c[1] = t; c[2] = t;
        }

        const float& operator[](int i) const
        {
            return c[i];
        }

        float& operator[](int i)
        {
            return c[i];
        }

        /* Addition */
        RGBSpectrum& operator+=(const RGBSpectrum& s2)
        {
            for (int i = 0; i < 3; i++)
                c[i] += s2.c[i];
            return *this;
        }
        RGBSpectrum operator+(const RGBSpectrum& s2)
        {
            RGBSpectrum ret;
            for (int i = 0; i < 3; i++)
                ret.c[i] += s2.c[i];
            return ret;
        }

        /* Subtraction */
        RGBSpectrum& operator-=(const RGBSpectrum& s2)
        {
            for (int i = 0; i < 3; i++)
                c[i] -= s2.c[i];
            return *this;
        }
        RGBSpectrum operator-(const RGBSpectrum& s2)
        {
            RGBSpectrum ret;
            for (int i = 0; i < 3; i++)
                ret.c[i] -= s2.c[i];
            return ret;
        }

        /* Multiplication */
        RGBSpectrum& operator*=(const RGBSpectrum& s2)
        {
            for (int i = 0; i < 3; i++)
                c[i] *= s2.c[i];
            return *this;
        }
        RGBSpectrum operator*(const RGBSpectrum& s2)
        {
            RGBSpectrum ret;
            for (int i = 0; i < 3; i++)
                ret.c[i] *= s2.c[i];
            return ret;
        }

        /* Division */
        RGBSpectrum& operator/=(const RGBSpectrum& s2)
        {
            for (int i = 0; i < 3; i++)
                c[i] /= s2.c[i] != 0.0f ? s2.c[i] : 1.0f;
            return *this;
        }
        RGBSpectrum operator/(const RGBSpectrum& s2)
        {
            RGBSpectrum ret;
            for (int i = 0; i < 3; i++)
                c[i] /= s2.c[i] != 0.0f ? s2.c[i] : 1.0f;
            return ret;
        }

        bool isBlack() const
        {
            for (int i = 0; i < 3; i++)
                if (c[i] != 0.0f) return false;
            return true;
        }

        float3 toXYZ() const 
        {
            return sRGBToXYZ(make_float3(c[0], c[1], c[2]));
        }

        float3 toRGB() const 
        {
            return make_float3(c[0], c[1], c[2]);
        }

        friend RGBSpectrum sqrtf(const RGBSpectrum& s)
        {
            RGBSpectrum ret;
            for (int i = 0; i < 3; i++)
                ret.c[i] = sqrtf(s.c[i]);
            return ret;
        }

        friend RGBSpectrum expf(const RGBSpectrum& s)
        {
            RGBSpectrum ret;
            for (int i = 0; i < 3; i++)
                ret.c[i] = expf(s.c[i]);
            return ret;
        }

        friend RGBSpectrum powf(const RGBSpectrum& s, float t)
        {
            RGBSpectrum ret;
            for (int i = 0; i < 3; i++)
                ret.c[i] = powf(s.c[i], t);
            return ret;
        }
    private:
        float c[3];
    };

     // SampledSpectrum ---------------------------------------------------------------
    template <int nSpectrumSamples>
    class SampledSpectrum {
    public:
        static constexpr int nSamples = nSpectrumSamples;

        SampledSpectrum(float t)
        {
            for (int i = 0; i < nSpectrumSamples; i++)
                c[i] = t;
        }

        static SampledSpectrum fromSample(float* lambda, float* v, int n)
        {
            /// @todo Sort with lambda if the spectrum is randomly stored.

            SampledSpectrum ss;
            for (int i = 0; i < nSpectrumSamples; i++)
            {
                float lambda0 = lerp(min_lambda, max_lambda, float(i) / nSpectrumSamples);
                float lambda1 = lerp(min_lambda, max_lambda, float(i+1) / nSpectrumSamples);
                ss.c[i] = averageSpectrumSamples(lambda, v, n, lamdba0, lambda1);
            }
            return ss;
        }

        float& operator[](int i) {
            return c[i];
        }

        /* Addition */
        SampledSpectrum& operator+=(const SampledSpectrum& s2)
        {
            for (int i = 0; i < nSpectrumSamples; i++)
                c[i] += s2.c[i];
            return *this;
        }
        SampledSpectrum operator+(const SampledSpectrum& s2)
        {
            SampledSpectrum ret;
            for (int i = 0; i < nSpectrumSamples; i++)
                ret.c[i] += s2.c[i];
            return ret;
        }

        /* Subtraction */
        SampledSpectrum& operator-=(const SampledSpectrum& s2)
        {
            for (int i = 0; i < nSpectrumSamples; i++)
                c[i] -= s2.c[i];
            return *this;
        }
        SampledSpectrum operator-(const SampledSpectrum& s2)
        {
            SampledSpectrum ret;
            for (int i = 0; i < nSpectrumSamples; i++)
                ret.c[i] -= s2.c[i];
            return ret;
        }

        /* Multiplication */
        SampledSpectrum& operator*=(const SampledSpectrum& s2)
        {
            for (int i = 0; i < nSpectrumSamples; i++)
                c[i] *= s2.c[i];
            return *this;
        }
        SampledSpectrum operator*(const SampledSpectrum& s2)
        {
            SampledSpectrum ret;
            for (int i = 0; i < nSpectrumSamples; i++)
                ret.c[i] *= s2.c[i];
            return ret;
        }

        /* Division */
        SampledSpectrum& operator/=(const SampledSpectrum& s2)
        {
            for (int i = 0; i < nSpectrumSamples; i++)
                c[i] /= s2.c[i] != 0.0f ? s2.c[i] : 1.0f;
            return *this;
        }
        SampledSpectrum operator/(const SampledSpectrum& s2)
        {
            SampledSpectrum ret;
            for (int i = 0; i < nSpectrumSamples; i++)
                c[i] /= s2.c[i] != 0.0f ? s2.c[i] : 1.0f;
            return ret;
        }

        bool isBlack() const
        {
            for (int i = 0; i < nSpectrumSamples; i++)
                if (c[i] != 0.0f) return false;
            return true;
        }

        float3 toXYZ() const 
        {
            float3 ret{0.0f, 0.0f, 0.0f};
            for (int i = 0; i < nSpectrumSamples; i++)
            {
                const float lambda = lerp(min_lambda, max_lambda, float(i) / nSpectrumSamples);
                ret.x += c[i] * CIE_X(lambda);
                ret.y += c[i] * CIE_Y(lambda);
                ret.z += c[i] * CIE_Z(lambda);                
            }
            return ret / CIE_Y_integral;
        }

        float3 toRGB() const 
        {
            float3 xyz = toXYZ();
            return XYZtoSRGB(xyz);
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

} // namespace prayground
