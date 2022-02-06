#pragma once

#include <optix.h>
#include <vector_types.h>
#include <prayground/math/vec_math.h>
#include <prayground/optix/macros.h>
#include <prayground/optix/helpers.h>


/**
 * 各関数や実装においてtemplate <Spectrum> とするが、
 * OptiXでは仮想関数呼び出しが禁止されているので、
 * SampledSpectrumとRGBSpectrumのメンバ関数の名前を同じにし、
 * コーディングルールによってコンパイルエラーを防ぐ */ 
template <int nSpectrumSamples>
class SampledSpectrum {
public:
    static constexpr int nSamples = nSpectrumSamples;

    SampledSpectrum(float t)
    {
        for (int i = 0; i < nSpectrumSamples; i++)
            c[i] = t;
    }

    float& operator[](int i) {
        return c[i];
    }

    /* Addition */
    SampledSpectrum &operator+=(const SampledSpectrum& s2)
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

HOSTDEVICE INLINE float3 linearTosRGB(const float3& c)
{
    float invGamma = 1.0f / 2.4f;
    float3 powed = make_float3(powf(c.x, invGamma), powf(c.y, invGamma), powf(c.z, invGamma));
    return make_float3(
        c.x < 0.0031308f ? 12.92f * c.x : 1.055f * powed.x - 0.055f, 
        c.y < 0.0031308f ? 12.92f * c.y : 1.055f * powed.y - 0.055f,
        c.z < 0.0031308f ? 12.92f * c.z : 1.055f * powed.z - 0.055f
    );
}

HOSTDEVICE INLINE RGBSpectrum linearToSRGB( const RGBSpectrum& c )
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
        c.x < 0.0404482f ? c.x/12.92 : powf((c.x + 0.055f)/1.055f, gamma),
        c.y < 0.0404482f ? c.y/12.92 : powf((c.y + 0.055f)/1.055f, gamma),
        c.z < 0.0404482f ? c.z/12.92 : powf((c.z + 0.055f)/1.055f, gamma)
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

HOSTDEVICE INLINE float4 color2float( const uchar4& c )
{
    return make_float4(
        static_cast<float>(c.x) / 255.0f,
        static_cast<float>(c.y) / 255.0f,
        static_cast<float>(c.z) / 255.0f,
        static_cast<float>(c.w) / 255.0f       
    );
}

HOSTDEVICE INLINE float3 color2float( const uchar3& c )
{
    return make_float3(
        static_cast<float>(c.x) / 255.0f,
        static_cast<float>(c.y) / 255.0f,
        static_cast<float>(c.z) / 255.0f
    );
}

INLINE HOSTDEVICE uchar3 make_color( const float3& c, bool gamma_enalbed=true)
{
    // first apply gamma, then convert to unsigned char
    float3 rgb = c;
    if (gamma_enalbed)
        rgb = toSRGB( clamp( c, 0.0f, 1.0f ) );
    return make_uchar3( quantizeUnsigned8Bits( rgb.x ), quantizeUnsigned8Bits( rgb.y ), quantizeUnsigned8Bits( rgb.z ));
}

INLINE HOSTDEVICE uchar4 make_color( const float4& c, bool gamma_enabled=true )
{
    uchar3 rgb = make_color( make_float3( c.x, c.y, c.z ), gamma_enabled );
    return make_uchar4( rgb.x, rgb.y, rgb.z, (unsigned char)( clamp( c.w, 0.0f, 1.0f ) * 255.0f ) );
}

/**
 * @ref https://qiita.com/yoya/items/96c36b069e74398796f3
 * 
 * @note CIE XYZ
 */
HOSTDEVICE INLINE float luminance( const float3& c )
{
    return 0.299f*c.x + 0.587f*c.y + 0.114f*c.z;
}
