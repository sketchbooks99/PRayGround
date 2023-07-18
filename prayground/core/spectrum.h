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

    // Forward declaration
    class SampledSpectrum;
    HOSTDEVICE INLINE Vec3f XYZToSRGB(const Vec3f& xyz);
    HOSTDEVICE INLINE void XYZToSRGB(float xyz2rgb[3]);
    HOSTDEVICE INLINE Vec3f sRGBToXYZ(const Vec3f& rgb);
    HOSTDEVICE INLINE void sRGBToXYZ(float rgb2xyz[3]);
    HOSTDEVICE INLINE Vec3f linearToSRGB(const Vec3f& c);
    HOSTDEVICE INLINE Vec3f sRGBToLinear(const Vec3f& c);
    HOSTDEVICE INLINE Vec4f color2float(const Vec4u& c);
    HOSTDEVICE INLINE Vec3f color2float(const Vec3u& c);
    HOSTDEVICE INLINE Vec3u make_color(const Vec3f& c, bool gamma_enabled = true);
    HOSTDEVICE INLINE Vec4u make_color(const Vec4f& c, bool gamma_enabled = true);
    HOSTDEVICE INLINE float luminance(const Vec3f& c);
    HOSTDEVICE INLINE float gauss(const float& x, const float& mu, const float& sigma1, const float& sigma2);
    HOSTDEVICE INLINE float CIE_X(const float& lambda);
    HOSTDEVICE INLINE float CIE_Y(const float& lambda);
    HOSTDEVICE INLINE float CIE_Z(const float& lambda);
    HOSTDEVICE INLINE float averageSpectrumSamples(const float* lambda, const float* v, int n, const float& lambda_start, const float& lambda_end);
    HOSTDEVICE INLINE float linearInterpSpectrumSamples(const float* lambda, const float* v, int n, const float& l);
    HOSTDEVICE INLINE float Lerp(const float a, const float b, const float t);

    /** 
    * @note
    * Could I use array to reconstruct spectrum from rgb as global variable on the device
    * if I would be able to initialize the array in compile-time?
    */
    namespace constants {
        constexpr int min_lambda = 380;
        constexpr int max_lambda = 720;
        constexpr int num_spectrum_samples = 81;
        constexpr float CIE_Y_integral = 106.911594f;

        constexpr float spectrum_lambda[num_spectrum_samples] = {
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
    }

    // SampledSpectrum ---------------------------------------------------------------
    struct SampledSpectrum {
        static constexpr int nSamples = constants::num_spectrum_samples;

        float c[nSamples];

#ifndef __CUDACC__ /// @note Not defined on CUDA kernel
        static HOSTDEVICE SampledSpectrum fromSample(const float* lambda, const float* v, int n)
        {
            /// @todo Sort with lambda if the spectrum is randomly stored.

            SampledSpectrum ss;
            for (int i = 0; i < nSamples; i++)
            {
                const float start_l = static_cast<float>(constants::min_lambda);
                const float end_l = static_cast<float>(constants::max_lambda);
                const float offset0 = float(i) / nSamples;
                const float offset1 = float(i + 1) / nSamples;
                float lambda0 = Lerp(start_l, end_l, offset0);
                float lambda1 = Lerp(start_l, end_l, offset1);
                ss.c[i] = averageSpectrumSamples(lambda, v, n, lambda0, lambda1);
            }
            return ss;
        }

        static HOST SampledSpectrum fromFile(const std::filesystem::path& filepath)
        {
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
        static HOSTDEVICE SampledSpectrum zero()
        {
            SampledSpectrum ret;
            for (int i = 0; i < nSamples; i++)
                ret.c[i] = 0.0f;
            return ret;
        }

        static HOSTDEVICE SampledSpectrum constant(const float t)
        {
            SampledSpectrum ret;
            for (int i = 0; i < nSamples; i++)
                ret.c[i] = t;
            return ret;
        }

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
            for (int i = 0; i < nSamples; i++)
                c[i] += s2.c[i];
            return *this;
        }
        HOSTDEVICE SampledSpectrum operator+(const SampledSpectrum& s2) const
        {
            SampledSpectrum ret = *this;
            for (int i = 0; i < nSamples; i++)
                ret.c[i] += s2.c[i];
            return ret;
        }

        /* Subtraction */
        HOSTDEVICE SampledSpectrum& operator-=(const SampledSpectrum& s2)
        {
            for (int i = 0; i < nSamples; i++)
                c[i] -= s2.c[i];
            return *this;
        }
        HOSTDEVICE SampledSpectrum operator-(const SampledSpectrum& s2) const
        {
            SampledSpectrum ret = *this;
            for (int i = 0; i < nSamples; i++)
                ret.c[i] -= s2.c[i];
            return ret;
        }

        /* Multiplication */
        HOSTDEVICE SampledSpectrum& operator*=(const SampledSpectrum& s2)
        {
            for (int i = 0; i < nSamples; i++)
                c[i] *= s2.c[i];
            return *this;
        }
        HOSTDEVICE SampledSpectrum operator*(const SampledSpectrum& s2) const
        {
            SampledSpectrum ret = *this;
            for (int i = 0; i < nSamples; i++)
                ret.c[i] *= s2.c[i];
            return ret;
        }
        HOSTDEVICE SampledSpectrum& operator*=(const float& t)
        {
            for (int i = 0; i < nSamples; i++)
                c[i] *= t;
            return *this;
        }
        HOSTDEVICE SampledSpectrum operator*(const float& t) const
        {
            SampledSpectrum ret = *this;
            for (int i = 0; i < nSamples; i++)
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
            for (int i = 0; i < nSamples; i++)
                c[i] /= s2.c[i] != 0.0f ? s2.c[i] : 1.0f;
            return *this;
        }
        HOSTDEVICE SampledSpectrum operator/(const SampledSpectrum& s2) const
        {
            SampledSpectrum ret = *this;
            for (int i = 0; i < nSamples; i++)
                ret.c[i] /= s2.c[i] != 0.0f ? s2.c[i] : 1.0f;
            return ret;
        }
        HOSTDEVICE SampledSpectrum& operator/=(const float& t)
        {
            for (int i = 0; i < nSamples; i++)
                c[i] /= t;
            return *this;
        }
        HOSTDEVICE SampledSpectrum operator/(const float& t) const
        {
            SampledSpectrum ret = *this;
            for (int i = 0; i < nSamples; i++)
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
            for (int i = 0; i < nSamples; i++)
                if (c[i] != 0.0f) return false;
            return true;
        }

        HOSTDEVICE Vec3f toXYZ() const
        {
            Vec3f ret{ 0.0f, 0.0f, 0.0f };
            for (int i = 0; i < nSamples; i++)
            {
                const float lambda = lerp(constants::min_lambda, constants::max_lambda, float(i) / nSamples);
                ret[0] += c[i] * CIE_X(lambda);
                ret[1] += c[i] * CIE_Y(lambda);
                ret[2] += c[i] * CIE_Z(lambda);
            }
            const float scale = float(constants::max_lambda - constants::min_lambda) / (constants::CIE_Y_integral * nSamples);

            return ret * scale;
        }

        HOSTDEVICE Vec3f toRGB() const
        {
            Vec3f xyz = toXYZ();
            return XYZToSRGB(xyz);
        }

        HOSTDEVICE float getSpectrumFromWavelength(const float& lambda) const
        {
            return linearInterpSpectrumSamples(constants::spectrum_lambda, c, nSamples, lambda);
        }

        HOSTDEVICE float y() const
        {
            float sum = 0.0f;
            for (int i = 0; i < nSamples; i++)
            {
                sum += c[i];
            }
            return sum;
        }

        friend SampledSpectrum sqrtf(const SampledSpectrum& s)
        {
            SampledSpectrum ret;
            for (int i = 0; i < nSamples; i++)
                ret.c[i] = sqrtf(s.c[i]);
            return ret;
        }

        friend SampledSpectrum expf(const SampledSpectrum& s)
        {
            SampledSpectrum ret;
            for (int i = 0; i < nSamples; i++)
                ret.c[i] = expf(s.c[i]);
            return ret;
        }

        friend SampledSpectrum powf(const SampledSpectrum& s, float t)
        {
            SampledSpectrum ret;
            for (int i = 0; i < nSamples; i++)
                ret.c[i] = powf(s.c[i], t);
            return ret;
        }
    };

    // Tables to reconstruct SampledSpectrum from RGB value.
    /// @ref An RGB to Spectrum Conversion for Reflectances, Smits 2000
    namespace constants {
        constexpr SampledSpectrum rgb2spectrum_white = {
            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.999997f,
            0.999985f, 0.999973f, 0.999962f, 0.999952f, 0.999941f, 0.999929f, 0.999917f, 0.999907f, 0.999876f, 0.999813f,
            0.999742f, 0.999671f, 0.999607f, 0.999543f, 0.99948f, 0.999409f, 0.999337f, 0.999296f, 0.999285f, 0.999274f,
            0.999263f, 0.999251f, 0.99924f, 0.999229f, 0.999219f, 0.999207f, 0.99923f, 0.999294f, 0.999357f, 0.999421f,
            0.999492f, 0.999564f, 0.999627f, 0.999691f, 0.999754f, 0.999808f, 0.999832f, 0.999853f, 0.999875f, 0.999896f,
            0.99992f, 0.999943f, 0.999965f, 0.999986f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
            1.0f
        };

        constexpr SampledSpectrum rgb2spectrum_cyan = {
            0.969497f, 0.96649f, 0.963483f, 0.960476f, 0.957469f, 0.954086f, 0.950704f, 0.947697f, 0.94469f, 0.944651f,
            0.951399f, 0.958321f, 0.964474f, 0.970627f, 0.97678f, 0.983702f, 0.990624f, 0.996777f, 1.00064f, 1.0007f, 
            1.0007f, 1.0007f, 1.0007f, 1.0007f, 1.0007f, 1.0007f, 1.0007f, 1.0007f, 1.0007f, 1.0007f,
            1.0007f, 1.0007f, 1.0007f, 1.0007f, 1.0007f, 1.0007f, 1.0007f, 1.0007f, 1.0007f, 1.0007f,
            1.0007f, 1.0007f, 1.0007f, 1.0007f, 1.0007f, 0.96295f, 0.864155f, 0.774764f, 0.685373f, 0.595981f,
            0.495416f, 0.394851f, 0.30546f, 0.216068f, 0.151914f, 0.132259f, 0.113625f, 0.0970615f, 0.0804981f, 0.0639347f, 
            0.0453009f, 0.0266671f, 0.0101037f, 0.000100209f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 
            0.0f
        };

        constexpr SampledSpectrum rgb2spectrum_magenta = {
            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.998919f,
            0.99523f, 0.991477f, 0.988141f, 0.984805f, 0.981469f, 0.977716f, 0.973963f, 0.970627f, 0.939169f, 0.860942f,
            0.772133f, 0.683325f, 0.604383f, 0.525442f, 0.446501f, 0.357692f, 0.268883f, 0.213235f, 0.189447f, 0.165847f, 
            0.139298f, 0.112748f, 0.0891481f, 0.0655483f, 0.0419485f, 0.0153988f, 0.00230198f, 0.00714035f, 0.0119895f, 0.0168386f,
            0.0222939f, 0.0277491f, 0.0325982f, 0.0374474f, 0.0422965f, 0.0810752f, 0.173741f, 0.2575f, 0.341258f, 0.425017f, 
            0.519246f, 0.613474f, 0.697233f, 0.780991f, 0.841711f, 0.862075f, 0.881508f, 0.89878f, 0.916053f, 0.933326f,
            0.952758f, 0.97219f, 0.989463f, 0.999896f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 0.999806f, 0.999373f, 0.998939f, 0.99845f, 0.997962f, 0.997528f,  0.997094f, 0.99666f, 
            0.996171f
        };

        constexpr SampledSpectrum rgb2spectrum_yellow = {
            0.0000947062f, 0.0000841186f, 0.000073531f, 0.0000629434f, 0.0000523557f, 0.0000404447f, 0.0000285336f, 0.000017946f, 0.00000735839f, 0.0037336f, 
            0.016477f, 0.0294397f, 0.040962f, 0.0524844f, 0.0640068f, 0.0769695f, 0.0899321f, 0.101455f, 0.130599f, 0.18905f, 
            0.255311f, 0.321572f, 0.380471f, 0.43937f, 0.498269f, 0.56453f, 0.630792f, 0.679824f, 0.715362f, 0.75082f,
            0.79071f, 0.8306f, 0.866058f, 0.901516f, 0.936974f, 0.976864f, 0.999987f, 1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.999982f, 0.999935f, 0.999893f, 0.999851f, 0.999808f,
            0.999761f, 0.999713f, 0.999671f, 0.999628f, 0.998096f, 0.993271f, 0.988387f, 0.984044f, 0.979702f, 0.97536f,
            0.970476f, 0.965591f, 0.961249f, 0.959041f, 0.960057f, 0.961236f, 0.962415f, 0.963464f, 0.964512f, 0.96556f,
            0.966739f, 0.967918f, 0.969231f, 0.970871f, 0.972512f, 0.974359f, 0.976205f, 0.977846f, 0.979487f, 0.981128f, 
            0.982974f
        };

        constexpr SampledSpectrum rgb2spectrum_red = {
            0.098569f, 0.0933069f, 0.0880449f, 0.0827828f, 0.0775208f, 0.071601f, 0.0656812f, 0.0604192f, 0.0551571f, 0.0498329f,
            0.0437007f, 0.0375649f, 0.0321108f, 0.0266567f, 0.0212027f, 0.0150668f, 0.00893101f, 0.00347695f, 0.0000515556, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0372223f, 0.134637f, 0.222778f, 0.31092f, 0.399062f,
            0.498222f, 0.597382f, 0.685524f, 0.773666f, 0.837958f, 0.860654f, 0.882386f, 0.901703f, 0.92102f, 0.940337f,
            0.962068f, 0.9838f, 1.00312f, 1.01478f, 1.0149f, 1.0149f, 1.0149f, 1.0149f, 1.0149f, 1.0149f,
            1.0149f, 1.0149f, 1.0149f, 1.0149f, 1.0149f, 1.0149f, 1.0149f, 1.0149f, 1.0149f, 1.0149f,
            1.0149f
        }; 

        constexpr SampledSpectrum rgb2spectrum_green = {
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.000936781f,
            0.00413439f, 0.00738698f, 0.0102782f, 0.0131693f, 0.0160605f, 0.0193131f, 0.0225657f, 0.0254569f, 0.0574544f, 0.137858f,
            0.229145f, 0.320431f, 0.401574f, 0.482718f, 0.563861f, 0.655148f, 0.746434f, 0.802617f, 0.824661f, 0.846504f,
            0.871076f, 0.895649f, 0.917491f, 0.939333f, 0.961176f, 0.985748f, 0.997078f, 0.990927f, 0.984764f, 0.978603f,
            0.97167f, 0.964738f, 0.958576f, 0.952414f, 0.946252f, 0.907499f, 0.817288f, 0.735773f, 0.654259f, 0.572745f,
            0.481042f, 0.389339f, 0.307824f, 0.22631f, 0.166735f, 0.145366f, 0.124886f, 0.106681f, 0.0884759f, 0.070271f,
            0.0497904f, 0.0293099f, 0.011105f, 0.000110141f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.000118189f, 0.000382479f, 0.000647169f, 0.000944946f, 0.00124272f, 0.00150741f, 0.0017721f, 0.00203679f,
            0.00233457
        }; 

        constexpr SampledSpectrum rgb2spectrum_blue = {
            1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.99628f,
            0.983584f, 0.970669f, 0.959189f, 0.947709f, 0.936229f, 0.923313f, 0.910398f, 0.898919f, 0.869683f, 0.810917f,
            0.744299f, 0.67768f, 0.618464f, 0.559247f, 0.500031f, 0.433412f, 0.366793f, 0.317693f, 0.282429f, 0.247246f,
            0.207665f, 0.168085f, 0.132902f, 0.0977197f, 0.062537f, 0.0229565f, 0.0000132998f, 0.0f, 0.0f, 0.0f, 
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0000134134f, 0.0000485177f, 0.0000802805f, 0.000112043f, 0.000143806f,
            0.000179539f, 0.000215273f, 0.000247036f, 0.000278798f, 0.00164274f, 0.00594942f, 0.01031f, 0.0141861f, 0.0180622f, 0.0219383f,
            0.0262989f, 0.0306595f, 0.0345356f, 0.0373546f, 0.0385777f, 0.0399356f, 0.0412934f, 0.0425004f, 0.0437074f, 0.0449144f,
            0.0462723f, 0.0476301f, 0.0483596f, 0.0484989f, 0.0486365f, 0.0487914f, 0.0489462f, 0.0490839f, 0.0492215f, 0.0493591f,
            0.049514
        };
    }

#ifndef __CUDACC__
    /* Stream function of spectrum classes */
    HOST inline std::ostream& operator<<(std::ostream& out, const SampledSpectrum& spd)
    {
        for (int i = 0; i < SampledSpectrum::nSamples; i++)
            out << spd[i] << ' ';
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
    HOSTDEVICE INLINE float gauss(const float& x, const float& mu, const float& sigma1, const float& sigma2)
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

    HOSTDEVICE INLINE float averageSpectrumSamples(const float* lambda, const float* v, int n, const float& lambda_start, const float& lambda_end)
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

    HOSTDEVICE INLINE SampledSpectrum rgb2spectrum(const Vec3f& rgb)
    {
        SampledSpectrum ret = SampledSpectrum::zero();
        const float r = rgb[0];
        const float g = rgb[1];
        const float b = rgb[2];

        if (r <= g && r <= b)
        {
            ret += constants::rgb2spectrum_white * r;
            if (g <= b)
            {
                ret += constants::rgb2spectrum_cyan * (g - r);
                ret += constants::rgb2spectrum_blue * (b - g);
            }
            else
            {
                ret += constants::rgb2spectrum_cyan * (b - r);
                ret += constants::rgb2spectrum_green * (g - b);
            }
        }
        else if (g <= r && g <= b)
        {
            ret += constants::rgb2spectrum_white * g;
            if (r <= g)
            {
                ret += constants::rgb2spectrum_magenta * (r - g);
                ret += constants::rgb2spectrum_blue * (b - r);
            }
            else
            {
                ret += constants::rgb2spectrum_magenta * (b - g);
                ret += constants::rgb2spectrum_red * (r - b);
            }
        }
        else // blue <= red && blue <= green
        {
            ret += constants::rgb2spectrum_white * b;
            if (r <= g)
            {
                ret += constants::rgb2spectrum_yellow * (r - b);
                ret += constants::rgb2spectrum_green * (g - r);
            }
            else
            {
                ret += constants::rgb2spectrum_yellow * (g - b);
                ret += constants::rgb2spectrum_red * (r - g);
            }
        }
        return ret;
    }

} // namespace prayground
