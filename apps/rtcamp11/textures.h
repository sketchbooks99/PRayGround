#pragma once


#include <prayground/core/texture.h>
#include <prayground/core/spectrum.h>

namespace prayground {

    template <typename T>
    class ProceduralWoodenTexture_ final : public Texture {
    public:
        using ColorType = T;
        struct Data
        {
            float scale;
            float frequency;
            float noise_amplitude;
            T light_wood_color;
            T dark_wood_color;
        };

#ifndef __CUDACC__
        ProceduralWoodenTexture_(
            const T& light_wood_color, const T& dark_wood_color, 
            float scale, float frequency, float noise_amplitude, int prg_id
        )
            : Texture(prg_id), 
            m_light_wood_color(light_wood_color), m_dark_wood_color(dark_wood_color),
            m_scale(scale), m_frequency(frequency), m_noise_amplitude(noise_amplitude)
        {}

        constexpr TextureType type() override
        {
            return TextureType::Custom;
        }

        ColorType eval(const Vec2f& texcoord) const
        {
            return m_light_wood_color;
        }

        void setLightWoodColor(const T& color)
        {
            m_light_wood_color = color;
        }
        const T& lightWoodColor() const
        {
            return m_light_wood_color;
        }

        void setDarkWoodColor(const T& color)
        {
            m_dark_wood_color = color;
        }
        const T& darkWoodColor() const
        {
            return m_dark_wood_color;
        }

        void setScale(float scale)
        {
            m_scale = scale;
        }
        float scale() const
        {
            return m_scale;
        }

        void setFrequency(float frequency)
        {
            m_frequency = frequency;
        }
        float frequency() const
        {
            return m_frequency;
        }

        void setNoiseAmplitude(float amplitude)
        {
            m_noise_amplitude = amplitude;
        }
        float noiseAmplitude() const
        {
            return m_noise_amplitude;
        }

        void copyToDevice() override
        {
            Data data = {
                .scale = m_scale,
                .frequency = m_frequency,
                .noise_amplitude = m_noise_amplitude,
                .light_wood_color = m_light_wood_color,
                .dark_wood_color = m_dark_wood_color
            };

            if (!d_data)
                 CUDA_CHECK(cudaMalloc(&d_data, sizeof(Data)));
            CUDA_CHECK(cudaMemcpy(
                d_data,
                &data, sizeof(Data),
                cudaMemcpyHostToDevice
            ));
        }

    private:
        T m_light_wood_color;
        T m_dark_wood_color;
        float m_scale;
        float m_frequency;
        float m_noise_amplitude;
#endif // __CUDACC__
    };

} // namespace prayground