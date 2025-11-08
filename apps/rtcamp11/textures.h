#pragma once


#include <prayground/core/texture.h>
#include <prayground/core/spectrum.h>
#include <prayground/math/noise.h>

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

    template <typename T>
    class StarNightTexture_ final : public Texture {
    public:
        using ColorType = T;
        struct Data
        {
            T base_color;
            RandomNoise::Data noise_data;
            float star_threshold;
            float star_intensity;
            Vec3f moon_dir;
            float moon_intensity;
        };

#ifndef __CUDACC__
        StarNightTexture_(
            const T& base_color,
            uint32_t noise_seed,
            int noise_width,
            int noise_height,
            int noise_depth,
            float star_threshold,
            float star_intensity,
            Vec3f moon_dir,
            float moon_intensity,
            int prg_id
        )
            : Texture(prg_id),
            m_base_color(base_color),
            m_seed(noise_seed),
            m_noise_width(noise_width),
            m_noise_height(noise_height),
            m_noise_depth(noise_depth),
            m_star_threshold(star_threshold),
            m_star_intensity(star_intensity),
            m_moon_dir(moon_dir),
            m_moon_intensity(moon_intensity)
        {}

        constexpr TextureType type() override
        {
            return TextureType::Custom;
        }

        void copyToDevice() override
        {
            RandomNoise::Data noise_data = {
                .seed = m_seed,
                .width = m_noise_width,
                .height = m_noise_height,
                .depth = m_noise_depth
            };

            Data data = {
                .base_color = m_base_color,
                .noise_data = noise_data,
                .star_threshold = m_star_threshold,
                .star_intensity = m_star_intensity,
                .moon_dir = m_moon_dir,
                .moon_intensity = m_moon_intensity
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
        T m_base_color;
        uint32_t m_seed;
        int m_noise_width;
        int m_noise_height;
        int m_noise_depth;
        float m_star_threshold;
        float m_star_intensity;
        Vec3f m_moon_dir;
        float m_moon_intensity;
#endif // __CUDACC__
    };

    template <typename T>
    class LeafTexture_ final : public Texture {
    public:
        using ColorType = T;
        struct Data
        {
            T base_color;
            T vein_color;
            uint32_t seed;
            float vein_density;
        };

#ifndef __CUDACC__
        LeafTexture_(
            const T& base_color,
            const T& vein_color,
            uint32_t seed,
            float vein_density,
            int prg_id
        )
            : Texture(prg_id),
            m_base_color(base_color),
            m_vein_color(vein_color),
            m_seed(seed),
            m_vein_density(vein_density)
        {}

        constexpr TextureType type() override
        {
            return TextureType::Custom;
        }

        void copyToDevice() override
        {
            Data data = {
                .base_color = m_base_color,
                .vein_color = m_vein_color,
                .seed = m_seed,
                .vein_density = m_vein_density
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
        T m_base_color;
        T m_vein_color;
        uint32_t m_seed;
        float m_vein_density;
#endif // __CUDACC__
    };

} // namespace prayground