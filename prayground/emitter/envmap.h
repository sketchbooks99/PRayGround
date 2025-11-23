#pragma once

#include <prayground/core/emitter.h>
#include <prayground/core/texture.h>

#ifndef __CUDACC__
    #include <filesystem>
#endif

/**
 * @brief Environment emitter. In general, emittance is evaluated by a miss program.
 */

namespace prayground {

    struct EnvmapSampleData {
        float* d_conditional_cdf; // [width * height]
        float* d_row_luminance;   // [height]
        float* d_marginal_cdf;    // [height]

        // Texture data
        Texture::Data texture;
        // Resolution
        uint32_t width;
        uint32_t height;
        // Total luminance
        float total_luminance;
    };

    class EnvironmentEmitter final : public Emitter {
    public:
        struct Data {
            Texture::Data texture;
        };

#ifndef __CUDACC__
        EnvironmentEmitter() = default;
        EnvironmentEmitter(const std::shared_ptr<Texture>& texture)
        : m_texture(texture) {}

        void copyToDevice() override;

        EmitterType type() const override { return EmitterType::Envmap; }
        void setTexture(const std::shared_ptr<Texture>& texture) { m_texture = texture; }
        std::shared_ptr<Texture> texture() const { return m_texture; }

        Data getData() const;
    private:
        std::shared_ptr<Texture> m_texture;

#endif
    };

    EnvmapSampleData createEnvmapSampleDataOnDevice(Vec4f* envmap_data, uint32_t width, uint32_t height);

} // namespace prayground