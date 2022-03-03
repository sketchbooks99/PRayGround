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

class EnvironmentEmitter final : public Emitter {
public:
    struct Data {
        Texture::Data tex_data;
    };

#ifndef __CUDACC__
    EnvironmentEmitter() = default;
    EnvironmentEmitter(const std::shared_ptr<Texture>& texture)
    : m_texture(texture) {}

    void copyToDevice() override;

    void free() override {}

    EmitterType type() const override { return EmitterType::Envmap; }
    std::shared_ptr<Texture> texture() const { return m_texture; }

    Data getData() const;
private:
    std::shared_ptr<Texture> m_texture;

#endif
};


} // ::prayground