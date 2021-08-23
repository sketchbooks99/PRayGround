#pragma once

#include <prayground/core/emitter.h>
#include <prayground/texture/constant.h>

#ifndef __CUDACC__
    #include <filesystem>
#endif

/**
 * @brief Environment emitter. In general, emittance is evaluated at a miss program.
 * 
 * @note 
 * - If you'd like to render with image based lighting, you must be use latitude-longitude format (sphere map).
 * - EnvironmentEmitter allows ordinary textures such as checker or constant.
 */

namespace prayground {

struct EnvironmentEmitterData {
    void* tex_data;
    unsigned int tex_program_id;
};

#ifndef __CUDACC__

class EnvironmentEmitter final : public Emitter {
public:
    explicit EnvironmentEmitter(const std::shared_ptr<Texture>& texture)
    : m_texture(texture) {}

    void copyToDevice() override;

    void free() override {}

    EmitterType type() const override { return EmitterType::Envmap; }
    std::shared_ptr<Texture> texture() const { return m_texture; }
private:
    std::shared_ptr<Texture> m_texture;
};

#endif

}