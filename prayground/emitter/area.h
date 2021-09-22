#pragma once

#ifndef __CUDACC__
#include <memory>
#include <prayground/core/texture.h>
#include <prayground/core/emitter.h>
#include <prayground/core/interaction.h>
#endif 

namespace prayground {

struct AreaEmitterData {
    void* tex_data;
    float intensity;
    bool twosided;
    unsigned int tex_program_id;
};

#ifndef __CUDACC__

class AreaEmitter final : public Emitter {
public:
    AreaEmitter(const std::shared_ptr<Texture>& texture, float intensity = 1.0f, bool twosided = true);

    SurfaceType surfaceType() const;

    void copyToDevice() override;
    void free() override;

    EmitterType type() const override { return EmitterType::Area; }
    std::shared_ptr<Texture> texture() const;
private:
    std::shared_ptr<Texture> m_texture;
    float m_intensity;
    bool m_twosided;
};

#endif // __CUDACC__

}