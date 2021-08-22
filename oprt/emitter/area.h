#pragma once

#ifndef __CUDACC__
#include <memory>
#include <oprt/core/texture.h>
#include <oprt/core/emitter.h>
#include <oprt/texture/constant.h>
#endif 

namespace oprt {

struct AreaEmitterData {
    void* tex_data;
    float strength;
    bool twosided;
    unsigned int tex_program_id;
};

#ifndef __CUDACC__

class AreaEmitter final : public Emitter {
public:
    AreaEmitter(const float3& color, float intensity=1.0f, bool twosided=true);
    AreaEmitter(const std::shared_ptr<Texture>& texture, float intensity = 1.0f, bool twosided = true);

    void copyToDevice() override;
    void free() override;

    void setProgramId(const int32_t prg_id);
    int32_t programId() const;

    EmitterType type() const override { return EmitterType::Area; }
private:
    std::shared_ptr<Texture> m_texture;
    float m_intensity;
    bool m_twosided;
    int32_t m_prg_id { -1 };
    
};

#endif // __CUDACC__

}