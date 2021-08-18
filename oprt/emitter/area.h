#pragma once

#include <memory>
#include <oprt/core/texture.h>
#include <oprt/core/emitter.h>
#include <oprt/texture/constant.h>
#include "cuda/area.cuh"

namespace oprt {

#ifndef __CUDACC__

class AreaEmitter final : public Emitter {
public:
    AreaEmitter(const float3& color, float intensity=1.0f, bool twosided=true);
    AreaEmitter(const std::shared_ptr<Texture>& texture, float intensity = 1.0f, bool twosided = true);

    void prepareData() override;
    void freeData() override;

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