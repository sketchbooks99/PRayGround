#pragma once

#include "../core/util.h"
#include "../core/shape.h"
#include "../core/texture.h"
#include "../core/emitter.h"

namespace oprt {

struct AreaEmitterData {
    void* texdata;
    float intensity;
    bool twosided;
    unsigned int tex_func_id;
};

class AreaEmitter final : public Emitter {
public:
    AreaEmitter(
        const std::shared_ptr<Shape>& shape, 
        const std::shared_ptr<Texture>& texture, 
        float intensity,
        bool twosided = true
    );

    void prepareData() override;

    EmitterType type() const { return EmitterType::Area; }
private:
    std::shared_ptr<Shape> m_shape;
    std::shared_ptr<Texture> m_texture;
    float m_intensity;
    bool m_twosided;
};

}