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
    AreaEmitter(Shape* shape, Texture* texture, float intensity) 
    : m_shape(std::make_shared<Shape>(shape))
    , m_texture(std::make_shared<Texture>(texture))
    , m_intensity(intensity) {}

    AreaEmitter(const std::shared_ptr<Shape>& shape, const std::shared_ptr<Texture>& texture, float intensity)
    : m_shape(shape), m_texture(texture), m_intensity(intensity) {}

    void prepareData() override
    {
        m_texture->prepareData();

        AreaEmitterData data = {
            m_texture->devicePtr(),
            m_intensity,
            
        }
    }

    EmitterType type() const { return EmitterType::Area; }
private:
    std::shared_ptr<Shape> m_shape;
    std::shared_ptr<Texture> m_texture;
    float m_intensity;
    bool m_twosided;
};

}