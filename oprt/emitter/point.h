#pragma once

#include "../core/emitter.h"

namespace oprt {

class PointEmitter : public Emitter 
{
public:
    PointEmitter(const float3& c, float intensity)
    : m_color(c), m_intensity(intensity) {}

    void prepareData() override
    {

    }

    EmitterType type() const override { return EmitterType::Point; }
private:
    float3 m_color;
    float m_intensity;
};

}