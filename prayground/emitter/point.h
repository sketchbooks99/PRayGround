#pragma once

#include <prayground/core/emitter.h>
#include <prayground/math/vec.h>

namespace prayground {

    class PointLight : public Emitter
    {
    public:
        struct Data {
            Vec3f position;
            Vec3f color;
            float intensity;
        };
#ifndef __CUDACC__
        PointLight(const Vec3f& position, const Vec3f& color, float intensity);

        EmitterType type() const override { return EmitterType::Point; }

        void copyToDevice() override;
        void free() override;

        Data getData() const;

        void setPosition(const Vec3f& position) { m_position = position; }
        const Vec3f& position() const { return m_position; }

        void setColor(const Vec3f& color) { m_color = color; }
        const Vec3f& color() const { return m_color; }

        void setIntensity(float intensity) { m_intensity = intensity; }
        const float intensity() const { return m_intensity; }
    private:
        Vec3f m_position;
        Vec3f m_color;
        float m_intensity;
#endif // __CUDACC__
    };
    
} // namespace prayground