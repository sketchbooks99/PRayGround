#pragma once

#ifndef __CUDACC__
#include <memory>
#endif 

#include <prayground/core/texture.h>
#include <prayground/core/emitter.h>
#include <prayground/core/interaction.h>

namespace prayground {

    class AreaEmitter final : public Emitter {
    public:
        struct Data {
            Texture::Data texture;
            float intensity;
            bool twosided;
        };
#ifndef __CUDACC__
        AreaEmitter(const SurfaceCallableID& surface_callable_id, const std::shared_ptr<Texture>& texture, float intensity = 1.0f, bool twosided = true);

        SurfaceType surfaceType() const;

        void copyToDevice() override;
        void free() override;

        EmitterType type() const override { return EmitterType::Area; }
        
        void setTexture(const std::shared_ptr<Texture>& texture);
        std::shared_ptr<Texture> texture() const;

        void setIntensity(float intensity);
        float intensity() const;

        Data getData() const;
    private:
        SurfaceCallableID m_surface_callable_id;
        std::shared_ptr<Texture> m_texture;
        float m_intensity;
        bool m_twosided;
#endif
    };

} // namespace prayground