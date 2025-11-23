#pragma once

#include <prayground/core/texture.h>

namespace prayground {

    class Thinfilm {
    public:
        struct Data {
            // Fresnel
            Vec3f ior;

            // Thin film
            Texture::Data thickness;
            float thickness_scale;
            float tf_ior;
            Vec3f extinction;
        };

#ifndef __CUDACC__
        Thinfilm() = default;

        Thinfilm(Vec3f ior, const std::shared_ptr<Texture>& thickness, float thickness_scale, float tf_ior, Vec3f extinction)
            : m_ior(ior), m_thickness(thickness), m_thickness_scale(thickness_scale), m_tf_ior(tf_ior), m_extinction(extinction) {}

        void copyToDevice() {
            if (m_thickness != nullptr && !m_thickness->devicePtr())
                m_thickness->copyToDevice();
        }

        void free() {
            if (m_thickness != nullptr && m_thickness->devicePtr())
                m_thickness->free();
        }

        Data getData() const {
            Texture::Data thickness_data = m_thickness != nullptr ? m_thickness->getData() : Texture::Data{nullptr, -1};
            return { m_ior, thickness_data, m_thickness_scale, m_tf_ior, m_extinction };
        }

        void setIor(Vec3f ior) {
            m_ior = ior;
        }
        Vec3f ior() const {
            return m_ior;
        }

        void setThickness(const std::shared_ptr<Texture>& thickness) {
            m_thickness = thickness;
        }
        std::shared_ptr<Texture> thickness() const {
            return m_thickness;
        }

        void setThicknessScale(float thickness_scale) {
            m_thickness_scale = thickness_scale;
        }
        float thicknessScale() const {
            return m_thickness_scale;
        }

        void setTfIor(float tf_ior) {
            m_tf_ior = tf_ior;
        }
        float tfIor() const {
            return m_tf_ior;
        }

        void setExtinction(Vec3f extinction) {
            m_extinction = extinction;
        }
        Vec3f extinction() const {
            return m_extinction;
        }

    private:
        Vec3f m_ior;
        std::shared_ptr<Texture> m_thickness;
        float m_thickness_scale;
        float m_tf_ior;
        Vec3f m_extinction;
#endif
    };

} // namespace prayground