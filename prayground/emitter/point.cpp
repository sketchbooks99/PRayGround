#include "point.h"

namespace prayground {
    PointLight::PointLight(const Vec3f& position, const Vec3f& color, float intensity)
        : m_position(position), m_color(color), m_intensity(intensity)
    {
    }

    void PointLight::copyToDevice()
    {
        auto data = this->getData();

        if (!d_data)
            CUDA_CHECK(cudaMalloc(&d_data, sizeof(Data)));

        CUDA_CHECK(cudaMemcpy( d_data, &data, sizeof(Data), cudaMemcpyHostToDevice ));
    }

    PointLight::Data PointLight::getData() const
    {
        return { m_position, m_color, m_intensity };
    }

} // namespace prayground