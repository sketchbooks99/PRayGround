#include "sph.h"

namespace prayground {

    // ------------------------------------------------------------------
    SPHParticle::SPHParticle() 
        : m_position(Vec3f(0.0f)), m_velocity(Vec3f(0.0f)), m_mass(1.0f), m_radius(0.1f)
    {}

    SPHParticle::SPHParticle(Vec3f position, Vec3f velocity, float mass, float radius)
        : m_position(position), m_velocity(velocity), m_mass(mass), m_radius(radius)
    {}

    // ------------------------------------------------------------------
    constexpr ShapeType SPHParticle::type()
    {
        return ShapeType::Custom;
    }

    // ------------------------------------------------------------------
    OptixBuildInput SPHParticle::createBuildInput()
    {
        if (d_data) cuda_free(d_data);
        return createSingleCustomBuildInput(d_aabb_buffer, this->bound(), m_sbt_index);
    }

    // ------------------------------------------------------------------
    uint32_t SPHParticle::numPrimitives() const
    {
        return 1u;
    }

    // ------------------------------------------------------------------
    void SPHParticle::copyToDevice()
    {
        Data data = this->getData();
        if (!d_data)
            CUDA_CHECK(cudaMalloc(&d_data, sizeof(Data)));
        CUDA_CHECK(cudaMemcpy(
            d_data, &data, sizeof(Data), cudaMemcpyHostToDevice
        ));
    }

    // ------------------------------------------------------------------
    void SPHParticle::free()
    {
        Shape::free();
        cuda_free(d_data);
    }

    // ------------------------------------------------------------------
    AABB SPHParticle::bound() const
    {
        return AABB(m_position - Vec3f(m_radius), m_position + Vec3f(m_radius));
    }

    // ------------------------------------------------------------------
    SPHParticle::Data SPHParticle::getData() const
    {
        return { m_position, m_velocity, m_mass, m_radius, 0.0f, 0.0f, Vec3f(0.0f) };
    }

} // namespace prayground