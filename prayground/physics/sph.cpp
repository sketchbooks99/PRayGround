#include "sph.h"
#include <prayground/physics/cuda/sph.cuh>

namespace prayground {

    // ------------------------------------------------------------------
    SPHParticles::SPHParticles() {

    }

    SPHParticles::SPHParticles(const std::vector<SPHParticles::Data>& particles)
    {
        // Allocate unique_ptr of particles 
        m_particles = std::make_unique<Data[]>(particles.size());
        memcpy(m_particles.get(), particles.data(), sizeof(Data) * particles.size());
        m_num_particles = static_cast<uint32_t>(particles.size());
    }

    SPHParticles::SPHParticles(Data* particles, uint32_t num_particles)
        : m_num_particles{ num_particles }
    {
        m_particles = std::make_unique<Data[]>(num_particles);
        memcpy(m_particles.get(), particles, sizeof(Data) * num_particles);
    }


    // ------------------------------------------------------------------
    constexpr ShapeType SPHParticles::type()
    {
        return ShapeType::Custom;
    }

    // ------------------------------------------------------------------
    OptixBuildInput SPHParticles::createBuildInput()
    {
        OptixBuildInput bi = {};

        std::vector<uint32_t> sbt_indices;

        uint32_t* input_flags = new uint32_t[m_num_particles];
        for (int i = 0; i < m_num_particles; i++) {
            input_flags[i] = OPTIX_GEOMETRY_FLAG_NONE;
            sbt_indices.push_back(m_sbt_index);
        }

        CUDABuffer<uint32_t> d_sbt_indices;
        d_sbt_indices.copyToDevice(sbt_indices);

        //OptixAabb aabb = static_cast<OptixAabb>(this->bound());
        //std::vector<OptixAabb> aabbs;
        //for (int i = 0; i < m_num_particles; i++) {
        //    auto p = m_particles.get()[i];
        //    aabbs.push_back(static_cast<OptixAabb>(AABB(p.position - Vec3f(p.radius), p.position + Vec3f(p.radius))));
        //}
        //CUDABuffer<OptixAabb> d_aabb;
        //d_aabb.copyToDevice(aabbs);
        //d_aabb_buffer = d_aabb.devicePtr();

        std::vector<OptixAabb> aabbs(m_num_particles);
        CUDABuffer<OptixAabb> d_aabb;
        d_aabb.copyToDevice(aabbs);

        // Copy particle buffer to device
        this->copyToDevice();
        updateParticleAABB((SPHParticles::Data*)d_data, m_num_particles, d_aabb.deviceData());
        CUDA_SYNC_CHECK();

        d_aabb_buffer = d_aabb.devicePtr();

        bi.type = static_cast<OptixBuildInputType>(type());
        bi.customPrimitiveArray.aabbBuffers = &d_aabb_buffer;
        bi.customPrimitiveArray.numPrimitives = static_cast<uint32_t>(m_num_particles);
        bi.customPrimitiveArray.flags = input_flags;
        bi.customPrimitiveArray.numSbtRecords = 1u;
        bi.customPrimitiveArray.sbtIndexOffsetBuffer = d_sbt_indices.devicePtr();
        bi.customPrimitiveArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
        bi.customPrimitiveArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);

        return bi;
    }

    // ------------------------------------------------------------------
    uint32_t SPHParticles::numPrimitives() const
    {
        return m_num_particles;
    }

    // ------------------------------------------------------------------
    void SPHParticles::copyToDevice()
    {
        if (!d_data)
            CUDA_CHECK(cudaMalloc(&d_data, sizeof(Data) * m_num_particles));
        CUDA_CHECK(cudaMemcpy(d_data, m_particles.get(), sizeof(Data) * m_num_particles, cudaMemcpyHostToDevice));
    }

    // ------------------------------------------------------------------
    void SPHParticles::free()
    {
        Shape::free();
        cuda_free(d_data);
    }

    // ------------------------------------------------------------------
    AABB SPHParticles::bound() const
    {
        AABB aabb;
        for (uint32_t i = 0; i < m_num_particles; i++) {
            auto p = m_particles.get()[i];
            aabb = AABB::merge(aabb, AABB(p.position - p.radius, p.position + p.radius));
        }
        return aabb;
    }

} // namespace prayground