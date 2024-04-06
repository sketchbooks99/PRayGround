#pragma once 

#ifndef __CUDACC__
#include <prayground/core/cudabuffer.h>
#endif

#include <prayground/math/vec.h>
#include <prayground/optix/macros.h>
#include <prayground/core/shape.h>

namespace prayground {

    class SPHParticles : public Shape {
    public:
        struct Data {
            Vec3f position;
            Vec3f velocity;
            float mass;
            float radius;

            /* Calculated by kernel function */
            float pressure;

            /* Reconstructed from position and kernel function */
            float density;

            /* F_pressure + F_viscosity + F_external */
            Vec3f force;
        };

#ifndef __CUDACC__

        SPHParticles();
        SPHParticles(const std::vector<Data>& particles);
        SPHParticles(Data* particles, uint32_t num_particles);

        constexpr ShapeType type() override;

        OptixBuildInput createBuildInput() override;

        uint32_t numPrimitives() const override;

        void copyToDevice() override;
        void free() override;

        AABB bound() const override;

    private:
        std::unique_ptr<Data[]> m_particles;
        uint32_t m_num_particles{ 0 };

        CUdeviceptr d_aabb_buffer{ 0 };
#endif // __CUDACC__
    };

    struct SPHConfig {
        float kernel_size;      // h
        float rest_density;     // rho0
        Vec3f external_force;   // f_ext
        float time_step;        // dt
        float stiffness;        // k 
    };

} // namespace prayground