#pragma once 

#include <prayground/math/vec.h>
#include <prayground/optix/macros.h>

namespace prayground {

    struct SPHParticle {
        Vec3f position;
        Vec3f velocity;
        float mass;
        float pressure;

        /* Reconstructed from position and kernel function */
        float density;

        /* F_pressure + F_viscosity + F_external */
        Vec3f force;
    };

    struct SPHConfig {
        float kernel_size;      // h
        float rest_density;     // rho0
        Vec3f external_force;   // f_ext
        float time_step;        // dt
        float stiffness;        // k 
    };

    // Entry point for SPH simulation on CUDA
    extern "C" HOST void solveSPH(
        SPHParticle* d_particles,   // Device pointer to particles
        uint32_t num_particles, 
        SPHConfig config
    );

} // namespace prayground