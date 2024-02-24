#pragma once 

#include <prayground/math/vec.h>
#include <prayground/optix/macros.h>

namespace prayground {

    struct SPHParticle {
        Vec3f position;
        Vec3f velocity;
        float mass;

        /* Reconstructed from position and kernel function */
        float rho;
    };

    // Entry point for SPH simulation on CUDA
    extern "C" HOST void solveSPH(SPHParticle* d_particles, uint32_t num_particles, float kernel_size, float time_step);

} // namespace prayground