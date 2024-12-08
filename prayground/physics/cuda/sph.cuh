#include <prayground/physics/sph.h>

namespace prayground {
    // Entry point for SPH simulation on CUDA
    extern "C" HOST void solveSPH(
        SPHParticles::Data* d_particles,   // Device pointer to particles
        uint32_t num_particles, 
        SPHConfig config, 
        AABB wall = AABB(Vec3f(-10000), Vec3f(10000))
    );

    extern "C" HOST void updateParticleAABB(
        const SPHParticles::Data * particles,
        uint32_t num_particles,
        OptixAabb * out_aabbs
    );
} // namespace prayground