#include <prayground/physics/sph.h>

namespace prayground {
    // Entry point for SPH simulation on CUDA
    extern "C" HOST void solveSPH(
        SPHParticle::Data* d_particles,   // Device pointer to particles
        uint32_t num_particles, 
        SPHConfig config
    );
} // namespace prayground