// Smoothed Particle Hydrodynamics 

#include <prayground/physics/sph.h>
#include <prayground/math/util.h>

namespace prayground {

    DEVIVE float cubicSpline(float q)
    {
        if (0.0f <= q <= 0.5f)
            return 6.0f * (pow3(q) - pow2(q)) + 1.0f;
        else if (0.5f < q <= 1.0f)
            return 2.0f * pow3(1.0f - q);
        else
            return 0.0f;
    }

    DEVICE float cubicSplineDerivative(float q)
    {
        if (0.0f <= q <= 0.5f)
            return 6.0f * (3.0f * pow2(q) - 2.0f * q);
        else if (0.5f < q <= 1.0f)
            return -6.0f * pow2(1.0f - q);
        else
            return 0.0f;
    }

    DEVICE float particleKernel(float r, float kernel_size)
    {
        auto q = r / kernel_size;
        auto norm_factor = 8.0f / (math::pi * pow3(kernel_size));
        return norm_factor * cubicSpline(q);
    }

    DEVICE float particleKernelDerivative(float r, float kernel_size)
    {
        auto q = r / kernel_size;
        auto norm_factor = 8.0f / (math::pi * pow3(kernel_size));
        return norm_factor * cubicSplineDerivative(q);
    }

    extern "C" GLOBAL void reconstructRho(
        SPHParticle* particles, uint32_t num_particles, float kernel_size) 
    {
        // Global thread ID equals particle index i
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_particles) return;

        auto pi = particles[idx];

        for (auto j = 0; j < num_particles; j++) {
            if (j == idx) continue;

            // Reconstruct rho from mass and kernel
            auto pj = particles[j];
            float r = length(pi.position - pj.position);

            // Ignore particles outside of kernel size
            if (r > kernel_size) continue;

            pi.rho += pj.mass * particleKernel(r, kernel_size);
        }
    }

    extern "C" GLOBAL void computeViscosity(
        SPHParticle* particles, uint32_t num_particles, float kernel_size, float time_step)
    {
        // Global thread ID equals particle index i
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_particles) return;

        auto pi = particles[idx];

        for (auto j = 0; j < num_particles; j++) {
            if (j == idx) continue;

            auto pj = particles[j];

            // Second-derivatives of vectorial field
            float dd_field = 0.0f;
            for (auto k = 0; k < num_particles; k++) {
                if (k == idx) continue;

                auto pj = particles[j];
                float r = length(pi.position - pj.position);

                dd_field += (pi.mass / pj.rho) * ((2.0f * particleKernelDerivative(r, kernel_size)) / r);
            }
            dd_field *= -1.0f;

            // Compute viscosity force
            auto viscosity_force = pi.mass * pi.velocity * dd_field;

            // Update particle velocity
            pi.velocity += (time_step / pi.mass) * (viscosity_force /* + external_force */);
        }
    }

    extern "C" HOST void solveSPH(SPHParticle* d_particles, uint32_t num_particles, float kernel_size, float time_step) 
    {
        constexpr int NUM_MAX_THREADS = 1024;
        constexpr int NUM_MAX_BLOCKS = 65536;

        // Determine thread size
        const int num_threads = min(num_particles, NUM_MAX_THREADS);
        dim3 threads_per_block(num_threads, 1);

        // Determine block size
        const int num_blocks = num_particles / num_threads + 1;
        dim3 block_dim(num_blocks, 1);

        reconstructRho<<<block_dim, threads_per_block>>>(d_particles, num_particles, kernel_size, time_step);
    }

} // namespace prayground