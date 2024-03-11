// Smoothed Particle Hydrodynamics 

#include <prayground/physics/cuda/sph.cuh>
#include <prayground/math/util.h>

namespace prayground {

    DEVICE float cubicSpline(float q)
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

    extern "C" GLOBAL void computeDensity(SPHParticle::Data* particles, uint32_t num_particles, SPHConfig config) 
    {
        // Global thread ID equals particle index i
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_particles) return;

        SPHParticle::Data& pi = particles[idx];

        const float h = config.kernel_size;

        for (auto j = 0; j < num_particles; j++) {
            if (j == idx) continue;

            // Reconstruct density from mass and kernel
            auto pj = particles[j];
            float r = length(pi.position - pj.position);

            // Ignore particles outside of kernel size
            if (r > h) continue;

            pi.density += pj.mass * particleKernel(r, h);
        }
    }

    extern "C" GLOBAL void computePressure(SPHParticle::Data* particles, uint32_t num_particles, SPHConfig config)
    {
        // Global thread ID equals particle index i
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_particles) return;

        SPHParticle::Data& pi = particles[idx];
        pi.pressure = config.stiffness * (pi.density - config.rest_density);
    }

    extern "C" GLOBAL void computeForce(SPHParticle::Data* particles, uint32_t num_particles, SPHConfig config)
    {
        // Global thread ID equals particle index i
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_particles) return;

        SPHParticle::Data& pi = particles[idx];

        Vec3f pressure_force(0.0f);
        Vec3f viscosity_force(0.0f);

        const float h = config.kernel_size;




        for (auto j = 0; j < num_particles; j++) {
            if (j == idx) continue;

            auto pj = particles[j];

            Vec3f pi2pj = pj.position - pi.position;
            float r = length(pi2pj);
            if (r > h) continue;

            viscosity_force += (pj.mass * (pi.velocity - pi.velocity) * 2.0f * particleKernelDerivative(r, h)) / (pj.density * r);

            pressure_force += -pi2pj * pj.mass * (pi.pressure / pow2(pi.density) + (pj.pressure / pow2(pj.density))) * particleKernelDerivative(r, h);
        }
        // Compute viscosity force
        viscosity_force *= -1.0f * pi.mass * pi.velocity;

        // Compute pressure force
        pressure_force *= -1.0f;

        pi.force = pressure_force + viscosity_force + config.external_force;
    }

    extern "C" GLOBAL void updateParticle(SPHParticle::Data* particles, uint32_t num_particles, SPHConfig config)
    {
        // Global thread ID equals particle index i
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_particles) return;

        SPHParticle::Data& pi = particles[idx];

        // Update velocity
        pi.velocity += config.time_step * pi.force / pi.mass;

        // Update position
        pi.position += config.time_step * pi.velocity;
    }

    extern "C" HOST void solveSPH(SPHParticle::Data* d_particles, uint32_t num_particles, SPHConfig config) 
    {
        constexpr int NUM_MAX_THREADS = 1024;
        constexpr int NUM_MAX_BLOCKS = 65536;

        // Determine thread size
        const int num_threads = min((int)num_particles, NUM_MAX_THREADS);
        dim3 threads_per_block(num_threads, 1);

        // Determine block size
        const int num_blocks = num_particles / num_threads + 1;
        dim3 block_dim(num_blocks, 1);

        computeDensity<<<block_dim, threads_per_block>>>(d_particles, num_particles, config);
        computePressure<<<block_dim, threads_per_block>>>(d_particles, num_particles, config);
        computeForce<<<block_dim, threads_per_block>>>(d_particles, num_particles, config);
        updateParticle<<<block_dim, threads_per_block>>>(d_particles, num_particles, config);
    }

} // namespace prayground