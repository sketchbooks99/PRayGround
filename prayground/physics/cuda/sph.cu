// Smoothed Particle Hydrodynamics 

#include <prayground/physics/cuda/sph.cuh>
#include <prayground/math/util.h>
#include <stdio.h>

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
        else {
            return 0.0f;
        }
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

    extern "C" GLOBAL void computeDensity(SPHParticles::Data * particles, uint32_t num_particles, SPHConfig config)
    {
        // Global thread ID equals particle index i
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_particles) return;

        SPHParticles::Data& pi = particles[idx];

        const float h = config.kernel_size;

        for (auto j = 0; j < num_particles; j++) {
            if (j == idx) continue;

            // Reconstruct density from mass and kernel
            auto pj = particles[j];
            float r = length(pi.position - pj.position);

            // Ignore particles outside of kernel size
            if (r < h) {
                pi.density += pj.mass * particleKernel(r, h);
            }
        }
    }

    extern "C" GLOBAL void computePressure(SPHParticles::Data * particles, uint32_t num_particles, SPHConfig config)
    {
        // Global thread ID equals particle index i
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_particles) return;

        SPHParticles::Data& pi = particles[idx];
        pi.pressure = config.stiffness * (pi.density - config.rest_density);
    }

    extern "C" GLOBAL void computeForce(SPHParticles::Data * particles, uint32_t num_particles, SPHConfig config)
    {
        // Global thread ID equals particle index i
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_particles) return;

        SPHParticles::Data& pi = particles[idx];

        Vec3f pressure_force(0.0f);
        Vec3f viscosity_force(0.0f);

        const float h = config.kernel_size;

        for (auto j = 0; j < num_particles; j++) {
            if (j == idx) continue;

            auto pj = particles[j];

            Vec3f pi2pj = pj.position - pi.position;
            auto dir = normalize(pi2pj);
            float r = length(pi2pj);
            if (r < h) {
                viscosity_force += config.viscosity * (pj.mass * (pj.velocity - pi.velocity) * 2.0f * particleKernelDerivative(r, h)) / pj.density;

                pressure_force += -dir * (pj.mass * (pj.pressure + pi.pressure)) * particleKernelDerivative(r, h) / (2.0f * pj.density);
            }
        }
        //viscosity_force *= config.viscosity;

        pressure_force *= -1.0f / pi.density;

        pi.force = pressure_force + viscosity_force + config.external_force;
    }

    extern "C" DEVICE void particleCollision(SPHParticles::Data * particles, uint32_t num_particles, SPHConfig config) 
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_particles) return;

        SPHParticles::Data& pi = particles[idx];

        for (auto j = 0; j < num_particles; j++) {
            if (j == idx) continue;

            auto pj = particles[j];
        }
    }

    extern "C" GLOBAL void updateParticle(SPHParticles::Data * particles, uint32_t num_particles, SPHConfig config, AABB wall)
    {
        // Global thread ID equals particle index i
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_particles) return;

        SPHParticles::Data& pi = particles[idx];

        float kd = 20.0f;
        float ks = 20.0f;
        Vec3f pos_offset(0.0f);
        if (pi.position.x() + pi.velocity.x() * config.time_step > wall.max().x() - 2.0f * pi.radius) {
            const float dist = abs(pi.position.x() - wall.max().x());
            pi.force += Vec3f(-1.0f, 0.0f, 0.0f) * (ks * dist + kd * dot(pi.velocity, Vec3f(1.0f, 0.0f, 0.0f)));
        }
        
        if (pi.position.x() + pi.velocity.x() * config.time_step < wall.min().x() + 2.0f * pi.radius) {
            const float dist = abs(pi.position.x() - wall.min().x());
            pi.force += Vec3f(1.0f, 0.0f, 0.0f) * (ks * dist + kd * dot(pi.velocity, Vec3f(-1.0f, 0.0f, 0.0)));
        }
        
        if (pi.position.y() + pi.velocity.y() * config.time_step > wall.max().y() - 2.0f * pi.radius) {
            const float dist = abs(pi.position.y() - wall.max().y());
            pi.force += Vec3f(0.0f, -1.0f, 0.0f) * (ks * dist + kd * dot(pi.velocity, Vec3f(0.0f, 1.0f, 0.0f)));
        }
        
        if (pi.position.y() + pi.velocity.y() * config.time_step < wall.min().y() + 2.0f * pi.radius) {
            const float dist = abs(pi.position.y() - wall.min().y());
            pi.force += Vec3f(0.0f, 1.0f, 0.0f) * (ks * dist + kd * dot(pi.velocity, Vec3f(0.0f, -1.0f, 0.0f)));
        }

        if (pi.position.z() + pi.velocity.z() * config.time_step > wall.max().z() - 2.0f * pi.radius) {
            const float dist = abs(pi.position.z() - wall.max().z());
            pi.force += Vec3f(0.0f, 0.0f, -1.0f) * (ks * dist + kd * dot(pi.velocity, Vec3f(0.0f, 0.0f, 1.0f)));
        }

        if (pi.position.z() + pi.velocity.z() * config.time_step < wall.min().z() + 2.0f * pi.radius) {
            const float dist = abs(pi.position.z() - wall.min().z());
            pi.force += Vec3f(0.0f, 0.0f, 1.0f) * (ks * dist + kd * dot(pi.velocity, Vec3f(0.0f, 0.0f, -1.0f)));
        }


        // Update velocity
        pi.velocity += config.time_step * pi.force / pi.mass;

        // Update position
        pi.position += config.time_step * pi.velocity + pos_offset;
    }

    extern "C" HOST void solveSPH(SPHParticles::Data* d_particles, uint32_t num_particles, SPHConfig config, AABB wall) 
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
        updateParticle<<<block_dim, threads_per_block>>>(d_particles, num_particles, config, wall);
    }

    // Update particle's AABB buffers on device
    extern "C" GLOBAL void updateAABB(const SPHParticles::Data * particles, uint32_t num_particles, OptixAabb * out_aabbs)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_particles) return;

        SPHParticles::Data p = particles[idx];

        out_aabbs[idx] = {
            p.position.x() - p.radius,
            p.position.y() - p.radius,
            p.position.z() - p.radius,
            p.position.x() + p.radius,
            p.position.y() + p.radius,
            p.position.z() + p.radius,
        };
    }

    extern "C" HOST void updateParticleAABB(const SPHParticles::Data * particles, uint32_t num_particles, OptixAabb * out_aabbs)
    {
        constexpr int NUM_MAX_THREADS = 1024;
        constexpr int NUM_MAX_BLOCKS = 65536;

        // Determine thread size
        const int num_threads = min((int)num_particles, NUM_MAX_THREADS);
        dim3 threads_per_block(num_threads, 1);

        // Determine block size
        const int num_blocks = num_particles / num_threads + 1;
        dim3 block_dim(num_blocks, 1);
        updateAABB<<<block_dim, threads_per_block>>>(particles, num_particles, out_aabbs);
    }

} // namespace prayground