#include "textures.cuh"
#include <cstdio>

namespace prayground {

    __global__ void bakeRockTextureKernel(
        PerlinNoiseData* d_perlin,
        RockTexture rock,
        uint32_t width,
        uint32_t height,
        float* output_buffer
    ) {
        const uint32_t ix = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t iy = blockIdx.y * blockDim.y + threadIdx.y;

        if (ix >= width || iy >= height) return;

        float u = (ix + 0.5f) / static_cast<float>(width);
        float v = (iy + 0.5f) / static_cast<float>(height);
        Vec3f p(u, v, 1.0f);

        float noise1 = d_perlin->turb(p * rock.noise1.scale, rock.noise1.depth);
        float noise2 = d_perlin->turb(p * rock.noise2.scale, rock.noise2.depth);
        
        float rock_value = rock.noise1.amplitude * noise1
                         + rock.noise2.amplitude * noise2;
        output_buffer[iy * width + ix] = rock_value;
    }

    extern "C" float* bakeRockTexture(
        uint32_t seed,
        RockTexture rock_data,
        uint32_t width,
        uint32_t height
    ) {
        // Generate Perlin Noise data on host using existing PerlinNoise class
        PerlinNoise host_perlin(seed);
        
        // Create host-side PerlinNoiseData structure
        PerlinNoiseData h_perlin_data;
        
        // Copy permutation tables (we need to extract from PerlinNoise)
        // Since PerlinNoise uses pointers, we'll generate fresh data
        unsigned int local_seed = seed;
        
        // Generate random vectors
        for (int i = 0; i < PerlinNoiseData::POINT_COUNT; i++) {
            const Vec3f rnd_v = Vec3f(rnd(local_seed), rnd(local_seed), rnd(local_seed)) * 2.0f - 1.0f;
            h_perlin_data.rnd_vec[i] = normalize(rnd_v);
        }
        
        // Generate permutation tables
        auto generatePerm = [&](int* perm) {
            for (int i = 0; i < PerlinNoiseData::POINT_COUNT; i++)
                perm[i] = i;
            
            // Permute
            for (int i = PerlinNoiseData::POINT_COUNT - 1; i > 0; i--) {
                int target = rndInt(local_seed, 0, i);
                int tmp = perm[i];
                perm[i] = perm[target];
                perm[target] = tmp;
            }
        };
        
        generatePerm(h_perlin_data.perm_x);
        generatePerm(h_perlin_data.perm_y);
        generatePerm(h_perlin_data.perm_z);
        
        // Copy PerlinNoiseData to device
        PerlinNoiseData* d_perlin_data;
        cudaError_t err = cudaMalloc(&d_perlin_data, sizeof(PerlinNoiseData));
        if (err != cudaSuccess) {
            printf("[ERROR] cudaMalloc failed for PerlinNoiseData: %s\n", cudaGetErrorString(err));
            return nullptr;
        }
        
        err = cudaMemcpy(d_perlin_data, &h_perlin_data, sizeof(PerlinNoiseData), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("[ERROR] cudaMemcpy failed for PerlinNoiseData: %s\n", cudaGetErrorString(err));
            cudaFree(d_perlin_data);
            return nullptr;
        }
        
        // Allocate texture buffer
        float* d_texture;
        size_t texture_size = width * height * sizeof(float);
        err = cudaMalloc(&d_texture, texture_size);
        if (err != cudaSuccess) {
            printf("[ERROR] cudaMalloc failed for rock texture (%zu bytes): %s\n", 
                   texture_size, cudaGetErrorString(err));
            cudaFree(d_perlin_data);
            return nullptr;
        }

        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                      (height + blockSize.y - 1) / blockSize.y);

        bakeRockTextureKernel<<<gridSize, blockSize>>>(
            d_perlin_data, rock_data, width, height, d_texture
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("[ERROR] bakeRockTextureKernel launch failed: %s\n", 
                   cudaGetErrorString(err));
            cudaFree(d_texture);
            cudaFree(d_perlin_data);
            return nullptr;
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("[ERROR] bakeRockTextureKernel execution failed: %s\n", 
                   cudaGetErrorString(err));
            cudaFree(d_texture);
            cudaFree(d_perlin_data);
            return nullptr;
        }

        // Clean up Perlin data
        cudaFree(d_perlin_data);
        
        return d_texture;
    }

    __global__ void generateBumpMapKernel(
        float* d_heightmap,
        uint32_t width,
        uint32_t height,
        float bump_strength,
        Vec4f* d_bumpmap
    ) {
        const uint32_t ix = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t iy = blockIdx.y * blockDim.y + threadIdx.y;

        if (ix >= width || iy >= height) return;

        float hL = (ix > 0) ? d_heightmap[iy * width + (ix - 1)] : d_heightmap[iy * width + ix];
        float hR = (ix < width - 1) ? d_heightmap[iy * width + (ix + 1)] : d_heightmap[iy * width + ix];
        float hD = (iy > 0) ? d_heightmap[(iy - 1) * width + ix] : d_heightmap[iy * width + ix];
        float hU = (iy < height - 1) ? d_heightmap[(iy + 1) * width + ix] : d_heightmap[iy * width + ix];

        // Create tangent vectors from height differences
        // Tangent along X axis: (2, hR - hL, 0)  (2 pixels apart in X)
        // Tangent along Z axis: (0, hU - hD, 2)  (2 pixels apart in Z)
        float h_size = 1.0f / (float)width;
        float v_size = 1.0f / (float)height;
        Vec3f tangent_x = normalize(Vec3f(h_size, 0.0f, (hR - hL) * bump_strength));
        Vec3f tangent_y = normalize(Vec3f(0.0f, v_size, (hU - hD) * bump_strength));
        
        // Normal is cross product of tangents
        Vec3f normal = normalize(cross(tangent_x, tangent_y));

        if (length(normal) == 0.0f) {
            printf("Normal is zero at pixel (%u, %u)\n", ix, iy);
            printf("hL: %f, hR: %f, hD: %f, hU: %f\n", hL, hR, hD, hU);
            printf("tangent_x: %f %f %f\n", tangent_x.x(), tangent_x.y(), tangent_x.z());
            printf("tangent_y: %f %f %f\n", tangent_y.x(), tangent_y.y(), tangent_y.z());
        }

        d_bumpmap[iy * width + ix] = Vec4f(normal.x(), normal.y(), normal.z(), 1.0f);
    }

    extern "C" Vec4f* createBumpTextureFromHeightmap(
        float* d_heightmap,
        uint32_t width,
        uint32_t height,
        float bump_strength
    ) {
        Vec4f* d_bumpmap;
        size_t bumpmap_size = width * height * sizeof(Vec4f);
        cudaError_t err = cudaMalloc(&d_bumpmap, bumpmap_size);
        if (err != cudaSuccess) {
            printf("[ERROR] cudaMalloc failed for bumpmap (%zu bytes): %s\n", 
                   bumpmap_size, cudaGetErrorString(err));
            return nullptr;
        }

        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                      (height + blockSize.y - 1) / blockSize.y);

        generateBumpMapKernel<<<gridSize, blockSize>>>(
            d_heightmap, width, height, bump_strength, d_bumpmap
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("[ERROR] generateBumpMapKernel launch failed: %s\n", 
                   cudaGetErrorString(err));
            cudaFree(d_bumpmap);
            return nullptr;
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("[ERROR] generateBumpMapKernel execution failed: %s\n", 
                   cudaGetErrorString(err));
            cudaFree(d_bumpmap);
            return nullptr;
        }

        return d_bumpmap;
    }
}