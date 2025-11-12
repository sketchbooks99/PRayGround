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
    
    // ===== Tree Bark Texture Generation =====
    
    // Worley/Cellular noise (2D) with anisotropic distance
    __device__ void worleyNoise2D(Vec2f p, float scale, uint32_t seed, float* f1, float* f2, float anisotropy = 1.0f) {
        p = p * scale;
        Vec2f cell = Vec2f(floor(p.x()), floor(p.y()));
        Vec2f frac = Vec2f(p.x() - cell.x(), p.y() - cell.y());
        
        *f1 = 1e10f;
        *f2 = 1e10f;
        
        // Check 3x3 neighboring cells
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                Vec2f neighbor = cell + Vec2f((float)x, (float)y);
                
                // Generate cell point using hash
                uint32_t h = tea<4>(seed, 
                    static_cast<uint32_t>(neighbor.x() * 374761393 + neighbor.y() * 668265263));
                
                float px = rnd(h);
                float py = rnd(h);
                Vec2f point = neighbor + Vec2f(px, py);
                
                // Anisotropic distance: stretch Y coordinate for distance calculation
                Vec2f diff = point - p;
                diff.y() *= anisotropy;  // Apply anisotropy to Y axis
                float dist = length(diff);
                
                if (dist < *f1) {
                    *f2 = *f1;
                    *f1 = dist;
                } else if (dist < *f2) {
                    *f2 = dist;
                }
            }
        }
    }
    
    // Smoothstep helper function
    __device__ inline float smoothstep(float edge0, float edge1, float x) {
        float t = fmaxf(0.0f, fminf(1.0f, (x - edge0) / (edge1 - edge0)));
        return t * t * (3.0f - 2.0f * t);
    }
    
    // Rough bark (Worley crackle)
    __device__ float roughBarkHeight(Vec2f uv, PerlinNoiseData* perlin, const TreeBarkTexture& bark) {
        // Use isotropic Voronoi (no UV stretching)
        float cell_scale = 8.0f;        // bark.rough.cell_scale
        float crack_depth = 0.5f;       // bark.rough.crack_depth (increased)
        float anisotropy = 1.5f;        // Anisotropic distance metric for wood grain
        
        float f1, f2;
        worleyNoise2D(uv, cell_scale, bark.seed, &f1, &f2, anisotropy);
        
        // Crack pattern (F2-F1)
        float crack = f2 - f1;
        float crack_threshold = 0.15f;
        float crack_mask = smoothstep(0.0f, crack_threshold, crack);
        
        // Base displacement with cracks
        float base = f1;  // Use full range 0-1
        float height = base * (0.5f + crack_mask * 0.5f);  // 0.5-1.0 range
        
        // Add crack depth (inverted crack_mask makes cracks deeper)
        height -= crack_depth * (1.0f - crack_mask);
        
        // Add Perlin noise for fine detail (ざらざら感)
        Vec3f p_fine = Vec3f(uv.x(), uv.y(), 0.5f) * 60.0f;
        float fine_noise = perlin->noise(p_fine) * 0.06f;
        
        // Ensure positive values and scale to reasonable range
        return fmaxf(0.0f, height + fine_noise + 0.5f);  // Offset to ensure positive
    }
    
    // Aged bark (Layered FBM with domain warping)
    __device__ float agedBarkHeight(Vec2f uv, PerlinNoiseData* perlin, const TreeBarkTexture& bark) {
        int octaves = 6;
        float scale = 5.0f;
        float warp_strength = 0.3f;
        float vertical_bias = 0.5f;
        
        // Domain warping for organic flow
        Vec3f p1 = Vec3f(uv.x(), uv.y(), 0.5f) * scale;
        float warp_x = perlin->turb(p1, octaves) * warp_strength;
        float warp_y = perlin->turb(p1 + Vec3f(5.2f, 1.3f, 0.0f), octaves) * warp_strength;
        
        Vec2f warped_uv = uv + Vec2f(warp_x, warp_y);
        
        // Vertical bias (wood grain flows vertically)
        warped_uv.y() += uv.y() * vertical_bias;
        
        // Multi-octave noise
        Vec3f p2 = Vec3f(warped_uv.x(), warped_uv.y(), 0.5f) * scale * 2.0f;
        float height = perlin->turb(p2, octaves);
        
        // Add coarse features
        Vec3f p3 = Vec3f(uv.x(), uv.y(), 0.5f) * scale * 0.5f;
        height += perlin->noise(p3) * 0.3f;
        
        // Normalize to positive range (turb returns 0-1 typically)
        return height * 0.5f + 0.5f;  // Ensure 0.5-1.0 range
    }
    
    // Smooth bark (Flow lines - 滑らかな流線型)
    __device__ float smoothBarkHeight(Vec2f uv, PerlinNoiseData* perlin, const TreeBarkTexture& bark) {
        float flow_scale = 3.0f;
        float flow_strength = 0.2f;
        float ripple_frequency = 30.0f;
        float smoothness = 2.0f;
        
        // Vertical flow direction
        float flow_noise = perlin->noise(Vec3f(uv.x() * flow_scale, 
                                                uv.y() * flow_scale * 0.3f, 
                                                0.5f));
        
        // Create smooth vertical streaks
        float flow_offset = flow_noise * flow_strength;
        float streak = sinf((uv.x() + flow_offset) * flow_scale * 10.0f);
        streak = powf(fabsf(streak), smoothness);
        
        // Subtle horizontal ripples (growth rings)
        float ripple = sinf(uv.y() * ripple_frequency) * 0.1f;
        
        // Very fine detail
        float detail = perlin->noise(Vec3f(uv.x(), uv.y(), 0.5f) * 50.0f) * 0.02f;
        
        return streak * 0.3f + ripple + detail;
    }
    
    // Smooth bark color (red-brown -> white -> green blend)
    __device__ Vec3f smoothBarkColor(Vec2f uv, PerlinNoiseData* perlin, const TreeBarkTexture& bark) {
        // Base colors: red-brown (base), white (bark), green (moss)
        Vec3f red_brown = Vec3f(0.4f, 0.2f, 0.1f);   // 赤茶色
        Vec3f white_bark = Vec3f(0.9f, 0.85f, 0.75f); // 白っぽい樹皮
        Vec3f green_moss = Vec3f(0.2f, 0.4f, 0.2f);   // 緑のこけ
        
        // Vertical gradient with noise variation
        float noise_variation = perlin->noise(Vec3f(uv.x() * 5.0f, uv.y() * 5.0f, 0.5f)) * 0.3f;
        float gradient = uv.y() + noise_variation;
        
        // Add vertical streaks for flow effect
        float flow_noise = perlin->noise(Vec3f(uv.x() * 3.0f, uv.y() * 1.0f, 0.7f));
        float streak = sinf((uv.x() + flow_noise * 0.2f) * 10.0f) * 0.5f + 0.5f;
        
        // Blend colors based on vertical position
        Vec3f color;
        if (gradient < 0.4f) {
            // Bottom: red-brown to white
            float t = gradient / 0.4f;
            color = red_brown * (1.0f - t) + white_bark * t;
        } else if (gradient < 0.7f) {
            // Middle: white dominant
            float t = (gradient - 0.4f) / 0.3f;
            color = white_bark * (1.0f - t * 0.5f) + green_moss * (t * 0.3f);
        } else {
            // Top: white to green (moss)
            float t = (gradient - 0.7f) / 0.3f;
            t = fminf(t, 1.0f);
            color = white_bark * (1.0f - t) + green_moss * t;
        }
        
        // Apply streak variation
        color = color * (0.8f + streak * 0.4f);
        
        return color;
    }

    
    __global__ void bakeTreeBarkTextureKernel(
        PerlinNoiseData* d_perlin,
        TreeBarkTexture* d_bark,  // Pass as pointer to device memory
        uint32_t width,
        uint32_t height,
        float* output_buffer
    ) {
        const uint32_t ix = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t iy = blockIdx.y * blockDim.y + threadIdx.y;

        if (ix >= width || iy >= height) return;

        float u = (ix + 0.5f) / static_cast<float>(width);
        float v = (iy + 0.5f) / static_cast<float>(height);
        Vec2f uv(u, v);

        float bark_height = 0.0f;
        
        // Debug: Print bark type for first pixel
        if (ix == 0 && iy == 0) {
            printf("[KERNEL DEBUG] d_bark->type = %d, bump_strength = %.4f\n", 
                   static_cast<int>(d_bark->type), d_bark->bump_strength);
        }
        
        switch(d_bark->type) {
            case BarkType::ROUGH:
                bark_height = roughBarkHeight(uv, d_perlin, *d_bark);
                if (ix == 0 && iy == 0) printf("[KERNEL] ROUGH bark_height = %.4f\n", bark_height);
                break;
            case BarkType::AGED:
                bark_height = agedBarkHeight(uv, d_perlin, *d_bark);
                if (ix == 0 && iy == 0) printf("[KERNEL] AGED bark_height = %.4f\n", bark_height);
                break;
            case BarkType::SMOOTH:
                bark_height = smoothBarkHeight(uv, d_perlin, *d_bark);
                if (ix == 0 && iy == 0) printf("[KERNEL] SMOOTH bark_height = %.4f\n", bark_height);
                break;
        }
        
        float final_value = bark_height * d_bark->bump_strength;
        if (ix == 0 && iy == 0) {
            printf("[KERNEL] final_value = %.4f\n", final_value);
        }
        
        output_buffer[iy * width + ix] = final_value;
    }
    
    extern "C" float* bakeTreeBarkTexture(
        TreeBarkTexture bark_data,
        uint32_t width,
        uint32_t height
    ) {
        // Debug: Print bump_strength at function entry
        printf("[bakeTreeBarkTexture] Received bump_strength = %.4f\n", bark_data.bump_strength);
        
        // Generate Perlin Noise data (for AGED and SMOOTH types)
        PerlinNoiseData h_perlin_data;
        unsigned int local_seed = bark_data.seed;
        
        // Generate random vectors
        for (int i = 0; i < PerlinNoiseData::POINT_COUNT; i++) {
            const Vec3f rnd_v = Vec3f(rnd(local_seed), rnd(local_seed), rnd(local_seed)) * 2.0f - 1.0f;
            h_perlin_data.rnd_vec[i] = normalize(rnd_v);
        }
        
        // Generate permutation tables
        auto generatePerm = [&](int* perm) {
            for (int i = 0; i < PerlinNoiseData::POINT_COUNT; i++)
                perm[i] = i;
            
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
        
        // Copy to device
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
        
        // Copy TreeBarkTexture to device
        TreeBarkTexture* d_bark_data;
        err = cudaMalloc(&d_bark_data, sizeof(TreeBarkTexture));
        if (err != cudaSuccess) {
            printf("[ERROR] cudaMalloc failed for TreeBarkTexture: %s\n", cudaGetErrorString(err));
            cudaFree(d_perlin_data);
            return nullptr;
        }
        
        err = cudaMemcpy(d_bark_data, &bark_data, sizeof(TreeBarkTexture), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("[ERROR] cudaMemcpy failed for TreeBarkTexture: %s\n", cudaGetErrorString(err));
            cudaFree(d_perlin_data);
            cudaFree(d_bark_data);
            return nullptr;
        }
        
        // Allocate texture buffer
        float* d_texture;
        size_t texture_size = width * height * sizeof(float);
        err = cudaMalloc(&d_texture, texture_size);
        if (err != cudaSuccess) {
            printf("[ERROR] cudaMalloc failed for bark texture (%zu bytes): %s\n", 
                   texture_size, cudaGetErrorString(err));
            cudaFree(d_perlin_data);
            cudaFree(d_bark_data);
            return nullptr;
        }

        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                      (height + blockSize.y - 1) / blockSize.y);

        printf("[HOST] About to launch kernel with bump_strength = %.4f\n", bark_data.bump_strength);

        bakeTreeBarkTextureKernel<<<gridSize, blockSize>>>(
            d_perlin_data, d_bark_data, width, height, d_texture
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("[ERROR] bakeTreeBarkTextureKernel launch failed: %s\n", 
                   cudaGetErrorString(err));
            cudaFree(d_texture);
            cudaFree(d_perlin_data);
            cudaFree(d_bark_data);
            return nullptr;
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("[ERROR] bakeTreeBarkTextureKernel execution failed: %s\n", 
                   cudaGetErrorString(err));
            cudaFree(d_texture);
            cudaFree(d_perlin_data);
            cudaFree(d_bark_data);
            return nullptr;
        }

        cudaFree(d_perlin_data);
        cudaFree(d_bark_data);
        
        return d_texture;
    }
    
    // Generate smooth bark color texture (not bumpmap)
    __global__ void bakeSmoothBarkColorKernel(
        PerlinNoiseData* d_perlin,
        TreeBarkTexture bark,
        uint32_t width,
        uint32_t height,
        Vec4f* output_buffer
    ) {
        const uint32_t ix = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t iy = blockIdx.y * blockDim.y + threadIdx.y;

        if (ix >= width || iy >= height) return;

        float u = (ix + 0.5f) / static_cast<float>(width);
        float v = (iy + 0.5f) / static_cast<float>(height);
        Vec2f uv(u, v);

        Vec3f color = smoothBarkColor(uv, d_perlin, bark);
        output_buffer[iy * width + ix] = Vec4f(color.x(), color.y(), color.z(), 1.0f);
    }
    
    extern "C" Vec4f* bakeSmoothBarkColorTexture(
        TreeBarkTexture bark_data,
        uint32_t width,
        uint32_t height
    ) {
        Vec4f* d_texture;
        size_t texture_size = width * height * sizeof(Vec4f);
        cudaError_t err = cudaMalloc(&d_texture, texture_size);
        if (err != cudaSuccess) {
            printf("[ERROR] cudaMalloc failed for smooth bark color texture: %s\n", 
                   cudaGetErrorString(err));
            return nullptr;
        }
        
        // Copy Perlin noise data to device
        PerlinNoiseData* d_perlin_data;
        err = cudaMalloc(&d_perlin_data, sizeof(PerlinNoiseData));
        if (err != cudaSuccess) {
            printf("[ERROR] cudaMalloc failed for PerlinNoiseData: %s\n", 
                   cudaGetErrorString(err));
            cudaFree(d_texture);
            return nullptr;
        }
        
        PerlinNoiseData h_perlin_data;
        
        // Generate random vectors and permutation tables (same as bakeRockTexture)
        unsigned int local_seed = bark_data.seed;
        for (int i = 0; i < PerlinNoiseData::POINT_COUNT; i++) {
            const Vec3f rnd_v = Vec3f(rnd(local_seed), rnd(local_seed), rnd(local_seed)) * 2.0f - 1.0f;
            h_perlin_data.rnd_vec[i] = normalize(rnd_v);
        }
        
        auto generatePerm = [&](int* perm) {
            for (int i = 0; i < PerlinNoiseData::POINT_COUNT; i++)
                perm[i] = i;
            
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
        
        err = cudaMemcpy(d_perlin_data, &h_perlin_data, sizeof(PerlinNoiseData), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            printf("[ERROR] cudaMemcpy failed for PerlinNoiseData: %s\n", 
                   cudaGetErrorString(err));
            cudaFree(d_texture);
            cudaFree(d_perlin_data);
            return nullptr;
        }
        
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                      (height + blockSize.y - 1) / blockSize.y);
        
        bakeSmoothBarkColorKernel<<<gridSize, blockSize>>>(
            d_perlin_data, bark_data, width, height, d_texture
        );
        
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("[ERROR] bakeSmoothBarkColorKernel launch failed: %s\n", 
                   cudaGetErrorString(err));
            cudaFree(d_texture);
            cudaFree(d_perlin_data);
            return nullptr;
        }
        
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("[ERROR] bakeSmoothBarkColorKernel execution failed: %s\n", 
                   cudaGetErrorString(err));
            cudaFree(d_texture);
            cudaFree(d_perlin_data);
            return nullptr;
        }
        
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

        // Create tangent vectors from height differences (Z-up coordinate system)
        // Tangent along X axis: (dx, 0, height_diff)
        // Tangent along Y axis: (0, dy, height_diff)
        float h_size = 1.0f / (float)width;
        float v_size = 1.0f / (float)height;
        Vec3f tangent_x = normalize(Vec3f(h_size, 0.0f, (hR - hL) * bump_strength));
        Vec3f tangent_y = normalize(Vec3f(0.0f, v_size, (hU - hD) * bump_strength));
        
        // Normal is cross product of tangents (Z-up: tangent_y × tangent_x for upward normal)
        Vec3f normal = normalize(cross(tangent_y, tangent_x));

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