#include "envmap.cuh"
#include "textures.h"
#include <prayground/math/noise.h>
#include <stdio.h>
#include <math.h>

using namespace prayground;


// ========================================
// CUDA Kernel: Bake procedural texture
// Same implementation as __direct_callable__star_night
// ========================================

__global__ void bakeTextureKernel(
    uint32_t width,
    uint32_t height,
    StarNightTexture::Data star_night_data,
    Vec4f* output_buffer
) {
    const uint32_t ix = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t iy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (ix >= width || iy >= height) return;
    
    // Calculate UV coordinates
    float u = (ix + 0.5f) / width;
    float v = (iy + 0.5f) / height;
    Vec2f uv(u, v);
    
    // Star night texture evaluation (same as __direct_callable__star_night)
    auto n = star_night_data.noise_data;
    RandomNoise r_noise(n.seed);
    uint32_t pixel_seed = n.seed + ix * height + iy;

    Vec3f base_color = Vec3f(star_night_data.base_color);

    float star_thres = star_night_data.star_threshold;
    float star_intensity = star_night_data.star_intensity;
    Vec3f moon_dir = normalize(star_night_data.moon_dir);
    float moon_intensity = star_night_data.moon_intensity;
    
    Vec3i p = Vec3i(
        static_cast<int>(uv.x() * n.width),
        static_cast<int>(uv.y() * n.height),
        0
    );
    float value = r_noise.noise(p);

    float theta = uv.y() * math::pi;
    float phi = uv.x() * 2.0f * math::pi;
    Vec3f p3d(
        sinf(theta) * cosf(phi),
        cosf(theta),
        sinf(theta) * sinf(phi)
    );

    float moon_proximity = dot(moon_dir, p3d);
    float moon_radius = 0.03f;
    
    Vec3f color(0.0f);
    if (moon_proximity > cosf(moon_radius)) {
        float moon_factor = (moon_proximity - cosf(moon_radius)) / (1.0f - cosf(moon_radius));
        color = Vec3f(0.9f, 0.75f, 0.6f) * (moon_intensity + moon_factor * moon_intensity);
    } 
    else if (value > star_thres) {
        // Use pixel-based seed for random color variation
        float color_rnd = rnd(pixel_seed);
        Vec3f star_color = Vec3f(1.0f, 1.0f, 0.9f);
        float intensity = star_intensity * rnd(pixel_seed);
        if (color_rnd < 0.1f) {
            star_color = Vec3f(1.0f, 0.8f, 0.6f);  // Warm star
        }
        else if (color_rnd < 0.2f) {
            star_color = Vec3f(0.6f, 0.8f, 1.0f);  // Cool star
        }
        color = star_color * intensity;
    }
    else {
        color = base_color;
    }
    
    const uint32_t idx = iy * width + ix;
    output_buffer[idx] = Vec4f(color.x(), color.y(), color.z(), 1.0f);
}

// ========================================
// Bake procedural texture - returns device pointer
// ========================================

extern "C" Vec4f* bakeProceduralTextureToDevice(
    StarNightTexture::Data star_night_data,
    uint32_t width,
    uint32_t height
) {
    printf("[Envmap] Baking procedural texture to device buffer (%dx%d)...\n", width, height);
    
    // Allocate device buffer for output (Vec4f for RGBA)
    Vec4f* d_output;
    cudaMalloc(&d_output, sizeof(Vec4f) * width * height);
    
    // Launch CUDA kernel
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    bakeTextureKernel<<<grid, block>>>(
        width, height, star_night_data, d_output
    );
    
    cudaDeviceSynchronize();
    
    printf("[Envmap] Texture baked successfully!\n");
    
    return d_output;
}

// ========================================
// Build CDF for importance sampling
// ========================================

extern "C" EnvmapSamplingData buildEnvmapSamplingDataFromDevice(
    Vec4f* d_colors,
    uint32_t width,
    uint32_t height,
    uint32_t texture_id,
    void* texture_device_data
) {
    printf("[Envmap] Building importance sampling data (%dx%d)...\n", width, height);
    
    // Allocate host memory
    Vec4f* h_colors = (Vec4f*)malloc(sizeof(Vec4f) * width * height);
    float* h_conditional_cdf = (float*)malloc(sizeof(float) * width * height);
    float* h_row_luminances = (float*)malloc(sizeof(float) * height);
    float* h_marginal_cdf = (float*)malloc(sizeof(float) * height);
    
    // Copy bitmap data from device to host
    cudaMemcpy(h_colors, d_colors, sizeof(Vec4f) * width * height, cudaMemcpyDeviceToHost);
    
    float total_luminance = 0.0f;
    const float pi = math::pi;
    
    // Build conditional CDF per row
    for (uint32_t v = 0; v < height; v++) {
        float row_sum = 0.0f;
        
        for (uint32_t u = 0; u < width; u++) {
            float uv_u = (u + 0.5f) / width;
            float uv_v = (v + 0.5f) / height;
            
            Vec4f color = h_colors[v * width + u];
            
            // Calculate luminance (Rec.709) - ignore alpha channel
            float luminance = 0.2126f * color.x() + 0.7152f * color.y() + 0.0722f * color.z();
            
            // Weight by sin(theta) for spherical parameterization
            float theta = uv_v * pi;
            float sin_theta = sinf(theta);
            if (sin_theta < 1e-6f) sin_theta = 1e-6f;
            
            luminance *= sin_theta;
            
            row_sum += luminance;
            h_conditional_cdf[v * width + u] = row_sum;
        }
        
        // Normalize conditional CDF
        if (row_sum > 0.0f) {
            for (uint32_t u = 0; u < width; u++) {
                h_conditional_cdf[v * width + u] /= row_sum;
            }
        }
        
        h_row_luminances[v] = row_sum;
        total_luminance += row_sum;
    }
    
    // Build marginal CDF
    float marginal_sum = 0.0f;
    for (uint32_t v = 0; v < height; v++) {
        marginal_sum += h_row_luminances[v];
        h_marginal_cdf[v] = marginal_sum;
    }
    
    // Normalize marginal CDF
    if (marginal_sum > 0.0f) {
        for (uint32_t v = 0; v < height; v++) {
            h_marginal_cdf[v] /= marginal_sum;
        }
    }
    
    printf("[Envmap] Total luminance: %.2f\n", total_luminance);
    
    // Allocate device memory for CDFs
    float* d_conditional_cdf;
    float* d_row_luminances;
    float* d_marginal_cdf;
    
    cudaMalloc(&d_conditional_cdf, sizeof(float) * width * height);
    cudaMalloc(&d_row_luminances, sizeof(float) * height);
    cudaMalloc(&d_marginal_cdf, sizeof(float) * height);
    
    cudaMemcpy(d_conditional_cdf, h_conditional_cdf,
               sizeof(float) * width * height, cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_luminances, h_row_luminances,
               sizeof(float) * height, cudaMemcpyHostToDevice);
    cudaMemcpy(d_marginal_cdf, h_marginal_cdf,
               sizeof(float) * height, cudaMemcpyHostToDevice);
    
    // Free host memory
    free(h_colors);
    free(h_conditional_cdf);
    free(h_row_luminances);
    free(h_marginal_cdf);
    
    // Create EnvmapSamplingData structure on host
    EnvmapSamplingData h_envmap_data;
    h_envmap_data.conditional_cdf = d_conditional_cdf;
    h_envmap_data.row_luminances = d_row_luminances;
    h_envmap_data.marginal_cdf = d_marginal_cdf;
    h_envmap_data.width = width;
    h_envmap_data.height = height;
    h_envmap_data.total_luminance = total_luminance;
    
    return h_envmap_data;
}
