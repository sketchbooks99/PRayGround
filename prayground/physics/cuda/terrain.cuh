#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <prayground/math/vec.h>

namespace prayground {

// ============================================================================
// Perlin Noise Utility Functions
// ============================================================================

// Hash function for pseudo-random gradient generation
__device__ inline unsigned int hash(unsigned int x) {
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x;
}

__device__ inline unsigned int hash(unsigned int x, unsigned int y, unsigned int seed) {
    return hash(x + hash(y + seed));
}

// Generate pseudo-random gradient vector
__device__ inline Vec2f gradient2D(int ix, int iy, unsigned int seed) {
    unsigned int h = hash(ix, iy, seed);
    float angle = (h & 0xFF) * (2.0f * M_PI / 256.0f);
    return Vec2f(cosf(angle), sinf(angle));
}

// Smoothstep interpolation (5th order for better smoothness)
__device__ inline float smoothstep(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

// Bilinear interpolation
__device__ inline float bilerp(float v00, float v10, float v01, float v11, float tx, float ty) {
    float a = v00 * (1.0f - tx) + v10 * tx;
    float b = v01 * (1.0f - tx) + v11 * tx;
    return a * (1.0f - ty) + b * ty;
}

// 2D Perlin noise at given position
__device__ inline float perlin2D(float x, float y, unsigned int seed) {
    // Grid cell coordinates
    int x0 = floorf(x);
    int y0 = floorf(y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    
    // Local coordinates within cell
    float sx = x - x0;
    float sy = y - y0;
    
    // Gradients at corners
    Vec2f g00 = gradient2D(x0, y0, seed);
    Vec2f g10 = gradient2D(x1, y0, seed);
    Vec2f g01 = gradient2D(x0, y1, seed);
    Vec2f g11 = gradient2D(x1, y1, seed);
    
    // Distance vectors from corners
    Vec2f d00 = Vec2f(sx, sy);
    Vec2f d10 = Vec2f(sx - 1.0f, sy);
    Vec2f d01 = Vec2f(sx, sy - 1.0f);
    Vec2f d11 = Vec2f(sx - 1.0f, sy - 1.0f);
    
    // Dot products
    float n00 = dot(g00, d00);
    float n10 = dot(g10, d10);
    float n01 = dot(g01, d01);
    float n11 = dot(g11, d11);
    
    // Smooth interpolation
    float tx = smoothstep(sx);
    float ty = smoothstep(sy);
    
    return bilerp(n00, n10, n01, n11, tx, ty);
}

// Multi-octave Perlin noise (FBM - Fractal Brownian Motion)
__device__ inline float perlinFBM(float x, float y, int octaves, float frequency, 
                                  float lacunarity, float persistence, unsigned int seed) {
    float value = 0.0f;
    float amplitude = 1.0f;
    float max_value = 0.0f;
    
    for (int i = 0; i < octaves; i++) {
        value += perlin2D(x * frequency, y * frequency, seed + i) * amplitude;
        max_value += amplitude;
        
        amplitude *= persistence;
        frequency *= lacunarity;
    }
    
    return value / max_value; // Normalize to [-1, 1]
}

// ============================================================================
// Heightmap Access Functions
// ============================================================================

// Get height at grid coordinates with bounds checking
__device__ inline float getHeight(const float* heightmap, int x, int z, int width, int height) {
    // Clamp to valid range (prevents edge artifacts)
    x = max(0, min(x, width - 1));
    z = max(0, min(z, height - 1));
    return heightmap[z * width + x];
}

// Set height at grid coordinates
__device__ inline void setHeight(float* heightmap, int x, int z, int width, int height, float value) {
    if (x >= 0 && x < width && z >= 0 && z < height)
        heightmap[z * width + x] = value;
}

// Bilinear interpolation of height at arbitrary position
__device__ inline float sampleHeight(const float* heightmap, float x, float z, int width, int height) {
    int x0 = floorf(x);
    int z0 = floorf(z);
    int x1 = x0 + 1;
    int z1 = z0 + 1;
    
    float fx = x - x0;
    float fz = z - z0;
    
    float h00 = getHeight(heightmap, x0, z0, width, height);
    float h10 = getHeight(heightmap, x1, z0, width, height);
    float h01 = getHeight(heightmap, x0, z1, width, height);
    float h11 = getHeight(heightmap, x1, z1, width, height);
    
    return bilerp(h00, h10, h01, h11, fx, fz);
}

// Calculate gradient (slope) at position
__device__ inline Vec2f calculateGradient(const float* heightmap, float x, float z, 
                                          int width, int height, float cell_size) {
    float h = sampleHeight(heightmap, x, z, width, height);
    float hx = sampleHeight(heightmap, x + 1.0f, z, width, height);
    float hz = sampleHeight(heightmap, x, z + 1.0f, width, height);
    
    return Vec2f((hx - h) / cell_size, (hz - h) / cell_size);
}

} // namespace prayground
