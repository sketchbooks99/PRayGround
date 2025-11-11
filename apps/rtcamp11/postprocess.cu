#include "bloom.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

namespace prayground {

// ----------------------------------------------------------------------------
// Device helper functions
// ----------------------------------------------------------------------------

__device__ inline float luminance(const Vec4f& color) {
    // Standard luminance calculation (Rec. 709)
    return 0.2126f * color.x() + 0.7152f * color.y() + 0.0722f * color.z();
}

__device__ inline float gaussian(float x, float sigma) {
    return expf(-(x * x) / (2.0f * sigma * sigma));
}

// ----------------------------------------------------------------------------
// Bright Pass Kernel
// ----------------------------------------------------------------------------

__global__ void brightPassKernel(
    const Vec4f* input,
    Vec4f* output,
    int width,
    int height,
    float threshold
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    Vec4f color = input[idx];
    
    // Calculate luminance
    float lum = luminance(color);
    
    // Extract only pixels above threshold
    if (lum > threshold) {
        // Preserve relative color ratios, scale by excess brightness
        float excess = lum - threshold;
        output[idx] = color * (excess / lum);
    } else {
        output[idx] = Vec4f(0.0f, 0.0f, 0.0f, color.w());
    }
}

void launchBrightPassKernel(
    const Vec4f* input,
    Vec4f* output,
    int width,
    int height,
    float threshold,
    cudaStream_t stream
) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    brightPassKernel<<<gridSize, blockSize, 0, stream>>>(
        input, output, width, height, threshold
    );
}

// ----------------------------------------------------------------------------
// Gaussian Blur Horizontal Kernel
// ----------------------------------------------------------------------------

__global__ void gaussianBlurHorizontalKernel(
    const Vec4f* input,
    Vec4f* output,
    int width,
    int height,
    int radius,
    float sigma
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    Vec4f sum(0.0f, 0.0f, 0.0f, 0.0f);
    float weight_sum = 0.0f;

    // Apply Gaussian blur in horizontal direction
    for (int dx = -radius; dx <= radius; dx++) {
        int sample_x = x + dx;
        
        // Clamp to image boundaries
        sample_x = max(0, min(width - 1, sample_x));
        
        int idx = y * width + sample_x;
        float weight = gaussian(static_cast<float>(dx), sigma);
        
        sum += input[idx] * weight;
        weight_sum += weight;
    }

    output[y * width + x] = sum / weight_sum;
}

void launchGaussianBlurHorizontalKernel(
    const Vec4f* input,
    Vec4f* output,
    int width,
    int height,
    int radius,
    float sigma,
    cudaStream_t stream
) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    gaussianBlurHorizontalKernel<<<gridSize, blockSize, 0, stream>>>(
        input, output, width, height, radius, sigma
    );
}

// ----------------------------------------------------------------------------
// Gaussian Blur Vertical Kernel
// ----------------------------------------------------------------------------

__global__ void gaussianBlurVerticalKernel(
    const Vec4f* input,
    Vec4f* output,
    int width,
    int height,
    int radius,
    float sigma
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    Vec4f sum(0.0f, 0.0f, 0.0f, 0.0f);
    float weight_sum = 0.0f;

    // Apply Gaussian blur in vertical direction
    for (int dy = -radius; dy <= radius; dy++) {
        int sample_y = y + dy;
        
        // Clamp to image boundaries
        sample_y = max(0, min(height - 1, sample_y));
        
        int idx = sample_y * width + x;
        float weight = gaussian(static_cast<float>(dy), sigma);
        
        sum += input[idx] * weight;
        weight_sum += weight;
    }

    output[y * width + x] = sum / weight_sum;
}

void launchGaussianBlurVerticalKernel(
    const Vec4f* input,
    Vec4f* output,
    int width,
    int height,
    int radius,
    float sigma,
    cudaStream_t stream
) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    gaussianBlurVerticalKernel<<<gridSize, blockSize, 0, stream>>>(
        input, output, width, height, radius, sigma
    );
}

// ----------------------------------------------------------------------------
// Additive Blend Kernel
// ----------------------------------------------------------------------------

__global__ void additiveBlendKernel(
    const Vec4f* original,
    const Vec4f* bloom,
    Vec4f* output,
    int width,
    int height,
    float intensity
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    
    // Additive blending: original + bloom * intensity (RGB only, preserve alpha)
    Vec4f orig = original[idx];
    Vec4f blm = bloom[idx];
    output[idx] = Vec4f(
        orig.x() + blm.x() * intensity,
        orig.y() + blm.y() * intensity,
        orig.z() + blm.z() * intensity,
        orig.w()
    );
}

void launchAdditiveBlendKernel(
    const Vec4f* original,
    const Vec4f* bloom,
    Vec4f* output,
    int width,
    int height,
    float intensity,
    cudaStream_t stream
) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    additiveBlendKernel<<<gridSize, blockSize, 0, stream>>>(
        original, bloom, output, width, height, intensity
    );
}

// ----------------------------------------------------------------------------
// Complete Bloom Effect
// ----------------------------------------------------------------------------

void applyBloomEffect(
    const Vec4f* input,
    Vec4f* output,
    Vec4f* temp_buffer1,
    Vec4f* temp_buffer2,
    int width,
    int height,
    const BloomParams& params,
    cudaStream_t stream
) {
    // Step 1: Extract bright pixels
    launchBrightPassKernel(input, temp_buffer1, width, height, params.threshold, stream);
    
    // Step 2: Gaussian blur horizontal pass
    launchGaussianBlurHorizontalKernel(temp_buffer1, temp_buffer2, width, height, 
                                       params.blur_radius, params.sigma, stream);
    
    // Step 3: Gaussian blur vertical pass
    launchGaussianBlurVerticalKernel(temp_buffer2, temp_buffer1, width, height,
                                     params.blur_radius, params.sigma, stream);
    
    // Step 4: Additive blend with original
    launchAdditiveBlendKernel(input, temp_buffer1, output, width, height, 
                             params.intensity, stream);
}

} // namespace prayground