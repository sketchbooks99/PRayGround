#pragma once

#include <cuda_runtime.h>
#include <prayground/math/vec.h>

namespace prayground {

// Bloom effect parameters
struct BloomParams {
    float threshold;           // Brightness threshold for bloom
    float intensity;          // Bloom intensity (blend factor)
    int blur_radius;          // Gaussian blur kernel radius
    float sigma;              // Gaussian blur sigma
};

// Extract bright pixels above threshold (Vec4f version)
void launchBrightPassKernel(
    const Vec4f* input,
    Vec4f* output,
    int width,
    int height,
    float threshold,
    cudaStream_t stream = 0
);

// Gaussian blur horizontal pass (Vec4f version)
void launchGaussianBlurHorizontalKernel(
    const Vec4f* input,
    Vec4f* output,
    int width,
    int height,
    int radius,
    float sigma,
    cudaStream_t stream = 0
);

// Gaussian blur vertical pass (Vec4f version)
void launchGaussianBlurVerticalKernel(
    const Vec4f* input,
    Vec4f* output,
    int width,
    int height,
    int radius,
    float sigma,
    cudaStream_t stream = 0
);

// Additive blend: result = original + bloom * intensity (Vec4f version)
void launchAdditiveBlendKernel(
    const Vec4f* original,
    const Vec4f* bloom,
    Vec4f* output,
    int width,
    int height,
    float intensity,
    cudaStream_t stream = 0
);

// Complete bloom effect (Vec4f version, all passes combined)
void applyBloomEffect(
    const Vec4f* input,
    Vec4f* output,
    Vec4f* temp_buffer1,
    Vec4f* temp_buffer2,
    int width,
    int height,
    const BloomParams& params,
    cudaStream_t stream = 0
);

// Firefly filtering parameters
struct FireflyFilterParams {
    float outlier_ratio;       // Threshold ratio for outlier detection (e.g., 2.5x brighter than neighbors)
    float min_luminance;       // Minimum luminance to consider (avoid false positives in dark areas)
};

// Firefly filtering: Replace bright outlier pixels with 8-neighbor average
void launchFireflyFilterKernel(
    const Vec4f* input,
    Vec4f* output,
    int width,
    int height,
    const FireflyFilterParams& params,
    cudaStream_t stream = 0
);

// Convergence checking: Returns ratio of converged pixels [0.0, 1.0]
float checkConvergenceRatio(
    const uint8_t* d_converged_buffer,
    int width,
    int height,
    cudaStream_t stream = 0
);

} // namespace prayground