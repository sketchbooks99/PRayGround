#include "svgf.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <prayground/optix/macros.h>
#include <cstdio>

namespace prayground {

// ============================================================================
// CUDA Kernels
// ============================================================================

// Helper: Convert Vec4f to Vec3f
__device__ __forceinline__ Vec3f toVec3f(const Vec4f& v) {
    return Vec3f(v.x(), v.y(), v.z());
}

// Helper: Convert Vec4f to Vec2f (for motion)
__device__ __forceinline__ Vec2f toVec2f(const Vec4f& v) {
    return Vec2f(v.x(), v.y());
}

// Helper: Convert Vec4f to Vec3f buffer
__global__ void convertVec4fToVec3fKernel(
    const Vec4f* input,
    Vec3f* output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    output[idx] = toVec3f(input[idx]);
}

// Albedo demodulation: RGB / Albedo → Irradiance (lighting only)
__global__ void albedoDemodulationKernel(
    const Vec4f* rgb_input,
    const Vec4f* albedo_input,
    Vec3f* irradiance_output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    Vec3f color = toVec3f(rgb_input[idx]);
    Vec3f albedo = toVec3f(albedo_input[idx]);
    
    // Demodulate: irradiance = color / albedo
    // Avoid division by zero or very dark albedo
    Vec3f irradiance(
        (albedo.x() > 1e-4f) ? color.x() / albedo.x() : color.x(),
        (albedo.y() > 1e-4f) ? color.y() / albedo.y() : color.y(),
        (albedo.z() > 1e-4f) ? color.z() / albedo.z() : color.z()
    );
    
    irradiance_output[idx] = irradiance;
}

// Albedo modulation: Filtered Irradiance × Albedo → RGB
__global__ void albedoModulationKernel(
    const Vec3f* irradiance_input,
    const Vec4f* albedo_input,
    Vec4f* rgb_output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    Vec3f irradiance = irradiance_input[idx];
    Vec3f albedo = toVec3f(albedo_input[idx]);
    
    // Modulate: color = irradiance × albedo
    Vec3f color = irradiance * albedo;
    
    rgb_output[idx] = Vec4f(color, 1.0f);
}

// Extract high-frequency detail
__global__ void extractDetailKernel(
    const Vec3f* original,
    const Vec3f* filtered,
    Vec3f* detail_output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    Vec3f orig = original[idx];
    Vec3f filt = filtered[idx];
    
    // Extract high-frequency detail
    detail_output[idx] = orig - filt;
}

// Restore detail to final image
__global__ void restoreDetailKernel(
    const Vec4f* base_input,
    const Vec3f* detail_input,
    const Vec4f* albedo_input,
    Vec4f* output,
    float detail_strength,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    Vec3f base = toVec3f(base_input[idx]);
    Vec3f detail = detail_input[idx];
    Vec3f albedo = toVec3f(albedo_input[idx]);
    
    // Apply detail modulated by albedo
    // This ensures detail is scaled by material reflectance
    Vec3f detail_modulated = detail * albedo;
    Vec3f result = base + detail_modulated * detail_strength;
    
    // Clamp to avoid negative values
    result = Vec3f(
        fmaxf(0.0f, result.x()),
        fmaxf(0.0f, result.y()),
        fmaxf(0.0f, result.z())
    );
    
    output[idx] = Vec4f(result, 1.0f);
}

// Temporal reprojection and accumulation (RGB version)
__global__ void temporalAccumulationKernel(
    const Vec3f* color_curr,
    const Vec4f* motion,
    Vec3f* color_prev,
    Vec3f* moment1_curr,
    Vec3f* moment2_curr,
    Vec3f* moment1_prev,
    Vec3f* moment2_prev,
    float* history_length,
    float alpha,
    int max_history,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    Vec3f curr_color = color_curr[idx];
    
    // Motion vector to previous frame (extract xy from Vec4f)
    Vec2f mv = toVec2f(motion[idx]);
    int prev_x = x + (int)(mv.x() * width);
    int prev_y = y + (int)(mv.y() * height);
    
    // Check if reprojection is valid
    bool valid = (prev_x >= 0 && prev_x < width && prev_y >= 0 && prev_y < height);
    
    float hist_len = 0.0f;
    Vec3f prev_color(0.0f);
    Vec3f prev_m1(0.0f);
    Vec3f prev_m2(0.0f);
    
    if (valid) {
        int prev_idx = prev_y * width + prev_x;
        prev_color = color_prev[prev_idx];
        prev_m1 = moment1_prev[prev_idx];
        prev_m2 = moment2_prev[prev_idx];
        hist_len = history_length[prev_idx];
        
        // Clamp history
        hist_len = fminf(hist_len + 1.0f, (float)max_history);
    } else {
        hist_len = 1.0f;
    }
    
    // Temporal blending
    float blend_alpha = valid ? fmaxf(alpha, 1.0f / hist_len) : 1.0f;
    
    Vec3f blended_color = curr_color * blend_alpha + prev_color * (1.0f - blend_alpha);
    Vec3f m1 = curr_color * blend_alpha + prev_m1 * (1.0f - blend_alpha);
    Vec3f m2 = curr_color * curr_color * blend_alpha + prev_m2 * (1.0f - blend_alpha);
    
    // Write results
    color_prev[idx] = blended_color;
    moment1_curr[idx] = m1;
    moment2_curr[idx] = m2;
    history_length[idx] = hist_len;
}

// Estimate variance from moments (RGB version)
__global__ void estimateVarianceKernel(
    const Vec3f* moment1,
    const Vec3f* moment2,
    Vec3f* variance,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    // Variance = E[X^2] - E[X]^2 (per channel)
    Vec3f m1 = moment1[idx];
    Vec3f m2 = moment2[idx];
    Vec3f var = Vec3f(
        fmaxf(0.0f, m2.x() - m1.x() * m1.x()),
        fmaxf(0.0f, m2.y() - m1.y() * m1.y()),
        fmaxf(0.0f, m2.z() - m1.z() * m1.z())
    );
    
    variance[idx] = var;
}

// Edge-stopping function
__device__ float computeWeight(
    const Vec4f& pos_center,
    const Vec4f& pos_sample,
    const Vec4f& normal_center,
    const Vec4f& normal_sample,
    float luma_center,
    float luma_sample,
    float var_center,
    float phi_position,
    float phi_normal,
    float phi_color,
    float sigma_z,
    float sigma_n,
    float sigma_l
) {
    // Convert Vec4f to Vec3f for calculations
    Vec3f pos_c = toVec3f(pos_center);
    Vec3f pos_s = toVec3f(pos_sample);
    Vec3f n_c = toVec3f(normal_center);
    Vec3f n_s = toVec3f(normal_sample);
    
    // Position weight (depth)
    // Special handling for environment map (very far distance)
    float dist = length(pos_c - pos_s);
    const float env_threshold = 1e7f;  // Environment map threshold
    bool both_env = (length(pos_c) > env_threshold) && (length(pos_s) > env_threshold);
    float w_z = both_env ? 1.0f : expf(-dist / (sigma_z * phi_position + 1e-6f));

    // Normal weight
    // For environment map, normals point in different directions (sphere surface)
    // so we skip normal comparison for environment pixels
    float w_n = 1.0f;
    if (!both_env) {
        float cos_angle = dot(n_c, n_s);
        cos_angle = fmaxf(0.0f, cos_angle);
        w_n = powf(cos_angle, sigma_n);
    }
    
    // Luminance weight (variance-guided for temporal, simple for spatial-only)
    float luma_diff = fabsf(luma_center - luma_sample);
    float variance_term = sqrtf(var_center);
    // If variance is very small (spatial-only mode), use fixed threshold
    if (variance_term < 1e-4f) {
        variance_term = 1.0f;  // Use phi_color as threshold directly
    }
    float w_l = expf(-luma_diff / (sigma_l * variance_term * phi_color + 1e-6f));
    
    return w_z * w_n * w_l;
}

// A-trous wavelet filter (single iteration)
// A-trous wavelet filter (single iteration) - RGB version
__global__ void atrousFilterKernel(
    const Vec3f* input,
    const Vec3f* variance,
    const Vec4f* position,
    const Vec4f* normal,
    Vec3f* output,
    int step_size,
    float phi_position,
    float phi_normal,
    float phi_color,
    float sigma_z,
    float sigma_n,
    float sigma_l,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    Vec4f pos_center = position[idx];
    Vec4f normal_center = normal[idx];
    Vec3f color_center = input[idx];
    Vec3f var_center = variance[idx];
    
    // Compute luminance for edge-stopping
    float luma_center = 0.2126f * color_center.x() + 0.7152f * color_center.y() + 0.0722f * color_center.z();
    float var_luma_center = 0.2126f * var_center.x() + 0.7152f * var_center.y() + 0.0722f * var_center.z();
    
    // 3x3 kernel with step size (for wavelet hierarchy)
    const int kernel[3][3] = {
        {1, 2, 1},
        {2, 4, 2},
        {1, 2, 1}
    };
    
    float sum_weight = 0.0f;
    Vec3f sum_color(0.0f);
    
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int sx = x + dx * step_size;
            int sy = y + dy * step_size;
            
            if (sx < 0 || sx >= width || sy < 0 || sy >= height)
                continue;
            
            int sidx = sy * width + sx;
            
            Vec4f pos_sample = position[sidx];
            Vec4f normal_sample = normal[sidx];
            Vec3f color_sample = input[sidx];
            
            // Compute luminance for edge-stopping
            float luma_sample = 0.2126f * color_sample.x() + 0.7152f * color_sample.y() + 0.0722f * color_sample.z();
            
            float weight = (float)kernel[dy + 1][dx + 1];
            weight *= computeWeight(
                pos_center, pos_sample,
                normal_center, normal_sample,
                luma_center, luma_sample,
                var_luma_center,
                phi_position, phi_normal, phi_color,
                sigma_z, sigma_n, sigma_l
            );
            
            sum_weight += weight;
            sum_color += color_sample * weight;
        }
    }
    
    output[idx] = sum_color / (sum_weight + 1e-6f);
}

// Convert Vec3f to Vec4f output
__global__ void convertVec3fToVec4fKernel(
    const Vec3f* input,
    Vec4f* output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    output[idx] = Vec4f(input[idx], 1.0f);
}

// ============================================================================
// SVGF Class Implementation
// ============================================================================

SVGF::SVGF() 
    : m_width(0)
    , m_height(0)
    , m_initialized(false)
    , m_frame_index(0)
{
    m_buffers = {};
}

SVGF::~SVGF() {
    free();
}

void SVGF::init(int width, int height) {
    if (m_initialized && m_width == width && m_height == height)
        return;
    
    if (m_initialized)
        free();
    
    m_width = width;
    m_height = height;
    
    allocateBuffers();
    reset();
    
    m_initialized = true;
}

void SVGF::allocateBuffers() {
    size_t rgb_size = m_width * m_height * sizeof(Vec3f);
    size_t float_size = m_width * m_height * sizeof(float);
    
    cudaMalloc(&m_buffers.color_curr, rgb_size);
    cudaMalloc(&m_buffers.color_prev, rgb_size);
    cudaMalloc(&m_buffers.moment1_curr, rgb_size);
    cudaMalloc(&m_buffers.moment2_curr, rgb_size);
    cudaMalloc(&m_buffers.moment1_prev, rgb_size);
    cudaMalloc(&m_buffers.moment2_prev, rgb_size);
    cudaMalloc(&m_buffers.variance, rgb_size);
    cudaMalloc(&m_buffers.filtered, rgb_size);
    cudaMalloc(&m_buffers.detail_buffer, rgb_size);  // For detail restoration
    cudaMalloc(&m_buffers.history_length, float_size);  // Still float
    
    m_buffers.width = m_width;
    m_buffers.height = m_height;
}

void SVGF::freeBuffers() {
    if (m_buffers.color_curr) cudaFree(m_buffers.color_curr);
    if (m_buffers.color_prev) cudaFree(m_buffers.color_prev);
    if (m_buffers.moment1_curr) cudaFree(m_buffers.moment1_curr);
    if (m_buffers.moment2_curr) cudaFree(m_buffers.moment2_curr);
    if (m_buffers.moment1_prev) cudaFree(m_buffers.moment1_prev);
    if (m_buffers.moment2_prev) cudaFree(m_buffers.moment2_prev);
    if (m_buffers.variance) cudaFree(m_buffers.variance);
    if (m_buffers.filtered) cudaFree(m_buffers.filtered);
    if (m_buffers.detail_buffer) cudaFree(m_buffers.detail_buffer);
    if (m_buffers.history_length) cudaFree(m_buffers.history_length);
    
    m_buffers = {};
}

void SVGF::free() {
    if (!m_initialized)
        return;
    
    freeBuffers();
    m_initialized = false;
}

void SVGF::reset() {
    size_t rgb_size = m_width * m_height * sizeof(Vec3f);
    size_t float_size = m_width * m_height * sizeof(float);
    
    cudaMemset(m_buffers.color_prev, 0, rgb_size);
    cudaMemset(m_buffers.moment1_prev, 0, rgb_size);
    cudaMemset(m_buffers.moment2_prev, 0, rgb_size);
    cudaMemset(m_buffers.history_length, 0, float_size);
    
    m_frame_index = 0;
}

void SVGF::temporalAccumulation(const SVGFGBuffer& gbuffer, CUstream stream) {
    dim3 block(16, 16);
    dim3 grid((m_width + block.x - 1) / block.x, (m_height + block.y - 1) / block.y);
    
    temporalAccumulationKernel<<<grid, block, 0, stream>>>(
        m_buffers.color_curr,
        gbuffer.motion,
        m_buffers.color_prev,
        m_buffers.moment1_curr,
        m_buffers.moment2_curr,
        m_buffers.moment1_prev,
        m_buffers.moment2_prev,
        m_buffers.history_length,
        m_params.alpha,
        m_params.max_history_length,
        m_width,
        m_height
    );
    
    // Swap prev/curr moments for next frame
    std::swap(m_buffers.moment1_prev, m_buffers.moment1_curr);
    std::swap(m_buffers.moment2_prev, m_buffers.moment2_curr);
}

void SVGF::estimateVariance(CUstream stream) {
    dim3 block(16, 16);
    dim3 grid((m_width + block.x - 1) / block.x, (m_height + block.y - 1) / block.y);
    
    estimateVarianceKernel<<<grid, block, 0, stream>>>(
        m_buffers.moment1_curr,
        m_buffers.moment2_curr,
        m_buffers.variance,
        m_width,
        m_height
    );
}

void SVGF::atrousFilter(const SVGFGBuffer& gbuffer, CUstream stream) {
    dim3 block(16, 16);
    dim3 grid((m_width + block.x - 1) / block.x, (m_height + block.y - 1) / block.y);
    
    Vec3f* ping = m_buffers.color_prev;
    Vec3f* pong = m_buffers.filtered;
    
    for (int i = 0; i < m_params.filter_iterations; i++) {
        int step_size = 1 << i;  // 1, 2, 4, 8, ...
        
        atrousFilterKernel<<<grid, block, 0, stream>>>(
            ping,
            m_buffers.variance,
            gbuffer.position,
            gbuffer.normal,
            pong,
            step_size,
            m_params.phi_position,
            m_params.phi_normal,
            m_params.phi_color,
            m_params.sigma_z,
            m_params.sigma_n,
            m_params.sigma_l,
            m_width,
            m_height
        );
        
        std::swap(ping, pong);
    }
    
    // Result is now in ping (which might be filtered or color_prev)
    if (ping != m_buffers.filtered) {
        cudaMemcpyAsync(
            m_buffers.filtered,
            ping,
            m_width * m_height * sizeof(Vec3f),
            cudaMemcpyDeviceToDevice,
            stream
        );
    }
}

void SVGF::filter(
    const Vec4f* color_input,
    const SVGFGBuffer& gbuffer,
    Vec4f* color_output,
    CUstream stream
) {
    if (!m_initialized) {
        printf("SVGF not initialized!");
        return;
    }
    
    dim3 block(16, 16);
    dim3 grid((m_width + block.x - 1) / block.x, (m_height + block.y - 1) / block.y);
    
    // 1. Albedo demodulation: RGB / Albedo → Irradiance (lighting only)
    albedoDemodulationKernel<<<grid, block, 0, stream>>>(
        color_input, gbuffer.albedo, m_buffers.color_curr, m_width, m_height
    );
    
    // Handle different filtering modes
    if (m_params.use_temporal && m_params.use_spatial) {
        // Full SVGF: Temporal + Spatial
        // 2. Temporal accumulation (RGB)
        temporalAccumulationKernel<<<grid, block, 0, stream>>>(
            m_buffers.color_curr,
            gbuffer.motion,
            m_buffers.color_prev,
            m_buffers.moment1_curr,
            m_buffers.moment2_curr,
            m_buffers.moment1_prev,
            m_buffers.moment2_prev,
            m_buffers.history_length,
            m_params.alpha,
            m_params.max_history_length,
            m_width,
            m_height
        );
        
        // 3. Estimate variance (RGB)
        estimateVarianceKernel<<<grid, block, 0, stream>>>(
            m_buffers.moment1_curr,
            m_buffers.moment2_curr,
            m_buffers.variance,
            m_width,
            m_height
        );
        
        // 4. Spatial filtering (A-trous) with variance
        atrousFilter(gbuffer, stream);
        
    } else if (m_params.use_temporal && !m_params.use_spatial) {
        // Temporal only: Skip spatial filtering
        // 2. Temporal accumulation (RGB)
        temporalAccumulationKernel<<<grid, block, 0, stream>>>(
            m_buffers.color_curr,
            gbuffer.motion,
            m_buffers.color_prev,
            m_buffers.moment1_curr,
            m_buffers.moment2_curr,
            m_buffers.moment1_prev,
            m_buffers.moment2_prev,
            m_buffers.history_length,
            m_params.alpha,
            m_params.max_history_length,
            m_width,
            m_height
        );
        
        // Copy temporal result directly to filtered output
        size_t rgb_size = m_width * m_height * sizeof(Vec3f);
        cudaMemcpyAsync(
            m_buffers.filtered,
            m_buffers.color_prev,
            rgb_size,
            cudaMemcpyDeviceToDevice,
            stream
        );
        
    } else if (!m_params.use_temporal && m_params.use_spatial) {
        // Spatial only: Skip temporal and use zero variance
        size_t rgb_size = m_width * m_height * sizeof(Vec3f);
        cudaMemsetAsync(m_buffers.variance, 0, rgb_size, stream);
        
        // Copy color_curr to color_prev for spatial filtering input
        cudaMemcpyAsync(
            m_buffers.color_prev, 
            m_buffers.color_curr,
            rgb_size,
            cudaMemcpyDeviceToDevice,
            stream
        );
        
        // For spatial-only mode, increase filter strength by using more iterations
        int original_iterations = m_params.filter_iterations;
        m_params.filter_iterations = 5;  // More aggressive filtering for spatial-only
        
        // Spatial filtering (A-trous)
        atrousFilter(gbuffer, stream);
        
        // Restore original iterations
        m_params.filter_iterations = original_iterations;
        
    } else {
        // Neither temporal nor spatial: Just copy input to output
        size_t rgb_size = m_width * m_height * sizeof(Vec3f);
        cudaMemcpyAsync(
            m_buffers.filtered,
            m_buffers.color_curr,
            rgb_size,
            cudaMemcpyDeviceToDevice,
            stream
        );
    }
    
    // Extract high-frequency detail from irradiance before modulation
    // This captures detail like bump maps (in lighting) that would be lost in filtering
    if (m_params.detail_strength > 0.0f) {
        // Always use color_prev as the source since:
        // - In temporal mode: color_prev contains accumulated result
        // - In spatial-only mode: color_curr was copied to color_prev before filtering
        // - In full SVGF: color_prev contains temporal accumulated result
        extractDetailKernel<<<grid, block, 0, stream>>>(
            m_buffers.color_prev,  // Pre-filtered irradiance (temporal accumulated or copied from curr)
            m_buffers.filtered,    // Filtered irradiance
            m_buffers.detail_buffer,  // Output: high-freq detail
            m_width,
            m_height
        );
    }
    
    // 5. Albedo modulation: Filtered Irradiance × Albedo → RGB
    albedoModulationKernel<<<grid, block, 0, stream>>>(
        m_buffers.filtered, gbuffer.albedo, color_output, m_width, m_height
    );
    
    // 6. Detail restoration: Add detail × albedo back to filtered result
    // This restores high-frequency detail (like bump maps, small stars) lost in spatial filtering
    if (m_params.detail_strength > 0.0f) {
        restoreDetailKernel<<<grid, block, 0, stream>>>(
            color_output,  // Base: modulated RGB
            m_buffers.detail_buffer,  // Detail from irradiance
            gbuffer.albedo,  // Albedo for proper detail scaling
            color_output,  // Output: in-place update
            m_params.detail_strength,
            m_width,
            m_height
        );
    }
    
    m_frame_index++;
}

} // namespace prayground
