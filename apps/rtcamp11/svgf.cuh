#pragma once

#include <prayground/math/vec.h>
#include <cuda_runtime.h>
#include <cuda.h>

namespace prayground {

// SVGF G-Buffer structure (using Vec4f for compatibility with existing params)
struct SVGFGBuffer {
    Vec4f* position;      // World space position (w unused)
    Vec4f* normal;        // World space normal (w unused)
    Vec4f* albedo;        // Surface albedo (w unused)
    Vec4f* motion;        // Motion vector (zw unused, xy = screen-space motion)
    int width;
    int height;
};

// SVGF intermediate buffers
struct SVGFBuffers {
    Vec3f* color_curr;        // Current frame color (RGB)
    Vec3f* color_prev;        // Previous frame color (RGB)
    Vec3f* moment1_curr;      // 1st moment (mean, RGB)
    Vec3f* moment2_curr;      // 2nd moment (for variance, RGB)
    Vec3f* moment1_prev;      // Previous 1st moment (RGB)
    Vec3f* moment2_prev;      // Previous 2nd moment (RGB)
    Vec3f* variance;          // Estimated variance (RGB)
    Vec3f* filtered;          // Filtered output (RGB)
    Vec3f* detail_buffer;     // High-frequency detail (for restoration)
    float* history_length;    // Temporal accumulation count
    int width;
    int height;
};

// SVGF parameters
struct SVGFParams {
    // Temporal parameters
    float alpha;                 // Temporal blending factor (default: 0.2)
    int max_history_length;      // Max frames to accumulate (default: 32)
    
    // Spatial filter parameters
    int filter_iterations;       // A-trous iterations (default: 4)
    float phi_color;            // Color weight threshold (default: 10.0)
    float phi_normal;           // Normal weight threshold (default: 32.0)
    float phi_position;         // Position weight threshold (default: 1.0)
    
    // Edge stopping
    float sigma_z;              // Depth sensitivity (default: 1.0)
    float sigma_n;              // Normal sensitivity (default: 128.0)
    float sigma_l;              // Luminance sensitivity (default: 4.0)
    
    SVGFParams()
        : alpha(0.2f)
        , max_history_length(32)
        , filter_iterations(4)
        , phi_color(10.0f)
        , phi_normal(32.0f)
        , phi_position(1.0f)
        , sigma_z(1.0f)
        , sigma_n(128.0f)
        , sigma_l(4.0f)
        , use_temporal(true)   // Enable temporal filtering by default
        , use_spatial(true)    // Enable spatial filtering by default
        , detail_strength(0.5f) // Detail restoration strength (0.0 = none, 1.0 = full detail)
    {}
    
    bool use_temporal;      // If false, only spatial filtering is applied
    bool use_spatial;       // If false, only temporal filtering is applied
    float detail_strength;  // High-frequency detail restoration strength
};

// SVGF class for managing buffers and execution
class SVGF {
public:
    SVGF();
    ~SVGF();
    
    void init(int width, int height);
    void free();
    
    // Main filtering pipeline
    void filter(
        const Vec4f* color_input,    // RGB color from ray tracing (Vec4f)
        const SVGFGBuffer& gbuffer,
        Vec4f* color_output,         // Filtered RGB output (Vec4f)
        CUstream stream = 0
    );
    
    // Reset temporal accumulation
    void reset();
    
    // Parameter access
    SVGFParams& params() { return m_params; }
    
private:
    SVGFBuffers m_buffers;
    SVGFParams m_params;
    
    int m_width;
    int m_height;
    bool m_initialized;
    int m_frame_index;
    
    // Internal pipeline stages
    void temporalAccumulation(const SVGFGBuffer& gbuffer, CUstream stream);
    void estimateVariance(CUstream stream);
    void atrousFilter(const SVGFGBuffer& gbuffer, CUstream stream);
    
    void allocateBuffers();
    void freeBuffers();
};

} // namespace prayground
