#pragma once

#include <optix.h>
#include <prayground/math/vec.h>
#include <prayground/math/matrix.h>
#include <prayground/texture/constant.h>
#include <prayground/texture/checker.h>
#include <prayground/physics/tree.h>
#include "envmap.cuh"
#include "textures.h"

#define DENOISE 0
#define USE_SVGF 0
#define SUBMISSION 1
#define INTERACTIVE 1

using namespace prayground;

using ConstantTexture = ConstantTexture_<Vec4f>;
using CheckerTexture = CheckerTexture_<Vec4f>;
using ProceduralWoodenTexture = ProceduralWoodenTexture_<Vec4f>;
using StarNightTexture = StarNightTexture_<Vec4f>;
using LeafTexture = LeafTexture_<Vec4f>;

// MIS (Multiple Importance Sampling) heuristic types
enum class MISHeuristic : uint32_t {
    Balance = 0,     // Balance heuristic: w = pdf_i / sum(pdf_j)
    PowerBeta2 = 1,  // Power heuristic with beta=2: w = pdf_i^2 / sum(pdf_j^2)
};

// POD structure for 4x4 matrix (compatible with __constant__ memory)
// Can be converted to Matrix4f using: Matrix4f(mat_data.data)
struct MatrixData {
    float data[12];  // First 12 elements (row-major), last row is [0,0,0,1]
};

struct AreaEmitterInfo
{
    void* shape_data;
    SurfaceInfo* surface_info;

    Matrix4f objToWorld;
    Matrix4f worldToObj;

    uint32_t sample_id;
    uint32_t pdf_id;
};

struct LightInteraction
{
    // A surface point on the light source in world coordinates
    Vec3f p;
    // Surface normal on the light source in world coordinates
    Vec3f n;
    // Texture coordinates on light source
    Vec2f uv;
    // Area of light source
    float area;
    // PDF of light source
    float pdf;
};

// Texture baking parameters
struct BakeTextureParams {
    uint32_t width;
    uint32_t height;
    uint32_t texture_id;
    void* texture_data;
    Vec3f* output_buffer;  // RGB float buffer
};

struct LaunchParams {
    uint32_t width;
    uint32_t height;
    uint32_t samples_per_launch;
    int32_t frame;
    uint32_t max_depth;
    
    // Adaptive sampling parameters
    bool use_adaptive_sampling;
    uint32_t adaptive_min_samples;      // Minimum samples before checking convergence
    
    // Stratified sampling parameters
    uint32_t stratified_dim;  // Grid dimension (e.g., 4 for 4x4=16spp, 8 for 8x8=64spp)
    
    // MIS (Multiple Importance Sampling) parameters
    MISHeuristic mis_heuristic;  // Balance or Power (beta=2)
    bool use_multi_light_sampling;  // Sample both area lights and environment map
    
    // Elapsed time from the start of rendering (in seconds)
    float elapsed_time;

    // Temporal anti-aliasing jitter (in pixel units)
    float2 taa_jitter;

    Vec4u* result_buffer;
    Vec4f* accum_buffer;

    // Float result buffer for post-processing (bloom) and denoising
    Vec4f* float_result_buffer;
    
    // Adaptive sampling buffers
    Vec4f* sum_buffer;         // Sum of samples (RGB)
    Vec4f* sum_squared_buffer; // Sum of squared samples (RGB)
    uint32_t* sample_count_buffer; // Per-pixel sample count
    uint8_t* converged_buffer; // Per-pixel convergence flag (0 = not converged, 1 = converged)

#if !SUBMISSION
    Vec4f* normal_buffer;
    Vec4f* albedo_buffer;
    Vec4f* uv_buffer;
#endif

#if USE_SVGF
    // For SVGF temporal filtering
    Vec4f* position_buffer;     // World-space position
    Vec4f* motion_buffer;        // Screen-space motion vector (current - previous)
    Vec4f* prev_position_buffer; // Previous frame position for motion calculation
    MatrixData prev_view_projection; // Previous frame's view-projection matrix (POD)
    MatrixData curr_view_projection; // Current frame's view-projection matrix (POD)
#endif

    OptixTraversableHandle handle;

    AreaEmitterInfo* lights;
    uint32_t n_lights;

    // Environment map importance sampling data
    EnvmapSamplingData* envmap_sampling_data;
    uint32_t envmap_sample_id;
    uint32_t envmap_pdf_id;
    uint32_t envmap_texture_id;
    void* envmap_texture_data;

    float white;

    bool enable_mis;
};

struct ProceduralTreeData {
    float gravity;           // Downward force (bending down)
    float coverage;          // Horizontal spreading force
    float vertically;        // Upward vertical force (resistance to gravity)
    float twist;             // Twist amount for spiral effect
    bool trunk_mode;         // If true, main trunk continues straight with side branches
    float scale;
    int depth;
    float radius;
    
    // Leaf parameters
    bool has_leaves;         // Whether to generate leaves
    float leaf_density;      // Density of leaves (0.0 - 1.0)
    int leaf_start_gen;      // Generation to start adding leaves (e.g., 2 = only on thin branches)
    float leaf_size;         // Size of each leaf
};